"""Tests for puckpushing_utils.py."""

import collections
import math

import numpy as np

from kinder.envs.puckpushing_utils import (
    GRAVITY,
    MAPS,
    THREE_ROUTES,
    PuckPhysics,
    BoxFrictionPrior,
    FrictionPrior,
    PuckWorld,
    StrikeEnd,
    sample_frictions,
)

FRICTIONS = (0.03, 0.07, 0.15)


def test_map_rectangles_tile_the_grid():
    """Walls and floor patches cover every cell exactly once."""
    puck_map = THREE_ROUTES
    area = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in puck_map.wall_rects())
    area += sum(
        (x1 - x0) * (y1 - y0) for _, (x0, y0, x1, y1) in puck_map.surface_rects()
    )
    assert math.isclose(area, puck_map.width * puck_map.height, rel_tol=1e-9)
    assert set(MAPS) == {"threeroutes"}


def test_slide_on_one_surface_stops_at_the_coulomb_distance():
    """A slide within one patch stops after v^2 / (2 mu g)."""
    world = PuckWorld(THREE_ROUTES)
    start = np.array([0.8, 3.4])  # blue corridor
    speed = 0.4
    result = world.slide(start, speed, 0.0, FRICTIONS)
    expected = speed**2 / (2 * FRICTIONS[0] * GRAVITY)
    assert result.hole is None
    assert math.isclose(result.stop[0] - start[0], expected, rel_tol=1e-6)
    assert math.isclose(result.stop[1], start[1], abs_tol=1e-9)


def test_slide_crossing_patches_uses_each_friction():
    """Kinetic energy drops by mu * g per meter on each patch crossed."""
    world = PuckWorld(THREE_ROUTES)
    # From tan at y=1.9 straight down into the green corridor (y < 1.6).
    start = np.array([1.0, 1.9])
    speed = 0.8
    result = world.slide(start, speed, -math.pi / 2, FRICTIONS)
    on_tan = start[1] - 1.6
    energy = speed**2 / 2 - FRICTIONS[1] * GRAVITY * on_tan
    on_green = energy / (FRICTIONS[2] * GRAVITY)
    assert math.isclose(result.stop[1], 1.6 - on_green, rel_tol=1e-6)


def test_wall_bounce_scales_the_normal_velocity():
    """A head-on bounce keeps restitution^2 of the remaining kinetic energy."""
    phys = PuckPhysics(wall_restitution=0.5)
    world = PuckWorld(THREE_ROUTES, phys)
    start = np.array([1.0, 3.4])
    speed = 1.0
    # Toward the top wall (y=4.0), so the puck touches at y = 4.0 - r.
    result = world.slide(start, speed, math.pi / 2, (0.03, 0.07, 0.15))
    contact = 4.0 - phys.puck_radius
    v2 = speed**2 - 2 * 0.03 * GRAVITY * (contact - start[1])
    back = (0.25 * v2) / (2 * 0.03 * GRAVITY)
    assert len(result.path) == 3
    assert math.isclose(result.stop[1], contact - back, rel_tol=1e-6)


def test_passing_over_a_hole_ends_the_slide():
    """The slide fails where the puck's center enters a hole."""
    world = PuckWorld(THREE_ROUTES)
    hole = world.holes[1]
    start = np.array([hole[0] - 0.8, hole[1] + 0.1])
    result = world.slide(start, 1.2, 0.0, (0.03, 0.07, 0.05))
    assert result.hole == 1
    gap = np.linalg.norm(result.stop - hole)
    assert math.isclose(gap, world.physics.hole_radius, rel_tol=1e-6)
    assert world.over_hole(result.stop + np.array([0.01, 0.0]))


def test_push_feasibility_and_driving():
    """The robot needs room behind the puck, and drives around walls."""
    world = PuckWorld(THREE_ROUTES)
    near_wall = np.array([0.55, 2.2])
    # The puck would hit the wall while still on the paddle.
    assert not world.push_feasible(near_wall, math.pi)
    # The robot cannot stand between the puck and the wall, even at an angle.
    assert not world.push_feasible(near_wall, 0.0)
    assert world.push_feasible(near_wall, math.pi / 2)
    # Straight behind is blocked by the wall, but standing at 60 degrees works.
    puck = np.array(THREE_ROUTES.puck_start)
    pose = world.push_pose(puck, 0.0)
    assert pose is not None and math.isclose(abs(pose[1]), math.radians(60))
    start = np.array(THREE_ROUTES.robot_start)
    # From the left side to the right room, around the central block.
    goal_side = np.array([4.4, 2.2])
    node = world.nav_node_of(start)
    distance = world.drive_distance(node, goal_side)
    assert math.isfinite(distance)
    assert distance >= np.linalg.norm(goal_side - start)
    path = world.drive_path(start, goal_side)
    assert np.allclose(path[0], start) and np.allclose(path[-1], goal_side)
    # The slot is too narrow for the robot.
    assert world.nav_node_of(np.array([2.6, 2.2])) == -1


def test_slide_bounces_off_the_robot():
    """A puck sliding back into the stopped paddle or the base rebounds off it."""
    world = PuckWorld(THREE_ROUTES)
    phys = world.physics
    e = phys.wall_restitution
    start = np.array([2.0, 3.0])
    v = 0.6
    decel = FRICTIONS[0] * GRAVITY
    # Paddle face 0.5 m behind the puck, square to the push (+x); the puck comes back.
    face = start - np.array([0.5 + phys.puck_radius, 0.0])
    robot = StrikeEnd(base=face - np.array([10.0, 0.0]), face=face, angle=0.0)
    result = world.slide(start, v, math.pi, FRICTIONS, robot=robot)
    v_hit = math.sqrt(v * v - 2 * decel * 0.5)
    expected = start[0] - 0.5 + (e * v_hit) ** 2 / (2 * decel)
    assert abs(result.stop[0] - expected) < 1e-6
    # Base: a disc of robot_radius, hit head on.
    base = start - np.array([0.5 + phys.robot_radius + phys.puck_radius, 0.0])
    robot = StrikeEnd(base=base, face=base + np.array([0.0, 10.0]), angle=0.0)
    result = world.slide(start, v, math.pi, FRICTIONS, robot=robot)
    assert abs(result.stop[0] - expected) < 1e-6


def test_driving_goes_around_the_puck():
    """The robot never drives through the puck, and the detour costs distance."""
    world = PuckWorld(THREE_ROUTES)
    phys = world.physics
    reach = phys.robot_radius + phys.puck_radius
    puck = np.array([2.6, 3.0])
    start, goal = np.array([1.6, 3.0]), np.array([3.6, 3.0])
    node = world.nav_node_of(start)
    assert world.drive_distance(node, goal, puck) > world.drive_distance(node, goal)
    path = np.array(world.drive_path(start, goal, puck))
    assert np.linalg.norm(path - puck, axis=1).min() >= reach - 1e-9
    # Push stations keep the base off the puck.
    for angle in np.linspace(-math.pi, math.pi, 24, endpoint=False):
        pose = world.push_pose(puck, float(angle))
        if pose is not None:
            assert world.robot_clear_of_puck(pose[0], puck)


def test_paddle_must_clear_the_walls():
    """A push that would sweep the paddle (wider than the puck) into a wall is
    infeasible, even if the puck itself clears it."""
    world = PuckWorld(THREE_ROUTES)
    phys = world.physics
    # Puck resting against the top wall of the blue corridor, pushed along it.
    top = THREE_ROUTES.height - THREE_ROUTES.cell_size
    gap = 0.5 * (phys.paddle_width - 2 * phys.puck_radius)
    puck = np.array([2.6, top - phys.puck_radius - 0.5 * gap])
    assert world.puck_path_clear(puck, puck + np.array([0.3, 0.0]))
    assert not world.push_feasible(puck, 0.0)
    # Away from the wall the same push is fine.
    assert world.push_feasible(puck - np.array([0.0, gap]), 0.0)


def test_push_stance():
    """A named stance puts the base at that heading relative to the push, and only
    that stance is tried."""
    world = PuckWorld(THREE_ROUTES)
    phys = world.physics
    # Middle of the blue corridor (y from 2.8 to 4.0).
    puck = np.array([2.6, 3.4])
    for stance in (0.0, math.radians(25), -math.radians(25)):
        pose = world.push_pose(puck, 0.0, stance)
        assert pose is not None
        base, heading = pose
        assert math.isclose(heading, stance, abs_tol=1e-9)
        contact = puck - np.array([phys.puck_radius, 0.0])
        face = base + phys.paddle_offset * np.array([math.cos(heading), math.sin(heading)])
        assert np.allclose(face, contact)
    # Stances beyond the limit are clipped to it.
    # (In the green corridor, y from 0.4 to 1.6, high enough for the base to stand
    # 60 degrees off below the puck.)
    clipped = world.push_pose(np.array([2.6, 1.3]), 0.0, math.radians(80))
    assert clipped is not None and math.isclose(clipped[1], phys.max_stance)
    # Near a side wall, straight behind may be blocked while an angled stance fits.
    top = THREE_ROUTES.height - THREE_ROUTES.cell_size
    near = np.array([2.6, top - phys.puck_radius - 0.2])
    blocked = [s for s in np.radians([-60, 0, 60]) if not world.push_feasible(near, -math.pi / 2, s)]
    assert world.push_feasible(near, -math.pi / 2) == (len(blocked) < 3)


def test_sample_frictions_respects_the_prior():
    """The default prior deals its levels to the surfaces in a random order; the box
    prior keeps each surface in its range."""
    rng = np.random.default_rng(0)
    prior = FrictionPrior(levels=(0.03, 0.08, 0.15), jitter=0.2)
    slippery = collections.Counter()
    for _ in range(600):
        sample = np.array(sample_frictions(rng, prior))
        ratios = np.sort(sample) / np.array(prior.levels)
        assert np.all((ratios >= 0.8 - 1e-9) & (ratios <= 1.2 + 1e-9))
        slippery[int(np.argmin(sample))] += 1
    assert all(150 <= slippery[k] <= 250 for k in range(3))
    box = BoxFrictionPrior(((0.02, 0.05), (0.05, 0.1), (0.1, 0.2)))
    for _ in range(20):
        sample = sample_frictions(rng, box)
        assert all(lo <= f <= hi for f, (lo, hi) in zip(sample, box.ranges))
