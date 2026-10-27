"""Tests for puckpushing_utils.py."""

import math

import numpy as np

from kinder.envs.puckpushing_utils import (
    GRAVITY,
    MAPS,
    THREE_ROUTES,
    PuckPhysics,
    PuckWorld,
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
    start = np.array([0.8, 3.4])  # laminate corridor
    speed = 0.4
    result = world.slide(start, speed, 0.0, FRICTIONS)
    expected = speed**2 / (2 * FRICTIONS[0] * GRAVITY)
    assert result.hole is None
    assert math.isclose(result.stop[0] - start[0], expected, rel_tol=1e-6)
    assert math.isclose(result.stop[1], start[1], abs_tol=1e-9)


def test_slide_crossing_patches_uses_each_friction():
    """Kinetic energy drops by mu * g per meter on each patch crossed."""
    world = PuckWorld(THREE_ROUTES)
    # From wood at y=1.9 straight down into the felt corridor (y < 1.6).
    start = np.array([1.0, 1.9])
    speed = 0.8
    result = world.slide(start, speed, -math.pi / 2, FRICTIONS)
    on_wood = start[1] - 1.6
    energy = speed**2 / 2 - FRICTIONS[1] * GRAVITY * on_wood
    on_felt = energy / (FRICTIONS[2] * GRAVITY)
    assert math.isclose(result.stop[1], 1.6 - on_felt, rel_tol=1e-6)


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


def test_sample_frictions_respects_the_prior():
    """Samples fall in each surface's range."""
    rng = np.random.default_rng(0)
    prior = ((0.02, 0.05), (0.05, 0.1), (0.1, 0.2))
    for _ in range(20):
        sample = sample_frictions(rng, prior)
        assert all(lo <= f <= hi for f, (lo, hi) in zip(sample, prior))
