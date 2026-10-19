"""Tests for SweepOffShelf3D: cubes on a cupboard shelf, a heavy box below its edge."""

import os
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import kinder
from kinder.envs.dynamic3d.sweep_off_shelf3d import ObjectCentricSweepOffShelf3DEnv
from kinder.envs.dynamic3d.utils import convert_yaw_to_quaternion

_TASK_CONFIG_PATH = (
    Path(kinder.__path__[0])
    / "envs"
    / "dynamic3d"
    / "tasks"
    / "SweepOffShelf3D"
    / "SweepOffShelf3D-o5.json"
)
_IDENTITY = [1.0, 0.0, 0.0, 0.0]
_CUBES = [f"cube_{i}" for i in range(5)]
# The Kinova Gen3 on the TidyBot is rated for a 4 kg payload.
_ARM_PAYLOAD_KG = 4.0


def _register_keeping_render_backend() -> None:
    chosen = {key: os.environ.get(key) for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM")}
    kinder.register_all_environments()
    for key, value in chosen.items():
        if value:
            os.environ[key] = value


def _settle(env, steps: int = 2000) -> None:
    sim = env._robot_env.sim  # pylint: disable=protected-access
    for _ in range(steps):
        sim.step()


def _model_and_data(env):
    sim = env._robot_env.sim  # pylint: disable=protected-access
    return sim.model.mj_model, sim.data.mj_data


@pytest.fixture(name="env", scope="module")
def _env():
    env = ObjectCentricSweepOffShelf3DEnv(
        task_config_path=str(_TASK_CONFIG_PATH), scene_bg=False, allow_state_access=True
    )
    try:
        yield env
    finally:
        env.close()


def test_sweep_off_shelf3d_registers():
    """The task folder registers like every other Dynamic3D task."""
    _register_keeping_render_backend()
    env = kinder.make("kinder/SweepOffShelf3D-o5-v0")
    env.close()


def test_cubes_start_on_the_second_shelf_from_the_bottom(env):
    """Cubes rest on shelf index 1, the board above the bottom board."""
    env.reset(seed=0)
    _settle(env, steps=1000)
    shelf = env.shelf_top_z(shelf=1)
    for name in _CUBES:
        position, _ = env.get_object(name).get_pose()
        assert position[2] == pytest.approx(shelf + 0.01, abs=3e-3)
        assert not env.cube_in_box(name)
    assert not env._check_goals()  # pylint: disable=protected-access


def test_the_wiper_starts_on_the_third_shelf(env):
    """The SweepIntoDrawer3D wiper lies on shelf index 2."""
    env.reset(seed=0)
    _settle(env, steps=1000)
    shelf = env.shelf_top_z(shelf=2)
    position, _ = env.get_object("wiper_0").get_pose()
    # The wiper's origin is its bottom face; soft contact lets it sink ~0.1 mm.
    assert shelf - 0.002 < position[2] < shelf + 0.06


def test_the_box_is_heavy_and_its_rim_is_below_the_cube_shelf(env):
    """Cubes pulled off the shelf edge fall in; the arm cannot lift the box."""
    env.reset(seed=0)
    model, _ = _model_and_data(env)
    assert float(model.body_subtreemass[model.body("box_0").id]) > 5 * _ARM_PAYLOAD_KG
    _, rim = env.box_interior()
    assert rim < env.shelf_top_z(shelf=1)


def test_the_box_stands_under_the_cube_shelfs_front_edge(env):
    """The box is on the floor against the open front, spanning most of the shelf."""
    env.reset(seed=0)
    (x0, y0, x1, y1), _ = env.box_interior()
    front = env.shelf_front_x()
    assert x0 < front <= x1 + 0.03
    assert y1 - y0 > 0.4


def test_a_cube_dropped_into_the_box_is_in_it(env):
    """A cube released above the box settles inside and counts."""
    env.reset(seed=0)
    (x0, y0, x1, y1), _ = env.box_interior()
    cube = env.get_object("cube_0")
    cube.set_pose([(x0 + x1) / 2, (y0 + y1) / 2, 0.3], _IDENTITY)
    cube.set_velocity([0.0] * 3, [0.0] * 3)
    _settle(env)
    assert env.cube_in_box("cube_0")


def test_every_cube_in_the_box_is_the_goal(env):
    """The goal is all five cubes inside the box."""
    env.reset(seed=0)
    (x0, y0, x1, _), _ = env.box_interior()
    for i, name in enumerate(_CUBES):
        cube = env.get_object(name)
        cube.set_pose([(x0 + x1) / 2, y0 + 0.06 + 0.08 * i, 0.3], _IDENTITY)
        cube.set_velocity([0.0] * 3, [0.0] * 3)
    _settle(env)
    assert env._check_goals()  # pylint: disable=protected-access


def test_the_declared_goal_region_is_the_box_interior(env):
    """The JSON goal region documents the goal; it must match the measured box."""
    env.reset(seed=0)
    (x0, y0, x1, y1), _ = env.box_interior()
    declared = env.task_config["regions"]["box_goal_region"]["ranges"][0]
    assert declared == pytest.approx([x0, y0, x1, y1], abs=1e-3)
    goal = env.task_config["goal_state"]
    assert sorted(pred[1] for pred in goal) == _CUBES
    assert all(pred[2] == "box_goal_region" for pred in goal)


def _catching_geoms(model):
    """The shelf boards and the box: the thin parts that catch falling objects."""
    box = model.body("box_0").id
    return [
        g
        for g in range(model.ngeom)
        if (model.geom(g).name or "").startswith("cupboard_1_shelf")
        or model.geom_bodyid[g] == box
    ]


def test_only_the_boards_and_box_have_stiff_contacts_with_priority(env):
    """Stiffened with priority so their parameters win over the soft wiper and
    cubes; the rest of the cupboard and the cubes are unchanged."""
    env.reset(seed=0)
    model, _ = _model_and_data(env)
    catching = set(_catching_geoms(model))
    assert len(catching) == 4 + 5
    for geom in range(model.ngeom):
        if geom in catching:
            assert model.geom_solref[geom][0] == pytest.approx(0.004)
            assert model.geom_priority[geom] == 1
        elif (model.geom(geom).name or "").startswith(("cupboard_1", "cube")):
            assert model.geom_solref[geom][0] == pytest.approx(0.02)


def _descendants(model, root):
    bodies = {root}
    for body in range(model.nbody):
        parent = body
        while parent not in (0, root):
            parent = int(model.body_parentid[parent])
        if parent == root:
            bodies.add(body)
    return bodies


def _gripper_geoms(model):
    """(pad geoms, every other collidable gripper geom)."""
    bodies = _descendants(model, model.body("robot_base").id)
    collidable = [
        g
        for g in range(model.ngeom)
        if int(model.geom_bodyid[g]) in bodies
        and (model.geom_contype[g] or model.geom_conaffinity[g])
    ]
    pads = [
        g for g in collidable if (model.geom(g).name or "").endswith(("pad1", "pad2"))
    ]
    return pads, [g for g in collidable if g not in pads]


def _wiper_geoms(model):
    wiper = model.body("wiper_0").id
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] == wiper]


def test_the_gripper_links_and_wiper_have_the_pads_stiff_contact(env):
    """Every gripper link and the wiper collide as stiffly as the pads, without the
    pads' priority: priority would also make their friction win, and friction is
    unchanged."""
    env.reset(seed=0)
    model, _ = _model_and_data(env)
    pads, links = _gripper_geoms(model)
    assert len(pads) == 4 and len(links) == 9
    for geom in links + _wiper_geoms(model):
        assert model.geom_solref[geom] == pytest.approx(model.geom_solref[pads[0]])
        assert model.geom_solimp[geom] == pytest.approx(model.geom_solimp[pads[0]])
        assert model.geom_priority[geom] == 0
        assert model.geom_friction[geom][0] == pytest.approx(1.0)


def _contacts_with(data, geoms, others):
    """{other geom: deepest penetration in metres} over contacts of geoms with
    others."""
    deepest: dict[int, float] = {}
    for i in range(data.ncon):
        contact = data.contact[i]
        for mine, other in (
            (contact.geom1, contact.geom2),
            (contact.geom2, contact.geom1),
        ):
            if mine in geoms and other in others:
                deepest[int(other)] = max(deepest.get(int(other), 0.0), -contact.dist)
    return deepest


# Controllers have pressed the held wiper into cubes, a shelf edge or a panel with
# 84-105 N; the handle slid up through the pads and sank up to 22.7 mm into the palm.
_PRESS_N = 100.0


def test_a_held_wiper_pressed_into_the_hand_stays_out_of_the_gripper_links(env):
    """A wiper pinched by its handle and pressed head-first against a board (the
    board's push on the wiper, and the arm's equal push on the hand) slides up into
    the palm. It must stop at the palm's surface rather than sink into it, and stay
    held by both pads."""
    env.reset(seed=0)
    sim = env._robot_env.sim  # pylint: disable=protected-access
    model, data = _model_and_data(env)
    pads, links = _gripper_geoms(model)
    wiper_geoms = set(_wiper_geoms(model))
    hand = model.body("robot_base").id
    wiper_body = model.body("wiper_0").id
    sim.forward()
    middle = (
        data.xpos[model.body("robot_left_pad").id]
        + data.xpos[model.body("robot_right_pad").id]
    ) / 2
    up = data.xpos[hand] - middle
    up /= np.linalg.norm(up)
    across = np.array([0.0, 1.0, 0.0])
    frame = np.column_stack([up, across, np.cross(up, across)])
    quaternion = Rotation.from_matrix(frame).as_quat(scalar_first=True).tolist()
    # The handle runs along the wiper's x to x = 0.115; pinched at x = 0.10 its end
    # starts clear of the palm.
    position = (middle - frame @ np.array([0.10, 0.0, 0.0075])).tolist()
    wiper = env.get_object("wiper_0")
    env._robot_env.ctrl["gripper"][:] = 255.0  # pylint: disable=protected-access
    for _ in range(1500):
        wiper.set_pose(position, quaternion)
        wiper.set_velocity([0.0] * 3, [0.0] * 3)
        sim.step()
    _settle(env, steps=500)
    assert set(_contacts_with(data, wiper_geoms, pads)) == set(pads)
    assert not _contacts_with(data, wiper_geoms, links)
    deepest = 0.0
    for step in range(4000):
        force = _PRESS_N * min(1.0, step / 1000) * up
        data.xfrc_applied[wiper_body, :3] = force
        data.xfrc_applied[hand, :3] = -force
        sim.step()
        deepest = max([deepest, *_contacts_with(data, wiper_geoms, links).values()])
    data.xfrc_applied[[wiper_body, hand]] = 0.0
    _settle(env, steps=1000)
    assert deepest < 3e-3
    held = _contacts_with(data, wiper_geoms, pads)
    assert {int(model.geom_bodyid[g]) for g in held} == {
        model.body("robot_left_pad").id,
        model.body("robot_right_pad").id,
    }


def test_a_cube_pinched_between_the_pads_stays_held(env):
    """The stiffer gripper links leave an ordinary cube grasp closing and holding."""
    env.reset(seed=0)
    sim = env._robot_env.sim  # pylint: disable=protected-access
    model, data = _model_and_data(env)
    pads, _ = _gripper_geoms(model)
    sim.forward()
    middle = data.geom_xpos[pads].mean(axis=0)
    cube = env.get_object("cube_0")
    body = model.body("cube_0").id
    cube_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == body}
    env._robot_env.ctrl["gripper"][:] = 255.0  # pylint: disable=protected-access
    for _ in range(1500):
        cube.set_pose(middle.tolist(), _IDENTITY)
        cube.set_velocity([0.0] * 3, [0.0] * 3)
        sim.step()
    _settle(env, steps=2000)
    held = _contacts_with(data, cube_geoms, pads)
    assert {int(model.geom_bodyid[g]) for g in held} == {
        model.body("robot_left_pad").id,
        model.body("robot_right_pad").id,
    }
    assert np.linalg.norm(cube.get_pose()[0] - middle) < 5e-3


@pytest.mark.parametrize("yaw", [0.0, 90.0])
# 0.239 m is the largest drop the shelf gap allows: the 0.254 m between boards less
# the wiper's 15 mm. A 0.25 m drop starts the wiper 11 mm inside the board above.
@pytest.mark.parametrize("drop", [0.1, 0.2, 0.239])
def test_a_dropped_wiper_lands_on_its_shelf(env, yaw, drop):
    """A wiper falling onto a shelf stays on it rather than passing through."""
    env.reset(seed=0)
    shelf = env.shelf_top_z(shelf=2)
    wiper = env.get_object("wiper_0")
    wiper.set_pose(
        [1.512, 0.038, shelf + drop], convert_yaw_to_quaternion(np.deg2rad(yaw))
    )
    wiper.set_velocity([0.0] * 3, [0.0] * 3)
    _settle(env, steps=1500)
    assert wiper.get_pose()[0][2] > shelf - 0.003


def test_cubes_falling_from_the_shelf_edge_stay_in_the_box(env):
    """A cube leaving the cube shelf at its front edge lands and stays in the box."""
    env.reset(seed=0)
    (_, y0, x1, _), _ = env.box_interior()
    start = env.shelf_top_z(shelf=1) + 0.011
    for i, name in enumerate(_CUBES):
        cube = env.get_object(name)
        cube.set_pose([x1 - 0.02, y0 + 0.06 + 0.08 * i, start], _IDENTITY)
        cube.set_velocity([-0.3, 0.0, 0.0], [0.0] * 3)
    _settle(env)
    for name in _CUBES:
        assert env.cube_in_box(name), name
        assert env.get_object(name).get_pose()[0][2] > 0.005


def test_a_cube_on_the_floor_beside_the_box_is_not_in_it(env):
    """Containment needs the cube inside the box walls, not merely near it."""
    env.reset(seed=0)
    (x0, y0, _, _), _ = env.box_interior()
    cube = env.get_object("cube_0")
    cube.set_pose([x0 - 0.1, y0, 0.05], _IDENTITY)
    cube.set_velocity([0.0] * 3, [0.0] * 3)
    _settle(env)
    assert not env.cube_in_box("cube_0")


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 1109584189, 1282264805])
def test_the_cupboard_stands_on_the_floor_at_one_height(env, seed):
    """The cupboard is a fixed fixture and never settles, so it must be placed on
    the floor; otherwise the shelves and the drop into the box change with the seed."""
    env.reset(seed=seed)
    model, data = _model_and_data(env)
    assert float(data.xpos[model.body("cupboard_1").id][2]) == pytest.approx(
        0.0, abs=1e-6
    )
    # The board above the bottom one: 0.254 m shelf height plus the board thickness.
    assert env.shelf_top_z(shelf=1) == pytest.approx(0.2794, abs=1e-4)


def _push_cube_on_the_shelf(env, *, above_centre: float, force_over_weight: float):
    """Rest cube_0 alone on the cube shelf, push it toward the front edge with a
    constant horizontal force applied `above_centre` metres above its centre, and return
    (distance slid, largest tilt in degrees)."""
    env.reset(seed=0)
    (x0, y0, x1, _), _ = env.box_interior()
    for i, name in enumerate(_CUBES[1:]):
        cube = env.get_object(name)
        cube.set_pose([(x0 + x1) / 2, y0 + 0.06 + 0.08 * i, 0.1], _IDENTITY)
        cube.set_velocity([0.0] * 3, [0.0] * 3)
    cube = env.get_object("cube_0")
    cube.set_pose([1.5, 0.0, env.shelf_top_z(shelf=1) + 0.0101], _IDENTITY)
    cube.set_velocity([0.0] * 3, [0.0] * 3)
    _settle(env, steps=500)
    sim = env._robot_env.sim  # pylint: disable=protected-access
    model, data = sim.model.mj_model, sim.data.mj_data
    body = model.body("cube_0").id
    weight = float(model.body_subtreemass[body]) * 9.81
    force = np.array([-force_over_weight * weight, 0.0, 0.0])
    torque = np.cross([0.0, 0.0, above_centre], force)
    start = data.xpos[body].copy()
    tilt = 0.0
    for _ in range(int(0.15 / model.opt.timestep)):
        data.xfrc_applied[body] = np.concatenate([force, torque])
        sim.step()
        up = data.xmat[body].reshape(3, 3)[2, 2]
        tilt = max(tilt, float(np.degrees(np.arccos(np.clip(abs(up), 0.0, 1.0)))))
    data.xfrc_applied[body] = 0.0
    return float(start[0] - data.xpos[body][0]), tilt


def test_a_cube_pushed_above_its_centre_slides_rather_than_tips(env):
    """A 2 cm cube on a wooden board, pushed 5 mm above its centre with 0.8 of its
    weight, slides. A rigid cube pushed at height h, half-width a, tips while static
    only if F h > m g a with F below mu m g, and while sliding only if
    F (h - a) > m g a (1 - mu): for dry wood on wood (mu 0.25-0.5) neither holds.
    With MuJoCo's default friction (mu = 1) it cannot slide and tips instead, which
    is how a wiper pulling cubes across the shelf rolled them over."""
    slid, tilt = _push_cube_on_the_shelf(env, above_centre=0.005, force_over_weight=0.8)
    assert tilt < 5.0
    assert slid > 0.01


def test_only_the_shelf_boards_have_wooden_friction(env):
    """The boards slide like wood; the box, the cubes, the wiper and the gripper pads
    keep their own friction."""
    env.reset(seed=0)
    model, _ = _model_and_data(env)
    for geom in range(model.ngeom):
        name = model.geom(geom).name or ""
        if name.startswith("cupboard_1_shelf"):
            assert 0.25 <= model.geom_friction[geom][0] <= 0.5
        elif model.geom_bodyid[geom] == model.body("box_0").id or (
            str(model.body(model.geom_bodyid[geom]).name).startswith(("cube", "wiper"))
        ):
            assert model.geom_friction[geom][0] == pytest.approx(1.0)


def _turn_box(env, yaw_deg, tilt_deg=0.0, shift=(0.0, 0.0)):
    """Turn the box in place about z by yaw_deg, then tip it about its own x by
    tilt_deg; return its rotation and position."""
    box = env.get_object("box_0")
    position, _ = box.get_pose()
    position = np.asarray(position, dtype=float) + np.array([*shift, 0.0])
    rotation = Rotation.from_euler("zx", [yaw_deg, tilt_deg], degrees=True)
    box.set_pose(position, rotation.as_quat()[[3, 0, 1, 2]])
    box.set_velocity([0.0] * 3, [0.0] * 3)
    return rotation, position


def _place_in_box_frame(env, name, rotation, position, local):
    """Put a cube at `local` in the box's frame, turned with the box."""
    env.get_object(name).set_pose(
        rotation.apply(local) + position, rotation.as_quat()[[3, 0, 1, 2]]
    )


def _box_half_extents(env):
    box = env.get_object("box_0")
    return box.length / 2 - box.wall_thickness, box.width / 2 - box.wall_thickness


@pytest.mark.parametrize("yaw", [-10.0, -2.0, -0.3, 0.3, 2.0, 10.0])
def test_a_cube_inside_a_turned_box_is_in_it(env, yaw):
    """A cube in a corner of a box turned about z still counts as in the box."""
    env.reset(seed=0)
    rotation, position = _turn_box(env, yaw)
    half_x, half_y = _box_half_extents(env)
    _place_in_box_frame(
        env, "cube_0", rotation, position, [half_x - 0.02, half_y - 0.02, 0.05]
    )
    assert env.cube_in_box("cube_0")


@pytest.mark.parametrize("yaw", [-10.0, -2.0, -0.3, 0.3, 2.0, 10.0])
def test_a_cube_outside_a_turned_box_is_not_in_it(env, yaw):
    """Just beyond a turned box's wall, or above its rim, is not in the box."""
    env.reset(seed=0)
    rotation, position = _turn_box(env, yaw)
    half_x, half_y = _box_half_extents(env)
    height = env.get_object("box_0").height
    outside = {
        "cube_0": [half_x + 0.02, 0.0, 0.05],
        "cube_1": [0.0, -half_y - 0.02, 0.05],
        "cube_2": [0.0, 0.0, height + 0.02],
    }
    for name, local in outside.items():
        _place_in_box_frame(env, name, rotation, position, local)
    for name in outside:
        assert not env.cube_in_box(name), name


def test_a_spot_the_turned_box_has_moved_away_from_is_not_in_it(env):
    """A spot inside the unturned footprint that the turned box no longer covers is
    outside: containment follows the box, not where it used to be."""
    env.reset(seed=0)
    half_x, half_y = _box_half_extents(env)
    position, _ = env.get_object("box_0").get_pose()
    spot = np.asarray(position, dtype=float) + [half_x - 0.015, half_y - 0.015, 0.05]
    cube = env.get_object("cube_0")
    cube.set_pose(spot, _IDENTITY)
    assert env.cube_in_box("cube_0")
    _turn_box(env, 10.0)
    cube.set_pose(spot, _IDENTITY)
    assert not env.cube_in_box("cube_0")


def test_unturned_containment_is_still_the_footprint_and_rim_test(env):
    """For an axis-aligned box, cube_in_box answers exactly as before."""
    env.reset(seed=0)
    (x0, y0, x1, y1), rim = env.box_interior()
    cube = env.get_object("cube_0")
    half = float(cube.size[0])
    cases = {
        (x0 + half + 1e-6, y0 + half + 1e-6, rim - half - 1e-4): True,
        (x1 - half - 1e-6, y1 - half - 1e-6, rim - half - 1e-4): True,
        (x0 + half - 1e-4, (y0 + y1) / 2, 0.05): False,
        ((x0 + x1) / 2, y1 - half + 1e-4, 0.05): False,
        ((x0 + x1) / 2, (y0 + y1) / 2, rim - half + 1e-4): False,
    }
    for xyz, inside in cases.items():
        cube.set_pose(list(xyz), _IDENTITY)
        assert env.cube_in_box("cube_0") is inside, xyz


def test_a_tipped_box_holds_nothing(env):
    """A box tipped well over is no longer a container, so nothing in its frame counts;
    a box tilted a degree on the floor still holds its cubes."""
    env.reset(seed=0)
    rotation, position = _turn_box(env, 0.0, tilt_deg=30.0)
    _place_in_box_frame(env, "cube_0", rotation, position, [0.0, 0.0, 0.05])
    assert not env.cube_in_box("cube_0")
    rotation, position = _turn_box(env, 0.0, tilt_deg=1.0)
    _place_in_box_frame(env, "cube_0", rotation, position, [0.0, 0.0, 0.05])
    assert env.cube_in_box("cube_0")


def test_an_episode_survives_a_box_nudged_a_fraction_of_a_degree(env):
    """A wiper nudging the box ~0.3 degrees and ~1.4 mm must not end the episode; every
    env.step checks the goal, which used to raise for a box not axis-aligned."""
    env.reset(seed=0)
    _turn_box(env, 0.28, shift=(0.0014, 0.0))
    env.action_space.seed(0)
    _, _, terminated, _, _ = env.step(env.action_space.sample())
    assert not terminated
