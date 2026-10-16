"""Tests for SweepOffShelf3D: cubes on a cupboard shelf, a heavy box below its edge."""

import os
from pathlib import Path

import numpy as np
import pytest

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


def test_only_the_boards_and_box_have_stiff_contacts(env):
    """Stiffened with priority so their parameters win over the soft wiper and
    cubes; nothing else in the scene is changed."""
    env.reset(seed=0)
    model, _ = _model_and_data(env)
    catching = set(_catching_geoms(model))
    assert len(catching) == 4 + 5
    for geom in range(model.ngeom):
        if geom in catching:
            assert model.geom_solref[geom][0] == pytest.approx(0.004)
            assert model.geom_priority[geom] == 1
        elif (model.geom(geom).name or "").startswith(("cupboard_1", "wiper", "cube")):
            assert model.geom_solref[geom][0] == pytest.approx(0.02)


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
