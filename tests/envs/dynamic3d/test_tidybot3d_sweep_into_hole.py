"""Tests for the TidyBot3D SweepIntoHole3D task."""

import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

import kinder
from kinder.envs.dynamic3d.sweep_into_hole3d import (
    PIT_DEPTH,
    PIT_X,
    PIT_Y,
    WHEEL_OFFSETS,
    ObjectCentricSweepIntoHole3DEnv,
    SweepIntoHole3DEnv,
)

_TASK_CONFIG_PATH = (
    Path(kinder.__path__[0])
    / "envs"
    / "dynamic3d"
    / "tasks"
    / "SweepIntoHole3D"
    / "SweepIntoHole3D-o5.json"
)
_MIMICLABS_MESHES = (
    Path(kinder.__path__[0])
    / "envs"
    / "dynamic3d"
    / "models"
    / "assets"
    / "mimiclabs_scenes"
    / "meshes"
)
_IDENTITY = [1.0, 0.0, 0.0, 0.0]
_CUBES = [f"cube_{i}" for i in range(5)]
# The Kinova Gen3 arm on the TidyBot is rated for a 4 kg payload; the shovel must
# be far beyond that so a grasp can close on it but never lift it.
_ARM_PAYLOAD_KG = 4.0


def _register_keeping_render_backend() -> None:
    # register_all_environments() rewrites the GL backend when DISPLAY is unset;
    # a backend chosen before the call must survive it, or the render context
    # created by kinder.make() below targets a platform this process cannot use.
    chosen = {key: os.environ.get(key) for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM")}
    kinder.register_all_environments()
    for key, value in chosen.items():
        if value:
            os.environ[key] = value


def _settle(env: ObjectCentricSweepIntoHole3DEnv, steps: int = 2000) -> None:
    # 2000 steps is one second at the model's 0.5 ms timestep; a 12 cm drop has
    # stopped moving well within that.
    sim = env._robot_env.sim  # pylint: disable=protected-access
    assert sim is not None
    for _ in range(steps):
        sim.step()


def _model_and_data(env: ObjectCentricSweepIntoHole3DEnv):
    # reset() rebuilds the simulator, so these are only valid when taken after it.
    sim = env._robot_env.sim  # pylint: disable=protected-access
    assert sim is not None
    return sim.model.mj_model, sim.data.mj_data


def _make(scene_bg: bool) -> ObjectCentricSweepIntoHole3DEnv:
    return ObjectCentricSweepIntoHole3DEnv(
        task_config_path=str(_TASK_CONFIG_PATH),
        scene_bg=scene_bg,
        allow_state_access=True,
    )


def _drop_and_settle(env: ObjectCentricSweepIntoHole3DEnv, name: str) -> float:
    cube = env.get_object(name)
    cube.set_pose([1.0, 0.0, 0.05], _IDENTITY)
    cube.set_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    _settle(env)
    position, _ = cube.get_pose()
    return float(position[2])


@pytest.fixture(name="env", scope="module")
def _env():
    env = _make(scene_bg=False)
    try:
        yield env
    finally:
        env.close()


def test_sweep_into_hole3d_registers_beside_the_other_dynamic3d_tasks():
    """The task folder registers through the filesystem-derived mechanism without
    breaking any sibling, and the registered id makes, resets and renders."""
    _register_keeping_render_backend()
    env_ids = kinder.get_all_env_ids()
    assert "kinder/SweepIntoHole3D-o5-v0" in env_ids
    # A malformed task JSON would have broken every other Dynamic3D registration.
    assert "kinder/Dustpan3D-o5-v0" in env_ids
    assert "kinder/Tossing3D-o1-v0" in env_ids
    made = kinder.make("kinder/SweepIntoHole3D-o5-v0", render_mode="rgb_array")
    try:
        assert isinstance(made.unwrapped, SweepIntoHole3DEnv)
        made.reset(seed=0)
        image = made.render()
        assert image.shape == (480, 640, 3)
    finally:
        made.close()


def test_reset_rests_every_cube_on_the_floor_outside_the_hole(env):
    """Cubes start on the slab in Dustpan3D's strip, none in the pit, goal unmet."""
    env.reset(seed=0)
    for name in _CUBES:
        position, _ = env.get_object(name).get_pose()
        assert position[2] == pytest.approx(0.01, abs=2e-3)
        assert 0.45 <= position[0] <= 0.6
        assert not env.cube_in_hole(name)
    assert not env._check_goals()  # pylint: disable=protected-access


def test_a_cube_dropped_over_the_pit_settles_on_the_pit_floor(env):
    """The pit floor is a real collider exactly PIT_DEPTH below the slab top."""
    env.reset(seed=0)
    assert _drop_and_settle(env, "cube_0") == pytest.approx(-PIT_DEPTH + 0.01, abs=1e-3)
    assert env.cube_in_hole("cube_0")


@pytest.mark.skipif(
    not _MIMICLABS_MESHES.is_dir(), reason="MimicLabs assets are not downloaded"
)
def test_the_pit_is_the_same_depth_in_the_lab_scene():
    """In mimiclabs-lab2 the lab's own floor, shifted down by the task's scene
    position, is the pit bottom; the slabs keep the walkable floor at z = 0."""
    lab = _make(scene_bg=True)
    try:
        lab.reset(seed=0)
        model, data = _model_and_data(lab)
        scene = model.body("scene").id
        assert float(data.xpos[scene][2]) == pytest.approx(-PIT_DEPTH)
        for name in _CUBES:
            assert lab.get_object(name).get_pose()[0][2] == pytest.approx(
                0.01, abs=2e-3
            )
        assert _drop_and_settle(lab, "cube_0") == pytest.approx(
            -PIT_DEPTH + 0.01, abs=1e-3
        )
        assert lab.cube_in_hole("cube_0")
    finally:
        lab.close()


@pytest.mark.skipif(
    not _MIMICLABS_MESHES.is_dir(), reason="MimicLabs assets are not downloaded"
)
def test_the_lab_pit_floor_is_white_and_covers_the_opening():
    """The lab's wood floor shows through as the pit bottom unless something covers
    it, and then the opening is hard to see; a white floor makes the pit read."""
    lab = _make(scene_bg=True)
    try:
        lab.reset(seed=0)
        model, data = _model_and_data(lab)
        pit = model.geom("pit_floor").id
        assert list(model.geom_rgba[pit][:3]) == pytest.approx([1.0, 1.0, 1.0])
        centre, half = data.geom_xpos[pit], model.geom_size[pit]
        assert centre[0] - half[0] <= PIT_X[0] and centre[0] + half[0] >= PIT_X[1]
        assert centre[1] - half[1] <= PIT_Y[0] and centre[1] + half[1] >= PIT_Y[1]
        assert centre[2] + half[2] == pytest.approx(-PIT_DEPTH, abs=1e-3)
    finally:
        lab.close()


@pytest.mark.skipif(
    not _MIMICLABS_MESHES.is_dir(), reason="MimicLabs assets are not downloaded"
)
def test_the_lab_floor_tiles_line_up_across_the_slab_seams():
    """MuJoCo maps a box's texture from its centre, so the planks only continue
    across seams when every slab centre is a whole number of texture periods."""
    lab = _make(scene_bg=True)
    try:
        lab.reset(seed=0)
        model, _ = _model_and_data(lab)
        slabs = [
            g
            for g in range(model.ngeom)
            if (model.geom(g).name or "").startswith("floor_slab_")
        ]
        assert len(slabs) == 4
        for geom in slabs:
            material = model.geom_matid[geom]
            assert model.mat_texuniform[material]
            period = 1.0 / model.mat_texrepeat[material]
            centre = model.geom_pos[geom][:2] / period
            assert centre == pytest.approx(np.round(centre), abs=1e-6)
    finally:
        lab.close()


def test_every_cube_in_the_pit_is_the_goal(env):
    """The goal is the conjunction of cube_in_hole over all five cubes."""
    env.reset(seed=0)
    for index, name in enumerate(_CUBES):
        env.get_object(name).set_pose([0.85 + 0.08 * index, 0.0, -0.07], _IDENTITY)
    assert all(env.cube_in_hole(name) for name in _CUBES)
    assert env._check_goals()  # pylint: disable=protected-access


@pytest.mark.parametrize(
    "position",
    [
        # Resting on the rim: half its corners hang over the opening.
        [PIT_X[0], 0.0, 0.01],
        # Below the rim but with its outer corners inside the wall.
        [PIT_X[0] + 0.005, 0.0, -0.01],
        # The same straddle on the far wall and on a side wall.
        [PIT_X[1], 0.0, 0.01],
        [1.0, PIT_Y[1], -0.05],
    ],
)
def test_a_cube_straddling_the_rim_is_not_in_the_hole(env, position):
    """Containment is corner-exact: one corner outside the opening is not in."""
    env.reset(seed=0)
    env.get_object("cube_0").set_pose(position, _IDENTITY)
    assert not env.cube_in_hole("cube_0")


@pytest.mark.parametrize("position", [[0.7, 0.0, 0.01], [1.0, 0.3, 0.01]])
def test_a_cube_resting_on_the_floor_beside_the_hole_is_not_in_it(env, position):
    """A cube on the slab next to the pit, on either axis, is not in."""
    env.reset(seed=0)
    env.get_object("cube_0").set_pose(position, _IDENTITY)
    assert not env.cube_in_hole("cube_0")


def test_a_cube_over_the_opening_counts_only_once_it_is_below_the_rim(env):
    """Inside the footprint is not enough; every corner must be below z = 0."""
    env.reset(seed=0)
    cube = env.get_object("cube_0")
    cube.set_pose([1.0, 0.0, 0.01], _IDENTITY)
    assert not env.cube_in_hole("cube_0")
    cube.set_pose([1.0, 0.0, -0.011], _IDENTITY)
    assert env.cube_in_hole("cube_0")


def test_the_shovel_is_far_beyond_the_arm_payload(env):
    """A closed grasp could hold the shovel's handle but never lift it."""
    env.reset(seed=0)
    model, _ = _model_and_data(env)
    assert model.body_subtreemass[model.body("shovel_0").id] >= 5 * _ARM_PAYLOAD_KG
    names = {model.geom(i).name for i in range(model.ngeom)}
    assert "shovel_0_handle" in names
    assert "shovel_0_grip" in names


def test_the_shovel_rests_beside_the_hole_clear_of_the_sweep_lane(env):
    """The shovel lies at its declared spot, off the cube-strip-to-pit lane, and
    stays put on the slab under physics."""
    env.reset(seed=0)
    position, _ = env.get_object("shovel_0").get_pose()
    assert position[0] == pytest.approx(1.05, abs=1e-3)
    assert position[1] == pytest.approx(0.55, abs=1e-3)
    _settle(env, steps=200)
    settled, _ = env.get_object("shovel_0").get_pose()
    assert np.allclose(settled[:2], position[:2], atol=5e-3)
    assert settled[2] > -1e-3


def test_the_pit_is_open_and_the_floor_around_it_is_solid(env):
    """Ray casts against the compiled model: the opening is PIT_DEPTH deep, the
    slab beside it is at z = 0, and the walls are vertical at the opening's edge."""
    env.reset(seed=0)
    model, data = _model_and_data(env)
    geom_id = np.zeros(1, dtype=np.int32)

    def hit_distance(origin, direction):
        return mujoco.mj_ray(  # pylint: disable=no-member
            model,
            data,
            np.array(origin, dtype=np.float64),
            np.array(direction, dtype=np.float64),
            None,
            1,
            -1,
            geom_id,
        )

    centre = [(PIT_X[0] + PIT_X[1]) / 2, 0.0]
    assert hit_distance([*centre, 1.0], [0, 0, -1]) == pytest.approx(1.0 + PIT_DEPTH)
    assert hit_distance([PIT_X[0] - 0.1, 0.0, 1.0], [0, 0, -1]) == pytest.approx(1.0)
    assert hit_distance([*centre[:1], 0.3, 1.0], [0, 0, -1]) == pytest.approx(1.0)
    # Vertical rims: a horizontal ray from inside the pit meets each wall exactly
    # at the opening's edge, at any depth.
    for depth in (-0.01, -PIT_DEPTH / 2, -PIT_DEPTH + 0.01):
        origin = [*centre, depth]
        assert hit_distance(origin, [-1, 0, 0]) == pytest.approx(centre[0] - PIT_X[0])
        assert hit_distance(origin, [1, 0, 0]) == pytest.approx(PIT_X[1] - centre[0])
        assert hit_distance(origin, [0, -1, 0]) == pytest.approx(-PIT_Y[0])
        assert hit_distance(origin, [0, 1, 0]) == pytest.approx(PIT_Y[1])


def test_the_robot_parks_at_the_near_rim_without_falling(env):
    """The base is a slide-joint body: parked at the near rim it holds its height
    and no robot geom goes below the floor."""
    env.reset(seed=0)
    model, data = _model_and_data(env)
    base = model.body("robot_base_link").id
    robot_geoms = [
        g
        for g in range(model.ngeom)
        if (model.geom(g).name or "").startswith("robot_")
        or (model.body(model.geom_bodyid[g]).name or "").startswith("robot")
    ]
    before = float(data.xpos[base][2])
    env._robot_env.set_robot_base_pos_yaw(  # pylint: disable=protected-access
        PIT_X[0] - 0.35, 0.0, 0.0
    )
    _settle(env, steps=300)
    assert float(data.xpos[base][2]) == pytest.approx(before, abs=1e-6)
    assert float(data.xpos[base][0]) == pytest.approx(PIT_X[0] - 0.35, abs=2e-2)
    assert min(float(data.geom_xpos[g][2]) for g in robot_geoms) >= 0.0


def _base_xy(env):
    model, data = _model_and_data(env)
    return np.array(data.xpos[model.body("robot_base_link").id][:2])


def _drive(env, *, base_step, steps=40, arm_step=None):
    # Actions are per-step deltas (the TidyBot env's default act_delta).
    arm = np.zeros(7) if arm_step is None else np.asarray(arm_step)
    for _ in range(steps):
        env.step(np.r_[base_step, arm, 0.0])


def test_the_wheels_are_the_real_tidybot_caster_positions():
    """The real TidyBot++ base puts each caster's steer axis 0.190 m forward or back
    and 0.170 m left or right of the vehicle centre; the wheel contact trails that
    by about 1.4 cm, so the steer axis is the wheel's position."""
    assert sorted(WHEEL_OFFSETS) == sorted(
        (sx * 0.190150, sy * 0.170150) for sx in (-1, 1) for sy in (-1, 1)
    )


def test_the_robot_is_not_stuck_at_reset(env):
    """The start pose keeps every wheel on the floor."""
    env.reset(seed=0)
    assert not env.robot_stuck_in_hole()


@pytest.mark.parametrize("inside", [0.02, -0.02])
def test_one_wheel_over_the_rim_is_stuck_and_none_is_not(env, inside):
    """Stuck as soon as one wheel point is inside the opening, not before."""
    env.reset(seed=0)
    robot = env._robot_env  # pylint: disable=protected-access
    front_wheel_x = max(offset[0] for offset in WHEEL_OFFSETS)
    # Offset in y so only the +y front wheel is over the opening.
    y = -0.1
    robot.set_robot_base_pos_yaw(PIT_X[0] - front_wheel_x + inside, y, 0.0)
    assert env.robot_stuck_in_hole() == (inside > 0)


def test_a_base_centred_over_the_pit_is_stuck(env):
    """The caster track is narrower than the opening, so a base driving down the
    middle of the pit cannot straddle it."""
    env.reset(seed=0)
    env._robot_env.set_robot_base_pos_yaw(1.0, 0.0, 0.0)  # pylint: disable=W0212
    assert env.robot_stuck_in_hole()


def test_a_stuck_base_cannot_drive_out_but_its_arm_still_moves(env):
    """Base commands are ignored while stuck; arm commands still act."""
    env.reset(seed=0)
    robot = env._robot_env  # pylint: disable=protected-access
    robot.set_robot_base_pos_yaw(1.0, 0.1, 0.0)
    assert env.robot_stuck_in_hole()
    held = _base_xy(env)
    arm_before = np.array(robot.qpos["arm"])
    _drive(env, base_step=[-0.02, -0.02, 0.0], arm_step=np.r_[0.3 / 40, np.zeros(6)])
    assert _base_xy(env) == pytest.approx(held, abs=1e-2)
    assert env.robot_stuck_in_hole()
    assert robot.qpos["arm"][0] - arm_before[0] > 0.03


def test_driving_into_the_pit_gets_stuck_at_the_rim(env):
    """A base driving toward the pit stops once a wheel crosses the rim."""
    env.reset(seed=0)
    robot = env._robot_env  # pylint: disable=protected-access
    front_wheel_x = max(offset[0] for offset in WHEEL_OFFSETS)
    robot.set_robot_base_pos_yaw(PIT_X[0] - front_wheel_x - 0.1, 0.1, 0.0)
    _drive(env, base_step=[0.02, 0.0, 0.0], steps=40)
    assert env.robot_stuck_in_hole()
    assert _base_xy(env)[0] + front_wheel_x < PIT_X[0] + 0.1


def test_teleporting_the_base_out_frees_it(env):
    """Setting the base pose directly is the only way out, and it works."""
    env.reset(seed=0)
    robot = env._robot_env  # pylint: disable=protected-access
    robot.set_robot_base_pos_yaw(1.0, 0.1, 0.0)
    assert env.robot_stuck_in_hole()
    robot.set_robot_base_pos_yaw(-0.4, -0.6, 0.0)
    assert not env.robot_stuck_in_hole()
    _drive(env, base_step=[0.005, 0.0, 0.0])
    assert _base_xy(env)[0] == pytest.approx(-0.2, abs=3e-2)


def _collides(model, first, second):
    return bool(
        model.geom_contype[first] & model.geom_conaffinity[second]
        or model.geom_contype[second] & model.geom_conaffinity[first]
    )


def _robot_geoms_by_part(model):
    gripper_root = model.body("robot_base").id
    parts = {"gripper": [], "arm_or_base": []}
    for geom in range(model.ngeom):
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        body = int(model.geom_bodyid[geom])
        if not (model.body(body).name or "").startswith("robot"):
            continue
        chain = body
        while chain and chain != gripper_root:
            chain = int(model.body_parentid[chain])
        parts["gripper" if chain == gripper_root else "arm_or_base"].append(geom)
    return parts


def test_cubes_collide_with_the_gripper_but_not_the_arm_or_base(env):
    """The robot must pick cubes out of the pit, so the gripper touches them; the
    arm and base still pass through them so driving and reaching do not shove
    them around."""
    env.reset(seed=0)
    model, _ = _model_and_data(env)
    parts = _robot_geoms_by_part(model)
    cubes = [
        g
        for g in range(model.ngeom)
        if (model.body(int(model.geom_bodyid[g])).name or "").startswith("cube_")
    ]
    assert parts["gripper"] and parts["arm_or_base"] and cubes
    for cube in cubes:
        assert all(_collides(model, cube, g) for g in parts["gripper"])
        assert not any(_collides(model, cube, g) for g in parts["arm_or_base"])
