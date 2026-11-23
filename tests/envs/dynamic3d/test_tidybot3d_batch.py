"""Tests for the parallel TidyBot environment on MuJoCo Warp.

These tests compare the batch environment with the single-world ``TidyBot3DEnv``. They
are skipped without ``mujoco_warp``.
"""

# pylint: disable=protected-access,wrong-import-position,wrong-import-order
# pylint: disable=ungrouped-imports

from pathlib import Path

import numpy as np
import pytest

wp = pytest.importorskip("warp")
pytest.importorskip("mujoco_warp")

import kinder  # noqa: E402
from kinder.envs.dynamic3d import tidybot_batch_env  # noqa: E402
from kinder.envs.dynamic3d.envs import TidyBot3DConfig  # noqa: E402
from kinder.envs.dynamic3d.tidybot_batch_env import (  # noqa: E402
    TIDYBOT_BATCH_TIMESTEP,
    TidyBot3DBatchEnv,
)
from kinder.utils import find_all_demo_files, load_demo  # noqa: E402

TASKS_DIR = Path(tidybot_batch_env.__file__).parent / "tasks"
PICK_PLACE = str(TASKS_DIR / "PickPlace3D" / "PickPlace3D-o1.json")
NUM_WORLDS = 8
# The time step of the single-world environment.
SINGLE_ENV_TIMESTEP = 0.0005


@pytest.fixture(name="pick_place_env", scope="module")
def _pick_place_env():
    env = TidyBot3DBatchEnv(PICK_PLACE, num_worlds=NUM_WORLDS, seed=0)
    yield env
    env.close()


def _copy_world_to_single_env(env, world):
    """Put the full simulation state of one world into the single environment."""
    env._mj_data.qpos[:] = env._data.qpos.numpy()[world]
    env._mj_data.qvel[:] = env._data.qvel.numpy()[world]
    env._mj_data.ctrl[:] = env._data.ctrl.numpy()[world]
    env._robot_env.sim.forward()


def _random_actions(env, seed):
    """Small base and arm motions, and a random gripper command."""
    rng = np.random.default_rng(seed)
    actions = rng.uniform(-0.05, 0.05, (env.num_worlds, env.action_dim))
    actions[:, 10] = rng.uniform(0.0, 1.0, env.num_worlds)
    return actions.astype(np.float32)


def test_reset_gives_valid_and_different_worlds(pick_place_env):
    """Each world has its own initial state, and a full reset gives it again."""
    env = pick_place_env
    obs = env.reset().numpy().copy()
    assert obs.shape == (NUM_WORLDS, env.obs_dim)
    assert np.all(np.isfinite(obs))
    for world in range(NUM_WORLDS):
        assert env.observation_space.contains(obs[world])
    assert np.abs(obs[0] - obs[1]).max() > 1e-3

    env.step(_random_actions(env, seed=0))
    assert np.allclose(env.reset().numpy(), obs)


def test_time_step(pick_place_env):
    """The default physics time step is larger than that of the single environment."""
    env = pick_place_env
    assert env.physics_timestep == TIDYBOT_BATCH_TIMESTEP
    assert env.decimation == 50
    assert np.isclose(env.step_dt, 0.1)
    # The single environment keeps its time step.
    assert env._mj_model.opt.timestep == SINGLE_ENV_TIMESTEP
    with pytest.raises(AssertionError, match="multiple"):
        TidyBot3DBatchEnv(PICK_PLACE, num_worlds=1, physics_timestep=0.003)


@pytest.mark.parametrize("use_arm_velocities", [False, True])
def test_control_matches_single_env(use_arm_velocities):
    """The control law of the batch environment is that of ``TidyBotRobotEnv``."""
    config = TidyBot3DConfig(use_arm_velocities=use_arm_velocities)
    env = TidyBot3DBatchEnv(PICK_PLACE, num_worlds=4, seed=0, config=config)
    assert env.action_dim == (18 if use_arm_velocities else 11)
    env.reset()
    actions = _random_actions(env, seed=1)
    # The controls of a step come from the state before the step.
    expected = []
    for world in range(4):
        _copy_world_to_single_env(env, world)
        expected.append(env._robot_env._action_to_ctrl(actions[world]))
    env.step(actions)
    ctrl = env._data.ctrl.numpy()
    assert np.allclose(ctrl, np.array(expected), rtol=1e-4, atol=1e-3)
    env.close()


def test_batch_matches_single_env():
    """With the same time step, a world follows the single-world environment."""
    env = TidyBot3DBatchEnv(
        PICK_PLACE,
        num_worlds=NUM_WORLDS,
        seed=0,
        physics_timestep=SINGLE_ENV_TIMESTEP,
    )
    single_env = env._single_env
    assert env.decimation == 200
    for world in (0, 3):
        env.reset()
        _copy_world_to_single_env(env, world)
        for step in range(5):
            actions = _random_actions(env, seed=step)
            batch_obs, _, _, _, _ = env.step(actions)
            single_obs, _, _, _, _ = single_env.step(actions[world])
        # float32 on the GPU against float64 on the CPU, for 0.5 s of motion: the
        # positions agree to millimeters, and the velocities are less exact.
        difference = np.abs(batch_obs.numpy()[world] - single_obs)
        velocity_columns = np.zeros(env.obs_dim, dtype=bool)
        velocity_columns[env._obs_qvel_column] = True
        assert difference[~velocity_columns].max() < 5e-3
        assert difference[velocity_columns].max() < 0.1
    env.close()


def test_goal_check_matches_single_env(pick_place_env):
    """The batch goal check agrees with the goal check of the single environment."""
    env = pick_place_env
    env.reset()
    # Move the cube to the goal region in each second world.
    cube = env._object_centric_env.get_object("cube1")
    cube_adr = int(env._mj_model.joint(cube.joint_name).qposadr[0])
    qpos = env._data.qpos.numpy()
    qpos[::2, cube_adr : cube_adr + 2] = [1.1, 0.5]
    env._data.qpos.assign(qpos)

    _, reward, terminated, truncated, info = env.step(env.default_action())
    terminated = terminated.numpy()
    assert terminated.tolist() == [world % 2 == 0 for world in range(NUM_WORLDS)]
    assert np.all(reward.numpy() == -1.0)
    assert not truncated.numpy().any()
    assert not info

    object_centric_env = env._object_centric_env
    for world in range(NUM_WORLDS):
        object_centric_env.set_state(env.get_object_centric_state(world))
        assert object_centric_env._check_goals() == terminated[world]


def test_masked_reset(pick_place_env):
    """A masked reset changes the worlds in the mask only."""
    env = pick_place_env
    initial = env.reset().numpy().copy()
    for step in range(3):
        obs, _, _, _, _ = env.step(_random_actions(env, seed=step))
    stepped = obs.numpy().copy()
    mask = np.zeros(NUM_WORLDS, dtype=bool)
    mask[[1, 4]] = True
    obs = env.reset(mask).numpy()
    assert np.allclose(obs[~mask], stepped[~mask])
    for world in (1, 4):
        # The world is in one of the stored initial states.
        assert np.isclose(obs[world], initial, atol=1e-6).all(axis=1).any()
    assert env._step_count.numpy().tolist() == [0 if m else 3 for m in mask]


def test_render(pick_place_env):
    """A world can be rendered with the camera of the single environment."""
    env = pick_place_env
    env.reset()
    image = env.render(world=1)
    assert image.ndim == 3 and image.shape[2] == 3
    assert image.std() > 1.0


def test_make_batch():
    """``kinder.make_batch`` makes the batch version of a registered environment."""
    kinder.register_all_environments()
    env = kinder.make_batch("kinder/PickPlace3D-o1-v0", num_worlds=2, seed=0)
    assert isinstance(env, TidyBot3DBatchEnv)
    assert env.reset().shape == (2, env.obs_dim)
    env.close()

    # The tossing tasks have velocity targets for the arm, as in Tossing3DEnv.
    env = kinder.make_batch("kinder/Tossing3D-o1-v0", num_worlds=2, seed=0)
    single_env = kinder.make("kinder/Tossing3D-o1-v0")
    assert env.action_dim == 18
    assert env.action_space.shape == single_env.action_space.shape
    assert env.observation_space.shape == single_env.observation_space.shape
    single_env.close()
    env.close()

    with pytest.raises(ValueError, match="no batch version"):
        kinder.make_batch("kinder/Motion2D-p0-v0", num_worlds=2)


def test_demonstration_reaches_goal():
    """A recorded demonstration reaches the goal with the default time step.

    The demonstration was recorded in ``TidyBot3DEnv`` (CPU MuJoCo, a time step of
    0.0005 s). Its actions are replayed with no feedback.
    """
    kinder.register_all_environments()
    demo_files = [f for f in find_all_demo_files() if f.parts[-3] == "Shelf3D-o1"]
    if not demo_files:
        pytest.skip("The demonstrations are not available")
    demo = load_demo(sorted(demo_files)[0])
    env = kinder.make_batch(demo["env_id"], num_worlds=1, seed=demo["seed"])
    # Start in the initial state of the demonstration: the reset of the single
    # environment with the seed of the demonstration.
    env._single_env.reset(seed=demo["seed"])
    mj_data = env._single_env._object_centric_env._robot_env.sim.data.mj_data
    env._start_qpos.assign(mj_data.qpos[None].astype(np.float32))
    env._start_qvel.assign(mj_data.qvel[None].astype(np.float32))
    env.reset()
    reached = False
    for action in demo["actions"]:
        _, _, terminated, _, _ = env.step(np.asarray(action, dtype=np.float32)[None])
        reached = reached or bool(terminated.numpy()[0])
    assert reached
    env.close()


def test_drifted_starts_are_replaced():
    """A start state in which a ground object moved too far is replaced."""
    task = str(TASKS_DIR / "Dynamo3D" / "Dynamo3D-o1.json")
    env = TidyBot3DBatchEnv(task, num_worlds=6, seed=0)
    drift_limit = env._object_centric_env.task_config["settle_drift_limit"]
    settled = env._start_qpos.numpy().copy()
    # Pretend that the chair of worlds 1 and 4 was placed far from where it is now.
    (chair_name,) = env._object_centric_env._ground_object_xy
    chair = env._object_centric_env.get_object(chair_name)
    chair_adr = int(env._mj_model.joint(chair.joint_name).qposadr[0])
    sampled = settled.copy()
    sampled[[1, 4], chair_adr] += 2 * drift_limit
    env._replace_drifted_starts(sampled)
    replaced = env._start_qpos.numpy()
    good = [0, 2, 3, 5]
    assert np.array_equal(replaced[good], settled[good])
    for world in (1, 4):
        assert any(np.array_equal(replaced[world], settled[g]) for g in good)
    env.close()


@pytest.mark.parametrize(
    "task",
    [
        "Shelf3D/Shelf3D-o8.json",
        "BalanceBeam3D/BalanceBeam3D-o3.json",
        "ConstrainedCupboard3D/ConstrainedCupboard3D-o6.json",
        "ScoopPour3D/ScoopPour3D-o10.json",
        "SortClutteredBlocks3D/SortClutteredBlocks3D-o4-"
        "sort_the_cluttered_blocks_into_bins.json",
        "SweepIntoDrawer3D/SweepIntoDrawer3D-o5.json",
        "SweepSimple3D/SweepSimple3D-o1-"
        "sweep_the_blocks_to_the_left_side_of_the_kitchen_island.json",
        "Rearrange3D/Rearrange3D-o2-"
        "put_the_boxed_drink_and_the_can_next_to_the_bowl.json",
    ],
)
def test_other_task_families(task):
    """The batch environment builds the larger tasks, and its states are settled."""
    env = TidyBot3DBatchEnv(str(TASKS_DIR / task), num_worlds=3, seed=0)
    obs = env.reset().numpy().copy()
    assert np.all(np.isfinite(obs))
    for world in range(3):
        assert env.observation_space.contains(obs[world])
    # The objects do not travel. (An object can still fall from a pile after the
    # settle time, as in CPU MuJoCo. So the angular velocities are not checked, and
    # the limit has a margin for it.)
    space = env.observation_space
    names = [
        feature
        for obj in space.constant_objects
        for feature in space.type_features[obj.type]
    ]
    linear_columns = [i for i, name in enumerate(names) if name in ("vx", "vy", "vz")]
    assert np.abs(obs[:, linear_columns]).max() < 0.1
    # The robot holds its pose with the default action.
    for _ in range(3):
        next_obs, _, terminated, _, _ = env.step(env.default_action())
    robot_columns = [i for i, name in enumerate(names) if name.startswith("pos_")]
    assert (
        np.abs(next_obs.numpy()[:, robot_columns] - obs[:, robot_columns]).max() < 0.02
    )
    assert not terminated.numpy().any()
    usage = env.buffer_use()
    assert usage["contacts"] < usage["contacts_limit"]
    assert usage["constraints"] < usage["constraints_limit"]
    env.close()
