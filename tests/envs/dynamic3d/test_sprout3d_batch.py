"""Tests for the parallel Sprout environment on MuJoCo Warp.

These tests compare the batch environment with the single-world ``Sprout3DEnv``. They
are skipped without ``mujoco_warp`` or without the (not distributed) Sprout model.
"""

# pylint: disable=protected-access,wrong-import-position,wrong-import-order
# pylint: disable=ungrouped-imports

import json
from pathlib import Path

import numpy as np
import pytest

from kinder.envs.dynamic3d.robots.sprout_robot_env import (
    NUM_SPROUT_JOINTS,
    dc_motor_torque,
    sprout_model_available,
)

wp = pytest.importorskip("warp")
pytest.importorskip("mujoco_warp")
if not sprout_model_available():
    pytest.skip("The Sprout model is not distributed", allow_module_level=True)

from kinder.envs.dynamic3d import sprout_batch_env  # noqa: E402
from kinder.envs.dynamic3d.sprout_batch_env import (  # noqa: E402
    Sprout3DBatchEnv,
    _dc_motor_kernel,
)

TASKS_DIR = Path(sprout_batch_env.__file__).parent / "tasks"
PICK_PLACE = str(TASKS_DIR / "SproutPickPlace3D" / "SproutPickPlace3D-o1.json")
DYNAMO = str(TASKS_DIR / "SproutDynamo3D" / "SproutDynamo3D-o1.json")
NUM_WORLDS = 8


@pytest.fixture(name="pick_place_env", scope="module")
def _pick_place_env():
    env = Sprout3DBatchEnv(PICK_PLACE, num_worlds=NUM_WORLDS, seed=0)
    yield env
    env.close()


def test_motor_kernel_matches_numpy():
    """The Warp motor kernel and the NumPy motor model give the same torques."""
    rng = np.random.default_rng(0)
    shape = (4, NUM_SPROUT_JOINTS)
    qpos = rng.uniform(-1, 1, shape)
    qvel = rng.uniform(-30, 30, shape)
    targets = rng.uniform(-2, 2, shape)
    stiffness = rng.uniform(5, 40, NUM_SPROUT_JOINTS)
    damping = rng.uniform(0.5, 2, NUM_SPROUT_JOINTS)
    effort_limit = rng.uniform(3, 30, NUM_SPROUT_JOINTS)
    saturation_effort = effort_limit * rng.uniform(1.2, 4, NUM_SPROUT_JOINTS)
    velocity_limit = rng.uniform(15, 40, NUM_SPROUT_JOINTS)
    expected = dc_motor_torque(
        targets,
        qpos,
        qvel,
        stiffness,
        damping,
        effort_limit,
        saturation_effort,
        velocity_limit,
    )

    def on_device(values):
        return wp.array(values, dtype=float)

    ctrl = wp.zeros(shape, dtype=float)
    wp.launch(
        _dc_motor_kernel,
        dim=shape,
        inputs=[
            on_device(qpos),
            on_device(qvel),
            on_device(targets),
            0,
            0,
            0,
            on_device(stiffness),
            on_device(damping),
            on_device(effort_limit),
            on_device(saturation_effort),
            on_device(velocity_limit),
        ],
        outputs=[ctrl],
    )
    assert np.allclose(ctrl.numpy(), expected, atol=1e-4)


def test_reset_gives_valid_and_different_worlds(pick_place_env):
    """Each world has its own initial state, and a full reset gives it again."""
    env = pick_place_env
    obs = env.reset().numpy()
    assert obs.shape == (NUM_WORLDS, env.obs_dim)
    assert np.all(np.isfinite(obs))
    for world in range(NUM_WORLDS):
        assert env.observation_space.contains(obs[world])
    assert np.abs(obs[0] - obs[1]).max() > 1e-3

    env.step(env.default_action())
    assert np.allclose(env.reset().numpy(), obs)


def test_batch_matches_single_env(pick_place_env):
    """A world of the batch environment follows the single-world environment."""
    env = pick_place_env
    single_env = env._single_env
    obs = env.reset().numpy()
    hold = env.default_action()
    hold_single = hold.numpy()[0]
    for world in (0, 3):
        env.reset()
        single_env.set_state(obs[world])
        for _ in range(10):
            batch_obs, _, _, _, _ = env.step(hold)
            single_obs, _, _, _, _ = single_env.step(hold_single)
        # float32 on the GPU against float64 on the CPU, for 0.2 s of contact: the
        # positions agree to millimeters, and the velocities are less exact.
        difference = np.abs(batch_obs.numpy()[world] - single_obs)
        velocity_columns = np.zeros(env.obs_dim, dtype=bool)
        velocity_columns[env._obs_qvel_column] = True
        assert difference[~velocity_columns].max() < 5e-3
        assert difference[velocity_columns].max() < 0.1


def test_goal_check_matches_single_env(pick_place_env):
    """The batch goal check agrees with the goal check of the single environment."""
    env = pick_place_env
    env.reset()
    # Move the cube to the goal region in each second world.
    cube = env._object_centric_env.get_object("cube1")
    cube_adr = int(env._mj_model.joint(cube.joint_name).qposadr[0])
    qpos = env._data.qpos.numpy()
    qpos[::2, cube_adr + 1] += 0.25
    env._data.qpos.assign(qpos)

    _, reward, terminated, truncated, info = env.step(env.default_action())
    terminated = terminated.numpy()
    assert terminated.tolist() == [world % 2 == 0 for world in range(NUM_WORLDS)]
    assert np.all(reward.numpy() == -1.0)
    assert not truncated.numpy().any()
    assert not info["fallen"].numpy().any()

    object_centric_env = env._object_centric_env
    for world in range(NUM_WORLDS):
        object_centric_env.set_state(env.get_object_centric_state(world))
        assert object_centric_env._check_goals() == terminated[world]


def test_masked_reset(pick_place_env):
    """A masked reset changes the worlds in the mask only."""
    env = pick_place_env
    initial = env.reset().numpy().copy()
    for _ in range(5):
        obs, _, _, _, _ = env.step(env.default_action())
    stepped = obs.numpy().copy()
    mask = np.zeros(NUM_WORLDS, dtype=bool)
    mask[[1, 4]] = True
    obs = env.reset(mask).numpy()
    assert np.allclose(obs[~mask], stepped[~mask])
    for world in (1, 4):
        # The world is in one of the stored initial states.
        assert np.isclose(obs[world], initial, atol=1e-6).all(axis=1).any()
    assert env._step_count.numpy().tolist() == [0 if m else 5 for m in mask]


def test_robot_goal_and_truncation():
    """A goal on the robot base works, and the horizon truncates the worlds."""
    env = Sprout3DBatchEnv(DYNAMO, num_worlds=4, seed=0, horizon=3)
    env.reset()
    # Stand the robot of world 2 in the goal region on the floor. The chair starts
    # in that region, so move the chair away first.
    chair = env._object_centric_env.get_object("obstacle_chair")
    chair_adr = int(env._mj_model.joint(chair.joint_name).qposadr[0])
    base_adr = env._robot_env.base_qpos_adr
    qpos = env._data.qpos.numpy()
    qpos[2, chair_adr : chair_adr + 2] = [5.0, 5.0]
    qpos[2, base_adr : base_adr + 2] = [1.0, 0.0]
    env._data.qpos.assign(qpos)
    for step in range(3):
        _, _, terminated, truncated, _ = env.step(env.default_action())
        assert terminated.numpy().tolist() == [False, False, True, False]
        assert truncated.numpy().all() == (step == 2)
    env.close()


def test_command_delay():
    """The motors get the targets of the configured number of physics steps ago."""
    default = None
    final_obs = {}
    # One control step is 10 physics steps. With a delay of 10 physics steps, the
    # motors use the action of the control step before at the end of a control
    # step. With a delay of 3 or 0, they use the action of the same control step.
    for delay, expected_offset in ((10, 1), (3, 0), (0, 0)):
        env = Sprout3DBatchEnv(
            PICK_PLACE,
            num_worlds=2,
            seed=0,
            command_delay_steps=(delay, delay),
            command_delay_update_period=0,
        )
        env.reset()
        default = env.default_action().numpy()
        actions = [default + 0.1 * np.sin(0.5 * step) for step in range(10)]
        for step, action in enumerate(actions):
            obs, _, _, _, _ = env.step(action)
            if delay > 0:
                expected = actions[max(step - expected_offset, 0)]
                assert np.allclose(env._delayed_targets.numpy(), expected, atol=1e-6)
                assert env._delay_lag.numpy().tolist() == [delay, delay]
        final_obs[delay] = obs.numpy().copy()
        env.close()
    # The delay changes the motion.
    assert np.abs(final_obs[10] - final_obs[0]).max() > 1e-2


def test_default_command_delay_is_random(pick_place_env):
    """By default, each world draws its own delay from the range of the RL tasks."""
    env = pick_place_env
    env.reset()
    assert (env._delay_min_lag, env._delay_max_lag) == (5, 20)
    assert env._delay_lag.numpy().tolist() == [0] * NUM_WORLDS
    for _ in range(60):  # 600 physics steps: more than one update period
        env.step(env.default_action())
    lags = env._delay_lag.numpy()
    assert np.all((lags >= 5) & (lags <= 20))
    assert len(set(lags.tolist())) > 1


def test_balanced_goal_matches_single_env(tmp_path):
    """The "balanced" goal predicate agrees with the single environment."""
    task = json.loads(
        (TASKS_DIR / "BalanceBeam3D" / "BalanceBeam3D-o3.json").read_text()
    )
    task["robots"] = {"sprout": {"robot": {}}}
    task["goal_state"] = [["balanced", "seesaw_1", 5.0]]
    task_path = tmp_path / "SproutBalanceBeam3D-o3.json"
    task_path.write_text(json.dumps(task))

    env = Sprout3DBatchEnv(str(task_path), num_worlds=4, seed=0)
    env.reset()
    # Tilt the beam in worlds 1 and 3.
    beam = env._mj_model.body("seesaw_1_beam")
    hinge_adr = int(env._mj_model.jnt_qposadr[beam.jntadr[0]])
    qpos = env._data.qpos.numpy()
    qpos[[1, 3], hinge_adr] = 0.4
    env._data.qpos.assign(qpos)
    _, _, terminated, _, _ = env.step(env.default_action())
    terminated = terminated.numpy()
    assert terminated.tolist() == [True, False, True, False]

    # The object-centric state does not have the angle of the beam, so copy the
    # full simulation state of each world to the single environment.
    object_centric_env = env._object_centric_env
    qpos = env._data.qpos.numpy()
    for world in range(4):
        env._mj_data.qpos[:] = qpos[world]
        env._robot_env.sim.forward()
        assert object_centric_env._check_goals() == terminated[world]
    env.close()
