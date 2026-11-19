"""Tests for the Sprout humanoid environments (single world, on the CPU).

The Sprout model is not distributed, so the tests that need it are skipped when the
model files are not in ``kinder/envs/dynamic3d/models/fauna_sprout``.
"""

# pylint: disable=protected-access

import numpy as np
import pytest

import kinder
from kinder.envs.dynamic3d.robots.sprout_robot_env import (
    NUM_SPROUT_JOINTS,
    SPROUT_SIMULATION_TIMESTEP,
    dc_motor_torque,
    load_sprout_model,
    sprout_model_available,
)

needs_sprout_model = pytest.mark.skipif(
    not sprout_model_available(), reason="The Sprout model is not distributed"
)

SPROUT_FAMILIES = {
    "SproutBalanceBeam3D": 1,
    "SproutConstrainedCupboard3D": 6,
    "SproutDynamo3D": 1,
    "SproutPickPlace3D": 1,
    "SproutRearrange3D": 10,
    "SproutScoopPour3D": 6,
    "SproutShelf3D": 6,
    "SproutSortClutteredBlocks3D": 4,
    "SproutSweepIntoDrawer3D": 1,
    "SproutSweepSimple3D": 6,
    "SproutTossing3D": 2,
}


def _sprout_env_ids() -> list[str]:
    """All registered Sprout environments (none without the Sprout model)."""
    kinder.register_all_environments()
    return sorted(i for i in kinder.get_all_env_ids() if i.startswith("kinder/Sprout"))


@needs_sprout_model
def test_sprout_envs_are_registered():
    """Each Sprout task family registers all its variants."""
    env_ids = _sprout_env_ids()
    for family, num_variants in SPROUT_FAMILIES.items():
        variants = [i for i in env_ids if i.startswith(f"kinder/{family}-")]
        assert len(variants) == num_variants, (family, variants)
    assert len(env_ids) == sum(SPROUT_FAMILIES.values())


def _motor_torque(target: float, position: float, velocity: float) -> float:
    """Torque of a motor with stiffness 10, damping 1, limit 4, stall 8, no-load 20."""
    one = np.ones(1)
    return float(
        dc_motor_torque(
            target * one,
            position * one,
            velocity * one,
            stiffness=10.0 * one,
            damping=1.0 * one,
            effort_limit=4.0 * one,
            saturation_effort=8.0 * one,
            velocity_limit=20.0 * one,
        )[0]
    )


def test_dc_motor_torque():
    """The motor model is a PD law, clipped to the torque-speed curve."""
    # Inside all limits: the PD law.
    assert np.isclose(_motor_torque(0.1, 0.0, 0.5), 10.0 * 0.1 - 0.5)
    # A large error at rest: the continuous effort limit.
    assert np.isclose(_motor_torque(1.0, 0.0, 0.0), 4.0)
    assert np.isclose(_motor_torque(-1.0, 0.0, 0.0), -4.0)
    # At 15 rad/s, the torque-speed curve permits 8 * (1 - 15 / 20) = 2 forward.
    assert np.isclose(_motor_torque(3.0, 0.0, 15.0), 2.0)
    # At the no-load speed, the motor cannot push forward.
    assert np.isclose(_motor_torque(5.0, 0.0, 20.0), 0.0)
    # Against the motion, the motor can brake with the effort limit.
    assert np.isclose(_motor_torque(-3.0, 0.0, 15.0), -4.0)


@needs_sprout_model
def test_sprout_model_parameters():
    """Each joint has a motor, and the edited XML has torque motors only."""
    model = load_sprout_model()
    assert len(model.joint_names) == NUM_SPROUT_JOINTS
    for values in (
        model.default_joint_positions,
        model.stiffness,
        model.damping,
        model.effort_limit,
        model.saturation_effort,
        model.velocity_limit,
    ):
        assert values.shape == (NUM_SPROUT_JOINTS,)
    assert np.all(model.effort_limit <= model.saturation_effort)
    assert "<position name=" not in model.xml_string
    assert model.xml_string.count("<motor name=") == NUM_SPROUT_JOINTS


@needs_sprout_model
@pytest.mark.parametrize("env_id", _sprout_env_ids())
def test_sprout_env_reset_and_step(env_id):
    """The environment runs at 50 Hz with a 0.002 s timestep, and it is
    deterministic."""
    kinder.register_all_environments()
    env = kinder.make(env_id)
    obs, _ = env.reset(seed=123)
    assert env.observation_space.contains(obs)
    assert env.action_space.shape == (NUM_SPROUT_JOINTS,)

    robot_env = env.unwrapped._object_centric_env._robot_env
    assert robot_env.sim.model.mj_model.opt.timestep == SPROUT_SIMULATION_TIMESTEP
    assert robot_env.decimation == 10

    # After reset, the robot stands in the default pose, at rest, and the goal does
    # not hold. The objects rest on a support (they do not fall).
    standing_height = robot_env.model.standing_height
    assert np.isclose(robot_env.qpos["base"][2], standing_height)
    assert np.allclose(
        robot_env.qpos["joints"], robot_env.model.default_joint_positions
    )
    assert np.all(robot_env.qvel["joints"] == 0.0)
    object_centric_env = env.unwrapped._object_centric_env
    assert not object_centric_env._check_goals()
    for mujoco_object in object_centric_env._objects:
        linear_velocity, _ = robot_env.get_joint_vel(mujoco_object.joint_name)
        assert np.abs(linear_velocity).max() < 0.05, mujoco_object.name
        # An object that the sampler could not place stays at the origin.
        position, _ = robot_env.get_joint_pos_quat(mujoco_object.joint_name)
        assert np.abs(position).max() > 0.02, mujoco_object.name
    time_before = robot_env.sim.data.mj_data.time

    hold = robot_env.model.default_joint_positions.astype(np.float32)
    trajectory = []
    for _ in range(10):
        obs, reward, terminated, truncated, _ = env.step(hold)
        trajectory.append(obs)
        assert reward == -1.0
        assert not terminated and not truncated
        assert np.all(np.isfinite(obs))
    assert np.isclose(robot_env.sim.data.mj_data.time - time_before, 10 * 0.02)
    # 0.2 s after reset, the robot has not fallen.
    assert robot_env.qpos["base"][2] > 0.9 * standing_height

    # The same seed and actions give the same trajectory.
    obs, _ = env.reset(seed=123)
    for expected in trajectory:
        obs, _, _, _, _ = env.step(hold)
        assert np.allclose(obs, expected, atol=1e-6)
    env.close()


@needs_sprout_model
def test_sprout_env_set_state():
    """A state that is set again gives the same next state."""
    kinder.register_all_environments()
    env = kinder.make("kinder/SproutPickPlace3D-o1-v0", allow_state_access=True)
    env.reset(seed=0)
    unwrapped = env.unwrapped
    hold = unwrapped._object_centric_env._robot_env.model.default_joint_positions
    hold = hold.astype(np.float32)
    for _ in range(5):
        env.step(hold)
    state = unwrapped.get_state()
    next_obs, _, _, _, _ = env.step(hold)
    unwrapped.set_state(state)
    assert np.allclose(unwrapped.get_state(), state)
    replayed_obs, _, _, _, _ = env.step(hold)
    assert np.allclose(replayed_obs, next_obs, atol=1e-3)
    env.close()


@needs_sprout_model
def test_sprout_pick_place_goal():
    """The goal holds when the cube is in the goal region on the table."""
    kinder.register_all_environments()
    env = kinder.make("kinder/SproutPickPlace3D-o1-v0", allow_state_access=True)
    env.reset(seed=0)
    object_centric_env = env.unwrapped._object_centric_env
    assert not object_centric_env._check_goals()

    cube = object_centric_env.get_object("cube1")
    position = np.array(cube.get_position())
    # The goal region is 0.25 m to the left of the initial region, on the table.
    position[1] += 0.25
    cube.set_pose(position, [1.0, 0.0, 0.0, 0.0])
    object_centric_env._robot_env.sim.forward()
    object_centric_env._current_state = object_centric_env._get_object_centric_state()
    assert object_centric_env._check_goals()
    env.close()


@needs_sprout_model
def test_sprout_command_delay():
    """The targets reach the motors after the configured number of physics steps."""
    kinder.register_all_environments()

    # The default is the random delay of the RL tasks.
    env = kinder.make("kinder/SproutPickPlace3D-o1-v0")
    env.reset(seed=0)
    robot_env = env.unwrapped._object_centric_env._robot_env
    assert robot_env.command_delay_steps == (5, 20)
    assert robot_env.command_delay_update_period == 500
    hold = robot_env.model.default_joint_positions.astype(np.float32)
    lags = set()
    for _ in range(120):  # 1200 physics steps: more than two update periods
        env.step(hold)
        lags.add(robot_env._delay_lag)
    assert max(lags) > 0
    assert all(lag == 0 or 5 <= lag <= 20 for lag in lags)
    env.close()

    # A fixed delay of 3 physics steps.
    env = kinder.make(
        "kinder/SproutPickPlace3D-o1-v0",
        command_delay_steps=(3, 3),
        command_delay_update_period=0,
    )
    env.reset(seed=0)
    robot_env = env.unwrapped._object_centric_env._robot_env
    robot_env.reset_command_delay()
    for k in range(10):
        targets = np.full(NUM_SPROUT_JOINTS, float(k), dtype=np.float32)
        delayed = robot_env._delay_targets(targets)
        # Before 3 steps have passed, the oldest target is used.
        assert np.all(delayed == max(k - 3, 0))
    env.close()

    # No delay.
    env = kinder.make("kinder/SproutPickPlace3D-o1-v0", command_delay_steps=(0, 0))
    env.reset(seed=0)
    robot_env = env.unwrapped._object_centric_env._robot_env
    targets = np.ones(NUM_SPROUT_JOINTS, dtype=np.float32)
    assert robot_env._delay_targets(targets) is targets
    env.close()


@needs_sprout_model
def test_sprout_dynamo_chair_mass():
    """The chair of the Sprout task has a realistic mass, not that of a solid mesh."""
    kinder.register_all_environments()
    for env_id, expected_mass in (
        ("kinder/SproutDynamo3D-o1-v0", 5.0),
        ("kinder/Dynamo3D-o1-v0", None),
    ):
        env = kinder.make(env_id)
        env.reset(seed=0)
        mj_model = env.unwrapped._object_centric_env._robot_env.sim.model.mj_model
        mass = float(mj_model.body("obstacle_chair").subtreemass[0])
        if expected_mass is None:
            assert mass > 100.0  # the TidyBot task is not changed
        else:
            assert np.isclose(mass, expected_mass)
        env.close()
