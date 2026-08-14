"""Tests for base_limbrepositioning3d.py.

The base class is abstract, so these exercise it through a minimal concrete subclass
rather than through the full LimbRepositioning3D environment.
"""

import numpy as np
import pybullet as p
import pytest
from pybullet_helpers.geometry import Pose, SE2Pose
from relational_structs.spaces import ObjectCentricStateSpace

from kinder.envs.dynamic3d.base_limbrepositioning3d import (
    Limb3DEnvConfig,
    ObjectCentricLimb3DRobotEnv,
)
from kinder.envs.dynamic3d.limb_scenes import LimbRepositioningSceneConfig
from kinder.envs.dynamic3d.limb_utils import (
    NUM_LIMB_JOINTS,
    NUM_ROBOT_JOINTS,
    LimbRepositioning3DObjectCentricState,
    LimbRepositioning3DRobotActionSpace,
)
from kinder.envs.kinematic3d.utils import extend_joints_to_include_fingers

# An isolated right arm, which needs no furniture and is reachable from this base pose.
LIMB_BASE_POSE = Pose.from_rpy((0.730, -0.396, 0.270), (0.0, 0.0, np.pi))
LIMB_INIT = (-0.403357, -0.447072, -0.092894, -0.501061, 0.0, 0.0)
LIMB_GOAL = (-0.7, 0.0, -0.03, -0.8, 0.0, 0.0)
ROBOT_BASE_POSE = SE2Pose(0.3566, 0.4849, -0.6971)
ROBOT_BASE_Z = -0.329


class _MinimalLimb3DEnv(ObjectCentricLimb3DRobotEnv[Limb3DEnvConfig]):
    """The smallest concrete environment the base class admits."""

    def _create_constant_initial_state(self):
        return self._get_obs()

    def goal_reached(self) -> bool:
        return False

    def _reset_bodies(self) -> None:
        self.robot.arm.set_joints(
            extend_joints_to_include_fingers(list(self._initial_robot_joints)),
            joint_velocities=[0.0] * len(self.robot.arm.arm_joints),
        )
        self.limb.set_joints(
            list(self.config.scene.limb_init_joint_positions),
            joint_velocities=[0.0] * NUM_LIMB_JOINTS,
        )
        self._regrasp_limb()


def _make_config(**kwargs) -> Limb3DEnvConfig:
    """A config for an isolated right arm, settling briefly to keep tests quick."""
    scene = LimbRepositioningSceneConfig(
        scene_type="isolated",
        limb_name="human-right-arm",
        limb_init_joint_positions=LIMB_INIT,
        limb_goal_joint_positions=LIMB_GOAL,
        limb_base_pose=LIMB_BASE_POSE,
    )
    return Limb3DEnvConfig(
        scene=scene,
        robot_base_home_pose=ROBOT_BASE_POSE,
        robot_base_z=ROBOT_BASE_Z,
        num_settle_steps=kwargs.pop("num_settle_steps", 100),
        **kwargs,
    )


@pytest.fixture(scope="module", name="env")
def _env():
    """A shared environment, since construction runs inverse kinematics and settles."""
    environment = _MinimalLimb3DEnv(config=_make_config())
    yield environment
    environment.close()


def test_action_space_is_one_torque_per_arm_joint(env):
    """The base class builds a torque action space from the config's limits."""
    assert isinstance(env.action_space, LimbRepositioning3DRobotActionSpace)
    assert env.action_space.shape == (NUM_ROBOT_JOINTS,)
    assert env.torque_action_space is env.action_space


def test_action_space_honours_the_configured_limits():
    """Torque limits come from the config rather than being hard-coded."""
    environment = _MinimalLimb3DEnv(
        config=_make_config(
            torque_lower_limits=(-3.0,) * NUM_ROBOT_JOINTS,
            torque_upper_limits=(4.0,) * NUM_ROBOT_JOINTS,
        )
    )
    try:
        assert np.all(environment.action_space.low == -3.0)
        assert np.all(environment.action_space.high == 4.0)
    finally:
        environment.close()


def test_observation_space_is_object_centric(env):
    """States round-trip through the observation space."""
    assert isinstance(env.observation_space, ObjectCentricStateSpace)
    obs, _ = env.reset()
    assert isinstance(obs, LimbRepositioning3DObjectCentricState)
    assert len(obs.limb_joint_positions) == NUM_LIMB_JOINTS
    assert len(obs.robot_joint_positions) == NUM_ROBOT_JOINTS


def test_robot_is_welded_to_the_limb(env):
    """The end effectors coincide, or the weld would yank the limb when stepping."""
    env.reset()
    robot_ee = env.robot.arm.get_end_effector_pose().position
    limb_ee = env.limb.get_end_effector_pose().position
    assert float(np.linalg.norm(np.subtract(robot_ee, limb_ee))) < 0.01


def test_weld_survives_stepping(env):
    """Applying torque moves the pair together rather than pulling them apart."""
    env.reset()
    action = np.full(NUM_ROBOT_JOINTS, env.action_space.high[0], dtype=np.float32)
    for _ in range(50):
        env.step(action)
    robot_ee = env.robot.arm.get_end_effector_pose().position
    limb_ee = env.limb.get_end_effector_pose().position
    assert float(np.linalg.norm(np.subtract(robot_ee, limb_ee))) < 0.05


def test_torque_moves_the_passive_limb(env):
    """The limb is passive, so it only moves because the robot moves it."""
    obs, _ = env.reset()
    start = np.array(obs.limb_joint_positions)
    action = np.full(NUM_ROBOT_JOINTS, env.action_space.high[0], dtype=np.float32)
    for _ in range(50):
        obs, _, _, _, _ = env.step(action)
    assert not np.allclose(start, obs.limb_joint_positions, atol=1e-4)


def test_zero_torque_leaves_a_settled_scene_at_rest(env):
    """With no torque and no gravity, a settled scene barely drifts."""
    obs, _ = env.reset()
    start = np.array(obs.limb_joint_positions)
    for _ in range(20):
        obs, _, _, _, _ = env.step(np.zeros(NUM_ROBOT_JOINTS, dtype=np.float32))
    assert np.allclose(start, obs.limb_joint_positions, atol=1e-2)


def test_reset_restores_the_initial_configuration(env):
    """Reset returns the limb to the config's initial joint positions."""
    env.reset()
    action = np.full(NUM_ROBOT_JOINTS, env.action_space.high[0], dtype=np.float32)
    for _ in range(30):
        env.step(action)
    obs, _ = env.reset()
    assert np.allclose(obs.limb_joint_positions, LIMB_INIT, atol=1e-3)


def test_reset_with_init_state_restores_that_state(env):
    """An init_state passed through options is restored rather than the default."""
    obs, _ = env.reset()
    action = np.full(NUM_ROBOT_JOINTS, env.action_space.high[0], dtype=np.float32)
    for _ in range(30):
        moved, _, _, _, _ = env.step(action)
    restored, _ = env.reset(options={"init_state": moved})
    assert np.allclose(
        restored.limb_joint_positions, moved.limb_joint_positions, atol=1e-3
    )
    assert not np.allclose(restored.limb_joint_positions, obs.limb_joint_positions)


def test_goal_joint_positions_come_from_the_config(env):
    """The goal is observable and matches the scene config."""
    obs, _ = env.reset()
    assert np.allclose(obs.limb_goal_joint_positions, LIMB_GOAL)


def test_render_returns_an_rgb_image(env):
    """Rendering honours the configured image size."""
    env.reset()
    img = env.render()
    assert img.shape == (
        env.config.render_image_height,
        env.config.render_image_width,
        3,
    )
    assert img.dtype == np.uint8


def test_get_transition_matches_stepping():
    """A state plus an action must determine the next state, momentum included."""
    env = _MinimalLimb3DEnv(config=_make_config(), allow_state_access=True)
    try:
        env.reset()
        action = np.full(NUM_ROBOT_JOINTS, 0.5, dtype=np.float32)
        # Build up real joint velocities, so that dropping them would show.
        for _ in range(100):
            env.step(action)
        state = env.get_state()
        stepped, _, _, _, _ = env.step(action)
        predicted = env.get_next_state(state, action)
        assert np.allclose(
            predicted.limb_joint_positions, stepped.limb_joint_positions, atol=1e-3
        )
        assert np.allclose(
            predicted.robot_joint_positions, stepped.robot_joint_positions, atol=1e-3
        )
    finally:
        env.close()


def test_a_coarse_dt_applies_torque_for_the_whole_interval():
    """One coarse step matches several fine ones, since PyBullet clears torques."""
    fine = _MinimalLimb3DEnv(config=_make_config())
    coarse = _MinimalLimb3DEnv(config=_make_config(dt=4.0 / 240.0))
    try:
        action = np.full(NUM_ROBOT_JOINTS, 1.0, dtype=np.float32)
        fine.reset()
        coarse.reset()
        for _ in range(20):
            for _ in range(4):
                fine_obs, _, _, _, _ = fine.step(action)
            coarse_obs, _, _, _, _ = coarse.step(action)
        assert np.allclose(
            coarse_obs.robot_joint_positions, fine_obs.robot_joint_positions, atol=1e-3
        )
    finally:
        fine.close()
        coarse.close()


def test_dt_must_be_a_whole_number_of_timesteps():
    """A dt that is not a multiple of the timestep fails loudly rather than silently."""
    with pytest.raises(AssertionError, match="PYBULLET_TIMESTEP"):
        _make_config(dt=1.0 / 100.0)


def test_render_fps_follows_dt():
    """Videos play at real speed only if the frame rate matches the control rate."""
    assert _make_config().render_fps == 240
    assert _make_config(dt=1.0 / 60.0).render_fps == 60


def test_default_config_builds_a_scene():
    """The config is usable without being handed a scene."""
    assert Limb3DEnvConfig().scene.scene_type == "isolated"


def test_close_disconnects_the_physics_client():
    """Closing releases the PyBullet connection."""
    environment = _MinimalLimb3DEnv(config=_make_config())
    client_id = environment.physics_client_id
    environment.close()
    assert not p.isConnected(physicsClientId=client_id)


def test_close_is_idempotent():
    """Closing twice is safe, which matters for fixture teardown."""
    environment = _MinimalLimb3DEnv(config=_make_config())
    environment.close()
    environment.close()
