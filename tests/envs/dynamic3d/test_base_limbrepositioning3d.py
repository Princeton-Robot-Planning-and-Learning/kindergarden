"""Tests for base_limbrepositioning3d.py.

The base class is abstract, so these exercise it through a minimal concrete subclass
rather than through the full LimbRepositioning3D environment.
"""

import numpy as np
import pybullet as p
import pytest
from pybullet_helpers.geometry import Pose, SE2Pose
from relational_structs import Object
from relational_structs.spaces import ObjectCentricStateSpace
from relational_structs.utils import create_state_from_dict

from kinder.envs.dynamic3d.base_limbrepositioning3d import (
    Limb3DEnvConfig,
    ObjectCentricLimb3DRobotEnv,
)
from kinder.envs.dynamic3d.limb_object_types import (
    Limb3DEnvTypeFeatures,
    Limb3DLimbType,
    Limb3DRobotType,
)
from kinder.envs.dynamic3d.limb_scenes import LimbRepositioningSceneConfig
from kinder.envs.dynamic3d.limb_utils import (
    NUM_LIMB_JOINTS,
    NUM_ROBOT_JOINTS,
    LimbRepositioning3DObjectCentricState,
    LimbRepositioning3DRobotActionSpace,
)

# An isolated right arm, which needs no furniture and is reachable from this base pose.
LIMB_BASE_POSE = Pose.from_rpy((0.730, -0.396, 0.270), (0.0, 0.0, np.pi))
LIMB_INIT = (-0.403357, -0.447072, -0.092894, -0.501061, 0.0, 0.0)
LIMB_GOAL = (-0.7, 0.0, -0.03, -0.8, 0.0, 0.0)
ROBOT_BASE_POSE = SE2Pose(0.3566, 0.4849, -0.6971)
ROBOT_BASE_Z = -0.329


class _MinimalLimb3DEnv(
    ObjectCentricLimb3DRobotEnv[LimbRepositioning3DObjectCentricState, Limb3DEnvConfig]
):
    """The smallest concrete environment the base class admits."""

    def _get_obs(self) -> LimbRepositioning3DObjectCentricState:
        base_pose = self.robot.get_base()
        robot_feats = {
            "pos_base_x": base_pose.x,
            "pos_base_y": base_pose.y,
            "pos_base_rot": base_pose.rot,
        }
        for i, value in enumerate(self._get_robot_joint_positions()):
            robot_feats[f"joint_{i + 1}"] = value
        for i, value in enumerate(self._get_robot_joint_velocities()):
            robot_feats[f"joint_vel_{i + 1}"] = value
        limb_feats = {}
        for i, value in enumerate(self.limb.get_joint_positions()):
            limb_feats[f"joint_{i + 1}"] = value
        for i, value in enumerate(self.limb.get_joint_velocities()):
            limb_feats[f"joint_vel_{i + 1}"] = value
        for i, value in enumerate(self.config.scene.limb_goal_joint_positions):
            limb_feats[f"goal_joint_{i + 1}"] = value
        state = create_state_from_dict(
            {
                Object("robot", Limb3DRobotType): robot_feats,
                Object("limb", Limb3DLimbType): limb_feats,
            },
            Limb3DEnvTypeFeatures,
            state_cls=LimbRepositioning3DObjectCentricState,
        )
        assert isinstance(state, LimbRepositioning3DObjectCentricState)
        return state

    def _create_constant_initial_state(self):
        return self._get_obs()

    def goal_reached(self) -> bool:
        return False

    def _reset_bodies(self) -> None:
        self.robot.arm.set_joints(
            self._extend_joints_with_fingers(list(self._initial_robot_joints))
        )
        self.limb.set_joints(list(self.config.scene.limb_init_joint_positions))
        self._regrasp_limb()

    def _set_state(self, state) -> None:
        self.robot.arm.set_joints(
            self._extend_joints_with_fingers(list(state.robot_joint_positions))
        )
        self.limb.set_joints(list(state.limb_joint_positions))
        self._reweld_limb()

    def step(self, action):
        self._apply_torques(list(action), [0.0] * NUM_LIMB_JOINTS)
        return self._get_obs(), -1.0, self.goal_reached(), False, {}


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


def test_isolated_scene_has_no_penetration(env):
    """An empty world gives the limb nothing to intrude into."""
    env.reset()
    assert env.limb_penetration() == pytest.approx(0.0)
    assert np.isinf(env.get_collision_clearance())


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
