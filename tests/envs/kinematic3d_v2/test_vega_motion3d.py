"""Tests for vega_motion3d.py."""

import numpy as np
import pytest
from gymnasium.wrappers import RecordVideo
from relational_structs.spaces import ObjectCentricBoxSpace

from tests.conftest import MAKE_VIDEOS

pytest.importorskip("prpl_kinematics")

# pylint: disable=wrong-import-position
from kinder.envs.kinematic3d_v2.object_types import (  # noqa: E402
    ARM_NUM_JOINTS,
)
from kinder.envs.kinematic3d_v2.vega_motion3d import (  # noqa: E402
    ObjectCentricVegaMotion3DEnv,
    VegaMotion3DEnv,
    VegaMotion3DObjectCentricState,
)


@pytest.fixture(scope="module")
def env():
    """Create a shared environment for all tests in this module."""
    environment = VegaMotion3DEnv(render_mode="rgb_array", use_gui=False)
    if MAKE_VIDEOS:
        environment = RecordVideo(environment, "unit_test_videos")
    yield environment
    environment.close()


@pytest.fixture(scope="module")
def object_centric_env():
    """Create a shared object-centric environment for all tests in this module."""
    environment = ObjectCentricVegaMotion3DEnv(
        render_mode="rgb_array", allow_state_access=True
    )
    yield environment
    environment.close()


def test_vega_motion3d_env(env):  # pylint: disable=redefined-outer-name
    """Tests for basic methods in the Vega motion3D env."""
    obs, _ = env.reset(seed=123)
    assert isinstance(obs, np.ndarray)
    assert env.action_space.shape == (ARM_NUM_JOINTS,)

    for _ in range(10):
        act = env.action_space.sample()
        assert isinstance(act, np.ndarray)
        obs, reward, _, _, _ = env.step(act)
        assert reward == -1.0

    img = env.render()
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.dtype == np.uint8


def test_vega_motion3d_reset_is_deterministic(
    env,
):  # pylint: disable=redefined-outer-name
    """The same seed should produce the same initial observation."""
    obs1, _ = env.reset(seed=123)
    obs2, _ = env.reset(seed=123)
    assert np.allclose(obs1, obs2)
    obs3, _ = env.reset(seed=456)
    assert not np.allclose(obs1, obs3)


def test_vega_motion3d_observation_space(
    env,
):  # pylint: disable=redefined-outer-name
    """The vectorized observation should round-trip through the box space."""
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    vec_obs, _ = env.reset(seed=123)
    obs = env.observation_space.devectorize(vec_obs)
    assert isinstance(obs, VegaMotion3DObjectCentricState)
    assert len(obs.arm_joint_positions) == ARM_NUM_JOINTS
    assert np.allclose(env.observation_space.vectorize(obs), vec_obs)


def test_vega_motion3d_target_is_reachable(
    object_centric_env,
):  # pylint: disable=redefined-outer-name
    """Every sampled target should carry a witness configuration reaching it.

    Targets are placed at the end-effector position of a sampled arm configuration, so
    reachability holds by construction rather than by an IK query against a fixed end-
    effector orientation, which over-constrained the reach (issue #150). The witness is
    sampled near home, so no target demands a wild swing of the arm.
    """
    # pylint: disable=protected-access
    for seed in range(3):
        obs, _ = object_centric_env.reset(seed=seed)
        assert not object_centric_env.goal_reached()
        witness = object_centric_env._target_witness_joints
        assert witness is not None
        home_joints = object_centric_env._home_arm_joint_positions
        window = object_centric_env.config.target_witness_joint_delta
        assert np.max(np.abs(witness - home_joints)) <= window
        config = {
            **object_centric_env.configuration,
            **object_centric_env._arm_space.to_configuration(witness),
        }
        position = object_centric_env.tree.forward_kinematics(
            object_centric_env._manipulator.ee_frame, config
        ).t
        assert np.allclose(position, obs.target_position, atol=1e-9)
        assert not object_centric_env._collision_checker.in_collision(config)


def test_vega_motion3d_goal_reached_by_witness(
    object_centric_env,
):  # pylint: disable=protected-access,redefined-outer-name
    """Driving the arm to the target's witness configuration should reach the goal."""
    # pylint: disable=protected-access
    object_centric_env.reset(seed=0)
    assert not object_centric_env.goal_reached()

    # Step toward the witness with actions the environment would accept, rather than
    # teleporting, so that action clipping and collision reverting are exercised too.
    goal_joints = object_centric_env._target_witness_joints
    max_mag = object_centric_env.config.max_action_mag
    terminated = False
    for _ in range(500):
        delta = np.clip(
            goal_joints - object_centric_env.arm_joint_positions, -max_mag, max_mag
        )
        _, _, terminated, _, _ = object_centric_env.step(delta)
        if terminated:
            break
    assert terminated
    assert object_centric_env.goal_reached()


def test_vega_motion3d_state_access(
    object_centric_env,
):  # pylint: disable=redefined-outer-name
    """Getting and setting states should round-trip."""
    obs, _ = object_centric_env.reset(seed=123)
    state = object_centric_env.get_state()
    action = np.array([0.05] * ARM_NUM_JOINTS)
    next_obs, _, _, _, _ = object_centric_env.step(action)
    assert not np.allclose(next_obs.arm_joint_positions, obs.arm_joint_positions)
    assert np.allclose(next_obs.target_position, obs.target_position)

    object_centric_env.set_state(state)
    restored = object_centric_env.get_state()
    assert np.allclose(restored.arm_joint_positions, obs.arm_joint_positions)
    assert np.allclose(restored.target_position, obs.target_position)


def test_vega_motion3d_registered():
    """The environment should be registered under the Kinematic3Dv2 category."""
    import kinder  # pylint: disable=import-outside-toplevel

    kinder.register_all_environments()
    assert "kinder/VegaMotion3D-v0" in kinder.get_all_env_ids()
    assert "VegaMotion3D" in kinder.get_env_categories()["Kinematic3Dv2"]
