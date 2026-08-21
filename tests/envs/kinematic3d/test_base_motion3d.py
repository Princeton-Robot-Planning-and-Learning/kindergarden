"""Tests for base_motion3d.py."""

from unittest.mock import patch

import numpy as np
import pytest
from gymnasium.wrappers import RecordVideo
from pybullet_helpers.motion_planning import (
    remap_se2_pose_plan_to_constant_distance,
    run_single_arm_mobile_base_motion_planning,
)
from relational_structs.spaces import ObjectCentricBoxSpace

import kinder
from kinder.envs.kinematic3d.base_motion3d import (
    BaseMotion3DEnv,
    BaseMotion3DEnvConfig,
    BaseMotion3DObjectCentricState,
    ObjectCentricBaseMotion3DEnv,
)
from tests.conftest import MAKE_VIDEOS


@pytest.fixture(scope="module")
def env():
    """Create a shared environment for all tests in this module."""
    environment = BaseMotion3DEnv(
        render_mode="rgb_array", use_gui=False, realistic_bg=False
    )
    if MAKE_VIDEOS:
        environment = RecordVideo(environment, "unit_test_videos")
    yield environment
    environment.close()


def test_base_motion3d_env(env):  # pylint: disable=redefined-outer-name
    """Tests for basic methods in base motion3D env."""
    obs, _ = env.reset(seed=123)
    assert isinstance(obs, np.ndarray)

    for _ in range(10):
        act = env.action_space.sample()
        assert isinstance(act, np.ndarray)
        obs, _, _, _, _ = env.step(act)


def test_standard_action_magnitude_and_base_action_clipping():
    """Base and arm deltas use the environment's configured action magnitude."""
    config = BaseMotion3DEnvConfig(max_action_mag=0.2)
    environment = BaseMotion3DEnv(config=config, use_gui=False, realistic_bg=False)
    try:
        vec_obs, _ = environment.reset(seed=123)
        assert np.all(environment.action_space.low[:10] == -0.2)
        assert np.all(environment.action_space.high[:10] == 0.2)

        initial_obs = environment.observation_space.devectorize(vec_obs)
        initial_state = BaseMotion3DObjectCentricState(
            initial_obs.data, initial_obs.type_features
        )
        out_of_range_action = np.array(
            [1.0, 0.0, 0.0] + [0.0] * 7 + [0.0], dtype=np.float32
        )
        next_vec_obs, _, _, _, _ = environment.step(out_of_range_action)
        next_obs = environment.observation_space.devectorize(next_vec_obs)
        next_state = BaseMotion3DObjectCentricState(
            next_obs.data, next_obs.type_features
        )

        assert np.isclose(next_state.base_pose.x - initial_state.base_pose.x, 0.2)
    finally:
        environment.close()

    # Uncomment to debug.
    # import pybullet as p
    # while True:
    #     p.getMouseEvents(env.unwrapped._object_centric_env.physics_client_id)


def test_motion_planning_in_base_motion3d_env(
    env,
):  # pylint: disable=redefined-outer-name
    """Proof of concept that motion planning works in this environment."""
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    config = (
        env.unwrapped._object_centric_env.config  # pylint: disable=protected-access
    )

    vec_obs, _ = env.reset(seed=123)
    # NOTE: we should soon make this smoother.
    oc_obs = env.observation_space.devectorize(vec_obs)
    obs = BaseMotion3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Create a simulator for planning.
    sim = ObjectCentricBaseMotion3DEnv(config=config, realistic_bg=False)

    base_plan = run_single_arm_mobile_base_motion_planning(
        sim.robot,
        sim.robot.base.get_pose(),
        obs.target_base_pose,
        collision_bodies=set(),
        seed=123,
    )
    assert base_plan is not None
    base_plan = remap_se2_pose_plan_to_constant_distance(
        base_plan, max_distance=config.max_action_mag
    )

    env.action_space.seed(123)
    for target_base_pose in base_plan[1:]:
        current_base_pose = obs.base_pose
        delta = target_base_pose - current_base_pose
        delta_lst = [delta.x, delta.y, delta.rot]
        action_lst = delta_lst + [0.0] * 7 + [0.0]
        action = np.array(action_lst, dtype=np.float32)
        vec_obs, _, done, _, _ = env.step(action)
        # NOTE: we should soon make this smoother.
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = BaseMotion3DObjectCentricState(oc_obs.data, oc_obs.type_features)
        if done:
            break
    else:
        assert False, "Plan did not reach goal"


def test_check_mobile_base_collisions_is_called(
    env,
):  # pylint: disable=redefined-outer-name
    """Test that check_mobile_base_collisions is called when there is a collision."""
    env.reset(seed=123)

    # Patch the check_mobile_base_collisions function
    with patch(
        "kinder.envs.kinematic3d.base_env.check_mobile_base_collisions"
    ) as mock_check_base:
        # Set return value to False (no collision)
        mock_check_base.return_value = False

        # Take an action that moves the base (first 3 elements are base actions)
        action = np.array([0.01, 0.01, 0.0] + [0.0] * 7 + [0.0], dtype=np.float32)
        env.step(action)

        # Verify that check_mobile_base_collisions was called
        assert mock_check_base.called, "check_mobile_base_collisions should be called"

        # Verify it was called with the correct arguments
        assert mock_check_base.call_count >= 1
        call_args = mock_check_base.call_args
        assert call_args is not None

        # Verify the robot base was passed as the first argument
        assert (
            call_args[0][0]
            == env.unwrapped._object_centric_env.robot.base  # pylint: disable=protected-access
        )


def test_base_motion3d_stays_in_bounds_with_many_small_base_actions():
    """BaseMotion3D should not drift outside its intended bounds."""
    environment = BaseMotion3DEnv(
        render_mode="rgb_array", use_gui=False, realistic_bg=True
    )
    if MAKE_VIDEOS:
        environment = RecordVideo(environment, "unit_test_videos")
    try:
        vec_obs, _ = environment.reset(seed=123)
        oc_obs = environment.observation_space.devectorize(vec_obs)
        obs = BaseMotion3DObjectCentricState(oc_obs.data, oc_obs.type_features)
        start_y = obs.base_pose.y
        config = (
            environment.unwrapped._object_centric_env.config  # pylint: disable=protected-access
        )

        desired_distance = 100.0
        step_distance = 0.1
        num_steps = int(desired_distance / step_distance)
        action = np.array(
            [0.0, -step_distance, 0.0] + [0.0] * 7 + [0.0], dtype=np.float32
        )

        terminated = False
        for _ in range(num_steps):
            vec_obs, _, terminated, _, _ = environment.step(action)
            assert not terminated

        oc_obs = environment.observation_space.devectorize(vec_obs)
        obs = BaseMotion3DObjectCentricState(oc_obs.data, oc_obs.type_features)

        assert obs.base_pose.y >= config.robot_base_pose_lower_bound.y
        assert obs.base_pose.y > start_y - desired_distance + step_distance
        assert not terminated
    finally:
        environment.close()


def test_reset_with_init_state():
    """Test that reset accepts init_state via options and restores state correctly."""
    kinder.register_all_environments()
    env = kinder.make(  # pylint: disable=redefined-outer-name
        "kinder/BaseMotion3D-v0", render_mode="rgb_array", realistic_bg=False
    )

    # Get initial observation after reset.
    vec_obs, _ = env.reset(seed=123)
    initial_oc_obs = env.observation_space.devectorize(vec_obs)
    initial_state = BaseMotion3DObjectCentricState(
        initial_oc_obs.data, initial_oc_obs.type_features
    )

    # Take some actions to change state.
    for _ in range(5):
        action = np.array([0.05, 0.02, 0.01] + [0.0] * 7 + [0.0], dtype=np.float32)
        env.step(action)

    # Verify state has changed.
    changed_vec_obs = (
        env.unwrapped._object_centric_env._get_obs()  # pylint: disable=protected-access
    )
    changed_state = BaseMotion3DObjectCentricState(
        changed_vec_obs.data, changed_vec_obs.type_features
    )
    assert not np.allclose(
        initial_state.base_pose.x, changed_state.base_pose.x, atol=1e-3
    ), "State should have changed after taking actions"

    # Reset with init_state to restore the original state.
    restored_vec_obs, _ = env.reset(options={"init_state": initial_state})
    restored_oc_obs = env.observation_space.devectorize(restored_vec_obs)
    restored_state = BaseMotion3DObjectCentricState(
        restored_oc_obs.data, restored_oc_obs.type_features
    )

    # Verify the state matches the initial state.
    assert np.allclose(
        initial_state.base_pose.x, restored_state.base_pose.x, atol=1e-6
    ), "Base pose x should match after reset with init_state"
    assert np.allclose(
        initial_state.base_pose.y, restored_state.base_pose.y, atol=1e-6
    ), "Base pose y should match after reset with init_state"
    assert np.allclose(
        initial_state.target_base_pose.x,
        restored_state.target_base_pose.x,
        atol=1e-6,
    ), "Target pose x should match after reset with init_state"
    assert np.allclose(
        initial_state.target_base_pose.y,
        restored_state.target_base_pose.y,
        atol=1e-6,
    ), "Target pose y should match after reset with init_state"

    env.close()
