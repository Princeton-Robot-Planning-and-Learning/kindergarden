"""Tests for limbrepositioning3d.py."""

import numpy as np
import pytest
from gymnasium.wrappers import RecordVideo
from relational_structs.spaces import ObjectCentricBoxSpace

import kinder
from kinder.envs.kinematic3d.limb_utils import NUM_LIMB_JOINTS, NUM_ROBOT_JOINTS
from kinder.envs.kinematic3d.limbrepositioning3d import (
    ALL_VARIANTS,
    LimbRepositioning3DEnv,
    ObjectCentricLimbRepositioning3DEnv,
    create_variant_config,
)
from kinder.envs.kinematic3d.limbs import get_sampling_bounds
from tests.conftest import MAKE_VIDEOS

# One variant per scene type, for the tests that step the simulation.
REPRESENTATIVE_VARIANTS = [
    "isolated-left-arm",
    "human-right-arm",
    "wheelchair-left-arm",
    "bed-right-leg",
]


@pytest.fixture(scope="module")
def env():
    """Create a shared environment for all tests in this module."""
    environment = LimbRepositioning3DEnv(
        variant="wheelchair-left-arm", render_mode="rgb_array", use_gui=False
    )
    if MAKE_VIDEOS:
        environment = RecordVideo(environment, "unit_test_videos")
    yield environment
    environment.close()


def test_limbrepositioning3d_env(env):  # pylint: disable=redefined-outer-name
    """Tests for basic methods in the limb repositioning environment."""
    obs, _ = env.reset(seed=123)
    assert isinstance(obs, np.ndarray)
    assert env.action_space.shape == (NUM_ROBOT_JOINTS,)

    for _ in range(10):
        act = env.action_space.sample()
        assert isinstance(act, np.ndarray)
        obs, reward, terminated, truncated, info = env.step(act)
        assert reward == -1.0
        assert not truncated
        assert "limb_distance_to_goal" in info
        if terminated:
            break

    img = env.render()
    assert img.shape == (512, 512, 3)
    assert img.dtype == np.uint8


def test_goal_is_not_reached_initially(env):  # pylint: disable=redefined-outer-name
    """The limb should start away from its goal, or the task would be trivial."""
    env.reset(seed=123)
    object_centric_env = (
        env.unwrapped._object_centric_env  # pylint: disable=protected-access
    )
    assert not object_centric_env.goal_reached()


def test_observation_contains_limb_and_goal(
    env,
):  # pylint: disable=redefined-outer-name
    """The limb's joint positions, velocities, and goal are all observable."""
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    vec_obs, _ = env.reset(seed=123)
    obs = env.observation_space.devectorize(vec_obs)

    limb = obs.get_object_from_name("limb")
    for i in range(1, NUM_LIMB_JOINTS + 1):
        for prefix in ("joint_", "joint_vel_", "goal_joint_"):
            assert obs.get(limb, f"{prefix}{i}") is not None

    config = create_variant_config("wheelchair-left-arm")
    goal = [obs.get(limb, f"goal_joint_{i}") for i in range(1, NUM_LIMB_JOINTS + 1)]
    assert np.allclose(goal, config.scene.limb_goal_joint_positions)


def test_torque_moves_the_passive_limb(env):  # pylint: disable=redefined-outer-name
    """The limb is passive, so it should only move because the robot moves it."""
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    vec_obs, _ = env.reset(seed=123)
    limb = env.observation_space.devectorize(vec_obs).get_object_from_name("limb")

    def limb_joints(vec):
        state = env.observation_space.devectorize(vec)
        return np.array(
            [state.get(limb, f"joint_{i}") for i in range(1, NUM_LIMB_JOINTS + 1)]
        )

    start = limb_joints(vec_obs)
    action = np.full(NUM_ROBOT_JOINTS, env.action_space.high[0], dtype=np.float32)
    for _ in range(50):
        vec_obs, _, terminated, _, _ = env.step(action)
        if terminated:
            break
    assert not np.allclose(start, limb_joints(vec_obs), atol=1e-4)


def test_reset_with_init_state(env):  # pylint: disable=redefined-outer-name
    """Reset() accepts an init_state via options and restores it."""
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    vec_obs, _ = env.reset(seed=123)
    initial_state = env.observation_space.devectorize(vec_obs)

    action = np.full(NUM_ROBOT_JOINTS, env.action_space.high[0], dtype=np.float32)
    for _ in range(20):
        env.step(action)

    restored_vec_obs, _ = env.reset(options={"init_state": initial_state})
    restored = env.observation_space.devectorize(restored_vec_obs)
    limb = restored.get_object_from_name("limb")
    for i in range(1, NUM_LIMB_JOINTS + 1):
        assert np.isclose(
            initial_state.get(limb, f"joint_{i}"),
            restored.get(limb, f"joint_{i}"),
            atol=1e-6,
        )


def test_reset_is_deterministic(env):  # pylint: disable=redefined-outer-name
    """Two resets with the same seed should give the same observation."""
    first, _ = env.reset(seed=123)
    action = np.full(NUM_ROBOT_JOINTS, env.action_space.high[0], dtype=np.float32)
    for _ in range(5):
        env.step(action)
    second, _ = env.reset(seed=123)
    assert np.allclose(first, second, atol=1e-6)


@pytest.mark.parametrize("variant", REPRESENTATIVE_VARIANTS)
def test_scene_types_build_and_step(variant):
    """Every kind of scene can be built, stepped, and rendered."""
    environment = ObjectCentricLimbRepositioning3DEnv(variant=variant)
    try:
        obs, _ = environment.reset(seed=0)
        assert len(obs.limb_joint_positions) == NUM_LIMB_JOINTS
        assert len(obs.robot_joint_positions) == NUM_ROBOT_JOINTS
        obs, _, _, _, _ = environment.step(np.zeros(NUM_ROBOT_JOINTS, dtype=np.float32))
        assert np.isfinite(obs.limb_distance_to_goal)
        assert environment.render().shape == (512, 512, 3)
    finally:
        environment.close()


def test_initial_state_varies_with_seed():
    """Resets with different seeds give different initial limb configurations."""
    environment = ObjectCentricLimbRepositioning3DEnv(variant="wheelchair-left-arm")
    try:
        configurations = []
        for seed in range(4):
            obs, _ = environment.reset(seed=seed)
            configurations.append(np.array(obs.limb_joint_positions))
        for i in range(1, len(configurations)):
            assert not np.allclose(configurations[0], configurations[i], atol=1e-3)
    finally:
        environment.close()


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_initial_state_within_joint_ranges(variant):
    """Sampled initial configurations stay within the limb's joint ranges."""
    environment = ObjectCentricLimbRepositioning3DEnv(variant=variant)
    try:
        nominal = np.array(environment.config.scene.limb_init_joint_positions)
        lower, upper = get_sampling_bounds(environment.config.scene.limb_name, nominal)
        for seed in range(5):
            obs, _ = environment.reset(seed=seed)
            joints = np.array(obs.limb_joint_positions)
            assert np.all(joints >= lower - 1e-6), f"{variant} below range at {seed}"
            assert np.all(joints <= upper + 1e-6), f"{variant} above range at {seed}"
    finally:
        environment.close()


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_robot_regrasps_sampled_initial_states(variant):
    """The weld stays tight for every sampled initial state."""
    environment = ObjectCentricLimbRepositioning3DEnv(variant=variant)
    try:
        for seed in range(3):
            environment.reset(seed=seed)
            error = float(
                np.linalg.norm(
                    np.subtract(
                        environment.robot.arm.get_end_effector_pose().position,
                        environment.limb.get_end_effector_pose().position,
                    )
                )
            )
            assert error < 0.01, f"{variant} seed {seed}: grasp offset {error:.3f} m"
    finally:
        environment.close()


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_robot_grasps_limb(variant):
    """The robot's end effector must reach the limb's grasp frame.

    Otherwise the weld between them is created with an offset, and the limb gets yanked
    as soon as the simulation starts.
    """
    environment = ObjectCentricLimbRepositioning3DEnv(variant=variant)
    try:
        robot_ee = environment.robot.arm.get_end_effector_pose().position
        limb_ee = environment.limb.get_end_effector_pose().position
        error = float(np.linalg.norm(np.subtract(robot_ee, limb_ee)))
        assert error < 0.01, f"{variant}: grasp offset of {error:.3f} m"
    finally:
        environment.close()


def test_registered_variants():
    """All sixteen variants are registered and can be created with make()."""
    kinder.register_all_environments()
    env_ids = kinder.get_all_env_ids()
    for variant in ALL_VARIANTS:
        assert f"kinder/LimbRepositioning3D-{variant}-v0" in env_ids
    assert "LimbRepositioning3D" in kinder.get_env_categories()["Kinematic3D"]

    environment = kinder.make(
        "kinder/LimbRepositioning3D-wheelchair-right-leg-v0", render_mode="rgb_array"
    )
    try:
        obs, _ = environment.reset(seed=0)
        assert isinstance(obs, np.ndarray)
    finally:
        environment.close()
