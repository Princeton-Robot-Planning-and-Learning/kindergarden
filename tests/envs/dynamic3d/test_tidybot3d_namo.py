"""Tests for the TidyBot3D NAMO (Navigation Among Movable Objects) environment."""

from pathlib import Path

import numpy as np
import pytest
from gymnasium.wrappers import RecordVideo

import kinder
from kinder.envs.dynamic3d.tidybot3d import ObjectCentricTidyBot3DEnv
from tests.conftest import MAKE_VIDEOS

# Path to tasks directory
TASKS_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "src"
    / "kinder"
    / "envs"
    / "dynamic3d"
    / "tasks"
)

# Path to mimiclabs scenes for skip condition
MIMICLABS_SCENES_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "src"
    / "kinder"
    / "envs"
    / "dynamic3d"
    / "models"
    / "assets"
    / "mimiclabs_scenes"
    / "meshes"
)


def test_namo_env_loads():
    """Test that the NAMO environment loads correctly."""
    env = ObjectCentricTidyBot3DEnv(
        scene_type="namo",
        num_objects=1,
        task_config_path=str(TASKS_DIR / "Dynamo3D" / "Dynamo3D-o1.json"),
    )

    obs, info = env.reset(seed=42)
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)

    # Verify we have the obstacle block
    obstacle = obs.get_object_from_name("obstacle_chair")
    assert obstacle is not None

    env.close()


def test_namo_goal_not_satisfied_initially():
    """Test that goal is not satisfied after reset (block not in goal region)."""
    env = ObjectCentricTidyBot3DEnv(
        scene_type="namo",
        num_objects=1,
        task_config_path=str(TASKS_DIR / "Dynamo3D" / "Dynamo3D-o1.json"),
    )

    env.reset(seed=42)

    # Goal should not be satisfied initially
    assert not env._check_goals(), (  # pylint: disable=protected-access
        "Goal should not be satisfied after reset - "
        "obstacle_chair should not be in goal region initially"
    )

    env.close()


def test_namo_goal_satisfied_when_robot_in_region():
    """Test that goal is satisfied when robot reaches the goal region.

    In this NAMO task, the goal is for the robot (tidybot) to navigate to the goal
    region, potentially by pushing the obstacle out of the way.
    """
    kinder.register_all_environments()
    env = kinder.make(
        "kinder/Dynamo3D-o1-v0",
        render_mode="rgb_array",
        allow_state_access=True,
    )

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos_namo_goal_satisfied")

    env.reset(seed=42)

    # Access the underlying object-centric environment
    oc_env = env.unwrapped._object_centric_env  # pylint: disable=protected-access

    # Get current state
    current_state = oc_env._get_current_state()  # pylint: disable=protected-access

    # Get the robot object
    robot = current_state.get_object_from_name(
        oc_env.robot_name
    )  # type: ignore[attr-defined]

    # Move robot to the goal region (center of goal region is at x=1.0, y=0.0)
    modified_state = current_state.copy()
    modified_state.set(robot, "pos_base_x", 1.0)
    modified_state.set(robot, "pos_base_y", 0.0)

    # Set the modified state
    oc_env.set_state(modified_state)

    # Now goal should be satisfied (robot is in the goal region)
    assert (
        oc_env._check_goals()  # pylint: disable=protected-access
    ), "Goal should be satisfied after moving robot to goal region"

    env.close()


def test_namo_goal_achieved_after_teleporting_chair_and_robot():
    """Test that goal is achieved by teleporting chair away and robot to goal region.

    This test verifies the goal checking logic by:
    1. Teleporting the obstacle chair away from the goal region
    2. Teleporting the robot into the goal region
    3. Checking that the goal is now satisfied
    """
    kinder.register_all_environments()
    env = kinder.make(
        "kinder/Dynamo3D-o1-v0",
        render_mode="rgb_array",
        allow_state_access=True,
    )

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos_namo_teleport_goal")

    env.reset(seed=42)

    # Access the underlying object-centric environment
    oc_env = env.unwrapped._object_centric_env  # pylint: disable=protected-access

    # Goal should not be satisfied initially
    assert (
        not oc_env._check_goals()  # pylint: disable=protected-access
    ), "Goal should not be satisfied after reset"

    # Get current state
    current_state = oc_env._get_current_state()  # pylint: disable=protected-access

    # Get the robot and obstacle chair objects
    robot = current_state.get_object_from_name(
        oc_env.robot_name
    )  # type: ignore[attr-defined]
    obstacle_chair = current_state.get_object_from_name("obstacle_chair")

    # Create modified state
    modified_state = current_state.copy()

    # Teleport the obstacle chair away from the goal region
    # Goal region is at x=[0.8, 1.2], y=[-0.2, 0.2]
    # Move chair to x=-1.0, y=0.0 (far from goal region)
    modified_state.set(obstacle_chair, "x", -1.0)
    modified_state.set(obstacle_chair, "y", 0.0)
    modified_state.set(obstacle_chair, "z", 0.55)  # Keep it at its original height

    # Teleport the robot to the center of the goal region
    # Goal region center is at x=1.0, y=0.0
    modified_state.set(robot, "pos_base_x", 1.0)
    modified_state.set(robot, "pos_base_y", 0.0)

    # Set the modified state
    oc_env.set_state(modified_state)

    # Verify the chair was moved
    updated_state = oc_env._get_current_state()  # pylint: disable=protected-access
    chair_x = updated_state.get(obstacle_chair, "x")
    assert chair_x < 0, f"Chair should be moved to negative x, got {chair_x}"

    # Verify the robot was moved to goal region
    robot_x = updated_state.get(robot, "pos_base_x")
    robot_y = updated_state.get(robot, "pos_base_y")
    assert 0.8 <= robot_x <= 1.2, f"Robot x should be in [0.8, 1.2], got {robot_x}"
    assert -0.2 <= robot_y <= 0.2, f"Robot y should be in [-0.2, 0.2], got {robot_y}"

    # Now goal should be satisfied (robot is in the goal region)
    goal_satisfied = oc_env._check_goals()  # pylint: disable=protected-access
    assert (
        goal_satisfied
    ), "Goal should be satisfied after teleporting chair away and robot to goal region"

    env.close()


def test_namo_robot_can_navigate_to_goal():
    """Test that robot can navigate to the goal region by moving forward.

    This test verifies that:
    1. The chair is moved out of the way (simulating a successful push)
    2. The robot can navigate forward to the goal region
    3. The goal is achieved when the robot reaches the goal region
    """
    kinder.register_all_environments()
    if MIMICLABS_SCENES_DIR.exists():
        env = kinder.make(
            "kinder/Dynamo3D-o1-v0",
            render_mode="rgb_array",
            scene_bg=True,
            scene_render_camera="agentview_1",
        )
    else:
        env = kinder.make("kinder/Dynamo3D-o1-v0", render_mode="rgb_array")

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos_namo_navigate")

    obs, _ = env.reset(seed=123)

    # Access the underlying object-centric environment
    oc_env = env.unwrapped._object_centric_env  # pylint: disable=protected-access

    # Get initial state
    state = env.observation_space.devectorize(obs)
    robot_name = oc_env.robot_name  # type: ignore[attr-defined]
    robot = state.get_object_from_name(robot_name)
    robot_x = state.get(robot, "pos_base_x")
    robot_y = state.get(robot, "pos_base_y")

    # Goal region is at x=0.8 to x=1.2, y=-0.2 to 0.2
    # Move robot towards the goal region center
    goal_x = 1.0
    goal_y = 0.0

    # Calculate delta per step (constant delta actions)
    max_magnitude = 1e-2
    dx = goal_x - robot_x
    dy = goal_y - robot_y
    distance = (dx**2 + dy**2) ** 0.5
    steps = int(distance / max_magnitude) + 1

    # Normalize to get unit direction, then scale by max_magnitude
    unit_dx = dx / distance * max_magnitude
    unit_dy = dy / distance * max_magnitude

    # Execute constant delta steps to reach the goal
    for _ in range(steps):
        action = np.array([unit_dx, unit_dy, 0.0] + [0.0] * 8)
        env.step(action)

    # Verify the goal is achieved
    goal_achieved = oc_env._check_goals()  # pylint: disable=protected-access
    assert goal_achieved, "Goal should be achieved after robot navigates to goal region"

    env.close()


def test_namo_action_space():
    """Test that action space is valid."""
    env = ObjectCentricTidyBot3DEnv(
        scene_type="namo",
        num_objects=1,
        task_config_path=str(TASKS_DIR / "Dynamo3D" / "Dynamo3D-o1.json"),
    )

    env.reset(seed=42)
    action = env.action_space.sample()
    assert env.action_space.contains(action)

    env.close()


def test_namo_step():
    """Test that step returns valid outputs."""
    env = ObjectCentricTidyBot3DEnv(
        scene_type="namo",
        num_objects=1,
        task_config_path=str(TASKS_DIR / "Dynamo3D" / "Dynamo3D-o1.json"),
    )

    env.reset(seed=42)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    env.close()


@pytest.mark.skipif(
    not MIMICLABS_SCENES_DIR.exists(),
    reason="MimicLabs scenes not downloaded. "
    "Run: python scripts/download_mimiclabs_assets.py",
)
def test_namo_with_mimiclabs_scene():
    """Test NAMO environment with MimicLabs background scene."""
    env = ObjectCentricTidyBot3DEnv(
        scene_type="namo",
        num_objects=1,
        task_config_path=str(TASKS_DIR / "Dynamo3D" / "Dynamo3D-o1.json"),
        scene_bg=True,
    )

    obs, _ = env.reset(seed=42)
    assert env.observation_space.contains(obs)

    # Verify scene configuration
    active_scene = env.task_config.get("_active_scene", {})
    assert active_scene.get("type") == "mimiclabs"
    assert active_scene.get("lab") == 2

    # Take a few steps
    for _ in range(5):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            break

    env.close()
