"""Tests for the TidyBot3D base motion environment."""

from pathlib import Path

import numpy as np
import pytest
from gymnasium.wrappers import RecordVideo
from relational_structs.spaces import ObjectCentricBoxSpace

import kinder
from tests.conftest import MAKE_VIDEOS

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


def test_straight_base_motion():
    """This environment is really simple: moving directly towards the target works."""

    kinder.register_all_environments()
    env = kinder.make("kinder/TidyBot3D-base_motion-o1-v0", render_mode="rgb_array")

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    # Extract the positions of the target and robot.
    obs, _ = env.reset(seed=123)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)
    robot_name = env.unwrapped._object_centric_env.robot_name  # type: ignore[attr-defined] # pylint: disable=protected-access
    target = state.get_object_from_name("cube1")
    robot = state.get_object_from_name(robot_name)
    target_x = state.get(target, "x")
    target_y = state.get(target, "y")
    robot_x = state.get(robot, "pos_base_x")
    robot_y = state.get(robot, "pos_base_y")
    robot_rot = state.get(robot, "pos_base_rot")

    # Actions are delta positions.
    max_magnitude = 1e-2
    dx = target_x - robot_x
    dy = target_y - robot_y
    distance = (dx**2 + dy**2) ** 0.5
    steps = int(distance / max_magnitude) + 1
    plan = []
    for i in range(1, steps + 1):
        frac = i / steps
        plan.append(np.array([frac * dx, frac * dy, robot_rot] + [0.0] * 8))

    # Execute the plan.
    for action in plan:
        _, _, done, _, _ = env.step(action)
        if done:  # success
            break
    else:
        assert False, "Failed to reach target"

    env.close()


@pytest.mark.skipif(
    not MIMICLABS_SCENES_DIR.exists(),
    reason="MimicLabs scenes not downloaded. "
    "Run: python scripts/download_mimiclabs_assets.py",
)
@pytest.mark.parametrize(
    "view", ["frontview", "agentview_1", "agentview_2", "robot_base", "robot_wrist"]
)
def test_straight_base_motion_mimiclabs(view):
    """Test base motion with MimicLabs background scene (uses lab5 for base_motion)."""

    kinder.register_all_environments()
    env = kinder.make(
        "kinder/TidyBot3D-base_motion-o1-v0",
        render_mode="rgb_array",
        scene_bg=True,  # Use default mimiclabs scene (lab5 for base_motion)
        scene_render_camera=f"{view}",
    )

    if MAKE_VIDEOS:
        env = RecordVideo(env, f"unit_test_videos_base_motion_view_{view}")

    # Extract the positions of the target and robot.
    obs, _ = env.reset(seed=123)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)
    robot_name = env.unwrapped._object_centric_env.robot_name  # type: ignore[attr-defined] # pylint: disable=protected-access
    target = state.get_object_from_name("cube1")
    robot = state.get_object_from_name(robot_name)
    target_x = state.get(target, "x")
    target_y = state.get(target, "y")
    robot_x = state.get(robot, "pos_base_x")
    robot_y = state.get(robot, "pos_base_y")
    robot_rot = state.get(robot, "pos_base_rot")

    # Actions are delta positions.
    max_magnitude = 1e-2
    dx = target_x - robot_x
    dy = target_y - robot_y
    distance = (dx**2 + dy**2) ** 0.5
    steps = int(distance / max_magnitude) + 1
    plan = []
    for i in range(1, steps + 1):
        frac = i / steps
        plan.append(np.array([frac * dx, frac * dy, robot_rot] + [0.0] * 8))

    # Execute the plan.
    for action in plan:
        _, _, done, _, _ = env.step(action)
        if done:  # success
            break
    else:
        assert False, "Failed to reach target with mimiclabs background"

    env.close()
