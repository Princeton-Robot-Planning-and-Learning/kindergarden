"""Reward functions for TidyBot tasks."""

from typing import Any, ClassVar

from relational_structs import ObjectCentricState

from kinder.envs.dynamic3d.object_types import MujocoTidyBotRobotObjectType


class TidyBotRewardCalculator:
    """Base class for TidyBot task rewards."""

    def __init__(self, scene_type: str, num_objects: int):
        self.scene_type = scene_type
        self.num_objects = num_objects
        self.episode_step = 0

    def calculate_reward(self, obs: dict[str, Any]) -> float:
        """Calculate reward based on current observation."""
        self.episode_step += 1
        base_reward = -1.0  # Small negative reward per timestep

        # Add task-specific rewards
        task_reward = self._calculate_task_reward(obs)

        return base_reward + task_reward

    def _calculate_task_reward(self, obs: dict[str, Any]) -> float:
        """Calculate task-specific reward.

        Override in subclasses.
        """
        _ = obs  # Unused in base class, overridden in subclasses
        return 0

    def is_terminated(self, obs: dict[str, Any]) -> bool:
        """Check if episode should terminate."""
        return self._is_task_completed(obs)

    def _is_task_completed(self, obs: dict[str, Any]) -> bool:
        """Check if task is completed.

        Override in subclasses.
        """
        _ = obs  # Unused in base class, overridden in subclasses
        return False


class BaseMotionRewardCalculator(TidyBotRewardCalculator):
    """Terminate when the robot is close enough to the target."""

    dist_thresh: ClassVar[float] = 8 * 1e-2

    def __init__(self) -> None:
        super().__init__(scene_type="base_motion", num_objects=1)

    def _is_task_completed(self, obs: dict[str, Any]) -> bool:
        state = obs["object_centric_state"]
        assert isinstance(state, ObjectCentricState)
        target = state.get_object_from_name("cube1")
        # Find the robot by type (name varies by config)
        robots = state.get_objects(MujocoTidyBotRobotObjectType)
        assert len(robots) == 1, f"Expected 1 robot, found {len(robots)}"
        robot = robots[0]
        target_x = state.get(target, "x")
        target_y = state.get(target, "y")
        robot_x = state.get(robot, "pos_base_x")
        robot_y = state.get(robot, "pos_base_y")
        dx = target_x - robot_x
        dy = target_y - robot_y
        distance = (dx**2 + dy**2) ** 0.5
        return distance <= self.dist_thresh


def create_reward_calculator(
    scene_type: str, num_objects: int
) -> TidyBotRewardCalculator:
    """Factory function to create appropriate reward calculator."""
    if scene_type == "base_motion":
        return BaseMotionRewardCalculator()
    return TidyBotRewardCalculator(scene_type, num_objects)
