"""Robot classes for dynamic3d environments."""

from kinder.envs.dynamic3d.robots.base import RobotEnv
from kinder.envs.dynamic3d.robots.fr3_robot_env import (
    FR3RobotActionSpace,
    FR3RobotEnv,
)
from kinder.envs.dynamic3d.robots.rby1a_robot_env import (
    RBY1ARobotActionSpace,
    RBY1ARobotEnv,
)
from kinder.envs.dynamic3d.robots.tidybot_robot_env import (
    TidyBot3DRobotActionSpace,
    TidyBotRobotEnv,
)

__all__ = [
    "FR3RobotActionSpace",
    "FR3RobotEnv",
    "RobotEnv",
    "RBY1ARobotActionSpace",
    "RBY1ARobotEnv",
    "TidyBot3DRobotActionSpace",
    "TidyBotRobotEnv",
]
