"""The public tossing action space includes the existing velocity controls."""

import gymnasium as gym
import numpy as np

import kinder
from kinder.envs.dynamic3d.robots.tidybot_robot_env import TidyBot3DRobotActionSpace
from kinder.envs.dynamic3d.task_families import Tossing3DEnv


def test_tossing_velocity_action_space() -> None:
    """Velocity targets are accepted while position/gripper bounds still apply."""
    space = TidyBot3DRobotActionSpace(use_arm_velocities=True)
    action = np.zeros(18, dtype=space.dtype)
    action[11:] = np.arange(7) * 10
    assert space.contains(action)
    action[3] = 0.101
    assert not space.contains(action)
    action[3] = 0
    action[10] = 1.01
    assert not space.contains(action)
    assert not space.contains(np.zeros(11, dtype=space.dtype))
    assert not space.contains(np.zeros((100, 18), dtype=space.dtype))
    assert TidyBot3DRobotActionSpace().shape == (11,)


def test_registered_and_direct_tossing_have_velocity_controls() -> None:
    """Both public constructors declare actions that can be executed directly."""
    kinder.register_all_environments()
    for env in (gym.make("kinder/Tossing3D-o1-v0"), Tossing3DEnv(num_objects=1)):
        try:
            env.reset(seed=0)
            assert env.action_space.shape == (18,)
            action = np.zeros(18, dtype=env.action_space.dtype)
            action[11] = 0.1
            assert env.action_space.contains(action)
            env.step(action)
        finally:
            env.close()
    config = gym.spec("kinder/Shelf3D-o1-v0").kwargs.get("config")
    assert config is None or not config.use_arm_velocities
