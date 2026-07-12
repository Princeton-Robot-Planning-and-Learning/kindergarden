"""Tests for the Franka FR3 FrankaPickPlace3D task."""

from pathlib import Path

import kinder
from kinder.envs.dynamic3d.envs import ObjectCentricFranka3DEnv

_TASK_CONFIG_PATH = (
    Path(kinder.__path__[0])
    / "envs"
    / "dynamic3d"
    / "tasks"
    / "FrankaPickPlace3D"
    / "FrankaPickPlace3D-o1.json"
)


def _make_env() -> ObjectCentricFranka3DEnv:
    return ObjectCentricFranka3DEnv(
        num_objects=1,
        task_config_path=str(_TASK_CONFIG_PATH),
        scene_bg=False,
        allow_state_access=True,
    )


def test_franka_pickplace_reset_and_step():
    """Test that the environment resets to a valid observation and steps."""
    env = _make_env()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs), "Observation not in observation space"
    assert isinstance(info, dict)
    assert env.action_space.shape == (8,)

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, _ = env.step(action)
    assert env.observation_space.contains(next_obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    env.close()


def test_franka_pickplace_robot_state():
    """Test that the robot appears in the state with its mount pose."""
    env = _make_env()
    obs, _ = env.reset(seed=0)
    robot_obj = obs.get_object_from_name("robot")

    # The mount is sampled from robot_mount_region ([0.13, -0.02, 0.17, 0.02]).
    assert 0.13 <= obs.get(robot_obj, "pos_base_x") <= 0.17
    assert -0.02 <= obs.get(robot_obj, "pos_base_y") <= 0.02
    env.close()


def test_franka_pickplace_goal_detection():
    """Test that the goal is detected when the cube is placed in the goal region."""
    env = _make_env()
    env.reset(seed=0)

    # After reset, the cube starts in cube_init_region, not the goal region.
    assert (
        not env._check_goals()  # pylint: disable=protected-access
    ), "Goals should not be satisfied after reset"

    # Teleport the cube to the center of goal_region. The desk is at
    # world (0.5, 0.0); the region is desk-local [[-0.05, 0.12, 0.15, 0.27]],
    # so the center is world (0.55, 0.195) on the desk surface (z=0.75).
    current_state = env._get_current_state()  # pylint: disable=protected-access
    cube = env._objects_dict["cube1"]  # pylint: disable=protected-access
    modified_state = current_state.copy()
    modified_state.set(cube.symbolic_object, "x", 0.55)
    modified_state.set(cube.symbolic_object, "y", 0.195)
    modified_state.set(cube.symbolic_object, "z", 0.79)
    env.set_state(modified_state)

    assert (
        env._check_goals()  # pylint: disable=protected-access
    ), "Goals should be satisfied with the cube in the goal region"
    env.close()
