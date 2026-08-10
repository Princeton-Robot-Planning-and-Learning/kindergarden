"""Tests for the TidyBot3D PickPlace3D task."""

from pathlib import Path

import kinder
from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv

_TASK_CONFIG_PATH = (
    Path(kinder.__path__[0])
    / "envs"
    / "dynamic3d"
    / "tasks"
    / "PickPlace3D"
    / "PickPlace3D-o1.json"
)


def _make_env() -> ObjectCentricTidyBot3DEnv:
    return ObjectCentricTidyBot3DEnv(
        num_objects=1,
        task_config_path=str(_TASK_CONFIG_PATH),
        scene_bg=False,
        allow_state_access=True,
    )


def test_pickplace_reset_and_step():
    """Test that the environment resets to a valid observation and steps."""
    env = _make_env()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs), "Observation not in observation space"
    assert isinstance(info, dict)

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, _ = env.step(action)
    assert env.observation_space.contains(next_obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    env.close()


def test_pickplace_goal_detection():
    """Test that the goal is detected when the cube is placed in the goal region."""
    env = _make_env()
    env.reset(seed=0)

    # After reset, the cube starts in cube_init_region, not the goal region.
    assert (
        not env._check_goals()  # pylint: disable=protected-access
    ), "Goals should not be satisfied after reset"

    # Teleport the cube to the center of goal_region ([1.0, 0.4, 1.2, 0.6]).
    current_state = env._get_current_state()  # pylint: disable=protected-access
    cube = env._objects_dict["cube1"]  # pylint: disable=protected-access
    modified_state = current_state.copy()
    modified_state.set(cube.symbolic_object, "x", 1.1)
    modified_state.set(cube.symbolic_object, "y", 0.5)
    modified_state.set(cube.symbolic_object, "z", 0.02)
    env.set_state(modified_state)

    assert (
        env._check_goals()  # pylint: disable=protected-access
    ), "Goals should be satisfied with the cube in the goal region"
    env.close()
