"""ScoopPour3D must transfer cubes to the advertised target bin."""

import json
from pathlib import Path

import pytest

import kinder
from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv

_TASKS = Path(kinder.__path__[0]) / "envs/dynamic3d/tasks/ScoopPour3D"


@pytest.mark.parametrize("count", [10, 100])
def test_scoop_target_config(count: int):
    """Both counts score every cube against the green bin."""
    config = json.loads((_TASKS / f"ScoopPour3D-o{count}.json").read_text())
    goals = config["goal_state"]
    assert len(goals) == count
    for _, _, region in goals:
        assert config["regions"][region]["target"] == "bin_green_0"


def test_scoop_target_follows_green_bin():
    """The target bin succeeds; the source bin and former target drawer fail."""
    env = ObjectCentricTidyBot3DEnv(
        num_objects=10,
        task_config_path=str(_TASKS / "ScoopPour3D-o10.json"),
        scene_bg=False,
        allow_state_access=True,
    )
    try:
        env.reset(seed=0)
        assert not env._check_goals()  # pylint: disable=protected-access
        state = env._get_current_state()  # pylint: disable=protected-access
        green = state.get_object_from_name("bin_green_0")
        state.set(green, "x", state.get(green, "x") + 0.3)
        for target, expected in [("bin_green_0", True), ("bin_yellow_0", False)]:
            obj = state.get_object_from_name(target)
            for i in range(10):
                cube = state.get_object_from_name(f"cube_{i}")
                state.set(cube, "x", state.get(obj, "x") + (i % 5 - 2) * 0.02)
                state.set(cube, "y", state.get(obj, "y") + (i // 5) * 0.02)
                state.set(cube, "z", state.get(obj, "z") + 0.025)
            env.set_state(state)
            assert env._check_goals() == expected  # pylint: disable=protected-access
        for i in range(10):
            cube = state.get_object_from_name(f"cube_{i}")
            for axis, value in zip("xyz", [0.5, 0, 0.3]):
                state.set(cube, axis, value)
        env.set_state(state)
        assert not env._check_goals()  # pylint: disable=protected-access
    finally:
        env.close()


def test_scoop_target_remains_successful_after_settling() -> None:
    """Ten separated cubes rest inside the target bin and satisfy its goal."""
    env = ObjectCentricTidyBot3DEnv(
        num_objects=10,
        task_config_path=str(_TASKS / "ScoopPour3D-o10.json"),
        scene_bg=False,
        allow_state_access=True,
    )
    try:
        env.reset(seed=0)
        state = env._get_current_state()  # pylint: disable=protected-access
        target = state.get_object_from_name("bin_green_0")
        for i in range(10):
            cube = state.get_object_from_name(f"cube_{i}")
            for axis, offset in zip(
                "xyz", [(i % 5 - 2) * 0.02, (i // 5 - 0.5) * 0.02, 0.04]
            ):
                state.set(cube, axis, state.get(target, axis) + offset)
            for axis in ["vx", "vy", "vz", "wx", "wy", "wz"]:
                state.set(cube, axis, 0)
        env.set_state(state)
        for _ in range(1000):
            env._robot_env.sim.step()  # pylint: disable=protected-access
        # Refresh the task state after advancing physics directly for settling.
        # pylint: disable=protected-access
        env._current_state = env._get_object_centric_state()
        # pylint: enable=protected-access
        assert env._check_goals()  # pylint: disable=protected-access
    finally:
        env.close()
