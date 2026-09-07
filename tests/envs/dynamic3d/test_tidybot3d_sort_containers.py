"""Regression tests for moving SortClutteredBlocks3D receptacles."""

from pathlib import Path

import numpy as np
import pytest

import kinder
from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv


@pytest.mark.parametrize("count", [4, 20])
def test_sort_goals_follow_bins(count: int):
    """Correct placements follow translated bins; the former table patch fails."""
    path = (
        Path(kinder.__path__[0])
        / "envs/dynamic3d/tasks/SortClutteredBlocks3D"
        / f"SortClutteredBlocks3D-o{count}-sort_the_cluttered_blocks_into_bins.json"
    )
    env = ObjectCentricTidyBot3DEnv(
        num_objects=count,
        task_config_path=str(path),
        scene_bg=False,
        allow_state_access=True,
    )
    try:
        env.reset(seed=0)
        assert not env._check_goals()  # pylint: disable=protected-access
        state = env._get_current_state()  # pylint: disable=protected-access
        # Raise all bins clear of the table and translate them by 30 cm.
        for color in ["red", "green", "blue", "yellow"]:
            obj = state.get_object_from_name(f"bin_{color}")
            state.set(obj, "x", state.get(obj, "x") + 0.3)
            state.set(obj, "z", state.get(obj, "z") + 0.3)
            for key, value in zip(["qw", "qx", "qy", "qz"], [1, 0, 0, 0]):
                state.set(obj, key, value)
        for _, name, region in env.task_config["goal_state"]:
            cube = state.get_object_from_name(name)
            target = env.task_config["regions"][region]["target"]
            bin_obj = state.get_object_from_name(target)
            for axis in "xyz":
                state.set(
                    cube, axis, state.get(bin_obj, axis) + (0.025 if axis == "z" else 0)
                )
        env.set_state(state)
        assert env._check_goals()  # pylint: disable=protected-access
        for _, name, _ in env.task_config["goal_state"]:
            cube = state.get_object_from_name(name)
            state.set(cube, "x", state.get(cube, "x") - 0.3)
        env.set_state(state)
        assert not env._check_goals()  # pylint: disable=protected-access
        # Goal tolerance stays above the bin floor, below its rim, and within walls.
        for color in ["red", "green", "blue", "yellow"]:
            bin_geometry = env.get_object(f"bin_{color}")
            region = bin_geometry.region_objects[f"table_1_object_goal_{color}_region"][
                0
            ]
            assert region.site_element is not None
            size = np.fromstring(region.site_element.attrib["size"], sep=" ")
            pos = np.fromstring(region.site_element.attrib["pos"], sep=" ")
            assert np.all(size[:2] <= 0.04)
            assert pos[2] - size[2] > 0.01
            assert pos[2] + size[2] < 0.1
    finally:
        env.close()


def test_twenty_sorted_cubes_remain_successful_after_settling() -> None:
    """Five separated cubes fit in each bin and remain scored after settling."""
    path = (
        Path(kinder.__path__[0])
        / "envs/dynamic3d/tasks/SortClutteredBlocks3D"
        / "SortClutteredBlocks3D-o20-sort_the_cluttered_blocks_into_bins.json"
    )
    env = ObjectCentricTidyBot3DEnv(
        num_objects=20,
        task_config_path=str(path),
        scene_bg=False,
        allow_state_access=True,
    )
    try:
        env.reset(seed=0)
        state = env._get_current_state()  # pylint: disable=protected-access
        counts: dict[str, int] = {}
        for _, name, region in env.task_config["goal_state"]:
            target = env.task_config["regions"][region]["target"]
            obj = state.get_object_from_name(target)
            cube = state.get_object_from_name(name)
            index = counts.get(target, 0)
            counts[target] = index + 1
            for axis, offset in zip(
                "xyz", [(index % 3 - 1) * 0.022, (index // 3 - 0.5) * 0.022, 0.045]
            ):
                state.set(cube, axis, state.get(obj, axis) + offset)
            for axis in ["vx", "vy", "vz", "wx", "wy", "wz"]:
                state.set(cube, axis, 0)
        env.set_state(state)
        for _ in range(1000):
            env._robot_env.sim.step()  # pylint: disable=protected-access
        env._current_state = (
            env._get_object_centric_state()
        )  # pylint: disable=protected-access
        assert env._check_goals()  # pylint: disable=protected-access
    finally:
        env.close()
