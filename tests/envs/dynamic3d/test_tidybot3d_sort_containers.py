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
            obj = env._objects_dict[f"bin_{color}"]  # pylint: disable=protected-access
            region = obj.region_objects[f"table_1_object_goal_{color}_region"][0]
            size = np.fromstring(region.site_element.get("size"), sep=" ")
            pos = np.fromstring(region.site_element.get("pos"), sep=" ")
            assert np.all(size[:2] <= 0.04)
            assert pos[2] - size[2] > 0.01
            assert pos[2] + size[2] < 0.1
    finally:
        env.close()
