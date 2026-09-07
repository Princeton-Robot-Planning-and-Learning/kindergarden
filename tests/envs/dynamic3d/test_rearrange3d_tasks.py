"""Rearrange goals accept objects resting upright beside an upright bowl."""

from pathlib import Path

import numpy as np
import pytest

import kinder
from kinder.envs.dynamic3d import envs

_TASKS = sorted((Path(envs.__file__).parent / "tasks" / "Rearrange3D").glob("*.json"))


@pytest.mark.parametrize("task_path", _TASKS, ids=lambda p: p.stem)
def test_rearrange_resting_goal(task_path: Path) -> None:
    """Keep physical resting heights while moving objects to their target side."""
    kinder.register_all_environments()
    env = kinder.make(f"kinder/{task_path.stem}-v0", allow_state_access=True)
    inner = getattr(env.unwrapped, "_object_centric_env")
    try:
        env.reset(seed=0)
        state = inner.get_state()
        bowl = state.get_object_from_name("bowl_0")
        goals = inner.task_config["goal_state"]
        for index, (_, name, region_name) in enumerate(goals):
            obj = state.get_object_from_name(name)
            bounds = inner.task_config["regions"][region_name]["ranges"][0]
            # Stay away from the bowl mesh, inside the requested planar region.
            upper = 2 if len(bounds) == 4 else 3
            x = (bounds[0] + bounds[upper]) / 2
            y = (bounds[1] + bounds[upper + 1]) / 2
            if abs(x) > 0:
                x = float(np.sign(x) * 0.12)
            elif abs(y) > 0:
                y = float(np.sign(y) * 0.12)
            else:
                y = 0.12 if index == 0 else -0.12
            state.set(obj, "x", state.get(bowl, "x") + x)
            state.set(obj, "y", state.get(bowl, "y") + y)
        inner.set_state(state)
        for _ in range(10):
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert inner._check_goals()  # pylint: disable=protected-access
        resting = inner.get_state()
        obj = resting.get_object_from_name(goals[0][1])
        for feature, offset in [("x", 0.5), ("z", 0.1)]:
            outside = resting.copy()
            outside.set(obj, feature, resting.get(obj, feature) + offset)
            inner.set_state(outside)
            assert not inner._check_goals()  # pylint: disable=protected-access
    finally:
        env.close()
        inner.close()
