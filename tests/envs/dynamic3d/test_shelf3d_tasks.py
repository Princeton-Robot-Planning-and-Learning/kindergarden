"""Tests for the registered Shelf3D task variants."""

import gymnasium
import pytest

import kinder
from kinder.envs.dynamic3d.task_families import Shelf3DEnv


@pytest.mark.parametrize("count", sorted(Shelf3DEnv.supported_counts))
def test_shelf3d_task_goal_covers_every_cube(count: int) -> None:
    """Each variant resets outside its goal, and the goal names every cube.

    The goal is also shown to be satisfiable: a state with every cube inside its
    shelf region passes the goal check.
    """
    kinder.register_all_environments()
    env = kinder.make(
        f"kinder/Shelf3D-o{count}-v0",
        render_mode="rgb_array",
        scene_bg=False,
        allow_state_access=True,
    )
    assert isinstance(env, gymnasium.Env)
    inner = getattr(env.unwrapped, "_object_centric_env")
    try:
        observation, _ = env.reset(seed=count)
        assert env.observation_space.contains(observation)
        state = inner.get_state()
        cubes = sorted(n for n in state.get_object_names() if n.startswith("cube"))
        assert len(cubes) == count
        goal_predicates = inner.task_config["goal_state"]
        assert sorted(name for _, name, _ in goal_predicates) == cubes
        assert not getattr(inner, "_check_goals")()

        goal_state = state.copy()
        fixtures = getattr(inner, "_fixtures_dict")
        robot_env = getattr(inner, "_robot_env")
        for index, (_, object_name, region_name) in enumerate(goal_predicates):
            region_config = inner.task_config["regions"][region_name]
            region = fixtures[region_config["target"]].region_objects[region_name][0]
            region.env = robot_env
            x_min, y_min, z_min, x_max, y_max, z_max = region.bbox
            obj = goal_state.get_object_from_name(object_name)
            # Spread the cubes along the shelf so they do not overlap.
            goal_state.set(obj, "x", x_min + (index + 0.5) * (x_max - x_min) / count)
            goal_state.set(obj, "y", (y_min + y_max) / 2)
            goal_state.set(obj, "z", (z_min + z_max) / 2)
        inner.set_state(goal_state)
        assert getattr(inner, "_check_goals")()
    finally:
        env.close()
        inner.close()
