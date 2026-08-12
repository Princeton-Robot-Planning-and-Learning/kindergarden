"""Tests for the registered ConstrainedCupboard3D task variants."""

import gymnasium
import pytest

import kinder


@pytest.mark.parametrize(("count", "expected_cupboards"), [(3, 6), (4, 6), (5, 11)])
def test_intermediate_constrained_cupboard_task(
    count: int, expected_cupboards: int
) -> None:
    """Each intermediate task registers, resets, and contains the declared rods."""
    kinder.register_all_environments()
    env = kinder.make(
        f"kinder/ConstrainedCupboard3D-o{count}-v0",
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
        names = state.get_object_names()
        assert sum(name.startswith("cuboid_") for name in names) == count
        assert len(inner.task_config["fixtures"]["cupboard"]) == expected_cupboards
        assert (
            len(inner.task_config["regions"]["cupboard_init_region"]["ranges"])
            == expected_cupboards
        )

        goal_state = state.copy()
        fixtures = getattr(inner, "_fixtures_dict")
        robot_env = getattr(inner, "_robot_env")
        for _, object_name, region_name in inner.task_config["goal_state"]:
            region_config = inner.task_config["regions"][region_name]
            region = fixtures[region_config["target"]].region_objects[region_name][0]
            region.env = robot_env
            x_min, y_min, z_min, x_max, y_max, z_max = region.bbox
            obj = goal_state.get_object_from_name(object_name)
            goal_state.set(obj, "x", (x_min + x_max) / 2)
            goal_state.set(obj, "y", (y_min + y_max) / 2)
            goal_state.set(obj, "z", (z_min + z_max) / 2)
        inner.set_state(goal_state)
        assert getattr(inner, "_check_goals")()
    finally:
        env.close()
        inner.close()
