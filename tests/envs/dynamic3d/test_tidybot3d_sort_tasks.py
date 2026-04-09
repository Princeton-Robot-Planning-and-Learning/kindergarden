"""Tests for TidyBot3D sort tasks."""

from pathlib import Path

import pytest

import kinder
from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv


def test_tidybot_lab2_fit_blocks_in_cupboard_goals():
    """Test that tidybot-lab2-o6-fit_the_blocks_in_the_cupboard achieves goals.

    Initializes the environment, places all cuboids in their goal regions, and verifies
    that env._check_goals() returns True.
    """
    tasks_root = (
        Path(kinder.__path__[0]).parent / "kinder" / "envs" / "dynamic3d" / "tasks"
    )
    task_config_path = (
        tasks_root / "sort" / "tidybot-lab2-o6-fit_the_blocks_in_the_cupboard.json"
    )

    if not task_config_path.exists():
        pytest.skip(
            f"Task config not found: {task_config_path}. "
            "This test requires the fit_the_blocks_in_the_cupboard task configuration."
        )

    # Initialize environment with the task config
    env = ObjectCentricTidyBot3DEnv(
        scene_type="lab2",
        num_objects=6,
        task_config_path=str(task_config_path),
        scene_bg=False,
        allow_state_access=True,
    )

    # Reset the environment
    env.reset(seed=0)

    # After reset, goals should not be satisfied
    assert (
        not env._check_goals()  # pylint: disable=protected-access
    ), "Goals should not be satisfied after reset"

    # Get the current state
    current_state = env._get_current_state()  # pylint: disable=protected-access

    # Get all objects
    objects_dict = env._objects_dict  # pylint: disable=protected-access

    # Get all cupboard fixtures
    cupboards_dict = {}
    for fixture in env._fixtures_dict.values():  # pylint: disable=protected-access
        if fixture.name.startswith("cupboard_"):
            cupboards_dict[fixture.name] = fixture

    assert len(cupboards_dict) >= 6, "Should have at least 6 cupboards"

    # Create a modified state with objects in their goal regions
    modified_state = current_state.copy()

    # Goal predicates from the task config:
    # ["on", "cuboid_0", "cupboard_0_shelf_1_goal"],
    # ["on", "cuboid_1", "cupboard_1_shelf_1_goal"],
    # ["on", "cuboid_2", "cupboard_2_shelf_1_goal"],
    # ["on", "cuboid_3", "cupboard_3_shelf_0_goal"],
    # ["on", "cuboid_4", "cupboard_4_shelf_0_goal"],
    # ["on", "cuboid_5", "cupboard_5_shelf_0_goal"]

    goal_mapping = {
        "cuboid_0": ("cupboard_0", "cupboard_0_shelf_1_goal"),
        "cuboid_1": ("cupboard_1", "cupboard_1_shelf_1_goal"),
        "cuboid_2": ("cupboard_2", "cupboard_2_shelf_1_goal"),
        "cuboid_3": ("cupboard_3", "cupboard_3_shelf_0_goal"),
        "cuboid_4": ("cupboard_4", "cupboard_4_shelf_0_goal"),
        "cuboid_5": ("cupboard_5", "cupboard_5_shelf_0_goal"),
    }

    # Place each cuboid in its goal region
    for cuboid_name, (cupboard_name, region_name) in goal_mapping.items():
        cuboid = objects_dict.get(cuboid_name)
        if cuboid is None:
            continue

        cupboard = cupboards_dict.get(cupboard_name)
        if cupboard is None:
            continue

        # Get the region object and compute the center of its bbox
        region = cupboard.region_objects[region_name][0]
        region.env = env._robot_env  # pylint: disable=protected-access
        bbox = region.bbox
        goal_pos = (
            (bbox[0] + bbox[3]) / 2.0,
            (bbox[1] + bbox[4]) / 2.0,
            (bbox[2] + bbox[5]) / 2.0,
        )
        goal_pos = list(goal_pos)
        goal_pos[0] -= 0.1  # slight offset in x to fit better

        # Set position in the modified state
        modified_state.set(cuboid.symbolic_object, "x", goal_pos[0])
        modified_state.set(cuboid.symbolic_object, "y", goal_pos[1])
        modified_state.set(cuboid.symbolic_object, "z", goal_pos[2])

        # Set 90-degree rotation around x-axis: quaternion components
        # qw=cos(45°), qx=sin(45°), qy=0, qz=0
        modified_state.set(cuboid.symbolic_object, "qw", 0.7071067811865476)
        modified_state.set(cuboid.symbolic_object, "qx", 0.7071067811865475)
        modified_state.set(cuboid.symbolic_object, "qy", 0.0)
        modified_state.set(cuboid.symbolic_object, "qz", 0.0)

    # Set the modified state in the environment
    env.set_state(modified_state)

    # Verify all cuboids are in their goal positions
    state_after = env._get_current_state()  # pylint: disable=protected-access

    for cuboid_name, (cupboard_name, region_name) in goal_mapping.items():
        cuboid = objects_dict.get(cuboid_name)
        if cuboid is None:
            continue

        cuboid_pos = [
            state_after.get(cuboid.symbolic_object, "x"),
            state_after.get(cuboid.symbolic_object, "y"),
            state_after.get(cuboid.symbolic_object, "z"),
        ]

        assert isinstance(
            cuboid_pos[0], (int, float)
        ), f"Cuboid {cuboid_name} X position should be numeric"
        assert isinstance(
            cuboid_pos[1], (int, float)
        ), f"Cuboid {cuboid_name} Y position should be numeric"
        assert isinstance(
            cuboid_pos[2], (int, float)
        ), f"Cuboid {cuboid_name} Z position should be numeric"

    # Now check that goals are satisfied
    goals_satisfied = env._check_goals()  # pylint: disable=protected-access
    assert (
        goals_satisfied
    ), "Goals should be satisfied after placing all cuboids in their goal regions"

    env.close()
