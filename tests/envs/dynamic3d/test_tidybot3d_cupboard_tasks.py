"""Tests for TidyBot3D shelf tasks."""

from pathlib import Path

import numpy as np
import pytest

from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv


def test_tidybot_cupboard_real_o1_goals():
    """Test that tidybot-cupboard_real-o1 task correctly checks goals.

    Places the cube in the goal region and verifies that env._check_goals() returns
    True. Then places the cube at 10 different points in front of the cupboard and
    verifies that env._check_goals() returns False. Renders and saves all states to a
    folder.
    """
    tasks_root = Path(__file__).parent / "test_tasks"
    task_config_path = tasks_root / "tidybot-cupboard_real-o1.json"

    if not task_config_path.exists():
        pytest.skip(
            f"Task config not found: {task_config_path}. "
            "This test requires the tidybot-cupboard_real-o1 task configuration."
        )

    # Initialize environment with the task config
    env = ObjectCentricTidyBot3DEnv(
        scene_type="simple",
        num_objects=1,
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

    # Get the cupboard fixture
    cupboard = None
    for fixture in env._fixtures_dict.values():  # pylint: disable=protected-access
        if fixture.name == "cupboard_1":
            cupboard = fixture
            break

    assert cupboard is not None, "Cupboard fixture should exist in environment"

    # Get the cube object
    cube = objects_dict.get("cube1")
    assert cube is not None, "Cube object should exist in environment"

    # # Output folder for renders
    # output_dir = Path(__file__).parent / "output_images" / "cupboard_real_o1"
    # output_dir.mkdir(parents=True, exist_ok=True)

    # Test 1: Place cube in goal region and verify goals are satisfied
    modified_state = current_state.copy()

    # Get the goal region bbox
    goal_region = cupboard.region_objects["cupboard_1_cube_1_goal_region"][0]
    goal_region.env = env._robot_env  # pylint: disable=protected-access
    bbox = goal_region.bbox
    goal_pos = (
        (bbox[0] + bbox[3]) / 2.0,
        (bbox[1] + bbox[4]) / 2.0,
        (bbox[2] + bbox[5]) / 2.0,
    )

    # Set position and rotation
    modified_state.set(cube.symbolic_object, "x", goal_pos[0])
    modified_state.set(cube.symbolic_object, "y", goal_pos[1])
    modified_state.set(cube.symbolic_object, "z", goal_pos[2])
    modified_state.set(cube.symbolic_object, "qw", 1.0)
    modified_state.set(cube.symbolic_object, "qx", 0.0)
    modified_state.set(cube.symbolic_object, "qy", 0.0)
    modified_state.set(cube.symbolic_object, "qz", 0.0)

    # Set the modified state
    env.set_state(modified_state)

    # Verify goals are satisfied
    goals_satisfied = env._check_goals()  # pylint: disable=protected-access
    assert goals_satisfied, "Goals should be satisfied when cube is in goal region"

    # Test 1b: Place cube at goal_pos with noise and verify goals still satisfied
    noisy_positions = [
        (goal_pos[0] + 0.05, goal_pos[1] + 0.05, goal_pos[2]),
        (goal_pos[0] - 0.05, goal_pos[1] + 0.05, goal_pos[2]),
        (goal_pos[0] + 0.05, goal_pos[1] - 0.05, goal_pos[2]),
        (goal_pos[0] - 0.05, goal_pos[1] - 0.05, goal_pos[2]),
        (goal_pos[0] + 0.08, goal_pos[1], goal_pos[2]),
        (goal_pos[0] - 0.08, goal_pos[1], goal_pos[2]),
        (goal_pos[0], goal_pos[1] + 0.08, goal_pos[2]),
        (goal_pos[0], goal_pos[1] - 0.08, goal_pos[2]),
        (goal_pos[0] + 0.06, goal_pos[1] + 0.06, goal_pos[2]),
        (goal_pos[0] - 0.06, goal_pos[1] - 0.06, goal_pos[2]),
    ]

    for idx, (x, y, z) in enumerate(noisy_positions, start=1):
        modified_state = current_state.copy()

        # Set position with noise
        modified_state.set(cube.symbolic_object, "x", x)
        modified_state.set(cube.symbolic_object, "y", y)
        modified_state.set(cube.symbolic_object, "z", z)
        modified_state.set(cube.symbolic_object, "qw", 1.0)
        modified_state.set(cube.symbolic_object, "qx", 0.0)
        modified_state.set(cube.symbolic_object, "qy", 0.0)
        modified_state.set(cube.symbolic_object, "qz", 0.0)

        # Set the modified state
        env.set_state(modified_state)

        # Render and save
        # image = env.render()
        # image_path = output_dir / f"01b_goal_with_noise_{idx}.png"
        # Image.fromarray(image).save(image_path)

        # Verify goals are still satisfied with noise
        print("expected result: True")  ##DEBUG##
        goals_satisfied = env._check_goals()  # pylint: disable=protected-access
        assert (
            goals_satisfied
        ), f"Goals should be satisfied with noise at position {idx}: ({x}, {y}, {z})"

    # Test 2: Place cube at 20 different points sampled from a bbox in front of
    # the cupboard and verify goals are NOT satisfied
    # Bbox: [x_min, y_min, z_min, x_max, y_max, z_max]
    start_of_shelf = 1.5 - 0.254 / 2  # cupboard x position - half shelf depth
    start_of_shelf -= 0.02  # extra offset so that block is always outside of shelf
    print("start_of_shelf:", start_of_shelf)  ##DEBUG## == 1.353
    front_bbox = np.array([1.28, -0.1, 0.4, start_of_shelf, 0.1, 0.6])
    rng = np.random.default_rng(seed=42)
    front_positions = []
    for _ in range(20):
        x = rng.uniform(front_bbox[0], front_bbox[3])
        y = rng.uniform(front_bbox[1], front_bbox[4])
        z = rng.uniform(front_bbox[2], front_bbox[5])
        front_positions.append((x, y, z))

    for idx, (x, y, z) in enumerate(front_positions, start=1):
        modified_state = current_state.copy()

        # Set position
        modified_state.set(cube.symbolic_object, "x", x)
        modified_state.set(cube.symbolic_object, "y", y)
        modified_state.set(cube.symbolic_object, "z", z)
        modified_state.set(cube.symbolic_object, "qw", 1.0)
        modified_state.set(cube.symbolic_object, "qx", 0.0)
        modified_state.set(cube.symbolic_object, "qy", 0.0)
        modified_state.set(cube.symbolic_object, "qz", 0.0)

        # Set the modified state
        env.set_state(modified_state)

        # # Render and save
        # image = env.render()
        # image_path = output_dir / f"{idx:02d}_front_pos_{idx}.png"
        # Image.fromarray(image).save(image_path)

        # Verify goals are NOT satisfied
        print()
        print("expected result: False")  ##DEBUG##
        goals_satisfied = env._check_goals()  # pylint: disable=protected-access
        assert (
            not goals_satisfied
        ), f"Goals should NOT be satisfied when cube is at front position {idx}"

    env.close()


def test_tidybot_lab2_kitchen_o5_sweep_blocks_goal():
    """Test that tidybot-lab2_kitchen-o5 task correctly checks goals for sweeping
    blocks.

    Places all cubes in the goal region and verifies that env._check_goals() returns
    True. Then places cubes at various points on the kitchen island and verifies that
    env._check_goals() returns False when not all cubes are in the goal region. Renders
    and saves all states to a folder.
    """
    tasks_root = Path(__file__).parent / "test_tasks"
    task_filename = (
        "tidybot-lab2_kitchen-o5-sweep_the_blocks_into_the_top_drawer"
        "_of_the_kitchen_island.json"
    )
    task_config_path = tasks_root / task_filename

    if not task_config_path.exists():
        pytest.skip(
            f"Task config not found: {task_config_path}. "
            "This test requires the tidybot-lab2_kitchen-o5 task configuration."
        )

    # Initialize environment with the task config
    env = ObjectCentricTidyBot3DEnv(
        scene_type="simple",
        num_objects=5,
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

    # Get the kitchen island fixture to find the goal region
    kitchen_island = None
    for fixture in env._fixtures_dict.values():  # pylint: disable=protected-access
        if fixture.name == "kitchen_island":
            kitchen_island = fixture
            break

    assert (
        kitchen_island is not None
    ), "Kitchen island fixture should exist in environment"

    # Get all cube objects
    cubes = {}
    for i in range(5):
        cube_key = f"cube_{i}"
        cube = objects_dict.get(cube_key)
        assert cube is not None, f"Cube object {cube_key} should exist in environment"
        cubes[cube_key] = cube

    # # Output folder for renders
    # output_dir = Path(__file__).parent / "output_images" / "kitchen_o5"
    # output_dir.mkdir(parents=True, exist_ok=True)

    # Test 1: Place all cubes in goal region and verify goals are satisfied
    modified_state = current_state.copy()

    # Get the goal region bbox
    goal_region_list = kitchen_island.region_objects[
        "kitchen_island_shelf_1_partition_1_region"
    ]
    goal_region = goal_region_list[0]
    goal_region.env = env._robot_env  # pylint: disable=protected-access
    bbox = goal_region.bbox
    goal_pos = (
        (bbox[0] + bbox[3]) / 2.0,
        (bbox[1] + bbox[4]) / 2.0,
        (bbox[2] + bbox[5]) / 2.0,
    )

    # Set all cubes to the goal position
    for cube_key, cube in cubes.items():
        modified_state.set(cube.symbolic_object, "x", goal_pos[0])
        modified_state.set(cube.symbolic_object, "y", goal_pos[1])
        modified_state.set(cube.symbolic_object, "z", goal_pos[2])
        modified_state.set(cube.symbolic_object, "qw", 1.0)
        modified_state.set(cube.symbolic_object, "qx", 0.0)
        modified_state.set(cube.symbolic_object, "qy", 0.0)
        modified_state.set(cube.symbolic_object, "qz", 0.0)

    # Set the modified state
    env.set_state(modified_state)

    # Verify goals are satisfied
    goals_satisfied = env._check_goals()  # pylint: disable=protected-access
    assert (
        goals_satisfied
    ), "Goals should be satisfied when all cubes are in goal region"

    # Test 2: Place all cubes with noise around goal position and verify
    # goals are still satisfied
    noisy_offsets = [
        (0.05, 0.05, 0.0),
        (-0.05, 0.05, 0.0),
        (0.05, -0.05, 0.0),
        (-0.05, -0.05, 0.0),
        (0.08, 0.0, 0.0),
    ]

    for idx, (dx, dy, dz) in enumerate(noisy_offsets, start=1):
        modified_state = current_state.copy()

        # Set all cubes to noisy goal positions
        for cube_key, cube in cubes.items():
            modified_state.set(cube.symbolic_object, "x", goal_pos[0] + dx)
            modified_state.set(cube.symbolic_object, "y", goal_pos[1] + dy)
            modified_state.set(cube.symbolic_object, "z", goal_pos[2] + dz)
            modified_state.set(cube.symbolic_object, "qw", 1.0)
            modified_state.set(cube.symbolic_object, "qx", 0.0)
            modified_state.set(cube.symbolic_object, "qy", 0.0)
            modified_state.set(cube.symbolic_object, "qz", 0.0)

        # Set the modified state
        env.set_state(modified_state)

        # # Render and save
        # image = env.render()
        # image_path = output_dir / f"01_all_cubes_in_goal_region_with_noise_{idx}.png"
        # Image.fromarray(image).save(image_path)

        # Verify goals are still satisfied with noise
        print("expected result: True")  ##DEBUG##
        goals_satisfied = env._check_goals()  # pylint: disable=protected-access
        assert (
            goals_satisfied
        ), f"Goals should be satisfied with noise offset ({dx}, {dy}, {dz})"

    # Test 3: Place cubes at various positions on the shelf but outside goal region
    # and verify goals are NOT satisfied
    off_goal_positions = [
        (
            bbox[0] - 0.05,
            bbox[1] - 0.05,
            bbox[2],
        ),  # Left and front of goal
        (
            bbox[3] + 0.05,
            bbox[1] - 0.05,
            bbox[2],
        ),  # Right and front of goal
        (
            (bbox[0] + bbox[3]) / 2.0,
            bbox[1] - 0.1,
            bbox[2],
        ),  # Front of goal
        (
            (bbox[0] + bbox[3]) / 2.0,
            bbox[4] + 0.05,
            bbox[2],
        ),  # Back of goal
        (
            (bbox[0] + bbox[3]) / 2.0,
            (bbox[1] + bbox[4]) / 2.0,
            bbox[5] + 0.05,
        ),  # Above goal
    ]

    for idx, (x, y, z) in enumerate(off_goal_positions, start=1):
        modified_state = current_state.copy()

        # Set all cubes to off-goal position
        for cube_key, cube in cubes.items():
            modified_state.set(cube.symbolic_object, "x", x)
            modified_state.set(cube.symbolic_object, "y", y)
            modified_state.set(cube.symbolic_object, "z", z)
            modified_state.set(cube.symbolic_object, "qw", 1.0)
            modified_state.set(cube.symbolic_object, "qx", 0.0)
            modified_state.set(cube.symbolic_object, "qy", 0.0)
            modified_state.set(cube.symbolic_object, "qz", 0.0)

        # Set the modified state
        env.set_state(modified_state)

        # # Render and save
        # image = env.render()
        # image_path = output_dir / f"02_all_cubes_off_goal_{idx}.png"
        # Image.fromarray(image).save(image_path)

        # Verify goals are NOT satisfied
        print()
        print("expected result: False")  ##DEBUG##
        goals_satisfied = env._check_goals()  # pylint: disable=protected-access
        assert (
            not goals_satisfied
        ), f"Goals should NOT be satisfied when all cubes are at off-goal position {idx}"

    # Test 4: Place only some cubes in goal region and verify goals are NOT satisfied
    modified_state = current_state.copy()

    # Place first 2 cubes in goal region, rest outside
    for i, (cube_key, cube) in enumerate(cubes.items()):
        if i < 2:
            # In goal region
            modified_state.set(cube.symbolic_object, "x", goal_pos[0])
            modified_state.set(cube.symbolic_object, "y", goal_pos[1])
            modified_state.set(cube.symbolic_object, "z", goal_pos[2])
        else:
            # Outside goal region
            modified_state.set(cube.symbolic_object, "x", bbox[0] - 0.1)
            modified_state.set(cube.symbolic_object, "y", bbox[1] - 0.1)
            modified_state.set(cube.symbolic_object, "z", bbox[2])
        modified_state.set(cube.symbolic_object, "qw", 1.0)
        modified_state.set(cube.symbolic_object, "qx", 0.0)
        modified_state.set(cube.symbolic_object, "qy", 0.0)
        modified_state.set(cube.symbolic_object, "qz", 0.0)

    # Set the modified state
    env.set_state(modified_state)

    # # Render and save
    # image = env.render()
    # image_path = output_dir / "03_partial_cubes_in_goal.png"
    # Image.fromarray(image).save(image_path)

    # Verify goals are NOT satisfied (only 2 out of 5 cubes in goal)
    print()
    print("expected result: False (only 2/5 cubes in goal)")  ##DEBUG##
    goals_satisfied = env._check_goals()  # pylint: disable=protected-access
    assert (
        not goals_satisfied
    ), "Goals should NOT be satisfied when only some cubes are in goal region"

    env.close()
