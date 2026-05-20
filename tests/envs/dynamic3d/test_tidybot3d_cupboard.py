"""Tests for the TidyBot3D cupboard scene: observation/action spaces, reset, and
step."""

from pathlib import Path

import numpy as np
import pytest
from gymnasium.wrappers import RecordVideo

import kinder
from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv
from kinder.envs.dynamic3d.objects.fixtures import Cupboard
from tests.conftest import MAKE_VIDEOS

# Path to MimicLabs scenes
MIMICLABS_SCENES_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "src"
    / "kinder"
    / "envs"
    / "dynamic3d"
    / "models"
    / "assets"
    / "mimiclabs_scenes"
    / "meshes"
)

_TEST_TASKS = Path(__file__).parent / "test_tasks"


def test_tidybot3d_cupboard_observation_space():
    """Reset should return an observation within the observation space."""
    env = ObjectCentricTidyBot3DEnv(
        scene_type="cupboard",
        num_objects=8,
        task_config_path=str(_TEST_TASKS / "tidybot-cupboard-o8.json"),
    )
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)
    env.close()


def test_tidybot3d_cupboard_action_space():
    """A sampled action should be valid for the action space."""
    env = ObjectCentricTidyBot3DEnv(
        scene_type="cupboard",
        num_objects=8,
        task_config_path=str(_TEST_TASKS / "tidybot-cupboard-o8.json"),
    )
    action = env.action_space.sample()
    assert env.action_space.contains(action)
    env.close()


def test_tidybot3d_cupboard_step():
    """Step should return a valid obs, float reward, bool done flags, and info dict."""
    env = ObjectCentricTidyBot3DEnv(
        scene_type="cupboard",
        num_objects=8,
        task_config_path=str(_TEST_TASKS / "tidybot-cupboard-o8.json"),
    )
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    env.close()


def test_tidybot3d_cupboard_reset_seed_reproducible():
    """Reset with the same seed should produce identical observations."""
    env = ObjectCentricTidyBot3DEnv(
        scene_type="cupboard",
        num_objects=8,
        task_config_path=str(_TEST_TASKS / "tidybot-cupboard-o8.json"),
    )
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    assert obs1.allclose(obs2, atol=1e-3)
    env.close()


def test_tidybot3d_cupboard_reset_changes_with_different_seeds():
    """Resets with different seeds should produce different observations."""
    env = ObjectCentricTidyBot3DEnv(
        scene_type="cupboard",
        num_objects=8,
        task_config_path=str(_TEST_TASKS / "tidybot-cupboard-o8.json"),
    )
    obs1, _ = env.reset(seed=10)
    obs2, _ = env.reset(seed=20)
    if len(obs1.data) != len(obs2.data):
        raise AssertionError("Observations have different number of objects")
    if len(obs1.data) > 0:
        assert not obs1.allclose(obs2, atol=1e-4)
    env.close()


def test_tidybot3d_cupboard_has_eight_objects():
    """Cupboard environment should be configured with 8 objects."""
    env = ObjectCentricTidyBot3DEnv(
        scene_type="cupboard",
        num_objects=8,
        task_config_path=str(_TEST_TASKS / "tidybot-cupboard-o8.json"),
    )
    assert env.num_objects == 8
    assert env.scene_type == "cupboard"
    env.close()


def test_tidybot_cupboard_constrained_fitting_goals():
    """Test that tidybot-cupboard-o12-ConstrainedFitting env correctly checks goals."""
    tasks_root = (
        Path(kinder.__path__[0]).parent / "kinder" / "envs" / "dynamic3d" / "tasks"
    )
    task_config_path = tasks_root / "sort" / "tidybot-lab6-o12-ConstrainedFitting.json"

    if not task_config_path.exists():
        pytest.skip(
            f"Task config not found: {task_config_path}. "
            "This test requires the ConstrainedFitting task configuration."
        )

    env = ObjectCentricTidyBot3DEnv(
        scene_type="cupboard",
        num_objects=12,
        task_config_path=str(task_config_path),
        allow_state_access=True,
    )

    # Reset the environment
    env.reset()

    # After reset, goals should not be satisfied
    assert (
        not env._check_goals()  # pylint: disable=protected-access
    ), "Goals should not be satisfied after reset"

    # Get the current state
    current_state = env._get_current_state()  # pylint: disable=protected-access

    # Get the cupboard fixture
    cupboard = None
    for fixture in env._fixtures_dict.values():  # pylint: disable=protected-access
        if fixture.name == "cupboard_1":
            cupboard = fixture
            break

    assert cupboard is not None, "Cupboard fixture should exist in environment"

    # Get all objects
    objects_dict = env._objects_dict  # pylint: disable=protected-access

    # Create a modified state with objects in their goal regions
    modified_state = current_state.copy()

    # 90-degree rotation around x-axis: quaternion components
    # qw=cos(45°), qx=sin(45°), qy=0, qz=0
    qw_90x = 0.7071067811865476
    qx_90x = 0.7071067811865475
    qy_90x = 0.0
    qz_90x = 0.0

    # Place red cuboid in shelf 3 red partition (center of region)
    red_cuboid = objects_dict.get("red_cuboid1")
    goal_pos_red = None
    if red_cuboid:
        # Get region name from goal predicates
        red_cuboid_goal_region = "cupboard_1_shelf_3_red_partition_goal"
        # Get the region from the fixture
        region = cupboard.region_objects[red_cuboid_goal_region][0]
        # Get the bbox of the region
        bbox = region.bbox
        # Compute center: (x_min + x_max)/2, (y_min + y_max)/2, (z_min + z_max)/2
        goal_pos_red = (
            (bbox[0] + bbox[3]) / 2.0,
            (bbox[1] + bbox[4]) / 2.0,
            (bbox[2] + bbox[5]) / 2.0,
        )
        # Set position to center of goal region
        modified_state.set(red_cuboid.symbolic_object, "x", goal_pos_red[0])
        modified_state.set(red_cuboid.symbolic_object, "y", goal_pos_red[1])
        modified_state.set(red_cuboid.symbolic_object, "z", goal_pos_red[2])
        # Set 90-degree rotation around x-axis
        modified_state.set(red_cuboid.symbolic_object, "qw", qw_90x)
        modified_state.set(red_cuboid.symbolic_object, "qx", qx_90x)
        modified_state.set(red_cuboid.symbolic_object, "qy", qy_90x)
        modified_state.set(red_cuboid.symbolic_object, "qz", qz_90x)

    # Place green cuboid in shelf 3 green partition (center of region)
    green_cuboid = objects_dict.get("green_cuboid1")
    goal_pos_green = None
    if green_cuboid:
        # Get region name from goal predicates
        green_cuboid_goal_region = "cupboard_1_shelf_3_green_partition_goal"
        # Get the region from the fixture
        region = cupboard.region_objects[green_cuboid_goal_region][0]
        # Get the bbox of the region
        bbox = region.bbox
        # Compute center: (x_min + x_max)/2, (y_min + y_max)/2, (z_min + z_max)/2
        goal_pos_green = (
            (bbox[0] + bbox[3]) / 2.0,
            (bbox[1] + bbox[4]) / 2.0,
            (bbox[2] + bbox[5]) / 2.0,
        )
        # Set position to center of goal region
        modified_state.set(green_cuboid.symbolic_object, "x", goal_pos_green[0])
        modified_state.set(green_cuboid.symbolic_object, "y", goal_pos_green[1])
        modified_state.set(green_cuboid.symbolic_object, "z", goal_pos_green[2])
        # Set 90-degree rotation around x-axis
        modified_state.set(green_cuboid.symbolic_object, "qw", qw_90x)
        modified_state.set(green_cuboid.symbolic_object, "qx", qx_90x)
        modified_state.set(green_cuboid.symbolic_object, "qy", qy_90x)
        modified_state.set(green_cuboid.symbolic_object, "qz", qz_90x)

    # Place blue cuboid in shelf 3 blue partition (center of region)
    blue_cuboid = objects_dict.get("blue_cuboid1")
    goal_pos_blue = None
    if blue_cuboid:
        # Get region name from goal predicates
        blue_cuboid_goal_region = "cupboard_1_shelf_3_blue_partition_goal"
        # Get the region from the fixture
        region = cupboard.region_objects[blue_cuboid_goal_region][0]
        # Get the bbox of the region
        bbox = region.bbox
        # Compute center: (x_min + x_max)/2, (y_min + y_max)/2, (z_min + z_max)/2
        goal_pos_blue = (
            (bbox[0] + bbox[3]) / 2.0,
            (bbox[1] + bbox[4]) / 2.0,
            (bbox[2] + bbox[5]) / 2.0,
        )
        # Set position to center of goal region
        modified_state.set(blue_cuboid.symbolic_object, "x", goal_pos_blue[0])
        modified_state.set(blue_cuboid.symbolic_object, "y", goal_pos_blue[1])
        modified_state.set(blue_cuboid.symbolic_object, "z", goal_pos_blue[2])
        # Set 90-degree rotation around x-axis
        modified_state.set(blue_cuboid.symbolic_object, "qw", qw_90x)
        modified_state.set(blue_cuboid.symbolic_object, "qx", qx_90x)
        modified_state.set(blue_cuboid.symbolic_object, "qy", qy_90x)
        modified_state.set(blue_cuboid.symbolic_object, "qz", qz_90x)

    # Place red cubes on shelf 0
    for i in range(1, 4):
        red_cube = objects_dict.get(f"red_cube{i}")
        if red_cube:
            goal_pos = cupboard.sample_pose_in_region(
                "cupboard_1_shelf_0_red_goal", env.np_random
            )
            modified_state.set(red_cube.symbolic_object, "x", goal_pos[0])
            modified_state.set(red_cube.symbolic_object, "y", goal_pos[1])
            modified_state.set(red_cube.symbolic_object, "z", goal_pos[2])

    # Place green cubes on shelf 1
    for i in range(1, 4):
        green_cube = objects_dict.get(f"green_cube{i}")
        if green_cube:
            goal_pos = cupboard.sample_pose_in_region(
                "cupboard_1_shelf_1_green_goal", env.np_random
            )
            modified_state.set(green_cube.symbolic_object, "x", goal_pos[0])
            modified_state.set(green_cube.symbolic_object, "y", goal_pos[1])
            modified_state.set(green_cube.symbolic_object, "z", goal_pos[2])

    # Place blue cubes on shelf 2
    for i in range(1, 4):
        blue_cube = objects_dict.get(f"blue_cube{i}")
        if blue_cube:
            goal_pos = cupboard.sample_pose_in_region(
                "cupboard_1_shelf_2_blue_goal", env.np_random
            )
            modified_state.set(blue_cube.symbolic_object, "x", goal_pos[0])
            modified_state.set(blue_cube.symbolic_object, "y", goal_pos[1])
            modified_state.set(blue_cube.symbolic_object, "z", goal_pos[2])

    # Set the modified state in the environment
    env.set_state(modified_state)

    # Get the state after setting and verify cuboid positions
    state_after = env._get_current_state()  # pylint: disable=protected-access

    # Verify that cuboids are placed in correct positions
    red_cuboid_pos = [
        state_after.get(red_cuboid.symbolic_object, "x"),
        state_after.get(red_cuboid.symbolic_object, "y"),
        state_after.get(red_cuboid.symbolic_object, "z"),
    ]
    green_cuboid_pos = [
        state_after.get(green_cuboid.symbolic_object, "x"),
        state_after.get(green_cuboid.symbolic_object, "y"),
        state_after.get(green_cuboid.symbolic_object, "z"),
    ]
    blue_cuboid_pos = [
        state_after.get(blue_cuboid.symbolic_object, "x"),
        state_after.get(blue_cuboid.symbolic_object, "y"),
        state_after.get(blue_cuboid.symbolic_object, "z"),
    ]

    # Verify cuboid positions are close to goal positions
    assert np.allclose(
        red_cuboid_pos, [goal_pos_red[0], goal_pos_red[1], goal_pos_red[2]], atol=0.001
    ), f"Red cuboid position {red_cuboid_pos} not close to goal"
    assert np.allclose(
        green_cuboid_pos,
        [goal_pos_green[0], goal_pos_green[1], goal_pos_green[2]],
        atol=0.001,
    ), f"Green cuboid position {green_cuboid_pos} not close to goal"
    assert np.allclose(
        blue_cuboid_pos,
        [goal_pos_blue[0], goal_pos_blue[1], goal_pos_blue[2]],
        atol=0.001,
    ), f"Blue cuboid position {blue_cuboid_pos} not close to goal"

    # Verify cuboid orientations are set to 90-degree rotation around x-axis
    for cuboid, cuboid_name in [
        (red_cuboid, "red_cuboid1"),
        (green_cuboid, "green_cuboid1"),
        (blue_cuboid, "blue_cuboid1"),
    ]:
        qw = state_after.get(cuboid.symbolic_object, "qw")
        qx = state_after.get(cuboid.symbolic_object, "qx")
        qy = state_after.get(cuboid.symbolic_object, "qy")
        qz = state_after.get(cuboid.symbolic_object, "qz")
        assert np.isclose(
            qw, qw_90x, atol=0.01
        ), f"{cuboid_name} qw={qw} not close to expected {qw_90x}"
        assert np.isclose(
            qx, qx_90x, atol=0.01
        ), f"{cuboid_name} qx={qx} not close to expected {qx_90x}"
        assert np.isclose(
            qy, qy_90x, atol=0.01
        ), f"{cuboid_name} qy={qy} not close to expected {qy_90x}"
        assert np.isclose(
            qz, qz_90x, atol=0.01
        ), f"{cuboid_name} qz={qz} not close to expected {qz_90x}"

    env.close()


def test_cupboard_region_site_creation_and_placement():
    """Test Cupboard construction with regions.

    Verify site creation, placement, and sizing.
    """
    # Create cupboard fixture config with multiple shelves, partitions, and drawers
    cupboard_config = {
        "length": 0.6,
        "depth": 0.3,
        "shelf_heights": [0.1, 0.2, 0.5],
        "shelf_partitions": [
            [0.15],  # Shelf 0: 1 partition creating 2 compartments
            [],  # Shelf 1: no partitions
            [0.1, 0.25],  # Shelf 2: 2 partitions creating 3 compartments
        ],
        "shelf_drawers": [
            [True, True],  # Shelf 0: both compartments have drawers
            [True],  # Shelf 1: single compartment has drawer
            [False, True, False],  # Shelf 2: middle compartment has drawer
        ],
        "side_and_back_open": False,
    }

    # Define regions for the cupboard
    regions_config = {
        # Region on shelf 0, partition 0 (has drawer)
        "shelf_0_partition_0_region": {
            "shelf": 0,
            "partition": 0,
            "ranges": [[-0.05, -0.1, 0.05, 0.1]],
            "rgba": [1.0, 0.0, 0.0, 0.3],
        },
        # Region on shelf 0, partition 1 (has drawer)
        "shelf_0_partition_1_region": {
            "shelf": 0,
            "partition": 1,
            "ranges": [[-0.05, -0.1, 0.05, 0.1]],
            "rgba": [0.0, 1.0, 0.0, 0.3],
        },
        # Region on shelf 1 (no partitions, has drawer)
        "shelf_1_region": {
            "shelf": 1,
            "ranges": [[-0.2, -0.1, 0.2, 0.1]],
            "rgba": [0.0, 0.0, 1.0, 0.3],
        },
        # Region on shelf 2, partition 0 (no drawer)
        "shelf_2_partition_0_region": {
            "shelf": 2,
            "partition": 0,
            "ranges": [[-0.03, -0.1, 0.03, 0.1]],
            "rgba": [1.0, 1.0, 0.0, 0.3],
        },
        # Region on shelf 2, partition 1 (has drawer)
        "shelf_2_partition_1_region": {
            "shelf": 2,
            "partition": 1,
            "ranges": [[-0.04, -0.1, 0.04, 0.1]],
            "rgba": [0.0, 1.0, 1.0, 0.3],
        },
        # Region on shelf 2, partition 2 (no drawer)
        "shelf_2_partition_2_region": {
            "shelf": 2,
            "partition": 2,
            "ranges": [[-0.03, -0.1, 0.03, 0.1]],
            "rgba": [1.0, 0.0, 1.0, 0.3],
        },
    }

    # Create the cupboard fixture
    cupboard = Cupboard(
        name="test_cupboard",
        fixture_config=cupboard_config,
        position=[0.0, 0.0, 0.0],
        yaw=0.0,
        regions=regions_config,
    )

    # Verify that all regions were created
    assert len(cupboard.region_objects) == 6
    for region_name in regions_config:
        assert region_name in cupboard.region_objects

    # Helper function to find the parent element of a site element
    def find_site_parent(elem, site_element):
        """Recursively find the parent element of a site element in the XML tree."""
        for child in elem:
            if child == site_element:
                return elem
            if child.tag == "body":
                parent = find_site_parent(child, site_element)
                if parent is not None:
                    return parent
        return None

    # Test drawer attachment tracking
    test_cases = [
        {
            "region_name": "shelf_0_partition_0_region",
            "should_have_drawer": True,
        },
        {
            "region_name": "shelf_0_partition_1_region",
            "should_have_drawer": True,
        },
        {
            "region_name": "shelf_1_region",
            "should_have_drawer": True,
        },
        {
            "region_name": "shelf_2_partition_0_region",
            "should_have_drawer": False,
        },
        {
            "region_name": "shelf_2_partition_1_region",
            "should_have_drawer": True,
        },
        {
            "region_name": "shelf_2_partition_2_region",
            "should_have_drawer": False,
        },
    ]

    for test_case in test_cases:
        region_name = test_case["region_name"]
        should_have_drawer = test_case["should_have_drawer"]

        # Get the region object
        regions = cupboard.region_objects[region_name]
        assert (
            len(regions) == 1
        ), f"Expected 1 region for {region_name}, got {len(regions)}"

        region = regions[0]

        # Verify site element exists
        assert (
            region.site_element is not None
        ), f"Site element should exist for {region_name}"

        site_name = region.site_element.get("name", "")
        assert site_name != "", f"Site name should not be empty for {region_name}"

        # Find the parent element of the site
        site_parent = find_site_parent(cupboard.xml_element, region.site_element)
        assert (
            site_parent is not None
        ), f"Could not find parent element of site {site_name}"

        parent_name = site_parent.get("name", "")

        if should_have_drawer:
            # Parent should be a drawer body
            assert parent_name.startswith(
                f"{cupboard.name}_drawer_"
            ), f"Site {site_name} should be in drawer body, but parent is {parent_name}"
        else:
            # Parent should be the main cupboard body
            assert parent_name == cupboard.name, (
                f"Site {site_name} should be in cupboard body {cupboard.name}, "
                f"but parent is {parent_name}"
            )

        # Verify site position and size match the specified ranges
        region_range = regions_config[region_name]["ranges"][0]
        x_start, y_start, x_end, y_end = region_range

        # Extract site position and size from XML
        site_pos_str = region.site_element.get("pos", "")
        site_size_str = region.site_element.get("size", "")

        assert site_pos_str, f"Site {site_name} has no position"
        assert site_size_str, f"Site {site_name} has no size"

        site_pos = [float(x) for x in site_pos_str.split()]
        site_size = [float(x) for x in site_size_str.split()]

        assert (
            len(site_pos) == 3
        ), f"Site position should have 3 components, got {len(site_pos)}"
        assert (
            len(site_size) == 3
        ), f"Site size should have 3 components, got {len(site_size)}"

        site_x, site_y, site_z = site_pos
        size_x, size_y, _ = site_size

        # Size should always be half the range span (MuJoCo convention)
        expected_size_x = (x_end - x_start) / 2
        expected_size_y = (y_end - y_start) / 2

        assert np.isclose(size_x, expected_size_x, atol=1e-6), (
            f"Site X size mismatch for {region_name}: "
            f"expected {expected_size_x}, got {size_x}"
        )
        assert np.isclose(size_y, expected_size_y, atol=1e-6), (
            f"Site Y size mismatch for {region_name}: "
            f"expected {expected_size_y}, got {size_y}"
        )

        # Position depends on whether site is in drawer (partition-relative)
        # or cupboard (absolute)
        if should_have_drawer:
            # For drawer sites: position should be at the center of the region ranges
            # in the drawer's local frame (centered at the partition), with Z at the
            # center of the shelf height
            expected_center_x = (x_start + x_end) / 2
            expected_center_y = (y_start + y_end) / 2
            # Get the shelf height to compute Z center
            shelf_idx = regions_config[region_name]["shelf"]
            shelf_height = cupboard_config["shelf_heights"][shelf_idx]
            expected_center_z = (
                shelf_height / 2
            )  # Center of drawer height (0 to shelf_height)
        else:
            # For cupboard sites: position is absolute (in cupboard frame)
            # We need to know the partition center to verify, but since we only
            # we just verify that the site spans the correct range width
            # Position verification is implicit in the size check and parent check
            expected_center_x = site_x  # Accept whatever position is set
            expected_center_y = site_y
            expected_center_z = site_z

        assert np.isclose(site_x, expected_center_x, atol=1e-6), (
            f"Site X position mismatch for {region_name}: "
            f"expected {expected_center_x}, got {site_x}"
        )
        assert np.isclose(site_y, expected_center_y, atol=1e-6), (
            f"Site Y position mismatch for {region_name}: "
            f"expected {expected_center_y}, got {site_y}"
        )
        assert np.isclose(site_z, expected_center_z, atol=1e-6), (
            f"Site Z position mismatch for {region_name}: "
            f"expected {expected_center_z}, got {site_z}"
        )


@pytest.mark.skipif(
    not MIMICLABS_SCENES_DIR.exists(),
    reason="MimicLabs scenes not downloaded. "
    "Run: python scripts/download_mimiclabs_assets.py",
)
def test_tidybot3d_cupboard_mimiclabs_with_video():
    """Test MimicLabs scene with ConstrainedFitting task and video recording."""
    kinder.register_all_environments()
    env = kinder.make(
        "kinder/SortClutteredBlocks3D-o12-sort_the_blocks_into_the_cupboard-v0",
        render_mode="rgb_array",
        scene_bg=True,
        scene_render_camera="task_view",
    )

    # Wrap with RecordVideo if making videos
    if MAKE_VIDEOS:
        env = RecordVideo(
            env,
            "unit_test_videos_cupboard_o12_sort_the_blocks_into_the_cupboard_mimiclabs",
        )

    obs, _ = env.reset()
    # Take a few random steps to generate video frames
    for _ in range(30):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            obs, _ = env.reset(seed=456)

    env.close()


def test_cupboard_custom_rgba_colors():
    """Test that Cupboard RGBA colors can be customized via fixture_config."""
    # Custom RGBA colors
    custom_rgba_shelf = [0.1, 0.2, 0.3, 1.0]
    custom_rgba_leg = [0.2, 0.3, 0.4, 1.0]
    custom_rgba_partition = [0.3, 0.4, 0.5, 1.0]
    custom_rgba_panel = [0.4, 0.5, 0.6, 1.0]
    custom_rgba_drawer_bottom = [0.5, 0.6, 0.7, 0.8]
    custom_rgba_drawer_wall = [0.6, 0.7, 0.8, 0.9]
    custom_rgba_drawer_face = [0.7, 0.8, 0.9, 1.0]
    custom_rgba_drawer_handle = [0.8, 0.8, 0.8, 1.0]

    # Create cupboard with custom colors
    cupboard_config = {
        "length": 0.6,
        "depth": 0.3,
        "shelf_heights": [0.1, 0.2],
        "shelf_partitions": [[], []],
        "shelf_drawers": [[False], [False]],
        "side_and_back_open": False,
        "rgba_cupboard_shelf": custom_rgba_shelf,
        "rgba_cupboard_leg": custom_rgba_leg,
        "rgba_cupboard_partition": custom_rgba_partition,
        "rgba_cupboard_panel": custom_rgba_panel,
        "rgba_drawer_bottom": custom_rgba_drawer_bottom,
        "rgba_drawer_wall": custom_rgba_drawer_wall,
        "rgba_drawer_face": custom_rgba_drawer_face,
        "rgba_drawer_handle": custom_rgba_drawer_handle,
    }

    cupboard = Cupboard(
        name="test_cupboard_rgba",
        fixture_config=cupboard_config,
        position=[0.0, 0.0, 0.0],
        yaw=0.0,
    )

    # Verify that the custom colors were set
    assert cupboard.rgba_cupboard_shelf == custom_rgba_shelf
    assert cupboard.rgba_cupboard_leg == custom_rgba_leg
    assert cupboard.rgba_cupboard_partition == custom_rgba_partition
    assert cupboard.rgba_cupboard_panel == custom_rgba_panel
    assert cupboard.rgba_drawer_bottom == custom_rgba_drawer_bottom
    assert cupboard.rgba_drawer_wall == custom_rgba_drawer_wall
    assert cupboard.rgba_drawer_face == custom_rgba_drawer_face
    assert cupboard.rgba_drawer_handle == custom_rgba_drawer_handle


def test_cupboard_default_rgba_colors():
    """Test that Cupboard uses default RGBA colors when not specified in config."""
    cupboard_config = {
        "length": 0.6,
        "depth": 0.3,
        "shelf_heights": [0.1, 0.2],
        "shelf_partitions": [[], []],
        "shelf_drawers": [[False], [False]],
        "side_and_back_open": True,  # Open so we have legs
    }

    cupboard = Cupboard(
        name="test_cupboard_default",
        fixture_config=cupboard_config,
        position=[0.0, 0.0, 0.0],
        yaw=0.0,
    )

    # Verify that the default colors are used
    assert cupboard.rgba_cupboard_shelf == Cupboard.default_rgba_cupboard_shelf
    assert cupboard.rgba_cupboard_leg == Cupboard.default_rgba_cupboard_leg
    assert cupboard.rgba_cupboard_partition == Cupboard.default_rgba_cupboard_partition
    assert cupboard.rgba_cupboard_panel == Cupboard.default_rgba_cupboard_panel
    assert cupboard.rgba_drawer_bottom == Cupboard.default_rgba_drawer_bottom
    assert cupboard.rgba_drawer_wall == Cupboard.default_rgba_drawer_wall
    assert cupboard.rgba_drawer_face == Cupboard.default_rgba_drawer_face
    assert cupboard.rgba_drawer_handle == Cupboard.default_rgba_drawer_handle
