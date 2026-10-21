"""Tests for placement sampling utilities for dynamic3d environments."""

import numpy as np
import pytest

from kinder.envs.dynamic3d.objects import MujocoGround, Table
from kinder.envs.dynamic3d.placement_samplers import (
    sample_collision_free_position,
)
from kinder.envs.dynamic3d.utils import bboxes_overlap


def create_mock_sampler(x_range=(-2.0, 2.0), y_range=(0.5, 2.5)):
    """Create a mock pos_yaw_sampler function for testing."""

    def sampler(region_name, np_random):
        _ = region_name  # Unused argument
        x = np_random.uniform(x_range[0], x_range[1])
        y = np_random.uniform(y_range[0], y_range[1])
        z = 0.0
        yaw = np_random.uniform(0.0, 2 * np.pi)
        return x, y, z, yaw

    return sampler


# Tests for sample_collision_free_position function


def test_no_existing_tables():
    """Test sampling when no tables are placed yet."""
    np_random = np.random.default_rng(42)
    table_config = {"shape": "rectangle", "length": 0.5, "width": 0.5, "height": 0.8}
    placed_bboxes = []

    # Get initial bounding box at origin
    initial_bbox = Table.get_bounding_box_from_config(
        np.array([0.0, 0.0, 0.0], dtype=np.float32), table_config
    )

    sampler = create_mock_sampler()
    pos, yaw, bbox = sample_collision_free_position(
        initial_bbox, placed_bboxes, np_random, "test_region", sampler
    )

    # Should return a valid position and yaw
    assert isinstance(pos, np.ndarray)
    assert pos.shape == (3,)
    assert pos[2] == 0.0  # z should always be 0
    assert isinstance(yaw, float)
    assert isinstance(bbox, list)
    assert len(bbox) == 6

    # Position should be within default ranges
    assert -2.0 <= pos[0] <= 2.0
    assert 0.5 <= pos[1] <= 2.5


def test_with_existing_tables():
    """Test sampling with existing tables."""
    np_random = np.random.default_rng(42)
    table_config = {"shape": "rectangle", "length": 0.5, "width": 0.5, "height": 0.8}
    # Place a table in the middle of the sampling area
    placed_bboxes = [[0.0, 1.0, 0.0, 1.0, 2.0, 0.8]]

    # Get initial bounding box at origin
    initial_bbox = Table.get_bounding_box_from_config(
        np.array([0.0, 0.0, 0.0], dtype=np.float32), table_config
    )

    sampler = create_mock_sampler()
    _, _, bbox = sample_collision_free_position(
        initial_bbox, placed_bboxes, np_random, "test_region", sampler
    )

    # Check that the sampled position doesn't create an overlapping bbox
    for existing_bbox in placed_bboxes:
        assert not bboxes_overlap(bbox, existing_bbox, margin=0.0)


def test_custom_ranges():
    """Test sampling with custom x and y ranges."""
    np_random = np.random.default_rng(42)
    table_config = {"shape": "rectangle", "length": 0.5, "width": 0.5, "height": 0.8}
    placed_bboxes = []
    x_range = (5.0, 6.0)
    y_range = (10.0, 11.0)

    # Get initial bounding box at origin
    initial_bbox = Table.get_bounding_box_from_config(
        np.array([0.0, 0.0, 0.0], dtype=np.float32), table_config
    )

    sampler = create_mock_sampler(x_range=x_range, y_range=y_range)
    pos, _, bbox = sample_collision_free_position(
        initial_bbox, placed_bboxes, np_random, "test_region", sampler
    )

    assert x_range[0] <= pos[0] <= x_range[1]
    assert y_range[0] <= pos[1] <= y_range[1]
    assert pos[2] == 0.0
    assert isinstance(bbox, list)


def test_deterministic_with_seed():
    """Test that sampling is deterministic with same seed."""
    table_config = {"shape": "rectangle", "length": 0.5, "width": 0.5, "height": 0.8}
    placed_bboxes = []

    # Get initial bounding box at origin
    initial_bbox = Table.get_bounding_box_from_config(
        np.array([0.0, 0.0, 0.0], dtype=np.float32), table_config
    )

    sampler = create_mock_sampler()

    # Sample with first generator
    rng1 = np.random.default_rng(123)
    pos1, yaw1, bbox1 = sample_collision_free_position(
        initial_bbox, placed_bboxes, rng1, "test_region", sampler
    )

    # Sample with second generator with same seed
    rng2 = np.random.default_rng(123)
    pos2, yaw2, bbox2 = sample_collision_free_position(
        initial_bbox, placed_bboxes, rng2, "test_region", sampler
    )

    np.testing.assert_array_equal(pos1, pos2)
    assert yaw1 == yaw2
    assert bbox1 == bbox2


def test_crowded_scenario_fallback():
    """Test fallback behavior when space is very crowded."""
    np_random = np.random.default_rng(42)
    # Create a scenario where it's very hard to find collision-free space
    large_table_config = {
        "shape": "rectangle",
        "length": 3.0,
        "width": 3.0,
        "height": 0.8,
    }
    placed_bboxes = [
        [-2.0, 0.5, 0.0, 0.0, 2.5, 0.8],  # Left side
        [0.0, 0.5, 0.0, 2.0, 2.5, 0.8],  # Right side
    ]

    # Get initial bounding box at origin
    initial_bbox = Table.get_bounding_box_from_config(
        np.array([0.0, 0.0, 0.0], dtype=np.float32), large_table_config
    )

    sampler = create_mock_sampler()
    # Should still return a position (though it might overlap)
    pos, yaw, bbox = sample_collision_free_position(
        initial_bbox, placed_bboxes, np_random, "test_region", sampler, max_attempts=5
    )

    assert isinstance(pos, np.ndarray)
    assert pos.shape == (3,)
    assert pos[2] == 0.0
    assert isinstance(yaw, float)
    assert isinstance(bbox, list)


def test_crowded_scenario_can_fail_instead_of_returning_invalid_fallback():
    """Partial in-episode resets must never silently place an object at origin."""
    np_random = np.random.default_rng(42)
    initial_bbox = Table.get_bounding_box_from_config(
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        {"shape": "rectangle", "length": 3.0, "width": 3.0, "height": 0.8},
    )
    with pytest.raises(RuntimeError, match="Could not find collision-free position"):
        sample_collision_free_position(
            initial_bbox,
            [[-100.0, -100.0, -100.0, 100.0, 100.0, 100.0]],
            np_random,
            "test_region",
            create_mock_sampler(),
            max_attempts=2,
            fail_on_exhaustion=True,
        )


def test_circular_table():
    """Test sampling with circular table configuration."""
    np_random = np.random.default_rng(42)
    circle_config = {"shape": "circle", "diameter": 0.8, "height": 0.8}
    placed_bboxes = []

    # Get initial bounding box at origin
    initial_bbox = Table.get_bounding_box_from_config(
        np.array([0.0, 0.0, 0.0], dtype=np.float32), circle_config
    )

    sampler = create_mock_sampler()
    pos, yaw, bbox = sample_collision_free_position(
        initial_bbox, placed_bboxes, np_random, "test_region", sampler
    )

    assert isinstance(pos, np.ndarray)
    assert pos.shape == (3,)
    assert pos[2] == 0.0
    assert isinstance(yaw, float)
    assert isinstance(bbox, list)


# Tests for MujocoGround.sample_pose_in_region method


def test_single_region():
    """Test sampling from a single region."""
    np_random = np.random.default_rng(42)
    regions = {
        "test_region": {
            "ranges": [[1.0, 2.0, 3.0, 4.0]],  # [x_start, y_start, x_end, y_end]
            "yaw_ranges": [(0.0, 360.0)],  # yaw range for each region
        }
    }
    ground = MujocoGround(regions=regions)

    x, y, z, yaw = ground.sample_pose_in_region("test_region", np_random)

    assert 1.0 <= x <= 3.0
    assert 2.0 <= y <= 4.0
    assert z > 0.0  # z should be above ground
    assert 0.0 <= yaw <= 2 * np.pi  # yaw in radians


def test_multiple_regions():
    """Test sampling from multiple regions."""
    np_random = np.random.default_rng(42)
    regions = {
        "test_region": {
            "ranges": [
                [0.0, 0.0, 1.0, 1.0],  # Region 1
                [5.0, 5.0, 6.0, 6.0],  # Region 2
                [10.0, 10.0, 11.0, 11.0],  # Region 3
            ],
            "yaw_ranges": [
                (0.0, 360.0),
                (0.0, 360.0),
                (0.0, 360.0),
            ],  # yaw range for each region
        }
    }
    ground = MujocoGround(regions=regions)

    # Sample many times to check all regions can be selected
    sampled_regions = set()
    for _ in range(100):
        x, y, z, yaw = ground.sample_pose_in_region("test_region", np_random)

        # Determine which region this sample came from
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            sampled_regions.add(0)
        elif 5.0 <= x <= 6.0 and 5.0 <= y <= 6.0:
            sampled_regions.add(1)
        elif 10.0 <= x <= 11.0 and 10.0 <= y <= 11.0:
            sampled_regions.add(2)

        assert z > 0.0
        assert 0.0 <= yaw <= 2 * np.pi  # yaw in radians

    # Should have sampled from all regions at least once
    assert len(sampled_regions) == 3


def test_ground_sampling_above_surface():
    """Test that ground sampling returns z coordinate above ground surface."""
    np_random = np.random.default_rng(42)
    regions = {
        "test_region": {
            "ranges": [[0.0, 0.0, 1.0, 1.0]],
            "yaw_ranges": [(0.0, 360.0)],  # yaw range for each region
        }
    }
    ground = MujocoGround(regions=regions)

    x, y, z, yaw = ground.sample_pose_in_region("test_region", np_random)

    assert 0.0 <= x <= 1.0
    assert 0.0 <= y <= 1.0
    assert z > 0.0  # Should be above ground
    assert 0.0 <= yaw <= 2 * np.pi  # yaw in radians


def test_deterministic_with_seed_pose():
    """Test that sampling is deterministic with same seed."""
    regions = {
        "test_region": {
            "ranges": [[0.0, 0.0, 1.0, 1.0]],
            "yaw_ranges": [(0.0, 360.0)],  # yaw range for each region
        }
    }
    ground = MujocoGround(regions=regions)

    # Sample with first generator
    rng1 = np.random.default_rng(123)
    pose1 = ground.sample_pose_in_region("test_region", rng1)

    # Sample with second generator with same seed
    rng2 = np.random.default_rng(123)
    pose2 = ground.sample_pose_in_region("test_region", rng2)

    assert pose1 == pose2


def test_empty_regions_dict():
    """Test that empty regions dict raises error when sampling."""
    np_random = np.random.default_rng(42)
    regions = {}
    ground = MujocoGround(regions=regions)

    with pytest.raises((ValueError, AssertionError, KeyError)):
        ground.sample_pose_in_region("nonexistent", np_random)


def test_invalid_region_format():
    """Test that invalid region format raises ValueError."""
    # Region with wrong number of elements
    regions = {
        "test_region": {
            "ranges": [[0.0, 0.0, 1.0]],  # Missing y_end
            "yaw_ranges": None,
        }
    }

    with pytest.raises((ValueError, IndexError)):
        MujocoGround(regions=regions)


def test_invalid_x_bounds():
    """Test that invalid x bounds raise ValueError."""
    np_random = np.random.default_rng(42)
    regions = {
        "test_region": {
            "ranges": [[1.0, 0.0, 0.0, 1.0]],  # x_start > x_end
            "yaw_ranges": None,
        }
    }
    ground = MujocoGround(regions=regions)

    with pytest.raises(ValueError, match="x_start .* must be less than x_end"):
        ground.sample_pose_in_region("test_region", np_random)


def test_invalid_y_bounds():
    """Test that invalid y bounds raise ValueError."""
    np_random = np.random.default_rng(42)
    regions = {
        "test_region": {
            "ranges": [[0.0, 1.0, 1.0, 0.0]],  # y_start > y_end
            "yaw_ranges": None,
        }
    }
    ground = MujocoGround(regions=regions)

    with pytest.raises(ValueError, match="y_start .* must be less than y_end"):
        ground.sample_pose_in_region("test_region", np_random)


def test_equal_bounds():
    """Test that invalid bounds raise ValueError."""
    np_random = np.random.default_rng(42)

    # x_start > x_end (invalid)
    regions = {
        "test_region": {
            "ranges": [[1.5, 0.0, 1.0, 1.0]],
            "yaw_ranges": [(0.0, 360.0)],  # yaw range for each region
        }
    }
    ground = MujocoGround(regions=regions)
    with pytest.raises(ValueError, match="x_start .* must be less than x_end"):
        ground.sample_pose_in_region("test_region", np_random)

    # y_start > y_end (invalid)
    regions = {
        "test_region": {
            "ranges": [[0.0, 1.5, 1.0, 1.0]],
            "yaw_ranges": [(0.0, 360.0)],  # yaw range for each region
        }
    }
    ground = MujocoGround(regions=regions)
    with pytest.raises(ValueError, match="y_start .* must be less than y_end"):
        ground.sample_pose_in_region("test_region", np_random)


def test_point_region():
    """Test sampling from a very small region."""
    np_random = np.random.default_rng(42)
    regions = {
        "test_region": {
            "ranges": [[0.0, 0.0, 0.001, 0.001]],  # Very small region
            "yaw_ranges": [(0.0, 360.0)],  # yaw range for each region
        }
    }
    ground = MujocoGround(regions=regions)

    x, y, z, yaw = ground.sample_pose_in_region("test_region", np_random)

    assert 0.0 <= x <= 0.001
    assert 0.0 <= y <= 0.001
    assert z > 0.0
    assert 0.0 <= yaw <= 2 * np.pi  # yaw in radians


def test_negative_coordinates():
    """Test sampling from regions with negative coordinates."""
    np_random = np.random.default_rng(42)
    regions = {
        "test_region": {
            "ranges": [[-2.0, -3.0, -1.0, -1.0]],
            "yaw_ranges": [(0.0, 360.0)],  # yaw range for each region
        }
    }
    ground = MujocoGround(regions=regions)

    x, y, z, yaw = ground.sample_pose_in_region("test_region", np_random)

    assert -2.0 <= x <= -1.0
    assert -3.0 <= y <= -1.0
    assert z > 0.0
    assert 0.0 <= yaw <= 2 * np.pi  # yaw in radians
