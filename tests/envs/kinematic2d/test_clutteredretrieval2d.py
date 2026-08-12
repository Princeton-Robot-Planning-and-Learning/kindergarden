"""Tests for clutteredretrieval2d.py."""

from gymnasium.spaces import Box
from gymnasium.wrappers import RecordVideo
from tomsgeoms2d.utils import geom2ds_intersect

import kinder
from kinder.envs.kinematic2d.clutteredretrieval2d import (
    ObjectCentricClutteredRetrieval2DEnv,
    TargetBlockType,
    TargetRegionType,
)
from kinder.envs.kinematic2d.object_types import CRVRobotType
from kinder.envs.kinematic2d.structs import SE2Pose
from kinder.envs.utils import object_to_multibody2d, rectangle_object_to_geom
from tests.conftest import MAKE_VIDEOS


def test_object_centric_clutteredretrieval2d_env():
    """Tests for ObjectCentricClutteredRetrieval2DEnv()."""
    # Test env creation and random actions.
    env = ObjectCentricClutteredRetrieval2DEnv(num_obstructions=25)

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    env.reset(seed=123)
    env.action_space.seed(123)
    for _ in range(10):
        action = env.action_space.sample()
        env.step(action)
    env.close()


def test_clutteredretrieval2d_observation_space():
    """Tests that observations are vectors with fixed dimensionality."""
    kinder.register_all_environments()
    env = kinder.make("kinder/ClutteredRetrieval2D-o10-v0")
    assert isinstance(env.observation_space, Box)
    for _ in range(5):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)


def test_clutteredretrieval2d_target_region_within_world_bounds():
    """Tests that the full rotated target region is sampled within the world."""
    env = ObjectCentricClutteredRetrieval2DEnv(num_obstructions=0)
    state, _ = env.reset(seed=5)
    target_region = state.get_objects(TargetRegionType)[0]
    target_region_geom = rectangle_object_to_geom(state, target_region, {})
    for x, y in target_region_geom.vertices:
        assert env.config.world_min_x <= x <= env.config.world_max_x
        assert env.config.world_min_y <= y <= env.config.world_max_y


def test_clutteredretrieval2d_target_starts_outside_region():
    """Tests that the target does not initially intersect its goal region."""
    env = ObjectCentricClutteredRetrieval2DEnv(num_obstructions=0)
    state, _ = env.reset(seed=54)
    target_block = state.get_objects(TargetBlockType)[0]
    target_region = state.get_objects(TargetRegionType)[0]
    target_block_geom = rectangle_object_to_geom(state, target_block, {})
    target_region_geom = rectangle_object_to_geom(state, target_region, {})
    assert not geom2ds_intersect(target_block_geom, target_region_geom)


def test_clutteredretrieval2d_free_space_check_flags_walled_in_robot():
    """Tests that the free-space check rejects a robot ringed by obstructions."""
    env = ObjectCentricClutteredRetrieval2DEnv(num_obstructions=0)
    env.reset(seed=0)
    # Four bars forming a closed box around the robot at the world center.
    ring = [
        (SE2Pose(1.0, 1.0, 0.0), (0.5, 0.05)),
        (SE2Pose(1.0, 1.45, 0.0), (0.5, 0.05)),
        (SE2Pose(1.0, 1.05, 0.0), (0.05, 0.4)),
        (SE2Pose(1.45, 1.05, 0.0), (0.05, 0.4)),
    ]
    state = env._create_initial_state(  # pylint: disable=protected-access
        SE2Pose(1.25, 1.25, 0.0),
        target_pose=SE2Pose(2.0, 2.0, 0.0),
        target_region_pose=SE2Pose(0.3, 0.3, 0.0),
        obstructions=ring,
    )
    check = env._initial_state_is_valid  # pylint: disable=protected-access
    assert not check(state)
    # The same layout without the ring leaves the robot in the main free space.
    state = env._create_initial_state(  # pylint: disable=protected-access
        SE2Pose(1.25, 1.25, 0.0),
        target_pose=SE2Pose(2.0, 2.0, 0.0),
        target_region_pose=SE2Pose(0.3, 0.3, 0.0),
    )
    assert check(state)


def test_clutteredretrieval2d_rejects_larger_enclosed_free_space():
    """Tests that an enclosed component is rejected even when it is the largest."""
    env = ObjectCentricClutteredRetrieval2DEnv(num_obstructions=0)
    env.reset(seed=0)
    # These bars enclose most of the world while leaving a smaller free-space
    # component around the outside of the ring.
    ring = [
        (SE2Pose(0.25, 0.25, 0.0), (2.0, 0.05)),
        (SE2Pose(0.25, 2.2, 0.0), (2.0, 0.05)),
        (SE2Pose(0.25, 0.3, 0.0), (0.05, 1.9)),
        (SE2Pose(2.2, 0.3, 0.0), (0.05, 1.9)),
    ]
    state = env._create_initial_state(  # pylint: disable=protected-access
        SE2Pose(1.25, 1.25, 0.0),
        target_pose=SE2Pose(1.8, 1.8, 0.0),
        target_region_pose=SE2Pose(0.3, 0.3, 0.0),
        obstructions=ring,
    )
    check = env._initial_state_is_valid  # pylint: disable=protected-access
    assert not check(state)


def test_clutteredretrieval2d_robot_spawns_in_main_free_space():
    """Tests that dense initial states never wall in the robot.

    Both seeds draw a layout with the robot inside the obstruction cluster on the first
    attempt, exercising the resampling loop.
    """
    env = ObjectCentricClutteredRetrieval2DEnv(num_obstructions=25)
    check = env._initial_state_is_valid  # pylint: disable=protected-access
    for seed in (48563862895646264, 1985872226597826804):
        state, _ = env.reset(seed=seed)
        assert check(state)


def test_clutteredretrieval2d_target_region_does_not_overlap_robot():
    """Tests that the target region does not initially overlap the robot."""
    env = ObjectCentricClutteredRetrieval2DEnv(num_obstructions=0)
    state, _ = env.reset(seed=91)
    robot = state.get_objects(CRVRobotType)[0]
    target_region = state.get_objects(TargetRegionType)[0]
    robot_multibody = object_to_multibody2d(robot, state, {})
    target_region_geom = rectangle_object_to_geom(state, target_region, {})
    assert not any(
        geom2ds_intersect(target_region_geom, body.geom)
        for body in robot_multibody.bodies
    )
