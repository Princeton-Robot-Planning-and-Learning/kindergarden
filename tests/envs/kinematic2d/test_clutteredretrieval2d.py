"""Tests for clutteredretrieval2d.py."""

from gymnasium.spaces import Box
from gymnasium.wrappers import RecordVideo

import kinder
from kinder.envs.kinematic2d.clutteredretrieval2d import (
    ObjectCentricClutteredRetrieval2DEnv,
    TargetRegionType,
)
from kinder.envs.utils import rectangle_object_to_geom
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
