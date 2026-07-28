"""Tests for realistic background loading in Kinematic3D environments."""

import numpy as np
import pytest
from gymnasium.wrappers import RecordVideo

import kinder
from tests.conftest import MAKE_VIDEOS


@pytest.fixture(scope="module", autouse=True)
def register_envs():
    """Register all environments before running tests."""
    kinder.register_all_environments()


def test_realistic_bg_loads_without_error():
    """Test that realistic_bg=True loads the background without errors."""
    env = kinder.make(
        "kinder/BaseMotion3D-v0",
        render_mode="rgb_array",
        use_gui=False,
        realistic_bg=True,
    )
    try:
        obs, _ = env.reset(seed=123)
        assert isinstance(obs, np.ndarray)

        # Take a few steps to make sure the environment works normally
        for _ in range(5):
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)
            assert isinstance(obs, np.ndarray)

        # Render to make sure the background is visible (no crash)
        img = env.render()
        assert img is not None
        assert isinstance(img, np.ndarray)
        assert len(img.shape) == 3  # Height x Width x Channels
    finally:
        env.close()


def test_realistic_bg_creates_body():
    """Test that realistic_bg=True creates a body in PyBullet."""
    env = kinder.make(
        "kinder/BaseMotion3D-v0",
        render_mode="rgb_array",
        use_gui=False,
        realistic_bg=True,
    )
    try:
        env.reset(seed=123)
        # Access the underlying object-centric env to check the background id
        oc_env = env.unwrapped._object_centric_env  # pylint: disable=protected-access
        assert oc_env._realistic_bg_id is not None  # pylint: disable=protected-access
        assert oc_env._realistic_bg_enabled  # pylint: disable=protected-access
        # The body ID should be a non-negative integer
        assert oc_env._realistic_bg_id >= 0  # pylint: disable=protected-access
    finally:
        env.close()


def test_realistic_bg_with_table3d():
    """Test realistic_bg with Table3D environment."""
    env = kinder.make(
        "kinder/Table3D-o1-v0",
        render_mode="rgb_array",
        use_gui=False,
        realistic_bg=True,
    )
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")
    try:
        obs, _ = env.reset(seed=456)
        assert isinstance(obs, np.ndarray)

        # Take a step
        for _ in range(10):
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)
            assert isinstance(obs, np.ndarray)
    finally:
        env.close()
