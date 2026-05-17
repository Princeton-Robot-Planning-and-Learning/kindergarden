"""Tests for stickbutton2d.py."""

from gymnasium.spaces import Box
from gymnasium.wrappers import RecordVideo

import kinder
from kinder.envs.kinematic2d.object_types import CircleType
from kinder.envs.kinematic2d.stickbutton2d import ObjectCentricStickButton2DEnv
from tests.conftest import MAKE_VIDEOS


def test_object_centric_stickbutton2d_env():
    """Tests for ObjectCentricMotion2DEnv()."""
    # Test env creation and random actions.
    env = ObjectCentricStickButton2DEnv(num_buttons=5)

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    env.reset(seed=123)
    env.action_space.seed(123)
    for _ in range(10):
        action = env.action_space.sample()
        env.step(action)
    env.close()


def test_stickbutton2d_observation_space():
    """Tests that observations are vectors with fixed dimensionality."""
    kinder.register_all_environments()
    env = kinder.make("kinder/StickButton2D-b5-v0")
    assert isinstance(env.observation_space, Box)
    for _ in range(5):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)


def test_stickbutton2d_action_space():
    """Tests that actions are vectors with fixed dimensionality."""
    kinder.register_all_environments()
    env = kinder.make("kinder/StickButton2D-b5-v0")
    assert isinstance(env.action_space, Box)
    env.reset()
    for _ in range(5):
        action = env.action_space.sample()
        assert env.action_space.contains(action)
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    env.close()


def test_stickbutton2d_termination():
    """Tests that the environment terminates when all buttons are pressed."""
    env = ObjectCentricStickButton2DEnv(num_buttons=5)
    state, _ = env.reset()

    # Manually press all buttons.
    buttons = state.get_objects(CircleType)
    for button in buttons:
        state = env.press_button(button)
    env.reset(options={"init_state": state})

    # Any action should now result in termination.
    action = env.action_space.sample()
    state, reward, terminated, _, _ = env.step(action)
    assert reward == 0.0
    assert terminated
