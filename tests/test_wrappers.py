"""Tests for NoisyObservation and NoisyAction wrappers."""

import numpy as np
import pytest

import kinder
from kinder.wrappers import NoisyAction, NoisyObservation

ENV_ID = "kinder/Obstruction2D-o0-v0"
SEED = 42


@pytest.fixture(autouse=True)
def _register():
    kinder.register_all_environments()


def _make_env(**kwargs):
    return kinder.make(ENV_ID, render_mode="rgb_array", **kwargs)


class TestNoisyObservation:
    """Tests for the NoisyObservation wrapper."""

    def test_observation_shape_preserved(self):
        """Observation shape is unchanged after wrapping."""
        env = NoisyObservation(_make_env(), noise_std=0.05)
        obs, _ = env.reset(seed=SEED)
        assert obs.shape == env.observation_space.shape
        obs2, _, _, _, _ = env.step(env.action_space.sample())
        assert obs2.shape == env.observation_space.shape
        env.close()

    def test_noise_is_applied(self):
        """Noisy obs should differ from clean obs (with high probability)."""
        clean_env = _make_env()
        noisy_env = NoisyObservation(_make_env(), noise_std=0.05)
        clean_obs, _ = clean_env.reset(seed=SEED)
        noisy_obs, _ = noisy_env.reset(seed=SEED)
        assert not np.allclose(clean_obs, noisy_obs)
        clean_env.close()
        noisy_env.close()

    def test_zero_noise_matches_clean(self):
        """Zero noise produces observations identical to the unwrapped env."""
        clean_env = _make_env()
        noisy_env = NoisyObservation(_make_env(), noise_std=0.0)
        clean_obs, _ = clean_env.reset(seed=SEED)
        noisy_obs, _ = noisy_env.reset(seed=SEED)
        np.testing.assert_array_equal(clean_obs, noisy_obs)
        clean_env.close()
        noisy_env.close()

    def test_per_dimension_noise_std(self):
        """Per-dimension noise_std array is accepted and works."""
        env = _make_env()
        obs, _ = env.reset(seed=SEED)
        noise_std = np.full(env.observation_space.shape, 0.01)
        env.close()

        wrapped = NoisyObservation(_make_env(), noise_std=noise_std)
        obs, _ = wrapped.reset(seed=SEED)
        assert obs.shape == wrapped.observation_space.shape
        wrapped.close()

    def test_mismatched_noise_std_shape_raises(self):
        """Mismatched noise_std array shape raises an error."""
        env = _make_env()
        wrong_shape = np.ones(3)
        with pytest.raises(AssertionError, match="noise_std shape"):
            NoisyObservation(env, noise_std=wrong_shape)
        env.close()


class TestNoisyAction:
    """Tests for the NoisyAction wrapper."""

    def test_action_stays_within_bounds(self):
        """Noisy actions are always clipped to action space bounds."""
        env = NoisyAction(_make_env(), noise_std=10.0)  # very high noise
        env.reset(seed=SEED)
        for _ in range(10):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                env.reset()
        env.close()

    def test_noise_is_applied_to_action(self):
        """With high noise, the executed action should differ from the input."""
        env = NoisyAction(_make_env(), noise_std=1.0)
        env.reset(seed=SEED)
        # Use a mid-range action so noise has room to perturb
        low = env.unwrapped.action_space.low  # type: ignore[union-attr]
        high = env.unwrapped.action_space.high  # type: ignore[union-attr]
        mid_action = (low + high) / 2.0
        # Step and check that the wrapper's action method perturbs
        noisy = env.action(mid_action.copy())
        assert not np.allclose(mid_action, noisy)
        env.close()

    def test_zero_noise_preserves_action(self):
        """Zero noise passes the action through unchanged."""
        env = NoisyAction(_make_env(), noise_std=0.0)
        env.reset(seed=SEED)
        action = env.action_space.sample()
        result = env.action(action.copy())
        np.testing.assert_array_equal(action, result)
        env.close()

    def test_per_dimension_noise_std(self):
        """Per-dimension noise_std array is accepted and works."""
        env = _make_env()
        noise_std = np.full(env.action_space.shape, 0.01)
        wrapped = NoisyAction(env, noise_std=noise_std)
        wrapped.reset(seed=SEED)
        wrapped.step(wrapped.action_space.sample())
        wrapped.close()

    def test_mismatched_noise_std_shape_raises(self):
        """Mismatched noise_std array shape raises an error."""
        env = _make_env()
        wrong_shape = np.ones(3)
        with pytest.raises(AssertionError, match="noise_std shape"):
            NoisyAction(env, noise_std=wrong_shape)
        env.close()


class TestComposition:
    """Tests for composing both wrappers together."""

    def test_both_wrappers_together(self):
        """NoisyObservation and NoisyAction can be stacked."""
        env = _make_env()
        env = NoisyObservation(env, noise_std=0.01)
        env = NoisyAction(env, noise_std=0.01)
        obs, _ = env.reset(seed=SEED)
        assert obs.shape == env.observation_space.shape
        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert obs.shape == env.observation_space.shape
        env.close()
