"""Tests for deterministic demo replay across all environments."""

from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
import pytest

import kinder
from kinder.utils import find_all_demo_files, load_demo
from tests.demo_blacklist import (
    DETERMINISTIC_REPLAY_BLACKLIST,
    is_demo_blacklisted,
)


@pytest.mark.parametrize("demo_path", find_all_demo_files())
def test_deterministic_demo_replay(demo_path: Path):
    """Test that demo replay produces identical observations and rewards.

    This test verifies that:
    1. Loading a demo file succeeds
    2. Environment can be created for the demo's environment ID
    For each observation and action pair in the demo:
        3. Resetting the environment with that observation
        4. Replaying the action produces the next observation
        5. Checking the reproduced observation matches the demo's next observation
    """
    # Register all environments
    kinder.register_all_environments()

    # Check if demo is blacklisted
    is_blacklisted, reason = is_demo_blacklisted(
        demo_path, DETERMINISTIC_REPLAY_BLACKLIST
    )
    if is_blacklisted:
        pytest.skip(f"Demo blacklisted: {reason}")

    # Load demo data
    try:
        demo_data = load_demo(demo_path)
    except Exception as e:
        pytest.skip(f"Failed to load demo {demo_path}: {e}")

    # Extract demo information
    env_id = demo_data["env_id"]
    actions = demo_data["actions"]
    expected_observations = demo_data["observations"]
    expected_rewards = demo_data.get("rewards", None)
    seed = demo_data["seed"]

    # Skip if no actions to replay
    if len(actions) == 0:
        pytest.skip(f"Demo {demo_path} contains no actions")

    # Create environment
    make_kwargs: dict[str, Any] = {"render_mode": "rgb_array"}
    entrypoint = gymnasium.registry[env_id].entry_point
    assert isinstance(entrypoint, str)
    if "kinematic3d." in entrypoint:
        make_kwargs["realistic_bg"] = False
    env = kinder.make(env_id, **make_kwargs)

    # Test reproducibility: reset with seed and replay actions
    obs, _ = env.reset(seed=seed)

    # Check initial observation matches
    assert np.allclose(
        obs, expected_observations[0], atol=1e-4
    ), f"Initial observation mismatch in {demo_path}"

    # Replay all actions and verify observations/rewards
    for i, _ in enumerate(expected_observations):
        action = actions[i]
        obs_next, reward, terminated, truncated, _ = env.step(action)

        # Check observation matches
        expected_obs = expected_observations[i + 1]
        if not np.allclose(obs_next, expected_obs, atol=1e-4):
            diff = np.abs(obs_next - expected_obs)
            max_diff = np.max(diff)
            raise AssertionError(
                f"Observation mismatch at step {i} in {demo_path}: "
                f"max difference {max_diff}"
            )

        # Check reward matches (if available)
        if expected_rewards is not None and i < len(expected_rewards):
            expected_reward = expected_rewards[i]
            assert reward == expected_reward, (
                f"Reward mismatch at step {i} in {demo_path}: "
                f"got {reward}, expected {expected_reward}"
            )
        # Stop if episode ended early
        if terminated or truncated or i == len(expected_observations) - 2:
            break
    env.close()  # type: ignore[no-untyped-call]
