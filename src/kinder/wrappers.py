"""Gymnasium wrappers for KinDER environments."""

from typing import Any

import gymnasium
import numpy as np
from gymnasium.spaces import Box
from numpy.typing import NDArray


class NoisyObservation(gymnasium.ObservationWrapper):
    """Adds Gaussian noise to vectorized observations.

    Designed for ConstantObjectKinDEREnv, which produces fixed-size float arrays
    (positions, angles, sizes, etc.).
    """

    def __init__(
        self,
        env: gymnasium.Env,
        noise_std: float | NDArray[np.floating] = 0.01,
    ) -> None:
        """Initialize.

        Args:
            env: The environment to wrap. Must have a Box observation space.
            noise_std: Standard deviation of the Gaussian noise. Can be a
                scalar (same noise for all dimensions) or a per-dimension
                array matching the observation shape.
        """
        super().__init__(env)
        assert isinstance(
            env.observation_space, Box
        ), f"Expected Box observation space, got {type(env.observation_space)}"
        if isinstance(noise_std, np.ndarray):
            assert noise_std.shape == env.observation_space.shape, (
                f"noise_std shape {noise_std.shape} does not match "
                f"observation space shape {env.observation_space.shape}"
            )
        self._noise_std = noise_std

    def observation(self, observation: NDArray[Any]) -> NDArray[Any]:
        """Add Gaussian noise to the observation."""
        noise = self.np_random.normal(
            loc=0.0, scale=self._noise_std, size=observation.shape
        )
        noisy_obs = observation + noise
        assert isinstance(self.observation_space, Box)
        noisy_obs = np.clip(
            noisy_obs, self.observation_space.low, self.observation_space.high
        )
        return noisy_obs.astype(observation.dtype)


class NoisyAction(gymnasium.ActionWrapper):
    """Adds Gaussian noise to continuous actions and clips to action space bounds.

    Designed for ConstantObjectKinDEREnv, which has continuous RobotActionSpace (a Box
    space with finite bounds).
    """

    def __init__(
        self,
        env: gymnasium.Env,
        noise_std: float | NDArray[np.floating] = 0.01,
    ) -> None:
        """Initialize.

        Args:
            env: The environment to wrap. Must have a Box action space.
            noise_std: Standard deviation of the Gaussian noise. Can be a
                scalar (same noise for all dimensions) or a per-dimension
                array matching the action shape.
        """
        assert isinstance(
            env.action_space, Box
        ), f"Expected Box action space, got {type(env.action_space)}"
        self._low = env.action_space.low.copy()
        self._high = env.action_space.high.copy()
        if isinstance(noise_std, np.ndarray):
            assert noise_std.shape == env.action_space.shape, (
                f"noise_std shape {noise_std.shape} does not match "
                f"action space shape {env.action_space.shape}"
            )
        self._noise_std = noise_std
        super().__init__(env)

    def action(self, action: NDArray[Any]) -> NDArray[Any]:
        """Add Gaussian noise to the action and clip to bounds."""
        noise = self.np_random.normal(loc=0.0, scale=self._noise_std, size=action.shape)
        noisy_action = action + noise
        clipped = np.clip(noisy_action, self._low, self._high)
        return clipped.astype(action.dtype)
