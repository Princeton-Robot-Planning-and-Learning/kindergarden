"""Unit tests for kinder.envs.dynamic3d.tidybot_rewards.

This test suite verifies the base behavior of TidyBotRewardCalculator and the
create_reward_calculator factory function, including reward calculation, episode step
tracking, and default termination logic.
"""

from kinder.envs.dynamic3d import tidybot_rewards


def test_reward_calculator_base_behavior():
    """Test the base TidyBotRewardCalculator: reward, episode_step increment, and
    termination logic.

    - Checks that calculate_reward returns the base reward (-0.01)
    when no task reward is present.
    - Verifies that episode_step increments with each call.
    - Ensures is_terminated returns False by default.
    """
    calc = tidybot_rewards.TidyBotRewardCalculator(scene_type="ground", num_objects=3)
    obs = {}
    # First call: should increment episode_step and return base reward
    reward = calc.calculate_reward(obs)
    assert reward == -1.0, "Base reward should be -1.0 when no task reward"
    assert calc.episode_step == 1
    # Second call: episode_step increments again
    reward2 = calc.calculate_reward(obs)
    assert reward2 == -1.0
    assert calc.episode_step == 2
    # is_terminated should be False by default
    assert not calc.is_terminated(obs)


def test_create_reward_calculator_factory():
    """Test the create_reward_calculator factory function.

    - Ensures the returned object is a TidyBotRewardCalculator.
    - Checks that scene_type and num_objects are set correctly.
    """
    calc = tidybot_rewards.create_reward_calculator("ground", 2)
    assert isinstance(calc, tidybot_rewards.TidyBotRewardCalculator)
    assert calc.scene_type == "ground"
    assert calc.num_objects == 2
