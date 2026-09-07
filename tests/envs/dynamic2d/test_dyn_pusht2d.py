"""Tests for dyn_pusht2d.py."""

import numpy as np
from gymnasium.spaces import Box
from tomsgeoms2d.structs import Tobject

import kinder
from kinder.envs.dynamic2d.dyn_pusht2d import DynPushT2DEnvConfig


def test_dyn_pusht2d_observation_space():
    """Tests that observations are vectors with fixed dimensionality."""
    kinder.register_all_environments()
    env = kinder.make("kinder/DynPushT2D-t1-v0")
    assert isinstance(env.observation_space, Box)
    for _ in range(5):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)


def test_dyn_pusht2d_action_space():
    """Tests that the actions are valid and the step function works."""
    kinder.register_all_environments()
    env = kinder.make("kinder/DynPushT2D-t1-v0")
    obs, _ = env.reset(seed=0)

    # Test that robot moves with delta actions
    for s in range(3):
        obs, _ = env.reset(seed=s)
        state = env.observation_space.devectorize(obs)
        name_to_object = {obj.name: obj for obj in state.data}
        robot_object = name_to_object["robot"]
        robot_x = state.get(robot_object, "x")
        robot_y = state.get(robot_object, "y")

        # Command robot to move with delta action
        # NOTE: Dynamic2D env has float64 action space
        delta_action = np.array([0.05, 0.05], dtype=np.float64)

        # After one step, robot should move in positive x and y
        obs_, _, _, _, _ = env.step(delta_action)
        state_ = env.observation_space.devectorize(obs_)
        robot_x_ = state_.get(robot_object, "x")
        robot_y_ = state_.get(robot_object, "y")

        # Robot should have moved in positive x and y directions
        assert np.isclose(robot_x_, robot_x + 0.05, atol=1e-5)
        assert np.isclose(robot_y_, robot_y + 0.05, atol=1e-5)


def test_dyn_pusht2d_random_actions():
    """Tests that observations are valid with random actions."""
    kinder.register_all_environments()
    env = kinder.make("kinder/DynPushT2D-t1-v0")
    assert isinstance(env.observation_space, Box)
    for _ in range(3):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            assert env.observation_space.contains(obs)
            assert isinstance(reward, (int, float))
            if terminated or truncated:
                break
    env.close()


def test_dyn_pusht2d_goal_achievement():
    """Tests that the goal can be achieved by moving the robot to the goal."""
    kinder.register_all_environments()
    env = kinder.make("kinder/DynPushT2D-t1-v0")
    obs, _ = env.reset(seed=42)
    state = env.observation_space.devectorize(obs)
    name_to_object = {obj.name: obj for obj in state.data}
    tblock_object = name_to_object["tblock"]
    goal_tblock_object = name_to_object["goal_tblock"]

    zero_action = np.array([0.0, 0.0], dtype=np.float32)
    _, _, terminated, _, _ = env.step(zero_action)
    assert not terminated

    # Move tblock to goal position
    new_state = state.copy()
    new_state.set(tblock_object, "x", state.get(goal_tblock_object, "x"))
    new_state.set(tblock_object, "y", state.get(goal_tblock_object, "y"))
    new_state.set(tblock_object, "theta", state.get(goal_tblock_object, "theta"))
    obs, _ = env.reset(options={"init_state": new_state})
    _, _, terminated, _, _ = env.step(zero_action)
    assert terminated


def test_dyn_pusht2d_goal_inside_world():
    """Tests that the sampled goal pose keeps the whole T-block inside the walls."""
    kinder.register_all_environments()
    env = kinder.make("kinder/DynPushT2D-t1-v0")
    config = DynPushT2DEnvConfig()
    for seed in range(200):
        obs, _ = env.reset(seed=seed)
        state = env.observation_space.devectorize(obs)
        name_to_object = {obj.name: obj for obj in state.data}
        goal = name_to_object["goal_tblock"]
        geom = Tobject(
            x=state.get(goal, "x"),
            y=state.get(goal, "y"),
            width=state.get(goal, "width"),
            length_horizontal=state.get(goal, "length_horizontal"),
            length_vertical=state.get(goal, "length_vertical"),
            theta=state.get(goal, "theta"),
        )
        xs, ys = zip(*geom.vertices)
        assert min(xs) >= config.world_min_x and max(xs) <= config.world_max_x, seed
        assert min(ys) >= config.world_min_y and max(ys) <= config.world_max_y, seed
    env.close()
