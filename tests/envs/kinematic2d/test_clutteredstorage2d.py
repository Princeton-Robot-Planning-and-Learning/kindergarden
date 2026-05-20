"""Tests for clutteredstoragel2d.py."""

import numpy as np
from gymnasium.spaces import Box
from gymnasium.wrappers import RecordVideo

import kinder
from kinder.envs.kinematic2d.clutteredstorage2d import (
    ObjectCentricClutteredStorage2DEnv,
    ShelfType,
    TargetBlockType,
)
from tests.conftest import MAKE_VIDEOS


def test_object_centric_clutteredstorage2d_env():
    """Tests for ObjectCentricClutteredStorage2DEnv()."""
    # Test env creation and random actions.
    env = ObjectCentricClutteredStorage2DEnv(num_blocks=1)

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    env.reset(seed=123)
    env.action_space.seed(123)
    for _ in range(10):
        action = env.action_space.sample()
        env.step(action)
    env.close()


def test_clutteredstorage2d_observation_space():
    """Tests that observations are vectors with fixed dimensionality."""
    kinder.register_all_environments()
    env = kinder.make("kinder/ClutteredStorage2D-b7-v0")
    assert isinstance(env.observation_space, Box)
    for _ in range(5):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)


def test_clutteredstorage2d_termination():
    """Tests that the environment terminates when all blocks are on the shelf."""
    env = ObjectCentricClutteredStorage2DEnv(num_blocks=3)
    state, _ = env.reset(seed=0)
    # Manually move the block into the shelf.
    shelf = state.get_objects(ShelfType)[0]
    blocks = state.get_objects(TargetBlockType)
    for block in blocks:
        # Move the block to the shelf.
        # The x, y positions are set to the (x1, y1) of the shelf
        state.set(block, "x", state.get(shelf, "x1"))
        state.set(block, "y", state.get(shelf, "y1"))
        state.set(block, "theta", 0.0)

    env.reset(options={"init_state": state})
    # Any action should now result in termination.
    action = env.action_space.sample()
    state, reward, terminated, _, _ = env.step(action)
    assert reward == -1.0
    assert terminated


def test_clutteredstorage2d_move_block():
    """Tests that the robot can attach and move one block."""
    kinder.register_all_environments()

    env = ObjectCentricClutteredStorage2DEnv(num_blocks=1)

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    obs, _ = env.reset()
    obj_name_to_obj = {o.name: o for o in obs}
    block0 = obj_name_to_obj["block0"]
    robot = obj_name_to_obj["robot"]
    # Unpack initial obs

    obs.set(robot, "x", 1.9)
    obs.set(robot, "y", 0.5)
    obs.set(robot, "theta", 0.0)

    obs.set(block0, "x", 1.3)
    obs.set(block0, "y", 0.5)
    obs.set(block0, "theta", np.pi / 3)

    robot_theta = obs.get(robot, "theta")
    block_theta0 = obs.get(block0, "theta")

    options = {
        "init_state": obs,
    }
    obs, _ = env.reset(options=options)

    # Action limits
    dtheta_max, darm_max = env.action_space.high[2], env.action_space.high[3]

    # 1) Rotate base to face the block
    desired_theta = block_theta0 + np.pi / 2
    if desired_theta >= np.pi:
        desired_theta -= np.pi
    diff = desired_theta - robot_theta
    # Rotate in chunks
    while abs(diff) > 1e-3:
        step = np.clip(diff, -dtheta_max, dtheta_max)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[2] = step
        obs, _, _, _, _ = env.step(action)
        robot_theta = obs.get(robot, "theta")
        diff = desired_theta - robot_theta

    # 2) Stretch arm to reach the block
    desired_arm = 0.5
    curr_arm = obs.get(robot, "arm_joint")
    while abs(curr_arm - desired_arm) > 1e-3:
        step = np.clip(desired_arm - curr_arm, -darm_max, darm_max)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[3] = step
        obs, _, _, _, _ = env.step(action)
        curr_arm = obs.get(robot, "arm_joint")

    # 3) Attach the block
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[4] = 1.0
    obs, _, _, _, _ = env.step(action)

    # 4) Move the block
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    curr_block_x = obs.get(block0, "x")
    action[0] = 0.05
    action[4] = 1.0
    obs, _, _, _, _ = env.step(action)
    new_block_x = obs.get(block0, "x")
    assert abs(new_block_x - 0.05 - curr_block_x) < 1e-3

    env.close()
