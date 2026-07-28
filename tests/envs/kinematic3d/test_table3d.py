"""Tests for table3d.py."""

from typing import Any

import numpy as np
import pytest
from gymnasium.wrappers import RecordVideo
from prpl_utils.utils import wrap_angle
from pybullet_helpers.geometry import Pose, SE2Pose
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    remap_joint_position_plan_to_constant_distance,
    run_single_arm_mobile_base_motion_planning,
    smoothly_follow_end_effector_path,
)
from relational_structs.spaces import ObjectCentricBoxSpace

from kinder.envs.kinematic3d.save_utils import DEFAULT_DEMOS_DIR, save_demo
from kinder.envs.kinematic3d.table3d import (
    ObjectCentricTable3DEnv,
    Table3DEnv,
    Table3DObjectCentricState,
)
from tests.conftest import MAKE_VIDEOS

# Flag to enable trajectory saving (can be controlled like MAKE_VIDEOS)
SAVE_TRAJECTORIES = MAKE_VIDEOS


@pytest.fixture(scope="module")
def env():
    """Create a shared environment for all tests in this module."""
    environment = Table3DEnv(
        num_cubes=2, use_gui=False, render_mode="rgb_array", realistic_bg=False
    )
    if MAKE_VIDEOS:
        environment = RecordVideo(environment, "unit_test_videos")
    yield environment
    environment.close()


def test_base_table3d_env(env):  # pylint: disable=redefined-outer-name
    """Tests for basic methods in base table3D env."""
    obs, _ = env.reset(seed=123)
    assert isinstance(obs, np.ndarray)

    for _ in range(10):
        act = env.action_space.sample()
        assert isinstance(act, np.ndarray)
        obs, _, _, _, _ = env.step(act)

    # Uncomment to debug.
    # import pybullet as p

    # while True:
    #     # p.getMouseEvents(env.unwrapped._object_centric_env.physics_client_id)
    #     p.stepSimulation(env.unwrapped._object_centric_env.physics_client_id)


def test_goal_reached_after_pick(env) -> None:  # pylint: disable=redefined-outer-name
    """Test moving in front of a block, picking it up, and lifting it clear of the
    table.

    The goal should be reached once the grasped cube is lifted above the table by more
    than the configured threshold.
    """
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    config = (
        env.unwrapped._object_centric_env.config  # pylint: disable=protected-access
    )

    seed = 123
    vec_obs, _ = env.reset(seed=seed)
    oc_obs = env.observation_space.devectorize(vec_obs)
    obs = Table3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Initialize trajectory collection
    traj_observations: list[Any] = [vec_obs.copy()]
    traj_actions: list[Any] = []
    traj_rewards: list[float] = []
    ep_terminated = False
    ep_truncated = False

    num_cubes = 2
    # Create a simulator for planning.
    sim = ObjectCentricTable3DEnv(
        num_cubes=num_cubes,
        config=config,
        use_gui=False,
        realistic_bg=False,
        allow_state_access=True,
    )
    sim.set_state(obs)

    if MAKE_VIDEOS:
        max_candidate_plans = 20
    else:
        max_candidate_plans = 1

    # Step 1: Move the base in front of cube1
    target_object_pose_temp = obs.get_object_pose("cube1").to_se2()
    target_object_pose = SE2Pose(
        target_object_pose_temp.x - 0.4,
        target_object_pose_temp.y,
        target_object_pose_temp.rot,
    )
    base_plan = run_single_arm_mobile_base_motion_planning(
        sim.robot,
        sim.robot.base.get_pose(),
        target_object_pose,
        collision_bodies=set(),
        seed=123,
    )
    assert base_plan is not None

    for target_base_pose in base_plan[1:]:
        current_base_pose = obs.base_pose
        base_pose_delta = target_base_pose - current_base_pose
        delta_lst = [base_pose_delta.x, base_pose_delta.y, base_pose_delta.rot]
        action_lst = delta_lst + [0.0] * 7 + [0.0]
        action = np.array(action_lst, dtype=np.float32)
        vec_obs, reward, terminated, truncated, _ = env.step(action)
        # Collect trajectory data
        traj_observations.append(vec_obs.copy())
        traj_actions.append(action.copy())
        traj_rewards.append(float(reward))
        ep_terminated = ep_terminated or terminated
        ep_truncated = ep_truncated or truncated
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = Table3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Step 2: Move arm to pre-grasp pose and then to grasp pose
    sim.set_state(obs)
    x, y, z = obs.get_object_pose("cube1").position
    dz = 0.05
    pre_grasp_pose = Pose.from_rpy((x, y, z + dz), (np.pi, 0, np.pi / 2))
    grasp_pose = Pose.from_rpy((x, y, z + 0.015), (np.pi, 0, np.pi / 2))

    joint_distance_fn = create_joint_distance_fn(sim.robot.arm)
    joint_plan = smoothly_follow_end_effector_path(
        sim.robot.arm,
        [sim.robot.arm.get_end_effector_pose(), pre_grasp_pose, grasp_pose],
        sim.robot.arm.get_joint_positions(),
        collision_ids=set(),
        joint_distance_fn=joint_distance_fn,
        max_smoothing_iters_per_step=max_candidate_plans,
    )
    assert joint_plan is not None

    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
    )

    for target_joints in joint_plan[1:]:
        delta = np.subtract(target_joints[:7], obs.joint_positions)
        delta_lst = [wrap_angle(a) for a in delta]
        action_lst = [0.0] * 3 + delta_lst + [0.0]
        action = np.array(action_lst, dtype=np.float32)
        vec_obs, reward, terminated, truncated, _ = env.step(action)
        # Collect trajectory data
        traj_observations.append(vec_obs.copy())
        traj_actions.append(action.copy())
        traj_rewards.append(float(reward))
        ep_terminated = ep_terminated or terminated
        ep_truncated = ep_truncated or truncated
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = Table3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Step 3: Close the gripper to grasp cube1 (takes multiple steps)
    for _ in range(5):
        action = np.array([0.0] * 3 + [0.0] * 7 + [-1.0], dtype=np.float32)
        vec_obs, reward, terminated, truncated, _ = env.step(action)
        # Collect trajectory data
        traj_observations.append(vec_obs.copy())
        traj_actions.append(action.copy())
        traj_rewards.append(float(reward))
        ep_terminated = ep_terminated or terminated
        ep_truncated = ep_truncated or truncated
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = Table3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # The cube should now be grasped
    assert obs.grasped_object == "cube1"

    # The goal should not yet be reached: the cube is grasped but still resting
    # on the table, well within the goal height threshold.
    assert not terminated

    # Step 4: Move up to lift the cube well above the goal height threshold.
    sim.set_state(obs)
    current_end_effector_pose = sim.robot.arm.get_end_effector_pose()
    lifted_pose = Pose(
        (
            current_end_effector_pose.position[0],
            current_end_effector_pose.position[1],
            current_end_effector_pose.position[2]
            + config.goal_height_above_table
            + 0.1,
        ),
        current_end_effector_pose.orientation,
    )
    joint_plan = smoothly_follow_end_effector_path(
        sim.robot.arm,
        [current_end_effector_pose, lifted_pose],
        sim.robot.arm.get_joint_positions(),
        collision_ids=set(),
        joint_distance_fn=joint_distance_fn,
        max_smoothing_iters_per_step=max_candidate_plans,
    )

    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
    )

    for target_joints in joint_plan[1:]:
        delta = np.subtract(target_joints[:7], obs.joint_positions)
        delta_lst = [wrap_angle(a) for a in delta]
        action_lst = [0.0] * 3 + delta_lst + [0.0]
        action = np.array(action_lst, dtype=np.float32)
        vec_obs, reward, terminated, truncated, _ = env.step(action)
        # Collect trajectory data
        traj_observations.append(vec_obs.copy())
        traj_actions.append(action.copy())
        traj_rewards.append(float(reward))
        ep_terminated = ep_terminated or terminated
        ep_truncated = ep_truncated or truncated
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = Table3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Verify cube is still grasped after lifting
    assert obs.grasped_object == "cube1"

    # The goal is reached once the grasped cube is lifted above the table by
    # more than the configured threshold.
    object_centric_env = (
        env.unwrapped._object_centric_env  # pylint: disable=protected-access
    )
    assert terminated
    assert object_centric_env.goal_reached()

    # Save trajectory to pickle file
    if SAVE_TRAJECTORIES and len(traj_actions) > 0:
        demo_path = save_demo(
            demo_dir=DEFAULT_DEMOS_DIR,
            env_id=f"kinder/Table3D-o{num_cubes}-v0",
            seed=seed,
            observations=traj_observations,
            actions=traj_actions,
            rewards=traj_rewards,
            terminated=ep_terminated,
            truncated=ep_truncated,
        )
        print(f"Trajectory saved to {demo_path}")
        print(f"  Observations: {len(traj_observations)}, Actions: {len(traj_actions)}")


def test_pick_place_after_moving(env) -> None:  # pylint: disable=redefined-outer-name
    """Test moving in front of a block, picking it up, and placing it."""
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    config = (
        env.unwrapped._object_centric_env.config  # pylint: disable=protected-access
    )

    seed = 123
    vec_obs, _ = env.reset(seed=seed)
    oc_obs = env.observation_space.devectorize(vec_obs)
    obs = Table3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Initialize trajectory collection
    traj_observations: list[Any] = [vec_obs.copy()]
    traj_actions: list[Any] = []
    traj_rewards: list[float] = []
    ep_terminated = False
    ep_truncated = False

    num_cubes = 2
    # Create a simulator for planning.
    sim = ObjectCentricTable3DEnv(
        num_cubes=num_cubes,
        config=config,
        use_gui=False,
        realistic_bg=False,
        allow_state_access=True,
    )
    sim.set_state(obs)

    if MAKE_VIDEOS:
        max_candidate_plans = 20
    else:
        max_candidate_plans = 1

    # Step 1: Move the base in front of cube1
    target_object_pose_temp = obs.get_object_pose("cube1").to_se2()
    target_object_pose = SE2Pose(
        target_object_pose_temp.x - 0.4,
        target_object_pose_temp.y,
        target_object_pose_temp.rot,
    )
    base_plan = run_single_arm_mobile_base_motion_planning(
        sim.robot,
        sim.robot.base.get_pose(),
        target_object_pose,
        collision_bodies=set(),
        seed=123,
    )
    assert base_plan is not None

    for target_base_pose in base_plan[1:]:
        current_base_pose = obs.base_pose
        base_pose_delta = target_base_pose - current_base_pose
        delta_lst = [base_pose_delta.x, base_pose_delta.y, base_pose_delta.rot]
        action_lst = delta_lst + [0.0] * 7 + [0.0]
        action = np.array(action_lst, dtype=np.float32)
        vec_obs, reward, terminated, truncated, _ = env.step(action)
        # Collect trajectory data
        traj_observations.append(vec_obs.copy())
        traj_actions.append(action.copy())
        traj_rewards.append(float(reward))
        ep_terminated = ep_terminated or terminated
        ep_truncated = ep_truncated or truncated
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = Table3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Step 2: Move arm to pre-grasp pose and then to grasp pose
    sim.set_state(obs)
    x, y, z = obs.get_object_pose("cube1").position
    dz = 0.05
    pre_grasp_pose = Pose.from_rpy((x, y, z + dz), (np.pi, 0, np.pi / 2))
    grasp_pose = Pose.from_rpy((x, y, z + 0.015), (np.pi, 0, np.pi / 2))

    joint_distance_fn = create_joint_distance_fn(sim.robot.arm)
    joint_plan = smoothly_follow_end_effector_path(
        sim.robot.arm,
        [sim.robot.arm.get_end_effector_pose(), pre_grasp_pose, grasp_pose],
        sim.robot.arm.get_joint_positions(),
        collision_ids=set(),
        joint_distance_fn=joint_distance_fn,
        max_smoothing_iters_per_step=max_candidate_plans,
    )
    assert joint_plan is not None

    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
    )

    for target_joints in joint_plan[1:]:
        delta = np.subtract(target_joints[:7], obs.joint_positions)
        delta_lst = [wrap_angle(a) for a in delta]
        action_lst = [0.0] * 3 + delta_lst + [0.0]
        action = np.array(action_lst, dtype=np.float32)
        vec_obs, reward, terminated, truncated, _ = env.step(action)
        # Collect trajectory data
        traj_observations.append(vec_obs.copy())
        traj_actions.append(action.copy())
        traj_rewards.append(float(reward))
        ep_terminated = ep_terminated or terminated
        ep_truncated = ep_truncated or truncated
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = Table3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Step 3: Close the gripper to grasp cube1 (takes multiple steps)
    for _ in range(5):
        action = np.array([0.0] * 3 + [0.0] * 7 + [-1.0], dtype=np.float32)
        vec_obs, reward, terminated, truncated, _ = env.step(action)
        # Collect trajectory data
        traj_observations.append(vec_obs.copy())
        traj_actions.append(action.copy())
        traj_rewards.append(float(reward))
        ep_terminated = ep_terminated or terminated
        ep_truncated = ep_truncated or truncated
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = Table3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # The cube should now be grasped
    assert obs.grasped_object == "cube1"

    # Step 4: Move up to lift the cube
    sim.set_state(obs)
    current_end_effector_pose = sim.robot.arm.get_end_effector_pose()
    lifted_pose = Pose(
        (
            current_end_effector_pose.position[0],
            current_end_effector_pose.position[1],
            current_end_effector_pose.position[2] + 0.1,
        ),
        current_end_effector_pose.orientation,
    )
    joint_plan = smoothly_follow_end_effector_path(
        sim.robot.arm,
        [current_end_effector_pose, lifted_pose],
        sim.robot.arm.get_joint_positions(),
        collision_ids=set(),
        joint_distance_fn=joint_distance_fn,
        max_smoothing_iters_per_step=max_candidate_plans,
    )

    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
    )

    for target_joints in joint_plan[1:]:
        delta = np.subtract(target_joints[:7], obs.joint_positions)
        delta_lst = [wrap_angle(a) for a in delta]
        action_lst = [0.0] * 3 + delta_lst + [0.0]
        action = np.array(action_lst, dtype=np.float32)
        vec_obs, reward, terminated, truncated, _ = env.step(action)
        # Collect trajectory data
        traj_observations.append(vec_obs.copy())
        traj_actions.append(action.copy())
        traj_rewards.append(float(reward))
        ep_terminated = ep_terminated or terminated
        ep_truncated = ep_truncated or truncated
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = Table3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Verify cube is still grasped after lifting
    assert obs.grasped_object == "cube1"

    # Step 5: Place it back down
    sim.set_state(obs)
    current_end_effector_pose = sim.robot.arm.get_end_effector_pose()
    placement_pose = Pose(
        (
            current_end_effector_pose.position[0] + 0.1,
            current_end_effector_pose.position[1],
            obs.get_cuboid_pose("table").position[2]
            + config.table_half_extents[2] / 2
            + obs.get_cuboid_half_extents("cube1")[2],
        ),
        current_end_effector_pose.orientation,
    )

    joint_plan = smoothly_follow_end_effector_path(
        sim.robot.arm,
        [current_end_effector_pose, placement_pose],
        sim.robot.arm.get_joint_positions(),
        collision_ids=set(),
        joint_distance_fn=joint_distance_fn,
        max_smoothing_iters_per_step=max_candidate_plans,
    )

    if joint_plan is not None:
        joint_plan = remap_joint_position_plan_to_constant_distance(
            joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
        )

        for target_joints in joint_plan[1:]:
            delta = np.subtract(target_joints[:7], obs.joint_positions)
            delta_lst = [wrap_angle(a) for a in delta]
            action_lst = [0.0] * 3 + delta_lst + [0.0]
            action = np.array(action_lst, dtype=np.float32)
            vec_obs, reward, terminated, truncated, _ = env.step(action)
            # Collect trajectory data
            traj_observations.append(vec_obs.copy())
            traj_actions.append(action.copy())
            traj_rewards.append(float(reward))
            ep_terminated = ep_terminated or terminated
            ep_truncated = ep_truncated or truncated
            oc_obs = env.observation_space.devectorize(vec_obs)
            obs = Table3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Debug: Check if cube is close to table
    sim.set_state(obs)
    cube_id = sim._cubes["cube1"]  # pylint: disable=protected-access
    # fmt: off
    surface_supports = sim._get_surfaces_supporting_object(  # pylint: disable=protected-access
        cube_id
    )
    # fmt: on
    print(f"Surface supports: {surface_supports}")  # Should not be empty!

    # Step 6: Open the gripper to place the cube
    for _ in range(5):
        action = np.array([0.0] * 3 + [0.0] * 7 + [1.0], dtype=np.float32)
        vec_obs, reward, terminated, truncated, _ = env.step(action)
        # Collect trajectory data
        traj_observations.append(vec_obs.copy())
        traj_actions.append(action.copy())
        traj_rewards.append(float(reward))
        ep_terminated = ep_terminated or terminated
        ep_truncated = ep_truncated or truncated
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = Table3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    assert obs.grasped_object is None, "Object not released"

    # Save trajectory to pickle file
    if SAVE_TRAJECTORIES and len(traj_actions) > 0:
        demo_path = save_demo(
            demo_dir=DEFAULT_DEMOS_DIR,
            env_id=f"kinder/Table3D-o{num_cubes}-v0",
            seed=seed,
            observations=traj_observations,
            actions=traj_actions,
            rewards=traj_rewards,
            terminated=ep_terminated,
            truncated=ep_truncated,
        )
        print(f"Trajectory saved to {demo_path}")
        print(f"  Observations: {len(traj_observations)}, Actions: {len(traj_actions)}")
