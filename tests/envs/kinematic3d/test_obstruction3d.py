"""Tests for obstruction3d.py."""

import numpy as np
import pytest
from gymnasium.wrappers import RecordVideo
from prpl_utils.utils import wrap_angle
from pybullet_helpers.geometry import Pose, multiply_poses, set_pose
from pybullet_helpers.inverse_kinematics import inverse_kinematics
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    remap_joint_position_plan_to_constant_distance,
    run_smooth_motion_planning_to_pose,
    smoothly_follow_end_effector_path,
)
from relational_structs.spaces import ObjectCentricBoxSpace

from kinder.envs.kinematic3d.obstruction3d import (
    ObjectCentricObstruction3DEnv,
    Obstruction3DEnv,
    Obstruction3DEnvConfig,
    Obstruction3DObjectCentricState,
)
from tests.conftest import MAKE_VIDEOS


@pytest.fixture(scope="module")
def env():
    """Create a shared environment for all tests in this module."""
    config = Obstruction3DEnvConfig(target_block_height=0.01)
    environment = Obstruction3DEnv(
        num_obstructions=0,
        config=config,
        use_gui=False,
        render_mode="rgb_array",
        realistic_bg=False,
    )
    if MAKE_VIDEOS:
        environment = RecordVideo(environment, "unit_test_videos")
    yield environment
    environment.close()


def test_obstruction3d_env(env):  # pylint: disable=redefined-outer-name
    """Tests for basic methods in obstruction3d env."""
    obs, _ = env.reset(seed=123)
    assert isinstance(obs, np.ndarray)

    for _ in range(10):
        act = env.action_space.sample()
        assert isinstance(obs, np.ndarray)
        obs, _, _, _, _ = env.step(act)

    # Uncomment to debug.
    # import pybullet as p
    # while True:
    #     p.getMouseEvents(env.unwrapped._object_centric_env.physics_client_id)


def test_pick_place_no_obstructions(env):  # pylint: disable=redefined-outer-name
    """Test that picking and placing succeeds when there are no obstructions."""
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    config = (
        env.unwrapped._object_centric_env.config  # pylint: disable=protected-access
    )

    vec_obs, _ = env.reset(seed=123)
    # NOTE: we should soon make this smoother.
    oc_obs = env.observation_space.devectorize(vec_obs)
    obs = Obstruction3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Create a simulator for planning.
    sim = ObjectCentricObstruction3DEnv(
        num_obstructions=0,
        config=config,
        realistic_bg=False,
        allow_state_access=True,
    )
    sim.set_state(obs)

    # Run motion planning.
    if MAKE_VIDEOS:  # make a smooth motion plan for videos
        max_candidate_plans = 20
    else:
        max_candidate_plans = 1

    # First, move to pre-grasp pose (top-down).
    x, y, z = obs.target_block_pose.position
    dz = 0.035
    pre_grasp_pose = Pose.from_rpy((x, y, z + dz), (np.pi, 0, np.pi / 2))
    joint_plan = run_smooth_motion_planning_to_pose(
        pre_grasp_pose,
        sim.robot.arm,
        collision_ids=sim._get_collision_object_ids(),  # pylint: disable=protected-access
        end_effector_frame_to_plan_frame=Pose.identity(),
        seed=123,
        max_candidate_plans=max_candidate_plans,
    )
    assert joint_plan is not None

    # Make sure we stay below the required max_action_mag by a fair amount.
    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
    )

    for target_joints in joint_plan[1:]:
        delta = np.subtract(target_joints[:7], obs.joint_positions)
        delta_lst = [wrap_angle(a) for a in delta]
        action_lst = [0.0] * 3 + delta_lst + [0.0]
        action = np.array(action_lst, dtype=np.float32)
        vec_obs, _, _, _, _ = env.step(action)
        # NOTE: we should soon make this smoother.
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = Obstruction3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Close the gripper to grasp.
    action = np.array([0.0] * 3 + [0.0] * 7 + [-1.0], dtype=np.float32)
    vec_obs, _, _, _, _ = env.step(action)
    # NOTE: we should soon make this smoother.
    oc_obs = env.observation_space.devectorize(vec_obs)
    obs = Obstruction3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # The target block should now be grasped.
    assert obs.grasped_object == "target_block"

    # Move up slightly to break contact with the table.
    sim.set_state(obs)
    current_end_effector_pose = sim.robot.arm.get_end_effector_pose()
    post_grasp_pose = Pose(
        (
            current_end_effector_pose.position[0],
            current_end_effector_pose.position[1],
            current_end_effector_pose.position[2] + 1e-2,
        ),
        current_end_effector_pose.orientation,
    )
    joint_distance_fn = create_joint_distance_fn(sim.robot.arm)
    joint_plan = smoothly_follow_end_effector_path(
        sim.robot.arm,
        [current_end_effector_pose, post_grasp_pose],
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
        vec_obs, _, _, _, _ = env.step(action)
        # NOTE: we should soon make this smoother.
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = Obstruction3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Determine placement pose and pre-placement pose. Place directly in the center of
    # the target region for this test.
    placement_padding = 1e-4  # leave some room to prevent collisions with surface
    block_placement_pose = Pose(
        (
            obs.target_region_pose.position[0],
            obs.target_region_pose.position[1],
            obs.target_region_pose.position[2]
            + obs.target_region_half_extents[2]
            + obs.target_block_half_extents[2]
            + placement_padding,
        ),
        obs.target_region_pose.orientation,
    )
    end_effector_placement_pose = multiply_poses(
        block_placement_pose,
        obs.grasped_object_transform,
    )
    end_effector_pre_placement_pose = Pose(
        (
            end_effector_placement_pose.position[0],
            end_effector_placement_pose.position[1],
            end_effector_placement_pose.position[2] + 1e-2,
        ),
        end_effector_placement_pose.orientation,
    )

    # We don't really have to motion plan here because there are no other objects, but
    # in general we would motion plan.
    sim.set_state(obs)
    current_end_effector_pose = sim.robot.arm.get_end_effector_pose()
    joint_plan = smoothly_follow_end_effector_path(
        sim.robot.arm,
        [
            current_end_effector_pose,
            end_effector_pre_placement_pose,
            end_effector_placement_pose,
        ],
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
        vec_obs, _, _, _, _ = env.step(action)
        # NOTE: we should soon make this smoother.
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = Obstruction3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Open the gripper to finish the placement. Should trigger "done" (goal reached).
    action = np.array([0.0] * 3 + [0.0] * 7 + [1.0], dtype=np.float32)
    vec_obs, _, done, _, _ = env.step(action)
    # NOTE: we should soon make this smoother.
    oc_obs = env.observation_space.devectorize(vec_obs)
    obs = Obstruction3DObjectCentricState(oc_obs.data, oc_obs.type_features)
    assert obs.grasped_object is None, "Object not released"
    assert done, "Goal not reached"

    # Uncomment to debug.
    # import pybullet as p
    # from pybullet_helpers.gui import visualize_pose
    # visualize_pose(end_effector_placement_pose, env.physics_client_id)
    # while True:
    #     p.getMouseEvents(env.physics_client_id)


def test_goal_requires_block_on_target_region():
    """Test that the goal is reached only when the block rests on the region."""
    config = Obstruction3DEnvConfig()
    oc_env = ObjectCentricObstruction3DEnv(
        num_obstructions=0, config=config, realistic_bg=False
    )
    obs, _ = oc_env.reset(seed=123)

    region_pose = obs.target_region_pose
    region_half = obs.target_region_half_extents
    block_half = obs.target_block_half_extents
    table_top = config.table_pose.position[2] + config.table_half_extents[2]
    block_id = oc_env._target_block_id  # pylint: disable=protected-access
    client_id = oc_env.physics_client_id

    # Block on the table right beside the region (1 mm lateral gap to the
    # region slab): not on the region, so the goal must not be reached.
    beside_pose = Pose(
        (
            region_pose.position[0],
            region_pose.position[1] + region_half[1] + block_half[1] + 1e-3,
            table_top + block_half[2],
        )
    )
    set_pose(block_id, beside_pose, client_id)
    assert not oc_env.goal_reached(), "Goal reached with block beside the region"

    # Block resting on top of the region: goal reached.
    on_region_pose = Pose(
        (
            region_pose.position[0],
            region_pose.position[1],
            region_pose.position[2] + region_half[2] + block_half[2] + 1e-4,
        )
    )
    set_pose(block_id, on_region_pose, client_id)
    assert oc_env.goal_reached(), "Goal not reached with block on the region"

    # Same placement but yawed (e.g. after a grasp/place cycle): still reached.
    yawed_pose = Pose.from_rpy(on_region_pose.position, (0.0, 0.0, np.pi / 4))
    set_pose(block_id, yawed_pose, client_id)
    assert oc_env.goal_reached(), "Goal not reached with yawed block on the region"

    # Block hovering well above the region: not reached.
    hover_pose = Pose(
        (
            on_region_pose.position[0],
            on_region_pose.position[1],
            on_region_pose.position[2] + 0.05,
        )
    )
    set_pose(block_id, hover_pose, client_id)
    assert not oc_env.goal_reached(), "Goal reached with block hovering above region"

    # Block on the table far away from the region: not reached.
    far_pose = Pose(
        (
            region_pose.position[0],
            region_pose.position[1] + 0.2,
            table_top + block_half[2],
        )
    )
    set_pose(block_id, far_pose, client_id)
    assert not oc_env.goal_reached(), "Goal reached with block far from region"

    oc_env.close()


def test_grasp_fails_when_fingers_collide_with_table():
    """Test that grasping fails when fingers collide with table during grasp."""
    # Create environment with no obstructions.
    config = Obstruction3DEnvConfig(
        target_block_height=0.015, target_block_size_scale=0.5
    )
    oc_env = ObjectCentricObstruction3DEnv(
        num_obstructions=0, config=config, realistic_bg=False
    )

    obs, _ = oc_env.reset(seed=456)

    # Position the gripper very low (close to the table surface) around the block.
    x, y, _ = obs.target_block_pose.position
    # Position gripper very close to table surface - when fingers close, they'll
    # collide with the table.
    grasp_z = 0.11  # Just barely above table surface
    low_grasp_pose = Pose.from_rpy((x, y, grasp_z), (np.pi, 0, np.pi / 2))

    # Use IK to get joint positions for this pose, then directly set the state.
    target_joints = inverse_kinematics(
        oc_env._robot_arm,  # pylint: disable=protected-access
        low_grasp_pose,
        validate=False,
    )
    assert target_joints is not None

    # Directly set robot state to this configuration.
    oc_env.robot.arm.set_joints(target_joints)
    oc_env._robot_arm.open_fingers()  # pylint: disable=protected-access

    # Attempt to grasp. This should fail because the fingers will collide with
    # the table when they close.
    close_action = np.array([0.0] * 3 + [0.0] * 7 + [-1.0], dtype=np.float32)
    obs, _, _, _, _ = oc_env.step(close_action)

    # The grasp should have failed - grasped_object should be None.
    assert obs.grasped_object is None, "Grasp should have failed due to table collision"
