"""Tests for transport3d.py."""

import numpy as np
import pytest
from gymnasium.wrappers import RecordVideo
from prpl_utils.utils import wrap_angle
from pybullet_helpers.geometry import Pose, SE2Pose
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    remap_joint_position_plan_to_constant_distance,
    remap_se2_pose_plan_to_constant_distance,
    run_motion_planning,
    run_single_arm_mobile_base_motion_planning,
    smoothly_follow_end_effector_path,
)
from relational_structs.spaces import ObjectCentricBoxSpace

from kinder.envs.kinematic3d.transport3d import (
    ObjectCentricTransport3DEnv,
    Transport3DEnv,
    Transport3DObjectCentricState,
)
from kinder.envs.kinematic3d.utils import extend_joints_to_include_fingers
from tests.conftest import MAKE_VIDEOS


@pytest.fixture(scope="module")
def env():
    """Create a shared environment for all tests in this module."""
    environment = Transport3DEnv(
        num_cubes=2,
        num_boxes=1,
        use_gui=False,
        render_mode="rgb_array",
        realistic_bg=False,
    )
    if MAKE_VIDEOS:
        environment = RecordVideo(environment, "unit_test_videos")
    yield environment
    environment.close()


def test_base_transport3d_env(env):  # pylint: disable=redefined-outer-name
    """Tests for basic methods in base transport3d env."""
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


def _execute_joint_plan(environment, joint_plan, obs):
    """Execute a joint space plan and return the final observation."""
    for target_joints in joint_plan[1:]:
        delta = np.subtract(target_joints[:7], obs.joint_positions)
        delta_lst = [wrap_angle(a) for a in delta]
        action_lst = [0.0] * 3 + delta_lst + [0.0]
        action = np.array(action_lst, dtype=np.float32)
        vec_obs, _, _, _, _ = environment.step(action)
        oc_obs = environment.observation_space.devectorize(vec_obs)
        obs = Transport3DObjectCentricState(oc_obs.data, oc_obs.type_features)
    return obs


def test_pick_place_after_moving(env):  # pylint: disable=redefined-outer-name
    """Test moving in front of a block or box, picking it up, and placing it."""
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    config = (
        env.unwrapped._object_centric_env.config  # pylint: disable=protected-access
    )

    vec_obs, _ = env.reset(seed=123)
    oc_obs = env.observation_space.devectorize(vec_obs)
    obs = Transport3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Create a simulator for planning.
    sim = ObjectCentricTransport3DEnv(
        num_cubes=2,
        num_boxes=1,
        config=config,
        realistic_bg=False,
        allow_state_access=True,
    )
    sim.set_state(obs)

    if MAKE_VIDEOS:
        max_candidate_plans = 20
    else:
        max_candidate_plans = 1

    # Step 1: Move the base in front of cube1 or box0
    target_object_pose_temp = obs.get_object_pose("box0").to_se2()
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
    base_plan = remap_se2_pose_plan_to_constant_distance(
        base_plan, max_distance=config.max_action_mag
    )

    for target_base_pose in base_plan[1:]:
        current_base_pose = obs.base_pose
        delta = target_base_pose - current_base_pose
        delta_lst = [delta.x, delta.y, delta.rot]
        action_lst = delta_lst + [0.0] * 7 + [0.0]
        action = np.array(action_lst, dtype=np.float32)
        vec_obs, _, _, _, _ = env.step(action)
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = Transport3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Step 2: Move arm to pre-grasp pose and then to grasp pose
    sim.set_state(obs)
    x, y, z = obs.get_object_pose("box0").position
    _, extent_y, extent_z = obs.get_object_half_extents("box0")
    dz = 0.05
    pre_grasp_pose = Pose.from_rpy(
        (x, y + extent_y, z + dz + extent_z / 2 + 0.1), (np.pi, 0, np.pi / 2)
    )
    grasp_pose = Pose.from_rpy(
        (x, y + extent_y, z + 0.015 + extent_z / 2 + 0.02), (np.pi, 0, np.pi / 2)
    )

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

    obs = _execute_joint_plan(env, joint_plan, obs)

    # Step 3: Close the gripper to grasp cube1 (takes multiple steps)
    for _ in range(5):
        action = np.array([0.0] * 3 + [0.0] * 7 + [-1.0], dtype=np.float32)
        vec_obs, _, _, _, _ = env.step(action)
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = Transport3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # The cube should now be grasped
    assert obs.grasped_object == "box0"

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

    obs = _execute_joint_plan(env, joint_plan, obs)

    # Verify cube is still grasped after lifting
    assert obs.grasped_object == "box0"

    # Step 5: Retract the arm
    sim.set_state(obs)
    joint_plan = run_motion_planning(
        sim.robot.arm,
        sim.robot.arm.get_joint_positions(),
        extend_joints_to_include_fingers(sim.config.initial_joints),
        collision_bodies=set(
            sim._get_collision_object_ids()  # pylint: disable=protected-access
            - {sim._grasped_object_id}  # pylint: disable=protected-access
        ),
        seed=123,
        physics_client_id=sim.physics_client_id,
        held_object=sim._grasped_object_id,  # pylint: disable=protected-access
        base_link_to_held_obj=sim._grasped_object_transform,  # pylint: disable=protected-access
    )
    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
    )
    obs = _execute_joint_plan(env, joint_plan, obs)

    # Verify cube is still grasped after lifting
    assert obs.grasped_object == "box0"

    # Step 6: Move the base in front of the table
    target_object_pose = SE2Pose(
        config.table_pose.position[0] - 0.65,
        config.table_pose.position[1],
        config.table_pose.orientation[2],
    )
    base_plan = run_single_arm_mobile_base_motion_planning(
        sim.robot,
        sim.robot.base.get_pose(),
        target_object_pose,
        collision_bodies=set(
            sim._get_collision_object_ids()  # pylint: disable=protected-access
        ),
        seed=123,
    )
    assert base_plan is not None
    base_plan = remap_se2_pose_plan_to_constant_distance(
        base_plan, max_distance=config.max_action_mag
    )

    for target_base_pose in base_plan[1:]:
        current_base_pose = obs.base_pose
        delta = target_base_pose - current_base_pose
        delta_lst = [delta.x, delta.y, delta.rot]
        action_lst = delta_lst + [0.0] * 7 + [0.0]
        action = np.array(action_lst, dtype=np.float32)
        vec_obs, _, _, _, _ = env.step(action)
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = Transport3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Step 7: Place it back down
    sim.set_state(obs)
    current_end_effector_pose = sim.robot.arm.get_end_effector_pose()
    placement_pose = Pose(
        (
            config.table_pose.position[0],
            config.table_pose.position[1],
            obs.get_object_pose("table").position[2]
            + config.table_half_extents[2]
            + obs.get_object_half_extents("box0")[2]
            + config.box_wall_thickness
            + 0.03,
        ),
        grasp_pose.orientation,
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

        obs = _execute_joint_plan(env, joint_plan, obs)

    # Step 8: Open the gripper to place the box
    for _ in range(5):
        action = np.array([0.0] * 3 + [0.0] * 7 + [1.0], dtype=np.float32)
        vec_obs, _, _, _, _ = env.step(action)
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = Transport3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    assert obs.grasped_object is None, "Object not released"
