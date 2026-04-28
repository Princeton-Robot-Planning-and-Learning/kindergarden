"""Tests for prpl3d.py."""

# pylint: disable=protected-access

import numpy as np
import pytest
from gymnasium.wrappers import RecordVideo
from prpl_utils.utils import wrap_angle
from pybullet_helpers.geometry import Pose, SE2Pose
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    remap_joint_position_plan_to_constant_distance,
    run_motion_planning,
    run_single_arm_mobile_base_motion_planning,
    run_smooth_motion_planning_to_pose,
    smoothly_follow_end_effector_path,
)
from relational_structs.spaces import ObjectCentricBoxSpace

from kinder.envs.kinematic3d.prpl3d import (
    ObjectCentricPrplLab3DEnv,
    PrplLab3DEnv,
    PrplLab3DObjectCentricState,
)
from kinder.envs.kinematic3d.utils import extend_joints_to_include_fingers
from tests.conftest import MAKE_VIDEOS


@pytest.fixture(scope="module")
def env():
    """Create a shared environment for all tests in this module."""
    environment = PrplLab3DEnv(
        num_cubes=2,
        use_gui=False,
        render_mode="rgb_array",
        realistic_bg=False,
    )
    if MAKE_VIDEOS:
        environment = RecordVideo(environment, "unit_test_videos")
    yield environment
    environment.close()


def _execute_base_plan(environment, base_plan, obs):
    """Execute a base motion plan and return the final observation."""
    for target_base_pose in base_plan[1:]:
        delta = target_base_pose - obs.base_pose
        action = np.array(
            [delta.x, delta.y, delta.rot] + [0.0] * 7 + [0.0], dtype=np.float32
        )
        vec_obs, _, _, _, _ = environment.step(action)
        oc_obs = environment.observation_space.devectorize(vec_obs)
        obs = PrplLab3DObjectCentricState(oc_obs.data, oc_obs.type_features)
    return obs


def _execute_joint_plan(environment, joint_plan, obs):
    """Execute a joint space plan and return the final observation."""
    for target_joints in joint_plan[1:]:
        delta = [
            wrap_angle(a) for a in np.subtract(target_joints[:7], obs.joint_positions)
        ]
        action = np.array([0.0] * 3 + delta + [0.0], dtype=np.float32)
        vec_obs, _, _, _, _ = environment.step(action)
        oc_obs = environment.observation_space.devectorize(vec_obs)
        obs = PrplLab3DObjectCentricState(oc_obs.data, oc_obs.type_features)
    return obs


def test_prpl_lab_3d_env(env):  # pylint: disable=redefined-outer-name
    """Tests for basic methods in the PRPL lab env."""
    obs, _ = env.reset(seed=123)
    assert isinstance(obs, np.ndarray)

    for _ in range(10):
        act = env.action_space.sample()
        assert isinstance(act, np.ndarray)
        obs, _, _, _, _ = env.step(act)


def test_camera_rendering(env):  # pylint: disable=redefined-outer-name
    """Test rendering from overview, base, and end-effector cameras."""
    env.reset(seed=123)

    oc_env = env.unwrapped._object_centric_env
    config = oc_env.config

    overview_image = oc_env.render()
    assert overview_image is not None
    assert overview_image.shape == (
        config.render_image_height,
        config.render_image_width,
        3,
    )
    assert overview_image.dtype == np.uint8

    base_image = oc_env.render_base_camera()
    assert base_image is not None
    assert base_image.shape == (
        config.base_camera_image_height,
        config.base_camera_image_width,
        3,
    )
    assert base_image.dtype == np.uint8

    ee_image = oc_env.render_ee_camera()
    assert ee_image is not None
    assert ee_image.shape == (
        config.ee_camera_image_height,
        config.ee_camera_image_width,
        3,
    )
    assert ee_image.dtype == np.uint8

    all_images = oc_env.render_all_cameras()
    assert isinstance(all_images, dict)
    assert set(all_images.keys()) == {"overview", "base", "wrist"}
    assert all_images["overview"].shape == overview_image.shape
    assert all_images["base"].shape == base_image.shape
    assert all_images["wrist"].shape == ee_image.shape

    for _ in range(5):
        act = env.action_space.sample()
        env.step(act)

    all_images_after = oc_env.render_all_cameras()
    assert all_images_after["overview"].shape == overview_image.shape
    assert all_images_after["base"].shape == base_image.shape
    assert all_images_after["wrist"].shape == ee_image.shape


def test_pick_place(env):  # pylint: disable=redefined-outer-name
    """Test picking a cube from the floor and placing it on the counter."""
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    config = env.unwrapped._object_centric_env.config

    vec_obs, _ = env.reset(seed=123)
    oc_obs = env.observation_space.devectorize(vec_obs)
    obs = PrplLab3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    sim = ObjectCentricPrplLab3DEnv(
        num_cubes=2,
        config=config,
        use_gui=False,
        realistic_bg=False,
        allow_state_access=True,
    )
    sim.set_state(obs)

    oc_env = env.unwrapped._object_centric_env

    if MAKE_VIDEOS:
        max_candidate_plans = 20
    else:
        max_candidate_plans = 1

    # Step 1: Move the base in front of cube1.
    cube_se2 = obs.get_object_pose("cube1").to_se2()
    target_base = SE2Pose(cube_se2.x, cube_se2.y - 0.5, np.pi / 2)
    base_plan = run_single_arm_mobile_base_motion_planning(
        sim.robot,
        sim.robot.base.get_pose(),
        target_base,
        collision_bodies=set(),
        seed=123,
    )
    assert base_plan is not None
    obs = _execute_base_plan(env, base_plan, obs)

    # Step 2: Move arm to pre-grasp pose.
    sim.set_state(obs)
    x, y, z = obs.get_object_pose("cube1").position
    pre_grasp_pose = Pose.from_rpy((x, y, z + 0.05), (np.pi, 0, np.pi / 2))
    grasp_pose = Pose.from_rpy((x, y, z + 0.005), (np.pi, 0, np.pi / 2))

    joint_distance_fn = create_joint_distance_fn(sim.robot.arm)
    joint_plan = run_smooth_motion_planning_to_pose(
        pre_grasp_pose,
        sim.robot.arm,
        collision_ids=set(),
        end_effector_frame_to_plan_frame=Pose.identity(),
        seed=123,
        max_candidate_plans=max_candidate_plans,
    )
    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
    )
    obs = _execute_joint_plan(env, joint_plan, obs)

    # Step 3: Descend to grasp pose.
    sim.set_state(obs)
    joint_plan = smoothly_follow_end_effector_path(
        sim.robot.arm,
        [sim.robot.arm.get_end_effector_pose(), grasp_pose],
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

    # Step 4: Close the gripper.
    for _ in range(5):
        action = np.array([0.0] * 3 + [0.0] * 7 + [-1.0], dtype=np.float32)
        vec_obs, _, _, _, _ = env.step(action)
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = PrplLab3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    assert obs.grasped_object == "cube1"

    # Step 5: Retract arm to home joints.
    sim.set_state(obs)
    joint_plan = run_motion_planning(
        sim.robot.arm,
        sim.robot.arm.get_joint_positions(),
        extend_joints_to_include_fingers(sim.config.initial_joints),
        collision_bodies=set(),
        seed=123,
        physics_client_id=sim.physics_client_id,
        held_object=sim._grasped_object_id,
        base_link_to_held_obj=sim._grasped_object_transform,
    )
    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
    )
    obs = _execute_joint_plan(env, joint_plan, obs)

    assert obs.grasped_object == "cube1"

    # Step 6: Move base to counter.
    sim.set_state(obs)
    counter_x = 0.3
    target_counter_base = SE2Pose(counter_x, 0.7, np.pi / 2)
    base_plan = run_single_arm_mobile_base_motion_planning(
        sim.robot,
        sim.robot.base.get_pose(),
        target_counter_base,
        collision_bodies=set(),
        seed=456,
    )
    assert base_plan is not None
    obs = _execute_base_plan(env, base_plan, obs)

    # Step 7: Place cube on counter surface.
    sim.set_state(obs)
    # EE just above the counter so the cube rests on the surface.
    counter_surface_z = 0.76
    place_z = counter_surface_z + config.block_half_extents[2]
    place_pose = Pose.from_rpy((counter_x, 1.6, place_z), (-np.pi / 2, np.pi, 0))

    joint_plan = run_smooth_motion_planning_to_pose(
        place_pose,
        sim.robot.arm,
        collision_ids=set(),
        end_effector_frame_to_plan_frame=Pose.identity(),
        seed=123,
        max_candidate_plans=max_candidate_plans,
        held_object=sim._grasped_object_id,
        base_link_to_held_obj=sim._grasped_object_transform,
    )
    assert joint_plan is not None
    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
    )
    obs = _execute_joint_plan(env, joint_plan, obs)

    assert obs.grasped_object == "cube1"

    # Step 8: Release the cube directly.
    oc_env._grasped_object = None
    oc_env._grasped_object_transform = None
    oc_env.robot.arm.open_fingers()

    # Sync obs after direct manipulation.
    action = np.zeros(11, dtype=np.float32)
    vec_obs, _, _, _, _ = env.step(action)
    oc_obs = env.observation_space.devectorize(vec_obs)
    obs = PrplLab3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    assert obs.grasped_object is None

    # Cube should be approximately at the target height.
    cube_z = obs.get_object_pose("cube1").position[2]
    assert abs(cube_z - place_z) < 0.15, f"Cube z {cube_z:.3f} not near counter"

    sim.close()
    env.close()
