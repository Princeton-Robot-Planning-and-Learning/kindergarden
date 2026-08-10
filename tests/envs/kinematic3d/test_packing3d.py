"""Tests for packing3d.py."""

from typing import Any, cast

import numpy as np
import pybullet as p
import pytest
from gymnasium.wrappers import RecordVideo
from prpl_utils.utils import wrap_angle
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses, set_pose
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    remap_joint_position_plan_to_constant_distance,
    run_smooth_motion_planning_to_pose,
    smoothly_follow_end_effector_path,
)
from pybullet_helpers.utils import get_triangle_vertices
from relational_structs import Object
from relational_structs.spaces import ObjectCentricBoxSpace
from shapely.geometry import Polygon

from kinder.envs.kinematic3d.object_types import (
    Kinematic3DCuboidType,
    Kinematic3DTriangleType,
)
from kinder.envs.kinematic3d.packing3d import (
    ObjectCentricPacking3DEnv,
    Packing3DEnv,
    Packing3DEnvConfig,
    Packing3DObjectCentricState,
)
from kinder.envs.kinematic3d.save_utils import DEFAULT_DEMOS_DIR, save_demo
from kinder.envs.kinematic3d.utils import is_inside
from tests.conftest import MAKE_VIDEOS

# Flag to enable trajectory saving (can be controlled like MAKE_VIDEOS)
SAVE_TRAJECTORIES = MAKE_VIDEOS


def test_packing3d_env_basic():
    """Basic smoke test for the packing3d environment."""
    for num_parts in [1, 2, 3]:
        env = Packing3DEnv(
            num_parts=num_parts, use_gui=False, realistic_bg=False
        )  # set use_gui=False to debug
        obs, _ = env.reset(seed=123)
        assert isinstance(obs, np.ndarray)

        for _ in range(10):
            act = env.action_space.sample()
            assert isinstance(obs, np.ndarray)
            obs, _, _, _, _ = env.step(act)

        env.close()


def test_packing3d_uses_standard_action_magnitude() -> None:
    """Packing3D uses the standard action bound used by the 3D envs."""
    env = Packing3DEnv(num_parts=1, use_gui=False, realistic_bg=False)
    assert np.all(env.action_space.low[:10] == -0.2)
    assert np.all(env.action_space.high[:10] == 0.2)
    assert env.unwrapped._object_centric_env.config.max_collision_check_step == 0.005
    env.close()


def test_packing3d_rejects_collision_tunneling_action() -> None:
    """A collision-free endpoint is invalid when the swept motion collides."""
    env = Packing3DEnv(
        num_parts=3,
        config=Packing3DEnvConfig(
            max_action_mag=0.4,
            max_collision_check_step=0.05,
        ),
        use_gui=False,
        realistic_bg=False,
    )
    obs, _ = env.reset(seed=7138484576005690180)

    # The arm crosses a collision before reaching a collision-free endpoint.
    action = np.array(
        [
            0.0417,
            -0.3489,
            0.0,
            -0.0018,
            0.3679,
            0.0094,
            0.0146,
            -0.0002,
            0.2311,
            0.0072,
            -1.0,
        ],
        dtype=np.float32,
    )
    next_obs, _, _, _, _ = env.step(action)

    assert np.allclose(next_obs[:11], obs[:11])
    assert np.all(next_obs[[38, 50, 62]] == 0.0)
    env.close()


def test_packing3d_goal_rejects_part_outside_rack_cavity():
    """A part on the outer rack footprint is not seated in its cavity."""
    env = ObjectCentricPacking3DEnv(
        num_parts=1,
        use_gui=False,
        realistic_bg=False,
        allow_state_access=True,
    )

    obs, _ = env.reset(seed=0)
    part = obs.get_object_from_name("part0")
    assert part.type == Kinematic3DTriangleType
    assert obs.get(part, "triangle_type") == 1

    rack_pose = obs.rack_pose
    rack_half_extents = obs.rack_half_extents
    side_a, side_b, _, triangle_type = obs.get_object_triangle_features("part0")
    assert triangle_type == 1

    triangle_vertices = get_triangle_vertices("right", (side_a, side_b))
    part_x = rack_pose.position[0] - rack_half_extents[0]
    part_y = rack_pose.position[1] - side_b / 2

    # set_state() restores triangle pose fields as raw PyBullet body poses.
    # Subtracting the centroid here would encode the old get_object_pose()
    # restoration behavior and place the true triangle footprint outside the rack.
    obs.set(part, "pose_x", part_x)
    obs.set(part, "pose_y", part_y)
    obs.set(part, "pose_z", rack_pose.position[2])
    obs.set(part, "pose_qx", 0.0)
    obs.set(part, "pose_qy", 0.0)
    obs.set(part, "pose_qz", 0.0)
    obs.set(part, "pose_qw", 1.0)
    obs.set(part, "grasp_active", 0.0)
    env.set_state(obs)

    rack_polygon = Polygon(
        [
            (
                rack_pose.position[0] - rack_half_extents[0],
                rack_pose.position[1] - rack_half_extents[1],
            ),
            (
                rack_pose.position[0] + rack_half_extents[0],
                rack_pose.position[1] - rack_half_extents[1],
            ),
            (
                rack_pose.position[0] + rack_half_extents[0],
                rack_pose.position[1] + rack_half_extents[1],
            ),
            (
                rack_pose.position[0] - rack_half_extents[0],
                rack_pose.position[1] + rack_half_extents[1],
            ),
        ]
    )
    triangle_polygon = Polygon(
        [(part_x + vx, part_y + vy) for vx, vy, _ in triangle_vertices]
    )

    assert rack_polygon.covers(triangle_polygon)

    updated_obs = env.get_state()
    assert not is_inside(
        updated_obs.rack_pose,
        updated_obs.rack_half_extents,
        updated_obs.get_object_pose("part0"),
        updated_obs.get_object_half_extents_packing3d("part0")[:3],
    )
    assert "part0" not in updated_obs.available_parts
    assert not env.goal_reached()

    env.close()


def _get_part_shape_name(obs: Packing3DObjectCentricState, part_name: str) -> str:
    part = obs.get_object_from_name(part_name)
    if part.type == Kinematic3DCuboidType:
        return "cuboid"
    assert part.type == Kinematic3DTriangleType
    triangle_type = obs.get_object_triangle_features(part_name)[3]
    return {0: "equilateral", 1: "right"}[int(triangle_type)]


def _place_part_on_rack_floor(
    env: ObjectCentricPacking3DEnv,
    part_name: str,
    target_xy: tuple[float, float],
    yaw: float = 0.0,
) -> None:
    part_id = env._part_ids[part_name]
    old_pose = get_pose(part_id, env.physics_client_id)
    orientation = Pose.from_rpy((0.0, 0.0, 0.0), (0.0, 0.0, yaw)).orientation
    set_pose(
        part_id,
        Pose(old_pose.position, orientation),
        env.physics_client_id,
    )
    vertices = env._get_part_convex_components(part_name)[0][0]
    polygon = Polygon(vertices[:, :2]).convex_hull
    rack_pose = get_pose(env._rack_id, env.physics_client_id)
    floor_z = (
        rack_pose.position[2]
        - env.config.rack_half_extents[2]
        + env.config.rack_wall_thickness
    )
    current_pose = get_pose(part_id, env.physics_client_id)
    set_pose(
        part_id,
        Pose(
            (
                current_pose.position[0] + target_xy[0] - polygon.centroid.x,
                current_pose.position[1] + target_xy[1] - polygon.centroid.y,
                current_pose.position[2] + floor_z + 0.002 - vertices[:, 2].min(),
            ),
            current_pose.orientation,
        ),
        env.physics_client_id,
    )


@pytest.mark.parametrize(
    ("triangle_probability", "seed", "expected_shapes"),
    [
        (1.0, 0, ("cuboid", "cuboid")),
        (0.5, 0, ("cuboid", "right")),
        (0.5, 5, ("cuboid", "equilateral")),
        (0.0, 0, ("right", "right")),
        (0.0, 1, ("equilateral", "right")),
        (0.0, 11, ("equilateral", "equilateral")),
    ],
)
def test_packing3d_goal_handles_every_part_shape_pair(
    triangle_probability: float,
    seed: int,
    expected_shapes: tuple[str, str],
) -> None:
    """All supported shape pairs reject overlap and accept clear packing."""
    env = ObjectCentricPacking3DEnv(
        num_parts=2,
        config=Packing3DEnvConfig(part_triangular_prob=triangle_probability),
        use_gui=False,
        realistic_bg=False,
        allow_state_access=True,
    )
    obs, _ = env.reset(seed=seed)
    assert (
        tuple(_get_part_shape_name(obs, f"part{i}") for i in range(2))
        == expected_shapes
    )
    rack_pose = get_pose(env._rack_id, env.physics_client_id)

    # Rotate one part to exercise non-axis-aligned cuboid and triangle geometry.
    # Both parts occupy the same supported rack region and must be rejected.
    target_xy = (rack_pose.position[0], rack_pose.position[1])
    for part_name in ("part0", "part1"):
        _place_part_on_rack_floor(
            env, part_name, target_xy, yaw=0.0 if part_name == "part0" else np.pi / 6
        )

    obs = env.get_state()
    assert env._part_is_seated_in_rack("part0", obs)
    assert env._part_is_seated_in_rack("part1", obs)
    assert env._parts_penetrate("part0", "part1")
    assert not env.goal_reached()

    # The same shapes are a valid terminal arrangement when separated.
    _place_part_on_rack_floor(
        env, "part0", (rack_pose.position[0], rack_pose.position[1] - 0.07)
    )
    _place_part_on_rack_floor(
        env, "part1", (rack_pose.position[0], rack_pose.position[1] + 0.07)
    )
    obs = env.get_state()
    assert env._part_is_seated_in_rack("part0", obs)
    assert env._part_is_seated_in_rack("part1", obs)
    assert not env._parts_penetrate("part0", "part1")
    assert env.goal_reached()
    env.close()


def test_packing3d_rejects_triangle_overlap_missed_by_pybullet() -> None:
    """Analytic geometry catches the mesh-query false negative from the replay."""
    env = ObjectCentricPacking3DEnv(
        num_parts=2,
        config=Packing3DEnvConfig(part_triangular_prob=0.0),
        use_gui=False,
        realistic_bg=False,
        allow_state_access=True,
    )
    obs, _ = env.reset(seed=0)
    assert tuple(_get_part_shape_name(obs, f"part{i}") for i in range(2)) == (
        "right",
        "right",
    )
    rack_pose = get_pose(env._rack_id, env.physics_client_id)
    _place_part_on_rack_floor(
        env, "part0", (rack_pose.position[0] - 0.003, rack_pose.position[1] - 0.028)
    )
    _place_part_on_rack_floor(
        env, "part1", (rack_pose.position[0] + 0.003, rack_pose.position[1] + 0.028)
    )

    part0_id = env._part_ids["part0"]
    part1_id = env._part_ids["part1"]
    assert not env._bodies_penetrate(part0_id, part1_id)
    assert env._parts_penetrate("part0", "part1")
    assert not env.goal_reached()

    # The same analytic check participates in swept validation while carrying a
    # triangle; the PyBullet-only result above must not allow this configuration.
    env._grasped_object = "part0"
    env._grasped_object_transform = multiply_poses(
        env.robot.arm.get_end_effector_pose().invert(),
        get_pose(part0_id, env.physics_client_id),
    )
    assert env._robot_or_held_object_collision_exists()
    env.close()


def test_packing3d_part_collision_includes_grasp_pegs() -> None:
    """The analytic part geometry includes the red peg, not just the main body."""
    env = ObjectCentricPacking3DEnv(
        num_parts=2,
        config=Packing3DEnvConfig(part_triangular_prob=1.0),
        use_gui=False,
        realistic_bg=False,
        allow_state_access=True,
    )
    env.reset(seed=0)
    first_id = env._part_ids["part0"]
    second_id = env._part_ids["part1"]
    set_pose(first_id, Pose((0.0, 0.0, 0.0)), env.physics_client_id)

    # The 2-cm-tall main cuboids are vertically separate, but the lower part's
    # 5-cm grasp peg penetrates the upper part's main body by 5 mm.
    set_pose(second_id, Pose((0.0, 0.0, 0.065)), env.physics_client_id)
    first_main = env._get_part_convex_components("part0")[0][0]
    second_main = env._get_part_convex_components("part1")[0][0]
    assert first_main[:, 2].max() < second_main[:, 2].min()
    assert env._parts_penetrate("part0", "part1")

    # Exact contact is allowed; only strict penetration is rejected.
    set_pose(second_id, Pose((0.0, 0.0, 0.07)), env.physics_client_id)
    assert not env._parts_penetrate("part0", "part1")
    env.close()


@pytest.mark.parametrize("triangle_probability", [0.0, 0.5, 1.0])
def test_packing3d_reset_avoids_analytic_part_overlap(
    triangle_probability: float,
) -> None:
    """Initial rejection sampling works for triangle and cuboid mixtures."""
    env = ObjectCentricPacking3DEnv(
        num_parts=3,
        config=Packing3DEnvConfig(part_triangular_prob=triangle_probability),
        use_gui=False,
        realistic_bg=False,
        allow_state_access=True,
    )
    for seed in range(10):
        env.reset(seed=seed)
        for first_index in range(3):
            for second_index in range(first_index + 1, 3):
                assert not env._parts_penetrate(
                    f"part{first_index}", f"part{second_index}"
                )
    env.close()


def test_packing3d_goal_rejects_rack_penetration_and_hovering() -> None:
    """Cavity containment alone is insufficient without valid floor support."""
    env = ObjectCentricPacking3DEnv(
        num_parts=1,
        use_gui=False,
        realistic_bg=False,
        allow_state_access=True,
    )
    env.reset(seed=0)
    rack_pose = get_pose(env._rack_id, env.physics_client_id)
    floor_z = (
        rack_pose.position[2]
        - env.config.rack_half_extents[2]
        + env.config.rack_wall_thickness
    )
    part_id = env._part_ids["part0"]
    part_pose = get_pose(part_id, env.physics_client_id)
    part_lower, _ = p.getAABB(part_id, 0, physicsClientId=env.physics_client_id)
    lower_z_offset = part_lower[2] - part_pose.position[2]
    xy = (rack_pose.position[0] - 0.03, rack_pose.position[1] - 0.02)

    # The footprint is inside and the part is near the floor, but it penetrates it.
    set_pose(
        part_id,
        Pose((*xy, floor_z - 0.002 - lower_z_offset)),
        env.physics_client_id,
    )
    assert not env.goal_reached()

    # The footprint remains inside, but the part is too high to be supported.
    set_pose(
        part_id,
        Pose((*xy, floor_z + 0.01 - lower_z_offset)),
        env.physics_client_id,
    )
    assert not env.goal_reached()
    env.close()


def get_target_object_from_obs(
    obs: Packing3DObjectCentricState,
) -> Object | None:
    """Get the target object from the observation."""
    available_parts = obs.available_parts
    if not available_parts:
        return None
    # For simplicity, just choose the first available part.
    target_part_name = available_parts[0]
    return obs.get_object_from_name(target_part_name)


def test_triangle_part_set_state_does_not_drift() -> None:
    """Repeated state restoration should not move triangular parts."""
    env = ObjectCentricPacking3DEnv(
        num_parts=1,
        realistic_bg=False,
        allow_state_access=True,
    )
    obs, _ = env.reset(seed=123)
    assert isinstance(obs, Packing3DObjectCentricState)

    part = obs.get_object_from_name("part0")
    initial_pose = (obs.get(part, "pose_x"), obs.get(part, "pose_y"))
    for _ in range(3):
        env.set_state(obs)
        obs = env.get_state()
        assert isinstance(obs, Packing3DObjectCentricState)

    part = obs.get_object_from_name("part0")
    final_pose = (obs.get(part, "pose_x"), obs.get(part, "pose_y"))
    np.testing.assert_allclose(final_pose, initial_pose)
    env.close()


def test_pick_place_on_rack() -> None:
    """Test that picking and placing can be executed for any object."""
    # Create the real environment.

    num_parts = 2
    seed = 123
    env = Packing3DEnv(
        num_parts=num_parts, use_gui=False, render_mode="rgb_array", realistic_bg=False
    )
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    obs_space = env.observation_space
    unwrapped_env = cast(Packing3DEnv, env.unwrapped)
    # pylint: disable-next=protected-access
    config = unwrapped_env._object_centric_env.config
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")  # type: ignore[assignment]

    vec_obs, _ = env.reset(seed=seed)
    # NOTE: we should soon make this smoother.
    oc_obs = obs_space.devectorize(vec_obs)
    obs = Packing3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Initialize trajectory collection
    traj_observations: list[Any] = [vec_obs.copy()]
    traj_actions: list[Any] = []
    traj_rewards: list[float] = []
    ep_terminated = False
    ep_truncated = False

    sim = ObjectCentricPacking3DEnv(
        num_parts=num_parts,
        config=config,
        realistic_bg=False,
        allow_state_access=True,
    )
    sim.reset()
    sim.set_state(obs)

    home_pos = sim.robot.arm.get_end_effector_pose()
    home_pos = Pose(
        (home_pos.position[0], home_pos.position[1], home_pos.position[2] + 0.2),
        home_pos.orientation,
    )

    # Run motion planning.
    if MAKE_VIDEOS:  # make a smooth motion plan for videos
        max_candidate_plans = 10
    else:
        max_candidate_plans = 1

    # sample placement coefficients for each part
    x_coeffs = np.linspace(-0.0, 0.0, num_parts)
    y_coeffs = np.linspace(-0.4, 0.4, num_parts)

    # First, move to pre-grasp pose (top-down).
    selected_object = get_target_object_from_obs(obs)
    assert selected_object is not None, "No target object selected"

    peg_height = 0.05

    while selected_object is not None:
        x, y, z = obs.part_poses[selected_object.name].position
        dz = 0.025 + peg_height * 2  # pre-grasp height
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
            vec_obs, reward, terminated, truncated, _ = env.step(action)
            # Collect trajectory data
            traj_observations.append(vec_obs.copy())
            traj_actions.append(action.copy())
            traj_rewards.append(float(reward))
            ep_terminated = ep_terminated or terminated
            ep_truncated = ep_truncated or truncated
            # NOTE: we should soon make this smoother.
            oc_obs = obs_space.devectorize(vec_obs)
            obs = Packing3DObjectCentricState(oc_obs.data, oc_obs.type_features)

        # Move down to grasp pose.
        sim.set_state(obs)
        current_end_effector_pose = sim.robot.arm.get_end_effector_pose()
        grasp_pose = Pose(
            (
                current_end_effector_pose.position[0],
                current_end_effector_pose.position[1],
                current_end_effector_pose.position[2] - peg_height - 0.02,
            ),
            current_end_effector_pose.orientation,
        )
        joint_plan = smoothly_follow_end_effector_path(
            sim.robot.arm,
            [current_end_effector_pose, grasp_pose],
            sim.robot.arm.get_joint_positions(),
            collision_ids=set(),
            joint_distance_fn=create_joint_distance_fn(sim.robot.arm),
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
            # NOTE: we should soon make this smoother.
            oc_obs = obs_space.devectorize(vec_obs)
            obs = Packing3DObjectCentricState(oc_obs.data, oc_obs.type_features)

        # Close the gripper to grasp.
        action = np.array([0.0] * 7 + [-1.0], dtype=np.float32)
        vec_obs, reward, terminated, truncated, _ = env.step(action)
        # Collect trajectory data
        traj_observations.append(vec_obs.copy())
        traj_actions.append(action.copy())
        traj_rewards.append(float(reward))
        ep_terminated = ep_terminated or terminated
        ep_truncated = ep_truncated or truncated
        # NOTE: we should soon make this smoother.
        oc_obs = obs_space.devectorize(vec_obs)
        obs = Packing3DObjectCentricState(oc_obs.data, oc_obs.type_features)

        assert obs.grasped_object == selected_object.name, "Object not grasped"

        # Move up slightly to break contact with the table.
        sim.set_state(obs)
        current_end_effector_pose = sim.robot.arm.get_end_effector_pose()
        post_grasp_pose = Pose(
            (
                current_end_effector_pose.position[0],
                current_end_effector_pose.position[1],
                current_end_effector_pose.position[2] + 0.1,
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
            vec_obs, reward, terminated, truncated, _ = env.step(action)
            # Collect trajectory data
            traj_observations.append(vec_obs.copy())
            traj_actions.append(action.copy())
            traj_rewards.append(float(reward))
            ep_terminated = ep_terminated or terminated
            ep_truncated = ep_truncated or truncated
            # NOTE: we should soon make this smoother.
            oc_obs = obs_space.devectorize(vec_obs)
            obs = Packing3DObjectCentricState(oc_obs.data, oc_obs.type_features)

        # Determine placement pose and pre-placement pose.
        # Place directly in the center of the target region for this test.
        placement_padding = 1e-3  # leave some room to prevent collisions with surface
        rack_pose = obs.rack_pose
        rack_half_extents = obs.rack_half_extents
        grasped_object = obs.grasped_object
        grasped_object_transform = obs.grasped_object_transform
        assert grasped_object is not None
        assert grasped_object_transform is not None
        block_placement_pose = Pose(
            (
                rack_pose.position[0] + x_coeffs[0] * rack_half_extents[0],
                rack_pose.position[1] + y_coeffs[0] * rack_half_extents[1],
                rack_pose.position[2]
                - obs.rack_half_extents[2]
                + 0.01
                + obs.get_object_half_extents_packing3d(grasped_object)[2]
                + placement_padding,
            ),
            obs.rack_pose.orientation,
        )
        end_effector_placement_pose = multiply_poses(
            block_placement_pose,
            grasped_object_transform,
        )
        end_effector_pre_placement_pose = Pose(
            (
                end_effector_placement_pose.position[0],
                end_effector_placement_pose.position[1],
                end_effector_placement_pose.position[2] + 0.1,
            ),
            end_effector_placement_pose.orientation,
        )

        # We don't really have to motion plan here because there
        # are no other objects, but in general we would motion plan.
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
            vec_obs, reward, terminated, truncated, _ = env.step(action)
            # Collect trajectory data
            traj_observations.append(vec_obs.copy())
            traj_actions.append(action.copy())
            traj_rewards.append(float(reward))
            ep_terminated = ep_terminated or terminated
            ep_truncated = ep_truncated or truncated
            # NOTE: we should soon make this smoother.
            oc_obs = obs_space.devectorize(vec_obs)
            obs = Packing3DObjectCentricState(oc_obs.data, oc_obs.type_features)

        # Open the gripper to finish the placement. Should trigger "done" (goal reached).
        action = np.array([0.0] * 7 + [1.0], dtype=np.float32)
        vec_obs, reward, done, truncated, _ = env.step(action)
        # Collect trajectory data
        traj_observations.append(vec_obs.copy())
        traj_actions.append(action.copy())
        traj_rewards.append(float(reward))
        ep_terminated = ep_terminated or done
        ep_truncated = ep_truncated or truncated
        # NOTE: we should soon make this smoother.
        oc_obs = obs_space.devectorize(vec_obs)
        obs = Packing3DObjectCentricState(oc_obs.data, oc_obs.type_features)
        assert obs.grasped_object is None, "Object not released"

        sim.set_state(obs)
        current_end_effector_pose = sim.robot.arm.get_end_effector_pose()
        end_effector_post_placement_pose = Pose(
            (
                current_end_effector_pose.position[0],
                current_end_effector_pose.position[1],
                current_end_effector_pose.position[2] + 0.05,
            ),
            current_end_effector_pose.orientation,
        )

        joint_plan = smoothly_follow_end_effector_path(
            sim.robot.arm,
            [
                current_end_effector_pose,
                end_effector_post_placement_pose,
                home_pos,
            ],
            sim.robot.arm.get_joint_positions(),
            collision_ids=sim._get_collision_object_ids(),  # pylint: disable=protected-access
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
            # NOTE: we should soon make this smoother.
            oc_obs = obs_space.devectorize(vec_obs)
            obs = Packing3DObjectCentricState(oc_obs.data, oc_obs.type_features)

        sim.set_state(obs)

        target_object = get_target_object_from_obs(obs)

        if target_object != selected_object:
            selected_object = target_object
            x_coeffs = x_coeffs[1:]
            y_coeffs = y_coeffs[1:]

    assert done, "Goal not reached"

    # Save trajectory to pickle file
    if SAVE_TRAJECTORIES and len(traj_actions) > 0:
        demo_path = save_demo(
            demo_dir=DEFAULT_DEMOS_DIR,
            env_id=f"kinder/Packing3D-p{num_parts}-v0",
            seed=seed,
            observations=traj_observations,
            actions=traj_actions,
            rewards=traj_rewards,
            terminated=ep_terminated,
            truncated=ep_truncated,
        )
        print(f"Trajectory saved to {demo_path}")
        print(f"  Observations: {len(traj_observations)}, Actions: {len(traj_actions)}")

    # Uncomment to debug.
    # import pybullet as p
    # from pybullet_helpers.gui import visualize_pose
    # visualize_pose(end_effector_placement_pose, env.physics_client_id)
    # while True:
    #     p.getMouseEvents(env.physics_client_id)
