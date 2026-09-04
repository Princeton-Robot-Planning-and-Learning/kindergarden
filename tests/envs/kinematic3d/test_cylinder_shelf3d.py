"""Tests for cylinder_shelf3d.py."""

# pylint: disable=protected-access

import numpy as np
import pytest
from pybullet_helpers.geometry import Pose, SE2Pose, set_pose
from pybullet_helpers.inverse_kinematics import inverse_kinematics
from pybullet_helpers.joint import get_jointwise_difference
from scipy.spatial.transform import Rotation

from kinder.envs.kinematic3d.cylinder_shelf3d import (
    CYLINDER_OBJECT_TYPE,
    CylinderShelf3DEnv,
    CylinderShelf3DEnvConfig,
    ObjectCentricCylinderShelf3DEnv,
)


@pytest.fixture(scope="module")
def env():
    """Create a shared object-centric environment for tests in this module."""
    environment = ObjectCentricCylinderShelf3DEnv(
        num_cylinders=2,
        use_gui=False,
        realistic_bg=False,
    )
    yield environment
    environment.close()


def test_cylinder_shelf3d_env(env):  # pylint: disable=redefined-outer-name
    """Basic reset / step / observation checks."""
    obs, _ = env.reset(seed=123)

    # Cylinders have per-index heights, cycled from the config, and are
    # tagged as cylinders in the state.
    for idx in range(2):
        obj = obs.get_object_from_name(f"cylinder{idx}")
        expected_half_height = env.config.get_cylinder_height(idx) / 2
        assert np.isclose(obs.get(obj, "half_extent_z"), expected_half_height)
        assert np.isclose(obs.get(obj, "half_extent_x"), env.config.cylinder_radius)
        assert np.isclose(obs.get(obj, "object_type"), CYLINDER_OBJECT_TYPE)
        # Resting on the ground.
        assert np.isclose(obs.get(obj, "pose_z"), expected_half_height)

    for _ in range(5):
        act = env.action_space.sample()
        obs, _, _, _, _ = env.step(act)


def test_cylinder_shelf3d_gym_registration():
    """The gym-registered variants construct and reset."""
    # pylint: disable=import-outside-toplevel
    import kinder

    kinder.register_all_environments()
    gym_env = kinder.make("kinder/KinematicCylinderShelf3D-o1-v0", realistic_bg=False)
    obs, _ = gym_env.reset(seed=0)
    assert isinstance(obs, np.ndarray)
    gym_env.close()


def _side_grasp_orientation(
    approach_yaw: float, approach_pitch: float = np.deg2rad(15)
) -> tuple[float, ...]:
    """End-effector orientation for a (near-)horizontal side grasp.

    Tool z (the approach axis) points along `approach_yaw`, tilted down by
    `approach_pitch`; tool x (the finger-closing axis) is horizontal and
    perpendicular to the approach, so the fingers straddle a vertical
    cylinder. The slight downward pitch widens the arm's reachable
    envelope at low grasp heights.
    """
    approach = np.array(
        [
            np.cos(approach_yaw) * np.cos(approach_pitch),
            np.sin(approach_yaw) * np.cos(approach_pitch),
            -np.sin(approach_pitch),
        ]
    )
    perp = np.array([-np.sin(approach_yaw), np.cos(approach_yaw), 0.0])
    rot_matrix = np.column_stack([perp, np.cross(approach, perp), approach])
    return tuple(Rotation.from_matrix(rot_matrix).as_quat())


def _drive_arm_to_joints(environment, obs, target_joints, max_steps=200, step_mag=0.05):
    """Step the env along the joint-space line to the target.

    The delta is scaled uniformly (rather than clipped per-joint) so the executed path
    follows the straight joint-space line between IK solutions. Stops early if a
    collision revert blocks progress; callers check the resulting state.
    """
    joint_infos = environment.robot.arm.get_arm_joint_infos()[:7]
    for _ in range(max_steps):
        delta = np.array(
            get_jointwise_difference(
                joint_infos, target_joints[:7], obs.joint_positions
            )
        )
        max_abs = np.max(np.abs(delta))
        if max_abs < 1e-4:
            return obs
        previous = np.array(obs.joint_positions)
        scaled = delta * min(1.0, step_mag / max_abs)
        action = np.array([0.0] * 3 + list(scaled) + [0.0], dtype=np.float32)
        obs, _, _, _, _ = environment.step(action)
        if np.allclose(previous, obs.joint_positions):
            # Collision revert; no further progress possible.
            return obs
    return obs


def test_side_grasp(env):  # pylint: disable=redefined-outer-name
    """A horizontal side grasp near the cylinder top succeeds."""
    obs, _ = env.reset(seed=123)

    cylinder_pose = obs.get_object_pose("cylinder0")
    cx, cy, _ = cylinder_pose.position
    height = env.config.get_cylinder_height(0)

    # Drive the base to face the cylinder from 0.6 m away, in steps bounded
    # by the action magnitude limit.
    approach_yaw = np.arctan2(cy, cx)
    target_base = SE2Pose(
        cx - 0.6 * np.cos(approach_yaw),
        cy - 0.6 * np.sin(approach_yaw),
        approach_yaw,
    )
    for _ in range(100):
        delta = target_base - obs.base_pose
        coords = [delta.x, delta.y, delta.rot]
        if np.max(np.abs(coords)) < 1e-4:
            break
        coords = np.clip(coords, -env.config.max_action_mag, env.config.max_action_mag)
        action = np.array(list(coords) + [0.0] * 7 + [0.0], dtype=np.float32)
        obs, _, _, _, _ = env.step(action)

    # Side grasp near the top of the cylinder, approached along the (slightly
    # pitched) approach axis. The end effector stops 2 cm short of the
    # cylinder axis so the fingers straddle the cylinder without contact,
    # while the grasp-detection zone still overlaps it.
    orientation = _side_grasp_orientation(approach_yaw)
    pitch = np.deg2rad(15)
    approach = np.array(
        [
            np.cos(approach_yaw) * np.cos(pitch),
            np.sin(approach_yaw) * np.cos(pitch),
            -np.sin(pitch),
        ]
    )
    axis_point = np.array([cx, cy, height - 0.05])
    grasp = Pose(tuple(axis_point - 0.02 * approach), orientation)
    pre_grasp = Pose(tuple(axis_point - 0.12 * approach), orientation)

    pre_grasp_joints = inverse_kinematics(
        env.robot.arm, pre_grasp, validate=True, set_joints=False
    )
    obs = _drive_arm_to_joints(env, obs, pre_grasp_joints)
    grasp_joints = inverse_kinematics(
        env.robot.arm, grasp, validate=True, set_joints=False
    )
    obs = _drive_arm_to_joints(env, obs, grasp_joints)

    # Close the gripper.
    for _ in range(5):
        action = np.array([0.0] * 10 + [-1.0], dtype=np.float32)
        obs, _, _, _, _ = env.step(action)
    assert obs.grasped_object == "cylinder0"

    # Retract; the cylinder comes along.
    obs = _drive_arm_to_joints(env, obs, env.config.initial_joints)
    assert obs.grasped_object == "cylinder0"
    obj = obs.get_object_from_name("cylinder0")
    assert obs.get(obj, "pose_z") > 0.3


def test_gym_env_wrapper():
    """The ConstantObjectKinDEREnv wrapper vectorizes and devectorizes."""
    gym_env = CylinderShelf3DEnv(num_cylinders=1, realistic_bg=False)
    vec_obs, _ = gym_env.reset(seed=42)
    state = gym_env.observation_space.devectorize(vec_obs)
    assert state.get_object_from_name("cylinder0") is not None
    assert state.get_object_from_name("shelf") is not None
    gym_env.close()


def test_omitted_layers_shelf_and_goal():
    """A shelf with the bottom inner board removed has one tall opening, and a cylinder
    standing on the bottom board inside the footprint reaches the goal."""
    config = CylinderShelf3DEnvConfig(
        shelf_omitted_layers=(1,),
        cylinder_heights=(0.233,),
        cylinder_radius=0.039,
    )
    assert config.get_present_layer_indices() == [0, 2, 3]
    openings = config.get_layer_openings()
    # The bottom opening spans up to the second present board.
    assert openings[0][1] == pytest.approx(
        2 * (config.shelf_spacing + config.shelf_height) - config.shelf_height
    )
    assert openings[-1][1] == float("inf")

    environment = ObjectCentricCylinderShelf3DEnv(
        num_cylinders=1, config=config, realistic_bg=False
    )
    environment.reset(seed=0)
    # Stand the cylinder on the bottom board, inside the footprint.
    shelf_x, shelf_y, _ = config.shelf_pose.position
    resting_z = config.get_layer_surface_zs()[0] + 0.233 / 2
    set_pose(
        environment._cylinders["cylinder0"],
        Pose((shelf_x, shelf_y, resting_z)),
        environment.physics_client_id,
    )
    assert environment.goal_reached()
    # On the floor next to the shelf: not a goal state.
    set_pose(
        environment._cylinders["cylinder0"],
        Pose((shelf_x, shelf_y - 1.0, 0.233 / 2)),
        environment.physics_client_id,
    )
    assert not environment.goal_reached()
    environment.close()


def test_real_restock_layout():
    """Explicit board heights, per-cylinder radii, staging boxes and init regions
    together model a measured physical restock scene: non-uniformly spaced boards
    (a big bottom opening for tall items), mixed-diameter cylinders starting inside
    open-top boxes whose walls are collision bodies."""
    board_half = 0.0127 / 2
    config = CylinderShelf3DEnvConfig(
        shelf_pose=Pose((1.5, 1.5, 0.0)),
        # Board top surfaces at 0.100, 0.538 and 0.800 m above the floor.
        shelf_layer_zs=(0.100 - board_half, 0.538 - board_half, 0.800 - board_half),
        cylinder_heights=(0.29, 0.208, 0.12),
        cylinder_radii=(0.0375, 0.0375, 0.0325),
        boxes=(
            (0.30, 0.70, 0.66, 0.98, 0.115, 0.2),  # rotated 0.2 rad about its center
            (0.30, 0.6975, 0.9325, 1.23, 0.215),
        ),
        cylinder_init_regions=(
            (0.36, 0.64, 0.99, 1.17),
            (0.36, 0.64, 0.99, 1.17),
            (0.36, 0.64, 0.72, 0.92),
        ),
    )
    surfaces = config.get_layer_surface_zs()
    assert surfaces == pytest.approx([0.100, 0.538, 0.800])
    openings = config.get_layer_openings()
    # Tall opening ~0.425, short opening ~0.249.
    assert openings[0][1] == pytest.approx(0.538 - 0.0127 - 0.100)
    assert openings[1][1] == pytest.approx(0.800 - 0.0127 - 0.538)
    assert config.get_cylinder_radius(2) == pytest.approx(0.0325)

    environment = ObjectCentricCylinderShelf3DEnv(
        num_cylinders=3, config=config, realistic_bg=False
    )
    try:
        # Four walls per box, all in the collision set.
        assert len(environment._box_wall_ids) == 8
        assert set(environment._box_wall_ids) <= environment._get_collision_object_ids()
        # Per-cylinder half extents follow the per-cylinder radii.
        assert environment._get_half_extents("cylinder2")[0] == pytest.approx(0.0325)
        obs, _ = environment.reset(seed=0)
        # Every cylinder spawned inside its configured region.
        for idx in range(3):
            x_lb, x_ub, y_lb, y_ub = config.get_cylinder_init_region(idx)
            pose = obs.get_object_pose(f"cylinder{idx}")
            assert x_lb <= pose.position[0] <= x_ub
            assert y_lb <= pose.position[1] <= y_ub
        # Goal check works off the explicit surfaces.
        shelf_x, shelf_y, _ = config.shelf_pose.position
        for idx, surface_index in ((0, 0), (1, 0), (2, 1)):
            height = config.get_cylinder_height(idx)
            set_pose(
                environment._cylinders[f"cylinder{idx}"],
                Pose(
                    (
                        shelf_x - 0.15 + 0.15 * idx,
                        shelf_y,
                        surfaces[surface_index] + height / 2,
                    )
                ),
                environment.physics_client_id,
            )
        assert environment.goal_reached()
    finally:
        environment.close()
