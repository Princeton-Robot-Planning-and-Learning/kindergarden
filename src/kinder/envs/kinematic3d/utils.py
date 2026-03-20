"""Utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose, SE2Pose, set_pose
from pybullet_helpers.inverse_kinematics import check_body_collisions
from pybullet_helpers.joint import JointPositions
from relational_structs import Object, ObjectCentricState
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon

from kinder.envs.kinematic3d.object_types import Kinematic3DCuboidType
from kinder.envs.utils import RobotActionSpace

# Path to the default realistic background OBJ file
DEFAULT_REALISTIC_BG_PATH = (
    Path(__file__).parent / "assets" / "Stage_v3_sc1_staging.obj"
)


def load_realistic_background(
    physics_client_id: int,
    obj_path: str | Path,
    position=(0, 0, 0),
    orientation=(0, 0, 0, 1),
    scale=(1, 1, 1),
) -> int:
    """Load an OBJ file as a visual-only background in PyBullet.

    Args:
        physics_client_id: PyBullet physics client ID.
        obj_path: Path to the OBJ file.
        position: Base position of the mesh (x, y, z).
        orientation: Base orientation as quaternion (x, y, z, w).
        scale: Mesh scale (sx, sy, sz).

    Returns:
        Body ID of the loaded background mesh.
    """
    visual_id = p.createVisualShape(
        p.GEOM_MESH,
        fileName=str(obj_path),
        meshScale=scale,
        physicsClientId=physics_client_id,
    )

    body_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_id,
        baseCollisionShapeIndex=-1,  # visual only
        basePosition=position,
        baseOrientation=orientation,
        physicsClientId=physics_client_id,
    )

    # Disable collision
    p.setCollisionFilterGroupMask(body_id, -1, 0, 0, physicsClientId=physics_client_id)

    return body_id


class Kinematic3DObjectCentricState(ObjectCentricState):
    """A state in the Kinematic3D environment.

    Inherits from ObjectCentricState but adds some conveninent look ups.
    """

    @property
    def robot(self) -> Object:
        """Assumes there is a unique robot object named "robot"."""
        return self.get_object_from_name("robot")

    @property
    def joint_positions(self) -> JointPositions:
        """The robot joint positions."""
        joint_names = [f"joint_{i}" for i in range(1, 8)]
        return [self.get(self.robot, n) for n in joint_names]

    @property
    def finger_state(self) -> float:
        """The robot finger state."""
        return self.get(self.robot, "finger_state")

    @property
    def grasped_object(self) -> str | None:
        """The name of the currently grasped object, or None if there is none."""
        grasped_objs: list[Object] = []
        for obj in self.get_objects(Kinematic3DCuboidType):
            if self.get(obj, "grasp_active") > 0.5:
                grasped_objs.append(obj)
        if not grasped_objs:
            return None
        assert len(grasped_objs) == 1, "Multiple objects should not be grasped"
        grasped_obj = grasped_objs[0]
        return grasped_obj.name

    @property
    def grasped_object_transform(self) -> Pose | None:
        """The grasped object transform, or None if there is no grasped object."""
        if self.grasped_object is None:
            return None
        robot = self.robot
        x = self.get(robot, "grasp_tf_x")
        y = self.get(robot, "grasp_tf_y")
        z = self.get(robot, "grasp_tf_z")
        qx = self.get(robot, "grasp_tf_qx")
        qy = self.get(robot, "grasp_tf_qy")
        qz = self.get(robot, "grasp_tf_qz")
        qw = self.get(robot, "grasp_tf_qw")
        grasp_tf = Pose((x, y, z), (qx, qy, qz, qw))
        return grasp_tf

    @property
    def base_pose(self) -> SE2Pose:
        """The pose of the base."""
        robot = self.get_object_from_name("robot")
        se2_pose = SE2Pose(
            self.get(robot, "pos_base_x"),
            self.get(robot, "pos_base_y"),
            self.get(robot, "pos_base_rot"),
        )
        return se2_pose

    def get_object_half_extents(self, name: str) -> tuple[float, float, float]:
        """The half extents of the object."""
        obj = self.get_object_from_name(name)
        return (
            self.get(obj, "half_extent_x"),
            self.get(obj, "half_extent_y"),
            self.get(obj, "half_extent_z"),
        )

    def get_object_pose(self, name: str) -> Pose:
        """The pose of the object."""
        obj = self.get_object_from_name(name)
        position = (
            self.get(obj, "pose_x"),
            self.get(obj, "pose_y"),
            self.get(obj, "pose_z"),
        )
        orientation = (
            self.get(obj, "pose_qx"),
            self.get(obj, "pose_qy"),
            self.get(obj, "pose_qz"),
            self.get(obj, "pose_qw"),
        )
        return Pose(position, orientation)


class Kinematic3DRobotActionSpace(RobotActionSpace):
    """An action space for a mobile manipulation with a 7 DOF robot that can open and
    close its gripper.

    Actions are bounded relative base position, rotation, and joint positions, and open
    / close.

    The open / close logic is: <-0.5 is close, >0.5 is open, and otherwise no change.
    """

    def __init__(
        self,
        max_magnitude: float = 0.05,
    ) -> None:
        low = np.array([-max_magnitude] * 3 + [-max_magnitude] * 7 + [-1.0])
        high = np.array([max_magnitude] * 3 + [max_magnitude] * 7 + [1.0])
        super().__init__(low, high)

    def create_markdown_description(self) -> str:
        """Create a markdown description with a table of action space entries."""
        # pylint: disable=line-too-long
        return """An action space for mobile manipulation with a 7 DOF robot that can open and close its gripper.

Actions are bounded relative base position, rotation, and joint positions, and open / close.

| **Index** | **Description** |
| --- | --- |
| 0 | delta base x |
| 1 | delta base y |
| 2 | delta base rotation |
| 3 | delta joint 1 |
| 4 | delta joint 2 |
| 5 | delta joint 3 |
| 6 | delta joint 4 |
| 7 | delta joint 5 |
| 8 | delta joint 6 |
| 9 | delta joint 7 |
| 10 | gripper open/close |

The open / close logic is: <-0.5 is close, >0.5 is open, and otherwise no change.
"""


def extend_joints_to_include_fingers(joint_positions: JointPositions) -> JointPositions:
    """Add 6 DOF for fingers."""
    assert len(joint_positions) == 7
    finger_joints = [0.0] * 6
    return list(joint_positions) + finger_joints


def remove_fingers_from_extended_joints(
    joint_positions: JointPositions,
) -> JointPositions:
    """Inverse of _extend_joints_to_include_fingers()."""
    assert len(joint_positions) == 13
    return joint_positions[:7]


def get_robot_action_from_gui_input(
    action_space: Kinematic3DRobotActionSpace, gui_input: dict[str, Any]
) -> NDArray[np.float32]:
    """Get the mapping from human inputs to actions, derived from action space."""
    # This will be implemented later - placeholder for now
    keys_pressed = gui_input["keys"]
    right_x, right_y = gui_input["right_stick"]
    left_x, left_y = gui_input["left_stick"]

    # Initialize the action.
    low = action_space.low
    high = action_space.high
    action = np.zeros(action_space.shape, action_space.dtype)

    def _rescale(x: float, lb: float, ub: float) -> float:
        """Rescale from [-1, 1] to [lb, ub]."""
        return lb + (x + 1) * (ub - lb) / 2

    # The right stick controls the x, y movement of the base.
    action[0] = _rescale(right_x, low[0], high[0])
    action[1] = _rescale(right_y, low[1], high[1])

    # The left stick controls the rotation of the base. Only the x axis
    # is used right now.
    action[2] = _rescale(left_x, low[2], high[2])
    action[4] = _rescale(left_y, low[4], high[4])

    # The w/s mouse keys are used to adjust the robot arm.
    if "a" in keys_pressed:
        action[5] = low[5]
    if "s" in keys_pressed:
        action[5] = high[5]

    # The space bar is used to close the gripper.
    # Open the gripper by default.
    if "d" in keys_pressed:
        action[6] = low[6]
    if "f" in keys_pressed:
        action[6] = high[6]

    return action


def is_on_top(
    poseA: Pose,
    halfA: tuple[float, float, float],
    poseB: Pose,
    halfB: tuple[float, float, float],
) -> bool:
    """Check if box B is on top of box A.

    Args:
        poseA: Pose of box A. (position, orientation as quaternion).
        halfA: Half extents of box A (hx, hy, hz).
        poseB: Pose of box B. (position, orientation as quaternion).
        halfB: Half extents of box B (hx, hy, hz).
    Returns:
        True if box B is on top of box A, False otherwise.
    """
    posA, quatA = poseA.position, poseA.orientation
    posB, quatB = poseB.position, poseB.orientation

    R_A = Rotation.from_quat(quatA).as_matrix()
    R_B = Rotation.from_quat(quatB).as_matrix()

    def corners(
        pos: tuple[float, float, float], R: np.ndarray, h: np.ndarray
    ) -> np.ndarray:
        local = np.array(
            [
                [sx * h[0], sy * h[1], sz * h[2]]
                for sx in [-1, 1]
                for sy in [-1, 1]
                for sz in [-1, 1]
            ]
        )
        return (R @ local.T).T + pos

    C_A = corners(posA, R_A, np.array(halfA))
    C_B = corners(posB, R_B, np.array(halfB))

    PA = Polygon(C_A[:, :2]).convex_hull
    PB = Polygon(C_B[:, :2]).convex_hull

    overlap = PA.intersects(PB)
    zA_max, zB_min = np.max(C_A[:, 2]), np.min(C_B[:, 2])

    return overlap and (zB_min >= zA_max)


def is_inside(
    poseA: Pose,
    halfA: tuple[float, float, float],
    poseB: Pose,
    halfB: tuple[float, float, float],
) -> bool:
    """Check if box B is inside box A.

    Args:
        poseA: Pose of box A. (position, orientation as quaternion).
        halfA: Half extents of box A (hx, hy, hz).
        poseB: Pose of box B. (position, orientation as quaternion).
        halfB: Half extents of box B (hx, hy, hz).
    Returns:
        True if box B is inside box A, False otherwise.
    """

    posA, quatA = poseA.position, poseA.orientation
    posB, quatB = poseB.position, poseB.orientation

    R_A = Rotation.from_quat(quatA).as_matrix()
    R_B = Rotation.from_quat(quatB).as_matrix()

    def corners(
        pos: tuple[float, float, float], R: np.ndarray, h: np.ndarray
    ) -> np.ndarray:
        local = np.array(
            [
                [sx * h[0], sy * h[1], sz * h[2]]
                for sx in [-1, 1]
                for sy in [-1, 1]
                for sz in [-1, 1]
            ]
        )
        return (R @ local.T).T + pos

    C_A = corners(posA, R_A, np.array(halfA))
    C_B = corners(posB, R_B, np.array(halfB))

    PA = Polygon(C_A[:, :2]).convex_hull
    PB = Polygon(C_B[:, :2]).convex_hull

    return PA.contains(PB)


def sample_collision_free_object_poses(
    object_ids: set[int],
    lb: tuple[float, float, float],
    ub: tuple[float, float, float],
    physics_client_id: int,
    rng: np.random.Generator,
    other_collision_ids: set[int],
    max_sampling_attempts: int = 100_000,
    use_box: bool = False,
    box_pose: Pose | None = None,
    table_pose: Pose | None = None,
    table_half_extents: tuple[float, float, float] | None = None,
    box_half_extents: tuple[float, float, float] | None = None,
) -> None:
    """Randomly reset the poses of objects in-place while avoiding collisions.

    NOTE: orientations not currently sampled.
    """
    collision_ids = set(other_collision_ids)

    for obj_id in sorted(object_ids):
        for _ in range(max_sampling_attempts):
            x, y, z = rng.uniform(lb, ub)
            pose = Pose((x, y, z))
            if use_box:
                assert (
                    table_pose is not None
                    and box_pose is not None
                    and table_half_extents is not None
                    and box_half_extents is not None
                )
                # Distance to closest edge of table (2D)
                point_2d = pose.position[:2]
                table_center_2d = table_pose.position[:2]
                table_he_2d = table_half_extents[:2]

                dx_table = max(
                    abs(point_2d[0] - table_center_2d[0]) - table_he_2d[0], 0.0
                )
                dy_table = max(
                    abs(point_2d[1] - table_center_2d[1]) - table_he_2d[1], 0.0
                )
                dist_to_table_edge = np.sqrt(dx_table**2 + dy_table**2)

                # Distance to closest edge of box (2D)
                box_center_2d = box_pose.position[:2]
                box_he_2d = box_half_extents[:2]

                dx_box = max(abs(point_2d[0] - box_center_2d[0]) - box_he_2d[0], 0.0)
                dy_box = max(abs(point_2d[1] - box_center_2d[1]) - box_he_2d[1], 0.0)
                dist_to_box_edge = np.sqrt(dx_box**2 + dy_box**2)

                # Require block to be closer to box edge than table edge
                if dist_to_box_edge >= dist_to_table_edge + 0.05:
                    continue
            set_pose(obj_id, pose, physics_client_id)

            if not any(
                check_body_collisions(obj_id, cid, physics_client_id)
                for cid in collision_ids
            ):
                collision_ids.add(obj_id)
                break
        else:
            raise RuntimeError(
                f"Failed to sample collision-free pose for object {obj_id}"
            )
