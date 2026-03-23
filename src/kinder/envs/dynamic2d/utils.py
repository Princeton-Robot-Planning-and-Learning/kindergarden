"""Utilities for Dynamic2D PyMunk-based environments."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pymunk
from numpy.typing import NDArray
from prpl_utils.motion_planning import BiRRT
from prpl_utils.utils import get_signed_angle_distance, wrap_angle
from pymunk import Body, Shape
from pymunk.vec2d import Vec2d
from relational_structs import (
    Object,
    ObjectCentricState,
)
from tomsgeoms2d.structs import Rectangle

from kinder.envs.dynamic2d.object_types import (
    KinRectangleType,
)
from kinder.envs.kinematic2d.structs import (
    MultiBody2D,
    SE2Pose,
    ZOrder,
)
from kinder.envs.utils import (
    BLACK,
    RobotActionSpace,
    get_se2_pose,
    kin_robot_to_multibody2d,
    state_2d_has_collision,
)

# Collision types from the basic_pymunk.py script
STATIC_COLLISION_TYPE = 0
DYNAMIC_COLLISION_TYPE = 1
ROBOT_COLLISION_TYPE = 2
FINGER_COLLISION_TYPE = 3
ARM_COLLISION_TYPE = 4
NON_GRASPABLE_COLLISION_TYPE = 5  # For small objects that cannot be grasped


class KinRobotActionSpace(RobotActionSpace):
    """An action space for a fingered robot with gripper control.

    Actions are bounded relative movements of the base, arm extension, and gripper
    opening/closing.
    """

    def __init__(
        self,
        min_dx: float = -5e-1,
        max_dx: float = 5e-1,
        min_dy: float = -5e-1,
        max_dy: float = 5e-1,
        min_dtheta: float = -np.pi / 16,
        max_dtheta: float = np.pi / 16,
        min_darm: float = -1e-1,
        max_darm: float = 1e-1,
        min_dgripper: float = -0.02,
        max_dgripper: float = 0.02,
    ) -> None:
        low = np.array([min_dx, min_dy, min_dtheta, min_darm, min_dgripper])
        high = np.array([max_dx, max_dy, max_dtheta, max_darm, max_dgripper])
        super().__init__(low, high, dtype=np.float64)

    def create_markdown_description(self) -> str:
        """Create a human-readable markdown description of this space."""
        features = [
            ("dx", "Change in robot x position (positive is right)"),
            ("dy", "Change in robot y position (positive is up)"),
            ("dtheta", "Change in robot angle in radians (positive is ccw)"),
            ("darm", "Change in robot arm length (positive is out)"),
            ("dgripper", "Change in gripper gap (positive is open)"),
        ]
        md_table_str = (
            "| **Index** | **Feature** | **Description** | **Min** | **Max** |"
        )
        md_table_str += "\n| --- | --- | --- | --- | --- |"
        for idx, (feature, description) in enumerate(features):
            lb = self.low[idx]
            ub = self.high[idx]
            md_table_str += (
                f"\n| {idx} | {feature} | {description} | {lb:.3f} | {ub:.3f} |"
            )
        return (
            f"The entries of an array in this Box space correspond to the "
            f"following action features:\n{md_table_str}\n"
        )


class DotRobotActionSpace(RobotActionSpace):
    """An action space for a simple dot robot (kinematic circle).

    Actions are bounded delta positions in 2D space.
    """

    def __init__(
        self,
        min_dx: float = -0.1,
        max_dx: float = 0.1,
        min_dy: float = -0.1,
        max_dy: float = 0.1,
    ) -> None:
        low = np.array([min_dx, min_dy])
        high = np.array([max_dx, max_dy])
        super().__init__(low, high, dtype=np.float64)

    def create_markdown_description(self) -> str:
        """Create a human-readable markdown description of this space."""
        features = [
            ("dx", "Delta x position for robot (positive is right)"),
            ("dy", "Delta y position for robot (positive is up)"),
        ]
        md_table_str = (
            "| **Index** | **Feature** | **Description** | **Min** | **Max** |"
        )
        md_table_str += "\n| --- | --- | --- | --- | --- |"
        for idx, (feature, description) in enumerate(features):
            lb = self.low[idx]
            ub = self.high[idx]
            md_table_str += (
                f"\n| {idx} | {feature} | {description} | {lb:.3f} | {ub:.3f} |"
            )
        return (
            f"The entries of an array in this Box space correspond to the "
            f"following action features:\n{md_table_str}\n"
        )


class DotRobot:
    """Simple dot robot implementation using PyMunk physics engine.

    The robot is a kinematic circle that can move to target positions using PD control.
    This is similar to the agent in the original PushT environment.
    """

    def __init__(
        self,
        init_pos: Vec2d = Vec2d(256.0, 256.0),
        radius: float = 15.0,
    ) -> None:
        # Robot parameters
        self.radius = radius

        # Track last robot state
        self._position = init_pos

        # Body and shape references
        self._body: pymunk.Body | None = None
        self._shape: pymunk.Shape | None = None
        self.create_body()

    def add_to_space(self, space: pymunk.Space) -> None:
        """Add robot to the PyMunk space."""
        assert self._body is not None and self._shape is not None
        space.add(self._body, self._shape)

    def create_body(self) -> None:
        """Create the robot body (kinematic circle)."""
        self._body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self._shape = pymunk.Circle(self._body, self.radius)
        self._shape.color = (50, 50, 255, 255)  # Blue circle
        self._shape.friction = 1
        self._shape.collision_type = ROBOT_COLLISION_TYPE
        self._shape.density = 1.0
        self._body.position = self._position

    @property
    def pose(self) -> SE2Pose:
        """Get the robot pose as SE2Pose."""
        assert self._body is not None
        return SE2Pose(
            x=self._body.position.x,
            y=self._body.position.y,
            theta=0.0,  # Dot robot has no orientation
        )

    @property
    def vel(self) -> Vec2d:
        """Get the robot velocity."""
        assert self._body is not None
        return self._body.velocity

    @property
    def body_id(self) -> int:
        """Get the body id in pymunk space."""
        assert self._body is not None
        return self._body.id

    def reset_position(
        self,
        x: float,
        y: float,
        vel: Vec2d | None = None,
    ) -> None:
        """Reset robot to specified position with optional velocity."""
        assert self._body is not None
        self._body.position = (x, y)
        if vel is not None:
            self._body.velocity = vel
        else:
            self._body.velocity = Vec2d(0.0, 0.0)

        # Update last state
        self.update_last_state()

    def revert_to_last_state(self) -> None:
        """Reset to last state and stay static when collide with static objects."""
        assert self._body is not None
        self._body.position = self._position
        self._body.velocity = Vec2d(0.0, 0.0)

    def update_last_state(self) -> None:
        """Update the last state tracking variables."""
        assert self._body is not None
        self._position = Vec2d(self._body.position.x, self._body.position.y)

    def update(self, velocity: Vec2d) -> None:
        """Update the robot velocity."""
        assert self._body is not None
        # Update robot last state
        self.update_last_state()
        # Update velocity
        self._body.velocity = velocity


class DotRobotPDController:
    """A simple PD controller for the DotRobot."""

    def __init__(
        self,
        kp: float = 100.0,
        kv: float = 20.0,
    ) -> None:
        self.kp = kp
        self.kv = kv

    def compute_control(
        self,
        robot: DotRobot,
        tgt_x: float,
        tgt_y: float,
        dt: float,
    ) -> Vec2d:
        """Compute velocity using PD control."""
        # Read current state
        curr_pos = Vec2d(robot.pose.x, robot.pose.y)
        curr_vel = robot.vel

        # PD control
        tgt_pos = Vec2d(tgt_x, tgt_y)
        zero_vel = Vec2d(0, 0)
        acceleration = self.kp * (tgt_pos - curr_pos) + self.kv * (zero_vel - curr_vel)
        new_vel = curr_vel + acceleration * dt

        return new_vel


class KinRobot:
    """Robot implementation using PyMunk physics engine with four bodies.

    The robot has a circular base, a rectangular gripper base (attached one arm and one
    right finger), and a left fingers. The gripper base is attached to the robot base
    via a kinematic arm that can extend and retract. The fingers can open and close to
    grasp objects.

    The robot can held objects by closing the fingers around them.

    The robot will be revert to the last valid state when colliding with static objects.

    The robot is controlled via setting the velocities of the bodies, which can be
    computed using a PD controller.
    """

    def __init__(
        self,
        init_pos: Vec2d = Vec2d(5.0, 5.0),
        base_radius: float = 0.4,
        arm_length_max: float = 0.8,
        gripper_base_width: float = 0.01,
        gripper_base_height: float = 0.1,
        gripper_finger_width: float = 0.1,
        gripper_finger_height: float = 0.01,
        finger_move_thresh: float = 0.001,
        grasping_theta_thresh: float = 0.1,
        base_collision_type: int = ROBOT_COLLISION_TYPE,
        arm_collision_type: int = ARM_COLLISION_TYPE,
        finger_collision_type: int = FINGER_COLLISION_TYPE,
    ) -> None:
        # Robot parameters
        self.base_radius = base_radius
        self.gripper_base_width = gripper_base_width
        self.gripper_base_height = gripper_base_height
        self.gripper_finger_width = gripper_finger_width
        self.gripper_finger_height = gripper_finger_height
        self.arm_length_max = arm_length_max
        self.gripper_gap_max = gripper_base_height
        self.finger_move_thresh = finger_move_thresh
        self.grasping_theta_thresh = grasping_theta_thresh
        self.base_collision_type = base_collision_type
        self.arm_collision_type = arm_collision_type
        self.finger_collision_type = finger_collision_type

        # Track last robot state
        self._base_position = init_pos
        self._base_angle = 0.0
        self._arm_length = base_radius
        self._gripper_gap = gripper_base_height
        self.held_objects: list[tuple[tuple[Body, list[Shape]], float, SE2Pose]] = []

        # Updated by env.step()
        self.is_opening_finger = False
        self.is_closing_finger = False

        # Body and shape references
        self.create_base()
        self.create_gripper_base()
        (self._left_finger_body, self._left_finger_shape), (
            self._right_finger_body,
            self._right_finger_shape,
        ) = self.create_finger()

    def add_to_space(self, space: pymunk.Space) -> None:
        """Add robot components to the PyMunk space."""
        space.add(self._base_body, self._base_shape)
        space.add(self._gripper_base_body, self._gripper_base_shape, self._arm_shape)
        space.add(self._left_finger_body, self._left_finger_shape)
        space.add(self._right_finger_body, self._right_finger_shape)

    def create_base(self) -> None:
        """Create the robot base."""
        self._base_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self._base_shape = pymunk.Circle(self._base_body, self.base_radius)
        self._base_shape.color = (255, 50, 50, 255)
        self._base_shape.friction = 1
        self._base_shape.collision_type = self.base_collision_type
        self._base_shape.density = 1.0
        self._base_body.angle = self._base_angle
        self._base_body.position = self._base_position

    @property
    def base_pose(self) -> SE2Pose:
        """Get the base pose as SE2Pose."""
        return SE2Pose(
            x=self._base_body.position.x,
            y=self._base_body.position.y,
            theta=self._base_body.angle,
        )

    @property
    def base_vel(self) -> tuple[Vec2d, float]:
        """Get the base linear and angular velocity."""
        return self._base_body.velocity, self._base_body.angular_velocity

    def create_gripper_base(self) -> None:
        """Create the gripper base."""
        half_w = self.gripper_base_width / 2
        half_h = self.gripper_base_height / 2
        vs = [
            (-half_w, half_h),
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
        ]
        self._gripper_base_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self._gripper_base_shape = pymunk.Poly(self._gripper_base_body, vs)
        self._gripper_base_shape.friction = 1
        self._gripper_base_shape.collision_type = self.arm_collision_type
        self._gripper_base_shape.density = 1.0

        vs_arm = [
            (-self.arm_length_max / 2, half_w),
            (-self.arm_length_max / 2, -half_w),
            (self.arm_length_max / 2, -half_w),
            (self.arm_length_max / 2, half_w),
        ]
        ts_arm = pymunk.Transform(tx=-self.arm_length_max / 2 - half_w, ty=0)
        self._arm_shape = pymunk.Poly(self._gripper_base_body, vs_arm, transform=ts_arm)
        self._arm_shape.friction = 1
        self._arm_shape.collision_type = self.arm_collision_type
        self._arm_shape.density = 1.0

        init_rel_pos = SE2Pose(x=self._arm_length, y=0.0, theta=0.0)
        init_pose = self.base_pose * init_rel_pos
        self._gripper_base_body.angle = init_pose.theta
        self._gripper_base_body.position = (init_pose.x, init_pose.y)

    @property
    def gripper_base_pose(self) -> SE2Pose:
        """Get the gripper base pose as SE2Pose."""
        return SE2Pose(
            x=self._gripper_base_body.position.x,
            y=self._gripper_base_body.position.y,
            theta=self._gripper_base_body.angle,
        )

    @property
    def gripper_base_vel(self) -> tuple[Vec2d, float]:
        """Get the gripper base linear and angular velocity."""
        return (
            self._gripper_base_body.velocity,
            self._gripper_base_body.angular_velocity,
        )

    def create_finger(
        self,
    ) -> tuple[tuple[pymunk.Body, pymunk.Shape], tuple[pymunk.Body, pymunk.Shape]]:
        """Create two gripper fingers."""
        half_w = self.gripper_finger_width / 2
        half_h = self.gripper_finger_height / 2
        vs = [
            (-half_w, half_h),
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
        ]
        finger_body_l = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        finger_shape_l = pymunk.Poly(finger_body_l, vs)
        finger_shape_l.friction = 1
        finger_shape_l.density = 1.0
        finger_shape_l.collision_type = self.finger_collision_type

        init_rel_pos = SE2Pose(x=half_w, y=self._gripper_gap / 2, theta=0.0)
        init_pose = self.gripper_base_pose * init_rel_pos
        finger_body_l.angle = init_pose.theta
        finger_body_l.position = (init_pose.x, init_pose.y)

        finger_body_r = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        finger_shape_r = pymunk.Poly(finger_body_r, vs)
        finger_shape_r.friction = 1
        finger_shape_r.density = 1.0
        # NOTE: Right finger uses arm collision type to disable grasping
        # collision handler, otherwise the two fingers collision will replicate.
        finger_shape_r.collision_type = self.arm_collision_type

        init_rel_pos = SE2Pose(x=half_w, y=-self._gripper_gap / 2, theta=0.0)
        init_pose = self.gripper_base_pose * init_rel_pos
        finger_body_r.angle = init_pose.theta
        finger_body_r.position = (init_pose.x, init_pose.y)
        return (finger_body_l, finger_shape_l), (finger_body_r, finger_shape_r)

    @property
    def finger_poses_l(self) -> SE2Pose:
        """Get the left finger pose as SE2Pose."""
        return SE2Pose(
            x=self._left_finger_body.position.x,
            y=self._left_finger_body.position.y,
            theta=self._left_finger_body.angle,
        )

    @property
    def finger_poses_r(self) -> SE2Pose:
        """Get the right finger pose as SE2Pose."""
        return SE2Pose(
            x=self._right_finger_body.position.x,
            y=self._right_finger_body.position.y,
            theta=self._right_finger_body.angle,
        )

    @property
    def finger_vel_l(self) -> tuple[Vec2d, float]:
        """Get the left finger linear and angular velocity."""
        return (
            self._left_finger_body.velocity,
            self._left_finger_body.angular_velocity,
        )

    @property
    def finger_vel_r(self) -> tuple[Vec2d, float]:
        """Get the right finger linear and angular velocity."""
        return (
            self._right_finger_body.velocity,
            self._right_finger_body.angular_velocity,
        )

    @property
    def held_object_vels(self) -> list[tuple[Vec2d, float]]:
        """Get the held object linear and angular velocity."""
        vel_list = []
        for obj, _, _ in self.held_objects:
            obj_body, _ = obj
            vel_list.append((obj_body.velocity, obj_body.angular_velocity))
        return vel_list

    @property
    def curr_gripper(self) -> float:
        """Get the current gripper opening."""
        relative_finger_pose_l = self.gripper_base_pose.inverse * self.finger_poses_l
        relative_finger_pose_r = self.gripper_base_pose.inverse * self.finger_poses_r
        return relative_finger_pose_l.y - relative_finger_pose_r.y

    @property
    def curr_l_finger_gap(self) -> float:
        """Get the current left finger gap from gripper base."""
        relative_finger_pose_l = self.gripper_base_pose.inverse * self.finger_poses_l
        return relative_finger_pose_l.y

    @property
    def curr_r_finger_gap(self) -> float:
        """Get the current right finger gap from gripper base."""
        relative_finger_pose_r = self.gripper_base_pose.inverse * self.finger_poses_r
        return relative_finger_pose_r.y

    @property
    def curr_arm_length(self) -> float:
        """Get the current arm length."""
        relative_pose = self.base_pose.inverse * self.gripper_base_pose
        return relative_pose.x

    @property
    def body_id(self) -> int:
        """Get the base id in pymunk space."""
        return self._base_body.id

    def reset_positions(
        self,
        base_x: float,
        base_y: float,
        base_theta: float,
        base_vel: tuple[Vec2d, float],
        arm_length: float,
        arm_vel: tuple[Vec2d, float],
        gripper_gap: float,
        gripper_vel_l: tuple[Vec2d, float],
        gripper_vel_r: tuple[Vec2d, float],
        helder_object_vels: list[Vec2d] | None = None,
    ) -> None:
        """Reset robot to specified positions with zero velocity."""
        self._base_body.angle = base_theta
        self._base_body.angular_velocity = base_vel[1]
        self._base_body.position = (base_x, base_y)
        self._base_body.velocity = base_vel[0]

        base_to_gripper = SE2Pose(x=arm_length, y=0.0, theta=0.0)
        gripper_pose = self.base_pose * base_to_gripper
        self._gripper_base_body.angle = gripper_pose.theta
        self._gripper_base_body.angular_velocity = arm_vel[1]
        self._gripper_base_body.position = (gripper_pose.x, gripper_pose.y)
        self._gripper_base_body.velocity = arm_vel[0]

        gripper_to_left_finger = SE2Pose(
            x=self.gripper_finger_width / 2,
            y=gripper_gap / 2,
            theta=0.0,
        )
        left_finger_pose = gripper_pose * gripper_to_left_finger
        self._left_finger_body.angle = left_finger_pose.theta
        self._left_finger_body.angular_velocity = gripper_vel_l[1]
        self._left_finger_body.position = (left_finger_pose.x, left_finger_pose.y)
        self._left_finger_body.velocity = gripper_vel_l[0]

        gripper_to_right_finger = SE2Pose(
            x=self.gripper_finger_width / 2,
            y=-gripper_gap / 2,
            theta=0.0,
        )
        right_finger_pose = gripper_pose * gripper_to_right_finger
        self._right_finger_body.angle = right_finger_pose.theta
        self._right_finger_body.angular_velocity = gripper_vel_r[1]
        self._right_finger_body.position = (right_finger_pose.x, right_finger_pose.y)
        self._right_finger_body.velocity = gripper_vel_r[0]

        # Reset held objects - they have the same velocity as gripper base
        if helder_object_vels is not None and len(helder_object_vels):
            assert len(helder_object_vels) == len(
                self.held_objects
            ), "Length of helder_object_vels must match the number of held objects."
            for i, (obj, _, relative_pose) in enumerate(self.held_objects):
                obj_body, _ = obj
                new_obj_pose = gripper_pose * relative_pose
                obj_body.angle = new_obj_pose.theta
                obj_body.position = (new_obj_pose.x, new_obj_pose.y)
                obj_body.velocity = helder_object_vels[i]
                obj_body.angular_velocity = self._gripper_base_body.angular_velocity

        # Update last state
        self.update_last_state()

    def revert_to_last_state(self) -> None:
        """Reset to last state and stay static when collide with static objects."""
        self._base_body.angle = self._base_angle
        self._base_body.position = self._base_position
        self._base_body.velocity = Vec2d(0.0, 0.0)
        self._base_body.angular_velocity = 0.0

        gripper_base_rel_pos = SE2Pose(x=self._arm_length, y=0.0, theta=0.0)
        gripper_base_pose = self.base_pose * gripper_base_rel_pos
        self._gripper_base_body.angle = gripper_base_pose.theta
        self._gripper_base_body.position = (gripper_base_pose.x, gripper_base_pose.y)
        self._gripper_base_body.velocity = Vec2d(0.0, 0.0)
        self._gripper_base_body.angular_velocity = 0.0

        left_finger_rel_pos = SE2Pose(
            x=self.gripper_finger_width / 2,
            y=self._gripper_gap / 2,
            theta=0.0,
        )
        left_finger_pose = gripper_base_pose * left_finger_rel_pos
        self._left_finger_body.angle = left_finger_pose.theta
        self._left_finger_body.position = (left_finger_pose.x, left_finger_pose.y)
        self._left_finger_body.velocity = Vec2d(0.0, 0.0)
        self._left_finger_body.angular_velocity = 0.0

        right_finger_rel_pos = SE2Pose(
            x=self.gripper_finger_width / 2,
            y=-self._gripper_gap / 2,
            theta=0.0,
        )
        right_finger_pose = gripper_base_pose * right_finger_rel_pos
        self._right_finger_body.angle = right_finger_pose.theta
        self._right_finger_body.position = (right_finger_pose.x, right_finger_pose.y)
        self._right_finger_body.velocity = Vec2d(0.0, 0.0)
        self._right_finger_body.angular_velocity = 0.0

        # Update held objects
        for obj, _, relative_pose in self.held_objects:
            obj_body, _ = obj
            new_obj_pose = gripper_base_pose * relative_pose
            obj_body.angle = new_obj_pose.theta
            obj_body.position = (new_obj_pose.x, new_obj_pose.y)
            obj_body.velocity = Vec2d(0.0, 0.0)
            obj_body.angular_velocity = 0.0

    def update_last_state(self) -> None:
        """Update the last state tracking variables."""
        self._base_position = Vec2d(
            self._base_body.position.x, self._base_body.position.y
        )
        self._base_angle = self._base_body.angle

        relative_pose = self.base_pose.inverse * self.gripper_base_pose
        self._arm_length = relative_pose.x
        relative_finger_pose_l = self.gripper_base_pose.inverse * self.finger_poses_l
        relative_finger_pose_r = self.gripper_base_pose.inverse * self.finger_poses_r
        self._gripper_gap = relative_finger_pose_l.y - relative_finger_pose_r.y

    def update(
        self,
        base_vel: Vec2d,
        base_ang_vel: float,
        gripper_base_vel: Vec2d,
        finger_vel_l: Vec2d,
        finger_vel_r: Vec2d,
        helder_object_vels: list[Vec2d],
    ) -> None:
        """Update the body velocities."""
        # Update robot last state
        self.update_last_state()
        # Update velocities
        self._base_body.velocity = base_vel
        self._base_body.angular_velocity = base_ang_vel
        # Calculate target gripper base
        # It only has relative translational velocity
        self._gripper_base_body.velocity = gripper_base_vel
        self._gripper_base_body.angular_velocity = base_ang_vel
        # Left Finger
        self._left_finger_body.velocity = finger_vel_l
        self._left_finger_body.angular_velocity = base_ang_vel
        # Right Finger
        self._right_finger_body.velocity = finger_vel_r
        self._right_finger_body.angular_velocity = base_ang_vel

        # Update held objects - they have the same velocity as gripper base
        for i, (obj, _, _) in enumerate(self.held_objects):
            obj_body, _ = obj
            obj_body.velocity = helder_object_vels[i]
            obj_body.angular_velocity = self._gripper_base_body.angular_velocity

    def is_grasping(
        self, contact_point_set: pymunk.ContactPointSet, tgt_body: pymunk.Body
    ) -> bool:
        """Check if robot is grasping a target body."""
        # Checker 0: If robot is closing gripper
        if not self.is_closing_finger:
            return False
        # Checker 1: If contact normal is roughly perpendicular
        # to gripper_base_body base
        normal = contact_point_set.normal
        if not self._gripper_base_body:
            return False
        dtheta = abs(self._gripper_base_body.angle - normal.angle)
        dtheta = min(dtheta, 2 * np.pi - dtheta)
        theta_ok = abs(dtheta - np.pi / 2) < self.grasping_theta_thresh
        if not theta_ok:
            return False
        # Checker 2: If exist contact points in hand and target body is within
        # the gripper height
        rel_body = self.gripper_base_pose.inverse * SE2Pose(
            x=tgt_body.position.x, y=tgt_body.position.y, theta=0.0
        )
        if abs(rel_body.y) > self.gripper_base_height / 2:
            return False
        for pt in contact_point_set.points:
            pt_a = pt.point_a
            p_a = SE2Pose(x=pt_a.x, y=pt_a.y, theta=0.0)
            rel_a = self.gripper_base_pose.inverse * p_a
            if (abs(rel_a.y) < self.gripper_base_height / 2) and (
                (rel_a.x < self.gripper_finger_width / 2) and rel_a.x > 0
            ):
                return True
        return False

    def add_to_hand(self, obj: tuple[Body, list[Shape]], mass: float) -> None:
        """Add an object to the robot's hand."""
        obj_body, _ = obj
        obj_pose = SE2Pose(
            x=obj_body.position.x, y=obj_body.position.y, theta=obj_body.angle
        )
        gripper_base_pose = SE2Pose(
            x=self._gripper_base_body.position.x,
            y=self._gripper_base_body.position.y,
            theta=self._gripper_base_body.angle,
        )
        relative_obj_pose = gripper_base_pose.inverse * obj_pose
        self.held_objects.append((obj, mass, relative_obj_pose))

    def body_in_hand(self, body_id: int) -> bool:
        """Check if a body is in the robot's hand."""
        for (obj_body, _), _, _ in self.held_objects:
            if obj_body.id == body_id:
                return True
        return False


class PDController:
    """A simple PD controller for the robot."""

    def __init__(
        self,
        kp_pos: float = 100.0,
        kv_pos: float = 20.0,
        kp_rot: float = 500.0,
        kv_rot: float = 50.0,
    ) -> None:
        self.kp_pos = kp_pos
        self.kv_pos = kv_pos
        self.kp_rot = kp_rot
        self.kv_rot = kv_rot

    def compute_control(
        self,
        robot: KinRobot,
        tgt_x: float,
        tgt_y: float,
        tgt_theta: float,
        tgt_arm: float,  # target arm length L*
        tgt_gripper: float,  # target finger opening g*
        dt: float,
    ) -> tuple[Vec2d, float, Vec2d, Vec2d, Vec2d, list[Vec2d]]:
        """Compute base vel, base ang vel, gripper-base vel (world), finger vel (world),
        and held object vels (world) using PD control."""
        # === 0) Read current state ===
        base_pos_curr = Vec2d(robot.base_pose.x, robot.base_pose.y)
        base_vel_curr = robot.base_vel[0]  # Vec2d(vx, vy) in world
        base_ang_curr = robot.base_pose.theta
        base_ang_vel_curr = robot.base_vel[1]  # scalar omega
        base_rot_omega_vec = Vec2d(
            math.cos(base_ang_curr + math.pi / 2), math.sin(base_ang_curr + math.pi / 2)
        )

        L_curr = robot.curr_arm_length  # current arm length (scalar)
        Ldot_curr = robot.gripper_base_vel[0]

        # If available (recommended), provide current gripper opening and its rate:
        tgt_gripper_l = tgt_gripper / 2
        tgt_gripper_r = -tgt_gripper / 2
        # NOTE: Should use the actual finger's pos as current
        # instead of using current gripper gap (will have translation error)
        g_curr_l = robot.curr_l_finger_gap
        g_curr_r = robot.curr_r_finger_gap
        finger_vel_abs_w_l = robot.finger_vel_l[0]  # Vec2d
        finger_vel_abs_w_r = robot.finger_vel_r[0]  # Vec2d

        # === 1) Base PD ===
        base_pos_tgt = Vec2d(tgt_x, tgt_y)
        a_base_lin = self.kp_pos * (base_pos_tgt - base_pos_curr) + self.kv_pos * (
            Vec2d(0, 0) - base_vel_curr
        )
        base_vel = base_vel_curr + a_base_lin * dt

        a_base_ang = self.kp_rot * (tgt_theta - base_ang_curr) + self.kv_rot * (
            0.0 - base_ang_vel_curr
        )
        base_ang_vel = base_ang_vel_curr + a_base_ang * dt

        # === 2) Arm prismatic rate via PD on length in the base frame ===
        # PD on arm length
        kp_arm = getattr(self, "kp_arm", self.kp_pos)
        kv_arm = getattr(self, "kv_arm", self.kv_pos)
        arm_center_omega_vec = (
            base_rot_omega_vec * L_curr * base_ang_vel_curr
        )  # omega x r
        # Extract prismatic vel from a moving base
        rel_Ldot_curr = (
            (Ldot_curr - base_vel_curr - arm_center_omega_vec).rotated(-base_ang_curr).x
        )  # R^T * v_gripper_base
        a_L = kp_arm * (tgt_arm - L_curr) + kv_arm * (0.0 - rel_Ldot_curr)

        # Integrate prismatic rate
        rel_Ldot_next = rel_Ldot_curr + a_L * dt
        # Note: We need to use the *next base_ang_vel* to compute the
        # world-frame gripper-base velocity
        v_gripper_base = (
            base_vel
            + base_rot_omega_vec * L_curr * base_ang_vel
            + Vec2d(rel_Ldot_next, 0.0).rotated(base_ang_curr)
        )

        # Held object vel (world), calculated the same way as gripper-base vel
        helde_object_vels = []
        for kin_obj, _, _ in robot.held_objects:
            obj_body, _ = kin_obj
            obj_x_world = obj_body.position.x
            obj_y_world = obj_body.position.y
            obj_pos = Vec2d(obj_x_world, obj_y_world)
            relative_pos = obj_pos - base_pos_curr
            obj_rot_omega_vec_base = relative_pos.normalized().rotated(math.pi / 2)
            # We assume held object does not have relative velocity in the gripper frame
            # So we can just use rel_Ldot_next in the x-dir.
            v_held_obj = (
                base_vel
                + obj_rot_omega_vec_base * relative_pos.length * base_ang_vel
                + Vec2d(rel_Ldot_next, 0.0).rotated(base_ang_curr)
            )
            helde_object_vels.append(v_held_obj)

        # === 3) Gripper-base world velocity = rigid motion + prismatic contribution ===
        # Use *updated* base_vel & base_ang_vel for consistency in this control step
        kp_finger = getattr(self, "kp_finger", self.kp_pos)
        kv_finger = getattr(self, "kv_finger", self.kv_pos)
        gripper_centr = Vec2d(robot.finger_poses_l.x, robot.finger_poses_l.y)
        # Extract the rotate omega x r contribution
        relative_pos = gripper_centr - base_pos_curr
        finger_rot_omega_vec_base = relative_pos.normalized().rotated(math.pi / 2)
        finger_rot_omega_vec = (
            finger_rot_omega_vec_base * relative_pos.length * base_ang_vel_curr
        )
        # We only care about y-dir as finger only moves in y in the base frame
        rel_gdot_curr = (
            (finger_vel_abs_w_l - base_vel_curr - finger_rot_omega_vec)
            .rotated(-base_ang_curr)
            .y
        )
        a_g = kp_finger * (tgt_gripper_l - g_curr_l) + kv_finger * (0.0 - rel_gdot_curr)
        rel_gdot_next = rel_gdot_curr + a_g * dt
        # Finger world velocity, similar to gripper-base vel but with the additional
        # prismatic part in y-dir
        finger_vel_l = (
            base_vel
            + finger_rot_omega_vec_base * relative_pos.length * base_ang_vel
            + Vec2d(rel_Ldot_next, rel_gdot_next).rotated(base_ang_curr)
        )

        gripper_centr = Vec2d(robot.finger_poses_r.x, robot.finger_poses_r.y)
        # Extract the rotate omega x r contribution
        relative_pos = gripper_centr - base_pos_curr
        finger_rot_omega_vec_base = relative_pos.normalized().rotated(math.pi / 2)
        finger_rot_omega_vec = (
            finger_rot_omega_vec_base * relative_pos.length * base_ang_vel_curr
        )
        # We only care about y-dir as finger only moves in y in the base frame
        rel_gdot_curr = (
            (finger_vel_abs_w_r - base_vel_curr - finger_rot_omega_vec)
            .rotated(-base_ang_curr)
            .y
        )
        a_g = kp_finger * (tgt_gripper_r - g_curr_r) + kv_finger * (0.0 - rel_gdot_curr)
        rel_gdot_next = rel_gdot_curr + a_g * dt
        # Finger world velocity, similar to gripper-base vel but with the additional
        # prismatic part in y-dir
        finger_vel_r = (
            base_vel
            + finger_rot_omega_vec_base * relative_pos.length * base_ang_vel
            + Vec2d(rel_Ldot_next, rel_gdot_next).rotated(base_ang_curr)
        )

        return (
            base_vel,
            base_ang_vel,
            v_gripper_base,
            finger_vel_l,
            finger_vel_r,
            helde_object_vels,
        )


def on_gripper_grasp(
    arbiter: pymunk.Arbiter, space: pymunk.Space, robot: KinRobot
) -> None:
    """Collision callback for gripper grasping objects."""
    dynamic_body = arbiter.bodies[0]
    if robot.is_grasping(arbiter.contact_point_set, dynamic_body):
        # Create a new kinematic object
        kinematic_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        kinematic_body.position = dynamic_body.position
        kinematic_body.angle = dynamic_body.angle
        shapes = dynamic_body.shapes
        new_shapes: list[Shape] = []
        for shape in shapes:
            copied_shape = shape.copy()
            copied_shape.body = kinematic_body
            # Held objects are considered part of the robot for collision purposes
            # NOTE: Reset with this has not being tested to pass yet
            # copied_shape.collision_type = ROBOT_COLLISION_TYPE
            new_shapes.append(copied_shape)
        space.add(kinematic_body, *new_shapes)
        robot.add_to_hand((kinematic_body, new_shapes), dynamic_body.mass)
        # Remove the dynamic body and attached shapes from the space
        space.remove(dynamic_body, *shapes)


def on_collision_w_static(
    arbiter: pymunk.Arbiter, space: pymunk.Space, robot: KinRobot
) -> None:
    """Collision callback for robot colliding with static objects."""
    del arbiter
    del space
    robot.revert_to_last_state()


def on_dot_robot_collision_w_static(
    arbiter: pymunk.Arbiter, space: pymunk.Space, robot: DotRobot
) -> None:
    """Collision callback for DotRobot colliding with static objects."""
    del arbiter
    del space
    robot.revert_to_last_state()


def create_walls_from_world_boundaries(
    world_min_x: float,
    world_max_x: float,
    world_min_y: float,
    world_max_y: float,
    min_dx: float,
    max_dx: float,
    min_dy: float,
    max_dy: float,
) -> dict[Object, dict[str, float]]:
    """Create wall objects and feature dicts based on world boundaries.

    Velocities are used to determine how large the walls need to be to avoid the
    possibility that the robot will transport over the wall.

    Left and right walls are considered "surfaces" (z_order=1) while top and bottom
    walls are considered "floors" (z_order=0). Otherwise there might be weird collision
    betweent left/right walls and top/bottom walls.
    """
    state_dict: dict[Object, dict[str, float]] = {}
    # Right wall.
    right_wall = Object("right_wall", KinRectangleType)
    side_wall_height = world_max_y - world_min_y
    state_dict[right_wall] = {
        "x": world_max_x + max_dx,
        "vx": 0.0,
        "y": (world_min_y + world_max_y) / 2,
        "vy": 0.0,
        "width": 2 * max_dx,  # 2x just for safety
        "height": side_wall_height,
        "theta": 0.0,
        "omega": 0.0,
        "static": True,
        "held": False,
        "color_r": BLACK[0],
        "color_g": BLACK[1],
        "color_b": BLACK[2],
        "z_order": ZOrder.ALL.value,
    }
    # Left wall.
    left_wall = Object("left_wall", KinRectangleType)
    state_dict[left_wall] = {
        "x": world_min_x + min_dx,
        "vx": 0.0,
        "y": (world_min_y + world_max_y) / 2,
        "vy": 0.0,
        "width": 2 * abs(min_dx),  # 2x just for safety
        "height": side_wall_height,
        "theta": 0.0,
        "omega": 0.0,
        "static": True,
        "held": False,
        "color_r": BLACK[0],
        "color_g": BLACK[1],
        "color_b": BLACK[2],
        "z_order": ZOrder.ALL.value,
    }
    # Top wall.
    top_wall = Object("top_wall", KinRectangleType)
    horiz_wall_width = 2 * 2 * abs(min_dx) + world_max_x - world_min_x
    state_dict[top_wall] = {
        "x": (world_min_x + world_max_x) / 2,
        "vx": 0.0,
        "y": world_max_y + max_dy,
        "vy": 0.0,
        "width": horiz_wall_width,
        "height": 2 * max_dy,
        "theta": 0.0,
        "omega": 0.0,
        "static": True,
        "held": False,
        "color_r": BLACK[0],
        "color_g": BLACK[1],
        "color_b": BLACK[2],
        "z_order": ZOrder.ALL.value,
    }
    # Bottom wall.
    bottom_wall = Object("bottom_wall", KinRectangleType)
    state_dict[bottom_wall] = {
        "x": (world_min_x + world_max_x) / 2,
        "vx": 0.0,
        "y": world_min_y + min_dy,
        "vy": 0.0,
        "width": horiz_wall_width,
        "height": 2 * abs(min_dy),
        "theta": 0.0,
        "omega": 0.0,
        "static": True,
        "held": False,
        "color_r": BLACK[0],
        "color_g": BLACK[1],
        "color_b": BLACK[2],
        "z_order": ZOrder.ALL.value,
    }
    return state_dict


def get_fingered_robot_action_from_gui_input(
    action_space: KinRobotActionSpace, gui_input: dict[str, Any]
) -> NDArray[np.float64]:
    """Get the mapping from human inputs to actions, derived from action space."""
    # This will be implemented later - placeholder for now
    keys_pressed = gui_input["keys"]
    right_x, right_y = gui_input["right_stick"]
    left_x, _ = gui_input["left_stick"]

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

    # The w/s mouse keys are used to adjust the robot arm.
    if "a" in keys_pressed:
        action[3] = low[3]
    if "s" in keys_pressed:
        action[3] = high[3]

    # The space bar is used to close the gripper.
    # Open the gripper by default.
    if "d" in keys_pressed:
        action[4] = low[4]
    if "f" in keys_pressed:
        action[4] = high[4]

    return action


def get_dot_robot_action_from_gui_input(
    action_space: DotRobotActionSpace, gui_input: dict[str, Any]
) -> NDArray[np.float64]:
    """Get the mapping from human inputs to actions, derived from action space."""
    # This will be implemented later - placeholder for now
    right_x, right_y = gui_input["right_stick"]

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

    return action


def get_tool_tip_position(
    state: ObjectCentricState, robot: Object
) -> tuple[float, float]:
    """Get the tip of the tool for the robot, which is defined as the center of the
    bottom edge of the gripper."""
    multibody = kin_robot_to_multibody2d(robot, state)
    gripper_geom = multibody.get_body("gripper_base").geom
    assert isinstance(gripper_geom, Rectangle)
    return (gripper_geom.x, gripper_geom.y)


def get_suctioned_objects(
    state: ObjectCentricState, robot: Object
) -> list[tuple[Object, SE2Pose]]:
    """Find objects that are in the suction zone of a CRVRobot and return the associated
    transform from gripper tool tip to suctioned object."""
    # If the robot's vacuum is not on, there are no suctioned objects.
    if state.get(robot, "vacuum") <= 0.5:
        return []
    gripper_x, gripper_y = get_tool_tip_position(state, robot)
    gripper_theta = state.get(robot, "theta")
    world_to_gripper = SE2Pose(gripper_x, gripper_y, gripper_theta)
    # Find MOVABLE objects in collision with the suction geom.
    suctioned_objects: list[tuple[Object, SE2Pose]] = []
    for obj in state:
        if "held" in state.type_features[obj.type]:
            if state.get(obj, "held"):
                world_to_obj = SE2Pose(
                    x=state.get(obj, "x"),
                    y=state.get(obj, "y"),
                    theta=state.get(obj, "theta"),
                )
                gripper_to_obj = world_to_gripper.inverse * world_to_obj
                suctioned_objects.append((obj, gripper_to_obj))
    return suctioned_objects


def snap_suctioned_objects(
    state: ObjectCentricState,
    robot: Object,
    suctioned_objs: list[tuple[Object, SE2Pose]],
) -> None:
    """Updates the state in-place."""
    gripper_x, gripper_y = get_tool_tip_position(state, robot)
    gripper_theta = state.get(robot, "theta")
    world_to_gripper = SE2Pose(gripper_x, gripper_y, gripper_theta)
    for obj, gripper_to_obj in suctioned_objs:
        world_to_obj = world_to_gripper * gripper_to_obj
        state.set(obj, "x", world_to_obj.x)
        state.set(obj, "y", world_to_obj.y)
        state.set(obj, "theta", world_to_obj.theta)


def get_held_objects(
    state: ObjectCentricState, robot: Object
) -> list[tuple[Object, SE2Pose]]:
    """Find objects held by the KinRobot's gripper and return the transform from the
    gripper tool-tip to each held object.

    Unlike ``get_suctioned_objects`` (which checks a vacuum flag), this checks
    the ``held`` attribute on every non-robot object in the state.
    """
    gripper_x, gripper_y = get_tool_tip_position(state, robot)
    gripper_theta = state.get(robot, "theta")
    world_to_gripper = SE2Pose(gripper_x, gripper_y, gripper_theta)
    held_objects: list[tuple[Object, SE2Pose]] = []
    for obj in state:
        if obj == robot:
            continue
        if "held" in state.type_features[obj.type]:
            if state.get(obj, "held"):
                world_to_obj = SE2Pose(
                    x=state.get(obj, "x"),
                    y=state.get(obj, "y"),
                    theta=state.get(obj, "theta"),
                )
                gripper_to_obj = world_to_gripper.inverse * world_to_obj
                held_objects.append((obj, gripper_to_obj))
    return held_objects


def snap_held_objects(
    state: ObjectCentricState,
    robot: Object,
    held_objs: list[tuple[Object, SE2Pose]],
) -> None:
    """Update held-object poses in-place so they stay rigidly attached to the robot's
    gripper tool-tip."""
    gripper_x, gripper_y = get_tool_tip_position(state, robot)
    gripper_theta = state.get(robot, "theta")
    world_to_gripper = SE2Pose(gripper_x, gripper_y, gripper_theta)
    for obj, gripper_to_obj in held_objs:
        world_to_obj = world_to_gripper * gripper_to_obj
        state.set(obj, "x", world_to_obj.x)
        state.set(obj, "y", world_to_obj.y)
        state.set(obj, "theta", world_to_obj.theta)


def run_motion_planning_for_kin_robot(
    state: ObjectCentricState,
    robot: Object,
    target_pose: SE2Pose,
    action_space: KinRobotActionSpace,
    static_object_body_cache: dict[Object, MultiBody2D] | None = None,
    seed: int = 0,
    num_attempts: int = 10,
    num_iters: int = 100,
    smooth_amt: int = 50,
    motion_border: list[float] | None = None,
) -> list[SE2Pose] | None:
    """Run BiRRT motion planning for a KinRobot to reach ``target_pose``.

    This plans a collision-free SE2 path for the robot base, snapping any
    held objects along with it.

    .. note::

        This planner assumes **no contact** between the robot/held objects
        and dynamic objects in the scene — every configuration along the
        planned path is checked for geometric intersection only.  In a
        dynamic (PyMunk) environment, physics-based contacts (pushing,
        friction, etc.) are not modelled, so the resulting path may be
        unreliable when the robot must interact with movable objects.
    """
    if static_object_body_cache is None:
        static_object_body_cache = {}

    rng = np.random.default_rng(seed)

    # Use the object positions in the state to create a rough room boundary.
    x_lb, x_ub, y_lb, y_ub = np.inf, -np.inf, np.inf, -np.inf
    for obj in state:
        pose = get_se2_pose(state, obj)
        x_lb = min(x_lb, pose.x)
        x_ub = max(x_ub, pose.x)
        y_lb = min(y_lb, pose.y)
        y_ub = max(y_ub, pose.y)

    if motion_border is not None:
        # NOTE: We might overwrite the motion border due to some
        # walls, e.g., pushpullhook2d.
        # It is tricky to use Z-order for collision checking in
        # that environment, so we just use the motion border to
        # set the sampling space for motion planning.
        x_lb = motion_border[0]
        x_ub = motion_border[1]
        y_lb = motion_border[2]
        y_ub = motion_border[3]

    # Create a static version of the state so that the geoms only need to be
    # instantiated once during motion planning (except for the robot).  Make
    # sure to not update the global cache because we don't want to carry over
    # static things that are not actually static.
    static_object_body_cache = static_object_body_cache.copy()
    held_objects = get_held_objects(state, robot)
    moving_objects = {robot} | {o for o, _ in held_objects}
    static_state = state.copy()
    for o in static_state:
        if o in moving_objects:
            continue
        static_state.set(o, "static", 1.0)

    # Set up the BiRRT methods.
    def sample_fn(_: SE2Pose) -> SE2Pose:
        """Sample a robot pose."""
        x = rng.uniform(x_lb, x_ub)
        y = rng.uniform(y_lb, y_ub)
        theta = rng.uniform(-np.pi, np.pi)
        return SE2Pose(x, y, theta)

    def extend_fn(pt1: SE2Pose, pt2: SE2Pose) -> Iterable[SE2Pose]:
        """Interpolate between the two poses respecting action-space bounds."""
        dx = pt2.x - pt1.x
        dy = pt2.y - pt1.y
        dtheta = get_signed_angle_distance(pt2.theta, pt1.theta)
        assert isinstance(action_space, KinRobotActionSpace)
        abs_x = action_space.high[0] if dx > 0 else action_space.low[0]
        abs_y = action_space.high[1] if dy > 0 else action_space.low[1]
        abs_theta = action_space.high[2] if dtheta > 0 else action_space.low[2]
        x_num_steps = int(dx / abs_x) + 1
        assert x_num_steps > 0
        y_num_steps = int(dy / abs_y) + 1
        assert y_num_steps > 0
        theta_num_steps = int(dtheta / abs_theta) + 1
        assert theta_num_steps > 0
        num_steps = max(x_num_steps, y_num_steps, theta_num_steps)
        x = pt1.x
        y = pt1.y
        theta = pt1.theta
        yield SE2Pose(x, y, theta)
        for _ in range(num_steps):
            x += dx / num_steps
            y += dy / num_steps
            theta = wrap_angle(theta + dtheta / num_steps)
            yield SE2Pose(x, y, theta)

    def collision_fn(pt: SE2Pose) -> bool:
        """Check for collisions if the robot were at this pose."""
        static_state.set(robot, "x", pt.x)
        static_state.set(robot, "y", pt.y)
        static_state.set(robot, "theta", pt.theta)

        # Snap held objects to the new robot pose.
        snap_held_objects(static_state, robot, held_objects)
        obstacle_objects = set(static_state) - moving_objects

        return state_2d_has_collision(
            static_state, moving_objects, obstacle_objects, static_object_body_cache
        )

    def distance_fn(pt1: SE2Pose, pt2: SE2Pose) -> float:
        """Return a distance between the two points."""
        dx = pt2.x - pt1.x
        dy = pt2.y - pt1.y
        dtheta = get_signed_angle_distance(pt2.theta, pt1.theta)
        return np.sqrt(dx**2 + dy**2) + abs(dtheta)

    birrt = BiRRT(
        sample_fn,
        extend_fn,
        collision_fn,
        distance_fn,
        rng,
        num_attempts,
        num_iters,
        smooth_amt,
    )

    initial_pose = get_se2_pose(state, robot)
    return birrt.query(initial_pose, target_pose)
