"""Base environment class for all Kinematic3D environments."""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import Any, Generic
from typing import Type as TypingType
from typing import TypeVar

import gymnasium
import numpy as np
import pybullet as p
from numpy.typing import NDArray
from pybullet_helpers.camera import capture_image
from pybullet_helpers.geometry import Pose, SE2Pose, get_pose, multiply_poses, set_pose
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.inverse_kinematics import (
    check_body_collisions,
    check_collisions_with_held_object,
    check_mobile_base_collisions,
    set_robot_joints_with_held_object,
)
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.robots import create_pybullet_mobile_robot
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from pybullet_helpers.utils import create_pybullet_block
from relational_structs import (
    Array,
    Object,
    ObjectCentricStateSpace,
    Type,
)
from relational_structs.utils import create_state_from_dict
from scipy.spatial.transform import Rotation

from kinder.core import KinDEREnvConfig, ObjectCentricKinDEREnv, RobotActionSpace
from kinder.envs.kinematic3d.object_types import (
    Kinematic3DCuboidType,
    Kinematic3DEnvTypeFeatures,
    Kinematic3DFixtureType,
    Kinematic3DPointType,
    Kinematic3DRobotType,
)
from kinder.envs.kinematic3d.utils import (
    DEFAULT_REALISTIC_BG_PATH,
    Kinematic3DObjectCentricState,
    Kinematic3DRobotActionSpace,
    extend_joints_to_include_fingers,
    get_robot_action_from_gui_input,
    load_realistic_background,
    remove_fingers_from_extended_joints,
)


@dataclass(frozen=True)
class Kinematic3DEnvConfig(KinDEREnvConfig):
    """Config for Kinematic3DEnv()."""

    # Robot.
    robot_name: str = "tidybot-kinova"
    robot_base_home_pose: SE2Pose = SE2Pose.identity()
    robot_base_pose_lower_bound: SE2Pose = SE2Pose(-10.0, -10.0, -np.pi)
    robot_base_pose_upper_bound: SE2Pose = SE2Pose(10.0, 10.0, np.pi)
    robot_base_z: float = 0.0
    initial_joints: JointPositions = field(
        # This is a retract position.
        default_factory=lambda: [
            0.0,  # "joint_1", starting at the robot base and going up to the gripper
            -0.35,  # "joint_2"
            -np.pi,  # "joint_3"
            -2.5,  # "joint_4"
            0.0,  # "joint_5"
            -0.87,  # "joint_6"
            np.pi / 2,  # "joint_7"
        ]
    )
    initial_finger_state: float = 0.0
    end_effector_viz_half_extents: tuple[float, float, float] = (0.01, 0.01, 0.035)
    end_effector_viz_color: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 0.0)
    max_action_mag: float = 0.4
    # Maximum coordinate increment between swept-motion collision checks. When unset,
    # only the action endpoint is checked, preserving the original transition.
    max_collision_check_step: float | None = None
    check_base_collisions: bool = False

    # This is used to check whether a grasped object can be placed on a surface.
    min_placement_dist: float = 5e-3

    # World bounds.
    x_lb: float = -1.5
    x_ub: float = 1.5
    y_lb: float = -1.5
    y_ub: float = 1.5

    # For rendering.
    render_dpi: int = 300
    render_fps: int = 20
    render_image_width: int = 640
    render_image_height: int = 360

    # Base camera (mounted on robot base) - matches dynamics3d tidybot
    # From tidybot.xml: pos="0.2525 0 0.335" euler="0 -0.7853981634 -1.5707963268"
    base_camera_offset: tuple[float, float, float] = (0.2525, 0.0, 0.335)
    base_camera_euler: tuple[float, float, float] = (0, -np.pi / 4, -np.pi / 2)
    base_camera_fov: float = 52.23384539951277
    base_camera_image_width: int = 640
    base_camera_image_height: int = 360

    # End-effector camera (mounted on wrist) - matches dynamics3d tidybot
    # From tidybot.xml: pos="0 -0.05639 -0.058475" quat="0 0 0 1"
    # add 2cm in z direction to avoid camera too close to the object/ground.
    ee_camera_offset: tuple[float, float, float] = (0.0, 0.04639, -0.108475)
    ee_camera_euler: tuple[float, float, float] = (np.pi, 0.0, 0.0)
    ee_camera_fov: float = 41.83792730009236
    ee_camera_image_width: int = 640
    ee_camera_image_height: int = 360

    # Realistic background settings.
    realistic_bg: bool = True
    realistic_bg_position: tuple[float, float, float] = (0.7, -1.5, -0.02)
    realistic_bg_euler: tuple[float, float, float] = (np.pi / 2, 0, 0.0)
    realistic_bg_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Floor, which can optionally be included as an object.
    floor_included_as_object: bool = False
    floor_z: float = 0.0
    floor_half_extents: tuple[float, float, float] = (x_ub - x_lb, y_ub - y_lb, 0.005)
    floor_pose: Pose = Pose(
        ((x_lb + x_ub) / 2, (y_lb + y_ub) / 2, floor_z - 2 * floor_half_extents[2])
    )

    def get_camera_kwargs(self) -> dict[str, Any]:
        """Get kwargs to pass to PyBullet camera."""
        return {
            "camera_target": (
                self.robot_base_home_pose.x,
                self.robot_base_home_pose.y,
                self.robot_base_z,
            ),
            "camera_yaw": 90,
            "camera_distance": 2.8,
            "camera_pitch": -30,
        }

    def _sample_block_on_block_pose(
        self,
        top_block_half_extents: tuple[float, float, float],
        bottom_block_half_extents: tuple[float, float, float],
        bottom_block_pose: Pose,
        rng: np.random.Generator,
    ) -> Pose:
        """Sample one block pose on top of another one, with no hanging allowed."""
        assert np.allclose(
            bottom_block_pose.orientation, (0, 0, 0, 1)
        ), "Not implemented"

        lb = (
            bottom_block_pose.position[0]
            - bottom_block_half_extents[0]
            + top_block_half_extents[0],
            bottom_block_pose.position[1]
            - bottom_block_half_extents[1]
            + top_block_half_extents[1],
            bottom_block_pose.position[2]
            + bottom_block_half_extents[2]
            + top_block_half_extents[2],
        )

        ub = (
            bottom_block_pose.position[0]
            + bottom_block_half_extents[0]
            - top_block_half_extents[0],
            bottom_block_pose.position[1]
            + bottom_block_half_extents[1]
            - top_block_half_extents[1],
            bottom_block_pose.position[2]
            + bottom_block_half_extents[2]
            + top_block_half_extents[2],
        )

        x, y, z = rng.uniform(lb, ub)

        return Pose((x, y, z))

    def _sample_block_on_block_pose_with_overhang(
        self,
        top_block_half_extents: tuple[float, float, float],
        bottom_block_half_extents: tuple[float, float, float],
        bottom_block_pose: Pose,
        rng: np.random.Generator,
        allowed_overhang_fraction: float = 0.25,
    ) -> Pose:
        """Sample one block pose on top of another one, where hanging is allowed."""
        assert np.allclose(
            bottom_block_pose.orientation, (0, 0, 0, 1)
        ), "Not implemented"

        lb = (
            bottom_block_pose.position[0]
            - bottom_block_half_extents[0]
            - top_block_half_extents[0] * allowed_overhang_fraction,
            bottom_block_pose.position[1]
            - bottom_block_half_extents[1]
            - top_block_half_extents[1] * allowed_overhang_fraction,
            bottom_block_pose.position[2]
            + bottom_block_half_extents[2]
            + top_block_half_extents[2],
        )

        ub = (
            bottom_block_pose.position[0]
            + bottom_block_half_extents[0]
            + top_block_half_extents[0] * allowed_overhang_fraction,
            bottom_block_pose.position[1]
            + bottom_block_half_extents[1]
            + top_block_half_extents[1] * allowed_overhang_fraction,
            bottom_block_pose.position[2]
            + bottom_block_half_extents[2]
            + top_block_half_extents[2],
        )

        x, y, z = rng.uniform(lb, ub)

        return Pose((x, y, z))


# Subclasses may extend the state.
_ObsType = TypeVar("_ObsType", bound=Kinematic3DObjectCentricState)
_ConfigType = TypeVar("_ConfigType", bound=Kinematic3DEnvConfig)


class ObjectCentricKinematic3DRobotEnv(
    ObjectCentricKinDEREnv[_ObsType, Array, _ConfigType],
    Generic[_ObsType, _ConfigType],
):
    """Base class for Kinematic3D environments."""

    def __init__(
        self,
        *args,
        use_gui: bool = False,
        realistic_bg: bool | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.use_gui = use_gui
        # Allow realistic_bg kwarg to override config value.
        self._realistic_bg_enabled = (
            realistic_bg if realistic_bg is not None else self.config.realistic_bg
        )

        # Create the PyBullet client.
        if use_gui:
            camera_info = self.config.get_camera_kwargs()
            self.physics_client_id = create_gui_connection(**camera_info)
        else:
            self.physics_client_id = p.connect(p.DIRECT)

        # Create robot.
        robot = create_pybullet_mobile_robot(
            self.config.robot_name,
            self.physics_client_id,
            base_z=self.config.robot_base_z,
            base_home_pose=self.config.robot_base_home_pose,
            base_pose_lower_bound=self.config.robot_base_pose_lower_bound,
            base_pose_upper_bound=self.config.robot_base_pose_upper_bound,
        )
        self.robot = robot
        self.robot.arm.set_joints(
            extend_joints_to_include_fingers(self.config.initial_joints)
        )

        # Show a visualization of the end effector.
        visual_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=self.config.end_effector_viz_half_extents,
            rgbaColor=self.config.end_effector_viz_color,
            physicsClientId=self.physics_client_id,
        )

        # Also create a collision body because we use it for grasp detection.
        collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=self.config.end_effector_viz_half_extents,
            physicsClientId=self.physics_client_id,
        )

        # Create the body for the end effector.
        end_effector_pose = self.robot.arm.get_end_effector_pose()
        self.end_effector_viz_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=end_effector_pose.position,
            baseOrientation=end_effector_pose.orientation,
            physicsClientId=self.physics_client_id,
        )

        # Track current held object.
        self._grasped_object: str | None = None
        self._grasped_object_transform: Pose | None = None

        # Load realistic background if enabled.
        self._realistic_bg_id: int | None = None
        if self._realistic_bg_enabled:
            rot = Rotation.from_euler(
                "zyx", self.config.realistic_bg_euler
            )  # MuJoCo convention
            self._realistic_bg_id = load_realistic_background(
                self.physics_client_id,
                obj_path=DEFAULT_REALISTIC_BG_PATH,  # Use default background
                position=self.config.realistic_bg_position,
                orientation=tuple(rot.as_quat()[[1, 2, 3, 0]]),
                scale=self.config.realistic_bg_scale,
            )

        # Optionally create floor.
        if self.config.floor_included_as_object:
            self.floor_id = create_pybullet_block(
                color=(0.5, 0.5, 0.5, 0.0),  # transparent
                half_extents=self.config.floor_half_extents,
                physics_client_id=self.physics_client_id,
            )
            set_pose(
                self.floor_id,
                self.config.floor_pose,
                self.physics_client_id,
            )

    @property
    @abc.abstractmethod
    def state_cls(self) -> TypingType[Kinematic3DObjectCentricState]:
        """The type of states in this environment."""

    @abc.abstractmethod
    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        """Create the constant initial state dict."""

    @abc.abstractmethod
    def _get_obs(self) -> _ObsType:
        """Get the current observation."""

    @abc.abstractmethod
    def goal_reached(self) -> bool:
        """Check if the goal is currently reached."""

    @abc.abstractmethod
    def _reset_objects(self) -> None:
        """Reset objects."""

    @abc.abstractmethod
    def _set_object_states(self, obs: _ObsType) -> None:
        """Reset the state of objects; helper for set_state()."""

    @abc.abstractmethod
    def _object_name_to_pybullet_id(self, object_name: str) -> int:
        """Look up the PyBullet ID for a given object name."""

    @abc.abstractmethod
    def _get_collision_object_ids(self) -> set[int]:
        """Get the collision object IDs."""

    @abc.abstractmethod
    def _get_movable_object_names(self) -> set[str]:
        """The names of objects that can be moved by the robot (grasped and placed)."""

    @abc.abstractmethod
    def _get_surface_object_names(self) -> set[str]:
        """The names of objects that can be used as surfaces for other objects.

        Note that surfaces might be movable, for example, consider block stacking.
        """

    @abc.abstractmethod
    def _get_half_extents(self, object_name: str) -> tuple[float, float, float]:
        """Get the half extents for a cuboid object."""

    def _get_triangle_features(
        self, object_name: str
    ) -> tuple[float, float, float, float]:
        """Return triangle parameters (side_a, side_b, depth, triangle_type).

        Subclasses that support triangles should override this. The default
        implementation returns zeros to provide a safe fallback for serialization.
        """
        assert object_name is not None
        return 0.0, 0.0, 0.0, 0.0

    @property
    def _robot_arm(self) -> FingeredSingleArmPyBulletRobot:
        robot_arm = self.robot.arm
        assert isinstance(robot_arm, FingeredSingleArmPyBulletRobot)
        return robot_arm

    @property
    def _grasped_object_id(self) -> int | None:
        if self._grasped_object is not None:
            return self._object_name_to_pybullet_id(self._grasped_object)
        return None

    @property
    def type_features(self) -> dict[Type, list[str]]:
        """The types and features for this environment."""
        return Kinematic3DEnvTypeFeatures

    def _create_observation_space(self, config: _ConfigType) -> ObjectCentricStateSpace:
        types = set(self.type_features)
        return ObjectCentricStateSpace(types, state_cls=self.state_cls)

    def _create_action_space(self, config: _ConfigType) -> RobotActionSpace:
        return Kinematic3DRobotActionSpace(max_magnitude=config.max_action_mag)

    def _create_constant_initial_state(self) -> _ObsType:
        initial_state_dict = self._create_constant_initial_state_dict()
        state = create_state_from_dict(
            initial_state_dict, Kinematic3DEnvTypeFeatures, state_cls=self.state_cls
        )
        # This is tricky for type annotation because we are dynamically setting the
        # class to be self.state_cls, which should be _ObsType.
        return state  # type: ignore

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[_ObsType, dict]:
        # Reset the random seed.
        gymnasium.Env.reset(self, seed=seed)

        # For testing purposes, the options may specify an initial state.
        if options is not None and "init_state" in options:
            self._set_state(options["init_state"])
        else:
            # Reset the held object info.
            self._grasped_object = None
            self._grasped_object_transform = None

            # Reset the robot.
            self._set_robot_and_held_object(
                self.config.robot_base_home_pose,
                self.config.initial_joints,
                self.config.initial_finger_state,
            )

            # Reset objects.
            self._reset_objects()

        return self._get_obs(), {}

    def _get_state(self) -> _ObsType:
        return self._get_obs()

    def _set_state(self, state: _ObsType) -> None:
        """Set the state of the environment to the given one."""
        self._set_robot_and_held_object(
            state.base_pose, state.joint_positions, state.finger_state
        )
        self._grasped_object = state.grasped_object
        self._grasped_object_transform = state.grasped_object_transform
        self._set_object_states(state)

    def _is_inside_object(
        self,
        obj_pose: Pose,
        grasped_object_pose: Pose,
        half_extents: tuple[float, float, float],
    ) -> bool:
        """Check if an object is inside the grasped object."""
        return (
            obj_pose.position[0] > grasped_object_pose.position[0] - half_extents[0]
            and obj_pose.position[0] < grasped_object_pose.position[0] + half_extents[0]
            and obj_pose.position[1] > grasped_object_pose.position[1] - half_extents[1]
            and obj_pose.position[1] < grasped_object_pose.position[1] + half_extents[1]
            and obj_pose.position[2] > grasped_object_pose.position[2] - half_extents[2]
            and obj_pose.position[2] < grasped_object_pose.position[2] + half_extents[2]
        )

    def _get_inside_objects(self) -> tuple[set[str], dict[str, Pose]]:
        """Compute which objects are inside the grasped container based on poses.

        Returns:
            Tuple of (set of object names, dict of name -> transform relative
            to robot EE).
        """
        if self._grasped_object is None:
            return set(), {}

        extent_threshold = 0.04
        half_extents = self._get_half_extents(self._grasped_object)
        if not all(e > extent_threshold for e in half_extents):
            return set(), {}

        grasped_object_id = self._grasped_object_id
        assert grasped_object_id is not None
        grasped_object_pose = get_pose(grasped_object_id, self.physics_client_id)
        world_to_robot = self.robot.arm.get_end_effector_pose()

        inside_names: set[str] = set()
        inside_transforms: dict[str, Pose] = {}

        for obj_name in self._get_movable_object_names():
            if obj_name == self._grasped_object:
                continue
            obj_id = self._object_name_to_pybullet_id(obj_name)
            obj_pose = get_pose(obj_id, self.physics_client_id)
            if self._is_inside_object(obj_pose, grasped_object_pose, half_extents):
                inside_names.add(obj_name)
                inside_transforms[obj_name] = multiply_poses(
                    world_to_robot.invert(), obj_pose
                )

        return inside_names, inside_transforms

    def _get_inside_object_ids(self) -> set[int]:
        """Get PyBullet IDs of objects inside the grasped container."""
        inside_names, _ = self._get_inside_objects()
        return {self._object_name_to_pybullet_id(name) for name in inside_names}

    def step(self, action: Array) -> tuple[_ObsType, float, bool, bool, dict]:
        # execute the base action
        base_action = np.clip(
            action[:3],
            -self.config.max_action_mag,
            self.config.max_action_mag,
        )
        current_base_pose = self.robot.get_base()
        next_base_pose = current_base_pose + SE2Pose(
            base_action[0], base_action[1], base_action[2]
        )

        # Store the current robot joints because we may need to revert in collision.
        current_joints = remove_fingers_from_extended_joints(
            self._robot_arm.get_joint_positions()
        )
        current_finger_state = self._robot_arm.get_finger_state()

        # Tentatively apply robot action.
        delta_arm_joints = action[-8:-1]
        # Clip the action to be within the allowed limits.
        delta_joints = np.clip(
            delta_arm_joints,
            -self.config.max_action_mag,
            self.config.max_action_mag,
        )

        next_joints = np.clip(
            current_joints + delta_joints,
            self._robot_arm.joint_lower_limits[:7],
            self._robot_arm.joint_upper_limits[:7],
        ).tolist()
        # Apply the motion incrementally so that an action cannot tunnel through a
        # collision and end at a collision-free configuration. This remains a purely
        # kinematic transition: intermediate states are used only for validity
        # checking, with no physics integration or contact response.
        max_delta = max(
            np.max(np.abs(base_action)),
            np.max(np.abs(np.asarray(next_joints) - np.asarray(current_joints))),
        )
        if self.config.max_collision_check_step is None:
            num_collision_checks = 1
        else:
            if self.config.max_collision_check_step <= 0:
                raise ValueError("max_collision_check_step must be positive")
            num_collision_checks = max(
                1, math.ceil(max_delta / self.config.max_collision_check_step)
            )
        motion_valid = True
        for i in range(1, num_collision_checks + 1):
            t = i / num_collision_checks
            intermediate_base_pose = current_base_pose + SE2Pose(
                t * base_action[0], t * base_action[1], t * base_action[2]
            )
            intermediate_joints = (
                np.asarray(current_joints)
                + t * (np.asarray(next_joints) - np.asarray(current_joints))
            ).tolist()
            self._set_robot_and_held_object(
                intermediate_base_pose,
                intermediate_joints,
                current_finger_state,
            )
            if self._robot_or_held_object_collision_exists():
                motion_valid = False
                break

        if not motion_valid:
            self._set_robot_and_held_object(
                current_base_pose, current_joints, current_finger_state
            )

        # Check for grasping.
        if action[-1] < -0.5:
            gripper_action = "close"
        elif action[-1] > 0.5:
            gripper_action = "open"
        else:
            gripper_action = "none"

        if gripper_action == "close" and self._grasped_object is None:
            # Check if an object is in collision with the end effector marker.
            # If multiple objects are in collision, treat this as a failed grasp.
            objects_in_grasp_zone: set[str] = set()
            # Perform collision detection one-time rather than once per check.
            p.performCollisionDetection(physicsClientId=self.physics_client_id)
            for obj in sorted(self._get_movable_object_names()):
                obj_id = self._object_name_to_pybullet_id(obj)
                if check_body_collisions(
                    obj_id,
                    self.end_effector_viz_id,
                    self.physics_client_id,
                    perform_collision_detection=False,
                ):
                    objects_in_grasp_zone.add(obj)
            # There must be exactly one object in the grasp zone to succeed.
            if len(objects_in_grasp_zone) == 1:
                self._grasped_object = next(iter(objects_in_grasp_zone))
                assert self._grasped_object_id is not None
                # Create grasp transform.
                world_to_robot = self.robot.arm.get_end_effector_pose()
                world_to_object = get_pose(
                    self._grasped_object_id, self.physics_client_id
                )
                self._grasped_object_transform = multiply_poses(
                    world_to_robot.invert(), world_to_object
                )

                while not (
                    check_body_collisions(
                        self._grasped_object_id,
                        self.robot.arm.robot_id,
                        self.physics_client_id,
                    )
                ):
                    # If the fingers are fully closed, stop.
                    intermediate_finger_state = self._robot_arm.get_finger_state()
                    closed_finger_state = self._robot_arm.closed_fingers_state
                    assert isinstance(intermediate_finger_state, float)
                    assert isinstance(closed_finger_state, float)
                    if intermediate_finger_state >= closed_finger_state - 1e-2:
                        break
                    next_finger_state = intermediate_finger_state + 1e-2
                    self._robot_arm.set_finger_state(next_finger_state)
                # Handle the edge case where the robot fingers penetrate the table as
                # the fingers close to grasp the object. This can happen with a gripper
                # that is not just a parallel jaw but has additional DOFs (robotiq).
                # Do not check collision with the tentatively held object.
                collision_bodies = self._get_collision_object_ids()
                collision_bodies -= {self._grasped_object_id}
                if check_collisions_with_held_object(
                    self.robot.arm,
                    collision_bodies,
                    self.physics_client_id,
                    held_object=None,
                    base_link_to_held_obj=self._grasped_object_transform,
                    joint_state=self.robot.arm.get_joint_positions(),
                ):
                    # Revert!
                    self._grasped_object = None
                    self._grasped_object_transform = None
                    self._set_robot_and_held_object(
                        current_base_pose, current_joints, current_finger_state
                    )

        # Check for ungrasping.
        elif gripper_action == "open" and self._grasped_object_id is not None:
            # Check if the held object is being placed on a surface. The rule is that
            # the distance between the object and the surface must be less than thresh.
            surface_supports = self._get_surfaces_supporting_object(
                self._grasped_object_id
            )
            # Placement is successful.
            if surface_supports:
                self._grasped_object = None
                self._grasped_object_transform = None
                self._robot_arm.open_fingers()

        reward = -1
        terminated = self.goal_reached()
        return self._get_obs(), reward, terminated, False, {}

    def render(self) -> NDArray[np.uint8]:  # type: ignore
        return capture_image(
            self.physics_client_id,
            image_width=self.config.render_image_width,
            image_height=self.config.render_image_height,
            **self.config.get_camera_kwargs(),
        )

    def render_base_camera(self) -> NDArray[np.uint8]:
        """Render from the base-mounted camera.

        The camera is attached to the robot base with the same pose as in
        dynamics3d tidybot: pos=(0.2525, 0, 0.335), euler=(0, -45°, -90°).
        """
        # Get current base pose
        base_pose = self.robot.get_base()

        base_pose_se3 = base_pose.to_se3(0.0)
        rot = Rotation.from_euler(
            "zyx", self.config.base_camera_euler
        )  # MuJoCo convention
        camera_to_base_transform = Pose(
            position=self.config.base_camera_offset,
            orientation=tuple(rot.as_quat()[[1, 2, 3, 0]]),  # (w,x,y,z) -> (x,y,z,w)
        )

        camera_pose = multiply_poses(base_pose_se3, camera_to_base_transform)

        # for debugging
        # from pybullet_helpers.gui import visualize_pose
        # visualize_pose(base_pose_se3, self.physics_client_id)
        # visualize_pose(camera_pose, self.physics_client_id)
        return capture_image(
            self.physics_client_id,
            specify_position=True,
            camera_position=camera_pose.position,
            camera_orientation=camera_pose.orientation,
            image_width=self.config.base_camera_image_width,
            image_height=self.config.base_camera_image_height,
            fov=self.config.base_camera_fov,
        )

    def render_ee_camera(self) -> NDArray[np.uint8]:
        """Render from the end-effector mounted camera.

        The camera is attached to the end-effector (wrist) with the same pose as
        in dynamics3d tidybot: pos=(0, -0.05639, -0.058475), quat=(0, 0, 0, 1).
        """
        # Get current end-effector pose
        ee_pose = self.robot.arm.get_end_effector_pose()

        rot = Rotation.from_euler(
            "zyx", self.config.ee_camera_euler
        )  # MuJoCo convention
        camera_to_ee_transform = Pose(
            position=self.config.ee_camera_offset,
            orientation=tuple(rot.as_quat()[[1, 2, 3, 0]]),  # (w,x,y,z) -> (x,y,z,w)
        )

        camera_pose = multiply_poses(ee_pose, camera_to_ee_transform)

        # for debugging
        # from pybullet_helpers.gui import visualize_pose
        # visualize_pose(ee_pose, self.physics_client_id)
        # visualize_pose(camera_pose, self.physics_client_id)

        return capture_image(
            self.physics_client_id,
            specify_position=True,
            camera_position=camera_pose.position,
            camera_orientation=camera_pose.orientation,
            image_width=self.config.ee_camera_image_width,
            image_height=self.config.ee_camera_image_height,
            fov=self.config.ee_camera_fov,
        )

    def render_all_cameras(self) -> dict[str, NDArray[np.uint8]]:
        """Render from all cameras and return as a dictionary.

        Returns:
            Dictionary with keys "overview", "base", "wrist" mapping to images.
        """
        return {
            "overview": self.render(),
            "base": self.render_base_camera(),
            "wrist": self.render_ee_camera(),
        }

    def _set_robot_and_held_object(
        self, base_pose: SE2Pose, joints: JointPositions, finger_state: float
    ) -> None:
        # Look at which objects are inside the box first, if any.
        _, inside_transforms = self._get_inside_objects()
        # First handle the base pose.
        self.robot.set_base(base_pose)
        # First handle the robot arm joints.
        set_robot_joints_with_held_object(
            self._robot_arm,
            self.physics_client_id,
            self._grasped_object_id,
            self._grasped_object_transform,
            extend_joints_to_include_fingers(joints),
        )
        # Handle inside objects (derived from current poses).
        for obj_name, obj_transform in inside_transforms.items():
            obj_id = self._object_name_to_pybullet_id(obj_name)
            set_robot_joints_with_held_object(
                self._robot_arm,
                self.physics_client_id,
                obj_id,
                obj_transform,
                extend_joints_to_include_fingers(joints),
            )
        # Now handle the fingers.
        self._robot_arm.set_finger_state(finger_state)
        # Update the end effector visualization.
        end_effector_pose = self._robot_arm.get_end_effector_pose()
        set_pose(self.end_effector_viz_id, end_effector_pose, self.physics_client_id)

    def _robot_or_held_object_collision_exists(self) -> bool:
        collision_bodies = self._get_collision_object_ids()
        if self._grasped_object_id is not None:
            collision_bodies.discard(self._grasped_object_id)
        collision_bodies -= self._get_inside_object_ids()
        if check_collisions_with_held_object(
            self.robot.arm,
            collision_bodies,
            self.physics_client_id,
            self._grasped_object_id,
            self._grasped_object_transform,
            self.robot.arm.get_joint_positions(),
        ):
            return True
        if not self.config.check_base_collisions:
            return False
        return check_mobile_base_collisions(
            self.robot.base,
            collision_bodies,
            self.physics_client_id,
        )

    def _get_surfaces_supporting_object(self, object_id: int) -> set[int]:
        thresh = self.config.min_placement_dist
        supporting_surface_ids: set[int] = set()
        for surface in self._get_surface_object_names():
            surface_id = self._object_name_to_pybullet_id(surface)
            if check_body_collisions(
                object_id, surface_id, self.physics_client_id, distance_threshold=thresh
            ):
                supporting_surface_ids.add(surface_id)
        return supporting_surface_ids

    def _create_state_dict(
        self, objects: list[tuple[str, Type]]
    ) -> dict[Object, dict[str, float]]:
        state_dict: dict[Object, dict[str, float]] = {}
        for object_name, object_type in objects:
            obj = Object(object_name, object_type)
            feats: dict[str, float] = {}
            # Handle robots.
            if object_type == Kinematic3DRobotType:
                # Add base pose.
                base_pose = self.robot.get_base()
                feats["pos_base_x"] = base_pose.x
                feats["pos_base_y"] = base_pose.y
                feats["pos_base_rot"] = base_pose.rot
                # Add joints.
                joints = remove_fingers_from_extended_joints(
                    self._robot_arm.get_joint_positions()
                )
                for i, v in enumerate(joints):
                    feats[f"joint_{i+1}"] = v
                # Add finger state.
                feats["finger_state"] = self._robot_arm.get_finger_state()
                # Add grasp.
                grasp_tf_feat_names = [
                    "grasp_tf_x",
                    "grasp_tf_y",
                    "grasp_tf_z",
                    "grasp_tf_qx",
                    "grasp_tf_qy",
                    "grasp_tf_qz",
                    "grasp_tf_qw",
                ]
                if self._grasped_object_transform is None:
                    feats["grasp_active"] = 0
                    for feat_name in grasp_tf_feat_names:
                        feats[feat_name] = 0
                else:
                    feats["grasp_active"] = 1
                    grasp_tf_feats = list(
                        self._grasped_object_transform.position
                    ) + list(self._grasped_object_transform.orientation)
                    for feat_name, feat in zip(
                        grasp_tf_feat_names, grasp_tf_feats, strict=True
                    ):
                        feats[feat_name] = feat
            # Handle cuboids.
            elif object_type == Kinematic3DCuboidType:
                # Add pose.
                body_id = self._object_name_to_pybullet_id(object_name)
                pose = get_pose(body_id, self.physics_client_id)
                pose_feat_names = [
                    "pose_x",
                    "pose_y",
                    "pose_z",
                    "pose_qx",
                    "pose_qy",
                    "pose_qz",
                    "pose_qw",
                ]
                pose_feats = list(pose.position) + list(pose.orientation)
                for feat_name, feat in zip(pose_feat_names, pose_feats, strict=True):
                    feats[feat_name] = feat
                # Add grasp active.
                if self._grasped_object == object_name:
                    feats["grasp_active"] = 1
                else:
                    feats["grasp_active"] = 0
                # Add half extents.
                half_extent_names = ["half_extent_x", "half_extent_y", "half_extent_z"]
                half_extents = self._get_half_extents(object_name)
                for feat_name, feat in zip(
                    half_extent_names, half_extents, strict=True
                ):
                    feats[feat_name] = feat
                feats["object_type"] = -1.0  # cuboid
            # Handle points.
            elif object_type == Kinematic3DPointType:
                # Add position.
                body_id = self._object_name_to_pybullet_id(object_name)
                pose = get_pose(body_id, self.physics_client_id)
                feats["x"] = pose.position[0]
                feats["y"] = pose.position[1]
                feats["z"] = pose.position[2]
            # Handle fixtures.
            elif object_type == Kinematic3DFixtureType:
                # Add pose.
                body_id = self._object_name_to_pybullet_id(object_name)
                pose = get_pose(body_id, self.physics_client_id)
                pose_feat_names = [
                    "pose_x",
                    "pose_y",
                    "pose_z",
                    "pose_qx",
                    "pose_qy",
                    "pose_qz",
                    "pose_qw",
                ]
                pose_feats = list(pose.position) + list(pose.orientation)
                for feat_name, feat in zip(pose_feat_names, pose_feats, strict=True):
                    feats[feat_name] = feat
            else:
                raise NotImplementedError(f"Unsupported object type: {object_type}")
            # Add feats to state dict.
            state_dict[obj] = feats
        return state_dict

    def get_action_from_gui_input(
        self, gui_input: dict[str, Any]
    ) -> NDArray[np.float32]:
        """Get the mapping from human inputs to actions."""
        # This will be implemented later
        assert isinstance(self.action_space, Kinematic3DRobotActionSpace)
        return get_robot_action_from_gui_input(self.action_space, gui_input)
