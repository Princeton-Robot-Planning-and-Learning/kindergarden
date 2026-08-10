"""Base environment class for all Kinematic3Dv2 environments."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Generic
from typing import Type as TypingType
from typing import TypeVar

import gymnasium
import numpy as np
import pybullet as p
from numpy.typing import NDArray
from prpl_kinematics.collision import PyBulletCollisionChecker
from prpl_kinematics.robots import Robot, make_vega
from prpl_kinematics.tree.kinematic_tree import Configuration
from prpl_kinematics.visualization.interface import CameraParams
from prpl_kinematics.visualization.pybullet_renderer import PyBulletRenderer
from relational_structs import Object, ObjectCentricState, ObjectCentricStateSpace, Type
from relational_structs.utils import create_state_from_dict
from spatialmath import SE3

from kinder.core import KinDEREnvConfig, ObjectCentricKinDEREnv, RobotActionSpace
from kinder.envs.kinematic3d_v2.object_types import (
    ARM_NUM_JOINTS,
    Kinematic3Dv2ArmRobotType,
    Kinematic3Dv2EnvTypeFeatures,
    Kinematic3Dv2PointType,
)

# Factories for the robots these environments can be configured with, keyed by the name
# used in Kinematic3Dv2EnvConfig.robot_name.
ROBOT_FACTORIES = {
    "vega": make_vega,
}


@dataclass(frozen=True)
class Kinematic3Dv2EnvConfig(KinDEREnvConfig):
    """Config for Kinematic3Dv2Env()."""

    # Robot. The manipulator names the entry in Robot.manipulators that this environment
    # actuates; every other joint stays at its home value.
    robot_name: str = "vega"
    manipulator: str = "right"

    # Actions are per-joint position deltas, in radians.
    max_action_mag: float = 0.1

    # For rendering.
    render_fps: int = 20
    render_image_width: int = 640
    render_image_height: int = 360
    camera_target: tuple[float, float, float] = (0.30, -0.30, 0.80)
    camera_distance: float = 1.6
    camera_yaw: float = -70.0
    camera_pitch: float = -10.0
    camera_fov: float = 60.0

    def get_camera_params(self) -> CameraParams:
        """The synthetic camera used by render()."""
        return CameraParams(
            target=self.camera_target,
            distance=self.camera_distance,
            yaw=self.camera_yaw,
            pitch=self.camera_pitch,
            width=self.render_image_width,
            height=self.render_image_height,
            fov=self.camera_fov,
        )


class Kinematic3Dv2ObjectCentricState(ObjectCentricState):
    """A state in a Kinematic3Dv2 environment.

    Inherits from ObjectCentricState but adds some convenient look ups.
    """

    @property
    def robot(self) -> Object:
        """Assumes there is a unique robot object named "robot"."""
        return self.get_object_from_name("robot")

    @property
    def arm_joint_positions(self) -> list[float]:
        """The positions of the actuated arm joints."""
        return [
            self.get(self.robot, f"joint_{i}") for i in range(1, ARM_NUM_JOINTS + 1)
        ]


class ArmJointDeltaActionSpace(RobotActionSpace):
    """An action space for an arm with ARM_NUM_JOINTS actuated joints.

    Actions are bounded relative joint positions. There is no gripper or base
    component: these environments move the arm and nothing else.
    """

    def __init__(self, max_magnitude: float = 0.1) -> None:
        low = np.array([-max_magnitude] * ARM_NUM_JOINTS)
        high = np.array([max_magnitude] * ARM_NUM_JOINTS)
        super().__init__(low, high)

    def create_markdown_description(self) -> str:
        """Create a markdown description with a table of action space entries."""
        rows = "\n".join(
            f"| {i} | delta joint {i + 1} |" for i in range(ARM_NUM_JOINTS)
        )
        return f"""An action space for an arm with {ARM_NUM_JOINTS} actuated joints.

Actions are bounded relative joint positions, in radians.

| **Index** | **Description** |
| --- | --- |
{rows}

Deltas are clipped to the configured maximum magnitude and the resulting joint positions
are clipped to the robot's joint limits. A motion that would put the robot in
self-collision is rejected and the arm stays where it was.
"""


_ObsType = TypeVar("_ObsType", bound=Kinematic3Dv2ObjectCentricState)
_ConfigType = TypeVar("_ConfigType", bound=Kinematic3Dv2EnvConfig)


class ObjectCentricKinematic3Dv2RobotEnv(
    ObjectCentricKinDEREnv[_ObsType, NDArray[Any], _ConfigType],
    Generic[_ObsType, _ConfigType],
):
    """Base class for object-centric Kinematic3Dv2 environments.

    The robot is a prpl_kinematics Robot over a KinematicTree. The environment holds one
    Configuration, which is the robot's home with the actuated arm joints overwritten;
    PyBullet is used only to draw that configuration and to check it for self-collision.
    """

    def __init__(
        self, config: _ConfigType, use_gui: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(config=config, **kwargs)
        self.use_gui = use_gui

        # Create the robot.
        self.robot: Robot = ROBOT_FACTORIES[self.config.robot_name]()
        self.tree = self.robot.tree
        self._manipulator = self.robot.manipulators[self.config.manipulator]
        self._arm_space = self.robot.groups[self._manipulator.group]
        assert self._arm_space.dimension == ARM_NUM_JOINTS, (
            f"Manipulator {self.config.manipulator} of {self.config.robot_name} has "
            f"{self._arm_space.dimension} joints; expected {ARM_NUM_JOINTS}."
        )
        self._joint_lower_limits, self._joint_upper_limits = self._arm_space.bounds()

        # The configuration of every joint, arm and otherwise. Non-arm joints keep their
        # home values for the lifetime of the environment.
        self._configuration: dict[str, list[float]] = {
            name: list(values) for name, values in self.robot.home.items()
        }
        self._home_arm_joint_positions = self._arm_space.to_vector(self.robot.home)

        # Subclasses add their scene geometry to the tree here, because the renderer and
        # the collision checker below each create one PyBullet body per shape at load
        # time and never revisit the tree's node set afterwards.
        self._create_scene_geometry()

        # Create the renderer.
        self.physics_client_id = p.connect(p.GUI if use_gui else p.DIRECT)
        self._renderer = PyBulletRenderer(self.physics_client_id)
        self._renderer.load(self.tree)

        # Create the collision checker. It gets its own client because its bodies carry
        # collision shapes but no visual ones, which PyBullet would otherwise draw into
        # the rendered image.
        self._collision_client_id = p.connect(p.DIRECT)
        self._collision_checker = PyBulletCollisionChecker(self._collision_client_id)
        self._collision_checker.load(self.tree)
        self._collision_checker.ignore(self.robot.allowed_collision_pairs)

    @property
    @abc.abstractmethod
    def state_cls(self) -> TypingType[Kinematic3Dv2ObjectCentricState]:
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

    def _create_scene_geometry(self) -> None:
        """Add any non-robot nodes to the kinematic tree.

        Called once, before the renderer and collision checker load the tree.
        """

    @property
    def configuration(self) -> Configuration:
        """The current configuration of every joint in the tree."""
        return self._configuration

    @property
    def arm_joint_positions(self) -> NDArray[np.float64]:
        """The positions of the actuated arm joints."""
        return self._arm_space.to_vector(self._configuration)

    @property
    def end_effector_pose(self) -> SE3:
        """The world-frame pose of the actuated manipulator's end effector."""
        return self.tree.forward_kinematics(
            self._manipulator.ee_frame, self._configuration
        )

    def set_arm_joint_positions(self, positions: NDArray[np.float64]) -> None:
        """Move the actuated arm joints to the given positions."""
        self._configuration.update(self._arm_space.to_configuration(positions))

    @property
    def type_features(self) -> dict[Type, list[str]]:
        """The types and features for this environment."""
        return Kinematic3Dv2EnvTypeFeatures

    def _create_observation_space(self, config: _ConfigType) -> ObjectCentricStateSpace:
        del config  # the observation space is the same for every config
        types = set(self.type_features)
        return ObjectCentricStateSpace(types, state_cls=self.state_cls)

    def _create_action_space(self, config: _ConfigType) -> RobotActionSpace:
        return ArmJointDeltaActionSpace(max_magnitude=config.max_action_mag)

    def _create_constant_initial_state(self) -> _ObsType:
        initial_state_dict = self._create_constant_initial_state_dict()
        state = create_state_from_dict(
            initial_state_dict, Kinematic3Dv2EnvTypeFeatures, state_cls=self.state_cls
        )
        # This is tricky for type annotation because we are dynamically setting the
        # class to be self.state_cls, which should be _ObsType.
        return state  # type: ignore

    def _create_state_dict(
        self, objects: list[tuple[str, Type]]
    ) -> dict[Object, dict[str, float]]:
        state_dict: dict[Object, dict[str, float]] = {}
        for object_name, object_type in objects:
            obj = Object(object_name, object_type)
            if object_type == Kinematic3Dv2ArmRobotType:
                joints = self.arm_joint_positions
                feats = {f"joint_{i + 1}": float(v) for i, v in enumerate(joints)}
            elif object_type == Kinematic3Dv2PointType:
                position = self.tree.forward_kinematics(
                    self._object_name_to_node_name(object_name), self._configuration
                ).t
                feats = {
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "z": float(position[2]),
                }
            else:
                raise ValueError(f"Unrecognized object type: {object_type}")
            state_dict[obj] = feats
        return state_dict

    def _object_name_to_node_name(self, object_name: str) -> str:
        """Look up the kinematic tree node backing a given object name.

        The default assumes objects are added to the tree under their own names.
        """
        return object_name

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
            self.set_arm_joint_positions(self._home_arm_joint_positions)
            self._reset_objects()

        return self._get_obs(), {}

    def step(self, action: NDArray[Any]) -> tuple[_ObsType, float, bool, bool, dict]:
        # Clip the action to be within the allowed limits.
        delta_joints = np.clip(
            action, -self.config.max_action_mag, self.config.max_action_mag
        )
        current_joints = self.arm_joint_positions
        next_joints = np.clip(
            current_joints + delta_joints,
            self._joint_lower_limits,
            self._joint_upper_limits,
        )

        # Tentatively apply the action, reverting if it puts the robot in collision.
        self.set_arm_joint_positions(next_joints)
        if self._collision_checker.in_collision(self._configuration):
            self.set_arm_joint_positions(current_joints)

        terminated = self.goal_reached()
        # Penalize every timestep to encourage reaching the goal quickly.
        reward = -1.0
        return self._get_obs(), reward, terminated, False, {}

    def render(self) -> NDArray[np.uint8]:  # type: ignore
        self._renderer.render(self._configuration)
        return self._renderer.capture_image(self.config.get_camera_params())

    def _get_state(self) -> _ObsType:
        return self._get_obs()

    def _set_state(self, state: _ObsType) -> None:
        """Set the state of the environment to the given one."""
        self.set_arm_joint_positions(np.array(state.arm_joint_positions))
        self._set_object_states(state)

    def close(self) -> None:
        for client_id in (self.physics_client_id, self._collision_client_id):
            if p.isConnected(physicsClientId=client_id):
                p.disconnect(physicsClientId=client_id)
