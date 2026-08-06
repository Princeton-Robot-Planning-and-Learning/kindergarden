"""Base environment class for all limb repositioning environments.

These are Kinematic3D environments, but use PyBullet forward dynamics with torque
control, rather than the joint position resets used by the rest of the category.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import gymnasium
import numpy as np
import pybullet as p
from numpy.typing import NDArray
from pybullet_helpers.camera import capture_image
from pybullet_helpers.geometry import SE2Pose, multiply_poses
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    inverse_kinematics,
    pybullet_inverse_kinematics,
)
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.robots import create_pybullet_mobile_robot
from relational_structs import Array, ObjectCentricStateSpace, Type

from kinder.core import KinDEREnvConfig, ObjectCentricKinDEREnv
from kinder.envs.kinematic3d.limb_scenes import (
    LimbRepositioningScene,
    LimbRepositioningSceneConfig,
    create_scene,
)
from kinder.envs.kinematic3d.limbs import HumanLimbPyBulletRobot
from kinder.envs.kinematic3d.object_types import (
    Limb3DEnvTypeFeatures,
)
from kinder.envs.kinematic3d.utils import (
    NUM_ROBOT_JOINTS,
    PYBULLET_TIMESTEP,
    JointTorques,
    LimbRepositioning3DObjectCentricState,
    LimbRepositioning3DRobotActionSpace,
)
from kinder.envs.utils import RobotActionSpace


@dataclass(frozen=True)
class Limb3DEnvConfig(KinDEREnvConfig):
    """Config for a torque-controlled PyBullet environment."""

    scene: LimbRepositioningSceneConfig = None  # type: ignore[assignment]
    robot_name: str = "tidybot-kinova"
    robot_base_home_pose: SE2Pose = SE2Pose.identity()
    robot_base_z: float = 0.0

    robot_initial_joints: tuple[float, ...] = (
        0.0,
        -0.35,
        -np.pi,
        -2.5,
        0.0,
        -0.87,
        np.pi / 2,
    )

    # Torque control
    dt: float = 1.0 / 240.0
    torque_lower_limits: tuple[float, ...] = (-1.0,) * NUM_ROBOT_JOINTS
    torque_upper_limits: tuple[float, ...] = (1.0,) * NUM_ROBOT_JOINTS

    # Off by default, so the limb resists only through its inertia and muscle tone.
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    num_settle_steps: int = 1000

    # Rendering.
    render_fps: int = 60
    render_image_width: int = 512
    render_image_height: int = 512
    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_distance: float = 3.0
    camera_yaw: float = 160.0
    camera_pitch: float = 200.0

    def get_camera_kwargs(self) -> dict[str, Any]:
        """Get kwargs to pass to the PyBullet camera."""
        return {
            "camera_target": self.camera_target,
            "camera_distance": self.camera_distance,
            "camera_yaw": self.camera_yaw,
            "camera_pitch": self.camera_pitch,
        }


_ObsType = TypeVar("_ObsType", bound=LimbRepositioning3DObjectCentricState)
_ConfigType = TypeVar("_ConfigType", bound=Limb3DEnvConfig)


class ObjectCentricLimb3DRobotEnv(
    ObjectCentricKinDEREnv[_ObsType, Array, _ConfigType],
    Generic[_ObsType, _ConfigType],
):
    """Base class for torque-controlled PyBullet environments.

    A robot arm is rigidly attached to a passive human limb and repositions it by
    applying joint torques.
    """

    def __init__(
        self, *args, use_gui: bool = False, realistic_bg: bool = False, **kwargs
    ) -> None:
        # Accepted and ignored so the kinematic3d tooling can build this environment.
        del realistic_bg
        super().__init__(*args, **kwargs)
        self.use_gui = use_gui

        if use_gui:
            self.physics_client_id = create_gui_connection(
                **self.config.get_camera_kwargs()
            )
        else:
            self.physics_client_id = p.connect(p.DIRECT)

        p.setPhysicsEngineParameter(
            fixedTimeStep=PYBULLET_TIMESTEP, physicsClientId=self.physics_client_id
        )
        p.setGravity(*self.config.gravity, physicsClientId=self.physics_client_id)

        # Build the static parts of the scene and the human, and create the robot.
        self.scene: LimbRepositioningScene = create_scene(
            self.config.scene, self.physics_client_id
        )
        self.robot = create_pybullet_mobile_robot(
            self.config.robot_name,
            self.physics_client_id,
            base_z=self.config.robot_base_z,
            base_home_pose=self.config.robot_base_home_pose,
        )
        self.robot.arm.set_joints(
            self._extend_joints_with_fingers(list(self.config.robot_initial_joints))
        )

        # Limbs already overlap the torso at rest, so record that as the baseline.
        self._limb_rest_clearance: dict[int, float] = self._measure_limb_clearance()

        # Move the arm so that it grasps the limb, then weld the two together.
        self._move_robot_to_grasp_limb()
        self._grasp_constraint_id = self._create_grasp_constraint()

        # Torque control requires turning off PyBullet's default joint motors.
        self._prepare_torque_control()

        self._initial_robot_joints = self._get_robot_joint_positions()
        self._settle()

    @property
    def limb(self) -> HumanLimbPyBulletRobot:
        """The passive limb that the robot is repositioning."""
        return self.scene.passive_limb

    @property
    def _robot_arm_joints(self) -> list[int]:
        """The PyBullet joint indices of the actuated arm joints, excluding fingers."""
        return list(self.robot.arm.arm_joints[:NUM_ROBOT_JOINTS])

    @property
    def torque_action_space(self) -> LimbRepositioning3DRobotActionSpace:
        """The action space, narrowed from gymnasium's generic Space type."""
        action_space = self.action_space
        assert isinstance(action_space, LimbRepositioning3DRobotActionSpace)
        return action_space

    def _get_robot_joint_positions(self) -> JointPositions:
        return list(self.robot.arm.get_joint_positions()[:NUM_ROBOT_JOINTS])

    def _get_robot_joint_velocities(self) -> JointPositions:
        return list(self.robot.arm.get_joint_velocities()[:NUM_ROBOT_JOINTS])

    @staticmethod
    def _extend_joints_with_fingers(joints: JointPositions) -> JointPositions:
        """Pad arm joint positions with the six Robotiq finger joints."""
        assert len(joints) == NUM_ROBOT_JOINTS
        return list(joints) + [0.0] * 6

    def _move_robot_to_grasp_limb(self) -> None:
        """Place the arm so that its end effector reaches the limb's grasp frame.

        The grasp pose is not always exactly reachable, and the constraint is built from
        wherever the arm ends up, so fall back to a best-effort solution.
        """
        limb_ee_pose = self.limb.get_end_effector_pose()
        robot_ee_pose = multiply_poses(limb_ee_pose, self.scene.grasp_transform)
        try:
            inverse_kinematics(self.robot.arm, robot_ee_pose, validate=False)
        except InverseKinematicsError:
            joint_positions = pybullet_inverse_kinematics(
                self.robot.arm, robot_ee_pose, validate=False, best_effort=True
            )
            self.robot.arm.set_joints(joint_positions)

    def _measure_limb_clearance(self) -> dict[int, float]:
        """Distance from the limb to each obstacle in its current configuration."""
        return {
            body_id: self._closest_distance(self.limb.robot_id, body_id)
            for body_id in self.scene.get_limb_obstacle_ids()
        }

    def _closest_distance(self, body_id: int, other_id: int) -> float:
        points = p.getClosestPoints(
            body_id, other_id, 0.5, physicsClientId=self.physics_client_id
        )
        return min((point[8] for point in points), default=float("inf"))

    def limb_penetration(self) -> float:
        """How far the limb intrudes past the overlap its model already has at rest."""
        worst = 0.0
        for body_id, distance in self._measure_limb_clearance().items():
            worst = min(worst, distance - self._limb_rest_clearance.get(body_id, 0.0))
        return worst

    def _reweld_limb(self) -> None:
        """Rebuild the weld from the current robot and limb poses."""
        p.removeConstraint(
            self._grasp_constraint_id, physicsClientId=self.physics_client_id
        )
        self._grasp_constraint_id = self._create_grasp_constraint()

    def _regrasp_limb(self) -> None:
        """Re-solve the grasp and rebuild the weld, after the limb has been moved."""
        self._move_robot_to_grasp_limb()
        self._reweld_limb()

    def _create_grasp_constraint(self) -> int:
        """Weld the robot's end effector to the limb's end effector."""
        constraint_tf = multiply_poses(
            self.limb.get_end_effector_pose().invert(),
            self.robot.arm.get_end_effector_pose(),
        )
        return p.createConstraint(
            self.robot.arm.robot_id,
            self.robot.arm.end_effector_id,
            self.limb.robot_id,
            self.limb.end_effector_id,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=constraint_tf.position,
            parentFrameOrientation=[0, 0, 0, 1],
            childFrameOrientation=constraint_tf.orientation,
            physicsClientId=self.physics_client_id,
        )

    def _prepare_torque_control(self) -> None:
        """Disable the default motors and joint friction so torques act directly.

        Collisions are disabled for the robot and the limb, so their interaction is
        mediated purely by the grasp constraint. See get_collision_clearance().
        """
        for body in (self.robot.arm, self.limb):
            p.setJointMotorControlArray(
                body.robot_id,
                body.arm_joints,
                p.VELOCITY_CONTROL,
                forces=np.zeros(len(body.arm_joints)),
                physicsClientId=self.physics_client_id,
            )
            for joint in range(
                p.getNumJoints(body.robot_id, physicsClientId=self.physics_client_id)
            ):
                p.enableJointForceTorqueSensor(
                    body.robot_id, joint, 1, physicsClientId=self.physics_client_id
                )
                p.changeDynamics(
                    body.robot_id,
                    joint,
                    jointDamping=0.0,
                    anisotropicFriction=0.0,
                    maxJointVelocity=5000,
                    linearDamping=0.0,
                    angularDamping=0.0,
                    lateralFriction=0.0,
                    spinningFriction=0.0,
                    rollingFriction=0.0,
                    contactStiffness=0.0,
                    contactDamping=0.0,
                    physicsClientId=self.physics_client_id,
                )
                p.setCollisionFilterGroupMask(
                    body.robot_id,
                    joint,
                    0,
                    0,
                    physicsClientId=self.physics_client_id,
                )

    def _settle(self) -> None:
        """Step the simulation with no applied torque so the scene comes to rest."""
        for _ in range(self.config.num_settle_steps):
            p.stepSimulation(physicsClientId=self.physics_client_id)

    def _apply_torques(
        self, robot_torque: JointTorques, limb_torque: JointTorques
    ) -> None:
        """Apply torques to both bodies and advance the simulation by one dt."""
        p.setJointMotorControlArray(
            self.robot.arm.robot_id,
            self._robot_arm_joints,
            p.TORQUE_CONTROL,
            forces=robot_torque,
            physicsClientId=self.physics_client_id,
        )
        p.setJointMotorControlArray(
            self.limb.robot_id,
            self.limb.arm_joints,
            p.TORQUE_CONTROL,
            forces=limb_torque,
            physicsClientId=self.physics_client_id,
        )
        num_substeps = max(1, int((self.config.dt + 1e-5) / PYBULLET_TIMESTEP))
        for _ in range(num_substeps):
            p.stepSimulation(physicsClientId=self.physics_client_id)

    def get_collision_clearance(self) -> float:
        """The smallest distance between the robot or limb and any static scene body.

        Negative values mean penetration.
        """
        clearance = np.inf
        for body_id in self.scene.get_scene_collision_ids():
            for other_id in (self.limb.robot_id, self.robot.arm.robot_id):
                points = p.getClosestPoints(
                    other_id,
                    body_id,
                    float(clearance),
                    physicsClientId=self.physics_client_id,
                )
                for point in points:
                    # The 9th element of a contact point is the contact distance.
                    clearance = min(clearance, point[8])
        return float(clearance)

    @property
    def type_features(self) -> dict[Type, list[str]]:
        return Limb3DEnvTypeFeatures

    @abc.abstractmethod
    def _get_obs(self) -> _ObsType:
        """Get the current observation."""

    @abc.abstractmethod
    def goal_reached(self) -> bool:
        """Check if the goal is currently reached."""

    @abc.abstractmethod
    def _reset_bodies(self) -> None:
        """Reset the robot and the limb to their initial positions and velocities."""

    @abc.abstractmethod
    def _set_state(self, state: _ObsType) -> None:
        """Set the state of the environment to the given one."""

    def _get_state(self) -> _ObsType:
        return self._get_obs()

    def _create_observation_space(self, config: _ConfigType) -> ObjectCentricStateSpace:
        del config
        return ObjectCentricStateSpace(
            set(self.type_features), state_cls=LimbRepositioning3DObjectCentricState
        )

    def _create_action_space(self, config: _ConfigType) -> RobotActionSpace:
        return LimbRepositioning3DRobotActionSpace(
            list(config.torque_lower_limits), list(config.torque_upper_limits)
        )

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[_ObsType, dict]:
        gymnasium.Env.reset(self, seed=seed)
        if options is not None and "init_state" in options:
            self._set_state(options["init_state"])
        else:
            self._reset_bodies()
        return self._get_obs(), {}

    def render(self) -> NDArray[np.uint8]:  # type: ignore
        return capture_image(
            self.physics_client_id,
            image_width=self.config.render_image_width,
            image_height=self.config.render_image_height,
            **self.config.get_camera_kwargs(),
        )

    def close(self) -> None:
        if p.isConnected(physicsClientId=self.physics_client_id):
            p.disconnect(physicsClientId=self.physics_client_id)
