"""Base environment class for all limb repositioning environments.

These are Dynamic3D environments, but run on PyBullet not the MuJoCo backend
used by the rest of the category.
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
from pybullet_helpers.joint import JointPositions, JointVelocities
from pybullet_helpers.robots import create_pybullet_mobile_robot
from relational_structs import Array, Object, ObjectCentricStateSpace, Type
from relational_structs.utils import create_state_from_dict

from kinder.core import KinDEREnvConfig, ObjectCentricKinDEREnv
from kinder.envs.dynamic3d.limb_object_types import (
    Limb3DEnvTypeFeatures,
    Limb3DLimbType,
    Limb3DRobotType,
)
from kinder.envs.dynamic3d.limb_scenes import (
    LimbRepositioningScene,
    LimbRepositioningSceneConfig,
    create_scene,
)
from kinder.envs.dynamic3d.limb_utils import (
    NUM_LIMB_JOINTS,
    NUM_ROBOT_JOINTS,
    PYBULLET_TIMESTEP,
    JointTorques,
    LimbRepositioning3DObjectCentricState,
    LimbRepositioning3DRobotActionSpace,
)
from kinder.envs.dynamic3d.limbs import HumanLimbPyBulletRobot
from kinder.envs.kinematic3d.utils import extend_joints_to_include_fingers
from kinder.envs.utils import RobotActionSpace

# Clearances are only queried within this radius, and saturate at it.
_CLEARANCE_QUERY_RADIUS = 0.5

# Default scene config for the limb repositioning environment.
_DEFAULT_SCENE = LimbRepositioningSceneConfig(
    scene_type="isolated",
    limb_name="human-right-arm",
    limb_init_joint_positions=(0.0,) * NUM_LIMB_JOINTS,
    limb_goal_joint_positions=(0.0,) * NUM_LIMB_JOINTS,
)


# Friction between the limb and the furniture
LIMB_FURNITURE_FRICTION = 0.5


@dataclass(frozen=True)
class Limb3DEnvConfig(KinDEREnvConfig):
    """Config for a torque-controlled PyBullet environment."""

    scene: LimbRepositioningSceneConfig = _DEFAULT_SCENE
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
    render_image_width: int = 512
    render_image_height: int = 512
    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_distance: float = 2.0
    camera_yaw: float = 140.0
    camera_pitch: float = -20.0

    def __post_init__(self) -> None:
        substeps = self.dt / PYBULLET_TIMESTEP
        assert (
            abs(substeps - round(substeps)) < 1e-9
        ), f"dt={self.dt} is not a whole multiple of PYBULLET_TIMESTEP"
        object.__setattr__(self, "render_fps", round(1.0 / self.dt))

    @property
    def num_substeps(self) -> int:
        """The number of simulation steps taken per environment step."""
        return round(self.dt / PYBULLET_TIMESTEP)

    def get_camera_kwargs(self) -> dict[str, Any]:
        """Get kwargs to pass to the PyBullet camera."""
        return {
            "camera_target": self.camera_target,
            "camera_distance": self.camera_distance,
            "camera_yaw": self.camera_yaw,
            "camera_pitch": self.camera_pitch,
        }


_ConfigType = TypeVar("_ConfigType", bound=Limb3DEnvConfig)


class ObjectCentricLimb3DRobotEnv(
    ObjectCentricKinDEREnv[LimbRepositioning3DObjectCentricState, Array, _ConfigType],
    Generic[_ConfigType],
):
    """Base class for torque-controlled PyBullet environments.

    A robot arm is rigidly attached to a passive human limb and repositions it by
    applying joint torques.

    Collision response is disabled for the robot and for the limb, and gravity is off by
    default, so nothing in the scene pushes back: the limb passes through the bed rather
    than resting on it. Overlap is still measured, by distance queries that ignore the
    collision filters, and reported as a cost by limb_penetration() and
    get_collision_clearance().
    """

    def __init__(self, *args, use_gui: bool = False, **kwargs) -> None:
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
            extend_joints_to_include_fingers(list(self.config.robot_initial_joints))
        )

        # Limbs already overlap the torso at rest, so record that as the baseline.
        self._limb_rest_clearance: dict[int, float] = self._measure_limb_clearance()

        # Move the arm so that it grasps the limb, then weld the two together.
        self._move_robot_to_grasp_limb()
        self._grasp_constraint_id = self._create_grasp_constraint()

        # Torque control requires turning off PyBullet's default joint motors.
        self._prepare_torque_control()

        # Record the joints after settling.
        self._settle()
        self._initial_robot_joints = self._get_robot_joint_positions()

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

    def _get_robot_joint_velocities(self) -> JointVelocities:
        return list(self.robot.arm.get_joint_velocities()[:NUM_ROBOT_JOINTS])

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
        """The closest distance between two bodies, saturating at the query radius."""
        points = p.getClosestPoints(
            body_id,
            other_id,
            _CLEARANCE_QUERY_RADIUS,
            physicsClientId=self.physics_client_id,
        )
        return min((point[8] for point in points), default=_CLEARANCE_QUERY_RADIUS)

    def limb_penetration(self) -> float:
        """How far the limb intrudes past the overlap its model already has at rest."""
        worst = 0.0
        for body_id, distance in self._measure_limb_clearance().items():
            worst = min(worst, distance - self._limb_rest_clearance.get(body_id, 0.0))
        return worst

    def limb_joint_limit_violation(self) -> float:
        """How far the limb's worst joint is outside its limits, in radians."""
        return self.limb.joint_limit_violation(self.limb.get_joint_positions())

    def get_collision_clearance(self) -> float:
        """The smallest distance between the robot or limb and any static scene body.

        Negative values mean penetration, and the result saturates at the query radius.
        """
        clearance = _CLEARANCE_QUERY_RADIUS
        for body_id in self.scene.get_scene_collision_ids():
            for other_id in (self.limb.robot_id, self.robot.arm.robot_id):
                points = p.getClosestPoints(
                    other_id,
                    body_id,
                    clearance,
                    physicsClientId=self.physics_client_id,
                )
                for point in points:
                    # The 9th element of a contact point is the contact distance.
                    clearance = min(clearance, point[8])
        return clearance

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
        """Enable contact between the limb and furniture, also the friction."""
        for body in (self.robot.arm, self.limb):
            p.setJointMotorControlArray(
                body.robot_id,
                body.arm_joints,
                p.VELOCITY_CONTROL,
                forces=np.zeros(len(body.arm_joints)),
                physicsClientId=self.physics_client_id,
            )
            num_joints = p.getNumJoints(
                body.robot_id, physicsClientId=self.physics_client_id
            )
            # Link indices start at -1, the base link, which would otherwise keep its
            # default collision group and damping.
            for joint in range(-1, num_joints):
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

        limb_links = range(
            -1,
            p.getNumJoints(self.limb.robot_id, physicsClientId=self.physics_client_id),
        )
        for furniture_id in self.scene.get_scene_collision_ids():
            furniture_links = range(
                -1, p.getNumJoints(furniture_id, physicsClientId=self.physics_client_id)
            )
            for limb_link in limb_links:
                p.changeDynamics(
                    self.limb.robot_id,
                    limb_link,
                    lateralFriction=LIMB_FURNITURE_FRICTION,
                    physicsClientId=self.physics_client_id,
                )
                for furniture_link in furniture_links:
                    p.setCollisionFilterPair(
                        self.limb.robot_id,
                        furniture_id,
                        limb_link,
                        furniture_link,
                        1,
                        physicsClientId=self.physics_client_id,
                    )

    def _settle(self) -> None:
        """Step the simulation with no applied torque so the scene comes to rest."""
        for _ in range(self.config.num_settle_steps):
            p.stepSimulation(physicsClientId=self.physics_client_id)

    def _apply_torques(
        self, robot_torque: JointTorques, extra_limb_torque: JointTorques | None = None
    ) -> None:
        """Apply torques to both bodies and advance the simulation by one dt."""
        for _ in range(self.config.num_substeps):
            limb_torque = self.limb.get_muscle_tone_torque()
            if extra_limb_torque is not None:
                limb_torque = list(np.add(limb_torque, extra_limb_torque))
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
            p.stepSimulation(physicsClientId=self.physics_client_id)

    @property
    def type_features(self) -> dict[Type, list[str]]:
        return Limb3DEnvTypeFeatures

    @abc.abstractmethod
    def goal_reached(self) -> bool:
        """Check if the goal is currently reached."""

    @abc.abstractmethod
    def _reset_bodies(self) -> None:
        """Reset the robot and the limb to their initial positions and velocities."""

    @property
    def _object_names(self) -> list[tuple[str, Type]]:
        """The objects that appear in the observation."""
        return [("robot", Limb3DRobotType), ("limb", Limb3DLimbType)]

    def _get_object_features(self, object_name: str) -> dict[str, float]:
        """The observed features of a single object."""
        if object_name == "robot":
            base_pose = self.robot.get_base()
            feats = {
                "pos_base_x": base_pose.x,
                "pos_base_y": base_pose.y,
                "pos_base_rot": base_pose.rot,
            }
            for i, value in enumerate(self._get_robot_joint_positions()):
                feats[f"joint_{i + 1}"] = value
            for i, value in enumerate(self._get_robot_joint_velocities()):
                feats[f"joint_vel_{i + 1}"] = value
            return feats
        if object_name == "limb":
            feats = {}
            for i, value in enumerate(self.limb.get_joint_positions()):
                feats[f"joint_{i + 1}"] = value
            for i, value in enumerate(self.limb.get_joint_velocities()):
                feats[f"joint_vel_{i + 1}"] = value
            for i, value in enumerate(self.config.scene.limb_goal_joint_positions):
                feats[f"goal_joint_{i + 1}"] = value
            return feats
        raise ValueError(f"Unknown object: {object_name}")

    def _get_obs(self) -> LimbRepositioning3DObjectCentricState:
        state = create_state_from_dict(
            {
                Object(name, object_type): self._get_object_features(name)
                for name, object_type in self._object_names
            },
            Limb3DEnvTypeFeatures,
            state_cls=LimbRepositioning3DObjectCentricState,
        )
        assert isinstance(state, LimbRepositioning3DObjectCentricState)
        return state

    def _set_state(self, state: LimbRepositioning3DObjectCentricState) -> None:
        """Restore positions and velocities, then rebuild the weld.

        Velocities are part of the state because the dynamics depend on them, so
        get_transition() would otherwise erase the system's momentum.
        """
        self.robot.arm.set_joints(
            extend_joints_to_include_fingers(state.robot_joint_positions),
            joint_velocities=extend_joints_to_include_fingers(
                state.robot_joint_velocities
            ),
        )
        self.limb.set_joints(
            state.limb_joint_positions, joint_velocities=state.limb_joint_velocities
        )
        self._reweld_limb()

    def _get_state(self) -> LimbRepositioning3DObjectCentricState:
        return self._get_obs()

    def _get_reward(self) -> float:
        """One unit of cost per step, unless a subclass shapes the reward."""
        return -1.0

    def _get_extra_limb_torque(self) -> JointTorques | None:
        """Torque applied to the limb on top of its muscle tone, e.g. a spasm."""
        return None

    def step(
        self, action: Array
    ) -> tuple[LimbRepositioning3DObjectCentricState, float, bool, bool, dict]:
        torque = np.clip(
            action, self.torque_action_space.low, self.torque_action_space.high
        )
        self._apply_torques(list(torque), self._get_extra_limb_torque())
        obs = self._get_obs()
        info = {
            "limb_distance_to_goal": obs.limb_distance_to_goal,
            "limb_joint_limit_violation": self.limb_joint_limit_violation(),
        }
        return obs, self._get_reward(), self.goal_reached(), False, info

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
    ) -> tuple[LimbRepositioning3DObjectCentricState, dict]:
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
