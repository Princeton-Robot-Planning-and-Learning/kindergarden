"""MassTransport3D with physics: the arm's motors cannot exceed its torque limits.

The scene, observations and actions are those of KinematicMassTransport3D (see
`kinder.envs.dynamic3d.kinematic_masstransport3d`). The difference is how a step is
carried out. The arm tracks its joint targets with position-controlled motors whose
force is capped at the scaled torque limits, and a held object hangs off the gripper
by a fixed constraint with its true mass. An overloaded arm therefore sags or fails to
lift instead of being reverted. While the tote is held, the items in it are pinned to
it the same way, so they load the arm through the tote but cannot slide around.

The base still moves kinematically, carrying what the robot holds with it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pybullet as p
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose, SE2Pose, get_pose, multiply_poses, set_pose
from pybullet_helpers.inverse_kinematics import check_body_collisions
from relational_structs import Array, ObjectCentricState

from kinder.core import ConstantObjectKinDEREnv
from kinder.envs.dynamic3d.kinematic_masstransport3d import (
    NUM_ARM_JOINTS,
    MassTransport3DEnvConfig,
    MassTransport3DObjectCentricState,
    ObjectCentricKinematicMassTransport3DEnv,
    arm_gravity_torques,
)
from kinder.envs.kinematic3d.base_env import ObjectCentricKinematic3DRobotEnv
from kinder.envs.kinematic3d.utils import remove_fingers_from_extended_joints

# The fixed constraint holding a grasped object is far stronger than the arm, so the
# arm's torque limits are what give way.
_GRASP_CONSTRAINT_FORCE = 1e4
# Force on the finger joints, which only hold their opening.
_FINGER_FORCE = 100.0
_ITEM_FRICTION = 1.0


class ObjectCentricMassTransport3DEnv(ObjectCentricKinematicMassTransport3DEnv):
    """Carry items of unknown mass across in a tote, with torque-limited motors."""

    def __init__(
        self,
        num_items: int = 8,
        config: MassTransport3DEnvConfig = MassTransport3DEnvConfig(),
        **kwargs,
    ) -> None:
        super().__init__(num_items=num_items, config=config, **kwargs)
        p.setTimeStep(config.physics_dt, physicsClientId=self.physics_client_id)
        # The end-effector marker is a static body used only to detect grasps; left
        # in contact, it would push back on the arm and on whatever it touches.
        p.setCollisionFilterGroupMask(
            self.end_effector_viz_id, -1, 0, 0, physicsClientId=self.physics_client_id
        )
        self._grasp_constraint: int | None = None
        # Constraints pinning the tote's contents to it while the tote is held.
        self._content_constraints: list[int] = []
        self._arm_targets: list[float] | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[MassTransport3DObjectCentricState, dict]:
        self._release_constraint()
        obs, info = super().reset(seed=seed, options=options)
        self._apply_masses()
        self._arm_targets = None
        if self._grasped_object is not None:
            self._attach(self._grasped_object)
        self._hold_current_joints()
        return obs, info

    def _apply_masses(self) -> None:
        """Give the items their hidden masses, and the tote its mass, in physics."""
        client = self.physics_client_id
        for name, body_id in self._items.items():
            p.changeDynamics(
                body_id,
                -1,
                mass=self._item_masses[name],
                lateralFriction=_ITEM_FRICTION,
                physicsClientId=client,
            )
        p.changeDynamics(
            self.tote_id,
            -1,
            mass=self.config.tote_mass,
            lateralFriction=_ITEM_FRICTION,
            physicsClientId=client,
        )
        for body_id in list(self._items.values()) + [self.tote_id]:
            p.resetBaseVelocity(body_id, (0, 0, 0), (0, 0, 0), physicsClientId=client)

    def _arm_joint_ids(self) -> list[int]:
        return list(self.robot.arm.arm_joints[:NUM_ARM_JOINTS])

    def _finger_joint_ids(self) -> list[int]:
        return list(self.robot.arm.arm_joints[NUM_ARM_JOINTS:])

    def _hold_current_joints(self) -> None:
        """Command the motors to hold the arm and fingers where they are."""
        joints = remove_fingers_from_extended_joints(
            self.robot.arm.get_joint_positions()
        )
        self._command_arm(list(joints))

    def _command_arm(self, arm_targets: list[float]) -> None:
        client = self.physics_client_id
        robot_id = self.robot.arm.robot_id
        self._arm_targets = list(arm_targets)
        p.setJointMotorControlArray(
            robot_id,
            self._arm_joint_ids(),
            p.POSITION_CONTROL,
            targetPositions=arm_targets,
            targetVelocities=[0.0] * NUM_ARM_JOINTS,
            forces=self.config.scaled_torque_limits.tolist(),
            positionGains=[self.config.motor_position_gain] * NUM_ARM_JOINTS,
            velocityGains=[self.config.motor_velocity_gain] * NUM_ARM_JOINTS,
            physicsClientId=client,
        )
        finger_ids = self._finger_joint_ids()
        finger_positions = [
            p.getJointState(robot_id, j, physicsClientId=client)[0] for j in finger_ids
        ]
        p.setJointMotorControlArray(
            robot_id,
            finger_ids,
            p.POSITION_CONTROL,
            targetPositions=finger_positions,
            forces=[_FINGER_FORCE] * len(finger_ids),
            physicsClientId=client,
        )

    def _set_arm_collisions(self, body_id: int, enabled: bool) -> None:
        client = self.physics_client_id
        robot_id = self.robot.arm.robot_id
        for link in range(-1, p.getNumJoints(robot_id, physicsClientId=client)):
            p.setCollisionFilterPair(
                robot_id, body_id, link, -1, int(enabled), physicsClientId=client
            )

    def _attach(self, name: str) -> None:
        """Hang object `name` off the tool link at its current relative pose."""
        client = self.physics_client_id
        arm = self.robot.arm
        body_id = self._object_name_to_pybullet_id(name)
        link_state = p.getLinkState(
            arm.robot_id,
            arm.tool_link_id,
            computeForwardKinematics=True,
            physicsClientId=client,
        )
        # createConstraint wants the parent frame in the link's center-of-mass frame.
        link_com = Pose(link_state[0], link_state[1])
        in_link = multiply_poses(link_com.invert(), get_pose(body_id, client))
        self._grasp_constraint = p.createConstraint(
            arm.robot_id,
            arm.tool_link_id,
            body_id,
            -1,
            p.JOINT_FIXED,
            (0, 0, 0),
            in_link.position,
            (0, 0, 0),
            parentFrameOrientation=in_link.orientation,
            physicsClientId=client,
        )
        p.changeConstraint(
            self._grasp_constraint,
            maxForce=_GRASP_CONSTRAINT_FORCE,
            physicsClientId=client,
        )
        self._set_arm_collisions(body_id, enabled=False)
        if name == "tote":
            for item in sorted(self._get_inside_objects()[0]):
                self._content_constraints.append(
                    self._pin(self._items[item], self.tote_id)
                )

    def _pin(self, child_id: int, parent_id: int) -> int:
        """Fix body `child_id` to body `parent_id` at their current relative pose."""
        client = self.physics_client_id
        in_parent = multiply_poses(
            get_pose(parent_id, client).invert(), get_pose(child_id, client)
        )
        constraint = p.createConstraint(
            parent_id,
            -1,
            child_id,
            -1,
            p.JOINT_FIXED,
            (0, 0, 0),
            in_parent.position,
            (0, 0, 0),
            parentFrameOrientation=in_parent.orientation,
            physicsClientId=client,
        )
        p.changeConstraint(
            constraint, maxForce=_GRASP_CONSTRAINT_FORCE, physicsClientId=client
        )
        return constraint

    def _release_constraint(self) -> None:
        for constraint in self._content_constraints:
            p.removeConstraint(constraint, physicsClientId=self.physics_client_id)
        self._content_constraints = []
        if self._grasp_constraint is None:
            return
        p.removeConstraint(
            self._grasp_constraint, physicsClientId=self.physics_client_id
        )
        self._grasp_constraint = None
        if self._grasped_object is not None:
            body_id = self._object_name_to_pybullet_id(self._grasped_object)
            self._set_arm_collisions(body_id, enabled=True)

    def _try_grasp(self) -> None:
        """Grasp the one object in the grasp zone, if there is exactly one."""
        p.performCollisionDetection(physicsClientId=self.physics_client_id)
        in_zone = [
            name
            for name in sorted(self._get_movable_object_names())
            if check_body_collisions(
                self._object_name_to_pybullet_id(name),
                self.end_effector_viz_id,
                self.physics_client_id,
                perform_collision_detection=False,
            )
        ]
        if len(in_zone) != 1:
            return
        name = in_zone[0]
        body_id = self._object_name_to_pybullet_id(name)
        world_to_ee = self.robot.arm.get_end_effector_pose()
        self._grasped_object = name
        self._grasped_object_transform = multiply_poses(
            world_to_ee.invert(), get_pose(body_id, self.physics_client_id)
        )
        self._attach(name)
        self._robot_arm.close_fingers()

    def _release(self) -> None:
        self._release_constraint()
        self._grasped_object = None
        self._grasped_object_transform = None
        self._robot_arm.open_fingers()

    def _move_base(self, base_action: NDArray[np.float64]) -> None:
        """Teleport the base, carrying the held object and the tote's contents."""
        client = self.physics_client_id
        carried: dict[int, Pose] = {}
        if self._grasped_object is not None:
            world_to_ee = self.robot.arm.get_end_effector_pose()
            names = [self._grasped_object] + sorted(self._get_inside_objects()[0])
            for name in names:
                body_id = self._object_name_to_pybullet_id(name)
                carried[body_id] = multiply_poses(
                    world_to_ee.invert(), get_pose(body_id, client)
                )
        # Same composition as the kinematic env's base step.
        base = self.robot.get_base()
        self.robot.set_base(
            base + SE2Pose(base_action[0], base_action[1], base_action[2])
        )
        world_to_ee = self.robot.arm.get_end_effector_pose()
        for body_id, in_ee in carried.items():
            set_pose(body_id, multiply_poses(world_to_ee, in_ee), client)
            p.resetBaseVelocity(body_id, (0, 0, 0), (0, 0, 0), physicsClientId=client)

    def _measured_torques(self) -> NDArray[np.float64]:
        client = self.physics_client_id
        robot_id = self.robot.arm.robot_id
        return np.array(
            [
                p.getJointState(robot_id, j, physicsClientId=client)[3]
                for j in self._arm_joint_ids()
            ]
        )

    def step(
        self, action: Array
    ) -> tuple[MassTransport3DObjectCentricState, float, bool, bool, dict]:
        config = self.config
        was_on_source_side = self._on_source_side()

        base_action = np.clip(action[:3], -config.max_action_mag, config.max_action_mag)
        if np.any(base_action != 0):
            self._move_base(base_action)

        current = np.asarray(
            remove_fingers_from_extended_joints(self.robot.arm.get_joint_positions())
        )
        delta = np.clip(action[-8:-1], -config.max_action_mag, config.max_action_mag)
        targets = np.clip(
            current + delta,
            self._robot_arm.joint_lower_limits[:NUM_ARM_JOINTS],
            self._robot_arm.joint_upper_limits[:NUM_ARM_JOINTS],
        )

        if action[-1] < -0.5 and self._grasped_object is None:
            self._try_grasp()
        elif action[-1] > 0.5 and self._grasped_object is not None:
            self._release()

        self._command_arm(targets.tolist())
        saturated = False
        limits = config.scaled_torque_limits
        for _ in range(config.physics_substeps):
            p.stepSimulation(physicsClientId=self.physics_client_id)
            torques = self._measured_torques()
            saturated |= bool(
                np.any(np.abs(torques) >= config.saturation_fraction * limits)
            )
        self._torque_violation = saturated

        set_pose(
            self.end_effector_viz_id,
            self.robot.arm.get_end_effector_pose(),
            self.physics_client_id,
        )

        if was_on_source_side and not self._on_source_side():
            self._num_trips += 1
        truncated = config.max_trips is not None and self._num_trips > config.max_trips

        info = self._measurement_info()
        reached = np.asarray(
            remove_fingers_from_extended_joints(self.robot.arm.get_joint_positions())
        )
        info["tracking_error"] = float(np.max(np.abs(reached - targets)))
        return self._get_obs(), -1.0, self.goal_reached(), truncated, info

    def _measurement_info(self) -> dict[str, Any]:
        measured = self._measured_torques()
        noise_std = self.config.torque_noise_std
        if noise_std > 0:
            measured = measured + self.np_random.normal(0.0, noise_std, measured.shape)
        arm_model = arm_gravity_torques(self.robot.arm, self.physics_client_id)
        return {
            "measured_torques": measured,
            "arm_torques": arm_model,
            # What the motors exert beyond the arm's own weight. At rest this is the
            # payload's gravity torque, as in the kinematic env.
            "payload_torques": measured - arm_model,
            "torque_violation": self._torque_violation,
            "num_trips": self._num_trips,
        }

    def _set_state(self, state: MassTransport3DObjectCentricState) -> None:
        self._release_constraint()
        super()._set_state(state)
        if self._grasped_object is not None:
            self._attach(self._grasped_object)
        self._hold_current_joints()


class MassTransport3DEnv(ConstantObjectKinDEREnv):
    """MassTransport3D env with a constant number of items."""

    def __init__(self, num_items: int = 8, **kwargs) -> None:
        self._num_items = num_items
        super().__init__(num_items=num_items, **kwargs)

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricKinematic3DRobotEnv:
        return ObjectCentricMassTransport3DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        names = ["robot", "source_table", "dest_table", "tote"]
        names += sorted(
            (o.name for o in exemplar_state if o.name.startswith("item")),
            key=lambda n: int(n[len("item") :]),
        )
        return names

    def _create_env_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return """A 3D environment where the robot must move every item from a source
               table directly onto a destination table, using a tote to carry several
               items per trip.

               Item masses are hidden. Physics is simulated: the arm tracks joint
               targets with motors capped at its torque limits, so an overloaded arm
               sags or fails to lift. The info dict reports the measured joint torques.
               """

    def _create_variant_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return "The number of items differs between variants. For example, MassTransport3D-o8 has 8 items."

    def _create_variant_specific_description(self) -> str:
        item_str = "1 item" if self._num_items == 1 else f"{self._num_items} items"
        return f"This variant has {item_str} to transport."

    def _create_reward_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return """The reward is a small negative reward (-1) per timestep until
               termination, which occurs when all items rest directly on the destination
               table."""

    def _create_references_markdown_description(self) -> str:
        return """N/A."""
