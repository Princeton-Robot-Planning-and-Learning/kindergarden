"""Utilities for the limb repositioning environments."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import numpy as np
import pybullet as p
from numpy.typing import NDArray
from pybullet_helpers.geometry import SE2Pose
from pybullet_helpers.joint import (
    JointInfo,
    JointPositions,
    JointVelocities,
    get_jointwise_difference,
)
from relational_structs import ObjectCentricState

from kinder.envs.utils import RobotActionSpace

JointTorques: TypeAlias = list[float]

ASSETS_DIR = Path(__file__).parent / "assets"
HUMAN_ASSETS_DIR = ASSETS_DIR / "human"
NUM_ROBOT_JOINTS = 7
NUM_LIMB_JOINTS = 6
PYBULLET_TIMESTEP = 1.0 / 240.0


def joint_position_distance(
    joint_infos: list[JointInfo],
    positions: JointPositions,
    other: JointPositions,
    weights: list[float] | None = None,
) -> float:
    """Weighted sum of per-joint differences, wrapping only the circular joints.

    Matches the "weighted_joints" metric in pybullet_helpers.motion_planning.
    """
    difference = get_jointwise_difference(joint_infos, positions, other)
    if weights is None:
        weights = [1.0] * len(joint_infos)
    return float(np.sum(np.multiply(weights, np.abs(difference))))


def _circular_joint_info(index: int) -> JointInfo:
    """A continuous joint, for callers with no physics client to query."""
    # PyBullet marks circular joints with an upper limit below the lower limit.
    return JointInfo(
        jointIndex=index,
        jointName=f"joint_{index}",
        jointType=p.JOINT_REVOLUTE,
        qIndex=-1,
        uIndex=-1,
        flags=0,
        jointDamping=0.0,
        jointFriction=0.0,
        jointLowerLimit=0.0,
        jointUpperLimit=-1.0,
        jointMaxForce=0.0,
        jointMaxVelocity=0.0,
        linkName=f"link_{index}",
        jointAxis=(0.0, 0.0, 1.0),
        parentFramePos=(0.0, 0.0, 0.0),
        parentFrameOrn=(0.0, 0.0, 0.0, 1.0),
        parentIndex=-1,
    )


# Every limb URDF joint is continuous.
LIMB_JOINT_INFOS = [_circular_joint_info(i) for i in range(NUM_LIMB_JOINTS)]


class LimbRepositioning3DObjectCentricState(ObjectCentricState):
    """A state in a limb repositioning environment.

    Inherits from ObjectCentricState but adds some convenient look ups.
    """

    @property
    def robot(self):
        """Assumes there is a unique robot object named "robot"."""
        return self.get_object_from_name("robot")

    @property
    def limb(self):
        """Assumes there is a unique passive limb object named "limb"."""
        return self.get_object_from_name("limb")

    @property
    def base_pose(self) -> SE2Pose:
        """The SE2 pose of the robot's mobile base."""
        return SE2Pose(
            self.get(self.robot, "pos_base_x"),
            self.get(self.robot, "pos_base_y"),
            self.get(self.robot, "pos_base_rot"),
        )

    @property
    def robot_joint_positions(self) -> JointPositions:
        """The robot arm joint positions."""
        return [
            self.get(self.robot, f"joint_{i}") for i in range(1, NUM_ROBOT_JOINTS + 1)
        ]

    @property
    def robot_joint_velocities(self) -> JointVelocities:
        """The robot arm joint velocities."""
        return [
            self.get(self.robot, f"joint_vel_{i}")
            for i in range(1, NUM_ROBOT_JOINTS + 1)
        ]

    @property
    def limb_joint_positions(self) -> JointPositions:
        """The passive limb joint positions."""
        return [
            self.get(self.limb, f"joint_{i}") for i in range(1, NUM_LIMB_JOINTS + 1)
        ]

    @property
    def limb_joint_velocities(self) -> JointVelocities:
        """The passive limb joint velocities."""
        return [
            self.get(self.limb, f"joint_vel_{i}") for i in range(1, NUM_LIMB_JOINTS + 1)
        ]

    @property
    def limb_goal_joint_positions(self) -> JointPositions:
        """The goal joint positions for the passive limb."""
        return [
            self.get(self.limb, f"goal_joint_{i}")
            for i in range(1, NUM_LIMB_JOINTS + 1)
        ]

    @property
    def limb_distance_to_goal(self) -> float:
        """The distance in joint space between the limb and its goal."""
        return joint_position_distance(
            LIMB_JOINT_INFOS, self.limb_joint_positions, self.limb_goal_joint_positions
        )


class LimbRepositioning3DRobotActionSpace(RobotActionSpace):
    """An action space for a torque-controlled 7 DOF robot arm.

    Actions are torques applied to each of the arm joints. The mobile base does not move
    during a repositioning episode, so it is not part of the action.
    """

    def __init__(
        self,
        torque_lower_limits: JointTorques,
        torque_upper_limits: JointTorques,
    ) -> None:
        assert len(torque_lower_limits) == len(torque_upper_limits) == NUM_ROBOT_JOINTS
        super().__init__(np.array(torque_lower_limits), np.array(torque_upper_limits))

    def create_markdown_description(self) -> str:
        """Create a markdown description with a table of action space entries."""
        # Explicit newlines, since docformatter reflows triple-quoted literals.
        lines = [
            "An action space for a torque-controlled 7 DOF robot arm.",
            "",
            "Actions are torques on each arm joint, clipped to the environment's torque "
            "limits. The base does not move, so it is not part of the action.",
            "",
            "| **Index** | **Description** |",
            "| --- | --- |",
        ]
        lines += [
            f"| {i - 1} | torque applied to robot joint {i} |"
            for i in range(1, NUM_ROBOT_JOINTS + 1)
        ]
        return "\n".join(lines)


def get_torque_action_from_gui_input(
    action_space: LimbRepositioning3DRobotActionSpace, gui_input: dict
) -> NDArray[np.float32]:
    """Map human inputs to joint torques, for manual control and debugging.

    Number keys 1-7 select a joint; the left stick's vertical axis sets the sign and
    magnitude of the torque applied to it.
    """
    torques = np.zeros(NUM_ROBOT_JOINTS, dtype=np.float32)
    keys_pressed = gui_input.get("keys", set())
    _, left_y = gui_input.get("left_stick", (0.0, 0.0))
    for i in range(NUM_ROBOT_JOINTS):
        if str(i + 1) in keys_pressed:
            scale = np.where(
                left_y >= 0, action_space.high[i], -action_space.low[i]
            ).item()
            torques[i] = left_y * scale
    return torques
