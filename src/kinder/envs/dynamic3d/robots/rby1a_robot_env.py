"""This module defines the RBY1ARobotEnv class, which is the base class for the RBY-1A
robot in simulation."""

from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray
from relational_structs import Array

from kinder.core import RobotActionSpace
from kinder.envs.dynamic3d.mujoco_utils import MjObs
from kinder.envs.dynamic3d.robots.base import RobotEnv


class RBY1ARobotActionSpace(RobotActionSpace):
    """An action in a MuJoCo environment; used to set sim.data.ctrl in MuJoCo."""

    def __init__(self) -> None:
        # Robot actions: joint positions for 2 base joints, 6 torso joints,
        # 7 right arm joints, 7 left arm joints, 2 head joints
        low = np.array([-300] * 24)
        high = np.array([300] * 24)
        super().__init__(low, high)

    def create_markdown_description(self) -> str:
        """Create a human-readable markdown description of this space."""
        return (
            """Actions: joint positions for 2 base joints, 6 torso joints,"""


            """7 right arm joints, 7 left arm joints, 2 head joints"""
        )


class RBY1ARobotEnv(RobotEnv):
    """This is the base class for RBY-1A environments that use MuJoCo for sim.

    It is still abstract: subclasses define rewards and add objects to the env.
    """

    def __init__(
        self,
        name: str,
        control_frequency: float,
        act_delta: bool = True,
        horizon: int = 1000,
        camera_names: list[str] | None = None,
        camera_width: int = 640,
        camera_height: int = 480,
        seed: int | None = None,
        show_viewer: bool = False,
    ) -> None:
        """
        Args:
            name: Name of the robot.
            control_frequency: Frequency at which control actions are applied (in Hz).
            act_delta: Whether to interpret actions as deltas or absolute values.
            horizon: Maximum number of steps per episode.
            camera_names: List of camera names to use for rendering.
            camera_width: Width of camera images.
            camera_height: Height of camera images.
            seed: Random seed for reproducibility.
            show_viewer: Whether to show the MuJoCo viewer.
        """
        super().__init__(
            control_frequency,
            horizon=horizon,
            camera_names=camera_names,
            camera_width=camera_width,
            camera_height=camera_height,
            seed=seed,
            show_viewer=show_viewer,
        )

        self.name = name
        self.act_delta = act_delta

        # Initialize robot state attributes
        self.joint_indices: list[int] = []
        self.joint_indices_ctrl: list[int] = []
        self.exclude_parts: list[str] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[MjObs, dict[str, Any]]:
        """Reset the RBY-1A robot environment.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options for resetting the environment.

        Returns:
            A tuple containing the observation and info dict.
        """
        # Access the original xml.
        assert options is not None and "xml" in options, "XML required to reset env"
        xml_string = options["xml"]
        # Insert the robot into the xml string.
        xml_string = self._insert_robot_into_xml(
            xml_string,
            str(Path(__file__).parents[1] / "models" / "rby1a"),
            "rby1a_model_v1.2.xml",
            str(Path(__file__).parents[1] / "models" / "rby1a"),
        )
        super().reset(seed=seed, options={"xml": xml_string})

        # Setup references to robot state/actuator buffers
        self._setup_robot_references()

        # Randomize the base pose of the robot in the sim
        self._randomize_base_pose()
        self._randomize_arm_and_torso_pose()

        return self.get_obs(), {}

    def _setup_robot_references(self) -> None:
        """Setup references to robot state/actuator buffers."""
        assert self.sim is not None, "Simulation must be initialized."

        robot_joint_names = {
            "base": [f"{self.name}_right_wheel", f"{self.name}_left_wheel"],
            "torso": [
                f"{self.name}_torso_0",
                f"{self.name}_torso_1",
                f"{self.name}_torso_2",
                f"{self.name}_torso_3",
                f"{self.name}_torso_4",
                f"{self.name}_torso_5",
            ],
            "right_arm": [
                f"{self.name}_right_arm_0",
                f"{self.name}_right_arm_1",
                f"{self.name}_right_arm_2",
                f"{self.name}_right_arm_3",
                f"{self.name}_right_arm_4",
                f"{self.name}_right_arm_5",
                f"{self.name}_right_arm_6",
            ],
            "left_arm": [
                f"{self.name}_left_arm_0",
                f"{self.name}_left_arm_1",
                f"{self.name}_left_arm_2",
                f"{self.name}_left_arm_3",
                f"{self.name}_left_arm_4",
                f"{self.name}_left_arm_5",
                f"{self.name}_left_arm_6",
            ],
            "head": [f"{self.name}_head_0", f"{self.name}_head_1"],
        }
        robot_actuator_names = {
            "base": [f"{self.name}_right_wheel_act", f"{self.name}_left_wheel_act"],
            "torso": [
                f"{self.name}_link1_act",
                f"{self.name}_link2_act",
                f"{self.name}_link3_act",
                f"{self.name}_link4_act",
                f"{self.name}_link5_act",
                f"{self.name}_link6_act",
            ],
            "right_arm": [
                f"{self.name}_right_arm_1_act",
                f"{self.name}_right_arm_2_act",
                f"{self.name}_right_arm_3_act",
                f"{self.name}_right_arm_4_act",
                f"{self.name}_right_arm_5_act",
                f"{self.name}_right_arm_6_act",
                f"{self.name}_right_arm_7_act",
            ],
            "left_arm": [
                f"{self.name}_left_arm_1_act",
                f"{self.name}_left_arm_2_act",
                f"{self.name}_left_arm_3_act",
                f"{self.name}_left_arm_4_act",
                f"{self.name}_left_arm_5_act",
                f"{self.name}_left_arm_6_act",
                f"{self.name}_left_arm_7_act",
            ],
            "head": [f"{self.name}_head_0_act", f"{self.name}_head_1_act"],
        }

        # Joint positions: joint_id corresponds to qpos index
        qpos_indices = {
            part: [
                self.sim.model.get_joint_qpos_addr(joint_name)
                for joint_name in joint_names
            ]
            for part, joint_names in robot_joint_names.items()
        }

        # Joint velocities: joint_id corresponds to qvel index
        qvel_indices = {
            part: [
                self.sim.model.get_joint_qvel_addr(joint_name)
                for joint_name in joint_names
            ]
            for part, joint_names in robot_joint_names.items()
        }

        # Actuators: actuator_id corresponds to ctrl index
        ctrl_indices = {
            part: [
                self.sim.model._actuator_name2id[  # pylint: disable=protected-access
                    actuator_name
                ]
                for actuator_name in actuator_names
            ]
            for part, actuator_names in robot_actuator_names.items()
        }

        # Verify indices are contiguous for slicing
        for part in qpos_indices:
            indices = qpos_indices[part]
            assert indices == list(
                range(min(indices), max(indices) + 1)
            ), f"Non-contiguous qpos indices for part {part}"
        for part in qvel_indices:
            indices = qvel_indices[part]
            assert indices == list(
                range(min(indices), max(indices) + 1)
            ), f"Non-contiguous qvel indices for part {part}"
        for part in ctrl_indices:
            indices = ctrl_indices[part]
            assert indices == list(
                range(min(indices), max(indices) + 1)
            ), f"Non-contiguous ctrl indices for part {part}"

        # Create views using correct slice ranges
        qpos_start_end = {
            part: (min(indices), max(indices) + 1)
            for part, indices in qpos_indices.items()
        }
        qvel_start_end = {
            part: (min(indices), max(indices) + 1)
            for part, indices in qvel_indices.items()
        }
        ctrl_start_end = {
            part: (min(indices), max(indices) + 1)
            for part, indices in ctrl_indices.items()
        }

        self.qpos = {
            part: self.sim.data.mj_data.qpos[start:end]
            for part, (start, end) in qpos_start_end.items()
        }
        self.qvel = {
            part: self.sim.data.mj_data.qvel[start:end]
            for part, (start, end) in qvel_start_end.items()
        }
        self.ctrl = {
            part: self.sim.data.mj_data.ctrl[start:end]
            for part, (start, end) in ctrl_start_end.items()
        }

        # Store all joint indices (in qvel) for which joint torques will be computed.
        self.joint_indices.clear()
        self.joint_indices_ctrl.clear()  # This could be used to set ctrl directly
        self.exclude_parts = ["base"]  # Exclude base joints from jacobian
        for part in qvel_indices:
            if part not in self.exclude_parts:  # exclude base joints from jacobian
                self.joint_indices.extend(qvel_indices[part])
                self.joint_indices_ctrl.extend(ctrl_indices[part])

    def set_robot_base_pos_yaw(self, x: float, y: float, yaw: float) -> None:
        """Set the robot's base position and yaw orientation.

        Args:
            x: X position of the robot base .
            y: Y position of the robot base.
            yaw: Yaw orientation of the robot base.
        """

    def _randomize_base_pose(self) -> None:
        """Randomize the base pose of the robot within defined limits."""
        assert (
            self.sim is not None
        ), "Simulation must be initialized before randomizing base pose."
        assert self.qpos["base"] is not None, "Base qpos must be initialized first"
        assert self.ctrl["base"] is not None, "Base ctrl must be initialized first"

        # Define limits for x, y, and theta
        left_limit = (-1.0, 1.0)
        right_limit = (-1.0, 1.0)
        # Sample random values within the limits
        left = self.np_random.uniform(*left_limit)
        right = self.np_random.uniform(*right_limit)
        # Set the base position and orientation in the simulation
        self.qpos["base"][:] = [left, right]
        self.ctrl["base"][:] = [left, right]
        self.sim.forward()  # Update the simulation state

    def _randomize_arm_and_torso_pose(self) -> None:
        """Randomize the arm and torso pose of the robot within defined limits."""
        assert (
            self.sim is not None
        ), "Simulation must be initialized before randomizing arm and torso pose."
        assert self.qpos["torso"] is not None, "Torso qpos must be initialized first"
        assert self.ctrl["torso"] is not None, "Torso ctrl must be initialized first"
        assert (
            self.qpos["right_arm"] is not None
        ), "Right arm qpos must be initialized first"
        assert (
            self.ctrl["right_arm"] is not None
        ), "Right arm ctrl must be initialized first"
        assert (
            self.qpos["left_arm"] is not None
        ), "Left arm qpos must be initialized first"
        assert (
            self.ctrl["left_arm"] is not None
        ), "Left arm ctrl must be initialized first"

        # Initial pose for torso and arms
        torso_pose = np.deg2rad([0.0, 45.0, -90.0, 45.0, 0.0, 0.0])
        right_arm_pose = np.deg2rad([0.0, -5.0, 0.0, -120.0, 0.0, 70.0, 0.0])
        left_arm_pose = np.deg2rad([0.0, 5.0, 0.0, -120.0, 0.0, 70.0, 0.0])

        # Define limits for torso and arms
        torso_limits = [(-0.5, 0.5)] * 6  # Example limits for 6 DOF torso
        arm_limits = [(-1.0, 1.0)] * 7  # Example limits for 7 DOF arms

        # Randomize the torso and arm poses within defined limits
        torso_pose = np.clip(
            torso_pose
            + np.deg2rad(
                [self.np_random.uniform() * 10 - 5 for _ in range(len(torso_pose))]
            ),
            [low for low, high in torso_limits],
            [high for low, high in torso_limits],
        )
        right_arm_pose = np.clip(
            right_arm_pose
            + np.deg2rad(
                [self.np_random.uniform() * 10 - 5 for _ in range(len(right_arm_pose))]
            ),
            [low for low, high in arm_limits],
            [high for low, high in arm_limits],
        )
        left_arm_pose = np.clip(
            left_arm_pose
            + np.deg2rad(
                [self.np_random.uniform() * 10 - 5 for _ in range(len(left_arm_pose))]
            ),
            [low for low, high in arm_limits],
            [high for low, high in arm_limits],
        )

        # Set the torso and arm positions in the simulation
        self.qpos["torso"][:] = torso_pose
        self.ctrl["torso"][:] = torso_pose
        self.qpos["right_arm"][:] = right_arm_pose
        self.ctrl["right_arm"][:] = right_arm_pose
        self.qpos["left_arm"][:] = left_arm_pose
        self.ctrl["left_arm"][:] = left_arm_pose

        self.sim.forward()  # Update the simulation state

    @property
    def jacobian_mat(self) -> NDArray[np.float64]:
        """Returns the pos and ori jacobian for the robot joints."""
        assert self.sim is not None, "Simulation must be initialized."
        body_name = "EE_BODY_R"  # End-effector body name (using right arm only)
        jacobian_pos = self.sim.data.get_body_jacp(  # type: ignore[no-untyped-call]
            body_name
        )[
            :, self.joint_indices
        ]  # (3, num_joints)
        jacobian_ori = self.sim.data.get_body_jacr(  # type: ignore[no-untyped-call]
            body_name
        )[
            :, self.joint_indices
        ]  # (3, num_joints)
        jacobian = np.concatenate([jacobian_pos, jacobian_ori], 0)  # (6, num_joints)
        return jacobian

    @property
    def mass_mat(self) -> NDArray[np.float64]:
        """Returns the mass matrix for the robot joints."""
        assert self.sim is not None, "Simulation must be initialized."
        mass_matrix: NDArray[np.float64] = np.ndarray(
            shape=(
                self.sim.model.mj_model.nv,
                self.sim.model.mj_model.nv,
            ),
            dtype=np.float64,
        )
        mujoco.mj_fullM(  # pylint: disable=no-member
            self.sim.model.mj_model,
            mass_matrix,
            self.sim.data.mj_data.qM,
        )
        mass_matrix = np.reshape(
            mass_matrix,
            (
                self.sim.model.mj_model.nv,
                self.sim.model.mj_model.nv,
            ),
        )
        mass_matrix = mass_matrix[self.joint_indices, :][:, self.joint_indices]
        return mass_matrix

    @property
    def lambda_mat(self) -> NDArray[np.float64]:
        """Returns the lambda matrix for the robot."""
        jacobian = self.jacobian_mat
        mass_matrix_inv = np.linalg.inv(self.mass_mat)

        # J M^-1 J^T
        lambda_full_inv = np.dot(
            np.dot(jacobian, mass_matrix_inv), jacobian.transpose()
        )

        # take the inverses, but zero out small singular values for stability
        lambda_full = np.linalg.pinv(lambda_full_inv)

        return lambda_full

    @property
    def torque_compensation(self) -> NDArray[np.float64]:
        """Return torque compensation values."""
        assert self.sim is not None, "Simulation must be initialized."
        return self.sim.data.mj_data.qfrc_bias[  # pylint: disable=protected-access
            self.joint_indices
        ]

    def step(self, action: Array) -> tuple[MjObs, float, bool, bool, dict[str, Any]]:
        """Step the RBY-1A robot environment with the given action.

        Args:
            action: The action to take in the environment.

        Returns:
            A tuple containing (observation, reward, terminated, truncated, info).
        """
        if self.act_delta:  # Interpret action as delta.
            # Compute absolute joint action.
            curr_qpos = np.concatenate([self.qpos[part] for part in self.qpos], -1)
            abs_action = curr_qpos + action
            return super().step(abs_action)
        # Use action as-is.
        return super().step(action)

    def reward(self, obs: MjObs) -> float:
        """Compute the reward from an observation.

        This is a placeholder implementation for the RBY-1A robot.

        Args:
            obs: The observation to compute reward from.

        Returns:
            The computed reward value.
        """
        # Placeholder reward - always returns 0.0
        return 0.0
