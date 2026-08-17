"""This module defines the FR3RobotEnv class, which is the base class for the Franka FR3
robot (with a Robotiq 2F-85 gripper) in simulation."""

from pathlib import Path
from typing import Any

import numpy as np
from relational_structs import Array

from kinder.core import RobotActionSpace
from kinder.envs.dynamic3d.mujoco_utils import MjObs
from kinder.envs.dynamic3d.robots.base import IndexedView, RobotEnv
from kinder.envs.dynamic3d.utils import convert_yaw_to_quaternion, quat_to_yaw


class FR3RobotActionSpace(RobotActionSpace):
    """An action in a MuJoCo environment; used to set sim.data.ctrl in MuJoCo."""

    def __init__(self) -> None:
        # FR3 actions: arm joints (7), gripper pos (1)
        low = np.array([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, 0.0])
        high = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0])
        super().__init__(low, high)

    def create_markdown_description(self) -> str:
        """Create a human-readable markdown description of this space."""
        return """Actions: arm joints (7), gripper pos (1)"""


class FR3RobotEnv(RobotEnv):
    """This is the base class for Franka FR3 environments that use MuJoCo for sim.

    The FR3 is a fixed-base arm: it has no mobile base, and its mount point is
    set once per episode via set_robot_base_pos_yaw (e.g., onto a desk surface
    at mount_height). The arm and gripper use MuJoCo position actuators, so
    target joint positions are written directly to ctrl.
    """

    # Home configuration (matches the menagerie "home" keyframe, which is
    # removed from fr3.xml because keyframes do not survive scene merging).
    HOME_QPOS: np.ndarray = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

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
        mount_height: float = 0.0,
    ) -> None:
        """
        Args:
            name: Name of the robot.
            control_frequency: Frequency at which control actions are applied (in Hz).
            act_delta: Whether to interpret actions as deltas or absolute values.
            horizon: Maximum number of steps per episode.
            camera_names: List of camera names for rendering.
            camera_width: Width of camera images.
            camera_height: Height of camera images.
            seed: Random seed for reproducibility.
            show_viewer: Whether to show the MuJoCo viewer.
            mount_height: Height (z) of the fixed base mount in meters, e.g. the
                surface height of the desk the arm is bolted to.
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
        self.mount_height = mount_height

    def _setup_robot_references(self) -> None:
        """Setup references to robot state/actuator buffers in the simulation data."""
        assert self.sim is not None, "Simulation must be initialized."

        arm_joint_names: list[str] = [f"{self.name}_fr3_joint{i}" for i in range(1, 8)]
        # Every joint of the Robotiq 2F-85, in model order. Only the two drivers are
        # actuated; the six passive coupler/spring-link/follower joints carry the grasp
        # geometry, so all eight are needed to say where the fingers actually are.
        gripper_joint_names = [
            f"{self.name}_right_driver_joint",
            f"{self.name}_right_coupler_joint",
            f"{self.name}_right_spring_link_joint",
            f"{self.name}_right_follower_joint",
            f"{self.name}_left_driver_joint",
            f"{self.name}_left_coupler_joint",
            f"{self.name}_left_spring_link_joint",
            f"{self.name}_left_follower_joint",
        ]
        # In fr3.xml, arm actuator names match the joint names.
        arm_actuator_names = arm_joint_names
        gripper_actuator_names = [f"{self.name}_fingers_actuator"]

        arm_qpos_indices = [
            self.sim.model.get_joint_qpos_addr(name) for name in arm_joint_names
        ]
        gripper_qpos_indices = [
            self.sim.model.get_joint_qpos_addr(name) for name in gripper_joint_names
        ]
        arm_qvel_indices = [
            self.sim.model.get_joint_qvel_addr(name) for name in arm_joint_names
        ]
        gripper_qvel_indices = [
            self.sim.model.get_joint_qvel_addr(name) for name in gripper_joint_names
        ]
        arm_ctrl_indices = [
            self.sim.model._actuator_name2id[name]  # pylint: disable=protected-access
            for name in arm_actuator_names
        ]
        gripper_ctrl_indices = [
            self.sim.model._actuator_name2id[name]  # pylint: disable=protected-access
            for name in gripper_actuator_names
        ]

        # Verify indices are contiguous for slicing
        assert arm_qpos_indices == list(
            range(min(arm_qpos_indices), max(arm_qpos_indices) + 1)
        ), "Arm qpos indices not contiguous"
        assert arm_qvel_indices == list(
            range(min(arm_qvel_indices), max(arm_qvel_indices) + 1)
        ), "Arm qvel indices not contiguous"
        assert arm_ctrl_indices == list(
            range(min(arm_ctrl_indices), max(arm_ctrl_indices) + 1)
        ), "Arm ctrl indices not contiguous"

        arm_qpos_start, arm_qpos_end = min(arm_qpos_indices), max(arm_qpos_indices) + 1
        arm_qvel_start, arm_qvel_end = min(arm_qvel_indices), max(arm_qvel_indices) + 1
        arm_ctrl_start, arm_ctrl_end = min(arm_ctrl_indices), max(arm_ctrl_indices) + 1

        self.qpos["arm"] = self.sim.data.mj_data.qpos[arm_qpos_start:arm_qpos_end]
        self.qvel["arm"] = self.sim.data.mj_data.qvel[arm_qvel_start:arm_qvel_end]
        self.ctrl["arm"] = self.sim.data.mj_data.ctrl[arm_ctrl_start:arm_ctrl_end]

        # Use an indexed view to maintain references for
        # non-contiguous gripper indices
        self.qpos["gripper"] = IndexedView(  # type: ignore[assignment]
            self.sim.data.mj_data.qpos, gripper_qpos_indices
        )
        self.qvel["gripper"] = IndexedView(  # type: ignore[assignment]
            self.sim.data.mj_data.qvel, gripper_qvel_indices
        )
        self.ctrl["gripper"] = IndexedView(  # type: ignore[assignment]
            self.sim.data.mj_data.ctrl, gripper_ctrl_indices
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[MjObs, dict[str, Any]]:
        """Reset the robot environment.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options, must contain 'xml' key.

        Returns:
            Tuple of observation and info dict.
        """
        # Access the original xml.
        assert options is not None and "xml" in options, "XML required to reset env"
        xml_string = options["xml"]

        # Insert the robot into the xml string.
        xml_string = self._insert_robot_into_xml(
            xml_string,
            str(Path(__file__).parents[1] / "models" / "franka_fr3"),
            "fr3.xml",
            str(Path(__file__).parents[1] / "models" / "assets"),
        )
        super().reset(seed=seed, options={"xml": xml_string})

        # Setup references to robot state/actuator buffers
        self._setup_robot_references()

        # Move the arm to its home configuration
        self._reset_arm_pose()

        return self.get_obs(), {}

    def set_robot_base_pos_yaw(self, x: float, y: float, yaw: float) -> None:
        """Set the fixed base mount to the specified position and orientation.

        The FR3 has no base joints; instead, the mount body itself is moved in the
        model, with z fixed at mount_height.
        """
        assert (
            self.sim is not None
        ), "Simulation must be initialized before setting base pose."

        body_name = f"{self.name}_base"
        body_id = self.sim.model._body_name2id[  # pylint: disable=protected-access
            body_name
        ]
        self.sim.model.mj_model.body_pos[body_id] = [x, y, self.mount_height]
        self.sim.model.mj_model.body_quat[body_id] = convert_yaw_to_quaternion(yaw)
        self.sim.forward()  # Update the simulation state

    def get_base_pos_yaw(self) -> tuple[float, float, float]:
        """Get the (x, y, yaw) pose of the fixed base mount."""
        assert self.sim is not None, "Simulation must be initialized."
        body_id = self.sim.model._body_name2id[  # pylint: disable=protected-access
            f"{self.name}_base"
        ]
        pos = self.sim.model.mj_model.body_pos[body_id]
        quat = self.sim.model.mj_model.body_quat[body_id]
        return float(pos[0]), float(pos[1]), quat_to_yaw(quat)

    def _reset_arm_pose(self) -> None:
        """Set the arm to the home configuration."""
        assert (
            self.sim is not None
        ), "Simulation must be initialized before resetting arm pose."
        assert self.qpos is not None, "Arm qpos must be initialized first"
        assert self.ctrl is not None, "Arm ctrl must be initialized first"

        self.qpos["arm"][:] = self.HOME_QPOS
        # Position actuators: hold the home configuration
        self.ctrl["arm"][:] = self.HOME_QPOS
        # Ctrl values: 0 = fully open, 255 = fully closed
        self.ctrl["gripper"][:] = 0.0
        self.sim.forward()  # Update the simulation state

    def step(self, action: Array) -> tuple[MjObs, float, bool, bool, dict[str, Any]]:
        """Take a step in the environment.

        The action space is joint positions: arm (7) + gripper (1). The arm
        and gripper use MuJoCo position actuators, so targets are written
        directly to ctrl (targets are clamped to joint limits by MuJoCo via
        actuator ctrlranges).

        Args:
            action: Action array with shape (8,):
                - [0:7]: Arm joint position targets (radians) or deltas
                - [7]: Gripper command in [0, 1] (0=open, 1=closed)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        action = action.copy()

        assert len(action) == 8, f"Action must have 8 elements, got {len(action)}"

        gripper_action = action[7] * 255.0

        # Compute target positions (absolute)
        if self.act_delta:
            # Interpret action as delta, compute absolute targets
            arm_targets = np.array(self.qpos["arm"]) + action[:7]
        else:
            arm_targets = action[:7]

        ctrl_action = np.concatenate([arm_targets, [gripper_action]])

        return super().step(ctrl_action)

    def reward(self, obs: MjObs) -> float:
        """Compute the reward from an observation.

        This is a placeholder implementation since FR3RobotEnv is used as a component in
        Franka3DEnv which handles rewards separately.
        """
        return 0.0
