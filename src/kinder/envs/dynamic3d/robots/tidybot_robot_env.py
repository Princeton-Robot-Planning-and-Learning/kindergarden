"""This module defines the TidyBotRobotEnv class, which is the base class for the
TidyBot robot in simulation."""

from pathlib import Path
from typing import Any

import numpy as np
from relational_structs import Array

from kinder.core import RobotActionSpace
from kinder.envs.dynamic3d.mujoco_utils import MjObs
from kinder.envs.dynamic3d.robots.base import IndexedView, RobotEnv


class TidyBot3DRobotActionSpace(RobotActionSpace):
    """An action in a MuJoCo environment; used to set sim.data.ctrl in MuJoCo."""

    def __init__(self) -> None:
        # TidyBot actions: base pos and yaw (3), arm joints (7), gripper pos (1)
        low = np.array(
            [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, 0.0]
        )
        high = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0])
        super().__init__(low, high)

    def create_markdown_description(self) -> str:
        """Create a human-readable markdown description of this space."""
        return """Actions: base pos and yaw (3), arm joints (7), gripper pos (1)"""


class TidyBotRobotEnv(RobotEnv):
    """This is the base class for TidyBot environments that use MuJoCo for sim.

    It is still abstract: subclasses define rewards and add objects to the env.

    The arm uses torque control with a PD controller that converts target joint
    positions to torques. The base uses position control directly.
    """

    # PD gains for arm torque control
    # With gravity compensation, PD control provides stable and accurate tracking
    # Lower gains for stability - can tune higher for better tracking if needed
    ARM_KP: np.ndarray = np.array([200.0, 200.0, 200.0, 200.0, 100.0, 100.0, 100.0])
    ARM_KD: np.ndarray = np.array([30.0, 30.0, 30.0, 30.0, 15.0, 15.0, 15.0])

    # Torque limits (Nm) - must match tidybot.xml actuator ctrlrange
    ARM_TORQUE_LIMITS: np.ndarray = np.array(
        [200.0, 200.0, 200.0, 200.0, 100.0, 100.0, 100.0]
    )

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
        arm_kp: np.ndarray | None = None,
        arm_kd: np.ndarray | None = None,
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
            arm_kp: Custom proportional gains for arm PD controller (7 values).
            arm_kd: Custom derivative gains for arm PD controller (7 values).
        """
        robot_camera_names = [f"{name}_base", f"{name}_wrist"]
        if camera_names is None:
            camera_names = []
        camera_names.extend(robot_camera_names)

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

        # Allow custom PD gains
        self.arm_kp = arm_kp if arm_kp is not None else self.ARM_KP.copy()
        self.arm_kd = arm_kd if arm_kd is not None else self.ARM_KD.copy()

        # Initialize arm qvel start and end indices
        self._arm_qvel_start: int = 0
        self._arm_qvel_end: int = 0

    def _setup_robot_references(self) -> None:
        """Setup references to robot state/actuator buffers in the simulation data."""
        assert self.sim is not None, "Simulation must be initialized."

        # Joint names for the base and arm
        base_joint_names: list[str] = [
            f"{self.name}_joint_x",
            f"{self.name}_joint_y",
            f"{self.name}_joint_th",
        ]
        arm_joint_names: list[str] = [
            f"{self.name}_joint_1",
            f"{self.name}_joint_2",
            f"{self.name}_joint_3",
            f"{self.name}_joint_4",
            f"{self.name}_joint_5",
            f"{self.name}_joint_6",
            f"{self.name}_joint_7",
        ]
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
        gripper_ctrl_joint_names = [f"{self.name}_fingers_actuator"]

        # Joint positions: joint_id corresponds to qpos index
        base_qpos_indices = [
            self.sim.model.get_joint_qpos_addr(name) for name in base_joint_names
        ]
        arm_qpos_indices = [
            self.sim.model.get_joint_qpos_addr(name) for name in arm_joint_names
        ]
        gripper_qpos_indices = [
            self.sim.model.get_joint_qpos_addr(name) for name in gripper_joint_names
        ]

        # Joint velocities: joint_id corresponds to qvel index
        base_qvel_indices = [
            self.sim.model.get_joint_qvel_addr(name) for name in base_joint_names
        ]
        arm_qvel_indices = [
            self.sim.model.get_joint_qvel_addr(name) for name in arm_joint_names
        ]
        gripper_qvel_indices = [
            self.sim.model.get_joint_qvel_addr(name) for name in gripper_joint_names
        ]

        # Actuators: actuator_id corresponds to ctrl index
        base_ctrl_indices = [
            self.sim.model._actuator_name2id[name]  # pylint: disable=protected-access
            for name in base_joint_names
        ]
        arm_ctrl_indices = [
            self.sim.model._actuator_name2id[name]  # pylint: disable=protected-access
            for name in arm_joint_names
        ]
        gripper_ctrl_indices = [
            self.sim.model._actuator_name2id[name]  # pylint: disable=protected-access
            for name in gripper_ctrl_joint_names
        ]

        # Verify indices are contiguous for slicing
        assert base_qpos_indices == list(
            range(min(base_qpos_indices), max(base_qpos_indices) + 1)
        ), "Base qpos indices not contiguous"
        assert arm_qpos_indices == list(
            range(min(arm_qpos_indices), max(arm_qpos_indices) + 1)
        ), "Arm qpos indices not contiguous"
        assert base_qvel_indices == list(
            range(min(base_qvel_indices), max(base_qvel_indices) + 1)
        ), "Base qvel indices not contiguous"
        assert arm_qvel_indices == list(
            range(min(arm_qvel_indices), max(arm_qvel_indices) + 1)
        ), "Arm qvel indices not contiguous"
        assert base_ctrl_indices == list(
            range(min(base_ctrl_indices), max(base_ctrl_indices) + 1)
        ), "Base ctrl indices not contiguous"
        assert arm_ctrl_indices == list(
            range(min(arm_ctrl_indices), max(arm_ctrl_indices) + 1)
        ), "Arm ctrl indices not contiguous"

        # Create views using correct slice ranges
        base_qpos_start, base_qpos_end = (
            min(base_qpos_indices),
            max(base_qpos_indices) + 1,
        )
        base_qvel_start, base_qvel_end = (
            min(base_qvel_indices),
            max(base_qvel_indices) + 1,
        )
        arm_qpos_start, arm_qpos_end = min(arm_qpos_indices), max(arm_qpos_indices) + 1
        arm_qvel_start, arm_qvel_end = (
            min(arm_qvel_indices),
            max(arm_qvel_indices) + 1,
        )
        base_ctrl_start, base_ctrl_end = (
            min(base_ctrl_indices),
            max(base_ctrl_indices) + 1,
        )
        arm_ctrl_start, arm_ctrl_end = min(arm_ctrl_indices), max(arm_ctrl_indices) + 1

        self.qpos["base"] = self.sim.data.mj_data.qpos[base_qpos_start:base_qpos_end]
        self.qvel["base"] = self.sim.data.mj_data.qvel[base_qvel_start:base_qvel_end]
        self.ctrl["base"] = self.sim.data.mj_data.ctrl[base_ctrl_start:base_ctrl_end]

        self.qpos["arm"] = self.sim.data.mj_data.qpos[arm_qpos_start:arm_qpos_end]
        self.qvel["arm"] = self.sim.data.mj_data.qvel[arm_qvel_start:arm_qvel_end]
        self.ctrl["arm"] = self.sim.data.mj_data.ctrl[arm_ctrl_start:arm_ctrl_end]

        # Store arm qvel indices for gravity compensation lookup
        self._arm_qvel_start = arm_qvel_start
        self._arm_qvel_end = arm_qvel_end

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
            str(Path(__file__).parents[1] / "models" / "stanford_tidybot"),
            "tidybot.xml",
            str(Path(__file__).parents[1] / "models" / "assets"),
        )
        super().reset(seed=seed, options={"xml": xml_string})

        # Setup references to robot state/actuator buffers
        self._setup_robot_references()

        # Randomize the arm pose of the robot in the sim
        self._randomize_arm_pose()

        return self.get_obs(), {}

    def set_robot_base_pos_yaw(self, x: float, y: float, yaw: float) -> None:
        """Set the base pose of the robot to the specified position and orientation."""
        assert (
            self.sim is not None
        ), "Simulation must be initialized before setting base pose."
        assert self.qpos is not None, "Base qpos must be initialized first"
        assert self.ctrl is not None, "Base ctrl must be initialized first"

        # Set the base position and orientation in the simulation
        self.qpos["base"][:] = [x, y, yaw]
        self.ctrl["base"][:] = [x, y, yaw]
        self.sim.forward()  # Update the simulation state

    def _randomize_arm_pose(self) -> None:
        """Randomize the arm pose of the robot within defined limits."""
        assert (
            self.sim is not None
        ), "Simulation must be initialized before randomizing base pose."
        assert self.qpos is not None, "Base qpos must be initialized first"
        assert self.ctrl is not None, "Base ctrl must be initialized first"

        # set to the retract configuration
        theta = np.deg2rad([0, -20, 180, -146, 0, -50, 90])
        # Set the arm joint positions in the simulation
        self.qpos["arm"][:] = theta
        # For torque control, set initial torque to 0 (will be computed by PD controller)
        self.ctrl["arm"][:] = 0.0
        self.sim.forward()  # Update the simulation state

    def _get_gravity_compensation(self) -> np.ndarray:
        """Get gravity compensation torques for the arm joints.

        MuJoCo stores the bias forces (gravity + Coriolis) in qfrc_bias.
        These torques, when applied, will counteract gravity.

        Returns:
            Gravity compensation torques for the 7 arm joints (Nm).
        """
        assert self.sim is not None, "Simulation must be initialized."
        # qfrc_bias contains C(q,v)*v + g(q) - the forces needed to counteract
        # gravity and Coriolis effects
        return self.sim.data.mj_data.qfrc_bias[
            self._arm_qvel_start : self._arm_qvel_end
        ].copy()

    def _compute_arm_torques(
        self,
        target_positions: np.ndarray,
        target_velocities: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute arm torques using PD control with gravity compensation.

        Uses the formula (position-only mode, target_velocities=None):
            torque = Kp * (target_pos - current_pos) - Kd * current_vel + gravity_comp

        Or with velocity tracking (target_velocities provided):
            torque = Kp * (target_pos - current_pos) +
            Kd * (target_vel - current_vel) + gravity_comp

        Gravity compensation counteracts gravitational forces, allowing
        accurate position tracking with just PD control. When target velocities
        are provided, the controller tracks both position and velocity, which is
        useful for dynamic manipulation tasks like tossing.

        Args:
            target_positions: Target joint positions for the 7 arm joints (radians).
            target_velocities: Target joint velocities for the 7 arm joints (rad/s).
                If None, uses damping mode (equivalent to target velocity of 0).

        Returns:
            Torques to apply to the arm joints (Nm), clipped to actuator limits.
        """
        current_positions = np.array(self.qpos["arm"])
        current_velocities = np.array(self.qvel["arm"])

        # Position error term
        position_error = target_positions - current_positions

        # Velocity term: damping mode vs tracking mode
        if target_velocities is None:
            # Damping mode: resist current velocity (equivalent to tracking vel=0)
            velocity_term = -self.arm_kd * current_velocities
        else:
            # Velocity tracking mode: track desired velocity
            velocity_error = target_velocities - current_velocities
            velocity_term = self.arm_kd * velocity_error

        pd_torques = self.arm_kp * position_error + velocity_term

        # Add gravity compensation (feedforward term)
        gravity_comp = self._get_gravity_compensation()
        torques = pd_torques + gravity_comp

        # Clip torques to actuator limits
        torques = np.clip(torques, -self.ARM_TORQUE_LIMITS, self.ARM_TORQUE_LIMITS)

        return torques

    def step(self, action: Array) -> tuple[MjObs, float, bool, bool, dict[str, Any]]:
        """Take a step in the environment.

        The action space is joint positions: base (3) + arm (7) + gripper (1).
        Optionally, arm velocity targets can be provided for dynamic manipulation.
        - Base: Uses position control directly (MuJoCo position actuators)
        - Arm: Converts target positions (and optionally velocities) to torques
        - Gripper: Uses tendon control with force range [0, 255]

        Args:
            action: Action array with shape (11,) or (18,). Or a 2-D array of
                such rows covering the control period, which MujocoEnv.step
                reads as a control schedule.

                Position-only mode (11,) - backward compatible:
                    - [0:3]: Base position targets (x, y, theta) or deltas
                    - [3:10]: Arm joint position targets (radians) or deltas
                    - [10]: Gripper command in [0, 1] (0=open, 1=closed)

                Position+Velocity mode (18,) - for dynamic manipulation:
                    - [0:3]: Base position targets (x, y, theta) or deltas
                    - [3:10]: Arm joint position targets (radians) or deltas
                    - [10]: Gripper command in [0, 1] (0=open, 1=closed)
                    - [11:18]: Arm joint velocity targets (rad/s)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        if np.ndim(action) == 2:
            return super().step(np.stack([self._action_to_ctrl(r) for r in action]))
        return super().step(self._action_to_ctrl(action))

    def _action_to_ctrl(self, action: Array) -> Array:
        """Translate one action row into this robot's ctrl vector; a step() helper."""
        action = action.copy()

        # Parse action based on length for backward compatibility
        if len(action) == 11:
            # Legacy position-only mode
            target_velocities = None
            gripper_action = action[10] * 255.0
            position_action = action[:10]
        elif len(action) == 18:
            # Position+velocity mode for dynamic manipulation (e.g., tossing)
            target_velocities = action[11:18]
            gripper_action = action[10] * 255.0
            position_action = action[:10]
        else:
            raise ValueError(
                f"Action must have 11 (position-only) or 18 (position+velocity) "
                f"elements, got {len(action)}"
            )

        # Ctrl values > 127 apply closing force, < 127 apply opening force;
        # hence, 0 = fully open, 255 = fully closed, 127 = no force applied.

        # Compute target positions (absolute)
        if self.act_delta:
            # Interpret action as delta, compute absolute targets
            curr_qpos = np.concatenate([self.qpos["base"], self.qpos["arm"]], -1)
            target_positions = curr_qpos + position_action
        else:
            target_positions = position_action

        # Split into base and arm targets
        base_targets = target_positions[:3]
        arm_targets = target_positions[3:10]

        # Compute arm torques using PD controller (with optional velocity tracking)
        arm_torques = self._compute_arm_torques(arm_targets, target_velocities)

        # Build the control array:
        # - Base: position targets (position actuators)
        # - Arm: torques (motor actuators)
        # - Gripper: force command
        ctrl_action = np.concatenate([base_targets, arm_torques, [gripper_action]])

        return ctrl_action

    def reward(self, obs: MjObs) -> float:
        """Compute the reward from an observation.

        This is a placeholder implementation since TidyBotRobotEnv is used as a
        component in TidyBot3DEnv which handles rewards separately.
        """
        return 0.0
