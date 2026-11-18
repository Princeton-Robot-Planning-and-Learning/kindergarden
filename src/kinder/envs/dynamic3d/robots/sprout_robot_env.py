"""This module defines the SproutRobotEnv class, which is the base class for the Fauna
Sprout humanoid robot in simulation.

The Sprout model is not public. The model files are read from ``models/fauna_sprout``,
which git ignores (see the README in that folder). The robot parameters that the model
does not have (motors, gains, default pose) are in ``sprout_config.json``.
"""

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray
from relational_structs import Array

from kinder.core import RobotActionSpace
from kinder.envs.dynamic3d.mujoco_utils import MjObs
from kinder.envs.dynamic3d.robots.base import RobotEnv

SPROUT_MODEL_DIR = Path(__file__).parents[1] / "models" / "fauna_sprout"
SPROUT_XML_NAME = "sprout.xml"
SPROUT_CONFIG_PATH = Path(__file__).parent / "sprout_config.json"

# The waist (1), the legs (10), the neck (2), the arms (12), and the grippers (2).
NUM_SPROUT_JOINTS = 27

# Physics settings of the RL tasks that train the Sprout policies. The environment
# uses the same settings, so that a trained policy behaves the same here.
SPROUT_SIMULATION_TIMESTEP = 0.002  # (in seconds)
SPROUT_CONTROL_FREQUENCY = 50  # (in Hz)
SPROUT_SOLVER_ITERATIONS = 10
# Control steps in reset() that let the objects settle (1 s, as for the other robots).
SPROUT_SETTLE_STEPS = 50
SPROUT_SOLVER_LS_ITERATIONS = 20

# Prefix for the default classes of the Sprout XML ("visual", "collision", ...), so
# that they cannot collide with the default classes of scene objects.
_DEFAULT_CLASS_PREFIX = "sprout_"

# Geom group of the Sprout collision shapes. The renderers do not show this group.
_COLLISION_GEOM_GROUP = "3"


def sprout_model_available() -> bool:
    """Whether the (not distributed) Sprout model files are in the models folder."""
    return (SPROUT_MODEL_DIR / SPROUT_XML_NAME).exists()


def dc_motor_torque(
    target_positions: NDArray[np.floating],
    positions: NDArray[np.floating],
    velocities: NDArray[np.floating],
    stiffness: NDArray[np.floating],
    damping: NDArray[np.floating],
    effort_limit: NDArray[np.floating],
    saturation_effort: NDArray[np.floating],
    velocity_limit: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute joint torques with the DC motor model of the Sprout RL tasks.

    A PD law on the joint position gives the torque (the velocity target is zero):
        torque = stiffness * (target - position) - damping * velocity

    The torque is then clipped to the torque-speed curve of the motor: the motor
    gives ``saturation_effort`` at zero velocity and no torque at
    ``velocity_limit``, and never more than ``effort_limit``. This is the model of
    mjlab's ``DcMotorActuator``, which the RL tasks use.
    """
    torque = stiffness * (target_positions - positions) - damping * velocities
    velocity_at_effort_limit = velocity_limit * (1 + effort_limit / saturation_effort)
    clipped_velocity = np.clip(
        velocities, -velocity_at_effort_limit, velocity_at_effort_limit
    )
    max_effort = np.minimum(
        saturation_effort * (1.0 - clipped_velocity / velocity_limit), effort_limit
    )
    min_effort = np.maximum(
        saturation_effort * (-1.0 - clipped_velocity / velocity_limit), -effort_limit
    )
    return np.clip(torque, min_effort, max_effort)


@dataclass(frozen=True)
class SproutModel:
    """The Sprout model: the edited robot XML and the per-joint parameters.

    All arrays have one entry per joint, in the order of ``joint_names``. This is the
    order of the actions.
    """

    xml_string: str
    joint_names: tuple[str, ...]
    standing_height: float  # torso height in the default pose (in meters)
    default_joint_positions: NDArray[np.float64]
    stiffness: NDArray[np.float64]
    damping: NDArray[np.float64]
    effort_limit: NDArray[np.float64]
    saturation_effort: NDArray[np.float64]
    velocity_limit: NDArray[np.float64]
    # Default command delay: (minimum, maximum) in physics steps, and the number of
    # physics steps between new random delays.
    command_delay_steps: tuple[int, int]
    command_delay_update_period: int


def _add_jaw_pads(root: ET.Element, jaw_pads: list[dict[str, Any]]) -> None:
    """Replace the gripper collision hulls with box pads that leave the jaw gap open.

    MuJoCo collides with the convex hull of a mesh, and the hulls of the gripper meshes
    fill the space between the jaws. The pads are for the left side; the right side is
    the left side turned 180 degrees about z.
    """
    bodies = {body.get("name"): body for body in root.iter("body")}
    geoms = {geom.get("name"): geom for geom in root.iter("geom")}
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for side, sign in (("left", 1.0), ("right", -1.0)):
        for link in ("gripper_lower_link", "gripper_upper_link"):
            hull = geoms[f"{side}_{link}_collision_0"]
            hull.set("contype", "0")
            hull.set("conaffinity", "0")
        # The box corners of the pads would touch the wrist and the forearm.
        for body1, body2 in (
            ("wrist_link", "gripper_upper_link"),
            ("radius_link", "gripper_lower_link"),
        ):
            ET.SubElement(
                contact, "exclude", body1=f"{side}_{body1}", body2=f"{side}_{body2}"
            )
        for pad in jaw_pads:
            half, center = pad["half_size"], pad["center"]
            turn = sign * pad["angle"]
            if pad["name"] == "knuckle":
                # Round, and on the hinge axis: a cylinder along x.
                geom_type = "cylinder"
                size = f"{max(half[1], half[2])} {half[0]}"
                quat = f"{np.sqrt(0.5)} 0 {np.sqrt(0.5)} 0"
            else:
                geom_type = "box"
                size = f"{half[0]} {half[1]} {half[2]}"
                quat = f"{np.cos(turn / 2)} {np.sin(turn / 2)} 0 0"
            ET.SubElement(
                bodies[f"{side}_{pad['link']}"],
                "geom",
                name=f"{side}_{pad['name']}_pad",
                type=geom_type,
                size=size,
                pos=f"{sign * center[0]} {sign * center[1]} {center[2]}",
                quat=quat,
                contype="1",
                conaffinity="1",
                condim="4",
                friction="1 0.01 0.0001",
                group=_COLLISION_GEOM_GROUP,
                rgba="1 0.28 0.1 0.5",
            )


@lru_cache(maxsize=1)
def load_sprout_model() -> SproutModel:
    """Load the Sprout XML and edit it in memory for the DC motor model.

    The vendor XML is not changed on disk. The edits are:
    - The position actuators become torque motors with the motor effort limits.
    - The joints get the armature of their motor.
    - The gripper tip sites and (if configured) the jaw collision pads are added.
    - The collision shapes move to a geom group that the renderers do not show.
    - The default classes get a prefix.
    """
    if not sprout_model_available():
        raise FileNotFoundError(
            f"The Sprout model is not in {SPROUT_MODEL_DIR}. The model is not "
            f"distributed; see the README in that folder."
        )
    with open(SPROUT_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    root = ET.parse(SPROUT_MODEL_DIR / SPROUT_XML_NAME).getroot()

    # The actuator order of the vendor XML defines the joint (and action) order.
    actuator_section = root.find("actuator")
    assert actuator_section is not None, "The Sprout XML has no actuators"
    joint_names = tuple(str(a.get("joint")) for a in actuator_section)
    assert len(joint_names) == NUM_SPROUT_JOINTS, (
        f"Expected {NUM_SPROUT_JOINTS} actuated joints in the Sprout XML, "
        f"got {len(joint_names)}"
    )

    # Find the motor and the PD stiffness of each joint.
    motor_of_joint: dict[str, dict[str, float]] = {}
    stiffness_of_joint: dict[str, float] = {}
    for group in config["actuator_groups"]:
        for joint_name in joint_names:
            if any(re.fullmatch(pattern, joint_name) for pattern in group["joints"]):
                assert (
                    joint_name not in motor_of_joint
                ), f"Joint {joint_name} matches more than one actuator group"
                motor_of_joint[joint_name] = config["motors"][group["motor"]]
                stiffness_of_joint[joint_name] = group["stiffness"]
    missing = set(joint_names) - set(motor_of_joint)
    assert not missing, f"Joints without an actuator group: {sorted(missing)}"

    # Replace the position actuators with torque motors.
    for actuator in list(actuator_section):
        actuator_section.remove(actuator)
    for joint_name in joint_names:
        limit = motor_of_joint[joint_name]["effort_limit"]
        ET.SubElement(
            actuator_section,
            "motor",
            name=joint_name,
            joint=joint_name,
            gear="1",
            ctrllimited="true",
            ctrlrange=f"{-limit} {limit}",
            forcelimited="true",
            forcerange=f"{-limit} {limit}",
        )
    for joint in root.iter("joint"):
        motor = motor_of_joint.get(joint.get("name", ""))
        if motor is not None:
            joint.set("armature", str(motor["armature"]))

    bodies = {body.get("name"): body for body in root.iter("body")}
    for site_name, site in config.get("gripper_sites", {}).items():
        ET.SubElement(
            bodies[site["body"]],
            "site",
            name=site_name,
            pos=" ".join(str(v) for v in site["pos"]),
        )
    if config.get("jaw_pads"):
        _add_jaw_pads(root, config["jaw_pads"])

    # Renders show the visual meshes only (MuJoCo hides geom group 3 by default).
    for default in root.iter("default"):
        if default.get("class") == "collision":
            for geom in default.findall("geom"):
                geom.set("group", _COLLISION_GEOM_GROUP)

    for element in root.iter():
        for key in ("class", "childclass"):
            if key in element.attrib:
                element.set(key, _DEFAULT_CLASS_PREFIX + element.attrib[key])

    def per_joint(key: str) -> NDArray[np.float64]:
        return np.array([motor_of_joint[name][key] for name in joint_names])

    default_pose = config["default_pose"]
    return SproutModel(
        xml_string=ET.tostring(root, encoding="unicode"),
        joint_names=joint_names,
        standing_height=float(config["standing_height"]),
        default_joint_positions=np.array(
            [default_pose.get(name, 0.0) for name in joint_names]
        ),
        stiffness=np.array([stiffness_of_joint[name] for name in joint_names]),
        damping=np.full(len(joint_names), float(config["damping"])),
        effort_limit=per_joint("effort_limit"),
        saturation_effort=per_joint("saturation_effort"),
        velocity_limit=per_joint("velocity_limit"),
        command_delay_steps=(
            int(config["command_delay"]["min_steps"]),
            int(config["command_delay"]["max_steps"]),
        ),
        command_delay_update_period=int(config["command_delay"]["update_period"]),
    )


class SproutRobotActionSpace(RobotActionSpace):
    """An action for the Sprout robot: one position target for each joint."""

    def __init__(self) -> None:
        # All joint ranges of the robot are inside [-pi, pi].
        low = np.full(NUM_SPROUT_JOINTS, -np.pi)
        high = np.full(NUM_SPROUT_JOINTS, np.pi)
        super().__init__(low, high)

    def create_markdown_description(self) -> str:
        """Create a human-readable markdown description of this space."""
        return (
            f"Actions: joint position targets ({NUM_SPROUT_JOINTS}, rad) for the "
            "waist (1), the legs (10), the neck (2), the arms (12), and the "
            "grippers (2), in the joint order of the robot model."
        )


class SproutRobotEnv(RobotEnv):
    """This is the base class for Sprout environments that use MuJoCo for sim.

    The Sprout is a humanoid with a floating base. The action is one position target
    for each joint, at 50 Hz. The DC motor model of the Sprout RL tasks converts the
    targets to joint torques at each physics step (see ``dc_motor_torque``).

    The targets reach the motors after a random command delay, as in the RL tasks:
    each ``command_delay_update_period`` physics steps, a new delay is drawn from
    ``command_delay_steps`` (minimum and maximum, in physics steps). The defaults are
    in ``sprout_config.json``; ``command_delay_steps=(0, 0)`` gives no delay.

    The robot has no balance controller: a policy must give the targets that keep it
    upright. The timestep and the solver settings are those of the RL tasks, not
    those of the other dynamic3d robots.
    """

    def __init__(
        self,
        name: str,
        control_frequency: float = SPROUT_CONTROL_FREQUENCY,
        act_delta: bool = False,
        horizon: int = 1000,
        camera_names: list[str] | None = None,
        camera_width: int = 640,
        camera_height: int = 480,
        seed: int | None = None,
        show_viewer: bool = False,
        command_delay_steps: tuple[int, int] | None = None,
        command_delay_update_period: int | None = None,
    ) -> None:
        """
        Args:
            name: Name of the robot.
            control_frequency: Frequency at which control actions are applied (in Hz).
            act_delta: Whether to interpret actions as deltas from the current
                joint positions (True) or as absolute joint position targets.
            horizon: Maximum number of steps per episode.
            camera_names: List of camera names for rendering.
            camera_width: Width of camera images.
            camera_height: Height of camera images.
            seed: Random seed for reproducibility.
            show_viewer: Whether to show the MuJoCo viewer.
            command_delay_steps: Minimum and maximum command delay, in physics
                steps. Default: the values of the RL tasks.
            command_delay_update_period: Physics steps between new random delays
                (0: a new delay at each step). Default: the value of the RL tasks.
        """
        robot_camera_names = [f"{name}_front_camera", f"{name}_side_camera"]
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
        self.model = load_sprout_model()

        # Number of physics steps in one control step.
        self.decimation = int(
            round(1.0 / (control_frequency * SPROUT_SIMULATION_TIMESTEP))
        )
        assert np.isclose(
            self.decimation * SPROUT_SIMULATION_TIMESTEP, 1.0 / control_frequency
        ), (
            f"The control period (1 / {control_frequency} s) must be a multiple "
            f"of the simulation timestep ({SPROUT_SIMULATION_TIMESTEP} s)."
        )

        # Command delay settings, and the state of the delay (see _delay_targets).
        self.command_delay_steps: tuple[int, int] = (
            self.model.command_delay_steps
            if command_delay_steps is None
            else (int(command_delay_steps[0]), int(command_delay_steps[1]))
        )
        self.command_delay_update_period: int = (
            self.model.command_delay_update_period
            if command_delay_update_period is None
            else command_delay_update_period
        )
        self._delay_buffer = np.zeros((1, NUM_SPROUT_JOINTS))  # past targets
        self._delay_lag = 0  # current delay (in physics steps)
        self._delay_count = 0  # targets written to the buffer
        self._delay_step = 0  # physics steps since the last delay reset
        self._delay_phase = 0  # offset of the delay updates

        # Addresses of the floating base and the joints (set in
        # _setup_robot_references).
        self.base_qpos_adr: int = 0
        self.base_qvel_adr: int = 0
        self.joint_qpos_adr: int = 0
        self.joint_qvel_adr: int = 0
        self.joint_ctrl_adr: int = 0

    def _create_sim(self, xml_string: str, *, defer_dynamics: bool = False) -> None:
        """Create the simulation with the physics settings of the Sprout RL tasks."""
        super()._create_sim(xml_string, defer_dynamics=defer_dynamics)
        assert self.sim is not None
        # pylint: disable=no-member
        opt = self.sim.model.mj_model.opt
        opt.timestep = SPROUT_SIMULATION_TIMESTEP
        opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        opt.impratio = 1.0
        opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        opt.iterations = SPROUT_SOLVER_ITERATIONS
        opt.ls_iterations = SPROUT_SOLVER_LS_ITERATIONS
        opt.tolerance = 1e-8
        opt.ls_tolerance = 0.01
        opt.ccd_iterations = 50

    def _setup_robot_references(self) -> None:
        """Setup references to robot state/actuator buffers in the simulation data."""
        assert self.sim is not None, "Simulation must be initialized."

        joint_names = [f"{self.name}_{n}" for n in self.model.joint_names]
        base_joint_name = f"{self.name}_floating_base"

        self.base_qpos_adr = self.sim.model.get_joint_qpos_addr(base_joint_name)
        self.base_qvel_adr = self.sim.model.get_joint_qvel_addr(base_joint_name)
        joint_qpos_indices = [
            self.sim.model.get_joint_qpos_addr(name) for name in joint_names
        ]
        joint_qvel_indices = [
            self.sim.model.get_joint_qvel_addr(name) for name in joint_names
        ]
        joint_ctrl_indices = [
            self.sim.model._actuator_name2id[name]  # pylint: disable=protected-access
            for name in joint_names
        ]

        # Verify indices are contiguous for slicing
        for label, indices in (
            ("qpos", joint_qpos_indices),
            ("qvel", joint_qvel_indices),
            ("ctrl", joint_ctrl_indices),
        ):
            assert indices == list(
                range(indices[0], indices[0] + NUM_SPROUT_JOINTS)
            ), f"Joint {label} indices not contiguous"
        self.joint_qpos_adr = joint_qpos_indices[0]
        self.joint_qvel_adr = joint_qvel_indices[0]
        self.joint_ctrl_adr = joint_ctrl_indices[0]

        mj_data = self.sim.data.mj_data
        # Base: position (3) and quaternion (4); linear (3) and angular (3) velocity.
        self.qpos["base"] = mj_data.qpos[self.base_qpos_adr : self.base_qpos_adr + 7]
        self.qvel["base"] = mj_data.qvel[self.base_qvel_adr : self.base_qvel_adr + 6]
        self.qpos["joints"] = mj_data.qpos[
            self.joint_qpos_adr : self.joint_qpos_adr + NUM_SPROUT_JOINTS
        ]
        self.qvel["joints"] = mj_data.qvel[
            self.joint_qvel_adr : self.joint_qvel_adr + NUM_SPROUT_JOINTS
        ]
        self.ctrl["joints"] = mj_data.ctrl[
            self.joint_ctrl_adr : self.joint_ctrl_adr + NUM_SPROUT_JOINTS
        ]

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
            str(SPROUT_MODEL_DIR),
            SPROUT_XML_NAME,
            str(SPROUT_MODEL_DIR),
            robot_xml_string=self.model.xml_string,
        )
        super().reset(seed=seed, options={**options, "xml": xml_string})

        # Setup references to robot state/actuator buffers
        self._setup_robot_references()

        # Put the robot in the default standing pose at the origin
        self.set_robot_base_pos_yaw(0.0, 0.0, 0.0)
        self.reset_command_delay()

        return self.get_obs(), {}

    def set_robot_base_pos_yaw(self, x: float, y: float, yaw: float) -> None:
        """Stand the robot at the given position and yaw, in the default pose."""
        assert (
            self.sim is not None
        ), "Simulation must be initialized before setting base pose."

        self.qpos["base"][:] = [
            x,
            y,
            self.model.standing_height,
            np.cos(yaw / 2),
            0.0,
            0.0,
            np.sin(yaw / 2),
        ]
        self.qvel["base"][:] = 0.0
        self.qpos["joints"][:] = self.model.default_joint_positions
        self.qvel["joints"][:] = 0.0
        self.ctrl["joints"][:] = 0.0
        self.sim.forward()  # Update the simulation state

    def hold_standing_pose(self, base_qpos: NDArray[np.float64]) -> None:
        """Put the robot back into the default pose at the given base pose, at rest.

        Used while the scene settles in reset(): the robot has no balance controller, so
        it cannot stand still by itself.
        """
        self.qpos["base"][:] = base_qpos
        self.qvel["base"][:] = 0.0
        self.qpos["joints"][:] = self.model.default_joint_positions
        self.qvel["joints"][:] = 0.0

    def reset_command_delay(self) -> None:
        """Start a new episode of the command delay: no delay until the first update.

        This is the reset of mjlab's delay buffer: the delay is zero until the first
        update, which comes at a random time within one update period.
        """
        min_lag, max_lag = self.command_delay_steps
        assert 0 <= min_lag <= max_lag, f"Invalid command delay {(min_lag, max_lag)}"
        self._delay_buffer = np.zeros((max_lag + 1, NUM_SPROUT_JOINTS))
        self._delay_lag = 0
        self._delay_count = 0
        self._delay_step = 0
        self._delay_phase = 0
        if self.command_delay_update_period > 0:
            self._delay_phase = int(
                self.np_random.integers(self.command_delay_update_period)
            )

    def clear_command_history(self) -> None:
        """Forget the past targets but keep the current delay.

        Used when the simulation state is set directly: the past targets are not a
        part of the state.
        """
        self._delay_count = 0

    def _delay_targets(self, targets: Array) -> Array:
        """Store the targets of this physics step and return the delayed targets."""
        min_lag, max_lag = self.command_delay_steps
        if max_lag == 0:
            return targets
        size = len(self._delay_buffer)
        self._delay_buffer[self._delay_count % size] = targets
        period = self.command_delay_update_period
        if period == 0 or (self._delay_step + self._delay_phase) % period == 0:
            self._delay_lag = int(self.np_random.integers(min_lag, max_lag + 1))
        # Early in an episode, the oldest target is used.
        lag = min(self._delay_lag, self._delay_count)
        delayed = self._delay_buffer[(self._delay_count - lag) % size]
        self._delay_count += 1
        self._delay_step += 1
        return delayed.astype(np.float32)

    def _update_ctrl(self, action: Array) -> None:
        """Write the motor torques for the given joint position targets to ctrl."""
        self.ctrl["joints"][:] = dc_motor_torque(
            action,
            self.qpos["joints"],
            self.qvel["joints"],
            self.model.stiffness,
            self.model.damping,
            self.model.effort_limit,
            self.model.saturation_effort,
            self.model.velocity_limit,
        )

    def step(self, action: Array) -> tuple[MjObs, float, bool, bool, dict[str, Any]]:
        """Take a step in the environment.

        The motor torques are computed again at each physics step, from the delayed
        targets and the current joint state.

        Args:
            action: Joint position targets (rad) with shape (27,), or deltas from
                the current joint positions if ``act_delta`` is set.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        assert self.sim is not None, "Simulation must be initialized before stepping."
        assert self.timestep is not None, "Timestep must be initialized."
        assert action.shape == (NUM_SPROUT_JOINTS,), (
            f"Action must have {NUM_SPROUT_JOINTS} elements, got shape "
            f"{action.shape}"
        )
        self.timestep += 1

        targets: Array = np.asarray(action, dtype=np.float32)
        if self.act_delta:
            targets = (self.qpos["joints"] + targets).astype(np.float32)

        for _ in range(self.decimation):
            self._update_ctrl(self._delay_targets(targets))
            self.sim.step()

        obs = self.get_obs()
        reward = self.reward(obs)
        terminated = self.timestep >= self.horizon
        truncated = False
        info: dict[str, object] = {}

        return obs, reward, terminated, truncated, info

    def reward(self, obs: MjObs) -> float:
        """Compute the reward from an observation.

        This is a placeholder implementation since SproutRobotEnv is used as a component
        in the object-centric environment, which handles rewards.
        """
        return 0.0
