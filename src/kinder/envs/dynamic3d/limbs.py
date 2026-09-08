"""PyBullet models of the human limbs that the robot repositions, and the joint limit
and muscle tone models that describe how they resist being moved.

These differ from the human models bundled with pybullet_helpers: some joint axes are
flipped, the joints are continuous, and the grasp frame is placed differently.
"""

import abc
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pybullet as p
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import (
    JointPositions,
    JointVelocities,
    get_joint_lower_limits,
    get_joint_upper_limits,
)
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot

from kinder.envs.dynamic3d.limb_utils import HUMAN_ASSETS_DIR, JointTorques


class BaseJointLimitsModel(abc.ABC):
    """Base model for joint limits."""

    @abc.abstractmethod
    def get_joint_limits(self) -> tuple[JointPositions, JointPositions]:
        """The (lower, upper) bound on each joint."""

    def check_joint_limits(self, joint_positions: JointPositions) -> bool:
        """Check if the given joint positions are within the limits."""
        lower_limits, upper_limits = self.get_joint_limits()
        return all(
            lower <= v <= upper
            for lower, v, upper in zip(
                lower_limits, joint_positions, upper_limits, strict=True
            )
        )


class NoJointLimitsModel(BaseJointLimitsModel):
    """A model where every joint position is allowed."""

    def __init__(self, num_joints: int) -> None:
        self._num_joints = num_joints

    def get_joint_limits(self) -> tuple[JointPositions, JointPositions]:
        return ([-np.inf] * self._num_joints, [np.inf] * self._num_joints)

    def check_joint_limits(self, joint_positions: JointPositions) -> bool:
        del joint_positions
        return True


class BoxJointLimitsModel(BaseJointLimitsModel):
    """A model where each joint is independently bounded."""

    def __init__(
        self, lower_limits: JointPositions, upper_limits: JointPositions
    ) -> None:
        assert len(lower_limits) == len(upper_limits)
        self._lower_limits = list(lower_limits)
        self._upper_limits = list(upper_limits)

    def get_joint_limits(self) -> tuple[JointPositions, JointPositions]:
        return (list(self._lower_limits), list(self._upper_limits))


ARM_LIMB_NAMES = ("human-left-arm", "human-right-arm")
LEG_LIMB_NAMES = ("human-left-leg", "human-right-leg")


def create_joint_limits_model(
    name: str, lower_limits: JointPositions, upper_limits: JointPositions
) -> BaseJointLimitsModel:
    """Create a joint limits model from its name."""
    if name == "none":
        return NoJointLimitsModel(len(lower_limits))
    if name == "box":
        return BoxJointLimitsModel(lower_limits, upper_limits)
    raise ValueError(f"Unrecognized joint limits model: {name}")


@dataclass(frozen=True)
class RangeOfMotion:
    """How far one person can move each joint, as a magnitude in degrees.

    The defaults are the AAOS normal adult values, from
    https://goniometer.io/range-of-motion.
    """

    shoulder_flexion: float = 180.0
    shoulder_extension: float = 60.0
    shoulder_abduction: float = 180.0
    shoulder_adduction: float = 40.0
    shoulder_internal_rotation: float = 70.0
    shoulder_external_rotation: float = 90.0
    elbow_flexion: float = 150.0
    elbow_extension: float = 0.0
    wrist_flexion: float = 80.0
    wrist_extension: float = 70.0
    wrist_radial_deviation: float = 20.0
    wrist_ulnar_deviation: float = 30.0
    hip_flexion: float = 120.0
    hip_extension: float = 30.0
    hip_abduction: float = 45.0
    hip_adduction: float = 30.0
    hip_internal_rotation: float = 45.0
    hip_external_rotation: float = 45.0
    knee_flexion: float = 135.0
    ankle_plantarflexion: float = 50.0
    ankle_dorsiflexion: float = 20.0
    ankle_inversion: float = 35.0
    ankle_eversion: float = 15.0

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be a non-negative number of degrees, got {value}"
                )

    def as_dict(self) -> dict[str, float]:
        """Every motion and its magnitude in degrees."""
        return {
            field.name: float(getattr(self, field.name))
            for field in dataclasses.fields(self)
        }

    def replace(self, **motions: float) -> "RangeOfMotion":
        """A copy with the named motions changed, rejecting unknown names."""
        unknown = sorted(set(motions) - set(MOTION_NAMES))
        if unknown:
            raise ValueError(
                f"Unknown range of motion names: {unknown}; expected some of "
                f"{list(MOTION_NAMES)}"
            )
        return dataclasses.replace(self, **motions)

    def scaled(self, factors: dict[str, float]) -> "RangeOfMotion":
        """A copy with each named motion multiplied by its factor."""
        return self.replace(
            **{name: getattr(self, name) * factor for name, factor in factors.items()}
        )


MOTION_NAMES: tuple[str, ...] = tuple(
    field.name for field in dataclasses.fields(RangeOfMotion)
)

DEFAULT_RANGE_OF_MOTION = RangeOfMotion()


def create_range_of_motion(**motions: float) -> RangeOfMotion:
    """Defines a range of motion, taking the default for every motion not named."""
    return DEFAULT_RANGE_OF_MOTION.replace(**motions)


def sample_range_of_motion(
    rng: np.random.Generator | int | None = None,
    relative_spread: float = 0.15,
    baseline: RangeOfMotion | None = None,
    symmetric: bool = True,
) -> dict[str, RangeOfMotion]:
    """Draw a person, as the range of motion of each of their four limbs.

    - Every motion is scaled by an independent uniform factor within `relative_spread` of
    the baseline.
    - `symmetric` gives both sides the same draw.
    """
    if not 0.0 <= relative_spread < 1.0:
        raise ValueError(f"relative_spread must be in [0, 1), got {relative_spread}")
    generator = (
        rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    )
    baseline = baseline or DEFAULT_RANGE_OF_MOTION

    def draw() -> RangeOfMotion:
        factors = generator.uniform(
            1.0 - relative_spread, 1.0 + relative_spread, size=len(MOTION_NAMES)
        )
        return baseline.scaled(dict(zip(MOTION_NAMES, factors, strict=True)))

    if symmetric:
        shared = draw()
        return {limb_name: shared for limb_name in ALL_LIMB_NAMES}
    return {limb_name: draw() for limb_name in ALL_LIMB_NAMES}


# Body segment mass fractions from assistive gym
ARM_MASS_FRACTIONS = {"upper_arm": 0.0330, "lower_arm": 0.0190, "hand": 0.0065}
LEG_MASS_FRACTIONS = {"upper_leg": 0.1050, "lower_leg": 0.0475, "foot": 0.0140}
TORSO_MASS_FRACTION = 0.550


@dataclass(frozen=True)
class BodyMass:
    """How much a person weighs, in kilograms.

    The default: assistive gym's 50th-percentile male is 78.4, its female 62.5.
    """

    total: float = 54.4

    def __post_init__(self) -> None:
        if not np.isfinite(self.total) or self.total <= 0.0:
            raise ValueError(f"total must be a positive number of kg, got {self.total}")

    def link_masses(self, limb_name: str) -> dict[str, float]:
        """What each of a limb's moving links weighs, by URDF link name."""
        if limb_name in ARM_LIMB_NAMES:
            fractions = ARM_MASS_FRACTIONS
        elif limb_name in LEG_LIMB_NAMES:
            fractions = LEG_MASS_FRACTIONS
        else:
            raise ValueError(f"Unknown human limb: {limb_name}")
        return {link: fraction * self.total for link, fraction in fractions.items()}

    def limb_mass(self, limb_name: str) -> float:
        """What a whole limb weighs."""
        return sum(self.link_masses(limb_name).values())

    @property
    def torso_mass(self) -> float:
        """What the torso and head weigh, the four limbs removed."""
        return TORSO_MASS_FRACTION * self.total


DEFAULT_BODY_MASS = BodyMass()


def _joint_ranges_deg(
    limb_name: str, range_of_motion: RangeOfMotion
) -> tuple[tuple[float, float], ...]:
    """The bounds on each joint, in degrees of URDF joint coordinates."""
    rom = range_of_motion
    wrist_deviation = min(rom.wrist_radial_deviation, rom.wrist_ulnar_deviation)
    wrist = (
        (-wrist_deviation, wrist_deviation),
        (-rom.wrist_extension, rom.wrist_flexion),
    )
    ankle_sagittal = (-rom.ankle_dorsiflexion, rom.ankle_plantarflexion)
    if limb_name == "human-left-arm":
        return (
            (-rom.shoulder_extension, rom.shoulder_flexion),
            (-rom.shoulder_adduction, rom.shoulder_abduction),
            (-rom.shoulder_internal_rotation, rom.shoulder_external_rotation),
            (-rom.elbow_extension, rom.elbow_flexion),
            *wrist,
        )
    if limb_name == "human-right-arm":
        return (
            (-rom.shoulder_flexion, rom.shoulder_extension),
            (-rom.shoulder_adduction, rom.shoulder_abduction),
            (-rom.shoulder_external_rotation, rom.shoulder_internal_rotation),
            (-rom.elbow_flexion, rom.elbow_extension),
            *wrist,
        )
    if limb_name == "human-left-leg":
        return (
            (-rom.hip_flexion, rom.hip_extension),
            (-rom.hip_abduction, rom.hip_adduction),
            (-rom.hip_internal_rotation, rom.hip_external_rotation),
            (0.0, rom.knee_flexion),
            ankle_sagittal,
            (-rom.ankle_eversion, rom.ankle_inversion),
        )
    if limb_name == "human-right-leg":
        return (
            (-rom.hip_flexion, rom.hip_extension),
            (-rom.hip_adduction, rom.hip_abduction),
            (-rom.hip_external_rotation, rom.hip_internal_rotation),
            (0.0, rom.knee_flexion),
            ankle_sagittal,
            (-rom.ankle_inversion, rom.ankle_eversion),
        )
    raise ValueError(f"Unknown human limb: {limb_name}")


def get_human_joint_limits(
    limb_name: str, range_of_motion: RangeOfMotion | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-joint anatomical bounds for a human limb, in URDF joint coordinates."""
    ranges = _joint_ranges_deg(limb_name, range_of_motion or DEFAULT_RANGE_OF_MOTION)
    lower, upper = np.deg2rad(np.array(ranges, dtype=np.float64)).T
    return lower, upper


def range_of_motion_admits(
    limb_name: str,
    joint_positions: JointPositions,
    range_of_motion: RangeOfMotion | None = None,
) -> bool:
    """Whether a limb configuration is reachable within this range of motion."""
    lower, upper = get_human_joint_limits(limb_name, range_of_motion)
    return bool(
        np.all(np.asarray(joint_positions) >= lower)
        and np.all(np.asarray(joint_positions) <= upper)
    )


def describe_joint_limit_violations(
    lower_limits: JointPositions,
    upper_limits: JointPositions,
    joint_positions: JointPositions,
) -> str:
    """Name each joint outside its limits, with its value and bounds."""
    return ", ".join(
        f"joint {i} at {np.rad2deg(v):.1f} deg is outside "
        f"[{np.rad2deg(lower):.1f}, {np.rad2deg(upper):.1f}]"
        for i, (lower, v, upper) in enumerate(
            zip(lower_limits, joint_positions, upper_limits, strict=True)
        )
        if not lower <= v <= upper
    )


def validate_human_joint_positions(
    limb_name: str,
    joint_positions: JointPositions,
    label: str,
    range_of_motion: RangeOfMotion | None = None,
    joint_limits_model_name: str = "box",
) -> None:
    """Raise if a configuration is anatomically impossible for this limb."""
    lower, upper = get_human_joint_limits(limb_name, range_of_motion)
    model = create_joint_limits_model(joint_limits_model_name, list(lower), list(upper))
    if model.check_joint_limits(list(joint_positions)):
        return
    violations = describe_joint_limit_violations(
        list(lower), list(upper), list(joint_positions)
    )
    if violations:
        raise ValueError(f"{label} is outside {limb_name}'s joint limits: {violations}")
    raise ValueError(
        f"{label} is within {limb_name}'s joint limits, but is not a configuration "
        f"the {joint_limits_model_name} model considers reachable"
    )


class BaseMuscleToneModel(abc.ABC):
    """Base muscle tone model."""

    @abc.abstractmethod
    def get_muscle_tone(
        self, joint_positions: JointPositions, joint_velocities: JointVelocities
    ) -> JointTorques:
        """Get the torque that the limb currently applies to itself."""


class NoMuscleToneModel(BaseMuscleToneModel):
    """A completely limp limb."""

    def get_muscle_tone(
        self, joint_positions: JointPositions, joint_velocities: JointVelocities
    ) -> JointTorques:
        num_joints = len(joint_positions)
        assert num_joints == len(joint_velocities)
        return [0.0] * num_joints


class SpringMuscleToneModel(BaseMuscleToneModel):
    """A spring-damper muscle tone model, saturating at `max_torque`.

    tone = min(max_torque, max(0, k|q - m| - b - c*qdot))

    Unvalidated: the constants keep the tone to a few N*m over the range of motion but
    are not fitted to a person.
    """

    def __init__(self, num_joints: int, max_torque: float = 5.0) -> None:
        self._num_joints = num_joints
        # Placeholders that should eventually be sampled rather than hard-coded.
        self._b = np.zeros(num_joints)
        self._k = 3.0 * np.eye(num_joints)
        self._m = np.zeros(num_joints)
        self._c = 0.5 * np.ones(num_joints)
        self._max_torque = max_torque

    def get_muscle_tone(
        self, joint_positions: JointPositions, joint_velocities: JointVelocities
    ) -> JointTorques:
        tone = np.maximum(
            np.zeros(self._num_joints),
            -(self._b - self._k @ np.abs(np.subtract(joint_positions, self._m)))
            - self._c * np.asarray(joint_velocities),
        )
        return list(np.minimum(tone, self._max_torque))


def create_muscle_tone_model(name: str, num_joints: int) -> BaseMuscleToneModel:
    """Create a muscle tone model from its name."""
    if name == "none":
        return NoMuscleToneModel()
    if name == "spring":
        return SpringMuscleToneModel(num_joints)
    raise ValueError(f"Unrecognized muscle tone model: {name}")


class HumanLimbPyBulletRobot(SingleArmPyBulletRobot):
    """Base class for a passive human limb.

    A limb is modelled as a robot, but is never commanded directly: it moves only in
    response to the forces the robot transmits through the grasp, plus its muscle tone.
    """

    def __init__(
        self,
        *args,
        joint_limits_model_name: str = "box",
        muscle_tone_model_name: str = "none",
        range_of_motion: RangeOfMotion | None = None,
        body_mass: BodyMass | None = None,
        **kwargs,
    ) -> None:
        # The base __init__ calls check_joint_limits() before the model exists.
        self._joint_limits_model_created = False
        self._range_of_motion = range_of_motion or DEFAULT_RANGE_OF_MOTION
        self._body_mass = body_mass or DEFAULT_BODY_MASS
        super().__init__(*args, **kwargs)
        self._apply_body_mass()
        self._joint_limits_model = self._create_joint_limits_model(
            joint_limits_model_name
        )
        self._joint_limits_model_created = True
        self._muscle_tone_model = create_muscle_tone_model(
            muscle_tone_model_name, len(self.arm_joints)
        )
        positions = self.get_joint_positions()
        if not self.check_joint_limits(positions):
            lower_limits, upper_limits = self._model_joint_limits()
            violations = (
                describe_joint_limit_violations(lower_limits, upper_limits, positions)
                or "the configuration is within the per-joint bounds but unreachable"
            )
            raise ValueError(
                f"{self.get_name()} was created outside its joint limits: {violations}"
            )

    def _create_joint_limits_model(self, name: str) -> BaseJointLimitsModel:
        lower_limits, upper_limits = get_human_joint_limits(
            self.get_name(), self._range_of_motion
        )
        return create_joint_limits_model(name, list(lower_limits), list(upper_limits))

    def _apply_body_mass(self) -> None:
        """Overwrite the URDF's placeholder masses, keeping its inertia in proportion."""
        masses = self._body_mass.link_masses(self.get_name())
        client = self.physics_client_id
        for joint in range(p.getNumJoints(self.robot_id, physicsClientId=client)):
            link = p.getJointInfo(self.robot_id, joint, physicsClientId=client)[12]
            mass = masses.get(link.decode())
            if mass is None:
                continue
            old_mass, _, inertia = p.getDynamicsInfo(
                self.robot_id, joint, physicsClientId=client
            )[:3]
            p.changeDynamics(
                self.robot_id,
                joint,
                mass=mass,
                localInertiaDiagonal=[v * mass / old_mass for v in inertia],
                physicsClientId=client,
            )

    @property
    def range_of_motion(self) -> RangeOfMotion:
        """How far this person can move each joint."""
        return self._range_of_motion

    @property
    def body_mass(self) -> BodyMass:
        """How much this person weighs."""
        return self._body_mass

    @property
    def joint_lower_limits(self) -> JointPositions:
        """Lower bound on the joint limits, from the limb's joint limits model."""
        return self._model_joint_limits()[0]

    @property
    def joint_upper_limits(self) -> JointPositions:
        """Upper bound on the joint limits, from the limb's joint limits model."""
        return self._model_joint_limits()[1]

    def _model_joint_limits(self) -> tuple[JointPositions, JointPositions]:
        """The joint limits, falling back to the URDF's until the model exists."""
        if not self._joint_limits_model_created:
            return (
                get_joint_lower_limits(
                    self.robot_id, self.arm_joints, self.physics_client_id
                ),
                get_joint_upper_limits(
                    self.robot_id, self.arm_joints, self.physics_client_id
                ),
            )
        return self._joint_limits_model.get_joint_limits()

    def check_joint_limits(self, joint_positions: JointPositions) -> bool:
        if not self._joint_limits_model_created:
            return super().check_joint_limits(joint_positions)
        return self._joint_limits_model.check_joint_limits(joint_positions)

    def joint_limit_violation(self, joint_positions: JointPositions) -> float:
        """How far the worst joint is outside its limits, in radians, or 0.0 if none."""
        lower_limits, upper_limits = self._model_joint_limits()
        return max(
            (
                max(lower - v, v - upper, 0.0)
                for lower, v, upper in zip(
                    lower_limits, joint_positions, upper_limits, strict=True
                )
            ),
            default=0.0,
        )

    @property
    def muscle_tone_model(self) -> BaseMuscleToneModel:
        """The muscle tone model for this limb."""
        return self._muscle_tone_model

    def get_muscle_tone_torque(self) -> JointTorques:
        """The torque the limb currently applies to itself."""
        pos = self.get_joint_positions()
        vel = self.get_joint_velocities()
        return self._muscle_tone_model.get_muscle_tone(pos, vel)

    @property
    def end_effector_name(self) -> str:
        return "grasp_fixed_joint"

    @property
    def tool_link_name(self) -> str:
        return "ee_link"


class HumanLeftArm(HumanLimbPyBulletRobot):
    """Human left arm."""

    @classmethod
    def get_name(cls) -> str:
        return "human-left-arm"

    @property
    def default_urdf_path(self) -> Path:
        return HUMAN_ASSETS_DIR / "left_arm_6dof_continuous.urdf"

    @property
    def default_home_joint_positions(self) -> JointPositions:
        return [0.0, 0.1, -0.1, 1.08786023, 0.0, 0.0]


class HumanRightArm(HumanLimbPyBulletRobot):
    """Human right arm."""

    @classmethod
    def get_name(cls) -> str:
        return "human-right-arm"

    @property
    def default_urdf_path(self) -> Path:
        return HUMAN_ASSETS_DIR / "right_arm_6dof_continuous.urdf"

    @property
    def default_home_joint_positions(self) -> JointPositions:
        return [0.0, 0.1, 0.1, -1.08786023, 0.0, 0.0]


class HumanLeftLeg(HumanLimbPyBulletRobot):
    """Human left leg."""

    @classmethod
    def get_name(cls) -> str:
        return "human-left-leg"

    @property
    def default_urdf_path(self) -> Path:
        return HUMAN_ASSETS_DIR / "left_leg_6dof_continuous.urdf"

    @property
    def default_home_joint_positions(self) -> JointPositions:
        return [0.0, 0.1, -0.1, 1.08786023, -0.14448669, 0.26559232]


class HumanRightLeg(HumanLimbPyBulletRobot):
    """Human right leg."""

    @classmethod
    def get_name(cls) -> str:
        return "human-right-leg"

    @property
    def default_urdf_path(self) -> Path:
        return HUMAN_ASSETS_DIR / "right_leg_6dof_continuous.urdf"

    @property
    def default_home_joint_positions(self) -> JointPositions:
        return [0.0, -0.1, 0.1, 1.08786023, -0.14448669, -0.26559232]


LIMB_NAME_TO_CLS: dict[str, type[HumanLimbPyBulletRobot]] = {
    cls.get_name(): cls
    for cls in (HumanLeftArm, HumanRightArm, HumanLeftLeg, HumanRightLeg)
}
ALL_LIMB_NAMES = tuple(LIMB_NAME_TO_CLS)


def create_human_limb(
    name: str,
    base_pose: Pose,
    home_joint_positions: JointPositions | None,
    joint_limits_model_name: str,
    muscle_tone_model_name: str,
    physics_client_id: int,
    range_of_motion: RangeOfMotion | None = None,
    body_mass: BodyMass | None = None,
) -> HumanLimbPyBulletRobot:
    """Create a human limb from its name."""
    if name not in LIMB_NAME_TO_CLS:
        raise ValueError(f"Unknown human limb: {name}")
    return LIMB_NAME_TO_CLS[name](
        physics_client_id=physics_client_id,
        base_pose=base_pose,
        home_joint_positions=home_joint_positions,
        joint_limits_model_name=joint_limits_model_name,
        muscle_tone_model_name=muscle_tone_model_name,
        range_of_motion=range_of_motion,
        body_mass=body_mass,
    )
