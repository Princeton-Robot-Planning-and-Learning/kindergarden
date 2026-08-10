"""PyBullet models of the human limbs that the robot repositions, and the joint limit
and muscle tone models that describe how they resist being moved.

These differ from the human models bundled with pybullet_helpers: some joint axes are
flipped, the joints are continuous, and the grasp frame is placed differently.
"""

import abc
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import JointPositions, JointVelocities
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot

from kinder.envs.dynamic3d.limb_utils import HUMAN_ASSETS_DIR, JointTorques


class BaseJointLimitsModel(abc.ABC):
    """Base model for joint limits."""

    @abc.abstractmethod
    def check_joint_limits(self, joint_positions: JointPositions) -> bool:
        """Check if the given joint positions are within the limits."""


class NoJointLimitsModel(BaseJointLimitsModel):
    """A model where every joint position is allowed."""

    def check_joint_limits(self, joint_positions: JointPositions) -> bool:
        del joint_positions
        return True


class BoxJointLimitsModel(BaseJointLimitsModel):
    """A model where each joint is independently bounded.

    A no-op on the limbs, whose URDF joints are continuous, so the bounds are +/-inf.
    """

    def __init__(
        self, lower_limits: JointPositions, upper_limits: JointPositions
    ) -> None:
        self._lower_limits = lower_limits
        self._upper_limits = upper_limits

    def check_joint_limits(self, joint_positions: JointPositions) -> bool:
        return all(
            lower <= v <= upper
            for lower, v, upper in zip(
                self._lower_limits,
                joint_positions,
                self._upper_limits,
                strict=True,
            )
        )


def create_joint_limits_model(
    name: str, lower_limits: JointPositions, upper_limits: JointPositions
) -> BaseJointLimitsModel:
    """Create a joint limits model from its name."""
    if name == "none":
        return NoJointLimitsModel()
    if name == "box":
        return BoxJointLimitsModel(lower_limits, upper_limits)
    raise ValueError(f"Unrecognized joint limits model: {name}")


# The limb URDFs use continuous joints, so anatomical bounds are given here instead.
_BALL_RANGE = (-np.pi / 2, np.pi / 2)
_DISTAL_RANGE = (-np.pi / 4, np.pi / 4)
_ELBOW_FLEXION = 2.53  # 145 degrees
_KNEE_FLEXION = 2.36  # 135 degrees

_HINGE_RANGES: dict[str, tuple[float, float]] = {
    "human-left-arm": (0.0, _ELBOW_FLEXION),
    "human-right-arm": (-_ELBOW_FLEXION, 0.0),
    "human-left-leg": (0.0, _KNEE_FLEXION),
    "human-right-leg": (0.0, _KNEE_FLEXION),
}


def get_sampling_bounds(
    limb_name: str, nominal: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Joint ranges widened to contain a nominal configuration that sits outside
    them."""
    lower, upper = np.array(get_human_joint_ranges(limb_name)).T
    return np.minimum(lower, nominal), np.maximum(upper, nominal)


def get_human_joint_ranges(limb_name: str) -> tuple[tuple[float, float], ...]:
    """Per-joint bounds for a human limb, in URDF joint coordinates."""
    if limb_name not in _HINGE_RANGES:
        raise ValueError(f"Unknown human limb: {limb_name}")
    return (
        _BALL_RANGE,
        _BALL_RANGE,
        _BALL_RANGE,
        _HINGE_RANGES[limb_name],
        _DISTAL_RANGE,
        _DISTAL_RANGE,
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
    """A spring-damper muscle tone model.

    Unvalidated: the tone is clamped to be one-directional and can exceed the robot's
    torque limit.
    """

    def __init__(self, num_joints: int) -> None:
        self._num_joints = num_joints
        # Placeholders that should eventually be sampled rather than hard-coded.
        self._b = -1 * np.ones(num_joints)
        self._k = np.eye(num_joints)
        self._m = 0.5 * np.ones(num_joints)
        self._c = 0.1 * np.ones(num_joints)

    def get_muscle_tone(
        self, joint_positions: JointPositions, joint_velocities: JointVelocities
    ) -> JointTorques:
        tone = np.maximum(
            np.zeros(self._num_joints),
            -(self._b - self._k @ np.abs(np.subtract(joint_positions, self._m)))
            - self._c * np.asarray(joint_velocities),
        )
        return list(tone)


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
        joint_limits_model_name: str = "none",
        muscle_tone_model_name: str = "none",
        **kwargs,
    ) -> None:
        # The base __init__ calls check_joint_limits() before the model exists.
        self._joint_limits_model_created = False
        super().__init__(*args, **kwargs)
        self._joint_limits_model = self._create_joint_limits_model(
            joint_limits_model_name
        )
        self._joint_limits_model_created = True
        self._muscle_tone_model = create_muscle_tone_model(
            muscle_tone_model_name, len(self.arm_joints)
        )

    def _create_joint_limits_model(self, name: str) -> BaseJointLimitsModel:
        return create_joint_limits_model(
            name, self.joint_lower_limits, self.joint_upper_limits
        )

    def check_joint_limits(self, joint_positions: JointPositions) -> bool:
        if not self._joint_limits_model_created:
            return super().check_joint_limits(joint_positions)
        return self._joint_limits_model.check_joint_limits(joint_positions)

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
        return [0.0, -0.1, 0.1, -1.08786023, 0.0, 0.0]


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
        return [0.0, -0.1, 0.1, -1.08786023, -0.14448669, -0.26559232]


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
        return [0.0, -0.1, 0.1, -1.08786023, -0.14448669, -0.26559232]


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
    )
