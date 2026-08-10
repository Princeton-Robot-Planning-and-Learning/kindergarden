"""Environment where the goal is to reposition a human limb.

The robot's end effector is rigidly attached to the limb's grasp frame, and must drive
the limb to a goal joint configuration using only joint torques. There are sixteen
variants, one per (scene, limb) pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose, SE2Pose, get_pose
from pybullet_helpers.joint import JointPositions
from relational_structs import Array, Object, ObjectCentricState
from relational_structs.utils import create_state_from_dict

from kinder.core import ConstantObjectKinDEREnv, FinalConfigMeta
from kinder.envs.kinematic3d.base_limbrepositioning3d import (
    Limb3DEnvConfig,
    ObjectCentricLimb3DRobotEnv,
)
from kinder.envs.kinematic3d.limb_scenes import (
    ALL_SCENE_TYPES,
    LimbRepositioningSceneConfig,
)
from kinder.envs.kinematic3d.limbs import ALL_LIMB_NAMES, get_sampling_bounds
from kinder.envs.kinematic3d.object_types import (
    Limb3DEnvTypeFeatures,
    Limb3DFixtureType,
    Limb3DLimbType,
    Limb3DRobotType,
)
from kinder.envs.kinematic3d.utils import (
    NUM_LIMB_JOINTS,
    NUM_ROBOT_JOINTS,
    LimbRepositioning3DObjectCentricState,
    get_torque_action_from_gui_input,
    joint_position_distance,
)

# Draws before a reset gives up and falls back to the nominal configuration.
_MAX_INIT_SAMPLE_ATTEMPTS = 20


@dataclass(frozen=True)
class LimbRepositioning3DEnvConfig(Limb3DEnvConfig, metaclass=FinalConfigMeta):
    """Config for LimbRepositioning3DEnv()."""

    # Radians from the goal joint configuration at which the goal counts as reached.
    goal_atol: float = 0.1

    # Probability that the limb spasms on a step, adding random torque to the action.
    spasm_probability: float = 0.0
    spasm_scale: float = 10000.0

    # Radians either side of the nominal configuration that a reset samples within.
    init_joint_noise: float = 0.26

    # Penetration a sampled initial configuration may add beyond the model's own overlap.
    init_max_penetration: float = 0.002


class ObjectCentricLimbRepositioning3DEnv(
    ObjectCentricLimb3DRobotEnv[
        LimbRepositioning3DObjectCentricState, LimbRepositioning3DEnvConfig
    ]
):
    """Object-centric version of LimbRepositioning3DEnv()."""

    def __init__(
        self,
        variant: str = "wheelchair-left-arm",
        config: LimbRepositioning3DEnvConfig | None = None,
        **kwargs,
    ) -> None:
        if config is None:
            config = create_variant_config(variant)
        self.variant = variant
        super().__init__(config=config, **kwargs)
        self._spasm_rng = np.random.default_rng(0)
        self._init_rng = np.random.default_rng(0)

    @property
    def _has_fixture(self) -> bool:
        return self.scene.furniture_id is not None

    @property
    def _object_names(self) -> list[tuple[str, Any]]:
        objects: list[tuple[str, Any]] = [
            ("robot", Limb3DRobotType),
            ("limb", Limb3DLimbType),
        ]
        if self._has_fixture:
            objects.append(("fixture", Limb3DFixtureType))
        return objects

    def _get_obs(self) -> LimbRepositioning3DObjectCentricState:
        state_dict: dict[Object, dict[str, float]] = {}
        for object_name, object_type in self._object_names:
            obj = Object(object_name, object_type)
            state_dict[obj] = self._get_object_features(object_name)
        state = create_state_from_dict(
            state_dict,
            Limb3DEnvTypeFeatures,
            state_cls=LimbRepositioning3DObjectCentricState,
        )
        assert isinstance(state, LimbRepositioning3DObjectCentricState)
        return state

    def _get_object_features(self, object_name: str) -> dict[str, float]:
        feats: dict[str, float] = {}
        if object_name == "robot":
            base_pose = self.robot.get_base()
            feats["pos_base_x"] = base_pose.x
            feats["pos_base_y"] = base_pose.y
            feats["pos_base_rot"] = base_pose.rot
            for i, v in enumerate(self._get_robot_joint_positions()):
                feats[f"joint_{i + 1}"] = v
            for i, v in enumerate(self._get_robot_joint_velocities()):
                feats[f"joint_vel_{i + 1}"] = v
        elif object_name == "limb":
            for i, v in enumerate(self.limb.get_joint_positions()):
                feats[f"joint_{i + 1}"] = v
            for i, v in enumerate(self.limb.get_joint_velocities()):
                feats[f"joint_vel_{i + 1}"] = v
            goal = self.config.scene.limb_goal_joint_positions
            for i, v in enumerate(goal):
                feats[f"goal_joint_{i + 1}"] = v
        elif object_name == "fixture":
            furniture_id = self.scene.furniture_id
            assert furniture_id is not None
            pose = get_pose(furniture_id, self.physics_client_id)
            values = list(pose.position) + list(pose.orientation)
            names = Limb3DEnvTypeFeatures[Limb3DFixtureType]
            feats = dict(zip(names, values, strict=True))
        else:
            raise ValueError(f"Unknown object: {object_name}")
        return feats

    def _create_constant_initial_state(self) -> LimbRepositioning3DObjectCentricState:
        # Every object in this environment already appears in the observation.
        state = create_state_from_dict(
            {},
            Limb3DEnvTypeFeatures,
            state_cls=LimbRepositioning3DObjectCentricState,
        )
        assert isinstance(state, LimbRepositioning3DObjectCentricState)
        return state

    def _sample_limb_init_joint_positions(self) -> JointPositions:
        """Perturb the nominal configuration without driving the limb into the scene."""
        nominal = np.array(self.config.scene.limb_init_joint_positions)
        noise = self.config.init_joint_noise
        lower, upper = get_sampling_bounds(self.config.scene.limb_name, nominal)
        low = np.maximum(nominal - noise, lower)
        high = np.minimum(nominal + noise, upper)
        for _ in range(_MAX_INIT_SAMPLE_ATTEMPTS):
            candidate = list(self._init_rng.uniform(low, high))
            self.limb.set_joints(candidate)
            if self.limb_penetration() > -self.config.init_max_penetration:
                return candidate
        # Fall back to the nominal configuration, which is collision free by definition.
        return list(nominal)

    def _reset_bodies(self) -> None:
        self.robot.arm.set_joints(
            self._extend_joints_with_fingers(list(self._initial_robot_joints)),
            joint_velocities=[0.0] * len(self.robot.arm.arm_joints),
        )
        self.limb.set_joints(
            self._sample_limb_init_joint_positions(),
            joint_velocities=[0.0] * NUM_LIMB_JOINTS,
        )
        self._regrasp_limb()

    def _set_state(self, state: LimbRepositioning3DObjectCentricState) -> None:
        self.robot.arm.set_joints(
            self._extend_joints_with_fingers(state.robot_joint_positions),
            joint_velocities=self._extend_joints_with_fingers(
                state.robot_joint_velocities
            ),
        )
        self.limb.set_joints(
            state.limb_joint_positions,
            joint_velocities=state.limb_joint_velocities,
        )
        self._reweld_limb()

    def _execute_spasm(self) -> None:
        """Apply an impulse of random torque to the passive limb."""
        cov = self.config.spasm_scale * np.eye(NUM_LIMB_JOINTS)
        spasm_torque = self._spasm_rng.multivariate_normal(
            np.zeros(NUM_LIMB_JOINTS), cov
        )
        self._apply_torques([0.0] * NUM_ROBOT_JOINTS, list(spasm_torque))

    def step(
        self, action: Array
    ) -> tuple[LimbRepositioning3DObjectCentricState, float, bool, bool, dict]:
        action_space = self.torque_action_space
        torque = np.clip(action, action_space.low, action_space.high)
        limb_torque = self.limb.get_muscle_tone_torque()
        self._apply_torques(list(torque), limb_torque)

        if self._spasm_rng.uniform() < self.config.spasm_probability:
            self._execute_spasm()

        obs = self._get_obs()
        reward = -1.0
        terminated = self.goal_reached()
        info = {"limb_distance_to_goal": obs.limb_distance_to_goal}
        return obs, reward, terminated, False, info

    def goal_reached(self) -> bool:
        distance = joint_position_distance(
            self.limb.get_joint_positions(),
            list(self.config.scene.limb_goal_joint_positions),
        )
        return distance < self.config.goal_atol

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[LimbRepositioning3DObjectCentricState, dict]:
        if seed is not None:
            self._spasm_rng = np.random.default_rng(seed)
            self._init_rng = np.random.default_rng(seed)
        return super().reset(seed=seed, options=options)

    def get_action_from_gui_input(
        self, gui_input: dict[str, Any]
    ) -> NDArray[np.float32]:
        return get_torque_action_from_gui_input(self.torque_action_space, gui_input)


class LimbRepositioning3DEnv(ConstantObjectKinDEREnv):
    """Limb repositioning environment with a constant number of objects."""

    def __init__(self, *args, variant: str = "wheelchair-left-arm", **kwargs) -> None:
        self.variant = variant
        super().__init__(*args, variant=variant, **kwargs)

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricLimbRepositioning3DEnv:
        return ObjectCentricLimbRepositioning3DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        names = ["robot", "limb"]
        if any(o.name == "fixture" for o in exemplar_state):
            names.append("fixture")
        return names

    # Implicit concatenation below, since docformatter rewraps triple quotes.

    def _create_env_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return (
            "A 3D task where the robot must reposition a passive human limb, as an assistive robot would when helping someone move an arm or a leg.\n"
            "\n"
            "The robot's end effector is welded to the limb's grasp frame and drives the limb to a goal joint configuration with joint torques. The limb has no actuation of its own. The goal is drawn as a translucent green copy of the limb.\n"
            "\n"
            "- Robot: a TidyBot Kinova Gen3 arm with 7 joints, on a base that stays put\n"
            "- Simulation: PyBullet forward dynamics, unlike Kinematic3D and Dynamic3D\n"
            "- Gravity: off by default; set `gravity` in the config to enable it\n"
            "\n"
            "On reset, each limb joint is perturbed within that limb's joint range around the variant's nominal starting configuration, and the draw is rejected if it pushes the limb into the scene, so the initial state varies with the seed while staying collision free. The robot re-solves its grasp for the sampled configuration.\n"
        )

    def _create_variant_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return (
            "The variants pair a scene (`isolated`, `human`, `wheelchair`, `bed`) with a limb (`left-arm`, `right-arm`, `left-leg`, `right-leg`).\n"
            "\n"
            "Each of the sixteen has its own robot placement, initial limb configuration, and goal.\n"
        )

    def _create_variant_specific_description(self) -> str:
        # pylint: disable=line-too-long
        scene = create_variant_config(self.variant).scene
        limb = scene.limb_name.removeprefix("human-").replace("-", " ")
        return f"Reposition the {limb} {_SCENE_DESCRIPTIONS[scene.scene_type]}."

    def _create_reward_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return (
            "A reward of -1 is given at every step until the goal is reached, so maximizing reward means repositioning the limb in as few steps as possible.\n"
            "\n"
            "The episode terminates when the limb's joint configuration comes within `goal_atol` (0.1 radians by default) of the goal.\n"
        )

    def _create_references_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return (
            "Ported from the `limb-manipulation` repository, which the human limb models and the repositioning tasks are taken from.\n"
            "\n"
            "https://github.com/empriselab/limb-manipulation\n"
            "\n"
            "The Franka Panda used there is replaced here by TidyBot.\n"
        )


# Initial and goal limb configurations, shared by the scenes without furniture.
_FREE_LIMB_INIT = (
    -0.40335719438145545,
    -0.44707218123342596,
    -0.09289427173389816,
    -0.5010609575327076,
    0.0,
    0.0,
)
_FREE_LIMB_GOAL = (-0.7, 0.0, -0.03, -0.8, 0.0, 0.0)

# Where the limb floats in the isolated scenes, and where it sits in the human scenes.
_FREE_LIMB_BASE_POSE = Pose.from_rpy((0.730, -0.396, 0.270), (0.0, 0.0, np.pi))

# Seated and lying postures for the limbs that are not being repositioned.
_WHEELCHAIR_TORSO_POSE = Pose.from_rpy((0.0, -0.15, 0.75), (-0.2, 0.0, 0.0))
_WHEELCHAIR_RESTING_JOINTS = {
    "left_arm_init_joint_positions": (0.0, -0.3, -0.2, -1.1, 0.0, 0.0),
    "right_arm_init_joint_positions": (0.0, 0.3, 0.2, -1.1, 0.0, 0.0),
    "left_leg_init_joint_positions": (-1.3, 0.0, 0.0, 1.57, 0.0, 0.0),
    "right_leg_init_joint_positions": (-1.3, 0.0, 0.0, 1.57, 0.0, 0.0),
}
# Bed offsets, applied to both the bed and the human on it.
_BED_HEIGHT_OFFSET = -0.25
_BED_LATERAL_OFFSET = 0.10
_BED_POSE = Pose.from_rpy((0.0, 0.0, 0.6 + _BED_HEIGHT_OFFSET), (0.0, 0.0, 0.0))


def _bed_torso_pose(limb_name: str) -> Pose:
    """The torso pose for a bed scene, shifted toward the limb being repositioned."""
    sign = 1.0 if "left" in limb_name else -1.0
    return Pose.from_rpy(
        (sign * _BED_LATERAL_OFFSET, 0.0, 0.9 + _BED_HEIGHT_OFFSET),
        (-np.pi / 2, 0.0, 0.0),
    )


# Cameras per scene type, with pitch in [-90, 90] or PyBullet renders upside down.
_FREE_CAMERA = {
    "camera_distance": 2.0,
    "camera_yaw": 140.0,
    "camera_pitch": -20.0,
}
_WHEELCHAIR_CAMERA = {
    "camera_distance": 3.0,
    "camera_yaw": 320.0,
    "camera_pitch": -30.0,
    "camera_target": _WHEELCHAIR_TORSO_POSE.position,
}
# The bed camera target is filled in per variant.
_BED_CAMERA = {
    "camera_distance": 3.0,
    "camera_yaw": 60.0,
    "camera_pitch": -40.0,
}

LIMB_NAMES = tuple(n.removeprefix("human-") for n in ALL_LIMB_NAMES)
SCENE_TYPES = ALL_SCENE_TYPES

# Used to describe each variant in the docs.
_SCENE_DESCRIPTIONS = {
    "isolated": "on its own in an empty world",
    "human": "of a human torso with all four limbs attached",
    "wheelchair": "of a human seated in a wheelchair",
    "bed": "of a human lying on a hospital bed",
}


def _mirror_arm(joints: tuple[float, ...]) -> tuple[float, ...]:
    """The arm configuration mirrored across the body's sagittal plane."""
    # The left arm URDF already flips all but the second and sixth axes.
    q = joints
    return (-q[0], q[1], -q[2], -q[3], -q[4], q[5])


def _flip_knee(joints: tuple[float, ...]) -> tuple[float, ...]:
    """The leg configuration with the knee, the fourth joint, bent the other way."""
    q = joints
    return (q[0], q[1], q[2], -q[3], q[4], q[5])


# Initial and goal joint positions, per variant.
_LIMB_TASKS: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "wheelchair-left-arm": (
        (
            0.40335719438145545,
            0.44707218123342596,
            -0.09289427173389816,
            0.5010609575327076,
            0.0,
            0.0,
        ),
        (0.7, 0.0, -0.03, 0.8, 0.0, 0.0),
    ),
    "wheelchair-right-arm": (
        (
            -0.40335719438145545,
            0.44707218123342596,
            0.09289427173389816,
            -0.5010609575327076,
            0.0,
            0.0,
        ),
        (-0.7, 0.0, -0.03, -0.8, 0.0, 0.0),
    ),
    "wheelchair-left-leg": (
        (-1.3, -0.2, 0.0, 0.5, 0.0, 0.0),
        (-1.3, 0.0, 0.0, 1.57, 0.0, 0.0),
    ),
    "wheelchair-right-leg": (
        (-1.3, 0.2, 0.0, 0.5, 0.0, 0.0),
        (-1.3, 0.0, 0.0, 1.57, 0.0, 0.0),
    ),
    "bed-left-arm": (
        _mirror_arm((0.0, 0.3, 0.0, 0.0, 0.0, 0.0)),
        _mirror_arm((-0.3, 0.0, -0.03, -0.2, 0.0, 0.0)),
    ),
    "bed-right-arm": (
        (0.0, 0.3, 0.0, 0.0, 0.0, 0.0),
        (-0.3, 0.0, -0.03, -0.2, 0.0, 0.0),
    ),
    "bed-left-leg": ((0.0,) * 6, (-0.5, -0.1, 0.0, 0.4, 0.0, 0.0)),
    "bed-right-leg": ((0.0,) * 6, (-0.5, 0.1, 0.0, 0.4, 0.0, 0.0)),
}
for _limb in LIMB_NAMES:
    for _scene in ("isolated", "human"):
        _LIMB_TASKS[f"{_scene}-{_limb}"] = (_FREE_LIMB_INIT, _FREE_LIMB_GOAL)


_LIMB_TASKS["human-left-arm"] = (
    _mirror_arm(_FREE_LIMB_INIT),
    _mirror_arm(_FREE_LIMB_GOAL),
)
for _scene in ("isolated", "human"):
    for _limb in ("left-leg", "right-leg"):
        _LIMB_TASKS[f"{_scene}-{_limb}"] = (
            _flip_knee(_FREE_LIMB_INIT),
            _flip_knee(_FREE_LIMB_GOAL),
        )

ALL_VARIANTS = tuple(f"{scene}-{limb}" for scene in SCENE_TYPES for limb in LIMB_NAMES)

# Where TidyBot stands, as (base SE2 pose, base height).
_ROBOT_PLACEMENTS: dict[str, tuple[SE2Pose, float]] = {
    "isolated-left-arm": (SE2Pose(1.0196, -0.6916, -0.3958), -0.701),
    "isolated-right-arm": (SE2Pose(0.3566, 0.4849, -0.6971), -0.329),
    "isolated-left-leg": (SE2Pose(0.1548, -0.1575, 2.1953), -0.965),
    "isolated-right-leg": (SE2Pose(0.1548, -0.1575, 2.1953), -0.965),
    "human-left-arm": (SE2Pose(1.0787, 0.1832, -0.8140), -0.726),
    "human-right-arm": (SE2Pose(0.3566, 0.4849, -0.6971), -0.329),
    "human-left-leg": (SE2Pose(0.1651, 0.4351, 0.4988), -0.976),
    "human-right-leg": (SE2Pose(0.1626, 0.4349, 0.5051), -0.976),
    "wheelchair-left-arm": (SE2Pose(0.9111, -0.6310, -2.6352), 0.335),
    "wheelchair-right-arm": (SE2Pose(-0.7040, -1.1983, 0.7716), 0.510),
    "wheelchair-left-leg": (SE2Pose(0.6645, -1.0218, -2.4336), 0.0),
    "wheelchair-right-leg": (SE2Pose(-0.6661, -0.5696, -0.3190), 0.0),
    "bed-left-arm": (SE2Pose(0.9078, 0.0224, -2.9401), 0.0),
    "bed-right-arm": (SE2Pose(-0.8557, 0.0837, -0.9312), 0.0),
    "bed-left-leg": (SE2Pose(0.7701, -0.9749, -1.7739), 0.495),
    "bed-right-leg": (SE2Pose(-0.8897, -0.8474, 1.4083), 0.177),
}


def create_variant_config(variant: str) -> LimbRepositioning3DEnvConfig:
    """Create the config for one of the sixteen variants, e.g. "bed-right-leg"."""
    if variant not in _LIMB_TASKS:
        raise ValueError(
            f"Unknown variant {variant!r}; expected one of {sorted(_LIMB_TASKS)}"
        )
    limb_init, limb_goal = _LIMB_TASKS[variant]
    base_pose, base_z = _ROBOT_PLACEMENTS[variant]
    scene_type, limb_suffix = variant.split("-", 1)
    limb_name = f"human-{limb_suffix}"

    scene_kwargs: dict[str, Any] = {
        "scene_type": scene_type,
        "limb_name": limb_name,
        "limb_init_joint_positions": limb_init,
        "limb_goal_joint_positions": limb_goal,
    }
    camera_kwargs: dict[str, Any] = {}

    if scene_type == "isolated":
        scene_kwargs["limb_base_pose"] = _FREE_LIMB_BASE_POSE
        camera_kwargs = dict(_FREE_CAMERA, camera_target=_FREE_LIMB_BASE_POSE.position)
    elif scene_type == "human":
        scene_kwargs["limb_base_pose"] = _FREE_LIMB_BASE_POSE
        scene_kwargs["use_limb_pose_to_initialize"] = True
        camera_kwargs = dict(_FREE_CAMERA, camera_target=_FREE_LIMB_BASE_POSE.position)
    elif scene_type == "wheelchair":
        scene_kwargs["torso_base_pose"] = _WHEELCHAIR_TORSO_POSE
        scene_kwargs.update(_WHEELCHAIR_RESTING_JOINTS)
        camera_kwargs = dict(_WHEELCHAIR_CAMERA)
    else:
        torso_pose = _bed_torso_pose(limb_name)
        scene_kwargs["torso_base_pose"] = torso_pose
        scene_kwargs["bed_pose"] = _BED_POSE
        camera_kwargs = dict(_BED_CAMERA, camera_target=torso_pose.position)

    return LimbRepositioning3DEnvConfig(
        scene=LimbRepositioningSceneConfig(**scene_kwargs),
        robot_base_home_pose=base_pose,
        robot_base_z=base_z,
        **camera_kwargs,
    )
