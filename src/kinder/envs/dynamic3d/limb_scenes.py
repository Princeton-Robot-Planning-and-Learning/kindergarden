"""Scenes that a human limb can be repositioned in.

There are four kinds of scene, in increasing order of clutter:

* isolated: the limb floats in an empty world, with no body attached.
* human: the limb is attached to a torso, along with the three other limbs.
* wheelchair: a human seated in a wheelchair, in a room.
* bed: a human lying on a hospital bed, in a room.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose, multiply_poses, set_pose
from pybullet_helpers.utils import create_pybullet_block

from kinder.envs.dynamic3d.limb_utils import ASSETS_DIR, NUM_LIMB_JOINTS
from kinder.envs.dynamic3d.limbs import (
    ALL_LIMB_NAMES,
    DEFAULT_BODY_MASS,
    DEFAULT_RANGE_OF_MOTION,
    BodyMass,
    HumanLimbPyBulletRobot,
    RangeOfMotion,
    create_human_limb,
    validate_human_joint_positions,
)

# Offsets from a limb's grasp frame to the robot's end effector
_GRIPPER_APPROACH_SPIN = Pose.from_rpy((0.0, 0.0, 0.0), (0.0, 0.0, np.pi / 2))
ARM_GRASP_TRANSFORM = multiply_poses(
    Pose.from_rpy((0.0, 0.0, 0.0), (0.0, -np.pi / 2, 0.0)), _GRIPPER_APPROACH_SPIN
)
LEG_GRASP_TRANSFORM = multiply_poses(
    Pose.from_rpy((0.0, 0.0, 0.0), (0.0, -np.pi / 2, -np.pi / 2)),
    _GRIPPER_APPROACH_SPIN,
)

_RESTING_JOINTS = (0.0,) * NUM_LIMB_JOINTS


@dataclass(frozen=True)
class LimbRepositioningSceneConfig:
    """Static description of a scene: what is in it and where."""

    # One of "isolated", "human", "wheelchair", "bed".
    scene_type: str
    # The limb being repositioned, e.g. "human-left-arm".
    limb_name: str
    limb_init_joint_positions: tuple[float, ...]
    limb_goal_joint_positions: tuple[float, ...]
    limb_joint_limits_model_name: str = "box"
    limb_muscle_tone_model_name: str = "none"
    limb_range_of_motion: RangeOfMotion | Mapping[str, RangeOfMotion] = (
        DEFAULT_RANGE_OF_MOTION
    )
    limb_body_mass: BodyMass | Mapping[str, BodyMass] = DEFAULT_BODY_MASS

    torso_base_pose: Pose = Pose.identity()
    limb_base_pose: Pose = Pose.identity()
    use_limb_pose_to_initialize: bool = False

    # Resting joint positions of the limbs that are not being repositioned.
    left_arm_init_joint_positions: tuple[float, ...] = _RESTING_JOINTS
    right_arm_init_joint_positions: tuple[float, ...] = _RESTING_JOINTS
    left_leg_init_joint_positions: tuple[float, ...] = _RESTING_JOINTS
    right_leg_init_joint_positions: tuple[float, ...] = _RESTING_JOINTS

    # Where each limb attaches to the torso.
    torso_to_left_arm: Pose = Pose.from_rpy((0.24, 0.048, 0.45), (0.0, 0.0, 0.0))
    torso_to_right_arm: Pose = Pose.from_rpy((-0.24, 0.048, 0.45), (0.0, 0.0, 0.0))
    torso_to_left_leg: Pose = Pose.from_rpy((0.11, 0.0, 0.0), (0.0, 0.0, 0.0))
    torso_to_right_leg: Pose = Pose.from_rpy((-0.11, 0.0, 0.0), (0.0, 0.0, 0.0))

    # Room, used by the wheelchair and bed scenes.
    floor_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    wall_poses: tuple[Pose, ...] = (
        Pose.from_rpy((0.0, 2.5, 0.0), (np.pi / 2, 0.0, np.pi / 2)),
        Pose.from_rpy((3.0, 0.0, 0.0), (np.pi / 2, 0.0, 0.0)),
        Pose.from_rpy((-3.0, 0.0, 0.0), (np.pi / 2, 0.0, 0.0)),
    )
    wall_half_extents: tuple[float, float, float] = (0.1, 3.0, 5.0)

    # Furniture.
    wheelchair_pose: Pose = Pose.from_rpy((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    bed_pose: Pose = Pose.from_rpy((0.0, 0.0, 0.6), (0.0, 0.0, 0.0))

    # Visualization of the goal configuration of the limb.
    show_goal: bool = True
    goal_rgba: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.5)

    def __post_init__(self) -> None:
        assert self.scene_type in ALL_SCENE_TYPES
        assert self.limb_name in ALL_LIMB_NAMES
        assert len(self.limb_init_joint_positions) == NUM_LIMB_JOINTS
        assert len(self.limb_goal_joint_positions) == NUM_LIMB_JOINTS
        self._validate_joint_limits()
        if self.scene_type == "isolated":
            return
        if self.use_limb_pose_to_initialize:
            torso = multiply_poses(self.limb_base_pose, self.torso_to_limb.invert())
            object.__setattr__(self, "torso_base_pose", torso)
        else:
            limb = multiply_poses(self.torso_base_pose, self.torso_to_limb)
            object.__setattr__(self, "limb_base_pose", limb)

    def _validate_joint_limits(self) -> None:
        """Reject a scene that starts, or aims, outside the person's joint limits."""
        if self.limb_joint_limits_model_name == "none":
            return
        limbs = (
            [self.limb_name] if self.scene_type == "isolated" else list(ALL_LIMB_NAMES)
        )
        for limb_name in limbs:
            validate_human_joint_positions(
                limb_name,
                list(self.get_limb_init_joint_positions(limb_name)),
                f"the initial configuration of {limb_name}",
                self.get_limb_range_of_motion(limb_name),
                self.limb_joint_limits_model_name,
            )
        validate_human_joint_positions(
            self.limb_name,
            list(self.limb_goal_joint_positions),
            f"the goal configuration of {self.limb_name}",
            self.get_limb_range_of_motion(self.limb_name),
            self.limb_joint_limits_model_name,
        )

    def get_torso_to_limb(self, limb_name: str) -> Pose:
        """The transform from the torso to any limb, active or not."""
        return {
            "human-left-arm": self.torso_to_left_arm,
            "human-right-arm": self.torso_to_right_arm,
            "human-left-leg": self.torso_to_left_leg,
            "human-right-leg": self.torso_to_right_leg,
        }[limb_name]

    @property
    def torso_to_limb(self) -> Pose:
        """The transform from the torso to the limb being repositioned."""
        return self.get_torso_to_limb(self.limb_name)

    @property
    def grasp_transform(self) -> Pose:
        """The transform from the limb's grasp frame to the robot's end effector."""
        return ARM_GRASP_TRANSFORM if "arm" in self.limb_name else LEG_GRASP_TRANSFORM

    def get_limb_init_joint_positions(self, limb_name: str) -> tuple[float, ...]:
        """The initial joint positions of any limb, active or not."""
        if limb_name == self.limb_name:
            return self.limb_init_joint_positions
        return {
            "human-left-arm": self.left_arm_init_joint_positions,
            "human-right-arm": self.right_arm_init_joint_positions,
            "human-left-leg": self.left_leg_init_joint_positions,
            "human-right-leg": self.right_leg_init_joint_positions,
        }[limb_name]

    def get_limb_range_of_motion(self, limb_name: str) -> RangeOfMotion:
        """The range of motion of any limb, active or not."""
        if isinstance(self.limb_range_of_motion, RangeOfMotion):
            return self.limb_range_of_motion
        return self.limb_range_of_motion.get(limb_name, DEFAULT_RANGE_OF_MOTION)

    def get_limb_body_mass(self, limb_name: str) -> BodyMass:
        """The body mass of any limb, active or not."""
        if isinstance(self.limb_body_mass, BodyMass):
            return self.limb_body_mass
        return self.limb_body_mass.get(limb_name, DEFAULT_BODY_MASS)

    def get_limb_base_pose(self, limb_name: str) -> Pose:
        """The base pose of any limb, active or not."""
        if self.scene_type == "isolated":
            assert limb_name == self.limb_name
            return self.limb_base_pose
        return multiply_poses(self.torso_base_pose, self.get_torso_to_limb(limb_name))


class LimbRepositioningScene(abc.ABC):
    """Everything in the world except the robot."""

    def __init__(
        self, scene_config: LimbRepositioningSceneConfig, physics_client_id: int
    ) -> None:
        self.scene_config = scene_config
        self.physics_client_id = physics_client_id
        self._create_bodies()
        self.passive_limb = self._create_passive_limb()
        if scene_config.show_goal:
            self._visualize_goal()

    @property
    def grasp_transform(self) -> Pose:
        """The transform from the limb's grasp frame to the robot's end effector."""
        return self.scene_config.grasp_transform

    @property
    def furniture_id(self) -> int | None:
        """The body the human is sitting or lying on, if the scene has one."""
        return None

    @abc.abstractmethod
    def _create_bodies(self) -> None:
        """Create everything except the passive limb."""

    @abc.abstractmethod
    def _create_passive_limb(self) -> HumanLimbPyBulletRobot:
        """Create (or look up) the limb that the robot repositions."""

    @abc.abstractmethod
    def get_scene_collision_ids(self) -> list[int]:
        """The bodies that the robot and limb should avoid."""

    def get_limb_obstacle_ids(self) -> list[int]:
        """The bodies the repositioned limb should not be driven into."""
        return self.get_scene_collision_ids()

    def _create_limb(
        self, limb_name: str, joint_positions: tuple[float, ...] | None = None
    ) -> HumanLimbPyBulletRobot:
        config = self.scene_config
        if joint_positions is None:
            joint_positions = config.get_limb_init_joint_positions(limb_name)
        return create_human_limb(
            limb_name,
            config.get_limb_base_pose(limb_name),
            list(joint_positions),
            config.limb_joint_limits_model_name,
            config.limb_muscle_tone_model_name,
            self.physics_client_id,
            config.get_limb_range_of_motion(limb_name),
            config.get_limb_body_mass(limb_name),
        )

    def _visualize_goal(self) -> None:
        """Show a translucent copy of the limb in its goal configuration."""
        config = self.scene_config
        goal_limb = self._create_limb(
            config.limb_name, config.limb_goal_joint_positions
        )
        num_joints = p.getNumJoints(
            goal_limb.robot_id, physicsClientId=self.physics_client_id
        )
        for joint in range(-1, num_joints):
            p.changeDynamics(
                goal_limb.robot_id,
                joint,
                mass=0.0,
                physicsClientId=self.physics_client_id,
            )
            p.changeVisualShape(
                goal_limb.robot_id,
                joint,
                rgbaColor=list(config.goal_rgba),
                physicsClientId=self.physics_client_id,
            )
            p.setCollisionFilterGroupMask(
                goal_limb.robot_id,
                joint,
                0,
                0,
                physicsClientId=self.physics_client_id,
            )
        self.goal_limb = goal_limb


class IsolatedLimbScene(LimbRepositioningScene):
    """A single limb floating in an empty world."""

    def _create_bodies(self) -> None:
        pass

    def _create_passive_limb(self) -> HumanLimbPyBulletRobot:
        return self._create_limb(self.scene_config.limb_name)

    def get_scene_collision_ids(self) -> list[int]:
        return []


class HumanScene(LimbRepositioningScene):
    """A whole human: a static torso and head, plus four limbs."""

    def _create_bodies(self) -> None:
        config = self.scene_config
        self.torso_id = p.loadURDF(
            str(ASSETS_DIR / "human" / "torso_and_head.urdf"),
            useFixedBase=True,
            physicsClientId=self.physics_client_id,
        )
        set_pose(self.torso_id, config.torso_base_pose, self.physics_client_id)

        self.limbs: dict[str, HumanLimbPyBulletRobot] = {
            limb_name: self._create_limb(limb_name) for limb_name in ALL_LIMB_NAMES
        }

        # Only the limb being repositioned participates in collision checking.
        for limb_name, limb in self.limbs.items():
            if limb_name == config.limb_name:
                continue
            num_joints = p.getNumJoints(
                limb.robot_id, physicsClientId=self.physics_client_id
            )
            for joint in range(num_joints):
                p.setCollisionFilterGroupMask(
                    limb.robot_id,
                    joint,
                    0,
                    0,
                    physicsClientId=self.physics_client_id,
                )

    def _create_passive_limb(self) -> HumanLimbPyBulletRobot:
        return self.limbs[self.scene_config.limb_name]

    def get_scene_collision_ids(self) -> list[int]:
        return []

    def get_limb_obstacle_ids(self) -> list[int]:
        others = [
            limb.robot_id
            for name, limb in self.limbs.items()
            if name != self.scene_config.limb_name
        ]
        return [self.torso_id] + others + super().get_limb_obstacle_ids()


class RoomScene(HumanScene):
    """A human in a room, with a textured floor and three walls."""

    def _create_bodies(self) -> None:
        super()._create_bodies()
        config = self.scene_config

        self.floor_id = p.loadURDF(
            str(ASSETS_DIR / "floor" / "floor.urdf"),
            config.floor_position,
            useFixedBase=True,
            physicsClientId=self.physics_client_id,
        )
        floor_texture_id = p.loadTexture(
            str(ASSETS_DIR / "floor" / "floor_texture.jpg"), self.physics_client_id
        )
        p.changeVisualShape(
            self.floor_id,
            -1,
            textureUniqueId=floor_texture_id,
            physicsClientId=self.physics_client_id,
        )

        self.wall_ids = [
            create_pybullet_block(
                (1.0, 1.0, 1.0, 1.0),
                config.wall_half_extents,
                self.physics_client_id,
            )
            for _ in config.wall_poses
        ]
        wall_texture_id = p.loadTexture(
            str(ASSETS_DIR / "light_blue_wall_texture.jpg"), self.physics_client_id
        )
        for wall_id, pose in zip(self.wall_ids, config.wall_poses, strict=True):
            p.changeVisualShape(
                wall_id,
                -1,
                textureUniqueId=wall_texture_id,
                physicsClientId=self.physics_client_id,
            )
            set_pose(wall_id, pose, self.physics_client_id)

        self._furniture_id = self._create_furniture()

    @property
    def furniture_id(self) -> int | None:
        return self._furniture_id

    @abc.abstractmethod
    def _create_furniture(self) -> int:
        """Create the thing the human is sitting or lying on."""

    def _load_furniture(self, urdf_path: Path, pose: Pose) -> int:
        body_id = p.loadURDF(
            str(urdf_path),
            useFixedBase=True,
            physicsClientId=self.physics_client_id,
        )
        set_pose(body_id, pose, self.physics_client_id)
        return body_id

    def get_scene_collision_ids(self) -> list[int]:
        return [self._furniture_id] + super().get_scene_collision_ids()


class WheelchairScene(RoomScene):
    """A human seated in a wheelchair."""

    def _create_furniture(self) -> int:
        return self._load_furniture(
            ASSETS_DIR / "wheelchair" / "wheelchair.urdf",
            self.scene_config.wheelchair_pose,
        )


class BedScene(RoomScene):
    """A human lying on a hospital bed."""

    def _create_furniture(self) -> int:
        return self._load_furniture(
            ASSETS_DIR / "hospital_bed" / "hospital_bed.urdf",
            self.scene_config.bed_pose,
        )


SCENE_TYPE_TO_CLS: dict[str, type[LimbRepositioningScene]] = {
    "isolated": IsolatedLimbScene,
    "human": HumanScene,
    "wheelchair": WheelchairScene,
    "bed": BedScene,
}
ALL_SCENE_TYPES = tuple(SCENE_TYPE_TO_CLS)


def create_scene(
    scene_config: LimbRepositioningSceneConfig, physics_client_id: int
) -> LimbRepositioningScene:
    """Create a scene from its config."""
    return SCENE_TYPE_TO_CLS[scene_config.scene_type](scene_config, physics_client_id)
