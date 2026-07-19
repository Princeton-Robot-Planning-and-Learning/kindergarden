"""Environment where the goal is to reach a target region."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Type as TypingType

import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose, SE2Pose, get_pose, set_pose
from relational_structs import Object, ObjectCentricState
from relational_structs.utils import create_state_from_dict

from kinder.core import ConstantObjectKinDEREnv, FinalConfigMeta
from kinder.envs.kinematic3d.base_env import (
    Kinematic3DEnvConfig,
    ObjectCentricKinematic3DRobotEnv,
)
from kinder.envs.kinematic3d.object_types import (
    Kinematic3DEnvTypeFeatures,
    Kinematic3DPointType,
    Kinematic3DRobotType,
)
from kinder.envs.kinematic3d.utils import Kinematic3DObjectCentricState


@dataclass(frozen=True)
class BaseMotion3DEnvConfig(Kinematic3DEnvConfig, metaclass=FinalConfigMeta):
    """Config for BaseMotion3DEnv()."""

    # Robot.
    robot_name: str = "tidybot-kinova"
    check_base_collisions: bool = True

    # Target.
    target_radius: float = 0.05
    target_z: float = 0.2
    target_color: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 0.5)
    target_lower_bound: SE2Pose = SE2Pose(-2, -2, -np.pi)
    target_upper_bound: SE2Pose = SE2Pose(2, 2, np.pi)


class BaseMotion3DObjectCentricState(Kinematic3DObjectCentricState):
    """A state in the BaseMotion3DEnv().

    Adds convenience methods on top of Kinematic3DObjectCentricState().
    """

    @property
    def target_base_pose(self) -> SE2Pose:
        """The pose of the base target, assuming the name "target"."""
        target = self.get_object_from_name("target")
        pose = Pose(
            (self.get(target, "x"), self.get(target, "y"), self.get(target, "z"))
        )
        se2_pose = pose.to_se2()
        return se2_pose


class ObjectCentricBaseMotion3DEnv(
    ObjectCentricKinematic3DRobotEnv[
        BaseMotion3DObjectCentricState, BaseMotion3DEnvConfig
    ]
):
    """Environment where the goal is to reach a target region."""

    def __init__(
        self, config: BaseMotion3DEnvConfig = BaseMotion3DEnvConfig(), **kwargs
    ) -> None:
        super().__init__(config=config, **kwargs)

        # Create target.
        visual_id = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self.config.target_radius,
            rgbaColor=self.config.target_color,
            physicsClientId=self.physics_client_id,
        )

        # Create the body.
        self.target_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_id,
            basePosition=(0, 0, 0),  # set in reset()
            baseOrientation=(0, 0, 0, 1),
            physicsClientId=self.physics_client_id,
        )

    @property
    def state_cls(self) -> TypingType[Kinematic3DObjectCentricState]:
        return BaseMotion3DObjectCentricState

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        # Neither the target nor the robot are constant in this env.
        return {}

    def _reset_objects(self) -> None:
        # Reset the target. Sample and check that the robot has not already reached it.
        target_pose: SE2Pose | None = None
        lb = self.config.target_lower_bound
        ub = self.config.target_upper_bound
        robot_base_pose = self.robot.base.get_pose()
        for _ in range(100_000):
            x, y, rot = self.np_random.uniform(
                (lb.x, lb.y, lb.rot), (ub.x, ub.y, ub.rot)
            )
            target_pose = SE2Pose(x, y, rot)
            # If the goal is already reached, keep sampling.
            if not self._robot_at_target_pose(robot_base_pose, target_pose):
                break
        else:
            raise RuntimeError("Failed to find reachable target position")
        target_se3_pose = target_pose.to_se3(self.config.target_z)
        set_pose(self.target_id, target_se3_pose, self.physics_client_id)

    def _set_object_states(self, obs: BaseMotion3DObjectCentricState) -> None:
        assert self.target_id is not None
        target_se3_pose = obs.target_base_pose.to_se3(0.0)
        set_pose(self.target_id, target_se3_pose, self.physics_client_id)

    def _object_name_to_pybullet_id(self, object_name: str) -> int:
        if object_name == "target":
            return self.target_id
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_collision_object_ids(self) -> set[int]:
        collision_ids = set()
        if self._realistic_bg_id is not None:
            collision_ids.add(self._realistic_bg_id)
        return collision_ids

    def _get_movable_object_names(self) -> set[str]:
        return set()

    def _get_surface_object_names(self) -> set[str]:
        return set()

    def _get_half_extents(self, object_name: str) -> tuple[float, float, float]:
        raise NotImplementedError("No objects have half extents")

    def _get_obs(self) -> BaseMotion3DObjectCentricState:
        state_dict = self._create_state_dict(
            [("robot", Kinematic3DRobotType), ("target", Kinematic3DPointType)]
        )
        state = create_state_from_dict(
            state_dict,
            Kinematic3DEnvTypeFeatures,
            state_cls=BaseMotion3DObjectCentricState,
        )
        assert isinstance(state, BaseMotion3DObjectCentricState)
        return state

    def _robot_at_target_pose(
        self, robot_base_pose: SE2Pose, target_pose: SE2Pose
    ) -> bool:
        dist = float(
            np.linalg.norm(
                np.array(
                    [
                        target_pose.x - robot_base_pose.x,
                        target_pose.y - robot_base_pose.y,
                    ]
                )
            )
        )
        return dist < self.config.target_radius

    def goal_reached(self) -> bool:
        robot_base_pose = self.robot.base.get_pose()
        target_se3_pose = get_pose(self.target_id, self.physics_client_id)
        target_pose = target_se3_pose.to_se2()
        return self._robot_at_target_pose(robot_base_pose, target_pose)


class BaseMotion3DEnv(ConstantObjectKinDEREnv):
    """Base motion 3D env with a constant number of objects."""

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricKinematic3DRobotEnv:
        return ObjectCentricBaseMotion3DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        return ["robot", "target"]

    def _create_env_markdown_description(self) -> str:
        """Create environment description."""
        # pylint: disable=line-too-long
        return """A simple 3D environment where the goal is to navigate to a target region."""

    def _create_variant_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return "This environment has only one variant."

    def _create_reward_markdown_description(self) -> str:
        """Create reward description."""
        # pylint: disable=line-too-long
        return (
            """The reward is -1 per timestep to encourage reaching the goal quickly."""
        )

    def _create_references_markdown_description(self) -> str:
        """Create references description."""
        # pylint: disable=line-too-long
        return """This is a very common kind of environment. The background is adapted from the [Replica dataset](https://arxiv.org/abs/1906.05797) (Straub et al., 2019)."""
