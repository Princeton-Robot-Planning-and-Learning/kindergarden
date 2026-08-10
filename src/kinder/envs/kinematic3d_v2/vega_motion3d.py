"""Environment where the goal is to reach a target region with a Dexmate Vega arm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Type as TypingType

import numpy as np
from prpl_kinematics.geometry.shapes import SphereShape
from prpl_kinematics.tree.kinematic_tree import Node
from relational_structs import Object, ObjectCentricState
from relational_structs.utils import create_state_from_dict
from spatialmath import SE3

from kinder.core import ConstantObjectKinDEREnv, FinalConfigMeta
from kinder.envs.kinematic3d_v2.base_env import (
    Kinematic3Dv2EnvConfig,
    Kinematic3Dv2ObjectCentricState,
    ObjectCentricKinematic3Dv2RobotEnv,
)
from kinder.envs.kinematic3d_v2.object_types import (
    Kinematic3Dv2ArmRobotType,
    Kinematic3Dv2EnvTypeFeatures,
    Kinematic3Dv2PointType,
)

# The name of the kinematic tree node holding the target sphere.
TARGET_NODE = "target"


@dataclass(frozen=True)
class VegaMotion3DEnvConfig(Kinematic3Dv2EnvConfig, metaclass=FinalConfigMeta):
    """Config for VegaMotion3DEnv()."""

    # Robot.
    robot_name: str = "vega"
    manipulator: str = "right"

    # Target. The bounds cover the region in front of and to the right of the torso that
    # the right arm can reach without folding back through the robot's own body.
    target_radius: float = 0.1
    target_color: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 0.5)
    target_lower_bound: tuple[float, float, float] = (0.30, -0.70, 0.60)
    target_upper_bound: tuple[float, float, float] = (0.70, -0.10, 1.10)


class VegaMotion3DObjectCentricState(Kinematic3Dv2ObjectCentricState):
    """A state in the VegaMotion3DEnv().

    Adds convenience methods on top of Kinematic3Dv2ObjectCentricState().
    """

    @property
    def target_position(self) -> tuple[float, float, float]:
        """The position of the target, assuming the name "target"."""
        target = self.get_object_from_name(TARGET_NODE)
        return (self.get(target, "x"), self.get(target, "y"), self.get(target, "z"))


class ObjectCentricVegaMotion3DEnv(
    ObjectCentricKinematic3Dv2RobotEnv[
        VegaMotion3DObjectCentricState, VegaMotion3DEnvConfig
    ]
):
    """Environment where the goal is to reach a target region with a Vega arm."""

    def __init__(
        self, config: VegaMotion3DEnvConfig = VegaMotion3DEnvConfig(), **kwargs
    ) -> None:
        super().__init__(config=config, **kwargs)
        # The orientation that targets are checked for reachability at. Holding the home
        # orientation fixed keeps the check cheap; the goal itself is position-only.
        self._target_rotation = np.array(self.end_effector_pose.R)

    def target_reach_pose(self, position: tuple[float, float, float]) -> SE3:
        """The end-effector pose an arm reaching ``position`` is asked to achieve."""
        return SE3.Rt(self._target_rotation, position)

    def _create_scene_geometry(self) -> None:
        # The target is a visual-only sphere: it marks the goal region rather than
        # obstructing the arm, so it is deliberately not given collision geometry.
        self.tree.add_node(
            Node(
                TARGET_NODE,
                visuals=[
                    SphereShape(
                        radius=self.config.target_radius,
                        color=self.config.target_color,
                    )
                ],
            )
        )
        self._set_target_position((0.0, 0.0, 0.0))  # set for real in reset()

    def _set_target_position(self, position: tuple[float, float, float]) -> None:
        self.tree.attach(TARGET_NODE, self.tree.root, SE3(*position))

    @property
    def state_cls(self) -> TypingType[Kinematic3Dv2ObjectCentricState]:
        return VegaMotion3DObjectCentricState

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        # Neither the target nor the robot are constant in this env.
        return {}

    def _reset_objects(self) -> None:
        # Reset the target. Sample and check reachability. The retry budget is small
        # because each attempt runs an IK solve, which costs tens of milliseconds; the
        # bounds are chosen so that the great majority of samples are reachable.
        for _ in range(1_000):
            position = self.np_random.uniform(
                self.config.target_lower_bound, self.config.target_upper_bound
            )
            target_pose = self.target_reach_pose(tuple(position))
            if self._manipulator.ik.solve(target_pose, self.configuration) is None:
                continue
            self._set_target_position(tuple(position))
            # If the goal is already reached, keep sampling.
            if not self.goal_reached():
                break
        else:
            raise RuntimeError("Failed to find reachable target position")

    def _set_object_states(self, obs: VegaMotion3DObjectCentricState) -> None:
        self._set_target_position(obs.target_position)

    def _get_obs(self) -> VegaMotion3DObjectCentricState:
        state_dict = self._create_state_dict(
            [
                ("robot", Kinematic3Dv2ArmRobotType),
                (TARGET_NODE, Kinematic3Dv2PointType),
            ]
        )
        state = create_state_from_dict(
            state_dict,
            Kinematic3Dv2EnvTypeFeatures,
            state_cls=VegaMotion3DObjectCentricState,
        )
        assert isinstance(state, VegaMotion3DObjectCentricState)
        return state

    def goal_reached(self) -> bool:
        target = self.tree.forward_kinematics(TARGET_NODE, self.configuration).t
        dist = float(np.linalg.norm(target - self.end_effector_pose.t))
        return dist < self.config.target_radius


class VegaMotion3DEnv(ConstantObjectKinDEREnv):
    """Vega motion 3D env with a constant number of objects."""

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricKinematic3Dv2RobotEnv:
        return ObjectCentricVegaMotion3DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        return ["robot", TARGET_NODE]

    def close(self) -> None:
        # Forward to the object-centric env so that its PyBullet clients are released.
        self._object_centric_env.close()

    def _create_env_markdown_description(self) -> str:
        """Create environment description."""
        # pylint: disable=line-too-long
        config = self._object_centric_env.config
        assert isinstance(config, VegaMotion3DEnvConfig)
        lower, upper = config.target_lower_bound, config.target_upper_bound
        return f"""A 3D environment where the goal is to reach a target sphere.

The robot is the {config.manipulator} arm of a bimanual Dexmate Vega 1U, which has 7 degrees of freedom. Vega's remaining joints -- the other arm, the lift, the torso flip, the head, and both grippers -- are held at their home values, so this is a pure arm motion problem. The target is a sphere with radius {config.target_radius:.3f}m positioned randomly within the workspace bounds, rejecting positions the arm cannot reach.

The workspace bounds are:
- X: [{lower[0]:.1f}, {upper[0]:.1f}]
- Y: [{lower[1]:.1f}, {upper[1]:.1f}]
- Z: [{lower[2]:.1f}, {upper[2]:.1f}]
"""

    def _create_variant_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return "This environment has only one variant."

    def _create_reward_markdown_description(self) -> str:
        """Create reward description."""
        # pylint: disable=line-too-long
        config = self._object_centric_env.config
        assert isinstance(config, VegaMotion3DEnvConfig)
        return f"""The reward structure is simple:
- **-1.0** penalty at every timestep until the goal is reached
- **Termination** occurs when the end effector is within {config.target_radius:.3f}m of the target center

This encourages the robot to reach the target as quickly as possible while avoiding infinite episodes.
"""

    def _create_references_markdown_description(self) -> str:
        """Create references description."""
        # pylint: disable=line-too-long
        return """This is a very common kind of environment. The robot is the [Dexmate Vega 1U](https://www.dexmate.ai/), modeled with [prpl_kinematics](https://github.com/Princeton-Robot-Planning-and-Learning/prpl-mono/tree/main/prpl-kinematics)."""
