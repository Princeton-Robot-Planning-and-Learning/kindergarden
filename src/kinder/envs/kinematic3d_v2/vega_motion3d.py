"""Environment where the goal is to reach a target region with a Dexmate Vega arm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Type as TypingType

import numpy as np
from numpy.typing import NDArray
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
    # Targets sit at the end-effector position of an arm configuration sampled within
    # this per-joint window (radians) around home, so every target admits a reaching
    # configuration at most this far from home on each joint. At 1.0 the sampled
    # positions still cover the full extent of the bounds above.
    target_witness_joint_delta: float = 1.0


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
        # The arm configuration whose end effector the current target was placed at,
        # recorded when the target is sampled. It certifies that the target is
        # reachable from home within the witness joint window; it is not a goal (the
        # goal is position-only).
        self._target_witness_joints: NDArray[np.float64] | None = None

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
        # Reset the target by sampling arm configurations near home and placing the
        # target at the end-effector position of one that lands inside the target
        # bounds, so every target is reachable by construction and no inverse
        # kinematics is needed. Generating targets the other way around -- sampling a
        # position and asking IK whether it is reachable -- requires picking an
        # end-effector orientation for the IK query, and a poor choice over-constrains
        # the reach (see issue #150). Sampling near home rather than over the whole
        # joint space keeps the demanded motion moderate; the already-reached check
        # below keeps it nonzero. A few percent of sampled configurations land in the
        # bounds, and each attempt is one forward-kinematics call, so the retry budget
        # is ample.
        delta = self.config.target_witness_joint_delta
        sample_lower = np.clip(
            self._home_arm_joint_positions - delta,
            self._joint_lower_limits,
            self._joint_upper_limits,
        )
        sample_upper = np.clip(
            self._home_arm_joint_positions + delta,
            self._joint_lower_limits,
            self._joint_upper_limits,
        )
        for _ in range(10_000):
            joints = self.np_random.uniform(sample_lower, sample_upper)
            config = {**self.configuration, **self._arm_space.to_configuration(joints)}
            position = self.tree.forward_kinematics(
                self._manipulator.ee_frame, config
            ).t
            if np.any(position < self.config.target_lower_bound) or np.any(
                position > self.config.target_upper_bound
            ):
                continue
            if self._collision_checker.in_collision(config):
                continue
            self._set_target_position(tuple(position))
            # If the goal is already reached, keep sampling.
            if not self.goal_reached():
                self._target_witness_joints = joints
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

The robot is the {config.manipulator} arm of a bimanual Dexmate Vega 1U, which has 7 degrees of freedom. Vega's remaining joints -- the other arm, the lift, the torso flip, the head, and both grippers -- are held at their home values, so this is a pure arm motion problem. The target is a sphere with radius {config.target_radius:.3f}m placed at the end-effector position of a collision-free arm configuration sampled within {config.target_witness_joint_delta:.1f} rad per joint of the home configuration, restricted to the workspace bounds. Every target is therefore reachable by construction, with a moderate motion from home.

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
