"""PyBullet environment where an object must be picked from the ground.

There may be other obstructing objects in the environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Type as TypingType

from pybullet_helpers.geometry import set_pose
from pybullet_helpers.utils import create_pybullet_block
from relational_structs import Object, ObjectCentricState
from relational_structs.utils import create_state_from_dict

from kinder.core import ConstantObjectKinDEREnv, FinalConfigMeta
from kinder.envs.kinematic3d.base_env import (
    Kinematic3DEnvConfig,
    ObjectCentricKinematic3DRobotEnv,
)
from kinder.envs.kinematic3d.object_types import (
    Kinematic3DCuboidType,
    Kinematic3DEnvTypeFeatures,
    Kinematic3DRobotType,
)
from kinder.envs.kinematic3d.utils import (
    Kinematic3DObjectCentricState,
    sample_collision_free_object_poses,
)
from kinder.envs.utils import PURPLE


@dataclass(frozen=True)
class Ground3DEnvConfig(Kinematic3DEnvConfig, metaclass=FinalConfigMeta):
    """Config for Ground3DEnv()."""

    # World bounds.
    x_lb: float = -1
    x_ub: float = 1
    y_lb: float = -1
    y_ub: float = 1

    # Blocks.
    block_size: float = 0.02  # cubes (height = width = length)
    block_rgba: tuple[float, float, float, float] = PURPLE + (1.0,)


class Ground3DObjectCentricState(Kinematic3DObjectCentricState):
    """A state in the GroundMotion3DEnv().

    Adds convenience methods on top of Kinematic3DObjectCentricState().
    """


class ObjectCentricGround3DEnv(
    ObjectCentricKinematic3DRobotEnv[Kinematic3DObjectCentricState, Ground3DEnvConfig]
):
    """PyBullet environment where an object must be picked from the ground.

    There may be other obstructing objects in the environment.
    """

    def __init__(
        self,
        num_cubes: int = 2,
        config: Ground3DEnvConfig = Ground3DEnvConfig(),
        **kwargs,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self._num_cubes = num_cubes

        # Create the cubes, but their poses will be reset (with collision checking) in
        # the reset() method.
        self._cubes: dict[str, int] = {}
        for idx in range(self._num_cubes):
            cube_id = create_pybullet_block(
                self.config.block_rgba,
                (
                    self.config.block_size / 2,
                    self.config.block_size / 2,
                    self.config.block_size / 2,
                ),
                physics_client_id=self.physics_client_id,
            )
            self._cubes[f"cube{idx}"] = cube_id

    @property
    def state_cls(self) -> TypingType[Kinematic3DObjectCentricState]:
        return Ground3DObjectCentricState

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        # No constant objects.
        return {}

    def _reset_objects(self) -> None:
        sample_collision_free_object_poses(
            object_ids=set(self._cubes.values()),
            lb=(self.config.x_lb, self.config.y_lb, self.config.block_size / 2),
            ub=(self.config.x_ub, self.config.y_ub, self.config.block_size / 2),
            physics_client_id=self.physics_client_id,
            rng=self.np_random,
            other_collision_ids={self.robot.base.robot_id},
        )

    def _set_object_states(self, obs: Kinematic3DObjectCentricState) -> None:
        assert isinstance(obs, Ground3DObjectCentricState)
        for cube_name, cube_id in self._cubes.items():
            assert cube_id is not None
            set_pose(
                cube_id,
                obs.get_object_pose(cube_name),
                self.physics_client_id,
            )

    def _object_name_to_pybullet_id(self, object_name: str) -> int:
        if object_name.startswith("cube"):
            return self._cubes[object_name]
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_collision_object_ids(self) -> set[int]:
        collision_ids = set(self._cubes.values())
        return collision_ids

    def _get_movable_object_names(self) -> set[str]:
        return set(self._cubes.keys())

    def _get_surface_object_names(self) -> set[str]:
        return set(self._cubes.keys())

    def _get_half_extents(self, object_name: str) -> tuple[float, float, float]:
        if object_name.startswith("cube"):
            return (
                self.config.block_size / 2,
                self.config.block_size / 2,
                self.config.block_size / 2,
            )
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_obs(self) -> Ground3DObjectCentricState:
        state_dict = self._create_state_dict(
            [("robot", Kinematic3DRobotType)]
            + [("cube" + str(i), Kinematic3DCuboidType) for i in range(self._num_cubes)]
        )
        state = create_state_from_dict(
            state_dict, Kinematic3DEnvTypeFeatures, state_cls=Ground3DObjectCentricState
        )
        assert isinstance(state, Ground3DObjectCentricState)
        return state

    def goal_reached(self) -> bool:
        return False


class Ground3DEnv(ConstantObjectKinDEREnv):
    """Ground 3D env with a constant number of objects."""

    def __init__(self, num_cubes: int = 2, **kwargs) -> None:
        self._num_cubes = num_cubes
        super().__init__(num_cubes=num_cubes, **kwargs)

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricKinematic3DRobotEnv:
        return ObjectCentricGround3DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        constant_objects = ["robot"]
        for obj in exemplar_state:
            if obj.name.startswith("cube"):
                constant_objects.append(obj.name)
        return constant_objects

    def _create_env_markdown_description(self) -> str:
        """Create environment description."""
        return (
            """A 3D environment where the goal is to pick up a cube from the ground."""
        )

    def _create_variant_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return "The number of cubes differs between environment variants. For example, Ground3D-o1 has 1 cube, while Ground3D-o3 has 3 cubes."

    def _create_variant_specific_description(self) -> str:
        if self._num_cubes == 1:
            return "This variant has 1 cube on the ground."
        return f"This variant has {self._num_cubes} cubes on the ground."

    def _create_reward_markdown_description(self) -> str:
        """Create reward description."""
        # pylint: disable=line-too-long
        return """The reward is a small negative reward (-0.01) per timestep to encourage exploration."""

    def _create_references_markdown_description(self) -> str:
        """Create references description."""
        # pylint: disable=line-too-long
        return """This is a very common kind of environment."""

    def _create_in_context_examples(self) -> str:
        """Create in-context examples for the environment."""
        return """**Example: Place Cube on Ground Target**

Initial State:
- Robot: base at (0.0, 0.0, 0.0), arm at home
- Cube: position (0.5, 0.0, 0.025)
- Ground target: position (1.0, 1.0), blue circle

Goal: Pick cube and place it on ground target.

Steps: Navigate, grasp cube, transport to target, place on ground.
"""
