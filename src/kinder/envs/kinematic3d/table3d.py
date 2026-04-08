"""PyBullet environment where an object must be picked from the table.

There may be other obstructing objects in the environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Type as TypingType

import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose, set_pose
from pybullet_helpers.inverse_kinematics import check_body_collisions
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
from kinder.envs.kinematic3d.utils import Kinematic3DObjectCentricState
from kinder.envs.utils import PURPLE


@dataclass(frozen=True)
class Table3DEnvConfig(Kinematic3DEnvConfig, metaclass=FinalConfigMeta):
    """Config for Table3DEnv()."""

    # Table.
    table_pose: Pose = Pose((0.6, 0.0, 0.2))
    table_rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    table_half_extents: tuple[float, float, float] = (0.2, 0.4, 0.2)
    table_texture: Path = (
        Path(__file__).parent / "assets" / "use_textures" / "light_wood_v3.png"
    )

    # World bounds.
    x_lb: float = -1
    x_ub: float = 1
    y_lb: float = -1
    y_ub: float = 1

    # Blocks.
    block_size: float = 0.05  # cubes (height = width = length)
    block_rgba: tuple[float, float, float, float] = PURPLE + (1.0,)

    def get_camera_kwargs(self) -> dict[str, Any]:
        """Get kwargs to pass to PyBullet camera."""
        return {
            "camera_target": (0, 0, 0),
            "camera_yaw": 90,
            "camera_distance": 2.0,
            "camera_pitch": -20,
        }

    def sample_block_on_table_pose(
        self, block_half_extents: tuple[float, float, float], rng: np.random.Generator
    ) -> Pose:
        """Sample an initial block pose given sampled half extents."""

        return self._sample_block_on_block_pose(
            block_half_extents, self.table_half_extents, self.table_pose, rng
        )


class Table3DObjectCentricState(Kinematic3DObjectCentricState):
    """A state in the Table3DEnv().

    Adds convenience methods on top of Kinematic3DObjectCentricState().
    """

    def get_cuboid_half_extents(self, name: str) -> tuple[float, float, float]:
        """The half extents of the cuboid."""
        obj = self.get_object_from_name(name)
        return (
            self.get(obj, "half_extent_x"),
            self.get(obj, "half_extent_y"),
            self.get(obj, "half_extent_z"),
        )

    def get_cuboid_pose(self, name: str) -> Pose:
        """The pose of the cuboid."""
        obj = self.get_object_from_name(name)
        position = (
            self.get(obj, "pose_x"),
            self.get(obj, "pose_y"),
            self.get(obj, "pose_z"),
        )
        orientation = (
            self.get(obj, "pose_qx"),
            self.get(obj, "pose_qy"),
            self.get(obj, "pose_qz"),
            self.get(obj, "pose_qw"),
        )
        return Pose(position, orientation)


class ObjectCentricTable3DEnv(
    ObjectCentricKinematic3DRobotEnv[Kinematic3DObjectCentricState, Table3DEnvConfig]
):
    """PyBullet environment where an object must be picked from the table.

    There may be other obstructing objects in the environment.
    """

    def __init__(
        self,
        num_cubes: int = 2,
        config: Table3DEnvConfig = Table3DEnvConfig(),
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

        # Create table.
        self.table_id = create_pybullet_block(
            self.config.table_rgba,
            half_extents=self.config.table_half_extents,
            physics_client_id=self.physics_client_id,
        )
        table_texture_id = p.loadTexture(
            str(self.config.table_texture), self.physics_client_id
        )
        p.changeVisualShape(
            self.table_id,
            -1,
            textureUniqueId=table_texture_id,
            physicsClientId=self.physics_client_id,
        )
        set_pose(self.table_id, self.config.table_pose, self.physics_client_id)

    @property
    def state_cls(self) -> TypingType[Kinematic3DObjectCentricState]:
        return Table3DObjectCentricState

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        return self._create_state_dict([("table", Kinematic3DCuboidType)])

    def _reset_objects(self) -> None:
        # Randomly sample collision-free positions for the cubes.
        # Also ensure that they are not in collision with the robot.
        # Samples the poses of the cubes
        for _ in range(100_000):
            for cube_name, cube_id in self._cubes.items():
                cube_half_extents = (
                    self.config.block_size / 2,
                    self.config.block_size / 2,
                    self.config.block_size / 2,
                )
                # add orientation later
                cube_pose = self.config.sample_block_on_table_pose(
                    cube_half_extents, self.np_random
                )
                set_pose(cube_id, cube_pose, self.physics_client_id)
            collision_free = True
            for cube_name, cube_id in self._cubes.items():
                for other_cube_name, other_cube_id in self._cubes.items():
                    if cube_name == other_cube_name:
                        continue
                    if check_body_collisions(
                        cube_id,
                        other_cube_id,
                        self.physics_client_id,
                    ):
                        collision_free = False
                        break

            if collision_free:
                break

        else:
            raise RuntimeError("Failed to sample collision-free cube poses")

    def _set_object_states(self, obs: Kinematic3DObjectCentricState) -> None:
        assert isinstance(obs, Table3DObjectCentricState)
        for cube_name, cube_id in self._cubes.items():
            assert cube_id is not None
            set_pose(
                cube_id,
                obs.get_object_pose(cube_name),
                self.physics_client_id,
            )

    def _object_name_to_pybullet_id(self, object_name: str) -> int:
        if object_name == "table":
            return self.table_id
        if object_name.startswith("cube"):
            return self._cubes[object_name]
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_collision_object_ids(self) -> set[int]:
        collision_ids = {self.table_id} | set(self._cubes.values())
        return collision_ids

    def _get_movable_object_names(self) -> set[str]:
        return set(self._cubes.keys())

    def _get_surface_object_names(self) -> set[str]:
        return {"table"}

    def _get_half_extents(self, object_name: str) -> tuple[float, float, float]:
        if object_name.startswith("cube"):
            return (
                self.config.block_size / 2,
                self.config.block_size / 2,
                self.config.block_size / 2,
            )
        if object_name == "table":
            return self.config.table_half_extents
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_obs(self) -> Table3DObjectCentricState:
        state_dict = self._create_state_dict(
            [("robot", Kinematic3DRobotType)]
            + [("table", Kinematic3DCuboidType)]
            + [("cube" + str(i), Kinematic3DCuboidType) for i in range(self._num_cubes)]
        )
        state = create_state_from_dict(
            state_dict, Kinematic3DEnvTypeFeatures, state_cls=Table3DObjectCentricState
        )
        assert isinstance(state, Table3DObjectCentricState)
        return state

    def goal_reached(self) -> bool:
        return False


class Table3DEnv(ConstantObjectKinDEREnv):
    """Table 3D env with a constant number of objects."""

    def __init__(self, num_cubes: int = 2, **kwargs) -> None:
        self._num_cubes = num_cubes
        super().__init__(num_cubes=num_cubes, **kwargs)

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricKinematic3DRobotEnv:
        return ObjectCentricTable3DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        constant_objects = ["robot", "table"]
        for obj in exemplar_state:
            if obj.name.startswith("cube"):
                constant_objects.append(obj.name)
        return constant_objects

    def _create_env_markdown_description(self) -> str:
        """Create environment description."""
        return (
            """A 3D environment where the goal is to pick up a cube from the table."""
        )

    def _create_variant_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return "The number of cubes differs between environment variants. For example, Table3D-o1 has 1 cube, while Table3D-o3 has 3 cubes."

    def _create_variant_specific_description(self) -> str:
        if self._num_cubes == 1:
            return "This variant has 1 cube on the table."
        return f"This variant has {self._num_cubes} cubes on the table."

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
        return """**Example: Place Cubes on Table Surface**

Initial State:
- Robot: base at (0.0, 0.0, 0.0)
- Table: rectangular surface at (0.6, 0.0, 0.2)
- Cubes: multiple cubes on ground surface

Goal: Pick cubes and place them on table surface.

Steps: Navigate, approach, grasp, transport, place on table.
"""
