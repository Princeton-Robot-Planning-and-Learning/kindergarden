"""PyBullet environment where an object must be picked from the ground and placed on a
shelf.

There may be other obstructing objects in the environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Type as TypingType

import pybullet as p
from pybullet_helpers.geometry import Pose, get_pose, set_pose
from pybullet_helpers.utils import create_pybullet_block, create_pybullet_shelf
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
    Kinematic3DFixtureType,
    Kinematic3DRobotType,
)
from kinder.envs.kinematic3d.utils import (
    Kinematic3DObjectCentricState,
    sample_collision_free_object_poses,
)


@dataclass(frozen=True)
class Shelf3DEnvConfig(Kinematic3DEnvConfig, metaclass=FinalConfigMeta):
    """Config for Shelf3DEnv()."""

    max_action_mag: float = 0.2
    specific_range: bool = False

    # Shelf.
    shelf_pose: Pose = Pose((2.0, 2.4, 0.02))
    if specific_range:
        shelf_pose = Pose((0.8, 0.7, 0.02))
    shelf_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    shelf_width: float = 0.60198
    shelf_depth: float = 0.254
    shelf_height: float = 0.0127
    shelf_spacing: float = 0.254
    shelf_support_width: float = 0.0127
    shelf_num_layers: int = 4
    shelf_texture: Path = Path(__file__).parent / "assets" / "dark-wood-texture.png"

    # World bounds.
    x_lb: float = -1.0
    x_ub: float = 1.0
    y_lb: float = -1.0
    y_ub: float = 1.0
    if specific_range:
        x_lb = 0.4
        x_ub = 0.5
        y_lb = -0.1
        y_ub = 0.1

    # Blocks.
    block_half_extents: tuple[float, float, float] = (0.05, 0.025, 0.025)
    block_rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)

    # Gripper.
    gripper_open_threshold: float = 0.01

    # Goal checking: tolerance below the first shelf layer for determining if an
    # object is "on the shelf". Objects must be above (shelf_pose.z + shelf_spacing
    # - on_shelf_z_tolerance) to count as placed.
    on_shelf_z_tolerance: float = 0.05

    def get_camera_kwargs(self) -> dict[str, Any]:
        """Get kwargs to pass to PyBullet camera."""
        return {
            "camera_target": (0, 0, 0),
            "camera_yaw": 0,
            "camera_distance": 2.0,
            "camera_pitch": -20,
        }

    def get_cube_texture(self, idx: int) -> Path:
        """Get a texture to wrap a cube given the index."""
        asset_dir = Path(__file__).parent / "assets"
        texture_filenames = [f"book{i}.jpg" for i in range(5)]
        texture_filename = texture_filenames[idx % len(texture_filenames)]
        return asset_dir / texture_filename


class Shelf3DObjectCentricState(Kinematic3DObjectCentricState):
    """A state in the Shelf3DEnv().

    Adds convenience methods on top of Kinematic3DObjectCentricState().
    """


class ObjectCentricShelf3DEnv(
    ObjectCentricKinematic3DRobotEnv[Kinematic3DObjectCentricState, Shelf3DEnvConfig]
):
    """PyBullet environment where objects must be picked from the ground and placed on a
    shelf."""

    def __init__(
        self,
        num_cubes: int = 2,
        config: Shelf3DEnvConfig = Shelf3DEnvConfig(),
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
                    self.config.block_half_extents[0],
                    self.config.block_half_extents[1],
                    self.config.block_half_extents[2],
                ),
                physics_client_id=self.physics_client_id,
            )
            self._cubes[f"cube{idx}"] = cube_id
            cube_texture_id = p.loadTexture(
                str(self.config.get_cube_texture(idx)), self.physics_client_id
            )
            p.changeVisualShape(
                cube_id,
                -1,
                textureUniqueId=cube_texture_id,
                physicsClientId=self.physics_client_id,
            )

        # Create shelf.
        self._shelf_id, self._shelf_surface_ids = create_pybullet_shelf(
            color=self.config.shelf_rgba,
            shelf_width=self.config.shelf_width,
            shelf_depth=self.config.shelf_depth,
            shelf_height=self.config.shelf_height,
            spacing=self.config.shelf_spacing,
            support_width=self.config.shelf_support_width,
            num_layers=self.config.shelf_num_layers,
            physics_client_id=self.physics_client_id,
        )
        set_pose(self._shelf_id, self.config.shelf_pose, self.physics_client_id)

        # NOTE: use this for repositioning the shelf visually (with GUI on).
        # from pybullet_helpers.gui import interactively_visualize_pose
        # interactively_visualize_pose(
        #     self.config.shelf_pose,
        #     self.physics_client_id,
        #     min_position=-10,
        #     max_position=10,
        #     object_id=self._shelf_id,
        # )

        shelf_texture_id = p.loadTexture(
            str(self.config.shelf_texture), self.physics_client_id
        )
        for shelf_link_id in range(
            p.getNumJoints(self._shelf_id, physicsClientId=self.physics_client_id)
        ):
            p.changeVisualShape(
                self._shelf_id,
                shelf_link_id,
                textureUniqueId=shelf_texture_id,
                physicsClientId=self.physics_client_id,
            )

    @property
    def state_cls(self) -> TypingType[Kinematic3DObjectCentricState]:
        return Shelf3DObjectCentricState

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        return self._create_state_dict([("shelf", Kinematic3DFixtureType)])

    def _reset_objects(self) -> None:
        sample_collision_free_object_poses(
            object_ids=set(self._cubes.values()),
            lb=(self.config.x_lb, self.config.y_lb, self.config.block_half_extents[2]),
            ub=(self.config.x_ub, self.config.y_ub, self.config.block_half_extents[2]),
            physics_client_id=self.physics_client_id,
            rng=self.np_random,
            other_collision_ids={self.robot.base.robot_id},
        )

    def _set_object_states(self, obs: Kinematic3DObjectCentricState) -> None:
        assert isinstance(obs, Shelf3DObjectCentricState)
        for cube_name, cube_id in self._cubes.items():
            assert cube_id is not None
            set_pose(
                cube_id,
                obs.get_object_pose(cube_name),
                self.physics_client_id,
            )

    def _object_name_to_pybullet_id(self, object_name: str) -> int:
        if object_name == "shelf":
            return self._shelf_id
        if object_name.startswith("cube"):
            return self._cubes[object_name]
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_collision_object_ids(self) -> set[int]:
        collision_ids = {self._shelf_id} | set(self._cubes.values())
        return collision_ids

    def _get_movable_object_names(self) -> set[str]:
        return set(self._cubes.keys())

    def _get_surface_object_names(self) -> set[str]:
        return {"shelf"}

    def _get_half_extents(self, object_name: str) -> tuple[float, float, float]:
        if object_name.startswith("cube"):
            return self.config.block_half_extents
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_obs(self) -> Shelf3DObjectCentricState:
        state_dict = self._create_state_dict(
            [("robot", Kinematic3DRobotType)]
            + [("shelf", Kinematic3DFixtureType)]
            + [("cube" + str(i), Kinematic3DCuboidType) for i in range(self._num_cubes)]
        )
        state = create_state_from_dict(
            state_dict, Kinematic3DEnvTypeFeatures, state_cls=Shelf3DObjectCentricState
        )
        assert isinstance(state, Shelf3DObjectCentricState)
        return state

    def goal_reached(self) -> bool:
        robot_gripper_pose = self._robot_arm.get_finger_state()
        if robot_gripper_pose > self.config.gripper_open_threshold:
            return False
        # Check that all cubes are above the first shelf layer (with tolerance).
        min_on_shelf_z = (
            self.config.shelf_pose.position[2]
            + self.config.shelf_spacing
            - self.config.on_shelf_z_tolerance
        )
        for _, cube_id in self._cubes.items():
            cube_pose = get_pose(cube_id, self.physics_client_id)
            if cube_pose.position[2] < min_on_shelf_z:
                return False

        return True


class Shelf3DEnv(ConstantObjectKinDEREnv):
    """Table 3D env with a constant number of objects."""

    def __init__(self, num_cubes: int = 2, **kwargs) -> None:
        self._num_cubes = num_cubes
        super().__init__(num_cubes=num_cubes, **kwargs)

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricKinematic3DRobotEnv:
        return ObjectCentricShelf3DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        constant_objects = ["robot", "shelf"]
        for obj in exemplar_state:
            if obj.name.startswith("cube"):
                constant_objects.append(obj.name)
        return constant_objects

    def _create_env_markdown_description(self) -> str:
        """Create environment description."""
        # pylint: disable=line-too-long
        return """A 3D environment where the goal is to pick up objects from the ground and place them onto a shelf."""

    def _create_variant_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return "The number of objects differs between environment variants. For example, Shelf3D-o1 has 1 object, while Shelf3D-o10 has 10 objects."

    def _create_variant_specific_description(self) -> str:
        if self._num_cubes == 1:
            return "This variant has 1 object to place on the shelf."
        return f"This variant has {self._num_cubes} objects to place on the shelf."

    def _create_reward_markdown_description(self) -> str:
        """Create reward description."""
        # pylint: disable=line-too-long
        return """The reward is -1 per timestep to encourage efficient task completion. The episode terminates successfully when all objects are placed on the shelf (i.e., above the first shelf layer) and the gripper is closed. The gripper must be closed to prevent accidental "success" while an object is still being held above the shelf."""

    def _create_references_markdown_description(self) -> str:
        """Create references description."""
        # pylint: disable=line-too-long
        return """This is a very common kind of environment. The background is adapted from the [Replica dataset](https://arxiv.org/abs/1906.05797) (Straub et al., 2019)."""

    def _create_in_context_examples(self) -> str:
        """Create in-context examples for the environment."""
        return """**Example: Place Objects on Shelf**

Initial State:
- Robot: base at (0.0, 0.0, 0.0)
- Shelf: multiple levels at different heights
- Objects: cubes on ground

Goal: Pick objects and place them on shelf levels.

Steps: Approach, grasp, transport, place on shelf, release.
"""
