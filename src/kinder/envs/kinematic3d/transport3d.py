"""PyBullet environment where cubes and boxes must be transported onto a table."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Type as TypingType

import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose, get_pose, set_pose
from pybullet_helpers.utils import create_pybullet_block, create_pybullet_hollow_box
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


@dataclass(frozen=True)
class Transport3DEnvConfig(Kinematic3DEnvConfig, metaclass=FinalConfigMeta):
    """Config for Transport3DEnv()."""

    max_action_mag: float = 0.2

    # Table.
    table_pose: Pose = Pose((0.6, 0.0, 0.2))
    table_rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    table_half_extents: tuple[float, float, float] = (0.2, 0.4, 0.2)
    table_texture: Path = (
        Path(__file__).parent / "assets" / "use_textures" / "light_wood_v3.png"
    )

    # Minimum distance between objects for placement.
    min_placement_dist: float = 0.01

    # Maximum sampling attempts for placing blocks on ground.
    max_ground_sampling_attempts: int = 100

    # Blocks.
    block_size: float = 0.05  # cubes (height = width = length)
    block_rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    block_texture_1: Path = (
        Path(__file__).parent / "assets" / "use_textures" / "yellow-grid.png"
    )
    block_texture_2: Path = (
        Path(__file__).parent / "assets" / "use_textures" / "metal.png"
    )

    # Box.
    box_half_extents: tuple[float, float, float] = (0.1, 0.15, 0.1)
    box_rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    box_wall_thickness: float = 0.01
    box_texture: Path = (
        Path(__file__).parent / "assets" / "use_textures" / "blue-flower.png"
    )

    # Floor.
    floor_included_as_object: bool = True

    # Gripper.
    gripper_open_threshold: float = 0.01

    # Goal thresholds.
    goal_height_threshold: float = 0.3
    goal_distance_threshold: float = 0.2

    def get_camera_kwargs(self) -> dict[str, Any]:
        """Get kwargs to pass to PyBullet camera."""
        return {
            "camera_target": (0, 0, 0),
            "camera_yaw": 90,
            "camera_distance": 2.0,
            "camera_pitch": -40,
        }

    def sample_block_on_table_pose(
        self, block_half_extents: tuple[float, float, float], rng: np.random.Generator
    ) -> Pose:
        """Sample an initial block pose given sampled half extents."""

        return self._sample_block_on_block_pose(
            block_half_extents, self.table_half_extents, self.table_pose, rng
        )

    def sample_block_on_ground(
        self, block_half_extents: tuple[float, float, float], rng: np.random.Generator
    ) -> Pose:
        """Sample an initial block pose given sampled half extents."""

        lb = (
            self.x_lb,
            self.y_lb,
            block_half_extents[2],
        )

        ub = (
            self.x_ub,
            self.y_ub,
            block_half_extents[2],
        )

        for _ in range(self.max_ground_sampling_attempts):
            x, y, z = rng.uniform(lb, ub)
            if (
                np.abs(x - self.table_pose.position[0]) > self.table_half_extents[0]
                and np.abs(y - self.table_pose.position[1]) > self.table_half_extents[1]
            ):
                break
        else:
            raise RuntimeError(
                f"Failed to sample collision-free block pose on ground after "
                f"{self.max_ground_sampling_attempts} attempts"
            )

        return Pose((x, y, z))

    def sample_block_in_box_pose(
        self,
        block_half_extents: tuple[float, float, float],
        box_pose: Pose,
        box_half_extents: tuple[float, float, float],
        box_wall_thickness: float,
        rng: np.random.Generator,
    ) -> Pose:
        """Sample an initial block pose given sampled half extents."""

        assert np.allclose(box_pose.orientation, (0, 0, 0, 1)), "Not implemented"

        lb = (
            box_pose.position[0] - box_half_extents[0] + block_half_extents[0],
            box_pose.position[1] - box_half_extents[1] + block_half_extents[1],
            box_pose.position[2] + block_half_extents[2] + box_wall_thickness,
        )

        ub = (
            box_pose.position[0] + box_half_extents[0] - block_half_extents[0],
            box_pose.position[1] + box_half_extents[1] - block_half_extents[1],
            box_pose.position[2] + block_half_extents[2] + box_wall_thickness,
        )

        x, y, z = rng.uniform(lb, ub)

        return Pose((x, y, z))

    def get_cube_texture(self, idx: int) -> Path:
        """Get a texture to wrap a cube given the index.

        Cycles through available textures if idx exceeds the number of textures.
        """
        textures = [self.block_texture_1, self.block_texture_2]
        return textures[idx % len(textures)]


class Transport3DObjectCentricState(Kinematic3DObjectCentricState):
    """A state in the Transport3DEnv()."""


class ObjectCentricTransport3DEnv(
    ObjectCentricKinematic3DRobotEnv[
        Kinematic3DObjectCentricState, Transport3DEnvConfig
    ]
):
    """PyBullet environment where cubes and boxes must be transported onto a table."""

    def __init__(
        self,
        num_cubes: int = 2,
        num_boxes: int = 1,
        config: Transport3DEnvConfig = Transport3DEnvConfig(),
        **kwargs,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self._num_cubes = num_cubes
        self._num_boxes = num_boxes

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
            cube_texture_id = p.loadTexture(
                str(self.config.get_cube_texture(idx)), self.physics_client_id
            )
            p.changeVisualShape(
                cube_id,
                -1,
                textureUniqueId=cube_texture_id,
                physicsClientId=self.physics_client_id,
            )

        # Create the boxes, but their poses will be reset (with collision checking) in
        # the reset() method.
        self._boxes: dict[str, int] = {}
        for idx in range(self._num_boxes):
            box_id = create_pybullet_hollow_box(
                self.config.box_rgba,
                self.config.box_half_extents,
                self.config.box_wall_thickness,
                physics_client_id=self.physics_client_id,
            )
            self._boxes[f"box{idx}"] = box_id
            box_texture_id = p.loadTexture(
                str(self.config.box_texture), self.physics_client_id
            )
            for box_link_id in range(
                p.getNumJoints(box_id, physicsClientId=self.physics_client_id)
            ):
                p.changeVisualShape(
                    box_id,
                    box_link_id,
                    textureUniqueId=box_texture_id,
                    physicsClientId=self.physics_client_id,
                )

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
        return Transport3DObjectCentricState

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        return self._create_state_dict([("table", Kinematic3DCuboidType)])

    def _reset_objects(self) -> None:
        # Randomly sample collision-free positions for the cubes.
        # Also ensure that they are not in collision with the robot.
        # Samples the poses of the cubes
        box_ids = set(self._boxes.values())
        sample_collision_free_object_poses(
            object_ids=box_ids,
            table_pose=self.config.table_pose,
            table_half_extents=self.config.table_half_extents,
            lb=(
                self.config.x_lb,
                self.config.y_lb,
                self.config.box_half_extents[2] + self.config.floor_z,
            ),
            ub=(
                self.config.x_ub,
                self.config.y_ub,
                self.config.box_half_extents[2] + self.config.floor_z,
            ),
            physics_client_id=self.physics_client_id,
            rng=self.np_random,
            other_collision_ids={self.robot.base.robot_id},
        )

        sample_collision_free_object_poses(
            use_box=True,
            box_pose=get_pose(self._boxes["box0"], self.physics_client_id),
            table_pose=self.config.table_pose,
            table_half_extents=self.config.table_half_extents,
            box_half_extents=self.config.box_half_extents,
            object_ids=set(self._cubes.values()),
            lb=(
                self.config.x_lb,
                self.config.y_lb,
                self.config.block_size / 2 + self.config.floor_z,
            ),
            ub=(
                self.config.x_ub,
                self.config.y_ub,
                self.config.block_size / 2 + self.config.floor_z,
            ),
            physics_client_id=self.physics_client_id,
            rng=self.np_random,
            other_collision_ids=box_ids | {self.robot.base.robot_id},
        )

    def _set_object_states(self, obs: Kinematic3DObjectCentricState) -> None:
        assert isinstance(obs, Transport3DObjectCentricState)
        for cube_name, cube_id in self._cubes.items():
            assert cube_id is not None
            set_pose(
                cube_id,
                obs.get_object_pose(cube_name),
                self.physics_client_id,
            )

        for box_name, box_id in self._boxes.items():
            assert box_id is not None
            set_pose(
                box_id,
                obs.get_object_pose(box_name),
                self.physics_client_id,
            )

    def _object_name_to_pybullet_id(self, object_name: str) -> int:
        if object_name == "table":
            return self.table_id
        if object_name.startswith("cube"):
            return self._cubes[object_name]
        if object_name.startswith("box"):
            return self._boxes[object_name]
        if object_name.startswith("floor"):
            assert self.config.floor_included_as_object
            return self.floor_id
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_collision_object_ids(self) -> set[int]:
        collision_ids = (
            {self.table_id} | set(self._cubes.values()) | set(self._boxes.values())
        )
        if self.config.floor_included_as_object:
            collision_ids.add(self.floor_id)
        return collision_ids

    def _get_movable_object_names(self) -> set[str]:
        return set(self._cubes.keys()) | set(self._boxes.keys())

    def _get_surface_object_names(self) -> set[str]:
        surfaces = {"table"} | set(self._boxes.keys())
        if self.config.floor_included_as_object:
            surfaces.add("floor")
        return surfaces

    def _get_half_extents(self, object_name: str) -> tuple[float, float, float]:
        if object_name.startswith("cube"):
            return (
                self.config.block_size / 2,
                self.config.block_size / 2,
                self.config.block_size / 2,
            )
        if object_name.startswith("box"):
            return (
                self.config.box_half_extents[0],
                self.config.box_half_extents[1],
                self.config.box_half_extents[2],
            )
        if object_name == "table":
            return self.config.table_half_extents
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_obs(self) -> Transport3DObjectCentricState:
        state_dict = self._create_state_dict(
            [("robot", Kinematic3DRobotType)]
            + [("table", Kinematic3DCuboidType)]
            + [("cube" + str(i), Kinematic3DCuboidType) for i in range(self._num_cubes)]
            + [("box" + str(i), Kinematic3DCuboidType) for i in range(self._num_boxes)]
        )
        state = create_state_from_dict(
            state_dict,
            Kinematic3DEnvTypeFeatures,
            state_cls=Transport3DObjectCentricState,
        )
        assert isinstance(state, Transport3DObjectCentricState)
        return state

    def goal_reached(self) -> bool:
        robot_gripper_pose = self._robot_arm.get_finger_state()
        robot_end_effector_pose = self._robot_arm.get_end_effector_pose()
        if robot_gripper_pose > self.config.gripper_open_threshold:
            return False
        for _, cube_id in self._cubes.items():
            cube_pose = get_pose(cube_id, self.physics_client_id)
            if (
                np.linalg.norm(
                    np.subtract(robot_end_effector_pose.position, cube_pose.position)
                )
                < self.config.goal_distance_threshold
            ):
                return False
            if cube_pose.position[2] < self.config.goal_height_threshold:
                return False
        for _, box_id in self._boxes.items():
            box_pose = get_pose(box_id, self.physics_client_id)
            if (
                np.linalg.norm(
                    np.subtract(robot_end_effector_pose.position, box_pose.position)
                )
                < self.config.goal_distance_threshold
            ):
                return False
            if box_pose.position[2] < self.config.goal_height_threshold:
                return False
        return True


class Transport3DEnv(ConstantObjectKinDEREnv):
    """Table Box 3D env with a constant number of objects."""

    def __init__(self, num_cubes: int = 2, num_boxes: int = 1, **kwargs) -> None:
        self._num_cubes = num_cubes
        self._num_boxes = num_boxes
        super().__init__(num_cubes=num_cubes, num_boxes=num_boxes, **kwargs)

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricKinematic3DRobotEnv:
        return ObjectCentricTransport3DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        constant_objects = ["robot", "table"]
        for obj in exemplar_state:
            if obj.name.startswith("cube"):
                constant_objects.append(obj.name)
            if obj.name.startswith("box"):
                constant_objects.append(obj.name)
        return constant_objects

    def _create_env_markdown_description(self) -> str:
        """Create environment description."""
        # pylint: disable=line-too-long
        return """A 3D environment where the goal is to place all objects, including one or more solid cubes and a box, on a table."""

    def _create_variant_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return "The number of cubes differs between environment variants. For example, Transport3D-o1 has 1 cube, while Transport3D-o2 has 2 cubes."

    def _create_variant_specific_description(self) -> str:
        cube_str = "1 cube" if self._num_cubes == 1 else f"{self._num_cubes} cubes"
        box_str = "1 box" if self._num_boxes == 1 else f"{self._num_boxes} boxes"
        return f"This variant has {cube_str} and {box_str} to transport onto the table."

    def _create_in_context_examples(self) -> str:
        """Create concrete examples showing strategies for solving the Transport3D task.

        Returns formatted examples with actual object positions and action sequences.
        """
        # pylint: disable=line-too-long
        return """
**Example 1: Two Cubes and One Box - Sequential Transport**

Initial State:
- Robot: base at (0.0, 0.0, 0.0), gripper open
- Table: center at (0.6, 0.0, 0.2), dimensions 0.4m × 0.8m × 0.4m, surface at z=0.4m
- Cube1: position (0.067, -0.673, 0.025) on ground, size 0.05m × 0.05m × 0.05m
- Cube0: position (0.879, -1.419, 0.025) on ground, size 0.05m × 0.05m × 0.05m
- Box0: position (0.355, -0.803, 0.1) on ground, dimensions 0.2m × 0.3m × 0.2m

Goal: Place all objects (cube1, cube0, box0) on the table (height > 0.3m, not held by robot)

Strategy: Prioritize objects by accessibility → Transport each sequentially → Place at different table locations

High-Level Plan:
1. pick(robot, cube1, grasp_offset=[0.55, 0.0])
   - Navigate robot to cube1 at (0.067, -0.673, 0.025)
   - Approach with base offset ensuring end effector can reach cube from position offset by (0.55, 0.0) relative to object
   - Lower gripper to z=0.025 (cube center height)
   - Close gripper to grasp cube1

2. place(robot, cube1, table, place_offset=[-0.10, -0.15])
   - Transport cube1 to table center (0.6, 0.0, 0.4)
   - Position cube at offset (-0.10, -0.15) from table center → final position (0.50, -0.15, 0.425)
   - Lower until cube contacts table surface
   - Open gripper to release cube1
   - Retreat to distance > 0.2m from cube

3. pick(robot, cube0, grasp_offset=[0.56, 0.2])
   - Navigate to cube0 at (0.879, -1.419, 0.025)
   - Approach with base offset (0.56, 0.2) for optimal reach angle
   - Lower and grasp cube0

4. place(robot, cube0, table, place_offset=[0.10, -0.15])
   - Transport cube0 to table
   - Place at offset (0.10, -0.15) from table center → final position (0.70, -0.15, 0.425)
   - Position next to cube1 but with separation to avoid collision
   - Release and retreat

5. pick(robot, box0, grasp_offset=[0.58, -0.2])
   - Navigate to box0 at (0.355, -0.803, 0.1)
   - Approach with base offset (0.58, -0.2)
   - Lower gripper to z=0.1 (box center height)
   - Grasp box from above/sides

6. place(robot, box0, table, place_offset=[0.00, 0.15])
   - Transport box0 to table
   - Place at offset (0.00, 0.15) from table center → final position (0.60, 0.15, 0.5)
   - Position box in available space away from cubes
   - Release and retreat

Goal Reached:
- Cube1 at (0.50, -0.15, 0.425): z=0.425m > 0.3m ✓
- Cube0 at (0.70, -0.15, 0.425): z=0.425m > 0.3m ✓
- Box0 at (0.60, 0.15, 0.5): z=0.5m > 0.3m ✓
- All objects on table, gripper open, not held ✓

Key Insights:
- Grasp offsets [dx, dy] determine robot base positioning relative to target object for optimal reachability
- Place offsets [dx, dy] specify object placement location relative to surface center
- Sequential transport: closest/easiest objects first, then farther objects
- Spatial planning: distribute objects across table surface to avoid collisions
- Table dimensions (0.4m × 0.8m) provide sufficient space for multiple objects

**Example 2: Two Cubes and One Box - Different Configuration**

Initial State:
- Robot: base at (0.0, 0.0, 0.0), gripper open
- Table: center at (0.6, 0.0, 0.2), surface at z=0.4m
- Cube0: position (-0.124, -0.684, 0.025) on ground, size 0.05m × 0.05m × 0.05m
- Cube1: position (-0.258, -1.148, 0.025) on ground, size 0.05m × 0.05m × 0.05m
- Box0: position (0.394, -1.115, 0.1) on ground, dimensions 0.2m × 0.3m × 0.2m

Goal: Place all objects on table

Strategy: Handle cubes first with consistent offsets → Transport box last

High-Level Plan:
1. pick(robot, cube0, grasp_offset=[0.55, 0.0])
   - Navigate to cube0 at (-0.124, -0.684, 0.025)
   - Standard grasp offset (0.55, 0.0) for straight approach
   - Grasp cube0

2. place(robot, cube0, table, place_offset=[-0.10, -0.15])
   - Transport to table
   - Place at offset (-0.10, -0.15) → position (0.50, -0.15, 0.425)
   - This is the same placement location as cube1 in Example 1 (reusable placement strategy)
   - Release and retreat

3. pick(robot, cube1, grasp_offset=[0.55, 0.0])
   - Navigate to cube1 at (-0.258, -1.148, 0.025)
   - Use same grasp offset (0.55, 0.0) as cube0 for consistency
   - Grasp cube1

4. place(robot, cube1, table, place_offset=[-0.10, 0.15])
   - Transport to table
   - Place at offset (-0.10, 0.15) → position (0.50, 0.15, 0.425)
   - Mirror y-position of cube0 placement (0.15 vs -0.15) for symmetric distribution
   - Release and retreat

5. pick(robot, box0, grasp_offset=[0.58, 0.0])
   - Navigate to box0 at (0.394, -1.115, 0.1)
   - Slightly larger grasp offset (0.58 vs 0.55) to account for box size
   - Grasp box

6. place(robot, box0, table, place_offset=[0.10, 0.00])
   - Transport to table
   - Place at offset (0.10, 0.00) → position (0.70, 0.00, 0.5)
   - Center box in y-direction, offset in x to avoid cubes
   - Release and retreat

Goal Reached:
- Cube0 at (0.50, -0.15, 0.425): z=0.425m > 0.3m ✓
- Cube1 at (0.50, 0.15, 0.425): z=0.425m > 0.3m ✓
- Box0 at (0.70, 0.00, 0.5): z=0.5m > 0.3m ✓

Key Insights:
- Consistent grasp offsets: similar objects (cubes) use same offset (0.55, 0.0)
- Symmetric placement: cubes placed with mirrored y-offsets (-0.15, 0.15) for spatial distribution
- Box placement: centered at y=0.0 with positive x-offset to use remaining table space
- Offset patterns can be reused across different initial configurations
- Small differences in grasp offset (0.55 vs 0.58) accommodate different object sizes

**General Offset Interpretation:**
- pick(grasp_offset=[dx, dy]): robot base positions at (object_x - dx, object_y - dy) to reach object with end effector
- place(place_offset=[dx, dy]): object final position is (surface_x + dx, surface_y + dy, surface_z + object_height/2)
- Positive grasp offset dx: robot approaches from behind object (negative x direction)
- Grasp offset magnitude (0.55-0.58): typical reach distance for this robot configuration
- Place offsets distribute objects: negative x for left side, positive x for right side, varied y for front/back
"""

    def _create_reward_markdown_description(self) -> str:
        """Create reward description."""
        # pylint: disable=line-too-long
        return """The reward is a small negative reward (-1) per timestep until termination, which occurs when all objects are on the table."""

    def _create_references_markdown_description(self) -> str:
        """Create references description."""
        # pylint: disable=line-too-long
        return """N/A."""
