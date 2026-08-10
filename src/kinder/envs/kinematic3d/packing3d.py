"""Environment where multiple objects must be packed into a rack without collisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Type as TypingType
from typing import cast

import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose, SE2Pose, get_pose, set_pose
from pybullet_helpers.inverse_kinematics import check_body_collisions
from pybullet_helpers.utils import (
    create_pybullet_block,
    create_pybullet_block_with_peg,
    create_pybullet_hollow_box,
    create_pybullet_triangle_with_peg,
    get_triangle_vertices,
)
from relational_structs import Object, ObjectCentricState, Type
from relational_structs.utils import create_state_from_dict
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon

from kinder.core import ConstantObjectKinDEREnv, FinalConfigMeta
from kinder.envs.kinematic3d.base_env import (
    Kinematic3DEnvConfig,
    ObjectCentricKinematic3DRobotEnv,
)
from kinder.envs.kinematic3d.object_types import (
    Kinematic3DCuboidType,
    Kinematic3DEnvTypeFeatures,
    Kinematic3DPointType,
    Kinematic3DRobotType,
    Kinematic3DTriangleType,
)
from kinder.envs.kinematic3d.utils import (
    Kinematic3DObjectCentricState,
    remove_fingers_from_extended_joints,
)
from kinder.envs.utils import PURPLE

# Dimensions used by pybullet_helpers' part grasp pegs.
_PART_PEG_HALF_EXTENTS = (0.01, 0.01, 0.025)
_PENETRATION_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Packing3DEnvConfig(Kinematic3DEnvConfig, metaclass=FinalConfigMeta):
    """Config for Packing3DEnv()."""

    # Robot.
    max_action_mag: float = 0.2
    max_collision_check_step: float = 0.005
    robot_base_home_pose: SE2Pose = SE2Pose(-0.12, 0, 0)
    robot_base_z: float = -0.4

    # Table.
    table_pose: Pose = Pose((0.3, 0.0, -0.175))
    table_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    table_half_extents: tuple[float, float, float] = (0.2, 0.4, 0.25)

    # Rack (target) region.
    rack_half_extents: tuple[float, float, float] = (0.1, 0.15, 0.02)
    rack_wall_thickness: float = 0.01
    rack_rgba: tuple[float, float, float, float] = PURPLE + (1.0,)

    # Parts.
    part_half_extents_lb: tuple[float, float, float, float] = (0.05, 0.05, 0.01, 0)
    part_half_extents_ub: tuple[float, float, float, float] = (0.05, 0.05, 0.01, 0)
    part_rgba: tuple[float, float, float, float] = (0.2, 0.6, 0.2, 1.0)

    # Triangle parts.
    part_triangle_depth: float = 0.02  # fixed depth for triangle parts
    part_triangle_side_lb: float = 0.1  # min side length for triangle parts
    part_triangle_side_ub: float = 0.1  # max side length for triangle parts

    # Realistic background settings.
    realistic_bg: bool = True
    realistic_bg_position: tuple[float, float, float] = (0.7, -1.5, -0.37)
    realistic_bg_euler: tuple[float, float, float] = (np.pi / 2, 0, 0.0)
    realistic_bg_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Probability a part is triangular
    part_triangular_prob: float = 0.5

    def get_camera_kwargs(self) -> dict[str, Any]:
        """Get kwargs to pass to PyBullet camera."""
        return {
            "camera_target": (0, 0, 0),
            "camera_yaw": 90,
            "camera_distance": 1.0,
            "camera_pitch": -20,
        }

    def _sample_block_on_block_pose(
        self,
        top_block_half_extents: tuple[float, float, float],
        bottom_block_half_extents: tuple[float, float, float],
        bottom_block_pose: Pose,
        rng: np.random.Generator,
    ) -> Pose:
        """Sample one block pose on top of another one, with no hanging allowed."""
        assert np.allclose(
            bottom_block_pose.orientation, (0, 0, 0, 1)
        ), "Not implemented"

        lb = (
            bottom_block_pose.position[0]
            - bottom_block_half_extents[0]
            + top_block_half_extents[0],
            bottom_block_pose.position[1]
            - bottom_block_half_extents[1]
            + top_block_half_extents[1],
            bottom_block_pose.position[2]
            + bottom_block_half_extents[2]
            + top_block_half_extents[2],
        )

        ub = (
            bottom_block_pose.position[0]
            + bottom_block_half_extents[0]
            - top_block_half_extents[0],
            bottom_block_pose.position[1]
            + bottom_block_half_extents[1]
            - top_block_half_extents[1],
            bottom_block_pose.position[2]
            + bottom_block_half_extents[2]
            + top_block_half_extents[2],
        )

        x, y, z = rng.uniform(lb, ub)

        return Pose((x, y, z))

    def sample_block_on_table_pose(
        self, block_half_extents: tuple[float, float, float], rng: np.random.Generator
    ) -> Pose:
        """Sample an initial block pose given sampled half extents."""

        return self._sample_block_on_block_pose(
            block_half_extents, self.table_half_extents, self.table_pose, rng
        )

    def sample_part_half_extents(
        self, rng: np.random.Generator
    ) -> tuple[float, float, float, float]:
        """Sample half extents of a cuboid object (or an approximate box of a triangle
        object)."""
        return tuple(rng.uniform(self.part_half_extents_lb, self.part_half_extents_ub))

    def sample_part_triangle_features(
        self, rng: np.random.Generator
    ) -> tuple[float, float, float, float]:
        """Sample triangle features (side_a, side_b, depth) of a triangle object.

        triangle_type is encoded as:
        0 = equilateral
        1 = right
        """
        triangle_type = rng.choice([0, 1])
        if triangle_type == 0:  # equilateral
            side = rng.uniform(self.part_triangle_side_lb, self.part_triangle_side_ub)
            base = height = side
        else:  # right
            base = rng.uniform(self.part_triangle_side_lb, self.part_triangle_side_ub)
            height = rng.uniform(self.part_triangle_side_lb, self.part_triangle_side_ub)
        return base, height, self.part_triangle_depth, float(triangle_type)


class Packing3DObjectCentricState(Kinematic3DObjectCentricState):
    """A state in the Packing3DEnv().

    Adds convenience methods on top of Kinematic3DObjectCentricState().
    """

    def get_object_pose(self, name: str) -> Pose:
        """The pose of the object."""
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

        if obj.type == Kinematic3DTriangleType:
            # For triangle objects, we need to adjust the position to match the center
            # of the triangular prism, since the pose is defined at the centroid of the
            # triangle base.
            side_a, side_b, _, triangle_type = self.get_object_triangle_features(name)
            vertices = get_triangle_vertices(
                {0: "equilateral", 1: "right"}[int(triangle_type)],
                (side_a, side_b),
            )
            centroid_x = sum(v[0] for v in vertices) / 3.0 + self.get(obj, "pose_x")
            centroid_y = sum(v[1] for v in vertices) / 3.0 + self.get(obj, "pose_y")
            position = (centroid_x, centroid_y, self.get(obj, "pose_z"))

        return Pose(position, orientation)

    def get_object_half_extents_packing3d(
        self, name: str
    ) -> tuple[float, float, float, float]:
        """Get the half extents of a cuboid object."""
        obj = self.get_object_from_name(name)
        return (
            (
                self.get(obj, "half_extent_x"),
                self.get(obj, "half_extent_y"),
                self.get(obj, "half_extent_z"),
                -1,
            )
            if obj.type == Kinematic3DCuboidType
            else (
                max(self.get_object_triangle_features(name)[:2]) / 2,
                max(self.get_object_triangle_features(name)[:2]) / 2,
                self.get_object_triangle_features(name)[2] / 2,
                self.get_object_triangle_features(name)[3],
            )
        )

    def get_object_triangle_features(
        self, name: str
    ) -> tuple[float, float, float, float]:
        """Get the triangle features (side_a, side_b, depth, triangle_type) of a
        triangle object."""
        obj = self.get_object_from_name(name)
        return (
            self.get(obj, "side_a"),
            self.get(obj, "side_b"),
            self.get(obj, "depth"),
            self.get(obj, "triangle_type"),
        )

    @staticmethod
    def _get_box_xy_polygon(
        pose: Pose,
        half_extents: tuple[float, float, float],
    ) -> Polygon:
        pos = np.array(pose.position)
        rot = Rotation.from_quat(pose.orientation).as_matrix()
        local_corners = np.array(
            [
                [sx * half_extents[0], sy * half_extents[1], sz * half_extents[2]]
                for sx in [-1, 1]
                for sy in [-1, 1]
                for sz in [-1, 1]
            ]
        )
        world_corners = (rot @ local_corners.T).T + pos
        return cast(Polygon, Polygon(world_corners[:, :2]).convex_hull)

    def _get_triangle_xy_polygon(self, name: str) -> Polygon:
        obj = self.get_object_from_name(name)
        position = np.array(
            [
                self.get(obj, "pose_x"),
                self.get(obj, "pose_y"),
                self.get(obj, "pose_z"),
            ]
        )
        orientation = (
            self.get(obj, "pose_qx"),
            self.get(obj, "pose_qy"),
            self.get(obj, "pose_qz"),
            self.get(obj, "pose_qw"),
        )
        side_a, side_b, _, triangle_type = self.get_object_triangle_features(name)
        local_vertices = np.array(
            get_triangle_vertices(
                {0: "equilateral", 1: "right"}[int(triangle_type)],
                (side_a, side_b),
            )
        )
        rot = Rotation.from_quat(orientation).as_matrix()
        world_vertices = (rot @ local_vertices.T).T + position
        return Polygon(world_vertices[:, :2])

    def is_part_inside_rack(self, part_name: str) -> bool:
        """Check whether a part's true footprint is inside the rack footprint."""
        part = self.get_object_from_name(part_name)
        rack_polygon = self._get_box_xy_polygon(self.rack_pose, self.rack_half_extents)
        if part.type == Kinematic3DCuboidType:
            part_polygon = self._get_box_xy_polygon(
                self.get_object_pose(part_name),
                self.get_object_half_extents_packing3d(part_name)[:3],
            )
        else:
            assert part.type == Kinematic3DTriangleType
            part_polygon = self._get_triangle_xy_polygon(part_name)
        return rack_polygon.buffer(1e-9).contains(part_polygon)

    @property
    def objects(self) -> list[Object]:
        """Get all objects in the state."""
        objects = list(self.data.keys())
        return objects

    @property
    def rack_half_extents(self) -> tuple[float, float, float]:
        """Get the half extents of the rack."""
        return self.get_object_half_extents_packing3d("rack")[:3]

    @property
    def rack_pose(self) -> Pose:
        """Get the pose of the rack."""
        return self.get_object_pose("rack")

    @property
    def part_poses(self) -> dict[str, Pose]:
        """Get the poses of all parts."""
        poses = {}
        for obj in self.objects:
            if obj.name.startswith("part"):
                poses[obj.name] = self.get_object_pose(obj.name)
        return poses

    @property
    def part_types(self) -> dict[str, Type]:
        """Get the types of all parts."""
        types = {}
        for obj in self.objects:
            if obj.name.startswith("part"):
                types[obj.name] = obj.type
        return types

    @property
    def part_features(
        self,
    ) -> dict[
        str,
        tuple[float, float, float, float],
    ]:
        """Get the features of all parts."""
        features = {}
        for obj in self.objects:
            if obj.name.startswith("part"):
                if obj.type == Kinematic3DCuboidType:
                    features[obj.name] = self.get_object_half_extents_packing3d(
                        obj.name
                    )
                elif obj.type == Kinematic3DTriangleType:
                    features[obj.name] = self.get_object_triangle_features(obj.name)
                else:
                    raise ValueError(f"Unsupported part type: {obj.type}")
        return features

    @property
    def available_parts(self) -> list[str]:
        """Get the names of all parts that are not currently grasped and not already
        placed on rack."""
        available_parts = []
        for obj in self.objects:
            if obj.name.startswith("part"):
                if self.get(obj, "grasp_active") < 0.5 and not self.is_part_inside_rack(
                    obj.name
                ):
                    available_parts.append(obj.name)
        return available_parts

    @property
    def grasped_object(self) -> str | None:
        """The name of the currently grasped object, or None if there is none."""
        grasped_objs: list[Object] = []
        for obj in self.get_objects(Kinematic3DCuboidType) + self.get_objects(
            Kinematic3DTriangleType
        ):
            if self.get(obj, "grasp_active") > 0.5:
                grasped_objs.append(obj)
        if not grasped_objs:
            return None
        assert len(grasped_objs) == 1, "Multiple objects should not be grasped"
        grasped_obj = grasped_objs[0]
        return grasped_obj.name


class ObjectCentricPacking3DEnv(
    ObjectCentricKinematic3DRobotEnv[Packing3DObjectCentricState, Packing3DEnvConfig]
):
    """Environment where small parts must be packed into a rack without collisions."""

    def __init__(
        self,
        num_parts: int = 2,
        config: Packing3DEnvConfig = Packing3DEnvConfig(),
        **kwargs,
    ) -> None:
        self._num_parts = num_parts
        super().__init__(config=config, **kwargs)

        # Create table.
        self.table_id = create_pybullet_block(
            self.config.table_rgba,
            half_extents=self.config.table_half_extents,
            physics_client_id=self.physics_client_id,
        )
        set_pose(self.table_id, self.config.table_pose, self.physics_client_id)

        # Rack (created in reset because geometry could be randomized later)
        self._rack_half_extents = self.config.rack_half_extents
        self._rack_id = create_pybullet_hollow_box(
            self.config.rack_rgba,
            half_extents=self._rack_half_extents,
            wall_thickness=self.config.rack_wall_thickness,
            physics_client_id=self.physics_client_id,
        )
        rack_pose = Pose(
            (
                self.config.table_pose.position[0],
                self.config.table_pose.position[1],
                self.config.table_pose.position[2]
                + self.config.table_half_extents[2]
                + self.config.rack_half_extents[2],
            )
        )
        set_pose(self._rack_id, rack_pose, self.physics_client_id)

        # Parts
        self._parts: dict[str, Object] = {}
        self._part_ids: dict[str, int] = {}
        self._part_ids_to_type: dict[int, Type] = {}
        self._part_id_to_half_extents: dict[int, tuple[float, float, float]] = {}
        self._part_ids_to_triangle_features: dict[
            int, tuple[float, float, float, float]
        ] = {}

    @property
    def state_cls(self) -> TypingType[Kinematic3DObjectCentricState]:
        return Packing3DObjectCentricState

    def _create_state_dict(
        self, objects: list[tuple[str, Type]]
    ) -> dict[Object, dict[str, float]]:
        state_dict: dict[Object, dict[str, float]] = {}
        for object_name, object_type in objects:
            obj = Object(object_name, object_type)
            feats: dict[str, float] = {}
            # Handle robots.
            if object_type == Kinematic3DRobotType:
                # Add base pose.
                base_pose = self.robot.get_base()
                feats["pos_base_x"] = base_pose.x
                feats["pos_base_y"] = base_pose.y
                feats["pos_base_rot"] = base_pose.rot
                # Add joints.
                joints = remove_fingers_from_extended_joints(
                    self._robot_arm.get_joint_positions()
                )
                for i, v in enumerate(joints):
                    feats[f"joint_{i+1}"] = v
                # Add finger state.
                feats["finger_state"] = self._robot_arm.get_finger_state()
                # Add grasp.
                grasp_tf_feat_names = [
                    "grasp_tf_x",
                    "grasp_tf_y",
                    "grasp_tf_z",
                    "grasp_tf_qx",
                    "grasp_tf_qy",
                    "grasp_tf_qz",
                    "grasp_tf_qw",
                ]
                if self._grasped_object_transform is None:
                    feats["grasp_active"] = 0
                    for feat_name in grasp_tf_feat_names:
                        feats[feat_name] = 0
                else:
                    feats["grasp_active"] = 1
                    grasp_tf_feats = list(
                        self._grasped_object_transform.position
                    ) + list(self._grasped_object_transform.orientation)
                    for feat_name, feat in zip(
                        grasp_tf_feat_names, grasp_tf_feats, strict=True
                    ):
                        feats[feat_name] = feat
            # Handle cuboids.
            elif object_type == Kinematic3DCuboidType:
                # Add pose.
                body_id = self._object_name_to_pybullet_id(object_name)
                pose = get_pose(body_id, self.physics_client_id)
                pose_feat_names = [
                    "pose_x",
                    "pose_y",
                    "pose_z",
                    "pose_qx",
                    "pose_qy",
                    "pose_qz",
                    "pose_qw",
                ]
                pose_feats = list(pose.position) + list(pose.orientation)
                for feat_name, feat in zip(pose_feat_names, pose_feats, strict=True):
                    feats[feat_name] = feat
                # Add grasp active.
                if self._grasped_object == object_name:
                    feats["grasp_active"] = 1
                else:
                    feats["grasp_active"] = 0
                # Add half extents.
                half_extent_names = ["half_extent_x", "half_extent_y", "half_extent_z"]
                half_extents = self._get_half_extents(object_name)
                for feat_name, feat in zip(
                    half_extent_names, half_extents, strict=True
                ):
                    feats[feat_name] = feat
                feats["object_type"] = -1.0  # cuboid
            # Handle points.
            elif object_type == Kinematic3DPointType:
                # Add position.
                body_id = self._object_name_to_pybullet_id(object_name)
                pose = get_pose(body_id, self.physics_client_id)
                feats["x"] = pose.position[0]
                feats["y"] = pose.position[1]
                feats["z"] = pose.position[2]
            # Handle triangles.
            elif object_type == Kinematic3DTriangleType:
                body_id = self._object_name_to_pybullet_id(object_name)
                pose = get_pose(body_id, self.physics_client_id)
                pose_feat_names = [
                    "pose_x",
                    "pose_y",
                    "pose_z",
                    "pose_qx",
                    "pose_qy",
                    "pose_qz",
                    "pose_qw",
                ]
                pose_feats = list(pose.position) + list(pose.orientation)
                for feat_name, feat in zip(pose_feat_names, pose_feats, strict=True):
                    feats[feat_name] = feat
                # Add grasp active.
                if self._grasped_object == object_name:
                    feats["grasp_active"] = 1
                else:
                    feats["grasp_active"] = 0
                # Triangle-specific features (fallback to 0).
                # Here we use _get_triangle_features implemented within the envs.
                a, b, depth, ttype = self._get_triangle_features(object_name)
                feats["triangle_type"] = float(ttype)
                feats["side_a"] = float(a)
                feats["side_b"] = float(b)
                feats["depth"] = float(depth)

            else:
                raise NotImplementedError(f"Unsupported object type: {object_type}")
            # Add feats to state dict.
            state_dict[obj] = feats
        return state_dict

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        return self._create_state_dict([("table", Kinematic3DCuboidType)])

    def _reset_objects(self) -> None:

        # Destroy previous parts.
        for old_id in set(self._part_ids.values()):
            if old_id is not None:
                p.removeBody(old_id, physicsClientId=self.physics_client_id)

        # Create parts and place them on the table with rejection sampling to avoid
        # initial collisions. Parts are modeled as cuboids and prisms with fixed z-depth.
        self._part_ids = {}
        self._part_ids_to_type = {}
        self._part_id_to_half_extents = {}
        self._part_ids_to_triangle_features = {}
        part_z_half_extent = self.config.part_half_extents_ub[
            2
        ]  # fixed z-depth for all parts
        for i in range(self._num_parts):
            name = f"part{i}"
            part_type = (
                Kinematic3DCuboidType
                if self.config.part_triangular_prob * self._num_parts >= i + 1
                else Kinematic3DTriangleType
            )

            # Sample part half extents from config.
            if part_type == Kinematic3DCuboidType:
                sampled = self.config.sample_part_half_extents(self.np_random)
                half_extents = (sampled[0], sampled[1], sampled[2])
                part_id = create_pybullet_block_with_peg(
                    self.config.part_rgba,
                    half_extents=half_extents,
                    physics_client_id=self.physics_client_id,
                )
                self._part_id_to_half_extents[part_id] = half_extents
                self._part_ids[name] = part_id
                self._part_ids_to_type[part_id] = Kinematic3DCuboidType
                self._part_ids_to_triangle_features[part_id] = (
                    sampled[0],
                    sampled[1],
                    sampled[2],
                    -1,
                )

            elif part_type == Kinematic3DTriangleType:
                side_a, side_b, depth, triangle_type = (
                    self.config.sample_part_triangle_features(self.np_random)
                )
                half_extents = (
                    max(side_a, side_b) / 2,
                    max(side_a, side_b) / 2,
                    depth / 2,
                )
                part_id = create_pybullet_triangle_with_peg(
                    self.config.part_rgba,
                    triangle_type={0: "equilateral", 1: "right"}[int(triangle_type)],
                    side_lengths=(side_a, side_b),
                    depth=depth,
                    physics_client_id=self.physics_client_id,
                )
                self._part_id_to_half_extents[part_id] = half_extents
                self._part_ids[name] = part_id
                self._part_ids_to_type[part_id] = Kinematic3DTriangleType
                self._part_ids_to_triangle_features[part_id] = (
                    side_a,
                    side_b,
                    depth,
                    triangle_type,
                )

            # Place part on table while avoiding collisions with other parts and
            # the rack (we allow parts to start outside the rack)
            for _ in range(100_000):
                # Sample a pose on the table surface.
                x = self.np_random.uniform(
                    self.config.table_pose.position[0]
                    - self.config.table_half_extents[0]
                    + half_extents[0],
                    self.config.table_pose.position[0]
                    + self.config.table_half_extents[0] * 0.5
                    - half_extents[0],
                )
                y = self.np_random.uniform(
                    self.config.table_pose.position[1]
                    - self.config.table_half_extents[1]
                    + half_extents[1],
                    self.config.table_pose.position[1]
                    + self.config.table_half_extents[1]
                    - half_extents[1],
                )
                z = (
                    self.config.table_pose.position[2]
                    + self.config.table_half_extents[2]
                    + part_z_half_extent * 2
                )

                # Check that objects are not initialized too close to rack
                rack_pose = get_pose(self._rack_id, self.physics_client_id)
                rack_x_min = (
                    rack_pose.position[0]
                    - 1.5 * self._rack_half_extents[0]
                    - half_extents[0]
                )
                rack_x_max = (
                    rack_pose.position[0]
                    + 1.5 * self._rack_half_extents[0]
                    + half_extents[0]
                )
                rack_y_min = (
                    rack_pose.position[1]
                    - 1.5 * self._rack_half_extents[1]
                    - half_extents[1]
                )
                rack_y_max = (
                    rack_pose.position[1]
                    + 1.5 * self._rack_half_extents[1]
                    + half_extents[1]
                )
                if rack_x_min <= x <= rack_x_max and rack_y_min <= y <= rack_y_max:
                    continue  # too close to rack

                set_pose(part_id, Pose((x, y, z)), self.physics_client_id)

                collision_exists = False
                for other_id in ({self._rack_id} | set(self._part_ids.values())) - {
                    part_id
                }:
                    if check_body_collisions(part_id, other_id, self.physics_client_id):
                        collision_exists = True
                        break
                if not collision_exists:
                    collision_exists = any(
                        self._parts_penetrate(name, other_name)
                        for other_name in self._part_ids
                        if other_name != name
                    )

                if not collision_exists:
                    break

    def _set_object_states(self, obs: Kinematic3DObjectCentricState) -> None:
        assert isinstance(obs, Packing3DObjectCentricState)
        # Update rack (recreate if half extents changed)
        if self._rack_id is not None:
            p.removeBody(self._rack_id, physicsClientId=self.physics_client_id)
        self._rack_half_extents = self.config.rack_half_extents
        self._rack_id = create_pybullet_hollow_box(
            PURPLE + (0.8,),
            half_extents=self._rack_half_extents,
            wall_thickness=self.config.rack_wall_thickness,
            physics_client_id=self.physics_client_id,
        )
        if self._rack_id is not None:
            # Rack pose expected as a cuboid in the state
            set_pose(self._rack_id, obs.get_object_pose("rack"), self.physics_client_id)

        parts = obs.part_poses
        assert (
            len(parts) == self._num_parts
        ), f"Expected {self._num_parts} parts, got {len(parts)}"

        # Update parts
        for i in range(self._num_parts):
            name = list(parts.keys())[i]
            pose = self._get_pybullet_pose_from_state(obs, name)
            part_id = self._object_name_to_pybullet_id(name)
            set_pose(part_id, pose, self.physics_client_id)

    def _get_pybullet_pose_from_state(
        self, obs: Packing3DObjectCentricState, name: str
    ) -> Pose:
        """Get the raw PyBullet pose from state features for state restoration."""
        obj = obs.get_object_from_name(name)
        return Pose(
            (
                obs.get(obj, "pose_x"),
                obs.get(obj, "pose_y"),
                obs.get(obj, "pose_z"),
            ),
            (
                obs.get(obj, "pose_qx"),
                obs.get(obj, "pose_qy"),
                obs.get(obj, "pose_qz"),
                obs.get(obj, "pose_qw"),
            ),
        )

    def _object_name_to_pybullet_id(self, object_name: str) -> int:
        if object_name == "rack":
            assert self._rack_id is not None
            return self._rack_id
        if object_name == "table":
            return self.table_id
        if object_name.startswith("part"):
            return self._part_ids[object_name]
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_collision_object_ids(self) -> set[int]:
        ids = {self.table_id}
        if self._rack_id is not None:
            ids.add(self._rack_id)
        ids |= set(self._part_ids.values())
        return ids

    def _get_movable_object_names(self) -> set[str]:
        return set(self._part_ids.keys())

    def _get_surface_object_names(self) -> set[str]:
        # The rack and table are surfaces.
        names = {"table"}
        if self._rack_id is not None:
            names.add("rack")
        return names

    @staticmethod
    def _box_local_geometry(
        half_extents: tuple[float, float, float],
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return vertices, face normals, and edge directions for a local box."""
        vertices = np.array(
            [
                [
                    center[0] + sx * half_extents[0],
                    center[1] + sy * half_extents[1],
                    center[2] + sz * half_extents[2],
                ]
                for sx in (-1, 1)
                for sy in (-1, 1)
                for sz in (-1, 1)
            ],
            dtype=float,
        )
        axes = np.eye(3)
        return vertices, axes, axes

    @staticmethod
    def _triangle_prism_local_geometry(
        triangle_type: float,
        side_a: float,
        side_b: float,
        depth: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return convex geometry for an equilateral or right triangular prism."""
        triangle_vertices = np.asarray(
            get_triangle_vertices(
                {0: "equilateral", 1: "right"}[int(triangle_type)],
                (side_a, side_b),
            ),
            dtype=float,
        )
        lower = triangle_vertices.copy()
        upper = triangle_vertices.copy()
        lower[:, 2] = -depth / 2
        upper[:, 2] = depth / 2
        vertices = np.concatenate((lower, upper))

        face_normals = [np.array([0.0, 0.0, 1.0])]
        edge_directions = [np.array([0.0, 0.0, 1.0])]
        for index in range(3):
            edge = triangle_vertices[(index + 1) % 3] - triangle_vertices[index]
            edge[2] = 0.0
            edge_directions.append(edge)
            face_normals.append(np.array([edge[1], -edge[0], 0.0]))
        return vertices, np.asarray(face_normals), np.asarray(edge_directions)

    def _get_part_convex_components(
        self, part_name: str
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return the world-frame convex components of a part and its grasp peg."""
        part_id = self._object_name_to_pybullet_id(part_name)
        part_type = self._part_ids_to_type[part_id]
        if part_type == Kinematic3DCuboidType:
            half_extents = self._part_id_to_half_extents[part_id]
            main_geometry = self._box_local_geometry(half_extents)
            peg_center = (0.0, 0.0, half_extents[2] + _PART_PEG_HALF_EXTENTS[2])
        else:
            assert part_type == Kinematic3DTriangleType
            side_a, side_b, depth, triangle_type = self._part_ids_to_triangle_features[
                part_id
            ]
            main_geometry = self._triangle_prism_local_geometry(
                triangle_type, side_a, side_b, depth
            )
            triangle_vertices = np.asarray(
                get_triangle_vertices(
                    {0: "equilateral", 1: "right"}[int(triangle_type)],
                    (side_a, side_b),
                ),
                dtype=float,
            )
            centroid = np.mean(triangle_vertices, axis=0)
            peg_center = (
                float(centroid[0]),
                float(centroid[1]),
                depth / 2 + _PART_PEG_HALF_EXTENTS[2],
            )
        peg_geometry = self._box_local_geometry(_PART_PEG_HALF_EXTENTS, peg_center)

        pose = get_pose(part_id, self.physics_client_id)
        rotation = Rotation.from_quat(pose.orientation).as_matrix()
        translation = np.asarray(pose.position)
        world_components = []
        for vertices, face_normals, edge_directions in (main_geometry, peg_geometry):
            world_components.append(
                (
                    (rotation @ vertices.T).T + translation,
                    (rotation @ face_normals.T).T,
                    (rotation @ edge_directions.T).T,
                )
            )
        return world_components

    @staticmethod
    def _convex_components_penetrate(
        first: tuple[np.ndarray, np.ndarray, np.ndarray],
        second: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> bool:
        """Use the separating axis theorem to detect strict 3D penetration."""
        first_vertices, first_normals, first_edges = first
        second_vertices, second_normals, second_edges = second
        candidate_axes = [*first_normals, *second_normals]
        candidate_axes.extend(
            np.cross(first_edge, second_edge)
            for first_edge in first_edges
            for second_edge in second_edges
        )
        for axis in candidate_axes:
            norm = np.linalg.norm(axis)
            if norm <= _PENETRATION_TOLERANCE:
                continue
            normalized_axis = axis / norm
            first_projection = first_vertices @ normalized_axis
            second_projection = second_vertices @ normalized_axis
            overlap = min(first_projection.max(), second_projection.max()) - max(
                first_projection.min(), second_projection.min()
            )
            if overlap <= _PENETRATION_TOLERANCE:
                return False
        return True

    def _parts_penetrate(self, first_name: str, second_name: str) -> bool:
        """Check exact part geometry, including pegs, without mesh contact queries."""
        first_components = self._get_part_convex_components(first_name)
        second_components = self._get_part_convex_components(second_name)
        return any(
            self._convex_components_penetrate(first, second)
            for first in first_components
            for second in second_components
        )

    def _robot_or_held_object_collision_exists(self) -> bool:
        """Check analytic collisions for a held part."""
        if super()._robot_or_held_object_collision_exists():
            return True
        if self._grasped_object is None:
            return False
        return any(
            self._parts_penetrate(self._grasped_object, other_name)
            for other_name in self._part_ids
            if other_name != self._grasped_object
        )

    def _get_half_extents(self, object_name: str) -> tuple[float, float, float]:
        if object_name == "rack":
            return self.config.rack_half_extents
        if object_name == "table":
            return self.config.table_half_extents
        assert object_name.startswith("part")
        part_id = self._object_name_to_pybullet_id(object_name)
        return self._part_id_to_half_extents[part_id]

    def _get_triangle_features(
        self, object_name: str
    ) -> tuple[float, float, float, float]:
        if not object_name.startswith("part"):
            raise ValueError(f"Object {object_name} is not a part")
        part_id = self._object_name_to_pybullet_id(object_name)
        if part_id not in self._part_ids_to_triangle_features:
            raise ValueError(f"Object {object_name} is not a triangle")
        return self._part_ids_to_triangle_features[part_id]

    def _get_obs(self) -> Packing3DObjectCentricState:
        state_dict = self._create_state_dict(
            [("robot", Kinematic3DRobotType), ("rack", Kinematic3DCuboidType)]
            + [
                (
                    f"part{i}",
                    self._part_ids_to_type[
                        self._object_name_to_pybullet_id(f"part{i}")
                    ],
                )
                for i in range(self._num_parts)
            ]
        )
        state = create_state_from_dict(
            state_dict,
            Kinematic3DEnvTypeFeatures,
            state_cls=Packing3DObjectCentricState,
        )
        assert isinstance(state, Packing3DObjectCentricState)
        return state

    def goal_reached(self) -> bool:
        if self._grasped_object is not None:
            return False
        obs = self._get_obs()
        part_names = []
        for i in range(self._num_parts):
            part_name = f"part{i}"
            if not self._part_is_seated_in_rack(part_name, obs):
                return False
            part_names.append(part_name)

        # Reject penetration between seated parts.
        for i, part_name in enumerate(part_names):
            for other_part_name in part_names[i + 1 :]:
                if self._parts_penetrate(part_name, other_part_name):
                    return False

        return True

    def _bodies_penetrate(self, body1: int, body2: int) -> bool:
        """Check for penetration while allowing nonpenetrating support contact."""
        contact_points = p.getClosestPoints(
            body1,
            body2,
            distance=0.0,
            physicsClientId=self.physics_client_id,
        )
        return any(point[8] < -1e-6 for point in contact_points)

    def _part_is_seated_in_rack(
        self, part_name: str, obs: Packing3DObjectCentricState
    ) -> bool:
        """Check that a part is supported by the floor of the rack cavity."""
        rack_half_extents = obs.rack_half_extents
        wall_thickness = self.config.rack_wall_thickness
        inner_half_extents = (
            rack_half_extents[0] - wall_thickness,
            rack_half_extents[1] - wall_thickness,
            rack_half_extents[2],
        )
        rack_inner_polygon = (
            obs._get_box_xy_polygon(  # pylint: disable=protected-access
                obs.rack_pose, inner_half_extents
            )
        )
        part_vertices = self._get_part_convex_components(part_name)[0][0]
        part_polygon = cast(Polygon, Polygon(part_vertices[:, :2]).convex_hull)
        if not rack_inner_polygon.buffer(1e-9).contains(part_polygon):
            return False

        part_id = self._object_name_to_pybullet_id(part_name)
        part_lower_z = float(part_vertices[:, 2].min())
        part_upper_z = float(part_vertices[:, 2].max())
        rack_floor_z = obs.rack_pose.position[2] - rack_half_extents[2] + wall_thickness
        rack_rim_z = obs.rack_pose.position[2] + rack_half_extents[2]
        tolerance = self.config.min_placement_dist
        if not np.isclose(part_lower_z, rack_floor_z, atol=tolerance):
            return False
        if part_upper_z > rack_rim_z + tolerance:
            return False

        if self._bodies_penetrate(part_id, self._rack_id):
            return False

        return self._rack_id in self._get_surfaces_supporting_object(part_id)


class Packing3DEnv(ConstantObjectKinDEREnv):
    """Packing 3D env with a constant number of objects."""

    def __init__(self, num_parts: int = 2, **kwargs) -> None:
        self._num_parts = num_parts
        super().__init__(num_parts=num_parts, **kwargs)

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricKinematic3DRobotEnv:
        return ObjectCentricPacking3DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        constant_objects = ["robot", "rack"]
        for obj in sorted(exemplar_state):
            if obj.name.startswith("part"):
                constant_objects.append(obj.name)
        return constant_objects

    def _create_env_markdown_description(self) -> str:
        """Create environment description."""
        # pylint: disable=line-too-long
        config = self._object_centric_env.config
        assert isinstance(config, Packing3DEnvConfig)
        return f"""A 3D packing environment where the goal is to place a set of parts into a rack without collisions.

The robot is a Kinova Gen-3 with 7 degrees of freedom that can grasp and manipulate objects. The environment consists of:
- A **table** with dimensions {config.table_half_extents[0]*2:.3f}m × {config.table_half_extents[1]*2:.3f}m × {config.table_half_extents[2]*2:.3f}m
- A **rack** (purple) with half-extents {config.rack_half_extents}
- **Parts** (green) that must be packed into the rack. Parts are sampled with half-extents in {config.part_half_extents_lb} to {config.part_half_extents_ub} and a probability {config.part_triangular_prob} of being triangle-shaped (triangles are represented as triangular prisms with depth {config.part_triangle_depth:.3f}m when used).
"""

    def _create_variant_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return "The number of parts to pack differs between environment variants. For example, Packing3D-p1 has 1 part, while Packing3D-p3 has 3 parts."

    def _create_variant_specific_description(self) -> str:
        if self._num_parts == 1:
            return "This variant has 1 part to pack into the rack."
        return f"This variant has {self._num_parts} parts to pack into the rack."

    def _create_action_space_markdown_description(self) -> str:
        """Create action space description."""
        # pylint: disable=line-too-long
        config = self._object_centric_env.config
        assert isinstance(config, Packing3DEnvConfig)
        return f"""Actions control the change in joint positions:
- **delta_arm_joints**: Change in joint positions for all {len(config.initial_joints)} joints (list of floats)

The action is a Packing3DAction dataclass with delta_arm_joints field. Each delta is clipped to the range [-{config.max_action_mag:.3f}, {config.max_action_mag:.3f}].

The resulting joint positions are clipped to the robot's joint limits before being applied. The robot can automatically grasp objects when the gripper is close enough and release them with appropriate actions.
"""

    def _create_reward_markdown_description(self) -> str:
        """Create reward description."""
        # pylint: disable=line-too-long
        return """The reward structure is simple:
- **-1.0** penalty at every timestep until the goal is reached
- **Termination** occurs when all parts are placed in the rack and none are grasped

The goal is considered reached when:
1. The robot is not currently grasping any part
2. Every part is resting on (supported by) the rack surface

Support is determined based on contact between a part and the rack within a small distance threshold (configured by the environment).

This encourages the robot to efficiently pack the parts into the rack while avoiding infinite episodes.
"""

    def _create_references_markdown_description(self) -> str:
        """Create references description."""
        # pylint: disable=line-too-long
        return """Packing tasks are common in robotics and automated warehousing literature. This environment is inspired by standard manipulation benchmarks and simple bin-packing problems; it’s intended as a deterministic, physics-based testbed.
"""
