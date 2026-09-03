"""PyBullet environment where tall cylinders must be picked from the ground and placed
on a shelf.

The cylinders are tall enough that a side grasp (horizontal approach) is required rather
than a top grasp. Because cylinders are rotationally symmetric, any horizontal approach
angle is a valid grasp direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Type as TypingType

import pybullet as p
from pybullet_helpers.geometry import Pose, get_pose, set_pose
from pybullet_helpers.utils import create_pybullet_block, create_pybullet_cylinder
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

# The "object_type" feature value for cylinders. Cuboids use -1.0 (see
# ObjectCentricKinematic3DRobotEnv._create_state_dict).
CYLINDER_OBJECT_TYPE = 1.0


@dataclass(frozen=True)
class CylinderShelf3DEnvConfig(Kinematic3DEnvConfig, metaclass=FinalConfigMeta):
    """Config for CylinderShelf3DEnv()."""

    max_action_mag: float = 0.2

    # Shelf.
    shelf_pose: Pose = Pose((2.0, 2.4, 0.02))
    shelf_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    shelf_width: float = 0.60198
    shelf_depth: float = 0.254
    shelf_height: float = 0.0127
    shelf_spacing: float = 0.254
    shelf_support_width: float = 0.0127
    shelf_num_layers: int = 4
    # Board indices absent from the physical shelf. Removing an inner board
    # merges the openings around it — e.g. (1,) models a shelf whose bottom
    # inner board was taken out, leaving one tall opening above the bottom
    # board for objects taller than the board spacing.
    shelf_omitted_layers: tuple[int, ...] = ()
    # Optional explicit board layout: each entry is a board CENTER's z offset from
    # shelf_pose.z, ascending. When set, this overrides the uniform
    # shelf_spacing/shelf_num_layers/shelf_omitted_layers layout — a re-staged physical
    # shelf's boards need not be evenly spaced (e.g. a big bottom opening for tall items
    # and a small upper one). The side supports then span exactly the given boards.
    shelf_layer_zs: tuple[float, ...] | None = None
    shelf_texture: Path = Path(__file__).parent / "assets" / "dark-wood-texture.png"

    # World bounds.
    x_lb: float = -1.0
    x_ub: float = 1.0
    y_lb: float = -1.0
    y_ub: float = 1.0

    # Open-top staging boxes the cylinders can start inside (real scenes stage the
    # stock in cardboard boxes, which is what rules out horizontal side grasps). Each
    # entry is a box's INNER floor extents plus its wall height, world frame:
    # (x_lo, x_hi, y_lo, y_hi, wall_height). Walls are real collision bodies; the top
    # and floor are open. Empty (the default) builds no boxes.
    boxes: tuple[tuple[float, float, float, float, float], ...] = ()
    box_wall_thickness: float = 0.012
    box_rgba: tuple[float, float, float, float] = (0.72, 0.56, 0.38, 1.0)

    # Cylinders. Heights are cycled by cylinder index, so multi-object
    # variants get cylinders of different heights. All heights should stay
    # comfortably below shelf_spacing so a placed cylinder fits between
    # shelf layers.
    cylinder_radius: float = 0.03
    # Optional per-cylinder radii, cycled by index like the heights; None means every
    # cylinder uses cylinder_radius. Real object sets rarely share one diameter.
    cylinder_radii: tuple[float, ...] | None = None
    cylinder_heights: tuple[float, ...] = (0.20, 0.15, 0.175)
    # Optional per-cylinder initial-pose regions (x_lo, x_ub, y_lb, y_ub), cycled by
    # index. When set, cylinder i's pose is sampled inside its region (e.g. inside a
    # staging box) instead of the world bounds.
    cylinder_init_regions: tuple[tuple[float, float, float, float], ...] | None = None
    cylinder_rgbas: tuple[tuple[float, float, float, float], ...] = (
        (0.8, 0.2, 0.2, 1.0),
        (0.2, 0.5, 0.8, 1.0),
        (0.2, 0.7, 0.3, 1.0),
        (0.8, 0.6, 0.2, 1.0),
        (0.6, 0.3, 0.7, 1.0),
    )

    # Gripper.
    gripper_open_threshold: float = 0.01

    # Goal checking: a cylinder counts as placed when it stands within the
    # shelf footprint with its resting height within this tolerance of some
    # present board's surface.
    on_shelf_z_tolerance: float = 0.05

    def get_camera_kwargs(self) -> dict[str, Any]:
        """Get kwargs to pass to PyBullet camera."""
        return {
            "camera_target": (0, 0, 0),
            "camera_yaw": 0,
            "camera_distance": 2.0,
            "camera_pitch": -20,
        }

    def get_cylinder_height(self, idx: int) -> float:
        """Get the height of the cylinder with the given index."""
        return self.cylinder_heights[idx % len(self.cylinder_heights)]

    def get_cylinder_radius(self, idx: int) -> float:
        """Get the radius of the cylinder with the given index."""
        if self.cylinder_radii is not None:
            return self.cylinder_radii[idx % len(self.cylinder_radii)]
        return self.cylinder_radius

    def get_cylinder_init_region(
        self, idx: int
    ) -> tuple[float, float, float, float] | None:
        """The (x_lb, x_ub, y_lb, y_ub) initial-pose region for cylinder ``idx``."""
        if self.cylinder_init_regions is None:
            return None
        return self.cylinder_init_regions[idx % len(self.cylinder_init_regions)]

    def get_present_layer_indices(self) -> list[int]:
        """Indices of the boards actually present on the shelf."""
        return [
            i
            for i in range(self.shelf_num_layers)
            if i not in self.shelf_omitted_layers
        ]

    def get_layer_center_z_offsets(self) -> list[float]:
        """Each present board's CENTER z offset from shelf_pose.z, ascending."""
        if self.shelf_layer_zs is not None:
            return list(self.shelf_layer_zs)
        return [
            i * (self.shelf_spacing + self.shelf_height)
            for i in self.get_present_layer_indices()
        ]

    def get_layer_surface_zs(self) -> list[float]:
        """World-frame z of each present board's top surface, ascending."""
        return [
            self.shelf_pose.position[2] + offset + self.shelf_height / 2
            for offset in self.get_layer_center_z_offsets()
        ]

    def get_layer_openings(self) -> list[tuple[float, float]]:
        """Per present board: (surface z, clear height above it).

        The clear height runs to the underside of the next present board, or
        infinity for the topmost board.
        """
        surface_zs = self.get_layer_surface_zs()
        openings = []
        for k, surface_z in enumerate(surface_zs):
            if k + 1 < len(surface_zs):
                clearance = surface_zs[k + 1] - self.shelf_height - surface_z
            else:
                clearance = float("inf")
            openings.append((surface_z, clearance))
        return openings

    def get_cylinder_rgba(self, idx: int) -> tuple[float, float, float, float]:
        """Get the color of the cylinder with the given index."""
        return self.cylinder_rgbas[idx % len(self.cylinder_rgbas)]


def _create_shelf_with_omitted_layers(
    config: CylinderShelf3DEnvConfig, physics_client_id: int
) -> tuple[int, set[int]]:
    """Build the shelf multibody, skipping the omitted board indices.

    Adapted from pybullet_helpers' create_pybullet_shelf, which only builds evenly
    spaced boards; the side supports still span the full height so the frame matches a
    shelf whose inner board was physically removed.
    """
    collision_shape_ids = []
    visual_shape_ids = []
    link_positions = []

    layer_center_zs = config.get_layer_center_z_offsets()
    for layer_z in layer_center_zs:
        half_extents = [
            config.shelf_width / 2,
            config.shelf_depth / 2,
            config.shelf_height / 2,
        ]
        collision_shape_ids.append(
            p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                physicsClientId=physics_client_id,
            )
        )
        visual_shape_ids.append(
            p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=config.shelf_rgba,
                physicsClientId=physics_client_id,
            )
        )
        link_positions.append([0, 0, layer_z])

    shelf_link_ids = set(range(len(layer_center_zs)))

    if config.shelf_layer_zs is not None:
        # Explicit layout: the frame stands on the floor (world z = 0) and rises to the
        # top board, like the physical unit does even when its lowest board is mounted
        # well above the ground.
        support_bottom = min(
            -config.shelf_pose.position[2],
            min(layer_center_zs) - config.shelf_height / 2,
        )
        support_top = max(layer_center_zs) + config.shelf_height / 2
        support_height = support_top - support_bottom
    else:
        # Uniform layout: the frame spans the full nominal shelf, including any
        # omitted boards' slots (a physically removed board leaves the frame intact).
        support_bottom = -config.shelf_height / 2
        support_height = (
            config.shelf_num_layers - 1
        ) * config.shelf_spacing + config.shelf_num_layers * config.shelf_height
    support_half_height = support_height / 2
    for x_offset in [
        -config.shelf_width / 2 - config.shelf_support_width / 2,
        config.shelf_width / 2 + config.shelf_support_width / 2,
    ]:
        half_extents = [
            config.shelf_support_width / 2,
            config.shelf_depth / 2,
            support_half_height,
        ]
        collision_shape_ids.append(
            p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                physicsClientId=physics_client_id,
            )
        )
        visual_shape_ids.append(
            p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=config.shelf_rgba,
                physicsClientId=physics_client_id,
            )
        )
        link_positions.append([x_offset, 0, support_bottom + support_half_height])

    num_links = len(collision_shape_ids)
    shelf_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=-1,
        basePosition=(0, 0, 0),  # set externally
        linkMasses=[0] * num_links,
        linkCollisionShapeIndices=collision_shape_ids,
        linkVisualShapeIndices=visual_shape_ids,
        linkPositions=link_positions,
        linkOrientations=[[0, 0, 0, 1]] * num_links,
        linkInertialFramePositions=[[0, 0, 0]] * num_links,
        linkInertialFrameOrientations=[[0, 0, 0, 1]] * num_links,
        linkParentIndices=[0] * num_links,
        linkJointTypes=[p.JOINT_FIXED] * num_links,
        linkJointAxis=[[0, 0, 0]] * num_links,
        physicsClientId=physics_client_id,
    )
    return shelf_id, shelf_link_ids


class CylinderShelf3DObjectCentricState(Kinematic3DObjectCentricState):
    """A state in the CylinderShelf3DEnv().

    Adds convenience methods on top of Kinematic3DObjectCentricState().
    """


class ObjectCentricCylinderShelf3DEnv(
    ObjectCentricKinematic3DRobotEnv[
        Kinematic3DObjectCentricState, CylinderShelf3DEnvConfig
    ]
):
    """PyBullet environment where tall cylinders must be picked from the ground and
    placed on a shelf."""

    def __init__(
        self,
        num_cylinders: int = 1,
        config: CylinderShelf3DEnvConfig = CylinderShelf3DEnvConfig(),
        **kwargs,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self._num_cylinders = num_cylinders

        # Create the cylinders, but their poses will be reset (with collision
        # checking) in the reset() method.
        self._cylinders: dict[str, int] = {}
        for idx in range(self._num_cylinders):
            cylinder_id = create_pybullet_cylinder(
                self.config.get_cylinder_rgba(idx),
                self.config.get_cylinder_radius(idx),
                self.config.get_cylinder_height(idx),
                physics_client_id=self.physics_client_id,
            )
            self._cylinders[f"cylinder{idx}"] = cylinder_id

        # Staging boxes: four walls each (open top, open floor), real collision
        # bodies so grasp approaches must come in from above.
        self._box_wall_ids: list[int] = []
        thickness = self.config.box_wall_thickness
        for x_lo, x_hi, y_lo, y_hi, wall_height in self.config.boxes:
            center_z = wall_height / 2
            wall_specs = [
                (
                    ((x_hi - x_lo) / 2 + 2 * thickness, thickness, center_z),
                    ((x_lo + x_hi) / 2, y_lo - thickness, center_z),
                ),
                (
                    ((x_hi - x_lo) / 2 + 2 * thickness, thickness, center_z),
                    ((x_lo + x_hi) / 2, y_hi + thickness, center_z),
                ),
                (
                    (thickness, (y_hi - y_lo) / 2, center_z),
                    (x_lo - thickness, (y_lo + y_hi) / 2, center_z),
                ),
                (
                    (thickness, (y_hi - y_lo) / 2, center_z),
                    (x_hi + thickness, (y_lo + y_hi) / 2, center_z),
                ),
            ]
            for half_extents, position in wall_specs:
                wall_id = create_pybullet_block(
                    self.config.box_rgba,
                    half_extents,
                    physics_client_id=self.physics_client_id,
                )
                set_pose(wall_id, Pose(position), self.physics_client_id)
                self._box_wall_ids.append(wall_id)

        # Create shelf.
        self._shelf_id, self._shelf_surface_ids = _create_shelf_with_omitted_layers(
            self.config, self.physics_client_id
        )
        set_pose(self._shelf_id, self.config.shelf_pose, self.physics_client_id)

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
        return CylinderShelf3DObjectCentricState

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        return self._create_state_dict([("shelf", Kinematic3DFixtureType)])

    def _reset_objects(self) -> None:
        # Sample one cylinder at a time because each has its own height, and
        # therefore its own resting z. The sampler adds each placed cylinder
        # to the collision set for the next one.
        placed_ids: set[int] = {self.robot.base.robot_id} | set(self._box_wall_ids)
        for idx in range(self._num_cylinders):
            cylinder_id = self._cylinders[f"cylinder{idx}"]
            half_height = self.config.get_cylinder_height(idx) / 2
            region = self.config.get_cylinder_init_region(idx)
            if region is not None:
                x_lb, x_ub, y_lb, y_ub = region
            else:
                x_lb, x_ub = self.config.x_lb, self.config.x_ub
                y_lb, y_ub = self.config.y_lb, self.config.y_ub
            sample_collision_free_object_poses(
                object_ids={cylinder_id},
                lb=(x_lb, y_lb, half_height),
                ub=(x_ub, y_ub, half_height),
                physics_client_id=self.physics_client_id,
                rng=self.np_random,
                other_collision_ids=placed_ids,
            )
            placed_ids.add(cylinder_id)

    def _set_object_states(self, obs: Kinematic3DObjectCentricState) -> None:
        assert isinstance(obs, CylinderShelf3DObjectCentricState)
        for cylinder_name, cylinder_id in self._cylinders.items():
            assert cylinder_id is not None
            set_pose(
                cylinder_id,
                obs.get_object_pose(cylinder_name),
                self.physics_client_id,
            )

    def _object_name_to_pybullet_id(self, object_name: str) -> int:
        if object_name == "shelf":
            return self._shelf_id
        if object_name.startswith("cylinder"):
            return self._cylinders[object_name]
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_collision_object_ids(self) -> set[int]:
        return (
            {self._shelf_id}
            | set(self._cylinders.values())
            | set(self._box_wall_ids)
        )

    def _get_movable_object_names(self) -> set[str]:
        return set(self._cylinders.keys())

    def _get_surface_object_names(self) -> set[str]:
        return {"shelf"}

    def _get_half_extents(self, object_name: str) -> tuple[float, float, float]:
        if object_name.startswith("cylinder"):
            idx = int(object_name[len("cylinder") :])
            radius = self.config.get_cylinder_radius(idx)
            return (radius, radius, self.config.get_cylinder_height(idx) / 2)
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_obs(self) -> CylinderShelf3DObjectCentricState:
        state_dict = self._create_state_dict(
            [("robot", Kinematic3DRobotType)]
            + [("shelf", Kinematic3DFixtureType)]
            + [
                ("cylinder" + str(i), Kinematic3DCuboidType)
                for i in range(self._num_cylinders)
            ]
        )
        state = create_state_from_dict(
            state_dict,
            Kinematic3DEnvTypeFeatures,
            state_cls=CylinderShelf3DObjectCentricState,
        )
        assert isinstance(state, CylinderShelf3DObjectCentricState)
        # The generic cuboid serializer tags every Kinematic3DCuboidType
        # object as a cuboid; re-tag the cylinders.
        for cylinder_name in self._cylinders:
            obj = state.get_object_from_name(cylinder_name)
            state.set(obj, "object_type", CYLINDER_OBJECT_TYPE)
        return state

    def goal_reached(self) -> bool:
        robot_gripper_pose = self._robot_arm.get_finger_state()
        if robot_gripper_pose > self.config.gripper_open_threshold:
            return False
        # Check that every cylinder stands within the shelf footprint with
        # its center at resting height on some present board.
        shelf_x, shelf_y, _ = self.config.shelf_pose.position
        surface_zs = self.config.get_layer_surface_zs()
        for idx, (_, cylinder_id) in enumerate(sorted(self._cylinders.items())):
            cylinder_pose = get_pose(cylinder_id, self.physics_client_id)
            x, y, z = cylinder_pose.position
            if abs(x - shelf_x) > self.config.shelf_width / 2:
                return False
            if abs(y - shelf_y) > self.config.shelf_depth / 2:
                return False
            resting_z_options = [
                surface_z + self.config.get_cylinder_height(idx) / 2
                for surface_z in surface_zs
            ]
            if not any(
                abs(z - resting_z) <= self.config.on_shelf_z_tolerance
                for resting_z in resting_z_options
            ):
                return False

        return True


class CylinderShelf3DEnv(ConstantObjectKinDEREnv):
    """Cylinder shelf 3D env with a constant number of objects."""

    def __init__(self, num_cylinders: int = 1, **kwargs) -> None:
        self._num_cylinders = num_cylinders
        super().__init__(num_cylinders=num_cylinders, **kwargs)

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricKinematic3DRobotEnv:
        return ObjectCentricCylinderShelf3DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        constant_objects = ["robot", "shelf"]
        for obj in exemplar_state:
            if obj.name.startswith("cylinder"):
                constant_objects.append(obj.name)
        return constant_objects

    def _create_env_markdown_description(self) -> str:
        """Create environment description."""
        # pylint: disable=line-too-long
        return """A 3D environment where the goal is to pick up tall cylinders from the ground and place them onto a shelf. The cylinders are tall enough that they must be grasped from the side rather than from the top."""

    def _create_variant_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return "The number of cylinders differs between environment variants. For example, CylinderShelf3D-o1 has 1 cylinder, while CylinderShelf3D-o3 has 3 cylinders of different heights."

    def _create_variant_specific_description(self) -> str:
        if self._num_cylinders == 1:
            return "This variant has 1 cylinder to place on the shelf."
        return (
            f"This variant has {self._num_cylinders} cylinders of different "
            "heights to place on the shelf."
        )

    def _create_reward_markdown_description(self) -> str:
        """Create reward description."""
        # pylint: disable=line-too-long
        return """The reward is -1 per timestep to encourage efficient task completion. The episode terminates successfully when all cylinders stand within the shelf footprint at resting height on one of the shelf boards and the gripper is closed. The gripper must be closed to prevent accidental "success" while a cylinder is still being held above the shelf."""

    def _create_references_markdown_description(self) -> str:
        """Create references description."""
        # pylint: disable=line-too-long
        return """This is a very common kind of environment. The background is adapted from the [Replica dataset](https://arxiv.org/abs/1906.05797) (Straub et al., 2019)."""
