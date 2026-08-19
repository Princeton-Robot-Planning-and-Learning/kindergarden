"""Base classes for MuJoCo objects and fixtures."""

from __future__ import annotations

import abc
import xml.etree.ElementTree as ET
from typing import Callable, TypeVar, Union, overload

import numpy as np
from numpy.typing import NDArray
from relational_structs import Object

from kinder.envs.dynamic3d import utils
from kinder.envs.dynamic3d.mujoco_utils import MujocoEnv
from kinder.envs.dynamic3d.object_types import (
    MujocoFixtureObjectType,
    MujocoMovableObjectType,
)

# Type variables for decorator type preservation
FixtureT = TypeVar("FixtureT", bound="MujocoFixture")
ObjectT = TypeVar("ObjectT", bound="MujocoObject")


class Region:
    """Represents a region in MuJoCo with site-based bounding box.

    The bounding box is always derived from the site element's pos and size attributes,
    ensuring consistency between the region's geometric definition and its XML
    representation.
    """

    def __init__(
        self,
        name: str,
        rgba: list[float] | None = None,
        site_element: ET.Element | None = None,
        env: MujocoEnv | None = None,
        parent_pos: NDArray[np.float32] | None = None,
        parent_yaw: float = 0.0,
    ) -> None:
        """Initialize a Region.

        Args:
            name: Name of the region
            rgba: RGBA color for visualization
            site_element: MuJoCo site XML element representing this region
            env: Optional MujocoEnv instance for getting absolute site position from
                simulation
            parent_pos: Position of the parent body [x, y, z] for coordinate
                transformation.
                        Used when sim is not available.
            parent_yaw: Yaw angle of the parent body in radians. Used to rotate region
                coordinates from parent-relative to world frame when sim unavailable.
        """
        self.name = name
        self.rgba = rgba if rgba is not None else [1.0, 0.0, 0.0, 0.0]
        self.site_element = site_element
        self.env = env
        self.parent_pos = (
            parent_pos if parent_pos is not None else np.array([0.0, 0.0, 0.0])
        )
        self.parent_yaw = parent_yaw

        # Get bbox in world frame at origin
        if self.site_element is None:
            self.bbox_at_origin = np.array([-0.0, -0.0, -0.0, 0.0, 0.0, 0.0])
        else:
            size_str = self.site_element.get("size", "0 0 0")
            size = np.array([float(v) for v in size_str.split()])
            self.bbox_at_origin = np.concatenate((-size, size))

    @property
    def bbox(self) -> list[float]:
        """Compute bounding box in world coordinates.

        First tries to get the site position from simulation. If unavailable,
        translates the relative bounding box by parent_pos and rotates by parent_yaw.

        Returns:
            Bounding box as [x_min, y_min, z_min, x_max, y_max, z_max] in world
            coordinates
        """
        if self.site_element is None:
            raise ValueError(
                f"Cannot compute bbox for region '{self.name}' without site_element"
            )

        # Get size from site element
        size_str = self.site_element.get("size", "0 0 0")
        size = np.array([float(v) for v in size_str.split()])

        # Try to get absolute position from simulation first
        if (
            self.env is not None
            and hasattr(self.env, "sim")
            and self.env.sim is not None
        ):
            site_name = self.site_element.get("name")
            if site_name is not None:
                try:
                    rot_mat = self.env.sim.data.get_site_xmat(site_name)
                    pos = self.env.sim.data.get_site_xpos(site_name)

                    # Compute current bbox
                    bbox_list: list[float] = self.bbox_at_origin.tolist()
                    rot_mat_float64: NDArray[np.float64] = np.array(
                        rot_mat, dtype=np.float64
                    )
                    rotated_bbox: list[float] = utils.rotate_bounding_box_3d(
                        bbox_list, rot_mat_float64
                    )
                    bbox_curr = np.array(rotated_bbox, dtype=np.float64)
                    bbox_curr += np.tile(pos, 2)
                    return bbox_curr.tolist()

                except ValueError:
                    # Fall through to XML-based computation
                    pass

        # Use XML-based computation with parent_pos and parent_yaw
        # Get site position from XML
        pos_str = self.site_element.get("pos", "0 0 0")
        relative_pos = np.array([float(v) for v in pos_str.split()])

        # Compute relative bounding box (site position +/- size)
        relative_bbox = [
            relative_pos[0] - size[0],  # x_min
            relative_pos[1] - size[1],  # y_min
            relative_pos[2] - size[2],  # z_min
            relative_pos[0] + size[0],  # x_max
            relative_pos[1] + size[1],  # y_max
            relative_pos[2] + size[2],  # z_max
        ]

        # Translate bounding box by parent_pos
        translated_bbox = utils.translate_bounding_box(relative_bbox, self.parent_pos)

        # Rotate bounding box by parent_yaw around the center of the parent's position
        center_x = self.parent_pos[0]
        center_y = self.parent_pos[1]
        rotated_bbox = utils.rotate_bounding_box_2d(
            translated_bbox, self.parent_yaw, (center_x, center_y)
        )

        return rotated_bbox

    def check_in_region(
        self,
        position: NDArray[np.float32],
        env: MujocoEnv | None = None,
    ) -> bool:
        """Check if a position is within this region's bounding box.

        Since bbox always returns world coordinates (via simulation or transform),
        we directly compare against the world-frame bounding box.

        Args:
            position: Position to check as [x, y, z] in world coordinates
            env: Optional MujocoEnv instance. If provided, uses env's simulation
                 to compute the absolute site position. Otherwise uses parent_pos
                 and parent_yaw for coordinate transformation.

        Returns:
            True if position is within the region, False otherwise
        """
        # Temporarily update env if provided
        original_env = self.env
        if env is not None:
            self.env = env

        try:
            # Get the bounding box in world coordinates
            bbox = self.bbox
            x_min, y_min, z_min, x_max, y_max, z_max = bbox

            # Check if position is within the bounds
            return bool(
                position[0] >= x_min
                and position[0] <= x_max
                and position[1] >= y_min
                and position[1] <= y_max
                and position[2] >= z_min
                and position[2] <= z_max
            )
        finally:
            # Restore original env
            self.env = original_env

    def visualize_region(self) -> None:
        """Visualize this region (site already created, nothing to do)."""
        # Visualization is handled at creation time via site element


REGISTERED_FIXTURES: dict[str, type[MujocoFixture]] = {}
REGISTERED_OBJECTS: dict[str, type[MujocoObject]] = {}


def register_fixture(cls: type[FixtureT]) -> type[FixtureT]:
    """Register fixture classes for TidyBot environments."""
    REGISTERED_FIXTURES[cls.__name__.lower()] = cls
    return cls


@overload
def register_object(cls: type[ObjectT]) -> type[ObjectT]: ...


@overload
def register_object(
    cls: None = None, name: str | None = None
) -> Callable[[type[ObjectT]], type[ObjectT]]: ...


def register_object(
    cls: type[ObjectT] | None = None, name: str | None = None
) -> type[ObjectT] | Callable[[type[ObjectT]], type[ObjectT]]:
    """Register object classes for TidyBot environments.

    Can be used as:
    - @register_object (uses class name in lowercase)
    - @register_object(name='custom_name') (uses provided name)

    Args:
        cls: The class to register (when used without parentheses)
        name: Optional name to register with (defaults to lowercase class name)

    Returns:
        The registered class or a decorator function
    """

    def decorator(c: type[ObjectT]) -> type[ObjectT]:
        registry_name = name if name is not None else c.__name__.lower()
        REGISTERED_OBJECTS[registry_name] = c
        c.REGISTERED_NAME = registry_name  # type: ignore[attr-defined]
        return c

    # If used without parentheses: @register_object
    if cls is not None:
        return decorator(cls)

    # If used with parentheses: @register_object() or @register_object(name='...')
    return decorator


def get_fixture_class(name: str) -> type[MujocoFixture]:
    """Get a fixture class by name.

    Args:
        name: Name of the fixture class (case-insensitive)

    Returns:
        The fixture class

    Raises:
        ValueError: If the fixture class is not found
    """
    name_lower = name.lower()
    if name_lower not in REGISTERED_FIXTURES:
        available_fixtures = list(REGISTERED_FIXTURES.keys())
        raise ValueError(
            f"Fixture class '{name}' not found. "
            f"Available fixtures: {available_fixtures}"
        )
    return REGISTERED_FIXTURES[name_lower]


def get_object_class(name: str) -> type[MujocoObject]:
    """Get an object class by name.

    Args:
        name: Name of the object class (case-insensitive)

    Returns:
        The object class

    Raises:
        ValueError: If the object class is not found
    """
    name_lower = name.lower()
    if name_lower not in REGISTERED_OBJECTS:
        available_objects = list(REGISTERED_OBJECTS.keys())
        raise ValueError(
            f"Object class '{name}' not found. "
            f"Available objects: {available_objects}"
        )
    return REGISTERED_OBJECTS[name_lower]


class MujocoObject:
    """Base class for MuJoCo objects with position and orientation control."""

    def __init__(
        self,
        name: str,
        env: MujocoEnv | None = None,
        options: dict | None = None,
    ) -> None:
        """Initialize a MujocoObject.

        Args:
            name: Name of the object body in the XML
            env: Reference to the environment (needed for position get/set operations)
            options: Optional dictionary of configuration options. Can include:
                - "regions": Optional dictionary of regions for this object
        """
        self.name = name
        self.joint_name = f"{name}_joint"
        self.env = env
        self.options = options if options is not None else {}
        self.regions = self.options.get("regions")

        # Create the corresponding Object for state representation key
        self.symbolic_object = Object(self.name, MujocoMovableObjectType)

        # Create regions if defined
        self.region_objects: dict[str, list[Region]] = {}

        self.xml_element: ET.Element  # To be defined in subclasses

    def _create_regions(self) -> None:
        """Create Region objects with site elements for each region.

        Each region's ranges are converted to 3D bounding boxes [x_min, y_min, z_min,
        x_max, y_max, z_max]. A 4-value range [x_start, y_start, x_end, y_end] spans a
        small, fixed height above the object's own local origin, as before; a 6-value
        range [x_start, y_start, z_start, x_end, y_end, z_end] carries real z bounds
        instead, for a region -- e.g. a goal box on a container -- that needs real
        interior height.
        """
        assert self.regions is not None, "Regions must be defined"
        assert (
            self.xml_element is not None
        ), "XML element must be defined to create regions"
        placement_threshold = 0.01  # 1cm tolerance for placement

        for region_name, region_config in self.regions.items():
            region_list: list[Region] = []

            for region_idx, region_range in enumerate(region_config["ranges"]):
                if len(region_range) == 4:
                    x_start, y_start, x_end, y_end = region_range
                    z_start, z_end = 0.0, 0.0
                elif len(region_range) == 6:
                    x_start, y_start, z_start, x_end, y_end, z_end = region_range
                else:
                    raise ValueError(
                        f"Region range must have 4 or 6 values "
                        f"[x_start, y_start, x_end, y_end] or "
                        f"[x_start, y_start, z_start, x_end, y_end, z_end], "
                        f"got {len(region_range)}"
                    )

                # Create 3D bounding box with z range and tolerance on all bounds
                bbox = [
                    x_start - placement_threshold,
                    y_start - placement_threshold,
                    z_start - placement_threshold,
                    x_end + placement_threshold,
                    y_end + placement_threshold,
                    z_end + placement_threshold,
                ]

                # Calculate center and half-sizes for MuJoCo box site
                x_min, y_min, z_min, x_max, y_max, z_max = bbox
                region_center_x = (x_min + x_max) / 2
                region_center_y = (y_min + y_max) / 2
                region_center_z = (z_min + z_max) / 2
                region_size_x = (x_max - x_min) / 2
                region_size_y = (y_max - y_min) / 2
                region_size_z = (z_max - z_min) / 2

                # Create site element for the region
                site = ET.Element("site")
                site.set("name", f"{self.name}_{region_name}_region_{region_idx}")
                site.set("type", "box")
                site.set("size", f"{region_size_x} {region_size_y} {region_size_z}")
                site.set(
                    "pos", f"{region_center_x} {region_center_y} {region_center_z}"
                )
                rgba_values = region_config.get("rgba", [1.0, 0.0, 0.0, 0.0])
                site.set("rgba", " ".join(map(str, rgba_values)))
                site.set("group", "0")

                # Create Region object
                # Get object position if environment is available
                obj_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                obj_yaw = 0.0
                if self.env is not None:
                    try:
                        obj_pos, obj_quat = self.env.get_joint_pos_quat(self.joint_name)
                        obj_pos = np.array(obj_pos, dtype=np.float32)
                        obj_yaw = utils.quat_to_yaw(obj_quat)
                    except Exception:
                        pass  # Use defaults if position can't be retrieved

                site_name = site.get("name")
                assert site_name is not None, "Site name must be set"
                region = Region(
                    name=site_name,
                    rgba=rgba_values,
                    site_element=site,
                    env=self.env,
                    parent_pos=obj_pos,
                    parent_yaw=obj_yaw,
                )
                region_list.append(region)

                # Append site element to xml_element
                self.xml_element.append(site)

            self.region_objects[region_name] = region_list

    def get_position(self) -> NDArray[np.float32]:
        """Get the object's current position.

        Returns:
            Position as [x, y, z] array

        Raises:
            ValueError: If environment is not set
        """
        if self.env is None:
            raise ValueError("Environment must be set to get position")

        pos, _ = self.env.get_joint_pos_quat(self.joint_name)
        return pos

    def get_orientation(self) -> NDArray[np.float32]:
        """Get the object's current orientation.

        Returns:
            Orientation as quaternion [w, x, y, z] array

        Raises:
            ValueError: If environment is not set
        """
        if self.env is None:
            raise ValueError("Environment must be set to get orientation")

        _, quat = self.env.get_joint_pos_quat(self.joint_name)
        return quat

    def get_pose(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Get the object's current position and orientation.

        Returns:
            Tuple of (position, quaternion)

        Raises:
            ValueError: If environment is not set
        """
        if self.env is None:
            raise ValueError("Environment must be set to get pose")

        return self.env.get_joint_pos_quat(self.joint_name)

    def set_position(self, position: Union[list[float], NDArray[np.float32]]) -> None:
        """Set the object's position.

        Args:
            position: New position as [x, y, z]

        Raises:
            ValueError: If environment is not set
        """
        if self.env is None:
            raise ValueError("Environment must be set to set position")

        # Get current orientation to preserve it
        _, current_quat = self.env.get_joint_pos_quat(self.joint_name)

        # Set new position with current orientation
        self.env.set_joint_pos_quat(self.joint_name, np.array(position), current_quat)

    def set_orientation(
        self, quaternion: Union[list[float], NDArray[np.float32]]
    ) -> None:
        """Set the object's orientation.

        Args:
            quaternion: New orientation as quaternion [w, x, y, z]

        Raises:
            ValueError: If environment is not set
        """
        if self.env is None:
            raise ValueError("Environment must be set to set orientation")

        # Get current position to preserve it
        current_pos, _ = self.env.get_joint_pos_quat(self.joint_name)

        # Set new orientation with current position
        self.env.set_joint_pos_quat(self.joint_name, current_pos, np.array(quaternion))

    def set_pose(
        self,
        position: Union[list[float], NDArray[np.float32]],
        quaternion: Union[list[float], NDArray[np.float32]],
    ) -> None:
        """Set the object's position and orientation.

        Args:
            position: New position as [x, y, z]
            quaternion: New orientation as quaternion [w, x, y, z]

        Raises:
            ValueError: If environment is not set
        """
        if self.env is None:
            raise ValueError("Environment must be set to set pose")

        self.env.set_joint_pos_quat(
            self.joint_name, np.array(position), np.array(quaternion)
        )

    def set_velocity(
        self,
        linear_velocity: Union[list[float], NDArray[np.float32]],
        angular_velocity: Union[list[float], NDArray[np.float32]],
    ) -> None:
        """Set the object's linear and angular velocity.

        Args:
            linear_velocity: New linear velocity as [vx, vy, vz]
            angular_velocity: New angular velocity as [wx, wy, wz]

        Raises:
            ValueError: If environment is not set
        """
        if self.env is None:
            raise ValueError("Environment must be set to set velocity")

        self.env.set_joint_vel(
            self.joint_name, np.array(linear_velocity), np.array(angular_velocity)
        )

    @abc.abstractmethod
    def get_bounding_box_dimensions(self) -> tuple[float, float, float]:
        """Get the bounding box dimensions for this object.

        These bounding box dimensions are independent from the object pose.
        """

    @staticmethod
    @abc.abstractmethod
    def get_bounding_box_from_config(
        pos: NDArray[np.float32], object_config: dict[str, str | float]
    ) -> list[float]:
        """Get the object's bounding box in world coordinates.

        Args:
            pos: Position of the object as [x, y, z] array
            object_config: Dictionary containing object configuration parameters

        Returns:
            Bounding box as [x_min, y_min, z_min, x_max, y_max, z_max] array
        """

    def _get_object_centric_data(self) -> dict[str, float]:
        """Get the object's current data.

        Returns:
            dict with current position and orientation

        Raises:
            ValueError: If environment is not set
        """
        if self.env is None:
            raise ValueError("Environment must be set to get state")

        pos, quat = self.env.get_joint_pos_quat(self.joint_name)
        linear_vel, angular_vel = self.env.get_joint_vel(self.joint_name)
        bb_x, bb_y, bb_z = self.get_bounding_box_dimensions()

        # Create and return the data
        obj_data = {
            "x": pos[0],
            "y": pos[1],
            "z": pos[2],
            "qw": quat[0],
            "qx": quat[1],
            "qy": quat[2],
            "qz": quat[3],
            "vx": linear_vel[0],
            "vy": linear_vel[1],
            "vz": linear_vel[2],
            "wx": angular_vel[0],
            "wy": angular_vel[1],
            "wz": angular_vel[2],
            "bb_x": bb_x,
            "bb_y": bb_y,
            "bb_z": bb_z,
        }
        return obj_data

    def get_object_centric_state(self) -> dict[Object, dict[str, float]]:
        """Get object-centric state with symbolic object as key.

        Returns:
            Dictionary mapping symbolic_object to its data for use in state creation
        """
        obj_data = self._get_object_centric_data()
        return {self.symbolic_object: obj_data}

    def check_in_region(
        self,
        position: NDArray[np.float32],
        region_name: str,
        env: MujocoEnv | None = None,
    ) -> bool:
        """Check if a given position is within the specified region.

        Args:
            position: Position as [x, y, z] array in world coordinates
            region_name: Name of the region to check
            env: Optional MujocoEnv instance for computing absolute site positions.
                 If not provided, uses object's env or falls back to parent_pos/yaw.

        Returns:
            True if the position is within the specified region, False otherwise

        Raises:
            ValueError: If regions are not defined or region not found
        """
        if self.regions is None:
            raise ValueError("Regions must be defined for this object")
        if region_name not in self.region_objects:
            raise ValueError(f"Region {region_name} not found in object regions")

        # Use provided env, otherwise fall back to object's env
        check_env = env if env is not None else self.env

        # Check if position is in any of the region objects
        region_list = self.region_objects[region_name]
        for region in region_list:
            if region.check_in_region(position, check_env):
                return True

        return False

    def visualize_regions(self) -> None:
        """Visualize the object's regions in the MuJoCo environment.

        This method is a no-op since regions are now added to the XML during
        _create_regions().
        """
        if self.regions is None:
            return

        for region_list in self.region_objects.values():
            for region in region_list:
                region.visualize_region()


class MujocoFixture(abc.ABC):
    """Base class for MuJoCo fixtures (static objects).

    These are non-movable objects, like tables, that cannot be manipulated by the robot,
    and cannot change position/orientation after sim initialization.
    """

    def __init__(
        self,
        name: str,
        fixture_config: dict[str, str | float],
        position: list[float] | NDArray[np.float32],
        yaw: float,
        regions: dict | None = None,
        env: MujocoEnv | None = None,
    ) -> None:
        """Initialize a MujocoFixture.

        Args:
            name: Name of the fixture body in the XML
            fixture_config: Dictionary containing fixture configuration
            position: Position of the fixture as [x, y, z]
            yaw: Yaw orientation of the fixture in radians
            env: Reference to the environment (needed for accessing joint data)
        """
        self.name = name
        self.fixture_config = fixture_config
        self.position = position
        self.yaw = yaw
        self.regions = regions
        self.env = env

        # Create the corresponding Object for state representation key
        self.symbolic_object = Object(self.name, MujocoFixtureObjectType)

        # Create regions if defined (to be called by subclasses after initialization)
        self.region_objects: dict[str, list[Region]] = {}

        self.xml_element: ET.Element  # To be defined in subclasses

    def _create_regions(self) -> None:
        """Create Region objects with site elements for each region.

        To be implemented by subclasses.
        """
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement _create_regions()"
        )

    def get_position(self) -> NDArray[np.float32]:
        """Get the fixture's position.

        Returns:
            Position as [x, y, z] array
        """
        return np.array(self.position)

    def get_orientation(self) -> list[float]:
        """Get the fixture's orientation.

        Returns:
            Orientation as quaternion [w, x, y, z] list
        """
        return utils.convert_yaw_to_quaternion(self.yaw)

    @staticmethod
    @abc.abstractmethod
    def get_bounding_box_from_config(
        pos: NDArray[np.float32], fixture_config: dict[str, str | float]
    ) -> list[float]:
        """Get the fixture's bounding box in world coordinates.

        Args:
            pos: Position of the fixture as [x, y, z] array
            fixture_config: Dictionary containing fixture configuration parameters

        Returns:
            Bounding box as [x_min, y_min, z_min, x_max, y_max, z_max] array
        """

    def _get_object_centric_data(self) -> dict[str, float]:
        """Get the object's current data.

        Returns:
            dict with current position and orientation

        Raises:
            ValueError: If environment is not set
        """
        pos = self.get_position()
        quat = self.get_orientation()

        # Create and return the data
        obj_data = {
            "x": pos[0],
            "y": pos[1],
            "z": pos[2],
            "qw": quat[0],
            "qx": quat[1],
            "qy": quat[2],
            "qz": quat[3],
        }
        return obj_data

    def get_object_centric_state(self) -> dict[Object, dict[str, float]]:
        """Get object-centric state with symbolic object as key.

        Returns:
            Dictionary mapping symbolic_object to its data for use in state creation
        """
        obj_data = self._get_object_centric_data()
        return {self.symbolic_object: obj_data}

    @abc.abstractmethod
    def _create_xml_element(self) -> ET.Element:
        """Create the XML Element for this fixture.

        Returns:
            ET.Element representing the fixture body
        """

    @abc.abstractmethod
    def sample_pose_in_region(
        self,
        region_name: str,
        np_random: np.random.Generator,
    ) -> tuple[float, float, float, float]:
        """Sample a pose (x, y, z, yaw) uniformly randomly from one of the provided
        regions.

        Args:
            region_name: Name of the region to sample from
            np_random: Random number generator

        Returns:
            Tuple of (x, y, z, yaw) coordinates in world coordinates (offset by
            fixture position), where yaw is in radians. The yaw range is read from
            self.regions[region_name]["yaw_ranges"] if it exists, otherwise
            defaults to (0.0, 360.0) degrees.

        Raises:
            ValueError: If regions list is empty or if any region has invalid bounds
        """

    @abc.abstractmethod
    def check_in_region(
        self,
        position: NDArray[np.float32],
        region_name: str,
        env: MujocoEnv | None = None,
    ) -> bool:
        """Check if a given position is within the specified region.

        Args:
            position: Position as [x, y, z] array in world coordinates
            region_name: Name of the region to check
            env: Optional MujocoEnv instance for computing absolute site positions.
        Returns:
            True if the position is within the specified region, False otherwise
        """

    def visualize_regions(self) -> None:
        """Visualize the fixture's regions in the MuJoCo environment.

        This method adds visual elements to the MuJoCo XML to represent the regions
        defined for this fixture.
        """
        if self.regions is None:
            return

        for region_list in self.region_objects.values():
            for region in region_list:
                region.visualize_region()
                if region.site_element is not None:
                    self.xml_element.append(region.site_element)


class MujocoGround:
    """A ground fixture for simple region sampling on the ground plane."""

    def __init__(
        self,
        regions: dict | None = None,
        worldbody: ET.Element | None = None,
    ) -> None:
        """Initialize a Ground object.

        Args:
            regions: Dictionary of regions defined on the ground plane
            worldbody: The MuJoCo worldbody XML element to add sites to
        """
        self.name = "ground"
        self.regions = regions
        self.worldbody = worldbody
        self.position = np.array([0.0, 0.0, 0.0])  # Ground at origin
        self.ground_placement_threshold = 0.05  # Default ground region threshold

        # Create regions if defined
        self.region_objects: dict[str, list[Region]] = {}
        if self.regions is not None:
            self._create_regions()

    def _create_regions(self) -> None:
        """Create Region objects with site elements for ground regions.

        Sites are added directly to the worldbody if it was provided during init.
        """
        assert self.regions is not None, "Regions must be defined"

        for region_name, region_config in self.regions.items():
            region_list: list[Region] = []

            for region_idx, region_range in enumerate(region_config["ranges"]):
                # Support both 4-value (x, y bounds) and 6-value (x, y, z bounds) ranges
                if len(region_range) == 4:
                    x_start, y_start, x_end, y_end = region_range
                    z_start, z_end = 0.0, self.ground_placement_threshold
                elif len(region_range) == 6:
                    x_start, y_start, z_start, x_end, y_end, z_end = region_range
                else:
                    raise ValueError(
                        f"Region range must have 4 or 6 values "
                        f"[x_start, y_start, x_end, y_end] or "
                        f"[x_start, y_start, z_start, x_end, y_end, z_end], "
                        f"got {len(region_range)}"
                    )

                # Create 3D bounding box on ground surface.
                # Sites must not go below ground (z >= 0), span from z_start to z_end
                bbox = [
                    x_start - self.ground_placement_threshold,
                    y_start - self.ground_placement_threshold,
                    max(0.0, z_start - self.ground_placement_threshold),
                    x_end + self.ground_placement_threshold,
                    y_end + self.ground_placement_threshold,
                    z_end + self.ground_placement_threshold,
                ]

                # Calculate center and half-sizes for MuJoCo box site
                x_min, y_min, z_min, x_max, y_max, z_max = bbox
                region_center_x = (x_min + x_max) / 2
                region_center_y = (y_min + y_max) / 2
                region_center_z = (z_min + z_max) / 2
                region_size_x = (x_max - x_min) / 2
                region_size_y = (y_max - y_min) / 2
                region_size_z = (z_max - z_min) / 2

                # Create site element for the region (in worldbody)
                site = ET.Element("site")
                site.set("name", f"{self.name}_{region_name}_region_{region_idx}")
                site.set("type", "box")
                site.set("size", f"{region_size_x} {region_size_y} {region_size_z}")
                site.set(
                    "pos", f"{region_center_x} {region_center_y} {region_center_z}"
                )
                rgba_values = region_config.get("rgba", [1.0, 0.0, 0.0, 0.0])
                site.set("rgba", " ".join(map(str, rgba_values)))
                site.set("group", "0")

                # Create Region object
                site_name = site.get("name")
                assert site_name is not None, "Site name must be set"
                region = Region(
                    name=site_name,
                    rgba=rgba_values,
                    site_element=site,
                    parent_pos=np.array(self.position, dtype=np.float32),
                    parent_yaw=0.0,  # Ground doesn't rotate
                )
                region_list.append(region)

                # Add site element to worldbody if provided
                if self.worldbody is not None:
                    self.worldbody.append(site)

            self.region_objects[region_name] = region_list

    def sample_pose_in_region(
        self,
        region_name: str,
        np_random: np.random.Generator,
    ) -> tuple[float, float, float, float]:
        """Sample a pose (x, y, z, yaw) uniformly randomly from one of the provided
        regions.

        For ground, this samples on the ground plane surface. Note that we are not
        using Region objects for this sampling; instead, we directly sample from the
        ranges specified in self.regions[region_name]["ranges"].

        Args:
            region_name: Name of the region to sample from
            np_random: Random number generator

        Returns:
            Tuple of (x, y, z, yaw) coordinates in world coordinates (offset by ground
            position), where yaw is in radians. The yaw range is read from
            self.regions[region_name]["yaw_ranges"] if it exists, otherwise
            defaults to (0.0, 360.0) degrees.

        Raises:
            ValueError: If regions list is empty or if any region has invalid bounds
        """
        assert self.regions is not None, "Regions must be defined"
        region_config = self.regions[region_name]

        # Randomly select one of the regions
        selected_range_index = np_random.choice(len(region_config["ranges"]))

        # Validate and unpack the selected region (supports 4-value and 6-value ranges)
        selected_range = region_config["ranges"][selected_range_index]
        if len(selected_range) == 4:
            x_start, y_start, x_end, y_end = selected_range  # type: ignore[misc]
            z_start, z_end = (
                0,
                2 * self.ground_placement_threshold,
            )  # Default: ground surface
        elif len(selected_range) == 6:
            (
                x_start,
                y_start,
                z_start,
                x_end,
                y_end,
                z_end,
            ) = selected_range  # type: ignore[misc]
        else:
            raise ValueError(
                f"Each region must have 4 or 6 values "
                f"[x_start, y_start, x_end, y_end] or "
                f"[x_start, y_start, x_end, y_end, z_start, z_end], "
                f"got {len(selected_range)}"
            )

        # Validate bounds
        if x_start > x_end:
            raise ValueError(f"x_start ({x_start}) must be less than x_end ({x_end})")
        if y_start > y_end:
            raise ValueError(f"y_start ({y_start}) must be less than y_end ({y_end})")
        if z_start > z_end:
            raise ValueError(f"z_start ({z_start}) must be less than z_end ({z_end})")

        # Sample uniformly within the selected region
        x = np_random.uniform(x_start, x_end)
        y = np_random.uniform(y_start, y_end)
        z = np_random.uniform(z_start, z_end)  # Sample z from the specified range

        # Sample yaw from the specified range (convert from degrees to radians)
        yaw_range = (0.0, 360.0)  # Default range
        if "yaw_ranges" in region_config:
            selected_yaw_range = region_config["yaw_ranges"][selected_range_index]
            yaw_range = selected_yaw_range
        yaw_deg = np_random.uniform(yaw_range[0], yaw_range[1])
        yaw = np.radians(yaw_deg)

        # Offset by the ground's position to get world coordinates
        world_x = x + self.position[0]
        world_y = y + self.position[1]
        world_z = z + self.position[2]

        return (world_x, world_y, world_z, yaw)

    def check_in_region(
        self,
        position: NDArray[np.float32],
        region_name: str,
        env: MujocoEnv | None = None,
    ) -> bool:
        """Check if a given position is within the specified region on the ground.

        Args:
            position: Position as [x, y, z] array in world coordinates
            region_name: Name of the region to check
            env: Optional MujocoEnv instance for computing absolute site positions.
        Returns:
            True if the position is within the specified region, False otherwise
        """
        if region_name not in self.region_objects:
            raise ValueError(f"Region {region_name} not found in ground regions")

        # Check if position is in any of the region objects
        region_list = self.region_objects[region_name]
        for region in region_list:
            if region.check_in_region(position, env):
                return True

        return False

    def visualize_regions(self) -> None:
        """Visualize the ground's regions.

        This is now a no-op since sites are added directly to the worldbody during
        _create_regions().
        """

    def __str__(self) -> str:
        """String representation of the ground."""
        return f"Ground(name='{self.name}', " f"position={self.position})"

    def __repr__(self) -> str:
        """Detailed string representation of the ground."""
        return f"Ground(name='{self.name}', " f"position={self.position})"
