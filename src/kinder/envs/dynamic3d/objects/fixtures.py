"""Fixture classes for TidyBot environments (Table, Cupboard, etc.)."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from relational_structs import Object

from kinder.envs.dynamic3d import utils
from kinder.envs.dynamic3d.mujoco_utils import MujocoEnv
from kinder.envs.dynamic3d.object_types import MujocoDrawerObjectType
from kinder.envs.dynamic3d.objects.base import (
    MujocoFixture,
    Region,
    register_fixture,
)


@register_fixture
class Table(MujocoFixture):
    """A table fixture."""

    DEFAULT_REGION_HEIGHT: float = 0.2  # 20cm height for regions
    DEFAULT_REGION_Z_OFFSET: float = 0.05  # 5cm offset above table surface
    # (assumes max possible half-height of any object is 5cm)

    # Default RGBA colors for table components
    default_rgba_table_top: list[float] = [0.8, 0.6, 0.4, 1.0]
    default_rgba_table_leg: list[float] = [0.6, 0.4, 0.2, 1.0]

    def __init__(
        self,
        name: str,
        fixture_config: dict[str, str | float],
        position: list[float] | NDArray[np.float32],
        yaw: float,
        regions: dict | None = None,
        env: MujocoEnv | None = None,
    ) -> None:
        """Initialize a Table object.

        Args:
            name: Name of the table body in the XML
            fixture_config: Dictionary containing table configuration with keys:
                - "shape": Shape of the table - "rectangle" or "circle"
                - "height": Table height in meters
                - "thickness": Table top thickness in meters
                - "length": Total table length in meters (for rectangle)
                - "width": Total table width in meters (for rectangle)
                - "diameter": Diameter of circular table in meters (for circle)
            position: Position of the table as [x, y, z]
            yaw: Yaw orientation of the table in radians
            env: Reference to the environment (needed for accessing joint data)
        """
        # Initialize base class
        super().__init__(name, fixture_config, position, yaw, regions, env)

        # Parse table configuration
        self.table_shape = str(self.fixture_config["shape"])
        self.table_height = float(self.fixture_config["height"])
        self.table_thickness = float(self.fixture_config["thickness"])
        self.leg_inset = 0.05

        # Parse RGBA colors from fixture config or use class defaults
        self.rgba_table_top: list[float] = self.fixture_config.get(
            "rgba_table_top", Table.default_rgba_table_top
        )  # type: ignore
        self.rgba_table_leg: list[float] = self.fixture_config.get(
            "rgba_table_leg", Table.default_rgba_table_leg
        )  # type: ignore

        # Optional parameters
        self.table_length: float | None = None
        self.table_width: float | None = None
        self.table_diameter: float | None = None

        # Shape-specific parameters
        if self.table_shape == "rectangle":
            self.table_length = float(self.fixture_config["length"])
            self.table_width = float(self.fixture_config["width"])
        elif self.table_shape == "circle":
            self.table_diameter = float(self.fixture_config["diameter"])
        else:
            raise ValueError(
                f"Unknown table shape: {self.table_shape}. "
                f"Must be 'rectangle' or 'circle'"
            )

        # Create the XML element
        self.xml_element = self._create_xml_element()

        # Create regions after all attributes are initialized
        if self.regions is not None:
            self._create_regions()

    def _create_xml_element(self) -> ET.Element:
        """Create the XML Element for this table.

        Returns:
            ET.Element representing the table body
        """
        # Create table body element
        table_body = ET.Element("body")
        table_body.set("name", self.name)
        position_str = " ".join(str(x) for x in self.position)
        table_body.set("pos", position_str)
        ori_quat = utils.convert_yaw_to_quaternion(self.yaw)
        orientation_str = " ".join(str(x) for x in ori_quat)
        table_body.set("quat", orientation_str)

        if self.table_shape == "rectangle":
            assert self.table_length is not None
            assert self.table_width is not None

            # Calculate MuJoCo geom sizes (half the actual dimensions)
            table_half_length = float(self.table_length) / 2
            table_half_width = float(self.table_width) / 2
            table_half_thickness = self.table_thickness / 2
            leg_radius = 0.02
            leg_half_height = (self.table_height - self.table_thickness) / 2

            # Calculate leg positions (inset from edges)
            leg_x_offset = table_half_length - self.leg_inset
            leg_y_offset = table_half_width - self.leg_inset
            leg_z_pos = leg_half_height  # Center of leg cylinder
            table_top_z_pos = self.table_height - table_half_thickness

            # Create rectangular table top geom
            table_top = ET.SubElement(table_body, "geom")
            table_top.set("name", f"{self.name}_top")
            table_top.set("type", "box")
            table_top.set(
                "size",
                f"{table_half_length} {table_half_width} {table_half_thickness}",
            )
            table_top.set("pos", f"0 0 {table_top_z_pos}")
            table_top.set("rgba", " ".join(map(str, self.rgba_table_top)))

            # Create table legs at four corners
            leg_positions = [
                (
                    f"{leg_x_offset} {leg_y_offset} {leg_z_pos}",
                    f"{self.name}_leg1",
                ),
                (
                    f"{-leg_x_offset} {leg_y_offset} {leg_z_pos}",
                    f"{self.name}_leg2",
                ),
                (
                    f"{leg_x_offset} {-leg_y_offset} {leg_z_pos}",
                    f"{self.name}_leg3",
                ),
                (
                    f"{-leg_x_offset} {-leg_y_offset} {leg_z_pos}",
                    f"{self.name}_leg4",
                ),
            ]

            for pos, name in leg_positions:
                leg = ET.SubElement(table_body, "geom")
                leg.set("name", name)
                leg.set("type", "cylinder")
                leg.set("size", f"{leg_radius} {leg_half_height}")
                leg.set("pos", pos)
                leg.set("rgba", " ".join(map(str, self.rgba_table_leg)))

        elif self.table_shape == "circle":
            assert self.table_diameter is not None

            # Calculate MuJoCo geom sizes for circular table
            table_radius = float(self.table_diameter) / 2
            table_half_thickness = self.table_thickness / 2
            leg_radius = 0.02
            leg_half_height = (self.table_height - self.table_thickness) / 2

            # Calculate leg positions (inset from edge on a circle)
            # Place 4 legs at 45-degree intervals from edge
            leg_distance_from_center = table_radius - self.leg_inset
            leg_z_pos = leg_half_height  # Center of leg cylinder
            table_top_z_pos = self.table_height - table_half_thickness

            # Create circular table top geom (using cylinder)
            table_top = ET.SubElement(table_body, "geom")
            table_top.set("name", f"{self.name}_top")
            table_top.set("type", "cylinder")
            table_top.set("size", f"{table_radius} {table_half_thickness}")
            table_top.set("pos", f"0 0 {table_top_z_pos}")
            table_top.set("rgba", " ".join(map(str, self.rgba_table_top)))

            # Create table legs at 4 positions around the circle
            # (at 45, 135, 225, 315 degrees)
            leg_angles = [
                math.pi / 4,
                3 * math.pi / 4,
                5 * math.pi / 4,
                7 * math.pi / 4,
            ]  # 45, 135, 225, 315 degrees

            for i, angle in enumerate(leg_angles, 1):
                leg_x = leg_distance_from_center * math.cos(angle)
                leg_y = leg_distance_from_center * math.sin(angle)

                leg = ET.SubElement(table_body, "geom")
                leg.set("name", f"{self.name}_leg{i}")
                leg.set("type", "cylinder")
                leg.set("size", f"{leg_radius} {leg_half_height}")
                leg.set("pos", f"{leg_x} {leg_y} {leg_z_pos}")
                leg.set("rgba", " ".join(map(str, self.rgba_table_leg)))

        else:
            raise ValueError(
                f"Unknown table shape: {self.table_shape}. "
                f"Must be 'rectangle' or 'circle'"
            )

        return table_body

    def _create_regions(self) -> None:
        """Create Region objects with site elements for each region.

        Each region's 2D ranges [x_start, y_start, x_end, y_end] are converted to 3D
        bounding boxes [x_min, y_min, z_min, x_max, y_max, z_max] where the z dimension
        spans from the table surface to DEFAULT_REGION_HEIGHT above it.

        Sites are attached to the table body.
        """
        assert self.regions is not None, "Regions must be defined"

        for region_name, region_config in self.regions.items():
            region_list: list = []

            for region_idx, region_range in enumerate(region_config["ranges"]):
                if len(region_range) == 4:
                    x_start, y_start, x_end, y_end = region_range
                    z_start = 0.0
                    z_end = self.DEFAULT_REGION_HEIGHT
                elif len(region_range) == 6:
                    x_start, y_start, z_start, x_end, y_end, z_end = region_range
                else:
                    raise ValueError(
                        f"Each region range must have exactly 4 values "
                        f"[x_start, y_start, x_end, y_end], "
                        f"got {len(region_range)} for region '{region_name}'"
                    )

                # Validate bounds
                if x_start >= x_end:
                    raise ValueError(
                        f"x_start ({x_start}) must be less than x_end ({x_end}) "
                        f"for region '{region_name}'"
                    )
                if y_start >= y_end:
                    raise ValueError(
                        f"y_start ({y_start}) must be less than y_end ({y_end}) "
                        f"for region '{region_name}'"
                    )
                if z_start >= z_end:
                    raise ValueError(
                        f"z_start ({z_start}) must be less than z_end ({z_end}) "
                        f"for region '{region_name}'"
                    )

                # Create 3D bounding box:
                # z_start is offset above table surface
                # z_end is DEFAULT_REGION_HEIGHT above z_start
                z_start += self.table_height
                z_end += self.table_height

                # Calculate center and half-sizes for MuJoCo box site
                region_center_x = (x_start + x_end) / 2
                region_center_y = (y_start + y_end) / 2
                region_center_z = (z_start + z_end) / 2
                region_size_x = (x_end - x_start) / 2
                region_size_y = (y_end - y_start) / 2
                region_size_z = (z_end - z_start) / 2

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
                site_name = f"{self.name}_{region_name}_region_{region_idx}"
                region = Region(
                    name=site_name,
                    rgba=rgba_values,
                    site_element=site,
                    parent_pos=np.array(self.position, dtype=np.float32),
                    parent_yaw=self.yaw,
                )
                region_list.append(region)

                # Append site element to xml_element
                self.xml_element.append(site)

            self.region_objects[region_name] = region_list

    @staticmethod
    def get_bounding_box_from_config(
        pos: NDArray[np.float32], fixture_config: dict[str, str | float]
    ) -> list[float]:
        """Get bounding box for a table given its position and config.

        Args:
            pos: Position of the table as [x, y, z] array
            fixture_config: Dictionary containing table configuration with keys:
                - "shape": Shape of the table - "rectangle" or "circle"
                - "length": Total table length in meters (for rectangle)
                - "width": Total table width in meters (for rectangle)
                - "diameter": Diameter of circular table in meters (for circle)
                - "height": Table height in meters

        Returns:
            Bounding box as [x_min, y_min, z_min, x_max, y_max, z_max]

        Raises:
            ValueError: If table shape is not supported
        """
        table_height = float(fixture_config["height"])
        z_min = pos[2]
        z_max = pos[2] + table_height

        if fixture_config["shape"] == "rectangle":
            half_length = float(fixture_config["length"]) / 2
            half_width = float(fixture_config["width"]) / 2
            return [
                pos[0] - half_length,  # x_min
                pos[1] - half_width,  # y_min
                z_min,
                pos[0] + half_length,  # x_max
                pos[1] + half_width,  # y_max
                z_max,
            ]
        if fixture_config["shape"] == "circle":
            radius = float(fixture_config["diameter"]) / 2
            return [
                pos[0] - radius,  # x_min
                pos[1] - radius,  # y_min
                z_min,
                pos[0] + radius,  # x_max
                pos[1] + radius,  # y_max
                z_max,
            ]

        raise ValueError(f"Unknown table shape: {fixture_config['shape']}")

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
            Tuple of (x, y, z, yaw) coordinates in world coordinates (offset by table
            position), where yaw is in radians. The yaw range is read from
            self.regions[region_name]["yaw_ranges"] if it exists, otherwise
            defaults to (0.0, 360.0) degrees.

        Raises:
            ValueError: If regions list is empty or if any region has invalid bounds
        """
        assert self.regions is not None, "Regions must be defined"
        assert region_name in self.region_objects, f"Region '{region_name}' not found"

        region_config = self.regions[region_name]
        region_list = self.region_objects[region_name]

        # Randomly select one of the regions
        selected_region_index = np_random.choice(len(region_list))
        selected_region = region_list[selected_region_index]
        selected_bbox = selected_region.bbox

        # Get yaw range for this region
        yaw_range = (0.0, 360.0)  # Default range
        if "yaw_ranges" in region_config:
            yaw_ranges = region_config["yaw_ranges"]
            if yaw_ranges and len(yaw_ranges) > selected_region_index:
                yaw_range = tuple(yaw_ranges[selected_region_index])

        # Sample pose from the 3D bounding box (already in world coordinates)
        x, y, z, yaw = utils.sample_pose_in_bbox_3d(selected_bbox, np_random, yaw_range)

        return (x, y, z, yaw)

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
        # Validate region exists
        assert self.regions is not None, "Regions must be defined"
        if region_name not in self.region_objects:
            raise ValueError(f"Region '{region_name}' not found")

        # Check if position is in any of the region objects
        region_list = self.region_objects[region_name]
        for region in region_list:
            if region.check_in_region(position, env):
                return True

        return False

    def visualize_regions(self) -> None:
        """Visualize the table's regions in the MuJoCo environment.

        This method is a no-op since regions are now added to the XML during
        _create_regions().
        """
        if self.regions is None:
            return

        for region_list in self.region_objects.values():
            for region in region_list:
                region.visualize_region()

    def __str__(self) -> str:
        """String representation of the table."""
        return (
            f"Table(name='{self.name}', shape='{self.table_shape}', "
            f"height={self.table_height}, thickness={self.table_thickness})"
        )

    def __repr__(self) -> str:
        """Detailed string representation of the table."""
        return (
            f"Table(name='{self.name}', "
            f"shape='{self.table_shape}', length={self.table_length}, "
            f"width={self.table_width}, diameter={self.table_diameter}, "
            f"height={self.table_height}, thickness={self.table_thickness}, "
            f"position={self.position}, leg_inset={self.leg_inset})"
        )


@register_fixture
class Cupboard(MujocoFixture):
    """A cupboard fixture with multiple shelves."""

    default_panel_thickness: float = 0.01  # 1cm thick side and back panels
    default_open_cupboard_leg_thickness: float = 0.03  # 3cm thick legs when open
    default_shelf_thickness: float = 0.02  # 2cm thick shelves
    default_partition_thickness: float = 0.01  # 1cm thick partitions
    default_drawer_damping: float = 50.0  # Damping for smooth sliding
    default_drawer_handle_depth: float = 0.04  # 4cm deep drawer handles
    default_drawer_wall_thickness: float = 0.003  # 3mm thick drawer walls
    default_drawer_bottom_thickness: float = 0.03  # 3cm thick drawer bottom
    # NOTE(VS): drawer bottom thickness less than 3cm was causing items to fall
    # through the drawer bottom

    # Default RGBA colors for cupboard and drawer components
    default_rgba_cupboard_shelf: list[float] = [0.8, 0.6, 0.4, 1.0]
    default_rgba_cupboard_top_shelf: list[float] = [0.9, 0.7, 0.5, 1.0]
    default_rgba_cupboard_leg: list[float] = [0.6, 0.4, 0.2, 1.0]
    default_rgba_cupboard_partition: list[float] = [0.7, 0.5, 0.3, 1.0]
    default_rgba_cupboard_panel: list[float] = [0.7, 0.5, 0.3, 1.0]
    default_rgba_drawer_bottom: list[float] = [0.6, 0.5, 0.4, 0.8]
    default_rgba_drawer_wall: list[float] = [0.5, 0.4, 0.3, 0.9]
    default_rgba_drawer_face: list[float] = [0.4, 0.3, 0.2, 1.0]
    default_rgba_drawer_handle: list[float] = [0.3, 0.3, 0.3, 1.0]

    def __init__(
        self,
        name: str,
        fixture_config: dict[str, str | float],
        position: list[float] | NDArray[np.float32],
        yaw: float,
        regions: dict | None = None,
        env: MujocoEnv | None = None,
    ) -> None:
        """Initialize a Cupboard object.

        Top View (from above, y pointing up):

                              y    shelves opening
                              ↑     ↑
                              |     |
                  ┌─────────────●──────────────┐
                  │            |               │
                  │            |               │
          depth   │            | (origin)      │ depth
                  │            |               │
                  │            |               │
                  └─────────────●──────────────┘
                  ←────────────────────────────→
                          length
                                    → x

        Args:
            name: Name of the cupboard body in the XML
            fixture_config: Dictionary containing cupboard configuration with keys:
                - "length": Total cupboard length in meters
                - "depth": Total cupboard depth in meters
                - "shelf_heights": List of distances between consecutive shelves
                  in meters
                - "shelf_partitions": List of lists, each containing partition
                  distances from left edge
                - "side_and_back_open": Boolean indicating if sides and back are
                  open
                - "shelf_thickness": Thickness of each shelf in meters
                  (optional, default 0.02)
            position: Position of the cupboard as [x, y, z]
            yaw: Yaw orientation of the cupboard in radians
            env: Optional reference to the MujocoEnv for accessing joint data
        """
        # Initialize base class
        super().__init__(name, fixture_config, position, yaw, regions, env)

        # Parse cupboard configuration
        self.cupboard_length = float(self.fixture_config["length"])
        self.cupboard_depth = float(self.fixture_config["depth"])

        # Handle shelf_heights - convert to list of floats
        shelf_heights_raw = self.fixture_config["shelf_heights"]
        self.shelf_heights: list[float] = (
            [float(h) for h in shelf_heights_raw]  # type: ignore
            if hasattr(shelf_heights_raw, "__iter__")
            and not isinstance(shelf_heights_raw, str)
            else [float(shelf_heights_raw)]
        )

        # Handle shelf_partitions - convert to list of lists of floats
        shelf_partitions_raw = self.fixture_config["shelf_partitions"]
        self.shelf_partitions: list[list[float]] = [
            [float(p) for p in partition_list]  # type: ignore
            for partition_list in shelf_partitions_raw  # type: ignore
        ]
        self.side_and_back_open: bool = bool(self.fixture_config["side_and_back_open"])
        self.shelf_thickness: float = float(
            self.fixture_config.get("shelf_thickness", Cupboard.default_shelf_thickness)
        )
        self.panel_thickness: float = float(
            self.fixture_config.get("panel_thickness", Cupboard.default_panel_thickness)
        )
        # Set leg thickness only when cupboard is open (has legs, not panels)
        self.leg_thickness: float = (
            float(
                self.fixture_config.get(
                    "leg_thickness", Cupboard.default_open_cupboard_leg_thickness
                )
            )
            if self.side_and_back_open
            else 0.0
        )

        # Handle shelf_drawers - convert to list of lists of bools
        shelf_drawers_raw: list[Any] = cast(
            list[Any], self.fixture_config.get("shelf_drawers", [])
        )
        self.shelf_drawers: list[list[bool]] = []
        if shelf_drawers_raw:
            for drawer_list in shelf_drawers_raw:  # type: ignore
                self.shelf_drawers.append(
                    [bool(d) for d in drawer_list]  # type: ignore
                )

        # Drawer parameters
        self.drawer_damping: float = float(
            self.fixture_config.get("drawer_damping", Cupboard.default_drawer_damping)
        )
        self.drawer_handle_depth: float = float(
            self.fixture_config.get(
                "drawer_handle_depth", Cupboard.default_drawer_handle_depth
            )
        )
        self.drawer_wall_thickness: float = float(
            self.fixture_config.get(
                "drawer_wall_thickness", Cupboard.default_drawer_wall_thickness
            )
        )
        self.drawer_bottom_thickness: float = float(
            self.fixture_config.get(
                "drawer_bottom_thickness", Cupboard.default_drawer_bottom_thickness
            )
        )

        # Parse RGBA colors from fixture config or use class defaults
        self.rgba_cupboard_shelf: list[float] = self.fixture_config.get(
            "rgba_cupboard_shelf", Cupboard.default_rgba_cupboard_shelf
        )  # type: ignore
        self.rgba_cupboard_top_shelf: list[float] = self.fixture_config.get(
            "rgba_cupboard_top_shelf", Cupboard.default_rgba_cupboard_top_shelf
        )  # type: ignore
        self.rgba_cupboard_leg: list[float] = self.fixture_config.get(
            "rgba_cupboard_leg", Cupboard.default_rgba_cupboard_leg
        )  # type: ignore
        self.rgba_cupboard_partition: list[float] = self.fixture_config.get(
            "rgba_cupboard_partition", Cupboard.default_rgba_cupboard_partition
        )  # type: ignore
        self.rgba_cupboard_panel: list[float] = self.fixture_config.get(
            "rgba_cupboard_panel", Cupboard.default_rgba_cupboard_panel
        )  # type: ignore
        self.rgba_drawer_bottom: list[float] = self.fixture_config.get(
            "rgba_drawer_bottom", Cupboard.default_rgba_drawer_bottom
        )  # type: ignore
        self.rgba_drawer_wall: list[float] = self.fixture_config.get(
            "rgba_drawer_wall", Cupboard.default_rgba_drawer_wall
        )  # type: ignore
        self.rgba_drawer_face: list[float] = self.fixture_config.get(
            "rgba_drawer_face", Cupboard.default_rgba_drawer_face
        )  # type: ignore
        self.rgba_drawer_handle: list[float] = self.fixture_config.get(
            "rgba_drawer_handle", Cupboard.default_rgba_drawer_handle
        )  # type: ignore
        # Max drawer slide distance (80% of shelf depth)
        self.drawer_max_slide: float = self.cupboard_depth * 0.8

        # Calculate derived properties
        self.num_shelves: int = len(self.shelf_heights) + 1  # +1 for the top shelf
        self.cupboard_height: float = (
            sum(self.shelf_heights) + self.num_shelves * self.shelf_thickness
        )

        # Validate configuration
        if len(self.shelf_heights) < 1:
            raise ValueError("Number of shelf heights must be at least 1")

        if len(self.shelf_partitions) != len(self.shelf_heights):
            raise ValueError(
                f"shelf_partitions must have {len(self.shelf_heights)} lists, "
                f"got {len(self.shelf_partitions)} (one list per shelf gap, "
                f"not including top shelf)"
            )

        # Validate partition positions
        for i, partitions in enumerate(self.shelf_partitions):
            for partition_pos in partitions:
                if (
                    partition_pos <= -self.cupboard_length / 2
                    or partition_pos >= self.cupboard_length / 2
                ):
                    raise ValueError(
                        f"Partition position {partition_pos} on shelf {i} must be "
                        f"strictly between -{self.cupboard_length/2} and "
                        f"{self.cupboard_length/2} "
                        f"(cupboard length is {self.cupboard_length})"
                    )

        # Validate shelf_drawers configuration if provided
        if self.shelf_drawers:
            if len(self.shelf_drawers) != len(self.shelf_heights):
                raise ValueError(
                    f"shelf_drawers must have {len(self.shelf_heights)} lists, "
                    f"got {len(self.shelf_drawers)} (one list per shelf gap, "
                    f"not including top shelf)"
                )
            for i, drawer_list in enumerate(self.shelf_drawers):
                expected_len = len(self.shelf_partitions[i]) + 1
                if len(drawer_list) != expected_len:
                    raise ValueError(
                        f"shelf_drawers[{i}] must have {expected_len} elements "
                        f"(num_partitions + 1), got {len(drawer_list)}"
                    )

        # Precompute shelf z positions for efficiency
        self._shelf_z_positions = self._compute_shelf_z_positions()

        # Initialize drawer joints list (populated during XML creation)
        self.drawer_joints: list[str] = []

        # Initialize mapping of drawer joint names to drawer symbolic objects
        # This will be populated in get_object_centric_state()
        self._drawer_symbolic_objects: dict[str, Object] = {}

        # Create the XML element
        self.xml_element = self._create_xml_element()

        # Initialize pending region sites list (populated by _create_regions)
        self._pending_region_sites: list[tuple[str, ET.Element]] = []

        # Create regions after all attributes are initialized
        if self.regions is not None:
            self._create_regions()

    def _compute_shelf_z_positions(self) -> list[float]:
        """Compute the z position of each shelf surface.

        Returns:
            List of z positions for each shelf surface (relative to cupboard base)
        """
        shelf_z_positions = []
        current_z = self.shelf_thickness / 2

        for i in range(self.num_shelves):
            # Z position of shelf surface (top of shelf)
            shelf_surface_z = current_z + self.shelf_thickness / 2
            shelf_z_positions.append(shelf_surface_z)

            # Move to next shelf if not the last one
            if i < len(self.shelf_heights):
                current_z += (
                    self.shelf_thickness / 2
                    + self.shelf_heights[i]
                    + self.shelf_thickness / 2
                )

        return shelf_z_positions

    def _create_regions(self) -> None:
        """Create Region objects with site elements for each region.

        For shelves with partitions: region config must include:
            - "shelf": shelf index
            - "partition": partition index (0 for first partition, 1 for second, etc.)
            - "ranges": [[x_start, y_start, x_end, y_end]] relative to partition center

        For non-partitioned shelves (top shelf): region config must include:
            - "shelf": shelf index (top shelf = num_shelves - 1)
            - "ranges": [[x_start, y_start, x_end, y_end]] relative to shelf center

        For drawers: if "drawer" is True, site is attached to drawer body instead
        of cupboard.

        Sites are positioned at the partition/compartment center.
        """
        assert self.regions is not None, "Regions must be defined"

        for region_name, region_config in self.regions.items():
            # Get the shelf index for this region
            if "shelf" not in region_config:
                raise ValueError(
                    f"Cupboard region '{region_name}' must specify 'shelf'"
                )
            shelf = region_config["shelf"]

            # Validate shelf index
            if shelf < 0 or shelf >= self.num_shelves:
                raise ValueError(
                    f"Shelf index {shelf} out of range for cupboard with "
                    f"{self.num_shelves} shelves in region '{region_name}'"
                )

            region_list: list = []

            # Determine the height available on this shelf
            if shelf < len(self.shelf_heights):
                shelf_height = self.shelf_heights[shelf]
            else:
                # For the top shelf, use the last shelf height as default
                shelf_height = self.shelf_heights[-1] if self.shelf_heights else 0.2

            # Check if this shelf has partitions
            has_partitions = (
                shelf < len(self.shelf_partitions) and self.shelf_partitions[shelf]
            )

            if has_partitions:
                # Region with partitions - must specify partition index
                if "partition" not in region_config:
                    raise ValueError(
                        f"Cupboard region '{region_name}' on shelf {shelf} "
                        f"has partitions but does not specify 'partition' index"
                    )
                partition_idx = region_config["partition"]
                partitions = self.shelf_partitions[shelf]

                # Validate partition index
                if partition_idx < 0 or partition_idx > len(partitions):
                    raise ValueError(
                        f"Partition index {partition_idx} out of range for "
                        f"shelf {shelf} with {len(partitions)} partitions "
                        f"in '{region_name}'"
                    )

                # Get the x bounds of this partition/compartment
                x_min, x_max = self._get_drawer_compartment_bounds(shelf, partition_idx)
                partition_center_x = (x_min + x_max) / 2
            else:
                # Non-partitioned shelf (top shelf) - entire shelf is one region
                if "partition" in region_config:
                    raise ValueError(
                        f"Cupboard region '{region_name}' on shelf {shelf} "
                        f"has no partitions but specifies 'partition' index"
                    )
                cupboard_half_length = self.cupboard_length / 2
                partition_center_x = 0.0  # Center of shelf
                x_min = -cupboard_half_length
                x_max = cupboard_half_length

            # Compute z bounds from shelf position and height
            z_min = self._shelf_z_positions[shelf]
            z_max = z_min + shelf_height
            # Compute relative z bounds with respect to shelf position
            partition_center_z = self._shelf_z_positions[shelf]
            z_min_relative = z_min - partition_center_z
            z_max_relative = z_max - partition_center_z

            # Get ranges - if not provided, use the computed x_min, x_max, y_min, y_max
            ranges = region_config.get("ranges")
            if ranges is None:
                # Compute y_min and y_max from cupboard depth
                cupboard_half_depth = self.cupboard_depth / 2
                y_min = -cupboard_half_depth
                y_max = cupboard_half_depth
                # Convert x bounds to be relative to partition center
                x_min_relative = x_min - partition_center_x
                x_max_relative = x_max - partition_center_x
                ranges = [[x_min_relative, y_min, x_max_relative, y_max]]

            # Parse each compartment-relative range, where ranges are with respect to
            # (partition_center_x, 0, partition_center_z)
            for region_idx, region_range in enumerate(ranges):
                if len(region_range) == 4:
                    x_start, y_start, x_end, y_end = region_range
                    z_start, z_end = z_min_relative, z_max_relative
                elif len(region_range) == 6:
                    x_start, y_start, z_start, x_end, y_end, z_end = region_range
                else:
                    raise ValueError(
                        f"Each region range must have 4 or 6 values "
                        f"[x_start, y_start, x_end, y_end] or "
                        f"[x_start, y_start, z_start, x_end, y_end, z_end], "
                        f"got {len(region_range)} for region '{region_name}'"
                    )

                # Validate bounds (ranges are relative to partition center)
                if x_start >= x_end:
                    raise ValueError(
                        f"x_start ({x_start}) must be less than x_end ({x_end}) "
                        f"for region '{region_name}'"
                    )
                if y_start >= y_end:
                    raise ValueError(
                        f"y_start ({y_start}) must be less than y_end ({y_end}) "
                        f"for region '{region_name}'"
                    )
                if z_start >= z_end:
                    raise ValueError(
                        f"z_start ({z_start}) must be less than z_end ({z_end}) "
                        f"for region '{region_name}'"
                    )

                # Determine if this compartment has a drawer (before coordinate
                # conversion)
                compartment_has_drawer = False
                if self.shelf_drawers and shelf < len(self.shelf_drawers):
                    shelf_drawers_list = self.shelf_drawers[shelf]
                    if has_partitions:
                        # For partitioned shelves, check if this specific partition
                        # has a drawer
                        if partition_idx < len(shelf_drawers_list):
                            compartment_has_drawer = shelf_drawers_list[partition_idx]
                    else:
                        # For non-partitioned shelves, check if the single compartment
                        # has a drawer
                        if shelf_drawers_list:
                            compartment_has_drawer = shelf_drawers_list[0]

                # Calculate center and half-sizes for MuJoCo box site
                if compartment_has_drawer:
                    # For drawer sites: keep partition-relative coordinates
                    # (drawer body has its own local frame centered at partition center)
                    region_center_x = (x_start + x_end) / 2
                    region_center_y = (y_start + y_end) / 2
                    region_center_z = (z_start + z_end) / 2  # Center of drawer height
                else:
                    # For cupboard sites: convert partition-relative to world-relative
                    region_center_x = (
                        partition_center_x + x_start + partition_center_x + x_end
                    ) / 2
                    region_center_y = (y_start + y_end) / 2
                    # Use pre-computed z bounds
                    region_center_z = (
                        partition_center_z + z_start + partition_center_z + z_end
                    ) / 2
                region_size_x = (x_end - x_start) / 2
                region_size_y = (y_end - y_start) / 2
                region_size_z = (z_end - z_start) / 2

                # Create site element for the region
                site = ET.Element("site")
                site_name = f"{self.name}_{region_name}_region_{region_idx}"
                site.set("name", site_name)
                site.set("type", "box")
                site.set("size", f"{region_size_x} {region_size_y} {region_size_z}")
                site.set(
                    "pos", f"{region_center_x} {region_center_y} {region_center_z}"
                )
                rgba_values = region_config.get("rgba", [1.0, 0.0, 0.0, 0.0])
                site.set("rgba", " ".join(map(str, rgba_values)))
                site.set("group", "0")

                # Create Region object (derived from site_element)
                # Determine parent pose: cupboard pose + drawer pose if applicable
                if compartment_has_drawer:
                    if has_partitions:
                        drawer_index = f"s{shelf}c{partition_idx}"
                    else:
                        drawer_index = f"s{shelf}c0"
                    # Calculate drawer pose from compartment geometry
                    # (this is the same calculation as in _create_drawer_body)
                    x_min_comp, x_max_comp = self._get_drawer_compartment_bounds(
                        shelf, partition_idx if has_partitions else 0
                    )
                    drawer_center_x = (x_min_comp + x_max_comp) / 2
                    shelf_z = self._shelf_z_positions[shelf]
                    depth_margin = (
                        self.leg_thickness
                        if self.side_and_back_open
                        else self.panel_thickness
                    )
                    drawer_y = depth_margin / 2
                    drawer_z = (
                        shelf_z + self.shelf_thickness / 2 + self.drawer_wall_thickness
                    )
                    # Parent pose is cupboard position + drawer position
                    drawer_pos = np.array(
                        [drawer_center_x, drawer_y, drawer_z],
                        dtype=np.float32,
                    )  # Note: (to fix) region bounds should account for
                    # drawer position offset
                    parent_pos = np.array(self.position, dtype=np.float32) + drawer_pos
                else:
                    # Parent pose is just the cupboard position
                    parent_pos = np.array(self.position, dtype=np.float32)

                region = Region(
                    name=site_name,
                    rgba=rgba_values,
                    site_element=site,
                    parent_pos=parent_pos,
                    parent_yaw=self.yaw,
                )
                region_list.append(region)

                # Store which body this site should be attached to
                if not hasattr(self, "_region_site_bodies"):
                    self._region_site_bodies: dict[str, str] = {}

                if compartment_has_drawer:
                    # Attach to drawer body
                    # Use the same naming convention as in _create_xml_element()
                    if has_partitions:
                        drawer_index = f"s{shelf}c{partition_idx}"
                    else:
                        drawer_index = f"s{shelf}c0"
                    self._region_site_bodies[site_name] = (
                        f"{self.name}_drawer_{drawer_index}"
                    )
                else:
                    # Attach to main cupboard body
                    self._region_site_bodies[site_name] = self.name

                # Append site element to the appropriate body
                # We'll do this after all regions are created to ensure drawer bodies
                # exist. Store the site for later appending
                self._pending_region_sites.append((site_name, site))

            self.region_objects[region_name] = region_list

        # Now append all pending sites to their target bodies
        self._append_region_sites_to_bodies()

    def _append_region_sites_to_bodies(self) -> None:
        """Append pending region sites to their target bodies in the XML.

        This is called after _create_regions() to ensure all drawer bodies have been
        created in the XML.
        """
        if not hasattr(self, "_pending_region_sites"):
            return

        # Build a map of body names to body elements
        body_map: dict[str, ET.Element] = {self.name: self.xml_element}

        # Recursively find all drawer bodies
        def find_bodies(parent_elem: ET.Element) -> None:
            for child in parent_elem:
                if child.tag == "body":
                    body_name = child.get("name", "")
                    if body_name:
                        body_map[body_name] = child

        find_bodies(self.xml_element)

        # Append each site to its target body
        for site_name, site_element in self._pending_region_sites:
            if (
                hasattr(self, "_region_site_bodies")
                and site_name in self._region_site_bodies
            ):
                body_name = self._region_site_bodies[site_name]
                if body_name in body_map:
                    body_map[body_name].append(site_element)
                else:
                    # Fallback to main cupboard body if target not found
                    self.xml_element.append(site_element)
            else:
                # Default: attach to main cupboard body
                self.xml_element.append(site_element)

        # Clear pending sites
        self._pending_region_sites = []

    def _get_drawer_compartment_bounds(
        self, shelf_index: int, compartment_index: int
    ) -> tuple[float, float]:
        """Get the x-bounds of a specific drawer compartment.

        Args:
            shelf_index: Index of the shelf (0-based)
            compartment_index: Index of the compartment (0-based)

        Returns:
            Tuple of (x_min, x_max) in cupboard-relative coordinates
        """
        cupboard_half_length = self.cupboard_length / 2
        partitions = self.shelf_partitions[shelf_index]

        # Sort partitions for consistent ordering
        sorted_partitions = sorted(partitions)

        # Account for side panel thickness when panels are present
        side_panel_extent = 0.0
        if not self.side_and_back_open:
            side_panel_extent = self.panel_thickness
        else:
            side_panel_extent = self.leg_thickness

        # Partition half-thickness for computing compartment edges
        partition_half_thickness = Cupboard.default_partition_thickness / 2

        if compartment_index == 0:
            # Leftmost compartment: start after left side panel
            x_min = -cupboard_half_length + side_panel_extent
            if sorted_partitions:
                # End at the left edge of the first partition
                x_max = sorted_partitions[0] - partition_half_thickness
            else:
                x_max = cupboard_half_length - side_panel_extent
        elif compartment_index == len(sorted_partitions):
            # Rightmost compartment: start at right edge of last partition,
            # end before right side panel
            x_min = sorted_partitions[-1] + partition_half_thickness
            x_max = cupboard_half_length - side_panel_extent
        else:
            # Middle compartment: bounded by two partitions
            x_min = sorted_partitions[compartment_index - 1] + partition_half_thickness
            x_max = sorted_partitions[compartment_index] - partition_half_thickness

        return x_min, x_max

    def _create_drawer_handle(
        self, handle_length: float, handle_depth: float
    ) -> ET.Element:
        """Create a handle body for a drawer.

        The handle consists of 3 box geoms:
        - One main box along the x-axis
        - Two smaller boxes attached perpendicularly (along y-axis) at the ends,
          positioned to span 80% of the handle length

        The origin is at the center of the main box in x, and the perpendicular
        boxes end at y=0 (extending from y=-handle_depth to y=0).

        Example handle layout (top view):
                        (x=0 ⬅️, y=0 ⬇️) <-- Origin at center of main box
                               ^
                               |
                               |
                               .
                ||                           ||  <-- Perpendicular attachments
                ||                           ||
                ||                           ||
        =============================================== <-- Main horizontal handle
        |<-------------- handle_length -------------->|

        Args:
            handle_length: Length of the main handle box (x-direction)
            handle_depth: Depth of the perpendicular boxes (y-direction)

        Returns:
            ET.Element representing the handle body
        """
        handle_body = ET.Element("body")

        handle_size = 0.006  # 6mm wide boxes for the handle
        handle_half_size = handle_size / 2

        # Main horizontal box (along x-axis)
        main_box = ET.SubElement(handle_body, "geom")
        main_box.set("type", "box")
        main_box.set(
            "size", f"{handle_length / 2} {handle_half_size} {handle_half_size}"
        )
        main_box.set("pos", f"0 {handle_depth - handle_half_size} 0")
        main_box.set("rgba", " ".join(map(str, self.rgba_drawer_handle)))

        # Two perpendicular boxes at the ends (along y-axis)
        # Positioned 10% from each end, so total span is 80% of handle_length
        # 10% inset from each end
        x_offset = (handle_length / 2 - handle_half_size) * 0.8
        # half length of perpendicular attachments
        perp_half_depth = handle_depth / 2

        for x_pos in [-x_offset, x_offset]:
            y_pos = perp_half_depth - handle_half_size
            perp_box = ET.SubElement(handle_body, "geom")
            perp_box.set("type", "box")
            perp_box.set("size", f"{handle_half_size} {y_pos} {handle_half_size}")
            perp_box.set("pos", f"{x_pos} {y_pos} 0")
            perp_box.set("rgba", " ".join(map(str, self.rgba_drawer_handle)))

        return handle_body

    def _create_drawer_body(
        self,
        drawer_index: str,
        shelf_z: float,
        shelf_half_thickness: float,
        shelf_height: float,
        x_min: float,
        x_max: float,
        cupboard_half_depth: float,
        compartment_index: int = 0,
        num_compartments: int = 1,
    ) -> ET.Element:
        """Create a drawer body with 5 geoms (front, back, left, right, bottom).

        Args:
            drawer_index: Unique identifier for the drawer (e.g., "shelf1_comp0")
            shelf_z: Z position of the shelf surface
            shelf_half_thickness: Half thickness of the shelf
            shelf_height: Height available for the drawer
            x_min: Minimum x coordinate of the drawer compartment
            x_max: Maximum x coordinate of the drawer compartment
            cupboard_half_depth: Half depth of the cupboard
            compartment_index: Index of this compartment (0-based)
            num_compartments: Total number of compartments on this shelf

        Returns:
            ET.Element representing the drawer body
        """
        # Create drawer body
        drawer_body = ET.Element("body")
        drawer_body.set("name", f"{self.name}_drawer_{drawer_index}")

        # x_min and x_max are the unadjusted compartment bounds
        # (representing actual panel/partition extents)
        # Drawer compartment dimensions
        drawer_length = x_max - x_min
        drawer_half_length = drawer_length / 2

        # Adjust for side panel width (when panels present) or leg thickness (when open)
        # This insets the drawable space from the compartment boundaries
        # Use multiple of wall thickness for clearance
        adjust_half_t = 4 * self.drawer_wall_thickness

        x_min_adjusted = x_min + adjust_half_t
        x_max_adjusted = x_max - adjust_half_t

        # Create drawer walls using thin geoms
        wall_t = self.drawer_wall_thickness
        wall_half_t = wall_t / 2
        bottom_t = self.drawer_bottom_thickness
        bottom_half_t = bottom_t / 2

        # Compute clearances for left and right sides
        # Using adjusted bounds (already inset by structural thickness)
        # So we only need wall thickness clearance from the adjusted boundaries
        left_clearance = wall_t
        right_clearance = wall_t

        # Adjust drawer center if left and right clearances are asymmetric
        drawer_center_shift = (left_clearance - right_clearance) / 2
        drawer_center_x = (x_min_adjusted + x_max_adjusted) / 2 + drawer_center_shift

        drawer_length_adjusted = x_max_adjusted - x_min_adjusted
        drawer_length_adjusted = (
            drawer_length_adjusted - left_clearance - right_clearance
        )
        drawer_half_length = drawer_length_adjusted / 2

        # Determine edge thickness based on cupboard configuration
        # When open: edges have legs; when closed: edges have panels
        edge_thickness = (
            self.leg_thickness if self.side_and_back_open else self.panel_thickness
        )

        # Depth margin: inset from back panel/leg to keep drawer inside
        # Use panel thickness for closed cupboard, leg thickness for open
        depth_margin = edge_thickness
        drawer_half_depth = cupboard_half_depth - depth_margin / 2

        # Drawer position: centered in inset compartment, at shelf surface
        # Adjust y position so front face is flush with shelf edge
        # at cupboard_half_depth - depth_margin/2
        drawer_y = depth_margin / 2
        drawer_z = shelf_z + shelf_half_thickness + bottom_half_t
        drawer_body.set("pos", f"{drawer_center_x} {drawer_y} {drawer_z}")

        # Create sliding joint inside the drawer body
        joint = ET.SubElement(drawer_body, "joint")
        joint_name = f"{self.name}_drawer_{drawer_index}_joint"
        joint.set("name", joint_name)
        joint.set("type", "slide")
        joint.set("axis", "0 1 0")  # Slide along Y axis (in/out)
        joint.set("range", f"0 {self.drawer_max_slide}")
        joint.set("damping", str(self.drawer_damping))
        # Track this joint for object-centric data
        self.drawer_joints.append(joint_name)

        # Vertical clearance: reduce wall height to avoid collision with shelf above
        vertical_clearance = 4 * wall_t
        wall_height = shelf_height - bottom_t - vertical_clearance
        wall_half_height = wall_height / 2
        wall_pos_z = bottom_half_t + wall_half_height  # Position above the bottom geom

        # Bottom geom (closes bottom of drawer) - align flush with side walls
        bottom = ET.SubElement(drawer_body, "geom")
        bottom.set("name", f"{self.name}_drawer_{drawer_index}_bottom")
        bottom.set("type", "box")
        bottom.set(
            "size",
            f"{drawer_half_length - wall_half_t} {drawer_half_depth} {bottom_half_t}",
        )
        bottom.set("pos", f"0 0 {bottom_half_t/2}")
        bottom.set("rgba", " ".join(map(str, self.rgba_drawer_bottom)))

        # Front geom (facing out towards user)
        front = ET.SubElement(drawer_body, "geom")
        front.set("name", f"{self.name}_drawer_{drawer_index}_front")
        front.set("type", "box")
        front.set(
            "size",
            f"{drawer_half_length - wall_half_t} {wall_half_t} {wall_half_height}",
        )
        front.set("pos", f"0 {drawer_half_depth - wall_half_t} {wall_pos_z}")
        front.set("rgba", " ".join(map(str, self.rgba_drawer_wall)))

        # Drawer face geom (spans full width accounting for partitions)
        partition_thickness = Cupboard.default_partition_thickness

        # Determine face left and right bounds
        # At side panels/legs (edges): extend by the appropriate edge thickness
        # At partitions (middle): extend by half partition thickness
        if compartment_index == 0:
            # Left side is at the edge, extend by edge thickness into it
            face_left = x_min - edge_thickness
        else:
            # Left side is at a partition, extend half partition thickness into it
            face_left = x_min - (0.8 * partition_thickness / 2)

        if compartment_index == num_compartments - 1:
            # Right side is at the edge, extend by edge thickness into it
            face_right = x_max + edge_thickness
        else:
            # Right side is at a partition, extend half partition thickness into it
            face_right = x_max + (0.8 * partition_thickness / 2)

        # Calculate face dimensions
        face_length = face_right - face_left
        face_half_length = face_length / 2
        face_center_x = (face_left + face_right) / 2

        face_thickness_half = self.shelf_thickness / 2

        # Position relative to drawer body center
        face_local_center_x = face_center_x - drawer_center_x
        face_local_center_y = drawer_half_depth + face_thickness_half
        # Face geom height: covers drawer wall plus shelf thickness plus extends
        # halfway into shelf above
        face_height = wall_half_height + shelf_half_thickness + shelf_half_thickness / 2
        # Shift face position upward to extend into the shelf above
        face_z_pos = wall_pos_z - wall_half_t + shelf_half_thickness / 4
        face = ET.SubElement(drawer_body, "geom")
        face.set("name", f"{self.name}_drawer_{drawer_index}_face")
        face.set("type", "box")
        face.set("size", f"{face_half_length} {face_thickness_half} {face_height}")
        face.set(
            "pos",
            f"{face_local_center_x} {face_local_center_y} {face_z_pos}",
        )
        face.set("rgba", " ".join(map(str, self.rgba_drawer_face)))

        # Back geom (facing away, allows sliding)
        back = ET.SubElement(drawer_body, "geom")
        back.set("name", f"{self.name}_drawer_{drawer_index}_back")
        back.set("type", "box")
        back.set(
            "size",
            f"{drawer_half_length - wall_half_t} {wall_half_t} {wall_half_height}",
        )
        back.set("pos", f"0 {-(drawer_half_depth - wall_half_t)} {wall_pos_z}")
        back.set("rgba", " ".join(map(str, self.rgba_drawer_wall)))

        # Left geom (left side wall) - inset to avoid partition/panel collision
        left = ET.SubElement(drawer_body, "geom")
        left.set("name", f"{self.name}_drawer_{drawer_index}_left")
        left.set("type", "box")
        left.set("size", f"{wall_half_t} {drawer_half_depth} {wall_half_height}")
        left.set(
            "pos",
            f"{-(drawer_half_length - 2*wall_half_t)} 0 {wall_pos_z}",
        )
        left.set("rgba", " ".join(map(str, self.rgba_drawer_wall)))

        # Right geom (right side wall) - inset to avoid partition/panel collision
        right = ET.SubElement(drawer_body, "geom")
        right.set("name", f"{self.name}_drawer_{drawer_index}_right")
        right.set("type", "box")
        right.set("size", f"{wall_half_t} {drawer_half_depth} {wall_half_height}")
        right.set(
            "pos",
            f"{drawer_half_length - 2*wall_half_t} 0 {wall_pos_z}",
        )
        right.set("rgba", " ".join(map(str, self.rgba_drawer_wall)))

        # Create handle body with 3 boxes (main + 2 perpendicular)
        # 40% of drawer width
        handle_length = (drawer_half_length - wall_half_t) * 0.4
        # total protrusion of handle body
        handle_depth = self.drawer_handle_depth
        handle_body = self._create_drawer_handle(handle_length, handle_depth)

        # Set handle body properties and position at center of face geom
        handle_body.set("name", f"{self.name}_drawer_{drawer_index}_handle")
        handle_body.set(
            "pos",
            f"{face_local_center_x} {face_local_center_y + face_thickness_half} "
            f"{face_z_pos}",
        )
        # Add handle body to drawer
        drawer_body.append(handle_body)

        return drawer_body

    def _create_xml_element(self) -> ET.Element:
        """Create the XML Element for this cupboard.

        Returns:
            ET.Element representing the cupboard body
        """
        # Create cupboard body element
        cupboard_body = ET.Element("body")
        cupboard_body.set("name", self.name)
        position_str = " ".join(str(x) for x in self.position)
        cupboard_body.set("pos", position_str)
        ori_quat = utils.convert_yaw_to_quaternion(self.yaw)
        orientation_str = " ".join(str(x) for x in ori_quat)
        cupboard_body.set("quat", orientation_str)

        # Calculate dimensions
        cupboard_half_length = self.cupboard_length / 2
        cupboard_half_depth = self.cupboard_depth / 2
        shelf_half_thickness = self.shelf_thickness / 2

        # Calculate the height of the topmost shelf
        top_shelf_z = shelf_half_thickness
        for height in self.shelf_heights:
            top_shelf_z += shelf_half_thickness + height + shelf_half_thickness

        # Make legs flush with the top shelf (leg height = top shelf height)
        leg_half_height = top_shelf_z / 2

        # Calculate leg positions (at the edges)
        leg_x_offset = cupboard_half_length - self.leg_thickness / 2
        leg_y_offset = cupboard_half_depth - self.leg_thickness / 2

        # Create vertical legs at four corners (only when cupboard is open)
        if self.side_and_back_open:
            leg_positions = [
                (f"{leg_x_offset} {leg_y_offset}", f"{self.name}_leg1"),
                (f"{-leg_x_offset} {leg_y_offset}", f"{self.name}_leg2"),
                (f"{leg_x_offset} {-leg_y_offset}", f"{self.name}_leg3"),
                (f"{-leg_x_offset} {-leg_y_offset}", f"{self.name}_leg4"),
            ]

            for pos, name in leg_positions:
                leg = ET.SubElement(cupboard_body, "geom")
                leg.set("name", name)
                leg.set("type", "box")
                leg.set(
                    "size",
                    f"{self.leg_thickness/2} {self.leg_thickness/2} {leg_half_height}",
                )
                leg.set(
                    "pos", f"{pos.split()[0]} {pos.split()[1]} {leg_half_height}"
                )  # Position leg center at half its height
                leg.set("rgba", " ".join(map(str, self.rgba_cupboard_leg)))

        # Calculate cumulative shelf positions
        current_z = shelf_half_thickness
        shelf_positions = [current_z]

        for height in self.shelf_heights:  # Include all heights to get the top shelf
            current_z += shelf_half_thickness + height + shelf_half_thickness
            shelf_positions.append(current_z)

        # Create horizontal shelves (including the top shelf)
        for i, shelf_z in enumerate(shelf_positions):
            shelf = ET.SubElement(cupboard_body, "geom")
            shelf.set("name", f"{self.name}_shelf{i+1}")
            shelf.set("type", "box")
            shelf.set(
                "size",
                f"{cupboard_half_length} {cupboard_half_depth} {shelf_half_thickness}",
            )
            shelf.set("pos", f"0 0 {shelf_z}")
            # Use different rgba for top shelf
            is_top_shelf = i == len(shelf_positions) - 1
            shelf_rgba = (
                self.rgba_cupboard_top_shelf
                if is_top_shelf
                else self.rgba_cupboard_shelf
            )
            shelf.set("rgba", " ".join(map(str, shelf_rgba)))

            # Create vertical partitions for this shelf
            # (if we have partition data for it)
            if i < len(self.shelf_partitions):
                partitions = self.shelf_partitions[i]
                shelf_height = (
                    self.shelf_heights[i]
                    if i < len(self.shelf_heights)
                    else self.shelf_heights[-1]
                )

                for j, partition_x in enumerate(partitions):
                    # partition_x is already in center-relative coordinates
                    # Calculate partition dimensions
                    partition_half_thickness = Cupboard.default_partition_thickness / 2
                    partition_half_height = shelf_height / 2
                    partition_z = shelf_z + shelf_half_thickness + partition_half_height

                    partition = ET.SubElement(cupboard_body, "geom")
                    partition.set("name", f"{self.name}_shelf{i+1}_partition{j+1}")
                    partition.set("type", "box")
                    partition.set(
                        "size",
                        f"{partition_half_thickness} {cupboard_half_depth} "
                        f"{partition_half_height}",
                    )
                    partition.set("pos", f"{partition_x} 0 {partition_z}")
                    partition.set(
                        "rgba", " ".join(map(str, self.rgba_cupboard_partition))
                    )

                # Create drawers for this shelf if configured
                if self.shelf_drawers and i < len(self.shelf_drawers):
                    drawer_list = self.shelf_drawers[i]
                    num_compartments = len(drawer_list)

                    for comp_idx in range(num_compartments):
                        if drawer_list[comp_idx]:
                            # Create drawer for this compartment
                            x_min, x_max = self._get_drawer_compartment_bounds(
                                i, comp_idx
                            )
                            drawer_index = f"s{i}c{comp_idx}"

                            # Create drawer body (includes joint inside)
                            drawer_body = self._create_drawer_body(
                                drawer_index,
                                shelf_z,
                                shelf_half_thickness,
                                shelf_height,
                                x_min,
                                x_max,
                                cupboard_half_depth,
                                compartment_index=comp_idx,
                                num_compartments=num_compartments,
                            )

                            # Add drawer body to cupboard
                            cupboard_body.append(drawer_body)

        # Create side and back panels if not open
        if not self.side_and_back_open:
            panel_half_thickness = self.panel_thickness / 2
            # Make panels flush with the top shelf (same height as legs)
            panel_half_height = top_shelf_z / 2

            # Back panel (at -Y edge)
            back_panel = ET.SubElement(cupboard_body, "geom")
            back_panel.set("name", f"{self.name}_back_panel")
            back_panel.set("type", "box")
            back_panel.set(
                "size",
                f"{cupboard_half_length} {panel_half_thickness} {panel_half_height}",
            )
            back_panel.set(
                "pos",
                f"0 {-cupboard_half_depth + panel_half_thickness} {panel_half_height}",
            )
            back_panel.set("rgba", " ".join(map(str, self.rgba_cupboard_panel)))

            # Left side panel (at -X edge)
            left_panel = ET.SubElement(cupboard_body, "geom")
            left_panel.set("name", f"{self.name}_left_panel")
            left_panel.set("type", "box")
            left_panel.set(
                "size",
                f"{panel_half_thickness} {cupboard_half_depth} {panel_half_height}",
            )
            left_panel.set(
                "pos",
                f"{-cupboard_half_length + panel_half_thickness} 0 {panel_half_height}",
            )
            left_panel.set("rgba", " ".join(map(str, self.rgba_cupboard_panel)))

            # Right side panel (at +X edge)
            right_panel = ET.SubElement(cupboard_body, "geom")
            right_panel.set("name", f"{self.name}_right_panel")
            right_panel.set("type", "box")
            right_panel.set(
                "size",
                f"{panel_half_thickness} {cupboard_half_depth} {panel_half_height}",
            )
            right_panel.set(
                "pos",
                f"{cupboard_half_length - panel_half_thickness} 0 {panel_half_height}",
            )
            right_panel.set("rgba", " ".join(map(str, self.rgba_cupboard_panel)))

        return cupboard_body

    @staticmethod
    def get_bounding_box_from_config(
        pos: NDArray[np.float32], fixture_config: dict[str, str | float]
    ) -> list[float]:
        """Get bounding box for a cupboard given its position and config.

        Args:
            pos: Position of the cupboard as [x, y, z] array
            fixture_config: Dictionary containing cupboard configuration with keys:
                - "length": Total cupboard length in meters
                - "depth": Total cupboard depth in meters
                - "shelf_heights": List of distances between consecutive shelves
                - "shelf_thickness": Thickness of each shelf in meters
                  (optional, default 0.02)

        Returns:
            Bounding box as [x_min, y_min, z_min, x_max, y_max, z_max]

        Raises:
            ValueError: If required keys are missing in fixture_config
        """
        if "length" not in fixture_config or "depth" not in fixture_config:
            raise ValueError("fixture_config must contain 'length' and 'depth' keys")

        half_length = float(fixture_config["length"]) / 2
        half_depth = float(fixture_config["depth"]) / 2

        # Calculate cupboard height from shelf configuration
        shelf_heights_config: list[float] = fixture_config.get(
            "shelf_heights", []
        )  # type: ignore
        shelf_heights_float: list[float] = [float(h) for h in shelf_heights_config]
        shelf_thickness = float(
            fixture_config.get("shelf_thickness", Cupboard.default_shelf_thickness)
        )
        num_shelves = len(shelf_heights_float) + 1  # +1 for the top shelf
        cupboard_height = sum(shelf_heights_float) + num_shelves * shelf_thickness

        return [
            pos[0] - half_length,  # x_min
            pos[1] - half_depth,  # y_min
            pos[2],  # z_min
            pos[0] + half_length,  # x_max
            pos[1] + half_depth,  # y_max
            pos[2] + cupboard_height,  # z_max
        ]

    def sample_pose_in_region(
        self,
        region_name: str,
        np_random: np.random.Generator,
    ) -> tuple[float, float, float, float]:
        """Sample a pose (x, y, z, yaw) uniformly randomly from one of the provided
        regions.

        For cupboards, this samples on the specified shelf surface.

        Args:
            region_name: Name of the region to sample from
            np_random: Random number generator

        Returns:
            Tuple of (x, y, z, yaw) coordinates in world coordinates (offset by cupboard
            position), where yaw is in radians. The yaw range is read from
            self.regions[region_name]["yaw_ranges"] if it exists, otherwise
            defaults to (0.0, 360.0) degrees.

        Raises:
            ValueError: If regions list is empty or if any region has invalid bounds
        """
        assert self.regions is not None, "Regions must be defined"
        assert region_name in self.region_objects, f"Region '{region_name}' not found"

        region_config = self.regions[region_name]
        region_list = self.region_objects[region_name]

        # Randomly select one of the regions
        selected_region_index = np_random.choice(len(region_list))
        selected_region = region_list[selected_region_index]
        selected_bbox = selected_region.bbox

        # Get yaw range for this region
        yaw_range = (0.0, 360.0)  # Default range
        if "yaw_ranges" in region_config:
            yaw_ranges = region_config["yaw_ranges"]
            if yaw_ranges and len(yaw_ranges) > selected_region_index:
                yaw_range = tuple(yaw_ranges[selected_region_index])

        # Sample pose from the 3D bounding box (already in world coordinates)
        x, y, z, yaw = utils.sample_pose_in_bbox_3d(selected_bbox, np_random, yaw_range)

        return (x, y, z, yaw)

    def check_in_region(
        self,
        position: NDArray[np.float32],
        region_name: str,
        env: MujocoEnv | None = None,
    ) -> bool:
        """Check if a given position is within the specified region.

        This checks if the position is within the region's 3D bounding box.
        Args:
            position: Position as [x, y, z] array in world coordinates
            region_name: Name of the region to check
            env: Optional MujocoEnv instance for computing absolute site positions.
        Returns:
            True if the position is within the specified region, False otherwise
        """
        # Validate region exists
        assert self.regions is not None, "Regions must be defined"
        if region_name not in self.region_objects:
            raise ValueError(f"Region '{region_name}' not found")

        # Check if position is in any of the region objects
        region_list = self.region_objects[region_name]
        for region in region_list:
            if region.check_in_region(position, env):
                return True

        return False

    def get_object_centric_state(self) -> dict[Object, dict[str, Any]]:
        """Get object-centric state for the cupboard, including drawer joint positions.

        Returns:
            Dictionary with symbolic_object as key for the cupboard and drawer objects,
            and data (including drawer positions) as values. Each drawer gets its own
            symbolic object as a key with its slide position as the value.

        Note:
            This method retrieves drawer joint positions from the MuJoCo simulation
            if the environment is set. If not, drawer positions default to 0.0.
        """
        # Get base fixture state from parent class
        state = super().get_object_centric_state()

        # Create symbolic objects for each drawer and add to state
        for joint_name in self.drawer_joints:
            # Create drawer symbolic object if not already created
            if joint_name not in self._drawer_symbolic_objects:
                # Create a symbolic object for this drawer using MujocoDrawerObjectType
                drawer_name = joint_name.replace("_joint", "")
                self._drawer_symbolic_objects[joint_name] = Object(
                    drawer_name, MujocoDrawerObjectType
                )

            # Get the symbolic object for this drawer
            drawer_symbolic_object = self._drawer_symbolic_objects[joint_name]

            # Initialize drawer state with default position
            drawer_data = {"pos": 0.0}

            # Retrieve drawer position from simulation if environment is available
            if self.env is not None:
                try:
                    pos, _ = self.env.get_joint_pos_quat(joint_name)
                    # Extract first dimension for slide joint
                    pos_value = float(pos[0]) if hasattr(pos, "__len__") else float(pos)
                    drawer_data["pos"] = pos_value
                except (ValueError, KeyError, AttributeError, TypeError):
                    # Joint not found or environment doesn't support it
                    # Keep default value of 0.0
                    pass

            # Add drawer state to dictionary
            state[drawer_symbolic_object] = drawer_data

        return state

    def visualize_regions(self) -> None:
        """Visualize the cupboard's regions in the MuJoCo environment.

        This method is a no-op since regions are now added to the XML during
        _create_regions().
        """
        if self.regions is None:
            return

        for region_list in self.region_objects.values():
            for region in region_list:
                region.visualize_region()

    def __str__(self) -> str:
        """String representation of the cupboard."""
        return (
            f"Cupboard(name='{self.name}', length={self.cupboard_length}, "
            f"depth={self.cupboard_depth}, height={self.cupboard_height}, "
            f"num_shelves={self.num_shelves}, shelf_heights={self.shelf_heights})"
        )

    def __repr__(self) -> str:
        """Detailed string representation of the cupboard."""
        return (
            f"Cupboard(name='{self.name}', "
            f"length={self.cupboard_length}, depth={self.cupboard_depth}, "
            f"height={self.cupboard_height}, num_shelves={self.num_shelves}, "
            f"shelf_heights={self.shelf_heights}, "
            f"shelf_partitions={self.shelf_partitions}, "
            f"shelf_thickness={self.shelf_thickness}, "
            f"side_and_back_open={self.side_and_back_open}, "
            f"position={self.position}, leg_thickness={self.leg_thickness})"
        )
