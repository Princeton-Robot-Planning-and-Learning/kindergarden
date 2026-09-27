"""Primitive MuJoCo object classes such as Cube and Cuboid."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
from numpy.typing import NDArray
from relational_structs import Object

from kinder.envs.dynamic3d.mujoco_utils import MujocoEnv
from kinder.envs.dynamic3d.object_types import MujocoMovableObjectType
from kinder.envs.dynamic3d.objects.base import MujocoObject, register_object


@register_object
class Cuboid(MujocoObject):
    """A cuboid (rectangular box) object for TidyBot environments."""

    default_edge_size: float = 0.02  # Default edge size in meters

    def __init__(
        self,
        name: str,
        env: MujocoEnv | None = None,
        options: dict | None = None,
    ) -> None:
        """Initialize a Cuboid object.

        Args:
            name: Name of the cuboid body in the XML
            options: Dictionary of cuboid options:
                - size: [x, y, z] dimensions as a list of three floats
                - rgba: Color of the cuboid (either string or [r, g, b, a] values)
                - mass: Mass of the cuboid
            env: Reference to the environment (needed for position get/set operations)
        """
        # Initialize base class
        super().__init__(name, env, options)

        # Override object type
        self.symbolic_object = Object(self.name, MujocoMovableObjectType)

        # Handle size parameter - must be a list of 3 dimensions
        default_size = Cuboid.default_edge_size
        size = self.options.get(
            "size",
            [default_size, default_size, default_size],
        )
        if isinstance(size, (int, float)):
            # If scalar provided, treat as cube
            self.size = [size, size, size]
        else:
            # Expect a list of [x, y, z]
            self.size = list(size)
            if len(self.size) != 3:
                raise ValueError(
                    f"Cuboid size must be a list of 3 values [x, y, z], "
                    f"got {len(self.size)} values"
                )

        # Handle rgba parameter - can be string or list of values
        rgba = self.options.get("rgba", [0.5, 0.7, 0.5, 1])
        if isinstance(rgba, str):
            self.rgba = rgba
        else:
            self.rgba = " ".join(str(x) for x in rgba)

        # Handle mass parameter with default
        self.mass = self.options.get("mass", 0.1)

        # Create the XML element
        self.xml_element = self._create_xml_element()

        if self.regions is not None:
            self._create_regions()

    def _create_xml_element(self) -> ET.Element:
        """Create the XML Element for this cuboid.

        The cuboid is created as a single box geom centered at the body's origin.
        The origin (0, 0, 0) is located at the center of the cuboid.
        The cuboid extends by size[i]/2 in each direction
        (+/- x, +/- y, +/- z) from the origin.

        Returns:
            ET.Element representing the cuboid body
        """
        # Create body element
        body = ET.Element("body", name=self.name)

        # Add freejoint for position/orientation control
        ET.SubElement(body, "freejoint", name=self.joint_name)

        # Add geom element with cuboid properties
        size_str = " ".join(str(x) for x in self.size)
        ET.SubElement(
            body,
            "geom",
            type="box",
            size=size_str,
            # friction="2.0 0.2 0.02",
            rgba=self.rgba,
            mass=str(self.mass),
        )

        return body

    def __str__(self) -> str:
        """String representation of the cuboid."""
        return (
            f"Cuboid(name='{self.name}', size={self.size}, "
            f"rgba='{self.rgba}', mass={self.mass})"
        )

    def __repr__(self) -> str:
        """Detailed string representation of the cuboid."""
        return (
            f"Cuboid(name='{self.name}', joint_name='{self.joint_name}', "
            f"size={self.size}, rgba='{self.rgba}', mass={self.mass})"
        )

    def get_bounding_box_dimensions(self) -> tuple[float, float, float]:
        return (2 * self.size[0], 2 * self.size[1], 2 * self.size[2])

    @staticmethod
    def get_bounding_box_from_config(
        pos: NDArray[np.float32], object_config: dict[str, str | float]
    ) -> list[float]:
        """Get bounding box for a cuboid given its position and config.

        Args:
            pos: Position of the cuboid as [x, y, z] array
            object_config: Dictionary containing cuboid configuration with keys:
                - "size": Either a scalar (for cube) or [x, y, z] half-extents

        Returns:
            Bounding box as [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        # Handle size parameter - can be scalar or list of 3 dimensions
        default_size = Cuboid.default_edge_size
        size = object_config.get("size", default_size)

        if isinstance(size, (int, float)):
            # Scalar size - cube
            half_extents = [float(size), float(size), float(size)]
        else:
            # List of [x, y, z] half-extents
            half_extents = [float(s) for s in size]  # type: ignore[union-attr]
            if len(half_extents) != 3:
                raise ValueError(
                    f"Cuboid size must be a scalar or list of 3 values [x, y, z], "
                    f"got {len(half_extents)} values"
                )

        return [
            pos[0] - half_extents[0],  # x_min
            pos[1] - half_extents[1],  # y_min
            pos[2] - half_extents[2],  # z_min
            pos[0] + half_extents[0],  # x_max
            pos[1] + half_extents[1],  # y_max
            pos[2] + half_extents[2],  # z_max
        ]


@register_object
class Cube(Cuboid):
    """A cube object for TidyBot environments.

    This is a special case of Cuboid where all dimensions are equal.
    """

    def __init__(
        self,
        name: str,
        env: MujocoEnv | None = None,
        options: dict | None = None,
    ) -> None:
        """Initialize a Cube object.

        Args:
            name: Name of the cube body in the XML
            options: Dictionary of cube options:
                - size: Size of the cube (either scalar or [x, y, z] dimensions)
                - rgba: Color of the cube (either string or [r, g, b, a] values)
                - mass: Mass of the cube
            env: Reference to the environment (needed for position get/set operations)
        """
        # Normalize size to scalar if all dimensions are equal
        if options is None:
            options = {}

        size = options.get("size", Cuboid.default_edge_size)
        if isinstance(size, (int, float)):
            # Already scalar, keep as is
            pass
        else:
            # Convert to list to check dimensions
            size_list = list(size)
            if len(size_list) == 3 and size_list[0] == size_list[1] == size_list[2]:
                # All dimensions equal, use scalar
                options = dict(options)  # Create a copy
                options["size"] = size_list[0]

        # Initialize parent Cuboid class
        super().__init__(name, env, options)

    def __str__(self) -> str:
        """String representation of the cube."""
        return (
            f"Cube(name='{self.name}', size={self.size}, "
            f"rgba='{self.rgba}', mass={self.mass})"
        )

    def __repr__(self) -> str:
        """Detailed string representation of the cube."""
        return (
            f"Cube(name='{self.name}', joint_name='{self.joint_name}', "
            f"size={self.size}, rgba='{self.rgba}', mass={self.mass})"
        )


@register_object
class Bin(MujocoObject):
    """A bin (rectangular container with open top) object for TidyBot environments.

    The bin is constructed using multiple MuJoCo box primitives:
    - 1 bottom panel
    - 4 wall panels (front, back, left, right)
    """

    # Outer dimensions of the bin
    default_length: float = 0.1  # Default bin length in meters
    default_width: float = 0.1  # Default bin width in meters
    default_height: float = 0.05  # Default bin height in meters
    default_wall_thickness: float = 0.005  # Default wall thickness in meters

    def __init__(
        self,
        name: str,
        env: MujocoEnv | None = None,
        options: dict | None = None,
    ) -> None:
        """Initialize a Bin object.

        Args:
            name: Name of the bin body in the XML
            options: Dictionary of bin options:
                - length: Length of bin (x dimension, outer)
                - width: Width of bin (y dimension, outer)
                - height: Height of bin (z dimension)
                - wall_thickness: Thickness of walls (default: 0.005)
                - rgba: Color of the bin (either string or [r, g, b, a] values)
                - mass: Mass of the bin
                - solref: Optional MuJoCo contact solref applied to every bin geom
                - solimp: Optional MuJoCo contact solimp applied to every bin geom
            env: Reference to the environment (needed for position get/set operations)
        """
        # Initialize base class
        super().__init__(name, env, options)

        # Override object type
        self.symbolic_object = Object(self.name, MujocoMovableObjectType)

        # Bin dimensions
        self.length = float(
            self.options.get("length", Bin.default_length)
        )  # x dimension
        self.width = float(self.options.get("width", Bin.default_width))  # y dimension
        self.height = float(
            self.options.get("height", Bin.default_height)
        )  # z dimension
        self.wall_thickness = float(
            self.options.get("wall_thickness", Bin.default_wall_thickness)
        )

        # Handle rgba parameter - can be string or list of values
        rgba = self.options.get("rgba", [0.5, 0.5, 0.5, 1])
        if isinstance(rgba, str):
            self.rgba = rgba
        else:
            self.rgba = " ".join(str(x) for x in rgba)

        # Handle mass parameter with default
        self.mass = self.options.get("mass", 0.1)

        # MuJoCo's default contact is soft enough that a fast-falling object can
        # sink past the floor panel's mid-plane, flipping the contact normal.
        self.solref: list[float] | None = self.options.get("solref")
        self.solimp: list[float] | None = self.options.get("solimp")

        # Create the XML element
        self.xml_element = self._create_xml_element()

        if self.regions is not None:
            self._create_regions()

    def _create_xml_element(self) -> ET.Element:
        """Create the XML Element for this bin using multiple box geoms.

        The bin is constructed from 5 box geoms:
        - 1 bottom panel: full outer dimensions (length x width),
          at z in [0, wall_thickness]
        - 4 wall panels: back, front, left, right walls with thickness
          wall_thickness, extending from z = wall_thickness to z = height

        The origin (0, 0, 0) is located at the base center of the bin
        (center of bottom surface). The bin extends in the positive z direction.

        Returns:
            ET.Element representing the bin body with all geoms
        """
        # Create body element
        body = ET.Element("body", name=self.name)

        # Add freejoint for position/orientation control
        ET.SubElement(body, "freejoint", name=self.joint_name)

        # Calculate half dimensions
        half_length = self.length / 2
        half_width = self.width / 2
        half_wall = self.wall_thickness / 2

        # Calculate inner dimensions
        inner_half_width = half_width - self.wall_thickness

        # Wall height (excluding bottom thickness)
        wall_height = self.height - self.wall_thickness
        half_wall_height = wall_height / 2

        # Mass distribution (divide among 5 components)
        component_mass = self.mass / 5.0

        # Bottom panel (full outer dimensions, at z = 0 to z = wall_thickness)
        bottom_size = [half_length, half_width, half_wall]
        bottom_pos = [0.0, 0.0, half_wall]
        ET.SubElement(
            body,
            "geom",
            type="box",
            size=" ".join(str(x) for x in bottom_size),
            pos=" ".join(str(x) for x in bottom_pos),
            rgba=self.rgba,
            mass=str(component_mass),
        )

        # Back wall (along x-axis, at -y edge)
        back_wall_size = [half_length, half_wall, half_wall_height]
        back_wall_z = self.wall_thickness + half_wall_height
        back_wall_pos = [0.0, -half_width + half_wall, back_wall_z]
        ET.SubElement(
            body,
            "geom",
            type="box",
            size=" ".join(str(x) for x in back_wall_size),
            pos=" ".join(str(x) for x in back_wall_pos),
            rgba=self.rgba,
            mass=str(component_mass),
        )

        # Front wall (along x-axis, at +y edge)
        front_wall_size = [half_length, half_wall, half_wall_height]
        front_wall_z = self.wall_thickness + half_wall_height
        front_wall_pos = [0.0, half_width - half_wall, front_wall_z]
        ET.SubElement(
            body,
            "geom",
            type="box",
            size=" ".join(str(x) for x in front_wall_size),
            pos=" ".join(str(x) for x in front_wall_pos),
            rgba=self.rgba,
            mass=str(component_mass),
        )

        # Left wall (along y-axis, at -x edge)
        left_wall_size = [half_wall, inner_half_width, half_wall_height]
        left_wall_z = self.wall_thickness + half_wall_height
        left_wall_pos = [-half_length + half_wall, 0.0, left_wall_z]
        ET.SubElement(
            body,
            "geom",
            type="box",
            size=" ".join(str(x) for x in left_wall_size),
            pos=" ".join(str(x) for x in left_wall_pos),
            rgba=self.rgba,
            mass=str(component_mass),
        )

        # Right wall (along y-axis, at +x edge)
        right_wall_size = [half_wall, inner_half_width, half_wall_height]
        right_wall_z = self.wall_thickness + half_wall_height
        right_wall_pos = [half_length - half_wall, 0.0, right_wall_z]
        ET.SubElement(
            body,
            "geom",
            type="box",
            size=" ".join(str(x) for x in right_wall_size),
            pos=" ".join(str(x) for x in right_wall_pos),
            rgba=self.rgba,
            mass=str(component_mass),
        )

        # Priority makes these parameters govern the contact instead of being
        # averaged with the other geom's defaults.
        if self.solref is not None or self.solimp is not None:
            for geom in body.iter("geom"):
                if self.solref is not None:
                    geom.set("solref", " ".join(str(x) for x in self.solref))
                if self.solimp is not None:
                    geom.set("solimp", " ".join(str(x) for x in self.solimp))
                geom.set("priority", "1")

        return body

    def get_bounding_box_dimensions(self) -> tuple[float, float, float]:
        """Get the bounding box dimensions for this bin.

        Returns:
            Tuple of (length, width, height) for the bounding box
        """
        return (self.length, self.width, self.height)

    @staticmethod
    def get_bounding_box_from_config(
        pos: NDArray[np.float32], object_config: dict[str, str | float]
    ) -> list[float]:
        """Get bounding box for a bin given its position and config.

        Args:
            pos: Position of the bin base as [x, y, z] array
            object_config: Dictionary containing bin configuration with keys:
                - "length": Length of bin (x dimension)
                - "width": Width of bin (y dimension)
                - "height": Height of bin (z dimension)

        Returns:
            Bounding box as [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        # Extract bin parameters
        length = float(object_config.get("length", 0.1))
        width = float(object_config.get("width", 0.1))
        height = float(object_config.get("height", 0.05))

        # Half-extents
        half_length = length / 2
        half_width = width / 2

        return [
            pos[0] - half_length,  # x_min
            pos[1] - half_width,  # y_min
            pos[2],  # z_min (at base)
            pos[0] + half_length,  # x_max
            pos[1] + half_width,  # y_max
            pos[2] + height,  # z_max
        ]

    def __str__(self) -> str:
        """String representation of the bin."""
        return (
            f"Bin(name='{self.name}', length={self.length}, "
            f"width={self.width}, height={self.height}, "
            f"wall_thickness={self.wall_thickness})"
        )

    def __repr__(self) -> str:
        """Detailed string representation of the bin."""
        return (
            f"Bin(name='{self.name}', joint_name='{self.joint_name}', "
            f"length={self.length}, width={self.width}, height={self.height}, "
            f"wall_thickness={self.wall_thickness}, rgba='{self.rgba}', "
            f"mass={self.mass})"
        )


@register_object
class Wiper(MujocoObject):
    """A wiper object composed of a long handle and a perpendicular blade head."""

    default_handle_width: float = 0.01  # Default handle width in meters
    default_handle_height: float = 0.01  # Default handle height in meters
    default_head_length: float = 0.15  # Default head length in meters
    default_head_height: float = 0.01  # Default head height in meters
    default_upright: bool = True  # If True, wiper is standing upright

    def __init__(
        self,
        name: str,
        env: MujocoEnv | None = None,
        options: dict | None = None,
    ) -> None:
        """Initialize a Wiper object.

        Args:
            name: Name of the wiper body in the XML
            options: Dictionary of wiper options:
                - handle_width: Width of the handle in both x and y dimensions
                  (default: 0.01)
                - handle_height: Height of the handle in z dimension (default: 0.01)
                - head_length: Length of the blade head in x dimension (default: 0.15)
                - head_height: Height of the blade head in z dimension (default: 0.01)
                - handle_rgba: Color of the handle (default: [0.5, 0.5, 0.5, 1])
                - head_rgba: Color of the head (default: [0.5, 0.5, 0.5, 1])
                - mass: Mass of the wiper (default: 0.1)
            env: Reference to the environment
        """
        super().__init__(name, env, options)

        # Override object type
        self.symbolic_object = Object(self.name, MujocoMovableObjectType)

        # Handle parameters
        self.handle_width = float(self.options.get("handle_width", 0.01))
        self.handle_height = float(self.options.get("handle_height", 0.01))

        # Blade head parameters
        self.head_length = float(self.options.get("head_length", 0.15))
        self.head_height = float(self.options.get("head_height", 0.01))
        self.upright = bool(self.options.get("upright", Wiper.default_upright))

        # Handle rgba parameter - can be string or list of values
        handle_rgba = self.options.get("handle_rgba", [0.5, 0.5, 0.5, 1])
        if isinstance(handle_rgba, str):
            self.handle_rgba = handle_rgba
        else:
            self.handle_rgba = " ".join(str(x) for x in handle_rgba)

        # Head rgba parameter - can be string or list of values
        head_rgba = self.options.get("head_rgba", [0.5, 0.5, 0.5, 1])
        if isinstance(head_rgba, str):
            self.head_rgba = head_rgba
        else:
            self.head_rgba = " ".join(str(x) for x in head_rgba)

        # Handle mass parameter with default
        self.mass = self.options.get("mass", 0.1)

        # Create the XML element
        self.xml_element = self._create_xml_element()

        if self.regions is not None:
            self._create_regions()

    def _create_xml_element(self) -> ET.Element:
        """Create the XML Element for this wiper.

        The wiper consists of two geoms: a handle and a blade head.

        When upright=True (handle extends upward in z, head extends horizontally):
        - Handle geom:
            size = [handle_width/2, handle_width/2, handle_height/2]
            pos = [0, 0, handle_height/2 + head_height]
            extends x: [-handle_width/2, handle_width/2]
            extends y: [-handle_width/2, handle_width/2]
            extends z: [head_height, head_height + handle_height]
        - Head geom (blade at bottom, perpendicular to handle):
            size = [head_length/2, handle_width/2, head_height/2]
            pos = [0, 0, head_height/2]
            extends x: [-head_length/2, head_length/2]
            extends y: [-handle_width/2, handle_width/2]
            extends z: [0, head_height]

        When upright=False (handle extends horizontally in x, head extends in y):
        - Handle geom (rod along x):
            size = [handle_height/2, handle_width/2, handle_width/2]
            pos = [head_height/2, 0, handle_width/2]
            extends x: [0, handle_height]
            extends y: [-handle_width/2, handle_width/2]
            extends z: [0, handle_width]
        - Head geom (blade perpendicular to handle, extends in y):
            size = [head_height/2, head_length/2, handle_width/2]
            pos = [-handle_height/2, 0, handle_width/2]
            extends x: [-handle_height, 0]
            extends y: [-head_length/2, head_length/2]
            extends z: [0, handle_width]

        Returns:
            ET.Element representing the wiper body with both geoms
        """
        # Create body element
        body = ET.Element("body", name=self.name)

        # Add freejoint for position/orientation control
        ET.SubElement(body, "freejoint", name=self.joint_name)

        # Mass distribution (divide between handle and head)
        component_mass = self.mass / 2.0

        if self.upright:
            # Upright orientation: handle extends in z, head extends in x
            # Handle: a box with square cross-section in x-y plane
            # MuJoCo box size is half-extent in each direction
            hw = self.handle_width / 2
            hh = self.handle_height / 2
            handle_size = f"{hw} {hw} {hh}"
            handle_pos_z = self.handle_height / 2 + self.head_height
            ET.SubElement(
                body,
                "geom",
                type="box",
                size=handle_size,
                pos=f"0 0 {handle_pos_z}",
                rgba=self.handle_rgba,
                mass=str(component_mass),
            )

            # Blade head: box at the end of the handle
            # Position: at the end of the handle along x-axis
            head_pos = f"0 0 {self.head_height / 2}"
            # Size: head_length in x, width in y, and head_height in z
            hl = self.head_length / 2
            hw = self.handle_width / 2
            hh = self.head_height / 2
            head_size = f"{hl} {hw} {hh}"
            ET.SubElement(
                body,
                "geom",
                type="box",
                size=head_size,
                pos=head_pos,
                rgba=self.head_rgba,
                mass=str(component_mass),
            )
        else:
            # Horizontal orientation: handle extends in x, head extends in y
            # Handle: rod extending along x-axis
            # size = [handle_height/2, handle_width/2, handle_width/2]
            # pos = [handle_height/2, 0, handle_width/2]
            hh = self.handle_height / 2
            hw = self.handle_width / 2
            hd = self.head_height / 2
            handle_size = f"{hh} {hw} {hw}"
            handle_pos = f"{hd} 0 {hw}"
            ET.SubElement(
                body,
                "geom",
                type="box",
                size=handle_size,
                pos=handle_pos,
                rgba=self.handle_rgba,
                mass=str(component_mass),
            )

            # Blade head: box extending along y-axis at end of handle
            # size = [head_height/2, head_length/2, handle_width/2]
            # pos = [-handle_height/2, 0, handle_width/2]
            hd = self.head_height / 2
            hl = self.head_length / 2
            hw = self.handle_width / 2
            hh = self.handle_height / 2
            head_size = f"{hd} {hl} {hw}"
            head_pos = f"{-hh} 0 {hw}"
            ET.SubElement(
                body,
                "geom",
                type="box",
                size=head_size,
                pos=head_pos,
                rgba=self.head_rgba,
                mass=str(component_mass),
            )

        return body

    def get_bounding_box_dimensions(self) -> tuple[float, float, float]:
        """Get the bounding box dimensions for this wiper.

        Returns:
            Tuple of (length, width, height) encompassing both handle and blade
        """
        if self.upright:
            # Upright: handle extends in z, head extends in x
            total_length = self.head_length
            total_width = self.handle_width
            total_height = self.head_height + self.handle_height
        else:
            # Horizontal: handle extends in x, head extends in y
            # Total x is handle_height + head_height (from handle to head)
            # Total y is head_length
            # Total z is handle_width
            total_length = self.handle_height + self.head_height
            total_width = self.head_length
            total_height = self.handle_width

        return (total_length, total_width, total_height)

    @staticmethod
    def get_bounding_box_from_config(
        pos: NDArray[np.float32], object_config: dict[str, str | float]
    ) -> list[float]:
        """Get bounding box for a wiper given its position and config.

        Args:
            pos: Position of the wiper as [x, y, z] array
            object_config: Dictionary containing wiper configuration with keys:
                - "handle_width": Width of the handle in x and y dimensions
                - "handle_height": Height of the handle in z dimension
                - "head_length": Length of the blade head in x dimension
                - "head_height": Height of the blade head in z dimension
                - "upright": Boolean indicating orientation
                  (True = upright, False = horizontal)

        Returns:
            Bounding box as [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        # Extract wiper parameters
        handle_width = float(object_config.get("handle_width", 0.01))
        handle_height = float(object_config.get("handle_height", 0.01))
        head_length = float(object_config.get("head_length", 0.15))
        head_height = float(object_config.get("head_height", 0.01))
        upright = bool(object_config.get("upright", Wiper.default_upright))

        if upright:
            # Upright orientation: handle extends in z, head extends in x
            # Head geom: size=[head_length/2, handle_width/2, head_height/2],
            # pos=[0, 0, head_height/2]
            #   Extends x: ±head_length/2, y: ±handle_width/2, z: [0, head_height]
            # Handle geom: size=[handle_width/2, handle_width/2, handle_height/2],
            # pos=[0, 0, head_height + handle_height/2]
            #   Extends x: ±handle_width/2, y: ±handle_width/2,
            #   z: [head_height, head_height + handle_height]

            # Overall bounds relative to body origin:
            # x: [-head_length/2, head_length/2] (head is longer)
            # y: [-handle_width/2, handle_width/2]
            # z: [0, head_height + handle_height]

            x_min = pos[0] - head_length / 2
            x_max = pos[0] + head_length / 2

            y_min = pos[1] - handle_width / 2
            y_max = pos[1] + handle_width / 2

            z_min = pos[2]
            z_max = pos[2] + head_height + handle_height
        else:
            # Horizontal orientation: handle extends in x, head extends in y
            # Handle geom: size=[handle_height/2, handle_width/2, handle_width/2],
            # pos=[head_height/2, 0, handle_width/2]
            #   Extends x: [head_height/2 - handle_height/2,
            #              head_height/2 + handle_height/2],
            #   y: ±handle_width/2, z: [0, handle_width]
            # Head geom: size=[head_height/2, head_length/2, handle_width/2],
            # pos=[-handle_height/2, 0, handle_width/2]
            #   Extends x: [-handle_height/2 - head_height/2,
            #              -handle_height/2 + head_height/2],
            #   y: ±head_length/2, z: [0, handle_width]

            # Overall bounds relative to body origin:
            # x: [-(handle_height + head_height)/2,
            #     (handle_height + head_height)/2]
            # y: [-head_length/2, head_length/2]
            # z: [0, handle_width]

            x_min = pos[0] - (handle_height + head_height) / 2
            x_max = pos[0] + (handle_height + head_height) / 2

            y_min = pos[1] - head_length / 2
            y_max = pos[1] + head_length / 2

            z_min = pos[2]
            z_max = pos[2] + handle_width

        return [x_min, y_min, z_min, x_max, y_max, z_max]

    def __str__(self) -> str:
        """String representation of the wiper."""
        return (
            f"Wiper(name='{self.name}', handle_width={self.handle_width}, "
            f"handle_height={self.handle_height}, head_length={self.head_length}, "
            f"head_height={self.head_height}, handle_rgba='{self.handle_rgba}', "
            f"head_rgba='{self.head_rgba}', mass={self.mass})"
        )

    def __repr__(self) -> str:
        """Detailed string representation of the wiper."""
        return (
            f"Wiper(name='{self.name}', joint_name='{self.joint_name}', "
            f"handle_width={self.handle_width}, handle_height={self.handle_height}, "
            f"head_length={self.head_length}, head_height={self.head_height}, "
            f"handle_rgba='{self.handle_rgba}', head_rgba='{self.head_rgba}', "
            f"mass={self.mass})"
        )


@register_object
class Scoop(MujocoObject):
    """A scoop object composed of a small bin with walls and a handle.

    The scoop consists of:
    - A bottom panel
    - 4 wall panels (front, back, left, right)
    - A handle attached at the back of the bin

    Coordinate system (body origin at 0, 0, 0):
    - x-axis: handle extends from -handle_length to 0, bin extends from 0 to
        (length + 2*wall_width)
    - y-axis: bin extends symmetrically from -(width + wall_width)/2 to
        +(width + wall_width)/2
    - z-axis: bin bottom at 0, extends to (height + wall_width)
    """

    # Inner dimensions of the scoop
    default_wall_width: float = 0.005  # Default wall thickness in meters
    default_length: float = 0.1  # Default bin length in meters
    default_width: float = 0.1  # Default bin width in meters
    default_height: float = 0.05  # Default bin height in meters
    default_handle_length: float = 0.05  # Default handle length in meters
    default_handle_width: float = 0.02  # Default handle width in meters
    default_handle_height: float = 0.02  # Default handle height in meters

    def __init__(
        self,
        name: str,
        env: MujocoEnv | None = None,
        options: dict | None = None,
    ) -> None:
        """Initialize a Scoop object.

        Args:
            name: Name of the scoop body in the XML
            options: Dictionary of scoop options:
                - length: Length of bin (x dimension, inner)
                - width: Width of bin (y dimension, inner)
                - height: Height of bin (z dimension)
                - wall_width: Thickness of walls (default: 0.005)
                - handle_length: Length of the handle (y direction)
                - handle_width: Width of the handle (x direction)
                - handle_height: Height of the handle (z direction)
                - rgba: Color of the scoop (default: [0.5, 0.5, 0.5, 1])
                - handle_rgba: Color of the handle (default: [0.5, 0.5, 0.5, 1])
                - mass: Total mass of the scoop (default: 0.1)
            env: Reference to the environment
        """
        super().__init__(name, env, options)

        # Override object type
        self.symbolic_object = Object(self.name, MujocoMovableObjectType)

        # Bin dimensions
        self.length = float(self.options.get("length", Scoop.default_length))
        self.width = float(self.options.get("width", Scoop.default_width))
        self.height = float(self.options.get("height", Scoop.default_height))
        self.wall_width = float(
            self.options.get("wall_width", Scoop.default_wall_width)
        )

        # Handle dimensions
        self.handle_length = float(
            self.options.get("handle_length", Scoop.default_handle_length)
        )
        self.handle_width = float(
            self.options.get("handle_width", Scoop.default_handle_width)
        )
        self.handle_height = float(
            self.options.get("handle_height", Scoop.default_handle_height)
        )

        # Handle rgba parameter - can be string or list of values
        handle_rgba = self.options.get("handle_rgba", [0.5, 0.5, 0.5, 1])
        if isinstance(handle_rgba, str):
            self.handle_rgba = handle_rgba
        else:
            self.handle_rgba = " ".join(str(x) for x in handle_rgba)

        # Bin rgba parameter - can be string or list of values
        rgba = self.options.get("rgba", [0.5, 0.5, 0.5, 1])
        if isinstance(rgba, str):
            self.rgba = rgba
        else:
            self.rgba = " ".join(str(x) for x in rgba)

        # Handle mass parameter with default
        self.mass = self.options.get("mass", 0.1)

        # Create the XML element
        self.xml_element = self._create_xml_element()

        if self.regions is not None:
            self._create_regions()

    def _create_xml_element(self) -> ET.Element:
        """Create the XML Element for this scoop.

        The scoop consists of:
        - 1 bottom panel: full outer dimensions including walls
          size = [outer_length/2, outer_width/2, wall_width/2]
          pos = [outer_length/2, 0, wall_width/2]
          extends x: [0, outer_length], y: [-outer_width/2, outer_width/2],
          z: [0, wall_width]

        - 4 wall panels: back, front, left, right walls extending from
          z = wall_width to z = wall_width + height
          * back wall (y = -outer_width/2):
            size = [outer_length/2, wall_width/2, height/2]
            extends x: [0, outer_length],
            y: [-outer_width/2, -outer_width/2 + wall_width]
          * front wall (y = +outer_width/2):
            size = [outer_length/2, wall_width/2, height/2]
            extends x: [0, outer_length],
            y: [outer_width/2 - wall_width, outer_width/2]
          * left wall (x = 0):
            size = [wall_width/2, width/2, height/2]
            extends x: [0, wall_width], y: [-width/2, width/2]
          * right wall (x = outer_length):
            size = [wall_width/2, width/2, height/2]
            extends x: [outer_length - wall_width, outer_length], y: [-width/2, width/2]

        - 1 handle: attached at the back-left corner, extending backward (negative x)
          size = [handle_length/2, handle_width/2, handle_height/2]
          pos = [-handle_length/2, 0, height + wall_width - handle_height/2]
          extends x: [-handle_length, 0], y: [-handle_width/2, handle_width/2],
          z: [height + wall_width - handle_height, height + wall_width]

        Returns:
            ET.Element representing the scoop body with all geoms
        """
        # Create body element
        body = ET.Element("body", name=self.name)

        # Add freejoint for position/orientation control
        ET.SubElement(body, "freejoint", name=self.joint_name)

        # Calculate half dimensions
        half_width = self.width / 2
        half_wall_width = self.wall_width / 2

        # Total outer dimensions (including walls)
        outer_length = self.length
        outer_half_length = outer_length / 2
        outer_width = self.width
        outer_half_width = outer_width / 2

        # Mass distribution
        bin_component_mass = (self.mass * 0.8) / 5.0  # 80% to bin
        handle_mass = self.mass * 0.2  # 20% to handle

        # Bottom panel (full outer dimensions)
        # Center at [(outer_length)/2, 0, wall_width/2]
        # extends from x: [0, outer_length], y: [-outer_width/2, outer_width/2]
        bottom_size = [outer_half_length, outer_half_width, half_wall_width]
        bottom_pos = [outer_half_length, 0.0, half_wall_width]
        ET.SubElement(
            body,
            "geom",
            type="box",
            size=" ".join(str(x) for x in bottom_size),
            pos=" ".join(str(x) for x in bottom_pos),
            rgba=self.rgba,
            mass=str(bin_component_mass),
        )

        # Wall heights (excluding bottom thickness)
        wall_height = self.height
        half_wall_height = wall_height / 2
        wall_z = self.wall_width + half_wall_height

        # Back wall (at y = -outer_width/2)
        back_wall_size = [outer_half_length, half_wall_width, half_wall_height]
        back_wall_pos = [outer_half_length, -outer_half_width + half_wall_width, wall_z]
        ET.SubElement(
            body,
            "geom",
            type="box",
            size=" ".join(str(x) for x in back_wall_size),
            pos=" ".join(str(x) for x in back_wall_pos),
            rgba=self.rgba,
            mass=str(bin_component_mass),
        )

        # Front wall (at y = +outer_width/2)
        front_wall_size = back_wall_size
        front_wall_pos = [outer_half_length, outer_half_width - half_wall_width, wall_z]
        ET.SubElement(
            body,
            "geom",
            type="box",
            size=" ".join(str(x) for x in front_wall_size),
            pos=" ".join(str(x) for x in front_wall_pos),
            rgba=self.rgba,
            mass=str(bin_component_mass),
        )

        # Left wall (at x = 0, inner half-width = width/2)
        left_wall_size = [half_wall_width, half_width, half_wall_height]
        left_wall_pos = [half_wall_width, 0.0, wall_z]
        ET.SubElement(
            body,
            "geom",
            type="box",
            size=" ".join(str(x) for x in left_wall_size),
            pos=" ".join(str(x) for x in left_wall_pos),
            rgba=self.rgba,
            mass=str(bin_component_mass),
        )

        # # Right wall (at x = outer_length)
        # right_wall_size = [half_wall_width, half_width, half_wall_height]
        # right_wall_pos = [outer_length - half_wall_width, 0.0, wall_z]
        # ET.SubElement(
        #     body,
        #     "geom",
        #     type="box",
        #     size=" ".join(str(x) for x in right_wall_size),
        #     pos=" ".join(str(x) for x in right_wall_pos),
        #     rgba=self.rgba,
        #     mass=str(bin_component_mass),
        # )

        # Handle: positioned at the back-left corner, extending along y
        # Handle position: at x=0, y extends from -(wall_width + handle_length)/2
        # to +(wall_width + handle_length)/2 (for symmetry)
        # z extends from height-handle_height to height (top flush with wall top)
        handle_half_length = self.handle_length / 2
        handle_half_width = self.handle_width / 2
        handle_half_height = self.handle_height / 2

        # Handle center position
        handle_x = -handle_half_length
        handle_y = 0.0  # Centered in y at back
        handle_z = self.height + self.wall_width - handle_half_height

        handle_size = [handle_half_length, handle_half_width, handle_half_height]
        handle_pos = [handle_x, handle_y, handle_z]
        ET.SubElement(
            body,
            "geom",
            type="box",
            size=" ".join(str(x) for x in handle_size),
            pos=" ".join(str(x) for x in handle_pos),
            rgba=self.handle_rgba,
            mass=str(handle_mass),
        )

        return body

    def get_bounding_box_dimensions(self) -> tuple[float, float, float]:
        """Get the bounding box dimensions for this scoop.

        Returns:
            Tuple of (length, width, height) including walls and handle
        """
        return (
            self.length + 2 * self.wall_width + self.handle_length,
            self.width + 2 * self.wall_width,
            self.height + self.wall_width,
        )

    @staticmethod
    def get_bounding_box_from_config(
        pos: NDArray[np.float32], object_config: dict[str, str | float]
    ) -> list[float]:
        """Get bounding box for a scoop given its position and config.

        Args:
            pos: Position of the scoop body center as [x, y, z]
            object_config: Dictionary containing scoop configuration with keys:
                - "length": Inner length of bin (x dimension)
                - "width": Inner width of bin (y dimension)
                - "height": Height of bin (z dimension)
                - "wall_width": Thickness of walls
                - "handle_length": Length of the handle
                - "handle_height": Height of the handle

        Returns:
            Bounding box as [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        # Extract scoop parameters
        length = float(object_config.get("length", 0.1))
        width = float(object_config.get("width", 0.1))
        height = float(object_config.get("height", 0.05))
        wall_width = float(object_config.get("wall_width", Scoop.default_wall_width))
        handle_length = float(object_config.get("handle_length", 0.05))

        # Outer dimensions (including walls)
        outer_length = length + 2 * wall_width
        outer_width = width + 2 * wall_width
        outer_half_width = outer_width / 2

        # Compute bounding box relative to body origin, then add pos offset
        # x extent: handle extends backward to -handle_length, bin extends
        # forward to outer_length
        x_min_relative = -handle_length
        x_max_relative = outer_length

        # y extent: symmetric around center
        y_min_relative = -outer_half_width
        y_max_relative = outer_half_width

        # z extent: bottom panel starts at 0, top of handle is at
        # height + wall_width
        z_min_relative = 0
        z_max_relative = height + wall_width

        return [
            pos[0] + x_min_relative,
            pos[1] + y_min_relative,
            pos[2] + z_min_relative,
            pos[0] + x_max_relative,
            pos[1] + y_max_relative,
            pos[2] + z_max_relative,
        ]

    def __str__(self) -> str:
        """String representation of the scoop."""
        return (
            f"Scoop(name='{self.name}', length={self.length}, "
            f"width={self.width}, height={self.height}, "
            f"wall_width={self.wall_width}, handle_length={self.handle_length}, "
            f"handle_width={self.handle_width}, handle_height={self.handle_height})"
        )

    def __repr__(self) -> str:
        """Detailed string representation of the scoop."""
        return (
            f"Scoop(name='{self.name}', joint_name='{self.joint_name}', "
            f"length={self.length}, width={self.width}, height={self.height}, "
            f"wall_width={self.wall_width}, handle_length={self.handle_length}, "
            f"handle_width={self.handle_width}, handle_height={self.handle_height}, "
            f"rgba='{self.rgba}', handle_rgba='{self.handle_rgba}', "
            f"mass={self.mass})"
        )
