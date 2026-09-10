"""RoboCasa object classes dynamically generated from model files."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from relational_structs import Object

from kinder.envs.dynamic3d.mujoco_utils import MujocoEnv
from kinder.envs.dynamic3d.object_types import MujocoMovableObjectType
from kinder.envs.dynamic3d.objects.base import (
    REGISTERED_OBJECTS,
    MujocoObject,
    register_object,
)
from kinder.envs.dynamic3d.objects.utils import euler_to_quat

# Get the path to the robocasa objects directory relative to this file
ROBOCASA_OBJECTS_DIR = (
    Path(__file__).parent.parent / "models" / "assets" / "robocasa_objects"
)
REPLICA_OBJECTS_DIR = (
    Path(__file__).parent.parent / "models" / "assets" / "replica_objects"
)


class RoboCasaObject(MujocoObject):
    """Base class for RoboCasa objects loaded from model.xml files."""

    # Class variables to be set by subclasses
    object_type_name: ClassVar[str] = ""
    model_dir: ClassVar[Path] = Path()

    def __init__(
        self,
        name: str,
        env: MujocoEnv | None = None,
        options: dict | None = None,
    ) -> None:
        """Initialize a RoboCasa object.

        Args:
            name: Name of the object body in the XML
            env: Reference to the environment
            options: Dictionary of object options
        """
        # Initialize base class
        super().__init__(name, env, options)

        # Override object type
        self.symbolic_object = Object(self.name, MujocoMovableObjectType)

        # Load the model.xml file and parse it
        model_xml_path = self.model_dir / "model.xml"
        if not model_xml_path.exists():
            raise FileNotFoundError(
                f"Model XML not found for {self.object_type_name}: {model_xml_path}"
            )

        # Parse the model.xml to extract asset and body information
        self.model_tree = ET.parse(str(model_xml_path))
        self.model_root = self.model_tree.getroot()

        # Extract assets (meshes, textures, materials)
        self.assets = self._extract_assets()

        # Create the XML element for this object
        self.xml_element = self._create_xml_element()

        # Calculate bounding box from sites in the model
        self.bounding_box = self._calculate_bounding_box()

        if self.regions is not None:
            self._create_regions()

    def _extract_assets(self) -> ET.Element:
        """Extract asset elements from the model.xml.

        Returns:
            ET.Element containing all assets (meshes, textures, materials)
        """
        asset_element = self.model_root.find("asset")
        if asset_element is None:
            raise ValueError(
                f"No asset element found in model.xml for {self.object_type_name}"
            )

        # Create a new asset container
        assets_container = ET.Element("assets")

        # Copy all mesh, texture, and material elements
        for child in asset_element:
            if child.tag in ["mesh", "texture", "material"]:
                # Create a copy of the element
                new_element = ET.Element(child.tag, child.attrib.copy())
                new_element.text = child.text
                new_element.tail = child.tail

                # Update file paths to be absolute
                if child.tag == "mesh" and "file" in child.attrib:
                    # Create absolute path to the mesh file
                    original_file = child.attrib["file"]
                    absolute_file = str(self.model_dir / original_file)
                    new_element.attrib["file"] = absolute_file

                    # Make mesh name unique by prefixing with object instance name
                    if "name" in child.attrib:
                        original_name = child.attrib["name"]
                        new_element.attrib["name"] = f"{self.name}_{original_name}"

                elif child.tag == "texture" and "file" in child.attrib:
                    # Create absolute path to the texture file
                    original_file = child.attrib["file"]
                    absolute_file = str(self.model_dir / original_file)
                    new_element.attrib["file"] = absolute_file

                    # Make texture name unique
                    if "name" in child.attrib:
                        original_name = child.attrib["name"]
                        new_element.attrib["name"] = f"{self.name}_{original_name}"

                elif child.tag == "material":
                    # Make material name unique
                    if "name" in child.attrib:
                        original_name = child.attrib["name"]
                        new_element.attrib["name"] = f"{self.name}_{original_name}"

                    # Update texture reference if present
                    if "texture" in child.attrib:
                        original_texture = child.attrib["texture"]
                        new_element.attrib["texture"] = (
                            f"{self.name}_{original_texture}"
                        )

                assets_container.append(new_element)

        return assets_container

    def _create_xml_element(self) -> ET.Element:
        """Create the XML Element for this RoboCasa object.

        Returns:
            ET.Element representing the object body
        """
        # Find the worldbody in the model
        worldbody = self.model_root.find("worldbody")
        if worldbody is None:
            raise ValueError(
                f"No worldbody found in model.xml for {self.object_type_name}"
            )

        # Find the object body (should be nested inside worldbody)
        # Structure is typically: worldbody -> body -> body[name="object"]
        object_body = None
        for body in worldbody.iter("body"):
            if body.attrib.get("name") == "object":
                object_body = body
                break

        if object_body is None:
            raise ValueError(
                f"No body with name='object' found in model.xml for "
                f"{self.object_type_name}"
            )

        # Create a new body element with our object name
        new_body = ET.Element("body", name=self.name)

        # Add freejoint for position/orientation control
        ET.SubElement(new_body, "freejoint", name=self.joint_name)

        # Copy all geom elements from the object body and update mesh references
        for geom in object_body.findall("geom"):
            new_geom = ET.Element("geom", geom.attrib.copy())
            new_geom.text = geom.text
            new_geom.tail = geom.tail

            # Update mesh reference to use the prefixed name
            if "mesh" in geom.attrib:
                original_mesh = geom.attrib["mesh"]
                new_geom.attrib["mesh"] = f"{self.name}_{original_mesh}"

            # Update material reference to use the prefixed name
            if "material" in geom.attrib:
                original_material = geom.attrib["material"]
                new_geom.attrib["material"] = f"{self.name}_{original_material}"

            new_body.append(new_geom)

        return new_body

    def _calculate_bounding_box(self) -> tuple[float, float, float]:
        """Calculate bounding box dimensions from site elements in the model.

        Returns:
            Tuple of (width, depth, height) for the bounding box
        """
        # Find the worldbody in the model
        worldbody = self.model_root.find("worldbody")
        if worldbody is None:
            # Default fallback bounding box
            return (0.05, 0.05, 0.05)

        # Look for site elements that define the bounding box
        # Sites are typically named: bottom_site, top_site, horizontal_radius_site
        sites: dict[str, Any] = {}
        for body in worldbody.iter("body"):
            for site in body.findall("site"):
                site_name = site.attrib.get("name", "")
                pos_str = site.attrib.get("pos", "0 0 0")
                pos = np.array([float(x) for x in pos_str.split()])
                sites[site_name] = pos

        # Calculate bounding box from sites
        if "bottom_site" in sites and "top_site" in sites:
            # Height from bottom to top
            height = abs(sites["top_site"][2] - sites["bottom_site"][2])
        else:
            height = 0.05  # Default

        if "horizontal_radius_site" in sites:
            # Width and depth from horizontal radius
            # The horizontal_radius_site gives the maximum extent in x and y
            radius_pos = sites["horizontal_radius_site"]
            width = 2 * abs(radius_pos[0])
            depth = 2 * abs(radius_pos[1])
        else:
            width = 0.05  # Default
            depth = 0.05  # Default

        return (width, depth, height)

    def get_bounding_box_dimensions(self) -> tuple[float, float, float]:
        """Get the bounding box dimensions for this object.

        Returns:
            Tuple of (width, depth, height) for the bounding box
        """
        return self.bounding_box

    def get_assets(self) -> ET.Element:
        """Get the asset elements that need to be added to the scene XML.

        Returns:
            ET.Element containing all assets for this object
        """
        return self.assets

    @staticmethod
    def get_bounding_box_from_config(
        pos: NDArray[np.float32], object_config: dict[str, str | float]
    ) -> list[float]:
        """Get bounding box for a RoboCasa object given its position and config.

        Args:
            pos: Position of the object as [x, y, z] array
            object_config: Dictionary containing RoboCasa object configuration.
                Should contain "object_type" key for the object type name.

        Returns:
            Bounding box as [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        # A task can state the placed model's footprint directly, as full
        # [x, y, z] extents in metres, when the model's sites do not describe it.
        footprint = object_config.get("footprint")
        if footprint is not None:
            half = [float(v) / 2 for v in footprint]  # type: ignore[union-attr]
            return [
                float(pos[0]) - half[0],
                float(pos[1]) - half[1],
                float(pos[2]) - half[2],
                float(pos[0]) + half[0],
                float(pos[1]) + half[1],
                float(pos[2]) + half[2],
            ]
        # Extract object type from config
        object_type_value = object_config.get("object_type", "")
        object_type = str(object_type_value) if object_type_value else ""
        if not object_type:
            # Try to use type name or fallback to default dimensions
            return [
                float(pos[0]) - 0.05,
                float(pos[1]) - 0.05,
                float(pos[2]) - 0.05,
                float(pos[0]) + 0.05,
                float(pos[1]) + 0.05,
                float(pos[2]) + 0.05,
            ]

        # Look up the registered object class to get model directory
        object_class = REGISTERED_OBJECTS.get(object_type)
        if object_class is None or not hasattr(object_class, "model_dir"):
            # Fallback to default dimensions if object not found
            return [
                float(pos[0]) - 0.05,
                float(pos[1]) - 0.05,
                float(pos[2]) - 0.05,
                float(pos[0]) + 0.05,
                float(pos[1]) + 0.05,
                float(pos[2]) + 0.05,
            ]

        # Load the model and calculate bounding box dimensions
        model_dir = getattr(object_class, "model_dir", Path())
        if not isinstance(model_dir, Path):
            model_dir = Path(str(model_dir))
        model_xml_path = model_dir / "model.xml"

        if not model_xml_path.exists():
            # Fallback to default dimensions
            return [
                float(pos[0]) - 0.05,
                float(pos[1]) - 0.05,
                float(pos[2]) - 0.05,
                float(pos[0]) + 0.05,
                float(pos[1]) + 0.05,
                float(pos[2]) + 0.05,
            ]

        # Parse model to extract bounding box dimensions
        try:
            model_tree = ET.parse(str(model_xml_path))
            model_root = model_tree.getroot()
            worldbody = model_root.find("worldbody")

            if worldbody is None:
                raise ValueError("No worldbody found in model.xml")

            # Extract sites to calculate bounding box
            sites_dict: dict[str, Any] = {}
            for body in worldbody.iter("body"):
                for site in body.findall("site"):
                    site_name = site.attrib.get("name", "")
                    pos_str = site.attrib.get("pos", "0 0 0")
                    site_pos = np.array([float(x) for x in pos_str.split()])
                    sites_dict[site_name] = site_pos

            # Calculate dimensions from sites
            if "bottom_site" in sites_dict and "top_site" in sites_dict:
                height = abs(sites_dict["top_site"][2] - sites_dict["bottom_site"][2])
            else:
                height = 0.05

            if "horizontal_radius_site" in sites_dict:
                radius_pos = sites_dict["horizontal_radius_site"]
                width = 2 * abs(radius_pos[0])
                depth = 2 * abs(radius_pos[1])
            else:
                width = 0.05
                depth = 0.05

            # Calculate half-extents
            half_width = width / 2
            half_depth = depth / 2
            half_height = height / 2

            # Return bounding box at given position
            return [
                float(pos[0]) - half_width,
                float(pos[1]) - half_depth,
                float(pos[2]) - half_height,
                float(pos[0]) + half_width,
                float(pos[1]) + half_depth,
                float(pos[2]) + half_height,
            ]
        except (ValueError, OSError, ET.ParseError) as e:
            # Fallback to default dimensions on error (parsing or file access issues)
            # pylint: disable=unused-variable
            _ = e
            return [
                float(pos[0]) - 0.05,
                float(pos[1]) - 0.05,
                float(pos[2]) - 0.05,
                float(pos[0]) + 0.05,
                float(pos[1]) + 0.05,
                float(pos[2]) + 0.05,
            ]

    def __str__(self) -> str:
        """String representation of the RoboCasa object."""
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"type='{self.object_type_name}')"
        )

    def __repr__(self) -> str:
        """Detailed string representation of the RoboCasa object."""
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"joint_name='{self.joint_name}', type='{self.object_type_name}', "
            f"bounding_box={self.bounding_box})"
        )


class ReplicaObject(RoboCasaObject):
    """Replica object class with support for pos_z and euler options.

    This class extends RoboCasaObject with additional options for positioning and
    rotating the object, which is useful for replica objects like furniture.
    """

    def _create_xml_element(self) -> ET.Element:
        """Create the XML Element for this Replica object.

        Extends the base class to support pos_z and euler options for
        positioning and rotating the object.

        Returns:
            ET.Element representing the object body
        """
        # Find the worldbody in the model
        worldbody = self.model_root.find("worldbody")
        if worldbody is None:
            raise ValueError(
                f"No worldbody found in model.xml for {self.object_type_name}"
            )

        # Find the object body (should be nested inside worldbody)
        # Structure is typically: worldbody -> body -> body[name="object"]
        object_body = None
        for body in worldbody.iter("body"):
            if body.attrib.get("name") == "object":
                object_body = body
                break

        if object_body is None:
            raise ValueError(
                f"No body with name='object' found in model.xml for "
                f"{self.object_type_name}"
            )

        # Create a new body element with our object name
        new_body = ET.Element("body", name=self.name)

        # Add freejoint for position/orientation control
        ET.SubElement(new_body, "freejoint", name=self.joint_name)

        # Get optional pos_z and euler from options (replica-specific)
        pos_z = self.options.get("pos_z", 0.0)
        euler = self.options.get("euler", None)

        # Copy all geom elements from the object body and update mesh references
        for geom in object_body.findall("geom"):
            new_geom = ET.Element("geom", geom.attrib.copy())
            new_geom.text = geom.text
            new_geom.tail = geom.tail

            # Update mesh reference to use the prefixed name
            if "mesh" in geom.attrib:
                original_mesh = geom.attrib["mesh"]
                new_geom.attrib["mesh"] = f"{self.name}_{original_mesh}"

            # Update material reference to use the prefixed name
            if "material" in geom.attrib:
                original_material = geom.attrib["material"]
                new_geom.attrib["material"] = f"{self.name}_{original_material}"

            # Apply pos_z offset if specified
            if pos_z != 0.0:
                # Get existing pos or default to "0 0 0"
                existing_pos = new_geom.attrib.get("pos", "0 0 0")
                pos_parts = [float(p) for p in existing_pos.split()]
                pos_parts[2] += pos_z  # Add to z component
                new_geom.attrib["pos"] = f"{pos_parts[0]} {pos_parts[1]} {pos_parts[2]}"

            # Apply euler rotation if specified (in degrees: [roll, pitch, yaw])
            if euler is not None:
                quat_str = euler_to_quat(euler)
                new_geom.attrib["quat"] = quat_str

            new_body.append(new_geom)

        return new_body


# Dynamically create and register classes for each object directory
def _create_robocasa_object_classes() -> None:
    """Scan the robocasa_objects directory and create classes for each object."""

    if not ROBOCASA_OBJECTS_DIR.exists():
        print(f"Warning: RoboCasa objects directory not found: {ROBOCASA_OBJECTS_DIR}")
        return

    # Iterate through all directories in the robocasa_objects folder
    for object_type_dir in sorted(ROBOCASA_OBJECTS_DIR.iterdir()):
        for object_dir in sorted(object_type_dir.iterdir()):
            if not object_dir.is_dir():
                continue

            # Check if model.xml exists
            model_xml = object_dir / "model.xml"
            if not model_xml.exists():
                continue

            # Extract object type name (directory name)
            object_type_name = object_dir.name

            # Create a class name (convert snake_case to PascalCase)
            # e.g., "apple_0" -> "RobocasaApple0"
            class_name_parts = ["Robocasa"] + [
                part.capitalize() for part in object_type_name.split("_")
            ]
            class_name = "".join(class_name_parts)

            # Create a new class dynamically
            new_class = type(
                class_name,
                (RoboCasaObject,),
                {
                    "object_type_name": object_type_name,
                    "model_dir": object_dir,
                    "__module__": __name__,
                },
            )

            # Register the class with robocasa_ prefix (e.g., "robocasa_apple_0")
            # pylint: disable=too-many-locals
            register_fn = register_object(name=f"robocasa_{object_type_name}")
            registered = register_fn(new_class)  # type: ignore
            new_class = registered

            # Add to module globals so it can be imported
            # pylint: disable=global-variable-undefined
            globals()[class_name] = new_class

    for object_dir in sorted(REPLICA_OBJECTS_DIR.iterdir()):
        if not object_dir.is_dir():
            continue

        # Check if model.xml exists
        model_xml = object_dir / "model.xml"
        if not model_xml.exists():
            continue

        # Extract object type name (directory name)
        object_type_name = object_dir.name

        # Create a class name (convert snake_case to PascalCase)
        # e.g., "frl_apartment_chair_01" -> "RobocasaFrlApartmentChair01"
        # Note: We keep the "Robocasa" prefix for backwards compatibility
        class_name_parts = ["Robocasa"] + [
            part.capitalize() for part in object_type_name.split("_")
        ]
        class_name = "".join(class_name_parts)

        # Create a new class dynamically, inheriting from ReplicaObject
        # to support pos_z and euler options
        new_class = type(
            class_name,
            (ReplicaObject,),
            {
                "object_type_name": object_type_name,
                "model_dir": object_dir,
                "__module__": __name__,
            },
        )

        # Register the class with multiple names for flexibility
        register_object(new_class)  # Registers as lowercase class name

        # Also register with the exact object type name (e.g., "frl_apartment_chair_01")
        # and with "robocasa_" prefix for backwards compatibility
        REGISTERED_OBJECTS[object_type_name] = new_class
        REGISTERED_OBJECTS[f"robocasa_{object_type_name}"] = new_class

        # Add to module globals so it can be imported
        globals()[class_name] = new_class


# Auto-generate classes on module import
_create_robocasa_object_classes()


# Export all dynamically created classes
# pylint: disable=unused-variable
_dynamic_exports = [
    name
    for name in globals()
    if name.startswith("Robocasa") and name not in ("RoboCasaObject", "ReplicaObject")
]
__all__ = [
    "RoboCasaObject",
    "ROBOCASA_OBJECTS_DIR",
] + _dynamic_exports
