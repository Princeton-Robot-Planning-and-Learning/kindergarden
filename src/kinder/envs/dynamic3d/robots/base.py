"""Base robot class for dynamic3d environments."""

import abc
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from relational_structs import Array

from kinder.envs.dynamic3d.mujoco_utils import MjObs, MujocoEnv


class IndexedView:
    """A view that provides indexed access to non-contiguous array elements."""

    def __init__(self, array: Any, indices: list[int]) -> None:
        self.array = array
        self.indices = indices

    def __setitem__(self, key: int | slice, value: Any) -> None:
        self.array[self.indices[key]] = value

    def __getitem__(self, key: int) -> Any:
        return self.array[self.indices[key]]

    def __len__(self) -> int:
        return len(self.indices)


class RobotEnv(MujocoEnv, abc.ABC):
    """Abstract base class for robots in dynamic3d environments."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the robot environment.

        Args:
            *args: Positional arguments passed to MujocoEnv.
            **kwargs: Keyword arguments passed to MujocoEnv.
        """
        super().__init__(*args, **kwargs)

        # Robot name for namespacing XML elements
        self.name: str | None = None

        # Robot state/actuator references (initialized in _setup_robot_references)
        self.qpos: dict[str, NDArray[np.float64]] = {}
        self.qvel: dict[str, NDArray[np.float64]] = {}
        self.ctrl: dict[str, NDArray[np.float64]] = {}

        # Track original names that were renamed
        self._renamed_names: list[str] = []

    def _rename_elements_recursively(self, element: ET.Element) -> None:
        """Recursively rename all 'name' attributes in XML elements.

        Appends the robot name prefix (self.name_) to any existing 'name'
        attribute values. This ensures XML elements are namespaced to avoid
        conflicts when multiple robots are in the same scene.

        Args:
            element: The XML element to process and its children.
        """
        if self.name is not None:
            # Append robot name prefix to the name attribute if it exists
            name_attr = element.get("name")
            if name_attr is not None:
                self._renamed_names.append(name_attr)
                element.set("name", f"{self.name}_{name_attr}")

            # Recursively process all child elements
            for child_elem in element:
                self._rename_elements_recursively(child_elem)

    def _update_attribute_references_recursively(self, element: ET.Element) -> None:
        """Recursively update all attribute values that reference renamed elements.

        Checks all attributes in the element and its children. If any attribute
        value matches a name in _renamed_names, prepends the robot name prefix.

        Args:
            element: The XML element to process and its children.
        """
        if self.name is not None:
            # Check all attributes of the current element
            for key, value in element.attrib.items():
                if value in self._renamed_names:
                    element.set(key, f"{self.name}_{value}")

        # Recursively process all child elements
        for child_elem in element:
            self._update_attribute_references_recursively(child_elem)

    def _insert_robot_into_xml(
        self, xml_string: str, models_dir: str, robot_xml_name: str, assets_dir: str
    ) -> str:
        """Insert the robot model into the provided XML string."""
        # Parse the provided XML string
        input_tree = ET.ElementTree(ET.fromstring(xml_string))
        input_root = input_tree.getroot()

        # Read the scene XML content
        models_dir_path = Path(models_dir)
        robot_path = models_dir_path / robot_xml_name
        assets_dir_path = Path(assets_dir)
        # NOTE: currently manually handling duplicate geoms.xml
        # by creating duplicate asset directories. Probably
        # handle that in code through recursive include.

        with open(robot_path, "r", encoding="utf-8") as f:
            robot_content = f.read()

        # Parse robot XML
        robot_tree = ET.ElementTree(ET.fromstring(robot_content))
        robot_root = robot_tree.getroot()
        if robot_root is None:
            raise ValueError("Missing robot element")

        # Update compiler meshdir to absolute path in robot content
        robot_compiler = robot_root.find("compiler")  # type: ignore[union-attr]
        if robot_compiler is not None:
            robot_compiler.set("meshdir", str(assets_dir_path.resolve()))

        # Helper function to recursively make include file paths absolute
        def make_include_paths_absolute(element: ET.Element) -> None:
            """Recursively process an element and its children to make include file
            paths absolute."""
            if element.tag == "include" and element.get("file") is not None:
                file_path = element.get("file")
                if file_path and not Path(file_path).is_absolute():
                    # Make the file path absolute relative to the models directory
                    absolute_path = models_dir_path / file_path
                    element.set("file", str(absolute_path.resolve()))

            # Recursively process all children
            for child_elem in element:
                make_include_paths_absolute(child_elem)

        # Rename elements in robot XML to namespace them with robot name
        for child in list(robot_root):
            if child.tag in ["worldbody", "tendon", "actuator", "equality"]:
                self._rename_elements_recursively(child)

        # Merge the robot content into the input XML
        # Copy all children from robot root to input root (except mujoco tag itself)
        for child in list(robot_root):
            if child.tag == "worldbody":
                # Merge worldbody content
                input_worldbody = input_root.find(  # type: ignore[union-attr]
                    "worldbody"
                )
                if input_worldbody is not None:
                    for robot_body in list(child):
                        # Process any include tags within robot_body and its children
                        make_include_paths_absolute(robot_body)
                        input_worldbody.append(robot_body)
                else:
                    input_root.append(child)  # type: ignore[union-attr]
            elif child.tag == "default":
                # Merge or append default sections
                input_section = input_root.find(child.tag)  # type: ignore[union-attr]
                if input_section is not None:
                    for sub_child in list(child):
                        input_section.append(sub_child)
                else:
                    input_root.append(child)  # type: ignore[union-attr]
            elif child.tag == "asset":
                # Merge or append asset sections
                input_section = input_root.find(child.tag)  # type: ignore[union-attr]
                if input_section is not None:
                    # Get existing asset names in the scene to avoid duplicates
                    # Track per asset type since MuJoCo allows same name for
                    # different types
                    existing_names: dict[str, set[str]] = {}
                    for existing_asset in input_section:
                        asset_tag = existing_asset.tag
                        asset_name = existing_asset.get("name")
                        if asset_name:
                            if asset_tag not in existing_names:
                                existing_names[asset_tag] = set()
                            existing_names[asset_tag].add(asset_name)

                    for sub_child in list(child):
                        # Skip if this asset name already exists in the scene (same type)
                        asset_tag = sub_child.tag
                        asset_name = sub_child.get("name")
                        if (
                            asset_name
                            and asset_tag in existing_names
                            and asset_name in existing_names[asset_tag]
                        ):
                            continue

                        # Check if the asset element has a "file" attribute
                        # and make it absolute
                        if sub_child.get("file") is not None:
                            file_path = sub_child.get("file")
                            if file_path and not Path(file_path).is_absolute():
                                # Make the file path absolute relative to the
                                # assets directory
                                absolute_path = assets_dir_path / file_path
                                sub_child.set("file", str(absolute_path.resolve()))
                        input_section.append(sub_child)
                else:
                    input_root.append(child)  # type: ignore[union-attr]
            elif child.tag == "compiler":
                # Merge compiler sections - preserve scene's meshdir if it exists
                input_compiler = input_root.find(child.tag)  # type: ignore[union-attr]
                if input_compiler is not None:
                    # Scene already has compiler - merge attributes from robot
                    # but DON'T override meshdir/texturedir if scene has them
                    scene_meshdir = input_compiler.get("meshdir")
                    scene_texturedir = input_compiler.get("texturedir")

                    # Merge all robot compiler attributes
                    for key, value in child.attrib.items():
                        # Skip meshdir/texturedir if scene already defined them
                        if key == "meshdir" and scene_meshdir:
                            continue
                        if key == "texturedir" and scene_texturedir:
                            continue
                        input_compiler.set(key, value)
                else:
                    # No compiler in scene, just append robot's compiler
                    input_root.append(child)  # type: ignore[union-attr]
            elif child.tag == "default":
                # Simply append default sections
                input_root.append(child)  # type: ignore[union-attr]
            else:
                # For other sections (actuator, contact, etc.), update any attribute
                # references to renamed elements since they refer to namespaced elements
                # in the worldbody, then append
                self._update_attribute_references_recursively(child)
                input_root.append(child)  # type: ignore[union-attr]

        if input_root is None:
            raise ValueError("input_root is None, cannot serialize to string")

        # Return the merged XML as string
        return ET.tostring(input_root, encoding="unicode")

    def _update_ctrl(self, action: Array) -> None:
        """Update control values from action array.

        Splits the action across this robot's ctrl parts in insertion order,
        rather than writing the full simulation ctrl array.

        Args:
            action: Action array to apply to robot controls.
        """
        start = 0
        for _, ctrl_part in self.ctrl.items():
            end = start + len(ctrl_part)
            ctrl_part[:] = action[start:end]
            start = end

    @abc.abstractmethod
    def reward(self, obs: MjObs) -> float:
        """Compute the reward from an observation.

        Args:
            obs: The observation to compute reward from.

        Returns:
            The computed reward value.
        """

    @abc.abstractmethod
    def set_robot_base_pos_yaw(self, x: float, y: float, yaw: float) -> None:
        """Set the robot's base position and yaw orientation.

        Args:
            x: X position of the robot base.
            y: Y position of the robot base.
            yaw: Yaw orientation of the robot base.
        """
