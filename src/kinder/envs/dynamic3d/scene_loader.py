"""Scene loader utilities for loading different types of MuJoCo scene XMLs."""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


class SceneLoader:
    """Factory class for loading different scene types."""

    @staticmethod
    def load_scene(
        scene_config: dict[str, Any],
        model_base_path: Path,
        room_layout: str | None = None,
    ) -> str:
        """Load scene XML based on configuration.

        Args:
            scene_config: Scene configuration dictionary with keys:
                - type: "simple" or "mimiclabs"
                - xml_path: (optional) path to scene XML file
                - lab: (optional, for mimiclabs) lab number (2-8)
            model_base_path: Base path to models directory
            room_layout: Optional task-owned room geometry, shared by backgrounds.

        Returns:
            XML string of the loaded scene
        """
        scene_type = scene_config.get("type", "simple")

        if scene_type == "mimiclabs":
            xml = MimicLabsSceneLoader.load(scene_config)
        elif scene_type == "simple":
            xml = SimpleSceneLoader.load(scene_config, model_base_path)
        else:
            raise ValueError(f"Unknown scene type: {scene_type}")
        if room_layout is None:
            return xml
        return SceneLoader._apply_room_layout(xml, model_base_path / room_layout)

    @staticmethod
    def _apply_room_layout(xml: str, layout_path: Path) -> str:
        """Replace matching visual walls with one explicit physical room layout."""
        root = ET.fromstring(xml)
        room = ET.parse(layout_path).getroot()
        assert room.tag == "body"
        floor_visual_size = room.attrib.pop("floor_visual_size", None)
        wall_names = {geom.get("name") for geom in room.iter("geom")}
        worldbody = root.find("worldbody")
        assert worldbody is not None
        for parent in worldbody.iter():
            for geom in list(parent.findall("geom")):
                if geom.get("name") in wall_names:
                    parent.remove(geom)
                elif floor_visual_size is not None and geom.get("type") == "plane":
                    # Plane X/Y sizes affect rendering, not its infinite collision.
                    size = geom.get("size", "0 0 0.1").split()
                    geom.set("size", f"{floor_visual_size} {size[2]}")
        materials = {mat.get("name") for mat in root.findall("asset/material")}
        for geom in room.iter("geom"):
            if geom.get("material") not in materials:
                geom.attrib.pop("material", None)
                geom.set("rgba", "1 1 1 1")
        worldbody.append(room)
        return ET.tostring(root, encoding="unicode")


class SimpleSceneLoader:
    """Loader for simple ground scenes."""

    @staticmethod
    def load(scene_config: dict[str, Any], model_base_path: Path) -> str:
        """Load a simple ground scene.

        Args:
            scene_config: Scene configuration with optional "xml_path" key
            model_base_path: Base path to models directory

        Returns:
            XML string of the scene
        """
        # Use provided xml_path or default to ground_scene.xml
        xml_filename = scene_config.get("xml_path", "ground_scene.xml")

        # If it's not an absolute path, make it relative to model_base_path
        if not Path(xml_filename).is_absolute():
            xml_path = model_base_path / xml_filename
        else:
            xml_path = Path(xml_filename)

        with open(xml_path, "r", encoding="utf-8") as f:
            return f.read()


class MimicLabsSceneLoader:
    """Loader for MimicLabs realistic background scenes."""

    @staticmethod
    def load(scene_config: dict[str, Any]) -> str:
        """Load a MimicLabs scene with proper path resolution.

        Args:
            scene_config: Scene configuration with keys:
                - lab: lab number (2-8), or
                - xml_path: relative path to scene XML
                - position: (optional) [x, y, z] position offset for the scene

        Returns:
            XML string with absolute paths for assets
        """
        # Get position offset (default to [0, 0, 0])
        position = scene_config.get("position", [0, 0, 0])
        # Resolve mimiclabs assets directory
        # Path(__file__) is at: kinder/src/kinder/envs/dynamic3d/scene_loader.py
        # MimicLabs scenes are stored in models/assets/mimiclabs_scenes/
        # (similar to how RoboCasa objects are in models/assets/robocasa_objects/)
        mimiclabs_scenes_dir = (
            Path(__file__).parent / "models" / "assets" / "mimiclabs_scenes"
        )

        # Check if mimiclabs_scenes directory exists
        if not mimiclabs_scenes_dir.exists():
            raise FileNotFoundError(
                f"MimicLabs scenes directory not found at: {mimiclabs_scenes_dir}\n"
                f"Please run: python scripts/download_mimiclabs_assets.py"
            )

        # Determine scene XML path
        if "xml_path" in scene_config:
            scene_xml_path = mimiclabs_scenes_dir / scene_config["xml_path"]
        elif "lab" in scene_config:
            lab_num = scene_config["lab"]
            scene_xml_path = mimiclabs_scenes_dir / f"lab{lab_num}.xml"
        else:
            raise ValueError(
                "MimicLabs scene config must specify either 'lab' or 'xml_path'"
            )

        if not scene_xml_path.exists():
            raise FileNotFoundError(
                f"MimicLabs scene not found at: {scene_xml_path}\n"
                f"Please run: python scripts/download_mimiclabs_assets.py"
            )

        # Load scene XML
        with open(scene_xml_path, "r", encoding="utf-8") as f:
            xml_string = f.read()

        # Parse and update asset paths to be absolute
        tree = ET.fromstring(xml_string)

        # Set meshdir and texturedir to absolute paths
        # NOTE: mimiclabs XML files already include "meshes/" and "textures/"
        # in file paths, so meshdir/texturedir should point to
        # mimiclabs_scenes directory
        meshdir = mimiclabs_scenes_dir
        texturedir = mimiclabs_scenes_dir

        # Update or create compiler section
        compiler = tree.find("compiler")
        if compiler is None:
            # Create compiler section and insert it at the beginning
            compiler = ET.Element("compiler")
            tree.insert(0, compiler)

        # Set absolute paths
        compiler.set("meshdir", str(meshdir.resolve()))
        if texturedir.exists():
            compiler.set("texturedir", str(texturedir.resolve()))

        # De-duplicate assets within the scene XML
        # Some mimiclabs scenes have duplicate material/texture/mesh definitions
        # NOTE: Deduplicate per asset type (tag), not globally, since MuJoCo allows
        # same name for different types (e.g., texture and material both named "X")
        asset_section = tree.find("asset")
        if asset_section is not None:
            seen_names: dict[str, set[str]] = {}  # tag -> set of names
            assets_to_remove = []
            for asset_elem in asset_section:
                asset_tag = asset_elem.tag
                asset_name = asset_elem.get("name")
                if asset_name:
                    if asset_tag not in seen_names:
                        seen_names[asset_tag] = set()
                    if asset_name in seen_names[asset_tag]:
                        # Mark for removal (duplicate within same type)
                        assets_to_remove.append(asset_elem)
                    else:
                        seen_names[asset_tag].add(asset_name)

            # Remove duplicate assets
            for asset_elem in assets_to_remove:
                asset_section.remove(asset_elem)

        # Apply position offset to the scene
        MimicLabsSceneLoader._apply_position_offset(tree, position)

        # Enable wall collisions if requested
        wall_collision = scene_config.get("wall_collision", False)
        if wall_collision:
            MimicLabsSceneLoader._enable_wall_collisions(tree)

        return ET.tostring(tree, encoding="unicode")

    @staticmethod
    def _apply_position_offset(tree: ET.Element, position: list[float]) -> None:
        """Apply position offset to scene by updating the scene body position.

        MimicLabs XMLs have all scene elements wrapped in a <body name="scene">
        element. This method updates that body's position to offset the entire
        scene while keeping task-specific objects (fixtures, robots) at their
        original positions.

        Args:
            tree: Root XML element
            position: [x, y, z] position offset
        """
        if position == [0, 0, 0]:
            return  # No offset needed

        worldbody = tree.find("worldbody")
        if worldbody is None:
            return

        # Find the scene body and update its position
        for body in worldbody.findall("body"):
            if body.get("name") == "scene":
                body.set("pos", f"{position[0]} {position[1]} {position[2]}")
                return

        # If no scene body found, log a warning (shouldn't happen with proper XMLs)
        logging.warning(
            "No <body name='scene'> found in MimicLabs XML. "
            "Position offset will not be applied."
        )

    @staticmethod
    def _enable_wall_collisions(tree: ET.Element) -> None:
        """Enable collisions for wall geoms in the scene.

        MimicLabs scenes have wall geoms with conaffinity="0" contype="0"
        (collisions disabled). This method enables collisions for these walls
        by setting conaffinity="1" contype="1".

        Args:
            tree: Root XML element
        """
        worldbody = tree.find("worldbody")
        if worldbody is None:
            return

        # Find the scene body
        scene_body = None
        for body in worldbody.findall("body"):
            if body.get("name") == "scene":
                scene_body = body
                break

        if scene_body is None:
            logging.warning(
                "No <body name='scene'> found in MimicLabs XML. "
                "Wall collisions cannot be enabled."
            )
            return

        # Find all wall geoms and enable collisions
        # Wall geoms typically have names containing "wall"
        for geom in scene_body.findall("geom"):
            geom_name = geom.get("name", "")
            if "wall" in geom_name.lower():
                geom.set("conaffinity", "1")
                geom.set("contype", "1")
