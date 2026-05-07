"""Other object classes loaded from flat XML files (e.g. fasteners)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from relational_structs import Object

from kinder.envs.dynamic3d.mujoco_utils import MujocoEnv
from kinder.envs.dynamic3d.object_types import MujocoMovableObjectType
from kinder.envs.dynamic3d.objects.base import MujocoObject, register_object
from kinder.envs.dynamic3d.objects.robocasa_objects import RoboCasaObject

OTHER_OBJECTS_DIR = (
    Path(__file__).parent.parent / "models" / "assets" / "other_objects"
)
FASTENERS_DIR = OTHER_OBJECTS_DIR / "fasteners"


class FastenerObject(RoboCasaObject):
    """Object loaded from a flat XML file in the fasteners directory.

    Unlike standard RoboCasa objects (which live in <category>/<name>/model.xml
    subdirectories and contain a body named "object"), fasteners are stored as
    a single flat .xml per variant whose top-level body is named after the
    fastener and already embeds a freejoint.  This class adapts that format to
    the interface expected by the environment.
    """

    # Subclasses set this to the XML stem, e.g. "screw_simple"
    fastener_name: ClassVar[str] = ""

    def __init__(
        self,
        name: str,
        env: MujocoEnv | None = None,
        options: dict | None = None,
    ) -> None:
        # Bypass RoboCasaObject.__init__, which expects model_dir / "model.xml".
        # Call MujocoObject.__init__ directly for the shared base setup.
        MujocoObject.__init__(self, name, env, options)
        self.symbolic_object = Object(self.name, MujocoMovableObjectType)

        # Set model_dir to FASTENERS_DIR so that _extract_assets() resolves
        # relative paths like "textures/metal.png" and "meshes/*.stl" correctly.
        self.model_dir = FASTENERS_DIR

        xml_path = FASTENERS_DIR / f"{self.fastener_name}.xml"
        if not xml_path.exists():
            raise FileNotFoundError(
                f"Fastener XML not found: {xml_path}"
            )

        self.model_tree = ET.parse(str(xml_path))
        self.model_root = self.model_tree.getroot()

        # Uniform scale factor applied to the mesh asset and all geom dimensions.
        # Useful for making real-world-scale fasteners visible in larger scenes.
        self.scale: float = float(self.options.get("scale", 1.0))

        # Reuse inherited helpers — they only need model_root, model_dir, name.
        self.assets = self._extract_assets()
        self.xml_element = self._create_xml_element()
        self.bounding_box = self._calculate_bounding_box()

        if self.regions is not None:
            self._create_regions()

    def _extract_assets(self) -> ET.Element:
        """Extract assets and apply uniform scale to all mesh elements."""
        container = super()._extract_assets()
        if self.scale != 1.0:
            scale_str = f"{self.scale} {self.scale} {self.scale}"
            for elem in container:
                if elem.tag == "mesh":
                    elem.attrib["scale"] = scale_str
        return container

    def _create_xml_element(self) -> ET.Element:
        """Build the MuJoCo body XML for this fastener.

        Fastener XMLs name their body after the fastener (e.g. "screw_simple")
        and already contain a freejoint child.  We extract only the geom
        elements, rename mesh/material references with the per-instance prefix,
        and inject a fresh freejoint so instance names stay unique.
        """
        worldbody = self.model_root.find("worldbody")
        if worldbody is None:
            raise ValueError(
                f"No <worldbody> in fastener XML for '{self.fastener_name}'"
            )

        # The first <body> child is the fastener body (named after the fastener).
        object_body = next(
            (child for child in worldbody if child.tag == "body"), None
        )
        if object_body is None:
            raise ValueError(
                f"No <body> element found in fastener XML for '{self.fastener_name}'"
            )

        new_body = ET.Element("body", name=self.name)
        ET.SubElement(new_body, "freejoint", name=self.joint_name)

        for geom in object_body.findall("geom"):
            new_geom = ET.Element("geom", geom.attrib.copy())
            if "name" in new_geom.attrib:
                new_geom.attrib["name"] = f"{self.name}_{new_geom.attrib['name']}"
            if "mesh" in geom.attrib:
                new_geom.attrib["mesh"] = f"{self.name}_{geom.attrib['mesh']}"
            if "material" in geom.attrib:
                new_geom.attrib["material"] = (
                    f"{self.name}_{geom.attrib['material']}"
                )
            if self.scale != 1.0:
                if "size" in new_geom.attrib:
                    scaled = [
                        str(float(v) * self.scale)
                        for v in new_geom.attrib["size"].split()
                    ]
                    new_geom.attrib["size"] = " ".join(scaled)
                if "pos" in new_geom.attrib:
                    scaled = [
                        str(float(v) * self.scale)
                        for v in new_geom.attrib["pos"].split()
                    ]
                    new_geom.attrib["pos"] = " ".join(scaled)
            new_body.append(new_geom)

        return new_body

    def _calculate_bounding_box(self) -> tuple[float, float, float]:
        """Compute bounding box from the collision cylinder geom in the XML.

        Returns (width, depth, height) representing the object footprint and
        height above its resting surface.  The cylinder's size attribute encodes
        (radius, half-height), both scaled by self.scale.
        """
        worldbody = self.model_root.find("worldbody")
        if worldbody is not None:
            for geom in worldbody.iter("geom"):
                if geom.get("type") == "cylinder" and "size" in geom.attrib:
                    r, half_h = (float(v) for v in geom.attrib["size"].split())
                    r *= self.scale
                    h = half_h * 2 * self.scale
                    return (r * 2, r * 2, h)
        fallback = 0.05 * self.scale
        return (fallback, fallback, fallback * 2)

    @classmethod
    def get_bounding_box_from_config(
        cls,
        pos: NDArray[np.float32],
        object_config: dict[str, str | float],
    ) -> list[float]:
        """Return a bottom-referenced bounding box for placement sampling.

        z_min is set to pos[2] (object base rests on the surface) so that the
        bottom-corner check inside sample_collision_free_position passes for
        fixture shelf regions whose z range starts exactly at the shelf surface.
        """
        scale = float(object_config.get("scale", 1.0))
        xml_path = FASTENERS_DIR / f"{cls.fastener_name}.xml"
        try:
            root = ET.parse(str(xml_path)).getroot()
            for geom in root.iter("geom"):
                if geom.get("type") == "cylinder" and "size" in geom.attrib:
                    r, half_h = (float(v) for v in geom.attrib["size"].split())
                    r *= scale
                    half_h *= scale
                    # z_min = -half_h so the sampler requires candidate_z >= shelf_z +
                    # half_h, placing the body centre half_h above the surface and the
                    # cylinder bottom exactly at the surface (no clipping).
                    return [
                        float(pos[0]) - r,
                        float(pos[1]) - r,
                        float(pos[2]) - half_h,
                        float(pos[0]) + r,
                        float(pos[1]) + r,
                        float(pos[2]) + half_h,
                    ]
        except (OSError, ET.ParseError):
            pass
        hw = 0.05 * scale
        return [
            float(pos[0]) - hw,
            float(pos[1]) - hw,
            float(pos[2]) - hw,
            float(pos[0]) + hw,
            float(pos[1]) + hw,
            float(pos[2]) + hw,
        ]

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"fastener='{self.fastener_name}')"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"joint_name='{self.joint_name}', fastener='{self.fastener_name}', "
            f"bounding_box={self.bounding_box})"
        )


def _create_fastener_classes() -> None:
    """Scan the fasteners directory and register one class per .xml file."""
    if not FASTENERS_DIR.exists():
        return

    for xml_file in sorted(FASTENERS_DIR.glob("*.xml")):
        fastener_name = xml_file.stem  # e.g. "screw_simple"
        class_name = "Fastener" + "".join(
            part.capitalize() for part in fastener_name.split("_")
        )
        new_class = type(
            class_name,
            (FastenerObject,),
            {"fastener_name": fastener_name, "__module__": __name__},
        )
        register_fn = register_object(name=f"fastener_{fastener_name}")
        # pylint: disable=global-variable-undefined
        globals()[class_name] = register_fn(new_class)


_create_fastener_classes()

_dynamic_exports = [name for name in globals() if name.startswith("Fastener")]
__all__ = [
    "FastenerObject",
    "OTHER_OBJECTS_DIR",
    "FASTENERS_DIR",
] + _dynamic_exports
