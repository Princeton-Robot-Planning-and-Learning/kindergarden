"""Render of the Sprout environments with the Filament renderer of MuJoCo 4.

MuJoCo 4 has two renderers. The classic renderer (OpenGL) is the one that the other
dynamic3d environments use. Filament is a physically-based renderer: it has soft
shadows, ambient occlusion, and reflections, and it shows the declared colors of the
scene (the classic renderer overexposes light surfaces).

The scene of a Kinder task is made for the classic renderer: its lights cast no shadows
and its materials have no physical properties. ``make_render_xml`` makes a copy of the
scene XML for the render only, with physical lights and materials. The physics model
does not change. ``SproutFilamentRenderer`` compiles this copy, and renders it with the
positions of the physics simulation.

Filament needs MuJoCo 4 (the ``mujoco`` package, not only ``mujoco-core``). Use
``filament_available()`` to check. No display is necessary.
"""

# Pylint does not see the members of the mujoco module, and Filament is only in
# MuJoCo 4.
# pylint: disable=no-member,import-error,no-name-in-module

import os
import weakref
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

# The key light: a spot light above the task area, with soft shadows. Its position is
# relative to the point that it lights. Intensities are in candela.
KEY_LIGHT_OFFSET = (2.2, -2.4, 4.0)
KEY_LIGHT_INTENSITY = 300000.0
KEY_LIGHT_BULB_RADIUS = 0.12  # a larger radius gives softer shadows
# The lights of the Kinder scene become fill lights (no shadows). This is the sum of
# their intensities.
FILL_LIGHT_INTENSITY = 30000.0
# The environment map gives the ambient light.
ENVIRONMENT_LIGHT_INTENSITY = 4500.0

# Roughness (0: mirror, 1: matte) and reflectance of the materials, by a word in the
# material name. The values are selected by eye.
MATERIAL_ROUGHNESS = {"floor": 0.35, "wall": 0.95, "table": 0.55}
DEFAULT_ROUGHNESS = 0.6
FLOOR_REFLECTANCE = 0.12

# Names in the Sprout model: its material, its color texture, and its normal map (a
# texture for small surface relief, which the classic renderer cannot use).
SPROUT_MATERIAL = "visualgeom"
SPROUT_COLOR_TEXTURE = "robot_diffuse"
SPROUT_NORMAL_TEXTURE = "robot_normal"

# All renderers of the process share one Filament context (each new context prints
# some lines). In MuJoCo 4.0, a context renders only one scene: a
# second scene in the same context gives an error. So one renderer at a time has its
# scene in the context (the "scene owner"). A renderer that renders becomes the owner,
# which takes about 0.1 s when the owner changes.
_SHARED_CONTEXT: Any = None
_SCENE_OWNER: "weakref.ReferenceType[SproutFilamentRenderer] | None" = None


def filament_available() -> bool:
    """Check if the Filament renderer can be imported (MuJoCo 4 or newer)."""
    try:
        # pylint: disable=import-outside-toplevel,unused-import
        from mujoco import mjrf  # noqa: F401
        from mujoco.rendering.filament import model_scene  # noqa: F401
    except ImportError:
        return False
    return True


def _get_shared_context() -> Any:
    global _SHARED_CONTEXT  # pylint: disable=global-statement
    if _SHARED_CONTEXT is None:
        from mujoco import mjrf  # pylint: disable=import-outside-toplevel

        # The render is offscreen. But with a display variable, Filament connects to
        # that display when it makes the context. Through SSH, this is slow, and it
        # blocks if the display does not answer. So hide the variables for this call.
        hidden = {
            name: os.environ.pop(name)
            for name in ("DISPLAY", "WAYLAND_DISPLAY")
            if name in os.environ
        }
        try:
            _SHARED_CONTEXT = mjrf.Context(mjrf.ContextConfig())
        finally:
            os.environ.update(hidden)
    return _SHARED_CONTEXT


def _get_or_add(parent: ET.Element, tag: str) -> ET.Element:
    child = parent.find(tag)
    return child if child is not None else ET.SubElement(parent, tag)


def _vector(values: Any) -> str:
    return " ".join(f"{float(value):.4f}" for value in values)


def make_render_xml(
    xml_string: str, light_target: tuple[float, float, float] | NDArray[np.float64]
) -> str:
    """Make the render copy of a scene XML: physical lights and materials.

    Bodies, joints, and geoms do not change, so the generalized positions (qpos) of
    the physics model are valid for the model of this XML.

    Args:
        xml_string: The scene XML of a Kinder task with the Sprout.
        light_target: The point on the floor that the key light lights (the task
            area, for example the position of the robot).
    """
    root = ET.fromstring(xml_string)
    asset = _get_or_add(root, "asset")
    worldbody = _get_or_add(root, "worldbody")

    # Lights. A light with an intensity selects the physical light mode of Filament:
    # the headlight goes off, and the lights have a distance falloff.
    fill_lights = list(root.iter("light"))
    for light in fill_lights:
        light.set("castshadow", "false")
        light.set("intensity", str(FILL_LIGHT_INTENSITY / len(fill_lights)))
        light.set("cutoff", "70")
        light.set("range", "40")
    target = np.asarray(light_target, dtype=float)
    key_position = target + np.array(KEY_LIGHT_OFFSET)
    key_direction = target + np.array([0.0, 0.0, 0.6]) - key_position
    ET.SubElement(
        worldbody,
        "light",
        name="render_key_light",
        type="spot",
        pos=_vector(key_position),
        dir=_vector(key_direction),
        cutoff="60",
        range="40",
        intensity=str(KEY_LIGHT_INTENSITY),
        bulbradius=str(KEY_LIGHT_BULB_RADIUS),
        castshadow="true",
    )
    environment_map = Path(mujoco.__file__).parent / "studio" / "assets" / "ibl.ktx"
    ET.SubElement(
        asset,
        "texture",
        name="render_environment_map",
        type="2d",
        file=str(environment_map),
    )
    ET.SubElement(
        worldbody,
        "light",
        name="render_environment_light",
        type="image",
        texture="render_environment_map",
        intensity=str(ENVIRONMENT_LIGHT_INTENSITY),
        castshadow="false",
    )

    # Materials. A roughness selects the physically-based shading of Filament.
    texture_names = {texture.get("name") for texture in root.iter("texture")}
    for material in root.iter("material"):
        name = material.get("name", "")
        roughness = DEFAULT_ROUGHNESS
        for word, value in MATERIAL_ROUGHNESS.items():
            if word in name.lower():
                roughness = value
        material.set("roughness", str(roughness))
        material.set("metallic", "0")
        if "floor" in name.lower():
            material.set("reflectance", str(FLOOR_REFLECTANCE))
        if name == SPROUT_MATERIAL and SPROUT_NORMAL_TEXTURE in texture_names:
            # A material with two textures needs "layer" elements.
            material.attrib.pop("texture", None)
            ET.SubElement(material, "layer", texture=SPROUT_COLOR_TEXTURE, role="rgb")
            ET.SubElement(
                material, "layer", texture=SPROUT_NORMAL_TEXTURE, role="normal"
            )
            material.set("roughness", "0.45")

    # The region markers are sites with alpha 0. The classic renderer does not show
    # them, but in Filament they cast shadows. Remove them.
    for parent in list(root.iter()):
        for site in parent.findall("site"):
            rgba = site.get("rgba", "").split()
            if len(rgba) == 4 and float(rgba[3]) == 0.0:
                parent.remove(site)

    # Render options of Filament (see "Rendering options" in the MuJoCo manual).
    _get_or_add(_get_or_add(root, "visual"), "quality").set("shadowsize", "4096")
    custom = _get_or_add(root, "custom")
    ET.SubElement(custom, "text", name="filament.shadows.type", data="pcss")
    ET.SubElement(custom, "numeric", name="filament.ao.bent_normals", data="1")
    return ET.tostring(root, encoding="unicode")


class SproutFilamentRenderer:
    """Renders one Sprout scene with Filament.

    Make a new renderer when the scene XML changes (a reset makes a new scene).

    Example:
        renderer = SproutFilamentRenderer(scene_xml, light_target=(0.0, 0.0, 0.0))
        image = renderer.render(mj_data, "task_view", width=640, height=480)
    """

    def __init__(
        self,
        xml_string: str,
        light_target: tuple[float, float, float] | NDArray[np.float64],
    ) -> None:
        self._model = mujoco.MjModel.from_xml_string(
            make_render_xml(xml_string, light_target)
        )
        self._data = mujoco.MjData(self._model)
        # The Filament objects of the scene. They exist while this renderer is the
        # scene owner (see _SCENE_OWNER).
        self._renderer: Any = None
        self._scene: Any = None
        self._pixels: NDArray[np.uint8] = np.empty((0, 0, 3), dtype=np.uint8)

    def _become_scene_owner(self) -> None:
        """Put the scene of this renderer into the shared context."""
        # pylint: disable=import-outside-toplevel,global-statement
        from mujoco.rendering.filament import model_scene, renderer

        global _SCENE_OWNER
        owner = _SCENE_OWNER() if _SCENE_OWNER is not None else None
        if owner is self:
            return
        if owner is not None:
            owner.close()
        context = _get_shared_context()
        self._renderer = renderer.Renderer(context)
        self._scene = model_scene.ModelScene(
            context, self._model, self._renderer.scene("scene")
        )
        self._renderer.scene("scene").set_msaa(True)  # smooth edges
        _SCENE_OWNER = weakref.ref(self)

    def close(self) -> None:
        """Remove the scene of this renderer from the shared context.

        The renderer stays valid: the next render() puts the scene back.
        """
        self._scene = None
        self._renderer = None
        self._pixels = np.empty((0, 0, 3), dtype=np.uint8)

    def render(
        self,
        mj_data: mujoco.MjData,
        camera_name: str | None,
        width: int,
        height: int,
    ) -> NDArray[np.uint8]:
        """Render the scene with the positions of ``mj_data``.

        Args:
            mj_data: The data of the physics simulation.
            camera_name: A camera of the model. None gives the default free camera.
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            The image, with shape (height, width, 3).
        """
        # pylint: disable=import-outside-toplevel
        from mujoco.rendering.filament import renderer

        assert mj_data.qpos.shape == self._data.qpos.shape, "Not the same scene"
        self._data.qpos[:] = mj_data.qpos
        self._data.mocap_pos[:] = mj_data.mocap_pos
        self._data.mocap_quat[:] = mj_data.mocap_quat
        mujoco.mj_kinematics(self._model, self._data)
        mujoco.mj_camlight(self._model, self._data)

        self._become_scene_owner()
        self._scene.update(self._data)
        if self._pixels.shape != (height, width, 3):
            self._pixels = np.empty((height, width, 3), dtype=np.uint8)
            self._renderer.target("image", self._pixels)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self._model, camera)
        if camera_name is not None:
            camera.type = int(mujoco.mjtCamera.mjCAMERA_FIXED)
            camera.fixedcamid = self._model.camera(camera_name).id
        view = renderer.View(
            scene="scene",
            target="image",
            camera=mujoco.mjv_camera2GLCamera(self._model, self._data, camera),
        )
        self._renderer.render([view])
        return self._pixels.copy()
