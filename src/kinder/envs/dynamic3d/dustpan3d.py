"""Covered lobby dustpan with a solid shell and an entrance ramp."""

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from kinder.envs.dynamic3d.broom3d import Broom3DEnv, ObjectCentricBroom3DEnv
from kinder.envs.dynamic3d.objects.base import register_object
from kinder.envs.dynamic3d.objects.primitive_objects import Cuboid


@register_object
class LobbyDustpan(Cuboid):
    """Separate convex shell pieces leave a genuinely hollow collection pocket."""

    @staticmethod
    def _width(x):
        return 0.25 - 0.02 * x / 0.32

    @staticmethod
    def _prism(x0, x1, z0, z1, thickness):
        return [
            (x, side * LobbyDustpan._width(x), z + dz)
            for x, z in ((x0, z0), (x1, z1))
            for side in (-1, 1)
            for dz in (0, thickness)
        ]

    def _meshes(self):
        meshes = {
            "bottom": self._prism(0.12, 0.32, 0.0, 0.0, 0.008),
            "top": self._prism(0.035, 0.32, 0.07, 0.13, 0.006),
            "ramp": [(0.0, side * self._width(0.0), 0.0) for side in (-1, 1)]
            + [
                (0.12, side * self._width(0.12), z)
                for side in (-1, 1)
                for z in (0.0, 0.03)
            ],
        }
        for name, side in (("left_wall", 1), ("right_wall", -1)):
            meshes[name] = [
                (x, side * (self._width(x) + width), z)
                for x, top in ((0.0, 0.042), (0.12, 0.095), (0.32, 0.13))
                for width in (0.0, 0.006)
                for z in (0.0, top)
            ]
        return meshes

    def get_assets(self):
        return [
            ET.Element(
                "mesh",
                name=f"{self.name}_{name}_mesh",
                vertex=" ".join(str(v) for p in points for v in p),
            )
            for name, points in self._meshes().items()
        ]

    def _create_xml_element(self):
        body = ET.Element("body", name=self.name)
        ET.SubElement(body, "freejoint", name=self.joint_name)
        colors = {"ramp": ".22 .23 .24 1", "top": ".06 .24 .30 1"}
        masses = {
            "bottom": 0.09,
            "top": 0.035,
            "ramp": 0.06,
            "left_wall": 0.035,
            "right_wall": 0.035,
        }
        for name in self._meshes():
            extra = (
                {"friction": ".4 .005 .0001", "priority": "1"} if name == "ramp" else {}
            )
            ET.SubElement(
                body,
                "geom",
                name=f"{self.name}_{name}",
                type="mesh",
                mesh=f"{self.name}_{name}_mesh",
                mass=str(masses[name]),
                rgba=colors.get(name, ".045 .17 .21 1"),
                **extra,
            )
        ET.SubElement(
            body,
            "geom",
            name=f"{self.name}_back_wall",
            type="box",
            pos=".322 0 .066",
            size=".004 .236 .066",
            mass=".025",
            rgba=".045 .17 .21 1",
        )
        # Rounded rim and rubber entry strip match the collision shell.
        for side in (-1, 1):
            ET.SubElement(
                body,
                "geom",
                name=f"{self.name}_rim{side}",
                type="capsule",
                fromto=f"0 {side*(self._width(0)+.003)} .042 .32 {side*(self._width(.32)+.003)} .133",
                size=".004",
                mass=".003",
                rgba=".025 .07 .08 1",
            )
        ET.SubElement(
            body,
            "geom",
            name=f"{self.name}_handle",
            type="capsule",
            fromto=".31 0 .11 .365 0 .96",
            size=".009",
            mass=".045",
            rgba=".6 .63 .65 1",
        )
        for side in (-1, 1):
            ET.SubElement(
                body,
                "geom",
                name=f"{self.name}_grip_side{side}",
                type="capsule",
                fromto=f".365 0 .94 .365 {side*.05} .99",
                size=".012",
                mass=".003",
                rgba=".035 .04 .045 1",
            )
        ET.SubElement(
            body,
            "geom",
            name=f"{self.name}_grip",
            type="capsule",
            fromto=".365 -.05 .99 .365 .05 .99",
            size=".012",
            mass=".008",
            rgba=".035 .04 .045 1",
        )
        # Native soft contacts permit millimetres of penetration, comparable to
        # this thin shell. Keep the collision surface aligned with the visible one.
        for geom in body.findall("geom"):
            geom.set("solref", ".004 1")
            geom.set("solimp", ".99 .999 .001")
            geom.set("priority", "1")
        return body

    def get_bounding_box_dimensions(self):
        return (0.377, 0.52, 1.002)

    @staticmethod
    def get_bounding_box_from_config(pos, object_config):
        return (np.tile(pos, 2) + [0.0, -0.26, 0.0, 0.377, 0.26, 1.002]).tolist()


class ObjectCentricDustpan3DEnv(ObjectCentricBroom3DEnv):
    graspable_tools = {"broom_grasp": "wiper_0", "dustpan_grasp": "dustpan_0"}

    def _grasp_geoms(self, tool):
        if tool == "wiper_0":
            return super()._grasp_geoms(tool)
        model = self._robot_env.sim.model.mj_model
        return {
            i
            for i in range(model.ngeom)
            if (model.geom(i).name or "").startswith(
                ("dustpan_0_handle", "dustpan_0_grip")
            )
        }

    def cube_in_pan(self, name):
        p, q = self._objects_dict["dustpan_0"].get_pose()
        cp, cq = self._objects_dict[name].get_pose()
        inverse = Rotation.from_quat(q[[1, 2, 3, 0]]).inv()
        rotation = Rotation.from_quat(cq[[1, 2, 3, 0]])
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        corners = inverse.apply(rotation.apply(signs * 0.01) + cp - p)
        x, y, z = corners.T
        roof = 0.07 + (0.13 - 0.07) * (x - 0.035) / (0.32 - 0.035)
        return bool(
            np.all(
                (x >= 0.125)
                & (x <= 0.316)
                & (np.abs(y) <= LobbyDustpan._width(x) - 0.004)
                & (z >= 0.006)
                & (z <= roof - 0.002)
            )
        )

    def _check_goals(self):
        return all(self.cube_in_pan(f"cube_{i}") for i in range(5))


class Dustpan3DEnv(Broom3DEnv):
    def _create_object_centric_env(self, *args, **kwargs):
        return ObjectCentricDustpan3DEnv(*args, **kwargs)
