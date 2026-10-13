"""Floor sweeping with tool-only debris contact and a handled retaining pan."""

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv, TidyBot3DEnv
from kinder.envs.dynamic3d.objects.base import register_object
from kinder.envs.dynamic3d.objects.primitive_objects import Cuboid


@register_object
class Dustpan(Cuboid):
    """Rigid ramp, collection pocket and accessible horizontal handle."""

    def _create_xml_element(self):
        body = ET.Element("body", name=self.name)
        ET.SubElement(body, "freejoint", name=self.joint_name)

        def box(name, size, pos, mass, rgba="0.15 0.35 0.7 1", **extra):
            return ET.SubElement(
                body,
                "geom",
                name=f"{self.name}_{name}",
                type="box",
                size=" ".join(map(str, size)),
                pos=" ".join(map(str, pos)),
                mass=str(mass * 0.4),
                rgba=rgba,
                **{"friction": "0.8 0.005 0.0001", **extra},
            )

        box("floor", [0.15, 0.23, 0.006], [0.15, 0.0, 0.006], 0.55)
        box("back", [0.006, 0.23, 0.055], [0.306, 0.0, 0.055], 0.12)
        for side in (-1, 1):
            box(f"side{side}", [0.156, 0.006, 0.055], [0.15, side * 0.236, 0.055], 0.12)
        # A shallow entrance ramp ends above the pocket floor. Its rear face
        # retains debris until the whole pan is tipped.
        ET.SubElement(
            body,
            "geom",
            name=f"{self.name}_ramp",
            type="mesh",
            mesh=f"{self.name}_ramp_mesh",
            mass=".072",
            rgba="0.15 0.35 0.7 1",
            friction=".4 .005 .0001",
        )
        box("retaining_ridge", [0.005, 0.23, 0.03275], [0.0, 0.0, 0.03275], 0.08)
        for side in (-1, 1):
            box(
                f"handle_post{side}",
                [0.009, 0.009, 0.225],
                [0.25, side * 0.07, 0.285],
                0.025,
                rgba="0.75 0.75 0.75 1",
            )
        box(
            "handle",
            [0.012, 0.082, 0.012],
            [0.25, 0.0, 0.51],
            0.035,
            rgba="0.95 0.8 0.15 1",
            condim="6",
            priority="1",
            friction="1.5 .03 .001",
        )
        return body

    def get_assets(self):
        # A solid wedge reaches the floor without the vertical front face of a
        # thick rotated box or the tunnelling risk of a paper-thin ramp.
        vertices = [(-0.30, y, 0.0) for y in (-0.23, 0.23)]
        vertices += [(0.0, y, z) for y in (-0.23, 0.23) for z in (0.0, 0.065)]
        return [
            ET.Element(
                "mesh",
                name=f"{self.name}_ramp_mesh",
                vertex=" ".join(str(v) for xyz in vertices for v in xyz),
            )
        ]

    def get_bounding_box_dimensions(self):
        return (0.618, 0.484, 0.522)

    @staticmethod
    def get_bounding_box_from_config(pos, object_config):
        return (np.tile(pos, 2) + np.array([-0.306, -0.242, 0.0, 0.312, 0.242, 0.522])).tolist()


@register_object
class Broom(Cuboid):
    """A rigid floor squeegee with a tall, robot-graspable handle."""

    def _create_xml_element(self):
        body = ET.Element("body", name=self.name)
        ET.SubElement(body, "freejoint", name=self.joint_name)
        ET.SubElement(
            body,
            "geom",
            name=f"{self.name}_head",
            type="box",
            size=".025 .19 .012",
            pos="0 0 .012",
            mass=".25",
            rgba=".15 .15 .15 1",
            friction=".1 .005 .0001",
            priority="1",
        )
        ET.SubElement(
            body,
            "geom",
            name=f"{self.name}_shaft",
            type="box",
            size=".01 .01 .20",
            pos="-.10 0 .197205",
            quat=".965925826 0 -.258819045 0",
            mass=".015",
            rgba=".85 .5 .15 1",
            friction="1.5 .03 .001",
            condim="6",
            priority="1",
        )
        ET.SubElement(
            body,
            "geom",
            name=f"{self.name}_handle",
            type="box",
            size=".01 .01 .070",
            pos="-.20 0 .440410",
            mass=".005",
            rgba=".85 .5 .15 1",
            friction="1.5 .03 .001",
            condim="6",
            priority="1",
        )
        return body

    def get_bounding_box_dimensions(self):
        return (0.235, 0.38, 0.511)

    @staticmethod
    def get_bounding_box_from_config(pos, object_config):
        return (np.tile(pos, 2) + np.array([-0.21, -0.19, 0.0, 0.025, 0.19, 0.511])).tolist()


class ObjectCentricSweepIntoDustpan3DEnv(ObjectCentricTidyBot3DEnv):
    @property
    def debris_names(self):
        return tuple(sorted(n for n in self._objects_dict if n.startswith("debris_")))

    def _initialize_robot_pose(self):
        super()._initialize_robot_pose()
        # Robot XML is composed after the scene XML; apply masks to the compiled
        # model before initial settling so every robot descendant is covered.
        model = self._robot_env.sim.model.mj_model
        for i in range(model.ngeom):
            if not (model.geom_contype[i] or model.geom_conaffinity[i]):
                continue
            body = int(model.geom_bodyid[i])
            category = "floor"
            while body:
                name = model.body(body).name or ""
                if name.startswith("robot_") or name == "robot":
                    category = "robot"
                    break
                if name.startswith("debris_"):
                    category = "debris"
                    break
                if name in ("broom_0", "dustpan_0"):
                    category = "tool"
                    break
                body = int(model.body_parentid[body])
            if category == "robot":
                # Preserve native robot self-collision filtering exactly.
                continue
            if category == "debris":
                model.geom_contype[i], model.geom_conaffinity[i] = 4, 28
                # The default contact softness is comparable to these pieces'
                # thickness; use a small contact margin for solid ramp contact.
                model.geom_priority[i] = 2
                model.geom_solref[i] = [0.004, 1.0]
                model.geom_solimp[i, :3] = [0.99, 0.999, 0.0001]
                model.geom_friction[i] = [0.3, 0.005, 0.0001]
            else:
                bit = 8 if category == "tool" else 16
                model.geom_contype[i] |= bit
                model.geom_conaffinity[i] |= 28

    def collision_groups(self):
        model = self._robot_env.sim.model.mj_model
        groups = {n: [] for n in ("robot", "debris", "broom", "dustpan", "floor")}
        for i in range(model.ngeom):
            if not (model.geom_contype[i] or model.geom_conaffinity[i]):
                continue
            body = int(model.geom_bodyid[i])
            category = "floor"
            while body:
                name = model.body(body).name or ""
                if name.startswith("robot_") or name == "robot":
                    category = "robot"
                    break
                if name.startswith("debris_"):
                    category = "debris"
                    break
                if name in ("broom_0", "dustpan_0"):
                    category = name.split("_")[0]
                    break
                body = int(model.body_parentid[body])
            groups[category].append(i)
        return groups

    def debris_in_pan(self, name):
        # Check every cube corner in the pan's local frame, not a world-aligned
        # approximation that would score pieces beneath a tilted or moved pan.
        pan = self._objects_dict["dustpan_0"]
        p, q = pan.get_pose()
        inv = Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]]).inv()
        obj = self._objects_dict[name]
        op, oq = obj.get_pose()
        rot = Rotation.from_quat(np.asarray(oq)[[1, 2, 3, 0]])
        half = np.asarray(obj.get_bounding_box_dimensions()) / 2
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        corners = inv.apply(rot.apply(signs * half) + op - p)
        return bool(
            np.all(corners >= [0.008, -0.229, 0.007]) and np.all(corners <= [0.299, 0.229, 0.109])
        )

    def _check_goals(self):
        return bool(self.debris_names) and all(self.debris_in_pan(n) for n in self.debris_names)


class SweepIntoDustpan3DEnv(TidyBot3DEnv):
    def _create_object_centric_env(self, *args, **kwargs):
        return ObjectCentricSweepIntoDustpan3DEnv(*args, **kwargs)
