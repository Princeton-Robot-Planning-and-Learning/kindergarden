"""Floor sweeping into a shallow pit, beside a shovel the arm cannot lift."""

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from kinder.envs.dynamic3d.broom3d import Broom3DEnv, ObjectCentricBroom3DEnv
from kinder.envs.dynamic3d.objects.base import register_object
from kinder.envs.dynamic3d.objects.primitive_objects import Cuboid

# The pit opening in world coordinates, and its depth below the floor at z = 0.
# A cube is in the hole when all its corners lie inside this footprint below z = 0.
PIT_X = (0.80, 1.25)
PIT_Y = (-0.20, 0.20)
PIT_DEPTH = 0.08

# The slabs reach as far as the ground plane they replace renders.
FLOOR_EXTENT = 10.0

# The real TidyBot++ base's caster steer axes, measured from the vehicle centre in
# the base frame (x forward), from its base controller's constants. Each wheel's
# ground contact trails its steer axis by about 1.4 cm, so the axis stands in for
# the wheel. The track is narrower than the pit, so a base cannot straddle it.
WHEEL_OFFSETS = (
    (0.190150, 0.170150),
    (0.190150, -0.170150),
    (-0.190150, 0.170150),
    (-0.190150, -0.170150),
)

# White so the opening reads against the lab's wood floor. The lab plane already sits
# at the pit bottom, so the cover rides half a millimetre above it to avoid z-fighting.
PIT_FLOOR_RGBA = "1 1 1 1"
PIT_FLOOR_LIFT = 0.0005


@register_object
class Shovel(Cuboid):
    """Flat on the floor, blade at +x, handle along -x; far beyond the arm payload.

    The body origin is the blade centre at floor level. The handle and grip carry the
    same geom name suffixes as the lobby dustpan's so a tool grasp can address them
    later; nothing here grasps.
    """

    BLADE_HALF = (0.125, 0.10, 0.004)
    HANDLE_LENGTH = 1.1
    HANDLE_RADIUS = 0.015
    GRIP_HALF_WIDTH = 0.05
    # Kilograms. The TidyBot's Gen3 arm is rated for 4 kg; ten times that keeps a
    # closed grasp from ever lifting the tool while a 40 kg free body stays tame
    # under MuJoCo's default contact parameters.
    MASSES = {"blade": 18.0, "socket": 2.0, "handle": 16.0, "grip": 4.0}

    def _create_xml_element(self):
        body = ET.Element("body", name=self.name)
        ET.SubElement(body, "freejoint", name=self.joint_name)
        bx, by, bz = self.BLADE_HALF
        ET.SubElement(
            body,
            "geom",
            name=f"{self.name}_blade",
            type="box",
            pos=f"0 0 {bz}",
            size=f"{bx} {by} {bz}",
            mass=str(self.MASSES["blade"]),
            rgba=".38 .39 .42 1",
        )
        handle_z = self.HANDLE_RADIUS
        socket_x = -bx - 0.03
        ET.SubElement(
            body,
            "geom",
            name=f"{self.name}_socket",
            type="box",
            pos=f"{socket_x} 0 {handle_z}",
            size=f"0.03 0.025 {handle_z}",
            mass=str(self.MASSES["socket"]),
            rgba=".30 .31 .33 1",
        )
        handle_end = -bx - self.HANDLE_LENGTH
        ET.SubElement(
            body,
            "geom",
            name=f"{self.name}_handle",
            type="capsule",
            fromto=f"{socket_x - 0.03} 0 {handle_z} {handle_end} 0 {handle_z}",
            size=str(self.HANDLE_RADIUS),
            mass=str(self.MASSES["handle"]),
            rgba=".55 .36 .16 1",
        )
        for side in (-1, 1):
            ET.SubElement(
                body,
                "geom",
                name=f"{self.name}_grip_side{side}",
                type="capsule",
                fromto=(
                    f"{handle_end} 0 {handle_z} "
                    f"{handle_end - 0.04} {side * self.GRIP_HALF_WIDTH} {handle_z}"
                ),
                size="0.012",
                mass="0.5",
                rgba=".035 .04 .045 1",
            )
        ET.SubElement(
            body,
            "geom",
            name=f"{self.name}_grip",
            type="capsule",
            fromto=(
                f"{handle_end - 0.04} {-self.GRIP_HALF_WIDTH} {handle_z} "
                f"{handle_end - 0.04} {self.GRIP_HALF_WIDTH} {handle_z}"
            ),
            size="0.012",
            mass=str(self.MASSES["grip"] - 1.0),
            rgba=".035 .04 .045 1",
        )
        return body

    @classmethod
    def _extents(cls):
        bx, by, _ = cls.BLADE_HALF
        return [-bx - cls.HANDLE_LENGTH - 0.052, -by, 0.0, bx, by, 0.03]

    def get_bounding_box_dimensions(self):
        lo, hi = self._extents()[:3], self._extents()[3:]
        return tuple(float(b - a) for a, b in zip(lo, hi))

    @staticmethod
    def get_bounding_box_from_config(pos, object_config):
        return (np.tile(pos, 2) + Shovel._extents()).tolist()


class ObjectCentricSweepIntoHole3DEnv(ObjectCentricBroom3DEnv):
    """Dustpan3D's layout with the dustpan replaced by a pit cut into the floor.

    A MuJoCo plane cannot have a hole, so the floor becomes four coplanar slabs whose
    inner faces are the pit's vertical walls, over a lowered plane. The base is a slide-
    joint body that never touches the floor, so slab seams cannot catch it; the only
    seams are the pit edges anyway.
    """

    def _create_scene_xml(self):
        root = ET.fromstring(super()._create_scene_xml())
        worldbody = root.find("worldbody")
        assert worldbody is not None
        floor = next(g for g in root.iter("geom") if g.get("name") == "floor")
        scene = self.task_config["_active_scene"]
        x0, x1 = PIT_X
        y0, y1 = PIT_Y
        pit_bottom = None
        if scene["type"] == "simple":
            # The plane drops below a darker box so the opening reads against a
            # flat white floor; the box is also what a dropped cube rests on.
            floor.set("pos", f"0 0 {-PIT_DEPTH - 0.02}")
            slab_look = {"material": "groundplane"}
            pit_bottom = ET.Element(
                "geom",
                name="pit_floor",
                type="box",
                pos=f"{(x0 + x1) / 2} {(y0 + y1) / 2} {-PIT_DEPTH - 0.01}",
                size=f"{(x1 - x0) / 2} {(y1 - y0) / 2} 0.01",
                rgba=".22 .22 .24 1",
            )
        else:
            # A lab floor cannot be cut, so the task lowers the whole lab by the
            # pit depth through its scene position: the lab's own floor plane is
            # then the pit bottom and the slabs are the walkable floor at z = 0.
            offset = scene.get("position", [0, 0, 0])
            if not np.isclose(offset[2], -PIT_DEPTH):
                raise ValueError(
                    f"the {scene['type']} scene must be lowered by exactly "
                    f"{PIT_DEPTH} m for the pit; its z offset is {offset[2]}"
                )
            material = floor.get("material")
            slab_look = {"material": material} if material else {}
            pit_bottom = ET.Element(
                "geom",
                name="pit_floor",
                type="box",
                pos=f"{(x0 + x1) / 2} {(y0 + y1) / 2} {-PIT_DEPTH + PIT_FLOOR_LIFT / 2}",
                size=f"{(x1 - x0) / 2} {(y1 - y0) / 2} {PIT_FLOOR_LIFT / 2}",
                rgba=PIT_FLOOR_RGBA,
            )
        slabs = self._slab_extents(root=root, material=slab_look.get("material"))
        for name, (xa, xb, ya, yb) in slabs.items():
            worldbody.append(
                ET.Element(
                    "geom",
                    name=f"floor_slab_{name}",
                    type="box",
                    pos=f"{(xa + xb) / 2} {(ya + yb) / 2} {-PIT_DEPTH / 2}",
                    size=f"{(xb - xa) / 2} {(yb - ya) / 2} {PIT_DEPTH / 2}",
                    **slab_look,
                )
            )
        if pit_bottom is not None:
            worldbody.append(pit_bottom)
        return ET.tostring(root, encoding="unicode")

    @staticmethod
    def _slab_extents(*, root, material):
        """Four slabs in a pinwheel around the opening, each centred on the floor
        texture's grid.

        MuJoCo maps a box's texture from the box's own centre, so tiles only line up
        across seams when every centre sits on a whole number of texture periods;
        each slab's two far edges are chosen to make that so. The pinwheel keeps
        every slab large and near-square: MuJoCo lights a box from its corners, so a
        long thin strip beside the pit shades visibly differently from its
        neighbours.
        """
        period = 0.0
        element = next(
            (m for m in root.iter("material") if m.get("name") == material), None
        )
        if element is not None and element.get("texuniform") == "true":
            repeat = float(element.get("texrepeat", "1 1").split()[0])
            period = 1.0 / repeat

        def far_edge(near, direction):
            # The far edge at about FLOOR_EXTENT whose slab centre is on the grid.
            far = direction * FLOOR_EXTENT
            if period:
                centre = round((near + far) / 2 / period) * period
                far = 2 * centre - near
            return far

        x0, x1 = PIT_X
        y0, y1 = PIT_Y
        return {
            "north": (x0, far_edge(x0, 1), y1, far_edge(y1, 1)),
            "east": (x1, far_edge(x1, 1), far_edge(y1, -1), y1),
            "south": (far_edge(x1, -1), x1, far_edge(y0, -1), y0),
            "west": (far_edge(x0, -1), x0, y0, far_edge(y0, 1)),
        }

    def cube_in_hole(self, name):
        """All eight corners inside the opening and below the floor: corner-exact,
        mirroring the dustpan's cube_in_pan."""
        cube = self._objects_dict[name]
        position, quaternion = cube.get_pose()
        rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]])
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        corners = rotation.apply(signs * np.asarray(cube.size)) + position
        x, y, z = corners.T
        return bool(
            np.all(
                (x >= PIT_X[0])
                & (x <= PIT_X[1])
                & (y >= PIT_Y[0])
                & (y <= PIT_Y[1])
                & (z < 0.0)
            )
        )

    def wheel_positions(self):
        """World xy of the four wheel points."""
        base = self._robot_env.qpos["base"]
        x, y, yaw = (float(v) for v in base)
        cos, sin = np.cos(yaw), np.sin(yaw)
        return np.array(
            [
                [x + cos * dx - sin * dy, y + sin * dx + cos * dy]
                for dx, dy in WHEEL_OFFSETS
            ]
        )

    def robot_stuck_in_hole(self):
        """Stuck as soon as one wheel is over the opening.

        The base is a slide-joint body that never touches the floor, so it cannot
        physically drop in; being stuck is this rule, and `step` enforces it.
        """
        wheels = self.wheel_positions()
        return bool(
            np.any(
                (wheels[:, 0] > PIT_X[0])
                & (wheels[:, 0] < PIT_X[1])
                & (wheels[:, 1] > PIT_Y[0])
                & (wheels[:, 1] < PIT_Y[1])
            )
        )

    def step(self, action):
        """A stuck base ignores base commands and holds where it is; the arm and
        gripper still act. Only setting the base pose directly frees it."""
        if self.robot_stuck_in_hole():
            action = np.array(action, dtype=float, copy=True)
            robot = self._robot_env
            hold = 0.0 if robot.act_delta else np.asarray(robot.qpos["base"])
            if action.ndim == 2:
                action[:, :3] = hold
            else:
                action[:3] = hold
        return super().step(action)

    def _check_goals(self):
        return all(
            self.cube_in_hole(name)
            for name in self._objects_dict
            if name.startswith("cube_")
        )


class SweepIntoHole3DEnv(Broom3DEnv):
    """Five cubes, one native wiper, a floor pit and a shovel too heavy to lift."""

    def _create_object_centric_env(self, *args, **kwargs):
        return ObjectCentricSweepIntoHole3DEnv(*args, **kwargs)
