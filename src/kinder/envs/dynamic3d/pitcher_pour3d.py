"""Pour a requested count of five cubes from a handled pitcher into a floor tray.

Uses the standard TidyBot robot, torque controllers, scene loading and rendering.
All seven movable objects have free joints. Harness initialization and the explicit
simulated human reset place the payload; robot skills use physical manipulation.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv, TidyBot3DEnv
from kinder.envs.dynamic3d.objects.base import register_object
from kinder.envs.dynamic3d.objects.primitive_objects import Bin, Cube


@register_object
class PouringPitcher(Bin):
    """Approximate ACCUPOUR 32 oz polypropylene pitcher, Amazon B08WX3JWDM.

    Product reference: https://www.amazon.com/dp/B08WX3JWDM
    The listing gives 4.41 x 5.91 x 7.01 inches overall and 0.12 kg. Rounded
    dimensions here are 112 mm across, 150 mm including handle/spout, 178 mm tall.
    Taper, 3 mm walls, spout profile and open-handle clearance are estimates,
    not manufacturer CAD or hardware calibration. The origin is the base centre;
    the spout points along +x and the handle sits on -x.
    """

    wall_segments = 32

    def cavity_ring(self, height: float, *, inner: bool = True) -> np.ndarray:
        """Cross section of the tapered shell, including the upper spout flare."""
        fraction = float(np.clip(height / self.height, 0, 1))
        angles = np.linspace(0, 2 * np.pi, self.wall_segments, endpoint=False)
        taper = 11 / 14 + 3 / 14 * fraction
        radii = (
            np.column_stack(
                (self.length / 2 * np.cos(angles), self.width / 2 * np.sin(angles))
            )
            * taper
        )
        flare = max(0.0, (fraction - 0.14 / 0.178) / (1 - 0.14 / 0.178))
        radii[:, 0] += 0.011 * flare * np.maximum(0, np.cos(angles)) ** 12
        if inner:
            radii -= self.wall_thickness * np.column_stack(
                (np.cos(angles), np.sin(angles))
            )
        return radii

    @property
    def handle_grasp_position(self) -> np.ndarray:
        """Centre of the upper handle grip, in the pitcher frame."""
        return np.array([-self.length / 2 - 0.021, 0, self.height * 0.135 / 0.178])

    def _create_xml_element(self) -> ET.Element:
        body = ET.Element("body", name=self.name)
        ET.SubElement(body, "freejoint", name=self.joint_name)
        self._wall_assets: list[ET.Element] = []
        # Separate convex wall panels preserve the hollow interior for collisions.
        # A single convex mesh of the whole pitcher would seal the mouth.
        levels = (0.0, self.height * 0.14 / 0.178, self.height)
        for band, (bottom, top) in enumerate(zip(levels[:-1], levels[1:])):
            rings = [
                np.column_stack(
                    (self.cavity_ring(z, inner=inner), np.full(self.wall_segments, z))
                )
                for z in (bottom, top)
                for inner in (False, True)
            ]
            for segment in range(self.wall_segments):
                name = f"{self.name}_wall_{band}_{segment}"
                vertices = np.concatenate(
                    [
                        ring[[segment, (segment + 1) % self.wall_segments]]
                        for ring in rings
                    ]
                )
                self._wall_assets.append(
                    ET.Element(
                        "mesh",
                        name=name,
                        vertex=" ".join(str(x) for x in vertices.ravel()),
                    )
                )
                ET.SubElement(
                    body,
                    "geom",
                    name=name,
                    type="mesh",
                    mesh=name,
                    mass=str(self.mass * 0.75 / (2 * self.wall_segments)),
                )
        # Coplanar box strips approximate the round floor. Box/box contacts give
        # small cubes a stable support manifold; a cylinder's convex contact
        # produced persistent rocking despite submillimetre pose changes.
        radius = self.length / 2 * 11 / 14
        strip_count = 16
        for index in range(strip_count):
            x = -radius + (index + 0.5) * 2 * radius / strip_count
            half_width = np.sqrt(radius * radius - x * x)
            ET.SubElement(
                body,
                "geom",
                name=f"{self.name}_bottom_{index}",
                type="box",
                size=f"{radius / strip_count} {half_width} {self.wall_thickness / 2}",
                pos=f"{x} 0 {self.wall_thickness / 2}",
                mass=str(self.mass * 0.15 / strip_count),
            )
        # Open-bottom hook: upper bridge plus outer grip, without a lower bridge.
        x = -self.length / 2
        for name, start, end in [
            ("handle_upper", [x + 0.004, 0, 0.162], [x - 0.021, 0, 0.162]),
            ("handle_grip", [x - 0.021, 0, 0.162], [x - 0.021, 0, 0.078]),
        ]:
            points = np.array([start, end])
            points[:, 2] *= self.height / 0.178
            ET.SubElement(
                body,
                "geom",
                name=f"{self.name}_{name}",
                type="capsule",
                fromto=" ".join(str(x) for x in points.ravel()),
                size="0.006",
                mass=str(self.mass * 0.05),
            )
        for geom in body.iter("geom"):
            geom.set("rgba", self.rgba)
            geom.set("friction", "0.7 0.005 0.0001")
            if self.solref is not None:
                geom.set("solref", " ".join(map(str, self.solref)))
            if self.solimp is not None:
                geom.set("solimp", " ".join(map(str, self.solimp)))
            geom.set("priority", "1")
        return body

    def get_assets(self) -> list[ET.Element]:
        """Return inline convex wall meshes; no downloaded asset is required."""
        return self._wall_assets

    def contains_points(self, points: np.ndarray, *, tolerance: float = 0.002) -> bool:
        """Check all points against the hollow tapered polygon, not its AABB."""
        if points[:, 2].min() < self.wall_thickness - tolerance:
            return False
        if points[:, 2].max() > self.height + tolerance:
            return False
        for point in points:
            ring = self.cavity_ring(float(point[2]))
            following = np.roll(ring, -1, axis=0)
            delta = following - ring
            u = np.clip(
                np.sum((point[:2] - ring) * delta, axis=1)
                / np.sum(delta * delta, axis=1),
                0,
                1,
            )
            distances = np.linalg.norm(point[:2] - ring - u[:, None] * delta, axis=1)
            if distances.min() <= tolerance:
                continue
            # Ray crossing also handles the slightly concave spout transitions.
            crossing = (ring[:, 1] > point[1]) != (following[:, 1] > point[1])
            edges = delta[crossing]
            starts = ring[crossing]
            intersections = (
                starts[:, 0] + (point[1] - starts[:, 1]) * edges[:, 0] / edges[:, 1]
            )
            if np.count_nonzero(intersections > point[0]) % 2 == 0:
                return False
        return True

    def get_bounding_box_dimensions(self) -> tuple[float, float, float]:
        return self.length + 0.038, self.width, self.height

    @staticmethod
    def get_bounding_box_from_config(
        pos: np.ndarray, object_config: dict[str, str | float]
    ) -> list[float]:
        bounds = Bin.get_bounding_box_from_config(pos, object_config)
        bounds[0] -= 0.027
        bounds[3] += 0.011
        return bounds


class ObjectCentricPitcherPour3DEnv(ObjectCentricTidyBot3DEnv):
    """Physical containment, settled outcomes, and explicit payload initialization."""

    cube_names = tuple(f"cube_{i}" for i in range(5))

    def __init__(self, *args: Any, requested_count: int = 5, **kwargs: Any) -> None:
        """Fix the requested split for this task instance; five preserves the default."""
        if (
            isinstance(requested_count, bool)
            or not isinstance(requested_count, int)
            or not 1 <= requested_count <= 5
        ):
            raise ValueError("requested_count must be an integer from 1 to 5")
        self.requested_count = requested_count
        super().__init__(*args, **kwargs)

    def _model_and_data(
        self,
    ) -> tuple[mujoco.MjModel, mujoco.MjData]:  # pylint: disable=no-member
        sim = self._robot_env.sim
        return sim.model.mj_model, sim.data.mj_data

    def _initialize_object_poses(self) -> None:
        super()._initialize_object_poses()
        self.respawn_payload_in_pitcher()

    def respawn_payload_in_pitcher(self) -> None:
        """Privileged harness/human reset of cubes only, never a robot skill.

        Represent a completed human intervention without animating tray pouring.
        Preserve both vessels and the robot; place all cubes in the pitcher's
        current frame and clear their velocities before recomputing contacts.
        """
        pitcher = self.get_object("pitcher")
        assert isinstance(pitcher, PouringPitcher)
        origin = pitcher.get_position().astype(float)
        orientation = pitcher.get_orientation().astype(float)
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, orientation)  # pylint: disable=no-member
        rotation = rotation.reshape(3, 3)
        angles = np.linspace(0, 2 * np.pi, len(self.cube_names), endpoint=False)
        offsets = 0.024 * np.column_stack((np.cos(angles), np.sin(angles)))
        for name, (x, y) in zip(self.cube_names, offsets):
            jitter = self.np_random.uniform(-0.0005, 0.0005, 2)
            cube = self.get_object(name)
            assert isinstance(cube, Cube)
            height = pitcher.wall_thickness + cube.size[2]
            cube.set_pose(
                origin + rotation @ np.array([x + jitter[0], y + jitter[1], height]),
                orientation,
            )
            cube.set_velocity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        model, data = self._model_and_data()
        mujoco.mj_forward(model, data)  # pylint: disable=no-member

    def cube_corners(self, name: str) -> np.ndarray:
        """Return the eight physical cube corners in world coordinates."""
        model, data = self._model_and_data()
        body = model.body(name).id
        geom = int(model.body_geomadr[body])
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        return (signs * model.geom_size[geom]) @ data.geom_xmat[geom].reshape(
            3, 3
        ).T + data.geom_xpos[geom]

    def cube_in_vessel(self, cube: str, vessel: str) -> bool:
        """Check whole-cube containment in the actual vessel frame."""
        model, data = self._model_and_data()
        body = model.body(vessel).id
        geometry = self.get_object(vessel)
        assert isinstance(geometry, Bin)
        local = (self.cube_corners(cube) - data.xpos[body]) @ data.xmat[body].reshape(
            3, 3
        )
        if isinstance(geometry, PouringPitcher):
            return geometry.contains_points(local)
        interior = (
            np.array([geometry.length, geometry.width]) / 2 - geometry.wall_thickness
        )
        return bool(
            np.all(np.abs(local[:, :2]) <= interior + 0.002)
            and local[:, 2].min() >= geometry.wall_thickness - 0.002
            and local[:, 2].max() <= geometry.height + 0.002
        )

    def cube_locations(self) -> list[str]:
        """Classify each cube using containment and floor contact height."""
        result = []
        for cube in self.cube_names:
            if self.cube_in_vessel(cube, "pitcher"):
                result.append("pitcher")
            elif self.cube_in_vessel(cube, "tray"):
                result.append("tray")
            else:
                corners = self.cube_corners(cube)
                result.append("floor" if corners[:, 2].min() < 0.005 else "air")
        return result

    def cubes_settled(self) -> bool:
        """Require every payload cube to have small linear and angular speed."""
        model, data = self._model_and_data()
        for name in self.cube_names:
            joint = int(model.body_jntadr[model.body(name).id])
            dof = int(model.jnt_dofadr[joint])
            if np.linalg.norm(data.qvel[dof : dof + 3]) > 0.025:
                return False
            if np.linalg.norm(data.qvel[dof + 3 : dof + 6]) > 0.25:
                return False
        return True

    def _check_goals(self) -> bool:
        locations = self.cube_locations()
        return (
            locations.count("tray") == self.requested_count
            and locations.count("pitcher") == 5 - self.requested_count
            and self.cubes_settled()
        )

    def physical_snapshot(self) -> dict[str, Any]:
        """Capture the physical state and measured outcomes for recording."""
        model, data = self._model_and_data()
        return {
            "requested_count": self.requested_count,
            "time": float(data.time),
            "qpos": data.qpos.tolist(),
            "qvel": data.qvel.tolist(),
            "ctrl": data.ctrl.tolist(),
            "locations": self.cube_locations(),
            "settled": self.cubes_settled(),
            "goal": self._check_goals(),
            "bodies": {
                name: {
                    "position": data.xpos[model.body(name).id].tolist(),
                    "quaternion": data.xquat[model.body(name).id].tolist(),
                }
                for name in ["pitcher", "tray", *self.cube_names]
            },
        }


class PitcherPour3DEnv(TidyBot3DEnv):
    """Native low-level KinDER environment with five cubes and two free vessels."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault(
            "task_config_path",
            str(Path(__file__).parent / "tasks/PitcherPour3D/PitcherPour3D-o5.json"),
        )
        kwargs.setdefault("scene_render_camera", "task_view")
        kwargs.setdefault("scene_bg", True)
        super().__init__(**kwargs)

    def _create_object_centric_env(
        self, *args: Any, **kwargs: Any
    ) -> ObjectCentricPitcherPour3DEnv:
        return ObjectCentricPitcherPour3DEnv(*args, **kwargs)

    def _create_variant_specific_description(self) -> str:
        scene = self._object_centric_env
        assert isinstance(scene, ObjectCentricPitcherPour3DEnv)
        return (
            f"Exactly {scene.requested_count} cubes must settle in the tray and "
            f"{5 - scene.requested_count} remain settled in the pitcher, with no spills."
        )
