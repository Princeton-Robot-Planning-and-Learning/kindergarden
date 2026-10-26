"""Pour five cubes from a physical handled pitcher into a floor tray.

Uses the standard TidyBot robot, torque controllers, scene loading and rendering.
All seven movable objects have free joints. Only harness initialization places
the payload; no robot skill may move a vessel by assigning its pose.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv, TidyBot3DEnv
from kinder.envs.dynamic3d.objects.base import register_object
from kinder.envs.dynamic3d.objects.primitive_objects import Bin


@register_object
class PouringPitcher(Bin):
    """A rectangular open pitcher with a collision-bearing vertical handle."""

    def _create_xml_element(self) -> ET.Element:
        body = super()._create_xml_element()
        # The handle is open through its centre and participates in collisions.
        x = -self.length / 2
        for name, position, size in [
            ("handle_lower", [x - 0.03, 0, 0.065], [0.03, 0.012, 0.008]),
            ("handle_upper", [x - 0.03, 0, 0.175], [0.03, 0.012, 0.008]),
            ("handle_grip", [x - 0.06, 0, 0.12], [0.009, 0.012, 0.055]),
        ]:
            ET.SubElement(
                body,
                "geom",
                name=f"{self.name}_{name}",
                type="box",
                pos=" ".join(map(str, position)),
                size=" ".join(map(str, size)),
                rgba=self.rgba,
                mass="0.002",
                friction="0.7 0.005 0.0001",
                solref="0.004 1",
                solimp="0.99 0.999 0.001",
            )
        return body

    def get_bounding_box_dimensions(self) -> tuple[float, float, float]:
        return self.length + 0.138, self.width, self.height

    @staticmethod
    def get_bounding_box_from_config(
        pos: np.ndarray, object_config: dict[str, str | float]
    ) -> list[float]:
        bounds = Bin.get_bounding_box_from_config(pos, object_config)
        bounds[0] -= 0.069
        return bounds


class ObjectCentricPitcherPour3DEnv(ObjectCentricTidyBot3DEnv):
    """Physical containment, settled outcomes, and explicit payload initialization."""

    cube_names = tuple(f"cube_{i}" for i in range(5))

    def _model_and_data(self) -> tuple[Any, Any]:
        sim = self._robot_env.sim
        return sim.model.mj_model, sim.data.mj_data

    def _initialize_object_poses(self) -> None:
        super()._initialize_object_poses()
        pitcher = self.get_object("pitcher")
        origin = pitcher.get_position().astype(float)
        offsets = [
            (-0.045, -0.045),
            (0.045, -0.045),
            (-0.045, 0.045),
            (0.045, 0.045),
            (0.0, 0.0),
        ]
        for name, (x, y) in zip(self.cube_names, offsets):
            jitter = self.np_random.uniform(-0.003, 0.003, 2)
            self.get_object(name).set_pose(
                origin + [x + jitter[0], y + jitter[1], 0.035],
                np.array([1.0, 0.0, 0.0, 0.0]),
            )

    def cube_corners(self, name: str) -> np.ndarray:
        model, data = self._model_and_data()
        body = model.body(name).id
        geom = int(model.body_geomadr[body])
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        return (signs * model.geom_size[geom]) @ data.geom_xmat[geom].reshape(
            3, 3
        ).T + data.geom_xpos[geom]

    def cube_in_vessel(self, cube: str, vessel: str) -> bool:
        model, data = self._model_and_data()
        body = model.body(vessel).id
        geometry = self.get_object(vessel)
        assert isinstance(geometry, Bin)
        local = (self.cube_corners(cube) - data.xpos[body]) @ data.xmat[body].reshape(
            3, 3
        )
        interior = (
            np.array([geometry.length, geometry.width]) / 2 - geometry.wall_thickness
        )
        return bool(
            np.all(np.abs(local[:, :2]) <= interior + 0.002)
            and local[:, 2].min() >= geometry.wall_thickness - 0.002
            and local[:, 2].max() <= geometry.height + 0.002
        )

    def cube_locations(self) -> list[str]:
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
        return self.cube_locations() == ["tray"] * 5 and self.cubes_settled()

    def physical_snapshot(self) -> dict[str, Any]:
        model, data = self._model_and_data()
        return {
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
