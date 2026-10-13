"""Sweep cubes off a cupboard shelf into a heavy box standing below its front edge."""

import re

import numpy as np
from scipy.spatial.transform import Rotation

from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv, TidyBot3DEnv

CUPBOARD = "cupboard_1"
BOX = "box_0"
_SHELF_GEOM = re.compile(rf"^{CUPBOARD}_shelf\d+$")


class ObjectCentricSweepOffShelf3DEnv(ObjectCentricTidyBot3DEnv):
    """Shelf3D's cupboard, cubes on one shelf, and a box too heavy for the arm."""

    def _model_and_data(self):
        sim = self._robot_env.sim
        return sim.model.mj_model, sim.data.mj_data

    def _shelf_geoms(self):
        model, data = self._model_and_data()
        geoms = [
            g for g in range(model.ngeom) if _SHELF_GEOM.match(model.geom(g).name or "")
        ]
        return sorted(geoms, key=lambda g: float(data.geom_xpos[g][2]))

    def _world_corners(self, geom):
        model, data = self._model_and_data()
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        local = signs * model.geom_size[geom]
        return local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]

    def shelf_top_z(self, shelf):
        """World height of the top surface of a board, 0 = the bottom board."""
        return float(self._world_corners(self._shelf_geoms()[shelf])[:, 2].max())

    def shelf_front_x(self):
        """World x of the cupboard's open front, the edge cubes are pulled over."""
        return float(
            min(self._world_corners(g)[:, 0].min() for g in self._shelf_geoms())
        )

    def box_interior(self):
        """((x0, y0, x1, y1), rim_z): the box's inner footprint and rim height."""
        box = self._objects_dict[BOX]
        position, quaternion = box.get_pose()
        yaw = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_euler("xyz")[2]
        if not np.isclose(np.sin(2 * yaw), 0.0, atol=1e-3):
            raise ValueError("box_interior assumes an axis-aligned box")
        half_x = box.length / 2 - box.wall_thickness
        half_y = box.width / 2 - box.wall_thickness
        if np.isclose(abs(np.sin(yaw)), 1.0, atol=1e-3):
            half_x, half_y = half_y, half_x
        x, y, z = (float(v) for v in position)
        rim = z + box.height
        return (x - half_x, y - half_y, x + half_x, y + half_y), rim

    def cube_in_box(self, name):
        """All eight cube corners inside the inner walls and below the rim."""
        (x0, y0, x1, y1), rim = self.box_interior()
        cube = self._objects_dict[name]
        position, quaternion = cube.get_pose()
        rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]])
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        corners = rotation.apply(signs * np.asarray(cube.size)) + position
        x, y, z = corners.T
        return bool(np.all((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1) & (z < rim)))

    def _check_goals(self):
        return all(
            self.cube_in_box(name)
            for name in self._objects_dict
            if name.startswith("cube_")
        )


class SweepOffShelf3DEnv(TidyBot3DEnv):
    """Sweep five cubes off the second shelf of Shelf3D's cupboard into a heavy box."""

    def _create_object_centric_env(self, *args, **kwargs):
        return ObjectCentricSweepOffShelf3DEnv(*args, **kwargs)
