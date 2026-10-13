"""Minimal native floor-sweeping task with robot/cube contact disabled."""

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv, TidyBot3DEnv
from kinder.envs.dynamic3d.robots.tool_grasp_robot_env import ToolGraspRobotEnv


class ObjectCentricBroom3DEnv(ObjectCentricTidyBot3DEnv):
    """Native assets and control, with contact-triggered tool grasp assistance."""

    graspable_tools = {"broom_grasp": "wiper_0"}

    _robot_env_classes = {
        **ObjectCentricTidyBot3DEnv._robot_env_classes,
        "tidybot": ToolGraspRobotEnv,
    }

    def _create_scene_xml(self):
        root = ET.fromstring(super()._create_scene_xml())
        equality = root.find("equality")
        if equality is None:
            equality = ET.SubElement(root, "equality")
        for constraint, tool in self.graspable_tools.items():
            ET.SubElement(
                equality,
                "weld",
                name=constraint,
                body1="robot_base",
                body2=tool,
                active="false",
            )
        return ET.tostring(root, encoding="unicode")

    def _update_grasp(self, closing):
        sim = self._robot_env.sim
        model, data = sim.model.mj_model, sim.data.mj_data
        constraints = [model.equality(name).id for name in self.graspable_tools]
        if not closing:
            data.eq_active[constraints] = False
            self._robot_env.broken_tool_grasps.clear()
            return
        # An overloaded grasp must be deliberately reopened before regrasping.
        if self._robot_env.broken_tool_grasps:
            return
        if any(data.eq_active[constraints]):
            return
        for name, tool in self.graspable_tools.items():
            if self._try_grasp(name, tool):
                return

    def _grasp_geoms(self, tool):
        model = self._robot_env.sim.model.mj_model
        body = model.body(tool).id
        return {
            max(
                (i for i in range(model.ngeom) if model.geom_bodyid[i] == body),
                key=lambda i: max(model.geom_size[i]),
            )
        }

    def _try_grasp(self, constraint_name, tool):
        sim = self._robot_env.sim
        model, data = sim.model.mj_model, sim.data.mj_data
        constraint = model.equality(constraint_name).id
        wiper_body = model.body(tool).id
        handles = self._grasp_geoms(tool)
        sides = set()
        for contact in data.contact:
            if contact.dist > 0 or not handles.intersection(
                (contact.geom1, contact.geom2)
            ):
                continue
            other = contact.geom2 if contact.geom1 in handles else contact.geom1
            name = model.geom(other).name or ""
            for side in ("left", "right"):
                if name.startswith(f"robot_{side}_pad"):
                    sides.add(side)
        if sides != {"left", "right"}:
            return
        # Preserve the current relative pose; engagement must not teleport a tool.
        palm = model.body("robot_base").id
        palm_r = Rotation.from_matrix(data.xmat[palm].reshape(3, 3))
        tool_r = Rotation.from_matrix(data.xmat[wiper_body].reshape(3, 3))
        model.eq_data[constraint, :3] = 0
        model.eq_data[constraint, 3:6] = palm_r.inv().apply(
            data.xpos[wiper_body] - data.xpos[palm]
        )
        model.eq_data[constraint, 6:10] = (palm_r.inv() * tool_r).as_quat()[
            [3, 0, 1, 2]
        ]
        data.eq_active[constraint] = True
        return True

    def step(self, action):
        action = np.asarray(action)
        if action.ndim != 1:
            raise ValueError(
                "Broom3D expects one native 11-dimensional action per step"
            )
        closing = action[10] > 0.5
        self._update_grasp(closing)
        result = super().step(action)
        self._update_grasp(closing)
        return result

    def collision_groups(self):
        model = self._robot_env.sim.model.mj_model
        groups = {name: [] for name in ("robot", "cube", "wiper", "floor")}
        for geom in range(model.ngeom):
            if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            body = int(model.geom_bodyid[geom])
            category = "floor"
            while body:
                name = model.body(body).name or ""
                if name.startswith("robot_") or name == "robot":
                    category = "robot"
                    break
                if name.startswith("cube_"):
                    category = "cube"
                    break
                if name == "wiper_0":
                    category = "wiper"
                    break
                body = int(model.body_parentid[body])
            groups[category].append(geom)
        return groups

    def _initialize_robot_pose(self):
        super()._initialize_robot_pose()
        model = self._robot_env.sim.model.mj_model
        for category, geoms in self.collision_groups().items():
            for geom in geoms:
                if category == "cube":
                    model.geom_contype[geom] = 4
                    model.geom_conaffinity[geom] = 28
                elif category != "robot":
                    model.geom_contype[geom] |= 8 if category == "wiper" else 16
                    model.geom_conaffinity[geom] |= 28


class Broom3DEnv(TidyBot3DEnv):
    """Five cubes, one native wiper and a floor goal region."""

    def _create_object_centric_env(self, *args, **kwargs):
        return ObjectCentricBroom3DEnv(*args, **kwargs)
