"""Native TidyBot control with bounded contact-triggered tool grasps."""

import mujoco
import numpy as np
from kinder.envs.dynamic3d.robots.tidybot_robot_env import TidyBotRobotEnv
from typing_extensions import override


class ToolGraspRobotEnv(TidyBotRobotEnv):
    """Release a separated tool rather than let its weld pull it back violently.

    The 3 cm limit bounds numerical grasp compliance; it is not a calibrated
    hardware force limit. Motors, gains, control frequency and contacts retain
    their native behavior. Check at every physics step, not only every action.
    """

    MAX_TOOL_GRASP_SEPARATION = 0.03
    TOOL_GRASP_NAMES = frozenset({"broom_grasp", "dustpan_grasp"})

    @override
    def _setup_robot_references(self):
        super()._setup_robot_references()
        model = self.sim.model.mj_model
        self._tool_grasp_ids = tuple(
            i for i in range(model.neq) if model.equality(i).name in self.TOOL_GRASP_NAMES
        )
        self.broken_tool_grasps = set()
        self.tool_grasp_break_events = []

    @override
    def _update_ctrl(self, action):
        model, data = self.sim.model.mj_model, self.sim.data.mj_data
        for constraint in self._tool_grasp_ids:
            if not data.eq_active[constraint]:
                continue
            rows = np.flatnonzero(
                (data.efc_type[: data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                & (data.efc_id[: data.nefc] == constraint)
            )
            if len(rows) < 3:
                continue
            separation = float(np.linalg.norm(data.efc_pos[rows[:3]]))
            if separation > self.MAX_TOOL_GRASP_SEPARATION:
                data.eq_active[constraint] = False
                self.broken_tool_grasps.add(constraint)
                self.tool_grasp_break_events.append(
                    dict(
                        time=float(data.time),
                        constraint=model.equality(constraint).name,
                        separation_m=separation,
                    )
                )
        super()._update_ctrl(action)
