"""Tossing with a fragile cube and a visual, bin-attached protective mat.

Only this variant adds damage events. Success and contact physics are unchanged.
"""

import math
import xml.etree.ElementTree as ET
from dataclasses import replace
from typing import Any

import mujoco
from kinder.envs.dynamic3d.envs import (
    ObjectCentricTidyBot3DEnv,
    TidyBot3DConfig,
    TidyBot3DEnv,
)
from kinder.envs.dynamic3d.task_families import Tossing3DEnv

# MuJoCo exposes these enum classes through its native extension.
# pylint: disable=no-member


def inside_mat(point: Any, center: Any, yaw: float, side: float) -> bool:
    """Test a world contact point against the bin's floor-aligned square."""
    dx, dy = point[0] - center[0], point[1] - center[1]
    c, s = math.cos(yaw), math.sin(yaw)
    return max(abs(c * dx + s * dy), abs(-s * dx + c * dy)) <= side / 2


class DamageTracker:
    """Track per-cube bare-ground contact onsets at physics frequency.

    Multiple contact points count once. A separated rebound can incur another
    charge. Human placement remains exempt until the cube is lifted clear.
    """

    def __init__(self, model: Any, *, mat_size: float, damage_cost: float) -> None:
        if not math.isfinite(mat_size) or mat_size <= 0:
            raise ValueError("mat_size must be finite and positive")
        if not math.isfinite(damage_cost) or damage_cost < 0:
            raise ValueError("damage_cost must be finite and nonnegative")
        self.model, self.mat_size, self.damage_cost = model, mat_size, damage_cost
        self.cubes = {
            i: model.body(i).name
            for i in range(model.nbody)
            if model.body(i).name.startswith("cube_")
        }
        self.bins = [
            i for i in range(model.nbody) if model.body(i).name.startswith("bin_")
        ]
        self.floor = {
            i
            for i in range(model.ngeom)
            if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
            and (model.geom_contype[i] or model.geom_conaffinity[i])
        }
        if not self.floor or not self.bins or not self.cubes:
            raise ValueError("Fragile tossing requires ground planes, bins and cubes")
        self.active: set[int] = set()
        self.exempt: set[int] = set()
        self.sequence = 0

    def exempt_placement(self) -> None:
        """Placement itself and ensuing settling are not robot damage."""
        self.active.clear()
        self.exempt = set(self.cubes)

    def mat_poses(self, data: Any) -> list[tuple[Any, float]]:
        """Return current bin floor positions and yaw angles."""
        result = []
        for body in self.bins:
            rotation = data.xmat[body].reshape(3, 3)
            yaw = math.atan2(rotation[1, 0], rotation[0, 0])
            result.append((data.xpos[body, :2], yaw))
        return result

    def update(self, data: Any) -> list[dict[str, Any]]:
        """Return newly charged bare-ground contacts at the current physics tick."""
        mats = self.mat_poses(data)
        contacts: dict[int, Any] = {}
        for contact in data.contact[: data.ncon]:
            if contact.dist > 0:
                continue
            a, b = int(contact.geom1), int(contact.geom2)
            cube_geom = b if a in self.floor else a if b in self.floor else None
            if cube_geom is None:
                continue
            body = int(self.model.geom_bodyid[cube_geom])
            if body in self.cubes and not any(
                inside_mat(contact.pos, center, yaw, self.mat_size)
                for center, yaw in mats
            ):
                contacts.setdefault(body, contact.pos.copy())
        # Release placement exemption once the cube is physically lifted clear.
        # Use oriented collision geometry, not a fixed center-height threshold.
        for body in tuple(self.exempt):
            geometries = [
                g
                for g in range(self.model.ngeom)
                if self.model.geom_bodyid[g] == body
                and self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
            ]
            bottom = min(
                (
                    data.geom_xpos[g, 2]
                    - sum(
                        abs(data.geom_xmat[g].reshape(3, 3)[2, axis])
                        * self.model.geom_size[g, axis]
                        for axis in range(3)
                    )
                    for g in geometries
                ),
                default=0,
            )
            if bottom > 0.005:
                self.exempt.remove(body)
        active = set(contacts) - self.exempt
        events = []
        for body in sorted(active - self.active):
            self.sequence += 1
            events.append(
                {
                    "event_id": self.sequence,
                    "cube": self.cubes[body],
                    "position": contacts[body].tolist(),
                    "simulation_time": float(data.time),
                    "cost": self.damage_cost,
                }
            )
        self.active = active
        return events


class ObjectCentricFragileTossing3DEnv(ObjectCentricTidyBot3DEnv):
    """Same task and controls with damage info and non-colliding mat visuals."""

    def __init__(
        self,
        *args: Any,
        mat_size: float = 4.0,
        damage_cost: float = 10.0,
        **kwargs: Any,
    ) -> None:
        if not math.isfinite(mat_size) or mat_size <= 0:
            raise ValueError("mat_size must be finite and positive")
        if not math.isfinite(damage_cost) or damage_cost < 0:
            raise ValueError("damage_cost must be finite and nonnegative")
        self.mat_size, self.damage_cost = mat_size, damage_cost
        self.damage_tracker: DamageTracker | None = None
        self.last_damage_events: list[dict[str, Any]] = []
        super().__init__(*args, **kwargs)

    def _create_scene_xml(self) -> str:
        root = ET.fromstring(super()._create_scene_xml())
        world = root.find("worldbody")
        assert world is not None
        bins = [
            b.get("name")
            for b in world.iter("body")
            if (b.get("name") or "").startswith("bin_")
        ]
        for name in bins:
            body = ET.SubElement(
                world, "body", name=f"protective_mat_{name}", mocap="true"
            )
            ET.SubElement(
                body,
                "geom",
                name=f"protective_mat_visual_{name}",
                type="box",
                size=f"{self.mat_size/2} {self.mat_size/2} .0005",
                rgba=".08 .65 .55 .45",
                contype="0",
                conaffinity="0",
                group="1",
                mass="0",
            )
        return ET.tostring(root, encoding="unicode")

    def reset(self, **kwargs: Any) -> Any:
        self.damage_tracker = None
        observation, info = super().reset(**kwargs)
        sim = self._robot_env.sim
        assert sim is not None
        self.damage_tracker = DamageTracker(
            sim.model.mj_model, mat_size=self.mat_size, damage_cost=self.damage_cost
        )
        self.damage_tracker.exempt_placement()
        self.last_damage_events = []
        self._update_mats()
        return observation, {**info, **self.damage_info()}

    def _update_mats(self) -> None:
        if self.damage_tracker is None:
            return
        sim = self._robot_env.sim
        assert sim is not None
        model, data = sim.model.mj_model, sim.data.mj_data
        for body, (center, yaw) in zip(
            self.damage_tracker.bins, self.damage_tracker.mat_poses(data)
        ):
            mat_body = model.body(f"protective_mat_{model.body(body).name}").id
            mocap = model.body_mocapid[mat_body]
            data.mocap_pos[mocap] = [*center, 0.001]
            data.mocap_quat[mocap] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]

    def damage_info(self) -> dict[str, Any]:
        """Expose current step events separately from task reward."""
        return {
            "damage_events": list(self.last_damage_events),
            "damage_cost": sum(e["cost"] for e in self.last_damage_events),
            "fragile_object": {
                "mat_size": self.mat_size,
                "damage_cost_per_impact": self.damage_cost,
            },
        }

    def step(self, action: Any) -> Any:
        sim = self._robot_env.sim
        assert sim is not None
        tracker = self.damage_tracker
        assert tracker is not None
        original = sim.step
        self.last_damage_events = []

        def tracked_step() -> None:
            original()
            self.last_damage_events.extend(tracker.update(sim.data.mj_data))
            self._update_mats()

        # Scoped to robot execution: initial settling and human placement do not
        # run the observer. Restore even if controller execution is interrupted.
        sim.step = tracked_step  # type: ignore[method-assign]
        try:
            observation, reward, terminated, truncated, info = super().step(action)
        finally:
            sim.step = original  # type: ignore[method-assign]
        return (
            observation,
            reward,
            terminated,
            truncated,
            {**info, **self.damage_info()},
        )

    def reset_ground_objects_to_regions(self, *args: Any, **kwargs: Any) -> Any:
        state = super().reset_ground_objects_to_regions(*args, **kwargs)
        if self.damage_tracker is not None:
            self.damage_tracker.exempt_placement()
        self.last_damage_events = []
        self._update_mats()
        return state

    def _set_state(self, state: Any) -> None:
        super()._set_state(state)
        if self.damage_tracker is not None:
            self.damage_tracker.exempt_placement()
        self.last_damage_events = []
        self._update_mats()

    def render(self) -> Any:
        self._update_mats()
        sim = self._robot_env.sim
        assert sim is not None
        sim.forward()
        return super().render()


class FragileTossing3DEnv(TidyBot3DEnv):
    """Configurable fragile variant, also accepting custom Tossing task layouts."""

    def __init__(
        self,
        *args: Any,
        num_objects: int = 1,
        task_config_path: str | None = None,
        **kwargs: Any,
    ) -> None:

        kwargs["config"] = replace(
            kwargs.pop("config", TidyBot3DConfig()), use_arm_velocities=True
        )
        super().__init__(
            *args,
            num_objects=num_objects,
            task_config_path=task_config_path
            or str(Tossing3DEnv.task_path(num_objects)),
            **kwargs,
        )

    def _create_object_centric_env(
        self, *args: Any, **kwargs: Any
    ) -> ObjectCentricFragileTossing3DEnv:
        return ObjectCentricFragileTossing3DEnv(*args, **kwargs)

    def _create_env_markdown_description(self) -> str:
        return (
            super()._create_env_markdown_description()
            + "\nThe cube is fragile. Bare-ground impacts outside the bin-attached "
            "protective squishy mat incur damage cost. The mat is non-colliding "
            "and contact physics is unchanged. See step info for damage events."
        )
