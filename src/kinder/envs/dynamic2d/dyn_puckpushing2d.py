"""Dynamic PuckPushing2D: strike a puck across a floor maze of unknown friction.

A mobile robot with a paddle pushes a puck to a goal through a maze (see
`kinder.envs.puckpushing_utils` for the maps and the shared physics). The floor is tiled
with surface patches, and each surface's sliding friction coefficient is hidden. Each
action is one push: the robot drives behind the puck, accelerates it with its paddle to
the commanded speed along the commanded heading, and lets go. The puck then slides until
friction stops it, bouncing off walls. If its center passes over a hole (a red circle),
the episode fails.

Top-down PyMunk has no floor contact, so the floor's Coulomb friction is applied to the
puck's velocity directly, using the patch under its center. The same model drives the
planners' analytic slide model.

The info dict reports what the robot measures: the release speed and the stop position,
both noisy. Planners must not read the true friction coefficients.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pymunk
from numpy.typing import NDArray
from relational_structs import (
    Array,
    Object,
    ObjectCentricState,
    ObjectCentricStateSpace,
    Type,
)
from relational_structs.utils import create_state_from_dict

from kinder.core import ConstantObjectKinDEREnv, FinalConfigMeta
from kinder.envs.dynamic2d.base_env import (
    Dynamic2DRobotEnvConfig,
    ObjectCentricDynamic2DRobotEnv,
)
from kinder.envs.dynamic2d.object_types import (
    Dynamic2DRobotEnvTypeFeatures,
    KinRectangleType,
    KinRobotType,
    SmallCircleType,
)
from kinder.envs.kinematic2d.structs import ZOrder
from kinder.envs.puckpushing_utils import (
    DEFAULT_FRICTION_PRIOR,
    GOAL_RGB,
    GRAVITY,
    HOLE_RGB,
    MAPS,
    PUCK_RGB,
    SURFACE_RGB,
    WALL_RGB,
    PuckPhysics,
    PuckWorld,
    PushActionSpace,
    sample_frictions,
)

PuckType = Type("puck", parent=SmallCircleType)
Dynamic2DRobotEnvTypeFeatures[PuckType] = list(
    Dynamic2DRobotEnvTypeFeatures[SmallCircleType]
)
HoleMarkerType = Type("hole_marker", parent=SmallCircleType)
Dynamic2DRobotEnvTypeFeatures[HoleMarkerType] = list(
    Dynamic2DRobotEnvTypeFeatures[SmallCircleType]
)
GoalMarkerType = Type("goal_marker", parent=SmallCircleType)
Dynamic2DRobotEnvTypeFeatures[GoalMarkerType] = list(
    Dynamic2DRobotEnvTypeFeatures[SmallCircleType]
)

_PUCK_COLLISION_TYPE = 11
_WALL_COLLISION_TYPE = 12
_PADDLE_COLLISION_TYPE = 13
_PADDLE_THICKNESS = 0.04
# The paddle starts this far behind the puck so the two do not overlap.
_PADDLE_GAP = 0.005


@dataclass(frozen=True)
class DynPuckPushing2DEnvConfig(Dynamic2DRobotEnvConfig, metaclass=FinalConfigMeta):
    """Config for DynPuckPushing2DEnv()."""

    map_name: str = "threeroutes"
    physics: PuckPhysics = PuckPhysics()
    friction_prior: tuple[tuple[float, float], ...] = DEFAULT_FRICTION_PRIOR

    # Physics. No gravity or damping in the plane; floor friction is applied directly.
    gravity_y: float = 0.0
    damping: float = 1.0
    sim_hz: int = 240
    stabilization_seconds: float = 0.0
    # A slide ends when the puck is slower than this, or after max_slide_seconds.
    stop_speed: float = 1e-3
    max_slide_seconds: float = 30.0

    # Matches the threeroutes map; other maps reset these from their own size.
    world_min_x: float = 0.0
    world_max_x: float = 5.2
    world_min_y: float = 0.0
    world_max_y: float = 3.6

    render_dpi: int = 100


class ObjectCentricDynPuckPushing2DEnv(
    ObjectCentricDynamic2DRobotEnv[DynPuckPushing2DEnvConfig]
):
    """Push a puck to a goal across floor patches of unknown friction."""

    def __init__(
        self,
        config: DynPuckPushing2DEnvConfig | None = None,
        map_name: str | None = None,
        **kwargs,
    ) -> None:
        config = config or DynPuckPushing2DEnvConfig()
        if map_name is not None:
            config = dataclasses.replace(config, map_name=map_name)
        puck_map = MAPS[config.map_name]
        config = dataclasses.replace(
            config,
            world_min_x=0.0,
            world_max_x=puck_map.width,
            world_min_y=0.0,
            world_max_y=puck_map.height,
        )
        self.world = PuckWorld(puck_map, config.physics)
        super().__init__(config, **kwargs)
        self._frictions: tuple[float, ...] = tuple(
            (lo + hi) / 2 for lo, hi in config.friction_prior
        )
        self._robot_pose = np.array([*puck_map.robot_start, 0.0])
        self._puck_body: pymunk.Body | None = None
        self._paddle_body: pymunk.Body | None = None
        self._outcome: str | None = None  # "goal", "hole" or None
        self._last_info: dict[str, Any] = {}
        # Built now so renders include the walls and patches from the first reset.
        self._initial_constant_state = self._create_constant_initial_state()

    # ------------------------------------------------------------------ spaces

    def _create_action_space(  # type: ignore[override]
        self, config: DynPuckPushing2DEnvConfig
    ) -> PushActionSpace:
        return PushActionSpace(config.physics.max_speed)

    # ------------------------------------------------------------------ hidden

    @property
    def frictions(self) -> tuple[float, ...]:
        """The true (hidden) friction coefficient per surface.

        For evaluation; planners must not read it.
        """
        return self._frictions

    # ------------------------------------------------------------------ state

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        state: dict[Object, dict[str, float]] = {}
        for i, (x0, y0, x1, y1) in enumerate(self.world.puck_map.wall_rects()):
            state[Object(f"wall{i}", KinRectangleType)] = _rect_features(
                x0, y0, x1, y1, WALL_RGB, ZOrder.SURFACE
            )
        for i, (surface, (x0, y0, x1, y1)) in enumerate(
            self.world.puck_map.surface_rects()
        ):
            state[Object(f"patch{i}", KinRectangleType)] = _rect_features(
                x0, y0, x1, y1, SURFACE_RGB[surface], ZOrder.FLOOR
            )
        return state

    def _sample_initial_state(self) -> ObjectCentricState:
        puck_map = self.world.puck_map
        robot = np.array([*puck_map.robot_start, 0.0])
        return self._create_state(robot, np.asarray(puck_map.puck_start))

    def _create_state(
        self, robot_pose: NDArray[np.float64], puck: NDArray[np.float64]
    ) -> ObjectCentricState:
        phys = self.config.physics
        state: dict[Object, dict[str, float]] = {}
        state[Object("robot", KinRobotType)] = {
            "x": robot_pose[0],
            "y": robot_pose[1],
            "theta": robot_pose[2],
            "vx_base": 0.0,
            "vy_base": 0.0,
            "omega_base": 0.0,
            "vx_arm": 0.0,
            "vy_arm": 0.0,
            "omega_arm": 0.0,
            "vx_gripper_l": 0.0,
            "vy_gripper_l": 0.0,
            "omega_gripper_l": 0.0,
            "vx_gripper_r": 0.0,
            "vy_gripper_r": 0.0,
            "omega_gripper_r": 0.0,
            "static": False,
            "base_radius": phys.robot_radius,
            # The paddle is drawn as the gripper base, with its face at the offset.
            "arm_joint": phys.paddle_offset - _PADDLE_THICKNESS / 2,
            "arm_length": phys.paddle_offset,
            "gripper_base_width": _PADDLE_THICKNESS,
            "gripper_base_height": phys.paddle_width,
            "finger_gap": 0.0,
            "finger_height": 0.01,
            "finger_width": 0.01,
        }
        state[Object("puck", PuckType)] = _circle_features(
            puck[0], puck[1], phys.puck_radius, PUCK_RGB, ZOrder.ALL, phys.puck_mass
        )
        goal = Object("goal", GoalMarkerType)
        state[goal] = _circle_features(
            float(self.world.goal[0]),
            float(self.world.goal[1]),
            phys.goal_radius,
            GOAL_RGB,
            ZOrder.SURFACE,
            0.0,
        )
        state[goal]["static"] = True
        for i, (hx, hy) in enumerate(self.world.holes):
            hole = Object(f"hole{i}", HoleMarkerType)
            state[hole] = _circle_features(
                hx, hy, phys.hole_radius, HOLE_RGB, ZOrder.SURFACE, 0.0
            )
            state[hole]["static"] = True
        return create_state_from_dict(state, Dynamic2DRobotEnvTypeFeatures)

    # ------------------------------------------------------------------ physics

    def _setup_physics_space(self) -> None:
        self.pymunk_space = pymunk.Space()
        self.pymunk_space.gravity = 0, 0
        self.pymunk_space.damping = self.config.damping
        self.pymunk_space.collision_slop = self.config.collision_slop
        self.robot = None

    def _add_state_to_space(self, state: ObjectCentricState) -> None:
        assert self.pymunk_space is not None
        phys = self.config.physics
        walls = pymunk.Body(body_type=pymunk.Body.STATIC)
        shapes = []
        for x0, y0, x1, y1 in self.world.puck_map.wall_rects():
            shape = pymunk.Poly(walls, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            shape.elasticity = 1.0  # times the puck's, which is the restitution
            shape.friction = 0.0
            shape.collision_type = _WALL_COLLISION_TYPE
            shapes.append(shape)
        self.pymunk_space.add(walls, *shapes)

        puck_obj = state.get_objects(PuckType)[0]
        moment = pymunk.moment_for_circle(phys.puck_mass, 0, phys.puck_radius)
        puck = pymunk.Body(phys.puck_mass, moment)
        puck.position = (state.get(puck_obj, "x"), state.get(puck_obj, "y"))
        puck.velocity = (state.get(puck_obj, "vx"), state.get(puck_obj, "vy"))
        puck_shape = pymunk.Circle(puck, phys.puck_radius)
        puck_shape.elasticity = phys.wall_restitution
        puck_shape.friction = 0.0
        puck_shape.collision_type = _PUCK_COLLISION_TYPE
        puck.velocity_func = self._floor_friction
        self.pymunk_space.add(puck, puck_shape)
        self._puck_body = puck
        self._state_obj_to_pymunk_body[puck_obj] = puck

        robot_obj = state.get_objects(KinRobotType)[0]
        self._robot_pose = np.array(
            [state.get(robot_obj, "x"), state.get(robot_obj, "y")]
            + [state.get(robot_obj, "theta")]
        )
        # The paddle's frame is at the middle of its face, facing +x.
        paddle = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        half_w = phys.paddle_width / 2
        paddle_shape = pymunk.Poly(
            paddle,
            [
                (-_PADDLE_THICKNESS, -half_w),
                (0.0, -half_w),
                (0.0, half_w),
                (-_PADDLE_THICKNESS, half_w),
            ],
        )
        paddle_shape.elasticity = 0.0
        paddle_shape.friction = 0.0
        paddle_shape.collision_type = _PADDLE_COLLISION_TYPE
        self.pymunk_space.add(paddle, paddle_shape)
        self._paddle_body = paddle
        self._place_paddle()

    def _floor_friction(
        self, body: pymunk.Body, gravity, damping: float, dt: float
    ) -> None:
        """Coulomb friction from the patch under the puck's center.

        Replaces PyMunk's velocity update, so gravity and damping are unused.
        """
        del gravity, damping
        velocity = body.velocity
        speed = velocity.length
        body.angular_velocity = 0.0
        if speed <= 0.0:
            return
        point = np.array([body.position.x, body.position.y])
        decel = self.world.friction_at(point, self._frictions) * GRAVITY * dt
        if speed <= decel:
            body.velocity = (0.0, 0.0)
        else:
            body.velocity = velocity * (1.0 - decel / speed)

    def _place_paddle(
        self,
        face: NDArray[np.float64] | None = None,
        angle: float | None = None,
    ) -> None:
        """Put the paddle face at `face`, facing `angle`; by default in front of the
        robot."""
        assert self._paddle_body is not None
        heading = float(self._robot_pose[2])
        if face is None:
            reach = self.config.physics.paddle_offset
            face = self._robot_pose[:2] + reach * np.array(
                [math.cos(heading), math.sin(heading)]
            )
        self._paddle_body.position = (float(face[0]), float(face[1]))
        self._paddle_body.angle = heading if angle is None else angle
        self._paddle_body.velocity = (0.0, 0.0)
        assert self.pymunk_space is not None
        self.pymunk_space.reindex_shapes_for_body(self._paddle_body)

    def _read_state_from_space(self) -> None:
        assert self._current_state is not None
        state = self._current_state.copy()
        puck_obj = state.get_objects(PuckType)[0]
        if self._puck_body is not None:
            state.set(puck_obj, "x", self._puck_body.position.x)
            state.set(puck_obj, "y", self._puck_body.position.y)
            state.set(puck_obj, "vx", self._puck_body.velocity.x)
            state.set(puck_obj, "vy", self._puck_body.velocity.y)
        robot_obj = state.get_objects(KinRobotType)[0]
        state.set(robot_obj, "x", self._robot_pose[0])
        state.set(robot_obj, "y", self._robot_pose[1])
        state.set(robot_obj, "theta", self._robot_pose[2])
        self._current_state = state

    # ------------------------------------------------------------------ reset/step

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[ObjectCentricState, dict]:
        """Reset.

        `options["frictions"]` may fix the hidden friction coefficients.
        """
        obs, info = super().reset(seed=seed, options=options)
        if options is not None and "frictions" in options:
            frictions = tuple(float(f) for f in options["frictions"])
            assert len(frictions) == len(self.config.friction_prior)
            self._frictions = frictions
        elif options is None or "init_state" not in options:
            self._frictions = sample_frictions(
                self.np_random, self.config.friction_prior
            )
        self._outcome = None
        self._last_info = {}
        return obs, info

    def step(self, action: Array) -> tuple[ObjectCentricState, float, bool, bool, dict]:
        assert self.action_space.contains(np.asarray(action, dtype=np.float64))
        assert self.pymunk_space is not None and self._puck_body is not None
        assert self._outcome is None, "Episode is over; call reset()"
        angle, speed = float(action[0]), float(action[1])
        puck = np.array([self._puck_body.position.x, self._puck_body.position.y])
        info: dict[str, Any] = {"push_feasible": False, "drive_distance": 0.0}

        if self.world.push_feasible(puck, angle):
            info.update(self._strike(puck, angle, speed))
        else:
            info.update(
                release_speed=0.0,
                stop_position=self._measure_position(puck),
                fell_in_hole=False,
                puck_trajectory=[puck.copy()],
            )

        stop = np.array([self._puck_body.position.x, self._puck_body.position.y])
        if info["fell_in_hole"]:
            self._outcome = "hole"
        elif self.world.in_goal(stop):
            self._outcome = "goal"
        info["goal_reached"] = self._outcome == "goal"
        self._last_info = info
        reward, terminated = self._get_reward_and_done()
        return self._get_obs(), reward, terminated, False, info

    def _strike(self, puck: NDArray[np.float64], angle: float, speed: float) -> dict:
        """Drive behind the puck, push it to speed with the paddle, let it slide."""
        assert self.pymunk_space is not None
        assert self._puck_body is not None and self._paddle_body is not None
        phys = self.config.physics
        rng = self.np_random
        # The robot aims and accelerates imperfectly.
        actual_angle = angle + float(rng.normal(0.0, phys.angle_noise))
        actual_speed = max(
            speed * (1.0 + float(rng.normal(0.0, phys.speed_noise))), 0.0
        )

        # The commanded push is feasible; a noisy heading may not be, and then the
        # robot stands as for the commanded one.
        pose = self.world.push_pose(puck, actual_angle) or self.world.push_pose(
            puck, angle
        )
        assert pose is not None
        station, heading = pose
        direction = np.array([math.cos(actual_angle), math.sin(actual_angle)])
        start_node = self.world.nav_node_of(self._robot_pose[:2])
        drive = self.world.drive_distance(start_node, station)
        self._robot_pose = np.array([*(station - _PADDLE_GAP * direction), heading])
        face = puck - (phys.puck_radius + _PADDLE_GAP) * direction
        self._place_paddle(face, actual_angle)
        # The base translates with the paddle during the strike.
        base_from_paddle = self._robot_pose[:2] - face

        dt = 1.0 / self.config.sim_hz
        trajectory = [puck.copy()]
        fell = False
        release_speed = 0.0
        # Accelerate uniformly over the acceleration distance, then stop dead.
        if actual_speed > 0:
            accel = actual_speed**2 / (2.0 * phys.accel_distance)
            num_accel = max(int(round(actual_speed / accel / dt)), 1)
            for k in range(num_accel):
                v = accel * (k + 1) * dt
                self._paddle_body.velocity = tuple(v * direction)
                self.pymunk_space.step(dt)
                self._robot_pose[:2] = (
                    np.array(self._paddle_body.position) + base_from_paddle
                )
                fell = fell or self._over_hole()
                trajectory.append(self._puck_position())
            self._paddle_body.velocity = (0.0, 0.0)
            release_speed = float(self._puck_body.velocity.length)
        # The robot is clear once it lets go.
        self.pymunk_space.remove(*self._paddle_body.shapes)
        num_slide = int(self.config.max_slide_seconds / dt)
        for _ in range(num_slide):
            if fell or self._puck_body.velocity.length < self.config.stop_speed:
                break
            self.pymunk_space.step(dt)
            fell = self._over_hole()
            trajectory.append(self._puck_position())
        self._puck_body.velocity = (0.0, 0.0)
        self.pymunk_space.add(*self._paddle_body.shapes)
        self._place_paddle()
        stop = self._puck_position()
        noisy_speed = release_speed + float(rng.normal(0.0, phys.speed_obs_noise))
        return {
            "push_feasible": True,
            "drive_distance": drive,
            "release_speed": max(noisy_speed, 0.0),
            "stop_position": self._measure_position(stop),
            "fell_in_hole": fell,
            # For analysis and videos only: planners observe the two measurements.
            "puck_trajectory": trajectory,
        }

    def _puck_position(self) -> NDArray[np.float64]:
        assert self._puck_body is not None
        return np.array([self._puck_body.position.x, self._puck_body.position.y])

    def _over_hole(self) -> bool:
        return self.world.over_hole(self._puck_position())

    def _measure_position(self, point: NDArray[np.float64]) -> NDArray[np.float64]:
        noise = self.config.physics.position_obs_noise
        return point + self.np_random.normal(0.0, noise, size=2)

    def _get_reward_and_done(self) -> tuple[float, bool]:
        return -1.0, self._outcome is not None

    def _get_info(self) -> dict:
        return dict(self._last_info)

    def get_action_from_gui_input(self, gui_input: dict[str, Any]) -> NDArray[Any]:
        raise NotImplementedError("PuckPushing has no GUI controls")


def _rect_features(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    rgb: tuple[float, float, float],
    z_order: ZOrder,
) -> dict[str, float]:
    return {
        "x": (x0 + x1) / 2,
        "y": (y0 + y1) / 2,
        "theta": 0.0,
        "vx": 0.0,
        "vy": 0.0,
        "omega": 0.0,
        "static": True,
        "held": False,
        "color_r": rgb[0],
        "color_g": rgb[1],
        "color_b": rgb[2],
        "z_order": z_order.value,
        "width": x1 - x0,
        "height": y1 - y0,
    }


def _circle_features(
    x: float,
    y: float,
    radius: float,
    rgb: tuple[float, float, float],
    z_order: ZOrder,
    mass: float,
) -> dict[str, float]:
    return {
        "x": x,
        "y": y,
        "theta": 0.0,
        "vx": 0.0,
        "vy": 0.0,
        "omega": 0.0,
        "static": False,
        "held": False,
        "color_r": rgb[0],
        "color_g": rgb[1],
        "color_b": rgb[2],
        "z_order": z_order.value,
        "radius": radius,
        "mass": mass,
    }


class DynPuckPushing2DEnv(ConstantObjectKinDEREnv):
    """Dynamic PuckPushing2D env with a constant set of objects."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert isinstance(
            self._object_centric_env.observation_space, ObjectCentricStateSpace
        )
        self.observation_space = self._object_centric_env.observation_space.to_box(
            self._constant_objects,
            self._object_centric_env.type_features,
            dtype=np.float64,
        )

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricDynamic2DRobotEnv:
        return ObjectCentricDynPuckPushing2DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        holes = sorted(o.name for o in exemplar_state if o.name.startswith("hole"))
        return ["robot", "puck", "goal"] + holes

    def _create_env_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return """A 2D physics environment where a mobile robot pushes a puck through a
               floor maze to a goal.

               The floor is tiled with three surfaces (laminate, wood, felt), each with
               a hidden sliding friction coefficient. Each action is one push: the robot
               drives behind the puck and strikes it with its paddle at the commanded
               heading and speed, with small execution noise. The puck slides until
               friction stops it, bouncing off walls. The episode fails if the puck's
               center passes over a hole (red circle).

               The info dict reports the noisy release speed and stop position, the
               driving distance, and whether the push was feasible.
               """

    def _create_variant_markdown_description(self) -> str:
        return "Variants differ in the maze."

    def _create_variant_specific_description(self) -> str:
        env = self._object_centric_env
        assert isinstance(env, ObjectCentricDynPuckPushing2DEnv)
        return f"This variant uses the {env.world.puck_map.name} maze."

    def _create_reward_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return """A reward of -1 per push.

               The episode terminates when the puck stops inside the goal circle, or
               fails when its center passes over a hole.
               """

    def _create_references_markdown_description(self) -> str:
        return "N/A."
