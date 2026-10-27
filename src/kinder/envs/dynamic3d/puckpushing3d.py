"""PuckPushing3D: TidyBot strikes a puck across a floor maze of unknown friction.

The 3D counterpart of `kinder.envs.dynamic2d.dyn_puckpushing2d`, on the same maps (see
`kinder.envs.puckpushing_utils`). The TidyBot holds a paddle low in front of its base.
Each action is one push: the robot drives behind the puck, the paddle accelerates the
puck to the commanded speed along the commanded heading and lets go, and the puck slides
until friction stops it, bouncing off the walls. Holes are red circles painted on the
floor; the episode fails if the puck's center passes over one.

PyBullet simulates the contacts (paddle on puck, puck on walls). Gravity is off and the
puck floats at its resting height, while the floor's Coulomb friction is applied to its
velocity from the patch under its center. This avoids Bullet's contacts catching on the
edges between floor tiles, and matches the planners' analytic slide model.

The robot base is teleported, as in the other Kinematic3D envs; during a strike it
follows the paddle. The info dict reports the noisy release speed and stop position.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any
from typing import Type as TypingType

import numpy as np
import pybullet as p
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose, SE2Pose, get_pose, set_pose
from pybullet_helpers.joint import JointPositions
from relational_structs import Array, Object, ObjectCentricState
from relational_structs.utils import create_state_from_dict

from kinder.core import ConstantObjectKinDEREnv, FinalConfigMeta
from kinder.envs.kinematic3d.base_env import (
    Kinematic3DEnvConfig,
    ObjectCentricKinematic3DRobotEnv,
)
from kinder.envs.kinematic3d.object_types import (
    Kinematic3DCuboidType,
    Kinematic3DEnvTypeFeatures,
    Kinematic3DPointType,
    Kinematic3DRobotType,
)
from kinder.envs.kinematic3d.utils import Kinematic3DObjectCentricState
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

# The paddle starts this far behind the puck so the two do not overlap.
_PADDLE_GAP = 0.005


@dataclass(frozen=True)
class PuckPushing3DEnvConfig(Kinematic3DEnvConfig, metaclass=FinalConfigMeta):
    """Config for PuckPushing3DEnv()."""

    map_name: str = "threeroutes"
    physics: PuckPhysics = PuckPhysics()
    friction_prior: tuple[tuple[float, float], ...] = DEFAULT_FRICTION_PRIOR

    wall_height: float = 0.3
    paddle_thickness: float = 0.04
    paddle_height: float = 0.15
    paddle_mass: float = 20.0
    # The arm holds the paddle with the gripper pointing down just behind it.
    initial_joints: JointPositions = field(
        default_factory=lambda: [2.439, -1.75, 0.866, -2.147, -2.156, -1.118, 1.648]
    )
    realistic_bg: bool = False

    physics_dt: float = 1.0 / 240.0
    # Bullet's contacts come out this much bouncier than the restitution it is given,
    # so the puck gets the target restitution plus this offset (calibrated by shooting
    # the puck at a wall).
    bullet_restitution_offset: float = -0.07
    stop_speed: float = 1e-3
    max_slide_seconds: float = 30.0

    def get_camera_kwargs(self) -> dict[str, Any]:
        puck_map = MAPS[self.map_name]
        return {
            "camera_target": (puck_map.width / 2, puck_map.height / 2, 0.0),
            "camera_yaw": 0,
            "camera_distance": 0.9 * max(puck_map.width, puck_map.height),
            "camera_pitch": -60,
        }


class PuckPushing3DObjectCentricState(Kinematic3DObjectCentricState):
    """A state in the PuckPushing3DEnv().

    The friction coefficients are not part of the state: they are what the robot must
    infer.
    """


class ObjectCentricPuckPushing3DEnv(
    ObjectCentricKinematic3DRobotEnv[
        PuckPushing3DObjectCentricState, PuckPushing3DEnvConfig
    ]
):
    """Push a puck to a goal across floor patches of unknown friction."""

    def __init__(
        self,
        config: PuckPushing3DEnvConfig | None = None,
        map_name: str | None = None,
        **kwargs,
    ) -> None:
        config = config or PuckPushing3DEnvConfig()
        if map_name is not None:
            config = dataclasses.replace(config, map_name=map_name)
        puck_map = MAPS[config.map_name]
        config = dataclasses.replace(
            config,
            robot_base_home_pose=SE2Pose(*puck_map.robot_start, 0.0),
            x_lb=0.0,
            x_ub=puck_map.width,
            y_lb=0.0,
            y_ub=puck_map.height,
        )
        self.world = PuckWorld(puck_map, config.physics)
        super().__init__(config=config, **kwargs)
        client = self.physics_client_id
        phys = config.physics
        p.setGravity(0, 0, 0, physicsClientId=client)
        p.setTimeStep(config.physics_dt, physicsClientId=client)
        # Without these, Bullet drops the restitution of slow or early contacts.
        p.setPhysicsEngineParameter(
            restitutionVelocityThreshold=0.0, physicsClientId=client
        )

        # Floor patches, holes and the goal are paint: visual only.
        for surface, (x0, y0, x1, y1) in puck_map.surface_rects():
            _visual_box(
                client,
                ((x1 - x0) / 2, (y1 - y0) / 2, 0.005),
                ((x0 + x1) / 2, (y0 + y1) / 2, -0.005),
                (*SURFACE_RGB[surface], 1.0),
            )
        self._hole_ids = [
            _visual_disc(client, phys.hole_radius, (hx, hy, 0.001), (*HOLE_RGB, 1.0))
            for hx, hy in self.world.holes
        ]
        self._goal_id = _visual_disc(
            client, phys.goal_radius, (*self.world.goal, 0.0005), (*GOAL_RGB, 1.0)
        )

        self._walls: dict[str, int] = {}
        self._wall_half_extents: dict[str, tuple[float, float, float]] = {}
        for i, (x0, y0, x1, y1) in enumerate(puck_map.wall_rects()):
            half = ((x1 - x0) / 2, (y1 - y0) / 2, config.wall_height / 2)
            collision = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=half, physicsClientId=client
            )
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half,
                rgbaColor=(*WALL_RGB, 1.0),
                physicsClientId=client,
            )
            wall = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=((x0 + x1) / 2, (y0 + y1) / 2, half[2]),
                physicsClientId=client,
            )
            # Bullet multiplies the two restitutions, so the puck's is the wall's.
            p.changeDynamics(
                wall,
                -1,
                restitution=1.0,
                lateralFriction=0.0,
                contactProcessingThreshold=0.0,
                physicsClientId=client,
            )
            self._walls[f"wall{i}"] = wall
            self._wall_half_extents[f"wall{i}"] = half

        collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=phys.puck_radius,
            height=phys.puck_height,
            physicsClientId=client,
        )
        visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=phys.puck_radius,
            length=phys.puck_height,
            rgbaColor=(*PUCK_RGB, 1.0),
            physicsClientId=client,
        )
        self.puck_id = p.createMultiBody(
            baseMass=phys.puck_mass,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=(*puck_map.puck_start, self.puck_z),
            physicsClientId=client,
        )
        p.changeDynamics(
            self.puck_id,
            -1,
            restitution=phys.wall_restitution + config.bullet_restitution_offset,
            lateralFriction=0.0,
            spinningFriction=0.0,
            rollingFriction=0.0,
            linearDamping=0.0,
            angularDamping=0.0,
            contactProcessingThreshold=0.0,
            # Bullet puts cylinder contacts on the rim; a huge inertia keeps those
            # impulses from tipping the puck, which stays flat with its spin zeroed.
            localInertiaDiagonal=(1e3, 1e3, 1e3),
            physicsClientId=client,
        )

        paddle_half = (
            config.paddle_thickness / 2,
            phys.paddle_width / 2,
            config.paddle_height / 2,
        )
        collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=paddle_half, physicsClientId=client
        )
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=paddle_half,
            rgbaColor=(0.2, 0.2, 0.8, 1.0),
            physicsClientId=client,
        )
        self.paddle_id = p.createMultiBody(
            baseMass=config.paddle_mass,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            physicsClientId=client,
        )
        p.changeDynamics(
            self.paddle_id,
            -1,
            restitution=0.0,
            lateralFriction=0.0,
            linearDamping=0.0,
            angularDamping=0.0,
            contactProcessingThreshold=0.0,
            physicsClientId=client,
        )

        # The paddle only touches the puck; the robot is clear of the puck once it
        # lets go, and the walls are cleared by the feasibility check.
        robot_bodies = [self.robot.base.robot_id, self.robot.arm.robot_id]
        for body in robot_bodies + [self.end_effector_viz_id]:
            for link in range(-1, p.getNumJoints(body, physicsClientId=client)):
                for other in (self.puck_id, self.paddle_id):
                    p.setCollisionFilterPair(
                        body, other, link, -1, 0, physicsClientId=client
                    )
        for wall in self._walls.values():
            p.setCollisionFilterPair(
                wall, self.paddle_id, -1, -1, 0, physicsClientId=client
            )

        self._frictions: tuple[float, ...] = tuple(
            (lo + hi) / 2 for lo, hi in config.friction_prior
        )
        self._outcome: str | None = None  # "goal", "hole" or None
        self._last_info: dict[str, Any] = {}

    @property
    def puck_z(self) -> float:
        """Height of the puck's center, resting on the floor."""
        return self.config.physics.puck_height / 2 + 1e-3

    @property
    def frictions(self) -> tuple[float, ...]:
        """The true (hidden) friction coefficient per surface.

        For evaluation; planners must not read it.
        """
        return self._frictions

    # ------------------------------------------------------------------ spaces

    def _create_action_space(  # type: ignore[override]
        self, config: PuckPushing3DEnvConfig
    ) -> PushActionSpace:
        return PushActionSpace(config.physics.max_speed)

    @property
    def state_cls(self) -> TypingType[Kinematic3DObjectCentricState]:
        return PuckPushing3DObjectCentricState

    # ------------------------------------------------------------------ objects

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        return self._create_state_dict(
            [(name, Kinematic3DCuboidType) for name in self._walls]
            + [("goal", Kinematic3DPointType)]
            + [(f"hole{i}", Kinematic3DPointType) for i in range(len(self._hole_ids))]
        )

    def _get_obs(self) -> PuckPushing3DObjectCentricState:
        state_dict = self._create_state_dict(
            [("robot", Kinematic3DRobotType), ("puck", Kinematic3DCuboidType)]
        )
        state = create_state_from_dict(
            state_dict,
            Kinematic3DEnvTypeFeatures,
            state_cls=PuckPushing3DObjectCentricState,
        )
        assert isinstance(state, PuckPushing3DObjectCentricState)
        return state

    def goal_reached(self) -> bool:
        return self._outcome == "goal"

    def _reset_objects(self) -> None:
        start = self.world.puck_map.puck_start
        set_pose(self.puck_id, Pose((*start, self.puck_z)), self.physics_client_id)
        self._stop_puck()
        self._place_paddle()

    def _set_object_states(self, obs: PuckPushing3DObjectCentricState) -> None:
        pose = obs.get_object_pose("puck")
        position = (pose.position[0], pose.position[1], self.puck_z)
        set_pose(self.puck_id, Pose(position), self.physics_client_id)
        self._stop_puck()
        self._place_paddle()

    def _object_name_to_pybullet_id(self, object_name: str) -> int:
        if object_name == "puck":
            return self.puck_id
        if object_name == "goal":
            return self._goal_id
        if object_name.startswith("hole"):
            return self._hole_ids[int(object_name[len("hole") :])]
        if object_name in self._walls:
            return self._walls[object_name]
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_collision_object_ids(self) -> set[int]:
        return set(self._walls.values())

    def _get_movable_object_names(self) -> set[str]:
        return set()

    def _get_surface_object_names(self) -> set[str]:
        return set()

    def _get_half_extents(self, object_name: str) -> tuple[float, float, float]:
        if object_name == "puck":
            phys = self.config.physics
            return (phys.puck_radius, phys.puck_radius, phys.puck_height / 2)
        if object_name in self._wall_half_extents:
            return self._wall_half_extents[object_name]
        raise ValueError(f"Unrecognized object name: {object_name}")

    # ------------------------------------------------------------------ physics

    def _puck_position(self) -> NDArray[np.float64]:
        position = get_pose(self.puck_id, self.physics_client_id).position
        return np.array(position[:2])

    def _puck_velocity(self) -> NDArray[np.float64]:
        linear, _ = p.getBaseVelocity(
            self.puck_id, physicsClientId=self.physics_client_id
        )
        return np.array(linear[:2])

    def _set_puck_velocity(self, velocity: NDArray[np.float64]) -> None:
        # Pinning z and spin keeps the puck flat on the floor without gravity.
        p.resetBaseVelocity(
            self.puck_id,
            (float(velocity[0]), float(velocity[1]), 0.0),
            (0.0, 0.0, 0.0),
            physicsClientId=self.physics_client_id,
        )

    def _stop_puck(self) -> None:
        self._set_puck_velocity(np.zeros(2))

    def _apply_floor_friction(self) -> None:
        """Coulomb friction from the patch under the puck's center, for one step."""
        velocity = self._puck_velocity()
        speed = float(np.linalg.norm(velocity))
        if speed <= 0.0:
            self._set_puck_velocity(velocity)
            return
        mu = self.world.friction_at(self._puck_position(), self._frictions)
        decel = mu * GRAVITY * self.config.physics_dt
        if speed <= decel:
            self._stop_puck()
        else:
            self._set_puck_velocity(velocity * (1.0 - decel / speed))

    def _paddle_pose(self, face: NDArray[np.float64], angle: float) -> Pose:
        """Paddle pose with the middle of its face at `face`, facing `angle`."""
        config = self.config
        back = config.paddle_thickness / 2
        x = face[0] - back * math.cos(angle)
        y = face[1] - back * math.sin(angle)
        z = config.paddle_height / 2 + 2e-3
        orientation = p.getQuaternionFromEuler((0.0, 0.0, angle))
        return Pose((x, y, z), orientation)

    def _place_paddle(
        self,
        face: NDArray[np.float64] | None = None,
        angle: float | None = None,
    ) -> None:
        """Put the paddle face at `face`, facing `angle`; by default in front of the
        robot."""
        base = self.robot.get_base()
        if face is None:
            reach = self.config.physics.paddle_offset
            face = np.array([base.x, base.y]) + reach * np.array(
                [math.cos(base.rot), math.sin(base.rot)]
            )
        pose = self._paddle_pose(face, base.rot if angle is None else angle)
        set_pose(self.paddle_id, pose, self.physics_client_id)
        p.resetBaseVelocity(
            self.paddle_id, (0, 0, 0), (0, 0, 0), physicsClientId=self.physics_client_id
        )

    def _set_paddle_puck_contact(self, enabled: bool) -> None:
        p.setCollisionFilterPair(
            self.paddle_id,
            self.puck_id,
            -1,
            -1,
            int(enabled),
            physicsClientId=self.physics_client_id,
        )

    # ------------------------------------------------------------------ reset/step

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[PuckPushing3DObjectCentricState, dict]:
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
        self._set_paddle_puck_contact(True)
        return obs, info

    def step(
        self, action: Array
    ) -> tuple[PuckPushing3DObjectCentricState, float, bool, bool, dict]:
        assert self._outcome is None, "Episode is over; call reset()"
        angle, speed = float(action[0]), float(action[1])
        assert self.action_space.contains(np.array([angle, speed]))
        puck = self._puck_position()
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
        if info["fell_in_hole"]:
            self._outcome = "hole"
        elif self.world.in_goal(self._puck_position()):
            self._outcome = "goal"
        info["goal_reached"] = self._outcome == "goal"
        self._last_info = info
        return self._get_obs(), -1.0, self._outcome is not None, False, info

    def _strike(self, puck: NDArray[np.float64], angle: float, speed: float) -> dict:
        """Drive behind the puck, push it to speed with the paddle, let it slide."""
        config = self.config
        phys = config.physics
        client = self.physics_client_id
        rng = self.np_random
        # The robot aims and accelerates imperfectly.
        actual_angle = angle + float(rng.normal(0.0, phys.angle_noise))
        actual_speed = max(
            speed * (1.0 + float(rng.normal(0.0, phys.speed_noise))), 0.0
        )
        direction = np.array([math.cos(actual_angle), math.sin(actual_angle)])

        # The commanded push is feasible; a noisy heading may not be, and then the
        # robot stands as for the commanded one.
        pose = self.world.push_pose(puck, actual_angle) or self.world.push_pose(
            puck, angle
        )
        assert pose is not None
        station, heading = pose
        start = self.robot.get_base()
        drive = self.world.drive_distance(
            self.world.nav_node_of(np.array([start.x, start.y])), station
        )
        base_xy = station - _PADDLE_GAP * direction
        self.robot.set_base(SE2Pose(base_xy[0], base_xy[1], heading))
        face = puck - (phys.puck_radius + _PADDLE_GAP) * direction
        self._place_paddle(face, actual_angle)
        self._set_paddle_puck_contact(True)
        paddle_start = np.array(get_pose(self.paddle_id, client).position[:2])
        base_from_paddle = base_xy - paddle_start

        dt = config.physics_dt
        trajectory = [puck.copy()]
        fell = False
        release_speed = 0.0
        if actual_speed > 0:
            accel = actual_speed**2 / (2.0 * phys.accel_distance)
            num_accel = max(int(round(actual_speed / accel / dt)), 1)
            for k in range(num_accel):
                v = accel * (k + 1) * dt * direction
                p.resetBaseVelocity(
                    self.paddle_id,
                    (float(v[0]), float(v[1]), 0.0),
                    (0.0, 0.0, 0.0),
                    physicsClientId=client,
                )
                p.stepSimulation(physicsClientId=client)
                self._apply_floor_friction()
                # The robot follows its paddle.
                paddle_xy = np.array(get_pose(self.paddle_id, client).position[:2])
                base_xy = paddle_xy + base_from_paddle
                self.robot.set_base(SE2Pose(base_xy[0], base_xy[1], heading))
                fell = fell or self.world.over_hole(self._puck_position())
                trajectory.append(self._puck_position())
            release_speed = float(np.linalg.norm(self._puck_velocity()))
        # Let go: stop the paddle, and keep it out of the puck's way.
        self._place_paddle()
        self._set_paddle_puck_contact(False)
        for _ in range(int(config.max_slide_seconds / dt)):
            if fell or np.linalg.norm(self._puck_velocity()) < config.stop_speed:
                break
            p.stepSimulation(physicsClientId=client)
            self._apply_floor_friction()
            p.resetBaseVelocity(
                self.paddle_id, (0, 0, 0), (0, 0, 0), physicsClientId=client
            )
            fell = self.world.over_hole(self._puck_position())
            trajectory.append(self._puck_position())
        self._stop_puck()
        stop = self._puck_position()
        set_pose(self.puck_id, Pose((stop[0], stop[1], self.puck_z)), client)
        self._place_paddle()
        self._set_paddle_puck_contact(True)
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

    def _measure_position(self, point: NDArray[np.float64]) -> NDArray[np.float64]:
        noise = self.config.physics.position_obs_noise
        return point + self.np_random.normal(0.0, noise, size=2)


def _visual_box(
    client: int,
    half_extents: tuple[float, float, float],
    position: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
) -> int:
    visual = p.createVisualShape(
        p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba, physicsClientId=client
    )
    return p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual,
        basePosition=position,
        physicsClientId=client,
    )


def _visual_disc(
    client: int,
    radius: float,
    position: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
) -> int:
    visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=radius,
        length=0.002,
        rgbaColor=rgba,
        physicsClientId=client,
    )
    return p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual,
        basePosition=position,
        physicsClientId=client,
    )


class PuckPushing3DEnv(ConstantObjectKinDEREnv):
    """PuckPushing3D env with a constant set of objects."""

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricKinematic3DRobotEnv:
        return ObjectCentricPuckPushing3DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        return ["robot", "puck"]

    def _create_env_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return """A 3D environment where a TidyBot pushes a puck through a floor maze to
               a goal with a paddle.

               The floor is tiled with three surfaces (laminate, wood, felt), each with
               a hidden sliding friction coefficient. Each action is one push: the robot
               drives behind the puck and strikes it at the commanded heading and speed,
               with small execution noise. The puck slides until friction stops it,
               bouncing off walls. The episode fails if the puck's center passes over a
               hole (red circle on the floor).

               The info dict reports the noisy release speed and stop position, the
               driving distance, and whether the push was feasible.
               """

    def _create_variant_markdown_description(self) -> str:
        return "Variants differ in the maze."

    def _create_variant_specific_description(self) -> str:
        env = self._object_centric_env
        assert isinstance(env, ObjectCentricPuckPushing3DEnv)
        return f"This variant uses the {env.world.puck_map.name} maze."

    def _create_reward_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return """A reward of -1 per push.

               The episode terminates when the puck stops inside the goal circle, or
               fails when its center passes over a hole.
               """

    def _create_references_markdown_description(self) -> str:
        return "N/A."
