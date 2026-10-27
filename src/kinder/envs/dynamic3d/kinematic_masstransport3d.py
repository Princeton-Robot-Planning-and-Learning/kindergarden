"""MassTransport3D without physics: torque limits enforced by a static check.

This is the fast, kinematic variant of `kinder.envs.dynamic3d.masstransport3d`, which
simulates physics. It also holds what both share: the config, the scene, and the
gravity-torque helpers that planners use to check a load.

Items of unknown mass are carried across in a tote.

Items start on a source table. The goal is to put every item directly on a destination
table. The robot can load several items into a single tote and carry the tote across, so
the task becomes multi-trip bin packing: which items go together in each trip.

Each item has a mass the robot cannot observe. Whenever the robot holds something, the
gravity torques that the arm and its payload put on the joints must stay within the
arm's torque limits. A motion that would exceed them is reverted, as a collision would
be. The info dict reports the measured holding torques, which are the only signal about
the masses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from typing import Type as TypingType

import numpy as np
import pybullet as p
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose, SE2Pose, get_pose, multiply_poses, set_pose
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.utils import create_pybullet_block
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
    Kinematic3DRobotType,
)
from kinder.envs.kinematic3d.utils import (
    Kinematic3DObjectCentricState,
    extend_joints_to_include_fingers,
)

_TEXTURES = Path(__file__).parents[1] / "kinematic3d" / "assets" / "use_textures"

# Number of arm joints whose torques are limited.
NUM_ARM_JOINTS = 7


@dataclass(frozen=True)
class MassTransport3DEnvConfig(Kinematic3DEnvConfig, metaclass=FinalConfigMeta):
    """Config for MassTransport3DEnv()."""

    max_action_mag: float = 0.2

    # The robot starts in front of the source table.
    robot_base_home_pose: SE2Pose = SE2Pose(0.0, -0.8, 0.0)

    # Tables. The source table is at negative y and the destination at positive y.
    # Shallow enough that one base pose in front of a table reaches all of it.
    source_table_pose: Pose = Pose((0.625, -0.8, 0.2))
    dest_table_pose: Pose = Pose((0.625, 0.8, 0.2))
    table_half_extents: tuple[float, float, float] = (0.175, 0.45, 0.2)
    table_rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    table_texture: Path = _TEXTURES / "light_wood_v3.png"

    # Items are cubes.
    item_size: float = 0.05
    item_rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    item_textures: tuple[Path, ...] = (
        _TEXTURES / "yellow-grid.png",
        _TEXTURES / "metal.png",
    )
    # Hidden item masses (kg) are drawn uniformly from this range at reset.
    item_mass_range: tuple[float, float] = (0.4, 1.6)
    # Items start in distinct cells of a grid this far apart, which leaves room for
    # the open fingers between neighbors. Cells near the tote are left out.
    item_grid_step: float = 0.1
    item_tote_clearance: float = 0.03

    # Tote. Its pose is its bounding-box center; the floor is one wall thick.
    tote_half_extents: tuple[float, float, float] = (0.16, 0.2, 0.06)
    tote_wall_thickness: float = 0.01
    tote_mass: float = 0.5
    tote_rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    tote_texture: Path = _TEXTURES / "blue-flower.png"
    # Tote (x, y) on the source table, relative to the table center.
    tote_initial_offset: tuple[float, float] = (0.005, -0.25)
    # The tote rests this far above the table, inside the release tolerance
    # (`min_placement_dist`), so a tote put back the way it was picked is not flush.
    tote_resting_clearance: float = 2.5e-3

    # Holding-torque limits (N m) of the Kinova Gen3 joints, scaled by a factor that
    # sets the task difficulty.
    torque_limits: tuple[float, ...] = (39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0)
    torque_limit_scale: float = 1.0
    gravity: float = 9.81
    # Std of the zero-mean Gaussian noise on the reported torque measurements.
    torque_noise_std: float = 0.0

    # A trip is a base move from the source side (y < 0) to the destination side.
    # The episode is truncated once more trips than this are made; None disables it.
    max_trips: int | None = None

    # Gripper.
    gripper_open_threshold: float = 0.01

    # Physics, used only by the dynamic variant. Each env step runs this many
    # physics steps, with the arm motors capped at the scaled torque limits.
    physics_dt: float = 1.0 / 240.0
    physics_substeps: int = 48
    motor_position_gain: float = 0.3
    motor_velocity_gain: float = 1.0
    # A joint whose commanded torque reaches this fraction of its limit is saturated.
    saturation_fraction: float = 0.99

    # World bounds (used for rendering and sampling).
    x_lb: float = -1.5
    x_ub: float = 1.5
    y_lb: float = -1.5
    y_ub: float = 1.5

    initial_joints: JointPositions = field(
        default_factory=lambda: np.deg2rad([0, -20, 180, -146, 0, -50, 90]).tolist()
    )

    def get_camera_kwargs(self) -> dict[str, Any]:
        """Get kwargs to pass to PyBullet camera."""
        return {
            "camera_target": (0.4, 0.0, 0.3),
            "camera_yaw": 90,
            "camera_distance": 2.6,
            "camera_pitch": -45,
        }

    @property
    def table_top_z(self) -> float:
        """Height of both table tops."""
        return self.source_table_pose.position[2] + self.table_half_extents[2]

    @property
    def tote_initial_pose(self) -> Pose:
        """Where the tote starts, resting on the source table."""
        x, y, _ = self.source_table_pose.position
        dx, dy = self.tote_initial_offset
        z = self.table_top_z + self.tote_half_extents[2] + self.tote_resting_clearance
        return Pose((x + dx, y + dy, z))

    @property
    def scaled_torque_limits(self) -> NDArray[np.float64]:
        """The torque limits after applying the difficulty scale."""
        return self.torque_limit_scale * np.asarray(self.torque_limits, dtype=float)


class MassTransport3DObjectCentricState(Kinematic3DObjectCentricState):
    """A state in the MassTransport3DEnv().

    Item masses are not part of the state: they are what the robot must infer.
    """


class ObjectCentricKinematicMassTransport3DEnv(
    ObjectCentricKinematic3DRobotEnv[
        MassTransport3DObjectCentricState, MassTransport3DEnvConfig
    ]
):
    """Carry items of unknown mass across in a tote; torque limits checked
    statically."""

    def __init__(
        self,
        num_items: int = 8,
        config: MassTransport3DEnvConfig = MassTransport3DEnvConfig(),
        **kwargs,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self._num_items = num_items
        # The kinematic sim never steps physics; gravity is only for inverse dynamics.
        p.setGravity(0, 0, -config.gravity, physicsClientId=self.physics_client_id)

        self._items: dict[str, int] = {}
        for idx in range(num_items):
            half = config.item_size / 2
            item_id = create_pybullet_block(
                config.item_rgba,
                (half, half, half),
                physics_client_id=self.physics_client_id,
            )
            texture = config.item_textures[idx % len(config.item_textures)]
            texture_id = p.loadTexture(str(texture), self.physics_client_id)
            p.changeVisualShape(
                item_id,
                -1,
                textureUniqueId=texture_id,
                physicsClientId=self.physics_client_id,
            )
            self._items[f"item{idx}"] = item_id

        self.tote_id = create_tote_body(
            config.tote_rgba,
            config.tote_half_extents,
            config.tote_wall_thickness,
            mass=config.tote_mass,
            physics_client_id=self.physics_client_id,
        )
        tote_texture_id = p.loadTexture(
            str(config.tote_texture), self.physics_client_id
        )
        p.changeVisualShape(
            self.tote_id,
            -1,
            textureUniqueId=tote_texture_id,
            physicsClientId=self.physics_client_id,
        )

        table_texture_id = p.loadTexture(
            str(config.table_texture), self.physics_client_id
        )
        self.source_table_id = self._create_table(config.source_table_pose)
        self.dest_table_id = self._create_table(config.dest_table_pose)
        for table_id in (self.source_table_id, self.dest_table_id):
            p.changeVisualShape(
                table_id,
                -1,
                textureUniqueId=table_texture_id,
                physicsClientId=self.physics_client_id,
            )

        self._item_masses: dict[str, float] = {}
        self._num_trips = 0
        self._torque_violation = False

    def _create_table(self, pose: Pose) -> int:
        table_id = create_pybullet_block(
            self.config.table_rgba,
            half_extents=self.config.table_half_extents,
            physics_client_id=self.physics_client_id,
        )
        set_pose(table_id, pose, self.physics_client_id)
        return table_id

    @property
    def state_cls(self) -> TypingType[Kinematic3DObjectCentricState]:
        return MassTransport3DObjectCentricState

    @property
    def item_names(self) -> list[str]:
        """Names of the items, in index order."""
        return list(self._items)

    @property
    def item_masses(self) -> dict[str, float]:
        """The true (hidden) item masses.

        For evaluation; planners must not read it.
        """
        return dict(self._item_masses)

    @property
    def num_trips(self) -> int:
        """Trips made since the last reset."""
        return self._num_trips

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[MassTransport3DObjectCentricState, dict]:
        """Reset.

        `options["item_masses"]` may fix the hidden masses.
        """
        obs, info = super().reset(seed=seed, options=options)
        if options is not None and "item_masses" in options:
            masses = dict(options["item_masses"])
            assert set(masses) == set(self._items), "Need one mass per item"
            self._item_masses = {name: float(masses[name]) for name in self._items}
        elif options is None or "init_state" not in options or not self._item_masses:
            low, high = self.config.item_mass_range
            self._item_masses = {
                name: float(self.np_random.uniform(low, high)) for name in self._items
            }
        self._num_trips = 0
        self._torque_violation = False
        return obs, info

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        return self._create_state_dict(
            [
                ("source_table", Kinematic3DCuboidType),
                ("dest_table", Kinematic3DCuboidType),
            ]
        )

    def _reset_objects(self) -> None:
        set_pose(self.tote_id, self.config.tote_initial_pose, self.physics_client_id)
        cells = self.item_grid_cells()
        if len(cells) < self._num_items:
            raise ValueError(
                f"{self._num_items} items do not fit in the {len(cells)} free cells"
            )
        chosen = self.np_random.choice(len(cells), size=self._num_items, replace=False)
        for item_id, index in zip(self._items.values(), chosen):
            set_pose(item_id, cells[index], self.physics_client_id)

    def item_grid_cells(self) -> list[Pose]:
        """Item poses on the grid of free cells on the source table."""
        config = self.config
        half = config.item_size / 2
        x, y, _ = config.source_table_pose.position
        hx, hy, _ = config.table_half_extents
        step = config.item_grid_step
        edge = step / 2
        xs = np.arange(x - hx + edge, x + hx - edge + 1e-9, step)
        ys = np.arange(y - hy + edge, y + hy - edge + 1e-9, step)
        tote_x, tote_y, _ = config.tote_initial_pose.position
        thx, thy, _ = config.tote_half_extents
        reach = half + config.item_tote_clearance
        z = config.table_top_z + half
        return [
            Pose((float(cx), float(cy), z))
            for cx in xs
            for cy in ys
            if abs(cx - tote_x) >= thx + reach or abs(cy - tote_y) >= thy + reach
        ]

    def _set_object_states(self, obs: Kinematic3DObjectCentricState) -> None:
        assert isinstance(obs, MassTransport3DObjectCentricState)
        for name, body_id in self._items.items():
            set_pose(body_id, obs.get_object_pose(name), self.physics_client_id)
        set_pose(self.tote_id, obs.get_object_pose("tote"), self.physics_client_id)

    def _object_name_to_pybullet_id(self, object_name: str) -> int:
        if object_name == "source_table":
            return self.source_table_id
        if object_name == "dest_table":
            return self.dest_table_id
        if object_name == "tote":
            return self.tote_id
        if object_name in self._items:
            return self._items[object_name]
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_collision_object_ids(self) -> set[int]:
        return {self.source_table_id, self.dest_table_id, self.tote_id} | set(
            self._items.values()
        )

    def _get_movable_object_names(self) -> set[str]:
        return set(self._items) | {"tote"}

    def _get_surface_object_names(self) -> set[str]:
        return {"source_table", "dest_table", "tote"}

    def _get_half_extents(self, object_name: str) -> tuple[float, float, float]:
        if object_name in self._items:
            half = self.config.item_size / 2
            return (half, half, half)
        if object_name == "tote":
            return self.config.tote_half_extents
        if object_name in ("source_table", "dest_table"):
            return self.config.table_half_extents
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_obs(self) -> MassTransport3DObjectCentricState:
        state_dict = self._create_state_dict(
            [("robot", Kinematic3DRobotType)]
            + [
                ("source_table", Kinematic3DCuboidType),
                ("dest_table", Kinematic3DCuboidType),
                ("tote", Kinematic3DCuboidType),
            ]
            + [(name, Kinematic3DCuboidType) for name in self._items]
        )
        state = create_state_from_dict(
            state_dict,
            Kinematic3DEnvTypeFeatures,
            state_cls=MassTransport3DObjectCentricState,
        )
        assert isinstance(state, MassTransport3DObjectCentricState)
        return state

    def get_payload(self) -> tuple[float, NDArray[np.float64] | None]:
        """Mass and world center of mass of whatever the robot holds.

        The tote's payload includes the items inside it.
        """
        if self._grasped_object is None:
            return 0.0, None
        names = [self._grasped_object]
        if self._grasped_object == "tote":
            names += sorted(self._get_inside_objects()[0])
        masses, centers = [], []
        for name in names:
            body_id = self._object_name_to_pybullet_id(name)
            position = np.asarray(get_pose(body_id, self.physics_client_id).position)
            if name == "tote":
                masses.append(self.config.tote_mass)
                # The floor is heavier than the open top, but the walls dominate.
                centers.append(position)
            else:
                masses.append(self._item_masses[name])
                centers.append(position)
        total = float(sum(masses))
        com = np.average(np.asarray(centers), axis=0, weights=masses)
        return total, com

    def compute_holding_torques(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Static (arm, payload) gravity torques at the current arm configuration."""
        arm_torques = arm_gravity_torques(self.robot.arm, self.physics_client_id)
        mass, com = self.get_payload()
        if com is None:
            return arm_torques, np.zeros(NUM_ARM_JOINTS)
        payload_torques = point_mass_torques(
            self.robot.arm,
            self.physics_client_id,
            com,
            mass * self.config.gravity,
        )
        return arm_torques, payload_torques

    def holding_torques_exceed_limits(self) -> bool:
        """Whether the current holding torques exceed the scaled limits."""
        arm_torques, payload_torques = self.compute_holding_torques()
        total = arm_torques + payload_torques
        return bool(np.any(np.abs(total) > self.config.scaled_torque_limits))

    def _robot_or_held_object_collision_exists(self) -> bool:
        # A pose the arm cannot hold is rejected like a colliding one.
        if super()._robot_or_held_object_collision_exists():
            return True
        if self._grasped_object is not None and self.holding_torques_exceed_limits():
            self._torque_violation = True
            return True
        return False

    def step(
        self, action: Array
    ) -> tuple[MassTransport3DObjectCentricState, float, bool, bool, dict]:
        self._torque_violation = False
        was_on_source_side = self._on_source_side()
        was_holding = self._grasped_object is not None
        _, reward, terminated, truncated, info = super().step(action)

        # A grasp on something too heavy to hold here slips.
        if (
            not was_holding
            and self._grasped_object is not None
            and self.holding_torques_exceed_limits()
        ):
            self._torque_violation = True
            self._grasped_object = None
            self._grasped_object_transform = None
            self._robot_arm.open_fingers()

        if was_on_source_side and not self._on_source_side():
            self._num_trips += 1
        if (
            self.config.max_trips is not None
            and self._num_trips > self.config.max_trips
        ):
            truncated = True

        info.update(self._measurement_info())
        obs = self._get_obs()
        terminated = self.goal_reached()
        return obs, reward, terminated, truncated, info

    def _on_source_side(self) -> bool:
        return self.robot.get_base().y < 0.0

    def _measurement_info(self) -> dict[str, Any]:
        arm_torques, payload_torques = self.compute_holding_torques()
        noise_std = self.config.torque_noise_std
        if noise_std > 0:
            payload_torques = payload_torques + self.np_random.normal(
                0.0, noise_std, size=payload_torques.shape
            )
        return {
            "arm_torques": arm_torques,
            "payload_torques": payload_torques,
            "torque_violation": self._torque_violation,
            "num_trips": self._num_trips,
        }

    def item_on_dest_table(self, name: str) -> bool:
        """Whether item `name` rests directly on the destination table."""
        if self._grasped_object == name:
            return False
        item_id = self._items[name]
        return self.dest_table_id in self._get_surfaces_supporting_object(item_id)

    def goal_reached(self) -> bool:
        if self._grasped_object is not None:
            return False
        return all(self.item_on_dest_table(name) for name in self._items)


def create_tote_body(
    color: tuple[float, float, float, float],
    half_extents: tuple[float, float, float],
    wall_thickness: float,
    mass: float,
    physics_client_id: int,
) -> int:
    """An open-top box as one rigid body, so physics can move it with its mass.

    Its origin is the center of its bounding box. The kinematic env never steps physics,
    so there the mass only matters to the dynamic subclass.
    """
    hx, hy, hz = half_extents
    half_wall = wall_thickness / 2
    parts = [
        ((hx, hy, half_wall), (0.0, 0.0, -hz + half_wall)),  # floor
        ((half_wall, hy, hz), (-hx + half_wall, 0.0, 0.0)),
        ((half_wall, hy, hz), (hx - half_wall, 0.0, 0.0)),
        ((hx, half_wall, hz), (0.0, -hy + half_wall, 0.0)),
        ((hx, half_wall, hz), (0.0, hy - half_wall, 0.0)),
    ]
    extents = [list(extent) for extent, _ in parts]
    positions = [list(position) for _, position in parts]
    collision_id = p.createCollisionShapeArray(
        shapeTypes=[p.GEOM_BOX] * len(parts),
        halfExtents=extents,
        collisionFramePositions=positions,
        physicsClientId=physics_client_id,
    )
    visual_id = p.createVisualShapeArray(
        shapeTypes=[p.GEOM_BOX] * len(parts),
        halfExtents=extents,
        visualFramePositions=positions,
        rgbaColors=[list(color)] * len(parts),
        physicsClientId=physics_client_id,
    )
    return p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
        physicsClientId=physics_client_id,
    )


def _movable_joint_positions(robot_id: int, physics_client_id: int) -> list[float]:
    """Positions of every non-fixed joint, in joint-index order."""
    positions = []
    for joint in range(p.getNumJoints(robot_id, physicsClientId=physics_client_id)):
        info = p.getJointInfo(robot_id, joint, physicsClientId=physics_client_id)
        if info[2] != p.JOINT_FIXED:
            positions.append(
                p.getJointState(robot_id, joint, physicsClientId=physics_client_id)[0]
            )
    return positions


def arm_gravity_torques(arm, physics_client_id: int) -> NDArray[np.float64]:
    """The torque gravity puts on the arm's own joints, by inverse dynamics at rest.

    The arm base only ever yaws, so world gravity is also gravity in its base frame.
    """
    positions = _movable_joint_positions(arm.robot_id, physics_client_id)
    rest = [0.0] * len(positions)
    torques = p.calculateInverseDynamics(
        arm.robot_id, positions, rest, rest, physicsClientId=physics_client_id
    )
    return np.asarray(torques[:NUM_ARM_JOINTS], dtype=np.float64)


def point_mass_torques(
    arm,
    physics_client_id: int,
    world_point: NDArray[np.float64],
    weight: float,
) -> NDArray[np.float64]:
    """Joint torques that hold a weight (N) hanging at `world_point` off the tool.

    The point is treated as rigidly attached to the tool link.
    """
    positions = _movable_joint_positions(arm.robot_id, physics_client_id)
    rest = [0.0] * len(positions)
    link_state = p.getLinkState(
        arm.robot_id,
        arm.tool_link_id,
        computeForwardKinematics=True,
        physicsClientId=physics_client_id,
    )
    # calculateJacobian wants the point in the link's center-of-mass frame.
    com_frame = Pose(link_state[0], link_state[1])
    local_point = multiply_poses(com_frame.invert(), Pose(tuple(world_point))).position
    linear_jacobian, _ = p.calculateJacobian(
        arm.robot_id,
        arm.tool_link_id,
        list(local_point),
        positions,
        rest,
        rest,
        physicsClientId=physics_client_id,
    )
    jacobian = np.asarray(linear_jacobian)[:, :NUM_ARM_JOINTS]
    return jacobian.T @ np.array([0.0, 0.0, weight])


def holding_torques_at(
    arm,
    physics_client_id: int,
    joints: JointPositions,
    payload_mass: float,
    payload_offset: Pose | None,
    gravity: float = 9.81,
) -> NDArray[np.float64]:
    """Total holding torques with the arm at `joints` (7 or extended).

    `payload_offset` is the payload center of mass in the end-effector frame. The arm is
    left at `joints`.
    """
    if len(joints) == NUM_ARM_JOINTS:
        joints = extend_joints_to_include_fingers(list(joints))
    arm.set_joints(list(joints))
    torques = arm_gravity_torques(arm, physics_client_id)
    if payload_offset is not None and payload_mass > 0:
        com = multiply_poses(arm.get_end_effector_pose(), payload_offset).position
        torques = torques + point_mass_torques(
            arm, physics_client_id, np.asarray(com), payload_mass * gravity
        )
    return torques


class KinematicMassTransport3DEnv(ConstantObjectKinDEREnv):
    """KinematicMassTransport3D env with a constant number of items."""

    def __init__(self, num_items: int = 8, **kwargs) -> None:
        self._num_items = num_items
        super().__init__(num_items=num_items, **kwargs)

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricKinematic3DRobotEnv:
        return ObjectCentricKinematicMassTransport3DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        names = ["robot", "source_table", "dest_table", "tote"]
        names += sorted(
            (o.name for o in exemplar_state if o.name.startswith("item")),
            key=lambda n: int(n[len("item") :]),
        )
        return names

    def _create_env_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return """A kinematic 3D environment (no physics) where the robot must move
               every item from a source table directly onto a destination table.

               A tote can carry several items per trip. Item masses are hidden, and the
               arm cannot hold a pose whose gravity torques exceed its joint torque
               limits, so heavy loads need more trips or a tucked-in carry pose. The
               info dict reports measured holding torques.
               """

    def _create_variant_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return "The number of items differs between variants. For example, KinematicMassTransport3D-o8 has 8 items."

    def _create_variant_specific_description(self) -> str:
        item_str = "1 item" if self._num_items == 1 else f"{self._num_items} items"
        return f"This variant has {item_str} to transport."

    def _create_reward_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return """The reward is a small negative reward (-1) per timestep until
               termination, which occurs when all items rest directly on the destination
               table."""

    def _create_references_markdown_description(self) -> str:
        return """N/A."""
