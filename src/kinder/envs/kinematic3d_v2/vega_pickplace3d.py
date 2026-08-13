"""Environment where a bimanual Dexmate Vega moves a cube to a target surface.

A cube starts somewhere on a table in front of the robot and must end up resting on a
target patch elsewhere on the table. Both arms are actuated and either arm can pick up,
carry, and release the cube, or pass it to the other arm mid-air. The cube and the
target patch are sampled uniformly over the table, with no constraint on which arm can
reach them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium
import numpy as np
import pybullet as p
from numpy.typing import NDArray
from prpl_kinematics.collision import PyBulletCollisionChecker
from prpl_kinematics.geometry.shapes import BoxShape
from prpl_kinematics.robots import Robot
from prpl_kinematics.tree.kinematic_tree import Node
from prpl_kinematics.visualization.pybullet_renderer import PyBulletRenderer
from relational_structs import (
    Object,
    ObjectCentricState,
    ObjectCentricStateSpace,
    Type,
)
from relational_structs.utils import create_state_from_dict
from spatialmath import SE3

from kinder.core import (
    ConstantObjectKinDEREnv,
    FinalConfigMeta,
    ObjectCentricKinDEREnv,
    RobotActionSpace,
)
from kinder.envs.kinematic3d_v2.base_env import ROBOT_FACTORIES, Kinematic3Dv2EnvConfig
from kinder.envs.kinematic3d_v2.object_types import (
    ARM_NUM_JOINTS,
    Kinematic3Dv2EnvTypeFeatures,
    Kinematic3Dv2GraspArmRobotType,
    Kinematic3Dv2PointType,
)

# Kinematic tree node names for the scene geometry.
TABLE_NODE = "pickplace_table"
CUBE_NODE = "cube"
TARGET_NODE = "target"

# The two arms, in the order their joints appear in actions and observations.
ARM_SIDES = ("left", "right")


@dataclass(frozen=True)
class VegaPickPlace3DEnvConfig(Kinematic3Dv2EnvConfig, metaclass=FinalConfigMeta):
    """Config for VegaPickPlace3DEnv()."""

    # Table. A solid block from the floor whose top surface is the work surface.
    # The home configuration must remain collision-free above it.
    table_x_bounds: tuple[float, float] = (0.40, 0.90)
    table_y_bounds: tuple[float, float] = (-0.80, 0.80)
    table_height: float = 0.55
    table_color: tuple[float, float, float, float] = (0.5, 0.35, 0.2, 1.0)

    # Cube.
    cube_half_size: float = 0.03
    cube_color: tuple[float, float, float, float] = (0.2, 0.4, 1.0, 1.0)

    # An arm holds the cube kinematically: a grasp succeeds whenever the arm's end
    # effector is within this distance of the cube center.
    grasp_radius: float = 0.10

    # Target patch, drawn flat on the table top.
    target_half_extents: tuple[float, float] = (0.10, 0.10)
    target_thickness: float = 0.005
    target_color: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 0.7)

    # Cube and target patch centers are sampled uniformly in this xy region, kept
    # inside the table edges, and at least min_cube_target_distance apart.
    sample_x_bounds: tuple[float, float] = (0.45, 0.80)
    sample_y_bounds: tuple[float, float] = (-0.70, 0.70)
    min_cube_target_distance: float = 0.25

    # Rendering: face the robot from across the table.
    camera_target: tuple[float, float, float] = (0.55, 0.0, 0.55)
    camera_distance: float = 2.2
    camera_yaw: float = 90.0
    camera_pitch: float = -35.0


class VegaPickPlace3DObjectCentricState(ObjectCentricState):
    """A state in the VegaPickPlace3DEnv()."""

    def arm(self, side: str) -> Object:
        """The arm object for the given side, "left" or "right"."""
        return self.get_object_from_name(f"{side}_arm")

    def arm_joint_positions(self, side: str) -> list[float]:
        """The joint positions of the given arm."""
        arm = self.arm(side)
        return [self.get(arm, f"joint_{i}") for i in range(1, ARM_NUM_JOINTS + 1)]

    def grasping(self, side: str) -> bool:
        """Whether the given arm is holding the cube."""
        return self.get(self.arm(side), "grasping") > 0.5

    @property
    def holder(self) -> str | None:
        """The side holding the cube, or None if the cube is free."""
        for side in ARM_SIDES:
            if self.grasping(side):
                return side
        return None

    def _position(self, name: str) -> tuple[float, float, float]:
        obj = self.get_object_from_name(name)
        return (self.get(obj, "x"), self.get(obj, "y"), self.get(obj, "z"))

    @property
    def cube_position(self) -> tuple[float, float, float]:
        """The position of the cube center."""
        return self._position(CUBE_NODE)

    @property
    def target_position(self) -> tuple[float, float, float]:
        """The position of the target patch center."""
        return self._position(TARGET_NODE)


class BimanualArmJointDeltaGraspActionSpace(RobotActionSpace):
    """An action space for two arms with ARM_NUM_JOINTS actuated joints each.

    The first 2 * ARM_NUM_JOINTS entries are bounded relative joint positions
    (left arm then right arm). The last two entries are grasp commands for the
    left and right arm: a value above zero asks the arm to hold the cube, a
    value at or below zero asks it to let go.
    """

    def __init__(self, max_magnitude: float = 0.1) -> None:
        low = np.array([-max_magnitude] * (2 * ARM_NUM_JOINTS) + [-1.0, -1.0])
        high = np.array([max_magnitude] * (2 * ARM_NUM_JOINTS) + [1.0, 1.0])
        super().__init__(low, high)

    def create_markdown_description(self) -> str:
        """Create a markdown description with a table of action space entries."""
        rows = []
        for a, side in enumerate(ARM_SIDES):
            for i in range(ARM_NUM_JOINTS):
                rows.append(
                    f"| {a * ARM_NUM_JOINTS + i} | delta {side} joint {i + 1} |"
                )
        for i, side in enumerate(ARM_SIDES):
            rows.append(f"| {2 * ARM_NUM_JOINTS + i} | {side} grasp command |")
        table = "\n".join(rows)
        return f"""An action space for two arms with {ARM_NUM_JOINTS} actuated joints each.

The first {2 * ARM_NUM_JOINTS} entries are bounded relative joint positions, in radians.
The last two entries are grasp commands: above zero asks the arm to hold the cube,
at or below zero asks it to let go. A grasp only succeeds while the arm's end effector
is close to the cube.

| **Index** | **Description** |
| --- | --- |
{table}

Joint deltas are clipped to the configured maximum magnitude and the resulting joint
positions are clipped to the robot's joint limits. A motion that would put an arm in
collision is rejected and that arm stays where it was.
"""


class ObjectCentricVegaPickPlace3DEnv(
    ObjectCentricKinDEREnv[
        VegaPickPlace3DObjectCentricState, NDArray[Any], VegaPickPlace3DEnvConfig
    ]
):
    """Environment where a bimanual Vega moves a cube to a target surface.

    Both arms are actuated. The cube is held kinematically: while grasped it is a child
    of the holding arm's end-effector frame in the kinematic tree, and a handover re-
    parents it from one gripper frame to the other.
    """

    def __init__(
        self,
        config: VegaPickPlace3DEnvConfig = VegaPickPlace3DEnvConfig(),
        use_gui: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self.use_gui = use_gui

        # Create the robot and address both arms.
        self.robot: Robot = ROBOT_FACTORIES[self.config.robot_name]()
        self.tree = self.robot.tree
        self._manipulators = {side: self.robot.manipulators[side] for side in ARM_SIDES}
        self._arm_spaces = {
            side: self.robot.groups[self._manipulators[side].group]
            for side in ARM_SIDES
        }
        for side, space in self._arm_spaces.items():
            assert space.dimension == ARM_NUM_JOINTS, (
                f"Manipulator {side} of {self.config.robot_name} has "
                f"{space.dimension} joints; expected {ARM_NUM_JOINTS}."
            )
        self._joint_limits = {
            side: space.bounds() for side, space in self._arm_spaces.items()
        }
        self._home_joint_positions = {
            side: space.to_vector(self.robot.home)
            for side, space in self._arm_spaces.items()
        }

        # The configuration of every joint. Non-arm joints keep their home values
        # for the lifetime of the environment.
        self._configuration: dict[str, list[float]] = {
            name: list(values) for name, values in self.robot.home.items()
        }

        # The side currently holding the cube, or None.
        self._holder: str | None = None

        # Scene geometry must be added before the renderer and the collision
        # checker load the tree, because each creates one PyBullet body per shape
        # at load time and never revisits the tree's node set afterwards.
        self._create_scene_geometry()

        # Create the renderer.
        self.physics_client_id = p.connect(p.GUI if use_gui else p.DIRECT)
        self._renderer = PyBulletRenderer(self.physics_client_id)
        self._renderer.load(self.tree)

        # Create the collision checker. It gets its own client because its bodies
        # carry collision shapes but no visual ones, which PyBullet would
        # otherwise draw into the rendered image.
        self._collision_client_id = p.connect(p.DIRECT)
        self._collision_checker = PyBulletCollisionChecker(self._collision_client_id)
        self._collision_checker.load(self.tree)
        self._collision_checker.ignore(self.robot.allowed_collision_pairs)

    def _create_scene_geometry(self) -> None:
        config = self.config
        # The table is a solid block from the floor. It has collision geometry, so
        # arm motions that would sweep through it are rejected.
        table_size = (
            config.table_x_bounds[1] - config.table_x_bounds[0],
            config.table_y_bounds[1] - config.table_y_bounds[0],
            config.table_height,
        )
        table_shape = BoxShape(size=table_size, color=config.table_color)
        self.tree.add_node(
            Node(TABLE_NODE, visuals=[table_shape], collisions=[table_shape])
        )
        table_center = (
            (config.table_x_bounds[0] + config.table_x_bounds[1]) / 2,
            (config.table_y_bounds[0] + config.table_y_bounds[1]) / 2,
            config.table_height / 2,
        )
        self.tree.attach(TABLE_NODE, self.tree.root, SE3(*table_center))

        # The cube is visual-only: it is held kinematically rather than through
        # contact, so collision geometry would only obstruct the grasp.
        cube_size = (config.cube_half_size * 2,) * 3
        self.tree.add_node(
            Node(
                CUBE_NODE,
                visuals=[BoxShape(size=cube_size, color=config.cube_color)],
            )
        )
        self.tree.attach(CUBE_NODE, self.tree.root, SE3(0.0, 0.0, 0.0))

        # The target patch marks the goal region on the table top; it is a
        # visual-only marker like the cube.
        patch_size = (
            config.target_half_extents[0] * 2,
            config.target_half_extents[1] * 2,
            config.target_thickness,
        )
        self.tree.add_node(
            Node(
                TARGET_NODE,
                visuals=[BoxShape(size=patch_size, color=config.target_color)],
            )
        )
        self.tree.attach(TARGET_NODE, self.tree.root, SE3(0.0, 0.0, 0.0))

    @property
    def configuration(self) -> dict[str, list[float]]:
        """The current configuration of every joint in the tree."""
        return self._configuration

    @property
    def holder(self) -> str | None:
        """The side currently holding the cube, or None."""
        return self._holder

    def arm_joint_positions(self, side: str) -> NDArray[np.float64]:
        """The positions of the actuated joints of the given arm."""
        return self._arm_spaces[side].to_vector(self._configuration)

    def set_arm_joint_positions(
        self, side: str, positions: NDArray[np.float64]
    ) -> None:
        """Move the actuated joints of the given arm to the given positions."""
        self._configuration.update(self._arm_spaces[side].to_configuration(positions))

    def end_effector_pose(self, side: str) -> SE3:
        """The world-frame pose of the given arm's end effector."""
        return self.tree.forward_kinematics(
            self._manipulators[side].ee_frame, self._configuration
        )

    def _node_position(self, node: str) -> NDArray[np.float64]:
        return self.tree.forward_kinematics(node, self._configuration).t

    @property
    def cube_resting_z(self) -> float:
        """The height of the cube center when the cube rests on the table."""
        return self.config.table_height + self.config.cube_half_size

    def _set_target_position(self, position: tuple[float, float, float]) -> None:
        self.tree.attach(TARGET_NODE, self.tree.root, SE3(*position))

    def _place_cube_in_world(self, position: tuple[float, float, float]) -> None:
        """Fix the cube in the world frame at the given position."""
        self.tree.attach(CUBE_NODE, self.tree.root, SE3(*position))
        self._holder = None

    def _attach_cube_to_arm(self, side: str) -> None:
        """Re-parent the cube onto the given arm's end-effector frame in place."""
        relative = self.end_effector_pose(side).inv() * SE3(
            *self._node_position(CUBE_NODE)
        )
        self.tree.attach(CUBE_NODE, self._manipulators[side].ee_frame, relative)
        self._holder = side

    def _drop_cube(self) -> None:
        """Release the cube and let it fall straight down onto its support.

        The cube lands on the table top when it is released over the table, and on the
        floor otherwise.
        """
        x, y, z = self._node_position(CUBE_NODE)
        config = self.config
        over_table = (
            config.table_x_bounds[0] <= x <= config.table_x_bounds[1]
            and config.table_y_bounds[0] <= y <= config.table_y_bounds[1]
            and z >= config.table_height
        )
        rest_z = self.cube_resting_z if over_table else config.cube_half_size
        self._place_cube_in_world((float(x), float(y), rest_z))

    def _update_grasp(self, wants_grasp: dict[str, bool]) -> None:
        """Apply the grasp commands after the arms have moved.

        A free cube is taken by the closest requesting arm within grasp range. A held
        cube passes to the other arm when that arm requests it within grasp range (a
        handover), and drops when the holder lets go.
        """
        cube = self._node_position(CUBE_NODE)

        def in_range(side: str) -> float | None:
            dist = float(np.linalg.norm(self.end_effector_pose(side).t - cube))
            return dist if dist < self.config.grasp_radius else None

        if self._holder is None:
            candidates = {
                side: dist
                for side in ARM_SIDES
                if wants_grasp[side] and (dist := in_range(side)) is not None
            }
            if candidates:
                self._attach_cube_to_arm(min(candidates, key=candidates.get))
            return
        other = ARM_SIDES[1 - ARM_SIDES.index(self._holder)]
        if wants_grasp[other] and in_range(other) is not None:
            self._attach_cube_to_arm(other)
        elif not wants_grasp[self._holder]:
            self._drop_cube()

    def goal_reached(self) -> bool:
        """The cube rests on the table with its center inside the target patch."""
        if self._holder is not None:
            return False
        cube = self._node_position(CUBE_NODE)
        target = self._node_position(TARGET_NODE)
        half_x, half_y = self.config.target_half_extents
        return (
            abs(cube[0] - target[0]) < half_x
            and abs(cube[1] - target[1]) < half_y
            and abs(cube[2] - self.cube_resting_z) < 1e-6
        )

    def _reset_objects(self) -> None:
        config = self.config
        low = (config.sample_x_bounds[0], config.sample_y_bounds[0])
        high = (config.sample_x_bounds[1], config.sample_y_bounds[1])
        cube_xy = self.np_random.uniform(low, high)
        self._place_cube_in_world(
            (float(cube_xy[0]), float(cube_xy[1]), self.cube_resting_z)
        )
        for _ in range(10_000):
            target_xy = self.np_random.uniform(low, high)
            if np.linalg.norm(target_xy - cube_xy) >= config.min_cube_target_distance:
                break
        else:
            raise RuntimeError("Failed to sample a target away from the cube")
        self._set_target_position(
            (
                float(target_xy[0]),
                float(target_xy[1]),
                config.table_height + config.target_thickness / 2,
            )
        )

    def _get_obs(self) -> VegaPickPlace3DObjectCentricState:
        state_dict: dict[Object, dict[str, float]] = {}
        for side in ARM_SIDES:
            joints = self.arm_joint_positions(side)
            feats = {f"joint_{i + 1}": float(v) for i, v in enumerate(joints)}
            feats["grasping"] = 1.0 if self._holder == side else 0.0
            state_dict[Object(f"{side}_arm", Kinematic3Dv2GraspArmRobotType)] = feats
        for node in (CUBE_NODE, TARGET_NODE):
            position = self._node_position(node)
            state_dict[Object(node, Kinematic3Dv2PointType)] = {
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]),
            }
        state = create_state_from_dict(
            state_dict,
            Kinematic3Dv2EnvTypeFeatures,
            state_cls=VegaPickPlace3DObjectCentricState,
        )
        assert isinstance(state, VegaPickPlace3DObjectCentricState)
        return state

    @property
    def type_features(self) -> dict[Type, list[str]]:
        """The types and features for this environment."""
        return Kinematic3Dv2EnvTypeFeatures

    def _create_observation_space(
        self, config: VegaPickPlace3DEnvConfig
    ) -> ObjectCentricStateSpace:
        del config  # the observation space is the same for every config
        types = set(self.type_features)
        return ObjectCentricStateSpace(
            types, state_cls=VegaPickPlace3DObjectCentricState
        )

    def _create_action_space(
        self, config: VegaPickPlace3DEnvConfig
    ) -> RobotActionSpace:
        return BimanualArmJointDeltaGraspActionSpace(
            max_magnitude=config.max_action_mag
        )

    def _create_constant_initial_state(self) -> VegaPickPlace3DObjectCentricState:
        # Nothing in this environment is constant across episodes.
        state = create_state_from_dict(
            {},
            Kinematic3Dv2EnvTypeFeatures,
            state_cls=VegaPickPlace3DObjectCentricState,
        )
        assert isinstance(state, VegaPickPlace3DObjectCentricState)
        return state

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[VegaPickPlace3DObjectCentricState, dict]:
        # Reset the random seed.
        gymnasium.Env.reset(self, seed=seed)

        # For testing purposes, the options may specify an initial state.
        if options is not None and "init_state" in options:
            self._set_state(options["init_state"])
        else:
            for side in ARM_SIDES:
                self.set_arm_joint_positions(side, self._home_joint_positions[side])
            self._reset_objects()

        return self._get_obs(), {}

    def step(
        self, action: NDArray[Any]
    ) -> tuple[VegaPickPlace3DObjectCentricState, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float64)

        # Move each arm independently: clip the deltas and the resulting joint
        # positions, then revert any arm whose motion would cause a collision.
        # The cube moves with the holding arm through the tree attachment.
        for i, side in enumerate(ARM_SIDES):
            delta = np.clip(
                action[i * ARM_NUM_JOINTS : (i + 1) * ARM_NUM_JOINTS],
                -self.config.max_action_mag,
                self.config.max_action_mag,
            )
            current = self.arm_joint_positions(side)
            lower, upper = self._joint_limits[side]
            self.set_arm_joint_positions(side, np.clip(current + delta, lower, upper))
            if self._collision_checker.in_collision(self._configuration):
                self.set_arm_joint_positions(side, current)

        wants_grasp = {
            side: bool(action[2 * ARM_NUM_JOINTS + i] > 0)
            for i, side in enumerate(ARM_SIDES)
        }
        self._update_grasp(wants_grasp)

        terminated = self.goal_reached()
        # Penalize every timestep to encourage reaching the goal quickly.
        reward = -1.0
        return self._get_obs(), reward, terminated, False, {}

    def render(self) -> NDArray[np.uint8]:  # type: ignore
        self._renderer.render(self._configuration)
        return self._renderer.capture_image(self.config.get_camera_params())

    def _get_state(self) -> VegaPickPlace3DObjectCentricState:
        return self._get_obs()

    def _set_state(self, state: VegaPickPlace3DObjectCentricState) -> None:
        """Set the state of the environment to the given one."""
        for side in ARM_SIDES:
            self.set_arm_joint_positions(
                side, np.array(state.arm_joint_positions(side))
            )
        self._set_target_position(state.target_position)
        # Place the cube in the world first so that attaching preserves its
        # world-frame position from the state.
        self._place_cube_in_world(state.cube_position)
        holder = state.holder
        if holder is not None:
            self._attach_cube_to_arm(holder)

    def close(self) -> None:
        for client_id in (self.physics_client_id, self._collision_client_id):
            if p.isConnected(physicsClientId=client_id):
                p.disconnect(physicsClientId=client_id)


class VegaPickPlace3DEnv(ConstantObjectKinDEREnv):
    """Vega pick-and-place 3D env with a constant number of objects."""

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricVegaPickPlace3DEnv:
        return ObjectCentricVegaPickPlace3DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        return ["left_arm", "right_arm", CUBE_NODE, TARGET_NODE]

    def close(self) -> None:
        # Forward to the object-centric env so that its PyBullet clients are released.
        self._object_centric_env.close()

    def _create_env_markdown_description(self) -> str:
        """Create environment description."""
        # pylint: disable=line-too-long
        config = self._object_centric_env.config
        assert isinstance(config, VegaPickPlace3DEnvConfig)
        return f"""A 3D environment where a cube on a table must be moved onto a target surface.

The robot is a bimanual Dexmate Vega 1U. Both 7-degree-of-freedom arms are actuated; the lift, the torso flip, the head, and both grippers are held at their home values. A cube rests on a table in front of the robot and a flat target patch marks a goal region elsewhere on the table. The episode ends when the cube rests on the table with its center inside the patch and neither arm is holding it.

Grasping is kinematic: an arm holds the cube whenever its grasp command is positive and its end effector is within {config.grasp_radius:.2f}m of the cube center. A held cube moves rigidly with the holding arm. The other arm can take the cube from the holder by requesting a grasp within range, so the cube can be passed between the arms. Releasing the cube drops it straight down onto the table (or the floor, if it is released away from the table).

The cube and the target patch positions are sampled uniformly over the table, so depending on the episode the cube and the target may each be reachable by one arm or both. Some episodes are solvable with a single arm; others require carrying the cube to the middle and passing it between the arms.
"""

    def _create_variant_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return "This environment has only one variant."

    def _create_reward_markdown_description(self) -> str:
        """Create reward description."""
        # pylint: disable=line-too-long
        return """The reward structure is simple:
- **-1.0** penalty at every timestep until the goal is reached
- **Termination** occurs when the cube rests on the table with its center inside the target patch and neither arm is holding it

This encourages the robot to deliver the cube as quickly as possible while avoiding infinite episodes.
"""

    def _create_references_markdown_description(self) -> str:
        """Create references description."""
        # pylint: disable=line-too-long
        return """Tabletop pick-and-place with an optional handover is a standard bimanual manipulation setting. The robot is the [Dexmate Vega 1U](https://www.dexmate.ai/), modeled with [prpl_kinematics](https://github.com/Princeton-Robot-Planning-and-Learning/prpl-mono/tree/main/prpl-kinematics)."""
