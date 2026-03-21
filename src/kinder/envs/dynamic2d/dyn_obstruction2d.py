"""Dynamic Obstruction 2D env using PyMunk physics."""

from dataclasses import dataclass

import numpy as np
import pymunk
from relational_structs import Object, ObjectCentricState, Type
from relational_structs.utils import create_state_from_dict

from kinder.core import ConstantObjectKinDEREnv, FinalConfigMeta
from kinder.envs.dynamic2d.base_env import (
    Dynamic2DRobotEnvConfig,
    ObjectCentricDynamic2DRobotEnv,
)
from kinder.envs.dynamic2d.object_types import (
    Dynamic2DRobotEnvTypeFeatures,
    DynRectangleType,
    KinRectangleType,
    KinRobotType,
)
from kinder.envs.dynamic2d.utils import (
    DYNAMIC_COLLISION_TYPE,
    ROBOT_COLLISION_TYPE,
    STATIC_COLLISION_TYPE,
    KinRobotActionSpace,
    create_walls_from_world_boundaries,
)
from kinder.envs.kinematic2d.structs import MultiBody2D, SE2Pose, ZOrder
from kinder.envs.kinematic2d.utils import is_inside, is_on
from kinder.envs.utils import PURPLE, sample_se2_pose, state_2d_has_collision

# Define custom object types for the obstruction environment
TargetBlockType = Type("target_block", parent=DynRectangleType)
TargetSurfaceType = Type("target_surface", parent=KinRectangleType)
Dynamic2DRobotEnvTypeFeatures[TargetBlockType] = list(
    Dynamic2DRobotEnvTypeFeatures[DynRectangleType]
)
Dynamic2DRobotEnvTypeFeatures[TargetSurfaceType] = list(
    Dynamic2DRobotEnvTypeFeatures[KinRectangleType]
)


@dataclass(frozen=True)
class DynObstruction2DEnvConfig(Dynamic2DRobotEnvConfig, metaclass=FinalConfigMeta):
    """Scene config for DynObstruction2DEnv()."""

    # World boundaries. Standard coordinate frame with (0, 0) in bottom left.
    world_min_x: float = 0.0
    world_max_x: float = 1 + np.sqrt(5)  # golden ratio :)
    world_min_y: float = 0.0
    world_max_y: float = 2.0

    # Robot parameters
    init_robot_pos: tuple[float, float] = (0.5, 0.5)
    robot_base_radius: float = 0.24
    robot_arm_length_max: float = 2 * robot_base_radius
    gripper_base_width: float = 0.06
    gripper_base_height: float = 0.32
    finger_gap_max = 0.32
    gripper_finger_width: float = 0.2
    gripper_finger_height: float = 0.06

    # Action space parameters.
    min_dx: float = -5e-2
    max_dx: float = 5e-2
    min_dy: float = -5e-2
    max_dy: float = 5e-2
    min_dtheta: float = -np.pi / 16
    max_dtheta: float = np.pi / 16
    min_darm: float = -1e-1
    max_darm: float = 1e-1
    min_dgripper: float = -0.02
    max_dgripper: float = 0.02

    # Controller parameters
    kp_pos: float = 50.0
    kv_pos: float = 5.0
    kp_rot: float = 50.0
    kv_rot: float = 5.0

    # Robot hyperparameters.
    robot_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(0.2, 0.2, -np.pi / 2),
        SE2Pose(0.8, 0.8, np.pi / 2),
    )

    # Table hyperparameters.
    table_rgb: tuple[float, float, float] = (0.75, 0.75, 0.75)
    table_height: float = 0.1
    table_width: float = world_max_x - world_min_x
    # The table pose is defined at the center
    table_pose: SE2Pose = SE2Pose(
        world_min_x + table_width / 2, world_min_y + table_height / 2, 0.0
    )

    # Target surface hyperparameters.
    target_surface_rgb: tuple[float, float, float] = PURPLE
    target_surface_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(world_min_x, table_pose.y, 0.0),
        SE2Pose(world_max_x - robot_base_radius, table_pose.y, 0.0),
    )
    target_surface_height: float = table_height
    # This adds to the width of the target block.
    target_surface_width_addition_bounds: tuple[float, float] = (
        robot_base_radius / 5,
        robot_base_radius / 2,
    )

    # Target block hyperparameters.
    target_block_rgb: tuple[float, float, float] = PURPLE
    target_block_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(
            world_min_x + robot_base_radius, table_pose.y + table_height + 1e-6, 0.0
        ),
        SE2Pose(
            world_max_x - robot_base_radius, table_pose.y + table_height + 1e-6, 0.0
        ),
    )
    target_block_height_bounds: tuple[float, float] = (
        robot_base_radius,
        2 * robot_base_radius,
    )
    target_block_width_bounds: tuple[float, float] = (
        gripper_base_height / 2,
        2 * robot_base_radius,
    )
    target_block_mass: float = 1.0

    # Obstruction hyperparameters (DYNAMIC).
    obstruction_rgb: tuple[float, float, float] = (0.75, 0.1, 0.1)
    obstruction_init_pose_bounds = (
        SE2Pose(
            world_min_x + robot_base_radius, table_pose.y + table_height + 1e-6, 0.0
        ),
        SE2Pose(
            world_max_x - robot_base_radius, table_pose.y + table_height + 1e-6, 0.0
        ),
    )
    obstruction_height_bounds: tuple[float, float] = (
        robot_base_radius,
        2 * robot_base_radius,
    )
    obstruction_width_bounds: tuple[float, float] = (
        gripper_base_height / 2,
        2 * robot_base_radius,
    )
    obstruction_block_mass: float = 1.0

    # NOTE: this is not the "real" probability, but rather, the probability
    # that we will attempt to sample the obstruction somewhere on the target
    # surface during each round of rejection sampling during reset().
    obstruction_init_on_target_prob: float = 0.9

    # For sampling initial states.
    max_initial_state_sampling_attempts: int = 10_000

    # For rendering.
    render_dpi: int = 250


class ObjectCentricDynObstruction2DEnv(
    ObjectCentricDynamic2DRobotEnv[DynObstruction2DEnvConfig]
):
    """Dynamic environment where a block must be placed on an obstructed target. Uses
    PyMunk physics simulation.

    Key difference from Kinematic2DEnv is that the robot can interact with dynamic
    objects with realistic physics (friction, collisions, etc). This means some objects
    should be *pushed* instead of *grasped*.
    """

    def __init__(
        self,
        num_obstructions: int = 2,
        config: DynObstruction2DEnvConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(config or DynObstruction2DEnvConfig(), **kwargs)
        self._num_obstructions = num_obstructions

        # Store object references for tracking
        self._target_block: Object | None = None
        self._target_surface: Object | None = None

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        init_state_dict: dict[Object, dict[str, float]] = {}

        # Create the table.
        table = Object("table", KinRectangleType)
        init_state_dict[table] = {
            "x": self.config.table_pose.x,
            "vx": 0.0,
            "y": self.config.table_pose.y,
            "vy": 0.0,
            "theta": self.config.table_pose.theta,
            "omega": 0.0,
            "width": self.config.table_width,
            "height": self.config.table_height,
            "static": True,
            "held": False,
            "color_r": self.config.table_rgb[0],
            "color_g": self.config.table_rgb[1],
            "color_b": self.config.table_rgb[2],
            "z_order": ZOrder.ALL.value,
        }

        # Create room walls.
        assert isinstance(self.action_space, KinRobotActionSpace)
        min_dx, min_dy = self.action_space.low[:2]
        max_dx, max_dy = self.action_space.high[:2]
        wall_state_dict = create_walls_from_world_boundaries(
            self.config.world_min_x,
            self.config.world_max_x,
            self.config.world_min_y,
            self.config.world_max_y,
            min_dx,
            max_dx,
            min_dy,
            max_dy,
        )
        init_state_dict.update(wall_state_dict)

        return init_state_dict

    def _sample_initial_state(self) -> ObjectCentricState:
        """Sample an initial state for the environment."""
        n = self.config.max_initial_state_sampling_attempts
        for _ in range(n):
            # Sample all randomized values.
            robot_pose = sample_se2_pose(
                self.config.robot_init_pose_bounds, self.np_random
            )
            target_block_pose = sample_se2_pose(
                self.config.target_block_init_pose_bounds, self.np_random
            )
            target_block_shape = (
                self.np_random.uniform(*self.config.target_block_width_bounds),
                self.np_random.uniform(*self.config.target_block_height_bounds),
            )
            target_surface_pose = sample_se2_pose(
                self.config.target_surface_init_pose_bounds, self.np_random
            )
            target_surface_width_addition = self.np_random.uniform(
                *self.config.target_surface_width_addition_bounds
            )
            target_surface_shape = (
                target_block_shape[0] + target_surface_width_addition,
                self.config.target_surface_height,
            )

            obstructions: list[tuple[SE2Pose, tuple[float, float]]] = []
            for _ in range(self._num_obstructions):
                obstruction_shape = (
                    self.np_random.uniform(*self.config.obstruction_width_bounds),
                    self.np_random.uniform(*self.config.obstruction_height_bounds),
                )
                obstruction_init_on_target = (
                    self.np_random.uniform()
                    < self.config.obstruction_init_on_target_prob
                )
                if obstruction_init_on_target:
                    old_lb, old_ub = self.config.obstruction_init_pose_bounds
                    new_x_lb = target_surface_pose.x - obstruction_shape[0]
                    new_x_ub = target_surface_pose.x + target_surface_shape[0]
                    new_lb = SE2Pose(new_x_lb, old_lb.y, old_lb.theta)
                    new_ub = SE2Pose(new_x_ub, old_ub.y, old_ub.theta)
                    pose_bounds = (new_lb, new_ub)
                else:
                    pose_bounds = self.config.obstruction_init_pose_bounds
                obstruction_pose = sample_se2_pose(pose_bounds, self.np_random)
                obstructions.append((obstruction_pose, obstruction_shape))

            state = self._create_initial_state(
                robot_pose,
                target_surface_pose,
                target_surface_shape,
                target_block_pose,
                target_block_shape,
                obstructions,
            )

            # Check initial state validity: goal not satisfied and no collisions.
            if self._target_satisfied(state, {}):
                continue
            full_state = state.copy()
            full_state.data.update(self.initial_constant_state.data)
            all_objects = set(full_state)
            # We use Kinematic2D collision checker for now, maybe need to update it.
            if state_2d_has_collision(full_state, all_objects, all_objects, {}):
                continue
            if self._surface_outside_table(full_state, {}):
                continue
            return state

        raise RuntimeError(f"Failed to sample initial state after {n} attempts")

    def _create_initial_state(
        self,
        robot_pose: SE2Pose,
        target_surface_pose: SE2Pose,
        target_surface_shape: tuple[float, float],
        target_block_pose: SE2Pose,
        target_block_shape: tuple[float, float],
        obstructions: list[tuple[SE2Pose, tuple[float, float]]],
    ) -> ObjectCentricState:
        # Shallow copy should be okay because the constant objects should not
        # ever change in this method.
        init_state_dict: dict[Object, dict[str, float]] = {}

        # Create the robot.
        robot = Object("robot", KinRobotType)
        init_state_dict[robot] = {
            "x": robot_pose.x,
            "y": robot_pose.y,
            "theta": robot_pose.theta,
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
            "base_radius": self.config.robot_base_radius,
            "arm_joint": self.config.robot_base_radius,
            "arm_length": self.config.robot_arm_length_max,
            "gripper_base_width": self.config.gripper_base_width,
            "gripper_base_height": self.config.gripper_base_height,
            "finger_gap": self.config.gripper_base_height,
            "finger_height": self.config.gripper_finger_height,
            "finger_width": self.config.gripper_finger_width,
        }

        # Create the target surface.
        target_surface = Object("target_surface", TargetSurfaceType)
        init_state_dict[target_surface] = {
            "x": target_surface_pose.x,
            "vx": 0.0,
            "y": target_surface_pose.y,
            "vy": 0.0,
            "theta": target_surface_pose.theta,
            "omega": 0.0,
            "width": target_surface_shape[0],
            "height": target_surface_shape[1],
            "static": True,
            "held": False,
            "color_r": self.config.target_surface_rgb[0],
            "color_g": self.config.target_surface_rgb[1],
            "color_b": self.config.target_surface_rgb[2],
            "z_order": ZOrder.NONE.value,
        }

        # Create the target block.
        target_block = Object("target_block", TargetBlockType)
        init_state_dict[target_block] = {
            "x": target_block_pose.x,
            "vx": 0.0,
            "y": target_block_pose.y + target_block_shape[1] / 2,
            "vy": 0.0,
            "theta": target_block_pose.theta,
            "omega": 0.0,
            "width": target_block_shape[0],
            "height": target_block_shape[1],
            "static": False,
            "held": False,
            "mass": self.config.target_block_mass,
            "color_r": self.config.target_block_rgb[0],
            "color_g": self.config.target_block_rgb[1],
            "color_b": self.config.target_block_rgb[2],
            "z_order": ZOrder.ALL.value,
        }

        # Create obstructions.
        for i, (obstruction_pose, obstruction_shape) in enumerate(obstructions):
            obstruction = Object(f"obstruction{i}", DynRectangleType)
            init_state_dict[obstruction] = {
                "x": obstruction_pose.x,
                "vx": 0.0,
                "y": obstruction_pose.y + obstruction_shape[1] / 2,
                "vy": 0.0,
                "theta": obstruction_pose.theta,
                "omega": 0.0,
                "mass": self.config.obstruction_block_mass,
                "width": obstruction_shape[0],
                "height": obstruction_shape[1],
                "static": False,
                "held": False,
                "color_r": self.config.obstruction_rgb[0],
                "color_g": self.config.obstruction_rgb[1],
                "color_b": self.config.obstruction_rgb[2],
                "z_order": ZOrder.ALL.value,
            }

        # Finalize state.
        return create_state_from_dict(init_state_dict, Dynamic2DRobotEnvTypeFeatures)

    def _add_state_to_space(self, state: ObjectCentricState) -> None:
        """Add objects from the state to the PyMunk space."""
        assert self.pymunk_space is not None, "Space not initialized"

        # Add static objects (table, walls)
        for obj in state:
            if obj.is_instance(KinRobotType):
                self._reset_robot_in_space(obj, state)
            else:
                # Everything else are rectangles in this environment.
                x = state.get(obj, "x")
                y = state.get(obj, "y")
                width = state.get(obj, "width")
                height = state.get(obj, "height")
                theta = state.get(obj, "theta")
                vx = state.get(obj, "vx")
                vy = state.get(obj, "vy")
                omega = state.get(obj, "omega")
                held = state.get(obj, "held")

                if (
                    (obj.name == "table")
                    or "wall" in obj.name
                    or obj.is_instance(TargetSurfaceType)
                ):
                    # Static objects
                    # We use Pymunk kinematic bodies for static objects
                    b2 = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
                    vs = [
                        (-width / 2, -height / 2),
                        (-width / 2, height / 2),
                        (width / 2, height / 2),
                        (width / 2, -height / 2),
                    ]
                    shape = pymunk.Poly(b2, vs)
                    shape.friction = 1.0
                    shape.density = 1.0
                    shape.mass = 1.0
                    shape.elasticity = 0.99
                    shape.collision_type = STATIC_COLLISION_TYPE
                    self.pymunk_space.add(b2, shape)
                    b2.angle = theta
                    b2.position = x, y
                    self._state_obj_to_pymunk_body[obj] = b2
                else:
                    # Target block and obstructions
                    mass = state.get(obj, "mass")
                    vs = [
                        (-width / 2, -height / 2),
                        (-width / 2, height / 2),
                        (width / 2, height / 2),
                        (width / 2, -height / 2),
                    ]

                    if not held:
                        # Dynamic objects
                        moment = pymunk.moment_for_box(mass, (width, height))
                        body = pymunk.Body()
                        shape = pymunk.Poly(body, vs)
                        shape.friction = 1.0
                        shape.density = 1.0
                        shape.collision_type = DYNAMIC_COLLISION_TYPE
                        shape.mass = mass
                        assert shape.body is not None
                        shape.body.moment = moment
                        shape.body.mass = mass
                        self.pymunk_space.add(body, shape)
                        body.angle = theta
                        body.position = x, y
                        body.velocity = vx, vy
                        body.angular_velocity = omega
                        self._state_obj_to_pymunk_body[obj] = body
                    else:
                        # Held dynamic objects are treated as kinematic
                        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
                        shape = pymunk.Poly(body, vs)
                        shape.friction = 1.0
                        shape.density = 1.0
                        shape.collision_type = ROBOT_COLLISION_TYPE
                        self.pymunk_space.add(body, shape)
                        body.angle = theta
                        body.position = x, y
                        body.velocity = vx, vy
                        body.angular_velocity = omega
                        # Add to robot hand
                        self._state_obj_to_pymunk_body[obj] = body
                        assert self.robot is not None, "Robot not initialized"
                        self.robot.add_to_hand((body, [shape]), mass)

    def _read_state_from_space(self) -> None:
        """Read the current state from the PyMunk space."""
        assert self.pymunk_space is not None, "Space not initialized"
        assert self._current_state is not None, "Current state not initialized"

        state = self._current_state.copy()

        # Update dynamic object positions from PyMunk simulation
        for obj in state:
            if state.get(obj, "static"):
                continue
            if obj.is_instance(KinRobotType):
                # Update robot state from its body
                assert self.robot is not None, "Robot not initialized"
                robot_obj = state.get_objects(KinRobotType)[0]
                state.set(robot_obj, "x", self.robot.base_pose.x)
                state.set(robot_obj, "y", self.robot.base_pose.y)
                state.set(robot_obj, "theta", self.robot.base_pose.theta)
                state.set(robot_obj, "vx_base", self.robot.base_vel[0].x)
                state.set(robot_obj, "vy_base", self.robot.base_vel[0].y)
                state.set(robot_obj, "omega_base", self.robot.base_vel[1])
                state.set(robot_obj, "arm_joint", self.robot.curr_arm_length)
                state.set(robot_obj, "vx_arm", self.robot.gripper_base_vel[0].x)
                state.set(robot_obj, "vy_arm", self.robot.gripper_base_vel[0].y)
                state.set(robot_obj, "omega_arm", self.robot.gripper_base_vel[1])
                state.set(robot_obj, "finger_gap", self.robot.curr_gripper)
                state.set(robot_obj, "vx_gripper_l", self.robot.finger_vel_l[0].x)
                state.set(robot_obj, "vy_gripper_l", self.robot.finger_vel_l[0].y)
                state.set(robot_obj, "omega_gripper_l", self.robot.finger_vel_l[1])
                state.set(robot_obj, "vx_gripper_r", self.robot.finger_vel_r[0].x)
                state.set(robot_obj, "vy_gripper_r", self.robot.finger_vel_r[0].y)
                state.set(robot_obj, "omega_gripper_r", self.robot.finger_vel_r[1])
            else:
                assert (
                    obj in self._state_obj_to_pymunk_body
                ), f"Object {obj.name} not found in pymunk body cache"
                pymunk_body = self._state_obj_to_pymunk_body[obj]
                # Update object state from body
                state.set(obj, "x", pymunk_body.position.x)
                state.set(obj, "y", pymunk_body.position.y)
                state.set(obj, "theta", pymunk_body.angle)
                state.set(obj, "vx", pymunk_body.velocity.x)
                state.set(obj, "vy", pymunk_body.velocity.y)
                state.set(obj, "omega", pymunk_body.angular_velocity)
                # Update held status
                assert self.robot is not None, "Robot not initialized"
                if self.robot.body_in_hand(pymunk_body.id):
                    state.set(obj, "held", True)
                else:
                    state.set(obj, "held", False)

        # Update the current state
        self._current_state = state

    def _target_satisfied(
        self,
        state: ObjectCentricState,
        static_object_body_cache: dict[Object, MultiBody2D],
    ) -> bool:
        """Check if the target condition is satisfied.

        This is borrowed from kinematic2d obstruction env for now.
        """
        target_objects = state.get_objects(TargetBlockType)
        assert len(target_objects) == 1
        target_object = target_objects[0]
        target_surfaces = state.get_objects(TargetSurfaceType)
        assert len(target_surfaces) == 1
        target_surface = target_surfaces[0]
        return is_on(state, target_object, target_surface, static_object_body_cache)

    def _surface_outside_table(
        self,
        state: ObjectCentricState,
        static_object_body_cache: dict[Object, MultiBody2D],
    ) -> bool:
        """Check if the target surface is outside the table boundaries."""
        target_surfaces = state.get_objects(TargetSurfaceType)
        assert len(target_surfaces) == 1
        target_surface = target_surfaces[0]
        table = state.get_objects(KinRectangleType)
        table = [obj for obj in table if obj.name == "table"]
        assert len(table) == 1
        table_instance = table[0]

        is_inside_table = is_inside(
            state,
            target_surface,
            table_instance,
            static_object_body_cache,
        )
        return not is_inside_table

    def _get_reward_and_done(self) -> tuple[float, bool]:
        """Calculate reward and termination."""
        # Terminate when target object is on the target surface. Give -1 reward
        # at every step until then to encourage fast completion.
        assert self._current_state is not None
        terminated = self._target_satisfied(
            self._current_state,
            self._static_object_body_cache,
        )
        return -1.0, terminated


class DynObstruction2DEnv(ConstantObjectKinDEREnv):
    """Dynamic Obstruction 2D env with a constant number of objects."""

    def __init__(self, num_obstructions: int = 2, **kwargs) -> None:
        self._num_obstructions = num_obstructions
        super().__init__(num_obstructions=num_obstructions, **kwargs)

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricDynamic2DRobotEnv:
        return ObjectCentricDynObstruction2DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        constant_objects = ["target_surface", "target_block"]
        for obj in sorted(exemplar_state):
            if obj.name.startswith("obstruct"):
                constant_objects.append(obj.name)
            if obj.name == "robot":
                constant_objects.append(obj.name)
        return constant_objects

    def _create_env_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return """A 2D physics-based environment where the goal is to place a target block onto a target surface using a two-fingered robot with PyMunk physics simulation. The block must be completely on the surface. The target surface may be initially obstructed.

The robot has a movable circular base and an extendable arm with gripper fingers. Objects can be grasped and released through gripper actions. All objects follow realistic physics including gravity, friction, and collisions.

Each object includes physics properties like mass, moment of inertia (for dynamic objects), and color information for rendering.
"""

    def _create_variant_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return "The number of obstructions differs between environment variants. For example, DynObstruction2D-o0 has no obstructions, while DynObstruction2D-o3 has 3 obstructions."

    def _create_variant_specific_description(self) -> str:
        if self._num_obstructions == 0:
            return "This variant has no obstructions."
        if self._num_obstructions == 1:
            return "This variant has 1 obstruction."
        return f"This variant has {self._num_obstructions} obstructions."

    def _create_reward_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return """A penalty of -1.0 is given at every time step until termination, which occurs when the target block is completely "on" the target surface. The "on" condition requires that the bottom vertices of the target block are within the bounds of the target surface.
"""

    def _create_references_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return """This is a dynamic version of Obstruction2D."""
