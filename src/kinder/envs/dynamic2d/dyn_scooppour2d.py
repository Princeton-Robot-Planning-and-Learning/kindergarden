"""Dynamic Scoop-Pour 2D env using PyMunk physics."""

from dataclasses import dataclass

import numpy as np
import pymunk
from relational_structs import Object, ObjectCentricState, Type
from relational_structs.utils import create_state_from_dict

from kinder.core import ConstantObjectKinDEREnv
from kinder.envs.dynamic2d.base_env import (
    Dynamic2DRobotEnvConfig,
    ObjectCentricDynamic2DRobotEnv,
)
from kinder.envs.dynamic2d.object_types import (
    Dynamic2DRobotEnvTypeFeatures,
    KinRectangleType,
    KinRobotType,
    LObjectType,
    SmallCircleType,
    SmallSquareType,
)
from kinder.envs.dynamic2d.utils import (
    DYNAMIC_COLLISION_TYPE,
    NON_GRASPABLE_COLLISION_TYPE,
    ROBOT_COLLISION_TYPE,
    STATIC_COLLISION_TYPE,
    KinRobotActionSpace,
    create_walls_from_world_boundaries,
)
from kinder.envs.kinematic2d.structs import SE2Pose, ZOrder
from kinder.envs.utils import (
    BLACK,
    BROWN,
    ORANGE,
    PURPLE,
    sample_se2_pose,
    state_2d_has_collision,
)

# Define custom object types for the scoop-pour environment
HookType = Type("hook", parent=LObjectType)
Dynamic2DRobotEnvTypeFeatures[HookType] = list(
    Dynamic2DRobotEnvTypeFeatures[LObjectType]
)


@dataclass(frozen=True)
class DynScoopPour2DEnvConfig(Dynamic2DRobotEnvConfig):
    """Scene config for DynScoopPour2DEnv()."""

    # World boundaries. Standard coordinate frame with (0, 0) in bottom left.
    world_min_x: float = 0.0
    world_max_x: float = 3.5
    world_min_y: float = 0.0
    world_max_y: float = 3.0

    # Robot parameters
    robot_base_radius: float = 0.2
    robot_arm_length_max: float = 2 * robot_base_radius
    gripper_base_width: float = 0.04
    gripper_base_height: float = 0.25
    gripper_finger_width: float = 0.12
    gripper_finger_height: float = 0.04

    # Action space parameters.
    min_dx: float = -3e-2
    max_dx: float = 3e-2
    min_dy: float = -3e-2
    max_dy: float = 3e-2
    min_dtheta: float = -np.pi / 32
    max_dtheta: float = np.pi / 32
    min_darm: float = -8e-2
    max_darm: float = 8e-2
    min_dgripper: float = -0.015
    max_dgripper: float = 0.015

    # Controller parameters
    kp_pos: float = 50.0
    kv_pos: float = 5.0
    kp_rot: float = 50.0
    kv_rot: float = 5.0

    # Robot hyperparameters.
    robot_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(0.5, 2.0, -np.pi),
        SE2Pose(3.0, 2.5, -np.pi / 2),
    )

    # Middle wall hyperparameters (half height of world).
    middle_wall_rgb: tuple[float, float, float] = BLACK
    middle_wall_x: float = (world_min_x + world_max_x) / 2
    middle_wall_y: float = world_max_y / 4  # Base position at quarter height
    middle_wall_width: float = 0.1
    middle_wall_height: float = world_max_y / 2  # Half the world height

    # Hook hyperparameters (L-shaped tool for scooping).
    hook_rgb: tuple[float, float, float] = BROWN
    hook_shape: tuple[float, float, float] = (
        gripper_base_height / 5,  # width
        0.5,  # length_side1 (horizontal bar)
        0.5,  # length_side2 (vertical bar)
    )
    hook_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(
            (middle_wall_x + world_max_x) / 2 + 0.5, gripper_base_height / 5, -np.pi / 2
        ),
        SE2Pose(world_max_x, world_max_y / 2 - 0.8, -np.pi / 2 + 1e-3),
    )
    hook_mass: float = 0.5

    # Small objects hyperparameters.
    small_object_rgb_circle: tuple[float, float, float] = ORANGE
    small_object_rgb_square: tuple[float, float, float] = PURPLE
    small_circle_radius_bounds: tuple[float, float] = (
        gripper_base_height / 4,
        gripper_base_height / 3,
    )
    small_square_size_bounds: tuple[float, float] = (
        gripper_base_height / 3,
        gripper_base_height / 2.8,
    )
    small_object_mass: float = 0.1
    small_object_init_x_bounds: tuple[float, float] = (
        world_min_x + 0.2,
        middle_wall_x - 0.3,
    )
    small_object_init_y_bounds: tuple[float, float] = (
        0.0,
        world_max_y / 2 - 0.2,
    )

    # Success threshold (fraction of small objects on right side).
    # very small for now, just for testing
    success_threshold: float = 0.5

    # For sampling initial states.
    max_initial_state_sampling_attempts: int = 10_000

    # We don't have gravity here, but we have damping.
    gravity_y: float = -1.0  # More realistic slight downward pull
    damping: float = 0.01  # Damping applied to all dynamic bodies
    stabilization_seconds: float = 30.0  # More steps needed

    # For rendering.
    render_dpi: int = 100


class ObjectCentricDynScoopPour2DEnv(
    ObjectCentricDynamic2DRobotEnv[DynScoopPour2DEnvConfig]
):
    """Object-centric dynamic 2D scoop-pour environment.

    The goal is to move small objects from the left side of a middle wall to the right
    side.
    """

    def __init__(
        self,
        num_small_circles: int = 15,
        num_small_squares: int = 15,
        config: DynScoopPour2DEnvConfig = DynScoopPour2DEnvConfig(),
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        self._num_small_circles = num_small_circles
        self._num_small_squares = num_small_squares

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        init_state_dict: dict[Object, dict[str, float]] = {}

        # Create the middle wall (half height of world).
        middle_wall = Object("middle_wall", KinRectangleType)
        init_state_dict[middle_wall] = {
            "x": self.config.middle_wall_x,
            "vx": 0.0,
            "y": self.config.middle_wall_y,
            "vy": 0.0,
            "theta": 0.0,
            "omega": 0.0,
            "width": self.config.middle_wall_width,
            "height": self.config.middle_wall_height,
            "static": True,
            "held": False,
            "color_r": self.config.middle_wall_rgb[0],
            "color_g": self.config.middle_wall_rgb[1],
            "color_b": self.config.middle_wall_rgb[2],
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
        static_objects = set(self.initial_constant_state)
        n = self.config.max_initial_state_sampling_attempts
        robot_pose = sample_se2_pose(self.config.robot_init_pose_bounds, self.np_random)

        # Sample hook pose
        for _ in range(n):
            hook_pose = sample_se2_pose(
                self.config.hook_init_pose_bounds, self.np_random
            )

            # Sample small objects positions on the left side
            small_circles_data = []
            for _ in range(self._num_small_circles):
                x = self.np_random.uniform(*self.config.small_object_init_x_bounds)
                y = self.np_random.uniform(*self.config.small_object_init_y_bounds)
                theta = self.np_random.uniform(-np.pi, np.pi)
                radius = self.np_random.uniform(*self.config.small_circle_radius_bounds)
                small_circles_data.append((SE2Pose(x, y, theta), radius))

            small_squares_data = []
            for _ in range(self._num_small_squares):
                x = self.np_random.uniform(*self.config.small_object_init_x_bounds)
                y = self.np_random.uniform(*self.config.small_object_init_y_bounds)
                theta = self.np_random.uniform(-np.pi, np.pi)
                size = self.np_random.uniform(*self.config.small_square_size_bounds)
                small_squares_data.append((SE2Pose(x, y, theta), size))

            state = self._create_initial_state(
                robot_pose,
                hook_pose=hook_pose,
                small_circles_data=small_circles_data,
                small_squares_data=small_squares_data,
            )

            robot = state.get_objects(KinRobotType)[0]
            hook = state.get_objects(HookType)[0]

            # Check for collisions with the robot and static objects.
            full_state = state.copy()
            full_state.data.update(self.initial_constant_state.data)

            # Don't allow initial collision between robot and hook/static objects
            if not state_2d_has_collision(
                full_state, {robot}, {hook} | static_objects, {}
            ):
                return state

        raise RuntimeError("Failed to sample initial state.")

    def _create_initial_state(
        self,
        robot_pose: SE2Pose,
        hook_pose: SE2Pose | None = None,
        small_circles_data: list[tuple[SE2Pose, float]] | None = None,
        small_squares_data: list[tuple[SE2Pose, float]] | None = None,
    ) -> ObjectCentricState:
        """Create initial state with robot, hook, and small objects."""
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
            "static": False,
            "base_radius": self.config.robot_base_radius,
            "arm_joint": self.config.robot_base_radius,
            "arm_length": self.config.robot_arm_length_max,
            "vx_arm": 0.0,
            "vy_arm": 0.0,
            "omega_arm": 0.0,
            "vx_gripper_l": 0.0,
            "vy_gripper_l": 0.0,
            "omega_gripper_l": 0.0,
            "vx_gripper_r": 0.0,
            "vy_gripper_r": 0.0,
            "omega_gripper_r": 0.0,
            "gripper_base_width": self.config.gripper_base_width,
            "gripper_base_height": self.config.gripper_base_height,
            "finger_gap": self.config.gripper_base_height,
            "finger_height": self.config.gripper_finger_height,
            "finger_width": self.config.gripper_finger_width,
        }

        # Create the hook.
        if hook_pose is not None:
            hook = Object("hook", HookType)
            init_state_dict[hook] = {
                "x": hook_pose.x,
                "vx": 0.0,
                "y": hook_pose.y,
                "vy": 0.0,
                "theta": hook_pose.theta,
                "omega": 0.0,
                "mass": self.config.hook_mass,
                "width": self.config.hook_shape[0],
                "length_side1": self.config.hook_shape[1],
                "length_side2": self.config.hook_shape[2],
                "static": False,
                "held": False,
                "color_r": self.config.hook_rgb[0],
                "color_g": self.config.hook_rgb[1],
                "color_b": self.config.hook_rgb[2],
                "z_order": ZOrder.SURFACE.value,
            }

        # Create small circle objects.
        if small_circles_data:
            for i, (pose, radius) in enumerate(small_circles_data):
                small_circle = Object(f"small_circle{i}", SmallCircleType)
                init_state_dict[small_circle] = {
                    "x": pose.x,
                    "vx": 0.0,
                    "y": pose.y,
                    "vy": 0.0,
                    "theta": pose.theta,
                    "omega": 0.0,
                    "radius": radius,
                    "mass": self.config.small_object_mass,
                    "static": False,
                    "held": False,
                    "color_r": self.config.small_object_rgb_circle[0],
                    "color_g": self.config.small_object_rgb_circle[1],
                    "color_b": self.config.small_object_rgb_circle[2],
                    "z_order": ZOrder.SURFACE.value,
                }

        # Create small square objects.
        if small_squares_data:
            for i, (pose, size) in enumerate(small_squares_data):
                small_square = Object(f"small_square{i}", SmallSquareType)
                init_state_dict[small_square] = {
                    "x": pose.x,
                    "vx": 0.0,
                    "y": pose.y,
                    "vy": 0.0,
                    "theta": pose.theta,
                    "omega": 0.0,
                    "size": size,
                    "mass": self.config.small_object_mass,
                    "static": False,
                    "held": False,
                    "color_r": self.config.small_object_rgb_square[0],
                    "color_g": self.config.small_object_rgb_square[1],
                    "color_b": self.config.small_object_rgb_square[2],
                    "z_order": ZOrder.SURFACE.value,
                }

        # Finalize state.
        return create_state_from_dict(init_state_dict, Dynamic2DRobotEnvTypeFeatures)

    def _add_state_to_space(self, state: ObjectCentricState) -> None:
        """Add objects from the state to the PyMunk space."""
        assert self.pymunk_space is not None, "Space not initialized"

        for obj in state:
            if obj.is_instance(KinRobotType):
                self._reset_robot_in_space(obj, state)
            elif "wall" in obj.name:
                # Static walls
                x = state.get(obj, "x")
                y = state.get(obj, "y")
                width = state.get(obj, "width")
                height = state.get(obj, "height")
                theta = state.get(obj, "theta")

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
            elif obj.is_instance(HookType):
                # L-shaped hook (graspable)
                mass = state.get(obj, "mass")
                x, y = state.get(obj, "x"), state.get(obj, "y")
                theta = state.get(obj, "theta")
                vx, vy = state.get(obj, "vx"), state.get(obj, "vy")
                omega = state.get(obj, "omega")
                held = state.get(obj, "held")
                l1 = state.get(obj, "length_side1")
                l2 = state.get(obj, "length_side2")
                w = state.get(obj, "width")

                # L-shape vertices (same as in dyn_pushpullhook2d.py)
                vertices = [
                    (0, 0),
                    (-l1, 0),
                    (-l1, -w),
                    (-w, -w),
                    (-w, -l2),
                    (0, -l2),
                    (0, -w),
                    (-w, 0),
                ]
                vs_l1 = (vertices[0], vertices[1], vertices[2], vertices[6])
                vs_l2 = (vertices[4], vertices[5], vertices[0], vertices[7])

                moment1 = pymunk.moment_for_poly(mass / 2, vs_l1)
                moment2 = pymunk.moment_for_poly(mass / 2, vs_l2)
                moment = moment1 + moment2

                if not held:
                    # Dynamic hook (can be grasped)
                    body = pymunk.Body(mass=mass, moment=moment)
                    shape1 = pymunk.Poly(body, vs_l1)
                    shape2 = pymunk.Poly(body, vs_l2)
                    shape1.friction = 1.0
                    shape1.density = 1.0
                    shape1.collision_type = DYNAMIC_COLLISION_TYPE
                    shape1.mass = mass / 2
                    shape2.friction = 1.0
                    shape2.density = 1.0
                    shape2.collision_type = DYNAMIC_COLLISION_TYPE
                    shape2.mass = mass / 2
                    self.pymunk_space.add(body, shape1, shape2)
                    body.angle = theta
                    body.position = x, y
                    body.velocity = vx, vy
                    body.angular_velocity = omega
                    self._state_obj_to_pymunk_body[obj] = body
                else:
                    # Held hook (kinematic)
                    body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
                    shape1 = pymunk.Poly(body, vs_l1)
                    shape2 = pymunk.Poly(body, vs_l2)
                    shape1.friction = 1.0
                    shape1.density = 1.0
                    shape1.collision_type = ROBOT_COLLISION_TYPE
                    shape1.mass = mass / 2
                    shape2.friction = 1.0
                    shape2.density = 1.0
                    shape2.collision_type = ROBOT_COLLISION_TYPE
                    shape2.mass = mass / 2
                    self.pymunk_space.add(body, shape1, shape2)
                    body.angle = theta
                    body.position = x, y
                    body.velocity = vx, vy
                    body.angular_velocity = omega
                    self._state_obj_to_pymunk_body[obj] = body
                    assert self.robot is not None, "Robot not initialized"
                    self.robot.add_to_hand((body, [shape1, shape2]), mass)
            elif obj.is_instance(SmallCircleType):
                # Small circle objects (non-graspable)
                mass = state.get(obj, "mass")
                x, y = state.get(obj, "x"), state.get(obj, "y")
                vx, vy = state.get(obj, "vx"), state.get(obj, "vy")
                omega = state.get(obj, "omega")
                radius = state.get(obj, "radius")

                moment = pymunk.moment_for_circle(mass, 0, radius)
                body = pymunk.Body(mass, moment)
                shape = pymunk.Circle(body, radius)  # type: ignore[assignment]
                shape.friction = 1.0
                shape.density = 1.0
                shape.collision_type = NON_GRASPABLE_COLLISION_TYPE
                shape.mass = mass
                self.pymunk_space.add(body, shape)
                body.position = x, y
                body.velocity = vx, vy
                body.angular_velocity = omega
                self._state_obj_to_pymunk_body[obj] = body
            elif obj.is_instance(SmallSquareType):
                # Small square objects (non-graspable)
                mass = state.get(obj, "mass")
                x, y = state.get(obj, "x"), state.get(obj, "y")
                theta = state.get(obj, "theta")
                vx, vy = state.get(obj, "vx"), state.get(obj, "vy")
                omega = state.get(obj, "omega")
                size = state.get(obj, "size")

                vs = [
                    (-size / 2, -size / 2),
                    (-size / 2, size / 2),
                    (size / 2, size / 2),
                    (size / 2, -size / 2),
                ]
                moment = pymunk.moment_for_poly(mass, vs)
                body = pymunk.Body(mass, moment)
                shape = pymunk.Poly(body, vs)
                shape.friction = 1.0
                shape.density = 1.0
                shape.collision_type = NON_GRASPABLE_COLLISION_TYPE
                shape.mass = mass
                self.pymunk_space.add(body, shape)
                body.angle = theta
                body.position = x, y
                body.velocity = vx, vy
                body.angular_velocity = omega
                self._state_obj_to_pymunk_body[obj] = body

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
                # Update robot state
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
                # Update all other dynamic objects
                assert (
                    obj in self._state_obj_to_pymunk_body
                ), f"Object {obj.name} not found in pymunk body cache"
                pymunk_body = self._state_obj_to_pymunk_body[obj]
                state.set(obj, "x", pymunk_body.position.x)
                state.set(obj, "y", pymunk_body.position.y)
                state.set(obj, "vx", pymunk_body.velocity.x)
                state.set(obj, "vy", pymunk_body.velocity.y)
                state.set(obj, "omega", pymunk_body.angular_velocity)

                # Only objects with theta need angle update
                if obj.is_instance(HookType) or obj.is_instance(SmallSquareType):
                    pymunk_angle = pymunk_body.angle
                    if pymunk_angle > np.pi:
                        pymunk_angle -= 2 * np.pi
                    elif pymunk_angle < -np.pi:
                        pymunk_angle += 2 * np.pi
                    state.set(obj, "theta", pymunk_angle)

                # Update held status (only hook can be held)
                if obj.is_instance(HookType):
                    assert self.robot is not None, "Robot not initialized"
                    if self.robot.body_in_hand(pymunk_body.id):
                        state.set(obj, "held", True)
                    else:
                        state.set(obj, "held", False)

        # Update the current state
        self._current_state = state

    def _get_reward_and_done(self) -> tuple[float, bool]:
        """Calculate reward and termination based on object positions.

        Success is achieved when a threshold fraction of small objects are on the right
        side of the middle wall.
        """
        assert self._current_state is not None

        # Count how many small objects are on the right side
        middle_wall_x = self.config.middle_wall_x
        total_objects = 0
        right_side_objects = 0

        for obj in self._current_state:
            if obj.is_instance(SmallCircleType) or obj.is_instance(SmallSquareType):
                total_objects += 1
                obj_x = self._current_state.get(obj, "x")
                obj_vx = self._current_state.get(obj, "vx")
                obj_vy = self._current_state.get(obj, "vy")
                static = (obj_vx**2 + obj_vy**2) < 1e-4
                obj_y = self._current_state.get(obj, "y")
                if (
                    obj_x > middle_wall_x
                    and obj_y < self.config.world_max_y / 4
                    and static
                ):
                    right_side_objects += 1

        # Calculate success
        if total_objects > 0:
            fraction_on_right = right_side_objects / total_objects
            terminated = fraction_on_right >= self.config.success_threshold
        else:
            terminated = False

        return -1.0, terminated


class DynScoopPour2DEnv(ConstantObjectKinDEREnv):
    """Dynamic Scoop-Pour 2D env with a constant number of objects."""

    def __init__(
        self, num_small_circles: int = 10, num_small_squares: int = 0, **kwargs
    ) -> None:
        self._num_small_circles = num_small_circles
        self._num_small_squares = num_small_squares
        super().__init__(
            num_small_circles=num_small_circles,
            num_small_squares=num_small_squares,
            **kwargs,
        )

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricDynScoopPour2DEnv:
        return ObjectCentricDynScoopPour2DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        constant_objects = ["robot", "hook"]
        for obj in sorted(exemplar_state):
            if obj.name.startswith("small_circle"):
                constant_objects.append(obj.name)
            elif obj.name.startswith("small_square"):
                constant_objects.append(obj.name)
        return constant_objects

    def _create_env_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return """A 2D physics-based environment where the goal is to move small objects from the left side of a middle wall to the right side. The middle wall is half the height of the world.

The robot has a movable circular base and an extendable arm with gripper fingers. An L-shaped hook is present as a kinematic object that can be grasped. Small objects are dynamic and follow PyMunk physics, but they cannot be grasped directly by the robot.

All objects include physics properties like mass, moment of inertia, and color information for rendering.
"""

    def _create_variant_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return "The number of small objects differs between environment variants. For example, DynScoopPour2D-o10 has 10 small objects, while DynScoopPour2D-o50 has 50 small objects."

    def _create_variant_specific_description(self) -> str:
        total = self._num_small_circles + self._num_small_squares
        if total == 1:
            return "This variant has 1 small object."
        return (
            f"This variant has {total} small objects "
            f"({self._num_small_circles} circles, {self._num_small_squares} squares)."
        )

    def _create_reward_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return """A penalty of -1.0 is given at every time step until termination, which occurs when at least 50% of the small objects have been moved to the right side of the middle wall."""

    def _create_references_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return """This is loosely inspired by the Kitchen2D environment from "Active model learning and diverse action sampling for task and motion planning" (Wang et al., 2018)."""
