"""Dynamic HookBlock 2D env using PyMunk physics."""

from dataclasses import dataclass

import numpy as np
import pymunk
from prpl_utils.utils import wrap_angle
from relational_structs import Object, ObjectCentricState, Type
from relational_structs.utils import create_state_from_dict

from kinder.core import ConstantObjectKinDEREnv
from kinder.envs.dynamic2d.base_env import (
    Dynamic2DRobotEnvConfig,
    ObjectCentricDynamic2DRobotEnv,
)
from kinder.envs.dynamic2d.object_types import (
    Dynamic2DRobotEnvTypeFeatures,
    DynRectangleType,
    KinRectangleType,
    KinRobotType,
    LObjectType,
)
from kinder.envs.dynamic2d.utils import (
    ARM_COLLISION_TYPE,
    DYNAMIC_COLLISION_TYPE,
    FINGER_COLLISION_TYPE,
    ROBOT_COLLISION_TYPE,
    STATIC_COLLISION_TYPE,
    KinRobot,
    KinRobotActionSpace,
    create_walls_from_world_boundaries,
    on_collision_w_static,
    on_gripper_grasp,
)
from kinder.envs.kinematic2d.structs import MultiBody2D, SE2Pose, ZOrder
from kinder.envs.utils import (
    BLACK,
    BROWN,
    ORANGE,
    object_to_multibody2d,
    sample_se2_pose,
    state_2d_has_collision,
)

TargetBlockType = Type("target_block", parent=DynRectangleType)
HookType = Type("hook", parent=LObjectType)
Dynamic2DRobotEnvTypeFeatures[TargetBlockType] = list(
    Dynamic2DRobotEnvTypeFeatures[DynRectangleType]
)
Dynamic2DRobotEnvTypeFeatures[HookType] = list(
    Dynamic2DRobotEnvTypeFeatures[LObjectType]
)


@dataclass(frozen=True)
class DynPushPullHook2DEnvConfig(Dynamic2DRobotEnvConfig):
    """Scene config for DynPushPullHook2DEnv()."""

    # World boundaries. Standard coordinate frame with (0, 0) in bottom left.
    world_min_x: float = 0.0
    world_max_x: float = 3.5
    world_min_y: float = 0.0
    world_max_y: float = 3.5

    # Robot parameters
    robot_base_radius: float = 0.24
    robot_arm_length_max: float = 2 * robot_base_radius
    gripper_base_width: float = 0.06
    gripper_base_height: float = 0.32
    gripper_finger_width: float = 0.2
    gripper_finger_height: float = 0.06

    # Action space parameters.
    min_dx: float = -5e-2
    max_dx: float = 5e-2
    min_dy: float = -5e-2
    max_dy: float = 5e-2
    min_dtheta: float = -np.pi / 48
    max_dtheta: float = np.pi / 48
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
        SE2Pose(0.3, 0.3, 0.0),
        SE2Pose(0.4, 0.4, np.pi / 2),
    )

    # Middle wall hyperparameters.
    middle_wall_rgb: tuple[float, float, float] = BLACK
    middle_wall_pose: tuple[float, float, float] = (
        (world_min_x + world_max_x) / 2,
        (world_min_y + world_max_y) / 2,
        0.0,
    )
    middle_wall_width: float = world_max_x - world_min_x
    middle_wall_height: float = 0.05

    # Hook hyperparameters.
    hook_rgb: tuple[float, float, float] = BROWN
    hook_shape: tuple[float, float, float] = (
        gripper_base_height / 3,
        (world_min_y + world_max_y) * 2 / 5,
        (world_min_y + world_max_y) / 6,
    )
    hook_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(world_min_x + hook_shape[1], hook_shape[2] * 1.5, -np.pi / 12),
        SE2Pose(
            world_max_x - hook_shape[0],
            (world_min_y + world_max_y) / 2 - hook_shape[2],
            np.pi / 12,
        ),
    )

    # Target block hyperparameters.
    target_block_rgb: tuple[float, float, float] = ORANGE
    target_block_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(
            world_min_x + hook_shape[2],
            (world_min_y + world_max_y) / 2 + hook_shape[2],
            -np.pi,
        ),
        SE2Pose(world_max_x - hook_shape[2], world_max_y - hook_shape[2], np.pi),
    )
    target_block_size_bounds: tuple[float, float] = (
        gripper_base_height / 2,
        hook_shape[2] - hook_shape[0],
    )
    target_block_mass: float = 1.0

    # Obstruction hyperparameters (DYNAMIC).
    obstruction_rgb: tuple[float, float, float] = BROWN
    obstruction_height_bounds: tuple[float, float] = (
        robot_base_radius / 2,
        hook_shape[2],
    )
    obstruction_width_bounds: tuple[float, float] = (
        robot_base_radius / 2,
        hook_shape[2],
    )
    obstruction_block_mass: float = 1.0
    # NOTE: obstruction poses are sampled using a 2D gaussian that is centered
    # at the target location. This hyperparameter controls the variance.
    # borrowed from clutteredretrieval2d
    obstruction_pose_init_distance_scale: float = 0.25

    # For sampling initial states.
    max_initial_state_sampling_attempts: int = 10_000

    # We don't have gravity here, but we have damping.
    gravity_y: float = 0.0
    damping: float = 0.01  # Damping applied to all dynamic bodies

    # For rendering.
    render_dpi: int = 250


class ObjectCentricDynPushPullHook2DEnv(
    ObjectCentricDynamic2DRobotEnv[DynPushPullHook2DEnvConfig]
):
    """Object-centric dynamic 2D push-pull-hook environment."""

    def __init__(
        self,
        num_obstructions: int = 2,
        config: DynPushPullHook2DEnvConfig = DynPushPullHook2DEnvConfig(),
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        self._num_obstructions = num_obstructions

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        init_state_dict: dict[Object, dict[str, float]] = {}

        # Create the middle wall.
        middle_wall = Object("middle_wall", KinRectangleType)
        init_state_dict[middle_wall] = {
            "x": self.config.middle_wall_pose[0],
            "vx": 0.0,
            "y": self.config.middle_wall_pose[1],
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
            "z_order": ZOrder.FLOOR.value,  # Middle wall does not collide with hook
        }

        # Create up-floor and down-floor, just for visualization.
        up_floor = Object("up_floor", KinRectangleType)
        init_state_dict[up_floor] = {
            "x": self.config.middle_wall_pose[0],
            "vx": 0.0,
            "y": (self.config.world_max_y + self.config.middle_wall_pose[1]) / 2,
            "vy": 0.0,
            "theta": 0.0,
            "omega": 0.0,
            "width": self.config.middle_wall_width,
            "height": (self.config.world_max_y - self.config.middle_wall_pose[1]),
            "static": True,
            "held": False,
            "color_r": BROWN[0],
            "color_g": BROWN[1],
            "color_b": BROWN[2],
            "z_order": ZOrder.FLOOR.value,
        }
        down_floor = Object("down_floor", KinRectangleType)
        init_state_dict[down_floor] = {
            "x": self.config.middle_wall_pose[0],
            "vx": 0.0,
            "y": self.config.middle_wall_pose[1] / 2,
            "vy": 0.0,
            "theta": 0.0,
            "omega": 0.0,
            "width": self.config.middle_wall_width,
            "height": (self.config.middle_wall_pose[1] - self.config.world_min_y),
            "static": True,
            "held": False,
            "color_r": ORANGE[0],
            "color_g": ORANGE[1],
            "color_b": ORANGE[2],
            "z_order": ZOrder.FLOOR.value,
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

    def _setup_physics_space(self) -> None:
        """Set up the PyMunk physics space."""
        self.pymunk_space = pymunk.Space()
        self.pymunk_space.gravity = 0, self.config.gravity_y
        self.pymunk_space.damping = self.config.damping
        self.pymunk_space.collision_slop = self.config.collision_slop

        # Create robot
        self.robot = KinRobot(
            init_pos=pymunk.Vec2d(*self.config.init_robot_pos),
            base_radius=self.config.robot_base_radius,
            arm_length_max=self.config.robot_arm_length_max,
            gripper_base_width=self.config.gripper_base_width,
            gripper_base_height=self.config.gripper_base_height,
            gripper_finger_width=self.config.gripper_finger_width,
            gripper_finger_height=self.config.gripper_finger_height,
        )
        self.robot.add_to_space(self.pymunk_space)

        # Set up collision handlers
        self.pymunk_space.on_collision(
            DYNAMIC_COLLISION_TYPE,
            FINGER_COLLISION_TYPE,
            post_solve=on_gripper_grasp,
            data=self.robot,
        )
        self.pymunk_space.on_collision(
            STATIC_COLLISION_TYPE,
            ROBOT_COLLISION_TYPE,
            pre_solve=on_collision_w_static,
            data=self.robot,
        )
        # NOTE: Arm and Finger static collisions are not handled here.

    def _sample_initial_state(self) -> ObjectCentricState:
        """Sample an initial state for the environment."""
        static_objects = set(self.initial_constant_state)
        n = self.config.max_initial_state_sampling_attempts
        robot_pose = sample_se2_pose(self.config.robot_init_pose_bounds, self.np_random)
        state = self._create_initial_state(robot_pose)
        robot = state.get_objects(KinRobotType)[0]
        # Check for collisions with the robot and static objects.
        full_state = state.copy()
        full_state.data.update(self.initial_constant_state.data)
        assert not state_2d_has_collision(full_state, {robot}, static_objects, {})
        for _ in range(n):
            target_pose = sample_se2_pose(
                self.config.target_block_init_pose_bounds, self.np_random
            )
            target_size = self.np_random.uniform(*self.config.target_block_size_bounds)
            hook_pose = sample_se2_pose(
                self.config.hook_init_pose_bounds, self.np_random
            )
            state = self._create_initial_state(
                robot_pose,
                target_pose=target_pose,
                target_size=target_size,
                hook_pose=hook_pose,
            )
            target_block = state.get_objects(TargetBlockType)[0]
            hook = state.get_objects(HookType)[0]
            full_state = state.copy()
            full_state.data.update(self.initial_constant_state.data)
            if not state_2d_has_collision(
                full_state, {target_block, robot}, {hook} | static_objects, {}
            ):
                break
        else:
            raise RuntimeError("Failed to sample target pose.")

        # Sample obstructions one by one. Assume that the scene is never so dense
        # that we need to resample earlier choices.
        obstructions: list[tuple[SE2Pose, tuple[float, float]]] = []
        for _ in range(self._num_obstructions):
            for _ in range(n):
                # Sample xy, relative to the target.
                x, y = self.np_random.normal(
                    loc=(target_pose.x, target_pose.y),
                    scale=self.config.obstruction_pose_init_distance_scale,
                    size=(2,),
                )
                # Make sure in bounds.
                if not (
                    self.config.world_min_x < x < self.config.world_max_x
                    and self.config.world_min_y < y < self.config.world_max_y
                ):
                    continue
                # Sample theta.
                theta = self.np_random.uniform(-np.pi, np.pi)
                # Check for collisions.
                obstruction_pose = SE2Pose(x, y, theta)
                # Sample shape.
                obstruction_shape = (
                    self.np_random.uniform(*self.config.obstruction_width_bounds),
                    self.np_random.uniform(*self.config.obstruction_height_bounds),
                )
                possible_obstructions = obstructions + [
                    (obstruction_pose, obstruction_shape)
                ]
                state = self._create_initial_state(
                    robot_pose,
                    target_pose=target_pose,
                    target_size=target_size,
                    hook_pose=hook_pose,
                    obstructions=possible_obstructions,
                )
                obj_name_to_obj = {o.name: o for o in state}
                full_state = state.copy()
                full_state.data.update(self.initial_constant_state.data)
                new_obstruction = obj_name_to_obj[f"obstruction{len(obstructions)}"]
                assert new_obstruction.name.startswith("obstruction")
                if not state_2d_has_collision(
                    full_state, {new_obstruction}, set(full_state), {}
                ):
                    break
            else:
                raise RuntimeError("Failed to sample obstruction pose.")
            # Update obstructions.
            obstructions = possible_obstructions
        # The state should already be finalized.
        return state

    def _create_initial_state(
        self,
        robot_pose: SE2Pose,
        target_pose: SE2Pose | None = None,
        target_size: float | None = None,
        hook_pose: SE2Pose | None = None,
        obstructions: list[tuple[SE2Pose, tuple[float, float]]] | None = None,
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

        # Create the target block.
        if target_pose is not None:
            assert target_size is not None
            target_block = Object("target_block", TargetBlockType)
            init_state_dict[target_block] = {
                "x": target_pose.x,
                "vx": 0.0,
                "y": target_pose.y,
                "vy": 0.0,
                "theta": target_pose.theta,
                "omega": 0.0,
                "width": target_size,
                "height": target_size,
                "static": False,
                "held": False,
                "mass": self.config.target_block_mass,
                "color_r": self.config.target_block_rgb[0],
                "color_g": self.config.target_block_rgb[1],
                "color_b": self.config.target_block_rgb[2],
                # Hook does not collide with middle wall
                "z_order": ZOrder.SURFACE.value,
            }

        # Create the hook.
        if hook_pose is not None:
            target_block = Object("hook", HookType)
            init_state_dict[target_block] = {
                "x": hook_pose.x,
                "vx": 0.0,
                "y": hook_pose.y,
                "vy": 0.0,
                "theta": hook_pose.theta,
                "omega": 0.0,
                "mass": self.config.obstruction_block_mass,
                "width": self.config.hook_shape[0],
                "length_side1": self.config.hook_shape[1],
                "length_side2": self.config.hook_shape[2],
                "static": False,
                "held": False,
                "color_r": self.config.hook_rgb[0],
                "color_g": self.config.hook_rgb[1],
                "color_b": self.config.hook_rgb[2],
                # Hook does not collide with middle wall
                "z_order": ZOrder.SURFACE.value,
            }

        # Create obstructions.
        if obstructions:
            for i, (obstruction_pose, obstruction_shape) in enumerate(obstructions):
                obstruction = Object(f"obstruction{i}", DynRectangleType)
                init_state_dict[obstruction] = {
                    "x": obstruction_pose.x,
                    "vx": 0.0,
                    "y": obstruction_pose.y,
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
                    "z_order": ZOrder.SURFACE.value,
                }

        # Finalize state.
        return create_state_from_dict(init_state_dict, Dynamic2DRobotEnvTypeFeatures)

    def _add_state_to_space(self, state: ObjectCentricState) -> None:
        """Add objects from the state to the PyMunk space.

        The robot must be processed first so that the gripper is at the
        correct position before any held objects call ``add_to_hand``
        (which computes a relative pose from the current gripper pose).
        """
        assert self.pymunk_space is not None, "Space not initialized"
        # Process the robot first, then everything else.
        robot_objs = [o for o in state if o.is_instance(KinRobotType)]
        other_objs = [o for o in state if not o.is_instance(KinRobotType)]
        for obj in robot_objs:
            self._reset_robot_in_space(obj, state)
        for obj in other_objs:
            # Everything else are rectangles in this environment.
            x = state.get(obj, "x")
            y = state.get(obj, "y")
            width = state.get(obj, "width")
            theta = state.get(obj, "theta")
            vx = state.get(obj, "vx")
            vy = state.get(obj, "vy")
            omega = state.get(obj, "omega")
            held = state.get(obj, "held")
            # Add static objects (table, walls)
            if "wall" in obj.name:
                # Static objects
                # We use Pymunk kinematic bodies for static objects
                height = state.get(obj, "height")
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
            elif obj.is_instance(DynRectangleType):
                # Target block and obstructions
                assert not held, "Blocks cannot be held in this env"
                height = state.get(obj, "height")
                mass = state.get(obj, "mass")
                vs = [
                    (-width / 2, -height / 2),
                    (-width / 2, height / 2),
                    (width / 2, height / 2),
                    (width / 2, -height / 2),
                ]
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
            elif obj.is_instance(HookType):
                mass = state.get(obj, "mass")
                x, y = state.get(obj, "x"), state.get(obj, "y")
                theta = state.get(obj, "theta")
                l1 = state.get(obj, "length_side1")
                l2 = state.get(obj, "length_side2")
                w = state.get(obj, "width")
                # Approximate moment of inertia for L-shape as two rectangles
                vertices = [
                    (0, 0),  # The right top vertex described in L101.
                    (-l1, 0),
                    (-l1, -w),
                    (-w, -w),
                    (-w, -l2),
                    (0, -l2),
                    (0, -w),
                    (-w, 0),
                ]
                vs_l1 = (
                    vertices[0],
                    vertices[1],
                    vertices[2],
                    vertices[6],
                )
                vs_l2 = (
                    vertices[4],
                    vertices[5],
                    vertices[0],
                    vertices[7],
                )
                moment1 = pymunk.moment_for_poly(mass / 2, vs_l1)
                moment2 = pymunk.moment_for_poly(mass / 2, vs_l2)
                moment = moment1 + moment2
                if not held:
                    # Dynamic objects
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
                    assert shape1.body is not None
                    assert shape2.body is not None
                    shape1.body.moment = moment1
                    shape1.body.mass = mass
                    shape2.body.moment = moment2
                    shape2.body.mass = mass
                    self.pymunk_space.add(body, shape1, shape2)
                    body.angle = theta
                    body.position = x, y
                    body.velocity = vx, vy
                    body.angular_velocity = omega
                    self._state_obj_to_pymunk_body[obj] = body
                else:
                    # Held dynamic objects are treated as kinematic.
                    # Match on_gripper_grasp: do NOT set velocity from
                    # state — the PD controller will set the correct
                    # velocity on the next step.
                    body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
                    shape1 = pymunk.Poly(body, vs_l1)
                    shape2 = pymunk.Poly(body, vs_l2)
                    shape1.friction = 1.0
                    shape1.density = 1.0
                    shape1.collision_type = ARM_COLLISION_TYPE
                    shape1.mass = mass / 2
                    shape2.friction = 1.0
                    shape2.density = 1.0
                    shape2.mass = mass / 2
                    shape2.collision_type = ARM_COLLISION_TYPE
                    self.pymunk_space.add(body, shape1, shape2)
                    body.angle = theta
                    body.position = x, y
                    # Add to robot hand
                    self._state_obj_to_pymunk_body[obj] = body
                    assert self.robot is not None, "Robot not initialized"
                    self.robot.add_to_hand((body, [shape1, shape2]), mass)
            else:
                assert "floor" in obj.name, "Unknown object type"

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
                state.set(obj, "theta", wrap_angle(pymunk_body.angle))
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
        """Check if the target condition is satisfied."""
        # If middle wall and target block geometrically intersect
        target_block = state.get_objects(TargetBlockType)[0]
        middle_wall = [
            o for o in self.initial_constant_state if o.name == "middle_wall"
        ][0]
        full_state = state.copy()
        full_state.data.update(self.initial_constant_state.data)
        target_block_body = object_to_multibody2d(
            target_block, full_state, static_object_body_cache
        )
        middle_wall_body = object_to_multibody2d(
            middle_wall, full_state, static_object_body_cache
        )
        if target_block_body.bodies[0].geom.intersects(middle_wall_body.bodies[0].geom):
            return True

        return False

    def _get_reward_and_done(self) -> tuple[float, bool]:
        """Calculate reward and termination."""
        assert self._current_state is not None
        terminated = self._target_satisfied(
            self._current_state,
            self._static_object_body_cache,
        )
        return -1.0, terminated


class DynPushPullHook2DEnv(ConstantObjectKinDEREnv):
    """Dynamic Push-Pull Hook 2D env with a constant number of objects."""

    def __init__(self, num_obstructions: int = 3, **kwargs) -> None:
        self._num_obstructions = num_obstructions
        super().__init__(num_obstructions=num_obstructions, **kwargs)

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricDynPushPullHook2DEnv:
        return ObjectCentricDynPushPullHook2DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        constant_objects = ["robot", "hook", "target_block"]
        for obj in sorted(exemplar_state):
            if obj.name.startswith("obstruction"):
                constant_objects.append(obj.name)
        return constant_objects

    def _create_env_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return """A 2D physics-based tool-use environment where a robot must use a hook to push/pull a target block onto a middle wall (goal surface). The target block is positioned in the upper region of the world, while the middle wall is located at the center. The robot must manipulate the hook to navigate the target block downward through obstacles.

The target block is initially surrounded by obstacle blocks.

The robot has a movable circular base and an extendable arm with gripper fingers. The hook is a kinematic object that can be grasped and used as a tool to indirectly manipulate the target block. All dynamic objects follow PyMunk physics including gravity, friction, and collisions.

Each object includes physics properties like mass, moment of inertia (for dynamic objects), and color information for rendering.
"""

    def _create_in_context_examples(self):
        # pylint: disable=line-too-long
        return """
**Example 1: Hook Manipulation to Push Target Block (With Obstructions)**

Initial State:
- Robot: position (0.362, 0.394), angle 0.306 rad (≈17.5°), arm length 0.48m (fully extended), gripper open (gap 0.32m)
- Hook: position (2.706, 1.013), angle -0.142 rad (≈-8.1°), L-shaped tool with dimensions width=0.107m, length_side1=1.4m, length_side2=0.583m, mass 1.0kg
- Target block: position (2.297, 2.833), angle -1.842 rad (≈-105.5°), dimensions 0.405m × 0.405m (square), mass 1.0kg, located in upper region
- Middle wall (goal surface): position (1.75, 1.75), black horizontal barrier spanning full world width (3.5m × 0.05m)
- Obstruction0: position (2.167, 2.499), angle -2.548 rad, dimensions 0.173m × 0.173m
- Obstruction1: position (1.854, 3.141), angle -0.859 rad, dimensions 0.224m × 0.192m
- Obstruction2: position (2.570, 3.203), angle -0.279 rad, dimensions 0.433m × 0.282m
- Obstruction3: position (2.813, 2.773), angle 0.788 rad, dimensions 0.228m × 0.340m
- Obstruction4: position (3.293, 2.627), angle 0.696 rad, dimensions 0.355m × 0.248m
- World boundaries: 3.5m × 3.5m, no gravity (gravity_y = 0.0), damping = 0.01

Goal: Use the L-shaped hook as a tool to push/pull the target block from the upper region (y ≈ 2.833) down to the middle wall at center (y = 1.75), navigating through clustered obstructions

Strategy: Grasp hook → Position hook near target → Use hook geometry to push/pull target downward to goal

High-Level Plan:
1. GraspHook(robot, hook, grasp_param=[0.621])
   - Navigate robot from starting position (0.362, 0.394) toward hook at (2.706, 1.013)
   - grasp_param 0.621: normalized grasp point along hook's longer arm (length_side1 = 1.4m)
     - 0.0 = grasp at corner vertex, 1.0 = grasp at far end of side1
     - 0.621 ≈ 0.87m from corner along the 1.4m arm (strategic balance point)
   - Approach hook, extend arm if needed, align gripper with hook geometry
   - Close gripper fingers (reduce finger_gap from 0.32m to match hook width 0.107m + finger considerations)
   - Hook becomes held, rigidly attached to robot gripper through physics constraints
   - Hook orientation relative to gripper is maintained during manipulation
   - Executed in 36 action steps

2. PreHook(robot, hook, target_block, position_params=[2.034, -0.040, -0.013])
   - Maneuver robot with held hook to position hook strategically near target block
   - position_params = [distance, offset_x, offset_y]:
     - distance 2.034: desired radial distance from target block center to robot base
     - offset_x -0.040: fine-tuning horizontal offset for hook positioning
     - offset_y -0.013: fine-tuning vertical offset for hook positioning
   - Goal: Position hook's L-shape to engage with target block geometry
   - Navigate around obstructions (5 obstacles scattered near target)
   - Hook's long arm (1.4m) provides reach to contact target from strategic angle
   - Robot maintains grasp throughout movement (gripper stays closed)
   - Damping (0.01) provides slight resistance to motion, requiring controlled movement
   - Executed in 39 action steps

3. HookDown(robot, hook, target_block, params=[0.0])
   - Execute pulling/pushing motion to drag target block downward toward middle wall
   - params 0.0: pulling strategy parameter (specific approach for downward motion)
   - Robot moves with hook, using hook's geometry as lever to apply force to target
   - L-shape hook design allows catching/pushing target block effectively
   - No gravity, so motion relies purely on robot force transmitted through hook
   - Target block must navigate past obstructions during descent
   - Physics simulation handles:
     - Contact forces between hook and target block
     - Collisions between target and obstructions (elastic collisions with friction)
     - Damping effects on all dynamic objects
   - Continue motion until target block intersects middle wall geometry
   - Success condition: target block geometry overlaps with middle wall (goal surface)
   - Executed in 25 action steps

Goal Reached:
- Target block moved from upper region (y ≈ 2.833) to middle wall (y = 1.75) ✓
- Hook used as extended tool to manipulate target indirectly ✓
- Navigated through 5 obstructions successfully ✓
- Hand still holding hook (HandEmpty predicate false, HoldingHook true) ✓
- Total actions: 36 + 39 + 25 = 100 steps

Key Insights:
- Hook is an L-shaped tool providing extended reach (1.4m + 0.583m arms)
- No gravity means objects don't fall naturally; robot must actively push/pull
- Damping (0.01) provides slight drag on all moving objects
- Grasp point selection (0.621) balances control and reach for hook manipulation
- PreHook positioning is critical for effective force application
- Target must intersect middle wall geometry to satisfy goal condition
- Obstructions add complexity but don't fundamentally change strategy
- Action space: [dx, dy, dtheta, darm, dgripper] with ranges ±0.05m, ±0.05m, ±π/48 rad, ±0.1m, ±0.02m per step

**Example 2: Simpler Hook Task (Fewer Obstructions)**

Initial State:
- Robot: position (0.35, 0.38), angle 0.5 rad, arm length 0.48m, gripper open
- Hook: position (2.50, 1.20), angle 0.0 rad, L-shaped with standard dimensions (width=0.107m, sides 1.4m × 0.583m)
- Target block: position (2.80, 2.60), angle 0.0 rad, dimensions 0.35m × 0.35m, mass 1.0kg
- Middle wall: position (1.75, 1.75), spanning full width (goal surface)
- Obstruction0: position (2.60, 2.40), dimensions 0.20m × 0.25m
- Obstruction1: position (2.95, 2.30), dimensions 0.18m × 0.22m
- World: 3.5m × 3.5m, no gravity, damping 0.01

Goal: Pull target block to middle wall using hook tool

Strategy: Grasp hook → Position hook above/beside target → Pull target down to goal

High-Level Plan:
1. GraspHook(robot, hook, grasp_param=[0.55])
   - Navigate from (0.35, 0.38) to hook at (2.50, 1.20)
   - Grasp at point 0.55 along hook's main arm (≈0.77m from corner)
   - Close gripper around hook width (0.107m)
   - Hook held securely by robot

2. PreHook(robot, hook, target_block, position_params=[1.80, 0.05, -0.08])
   - Position robot with hook near target at (2.80, 2.60)
   - distance 1.80: closer approach for better control
   - offset_x 0.05, offset_y -0.08: position hook to catch target from side
   - Align hook's L-shape to engage target block effectively
   - Navigate around 2 obstructions during approach

3. HookDown(robot, hook, target_block, params=[0.0])
   - Execute downward pulling motion with hook
   - Use hook's long arm to maintain contact with target
   - Pull target from y ≈ 2.60 down to middle wall at y = 1.75
   - Distance traveled: ~0.85m downward
   - Obstructions passively affected by target motion (collisions handled by physics)
   - Goal achieved when target intersects middle wall

Goal Reached:
- Target on middle wall ✓
- Hook used effectively as pulling tool ✓
- Fewer obstructions meant faster execution ✓

Key Insights:
- With fewer obstructions, PreHook positioning is simpler
- Grasp point (0.55) closer to center provides balanced control
- Hook's length (1.4m main arm) allows reaching target from safe distance
- Damping slows objects gradually, preventing uncontrolled motion

**Example 3: Alternative Hook Strategy (Pushing vs. Pulling)**

Initial State:
- Robot: position (0.40, 0.35), angle 0.2 rad, arm retracted (0.24m initially, extends to 0.48m)
- Hook: position (2.30, 0.90), angle 0.3 rad, standard L-shape dimensions
- Target block: position (2.00, 3.00), angle 0.5 rad, dimensions 0.40m × 0.40m
- Middle wall: center at (1.75, 1.75), goal surface
- Obstruction0: position (1.85, 2.70), dimensions 0.25m × 0.20m
- Obstruction1: position (2.15, 2.85), dimensions 0.22m × 0.28m
- Obstruction2: position (2.30, 2.50), dimensions 0.19m × 0.24m

Goal: Push target block downward to middle wall

Strategy: Grasp hook → Position hook above target → Push target down using hook's perpendicular arm

High-Level Plan:
1. GraspHook(robot, hook, grasp_param=[0.70])
   - Grasp hook at 0.70 position (≈0.98m from corner, further along main arm)
   - Provides leverage for pushing motion
   - Navigate from start to hook position

2. PreHook(robot, hook, target_block, position_params=[2.20, -0.10, 0.15])
   - Position robot above target (higher y-offset 0.15)
   - distance 2.20: maintain clearance for pushing motion
   - Hook oriented to use shorter arm (0.583m side) as pushing surface
   - Prepare for downward pushing force

3. HookDown(robot, hook, target_block, params=[0.0])
   - Push target downward using hook's perpendicular geometry
   - Robot advances downward while maintaining hook contact
   - Target block pushed through obstacle field
   - Continue until target reaches middle wall at y = 1.75
   - Physics handles force transmission from robot → hook → target → obstructions

Goal Reached:
- Target pushed to middle wall successfully ✓
- Pushing strategy effective for obstacles in path ✓
- Hook's L-shape versatility demonstrated ✓

Key Insights:
- Grasp point selection (0.70) affects leverage and control
- Hook can be used for both pulling (catching) and pushing motions
- PreHook positioning determines whether hook pulls or pushes
- L-shape geometry provides multiple contact surfaces for manipulation
- Obstructions may need to be pushed aside rather than avoided

**General Operation Interpretation:**

- GraspHook(robot, hook, grasp_param=[ratio]):
  - Navigate robot to hook location
  - grasp_param (0.0 to 1.0): normalized position along hook's main arm (length_side1)
    - 0.0 = corner vertex (where two arms meet)
    - 0.5 = midpoint of main arm (0.7m from corner if arm is 1.4m)
    - 1.0 = far end of main arm (full 1.4m from corner)
  - Extend arm, align gripper with hook geometry
  - Close gripper fingers to secure hook (width ≈ 0.107m)
  - Hook becomes held, moves rigidly with robot

- PreHook(robot, hook, target_block, position_params=[distance, offset_x, offset_y]):
  - Navigate robot with held hook to strategic position near target
  - distance: radial distance from target center to robot base
  - offset_x, offset_y: fine-tuning adjustments for hook alignment
  - Goal: Position hook's L-shape to effectively engage with target
  - Maintain grasp throughout navigation
  - Avoid collisions with obstructions during approach

- HookDown(robot, hook, target_block, params=[strategy]):
  - Execute pushing/pulling motion to move target toward middle wall
  - strategy param: approach for applying force (0.0 indicates default downward motion)
  - Robot moves while maintaining hook contact with target
  - Hook geometry acts as force transmission mechanism
  - Continue until target intersects middle wall (goal condition)
  - Physics handles all contact forces and collisions

**Environment Properties:**

World Configuration:
- Dimensions: 3.5m × 3.5m square space
- Middle wall: 3.5m wide × 0.05m tall, positioned at center (1.75, 1.75)
- Upper region (y > 1.75): brown floor (target spawn area)
- Lower region (y < 1.75): orange floor (robot spawn area)
- No gravity (gravity_y = 0.0): objects don't fall, must be pushed/pulled
- Damping = 0.01: slight resistance to motion for all dynamic bodies

Physics Properties:
- All dynamic objects: mass 1.0kg, friction 1.0, elasticity 0.99 (nearly elastic collisions)
- Hook: L-shaped rigid body with two perpendicular arms
  - Main arm (side1): 1.4m length
  - Perpendicular arm (side2): 0.583m length
  - Width: 0.107m (thickness of L-shape)
  - Moment of inertia: computed as sum of two rectangular components
- Target block: square dynamic object (typically 0.3-0.5m per side)
- Obstructions: rectangular dynamic objects (various sizes, 0.1-0.6m range)

Action Space:
- 5D continuous: [dx, dy, dtheta, darm, dgripper]
- dx, dy: Base displacement (range ±0.05m per step)
- dtheta: Angular rotation (range ±π/48 rad ≈ ±3.75° per step, slower than DynObstruction2D)
- darm: Arm extension/retraction (range ±0.1m per step, arm length 0.24-0.48m)
- dgripper: Finger gap change (range ±0.02m per step, max gap 0.32m)

Goal Condition:
- Success when target block geometry intersects middle wall geometry (shapely intersection test)
- Target must physically overlap with middle wall's bounding box
- Different from "on top" condition in DynObstruction2D
- Robot may still be holding hook at task completion

**Strategy Selection:**

Hook Manipulation Approaches:
1. **Pulling Strategy**: Position hook on far side of target, pull toward middle wall
   - Effective when obstructions are on near side
   - Hook's long arm (1.4m) provides reach
   - Grasp point 0.5-0.7 for balanced control

2. **Pushing Strategy**: Position hook behind target, push toward middle wall
   - Effective when clear path exists below target
   - Use hook's perpendicular arm (0.583m) as pushing surface
   - Grasp point 0.6-0.8 for leverage

3. **Obstacle Navigation**: 
   - With few obstacles (0-2): direct approach feasible
   - With many obstacles (3-5): careful PreHook positioning required
   - May need to push obstructions aside during HookDown

Grasp Point Guidelines:
- 0.3-0.5: Close to corner, good for fine control, shorter effective reach
- 0.5-0.7: Balanced position, versatile for both pulling and pushing
- 0.7-0.9: Far from corner, maximum reach, requires more force control

Key Differences from DynObstruction2D:
- Tool use (hook) vs. direct manipulation (gripper)
- No gravity: requires active force application throughout
- Goal is intersection (not "on top"): less precise positioning needed
- L-shaped tool geometry provides multiple manipulation strategies
- Damping is lower (0.01 vs. default), affecting motion dynamics
"""

    def _create_variant_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return "The number of obstructions differs between environment variants. For example, DynPushPullHook2D-o0 has no obstructions, while DynPushPullHook2D-o5 has 5 obstructions."

    def _create_variant_specific_description(self) -> str:
        if self._num_obstructions == 0:
            return "This variant has no obstructions."
        if self._num_obstructions == 1:
            return "This variant has 1 obstruction."
        return f"This variant has {self._num_obstructions} obstructions."

    def _create_reward_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return """A penalty of -1.0 is given at every time step until termination, which occurs when the target block reaches the middle wall (goal surface)."""

    def _create_references_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return """This is a dynamic version of PushPullHook2D."""
