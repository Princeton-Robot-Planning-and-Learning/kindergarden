"""Dynamic PushT environment using PyMunk physics."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pymunk
from numpy.typing import NDArray
from relational_structs import Object, ObjectCentricState, ObjectCentricStateSpace, Type
from relational_structs.utils import create_state_from_dict

from kinder.core import ConstantObjectKinDEREnv, FinalConfigMeta
from kinder.envs.dynamic2d.base_env import (
    Dynamic2DRobotEnvConfig,
    ObjectCentricDynamic2DRobotEnv,
)
from kinder.envs.dynamic2d.object_types import (
    DotRobotType,
    Dynamic2DRobotEnvTypeFeatures,
    TObjectType,
)
from kinder.envs.dynamic2d.utils import (
    DYNAMIC_COLLISION_TYPE,
    ROBOT_COLLISION_TYPE,
    STATIC_COLLISION_TYPE,
    DotRobot,
    DotRobotActionSpace,
    DotRobotPDController,
    create_walls_from_world_boundaries,
    get_dot_robot_action_from_gui_input,
    on_dot_robot_collision_w_static,
)
from kinder.envs.kinematic2d.structs import SE2Pose, ZOrder
from kinder.envs.utils import sample_se2_pose, state_2d_has_collision

# Define custom object types for the PushT environment
GoalTBlockType = Type("goal_tblock", parent=TObjectType)
Dynamic2DRobotEnvTypeFeatures[GoalTBlockType] = list(
    Dynamic2DRobotEnvTypeFeatures[TObjectType]
)


@dataclass(frozen=True)
class DynPushT2DEnvConfig(Dynamic2DRobotEnvConfig, metaclass=FinalConfigMeta):
    """Scene config for DynPushT2DEnv()."""

    # World boundaries (scaled from original 512x512 to 0-10 range)
    world_min_x: float = 0.0
    world_max_x: float = 5.0
    world_min_y: float = 0.0
    world_max_y: float = 5.0

    # Robot parameters (scaled from original 15 radius)
    init_robot_pos: tuple[float, float] = (1.0, 1.0)
    robot_radius: float = 0.1

    # Action space parameters (2D delta position for dot robot)
    min_dx: float = -5e-2
    max_dx: float = 5e-2
    min_dy: float = -5e-2
    max_dy: float = 5e-2

    # Controller parameters (from original PushT)
    kp: float = 50.0
    kv: float = 5.0

    # Physics parameters
    gravity_y: float = 0.0  # No gravity in PushT
    damping: float = 0.0  # No damping in original PushT
    collision_slop: float = 0.001
    control_hz: int = 10  # 10 Hz control like original
    sim_hz: int = 100  # 100 Hz simulation like original

    # T-block hyperparameters (scaled from original 50x100 with scale=30)
    tblock_rgb: tuple[float, float, float] = (0.5, 0.5, 0.5)  # Gray
    tblock_width_bounds: tuple[float, float] = (0.1, 0.15)  # scale=30 -> 30/512*10
    tblock_length_horizontal_bounds: tuple[float, float] = (1.0, 1.25)  # 4*scale
    tblock_length_vertical_bounds: tuple[float, float] = (1.0, 1.25)  # 4*scale
    tblock_mass: float = 1.0

    # Goal pose bounds (randomized)
    goal_rgb: tuple[float, float, float] = (0.5, 1.0, 0.5)  # Light green
    goal_tblock_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(1.0, 1.0, -np.pi),
        SE2Pose(4.0, 4.0, np.pi),
    )

    # Robot init pose bounds (scaled from original 50-450 in 512 space)
    robot_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(1.0, 1.0, 0.0),
        SE2Pose(2.0, 2.0, 0.0),
    )

    # T-block init pose bounds (scaled from original 100-400 in 512 space)
    tblock_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(1.0, 1.0, -np.pi),
        SE2Pose(4.0, 4.0, np.pi),
    )

    # Success threshold (we use dx, dy, dtheta threshold here, instead of coverage)
    success_dx_threshold: float = 0.03
    success_dy_threshold: float = 0.03
    success_dtheta_threshold: float = np.deg2rad(8)  # 10 degrees

    # For sampling initial states
    max_initial_state_sampling_attempts: int = 10_000

    # For rendering
    render_dpi: int = 50


class ObjectCentricDynPushT2DEnv(ObjectCentricDynamic2DRobotEnv[DynPushT2DEnvConfig]):
    """Dynamic PushT environment where a dot robot must push a T-shaped block to match a
    goal pose. Uses PyMunk physics simulation.

    This is a port of the original PushT environment to the ObjectCentric framework.
    """

    def __init__(
        self,
        config: DynPushT2DEnvConfig | None = None,
        num_tee: int = 1,  # noqa: ARG002
        **kwargs,
    ) -> None:
        super().__init__(config or DynPushT2DEnvConfig(), **kwargs)

        # Override robot and controller with DotRobot
        self.num_tee = num_tee
        self.dot_robot: DotRobot | None = None
        self.dot_pd_controller = DotRobotPDController(
            kp=self.config.kp,
            kv=self.config.kv,
        )

        # Store object references for tracking
        self._tblock: Object | None = None
        self._goal_tblock: Object | None = None
        self._goal_body: pymunk.Body | None = None

    def _create_action_space(self, config: DynPushT2DEnvConfig) -> DotRobotActionSpace:
        """Override to use DotRobotActionSpace."""
        return DotRobotActionSpace(
            min_dx=config.min_dx,
            max_dx=config.max_dx,
            min_dy=config.min_dy,
            max_dy=config.max_dy,
        )

    def _setup_physics_space(self) -> None:
        """Set up the PyMunk physics space with DotRobot."""
        self.pymunk_space = pymunk.Space()
        self.pymunk_space.gravity = 0, self.config.gravity_y
        self.pymunk_space.damping = self.config.damping
        self.pymunk_space.collision_slop = self.config.collision_slop

        # Create DotRobot instead of KinRobot
        self.dot_robot = DotRobot(
            init_pos=pymunk.Vec2d(*self.config.init_robot_pos),
            radius=self.config.robot_radius,
        )
        self.dot_robot.add_to_space(self.pymunk_space)

        # Set up collision handlers
        self.pymunk_space.on_collision(
            STATIC_COLLISION_TYPE,
            ROBOT_COLLISION_TYPE,
            pre_solve=on_dot_robot_collision_w_static,
            data=self.dot_robot,
        )

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        """Create constant objects (walls and goal pose)."""
        init_state_dict: dict[Object, dict[str, float]] = {}

        # Create room walls using world boundaries
        wall_state_dict = create_walls_from_world_boundaries(
            self.config.world_min_x,
            self.config.world_max_x,
            self.config.world_min_y,
            self.config.world_max_y,
            self.config.min_dx,
            self.config.max_dx,
            self.config.min_dy,
            self.config.max_dy,
        )
        init_state_dict.update(wall_state_dict)

        return init_state_dict

    def _sample_initial_state(self) -> ObjectCentricState:
        """Sample an initial state for the environment."""
        n = self.config.max_initial_state_sampling_attempts
        for _ in range(n):
            # Sample all randomized values
            robot_pose = sample_se2_pose(
                self.config.robot_init_pose_bounds, self.np_random
            )
            tblock_pose = sample_se2_pose(
                self.config.tblock_init_pose_bounds, self.np_random
            )
            tblock_width = self.np_random.uniform(*self.config.tblock_width_bounds)
            tblock_length_horizontal = self.np_random.uniform(
                *self.config.tblock_length_horizontal_bounds
            )
            tblock_length_vertical = self.np_random.uniform(
                *self.config.tblock_length_vertical_bounds
            )
            goal_tblock_pose = sample_se2_pose(
                self.config.goal_tblock_pose_bounds, self.np_random
            )

            state = self._create_initial_state(
                robot_pose,
                tblock_pose,
                tblock_width,
                tblock_length_horizontal,
                tblock_length_vertical,
                goal_tblock_pose,
            )

            full_state = state.copy()
            full_state.data.update(self.initial_constant_state.data)
            all_objects = set(full_state)
            # We use Kinematic2D collision checker for now
            if state_2d_has_collision(full_state, all_objects, all_objects, {}):
                continue
            # The goal marker is static and the collision checker skips static-static
            # pairs, so the check above never tests the goal pose against the walls.
            # Test it by placing the movable T-block at the goal pose.
            goal_check_state = full_state.copy()
            tblock = Object("tblock", TObjectType)
            goal_check_state.set(tblock, "x", goal_tblock_pose.x)
            goal_check_state.set(tblock, "y", goal_tblock_pose.y)
            goal_check_state.set(tblock, "theta", goal_tblock_pose.theta)
            static_objects = set(self.initial_constant_state)
            if state_2d_has_collision(goal_check_state, {tblock}, static_objects, {}):
                continue
            return state

        raise RuntimeError(f"Failed to sample initial state after {n} attempts")

    def _create_initial_state(
        self,
        robot_pose: SE2Pose,
        tblock_pose: SE2Pose,
        tblock_width: float,
        tblock_length_horizontal: float,
        tblock_length_vertical: float,
        goal_tblock_pose: SE2Pose,
    ) -> ObjectCentricState:
        """Create initial state with robot and T-block."""
        init_state_dict: dict[Object, dict[str, float]] = {}

        # Create the DotRobot
        robot = Object("robot", DotRobotType)
        init_state_dict[robot] = {
            "x": robot_pose.x,
            "y": robot_pose.y,
            "theta": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "omega": 0.0,
            "static": False,
            "held": False,
            "color_r": 50 / 255,
            "color_g": 50 / 255,
            "color_b": 255 / 255,
            "z_order": ZOrder.ALL.value,
            "radius": self.config.robot_radius,
        }

        # Create the T-block
        tblock = Object("tblock", TObjectType)
        init_state_dict[tblock] = {
            "x": tblock_pose.x,
            "y": tblock_pose.y,
            "theta": tblock_pose.theta,
            "vx": 0.0,
            "vy": 0.0,
            "omega": 0.0,
            "width": tblock_width,
            "length_horizontal": tblock_length_horizontal,
            "length_vertical": tblock_length_vertical,
            "static": False,
            "held": False,
            "mass": self.config.tblock_mass,
            "color_r": self.config.tblock_rgb[0],
            "color_g": self.config.tblock_rgb[1],
            "color_b": self.config.tblock_rgb[2],
            "z_order": ZOrder.SURFACE.value,
        }

        # Create the goal T-block (for visualization only, not in physics)
        goal_tblock = Object("goal_tblock", GoalTBlockType)
        self._goal_tblock = goal_tblock  # store reference
        init_state_dict[goal_tblock] = {
            "x": goal_tblock_pose.x,
            "y": goal_tblock_pose.y,
            "theta": goal_tblock_pose.theta,
            "vx": 0.0,
            "vy": 0.0,
            "omega": 0.0,
            "width": tblock_width,
            "length_horizontal": tblock_length_horizontal,
            "length_vertical": tblock_length_vertical,
            "static": True,  # Mark as static so it doesn't get added to physics
            "held": False,
            "mass": self.config.tblock_mass,
            "color_r": self.config.goal_rgb[0],
            "color_g": self.config.goal_rgb[1],
            "color_b": self.config.goal_rgb[2],
            "z_order": ZOrder.FLOOR.value,  # Render in background
        }

        # Finalize state
        return create_state_from_dict(init_state_dict, Dynamic2DRobotEnvTypeFeatures)

    def _add_state_to_space(self, state: ObjectCentricState) -> None:
        """Add objects from the state to the PyMunk space."""
        assert self.pymunk_space is not None, "Space not initialized"

        for obj in state:
            if obj.is_instance(DotRobotType):
                self._reset_robot_in_space(obj, state)
            elif obj.is_instance(GoalTBlockType):
                # Skip adding goal T-block to physics space
                continue
            elif obj.is_instance(TObjectType):
                # Add T-block
                x = state.get(obj, "x")
                y = state.get(obj, "y")
                theta = state.get(obj, "theta")
                vx = state.get(obj, "vx")
                vy = state.get(obj, "vy")
                omega = state.get(obj, "omega")
                width = state.get(obj, "width")
                length_horizontal = state.get(obj, "length_horizontal")
                length_vertical = state.get(obj, "length_vertical")
                mass = state.get(obj, "mass")

                # Create T-shape using two rectangles
                moment = pymunk.moment_for_box(
                    mass, (length_horizontal, width)
                ) + pymunk.moment_for_box(mass, (width, length_vertical))
                body = pymunk.Body(mass, moment)

                # Horizontal bar vertices (local frame)
                lh = length_horizontal
                lv = length_vertical
                w = width
                vs_horizontal = [
                    (-lh / 2, 0),
                    (-lh / 2, -w),
                    (lh / 2, -w),
                    (lh / 2, 0),
                ]
                # Vertical bar vertices (local frame)
                vs_vertical = [
                    (-w / 2, -w),
                    (-w / 2, -w - lv),
                    (w / 2, -w - lv),
                    (w / 2, -w),
                ]

                shape1 = pymunk.Poly(body, vs_horizontal)
                shape2 = pymunk.Poly(body, vs_vertical)

                shape1.friction = 1.0
                shape2.friction = 1.0
                shape1.density = 1.0
                shape2.density = 1.0
                shape1.collision_type = DYNAMIC_COLLISION_TYPE
                shape2.collision_type = DYNAMIC_COLLISION_TYPE

                # Set center of gravity
                self.pymunk_space.add(body, shape1, shape2)
                # NOTE: Importantly, set angle before position for T-Object
                # otherwise the set position will be rotated by the angle later.
                body.angle = theta
                body.position = (x, y)
                body.velocity = vx, vy
                body.angular_velocity = omega
                self._state_obj_to_pymunk_body[obj] = body
                self._tblock = obj  # store reference
            else:
                # Static objects (walls)
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
                b2.position = x, y
                b2.angle = theta
                self._state_obj_to_pymunk_body[obj] = b2

    def _reset_robot_in_space(self, obj: Object, state: ObjectCentricState) -> None:
        """Reset the DotRobot in the PyMunk space."""
        assert self.pymunk_space is not None, "Space not initialized"
        robot_x = state.get(obj, "x")
        robot_y = state.get(obj, "y")
        robot_vx = state.get(obj, "vx")
        robot_vy = state.get(obj, "vy")

        assert self.dot_robot is not None, "Robot not initialized"
        self.dot_robot.reset_position(
            x=robot_x,
            y=robot_y,
            vel=pymunk.Vec2d(robot_vx, robot_vy),
        )

    def _read_state_from_space(self) -> None:
        """Read the current state from the PyMunk space."""
        assert self.pymunk_space is not None, "Space not initialized"
        assert self._current_state is not None, "Current state not initialized"

        state = self._current_state.copy()

        # Update dynamic object positions from PyMunk simulation
        for obj in state:
            if state.get(obj, "static"):
                continue
            if obj.is_instance(DotRobotType):
                # Update robot state
                assert self.dot_robot is not None, "Robot not initialized"
                robot_obj = state.get_objects(DotRobotType)[0]
                state.set(robot_obj, "x", self.dot_robot.pose.x)
                state.set(robot_obj, "y", self.dot_robot.pose.y)
                state.set(robot_obj, "vx", self.dot_robot.vel.x)
                state.set(robot_obj, "vy", self.dot_robot.vel.y)
            else:
                # Update T-block state
                assert (
                    obj in self._state_obj_to_pymunk_body
                ), f"Object {obj.name} not found in pymunk body cache"
                pymunk_body = self._state_obj_to_pymunk_body[obj]
                state.set(obj, "x", pymunk_body.position.x)
                state.set(obj, "y", pymunk_body.position.y)
                # Angle should be in [-pi, pi]
                if pymunk_body.angle > np.pi:
                    pymunk_body.angle -= 2 * np.pi
                elif pymunk_body.angle < -np.pi:
                    pymunk_body.angle += 2 * np.pi
                state.set(obj, "theta", pymunk_body.angle)
                state.set(obj, "vx", pymunk_body.velocity.x)
                state.set(obj, "vy", pymunk_body.velocity.y)
                state.set(obj, "omega", pymunk_body.angular_velocity)

        # Update the current state
        self._current_state = state

    def step(
        self, action: np.ndarray
    ) -> tuple[ObjectCentricState, float, bool, bool, dict]:
        """Override step to use DotRobot control."""
        assert self.action_space.contains(action)
        dx, dy = action
        assert self._current_state is not None, "Need to call reset()"
        assert self.pymunk_space is not None, "Space not initialized"
        assert self.dot_robot is not None, "Robot not initialized"

        # Compute target position from current position + delta
        curr_pos = self.dot_robot.pose
        tgt_x = curr_pos.x + dx
        tgt_y = curr_pos.y + dy

        # Calculate simulation parameters
        sim_dt = 1.0 / self.config.sim_hz
        control_dt = 1.0 / self.config.control_hz
        n_steps = self.config.sim_hz // self.config.control_hz

        # Multi-step simulation like original PushT
        for _ in range(n_steps):
            # Use PD control to compute velocity
            velocity = self.dot_pd_controller.compute_control(
                self.dot_robot,
                tgt_x,
                tgt_y,
                control_dt,
            )
            # Update robot velocity
            self.dot_robot.update(velocity)

            # Step physics simulation (more fine-grained than control freq)
            for _ in range(self.config.sim_hz // self.config.control_hz):
                self.pymunk_space.step(sim_dt)

        reward, terminated = self._get_reward_and_done()
        truncated = False  # no maximum horizon, by default
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def _get_reward_and_done(self) -> tuple[float, bool]:
        """Calculate reward and termination based on position/angle thresholds."""
        assert self._current_state is not None
        assert self._tblock is not None
        assert self._goal_tblock is not None
        tblock_x = self._current_state.get(self._tblock, "x")
        tblock_y = self._current_state.get(self._tblock, "y")
        tblock_theta = self._current_state.get(self._tblock, "theta")

        tblock_goal_x = self._current_state.get(self._goal_tblock, "x")
        tblock_goal_y = self._current_state.get(self._goal_tblock, "y")
        tblock_goal_theta = self._current_state.get(self._goal_tblock, "theta")

        dx_abs_ok = abs(tblock_x - tblock_goal_x) < self.config.success_dx_threshold
        dy_abs_ok = abs(tblock_y - tblock_goal_y) < self.config.success_dy_threshold
        dtheta_abs_ok = (
            abs(tblock_theta - tblock_goal_theta) < self.config.success_dtheta_threshold
        )
        terminated = dx_abs_ok and dy_abs_ok and dtheta_abs_ok

        return -1.0, terminated

    def get_action_from_gui_input(self, gui_input: dict[str, Any]) -> NDArray[Any]:
        """Get the mapping from human inputs to actions."""
        # This will be implemented later
        assert isinstance(self.action_space, DotRobotActionSpace)
        return get_dot_robot_action_from_gui_input(self.action_space, gui_input)


class DynPushT2DEnv(ConstantObjectKinDEREnv):
    """Dynamic PushT2D env with a constant number of objects."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # This is a Box space with some extra functionality to allow easy vectorizing.
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
        return ObjectCentricDynPushT2DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState  # noqa: ARG002
    ) -> list[str]:
        constant_objects = ["tblock", "robot", "goal_tblock"]
        return constant_objects

    def _create_env_markdown_description(self) -> str:
        """Create environment description."""
        # pylint: disable=line-too-long
        return """A 2D physics-based environment where the goal is to push a T-shaped block to match a goal pose using a simple dot robot (kinematic circle) with PyMunk physics simulation.

The T-shaped block must be positioned within small position and orientation thresholds of the goal.
"""

    def _create_variant_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return "This environment has only one variant."

    def _create_variant_specific_description(self) -> str:
        return "This variant has a T-shaped block to push to a goal pose."

    def _create_reward_markdown_description(self) -> str:
        """Create reward description."""
        # pylint: disable=line-too-long
        return """A penalty of -1.0 is given at every time step until the T-block is aligned with the goal pose within specified thresholds.

**Termination Condition**: The episode terminates when all of the following conditions are met:
- Position error in X: |x - x_goal| < 0.03
- Position error in Y: |y - y_goal| < 0.03
- Orientation error: |θ - θ_goal| < 8 degrees

These thresholds ensure the T-block is precisely aligned with the goal pose.
"""

    def _create_references_markdown_description(self) -> str:
        """Create references description."""
        # pylint: disable=line-too-long
        return """This implementation is based on the Push-T environment introduced in the Diffusion Policy paper (Chi et al., 2023)."""
