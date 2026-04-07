"""Environment where only 2D motion planning is needed to reach a goal region."""

from dataclasses import dataclass

import numpy as np
from relational_structs import Object, ObjectCentricState, Type
from relational_structs.utils import create_state_from_dict

from kinder.core import ConstantObjectKinDEREnv, FinalConfigMeta
from kinder.envs.kinematic2d.base_env import (
    Kinematic2DRobotEnvConfig,
    ObjectCentricKinematic2DRobotEnv,
)
from kinder.envs.kinematic2d.object_types import (
    CRVRobotType,
    Kinematic2DRobotEnvTypeFeatures,
    RectangleType,
)
from kinder.envs.kinematic2d.structs import SE2Pose, ZOrder
from kinder.envs.kinematic2d.utils import (
    CRVRobotActionSpace,
    create_walls_from_world_boundaries,
    rectangle_object_to_geom,
)
from kinder.envs.utils import BLACK, PURPLE, sample_se2_pose, state_2d_has_collision

TargetRegionType = Type("target_region", parent=RectangleType)
Kinematic2DRobotEnvTypeFeatures[TargetRegionType] = list(
    Kinematic2DRobotEnvTypeFeatures[RectangleType]
)


@dataclass(frozen=True)
class Motion2DEnvConfig(Kinematic2DRobotEnvConfig, metaclass=FinalConfigMeta):
    """Config for Motion2DEnv()."""

    # World boundaries. Standard coordinate frame with (0, 0) in bottom left.
    world_min_x: float = 0.0
    world_max_x: float = 2.5
    world_min_y: float = 0.0
    world_max_y: float = 2.5

    # Action space parameters.
    min_dx: float = -5e-2
    max_dx: float = 5e-2
    min_dy: float = -5e-2
    max_dy: float = 5e-2
    min_dtheta: float = -np.pi / 16
    max_dtheta: float = np.pi / 16
    # NOTE: these should not be needed, but they are included for consistency
    # with the other kinematic2d environments.
    min_darm: float = -1e-1
    max_darm: float = 1e-1
    min_vac: float = 0.0
    max_vac: float = 1.0

    # Robot hyperparameters.
    robot_base_radius: float = 0.1
    robot_arm_length: float = 2 * robot_base_radius
    robot_gripper_height: float = 0.07
    robot_gripper_width: float = 0.01
    # The robot starts on the left side of the screen.
    robot_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(
            world_min_x + 2.5 * robot_base_radius,
            world_min_y + 3 * robot_base_radius,
            -np.pi,
        ),
        SE2Pose(
            world_min_x + 3 * robot_base_radius,
            world_max_y - 3 * robot_base_radius,
            np.pi,
        ),
    )

    # Target region hyperparameters.
    target_region_rgb: tuple[float, float, float] = PURPLE
    # The target region starts on the right side of the screen.
    target_region_init_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(
            world_max_x - 3 * robot_base_radius,
            world_min_y + 3 * robot_base_radius,
            0,
        ),
        SE2Pose(
            world_max_x - 2.5 * robot_base_radius,
            world_max_y - 3 * robot_base_radius,
            0,
        ),
    )
    target_region_shape: tuple[float, float] = (
        2.5 * robot_base_radius,
        2.5 * robot_base_radius,
    )

    # Obstacle hyperparameters.
    obstacle_rgb: tuple[float, float, float] = BLACK
    obstacle_width: float = robot_base_radius / 10
    obstacle_min_x: float = robot_init_pose_bounds[1].x + 2 * robot_base_radius
    obstacle_max_x: float = target_region_init_bounds[0].x - (
        2 * robot_base_radius + obstacle_width
    )
    obstacle_passage_height_bounds: tuple[float, float] = (
        2.5 * robot_base_radius,
        4.0 * robot_base_radius,
    )
    obstacle_passage_y_bounds: tuple[float, float] = (
        world_min_y + 2 * robot_base_radius,
        world_max_y - 2 * robot_base_radius,
    )

    # For rendering.
    render_dpi: int = 300
    render_fps: int = 20


class ObjectCentricMotion2DEnv(ObjectCentricKinematic2DRobotEnv[Motion2DEnvConfig]):
    """Only 2D motion planning is needed to reach a goal region.

    This is an object-centric environment. The vectorized version with Box spaces is
    defined below.
    """

    def __init__(
        self,
        num_passages: int = 2,
        config: Motion2DEnvConfig = Motion2DEnvConfig(),
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        self._num_passages = num_passages

    def _sample_initial_state(self) -> ObjectCentricState:
        # Sample initial robot pose.
        robot_pose = sample_se2_pose(self.config.robot_init_pose_bounds, self.np_random)
        # Sample initial target region pose.
        target_region_pose = sample_se2_pose(
            self.config.target_region_init_bounds, self.np_random
        )
        # Sample obstacles to form vertical narrow passages.
        obstacles: list[tuple[SE2Pose, tuple[float, float]]] = []
        min_x = self.config.obstacle_min_x
        max_x = self.config.obstacle_max_x
        if self._num_passages > 1:
            x_dist_between_passages = (
                max_x - min_x - self._num_passages * self.config.obstacle_width
            ) / (self._num_passages - 1)
            assert x_dist_between_passages > 2 * self.config.robot_base_radius
        else:
            x_dist_between_passages = 0.0  # not used
        for i in range(self._num_passages):
            # Sample the passage parameters.
            passage_y = self.np_random.uniform(*self.config.obstacle_passage_y_bounds)
            passage_height = self.np_random.uniform(
                *self.config.obstacle_passage_height_bounds
            )
            x = min_x + i * (self.config.obstacle_width + x_dist_between_passages)
            # Create the bottom obstacle.
            y = self.config.world_min_y
            height = passage_y - y
            pose = SE2Pose(x, y, 0.0)
            shape = (self.config.obstacle_width, height)
            obstacles.append((pose, shape))
            # Create the top obstacle.
            y = y + height + passage_height
            pose = SE2Pose(x, y, 0.0)
            height = self.config.world_max_y - y
            shape = (self.config.obstacle_width, height)
            obstacles.append((pose, shape))

        state = self._create_initial_state(robot_pose, target_region_pose, obstacles)

        # Sanity check.
        robot = state.get_objects(CRVRobotType)[0]
        target_region = state.get_objects(TargetRegionType)[0]
        assert not state_2d_has_collision(state, {robot, target_region}, set(state), {})

        return state

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        init_state_dict: dict[Object, dict[str, float]] = {}

        # Create room walls.
        assert isinstance(self.action_space, CRVRobotActionSpace)
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

    def _create_initial_state(
        self,
        robot_pose: SE2Pose,
        target_region_pose: SE2Pose | None = None,
        obstacles: list[tuple[SE2Pose, tuple[float, float]]] | None = None,
    ) -> ObjectCentricState:
        # Shallow copy should be okay because the constant objects should not
        # ever change in this method.
        init_state_dict: dict[Object, dict[str, float]] = {}

        # Create the robot.
        robot = Object("robot", CRVRobotType)
        init_state_dict[robot] = {
            "x": robot_pose.x,
            "y": robot_pose.y,
            "theta": robot_pose.theta,
            "base_radius": self.config.robot_base_radius,
            "arm_joint": self.config.robot_base_radius,  # arm is fully retracted
            "arm_length": self.config.robot_arm_length,
            "vacuum": 0.0,  # vacuum is off
            "gripper_height": self.config.robot_gripper_height,
            "gripper_width": self.config.robot_gripper_width,
        }

        # Create the target region.
        if target_region_pose is not None:
            target_region = Object("target_region", TargetRegionType)
            init_state_dict[target_region] = {
                "x": target_region_pose.x,
                "y": target_region_pose.y,
                "theta": target_region_pose.theta,
                "width": self.config.target_region_shape[0],
                "height": self.config.target_region_shape[1],
                "static": True,  # NOTE
                "color_r": self.config.target_region_rgb[0],
                "color_g": self.config.target_region_rgb[1],
                "color_b": self.config.target_region_rgb[2],
                "z_order": ZOrder.NONE.value,
            }

        # Create the obstacles.
        if obstacles:
            for i, (obstacle_pose, obstacle_shape) in enumerate(obstacles):
                obstacle = Object(f"obstacle{i}", RectangleType)
                init_state_dict[obstacle] = {
                    "x": obstacle_pose.x,
                    "y": obstacle_pose.y,
                    "theta": obstacle_pose.theta,
                    "width": obstacle_shape[0],
                    "height": obstacle_shape[1],
                    "static": True,  # NOTE
                    "color_r": self.config.obstacle_rgb[0],
                    "color_g": self.config.obstacle_rgb[1],
                    "color_b": self.config.obstacle_rgb[2],
                    "z_order": ZOrder.ALL.value,
                }

        # Finalize state.
        return create_state_from_dict(init_state_dict, Kinematic2DRobotEnvTypeFeatures)

    def _get_reward_and_done(self) -> tuple[float, bool]:
        # Terminate when the robot position is in the target region.
        assert self._current_state is not None
        robot = self._current_state.get_objects(CRVRobotType)[0]
        x = self._current_state.get(robot, "x")
        y = self._current_state.get(robot, "y")
        target_region = self._current_state.get_objects(TargetRegionType)[0]
        # NOTE: the type: ignore can be removed after refactor.
        target_region_geom = rectangle_object_to_geom(
            self._current_state,
            target_region,
            self._static_object_body_cache,
        )
        terminated = target_region_geom.contains_point(x, y)
        return -1.0, terminated


class Motion2DEnv(ConstantObjectKinDEREnv):
    """Motion 2D env with a constant number of objects."""

    def __init__(self, num_passages: int = 3, **kwargs) -> None:
        self._num_passages = num_passages
        super().__init__(num_passages=num_passages, **kwargs)

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricKinematic2DRobotEnv:
        return ObjectCentricMotion2DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        constant_objects = ["robot", "target_region"]
        for obj in sorted(exemplar_state):
            if obj.name.startswith("obstacle"):
                constant_objects.append(obj.name)
        return constant_objects

    def _create_env_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return """A 2D environment where the goal is to reach a target region while avoiding static obstacles.

There may be narrow passages.
        
The robot has a movable circular base and a retractable arm with a rectangular vacuum end effector. The arm and vacuum do not need to be used in this environment.
"""

    def _create_in_context_examples(self) -> str:
        """Create concrete examples showing strategies for solving the Motion2D task.

        Returns formatted examples with actual object positions and action sequences.
        """
        # pylint: disable=line-too-long
        return """
**Example 1: Open Space (No Obstacles)**

Initial State:
- Robot: position (0.3, 1.25, 0.0), base radius 0.1m
- Target Region: center (2.25, 1.25), size 0.25m × 0.25m (rectangle from 2.125 to 2.375 in both x and y)
- World: 2.5m × 2.5m, no obstacles

Goal: Robot base position must be inside target region

Strategy: Navigate directly to target region using collision-free motion planning

High-Level Plan:
1. move_to_tgt_from_no_passage(robot, target_region, params=[0.5, 0.5, 0.5])
   - Navigate robot directly to target region center
   - params=[0.5, 0.5, 0.5] specifies:
     - rel_x=0.5: target x-position at center of target region (50% across width)
     - rel_y=0.5: target y-position at center of target region (50% across height)
     - rel_theta=0.5: normalized angle (0.5 → 0° orientation after denormalization)
   - Motion planning computes collision-free path from (0.3, 1.25) to (2.25, 1.25)
   - No obstacles present, so direct path is optimal
   - Controller executes waypoint sequence to reach target

Goal Reached:
- Robot position (2.25, 1.25) is inside target region [2.125, 2.375] × [2.125, 2.375] ✓
- Efficient direct navigation with no obstacle avoidance needed ✓

Key Insights:
- params=[rel_x, rel_y, rel_theta] are normalized coordinates (0.0 to 1.0)
- rel_x, rel_y specify target position within rectangular target region
- rel_theta converts to absolute angle: abs_theta = rel_theta * 2π - π
- Motion planning ensures collision-free path

**Example 2: Single Narrow Passage**

Initial State:
- Robot: position (0.3, 1.25, 0.0), base radius 0.1m
- Target Region: center (2.25, 1.25), size 0.25m × 0.25m
- Obstacles: Two rectangles forming vertical wall at x=1.25 with passage opening
  - obstacle_bottom: center (1.25, 0.5), dimensions (0.01m × 1.0m), covers y=0.0 to y=1.0
  - obstacle_top: center (1.25, 2.0), dimensions (0.01m × 1.0m), covers y=1.5 to y=2.5
  - Passage opening: y=1.0 to y=1.5 (height 0.5m)

Goal: Navigate through passage to reach target region

Strategy: Use passage-aware motion planning to navigate through opening

High-Level Plan:
1. move_to_tgt_from_passage(robot, target_region, obstacle_bottom, obstacle_top, params=[0.5, 0.5, 0.5])
   - Navigate from starting position through passage to target region
   - params=[0.5, 0.5, 0.5] specifies center of target region
   - Motion planner considers obstacle_bottom and obstacle_top to identify passage
   - Computes path that passes through opening at y≈1.25 (center of 1.0-1.5 range)
   - Robot diameter (0.2m) fits through passage opening (0.5m tall) with clearance
   - Controller executes waypoints: approach passage → pass through → reach target

Goal Reached:
- Robot successfully navigated through narrow passage ✓
- Robot position inside target region ✓
- Collision-free path maintained throughout ✓

Key Insights:
- Passage-aware controller identifies opening between obstacles
- Robot base radius (0.1m) requires passage height > 0.2m for safe navigation
- Motion planning automatically centers robot in passage for maximum clearance
- Single skill call handles entire navigation sequence

**Example 3: Multiple Passages - Sequential Navigation**

Initial State:
- Robot: position (0.3, 1.25, 0.0), base radius 0.1m
- Target Region: center (2.25, 2.0), size 0.25m × 0.25m
- Obstacles: Six rectangles forming three vertical walls with passages
  - Wall 1 at x=0.7: obstacles at y=[0.0, 0.5] and y=[1.0, 2.5], passage y=0.5 to y=1.0
  - Wall 2 at x=1.25: obstacles at y=[0.0, 1.5] and y=[2.0, 2.5], passage y=1.5 to y=2.0
  - Wall 3 at x=1.8: obstacles at y=[0.0, 1.8] and y=[2.3, 2.5], passage y=1.8 to y=2.3

Goal: Navigate through all three passages to reach target at (2.25, 2.0)

Strategy: Chain passage navigation skills sequentially

High-Level Plan:
1. move_to_passage_from_no_passage(robot, wall1_bottom, wall1_top, params=[0.5, 0.5, 0.5])
   - Navigate from starting position (0.3, 1.25) to first passage
   - params specify position within passage opening (center at y=0.75)
   - Motion planner computes path to passage entrance

2. move_to_passage_from_passage(robot, wall2_bottom, wall2_top, wall1_bottom, wall1_top, params=[0.5, 0.5, 0.5])
   - Navigate from first passage to second passage
   - Current position: past wall 1, approaching wall 2
   - Target: passage 2 center at y=1.75 (between 1.5 and 2.0)
   - Motion planning accounts for both current and target passage constraints

3. move_to_tgt_from_passage(robot, target_region, wall3_bottom, wall3_top, params=[0.5, 0.8, 0.5])
   - Navigate from second passage through third passage to target
   - params=[0.5, 0.8, 0.5]: target position at (50% width, 80% height, 0° angle)
   - Final position: (2.25, 2.0) center of target region
   - Motion planner handles third passage automatically en route to target

Goal Reached:
- Robot successfully navigated through 3 passages ✓
- Robot position inside target region ✓
- Optimal path selected based on passage alignments ✓

Key Insights:
- Multiple passages handled by chaining passage navigation skills
- Each skill considers obstacles to compute collision-free waypoints
- Params allow fine-tuning target position within passage or region
- Motion planning optimizes path through available openings

**Example 4: Passage with Non-Centered Target**

Initial State:
- Robot: position (0.3, 0.5, 0.0), base radius 0.1m
- Target Region: center (2.25, 2.0), size 0.25m × 0.25m
- Obstacles: Two rectangles forming wall at x=1.25 with passage
  - obstacle_bottom: covers y=0.0 to y=1.5
  - obstacle_top: covers y=2.0 to y=2.5
  - Passage opening: y=1.5 to y=2.0 (height 0.5m)

Goal: Navigate to target near top edge of target region

Strategy: Use parameterized position within target region

High-Level Plan:
1. move_to_tgt_from_passage(robot, target_region, obstacle_bottom, obstacle_top, params=[0.5, 0.8, 0.25])
   - Navigate through passage to specific position within target region
   - params=[0.5, 0.8, 0.25] specifies:
     - rel_x=0.5: center of target region width (x ≈ 2.25)
     - rel_y=0.8: 80% up the target region height (y ≈ 2.15)
     - rel_theta=0.25: normalized angle (-45° after denormalization)
   - Motion planner:
     - Computes path from (0.3, 0.5) to passage at y≈1.75
     - Routes through passage opening (y=1.5 to y=2.0)
     - Continues to target position (2.25, 2.15) with orientation -45°
   - Controller executes waypoints with collision avoidance

Goal Reached:
- Robot at (2.25, 2.15) inside target region ✓
- Precise positioning achieved via parameterization ✓
- Efficient path through available passage ✓

Key Insights:
- Non-centered params (e.g., 0.8 vs 0.5) allow precise positioning
- Useful for tasks requiring specific target region locations
- Motion planning handles geometry constraints automatically

**General Parameterized Skill Interpretation:**

Skills:
- move_to_tgt_from_no_passage(robot, target, params=[rel_x, rel_y, rel_theta]):
  - Direct navigation when no obstacles block path to target
  - Params specify normalized target position and orientation

- move_to_tgt_from_passage(robot, target, obstacle1, obstacle2, params=[rel_x, rel_y, rel_theta]):
  - Navigate through passage defined by obstacle1/obstacle2 to reach target
  - Planner ensures path passes through opening between obstacles

- move_to_passage_from_no_passage(robot, obstacle1, obstacle2, params=[rel_x, rel_y, rel_theta]):
  - Navigate from open space to passage entrance
  - Params specify position within passage opening

- move_to_passage_from_passage(robot, obstacle1, obstacle2, obstacle3, obstacle4, params=[rel_x, rel_y, rel_theta]):
  - Navigate from one passage to another
  - obstacle1/2 define target passage, obstacle3/4 define current passage context

Parameters:
- rel_x ∈ [0, 1]: normalized x-position within target region or passage
  - 0.0 = left edge, 0.5 = center, 1.0 = right edge
- rel_y ∈ [0, 1]: normalized y-position within target region or passage
  - 0.0 = bottom edge, 0.5 = center, 1.0 = top edge
- rel_theta ∈ [0, 1]: normalized orientation
  - Converts to absolute: abs_theta = rel_theta × 2π - π
  - 0.0 → -π, 0.5 → 0, 1.0 → π

Motion Planning:
- Each skill uses RRT-based motion planning for collision-free paths
- Plans sequences of SE2 poses (x, y, theta) as waypoints
- Controller tracks waypoints using action space [dx, dy, dtheta, darm, vac]
- Arm (darm) and vacuum (vac) not used in this environment (stay at 0)
"""

    def _create_variant_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return "The number of narrow passages differs between environment variants. For example, Motion2D-p0 has no passages (open space), while Motion2D-p5 has 5 narrow passages."

    def _create_variant_specific_description(self) -> str:
        if self._num_passages == 0:
            return "This variant has no narrow passages (open space)."
        if self._num_passages == 1:
            return "This variant has 1 narrow passage."
        return f"This variant has {self._num_passages} narrow passages."

    def _create_reward_markdown_description(self) -> str:
        return "A penalty of -1.0 is given at every time step until termination, which occurs when the robot's position is within the target region.\n"  # pylint: disable=line-too-long

    def _create_references_markdown_description(self) -> str:
        return "Narrow passages are a classic challenge in motion planning.\n"  # pylint: disable=line-too-long
