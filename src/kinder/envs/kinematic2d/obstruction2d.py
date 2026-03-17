"""Obstruction 2D env."""

import inspect
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
from kinder.envs.kinematic2d.structs import MultiBody2D, SE2Pose, ZOrder
from kinder.envs.kinematic2d.utils import (
    CRVRobotActionSpace,
    create_walls_from_world_boundaries,
    is_inside,
    is_on,
)
from kinder.envs.utils import PURPLE, sample_se2_pose, state_2d_has_collision

TargetBlockType = Type("target_block", parent=RectangleType)
TargetSurfaceType = Type("target_surface", parent=RectangleType)
Kinematic2DRobotEnvTypeFeatures[TargetBlockType] = list(
    Kinematic2DRobotEnvTypeFeatures[RectangleType]
)
Kinematic2DRobotEnvTypeFeatures[TargetSurfaceType] = list(
    Kinematic2DRobotEnvTypeFeatures[RectangleType]
)


@dataclass(frozen=True)
class Obstruction2DEnvConfig(Kinematic2DRobotEnvConfig, metaclass=FinalConfigMeta):
    """Config for Obstruction2DEnv()."""

    # World boundaries. Standard coordinate frame with (0, 0) in bottom left.
    world_min_x: float = 0.0
    world_max_x: float = (1 + np.sqrt(5)) / 2  # golden ratio :)
    world_min_y: float = 0.0
    world_max_y: float = 1.0

    # Action space parameters.
    min_dx: float = -5e-2
    max_dx: float = 5e-2
    min_dy: float = -5e-2
    max_dy: float = 5e-2
    min_dtheta: float = -np.pi / 16
    max_dtheta: float = np.pi / 16
    min_darm: float = -1e-1
    max_darm: float = 1e-1
    min_vac: float = 0.0
    max_vac: float = 1.0

    # Robot hyperparameters.
    robot_base_radius: float = 0.1
    robot_arm_length: float = 2 * robot_base_radius
    robot_gripper_height: float = 0.07
    robot_gripper_width: float = 0.01
    robot_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(
            world_min_x + robot_base_radius,
            world_max_y - 2 * robot_base_radius,
            -np.pi / 2,
        ),
        SE2Pose(
            world_max_x - robot_base_radius, world_max_y - robot_base_radius, -np.pi / 2
        ),
    )

    # Table hyperparameters.
    table_rgb: tuple[float, float, float] = (0.75, 0.75, 0.75)
    table_height: float = 0.1
    table_width: float = world_max_x - world_min_x
    # The table pose is defined relative to the bottom left hand corner.
    table_pose: SE2Pose = SE2Pose(world_min_x, world_min_y, 0.0)

    # Target surface hyperparameters.
    target_surface_rgb: tuple[float, float, float] = PURPLE
    target_surface_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(world_min_x + robot_base_radius, table_pose.y, 0.0),
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
        robot_base_radius / 2,
        2 * robot_base_radius,
    )
    target_block_width_bounds: tuple[float, float] = (
        robot_base_radius / 2,
        2 * robot_base_radius,
    )

    # Obstruction hyperparameters.
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
        robot_base_radius / 2,
        2 * robot_base_radius,
    )
    obstruction_width_bounds: tuple[float, float] = (
        robot_base_radius / 2,
        2 * robot_base_radius,
    )
    # NOTE: this is not the "real" probability, but rather, the probability
    # that we will attempt to sample the obstruction somewhere on the target
    # surface during each round of rejection sampling during reset().
    obstruction_init_on_target_prob: float = 0.9

    # For sampling initial states.
    max_initial_state_sampling_attempts: int = 10_000

    # For rendering.
    render_dpi: int = 250


class ObjectCentricObstruction2DEnv(
    ObjectCentricKinematic2DRobotEnv[Obstruction2DEnvConfig]
):
    """Environment where a block must be placed on an obstructed target."""

    def __init__(
        self,
        num_obstructions: int = 2,
        config: Obstruction2DEnvConfig = Obstruction2DEnvConfig(),
        **kwargs,
    ) -> None:
        if num_obstructions < 0:
            raise ValueError(
                f"num_obstructions must be non-negative, got {num_obstructions}"
            )
        super().__init__(config, **kwargs)
        self._num_obstructions = num_obstructions

    def _sample_initial_state(self) -> ObjectCentricState:
        static_objects = set(self.initial_constant_state)
        assert not state_2d_has_collision(
            self.initial_constant_state,
            static_objects,
            static_objects,
            {},
        )
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
            if state_2d_has_collision(full_state, all_objects, all_objects, {}):
                continue
            if self._surface_outside_table(full_state, {}):
                continue
            return state
        raise RuntimeError(f"Failed to sample initial state after {n} attempts")

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        init_state_dict: dict[Object, dict[str, float]] = {}

        # Create the table.
        table = Object("table", RectangleType)
        init_state_dict[table] = {
            "x": self.config.table_pose.x,
            "y": self.config.table_pose.y,
            "theta": self.config.table_pose.theta,
            "width": self.config.table_width,
            "height": self.config.table_height,
            "static": True,
            "color_r": self.config.table_rgb[0],
            "color_g": self.config.table_rgb[1],
            "color_b": self.config.table_rgb[2],
            "z_order": ZOrder.ALL.value,
        }

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

        # Create the target surface.
        target_surface = Object("target_surface", TargetSurfaceType)
        init_state_dict[target_surface] = {
            "x": target_surface_pose.x,
            "y": target_surface_pose.y,
            "theta": target_surface_pose.theta,
            "width": target_surface_shape[0],
            "height": target_surface_shape[1],
            "static": True,
            "color_r": self.config.target_surface_rgb[0],
            "color_g": self.config.target_surface_rgb[1],
            "color_b": self.config.target_surface_rgb[2],
            "z_order": ZOrder.NONE.value,
        }

        # Create the target block.
        target_block = Object("target_block", TargetBlockType)
        init_state_dict[target_block] = {
            "x": target_block_pose.x,
            "y": target_block_pose.y,
            "theta": target_block_pose.theta,
            "width": target_block_shape[0],
            "height": target_block_shape[1],
            "static": False,
            "color_r": self.config.target_block_rgb[0],
            "color_g": self.config.target_block_rgb[1],
            "color_b": self.config.target_block_rgb[2],
            "z_order": ZOrder.ALL.value,
        }

        # Create obstructions.
        for i, (obstruction_pose, obstruction_shape) in enumerate(obstructions):
            obstruction = Object(f"obstruction{i}", RectangleType)
            init_state_dict[obstruction] = {
                "x": obstruction_pose.x,
                "y": obstruction_pose.y,
                "theta": obstruction_pose.theta,
                "width": obstruction_shape[0],
                "height": obstruction_shape[1],
                "static": False,
                "color_r": self.config.obstruction_rgb[0],
                "color_g": self.config.obstruction_rgb[1],
                "color_b": self.config.obstruction_rgb[2],
                "z_order": ZOrder.ALL.value,
            }

        # Finalize state.
        return create_state_from_dict(init_state_dict, Kinematic2DRobotEnvTypeFeatures)

    def _target_satisfied(
        self,
        state: ObjectCentricState,
        static_object_body_cache: dict[Object, MultiBody2D],
    ) -> bool:
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
        table = state.get_objects(RectangleType)
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
        # Terminate when target object is on the target surface. Give -1 reward
        # at every step until then to encourage fast completion.
        assert self._current_state is not None
        terminated = self._target_satisfied(
            self._current_state,
            self._static_object_body_cache,
        )
        return -1.0, terminated


class Obstruction2DEnv(ConstantObjectKinDEREnv):
    """Obstruction 2D env with a constant number of objects."""

    def __init__(self, num_obstructions: int = 2, **kwargs) -> None:
        self._num_obstructions = num_obstructions
        super().__init__(num_obstructions=num_obstructions, **kwargs)

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricKinematic2DRobotEnv:
        return ObjectCentricObstruction2DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        constant_objects = ["robot", "target_surface", "target_block"]
        for obj in sorted(exemplar_state):
            if obj.name.startswith("obstruct"):
                constant_objects.append(obj.name)
        return constant_objects

    def _create_env_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return """A 2D environment where the goal is to place a target block onto a target surface. The block must be completely contained within the surface boundaries.

The target surface may be initially obstructed.

The robot has a movable circular base and a retractable arm with a rectangular vacuum end effector. Objects can be grasped and ungrasped when the end effector makes contact.
"""

    def _create_variant_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return "The number of obstructions differs between environment variants. For example, Obstruction2D-o0 has no obstructions, while Obstruction2D-o4 has 4 obstructions."

    def _create_variant_specific_description(self) -> str:
        if self._num_obstructions == 0:
            return "This variant has no obstructions."
        if self._num_obstructions == 1:
            return "This variant has 1 obstruction."
        return f"This variant has {self._num_obstructions} obstructions."

    def _create_reward_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return f"""A penalty of -1.0 is given at every time step until termination, which occurs when the target block is "on" the target surface. The definition of "on" is given below:
```python
{inspect.getsource(is_on)}```
"""

    def _create_references_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return """Similar environments have been used many times, especially in the task and motion planning literature. We took inspiration especially from the "1D Continuous TAMP" environment in [PDDLStream](https://github.com/caelan/pddlstream).
"""
