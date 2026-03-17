"""Environment where a block must be retrieved amidst clutter."""

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
    is_inside,
)
from kinder.envs.utils import BROWN, PURPLE, sample_se2_pose, state_2d_has_collision

TargetBlockType = Type("target_block", parent=RectangleType)
TargetRegionType = Type("target_region", parent=RectangleType)
Kinematic2DRobotEnvTypeFeatures[TargetBlockType] = list(
    Kinematic2DRobotEnvTypeFeatures[RectangleType]
)
Kinematic2DRobotEnvTypeFeatures[TargetRegionType] = list(
    Kinematic2DRobotEnvTypeFeatures[RectangleType]
)


@dataclass(frozen=True)
class ClutteredRetrieval2DEnvConfig(
    Kinematic2DRobotEnvConfig, metaclass=FinalConfigMeta
):
    """Config for ClutteredRetrieval2DEnv()."""

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
            world_min_x + 4 * robot_base_radius,
            world_min_y + 4 * robot_base_radius,
            -np.pi,
        ),
        SE2Pose(
            world_max_x - 4 * robot_base_radius,
            world_max_y - 4 * robot_base_radius,
            np.pi,
        ),
    )

    # Target block hyperparameters.
    target_block_rgb: tuple[float, float, float] = PURPLE
    target_block_init_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(
            world_min_x + robot_base_radius,
            world_min_y + robot_base_radius,
            -np.pi,
        ),
        SE2Pose(
            world_max_x - robot_base_radius,
            world_max_y - robot_base_radius,
            np.pi,
        ),
    )
    target_block_shape: tuple[float, float] = (
        2 * robot_gripper_height,
        2 * robot_gripper_height,
    )

    # Target region hyperparameters.
    target_region_rgb: tuple[float, float, float] = BROWN
    target_region_init_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(
            world_min_x + robot_base_radius,
            world_min_y + robot_base_radius,
            -np.pi,
        ),
        SE2Pose(
            world_max_x - robot_base_radius,
            world_max_y - robot_base_radius,
            np.pi,
        ),
    )
    target_region_shape: tuple[float, float] = (
        3 * robot_gripper_height,
        3 * robot_gripper_height,
    )

    # Obstruction hyperparameters.
    obstruction_rgb: tuple[float, float, float] = (0.75, 0.1, 0.1)

    obstruction_height_bounds: tuple[float, float] = (
        robot_base_radius / 2,
        2 * robot_base_radius,
    )
    obstruction_width_bounds: tuple[float, float] = (
        robot_base_radius / 2,
        2 * robot_base_radius,
    )
    # NOTE: obstruction poses are sampled using a 2D gaussian that is centered
    # at the target location. This hyperparameter controls the variance.
    obstruction_pose_init_distance_scale: float = 0.25

    # For initial state sampling.
    max_init_sampling_attempts: int = 10_000

    # For rendering.
    render_dpi: int = 300
    render_fps: int = 20


class ObjectCentricClutteredRetrieval2DEnv(
    ObjectCentricKinematic2DRobotEnv[ClutteredRetrieval2DEnvConfig]
):
    """Environment where a block must be retrieved amidst clutter.

    This is an object-centric environment. The vectorized version with Box spaces is
    defined below.
    """

    def __init__(
        self,
        num_obstructions: int = 2,
        config: ClutteredRetrieval2DEnvConfig = ClutteredRetrieval2DEnvConfig(),
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
        robot_pose = sample_se2_pose(self.config.robot_init_pose_bounds, self.np_random)
        state = self._create_initial_state(robot_pose)
        robot = state.get_objects(CRVRobotType)[0]
        # Check for collisions with the robot and static objects.
        full_state = state.copy()
        full_state.data.update(self.initial_constant_state.data)
        assert not state_2d_has_collision(full_state, {robot}, static_objects, {})
        # Sample target pose and check for collisions with robot and static objects.
        for _ in range(self.config.max_init_sampling_attempts):
            target_pose = sample_se2_pose(
                self.config.target_block_init_bounds, self.np_random
            )
            target_region_pose = sample_se2_pose(
                self.config.target_region_init_bounds, self.np_random
            )
            state = self._create_initial_state(
                robot_pose,
                target_pose=target_pose,
                target_region_pose=target_region_pose,
            )
            target_block = state.get_objects(TargetBlockType)[0]
            target_region = state.get_objects(TargetRegionType)[0]
            full_state = state.copy()
            full_state.data.update(self.initial_constant_state.data)
            # Check that target_region doesn't collide with robot or static objects.
            if state_2d_has_collision(
                full_state, {target_region}, {robot} | static_objects, {}
            ):
                continue
            # Check that target_block doesn't collide with obstacles.
            if not state_2d_has_collision(
                full_state, {target_block}, {robot, target_region} | static_objects, {}
            ):
                break
        else:
            raise RuntimeError("Failed to sample target pose.")
        # Sample obstructions one by one. Assume that the scene is never so dense
        # that we need to resample earlier choices.
        obstructions: list[tuple[SE2Pose, tuple[float, float]]] = []
        for _ in range(self._num_obstructions):
            for _ in range(self.config.max_init_sampling_attempts):
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
                    target_region_pose=target_region_pose,
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
        target_pose: SE2Pose | None = None,
        target_region_pose: SE2Pose | None = None,
        obstructions: list[tuple[SE2Pose, tuple[float, float]]] | None = None,
    ) -> ObjectCentricState:
        # Shallow copy should be okay because the constant objects should not
        # ever change in this method.
        assert self.initial_constant_state is not None
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

        # Create the target block.
        if target_pose is not None:
            target_block = Object("target_block", TargetBlockType)
            init_state_dict[target_block] = {
                "x": target_pose.x,
                "y": target_pose.y,
                "theta": target_pose.theta,
                "width": self.config.target_block_shape[0],
                "height": self.config.target_block_shape[1],
                "static": False,
                "color_r": self.config.target_block_rgb[0],
                "color_g": self.config.target_block_rgb[1],
                "color_b": self.config.target_block_rgb[2],
                "z_order": ZOrder.SURFACE.value,
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
                "static": True,
                "color_r": self.config.target_region_rgb[0],
                "color_g": self.config.target_region_rgb[1],
                "color_b": self.config.target_region_rgb[2],
                "z_order": ZOrder.NONE.value,
            }

        # Create the obstructions.
        if obstructions:
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

    def _get_reward_and_done(self) -> tuple[float, bool]:
        # Terminate when the target block is inside the target region.
        assert self._current_state is not None
        target_object = self._current_state.get_objects(TargetBlockType)[0]
        target_region = self._current_state.get_objects(TargetRegionType)[0]
        terminated = is_inside(
            self._current_state,
            target_object,
            target_region,
            self._static_object_body_cache,
        )
        return -1.0, terminated


class ClutteredRetrieval2DEnv(ConstantObjectKinDEREnv):
    """Cluttered retrieval 2D env with a constant number of objects."""

    def __init__(self, num_obstructions: int = 5, **kwargs) -> None:
        self._num_obstructions = num_obstructions
        super().__init__(num_obstructions=num_obstructions, **kwargs)

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricKinematic2DRobotEnv:
        return ObjectCentricClutteredRetrieval2DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        constant_objects = ["robot", "target_block", "target_region"]
        for obj in sorted(exemplar_state):
            if obj.name.startswith("obstruction"):
                constant_objects.append(obj.name)
        return constant_objects

    def _create_env_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return """A 2D environment where the goal is to retrieve a target block and place it inside a target region.

The target block may be initially obstructed.

The robot has a movable circular base and a retractable arm with a rectangular vacuum end effector. Objects can be grasped and ungrasped when the end effector makes contact.
"""

    def _create_variant_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return "The number of obstructions differs between environment variants. For example, ClutteredRetrieval2D-o5 has 5 obstructions."

    def _create_variant_specific_description(self) -> str:
        if self._num_obstructions == 0:
            return "This variant has no obstructions."
        if self._num_obstructions == 1:
            return "This variant has 1 obstruction."
        return f"This variant has {self._num_obstructions} obstructions."

    def _create_reward_markdown_description(self) -> str:
        return "A penalty of -1.0 is given at every time step until termination, which occurs when the target block is inside the target region.\n"  # pylint: disable=line-too-long

    def _create_references_markdown_description(self) -> str:
        return 'Similar environments have been considered by many others, especially in the task and motion planning literature, e.g., "Combined Task and Motion Planning Through an Extensible Planner-Independent Interface Layer" (Srivastava et al., ICRA 2014).\n'  # pylint: disable=line-too-long
