"""Environment with a stick and buttons that need to be pressed."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from relational_structs import Object, ObjectCentricState
from relational_structs.utils import create_state_from_dict

from kinder.core import ConstantObjectKinDEREnv, FinalConfigMeta
from kinder.envs.kinematic2d.base_env import (
    Kinematic2DRobotEnvConfig,
    ObjectCentricKinematic2DRobotEnv,
)
from kinder.envs.kinematic2d.object_types import (
    CircleType,
    CRVRobotType,
    Kinematic2DRobotEnvTypeFeatures,
    LObjectType,
    RectangleType,
)
from kinder.envs.kinematic2d.structs import SE2Pose, ZOrder
from kinder.envs.kinematic2d.utils import (
    CRVRobotActionSpace,
    create_walls_from_world_boundaries,
    move_objects_in_contact,
)
from kinder.envs.utils import BLACK, BROWN, sample_se2_pose, state_2d_has_collision


@dataclass(frozen=True)
class PushPullHook2DEnvConfig(Kinematic2DRobotEnvConfig, metaclass=FinalConfigMeta):
    """Config for PushPullHook2DEnv()."""

    # World boundaries. Standard coordinate frame with (0, 0) in bottom left.
    world_min_x: float = 0.0
    world_max_x: float = 3.5
    world_min_y: float = 0.0
    world_max_y: float = 2.5

    # Button boundaries.
    button_min_x: float = 1.2
    button_max_x: float = 2.4
    button_min_y: float = (world_max_y - world_min_y) / 2
    button_max_y: float = 2.0

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
    # The robot starts on the bottom (off the table).
    robot_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(
            world_min_x + 3 * robot_base_radius,
            world_min_y + 3 * robot_base_radius,
            -np.pi,
        ),
        SE2Pose(
            world_max_x - 3 * robot_base_radius,
            world_min_y + (world_max_y - world_min_y) / 2 - 3 * robot_base_radius,
            np.pi,
        ),
    )

    # Table hyperparameters.
    table_rgb: tuple[float, float, float] = BLACK
    table_pose: SE2Pose = SE2Pose(
        x=world_min_x,
        y=world_min_y + (world_max_y - world_min_y) / 2,
        theta=0,
    )
    table_shape: tuple[float, float] = (
        world_max_x - world_min_x,
        (world_max_y - world_min_y) / 2,
    )

    # Hook hyperparameters.
    hook_rgb: tuple[float, float, float] = BROWN
    hook_shape: tuple[float, float, float] = (
        robot_base_radius / 2,
        table_shape[1],
        table_shape[1] / 2,
    )
    hook_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(world_min_x, table_pose.y - hook_shape[1] / 4, np.pi / 4),
        SE2Pose(
            world_max_x - hook_shape[0], table_pose.y + hook_shape[1] / 4, 3 * np.pi / 4
        ),
    )

    # Fixed Button hyperparameters.
    target_button_unpressed_rgb: tuple[float, float, float] = (0.9, 0.0, 0.0)
    target_button_pressed_rgb: tuple[float, float, float] = (0.0, 0.9, 0.0)
    target_button_radius: float = robot_base_radius / 2
    target_button_init_position_bounds: tuple[
        tuple[float, float], tuple[float, float]
    ] = (
        (button_min_x + target_button_radius, button_min_y + target_button_radius),
        (button_max_x - target_button_radius, button_max_y - target_button_radius),
    )

    # Movable Button hyperparameters.
    movable_button_unpressed_rgb: tuple[float, float, float] = (0.0, 0.0, 0.9)
    movable_button_pressed_rgb: tuple[float, float, float] = (0.0, 0.9, 0.0)
    movable_button_radius: float = robot_base_radius / 2
    movable_button_init_position_bounds: tuple[
        tuple[float, float], tuple[float, float]
    ] = (
        (button_min_x + movable_button_radius, button_min_y + movable_button_radius),
        (button_max_x - movable_button_radius, button_max_y - movable_button_radius),
    )

    # For initial state sampling.
    max_init_sampling_attempts: int = 10_000

    # For rendering.

    render_dpi: int = 300
    render_fps: int = 20


# Object-centric environment class
class ObjectCentricPushPullHook2DEnv(
    ObjectCentricKinematic2DRobotEnv[PushPullHook2DEnvConfig]
):
    """Environment with a hook, a movable button and a target button.

    The target button is pressed only when the movable button is in contact with it.
    """

    def __init__(
        self, config: PushPullHook2DEnvConfig = PushPullHook2DEnvConfig(), **kwargs
    ) -> None:
        super().__init__(config, **kwargs)

    def _sample_initial_state(self) -> ObjectCentricState:
        # Sample initial robot pose.
        robot_pose = sample_se2_pose(self.config.robot_init_pose_bounds, self.np_random)

        # Sample hook pose.
        for _ in range(self.config.max_init_sampling_attempts):
            hook_pose = sample_se2_pose(
                self.config.hook_init_pose_bounds, self.np_random
            )
            movable_button_pose = tuple(
                self.np_random.uniform(*self.config.movable_button_init_position_bounds)
            )
            target_button_pose = tuple(
                self.np_random.uniform(*self.config.target_button_init_position_bounds)
            )
            state = self._create_initial_state(
                robot_pose,
                hook_pose=hook_pose,
                movable_button_pose=movable_button_pose,
                target_button_pose=target_button_pose,
            )
            obj_name_to_obj = {o.name: o for o in state}
            hook = obj_name_to_obj["hook"]
            movable_button = obj_name_to_obj["movable_button"]
            target_button = obj_name_to_obj["target_button"]

            dist_movable_button = np.linalg.norm(
                np.array(
                    [
                        state.get(target_button, "x") - state.get(movable_button, "x"),
                        state.get(target_button, "y") - state.get(movable_button, "y"),
                    ]
                )
            )

            full_state = state.copy()
            full_state.data.update(self.initial_constant_state.data)
            if (
                not state_2d_has_collision(
                    full_state,
                    {hook, movable_button, target_button},
                    set(full_state),
                    {},
                )
                and 3 * self.config.movable_button_radius
                < dist_movable_button
                < 6 * self.config.movable_button_radius
            ):
                break
        else:
            raise RuntimeError("Failed to sample valid poses for all objects.")

        # Recreate state now with no-collision buttons.
        state = self._create_initial_state(
            robot_pose,
            hook_pose=hook_pose,
            movable_button_pose=movable_button_pose,
            target_button_pose=target_button_pose,
            target_button_z_order=ZOrder.NONE,
        )
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

        # Create the table.
        table = Object("table", RectangleType)
        init_state_dict[table] = {
            "x": self.config.table_pose.x,
            "y": self.config.table_pose.y,
            "theta": self.config.table_pose.theta,
            "width": self.config.table_shape[0],
            "height": self.config.table_shape[1],
            "static": True,
            "color_r": self.config.table_rgb[0],
            "color_g": self.config.table_rgb[1],
            "color_b": self.config.table_rgb[2],
            "z_order": ZOrder.FLOOR.value,
        }

        return init_state_dict

    def _create_initial_state(
        self,
        robot_pose: SE2Pose,
        hook_pose: SE2Pose,
        movable_button_pose: tuple[float, float],
        target_button_pose: tuple[float, float],
        target_button_z_order: ZOrder = ZOrder.NONE,
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

        # Create the hook.
        hook = Object("hook", LObjectType)
        init_state_dict[hook] = {
            "x": hook_pose.x,
            "y": hook_pose.y,
            "theta": hook_pose.theta,
            "width": self.config.hook_shape[0],
            "length_side1": self.config.hook_shape[1],
            "length_side2": self.config.hook_shape[2],
            "static": False,
            "color_r": self.config.hook_rgb[0],
            "color_g": self.config.hook_rgb[1],
            "color_b": self.config.hook_rgb[2],
            "z_order": ZOrder.SURFACE.value,
        }

        # Create the buttons.
        if len(movable_button_pose) > 0:
            movable_button = Object("movable_button", CircleType)
            init_state_dict[movable_button] = {
                "x": movable_button_pose[0],
                "y": movable_button_pose[1],
                "theta": 0,
                "radius": self.config.movable_button_radius,
                "static": False,
                "color_r": self.config.movable_button_unpressed_rgb[0],
                "color_g": self.config.movable_button_unpressed_rgb[1],
                "color_b": self.config.movable_button_unpressed_rgb[2],
                "z_order": ZOrder.SURFACE.value,
            }

        if len(target_button_pose) > 0:
            target_button = Object("target_button", CircleType)
            init_state_dict[target_button] = {
                "x": target_button_pose[0],
                "y": target_button_pose[1],
                "theta": 0,
                "radius": self.config.target_button_radius,
                "static": True,
                "color_r": self.config.target_button_unpressed_rgb[0],
                "color_g": self.config.target_button_unpressed_rgb[1],
                "color_b": self.config.target_button_unpressed_rgb[2],
                "z_order": target_button_z_order.value,
            }

        # Finalize state.
        return create_state_from_dict(init_state_dict, Kinematic2DRobotEnvTypeFeatures)

    def press_button(self, button: Object) -> ObjectCentricState:
        """Press a button by changing its color."""
        assert self._current_state is not None
        if button.name == "movable_button":
            self._current_state.set(
                button, "color_r", self.config.movable_button_pressed_rgb[0]
            )
            self._current_state.set(
                button, "color_g", self.config.movable_button_pressed_rgb[1]
            )
            self._current_state.set(
                button, "color_b", self.config.movable_button_pressed_rgb[2]
            )
        elif button.name == "target_button":
            self._current_state.set(
                button, "color_r", self.config.target_button_pressed_rgb[0]
            )
            self._current_state.set(
                button, "color_g", self.config.target_button_pressed_rgb[1]
            )
            self._current_state.set(
                button, "color_b", self.config.target_button_pressed_rgb[2]
            )
            del self._static_object_body_cache[button]
        return self._current_state

    def release_button(self, button: Object) -> ObjectCentricState:
        """Release a button by changing its color back."""
        assert self._current_state is not None
        if button.name == "movable_button":
            self._current_state.set(
                button, "color_r", self.config.movable_button_unpressed_rgb[0]
            )
            self._current_state.set(
                button, "color_g", self.config.movable_button_unpressed_rgb[1]
            )
            self._current_state.set(
                button, "color_b", self.config.movable_button_unpressed_rgb[2]
            )
        elif button.name == "target_button":
            self._current_state.set(
                button, "color_r", self.config.target_button_unpressed_rgb[0]
            )
            self._current_state.set(
                button, "color_g", self.config.target_button_unpressed_rgb[1]
            )
            self._current_state.set(
                button, "color_b", self.config.target_button_unpressed_rgb[2]
            )
            del self._static_object_body_cache[button]
        return self._current_state

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[ObjectCentricState, float, bool, bool, dict]:
        super().step(action)
        assert self._current_state is not None
        assert self.initial_constant_state is not None
        obj_name_to_obj = {o.name: o for o in self._current_state}
        movable_button = obj_name_to_obj["movable_button"]
        target_button = obj_name_to_obj["target_button"]

        # Check if movable button is in contact with target button (success)
        button_to_target = np.array(
            [
                self._current_state.get(target_button, "x")
                - self._current_state.get(movable_button, "x"),
                self._current_state.get(target_button, "y")
                - self._current_state.get(movable_button, "y"),
            ]
        )
        dist = np.linalg.norm(button_to_target)
        success = dist < self.config.target_button_radius * 2
        if success:
            self.press_button(movable_button)
            self.press_button(target_button)

        reward, terminated = self._get_reward_and_done()
        truncated = False
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def get_objects_to_move(
        self,
        state: ObjectCentricState,
        suctioned_objs: list[tuple[Object, SE2Pose]],
    ) -> tuple[ObjectCentricState, set[tuple[Object, SE2Pose]]]:
        """Get the set of objects that should be moved based on the current state and
        robot actions."""
        robots = [o for o in state if o.is_instance(CRVRobotType)]
        assert len(robots) == 1, "Multi-robot not yet supported"
        robot = robots[0]

        state, moved_objects = move_objects_in_contact(state, robot, suctioned_objs)
        return state, moved_objects

    def _get_reward_and_done(self) -> tuple[float, bool]:
        # Success if both buttons are pressed
        terminated = True
        assert self._current_state is not None

        obj_name_to_obj = {o.name: o for o in self._current_state}
        movable_button = obj_name_to_obj["movable_button"]
        target_button = obj_name_to_obj["target_button"]
        movable_color = (
            self._current_state.get(movable_button, "color_r"),
            self._current_state.get(movable_button, "color_g"),
            self._current_state.get(movable_button, "color_b"),
        )
        target_color = (
            self._current_state.get(target_button, "color_r"),
            self._current_state.get(target_button, "color_g"),
            self._current_state.get(target_button, "color_b"),
        )
        terminated = np.allclose(
            movable_color, self.config.movable_button_pressed_rgb
        ) and np.allclose(target_color, self.config.target_button_pressed_rgb)
        reward = 0.0 if terminated else -1.0
        return reward, terminated


# Main env class
class PushPullHook2DEnv(ConstantObjectKinDEREnv):
    """Push-pull hook 2D env with a constant number of objects."""

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricKinematic2DRobotEnv:
        return ObjectCentricPushPullHook2DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        constant_objects = ["robot", "hook", "movable_button", "target_button"]
        return constant_objects

    def _create_env_markdown_description(self) -> str:
        return (
            "A 2D environment with a robot, a hook (L-shape), a movable button, "
            "and a target button. "
            "The goal is to move the movable button onto the target button. "
            "The robot has a movable circular base and a retractable arm with a rectangular vacuum end effector. " # pylint: disable=line-too-long
            "Objects can be grasped and ungrasped when the end effector makes contact."
        )

    def _create_variant_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return "This environment has only one variant."

    def _create_variant_specific_description(self) -> str:
        return "This variant has one hook, one movable button, and one target button."

    def _create_reward_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return (
            "A penalty of -1.0 is given at every time step until both the movable button "
            "and the target button are pressed (i.e., in contact and colored green, termination)."
        )

    def _create_references_markdown_description(self) -> str:
        return (
            "This environment is inspired by StickButton2DEnv but uses a "
            "hook and push-pull mechanics."
        )
