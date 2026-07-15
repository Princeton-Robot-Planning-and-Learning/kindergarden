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
    RectangleType,
)
from kinder.envs.kinematic2d.structs import SE2Pose, ZOrder
from kinder.envs.kinematic2d.utils import (
    CRVRobotActionSpace,
    create_walls_from_world_boundaries,
)
from kinder.envs.utils import BLACK, BROWN, sample_se2_pose, state_2d_has_collision


@dataclass(frozen=True)
class StickButton2DEnvConfig(Kinematic2DRobotEnvConfig, metaclass=FinalConfigMeta):
    """Config for StickButton2DEnv()."""

    # World boundaries. Standard coordinate frame with (0, 0) in bottom left.
    world_min_x: float = 0.0
    world_max_x: float = 3.5
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

    # Stick hyperparameters.
    stick_rgb: tuple[float, float, float] = BROWN
    stick_shape: tuple[float, float] = (robot_base_radius / 2, table_shape[1])
    stick_init_pose_bounds: tuple[SE2Pose, SE2Pose] = (
        SE2Pose(world_min_x, table_pose.y - stick_shape[1] / 2, 0),
        SE2Pose(world_max_x - stick_shape[0], table_pose.y - stick_shape[1] / 10, 0),
    )

    # Button hyperparameters.
    button_unpressed_rgb: tuple[float, float, float] = (0.9, 0.0, 0.0)
    button_pressed_rgb: tuple[float, float, float] = (0.0, 0.9, 0.0)
    button_radius: float = robot_base_radius / 2
    button_init_position_bounds: tuple[tuple[float, float], tuple[float, float]] = (
        (world_min_x + button_radius, world_min_y + button_radius),
        (world_max_x - button_radius, world_max_y - button_radius),
    )

    # For initial state sampling.
    max_init_sampling_attempts: int = 10_000

    # For rendering.
    render_dpi: int = 300
    render_fps: int = 20


class ObjectCentricStickButton2DEnv(
    ObjectCentricKinematic2DRobotEnv[StickButton2DEnvConfig]
):
    """Environment with a stick and buttons that need to be pressed.

    The robot cannot directly press buttons that are on the table but can directly press
    buttons that are on the floor (by touching them).

    The stick can be used to press buttons on the table (by touch).

    This is an object-centric environment. The vectorized version with Box spaces is
    defined below.
    """

    def __init__(
        self,
        num_buttons: int = 2,
        config: StickButton2DEnvConfig = StickButton2DEnvConfig(),
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        self._num_buttons = num_buttons

    def _sample_initial_state(self) -> ObjectCentricState:
        # Sample initial robot pose.
        robot_pose = sample_se2_pose(self.config.robot_init_pose_bounds, self.np_random)
        # Sample stick pose.
        for _ in range(self.config.max_init_sampling_attempts):
            stick_pose = sample_se2_pose(
                self.config.stick_init_pose_bounds, self.np_random
            )
            state = self._create_initial_state(
                robot_pose,
                stick_pose=stick_pose,
                button_positions=[],
            )
            obj_name_to_obj = {o.name: o for o in state}
            stick = obj_name_to_obj["stick"]
            full_state = state.copy()
            full_state.data.update(self.initial_constant_state.data)
            if not state_2d_has_collision(full_state, {stick}, set(full_state), {}):
                break
        else:
            raise RuntimeError("Failed to sample target pose.")

        # Sample button positions. Assume that the scene is never so dense
        # that we need to resample earlier choices.
        button_positions: list[tuple[float, float]] = []
        for _ in range(self._num_buttons):
            while True:
                button_position = tuple(
                    self.np_random.uniform(*self.config.button_init_position_bounds)
                )
                new_button_positions = button_positions + [button_position]
                state = self._create_initial_state(
                    robot_pose,
                    stick_pose=stick_pose,
                    button_positions=new_button_positions,
                    button_z_order=ZOrder.SURFACE,
                )
                obj_name_to_obj = {o.name: o for o in state}
                new_button = obj_name_to_obj[f"button{len(button_positions)}"]
                full_state = state.copy()
                full_state.data.update(self.initial_constant_state.data)
                if not state_2d_has_collision(
                    full_state, {new_button}, set(full_state), {}
                ):
                    button_positions.append(button_position)
                    break

        # Recreate state now with no-collision buttons.
        state = self._create_initial_state(
            robot_pose,
            stick_pose=stick_pose,
            button_positions=button_positions,
            button_z_order=ZOrder.NONE,
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
        stick_pose: SE2Pose,
        button_positions: list[tuple[float, float]],
        button_z_order: ZOrder = ZOrder.NONE,
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

        # Create the stick.
        stick = Object("stick", RectangleType)
        init_state_dict[stick] = {
            "x": stick_pose.x,
            "y": stick_pose.y,
            "theta": stick_pose.theta,
            "width": self.config.stick_shape[0],
            "height": self.config.stick_shape[1],
            "static": False,
            "color_r": self.config.stick_rgb[0],
            "color_g": self.config.stick_rgb[1],
            "color_b": self.config.stick_rgb[2],
            "z_order": ZOrder.SURFACE.value,
        }

        # Create the buttons.
        for button_idx, button_position in enumerate(button_positions):
            button = Object(f"button{button_idx}", CircleType)
            init_state_dict[button] = {
                "x": button_position[0],
                "y": button_position[1],
                "theta": 0,
                "radius": self.config.button_radius,
                "static": True,
                "color_r": self.config.button_unpressed_rgb[0],
                "color_g": self.config.button_unpressed_rgb[1],
                "color_b": self.config.button_unpressed_rgb[2],
                "z_order": button_z_order.value,
            }

        # Finalize state.
        return create_state_from_dict(init_state_dict, Kinematic2DRobotEnvTypeFeatures)

    def press_button(self, button: Object) -> ObjectCentricState:
        """Press a button by changing its color."""
        assert self._current_state is not None
        self._current_state.set(button, "color_r", self.config.button_pressed_rgb[0])
        self._current_state.set(button, "color_g", self.config.button_pressed_rgb[1])
        self._current_state.set(button, "color_b", self.config.button_pressed_rgb[2])
        return self._current_state

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[ObjectCentricState, float, bool, bool, dict]:
        # For any button in contact with either the robot or the stick, change
        # color to pressed.
        super().step(action)
        assert self._current_state is not None
        assert self.initial_constant_state is not None
        newly_pressed_buttons: set[Object] = set()
        obj_name_to_obj = {o.name: o for o in self._current_state}
        robot = obj_name_to_obj["robot"]
        stick = obj_name_to_obj["stick"]
        full_state = self._current_state.copy()
        full_state.data.update(self.initial_constant_state.data)
        for button in self._current_state.get_objects(CircleType):
            if state_2d_has_collision(
                full_state,
                {button},
                {robot, stick},
                self._static_object_body_cache,
                ignore_z_orders=True,
            ):
                newly_pressed_buttons.add(button)
        # Change colors.
        for button in newly_pressed_buttons:
            self.press_button(button)
            # This is hacky, but it's the easiest way to force re-rendering.
            del self._static_object_body_cache[button]

        reward, terminated = self._get_reward_and_done()
        truncated = False  # no maximum horizon, by default
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def _get_reward_and_done(self) -> tuple[float, bool]:
        terminated = True
        assert self._current_state is not None
        for button in self._current_state.get_objects(CircleType):
            color = (
                self._current_state.get(button, "color_r"),
                self._current_state.get(button, "color_g"),
                self._current_state.get(button, "color_b"),
            )
            if not np.allclose(color, self.config.button_pressed_rgb):
                terminated = False
                break
        reward = 0.0 if terminated else -1.0
        return reward, terminated


class StickButton2DEnv(ConstantObjectKinDEREnv):
    """Stick button 2D env with a constant number of objects."""

    def __init__(self, num_buttons: int = 3, **kwargs) -> None:
        self._num_buttons = num_buttons
        super().__init__(num_buttons=num_buttons, **kwargs)

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricKinematic2DRobotEnv:
        return ObjectCentricStickButton2DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        constant_objects = ["robot", "stick"]
        for obj in sorted(exemplar_state):
            if obj.name.startswith("button"):
                constant_objects.append(obj.name)
        return constant_objects

    def _create_env_markdown_description(self) -> str:
        num_buttons = len(self._constant_objects) - 2
        # pylint: disable=line-too-long
        return f"""A 2D environment where the goal is to touch all buttons.

In this environment, there are always {num_buttons} buttons.

The robot has a movable circular base and a retractable arm with a rectangular vacuum end effector. Objects can be grasped and ungrasped when the end effector makes contact.
"""

    def _create_variant_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return "The number of buttons differs between environment variants. For example, StickButton2D-b1 has 1 button, while StickButton2D-b10 has 10 buttons."

    def _create_variant_specific_description(self) -> str:
        if self._num_buttons == 1:
            return "This variant has 1 button to press."
        return f"This variant has {self._num_buttons} buttons to press."

    def _create_reward_markdown_description(self) -> str:
        return "A penalty of -1.0 is given at every time step until all buttons have been pressed (termination).\n"  # pylint: disable=line-too-long

    def _create_references_markdown_description(self) -> str:
        return 'This environment is based on the Stick Button environment that was originally introduced in "Learning Neuro-Symbolic Skills for Bilevel Planning" (Silver et al., CoRL 2022). This version is simplified in that the robot or stick need only make contact with a button to press it, rather than explicitly pressing. Also, the full stick works for pressing, not just the tip.\n'  # pylint: disable=line-too-long
