"""A parallel (batch) Sprout environment that runs on MuJoCo Warp.

``Sprout3DBatchEnv`` steps many copies ("worlds") of one Sprout task at the same time,
on the GPU. ``Dynamic3DBatchEnv`` (in ``batch_env.py``) has the parts that are the same
for all robots. This module adds the parts of the Sprout: the motor model, the command
delay, the hold of the robot in the settle time, and the fall check.

The single-world ``Sprout3DEnv`` stays the reference: one world of the batch environment
is the same task as ``Sprout3DEnv``.

This module needs ``mujoco_warp``. See ``requirements/dynamic3d_warp.txt``.
"""

# The batch environment reads the internals of the single-world environment that it
# is built on.
# pylint: disable=protected-access

# The Warp kernels use Warp types as annotations, which mypy cannot check.
# mypy: disable-error-code="valid-type,misc,no-untyped-call,var-annotated,assignment"

from typing import Any

import mujoco_warp as mjw
import numpy as np
import warp as wp

from kinder.envs.dynamic3d.batch_env import Dynamic3DBatchEnv, ObservationMapEntry
from kinder.envs.dynamic3d.envs import Sprout3DEnv
from kinder.envs.dynamic3d.robots.sprout_robot_env import (
    NUM_SPROUT_JOINTS,
    SPROUT_SETTLE_STEPS,
    SproutRobotEnv,
)

# Feature names of the robot base, in the order of the qpos and qvel of its free joint.
_BASE_POS_FEATURES = [f"pos_base_{n}" for n in ["x", "y", "z", "qw", "qx", "qy", "qz"]]
_BASE_VEL_FEATURES = [f"vel_base_{n}" for n in ["x", "y", "z", "wx", "wy", "wz"]]


@wp.kernel
def _dc_motor_kernel(
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    targets: wp.array2d(dtype=float),
    joint_qpos_adr: int,
    joint_qvel_adr: int,
    joint_ctrl_adr: int,
    stiffness: wp.array(dtype=float),
    damping: wp.array(dtype=float),
    effort_limit: wp.array(dtype=float),
    saturation_effort: wp.array(dtype=float),
    velocity_limit: wp.array(dtype=float),
    ctrl: wp.array2d(dtype=float),
):
    """The DC motor model: the same law as ``dc_motor_torque`` in the robot module."""
    world, joint = wp.tid()
    velocity = qvel[world, joint_qvel_adr + joint]
    position_error = targets[world, joint] - qpos[world, joint_qpos_adr + joint]
    torque = stiffness[joint] * position_error - damping[joint] * velocity
    velocity_at_effort_limit = velocity_limit[joint] * (
        1.0 + effort_limit[joint] / saturation_effort[joint]
    )
    clipped_velocity = wp.clamp(
        velocity, -velocity_at_effort_limit, velocity_at_effort_limit
    )
    max_effort = wp.min(
        saturation_effort[joint] * (1.0 - clipped_velocity / velocity_limit[joint]),
        effort_limit[joint],
    )
    min_effort = wp.max(
        saturation_effort[joint] * (-1.0 - clipped_velocity / velocity_limit[joint]),
        -effort_limit[joint],
    )
    ctrl[world, joint_ctrl_adr + joint] = wp.clamp(torque, min_effort, max_effort)


@wp.kernel
def _command_delay_kernel(
    targets: wp.array2d(dtype=float),
    min_lag: int,
    max_lag: int,
    update_period: int,
    seed: int,
    phase: wp.array(dtype=int),
    lag: wp.array(dtype=int),
    count: wp.array(dtype=int),
    step: wp.array(dtype=int),
    buffer: wp.array3d(dtype=float),
    delayed_targets: wp.array2d(dtype=float),
):
    """The command delay: the same logic as ``SproutRobotEnv._delay_targets``.

    Each world stores the targets of the last physics steps, and the motors get the
    targets of ``lag`` steps ago. Each world draws a new lag from [min_lag, max_lag]
    each ``update_period`` steps.
    """
    world = wp.tid()
    size = buffer.shape[1]
    write = count[world] % size
    for joint in range(targets.shape[1]):
        buffer[world, write, joint] = targets[world, joint]
    update = int(0)
    if update_period == 0:
        update = 1
    elif (step[world] + phase[world]) % update_period == 0:
        update = 1
    if update == 1:
        state = wp.rand_init(seed, world * 1000003 + phase[world] * 7919 + step[world])
        lag[world] = wp.randi(state, min_lag, max_lag + 1)
    # Early in an episode, the oldest target is used.
    read = (count[world] - wp.min(lag[world], count[world])) % size
    for joint in range(targets.shape[1]):
        delayed_targets[world, joint] = buffer[world, read, joint]
    count[world] = count[world] + 1
    step[world] = step[world] + 1


@wp.kernel
def _reset_control_kernel(
    reset_mask: wp.array(dtype=wp.bool),
    default_targets: wp.array(dtype=float),
    new_delay_phase: wp.array(dtype=int),
    targets: wp.array2d(dtype=float),
    delay_phase: wp.array(dtype=int),
    delay_lag: wp.array(dtype=int),
    delay_count: wp.array(dtype=int),
    delay_step: wp.array(dtype=int),
):
    """Reset the targets and the command delay of each world that is in the mask."""
    world = wp.tid()
    if not reset_mask[world]:
        return
    for i in range(targets.shape[1]):
        targets[world, i] = default_targets[i]
    # A new episode of the command delay: no delay until the first update.
    delay_phase[world] = new_delay_phase[world]
    delay_lag[world] = 0
    delay_count[world] = 0
    delay_step[world] = 0


@wp.kernel
def _hold_robot_kernel(
    start_qpos: wp.array2d(dtype=float),
    base_qpos_adr: int,
    base_qvel_adr: int,
    joint_qpos_adr: int,
    joint_qvel_adr: int,
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
):
    """Put the robot of each world back into its initial pose, at rest."""
    world = wp.tid()
    for i in range(7):
        qpos[world, base_qpos_adr + i] = start_qpos[world, base_qpos_adr + i]
    for i in range(6):
        qvel[world, base_qvel_adr + i] = 0.0
    for i in range(NUM_SPROUT_JOINTS):
        qpos[world, joint_qpos_adr + i] = start_qpos[world, joint_qpos_adr + i]
        qvel[world, joint_qvel_adr + i] = 0.0


@wp.kernel
def _fallen_kernel(
    qpos: wp.array2d(dtype=float),
    base_height_adr: int,
    fallen_height: float,
    fallen: wp.array(dtype=wp.bool),
):
    """Find the worlds in which the robot base is below the fallen height."""
    world = wp.tid()
    fallen[world] = qpos[world, base_height_adr] < fallen_height


class Sprout3DBatchEnv(Dynamic3DBatchEnv):
    """Many parallel worlds of one Sprout task, on MuJoCo Warp.

    See ``Dynamic3DBatchEnv`` for the properties that all batch environments have.

    The action is one position target (rad) for each of the 27 joints, at 50 Hz. The
    targets reach the motors after a random command delay, as in the RL tasks and
    in ``SproutRobotEnv``. Each world has its own delay.

    ``info`` of ``step`` has ``fallen``: true where the torso is below the fallen
    height.

    Example:
        env = Sprout3DBatchEnv(task_config_path, num_worlds=1024, seed=0)
        obs = env.reset()
        obs, reward, terminated, truncated, info = env.step(env.default_action())
    """

    _robot_env: SproutRobotEnv

    def __init__(
        self,
        task_config_path: str,
        num_worlds: int,
        seed: int | None = None,
        device: str | None = None,
        horizon: int = 1000,
        num_settle_steps: int = SPROUT_SETTLE_STEPS,
        fallen_height: float = 0.3,
        nconmax: int | None = None,
        njmax: int | None = None,
        command_delay_steps: tuple[int, int] | None = None,
        command_delay_update_period: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            task_config_path: Task JSON with a "sprout" robot.
            num_worlds: Number of parallel worlds.
            seed: Random seed for the fixture poses and the initial states.
            device: Warp device, for example "cuda:0" or "cpu". Default: the
                preferred Warp device.
            horizon: Number of steps after which a world is truncated.
            num_settle_steps: Control steps that let the objects settle before the
                initial states are stored. The robot is held in its standing pose
                for this time, as in ``Sprout3DEnv.reset``.
            fallen_height: A world reports ``fallen`` when the torso is below this
                height (in meters).
            nconmax: Contact buffer size per world. Default: from the scene.
            njmax: Constraint buffer size per world. Default: from the scene.
            command_delay_steps: Minimum and maximum command delay, in physics
                steps; (0, 0) gives no delay. Default: the values of the RL tasks.
            command_delay_update_period: Physics steps between new random delays.
                Default: the value of the RL tasks.
            **kwargs: Passed to ``Sprout3DEnv`` (for example ``scene_bg``).
        """
        self._fallen_height = fallen_height
        single_env = Sprout3DEnv(
            task_config_path=task_config_path,
            allow_state_access=True,
            command_delay_steps=command_delay_steps,
            command_delay_update_period=command_delay_update_period,
            **kwargs,
        )
        super().__init__(
            single_env,
            num_worlds,
            seed=seed,
            device=device,
            horizon=horizon,
            num_settle_steps=num_settle_steps,
            nconmax=nconmax,
            njmax=njmax,
        )

    # ----- Robot interface -----

    def _setup_robot(self) -> None:
        robot_env = self._robot_env
        assert isinstance(robot_env, SproutRobotEnv)
        assert self.decimation == robot_env.decimation
        num_worlds = self.num_worlds

        # The motor model parameters, one entry per joint.
        model = robot_env.model
        self._stiffness = self._to_device(model.stiffness, float)
        self._damping = self._to_device(model.damping, float)
        self._effort_limit = self._to_device(model.effort_limit, float)
        self._saturation_effort = self._to_device(model.saturation_effort, float)
        self._velocity_limit = self._to_device(model.velocity_limit, float)
        self._default_targets = self._to_device(model.default_joint_positions, float)
        self._fallen = wp.zeros(num_worlds, wp.bool, self.device)

        # The state of the command delay (see _command_delay_kernel).
        self._delay_min_lag, self._delay_max_lag = robot_env.command_delay_steps
        self._delay_update_period = robot_env.command_delay_update_period
        self._delay_seed = int(self._rng.integers(2**31 - 1))
        self._delay_buffer = wp.zeros(
            (num_worlds, self._delay_max_lag + 1, self.action_dim), float, self.device
        )
        self._delayed_targets = wp.zeros_like(self._action)
        self._delay_phase = wp.zeros(num_worlds, int, self.device)
        self._new_delay_phase = wp.zeros(num_worlds, int, self.device)
        self._delay_lag = wp.zeros(num_worlds, int, self.device)
        self._delay_count = wp.zeros(num_worlds, int, self.device)
        self._delay_step = wp.zeros(num_worlds, int, self.device)

    def _robot_observation_map(
        self, column: dict[str, int]
    ) -> list[ObservationMapEntry]:
        robot_env = self._robot_env
        entries: list[ObservationMapEntry] = []
        for i, name in enumerate(_BASE_POS_FEATURES):
            entries.append((column[name], "qpos", robot_env.base_qpos_adr + i, 1.0))
        for i, name in enumerate(_BASE_VEL_FEATURES):
            entries.append((column[name], "qvel", robot_env.base_qvel_adr + i, 1.0))
        for i in range(NUM_SPROUT_JOINTS):
            entries.append(
                (column[f"pos_joint{i + 1}"], "qpos", robot_env.joint_qpos_adr + i, 1.0)
            )
            entries.append(
                (column[f"vel_joint{i + 1}"], "qvel", robot_env.joint_qvel_adr + i, 1.0)
            )
        return entries

    def _robot_base_qpos_adr(self) -> int:
        return self._robot_env.base_qpos_adr

    def _simulate_control_step(self) -> None:
        """One control step: decimation times the command delay, the motor model, and
        one physics step."""
        robot_env = self._robot_env
        motor_targets = self._action
        for _ in range(self.decimation):
            if self._delay_max_lag > 0:
                motor_targets = self._delayed_targets
                wp.launch(
                    _command_delay_kernel,
                    dim=self.num_worlds,
                    inputs=[
                        self._action,
                        self._delay_min_lag,
                        self._delay_max_lag,
                        self._delay_update_period,
                        self._delay_seed,
                        self._delay_phase,
                    ],
                    outputs=[
                        self._delay_lag,
                        self._delay_count,
                        self._delay_step,
                        self._delay_buffer,
                        self._delayed_targets,
                    ],
                    device=self.device,
                )
            wp.launch(
                _dc_motor_kernel,
                dim=(self.num_worlds, self.action_dim),
                inputs=[
                    self._data.qpos,
                    self._data.qvel,
                    motor_targets,
                    robot_env.joint_qpos_adr,
                    robot_env.joint_qvel_adr,
                    robot_env.joint_ctrl_adr,
                    self._stiffness,
                    self._damping,
                    self._effort_limit,
                    self._saturation_effort,
                    self._velocity_limit,
                ],
                outputs=[self._data.ctrl],
                device=self.device,
            )
            mjw.step(self._model, self._data)

    def default_action(self) -> wp.array:
        """The action that holds the default standing pose, for all worlds."""
        return self._to_device(
            np.tile(self._default_targets.numpy(), (self.num_worlds, 1)), float
        )

    def _reset_robot(self) -> None:
        self._new_delay_phase.assign(
            self._rng.integers(
                max(self._delay_update_period, 1), size=self.num_worlds, dtype=np.int32
            )
        )
        wp.launch(
            _reset_control_kernel,
            dim=self.num_worlds,
            inputs=[self._reset_mask, self._default_targets, self._new_delay_phase],
            outputs=[
                self._action,
                self._delay_phase,
                self._delay_lag,
                self._delay_count,
                self._delay_step,
            ],
            device=self.device,
        )

    def _after_settle_step(self) -> None:
        """Hold the robot in its standing pose, as in ``Sprout3DEnv.reset``."""
        robot_env = self._robot_env
        wp.launch(
            _hold_robot_kernel,
            dim=self.num_worlds,
            inputs=[
                self._start_qpos,
                robot_env.base_qpos_adr,
                robot_env.base_qvel_adr,
                robot_env.joint_qpos_adr,
                robot_env.joint_qvel_adr,
            ],
            outputs=[self._data.qpos, self._data.qvel],
            device=self.device,
        )

    def _get_info(self) -> dict[str, wp.array]:
        wp.launch(
            _fallen_kernel,
            dim=self.num_worlds,
            inputs=[
                self._data.qpos,
                self._robot_env.base_qpos_adr + 2,
                self._fallen_height,
            ],
            outputs=[self._fallen],
            device=self.device,
        )
        return {"fallen": self._fallen}
