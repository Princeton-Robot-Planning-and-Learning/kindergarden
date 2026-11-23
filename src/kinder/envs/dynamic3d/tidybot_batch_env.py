"""A parallel (batch) TidyBot environment that runs on MuJoCo Warp.

``TidyBot3DBatchEnv`` steps many copies ("worlds") of one TidyBot task at the same time,
on the GPU. ``Dynamic3DBatchEnv`` (in ``batch_env.py``) has the parts that are the same
for all robots. This module adds the control law of the TidyBot.

The single-world ``TidyBot3DEnv`` stays the reference: one world of the batch
environment is the same task as ``TidyBot3DEnv``, with one difference. The physics time
step is larger by default (see ``TIDYBOT_BATCH_TIMESTEP``).

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
from kinder.envs.dynamic3d.envs import TidyBot3DConfig, TidyBot3DEnv
from kinder.envs.dynamic3d.robots.tidybot_robot_env import TidyBotRobotEnv

# The physics time step of the batch environment, in seconds. ``TidyBot3DEnv`` uses
# 0.0005 s, which is 200 physics steps for each control step (0.1 s). The batch
# environment uses 0.002 s (50 physics steps), which is 4 times faster.
#
# 0.002 s is the largest step that keeps the contact stiffness of the gripper: the
# gripper pads have a contact time constant of 0.004 s (solref in tidybot.xml), and
# MuJoCo needs a time constant of at least two time steps. In replays of recorded
# pick-and-place demonstrations, 0.002 s gives the same success rate as 0.0005 s,
# and 0.004 s fails (see docs/batch_envs.md). Pass ``physics_timestep=0.0005`` for
# the time step of ``TidyBot3DEnv``.
TIDYBOT_BATCH_TIMESTEP = 0.002

NUM_BASE_JOINTS = 3  # x, y, and yaw
NUM_ARM_JOINTS = 7


@wp.kernel
def _tidybot_control_kernel(
    action: wp.array2d(dtype=float),
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    qfrc_bias: wp.array2d(dtype=float),
    base_qpos_adr: int,
    arm_qpos_adr: int,
    arm_qvel_adr: int,
    base_ctrl_adr: int,
    arm_ctrl_adr: int,
    gripper_ctrl_adr: int,
    arm_kp: wp.array(dtype=float),
    arm_kd: wp.array(dtype=float),
    arm_torque_limit: wp.array(dtype=float),
    act_delta: int,
    track_arm_velocity: int,
    ctrl: wp.array2d(dtype=float),
):
    """The control law of the TidyBot: the same as ``TidyBotRobotEnv._action_to_ctrl``.

    The action is: base targets (3), arm joint targets (7), gripper command (1), and,
    if ``track_arm_velocity`` is 1, arm joint velocity targets (7).
    """
    world = wp.tid()
    # Base: position actuators. The control is the target position.
    for i in range(3):
        target = action[world, i]
        if act_delta == 1:
            target += qpos[world, base_qpos_adr + i]
        ctrl[world, base_ctrl_adr + i] = target
    # Arm: torque actuators. The torque is a PD law plus gravity compensation.
    for i in range(7):
        position = qpos[world, arm_qpos_adr + i]
        velocity = qvel[world, arm_qvel_adr + i]
        target = action[world, 3 + i]
        if act_delta == 1:
            target += position
        torque = arm_kp[i] * (target - position)
        if track_arm_velocity == 1:
            torque += arm_kd[i] * (action[world, 11 + i] - velocity)
        else:
            torque -= arm_kd[i] * velocity
        torque += qfrc_bias[world, arm_qvel_adr + i]
        ctrl[world, arm_ctrl_adr + i] = wp.clamp(
            torque, -arm_torque_limit[i], arm_torque_limit[i]
        )
    # Gripper: a command from 0 (open) to 1 (closed), as a force command to 255.
    ctrl[world, gripper_ctrl_adr] = action[world, 10] * 255.0


@wp.kernel
def _reset_base_ctrl_kernel(
    reset_mask: wp.array(dtype=wp.bool),
    qpos: wp.array2d(dtype=float),
    base_qpos_adr: int,
    base_ctrl_adr: int,
    ctrl: wp.array2d(dtype=float),
):
    """Set the base targets of each world in the mask to the base position."""
    world = wp.tid()
    if not reset_mask[world]:
        return
    for i in range(3):
        ctrl[world, base_ctrl_adr + i] = qpos[world, base_qpos_adr + i]


class TidyBot3DBatchEnv(Dynamic3DBatchEnv):
    """Many parallel worlds of one TidyBot task, on MuJoCo Warp.

    See ``Dynamic3DBatchEnv`` for the properties that all batch environments have.

    The action is that of ``TidyBot3DEnv``, at 10 Hz: base targets (3), arm joint
    targets (7), and a gripper command (1). With the default configuration, the base
    and arm targets are relative to the current positions. As in ``TidyBot3DEnv``, the
    arm torques are computed one time for each control step.

    Example:
        env = TidyBot3DBatchEnv(task_config_path, num_worlds=1024, seed=0)
        obs = env.reset()
        obs, reward, terminated, truncated, info = env.step(env.default_action())
    """

    _robot_env: TidyBotRobotEnv

    def __init__(
        self,
        task_config_path: str,
        num_worlds: int,
        seed: int | None = None,
        device: str | None = None,
        horizon: int = 1000,
        num_settle_steps: int = 10,
        physics_timestep: float = TIDYBOT_BATCH_TIMESTEP,
        nconmax: int | None = None,
        njmax: int | None = None,
        config: TidyBot3DConfig = TidyBot3DConfig(),
        **kwargs: Any,
    ) -> None:
        """
        Args:
            task_config_path: Task JSON with a "tidybot" robot.
            num_worlds: Number of parallel worlds.
            seed: Random seed for the fixture poses and the initial states.
            device: Warp device, for example "cuda:0" or "cpu". Default: the
                preferred Warp device.
            horizon: Number of steps after which a world is truncated.
            num_settle_steps: Control steps that let the objects settle before the
                initial states are stored, as in ``TidyBot3DEnv.reset``.
            physics_timestep: Time of one physics step, in seconds. The control
                period (0.1 s) must be a multiple of it.
            nconmax: Contact buffer size per world. Default: from the scene.
            njmax: Constraint buffer size per world. Default: from the scene.
            config: The configuration of ``TidyBot3DEnv``. The tossing tasks use
                ``TidyBot3DConfig(use_arm_velocities=True)``.
            **kwargs: Passed to ``TidyBot3DEnv`` (for example ``scene_bg``).
        """
        single_env = TidyBot3DEnv(
            config=config,
            task_config_path=task_config_path,
            allow_state_access=True,
            **kwargs,
        )
        super().__init__(
            single_env,
            num_worlds,
            seed=seed,
            device=device,
            horizon=horizon,
            num_settle_steps=num_settle_steps,
            physics_timestep=physics_timestep,
            nconmax=nconmax,
            njmax=njmax,
        )

    # ----- Robot interface -----

    def _setup_robot(self) -> None:
        robot_env = self._robot_env
        assert isinstance(robot_env, TidyBotRobotEnv)
        mj_model = self._mj_model
        name = robot_env.name

        # The base joints (x, y, yaw) and the arm joints have consecutive addresses,
        # so the address of the first joint is sufficient.
        self._base_qpos_adr = int(mj_model.joint(f"{name}_joint_x").qposadr[0])
        self._base_qvel_adr = int(mj_model.joint(f"{name}_joint_x").dofadr[0])
        self._arm_qpos_adr = int(mj_model.joint(f"{name}_joint_1").qposadr[0])
        self._arm_qvel_adr = int(mj_model.joint(f"{name}_joint_1").dofadr[0])
        self._gripper_qvel_adr = int(
            mj_model.joint(f"{name}_right_driver_joint").dofadr[0]
        )
        self._base_ctrl_adr = int(mj_model.actuator(f"{name}_joint_x").id)
        self._arm_ctrl_adr = int(mj_model.actuator(f"{name}_joint_1").id)
        self._gripper_ctrl_adr = int(mj_model.actuator(f"{name}_fingers_actuator").id)

        self._arm_kp = self._to_device(robot_env.arm_kp, float)
        self._arm_kd = self._to_device(robot_env.arm_kd, float)
        self._arm_torque_limit = self._to_device(robot_env.ARM_TORQUE_LIMITS, float)
        self._act_delta = int(robot_env.act_delta)
        # An action with 18 values has velocity targets for the arm.
        self._track_arm_velocity = int(self.action_dim == 18)

    def _robot_observation_map(
        self, column: dict[str, int]
    ) -> list[ObservationMapEntry]:
        entries: list[ObservationMapEntry] = []
        for i, name in enumerate(["x", "y", "rot"]):
            entries.append(
                (column[f"pos_base_{name}"], "qpos", self._base_qpos_adr + i, 1)
            )
            entries.append(
                (column[f"vel_base_{name}"], "qvel", self._base_qvel_adr + i, 1)
            )
        for i in range(NUM_ARM_JOINTS):
            pos_column = column[f"pos_arm_joint{i + 1}"]
            vel_column = column[f"vel_arm_joint{i + 1}"]
            entries.append((pos_column, "qpos", self._arm_qpos_adr + i, 1.0))
            entries.append((vel_column, "qvel", self._arm_qvel_adr + i, 1.0))
        # The gripper position feature is the gripper command, from 0 to 1.
        entries.append((column["pos_gripper"], "ctrl", self._gripper_ctrl_adr, 1 / 255))
        entries.append((column["vel_gripper"], "qvel", self._gripper_qvel_adr, 1.0))
        return entries

    def _robot_base_qpos_adr(self) -> int:
        return self._base_qpos_adr

    def _simulate_control_step(self) -> None:
        """One control step: the control law, and decimation physics steps."""
        wp.launch(
            _tidybot_control_kernel,
            dim=self.num_worlds,
            inputs=[
                self._action,
                self._data.qpos,
                self._data.qvel,
                self._data.qfrc_bias,
                self._base_qpos_adr,
                self._arm_qpos_adr,
                self._arm_qvel_adr,
                self._base_ctrl_adr,
                self._arm_ctrl_adr,
                self._gripper_ctrl_adr,
                self._arm_kp,
                self._arm_kd,
                self._arm_torque_limit,
                self._act_delta,
                self._track_arm_velocity,
            ],
            outputs=[self._data.ctrl],
            device=self.device,
        )
        for _ in range(self.decimation):
            mjw.step(self._model, self._data)

    def default_action(self) -> wp.array:
        """The action that holds the robot in its current pose, for all worlds.

        The gripper command is 0 (open).
        """
        action = np.zeros((self.num_worlds, self.action_dim), dtype=np.float32)
        if not self._act_delta:
            qpos = self._data.qpos.numpy()
            base = slice(self._base_qpos_adr, self._base_qpos_adr + NUM_BASE_JOINTS)
            arm = slice(self._arm_qpos_adr, self._arm_qpos_adr + NUM_ARM_JOINTS)
            action[:, :3] = qpos[:, base]
            action[:, 3:10] = qpos[:, arm]
        return self._to_device(action, float)

    def _reset_robot(self) -> None:
        wp.launch(
            _reset_base_ctrl_kernel,
            dim=self.num_worlds,
            inputs=[
                self._reset_mask,
                self._data.qpos,
                self._base_qpos_adr,
                self._base_ctrl_adr,
            ],
            outputs=[self._data.ctrl],
            device=self.device,
        )
