"""A parallel (batch) Sprout environment that runs on MuJoCo Warp.

``Sprout3DBatchEnv`` steps many copies ("worlds") of one Sprout task at the same time,
on the GPU. It is not a gymnasium environment: it is a plain class whose methods take
and return arrays with one row per world.

The single-world ``Sprout3DEnv`` stays the reference. The batch environment uses it to
build the scene, to sample the initial states, and to define the observation vector. One
world of the batch environment is thus the same task as ``Sprout3DEnv``.

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
from numpy.typing import NDArray
from relational_structs import ObjectCentricState
from relational_structs.spaces import ObjectCentricBoxSpace

from kinder.envs.dynamic3d.envs import ObjectCentricSprout3DEnv, Sprout3DEnv
from kinder.envs.dynamic3d.object_types import (
    MujocoDrawerObjectType,
    MujocoSproutRobotObjectType,
)
from kinder.envs.dynamic3d.robots.sprout_robot_env import (
    NUM_SPROUT_JOINTS,
    SPROUT_SETTLE_STEPS,
    SproutRobotEnv,
)

# Feature names of an object with a free joint, in the order of its qpos and qvel.
_FREE_JOINT_POS_FEATURES = ["x", "y", "z", "qw", "qx", "qy", "qz"]
_FREE_JOINT_VEL_FEATURES = ["vx", "vy", "vz", "wx", "wy", "wz"]
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
def _gather_kernel(
    source: wp.array2d(dtype=float),
    source_adr: wp.array(dtype=int),
    obs_column: wp.array(dtype=int),
    obs: wp.array2d(dtype=float),
):
    """Copy entries of qpos (or qvel) into their columns of the observation."""
    world, i = wp.tid()
    obs[world, obs_column[i]] = source[world, source_adr[i]]


@wp.kernel
def _goal_kernel(
    qpos: wp.array2d(dtype=float),
    xmat: wp.array2d(dtype=wp.mat33),
    site_xpos: wp.array2d(dtype=wp.vec3),
    site_xmat: wp.array2d(dtype=wp.mat33),
    predicate_kind: wp.array(dtype=int),
    predicate_qpos_adr: wp.array(dtype=int),
    predicate_on_floor: wp.array(dtype=int),
    predicate_site_start: wp.array(dtype=int),
    predicate_site_count: wp.array(dtype=int),
    predicate_body_id: wp.array(dtype=int),
    predicate_min_up: wp.array(dtype=float),
    site_id: wp.array(dtype=int),
    site_half_size: wp.array(dtype=wp.vec3),
    num_predicates: int,
    require_all: int,
    success: wp.array(dtype=wp.bool),
):
    """Check the goal predicates of each world.

    An "on" predicate (kind 0) holds if the position of its object is inside the world-
    frame bounding box of one of the sites of its region. This is the check of
    ``Region.check_in_region``.

    A "balanced" predicate (kind 1) holds if the up axis of its body is near the world
    up axis. This is the check of ``GeneratedSeesaw.is_balanced``.
    """
    world = wp.tid()
    num_true = int(0)
    for p in range(num_predicates):
        holds = int(0)
        if predicate_kind[p] == 1:
            if xmat[world, predicate_body_id[p]][2, 2] >= predicate_min_up[p]:
                holds = 1
        else:
            adr = predicate_qpos_adr[p]
            position = wp.vec3(
                qpos[world, adr], qpos[world, adr + 1], qpos[world, adr + 2]
            )
            if predicate_on_floor[p] == 1:
                # The robot base counts as a point on the floor.
                position = wp.vec3(position[0], position[1], 0.0)
            first_site = predicate_site_start[p]
            for k in range(first_site, first_site + predicate_site_count[p]):
                rot = site_xmat[world, site_id[k]]
                half = site_half_size[k]
                # Half extents of the axis-aligned box around the rotated site box.
                extent = wp.vec3(
                    wp.abs(rot[0, 0]) * half[0]
                    + wp.abs(rot[0, 1]) * half[1]
                    + wp.abs(rot[0, 2]) * half[2],
                    wp.abs(rot[1, 0]) * half[0]
                    + wp.abs(rot[1, 1]) * half[1]
                    + wp.abs(rot[1, 2]) * half[2],
                    wp.abs(rot[2, 0]) * half[0]
                    + wp.abs(rot[2, 1]) * half[1]
                    + wp.abs(rot[2, 2]) * half[2],
                )
                offset = position - site_xpos[world, site_id[k]]
                if (
                    wp.abs(offset[0]) <= extent[0]
                    and wp.abs(offset[1]) <= extent[1]
                    and wp.abs(offset[2]) <= extent[2]
                ):
                    holds = 1
        num_true += holds
    if num_predicates == 0:
        success[world] = False
    elif require_all == 1:
        success[world] = num_true == num_predicates
    else:
        success[world] = num_true > 0


@wp.kernel
def _episode_kernel(
    qpos: wp.array2d(dtype=float),
    success: wp.array(dtype=wp.bool),
    base_height_adr: int,
    fallen_height: float,
    horizon: int,
    step_count: wp.array(dtype=int),
    reward: wp.array(dtype=float),
    terminated: wp.array(dtype=wp.bool),
    truncated: wp.array(dtype=wp.bool),
    fallen: wp.array(dtype=wp.bool),
):
    """Compute the reward and the end-of-episode flags of each world."""
    world = wp.tid()
    step_count[world] = step_count[world] + 1
    reward[world] = -1.0
    terminated[world] = success[world]
    truncated[world] = step_count[world] >= horizon
    fallen[world] = qpos[world, base_height_adr] < fallen_height


@wp.kernel
def _reset_kernel(
    reset_mask: wp.array(dtype=wp.bool),
    start_index: wp.array(dtype=int),
    start_qpos: wp.array2d(dtype=float),
    start_qvel: wp.array2d(dtype=float),
    default_targets: wp.array(dtype=float),
    new_delay_phase: wp.array(dtype=int),
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    targets: wp.array2d(dtype=float),
    step_count: wp.array(dtype=int),
    delay_phase: wp.array(dtype=int),
    delay_lag: wp.array(dtype=int),
    delay_count: wp.array(dtype=int),
    delay_step: wp.array(dtype=int),
):
    """Put each world that is in the mask into its initial state."""
    world = wp.tid()
    if not reset_mask[world]:
        return
    start = start_index[world]
    for i in range(qpos.shape[1]):
        qpos[world, i] = start_qpos[start, i]
    for i in range(qvel.shape[1]):
        qvel[world, i] = start_qvel[start, i]
    for i in range(targets.shape[1]):
        targets[world, i] = default_targets[i]
    step_count[world] = 0
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


class Sprout3DBatchEnv:
    """Many parallel worlds of one Sprout task, on MuJoCo Warp.

    All worlds use one compiled model: the fixtures have the same poses in all
    worlds. Each world has its own initial poses of the objects and the robot.

    The arrays that ``reset`` and ``step`` return are Warp arrays on the simulation
    device, with one row per world. The environment reuses these arrays: each step
    writes the new values into them. Use ``array.numpy()`` for a NumPy copy, or
    ``warp.to_torch(array)`` for a torch tensor without a copy.

    The targets reach the motors after a random command delay, as in the RL tasks
    and in ``SproutRobotEnv``. Each world has its own delay.

    Worlds do not reset by themselves. Call ``reset(mask)`` for the worlds that are
    done.

    Example:
        env = Sprout3DBatchEnv(task_config_path, num_worlds=1024, seed=0)
        obs = env.reset()
        obs, reward, terminated, truncated, info = env.step(env.default_action())
    """

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
        self.num_worlds = num_worlds
        self.horizon = horizon
        self.device = wp.get_device(device)
        self._rng = np.random.default_rng(seed)

        # The single-world environment builds the scene and compiles the model.
        self._single_env = Sprout3DEnv(
            task_config_path=task_config_path,
            allow_state_access=True,
            command_delay_steps=command_delay_steps,
            command_delay_update_period=command_delay_update_period,
            **kwargs,
        )
        self._single_env.reset(seed=seed)
        object_centric_env = self._single_env._object_centric_env
        assert isinstance(object_centric_env, ObjectCentricSprout3DEnv)
        self._object_centric_env = object_centric_env
        robot_env = object_centric_env._robot_env
        assert isinstance(robot_env, SproutRobotEnv) and robot_env.sim is not None
        self._robot_env = robot_env
        self._mj_model = robot_env.sim.model.mj_model
        self._mj_data = robot_env.sim.data.mj_data
        self.decimation = robot_env.decimation
        self._delay_min_lag, self._delay_max_lag = robot_env.command_delay_steps
        self._delay_update_period = robot_env.command_delay_update_period
        self._delay_seed = int(self._rng.integers(2**31 - 1))
        self.step_dt = self.decimation * self._mj_model.opt.timestep

        assert isinstance(self._single_env.observation_space, ObjectCentricBoxSpace)
        self.observation_space = self._single_env.observation_space
        self.action_space = self._single_env.action_space
        (self.obs_dim,) = self.observation_space.shape
        self.action_dim = NUM_SPROUT_JOINTS

        # Each world gets its own initial poses of the objects and the robot.
        initial_qpos = np.stack(
            [object_centric_env.sample_initial_qpos() for _ in range(num_worlds)]
        )

        # Put the model and the data on the device. A buffer that is too small can
        # stop the process, so the defaults leave a margin over the settled scene (a
        # robot that falls makes more contacts). The GPU memory grows with
        # num_worlds * njmax, so a large scene permits fewer worlds.
        self._model = mjw.put_model(self._mj_model)
        self._data = mjw.put_data(
            self._mj_model,
            self._mj_data,
            nworld=num_worlds,
            nconmax=(
                nconmax if nconmax is not None else max(256, 2 * self._mj_data.ncon)
            ),
            njmax=njmax if njmax is not None else max(800, 2 * self._mj_data.nefc),
        )

        def to_device(values: Any, dtype: Any) -> wp.array:
            return wp.array(values, dtype=dtype, device=self.device)

        # The motor model parameters, one entry per joint.
        model = robot_env.model
        self._stiffness = to_device(model.stiffness, float)
        self._damping = to_device(model.damping, float)
        self._effort_limit = to_device(model.effort_limit, float)
        self._saturation_effort = to_device(model.saturation_effort, float)
        self._velocity_limit = to_device(model.velocity_limit, float)
        self._default_targets = to_device(model.default_joint_positions, float)
        self._fallen_height = fallen_height

        # Per-world arrays.
        self._targets = wp.zeros((num_worlds, self.action_dim), float, self.device)
        self._step_count = wp.zeros(num_worlds, int, self.device)
        self._reset_mask = wp.zeros(num_worlds, wp.bool, self.device)
        self._start_index = to_device(np.arange(num_worlds), int)
        self._obs = wp.zeros((num_worlds, self.obs_dim), float, self.device)
        self._reward = wp.zeros(num_worlds, float, self.device)
        self._success = wp.zeros(num_worlds, wp.bool, self.device)
        self._terminated = wp.zeros(num_worlds, wp.bool, self.device)
        self._truncated = wp.zeros(num_worlds, wp.bool, self.device)
        self._fallen = wp.zeros(num_worlds, wp.bool, self.device)

        # The state of the command delay (see _command_delay_kernel).
        self._delay_buffer = wp.zeros(
            (num_worlds, self._delay_max_lag + 1, self.action_dim), float, self.device
        )
        self._delayed_targets = wp.zeros_like(self._targets)
        self._delay_phase = wp.zeros(num_worlds, int, self.device)
        self._new_delay_phase = wp.zeros(num_worlds, int, self.device)
        self._delay_lag = wp.zeros(num_worlds, int, self.device)
        self._delay_count = wp.zeros(num_worlds, int, self.device)
        self._delay_step = wp.zeros(num_worlds, int, self.device)

        self._setup_observation_map()
        self._setup_goal_predicates()

        # Record one control step as a graph on the GPU, so that a step is one
        # launch and not many small ones. This runs the step once.
        self._graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture(self.device) as capture:
                self._simulate_control_step()
            self._graph = capture.graph

        # Let the scene settle, and store the settled states as the initial states.
        self._start_qpos = to_device(initial_qpos, float)
        self._start_qvel = wp.zeros((num_worlds, self._mj_model.nv), float, self.device)
        self.reset()
        for _ in range(num_settle_steps):
            self._launch_control_step()
            # As in Sprout3DEnv.reset, the robot is held in its standing pose.
            wp.launch(
                _hold_robot_kernel,
                dim=num_worlds,
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
        wp.copy(self._start_qpos, self._data.qpos)
        wp.copy(self._start_qvel, self._data.qvel)
        self.reset()

    # ----- Setup -----

    def _setup_observation_map(self) -> None:
        """Find the qpos or qvel entry of each observation column that changes.

        The observation of one world is the vector of ``Sprout3DEnv``: the features of
        each object, in the order of the observation space. The columns that never
        change (fixture poses, bounding boxes) are written once.
        """
        mj_model = self._mj_model
        robot_env = self._robot_env
        objects_dict = self._object_centric_env._objects_dict
        qpos_map: list[tuple[int, int]] = []  # (observation column, qpos address)
        qvel_map: list[tuple[int, int]] = []  # (observation column, qvel address)

        offset = 0
        for obj in self.observation_space.constant_objects:
            features = self.observation_space.type_features[obj.type]
            column = {name: offset + i for i, name in enumerate(features)}
            if obj.type == MujocoSproutRobotObjectType:
                for i, name in enumerate(_BASE_POS_FEATURES):
                    qpos_map.append((column[name], robot_env.base_qpos_adr + i))
                for i, name in enumerate(_BASE_VEL_FEATURES):
                    qvel_map.append((column[name], robot_env.base_qvel_adr + i))
                for i in range(NUM_SPROUT_JOINTS):
                    qpos_map.append(
                        (column[f"pos_joint{i + 1}"], robot_env.joint_qpos_adr + i)
                    )
                    qvel_map.append(
                        (column[f"vel_joint{i + 1}"], robot_env.joint_qvel_adr + i)
                    )
            elif obj.name in objects_dict:
                # A movable object: its pose and velocity are those of its free joint.
                joint = mj_model.joint(objects_dict[obj.name].joint_name)
                for i, name in enumerate(_FREE_JOINT_POS_FEATURES):
                    qpos_map.append((column[name], int(joint.qposadr[0]) + i))
                for i, name in enumerate(_FREE_JOINT_VEL_FEATURES):
                    if name in column:
                        qvel_map.append((column[name], int(joint.dofadr[0]) + i))
            elif obj.is_instance(MujocoDrawerObjectType):
                joint = mj_model.joint(f"{obj.name}_joint")
                qpos_map.append((column["pos"], int(joint.qposadr[0])))
            offset += len(features)

        self._obs_qpos_column = np.array([c for c, _ in qpos_map], dtype=np.int32)
        self._obs_qpos_adr = np.array([a for _, a in qpos_map], dtype=np.int32)
        self._obs_qvel_column = np.array([c for c, _ in qvel_map], dtype=np.int32)
        self._obs_qvel_adr = np.array([a for _, a in qvel_map], dtype=np.int32)
        self._check_observation_map()

        # Write the columns that never change, for all worlds.
        static_obs = self._single_env.get_state().astype(np.float32)
        self._obs.assign(np.tile(static_obs, (self.num_worlds, 1)))
        self._obs_qpos_column_wp = wp.array(
            self._obs_qpos_column, dtype=int, device=self.device
        )
        self._obs_qpos_adr_wp = wp.array(
            self._obs_qpos_adr, dtype=int, device=self.device
        )
        self._obs_qvel_column_wp = wp.array(
            self._obs_qvel_column, dtype=int, device=self.device
        )
        self._obs_qvel_adr_wp = wp.array(
            self._obs_qvel_adr, dtype=int, device=self.device
        )

    def _check_observation_map(self) -> None:
        """Check the observation map against ``Sprout3DEnv`` on a random state.

        This fails for a task with an object whose features are not a copy of qpos or
        qvel entries, which the batch environment does not support.
        """
        mj_data = self._mj_data
        saved_qpos, saved_qvel = mj_data.qpos.copy(), mj_data.qvel.copy()
        reference = self._single_env.get_state()
        mj_data.qpos[:] = self._rng.uniform(-1.0, 1.0, mj_data.qpos.shape)
        mj_data.qvel[:] = self._rng.uniform(-1.0, 1.0, mj_data.qvel.shape)
        expected = self.observation_space.vectorize(
            self._object_centric_env._get_object_centric_state()
        )
        mapped = reference.copy()
        mapped[self._obs_qpos_column] = mj_data.qpos[self._obs_qpos_adr]
        mapped[self._obs_qvel_column] = mj_data.qvel[self._obs_qvel_adr]
        mj_data.qpos[:], mj_data.qvel[:] = saved_qpos, saved_qvel
        mismatch = np.nonzero(~np.isclose(mapped, expected, atol=1e-5))[0]
        if len(mismatch) > 0:
            raise NotImplementedError(
                f"The batch environment cannot build observation columns "
                f"{mismatch.tolist()} of this task from qpos and qvel."
            )

    def _setup_goal_predicates(self) -> None:
        """Convert the goal predicates of the task to arrays for the goal kernel."""
        env = self._object_centric_env
        mj_model = self._mj_model
        goal_predicates = list(env.task_config.get("goal_state", []))
        require_all = True
        if goal_predicates and goal_predicates[0] in ("and", "or"):
            require_all = goal_predicates[0] == "and"
            goal_predicates = goal_predicates[1:]

        # One entry per predicate. An entry that a predicate kind does not use is 0.
        kind, qpos_adr, on_floor, site_start, site_count = [], [], [], [], []
        body_id, min_up = [], []
        site_ids: list[int] = []
        site_half_sizes: list[NDArray[np.float64]] = []
        for predicate in goal_predicates:
            if predicate[0] == "balanced":
                # ["balanced", seesaw name, tolerance in degrees]
                tolerance_degrees = float(predicate[2]) if len(predicate) > 2 else 5.0
                kind.append(1)
                body_id.append(mj_model.body(f"{predicate[1]}_beam").id)
                min_up.append(np.cos(np.radians(tolerance_degrees)))
                qpos_adr.append(0)
                on_floor.append(0)
                site_start.append(0)
                site_count.append(0)
                continue
            if predicate[0] not in ("on", "in"):
                raise NotImplementedError(
                    f"Goal predicate {predicate[0]} is not implemented in the "
                    f"batch environment"
                )
            kind.append(0)
            body_id.append(0)
            min_up.append(0.0)
            object_name, region_name = predicate[1], predicate[2]
            if object_name == env.robot_name:
                qpos_adr.append(self._robot_env.base_qpos_adr)
                on_floor.append(1)
            else:
                joint_name = env._objects_dict[object_name].joint_name
                qpos_adr.append(int(mj_model.joint(joint_name).qposadr[0]))
                on_floor.append(0)

            # Find the entity (the ground, a fixture, or an object) with the region.
            target = env.task_config["regions"][region_name]["target"]
            entity: Any
            if target == "ground":
                entity = env._ground_fixture
            elif target in env._fixtures_dict:
                entity = env._fixtures_dict[target]
            else:
                entity = env._objects_dict[target]
            regions = entity.region_objects[region_name]
            site_start.append(len(site_ids))
            site_count.append(len(regions))
            for region in regions:
                site_name = region.site_element.get("name")
                site_ids.append(mj_model.site(site_name).id)
                site_half_sizes.append(region.bbox_at_origin[3:])

        def ints(values: list[int]) -> wp.array:
            return wp.array(np.array(values, dtype=np.int32), int, device=self.device)

        self._num_predicates = len(goal_predicates)
        self._require_all = int(require_all)
        self._predicate_kind = ints(kind)
        self._predicate_qpos_adr = ints(qpos_adr)
        self._predicate_on_floor = ints(on_floor)
        self._predicate_site_start = ints(site_start)
        self._predicate_site_count = ints(site_count)
        self._predicate_body_id = ints(body_id)
        self._predicate_min_up = wp.array(
            np.array(min_up, dtype=np.float32), float, device=self.device
        )
        self._site_id = ints(site_ids)
        self._site_half_size = wp.array(
            np.array(site_half_sizes, dtype=np.float32).reshape(-1, 3),
            wp.vec3,
            device=self.device,
        )

    # ----- Simulation -----

    def _simulate_control_step(self) -> None:
        """One control step: decimation times the command delay, the motor model, and
        one physics step."""
        robot_env = self._robot_env
        motor_targets = self._targets
        for _ in range(self.decimation):
            if self._delay_max_lag > 0:
                motor_targets = self._delayed_targets
                wp.launch(
                    _command_delay_kernel,
                    dim=self.num_worlds,
                    inputs=[
                        self._targets,
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

    def _launch_control_step(self) -> None:
        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self._simulate_control_step()

    def _update_observation(self) -> None:
        for source, adr, column in (
            (self._data.qpos, self._obs_qpos_adr_wp, self._obs_qpos_column_wp),
            (self._data.qvel, self._obs_qvel_adr_wp, self._obs_qvel_column_wp),
        ):
            wp.launch(
                _gather_kernel,
                dim=(self.num_worlds, len(adr)),
                inputs=[source, adr, column],
                outputs=[self._obs],
                device=self.device,
            )

    def _update_success(self) -> None:
        wp.launch(
            _goal_kernel,
            dim=self.num_worlds,
            inputs=[
                self._data.qpos,
                self._data.xmat,
                self._data.site_xpos,
                self._data.site_xmat,
                self._predicate_kind,
                self._predicate_qpos_adr,
                self._predicate_on_floor,
                self._predicate_site_start,
                self._predicate_site_count,
                self._predicate_body_id,
                self._predicate_min_up,
                self._site_id,
                self._site_half_size,
                self._num_predicates,
                self._require_all,
            ],
            outputs=[self._success],
            device=self.device,
        )

    # ----- Public interface -----

    def default_action(self) -> wp.array:
        """The action that holds the default standing pose, for all worlds."""
        return wp.array(
            np.tile(self._default_targets.numpy(), (self.num_worlds, 1)),
            dtype=float,
            device=self.device,
        )

    def reset(self, mask: Any | None = None) -> wp.array:
        """Put worlds into an initial state and return the observation of all worlds.

        Args:
            mask: Boolean array with one entry per world: the worlds to reset.
                Default: all worlds, and world ``i`` gets initial state ``i``, so
                that a full reset always gives the same states. With a mask, each
                world in the mask gets a random initial state.
        """
        if mask is None:
            self._reset_mask.fill_(True)
            self._start_index.assign(np.arange(self.num_worlds, dtype=np.int32))
        else:
            self._copy_to(self._reset_mask, mask, np.bool_)
            self._start_index.assign(
                self._rng.integers(
                    self.num_worlds, size=self.num_worlds, dtype=np.int32
                )
            )
        self._new_delay_phase.assign(
            self._rng.integers(
                max(self._delay_update_period, 1), size=self.num_worlds, dtype=np.int32
            )
        )
        mjw.reset_data(self._model, self._data, reset=self._reset_mask)
        wp.launch(
            _reset_kernel,
            dim=self.num_worlds,
            inputs=[
                self._reset_mask,
                self._start_index,
                self._start_qpos,
                self._start_qvel,
                self._default_targets,
                self._new_delay_phase,
            ],
            outputs=[
                self._data.qpos,
                self._data.qvel,
                self._targets,
                self._step_count,
                self._delay_phase,
                self._delay_lag,
                self._delay_count,
                self._delay_step,
            ],
            device=self.device,
        )
        # Update the body and site poses for the new states.
        mjw.forward(self._model, self._data)
        self._update_observation()
        return self._obs

    def step(
        self, action: Any
    ) -> tuple[wp.array, wp.array, wp.array, wp.array, dict[str, wp.array]]:
        """Advance all worlds by one control step (0.02 s).

        Args:
            action: Joint position targets (rad), with shape (num_worlds, 27). A
                Warp array, a NumPy array, or an array with DLPack support (for
                example a torch tensor).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info), with one
            row per world. ``terminated`` is true where the goal is reached. ``info``
            has ``fallen``: true where the torso is below the fallen height.
        """
        self._copy_to(self._targets, action, np.float32)
        self._launch_control_step()
        self._update_observation()
        self._update_success()
        wp.launch(
            _episode_kernel,
            dim=self.num_worlds,
            inputs=[
                self._data.qpos,
                self._success,
                self._robot_env.base_qpos_adr + 2,
                self._fallen_height,
                self.horizon,
            ],
            outputs=[
                self._step_count,
                self._reward,
                self._terminated,
                self._truncated,
                self._fallen,
            ],
            device=self.device,
        )
        info = {"fallen": self._fallen}
        return self._obs, self._reward, self._terminated, self._truncated, info

    def _copy_to(self, destination: wp.array, values: Any, dtype: Any) -> None:
        """Copy array-like values into a device array of the same shape."""
        if isinstance(values, wp.array):
            source = values
        elif isinstance(values, np.ndarray) or not hasattr(values, "__dlpack__"):
            source = wp.array(
                np.asarray(values, dtype=dtype),
                dtype=destination.dtype,
                device=self.device,
            )
        else:
            source = wp.from_dlpack(values)
        assert (
            source.shape == destination.shape
        ), f"Expected an array with shape {destination.shape}, got {source.shape}"
        wp.copy(destination, source)

    def get_object_centric_state(self, world: int) -> ObjectCentricState:
        """Get the object-centric state of one world (for example for a planner)."""
        return self.observation_space.devectorize(self._obs.numpy()[world])

    def render(self, world: int = 0) -> NDArray[np.uint8]:
        """Render one world with the cameras of the single-world environment."""
        self._single_env.set_state(self._obs.numpy()[world])
        return self._single_env.render()  # type: ignore[no-any-return]

    def buffer_use(self) -> dict[str, int]:
        """Report the use of the contact and constraint buffers.

        ``contacts`` is the number of contacts in all worlds, and ``constraints`` is the
        largest number of constraints in one world. Make ``nconmax`` or ``njmax`` larger
        when a value comes near its limit.
        """
        return {
            "contacts": int(self._data.nacon.numpy()[0]),
            "contacts_limit": int(self._data.naconmax),
            "constraints": int(self._data.nefc.numpy().max()),
            "constraints_limit": int(self._data.njmax),
        }

    def close(self) -> None:
        """Close the environment and free resources."""
        self._single_env.close()
