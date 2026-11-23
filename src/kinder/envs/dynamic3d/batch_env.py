"""The base class of the parallel (batch) dynamic3d environments on MuJoCo Warp.

A batch environment steps many copies ("worlds") of one dynamic3d task at the same
time, on the GPU. It is not a gymnasium environment: it is a plain class whose methods
take and return arrays with one row per world.

The single-world environment of the task (for example ``TidyBot3DEnv``) stays the
reference. The batch environment uses it to build the scene, to sample the initial
states, and to define the observation vector. One world of the batch environment is
thus the same task as the single-world environment.

``Dynamic3DBatchEnv`` has the parts that are the same for all robots: the model on the
GPU, the observation, the goal check, the reset, and the episode flags. A subclass adds
the control law of one robot:

- ``Sprout3DBatchEnv`` in ``sprout_batch_env.py``.
- ``TidyBot3DBatchEnv`` in ``tidybot_batch_env.py``.

This module needs ``mujoco_warp``. See ``requirements/dynamic3d_warp.txt``.
"""

# The batch environment reads the internals of the single-world environment that it
# is built on.
# pylint: disable=protected-access

# The Warp kernels use Warp types as annotations, which mypy cannot check.
# mypy: disable-error-code="valid-type,misc,no-untyped-call,var-annotated,assignment"

import abc
from typing import Any

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from numpy.typing import NDArray
from relational_structs import ObjectCentricState
from relational_structs.spaces import ObjectCentricBoxSpace

from kinder.core import ConstantObjectKinDEREnv
from kinder.envs.dynamic3d.envs import ObjectCentricRobotEnv
from kinder.envs.dynamic3d.object_types import MujocoDrawerObjectType

# Feature names of an object with a free joint, in the order of its qpos and qvel.
_FREE_JOINT_POS_FEATURES = ["x", "y", "z", "qw", "qx", "qy", "qz"]
_FREE_JOINT_VEL_FEATURES = ["vx", "vy", "vz", "wx", "wy", "wz"]

# One entry of the observation map: the observation column, the source array ("qpos",
# "qvel", or "ctrl"), the address in the source array, and a scale factor.
ObservationMapEntry = tuple[int, str, int, float]


@wp.kernel
def _gather_kernel(
    source: wp.array2d(dtype=float),
    source_adr: wp.array(dtype=int),
    obs_column: wp.array(dtype=int),
    scale: wp.array(dtype=float),
    obs: wp.array2d(dtype=float),
):
    """Copy entries of qpos (or qvel, or ctrl) into their observation columns."""
    world, i = wp.tid()
    obs[world, obs_column[i]] = scale[i] * source[world, source_adr[i]]


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
    success: wp.array(dtype=wp.bool),
    horizon: int,
    step_count: wp.array(dtype=int),
    reward: wp.array(dtype=float),
    terminated: wp.array(dtype=wp.bool),
    truncated: wp.array(dtype=wp.bool),
):
    """Compute the reward and the end-of-episode flags of each world."""
    world = wp.tid()
    step_count[world] = step_count[world] + 1
    reward[world] = -1.0
    terminated[world] = success[world]
    truncated[world] = step_count[world] >= horizon


@wp.kernel
def _reset_state_kernel(
    reset_mask: wp.array(dtype=wp.bool),
    start_index: wp.array(dtype=int),
    start_qpos: wp.array2d(dtype=float),
    start_qvel: wp.array2d(dtype=float),
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    step_count: wp.array(dtype=int),
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
    step_count[world] = 0


class Dynamic3DBatchEnv(abc.ABC):
    """Many parallel worlds of one dynamic3d task, on MuJoCo Warp.

    All worlds use one compiled model: the fixtures have the same poses in all
    worlds. Each world has its own initial poses of the objects and the robot.

    The arrays that ``reset`` and ``step`` return are Warp arrays on the simulation
    device, with one row per world. The environment reuses these arrays: each step
    writes the new values into them. Use ``array.numpy()`` for a NumPy copy, or
    ``warp.to_torch(array)`` for a torch tensor without a copy.

    MuJoCo Warp is not identical to CPU MuJoCo: a world and the single-world
    environment drift apart when the robot moves. Objects at rest agree, because the
    worlds use the box collider of CPU MuJoCo (see the note at ``put_model``).

    Worlds do not reset by themselves. Call ``reset(mask)`` for the worlds that are
    done.

    A subclass for a robot implements the methods in the "Robot interface" section.
    """

    def __init__(
        self,
        single_env: ConstantObjectKinDEREnv,
        num_worlds: int,
        seed: int | None = None,
        device: str | None = None,
        horizon: int = 1000,
        num_settle_steps: int = 10,
        physics_timestep: float | None = None,
        nconmax: int | None = None,
        njmax: int | None = None,
    ) -> None:
        """
        Args:
            single_env: The single-world environment of the task, made with
                ``allow_state_access=True``.
            num_worlds: Number of parallel worlds.
            seed: Random seed for the fixture poses and the initial states.
            device: Warp device, for example "cuda:0" or "cpu". Default: the
                preferred Warp device.
            horizon: Number of steps after which a world is truncated.
            num_settle_steps: Control steps that let the objects settle before the
                initial states are stored, as in the reset of the single-world
                environment.
            physics_timestep: Time of one physics step, in seconds. The control
                period must be a multiple of it. Default: the time step of the
                single-world environment.
            nconmax: Contact buffer size per world. Default: from the scene.
            njmax: Constraint buffer size per world. Default: from the scene.
        """
        self.num_worlds = num_worlds
        self.horizon = horizon
        self.device = wp.get_device(device)
        self._rng = np.random.default_rng(seed)

        # The single-world environment builds the scene and compiles the model.
        self._single_env = single_env
        single_env.reset(seed=seed)
        object_centric_env = single_env._object_centric_env
        assert isinstance(object_centric_env, ObjectCentricRobotEnv)
        self._object_centric_env = object_centric_env
        robot_env = object_centric_env._robot_env
        assert robot_env.sim is not None
        self._robot_env = robot_env
        self._mj_model = robot_env.sim.model.mj_model
        self._mj_data = robot_env.sim.data.mj_data

        # One control step is ``decimation`` physics steps.
        self.step_dt = 1.0 / robot_env.control_frequency
        if physics_timestep is None:
            physics_timestep = float(self._mj_model.opt.timestep)
        self.physics_timestep = physics_timestep
        self.decimation = int(round(self.step_dt / physics_timestep))
        assert np.isclose(self.decimation * physics_timestep, self.step_dt), (
            f"The control period ({self.step_dt} s) must be a multiple of the "
            f"physics time step ({physics_timestep} s)."
        )

        assert isinstance(single_env.observation_space, ObjectCentricBoxSpace)
        self.observation_space = single_env.observation_space
        self.action_space = single_env.action_space
        (self.obs_dim,) = self.observation_space.shape
        assert self.action_space.shape is not None
        (self.action_dim,) = self.action_space.shape

        # The contacts of the settled scene give the default buffer sizes. Read them
        # now: the samples below put the objects above their supports.
        settled_contacts = int(self._mj_data.ncon)
        settled_constraints = int(self._mj_data.nefc)

        # Each world gets its own initial poses of the objects and the robot.
        sampled_qpos = np.stack(
            [object_centric_env.sample_initial_qpos() for _ in range(num_worlds)]
        )

        # Put the model and the data on the device. A buffer that is too small can
        # stop the process, so the defaults leave a margin over the settled scene (a
        # robot in contact with objects makes more contacts). The GPU memory grows
        # with num_worlds * njmax, so a large scene permits fewer worlds.
        #
        # For two boxes, MuJoCo Warp uses its general convex collider by default.
        # That collider gives all contact points of a face the same depth, so a box
        # on a flat surface has no stiffness against tilt: small cubes rock and sink
        # into their support. This flag selects the box collider of CPU MuJoCo. It
        # is set for put_model only, because in CPU MuJoCo it has a different effect.
        # The time step is also set for put_model only: the single-world environment
        # keeps its own time step.
        # pylint: disable-next=no-member
        native_ccd_off = int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
        disable_flags = self._mj_model.opt.disableflags
        single_env_timestep = self._mj_model.opt.timestep
        self._mj_model.opt.disableflags = disable_flags | native_ccd_off
        self._mj_model.opt.timestep = physics_timestep
        self._model = mjw.put_model(self._mj_model)
        self._mj_model.opt.disableflags = disable_flags
        self._mj_model.opt.timestep = single_env_timestep
        self._data = mjw.put_data(
            self._mj_model,
            self._mj_data,
            nworld=num_worlds,
            nconmax=(
                nconmax if nconmax is not None else max(256, 2 * settled_contacts)
            ),
            njmax=njmax if njmax is not None else max(800, 2 * settled_constraints),
        )

        # Per-world arrays.
        self._action = wp.zeros((num_worlds, self.action_dim), float, self.device)
        self._step_count = wp.zeros(num_worlds, int, self.device)
        self._reset_mask = wp.zeros(num_worlds, wp.bool, self.device)
        self._start_index = self._to_device(np.arange(num_worlds), int)
        self._obs = wp.zeros((num_worlds, self.obs_dim), float, self.device)
        self._reward = wp.zeros(num_worlds, float, self.device)
        self._success = wp.zeros(num_worlds, wp.bool, self.device)
        self._terminated = wp.zeros(num_worlds, wp.bool, self.device)
        self._truncated = wp.zeros(num_worlds, wp.bool, self.device)

        self._setup_robot()
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
        self._start_qpos = self._to_device(sampled_qpos, float)
        self._start_qvel = wp.zeros((num_worlds, self._mj_model.nv), float, self.device)
        self.reset()
        wp.copy(self._action, self.default_action())
        for _ in range(num_settle_steps):
            self._launch_control_step()
            self._after_settle_step()
        wp.copy(self._start_qpos, self._data.qpos)
        wp.copy(self._start_qvel, self._data.qvel)
        self._replace_drifted_starts(sampled_qpos)
        self.reset()

    # ----- Robot interface: a subclass implements these methods -----

    @abc.abstractmethod
    def _setup_robot(self) -> None:
        """Make the device arrays of the robot (gains, limits, state of the control).

        The model and the data are on the device when this runs.
        """

    @abc.abstractmethod
    def _robot_observation_map(
        self, column: dict[str, int]
    ) -> list[ObservationMapEntry]:
        """Give the source of each robot feature in the observation.

        Args:
            column: The observation column of each feature of the robot.
        """

    @abc.abstractmethod
    def _robot_base_qpos_adr(self) -> int:
        """The address in qpos of the x position of the robot base (y is the next
        entry).

        A goal predicate on the robot uses it.
        """

    @abc.abstractmethod
    def _simulate_control_step(self) -> None:
        """One control step: the control law of the robot, which reads ``self._action``
        and writes ``self._data.ctrl``, and ``self.decimation`` physics steps."""

    @abc.abstractmethod
    def default_action(self) -> wp.array:
        """The action that holds the robot still, for all worlds."""

    def _reset_robot(self) -> None:
        """Reset the state of the control in the worlds of ``self._reset_mask``.

        The positions and velocities of these worlds are in their initial state when
        this runs. Default: nothing.
        """

    def _after_settle_step(self) -> None:
        """Runs after each control step of the settle time.

        Default: nothing.
        """

    def _get_info(self) -> dict[str, wp.array]:
        """The ``info`` of ``step``.

        Default: empty.
        """
        return {}

    # ----- Setup -----

    def _to_device(self, values: Any, dtype: Any) -> wp.array:
        return wp.array(values, dtype=dtype, device=self.device)

    def _setup_observation_map(self) -> None:
        """Find the source of each observation column that changes.

        The observation of one world is the vector of the single-world environment:
        the features of each object, in the order of the observation space. The
        columns that never change (fixture poses, bounding boxes) are written once.
        """
        mj_model = self._mj_model
        objects_dict = self._object_centric_env._objects_dict
        entries: list[ObservationMapEntry] = []

        offset = 0
        for obj in self.observation_space.constant_objects:
            features = self.observation_space.type_features[obj.type]
            column = {name: offset + i for i, name in enumerate(features)}
            if obj.name == self._object_centric_env.robot_name:
                entries.extend(self._robot_observation_map(column))
            elif obj.name in objects_dict:
                # A movable object: its pose and velocity are those of its free joint.
                joint = mj_model.joint(objects_dict[obj.name].joint_name)
                for i, name in enumerate(_FREE_JOINT_POS_FEATURES):
                    entries.append((column[name], "qpos", int(joint.qposadr[0]) + i, 1))
                for i, name in enumerate(_FREE_JOINT_VEL_FEATURES):
                    if name in column:
                        adr = int(joint.dofadr[0]) + i
                        entries.append((column[name], "qvel", adr, 1.0))
            elif obj.is_instance(MujocoDrawerObjectType):
                joint = mj_model.joint(f"{obj.name}_joint")
                entries.append((column["pos"], "qpos", int(joint.qposadr[0]), 1.0))
            offset += len(features)

        # For each source array: the observation columns, the addresses in the
        # source, and the scale factors.
        self._obs_map: dict[str, tuple[NDArray, NDArray, NDArray]] = {}
        for source in ("qpos", "qvel", "ctrl"):
            rows = [entry for entry in entries if entry[1] == source]
            self._obs_map[source] = (
                np.array([row[0] for row in rows], dtype=np.int32),
                np.array([row[2] for row in rows], dtype=np.int32),
                np.array([row[3] for row in rows], dtype=np.float32),
            )
        self._obs_qvel_column = self._obs_map["qvel"][0]
        self._check_observation_map()
        self._obs_map_wp = {
            source: (
                self._to_device(columns, int),
                self._to_device(addresses, int),
                self._to_device(scales, float),
            )
            for source, (columns, addresses, scales) in self._obs_map.items()
        }

        # Write the columns that never change, for all worlds.
        static_obs = self._single_env.get_state().astype(np.float32)
        self._obs.assign(np.tile(static_obs, (self.num_worlds, 1)))

    def _check_observation_map(self) -> None:
        """Check the observation map against the single-world environment, on a random
        state.

        This fails for a task with an object whose features are not a copy of qpos,
        qvel, or ctrl entries, which the batch environment does not support.
        """
        mj_data = self._mj_data
        sources = {"qpos": mj_data.qpos, "qvel": mj_data.qvel, "ctrl": mj_data.ctrl}
        saved = {name: values.copy() for name, values in sources.items()}
        mapped = self._single_env.get_state().copy()
        for name, values in sources.items():
            values[:] = self._rng.uniform(-1.0, 1.0, values.shape)
            columns, addresses, scales = self._obs_map[name]
            mapped[columns] = scales * values[addresses]
        expected = self.observation_space.vectorize(
            self._object_centric_env._get_object_centric_state()
        )
        for name, values in sources.items():
            values[:] = saved[name]
        mismatch = np.nonzero(~np.isclose(mapped, expected, atol=1e-5))[0]
        if len(mismatch) > 0:
            raise NotImplementedError(
                f"The batch environment cannot build observation columns "
                f"{mismatch.tolist()} of this task from qpos, qvel, and ctrl."
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
                qpos_adr.append(self._robot_base_qpos_adr())
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
            return self._to_device(np.array(values, dtype=np.int32), int)

        self._num_predicates = len(goal_predicates)
        self._require_all = int(require_all)
        self._predicate_kind = ints(kind)
        self._predicate_qpos_adr = ints(qpos_adr)
        self._predicate_on_floor = ints(on_floor)
        self._predicate_site_start = ints(site_start)
        self._predicate_site_count = ints(site_count)
        self._predicate_body_id = ints(body_id)
        self._predicate_min_up = self._to_device(
            np.array(min_up, dtype=np.float32), float
        )
        self._site_id = ints(site_ids)
        self._site_half_size = self._to_device(
            np.array(site_half_sizes, dtype=np.float32).reshape(-1, 3), wp.vec3
        )

    def _replace_drifted_starts(self, sampled_qpos: NDArray[np.float64]) -> None:
        """Replace the initial states in which an object moved too far in the settle
        time.

        A task can set ``settle_drift_limit``: objects that start in contact with
        another object are pushed away by the physics. The single-world environment
        samples such a start again. Here, a start state with too much drift is replaced
        with a copy of a random start state that is good.
        """
        env = self._object_centric_env
        drift_limit = env.task_config.get("settle_drift_limit")
        if drift_limit is None or not env._ground_object_xy:
            return
        settled_qpos = self._start_qpos.numpy()
        settled_qvel = self._start_qvel.numpy()
        drift = np.zeros(self.num_worlds)
        for name in env._ground_object_xy:
            joint = self._mj_model.joint(env._objects_dict[name].joint_name)
            xy = slice(int(joint.qposadr[0]), int(joint.qposadr[0]) + 2)
            distance = np.linalg.norm(settled_qpos[:, xy] - sampled_qpos[:, xy], axis=1)
            drift = np.maximum(drift, distance)
        good = np.nonzero(drift <= drift_limit)[0]
        bad = np.nonzero(drift > drift_limit)[0]
        if len(good) == 0:
            raise RuntimeError(
                f"In all {self.num_worlds} worlds, the ground objects drift more "
                f"than {drift_limit} m in the settle time."
            )
        replacement = self._rng.choice(good, size=len(bad))
        settled_qpos[bad] = settled_qpos[replacement]
        settled_qvel[bad] = settled_qvel[replacement]
        self._start_qpos.assign(settled_qpos)
        self._start_qvel.assign(settled_qvel)

    # ----- Simulation -----

    def _launch_control_step(self) -> None:
        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self._simulate_control_step()

    def _update_observation(self) -> None:
        sources = {
            "qpos": self._data.qpos,
            "qvel": self._data.qvel,
            "ctrl": self._data.ctrl,
        }
        for name, (columns, addresses, scales) in self._obs_map_wp.items():
            if len(columns) == 0:
                continue
            wp.launch(
                _gather_kernel,
                dim=(self.num_worlds, len(columns)),
                inputs=[sources[name], addresses, columns, scales],
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
        mjw.reset_data(self._model, self._data, reset=self._reset_mask)
        wp.launch(
            _reset_state_kernel,
            dim=self.num_worlds,
            inputs=[
                self._reset_mask,
                self._start_index,
                self._start_qpos,
                self._start_qvel,
            ],
            outputs=[self._data.qpos, self._data.qvel, self._step_count],
            device=self.device,
        )
        self._reset_robot()
        # Update the body and site poses for the new states.
        mjw.forward(self._model, self._data)
        self._update_observation()
        return self._obs

    def step(
        self, action: Any
    ) -> tuple[wp.array, wp.array, wp.array, wp.array, dict[str, wp.array]]:
        """Advance all worlds by one control step (``step_dt`` seconds).

        Args:
            action: The actions, with shape (num_worlds, action_dim). One row is an
                action of the single-world environment. A Warp array, a NumPy
                array, or an array with DLPack support (for example a torch tensor).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info), with one
            row per world. ``terminated`` is true where the goal is reached.
        """
        self._copy_to(self._action, action, np.float32)
        self._launch_control_step()
        self._update_observation()
        self._update_success()
        wp.launch(
            _episode_kernel,
            dim=self.num_worlds,
            inputs=[self._success, self.horizon],
            outputs=[
                self._step_count,
                self._reward,
                self._terminated,
                self._truncated,
            ],
            device=self.device,
        )
        info = self._get_info()
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
