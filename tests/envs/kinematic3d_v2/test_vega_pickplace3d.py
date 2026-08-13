"""Tests for vega_pickplace3d.py."""

import numpy as np
import pytest
from gymnasium.wrappers import RecordVideo
from relational_structs.spaces import ObjectCentricBoxSpace

from tests.conftest import MAKE_VIDEOS

pytest.importorskip("prpl_kinematics")

# pylint: disable=wrong-import-position
from kinder.envs.kinematic3d_v2.object_types import (  # noqa: E402
    ARM_NUM_JOINTS,
)
from kinder.envs.kinematic3d_v2.vega_pickplace3d import (  # noqa: E402
    ARM_SIDES,
    CUBE_NODE,
    ObjectCentricVegaPickPlace3DEnv,
    VegaPickPlace3DEnv,
    VegaPickPlace3DObjectCentricState,
)

# 14 joint deltas plus 2 grasp commands.
ACTION_DIM = 2 * ARM_NUM_JOINTS + 2


@pytest.fixture(scope="module", name="env")
def env_fixture():
    """Create a shared environment for all tests in this module."""
    environment = VegaPickPlace3DEnv(render_mode="rgb_array", use_gui=False)
    if MAKE_VIDEOS:
        environment = RecordVideo(environment, "unit_test_videos")
    yield environment
    environment.close()


@pytest.fixture(scope="module", name="object_centric_env")
def object_centric_env_fixture():
    """Create a shared object-centric environment for all tests in this module."""
    environment = ObjectCentricVegaPickPlace3DEnv(
        render_mode="rgb_array", allow_state_access=True
    )
    yield environment
    environment.close()


def _state_with_cube_at(
    env: ObjectCentricVegaPickPlace3DEnv,
    position,
    holder: str | None = None,
) -> VegaPickPlace3DObjectCentricState:
    """A copy of the current state with the cube moved, and optionally held."""
    state = env.get_state().copy()
    cube = state.get_object_from_name(CUBE_NODE)
    for feature, value in zip("xyz", position):
        state.set(cube, feature, float(value))
    if holder is not None:
        state.set(state.get_object_from_name(f"{holder}_arm"), "grasping", 1.0)
    return state


def test_vega_pickplace3d_env(env):
    """Tests for basic methods in the Vega pick-place 3D env."""
    obs, _ = env.reset(seed=123)
    assert isinstance(obs, np.ndarray)
    assert env.action_space.shape == (ACTION_DIM,)

    for _ in range(10):
        act = env.action_space.sample()
        assert isinstance(act, np.ndarray)
        obs, reward, _, _, _ = env.step(act)
        assert reward == -1.0

    img = env.render()
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.dtype == np.uint8


def test_vega_pickplace3d_reset_is_deterministic(env):
    """The same seed should produce the same initial observation."""
    obs1, _ = env.reset(seed=123)
    obs2, _ = env.reset(seed=123)
    assert np.allclose(obs1, obs2)
    obs3, _ = env.reset(seed=456)
    assert not np.allclose(obs1, obs3)


def test_vega_pickplace3d_observation_space(env):
    """The vectorized observation should round-trip through the box space."""
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    vec_obs, _ = env.reset(seed=123)
    obs = env.observation_space.devectorize(vec_obs)
    assert isinstance(obs, VegaPickPlace3DObjectCentricState)
    for side in ARM_SIDES:
        assert len(obs.arm_joint_positions(side)) == ARM_NUM_JOINTS
    assert np.allclose(env.observation_space.vectorize(obs), vec_obs)


def test_vega_pickplace3d_reset_samples_on_table(object_centric_env):
    """The cube and the target should be sampled on the table, well apart."""
    config = object_centric_env.config
    for seed in range(10):
        obs, _ = object_centric_env.reset(seed=seed)
        assert not object_centric_env.goal_reached()
        assert obs.holder is None
        cube = obs.cube_position
        target = obs.target_position
        for position in (cube, target):
            assert config.sample_x_bounds[0] <= position[0] <= config.sample_x_bounds[1]
            assert config.sample_y_bounds[0] <= position[1] <= config.sample_y_bounds[1]
        assert np.isclose(cube[2], object_centric_env.cube_resting_z)
        distance = np.linalg.norm(np.subtract(cube[:2], target[:2]))
        assert distance >= config.min_cube_target_distance


def test_vega_pickplace3d_grasp_move_release(object_centric_env):
    """An arm within grasp range should pick up, carry, and drop the cube."""
    object_centric_env.reset(seed=0)
    ee_position = object_centric_env.end_effector_pose("left").t
    init_state = _state_with_cube_at(object_centric_env, ee_position)
    obs, _ = object_centric_env.reset(options={"init_state": init_state})
    assert obs.holder is None

    # Requesting a grasp within range attaches the cube to the arm.
    action = np.zeros(ACTION_DIM)
    action[2 * ARM_NUM_JOINTS] = 1.0
    obs, _, _, _, _ = object_centric_env.step(action)
    assert obs.grasping("left")
    assert not obs.grasping("right")

    # The cube moves rigidly with the holding arm.
    action = np.zeros(ACTION_DIM)
    action[0] = 0.05
    action[2 * ARM_NUM_JOINTS] = 1.0
    before = np.array(obs.cube_position)
    obs, _, _, _, _ = object_centric_env.step(action)
    after = np.array(obs.cube_position)
    assert obs.grasping("left")
    assert np.linalg.norm(after - before) > 1e-4

    # Letting go drops the cube onto the table.
    action = np.zeros(ACTION_DIM)
    action[2 * ARM_NUM_JOINTS] = -1.0
    obs, _, _, _, _ = object_centric_env.step(action)
    assert obs.holder is None
    assert np.isclose(obs.cube_position[2], object_centric_env.cube_resting_z)


def test_vega_pickplace3d_out_of_range_grasp_fails(object_centric_env):
    """A grasp request does nothing when the cube is out of reach."""
    object_centric_env.reset(seed=0)
    action = np.zeros(ACTION_DIM)
    action[2 * ARM_NUM_JOINTS] = 1.0
    action[2 * ARM_NUM_JOINTS + 1] = 1.0
    obs, _, _, _, _ = object_centric_env.step(action)
    assert obs.holder is None


def test_vega_pickplace3d_handover(object_centric_env):
    """The other arm should take the cube from the holder when within range."""
    object_centric_env.reset(seed=0)
    ee_position = object_centric_env.end_effector_pose("right").t
    init_state = _state_with_cube_at(object_centric_env, ee_position, holder="left")
    obs, _ = object_centric_env.reset(options={"init_state": init_state})
    assert obs.holder == "left"

    action = np.zeros(ACTION_DIM)
    action[2 * ARM_NUM_JOINTS] = 1.0
    action[2 * ARM_NUM_JOINTS + 1] = 1.0
    obs, _, _, _, _ = object_centric_env.step(action)
    assert obs.holder == "right"

    # The cube stays where it was through the handover.
    assert np.allclose(obs.cube_position, ee_position, atol=1e-9)


def test_vega_pickplace3d_goal(object_centric_env):
    """Dropping the cube over the target patch should reach the goal."""
    obs, _ = object_centric_env.reset(seed=0)
    target = obs.target_position
    above_target = (target[0], target[1], 0.75)
    init_state = _state_with_cube_at(object_centric_env, above_target, holder="left")
    obs, _ = object_centric_env.reset(options={"init_state": init_state})
    assert not object_centric_env.goal_reached()

    action = np.zeros(ACTION_DIM)
    action[2 * ARM_NUM_JOINTS] = -1.0
    obs, _, terminated, _, _ = object_centric_env.step(action)
    assert terminated
    assert obs.holder is None
    assert np.isclose(obs.cube_position[2], object_centric_env.cube_resting_z)


def test_vega_pickplace3d_state_access(object_centric_env):
    """Getting and setting states should round-trip, including a held cube."""
    obs, _ = object_centric_env.reset(seed=123)
    ee_position = object_centric_env.end_effector_pose("left").t
    init_state = _state_with_cube_at(object_centric_env, ee_position, holder="left")
    object_centric_env.reset(options={"init_state": init_state})
    state = object_centric_env.get_state()
    assert state.holder == "left"

    action = np.zeros(ACTION_DIM)
    action[1] = 0.05
    action[2 * ARM_NUM_JOINTS] = 1.0
    next_obs, _, _, _, _ = object_centric_env.step(action)
    assert not np.allclose(
        next_obs.arm_joint_positions("left"), state.arm_joint_positions("left")
    )

    object_centric_env.set_state(state)
    restored = object_centric_env.get_state()
    assert restored.holder == "left"
    for side in ARM_SIDES:
        assert np.allclose(
            restored.arm_joint_positions(side), state.arm_joint_positions(side)
        )
    assert np.allclose(restored.cube_position, state.cube_position)
    assert np.allclose(restored.target_position, state.target_position)

    # A held cube keeps moving with the arm after the state is restored.
    next_obs, _, _, _, _ = object_centric_env.step(action)
    assert next_obs.holder == "left"
    assert not np.allclose(next_obs.cube_position, state.cube_position)


def test_vega_pickplace3d_registered():
    """The environment should be registered under the Kinematic3Dv2 category."""
    import kinder  # pylint: disable=import-outside-toplevel

    kinder.register_all_environments()
    assert "kinder/VegaPickPlace3D-v0" in kinder.get_all_env_ids()
    assert "VegaPickPlace3D" in kinder.get_env_categories()["Kinematic3Dv2"]
