"""Tests for the TidyBot3D Tossing3D task."""

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

import kinder
from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv, TidyBot3DConfig
from kinder.envs.dynamic3d.robots.tidybot_robot_env import TidyBot3DRobotActionSpace
from kinder.envs.dynamic3d.task_families import Tossing3DEnv

_TASK_CONFIG_PATH = (
    Path(kinder.__path__[0])
    / "envs"
    / "dynamic3d"
    / "tasks"
    / "Tossing3D"
    / "Tossing3D-o1.json"
)


def _make_env(count: int = 1) -> ObjectCentricTidyBot3DEnv:
    return ObjectCentricTidyBot3DEnv(
        num_objects=count,
        task_config_path=str(_TASK_CONFIG_PATH.with_name(f"Tossing3D-o{count}.json")),
        scene_bg=False,
        allow_state_access=True,
    )


def _put_cube_at(env: ObjectCentricTidyBot3DEnv, x: float, y: float, z: float) -> None:
    """Teleport cube_0 to the given world position."""
    modified_state = env._get_current_state()  # pylint: disable=protected-access
    cube = modified_state.get_object_from_name("cube_0")
    modified_state.set(cube, "x", x)
    modified_state.set(cube, "y", y)
    modified_state.set(cube, "z", z)
    env.set_state(modified_state)


def test_tossing3d_cube_in_bin_is_a_success():
    """Test that a cube resting in the bin satisfies the goal.

    Tossing3D asks the robot to toss a cube into the bin, so the bin's footprint has to
    cover blocks_goal_region. If the bin drifts off the region, a perfectly executed
    toss scores a failure.
    """
    env = _make_env()
    env.reset(seed=0)

    # The cube starts in blocks_init_region, so the goal is not yet satisfied.
    assert (
        not env._check_goals()  # pylint: disable=protected-access
    ), "Goals should not be satisfied after reset"

    # Place the cube where it comes to rest on the floor of the bin: the bin's
    # centre in x/y, and one wall thickness plus one cube half-extent up in z.
    # z is not discriminating here -- blocks_goal_region now lives on bin_0, inflated
    # by MujocoObject's per-object placement threshold (1cm), so in the bin's local
    # frame the region spans z in [-0.01, 0.11] and _check_goals cannot tell a cube
    # in the bin from one on the floor beneath it. The x/y comparison is what this
    # assertion rests on.
    current_state = env._get_current_state()  # pylint: disable=protected-access
    bin_obj = current_state.get_object_from_name("bin_0")
    bin_config = env.task_config["objects"]["bin"]["bin_0"]
    cube_half_extent = env.task_config["objects"]["cube"]["cube_0"]["size"]
    _put_cube_at(
        env,
        current_state.get(bin_obj, "x"),
        current_state.get(bin_obj, "y"),
        bin_config["wall_thickness"] + cube_half_extent,
    )

    assert env._check_goals(), (  # pylint: disable=protected-access
        "Goals should be satisfied with the cube resting in the bin, but the bin "
        "lies outside blocks_goal_region"
    )

    env.close()


def test_tossing3d_cube_short_of_the_bin_is_not_a_success():
    """Test that a cube on the floor that never reached the bin fails the goal.

    blocks_goal_region reaches down to the floor, so it only describes "in the bin"
    while the bin's footprint covers it. A bin offset from the region leaves floor
    positions that score a success without the cube ever entering the bin.
    """
    env = _make_env()
    env.reset(seed=0)

    current_state = env._get_current_state()  # pylint: disable=protected-access
    bin_obj = current_state.get_object_from_name("bin_0")
    bin_config = env.task_config["objects"]["bin"]["bin_0"]
    cube_half_extent = env.task_config["objects"]["cube"]["cube_0"]["size"]

    # A point on the floor one full bin-length short of the bin: outside the
    # bin's footprint entirely, so the cube is lying on the ground.
    _put_cube_at(
        env,
        current_state.get(bin_obj, "x") - bin_config["length"],
        current_state.get(bin_obj, "y"),
        cube_half_extent,
    )

    assert (
        not env._check_goals()  # pylint: disable=protected-access
    ), "Goals should not be satisfied with the cube on the floor short of the bin"

    env.close()


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_tossing3d_goal_region_is_covered_by_the_bin(seed: int):
    """Test that the bin's footprint covers blocks_goal_region in x and y, at whatever
    position bin_init_region actually placed the bin -- swept across several seeds, since
    bin_init_region now samples a real range rather than one fixed point.

    The two tests above sample single points, so they only detect a large drift. This
    one pins the invariant they rely on: because blocks_goal_region reaches down to the
    floor, it encodes "in the bin" only while the bin's footprint covers it in x and
    y -- in the bin's own local frame the region spans z in [-0.01, 0.11], which
    reaches below the bin's floor, so a success position is not necessarily inside the
    bin at all.

    Before `blocks_goal_region.target` became `bin_0` (previously `ground`), this
    coverage held only because `bin_init_region` was a zero-width range placing the bin
    exactly on the region's own scene-fixed centre -- "no margin to spare", as this test
    used to say. Now the region is a site on the bin's own body, so coverage is
    guaranteed by construction at any bin position, which is exactly what sweeping
    several seeds here demonstrates.
    """
    env = _make_env()
    env.reset(seed=seed)

    # _check_goals tests membership against the region's inflated bounding box, so
    # compare against that rather than against the task config's raw ranges. The region
    # lives on bin_0 itself now, not the ground fixture.
    bin_obj = env._objects_dict["bin_0"]  # pylint: disable=protected-access
    region = bin_obj.region_objects["blocks_goal_region"][0]
    x_min, y_min, _, x_max, y_max, _ = region.bbox

    current_state = env._get_current_state()  # pylint: disable=protected-access
    bin_symbolic = current_state.get_object_from_name("bin_0")
    bin_config = env.task_config["objects"]["bin"]["bin_0"]
    bin_x = current_state.get(bin_symbolic, "x")
    bin_y = current_state.get(bin_symbolic, "y")

    # A few mm of slack for physics settling -- still tight enough to catch any real
    # drift between the bin's footprint and its own attached region.
    tol = 0.002
    assert (
        bin_x - bin_config["length"] / 2 <= x_min + tol
    ), "Bin does not cover the low-x edge of blocks_goal_region"
    assert (
        bin_x + bin_config["length"] / 2 >= x_max - tol
    ), "Bin does not cover the high-x edge of blocks_goal_region"
    assert (
        bin_y - bin_config["width"] / 2 <= y_min + tol
    ), "Bin does not cover the low-y edge of blocks_goal_region"
    assert (
        bin_y + bin_config["width"] / 2 >= y_max - tol
    ), "Bin does not cover the high-y edge of blocks_goal_region"

    env.close()


@pytest.mark.parametrize("count", [1, 2])
def test_tossing_goal_follows_displaced_bin(count: int):
    """Scoring follows a displaced bin for both object counts."""
    env = _make_env(count)
    try:
        env.reset(seed=0)
        state = env._get_current_state()  # pylint: disable=protected-access
        bin_obj = state.get_object_from_name("bin_0")
        original_x = state.get(bin_obj, "x")
        state.set(bin_obj, "x", original_x + 0.4)
        for i in range(count):
            cube = state.get_object_from_name(f"cube_{i}")
            state.set(cube, "x", original_x + 0.4)
            state.set(cube, "y", state.get(bin_obj, "y") + 0.06 * i)
            state.set(cube, "z", state.get(bin_obj, "z") + 0.045)
        env.set_state(state)
        assert env._check_goals()  # pylint: disable=protected-access
        for i in range(count):
            cube = state.get_object_from_name(f"cube_{i}")
            state.set(cube, "x", original_x)
        env.set_state(state)
        assert not env._check_goals()  # pylint: disable=protected-access
    finally:
        env.close()


def test_tossing_velocity_action_space() -> None:
    """Velocity targets are accepted while position/gripper bounds still apply."""
    space = TidyBot3DRobotActionSpace(use_arm_velocities=True)
    action = np.zeros(18, dtype=space.dtype)
    action[11:] = np.arange(7) * 10
    assert space.contains(action)
    action[3] = 0.101
    assert not space.contains(action)
    action[3] = 0
    action[10] = 1.01
    assert not space.contains(action)
    assert not space.contains(np.zeros(11, dtype=space.dtype))
    assert not space.contains(np.zeros((100, 18), dtype=space.dtype))
    assert TidyBot3DRobotActionSpace().shape == (11,)


def test_registered_and_direct_tossing_have_velocity_controls() -> None:
    """Both public constructors declare actions that can be executed directly."""
    kinder.register_all_environments()
    for env in (gym.make("kinder/Tossing3D-o1-v0"), Tossing3DEnv(num_objects=1)):
        try:
            env.reset(seed=0)
            assert env.action_space.shape == (18,)
            action = np.zeros(18, dtype=env.action_space.dtype)
            action[11] = 0.1
            assert env.action_space.contains(action)
            env.step(action)
        finally:
            env.close()
    config = gym.spec("kinder/Shelf3D-o1-v0").kwargs.get("config")
    assert config is None or not config.use_arm_velocities


def test_custom_horizon_preserves_tossing_velocity_controls() -> None:
    """Changing episode duration must not silently change the public action shape."""
    config = TidyBot3DConfig(horizon=2000)
    env = Tossing3DEnv(num_objects=1, config=config)
    try:
        assert env.action_space.shape == (18,)
        actual_config = (
            env._object_centric_env.config
        )  # pylint: disable=protected-access
        assert actual_config.horizon == 2000
        assert not config.use_arm_velocities
    finally:
        env.close()
