"""Tests for the TidyBot3D Tossing3D task."""

from pathlib import Path

import numpy as np

import kinder
from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv

_TASK_CONFIG_PATH = (
    Path(kinder.__path__[0])
    / "envs"
    / "dynamic3d"
    / "tasks"
    / "Tossing3D"
    / "Tossing3D-o1.json"
)


def _make_env() -> ObjectCentricTidyBot3DEnv:
    return ObjectCentricTidyBot3DEnv(
        num_objects=1,
        task_config_path=str(_TASK_CONFIG_PATH),
        scene_bg=False,
        allow_state_access=True,
    )


def _goal_reached(env: ObjectCentricTidyBot3DEnv) -> bool:
    """Report whether the task's goal_state predicates are satisfied."""
    return env._check_goals()  # pylint: disable=protected-access


def _put_cube_at(env: ObjectCentricTidyBot3DEnv, position: np.ndarray) -> None:
    """Teleport cube_0 to the given world position."""
    state = env._get_current_state()  # pylint: disable=protected-access
    cube = env._objects_dict["cube_0"]  # pylint: disable=protected-access
    modified_state = state.copy()
    for feature, value in zip("xyz", position, strict=True):
        modified_state.set(cube.symbolic_object, feature, float(value))
    env.set_state(modified_state)


def test_tossing3d_cube_in_bin_is_a_success():
    """A cube resting in the bin must satisfy the goal.

    Tossing3D asks the robot to toss a cube into the bin, so the bin must lie inside
    blocks_goal_region. If the bin drifts outside that region, a perfectly executed toss
    scores a failure.
    """
    env = _make_env()
    env.reset(seed=0)

    # The cube starts in blocks_init_region, so the goal is not yet satisfied.
    assert not _goal_reached(env)

    # Place the cube where it comes to rest on the floor of the bin: the bin's
    # centre in x/y, and one wall thickness plus one cube half-extent up in z.
    # z is not discriminating here -- the inflated region spans z in [0, 0.15],
    # so _check_goals cannot tell a cube in the bin from one on the floor
    # beneath it. The x/y comparison is what this assertion rests on.
    state = env._get_current_state()  # pylint: disable=protected-access
    bin_obj = state.get_object_from_name("bin_0")
    bin_config = env.task_config["objects"]["bin"]["bin_0"]
    cube_size = env.task_config["objects"]["cube"]["cube_0"]["size"]
    resting_position = np.array(
        [
            state.get(bin_obj, "x"),
            state.get(bin_obj, "y"),
            bin_config["wall_thickness"] + cube_size,
        ]
    )
    _put_cube_at(env, resting_position)

    assert _goal_reached(env), (
        "A cube resting in the bin must score a success, but the bin lies "
        "outside blocks_goal_region"
    )
    env.close()


def test_tossing3d_cube_short_of_the_bin_is_not_a_success():
    """A cube on the floor that never reached the bin must not satisfy the goal.

    blocks_goal_region reaches down to the floor, so it only describes "in the bin"
    while the bin's footprint covers it. A bin offset from the region leaves floor
    positions that score a success without the cube ever entering the bin.
    """
    env = _make_env()
    env.reset(seed=0)

    state = env._get_current_state()  # pylint: disable=protected-access
    bin_obj = state.get_object_from_name("bin_0")
    bin_config = env.task_config["objects"]["bin"]["bin_0"]
    cube_size = env.task_config["objects"]["cube"]["cube_0"]["size"]

    # A point on the floor one full bin-length short of the bin: outside the
    # bin's footprint entirely, so the cube is lying on the ground.
    short_position = np.array(
        [
            state.get(bin_obj, "x") - bin_config["length"],
            state.get(bin_obj, "y"),
            cube_size,
        ]
    )
    _put_cube_at(env, short_position)

    assert not _goal_reached(
        env
    ), "A cube on the floor short of the bin must not score a success"
    env.close()


def test_tossing3d_goal_region_is_covered_by_the_bin():
    """Every position that scores a success must lie inside the bin's footprint.

    The two tests above sample single points, so they only detect a large drift.
    This one pins the invariant they rely on: because blocks_goal_region reaches
    down to the floor, it encodes "in the bin" only while the bin's footprint
    covers it in x and y.
    """
    env = _make_env()
    env.reset(seed=0)

    region = env._ground_fixture.region_objects[  # pylint: disable=protected-access
        "blocks_goal_region"
    ][0]
    x_min, y_min, _, x_max, y_max, _ = region.bbox

    state = env._get_current_state()  # pylint: disable=protected-access
    bin_obj = state.get_object_from_name("bin_0")
    bin_config = env.task_config["objects"]["bin"]["bin_0"]
    bin_x = state.get(bin_obj, "x")
    bin_y = state.get(bin_obj, "y")

    # bin_init_region samples over a 1 mm range, so allow that much slack.
    tol = 0.002
    assert bin_x - bin_config["length"] / 2 <= x_min + tol
    assert bin_x + bin_config["length"] / 2 >= x_max - tol
    assert bin_y - bin_config["width"] / 2 <= y_min + tol
    assert bin_y + bin_config["width"] / 2 >= y_max - tol
    env.close()
