"""Tests for the TidyBot3D Tossing3D task."""

from pathlib import Path

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


def _put_cube_at(env: ObjectCentricTidyBot3DEnv, x: float, y: float, z: float) -> None:
    """Teleport cube_0 to the given world position."""
    current_state = env._get_current_state()  # pylint: disable=protected-access
    cube = env._objects_dict["cube_0"]  # pylint: disable=protected-access
    modified_state = current_state.copy()
    modified_state.set(cube.symbolic_object, "x", x)
    modified_state.set(cube.symbolic_object, "y", y)
    modified_state.set(cube.symbolic_object, "z", z)
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
    # z is not discriminating here -- ground regions are inflated by
    # MujocoGround.ground_placement_threshold, so blocks_goal_region spans z in
    # [0, 0.15] and _check_goals cannot tell a cube in the bin from one on the
    # floor beneath it. The x/y comparison is what this assertion rests on.
    current_state = env._get_current_state()  # pylint: disable=protected-access
    bin_obj = current_state.get_object_from_name("bin_0")
    bin_config = env.task_config["objects"]["bin"]["bin_0"]
    cube_size = env.task_config["objects"]["cube"]["cube_0"]["size"]
    _put_cube_at(
        env,
        current_state.get(bin_obj, "x"),
        current_state.get(bin_obj, "y"),
        bin_config["wall_thickness"] + cube_size,
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
    cube_size = env.task_config["objects"]["cube"]["cube_0"]["size"]

    # A point on the floor one full bin-length short of the bin: outside the
    # bin's footprint entirely, so the cube is lying on the ground.
    _put_cube_at(
        env,
        current_state.get(bin_obj, "x") - bin_config["length"],
        current_state.get(bin_obj, "y"),
        cube_size,
    )

    assert (
        not env._check_goals()  # pylint: disable=protected-access
    ), "Goals should not be satisfied with the cube on the floor short of the bin"

    env.close()


def test_tossing3d_goal_region_is_covered_by_the_bin():
    """Test that every position scoring a success lies inside the bin's footprint.

    The two tests above sample single points, so they only detect a large drift. This
    one pins the invariant they rely on: because blocks_goal_region reaches down to the
    floor, it encodes "in the bin" only while the bin's footprint covers it in x and y.
    """
    env = _make_env()
    env.reset(seed=0)

    # _check_goals tests membership against the region's inflated bounding box,
    # so compare against that rather than against the task config's raw ranges.
    region = env._ground_fixture.region_objects[  # pylint: disable=protected-access
        "blocks_goal_region"
    ][0]
    x_min, y_min, _, x_max, y_max, _ = region.bbox

    current_state = env._get_current_state()  # pylint: disable=protected-access
    bin_obj = current_state.get_object_from_name("bin_0")
    bin_config = env.task_config["objects"]["bin"]["bin_0"]
    bin_x = current_state.get(bin_obj, "x")
    bin_y = current_state.get(bin_obj, "y")

    # bin_init_region spans 1 mm in x and in y, so the sampled bin pose sits up to
    # that far off the nominal one; 2 mm of slack covers it.
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
