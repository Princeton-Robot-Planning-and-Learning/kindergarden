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
    # z is not discriminating here -- ground regions are inflated by
    # MujocoGround.ground_placement_threshold, so blocks_goal_region spans z in
    # [0, 0.15] and _check_goals cannot tell a cube in the bin from one on the
    # floor beneath it. The x/y comparison is what this assertion rests on.
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


def test_tossing3d_goal_region_is_covered_by_the_bin():
    """Test that the bin's footprint still covers blocks_goal_region in x and y.

    The two tests above sample single points, so they only detect a large drift. This
    one pins the invariant they rely on: because blocks_goal_region reaches down to the
    floor, it encodes "in the bin" only while the bin's footprint covers it in x and y.
    Only x and y -- in z the region spans [0, 0.15], which reaches below the bin's
    floor, so a success position is not necessarily inside the bin at all.
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

    # The bin's footprint is exactly the size of the inflated goal region -- both
    # are 0.3 x 0.3, since the region's raw half-extent of 0.10 plus 0.05 of ground
    # inflation is the bin's length / 2. So the two coincide only when the bin's
    # centre lands on the region's centre, and there is no margin to spare.
    # bin_init_region spans 1 mm in x and in y and the bin settles slightly, so in
    # practice each edge overhangs or falls short by up to about 1 mm. The tolerance
    # absorbs that placement jitter; it is not slack over an invariant that holds
    # exactly. It is still tight enough to catch any real drift -- a few mm is
    # already the regime where a cube on the bare floor scores a success.
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
