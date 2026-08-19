"""Tests for the TidyBot3D Tossing3D task."""

from pathlib import Path

import pytest

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
