"""Tests for the TidyBot3D balance (seesaw) task.

This test suite verifies that when blocks are placed on a seesaw beam such that torques
are balanced (two small blocks on one side, one large block on the other), the balance
goal is achieved and the environment reports success.
"""

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from relational_structs import Object, ObjectCentricState
from scipy.spatial.transform import Rotation

from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv
from kinder.envs.dynamic3d.objects.generated_objects import GeneratedSeesaw

_TEST_TASKS = Path(__file__).parent / "test_tasks"


def _get_balance_env() -> ObjectCentricTidyBot3DEnv:
    """Create and return a balance task environment."""
    return ObjectCentricTidyBot3DEnv(
        scene_type="balance",
        num_objects=4,
        task_config_path=str(_TEST_TASKS / "tidybot-balance-o4.json"),
        allow_state_access=True,
    )


def _assert_block_on_seesaw(
    block: Object,
    state: ObjectCentricState,
    seesaw: GeneratedSeesaw,
    tolerance: float = 0.02,
) -> None:
    """Assert that a block is on the seesaw beam, with detailed error message.

    Args:
        block: The block object to check
        state: The current object-centric state
        seesaw: The seesaw object (uses seesaw.is_object_on_beam method)
        tolerance: Extra tolerance for position checks (default 0.02m)

    Raises:
        AssertionError: If the block is not on the seesaw beam
    """
    block_pos = np.array(
        [state.get(block, "x"), state.get(block, "y"), state.get(block, "z")],
        dtype=np.float32,
    )

    assert seesaw.is_object_on_beam(block_pos, tolerance), (
        f"Block {block.name} at position [{block_pos[0]:.3f}, {block_pos[1]:.3f}, "
        f"{block_pos[2]:.3f}] is not on the seesaw beam."
    )


def _place_block_on_seesaw(
    block: Object,
    state: ObjectCentricState,
    seesaw_pos: NDArray[Any],
    beam_height: float,
    x_offset: float,
    y_offset: float = 0.0,
    height_offset: float = 0.015,
) -> None:
    """Place a block on the seesaw at specified position.

    Args:
        block: The block object to place
        state: The object-centric state to modify (modified in place)
        seesaw_pos: The seesaw's position [x, y, z]
        beam_height: Total height of beam surface above seesaw origin
        x_offset: X offset from seesaw center (negative = left, positive = right)
        y_offset: Y offset from seesaw center (default 0)
        height_offset: Height above beam surface (default 0.015m)
    """
    state.set(block, "x", seesaw_pos[0] + x_offset)
    state.set(block, "y", seesaw_pos[1] + y_offset)
    state.set(block, "z", seesaw_pos[2] + beam_height + height_offset)
    # Set identity quaternion (upright)
    state.set(block, "qw", 1.0)
    state.set(block, "qx", 0.0)
    state.set(block, "qy", 0.0)
    state.set(block, "qz", 0.0)
    # Zero velocity
    state.set(block, "vx", 0.0)
    state.set(block, "vy", 0.0)
    state.set(block, "vz", 0.0)
    state.set(block, "wx", 0.0)
    state.set(block, "wy", 0.0)
    state.set(block, "wz", 0.0)


def test_tidybot3d_balance_observation_space():
    """Reset should return an observation within the observation space."""
    env = _get_balance_env()
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)
    env.close()


def test_tidybot3d_balance_action_space():
    """A sampled action should be valid for the action space."""
    env = _get_balance_env()
    action = env.action_space.sample()
    assert env.action_space.contains(action)
    env.close()


def test_tidybot3d_balance_step():
    """Step should return a valid obs, float reward, bool done flags, and info dict."""
    env = _get_balance_env()
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    env.close()


def test_tidybot3d_balance_seesaw_exists():
    """Verify that the seesaw object is created and accessible."""
    env = _get_balance_env()
    env.reset()

    # Check that seesaw_1 exists in the objects dict
    assert "seesaw_1" in env._objects_dict  # pylint: disable=protected-access
    seesaw = env._objects_dict["seesaw_1"]  # pylint: disable=protected-access
    assert isinstance(seesaw, GeneratedSeesaw)

    env.close()


def test_tidybot3d_balance_can_get_tilt_angle():
    """Verify that we can get the beam tilt angle from the seesaw."""
    env = _get_balance_env()
    env.reset()

    seesaw = env._objects_dict["seesaw_1"]  # pylint: disable=protected-access
    assert isinstance(seesaw, GeneratedSeesaw)

    # Get the tilt angle - should return a float
    angle = seesaw.get_beam_tilt_angle()
    assert isinstance(angle, float)

    angle_deg = seesaw.get_beam_tilt_angle_degrees()
    assert isinstance(angle_deg, float)

    # Verify conversion is correct
    assert np.isclose(np.degrees(angle), angle_deg, atol=1e-6)

    env.close()


def test_tidybot3d_balance_seesaw_starts_level():
    """After reset, the seesaw beam should start approximately level.

    The seesaw has hinge_stiffness that keeps it centered when no external forces are
    applied, so it naturally starts balanced.
    """
    env = _get_balance_env()
    env.reset()

    seesaw = env._objects_dict["seesaw_1"]  # pylint: disable=protected-access
    assert isinstance(seesaw, GeneratedSeesaw)

    # The seesaw should start approximately level due to hinge stiffness
    initial_angle = abs(seesaw.get_beam_tilt_angle_degrees())
    assert initial_angle < 10.0, (
        f"Seesaw should start approximately level. Tilt angle: "
        f"{initial_angle:.2f} degrees"
    )

    env.close()


def test_tidybot3d_balance_goal_satisfied_with_balanced_blocks():
    """Test that when blocks are placed to balance the seesaw, the goal is achieved.

    The balance task has:
    - Two small blocks (mass=0.05 each, total=0.10)
    - One large block (mass=0.10)

    When placed at equal distances from the pivot on opposite sides, the torques balance:
    - Left side: 2 * 0.05 * d = 0.10 * d
    - Right side: 0.10 * d
    """
    env = _get_balance_env()
    env.reset()

    # Get the seesaw and block objects
    seesaw = env._objects_dict["seesaw_1"]  # pylint: disable=protected-access
    assert isinstance(seesaw, GeneratedSeesaw)

    # Get seesaw position and dimensions
    seesaw_pos, _ = (
        env._robot_env.get_joint_pos_quat(  # pylint: disable=protected-access
            seesaw.joint_name
        )
    )
    beam_half_length = seesaw.beam_length / 2
    beam_height = seesaw.pivot_height + seesaw.beam_clearance + seesaw.beam_thickness

    # Get the current state
    current_state = env._get_current_state()  # pylint: disable=protected-access
    modified_state = current_state.copy()

    # Distance from pivot for block placement (use 70% of half beam length)
    placement_distance = beam_half_length * 0.7

    # Get block objects
    small_block_1 = current_state.get_object_from_name("small_block_1")
    small_block_2 = current_state.get_object_from_name("small_block_2")
    large_block = current_state.get_object_from_name("large_block")
    blocks = [small_block_1, small_block_2, large_block]

    # Get block sizes from task config
    small_block_size = 0.012

    # Place small blocks on left side (negative X direction)
    _place_block_on_seesaw(
        small_block_1,
        modified_state,
        seesaw_pos,
        beam_height,
        x_offset=-placement_distance,
        y_offset=-small_block_size,
    )
    _place_block_on_seesaw(
        small_block_2,
        modified_state,
        seesaw_pos,
        beam_height,
        x_offset=-placement_distance,
        y_offset=small_block_size,
    )
    # Place large block on right side (positive X direction)
    _place_block_on_seesaw(
        large_block,
        modified_state,
        seesaw_pos,
        beam_height,
        x_offset=placement_distance,
        y_offset=0.0,
    )

    # Set the modified state
    env.set_state(modified_state)

    # Run simulation for a few steps to let physics settle
    action_shape = env.action_space.shape
    assert action_shape is not None, "Action space shape should not be None"
    null_action = np.zeros(action_shape, dtype=np.float32)
    for _ in range(50):
        env.step(null_action)

    # Verify blocks are still on the seesaw after simulation
    final_state = env._get_current_state()  # pylint: disable=protected-access
    for block in blocks:
        _assert_block_on_seesaw(block, final_state, seesaw)

    # Check if the seesaw is balanced
    tilt_angle = abs(seesaw.get_beam_tilt_angle_degrees())
    tolerance = 5.0  # From task config goal_state

    # The seesaw should be balanced (within tolerance)
    assert seesaw.is_balanced(tolerance), (
        f"Seesaw should be balanced with equal torques. "
        f"Tilt angle: {tilt_angle:.2f} degrees (tolerance: {tolerance} degrees)"
    )

    # The goal should now be satisfied
    assert (
        env._check_goals()  # pylint: disable=protected-access
    ), "Balance goal should be satisfied when blocks are placed with equal torques"

    env.close()


def test_tidybot3d_balance_unbalanced_blocks_creates_tilt():
    """Test that placing all blocks on one side creates measurable tilt.

    Note: The seesaw has significant hinge_stiffness (1.2) which limits the
    maximum tilt angle. This test verifies that unbalanced loading creates
    SOME tilt, demonstrating the physics simulation works correctly.
    """
    env = _get_balance_env()
    env.reset()

    # Get the seesaw and record initial tilt
    seesaw = env._objects_dict["seesaw_1"]  # pylint: disable=protected-access
    assert isinstance(seesaw, GeneratedSeesaw)
    initial_tilt = seesaw.get_beam_tilt_angle_degrees()

    # Get seesaw position and dimensions
    seesaw_pos, _ = (
        env._robot_env.get_joint_pos_quat(  # pylint: disable=protected-access
            seesaw.joint_name
        )
    )
    beam_half_length = seesaw.beam_length / 2
    beam_height = seesaw.pivot_height + seesaw.beam_clearance + seesaw.beam_thickness

    # Get the current state
    current_state = env._get_current_state()  # pylint: disable=protected-access
    modified_state = current_state.copy()

    # Place ALL blocks on one side to create maximum imbalance
    small_block_1 = current_state.get_object_from_name("small_block_1")
    small_block_2 = current_state.get_object_from_name("small_block_2")
    large_block = current_state.get_object_from_name("large_block")
    blocks = [small_block_1, small_block_2, large_block]

    # Use placement near the end of the beam for maximum torque
    placement_distance = beam_half_length * 0.85
    small_block_size = 0.012

    # Place all blocks on the left side (creates torque imbalance)
    for i, block in enumerate(blocks):
        y_offset = (i - 1) * small_block_size * 2
        _place_block_on_seesaw(
            block,
            modified_state,
            seesaw_pos,
            beam_height,
            x_offset=-placement_distance,
            y_offset=y_offset,
        )

    # Set the modified state
    env.set_state(modified_state)

    # Run simulation for many steps to let physics settle
    action_shape = env.action_space.shape
    assert action_shape is not None, "Action space shape should not be None"
    null_action = np.zeros(action_shape, dtype=np.float32)
    for _ in range(100):
        env.step(null_action)

    # Verify blocks are still on the seesaw after simulation
    final_state = env._get_current_state()  # pylint: disable=protected-access
    for block in blocks:
        _assert_block_on_seesaw(block, final_state, seesaw)

    # The seesaw should have tilted from initial position
    final_tilt = seesaw.get_beam_tilt_angle_degrees()
    tilt_change = abs(final_tilt - initial_tilt)

    # Verify some tilt occurred (blocks created an imbalance)
    assert tilt_change > 0.5, (
        f"Placing all blocks on one side should create measurable tilt. "
        f"Initial: {initial_tilt:.2f}, Final: {final_tilt:.2f}, "
        f"Change: {tilt_change:.2f}"
    )

    # The tilt should be in the expected direction (negative = left side down)
    assert final_tilt < 0, (
        f"Seesaw should tilt left (negative angle) when blocks are on left side. "
        f"Tilt angle: {final_tilt:.2f} degrees"
    )

    env.close()


def test_tidybot3d_balance_reward_negative_per_timestep():
    """Test that the reward is negative per timestep (base reward)."""
    env = _get_balance_env()
    env.reset()

    action = env.action_space.sample()
    _, reward, _, _, _ = env.step(action)

    # Base reward should be -0.01 per timestep (from TidyBotRewardCalculator)
    assert reward < 0, "Reward should be negative per timestep"

    env.close()


# The shipped variant, as opposed to the o4 fixture above. It spawns the seesaw with a
# 90 degree yaw, which is what makes it the interesting case here.
_SHIPPED_TASK = "tasks/BalanceBeam3D/BalanceBeam3D-o3.json"


def _get_shipped_balancebeam_env() -> ObjectCentricTidyBot3DEnv:
    """Create the BalanceBeam3D-o3 environment that ships with kinder."""
    return ObjectCentricTidyBot3DEnv(
        scene_type="balance",
        num_objects=3,
        task_config_path=_SHIPPED_TASK,
        allow_state_access=True,
    )


def _seesaw_yaw(env: ObjectCentricTidyBot3DEnv, seesaw: GeneratedSeesaw) -> float:
    """Return the seesaw's yaw in radians."""
    _pos, quat = env._robot_env.get_joint_pos_quat(  # pylint: disable=protected-access
        seesaw.joint_name
    )
    # MuJoCo quaternions are (w, x, y, z); scipy wants (x, y, z, w).
    rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    return float(rotation.as_euler("xyz")[2])


def test_balancebeam3d_is_not_solved_by_doing_nothing():
    """The shipped task must not be satisfied by the state it starts in.

    An empty beam sits level, so a goal of ["balanced", ...] alone is true at reset:
    every policy, including one that emits zero actions, scored a solve. The goal has
    to name the cubes it wants on the beam for the task to mean anything.
    """
    env = _get_shipped_balancebeam_env()
    env.reset(seed=0)

    assert not env._check_goals()  # pylint: disable=protected-access

    action_shape = env.action_space.shape
    assert action_shape is not None
    null_action = np.zeros(action_shape, dtype=np.float32)
    for _ in range(20):
        env.step(null_action)
        assert not env._check_goals()  # pylint: disable=protected-access

    env.close()


def test_balancebeam3d_is_solved_when_the_cubes_sit_balanced_on_the_beam():
    """Placing the three cubes to balance the beam satisfies the shipped goal.

    The counterpart to the test above: tightening the goal must not make the task
    impossible. Offsets are along the beam, so this also covers the yawed seesaw.
    """
    env = _get_shipped_balancebeam_env()
    env.reset(seed=0)

    seesaw = env._objects_dict["seesaw_1"]  # pylint: disable=protected-access
    assert isinstance(seesaw, GeneratedSeesaw)
    seesaw_pos, _ = (
        env._robot_env.get_joint_pos_quat(  # pylint: disable=protected-access
            seesaw.joint_name
        )
    )
    yaw = _seesaw_yaw(env, seesaw)
    beam_height = seesaw.pivot_height + seesaw.beam_clearance + seesaw.beam_thickness
    placement_distance = (seesaw.beam_length / 2) * 0.7

    state = env._get_current_state().copy()  # pylint: disable=protected-access

    def place(name: str, along: float, across: float = 0.0) -> Object:
        """Put a cube on the beam at (along, across) in the seesaw's own frame."""
        block = state.get_object_from_name(name)
        offset = Rotation.from_euler("z", yaw).apply([along, across, 0.0])
        _place_block_on_seesaw(
            block,
            state,
            np.array(
                [seesaw_pos[0] + offset[0], seesaw_pos[1] + offset[1], seesaw_pos[2]]
            ),
            beam_height,
            x_offset=0.0,
        )
        return block

    # Two small cubes (0.05 each) against one large cube (0.10) at equal distances.
    small_block_size = 0.012
    blocks = [
        place("small_block_1", -placement_distance, -small_block_size),
        place("small_block_2", -placement_distance, small_block_size),
        place("large_block", placement_distance),
    ]
    env.set_state(state)

    action_shape = env.action_space.shape
    assert action_shape is not None
    null_action = np.zeros(action_shape, dtype=np.float32)
    for _ in range(50):
        env.step(null_action)

    final_state = env._get_current_state()  # pylint: disable=protected-access
    for block in blocks:
        _assert_block_on_seesaw(block, final_state, seesaw)
    assert seesaw.is_balanced(5.0)
    assert env._check_goals()  # pylint: disable=protected-access

    env.close()


def test_is_object_on_beam_measures_along_the_beam_not_along_world_x():
    """A yawed beam's extents follow the beam, not the world axes.

    The check compared world x against half the beam *length* and world y against half
    its *width*. On the shipped task the seesaw is yawed 90 degrees, so those two limits
    were swapped and a cube resting mid-beam read as off it -- 0.18m of beam judged
    against a 0.035m half-width.
    """
    env = _get_shipped_balancebeam_env()
    env.reset(seed=0)

    seesaw = env._objects_dict["seesaw_1"]  # pylint: disable=protected-access
    assert isinstance(seesaw, GeneratedSeesaw)
    seesaw_pos, _ = (
        env._robot_env.get_joint_pos_quat(  # pylint: disable=protected-access
            seesaw.joint_name
        )
    )
    yaw = _seesaw_yaw(env, seesaw)
    assert not np.isclose(yaw, 0.0), "this task is only interesting while it is yawed"

    beam_height = seesaw.pivot_height + seesaw.beam_clearance + seesaw.beam_thickness
    height = seesaw_pos[2] + beam_height + 0.015

    def world_point(along: float, across: float) -> NDArray[np.float32]:
        offset = Rotation.from_euler("z", yaw).apply([along, across, 0.0])
        return np.array(
            [seesaw_pos[0] + offset[0], seesaw_pos[1] + offset[1], height],
            dtype=np.float32,
        )

    near_the_end = (seesaw.beam_length / 2) * 0.9
    assert seesaw.is_object_on_beam(world_point(near_the_end, 0.0))
    assert seesaw.is_object_on_beam(world_point(-near_the_end, 0.0))
    # Past the end of the beam, and off its side, are both still off the beam.
    assert not seesaw.is_object_on_beam(world_point(seesaw.beam_length, 0.0))
    assert not seesaw.is_object_on_beam(world_point(0.0, seesaw.beam_width))

    env.close()


def test_beam_goal_region_is_derived_from_the_seesaw_it_is_attached_to():
    """The goal region's extents come from the beam, not from numbers in the task JSON.

    BalanceBeam3D-o3 declares `beam_goal_region` with a target and no ranges, so a task
    never restates beam_length or beam_width and the region cannot go stale when the
    seesaw's geometry changes.
    """
    env = _get_shipped_balancebeam_env()
    env.reset(seed=0)

    seesaw = env._objects_dict["seesaw_1"]  # pylint: disable=protected-access
    assert isinstance(seesaw, GeneratedSeesaw)

    x_min, y_min, _z_min, x_max, y_max, _z_max = seesaw.beam_surface_range()
    assert x_max - x_min == seesaw.beam_length
    assert y_max - y_min == seesaw.beam_width

    # And the region actually built from it tracks the beam through the seesaw's yaw:
    # at 90 degrees the beam's length lies along world y, its width along world x.
    (region,) = seesaw.region_objects["beam_goal_region"]
    world = region.bbox
    tolerance = 0.01  # placement_threshold, applied to every bound by _create_regions
    # Within a millimetre: the seesaw is a free body that has settled under physics, so
    # its yaw is a hair off 90 degrees and the world-aligned bound inflates by that much.
    assert np.isclose(world[3] - world[0], seesaw.beam_width + 2 * tolerance, atol=1e-3)
    assert np.isclose(
        world[4] - world[1], seesaw.beam_length + 2 * tolerance, atol=1e-3
    )


def test_a_cube_beside_the_beam_is_not_in_the_goal_region():
    """The region is the beam's footprint, not a box loose enough to catch the floor.

    Worth pinning: the region rotates with the seesaw, and a rotation that fell back to
    an axis-aligned bound would quietly widen it.
    """
    env = _get_shipped_balancebeam_env()
    env.reset(seed=0)

    seesaw = env._objects_dict["seesaw_1"]  # pylint: disable=protected-access
    assert isinstance(seesaw, GeneratedSeesaw)
    seesaw_pos, _ = (
        env._robot_env.get_joint_pos_quat(  # pylint: disable=protected-access
            seesaw.joint_name
        )
    )
    yaw = _seesaw_yaw(env, seesaw)
    (region,) = seesaw.region_objects["beam_goal_region"]

    beam_height = seesaw.pivot_height + seesaw.beam_clearance + seesaw.beam_thickness

    def world_point(along: float, across: float) -> NDArray[np.float32]:
        offset = Rotation.from_euler("z", yaw).apply([along, across, 0.0])
        return np.array(
            [
                seesaw_pos[0] + offset[0],
                seesaw_pos[1] + offset[1],
                seesaw_pos[2] + beam_height + 0.015,
            ],
            dtype=np.float32,
        )

    assert region.check_in_region(world_point(0.0, 0.0))
    assert region.check_in_region(world_point((seesaw.beam_length / 2) * 0.9, 0.0))
    # Off the side of the beam, and past its end.
    assert not region.check_in_region(world_point(0.0, seesaw.beam_width))
    assert not region.check_in_region(world_point(seesaw.beam_length, 0.0))
    # And lying on the ground beneath the beam is not on the beam.
    on_the_ground = world_point((seesaw.beam_length / 2) * 0.5, 0.0)
    on_the_ground[2] = np.float32(seesaw_pos[2] + 0.018)
    assert not region.check_in_region(on_the_ground)
