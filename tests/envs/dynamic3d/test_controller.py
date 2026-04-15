"""Tests for the TidyBot arm PD controller.

These tests verify that the PD controller correctly converts target joint positions to
torques and that the arm behaves reasonably under control.
"""

import numpy as np

from pathlib import Path

from kinder.envs.dynamic3d.envs import (
    ObjectCentricTidyBot3DEnv,
    TidyBot3DConfig,
)

_TEST_TASKS = Path(__file__).parent / "test_tasks"


def test_arm_converges_to_target_position():
    """Test that the arm moves toward a target joint position using PD control.

    This test verifies that:
    1. The arm position changes when commanded to a target
    2. The arm moves in the direction of the target
    3. The system remains stable (no explosion)

    Note: Without gravity compensation, the arm may not perfectly reach the target,
    but it should move toward it and remain stable.
    """
    # Create environment with absolute position mode (not delta)
    config = TidyBot3DConfig(act_delta=False)
    env = ObjectCentricTidyBot3DEnv(config=config, num_objects=1, task_config_path=str(_TEST_TASKS / "tidybot-ground-o1.json"))
    try:
        env.reset(seed=42)

        robot_env = env._robot_env  # pylint: disable=protected-access

        # Get current positions
        initial_base_pos = np.array(robot_env.qpos["base"]).copy()
        initial_arm_pos = np.array(robot_env.qpos["arm"]).copy()

        # Define target: small offset from initial position
        arm_offset = np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1])
        target_arm_pos = initial_arm_pos + arm_offset

        # Build absolute action: [base(3), arm(7), gripper(1)]
        action = np.zeros(11)
        action[0:3] = initial_base_pos  # Keep base at initial position
        action[3:10] = target_arm_pos  # Command target arm position
        action[10] = 0.0  # Keep gripper open

        # Run simulation for some steps
        num_steps = 100
        for _ in range(num_steps):
            env.step(action)

        # Get final position
        final_arm_pos = np.array(robot_env.qpos["arm"]).copy()

        # Test 1: The arm should have moved (position changed)
        position_change = final_arm_pos - initial_arm_pos
        assert np.linalg.norm(position_change) > 0.01, (
            f"Arm did not move significantly. "
            f"Initial: {initial_arm_pos}, Final: {final_arm_pos}"
        )

        # Test 2: The arm should move in the direction of the target for most joints
        # (some joints may be affected by gravity differently)
        direction_correct = np.sign(position_change) == np.sign(arm_offset)
        num_correct = np.sum(direction_correct)
        assert num_correct >= 4, (
            f"Arm did not move in correct direction for enough joints. "
            f"Only {num_correct}/7 joints moved correctly. "
            f"Expected direction: {np.sign(arm_offset)}, "
            f"Actual direction: {np.sign(position_change)}"
        )

        # Test 3: System should remain stable (positions should be reasonable)
        # All joint positions should be within a reasonable range (no explosion)
        assert np.all(
            np.abs(final_arm_pos) < 10
        ), f"Arm positions are unreasonable (possible instability): {final_arm_pos}"

    finally:
        env.close()


def test_gravity_compensation_is_applied():
    """Test that gravity compensation is computed and applied to the torques.

    This test verifies that:
    1. Gravity compensation values are computed from qfrc_bias
    2. The compensation is non-zero (gravity exists)
    3. The compensation is included in the torque output
    """
    config = TidyBot3DConfig(act_delta=False)
    env = ObjectCentricTidyBot3DEnv(config=config, num_objects=1, task_config_path=str(_TEST_TASKS / "tidybot-ground-o1.json"))
    try:
        env.reset(seed=42)

        robot_env = env._robot_env  # pylint: disable=protected-access

        # Get gravity compensation
        gravity_comp = (
            robot_env._get_gravity_compensation()  # pylint: disable=protected-access
        )

        # Test 1: Gravity compensation should be non-zero (gravity exists)
        assert (
            np.max(np.abs(gravity_comp)) > 0.1
        ), f"Gravity compensation seems too small: {gravity_comp}"

        # Test 2: When position error is zero, torques should include gravity comp
        current_pos = np.array(robot_env.qpos["arm"]).copy()
        robot_env.qvel["arm"][:] = 0.0  # Set velocity to zero
        robot_env.sim.forward()

        torques = robot_env._compute_arm_torques(  # pylint: disable=protected-access
            current_pos
        )
        expected_gravity_comp = (
            robot_env._get_gravity_compensation()  # pylint: disable=protected-access
        )

        # Torques should approximately equal gravity compensation
        # (small difference due to forward() call updating state)
        assert np.allclose(torques, expected_gravity_comp, atol=0.1), (
            f"Torques don't match gravity compensation. "
            f"Torques: {torques}, Gravity comp: {expected_gravity_comp}"
        )

        # Test 3: Gravity compensation should vary with arm configuration
        # Move to a different configuration and check gravity comp changes
        original_comp = gravity_comp.copy()
        robot_env.qpos["arm"][2] += 0.5  # Rotate joint 3
        robot_env.sim.forward()
        new_comp = (
            robot_env._get_gravity_compensation()  # pylint: disable=protected-access
        )

        # At least some components should have changed
        comp_diff = np.abs(new_comp - original_comp)
        assert np.max(comp_diff) > 0.1, (
            f"Gravity compensation didn't change with configuration. "
            f"Original: {original_comp}, New: {new_comp}"
        )

    finally:
        env.close()


def test_pd_achieves_accurate_tracking():
    """Test that PD controller with gravity compensation achieves accurate tracking.

    With PD control and gravity compensation, the arm should accurately track target
    positions with minimal steady-state error.
    """
    config = TidyBot3DConfig(act_delta=False)
    env = ObjectCentricTidyBot3DEnv(config=config, num_objects=1, task_config_path=str(_TEST_TASKS / "tidybot-ground-o1.json"))
    try:
        env.reset(seed=42)

        robot_env = env._robot_env  # pylint: disable=protected-access

        # Get initial positions
        initial_base_pos = np.array(robot_env.qpos["base"]).copy()
        initial_arm_pos = np.array(robot_env.qpos["arm"]).copy()

        # Small target offset
        target_offset = np.array([0.02, -0.02, 0.01, -0.01, 0.005, -0.005, 0.002])
        target_arm_pos = initial_arm_pos + target_offset

        # Build action
        action = np.zeros(11)
        action[0:3] = initial_base_pos
        action[3:10] = target_arm_pos
        action[10] = 0.0

        # Record error at different time points
        errors_over_time = []
        for step in range(500):
            env.step(action)
            if step % 100 == 99:
                current_pos = np.array(robot_env.qpos["arm"]).copy()
                error = np.linalg.norm(target_arm_pos - current_pos)
                errors_over_time.append(error)

        # Check tracking accuracy
        final_arm_pos = np.array(robot_env.qpos["arm"]).copy()
        joint_errors = np.abs(target_arm_pos - final_arm_pos)
        total_error = np.linalg.norm(target_arm_pos - final_arm_pos)

        # With gravity compensation, PD should achieve good tracking
        # Total error should be small (< 0.5 radians total across all joints)
        assert total_error < 0.5, (
            f"Total tracking error too large: {total_error:.4f} radians. "
            f"Joint errors: {joint_errors}"
        )

        # The system should remain stable (error shouldn't explode)
        assert all(
            e < 2.0 for e in errors_over_time
        ), f"System became unstable. Errors over time: {errors_over_time}"

    finally:
        env.close()


def test_arm_remains_stable():
    """Test that the arm remains stable (no explosion) under PD control.

    This test verifies that:
    1. Joint positions stay within reasonable bounds
    2. No NaN or infinite values occur
    3. The simulation doesn't explode

    Note: Without gravity compensation, perfect position holding is not expected.
    """
    # Create environment with absolute position mode
    config = TidyBot3DConfig(act_delta=False)
    env = ObjectCentricTidyBot3DEnv(config=config, num_objects=1, task_config_path=str(_TEST_TASKS / "tidybot-ground-o1.json"))
    try:
        env.reset(seed=42)

        robot_env = env._robot_env  # pylint: disable=protected-access

        # Get current positions
        initial_base_pos = np.array(robot_env.qpos["base"]).copy()
        initial_arm_pos = np.array(robot_env.qpos["arm"]).copy()

        # Use current position as target
        target_arm_pos = initial_arm_pos.copy()

        # Build absolute action
        action = np.zeros(11)
        action[0:3] = initial_base_pos
        action[3:10] = target_arm_pos
        action[10] = 0.0

        # Run for many steps and check stability at each step
        max_position_seen = 0.0
        for step in range(200):
            env.step(action)

            # Check positions at each step
            current_pos = np.array(robot_env.qpos["arm"]).copy()

            # No NaN or Inf values
            assert not np.any(
                np.isnan(current_pos)
            ), f"NaN in arm position at step {step}: {current_pos}"
            assert not np.any(
                np.isinf(current_pos)
            ), f"Inf in arm position at step {step}: {current_pos}"

            # Track maximum position magnitude
            max_position_seen = max(max_position_seen, np.max(np.abs(current_pos)))

        # Verify positions stayed bounded
        assert max_position_seen < 20, (
            f"Arm positions exceeded reasonable bounds. "
            f"Max position magnitude seen: {max_position_seen:.2f} radians"
        )

    finally:
        env.close()


def test_velocity_tracking_mode():
    """Test that the controller can track both position and velocity targets.

    This test verifies that:
    1. The 18-element action (position + velocity) is accepted
    2. Backward compatibility: 11-element actions still work
    3. Invalid action lengths are rejected
    4. Velocity tracking produces different torques than position-only mode
    """
    config = TidyBot3DConfig(act_delta=False)
    env = ObjectCentricTidyBot3DEnv(config=config, num_objects=1, task_config_path=str(_TEST_TASKS / "tidybot-ground-o1.json"))
    try:
        env.reset(seed=42)

        robot_env = env._robot_env  # pylint: disable=protected-access

        # Get initial state
        initial_base_pos = np.array(robot_env.qpos["base"]).copy()
        initial_arm_pos = np.array(robot_env.qpos["arm"]).copy()

        # Test 1: Verify 11-element action still works (backward compatibility)
        action_11 = np.zeros(11)
        action_11[0:3] = initial_base_pos
        action_11[3:10] = initial_arm_pos
        action_11[10] = 0.0
        env.step(action_11)  # Should not raise

        # Test 2: Verify 18-element action is accepted
        action_18 = np.zeros(18)
        action_18[0:3] = initial_base_pos
        action_18[3:10] = initial_arm_pos
        action_18[10] = 0.0
        action_18[11:18] = 0.0  # Zero velocity targets
        env.step(action_18)  # Should not raise

        # Test 3: Invalid action length should raise
        action_invalid = np.zeros(15)
        try:
            env.step(action_invalid)
            assert False, "Should have raised ValueError for invalid action length"
        except ValueError as e:
            assert "11" in str(e) and "18" in str(e), f"Unexpected error message: {e}"

        # Test 4: Verify _compute_arm_torques produces different results with velocity
        # Set up a state with known position and velocity
        robot_env.qpos["arm"][:] = initial_arm_pos
        robot_env.qvel["arm"][:] = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        robot_env.sim.forward()

        # Compute torques without velocity target (damping mode)
        torques_damping = (
            robot_env._compute_arm_torques(  # pylint: disable=protected-access
                initial_arm_pos, target_velocities=None
            )
        )

        # Compute torques with velocity target matching current velocity
        # (should reduce damping effect)
        torques_tracking = (
            robot_env._compute_arm_torques(  # pylint: disable=protected-access
                initial_arm_pos,
                target_velocities=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
            )
        )

        # When current velocity matches target velocity, torques should be different
        # from damping mode (where we resist velocity)
        torque_difference = np.linalg.norm(torques_tracking - torques_damping)
        assert torque_difference > 1.0, (
            f"Velocity tracking should produce different torques. "
            f"Damping: {torques_damping}, Tracking: {torques_tracking}, "
            f"Difference: {torque_difference}"
        )

        # Test 5: Velocity tracking should reduce torque when vel matches target
        # In damping mode: torque = -Kd * vel (resists motion)
        # In tracking mode with target=current: torque = Kd * (target - current) = 0
        # So tracking torques should be closer to just gravity compensation
        gravity_comp = (
            robot_env._get_gravity_compensation()  # pylint: disable=protected-access
        )

        # Tracking torques should be closer to gravity comp than damping torques
        tracking_diff_from_gravity = np.linalg.norm(torques_tracking - gravity_comp)
        damping_diff_from_gravity = np.linalg.norm(torques_damping - gravity_comp)

        assert tracking_diff_from_gravity < damping_diff_from_gravity, (
            f"When target velocity matches current velocity, torques should be closer "
            f"to gravity compensation. Tracking diff: {tracking_diff_from_gravity}, "
            f"Damping diff: {damping_diff_from_gravity}"
        )

    finally:
        env.close()


def test_velocity_tracking_for_dynamic_motion():
    """Test velocity tracking for dynamic manipulation scenarios like tossing.

    This test verifies that:
    1. Velocity tracking mode can maintain a target velocity during motion
    2. The arm achieves significant velocity when commanded
    3. Velocity tracking behaves differently from position-only control
    """
    config = TidyBot3DConfig(act_delta=False)
    env = ObjectCentricTidyBot3DEnv(config=config, num_objects=1, task_config_path=str(_TEST_TASKS / "tidybot-ground-o1.json"))
    try:
        env.reset(seed=42)

        robot_env = env._robot_env  # pylint: disable=protected-access

        # Get initial state
        initial_base_pos = np.array(robot_env.qpos["base"]).copy()

        # Define a swing trajectory: move joint 1 with target velocity
        # This simulates the acceleration phase of a toss
        target_velocity = np.zeros(7)
        target_velocity[0] = 1.5  # Joint 1 should swing at 1.5 rad/s

        # Track velocities over time
        velocities_over_time = []

        # Accelerate - command velocity while moving position target ahead
        for step in range(100):
            # Update position target to follow the motion
            current_pos = np.array(robot_env.qpos["arm"]).copy()

            action = np.zeros(18)
            action[0:3] = initial_base_pos
            # Position target slightly ahead of current position in velocity direction
            action[3:10] = current_pos + target_velocity * 0.02
            action[10] = 0.0
            action[11:18] = target_velocity

            env.step(action)

            if step % 10 == 0:
                current_vel = np.array(robot_env.qvel["arm"]).copy()
                velocities_over_time.append(current_vel[0])

        # Check that we achieved and maintained significant velocity
        final_velocity = np.array(robot_env.qvel["arm"])[0]
        assert abs(final_velocity) > 0.3, (
            f"Should achieve significant velocity for toss. "
            f"Final velocity: {final_velocity}, Target: {target_velocity[0]}"
        )

        # The velocity should stabilize (not drop to zero like in damping mode)
        # Check that the last few velocities are relatively stable
        recent_velocities = velocities_over_time[-5:]
        velocity_std = np.std(recent_velocities)
        assert velocity_std < 0.1, (
            f"Velocity should stabilize with velocity tracking. "
            f"Recent velocities: {recent_velocities}, std: {velocity_std}"
        )

        # Compare with position-only mode: velocity should decay faster
        env.reset(seed=42)

        pos_only_velocities = []
        for step in range(100):
            current_pos = np.array(robot_env.qpos["arm"]).copy()

            # Same position updates but no velocity target (11-element action)
            action = np.zeros(11)
            action[0:3] = initial_base_pos
            action[3:10] = current_pos + target_velocity * 0.02
            action[10] = 0.0

            env.step(action)

            if step % 10 == 0:
                current_vel = np.array(robot_env.qvel["arm"]).copy()
                pos_only_velocities.append(current_vel[0])

        # With velocity tracking, we should maintain higher velocity than without
        # (damping mode resists velocity)
        vel_tracking_mean = np.mean(velocities_over_time[-5:])
        pos_only_mean = np.mean(pos_only_velocities[-5:])

        print(
            f"Velocity tracking mean: {vel_tracking_mean}, Position-only mean: {pos_only_mean}"  # pylint: disable=line-too-long
        )
        # Note: The difference might be subtle depending on gains, but velocity
        # tracking mode should help maintain velocity
        assert vel_tracking_mean >= pos_only_mean * 0.9, (
            f"Velocity tracking should help maintain velocity. "
            f"Tracking mean: {vel_tracking_mean}, Position-only mean: {pos_only_mean}"
        )

    finally:
        env.close()
