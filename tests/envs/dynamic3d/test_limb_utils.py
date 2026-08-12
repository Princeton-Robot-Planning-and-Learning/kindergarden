"""Tests for the limb object types and the torque-control utilities."""

import numpy as np
import pytest
from relational_structs import Object
from relational_structs.utils import create_state_from_dict

from kinder.envs.dynamic3d.limb_object_types import (
    Limb3DEnvTypeFeatures,
    Limb3DFixtureType,
    Limb3DLimbType,
    Limb3DRobotType,
)
from kinder.envs.dynamic3d.limb_utils import (
    LIMB_JOINT_INFOS,
    NUM_LIMB_JOINTS,
    NUM_ROBOT_JOINTS,
    LimbRepositioning3DObjectCentricState,
    LimbRepositioning3DRobotActionSpace,
    get_torque_action_from_gui_input,
    joint_position_distance,
)


def _limited_joint_info(index: int):
    """A joint with finite limits, which must not be wrapped."""
    return LIMB_JOINT_INFOS[index]._replace(jointLowerLimit=-2.66, jointUpperLimit=2.66)


def _make_state(
    limb_joints,
    goal_joints,
    limb_velocities=None,
    robot_joints=None,
    base=(0.0, 0.0, 0.0),
):
    """Build a state with one robot and one limb, for the property lookups."""
    limb_velocities = limb_velocities or [0.0] * NUM_LIMB_JOINTS
    robot_joints = robot_joints or [0.0] * NUM_ROBOT_JOINTS
    robot_feats = {
        "pos_base_x": base[0],
        "pos_base_y": base[1],
        "pos_base_rot": base[2],
    }
    for i, value in enumerate(robot_joints):
        robot_feats[f"joint_{i + 1}"] = value
        robot_feats[f"joint_vel_{i + 1}"] = 0.0
    limb_feats = {}
    for i in range(NUM_LIMB_JOINTS):
        limb_feats[f"joint_{i + 1}"] = limb_joints[i]
        limb_feats[f"joint_vel_{i + 1}"] = limb_velocities[i]
        limb_feats[f"goal_joint_{i + 1}"] = goal_joints[i]
    state = create_state_from_dict(
        {
            Object("robot", Limb3DRobotType): robot_feats,
            Object("limb", Limb3DLimbType): limb_feats,
        },
        Limb3DEnvTypeFeatures,
        state_cls=LimbRepositioning3DObjectCentricState,
    )
    assert isinstance(state, LimbRepositioning3DObjectCentricState)
    return state


def test_object_type_feature_counts():
    """The types carry positions and velocities for torque control."""
    assert len(Limb3DEnvTypeFeatures[Limb3DRobotType]) == 3 + 2 * NUM_ROBOT_JOINTS
    assert len(Limb3DEnvTypeFeatures[Limb3DLimbType]) == 3 * NUM_LIMB_JOINTS
    assert len(Limb3DEnvTypeFeatures[Limb3DFixtureType]) == 7


def test_joint_position_distance_is_zero_for_identical_configurations():
    """A configuration is zero distance from itself."""
    positions = [0.1, -0.2, 0.3, 1.0, -1.0, 0.0]
    assert joint_position_distance(
        LIMB_JOINT_INFOS, positions, positions
    ) == pytest.approx(0.0)


def test_joint_position_distance_wraps_a_full_turn():
    """A circular joint wound a full turn counts as being where it looks."""
    positions = [0.0] * NUM_LIMB_JOINTS
    wound = [2 * np.pi] + [0.0] * (NUM_LIMB_JOINTS - 1)
    assert joint_position_distance(LIMB_JOINT_INFOS, positions, wound) == pytest.approx(
        0.0, abs=1e-9
    )


def test_joint_position_distance_takes_the_short_way_around():
    """Crossing pi is measured the short way, not the long way."""
    a = [0.9 * np.pi] + [0.0] * (NUM_LIMB_JOINTS - 1)
    b = [-0.9 * np.pi] + [0.0] * (NUM_LIMB_JOINTS - 1)
    assert joint_position_distance(LIMB_JOINT_INFOS, a, b) == pytest.approx(0.2 * np.pi)


def test_joint_position_distance_sums_over_joints():
    """Per-joint differences combine as a weighted sum of absolute values."""
    a = [0.0] * NUM_LIMB_JOINTS
    b = [0.3, 0.4] + [0.0] * (NUM_LIMB_JOINTS - 2)
    assert joint_position_distance(LIMB_JOINT_INFOS, a, b) == pytest.approx(0.7)


def test_joint_position_distance_applies_weights():
    """Weights scale each joint's contribution."""
    a = [0.0] * NUM_LIMB_JOINTS
    b = [0.3, 0.4] + [0.0] * (NUM_LIMB_JOINTS - 2)
    weights = [2.0, 0.5] + [1.0] * (NUM_LIMB_JOINTS - 2)
    assert joint_position_distance(
        LIMB_JOINT_INFOS, a, b, weights=weights
    ) == pytest.approx(0.8)


def test_joint_position_distance_does_not_wrap_limited_joints():
    """A limited joint cannot pass through the wrap point, so it is not wrapped."""
    limited = [_limited_joint_info(i) for i in range(NUM_LIMB_JOINTS)]
    a = [2.6] + [0.0] * (NUM_LIMB_JOINTS - 1)
    b = [-2.6] + [0.0] * (NUM_LIMB_JOINTS - 1)
    assert joint_position_distance(limited, a, b) == pytest.approx(5.2)
    assert joint_position_distance(LIMB_JOINT_INFOS, a, b) == pytest.approx(
        2 * np.pi - 5.2
    )


def test_action_space_shape_and_bounds():
    """The action space is one torque per arm joint, within the given limits."""
    space = LimbRepositioning3DRobotActionSpace(
        [-2.0] * NUM_ROBOT_JOINTS, [3.0] * NUM_ROBOT_JOINTS
    )
    assert space.shape == (NUM_ROBOT_JOINTS,)
    assert np.all(space.low == -2.0)
    assert np.all(space.high == 3.0)
    for _ in range(5):
        assert space.contains(space.sample())


def test_action_space_rejects_mismatched_limits():
    """Limits that are not one per arm joint are a programming error."""
    with pytest.raises(AssertionError):
        LimbRepositioning3DRobotActionSpace([-1.0], [1.0])


def test_action_space_markdown_has_a_row_per_joint():
    """The generated docs table describes every joint torque."""
    space = LimbRepositioning3DRobotActionSpace(
        [-1.0] * NUM_ROBOT_JOINTS, [1.0] * NUM_ROBOT_JOINTS
    )
    description = space.create_markdown_description()
    assert "| **Index** | **Description** |" in description
    for i in range(1, NUM_ROBOT_JOINTS + 1):
        assert f"| {i - 1} | torque applied to robot joint {i} |" in description


def test_gui_input_is_zero_without_a_selected_joint():
    """No number key held means no torque."""
    space = LimbRepositioning3DRobotActionSpace(
        [-1.0] * NUM_ROBOT_JOINTS, [1.0] * NUM_ROBOT_JOINTS
    )
    action = get_torque_action_from_gui_input(
        space, {"keys": set(), "left_stick": (0.0, 1.0)}
    )
    assert np.all(action == 0.0)


def test_gui_input_drives_only_the_selected_joint():
    """Holding a number key applies torque to that joint alone."""
    space = LimbRepositioning3DRobotActionSpace(
        [-1.0] * NUM_ROBOT_JOINTS, [2.0] * NUM_ROBOT_JOINTS
    )
    action = get_torque_action_from_gui_input(
        space, {"keys": {"3"}, "left_stick": (0.0, 1.0)}
    )
    assert action[2] == pytest.approx(2.0)
    assert np.all(np.delete(action, 2) == 0.0)


def test_gui_input_scales_by_the_matching_limit_sign():
    """A negative stick scales by the lower limit, so the torque stays in bounds."""
    space = LimbRepositioning3DRobotActionSpace(
        [-4.0] * NUM_ROBOT_JOINTS, [2.0] * NUM_ROBOT_JOINTS
    )
    action = get_torque_action_from_gui_input(
        space, {"keys": {"1"}, "left_stick": (0.0, -1.0)}
    )
    assert action[0] == pytest.approx(-4.0)
    assert space.contains(action)


def test_state_exposes_limb_and_robot_joints():
    """The convenience properties read back what was written."""
    limb_joints = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    goal_joints = [0.0] * NUM_LIMB_JOINTS
    velocities = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    robot_joints = [0.7] * NUM_ROBOT_JOINTS
    state = _make_state(
        limb_joints, goal_joints, velocities, robot_joints, base=(1.0, 2.0, 0.5)
    )
    assert np.allclose(state.limb_joint_positions, limb_joints)
    assert np.allclose(state.limb_goal_joint_positions, goal_joints)
    assert np.allclose(state.limb_joint_velocities, velocities)
    assert np.allclose(state.robot_joint_positions, robot_joints)
    assert np.allclose(state.robot_joint_velocities, [0.0] * NUM_ROBOT_JOINTS)
    assert state.base_pose.x == pytest.approx(1.0)
    assert state.base_pose.y == pytest.approx(2.0)
    assert state.base_pose.rot == pytest.approx(0.5)


def test_state_distance_to_goal_matches_joint_distance():
    """limb_distance_to_goal is the wrap-aware distance between limb and goal."""
    limb_joints = [0.3, 0.4, 0.0, 0.0, 0.0, 0.0]
    goal_joints = [0.0] * NUM_LIMB_JOINTS
    state = _make_state(limb_joints, goal_joints)
    assert state.limb_distance_to_goal == pytest.approx(0.7)


def test_state_distance_to_goal_is_zero_at_the_goal():
    """A limb sitting on its goal is zero distance away."""
    goal_joints = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6]
    state = _make_state(list(goal_joints), goal_joints)
    assert state.limb_distance_to_goal == pytest.approx(0.0)
