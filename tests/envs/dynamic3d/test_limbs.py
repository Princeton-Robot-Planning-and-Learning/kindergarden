"""Tests for limbs.py."""

import numpy as np
import pybullet as p
import pytest
from pybullet_helpers.geometry import Pose

from kinder.envs.dynamic3d.limbs import (
    ALL_LIMB_NAMES,
    LIMB_NAME_TO_CLS,
    BoxJointLimitsModel,
    HumanLimbPyBulletRobot,
    NoJointLimitsModel,
    NoMuscleToneModel,
    SpringMuscleToneModel,
    create_human_limb,
    create_joint_limits_model,
    create_muscle_tone_model,
    get_human_joint_ranges,
    get_sampling_bounds,
)
from kinder.envs.dynamic3d.limb_utils import NUM_LIMB_JOINTS


@pytest.fixture(name="physics_client_id")
def _physics_client_id():
    """A headless PyBullet connection, torn down after each test."""
    client_id = p.connect(p.DIRECT)
    yield client_id
    p.disconnect(physicsClientId=client_id)


def test_all_four_limbs_are_registered():
    """There is one class per arm and leg, keyed by name."""
    assert set(ALL_LIMB_NAMES) == {
        "human-left-arm",
        "human-right-arm",
        "human-left-leg",
        "human-right-leg",
    }
    for name in ALL_LIMB_NAMES:
        assert LIMB_NAME_TO_CLS[name].get_name() == name


def test_no_joint_limits_model_allows_everything():
    """The default model is a no-op, matching the URDFs' continuous joints."""
    model = NoJointLimitsModel()
    assert model.check_joint_limits([1e6] * NUM_LIMB_JOINTS)


def test_box_joint_limits_model_bounds_each_joint():
    """Each joint is checked independently against its own bounds."""
    model = BoxJointLimitsModel([-1.0] * NUM_LIMB_JOINTS, [1.0] * NUM_LIMB_JOINTS)
    assert model.check_joint_limits([0.0] * NUM_LIMB_JOINTS)
    assert model.check_joint_limits([1.0] * NUM_LIMB_JOINTS)
    assert not model.check_joint_limits([0.0, 0.0, 0.0, 0.0, 0.0, 1.5])
    assert not model.check_joint_limits([-1.5, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_create_joint_limits_model_by_name():
    """The factory maps names to models and rejects anything else."""
    lower, upper = [-1.0] * NUM_LIMB_JOINTS, [1.0] * NUM_LIMB_JOINTS
    assert isinstance(
        create_joint_limits_model("none", lower, upper), NoJointLimitsModel
    )
    assert isinstance(
        create_joint_limits_model("box", lower, upper), BoxJointLimitsModel
    )
    with pytest.raises(ValueError, match="Unrecognized joint limits model"):
        create_joint_limits_model("nonsense", lower, upper)


def test_no_muscle_tone_model_is_limp():
    """A limp limb applies no torque to itself."""
    model = NoMuscleToneModel()
    tone = model.get_muscle_tone([0.5] * NUM_LIMB_JOINTS, [0.1] * NUM_LIMB_JOINTS)
    assert tone == [0.0] * NUM_LIMB_JOINTS


def test_spring_muscle_tone_model_is_one_directional():
    """The spring model is clamped at zero, so it never pulls negative."""
    model = SpringMuscleToneModel(NUM_LIMB_JOINTS)
    for positions in ([0.0] * NUM_LIMB_JOINTS, [2.0] * NUM_LIMB_JOINTS):
        tone = model.get_muscle_tone(positions, [0.0] * NUM_LIMB_JOINTS)
        assert len(tone) == NUM_LIMB_JOINTS
        assert all(value >= 0.0 for value in tone)


def test_spring_muscle_tone_grows_with_displacement():
    """Pushing the limb further from its rest point meets more resistance."""
    model = SpringMuscleToneModel(NUM_LIMB_JOINTS)
    velocities = [0.0] * NUM_LIMB_JOINTS
    near = sum(model.get_muscle_tone([0.5] * NUM_LIMB_JOINTS, velocities))
    far = sum(model.get_muscle_tone([3.0] * NUM_LIMB_JOINTS, velocities))
    assert far > near


def test_create_muscle_tone_model_by_name():
    """The factory maps names to models and rejects anything else."""
    assert isinstance(
        create_muscle_tone_model("none", NUM_LIMB_JOINTS), NoMuscleToneModel
    )
    assert isinstance(
        create_muscle_tone_model("spring", NUM_LIMB_JOINTS), SpringMuscleToneModel
    )
    with pytest.raises(ValueError, match="Unrecognized muscle tone model"):
        create_muscle_tone_model("nonsense", NUM_LIMB_JOINTS)


@pytest.mark.parametrize("limb_name", ALL_LIMB_NAMES)
def test_joint_ranges_are_well_formed(limb_name):
    """Every limb has one ordered range per joint."""
    ranges = get_human_joint_ranges(limb_name)
    assert len(ranges) == NUM_LIMB_JOINTS
    for lower, upper in ranges:
        assert lower <= upper


def test_joint_ranges_reject_unknown_limbs():
    """An unknown limb name is an error rather than a silent default."""
    with pytest.raises(ValueError, match="Unknown human limb"):
        get_human_joint_ranges("human-third-arm")


def test_elbow_and_knee_bend_one_way():
    """The hinge joint is one-directional, and mirrored between left and right arms."""
    left_arm = get_human_joint_ranges("human-left-arm")[3]
    right_arm = get_human_joint_ranges("human-right-arm")[3]
    assert left_arm[0] == 0.0 and left_arm[1] > 0.0
    assert right_arm[0] < 0.0 and right_arm[1] == 0.0
    for leg in ("human-left-leg", "human-right-leg"):
        knee = get_human_joint_ranges(leg)[3]
        assert knee[0] == 0.0 and knee[1] > 0.0


@pytest.mark.parametrize("limb_name", ALL_LIMB_NAMES)
def test_sampling_bounds_contain_the_nominal_configuration(limb_name):
    """Bounds are widened so a nominal pose outside the anatomical range still fits."""
    nominal = np.full(NUM_LIMB_JOINTS, 5.0)
    lower, upper = get_sampling_bounds(limb_name, nominal)
    assert np.all(lower <= nominal)
    assert np.all(nominal <= upper)


@pytest.mark.parametrize("limb_name", ALL_LIMB_NAMES)
def test_sampling_bounds_keep_the_anatomical_range(limb_name):
    """A nominal pose inside the range leaves the range untouched."""
    nominal = np.zeros(NUM_LIMB_JOINTS)
    lower, upper = get_sampling_bounds(limb_name, nominal)
    ranges = np.array(get_human_joint_ranges(limb_name))
    assert np.all(lower <= ranges[:, 0])
    assert np.all(upper >= ranges[:, 1])


@pytest.mark.parametrize("limb_name", ALL_LIMB_NAMES)
def test_create_human_limb(limb_name, physics_client_id):
    """Every limb loads with six joints and the grasp frame the weld needs."""
    limb = create_human_limb(
        limb_name,
        Pose.identity(),
        None,
        "none",
        "none",
        physics_client_id,
    )
    assert isinstance(limb, HumanLimbPyBulletRobot)
    assert len(limb.arm_joints) == NUM_LIMB_JOINTS
    assert limb.end_effector_name == "grasp_fixed_joint"
    assert limb.tool_link_name == "ee_link"
    assert limb.get_end_effector_pose() is not None


def test_create_human_limb_rejects_unknown_names(physics_client_id):
    """An unknown limb name is an error rather than a silent default."""
    with pytest.raises(ValueError, match="Unknown human limb"):
        create_human_limb(
            "human-third-arm",
            Pose.identity(),
            None,
            "none",
            "none",
            physics_client_id,
        )


@pytest.mark.parametrize("limb_name", ALL_LIMB_NAMES)
def test_limb_starts_at_its_home_joint_positions(limb_name, physics_client_id):
    """Passing home joint positions places the limb there."""
    home = [0.1, 0.2, 0.3, -0.4, 0.5, -0.6]
    limb = create_human_limb(
        limb_name, Pose.identity(), home, "none", "none", physics_client_id
    )
    assert np.allclose(limb.get_joint_positions(), home, atol=1e-6)


def test_limb_muscle_tone_torque_matches_its_model(physics_client_id):
    """A limp limb reports no self-torque; a spring limb reports some."""
    limp = create_human_limb(
        "human-left-arm", Pose.identity(), None, "none", "none", physics_client_id
    )
    assert limp.get_muscle_tone_torque() == [0.0] * NUM_LIMB_JOINTS

    springy = create_human_limb(
        "human-right-arm", Pose.identity(), None, "none", "spring", physics_client_id
    )
    assert isinstance(springy.muscle_tone_model, SpringMuscleToneModel)
    assert len(springy.get_muscle_tone_torque()) == NUM_LIMB_JOINTS


def test_box_joint_limits_apply_to_a_loaded_limb(physics_client_id):
    """The limb defers to its joint limits model once construction has finished."""
    limb = create_human_limb(
        "human-left-arm", Pose.identity(), None, "box", "none", physics_client_id
    )
    # The URDF joints are continuous, so the box bounds are infinite and admit anything.
    assert limb.check_joint_limits([1e6] * NUM_LIMB_JOINTS)
