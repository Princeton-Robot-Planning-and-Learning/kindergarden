"""Tests for limbs.py."""

import numpy as np
import pybullet as p
import pytest
from pybullet_helpers.geometry import Pose

from kinder.envs.dynamic3d.limb_utils import NUM_LIMB_JOINTS
from kinder.envs.dynamic3d.limbs import (
    ALL_LIMB_NAMES,
    DEFAULT_BODY_MASS,
    DEFAULT_RANGE_OF_MOTION,
    LIMB_NAME_TO_CLS,
    BodyMass,
    BoxJointLimitsModel,
    HumanLimbPyBulletRobot,
    NoJointLimitsModel,
    NoMuscleToneModel,
    SpringMuscleToneModel,
    create_human_limb,
    create_joint_limits_model,
    create_muscle_tone_model,
    create_range_of_motion,
    get_human_joint_limits,
    range_of_motion_admits,
    sample_range_of_motion,
    validate_human_joint_positions,
)


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
    model = NoJointLimitsModel(NUM_LIMB_JOINTS)
    assert model.check_joint_limits([1e6] * NUM_LIMB_JOINTS)
    lower, upper = model.get_joint_limits()
    assert lower == [-np.inf] * NUM_LIMB_JOINTS
    assert upper == [np.inf] * NUM_LIMB_JOINTS


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
    lower, upper = get_human_joint_limits(limb_name)
    assert len(lower) == len(upper) == NUM_LIMB_JOINTS
    assert all(lo <= up for lo, up in zip(lower, upper, strict=True))


def test_joint_ranges_reject_unknown_limbs():
    """An unknown limb name is an error rather than a silent default."""
    with pytest.raises(ValueError, match="Unknown human limb"):
        get_human_joint_limits("human-third-arm")


def test_elbow_and_knee_bend_one_way():
    """The hinge joint is one-directional, and mirrored between left and right arms."""
    left_lower, left_upper = get_human_joint_limits("human-left-arm")
    right_lower, right_upper = get_human_joint_limits("human-right-arm")
    assert left_lower[3] == 0.0 and left_upper[3] > 0.0
    assert right_lower[3] < 0.0 and right_upper[3] == 0.0
    for leg in ("human-left-leg", "human-right-leg"):
        knee_lower, knee_upper = get_human_joint_limits(leg)
        assert knee_lower[3] == 0.0 and knee_upper[3] > 0.0


@pytest.mark.parametrize(
    "left_name,right_name,mirrored_joints",
    [
        ("human-left-arm", "human-right-arm", (0, 2, 3)),
        ("human-left-leg", "human-right-leg", (1, 2, 5)),
    ],
)
def test_joint_ranges_are_mirrored_between_left_and_right(
    left_name, right_name, mirrored_joints
):
    """The two sides share joint axes, so midline-relative motions flip sign."""
    left = np.column_stack(get_human_joint_limits(left_name))
    right = np.column_stack(get_human_joint_limits(right_name))
    for joint in range(NUM_LIMB_JOINTS):
        if joint in mirrored_joints:
            assert np.allclose(left[joint], -right[joint][::-1])
        else:
            assert np.allclose(left[joint], right[joint])


def test_validate_human_joint_positions():
    """Validation names the joint and its bounds, and passes a legal configuration."""
    validate_human_joint_positions(
        "human-left-arm", [0.0] * NUM_LIMB_JOINTS, "the test configuration"
    )
    with pytest.raises(ValueError, match="joint 3 at -60.0 deg is outside"):
        validate_human_joint_positions(
            "human-left-arm",
            [0.0, 0.0, 0.0, -np.pi / 3, 0.0, 0.0],
            "the test configuration",
        )


def test_create_range_of_motion_defaults_the_rest():
    """One named motion is changed and every other keeps its default."""
    stiff_knee = create_range_of_motion(knee_flexion=90.0)
    assert stiff_knee.knee_flexion == 90.0
    default = DEFAULT_RANGE_OF_MOTION.as_dict()
    for name, value in stiff_knee.as_dict().items():
        if name != "knee_flexion":
            assert value == default[name]


def test_range_of_motion_rejects_bad_input():
    """A typo is an error rather than a silently ignored setting."""
    with pytest.raises(ValueError, match="Unknown range of motion names"):
        create_range_of_motion(knee_flexation=90.0)
    with pytest.raises(ValueError, match="non-negative"):
        create_range_of_motion(knee_flexion=-10.0)
    with pytest.raises(ValueError, match="relative_spread"):
        sample_range_of_motion(0, relative_spread=1.0)


def test_sampled_ranges_of_motion_stay_near_the_default():
    """A draw is a plausible person: every motion within the spread of the baseline."""
    default = DEFAULT_RANGE_OF_MOTION.as_dict()
    for seed in range(10):
        person = sample_range_of_motion(seed, relative_spread=0.15)
        assert set(person) == set(ALL_LIMB_NAMES)
        for limb in person.values():
            for name, value in limb.as_dict().items():
                assert (
                    0.85 * default[name] - 1e-9 <= value <= 1.15 * default[name] + 1e-9
                )


def test_a_symmetric_person_has_one_range_of_motion():
    """Both sides of a symmetric draw are the same person; an asymmetric one is not."""
    symmetric = sample_range_of_motion(0)
    assert len(set(symmetric.values())) == 1
    asymmetric = sample_range_of_motion(0, symmetric=False)
    assert len(set(asymmetric.values())) == len(ALL_LIMB_NAMES)


def test_sampled_ranges_of_motion_are_reproducible_and_varied():
    """The same seed gives the same person, and different seeds do not."""
    assert sample_range_of_motion(0) == sample_range_of_motion(0)
    assert sample_range_of_motion(0) != sample_range_of_motion(1)
    unperturbed = sample_range_of_motion(0, relative_spread=0.0)
    assert set(unperturbed.values()) == {DEFAULT_RANGE_OF_MOTION}


def test_range_of_motion_admits():
    """A configuration is reachable for one person and not for a stiffer one."""
    bent_knee = [0.0, 0.0, 0.0, np.deg2rad(120.0), 0.0, 0.0]
    assert range_of_motion_admits("human-left-leg", bent_knee)
    stiff_knee = create_range_of_motion(knee_flexion=90.0)
    assert not range_of_motion_admits("human-left-leg", bent_knee, stiff_knee)


def test_a_custom_range_of_motion_reaches_a_loaded_limb(physics_client_id):
    """The limb reports the bounds of the person it was built for."""
    stiff_knee = create_range_of_motion(knee_flexion=90.0)
    limb = create_human_limb(
        "human-left-leg",
        Pose.identity(),
        [0.0] * NUM_LIMB_JOINTS,
        "box",
        "none",
        physics_client_id,
        stiff_knee,
    )
    assert limb.range_of_motion == stiff_knee
    assert np.isclose(limb.joint_upper_limits[3], np.deg2rad(90.0))
    assert not limb.check_joint_limits([0.0, 0.0, 0.0, np.deg2rad(120.0), 0.0, 0.0])


def test_body_mass_splits_a_person_into_segments():
    """Winter's fractions, as assistive gym applies them to a 78.4 kg male."""
    mass = BodyMass(78.4)
    assert mass.link_masses("human-right-arm") == pytest.approx(
        {"upper_arm": 2.5872, "lower_arm": 1.4896, "hand": 0.5096}
    )
    assert mass.link_masses("human-right-leg") == pytest.approx(
        {"upper_leg": 8.232, "lower_leg": 3.724, "foot": 1.0976}
    )
    # A leg is much heavier than an arm, which the placeholder masses did not capture.
    assert mass.limb_mass("human-right-leg") / mass.limb_mass(
        "human-right-arm"
    ) == pytest.approx(2.846, abs=1e-3)
    # The torso is whatever the limbs leave, so the fractions have to close.
    limbs = sum(mass.limb_mass(limb_name) for limb_name in ALL_LIMB_NAMES)
    assert mass.torso_mass == pytest.approx(43.12)
    assert limbs + mass.torso_mass == pytest.approx(mass.total)


def test_body_mass_scales_with_the_person():
    """Every segment is a fixed fraction, so a lighter person scales throughout."""
    heavy, light = BodyMass(78.4), BodyMass(62.5)
    for limb_name in ALL_LIMB_NAMES:
        assert light.limb_mass(limb_name) / heavy.limb_mass(limb_name) == pytest.approx(
            62.5 / 78.4
        )
    assert light.torso_mass / heavy.torso_mass == pytest.approx(62.5 / 78.4)


def test_body_mass_rejects_nonsense():
    """A person with no mass is a caller bug, as is an unknown limb."""
    for total in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="positive number of kg"):
            BodyMass(total)
    with pytest.raises(ValueError, match="Unknown human limb"):
        DEFAULT_BODY_MASS.link_masses("human-third-arm")


@pytest.mark.parametrize("limb_name", ALL_LIMB_NAMES)
def test_a_limb_is_built_with_a_person_s_mass(limb_name, physics_client_id):
    """The URDF ships 1 kg placeholders, so the mass has to be applied on load."""
    limb = create_human_limb(
        limb_name, Pose.identity(), None, "none", "none", physics_client_id
    )
    total = sum(
        p.getDynamicsInfo(limb.robot_id, j, physicsClientId=physics_client_id)[0]
        for j in range(p.getNumJoints(limb.robot_id, physicsClientId=physics_client_id))
    )
    assert np.isclose(total, DEFAULT_BODY_MASS.limb_mass(limb_name))
    assert not np.isclose(total, 3.0)


def test_scaling_a_limb_s_mass_keeps_the_urdf_inertia(physics_client_id):
    """Setting mass alone would swap the URDF's inertia for a bounding box."""
    limb = create_human_limb(
        "human-right-arm", Pose.identity(), None, "none", "none", physics_client_id
    )
    unscaled = p.loadURDF(
        str(limb.urdf_path),
        useFixedBase=True,
        flags=p.URDF_USE_INERTIA_FROM_FILE,
        physicsClientId=physics_client_id,
    )
    for link in (3, 4, 6):
        mass, _, inertia = p.getDynamicsInfo(
            limb.robot_id, link, physicsClientId=physics_client_id
        )[:3]
        at_one_kg = p.getDynamicsInfo(
            unscaled, link, physicsClientId=physics_client_id
        )[2]
        assert np.allclose(inertia, np.array(at_one_kg) * mass, atol=1e-4)


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
    assert not limb.check_joint_limits([1e6] * NUM_LIMB_JOINTS)
    assert limb.check_joint_limits([0.0] * NUM_LIMB_JOINTS)


@pytest.mark.parametrize("limb_name", ALL_LIMB_NAMES)
def test_a_bounded_limb_reports_its_anatomical_limits(limb_name, physics_client_id):
    """A box-limited limb reports the anatomical bounds, so IK and planning see them."""
    limb = create_human_limb(
        limb_name, Pose.identity(), None, "box", "none", physics_client_id
    )
    lower, upper = get_human_joint_limits(limb_name)
    assert np.allclose(limb.joint_lower_limits, lower)
    assert np.allclose(limb.joint_upper_limits, upper)


@pytest.mark.parametrize("limb_name", ALL_LIMB_NAMES)
def test_an_unbounded_limb_reports_infinite_limits(limb_name, physics_client_id):
    """Opting out restores the URDF's continuous joints, so nothing is ever violated."""
    limb = create_human_limb(
        limb_name, Pose.identity(), None, "none", "none", physics_client_id
    )
    assert np.all(np.isinf(limb.joint_lower_limits))
    assert np.all(np.isinf(limb.joint_upper_limits))
    assert limb.joint_limit_violation([1e6] * NUM_LIMB_JOINTS) == 0.0


def test_a_limb_cannot_be_created_outside_its_limits(physics_client_id):
    """An impossible home configuration is rejected at construction."""
    hyperextended_elbow = [0.0, 0.0, 0.0, -1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="outside its joint limits"):
        create_human_limb(
            "human-left-arm",
            Pose.identity(),
            hyperextended_elbow,
            "box",
            "none",
            physics_client_id,
        )


def test_joint_limit_violation_measures_the_worst_joint(physics_client_id):
    """The violation is the largest overshoot in radians, and zero when there is none."""
    limb = create_human_limb(
        "human-left-arm", Pose.identity(), None, "box", "none", physics_client_id
    )
    assert limb.joint_limit_violation([0.0] * NUM_LIMB_JOINTS) == 0.0
    violation = limb.joint_limit_violation([0.0, 0.0, 0.0, -0.25, 0.0, 0.1])
    assert np.isclose(violation, 0.25)
