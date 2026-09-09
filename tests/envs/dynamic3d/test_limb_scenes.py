"""Tests for limb_scenes.py."""

import inspect

import numpy as np
import pybullet as p
import pytest
from pybullet_helpers.geometry import Pose, multiply_poses

from kinder.envs.dynamic3d.limb_scenes import (
    ALL_SCENE_TYPES,
    ARM_GRASP_TRANSFORM,
    LEG_GRASP_TRANSFORM,
    BedScene,
    HumanScene,
    IsolatedLimbScene,
    LimbRepositioningSceneConfig,
    RoomScene,
    WheelchairScene,
    create_scene,
)
from kinder.envs.dynamic3d.limb_utils import NUM_LIMB_JOINTS
from kinder.envs.dynamic3d.limbs import (
    ALL_LIMB_NAMES,
    DEFAULT_BODY_MASS,
    DEFAULT_RANGE_OF_MOTION,
    HumanLimbPyBulletRobot,
    create_range_of_motion,
)

INIT_JOINTS = (0.0, -0.1, 0.1, 0.0, 0.0, 0.0)
GOAL_JOINTS = (0.0, 0.2, 0.1, 0.0, 0.0, 0.0)


@pytest.fixture(name="physics_client_id")
def _physics_client_id():
    """A headless PyBullet connection, torn down after each test."""
    client_id = p.connect(p.DIRECT)
    yield client_id
    p.disconnect(physicsClientId=client_id)


def _make_config(scene_type="isolated", limb_name="human-left-arm", **kwargs):
    """A minimal valid scene config."""
    kwargs.setdefault("limb_init_joint_positions", INIT_JOINTS)
    kwargs.setdefault("limb_goal_joint_positions", GOAL_JOINTS)
    return LimbRepositioningSceneConfig(
        scene_type=scene_type,
        limb_name=limb_name,
        **kwargs,
    )


def test_all_scene_types_are_registered():
    """There are four kinds of scene, in increasing order of clutter."""
    assert set(ALL_SCENE_TYPES) == {"isolated", "human", "wheelchair", "bed"}


def test_config_rejects_unknown_scene_type():
    """An unknown scene type is caught at construction."""
    with pytest.raises(AssertionError):
        _make_config(scene_type="spaceship")


def test_config_rejects_unknown_limb_name():
    """An unknown limb name is caught at construction."""
    with pytest.raises(AssertionError):
        _make_config(limb_name="human-third-arm")


@pytest.mark.parametrize(
    "field", ["limb_init_joint_positions", "limb_goal_joint_positions"]
)
def test_config_rejects_wrong_joint_count(field):
    """Both the initial and goal configurations need one entry per joint."""
    with pytest.raises(AssertionError):
        _make_config(**{field: (0.0, 0.0)})


@pytest.mark.parametrize(
    "field", ["limb_init_joint_positions", "limb_goal_joint_positions"]
)
def test_config_rejects_anatomically_impossible_configurations(field):
    """An anatomically impossible initial or goal configuration is rejected."""
    hyperextended = (0.0, 0.0, 0.0, -1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="outside human-left-arm's joint limits"):
        _make_config(**{field: hyperextended})


def test_config_rejects_impossible_resting_configurations():
    """The limbs that are not being repositioned are held to the same limits."""
    hyperextended = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="outside human-right-arm's joint limits"):
        _make_config(scene_type="human", right_arm_init_joint_positions=hyperextended)


def test_limbs_can_have_different_ranges_of_motion():
    """One stiff side, which a single whole-person range of motion cannot express."""
    stiff = create_range_of_motion(elbow_flexion=90.0)
    config = _make_config(
        scene_type="human", limb_range_of_motion={"human-right-arm": stiff}
    )
    assert config.get_limb_range_of_motion("human-right-arm") == stiff
    for limb_name in ("human-left-arm", "human-left-leg", "human-right-leg"):
        assert config.get_limb_range_of_motion(limb_name) == DEFAULT_RANGE_OF_MOTION


def test_a_per_limb_range_of_motion_is_validated_against_that_limb():
    """The stiff limb's own bounds are what its resting configuration must fit."""
    bent_elbow = (0.0, 0.0, 0.0, -np.deg2rad(120.0), 0.0, 0.0)
    _make_config(scene_type="human", right_arm_init_joint_positions=bent_elbow)
    with pytest.raises(ValueError, match="outside human-right-arm's joint limits"):
        _make_config(
            scene_type="human",
            right_arm_init_joint_positions=bent_elbow,
            limb_range_of_motion={
                "human-right-arm": create_range_of_motion(elbow_flexion=90.0)
            },
        )


def test_config_can_opt_out_of_joint_limits():
    """Opting out is what makes the anatomically impossible configurations loadable."""
    config = _make_config(
        limb_goal_joint_positions=(0.0, 0.0, 0.0, -1.0, 0.0, 0.0),
        limb_joint_limits_model_name="none",
    )
    assert config.limb_goal_joint_positions == (0.0, 0.0, 0.0, -1.0, 0.0, 0.0)


def test_isolated_scene_keeps_the_given_limb_base_pose():
    """With no torso, the limb's base pose is used as given."""
    limb_pose = Pose.from_rpy((1.0, 2.0, 3.0), (0.0, 0.0, 0.0))
    config = _make_config(scene_type="isolated", limb_base_pose=limb_pose)
    assert np.allclose(config.limb_base_pose.position, limb_pose.position)
    assert np.allclose(
        config.get_limb_base_pose(config.limb_name).position, (1.0, 2.0, 3.0)
    )


def test_limb_base_pose_is_derived_from_the_torso():
    """By default the torso is placed and the limb follows from the attachment."""
    torso = Pose.from_rpy((0.5, 0.0, 0.0), (0.0, 0.0, 0.0))
    config = _make_config(
        scene_type="human", limb_name="human-left-arm", torso_base_pose=torso
    )
    expected = multiply_poses(torso, config.torso_to_left_arm)
    assert np.allclose(config.limb_base_pose.position, expected.position)


def test_torso_pose_is_derived_from_the_limb_when_requested():
    """use_limb_pose_to_initialize inverts the relationship."""
    limb_pose = Pose.from_rpy((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    config = _make_config(
        scene_type="human",
        limb_name="human-left-arm",
        limb_base_pose=limb_pose,
        use_limb_pose_to_initialize=True,
    )
    assert np.allclose(config.limb_base_pose.position, limb_pose.position)
    recovered = multiply_poses(config.torso_base_pose, config.torso_to_left_arm)
    assert np.allclose(recovered.position, limb_pose.position, atol=1e-6)


@pytest.mark.parametrize("limb_name", ALL_LIMB_NAMES)
def test_grasp_transform_depends_on_arm_or_leg(limb_name):
    """Arms and legs are approached from different directions."""
    config = _make_config(limb_name=limb_name)
    expected = ARM_GRASP_TRANSFORM if "arm" in limb_name else LEG_GRASP_TRANSFORM
    assert np.allclose(config.grasp_transform.position, expected.position)
    assert np.allclose(config.grasp_transform.orientation, expected.orientation)


@pytest.mark.parametrize("limb_name", ALL_LIMB_NAMES)
def test_torso_to_limb_is_defined_for_every_limb(limb_name):
    """Each of the four limbs has an attachment point on the torso."""
    config = _make_config(scene_type="human", limb_name=limb_name)
    assert np.allclose(
        config.torso_to_limb.position, config.get_torso_to_limb(limb_name).position
    )


def test_active_limb_uses_the_task_joint_positions():
    """The limb being repositioned starts at the task's initial configuration."""
    config = _make_config(scene_type="human", limb_name="human-left-arm")
    assert config.get_limb_init_joint_positions("human-left-arm") == INIT_JOINTS


def test_inactive_limbs_use_their_resting_joint_positions():
    """The other limbs rest, independent of the task."""
    resting = (-0.1,) * NUM_LIMB_JOINTS
    config = _make_config(
        scene_type="human",
        limb_name="human-left-arm",
        right_arm_init_joint_positions=resting,
    )
    assert config.get_limb_init_joint_positions("human-right-arm") == resting


@pytest.mark.parametrize(
    "scene_type,expected_cls",
    [
        ("isolated", IsolatedLimbScene),
        ("human", HumanScene),
        ("wheelchair", WheelchairScene),
        ("bed", BedScene),
    ],
)
def test_create_scene_builds_the_right_class(
    scene_type, expected_cls, physics_client_id
):
    """Every scene type builds and exposes a passive limb."""
    config = _make_config(scene_type=scene_type)
    scene = create_scene(config, physics_client_id)
    assert isinstance(scene, expected_cls)
    assert isinstance(scene.passive_limb, HumanLimbPyBulletRobot)
    assert len(scene.passive_limb.arm_joints) == NUM_LIMB_JOINTS


def test_isolated_scene_is_empty(physics_client_id):
    """Nothing to collide with when the limb floats in an empty world."""
    scene = create_scene(_make_config(scene_type="isolated"), physics_client_id)
    assert scene.get_scene_collision_ids() == []
    assert scene.get_limb_obstacle_ids() == []
    assert scene.furniture_id is None


def test_human_scene_has_four_limbs_and_a_torso(physics_client_id):
    """The whole body is present, and the other limbs are obstacles."""
    config = _make_config(scene_type="human", limb_name="human-left-arm")
    scene = create_scene(config, physics_client_id)
    assert isinstance(scene, HumanScene)
    assert set(scene.limbs) == set(ALL_LIMB_NAMES)
    assert scene.passive_limb is scene.limbs["human-left-arm"]
    obstacles = scene.get_limb_obstacle_ids()
    assert scene.torso_id in obstacles
    # The three limbs that are not being repositioned.
    assert len(obstacles) == 1 + len(ALL_LIMB_NAMES) - 1
    assert scene.furniture_id is None


def test_human_scene_torso_stays_where_it_is_put(physics_client_id):
    """Check that the torso is fixed, not dynamic, and does not fall under gravity."""
    config = _make_config(scene_type="human", limb_name="human-left-arm")
    scene = create_scene(config, physics_client_id)
    mass = p.getDynamicsInfo(scene.torso_id, -1, physicsClientId=physics_client_id)[0]
    assert mass == 0.0, "a fixed base is a zero-mass base"
    assert DEFAULT_BODY_MASS.torso_mass > 0.0, "the person still has a torso mass"

    p.setGravity(0, 0, -9.81, physicsClientId=physics_client_id)
    before = p.getBasePositionAndOrientation(
        scene.torso_id, physicsClientId=physics_client_id
    )[0]
    for _ in range(240):
        p.stepSimulation(physicsClientId=physics_client_id)
    after = p.getBasePositionAndOrientation(
        scene.torso_id, physicsClientId=physics_client_id
    )[0]
    assert np.allclose(before, after)


@pytest.mark.parametrize("scene_type", ["wheelchair", "bed"])
def test_room_scenes_have_furniture_and_walls(scene_type, physics_client_id):
    """The furniture is a collision body and the room is built around it."""
    scene = create_scene(_make_config(scene_type=scene_type), physics_client_id)
    assert scene.furniture_id is not None
    assert scene.furniture_id in scene.get_scene_collision_ids()
    assert len(scene.wall_ids) == len(scene.scene_config.wall_poses)
    assert scene.floor_id >= 0


def test_goal_limb_is_shown_when_requested(physics_client_id):
    """The translucent goal copy is created only when show_goal is set."""
    scene = create_scene(
        _make_config(scene_type="isolated", show_goal=True), physics_client_id
    )
    assert scene.goal_limb is not None
    assert np.allclose(scene.goal_limb.get_joint_positions(), GOAL_JOINTS, atol=1e-6)


def test_goal_limb_is_omitted_when_not_requested(physics_client_id):
    """No goal visualization means no extra body in the world."""
    scene = create_scene(
        _make_config(scene_type="isolated", show_goal=False), physics_client_id
    )
    assert not hasattr(scene, "goal_limb")


def test_passive_limb_starts_at_the_initial_configuration(physics_client_id):
    """The scene places the limb at the task's initial joint positions."""
    scene = create_scene(_make_config(scene_type="isolated"), physics_client_id)
    assert np.allclose(scene.passive_limb.get_joint_positions(), INIT_JOINTS, atol=1e-6)


def test_room_scenes_share_a_public_base():
    """RoomScene is the abstract base for the scenes with furniture."""
    assert inspect.isabstract(RoomScene)
    assert issubclass(WheelchairScene, RoomScene)
    assert issubclass(BedScene, RoomScene)
