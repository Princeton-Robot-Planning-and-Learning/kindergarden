"""Define object types for the TidyBot environment."""

from relational_structs import Type

MujocoObjectTypeFeatures: dict[Type, list[str]] = {}

# Every joint of the Robotiq 2F-85, in model order, suffixed onto the robot's name.
# Only the two driver joints are actuated; the other six are the passive linkage that
# carries the grasp geometry, and all eight are needed to restore a pose. The TidyBot
# and the FR3 carry the same gripper under the same names.
ROBOTIQ_2F85_JOINT_SUFFIXES = (
    "right_driver_joint",
    "right_coupler_joint",
    "right_spring_link_joint",
    "right_follower_joint",
    "left_driver_joint",
    "left_coupler_joint",
    "left_spring_link_joint",
    "left_follower_joint",
)

# Feature names for those joints. `pos_gripper` is the commanded ctrl value in [0, 1],
# which does not determine where the fingers are: a gripper closed on an object stops
# short of one closed on air, and both report the same command.
GRIPPER_JOINT_POS_FEATURES = [
    f"pos_gripper_joint{i}" for i in range(1, len(ROBOTIQ_2F85_JOINT_SUFFIXES) + 1)
]
GRIPPER_JOINT_VEL_FEATURES = [
    f"vel_gripper_joint{i}" for i in range(1, len(ROBOTIQ_2F85_JOINT_SUFFIXES) + 1)
]

# Parent class for different object types, whether they are movable or not.
MujocoObjectType = Type("mujoco_object")
MujocoObjectTypeFeatures[MujocoObjectType] = [
    # Position.
    "x",
    "y",
    "z",
    # Orientation (quaternion).
    "qw",
    "qx",
    "qy",
    "qz",
]

MujocoMovableObjectType = Type("mujoco_movable_object", parent=MujocoObjectType)
MujocoObjectTypeFeatures[MujocoMovableObjectType] = [
    # Position.
    "x",
    "y",
    "z",
    # Orientation (quaternion).
    "qw",
    "qx",
    "qy",
    "qz",
    # Linear velocity.
    "vx",
    "vy",
    "vz",
    # Angular velocity.
    "wx",
    "wy",
    "wz",
    # Bounding box dimensions (full, not half).
    "bb_x",
    "bb_y",
    "bb_z",
]

MujocoFixtureObjectType = Type("mujoco_fixture", parent=MujocoObjectType)
MujocoObjectTypeFeatures[MujocoFixtureObjectType] = [
    # Position.
    "x",
    "y",
    "z",
    # Orientation (quaternion).
    "qw",
    "qx",
    "qy",
    "qz",
]

MujocoDrawerObjectType = Type("mujoco_drawer", parent=MujocoObjectType)
MujocoObjectTypeFeatures[MujocoDrawerObjectType] = [
    # Drawer slide position (0.0 to max_slide).
    "pos",
]

MujocoTidyBotRobotObjectType = Type("mujoco_tidybot_robot")
MujocoObjectTypeFeatures[MujocoTidyBotRobotObjectType] = [
    "pos_base_x",
    "pos_base_y",
    "pos_base_rot",
    "pos_arm_joint1",
    "pos_arm_joint2",
    "pos_arm_joint3",
    "pos_arm_joint4",
    "pos_arm_joint5",
    "pos_arm_joint6",
    "pos_arm_joint7",
    "pos_gripper",
    *GRIPPER_JOINT_POS_FEATURES,
    "vel_base_x",
    "vel_base_y",
    "vel_base_rot",
    "vel_arm_joint1",
    "vel_arm_joint2",
    "vel_arm_joint3",
    "vel_arm_joint4",
    "vel_arm_joint5",
    "vel_arm_joint6",
    "vel_arm_joint7",
    "vel_gripper",
    *GRIPPER_JOINT_VEL_FEATURES,
]

MujocoFR3RobotObjectType = Type("mujoco_fr3_robot")
MujocoObjectTypeFeatures[MujocoFR3RobotObjectType] = [
    # Fixed base mount pose (set once per episode, no velocities).
    "pos_base_x",
    "pos_base_y",
    "pos_base_rot",
    "pos_arm_joint1",
    "pos_arm_joint2",
    "pos_arm_joint3",
    "pos_arm_joint4",
    "pos_arm_joint5",
    "pos_arm_joint6",
    "pos_arm_joint7",
    "pos_gripper",
    *GRIPPER_JOINT_POS_FEATURES,
    "vel_arm_joint1",
    "vel_arm_joint2",
    "vel_arm_joint3",
    "vel_arm_joint4",
    "vel_arm_joint5",
    "vel_arm_joint6",
    "vel_arm_joint7",
    "vel_gripper",
    *GRIPPER_JOINT_VEL_FEATURES,
]

MujocoRBY1ARobotObjectType = Type("mujoco_rby1a_robot")
MujocoObjectTypeFeatures[MujocoRBY1ARobotObjectType] = [
    "pos_base_right",
    "pos_base_left",
    # TODO add more attributes  # pylint: disable=fixme
]
