"""Define object types for the TidyBot environment."""

from relational_structs import Type

MujocoObjectTypeFeatures: dict[Type, list[str]] = {}

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
    # The commanded control in [0, 1], then where the fingers actually are.
    "pos_gripper",
    "pos_gripper_joint1",
    "pos_gripper_joint2",
    "pos_gripper_joint3",
    "pos_gripper_joint4",
    "pos_gripper_joint5",
    "pos_gripper_joint6",
    "pos_gripper_joint7",
    "pos_gripper_joint8",
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
    "vel_gripper_joint1",
    "vel_gripper_joint2",
    "vel_gripper_joint3",
    "vel_gripper_joint4",
    "vel_gripper_joint5",
    "vel_gripper_joint6",
    "vel_gripper_joint7",
    "vel_gripper_joint8",
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
    # The commanded control in [0, 1], then where the fingers actually are. The FR3
    # carries the same Robotiq 2F-85 as the TidyBot, read through the same helpers.
    "pos_gripper",
    "pos_gripper_joint1",
    "pos_gripper_joint2",
    "pos_gripper_joint3",
    "pos_gripper_joint4",
    "pos_gripper_joint5",
    "pos_gripper_joint6",
    "pos_gripper_joint7",
    "pos_gripper_joint8",
    "vel_arm_joint1",
    "vel_arm_joint2",
    "vel_arm_joint3",
    "vel_arm_joint4",
    "vel_arm_joint5",
    "vel_arm_joint6",
    "vel_arm_joint7",
    "vel_gripper",
    "vel_gripper_joint1",
    "vel_gripper_joint2",
    "vel_gripper_joint3",
    "vel_gripper_joint4",
    "vel_gripper_joint5",
    "vel_gripper_joint6",
    "vel_gripper_joint7",
    "vel_gripper_joint8",
]

MujocoRBY1ARobotObjectType = Type("mujoco_rby1a_robot")
MujocoObjectTypeFeatures[MujocoRBY1ARobotObjectType] = [
    "pos_base_right",
    "pos_base_left",
    # TODO add more attributes  # pylint: disable=fixme
]
