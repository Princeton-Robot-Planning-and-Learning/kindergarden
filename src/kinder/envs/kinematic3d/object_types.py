"""Object types that are common across different environments."""

from relational_structs import Type

Kinematic3DEnvTypeFeatures: dict[Type, list[str]] = {}

# The robot, which is a 7DOF arm, has joint positions and grasp features.
# Note that we must store the finger state if we want to have different grasps for
# different sized objects.
Kinematic3DRobotType = Type("Kinematic3DRobot")
Kinematic3DEnvTypeFeatures[Kinematic3DRobotType] = [
    "pos_base_x",
    "pos_base_y",
    "pos_base_rot",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
    "finger_state",
    "grasp_active",
    "grasp_tf_x",
    "grasp_tf_y",
    "grasp_tf_z",
    "grasp_tf_qx",
    "grasp_tf_qy",
    "grasp_tf_qz",
    "grasp_tf_qw",
]

# Cuboid objects have poses, grasp features, and half extents.
Kinematic3DCuboidType = Type("Kinematic3DCuboid")
Kinematic3DEnvTypeFeatures[Kinematic3DCuboidType] = [
    "pose_x",
    "pose_y",
    "pose_z",
    "pose_qx",
    "pose_qy",
    "pose_qz",
    "pose_qw",
    "grasp_active",
    "object_type",
    # encoded as an int or small float category just
    # like triangle_type to make things uniform
    "half_extent_x",
    "half_extent_y",
    "half_extent_z",
]

# Triangle objects: parameterize by triangle kind and side lengths (a,b,c)
# plus a thickness/depth along Z. Pose and grasp_active included.
Kinematic3DTriangleType = Type("Kinematic3DTriangle")
Kinematic3DEnvTypeFeatures[Kinematic3DTriangleType] = [
    "pose_x",
    "pose_y",
    "pose_z",
    "pose_qx",
    "pose_qy",
    "pose_qz",
    "pose_qw",
    "grasp_active",
    # Triangle specification: either equilateral/isosceles/scalene etc.
    # The consumer can interpret these fields; they are numeric features.
    "triangle_type",  # encoded as an int or small float category
    "side_a",
    "side_b",
    "depth",
]

# A point is just a position. For example, it could be a target point to reach.
Kinematic3DPointType = Type("Kinematic3DPoint")
Kinematic3DEnvTypeFeatures[Kinematic3DPointType] = [
    "x",
    "y",
    "z",
]


# Fixtures: static objects with a pose.
Kinematic3DFixtureType = Type("Kinematic3DFixture")
Kinematic3DEnvTypeFeatures[Kinematic3DFixtureType] = [
    "pose_x",
    "pose_y",
    "pose_z",
    "pose_qx",
    "pose_qy",
    "pose_qz",
    "pose_qw",
]


Limb3DEnvTypeFeatures: dict[Type, list[str]] = {}

# Torque control means the state needs joint velocities as well as positions.
Limb3DRobotType = Type("Limb3DRobot")
Limb3DEnvTypeFeatures[Limb3DRobotType] = (
    [
        "pos_base_x",
        "pos_base_y",
        "pos_base_rot",
    ]
    + [f"joint_{i}" for i in range(1, 8)]
    + [f"joint_vel_{i}" for i in range(1, 8)]
)

# A passive human limb (arm or leg) with 6 DOF.
Limb3DLimbType = Type("Limb3DLimb")
Limb3DEnvTypeFeatures[Limb3DLimbType] = (
    [f"joint_{i}" for i in range(1, 7)]
    + [f"joint_vel_{i}" for i in range(1, 7)]
    + [f"goal_joint_{i}" for i in range(1, 7)]
)

# Fixtures are static scene objects with a pose, e.g., the wheelchair or the bed.
Limb3DFixtureType = Type("Limb3DFixture")
Limb3DEnvTypeFeatures[Limb3DFixtureType] = [
    "pose_x",
    "pose_y",
    "pose_z",
    "pose_qx",
    "pose_qy",
    "pose_qz",
    "pose_qw",
]
