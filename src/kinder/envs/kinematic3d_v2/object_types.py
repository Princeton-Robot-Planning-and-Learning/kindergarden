"""Object types that are common across Kinematic3Dv2 environments."""

from relational_structs import Type

# Every arm modeled by these environments has this many actuated joints. Robots with
# more joints (a lift, a torso, a second arm) hold the others at their home values, so
# that observations and actions stay fixed-dimensional across robots.
ARM_NUM_JOINTS = 7

Kinematic3Dv2EnvTypeFeatures: dict[Type, list[str]] = {}

# An arm robot is described by its actuated joint positions alone. There are no base or
# grasp features: these environments move the arm and nothing else.
Kinematic3Dv2ArmRobotType = Type("Kinematic3Dv2ArmRobot")
Kinematic3Dv2EnvTypeFeatures[Kinematic3Dv2ArmRobotType] = [
    f"joint_{i}" for i in range(1, ARM_NUM_JOINTS + 1)
]

# An arm that can also hold an object: its joint positions plus a binary "grasping"
# feature that is 1.0 while the arm is holding something.
Kinematic3Dv2GraspArmRobotType = Type("Kinematic3Dv2GraspArmRobot")
Kinematic3Dv2EnvTypeFeatures[Kinematic3Dv2GraspArmRobotType] = [
    f"joint_{i}" for i in range(1, ARM_NUM_JOINTS + 1)
] + ["grasping"]

# A point is just a position. For example, it could be a target point to reach.
Kinematic3Dv2PointType = Type("Kinematic3Dv2Point")
Kinematic3Dv2EnvTypeFeatures[Kinematic3Dv2PointType] = [
    "x",
    "y",
    "z",
]
