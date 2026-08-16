"""Object types for the limb repositioning environments."""

from relational_structs import Type

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
