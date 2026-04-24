# PrplLab3D

## Description
A 3D manipulation task set in the PRPL robotics lab. The robot opens lower cabinet
doors and places objects inside the open cabinets.

The robot has a holonomic mobile base with powered casters and a Kinova Gen3 arm.
The scene is loaded from the PRPL lab URDF, which includes a kitchen-style counter
with lower cabinets, a sink, stove, microwave, and other fixtures.

The robot can control:
- Base pose (x, y, theta)
- Arm position (x, y, z)
- Arm orientation (quaternion)
- Gripper position (open/close)

## Available Variants
The variants differ in the number of cubes to place.

- `kinder/PrplLab3D-o1-v0` — 1 cube
- `kinder/PrplLab3D-o2-v0` — 2 cubes

## Observation Space
*(Differs per variant)*

| Object | Features |
|---|---|
| `robot` | base pose (x, y, rot), arm joints (7), gripper state |
| `prpl_lab` | fixture pose |
| `cube0`, `cube1`, … | position, orientation |

## Action Space
Actions: base pos and yaw (3), arm joints (7), gripper pos (1)

## Rewards
No reward defined for this demo environment.

## References
PRPL Lab URDF from the Princeton Robot Planning and Learning group.
