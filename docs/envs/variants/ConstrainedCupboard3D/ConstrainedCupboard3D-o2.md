# ConstrainedCupboard3D-o2

## Usage
```python
import kinder
kinder.register_all_environments()
env = kinder.make("kinder/ConstrainedCupboard3D-o2-v0")
```

## Description
Place two rods onto open shelves across three small cupboards with varying shelf heights.

## Initial State Distribution
![initial state GIF](../../assets/initial_state_gifs/variants/ConstrainedCupboard3D-o2.gif)

## Random Action Behavior
![random action GIF](../../assets/random_action_gifs/variants/ConstrainedCupboard3D-o2.gif)

**Random Action Stats**: Total Reward: -25.00, Success: No, Steps: 25

## Example Demonstration
*(No demonstration GIFs available)*

## Observation Space
The entries of an array in this Box space correspond to the following object features:
| **Index** | **Object** | **Feature** |
| --- | --- | --- |
| 0 | cuboid_0 | x |
| 1 | cuboid_0 | y |
| 2 | cuboid_0 | z |
| 3 | cuboid_0 | qw |
| 4 | cuboid_0 | qx |
| 5 | cuboid_0 | qy |
| 6 | cuboid_0 | qz |
| 7 | cuboid_0 | vx |
| 8 | cuboid_0 | vy |
| 9 | cuboid_0 | vz |
| 10 | cuboid_0 | wx |
| 11 | cuboid_0 | wy |
| 12 | cuboid_0 | wz |
| 13 | cuboid_0 | bb_x |
| 14 | cuboid_0 | bb_y |
| 15 | cuboid_0 | bb_z |
| 16 | cuboid_1 | x |
| 17 | cuboid_1 | y |
| 18 | cuboid_1 | z |
| 19 | cuboid_1 | qw |
| 20 | cuboid_1 | qx |
| 21 | cuboid_1 | qy |
| 22 | cuboid_1 | qz |
| 23 | cuboid_1 | vx |
| 24 | cuboid_1 | vy |
| 25 | cuboid_1 | vz |
| 26 | cuboid_1 | wx |
| 27 | cuboid_1 | wy |
| 28 | cuboid_1 | wz |
| 29 | cuboid_1 | bb_x |
| 30 | cuboid_1 | bb_y |
| 31 | cuboid_1 | bb_z |
| 32 | cupboard_0 | x |
| 33 | cupboard_0 | y |
| 34 | cupboard_0 | z |
| 35 | cupboard_0 | qw |
| 36 | cupboard_0 | qx |
| 37 | cupboard_0 | qy |
| 38 | cupboard_0 | qz |
| 39 | cupboard_1 | x |
| 40 | cupboard_1 | y |
| 41 | cupboard_1 | z |
| 42 | cupboard_1 | qw |
| 43 | cupboard_1 | qx |
| 44 | cupboard_1 | qy |
| 45 | cupboard_1 | qz |
| 46 | cupboard_2 | x |
| 47 | cupboard_2 | y |
| 48 | cupboard_2 | z |
| 49 | cupboard_2 | qw |
| 50 | cupboard_2 | qx |
| 51 | cupboard_2 | qy |
| 52 | cupboard_2 | qz |
| 53 | robot | pos_base_x |
| 54 | robot | pos_base_y |
| 55 | robot | pos_base_rot |
| 56 | robot | pos_arm_joint1 |
| 57 | robot | pos_arm_joint2 |
| 58 | robot | pos_arm_joint3 |
| 59 | robot | pos_arm_joint4 |
| 60 | robot | pos_arm_joint5 |
| 61 | robot | pos_arm_joint6 |
| 62 | robot | pos_arm_joint7 |
| 63 | robot | pos_gripper |
| 64 | robot | vel_base_x |
| 65 | robot | vel_base_y |
| 66 | robot | vel_base_rot |
| 67 | robot | vel_arm_joint1 |
| 68 | robot | vel_arm_joint2 |
| 69 | robot | vel_arm_joint3 |
| 70 | robot | vel_arm_joint4 |
| 71 | robot | vel_arm_joint5 |
| 72 | robot | vel_arm_joint6 |
| 73 | robot | vel_arm_joint7 |
| 74 | robot | vel_gripper |
