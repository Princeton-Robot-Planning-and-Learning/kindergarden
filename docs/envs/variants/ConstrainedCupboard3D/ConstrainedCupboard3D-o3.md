# ConstrainedCupboard3D-o3

## Usage
```python
import kinder
kinder.register_all_environments()
env = kinder.make("kinder/ConstrainedCupboard3D-o3-v0")
```

## Description
Place three rods onto open shelves across an intermediate six-cupboard fixture.

## Initial State Distribution
![initial state GIF](../../assets/initial_state_gifs/variants/ConstrainedCupboard3D-o3.gif)

## Random Action Behavior
![random action GIF](../../assets/random_action_gifs/variants/ConstrainedCupboard3D-o3.gif)

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
| 32 | cuboid_2 | x |
| 33 | cuboid_2 | y |
| 34 | cuboid_2 | z |
| 35 | cuboid_2 | qw |
| 36 | cuboid_2 | qx |
| 37 | cuboid_2 | qy |
| 38 | cuboid_2 | qz |
| 39 | cuboid_2 | vx |
| 40 | cuboid_2 | vy |
| 41 | cuboid_2 | vz |
| 42 | cuboid_2 | wx |
| 43 | cuboid_2 | wy |
| 44 | cuboid_2 | wz |
| 45 | cuboid_2 | bb_x |
| 46 | cuboid_2 | bb_y |
| 47 | cuboid_2 | bb_z |
| 48 | cupboard_0 | x |
| 49 | cupboard_0 | y |
| 50 | cupboard_0 | z |
| 51 | cupboard_0 | qw |
| 52 | cupboard_0 | qx |
| 53 | cupboard_0 | qy |
| 54 | cupboard_0 | qz |
| 55 | cupboard_1 | x |
| 56 | cupboard_1 | y |
| 57 | cupboard_1 | z |
| 58 | cupboard_1 | qw |
| 59 | cupboard_1 | qx |
| 60 | cupboard_1 | qy |
| 61 | cupboard_1 | qz |
| 62 | cupboard_2 | x |
| 63 | cupboard_2 | y |
| 64 | cupboard_2 | z |
| 65 | cupboard_2 | qw |
| 66 | cupboard_2 | qx |
| 67 | cupboard_2 | qy |
| 68 | cupboard_2 | qz |
| 69 | cupboard_3 | x |
| 70 | cupboard_3 | y |
| 71 | cupboard_3 | z |
| 72 | cupboard_3 | qw |
| 73 | cupboard_3 | qx |
| 74 | cupboard_3 | qy |
| 75 | cupboard_3 | qz |
| 76 | cupboard_4 | x |
| 77 | cupboard_4 | y |
| 78 | cupboard_4 | z |
| 79 | cupboard_4 | qw |
| 80 | cupboard_4 | qx |
| 81 | cupboard_4 | qy |
| 82 | cupboard_4 | qz |
| 83 | cupboard_5 | x |
| 84 | cupboard_5 | y |
| 85 | cupboard_5 | z |
| 86 | cupboard_5 | qw |
| 87 | cupboard_5 | qx |
| 88 | cupboard_5 | qy |
| 89 | cupboard_5 | qz |
| 90 | robot | pos_base_x |
| 91 | robot | pos_base_y |
| 92 | robot | pos_base_rot |
| 93 | robot | pos_arm_joint1 |
| 94 | robot | pos_arm_joint2 |
| 95 | robot | pos_arm_joint3 |
| 96 | robot | pos_arm_joint4 |
| 97 | robot | pos_arm_joint5 |
| 98 | robot | pos_arm_joint6 |
| 99 | robot | pos_arm_joint7 |
| 100 | robot | pos_gripper |
| 101 | robot | vel_base_x |
| 102 | robot | vel_base_y |
| 103 | robot | vel_base_rot |
| 104 | robot | vel_arm_joint1 |
| 105 | robot | vel_arm_joint2 |
| 106 | robot | vel_arm_joint3 |
| 107 | robot | vel_arm_joint4 |
| 108 | robot | vel_arm_joint5 |
| 109 | robot | vel_arm_joint6 |
| 110 | robot | vel_arm_joint7 |
| 111 | robot | vel_gripper |
