# ConstrainedCupboard3D-o4

## Usage
```python
import kinder
kinder.register_all_environments()
env = kinder.make("kinder/ConstrainedCupboard3D-o4-v0")
```

## Description
Place four rods onto open shelves across an intermediate six-cupboard fixture.

## Initial State Distribution
![initial state GIF](../../assets/initial_state_gifs/variants/ConstrainedCupboard3D-o4.gif)

## Random Action Behavior
![random action GIF](../../assets/random_action_gifs/variants/ConstrainedCupboard3D-o4.gif)

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
| 48 | cuboid_3 | x |
| 49 | cuboid_3 | y |
| 50 | cuboid_3 | z |
| 51 | cuboid_3 | qw |
| 52 | cuboid_3 | qx |
| 53 | cuboid_3 | qy |
| 54 | cuboid_3 | qz |
| 55 | cuboid_3 | vx |
| 56 | cuboid_3 | vy |
| 57 | cuboid_3 | vz |
| 58 | cuboid_3 | wx |
| 59 | cuboid_3 | wy |
| 60 | cuboid_3 | wz |
| 61 | cuboid_3 | bb_x |
| 62 | cuboid_3 | bb_y |
| 63 | cuboid_3 | bb_z |
| 64 | cupboard_0 | x |
| 65 | cupboard_0 | y |
| 66 | cupboard_0 | z |
| 67 | cupboard_0 | qw |
| 68 | cupboard_0 | qx |
| 69 | cupboard_0 | qy |
| 70 | cupboard_0 | qz |
| 71 | cupboard_1 | x |
| 72 | cupboard_1 | y |
| 73 | cupboard_1 | z |
| 74 | cupboard_1 | qw |
| 75 | cupboard_1 | qx |
| 76 | cupboard_1 | qy |
| 77 | cupboard_1 | qz |
| 78 | cupboard_2 | x |
| 79 | cupboard_2 | y |
| 80 | cupboard_2 | z |
| 81 | cupboard_2 | qw |
| 82 | cupboard_2 | qx |
| 83 | cupboard_2 | qy |
| 84 | cupboard_2 | qz |
| 85 | cupboard_3 | x |
| 86 | cupboard_3 | y |
| 87 | cupboard_3 | z |
| 88 | cupboard_3 | qw |
| 89 | cupboard_3 | qx |
| 90 | cupboard_3 | qy |
| 91 | cupboard_3 | qz |
| 92 | cupboard_4 | x |
| 93 | cupboard_4 | y |
| 94 | cupboard_4 | z |
| 95 | cupboard_4 | qw |
| 96 | cupboard_4 | qx |
| 97 | cupboard_4 | qy |
| 98 | cupboard_4 | qz |
| 99 | cupboard_5 | x |
| 100 | cupboard_5 | y |
| 101 | cupboard_5 | z |
| 102 | cupboard_5 | qw |
| 103 | cupboard_5 | qx |
| 104 | cupboard_5 | qy |
| 105 | cupboard_5 | qz |
| 106 | robot | pos_base_x |
| 107 | robot | pos_base_y |
| 108 | robot | pos_base_rot |
| 109 | robot | pos_arm_joint1 |
| 110 | robot | pos_arm_joint2 |
| 111 | robot | pos_arm_joint3 |
| 112 | robot | pos_arm_joint4 |
| 113 | robot | pos_arm_joint5 |
| 114 | robot | pos_arm_joint6 |
| 115 | robot | pos_arm_joint7 |
| 116 | robot | pos_gripper |
| 117 | robot | vel_base_x |
| 118 | robot | vel_base_y |
| 119 | robot | vel_base_rot |
| 120 | robot | vel_arm_joint1 |
| 121 | robot | vel_arm_joint2 |
| 122 | robot | vel_arm_joint3 |
| 123 | robot | vel_arm_joint4 |
| 124 | robot | vel_arm_joint5 |
| 125 | robot | vel_arm_joint6 |
| 126 | robot | vel_arm_joint7 |
| 127 | robot | vel_gripper |
