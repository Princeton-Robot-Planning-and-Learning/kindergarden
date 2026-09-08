# Shelf3D-o6

## Usage
```python
import kinder
kinder.register_all_environments()
env = kinder.make("kinder/Shelf3D-o6-v0")
```

## Description
Pick up six cubes from the floor and place them onto a space-constrained cupboard shelf.

## Initial State Distribution
![initial state GIF](../../assets/initial_state_gifs/variants/Shelf3D-o6.gif)

## Random Action Behavior
![random action GIF](../../assets/random_action_gifs/variants/Shelf3D-o6.gif)

**Random Action Stats**: Total Reward: -25.00, Success: No, Steps: 25

## Example Demonstration
*(No demonstration GIFs available)*

## Observation Space
The entries of an array in this Box space correspond to the following object features:
| **Index** | **Object** | **Feature** |
| --- | --- | --- |
| 0 | cube1 | x |
| 1 | cube1 | y |
| 2 | cube1 | z |
| 3 | cube1 | qw |
| 4 | cube1 | qx |
| 5 | cube1 | qy |
| 6 | cube1 | qz |
| 7 | cube1 | vx |
| 8 | cube1 | vy |
| 9 | cube1 | vz |
| 10 | cube1 | wx |
| 11 | cube1 | wy |
| 12 | cube1 | wz |
| 13 | cube1 | bb_x |
| 14 | cube1 | bb_y |
| 15 | cube1 | bb_z |
| 16 | cube2 | x |
| 17 | cube2 | y |
| 18 | cube2 | z |
| 19 | cube2 | qw |
| 20 | cube2 | qx |
| 21 | cube2 | qy |
| 22 | cube2 | qz |
| 23 | cube2 | vx |
| 24 | cube2 | vy |
| 25 | cube2 | vz |
| 26 | cube2 | wx |
| 27 | cube2 | wy |
| 28 | cube2 | wz |
| 29 | cube2 | bb_x |
| 30 | cube2 | bb_y |
| 31 | cube2 | bb_z |
| 32 | cube3 | x |
| 33 | cube3 | y |
| 34 | cube3 | z |
| 35 | cube3 | qw |
| 36 | cube3 | qx |
| 37 | cube3 | qy |
| 38 | cube3 | qz |
| 39 | cube3 | vx |
| 40 | cube3 | vy |
| 41 | cube3 | vz |
| 42 | cube3 | wx |
| 43 | cube3 | wy |
| 44 | cube3 | wz |
| 45 | cube3 | bb_x |
| 46 | cube3 | bb_y |
| 47 | cube3 | bb_z |
| 48 | cube4 | x |
| 49 | cube4 | y |
| 50 | cube4 | z |
| 51 | cube4 | qw |
| 52 | cube4 | qx |
| 53 | cube4 | qy |
| 54 | cube4 | qz |
| 55 | cube4 | vx |
| 56 | cube4 | vy |
| 57 | cube4 | vz |
| 58 | cube4 | wx |
| 59 | cube4 | wy |
| 60 | cube4 | wz |
| 61 | cube4 | bb_x |
| 62 | cube4 | bb_y |
| 63 | cube4 | bb_z |
| 64 | cube5 | x |
| 65 | cube5 | y |
| 66 | cube5 | z |
| 67 | cube5 | qw |
| 68 | cube5 | qx |
| 69 | cube5 | qy |
| 70 | cube5 | qz |
| 71 | cube5 | vx |
| 72 | cube5 | vy |
| 73 | cube5 | vz |
| 74 | cube5 | wx |
| 75 | cube5 | wy |
| 76 | cube5 | wz |
| 77 | cube5 | bb_x |
| 78 | cube5 | bb_y |
| 79 | cube5 | bb_z |
| 80 | cube6 | x |
| 81 | cube6 | y |
| 82 | cube6 | z |
| 83 | cube6 | qw |
| 84 | cube6 | qx |
| 85 | cube6 | qy |
| 86 | cube6 | qz |
| 87 | cube6 | vx |
| 88 | cube6 | vy |
| 89 | cube6 | vz |
| 90 | cube6 | wx |
| 91 | cube6 | wy |
| 92 | cube6 | wz |
| 93 | cube6 | bb_x |
| 94 | cube6 | bb_y |
| 95 | cube6 | bb_z |
| 96 | cupboard_1 | x |
| 97 | cupboard_1 | y |
| 98 | cupboard_1 | z |
| 99 | cupboard_1 | qw |
| 100 | cupboard_1 | qx |
| 101 | cupboard_1 | qy |
| 102 | cupboard_1 | qz |
| 103 | robot | pos_base_x |
| 104 | robot | pos_base_y |
| 105 | robot | pos_base_rot |
| 106 | robot | pos_arm_joint1 |
| 107 | robot | pos_arm_joint2 |
| 108 | robot | pos_arm_joint3 |
| 109 | robot | pos_arm_joint4 |
| 110 | robot | pos_arm_joint5 |
| 111 | robot | pos_arm_joint6 |
| 112 | robot | pos_arm_joint7 |
| 113 | robot | pos_gripper |
| 114 | robot | vel_base_x |
| 115 | robot | vel_base_y |
| 116 | robot | vel_base_rot |
| 117 | robot | vel_arm_joint1 |
| 118 | robot | vel_arm_joint2 |
| 119 | robot | vel_arm_joint3 |
| 120 | robot | vel_arm_joint4 |
| 121 | robot | vel_arm_joint5 |
| 122 | robot | vel_arm_joint6 |
| 123 | robot | vel_arm_joint7 |
| 124 | robot | vel_gripper |
