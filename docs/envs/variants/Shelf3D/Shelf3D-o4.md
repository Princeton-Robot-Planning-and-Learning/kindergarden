# Shelf3D-o4

## Usage
```python
import kinder
kinder.register_all_environments()
env = kinder.make("kinder/Shelf3D-o4-v0")
```

## Description
Pick up four cubes from the floor and place them onto a space-constrained cupboard shelf.

## Initial State Distribution
![initial state GIF](../../assets/initial_state_gifs/variants/Shelf3D-o4.gif)

## Random Action Behavior
![random action GIF](../../assets/random_action_gifs/variants/Shelf3D-o4.gif)

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
| 64 | cupboard_1 | x |
| 65 | cupboard_1 | y |
| 66 | cupboard_1 | z |
| 67 | cupboard_1 | qw |
| 68 | cupboard_1 | qx |
| 69 | cupboard_1 | qy |
| 70 | cupboard_1 | qz |
| 71 | robot | pos_base_x |
| 72 | robot | pos_base_y |
| 73 | robot | pos_base_rot |
| 74 | robot | pos_arm_joint1 |
| 75 | robot | pos_arm_joint2 |
| 76 | robot | pos_arm_joint3 |
| 77 | robot | pos_arm_joint4 |
| 78 | robot | pos_arm_joint5 |
| 79 | robot | pos_arm_joint6 |
| 80 | robot | pos_arm_joint7 |
| 81 | robot | pos_gripper |
| 82 | robot | vel_base_x |
| 83 | robot | vel_base_y |
| 84 | robot | vel_base_rot |
| 85 | robot | vel_arm_joint1 |
| 86 | robot | vel_arm_joint2 |
| 87 | robot | vel_arm_joint3 |
| 88 | robot | vel_arm_joint4 |
| 89 | robot | vel_arm_joint5 |
| 90 | robot | vel_arm_joint6 |
| 91 | robot | vel_arm_joint7 |
| 92 | robot | vel_gripper |
