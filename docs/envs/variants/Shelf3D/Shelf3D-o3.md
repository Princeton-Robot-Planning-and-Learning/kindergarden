# Shelf3D-o3

## Usage
```python
import kinder
kinder.register_all_environments()
env = kinder.make("kinder/Shelf3D-o3-v0")
```

## Description
Pick up three cubes from the floor and place them onto a space-constrained cupboard shelf.

## Initial State Distribution
![initial state GIF](../../assets/initial_state_gifs/variants/Shelf3D-o3.gif)

## Random Action Behavior
![random action GIF](../../assets/random_action_gifs/variants/Shelf3D-o3.gif)

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
| 48 | cupboard_1 | x |
| 49 | cupboard_1 | y |
| 50 | cupboard_1 | z |
| 51 | cupboard_1 | qw |
| 52 | cupboard_1 | qx |
| 53 | cupboard_1 | qy |
| 54 | cupboard_1 | qz |
| 55 | robot | pos_base_x |
| 56 | robot | pos_base_y |
| 57 | robot | pos_base_rot |
| 58 | robot | pos_arm_joint1 |
| 59 | robot | pos_arm_joint2 |
| 60 | robot | pos_arm_joint3 |
| 61 | robot | pos_arm_joint4 |
| 62 | robot | pos_arm_joint5 |
| 63 | robot | pos_arm_joint6 |
| 64 | robot | pos_arm_joint7 |
| 65 | robot | pos_gripper |
| 66 | robot | vel_base_x |
| 67 | robot | vel_base_y |
| 68 | robot | vel_base_rot |
| 69 | robot | vel_arm_joint1 |
| 70 | robot | vel_arm_joint2 |
| 71 | robot | vel_arm_joint3 |
| 72 | robot | vel_arm_joint4 |
| 73 | robot | vel_arm_joint5 |
| 74 | robot | vel_arm_joint6 |
| 75 | robot | vel_arm_joint7 |
| 76 | robot | vel_gripper |
