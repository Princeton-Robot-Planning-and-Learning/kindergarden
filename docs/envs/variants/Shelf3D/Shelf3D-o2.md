# Shelf3D-o2

## Usage
```python
import kinder
kinder.register_all_environments()
env = kinder.make("kinder/Shelf3D-o2-v0")
```

## Description
This variant uses the 'ground' scene type with 3 objects.

## Initial State Distribution
![initial state GIF](../../assets/initial_state_gifs/variants/Shelf3D-o2.gif)

## Random Action Behavior
![random action GIF](../../assets/random_action_gifs/variants/Shelf3D-o2.gif)

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
| 32 | cupboard_1 | x |
| 33 | cupboard_1 | y |
| 34 | cupboard_1 | z |
| 35 | cupboard_1 | qw |
| 36 | cupboard_1 | qx |
| 37 | cupboard_1 | qy |
| 38 | cupboard_1 | qz |
| 39 | robot | pos_base_x |
| 40 | robot | pos_base_y |
| 41 | robot | pos_base_rot |
| 42 | robot | pos_arm_joint1 |
| 43 | robot | pos_arm_joint2 |
| 44 | robot | pos_arm_joint3 |
| 45 | robot | pos_arm_joint4 |
| 46 | robot | pos_arm_joint5 |
| 47 | robot | pos_arm_joint6 |
| 48 | robot | pos_arm_joint7 |
| 49 | robot | pos_gripper |
| 50 | robot | vel_base_x |
| 51 | robot | vel_base_y |
| 52 | robot | vel_base_rot |
| 53 | robot | vel_arm_joint1 |
| 54 | robot | vel_arm_joint2 |
| 55 | robot | vel_arm_joint3 |
| 56 | robot | vel_arm_joint4 |
| 57 | robot | vel_arm_joint5 |
| 58 | robot | vel_arm_joint6 |
| 59 | robot | vel_arm_joint7 |
| 60 | robot | vel_gripper |
