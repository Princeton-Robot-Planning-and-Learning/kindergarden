# Tossing3D-o2

## Usage
```python
import kinder
kinder.register_all_environments()
env = kinder.make("kinder/Tossing3D-o2-v0")
```

## Description
This variant uses the 'ground' scene type with 3 objects.

## Initial State Distribution
![initial state GIF](../../assets/initial_state_gifs/variants/Tossing3D-o2.gif)

## Random Action Behavior
![random action GIF](../../assets/random_action_gifs/variants/Tossing3D-o2.gif)

**Random Action Stats**: Total Reward: -25.00, Success: No, Steps: 25

## Example Demonstration
*(No demonstration GIFs available)*

## Observation Space
The entries of an array in this Box space correspond to the following object features:
| **Index** | **Object** | **Feature** |
| --- | --- | --- |
| 0 | bin_0 | x |
| 1 | bin_0 | y |
| 2 | bin_0 | z |
| 3 | bin_0 | qw |
| 4 | bin_0 | qx |
| 5 | bin_0 | qy |
| 6 | bin_0 | qz |
| 7 | bin_0 | vx |
| 8 | bin_0 | vy |
| 9 | bin_0 | vz |
| 10 | bin_0 | wx |
| 11 | bin_0 | wy |
| 12 | bin_0 | wz |
| 13 | bin_0 | bb_x |
| 14 | bin_0 | bb_y |
| 15 | bin_0 | bb_z |
| 16 | cube_0 | x |
| 17 | cube_0 | y |
| 18 | cube_0 | z |
| 19 | cube_0 | qw |
| 20 | cube_0 | qx |
| 21 | cube_0 | qy |
| 22 | cube_0 | qz |
| 23 | cube_0 | vx |
| 24 | cube_0 | vy |
| 25 | cube_0 | vz |
| 26 | cube_0 | wx |
| 27 | cube_0 | wy |
| 28 | cube_0 | wz |
| 29 | cube_0 | bb_x |
| 30 | cube_0 | bb_y |
| 31 | cube_0 | bb_z |
| 32 | cube_1 | x |
| 33 | cube_1 | y |
| 34 | cube_1 | z |
| 35 | cube_1 | qw |
| 36 | cube_1 | qx |
| 37 | cube_1 | qy |
| 38 | cube_1 | qz |
| 39 | cube_1 | vx |
| 40 | cube_1 | vy |
| 41 | cube_1 | vz |
| 42 | cube_1 | wx |
| 43 | cube_1 | wy |
| 44 | cube_1 | wz |
| 45 | cube_1 | bb_x |
| 46 | cube_1 | bb_y |
| 47 | cube_1 | bb_z |
| 48 | cuboid_barrier | x |
| 49 | cuboid_barrier | y |
| 50 | cuboid_barrier | z |
| 51 | cuboid_barrier | qw |
| 52 | cuboid_barrier | qx |
| 53 | cuboid_barrier | qy |
| 54 | cuboid_barrier | qz |
| 55 | cuboid_barrier | vx |
| 56 | cuboid_barrier | vy |
| 57 | cuboid_barrier | vz |
| 58 | cuboid_barrier | wx |
| 59 | cuboid_barrier | wy |
| 60 | cuboid_barrier | wz |
| 61 | cuboid_barrier | bb_x |
| 62 | cuboid_barrier | bb_y |
| 63 | cuboid_barrier | bb_z |
| 64 | robot | pos_base_x |
| 65 | robot | pos_base_y |
| 66 | robot | pos_base_rot |
| 67 | robot | pos_arm_joint1 |
| 68 | robot | pos_arm_joint2 |
| 69 | robot | pos_arm_joint3 |
| 70 | robot | pos_arm_joint4 |
| 71 | robot | pos_arm_joint5 |
| 72 | robot | pos_arm_joint6 |
| 73 | robot | pos_arm_joint7 |
| 74 | robot | pos_gripper |
| 75 | robot | vel_base_x |
| 76 | robot | vel_base_y |
| 77 | robot | vel_base_rot |
| 78 | robot | vel_arm_joint1 |
| 79 | robot | vel_arm_joint2 |
| 80 | robot | vel_arm_joint3 |
| 81 | robot | vel_arm_joint4 |
| 82 | robot | vel_arm_joint5 |
| 83 | robot | vel_arm_joint6 |
| 84 | robot | vel_arm_joint7 |
| 85 | robot | vel_gripper |
