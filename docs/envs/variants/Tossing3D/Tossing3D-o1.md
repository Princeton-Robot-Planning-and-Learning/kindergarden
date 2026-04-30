# Tossing3D-o1

## Usage
```python
import kinder
kinder.register_all_environments()
env = kinder.make("kinder/Tossing3D-o1-v0")
```

## Description
Toss one cube into the bin from beyond the robot's reachable distance.

## Initial State Distribution
![initial state GIF](../../assets/initial_state_gifs/variants/Tossing3D-o1.gif)

## Random Action Behavior
![random action GIF](../../assets/random_action_gifs/variants/Tossing3D-o1.gif)

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
| 32 | cuboid_barrier | x |
| 33 | cuboid_barrier | y |
| 34 | cuboid_barrier | z |
| 35 | cuboid_barrier | qw |
| 36 | cuboid_barrier | qx |
| 37 | cuboid_barrier | qy |
| 38 | cuboid_barrier | qz |
| 39 | cuboid_barrier | vx |
| 40 | cuboid_barrier | vy |
| 41 | cuboid_barrier | vz |
| 42 | cuboid_barrier | wx |
| 43 | cuboid_barrier | wy |
| 44 | cuboid_barrier | wz |
| 45 | cuboid_barrier | bb_x |
| 46 | cuboid_barrier | bb_y |
| 47 | cuboid_barrier | bb_z |
| 48 | robot | pos_base_x |
| 49 | robot | pos_base_y |
| 50 | robot | pos_base_rot |
| 51 | robot | pos_arm_joint1 |
| 52 | robot | pos_arm_joint2 |
| 53 | robot | pos_arm_joint3 |
| 54 | robot | pos_arm_joint4 |
| 55 | robot | pos_arm_joint5 |
| 56 | robot | pos_arm_joint6 |
| 57 | robot | pos_arm_joint7 |
| 58 | robot | pos_gripper |
| 59 | robot | vel_base_x |
| 60 | robot | vel_base_y |
| 61 | robot | vel_base_rot |
| 62 | robot | vel_arm_joint1 |
| 63 | robot | vel_arm_joint2 |
| 64 | robot | vel_arm_joint3 |
| 65 | robot | vel_arm_joint4 |
| 66 | robot | vel_arm_joint5 |
| 67 | robot | vel_arm_joint6 |
| 68 | robot | vel_arm_joint7 |
| 69 | robot | vel_gripper |
