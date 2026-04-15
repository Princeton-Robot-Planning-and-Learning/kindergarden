# ConstrainedCupboard3D-o1

## Usage
```python
import kinder
env = kinder.make("kinder/ConstrainedCupboard3D-o1-v0")
```

## Description
This variant uses the 'ground' scene type with 3 objects.

## Initial State Distribution
![initial state GIF](../../assets/initial_state_gifs/variants/ConstrainedCupboard3D-o1.gif)

## Random Action Behavior
![random action GIF](../../assets/random_action_gifs/variants/ConstrainedCupboard3D-o1.gif)

**Random Action Stats**: Total Reward: -0.25, Success: No, Steps: 25

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
| 16 | cupboard_0 | x |
| 17 | cupboard_0 | y |
| 18 | cupboard_0 | z |
| 19 | cupboard_0 | qw |
| 20 | cupboard_0 | qx |
| 21 | cupboard_0 | qy |
| 22 | cupboard_0 | qz |
| 23 | cupboard_1 | x |
| 24 | cupboard_1 | y |
| 25 | cupboard_1 | z |
| 26 | cupboard_1 | qw |
| 27 | cupboard_1 | qx |
| 28 | cupboard_1 | qy |
| 29 | cupboard_1 | qz |
| 30 | cupboard_2 | x |
| 31 | cupboard_2 | y |
| 32 | cupboard_2 | z |
| 33 | cupboard_2 | qw |
| 34 | cupboard_2 | qx |
| 35 | cupboard_2 | qy |
| 36 | cupboard_2 | qz |
| 37 | robot | pos_base_x |
| 38 | robot | pos_base_y |
| 39 | robot | pos_base_rot |
| 40 | robot | pos_arm_joint1 |
| 41 | robot | pos_arm_joint2 |
| 42 | robot | pos_arm_joint3 |
| 43 | robot | pos_arm_joint4 |
| 44 | robot | pos_arm_joint5 |
| 45 | robot | pos_arm_joint6 |
| 46 | robot | pos_arm_joint7 |
| 47 | robot | pos_gripper |
| 48 | robot | vel_base_x |
| 49 | robot | vel_base_y |
| 50 | robot | vel_base_rot |
| 51 | robot | vel_arm_joint1 |
| 52 | robot | vel_arm_joint2 |
| 53 | robot | vel_arm_joint3 |
| 54 | robot | vel_arm_joint4 |
| 55 | robot | vel_arm_joint5 |
| 56 | robot | vel_arm_joint6 |
| 57 | robot | vel_arm_joint7 |
| 58 | robot | vel_gripper |
