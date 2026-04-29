# Dynamo3D-o1

## Usage
```python
import kinder
kinder.register_all_environments()
env = kinder.make("kinder/Dynamo3D-o1-v0")
```

## Description
This variant uses the 'ground' scene type with 3 objects.

## Initial State Distribution
![initial state GIF](../../assets/initial_state_gifs/variants/Dynamo3D-o1.gif)

## Random Action Behavior
![random action GIF](../../assets/random_action_gifs/variants/Dynamo3D-o1.gif)

**Random Action Stats**: Total Reward: -25.00, Success: No, Steps: 25

## Example Demonstration
*(No demonstration GIFs available)*

## Observation Space
The entries of an array in this Box space correspond to the following object features:
| **Index** | **Object** | **Feature** |
| --- | --- | --- |
| 0 | obstacle_chair | x |
| 1 | obstacle_chair | y |
| 2 | obstacle_chair | z |
| 3 | obstacle_chair | qw |
| 4 | obstacle_chair | qx |
| 5 | obstacle_chair | qy |
| 6 | obstacle_chair | qz |
| 7 | obstacle_chair | vx |
| 8 | obstacle_chair | vy |
| 9 | obstacle_chair | vz |
| 10 | obstacle_chair | wx |
| 11 | obstacle_chair | wy |
| 12 | obstacle_chair | wz |
| 13 | obstacle_chair | bb_x |
| 14 | obstacle_chair | bb_y |
| 15 | obstacle_chair | bb_z |
| 16 | robot | pos_base_x |
| 17 | robot | pos_base_y |
| 18 | robot | pos_base_rot |
| 19 | robot | pos_arm_joint1 |
| 20 | robot | pos_arm_joint2 |
| 21 | robot | pos_arm_joint3 |
| 22 | robot | pos_arm_joint4 |
| 23 | robot | pos_arm_joint5 |
| 24 | robot | pos_arm_joint6 |
| 25 | robot | pos_arm_joint7 |
| 26 | robot | pos_gripper |
| 27 | robot | vel_base_x |
| 28 | robot | vel_base_y |
| 29 | robot | vel_base_rot |
| 30 | robot | vel_arm_joint1 |
| 31 | robot | vel_arm_joint2 |
| 32 | robot | vel_arm_joint3 |
| 33 | robot | vel_arm_joint4 |
| 34 | robot | vel_arm_joint5 |
| 35 | robot | vel_arm_joint6 |
| 36 | robot | vel_arm_joint7 |
| 37 | robot | vel_gripper |
