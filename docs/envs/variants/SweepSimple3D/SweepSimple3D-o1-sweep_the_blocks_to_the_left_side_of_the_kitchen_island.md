# SweepSimple3D-o1-sweep_the_blocks_to_the_left_side_of_the_kitchen_island

## Usage
```python
import kinder
kinder.register_all_environments()
env = kinder.make("kinder/SweepSimple3D-o1-sweep_the_blocks_to_the_left_side_of_the_kitchen_island-v0")
```

## Description
This variant uses the 'ground' scene type with 3 objects.

## Initial State Distribution
![initial state GIF](../../assets/initial_state_gifs/variants/SweepSimple3D-o1-sweep_the_blocks_to_the_left_side_of_the_kitchen_island.gif)

## Random Action Behavior
![random action GIF](../../assets/random_action_gifs/variants/SweepSimple3D-o1-sweep_the_blocks_to_the_left_side_of_the_kitchen_island.gif)

**Random Action Stats**: Total Reward: -25.00, Success: No, Steps: 25

## Example Demonstration
*(No demonstration GIFs available)*

## Observation Space
The entries of an array in this Box space correspond to the following object features:
| **Index** | **Object** | **Feature** |
| --- | --- | --- |
| 0 | cube_0 | x |
| 1 | cube_0 | y |
| 2 | cube_0 | z |
| 3 | cube_0 | qw |
| 4 | cube_0 | qx |
| 5 | cube_0 | qy |
| 6 | cube_0 | qz |
| 7 | cube_0 | vx |
| 8 | cube_0 | vy |
| 9 | cube_0 | vz |
| 10 | cube_0 | wx |
| 11 | cube_0 | wy |
| 12 | cube_0 | wz |
| 13 | cube_0 | bb_x |
| 14 | cube_0 | bb_y |
| 15 | cube_0 | bb_z |
| 16 | kitchen_cooking_area | x |
| 17 | kitchen_cooking_area | y |
| 18 | kitchen_cooking_area | z |
| 19 | kitchen_cooking_area | qw |
| 20 | kitchen_cooking_area | qx |
| 21 | kitchen_cooking_area | qy |
| 22 | kitchen_cooking_area | qz |
| 23 | kitchen_cooking_area_drawer_s0c1 | pos |
| 24 | kitchen_cooking_area_drawer_s1c1 | pos |
| 25 | kitchen_cooking_area_upper | x |
| 26 | kitchen_cooking_area_upper | y |
| 27 | kitchen_cooking_area_upper | z |
| 28 | kitchen_cooking_area_upper | qw |
| 29 | kitchen_cooking_area_upper | qx |
| 30 | kitchen_cooking_area_upper | qy |
| 31 | kitchen_cooking_area_upper | qz |
| 32 | kitchen_island | x |
| 33 | kitchen_island | y |
| 34 | kitchen_island | z |
| 35 | kitchen_island | qw |
| 36 | kitchen_island | qx |
| 37 | kitchen_island | qy |
| 38 | kitchen_island | qz |
| 39 | kitchen_island_drawer_s0c0 | pos |
| 40 | kitchen_island_drawer_s0c1 | pos |
| 41 | kitchen_island_drawer_s0c2 | pos |
| 42 | kitchen_island_drawer_s1c0 | pos |
| 43 | kitchen_island_drawer_s1c1 | pos |
| 44 | kitchen_island_drawer_s1c2 | pos |
| 45 | kitchen_left_corner | x |
| 46 | kitchen_left_corner | y |
| 47 | kitchen_left_corner | z |
| 48 | kitchen_left_corner | qw |
| 49 | kitchen_left_corner | qx |
| 50 | kitchen_left_corner | qy |
| 51 | kitchen_left_corner | qz |
| 52 | kitchen_left_side | x |
| 53 | kitchen_left_side | y |
| 54 | kitchen_left_side | z |
| 55 | kitchen_left_side | qw |
| 56 | kitchen_left_side | qx |
| 57 | kitchen_left_side | qy |
| 58 | kitchen_left_side | qz |
| 59 | kitchen_left_side_drawer_s0c1 | pos |
| 60 | kitchen_left_side_drawer_s1c1 | pos |
| 61 | robot | pos_base_x |
| 62 | robot | pos_base_y |
| 63 | robot | pos_base_rot |
| 64 | robot | pos_arm_joint1 |
| 65 | robot | pos_arm_joint2 |
| 66 | robot | pos_arm_joint3 |
| 67 | robot | pos_arm_joint4 |
| 68 | robot | pos_arm_joint5 |
| 69 | robot | pos_arm_joint6 |
| 70 | robot | pos_arm_joint7 |
| 71 | robot | pos_gripper |
| 72 | robot | vel_base_x |
| 73 | robot | vel_base_y |
| 74 | robot | vel_base_rot |
| 75 | robot | vel_arm_joint1 |
| 76 | robot | vel_arm_joint2 |
| 77 | robot | vel_arm_joint3 |
| 78 | robot | vel_arm_joint4 |
| 79 | robot | vel_arm_joint5 |
| 80 | robot | vel_arm_joint6 |
| 81 | robot | vel_arm_joint7 |
| 82 | robot | vel_gripper |
| 83 | wiper_0 | x |
| 84 | wiper_0 | y |
| 85 | wiper_0 | z |
| 86 | wiper_0 | qw |
| 87 | wiper_0 | qx |
| 88 | wiper_0 | qy |
| 89 | wiper_0 | qz |
| 90 | wiper_0 | vx |
| 91 | wiper_0 | vy |
| 92 | wiper_0 | vz |
| 93 | wiper_0 | wx |
| 94 | wiper_0 | wy |
| 95 | wiper_0 | wz |
| 96 | wiper_0 | bb_x |
| 97 | wiper_0 | bb_y |
| 98 | wiper_0 | bb_z |
