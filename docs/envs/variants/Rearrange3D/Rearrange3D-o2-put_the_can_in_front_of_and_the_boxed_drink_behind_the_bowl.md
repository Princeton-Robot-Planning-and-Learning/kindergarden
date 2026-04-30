# Rearrange3D-o2-put_the_can_in_front_of_and_the_boxed_drink_behind_the_bowl

## Usage
```python
import kinder
kinder.register_all_environments()
env = kinder.make("kinder/Rearrange3D-o2-put_the_can_in_front_of_and_the_boxed_drink_behind_the_bowl-v0")
```

## Description
Place the can in front of the bowl and the boxed drink behind it.

## Initial State Distribution
![initial state GIF](../../assets/initial_state_gifs/variants/Rearrange3D-o2-put_the_can_in_front_of_and_the_boxed_drink_behind_the_bowl.gif)

## Random Action Behavior
![random action GIF](../../assets/random_action_gifs/variants/Rearrange3D-o2-put_the_can_in_front_of_and_the_boxed_drink_behind_the_bowl.gif)

**Random Action Stats**: Total Reward: -25.00, Success: No, Steps: 25

## Example Demonstration
*(No demonstration GIFs available)*

## Observation Space
The entries of an array in this Box space correspond to the following object features:
| **Index** | **Object** | **Feature** |
| --- | --- | --- |
| 0 | bowl_0 | x |
| 1 | bowl_0 | y |
| 2 | bowl_0 | z |
| 3 | bowl_0 | qw |
| 4 | bowl_0 | qx |
| 5 | bowl_0 | qy |
| 6 | bowl_0 | qz |
| 7 | bowl_0 | vx |
| 8 | bowl_0 | vy |
| 9 | bowl_0 | vz |
| 10 | bowl_0 | wx |
| 11 | bowl_0 | wy |
| 12 | bowl_0 | wz |
| 13 | bowl_0 | bb_x |
| 14 | bowl_0 | bb_y |
| 15 | bowl_0 | bb_z |
| 16 | boxed_drink_0 | x |
| 17 | boxed_drink_0 | y |
| 18 | boxed_drink_0 | z |
| 19 | boxed_drink_0 | qw |
| 20 | boxed_drink_0 | qx |
| 21 | boxed_drink_0 | qy |
| 22 | boxed_drink_0 | qz |
| 23 | boxed_drink_0 | vx |
| 24 | boxed_drink_0 | vy |
| 25 | boxed_drink_0 | vz |
| 26 | boxed_drink_0 | wx |
| 27 | boxed_drink_0 | wy |
| 28 | boxed_drink_0 | wz |
| 29 | boxed_drink_0 | bb_x |
| 30 | boxed_drink_0 | bb_y |
| 31 | boxed_drink_0 | bb_z |
| 32 | can_0 | x |
| 33 | can_0 | y |
| 34 | can_0 | z |
| 35 | can_0 | qw |
| 36 | can_0 | qx |
| 37 | can_0 | qy |
| 38 | can_0 | qz |
| 39 | can_0 | vx |
| 40 | can_0 | vy |
| 41 | can_0 | vz |
| 42 | can_0 | wx |
| 43 | can_0 | wy |
| 44 | can_0 | wz |
| 45 | can_0 | bb_x |
| 46 | can_0 | bb_y |
| 47 | can_0 | bb_z |
| 48 | kitchen_cooking_area | x |
| 49 | kitchen_cooking_area | y |
| 50 | kitchen_cooking_area | z |
| 51 | kitchen_cooking_area | qw |
| 52 | kitchen_cooking_area | qx |
| 53 | kitchen_cooking_area | qy |
| 54 | kitchen_cooking_area | qz |
| 55 | kitchen_cooking_area_drawer_s0c1 | pos |
| 56 | kitchen_cooking_area_drawer_s1c1 | pos |
| 57 | kitchen_cooking_area_upper | x |
| 58 | kitchen_cooking_area_upper | y |
| 59 | kitchen_cooking_area_upper | z |
| 60 | kitchen_cooking_area_upper | qw |
| 61 | kitchen_cooking_area_upper | qx |
| 62 | kitchen_cooking_area_upper | qy |
| 63 | kitchen_cooking_area_upper | qz |
| 64 | kitchen_island | x |
| 65 | kitchen_island | y |
| 66 | kitchen_island | z |
| 67 | kitchen_island | qw |
| 68 | kitchen_island | qx |
| 69 | kitchen_island | qy |
| 70 | kitchen_island | qz |
| 71 | kitchen_island_drawer_s0c0 | pos |
| 72 | kitchen_island_drawer_s0c1 | pos |
| 73 | kitchen_island_drawer_s0c2 | pos |
| 74 | kitchen_island_drawer_s1c0 | pos |
| 75 | kitchen_island_drawer_s1c1 | pos |
| 76 | kitchen_island_drawer_s1c2 | pos |
| 77 | kitchen_left_corner | x |
| 78 | kitchen_left_corner | y |
| 79 | kitchen_left_corner | z |
| 80 | kitchen_left_corner | qw |
| 81 | kitchen_left_corner | qx |
| 82 | kitchen_left_corner | qy |
| 83 | kitchen_left_corner | qz |
| 84 | kitchen_left_side | x |
| 85 | kitchen_left_side | y |
| 86 | kitchen_left_side | z |
| 87 | kitchen_left_side | qw |
| 88 | kitchen_left_side | qx |
| 89 | kitchen_left_side | qy |
| 90 | kitchen_left_side | qz |
| 91 | kitchen_left_side_drawer_s0c1 | pos |
| 92 | kitchen_left_side_drawer_s1c1 | pos |
| 93 | robot | pos_base_x |
| 94 | robot | pos_base_y |
| 95 | robot | pos_base_rot |
| 96 | robot | pos_arm_joint1 |
| 97 | robot | pos_arm_joint2 |
| 98 | robot | pos_arm_joint3 |
| 99 | robot | pos_arm_joint4 |
| 100 | robot | pos_arm_joint5 |
| 101 | robot | pos_arm_joint6 |
| 102 | robot | pos_arm_joint7 |
| 103 | robot | pos_gripper |
| 104 | robot | vel_base_x |
| 105 | robot | vel_base_y |
| 106 | robot | vel_base_rot |
| 107 | robot | vel_arm_joint1 |
| 108 | robot | vel_arm_joint2 |
| 109 | robot | vel_arm_joint3 |
| 110 | robot | vel_arm_joint4 |
| 111 | robot | vel_arm_joint5 |
| 112 | robot | vel_arm_joint6 |
| 113 | robot | vel_arm_joint7 |
| 114 | robot | vel_gripper |
