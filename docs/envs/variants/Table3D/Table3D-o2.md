# Table3D-o2

## Usage
```python
import kinder
kinder.register_all_environments()
env = kinder.make("kinder/Table3D-o2-v0")
```

## Description
This variant has 2 cubes on the table.

## Initial State Distribution
![initial state GIF](../../assets/initial_state_gifs/variants/Table3D-o2.gif)

## Random Action Behavior
![random action GIF](../../assets/random_action_gifs/variants/Table3D-o2.gif)

**Random Action Stats**: Total Reward: -25.00, Success: No, Steps: 25

## Example Demonstration
![demo GIF](../../assets/demo_gifs/Table3D-o2/Table3D-o2_1770158668.gif)

**Demo Stats**: Total Reward: -37.00, Success: No, Steps: 37

## Observation Space
The entries of an array in this Box space correspond to the following object features:
| **Index** | **Object** | **Feature** |
| --- | --- | --- |
| 0 | robot | pos_base_x |
| 1 | robot | pos_base_y |
| 2 | robot | pos_base_rot |
| 3 | robot | joint_1 |
| 4 | robot | joint_2 |
| 5 | robot | joint_3 |
| 6 | robot | joint_4 |
| 7 | robot | joint_5 |
| 8 | robot | joint_6 |
| 9 | robot | joint_7 |
| 10 | robot | finger_state |
| 11 | robot | grasp_active |
| 12 | robot | grasp_tf_x |
| 13 | robot | grasp_tf_y |
| 14 | robot | grasp_tf_z |
| 15 | robot | grasp_tf_qx |
| 16 | robot | grasp_tf_qy |
| 17 | robot | grasp_tf_qz |
| 18 | robot | grasp_tf_qw |
| 19 | table | pose_x |
| 20 | table | pose_y |
| 21 | table | pose_z |
| 22 | table | pose_qx |
| 23 | table | pose_qy |
| 24 | table | pose_qz |
| 25 | table | pose_qw |
| 26 | table | grasp_active |
| 27 | table | object_type |
| 28 | table | half_extent_x |
| 29 | table | half_extent_y |
| 30 | table | half_extent_z |
| 31 | cube0 | pose_x |
| 32 | cube0 | pose_y |
| 33 | cube0 | pose_z |
| 34 | cube0 | pose_qx |
| 35 | cube0 | pose_qy |
| 36 | cube0 | pose_qz |
| 37 | cube0 | pose_qw |
| 38 | cube0 | grasp_active |
| 39 | cube0 | object_type |
| 40 | cube0 | half_extent_x |
| 41 | cube0 | half_extent_y |
| 42 | cube0 | half_extent_z |
| 43 | cube1 | pose_x |
| 44 | cube1 | pose_y |
| 45 | cube1 | pose_z |
| 46 | cube1 | pose_qx |
| 47 | cube1 | pose_qy |
| 48 | cube1 | pose_qz |
| 49 | cube1 | pose_qw |
| 50 | cube1 | grasp_active |
| 51 | cube1 | object_type |
| 52 | cube1 | half_extent_x |
| 53 | cube1 | half_extent_y |
| 54 | cube1 | half_extent_z |
