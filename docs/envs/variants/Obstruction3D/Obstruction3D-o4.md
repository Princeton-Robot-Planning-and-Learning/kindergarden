# Obstruction3D-o4

## Usage
```python
import kinder
kinder.register_all_environments()
env = kinder.make("kinder/Obstruction3D-o4-v0")
```

## Description
This variant has 4 obstructions to clear.

## Initial State Distribution
![initial state GIF](../../assets/initial_state_gifs/variants/Obstruction3D-o4.gif)

## Random Action Behavior
![random action GIF](../../assets/random_action_gifs/variants/Obstruction3D-o4.gif)

**Random Action Stats**: Total Reward: -25.00, Success: No, Steps: 25

## Example Demonstration
*(No demonstration GIFs available)*

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
| 19 | target_region | pose_x |
| 20 | target_region | pose_y |
| 21 | target_region | pose_z |
| 22 | target_region | pose_qx |
| 23 | target_region | pose_qy |
| 24 | target_region | pose_qz |
| 25 | target_region | pose_qw |
| 26 | target_region | grasp_active |
| 27 | target_region | object_type |
| 28 | target_region | half_extent_x |
| 29 | target_region | half_extent_y |
| 30 | target_region | half_extent_z |
| 31 | target_block | pose_x |
| 32 | target_block | pose_y |
| 33 | target_block | pose_z |
| 34 | target_block | pose_qx |
| 35 | target_block | pose_qy |
| 36 | target_block | pose_qz |
| 37 | target_block | pose_qw |
| 38 | target_block | grasp_active |
| 39 | target_block | object_type |
| 40 | target_block | half_extent_x |
| 41 | target_block | half_extent_y |
| 42 | target_block | half_extent_z |
| 43 | obstruction0 | pose_x |
| 44 | obstruction0 | pose_y |
| 45 | obstruction0 | pose_z |
| 46 | obstruction0 | pose_qx |
| 47 | obstruction0 | pose_qy |
| 48 | obstruction0 | pose_qz |
| 49 | obstruction0 | pose_qw |
| 50 | obstruction0 | grasp_active |
| 51 | obstruction0 | object_type |
| 52 | obstruction0 | half_extent_x |
| 53 | obstruction0 | half_extent_y |
| 54 | obstruction0 | half_extent_z |
| 55 | obstruction1 | pose_x |
| 56 | obstruction1 | pose_y |
| 57 | obstruction1 | pose_z |
| 58 | obstruction1 | pose_qx |
| 59 | obstruction1 | pose_qy |
| 60 | obstruction1 | pose_qz |
| 61 | obstruction1 | pose_qw |
| 62 | obstruction1 | grasp_active |
| 63 | obstruction1 | object_type |
| 64 | obstruction1 | half_extent_x |
| 65 | obstruction1 | half_extent_y |
| 66 | obstruction1 | half_extent_z |
| 67 | obstruction2 | pose_x |
| 68 | obstruction2 | pose_y |
| 69 | obstruction2 | pose_z |
| 70 | obstruction2 | pose_qx |
| 71 | obstruction2 | pose_qy |
| 72 | obstruction2 | pose_qz |
| 73 | obstruction2 | pose_qw |
| 74 | obstruction2 | grasp_active |
| 75 | obstruction2 | object_type |
| 76 | obstruction2 | half_extent_x |
| 77 | obstruction2 | half_extent_y |
| 78 | obstruction2 | half_extent_z |
| 79 | obstruction3 | pose_x |
| 80 | obstruction3 | pose_y |
| 81 | obstruction3 | pose_z |
| 82 | obstruction3 | pose_qx |
| 83 | obstruction3 | pose_qy |
| 84 | obstruction3 | pose_qz |
| 85 | obstruction3 | pose_qw |
| 86 | obstruction3 | grasp_active |
| 87 | obstruction3 | object_type |
| 88 | obstruction3 | half_extent_x |
| 89 | obstruction3 | half_extent_y |
| 90 | obstruction3 | half_extent_z |
