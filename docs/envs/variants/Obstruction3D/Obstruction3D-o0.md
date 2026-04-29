# Obstruction3D-o0

## Usage
```python
import kinder
kinder.register_all_environments()
env = kinder.make("kinder/Obstruction3D-o0-v0")
```

## Description
This variant has no obstructions.

## Initial State Distribution
![initial state GIF](../../assets/initial_state_gifs/variants/Obstruction3D-o0.gif)

## Random Action Behavior
![random action GIF](../../assets/random_action_gifs/variants/Obstruction3D-o0.gif)

**Random Action Stats**: Total Reward: -25.00, Success: No, Steps: 25

## Example Demonstration
![demo GIF](../../assets/demo_gifs/Obstruction3D-o0/Obstruction3D-o0_1770245184.gif)

**Demo Stats**: Total Reward: -46.00, Success: Yes, Steps: 61

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
