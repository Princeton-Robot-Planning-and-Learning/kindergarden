# PrplLab3D-o1

## Usage
```python
import kinder
kinder.register_all_environments()
env = kinder.make("kinder/PrplLab3D-o1-v0")
```

## Description
This variant has 1 cube to place on the counter.

## Initial State Distribution
![initial state GIF](../../assets/initial_state_gifs/variants/PrplLab3D-o1.gif)

## Random Action Behavior
![random action GIF](../../assets/random_action_gifs/variants/PrplLab3D-o1.gif)

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
| 19 | prpl_lab | pose_x |
| 20 | prpl_lab | pose_y |
| 21 | prpl_lab | pose_z |
| 22 | prpl_lab | pose_qx |
| 23 | prpl_lab | pose_qy |
| 24 | prpl_lab | pose_qz |
| 25 | prpl_lab | pose_qw |
| 26 | cube0 | pose_x |
| 27 | cube0 | pose_y |
| 28 | cube0 | pose_z |
| 29 | cube0 | pose_qx |
| 30 | cube0 | pose_qy |
| 31 | cube0 | pose_qz |
| 32 | cube0 | pose_qw |
| 33 | cube0 | grasp_active |
| 34 | cube0 | object_type |
| 35 | cube0 | half_extent_x |
| 36 | cube0 | half_extent_y |
| 37 | cube0 | half_extent_z |
