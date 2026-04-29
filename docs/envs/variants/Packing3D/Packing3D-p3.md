# Packing3D-p3

## Usage
```python
import kinder
kinder.register_all_environments()
env = kinder.make("kinder/Packing3D-p3-v0")
```

## Description
This variant has 3 parts to pack into the rack.

## Initial State Distribution
![initial state GIF](../../assets/initial_state_gifs/variants/Packing3D-p3.gif)

## Random Action Behavior
![random action GIF](../../assets/random_action_gifs/variants/Packing3D-p3.gif)

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
| 19 | rack | pose_x |
| 20 | rack | pose_y |
| 21 | rack | pose_z |
| 22 | rack | pose_qx |
| 23 | rack | pose_qy |
| 24 | rack | pose_qz |
| 25 | rack | pose_qw |
| 26 | rack | grasp_active |
| 27 | rack | object_type |
| 28 | rack | half_extent_x |
| 29 | rack | half_extent_y |
| 30 | rack | half_extent_z |
| 31 | part0 | pose_x |
| 32 | part0 | pose_y |
| 33 | part0 | pose_z |
| 34 | part0 | pose_qx |
| 35 | part0 | pose_qy |
| 36 | part0 | pose_qz |
| 37 | part0 | pose_qw |
| 38 | part0 | grasp_active |
| 39 | part0 | object_type |
| 40 | part0 | half_extent_x |
| 41 | part0 | half_extent_y |
| 42 | part0 | half_extent_z |
| 43 | part1 | pose_x |
| 44 | part1 | pose_y |
| 45 | part1 | pose_z |
| 46 | part1 | pose_qx |
| 47 | part1 | pose_qy |
| 48 | part1 | pose_qz |
| 49 | part1 | pose_qw |
| 50 | part1 | grasp_active |
| 51 | part1 | triangle_type |
| 52 | part1 | side_a |
| 53 | part1 | side_b |
| 54 | part1 | depth |
| 55 | part2 | pose_x |
| 56 | part2 | pose_y |
| 57 | part2 | pose_z |
| 58 | part2 | pose_qx |
| 59 | part2 | pose_qy |
| 60 | part2 | pose_qz |
| 61 | part2 | pose_qw |
| 62 | part2 | grasp_active |
| 63 | part2 | triangle_type |
| 64 | part2 | side_a |
| 65 | part2 | side_b |
| 66 | part2 | depth |
