# Motion2D-p0

## Usage
```python
import kinder
kinder.register_all_environments()
env = kinder.make("kinder/Motion2D-p0-v0")
```

## Description
This variant has no narrow passages (open space).

## Initial State Distribution
![initial state GIF](../../assets/initial_state_gifs/variants/Motion2D-p0.gif)

## Random Action Behavior
![random action GIF](../../assets/random_action_gifs/variants/Motion2D-p0.gif)

**Random Action Stats**: Total Reward: -25.00, Success: No, Steps: 25

## Example Demonstration
![demo GIF](../../assets/demo_gifs/Motion2D-p0/Motion2D-p0.gif)

**Demo Stats**: Total Reward: -43.00, Success: Yes, Steps: 43

## Observation Space
The entries of an array in this Box space correspond to the following object features:
| **Index** | **Object** | **Feature** |
| --- | --- | --- |
| 0 | robot | x |
| 1 | robot | y |
| 2 | robot | theta |
| 3 | robot | base_radius |
| 4 | robot | arm_joint |
| 5 | robot | arm_length |
| 6 | robot | vacuum |
| 7 | robot | gripper_height |
| 8 | robot | gripper_width |
| 9 | target_region | x |
| 10 | target_region | y |
| 11 | target_region | theta |
| 12 | target_region | static |
| 13 | target_region | color_r |
| 14 | target_region | color_g |
| 15 | target_region | color_b |
| 16 | target_region | z_order |
| 17 | target_region | width |
| 18 | target_region | height |
