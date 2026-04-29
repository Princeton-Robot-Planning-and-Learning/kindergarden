# ClutteredRetrieval2D-o1

## Usage
```python
import kinder
kinder.register_all_environments()
env = kinder.make("kinder/ClutteredRetrieval2D-o1-v0")
```

## Description
This variant has 1 obstruction.

## Initial State Distribution
![initial state GIF](../../assets/initial_state_gifs/variants/ClutteredRetrieval2D-o1.gif)

## Random Action Behavior
![random action GIF](../../assets/random_action_gifs/variants/ClutteredRetrieval2D-o1.gif)

**Random Action Stats**: Total Reward: -25.00, Success: No, Steps: 25

## Example Demonstration
*(No demonstration GIFs available)*

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
| 9 | target_block | x |
| 10 | target_block | y |
| 11 | target_block | theta |
| 12 | target_block | static |
| 13 | target_block | color_r |
| 14 | target_block | color_g |
| 15 | target_block | color_b |
| 16 | target_block | z_order |
| 17 | target_block | width |
| 18 | target_block | height |
| 19 | target_region | x |
| 20 | target_region | y |
| 21 | target_region | theta |
| 22 | target_region | static |
| 23 | target_region | color_r |
| 24 | target_region | color_g |
| 25 | target_region | color_b |
| 26 | target_region | z_order |
| 27 | target_region | width |
| 28 | target_region | height |
| 29 | obstruction0 | x |
| 30 | obstruction0 | y |
| 31 | obstruction0 | theta |
| 32 | obstruction0 | static |
| 33 | obstruction0 | color_r |
| 34 | obstruction0 | color_g |
| 35 | obstruction0 | color_b |
| 36 | obstruction0 | z_order |
| 37 | obstruction0 | width |
| 38 | obstruction0 | height |
