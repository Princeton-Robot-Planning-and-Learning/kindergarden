# DynObstruction2D-o1

## Usage
```python
import kinder
kinder.register_all_environments()
env = kinder.make("kinder/DynObstruction2D-o1-v0")
```

## Description
This variant has 1 obstruction.

## Initial State Distribution
![initial state GIF](../../assets/initial_state_gifs/variants/DynObstruction2D-o1.gif)

## Random Action Behavior
![random action GIF](../../assets/random_action_gifs/variants/DynObstruction2D-o1.gif)

**Random Action Stats**: Total Reward: -25.00, Success: No, Steps: 25

## Example Demonstration
*(No demonstration GIFs available)*

## Observation Space
The entries of an array in this Box space correspond to the following object features:
| **Index** | **Object** | **Feature** |
| --- | --- | --- |
| 0 | target_surface | x |
| 1 | target_surface | y |
| 2 | target_surface | theta |
| 3 | target_surface | vx |
| 4 | target_surface | vy |
| 5 | target_surface | omega |
| 6 | target_surface | static |
| 7 | target_surface | held |
| 8 | target_surface | color_r |
| 9 | target_surface | color_g |
| 10 | target_surface | color_b |
| 11 | target_surface | z_order |
| 12 | target_surface | width |
| 13 | target_surface | height |
| 14 | target_block | x |
| 15 | target_block | y |
| 16 | target_block | theta |
| 17 | target_block | vx |
| 18 | target_block | vy |
| 19 | target_block | omega |
| 20 | target_block | static |
| 21 | target_block | held |
| 22 | target_block | color_r |
| 23 | target_block | color_g |
| 24 | target_block | color_b |
| 25 | target_block | z_order |
| 26 | target_block | width |
| 27 | target_block | height |
| 28 | target_block | mass |
| 29 | obstruction0 | x |
| 30 | obstruction0 | y |
| 31 | obstruction0 | theta |
| 32 | obstruction0 | vx |
| 33 | obstruction0 | vy |
| 34 | obstruction0 | omega |
| 35 | obstruction0 | static |
| 36 | obstruction0 | held |
| 37 | obstruction0 | color_r |
| 38 | obstruction0 | color_g |
| 39 | obstruction0 | color_b |
| 40 | obstruction0 | z_order |
| 41 | obstruction0 | width |
| 42 | obstruction0 | height |
| 43 | obstruction0 | mass |
| 44 | robot | x |
| 45 | robot | y |
| 46 | robot | theta |
| 47 | robot | vx_base |
| 48 | robot | vy_base |
| 49 | robot | omega_base |
| 50 | robot | vx_arm |
| 51 | robot | vy_arm |
| 52 | robot | omega_arm |
| 53 | robot | vx_gripper_l |
| 54 | robot | vy_gripper_l |
| 55 | robot | omega_gripper_l |
| 56 | robot | vx_gripper_r |
| 57 | robot | vy_gripper_r |
| 58 | robot | omega_gripper_r |
| 59 | robot | static |
| 60 | robot | base_radius |
| 61 | robot | arm_joint |
| 62 | robot | arm_length |
| 63 | robot | gripper_base_width |
| 64 | robot | gripper_base_height |
| 65 | robot | finger_gap |
| 66 | robot | finger_height |
| 67 | robot | finger_width |
