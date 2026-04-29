# DynObstruction2D-o2

## Usage
```python
import kinder
kinder.register_all_environments()
env = kinder.make("kinder/DynObstruction2D-o2-v0")
```

## Description
This variant has 2 obstructions.

## Initial State Distribution
![initial state GIF](../../assets/initial_state_gifs/variants/DynObstruction2D-o2.gif)

## Random Action Behavior
![random action GIF](../../assets/random_action_gifs/variants/DynObstruction2D-o2.gif)

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
| 44 | obstruction1 | x |
| 45 | obstruction1 | y |
| 46 | obstruction1 | theta |
| 47 | obstruction1 | vx |
| 48 | obstruction1 | vy |
| 49 | obstruction1 | omega |
| 50 | obstruction1 | static |
| 51 | obstruction1 | held |
| 52 | obstruction1 | color_r |
| 53 | obstruction1 | color_g |
| 54 | obstruction1 | color_b |
| 55 | obstruction1 | z_order |
| 56 | obstruction1 | width |
| 57 | obstruction1 | height |
| 58 | obstruction1 | mass |
| 59 | robot | x |
| 60 | robot | y |
| 61 | robot | theta |
| 62 | robot | vx_base |
| 63 | robot | vy_base |
| 64 | robot | omega_base |
| 65 | robot | vx_arm |
| 66 | robot | vy_arm |
| 67 | robot | omega_arm |
| 68 | robot | vx_gripper_l |
| 69 | robot | vy_gripper_l |
| 70 | robot | omega_gripper_l |
| 71 | robot | vx_gripper_r |
| 72 | robot | vy_gripper_r |
| 73 | robot | omega_gripper_r |
| 74 | robot | static |
| 75 | robot | base_radius |
| 76 | robot | arm_joint |
| 77 | robot | arm_length |
| 78 | robot | gripper_base_width |
| 79 | robot | gripper_base_height |
| 80 | robot | finger_gap |
| 81 | robot | finger_height |
| 82 | robot | finger_width |
