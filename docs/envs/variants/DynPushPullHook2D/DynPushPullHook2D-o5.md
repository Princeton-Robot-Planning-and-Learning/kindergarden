# DynPushPullHook2D-o5

## Usage
```python
import kinder
kinder.register_all_environments()
env = kinder.make("kinder/DynPushPullHook2D-o5-v0")
```

## Description
This variant has 5 obstructions.

## Initial State Distribution
![initial state GIF](../../assets/initial_state_gifs/variants/DynPushPullHook2D-o5.gif)

## Random Action Behavior
![random action GIF](../../assets/random_action_gifs/variants/DynPushPullHook2D-o5.gif)

**Random Action Stats**: Total Reward: -25.00, Success: No, Steps: 25

## Example Demonstration
![demo GIF](../../assets/demo_gifs/DynPushPullHook2D-o5/DynPushPullHook2D-o5_seed2_1768427291.gif)

**Demo Stats**: Total Reward: -194.00, Success: Yes, Steps: 194

## Observation Space
The entries of an array in this Box space correspond to the following object features:
| **Index** | **Object** | **Feature** |
| --- | --- | --- |
| 0 | robot | x |
| 1 | robot | y |
| 2 | robot | theta |
| 3 | robot | vx_base |
| 4 | robot | vy_base |
| 5 | robot | omega_base |
| 6 | robot | vx_arm |
| 7 | robot | vy_arm |
| 8 | robot | omega_arm |
| 9 | robot | vx_gripper_l |
| 10 | robot | vy_gripper_l |
| 11 | robot | omega_gripper_l |
| 12 | robot | vx_gripper_r |
| 13 | robot | vy_gripper_r |
| 14 | robot | omega_gripper_r |
| 15 | robot | static |
| 16 | robot | base_radius |
| 17 | robot | arm_joint |
| 18 | robot | arm_length |
| 19 | robot | gripper_base_width |
| 20 | robot | gripper_base_height |
| 21 | robot | finger_gap |
| 22 | robot | finger_height |
| 23 | robot | finger_width |
| 24 | hook | x |
| 25 | hook | y |
| 26 | hook | theta |
| 27 | hook | vx |
| 28 | hook | vy |
| 29 | hook | omega |
| 30 | hook | static |
| 31 | hook | held |
| 32 | hook | color_r |
| 33 | hook | color_g |
| 34 | hook | color_b |
| 35 | hook | z_order |
| 36 | hook | width |
| 37 | hook | length_side1 |
| 38 | hook | length_side2 |
| 39 | hook | mass |
| 40 | target_block | x |
| 41 | target_block | y |
| 42 | target_block | theta |
| 43 | target_block | vx |
| 44 | target_block | vy |
| 45 | target_block | omega |
| 46 | target_block | static |
| 47 | target_block | held |
| 48 | target_block | color_r |
| 49 | target_block | color_g |
| 50 | target_block | color_b |
| 51 | target_block | z_order |
| 52 | target_block | width |
| 53 | target_block | height |
| 54 | target_block | mass |
| 55 | obstruction0 | x |
| 56 | obstruction0 | y |
| 57 | obstruction0 | theta |
| 58 | obstruction0 | vx |
| 59 | obstruction0 | vy |
| 60 | obstruction0 | omega |
| 61 | obstruction0 | static |
| 62 | obstruction0 | held |
| 63 | obstruction0 | color_r |
| 64 | obstruction0 | color_g |
| 65 | obstruction0 | color_b |
| 66 | obstruction0 | z_order |
| 67 | obstruction0 | width |
| 68 | obstruction0 | height |
| 69 | obstruction0 | mass |
| 70 | obstruction1 | x |
| 71 | obstruction1 | y |
| 72 | obstruction1 | theta |
| 73 | obstruction1 | vx |
| 74 | obstruction1 | vy |
| 75 | obstruction1 | omega |
| 76 | obstruction1 | static |
| 77 | obstruction1 | held |
| 78 | obstruction1 | color_r |
| 79 | obstruction1 | color_g |
| 80 | obstruction1 | color_b |
| 81 | obstruction1 | z_order |
| 82 | obstruction1 | width |
| 83 | obstruction1 | height |
| 84 | obstruction1 | mass |
| 85 | obstruction2 | x |
| 86 | obstruction2 | y |
| 87 | obstruction2 | theta |
| 88 | obstruction2 | vx |
| 89 | obstruction2 | vy |
| 90 | obstruction2 | omega |
| 91 | obstruction2 | static |
| 92 | obstruction2 | held |
| 93 | obstruction2 | color_r |
| 94 | obstruction2 | color_g |
| 95 | obstruction2 | color_b |
| 96 | obstruction2 | z_order |
| 97 | obstruction2 | width |
| 98 | obstruction2 | height |
| 99 | obstruction2 | mass |
| 100 | obstruction3 | x |
| 101 | obstruction3 | y |
| 102 | obstruction3 | theta |
| 103 | obstruction3 | vx |
| 104 | obstruction3 | vy |
| 105 | obstruction3 | omega |
| 106 | obstruction3 | static |
| 107 | obstruction3 | held |
| 108 | obstruction3 | color_r |
| 109 | obstruction3 | color_g |
| 110 | obstruction3 | color_b |
| 111 | obstruction3 | z_order |
| 112 | obstruction3 | width |
| 113 | obstruction3 | height |
| 114 | obstruction3 | mass |
| 115 | obstruction4 | x |
| 116 | obstruction4 | y |
| 117 | obstruction4 | theta |
| 118 | obstruction4 | vx |
| 119 | obstruction4 | vy |
| 120 | obstruction4 | omega |
| 121 | obstruction4 | static |
| 122 | obstruction4 | held |
| 123 | obstruction4 | color_r |
| 124 | obstruction4 | color_g |
| 125 | obstruction4 | color_b |
| 126 | obstruction4 | z_order |
| 127 | obstruction4 | width |
| 128 | obstruction4 | height |
| 129 | obstruction4 | mass |
