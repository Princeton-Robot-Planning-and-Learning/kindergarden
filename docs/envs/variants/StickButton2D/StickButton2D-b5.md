# StickButton2D-b5

## Usage
```python
import kinder
kinder.register_all_environments()
env = kinder.make("kinder/StickButton2D-b5-v0")
```

## Description
This variant has 5 buttons to press.

## Initial State Distribution
![initial state GIF](../../assets/initial_state_gifs/variants/StickButton2D-b5.gif)

## Random Action Behavior
![random action GIF](../../assets/random_action_gifs/variants/StickButton2D-b5.gif)

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
| 9 | stick | x |
| 10 | stick | y |
| 11 | stick | theta |
| 12 | stick | static |
| 13 | stick | color_r |
| 14 | stick | color_g |
| 15 | stick | color_b |
| 16 | stick | z_order |
| 17 | stick | width |
| 18 | stick | height |
| 19 | button0 | x |
| 20 | button0 | y |
| 21 | button0 | theta |
| 22 | button0 | static |
| 23 | button0 | color_r |
| 24 | button0 | color_g |
| 25 | button0 | color_b |
| 26 | button0 | z_order |
| 27 | button0 | radius |
| 28 | button1 | x |
| 29 | button1 | y |
| 30 | button1 | theta |
| 31 | button1 | static |
| 32 | button1 | color_r |
| 33 | button1 | color_g |
| 34 | button1 | color_b |
| 35 | button1 | z_order |
| 36 | button1 | radius |
| 37 | button2 | x |
| 38 | button2 | y |
| 39 | button2 | theta |
| 40 | button2 | static |
| 41 | button2 | color_r |
| 42 | button2 | color_g |
| 43 | button2 | color_b |
| 44 | button2 | z_order |
| 45 | button2 | radius |
| 46 | button3 | x |
| 47 | button3 | y |
| 48 | button3 | theta |
| 49 | button3 | static |
| 50 | button3 | color_r |
| 51 | button3 | color_g |
| 52 | button3 | color_b |
| 53 | button3 | z_order |
| 54 | button3 | radius |
| 55 | button4 | x |
| 56 | button4 | y |
| 57 | button4 | theta |
| 58 | button4 | static |
| 59 | button4 | color_r |
| 60 | button4 | color_g |
| 61 | button4 | color_b |
| 62 | button4 | z_order |
| 63 | button4 | radius |
