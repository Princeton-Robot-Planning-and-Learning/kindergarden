# VegaPickPlace3D

## Usage
```python
import kinder
kinder.register_all_environments()
env = kinder.make("kinder/VegaPickPlace3D-v0")
```

## Description
No variant-specific description available.

## Initial State Distribution
![initial state GIF](../../assets/initial_state_gifs/variants/VegaPickPlace3D.gif)

## Random Action Behavior
![random action GIF](../../assets/random_action_gifs/variants/VegaPickPlace3D.gif)

**Random Action Stats**: Total Reward: -25.00, Success: No, Steps: 25

## Example Demonstration
*(No demonstration GIFs available)*

## Observation Space
The entries of an array in this Box space correspond to the following object features:
| **Index** | **Object** | **Feature** |
| --- | --- | --- |
| 0 | left_arm | joint_1 |
| 1 | left_arm | joint_2 |
| 2 | left_arm | joint_3 |
| 3 | left_arm | joint_4 |
| 4 | left_arm | joint_5 |
| 5 | left_arm | joint_6 |
| 6 | left_arm | joint_7 |
| 7 | left_arm | grasping |
| 8 | right_arm | joint_1 |
| 9 | right_arm | joint_2 |
| 10 | right_arm | joint_3 |
| 11 | right_arm | joint_4 |
| 12 | right_arm | joint_5 |
| 13 | right_arm | joint_6 |
| 14 | right_arm | joint_7 |
| 15 | right_arm | grasping |
| 16 | cube | x |
| 17 | cube | y |
| 18 | cube | z |
| 19 | target | x |
| 20 | target | y |
| 21 | target | z |
