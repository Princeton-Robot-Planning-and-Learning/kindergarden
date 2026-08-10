# VegaMotion3D

## Usage
```python
import kinder
kinder.register_all_environments()
env = kinder.make("kinder/VegaMotion3D-v0")
```

## Description
No variant-specific description available.

## Initial State Distribution
![initial state GIF](../../assets/initial_state_gifs/variants/VegaMotion3D.gif)

## Random Action Behavior
![random action GIF](../../assets/random_action_gifs/variants/VegaMotion3D.gif)

**Random Action Stats**: Total Reward: -25.00, Success: No, Steps: 25

## Example Demonstration
*(No demonstration GIFs available)*

## Observation Space
The entries of an array in this Box space correspond to the following object features:
| **Index** | **Object** | **Feature** |
| --- | --- | --- |
| 0 | robot | joint_1 |
| 1 | robot | joint_2 |
| 2 | robot | joint_3 |
| 3 | robot | joint_4 |
| 4 | robot | joint_5 |
| 5 | robot | joint_6 |
| 6 | robot | joint_7 |
| 7 | target | x |
| 8 | target | y |
| 9 | target | z |
