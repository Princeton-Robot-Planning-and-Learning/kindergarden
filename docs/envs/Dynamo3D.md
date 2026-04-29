# Dynamo3D

![random action GIF](assets/random_action_gifs/Dynamo3D.gif)

**Random Action Stats**: Total Reward: -25.00, Success: No, Steps: 25

## Description
A 3D navigation-among-movable-objects task where the robot must reach a goal region on the floor while several chair obstacles are placed in the workspace and may need to be moved out of the way.

The robot has a holonomic mobile base with powered casters and a Kinova Gen3 arm.

The robot can control:
- Base pose (x, y, theta)
- Arm position (x, y, z)
- Arm orientation (quaternion)
- Gripper position (open/close)


## Available Variants
The variants differ in the number of obstacle chairs placed in the workspace.

- [`kinder/Dynamo3D-o12-v0`](variants/Dynamo3D/Dynamo3D-o12.md) (o12)
- [`kinder/Dynamo3D-o3-v0`](variants/Dynamo3D/Dynamo3D-o3.md) (o3)
- [`kinder/Dynamo3D-o1-v0`](variants/Dynamo3D/Dynamo3D-o1.md) (o1)

## Initial State Distribution
![initial state GIF](assets/initial_state_gifs/Dynamo3D.gif)

## Example Demonstration
![demo GIF](assets/group_gifs/Dynamo3D.gif)

## Observation Space
*(Differs per variant, see individual variant pages)*

## Action Space
Actions: base pos and yaw (3), arm joints (7), gripper pos (1)

## Rewards
The primary reward is for successfully placing objects at their target locations.
- A reward of +1.0 is given for each object placed within a 5cm tolerance of its target.
- A smaller positive reward is given for objects within a 10cm tolerance to guide the robot.
- A small negative reward (-0.01) is applied at each timestep to encourage efficiency.
The episode terminates when all objects are placed at their respective targets.


## References
TidyBot++: An Open-Source Holonomic Mobile Manipulator
for Robot Learning
- Jimmy Wu, William Chong, Robert Holmberg, Aaditya Prasad, Yihuai Gao,
  Oussama Khatib, Shuran Song, Szymon Rusinkiewicz, Jeannette Bohg
- Conference on Robot Learning (CoRL), 2024

https://github.com/tidybot2/tidybot2
