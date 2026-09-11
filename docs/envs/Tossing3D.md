# Tossing3D

![random action GIF](assets/random_action_gifs/Tossing3D.gif)

**Random Action Stats**: Total Reward: -25.00, Success: No, Steps: 25

## Description
A 3D task where an object that is initially on the floor must be transferred to the bin. The robot must toss the object into a bin, since it cannot reach the goal position due to an immovable obstacle.

The robot has a holonomic mobile base with powered casters and a Kinova Gen3 arm.

The robot can control:
- Base pose (x, y, theta)
- Arm position (x, y, z)
- Arm orientation (quaternion)
- Gripper position (open/close)


## Available Variants
The variants require tossing different numbers of objects into the bin.

- [`kinder/Tossing3D-o2-v0`](variants/Tossing3D/Tossing3D-o2.md) (o2)
- [`kinder/Tossing3D-o1-v0`](variants/Tossing3D/Tossing3D-o1.md) (o1)

## Initial State Distribution
![initial state GIF](assets/initial_state_gifs/Tossing3D.gif)

## Example Demonstration
![demo GIF](assets/group_gifs/Tossing3D.gif)

## Observation Space
*(Differs per variant, see individual variant pages)*

## Action Space
The 18D action at the default 10 Hz contains base position/yaw deltas (3), arm joint position deltas (7), gripper position (1; 0=open, 1=closed), and absolute arm joint velocity targets (7, rad/s). Position deltas remain bounded by ±0.1 and the gripper by [0, 1]. Velocity targets have no separate command bound; the existing controller limits motor torques. A velocity target is not a guaranteed physical joint speed. Substep schedules remain a separate low-level interface, outside this 18D Box.

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
