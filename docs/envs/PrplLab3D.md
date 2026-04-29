# PrplLab3D

![random action GIF](assets/random_action_gifs/PrplLab3D.gif)

**Random Action Stats**: Total Reward: -25.00, Success: No, Steps: 25

## Description
A 3D environment loaded from the PRPL lab URDF. The robot picks cubes from the floor and places them on the countertop.

## Available Variants
The number of cubes differs between variants. For example, PrplLab3D-o1 has 1 cube and PrplLab3D-o2 has 2 cubes.

- [`kinder/PrplLab3D-o1-v0`](variants/PrplLab3D/PrplLab3D-o1.md) (o1)
- [`kinder/PrplLab3D-o2-v0`](variants/PrplLab3D/PrplLab3D-o2.md) (o2)

## Initial State Distribution
![initial state GIF](assets/initial_state_gifs/PrplLab3D.gif)

## Example Demonstration
*(No demonstration GIFs available)*

## Observation Space
*(Differs per variant, see individual variant pages)*

## Action Space
An action space for mobile manipulation with a 7 DOF robot that can open and close its gripper.

Actions are bounded relative base position, rotation, and joint positions, and open / close.

| **Index** | **Description** |
| --- | --- |
| 0 | delta base x |
| 1 | delta base y |
| 2 | delta base rotation |
| 3 | delta joint 1 |
| 4 | delta joint 2 |
| 5 | delta joint 3 |
| 6 | delta joint 4 |
| 7 | delta joint 5 |
| 8 | delta joint 6 |
| 9 | delta joint 7 |
| 10 | gripper open/close |

The open / close logic is: <-0.5 is close, >0.5 is open, and otherwise no change.


## Rewards
The reward is -1 per timestep. The episode terminates successfullywhen all cubes are on the countertop.

## References
