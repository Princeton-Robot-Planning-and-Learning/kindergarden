# DynScoopPour2D-o10

## Usage
```python
import kinder
kinder.register_all_environments()
env = kinder.make("kinder/DynScoopPour2D-o10-v0")
```

## Description
This variant has 10 small objects (5 circles, 5 squares).

## Initial State Distribution
![initial state GIF](../../assets/initial_state_gifs/variants/DynScoopPour2D-o10.gif)

## Random Action Behavior
![random action GIF](../../assets/random_action_gifs/variants/DynScoopPour2D-o10.gif)

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
| 40 | small_circle0 | x |
| 41 | small_circle0 | y |
| 42 | small_circle0 | theta |
| 43 | small_circle0 | vx |
| 44 | small_circle0 | vy |
| 45 | small_circle0 | omega |
| 46 | small_circle0 | static |
| 47 | small_circle0 | held |
| 48 | small_circle0 | color_r |
| 49 | small_circle0 | color_g |
| 50 | small_circle0 | color_b |
| 51 | small_circle0 | z_order |
| 52 | small_circle0 | radius |
| 53 | small_circle0 | mass |
| 54 | small_circle1 | x |
| 55 | small_circle1 | y |
| 56 | small_circle1 | theta |
| 57 | small_circle1 | vx |
| 58 | small_circle1 | vy |
| 59 | small_circle1 | omega |
| 60 | small_circle1 | static |
| 61 | small_circle1 | held |
| 62 | small_circle1 | color_r |
| 63 | small_circle1 | color_g |
| 64 | small_circle1 | color_b |
| 65 | small_circle1 | z_order |
| 66 | small_circle1 | radius |
| 67 | small_circle1 | mass |
| 68 | small_circle2 | x |
| 69 | small_circle2 | y |
| 70 | small_circle2 | theta |
| 71 | small_circle2 | vx |
| 72 | small_circle2 | vy |
| 73 | small_circle2 | omega |
| 74 | small_circle2 | static |
| 75 | small_circle2 | held |
| 76 | small_circle2 | color_r |
| 77 | small_circle2 | color_g |
| 78 | small_circle2 | color_b |
| 79 | small_circle2 | z_order |
| 80 | small_circle2 | radius |
| 81 | small_circle2 | mass |
| 82 | small_circle3 | x |
| 83 | small_circle3 | y |
| 84 | small_circle3 | theta |
| 85 | small_circle3 | vx |
| 86 | small_circle3 | vy |
| 87 | small_circle3 | omega |
| 88 | small_circle3 | static |
| 89 | small_circle3 | held |
| 90 | small_circle3 | color_r |
| 91 | small_circle3 | color_g |
| 92 | small_circle3 | color_b |
| 93 | small_circle3 | z_order |
| 94 | small_circle3 | radius |
| 95 | small_circle3 | mass |
| 96 | small_circle4 | x |
| 97 | small_circle4 | y |
| 98 | small_circle4 | theta |
| 99 | small_circle4 | vx |
| 100 | small_circle4 | vy |
| 101 | small_circle4 | omega |
| 102 | small_circle4 | static |
| 103 | small_circle4 | held |
| 104 | small_circle4 | color_r |
| 105 | small_circle4 | color_g |
| 106 | small_circle4 | color_b |
| 107 | small_circle4 | z_order |
| 108 | small_circle4 | radius |
| 109 | small_circle4 | mass |
| 110 | small_square0 | x |
| 111 | small_square0 | y |
| 112 | small_square0 | theta |
| 113 | small_square0 | vx |
| 114 | small_square0 | vy |
| 115 | small_square0 | omega |
| 116 | small_square0 | static |
| 117 | small_square0 | held |
| 118 | small_square0 | color_r |
| 119 | small_square0 | color_g |
| 120 | small_square0 | color_b |
| 121 | small_square0 | z_order |
| 122 | small_square0 | size |
| 123 | small_square0 | mass |
| 124 | small_square1 | x |
| 125 | small_square1 | y |
| 126 | small_square1 | theta |
| 127 | small_square1 | vx |
| 128 | small_square1 | vy |
| 129 | small_square1 | omega |
| 130 | small_square1 | static |
| 131 | small_square1 | held |
| 132 | small_square1 | color_r |
| 133 | small_square1 | color_g |
| 134 | small_square1 | color_b |
| 135 | small_square1 | z_order |
| 136 | small_square1 | size |
| 137 | small_square1 | mass |
| 138 | small_square2 | x |
| 139 | small_square2 | y |
| 140 | small_square2 | theta |
| 141 | small_square2 | vx |
| 142 | small_square2 | vy |
| 143 | small_square2 | omega |
| 144 | small_square2 | static |
| 145 | small_square2 | held |
| 146 | small_square2 | color_r |
| 147 | small_square2 | color_g |
| 148 | small_square2 | color_b |
| 149 | small_square2 | z_order |
| 150 | small_square2 | size |
| 151 | small_square2 | mass |
| 152 | small_square3 | x |
| 153 | small_square3 | y |
| 154 | small_square3 | theta |
| 155 | small_square3 | vx |
| 156 | small_square3 | vy |
| 157 | small_square3 | omega |
| 158 | small_square3 | static |
| 159 | small_square3 | held |
| 160 | small_square3 | color_r |
| 161 | small_square3 | color_g |
| 162 | small_square3 | color_b |
| 163 | small_square3 | z_order |
| 164 | small_square3 | size |
| 165 | small_square3 | mass |
| 166 | small_square4 | x |
| 167 | small_square4 | y |
| 168 | small_square4 | theta |
| 169 | small_square4 | vx |
| 170 | small_square4 | vy |
| 171 | small_square4 | omega |
| 172 | small_square4 | static |
| 173 | small_square4 | held |
| 174 | small_square4 | color_r |
| 175 | small_square4 | color_g |
| 176 | small_square4 | color_b |
| 177 | small_square4 | z_order |
| 178 | small_square4 | size |
| 179 | small_square4 | mass |
