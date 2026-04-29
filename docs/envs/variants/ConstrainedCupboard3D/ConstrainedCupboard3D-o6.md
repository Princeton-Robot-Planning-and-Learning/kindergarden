# ConstrainedCupboard3D-o6

## Usage
```python
import kinder
kinder.register_all_environments()
env = kinder.make("kinder/ConstrainedCupboard3D-o6-v0")
```

## Description
This variant uses the 'ground' scene type with 3 objects.

## Initial State Distribution
![initial state GIF](../../assets/initial_state_gifs/variants/ConstrainedCupboard3D-o6.gif)

## Random Action Behavior
![random action GIF](../../assets/random_action_gifs/variants/ConstrainedCupboard3D-o6.gif)

**Random Action Stats**: Total Reward: -25.00, Success: No, Steps: 25

## Example Demonstration
*(No demonstration GIFs available)*

## Observation Space
The entries of an array in this Box space correspond to the following object features:
| **Index** | **Object** | **Feature** |
| --- | --- | --- |
| 0 | cuboid_0 | x |
| 1 | cuboid_0 | y |
| 2 | cuboid_0 | z |
| 3 | cuboid_0 | qw |
| 4 | cuboid_0 | qx |
| 5 | cuboid_0 | qy |
| 6 | cuboid_0 | qz |
| 7 | cuboid_0 | vx |
| 8 | cuboid_0 | vy |
| 9 | cuboid_0 | vz |
| 10 | cuboid_0 | wx |
| 11 | cuboid_0 | wy |
| 12 | cuboid_0 | wz |
| 13 | cuboid_0 | bb_x |
| 14 | cuboid_0 | bb_y |
| 15 | cuboid_0 | bb_z |
| 16 | cuboid_1 | x |
| 17 | cuboid_1 | y |
| 18 | cuboid_1 | z |
| 19 | cuboid_1 | qw |
| 20 | cuboid_1 | qx |
| 21 | cuboid_1 | qy |
| 22 | cuboid_1 | qz |
| 23 | cuboid_1 | vx |
| 24 | cuboid_1 | vy |
| 25 | cuboid_1 | vz |
| 26 | cuboid_1 | wx |
| 27 | cuboid_1 | wy |
| 28 | cuboid_1 | wz |
| 29 | cuboid_1 | bb_x |
| 30 | cuboid_1 | bb_y |
| 31 | cuboid_1 | bb_z |
| 32 | cuboid_2 | x |
| 33 | cuboid_2 | y |
| 34 | cuboid_2 | z |
| 35 | cuboid_2 | qw |
| 36 | cuboid_2 | qx |
| 37 | cuboid_2 | qy |
| 38 | cuboid_2 | qz |
| 39 | cuboid_2 | vx |
| 40 | cuboid_2 | vy |
| 41 | cuboid_2 | vz |
| 42 | cuboid_2 | wx |
| 43 | cuboid_2 | wy |
| 44 | cuboid_2 | wz |
| 45 | cuboid_2 | bb_x |
| 46 | cuboid_2 | bb_y |
| 47 | cuboid_2 | bb_z |
| 48 | cuboid_3 | x |
| 49 | cuboid_3 | y |
| 50 | cuboid_3 | z |
| 51 | cuboid_3 | qw |
| 52 | cuboid_3 | qx |
| 53 | cuboid_3 | qy |
| 54 | cuboid_3 | qz |
| 55 | cuboid_3 | vx |
| 56 | cuboid_3 | vy |
| 57 | cuboid_3 | vz |
| 58 | cuboid_3 | wx |
| 59 | cuboid_3 | wy |
| 60 | cuboid_3 | wz |
| 61 | cuboid_3 | bb_x |
| 62 | cuboid_3 | bb_y |
| 63 | cuboid_3 | bb_z |
| 64 | cuboid_4 | x |
| 65 | cuboid_4 | y |
| 66 | cuboid_4 | z |
| 67 | cuboid_4 | qw |
| 68 | cuboid_4 | qx |
| 69 | cuboid_4 | qy |
| 70 | cuboid_4 | qz |
| 71 | cuboid_4 | vx |
| 72 | cuboid_4 | vy |
| 73 | cuboid_4 | vz |
| 74 | cuboid_4 | wx |
| 75 | cuboid_4 | wy |
| 76 | cuboid_4 | wz |
| 77 | cuboid_4 | bb_x |
| 78 | cuboid_4 | bb_y |
| 79 | cuboid_4 | bb_z |
| 80 | cuboid_5 | x |
| 81 | cuboid_5 | y |
| 82 | cuboid_5 | z |
| 83 | cuboid_5 | qw |
| 84 | cuboid_5 | qx |
| 85 | cuboid_5 | qy |
| 86 | cuboid_5 | qz |
| 87 | cuboid_5 | vx |
| 88 | cuboid_5 | vy |
| 89 | cuboid_5 | vz |
| 90 | cuboid_5 | wx |
| 91 | cuboid_5 | wy |
| 92 | cuboid_5 | wz |
| 93 | cuboid_5 | bb_x |
| 94 | cuboid_5 | bb_y |
| 95 | cuboid_5 | bb_z |
| 96 | cupboard_0 | x |
| 97 | cupboard_0 | y |
| 98 | cupboard_0 | z |
| 99 | cupboard_0 | qw |
| 100 | cupboard_0 | qx |
| 101 | cupboard_0 | qy |
| 102 | cupboard_0 | qz |
| 103 | cupboard_1 | x |
| 104 | cupboard_1 | y |
| 105 | cupboard_1 | z |
| 106 | cupboard_1 | qw |
| 107 | cupboard_1 | qx |
| 108 | cupboard_1 | qy |
| 109 | cupboard_1 | qz |
| 110 | cupboard_10 | x |
| 111 | cupboard_10 | y |
| 112 | cupboard_10 | z |
| 113 | cupboard_10 | qw |
| 114 | cupboard_10 | qx |
| 115 | cupboard_10 | qy |
| 116 | cupboard_10 | qz |
| 117 | cupboard_2 | x |
| 118 | cupboard_2 | y |
| 119 | cupboard_2 | z |
| 120 | cupboard_2 | qw |
| 121 | cupboard_2 | qx |
| 122 | cupboard_2 | qy |
| 123 | cupboard_2 | qz |
| 124 | cupboard_3 | x |
| 125 | cupboard_3 | y |
| 126 | cupboard_3 | z |
| 127 | cupboard_3 | qw |
| 128 | cupboard_3 | qx |
| 129 | cupboard_3 | qy |
| 130 | cupboard_3 | qz |
| 131 | cupboard_4 | x |
| 132 | cupboard_4 | y |
| 133 | cupboard_4 | z |
| 134 | cupboard_4 | qw |
| 135 | cupboard_4 | qx |
| 136 | cupboard_4 | qy |
| 137 | cupboard_4 | qz |
| 138 | cupboard_5 | x |
| 139 | cupboard_5 | y |
| 140 | cupboard_5 | z |
| 141 | cupboard_5 | qw |
| 142 | cupboard_5 | qx |
| 143 | cupboard_5 | qy |
| 144 | cupboard_5 | qz |
| 145 | cupboard_6 | x |
| 146 | cupboard_6 | y |
| 147 | cupboard_6 | z |
| 148 | cupboard_6 | qw |
| 149 | cupboard_6 | qx |
| 150 | cupboard_6 | qy |
| 151 | cupboard_6 | qz |
| 152 | cupboard_7 | x |
| 153 | cupboard_7 | y |
| 154 | cupboard_7 | z |
| 155 | cupboard_7 | qw |
| 156 | cupboard_7 | qx |
| 157 | cupboard_7 | qy |
| 158 | cupboard_7 | qz |
| 159 | cupboard_8 | x |
| 160 | cupboard_8 | y |
| 161 | cupboard_8 | z |
| 162 | cupboard_8 | qw |
| 163 | cupboard_8 | qx |
| 164 | cupboard_8 | qy |
| 165 | cupboard_8 | qz |
| 166 | cupboard_9 | x |
| 167 | cupboard_9 | y |
| 168 | cupboard_9 | z |
| 169 | cupboard_9 | qw |
| 170 | cupboard_9 | qx |
| 171 | cupboard_9 | qy |
| 172 | cupboard_9 | qz |
| 173 | robot | pos_base_x |
| 174 | robot | pos_base_y |
| 175 | robot | pos_base_rot |
| 176 | robot | pos_arm_joint1 |
| 177 | robot | pos_arm_joint2 |
| 178 | robot | pos_arm_joint3 |
| 179 | robot | pos_arm_joint4 |
| 180 | robot | pos_arm_joint5 |
| 181 | robot | pos_arm_joint6 |
| 182 | robot | pos_arm_joint7 |
| 183 | robot | pos_gripper |
| 184 | robot | vel_base_x |
| 185 | robot | vel_base_y |
| 186 | robot | vel_base_rot |
| 187 | robot | vel_arm_joint1 |
| 188 | robot | vel_arm_joint2 |
| 189 | robot | vel_arm_joint3 |
| 190 | robot | vel_arm_joint4 |
| 191 | robot | vel_arm_joint5 |
| 192 | robot | vel_arm_joint6 |
| 193 | robot | vel_arm_joint7 |
| 194 | robot | vel_gripper |
