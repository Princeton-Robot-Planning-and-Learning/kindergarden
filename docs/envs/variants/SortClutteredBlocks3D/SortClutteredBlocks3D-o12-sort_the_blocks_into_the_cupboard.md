# SortClutteredBlocks3D-o12-sort_the_blocks_into_the_cupboard

## Usage
```python
import kinder
kinder.register_all_environments()
env = kinder.make("kinder/SortClutteredBlocks3D-o12-sort_the_blocks_into_the_cupboard-v0")
```

## Description
Sort 9 cubes and 3 cuboids by color into the matching shelves of a single cupboard.

## Initial State Distribution
![initial state GIF](../../assets/initial_state_gifs/variants/SortClutteredBlocks3D-o12-sort_the_blocks_into_the_cupboard.gif)

## Random Action Behavior
![random action GIF](../../assets/random_action_gifs/variants/SortClutteredBlocks3D-o12-sort_the_blocks_into_the_cupboard.gif)

**Random Action Stats**: Total Reward: -25.00, Success: No, Steps: 25

## Example Demonstration
*(No demonstration GIFs available)*

## Observation Space
The entries of an array in this Box space correspond to the following object features:
| **Index** | **Object** | **Feature** |
| --- | --- | --- |
| 0 | blue_cube1 | x |
| 1 | blue_cube1 | y |
| 2 | blue_cube1 | z |
| 3 | blue_cube1 | qw |
| 4 | blue_cube1 | qx |
| 5 | blue_cube1 | qy |
| 6 | blue_cube1 | qz |
| 7 | blue_cube1 | vx |
| 8 | blue_cube1 | vy |
| 9 | blue_cube1 | vz |
| 10 | blue_cube1 | wx |
| 11 | blue_cube1 | wy |
| 12 | blue_cube1 | wz |
| 13 | blue_cube1 | bb_x |
| 14 | blue_cube1 | bb_y |
| 15 | blue_cube1 | bb_z |
| 16 | blue_cube2 | x |
| 17 | blue_cube2 | y |
| 18 | blue_cube2 | z |
| 19 | blue_cube2 | qw |
| 20 | blue_cube2 | qx |
| 21 | blue_cube2 | qy |
| 22 | blue_cube2 | qz |
| 23 | blue_cube2 | vx |
| 24 | blue_cube2 | vy |
| 25 | blue_cube2 | vz |
| 26 | blue_cube2 | wx |
| 27 | blue_cube2 | wy |
| 28 | blue_cube2 | wz |
| 29 | blue_cube2 | bb_x |
| 30 | blue_cube2 | bb_y |
| 31 | blue_cube2 | bb_z |
| 32 | blue_cube3 | x |
| 33 | blue_cube3 | y |
| 34 | blue_cube3 | z |
| 35 | blue_cube3 | qw |
| 36 | blue_cube3 | qx |
| 37 | blue_cube3 | qy |
| 38 | blue_cube3 | qz |
| 39 | blue_cube3 | vx |
| 40 | blue_cube3 | vy |
| 41 | blue_cube3 | vz |
| 42 | blue_cube3 | wx |
| 43 | blue_cube3 | wy |
| 44 | blue_cube3 | wz |
| 45 | blue_cube3 | bb_x |
| 46 | blue_cube3 | bb_y |
| 47 | blue_cube3 | bb_z |
| 48 | blue_cuboid1 | x |
| 49 | blue_cuboid1 | y |
| 50 | blue_cuboid1 | z |
| 51 | blue_cuboid1 | qw |
| 52 | blue_cuboid1 | qx |
| 53 | blue_cuboid1 | qy |
| 54 | blue_cuboid1 | qz |
| 55 | blue_cuboid1 | vx |
| 56 | blue_cuboid1 | vy |
| 57 | blue_cuboid1 | vz |
| 58 | blue_cuboid1 | wx |
| 59 | blue_cuboid1 | wy |
| 60 | blue_cuboid1 | wz |
| 61 | blue_cuboid1 | bb_x |
| 62 | blue_cuboid1 | bb_y |
| 63 | blue_cuboid1 | bb_z |
| 64 | cupboard_0 | x |
| 65 | cupboard_0 | y |
| 66 | cupboard_0 | z |
| 67 | cupboard_0 | qw |
| 68 | cupboard_0 | qx |
| 69 | cupboard_0 | qy |
| 70 | cupboard_0 | qz |
| 71 | green_cube1 | x |
| 72 | green_cube1 | y |
| 73 | green_cube1 | z |
| 74 | green_cube1 | qw |
| 75 | green_cube1 | qx |
| 76 | green_cube1 | qy |
| 77 | green_cube1 | qz |
| 78 | green_cube1 | vx |
| 79 | green_cube1 | vy |
| 80 | green_cube1 | vz |
| 81 | green_cube1 | wx |
| 82 | green_cube1 | wy |
| 83 | green_cube1 | wz |
| 84 | green_cube1 | bb_x |
| 85 | green_cube1 | bb_y |
| 86 | green_cube1 | bb_z |
| 87 | green_cube2 | x |
| 88 | green_cube2 | y |
| 89 | green_cube2 | z |
| 90 | green_cube2 | qw |
| 91 | green_cube2 | qx |
| 92 | green_cube2 | qy |
| 93 | green_cube2 | qz |
| 94 | green_cube2 | vx |
| 95 | green_cube2 | vy |
| 96 | green_cube2 | vz |
| 97 | green_cube2 | wx |
| 98 | green_cube2 | wy |
| 99 | green_cube2 | wz |
| 100 | green_cube2 | bb_x |
| 101 | green_cube2 | bb_y |
| 102 | green_cube2 | bb_z |
| 103 | green_cube3 | x |
| 104 | green_cube3 | y |
| 105 | green_cube3 | z |
| 106 | green_cube3 | qw |
| 107 | green_cube3 | qx |
| 108 | green_cube3 | qy |
| 109 | green_cube3 | qz |
| 110 | green_cube3 | vx |
| 111 | green_cube3 | vy |
| 112 | green_cube3 | vz |
| 113 | green_cube3 | wx |
| 114 | green_cube3 | wy |
| 115 | green_cube3 | wz |
| 116 | green_cube3 | bb_x |
| 117 | green_cube3 | bb_y |
| 118 | green_cube3 | bb_z |
| 119 | green_cuboid1 | x |
| 120 | green_cuboid1 | y |
| 121 | green_cuboid1 | z |
| 122 | green_cuboid1 | qw |
| 123 | green_cuboid1 | qx |
| 124 | green_cuboid1 | qy |
| 125 | green_cuboid1 | qz |
| 126 | green_cuboid1 | vx |
| 127 | green_cuboid1 | vy |
| 128 | green_cuboid1 | vz |
| 129 | green_cuboid1 | wx |
| 130 | green_cuboid1 | wy |
| 131 | green_cuboid1 | wz |
| 132 | green_cuboid1 | bb_x |
| 133 | green_cuboid1 | bb_y |
| 134 | green_cuboid1 | bb_z |
| 135 | red_cube1 | x |
| 136 | red_cube1 | y |
| 137 | red_cube1 | z |
| 138 | red_cube1 | qw |
| 139 | red_cube1 | qx |
| 140 | red_cube1 | qy |
| 141 | red_cube1 | qz |
| 142 | red_cube1 | vx |
| 143 | red_cube1 | vy |
| 144 | red_cube1 | vz |
| 145 | red_cube1 | wx |
| 146 | red_cube1 | wy |
| 147 | red_cube1 | wz |
| 148 | red_cube1 | bb_x |
| 149 | red_cube1 | bb_y |
| 150 | red_cube1 | bb_z |
| 151 | red_cube2 | x |
| 152 | red_cube2 | y |
| 153 | red_cube2 | z |
| 154 | red_cube2 | qw |
| 155 | red_cube2 | qx |
| 156 | red_cube2 | qy |
| 157 | red_cube2 | qz |
| 158 | red_cube2 | vx |
| 159 | red_cube2 | vy |
| 160 | red_cube2 | vz |
| 161 | red_cube2 | wx |
| 162 | red_cube2 | wy |
| 163 | red_cube2 | wz |
| 164 | red_cube2 | bb_x |
| 165 | red_cube2 | bb_y |
| 166 | red_cube2 | bb_z |
| 167 | red_cube3 | x |
| 168 | red_cube3 | y |
| 169 | red_cube3 | z |
| 170 | red_cube3 | qw |
| 171 | red_cube3 | qx |
| 172 | red_cube3 | qy |
| 173 | red_cube3 | qz |
| 174 | red_cube3 | vx |
| 175 | red_cube3 | vy |
| 176 | red_cube3 | vz |
| 177 | red_cube3 | wx |
| 178 | red_cube3 | wy |
| 179 | red_cube3 | wz |
| 180 | red_cube3 | bb_x |
| 181 | red_cube3 | bb_y |
| 182 | red_cube3 | bb_z |
| 183 | red_cuboid1 | x |
| 184 | red_cuboid1 | y |
| 185 | red_cuboid1 | z |
| 186 | red_cuboid1 | qw |
| 187 | red_cuboid1 | qx |
| 188 | red_cuboid1 | qy |
| 189 | red_cuboid1 | qz |
| 190 | red_cuboid1 | vx |
| 191 | red_cuboid1 | vy |
| 192 | red_cuboid1 | vz |
| 193 | red_cuboid1 | wx |
| 194 | red_cuboid1 | wy |
| 195 | red_cuboid1 | wz |
| 196 | red_cuboid1 | bb_x |
| 197 | red_cuboid1 | bb_y |
| 198 | red_cuboid1 | bb_z |
| 199 | robot | pos_base_x |
| 200 | robot | pos_base_y |
| 201 | robot | pos_base_rot |
| 202 | robot | pos_arm_joint1 |
| 203 | robot | pos_arm_joint2 |
| 204 | robot | pos_arm_joint3 |
| 205 | robot | pos_arm_joint4 |
| 206 | robot | pos_arm_joint5 |
| 207 | robot | pos_arm_joint6 |
| 208 | robot | pos_arm_joint7 |
| 209 | robot | pos_gripper |
| 210 | robot | vel_base_x |
| 211 | robot | vel_base_y |
| 212 | robot | vel_base_rot |
| 213 | robot | vel_arm_joint1 |
| 214 | robot | vel_arm_joint2 |
| 215 | robot | vel_arm_joint3 |
| 216 | robot | vel_arm_joint4 |
| 217 | robot | vel_arm_joint5 |
| 218 | robot | vel_arm_joint6 |
| 219 | robot | vel_arm_joint7 |
| 220 | robot | vel_gripper |
