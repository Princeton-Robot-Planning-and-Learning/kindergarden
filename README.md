# KinDER

A physical reasoning benchmark for robot learning and planning.

## Website

See [https://prpl.group/kinder-site/](https://prpl.group/kinder-site/) for documentation and tutorials.

### Requirements
1. Python >=3.10, <3.13
2. Tested on MacOS 13-15 and Ubuntu 22.04 (but we aim to support most platforms)

## Installation

### From PyPI

```bash
pip install kindergarden          # core only
pip install kindergarden[all]     # all environment dependencies
pip install kindergarden[dynamic2d]     # only dynamic2d environments
pip install kindergarden[kinematic2d]   # only kinematic2d environments
pip install kindergarden[kinematic3d]   # only kinematic3d environments
pip install kindergarden[dynamic3d]     # only dynamic3d environments
```

You can also combine extras: `pip install kindergarden[kinematic2d,kinematic3d]`

### From Source

We strongly recommend [uv](https://docs.astral.sh/uv/getting-started/installation/), but other standard setups work too.

```bash
git clone https://github.com/Princeton-Robot-Planning-and-Learning/kindergarden.git
cd kindergarden
uv pip install -e ".[develop]"   # all dependencies + dev tools
```

Or install only what you need:
```bash
uv pip install -e .              # core only
uv pip install -e ".[all]"      # all environment dependencies
uv pip install -e ".[dynamic2d]" # only dynamic2d environments
```

To check the installation, run `./run_ci_checks.sh`. It should complete with all green successes.

## Usage Example

### Basic Usage (Gym API)

```python
import kinder
kinder.register_all_environments()
env = kinder.make("kinder/Obstruction2D-o3-v0")  # 3 obstructions
obs, info = env.reset()  # procedural generation
action = env.action_space.sample()
next_obs, reward, terminated, truncated, info = env.step(action)
img = env.render()  
```

### Object-Centric States

All environments in KinDER use object-centric states. For example:

```python
from kinder.envs.kinematic2d.obstruction2d import ObjectCentricObstruction2DEnv
env = ObjectCentricObstruction2DEnv(num_obstructions=3)
obs, _ = env.reset(seed=123)
print(obs.pretty_str())
```
Here, `obs` is an [ObjectCentricState](https://github.com/tomsilver/relational-structs/blob/main/src/relational_structs/object_centric_state.py#L25), and the printout is:
```
############################################################### STATE ###############################################################
type: crv_robot           x         y    theta    base_radius    arm_joint    arm_length    vacuum    gripper_height    gripper_width
-----------------  --------  --------  -------  -------------  -----------  ------------  --------  ----------------  ---------------
robot              0.885039  0.803795  -1.5708            0.1          0.1           0.2         0              0.07             0.01

type: rectangle           x         y    theta    static    color_r    color_g    color_b    z_order      width     height
-----------------  --------  --------  -------  --------  ---------  ---------  ---------  ---------  ---------  ---------
obstruction0       0.422462  0.100001        0         0       0.75        0.1        0.1        100  0.132224   0.0766399
obstruction1       0.804663  0.100001        0         0       0.75        0.1        0.1        100  0.0805652  0.0955062
obstruction2       0.559246  0.100001        0         0       0.75        0.1        0.1        100  0.12608    0.180172

type: target_block          x         y    theta    static    color_r    color_g    color_b    z_order     width    height
--------------------  -------  --------  -------  --------  ---------  ---------  ---------  ---------  --------  --------
target_block          1.20082  0.100001        0         0   0.501961          0   0.501961        100  0.138302  0.155183

type: target_surface           x    y    theta    static    color_r    color_g    color_b    z_order     width    height
----------------------  --------  ---  -------  --------  ---------  ---------  ---------  ---------  --------  --------
target_surface          0.499675    0        0         1   0.501961          0   0.501961        101  0.180286       0.1
#####################################################################################################################################
```

For compatibility with baselines, the observations provided by the main environments are vectors. It is easy to convert between vectors and object-centric states. For example:
```python
import kinder
kinder.register_all_environments()
env = kinder.make("kinder/Obstruction2D-o3-v0")
vec_obs, _ = env.reset(seed=123)
object_centric_obs = env.observation_space.devectorize(vec_obs)
recovered_vec_obs = env.observation_space.vectorize(object_centric_obs)
```

## Quick Environment Reference

| Environment | Category | Example Environment ID |
|---|---|---|
| ClutteredRetrieval2D | Kinematic2D | `kinder/ClutteredRetrieval2D-o10-v0` |
| Motion2D | Kinematic2D | `kinder/Motion2D-p5-v0` |
| Obstruction2D | Kinematic2D | `kinder/Obstruction2D-o4-v0` |
| PushPullHook2D | Kinematic2D | `kinder/PushPullHook2D-v0` |
| ClutteredStorage2D | Kinematic2D | `kinder/ClutteredStorage2D-b15-v0` |
| StickButton2D | Kinematic2D | `kinder/StickButton2D-b10-v0` |
| DynObstruction2D | Dynamic2D | `kinder/DynObstruction2D-o3-v0` |
| DynPushPullHook2D | Dynamic2D | `kinder/DynPushPullHook2D-o5-v0` |
| DynPushT2D | Dynamic2D | `kinder/DynPushT2D-t1-v0` |
| DynScoopPour2D | Dynamic2D | `kinder/DynScoopPour2D-o50-v0` |
| Obstruction3D | Kinematic3D | `kinder/Obstruction3D-o4-v0` |
| Packing3D | Kinematic3D | `kinder/Packing3D-p3-v0` |
| Table3D | Kinematic3D | `kinder/Table3D-o3-v0` |
| Transport3D | Kinematic3D | `kinder/Transport3D-o2-v0` |
| BaseMotion3D | Kinematic3D | `kinder/BaseMotion3D-v0` |
| Shelf3D | Kinematic3D | `kinder/Shelf3D-o10-v0` |
| ConstrainedCupboard3D | Dynamic3D | `kinder/ConstrainedCupboard3D-o6-v0` |
| SortClutteredBlocks3D | Dynamic3D | `kinder/SortClutteredBlocks3D-o20-sort_the_cluttered_blocks_into_bowls-v0` |
| Rearrange3D | Dynamic3D | `kinder/Rearrange3D-o2-put_the_boxed_drink_and_the_can_next_to_the_bowl-v0` |
| SweepSimple3D | Dynamic3D | `kinder/SweepSimple3D-o50-sweep_the_blocks_to_the_left_side_of_the_kitchen_island-v0` |
| Dynamo3D | Dynamic3D | `kinder/Dynamo3D-o1-v0` |
| Tossing3D | Dynamic3D | `kinder/Tossing3D-o1-v0` |
| ScoopPour3D | Dynamic3D | `kinder/ScoopPour3D-o10-v0` |
| BalanceBeam3D | Dynamic3D | `kinder/BalanceBeam3D-o3-v0` |
| SweepIntoDrawer3D | Dynamic3D | `kinder/SweepIntoDrawer3D-o5-v0` |

## Contributing

### General Guidelines
* All checks must pass before code is merged (see `./run_ci_checks.sh`)
* All code goes through the pull request review process

### Adding New Environments
Some new environment requests are in Issues. To add a new environment, please see the examples in `src/kinder/env`. Also consider:
* Environments are registered in `src/kinder/__init__.py`
* Each environment should have at least one demonstration (see `scripts/collect_demos.py`)
* After collecting a demonstraction, create a video with `scripts/generate_demo_video.py`, which will be used in the autogenerated documentation
