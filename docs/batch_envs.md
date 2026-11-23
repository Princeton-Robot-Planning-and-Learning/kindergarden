# Batch environments on the GPU (MuJoCo Warp)

A batch environment runs many copies ("worlds") of one dynamic3d task at the same
time on the GPU, with [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp).
It is for the fast evaluation of a policy on many episodes.

```python
import kinder

kinder.register_all_environments()
env = kinder.make("kinder/Shelf3D-o1-v0")                         # CPU, one world
env = kinder.make_batch("kinder/Shelf3D-o1-v0", num_worlds=1024)  # GPU, 1024 worlds
```

Batch environments exist for the dynamic3d tasks with the TidyBot (46 tasks) and with
the Sprout (44 tasks, see [sprout.md](sprout.md)).

- [CPU or GPU: how to select](#cpu-or-gpu-how-to-select)
- [Installation](#installation)
- [Usage](#usage)
- [The TidyBot batch environment](#the-tidybot-batch-environment)
- [The physics time step of the TidyBot](#the-physics-time-step-of-the-tidybot)
- [Speed](#speed)
- [Limits](#limits)
- [How to add a robot](#how-to-add-a-robot)
- [To do](#to-do)

## CPU or GPU: how to select

There are three modes. The task, the action, and the observation vector are the same
in all three.

| Mode | How | What you get |
| --- | --- | --- |
| One world on the CPU | `kinder.make(env_id)` | A Gymnasium environment. The reference physics. Camera images. |
| A process pool on the CPU | `gymnasium.vector.AsyncVectorEnv` | Some tens of worlds, one for each CPU core. The same physics and the same features as one world. |
| Many worlds on the GPU | `kinder.make_batch(env_id, num_worlds)` | Hundreds to thousands of worlds. State observations only. |

Use the GPU mode when all of these conditions are true:

- Your policy computes the actions of all worlds in one call (for example a neural
  network on the state vector).
- Your policy does not need camera images or point clouds.
- You want many episodes (hundreds or more), for example for a success rate with a
  small confidence interval, or for reinforcement learning on states.
- You have an NVIDIA GPU, and you can use the Warp install (see
  [Installation](#installation)).

Use a CPU mode in these cases:

- **Your policy handles one world at a time**, for example a planner with inverse
  kinematics. The simulation is then not the slow part, and the GPU mode gives no
  gain. Use the process pool if you want more than one episode at a time.
- **You need images.** Only the CPU environments give camera images in the
  observation loop.
- **You need the reference physics.** Published results and recorded demonstrations
  use CPU MuJoCo with a time step of 0.0005 s. The GPU mode is near to it, but not
  identical (see [Limits](#limits)).
- **You debug a task or a policy.** One world on the CPU is the simplest.

A common procedure is to develop with one world on the CPU, to evaluate on the GPU,
and to confirm the final numbers on a small CPU run.

The process pool on the CPU needs the "spawn" start method. With the default start
method, the render context of the workers can fail.

```python
import gymnasium
import kinder


def make_env():
    kinder.register_all_environments()
    return kinder.make("kinder/Shelf3D-o1-v0")


if __name__ == "__main__":
    envs = gymnasium.vector.AsyncVectorEnv([make_env] * 16, context="spawn")
    obs, info = envs.reset(seed=0)
    obs, reward, terminated, truncated, info = envs.step(envs.action_space.sample())
```

## Installation

MuJoCo Warp needs a newer MuJoCo than the base install of Kinder (MuJoCo 3.3). The
batch environments have their own requirements file. We recommend MuJoCo 4 (tested
on 4.0.1), with Python 3.11 or 3.12.

```bash
pip install --no-deps -e .
pip install -r requirements/dynamic3d_warp.txt
```

The CPU environments also run in this install. Thus one install is sufficient for the
three modes. Their trajectories are not identical to those on MuJoCo 3.3.

## Usage

A batch environment is not a Gymnasium environment. It is a small class: `reset` and
`step` work on all worlds at the same time, and their arrays stay on the GPU.

```python
import kinder

kinder.register_all_environments()
env = kinder.make_batch("kinder/Shelf3D-o1-v0", num_worlds=1024, seed=0)

obs = env.reset()  # Warp array, one row for each world
for _ in range(100):
    action = env.default_action()  # replace this with the output of a policy
    obs, reward, terminated, truncated, info = env.step(action)

# Worlds do not reset by themselves.
done = terminated.numpy() | truncated.numpy()
env.reset(mask=done)

state = env.get_object_centric_state(world=0)  # for a planner
image = env.render(world=0)
env.close()
```

Facts about all batch environments:

- One row of `obs` is the observation vector of the CPU environment, and one row of
  the action is an action of the CPU environment. `env.observation_space` and
  `env.action_space` are the spaces of one world.
- `obs`, `reward`, `terminated`, and `truncated` are Warp arrays. Use
  `array.numpy()` for a NumPy copy, or `warp.to_torch(array)` for a torch tensor
  with no copy. `step` accepts Warp arrays, NumPy arrays, and torch tensors.
- `terminated` is true when the goal of the task is reached. `truncated` is true
  after `horizon` steps. The reward is -1 at each step.
- All worlds use one compiled model. The fixtures have the same poses in all worlds.
  Each world has its own start poses of the objects and the robot.
- `reset()` puts world `i` into start state `i`, so a full reset always gives the same
  states. `reset(mask)` puts each world in the mask into a random start state.
- `render(world)` renders one world with the camera of the CPU environment. It is for
  videos and checks, not for image observations.

Options of `make_batch` (keyword arguments of the batch environment class):

| Option | Default | Meaning |
| --- | --- | --- |
| `seed` | `None` | Seed of the fixture poses and the start states. |
| `device` | Warp default | For example `"cuda:0"`. `"cpu"` works, but it is slow. |
| `horizon` | 1000 | Steps before `truncated`. |
| `physics_timestep` | TidyBot: 0.002 | Time of one physics step, in seconds. |
| `nconmax`, `njmax` | from the scene | Contact and constraint buffer sizes for each world. See `env.buffer_use()`. |
| `scene_bg` | `False` | The textured lab scene (passed to the CPU environment). |

You can also use the classes directly, with the path of a task file:

```python
from kinder.envs.dynamic3d.tidybot_batch_env import TidyBot3DBatchEnv

env = TidyBot3DBatchEnv(task_config_path, num_worlds=1024, seed=0)
```

## The TidyBot batch environment

`TidyBot3DBatchEnv` has the action of `TidyBot3DEnv`, at 10 Hz:

- Base targets: x, y, and yaw (3 values).
- Arm joint targets (7 values).
- Gripper command, from 0 (open) to 1 (closed).
- For the tossing tasks: arm joint velocity targets (7 values). `make_batch` selects
  this action for the `Tossing3D` environments.

With the default configuration, the base and arm targets are relative to the current
positions. The control law is that of `TidyBotRobotEnv`: position actuators for the
base, and a PD law with gravity compensation for the arm. As in the CPU environment,
the arm torques are computed one time for each control step.

The tests compare the batch environment with `TidyBot3DEnv`: the controls of one
step, the motion over 0.5 s with the same time step, and the goal check.

## The physics time step of the TidyBot

`TidyBot3DEnv` uses a physics time step of 0.0005 s. One control step (0.1 s) is thus
200 physics steps. `TidyBot3DBatchEnv` uses 0.002 s by default, which is 50 physics
steps for each control step and 4 times faster.

We selected 0.002 s with replays of recorded demonstrations (the files in `demos/`,
recorded on CPU MuJoCo with 0.0005 s). A replay applies the recorded actions with no
feedback, so it is sensitive to changes in the physics. The table shows the number of
replays that reach the goal.

| Time step (s) | Physics steps for each control step | `Shelf3D-o1` CPU | `Shelf3D-o1` GPU | `SweepIntoDrawer3D-o5` CPU | `SweepIntoDrawer3D-o5` GPU |
| --- | --- | --- | --- | --- | --- |
| 0.0005 (CPU environment) | 200 | 9 of 9 | 9 of 9 | 8 of 20 | 9 of 20 |
| 0.001 | 100 | 9 of 9 | 9 of 9 | 7 of 20 | 6 of 20 |
| **0.002 (batch default)** | 50 | 9 of 9 | 9 of 9 | 7 of 20 | 7 of 20 |
| 0.0025 | 40 | 9 of 9 | 9 of 9 | 4 of 20 | 7 of 20 |
| 0.004 | 25 | 1 of 9 | 0 of 9 | 4 of 20 | not tested |
| 0.005 | 20 | 0 of 9 | not tested | 5 of 20 | not tested |

Notes:

- The pick and place of `Shelf3D-o1` works up to 0.0025 s, and it fails at 0.004 s.
- The sweep demonstrations come from an older version of the task, and their replay
  is not deterministic (they are on the replay blacklist of the tests). Only 8 or 9
  of 20 reach the goal with the reference time step. With 0.002 s, 7 of 20 reach it,
  on the CPU and on the GPU. This difference of 1 or 2 replays is as large as the
  variation between runs (for example 6 and 7 on the GPU for 0.001 s and 0.002 s),
  so this task shows no clear loss, but it cannot exclude a small one. On the CPU,
  the rate decreases to 4 or 5 of 20 from 0.0025 s.
- The demonstrations of `ConstrainedCupboard3D-o1` also come from an older version of
  the task: no replay reaches the goal, with any time step. They show how much the
  motion changes: after about 430 steps, the largest difference in the final
  observation vector between 0.002 s and 0.0005 s was 0.003 (median of 20 replays).
- Objects at rest are not sensitive to the time step. In the tasks with piles of
  cubes (`ScoopPour3D-o30`, `ScoopPour3D-o100`, and `SortClutteredBlocks3D-o20`),
  0 to 4 of some hundreds of objects move more than 1 cm in the first second after
  the reset. This is the same for 0.002 s on the GPU, 0.0005 s on the GPU, and the
  CPU environment.

There is also a reason in the model. The gripper pads of the TidyBot have a contact
time constant of 0.004 s (`solref` in `tidybot.xml`), and the gripper linkage has
0.005 s. MuJoCo needs a time constant of at least two time steps, and it makes a
smaller one larger. Up to 0.002 s, the contacts of the gripper keep their stiffness.
With a larger time step, MuJoCo makes them softer, and the grasp becomes weaker.

Thus 0.002 s is the largest time step with no clear loss in these checks. Pass
`physics_timestep=0.0005` for the time step of the CPU environment, for example to
confirm a result. The control period (0.1 s) must be a multiple of the time step.

The Sprout batch environment uses the time step of the Sprout CPU environment
(0.002 s, 10 physics steps for each control step at 50 Hz).

## Speed

Control steps for each second of wall time, as the sum of all worlds. One control
step is 0.1 s of simulation time. The actions are small random motions of the base and
the arm.

| Task | One CPU world | 32 CPU processes | GPU, 1024 worlds | GPU, more worlds |
| --- | --- | --- | --- | --- |
| `PickPlace3D-o1` | 400 | 2,840 | 36,400 | 87,300 (4096 worlds) |
| `Shelf3D-o8` | 205 | 1,360 | 20,900 | 27,400 (2048 worlds) |
| `ScoopPour3D-o10` | 65 | 520 | 5,900 (512 worlds) | 8,700 (1024 worlds) |

Conditions: MuJoCo 4.0.1 and MuJoCo Warp 4.0.1, an NVIDIA RTX 5090, and a CPU with 32
logical cores. A different job used a part of the GPU and the CPU at the time of the
measurement, so the numbers are low estimates.

What the numbers show:

- The GPU mode is 130 to 220 times faster than one CPU world, and 17 to 31 times
  faster than the process pool.
- The larger time step gives a factor of almost 4 of this. With 0.0005 s,
  `PickPlace3D-o1` with 4096 worlds gives 23,000 control steps for each second.
- A scene with more objects is slower, and it permits fewer worlds.
- For example, 1000 episodes of 500 steps of `Shelf3D-o8` are 41 minutes with one CPU
  world, 6 minutes with the process pool, and 18 seconds on the GPU. These times do
  not include the time of the policy.

Measure your machine and your task with this script:

```bash
python scripts/benchmark_batch_env.py kinder/Shelf3D-o1-v0 --num-worlds 1024
```

The construction of a batch environment takes some seconds. The first time, Warp
also compiles its kernels, which takes more time.

## Limits

- **State observations only.** There are no camera images and no point clouds in the
  observation loop.
- **The policy must run in batch** to get the gain. A policy that handles one world
  at a time on the CPU is the slow part, not the simulation.
- **MuJoCo Warp is not identical to CPU MuJoCo.** A world and the CPU environment
  drift apart when the robot moves (float32 on the GPU, float64 on the CPU). Objects
  at rest agree. Success rates agree in our replays; single trajectories do not.
- **The TidyBot time step is larger** than in the CPU environment (see above).
- **One fixture layout for each batch.** The CPU environment samples the fixture
  poses again at each reset. A batch environment samples them one time, with its
  seed. For more layouts, make more batch environments with different seeds.
- **The reward is -1 at each step** for all tasks. The goal check has the predicates
  `on`, `in`, and `balanced`, which are all predicates of the current task files.
- **No control schedule.** `TidyBot3DEnv.step` also accepts a schedule of actions
  within one control step. The batch environment accepts one action for each step.
- **Robots.** There are batch environments for the TidyBot and the Sprout. There is
  none for the Franka arm (1 task) and the RBY1A (no tasks).
- **Not a Gymnasium environment.** Worlds do not reset by themselves.
- **GPU memory** grows with `num_worlds * njmax`. A large scene permits fewer worlds.
- **Not in the cloud CI.** The cloud CI has no `mujoco_warp`, so it skips the batch
  tests. Run them on a machine with the Warp install:

  ```bash
  pytest tests/envs/dynamic3d/test_tidybot3d_batch.py
  ```

## How to add a robot

`Dynamic3DBatchEnv` (in `src/kinder/envs/dynamic3d/batch_env.py`) has the parts that
are the same for all robots: the model on the GPU, the observation, the goal check,
the reset, and the episode flags. A subclass for a robot implements these methods:

| Method | Function |
| --- | --- |
| `_setup_robot()` | Make the device arrays of the robot (gains, limits, addresses). |
| `_robot_observation_map(column)` | Give the source (`qpos`, `qvel`, or `ctrl` entry) of each robot feature. |
| `_robot_base_qpos_adr()` | The address of the base position, for goals on the robot. |
| `_simulate_control_step()` | The control law (a Warp kernel) and the physics steps. |
| `default_action()` | The action that holds the robot still. |
| `_reset_robot()` (optional) | Reset the state of the control. |
| `_after_settle_step()` (optional) | For a robot that cannot stand still in the settle time. |
| `_get_info()` (optional) | The `info` of `step`. |

`tidybot_batch_env.py` is the shortest example (about 250 lines). Add the robot to
`kinder.make_batch` in `src/kinder/__init__.py`.

## To do

- [ ] A batch environment for the Franka arm.
- [ ] Batch camera images (MuJoCo Warp has a batch renderer; it is not connected).
- [ ] A check of the start states in tasks with piles of objects, where an object can
  still fall after the settle time.
- [ ] The cloud CI for the batch tests.
