# Running the Shelf3D Environment

## Context

The user wants to run the Shelf3D environment from this repo. The `kinder` package is **not currently installed** (`ModuleNotFoundError: No module named 'kinder'`), so the first step is installation. Shelf3D is a PyBullet-based 3D robot manipulation environment where a mobile arm robot picks objects off the floor and places them on a shelf.

---

## Step 1: Install the package

From the repo root (`/home/ian/Documents/kindergarden`):

```bash
python3 -m pip install -e ".[kinematic3d]"
```

This installs `kindergarden` in editable mode with the PyBullet dependencies (`pybullet-arm64>=3.2.8`, `pybullet_helpers>=0.1.0`) needed by Shelf3D.

> **Note:** `pip` is not on PATH — use `python3 -m pip`.

---

## Step 2: Verify the install

```python
import kinder
kinder.register_all_environments()
import gymnasium
env = gymnasium.make("kinder/KinematicShelf3D-o2-v0")
obs, info = env.reset(seed=123)
print(obs.shape)  # Should print something like (50,)
env.close()
```

---

## Step 3: Basic usage

```python
import kinder
kinder.register_all_environments()
import gymnasium

env = gymnasium.make("kinder/KinematicShelf3D-o2-v0")  # 2 cubes
obs, info = env.reset(seed=123)

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

env.close()
```

Or directly (more control):

```python
from kinder.envs.kinematic3d.shelf3d import Shelf3DEnv

env = Shelf3DEnv(num_cubes=2, use_gui=False, render_mode="rgb_array")
obs, _ = env.reset(seed=123)
```

Use `use_gui=True` to open a PyBullet viewer window.

---

## Available variants

| ID | Objects |
|----|---------|
| `kinder/KinematicShelf3D-o1-v0` | 1 cube |
| `kinder/KinematicShelf3D-o2-v0` | 2 cubes |
| `kinder/KinematicShelf3D-o3-v0` | 3 cubes |
| `kinder/KinematicShelf3D-o5-v0` | 5 cubes |
| `kinder/KinematicShelf3D-o10-v0` | 10 cubes |

---

## Run the existing tests

```bash
python3 -m pytest tests/envs/kinematic3d/test_shelf3d.py -v
```

Key test: `test_pick_place` — executes a full motion-planned pick-and-place sequence.

---

## Critical files

- [src/kinder/envs/kinematic3d/shelf3d.py](src/kinder/envs/kinematic3d/shelf3d.py) — main environment
- [tests/envs/kinematic3d/test_shelf3d.py](tests/envs/kinematic3d/test_shelf3d.py) — full usage examples
- [pyproject.toml](pyproject.toml) — dependencies
- [src/kinder/__init__.py](src/kinder/__init__.py#L343-L359) — gym registration

---

## Action space (11-dim)

| Indices | Meaning |
|---------|---------|
| 0–2 | Base delta (x, y, yaw) |
| 3–9 | Arm joint deltas (7-DOF Kinova Gen3) |
| 10 | Gripper: -1.0 = close, +1.0 = open |
