"""Measure the speed of the CPU modes and the GPU mode of one dynamic3d task.

The script steps the task with small random actions in three modes, and prints the
number of control steps for each second of wall time (the sum of all worlds):

- One world on the CPU (``kinder.make``).
- A process pool on the CPU (``gymnasium.vector.AsyncVectorEnv``).
- Many worlds on the GPU (``kinder.make_batch``).

See docs/batch_envs.md for how to select a mode.

Example:

    python scripts/benchmark_batch_env.py kinder/Shelf3D-o1-v0 --num-worlds 1024
"""

# Warp has no type annotations.
# mypy: disable-error-code="no-untyped-call,var-annotated"

from __future__ import annotations

import argparse
import os
import time
from functools import partial

import gymnasium
import numpy as np

import kinder

ACTION_SCALE = 0.02  # the size of the random actions


def _make_env(env_id: str) -> gymnasium.Env:
    kinder.register_all_environments()
    return kinder.make(env_id)


def benchmark_single(env_id: str, num_steps: int) -> float:
    """One world on the CPU."""
    env = _make_env(env_id)
    env.reset(seed=0)
    start = time.perf_counter()
    for _ in range(num_steps):
        env.step(ACTION_SCALE * env.action_space.sample())
    rate = num_steps / (time.perf_counter() - start)
    env.close()  # type: ignore[no-untyped-call]
    return rate


def benchmark_pool(env_id: str, num_processes: int, num_steps: int) -> float:
    """A process pool on the CPU.

    The "spawn" start method gives each process its own render context.
    """
    envs = gymnasium.vector.AsyncVectorEnv(
        [partial(_make_env, env_id)] * num_processes, context="spawn"
    )
    envs.reset(seed=0)
    start = time.perf_counter()
    for _ in range(num_steps):
        envs.step(ACTION_SCALE * envs.action_space.sample())
    rate = num_steps * num_processes / (time.perf_counter() - start)
    envs.close()
    return rate


def benchmark_batch(env_id: str, num_worlds: int, num_steps: int) -> float:
    """Many worlds on the GPU."""
    import warp as wp  # pylint: disable=import-outside-toplevel

    env = kinder.make_batch(env_id, num_worlds, seed=0)
    env.reset()
    rng = np.random.default_rng(0)
    shape = (num_worlds, env.action_dim)
    actions = [
        wp.array(ACTION_SCALE * rng.uniform(-1, 1, shape), float, device=env.device)
        for _ in range(4)
    ]
    env.step(actions[0])
    wp.synchronize()
    start = time.perf_counter()
    for step in range(num_steps):
        env.step(actions[step % 4])
    wp.synchronize()
    rate = num_steps * num_worlds / (time.perf_counter() - start)
    env.close()
    return rate


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("env_id", help='for example "kinder/Shelf3D-o1-v0"')
    parser.add_argument("--num-worlds", type=int, default=1024, help="GPU worlds")
    parser.add_argument("--num-processes", type=int, default=os.cpu_count())
    parser.add_argument("--num-steps", type=int, default=30, help="control steps")
    args = parser.parse_args()

    kinder.register_all_environments()
    rate = benchmark_single(args.env_id, args.num_steps)
    print(f"One world on the CPU: {rate:.0f} control steps/s")
    rate = benchmark_pool(args.env_id, args.num_processes, args.num_steps)
    print(f"{args.num_processes} processes on the CPU: {rate:.0f} control steps/s")
    rate = benchmark_batch(args.env_id, args.num_worlds, args.num_steps)
    print(f"{args.num_worlds} worlds on the GPU: {rate:.0f} control steps/s")


if __name__ == "__main__":
    _main()
