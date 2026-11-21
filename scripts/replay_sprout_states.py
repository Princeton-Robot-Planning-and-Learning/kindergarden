"""Replay a recorded run of the Sprout in the renderer of a Kinder task.

The policies of the Sprout run in other simulators (for example mjlab, with a scene
that ``scripts/export_sprout_scene.py`` wrote). This script shows such a run in the
Kinder scene: it reads the robot and object poses of each control step from a file,
puts them into the simulation state, and writes a video.

The states file is a NumPy ``.npz`` archive with these arrays. The first two
dimensions are (steps, worlds):

- ``robot_root``: position (3) and quaternion (4, w first) of the robot base.
- ``robot_joints``: the joint positions, in the order of ``joint_names``.
- ``joint_names``: the joint names of the vendor model.
- ``object_<name>``: position (3) and quaternion (4) of each object.
- ``step_dt``: the time of one step, in seconds.
- ``titles`` (optional): a text for each step, shown in the video.
- ``reached`` (optional): for each world, whether the run reached the goal.

Example:

    python scripts/replay_sprout_states.py kinder/SproutShelf3D-o1-v0 run.npz \
        run.mp4 --gif run.gif --speed 2
"""

# pylint: disable=protected-access

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import imageio.v2 as iio
import imageio_ffmpeg
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

import kinder
from kinder.envs.dynamic3d.robots.sprout_robot_env import SproutRobotEnv

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _add_title(image: NDArray[np.uint8], title: str) -> NDArray[np.uint8]:
    """Write a title on a dark band at the top of an image."""
    picture = Image.fromarray(image)
    draw = ImageDraw.Draw(picture)
    try:
        font: ImageFont.ImageFont | ImageFont.FreeTypeFont = ImageFont.truetype(
            FONT_PATH, 16
        )
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, picture.width, 26], fill=(20, 20, 20))
    draw.text((8, 4), title, fill=(255, 255, 255), font=font)
    return np.asarray(picture)


def write_gif(video: Path, gif: Path, fps: int, width: int) -> None:
    """Convert a video to a small GIF, with the ffmpeg program of imageio.

    One color palette for the full video and a low frame rate keep the file small.
    """
    filters = (
        f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=max_colors=128:stats_mode=diff[palette];"
        "[b][palette]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"
    )
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [ffmpeg, "-y", "-loglevel", "error", "-i", str(video)]
    subprocess.run([*command, "-filter_complex", filters, str(gif)], check=True)


def replay(
    env_id: str,
    states_path: Path,
    world: int | None,
    speed: float,
    video_fps: int,
) -> tuple[list[NDArray[np.uint8]], int]:
    """Render one world of a states file in a Kinder task.

    Args:
        env_id: The Kinder task, for example "kinder/SproutShelf3D-o1-v0".
        states_path: The states file (see the module docstring).
        world: The world to show. None: the first world that reached the goal.
        speed: The video speed, as a multiple of real time.
        video_fps: Frames for each second of video.

    Returns:
        The frames, and the world that they show.
    """
    states = np.load(states_path)
    if world is None:
        reached = states["reached"] if "reached" in states else np.zeros(1, bool)
        world = int(np.argmax(reached)) if reached.any() else 0

    kinder.register_all_environments()
    env = kinder.make(env_id, render_mode="rgb_array", scene_bg=True)
    env.reset(seed=0)
    robot_env = env.unwrapped._object_centric_env._robot_env  # type: ignore
    assert isinstance(robot_env, SproutRobotEnv) and robot_env.sim is not None
    mj_model = robot_env.sim.model.mj_model
    mj_data = robot_env.sim.data.mj_data

    # Where each recorded value goes in the generalized positions (qpos).
    joint_order = [
        robot_env.model.joint_names.index(str(name)) for name in states["joint_names"]
    ]
    joint_adr = robot_env.joint_qpos_adr + np.array(joint_order)
    object_adr = {}
    for key in states.files:
        if key.startswith("object_"):
            body = mj_model.body(key.removeprefix("object_"))
            object_adr[key] = int(mj_model.jnt_qposadr[body.jntadr[0]])

    num_steps = states["robot_root"].shape[0]
    steps_per_frame = speed / (video_fps * float(states["step_dt"]))
    frames = []
    for frame_index in range(int(num_steps / steps_per_frame)):
        step = int(frame_index * steps_per_frame)
        base = robot_env.base_qpos_adr
        mj_data.qpos[base : base + 7] = states["robot_root"][step, world]
        mj_data.qpos[joint_adr] = states["robot_joints"][step, world]
        for key, adr in object_adr.items():
            mj_data.qpos[adr : adr + 7] = states[key][step, world]
        robot_env.sim.forward()
        image = env.render()
        assert isinstance(image, np.ndarray)
        if "titles" in states:
            seconds = step * float(states["step_dt"])
            title = f"{states['titles'][step]}   (t = {seconds:.0f} s, {speed:g}x)"
            image = _add_title(image, title)
        frames.append(image)
    env.close()
    return frames, world


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("env_id", help='for example "kinder/SproutShelf3D-o1-v0"')
    parser.add_argument("states", type=Path, help="the states file (.npz)")
    parser.add_argument("video", type=Path, help="the video file to write (.mp4)")
    parser.add_argument("--world", type=int, help="default: a world with success")
    parser.add_argument("--speed", type=float, default=1.0, help="1: real time")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--gif", type=Path, help="also write a small GIF")
    parser.add_argument("--gif-fps", type=int, default=10)
    parser.add_argument("--gif-width", type=int, default=480)
    args = parser.parse_args()

    frames, world = replay(args.env_id, args.states, args.world, args.speed, args.fps)
    args.video.parent.mkdir(parents=True, exist_ok=True)
    iio.mimsave(args.video, frames, fps=args.fps, macro_block_size=8)  # type: ignore
    print(f"Wrote {args.video}: world {world}, {len(frames) / args.fps:.0f} s")
    if args.gif is not None:
        write_gif(args.video, args.gif, args.gif_fps, args.gif_width)
        print(f"Wrote {args.gif}: {args.gif.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    _main()
