"""Reproduce and film the `set_state` gripper-restore bug this PR fixes.

Companion script for "kinder: carry the gripper's joints in the state, and restore
them in set_state" (this PR). Demonstrates the concrete failure mode with a real
grasp -> release -> restore -> settle sequence, not just the unit test's assertions,
and renders an annotated video of it.

Four phases:
  1. PICKUP  -- approach open, teleport the cube to the pinch site (as
                `test_tidybot3d_restoring_a_grasp_after_a_release_restores_the_grasp`
                does -- MuJoCo friction alone isn't asked to perform the grasp), then
                close the fingers around it and snapshot the resulting state.
  2. RELEASE -- open the fingers; the cube falls under gravity. Confirms the release
                itself works, so a later non-drop in phase 4 isn't an artifact of a
                cube that was never really let go.
  3. RESTORE -- `env.set_state(holding_state)`, the exact call this PR changes.
  4. DROP    -- real `mj_step()`s with NO commanded action, so nothing re-closes the
                gripper and confounds the result. Tracks cube height and gripper qpos
                every captured frame.

Usage: run once on a pre-fix checkout and once on a post-fix checkout, diff the two
reports/videos:

    git checkout <pre-fix commit>
    python repro_gripper_restore_cube_drop.py --out-dir /tmp/before
    git checkout <this branch tip>
    python repro_gripper_restore_cube_drop.py --out-dir /tmp/after

On this repo's pre-fix parent (379f7cd), the measured result is NOT a dramatic visible
drop: at the moment of `set_state`, `right_driver` qpos is still open (~0.003 rad,
matching wherever RELEASE left it) even though the cube was correctly teleported back
between the pads -- but because the driver is a real position actuator, its `ctrl`
target *was* correctly restored, and it closes back around the cube fast enough (over
the free-physics DROP phase) to re-catch it before it falls out. So "the fingers were
briefly in the wrong place" is real and measurable (this script reports the qpos
discrepancy directly), but "the cube visibly falls" is not the reliable way to observe
it -- the qpos numbers are.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import kinder
from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv, TidyBot3DConfig

_TASK_CONFIG_PATH = (
    Path(kinder.__path__[0])
    / "envs"
    / "dynamic3d"
    / "tasks"
    / "Tossing3D"
    / "Tossing3D-o1.json"
)
_GRIPPER_JOINT_SUFFIXES = (
    "right_driver_joint",
    "right_coupler_joint",
    "right_spring_link_joint",
    "right_follower_joint",
    "left_driver_joint",
    "left_coupler_joint",
    "left_spring_link_joint",
    "left_follower_joint",
)

_DROP_PHASE_STEPS = 140
_DROP_RENDER_EVERY = 4
_CAMERA_DISTANCE = 0.55
_CAMERA_AZIMUTH = 180
_CAMERA_ELEVATION = -18
_GHOST_TINT = np.array([70, 210, 255])
_GHOST_ALPHA = 0.5
_GHOST_DIFF_THRESHOLD = 30
_MARKER_RGBA = np.array([1.0, 0.1, 0.85, 0.85])


def _gripper_qpos(robot_env) -> np.ndarray:
    model = robot_env.sim.model
    names = [f"{robot_env.name}_{s}" for s in _GRIPPER_JOINT_SUFFIXES]
    qpos = robot_env.sim.data.mj_data.qpos
    return np.array([qpos[model.get_joint_qpos_addr(n)] for n in names], dtype=float)


def _cube_z(env: ObjectCentricTidyBot3DEnv) -> float:
    state = env._get_current_state()  # noqa: SLF001
    cube = state.get_object_from_name("cube_0")
    return float(state.get(cube, "z"))


def film(
    seed: int,
) -> tuple[np.ndarray, list, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Run the four-phase repro and return (frames, phase_labels, cube_z, gripper_qpos,
    ghost_frame, report)."""
    config = TidyBot3DConfig(camera_width=640, camera_height=480)
    env = ObjectCentricTidyBot3DEnv(
        config=config,
        num_objects=1,
        task_config_path=str(_TASK_CONFIG_PATH),
        scene_bg=True,
        allow_state_access=True,
        scene_render_camera="task_view",
    )
    env.reset(seed=seed)
    render_fps = int(env.metadata["render_fps"])
    robot_env = env._robot_env  # noqa: SLF001
    mj_model = robot_env.sim.model.mj_model
    mj_data = robot_env.sim.data.mj_data

    mujoco.mj_forward(mj_model, mj_data)
    right_pad = mj_data.body(f"{robot_env.name}_right_pad").xpos.copy()
    left_pad = mj_data.body(f"{robot_env.name}_left_pad").xpos.copy()
    lookat = (right_pad + left_pad) / 2.0

    renderer = mujoco.Renderer(mj_model, height=480, width=480)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = _CAMERA_DISTANCE
    cam.azimuth = _CAMERA_AZIMUTH
    cam.elevation = _CAMERA_ELEVATION

    def render_frame() -> np.ndarray:
        renderer.update_scene(mj_data, camera=cam)
        return renderer.render().copy()

    frames: list = []
    phase_label: list = []
    cube_z_hist: list = []
    qpos_hist: list = []

    def snapshot(label: str, n: int = 1) -> None:
        cz, qp = _cube_z(env), _gripper_qpos(robot_env)
        for _ in range(n):
            frames.append(render_frame())
            phase_label.append(label)
            cube_z_hist.append(cz)
            qpos_hist.append(qp.copy())

    def drive(gripper_target: float, steps: int, label: str) -> None:
        action = np.zeros(11, dtype=np.float32)
        action[10] = gripper_target
        for _ in range(steps):
            env.step(action)
            frames.append(render_frame())
            phase_label.append(label)
            cube_z_hist.append(_cube_z(env))
            qpos_hist.append(_gripper_qpos(robot_env))

    # === PHASE 1: PICKUP ===
    drive(0.0, 40, "PHASE 1: PICKUP (approach, open)")
    pinch = np.array(robot_env.sim.data.get_site_xpos("robot_pinch_site"), dtype=float)
    state = env._get_current_state()  # noqa: SLF001
    cube = state.get_object_from_name("cube_0")
    state.set(cube, "x", pinch[0])
    state.set(cube, "y", pinch[1])
    state.set(cube, "z", pinch[2])
    env.set_state(state)
    snapshot("PHASE 1: PICKUP (cube placed at pinch site)", n=5)
    drive(1.0, 50, "PHASE 1: PICKUP (closing)")

    holding_state = env._get_current_state()  # noqa: SLF001
    held_z = _cube_z(env)
    held_x = float(holding_state.get(cube, "x"))
    held_y = float(holding_state.get(cube, "y"))
    held_qpos = _gripper_qpos(robot_env)
    snapshot_frame_idx = len(frames)
    snapshot_qpos_full = mj_data.qpos.copy()

    # One extra render at the snapshotted full qpos, reused as a ghost overlay for every
    # subsequent frame in the compose step below (the pose is static, so one render
    # suffices -- no need to re-render it per frame).
    ghost_data = mujoco.MjData(mj_model)
    ghost_data.qpos[:] = snapshot_qpos_full
    mujoco.mj_forward(mj_model, ghost_data)
    renderer.update_scene(ghost_data, camera=cam)
    ghost_frame = renderer.render().copy()
    renderer.update_scene(mj_data, camera=cam)

    # === PHASE 2: RELEASE ===
    drive(0.0, 90, "PHASE 2: RELEASE (open, cube falls)")
    released_z = _cube_z(env)

    # === PHASE 3: RESTORE -- the call this PR fixes ===
    env.set_state(holding_state)
    restored_z_instant = _cube_z(env)
    restored_qpos_instant = _gripper_qpos(robot_env)
    snapshot("PHASE 3: RESTORE (set_state to held state)", n=15)

    # === PHASE 4: DROP -- real physics, NO commanded action ===
    for step in range(_DROP_PHASE_STEPS):
        mujoco.mj_step(mj_model, mj_data)
        if step % _DROP_RENDER_EVERY == 0:
            frames.append(render_frame())
            phase_label.append("PHASE 4: DROP -- physics only, no action commanded")
            cube_z_hist.append(_cube_z(env))
            qpos_hist.append(_gripper_qpos(robot_env))

    final_z = _cube_z(env)
    final_qpos = _gripper_qpos(robot_env)
    env.close()

    report = {
        "seed": seed,
        "render_fps": render_fps,
        "snapshot_frame_idx": snapshot_frame_idx,
        "held_x": held_x,
        "held_y": held_y,
        "held_z": held_z,
        "released_z": released_z,
        "restored_z_instant": restored_z_instant,
        "final_z_after_drop_phase": final_z,
        "held_right_driver_qpos": float(held_qpos[0]),
        "restored_right_driver_qpos_instant": float(restored_qpos_instant[0]),
        "final_right_driver_qpos": float(final_qpos[0]),
        "cube_fell_again": bool(final_z < restored_z_instant - 0.02),
        "n_frames": len(frames),
    }
    return (
        np.stack(frames),
        phase_label,
        np.array(cube_z_hist),
        np.stack(qpos_hist),
        ghost_frame,
        report,
    )


def compose_video(
    frames: np.ndarray,
    phase_label: list,
    cube_z_hist: np.ndarray,
    qpos_hist: np.ndarray,
    ghost_frame: np.ndarray,
    report: dict,
    out_path: Path,
) -> None:
    """Render the annotated video: main view (with a persistent marker at the
    snapshotted cube position and a ghost overlay of the robot at snapshot time from
    that frame onward) plus a live cube-height / gripper-qpos trace plot below it."""
    font_b = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 18
    )
    font_s = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 12
    )

    main_w, main_h = 480, 480
    plot_w, plot_h = 480, 220
    top_bar_h = 70
    canvas_w = main_w
    canvas_h = top_bar_h + main_h + plot_h
    bg = (18, 22, 30)

    phase_color = {}
    for lbl in set(phase_label):
        if "PICKUP" in lbl:
            phase_color[lbl] = (110, 210, 120)
        elif "RELEASE" in lbl:
            phase_color[lbl] = (230, 140, 40)
        elif "RESTORE" in lbl:
            phase_color[lbl] = (110, 200, 230)
        elif "DROP" in lbl:
            phase_color[lbl] = (220, 100, 100)

    n = len(frames)
    fig, ax = plt.subplots(figsize=(plot_w / 100, plot_h / 100), dpi=100)
    fig.patch.set_facecolor("#12151b")
    ax.set_facecolor("#12151b")
    xs = np.arange(n)
    ax.plot(xs, cube_z_hist, color="#6ec8e6", linewidth=2.0, label="cube z (m)")
    ax.plot(
        xs,
        qpos_hist[:, 0] * 0.3 + cube_z_hist[0] - 0.15,
        color="#e8963e",
        linewidth=1.4,
        label="right_driver qpos (scaled, offset)",
    )
    for i in range(1, n):
        if phase_label[i] != phase_label[i - 1]:
            ax.axvline(i, color="#444a58", linewidth=1, linestyle=":")
    ax.axhline(
        report["held_z"], color="#7fd88f", linewidth=1, linestyle="--", alpha=0.6
    )
    ax.axvline(report["snapshot_frame_idx"], color="#ff2ad4", linewidth=1.6, alpha=0.9)
    ax.annotate(
        "SNAPSHOT",
        xy=(report["snapshot_frame_idx"], report["held_z"]),
        xytext=(report["snapshot_frame_idx"] + 4, report["held_z"] + 0.05),
        color="#ff2ad4",
        fontsize=7,
        fontweight="bold",
    )
    ax.set_xlim(0, n - 1)
    ax.tick_params(colors="#8b93a1", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#2a2f3a")
    ax.legend(
        loc="upper center",
        fontsize=6.5,
        facecolor="#12151b",
        edgecolor="#2a2f3a",
        labelcolor="#e6e8ec",
        ncol=1,
        bbox_to_anchor=(0.5, 1.28),
    )
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    bbox = ax.get_position()
    plot_base = (
        np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        .reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        .copy()
    )
    plt.close(fig)
    plot_base_img = Image.fromarray(plot_base).resize((plot_w, plot_h))
    axes_left_px = bbox.x0 * plot_w
    axes_right_px = bbox.x1 * plot_w

    ghost_int = ghost_frame.astype(int)
    out_frames = []
    for i, (frame, label, cz, qp) in enumerate(
        zip(frames, phase_label, cube_z_hist, qpos_hist)
    ):
        if i >= report["snapshot_frame_idx"]:
            live = frame.astype(int)
            diff = np.abs(live - ghost_int).sum(axis=-1)
            mask = diff > _GHOST_DIFF_THRESHOLD
            blended = (
                live.astype(float) * (1 - _GHOST_ALPHA)
                + (0.4 * ghost_int.astype(float) + 0.6 * _GHOST_TINT) * _GHOST_ALPHA
            )
            frame_out = frame.copy()
            frame_out[mask] = blended[mask].astype(np.uint8)
        else:
            frame_out = frame

        canvas = Image.new("RGB", (canvas_w, canvas_h), bg)
        draw = ImageDraw.Draw(canvas)
        main_img = Image.fromarray(frame_out).resize((main_w, main_h), Image.LANCZOS)
        canvas.paste(main_img, (0, top_bar_h))

        draw.rectangle([0, 0, canvas_w, top_bar_h], fill=(10, 13, 18))
        draw.text(
            (12, 6),
            "does set_state actually drop the cube?",
            font=font_b,
            fill=(240, 240, 245),
        )
        draw.text(
            (12, 30),
            f"seed={report['seed']} · Tossing3D-o1 · cyan ghost = @ snapshot",
            font=font_s,
            fill=(150, 155, 165),
        )
        draw.text(
            (12, 48), label, font=font_s, fill=phase_color.get(label, (240, 240, 245))
        )

        plot_img = plot_base_img.copy()
        pdraw = ImageDraw.Draw(plot_img)
        frac = i / (n - 1)
        cursor_x = int(axes_left_px + frac * (axes_right_px - axes_left_px))
        pdraw.line([(cursor_x, 0), (cursor_x, plot_h)], fill=(230, 230, 235), width=1)
        canvas.paste(plot_img, (0, top_bar_h + main_h))

        snap_marker = "  <-- SNAPSHOT" if i == report["snapshot_frame_idx"] else ""
        draw.text(
            (12, canvas_h - 18),
            f"f{i + 1}/{n}  cube_z={cz:.4f}  r_driver_qpos={qp[0]:+.4f}{snap_marker}",
            font=font_s,
            fill=(255, 42, 212) if snap_marker else (150, 155, 165),
        )
        out_frames.append(np.array(canvas))

    writer = imageio.get_writer(
        str(out_path),
        fps=report["render_fps"],
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    for f in out_frames:
        writer.append_data(f)
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames, phase_label, cube_z_hist, qpos_hist, ghost_frame, report = film(args.seed)
    compose_video(
        frames,
        phase_label,
        cube_z_hist,
        qpos_hist,
        ghost_frame,
        report,
        args.out_dir / "gripper_restore_cube_drop_repro.mp4",
    )
    with open(
        args.out_dir / "gripper_restore_cube_drop_report.json", "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
