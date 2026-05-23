"""Continuously record RGB + depth from the cameras in camera_extrinsics.json.

For each serial listed in the extrinsics JSON, opens an aligned RealSense
pipeline and writes frames to disk:

  out/<serial>/rgb/<frame_idx:06d>.png      (BGR uint8)
  out/<serial>/depth/<frame_idx:06d>.png    (uint16, millimetres)
  out/<serial>/timestamps.csv               (frame_idx, ts_unix, ts_device_ms)

Both cameras stream in parallel from a single process (one thread per
device). Stops on Ctrl+C, after --duration seconds, or after --max-frames
frames per camera.

Before streaming, the bimanual follower arms are driven to a "park" pose
that lifts them up and swings the elbow outward, so they don't appear in
the recording. Same park pose as
/home/yunshuang/depth_lerobot/sim_3d_bimanual-main/generate_goal_pc.py.
Disable with --no-park if the robot isn't powered.

Depth is aligned to color and stored as uint16 millimetres so it round-
trips through PNG losslessly (matches the convention in
/home/yunshuang/depth_lerobot/pointcloud_generation/get_images.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


DEFAULT_EXTRINSICS = "/home/yunshuang/3D_Bimanual_repo/scripts/data_collection/camera_extrinsics.json"
DEFAULT_OUT = "/home/yunshuang/Binomap/recordings"
WARMUP_FRAMES = 30

# ── Park-pose constants (mirrored from sim_3d_bimanual-main/generate_goal_pc.py).
# Joint indices on Trossen wxai_v0: j0 base yaw, j1 shoulder pitch, j2 elbow,
# j3 wrist pitch, j4 wrist roll, j5 wrist yaw, j6 gripper (m).
_HOME_POSE_PER_ARM = [0.0, np.pi / 3, np.pi / 6, np.pi / 5, 0.0, 0.0, 0.0]
_SWING_JOINT_IDX = 2  # elbow
_SWING_ANGLE_RAD = 0.7000
_LIFT_J1_RAD = 1.0  # subtract from shoulder pitch
_PARK_BLEND_S = 6.0
_SETTLE_SECONDS = _PARK_BLEND_S + 1.0

_lifted_base = list(_HOME_POSE_PER_ARM)
_lifted_base[1] = max(0.0, _HOME_POSE_PER_ARM[1] - _LIFT_J1_RAD)
_LEFT_PARK_POSE = list(_lifted_base)
_LEFT_PARK_POSE[_SWING_JOINT_IDX] = +_SWING_ANGLE_RAD
_RIGHT_PARK_POSE = list(_lifted_base)
_RIGHT_PARK_POSE[_SWING_JOINT_IDX] = +_SWING_ANGLE_RAD
_PER_ARM_PARK_POSES = {"left": _LEFT_PARK_POSE, "right": _RIGHT_PARK_POSE}


def park_robot_arms():
    """Drive both follower arms to the park pose. Returns the connected robot
    so the caller can disconnect it after recording finishes.

    Uses lerobot's TrossenAIStationary robot path. Bypasses send_action's short
    MIN_TIME_TO_MOVE by calling arm.driver.set_all_positions with a long blend.
    """
    # Imported lazily so --no-park works without lerobot installed.
    from lerobot.common.robot_devices.robots.configs import (
        TrossenAIStationaryRobotConfig,
    )
    from lerobot.common.robot_devices.robots.utils import make_robot_from_config

    robot_cfg = TrossenAIStationaryRobotConfig(camera_interface="intel_realsense")
    # We open the cameras ourselves via pyrealsense2; don't let lerobot grab them.
    robot_cfg.cameras.clear()
    robot = make_robot_from_config(robot_cfg)
    robot.leader_arms = {}
    robot.cameras = {}
    robot.connect()

    print(
        f"[park] Moving {len(robot.follower_arms)} arm(s) over"
        f" {_PARK_BLEND_S:.1f}s (swing j{_SWING_JOINT_IDX}"
        f" ±{_SWING_ANGLE_RAD:.4f} rad, lift j1 by {_LIFT_J1_RAD:.2f} rad)"
    )
    for arm_name, arm in robot.follower_arms.items():
        pose = _PER_ARM_PARK_POSES.get(arm_name, list(_HOME_POSE_PER_ARM))
        print(f"[park]   {arm_name}: {pose}")
        arm.driver.set_all_positions(pose, _PARK_BLEND_S, False)
    time.sleep(_SETTLE_SECONDS)
    print("[park] arms settled.")
    return robot


_stop = threading.Event()


def _handle_sigint(signum, frame):
    print("\n[main] Ctrl+C received, stopping...")
    _stop.set()


def _load_camera_specs(json_path: str):
    """Returns list of (serial, arm, width, height, fps_hint)."""
    with open(json_path, "r") as f:
        raw = json.load(f)
    cams = []
    for serial, entry in raw.items():
        intr = entry["intrinsics"]
        cams.append(
            (
                serial,
                entry.get("arm", "?"),
                int(intr["width"]),
                int(intr["height"]),
            )
        )
    return cams


def _record_one(
    serial: str,
    arm: str,
    width: int,
    height: int,
    fps: int,
    out_root: Path,
    max_frames: int | None,
    warmup_barrier: threading.Barrier | None = None,
):
    cam_dir = out_root / serial
    rgb_dir = cam_dir / "rgb"
    depth_dir = cam_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    ts_path = cam_dir / "timestamps.csv"
    ts_fp = open(ts_path, "w", newline="")
    ts_writer = csv.writer(ts_fp)
    ts_writer.writerow(["frame_idx", "ts_unix", "ts_device_ms"])

    n_written = 0
    try:
        # Warmup so auto-exposure settles.
        for _ in range(WARMUP_FRAMES):
            if _stop.is_set():
                return
            pipeline.wait_for_frames()
        print(f"[{serial} ({arm})] warmup done, recording @ {fps} Hz")

        # Synchronize with other cameras + main: wait until ALL cameras have
        # finished warmup before any of them starts writing frames. Main is
        # also at the barrier; it starts the --duration timer right after.
        if warmup_barrier is not None:
            try:
                warmup_barrier.wait(timeout=30.0)
            except threading.BrokenBarrierError:
                print(f"[{serial}] warmup barrier broken; proceeding anyway")

        t_start = time.time()
        while not _stop.is_set():
            try:
                frames = pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError as e:
                print(f"[{serial}] wait_for_frames failed: {e}")
                continue
            frames = align.process(frames)
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                continue

            color_img = np.asanyarray(color.get_data())
            depth_raw = np.asanyarray(depth.get_data())
            depth_mm_f = depth_raw.astype(np.float32) * depth_scale * 1000.0
            depth_mm = np.clip(depth_mm_f, 0, np.iinfo(np.uint16).max).astype(
                np.uint16
            )

            idx = n_written
            cv2.imwrite(str(rgb_dir / f"{idx:06d}.png"), color_img)
            cv2.imwrite(str(depth_dir / f"{idx:06d}.png"), depth_mm)
            ts_writer.writerow(
                [idx, f"{time.time():.6f}", f"{color.get_timestamp():.3f}"]
            )
            n_written += 1

            if max_frames is not None and n_written >= max_frames:
                print(f"[{serial}] reached max_frames={max_frames}")
                break

            if n_written % fps == 0:
                elapsed = time.time() - t_start
                print(
                    f"[{serial}] {n_written} frames written"
                    f" ({n_written / max(elapsed, 1e-6):.1f} fps)"
                )
    finally:
        pipeline.stop()
        ts_fp.close()
        print(f"[{serial}] stopped. {n_written} frames -> {cam_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extrinsics",
        type=str,
        default=DEFAULT_EXTRINSICS,
        help="Path to camera_extrinsics.json",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUT,
        help="Output root directory. Each serial gets a subfolder.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after this many seconds (in addition to Ctrl+C).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop each camera after this many frames.",
    )
    parser.add_argument(
        "--no-park",
        action="store_true",
        help="Skip the robot-arm parking step (use if the robot is off).",
    )
    parser.add_argument(
        "--countdown",
        type=int,
        default=3,
        help="Seconds of '3..2..1..GO' countdown after parking, before recording "
             "threads start. Gives you time to get into position. Set to 0 to skip.",
    )
    args = parser.parse_args()

    cams = _load_camera_specs(args.extrinsics)
    if not cams:
        raise SystemExit(f"No cameras found in {args.extrinsics}")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # Copy the extrinsics alongside the recording so it's self-contained.
    with open(args.extrinsics, "r") as f:
        extr_raw = f.read()
    (out_root / "camera_extrinsics.json").write_text(extr_raw)

    print(f"[main] Recording {len(cams)} cameras -> {out_root}")
    for serial, arm, w, h, in cams:
        print(f"  - {serial} ({arm})  {w}x{h}")

    signal.signal(signal.SIGINT, _handle_sigint)

    robot = None
    if not args.no_park:
        robot = park_robot_arms()
    else:
        print("[main] --no-park set; skipping arm parking.")

    if args.countdown > 0:
        print(f"[countdown] starting recording in {args.countdown} s — get ready!")
        for s in range(args.countdown, 0, -1):
            print(f"  {s}...", flush=True)
            time.sleep(1.0)
        print(f"  GO — recording {args.duration if args.duration else 'until Ctrl-C'}.\n", flush=True)

    try:
        # +1 for main; threads wait at this barrier after warmup completes.
        warmup_barrier = threading.Barrier(len(cams) + 1)
        threads = []
        for serial, arm, w, h in cams:
            t = threading.Thread(
                target=_record_one,
                args=(serial, arm, w, h, args.fps, out_root, args.max_frames,
                      warmup_barrier),
                name=f"rec-{serial}",
                daemon=True,
            )
            t.start()
            threads.append(t)

        # Wait for ALL cameras to finish warmup before starting the duration
        # timer — RealSense auto-exposure settling takes ~2-3 s, and prior to
        # this barrier the timer would eat into the requested recording time.
        try:
            warmup_barrier.wait(timeout=30.0)
            print(f"[main] all cameras warmed up; starting --duration timer.")
        except threading.BrokenBarrierError:
            print("[main] warmup barrier broken; starting timer anyway")

        if args.duration is not None:
            deadline = time.time() + args.duration
            while time.time() < deadline and not _stop.is_set():
                if not any(t.is_alive() for t in threads):
                    break
                time.sleep(0.1)
            _stop.set()

        for t in threads:
            t.join()
        print("[main] All cameras stopped.")
    finally:
        if robot is not None and getattr(robot, "is_connected", False):
            try:
                robot.disconnect()
            except Exception as exc:
                print(f"[main] robot.disconnect() raised: {exc}")


if __name__ == "__main__":
    main()
