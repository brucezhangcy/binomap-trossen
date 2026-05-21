"""Replay a BiNoMaP joint trajectory on the real Trossen ALOHA bimanual arms.

Loads `joint_L` and `joint_R` (per-arm, 6 DoF, 30 Hz) from a
`trossen_replay_ik_log.npz` produced by `replay_trossen_ik.py` and streams
them to the real follower arms via the LeRobot Trossen driver. The
intent is to reproduce on hardware the motion shown in
`trossen_replay_ik_pos.mp4` — same joint configurations, same time grid.

Why the IK log: those joints were already solved against the Trossen URDF
and validated in MuJoCo. Re-solving here would just duplicate work and risk
diverging from the sim render.

Layout mapping:
    sim log → real lerobot 14-D action
    joint_L[k] (6,)       → action[0:6]   (left arm joints 0..5)
    gripper                → action[6]     (closed, 0.0 m)
    joint_R[k] (6,)       → action[7:13]  (right arm joints 0..5)
    gripper                → action[13]    (closed, 0.0 m)

Smoothing/timing follows `replay_state.py` exactly: per-step
MIN_TIME_TO_MOVE is set equal to the inter-action sleep so each
trapezoidal interpolation completes just as the next command arrives,
and successive commands blend.

Usage::

    # Dry run — print what would be sent, no hardware
    python replay_real.py --dry-run

    # Half-speed first hardware run (default)
    python replay_real.py

    # Full speed (matches sim mp4 fps)
    python replay_real.py --speed 1.0

    # Different bundle
    python replay_real.py --log outputs/recordings/wrist/trossen_replay_ik_log.npz
"""

import argparse
import math
import pathlib
import time

import numpy as np
import torch

from lerobot.common.robot_devices.robots.configs import TrossenAIStationaryRobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config


DEFAULT_LOG = pathlib.Path(__file__).parent / "outputs" / "recordings_1" / "wrist" / "trossen_replay_ik_log.npz"

# Per-arm 7-DoF home (6 joints + gripper). Matches the lab's standard home in
# move_robot.py / replay_state.py — compact pose that keeps arms inside camera FoV.
HOME_POSE_PER_ARM = [0.0, 0.2618, 0.2618, 0.0, 0.0, 0.0, 0.0]

# Per-joint Trossen WX AI hardware limits (rad), per arm. Same numbers used by
# inference_dp3.py and replay_state.py. We clamp every commanded pose to these
# before sending, even though the sim IK already respected them — defense in
# depth in case a future log npz drifts.
_HW_LIMITS = [
    (-math.pi,     math.pi),
    (0.0,          math.pi),
    (0.0,          2.3562),
    (-math.pi / 2, math.pi / 2),
    (-math.pi / 2, math.pi / 2),
    (-math.pi,     math.pi),
    (-0.001,       0.04),    # gripper (linear, meters)
]
_HW_MINS = torch.tensor([lo for lo, _ in _HW_LIMITS * 2], dtype=torch.float32)
_HW_MAXS = torch.tensor([hi for _, hi in _HW_LIMITS * 2], dtype=torch.float32)

FIRST_STEP_ABORT_RAD = 2.5   # bail before sending if a commanded step jumps further than this


def build_actions_from_log(joint_L: np.ndarray, joint_R: np.ndarray, gripper: float = 0.0) -> torch.Tensor:
    """Stitch (N,6) per-arm joints into the lerobot (N,14) action layout."""
    n = joint_L.shape[0]
    assert joint_R.shape == joint_L.shape, "L/R joint trajectories must have the same shape"
    grip = np.full((n, 1), gripper, dtype=np.float32)
    actions = np.concatenate(
        [joint_L.astype(np.float32), grip, joint_R.astype(np.float32), grip],
        axis=1,
    )  # (N, 14)
    return torch.from_numpy(actions)


def clamp_to_hw_limits(action: torch.Tensor) -> torch.Tensor:
    return torch.clamp(action, min=_HW_MINS.to(action.device), max=_HW_MAXS.to(action.device))


def slow_move_to(robot, target_14: torch.Tensor, move_time_s: float, extra_settle_s: float) -> None:
    """Send a single pose with a temporarily slowed driver, then settle.

    Same shape as `replay_state.slow_move_to`. The driver-level
    `MIN_TIME_TO_MOVE` controls the trapezoidal interpolation duration that
    the firmware uses to reach the goal.
    """
    saved = {}
    for name, arm in robot.follower_arms.items():
        saved[name] = arm.MIN_TIME_TO_MOVE
        arm.MIN_TIME_TO_MOVE = move_time_s
    try:
        robot.send_action(target_14)
        time.sleep(move_time_s + extra_settle_s)
    finally:
        for name, arm in robot.follower_arms.items():
            arm.MIN_TIME_TO_MOVE = saved[name]


def emergency_stop(robot, home_14: torch.Tensor, home_move_time_s: float) -> None:
    """Ctrl-C handler: freeze each arm at its current pose, settle 2 s, slow-move HOME.

    Sequence:
      1. Drop each follower into velocity mode and send zero velocities → the
         firmware brakes the current trapezoid within a few ms and holds the
         current joint angles. This is the "instant freeze" step.
      2. Switch each follower back to position mode (so subsequent send_action
         calls behave normally).
      3. Sleep 2 s so the operator can inspect the frozen state.
      4. Slow-move back to HOME at the same pace as the initial HOME move.

    The script's outer `finally: robot.disconnect()` runs after this and will
    do its own home→sleep transition; calling slow-move-HOME here means the
    disconnect's home-move is a no-op (already there).

    A second Ctrl-C during this sequence propagates as KeyboardInterrupt and
    skips the slow move to HOME — caller should let it fall through to
    disconnect.
    """
    import trossen_arm as trossen

    print("\n!! EMERGENCY STOP — zeroing velocity, freezing at current pose.")
    for name, arm in robot.follower_arms.items():
        try:
            n = arm.driver.get_num_joints()
            arm.driver.set_all_modes(trossen.Mode.velocity)
            arm.driver.set_all_velocities([0.0] * n, 0.0, False)
            arm.driver.set_all_modes(trossen.Mode.position)
        except Exception as e:
            print(f"  [{name}] velocity-zero failed: {e}")
    print("frozen. 2.0 s pause for inspection, then slow move to HOME.")
    print("(press Ctrl-C again to skip the HOME move and disconnect immediately)")
    time.sleep(2.0)
    slow_move_to(robot, home_14, home_move_time_s, 0.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=pathlib.Path, default=DEFAULT_LOG,
                    help=f"Path to trossen_replay_ik_log.npz (default: {DEFAULT_LOG}).")
    ap.add_argument("--speed", type=float, default=0.5,
                    help="Playback speed multiplier vs the source 30 Hz grid. "
                         "0.5 = half-speed (period 66 ms). Recommended for first runs.")
    ap.add_argument("--source-fps", type=float, default=30.0,
                    help="Nominal frame rate of the joint log. Trajectories from "
                         "replay_trossen_ik.py are on a 30 Hz uniform grid.")
    ap.add_argument("--max-step-delta", type=float, default=0.05,
                    help="Per-joint cap on |action - current_state| each step "
                         "passed to lerobot's max_relative_target clamp (rad). "
                         "0.05 rad ≈ 2.9° / step; further tightens the trapezoid "
                         "limit imposed by MIN_TIME_TO_MOVE.")
    ap.add_argument("--home-move-time-s", type=float, default=3.0,
                    help="Interpolation time for the initial slow move to HOME.")
    ap.add_argument("--home-settle-s", type=float, default=1.0,
                    help="Sleep at HOME after the move finishes, before warm-up to start.")
    ap.add_argument("--start-move-time-s", type=float, default=3.0,
                    help="Interpolation time for the slow move from HOME to the "
                         "trajectory's first frame (joint_L[0], joint_R[0]).")
    ap.add_argument("--start-settle-s", type=float, default=1.0,
                    help="Sleep at the trajectory start pose before streaming begins.")
    ap.add_argument("--gripper", type=float, default=0.0,
                    help="Gripper command (meters) held constant for the whole replay. "
                         "0.0 = closed (BiNoMaP is non-prehensile).")
    ap.add_argument("--start", type=int, default=None,
                    help="First frame index to replay. If unset, defaults to "
                         "max(first_valid_L, first_valid_R) from the sibling "
                         "trajectory_smoothed.npz so we skip the leading frames "
                         "that replay_trossen_ik.py back-filled (which would "
                         "otherwise create a step-jump when streaming starts).")
    ap.add_argument("--end", type=int, default=None, help="One past last frame index (default: all).")
    ap.add_argument("--stride", type=int, default=1, help="Replay every Nth recorded frame.")
    ap.add_argument("--smoothed-npz", type=pathlib.Path, default=None,
                    help="Path to trajectory_smoothed.npz (carries valid_L/valid_R masks). "
                         "Defaults to <log_dir>/trajectory_smoothed.npz. Only consulted "
                         "when --start is unset.")
    ap.add_argument("--no-skip-leading-invalid", dest="skip_leading_invalid",
                    action="store_false", default=True,
                    help="Don't auto-skip leading back-filled frames. Use this if you "
                         "want to replay the full N frames including the leading plateau.")
    ap.add_argument("--record", action="store_true",
                    help="Capture RGB frames from the Intel RealSense cameras during "
                         "streaming and write one mp4 per camera. Requires the cameras "
                         "to be plugged in; will block robot.connect() if any are missing.")
    ap.add_argument("--record-dir", type=pathlib.Path, default=None,
                    help="Where to write the per-camera mp4s. "
                         "Defaults to <log_dir>/real_run_<YYYYMMDD_HHMMSS>/.")
    ap.add_argument("--cameras", nargs="+", default=None,
                    help="Subset of camera names to record (e.g. cam_high cam_low). "
                         "Default: record every camera in the connected robot config.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Don't connect to hardware; just print the resolved 14-D actions + timing.")
    args = ap.parse_args()

    if not args.log.exists():
        raise FileNotFoundError(args.log)
    if args.speed <= 0:
        raise ValueError("--speed must be positive")

    bundle = np.load(args.log)
    joint_L = bundle["joint_L"]  # (N, 6)
    joint_R = bundle["joint_R"]  # (N, 6)
    N_total = joint_L.shape[0]

    # Resolve --start. When unset, peek at trajectory_smoothed.npz's valid masks
    # and skip the leading frames that replay_trossen_ik.py back-filled to equal
    # the first jointly-valid IK output. Those leading frames are followed by a
    # step-jump at the back-fill boundary (e.g. on recordings_1/wrist: R is
    # invalid for the first 47 frames → joint_1 jumps 0.197 rad on frame 47→48).
    # Starting at the first jointly-valid frame turns that step-jump into the
    # tail end of the slow_move_to() warm-up.
    start = args.start
    if start is None:
        if not args.skip_leading_invalid:
            start = 0
        else:
            smoothed = args.smoothed_npz or (args.log.parent / "trajectory_smoothed.npz")
            if smoothed.exists():
                src = np.load(smoothed)
                first_L = int(np.where(src["valid_L"])[0][0]) if src["valid_L"].any() else 0
                first_R = int(np.where(src["valid_R"])[0][0]) if src["valid_R"].any() else 0
                start = max(first_L, first_R)
                print(f"Auto-skip leading invalid: smoothed.valid first L={first_L} R={first_R} "
                      f"→ --start {start} (skipping {start} back-filled frames). "
                      f"Pass --start 0 or --no-skip-leading-invalid to disable.")
            else:
                start = 0
                print(f"WARNING: --start unset and {smoothed} not found; defaulting to --start 0. "
                      f"Leading back-filled frames may cause step-jumps.")

    end = args.end if args.end is not None else N_total
    joint_L = joint_L[start:end:args.stride]
    joint_R = joint_R[start:end:args.stride]
    T = joint_L.shape[0]
    # Carry the resolved start forward so the summary prints the actual value used.
    args.start = start

    actions = build_actions_from_log(joint_L, joint_R, gripper=args.gripper)
    actions = clamp_to_hw_limits(actions)

    period = 1.0 / (args.source_fps * args.speed)
    wallclock_s = T * period
    print(f"Loaded {args.log.name}: {T} frames "
          f"(start={args.start}, stride={args.stride}, fps={args.source_fps:.1f}, speed={args.speed:.2f}x)")
    print(f"Period {period*1000:.1f} ms/step  → wall-clock ≈ {wallclock_s:.1f} s")
    print(f"Action layout: 14-D = [L0..L5 | gripper={args.gripper:.3f} | R0..R5 | gripper={args.gripper:.3f}]")

    if args.dry_run:
        print("\n--- dry run: first 3 actions ---")
        for t in range(min(T, 3)):
            print(f"  action[{t}] = {actions[t].numpy().round(4)}")
        if T > 3:
            print(f"  ... ({T - 3} more frames)")
        max_step = (actions[1:] - actions[:-1]).abs().max(dim=0).values.numpy().round(4) if T > 1 else None
        if max_step is not None:
            print(f"Per-joint max frame-to-frame delta (rad): {max_step}")
        return

    # ── Hardware path ────────────────────────────────────────────────────────
    # min_time_to_move_multiplier sets the driver's MIN_TIME_TO_MOVE via
    # MIN_TIME_TO_MOVE = multiplier / fps. We override per-arm to period below;
    # the constructor value is just a placeholder.
    robot_cfg = TrossenAIStationaryRobotConfig(
        camera_interface="intel_realsense",
        max_relative_target=args.max_step_delta,
        min_time_to_move_multiplier=1.0,
    )
    # When NOT recording, clear cameras so robot.connect() doesn't require a
    # working RealSense rig. When recording, keep the default cam_high / cam_low
    # (optionally filter to a subset via --cameras).
    if not args.record:
        robot_cfg.cameras = {}
    elif args.cameras:
        missing = [c for c in args.cameras if c not in robot_cfg.cameras]
        if missing:
            raise ValueError(
                f"--cameras references unknown names {missing}. Available in default "
                f"Stationary config: {list(robot_cfg.cameras.keys())}")
        robot_cfg.cameras = {k: v for k, v in robot_cfg.cameras.items() if k in args.cameras}
    robot = make_robot_from_config(robot_cfg)
    robot.leader_arms = {}
    if not args.record:
        robot.cameras = {}
    robot.connect()

    # Set up mp4 writers AFTER connect so we use the actual list of cameras the
    # driver successfully initialized (one camera failing to open shouldn't
    # silently drop it from the record set).
    video_writers = {}
    record_dir = None
    if args.record:
        import imageio.v2 as imageio
        stamp = time.strftime("%Y%m%d_%H%M%S")
        record_dir = args.record_dir or (args.log.parent / f"real_run_{stamp}")
        record_dir.mkdir(parents=True, exist_ok=True)
        # Use the wall-clock playback fps so the mp4 timeline matches reality.
        playback_fps = max(1, int(round(1.0 / period)))
        cam_names = list(robot.cameras.keys())
        if not cam_names:
            raise RuntimeError("--record set but no cameras present in robot after connect.")
        for cam_name in cam_names:
            mp4_path = record_dir / f"{stamp}_real_{cam_name}.mp4"
            video_writers[cam_name] = imageio.get_writer(str(mp4_path), fps=playback_fps, codec="libx264")
            print(f"Recording {cam_name} → {mp4_path}")

    try:
        n_arms = len(robot.follower_arms)
        if actions.shape[-1] != n_arms * 7:
            raise RuntimeError(
                f"action dim {actions.shape[-1]} != follower DOF {n_arms * 7}; "
                f"this script assumes a 2-arm Trossen stationary setup.")

        home = clamp_to_hw_limits(torch.tensor(HOME_POSE_PER_ARM * n_arms, dtype=torch.float32))
        print(f"Slow move HOME (move_time={args.home_move_time_s}s, settle={args.home_settle_s}s).")
        slow_move_to(robot, home, args.home_move_time_s, args.home_settle_s)

        # Pre-stream check: refuse if the very first commanded pose is implausibly
        # far from the arms' current state. Mirrors the safety pattern from
        # inference_dp3.py / 3d-diffusion-policy/replay_actions.py.
        obs = robot.capture_observation()
        cur = obs["observation.state"]
        first = actions[0]
        delta = (first - cur).abs()
        worst_joint = int(delta.argmax().item())
        worst = float(delta[worst_joint].item())
        print(f"start pose vs current: max |Δ| = {worst:.3f} rad on joint {worst_joint}")
        if worst > FIRST_STEP_ABORT_RAD:
            print(f"REFUSING: start pose is {worst:.3f} rad away (> {FIRST_STEP_ABORT_RAD}). "
                  f"Inspect the trajectory or HOME_POSE_PER_ARM and re-run.")
            return

        print(f"Slow move HOME → trajectory start (move_time={args.start_move_time_s}s, "
              f"settle={args.start_settle_s}s).")
        slow_move_to(robot, first, args.start_move_time_s, args.start_settle_s)

        # Lock in the streaming MIN_TIME_TO_MOVE == period so consecutive
        # trapezoids blend instead of decelerating to a halt each tick.
        for arm in robot.follower_arms.values():
            arm.MIN_TIME_TO_MOVE = period

        print(f"Streaming {T} poses, {period*1000:.1f} ms each "
              f"(press Ctrl-C to abort — arms will freeze, then slow-move to HOME).")
        report_every = max(1, int(round(1.0 / period)))   # ~1 print per second of wall-clock
        try:
            for t in range(T):
                step_start = time.perf_counter()
                if video_writers:
                    # capture_observation reads joint state AND triggers a frame
                    # grab from each enabled camera. Writing the RGBs happens
                    # after send_action so the actuator command isn't waiting on
                    # disk I/O.
                    obs = robot.capture_observation()
                robot.send_action(actions[t])
                if video_writers:
                    for cam_name, writer in video_writers.items():
                        rgb = obs[f"observation.images.{cam_name}"]
                        if hasattr(rgb, "numpy"):
                            rgb = rgb.numpy()
                        if rgb.ndim == 3 and rgb.shape[0] == 3:
                            rgb = rgb.transpose(1, 2, 0)
                        if rgb.dtype != np.uint8:
                            rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
                        writer.append_data(rgb)
                elapsed = time.perf_counter() - step_start
                time.sleep(max(0.0, period - elapsed))
                if t % report_every == 0:
                    print(f"  [t={t:04d}/{T}] sent.", flush=True)

            print("Replay complete.")
            # Return arms to HOME so the next person doesn't inherit the final pose.
            print("Slow move back to HOME.")
            slow_move_to(robot, home, args.home_move_time_s, 0.5)
        except KeyboardInterrupt:
            try:
                emergency_stop(robot, home, args.home_move_time_s)
            except KeyboardInterrupt:
                print("\nSecond Ctrl-C — skipping HOME return, going straight to disconnect.")

    finally:
        for cam_name, writer in video_writers.items():
            try:
                writer.close()
                print(f"Closed {cam_name} mp4.")
            except Exception as e:
                print(f"  failed to close {cam_name} writer: {e}")
        if record_dir is not None:
            print(f"Recording saved to {record_dir}")
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
