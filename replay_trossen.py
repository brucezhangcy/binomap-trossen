"""
BiNoMaP Trossen MuJoCo Replay (validation in simulation).

Loads a trajectory_smoothed.npz bundle, applies a frame transform from the
demonstrator's world frame (set by hand-eye calibration) to the Trossen
ALOHA MuJoCo world frame, and replays the per-arm 6-DoF pose sequence via
the scene_mocap.xml mocap-based EE control (mocap bodies weld-constrained
to the last arm link).

Trossen scene_mocap.xml reference points (extracted from the XML):
  mocap_left  rest:  (-0.019982,  0.212613, 0.202586)  m
  mocap_right rest:  (-0.019982, -0.212613, 0.202586)  m
  tabletop top z  :   0.02  m   (matches our world's tabletop z)
  arm base spacing:   0.915 m along y axis

Frame conventions
-----------------
Our world (camera_extrinsics.json):
  z up, tabletop ≈ 0.02 m, quaternion (x, y, z, w)  [scipy convention]
Trossen MuJoCo world:
  z up, floor at z=0, tabletop top at z=0.02 m, quaternion (w, x, y, z)
  [MuJoCo convention]

v1 alignment = translation-only (no rotation), so the bimanual midpoint of
our trajectory lands at the Trossen workspace center while preserving the
hand-above-table delta. Add a yaw correction in v2 if the visible motion
heads off-axis.

Usage
-----
  python replay_trossen.py \\
      --in_npz outputs/recordings_1/wrist/trajectory_smoothed.npz \\
      --scene_xml trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/scene_mocap.xml \\
      --speed 0.25 [--headless] [--record_mp4 out.mp4]
"""
import argparse
import os
import time
from pathlib import Path

# Headless OpenGL for off-screen rendering on servers without a display.
# Must be set BEFORE importing mujoco. EGL is preferred (GPU); osmesa is software fallback.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
import mujoco.viewer  # imported at module scope so conditional use doesn't shadow `mujoco`

# Trossen scene reference points
TROSSEN_MOCAP_LEFT_REST  = np.array([-0.019982,  0.212613, 0.202586])
TROSSEN_MOCAP_RIGHT_REST = np.array([-0.019982, -0.212613, 0.202586])
TROSSEN_WORKSPACE_CENTER = 0.5 * (TROSSEN_MOCAP_LEFT_REST + TROSSEN_MOCAP_RIGHT_REST)
TROSSEN_TABLE_Z = 0.02

# Safety bounds (Trossen world frame; loose enough not to false-alarm)
WS_XMIN, WS_XMAX = -0.5, 0.5
WS_YMIN, WS_YMAX = -0.6, 0.6
WS_ZMIN, WS_ZMAX = 0.0, 0.8

V_MAX_M_PER_S = 3.0  # generous human-hand cap


def quat_xyzw_to_wxyz(q):
    """scipy (x,y,z,w) → MuJoCo (w,x,y,z). Same orientation, different ordering."""
    return np.array([q[3], q[0], q[1], q[2]])


def compute_frame_alignment(p_L, p_R, valid_L, valid_R, our_table_z=0.02):
    """Translate-only alignment (v1).

    Maps the bimanual centroid of our trajectory to the Trossen workspace center.
    Preserves the hand-above-table z delta (tabletop is at z=0.02 in both frames).
    """
    cL = p_L[valid_L].mean(axis=0)
    cR = p_R[valid_R].mean(axis=0)
    ours_mid = 0.5 * (cL + cR)
    target = TROSSEN_WORKSPACE_CENTER.copy()
    target[2] = TROSSEN_TABLE_Z + (ours_mid[2] - our_table_z)
    offset = target - ours_mid
    return offset, ours_mid, target


def workspace_violation(p, valid):
    """Return count of valid samples falling outside the workspace AABB."""
    if not valid.any():
        return 0
    p_v = p[valid]
    bad = ((p_v[:, 0] < WS_XMIN) | (p_v[:, 0] > WS_XMAX) |
           (p_v[:, 1] < WS_YMIN) | (p_v[:, 1] > WS_YMAX) |
           (p_v[:, 2] < WS_ZMIN) | (p_v[:, 2] > WS_ZMAX))
    return int(bad.sum())


def velocity_check(p, valid, ts_ms, v_max=V_MAX_M_PER_S):
    """Count consecutive-frame velocity violations (instantaneous m/s > v_max)."""
    n = 0
    last_k = None
    for k in np.where(valid)[0]:
        if last_k is not None:
            dt = (ts_ms[k] - ts_ms[last_k]) / 1000.0
            if dt > 0:
                v = np.linalg.norm(p[k] - p[last_k]) / dt
                if v > v_max:
                    n += 1
        last_k = k
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", required=True)
    ap.add_argument("--scene_xml",
                    default="/data/bruce/BiNoMaP/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/scene_mocap.xml")
    ap.add_argument("--speed", type=float, default=0.25,
                    help="playback speed multiplier (1.0 = real-time, 0.25 = quarter speed)")
    ap.add_argument("--fps", type=float, default=30.0, help="trajectory grid fps")
    ap.add_argument("--headless", action="store_true", help="no viewer")
    ap.add_argument("--record_mp4", type=str, default=None,
                    help="write an mp4 of the viewer using mujoco.Renderer (off-screen)")
    ap.add_argument("--mp4_fps", type=float, default=30.0)
    ap.add_argument("--our_table_z", type=float, default=0.02)
    ap.add_argument("--settle_steps", type=int, default=200,
                    help="physics steps to settle the arms into the initial pose before replay")
    ap.add_argument("--orientation_mode", choices=["initial", "from_trajectory"], default="from_trajectory",
                    help="initial = keep mocap_quat at scene-load gripper rest pose (position-only); "
                         "from_trajectory = use the smoothed R_L/R_R (default; gives IK a real orientation goal)")
    ap.add_argument("--warmup_seconds", type=float, default=1.0,
                    help="seconds to slowly interpolate mocap from rest pose to trajectory frame 0, "
                         "giving the IK time to converge before replay starts")
    ap.add_argument("--render_w", type=int, default=640)
    ap.add_argument("--render_h", type=int, default=480)
    args = ap.parse_args()

    bundle = np.load(args.in_npz)
    ts_ms = bundle["ts_ms"]
    p_L = bundle["p_L"].copy()
    q_L = bundle["q_L"]
    valid_L = bundle["valid_L"]
    p_R = bundle["p_R"].copy()
    q_R = bundle["q_R"]
    valid_R = bundle["valid_R"]
    N = len(ts_ms)

    print(f"=== BiNoMaP Trossen MuJoCo Replay ===")
    print(f"input: {args.in_npz}  N={N} frames")
    print(f"  valid_L: {int(valid_L.sum())}/{N}   valid_R: {int(valid_R.sum())}/{N}")
    print(f"  p_L (our world):  x=[{p_L[valid_L,0].min():.3f},{p_L[valid_L,0].max():.3f}] "
          f"y=[{p_L[valid_L,1].min():.3f},{p_L[valid_L,1].max():.3f}] "
          f"z=[{p_L[valid_L,2].min():.3f},{p_L[valid_L,2].max():.3f}]")
    print(f"  p_R (our world):  x=[{p_R[valid_R,0].min():.3f},{p_R[valid_R,0].max():.3f}] "
          f"y=[{p_R[valid_R,1].min():.3f},{p_R[valid_R,1].max():.3f}] "
          f"z=[{p_R[valid_R,2].min():.3f},{p_R[valid_R,2].max():.3f}]")

    # Frame alignment
    offset, ours_mid, target = compute_frame_alignment(
        p_L, p_R, valid_L, valid_R, our_table_z=args.our_table_z)
    print(f"\nframe alignment (translate-only):")
    print(f"  our midpoint:    {ours_mid.round(3)}")
    print(f"  trossen target:  {target.round(3)}")
    print(f"  offset applied:  {offset.round(3)}")

    p_L_t = p_L + offset[None, :]
    p_R_t = p_R + offset[None, :]

    # Quaternion conversion
    q_L_mj = np.stack([quat_xyzw_to_wxyz(q_L[i]) for i in range(N)])
    q_R_mj = np.stack([quat_xyzw_to_wxyz(q_R[i]) for i in range(N)])

    # Safety pre-checks
    bad_L = workspace_violation(p_L_t, valid_L)
    bad_R = workspace_violation(p_R_t, valid_R)
    velL = velocity_check(p_L_t, valid_L, ts_ms)
    velR = velocity_check(p_R_t, valid_R, ts_ms)
    print(f"\nsafety pre-checks:")
    print(f"  L arm: workspace_violations={bad_L}  velocity_violations={velL}")
    print(f"  R arm: workspace_violations={bad_R}  velocity_violations={velR}")

    # Load model
    print(f"\nloading {args.scene_xml}")
    model = mujoco.MjModel.from_xml_path(args.scene_xml)
    data = mujoco.MjData(model)
    mocap_L_id = int(model.body("mocap_left").mocapid[0])
    mocap_R_id = int(model.body("mocap_right").mocapid[0])
    ee_L_body_id = model.body("follower_left_link_6").id
    ee_R_body_id = model.body("follower_right_link_6").id
    print(f"  mocap_left mocapid={mocap_L_id}  mocap_right mocapid={mocap_R_id}")
    print(f"  ee left body id={ee_L_body_id}  ee right body id={ee_R_body_id}")

    # Load the home keyframe and run a few unbiased steps to find the EE's
    # natural rest orientation. The XML's default mocap_quat=(1,0,0,0) does NOT
    # match link_6's home orientation, so a pure "initial" mode using identity
    # would force the IK to contort the arm. Using link_6's actual home xquat
    # eliminates the orientation-mismatch error.
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    # Park mocaps over each EE so the weld constraint isn't fighting reality.
    data.mocap_pos[mocap_L_id] = data.xpos[ee_L_body_id].copy()
    data.mocap_quat[mocap_L_id] = data.xquat[ee_L_body_id].copy()
    data.mocap_pos[mocap_R_id] = data.xpos[ee_R_body_id].copy()
    data.mocap_quat[mocap_R_id] = data.xquat[ee_R_body_id].copy()
    for _ in range(50):
        mujoco.mj_step(model, data)
    rest_qL = data.xquat[ee_L_body_id].copy()
    rest_qR = data.xquat[ee_R_body_id].copy()
    rest_pL = data.xpos[ee_L_body_id].copy()
    rest_pR = data.xpos[ee_R_body_id].copy()
    print(f"  EE rest pose L: pos={rest_pL.round(3)} quat={rest_qL.round(3)}")
    print(f"  EE rest pose R: pos={rest_pR.round(3)} quat={rest_qR.round(3)}")
    print(f"  orientation_mode: {args.orientation_mode}")

    # Determine the targets to lerp toward
    first_L = int(np.where(valid_L)[0][0])
    first_R = int(np.where(valid_R)[0][0])
    target_pL0 = p_L_t[first_L]
    target_pR0 = p_R_t[first_R]
    target_qL0 = q_L_mj[first_L] if args.orientation_mode == "from_trajectory" else rest_qL
    target_qR0 = q_R_mj[first_R] if args.orientation_mode == "from_trajectory" else rest_qR

    # Warm-up: lerp slowly from rest pose to the trajectory's first frame, giving
    # the IK time to converge. Otherwise the weld constraint sees a sudden ~30cm
    # jump on step 1 and the arm "snaps" toward an unreachable pose.
    warmup_steps = max(int(args.warmup_seconds / model.opt.timestep), 1)
    print(f"  warmup: {warmup_steps} steps ({args.warmup_seconds:.2f}s) lerping rest → frame 0 target")
    for s in range(warmup_steps):
        alpha = (s + 1) / warmup_steps
        data.mocap_pos[mocap_L_id] = (1 - alpha) * rest_pL + alpha * target_pL0
        data.mocap_pos[mocap_R_id] = (1 - alpha) * rest_pR + alpha * target_pR0
        # quaternion lerp via SLERP-ish (here just simple lerp + renormalize; quats are close enough)
        qL = (1 - alpha) * rest_qL + alpha * target_qL0
        qR = (1 - alpha) * rest_qR + alpha * target_qR0
        data.mocap_quat[mocap_L_id] = qL / max(np.linalg.norm(qL), 1e-9)
        data.mocap_quat[mocap_R_id] = qR / max(np.linalg.norm(qR), 1e-9)
        mujoco.mj_step(model, data)
    # snap exact final targets
    data.mocap_pos[mocap_L_id] = target_pL0
    data.mocap_pos[mocap_R_id] = target_pR0
    data.mocap_quat[mocap_L_id] = target_qL0
    data.mocap_quat[mocap_R_id] = target_qR0
    ee_L0 = data.xpos[ee_L_body_id].copy()
    ee_R0 = data.xpos[ee_R_body_id].copy()
    print(f"  after warmup: ee_L={ee_L0.round(3)}  target={target_pL0.round(3)}  err={np.linalg.norm(ee_L0-target_pL0)*1000:.1f}mm")
    print(f"  after warmup: ee_R={ee_R0.round(3)}  target={target_pR0.round(3)}  err={np.linalg.norm(ee_R0-target_pR0)*1000:.1f}mm")

    # Replay
    timestep = model.opt.timestep
    substeps_per_frame = max(1, int(round(1.0 / args.fps / timestep)))
    sleep_per_frame = 1.0 / (args.fps * max(args.speed, 1e-6))
    print(f"\nreplay: timestep={timestep:.4f}s  substeps/frame={substeps_per_frame}  "
          f"speed={args.speed}x  realtime_dt/frame={sleep_per_frame*1000:.1f}ms")

    track_err_L = []
    track_err_R = []
    log_target_L = []
    log_target_R = []
    log_actual_L = []
    log_actual_R = []
    last_pL = data.mocap_pos[mocap_L_id].copy()
    last_qL = data.mocap_quat[mocap_L_id].copy()
    last_pR = data.mocap_pos[mocap_R_id].copy()
    last_qR = data.mocap_quat[mocap_R_id].copy()

    viewer = None
    if not args.headless and not args.record_mp4:
        viewer = mujoco.viewer.launch_passive(model, data)

    renderer = None
    mp4_writer = None
    if args.record_mp4:
        import imageio
        renderer = mujoco.Renderer(model, height=args.render_h, width=args.render_w)
        mp4_writer = imageio.get_writer(args.record_mp4, fps=args.mp4_fps, codec="libx264")

    t_start = time.time()
    try:
        for k in range(N):
            if valid_L[k]:
                last_pL[:] = p_L_t[k]
                if args.orientation_mode == "from_trajectory":
                    last_qL[:] = q_L_mj[k]
            if valid_R[k]:
                last_pR[:] = p_R_t[k]
                if args.orientation_mode == "from_trajectory":
                    last_qR[:] = q_R_mj[k]
            data.mocap_pos[mocap_L_id] = last_pL
            data.mocap_quat[mocap_L_id] = last_qL
            data.mocap_pos[mocap_R_id] = last_pR
            data.mocap_quat[mocap_R_id] = last_qR

            for _ in range(substeps_per_frame):
                mujoco.mj_step(model, data)

            ee_L_pos = data.xpos[ee_L_body_id]
            ee_R_pos = data.xpos[ee_R_body_id]
            track_err_L.append(float(np.linalg.norm(ee_L_pos - last_pL)))
            track_err_R.append(float(np.linalg.norm(ee_R_pos - last_pR)))
            log_target_L.append(last_pL.copy())
            log_target_R.append(last_pR.copy())
            log_actual_L.append(ee_L_pos.copy())
            log_actual_R.append(ee_R_pos.copy())

            if viewer is not None:
                viewer.sync()
                elapsed = time.time() - t_start
                expected = (k + 1) * sleep_per_frame
                if elapsed < expected:
                    time.sleep(expected - elapsed)
            elif renderer is not None:
                renderer.update_scene(data, camera="cam_high")
                mp4_writer.append_data(renderer.render())

            if (k % 50) == 0:
                print(f"  frame {k:3d}/{N}  errL={track_err_L[-1]*1000:5.1f}mm  errR={track_err_R[-1]*1000:5.1f}mm  "
                      f"pL_world={last_pL.round(3)}", flush=True)
    finally:
        if viewer is not None:
            viewer.close()
        if mp4_writer is not None:
            mp4_writer.close()

    track_err_L = np.array(track_err_L)
    track_err_R = np.array(track_err_R)
    tgt_L = np.stack(log_target_L)
    tgt_R = np.stack(log_target_R)
    act_L = np.stack(log_actual_L)
    act_R = np.stack(log_actual_R)
    print(f"\n=== absolute tracking error (EE actual vs mocap target) ===")
    print(f"  L arm: mean={track_err_L.mean()*1000:.1f}mm  "
          f"p95={np.percentile(track_err_L, 95)*1000:.1f}mm  "
          f"max={track_err_L.max()*1000:.1f}mm")
    print(f"  R arm: mean={track_err_R.mean()*1000:.1f}mm  "
          f"p95={np.percentile(track_err_R, 95)*1000:.1f}mm  "
          f"max={track_err_R.max()*1000:.1f}mm")

    # Shape-match: subtract per-arm constant offset and recompute residual.
    # This tells us "does the arm follow the shape of the trajectory" independently
    # of the IK's failure to reach exact absolute pose.
    offset_L = (act_L - tgt_L).mean(axis=0)
    offset_R = (act_R - tgt_R).mean(axis=0)
    res_L = np.linalg.norm((act_L - offset_L) - tgt_L, axis=1)
    res_R = np.linalg.norm((act_R - offset_R) - tgt_R, axis=1)
    print(f"\n=== shape-match residual (after subtracting constant per-arm offset) ===")
    print(f"  L arm: constant offset={offset_L.round(3)*1000} mm  "
          f"residual mean={res_L.mean()*1000:.1f}mm  max={res_L.max()*1000:.1f}mm")
    print(f"  R arm: constant offset={offset_R.round(3)*1000} mm  "
          f"residual mean={res_R.mean()*1000:.1f}mm  max={res_R.max()*1000:.1f}mm")
    print(f"\nINTERPRETATION:")
    print(f"  Large absolute error + small residual → arms are following the SHAPE of")
    print(f"  the trajectory but at a fixed kinematic offset (mocap+weld IK limitation).")
    print(f"  This is sufficient to verify trajectory executability/safety; for tight")
    print(f"  absolute tracking, switch to joint-space IK with scene_joint.xml.")

    # Save EE-position log next to the input npz for diagnostic plotting
    log_path = Path(args.in_npz).parent / "trossen_replay_log.npz"
    np.savez(log_path,
             target_L=tgt_L, target_R=tgt_R,
             actual_L=act_L, actual_R=act_R,
             track_err_L=track_err_L, track_err_R=track_err_R,
             offset_L=offset_L, offset_R=offset_R,
             res_L=res_L, res_R=res_R)
    print(f"\nsaved {log_path}")
    print(f"replay wall-clock: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
