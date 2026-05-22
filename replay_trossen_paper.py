"""
Paper-strict Trossen MuJoCo replay.

What's different from replay_trossen_ik.py:
  - Orientation is ALWAYS tracked (no --position_only). Paper does not throw
    rotation away; this pipeline honors it.
  - No learned `compute_orientation_remap`. Instead a single hard-coded
    R_align maps the gripper's local axes onto the hand's local axes
    (paper Algorithm 1's basis: v_x, v_y=fingers, v_z=palm-normal).
    Default R_align = R_y(-90°)  ⇒  gripper-local +x (tool axis) aligns with
    hand-local +z (palm normal = where hand pushes the box).
    Was visually verified in:
      - calibrate_gripper_align.py  (world-frame 3D arrows)
      - calibrate_orient_rgb_overlay.py (RGB overlay on source frame)
  - Per-frame 6-DoF → 3-DoF fallback is OFF by default. Paper-strict: if IK
    can't reach a (p, R) target, log it; do NOT silently drop orientation
    on that frame. Use --allow_pos_fallback to opt in.

Everything else (frame alignment, gripper-tip-at-wrist via tip_offset_local,
joint Gaussian smoothing, boundary pinning, back-fill, joint-space IK
with damped LS) is identical to replay_trossen_ik.py — reused via import.

Usage:
  python replay_trossen_paper.py \
      --in_npz outputs/recordings_1/wrist/trajectory_smoothed.npz \
      --record_mp4 outputs/recordings_1/wrist/replay_paper_oriented.mp4 \
      --gripper_offset_m 0.09
"""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
import mujoco.viewer

from replay_trossen_ik import (
    quat_xyzw_to_wxyz, quat_wxyz_to_rotmat, rotmat_to_quat_wxyz,
    compute_frame_alignment, get_arm_indices, ik_solve,
    TROSSEN_TABLE_Z,
)


def euler_zyx_to_rotmat(rx_deg, ry_deg, rz_deg):
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", required=True)
    ap.add_argument("--scene_xml",
                    default="/data/bruce/BiNoMaP/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/scene_joint.xml")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--record_mp4", type=str, default=None)
    ap.add_argument("--mp4_fps", type=float, default=30.0)
    ap.add_argument("--our_table_z", type=float, default=0.02)
    ap.add_argument("--gripper_offset_m", type=float, default=0.09,
                    help="Trossen gripper tip offset along link_6 local +x")
    ap.add_argument("--align_euler_deg", default="0,-90,0",
                    help="Calibrated gripper-local→hand-local mapping. Default "
                         "puts gripper-pointing on Algorithm-1's palm normal.")
    ap.add_argument("--smooth_joints_sigma", type=float, default=2.0)
    ap.add_argument("--use_raw_R", action="store_true", default=True,
                    help="Use Algorithm-1 raw R instead of Stage 2a SLERPed R "
                         "(default ON: avoids 178°-rotation SLERP artifact in recordings_1 R-arm).")
    ap.add_argument("--no_use_raw_R", dest="use_raw_R", action="store_false")
    ap.add_argument("--smooth_R_sigma", type=float, default=5.0,
                    help="Temporal Gaussian on orientation TARGETS before IK. "
                         "Stage 2a's SLERP-between-anchors leaves frame-to-frame "
                         "jitter (~3°/frame on R from Algorithm 1 noise). IK "
                         "faithfully tracks that jitter into joint twitch. "
                         "Smoothing the target sequence in axis-angle space first "
                         "(σ in frames) removes the source. Set to 0 to disable.")
    ap.add_argument("--allow_pos_fallback", action="store_true",
                    help="Per-frame fallback to position-only IK when 6-DoF "
                         "fails. OFF by default in paper-strict mode.")
    ap.add_argument("--pos_fallback_thresh_mm", type=float, default=20.0)
    ap.add_argument("--render_w", type=int, default=640)
    ap.add_argument("--render_h", type=int, default=480)
    ap.add_argument("--camera", default="cam_high")
    ap.add_argument("--warmup_seconds", type=float, default=1.0)
    args = ap.parse_args()

    rx, ry, rz = [float(x) for x in args.align_euler_deg.split(",")]
    R_align = euler_zyx_to_rotmat(rx, ry, rz)

    bundle = np.load(args.in_npz)
    ts_ms = bundle["ts_ms"]
    p_L = bundle["p_L"].copy()
    p_R = bundle["p_R"].copy()
    valid_L = bundle["valid_L"]
    valid_R = bundle["valid_R"]
    N = len(ts_ms)
    # IMPORTANT for R-arm: Stage 2a's SLERP between sparse anchors produces a
    # ~178° rotation across the 15-frame [anchor_177 → anchor_192] segment
    # (anchors picked from a noisy Algorithm 1 output, not the actual hand).
    # That's where the "vibrate twisting" comes from. We bypass the SLERPed R
    # and use the RAW Algorithm-1 rotations + our own median+gaussian denoising.
    if args.use_raw_R and "R_L_raw" in bundle.files:
        R_L_world = bundle["R_L_raw"]
        R_R_world = bundle["R_R_raw"]
        print("orientation source: RAW Algorithm-1 R (bypassing Stage 2a SLERP)")
    else:
        R_L_world = bundle["R_L"]
        R_R_world = bundle["R_R"]
        print("orientation source: Stage 2a SLERPed R")

    print(f"=== Trossen Paper-Strict Replay (orientation ON) ===")
    print(f"input: {args.in_npz}  N={N}  valid_L={int(valid_L.sum())}  valid_R={int(valid_R.sum())}")
    print(f"R_align (euler ZYX {rx},{ry},{rz}°):")
    print(R_align.round(3))
    print(f"  gripper-local +x → hand-local axis: {(R_align @ np.array([1,0,0])).round(3)}  "
          f"(0,0,1 means gripper points along palm normal)")

    offset, ours_mid, target = compute_frame_alignment(p_L, p_R, valid_L, valid_R, args.our_table_z)
    print(f"frame offset = {offset.round(3)}")
    p_L_t = p_L + offset[None, :]
    p_R_t = p_R + offset[None, :]

    # Apply R_align per frame to convert hand-frame world rotation → gripper-frame
    # world rotation.
    R_grip_L_arr = np.zeros((N, 3, 3))
    R_grip_R_arr = np.zeros((N, 3, 3))
    for k in range(N):
        R_grip_L_arr[k] = R_L_world[k] @ R_align
        R_grip_R_arr[k] = R_R_world[k] @ R_align

    # Temporal orientation smoothing over valid frames. Componentwise Gaussian on
    # the rotation matrices with re-orthonormalization (SVD projection back to
    # SO(3)) is a cheap and accurate approximation for small per-frame deltas.
    # Robust outlier rejection on orientation. WiLoR-mini sometimes mislabels
    # right-as-left on R cam; on those frames it also X-flips the input image,
    # and Algorithm 1's sign-flip then makes v_z point 180° wrong. The resulting
    # bimodal cluster (correct R + 180°-flipped R) drives SLERP into 11.9°/frame
    # spurious "wrist twists" between anchors. We detect frames whose orientation
    # is >threshold from the windowed-mean rotation and either flip them back
    # (if the flip is ~180°) or mark them invalid for orientation tracking.
    from scipy.spatial.transform import Rotation as Rot
    def reject_flipped(R_seq, valid, win=15, thresh_deg=90.0):
        out = R_seq.copy(); out_valid = valid.copy()
        idx = np.where(valid)[0]
        if len(idx) < 5:
            return out, out_valid, 0
        n_flipped = 0
        for n, k in enumerate(idx):
            lo = max(0, n - win); hi = min(len(idx), n + win + 1)
            neigh_k = [idx[i] for i in range(lo, hi) if idx[i] != k]
            if len(neigh_k) < 5:
                continue
            mean_R = Rot.from_matrix(R_seq[neigh_k]).mean().as_matrix()
            rel = mean_R.T @ R_seq[k]
            ang_deg = np.degrees(np.arccos(np.clip((np.trace(rel) - 1) * 0.5, -1, 1)))
            if ang_deg > thresh_deg:
                # Try flipping (180° around the gripper-pointing axis = local +x)
                # If flip brings it close to mean, accept the flip; else mark invalid.
                R180 = R_seq[k] @ np.diag([1.0, -1.0, -1.0])  # 180° around +x (gripper tool axis)
                rel2 = mean_R.T @ R180
                ang2 = np.degrees(np.arccos(np.clip((np.trace(rel2) - 1) * 0.5, -1, 1)))
                if ang2 < thresh_deg:
                    out[k] = R180
                    n_flipped += 1
                else:
                    out_valid[k] = False
        return out, out_valid, n_flipped
    R_grip_L_arr, valid_L_orient, nfL = reject_flipped(R_grip_L_arr, valid_L)
    R_grip_R_arr, valid_R_orient, nfR = reject_flipped(R_grip_R_arr, valid_R)
    print(f"orientation outlier rejection (180° flips):  L flipped-back={nfL}  R flipped-back={nfR}")
    print(f"  R-arm valid orient: {int(valid_R_orient.sum())}/{int(valid_R.sum())}  "
          f"(rejected {int(valid_R.sum())-int(valid_R_orient.sum())} as un-rescuable outliers)")

    if args.smooth_R_sigma > 0:
        from scipy.ndimage import gaussian_filter1d, median_filter
        def smooth_R_seq(R_seq, valid):
            """Median-filter (rejects spikes) then Gaussian-smooth (denoise),
            then SVD-project each frame back to SO(3). No boundary pinning —
            for orientation, drift of a few degrees at the boundary is far
            preferable to a discontinuous pin → smooth transition."""
            idx = np.where(valid)[0]
            if len(idx) < 2:
                return R_seq.copy()
            seg = R_seq[idx].copy()                # (n_valid, 3, 3)
            # 7-frame median filter on each entry (rejects per-frame outliers)
            for i in range(3):
                for j in range(3):
                    seg[:, i, j] = median_filter(seg[:, i, j], size=7, mode="nearest")
            # Then Gaussian denoise
            for i in range(3):
                for j in range(3):
                    seg[:, i, j] = gaussian_filter1d(seg[:, i, j], sigma=args.smooth_R_sigma)
            R_smooth = R_seq.copy()
            for n, k in enumerate(idx):
                U, _, Vt = np.linalg.svd(seg[n])
                R = U @ Vt
                if np.linalg.det(R) < 0:
                    U[:, -1] *= -1
                    R = U @ Vt
                R_smooth[k] = R
            return R_smooth
        R_grip_L_arr = smooth_R_seq(R_grip_L_arr, valid_L)
        R_grip_R_arr = smooth_R_seq(R_grip_R_arr, valid_R)
        # Stats: post-smoothing frame-to-frame angle
        def jitter_stats(R, v):
            idx = np.where(v)[0]
            angs = []
            for i in range(len(idx)-1):
                if idx[i+1]-idx[i] == 1:
                    dR = R[idx[i]].T @ R[idx[i+1]]
                    cos_th = np.clip((np.trace(dR)-1)*0.5, -1, 1)
                    angs.append(np.degrees(np.arccos(cos_th)))
            return np.mean(angs), np.max(angs) if angs else (0, 0)
        m_L, x_L = jitter_stats(R_grip_L_arr, valid_L)
        m_R, x_R = jitter_stats(R_grip_R_arr, valid_R)
        print(f"orientation target smoothing: 7-frame median + gaussian σ={args.smooth_R_sigma}")
        print(f"  R-target frame-to-frame angle:  L mean={m_L:.2f}° max={x_L:.1f}°   "
              f"R mean={m_R:.2f}° max={x_R:.1f}°  (target < 1°/frame)")

    q_L_mj = np.stack([rotmat_to_quat_wxyz(R_grip_L_arr[k]) for k in range(N)])
    q_R_mj = np.stack([rotmat_to_quat_wxyz(R_grip_R_arr[k]) for k in range(N)])

    tip_offset_local = np.array([args.gripper_offset_m, 0.0, 0.0])
    print(f"gripper tip offset: {args.gripper_offset_m:+.3f}m along link_6 +x")

    model = mujoco.MjModel.from_xml_path(args.scene_xml)
    data = mujoco.MjData(model)
    arm_L = get_arm_indices(model, "left")
    arm_R = get_arm_indices(model, "right")

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    home_qL = data.qpos[arm_L[0]].copy()
    home_qR = data.qpos[arm_R[0]].copy()
    timestep = model.opt.timestep
    print(f"model: nq={model.nq} nv={model.nv} timestep={timestep}")

    # ===== IK pass =====
    print(f"\n--- IK pass over {N} frames (6-DoF, paper-strict) ---")
    joint_targets_L = np.tile(home_qL, (N, 1))
    joint_targets_R = np.tile(home_qR, (N, 1))
    ik_pos_err_L = np.full(N, np.nan)
    ik_pos_err_R = np.full(N, np.nan)
    ik_rot_err_L = np.full(N, np.nan)
    ik_rot_err_R = np.full(N, np.nan)
    ik_fail_L = ik_fail_R = 0
    n_fallback_L = n_fallback_R = 0
    POS_FALLBACK_M = args.pos_fallback_thresh_mm / 1000.0
    last_qL = home_qL.copy(); last_qR = home_qR.copy()
    t0 = time.time()

    def do_ik(arm_info, target_pos, target_quat, last_q, allow_fb):
        data.qpos[arm_info[0]] = last_q
        pe, re, it, conv = ik_solve(model, data, target_pos, target_quat, arm_info,
                                    use_orientation=True,
                                    tip_offset_local=tip_offset_local)
        fb_used = False
        if allow_fb and (not conv or pe > POS_FALLBACK_M):
            data.qpos[arm_info[0]] = last_q
            pe, re, it, conv = ik_solve(model, data, target_pos, target_quat, arm_info,
                                        use_orientation=False,
                                        tip_offset_local=tip_offset_local)
            fb_used = True
        return pe, re, conv, data.qpos[arm_info[0]].copy(), fb_used

    for k in range(N):
        if valid_L[k]:
            pe, re, conv, q_out, fb = do_ik(arm_L, p_L_t[k], q_L_mj[k], last_qL, args.allow_pos_fallback)
            joint_targets_L[k] = q_out; last_qL = q_out
            ik_pos_err_L[k] = pe; ik_rot_err_L[k] = re
            if fb: n_fallback_L += 1
            if not conv: ik_fail_L += 1
        else:
            joint_targets_L[k] = last_qL

        if valid_R[k]:
            pe, re, conv, q_out, fb = do_ik(arm_R, p_R_t[k], q_R_mj[k], last_qR, args.allow_pos_fallback)
            joint_targets_R[k] = q_out; last_qR = q_out
            ik_pos_err_R[k] = pe; ik_rot_err_R[k] = re
            if fb: n_fallback_R += 1
            if not conv: ik_fail_R += 1
        else:
            joint_targets_R[k] = last_qR

        if (k % 25) == 0:
            print(f"  IK frame {k:3d}/{N}  "
                  f"L pos={ik_pos_err_L[k]*1000:5.1f}mm rot={np.degrees(ik_rot_err_L[k]):5.1f}°  "
                  f"R pos={ik_pos_err_R[k]*1000:5.1f}mm rot={np.degrees(ik_rot_err_R[k]):5.1f}°")
    print(f"IK pass done in {time.time()-t0:.1f}s. failed (not converged) L={ik_fail_L}/{int(valid_L.sum())}  "
          f"R={ik_fail_R}/{int(valid_R.sum())}")
    if args.allow_pos_fallback:
        print(f"  position-only fallback used: L={n_fallback_L}  R={n_fallback_R}")
    else:
        print(f"  position-only fallback DISABLED (paper-strict)")

    first_valid_L = int(np.where(valid_L)[0][0])
    first_valid_R = int(np.where(valid_R)[0][0])
    last_valid_L = int(np.where(valid_L)[0][-1])
    last_valid_R = int(np.where(valid_R)[0][-1])

    # Fill interior invalid gaps via joint-space linear interpolation between
    # the surrounding valid IK solutions. Without this, joint_targets[k] during
    # a gap holds the last valid config and then jumps to the next valid one
    # — and because j5 (wrist roll) is in the null space of gripper-pointing,
    # IK can pick a wildly different j5 after a long gap. Linear-interp in
    # joint space gives a smooth ramp instead of a held → jump.
    def interp_gaps(joint_targets, valid):
        idx = np.where(valid)[0]
        if len(idx) < 2:
            return
        n_gaps_filled = 0
        for i in range(len(idx) - 1):
            k0, k1 = idx[i], idx[i + 1]
            if k1 - k0 <= 1:
                continue
            q0, q1 = joint_targets[k0], joint_targets[k1]
            for kk in range(k0 + 1, k1):
                a = (kk - k0) / (k1 - k0)
                joint_targets[kk] = (1 - a) * q0 + a * q1
            n_gaps_filled += 1
        return n_gaps_filled
    nL = interp_gaps(joint_targets_L, valid_L)
    nR = interp_gaps(joint_targets_R, valid_R)
    print(f"interior gap fill (joint-space lerp between valid IK solutions): L={nL} gaps  R={nR} gaps")

    if args.smooth_joints_sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        pinned_L0 = joint_targets_L[first_valid_L].copy()
        pinned_R0 = joint_targets_R[first_valid_R].copy()
        pinned_L1 = joint_targets_L[last_valid_L].copy()
        pinned_R1 = joint_targets_R[last_valid_R].copy()
        for j in range(6):
            joint_targets_L[:, j] = gaussian_filter1d(joint_targets_L[:, j], sigma=args.smooth_joints_sigma)
            joint_targets_R[:, j] = gaussian_filter1d(joint_targets_R[:, j], sigma=args.smooth_joints_sigma)
        joint_targets_L[: first_valid_L + 1] = pinned_L0
        joint_targets_R[: first_valid_R + 1] = pinned_R0
        joint_targets_L[last_valid_L:] = pinned_L1
        joint_targets_R[last_valid_R:] = pinned_R1
        print(f"joint smoothing: σ={args.smooth_joints_sigma}, boundaries pinned")

    if first_valid_L > 0:
        joint_targets_L[:first_valid_L] = joint_targets_L[first_valid_L]
    if first_valid_R > 0:
        joint_targets_R[:first_valid_R] = joint_targets_R[first_valid_R]

    vLm = ~np.isnan(ik_pos_err_L); vRm = ~np.isnan(ik_pos_err_R)
    print(f"IK pos err (m): L mean={ik_pos_err_L[vLm].mean()*1000:.1f}mm max={ik_pos_err_L[vLm].max()*1000:.1f}mm  "
          f"R mean={ik_pos_err_R[vRm].mean()*1000:.1f}mm max={ik_pos_err_R[vRm].max()*1000:.1f}mm")
    print(f"IK rot err (deg): L mean={np.degrees(ik_rot_err_L[vLm]).mean():.1f} max={np.degrees(ik_rot_err_L[vLm]).max():.1f}  "
          f"R mean={np.degrees(ik_rot_err_R[vRm]).mean():.1f} max={np.degrees(ik_rot_err_R[vRm]).max():.1f}")

    # ===== Replay =====
    print(f"\n--- replaying with joint-position control ---")
    mujoco.mj_resetDataKeyframe(model, data, 0)
    substeps = max(1, int(round(1.0 / args.fps / timestep)))
    warmup_steps = max(int(args.warmup_seconds / timestep), 1)
    print(f"substeps/frame={substeps}  warmup={warmup_steps} steps")
    for s in range(warmup_steps):
        a = (s + 1) / warmup_steps
        data.ctrl[arm_L[2]] = (1 - a) * home_qL + a * joint_targets_L[0]
        data.ctrl[arm_R[2]] = (1 - a) * home_qR + a * joint_targets_R[0]
        mujoco.mj_step(model, data)

    renderer = mp4 = viewer = None
    if args.record_mp4:
        import imageio
        renderer = mujoco.Renderer(model, height=args.render_h, width=args.render_w)
        mp4 = imageio.get_writer(args.record_mp4, fps=args.mp4_fps, codec="libx264")
    elif not args.headless:
        viewer = mujoco.viewer.launch_passive(model, data)

    actual_pos_L = np.zeros((N, 3)); actual_pos_R = np.zeros((N, 3))
    try:
        for k in range(N):
            data.ctrl[arm_L[2]] = joint_targets_L[k]
            data.ctrl[arm_R[2]] = joint_targets_R[k]
            for _ in range(substeps):
                mujoco.mj_step(model, data)
            actual_pos_L[k] = data.xpos[arm_L[3]].copy()
            actual_pos_R[k] = data.xpos[arm_R[3]].copy()
            if viewer is not None:
                viewer.sync()
            elif renderer is not None:
                renderer.update_scene(data, camera=args.camera)
                mp4.append_data(renderer.render())
            if (k % 50) == 0:
                print(f"  replay frame {k:3d}/{N}")
    finally:
        if mp4 is not None: mp4.close()
        if viewer is not None: viewer.close()

    log_path = Path(args.in_npz).parent / "trossen_replay_paper_log.npz"
    np.savez(log_path,
             target_L=p_L_t, target_R=p_R_t,
             actual_L=actual_pos_L, actual_R=actual_pos_R,
             joint_L=joint_targets_L, joint_R=joint_targets_R,
             ik_pos_err_L=ik_pos_err_L, ik_pos_err_R=ik_pos_err_R,
             ik_rot_err_L=ik_rot_err_L, ik_rot_err_R=ik_rot_err_R,
             R_align=R_align, frame_offset=offset)
    print(f"\nsaved {log_path}")
    if args.record_mp4:
        print(f"saved {args.record_mp4}")


if __name__ == "__main__":
    main()
