"""End-to-end box-aligned replay prep. One command does the whole pipeline.

Pipeline:
  1. (optional) Capture current box via `boxes/capture_box.py --name current`
     [skipped if --current-ply is passed]
  2. Compute Δxy = current_box_AABB_center − success_box_AABB_center
  3. Translate trajectory bundle by Δ (p_L, p_R offset for every valid frame)
  4. Re-run sim IK (replay_trossen_ik.py) on the aligned bundle, using
     soft 6-DoF (--rot-weight 0.2) by default. This overwrites
     trossen_replay_ik_log.npz.
  5. Print the final hardware-replay command.

Run this from the depth_lerobot env (needs open3d). The sim IK step is
spawned in trossen_sim env via subprocess.

Box-pose PLYs live next to the trajectory they align: by default the script
looks for / writes them under `<trajectory.parent>/box_poses/`. Pass
`--box-pose-dir` to override.

CLI:
  /home/yunshuang/anaconda3/envs/depth_lerobot/bin/python box_align_pipeline.py \\
      --success-ply outputs/recordings_2/wrist/box_poses/success_K11_brown.ply \\
      --trajectory  outputs/recordings_2/wrist/trajectory_contact_adj_K11_Ronly_dRyStart-30_dRzStart-30.npz

  # skip the SAM2 capture step (you already have a current.ply):
  ... --current-ply outputs/recordings_2/wrist/box_poses/current.ply

  # keep z delta too (default: zero Δz, the table is fixed):
  ... --apply-dz
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
import open3d as o3d


REPO_ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_SCENE_XML = pathlib.Path(
    "/home/bruce/Aloha_real/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/scene_joint.xml")
TROSSEN_SIM_PY = pathlib.Path("/home/bruce/miniconda3/envs/trossen_sim/bin/python")


def aabb_center(ply: pathlib.Path) -> np.ndarray:
    pts = np.asarray(o3d.io.read_point_cloud(str(ply)).points)
    if len(pts) == 0:
        raise RuntimeError(f"empty point cloud: {ply}")
    return 0.5 * (pts.min(0) + pts.max(0))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--success-ply", type=pathlib.Path, required=True,
                    help="PLY of the box pose at the success run.")
    ap.add_argument("--trajectory", type=pathlib.Path, required=True,
                    help="Trajectory bundle npz (e.g. trajectory_contact_adj_K11_*.npz).")
    ap.add_argument("--current-ply", type=pathlib.Path, default=None,
                    help="PLY of current box pose. If unset, runs `boxes/capture_box.py "
                         "--name current` first (interactive click).")
    ap.add_argument("--apply-dz", action="store_true",
                    help="Apply Δz too (default: zero Δz since the table is fixed).")
    ap.add_argument("--max-delta-mm", type=float, default=200.0,
                    help="Refuse if |Δ| exceeds this many mm. Default 200.")
    ap.add_argument("--scene-xml", type=pathlib.Path, default=DEFAULT_SCENE_XML)
    ap.add_argument("--rot-weight", type=float, default=0.2,
                    help="Soft DoF weight for sim IK. Default 0.2 (matches successful K=11).")
    ap.add_argument("--gripper-offset-m", type=float, default=0.09,
                    help="Gripper tip offset for the IK (matches successful K=11).")
    ap.add_argument("--out-npz", type=pathlib.Path, default=None,
                    help="Aligned bundle output path. Default: <trajectory>_box_aligned.npz")
    ap.add_argument("--skip-capture", action="store_true",
                    help="Treat --current-ply as already up to date; don't recapture.")
    ap.add_argument("--box-pose-dir", type=pathlib.Path, default=None,
                    help="Directory holding success/current PLYs. "
                         "Default: <trajectory.parent>/box_poses/")
    args = ap.parse_args()

    # Resolve all input/output paths to absolute so .relative_to(REPO_ROOT) works
    # regardless of the cwd the script is run from.
    args.success_ply = args.success_ply.resolve()
    args.trajectory  = args.trajectory.resolve()
    if args.current_ply is not None:
        args.current_ply = args.current_ply.resolve()
    if args.out_npz is not None:
        args.out_npz = args.out_npz.resolve()
    args.scene_xml = args.scene_xml.resolve() if args.scene_xml.exists() else args.scene_xml

    box_pose_dir = (args.box_pose_dir.resolve()
                    if args.box_pose_dir is not None
                    else args.trajectory.parent / "box_poses")
    box_pose_dir.mkdir(parents=True, exist_ok=True)

    def rel(p):
        """Display path: relative to REPO_ROOT if possible, absolute otherwise."""
        p = pathlib.Path(p).resolve()
        try:
            return p.relative_to(REPO_ROOT)
        except ValueError:
            return p

    # ── 1. capture current box (optional) ─────────────────────────────────
    current_ply = args.current_ply
    if current_ply is None and not args.skip_capture:
        current_ply = box_pose_dir / "current.ply"
        print(f"[1/5] capturing current box → {current_ply}")
        capture_script = REPO_ROOT / "boxes" / "capture_box.py"
        if not capture_script.exists():
            raise SystemExit(f"capture script not found: {capture_script}\n"
                             f"Pass --current-ply explicitly to skip this step.")
        rc = subprocess.call([sys.executable, str(capture_script),
                              "--name", "current",
                              "--out-dir", str(box_pose_dir)])
        if rc != 0:
            raise SystemExit(f"capture_box.py exited {rc}")
    elif current_ply is None and args.skip_capture:
        raise SystemExit("--skip-capture requires --current-ply")
    else:
        print(f"[1/5] using existing current PLY: {current_ply}")

    if not current_ply.exists():
        raise SystemExit(f"current PLY not found: {current_ply}")
    if not args.success_ply.exists():
        raise SystemExit(f"success PLY not found: {args.success_ply}")

    # ── 2. compute delta ───────────────────────────────────────────────────
    print(f"\n[2/5] computing Δ from box AABB centers")
    p_success = aabb_center(args.success_ply)
    p_current = aabb_center(current_ply)
    delta = p_current - p_success
    if not args.apply_dz:
        delta[2] = 0.0
    delta_mm = float(np.linalg.norm(delta) * 1000)
    print(f"  success: {(p_success*1000).round(1).tolist()} mm   "
          f"({args.success_ply.name})")
    print(f"  current: {(p_current*1000).round(1).tolist()} mm   ({current_ply.name})")
    print(f"  Δ      : {(delta*1000).round(2).tolist()} mm   |Δ|={delta_mm:.1f} mm   "
          f"{'(z kept)' if args.apply_dz else '(z zeroed)'}")
    if delta_mm > args.max_delta_mm:
        raise SystemExit(f"|Δ|={delta_mm:.0f}mm > --max-delta-mm={args.max_delta_mm:.0f}. "
                         f"Box may be mislocalized — re-check capture or bump threshold.")

    # ── 3. translate trajectory ───────────────────────────────────────────
    print(f"\n[3/5] translating trajectory by Δ")
    if not args.trajectory.exists():
        raise SystemExit(f"trajectory not found: {args.trajectory}")
    bundle = dict(np.load(args.trajectory))
    if "p_L" not in bundle or "p_R" not in bundle:
        raise SystemExit(f"bundle missing p_L / p_R keys: {args.trajectory}")
    vL = bundle["valid_L"].astype(bool); vR = bundle["valid_R"].astype(bool)
    bundle["p_L_pre_align"] = bundle["p_L"].copy()
    bundle["p_R_pre_align"] = bundle["p_R"].copy()
    p_L = bundle["p_L"].copy(); p_R = bundle["p_R"].copy()
    p_L[vL] += delta; p_R[vR] += delta
    bundle["p_L"] = p_L; bundle["p_R"] = p_R
    bundle["box_align_delta_xyz_m"]  = delta.astype(np.float64)
    bundle["box_align_success_ply"]  = np.asarray(str(args.success_ply.resolve()))
    bundle["box_align_current_ply"]  = np.asarray(str(current_ply.resolve()))
    bundle["box_align_apply_dz"]     = np.bool_(args.apply_dz)

    out_npz = args.out_npz or args.trajectory.with_name(args.trajectory.stem + "_box_aligned.npz")
    np.savez_compressed(out_npz, **bundle)
    out_npz = out_npz.resolve()
    print(f"  saved aligned bundle → {rel(out_npz)}")

    # Verify translation actually took effect by sampling before/after.
    both = vL & vR
    if both.any():
        idxs = np.where(both)[0]
        sample = [int(idxs[0]), int(idxs[len(idxs)//2]), int(idxs[-1])]
        print(f"  translation check (frames {sample}):")
        for t in sample:
            d_L = (p_L[t] - bundle["p_L_pre_align"][t]) * 1000
            d_R = (p_R[t] - bundle["p_R_pre_align"][t]) * 1000
            print(f"    t={t:3d}  L Δ={d_L.round(2).tolist()} mm  R Δ={d_R.round(2).tolist()} mm")
        # All shifts should equal `delta` for valid frames — sanity confirm
        max_dev = max(
            float(np.abs((p_L[both] - bundle["p_L_pre_align"][both]) - delta).max()),
            float(np.abs((p_R[both] - bundle["p_R_pre_align"][both]) - delta).max()),
        )
        if max_dev > 1e-9:
            print(f"  ⚠ shift varies per-frame by up to {max_dev*1000:.3f} mm — should be constant.")
        else:
            print(f"  ✓ shift is uniform across all valid frames")
    if abs(delta).max() < 1e-4:
        print(f"  ⚠ Δ ≈ 0 — no translation applied. Did the current box actually move?")

    # ── 4. re-run sim IK (subprocess to trossen_sim env) ──────────────────
    print(f"\n[4/5] re-running sim IK (replay_trossen_ik.py, rot-weight {args.rot_weight})")
    if not TROSSEN_SIM_PY.exists():
        raise SystemExit(f"trossen_sim python not found: {TROSSEN_SIM_PY}")
    sim_mp4 = out_npz.with_suffix(".mp4").name
    sim_mp4_path = out_npz.parent / f"replay_{out_npz.stem}.mp4"
    # Write the IK log directly to the box-aligned name via --out-log so the
    # existing default `trossen_replay_ik_log.npz` (matching the un-shifted
    # trajectory) is never opened for writing.
    ik_log = out_npz.parent / (out_npz.stem + "_iklog.npz")
    env = os.environ.copy(); env["MUJOCO_GL"] = "egl"
    rc = subprocess.call([
        str(TROSSEN_SIM_PY), str(REPO_ROOT / "replay_trossen_ik.py"),
        "--in_npz", str(out_npz),
        "--scene_xml", str(args.scene_xml),
        "--record_mp4", str(sim_mp4_path),
        "--rot-weight", str(args.rot_weight),
        "--warmup_seconds", "2.0",
        "--gripper_offset_m", str(args.gripper_offset_m),
        "--out-log", str(ik_log),
    ], env=env)
    if rc != 0:
        raise SystemExit(f"sim IK exited {rc}")
    print(f"  IK log saved as    → {rel(ik_log)}  (original log untouched)")
    print(f"  sim render         → {rel(sim_mp4_path)}")

    # IK quality + joint-config sanity (did the joints actually change vs original?)
    log = np.load(ik_log)
    valL = ~np.isnan(log['ik_pos_err_L']); valR = ~np.isnan(log['ik_pos_err_R'])
    pe_L_mean = log['ik_pos_err_L'][valL].mean() * 1000
    pe_L_max  = log['ik_pos_err_L'][valL].max()  * 1000
    pe_R_mean = log['ik_pos_err_R'][valR].mean() * 1000
    pe_R_max  = log['ik_pos_err_R'][valR].max()  * 1000
    re_L_max  = np.degrees(log['ik_rot_err_L'][valL]).max()
    re_R_max  = np.degrees(log['ik_rot_err_R'][valR]).max()
    print(f"  IK quality:")
    print(f"    L pos err  mean={pe_L_mean:.2f}mm  max={pe_L_max:.2f}mm   rot err max={re_L_max:.1f}°")
    print(f"    R pos err  mean={pe_R_mean:.2f}mm  max={pe_R_max:.2f}mm   rot err max={re_R_max:.1f}°")
    if pe_L_max > 5 or pe_R_max > 5:
        print(f"    ⚠ IK pos err > 5 mm — possible reachability issue in new box region.")

    # Compare joint configs to original IK log to confirm the translation
    # actually changed the joints (= proves the new run is doing something different).
    orig_log_path = out_npz.parent / "trossen_replay_ik_log.npz"
    if orig_log_path.exists() and orig_log_path != ik_log:
        orig = np.load(orig_log_path)
        if 'joint_L' in orig and orig['joint_L'].shape == log['joint_L'].shape:
            dj_L = np.abs(log['joint_L'] - orig['joint_L'])
            dj_R = np.abs(log['joint_R'] - orig['joint_R'])
            both = (valL & valR)
            if both.any():
                print(f"  joint-config Δ vs original (rad):")
                print(f"    L max={dj_L[both].max():.4f}  mean={dj_L[both].mean():.4f}")
                print(f"    R max={dj_R[both].max():.4f}  mean={dj_R[both].mean():.4f}")
                if dj_L[both].max() < 1e-4 and dj_R[both].max() < 1e-4:
                    print(f"    ⚠ joints are identical to original — box align had no effect on output.")

    # ── 5. print hardware replay command ──────────────────────────────────
    print(f"\n[5/5] ready — replay on the real robot:\n")
    print(f"  source /home/yigit/miniconda3/etc/profile.d/conda.sh && conda activate lerobot && \\")
    print(f"  cd {REPO_ROOT} && \\")
    print(f"  MKL_SERVICE_FORCE_INTEL=1 MKL_THREADING_LAYER=GNU \\")
    print(f"  python replay_real.py \\")
    print(f"      --log {rel(ik_log)} \\")
    print(f"      --max-step-delta 0.2")
    print(f"  # add '--d455 low' to also record from the D455 workspace cam.\n")


if __name__ == "__main__":
    main()
