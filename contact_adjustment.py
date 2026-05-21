"""BiNoMaP Stage 2b — Geometry-aware iterative contact adjustment.

Per BiNoMaP §3.3 (https://arxiv.org/abs/2509.21256), after Stage 2a (motion
smoothness) the primary arm's start-of-trajectory contact point is shifted
toward the object surface, then the SAME scaling factor is applied uniformly
to that arm's entire trajectory. The intent is to close the gap between the
human demonstrator's wrist position (which sits 5-10 cm behind the contact
point) and where the robot's gripper actually needs to make contact.

Algorithm (one invocation = one iteration):
  1. Pick primary arm (default: right — the active "lift/pivot" hand in
     both `recordings_1/wrist` and `recordings_one_hand_fix_*` demos).
  2. d_k = 5 mm * 0.85^(k-1)   (paper Eq. 11)
  3. Find first jointly-valid frame t_s.
  4. Closest point on object cloud to p_pri[t_s]: c_pri at distance D_orig.
  5. Shift p_pri[t_s] toward c_pri by (D_orig - d_k):
        new_p_pri_ts = p_pri[t_s] + (D_orig - d_k) * (c_pri - p_pri[t_s]) / D_orig
  6. s_k = ‖new_p_pri_ts − p_sec[t_s]‖ / ‖p_pri[t_s] − p_sec[t_s]‖
  7. ∀ t with valid_pri[t]:  p_pri[t] = p_sec[t] + s_k * (p_pri[t] − p_sec[t])
  8. Save adjusted bundle + 3D before/after PNG.

How you use this:
  Start with k=1.  Run the IK pass on the adjusted bundle, run replay_real.py.
  If the gripper still doesn't quite touch the box → re-run with k=2.  Repeat
  up to k=10. The paper notes 1-3 iterations usually suffice for plain boxes.

Inputs (defaults shown):
  --smoothed PATH        trajectory_smoothed.npz from smooth_trajectory.py
  --object-ply PATH      outputs/brown_box/brown_box.ply  (from brown_box_pointcloud.py)
  --iteration K          1..10            (paper default schedule)
  --distance-mm D        bypasses --iteration if set; for ad-hoc experiments
  --primary {left,right} default right

Frame note:  both inputs are assumed to be in the SAME world coordinate frame
(the camera_extrinsics.json frame, which both `extract_trajectory.py` and
`brown_box_pointcloud.py` use).  No frame remapping is done here.  If the box
on the table is far from where the demo box was, D_orig will be large (>10
cm) — that's a signal to reposition the box or recapture the demo.
"""

import argparse
import pathlib

import numpy as np
import open3d as o3d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


D1_M = 5e-3       # safety distance for iteration 1: 5 mm
GAMMA = 0.85      # decay rate per iteration
MAX_K = 10        # paper-recommended cap


def closest_point(p: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Brute-force closest-point query: returns (point, distance)."""
    diffs = pts - p[None, :]
    d2 = np.einsum("ij,ij->i", diffs, diffs)
    i = int(np.argmin(d2))
    return pts[i], float(np.sqrt(d2[i]))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoothed", type=pathlib.Path, required=True,
                    help="trajectory_smoothed.npz from smooth_trajectory.py")
    ap.add_argument("--object-ply", type=pathlib.Path,
                    default=pathlib.Path(__file__).parent / "outputs" / "brown_box" / "brown_box.ply",
                    help="object point cloud (default: outputs/brown_box/brown_box.ply)")
    ap.add_argument("--iteration", type=int, default=1,
                    help="iteration number k. d_k = 5mm * 0.85^(k-1). 1..10 per paper.")
    ap.add_argument("--distance-mm", type=float, default=None,
                    help="Override d_k directly in mm (bypasses --iteration).")
    ap.add_argument("--primary", choices=["left", "right"], default="right",
                    help="Which arm gets shifted (in --apply-to primary mode). Default: right.")
    ap.add_argument("--apply-to", choices=["primary", "both"], default="primary",
                    help="primary: only adjust the --primary arm (paper §3.3 default). "
                         "both: independently bring each arm to distance d_k from the box "
                         "(forces --mode translate; each arm gets its own translation toward "
                         "its closest box point). Useful for setting up a wide bimanual "
                         "starting envelope and iterating d_k downward to find contact.")
    ap.add_argument("--mode", choices=["scale", "translate"], default="scale",
                    help="How to propagate the t_s adjustment to the rest of the "
                         "trajectory. 'scale' (paper Eq. 11): scale primary relative "
                         "to secondary by s_k=|new_pri-sec|/|orig_pri-sec| — preserves "
                         "the secondary as a moving reference, but uniform scaling "
                         "can shrink the inter-arm gap at later frames when the demo's "
                         "L and R were already close (gripper collision risk on box-pivot "
                         "demos). 'translate': shift primary by a constant vector "
                         "(new_pri_ts - orig_pri_ts) — preserves the demo's inter-arm "
                         "distance evolution exactly, only the absolute primary position "
                         "is shifted.")
    ap.add_argument("--lateral-spread-mm", type=float, default=0.0,
                    help="Push L by +S/2 in +y AND R by -S/2 in -y (S = this value, mm). "
                         "Widens the inter-arm gap along the y axis without changing z. "
                         "Decoupled from --distance-mm / --apply-to — these are pure "
                         "lateral shifts, applied to every valid frame.")
    ap.add_argument("--lower-mm", type=float, default=0.0,
                    help="Shift BOTH arms down in z by this many mm "
                         "(positive = lower, negative = raise). Acts as a default "
                         "for --lower-L-mm and --lower-R-mm if those are unset.")
    ap.add_argument("--lower-L-mm", type=float, default=None,
                    help="Lower L only by this many mm. Overrides --lower-mm for L. "
                         "Use 0 or negative when L is already descending to the table "
                         "in the demo (the pivot's end frames put the L gripper tip at "
                         "the floor; more lowering crashes).")
    ap.add_argument("--lower-R-mm", type=float, default=None,
                    help="Lower R only by this many mm. Overrides --lower-mm for R. "
                         "Use larger values when R sits high above the box (typical "
                         "in single-hand-active demos where the passive arm never "
                         "approaches the object).")
    ap.add_argument("--lower-R-start-mm", type=float, default=None,
                    help="Lower R's STARTING pose by this many mm, linearly decaying "
                         "to 0 by the last valid R frame. Preserves R's end-of-motion "
                         "position from the demo (where R's lifted away from the box "
                         "anyway). Mutually exclusive with --lower-R-mm.")
    ap.add_argument("--lower-L-start-mm", type=float, default=None,
                    help="Same as --lower-R-start-mm but for L. Mutually exclusive "
                         "with --lower-L-mm.")
    ap.add_argument("--shift-R-y-start-mm", type=float, default=None,
                    help="Shift R's STARTING pose in y by this many mm, linearly "
                         "decaying to 0 by the last valid R frame. Preserves R's "
                         "end-of-motion y position from the demo. Negative = toward -y "
                         "(away from L for our box-pivot geometry).")
    ap.add_argument("--shift-L-y-start-mm", type=float, default=None,
                    help="Same as --shift-R-y-start-mm but for L. Positive = toward +y.")
    ap.add_argument("--out-dir", type=pathlib.Path, default=None,
                    help="Output dir (default: same as --smoothed parent)")
    ap.add_argument("--tag", type=str, default=None,
                    help="Optional suffix for output filenames. Default: 'iter{K}' "
                         "(or 'd{Dmm:.1f}mm' if --distance-mm given).")
    args = ap.parse_args()

    if not (1 <= args.iteration <= MAX_K):
        raise ValueError(f"iteration must be in [1, {MAX_K}]")
    if not args.smoothed.exists():
        raise FileNotFoundError(args.smoothed)
    if not args.object_ply.exists():
        raise FileNotFoundError(args.object_ply)

    out_dir = args.out_dir or args.smoothed.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load inputs ──────────────────────────────────────────────────────────
    bundle = dict(np.load(args.smoothed))
    required = ["p_L", "p_R", "valid_L", "valid_R"]
    for k in required:
        if k not in bundle:
            raise KeyError(f"--smoothed missing required key '{k}'")
    p_L = bundle["p_L"].astype(np.float64)
    p_R = bundle["p_R"].astype(np.float64)
    valid_L = bundle["valid_L"].astype(bool)
    valid_R = bundle["valid_R"].astype(bool)
    N = p_L.shape[0]
    print(f"smoothed bundle: N={N}  valid_L={int(valid_L.sum())}  valid_R={int(valid_R.sum())}")

    obj = np.asarray(o3d.io.read_point_cloud(str(args.object_ply)).points)
    if len(obj) == 0:
        raise RuntimeError(f"empty point cloud at {args.object_ply}")
    print(f"object cloud: {len(obj):,} pts  AABB center "
          f"{obj.mean(0).round(3).tolist()}  extent {(obj.max(0)-obj.min(0)).round(3).tolist()}")

    # ── Lateral spread + lower (geometry-decoupled mode) ────────────────────
    # If any of spread/lower/lower-L/lower-R/lower-*-start is set, this is a
    # pure translational widen/lower pass — no distance-to-box logic, no scaling.
    if args.lower_L_mm is not None and args.lower_L_start_mm is not None:
        raise ValueError("--lower-L-mm and --lower-L-start-mm are mutually exclusive")
    if args.lower_R_mm is not None and args.lower_R_start_mm is not None:
        raise ValueError("--lower-R-mm and --lower-R-start-mm are mutually exclusive")
    lower_L = args.lower_L_mm if args.lower_L_mm is not None else args.lower_mm
    lower_R = args.lower_R_mm if args.lower_R_mm is not None else args.lower_mm
    lower_L_start = args.lower_L_start_mm or 0.0
    lower_R_start = args.lower_R_start_mm or 0.0
    shift_L_y_start = args.shift_L_y_start_mm or 0.0
    shift_R_y_start = args.shift_R_y_start_mm or 0.0
    if (args.lateral_spread_mm != 0.0 or lower_L != 0.0 or lower_R != 0.0
            or lower_L_start != 0.0 or lower_R_start != 0.0
            or shift_L_y_start != 0.0 or shift_R_y_start != 0.0):
        spread_half_y = args.lateral_spread_mm / 2.0 * 1e-3
        # Per-frame z decay helpers — return 1.0 at t_first, 0.0 at t_last, linear in between.
        def decay_lower(vmask, t):
            if not vmask.any():
                return 0.0
            t_first = int(np.where(vmask)[0][0])
            t_last  = int(np.where(vmask)[0][-1])
            if t <= t_first or t_last == t_first:
                return 1.0
            if t >= t_last:
                return 0.0
            return (t_last - t) / (t_last - t_first)

        print(f"lateral mode: spread={args.lateral_spread_mm:.1f} mm  "
              f"lower_L={lower_L:.1f} mm  lower_R={lower_R:.1f} mm  "
              f"lower_L_start={lower_L_start:.1f} mm  lower_R_start={lower_R_start:.1f} mm")

        p_L_new = p_L.copy(); p_R_new = p_R.copy()
        for t in range(N):
            if valid_L[t]:
                dL = decay_lower(valid_L, t)
                shift_z = -(lower_L + lower_L_start * (dL if lower_L_start != 0.0 else 0.0)) * 1e-3
                shift_y = spread_half_y + shift_L_y_start * (dL if shift_L_y_start != 0.0 else 0.0) * 1e-3
                p_L_new[t] = p_L[t] + np.array([0.0, shift_y, shift_z])
            if valid_R[t]:
                dR = decay_lower(valid_R, t)
                shift_z = -(lower_R + lower_R_start * (dR if lower_R_start != 0.0 else 0.0)) * 1e-3
                shift_y = -spread_half_y + shift_R_y_start * (dR if shift_R_y_start != 0.0 else 0.0) * 1e-3
                p_R_new[t] = p_R[t] + np.array([0.0, shift_y, shift_z])
        d_orig = np.linalg.norm(p_L - p_R, axis=1)
        d_new  = np.linalg.norm(p_L_new - p_R_new, axis=1)
        both   = valid_L & valid_R
        if both.any():
            print(f"inter-arm 3D distance (mm)   orig: min={d_orig[both].min()*1000:.0f}  mean={d_orig[both].mean()*1000:.0f}  max={d_orig[both].max()*1000:.0f}")
            print(f"                             new:  min={d_new[both].min()*1000:.0f}  mean={d_new[both].mean()*1000:.0f}  max={d_new[both].max()*1000:.0f}")
        # Safety: warn if the new L or R z minimum is below the table (z=0).
        # The gripper extends ~9 cm below the wrist when pointing down, so a
        # wrist below ~90 mm typically means the gripper tip is colliding with
        # the table.
        Lz_min = p_L_new[valid_L, 2].min()
        Rz_min = p_R_new[valid_R, 2].min()
        TIP_LEN_M = 0.09
        print(f"min wrist z (mm):  L = {Lz_min*1000:.1f}   R = {Rz_min*1000:.1f}   (table at z=0)")
        if Lz_min < TIP_LEN_M:
            print(f"  ⚠ L wrist min ({Lz_min*1000:.1f}mm) < gripper length (90mm) → L gripper TIP likely at or below table. Raise L or reduce lower_L.")
        if Rz_min < TIP_LEN_M:
            print(f"  ⚠ R wrist min ({Rz_min*1000:.1f}mm) < gripper length (90mm) → R gripper TIP likely at or below table. Raise R or reduce lower_R.")
        out_bundle = dict(bundle)
        out_bundle["p_L_pre_contact_adj"] = bundle["p_L"]
        out_bundle["p_R_pre_contact_adj"] = bundle["p_R"]
        out_bundle["p_L"] = p_L_new
        out_bundle["p_R"] = p_R_new
        out_bundle["contact_adj_lateral_spread_m"]   = np.float64(args.lateral_spread_mm * 1e-3)
        out_bundle["contact_adj_lower_L_m"]          = np.float64(lower_L * 1e-3)
        out_bundle["contact_adj_lower_R_m"]          = np.float64(lower_R * 1e-3)
        out_bundle["contact_adj_lower_L_start_m"]    = np.float64(lower_L_start * 1e-3)
        out_bundle["contact_adj_lower_R_start_m"]    = np.float64(lower_R_start * 1e-3)
        # Tag = "spread{S}" + per-arm "lowerL{V}" / "lowerLstart{V}" if nonzero.
        parts = [f"spread{int(args.lateral_spread_mm)}"]
        if lower_L_start != 0.0:
            parts.append(f"lowerLstart{int(lower_L_start)}")
        elif lower_L != 0.0:
            parts.append(f"lowerL{int(lower_L)}")
        if lower_R_start != 0.0:
            parts.append(f"lowerRstart{int(lower_R_start)}")
        elif lower_R != 0.0:
            parts.append(f"lowerR{int(lower_R)}")
        tag = args.tag or "_".join(parts)
        out_npz = out_dir / f"trajectory_contact_adj_{tag}.npz"
        np.savez_compressed(out_npz, **out_bundle)
        print(f"\nsaved {out_npz}")
        print(f"\nNext: re-run sim IK on the lateral bundle:")
        print(f"  python replay_trossen_ik.py \\")
        print(f"    --in_npz {out_npz} \\")
        print(f"    --scene_xml /home/bruce/Aloha_real/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/scene_joint.xml \\")
        print(f"    --record_mp4 {out_dir / f'trossen_replay_contact_adj_{tag}.mp4'} \\")
        print(f"    --position_only --warmup_seconds 2.0 --gripper_offset_m 0.09")
        return

    # ── Target distance d_k ─────────────────────────────────────────────────
    if args.distance_mm is not None:
        d_k = args.distance_mm * 1e-3
        tag = args.tag or f"d{args.distance_mm:.1f}mm_{args.mode}"
        print(f"using --distance-mm override: d_k = {d_k*1000:.3f} mm")
    else:
        d_k = D1_M * (GAMMA ** (args.iteration - 1))
        tag = args.tag or f"iter{args.iteration}_{args.mode}"
        print(f"iteration k={args.iteration}: d_k = {d_k*1000:.3f} mm "
              f"(= 5.000 * 0.85^{args.iteration - 1})")

    # ── Identify primary / secondary ────────────────────────────────────────
    if args.primary == "right":
        p_pri, valid_pri = p_R, valid_R
        p_sec, valid_sec = p_L, valid_L
        pri_name, sec_name = "R", "L"
    else:
        p_pri, valid_pri = p_L, valid_L
        p_sec, valid_sec = p_R, valid_R
        pri_name, sec_name = "L", "R"

    both_valid = valid_pri & valid_sec
    if not both_valid.any():
        raise RuntimeError("no jointly-valid frames in trajectory")
    t_s = int(np.where(both_valid)[0][0])
    print(f"t_s = {t_s}  (first jointly-valid frame)")

    # ── Bilateral mode short-circuit ────────────────────────────────────────
    if args.apply_to == "both":
        if args.mode != "translate":
            print("--apply-to both forces --mode translate (scale would couple the arms).")
        # Each arm gets its own translation toward its own closest box point so
        # both end up d_k away. This is the bilateral extension to the paper —
        # useful when the demo had only one hand on the box and we need to
        # bring both grippers in (or push both out) by independent shifts.
        p_L_new = p_L.copy(); p_R_new = p_R.copy()
        deltas = {}
        for label, vmask, parr, p_new in [("L", valid_L, p_L, p_L_new),
                                          ("R", valid_R, p_R, p_R_new)]:
            t_first = int(np.where(vmask)[0][0]) if vmask.any() else 0
            c_arm, D_arm = closest_point(parr[t_first], obj)
            if D_arm < 1e-9:
                print(f"  {label}: D_orig = 0 (already on box); skipping.")
                deltas[label] = np.zeros(3); continue
            dir_arm = (c_arm - parr[t_first]) / D_arm
            shift   = (D_arm - d_k) * dir_arm     # signed: + toward box, − away
            deltas[label] = shift
            for t in range(N):
                if vmask[t]:
                    p_new[t] = parr[t] + shift
            sign = "toward" if (D_arm > d_k) else "away from"
            print(f"  {label}: t_first={t_first}  D_orig={D_arm*1000:.1f}mm  d_k={d_k*1000:.1f}mm  "
                  f"shift={sign} box by {abs(D_arm - d_k)*1000:.1f}mm  vec={shift.round(4).tolist()}")

        # Save with bilateral suffix
        out_bundle = dict(bundle)
        out_bundle["p_L_pre_contact_adj"] = bundle["p_L"]
        out_bundle["p_R_pre_contact_adj"] = bundle["p_R"]
        out_bundle["p_L"] = p_L_new
        out_bundle["p_R"] = p_R_new
        out_bundle["contact_adj_iteration"]  = np.int64(args.iteration if args.distance_mm is None else 0)
        out_bundle["contact_adj_d_k_m"]      = np.float64(d_k)
        out_bundle["contact_adj_primary"]    = np.asarray("both")
        out_bundle["contact_adj_object_ply"] = np.asarray(str(args.object_ply))
        out_bundle["contact_adj_delta_L_m"]  = deltas["L"]
        out_bundle["contact_adj_delta_R_m"]  = deltas["R"]
        out_npz = out_dir / f"trajectory_contact_adj_{tag}_both.npz"
        np.savez_compressed(out_npz, **out_bundle)
        print(f"\nsaved {out_npz}")

        # New inter-arm distance summary
        d_orig = np.linalg.norm(p_L - p_R, axis=1)
        d_new  = np.linalg.norm(p_L_new - p_R_new, axis=1)
        both   = valid_L & valid_R
        if both.any():
            print(f"inter-arm distance (mm)   orig: min={d_orig[both].min()*1000:.0f}  mean={d_orig[both].mean()*1000:.0f}  max={d_orig[both].max()*1000:.0f}")
            print(f"                          new:  min={d_new[both].min()*1000:.0f}  mean={d_new[both].mean()*1000:.0f}  max={d_new[both].max()*1000:.0f}")

        # Next-step hint
        print(f"\nNext: re-run sim IK on the bilateral bundle:")
        print(f"  python replay_trossen_ik.py \\")
        print(f"    --in_npz {out_npz} \\")
        print(f"    --scene_xml /home/bruce/Aloha_real/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/scene_joint.xml \\")
        print(f"    --record_mp4 {out_dir / f'trossen_replay_contact_adj_{tag}_both.mp4'} \\")
        print(f"    --position_only --warmup_seconds 2.0 --gripper_offset_m 0.09")
        return

    # ── Geometric adjustment ────────────────────────────────────────────────
    c_pri, D_orig = closest_point(p_pri[t_s], obj)
    print(f"\n{pri_name}[t_s] = {p_pri[t_s].round(3).tolist()}")
    print(f"closest object pt = {c_pri.round(3).tolist()}")
    print(f"D_orig = {D_orig*1000:.2f} mm     (target d_k = {d_k*1000:.2f} mm)")

    if D_orig < d_k:
        print(f"\nWARNING: D_orig ({D_orig*1000:.2f} mm) < d_k ({d_k*1000:.2f} mm).")
        print(f"  The {pri_name} arm starts CLOSER to the object than the safety margin.")
        print(f"  Adjusting would push it AWAY, which is not the paper's intent.")
        print(f"  Skipping — your trajectory already touches (or penetrates) the object.")
        return

    shift_dir = (c_pri - p_pri[t_s]) / D_orig
    shift_mag = D_orig - d_k
    new_p_pri_ts = p_pri[t_s] + shift_mag * shift_dir
    print(f"shift {pri_name}[t_s] by {shift_mag*1000:.2f} mm along "
          f"{shift_dir.round(3).tolist()}  →  {new_p_pri_ts.round(3).tolist()}")

    L_orig = float(np.linalg.norm(p_pri[t_s] - p_sec[t_s]))
    L_new  = float(np.linalg.norm(new_p_pri_ts - p_sec[t_s]))
    if L_orig < 1e-6:
        raise RuntimeError(f"inter-arm distance at t_s is ~0; can't compute scaling")
    s_k = L_new / L_orig
    print(f"inter-arm {pri_name}↔{sec_name} at t_s: {L_orig*1000:.1f} mm  →  {L_new*1000:.1f} mm     "
          f"s_k = {s_k:.4f}  (used only for --mode scale)")

    # ── Apply adjustment to every primary-valid frame ───────────────────────
    p_pri_new = p_pri.copy()
    if args.mode == "scale":
        # Paper Eq. 11 — uniform scaling relative to secondary arm.
        for t in range(N):
            if valid_pri[t]:
                p_pri_new[t] = p_sec[t] + s_k * (p_pri[t] - p_sec[t])
    else:  # translate
        # Constant translation — preserves the demo's inter-arm distance at every frame.
        delta = new_p_pri_ts - p_pri[t_s]
        for t in range(N):
            if valid_pri[t]:
                p_pri_new[t] = p_pri[t] + delta
        print(f"--mode translate: shift = {(delta * 1000).round(2).tolist()} mm "
              f"(applied to every {pri_name}-valid frame)")
    print(f"--mode {args.mode}")

    delta = np.linalg.norm(p_pri_new - p_pri, axis=1)
    valid_delta = delta[valid_pri]
    print(f"\nprimary-arm shift across {int(valid_pri.sum())} valid frames:")
    print(f"  mean = {valid_delta.mean()*1000:.2f} mm")
    print(f"  max  = {valid_delta.max()*1000:.2f} mm  at frame {int(np.argmax(delta))}")
    print(f"  min  = {valid_delta.min()*1000:.2f} mm")

    # ── Save adjusted bundle ────────────────────────────────────────────────
    out_bundle = dict(bundle)  # shallow copy; we'll overwrite p_pri
    if args.primary == "right":
        out_bundle["p_R_pre_contact_adj"] = bundle["p_R"]  # archive original
        out_bundle["p_R"] = p_pri_new
    else:
        out_bundle["p_L_pre_contact_adj"] = bundle["p_L"]
        out_bundle["p_L"] = p_pri_new
    out_bundle["contact_adj_iteration"]   = np.int64(args.iteration)
    out_bundle["contact_adj_d_k_m"]       = np.float64(d_k)
    out_bundle["contact_adj_s_k"]         = np.float64(s_k)
    out_bundle["contact_adj_primary"]     = np.asarray(args.primary)
    out_bundle["contact_adj_object_ply"]  = np.asarray(str(args.object_ply))
    out_npz = out_dir / f"trajectory_contact_adj_{tag}.npz"
    np.savez_compressed(out_npz, **out_bundle)
    print(f"\nsaved {out_npz}")

    # ── Visualization ───────────────────────────────────────────────────────
    rng = np.random.default_rng(0)
    obj_sub = obj[rng.choice(len(obj), size=min(2500, len(obj)), replace=False)]

    fig = plt.figure(figsize=(13, 5.5))
    titles = [
        f"BEFORE — {pri_name}[t_s] {D_orig*1000:.1f} mm from object",
        f"AFTER  — k={args.iteration}, d_k={d_k*1000:.2f} mm, s_k={s_k:.4f}",
    ]
    p_pris = [p_pri, p_pri_new]
    starts = [p_pri[t_s], new_p_pri_ts]

    # Shared bounds so the two panels are visually comparable
    allp = np.concatenate(
        [p_pri[valid_pri], p_pri_new[valid_pri], p_sec[valid_sec], obj_sub], axis=0)
    mn, mx = allp.min(0) - 0.03, allp.max(0) + 0.03

    for col in range(2):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        ax.scatter(obj_sub[:, 0], obj_sub[:, 1], obj_sub[:, 2],
                   c="saddlebrown", s=2, alpha=0.4, label="object")
        ax.plot(p_pris[col][valid_pri, 0], p_pris[col][valid_pri, 1], p_pris[col][valid_pri, 2],
                color=("steelblue" if col == 0 else "crimson"), lw=1.2,
                label=f"{pri_name} {'orig' if col == 0 else 'adjusted'}")
        ax.plot(p_sec[valid_sec, 0], p_sec[valid_sec, 1], p_sec[valid_sec, 2],
                "g-", lw=1.0, label=f"{sec_name} (unchanged)")
        ax.scatter(*starts[col], c=("steelblue" if col == 0 else "crimson"),
                   s=50, marker="o", label=f"{pri_name}[t_s]")
        ax.scatter(*p_sec[t_s], c="green", s=50, marker="o", label=f"{sec_name}[t_s]")
        ax.scatter(*c_pri, c="red", s=60, marker="x", label="closest obj pt")
        ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[1], mx[1]); ax.set_zlim(mn[2], mx[2])
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
        ax.view_init(elev=22, azim=-65)
        ax.set_title(titles[col], fontsize=10)
        ax.legend(loc="upper right", fontsize=7)

    fig.suptitle(
        f"BiNoMaP Stage 2b contact adjustment — primary={args.primary}, "
        f"smoothed={args.smoothed.parent.name}/{args.smoothed.name}",
        fontsize=11,
    )
    fig.tight_layout()
    out_png = out_dir / f"trajectory_contact_adj_{tag}.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"saved {out_png}")

    # ── Next steps hint ─────────────────────────────────────────────────────
    print(f"\nNext: re-run the sim IK on the adjusted bundle:")
    print(f"  python replay_trossen_ik.py \\")
    print(f"    --in_npz {out_npz} \\")
    print(f"    --scene_xml /home/bruce/Aloha_real/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/scene_joint.xml \\")
    print(f"    --record_mp4 {out_dir / f'trossen_replay_contact_adj_{tag}.mp4'} \\")
    print(f"    --position_only --warmup_seconds 2.0")
    print(f"\nThen on real robot:")
    print(f"  python replay_real.py --log {out_dir / 'trossen_replay_ik_log.npz'} --max-step-delta 0.2")


if __name__ == "__main__":
    main()
