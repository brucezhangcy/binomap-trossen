"""Translate a trajectory bundle by Δ = (current_box_pose − success_box_pose).

BiNoMaP §3.3 mentions a VLM that computes (Δx, Δy) between the demo's object
pose and the new scene's object pose, then offsets the bimanual trajectory by
that delta so the same primitive runs on the new scene. We replace the VLM
with SAM2-based box capture (see [record_box_pose.py](record_box_pose.py)):
given a SUCCESS reference pose and a CURRENT box pose, compute Δ and apply
it as a rigid translation to p_L and p_R for every valid frame.

By default Δz is zeroed out (the table is fixed; only x/y position varies
between scene placements). Pass `--apply-dz` to keep the z component.

Output: a new bundle (same npz schema as the input) with `p_L_box_aligned`
and `p_R_box_aligned` overwriting `p_L`/`p_R`, plus metadata recording the
delta and the source pose JSONs. Originals preserved under `p_L_pre_align`
and `p_R_pre_align` for comparison.

CLI:
  python box_align_trajectory.py \\
      --in_npz  outputs/recordings_2/wrist/trajectory_contact_adj_K11_*.npz \\
      --success outputs/recordings_2/wrist/box_poses/success_K11_brown.json \\
      --current outputs/recordings_2/wrist/box_poses/current.json \\
      --out_npz outputs/recordings_2/wrist/trajectory_K11_box_aligned.npz

Then re-run the IK + replay chain:
  /home/bruce/miniconda3/envs/trossen_sim/bin/python replay_trossen_ik.py \\
      --in_npz outputs/recordings_2/wrist/trajectory_K11_box_aligned.npz \\
      --scene_xml /home/bruce/Aloha_real/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/scene_joint.xml \\
      --record_mp4 outputs/recordings_2/wrist/replay_K11_box_aligned.mp4 \\
      --rot-weight 0.2 --warmup_seconds 2.0 --gripper_offset_m 0.09

  python replay_real.py --log outputs/recordings_2/wrist/trossen_replay_ik_log.npz \\
      --max-step-delta 0.2
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def load_box_pose(path: pathlib.Path) -> np.ndarray:
    with path.open() as f:
        data = json.load(f)
    return np.asarray(data["box_aabb_center_xyz_m"], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_npz", type=pathlib.Path, required=True,
                    help="Trajectory bundle (e.g. trajectory_contact_adj_K11_*.npz)")
    ap.add_argument("--success", type=pathlib.Path, required=True,
                    help="JSON of the box pose at the SUCCESS run (record_box_pose.py output).")
    ap.add_argument("--current", type=pathlib.Path, required=True,
                    help="JSON of the box pose at the CURRENT run.")
    ap.add_argument("--out_npz", type=pathlib.Path, default=None,
                    help="Output bundle (default: <in_npz>_box_aligned.npz)")
    ap.add_argument("--apply-dz", action="store_true",
                    help="Apply the z component of Δ too. Default: zero Δz (table is fixed).")
    ap.add_argument("--max-delta-mm", type=float, default=200.0,
                    help="Refuse if |Δ| exceeds this many mm (sanity gate against using "
                         "a wildly mis-localized box). Default 200 mm.")
    args = ap.parse_args()

    if not args.in_npz.exists():
        raise SystemExit(f"input bundle not found: {args.in_npz}")
    if not args.success.exists():
        raise SystemExit(f"success pose JSON not found: {args.success}")
    if not args.current.exists():
        raise SystemExit(f"current pose JSON not found: {args.current}")

    p_success = load_box_pose(args.success)
    p_current = load_box_pose(args.current)
    delta = p_current - p_success
    if not args.apply_dz:
        delta[2] = 0.0

    delta_mm = float(np.linalg.norm(delta) * 1000)
    print(f"box pose Δ (current − success): {(delta * 1000).round(2).tolist()} mm")
    print(f"  success center: {(p_success * 1000).round(1).tolist()} mm")
    print(f"  current center: {(p_current * 1000).round(1).tolist()} mm")
    print(f"  Δ magnitude   : {delta_mm:.1f} mm   "
          f"({'apply_dz=ON' if args.apply_dz else 'Δz zeroed (--apply-dz to keep)'})")
    if delta_mm > args.max_delta_mm:
        raise SystemExit(
            f"|Δ|={delta_mm:.0f}mm exceeds --max-delta-mm={args.max_delta_mm:.0f}. "
            f"Box localization may be wrong. Re-capture, or pass a larger threshold.")

    # Load bundle, apply translation to valid frames
    bundle = dict(np.load(args.in_npz))
    if "p_L" not in bundle or "p_R" not in bundle:
        raise SystemExit("input bundle missing required p_L / p_R keys")
    vL = bundle["valid_L"].astype(bool)
    vR = bundle["valid_R"].astype(bool)
    print(f"\nbundle: N={bundle['p_L'].shape[0]}  valid_L={int(vL.sum())}  valid_R={int(vR.sum())}")

    bundle["p_L_pre_align"] = bundle["p_L"].copy()
    bundle["p_R_pre_align"] = bundle["p_R"].copy()
    p_L_new = bundle["p_L"].copy()
    p_R_new = bundle["p_R"].copy()
    p_L_new[vL] += delta
    p_R_new[vR] += delta
    bundle["p_L"] = p_L_new
    bundle["p_R"] = p_R_new

    # Stamp metadata
    bundle["box_align_delta_xyz_m"]  = delta.astype(np.float64)
    bundle["box_align_success_json"] = np.asarray(str(args.success.resolve()))
    bundle["box_align_current_json"] = np.asarray(str(args.current.resolve()))
    bundle["box_align_apply_dz"]     = np.bool_(args.apply_dz)

    out = args.out_npz or args.in_npz.with_name(args.in_npz.stem + "_box_aligned.npz")
    np.savez_compressed(out, **bundle)
    print(f"\nsaved {out}")
    print(f"  p_L_pre_align / p_R_pre_align preserved in the bundle for comparison.")


if __name__ == "__main__":
    main()
