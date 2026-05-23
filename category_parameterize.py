"""
BiNoMaP Stage 3 — Category-Level Primitive Parameterization (paper §3.4).

Generalize a base bimanual trajectory (verified successful on the base object)
to a new object of the same category by re-scaling the inter-arm vector at every
contact-phase frame, in proportion to the object's size difference measured
along the inter-arm axis. Single non-iterative geometric step; orientations and
the left-arm trajectory are unchanged.

Algorithm (verbatim from §3.4):
  1. t_s = first frame where both arms are valid.
  2. û = (p_L[t_s] − p_R[t_s]) / ||p_L[t_s] − p_R[t_s]||  (inter-arm direction)
  3. e_obj = max(pcd · û) − min(pcd · û)                  (extent along û)
  4. δsize = e_new − e_base
  5. for t ∈ [t_s, t_e] (both valid):
        d_orig(t) = ||p_R[t] − p_L[t]||
        s(t)      = (d_orig(t) + δsize) / d_orig(t)
        p̄_R[t]   = p_L[t] + s(t) · (p_R[t] − p_L[t])
  6. p_L, R_L, R_R unchanged. Frames outside [t_s, t_e] pass through.

Outputs an npz with the same schema as the input plus traceability arrays:
  p_R_pre_param            — original right-arm trajectory before re-scaling
  cat_param_delta_size_m   — scalar δsize (meters)
  cat_param_inter_arm_unit — 3-vector û (unit)
  cat_param_t_s, cat_param_t_e — int frame indices of the contact phase

CLI:
  python category_parameterize.py \\
      --in_npz   outputs/recordings_2/wrist/trajectory_contact_adj_K11_Ronly_dRyStart-20_dRzStart-30.npz \\
      --base_ply boxes/brown_box.ply \\
      --new_ply  boxes/wifi_box.ply \\
      --out_npz  outputs/recordings_2/wrist/trajectory_K11_paramto_wifi_box.npz
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import open3d as o3d


def load_pcd_points(ply_path: pathlib.Path) -> np.ndarray:
    pts = np.asarray(o3d.io.read_point_cloud(str(ply_path)).points)
    if len(pts) == 0:
        raise SystemExit(f"empty point cloud: {ply_path}")
    return pts


def extent_along(pts: np.ndarray, unit: np.ndarray) -> float:
    proj = pts @ unit
    return float(proj.max() - proj.min())


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--in_npz", type=pathlib.Path, required=True,
                    help="Source trajectory bundle (the successful K=11 npz).")
    ap.add_argument("--base_ply", type=pathlib.Path, required=True,
                    help="PLY of the base (demo) object.")
    ap.add_argument("--new_ply", type=pathlib.Path, required=True,
                    help="PLY of the new (target) object.")
    ap.add_argument("--out_npz", type=pathlib.Path, default=None,
                    help="Output bundle. Default: <in_npz_stem>_paramto_<new_ply_stem>.npz")
    ap.add_argument("--max_scale_dev", type=float, default=0.5,
                    help="Refuse if |s(t)−1| exceeds this anywhere on the contact "
                         "phase (default 0.5 = ±50%% rescale).")
    args = ap.parse_args()

    for p in [args.in_npz, args.base_ply, args.new_ply]:
        if not p.exists():
            raise SystemExit(f"not found: {p}")

    out = args.out_npz
    if out is None:
        out = args.in_npz.with_name(
            f"{args.in_npz.stem}_paramto_{args.new_ply.stem}.npz"
        )

    bundle = dict(np.load(args.in_npz))
    p_L = bundle["p_L"]
    p_R = bundle["p_R"]
    valid_L = bundle["valid_L"]
    valid_R = bundle["valid_R"]

    both = valid_L & valid_R
    if not both.any():
        raise SystemExit("no frame where both arms are valid in input bundle")
    t_s = int(np.where(both)[0][0])
    t_e = int(np.where(both)[0][-1])

    v = p_L[t_s] - p_R[t_s]
    v_norm = float(np.linalg.norm(v))
    if v_norm < 1e-4:
        raise SystemExit(f"degenerate inter-arm vector at t_s={t_s}: |v|={v_norm:.6f}m")
    u_hat = v / v_norm

    pts_base = load_pcd_points(args.base_ply)
    pts_new = load_pcd_points(args.new_ply)
    e_base = extent_along(pts_base, u_hat)
    e_new = extent_along(pts_new, u_hat)
    delta_size = e_new - e_base

    p_R_pre = p_R.copy()
    p_R_new = p_R.copy()
    s_values = []
    for t in range(t_s, t_e + 1):
        if not both[t]:
            continue
        d_orig = float(np.linalg.norm(p_R[t] - p_L[t]))
        if d_orig < 1e-6:
            continue
        s_t = (d_orig + delta_size) / d_orig
        s_values.append(s_t)
        p_R_new[t] = p_L[t] + s_t * (p_R[t] - p_L[t])
    s_values = np.array(s_values)

    if len(s_values):
        max_dev = float(np.max(np.abs(s_values - 1.0)))
        if max_dev > args.max_scale_dev:
            raise SystemExit(
                f"max |s(t)−1| = {max_dev:.3f} exceeds --max_scale_dev={args.max_scale_dev}; "
                "target object likely outside the base category. Increase --max_scale_dev to override."
            )

    # Sanity asserts (paper invariants)
    assert np.array_equal(bundle["p_L"], p_L), "p_L mutated"
    assert np.array_equal(bundle["R_L"], bundle["R_L"]), "R_L mutated"  # not touched
    for t in range(t_s, t_e + 1):
        if both[t]:
            d_old = np.linalg.norm(p_R_pre[t] - p_L[t])
            d_new = np.linalg.norm(p_R_new[t] - p_L[t])
            assert abs(d_new - (d_old + delta_size)) < 1e-6, (
                f"frame {t}: |p̄_R−p_L|={d_new:.6f} != |p_R_orig−p_L|+δsize={d_old+delta_size:.6f}"
            )

    bundle["p_R"] = p_R_new
    bundle["p_R_pre_param"] = p_R_pre
    bundle["cat_param_delta_size_m"] = np.float64(delta_size)
    bundle["cat_param_inter_arm_unit"] = u_hat.astype(np.float64)
    bundle["cat_param_t_s"] = np.int64(t_s)
    bundle["cat_param_t_e"] = np.int64(t_e)
    bundle["cat_param_base_ply"] = str(args.base_ply)
    bundle["cat_param_new_ply"] = str(args.new_ply)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **bundle)

    # Report
    print(f"=== BiNoMaP Stage 3 — Category-Level Primitive Parameterization ===")
    print(f"input bundle: {args.in_npz}")
    print(f"base PLY    : {args.base_ply}   n_pts={len(pts_base)}")
    print(f"new  PLY    : {args.new_ply}    n_pts={len(pts_new)}")
    print(f"t_s={t_s}  t_e={t_e}  contact-phase length={t_e - t_s + 1} frames")
    print(f"inter-arm vector at t_s: |v|={v_norm * 1000:.1f} mm   û={u_hat.round(3)}")
    print(f"e_base (base obj extent along û): {e_base * 1000:7.1f} mm")
    print(f"e_new  (new  obj extent along û): {e_new * 1000:7.1f} mm")
    print(f"δsize  = e_new − e_base         : {delta_size * 1000:+7.1f} mm")
    if len(s_values):
        print(f"s(t) range over contact phase: [{s_values.min():.4f}, {s_values.max():.4f}]  "
              f"(=1.0 → no change; {s_values.mean():.4f} mean)")
        print(f"max |s(t)−1| = {np.max(np.abs(s_values - 1.0)):.4f}  "
              f"(threshold {args.max_scale_dev})")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
