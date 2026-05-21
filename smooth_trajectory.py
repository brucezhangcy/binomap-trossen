"""
BiNoMaP §3.3 Stage 2a — Motion Smoothness Optimization.

Takes a trajectory.npz produced by extract_trajectory.py and produces
trajectory_smoothed.npz with positions denoised and rotations regularized.

Per arm (independently):
  1. Plane fit  – least-squares plane through valid 3D positions (||n||=1).
  2. Project all valid positions onto that plane → 2D in-plane coords.
  3. Cubic B-spline smoothing on (x_2d(t), y_2d(t)) → smoothed in-plane curve.
  4. Lift back to 3D (smoothed positions are exactly coplanar).
  5. Anchor selection  𝒦 = {start, end} ∪ top-n intermediate frames with
     smallest positional residual (raw vs smoothed). Default top-n=3 (paper Table 3).
  6. SLERP between consecutive anchors → smoothed rotations.

Usage:
  python smooth_trajectory.py \
      --in_npz outputs/recordings/wrist/trajectory.npz \
      --out_dir outputs/recordings/wrist
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
from scipy.interpolate import splev, splrep

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse SO(3) helpers from extract_trajectory.py
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_trajectory import quat_to_rotmat, rotmat_to_quat, slerp_quat


# ---------------- plane fit + in-plane basis ----------------
def fit_plane(P):
    """SVD-based least-squares plane fit.
    P: (N, 3). Returns (n, b, origin) with n unit, n·x + b = 0 the plane eqn.
    origin = mean(P) (used as in-plane origin)."""
    origin = P.mean(axis=0)
    Pc = P - origin
    _, _, Vt = np.linalg.svd(Pc, full_matrices=False)
    n = Vt[-1]
    n = n / max(np.linalg.norm(n), 1e-12)
    b = -float(n @ origin)
    return n, b, origin


def plane_basis(n):
    """Return two orthonormal in-plane basis vectors (u, v) given normal n."""
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, helper)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


# ---------------- position smoothing ----------------
def smooth_positions(P, t_ms, smoothing=None, sigma_m=0.005):
    """Plane fit + project + cubic B-spline smooth + lift back to 3D.

    P:        (N, 3) valid positions in world frame [m]
    t_ms:     (N,) timestamps in ms (B-spline parameter; converted to seconds)
    smoothing:  scipy splrep `s`. If None, auto = N * sigma_m**2 (≈ N point-noise variance)
    sigma_m:    expected per-point noise [m] used for auto smoothing

    Returns: (P_smooth (N,3), info dict with plane parameters and chosen s)
    """
    N = P.shape[0]
    n, b, origin = fit_plane(P)
    u, v = plane_basis(n)
    Pc = P - origin
    x_2d = Pc @ u
    y_2d = Pc @ v

    t_s = (t_ms - t_ms[0]) / 1000.0  # seconds, anchored at 0 for numerical stability
    if smoothing is None:
        smoothing = N * sigma_m ** 2

    tck_x = splrep(t_s, x_2d, k=3, s=smoothing)
    tck_y = splrep(t_s, y_2d, k=3, s=smoothing)
    x_smooth = splev(t_s, tck_x)
    y_smooth = splev(t_s, tck_y)

    P_smooth = origin + x_smooth[:, None] * u + y_smooth[:, None] * v
    residual = np.linalg.norm(P_smooth - P, axis=1)
    info = {
        "plane_normal": n,
        "plane_offset": b,
        "plane_origin": origin,
        "smoothing_s": smoothing,
        "in_plane_thickness_mm": float(np.std(Pc @ n) * 1000),  # how planar the raw is
        "residual_mean_mm": float(residual.mean() * 1000),
        "residual_max_mm": float(residual.max() * 1000),
    }
    return P_smooth, info


# ---------------- rotation smoothing ----------------
def select_anchors(P_raw, P_smooth, top_n=3):
    """𝒦 = {start, end} ∪ top-n intermediate frames with smallest positional error."""
    N = P_raw.shape[0]
    if N <= 2:
        return np.arange(N)
    err = np.linalg.norm(P_raw - P_smooth, axis=1)
    middle_indices = np.arange(1, N - 1)
    middle_err = err[1:-1]
    if len(middle_indices) <= top_n:
        intermediate = middle_indices
    else:
        order = np.argsort(middle_err)[:top_n]
        intermediate = np.sort(middle_indices[order])
    return np.unique(np.concatenate([[0], intermediate, [N - 1]]))


def smooth_rotations(Q, t_ms, anchors):
    """SLERP between consecutive anchor frames. Q: (N, 4) xyzw."""
    Q_smooth = Q.copy()
    for i in range(len(anchors) - 1):
        a, b = anchors[i], anchors[i + 1]
        if b - a <= 1:
            continue
        t_a, t_b = t_ms[a], t_ms[b]
        q_a, q_b = Q[a], Q[b]
        denom = max(t_b - t_a, 1e-6)
        for k in range(a + 1, b):
            alpha = (t_ms[k] - t_a) / denom
            Q_smooth[k] = slerp_quat(q_a, q_b, alpha)
    return Q_smooth


# ---------------- single-arm wrapper ----------------
def smooth_arm(p_raw, R_raw, q_raw, valid_mask, ts_ms, smoothing=None, sigma_m=0.005, top_n=3):
    """Apply §3.3 smoothing to one arm. Only operates on valid frames; invalid stays as-is."""
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) < 5:
        return p_raw.copy(), R_raw.copy(), q_raw.copy(), None, np.array([], dtype=int)

    P_v = p_raw[valid_idx]
    Q_v = q_raw[valid_idx]
    t_v = ts_ms[valid_idx]

    P_smooth_v, info = smooth_positions(P_v, t_v, smoothing=smoothing, sigma_m=sigma_m)
    anchors_local = select_anchors(P_v, P_smooth_v, top_n=top_n)
    Q_smooth_v = smooth_rotations(Q_v, t_v, anchors_local)

    p_smooth = p_raw.copy()
    R_smooth = R_raw.copy()
    q_smooth = q_raw.copy()
    for i, idx in enumerate(valid_idx):
        p_smooth[idx] = P_smooth_v[i]
        q_smooth[idx] = Q_smooth_v[i]
        R_smooth[idx] = quat_to_rotmat(Q_smooth_v[i])

    anchors_global = valid_idx[anchors_local]
    return p_smooth, R_smooth, q_smooth, info, anchors_global


# ---------------- IO + plot ----------------
def write_csv(path, b):
    with open(path, "w") as f:
        f.write(
            "ts_ms,"
            "valid_L,interp_L,side_L,"
            "pLx,pLy,pLz,qLx,qLy,qLz,qLw,"
            "pLx_raw,pLy_raw,pLz_raw,"
            "valid_R,interp_R,side_R,"
            "pRx,pRy,pRz,qRx,qRy,qRz,qRw,"
            "pRx_raw,pRy_raw,pRz_raw\n"
        )
        for k in range(len(b["ts_ms"])):
            f.write(
                f"{b['ts_ms'][k]:.3f},"
                f"{int(b['valid_L'][k])},{int(b['interp_L'][k])},{int(b['side_L_detected'][k])},"
                f"{b['p_L'][k,0]:.6f},{b['p_L'][k,1]:.6f},{b['p_L'][k,2]:.6f},"
                f"{b['q_L'][k,0]:.6f},{b['q_L'][k,1]:.6f},{b['q_L'][k,2]:.6f},{b['q_L'][k,3]:.6f},"
                f"{b['p_L_raw'][k,0]:.6f},{b['p_L_raw'][k,1]:.6f},{b['p_L_raw'][k,2]:.6f},"
                f"{int(b['valid_R'][k])},{int(b['interp_R'][k])},{int(b['side_R_detected'][k])},"
                f"{b['p_R'][k,0]:.6f},{b['p_R'][k,1]:.6f},{b['p_R'][k,2]:.6f},"
                f"{b['q_R'][k,0]:.6f},{b['q_R'][k,1]:.6f},{b['q_R'][k,2]:.6f},{b['q_R'][k,3]:.6f},"
                f"{b['p_R_raw'][k,0]:.6f},{b['p_R_raw'][k,1]:.6f},{b['p_R_raw'][k,2]:.6f}\n"
            )


def plot_compare(b, out_png, title, anchors_L=None, anchors_R=None):
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    if b["valid_L"].any():
        m = b["valid_L"]
        ax.scatter(b["p_L_raw"][m, 0], b["p_L_raw"][m, 1], b["p_L_raw"][m, 2],
                   c="#9bbcff", s=8, alpha=0.7, label="L raw")
        ax.plot(b["p_L"][m, 0], b["p_L"][m, 1], b["p_L"][m, 2],
                "b-", linewidth=1.8, label="L smoothed")
        if anchors_L is not None and len(anchors_L):
            ax.scatter(b["p_L"][anchors_L, 0], b["p_L"][anchors_L, 1], b["p_L"][anchors_L, 2],
                       c="b", s=60, marker="*", edgecolor="k", label="L anchors")
    if b["valid_R"].any():
        m = b["valid_R"]
        ax.scatter(b["p_R_raw"][m, 0], b["p_R_raw"][m, 1], b["p_R_raw"][m, 2],
                   c="#ffb09b", s=8, alpha=0.7, label="R raw")
        ax.plot(b["p_R"][m, 0], b["p_R"][m, 1], b["p_R"][m, 2],
                "r-", linewidth=1.8, label="R smoothed")
        if anchors_R is not None and len(anchors_R):
            ax.scatter(b["p_R"][anchors_R, 0], b["p_R"][anchors_R, 1], b["p_R"][anchors_R, 2],
                       c="r", s=60, marker="*", edgecolor="k", label="R anchors")
    xx, yy = np.meshgrid(np.linspace(-0.5, 0.5, 5), np.linspace(-0.5, 0.5, 5))
    ax.plot_surface(xx, yy, 0.02 * np.ones_like(xx), alpha=0.1, color="gray")
    ax.set_xlabel("X world [m]")
    ax.set_ylabel("Y world [m]")
    ax.set_zlabel("Z world [m]")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--smoothing", type=float, default=None,
                    help="scipy splrep s. Default: N * sigma_m**2")
    ap.add_argument("--sigma_m", type=float, default=0.005,
                    help="expected per-point noise in meters (default 5 mm)")
    ap.add_argument("--top_n", type=int, default=3,
                    help="intermediate anchor count (paper Table 3 best=3)")
    args = ap.parse_args()

    print("=== BiNoMaP §3.3 Motion Smoothness Optimization ===")
    print(f"  in_npz:    {args.in_npz}")
    print(f"  out_dir:   {args.out_dir}")
    print(f"  sigma_m:   {args.sigma_m}")
    print(f"  smoothing: {'auto' if args.smoothing is None else args.smoothing}")
    print(f"  top_n:     {args.top_n}")

    data = np.load(args.in_npz)
    bundle = {k: data[k] for k in data.files}

    print("\n--- smoothing L arm ---")
    p_L_s, R_L_s, q_L_s, info_L, anchors_L = smooth_arm(
        bundle["p_L"], bundle["R_L"], bundle["q_L"], bundle["valid_L"], bundle["ts_ms"],
        smoothing=args.smoothing, sigma_m=args.sigma_m, top_n=args.top_n,
    )
    if info_L is not None:
        print(f"  plane normal:           {info_L['plane_normal']}")
        print(f"  raw in-plane thickness: {info_L['in_plane_thickness_mm']:.2f} mm  (smaller = more planar)")
        print(f"  smoothed residual mean: {info_L['residual_mean_mm']:.2f} mm   (raw→smooth)")
        print(f"  smoothed residual max:  {info_L['residual_max_mm']:.2f} mm")
        print(f"  smoothing s used:       {info_L['smoothing_s']:.2e}")
        print(f"  anchors (global idx):   {anchors_L.tolist()}")

    print("\n--- smoothing R arm ---")
    p_R_s, R_R_s, q_R_s, info_R, anchors_R = smooth_arm(
        bundle["p_R"], bundle["R_R"], bundle["q_R"], bundle["valid_R"], bundle["ts_ms"],
        smoothing=args.smoothing, sigma_m=args.sigma_m, top_n=args.top_n,
    )
    if info_R is not None:
        print(f"  plane normal:           {info_R['plane_normal']}")
        print(f"  raw in-plane thickness: {info_R['in_plane_thickness_mm']:.2f} mm")
        print(f"  smoothed residual mean: {info_R['residual_mean_mm']:.2f} mm")
        print(f"  smoothed residual max:  {info_R['residual_max_mm']:.2f} mm")
        print(f"  smoothing s used:       {info_R['smoothing_s']:.2e}")
        print(f"  anchors (global idx):   {anchors_R.tolist()}")

    out_bundle = dict(bundle)
    out_bundle["p_L_raw"] = bundle["p_L"]
    out_bundle["p_R_raw"] = bundle["p_R"]
    out_bundle["R_L_raw"] = bundle["R_L"]
    out_bundle["R_R_raw"] = bundle["R_R"]
    out_bundle["q_L_raw"] = bundle["q_L"]
    out_bundle["q_R_raw"] = bundle["q_R"]
    out_bundle["p_L"], out_bundle["R_L"], out_bundle["q_L"] = p_L_s, R_L_s, q_L_s
    out_bundle["p_R"], out_bundle["R_R"], out_bundle["q_R"] = p_R_s, R_R_s, q_R_s
    out_bundle["anchors_L"] = anchors_L
    out_bundle["anchors_R"] = anchors_R

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "trajectory_smoothed.npz", **out_bundle)
    write_csv(out_dir / "trajectory_smoothed.csv", out_bundle)
    title = f"BiNoMaP Stage 2a Smoothed — {Path(args.in_npz).parent.parent.name}/{Path(args.in_npz).parent.name}"
    plot_compare(out_bundle, out_dir / "trajectory_smoothed.png", title, anchors_L, anchors_R)

    print(f"\nsaved:")
    print(f"  {out_dir/'trajectory_smoothed.npz'}")
    print(f"  {out_dir/'trajectory_smoothed.csv'}")
    print(f"  {out_dir/'trajectory_smoothed.png'}")


if __name__ == "__main__":
    main()
