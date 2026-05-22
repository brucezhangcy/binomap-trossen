"""
Calibration tool for the paper-strict orientation pipeline.

What this verifies:
  Algorithm 1 (BiNoMaP §3.2) builds R_hand_world = [v_x | v_y | v_z] columns
  from the hand's 3D landmarks:
      v_z = palm normal       (where the palm faces / where the hand pushes)
      v_y = fingers open/close (perpendicular to palm, in finger plane)
      v_x = v_y × v_z         (right-hand frame completion)
  We want to map this to the Trossen gripper:
      Trossen link_6 local +x = gripper pointing direction (tool axis)
      Trossen link_6 local +y or +z = finger open/close
  So we drive the IK target with R_gripper_world = R_hand_world @ R_align, where
  R_align rotates the gripper's local axes onto the hand's local axes.

How to use:
  python calibrate_gripper_align.py \
      --in_npz outputs/recordings_1/wrist/trajectory_smoothed.npz \
      --align_euler_deg 0,-90,0 \
      --out outputs/recordings_1/wrist/orient_calib.png \
      --sample_every 30
  Inspect the plot. The BLACK gripper-pointing arrow should overlay the
  GREEN palm-normal arrow at every sampled frame and visually point toward
  where the hand is pushing the box. Tweak --align_euler_deg if it does not.

  Convention for --align_euler_deg "rx,ry,rz" (degrees, intrinsic ZYX):
      R_align = Rz(rz) @ Ry(ry) @ Rx(rx)
  Default 0,-90,0  → maps gripper-local +x onto +z (palm normal). This is
  the geometric prior — start here, then iterate visually if the arrows do
  not line up with the hand pushing direction.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


def euler_zyx_to_rotmat(rx_deg, ry_deg, rz_deg):
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def draw_frame_arrows(ax, origin, R, length=0.04, lw=1.5, alpha=0.9,
                      colors=("#ff5252", "#22cc55", "#3a82ff"),
                      label_axes=False):
    """Draw the three columns of R as RGB arrows from `origin`."""
    for j, c in enumerate(colors):
        v = R[:, j] * length
        ax.quiver(*origin, *v, color=c, linewidth=lw, alpha=alpha,
                  arrow_length_ratio=0.25)
        if label_axes:
            tip = origin + v
            ax.text(tip[0], tip[1], tip[2], "xyz"[j], color=c, fontsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--align_euler_deg", default="0,-90,0",
                    help="Trossen-gripper-local → hand-local rotation, intrinsic ZYX")
    ap.add_argument("--sample_every", type=int, default=30,
                    help="Draw arrows every N frames")
    ap.add_argument("--arrow_len_m", type=float, default=0.05)
    ap.add_argument("--arm", choices=["both", "L", "R"], default="both")
    ap.add_argument("--elev", type=float, default=22)
    ap.add_argument("--azim", type=float, default=-58)
    args = ap.parse_args()

    d = np.load(args.in_npz)
    p_L, R_L, vL = d["p_L"], d["R_L"], d["valid_L"]
    p_R, R_R, vR = d["p_R"], d["R_R"], d["valid_R"]
    N = len(p_L)
    rx, ry, rz = [float(x) for x in args.align_euler_deg.split(",")]
    R_align = euler_zyx_to_rotmat(rx, ry, rz)
    print(f"R_align (gripper-local → hand-local) for euler ZYX = ({rx},{ry},{rz}) deg:")
    print(R_align.round(3))
    # Sanity: which gripper-local axis maps to hand-local +z (palm normal)?
    g_in_hand = R_align.T @ np.array([0, 0, 1])
    print(f"hand-local +z (palm normal) lives along gripper-local axis: {g_in_hand.round(3)}")
    print(f"gripper-local +x maps to hand-local axis: {(R_align @ np.array([1,0,0])).round(3)}")
    print(f"   → if this ≈ (0,0,1), gripper-pointing matches palm normal ✓")

    arms = []
    if args.arm in ("both", "L"):
        arms.append(("L", p_L, R_L, vL, "#1565c0", "left"))
    if args.arm in ("both", "R"):
        arms.append(("R", p_R, R_R, vR, "#c62828", "right"))

    n_panels = len(arms)
    fig = plt.figure(figsize=(8 * n_panels, 9))
    L = args.arrow_len_m
    panels = []
    for pi, (tag, p, R, valid, col_path, label) in enumerate(arms):
        ax = fig.add_subplot(1, n_panels, pi + 1, projection="3d")
        panels.append(ax)
        idx_valid = np.where(valid)[0]
        if len(idx_valid):
            ax.plot(p[idx_valid, 0], p[idx_valid, 1], p[idx_valid, 2],
                    color=col_path, lw=1.2, alpha=0.5)
        for k in idx_valid[::args.sample_every]:
            R_hand = R[k]
            R_gripper = R_hand @ R_align
            draw_frame_arrows(ax, p[k], R_hand, length=L * 0.9,
                              colors=("#ff5252", "#22cc55", "#3a82ff"),
                              lw=1.8, alpha=0.85)
            gp = R_gripper @ np.array([1.0, 0, 0])
            ax.quiver(*p[k], *(gp * L * 1.4),
                      color="black", linewidth=2.5, arrow_length_ratio=0.18, alpha=0.95)
            ax.scatter(*p[k], color=col_path, s=18)
            ax.text(p[k, 0], p[k, 1], p[k, 2] + 0.005, f"f{k}", fontsize=7,
                    color=col_path)
        # zoom to data
        if len(idx_valid):
            pmin = p[idx_valid].min(0) - 0.08
            pmax = p[idx_valid].max(0) + 0.08
            ctr = 0.5 * (pmin + pmax)
            span = max((pmax - pmin).max(), 0.2)
            ax.set_xlim(ctr[0] - span / 2, ctr[0] + span / 2)
            ax.set_ylim(ctr[1] - span / 2, ctr[1] + span / 2)
            ax.set_zlim(ctr[2] - span / 2, ctr[2] + span / 2)
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
        ax.set_title(f"{label} arm     "
                     f"R_align ZYX=({rx:.0f},{ry:.0f},{rz:.0f})°")
        ax.set_box_aspect((1, 1, 1))
    # Single legend
    legend_h = [
        plt.Line2D([0], [0], color="#ff5252", lw=2, label="hand v_x"),
        plt.Line2D([0], [0], color="#22cc55", lw=2, label="hand v_y (fingers open/close)"),
        plt.Line2D([0], [0], color="#3a82ff", lw=2, label="hand v_z (palm normal = where hand pushes)"),
        plt.Line2D([0], [0], color="black",   lw=3, label="gripper pointing (= R_hand @ R_align @ +x_local)"),
    ]
    fig.legend(handles=legend_h, loc="upper center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 0.98))

    fig.suptitle(f"Algorithm-1 hand frame vs gripper-pointing axis     [{Path(args.in_npz).name}]",
                 y=0.995, fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=140)
    print(f"\nplot → {args.out}")

    # Stats: angular gap between gripper-pointing and palm normal across valid frames
    for tag, p, R, valid, _, label in arms:
        idx = np.where(valid)[0]
        if len(idx) == 0:
            continue
        cos_gap = []
        for k in idx:
            v_palm = R[k] @ np.array([0, 0, 1])
            v_grip = R[k] @ R_align @ np.array([1, 0, 0])
            cos_gap.append(np.dot(v_palm, v_grip))
        cos_gap = np.array(cos_gap)
        cos_gap = np.clip(cos_gap, -1, 1)
        ang = np.rad2deg(np.arccos(cos_gap))
        print(f"  {label} arm: gripper-pointing ↔ palm-normal angle: "
              f"mean={ang.mean():.1f}° median={np.median(ang):.1f}° max={ang.max():.1f}°  "
              f"(0° = aligned)")


if __name__ == "__main__":
    main()
