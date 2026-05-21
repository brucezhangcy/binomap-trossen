"""
Generate an MP4 animation of a bimanual trajectory over time.

Each frame of the animation = one grid timestamp. Shows:
- accumulated trajectory up to time t (line)
- current EE position at time t (bold marker)
- gripper-frame triad (x/y/z basis vectors) at current pose
- elapsed time + frame index in the title

Works on both trajectory.npz and trajectory_smoothed.npz.

Usage:
  python animate_trajectory.py --in_npz outputs/recordings/wrist/trajectory.npz \
                               --out_mp4 outputs/recordings/wrist/trajectory.mp4 \
                               [--fps 30] [--triad_size 0.05]
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", required=True)
    ap.add_argument("--out_mp4", required=True)
    ap.add_argument("--fps", type=float, default=30.0, help="animation playback fps")
    ap.add_argument("--triad_size", type=float, default=0.05, help="triad arm length in meters")
    ap.add_argument("--stride", type=int, default=1, help="render every Nth grid frame (1=all)")
    args = ap.parse_args()

    d = np.load(args.in_npz)
    ts = d["ts_ms"]
    p_L, R_L, valid_L = d["p_L"], d["R_L"], d["valid_L"]
    p_R, R_R, valid_R = d["p_R"], d["R_R"], d["valid_R"]
    N = len(ts)

    # Determine plot extents from all valid samples + a small pad
    all_p = np.concatenate([p_L[valid_L], p_R[valid_R]], axis=0)
    pad = 0.05
    xmin, ymin, zmin = all_p.min(axis=0) - pad
    xmax, ymax, zmax = all_p.max(axis=0) + pad

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel("X world [m]")
    ax.set_ylabel("Y world [m]")
    ax.set_zlabel("Z world [m]")

    # Tabletop plane reference
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 4), np.linspace(ymin, ymax, 4))
    ax.plot_surface(xx, yy, 0.02 * np.ones_like(xx), alpha=0.08, color="gray")

    # Precompute cumulative line objects per side
    trail_L, = ax.plot([], [], [], "b-", linewidth=1.2, alpha=0.6, label="L trail")
    trail_R, = ax.plot([], [], [], "r-", linewidth=1.2, alpha=0.6, label="R trail")
    head_L, = ax.plot([], [], [], "bo", markersize=8, markeredgecolor="k", label="L EE")
    head_R, = ax.plot([], [], [], "ro", markersize=8, markeredgecolor="k", label="R EE")

    # Triad quivers (will be re-drawn each frame)
    triad_artists = []
    ax.legend(loc="upper right", fontsize=8)

    title = ax.set_title("")

    # NaN-fill invalid frames so matplotlib breaks the trail line at gaps
    # (otherwise it draws straight shortcuts across long detection holes —
    # visually looks like spikes radiating from the trajectory)
    p_L_nan = np.where(valid_L[:, None], p_L, np.nan)
    p_R_nan = np.where(valid_R[:, None], p_R, np.nan)

    frames_to_render = list(range(0, N, args.stride))
    t0 = ts[0]

    writer = FFMpegWriter(fps=args.fps, bitrate=2400)
    with writer.saving(fig, args.out_mp4, dpi=120):
        for f_idx, k in enumerate(frames_to_render):
            # accumulated trails: use NaN-filled arrays so the line breaks at
            # invalid frames instead of shortcutting straight across the gap
            pL_slice = p_L_nan[: k + 1]
            pR_slice = p_R_nan[: k + 1]
            trail_L.set_data(pL_slice[:, 0], pL_slice[:, 1])
            trail_L.set_3d_properties(pL_slice[:, 2])
            trail_R.set_data(pR_slice[:, 0], pR_slice[:, 1])
            trail_R.set_3d_properties(pR_slice[:, 2])

            # current EE head
            if valid_L[k]:
                head_L.set_data([p_L[k, 0]], [p_L[k, 1]])
                head_L.set_3d_properties([p_L[k, 2]])
            if valid_R[k]:
                head_R.set_data([p_R[k, 0]], [p_R[k, 1]])
                head_R.set_3d_properties([p_R[k, 2]])

            # remove previous triads
            for a in triad_artists:
                a.remove()
            triad_artists.clear()
            # draw triads at current valid pose (x=red, y=green, z=blue conventional)
            for valid, p_arr, R_arr, base_col in (
                (valid_L, p_L, R_L, "blue"),
                (valid_R, p_R, R_R, "red"),
            ):
                if not valid[k]:
                    continue
                o = p_arr[k]
                R = R_arr[k]
                for axis_i, col in enumerate(["#ff5050", "#50ff50", "#5050ff"]):
                    v = R[:, axis_i] * args.triad_size
                    art = ax.quiver(
                        o[0], o[1], o[2], v[0], v[1], v[2],
                        color=col, linewidth=1.5, arrow_length_ratio=0.25,
                    )
                    triad_artists.append(art)

            elapsed_s = (ts[k] - t0) / 1000.0
            title.set_text(
                f"{Path(args.in_npz).parent.parent.name}/{Path(args.in_npz).parent.name}  "
                f"frame {k}/{N-1}  t={elapsed_s:.2f}s  "
                f"L{'✓' if valid_L[k] else '✗'} R{'✓' if valid_R[k] else '✗'}"
            )
            writer.grab_frame()

    plt.close(fig)
    print(f"wrote {args.out_mp4}  ({len(frames_to_render)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
