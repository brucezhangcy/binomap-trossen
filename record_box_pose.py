"""Save a box's world-frame AABB center to a tagged JSON file.

Use case: BiNoMaP §3.3 talks about a VLM that computes (Δx, Δy) between the
demonstration's object pose and the current object pose, then offsets the
trajectory by that delta. We don't have a VLM — we have SAM2 +
[capture_box.py](boxes/capture_box.py) which gives us a per-capture point
cloud. This script extracts the box pose from any captured PLY and saves it
to a JSON pose-reference file.

Workflow:
  1. After a successful real-robot flip with a particular trajectory K, run
     `boxes/capture_box.py --name success_K11_brown --out-dir
     outputs/recordings_2/wrist/box_poses` to capture the box pose that
     worked. Then `record_box_pose.py --ply
     outputs/recordings_2/wrist/box_poses/success_K11_brown.ply --name
     success_K11_brown` to lock that as the success reference.
  2. Before each subsequent replay, recapture the current box (e.g. after
     the box has moved) and run `record_box_pose.py` to log its new pose.
  3. Feed both JSONs to `box_align_trajectory.py` to compute Δxy and produce
     a box-aligned trajectory bundle.

CLI:
  python record_box_pose.py \\
      --ply outputs/recordings_2/wrist/box_poses/success_K11_brown.ply \\
      --name success_K11_brown
  python record_box_pose.py \\
      --ply outputs/recordings_2/wrist/box_poses/current.ply --name current
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib

import numpy as np
import open3d as o3d


REPO_ROOT = pathlib.Path(__file__).resolve().parent


def aabb_center_extent(ply_path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (center_xyz, extent_xyz) of the PLY's axis-aligned bounding box."""
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        raise RuntimeError(f"empty point cloud at {ply_path}")
    aabb_min, aabb_max = pts.min(axis=0), pts.max(axis=0)
    return 0.5 * (aabb_min + aabb_max), aabb_max - aabb_min


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", type=pathlib.Path, required=True,
                    help="Path to the box PLY (e.g. produced by boxes/capture_box.py).")
    ap.add_argument("--name", type=str, required=True,
                    help="Pose label. Saved as <out-dir>/<name>.json. "
                         "Use a descriptive tag like 'success_K11_brown' or 'current'.")
    ap.add_argument("--out-dir", type=pathlib.Path, default=None,
                    help="Where to save the pose JSON. Default: same dir as --ply.")
    ap.add_argument("--note", type=str, default="",
                    help="Free-text note (e.g. 'success after K=11 box-flip on recordings_2').")
    args = ap.parse_args()

    if not args.ply.exists():
        raise SystemExit(f"PLY not found: {args.ply}")

    center, extent = aabb_center_extent(args.ply)
    out_dir = args.out_dir if args.out_dir is not None else args.ply.resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{args.name}.json"

    payload = {
        "name": args.name,
        "captured_at_iso": datetime.datetime.now().isoformat(timespec="seconds"),
        "source_ply": str(args.ply.resolve()),
        "box_aabb_center_xyz_m": [float(v) for v in center],
        "box_aabb_extent_xyz_m": [float(v) for v in extent],
        "note": args.note,
    }
    with out_json.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"saved {out_json}")
    print(f"  center (m): {center.round(4).tolist()}")
    print(f"  extent (m): {extent.round(4).tolist()}")
    if args.note:
        print(f"  note: {args.note}")


if __name__ == "__main__":
    main()
