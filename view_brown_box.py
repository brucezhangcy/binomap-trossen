"""Open `binomap/brownbox/brown_box.ply` in an Open3D window."""

from __future__ import annotations

import argparse
import pathlib

import open3d as o3d

DEFAULT_PLY = pathlib.Path(__file__).resolve().parent / "brownbox" / "brown_box.ply"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ply", type=pathlib.Path, default=DEFAULT_PLY)
    args = ap.parse_args()

    if not args.ply.exists():
        raise SystemExit(f"missing {args.ply}")

    pcd = o3d.io.read_point_cloud(str(args.ply))
    print(f"{args.ply}: {len(pcd.points):,} pts")
    o3d.visualization.draw_geometries([pcd], window_name=args.ply.name)


if __name__ == "__main__":
    main()
