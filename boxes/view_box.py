"""Open a saved box point cloud in an Open3D window.

Usage::

    python view_box.py --name brown_box       # opens brown_box.ply
    python view_box.py --name cyan_box        # opens cyan_box.ply
    python view_box.py --ply other/path.ply   # any PLY anywhere
"""

from __future__ import annotations

import argparse
import pathlib

import open3d as o3d

HERE = pathlib.Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", help="Box label, e.g. brown_box (resolves to <name>.ply next to this script).")
    ap.add_argument("--ply", type=pathlib.Path,
                    help="Explicit PLY path. Overrides --name.")
    args = ap.parse_args()

    if args.ply is None and args.name is None:
        raise SystemExit("pass either --name <label> or --ply <path>")
    ply = args.ply if args.ply is not None else (HERE / f"{args.name}.ply")
    if not ply.exists():
        raise SystemExit(f"missing {ply}")

    pcd = o3d.io.read_point_cloud(str(ply))
    print(f"{ply}: {len(pcd.points):,} pts")
    o3d.visualization.draw_geometries([pcd], window_name=ply.name)


if __name__ == "__main__":
    main()
