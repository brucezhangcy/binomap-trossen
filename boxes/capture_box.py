"""Capture both calibrated RealSense cameras, segment an object with SAM2,
and fuse the masked pixels into a world-frame point cloud.

Color-agnostic: SAM2 segments from a single click, not from color, so the
same script works for a brown box, black box, cyan box, plastic box, mug,
etc. Pass ``--name`` so the outputs are labelled.

Pipeline:
  1. Live-capture one aligned RGBD pair per camera serial in
     `caliberation/camera_extrinsics.json` (pyrealsense2 directly).
  2. For each camera:
       - show the RGB; user clicks the object once
       - SAM2 (hiera-large) predicts a per-pixel mask for that click
       - unproject only the masked pixels to a camera-frame point cloud
       - transform to world frame via the calibrated extrinsic
  3. Merge masked clouds, crop to a workspace AABB, keep the largest DBSCAN
     cluster (drops stray blobs if SAM grabs a shadow strip).
  4. Save `<name>.ply` and `<name>_scene.ply` next to this script, then open
     an Open3D viewer with a green AABB.

Usage::

    /home/yunshuang/anaconda3/envs/depth_lerobot/bin/python \\
        binomap/boxes/capture_box.py --name cyan_box
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "pointcloud_generation"))
from generate_pointcloud import create_pointcloud_from_arrays  # noqa: E402

EXTRINSICS_JSON = REPO_ROOT / "caliberation" / "camera_extrinsics.json"
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent

SAM2_CHECKPOINT = pathlib.Path(
    "/home/yunshuang/depth_lerobot/sim_3d_bimanual-main/checkpoints/sam2/sam2_hiera_large.pt"
)
SAM2_CONFIG_NAME = "configs/sam2/sam2_hiera_l.yaml"
SAM2_DEVICE = "cuda"

CAPTURE_WIDTH, CAPTURE_HEIGHT, CAPTURE_FPS = 640, 480, 30
WARMUP_FRAMES = 30

WORKSPACE_MIN = np.array([-0.6, -0.6, -0.05])
WORKSPACE_MAX = np.array([ 0.6,  0.6,  0.40])

CLUSTER_EPS_M = 0.008  # tightened from 0.015 (May 2026): a 15 mm eps was
                       # bridging the box's masked points to nearby objects via
                       # depth speckle, producing a "two-box" merged cloud.
                       # 8 mm separates them reliably for boxes on a table.
CLUSTER_MIN_POINTS = 30


def capture_rgbd(serial: str) -> tuple[np.ndarray, np.ndarray]:
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, CAPTURE_WIDTH, CAPTURE_HEIGHT, rs.format.bgr8, CAPTURE_FPS)
    cfg.enable_stream(rs.stream.depth, CAPTURE_WIDTH, CAPTURE_HEIGHT, rs.format.z16, CAPTURE_FPS)
    profile = pipeline.start(cfg)
    align = rs.align(rs.stream.color)
    try:
        for _ in range(WARMUP_FRAMES):
            pipeline.wait_for_frames()
        frames = align.process(pipeline.wait_for_frames())
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError(f"[{serial}] missing color/depth on capture")
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        bgr = np.asanyarray(color.get_data())
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth_mm_f = np.asanyarray(depth.get_data()).astype(np.float32) * depth_scale * 1000.0
        depth_mm = np.clip(depth_mm_f, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        return rgb, depth_mm
    finally:
        pipeline.stop()


def load_extrinsics() -> dict[str, dict]:
    with open(EXTRINSICS_JSON) as f:
        return json.load(f)


def load_sam2_predictor():
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    if not SAM2_CHECKPOINT.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found at {SAM2_CHECKPOINT}")
    model = build_sam2(SAM2_CONFIG_NAME, str(SAM2_CHECKPOINT), device=SAM2_DEVICE)
    return SAM2ImagePredictor(model)


def capture_click(rgb: np.ndarray, serial: str, name: str) -> tuple[int, int]:
    clicked = {"xy": None}

    def _cb(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["xy"] = (x, y)

    winname = f"cam {serial}: click the {name.replace('_', ' ')} (ESC to abort)"
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(winname, _cb)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    while clicked["xy"] is None:
        cv2.imshow(winname, bgr)
        if (cv2.waitKey(20) & 0xFF) == 27:
            break
    cv2.destroyWindow(winname)
    if clicked["xy"] is None:
        raise RuntimeError(f"[{serial}] click cancelled (ESC)")
    return clicked["xy"]


def preview_and_confirm_mask(rgb: np.ndarray, mask: np.ndarray, serial: str) -> bool:
    """Show SAM2 mask overlay on the RGB frame. Return True to accept, False to retry."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    overlay = bgr.copy()
    overlay[mask] = (0, 255, 0)
    blended = cv2.addWeighted(bgr, 0.5, overlay, 0.5, 0)
    cv2.putText(blended, "y = accept   n / any key = re-click   ESC = abort",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    winname = f"cam {serial}: SAM2 mask preview"
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    while True:
        cv2.imshow(winname, blended)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            cv2.destroyWindow(winname)
            raise RuntimeError(f"[{serial}] preview cancelled (ESC)")
        if k == ord('y'):
            cv2.destroyWindow(winname)
            return True
        if k != 255:
            cv2.destroyWindow(winname)
            return False


def sam2_mask_from_point(predictor, rgb: np.ndarray, point: tuple[int, int]) -> np.ndarray:
    predictor.set_image(rgb)
    masks, scores, _ = predictor.predict(
        point_coords=np.array([list(point)], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int32),
        multimask_output=True,
    )
    if masks is None or len(masks) == 0:
        raise RuntimeError("SAM2 returned no masks for the clicked point")
    best = int(np.argmax(scores))
    mask = masks[best].astype(bool)
    h, w = rgb.shape[:2]
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    print(f"  mask: {int(mask.sum()):,} px ({100 * mask.mean():.1f}% of frame), score={scores[best]:.3f}")
    return mask


def masked_pointcloud(rgb, depth_mm, mask, intrinsics) -> o3d.geometry.PointCloud:
    depth_masked = depth_mm.copy()
    depth_masked[~mask] = 0
    return create_pointcloud_from_arrays(rgb, depth_masked, intrinsics)


def crop_to_workspace(pcd):
    aabb = o3d.geometry.AxisAlignedBoundingBox(WORKSPACE_MIN, WORKSPACE_MAX)
    return pcd.crop(aabb)


def keep_largest_cluster(pcd):
    if len(pcd.points) == 0:
        return pcd
    labels = np.array(pcd.cluster_dbscan(eps=CLUSTER_EPS_M,
                                          min_points=CLUSTER_MIN_POINTS,
                                          print_progress=False))
    if labels.max() < 0:
        print("DBSCAN: no clusters survived, keeping all masked points as-is")
        return pcd
    counts = np.bincount(labels[labels >= 0])
    best = int(np.argmax(counts))
    idx = np.where(labels == best)[0]
    print(f"largest cluster: label={best}, {len(idx):,} pts "
          f"(of {labels.max() + 1} clusters, sizes={counts.tolist()})")
    return pcd.select_by_index(idx)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True,
                    help="Label for this object, e.g. brown_box, black_box, cyan_box. "
                         "Used to name the output PLYs (<name>.ply + <name>_scene.ply).")
    ap.add_argument("--out-dir", type=pathlib.Path, default=OUTPUT_DIR,
                    help=f"Where to save the PLYs. Default: {OUTPUT_DIR}")
    ap.add_argument("--no-viewer", action="store_true",
                    help="Skip the final Open3D viewer; just write the PLYs and exit.")
    ap.add_argument("--preview-mask", action="store_true",
                    help="After each SAM2 click, show the mask overlay and wait for "
                         "y=accept / any other key=retry / ESC=abort. Prevents the "
                         "'two boxes in cloud' problem when SAM2 spills onto adjacent "
                         "objects with similar color/texture.")
    args = ap.parse_args()

    if not EXTRINSICS_JSON.exists():
        raise SystemExit(f"missing {EXTRINSICS_JSON} — run calibration first")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cams = load_extrinsics()
    print(f"cameras in extrinsics: {list(cams.keys())}")

    print("loading SAM2…")
    predictor = load_sam2_predictor()

    scene = o3d.geometry.PointCloud()
    box = o3d.geometry.PointCloud()
    for serial, entry in cams.items():
        print(f"\n[{serial}] capturing…")
        rgb, depth_mm = capture_rgbd(serial)

        full = create_pointcloud_from_arrays(rgb, depth_mm, entry["intrinsics"])
        full.transform(np.array(entry["transform_camera_to_world"]))
        scene += full

        # Click loop: re-click until preview accepted (or skip preview entirely).
        while True:
            print(f"[{serial}] click the {args.name.replace('_', ' ')}…")
            point = capture_click(rgb, serial, args.name)
            print(f"  click at pixel {point}")
            mask = sam2_mask_from_point(predictor, rgb, point)
            if not args.preview_mask:
                break
            if preview_and_confirm_mask(rgb, mask, serial):
                break
            print(f"  [{serial}] mask rejected — re-click")

        masked = masked_pointcloud(rgb, depth_mm, mask, entry["intrinsics"])
        masked.transform(np.array(entry["transform_camera_to_world"]))
        print(f"[{serial}] masked cloud: {len(masked.points):,} pts")
        box += masked

        time.sleep(0.3)

    scene_path = out_dir / f"{args.name}_scene.ply"
    o3d.io.write_point_cloud(str(scene_path), scene)
    print(f"\nsaved {scene_path}: {len(scene.points):,} pts (full merged scene)")

    print(f"merged masked cloud: {len(box.points):,} pts")
    box = crop_to_workspace(box)
    print(f"after workspace crop: {len(box.points):,} pts")
    box = keep_largest_cluster(box)

    box_path = out_dir / f"{args.name}.ply"
    o3d.io.write_point_cloud(str(box_path), box)
    print(f"saved {box_path}: {len(box.points):,} pts")

    if args.no_viewer:
        return

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    geometries = [box, axes]
    if len(box.points) > 0:
        bbox = box.get_axis_aligned_bounding_box()
        bbox.color = (0.1, 0.9, 0.1)
        geometries.append(bbox)
        ext = bbox.get_extent()
        ctr = bbox.get_center()
        print(f"\n{args.name} AABB extent (m): x={ext[0]:.3f} y={ext[1]:.3f} z={ext[2]:.3f}")
        print(f"{args.name} AABB center (m): [{ctr[0]:+.3f}, {ctr[1]:+.3f}, {ctr[2]:+.3f}]")

    o3d.visualization.draw_geometries(
        geometries, window_name=f"{args.name} (SAM2 + RGBD fusion)"
    )


if __name__ == "__main__":
    main()
