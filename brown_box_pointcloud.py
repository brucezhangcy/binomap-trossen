"""Capture both calibrated RealSense cameras, segment the brown box with SAM2,
and fuse the masked pixels into a world-frame point cloud.

Pipeline:
  1. Live-capture one aligned RGBD pair per camera serial in
     `caliberation/camera_extrinsics.json` (pyrealsense2 directly — no robot).
  2. For each camera:
       - show the RGB; user clicks the brown box once
       - SAM2 (hiera-large) predicts a per-pixel mask for that click
       - unproject only the masked pixels to a camera-frame point cloud
       - transform to world frame via the calibrated extrinsic
  3. Merge masked clouds, crop to a workspace AABB (paranoia against bleed-
     through into far background), and keep the largest DBSCAN cluster
     (drops the occasional disconnected blob if SAM grabs a stray region).
  4. Save `brown_box.ply` and `scene.ply` (unmasked, for sanity-checking the
     extrinsics), and open an Open3D viewer with a green AABB around the box.

Run with the depth_lerobot env where SAM2 + torch + pyrealsense2 + open3d
are installed::

    /home/yunshuang/anaconda3/envs/depth_lerobot/bin/python \\
        binomap/brown_box_pointcloud.py
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

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "pointcloud_generation"))
from generate_pointcloud import create_pointcloud_from_arrays  # noqa: E402

EXTRINSICS_JSON = REPO_ROOT / "caliberation" / "camera_extrinsics.json"
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs" / "brown_box"

# SAM2 checkpoint shipped with the original sim_3d_bimanual-main install on
# this machine. Stays valid as long as that tree is on disk.
SAM2_CHECKPOINT = pathlib.Path(
    "/home/yunshuang/depth_lerobot/sim_3d_bimanual-main/checkpoints/sam2/sam2_hiera_large.pt"
)
SAM2_CONFIG_NAME = "configs/sam2/sam2_hiera_l.yaml"
SAM2_DEVICE = "cuda"

CAPTURE_WIDTH, CAPTURE_HEIGHT, CAPTURE_FPS = 640, 480, 30
WARMUP_FRAMES = 30

# Workspace AABB in world frame (meters). Generous box around the tabletop —
# the SAM mask already does the heavy lifting; this just catches the edge
# case where SAM bleeds into the floor/wall behind the box.
WORKSPACE_MIN = np.array([-0.6, -0.6, -0.05])
WORKSPACE_MAX = np.array([ 0.6,  0.6,  0.40])

# Drop disconnected blobs (SAM occasionally pulls in a thin shadow strip).
CLUSTER_EPS_M = 0.015
CLUSTER_MIN_POINTS = 30


def capture_rgbd(serial: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (rgb HxWx3 uint8, depth_mm HxW uint16)."""
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


# ── SAM2 ─────────────────────────────────────────────────────────────────────

def load_sam2_predictor():
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    if not SAM2_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found at {SAM2_CHECKPOINT}. "
            f"Download sam2_hiera_large.pt from facebookresearch/sam2 or update SAM2_CHECKPOINT."
        )
    model = build_sam2(SAM2_CONFIG_NAME, str(SAM2_CHECKPOINT), device=SAM2_DEVICE)
    return SAM2ImagePredictor(model)


def capture_click(rgb: np.ndarray, serial: str) -> tuple[int, int]:
    """Show RGB in an OpenCV window; return the (x, y) of the user's first click."""
    clicked = {"xy": None}

    def _cb(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["xy"] = (x, y)

    winname = f"cam {serial}: click the brown box (ESC to abort)"
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


def sam2_mask_from_point(predictor, rgb: np.ndarray, point: tuple[int, int]) -> np.ndarray:
    """Run SAM2 with a single positive point. Returns the best-scored bool mask (H, W)."""
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


# ── point cloud construction ─────────────────────────────────────────────────

def masked_pointcloud(rgb: np.ndarray, depth_mm: np.ndarray, mask: np.ndarray,
                      intrinsics: dict) -> o3d.geometry.PointCloud:
    """Build a per-camera point cloud restricted to mask==True.

    We do this by zeroing depth outside the mask, then reusing
    create_pointcloud_from_arrays (which drops zero/out-of-range depth).
    Keeps the unprojection math in one place.
    """
    depth_masked = depth_mm.copy()
    depth_masked[~mask] = 0
    return create_pointcloud_from_arrays(rgb, depth_masked, intrinsics)


def crop_to_workspace(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    aabb = o3d.geometry.AxisAlignedBoundingBox(WORKSPACE_MIN, WORKSPACE_MAX)
    return pcd.crop(aabb)


def keep_largest_cluster(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    if len(pcd.points) == 0:
        return pcd
    labels = np.array(pcd.cluster_dbscan(eps=CLUSTER_EPS_M, min_points=CLUSTER_MIN_POINTS,
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


def parse_auto_clicks(values: list[str] | None) -> dict[str, tuple[int, int]]:
    """Parse --auto-click SERIAL,X,Y entries into {serial: (x, y)}."""
    out: dict[str, tuple[int, int]] = {}
    if not values:
        return out
    for v in values:
        parts = v.split(",")
        if len(parts) != 3:
            raise ValueError(f"--auto-click must be SERIAL,X,Y; got {v!r}")
        serial, x, y = parts[0].strip(), int(parts[1]), int(parts[2])
        out[serial] = (x, y)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--auto-click", action="append", default=None,
                    help="Skip the interactive cv2 click for a camera. Format: "
                         "SERIAL,X,Y (repeatable). Cameras not listed still use "
                         "the interactive click. Example: "
                         "--auto-click 333422304645,359,263 "
                         "--auto-click 338122302972,361,226")
    ap.add_argument("--no-viewer", action="store_true",
                    help="Skip the Open3D draw_geometries call at the end. "
                         "Required for headless / background runs.")
    args = ap.parse_args()
    auto_clicks = parse_auto_clicks(args.auto_click)

    if not EXTRINSICS_JSON.exists():
        raise SystemExit(f"missing {EXTRINSICS_JSON} — run calibration first")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cams = load_extrinsics()
    print(f"cameras in extrinsics: {list(cams.keys())}")
    if auto_clicks:
        print(f"auto-click overrides: {auto_clicks}")

    print("loading SAM2…")
    predictor = load_sam2_predictor()

    scene = o3d.geometry.PointCloud()
    box = o3d.geometry.PointCloud()
    for serial, entry in cams.items():
        print(f"\n[{serial}] capturing…")
        rgb, depth_mm = capture_rgbd(serial)

        # Full-frame cloud for sanity-checking extrinsics.
        full = create_pointcloud_from_arrays(rgb, depth_mm, entry["intrinsics"])
        full.transform(np.array(entry["transform_camera_to_world"]))
        scene += full

        if serial in auto_clicks:
            point = auto_clicks[serial]
            print(f"[{serial}] auto-click at pixel {point} (from --auto-click)")
        else:
            print(f"[{serial}] click the brown box…")
            point = capture_click(rgb, serial)
            print(f"  click at pixel {point}")
        mask = sam2_mask_from_point(predictor, rgb, point)

        masked = masked_pointcloud(rgb, depth_mm, mask, entry["intrinsics"])
        masked.transform(np.array(entry["transform_camera_to_world"]))
        print(f"[{serial}] masked cloud: {len(masked.points):,} pts")
        box += masked

        time.sleep(0.3)  # USB settle before next device

    scene_path = OUTPUT_DIR / "scene.ply"
    o3d.io.write_point_cloud(str(scene_path), scene)
    print(f"\nsaved {scene_path}: {len(scene.points):,} pts (full merged scene)")

    print(f"merged masked cloud: {len(box.points):,} pts")
    box = crop_to_workspace(box)
    print(f"after workspace crop: {len(box.points):,} pts")
    box = keep_largest_cluster(box)

    box_path = OUTPUT_DIR / "brown_box.ply"
    o3d.io.write_point_cloud(str(box_path), box)
    print(f"saved {box_path}: {len(box.points):,} pts")

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    geometries = [box, axes]
    if len(box.points) > 0:
        bbox = box.get_axis_aligned_bounding_box()
        bbox.color = (0.1, 0.9, 0.1)
        geometries.append(bbox)
        ext = bbox.get_extent()
        ctr = bbox.get_center()
        print(f"\nbrown box AABB extent (m): x={ext[0]:.3f} y={ext[1]:.3f} z={ext[2]:.3f}")
        print(f"brown box AABB center (m): [{ctr[0]:+.3f}, {ctr[1]:+.3f}, {ctr[2]:+.3f}]")

    if args.no_viewer:
        print("(--no-viewer set, skipping Open3D draw_geometries)")
    else:
        o3d.visualization.draw_geometries(geometries, window_name="Brown box (SAM2 + RGBD fusion)")


if __name__ == "__main__":
    main()
