"""
Drop-in replacement for extract_trajectory.py that uses MediaPipe Hands instead
of WiLoR + YOLO. Same CLI, same output schema.

Why: on clips where the hand pose is unusual (bracing a box from underneath,
knuckles-down, partial occlusion behind an object), WiLoR-mini's bundled YOLO
detector fails — sometimes misses the hand entirely, sometimes false-positives
on nearby high-contrast objects (white shipping labels, etc.). MediaPipe
Hands' detector has a different training distribution and reliably catches
these cases.

Detector: Google's MediaPipe Hand Landmarker (task API), loaded from
hand_landmarker.task in this directory. Provides:
  - 21 image-space landmarks (normalized 0-1)
  - 21 "world" 3D landmarks (wrist-relative meters, approximate)
  - handedness label + confidence

We use the wrist 2D landmark + the scene depth map for metric 3D position
(identical to the WiLoR variant — the depth deprojection step is the only
source of metric position in either pipeline). For SO(3) orientation we feed
the 3D world landmarks through Algorithm 1's cross-products (orientation is
relative, so any consistent landmark frame works).
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Reuse all the existing pipeline pieces (helpers + filter + smoothing).
# extract_trajectory.py imports torch at module level (for WiLoR), which is
# fine in the wilor env that already has torch.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_trajectory import (
    algorithm1_so3,
    load_clip,
    deproject,
    in_workspace,
    filter_outliers_by_neighbors,
    fill_gaps,
    resample_to_grid,
    align_bimanual,
    filter_bundle_outliers,
    save_outputs,
    plot_trajectory,
    IDX_WRIST,
    IDX_THUMB_TIP,
    IDX_INDEX_TIP,
    IDX_RING_TIP,
    SERIAL_L,
    SERIAL_R,
)


HAND_LANDMARKER_MODEL = str(Path(__file__).resolve().parent / "hand_landmarker.task")


class MediaPipeRunner:
    """Wraps MediaPipe Hand Landmarker behind the same interface as WilorRunner.

    Returns list of detection dicts shaped like WiLoR-mini's output so we can
    reuse pick_detection() unchanged."""

    def __init__(self, min_det_conf=0.3, min_pres_conf=0.3, num_hands=2):
        opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL),
            num_hands=num_hands,
            min_hand_detection_confidence=min_det_conf,
            min_hand_presence_confidence=min_pres_conf,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(opts)

    def __call__(self, rgb_img):
        H, W = rgb_img.shape[:2]
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        res = self.landmarker.detect(mp_img)
        if not res.hand_landmarks:
            return []
        out = []
        for i, lms_2d in enumerate(res.hand_landmarks):
            lms_3d = res.hand_world_landmarks[i]
            handed = res.handedness[i][0].category_name  # "Left" or "Right"
            is_right = 1.0 if handed == "Right" else 0.0
            kp2d = np.array([[lm.x * W, lm.y * H] for lm in lms_2d], dtype=np.float32)
            kp3d = np.array([[lm.x, lm.y, lm.z] for lm in lms_3d], dtype=np.float32)
            x1, y1 = kp2d.min(0); x2, y2 = kp2d.max(0)
            out.append({
                "hand_bbox": [float(x1), float(y1), float(x2), float(y2)],
                "is_right": is_right,
                "wilor_preds": {
                    "pred_keypoints_2d": kp2d[None, :, :],
                    "pred_keypoints_3d": kp3d[None, :, :],
                },
            })
        return out


def pick_detection(detections, expected_side):
    if not detections:
        return None, True
    expected_is_right = 1.0 if expected_side == "R" else 0.0
    def area(d):
        b = d["hand_bbox"]
        return (b[2] - b[0]) * (b[3] - b[1])
    matching = [d for d in detections if float(d["is_right"]) == expected_is_right]
    if matching:
        return max(matching, key=area), False
    return max(detections, key=area), True


def process_camera(cam, expected_side, runner, position_source="wrist",
                   depth_min_mm=100.0, depth_max_mm=1500.0,
                   workspace_bounds=(-0.7, 0.7, -0.7, 0.7, -0.05, 0.7),
                   log_every=100):
    intr = cam["intrinsics"]
    T_c2w = cam["T_cam2world"]
    R_c2w = T_c2w[:3, :3]
    t_c2w = T_c2w[:3, 3]
    N = len(cam["frames"])
    ts_ms = np.zeros(N)
    p_world = np.zeros((N, 3))
    R_world = np.zeros((N, 3, 3))
    valid = np.zeros(N, dtype=bool)
    side_det = -1 * np.ones(N, dtype=np.int8)
    fallback_used = np.zeros(N, dtype=bool)
    depth_mm_arr = np.zeros(N)
    n_no_det = n_no_depth = n_out_ws = 0
    t0 = time.time()
    for k, (frame_idx, _ts_unix, t_dev_ms) in enumerate(cam["frames"]):
        ts_ms[k] = t_dev_ms
        bgr = cv2.imread(str(cam["rgb_dir"] / f"{frame_idx:06d}.png"))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(cam["depth_dir"] / f"{frame_idx:06d}.png"), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.dtype != np.uint16:
            continue
        dets = runner(rgb)
        chosen, fb = pick_detection(dets, expected_side)
        if chosen is None:
            n_no_det += 1
            continue
        wp = chosen["wilor_preds"]
        kp2d = wp["pred_keypoints_2d"][0]
        kp3d = wp["pred_keypoints_3d"][0]
        side_int = int(round(float(chosen["is_right"])))
        side_det[k] = side_int
        fallback_used[k] = fb
        if position_source == "wrist":
            u_c, v_c = kp2d[IDX_WRIST, 0], kp2d[IDX_WRIST, 1]
        else:
            u_c = 0.5 * (kp2d[IDX_THUMB_TIP, 0] + kp2d[IDX_INDEX_TIP, 0])
            v_c = 0.5 * (kp2d[IDX_THUMB_TIP, 1] + kp2d[IDX_INDEX_TIP, 1])
        p_cam, ok_depth, z_mm = deproject(u_c, v_c, depth, intr,
                                          depth_min_mm=depth_min_mm,
                                          depth_max_mm=depth_max_mm)
        depth_mm_arr[k] = z_mm
        if not ok_depth:
            n_no_depth += 1
            continue
        p_w = R_c2w @ p_cam + t_c2w
        if not in_workspace(p_w, workspace_bounds):
            n_out_ws += 1
            continue
        side_for_alg = "R" if side_int == 1 else "L"
        R_cam = algorithm1_so3(kp3d, side_for_alg)
        p_world[k] = p_w
        R_world[k] = R_c2w @ R_cam
        valid[k] = True
        if k % log_every == 0:
            print(f"  [{k}/{N}] ok side={side_for_alg} fb={fb} z={z_mm/1000:.3f}m p_w={p_world[k].round(3)}", flush=True)
    dt = time.time() - t0
    print(f"  cam done in {dt:.1f}s ({N/dt:.1f} fps). valid={valid.sum()}/{N}  "
          f"no_det={n_no_det}  no_depth={n_no_depth}  out_workspace={n_out_ws}", flush=True)
    return {
        "ts_ms": ts_ms, "p_world": p_world, "R_world": R_world,
        "valid": valid, "side_detected": side_det,
        "fallback_used": fallback_used, "depth_mm": depth_mm_arr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_frames", type=int, default=-1)
    ap.add_argument("--max_gap_frames", type=int, default=10)
    ap.add_argument("--position_source", choices=["wrist", "midpoint"], default="wrist")
    ap.add_argument("--depth_min_mm", type=float, default=100.0)
    ap.add_argument("--depth_max_mm", type=float, default=1500.0)
    ap.add_argument("--workspace_box", default="-0.7,0.7,-0.7,0.7,-0.05,0.7")
    ap.add_argument("--outlier_window", type=int, default=5)
    ap.add_argument("--bundle_k_neighbors", type=int, default=10)
    ap.add_argument("--outlier_max_dev_mm", type=float, default=100.0)
    ap.add_argument("--min_det_conf", type=float, default=0.3)
    args = ap.parse_args()
    ws = tuple(float(x) for x in args.workspace_box.split(","))

    print("=== BiNoMaP Stage 1 trajectory extraction (MediaPipe) ===")
    print(f"clip: {args.clip}\nout_dir: {args.out_dir}\nposition_source: {args.position_source}\n"
          f"min_det_conf: {args.min_det_conf}")

    cams = load_clip(args.clip)
    if args.max_frames > 0:
        for s in cams:
            cams[s]["frames"] = cams[s]["frames"][: args.max_frames]

    runner = MediaPipeRunner(min_det_conf=args.min_det_conf, min_pres_conf=args.min_det_conf)

    print(f"\n--- processing left-arm cam {SERIAL_L} (expected side=L) ---")
    traj_L = process_camera(cams[SERIAL_L], "L", runner, args.position_source,
                            args.depth_min_mm, args.depth_max_mm, ws)
    print(f"\n--- processing right-arm cam {SERIAL_R} (expected side=R) ---")
    traj_R = process_camera(cams[SERIAL_R], "R", runner, args.position_source,
                            args.depth_min_mm, args.depth_max_mm, ws)

    print(f"\n--- per-camera outlier filter ---")
    n_pre_L = int(traj_L["valid"].sum()); n_pre_R = int(traj_R["valid"].sum())
    traj_L = filter_outliers_by_neighbors(traj_L, args.outlier_window, args.outlier_max_dev_mm)
    traj_R = filter_outliers_by_neighbors(traj_R, args.outlier_window, args.outlier_max_dev_mm)
    print(f"  L: rejected {traj_L['n_rejected']}/{n_pre_L}   R: rejected {traj_R['n_rejected']}/{n_pre_R}")

    print(f"\n--- per-camera fill_gaps (max_gap={args.max_gap_frames}) ---")
    traj_L = fill_gaps(traj_L, max_gap_frames=args.max_gap_frames)
    traj_R = fill_gaps(traj_R, max_gap_frames=args.max_gap_frames)
    print(f"  L after fill: {int(traj_L['valid'].sum())}/{len(traj_L['valid'])} (filled {traj_L['n_filled']})")
    print(f"  R after fill: {int(traj_R['valid'].sum())}/{len(traj_R['valid'])} (filled {traj_R['n_filled']})")

    print("\n--- aligning bimanual trajectory ---")
    bundle = align_bimanual(traj_L, traj_R, fps=30.0)

    print(f"\n--- bundle outlier filter ---")
    bundle, n_rej = filter_bundle_outliers(bundle, args.bundle_k_neighbors, args.outlier_max_dev_mm)
    print(f"  L grid rej={n_rej['L']}  R grid rej={n_rej['R']}")

    for k in range(min(10, len(bundle["ts_ms"]))):
        if bundle["valid_L"][k]:
            R = bundle["R_L"][k]
            assert abs(np.linalg.det(R) - 1) < 1e-3
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-3)
    print("SO(3) sanity OK on first 10 valid samples")

    save_outputs(args.out_dir, bundle)
    plot_trajectory(bundle, Path(args.out_dir) / "trajectory.png",
                    title=f"BiNoMaP P_coarse (MediaPipe) — {Path(args.clip).name}")

    print(f"\n=== summary ===")
    print(f"  L cam valid (after fill): {int(traj_L['valid'].sum())}/{len(traj_L['valid'])}")
    print(f"  R cam valid (after fill): {int(traj_R['valid'].sum())}/{len(traj_R['valid'])}")
    print(f"  bimanual grid: {len(bundle['ts_ms'])}   valid_L={int(bundle['valid_L'].sum())}   valid_R={int(bundle['valid_R'].sum())}")
    print(f"  L cam side_detected: L={(traj_L['side_detected']==0).sum()}  R={(traj_L['side_detected']==1).sum()}  miss={(traj_L['side_detected']==-1).sum()}  fallback={int(traj_L['fallback_used'].sum())}")
    print(f"  R cam side_detected: L={(traj_R['side_detected']==0).sum()}  R={(traj_R['side_detected']==1).sum()}  miss={(traj_R['side_detected']==-1).sum()}  fallback={int(traj_R['fallback_used'].sum())}")


if __name__ == "__main__":
    main()
