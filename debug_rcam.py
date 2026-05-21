"""Debug R cam frames: detect hand, dump wrist 2D pixel, depth, deprojected 3D point."""
import sys, json, numpy as np, cv2, torch
from pathlib import Path

CLIP = Path("/data/bruce/BiNoMaP/recordings_one_hand_fix_7s")
SERIAL = "338122302972"
FRAMES_TO_CHECK = [30, 60, 90, 120, 150, 180]

# Load extrinsics
extr = json.loads((CLIP / "camera_extrinsics.json").read_text())[SERIAL]
fx, fy, cx, cy = (extr["intrinsics"]["fx"], extr["intrinsics"]["fy"],
                  extr["intrinsics"]["cx"], extr["intrinsics"]["cy"])
T_c2w = np.array(extr["transform_camera_to_world"])

from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
pipe = WiLorHandPose3dEstimationPipeline(device=torch.device("cuda:0"), dtype=torch.float32)

IDX_WRIST = 0
IDX_THUMB = 4
IDX_INDEX = 8

for k in FRAMES_TO_CHECK:
    rgb_path = CLIP / SERIAL / "rgb" / f"{k:06d}.png"
    dep_path = CLIP / SERIAL / "depth" / f"{k:06d}.png"
    bgr = cv2.imread(str(rgb_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(dep_path), cv2.IMREAD_UNCHANGED)  # uint16 mm
    outs = pipe.predict(rgb)
    print(f"\n=== frame {k} ===  detections: {len(outs)}")
    for i, det in enumerate(outs):
        kp2d = det["wilor_preds"]["pred_keypoints_2d"][0]
        uw, vw = int(round(kp2d[IDX_WRIST, 0])), int(round(kp2d[IDX_WRIST, 1]))
        ut, vt = int(round(kp2d[IDX_THUMB, 0])), int(round(kp2d[IDX_THUMB, 1]))
        ui, vi = int(round(kp2d[IDX_INDEX, 0])), int(round(kp2d[IDX_INDEX, 1]))
        side_label = "L" if det["is_right"] == 0 else "R"
        d_wrist_mm = depth[vw, uw] if 0 <= uw < depth.shape[1] and 0 <= vw < depth.shape[0] else -1
        d_thumb_mm = depth[vt, ut] if 0 <= ut < depth.shape[1] and 0 <= vt < depth.shape[0] else -1
        d_index_mm = depth[vi, ui] if 0 <= ui < depth.shape[1] and 0 <= vi < depth.shape[0] else -1
        print(f"  det[{i}] YOLO_label={side_label}  bbox={[int(b) for b in det['hand_bbox']]}")
        print(f"    wrist 2D=({uw},{vw})   depth={d_wrist_mm}mm")
        print(f"    thumb 2D=({ut},{vt})   depth={d_thumb_mm}mm")
        print(f"    index 2D=({ui},{vi})   depth={d_index_mm}mm")
        # Deproject wrist
        if d_wrist_mm > 0:
            z = d_wrist_mm / 1000.0
            x = (uw - cx) * z / fx
            y = (vw - cy) * z / fy
            p_cam = np.array([x, y, z])
            p_world = T_c2w[:3, :3] @ p_cam + T_c2w[:3, 3]
            print(f"    wrist p_cam={p_cam.round(3)}  p_world={p_world.round(3)}")

        # Render overlay: RGB with wrist (red), thumb (green), index (blue) circles + bbox
        vis = bgr.copy()
        x1, y1, x2, y2 = [int(b) for b in det["hand_bbox"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.circle(vis, (uw, vw), 8, (0, 0, 255), -1)   # wrist red
        cv2.circle(vis, (ut, vt), 5, (0, 255, 0), -1)   # thumb green
        cv2.circle(vis, (ui, vi), 5, (255, 0, 0), -1)   # index blue
        cv2.putText(vis, f"f{k} det{i} {side_label} wrist_d={d_wrist_mm}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        Path("/data/bruce/BiNoMaP/_inspect").mkdir(exist_ok=True)
        cv2.imwrite(f"/data/bruce/BiNoMaP/_inspect/dbg_R_f{k}_det{i}.png", vis)

        # Depth-image overlay too (false-colored)
        dep_vis = cv2.applyColorMap((depth.astype(np.float32) / 10.0).clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.circle(dep_vis, (uw, vw), 8, (255, 255, 255), -1)
        cv2.imwrite(f"/data/bruce/BiNoMaP/_inspect/dbg_R_f{k}_det{i}_depth.png", dep_vis)
print("\nsaved overlays to /data/bruce/BiNoMaP/_inspect/dbg_R_f*.png")
