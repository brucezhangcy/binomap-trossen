"""Test MediaPipe Hands (Task API) on the 6 R-cam frames WiLoR/YOLO failed on."""
import json, numpy as np, cv2
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

CLIP = Path("/data/bruce/BiNoMaP/recordings_one_hand_fix_7s")
SERIAL = "338122302972"
FRAMES = [30, 60, 90, 120, 150, 180]
MODEL = "/data/bruce/BiNoMaP/hand_landmarker.task"

extr = json.loads((CLIP / "camera_extrinsics.json").read_text())[SERIAL]
fx, fy, cx, cy = (extr["intrinsics"]["fx"], extr["intrinsics"]["fy"],
                  extr["intrinsics"]["cx"], extr["intrinsics"]["cy"])
T_c2w = np.array(extr["transform_camera_to_world"])

opts = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL),
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    running_mode=vision.RunningMode.IMAGE,
)
hl = vision.HandLandmarker.create_from_options(opts)

OUT = Path("/data/bruce/BiNoMaP/_inspect")
OUT.mkdir(exist_ok=True)

for k in FRAMES:
    rgb_path = CLIP / SERIAL / "rgb" / f"{k:06d}.png"
    dep_path = CLIP / SERIAL / "depth" / f"{k:06d}.png"
    bgr = cv2.imread(str(rgb_path))
    H, W = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(dep_path), cv2.IMREAD_UNCHANGED)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = hl.detect(mp_img)
    n = len(res.hand_landmarks) if res.hand_landmarks else 0
    print(f"\n=== frame {k} ===  MediaPipe detected {n} hand(s)")
    vis = bgr.copy()
    if n:
        for i, lms in enumerate(res.hand_landmarks):
            handed = res.handedness[i][0].category_name
            score = res.handedness[i][0].score
            w_lm = lms[0]  # wrist
            uw, vw = int(round(w_lm.x * W)), int(round(w_lm.y * H))
            d_mm = int(depth[vw, uw]) if 0 <= uw < W and 0 <= vw < H else -1
            print(f"  hand[{i}] handed={handed} score={score:.2f}  wrist 2D=({uw},{vw}) depth={d_mm}mm")
            if d_mm > 0:
                z = d_mm / 1000.0
                p_cam = np.array([(uw - cx) * z / fx, (vw - cy) * z / fy, z])
                p_world = T_c2w[:3, :3] @ p_cam + T_c2w[:3, 3]
                print(f"    p_world={p_world.round(3)}")
            for lm in lms:
                cv2.circle(vis, (int(round(lm.x * W)), int(round(lm.y * H))), 3, (0, 255, 255), -1)
            cv2.circle(vis, (uw, vw), 8, (0, 0, 255), -1)
            cv2.putText(vis, f"{handed} {score:.2f}", (uw + 12, vw),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(str(OUT / f"dbg_R_mp_f{k}.png"), vis)
print("\nsaved → /data/bruce/BiNoMaP/_inspect/dbg_R_mp_f*.png")
