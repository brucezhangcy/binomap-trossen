"""Sanity-check R cam: render a few frames with the MediaPipe detection overlaid
to confirm we're tracking the actual right hand (not a stray left-hand corner)."""
import cv2, numpy as np
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

CLIP = Path("/data/bruce/BiNoMaP/recordings_one_hand_fix_7s")
SERIAL = "338122302972"
FRAMES = [10, 50, 100, 150, 200, 230]
MODEL = "/data/bruce/BiNoMaP/hand_landmarker.task"

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
    bgr = cv2.imread(str(rgb_path))
    H, W = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = hl.detect(mp_img)
    n = len(res.hand_landmarks) if res.hand_landmarks else 0
    vis = bgr.copy()
    print(f"f{k}: {n} hand(s)")
    if n:
        for i, lms in enumerate(res.hand_landmarks):
            handed = res.handedness[i][0].category_name
            score = res.handedness[i][0].score
            color = (0, 255, 0) if handed == "Right" else (0, 100, 255)  # green=R, orange=L
            for lm in lms:
                cv2.circle(vis, (int(lm.x * W), int(lm.y * H)), 3, color, -1)
            uw, vw = int(lms[0].x * W), int(lms[0].y * H)
            cv2.circle(vis, (uw, vw), 8, (0, 0, 255), 2)
            cv2.putText(vis, f"{handed} {score:.2f}", (uw + 10, vw),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            print(f"  hand[{i}] handed={handed} score={score:.2f} wrist=({uw},{vw})")
    cv2.imwrite(str(OUT / f"verify_R_mp_f{k}.png"), vis)
print(f"\nsaved → {OUT}/verify_R_mp_f*.png")
