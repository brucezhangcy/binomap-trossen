"""
Companion verifier for calibrate_gripper_align.py.

Projects the hand-frame axes (Algorithm 1 v_x/v_y/v_z) and the proposed
gripper-pointing axis (R_hand @ R_align @ +x_local) into the source RGB
image for a handful of sampled frames. Use it to eyeball whether v_z really
points into the box (where the hand is pushing), so you know whether the
default R_align = R_y(-90°) is the correct choice.

The npz produced by extract_trajectory.py contains world-frame rotations
(R_world = T_cam2world[:3,:3] @ R_cam), so we map back to camera frame by
left-multiplying by R_cam2world.T.
"""
import argparse, json
from pathlib import Path

import cv2
import numpy as np


def euler_zyx_to_rotmat(rx_deg, ry_deg, rz_deg):
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def project_cam_to_pixel(p_cam, intr):
    """Pinhole projection. p_cam is (3,) meters. Returns (u, v) pixels."""
    if p_cam[2] <= 1e-3:
        return None
    u = intr["fx"] * p_cam[0] / p_cam[2] + intr["cx"]
    v = intr["fy"] * p_cam[1] / p_cam[2] + intr["cy"]
    return float(u), float(v)


def draw_axis(img, intr, T_w2c, p_world, R_world, length_m=0.05, lw=2,
              colors=((0, 0, 255), (0, 255, 0), (255, 0, 0))):
    """Project the three columns of R_world from p_world into the image."""
    R_w2c = T_w2c[:3, :3]
    t_w2c = T_w2c[:3, 3]
    p_cam = R_w2c @ p_world + t_w2c
    base_px = project_cam_to_pixel(p_cam, intr)
    if base_px is None:
        return
    for j in range(3):
        v_w = R_world[:, j]
        tip_cam = R_w2c @ (p_world + v_w * length_m) + t_w2c
        tip_px = project_cam_to_pixel(tip_cam, intr)
        if tip_px is None:
            continue
        cv2.arrowedLine(img,
                        (int(round(base_px[0])), int(round(base_px[1]))),
                        (int(round(tip_px[0])), int(round(tip_px[1]))),
                        colors[j], lw, tipLength=0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", required=True)
    ap.add_argument("--clip_dir", required=True,
                    help="Recording directory containing camera_extrinsics.json + per-serial rgb/")
    ap.add_argument("--cam_serial_L", default="333422304645")
    ap.add_argument("--cam_serial_R", default="338122302972")
    ap.add_argument("--align_euler_deg", default="0,-90,0")
    ap.add_argument("--frames", default="0,40,80,120,160,200",
                    help="Comma-separated grid indices to render")
    ap.add_argument("--arrow_len_m", type=float, default=0.10)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    rx, ry, rz = [float(x) for x in args.align_euler_deg.split(",")]
    R_align = euler_zyx_to_rotmat(rx, ry, rz)
    print(f"R_align (euler ZYX {rx},{ry},{rz}°):")
    print(R_align.round(3))

    d = np.load(args.in_npz)
    p_L, R_L, vL, ts = d["p_L"], d["R_L"], d["valid_L"], d["ts_ms"]
    p_R, R_R, vR = d["p_R"], d["R_R"], d["valid_R"]

    extr_path = Path(args.clip_dir) / "camera_extrinsics.json"
    extr = json.loads(extr_path.read_text())

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    sel_frames = [int(x) for x in args.frames.split(",")]

    for arm_tag, serial, p_arr, R_arr, valid in [
        ("L", args.cam_serial_L, p_L, R_L, vL),
        ("R", args.cam_serial_R, p_R, R_R, vR),
    ]:
        cam = extr[serial]
        intr = cam["intrinsics"]
        T_c2w = np.array(cam["transform_camera_to_world"])
        T_w2c = np.linalg.inv(T_c2w)

        # match grid frame → source RGB filename via nearest timestamp
        ts_path = Path(args.clip_dir) / serial / "timestamps.csv"
        ts_rows = ts_path.read_text().strip().splitlines()[1:]
        frame_idxs = [int(r.split(",")[0]) for r in ts_rows]
        frame_ts_ms = [float(r.split(",")[2]) for r in ts_rows]
        rgb_dir = Path(args.clip_dir) / serial / "rgb"

        for k in sel_frames:
            if k >= len(ts) or not valid[k]:
                print(f"  {arm_tag} frame {k}: invalid, skipping")
                continue
            t_target = ts[k]
            i_src = int(np.argmin(np.abs(np.array(frame_ts_ms) - t_target)))
            src_idx = frame_idxs[i_src]
            img_path = rgb_dir / f"{src_idx:06d}.png"
            if not img_path.exists():
                print(f"  {arm_tag} frame {k}: no rgb at {img_path}")
                continue
            img = cv2.imread(str(img_path))
            # Upscale 3x for readability
            img = cv2.resize(img, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_LINEAR)
            intr = {**intr, "fx": intr["fx"] * 3, "fy": intr["fy"] * 3,
                    "cx": intr["cx"] * 3, "cy": intr["cy"] * 3}

            R_w2c = T_w2c[:3, :3]; t_w2c = T_w2c[:3, 3]
            p_cam = R_w2c @ p_arr[k] + t_w2c
            base_px = project_cam_to_pixel(p_cam, intr)
            if base_px is None:
                continue
            base = (int(base_px[0]), int(base_px[1]))

            # Draw v_x (red), v_y (green) thin; v_z (blue) THICK and labeled
            R_hand = R_arr[k]
            axis_specs = [
                (R_hand[:, 0], (50, 50, 220), 2, 0.7, "vx"),
                (R_hand[:, 1], (50, 200, 50), 2, 0.7, "vy"),
                (R_hand[:, 2], (255, 80, 0),  5, 1.5, "vz=palm normal"),
            ]
            for v_w, col, lw, mult, lbl in axis_specs:
                tip_cam = R_w2c @ (p_arr[k] + v_w * args.arrow_len_m * mult) + t_w2c
                tip_px = project_cam_to_pixel(tip_cam, intr)
                if not tip_px:
                    continue
                tip = (int(tip_px[0]), int(tip_px[1]))
                cv2.arrowedLine(img, base, tip, col, lw, tipLength=0.20)
                cv2.putText(img, lbl, (tip[0] + 4, tip[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

            # Gripper-pointing (BLACK), should overlay vz at align=(0,-90,0)
            R_grip = R_hand @ R_align
            grip_w = R_grip[:, 0]
            tip_cam = R_w2c @ (p_arr[k] + grip_w * args.arrow_len_m * 1.5) + t_w2c
            tip_px = project_cam_to_pixel(tip_cam, intr)
            if tip_px:
                tip = (int(tip_px[0]), int(tip_px[1]))
                cv2.arrowedLine(img, base, tip, (0, 0, 0), 3, tipLength=0.18)
                cv2.putText(img, "gripper+x", (tip[0] + 4, tip[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            cv2.circle(img, base, 6, (255, 255, 255), -1)
            cv2.circle(img, base, 6, (0, 0, 0), 2)

            H = img.shape[0]
            cv2.putText(img, f"{arm_tag} cam  k={k}  src_idx={src_idx}  R_align ZYX=({rx:.0f},{ry:.0f},{rz:.0f})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, "BLUE vz = palm normal (paper says: where hand pushes box)",
                        (10, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 0), 2)
            cv2.putText(img, "BLACK gripper+x = where Trossen gripper would point",
                        (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            out_path = Path(args.out_dir) / f"verify_{arm_tag}_f{k:03d}.png"
            cv2.imwrite(str(out_path), img)
            print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
