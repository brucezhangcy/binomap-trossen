"""
BiNoMaP Stage 1 — Bimanual wrist-trajectory extraction for ALOHA Trossen replay.

Pipeline per clip directory (e.g. /data/bruce/BiNoMaP/recordings):
  1. Per camera (left-arm cam 333422304645, right-arm cam 338122302972):
       - Run WiLoR (via wilor-mini) on each RGB frame
       - Pick the detection matching the expected hand side (fallback: best confidence)
       - Position: project WRIST 2D pixel (MANO joint 0) → deproject with depth +
         intrinsics to get p_cam in meters
         (NOTE: BiNoMaP §3.2 uses midpoint(thumb_tip, index_tip) to align with the
          parallel-jaw gripper contact point. Mentor explicitly asked for wrist,
          so we use joint 0 instead.)
       - Compute SO(3) orientation R_cam via BiNoMaP Algorithm 1
         (cross products on wrist/index_tip/ring_tip; sign-corrected for left hand)
         R_cam is anchored at the wrist origin, so it naturally pairs with the
         wrist position.
       - Lift (p_cam, R_cam) into world frame via transform_camera_to_world
  2. Time-align the two single-arm streams on a common 30 Hz grid in their
     ts_device_ms overlap window (linear interp on position, SLERP on rotation)
  3. Save outputs: trajectory.npz, trajectory.csv, trajectory.png

Output bundle (npz):
  ts_ms              (T,)      reference grid timestamps (device_ms, monotonic)
  p_L                (T, 3)    left-arm contact point in world frame [m]
  R_L                (T, 3, 3) left-arm rotation in world frame (SO(3))
  q_L                (T, 4)    left-arm rotation as quaternion (x,y,z,w)
  valid_L            (T,)      bool, True if a usable sample exists at this grid pt
  interp_L           (T,)      bool, True if this sample required gap-fill
                                (i.e., the underlying per-camera frame was missing
                                 and filled via linear/SLERP interpolation)
  side_L_detected    (T,)      0=left-hand, 1=right-hand, -1=missing
  p_R, R_R, q_R, valid_R, interp_R, side_R_detected — symmetric for right arm
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
    WiLorHandPose3dEstimationPipeline,
)

# ---------------- MANO/OpenPose joint indices (verified in wilor/models/mano_wrapper.py) ----------------
IDX_WRIST = 0
IDX_THUMB_TIP = 4
IDX_INDEX_TIP = 8
IDX_RING_TIP = 16

# Camera serial → arm assignment (from camera_extrinsics.json)
SERIAL_L = "333422304645"  # left arm
SERIAL_R = "338122302972"  # right arm


# ---------------- utility math ----------------
def normalize(v, eps=1e-8):
    n = np.linalg.norm(v)
    return v / (n + eps)


def algorithm1_so3(kp3d, side):
    """
    BiNoMaP Algorithm 1: build a 3x3 rotation matrix from MANO keypoints.
      v_z = (l_iw × l_rw) / |...|     palm normal (gripper approach)
      v_y = (l_iw - 0.5*(l_iw + l_rw)) / |...|   gripper open/close
      v_x = v_y × v_z                  right-handed frame

    For left-hand detections, WiLoR-mini already X-flipped the keypoints so the
    geometry has left-hand chirality; the resulting cross-product palm normal
    then points opposite to the right-hand case. We negate v_z (and re-derive
    v_x) so that v_z consistently points "out of the palm" for both hands.

    Args:
      kp3d:  (21, 3) MANO joint positions in camera-aligned local frame [m]
      side:  'L' or 'R'
    Returns:
      R: (3, 3) rotation matrix in SO(3), columns = [v_x, v_y, v_z]
    """
    p_wri = kp3d[IDX_WRIST]
    p_ind = kp3d[IDX_INDEX_TIP]
    p_ring = kp3d[IDX_RING_TIP]
    l_iw = p_ind - p_wri
    l_rw = p_ring - p_wri

    v_z = np.cross(l_iw, l_rw)
    if side == "L":
        v_z = -v_z  # sign-correct for left-hand chirality (see docstring)
    v_z = normalize(v_z)

    v_y = l_iw - 0.5 * (l_iw + l_rw)  # equals 0.5*(l_iw - l_rw)
    v_y = normalize(v_y)

    v_x = np.cross(v_y, v_z)  # v_x ⟂ v_y, v_z; already unit length
    R = np.stack([v_x, v_y, v_z], axis=1)  # columns are x,y,z basis vectors
    return R


def rotmat_to_quat(R):
    """Rotation matrix → quaternion (x, y, z, w). Robust to all four cases."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = 2.0 * np.sqrt(1.0 + tr)
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return np.array([x, y, z, w])


def slerp_quat(q0, q1, t):
    """SLERP between two unit quaternions (x,y,z,w)."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    theta_0 = np.arccos(np.clip(d, -1.0, 1.0))
    theta = theta_0 * t
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - d * np.sin(theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return s0 * q0 + s1 * q1


def quat_to_rotmat(q):
    """Quaternion (x,y,z,w) → 3x3 rotation matrix."""
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n if n > 0 else 0.0
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array([
        [1 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1 - (xx + yy)],
    ])


# ---------------- data loading ----------------
def load_clip(clip_dir):
    """Parse camera_extrinsics.json and assemble per-camera metadata."""
    clip_dir = Path(clip_dir)
    extr = json.loads((clip_dir / "camera_extrinsics.json").read_text())
    cams = {}
    for serial, info in extr.items():
        cam_root = clip_dir / serial
        ts_csv = cam_root / "timestamps.csv"
        rows = []
        with open(ts_csv) as f:
            header = f.readline().strip().split(",")
            for line in f:
                parts = line.strip().split(",")
                rows.append((int(parts[0]), float(parts[1]), float(parts[2])))
        K = info["intrinsics"]
        cams[serial] = {
            "arm": info["arm"],  # 'left' or 'right'
            "intrinsics": np.array([K["fx"], K["fy"], K["cx"], K["cy"]]),
            "img_w": K["width"],
            "img_h": K["height"],
            "T_cam2world": np.array(info["transform_camera_to_world"]),
            "frames": rows,  # list of (frame_idx, ts_unix, ts_device_ms)
            "rgb_dir": cam_root / "rgb",
            "depth_dir": cam_root / "depth",
        }
    return cams


def deproject(u, v, depth_img_mm, intr, depth_window=3, depth_min_mm=100.0, depth_max_mm=1500.0):
    """
    Project pixel (u, v) into camera-frame 3D point using depth image (uint16 mm).
    Falls back to a small-window median if center pixel is zero (depth dropout).
    Rejects depths outside [depth_min_mm, depth_max_mm] (workspace sanity check):
    these typically indicate the wrist pixel landed on background (wall behind table)
    or on the demonstrator's body — both produce wildly wrong 3D points.
    Returns (p_cam, valid_bool, used_depth_mm).
    """
    fx, fy, cx, cy = intr
    H, W = depth_img_mm.shape
    u_i, v_i = int(round(u)), int(round(v))
    if u_i < 0 or u_i >= W or v_i < 0 or v_i >= H:
        return np.zeros(3), False, 0.0
    z_mm = float(depth_img_mm[v_i, u_i])
    if z_mm <= 0:
        r = depth_window
        u0, u1 = max(0, u_i - r), min(W, u_i + r + 1)
        v0, v1 = max(0, v_i - r), min(H, v_i + r + 1)
        patch = depth_img_mm[v0:v1, u0:u1].astype(np.float32)
        good = patch[patch > 0]
        if good.size == 0:
            return np.zeros(3), False, 0.0
        z_mm = float(np.median(good))
    # depth sanity: reject background hits and impossibly-close readings
    if z_mm < depth_min_mm or z_mm > depth_max_mm:
        return np.zeros(3), False, z_mm
    z = z_mm / 1000.0  # mm → m
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z]), True, z_mm


def in_workspace(p_world, bounds):
    """bounds = (xmin, xmax, ymin, ymax, zmin, zmax). Returns True if p_world inside."""
    return (bounds[0] <= p_world[0] <= bounds[1] and
            bounds[2] <= p_world[1] <= bounds[3] and
            bounds[4] <= p_world[2] <= bounds[5])


# ---------------- WiLoR wrapper ----------------
class WilorRunner:
    def __init__(self, device="cuda:0"):
        self.device = torch.device(device)
        self.pipe = WiLorHandPose3dEstimationPipeline(device=self.device, dtype=torch.float32)

    def __call__(self, rgb_img):
        """rgb_img: HxWx3 uint8 RGB. Returns list of det dicts with 'is_right', 'hand_bbox', 'wilor_preds'."""
        return self.pipe.predict(rgb_img)


def pick_detection(detections, expected_side):
    """
    Pick the detection most consistent with the expected hand side.
    expected_side ∈ {'L', 'R'} → expected is_right in {0, 1}.
    Strategy:
      1. Filter detections matching expected side; if any, return the largest bbox.
      2. Else return the largest bbox among all detections (mark as fallback).
    Returns (chosen_det, fallback_used: bool) or (None, True) if no detections.
    """
    if len(detections) == 0:
        return None, True
    expected_is_right = 1.0 if expected_side == "R" else 0.0

    def bbox_area(det):
        b = det["hand_bbox"]
        return (b[2] - b[0]) * (b[3] - b[1])

    matching = [d for d in detections if float(d["is_right"]) == expected_is_right]
    if matching:
        return max(matching, key=bbox_area), False
    return max(detections, key=bbox_area), True


# ---------------- per-camera processing ----------------
def process_camera(cam, expected_side, runner, position_source="wrist",
                   depth_min_mm=100.0, depth_max_mm=1500.0,
                   workspace_bounds=(-0.7, 0.7, -0.7, 0.7, -0.05, 0.7),
                   log_every=50):
    """
    Run WiLoR on every frame of one camera, return per-frame trajectory in world frame.

    Returns dict with keys (all numpy arrays of length N = len(cam['frames'])):
      ts_ms (N,)             timestamps in device_ms
      p_world (N, 3)         contact point in world frame [m]
      R_world (N, 3, 3)      rotation in world frame
      valid (N,)             bool
      side_detected (N,)     -1 missing, 0 = left-hand, 1 = right-hand
      fallback_used (N,)     bool — chose a detection whose side didn't match expected
      depth_mm (N,)          depth (mm) used for deprojection (for diagnostics)
    """
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
    depth_mm = np.zeros(N)

    t0 = time.time()
    n_no_det = 0
    n_no_depth = 0
    n_out_workspace = 0
    for k, (frame_idx, _ts_unix, t_dev_ms) in enumerate(cam["frames"]):
        ts_ms[k] = t_dev_ms
        rgb_path = cam["rgb_dir"] / f"{frame_idx:06d}.png"
        depth_path = cam["depth_dir"] / f"{frame_idx:06d}.png"
        bgr = cv2.imread(str(rgb_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.dtype != np.uint16:
            continue

        dets = runner(rgb)
        chosen, fb = pick_detection(dets, expected_side)
        if chosen is None:
            n_no_det += 1
            if k % log_every == 0:
                print(f"  [{k}/{N}] no det", flush=True)
            continue
        wp = chosen["wilor_preds"]
        kp2d = wp["pred_keypoints_2d"][0]  # (21, 2)
        kp3d = wp["pred_keypoints_3d"][0]  # (21, 3)
        side_int = int(round(float(chosen["is_right"])))  # 0=left, 1=right
        side_det[k] = side_int
        fallback_used[k] = fb

        # Position pixel: 'wrist' (MANO joint 0) or 'midpoint' (BiNoMaP §3.2,
        # midpoint of thumb_tip and index_tip). Mentor asked to try both.
        if position_source == "wrist":
            u_c = kp2d[IDX_WRIST, 0]
            v_c = kp2d[IDX_WRIST, 1]
        elif position_source == "midpoint":
            u_c = 0.5 * (kp2d[IDX_THUMB_TIP, 0] + kp2d[IDX_INDEX_TIP, 0])
            v_c = 0.5 * (kp2d[IDX_THUMB_TIP, 1] + kp2d[IDX_INDEX_TIP, 1])
        else:
            raise ValueError(f"unknown position_source={position_source!r}; expected 'wrist' or 'midpoint'")
        p_cam, ok_depth, z_mm = deproject(u_c, v_c, depth, intr,
                                          depth_min_mm=depth_min_mm,
                                          depth_max_mm=depth_max_mm)
        depth_mm[k] = z_mm
        if not ok_depth:
            n_no_depth += 1
            if k % log_every == 0:
                print(f"  [{k}/{N}] no depth at ({u_c:.1f},{v_c:.1f})  z_mm={z_mm:.0f}", flush=True)
            continue

        # Lift to world frame
        p_w = R_c2w @ p_cam + t_c2w
        # Workspace bounds sanity: reject if outside the calibrated workspace box
        if not in_workspace(p_w, workspace_bounds):
            n_out_workspace += 1
            if k % log_every == 0:
                print(f"  [{k}/{N}] out-of-workspace p_w={p_w}  bounds={workspace_bounds}", flush=True)
            continue

        # Algorithm 1 rotation, in camera frame
        side_for_alg = "R" if side_int == 1 else "L"
        R_cam = algorithm1_so3(kp3d, side_for_alg)

        p_world[k] = p_w
        R_world[k] = R_c2w @ R_cam
        valid[k] = True
        if k % log_every == 0:
            print(
                f"  [{k}/{N}] ok side={side_for_alg} fb={fb} z={z_mm/1000:.3f}m p_w={p_world[k]}",
                flush=True,
            )
    dt = time.time() - t0
    print(
        f"  cam done in {dt:.1f}s ({N/dt:.1f} fps). valid={valid.sum()}/{N}  no_det={n_no_det}  no_depth={n_no_depth}  out_workspace={n_out_workspace}",
        flush=True,
    )
    return {
        "ts_ms": ts_ms,
        "p_world": p_world,
        "R_world": R_world,
        "valid": valid,
        "side_detected": side_det,
        "fallback_used": fallback_used,
        "depth_mm": depth_mm,
    }


# ---------------- per-camera outlier rejection ----------------
def filter_outliers_by_neighbors(traj, window_frames=5, max_dev_mm=100.0):
    """
    Reject raw-valid frames whose position is more than max_dev_mm from the
    median of their temporal neighbors (±window_frames). Catches single-frame
    detections where the depth happened to pick up a wrong-but-plausible value
    (e.g., box surface instead of hand) — these pass the absolute depth/workspace
    bounds but are spatially inconsistent with the surrounding trajectory.

    Operates only on raw-valid frames. Invalidated frames will be re-bridged by
    fill_gaps using the surviving good neighbors.

    Returns a new traj with 'valid' tightened and adds 'n_rejected' int.
    """
    N = len(traj["ts_ms"])
    valid = traj["valid"].copy()
    p = traj["p_world"]
    valid_orig = traj["valid"].copy()
    n_rejected = 0

    valid_idx_orig = np.where(valid_orig)[0]
    if len(valid_idx_orig) < 3:
        out = dict(traj)
        out["valid"] = valid
        out["n_rejected"] = 0
        return out

    for k in valid_idx_orig:
        # collect neighbor positions within ±window_frames index distance
        lo = max(0, k - window_frames)
        hi = min(N, k + window_frames + 1)
        nbr_mask = valid_orig[lo:hi].copy()
        # exclude self
        nbr_mask[k - lo] = False
        if nbr_mask.sum() < 2:
            continue  # not enough context to judge
        nbr_pos = p[lo:hi][nbr_mask]
        med = np.median(nbr_pos, axis=0)
        if np.linalg.norm(p[k] - med) * 1000 > max_dev_mm:
            valid[k] = False
            n_rejected += 1

    out = dict(traj)
    out["valid"] = valid
    out["n_rejected"] = n_rejected
    return out


# ---------------- per-camera gap fill ----------------
def fill_gaps(traj, max_gap_frames=10):
    """
    Linearly interpolate invalid frames in a per-camera trajectory if the
    nearest valid neighbors on both sides are within max_gap_frames.

    Position: linear interp between neighbors. Rotation: SLERP via quaternions.
    Boundaries (no valid sample on one side) are NOT extrapolated.

    Returns a new traj dict with:
      - 'valid' denser (formerly-invalid frames now True if they were filled)
      - new 'interpolated' array (True for filled frames, False for raw observations)
      - 'n_filled' int count
    The original 'p_world', 'R_world', 'side_detected', 'fallback_used' arrays
    are copied (filled values written for newly-valid frames).
    """
    N = len(traj["ts_ms"])
    ts = traj["ts_ms"]
    valid_orig = traj["valid"].copy()
    valid = valid_orig.copy()
    p_world = traj["p_world"].copy()
    R_world = traj["R_world"].copy()
    interpolated = np.zeros(N, dtype=bool)
    n_filled = 0

    valid_indices = np.where(valid_orig)[0]
    if len(valid_indices) < 2:
        out = dict(traj)
        out["valid"] = valid
        out["interpolated"] = interpolated
        out["n_filled"] = 0
        return out

    for k in range(N):
        if valid_orig[k]:
            continue
        # nearest original-valid neighbors
        pos = np.searchsorted(valid_indices, k)
        if pos == 0 or pos == len(valid_indices):
            continue  # boundary: no neighbor on one side
        i_left = int(valid_indices[pos - 1])
        i_right = int(valid_indices[pos])
        gap = max(k - i_left, i_right - k)
        if gap > max_gap_frames:
            continue
        t_l, t_r, t_k = ts[i_left], ts[i_right], ts[k]
        alpha = (t_k - t_l) / max(t_r - t_l, 1e-6)
        p_world[k] = (1 - alpha) * traj["p_world"][i_left] + alpha * traj["p_world"][i_right]
        q_l = rotmat_to_quat(traj["R_world"][i_left])
        q_r = rotmat_to_quat(traj["R_world"][i_right])
        q_k = slerp_quat(q_l, q_r, alpha)
        R_world[k] = quat_to_rotmat(q_k)
        valid[k] = True
        interpolated[k] = True
        n_filled += 1

    out = dict(traj)
    out["valid"] = valid
    out["p_world"] = p_world
    out["R_world"] = R_world
    out["interpolated"] = interpolated
    out["n_filled"] = n_filled
    return out


# ---------------- bimanual time alignment ----------------
def resample_to_grid(traj, grid_ms, max_gap_ms=33.0):
    """
    Resample one camera's trajectory onto a target time grid.
    Linear interp on position, SLERP on rotation between nearest *valid* anchors.
    Mark sample invalid if the further anchor is >max_gap_ms away.

    Propagates per-camera 'interpolated' flag: out_interp[k] = True if either
    anchor used in resampling was itself an interpolated (gap-filled) frame.
    """
    ts = traj["ts_ms"]
    p = traj["p_world"]
    R = traj["R_world"]
    val = traj["valid"]
    side = traj["side_detected"]
    interp_in = traj.get("interpolated", np.zeros(len(ts), dtype=bool))

    # Pre-compute valid index list and quaternions
    valid_idx = np.where(val)[0]
    if len(valid_idx) == 0:
        N = grid_ms.shape[0]
        return {
            "p": np.zeros((N, 3)),
            "R": np.tile(np.eye(3), (N, 1, 1)),
            "q": np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (N, 1)),
            "valid": np.zeros(N, dtype=bool),
            "interp": np.zeros(N, dtype=bool),
            "side": -1 * np.ones(N, dtype=np.int8),
        }
    valid_ts = ts[valid_idx]
    valid_p = p[valid_idx]
    valid_q = np.stack([rotmat_to_quat(R[i]) for i in valid_idx])
    valid_side = side[valid_idx]
    valid_interp = interp_in[valid_idx]

    N = grid_ms.shape[0]
    out_p = np.zeros((N, 3))
    out_q = np.zeros((N, 4))
    out_R = np.zeros((N, 3, 3))
    out_valid = np.zeros(N, dtype=bool)
    out_interp = np.zeros(N, dtype=bool)
    out_side = -1 * np.ones(N, dtype=np.int8)

    for k, t in enumerate(grid_ms):
        # Find nearest valid anchors around t
        right = np.searchsorted(valid_ts, t)
        left = right - 1
        # Boundary handling
        if right == 0:
            i = 0
            if abs(valid_ts[i] - t) > max_gap_ms:
                continue
            out_p[k] = valid_p[i]
            out_q[k] = valid_q[i]
            out_R[k] = quat_to_rotmat(out_q[k])
            out_valid[k] = True
            out_interp[k] = bool(valid_interp[i])
            out_side[k] = valid_side[i]
            continue
        if right == len(valid_ts):
            i = right - 1
            if abs(valid_ts[i] - t) > max_gap_ms:
                continue
            out_p[k] = valid_p[i]
            out_q[k] = valid_q[i]
            out_R[k] = quat_to_rotmat(out_q[k])
            out_valid[k] = True
            out_interp[k] = bool(valid_interp[i])
            out_side[k] = valid_side[i]
            continue
        t_l, t_r = valid_ts[left], valid_ts[right]
        gap = max(abs(t_l - t), abs(t_r - t))
        if gap > max_gap_ms:
            # Allow interpolation only if both anchors are reasonably close
            continue
        alpha = (t - t_l) / max(t_r - t_l, 1e-6)
        out_p[k] = (1 - alpha) * valid_p[left] + alpha * valid_p[right]
        out_q[k] = slerp_quat(valid_q[left], valid_q[right], alpha)
        out_R[k] = quat_to_rotmat(out_q[k])
        out_valid[k] = True
        out_interp[k] = bool(valid_interp[left] or valid_interp[right])
        # Take side of nearer anchor
        out_side[k] = valid_side[left] if abs(t - t_l) <= abs(t - t_r) else valid_side[right]
    return {"p": out_p, "R": out_R, "q": out_q, "valid": out_valid, "interp": out_interp, "side": out_side}


def filter_bundle_outliers(bundle, k_neighbors=10, max_dev_mm=100.0, max_iters=5):
    """Bundle-level outlier filter. Catches grid samples whose position deviates
    from local context, regardless of whether they originated from raw detections
    or from fill_gaps with a bad anchor.

    Reference set = RAW (non-interp) valid grid samples only. This prevents a
    cluster of bad interpolated samples from polluting their own reference median.

    For each valid grid frame:
      - find K nearest RAW valid samples by time index (excluding self)
      - if no raw samples exist, fall back to K nearest valid samples
      - reject if |p[k] - median(neighbors)| > max_dev_mm

    Runs iteratively up to max_iters passes; each pass uses the current valid
    mask as input, so newly-invalidated samples don't pollute later passes.
    Stops early if a pass rejects nothing.
    """
    n_rej_total = {"L": 0, "R": 0}
    for side in ("L", "R"):
        valid_key = f"valid_{side}"
        interp_key = f"interp_{side}"
        p_key = f"p_{side}"
        valid = bundle[valid_key]
        p = bundle[p_key]
        interp = bundle[interp_key]
        for _it in range(max_iters):
            valid_now = valid.copy()
            raw_idx = np.where(valid_now & ~interp)[0]
            valid_idx = np.where(valid_now)[0]
            if len(valid_idx) < k_neighbors + 1:
                break
            rej_this_iter = 0
            for k in valid_idx:
                # Prefer RAW-only references; fall back to any valid if no raw available
                if len(raw_idx) >= 2:
                    candidates = raw_idx[raw_idx != k]
                else:
                    candidates = valid_idx[valid_idx != k]
                nearest = candidates[np.argsort(np.abs(candidates - k))[:k_neighbors]]
                if len(nearest) < 2:
                    continue
                med = np.median(p[nearest], axis=0)
                if np.linalg.norm(p[k] - med) * 1000 > max_dev_mm:
                    valid[k] = False
                    interp[k] = False
                    rej_this_iter += 1
            n_rej_total[side] += rej_this_iter
            if rej_this_iter == 0:
                break
        bundle[valid_key] = valid
        bundle[interp_key] = interp
    return bundle, n_rej_total


def align_bimanual(traj_L, traj_R, fps=30.0):
    """
    Build a common 30 Hz time grid over the device_ms overlap window of both cameras,
    then resample each onto it.
    """
    ts_L, ts_R = traj_L["ts_ms"], traj_R["ts_ms"]
    t_start = max(ts_L[0], ts_R[0])
    t_end = min(ts_L[-1], ts_R[-1])
    if t_end <= t_start:
        raise RuntimeError(f"No time overlap between cameras (L:{ts_L[0]:.0f}-{ts_L[-1]:.0f}, R:{ts_R[0]:.0f}-{ts_R[-1]:.0f})")
    step_ms = 1000.0 / fps
    N = int(np.floor((t_end - t_start) / step_ms)) + 1
    grid = t_start + np.arange(N) * step_ms
    print(f"  bimanual grid: {N} samples @ {fps} fps over [{t_start:.0f}, {t_end:.0f}] ms (span {t_end-t_start:.0f} ms)")
    res_L = resample_to_grid(traj_L, grid)
    res_R = resample_to_grid(traj_R, grid)
    return {
        "ts_ms": grid,
        "p_L": res_L["p"], "R_L": res_L["R"], "q_L": res_L["q"],
        "valid_L": res_L["valid"], "interp_L": res_L["interp"], "side_L_detected": res_L["side"],
        "p_R": res_R["p"], "R_R": res_R["R"], "q_R": res_R["q"],
        "valid_R": res_R["valid"], "interp_R": res_R["interp"], "side_R_detected": res_R["side"],
    }


# ---------------- IO and viz ----------------
def save_outputs(out_dir, bundle):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "trajectory.npz", **bundle)

    csv_path = out_dir / "trajectory.csv"
    with open(csv_path, "w") as f:
        f.write(
            "ts_ms,"
            "valid_L,interp_L,side_L,pLx,pLy,pLz,qLx,qLy,qLz,qLw,"
            "valid_R,interp_R,side_R,pRx,pRy,pRz,qRx,qRy,qRz,qRw\n"
        )
        for k in range(len(bundle["ts_ms"])):
            f.write(
                f"{bundle['ts_ms'][k]:.3f},"
                f"{int(bundle['valid_L'][k])},{int(bundle['interp_L'][k])},{int(bundle['side_L_detected'][k])},"
                f"{bundle['p_L'][k,0]:.6f},{bundle['p_L'][k,1]:.6f},{bundle['p_L'][k,2]:.6f},"
                f"{bundle['q_L'][k,0]:.6f},{bundle['q_L'][k,1]:.6f},{bundle['q_L'][k,2]:.6f},{bundle['q_L'][k,3]:.6f},"
                f"{int(bundle['valid_R'][k])},{int(bundle['interp_R'][k])},{int(bundle['side_R_detected'][k])},"
                f"{bundle['p_R'][k,0]:.6f},{bundle['p_R'][k,1]:.6f},{bundle['p_R'][k,2]:.6f},"
                f"{bundle['q_R'][k,0]:.6f},{bundle['q_R'][k,1]:.6f},{bundle['q_R'][k,2]:.6f},{bundle['q_R'][k,3]:.6f}\n"
            )
    print(f"  saved {out_dir/'trajectory.npz'} and {csv_path}")


def plot_trajectory(bundle, out_png, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    if bundle["valid_L"].any():
        m_all = bundle["valid_L"]
        m_raw = m_all & ~bundle["interp_L"]
        m_int = m_all & bundle["interp_L"]
        p_all = bundle["p_L"][m_all]
        ax.plot(p_all[:, 0], p_all[:, 1], p_all[:, 2], "-", color="#9bbcff", linewidth=1, alpha=0.6)
        if m_raw.any():
            p = bundle["p_L"][m_raw]
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], c="b", s=10, label=f"L raw (n={m_raw.sum()})")
        if m_int.any():
            p = bundle["p_L"][m_int]
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], c="#9bbcff", s=10, marker="x", label=f"L interp (n={m_int.sum()})")
        # start/end of full valid stream
        p_all_idx = np.where(m_all)[0]
        ax.scatter(bundle["p_L"][p_all_idx[0], 0], bundle["p_L"][p_all_idx[0], 1], bundle["p_L"][p_all_idx[0], 2],
                   c="b", s=80, marker="o", edgecolor="k", label="L start")
        ax.scatter(bundle["p_L"][p_all_idx[-1], 0], bundle["p_L"][p_all_idx[-1], 1], bundle["p_L"][p_all_idx[-1], 2],
                   c="b", s=80, marker="^", edgecolor="k", label="L end")
    if bundle["valid_R"].any():
        m_all = bundle["valid_R"]
        m_raw = m_all & ~bundle["interp_R"]
        m_int = m_all & bundle["interp_R"]
        p_all = bundle["p_R"][m_all]
        ax.plot(p_all[:, 0], p_all[:, 1], p_all[:, 2], "-", color="#ffb09b", linewidth=1, alpha=0.6)
        if m_raw.any():
            p = bundle["p_R"][m_raw]
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], c="r", s=10, label=f"R raw (n={m_raw.sum()})")
        if m_int.any():
            p = bundle["p_R"][m_int]
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], c="#ffb09b", s=10, marker="x", label=f"R interp (n={m_int.sum()})")
        p_all_idx = np.where(m_all)[0]
        ax.scatter(bundle["p_R"][p_all_idx[0], 0], bundle["p_R"][p_all_idx[0], 1], bundle["p_R"][p_all_idx[0], 2],
                   c="r", s=80, marker="o", edgecolor="k", label="R start")
        ax.scatter(bundle["p_R"][p_all_idx[-1], 0], bundle["p_R"][p_all_idx[-1], 1], bundle["p_R"][p_all_idx[-1], 2],
                   c="r", s=80, marker="^", edgecolor="k", label="R end")
    # Tabletop plane at z≈0.02 m (from extrinsics tabletop_refine)
    xx, yy = np.meshgrid(np.linspace(-0.5, 0.5, 5), np.linspace(-0.5, 0.5, 5))
    ax.plot_surface(xx, yy, 0.02 * np.ones_like(xx), alpha=0.1, color="gray")
    ax.set_xlabel("X world [m]")
    ax.set_ylabel("Y world [m]")
    ax.set_zlabel("Z world [m]")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()
    print(f"  plot → {out_png}")


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, help="Path to clip dir (contains camera_extrinsics.json + per-serial subfolders)")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_frames", type=int, default=-1, help="Debug: cap frames per camera (-1 = all)")
    ap.add_argument("--max_gap_frames", type=int, default=10,
                    help="Per-camera: max consecutive invalid frames to linearly interpolate over (default 10 = ~333 ms at 30 fps)")
    ap.add_argument("--position_source", choices=["wrist", "midpoint"], default="wrist",
                    help="Position pixel: 'wrist' (MANO joint 0) or 'midpoint' (thumb_tip + index_tip midpoint, BiNoMaP §3.2)")
    ap.add_argument("--depth_min_mm", type=float, default=100.0,
                    help="Reject deprojections with depth < this (mm). Default 100mm.")
    ap.add_argument("--depth_max_mm", type=float, default=1500.0,
                    help="Reject deprojections with depth > this (mm). Default 1500mm (workspace is <1m).")
    ap.add_argument("--workspace_box", default="-0.7,0.7,-0.7,0.7,-0.05,0.7",
                    help="World-frame bounding box xmin,xmax,ymin,ymax,zmin,zmax in meters. "
                         "Samples outside are rejected as outliers.")
    ap.add_argument("--outlier_window", type=int, default=5,
                    help="Per-camera rolling-median outlier filter: window of neighbor frames (±N) per raw frame.")
    ap.add_argument("--bundle_k_neighbors", type=int, default=10,
                    help="Bundle-level outlier filter: number of K-nearest valid neighbors by time to compare each grid frame against.")
    ap.add_argument("--outlier_max_dev_mm", type=float, default=100.0,
                    help="Outlier filters: reject frame if it deviates from neighbor median by > this (mm).")
    args = ap.parse_args()
    ws = tuple(float(x) for x in args.workspace_box.split(","))
    assert len(ws) == 6, f"--workspace_box needs 6 floats, got {len(ws)}"

    print(f"=== BiNoMaP Stage 1 trajectory extraction ===")
    print(f"clip:    {args.clip}")
    print(f"out_dir: {args.out_dir}")
    print(f"device:  {args.device}")

    cams = load_clip(args.clip)
    print(f"loaded {len(cams)} cameras: " + ", ".join(f"{s}({c['arm']})" for s, c in cams.items()))

    if args.max_frames > 0:
        for s in cams:
            cams[s]["frames"] = cams[s]["frames"][: args.max_frames]
            print(f"  (debug) truncated {s} to {len(cams[s]['frames'])} frames")

    runner = WilorRunner(device=args.device)

    print(f"position_source: {args.position_source}")
    print(f"depth bounds:    [{args.depth_min_mm}, {args.depth_max_mm}] mm")
    print(f"workspace box:   {ws}")
    print(f"\n--- processing left-arm cam {SERIAL_L} (expected side=L) ---")
    traj_L = process_camera(cams[SERIAL_L], expected_side="L", runner=runner,
                            position_source=args.position_source,
                            depth_min_mm=args.depth_min_mm,
                            depth_max_mm=args.depth_max_mm,
                            workspace_bounds=ws)
    print(f"\n--- processing right-arm cam {SERIAL_R} (expected side=R) ---")
    traj_R = process_camera(cams[SERIAL_R], expected_side="R", runner=runner,
                            position_source=args.position_source,
                            depth_min_mm=args.depth_min_mm,
                            depth_max_mm=args.depth_max_mm,
                            workspace_bounds=ws)

    print(f"\n--- per-camera outlier filter (window=±{args.outlier_window}, max_dev={args.outlier_max_dev_mm}mm) ---")
    n_before_L = int(traj_L["valid"].sum())
    n_before_R = int(traj_R["valid"].sum())
    traj_L = filter_outliers_by_neighbors(traj_L, args.outlier_window, args.outlier_max_dev_mm)
    traj_R = filter_outliers_by_neighbors(traj_R, args.outlier_window, args.outlier_max_dev_mm)
    print(f"  L cam: rejected {traj_L['n_rejected']} of {n_before_L} raw frames")
    print(f"  R cam: rejected {traj_R['n_rejected']} of {n_before_R} raw frames")

    print(f"\n--- per-camera gap fill (max_gap_frames={args.max_gap_frames}) ---")
    n_raw_L = int(traj_L["valid"].sum())
    n_raw_R = int(traj_R["valid"].sum())
    traj_L = fill_gaps(traj_L, max_gap_frames=args.max_gap_frames)
    traj_R = fill_gaps(traj_R, max_gap_frames=args.max_gap_frames)
    print(f"  L cam: raw_valid={n_raw_L}  filled={traj_L['n_filled']}  after_valid={int(traj_L['valid'].sum())}/{len(traj_L['valid'])}")
    print(f"  R cam: raw_valid={n_raw_R}  filled={traj_R['n_filled']}  after_valid={int(traj_R['valid'].sum())}/{len(traj_R['valid'])}")

    print(f"\n--- aligning bimanual trajectory ---")
    bundle = align_bimanual(traj_L, traj_R, fps=30.0)

    print(f"\n--- bundle-level outlier filter (k_neighbors={args.bundle_k_neighbors}, max_dev={args.outlier_max_dev_mm}mm) ---")
    bundle, n_bundle_rej = filter_bundle_outliers(bundle, args.bundle_k_neighbors, args.outlier_max_dev_mm)
    print(f"  L grid: rejected {n_bundle_rej['L']} samples")
    print(f"  R grid: rejected {n_bundle_rej['R']} samples")

    # SO(3) sanity
    n_check = min(10, len(bundle["ts_ms"]))
    for k in range(n_check):
        if bundle["valid_L"][k]:
            R = bundle["R_L"][k]
            assert abs(np.linalg.det(R) - 1) < 1e-3, f"det(R_L[{k}])={np.linalg.det(R)}"
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-3)
    print(f"SO(3) sanity check passed on first {n_check} valid samples")

    save_outputs(args.out_dir, bundle)
    plot_trajectory(bundle, Path(args.out_dir) / "trajectory.png", title=f"BiNoMaP P_coarse — {Path(args.clip).name}")

    # Stats summary
    pct_L = 100.0 * traj_L["valid"].sum() / len(traj_L["valid"])
    pct_R = 100.0 * traj_R["valid"].sum() / len(traj_R["valid"])
    print(f"\n=== summary ===")
    print(f"  left-cam valid frames (after fill):  {traj_L['valid'].sum()}/{len(traj_L['valid'])}  ({pct_L:.1f}%)  [{traj_L['n_filled']} filled]")
    print(f"  right-cam valid frames (after fill): {traj_R['valid'].sum()}/{len(traj_R['valid'])}  ({pct_R:.1f}%)  [{traj_R['n_filled']} filled]")
    print(f"  bimanual grid samples:  {len(bundle['ts_ms'])}")
    print(f"    valid_L={bundle['valid_L'].sum()}  (interp_L={bundle['interp_L'].sum()})")
    print(f"    valid_R={bundle['valid_R'].sum()}  (interp_R={bundle['interp_R'].sum()})")
    print(f"  left-cam side_detected: L={(traj_L['side_detected']==0).sum()}  R={(traj_L['side_detected']==1).sum()}  miss={(traj_L['side_detected']==-1).sum()}  fallback={traj_L['fallback_used'].sum()}")
    print(f"  right-cam side_detected: L={(traj_R['side_detected']==0).sum()}  R={(traj_R['side_detected']==1).sum()}  miss={(traj_R['side_detected']==-1).sum()}  fallback={traj_R['fallback_used'].sum()}")


if __name__ == "__main__":
    main()
