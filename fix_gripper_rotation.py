"""Override per-frame gripper rotations in a smoothed bundle so each gripper's
palm normal points at the OTHER gripper — i.e. perpendicular to the box face
each hand is contacting in a sandwich grip.

Why: MediaPipe + Algorithm-1 sometimes lands palm normals in roughly the right
direction at the start, then they go stale as the box rotates during a flip
demo (the underlying MANO frame doesn't track the box-axis flip). For a clean
sandwich grip the geometric constraint is simple: the gripper's approach axis
is just the inter-arm direction.

Each new rotation:
  z_axis (palm normal / approach) = unit(p_other - p_self)
  x_axis (gripper opening line)   = world_up - (world_up · z) z, then normalized
  y_axis                          = z × x

If z is too close to world_up (rare for our setup), fall back to world_right.

CLI:
  python fix_gripper_rotation.py --in_npz outputs/recordings_2/wrist/trajectory_smoothed.npz \\
                                 --out_npz outputs/recordings_2/wrist/trajectory_smoothed_fixed_rot.npz
"""
import argparse
import pathlib

import numpy as np


def rotmat_to_quat_xyzw(R):
    """3x3 rotation matrix → quaternion (x, y, z, w). Robust shepperd's method."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


def build_rotation_from_z(z_axis, world_up=np.array([0.0, 0.0, 1.0])):
    """Return a right-handed orthonormal R with z column = z_axis.

    x = world_up - (world_up · z) z, normalized. If degenerate, use world_x.
    y = z × x.
    """
    z = z_axis / np.linalg.norm(z_axis)
    x = world_up - (world_up @ z) * z
    if np.linalg.norm(x) < 1e-3:
        # z is nearly parallel to world_up — pick world_x as fallback "x in plane"
        fallback = np.array([1.0, 0.0, 0.0])
        x = fallback - (fallback @ z) * z
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)
    return np.column_stack([x, y, z])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_npz", type=pathlib.Path, required=True)
    ap.add_argument("--out_npz", type=pathlib.Path, default=None,
                    help="Default: <in_npz>_fixed_rot.npz")
    args = ap.parse_args()

    out = args.out_npz or args.in_npz.with_name(args.in_npz.stem + "_fixed_rot.npz")
    b = dict(np.load(args.in_npz))
    pL, pR = b["p_L"], b["p_R"]
    vL, vR = b["valid_L"].astype(bool), b["valid_R"].astype(bool)
    N = pL.shape[0]

    print(f"input: {args.in_npz}  N={N}")
    print(f"valid_L={int(vL.sum())}  valid_R={int(vR.sum())}")

    R_L_new = b["R_L"].copy()
    R_R_new = b["R_R"].copy()
    q_L_new = b["q_L"].copy() if "q_L" in b else np.zeros((N, 4))
    q_R_new = b["q_R"].copy() if "q_R" in b else np.zeros((N, 4))

    n_fixed = 0
    for t in range(N):
        # Need both arms valid to define inter-arm axis. For singly-valid frames,
        # leave the rotation alone.
        if not (vL[t] and vR[t]):
            continue
        v = pR[t] - pL[t]
        if np.linalg.norm(v) < 1e-6:
            continue
        R_L_new[t] = build_rotation_from_z(+v)   # L palm normal points TOWARD R
        R_R_new[t] = build_rotation_from_z(-v)   # R palm normal points TOWARD L
        q_L_new[t] = rotmat_to_quat_xyzw(R_L_new[t])
        q_R_new[t] = rotmat_to_quat_xyzw(R_R_new[t])
        n_fixed += 1

    print(f"\nrewrote rotations on {n_fixed} jointly-valid frames")

    # Sanity-check the new alignment
    both = vL & vR
    if both.any():
        dots_L, dots_R = [], []
        for t in np.where(both)[0]:
            v = pR[t] - pL[t]; v = v / np.linalg.norm(v)
            dots_L.append(float(R_L_new[t, :, 2] @ v))
            dots_R.append(float(R_R_new[t, :, 2] @ (-v)))
        print(f"sanity:  L palm·(L→R)  min={min(dots_L):.4f}  mean={np.mean(dots_L):.4f}  "
              f"max={max(dots_L):.4f}  (should be 1.0)")
        print(f"         R palm·(R→L)  min={min(dots_R):.4f}  mean={np.mean(dots_R):.4f}  "
              f"max={max(dots_R):.4f}  (should be 1.0)")

    # Preserve originals for comparison + write the fixed bundle.
    b["R_L_orig_rot"] = b["R_L"]
    b["R_R_orig_rot"] = b["R_R"]
    b["q_L_orig_rot"] = b.get("q_L", np.zeros((N, 4)))
    b["q_R_orig_rot"] = b.get("q_R", np.zeros((N, 4)))
    b["R_L"] = R_L_new
    b["R_R"] = R_R_new
    b["q_L"] = q_L_new
    b["q_R"] = q_R_new
    b["rotation_fix_applied"] = np.bool_(True)
    np.savez_compressed(out, **b)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
