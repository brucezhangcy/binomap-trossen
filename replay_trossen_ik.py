"""
BiNoMaP Trossen MuJoCo Replay v2 — joint-space IK.

v1 used MuJoCo's mocap+weld constraint (a soft spring); when (p, R) was
unreachable the EE settled at a least-squares compromise ~14 cm from target.
v2 does **real inverse kinematics**: per-frame damped least-squares solve on
the 6-DOF kinematic chain, then send joint targets to the position-controlled
actuators in scene_joint.xml.

Pipeline:
  1. Load scene_joint.xml (14 position actuators: 6 joints + 1 gripper per arm).
  2. Frame transform (translate-only) from our world to Trossen world.
  3. For each grid frame: damped-LS IK on each arm → 6-DoF joint target.
  4. Replay loop: send ctrl = joint_targets, step physics, render.

CLI:
  python replay_trossen_ik.py \\
      --in_npz outputs/recordings_1/wrist/trajectory_smoothed.npz \\
      --record_mp4 outputs/recordings_1/wrist/trossen_replay_ik.mp4 \\
      [--position_only] [--speed 1.0]
"""
import argparse
import os
import time
from pathlib import Path

# Headless OpenGL must be set before importing mujoco
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
import mujoco.viewer

# Trossen scene reference points (from scene_mocap.xml — same world frame)
TROSSEN_MOCAP_LEFT_REST = np.array([-0.019982, 0.212613, 0.202586])
TROSSEN_MOCAP_RIGHT_REST = np.array([-0.019982, -0.212613, 0.202586])
TROSSEN_WORKSPACE_CENTER = 0.5 * (TROSSEN_MOCAP_LEFT_REST + TROSSEN_MOCAP_RIGHT_REST)
TROSSEN_TABLE_Z = 0.02


def quat_xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]])


def quat_wxyz_to_rotmat(q_wxyz):
    """MuJoCo (w, x, y, z) → 3x3 rotation matrix via mujoco helper."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q_wxyz)
    return R.reshape(3, 3)


def rotmat_to_quat_wxyz(R):
    """3x3 rotation matrix → MuJoCo (w, x, y, z)."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


def compute_orientation_remap(rest_quat_mj, our_quat_xyzw_anchor):
    """Build R_offset (3,3) such that R_offset @ R_ours_anchor = R_trossen_rest.

    Equivalently: at the anchor orientation, the remapped target equals the
    Trossen EE's natural rest orientation. Later frames are remapped consistently
    so the entire orientation trajectory lands inside Trossen's reachable envelope.
    Matches the spirit of BiNoMaP §Appendix E's 'XYZ axis remap' for
    cross-embodiment transfer, generalised to a full SO(3) rotation (per arm).
    """
    R_trossen = quat_wxyz_to_rotmat(rest_quat_mj)
    our_wxyz = quat_xyzw_to_wxyz(our_quat_xyzw_anchor)
    R_ours = quat_wxyz_to_rotmat(our_wxyz)
    return R_trossen @ R_ours.T


def mean_quat_xyzw(quats_xyzw, valid_mask):
    """Average orientation across valid frames (Markley's method-ish).

    Pre-flips all quaternions onto the same hemisphere (sign disambiguation),
    then takes the elementwise mean and renormalizes. Good enough when the
    trajectory's rotations stay within ~90° of the reference; for wider spreads
    use principal-eigenvector method (not implemented; not needed here)."""
    idx = np.where(valid_mask)[0]
    ref = quats_xyzw[idx[0]]
    flipped = quats_xyzw[idx].copy()
    for i in range(len(flipped)):
        if np.dot(flipped[i], ref) < 0:
            flipped[i] = -flipped[i]
    m = flipped.mean(axis=0)
    return m / max(np.linalg.norm(m), 1e-12)


def rot_mat_log(R):
    """Axis-angle (rotation vector, 3,) from rotation matrix (3,3)."""
    cos_theta = (np.trace(R) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if abs(theta) < 1e-8:
        return np.zeros(3)
    if abs(theta - np.pi) < 1e-6:
        # near-pi case: extract axis from diag of R + I
        d = np.diag(R) + 1.0
        axis = np.sqrt(np.maximum(d * 0.5, 0))
        # disambiguate signs from off-diagonals
        if R[0, 1] < 0: axis[1] = -axis[1]
        if R[0, 2] < 0: axis[2] = -axis[2]
        return theta * axis
    skew = (R - R.T) * (0.5 / np.sin(theta))
    return theta * np.array([skew[2, 1], skew[0, 2], skew[1, 0]])


def compute_frame_alignment(p_L, p_R, valid_L, valid_R, our_table_z=0.02):
    cL = p_L[valid_L].mean(axis=0)
    cR = p_R[valid_R].mean(axis=0)
    ours_mid = 0.5 * (cL + cR)
    target = TROSSEN_WORKSPACE_CENTER.copy()
    target[2] = TROSSEN_TABLE_Z + (ours_mid[2] - our_table_z)
    offset = target - ours_mid
    return offset, ours_mid, target


def get_arm_indices(model, side):
    """Return (q_idx, v_idx, act_idx, ee_body_id, jnt_range) for one arm.
    side ∈ {'left', 'right'}."""
    joint_names = [f"follower_{side}_joint_{i}" for i in range(6)]
    q_idx = np.array([model.joint(j).qposadr[0] for j in joint_names])
    v_idx = np.array([model.joint(j).dofadr[0] for j in joint_names])
    act_idx = np.array([model.actuator(n).id for n in joint_names])  # actuator name == joint name
    ee_body_id = model.body(f"follower_{side}_link_6").id
    jnt_range = np.array([model.jnt_range[model.joint(j).id] for j in joint_names])
    return q_idx, v_idx, act_idx, ee_body_id, jnt_range


def ik_solve(model, data, target_pos, target_quat, arm_info,
             use_orientation=True, max_iters=200,
             tol_pos=1e-3, tol_rot=0.02,
             damping=0.05, step_clip=0.3,
             tip_offset_local=None):
    """Damped LS IK for one arm. Modifies data.qpos[q_idx] in place.

    tip_offset_local (3,) — offset in link_6 LOCAL frame of the point that IK
    should drive to target_pos. Default None = link_6 body origin (legacy).
    Pass e.g. [0.09, 0, 0] to make IK solve "gripper tip at target" (Trossen
    gripper extends ~9cm along link_6 local +x from link_6 origin).
    This decouples the IK target from the link_6 body, so position-only IK
    truly tracks the tip rather than link_6.

    Returns (pos_err [m], rot_err [rad], iters, converged_bool)."""
    q_idx, v_idx, _, ee_body_id, jnt_range = arm_info

    # Target rotation matrix from quat (w,x,y,z)
    target_R_flat = np.zeros(9)
    mujoco.mju_quat2Mat(target_R_flat, target_quat)
    target_R = target_R_flat.reshape(3, 3)

    if tip_offset_local is None:
        tip_offset_local = np.zeros(3)

    pos_err_norm = np.inf
    rot_err_norm = np.inf
    for it in range(max_iters):
        # Forward kinematics
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)

        # Compute IK control point: link_6 body origin + R_link_6 @ tip_offset_local
        link6_pos = data.xpos[ee_body_id]
        link6_R = data.xmat[ee_body_id].reshape(3, 3)
        cur_pos = (link6_pos + link6_R @ tip_offset_local).copy()
        cur_R = link6_R.copy()

        pos_err = target_pos - cur_pos
        if use_orientation:
            R_err = target_R @ cur_R.T
            rot_err = rot_mat_log(R_err)
        else:
            rot_err = np.zeros(3)

        pos_err_norm = float(np.linalg.norm(pos_err))
        rot_err_norm = float(np.linalg.norm(rot_err))

        if pos_err_norm < tol_pos and (not use_orientation or rot_err_norm < tol_rot):
            return pos_err_norm, rot_err_norm, it + 1, True

        # Jacobian at the IK control point (not link_6 origin)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, jacr, cur_pos, ee_body_id)

        if use_orientation:
            J = np.vstack([jacp[:, v_idx], jacr[:, v_idx]])  # (6, 6)
            err = np.concatenate([pos_err, rot_err])
            n = 6
        else:
            J = jacp[:, v_idx]  # (3, 6)
            err = pos_err
            n = 3

        # Damped least squares: dq = J^T (J J^T + λ²I)^-1 e
        damp = damping ** 2 * np.eye(n)
        dq = J.T @ np.linalg.solve(J @ J.T + damp, err)

        # Clip step magnitude per joint to avoid divergence
        dq = np.clip(dq, -step_clip, step_clip)

        new_q = data.qpos[q_idx] + dq
        data.qpos[q_idx] = np.clip(new_q, jnt_range[:, 0], jnt_range[:, 1])

    return pos_err_norm, rot_err_norm, max_iters, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", required=True)
    ap.add_argument("--scene_xml",
                    default="/data/bruce/BiNoMaP/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/scene_joint.xml")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--record_mp4", type=str, default=None)
    ap.add_argument("--mp4_fps", type=float, default=30.0)
    ap.add_argument("--our_table_z", type=float, default=0.02)
    ap.add_argument("--position_only", action="store_true",
                    help="Solve IK for position only (more permissive, ignores rotation)")
    ap.add_argument("--orientation_remap", action="store_true", default=True,
                    help="Apply a fixed per-arm SO(3) remap so Algorithm-1 orientations land in "
                         "Trossen's reachable envelope. Calibrated from the first valid frame "
                         "matching the EE's home-keyframe rest orientation. (BiNoMaP §App E)")
    ap.add_argument("--no_orientation_remap", dest="orientation_remap", action="store_false",
                    help="Disable orientation remap (use raw Algorithm-1 orientations).")
    ap.add_argument("--smooth_joints_sigma", type=float, default=2.0,
                    help="Gaussian filter sigma (in frames) applied to IK-output joint trajectories. "
                         "Removes visible wrist 'twist' artifacts from IK branch-jumping. 0 = disabled.")
    ap.add_argument("--gripper_offset_m", type=float, default=0.0,
                    help="Push each wrist target along the trajectory's approach axis (R column 2) "
                         "by this distance (meters). Compensates for Trossen gripper being longer than "
                         "the human hand's wrist-to-fingertip-midpoint (≈3 cm for ALOHA-WidowX vs human). "
                         "Sign convention: + moves wrist OUTWARD (away from the box-holding direction).")
    ap.add_argument("--render_w", type=int, default=640)
    ap.add_argument("--render_h", type=int, default=480)
    ap.add_argument("--camera", default="cam_high",
                    help="MuJoCo camera name for offscreen rendering")
    ap.add_argument("--warmup_seconds", type=float, default=1.0)
    args = ap.parse_args()

    bundle = np.load(args.in_npz)
    ts_ms = bundle["ts_ms"]
    p_L = bundle["p_L"].copy()
    q_L = bundle["q_L"]
    valid_L = bundle["valid_L"]
    p_R = bundle["p_R"].copy()
    q_R = bundle["q_R"]
    valid_R = bundle["valid_R"]
    N = len(ts_ms)

    print(f"=== Trossen Joint-Space IK Replay ===")
    print(f"input: {args.in_npz}  N={N}  valid_L={int(valid_L.sum())}  valid_R={int(valid_R.sum())}")
    print(f"orientation: {'IGNORED' if args.position_only else 'tracked'}")

    # Frame alignment (translate-only)
    offset, ours_mid, target = compute_frame_alignment(p_L, p_R, valid_L, valid_R, args.our_table_z)
    print(f"offset = {offset.round(3)}  (ours_mid={ours_mid.round(3)} → trossen_target={target.round(3)})")
    p_L_t = p_L + offset[None, :]
    p_R_t = p_R + offset[None, :]

    # Gripper-length compensation. Trossen ALOHA's gripper extends ~9 cm
    # forward from link_6 along link_6 local +x. The trajectory describes
    # where the human WRIST was; for the human, the gripper-contact point was
    # ~6 cm forward from that wrist. To make the Trossen GRIPPER TIP track the
    # human wrist (Bruce's directive), we pass tip_offset_local=[d, 0, 0] to
    # the IK so it solves for "gripper tip at wrist target" instead of
    # "link_6 at wrist target". d = +0.09 m for Trossen.
    # This works correctly in position-only mode because the offset is in
    # link_6 LOCAL coordinates (rotates with the joint chain), not in world.
    tip_offset_local = np.array([args.gripper_offset_m, 0.0, 0.0])
    if args.gripper_offset_m != 0.0:
        print(f"gripper IK control point: link_6 + {args.gripper_offset_m:+.3f}m × link_6_local_+x "
              f"(IK targets the gripper tip, not link_6)")

    q_L_mj = np.stack([quat_xyzw_to_wxyz(q_L[i]) for i in range(N)])
    q_R_mj = np.stack([quat_xyzw_to_wxyz(q_R[i]) for i in range(N)])

    # Load model
    model = mujoco.MjModel.from_xml_path(args.scene_xml)
    data = mujoco.MjData(model)
    arm_L = get_arm_indices(model, "left")
    arm_R = get_arm_indices(model, "right")

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    home_qL = data.qpos[arm_L[0]].copy()
    home_qR = data.qpos[arm_R[0]].copy()
    timestep = model.opt.timestep
    print(f"model: nq={model.nq} nv={model.nv} nu={model.nu} timestep={timestep}")
    print(f"home_qL = {home_qL.round(2)}")
    print(f"home_qR = {home_qR.round(2)}")

    # === Orientation remap (BiNoMaP §App E "XYZ axis remap", generalised) ===
    # Compute a fixed SO(3) rotation per arm that maps Algorithm-1's hand
    # orientations (built from human finger cross-products in the demonstrator's
    # frame) into Trossen's reachable orientation envelope. Calibration anchor:
    # at the first valid frame, the remapped orientation target should equal
    # the EE's natural rest orientation at the home keyframe. Later frames
    # inherit the same offset, so the *relative* orientation evolution from
    # the human video is preserved while the absolute frame is aligned.
    if args.orientation_remap and not args.position_only:
        # Need the EE's rest orientation in world frame after kinematics
        mujoco.mj_kinematics(model, data)
        rest_qL_mj = data.xquat[arm_L[3]].copy()  # (w, x, y, z) in world
        rest_qR_mj = data.xquat[arm_R[3]].copy()
        # Anchor = MEAN orientation across all valid frames (per arm). This
        # spreads the IK reachability load across the whole trajectory instead
        # of overfitting to the first frame (where R might happen to be in a
        # reachable corner but L drifts out as the trajectory evolves).
        anchor_qL = mean_quat_xyzw(q_L, valid_L)
        anchor_qR = mean_quat_xyzw(q_R, valid_R)
        R_remap_L = compute_orientation_remap(rest_qL_mj, anchor_qL)
        R_remap_R = compute_orientation_remap(rest_qR_mj, anchor_qR)
        print(f"orientation remap (anchor = mean-quaternion across valid frames):")
        print(f"  L anchor xyzw = {anchor_qL.round(3)}")
        print(f"  R anchor xyzw = {anchor_qR.round(3)}")

        # Apply remap to every frame's quaternion target (in place)
        for k in range(N):
            R_ours_L = quat_wxyz_to_rotmat(q_L_mj[k])
            R_remapped_L = R_remap_L @ R_ours_L
            q_L_mj[k] = rotmat_to_quat_wxyz(R_remapped_L)
            R_ours_R = quat_wxyz_to_rotmat(q_R_mj[k])
            R_remapped_R = R_remap_R @ R_ours_R
            q_R_mj[k] = rotmat_to_quat_wxyz(R_remapped_R)
    else:
        print(f"orientation remap: {'OFF (--no_orientation_remap)' if not args.orientation_remap else 'N/A (position_only)'}")

    # === Precompute joint targets via IK for every frame ===
    print(f"\n--- IK pass over {N} frames ---")
    joint_targets_L = np.tile(home_qL, (N, 1))
    joint_targets_R = np.tile(home_qR, (N, 1))
    ik_pos_err_L = np.full(N, np.nan)
    ik_pos_err_R = np.full(N, np.nan)
    ik_rot_err_L = np.full(N, np.nan)
    ik_rot_err_R = np.full(N, np.nan)
    ik_fail_L = 0
    ik_fail_R = 0

    last_qL = home_qL.copy()
    last_qR = home_qR.copy()
    n_fallback_L = 0
    n_fallback_R = 0
    # Per-frame fallback: if 6-DoF IK fails to converge or returns large pos error,
    # retry the same frame in position-only mode. This preserves trajectory rotation
    # everywhere it's reachable, and avoids position-vs-rotation compromise (which
    # otherwise drags the arm into bad poses, e.g. R-arm drifts toward L-arm at the
    # end of the box-pivot motion when human wrist rotates beyond Trossen joint limits).
    POS_FALLBACK_THRESH_M = 0.02   # 20 mm position error → drop rotation constraint
    t0 = time.time()
    for k in range(N):
        if valid_L[k]:
            data.qpos[arm_L[0]] = last_qL  # warm-start from previous solution
            pe, re, it, conv = ik_solve(model, data, p_L_t[k], q_L_mj[k], arm_L,
                                        use_orientation=not args.position_only,
                                        tip_offset_local=tip_offset_local)
            if (not args.position_only) and (not conv or pe > POS_FALLBACK_THRESH_M):
                # 6-DoF failed → retry position-only
                data.qpos[arm_L[0]] = last_qL
                pe, re, it, conv = ik_solve(model, data, p_L_t[k], q_L_mj[k], arm_L,
                                            use_orientation=False,
                                            tip_offset_local=tip_offset_local)
                n_fallback_L += 1
            joint_targets_L[k] = data.qpos[arm_L[0]]
            last_qL = joint_targets_L[k].copy()
            ik_pos_err_L[k] = pe
            ik_rot_err_L[k] = re
            if not conv:
                ik_fail_L += 1
        else:
            joint_targets_L[k] = last_qL

        if valid_R[k]:
            data.qpos[arm_R[0]] = last_qR
            pe, re, it, conv = ik_solve(model, data, p_R_t[k], q_R_mj[k], arm_R,
                                        use_orientation=not args.position_only,
                                        tip_offset_local=tip_offset_local)
            if (not args.position_only) and (not conv or pe > POS_FALLBACK_THRESH_M):
                data.qpos[arm_R[0]] = last_qR
                pe, re, it, conv = ik_solve(model, data, p_R_t[k], q_R_mj[k], arm_R,
                                            use_orientation=False,
                                            tip_offset_local=tip_offset_local)
                n_fallback_R += 1
            joint_targets_R[k] = data.qpos[arm_R[0]]
            last_qR = joint_targets_R[k].copy()
            ik_pos_err_R[k] = pe
            ik_rot_err_R[k] = re
            if not conv:
                ik_fail_R += 1
        else:
            joint_targets_R[k] = last_qR

        if (k % 25) == 0:
            print(f"  IK frame {k:3d}/{N}  L_pos={ik_pos_err_L[k]*1000:5.1f}mm L_rot={np.degrees(ik_rot_err_L[k]):5.1f}deg  "
                  f"R_pos={ik_pos_err_R[k]*1000:5.1f}mm R_rot={np.degrees(ik_rot_err_R[k]):5.1f}deg")
    print(f"IK pass done in {time.time()-t0:.1f}s. fail count L={ik_fail_L}/{int(valid_L.sum())}  "
          f"R={ik_fail_R}/{int(valid_R.sum())}")
    print(f"  per-frame position-only fallback (6-DoF→3-DoF when unreachable): "
          f"L={n_fallback_L}  R={n_fallback_R}")

    # First/last valid frames per arm — used by both smoothing (pin boundaries)
    # and the back-fill section below.
    first_valid_L = int(np.where(valid_L)[0][0])
    first_valid_R = int(np.where(valid_R)[0][0])
    last_valid_L = int(np.where(valid_L)[0][-1])
    last_valid_R = int(np.where(valid_R)[0][-1])

    # === Joint-space smoothing of IK output ===
    # Per-frame IK can "jump branches" when targets push it into different
    # solution basins (kinematic redundancy: same EE pose, multiple joint configs).
    # The result is visible joint-axis "twists" in the rendered video even when
    # EE tracking is fine. A small Gaussian on each joint independently removes
    # the twists; EE deviation from smoothing is sub-cm.
    #
    # First/last valid frames per arm are PINNED to their pre-smoothing IK value
    # so the start-point alignment to the human wrist trajectory is exact. The
    # smoothing kernel would otherwise drift these boundary frames toward
    # adjacent IK solutions (worst case: ~70 mm tip error if the trajectory has
    # fast motion right after the first valid frame, as in recordings_1 R-arm).
    if args.smooth_joints_sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        pinned_L = joint_targets_L[first_valid_L].copy()
        pinned_R = joint_targets_R[first_valid_R].copy()
        last_valid_L = int(np.where(valid_L)[0][-1])
        last_valid_R = int(np.where(valid_R)[0][-1])
        pinned_L_end = joint_targets_L[last_valid_L].copy()
        pinned_R_end = joint_targets_R[last_valid_R].copy()
        for j in range(6):
            joint_targets_L[:, j] = gaussian_filter1d(joint_targets_L[:, j], sigma=args.smooth_joints_sigma)
            joint_targets_R[:, j] = gaussian_filter1d(joint_targets_R[:, j], sigma=args.smooth_joints_sigma)
        # Restore pinned boundaries + back-fill before first-valid (since back-fill
        # ran before smoothing, it was also distorted)
        joint_targets_L[: first_valid_L + 1] = pinned_L
        joint_targets_R[: first_valid_R + 1] = pinned_R
        joint_targets_L[last_valid_L:] = pinned_L_end
        joint_targets_R[last_valid_R:] = pinned_R_end
        print(f"  joint-space smoothing applied: gaussian σ={args.smooth_joints_sigma} frames")
        print(f"    pinned first/last valid frames: L[{first_valid_L}, {last_valid_L}]  R[{first_valid_R}, {last_valid_R}]")

    # Back-fill pre-first-valid joint targets per arm: when an arm has no valid
    # detection for the first K frames (e.g. R-cam in recordings_1 misses
    # frames 0-44), joint_targets_*[0:K] would otherwise stay at home pose,
    # then jump to a far-away target on frame K. The position controller can't
    # track that jump in 33 ms → big spike at the start of the replay.
    # Filling those leading frames with the first valid frame's config lets
    # the warmup lerp smoothly to that config and the replay start clean.
    # (first_valid_L/R are already computed above for the joint smoothing pin.)
    if first_valid_L > 0:
        joint_targets_L[:first_valid_L] = joint_targets_L[first_valid_L]
    if first_valid_R > 0:
        joint_targets_R[:first_valid_R] = joint_targets_R[first_valid_R]
    print(f"first valid frame: L={first_valid_L}  R={first_valid_R}  "
          f"(back-filled {first_valid_L} L frames, {first_valid_R} R frames to remove startup discontinuity)")
    vL = ~np.isnan(ik_pos_err_L)
    vR = ~np.isnan(ik_pos_err_R)
    print(f"IK pos err (m): L mean={ik_pos_err_L[vL].mean()*1000:.1f}mm max={ik_pos_err_L[vL].max()*1000:.1f}mm  "
          f"R mean={ik_pos_err_R[vR].mean()*1000:.1f}mm max={ik_pos_err_R[vR].max()*1000:.1f}mm")
    if not args.position_only:
        print(f"IK rot err (deg): L mean={np.degrees(ik_rot_err_L[vL].mean()):.1f} max={np.degrees(ik_rot_err_L[vL].max()):.1f}  "
              f"R mean={np.degrees(ik_rot_err_R[vR].mean()):.1f} max={np.degrees(ik_rot_err_R[vR].max()):.1f}")

    # === Replay sim with joint targets ===
    print(f"\n--- replaying with joint-position control ---")
    mujoco.mj_resetDataKeyframe(model, data, 0)
    substeps_per_frame = max(1, int(round(1.0 / args.fps / timestep)))
    print(f"substeps/frame={substeps_per_frame}")

    # Warmup: lerp ctrl from home → frame 0 targets, give the position controllers
    # time to track in
    warmup_steps = max(int(args.warmup_seconds / timestep), 1)
    print(f"warmup: {warmup_steps} steps ({args.warmup_seconds:.2f}s)")
    for s in range(warmup_steps):
        a = (s + 1) / warmup_steps
        ctrl_L = (1 - a) * home_qL + a * joint_targets_L[0]
        ctrl_R = (1 - a) * home_qR + a * joint_targets_R[0]
        data.ctrl[arm_L[2]] = ctrl_L
        data.ctrl[arm_R[2]] = ctrl_R
        mujoco.mj_step(model, data)

    # Setup renderer
    renderer = None
    mp4_writer = None
    viewer = None
    if args.record_mp4:
        import imageio
        renderer = mujoco.Renderer(model, height=args.render_h, width=args.render_w)
        mp4_writer = imageio.get_writer(args.record_mp4, fps=args.mp4_fps, codec="libx264")
    elif not args.headless:
        viewer = mujoco.viewer.launch_passive(model, data)

    actual_pos_L = np.zeros((N, 3))
    actual_pos_R = np.zeros((N, 3))
    sleep_per_frame = 1.0 / (args.fps * max(args.speed, 1e-6))

    t_start = time.time()
    try:
        for k in range(N):
            data.ctrl[arm_L[2]] = joint_targets_L[k]
            data.ctrl[arm_R[2]] = joint_targets_R[k]
            for _ in range(substeps_per_frame):
                mujoco.mj_step(model, data)
            actual_pos_L[k] = data.xpos[arm_L[3]].copy()
            actual_pos_R[k] = data.xpos[arm_R[3]].copy()

            if viewer is not None:
                viewer.sync()
                elapsed = time.time() - t_start
                expected = (k + 1) * sleep_per_frame
                if elapsed < expected:
                    time.sleep(expected - elapsed)
            elif renderer is not None:
                renderer.update_scene(data, camera=args.camera)
                mp4_writer.append_data(renderer.render())

            if (k % 50) == 0:
                err = np.linalg.norm(actual_pos_L[k] - p_L_t[k]) * 1000
                print(f"  replay frame {k:3d}/{N}  L_track_err={err:5.1f}mm")
    finally:
        if mp4_writer is not None:
            mp4_writer.close()
        if viewer is not None:
            viewer.close()

    # Stats
    err_L = np.linalg.norm(actual_pos_L - p_L_t, axis=1)
    err_R = np.linalg.norm(actual_pos_R - p_R_t, axis=1)
    print(f"\n=== Actual tracking (sim EE vs trajectory target, after IK + low-level position control) ===")
    print(f"  L arm: mean={err_L[valid_L].mean()*1000:.1f}mm  "
          f"p95={np.percentile(err_L[valid_L], 95)*1000:.1f}mm  "
          f"max={err_L[valid_L].max()*1000:.1f}mm")
    print(f"  R arm: mean={err_R[valid_R].mean()*1000:.1f}mm  "
          f"p95={np.percentile(err_R[valid_R], 95)*1000:.1f}mm  "
          f"max={err_R[valid_R].max()*1000:.1f}mm")

    log_path = Path(args.in_npz).parent / "trossen_replay_ik_log.npz"
    np.savez(log_path,
             target_L=p_L_t, target_R=p_R_t,
             actual_L=actual_pos_L, actual_R=actual_pos_R,
             joint_L=joint_targets_L, joint_R=joint_targets_R,
             ik_pos_err_L=ik_pos_err_L, ik_pos_err_R=ik_pos_err_R,
             ik_rot_err_L=ik_rot_err_L, ik_rot_err_R=ik_rot_err_R,
             frame_offset=offset)
    print(f"\nsaved {log_path}")
    if args.record_mp4:
        print(f"saved {args.record_mp4}")
    print(f"wall-clock total: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
