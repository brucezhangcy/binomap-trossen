# BiNoMaP — Bimanual wrist-trajectory extraction + Trossen ALOHA replay

Reproduction of [BiNoMaP](https://arxiv.org/abs/2509.21256)'s Stage 1 (hand-trajectory extraction from human demonstration video) on a custom 2-camera RGB-D rig, followed by replay on the Trossen ALOHA bimanual MuJoCo simulator.

## Pipeline

```
2 RGB-D wrist-cam recordings (left + right arm)
        │
        ▼  extract_trajectory.py
WiLoR per-frame hand reconstruction
        │  + Algorithm 1 (cross-product SO(3) from MANO joints)
        │  + depth deprojection (real metric scale via intrinsics)
        │  + camera→world transform (hand-eye calibration)
        │  + per-camera outlier filter (depth bound + workspace AABB + rolling median)
        │  + fill_gaps (linear interp + SLERP across short detection gaps)
        ▼
trajectory.npz  (per-arm 6-DoF pose on uniform 30 Hz grid)
        │
        ▼  smooth_trajectory.py
BiNoMaP §3.3 Stage 2a motion smoothness optimization
        │  – plane fit (least squares + coplanarity constraint)
        │  – 2D cubic B-spline in plane
        │  – SLERP-between-anchors rotation smoothing
        ▼
trajectory_smoothed.npz
        │
        ▼  replay_trossen_ik.py
Trossen ALOHA MuJoCo joint-space replay
        │  – frame alignment (translate-only, our world → Trossen world)
        │  – optional orientation remap (SO(3) calibrated from mean quaternion)
        │  – optional gripper-length compensation (per-frame offset along approach axis)
        │  – per-frame damped-LS IK on the kinematic Jacobian (6 joints/arm)
        │  – per-frame position-only fallback (when 6-DoF is unreachable)
        │  – Gaussian smoothing on joint trajectories
        │  – position controllers in scene_joint.xml
        ▼
trossen_replay_ik_pos.mp4  (sim render: arms executing the trajectory)
trossen_replay_ik_log.npz  (per-frame target vs actual EE positions)
```

## Files

| file | purpose |
|---|---|
| `extract_trajectory.py` | Stage 1 pipeline: WiLoR → Algorithm 1 → world frame |
| `smooth_trajectory.py` | Stage 2a smoothing (plane fit + B-spline + SLERP) |
| `animate_trajectory.py` | matplotlib 3D animation of any trajectory bundle |
| `replay_trossen.py` | v1 sim replay (mocap+weld constraint — kept for reference) |
| `replay_trossen_ik.py` | v2 sim replay (real joint-space IK + position control) |
| `outputs/` | per-bundle results: trajectory + smoothed + sim replay |
| `dev_log.md` | complete implementation log (design, ablations, decisions) |

## Quick-start

Two conda environments because the trajectory extraction needs WiLoR (PyTorch + MANO) and the Trossen sim needs MuJoCo + scipy:

```bash
# Env 1: trajectory extraction
conda create -n wilor python=3.10
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install --no-build-isolation chumpy
pip install --no-build-isolation git+https://github.com/warmshao/WiLoR-mini.git
pip install opencv-python scipy matplotlib

# Env 2: Trossen MuJoCo sim
conda create -n trossen_sim python=3.10
git clone https://github.com/TrossenRobotics/trossen_arm_mujoco.git
cd trossen_arm_mujoco && pip install -e .
pip install imageio imageio-ffmpeg
```

Then for each demonstration clip (RGB-D PNG sequences from two wrist cams + `camera_extrinsics.json`):

```bash
# Stage 1: extract per-arm trajectory
python extract_trajectory.py --clip path/to/clip --out_dir outputs/clip/wrist

# Stage 2a: smooth
python smooth_trajectory.py --in_npz outputs/clip/wrist/trajectory.npz \
                            --out_dir outputs/clip/wrist

# Trossen replay (sim)
python replay_trossen_ik.py --in_npz outputs/clip/wrist/trajectory_smoothed.npz \
                            --record_mp4 outputs/clip/wrist/trossen_replay_ik_pos.mp4 \
                            --position_only --warmup_seconds 2.0
```

## Adaptations from the paper

The paper uses one binocular camera viewing both hands. This rig has two independent wrist-mounted cams, one per arm — extra pipeline steps were needed:

1. **Per-camera single-hand extraction** with WiLoR-mini.
2. **Cross-camera time alignment** on `ts_device_ms` (no hardware sync), 30 Hz uniform grid.
3. **Cross-embodiment frame transform** to the Trossen world frame (translate + optional SO(3) remap, see BiNoMaP §Appendix E for the spirit of the approach).
4. **Custom joint-space IK** in MuJoCo (Trossen MuJoCo's mocap+weld is a soft constraint, not a real IK solver).

See [dev_log.md](dev_log.md) for the full design log, ablations, and decisions.

## Status

- Stage 1 (trajectory extraction): ✅ implemented, mm-level accuracy verified
- Stage 2a (smoothing): ✅ implemented
- Stage 2b (iterative real-robot contact adjustment): ⏸ deferred — needs real Trossen hardware
- Stage 3 (category-level size parameterization): ⏸ deferred

## References

- BiNoMaP paper: https://arxiv.org/abs/2509.21256 — Huayi Zhou, Kui Jia, "BiNoMaP: Bimanual Non-Prehensile Manipulation Primitives"
- WiLoR-mini: https://github.com/warmshao/WiLoR-mini
- Trossen MuJoCo: https://github.com/TrossenRobotics/trossen_arm_mujoco
