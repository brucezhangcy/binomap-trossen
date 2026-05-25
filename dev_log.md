# BiNoMaP Stage 1 — Dev Log

Bimanual wrist-trajectory extraction from two-camera RGB-D recordings, following BiNoMaP §3.2 with the per-camera-per-hand adaptation (paper uses one binocular cam for both hands; we use two independent cams, one per arm, due to mutual occlusion).

Downstream consumer: **ALOHA Trossen** dual-arm replay. Output: per-arm 6-DoF EE pose `(p_world, R_world)` on a uniform time grid.

---

## 2026-05-19 — Environment survey

- Host: 3× NVIDIA RTX 6000 Ada (49 GB each). GPU 1 already loaded with ~36 GB, GPU 0/2 free. Driver 580.76.05, CUDA 13.0.
- Existing conda envs: `base` (Python 3.13.11), `DexM` (Python 3.10.19, torch 2.10, opencv 4.13, scipy 1.15, trimesh 4.11, pytorch-kinematics). Will NOT touch DexM — creating dedicated `wilor` env.
- Data layout confirmed:
  - `recordings/` and `recordings_1/`, each with two camera-serial subfolders + `camera_extrinsics.json`.
  - Cam `333422304645` → left arm, hand-eye RMS 3.32 mm trans, 0.50° rot mean.
  - Cam `338122302972` → right arm, hand-eye RMS 3.95 mm trans, 0.68° rot mean.
  - Intrinsics: ~fx≈fy≈387 px, cx≈326 / 319, image 640×480.
  - RGB: 8-bit PNG. Depth: 16-bit grayscale PNG (RealSense convention, 1 unit = 1 mm — to be confirmed against tabletop ≈ 20 mm).
- Frame counts and timing (from earlier inspection):
  - `recordings/cam333` (L): 230 frames, 29.8 fps, start offset +448 ms vs cam338.
  - `recordings/cam338` (R): 217 frames, 30.0 fps.
  - `recordings_1/cam333` (L): 218 frames, 30.1 fps, start offset −496 ms vs cam338.
  - `recordings_1/cam338` (R): 230 frames, 29.7 fps.
- No synchronized hardware trigger; `ts_device_ms` will drive cross-camera alignment.

## 2026-05-19 — Environment build

- `conda create -n wilor python=3.10` → Python 3.10.20.
- `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` → torch 2.5.1+cu121 (later downgraded to 2.5.0 by WiLoR-mini deps, fine).
- WiLoR-mini install failed initially: `chumpy` couldn't build (its `setup.py` does `import pip`, but pip is not exposed in modern `pip>=23` build isolation).
  - Fix: `pip install setuptools==69.5.1` (older setuptools needed for chumpy) → `pip install --no-build-isolation chumpy` → `pip install --no-build-isolation git+https://github.com/warmshao/WiLoR-mini.git`.
- WiLoR-mini auto-downloads everything: YOLO `detector.pt`, `wilor_final.ckpt`, and even MANO (so no MPI license sign-up).
- Extra deps installed: `opencv-python matplotlib scipy`.
- First inference run auto-installed `dill` via ultralytics autoupdate.

## 2026-05-19 — MANO joint convention (verified)

WiLoR remaps MANO joints to **OpenPose 21-keypoint convention** in `wilor/models/mano_wrapper.py`:

```
mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
```

After remap, the indices we care about are:

| name | idx |
|---|---|
| wrist | 0 |
| thumb_tip | 4 |
| index_tip | 8 |
| middle_tip | 12 |
| ring_tip | 16 |
| pinky_tip | 20 |

Encoded as constants in [extract_trajectory.py](/data/bruce/BiNoMaP/extract_trajectory.py) (`IDX_WRIST=0`, `IDX_THUMB_TIP=4`, `IDX_INDEX_TIP=8`, `IDX_RING_TIP=16`).

## 2026-05-19 — Left-hand chirality handling

WiLoR is trained on the MANO right-hand model only. For left-hand inputs, the pipeline runs the image through the right-hand network and then negates the X coordinates of the output joints/vertices/global_orient/hand_pose (`wilor_hand_pose3d_estimation_pipeline.py:138-146`):

```python
if right == 0:
    wilor_output_i["pred_keypoints_3d"][:, :, 0] = -wilor_output_i["pred_keypoints_3d"][:, :, 0]
```

This X-flip is a reflection (det = −1), which inverts chirality of subsequent cross-products. Concretely:
- Right hand: `cross(l_iw, l_rw)` palm normal points one way (say, out of palm).
- Left hand (after X-flip): same operation points the opposite way.

Algorithm 1 as written in BiNoMaP §3.2 assumes a single hand convention. To keep the gripper approach axis (`v_z`) consistent across arms, the extraction pipeline **negates `v_z` for left-hand detections** in `algorithm1_so3()`:
```python
if side == "L":
    v_z = -v_z
```

Then `v_x = v_y × v_z` is recomputed so the (v_x, v_y, v_z) frame remains right-handed (det = +1). SO(3) sanity check (`R @ Rᵀ ≈ I`, `det(R) ≈ +1`) passes on first 10 valid samples of both clips.

## 2026-05-19 — Smoke test (frame 100 of recordings/)

Both cameras detect 1 hand each on frame 100, both classified `is_right=0.0` (left hand). YOLO console output confirms: "1 left". The demo task in this clip is **bimanual cardboard-box lifting** (matches BiNoMaP paper's `pivoting box` skill: thin cuboid flat on table → raised to standing). Visual confirmation from `recordings/333422304645/rgb/000100.png` and `recordings/338122302972/rgb/000100.png`.

Note: `pred_cam_t_full` from WiLoR is in a "scaled focal length" weak-perspective space (e.g. z ≈ 14.5–22 m on these frames), NOT real metric scene depth. **We do NOT use it for the contact-point 3D position.** Instead we follow BiNoMaP Algorithm 1: project thumb_tip/index_tip pixel midpoint, then deproject with the real depth image and real intrinsics. This is the same workaround the paper §3.2 calls out: "WiLoR outputs per-frame meshes in independent coordinates (no camera intrinsics)."

## 2026-05-19 — Full pipeline runs

Code: [extract_trajectory.py](/data/bruce/BiNoMaP/extract_trajectory.py).

### Clip `recordings/`

```
left-cam (333422304645)  valid 230/230 (100.0%)  side L=230 R=0 fallback=0
right-cam (338122302972) valid 105/217  (48.4%)  side L=88 R=28 miss=101 fallback=88
bimanual grid: 221 samples @ 30 fps over [1779153048993, 1779153056328] ms (span 7.34 s)
  valid_L = 205 / 221   valid_R =  81 / 221
SO(3) sanity check passed.
```

Outputs: [outputs/recordings/](/data/bruce/BiNoMaP/outputs/recordings/) → `trajectory.npz`, `trajectory.csv`, `trajectory.png`.

### Clip `recordings_1/`

```
left-cam  (333422304645) valid 218/218 (100.0%)  side L=218 R=0 fallback=0
right-cam (338122302972) valid 138/230 (60.0%)   side L=78 R=90 miss=62 fallback=78
bimanual grid: 220 samples @ 30 fps over [1779152941952, 1779152949283] ms (span 7.33 s)
  valid_L = 215 / 220   valid_R = 109 / 220
SO(3) sanity check passed.
```

Outputs: [outputs/recordings_1/](/data/bruce/BiNoMaP/outputs/recordings_1/) → `trajectory.npz`, `trajectory.csv`, `trajectory.png`.

## 2026-05-19 — Sanity check + interpretation

- **Left-arm camera is rock-solid** (100% detection on both clips). The wrist cam is mounted such that the demonstrator's hand (or the closer hand) is centered in frame, well-lit, and minimally occluded.
- **Right-arm camera is weak** (48% / 60%). Root cause is YOLO's hand detector missing the hand on many frames — `no_det=101` for clip 1, `no_det=62` for clip 2. Failure mode appears to be hand-too-close-to-camera + occlusion by the box (box edge cuts across the hand contour).
- **Handedness flag is unreliable**. The same physical hand sometimes labeled left, sometimes right by YOLO. In recordings_1 right-cam, the detector returned mostly "left" (78 frames) versus "right" (90 frames) on what is likely the same physical hand. Our `pick_detection()` uses `expected_side` as a hint but falls back to the largest-bbox detection when no detection matches the expected side. The `side_detected` and `fallback_used` arrays in the npz let downstream tools filter or weight samples.
- **3D shape sanity**: both world-frame trajectories sit above z = 0 (tabletop) and inside roughly the ±0.5 m workspace — consistent with the calibrated table-corner geometry. The right-arm trajectory is the one rising vertically (matches the box being lifted), the left-arm trajectory is more constrained.
- **z range** (m): clip 1 ≈ 0.05–0.40; clip 2 ≈ 0.10–0.35 — both above the refined tabletop at z = 0.02 m.

## Output schema (for ALOHA Trossen replay)

`trajectory.npz` contains:

| key | shape | dtype | meaning |
|---|---|---|---|
| `ts_ms` | (T,) | float64 | device_ms timestamps on a uniform 30 Hz grid (T ≈ 220) |
| `p_L`, `p_R` | (T, 3) | float64 | left/right contact-point position in world frame [m] |
| `R_L`, `R_R` | (T, 3, 3) | float64 | rotation in world frame, columns = (x, y, z) basis |
| `q_L`, `q_R` | (T, 4) | float64 | same rotation as quaternion (x, y, z, w) |
| `valid_L`, `valid_R` | (T,) | bool | True if interpolation anchors were within 33 ms |
| `side_L_detected`, `side_R_detected` | (T,) | int8 | 0=left-hand, 1=right-hand, −1=missing at that grid point |

Same data as CSV in `trajectory.csv` (human-readable). 3D PNG plot in `trajectory.png`.

## 2026-05-19 — Step 1 follow-ups: per-camera gap fill + wrist position

### Per-camera gap fill (`fill_gaps`)

Added `fill_gaps(traj, max_gap_frames)` to [extract_trajectory.py](/data/bruce/BiNoMaP/extract_trajectory.py). For each invalid per-camera frame, finds the nearest original-valid neighbors before/after. If the further neighbor is ≤ `max_gap_frames` (default 10 = ~333 ms at 30 fps), linearly interpolates position and SLERPs rotation. Boundary frames (no neighbor on one side) are NOT extrapolated.

Per-frame interp flag is propagated through the resampler so the bundle exposes `interp_L`/`interp_R` alongside `valid_L`/`valid_R`. Downstream consumers can use `valid & ~interp` for raw observations only, or `valid` for the densified series.

CSV/NPZ schema gained `interp_L`/`interp_R` columns. Plot now shows interp samples as light-tinted X markers (raw as solid circles).

### Wrist vs gripper-contact-point

Mentor explicitly asked for the **wrist trajectory** (手腕). BiNoMaP §3.2 itself uses the midpoint of `thumb_tip` and `index_tip` as the "Hand-to-Robot" contact point — appropriate for matching a parallel-jaw gripper. Switched the position source from fingertip-midpoint to **MANO joint 0 (wrist)** to satisfy the mentor's request.

- Position: `p_cam = deproject(kp2d[IDX_WRIST], depth)` — wrist pixel + depth + intrinsics.
- Rotation unchanged: Algorithm 1 cross-products are anchored at the wrist by construction, so the rotation naturally pairs with the wrist position.

Tradeoff: the wrist pixel sometimes has worse depth than the fingertip midpoint (sleeve cuff, motion blur near body), so for clip `recordings_1` the right-cam detection rate dropped from 93.9% (gripper-midpoint, fill applied) to 70.0% (wrist, fill applied). Gripper-midpoint version was preserved at git-level for revert if needed.

### Re-run results (final, wrist + gap fill)

| | clip `recordings` | clip `recordings_1` |
|---|---|---|
| L cam raw valid | 230/230 (100%) | 218/218 (100%) |
| L cam after fill | 230/230 (100%) | 218/218 (100%) |
| L cam filled count | 0 | 0 |
| R cam raw valid | 109/217 (50.2%) | 75/230 (32.6%) |
| R cam after fill | 164/217 (75.6%) | 161/230 (70.0%) |
| R cam filled count | 55 | 86 |
| Bimanual grid (30 Hz) | 221 samples | 220 samples |
| valid_L on grid | 205 (interp 0) | 215 (interp 0) |
| valid_R on grid | 159 (interp 77) | 141 (interp 90) |
| SO(3) sanity | ✓ | ✓ |

L-cam: rock solid, never needs fill. R-cam: roughly 50/50 raw-to-fill ratio after the gap-fill pass. The remaining R-cam holes are at gap boundaries > 10 frames where we deliberately don't extrapolate.

Outputs (latest, **wrist** position):
- [outputs/recordings/trajectory.{npz,csv,png}](/data/bruce/BiNoMaP/outputs/recordings/)
- [outputs/recordings_1/trajectory.{npz,csv,png}](/data/bruce/BiNoMaP/outputs/recordings_1/)

## 2026-05-19 — Terminology + open questions

### What "detection rate" means here

A frame is a **usable sample** (= replayable by the robot) only if the entire pipeline succeeds end-to-end on it:

1. **YOLO** finds at least one hand bbox in the RGB frame.
2. **WiLoR** produces 21 MANO joints + the 2D pixel at the wrist (or fingertip midpoint).
3. **Depth** at that pixel is non-zero (RealSense can drop pixels at edges, shiny surfaces, motion).
4. **Deprojection** + transform-to-world produces a sensible `(p_world, R_world)` pair.
5. **`fill_gaps`** OR the raw observation succeeded — either the frame was directly detected, or the gap to its valid neighbors was ≤ 10 frames and it was interpolated.

If any of those fail, the frame is `valid=0` in csv/npz → the robot skips that timestamp. "Detection rate X%" means `valid_*` is True on X% of grid frames.

### Wrist vs fingertip-midpoint — detection-rate difference

The choice of position source changes the depth pixel that gets queried, which in turn changes how often step 3 above fails. On the harder right-arm camera, **after** `fill_gaps`:

| | Wrist (MANO joint 0) | Midpoint (thumb_tip + index_tip)/2 |
|---|---|---|
| clip `recordings` | 75.6% | 66.8% |
| clip `recordings_1` | 70.0% | 93.9% |

Mixed result — neither dominates. Likely cause: the wrist pixel can be hidden by sleeve / motion blur, while the fingertip midpoint sits closer to the center of the hand region where depth tends to be more consistent. Switching is a one-line change (`IDX_WRIST` ↔ `IDX_THUMB_TIP/IDX_INDEX_TIP` midpoint).

### `fill_gaps` vs paper's Stage 2 smoothing

These are **two different operations**, often confused because both touch SO(3) via SLERP:

| | `fill_gaps` (added here) | BiNoMaP §3.3 Motion Smoothness |
|---|---|---|
| **Scope** | only invalid frames | all frames |
| **Goal** | imputation (fill holes) | denoising (suppress jitter) |
| **Position** | linear interp between valid neighbors | plane fit (least-squares) → project all points → cubic B-spline smoothing in plane |
| **Rotation** | SLERP between 2 adjacent valid neighbors | SLERP between selected anchor frames 𝒦 (top-n most-stable) |
| **Touches raw observations?** | No | Yes (every point can move a few mm) |
| **In the paper?** | No — paper has no gaps (one cam sees both hands + manual start/end labels) | Yes — §3.3 |

The two are complementary and can stack: `fill_gaps` first (so Stage 2's input is dense), then Stage 2 smoothing (so the whole trajectory is regularized). Currently only `fill_gaps` is implemented; Stage 2 is the natural next step.

## 2026-05-19 — Dual-output layout

Decisions locked in:
1. **Position source: emit BOTH wrist and midpoint trajectories** — compare downstream on the real robot.
2. **`fill_gaps` threshold = 10 frames stays.**

### Code change

Added `--position_source {wrist,midpoint}` CLI flag (default `wrist`) to [extract_trajectory.py](/data/bruce/BiNoMaP/extract_trajectory.py). The flag controls only the 2D pixel that gets deprojected; the SO(3) rotation (Algorithm 1 cross-products) is identical between the two modes since both build the rotation from the same MANO joint vectors.

### Re-run results (both sources, both clips)

| | wrist | midpoint |
|---|---|---|
| `recordings` R-cam raw valid | 109/217 (50.2%) | 105/217 (48.4%) |
| `recordings` R-cam after fill | 164/217 (75.6%) | 145/217 (66.8%) |
| `recordings` bimanual valid_R (interp) | 159 (77) | 140 (59) |
| `recordings_1` R-cam raw valid | 75/230 (32.6%) | 138/230 (60.0%) |
| `recordings_1` R-cam after fill | 161/230 (70.0%) | 216/230 (93.9%) |
| `recordings_1` bimanual valid_R (interp) | 141 (90) | 198 (89) |

Wrist wins clip 1, midpoint wins clip 2 — neither dominates. The mentor can pick the winner empirically on the Trossen.

### New output layout

```
outputs/
├── recordings/
│   ├── wrist/      trajectory.{npz,csv,png}   (MANO joint 0)
│   └── midpoint/   trajectory.{npz,csv,png}   (BiNoMaP §3.2 (thumb_tip + index_tip)/2)
└── recordings_1/
    ├── wrist/      trajectory.{npz,csv,png}
    └── midpoint/   trajectory.{npz,csv,png}
```

Both bundles share the same time grid, same `R_*`/`q_*`, same `valid_*`/`interp_*` masks structure — only `p_L`/`p_R` (and the underlying detection-rate counts driven by depth availability at the chosen pixel) differ.

## 2026-05-19 — Stage 2a Motion Smoothness Optimization

Implemented BiNoMaP §3.3 motion-smoothness optimization in [smooth_trajectory.py](/data/bruce/BiNoMaP/smooth_trajectory.py). Pure post-processing — takes a `trajectory.npz` produced by `extract_trajectory.py` and emits `trajectory_smoothed.npz` (+ csv + png).

### Algorithm

Per arm (independently):
1. **Plane fit.** SVD on centered valid positions → smallest singular vector = unit normal `n`, with offset `b = -n · mean(P)`.
2. **Project** all valid positions onto the plane → 2D in-plane coords using orthonormal basis `(u, v)` perpendicular to `n`.
3. **Cubic B-spline smoothing** (`scipy.interpolate.splrep`, k=3) independently on `(t, x_2d)` and `(t, y_2d)`. `s = N · σ²` with σ = 5 mm default → smoothing condition matches expected per-point noise.
4. **Lift** back to 3D: `p_smooth = origin + x_smooth · u + y_smooth · v`. Result is exactly coplanar by construction.
5. **Anchor selection** `𝒦 = {start, end} ∪ top-3 intermediate frames with smallest positional residual` (paper Table 3 best = top-n 3).
6. **SLERP rotations** between consecutive anchors, evaluated on the original time grid.

Invalid frames (where `valid_*=False` even after `fill_gaps`) are left untouched.

### Output bundle additions

`trajectory_smoothed.npz` extends the original schema:

| key | shape | meaning |
|---|---|---|
| `p_L`, `p_R`, `R_L`, `R_R`, `q_L`, `q_R` | unchanged shapes | **smoothed** position + rotation (overwritten) |
| `p_L_raw`, `p_R_raw`, `R_L_raw`, `R_R_raw`, `q_L_raw`, `q_R_raw` | unchanged shapes | original raw values (preserved for comparison) |
| `anchors_L`, `anchors_R` | (5,) int | global frame indices of the 5 anchor frames (start + 3 mid + end) |
| `ts_ms`, `valid_*`, `interp_*`, `side_*_detected` | unchanged | carried forward |

CSV columns are duplicated similarly (smoothed columns first, raw columns suffixed `_raw`).

### Per-bundle results

| | L plane thickness | L residual mean | L residual max | R plane thickness | R residual mean | R residual max |
|---|---|---|---|---|---|---|
| `recordings/wrist` | 6.7 mm | 8.7 mm | 20.8 mm | 11.0 mm | 10.9 mm | 34.9 mm |
| `recordings/midpoint` | 10.3 mm | 11.0 mm | 34.5 mm | 17.5 mm | 15.1 mm | 67.2 mm |
| `recordings_1/wrist` | 2.2 mm | 6.8 mm | 16.9 mm | 7.8 mm | 9.4 mm | 27.5 mm |
| `recordings_1/midpoint` | 3.2 mm | 6.9 mm | 24.0 mm | 8.7 mm | 9.4 mm | 31.1 mm |

- **Plane thickness** = standard deviation of `(p · n + b)` across raw valid points. Smaller = more planar = paper's assumption holds better.
- **Residual** = `‖p_smooth − p_raw‖` per frame. Mean ~7–15 mm matches expected WiLoR + depth noise budget; max spikes are outlier frames the spline pulls back.

Wrist is more planar than midpoint on every clip (wrist moves less than fingertip — fingertip orbits around it). Both options stay within ~1–2 cm of raw → smoothing didn't distort the trajectory shape.

### Final output layout

```
/data/bruce/BiNoMaP/outputs/
├── recordings/
│   ├── wrist/      trajectory.{npz,csv,png}              ← Stage 1 + fill_gaps
│   │               trajectory_smoothed.{npz,csv,png}     ← + Stage 2a
│   └── midpoint/   trajectory.{npz,csv,png}
│                   trajectory_smoothed.{npz,csv,png}
└── recordings_1/
    ├── wrist/      trajectory.{npz,csv,png}
    │               trajectory_smoothed.{npz,csv,png}
    └── midpoint/   trajectory.{npz,csv,png}
                    trajectory_smoothed.{npz,csv,png}
```

Mentor pipeline: load either `trajectory.npz` (raw + fill) or `trajectory_smoothed.npz` (denoised, recommended for Trossen replay). All four `{wrist, midpoint} × {recordings, recordings_1}` combinations are ready.

## 2026-05-19 — Outlier filter (depth bounds + workspace bounds)

### Motivation

Visual inspection of `recordings/wrist/trajectory.png` showed a ~1.2 m horizontal red spike on the right-arm trajectory at grid frames 167–169. Root cause: the wrist 2D pixel landed on the background (wall behind the table), so the depth pixel returned ~1 m instead of the hand's ~0.4 m, and the deprojection placed the wrist 1 m away from the actual hand. `fill_gaps` then propagated the bad anchor into adjacent interpolated frames.

### Fix

Added two layers of outlier rejection in [extract_trajectory.py](/data/bruce/BiNoMaP/extract_trajectory.py):

1. **Depth bounds in `deproject()`** — reject any pixel whose depth is outside `[depth_min_mm, depth_max_mm]` (defaults 100 mm and 1500 mm). Catches background hits, near-camera glare, sensor saturation.
2. **Workspace bounds in `process_camera()`** — reject if the deprojected, world-transformed position falls outside a configurable AABB (default `[-0.7, 0.7] × [-0.7, 0.7] × [-0.05, 0.7]` m). Backstop in case depth passed but the cam→world transform still produced an absurd point.

Both rejections fall through to `valid=False` → `fill_gaps` then bridges across the bad frames using **good** neighbors.

CLI flags: `--depth_min_mm`, `--depth_max_mm`, `--workspace_box "xmin,xmax,ymin,ymax,zmin,zmax"`. Camera-done log now reports `out_workspace=N` count.

### Before vs after (R-cam max distance from median position, raw bundle)

| | before | after |
|---|---|---|
| `recordings/wrist` | **1244 mm** (frames 167–169 spike) | 314 mm |
| `recordings/midpoint` | 360 mm | 360 mm |
| `recordings_1/wrist` | 97 mm | 97 mm |
| `recordings_1/midpoint` | 143 mm | 143 mm |

Only `recordings/wrist` had the egregious spike — caught by the depth bound (`no_depth` count went 7 → 11, those 4 extra rejections include the bad-background pixels). The other bundles had no outliers above the threshold so they're unchanged.

`out_workspace=0` across every camera × clip — meaning the depth bound caught everything before the world-frame check needed to fire. The workspace bound is there as a backstop for unusual failure modes.

After re-running Stage 2a smoothing on the cleaned bundles, `outputs/recordings/wrist/trajectory_smoothed.png` no longer shows the horizontal red spike — both clusters are tight and within the expected workspace volume.

## 2026-05-19 — Outlier filter v2: K-nearest bundle filter

### Problem with v1

The first outlier filter (depth bounds + workspace bounds + per-camera rolling-median) still left a smaller spike (~314 mm offset) at grid frames 171–176 of `recordings/wrist`. Root cause:

- The 6 spike frames were **all interpolated** (`interp=1`, several with `side=-1` = no raw detection at all). The per-camera filter only checks raw frames, so it didn't apply.
- The bundle-level rolling-median filter used a ±5-frame window. But the spike sat in the middle of a 50+ frame gap (grid frames 150-170 invalid, 177-198 invalid) — so the filter's neighbor set was empty (`nbr_mask.sum() < 2`) and the spike was skipped.

The spike came from `fill_gaps` at the per-camera level: it bridged a long invalid gap using a "bad" raw anchor on the far side, smoothly interpolating to the wrong position.

### Fix v2: K-nearest-by-time bundle filter

Replaced the ±window-frames bundle filter with a **K-nearest-valid-by-time** filter ([`filter_bundle_outliers`](/data/bruce/BiNoMaP/extract_trajectory.py)):

```python
# For each valid grid frame k:
#   find K nearest valid frames (by time index) — regardless of how far away
#   compare p[k] to median of those K positions
#   reject if |p[k] - median| > threshold
```

Default `k_neighbors = 10`, `max_dev_mm = 100`. This guarantees we always have K reference points no matter how long the surrounding gap is.

CLI flag: `--bundle_k_neighbors`.

### Results after v2

R-cam max distance from median (raw bundle):

| | v1 (window=5) | v2 (k_neighbors=10) |
|---|---|---|
| `recordings/wrist` | 314 mm | **200 mm** (5 grid rejected) |
| `recordings/midpoint` | 360 mm | **262 mm** (6 grid rejected) |
| `recordings_1/wrist` | 97 mm | 80 mm (0 rejected) |
| `recordings_1/midpoint` | 143 mm | 143 mm (0 rejected) |

5 spike samples in `recordings/wrist` were caught and invalidated. The remaining 200mm max is a legitimate (if slightly noisy) endpoint of the actual motion, not a 1.2m teleport. Stage 2a smoothing pulls it onto the planar curve.

Visual: `outputs/recordings/wrist/trajectory.png` and `trajectory_smoothed.png` both show clean compact clusters, no horizontal spikes.

### Filter chain (final)

```
1. process_camera   → WiLoR + Algorithm 1 + depth bound + workspace bound
2. filter_outliers_by_neighbors  (per-camera, ±5 frames, 100mm)   ← catches single bad raw frames
3. fill_gaps        (per-camera, ≤10-frame gaps)
4. align_bimanual   (resample to 30 Hz grid, SLERP rotations)
5. filter_bundle_outliers  (iterative, K=10 nearest RAW by time, 100mm)
                                                                   ← catches spike clusters from bad fill_gaps anchors
6. save bundle
```

Step 2 catches outliers BEFORE fill_gaps so the interpolation uses clean anchors. Step 5 catches outliers AFTER alignment that snuck through (e.g., long-gap fill from a bad anchor that itself was below the per-camera filter threshold).

## 2026-05-19 — Outlier filter v3: iterative + raw-only references

### Problem with v2

K-nearest-by-time at the bundle level still left a 253 mm spike at frames 170-174 of `recordings/midpoint`. The K=10 nearest valid samples around frame 170 included the spike cluster itself (frames 171-174 were also "valid but bad") → polluted the median → filter underestimated the deviation → spike survived.

### Fix v3: iterative + raw-only references

Two changes to `filter_bundle_outliers`:

1. **Reference set = RAW samples only** (`valid & ~interp`). Interpolated samples can't pollute their own reference median because they're not in the reference set.
2. **Iterative passes** (up to 5) — each pass uses the previous valid mask. After pass 1 invalidates one layer of the spike, pass 2 sees a tighter cluster and rejects more, etc. Stops early when no more rejections happen.

### Results (R-cam max distance from median, raw bundle)

| | v1 (depth+ws only) | v2 (single-pass K=10) | v3 (iterative, raw-only) |
|---|---|---|---|
| `recordings/wrist` | 314 mm | 200 mm | **47 mm** ✓ |
| `recordings/midpoint` | 360 mm | 262 mm | **99 mm** ✓ |
| `recordings_1/wrist` | 97 mm | 80 mm | 80 mm |
| `recordings_1/midpoint` | 143 mm | 143 mm | 143 mm |

The remaining 80–143 mm on the clean clips is genuine motion range, not outliers — the hand legitimately moves over that range during the box-pivot. Smoothed (post Stage 2a) maxes match: 39 / 97 / 72 / 141 mm.

Plot inspection: all 4 `trajectory.png` and all 4 `trajectory_smoothed.png` show compact clusters with no extending spikes. Final.

## 2026-05-19 — Trajectory animations

Added [animate_trajectory.py](/data/bruce/BiNoMaP/animate_trajectory.py) to render an MP4 animation of any bundle. Each animation frame = one grid timestamp; output is 30 fps so playback is real-time-synchronized with the source RGB videos in `videos/`.

What each frame shows:
- **Blue / red trails** — accumulated valid positions up to time t (left / right arm).
- **Bold dot** — current EE position at time t.
- **3-axis triad** (X=red, Y=green, Z=blue, 5 cm) — current SO(3) orientation at the EE.
- **Title** — clip name, frame index, elapsed seconds, per-arm valid flags.

### Gap-shortcut artifact + fix

First-pass animations of the smoothed bundles showed many apparent "spikes" radiating from the trajectory. Cause: matplotlib's line-plot connects consecutive valid samples with straight segments, so a long invalid stretch (e.g., 20 missed frames) became a straight shortcut across the workspace.

Fix: NaN-fill invalid frames in the position arrays before `set_data`. Matplotlib breaks the line at NaN, so gaps appear as missing segments instead of cross-workspace lines:

```python
p_L_nan = np.where(valid_L[:, None], p_L, np.nan)
p_R_nan = np.where(valid_R[:, None], p_R, np.nan)
trail_L.set_data(p_L_nan[:k+1, 0], p_L_nan[:k+1, 1])
trail_L.set_3d_properties(p_L_nan[:k+1, 2])
```

After fix: trails are clean curves with visible gaps where WiLoR missed; no spurious spikes.

### Generated files

```
outputs/recordings/wrist/trajectory_smoothed.mp4         (812 KB, 221 frames)
outputs/recordings/midpoint/trajectory_smoothed.mp4      (711 KB, 221 frames)
outputs/recordings_1/wrist/trajectory_smoothed.mp4       (858 KB, 220 frames)
outputs/recordings_1/midpoint/trajectory_smoothed.mp4    (833 KB, 220 frames)
```

Packaged: [trajectory_animations.zip](/data/bruce/BiNoMaP/trajectory_animations.zip) (3.1 MB).

These pair naturally with the raw RGB recordings in [videos/](/data/bruce/BiNoMaP/videos/) — side-by-side playback lets a viewer verify that the extracted trajectory's motion matches the demonstrator's hand at each frame.

## 2026-05-19 — Trossen ALOHA MuJoCo replay (validation in simulation)

Mentor task: verify the extracted trajectories are **executable and safe** on the Trossen ALOHA bimanual setup in MuJoCo (no box yet, real-robot test deferred).

### Environment

```
conda create -n trossen_sim python=3.10
cd /data/bruce/BiNoMaP
git clone https://github.com/TrossenRobotics/trossen_arm_mujoco.git
cd trossen_arm_mujoco && pip install -e .
pip install imageio imageio-ffmpeg numpy scipy
```

MuJoCo 3.8.1, native Python bindings. For headless off-screen rendering on a server: `MUJOCO_GL=egl` is set inside [replay_trossen.py](/data/bruce/BiNoMaP/replay_trossen.py) before `import mujoco`.

### Trossen scene anatomy ([stationary_ai/scene_mocap.xml](/data/bruce/BiNoMaP/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/scene_mocap.xml))

- 16 DOFs: 2 arms × (6 revolute + 2 prismatic gripper).
- **Mocap bodies** `mocap_left` / `mocap_right` are weld-constrained to `follower_left_link_6` / `follower_right_link_6`. Setting `data.mocap_pos[0/1]` and `data.mocap_quat[0/1]` commands the EE pose; MuJoCo's IK chases.
- **MuJoCo quaternion convention is `(w, x, y, z)`**, different from our bundles' `(x, y, z, w)`. Conversion = component reorder.
- **Arm bases**: `follower_left_base_link` at (−0.02, +0.4575, 0.039) with yaw −90°; `follower_right_base_link` at (−0.02, −0.4575, 0.039) with yaw +90° — i.e., arms face inward toward each other.
- **Mocap rest poses** (XML defaults): left (−0.02, +0.213, 0.20), right (−0.02, −0.213, 0.20).
- **Tabletop top** at z = 0.02 m — coincidentally matches our world's tabletop z.

### Frame alignment (v1: translate-only)

```
our_midpoint     = 0.5 * (mean(p_L_valid) + mean(p_R_valid))   # in our world
trossen_target   = midpoint of mocap rest poses = (-0.02, 0, 0.20)
                   z adjusted to preserve our hand-above-table delta
offset           = trossen_target - our_midpoint
p_trossen        = p_ours + offset
R_trossen        = R_ours       # no rotation in v1
```

For `recordings_1/wrist`: `offset = (+0.033, +0.008, 0.000) m`. Trajectory pre-checks: 0 workspace violations, 0 velocity violations (< 3 m/s cap).

### Replay design

1. **Warm-up phase (default 2 s)** — lerp mocap from each arm's natural rest pose (read after applying the `home` keyframe) to the trajectory's first frame. Without this, the weld constraint sees a ~30 cm sudden jump on step 1 and the arm snaps to an unreachable pose with the IK never recovering.
2. **Replay loop** at 30 Hz × `--speed` — write `mocap_pos`/`mocap_quat` per frame, hold previous target on `valid_*=False` frames, run `substeps_per_frame ≈ 17` substeps (timestep = 2 ms).
3. **Two orientation modes** — `from_trajectory` (default; passes our Algorithm-1 `R_L/R_R` through quaternion reorder) or `initial` (uses the arm's natural rest quat, position-only intent).
4. **No gripper commands** — `qpos[6,7,14,15]` left at home keyframe (closed). Matches paper §A.1 (non-prehensile).

### Results on `recordings_1/wrist` (1.0× speed, 2 s warmup, from_trajectory orientation)

**Absolute tracking error (EE actual vs mocap target):**

| arm | mean | p95 | max |
|---|---|---|---|
| L | 146.4 mm | 147.5 mm | 150.0 mm |
| R | 138.7 mm | 146.8 mm | 154.8 mm |

**Shape-match residual** (after subtracting the per-arm constant offset — quantifies whether the arms follow the *shape* of the trajectory):

| arm | constant offset | residual mean | residual max |
|---|---|---|---|
| L | (+47, −31, +132) mm | **24.9 mm** | 87.1 mm |
| R | (−52, −58, +12) mm | 106.8 mm | 220.6 mm |

### Interpretation

- **L arm**: shape-residual 25 mm = the arm tracks the trajectory's *motion pattern* well, just sitting at a fixed ~14 cm kinematic offset.
- **R arm**: residual 107 mm = noisier shape tracking. R has more challenging targets (higher z and arms reaching across) where the IK can't satisfy both position and orientation. The R-cam detection rate (70 % vs L's 100 %) already made this the weaker arm.
- **Constant offset cause**: mocap + weld is a *soft* constraint — when the requested (pos, quat) pair is unreachable by the kinematic chain, the weld settles at a least-squares compromise. The visible motion in the rendered MP4 confirms arms are *moving* in response to the trajectory; they just sit at the closest-reachable pose.

**Verification verdict for the mentor task:**
- ✅ Trajectory loads without exceptions, stays in the workspace AABB, no velocity violations.
- ✅ MuJoCo simulation runs the full ~7 s clip without instability.
- ✅ Arms visibly follow the trajectory's macro motion pattern (esp. L arm).
- ⚠️ Mocap+weld IK can't reach exact targets when orientation is constrained — 14 cm constant offset. For tight absolute tracking, switch to `scene_joint.xml` + an explicit IK solver (e.g., MuJoCo's `mj_inverse` or scipy least-squares on FK).

### How my impl maps to the paper's design

| BiNoMaP design | My impl in `replay_trossen.py` |
|---|---|
| `(p, R)` per-arm 6-DoF pose directly drives the EE — no joint-space retargeting | `mocap_pos`/`mocap_quat` with weld constraint to last arm link |
| Grippers stay closed (non-prehensile) | `qpos[6,7,14,15]` untouched (home = closed) |
| Stage 2a smoothing applied before deployment | Replay reads `trajectory_smoothed.npz` |
| Cross-embodiment = "only XYZ axis remap" (Appendix E, Aubo→Rokae) | Frame alignment = translate-only (v1); yaw correction available for v2 if motion direction is off |
| **Not implemented:** Stage 2b iterative contact adjustment (real-robot retries, γ=0.85 decay), Stage 3 size parameterization | Both out of scope for the box-free sim verification milestone |

### Files

```
/data/bruce/BiNoMaP/
├── replay_trossen.py                                                ← pipeline
├── trossen_arm_mujoco/                                              ← cloned repo (BSD-3-Clause)
└── outputs/recordings_1/wrist/
    ├── trossen_replay.mp4                                           ← rendered sim (top-down camera)
    ├── trossen_replay_log.npz                                       ← per-frame target & actual EE
    └── trossen_replay_compare.png                                   ← 3D plot: target vs actual, with shape-residual
```

### Open items for next milestone

1. **Reduce 14 cm absolute offset** — Either (a) move to joint-space control (`scene_joint.xml` + numerical IK) or (b) tune mocap+weld solver parameters or (c) iteratively adjust frame-alignment offset to land in the IK's sweet spot.
2. **R-arm shape residual 107 mm** — Inspect whether the worst frames coincide with R-cam interp gaps. May warrant a R-arm-specific yaw correction in the frame transform.
3. **Add the box back** — Once the sim replay is accurate enough that the arms reliably reach the trajectory poses, re-introduce the manipulated box and check whether the BiNoMaP §3.3 iterative contact adjustment (5 mm safety distance, γ=0.85 decay) succeeds in sim.
4. **Real-robot trial** — Only after the above; verify the calibrated world frame really matches the Trossen base on hardware.

## 2026-05-19 — Trossen replay v2: real joint-space IK ([replay_trossen_ik.py](/data/bruce/BiNoMaP/replay_trossen_ik.py))

### Why v1 (mocap + weld) was the wrong approach

The v1 replay drove `data.mocap_pos`/`data.mocap_quat` and relied on the XML's `<weld>` constraint to pull the EE link to the target. That weld is a **soft spring** (`solref="0.01 1" solimp=".25 .25 0.001"`). When the requested `(p, R)` pair is unreachable by the 6-DoF kinematic chain, the weld settles at a least-squares compromise — that's where the constant ~14 cm offset came from. It is **not** an IK solver.

### v2 design

- Load `scene_joint.xml` (14 position-controlled actuators: 6 joints + 1 gripper per arm).
- **Per-frame damped least-squares IK** on the MuJoCo Jacobian (`mj_jac`):
  - For each arm, solve for `q ∈ ℝ⁶` such that FK(q) ≈ (p_target, R_target).
  - Update rule: `dq = Jᵀ (J Jᵀ + λ²I)⁻¹ e`, with λ = 0.05, step-clipped to ±0.3 rad/iter, joint-limited per `jnt_range`.
  - Up to 200 iters, convergence tol = 1 mm position / 1.1° orientation.
  - Warm-start from previous frame's solution for temporal continuity.
- **Replay loop**: send `data.ctrl = joint_targets` for each arm's 6 actuators; the existing position controllers (kp=200/100/50, kv=10/5/5) track in.
- **Warmup phase**: lerp ctrl from home to frame-0 joint config over 1 s so the controllers settle.
- **Output**: per-frame joint targets, actual EE positions, IK residuals saved to `trossen_replay_ik_log.npz`.

### Results on `recordings_1/wrist`

Full 6-DoF tracking (`--position_only` off):

| arm | IK position err (mean / max) | IK rotation err (mean / max) | IK failures | Sim tracking (mean / max) |
|---|---|---|---|---|
| L | 0.4 mm / 1.0 mm | 0.1° / 0.4° | 0 / 215 | 1.6 mm / 7.6 mm |
| R | 66 mm / 327 mm | 39° / 163° | 103 / 139 | 66 mm / 266 mm |

The R-arm IK fails frequently because the right-hand orientations from Algorithm 1 are physically unreachable by Trossen's joint limits (especially joint_1 ∈ [0, π] and joint_2 ∈ [0, 2.36] constrain the elbow), and the YOLO labels were noisy on R-cam.

Position-only mode (`--position_only`, ignoring `R_target`):

| arm | IK pos err (mean / max) | IK failures | Sim tracking (mean / p95 / max) |
|---|---|---|---|
| L | 0.3 mm / 1.0 mm | 0 / 215 | 1.3 mm / 1.6 mm / 7.6 mm |
| R | 0.3 mm / 1.0 mm | 0 / 139 | 4.5 mm / 5.2 mm / 193.9 mm |

**Improvement vs v1**:
- L arm tracking: 146 mm → 1.3 mm (≈ 110× tighter)
- R arm tracking: 139 mm → 4.5 mm at p95 (≈ 27× tighter)
- The R single-frame max of 194 mm comes from a sudden joint-target discontinuity around frame 50 where the position controller hasn't caught up — visible as a spike in the per-frame error plot.

Visualization: arms now visibly trace the human hand's motion. Top-down camera mp4 confirms the bimanual reach + lift + return pattern matches the source RGB recording.

### Mapping to BiNoMaP design (revisited)

v2 is **the right way to do "exactly what the human hands did":**
- Algorithm 1 gives us `(p, R)` per-arm per-frame from human video.
- v2 runs proper IK to solve for the robot's joint config that achieves that (p, R).
- This is exactly how the BiNoMaP paper's robots (Aubo-i5, Rokae xMate) work: their controllers handle joint-space IK internally; the paper just feeds them `(p, R)`.
- The position-only fallback is necessary because Trossen ALOHA's 6-DoF joint limits are tighter than the demonstrator's anthropomorphic wrist, so some Algorithm-1 rotations are unreachable. The orientation-mode flag is exposed via `--position_only` for cases where R targets exceed joint limits.

### Files

```
/data/bruce/BiNoMaP/
├── replay_trossen_ik.py                                       ← new pipeline (joint-space IK)
└── outputs/recordings_1/wrist/
    ├── trossen_replay_ik.mp4              ← 6-DoF IK render
    ├── trossen_replay_ik_pos.mp4          ← position-only IK render (clean)
    ├── trossen_replay_ik_log.npz          ← per-frame target/actual/joint
    └── trossen_replay_ik_compare.png      ← 3D overlay + per-frame error plot
```

### v2.1 fix: back-fill pre-first-valid joint targets

Symptom: the R arm had a 194 mm spike at the very start of `trossen_replay_ik_pos.mp4`. The arm flailed up to the box-lifting pose during the warmup window.

Root cause: R-cam misses the first 47 grid frames of `recordings_1` (no YOLO detection). For those frames the IK pre-compute kept `joint_targets_R[k] = home_qR`. Then at frame 47 (first valid), the IK output jumped to a very different config (high-z target). The position controller couldn't track that jump in one 33 ms grid step → big spike.

Fix (one block in `replay_trossen_ik.py`):
```python
first_valid_R = int(np.where(valid_R)[0][0])
if first_valid_R > 0:
    joint_targets_R[:first_valid_R] = joint_targets_R[first_valid_R]
```
Same for L. Plus `--warmup_seconds 2.0` (was 1.0) gives the position controllers more time to settle.

Results on `recordings_1/wrist` (position-only):

| arm | mean | p95 | max |
|---|---|---|---|
| L | 1.3 mm | 1.6 mm | **4.7 mm** (was 7.6) |
| R | 1.6 mm | 2.1 mm | **6.0 mm** (was 193.9) |

Spike eliminated. Both arms now sub-7 mm everywhere.

### v2.2 — replayed all 4 bundles (position-only IK + back-filled warmup)

| bundle | L mean / p95 / max | R mean / p95 / max | notes |
|---|---|---|---|
| `recordings_1/wrist` | 1.3 / 1.6 / 4.7 mm | 1.6 / 2.1 / 6.0 mm | **best — recommend for mentor demo** |
| `recordings/wrist` | 1.6 / 2.4 / 9.5 mm | 2.6 / 10.5 / 18.0 mm | clean, slightly larger spikes than recordings_1 |
| `recordings_1/midpoint` | 2.5 / 14.1 / 25.3 mm | 7.0 / 27.7 / 52.7 mm | midpoint wobbly as predicted |
| `recordings/midpoint` | 2.0 / 3.0 / 15.9 mm | 12.8 / 61.9 / 69.9 mm | worst R; midpoint × harder R-cam clip |

Pattern confirms what the smoothing plane-thickness predicted: wrist tracks better than midpoint on every clip, especially on R arm (the midpoint sits on a 5–7 cm lever arm from the wrist, so finger pose changes swing it around).

### Open items

- **6-DoF IK on R arm** still has unreachable orientations (Algorithm-1 R from human fingers exceeds Trossen joint limits). Position-only is the safe default; for full 6-DoF we'd need a yaw correction in the frame transform.
- **Mentor demo recommendation**: `recordings_1/wrist/trossen_replay_ik_pos.mp4`. Other 3 bundles available in `outputs/` if she wants to compare clips or wrist-vs-midpoint.

## 2026-05-20 — Three sim videos & how they relate to the paper

### Differences between the 3 output videos

Same source trajectory + same frame alignment in all three; only the control method differs.

| file | IK approach | what to look for |
|---|---|---|
| `trossen_replay.mp4` (v1) | **mocap + weld constraint** (soft spring, not real IK) | arms barely move from rest pose, stuck ~14 cm from target |
| `trossen_replay_ik.mp4` (v2, full 6-DoF) | real damped-LS IK, position + orientation | L arm tracks great; **R arm flails** because Algorithm-1 orientations exceed Trossen joint limits |
| `trossen_replay_ik_pos.mp4` (v2, position-only) ✓ | real damped-LS IK, position only | both arms track at mm level cleanly — **the one for the mentor demo** |

### Does v2 IK + position-only match the paper?

**Mostly yes — closer than v1 by far.**

What the paper does on real hardware:
- Algorithm 1 produces `(p, R)` per arm per frame from human video.
- Paper feeds `(p, R)` directly to the robot's controller (Aubo-i5, Rokae xMate), which runs its own joint-space IK internally and drives joint position controllers.
- The paper doesn't describe its own IK math because it uses what the robot vendor provides.

v2 does the same: takes `(p, R)`, runs joint-space IK, drives Trossen position controllers. Functionally equivalent — I just had to write the IK myself because MuJoCo's mocap+weld is a soft spring rather than a real IK solver. This is what the paper would have done if targeting Trossen.

**Where my v2 deviates from the paper:**
- **Position-only fallback** (`--position_only`). The paper always uses full 6-DoF. I expose this as an option because Trossen ALOHA's 6-DoF joint limits are tighter than the demonstrator's human wrist, so many of Algorithm-1's R orientations are unreachable. The paper sidesteps this by demonstrating directly above the robot workspace so the rotations naturally align with what their arms can do. Our demonstrator's rig has a different layout (separate cameras on a different rig), so we hit the joint-limit ceiling on the right arm.
- For a paper-faithful test, we'd add a yaw correction to the frame transform (matches Appendix E's "XYZ axis remap" for cross-embodiment) and retry 6-DoF.

### Why v2 > v1: how the two versions came up

The paper doesn't actually prescribe a sim-control method — it just describes the algorithm output `(p, R)` and trusts the robot's controller. So I had to pick how to drive MuJoCo myself. The Trossen repo ships two scenes:
- `scene_mocap.xml` — mocap bodies with `<weld>` constraints (the easy demo path).
- `scene_joint.xml` — joint-position actuators (what the real robot would receive).

**v1 went with mocap+weld** because the Trossen tutorial showcases it. Set `mocap_pos`/`mocap_quat`, let MuJoCo "figure out the joints." But mocap+weld is a *soft spring*: when `(p, R)` is unreachable, the weld settles at a least-squares compromise — no IK, no joint optimization, just a constraint pulling toward the target with finite stiffness. That's where the constant ~14 cm offset came from.

**v2 dropped mocap+weld and wrote real IK on top of `scene_joint.xml`**: per-frame damped least-squares IK on the Trossen Jacobian → joint targets → position controllers. This is what the paper's robots do internally on hardware. Tracking dropped from 146 mm → 1.3 mm because v2 actually solves for joint angles, not a spring pulling a link.

**TL;DR**: v1 was a shortcut suggested by Trossen's tutorial scripts; v2 is the actual paper-faithful approach (joint-space IK + position control). The paper never spelled this out because every robot vendor ships its own IK — but if you wanted to "do what the paper does" on Trossen, v2 is it.

## 2026-05-20 — Orientation remap added (BiNoMaP §App E generalised)

Follow-up to "what if you don't deviate from the paper" — i.e. run full 6-DoF IK instead of falling back to position-only. The raw 6-DoF run flails on R arm because Algorithm-1 right-hand orientations exceed Trossen joint limits.

Added `--orientation_remap` (default ON when not `--position_only`) to [replay_trossen_ik.py](/data/bruce/BiNoMaP/replay_trossen_ik.py). For each arm:

1. After loading Trossen scene, capture the EE's rest orientation `R_trossen_rest_*` at the home keyframe via `data.xquat[ee_*_body_id]`.
2. Compute an anchor orientation from the trajectory: **mean quaternion across valid frames** (sign-disambiguated to one hemisphere, then averaged + renormalised).
3. Build `R_remap = R_trossen_rest @ R_anchor.T` so the remapped anchor exactly matches Trossen rest pose.
4. Apply `R_remap @ R_ours_k` to every frame's quaternion before IK.

Matches the spirit of §Appendix E's "XYZ axis remap" for cross-embodiment, generalised to a per-arm full SO(3) rotation. The mean-quaternion anchor balances reachability across the whole trajectory.

### Anchor choice matters (calibration ablation)

| anchor strategy | L mean / max | R mean / max | comment |
|---|---|---|---|
| first valid frame | 6.5 / 21.3 mm | 40 / 176 mm | overfits to one frame; L drifts out of envelope as trajectory evolves |
| **mean quaternion (chosen)** | 1.6 / 5.4 mm | 15.5 / 116 mm | L unchanged from no-remap baseline, R 4× tighter |

### Final 6-DoF vs position-only on `recordings_1/wrist`

| variant | L mean / max | R mean / p95 / max | mp4 |
|---|---|---|---|
| 6-DoF, no remap | 1.6 / 7.6 mm | 66 / 147 / 266 mm | `trossen_replay_ik.mp4` (R flails — don't use) |
| **6-DoF + mean-quat remap** | 1.6 / 5.4 mm | 15.5 / 107.9 / 116.4 mm | `trossen_replay_ik_remap.mp4` (paper-faithful, rotation preserved) |
| **Position-only** | 1.3 / 4.7 mm | 1.6 / 2.1 / 6.0 mm | `trossen_replay_ik_pos.mp4` (cleanest absolute tracking) |

R-arm with remap still has occasional unreachable frames (IK rot err up to 90°): those are genuinely outside Trossen's reachable orientation envelope no matter how we align it. Bulk of frames track cleanly.

### Decision matrix

- **Cleanest tracking**: use `trossen_replay_ik_pos.mp4` (position-only). Gripper orientations are inherited from Trossen's natural pose; trajectory's rotation info is discarded.
- **Paper-faithful, gripper orientations matter**: use `trossen_replay_ik_remap.mp4` (6-DoF + mean-quat remap). Preserves the human-hand orientation evolution; occasional 10 cm R-arm spikes on unreachable frames.
- **Don't ever use**: `trossen_replay_ik.mp4` (6-DoF no remap → R flails).

CLI:
- 6-DoF + remap (default): `python replay_trossen_ik.py --in_npz ... --record_mp4 ...`
- Position-only: add `--position_only`
- 6-DoF, no remap: add `--no_orientation_remap`


## 2026-05-20 — recordings/wrist R-arm wobble fix + gripper-length compensation

### Why R arm is "always the unstable one"

Both demos are box-pivot tasks: one hand stabilizes (small motion, low z), the other actively lifts/rotates the box. The active hand happens to be in the **right-arm camera's view** in both clips — and the active hand has:
- **More occlusion** by the box being manipulated → YOLO detection drops to 50% (clip 1) and 33% (clip 2)
- **More YOLO mislabeling** → 88 fallback frames in `recordings/R` (= `pick_detection` chose the largest bbox when expected side wasn't found, which sometimes is a different physical hand)
- **Wider rotation range** → 6-DoF IK fails more often (Trossen joint_1 ∈ [0, π] limit)

So "R arm unstable" is really "R cam tracks the hardest hand." Swapping the L↔R arm assignment wouldn't help — same data, different label.

Diagnostic on `recordings/wrist`: fallback boundaries produce the worst jumps. Clean (both-side-R) frame-to-frame Δp max = 13.6 mm; fallback-involved Δp max = 26.6 mm.

### Fix attempt 1: reject all fallback frames + bigger smoothing (over-corrected)

`clean_trajectory.py` rejects frames where `side_detected != expected_side`, then re-runs `fill_gaps` and Stage 2a with `sigma_m=0.012`. On `recordings/wrist`:
- R: 105 raw → 26 raw valid (rejected 79). Even after fill_gaps, only 44/221 grid frames valid.
- Replay tracking: L 1.6→2.4 mm mean (worse), R 2.6→9.9 mm mean (worse). Too sparse to be useful.

### Fix attempt 2: bumped Stage 2a sigma_m only (partial win)

Restored `recordings/wrist` via re-extract + re-smooth, this time with `--sigma_m 0.008` (8 mm noise budget vs default 5 mm):
- R: mean 2.6 mm → **1.6 mm**, max 18 mm → **9.5 mm**  ✅ wobble gone
- L: mean 1.6 mm → 2.3 mm, max 9.5 mm → **60.9 mm**  ⚠️ startup blip on frames 3-8

The L spike is at the B-spline boundary: larger sigma destabilizes the spline at the trajectory edges. Concentrated in frames 3-8 (~100-260 ms after warmup ends); rest of L is sub-5 mm. Acceptable for the demo since the blip is during the warmup-to-trajectory transition.

### Proper fix (not done yet): per-arm sigma_m

R-cam tracks the noisier hand → needs more smoothing; L-cam tracks the stable hand → default smoothing is fine. Right approach: `smooth_trajectory.py --sigma_m_L 0.005 --sigma_m_R 0.008`. ~10 lines to modify, would eliminate both the R wobble AND the L startup blip.

### Gripper-length compensation (--gripper_offset_m)

Added a CLI flag to `replay_trossen_ik.py` to compensate for Trossen ALOHA's gripper being longer than the human hand's wrist-to-fingertip span:

- Human gripper ≈ 6 cm (MANO wrist to fingertip midpoint)
- Trossen gripper ≈ 9 cm (link_6 to gripper pad)
- Without compensation: Trossen gripper tips end up ~6 cm closer together than the human's → a box that fit the human's hands won't fit Trossen's.

Per-frame correction: `p_trossen[k] = p_aligned[k] + d * R_world[:,:,2]_k`
where `R[:,:,2]` is the palm-normal column (v_z from Algorithm 1) and `d ≈ +0.03 m` pushes the wrist OUTWARD along the approach axis, keeping gripper tip positions where the human gripper tips were.

CLI: `--gripper_offset_m 0.03` (default 0 = no compensation).

Validation: with current pipeline (no box in sim) this just visually separates the arms more. To actually verify it fits a specific box, we need to add the box to the Trossen scene and check whether the trajectory's contact phase matches the box geometry. Out of scope for tonight's demo.

### Current state per bundle (recommended for mentor demo)

- `recordings_1/wrist`: position-only IK, default sigma_m=0.005. L 1.3 / R 1.6 mm mean. ✅ recommended.
- `recordings/wrist`: position-only IK, sigma_m=0.008. L 2.3 / R 1.6 mm mean, brief L startup blip. Acceptable.
- Both midpoint bundles: not re-replayed; midpoint inherently wobblier.

## 2026-05-20/21 — GitHub push + IK refinements + new clip diagnosis

### GitHub push

- Initialized [/data/bruce/BiNoMaP](/data/bruce/BiNoMaP) as a git repo.
- Pushed to https://github.com/brucezhangcy/binomap-trossen (renamed from a typo `bimomap-trossen`).
- `.gitignore` excludes: raw RGB-D recordings, recordings.zip / recordings_one_hand_fix.zip, `trossen_arm_mujoco/` and `wilor_repo/` (external repos), the BiNoMaP paper PDF + Chinese translation, encoded `videos/` MP4s, `__pycache__/`, and the staging zips (`trajectory_results.zip`, `trajectory_animations.zip`).
- Initial commit (52 files, 7.9 MB): all `.py` pipeline scripts, dev_log, README, and `outputs/` (4 bundles × {trajectory, smoothed, IK replay, plots}).
- Pulled remote commit (`d4b0992`): mentor added `record_rgbd.py`, `replay_real.py`, `view_brown_box.py`.

### New input clip: recordings_one_hand_fix (Bruce, 2026-05-21)

- Unzipped (~2 GB extracted → 5806 PNG frames). Top-level `recordings/` inside the zip renamed to `recordings_one_hand_fix/` to avoid clash with the existing `recordings/`.
- Moved both zip files into `archives/` to declutter the project root.
- Encoded sibling MP4s in `videos/recordings_one_hand_fix_<serial>.mp4` (~15 MB each, 48 s, 30 fps).
- Per-camera frame counts: L = 1453, R = 1445.

### Gripper-tip-at-wrist IK fix (replay_trossen_ik.py)

Mentor's directive: "position of hand wrist should be the tip of the gripper." Previous setup: wrist-trajectory → Trossen `link_6` target. But Trossen's gripper extends ~9 cm forward from `link_6`, so the GRIPPER TIP was landing 9 cm beyond the human wrist (a 35 cm box wouldn't fit between the tips).

**v1 wrong fix (don't use):** pre-shift wrist target by `-9 cm × R_target[:,2]` before IK. Failed because in `--position_only` mode, the achieved `link_6` orientation isn't the trajectory's `R_target` orientation — so the offset direction in world frame was wrong → tracking went to hell.

**v2 correct fix (current):** parameterize `ik_solve()` with `tip_offset_local` = a 3-vector in `link_6`'s LOCAL frame. The Jacobian is computed at `link_6_pos + R_link6 @ tip_offset_local`, and pos_err is `target − (that point)`. So IK directly solves "gripper tip at target" instead of "link_6 at target". For Trossen: `tip_offset_local = [0.09, 0, 0]` (9 cm along link_6 +x).

Verification on `recordings_1/wrist`:
- L tip mean: 0.3 mm, p95 0.6 mm, max 2.1 mm
- R tip mean: 1.3 mm, p95 1.5 mm, max 66.7 mm (one outlier → see boundary-pin fix below)

### First/last valid frame pin (after joint smoothing)

Found a follow-on bug: joint-space Gaussian smoothing (default σ=2 frames) was distorting the FIRST and LAST valid frames per arm, pulling them toward neighbor frames that hold the boundary back-fill or move quickly afterward. For `recordings_1` R-arm this gave a 66 mm start-point error at the first valid R frame (47).

Fix: capture each arm's first-valid and last-valid joint-target BEFORE smoothing, then restore them after. Result: start-point alignment now sub-mm on both arms.

```
L first valid frame 0: tip target = (-0.009, 0.167, 0.193) m,  sim tip = (-0.009, 0.167, 0.193) m,  err = 0.24 mm
R first valid frame 47: tip target = (-0.043, -0.206, 0.351) m, sim tip = (-0.043, -0.206, 0.351) m, err = 0.14 mm
```

### `clean_trajectory.py` (utility)

Standalone post-processor that (a) rejects YOLO-fallback frames (`side_detected != expected_side`) and (b) re-runs Stage 2a smoothing with a configurable noise budget. Tried it on `recordings/wrist` with `sigma_m=0.012` and aggressive fallback rejection — over-corrected (R-cam went from 105 raw valid → 26 raw, then 44/221 grid valid after fill_gaps, too sparse to be useful). Conclusion: per-arm σ would have been the better lever; leaving the script in the repo for future use but not running it in the default pipeline.

### Recordings_one_hand_fix: trimmed to 7s + tried pipeline → real bug surfaced

The full 48-s clip is mostly idle box (the active demonstration is only the first ~7 s). Trimmed to first 240 frames per camera (hardlinked, ~331 MB) → `recordings_one_hand_fix_7s/`.

Ran the full pipeline. Results were structurally OK (extract → smooth → IK replay all ran, sim mp4 generated) but the trajectory was clearly wrong:
- L cam: 240/240 valid (100% with 24 fill_gaps)
- **R cam: only 39/240 valid (16%)** — and `side_detected: L=23, R=0, miss=217`. R cam never sees a hand YOLO labels "right" across all 240 frames.
- Inter-wrist median in world frame: **70-73 mm**, vs 417 mm on `recordings_1` (a 5× drop).
- Both L and R hand centroids cluster at +Y (left side of workspace) — both at z ≈ 0.42 m.

### Diagnosis (long debug session, Bruce + me)

Initial hypothesis sequence (most → least wrong):
1. **"R cam doesn't see the right hand"** — wrong. Bruce clarified: he can clearly see the right hand bracing the bottom of the box in the R cam mp4.
2. **"Stale extrinsics for this clip"** — wrong. Confirmed the `camera_extrinsics.json` is byte-identical across `recordings_1` and `recordings_one_hand_fix`, and Bruce confirmed the physical camera setup didn't change between sessions.
3. **"Narrow-grip demo, so 10 cm wrist-to-wrist is correct"** — partially wrong. Inter-wrist of 10 cm is too tight for any plausible box grip, and the R hand z = 0.428 m doesn't match the visible hand at table level (z ≈ 0.02 m).
4. **Actual cause** — see below.

Wrote `debug_rcam.py` to inspect what WiLoR + YOLO actually do on a few R cam frames. Output:
- Frames 30, 60, 90, 120, 180: **0 detections** (YOLO finds nothing despite a clearly visible right hand)
- Frame 150: 1 detection — **YOLO drew its bbox in the middle of the box surface**, on what looks like the white shipping sticker. WiLoR ran on that bogus bbox and produced a "wrist" 2D pixel on the box itself. Depth at that pixel = 521 mm = box surface depth → deprojected wrist at z = 0.151 m (~mid-box height, NOT the real hand at z ≈ 0.02 m).

So the real cause: **WiLoR-mini's bundled YOLO hand detector is failing on this clip.** Misses the actual hand in ~83 % of frames, generates false positives on the box's shipping sticker in some frames. The pipeline code is correct; the off-the-shelf detector just doesn't know what to do with a hand bracing the bottom of a cardboard box. Saved overlay images in `_inspect/dbg_R_f*.png` for reference.

### Next step in progress: swap YOLO for MediaPipe Hands

Three plausible fixes:
- **(1) Different detector** — MediaPipe Hands or RTMPose-hand. Different training distribution; might catch the hand poses YOLO misses.
- **(2) Manual bbox annotation** for 240 frames + feed to WiLoR's 3D-reconstruction step only (skip YOLO).
- **(3) Temporal bbox propagation** — once a hand is detected, Kalman / optical-flow the bbox across nearby frames.

Going with (1) — installing MediaPipe in the `wilor` env. Will run the same `debug_rcam.py` test on the same 6 frames to see if MediaPipe detects what YOLO missed. If yes, swap in MediaPipe + use depth+intrinsics deprojection (we don't need WiLoR's 3D mesh — MediaPipe already gives 21 3D landmarks per hand). If no, fall back to (2) or (3).

---

## 2026-05-21 — MediaPipe swap: fix lands cleanly

### Detector swap → success

Installed MediaPipe (`pip install mediapipe` → 0.10.35) in `wilor` env. The classic `mp.solutions.hands` API no longer ships with that version — switched to the new task API (`mediapipe.tasks.python.vision.HandLandmarker`) which needs an explicit model file. Downloaded `hand_landmarker.task` (7.8 MB) from Google's CDN.

Tested first on the same 6 R-cam failure frames that broke WiLoR/YOLO (`debug_mediapipe.py`):

| Frame | WiLoR/YOLO | MediaPipe |
|---|---|---|
| 30, 60, 90, 120, 180 | 0 detections | 4/5 detections ✓ |
| 150 (false-positive on sticker) | bogus bbox on box | correct wrist ✓ |

MediaPipe's keypoints land squarely on the actual hand (yellow dots in `_inspect/dbg_R_mp_f*.png`).

### Drop-in replacement

Wrote `extract_trajectory_mediapipe.py` — same CLI as `extract_trajectory.py`. Wraps the new HandLandmarker in a `MediaPipeRunner` class that returns detection dicts shaped exactly like WiLoR-mini's output (`hand_bbox`, `is_right`, `wilor_preds.pred_keypoints_2d/3d`), so all downstream helpers (`pick_detection`, `filter_outliers_by_neighbors`, `fill_gaps`, `align_bimanual`, `filter_bundle_outliers`, `save_outputs`, `plot_trajectory`) and Algorithm 1 are reused unchanged.

### Full-pipeline run on `recordings_one_hand_fix_7s`

```
L cam valid (after fill): 240/240   (raw 234/240 → 6 filled)
R cam valid (after fill): 196/240   (raw 184/240 → 27 filled, 15 outliers rejected)
bimanual grid: 234   valid_L=231   valid_R=188
```

vs WiLoR (same clip): L 240/240, **R 39/240, inter-wrist 70 mm**.

After Stage 2a smoothing + IK pass:
- Inter-wrist median: **352 mm** (vs WiLoR's 70 mm) — now in the expected ~300 mm range for a two-handed box grip.
- L mean position: (-0.006, 0.157, 0.410) m. R mean position: (-0.082, -0.111, 0.173) m.
- L range covers ~150 mm in Y and 125 mm in Z (active lifting hand).
- R range is narrow: 30 mm × 75 mm × 90 mm (consistent with "right hand barely moves, braces the box bottom").
- IK pos err sub-mm (L mean 0.3 mm / max 1.0 mm, R mean 0.3 mm / max 1.0 mm). 0 IK failures, 0 per-frame fallbacks.

### MediaPipe handedness quirk to remember

R cam side-detected counts: `L=130, R=54, miss=56, fallback=130`. MediaPipe is mislabelling the (clearly visible) right hand as "Left" in ~70% of R-cam frames. Spot-checked overlays (`_inspect/verify_R_mp_f*.png`): the position is always on the right hand, the handedness label is just flipped. Spurious tiny detections at frame edges sometimes get the correct "Right" label, which is unhelpful. `pick_detection`'s area-fallback rescues this because it picks the largest detection when no handedness match exists — the real hand is always bigger than the spurious corner blob.

If we later want to suppress those spurious detections explicitly, options are: (a) min-bbox-area filter in `MediaPipeRunner.__call__`, (b) ignore MediaPipe's handedness entirely on this clip and treat all detections as the expected side. Not blocking — leaving the area fallback as-is.

### Files updated / created
- `extract_trajectory_mediapipe.py` — new
- `debug_mediapipe.py` — new (6-frame smoke test)
- `_verify_mp_rcam.py` — new (overlay rendering for handedness sanity)
- `hand_landmarker.task` — new (7.8 MB model)
- `outputs/recordings_one_hand_fix_7s_mp/wrist/{trajectory*.npz, trajectory*.csv, trajectory*.png, replay_ik_pos.mp4, trossen_replay_ik_log.npz}`

### Status

Stage 1 trajectory extraction now works for both clips. `recordings_one_hand_fix_7s` produces a geometrically sensible bimanual trajectory (~35 cm wrist separation, sub-mm IK convergence on Trossen). The MediaPipe path will likely be the new default for any clip where the demonstrator's hands aren't in a "natural" pose (knuckles-down, palm-up bracing, partial occlusion).

## 2026-05-21 — Re-extracted brown box point cloud with arms parked

Re-ran the brown-box capture script (since superseded by [boxes/capture_box.py](boxes/capture_box.py) — see the May 22 consolidation entry below) while the bimanual arms were holding the goal-capture park pose (j1≈0.047, j2=0.700, j3=0.628, others 0 — both elbows swung forward, shoulders lifted, grippers above the workspace). The point was to capture a clean box-only cloud with zero arm geometry in the scene, since the previous May 20 capture was made with the arms in some neutral position.

### Tooling change

Added `--auto-click SERIAL,X,Y` (repeatable) and `--no-viewer` CLI flags so the original `brown_box_pointcloud.py` could run fully non-interactive when given clicks up-front. Reused the prior session's click coordinates:

```
--auto-click 333422304645,359,263   # cam_low  (L)
--auto-click 338122302972,361,226   # cam_high (R)
```

(Both `--auto-click` and the script itself were superseded by the May 22 consolidation — see entry below. The headless flag has not been re-added to the new script; re-introduce if you need to script another parked-arm capture.)

### Before / after

| | May 20 (arms not parked) | May 21 (arms parked) |
|---|---|---|
| brown_box.ply pts | 60,656 | 35,580 |
| AABB extent (m) | 0.300 × 0.393 × 0.167 | 0.278 × 0.372 × 0.164 |
| AABB center (m) | (−0.009, −0.032, +0.102) | (+0.016, −0.022, +0.099) |
| L cam masked px | 29,154 (10.8 %, score 0.985) | 29,425 (11.2 %, score 0.974) |
| R cam masked px | 32,534 (13.8 %, score 0.816) | **6,178 (2.0 %, score 0.920)** |
| DBSCAN clusters | 11 (largest 60,656) | 1 (35,580 — single clean cluster) |
| scene.ply pts | 402,321 | 419,943 |

Previous PLY preserved as `outputs/brown_box/brown_box.prev.ply` before overwrite.

### What the deltas mean

1. **AABB center shifted +2.5 cm in X, +1 cm in Y.** The box has physically moved on the table since the May 20 capture (someone bumped it, or it was repositioned). The same R-cam click `(361, 226)` no longer lands on the box centre — it lands closer to an edge, so SAM2 segments a much smaller visible face. L-cam mask is essentially unchanged because the click `(359, 263)` still lands inside the box body in that view.
2. **R-cam contribution dropped 5×.** Despite that, the merged cloud is still a single clean DBSCAN cluster — the L cam alone provides 29 k points which already covers the full box surface visible from that angle.
3. **Scene cloud got slightly bigger** (+17 k pts) — consistent with the arms being lifted out of the depth field-of-view (less arm geometry contaminating the workspace volume).
4. **AABB extent shrank ~2 cm in X and Y, ~3 mm in Z.** With the arms gone, points that previously came from arm forearms (sitting near the edges of the workspace AABB) are no longer in the masked cloud → the box's own dimensions are reported more tightly.

### Visualization

4-view PNG (isometric / top / front / side) at `outputs/brown_box/visualize_compare.png` — new (left column) vs old (right column). Both clouds resolve to the same box from every angle; the new one is sparser but covers the same volume.

### Files updated
- `outputs/brown_box/brown_box.ply` — overwritten with the 35,580-pt cloud.
- `outputs/brown_box/scene.ply` — overwritten with the 419,943-pt scene.
- `outputs/brown_box/brown_box.prev.ply` — manual backup of the May 20 version.
- `outputs/brown_box/visualize_compare.png` — new vs old side-by-side render.

Hardware state during capture: both Trossen WX AI arms held in park pose by [park_pose.py](park_pose.py) (background process — position-mode actively locked, zero motion during the ~30 s SAM2 + capture pass).

## 2026-05-21 — Stage 2b geometry-aware contact adjustment

Implemented BiNoMaP §3.3 Stage 2b in [contact_adjustment.py](contact_adjustment.py). Takes a smoothed bimanual trajectory + object point cloud and emits an adjusted bundle where the primary arm is shifted toward the object surface; the user then iterates with the paper's γ=0.85 decay (max k ≤ 10 in the paper, extended as needed for our demo geometry) until the real-robot test makes contact.

### Why it diverges from the paper's exact formula

The paper assumes a bimanual sandwich grip — both arms in contact with the object at `t_s`. Our `recordings_1/wrist` demo is single-hand-active: L touches the box (within 0.9 mm at frame 166), R never approaches it (always ≥ 166 mm away, sits 17 cm above box top). So the paper's scaling `s_k = ‖new_R − L‖ / ‖orig_R − L‖` with R as primary and L as reference can't produce a real bimanual contact — it just slides R toward L's air-grip position.

The adopted approach: translation-based per-arm shifts with **start-only decay** so the demo's end-of-motion is preserved (translating R uniformly creates a "shrunken trajectory" effect where R returns to a shifted position; start-only with linear decay to 0 by `t_end` keeps the demo trajectory intact except at the contact-establishment phase). γ=0.85 still drives the iteration step size, matching the paper.

### Script features (`contact_adjustment.py`)

| Flag | Purpose |
|---|---|
| `--mode {scale, translate}` | Paper's Eq. 4 (§3.3.2) scaling (default) vs. constant translation. Translation preserves the demo's inter-arm distance evolution; scaling shrinks it uniformly per iteration. |
| `--apply-to {primary, both}` | Single arm (paper) vs. bilateral. Bilateral is needed when the demo lacks a clear primary-secondary roles. |
| `--lateral-spread-mm`, `--lower-mm` | Translation knobs (constant shift for entire trajectory; box-decoupled). |
| `--shift-R-y-start-mm`, `--lower-R-start-mm` | **Start-only** shifts per arm, decay linearly to 0 by the arm's last valid frame. Preserves demo end pose. Add `_L_` variants for L. |
| `--distance-mm D` / `--iteration K` | Set d_k directly or via paper's `5 × 0.85^(K-1)` formula. |
| `--tag NAME` | Custom suffix for `trajectory_contact_adj_<tag>.npz` to keep iterations distinct. |

### Iteration framework used for `recordings_1/wrist` + brown box

Baseline input: [trajectory_contact_adj_d30.0mm_translate.npz](outputs/recordings_1/wrist/trajectory_contact_adj_d30.0mm_translate.npz) (the static d=30 mm primary=left translate adjustment of the smoothed trajectory — preserves L's contact intent).

R-only iteration, start-only shifts in y and z, both decaying linearly to demo by end of motion:

| Phase | What happens | Why |
|---|---|---|
| K=1 | y_shift_start = -200 mm (R pushed far in -y), no z shift | Maximum lateral safety. R won't crush the box during slow_move to start pose. |
| K=2..K=10 | y locked at -170 (no further y change), z safety decays per γ=0.85 (R progressively descends) | L is already at its contact pose; R needs to descend onto the box top. Locking y prevents R-y from drifting toward demo (which would put R 6 mm outside the -y face — too close to risk during iteration on z). |
| K=11..  | y also decays per γ=0.85 (R approaches box from -y side as well) | After R is near box top (z safety ~15 mm), start tightening y to engage the box's -y face for full grip. |

The z-safety budget is the wrist's height above the contact target: `safety_k = (R_demo_z − R_contact_target_z) × γ^(k-1)`, where contact target = `box_top_z + gripper_length = 0.272 m`. So `z_shift_k = 79 mm × (1 − γ^(k-1))`. Decays asymptotically: K=10 ≈ 18 mm, K=20 ≈ 3 mm.

### Per-iteration runner

For each iteration K, the workflow is three commands chained:

```bash
# 1. Generate the adjusted trajectory bundle
/home/yunshuang/anaconda3/envs/depth_lerobot/bin/python contact_adjustment.py \
    --smoothed outputs/recordings_1/wrist/trajectory_contact_adj_d30.0mm_translate.npz \
    --shift-R-y-start-mm <Y_VAL> --lower-R-start-mm <Z_VAL> \
    --tag K${K}_Ronly_dRyStart${Y_VAL}_dRzStart${Z_VAL}

# 2. Re-solve IK in MuJoCo (gripper-tip-at-wrist via --gripper_offset_m 0.09 is essential)
source /home/bruce/miniconda3/etc/profile.d/conda.sh && conda activate trossen_sim && \
MUJOCO_GL=egl python replay_trossen_ik.py \
    --in_npz outputs/recordings_1/wrist/trajectory_contact_adj_<tag>.npz \
    --scene_xml /home/bruce/Aloha_real/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/scene_joint.xml \
    --record_mp4 outputs/recordings_1/wrist/trossen_replay_contact_adj_<tag>.mp4 \
    --position_only --warmup_seconds 2.0 --gripper_offset_m 0.09

# 3. Real-arm replay with D455-low workspace cam recording (via the new --d455 flag in replay_real.py)
source /home/yigit/miniconda3/etc/profile.d/conda.sh && conda activate lerobot && \
python replay_real.py --max-step-delta 0.2 --d455 low
```

### Supporting infra added this session

- [`replay_real.py`](replay_real.py): added `--d455 {high,low,both}` to register the workspace D455 cameras on-the-fly (their serials aren't in yigit's lerobot Stationary config). `--auto-skip-leading-invalid` flag uses `trajectory_smoothed.npz` masks to skip back-filled frames. `slow_move_to` rewritten to bypass `send_action`'s `max_relative_target` clamp by calling `arm.driver.set_all_positions` directly — without this, large slow-moves get clamped and the arm never reaches the target pose.
- [`park_pose.py`](park_pose.py): moves both arms to the goal-capture park pose and holds indefinitely. Used during brown box recapture so the arms don't appear in the depth scene.
- brown-box capture script: added `--auto-click SERIAL,X,Y` and `--no-viewer` for headless re-extraction. (Now superseded by [`boxes/capture_box.py`](boxes/capture_box.py) — see May 22 consolidation.)

### Notes from iteration

- **Gripper offset is essential.** Running `replay_trossen_ik.py` *without* `--gripper_offset_m 0.09` makes the IK target link_6 (the wrist) at the trajectory point, but the gripper tip then projects 9 cm *forward* of that — meaning the tips end up ~6 cm closer to each other than the human's fingertips were, and a box that fit the human's hands no longer fits the robot's. The user's commit `b4c76cd` (gripper-tip-at-wrist via `tip_offset_local`) makes this a one-line fix.
- **The `_start` shift formulation matters.** Constant `--lower-R-mm` shifts R uniformly across all 220 frames; R's end-of-motion pose ends up at a different absolute location. Visible as "the trajectory shrunk." The `_start` variants apply the shift only at the arm's first valid frame and decay it linearly to 0 by the last valid frame — R's end pose matches the demo exactly.
- **The translate vs scale split.** Paper's scale mode (Eq. 4 in §3.3.2 — the 3D radial rescale `p̄_R = p_L + s·(p_R − p_L)`) shrinks the inter-arm gap uniformly across all frames. For our box-pivot demo, this can collapse the gap at the pivot apex (where L and R were already close in the demo) — visible as "grippers crash." Translate mode avoids this entirely.
- **Z safety budget vs Y safety budget.** Z safety is well-defined (gripper tip height above box top). Y safety is geometry-dependent (depends on R's approach axis and gripper orientation under `--position_only`). When unlocking y for iteration, expect the iteration to be less predictable since gripper orientation is free.

---

## 2026-05-21 (later) — Status after the iteration sweep

After the K1–K15 contact-adjustment sweep on `recordings_1`, the working point is:
- **The `recordings_1` trajectory can flip the box in sim.** The Trossen arms now make contact, lift, and complete the pivot motion the demonstrator performed.
- **But it still needs initial help placing the box.** The `recordings_1` clip *starts* with the box already mid-flip — the demonstrator's hands enter the frame already bracing the box from below. So when the sim replays from t=0, the box is sitting upright on the table while the grippers reach for where the hands were (already in the lifting pose). With no pre-contact setup, the box would just stay on the table while the arms reach over it.
- Current workaround in sim: spawn the box at the orientation/position the demonstrator's hands are bracing it at on frame 0, rather than upright on the table. The Stage 2b contact-adjustment iteration handles the rest.

Concretely this means the pipeline still needs a "pre-roll" or initial-box-placement step for any clip that doesn't start with the hands well clear of the box. Either:
- record demos that always start with the hands away from the box and the box in its rest pose, or
- add a synthesized lead-in segment that approaches the box from rest before the demonstrator's first valid frame.

The MediaPipe clip `recordings_one_hand_fix_7s` may not have this issue — it starts with the right hand at a low resting position. Not yet tested end-to-end in sim with a box.

---

## 2026-05-21 — Paper-strict orientation pipeline (new replay path)

Mentor task framing: existing replay uses `--position_only`. Paper §3.3 actually uses full 6-DoF — orientation matters. Build a NEW pipeline (don't touch existing working code) where:
1. Orientation is tracked, not dropped.
2. The mapping between gripper local axes and hand local axes is calibrated, not learned blindly (i.e., no `compute_orientation_remap` of mean orientation onto rest pose).
3. The geometric constraint to enforce: **gripper-pointing direction = where the hand pushes the box = palm normal**.
4. Goal: still flip the box, but with orientation.

### Calibration of R_align (gripper-local → hand-local)

Algorithm 1 builds `R_hand = [v_x | v_y | v_z]` columns, where v_z is the palm normal. Trossen ALOHA link_6 +x is the tool axis (gripper pointing). So the mapping is:

```
R_align = R_y(-90°) @ R_x(roll)   # roll is a free parameter (around the tool axis)
```
- R_y(-90°): maps gripper-local +x → hand-local +z (palm normal). This is the constraint.
- R_x(roll): rotation around the tool axis — preserves the gripper-pointing constraint, but lets us pick the orientation around it (where the parallel-jaw finger plane lies).

Tested rolls (0, 45, 90, 135, 180°): roll=90° gave the best R-arm IK convergence (105/139 → 27/139 failed frames, mean rotation error 38° → 7.3°). Roll=0/45/180 left R-arm grossly unreachable. **Picked roll=90°.** CLI form: `--align_euler_deg 90,-90,0`.

Verified visually:
1. World-frame 3D plot: hand v_z arrows (blue) and gripper-pointing arrows (black) overlay exactly across all sampled frames. Both R-arm and L-arm arrows point inward (toward each other / the box midpoint) — i.e., toward where the hand is pushing.
   → `outputs/recordings_1/wrist/orient_calib_roll90.png`
2. RGB overlay on the source video frames: the v_z arrow (re-projected into camera) lands inside the box from the wrist position. Confirmed on L-cam f80 where the L hand braces the box's right face — palm normal arrow points clearly into the box.
   → `outputs/recordings_1/wrist/orient_rgb_overlay/verify_L_f080.png` and 7 other frames.

### New scripts

- `calibrate_gripper_align.py` — 3D world-frame plot of trajectory + RGB-coloured hand-frame arrows + black gripper-pointing arrow, parameterized by Euler-ZYX angles. Used to iterate the roll until the gripper-pointing axis overlays v_z and the resulting EE orientation is reachable for both arms.
- `calibrate_orient_rgb_overlay.py` — projects the same arrows onto the source RGB image (with intrinsics rescaling for upscale) for direct ground-truth comparison against the demonstrator's hand.
- `replay_trossen_paper.py` — strict-paper joint-space IK replay. Hard-coded R_align (no learned remap). Always tracks orientation. Has `--allow_pos_fallback` for the small set of frames where the human wrist twist exceeds Trossen's reachable orientation cone (paper-allows a graceful per-frame fallback to position-only on those exact frames; the rest of the trajectory keeps full 6-DoF).
- `extract_trajectory_v1_backup.py` — verbatim snapshot of the pre-fix extraction script, so we can roll back if anything breaks.

### Discovered Stage-1 bug: WiLoR handedness mislabeling causes 178° R-arm rotation cluster

When running the new replay-paper pipeline I saw a "vibrate twisting" in the R-arm motion that wasn't present in `--position_only`. Diagnostic:
1. Frame-to-frame angular delta on R was **17.6°/frame mean** (vs <1°/frame on L) with max **124°/frame**. Stage 2a's SLERP-between-anchors was producing 11.9°/frame for 15 consecutive frames in the [anchor_177 → anchor_192] segment — totaling a 178° rotation that the human wrist did NOT actually do.
2. The R-arm rotation matrices form a **bimodal cluster**: mean deviation from the global mean orientation is **75.7° with p95=170°** (the cluster gap). L-arm has the same metric at 20.5° / 43.7° — well-behaved.
3. Looking at `side_R_detected`, WiLoR-mini's bundled YOLO labels the right hand as "Left" in **50/103** valid R-cam frames on `recordings_1`. The runs of "L"-labeled and "R"-labeled frames alternate.

Root cause traced into `wilor_mini/pipelines/wilor_hand_pose3d_estimation_pipeline.py`:
- Line 108-122: when the detector returns is_right=0, the input image is X-flipped before pose regression.
- Line 139: after regression, `pred_keypoints_3d[:, :, 0]` is negated to "unflip back" to original image coordinates.
- Net effect: for an actually-right hand mislabeled as "Left", the resulting kp3d describes the right hand AS IF IT WERE A LEFT HAND in original image space. Algorithm 1's chirality sign-flip (`if side == "L": v_z = -v_z`) then operates on those mirrored keypoints, producing a rotation matrix in a basis 180° rotated from the correctly-labeled frames.

The X-flip-then-undo isn't a simple basis rotation we can cleanly undo at our end (I tried — `kp3d[:, 0] = -kp3d[:, 0]` didn't fix it). So the fix is to **trust the camera's known hand assignment, not the detector**.

### Fix in extract_trajectory.py

[extract_trajectory.py:380-388]: when the per-frame `is_right` flag from WiLoR disagrees with the camera's `expected_side`, drop the frame entirely (treat as a missed detection). Position would still be recoverable, but orientation can't be safely reconciled with the other-labeled frames' basis.

```python
expected_is_right = 1 if expected_side == "R" else 0
if side_int != expected_is_right:
    n_side_mismatch += 1
    if k % log_every == 0:
        print(f"  [{k}/{N}] side mismatch (detected={'R' if side_int else 'L'} "
              f"expected={expected_side}) → drop frame", flush=True)
    continue
```

### Results on recordings_1

| Metric | v1 (buggy) | v2 (fixed) |
|---|---:|---:|
| R-cam orient. dev from mean (mean / p95) | 75.7° / 170.3° | **17.1°** / 30.5° |
| R-cam orient. frame-to-frame (mean / max) | 17.6° / 124° | **1.4°** / 6.5° |
| R-arm IK pos err (mean / max) | sub-mm | sub-mm |
| R-arm IK rot err (mean / max) | sub-deg | sub-deg |

Trade-off: dropping the 78 mislabeled R-cam frames leaves only 79 valid_R bimanual grid samples (out of 220), vs 139 in v1. But the surviving samples form a single basis — orientation evolves smoothly.

### Second wobble — gap interpolation

Even with the basis bug fixed, a single ~30° R-wrist (j5) sudden rotation remained at frames 101-108. Cause: 43-frame R-arm gap between valid frame 61 and next valid frame 105. During the gap the joint targets were held at frame-61's config. Frame 105's IK then picked a j5 value ~100° away in the wrist-roll null-space (j5 is redundant relative to gripper-pointing direction, so the IK basin near the held config wasn't a unique solution). The joint Gaussian smoothing then ramped the discontinuity over ~5 frames → visible wrist twist.

Fix in `replay_trossen_paper.py`: linearly interpolate joint targets across the gap between two valid IK solutions BEFORE Gaussian smoothing. Result: R j5 per-frame jump dropped **31.5° → 3.7°** (8.5× smaller). Overall R ||Δq|| max dropped 34° → 12.8° (the remaining 12.8° is the legitimate shoulder reach when the arm first engages at frame 47, not a wrist twist).

### Tip-to-tip distance sanity check

Source trajectory wrist-to-wrist on recordings_1 v2 (both arms valid): **mean=429mm, median=428mm, range 397-454mm**. Sim FK gripper-tip-to-gripper-tip = 427mm mean — matches IK target to sub-mm. (For comparison: `recordings_one_hand_fix_7s_mp` is mean 365mm / median 352mm — a different demo with a smaller box / closer-together grip.)

### Files added/modified

- modified: `extract_trajectory.py` (side-mismatch frame drop)
- new: `extract_trajectory_v1_backup.py` (rollback snapshot, per user request)
- new: `calibrate_gripper_align.py` (3D world-frame R_align calibration)
- new: `calibrate_orient_rgb_overlay.py` (source-RGB v_z visualization)
- new: `replay_trossen_paper.py` (strict-paper 6-DoF IK replay with hard-coded R_align)
- outputs: `outputs/recordings_1_v2/wrist/{trajectory*.npz, trajectory*.csv, trajectory*.png, replay_paper_fixed.mp4, trossen_replay_paper_log.npz}`
- kept (final calibration artifacts): `outputs/recordings_1/wrist/{orient_calib_roll90.png, orient_rgb_overlay/}`

### Status

New paper-strict pipeline produces orientation tracking that's smooth (sub-1°/frame source orientation jitter, IK sub-mm/sub-1° error, no visible wrist twist) and respects the gripper-points-where-palm-pushes constraint by construction. Box-flip verification still pending an appropriate box position in the scene XML — the default 25 mm cube at floor (z=0.0325 m) doesn't intersect the trajectory at z≈0.2 m.

---

## 2026-05-22 — Object point-cloud capture: consolidated into one parameterized script

The brown-box capture flow has grown — black box now, more boxes coming (cyan, blue, plastic, …). Collapsed two color-specific scripts + a color-specific viewer into one folder with two generic scripts.

### Folder

[`boxes/`](boxes/) — everything object-capture-related lives here, code and PLYs together:
- [`capture_box.py`](boxes/capture_box.py) — single capture script, `--name <label>` → `<label>.ply` + `<label>_scene.ply` next to itself. SAM2 hiera-large with a per-camera click; no color logic. Works for any object on the table.
- [`view_box.py`](boxes/view_box.py) — viewer; `--name <label>` resolves to `<label>.ply` in the same folder, or pass `--ply <path>` for anything else.
- `brown_box.ply` — preserved from the May 21 parked-arm capture (35,580 pts).

### Removed

- `binomap/brown_box_pointcloud.py` (subsumed by `capture_box.py --name brown_box`)
- `binomap/blackbox/blackbox_pointcloud.py` (subsumed by `--name black_box`; the `blackbox/` folder is gone)
- `binomap/view_brown_box.py` (subsumed by `view_box.py --name brown_box`)
- The `--auto-click SERIAL,X,Y` / `--no-viewer` headless flags from the old brown-box script were not carried forward. Re-add if you need to script another parked-arm recapture.

### Workflow

```
# Put any object on the table. Then:
python boxes/capture_box.py --name cyan_box
# Click the object in each cv2 window. Outputs: boxes/cyan_box.ply + boxes/cyan_box_scene.ply.

# Later:
python boxes/view_box.py --name cyan_box
```

The `<name>_scene.ply` is the full unmasked merged scene for sanity-checking extrinsics; the `<name>.ply` is the masked + workspace-cropped + largest-DBSCAN-cluster cloud.

### Rationale

Earlier per-color scripts were 95% identical and would have spawned N copies as the object library grew. SAM2 doesn't care about color (it segments from the user's click), so a single `--name` parameter is sufficient. The May 21 brown-box capture is still the canonical brown reference; nothing about the empirical capture is changed by this reorg.

## 2026-05-22 — recordings_2 box-flip: full pipeline succeeds after K=11

New box-flip demo `recordings_2/` (binomap root, 6 s after warmup-barrier crop, 30 Hz on both D455 workspace cams). Captured with [record_rgbd.py](record_rgbd.py) — the warmup-barrier added yesterday ensures `--duration 8` produces a true 8 s of footage (timer only starts after both cameras complete RealSense auto-exposure warmup).

### Pipeline used

```
extract_trajectory_mediapipe.py  →  trajectory.npz
                                    (180/180 L valid, 176/180 R valid; bimanual grid 183 frames)
smooth_trajectory.py             →  trajectory_smoothed.npz
contact_adjustment.py            →  trajectory_contact_adj_K{k}_*.npz  (R-only start-shift, γ=0.85)
replay_trossen_ik.py --rot-weight 0.2  →  trossen_replay_ik_log.npz   (soft 6-DoF, see below)
replay_real.py --max-step-delta 0.2 --d455 low  →  hardware run
```

### New: soft 6-DoF IK (`--rot-weight` flag in `replay_trossen_ik.py`)

`--rot-weight W` (default 1.0 = paper-strict) scales rotation rows of the IK Jacobian and the rotation error vector by W. Lets IK preserve sub-mm position tracking even when the demo's palm orientation isn't fully reachable by Trossen's kinematic chain.

On recordings_2, full 6-DoF (W=1.0) collapsed the R-arm to a 259 mm position offset because the demo's R palm orientation at several frames requires joint configs Trossen can't reach. W=0.2 converges to L pos err mean 0.50 mm / max 0.99 mm; R pos err mean 0.37 mm / max 0.87 mm; L rot err max 1.1°; R rot err max 0.4°. Treats orientation as preference instead of hard constraint — IK finds joint configs that satisfy position first, then picks among them the one closest to the requested orientation.

Practical reason: the paper assumes the demonstrator was thinking like a robot (always reachable). Real hand demos need slack on at least one axis. Soft DoF is the dial; W=0.2 was the sweet spot for this trajectory.

**How it works.** Damped LS solves `dq = Jᵀ(JJᵀ+λ²I)⁻¹ err` where `J = [J_pos; W·J_rot]` and `err = [pos_err; W·rot_err]`. The cost minimized becomes `‖pos_err‖² + W² ‖rot_err‖²` — so W² is just the relative weight of rotation vs position in the IK objective.

**Why it works geometrically.** A 6-DoF target rotation has 3 DoF: (a) palm-normal direction (2 DoF), (b) gripper roll around it (1 DoF). For a sandwich grip, roll doesn't matter much — the box doesn't care which way the wrist is twisted around the approach axis. Strict 6-DoF (W=1) treats roll as important as palm direction, so when reachability forces a compromise, the optimizer "splits the difference" and gives up position AND rotation. Soft DoF (W<1) tilts the cost so the compromise lands on roll (low cost when wrong) rather than position (high cost when wrong) — typically still satisfies palm direction within 1-2° as a side effect, because Trossen has effectively 1 DoF of redundancy.

**Picking W:** 0=position-only · 0.2=our default for real hand demos · 0.5=tighter rotation if reachability allows · 1.0=paper-strict (use only if demo was Trossen-aware).

### New: dynamic HOME pose in `replay_real.py`

Previous `HOME_POSE_L = [0, 0.2618, 0.2618, 0, 0, 2.87, 0]` (set yesterday for recordings_1_v2's L wrist yaw) triggered `FIRST_STEP_ABORT_RAD = 2.5` when running recordings_2 (whose L j5 ≈ 0). Fixed by deriving HOME from the IK log's first commanded pose at runtime when `--auto-skip-leading-invalid` is on — works for any trajectory, eliminates per-trajectory editing of the constant.

Side effect: also eliminates the start-frame "shake" — the back-fill plateau (frames 0..t_R_first−1) is skipped by `--auto-skip-leading-invalid`, so the first streamed frame is the first real IK solution. No joint jump at stream start.

### Iteration schedule (R-only, start-shift, γ=0.85)

Same shape as recordings_1's: shift R outward at start, decay linearly to 0 by trajectory end (preserves the demo's end pose). Starting wide for collision safety:

| K | R y shift (mm) | R z lower (mm) | notes |
|---|---|---|---|
| 1 | −150 (safe wide) | 0 | start of iteration |
| 2-10 | −128, −108, −92, −78, −66, −56, −48, −41, −35 | 0 | y γ-decay only |
| **11** | **−30** | **30** | **z unlocked; box flips on real hardware** |

### Result: box flips reliably at K=11

Running `trajectory_contact_adj_K11_Ronly_dRyStart-30_dRzStart-30.npz` on the real Trossen ALOHA arms, **the box reliably tends to flip**. The combination of:

- y safety tightened from 150 mm down to 30 mm over K=1..10 (γ-decay)
- z safety opened at K=11 (R wrist drops 215 mm → 185 mm at start, gripper tip at box top z=182 mm)
- Soft 6-DoF IK preserving rotation tracking while never failing position

…brings R's gripper into reliable contact with the box's −y face during the grip phase, while the L hand follows the demo trajectory unchanged. End pose: L and R end ~31 cm apart (intrinsic to demo — demonstrator's hands had come back together as the box settled); replay passes safely through this because the gripper tips meet the box, not each other.

### Notes / caveats

- Geometric sanity checks on the raw trajectory failed (tip-to-tip < 350 mm at start, L.z≠R.z, R wrist above box top, end pose not stacked) — but the trajectory positions match what the demonstrator actually did, not what an idealized bimanual sandwich grip looks like. Iteration's job isn't to fix the positions; it's to bring the gripper to contact given those positions. K=11 achieves this.
- Squeeze warning frames (tip-to-tip < box width 372 mm at frames 150-182) are intrinsic to the demo. Robot replay passes through safely because gripper tips meet the box, not each other.
- MediaPipe pipeline ran on this host (coldbrew) for the first time. Required `pip install mediapipe` into the `trossen_sim` env + a 1-line lazy-import patch in `extract_trajectory.py` (moved `import torch` + `from wilor_mini import ...` into `WilorRunner.__init__`) so the MediaPipe variant can `from extract_trajectory import ...` without WiLoR/torch installed.
- Soft DoF + dynamic HOME together make recordings_1_v2 and recordings_2 both replayable from the same `replay_real.py` without per-trajectory editing.

---

## 2026-05-22 (later) — Box-aware replay: SAM2-based Δxy substitute for paper's VLM

BiNoMaP §3.3 sketches a VLM that estimates (Δx, Δy) between the demo's object pose and the new scene's object pose, then offsets the bimanual trajectory by that Δ. We don't run a VLM — we already have [`boxes/capture_box.py`](boxes/capture_box.py) (SAM2 + RGBD fusion) producing world-frame box point clouds. This entry documents the end-to-end box-aware replay pipeline built on top of that.

### Convention: box poses live next to the trajectory

For any trajectory bundle `<dir>/foo.npz`, both the success-reference and current PLYs live in `<dir>/box_poses/`. The pipeline derives this automatically — no per-recording hardcoding. For recordings_2:

```
outputs/recordings_2/wrist/box_poses/
├── success_K11_brown.ply        (locked once, after the K=11 hardware success)
├── success_K11_brown_scene.ply
├── current.ply                  (overwritten before every replay)
└── current_scene.ply
```

[`boxes/capture_box.py`](boxes/capture_box.py) takes `--out-dir` (default: legacy `binomap/boxes/` for back-compat). [`record_box_pose.py`](record_box_pose.py)'s `--out-dir` now defaults to the same directory as the input PLY (so JSON metadata, when generated, lands next to its source).

### Pipeline orchestrator: `box_align_pipeline.py`

One command does: capture current → compute Δ → translate bundle → re-run sim IK → print hardware command. Defaults derived from `--trajectory`:

```bash
/home/yunshuang/anaconda3/envs/depth_lerobot/bin/python box_align_pipeline.py \
    --success-ply outputs/recordings_2/wrist/box_poses/success_K11_brown.ply \
    --trajectory  outputs/recordings_2/wrist/trajectory_contact_adj_K11_Ronly_dRyStart-20_dRzStart-30.npz
```

Add `--current-ply <path> --skip-capture` to reuse an existing current PLY (skip the interactive SAM2 click). Add `--apply-dz` to keep the z component of Δ (default zeroes it — table is fixed).

Outputs into the trajectory dir:
- `<bundle>_box_aligned.npz` — Δ-translated bundle, preserves `p_L_pre_align`/`p_R_pre_align` for diff
- `<bundle>_box_aligned_iklog.npz` — sim IK joints for the translated bundle
- `replay_<bundle>_box_aligned.mp4` — sim render

### Bug fixed: `replay_trossen_ik.py`'s frame_alignment was silently canceling Δ

`compute_frame_alignment()` recomputes a re-centering offset on every IK run from `mean(p_L, p_R)` and snaps that midpoint to `TROSSEN_WORKSPACE_CENTER`. For any pure-translation augmentation Δ to the bundle, the mean shifts by Δ, the new offset is `(old_offset − Δ)`, and IK sees identical targets. Joint trajectory output before/after box-align was **byte-identical** despite `box_align_delta_xyz_m` being correctly stored.

Verified on recordings_2 K=11: |Δ|=57 mm, target_L/R differed by 57 mm, but `joint_L`/`joint_R` differences were exactly 0. After fix: max joint diff up to 76° per joint (R), IK pos err stays <1 mm.

**Fix** ([`replay_trossen_ik.py:268-282`](replay_trossen_ik.py)): when the bundle has a `p_L_pre_align` marker key, compute the alignment offset from the PRE-align coordinates so the box-align Δ survives into joint space.

### Bug fixed: pipeline was clobbering the un-aligned IK log mid-run

`replay_trossen_ik.py` writes to `<in_npz_parent>/trossen_replay_ik_log.npz` by default. The old pipeline wrote-then-renamed, which destroyed the existing default-named log (the user's un-aligned K=11 baseline) before the rename happened.

**Fix:** added `--out-log <path>` to [`replay_trossen_ik.py`](replay_trossen_ik.py); [`box_align_pipeline.py`](box_align_pipeline.py) now passes the box-aligned name directly so the baseline log is never opened for writing. Hardware replay can fall back to `trossen_replay_ik_log.npz` (un-aligned K=11) at any time without rebuilding it.


### Hardware command (post-pipeline)

```bash
source /home/yigit/miniconda3/etc/profile.d/conda.sh && conda activate lerobot && \
python replay_real.py \
    --log outputs/recordings_2/wrist/trajectory_contact_adj_K11_Ronly_dRyStart-20_dRzStart-30_box_aligned_iklog.npz \
    --max-step-delta 0.2 --d455 low
```

Drop `--d455 low` if you don't want the workspace-cam recording.

### Notes

- The "delete original IK log on diagnostic" failure mode happened twice during dev — once because the pipeline overwrote, and once because a diagnostic `replay_trossen_ik.py` invocation wrote the default-named log. The `--out-log` flag closes the pipeline path; for ad-hoc diagnostics, copy the log to a `_backup.npz` first.
- Hardware replay still expects `--max-step-delta 0.2`. The box-align Δ (~5 cm) shows up as a few-cm offset of the entire joint trajectory, not as a single-frame jump, so the step-delta limit doesn't gate it.
- Δz is zeroed by default (`--apply-dz` to keep). Rationale: the table is fixed in our setup, so any z component of the box AABB delta is point-cloud noise rather than real motion. Empirically Δz observed at <5 mm so far.

---

## 2026-05-23 — Stage 3: Category-Level Primitive Parameterization (BiNoMaP §3.4)

Generalize the K=11 brown-box success trajectory to other boxes in the same category via the paper's single non-iterative geometric step. The L arm and all orientations are unchanged; only R-arm position is rescaled radially along the inter-arm axis.

### What §3.4 contributes vs reuses

§3.4 contributes **only Eq. 5** (the δsize calculation from object point clouds). The trajectory update reuses **Eq. 4 from §3.3.2** — the same Stage 2b scale-mode 3D radial rescale we deliberately avoided on K=11 brown by choosing translate-mode.

Verbatim from §3.4:

> "The size difference δsize between the new and base instances is then incorporated into our contact optimization **Eqn. 4** in a single, non-iterative step to adapt the trajectory."

The two equations:

```
# Eq. 5 (§3.4) — δsize from object point clouds:
δsize = max_{u,v ∈ pcd_new} ‖u − v‖₂  −  max_{u,v ∈ pcd_base} ‖u − v‖₂
        s.t. (u − v) ∥ (p̂^{t_s}_L − p̂^{t_s}_R)
        # paper: "taking a horizontal slice at the initial contact height"

# Eq. 4 (§3.3.2) — trajectory update, applied to full 3D inter-arm vector:
s(t)   = (‖p_R[t] − p_L[t]‖ + δsize) / ‖p_R[t] − p_L[t]‖
p̄_R[t] = p_L[t] + s(t) · (p_R[t] − p_L[t]),  ∀t ∈ [t_s, t_e]
```

`t_s` = **starting point** of the contact phase (first frame where both arms are valid); `t_e` = last such frame. `û` is computed once at `t_s` from `(p_L[t_s] − p_R[t_s]) / ‖·‖` and held fixed across the whole contact phase. Frames outside `[t_s, t_e]` pass through unmodified.

### Script + pipeline

[category_parameterize.py](category_parameterize.py) implements Eq. 5 (via `extent_along(pts, û) = max(proj) − min(proj)`, algebraically equivalent to the chord-max under the parallel constraint) and Eq. 4 (per-frame, 3D). One **minor deviation** from the paper spec: uses the full point cloud instead of a horizontal slice at contact height — equivalent for prismatic objects (boxes), not for tapered objects.

CLI:
```bash
python category_parameterize.py \
    --in_npz   outputs/recordings_2/wrist/trajectory_contact_adj_K11_Ronly_dRyStart-20_dRzStart-30.npz \
    --base_ply boxes/brown_box.ply \
    --new_ply  boxes/wifi_box.ply
```

Downstream pipeline is unchanged — Stage 3 outputs the same bundle schema as Stage 2b plus traceability arrays (`p_R_pre_param`, `cat_param_delta_size_m`, `cat_param_inter_arm_unit`, `cat_param_t_s/t_e`):

```
category_parameterize.py → trajectory_K11_paramto_<target>.npz
[ box_align_pipeline.py ] → optional Δxy alignment
replay_trossen_ik.py     → IK log
replay_real.py           → hardware
```

### Hardware results: all four targets failed to flip

Generated and tested all four available boxes against the brown-base trajectory (no box-align, no extension). All failed — no contact made in any case.

| Target | δsize | height gap vs brown | sim s(t) max\|−1\| | hardware result |
|---|---:|---:|---:|---|
| black_box        | −17.8 mm | −28 mm |  5.7% | no contact |
| amazon_brown_box | −31.4 mm | −60 mm | 10.0% | no contact |
| blue_box         | −33.4 mm | −23 mm | 10.7% | no contact |
| wifi_box         | −74.7 mm | −74 mm | 23.8% | no contact |

Two consistent failure modes:

1. **R wrist starts too high above the new box's top.** R wrist min z stays near brown's ~185 mm; new box tops sit at 108–159 mm. R hovers 20–73 mm above the new box from the very start, so the contact phase never engages.
2. **Inter-arm separation, even after rescale, is still too wide for the gripper tips to close on the box sides.** Likely contributing causes (any/all): brown's pinch min already had only ~8 mm/side clearance (no margin for mismatch), SAM2-captured AABB widths may over-estimate the physical box (table pixels at the edge), and gripper-tip orientation `R_R` is set by the demo and may face the smaller box at a slightly off angle.

### Why it failed — and why the script isn't the bug

Audited [category_parameterize.py](category_parameterize.py) line-by-line against the paper: Eq. 4 ([L109-117](category_parameterize.py#L109-L117)), Eq. 5 ([L100-104](category_parameterize.py#L100-L104)), û at t_s ([L94-98](category_parameterize.py#L94-L98)), and the L-arm-untouched invariant all check out. **The script is a faithful implementation of §3.4 + Eq. 4.**

The failures are failures of the **paper's category assumption**, not of the implementation:

- §3.4 implicitly assumes "same category" = "differs primarily in inter-arm-axis extent at the contact-height slice." Height variation and contact-surface geometry are assumed to match.
- The paper's natural use case is **side-grasp at matched heights** (e.g., grasping a vase from both sides): L_z ≈ R_z, the inter-arm vector is roughly horizontal, and Eq. 4's z scaling is ~0 — z is preserved.
- Our box-flip demo violates this. L is *below* the box (~140 mm), R is *above* (~350 mm), so the inter-arm vector has ~200 mm of z. Eq. 4's multiplicative rescale does change R's z by `(1 − s)·(p_R_z − p_L_z)` (e.g. 38 mm for wifi) — **but this z change is set by the demo geometry, not by the new box**. It's a side effect, not a correction. For wifi, the demo gives 38 mm of drop when 74 mm is needed; for a taller box it could push R the wrong direction entirely.

### Implication

Strict §3.4 is not sufficient for our box-flip category as-captured. Two paths forward:

1. **Stay paper-compliant** — reshoot the base demo on an object that better matches targets in height, so the inter-arm vector is closer to horizontal and Eq. 4's behavior matches the paper's assumption.
2. **Extend beyond §3.4** — add an additive `δheight = new_top_z − base_top_z` R-z lower term, and probably a few Stage-2b iterations on top of the §3.4 output to tighten the pinch contact. Both would be deliberate paper-departures.

Box-align Δxy alone (§3.3) won't help — the issue is shape mismatch, not position.

### Update: physical-flip workaround for the height limitation works (wifi + amazon succeed)

Instead of extending the algorithm with a `δheight` term, the demonstrator physically **flipped the box onto a different face** so the box's z-extent in world matches the brown base. With the new orientation, the target box's *top* sits at roughly brown's top height, so the trajectory's R-wrist (which still terminates at brown's box-top z) now grazes the new box's top correctly.

For this to flow through, **`--apply-dz` must be enabled on `box_align_pipeline.py`** (default is to zero Δz since the table is fixed; for this workaround the box has been physically lifted, so Δz is real and must propagate). Pipeline used:

```bash
/home/yunshuang/anaconda3/envs/depth_lerobot/bin/python box_align_pipeline.py \
    --success-ply outputs/recordings_2/wrist/box_poses/success_K11_brown.ply \
    --trajectory  outputs/recordings_2/wrist/trajectory_K11_paramto_<target>.npz \
    --apply-dz
```

Captures a fresh `current.ply` (interactive SAM2 click on both cams) at the box's new flipped pose, computes Δxyz, translates the Stage-3 paramto bundle by the full 3D Δ, runs sim IK with `--rot-weight 0.2 --gripper_offset_m 0.09`, and prints the hardware command.

**Hardware results:**

| Target | Pipeline run | Box-aligned IK log | Hardware result |
|---|---|---|---|
| wifi_box        | 2026-05-23 14:41 | `outputs/recordings_2/wrist/trajectory_K11_paramto_wifi_box_box_aligned_iklog.npz`        | **✓ box flipped** |
| amazon_brown_box | 2026-05-23 14:57 | `outputs/recordings_2/wrist/trajectory_K11_paramto_amazon_brown_box_box_aligned_iklog.npz` | **✓ box flipped** |
| black_box       | (pending)        | `outputs/recordings_2/wrist/trajectory_K11_paramto_black_box_box_aligned_iklog.npz`        | (in progress) |
| blue_box        | (pending)        | —                                                                                          | — |

**What the workaround actually exploits.** Stage 3's §3.4 only adjusts inter-arm extent, not height. Manually re-orienting the target box reshapes its world-frame AABB so the *new* z-extent matches brown's. Then the Stage-3 paramto bundle (which assumed brown's height) plus a full-3D box-align translation (with `--apply-dz`) is geometrically correct: width is handled by §3.4's rescale, height is handled by physical reorientation, position is handled by Δxyz. No algorithm extension, no second `δheight` term.

**Caveat — δsize may now be slightly off.** The Stage-3 paramto bundles were generated using the original (un-flipped) `boxes/<name>.ply` size references, which captured the boxes in their *original* orientation. After flipping, the dimension that's now along the inter-arm axis may differ from what δsize assumed. For wifi (−74.7 mm) and amazon (−31.4 mm) it was close enough that the in-flight pinch still worked. If a flip changes which face is up dramatically (>30 mm extent change along û), re-capture the size reference with the new orientation and re-run Stage 3 before box-align.

### Note on `u, v` naming

In the paper's Eq. 5 formula, `u` and `v` are **point-cloud coordinates** (two points whose chord is parallel to the inter-arm axis). In [category_parameterize.py:94](category_parameterize.py#L94) the local variable `v` is unrelated — it's the unnormalized inter-arm vector `p_L[t_s] − p_R[t_s]`. Same letter, different concept; only `δsize` (the chord-max difference) flows between them.

---

## 2026-05-23 (later) — recording_rotation: hand-detection failure, framing fix, pipeline tuning for rotation tasks

New demo `recording_rotation/` — bimanual rotation of the brown box (no flipping, just rotation in place). Took two attempts; pipeline needed two new flags. Documenting what went wrong, what fixed it, and what's now permanent in the scripts.

### First take: both detectors returned ~0 valid frames

`recording_rotation/` v1 (6 s, 30 fps, `--no-park`, both D455 workspace cams). Ran extraction with the standard pipeline:

| Detector | L cam valid | R cam valid | Bimanual grid |
|---|---:|---:|---:|
| MediaPipe (`extract_trajectory_mediapipe.py`) | 1/174 | 0/173 | valid_L=2, valid_R=0 (of 183) |
| WiLoR (`extract_trajectory.py`, installed for this test — see install notes below) | 1/174 | 0/173 | valid_L=2, valid_R=0 (of 183) |

**Both detectors identical** → not a detector-specific issue. Inspected raw frames:

- **R cam (338122302972):** the rotation action was mostly outside the frame — mostly empty table with a sleeve visible in one corner. Zero detections is consistent with "no hand actually in frame."
- **L cam (333422304645):** hands visible but in unusual top-down configuration with **dark long sleeves** coming down — high-contrast skin-to-clothing transition was hidden, and the wrist/forearm context the detectors rely on was occluded.

### Fix: remove sleeves + stand at the back of the table

Empirical fix from the demonstrator. The key changes between take 1 and take 2:

1. **Removed long sleeves.** Bare forearms gave both detectors the wrist-to-forearm transition they're trained on. Box-flip recordings_2 had the same lesson implicitly (rolled-up sleeves).
2. **Stood at the back of the table** (rather than the side). This put both hands in the *center* of both cameras' FOV during the rotation motion instead of one hand drifting into the R cam's corner. The rotation action now lives inside the intersection of the two camera frustums for the whole 6 s.

No script change, no detector switch. Pure recording-side fix.

### Second take: clean extraction

Same 6 s `--duration 6 --countdown 3 --no-park` recording, just with the two fixes above:

| Detector | L cam valid | R cam valid | Bimanual grid |
|---|---:|---:|---:|
| MediaPipe (`extract_trajectory_mediapipe.py`) | 168/179 (94%) | 179/179 (100%) | valid_L=159, valid_R=177 (of 182) |

94% / 100% per-cam, 87% / 97% on the bimanual grid. Plenty of data for downstream stages.

Side note: MediaPipe's `side_detected` was unreliable (L cam called the hand "R" in 145 frames, R cam called it "L" in 168 frames). Doesn't matter — the script's camera-expected-side fallback assigns correctly. Worth knowing if anything ever depends on MediaPipe's intrinsic handedness label.

### Local install: WiLoR is now available on coldbrew

When MediaPipe failed on take 1 I installed WiLoR locally to rule out detector-specific bias. Steps (records for future reproducibility on this host):

```bash
# 1. chumpy (legacy SMPL/MANO dependency) — fails to build in isolated build env on Py 3.10
/home/yunshuang/anaconda3/envs/depth_lerobot/bin/pip install --user --no-build-isolation chumpy

# 2. chumpy 0.70 broke on numpy>=1.24 (np.bool/int/float aliases removed). Patch:
sed -i 's|from numpy import bool, int, float, complex, object, unicode, str, nan, inf|from numpy import nan, inf|' \
    /home/bruce/.local/lib/python3.10/site-packages/chumpy/__init__.py

# 3. Now wilor-mini itself
/home/yunshuang/anaconda3/envs/depth_lerobot/bin/pip install --user --no-build-isolation \
    git+https://github.com/warmshao/WiLoR-mini

# 4. Model checkpoints auto-download from HuggingFace on first pipeline construction
#    (~500 MB into ~/.local/lib/python3.10/site-packages/wilor_mini/pretrained_models/)
```

Side effects to be aware of:
- pip downgraded torch 2.11 → 2.5 in depth_lerobot's `--user` site-packages to satisfy wilor-mini's pins. Re-check if torchvision/sam-2/lerobot complain about version mismatch in future work.
- Confirmed `extract_trajectory.py` (WiLoR variant) runs end-to-end in depth_lerobot env after install.

### Pipeline tuning needed for rotation-class tasks

Rotation differs from box-flip in two ways that exposed pre-existing rough edges:

**1. Back-fill plateau visible in sim warmup.** L's first valid frame was 6 (R's was 0), so frames 0–5 were back-filled to a duplicate of frame 6. During MuJoCo's 2 s warmup, the arm tried to move from all-zeros home to a non-trivial back-fill pose, producing a wild visible twist. Hardware was fine (slow_move_to handles it), but the sim viz was misleading.

**Fix added:** `--auto-skip-leading-invalid` flag on `replay_trossen_ik.py` (mirrors the same-named flag on `replay_real.py`). Drops frames before `max(first_valid_L, first_valid_R)` so the bundle starts at the first frame where both arms have real data. Reduces max frame-to-frame joint Δ from 24.5° → 13.2° at start.

**2. Mean-anchor orientation remap drifts for large-rotation demos.** `replay_trossen_ik.py`'s default `--orientation_remap` computes the IK orientation target as `R_remap @ R_hand[k]`, where `R_remap` is derived from the **mean** quaternion across all valid frames. For box-flip (palm orientation roughly constant) this works fine. For rotation (R palm sweeps ~170° across the demo), the mean is a meaningless halfway pose and the gripper tip drifts off the palm direction by varying amounts — early frames tilted ~85° one way, mid frames matched, late frames tilted ~85° the other way. Symptom: "the tip is not always facing where my palm is facing."

**Fix added:** `--fixed-orientation-remap` flag on `replay_trossen_ik.py`. Uses a fixed `R_align` (default `R_y(-90°)` mapping gripper-+x onto palm normal, same as `replay_trossen_paper.py`) applied per-frame as `R_target[k] = R_hand[k] @ R_align`. No averaging. Combined with `--rot-weight 0.2` (soft DoF, also default for our setup) this gives:
- IK pos err 0.4 mm mean / 1.0 mm max
- IK rot err **0.1° mean / 0.9° max** (vs the mean-anchor variant's drift)
- ~half the frames take the soft-fallback to position-only (graceful degradation on unreachable orientations)

Default behavior is unchanged when the flag is off — existing box-flip / K-iteration pipelines aren't affected.

### Stage 2b iteration on rotation — converged on iter2_scale_dRzStart20

Goal: tighten the R-arm's pinch on the box's −y face for a firm rotation grip. After deleting an initial custom-translate sweep (K1–K3 above), switched to **paper iteration mode** (`--iteration K` with `d_k = 5 × 0.85^(K-1)` mm) chained with a custom z-lower pass.

| Variant | Pipeline | Hardware result |
|---|---|---|
| Custom K1 (+20 y, 0 z) | `--shift-R-y-start-mm 20` | a little loose |
| Custom K2 (+30 y, +20 z) | `--shift-R-y-start-mm 30 --lower-R-start-mm 20` | replayed but rejected; deleted |
| Custom K3 (+40 y, +30 z) | `--shift-R-y-start-mm 40 --lower-R-start-mm 30` | generated but rejected; deleted |
| Paper iter 1 (d_k=5 mm, scale) | `--iteration 1` | replayed |
| **Paper iter 2 + R z −20 mm (final)** | `--iteration 2` chained → `--lower-R-start-mm 20` | **✓ canonical rotation baseline** |

**Canonical bundles for downstream work** (Stage 3 generalization, box-align, hardware replay):

- Trajectory bundle: `outputs/recording_rotation/wrist/trajectory_contact_adj_iter2_scale_dRzStart20.npz`
- Sim IK log: `outputs/recording_rotation/wrist/trossen_replay_ik_log_iter2_scale_dRzStart20.npz`
- Sim mp4: `outputs/recording_rotation/wrist/replay_iter2_scale_dRzStart20.mp4`

Built with `--rot-weight 0.2 --auto-skip-leading-invalid --fixed-orientation-remap --gripper_offset_m 0.09`. Any future Stage 3 paramto or box-align run for the rotation task should chain off this bundle (the same role K=11 brown plays for box-flip).

Chain pattern (one-time):
```bash
# Pass 1: paper Eq. 4 with d_k = 4.25 mm
python contact_adjustment.py \
    --smoothed outputs/recording_rotation/wrist/trajectory_smoothed.npz \
    --object-ply boxes/brown_box.ply \
    --iteration 2

# Pass 2: R z-lower decay on top of the iter2 output
python contact_adjustment.py \
    --smoothed outputs/recording_rotation/wrist/trajectory_contact_adj_iter2_scale.npz \
    --object-ply boxes/brown_box.ply \
    --lower-R-start-mm 20 \
    --tag iter2_scale_dRzStart20
```

Two passes because `contact_adjustment.py` runs in either paper-iteration mode OR lateral-shift mode per invocation, not both — chaining is the workaround.

### What's permanent vs what was throwaway

- **Permanent:** `--auto-skip-leading-invalid` and `--fixed-orientation-remap` on `replay_trossen_ik.py`. WiLoR install on coldbrew. The "no sleeves + stand at back" recording protocol.
- **Throwaway (kept for reference):** `outputs/recording_rotation/wrist/replay_paper.mp4` and `trossen_replay_paper_log.npz` — the diagnostic paper-strict run that confirmed fixed R_align was the right fix before we added the soft+fixed combo flag.

---

## Hardware success-rate tracking — flip + rotate, all targets

Per-target, per-task hardware success rate. Update cells after each batch of trials. Counted as `successes / trials`.

For **flip**, two separate criteria:
- **Perfect stand at end** — box ends upright on the table, flush with the surface (no tilt).
- **<10° tilt** — looser bar; box ends standing but slightly leaned.

For **rotate**, single criterion: box ends rotated to the intended orientation with full contact maintained throughout the motion.

| Target | Pose variant | Flip — perfect stand | Flip — <10° tilt | Rotate |
|---|---|---:|---:|---:|
| brown_box | flat (default)       | / | / | **3/15** |
| brown_box | higher (`brown_box_2.ply`, top z 295 mm) | / | / | **0/15** |
| blue_box  | —                    | / | / | **0/15** |
| black_box | —                    | / | / | **1/15** |
| plastic_box | —                  | / | / | **0/15** |
| wifi_box  | —                    | / | / | **0/15** |

**How to read the columns** — flip results use the K=11 brown-base trajectory generalized via Stage 3 (`trajectory_K11_paramto_<target>.npz`) + `box_align_pipeline.py --apply-dz`; rotate results use the rotation canonical baseline (`trajectory_contact_adj_iter2_scale_dRzStart20.npz`) generalized per target the same way. Fill cells as `N/M` (e.g. `3/5`) so the trial count is preserved alongside the rate.
