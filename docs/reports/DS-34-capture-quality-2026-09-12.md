# DS-34 — Capture image QA and framing helper

Date: 2026-09-12  
Status: evaluation-only implementation; no capture queue or live run created.

## Decision

The finalized `train-station-spawn-208-96-orbit-a` artifact has consistent capture/ground-truth scene identity and bounds, six complete hashed screenshots, matching camera/frame/resolution cardinality, converged poses, closed screens, and matching decoded, framebuffer, window, requested, and calibrated image dimensions (1600×900). It passes capture-integrity QA but is **not eligible as a building-promotion holdout**: only 1/6 views meets the provisional per-view policy, below the minimum 4. Keep it as integrity-valid adverse-condition data.

The rejection is driven by a near-black frame, persistent dark upper bands consistent with the previously inspected overhead terrain, and a small projected target envelope in three views. HUD-visible frames 00, 01, and 03 are tagged as ordinary-input coverage and are not rejected solely for showing the HUD. All region corners and occupied block centers project inside all six image bounds, so this audit detects no projected truncation. Projection is a 2D camera projection only: occupancy/coverage says nothing about depth-tested visibility or occlusion.

| Frame | Near-black | Clipped | Luma mean / std | Entropy (bits) | Top 10% dark | Projected GT envelope W×H | Usable |
|---|---:|---:|---:|---:|---:|---:|:---:|
| 00 | 0.3369 | 0.0000 | 71.59 / 66.34 | 6.783 | 0.4979 | 0.2926×0.2652 | no |
| 01 | 0.4673 | 0.0000 | 65.61 / 66.49 | 6.466 | 0.5677 | 0.2670×0.3240 | no |
| 02 | 0.9525 | 0.0000 | 10.09 / 5.46 | 3.347 | 0.9995 | 0.3204×0.3319 | no |
| 03 | 0.3841 | 0.0015 | 79.21 / 71.06 | 6.874 | 0.4317 | 0.3821×0.3299 | no |
| 04 | 0.1683 | 0.0000 | 117.45 / 71.07 | 5.616 | 0.3858 | 0.2514×0.2869 | no |
| 05 | 0.1238 | 0.0000 | 125.14 / 72.70 | 5.086 | 0.2634 | 0.3920×0.4337 | yes |

## QA policy and implementation

`src/mc_photogrammetry/dataset/quality.py` provides read-only per-frame and scene reports, plus an orbit-framing helper. Integrity checks cover screenshot decode/hash, capture completion/errors, pose/screen state, exact frame-camera cardinality, capture/ground-truth scene ID and region agreement, optional resolution cardinality, and actual screenshot size against framebuffer, actual and requested window sizes, declared resolution, and camera intrinsics. Integrity validity is distinct from promotion eligibility and adverse tagging; clean HUD-visible coverage or an insufficient view count alone does not make a scene adverse, while integrity-valid gamma/night variants can coexist with four other usable views.

The policy defaults are **provisional promotion gates**, not empirically validated general camera-quality thresholds. They were frozen and exercised against synthetic bright, saturated, low-contrast, black, dark-band, usable-variant, and truncated-projection controls; they were not fitted or tuned on Train Station. Defaults: near-black threshold 16/255 and severe fraction 0.80; clipped threshold 250/255 and maximum fraction 0.10; minimum luminance standard deviation 8 and entropy 4 bits; top-band height 10% and maximum dark fraction 0.40; projected GT envelope width and height each at least 0.30; all eight region corners in frame and at least 0.95 occupied-center coverage; minimum four usable views. A projected GT region that is truncated or fails coverage cannot pass merely by having a large in-frame envelope.

`derive_orbit_framing` accepts positive region dimensions, vertical FOV in degrees, aspect ratio, target width/height occupancy, and a clearance. It returns a horizontal camera radius and elevation or raises `ValueError` for invalid/infeasible inputs so the caller can abstain. It approximates both projected extents from the horizontal region diagonal and uses the project's **vertical FOV** convention; returned occupancy is a planning estimate, not a camera visibility guarantee. It is not wired into capture planning or reconstruction.

The new report contracts are exported additively from `mc_photogrammetry.dataset`. No default CLI, reconstruction input, dataset factory behavior, or capture artifact was changed.

## Canal Houses reconnaissance hypothesis

No Canal Houses capture, world, or visual artifact was found in the inspected project paths. The available coordinates alone are insufficient to establish a building region. For lead visual review only, a broad unmaterialized scout hypothesis is min `[-26, -64, -34]`, max `[13, -8, 5]`, including both the level spawn `(0, -63, 0)` and saved player near `(-6.3, -31.4, -13.8)`. This may include substantial irrelevant terrain and is not a promotion region; excluding spawn could instead omit lower geometry. If approved after visual inspection, a provisional six-view orbit would look toward the hypothesis center from azimuths 0°, 60°, 120°, 180°, 240°, and 300° using 70° vertical FOV and 16:9 framing. The helper estimates about 58.15 blocks horizontal radius, 0° elevation, and projected bounds occupancy 0.391×0.700. Nothing was materialized or queued.

## Validation and scope

Validation: `python -m pytest tests/test_capture_quality.py tests/test_capture_dataset.py tests/test_capture_batch.py -o addopts= -p no:cacheprovider -q --deselect=tests/test_capture_dataset.py::test_plan_is_identical_in_a_fresh_process --deselect=tests/test_capture_batch.py::test_cli_materializes_a_unicode_identity_on_a_strict_cp1252_console` passed **113**, skipped **1**, and deselected exactly those **2** child-process tests (4.04s). `py_compile` passed for the QA module, package exports, and QA tests. Synthetic tests cover dark/clipped/low-contrast behavior, HUD semantics, a dark variant with four usable views, clean-but-insufficient-view semantics, artifact cardinality/identity/region mismatches, actual window size, projected truncation, framing validity, package exports, and the read-only Train Station hold result. No live client, subprocess, junction, queue, worker, Defender setting, or source world operation was performed. No real/v4/C4 mask or label informed thresholds. No stage or commit.
