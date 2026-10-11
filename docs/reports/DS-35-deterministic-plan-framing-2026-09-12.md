# DS-35 — Deterministic projected-region framing

Date: 2026-09-12

Status: implemented in the evaluation-only dataset planner; no queue or live capture created.

## Planner behavior

Each planned view keeps the existing seeded azimuth, elevation, FOV, resolution, weather, and setting-stratum draws. A deterministic radius fitter now evaluates the continuous voxel-volume AABB `[min, max + 1]` using all eight corners projected with that view's actual camera FOV and image dimensions. It chooses the feasible sampled radius whose dominant envelope dimension `max(width_fraction, height_fraction)` is closest to 0.58, constrained to `[0.45, 0.70]`, with all corners inside the half-open pixel rectangle `0 <= x < width`, `0 <= y < height`. The returned `bounds_projected_occupancy` records both W/H fractions; these are 2D envelope measurements, not depth-tested visibility.

Both player feet and rendered eye must be at least 2.0 blocks from the continuous region AABB. The fitter brackets that clearance and resolves its lower radius with 40 deterministic bisection steps, then samples 64 logarithmic radii up to the existing cap `max(16, 8 × region_diagonal)` and uses 24 bisection steps around sampled crossings of the 0.58 target. Its clearance comparison uses `1e-6` tolerance and adds `1e-4` blocks after the clearance root. If no candidate satisfies clearance, full corner coverage, and the occupancy band before the distance cap, planning raises `DatasetSchemaError` with the failed view, cap, band, and nearest sampled dominant envelope.

The default target is now the center of `[min, max + 1]`, including half-block coordinates where appropriate. Explicit targets are unchanged. Camera pose serialization, `CapturePlan` schema, validation, frame IDs, setting strata, and seeded random draws remain unchanged. Projection continues to use the project's vertical-FOV convention. No capture CLI or reconstruction path was changed.

## Synthetic before/after measurement

For each region, the old planner was reproduced with seed 23 and six views, retaining the same per-view FOV and 16:9 resolution choices; the old radius stopped at the first 1.25× expansion with complete corners. The new plan used the same random draws and selected max(W,H) = 0.580 for all six views. Range columns list the minimum and maximum across the six views; clearances are minimum point-to-AABB distances over both feet and eye.

| Synthetic region (W×H×D) | Old dominant min/median/max | Old W range | Old H range | New W range | New H range | Old min clearance | New min clearance |
|---|---:|---:|---:|---:|---:|---:|---:|
| Box (20×20×20) | 0.449 / 0.520 / 0.809 | 0.256–0.438 | 0.449–0.809 | 0.289–0.454 | 0.580–0.580 | 16.60 | 13.37 |
| Tall (8×40×12) | 0.551 / 0.744 / 0.813 | 0.109–0.162 | 0.551–0.813 | 0.077–0.131 | 0.580–0.580 | 31.23 | 29.54 |
| Wide (60×8×12) | 0.312 / 0.485 / 0.558 | 0.236–0.558 | 0.153–0.487 | 0.383–0.580 | 0.173–0.580 | 24.85 | 12.91 |
| Deep (10×15×48) | 0.307 / 0.528 / 0.559 | 0.111–0.559 | 0.306–0.535 | 0.205–0.580 | 0.513–0.580 | 17.66 | 10.96 |

The old plan already cleared these synthetic bounds, but envelope sizes varied from 0.307 to 0.813 across views. The new plans keep both feet and eye outside each region with at least the 2-block clearance and fit the chosen occupancy band. Separate tests passed all 9 combinations of tall/wide/deep synthetic regions with 16:9, 4:3, and portrait resolutions. A one-block synthetic region with 90° vertical FOV demonstrates explicit infeasibility when the 2-block clearance cannot also produce the 0.45 minimum occupancy. Right- and bottom-border pixels are rejected; the continuous AABB-center and explicit-target controls pass.

## Validation and scope

Focused command:

```text
python -m pytest tests/test_capture_dataset.py tests/test_capture_quality.py tests/test_capture_batch.py -o addopts= -p no:cacheprovider -q --deselect=tests/test_capture_dataset.py::test_plan_is_identical_in_a_fresh_process --deselect=tests/test_capture_batch.py::test_cli_materializes_a_unicode_identity_on_a_strict_cp1252_console
```

Result: **129 passed, 1 skipped, 2 deselected** in 4.93s. The deselections are exactly the two named tests that launch child processes. `py_compile` passed for the planner, additive package exports, and focused dataset tests; a trailing-whitespace scan passed for the source, tests, and report. No Train Station or Canal metadata, real labels, v4/C4 evaluation, live client, queue, materialization, subprocess test, source-save, junction, Defender, or Minecraft operation was used. No stage or commit.

SHA256: `plan.py` b585f96a38215c5e5ca338c3a295b45e5020ea4a695e670351ca3997fb91e408; `dataset/__init__.py` b928cc32c4047f543f5ed621888744aa2a0f12b4d612a326e14c7affae891a26; `test_capture_dataset.py` c1ef9deb8e0228ef54999d0a290aa13900d2d4b48fa29c6eb031ee87a4b3bb96.
