# M1 baseline and contracts

Completed 2026-09-07 on the existing Python 3.10 virtual environment.

## Reproducibility

- Local Git baseline: `65d2a16` (`Snapshot reconstruction baseline before M1`). Fixture media,
  ground-truth litematics, texture caches, generated exports, environments, and scratch artifacts
  are excluded by `.gitignore`.
- Verification: `./.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider` passed; the
  post-M1 collection contains 252 tests.
- Local asset: `%APPDATA%/.minecraft/versions/1.21.5/1.21.5.jar`, 28,984,409 bytes. Each report
  records its SHA-256, image/reference/manifest hashes, cache schema, environment, dependencies,
  effective settings, timings, and peak Python heap.

## Supplied-camera diagnostic baselines

These runs intentionally consume the fixture camera manifest and scene bounds. They are evaluation
diagnostics, not image-only production results. Visibility used all decoded pixels because a frozen
fixture mask has not yet been supplied; this limitation is recorded in each JSON report.

| Fixture | Visible exact-state recovery | Visible geometry F1 | Reconstruction / total | Peak Python heap |
| --- | ---: | ---: | ---: | ---: |
| Template 2 | 79.53% | 92.81% | 34.97 s / 41.17 s | 206.61 MiB |
| Template 6 | 29.72% | 85.46% | 203.21 s / 213.48 s | 286.25 MiB |

The reports are [Template 2](../exports/baselines/m1-template-2-supplied-camera.json) and
[Template 6](../exports/baselines/m1-template-6-supplied-camera.json). Both set
`qualifies_image_only_production` and the promotion gate to `false`; their supplied manifests make
them ineligible regardless of score.

## Contracts and benchmark rules

- `pipeline.contracts` now defines `FrameObservation`, explicit crop/resize transforms, intrinsics,
  pose, `CameraSolution`, `LatticeScene`, `RenderResult`, and `ReconstructionResult`.
- `evaluation.visible_metrics` separates visible exact-state recovery, visible geometry precision,
  recall, F1, and phantom counts from legacy whole-reference comparisons. It has no fixture file
  access.
- `scripts/benchmark_templates.py` requires an explicit `--reference`, `--camera-manifest`, and
  `--scene`. Its modes are `image-only`, `supplied-camera`, and `reference-assisted`; the current
  reconstructor rejects the image-only mode rather than pretending supplied cameras are derived.
- Reference-palette appearance calibration is available only in `reference-assisted` mode.

M2 may use the camera and renderer contract types as the shared interface. Claude's M3 experiment
remains read-only on the shared projection implementation until its calibration evidence is ready.
