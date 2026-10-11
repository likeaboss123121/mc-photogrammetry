# DS-44 — QA-compatible anisotropic framing

Date: 2026-09-12

Status: implemented in the evaluation-only dataset planner. No capture queue, staging, or live client work was performed.

## Planner contract

Planning and capture-quality review now share the same occupancy floor constants from `dataset/framing_contract.py`; the existing QA defaults remain **0.30 projected width and 0.30 projected height**. The planner evaluates each view's continuous region AABB `[min, max + 1]` by projecting all eight corners with that view's vertical FOV convention, resolution, and camera. The full projected corner set must fit in the half-open image rectangle `0 <= x < width`, `0 <= y < height`; the projected envelope must meet both per-axis floors and its dominant dimension must not exceed the safe cap **0.85**. Both player feet and rendered eye retain the existing **2-block minimum clearance**. These are image-envelope measures, not depth-tested visibility.

The existing dominant occupancy band **[0.45, 0.70]**, target **0.58**, remains preferred. For each seeded azimuth, a deterministic elevation grid from 0.0 to 1.4 radians in 0.1-radian steps and the existing bounded radius search are evaluated. If no preferred-band camera meets the complete contract at that azimuth, local azimuth offsets of ±0.2 and then ±0.4 radians are tried. A preferred-band candidate at the nearest feasible offset wins; only when no preferred-band solution exists in this bounded neighborhood may a candidate between 0.70 and 0.85 be selected. If no candidate meets both axis floors, full-corner coverage, clearance, and the existing maximum-distance cap, planning raises `DatasetSchemaError` with the occupancy/coverage constraints.

Radius search remains deterministic: 64 logarithmic samples between the clearance boundary and `max(16, 8 × region_diagonal)`, the seeded preferred radius, and bounded bisection around sampled crossings of 0.58. Random draws and setting strata remain unchanged. Existing request schemas already support **4–8 views** (`MIN_VIEWS`/`MAX_VIEWS`, `RegionRequest.views`, and the planner CLI); no schema change was needed to request eight views.

## Synthetic evidence

The Canal-Town-like synthetic AABB is 17×23×16 voxels. With a frontal azimuth, 70° vertical FOV, and 1600×900 resolution, the selected projection measured **W=0.3814, H=0.5800**, all 8 corners in-frame, feet clearance **16.500** blocks, and eye clearance **18.003** blocks. A separate constrained-orientation control exercises the safe fallback: **W=0.3115, H=0.8445**, with all corners in-frame and both axes above 0.30 while exceeding the preferred 0.70 ceiling.

Generated plans are also checked for all jobs across 12 synthetic combinations: tall 8×40×12, wide/flat 60×8×12, deep 10×15×48, and 17×23×16 regions at 16:9, 4:3, and portrait resolutions. Each planned camera satisfies both QA axis floors, the 0.85 cap, full-corner coverage, and clearance. A separate validation regression scales a valid camera outward and confirms `validate_plan` rejects its undersized projected occupancy. A one-voxel region continues to fail clearly when clearance and occupancy cannot both be satisfied.

## Validation and scope

Focused command:

```text
.venv\Scripts\python.exe -m pytest -o addopts= -ra --tb=short tests\test_capture_dataset.py tests\test_capture_quality.py
```

Result before the final fallback-only test was added: **108 passed, 1 skipped** in 21.63s; the skip is the existing Windows symlink-permission test. The final frontal and fallback controls then passed **2/2** in 0.27s. Full Python regression passed **1,021 tests**, with the same single symlink-permission skip, in 198.63s; it exercised the completed planner and validation source, before only the additional fallback assertion was added.

No Train Station/Canal capture artifact, real label, C4/v4 score, reconstruction path, dataset, or queue was read or changed for implementation or tuning. The new planner dependency is a two-constant contract module; no QA artifact loader or reference/manifest input was added to planning.

SHA-256 at closeout:

- `plan.py`: `52c2ea38a79aa945576b59a231f3ebe5c3813edf8d634427148bcb9f03f0ec58`
- `quality.py`: `9c7a25513dad0d6947ffff3f2c15e81e44731371e28455d333428ba688d1de0b`
- `framing_contract.py`: `bce0db097b55c714452f58e668b374f0a4be6b9456423b12120e2a09a91bd24a`
- `test_capture_dataset.py`: `7fc3346c6e72a5b3a199b03a38a0349c3e645d11ab3d4160b264061eb07459dd`
- `test_capture_quality.py`: `53a82608b7d261be71a3bc3c7863af89a2b194c7d2b158c671f13f5a3f1dcc42`
