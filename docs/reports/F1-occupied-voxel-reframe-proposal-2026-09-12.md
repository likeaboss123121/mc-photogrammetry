# Factory F1: occupied-voxel evaluation reframe proposal (2026-09-12)

## Finding

The frozen scene `exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox` is capture-integrity-valid, but remains promotion-ineligible at 0/8 usable views. QA measures projected occupied unit-voxel support, with block-center coverage reported separately. On the saved cameras, occupied-envelope width is 0.2174-0.2459 and height is 0.4129-0.5714; all eight views have full requested-region corner coverage and full occupied-center coverage. The sole per-frame promotion reason is `small_projected_gt_occupancy`. The earlier center-point width metric was 0.1982-0.2234; projecting occupied voxel corners corrects the physical support measure, but still does not meet the frozen 0.30 width floor.

## Evaluation-only proposal

Added `propose_evaluation_only_reframe` in `dataset/quality.py`. It uses only finalized ground-truth occupied block records, requested bounds, and an existing camera. It moves the camera eye inward along the existing region-center ray while preserving yaw, pitch, roll, vertical FOV, intrinsics, and eye height. It searches 96 deterministic scales from 1.0 to 0.05 and selects the largest feasible scale. Feasibility requires occupied voxel width/height >= 0.30, occupied voxel corners and all requested-region corners inside the half-open image, occupied-center coverage >= 0.95, projected requested-region envelope <= 0.85, and both feet and eye at least 2 blocks from the continuous region AABB. Limits share `framing_contract.py` with the planner. If no sampled camera passes, the typed result abstains. The helper is not exported from the dataset package and is absent from normal CLI, reconstruction, and pipeline sources; it cannot alter capture plans or artifacts.

Read-only projection of the existing eight GT/camera pairs produced:

| Frame | Scale | Proposed feet (x,y,z) | yaw / pitch | Occupied W/H | Region W/H | Min feet/eye clearance |
|---|---:|---|---|---|---|---:|
| 00 | 0.86 | (164.372, 88.024, 386.652) | 173.945 / 34.377 | 0.3004 / 0.6718 | 0.5622 / 0.6747 | 9.65 / 9.67 |
| 01 | 0.72 | (173.758, 88.966, 379.706) | 133.561 / 40.107 | 0.3023 / 0.7897 | 0.5088 / 0.8077 | 3.86 / 4.18 |
| 02 | 0.85 | (172.408, 101.381, 369.362) | 92.092 / 68.755 | 0.3002 / 0.5512 | 0.4734 / 0.6979 | 12.46 / 14.07 |
| 03 | 0.71 | (172.695, 94.533, 358.268) | 43.528 / 51.566 | 0.3013 / 0.7432 | 0.5285 / 0.8191 | 6.40 / 7.84 |
| 04 | abstain | unchanged | unchanged | saved 0.2252 / 0.5247 | no feasible fit | - |
| 05 | 0.77 | (158.282, 110.365, 364.806) | -45.164 / 80.214 | 0.3017 / 0.5299 | 0.4804 / 0.8356 | 21.36 / 22.98 |
| 06 | 0.80 | (151.274, 104.802, 369.649) | -93.307 / 68.755 | 0.3005 / 0.5880 | 0.4512 / 0.7411 | 16.04 / 17.63 |
| 07 | 0.77 | (154.443, 108.601, 378.845) | -140.702 / 68.755 | 0.3035 / 0.5808 | 0.4653 / 0.7642 | 19.69 / 21.30 |

For view 04 the radial proposal honestly abstains. At scale 0.80, occupied width remains 0.284 with all requested-region corners inside. At scale 0.76, occupied width reaches 0.303 but requested-region corner coverage falls to 0.75, so no candidate satisfies both requirements. This is a geometric incompatibility on that ray, not a threshold change. The projections are not depth-tested visibility claims; they do not certify rendered appearance or occlusion.

The saved capture and GT files were read only; no plan, screenshot, queue, or capture was changed. The proposed positions are diagnostic output only and have not been sent to a client.

## Validation

Focused command: `.venv\\Scripts\\python.exe -m pytest tests/test_capture_quality.py tests/test_capture_dataset.py -q -o addopts=''` -> **116 passed, 1 skipped** in 27.48 s. The single skip is the existing platform symlink-permission test. New synthetic coverage checks deterministic inward framing and preserved camera calibration, floor/coverage/clearance gates, typed abstention for mis-aimed and infeasible inputs, and a poison boundary excluding the helper from normal CLI/reconstruction/pipeline code.

SHA-256 (lowercase hex):

- `src/mc_photogrammetry/dataset/quality.py`: `8f4943696de51660c36f5e06ca99a49c2ae74b35038ec636d44f0329e97e8b6e`
- `src/mc_photogrammetry/dataset/plan.py`: `422c14743782d952996f3d3198d13659305a0c7b45ee9c4761d873d0cda335ac`
- `src/mc_photogrammetry/dataset/framing_contract.py`: `369b9673e8d5a3c16c683247abf2e4eb0dfd9e60bcaa8ad51771c95077388bc9`
- `tests/test_capture_quality.py`: `e79c9a75d6d9b585155b74afadd3b6a2b1483679122a8720a406e11368c9364d`
- frozen `capture.json`: `c6c44d43408779006854c81742a68d00cfe5b546f506ffa08ea59ba7b812abe6`
- frozen `ground_truth.json`: `36114fbee61d7b358927095a159d56dc5762e02c5a52a52e8a12ae17ce4a35a`

No full suite, live capture, launch, staging, label/C4 scoring, reconstruction integration, or git staging/commit was performed.
