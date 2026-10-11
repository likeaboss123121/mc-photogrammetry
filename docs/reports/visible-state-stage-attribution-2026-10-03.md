# Visible-state stage attribution — 2026-10-03

## Scope

Three supplied-camera diagnostics used the same reviewed T2/T6 screenshots, camera manifests, scene bounds, reference Litematics, vanilla 1.21.5 JAR and exact evaluator-pinned mask domains. They measure the reconstruction/classification stages conditional on supplied cameras; none is an image-only product result. Both pinned baseline runs use default settings. The T6 ablation changes only `palette_support=0.0` and is deliberately gate-ineligible.

The tracked JSON mask specs under `docs/evaluation_masks/templates-{2,6}` are the reviewed source of truth. Their file hashes and generated usable-domain hashes match `docs/reports/evaluator-fixture-pins.json` exactly. The ignored PNG derivatives under `.masks/` were stale duplicates and were not used.

## Results

| Fixture/run | Visible exact | Geometry F1 (P/R) | Palette | Support | Cell candidates | Raw winner | Smoothed | Placed | Final missing / wrong state |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T2 pinned default | 480/667 (71.96%) | 91.88% (93.08/90.70) | 599 (89.81%) | 562 (84.26%) | 561 (84.11%) | 459 (68.82%) | 480 | 480 | 62 / 125 |
| T6 pinned default | 350/2,007 (17.44%) | 85.39% (80.88/90.43) | 1,920 (95.67%) | 393 (19.58%) | 393 (19.58%) | 329 (16.39%) | 350 | 350 | 192 / 1,465 |
| T6 no palette pruning (`palette_support=0`) | 511/2,007 (25.46%) | 85.39% (80.88/90.43) | 1,920 (95.67%) | 1,920 (95.67%) | 788 (39.26%) | 470 (23.42%) | 519 | 511 | 192 / 1,304 |

The default support rounds reduce the 726-state canonical palette to six states on both fixtures. On T2, palette-to-support loses 37 exact visible occurrences; cell-candidate-to-raw-winner loses 102 state matches. Smoothing changes 111 labels (66 gains, 45 losses; net +21). T2’s candidate set contains the right state often enough to make learned reranking a plausible first appearance-model experiment, although 68 visible references are already outside the canonical palette and 62 have no final candidate block.

On T6, palette-to-support removes 1,527 exact visible occurrences (1 geometry absence, 1,526 states pruned); this is the dominant default-path loss. Disabling only that pruning restores 1,527 global state availabilities and improves the final score by 161 cells (+8.02 percentage points), with unchanged visible geometry metrics and nearly unchanged reconstruction time (419.8 s default, 417.6 s ablation). It does not approach the 80% gate: only 788/2,007 (39.26%) retain the correct state in the per-cell candidate set, and 318 of those are missed by the raw winner. T6 therefore needs better local candidate generation/recall as well as ranking; simply keeping every palette state is not an adequate fix.

Checkpoints are not a monotone loss funnel. T2 smoothing loses 45 but gains 66; the T6 ablation smoothing loses 10 but gains 59, then placement omits eight blocks. Keep geometry misses (no placed block) separate from matched wrong states.

## Reproduction and artifacts

Commands used PowerShell, `.venv\Scripts\python.exe`, `PYTHONPATH=src`, supplied-camera mode, local `%APPDATA%\.minecraft\versions\1.21.5\1.21.5.jar`, all fixture-pinned input paths, `--fixture-role template-{2,6}`, each reviewed per-frame JSON mask spec, and `--stage-trace` plus `--report`. The T6 ablation added `--palette-support-override 0.0`; its acceptance gate is ineligible by design.

| Run | Report | Trace | Candidate Litematic |
|---|---|---|---|
| T2 default | `.tmp-stagewise-pinned-template-2-report.json` — SHA-256 `EC86FB6623FB605AEC3C0C124A0EA2D16F61713CC39B9C7CA35328B89F7B55C4` | `.tmp-stagewise-pinned-template-2-trace.json` — `870E087D0E4FA67BAA02FF051FA7CCB95D205070ADB2027A4B8BF1D09E8EFEE9` | `.tmp-stagewise-pinned-template-2-output.litematic` — `24C091C50FEE1D10AAA918FDA7B30908B0DD0380F3E7FCADD2543F470E3FA7B1` |
| T6 default | `.tmp-stagewise-pinned-template-6-report.json` — `2FB45F281038D3A8911E9B55B5413145AF23465815A7F4BB34F3B0B6CBDD5422` | `.tmp-stagewise-pinned-template-6-trace.json` — `D43A5D5136B76CC67E5B035E548D5D229E538B5F3CF4B4FFAF53A563D05D581C` | `.tmp-stagewise-pinned-template-6-output.litematic` — `4031BF9D854981361F0B95E5080F55FF3801CFC0345691E80B32DD5D0B574692` |
| T6 no-prune | `.tmp-stagewise-t6-no-prune-report.json` — `7B8ECEE749A0977FAAC3CE1E2BC502B0B451FCDA5BF5913C5ECD20F816B9BD58` | `.tmp-stagewise-t6-no-prune-trace.json` — `5099C72998CA3211A7A7733CC8781A08569D50DE119F41A984D62F42E20CF199` | `.tmp-stagewise-t6-no-prune-output.litematic` — `BD521592E1004D494BF5495F1D04C550E11E58BA93E5259E613897F17946BF17` |

## Next experiment

Train a local vanilla face-patch model as an auxiliary candidate proposer/ranker. First establish factory support for exact visible-face/block labels; split by source build, not by screenshot/view. Evaluate the model on oracle-visible crops, then on the reconstructor’s actual sampled patches. Report exact-state top-1, top-k candidate recall, calibration/abstention, and end-to-end pinned visible metrics separately. T2 tests reranking where truth is already in the candidate set; T6 must measure whether the model can raise per-cell recall beyond 39.26% without relying on global support pruning. Do not treat requested shader/resource-pack metadata as proof a pack was active. Keep camera/pose/bounds recovery as a separate upstream image-only gate.
