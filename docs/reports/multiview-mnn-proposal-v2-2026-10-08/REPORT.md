# Additive v2 paired proposal screen: bounded synthetic fixtures and development-only calibration

This v2 diagnostic is additive to the frozen v1 screen (`docs/reports/multiview-mnn-proposal-2026-10-08/REPORT.md`, whose terminal held-out `SIFT feature budget exceeded` refusal and receipts are immutable). No v1 file, pin, artifact or run record was edited. The new module copies v1's pure policy, witness, geometry-adapter, partition, truth-isolation, deterministic-order and atomic-publication helpers, and changes only the report schema/pins, the independent scene-group seeds and fixture generator, and the calibration entry point.

## Frozen basis and pins

- v1 source `scripts/evaluation/diagnose_multiview_mnn_proposals.py` SHA256 `6b4437e60a0739673ddb6b70574b03d63866af9cbae6607b8a639b020423ca40` (37238 bytes) and v1 tests `tests/test_diagnose_multiview_mnn_proposals.py` SHA256 `7765d1d99bb61c57a2f14b30726cb11dddff142db10e10bec009f85c17ca76f0` (9204 bytes) are unchanged and recorded in the v2 configuration.
- Shared extractor `scripts/evaluation/diagnose_multiview_matcher_ablation.py` SHA256 `8907fa39bbdf7e2b04960fa8c5565757dc94227502c6272038823cc838d8b02e`; low-contrast SIFT variant `sift-low-wide` (`contrastThreshold .01`, requested 668 features) is used unchanged.
- Current geometry `scripts/evaluation/diagnose_multiview_plane_tracks.py` SHA256 `f69be2ef78e8ae90a2bad09f973ed1ce3ec37d1a3a99f77101ecb68f9b2291b0` is pinned in the v2 configuration and validated on every entry point. v1's stale `97ea…` pin is not reused. The nested geometry report stays `projective-plane-tracks-v4` with unchanged geometry calls, RANSAC/fit-held selection, track and coverage functions.
- v2 schema is `opencv-mnn-proposals-v2`; the development calibration schema is `opencv-mnn-synthetic-calibration-v2`. Config SHA256 is `8f0cb5ed19ed1a382aa3290bf9e1571384e5a4a70692536d714f61ada46f3fe9`.

## Preserved policy semantics

Arm A is the accepted reciprocal Lowe .85 policy; Arm B is strict mutual nearest, requiring each first L2 distance to be strictly less than its second, with exact nearest-distance ties abstaining. Both arms share one two-neighbor query per direction per pair, receive the same ordered extracted features, and pass through the same adapter-checked geometry ledger. Exact nearest ties abstain and are counted. No truth argument reaches detection, matching, fit/held partitioning, geometry or acceptance; the oracle is evaluator-only. The hard 700-feature refusal is re-asserted in a v2 extraction wrapper and again at every per-frame count readout, with no truncation or candidate top-up anywhere. Per-image/pair/work/artifact limits (`MAX_FEATURES 700`, `MAX_DESCRIPTOR_COMPARISONS 12000000`, `MAX_CYCLE_COMBINATIONS 1000`, `MAX_SHARED_COMPARISONS 188160000`, `MAX_ARTIFACT_BYTES 96000000`) are unchanged.

## Fixture resource repair

The v1 fixture generated texture-heavy families that reached ~688 detected features on development seeds and crossed 700 on the frozen held-out panel, producing the v1 terminal refusal. v2 uses entirely new independent scene-group seeds (development 83011/83021/83031/83041, held-out 93011/93021/93031/93041; none reused from v1) and a development-calibrated, complexity-bounded generator: fewer low-contrast strokes, 48px tiles with heavier blur and explicit brick borders, 12px source upsampled for the mixed-scale family, a dedicated independently permuted 48px tile for `incompatible-tiles`, and noise perturbation reduced from ±8 to ±4. Transforms, case kinds, gates, extractor, geometry and caps are unchanged.

Detected and retained counts are recorded for all 96 development frames in `development.json` (`frame_feature_counts` per case, plus per-pair `detected_feature_counts`/`retained_feature_counts`). Retained equals detected in every frame. Per-family detected range:

| Family | min detected | max detected |
| --- | --- | --- |
| weak-contrast | 66 | 325 |
| repeated-brick-tile | 58 | 166 |
| mixed-scale-pixel | 34 | 154 |
| sparse-corners | 72 | 238 |

Panel maximum is 325 of 700; no development frame refused. A development frame above 700 raises inside extraction, or in the panel loop before any match-policy metric is computed or published.

## Fresh 32-case development calibration

One run, `.venv\Scripts\python.exe -B -c "… calibrate_synthetic_dev(); persist_calibration(…)"`, completed 32/32 development cases in 2.709 s (2026-10-08T20:41:01Z). `development.json` is 248700 bytes, SHA256 `8135104eacd2dcb1109733f992c817fe89c6c993544dbdae4a73011df29c7b91`, and replays byte-identically through `canonical_calibration_bytes`. Actual shared descriptor comparisons 2 961 086 of the 188 160 000 budget. `evaluation_split` is `dev` and `heldout_run` is false; the held-out groups stay declared and disjoint but were never generated.

Pooled development aggregates (Arm B strict mutual nearest vs Arm A reciprocal Lowe .85): Arm B true-proposal recall 0.6324 vs 0.6026, partition-eligible positive pairs 30 vs 25, accepted support precision 0.9994 vs 0.9993, accepted pairs 1565 vs 1488, correct-geometry pairs 20 vs 19, and zero accepted pairs on the 8 pure-negative cases for both arms. Only 4 of 16 positive cases produced a coherent plane track per arm.

Configured gate support on development:

| Gate | Development support |
| --- | --- |
| `family_positive_and_precision` | **false** (weak-contrast 1/4, repeated-brick-tile 0/4, mixed-scale-pixel 0/4, sparse-corners 3/4) |
| `pooled_precision` | true |
| `pure_negative_zero` | true |
| `structural_invalid_zero` | **false** (sparse-corners `multi-plane-corner` yields 1 invalid cross-surface track per arm) |
| `no_a_positive_loss` | true |
| `strict_recall_gain` | true |
| `strict_partition_gain` | true |
| `correct_geometry_pair` | true |
| `a_not_saturated` | true |

Recorded decision: **no-go** on development support only (`inconclusive` is not claimed because Arm A is not saturated). The reduced-complexity fixtures are resource-feasible but do not carry per-family positive support; this is a development finding, not held-out evidence, and no threshold, gate, extractor or geometry value was changed in response. Development results do not establish held-out generalization or authorize threshold changes.

## Checks performed

- Focused v2 suite: `.\.venv\Scripts\python.exe -B -m pytest tests/test_diagnose_multiview_mnn_proposals_v2.py -q -s --override-ini addopts= --tb=short` → 23 passed, exit 0, 1.02 s.
- Covered: byte-identical Arm A geometry to the accepted low-wide variant; deterministic shared queries, witnesses and partition identities; strict-tie abstention vs ratio policy distinctness; resealed policy-id/tie-rule/adapter/tie-count/nearest-identity/survivor-count/proposal-identity/geometry-identity/frame-dimension/shared-work mutation refusal before publication; truth isolation and pin-before-extraction refusal (ablation and geometry pins); per-policy and shared work refusal; output-cap refusal and publication-failure cleanup; exact 700-feature boundary accepted and 701 refused before nearest; development case above 700 failing before match-policy metrics; dev-only calibration with recorded per-frame counts, atomic no-clobber publication, injected link-failure cleanup, and resealed over-cap/held-out split refusal; frozen 8-group schedule with disjoint dev/held-out groups and `synthetic_cases("heldout")` refusal; current geometry pin vs stale v1 pin and frozen v1 basis hashes.
- Syntax compiled without emitting files; full-file trailing whitespace 0 lines in both v2 files; persisted `development.json` replays byte-identically.

## Scope and stop

The only run performed is focused v2 tests plus the 32-case development calibration. v1 was not rerun, and no v2 held-out case, ordinary image, capture/client, truth, training, download or production integration was touched. Authority stays with Astra exact review and lead assessment before any separately authorized v2 held-out task.
