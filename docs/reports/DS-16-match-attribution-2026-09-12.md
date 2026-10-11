# DS-16 — Reference-Assisted Attribution of DS-14 Joint-Match Failures

- Date: 2026-09-12 · Author: DeepSeek DS-16-report · Status: evaluation-only; every label below is reference-assisted and must never enter a production path.
- Directive: `AGENT_COMMS.md` 2026-09-12T11:53:00-05:00 (GPT-6 Astra → DeepSeek worker B, cc Luna). An earlier DS-16 (`scripts/attribute_joint_matches.py`) was retired before any owned write (11:45/11:47 entries); only the restarted scope is reported.
- Owned artifacts read, never rerun: `scripts/diagnose_match_attribution.py`; `exports/diagnostics/ds16/{match_candidates.json, match_candidates.npz, match_attribution.json, match_attribution.csv, attribute_run.log, attribute_run_corrected.log}`.
- Authoritative result: `exports/diagnostics/ds16/match_attribution.json`, sha256 `72da240a1992b31640ddbfb4e4bd0b6fb6c1b02e4ede1abd6a7507b74ea7a038` (re-verified read-only by `Get-FileHash`; also printed by the corrected log). Superseded earlier write `a048518b129e7b4dc24cc58f0b96b52b12e338e728ef99ef6c58082d1871918d`; both logs carry identical console summaries (attribution `{yaw_baseline_scoring 3, descriptor_candidate_recall 6, insufficient_overlap 28}`, cells within-5° `{0, 0, 6, 6}`), so the corrected run changed bytes, not outcome.

## 1. Provenance: blind freeze vs reference attribution

Phase 1 (`blind_candidate_extraction`, 2026-09-12T11:58:24-0500, recipe `--freeze-only`) opened only the 15 frozen screenshots, derived GUI usable masks and the image-derived M3 calibration artifacts. It re-extracted DS-14 reciprocal top-k candidates and predictions with the exact DS-14 settings and froze them:

| Frozen phase-1 artifact | sha256 |
|---|---|
| `match_candidates.json` | `e062b985fd10ce0ff91b69d4ea2b8047977137f3aa686f63223f645237e02d93` |
| `match_candidates.npz` | `3f7b4555b45ae848960f9f931e9984b1e888ed7083566f49467d37b8982a8ed4` |
| canonical predictions digest (37 pairs × 5 fields) | `2887755e1f6cb6f00042da3fe20c0a6d5f131eae7623848f32f9e29cb3cba8ea` |
| DS-15 cross-check (`joint_estimates.json`, sha256 `40fe2880c90f9afc2dda8a9d0d935ad97b7bda8078e5a6ae17644991679ed018`) | 37 pairs compared, **0 mismatches** |

Phase 2 (`reference_assisted_attribution`, 2026-09-12T12:19:53-0500, recipe `--attribute-only`) opened reference data only after that freeze was digested: litematic/manifest/scene sha256 for `templates 2` (`519034a6…`, `4f743e4f…`, `9620608a…`), `templates 6` (`874b1980…`, `836ad62f…`, `0a552818…`), `templates 3` (`671973c3…`, `92dc88ec…`, `098f1e79…`). Extraction, thresholds and the DS-14/DS-15 artifacts were untouched; source `c6dba2cf4610876db7d24361eb13fc9ea3ceca0c` (`dirty=True`), Python 3.10.11 / numpy 2.2.6 / OpenCV 5.0.0.

Frozen matcher/gates used identically in both phases: `neighbors=5`, `max_candidate_edges=8192`, `max_features=4000`, `ratio_test=0.75`, `graph_distance_ceiling=null`; `PairGates(min_correspondences=8, min_inliers=8, min_inlier_fraction=0.5, inlier_residual=1.0°, min_parallax=3.0°, min_cheirality_fraction=0.75, min_spatial_bins=4)`. Preregistered oracle constants: distance ≤ **0.5 blocks**, face-normal cosine ≥ **cos 30°**, epipolar necessary ≤ **1.0°**, render scale **0.35**, alignment median ≤ 10 px. Pair universe: all C(n,2) — T2 3 + T6 6 + t3-sparse-8 28 = **37 pairs**, 15 frames.

## 2. The 2×2 candidate cells are DS-11 on fixed edge sets — not DS-14 end-to-end

The directive sketched four cells as candidates/oracle-edges × current/reference cameras. The implementation runs all four through the public DS-11 `solve_pair` adjudicator on **fixed edge sets** (all reciprocal candidate edges, or all oracle-consistent keypoint pairs). It is therefore *not* DS-14's pipeline: DS-14 selects one-to-one matches and scores yaw/baseline hypotheses. The frozen DS-14 end-to-end result is reported separately. This distinction must not be collapsed.

| Cell (DS-11 `solve_pair`, frozen edge set) | evaluated (≥8 matches) | solved | within 5° | reason counts |
|---|---|---|---|---|
| `candidates_current` (57832 reciprocal candidate edges / M3 cameras) | 37 | **0** | 0 | `low_inlier_fraction` ×37 |
| `candidates_reference` (same edges / manifest cameras) | 37 | **0** | 0 | `low_inlier_fraction` ×37 |
| `oracle_current` (all oracle-consistent pairs / M3 cameras) | 9 | 6 | 6 (precision 1.0) | `insufficient_correspondences` ×28, `ambiguous_yaw_class` ×2, `cheirality_failed` ×1 |
| `oracle_reference` (same edges / manifest cameras) | 9 | 6 | 6 (precision 1.0) | identical counts |

- `oracle_current` median rotation error **0.1498°**, median gauge-aligned baseline error **0.1383°**; `oracle_reference` median rotation error **0.0°**, median baseline error **0.1659°**. Current M3 cameras match reference cameras on every oracle cell outcome → the ladder's `camera_calibration` branch (requires `oracle_reference` success *and* `oracle_current` failure) was unreachable; **0 camera-calibration attributions is absence of discriminative evidence, not proof of perfect calibration.**
- The candidate column is degenerate: 156/57815 hit candidate edges are geometric-true (**0.270%**), far below the 0.5 inlier-fraction gate, so both candidate cells are rejected for all 37 pairs by construction. A 0/37 `candidates_reference` outcome therefore cannot be read as measured "candidate recall".
- **DS-14 end-to-end (frozen, independently reproduced by phase 1):** solved **8/37**, class correct **4**, i.e. 4 false solves; 29 refusals. Per set: T2 2 solved/2 correct, T6 3 solved/2 correct, t3-sparse-8 3 solved/**0** correct. Re-running `match_feature_frames` with reference cameras solved **5/37** (`ds14_end_to_end.reference_camera_solved`). Note the reference-camera run labels classes in the reference gauge (expected class 0), so its class accuracy is not comparable to `reference_yaw_class`; only the solved count is reported.

## 3. Correspondence evidence (all 37 pairs, aggregate)

- **True-edge prevalence:** 156 geometric-true of 57815 hit candidate edges = **0.270%** (T2 52/5323 = 0.98%; T6 104/23675 = 0.44%; t3-sparse-8 0/28834 = 0.00%). Strict (same block **and** face) 3; coordinate-only 4. 17 edges had no usable hit and are unscored.
- **Recall** against the many-to-many oracle truth set (60000 truths total): directed top-k **0.005417**, reciprocal **0.002600**, selected **0.000450**. Selected matches 3649, selected-true 27, precision **0.007399**.
- **One-to-one loss (measured, never attributed):** 133 of the 156 geometric-true reciprocal edges were dropped by DS-14's one-to-one selection (**85.3%**). The `one_to_one_selection` label was unreachable because its branch requires `candidates_reference.within_5deg`, false for all 37.
- **Manifest-pose control (non-tautological):** epipolar ≤1.0° **and** cheirality true for 741 hit edges vs 558 for the seed-0 shuffled-endpoint control (**+183**; 1.28% vs 0.97% of hits). The excess is concentrated in T2/T6 (559 vs 393, **+42%**) and is only +17 in t3-sparse-8 (182 vs 165, **+10%**).
- **Descriptor distance** (L2 on raw upright SIFT descriptors): true mean 228.70 (n=156, range 76.73–373.40); false mean 288.87 (n=57659, range 37.82–467.31). True edges are lower on average but heavily overlap; in 3 of the 8 pairs with true edges (2, 4, 8) the true-edge median *exceeds* the false-edge median.
- **Spatial distribution:** true edges occupy 21 left / 26 right of the 4×4 bins aggregated over the 9 evaluable pairs, vs 342/333 for false edges — true correspondences are confined to a small image region, consistent with small usable overlap.

## 4. Per-set and per-pair attribution

| Set | pairs | candidate edges | geometric true | oracle truths | DS-14 solved / correct | `oracle_reference` within 5° | attribution |
|---|---|---|---|---|---|---|---|
| T2 | 3 | 5323 | 52 | 20845 | 2 / 2 | 1 | yaw_baseline_scoring 2, descriptor_candidate_recall 1 |
| T6 | 6 | 23675 | 104 | 39155 | 3 / 2 | 5 | descriptor_candidate_recall 5, yaw_baseline_scoring 1 |
| t3-sparse-8 | 28 | 28834 | 0 | 0 | 3 / 0 | 0 | insufficient_overlap 28 |

Preregistered total: `descriptor_candidate_recall` **6** (pairs 2, 3, 4, 6, 7, 8), `yaw_baseline_scoring` **3** (pairs 0, 1, 5), `insufficient_overlap` **28** (all t3-sparse-8); `camera_calibration` and `one_to_one_selection` **0**.

- **6 oracle-success pairs (2, 3, 4, 6, 7, 8)** are the only pairs where the reference edge set solves under both camera regimes; DS-14 end-to-end on them solved 4, got 3 classes right (2, 4, 7), mis-classed pair 6 (class 3 vs reference 1), and refused pairs 3 and 8. Their reciprocal graphs retain only 1, 0, 7, 27, 44, 11 of 820–8839 oracle truths. The `descriptor_candidate_recall` label is assigned by the ladder exactly when `oracle_reference` succeeds and `candidates_reference` fails; because the latter fails for all 37 pairs, that label carries no per-pair discrimination. The quantitative support for a recall/selection bottleneck is §3, not the 0/37 cell.
- **3 yaw/baseline pairs (0, 1, 5)** fail `oracle_current` *and* `oracle_reference` with ≥408 oracle truths each — `ambiguous_yaw_class` (0, 5) and `cheirality_failed` (1). Perfect correspondences plus true cameras still do not yield a pose, so the residual failure is in adjudication/gating, not in data.
- **28 t3-sparse-8 pairs** have **0 oracle truths and 0 geometric-true candidate edges**, so the `<8 oracle_truths` branch assigns `insufficient_overlap` by construction. This label describes oracle coverage, not image overlap.

Correct-vs-wrong yaw support among DS-14's 8 solves (winner support / ambiguity margin / true-class support):

| Solve | pair | chosen class | true class | margin | true-class support | verdict |
|---|---|---|---|---|---|---|
| correct | 0 | 3 | 3 | 3 | 106 (winner) | thin margin separates class 3 (106) from class 1 (103) |
| correct | 2 | 2 | 2 | 28 | 142 (winner) | – |
| correct | 4 | 3 | 3 | 3 | 152 (winner) | runner-up class 0 at 149 |
| correct | 7 | 3 | 3 | 37 | 248 (winner) | – |
| false | 6 | 3 | 1 | 7 | 218 | chosen 243 > true 218 |
| false | 15 | 2 | 1 | 7 | 87 | chosen 108 > true 87 |
| false | 18 | 3 | 2 | 3 | 53 | chosen 56 > true 53 |
| false | 34 | 3 | 0 | 2 | 20 | chosen 23 > true 20 |

Support margins do not separate correct from incorrect solves (correct margins 3, 28, 3, 37; false 7, 7, 3, 2), and the dominant refusal mechanism across hypotheses is the 0.75 cheirality gate (`cheirality_failed` dominates the per-pair hypothesis lists).

## 5. Limitations that bound every claim above

- **Oracle cap / effective evaluation cap.** The `<8 oracle_truths → insufficient_overlap` rule removes 28/37 pairs (76%) from attribution, and the oracle-success ceiling itself is only 6/9 evaluable pairs (6/37 of all pairs). Nothing here explains the other 31 pairs, and no pair with zero oracle truths can be attributed at all.
- **Oracle set is uncapped and many-to-many.** Truth = *every* (left, right) keypoint pair within 0.5 blocks with compatible normals, so dense regions contribute N_L×N_R truths (per-pair counts 408–19617, total 60000; the round total is arithmetic, not a clamp — `_truth_edges` applies no cap). Recall denominators are inflated by redundant combinations that no top-k matcher could be expected to select, so the reported recalls are pessimistic and should be recomputed per left keypoint ("best-of") before any tuning decision.
- **Tolerance is a single frozen point, never swept.** 0.5 blocks and cos 30° were fixed before the run; all "true/false" and zero-overlap statements are conditional on them.
- **Render scale 0.35 quantises geometry.** Ownership/depth are sampled at 35% resolution by nearest pixel (`rint`), so world hits carry ≈0.5 render-px lateral error (≈0.5/0.35 ≈ 1.4 full-resolution px) and the block coordinate is quantised to the render grid; effective block tolerance also depends on depth. The 10 px alignment gate passed for all 15 frames (hit fraction 0.997–1.000, per-frame median round-trip error 1.081–1.193 px, p90 1.529–1.659 px, `within_2px` ≥ 0.997), but that check re-projects through the *same* manifest camera and is therefore **tautological with respect to camera-vs-photo registration**; its ~1.15 px median is of the order of the 0.35-scale quantisation floor, so it cannot detect a few-pixel registration error.
- **Chance control is a single sample.** One fixed permutation (rng seed 0), no repeats, variance or interval; the epipolar test is a loose necessary condition (1.28% of hit edges pass with cheirality); and the true count is masked to hit edges while the chance count is not (17-edge asymmetry). Treat +183 (and the +17 for t3) as weak.
- **Priority ladder is conjunctive.** Unreachable branches (see §2 and §3) mean "0 camera_calibration" and "0 one_to_one_selection" are no-evidence findings; the large measured one-to-one loss (133/156) is real but deliberately unattributed by the frozen ladder.
- **T3 misalignment is not claimed.** All 28 t3-sparse-8 pairs show zero oracle truths and zero geometric-true edges, while the manifest-pose control shows only a weak positive excess (182 vs 165). Zero sub-0.5-block agreement is equally compatible with renderer/scale or scene-transform quantisation at 0.35, with the 0.5-block gate under sparse keypoints (83–1722 per frame), and with genuinely small overlap; the self-consistency check cannot distinguish these. This report neither asserts nor excludes a templates-3 registration defect; that requires the bounded sweep below.
- **Scope:** all of the above is DS-16 evaluation over frozen T2/T6/t3-sparse-8 acceptance fixtures (never training data). No edit was made outside this report; no test, rerun, delegation, stage, commit or self-chain was performed.

## 6. Bounded next experiment (DS-16B, evaluation-only, preregistered)

Reload only the frozen `match_candidates.npz` + `match_candidates.json` (digests above) and rerun attribution on the **9 pairs with ≥8 oracle truths**, reporting two fixed read-outs with the same DS-11 gates: (a) oracle-geometry sweep distance ∈ {0.25, 0.5, 1.0, 2.0} blocks × normal ∈ {cos 30°, cos 45°, cos 60°} at render scale 0.35 (12 preregistered cells) to test whether the 0.270% prevalence and the t3 zero-overlap are tolerance artefacts; (b) per-left-keypoint best-of recall beside the current many-to-many recall, plus one `solve_pair` cell on the intersection of the reciprocal graph with the oracle truth set, to separate residual recall loss from one-to-one selection and the 0.5 inlier/0.75 cheirality gates. Stop rule: if the 9-pair oracle ceiling stays ≤8/9 solved or best-of recall stays <5%, conclude the correspondence representation (descriptor/ratio/top-k), not the pose adjudicator, is binding — and do not tune any threshold. Do not extend to t3 misalignment claims until the sweep reports. No re-detection, no production/test/threshold edits.
