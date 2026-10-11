# DS-14 — Corrected joint matcher and frozen real-pair experiment

DeepSeek CLI DS-14, 2026-09-12. Owned paths only: new
`src/mc_photogrammetry/pipeline/joint_matching.py`, new `tests/test_joint_matching.py`,
new `scripts/diagnose_joint_pairs.py`, the DS-13 correction section, new
`docs/reports/DS-14-joint-matching-2026-09-12.md`, generated
`exports/diagnostics/ds14/`, plus the additive correspondence.py/test changes
allowed by the assignment. `connected_camera.py`, DS-12 defaults and frozen
artifacts, residual/raycast/shape, schema, CLI/desktop, benchmark, annotations and
every Luna path are untouched. No dependency added; nothing staged or committed;
no full suite, no benchmark. Revision: HEAD `c6dba2c` plus the accepted
DS-3/4F/5/7/9/10/11/12/13-DS14 working-tree edits (`git status` dirty, as before).

## 1. What was built

One module, one public pair entry point plus an injectable feature seam:

* `match_and_solve_pair(left_id, left_rgb, right_id, right_rgb, *, left_camera,
  right_camera, usable_masks=None, settings=None, matcher=None, gates=None)` —
  detects features (blind: pixels + authoritative usable mask only), builds the
  reciprocal top-k graph, adjudicates it geometrically, and returns a
  `JointPairResult` whose `pose` is DS-11's own `PairPose`.
* `match_feature_frames(left, right, left_camera, right_camera, *, matcher, gates)` —
  the injectable seam: tests supply `FrameFeatures` (keypoints + descriptors)
  directly, so the geometric stage is exercised without SIFT.

`JointMatchSettings` carries every declared bound; `HypothesisScore` reports each
(class, baseline mode) candidate; `JOINT_REFUSAL_REASONS` is module-local (never
appended to `connected_camera.REFUSAL_REASONS`). Additive DS-12 surface:
`reciprocal_topk_graph`, `boundary_distances`, `descriptor_footprint_keep`,
`FrameFeatures.sizes` / `FrameFeatures.boundary_dropped` (defaults `None`/`0`), and
`CorrespondenceSettings.mask_boundary_pad_px` (default `0.0`, the frozen path).
No existing DS-12 test was modified; three additive controls were appended.

## 2. The corrected objective, point by point

DS-13 §9 superseded DS-13 §4's objective. Implementation map:

| Correction | Code |
| --- | --- |
| No all-edge median gate | Winner gates read only selected support; `median_all_edge_residual_degrees` is reported per hypothesis and used nowhere |
| Support floor on `min(raw keypoints)` | `support_floor = max(min_support_absolute, ceil(0.02 · min(raw_left, raw_right)))`, computed after the boundary filter |
| 2 endpoint-distinct edges | `_class_modes` samples 2 edges with distinct left **and** distinct right indices, rejects `s1/s0 < min_singular_ratio` samples |
| Explicit ambiguity margin | `ambiguity_support_margin = 2`: the winner must beat every floor-meeting competitor (other class **or** other baseline mode) by ≥ 2 selected edges, else `ambiguous_hypothesis` |
| Rank by support first | Sort key `(-support, -min(bins), median selected residual, descriptor cost, class, mode)` |
| Refit before freezing, public DS-11 check | `_select_and_refit` SVD-refits `t` on the greedy support to a fixed point; the final selected set goes to public `solve_pair`, whose class must match (`selection_unstable` otherwise) |

## 3. Algorithm

**Graph.** `reciprocal_topk_graph(k=5, |E| ≤ 8192, distance_ceiling=None)` takes
each keypoint's 5 nearest descriptors in both directions; an edge survives only
when both endpoints agree. Edges are sorted by `(distance, left, right)` and
truncated to the cap, so truncation drops the least-trustworthy edges first and is
reported as `truncated` (this run: 0 for all six pairs, max 4186 edges).

**Mask-boundary descriptors.** With authoritative `U`, the detection domain is
`{p : dist(p, ~U) > pad + 0.5·size}` evaluated on `cv2.distanceTransform(U)`.
`pad = 8.0` in the matcher, `0.0` (short-circuit no-op) in DS-12's defaults. The
filter only shrinks `U`; excluded-RGB zeroing and the missing/wrong-shape mask
semantics are unchanged.

**Per class `k∈{0..3}`, `R_k = R_r Ry(90k) R_lᵀ`.** Rows `a = (R_k u_l) × u_r`.
The all-edge SVD null vector is one candidate (DS-11's rule); then 512 seeded
2-edge samples. Each candidate scores **every** edge by `ρ = asin(|a·t|/‖a‖)`;
candidates are ordered by `(inlier count, inlier median, draw order)` and the best
few angularly distinct modes (≥15°, ≤3 per class) reaching the support floor are
kept. Each kept mode is greedily selected — edges ordered by `(ρ, descriptor
distance, left index, right index)`, accepted only when neither endpoint is used —
then SVD-refitted on the selected support and re-selected to a fixed point (≤10
iterations). No all-edge statistic gates.

**Gates.** Ranked winner must clear: support ≥ floor; ≥4 of 16 bins on **both**
frames; `s1/s0 ≥ 1e-3`; median parallax ≥ 3°; cheirality ≥ 0.75; and the ≥2-edge
ambiguity margin over the best floor-meeting competitor. Then `solve_pair` is the
compatibility check: a refusal is `solve_pair_refused`, a class mismatch is
`selection_unstable`, otherwise the result is solved with DS-11's pose.

## 4. Controls (`tests/test_joint_matching.py`, 10 tests)

| Control | What it pins |
| --- | --- |
| `test_exact_scene_solves_through_public_ds11_pose` | class 1, rotation ≤1°, baseline ≤1°, support/bin/parallax/cheirality/margin gates, one-to-one, DS-11 pose class equality, `truncated == 0` |
| `test_selected_support_is_one_to_one_and_deterministic` | two runs bit-identical matches, class, support and per-hypothesis supports |
| `test_unequal_explicit_intrinsics_still_solve` | 1280×720 vs 1024×768 with different focals/principal points, class 1, rotation/baseline ≤1° |
| `test_repeated_descriptors_plus_unique_minority_solves_correct_class` | 40 near-identical descriptors on random unrelated pixels plus 20 consistent unique points: solves class 1 from the selected support (the all-edge median is diagnostic only) |
| `test_two_competing_baselines_refuse_ambiguous` | same yaw class, two distinct baselines with equal support: `ambiguous_hypothesis`, no vector |
| `test_identity_self_match_refuses` | identical pixels/descriptors both sides: typed refusal, no vector |
| `test_insufficient_support_refuses_with_typed_reason` | 5 points: support < floor, typed refusal, no vector |
| `test_edge_cap_truncates_and_is_reported` | `max_candidate_edges=8`: `candidate_edges == 8`, `truncated == total − 8` |
| `test_boundary_footprint_filter_and_excluded_rgb_invariance` | `boundary_dropped > 0`; mutating RGB outside `U` to 255/0 leaves matches, edges, drops, keypoints and status identical |
| `test_default_detection_keeps_frozen_ds12_boundary_behaviour` | default features report `boundary_dropped == (0, 0)` |

Additive `tests/test_correspondence.py` controls: exact reciprocal edge set /
cap / ceiling / determinism for `reciprocal_topk_graph`; `descriptor_footprint_keep`
rejects only boundary footprints and `pad_px=0` is the frozen no-op;
`detect_features` reports `boundary_dropped == 0` by default and `> 0` only when
`mask_boundary_pad_px > 0`, with every survivor inside `U`.

## 5. Frozen real-pair diagnostic — provenance and phase split

`scripts/diagnose_joint_pairs.py` repeats DS-12's exact six frozen pairs by
importing DS-12's `PAIR_SPECS` verbatim, with the same image-derived M3 calibration
artifacts and `usable_masks=None`. Phase 1 (`--extract-only`) writes
`joint_estimates.json` + `.csv` and opens no manifest; `load_camera_manifest` is
imported inside `run_evaluation` only. Phase 2 (`--evaluate-only`) loads the frozen
file, records its SHA-256, then opens `cameras.json`. The run order was
`--extract-only` (twice) → `--evaluate-only`; no matcher/solver/correspondence
parameter was changed after the reference was read.

Settings of record (echoed into the artifact): `neighbors=5`,
`max_candidate_edges=8192`, `max_features=4000`, `mask_boundary_pad_px=8.0`,
`ransac_seed=0`, `ransac_iterations=512`, `inlier_residual_degrees=1.0`,
`refit_iterations=10`, `min_support_absolute=8`, `min_support_fraction=0.02`,
`spatial_grid=(4,4)`, `min_spatial_bins=4`, `min_parallax_degrees=3.0`,
`min_cheirality_fraction=0.75`, `min_singular_ratio=1e-3`,
`ambiguity_support_margin=2`, `mode_separation_degrees=15.0`,
`max_modes_per_class=3`; DS-11 `PairGates()` defaults.

Environment: Python 3.10.11, NumPy 2.2.6, OpenCV 5.0.0; git `c6dba2c`, dirty.
Calibration artifacts `435c19bb…`, `a24751ab…`, `bbbd3a0d…`; manifests
`4f743e4f…`, `836ad62f…`, `92dc88ec…` (opened only in phase 2).

**Determinism.** Two independent phase-1 runs produced the identical
`canonical_pairs_digest`
`dff6ab40258601b6b4c06a6b78cbb7e103edb3d74505e83ddf93dbfb105dd723` (outcome
payload with wall-clock fields stripped); the second run is kept under
`exports/diagnostics/ds14/repeat_check/`. The two files differ only in the
`generated` timestamp, per-pair wall-clock and their own path field.

## 6. Phase-1 result (blind, manifest-free)

`solved 3/6`. Raw keypoints fall (97/51, 51/63, 77/75, 75/61, 52/60, 60/63) are
descriptor-footprint drops; the graph is 752–4186 reciprocal edges, none truncated.

| fixture | pair (suffix) | status | reason | raw L/R | drop L/R | edges | support/floor | bins L/R | med ρ | parallax | cheir | margin | class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T2 | 13.22.14→13.24.24 | solved | — | 1189/868 | 97/51 | 1434 | 106/18 | 6/6 | 0.35° | 120.1° | 1.00 | 3 | 3 |
| T2 | 13.24.24→13.24.33 | solved | — | 868/1505 | 51/63 | 1729 | 142/18 | 8/5 | 0.42° | 118.6° | 0.86 | 28 | 2 |
| T6 | 18.01.09→18.01.14 | refused | `cheirality_failed` | 1722/2485 | 77/75 | 3495 | 210/35 | 5/9 | 0.41° | 158.7° | 0.69 | 4 | — |
| T6 | 18.01.14→18.01.18 | solved | — | 2485/1749 | 75/61 | 4186 | 243/35 | 6/7 | 0.34° | 145.7° | 0.94 | 7 | 3 |
| T3 | 20.03.32→20.03.55 | refused | `cheirality_failed` | 921/260 | 52/60 | 899 | 79/8 | 8/7 | 0.31° | 113.4° | 0.70 | 14 | — |
| T3 | 20.03.55→20.04.18 | refused | `cheirality_failed` | 260/605 | 60/63 | 752 | 62/8 | 6/5 | 0.42° | 30.5° | 0.23 | 16 | — |

## 7. Phase-2 reference evaluation

`solved 3/6`, `class correct 2/6`, `within 5° rotation 2`, median rotation error
0.250°, median gauge-aligned baseline error 14.060°.

| fixture | pair | joint | solved class | ref class | class ok | rotation err | baseline err (gauge) | baseline err (signless) | support/floor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T2 | 13.22.14→13.24.24 | solved | 3 | 3 | **yes** | 0.222° | 9.586° | 81.094° | 106/18 |
| T2 | 13.24.24→13.24.33 | solved | 2 | 2 | **yes** | 0.250° | 14.060° | 89.019° | 142/18 |
| T6 | 18.01.09→18.01.14 | refused (`cheirality_failed`) | — | 2 | — | — | — | — | 210/35 |
| T6 | 18.01.14→18.01.18 | solved | 3 | 1 | **no** | 179.950° | 18.342° | 18.342° | 243/35 |
| T3 | 20.03.32→20.03.55 | refused (`cheirality_failed`) | — | 0 | — | — | — | — | 79/8 |
| T3 | 20.03.55→20.04.18 | refused (`cheirality_failed`) | — | 3 | — | — | — | — | 62/8 |

Reading, without retuning: the corrected objective does solve real pairs where
DS-12's funnel was empty on five of six (its sixth pair was feedable and refused by
DS-11) — but **two of the three solves are only class-consistent, not geometrically
accurate, and one is a class flip**. The two T2 classes are exact (0.22°/0.25° rotation), yet their baselines
are 9.6°/14.1° off after gauge removal and 81–89° off signless, with implausible
120°/118° parallax: those supports are the low-residual near-degenerate mode, not
a usable direction. The T6 18.01.14→18.01.18 solve has 243 selected edges over
4186 candidate edges, a 0.34° median and 145.7° parallax, and lands on the
180°-flipped class (rotation error 179.95°) — the repetition trap resurfacing at
the **class** level because RANSAC support over a mostly-false 4186-edge graph is
inflated by chance (~2.2% of edges fall in a 1° band, so >90 expected) and the
7-edge ambiguity margin does not separate the true mode. The three refusals are
honest typed `cheirality_failed` abstentions; the support floors were met, so the
gate that fired is cheirality, not support. No threshold was changed after
reading any of this.

## 8. Limits and non-claims

1. **No production promotion and no accuracy claim.** 2/6 class-correct solves on
   frozen real pairs is a diagnostic result, not an M3 gate. The M3 gate and all
   component/position/scale surfaces are untouched.
2. **Support is not discriminative at large `|E|`.** On T6b, 243/4186 selected
   edges came from chance alignment. The declared ambiguity margin (2 edges) and
   spatial/parallax/cheirality gates did not catch it; a support *fraction* or an
   upper parallax bound would be a new declared threshold, which this assignment
   forbids changing after the reference read. Recorded as the headline finding.
3. **High-parallax, low-residual solutions are not validated.** The 113–159°
   parallax winners show the 3° parallax floor is a lower bound only.
4. **Boundary filtering is a declared heuristic** (`pad + 0.5·size`, pad 8 px),
   pinned by a synthetic control, not by fixture outcomes; it dropped 51–97
   keypoints per frame and changed no DS-12 default.
5. **The all-edge median is never a gate**, so a reader cannot use it to explain a
   solve; it is reported per hypothesis for transparency only.
6. **Diagnostic scope.** Six adjacent frozen pairs, `usable_masks=None`, one
   settings set, single-run wall-clock (0.5–1.6 s/pair) on a dirty tree; not a
   benchmark and not an M6 measurement.
7. No components, no positions, no scale/`scale_known`, no schema/CLI/desktop
   wiring, no lattice/ID use, and no claim that the corrected matcher generalizes
   beyond these six pairs.

## 9. Evidence

```
.venv\Scripts\python.exe -m pytest tests/test_joint_matching.py tests/test_correspondence.py \
  tests/test_connected_camera.py tests/test_camera.py -p no:cacheprovider
52 passed in 7.85s
```
(10 new joint controls, 3 additive correspondence controls, 12 DS-11, 19 camera. No
other test file was run; no existing test was modified.)

Run order: `--extract-only` (frozen `exports/diagnostics/ds14/joint_estimates.json`
sha256 `8d424e6c02ff623264b6fb197a9adb0e21bf6695011a452726ed4f979d243898`), then a
second `--extract-only` under `repeat_check/` (same canonical digest), then
`--evaluate-only` (`joint_evaluation.json` sha256
`b52b862826f10d8af67e8411b6db0fdd5e235c18b8be79681d77f5cf9e5c1ca2`). CSVs:
`joint_estimates.csv` `cd8ae6d0…`, `joint_evaluation.csv` `6afd02fa…`. The frozen
bytes of record are the phase-1 JSON; phase 2 recorded that digest before opening
any manifest.
