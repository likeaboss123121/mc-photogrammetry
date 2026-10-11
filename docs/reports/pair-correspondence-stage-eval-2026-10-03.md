# DS-R5I / DS-R5I-R1 evaluator-only pair-correspondence stage eval (2026-10-03)

Status: **complete, stopped for lead review.** Evaluation-only. No production
matcher, threshold, solver, project or promotion decision was changed, and no
reconstruction or user metadata path was touched.

This report now carries **two provenance records**: the original DS-R5I run (its
sealed artifact preserved unchanged as `legacy`) and the DS-R5I-R1 correction run.
The DS-R5I `repeated_texture` bucket was overbroad (it fired on any two different
face identities that shared a block *name*, including different slots on the same
coordinate, without proving texture identity). Every DS-R5I
repeated-texture/appearance count is therefore **invalid** and is struck from the
conclusions; only the DS-R5I-R1 proxy counts are reported as findings. Pair
positive counts and the pooled surface precision/recall *definitions* were
unaffected by that bucket, but the truth denominator was inflated by repeated
descriptor keypoints at the same pixel and is corrected here.

## 1. Scope and owned paths

DS-R5I asked for an evaluation-only per-pair matcher scorecard over the finalized
dataset-factory exports: hash-pin every `exports/datasets/**/index.json`, group
strictly by `scene_group`, keep duplicate Canal Town exports in the same fold,
define strict positive pairs by the same world-surface point within a
preregistered tolerance, and report pair precision over returned match identities
and pair recall over the explicitly constructed truth-visible keypoint-pair
denominator.

DS-R5I-R1 asked to correct only the evaluation-only classifier/denominator and its
tests/report: rename the appearance proxy so it makes no verified-texture claim,
count it only for the same exact `BlockState` on the same face slot at a
different block coordinate, classify same-coordinate/different-slot as wrong
surface, key the truth denominator and returned identity by the same unique pixel
pair, and preserve the original sealed artifact as `legacy`.

| Path | DS-R5I sha256 | DS-R5I-R1 sha256 | Notes |
| --- | --- | --- | --- |
| `src/mc_photogrammetry/evaluation_pair_correspondence.py` | `d2a328bd8f95e8a1f5f0aa449f438596084c4ef19a705adac6e19fcbf7533510` | `bb6562a825986b1ffd62a6fbb9c63fdfa34921a69814873645e166a0fcd43a72` | evaluation module |
| `scripts/evaluation/evaluate_pair_correspondence.py` | `3a90e8d97595e170b1ad1190ea077d17c2ae6f68ab49b8e878fd128b969dd8ad` | `c2ac759a3cb5a3e5512f04c8d1dec2b74e8b4b8b5870bf15049bcacff79e7fdf` | one-shot runner |
| `tests/test_evaluation_pair_correspondence.py` | `00a71f6e74ee69e77fec0e2c90aa6d30bd6f7091d02eb12a8f492c55eb52df81` | `d683499a193655e8680ced977b2a7d0c9e08d11c565aa48e537acd2ef81c219e` | focused tests |
| `docs/reports/pair-correspondence-stage-eval-2026-10-03.md` | this file | this file | report |
| `.tmp-pair-correspondence-eval-20261003.json` | file `8dec1dcd56a8e495958ec44a52759147ec96c75caee485b67f91a3b1c08fe5bd`, internal seal `2d88a45bb8a6c2237d2ed3bdf2d3d25042da41dea40f0befab6ab020d0860932`, 95,404 bytes | unchanged (**legacy**, preserved) | original sealed output |
| `.tmp-pair-correspondence-eval-r1-20261003.json` | absent | file `7faf44c096b51085fb51477c403149ffdb2dbbf00b29d5a7f948003d2fd2784d`, internal seal `499a9c1b616eb48f573179703adf7fb0883d2ee526c190a7ae50422470179fd8`, 97,289 bytes | corrected sealed output |

The original DS-R5I artifact was not rewritten: its file hash and internal seal
still reproduce exactly. The R1 output path was confirmed absent immediately
before the run, and no scratch file, scratch directory, partial output,
junction/reparse point, background process, install/download, Minecraft/client/
capture, or system write was used.

## 2. Method

### 2.1 Truth model (evaluation-only)

* World model is the registered `ground_truth.json` region block states; air is
  absent. Block coordinates are world coordinates, as recorded by the factory.
* Camera pose is rebuilt from the recorded **actual** eye position and
  yaw/pitch in `capture.json` (`actual.eye_position`, `actual.yaw_degrees`,
  `actual.pitch_degrees`, `actual.fov_degrees`) with the planned intrinsics;
  the planned pose is only a fallback. This is the pose the client reported, not
  the one requested.
* Visibility is a per-query front-to-back axis-aligned voxel walk ("sparse
  z-buffer"). For a pixel the first occupied cell owns the surface; the last axis
  crossed is the entry face (e.g. stepping `-z` enters `south`) and the entry
  point is the world surface point. The ray is built with the production camera
  projection (`export.visibility.ray_directions`).
* **Strict positive pair:** two keypoints, one per frame, whose owned surfaces are
  the same `(block coordinate, face slot)` and whose world surface points agree
  within `SURFACE_TOLERANCE_BLOCKS = 0.0625` (one vanilla texel, 1/16 block).
  Both endpoints have truth hits, so a strict positive is jointly visible.
* **Appearance proxy (corrected, DS-R5I-R1):**
  `same_state_same_slot_other_block` requires the **same exact `BlockState`**
  (name *and* properties) on the **same face slot** at two **different block
  coordinates**. Same-coordinate/different-slot, different-slot at different
  coordinates, and same-name/different-properties are all `wrong_surface`. **No
  block texture is read, hashed or compared anywhere in the pipeline**, so this
  bucket makes no texture-identity claim and is not texture ground truth.
* **Unique pixel-pair identity (corrected, DS-R5I-R1):** the truth denominator
  and the returned identity used for recall are both the set of unique
  `(left pixel, right pixel)` pairs. Repeated descriptor keypoints at exactly the
  same pixel collapse to one identity; distinct pixels are never collapsed. The
  DS-R5I denominator counted feature-array indices instead, so repeated
  keypoints at one pixel inflated it.
* Partial-model limitation: bottom/top slabs, stairs, trapdoors, doors, fences,
  walls, panes, carpet, plants, fluids and similar models are unavailable without
  the vanilla model assets. They occlude as full cubes but are excluded from
  truth ownership, so surfaces behind them leave the denominator instead of being
  mislabelled. `_slab` with `type=double` is the one partial family kept as a full
  cube. This is recorded in the sealed artifact under `renderer_geometry`.

### 2.2 Metrics

* **Pair precision** = `positive / returned` over returned match identities.
  Returned identities come from the production
  `pipeline.correspondence.detect_features` + `select_mutual_matches`; a match
  with no truth surface at one endpoint is `unattributed` and counts against
  precision. Returned rows are deduplicated by identity before classification.
  `PlaneGraphTrace` hypothesis counts are never used.
* **Pair recall** = `truth_pairs_returned / truth_pairs`, where `truth_pairs` is
  the explicitly constructed truth-visible keypoint-pair denominator (unique
  pixel pairs on the same face within tolerance).
* Undefined denominators are `null` with a typed `undefined_reason`
  (`no_returned_matches`, `no_truth_keypoint_pairs`), never zero.
* Per-stage funnel/refusal: cached per-frame SIFT features plus the production
  mutual matcher, with the production refusal order mirrored
  (`unusable_frame`, `insufficient_keypoints`, `insufficient_matches`,
  `repetition_risk`, `ok`). Features are detected once per frame and reused.
* Sealed schema is `mc-photogrammetry-pair-correspondence-scorecard-v2`; the
  v1 schema used the invalid `repeated_texture` field.

## 3. Inputs and hash pins

`discover_finalized_entries` recomputed every listed scene-file hash and
screenshot hash before admitting an entry; all pins matched for both runs. Run
preconditions also confirmed every declared screenshot dimension equals the PNG
IHDR. The index pins are unchanged between runs:

| Dataset | `index.json` sha256 | Scene | `scene_group` | Frames |
| --- | --- | --- | --- | --- |
| `smoke-train-station-ds20c-2026-09-12` | `a2a21eb49d4b7664f603eaaca77bf8a8095defee520a2c64dc2592fb1625a3e0` | `train-station-spawn-208-96` | `1757a5ad7848…` | 4 |
| `pilot-batch-train-station-2026-09-12` | `4ef40f8be5663108036bd3827e298286c93747a71ed9bbb4dcd2a3266ecf4837` | `train-station-spawn-208-96-orbit-a` | `1757a5ad7848…` | 6 |
| `recon-canal-houses-2026-09-12` | `5d0db614b1802bd7fcd047c390dd4637eb20dcd88f550d76a624d3bf6ba6fae9` | `canal-houses-test-recon-player-spawn` | `3e9dfb4340e1…` | 4 |
| `fresh-c4-canal-town-2026-09-12` | `d3dc69b048bf8b827d75df402aa041445678ae63ba5124e6eefc6586278413dd` | `canal-town-architectural-component-bbox` | `74f080aab201…` | 4 |
| `fresh-c4-canal-town-framed-2026-09-12` | `0637caba1a523bbdf7351dedd7c4f93bfdb611f0b549fc54b3a781f3d25ee46b` | `canal-town-architectural-component-bbox` | `74f080aab201…` | 8 |

Exact independent counts: **5 finalized entries, 3 independent topology groups,
26 frames, 61 unordered frame pairs.** The R1 pre-run check re-verified all five
entries, 3 groups and 3 folds with empty split/fold leakage lists.

### 3.1 Deterministic grouped split manifest

Grouping key is strictly `scene_group`. The two Canal Town exports share
`scene_id canal-town-architectural-component-bbox` and group `74f080aab201…`
and therefore travel together. The two train-station captures share topology
group `1757a5ad7848…` under different `scene_id`s. Leave-one-group-out folds are
purely descriptive (nothing is trained):

| Fold | Held-out group | Retained groups | Split-leakage | Fold-leakage |
| --- | --- | --- | --- | --- |
| `heldout-1757a5ad7848` | `1757a5ad7848…` (2 entries) | `3e9dfb4340e1…`, `74f080aab201…` | none | none |
| `heldout-3e9dfb4340e1` | `3e9dfb4340e1…` (1 entry) | `1757a5ad7848…`, `74f080aab201…` | none | none |
| `heldout-74f080aab201` | `74f080aab201…` (2 entries) | `1757a5ad7848…`, `3e9dfb4340e1…` | none | none |

Every index split is `train`, so the index split carries no signal; leakage
checks are group/scene-id consistency plus "no group or duplicated scene on both
sides of a fold", and all are empty. No scene id maps to more than one group.

## 4. Run provenance

### 4.1 DS-R5I original (preserved as `legacy`; appearance counts invalid)

* Sources: module `d2a328bd…`, runner `3a90e8d9…`, tests `00a71f6e…`.
* Command, foreground, no retry, 19.06 s wall, exit 0:
  `$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe scripts\evaluation\evaluate_pair_correspondence.py`
* Output `.tmp-pair-correspondence-eval-20261003.json`: 95,404 bytes, file
  `8dec1dcd56a8e495958ec44a52759147ec96c75caee485b67f91a3b1c08fe5bd`, internal
  seal `2d88a45bb8a6c2237d2ed3bdf2d3d25042da41dea40f0befab6ab020d0860932`
  (independently replayed and matched). **Unchanged.**
* The legacy `repeated_texture` field (pooled 194; group `74f080aab201…` 193;
  scenes 119/74/1) and its "texture-candidate loss" conclusion are **invalid**:
  the classifier matched on block *name* across different face identities,
  including different slots on one coordinate, and never proved texture identity.

### 4.2 DS-R5I-R1 correction (authoritative)

* Sources: module `bb6562a8…`, runner `c2ac759a…`, tests `d683499a…`.
* Focused tests and `py_compile` passed before the run (section 6).
* R1 pre-run check: all five factory input index pins re-verified, 3 groups,
  empty leakage lists, new output path absent.
* Command, foreground, no retry, 19.15 s wall, exit 0:
  `$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe scripts\evaluation\evaluate_pair_correspondence.py`
* Output `.tmp-pair-correspondence-eval-r1-20261003.json`: 97,289 bytes, file
  `7faf44c096b51085fb51477c403149ffdb2dbbf00b29d5a7f948003d2fd2784d`, internal
  seal `499a9c1b616eb48f573179703adf7fb0883d2ee526c190a7ae50422470179fd8`
  (independently replayed and matched).
* `PYTHONPATH=src` is required because the venv's `mc_photogrammetry.pth` points
  at a stale path (`C:/Users/Likea/Documents/mc photogrammetry project/src`), and
  the runner refuses to overwrite an existing output.

## 5. Results (DS-R5I-R1 corrected)

### 5.1 Per-scene

| Dataset | Frames | Pairs | Returned | Positive | Appearance proxy | Wrong-surface | Unattributed | Truth pairs | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `smoke-train-station-ds20c` | 4 | 6 | 8 | 0 | 0 | 0 | 8 | 0 | 0.0 | undefined |
| `pilot-batch-train-station` | 6 | 15 | 18 | 0 | 0 | 3 | 15 | 0 | 0.0 | undefined |
| `recon-canal-houses` | 4 | 6 | 10 | 0 | 0 | 1 | 9 | 0 | 0.0 | undefined |
| `fresh-c4-canal-town` | 4 | 6 | 218 | 0 | 11 | 109 | 98 | 2 | 0.0 | 0.0 |
| `fresh-c4-canal-town-framed` | 8 | 28 | 637 | 6 | 50 | 29 | 552 | 1,325 | 0.00942 | 0.00453 |

`same_face_out_of_tolerance` was 0 in every scene. Across all 61 pairs the
production funnel refused 59: 27 `insufficient_matches`, 6
`insufficient_keypoints`, and 26 `repetition_risk`; only 2 pairs reached `ok`
(both Canal Town). The two train-station captures and canal houses never reached
`ok` at all.

### 5.2 Per-group and pooled

| Group (`scene_group`) | Entries | Pairs | Returned | Positive | Appearance proxy | Wrong-surface | Unattributed | Truth pairs | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `1757a5ad7848…` | 2 | 21 | 26 | 0 | 0 | 3 | 23 | 0 | 0.0 | undefined |
| `3e9dfb4340e1…` | 1 | 6 | 10 | 0 | 0 | 1 | 9 | 0 | 0.0 | undefined |
| `74f080aab201…` | 2 | 34 | 855 | 6 | 61 | 138 | 650 | 1,327 | 0.007018 | 0.004521 |

**Pooled (ratio of sums):** returned 891, positive 6, appearance proxy 61,
wrong-surface 142, unattributed 682, same-face-out-of-tolerance 0, truth pairs
1,327, truth-pairs-returned 6 → **precision 0.006734**, **recall 0.004521**.

**Macro over groups:** precision `0.002339` (3/3 groups defined); recall
`0.004521` (1/3 groups defined; the two train-station/canal-house groups have no
truth keypoint pairs, so their recall is `null`, not zero).

Derived from the sealed counts (not separately sealed): precision restricted to
attributed returned matches is `6 / (891 − 682) = 0.028708`. The sealed artifact
reports the assigned all-returned precision; this derived value is reported here
only to separate "wrong" from "no truth".

`positive_identity_mismatch` is 0: `positive` equals `truth_pairs_returned`, so
the precision numerator and recall numerator are the same six identities.

### 5.3 Raw mutual-matcher gate stratification

Derived from the sealed R1 per-pair rows (no rerun). `ok` is the upstream extract-pair gate, not the later joint pose solver:

| Raw gate status | Frame pairs | Returned | Positive | Truth pairs | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `insufficient_keypoints` | 6 | 10 | 0 | 0 | 0.0 | undefined |
| `insufficient_matches` | 27 | 86 | 0 | 47 | 0.0 | 0.0 |
| `repetition_risk` | 26 | 664 | 4 | 1,212 | 0.00602 | 0.00330 |
| `ok` | 2 | 131 | 2 | 68 | 0.01527 | 0.02941 |

Only 2/61 pairs pass this raw correspondence gate. Even those pairs have 2/131
strict-positive returned identities; this is evidence that the raw mutual-SIFT
stage needs better correspondences and/or eligible views. This scorecard does
not measure the later `joint_matching.match_feature_frames` selections or pose
refusals; DS-R5J is the next stage-specific evaluation.

### 5.3 Effect of the correction

* Truth denominator: 2,525 (DS-R5I, index-counted, **invalid/inflated**) →
  **1,327** unique pixel pairs. Recall: 0.002376 → **0.004521**. Only the
  Canal Town framed scene changed (2,523 → 1,325); all other scenes keep their
  (mostly zero/two-pair) denominators.
* Appearance bucket: 194 (invalid) → **61** corrected proxy. The 133 removed
  identities are now `wrong_surface` (9 → 142), because they were
  same-name/different-slot or otherwise not the exact-state/same-slot proxy.
* Precision numerator/denominator: unchanged (6 / 891), because the six positives
  are strict same-face/tolerance matches and the invalid bucket never contributed
  to `positive`.

### 5.4 Renderer / geometry

26 frames, 43,945 SIFT keypoints. Only 10,072 (22.9%) received a truth surface:
`exceeded_max_distance` 27,892 (rays that never met a registered block, i.e.
background or world geometry outside the exported region), `excluded_partial_owner`
5,981 (partial-model occluder that is excluded from truth), and `hit` 10,072.
This is the dominant geometry limitation and the main reason 682/891 returned
matches are `unattributed`.

## 6. Separated losses (corrected)

* **Evaluation-coverage censoring.** The exported region is a small bounding box
  within each screenshot. 682 returned matches (76.5%) have an endpoint without
  a registered truth surface and are `unattributed`. These are unscored by the
  reference coverage, not proven false matches or reconstruction-geometry errors;
  they lower the all-returned precision bound. The two non-Canal-Town groups
  have no truth keypoint-pair denominator. Partial-model exclusion also removes
  5,981 keypoints.
* **Wrong-surface loss.** 142 returned matches land on a different surface; 133 of
  them are the previously mislabelled same-name/different-slot and same-name/
  different-property cases now correctly counted as wrong. On the one scene with a
  usable denominator (Canal Town), only 6 of 1,327 truth-visible keypoint pairs
  were returned; even over attributed matches precision is 2.87% (6/209).
* **Appearance-proxy loss.** 61 returned matches (6.8% of all returns) are the
  exact-state/same-slot/other-block proxy. This bucket is *not* texture ground
  truth (no texture identity exists in the pipeline) and is an appearance-based
  ambiguity signal only. It is no longer the largest identified failure mode.

## 7. Tests and exact commands

Focused suite (focused tests only, plus `py_compile`), run **before** the single
R1 corpus evaluation:

```
.\.venv\Scripts\python.exe -m pytest -o addopts='' -q tests\test_evaluation_pair_correspondence.py
→ 29 passed in 0.43s
.\.venv\Scripts\python.exe -m py_compile src\mc_photogrammetry\evaluation_pair_correspondence.py scripts\evaluation\evaluate_pair_correspondence.py tests\test_evaluation_pair_correspondence.py
→ exit 0
```

Coverage includes exact same-surface positive, the exact-state/same-slot/
other-block appearance proxy (not a positive), same-block/different-slot as wrong
surface, same-name/different-properties as wrong surface, different-coordinate/
different-slot as wrong surface, exact and near-tolerance boundaries,
occlusion in one frame (wrong surface and no-truth `unattributed`), duplicate
keypoint coordinates collapsing to one truth pair, distinct pixels never
collapsing, duplicate returned identities counted once in precision, non-crossing
duplicate exports across folds, explicit undefined denominators, pooled
ratio-of-sums vs macro arithmetic, the synthetic voxel raycast (entry face, point,
partial exclusion, `type=double` slab), hash-pin discovery refusal on a changed
screenshot, runner overwrite refusal, and an end-to-end synthetic capture through
`build_scene` → `evaluate_scene`.

Single authorized R1 corpus run, foreground, no retry, 19.15 s wall, exit 0:

```
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\evaluation\evaluate_pair_correspondence.py
```

The R1 output self-seal was independently replayed and matched; the legacy
artifact's file hash and self-seal were re-verified as unchanged.

## 8. Limitations

* Pair hypotheses are not truth correspondences. Precision and recall are scored
  against an evaluator-built truth model, not against independent labels or
  ground-truth annotations.
* **No texture identity is available.** The pipeline never reads, hashes or
  compares a block texture. `same_state_same_slot_other_block` is only an
  appearance proxy (exact `BlockState` + same face slot + different block
  coordinate) and is **not texture ground truth**; the corrected counts must not
  be read as a texture-confusion rate.
* The unique-pixel-pair identity deduplicates keypoints at exactly the same pixel
  only. It makes no claim about near-duplicate keypoints at neighbouring pixels.
* Only three independent topology groups exist and all index splits are `train`;
  leave-one-group-out is exploratory and **no generalization claim is made**.
* Partial-model geometry is approximated: partial families occlude as full cubes
  but are excluded from ownership. Absolute precision/recall are affected by this
  renderer limitation, and the numbers are not comparable to a full-model
  renderer.
* The same-surface tolerance is preregistered at one vanilla texel and not swept;
  no sensitivity analysis is reported.
* The exported ground-truth region covers only part of each screenshot; the
  resulting truth sparsity dominates the `unattributed` count and makes two
  groups' recall undefined.
* One corpus and one run per revision; no matcher, threshold, solver, production,
  project or promotion change, and geometry remains unavailable.

## 9. Stop condition

Stopped for lead review. No production edit, no threshold change, no promotion,
and no follow-on run was made. The legacy DS-R5I artifact is preserved unchanged;
the corrected R1 artifact is the new authority. Background/jobs, subagents,
junctions, recursive deletion, obfuscated commands, generated executables,
installs/downloads, Defender changes and system writes were not used; the
pre-existing dirty work tree was preserved.

## 10. DS-R5J-R1 errata (unique pixel-pair scoring identity)

Appended 2026-10-03 after the lead review of the DS-R5J corpus artifact. This
errata corrects the evaluation-only scorer; it **does not recalculate or rewrite
any sealed artifact**. Both historical R5I artifacts and the as-run R5J artifact
are preserved byte-for-byte (section 10.4).

### 10.1 Defect

`score_pair_matches` deduplicated the classification counts (`positive`,
`same_face_out_of_tolerance`, `same_state_same_slot_other_block`, `wrong_surface`,
`unattributed`) and the returned/precision denominator by the **keypoint index
pair**, while `truth_pairs_returned` intersected the **unique pixel-pair** set.
Two keypoint indices at identical pixel coordinates could therefore inflate
`returned` and every category count while the recall numerator stayed pixel-keyed,
and an `unattributed` row (whose `SurfaceHit` is `None`) carried no pixel at all,
so it could only be deduplicated by index. The sealed DS-R5J artifact flags
`consistency.ok = false` with 40 candidate-stage `positive_identity_mismatch`.

### 10.2 Correction (this revision)

* `score_pair_matches` now takes the detected keypoint arrays `left_points` and
  `right_points`, validates every row index against both the truth table and the
  pixel table, and derives every returned identity as the unique
  `(left pixel, right pixel)` pair read from those arrays. There is **no
  index/pixel hybrid fallback**.
* Classification counts, `returned` (the precision denominator) and
  `truth_pairs_returned` (the recall numerator) now share exactly one identity
  space, **including rows with no truth surface**.
* One pixel-pair identity whose rows disagree on the surface verdict raises
  `PairCorrespondenceEvaluationError`: a ray is a function of its pixel, so a
  disagreement means a malformed truth table.
* `_score_with_selection` (the mutual-SIFT caller in this module) threads
  `left.points`/`right.points`; the R5J runner threads the same arrays through its
  raw mutual, reciprocal candidate and selected stage calls (see the R5J report
  errata).

### 10.3 Effect on this DS-R5I-R1 scorecard

**None on the reported mutual-SIFT counts.** For this stage the returned rows pass
through `match_indices`, which already canonicalises every match row to one index
per pixel, so index-pair deduplication and pixel-pair deduplication coincide; the
six positives and the 891/1,327 pooled denominators are unchanged. The correction
makes that equivalence structural rather than incidental. No sealed artifact was
rerun, recomputed or rewritten, and the section 5-6 numbers remain the authority
for the mutual-SIFT stage.

### 10.4 Files and checks (this revision)

| Path | sha256 |
| --- | --- |
| `src/mc_photogrammetry/evaluation_pair_correspondence.py` | `18a0b3a60394c7bbc63e3f3d4a1eb24511d5164c35b5d80629d0c9a78cdef6de` |
| `tests/test_evaluation_pair_correspondence.py` | `ab415506cea584c2f03da7f215ce687d12e211794959cdc90a88fc08ae071c1a` |

Focused tests (adds duplicate attributed-index dedupe, duplicate unattributed-index
dedupe from the point arrays, distinct-pixel preservation, category totals summing
to unique returned pairs with agreeing precision/recall denominators, conflicting
verdict rejection and pixel-table bound validation):

```
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m pytest -o addopts='' -q tests\test_evaluation_pair_correspondence.py
→ 35 passed in 0.41s

.\.venv\Scripts\python.exe -m py_compile src\mc_photogrammetry\evaluation_pair_correspondence.py tests\test_evaluation_pair_correspondence.py
→ exit 0
```

Sealed artifacts re-hashed read-only after the edit and unchanged:
`.tmp-pair-correspondence-eval-20261003.json`
`8dec1dcd56a8e495958ec44a52759147ec96c75caee485b67f91a3b1c08fe5bd` (95,404
bytes), `.tmp-pair-correspondence-eval-r1-20261003.json`
`7faf44c096b51085fb51477c403149ffdb2dbbf00b29d5a7f948003d2fd2784d` (97,289
bytes), and `.tmp-joint-matcher-stage-20261003.json`
`5a803a5e6cabec57b35e1cf3a349cfa800117be350a07c81c5183123bebce571` (910,663
bytes). No evaluator or corpus run was made.
