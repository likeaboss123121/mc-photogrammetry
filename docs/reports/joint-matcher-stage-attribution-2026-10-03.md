# DS-R5J evaluator-only joint-matcher stage attribution (2026-10-03)

Status: **implementation and focused tests complete; the single authorized corpus
evaluation has NOT been run. Stopped for lead review.**

Evaluation-only. No production module, threshold, matcher, solver, project or
promotion decision was changed, and no corpus run, capture, installation, job,
subagent, junction, recursive deletion or system write was used. The owned output
path `.tmp-joint-matcher-stage-20261003.json` is **absent**.

This report is the preregistration and the implementation/test record for the
DS-R5J stage ledger. It deliberately contains **no corpus results**: DS-R5J
authorizes exactly one foreground corpus evaluation only *after* the lead has
reviewed this implementation and its tests, and it forbids an overwrite or retry.
The results section is therefore reserved and empty; it will be completed from the
single sealed artifact if and when the lead authorizes the run.

## 1. Scope and owned paths

DS-R5J extends the DS-R5I-R1 truth ledger from the raw mutual-SIFT correspondence
stage to the actual production `pipeline.joint_matching.match_feature_frames`
output and its refusal/pose stage. It reuses the existing R5I scene/frame/hash
verification and truth construction, changes no production code, and keeps the
required oracle-camera diagnostic clearly separate from and labelled relative to
the default image-only reconstruction path.

| Path | State | sha256 | Notes |
| --- | --- | --- | --- |
| `scripts/evaluation/evaluate_joint_matcher_stage.py` | created | `240bb7c0a42900a28cf3aff618fceecd73943fc41978ffee40fbdb6300d92be6` | one-shot runner + stage ledger (review snapshot) |
| `tests/test_joint_matcher_stage_eval.py` | created | `76353ff15c36393a4aada798a6a1995ff3bf93fe3bc2e3079a2ba815af96151a` | focused tests (review snapshot) |
| `docs/reports/joint-matcher-stage-attribution-2026-10-03.md` | this file | — | preregistration + implementation/test record |
| `.tmp-joint-matcher-stage-20261003.json` | **absent** (confirmed) | — | reserved for the single authorized run |

The implementation lives entirely in the owned script; it defines the evaluation
dataclasses and functions there and imports the R5I truth machinery and the
production interfaces. No file under `src/` was modified.

## 2. Preregistered design

### 2.1 Assisted oracle-camera policy

The joint matcher's geometric stage consumes `CameraEstimate` records; the default
image-only reconstruction path derives those from images, but this evaluation has
no image-derived calibration artifact. DS-R5J therefore builds the estimates from
the **exact factory capture metadata**:

* the frame's recorded *actual* pose (eye position, yaw, pitch, roll) with the
  planned pose only as fallback — exactly the pose DS-R5I already uses for truth;
* intrinsics from the factory field of view at the frame resolution
  (`intrinsics_for`, the production calibration).

Every frame record carries `camera_source = "oracle_factory_capture_metadata"` and
`assisted = true`; the sealed payload carries a top-level `camera_policy` block
with `assisted = true` and `default_image_only_reconstruction = false`, and the
limitations state plainly that this is an oracle aid that makes no image-derived
calibration claim. Feature extraction still reads only pixels and the authoritative
usable mask; the oracle cameras reach the matcher's geometric stage alone. This is
a single-mode diagnostic, so there is no default image-only mode to confuse it with.

### 2.2 Reused truth ledger (unchanged)

* Hash-pinned `exports/datasets/**/index.json` entries and screenshots via
  `discover_finalized_entries` (every listed scene-file and screenshot hash is
  recomputed before admission).
* `build_scene` reads the recorded camera pose, frame resolutions and the
  registered `ground_truth.json` region occupancy.
* Truth surfaces come from the DS-R5I sparse-z-buffer raycast
  (`raycast_surface`): first occupied cell owns the surface, the entry face is the
  slot, the entry point is the world surface point.
* A strict positive pair is two keypoints, one per frame, on the same
  `(block coordinate, face slot)` within the preregistered
  `SURFACE_TOLERANCE_BLOCKS = 0.0625` (one vanilla texel).
* `same_state_same_slot_other_block` remains an **appearance proxy only** (same
  exact `BlockState`, same face slot, different block coordinate). No texture is
  ever read, hashed or compared; the bucket is not texture ground truth.
* Truth and every returned identity use the **unique pixel pair**
  `(left pixel, right pixel)`: repeated keypoints at exactly one pixel collapse,
  distinct pixels never collapse.

The joint detection path mirrors production `match_and_solve_pair`: features are
detected with
`CorrespondenceSettings(max_features=joint.max_features=4000,
mask_boundary_pad_px=joint.mask_boundary_pad_px=8.0)` and `ratio_test=0.75`, so the
descriptor-footprint boundary drop is the production one. The matcher is called as
the unchanged `match_feature_frames(features, features, oracle_left, oracle_right,
matcher=JointMatchSettings(), ratio_test=0.75)`.

### 2.3 Stage ledger per unordered frame pair

For every unordered frame pair of every scene, in stage order:

1. **Raw mutual-SIFT baseline** — `select_mutual_matches` rows, scored against the
   truth model.
2. **Reciprocal top-k candidate graph** — `reciprocal_topk_graph` edge count,
   untruncated total, truncation, and the candidate edges scored against truth.
3. **Joint selection** — `JointPairResult.matches` scored against truth.
4. **Support and gates** — support, support floor, spatial bins, selected median
   residual, descriptor cost, parallax, cheirality, singular ratio, ambiguity
   margin, plus every ranked hypothesis with its own gate verdict and refusal.
5. **Solved/refused reason** — the module-local `status`/`reason`, the DS-11 pose
   facts (`right_yaw_class`, pose reason, inliers, inlier fraction, median rho).
6. **Selection availability on refusals** — see 2.4.
7. **Denominator split** — see 2.5.

### 2.4 Refusal versus pose abstention

`describe_outcome` classifies each result on two independent axes:

| selection state | meaning |
| --- | --- |
| `selected_solved` | the pair solved |
| `selected_on_refusal` | the pair refused but still carries a non-empty best one-to-one selection |
| `absent_selection` | the pair refused with no selection at all (early refusal) |

| pose state | meaning |
| --- | --- |
| `pose_solved` | DS-11 solved the selection |
| `pose_class_mismatch` | DS-11 solved a *different* yaw class (`selection_unstable` refusal) |
| `pose_abstained` | DS-11 ran on the selection and refused (`solve_pair_refused`) |
| `pose_not_attempted` | no DS-11 pose was produced |

The joint `status` is authoritative, **not** `JointPairResult.solved`: a
`selection_unstable` refusal carries a solved DS-11 pose, so that property is
`True` while the pair was refused. The pooled and per-group records report
`refusals_with_selected_matches`, `refusals_without_selected_matches` and
`refusals_with_pose_attempt`, so an absent selection is never conflated with a
pose abstention.

### 2.5 Correspondence versus geometry denominators

Each pair reports both denominator families explicitly:

* **correspondence denominator** — `correspondence_truth_pairs`: the
  truth-visible unique-pixel-pair keypoint pairs (the recall denominator);
* **geometry denominators** — `geometry_mutual_returned`,
  `geometry_candidate_edges`, `geometry_selected_returned`: the returned-identity
  counts of the raw mutual baseline, the reciprocal candidate graph and the joint
  selection (the precision denominators);
* **reach counts** — `mutual_correspondence_reach`,
  `candidate_correspondence_reach`, `selected_correspondence_reach`: how many
  truth pairs each stage's returned identities actually cover. Comparing candidate
  reach (the geometric reachability upper bound) with selected reach attributes a
  loss to the candidate-graph/correspondence stage or to the geometry-selection
  stage.

Undefined denominators are `null` with a typed `undefined_reason`
(`no_returned_matches`, `no_truth_keypoint_pairs`), never zero; pooled metrics are
ratios of sums, macro metrics skip undefined groups and report the count.

### 2.6 Grouping and leakage

Grouping is strictly by `scene_group` via the existing `build_group_manifest`,
with duplicate exports of one scene kept together. Split-leakage and
leave-one-group-out fold leakage are checked (`check_split_leakage`,
`check_fold_leakage`) and recorded in the sealed payload; a manifest with leakage
raises before any evaluation. Only three independent topology groups exist, all
index splits are `train`, so the folds are descriptive/exploratory and **no
generalization claim is made**. No production thresholds are tuned, and no
per-group or per-scene parameter fitting exists.

## 3. Sealed output

The runner refuses to overwrite an existing output (exit 2). On success it writes
one sealed JSON record with schema
`mc-photogrammetry-joint-matcher-stage-attribution-v1`, including the camera
policy, the exact joint/detection parameters, the hash-pinned inputs and group
manifest, per-scene frames (with the assisted camera record) and pairs, per-group
and pooled stage summaries, the macro metrics, the explicit stage-attribution and
denominator blocks, the renderer/geometry policy plus keypoint truth coverage, an
internal `seal_sha256` self-seal, and the limitations list.

## 4. Focused tests and exact commands

Focused suite and `py_compile`, run before and independently of any corpus
evaluation:

```
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m pytest -o addopts='' -q tests\test_joint_matcher_stage_eval.py
→ 18 passed in 0.96s

.\.venv\Scripts\python.exe -m py_compile scripts\evaluation\evaluate_joint_matcher_stage.py tests\test_joint_matcher_stage_eval.py
→ exit 0

.\.venv\Scripts\python.exe -m pytest -o addopts='' -q tests\test_joint_matcher_stage_eval.py tests\test_evaluation_pair_correspondence.py
→ 47 passed in 1.09s
```

Coverage:

* **Oracle camera** — exact factory pose/intrinsics copy (vertical FOV focal),
  `camera_record` assisted label, and the flag/`camera_source` on the record.
* **Production detection settings** mirror `match_and_solve_pair`
  (`max_features=4000`, `mask_boundary_pad_px=8.0`, `ratio_test=0.75`).
* **Synthetic same surface** — 20 independently projected 3D points with one
  matching truth surface each solve through the real `match_feature_frames`;
  selected precision/recall are exactly 1.0 and every stage denominator is
  reported.
* **Synthetic wrong surface** — the selected set has zero truth pairs, zero
  positives, all `wrong_surface`, precision 0.0 and `null` recall.
* **Synthetic occluded surface** — one frame has no truth surface, so the selected
  set is all `unattributed`, with zero positives and `null` recall.
* **Denominator split** — `pair_record` exposes correspondence versus geometry
  denominators and stage precision/recall.
* **Refusal with selected matches** — absent selection (no matches, no pose),
  selection without a pose attempt, selection with `pose_abstained`, and the
  `pose_class_mismatch` case where a *refused* pair carries a *solved* DS-11 pose.
* **Refusal flow-through** — a monkeypatched refusal with a non-empty selection is
  attributed end-to-end as `selected_on_refusal`/`pose_abstained`, and an empty
  refusal as `absent_selection`.
* **Pixel dedupe** — a duplicated pixel pair is one returned identity and one
  positive (precision denominator 1); distinct pixel pairs are never collapsed.
* **Pooled accounting** — solved/refused, refusal reasons, selection/pose state
  histograms and the refusal-with-selection counters.
* **Payload assembly** — `build_payload` seals, labels the assisted camera, is
  JSON-safe and its self-seal reproduces (scenes and features injected, no corpus).
* **Fold leakage** — duplicate exports stay in one fold; same scene in two groups
  and a group on both sides of a fold are rejected/detected.
* **Runner guards** — overwrite refusal (exit 2) and failure-without-writing when
  the export root is missing (exit 1, no output).

## 5. Corpus run status

**Not run.** DS-R5J authorizes exactly one foreground corpus evaluation *after*
lead review of this implementation and its tests. No hash pin has been re-verified
against the finalized exports in this session, and the reserved output path
`.tmp-joint-matcher-stage-20261003.json` remains absent (confirmed immediately
before this report was written).

When the lead authorizes it, the single run is:

```
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\evaluation\evaluate_joint_matcher_stage.py
```

It must re-verify all index/screenshot pins, the split/fold leakage lists and the
absence of the output path, run once in the foreground with no retry, and stop for
lead review. `PYTHONPATH=src` is required because the venv's
`mc_photogrammetry.pth` points at a stale path (DS-R5I-R1 report, section 4.2).

### 5.1 Results (reserved)

Reserved for the single authorized run. The report will then record the source and
artifact hashes, the exact command and wall time, the pooled/per-group/per-scene
tables, the raw mutual versus candidate-graph versus joint-selected stratification,
the refusal/pose-state counts, and the correspondence-versus-geometry denominator
attribution.

## 6. Limitations

* **Assisted camera.** The oracle factory camera metadata is an evaluation aid, not
  image-derived calibration. This diagnostic is not the default image-only
  reconstruction path and its solve rates must not be read as image-only results.
  It is not a reconstruction and produces no geometry.
* Pair hypotheses are not truth correspondences; precision/recall are scored
  against an evaluator-built truth model, not independent labels.
* **No texture identity** exists anywhere in the pipeline.
  `same_state_same_slot_other_block` is an appearance proxy only and is not
  texture ground truth.
* The exported ground-truth region covers only part of each screenshot; returned
  matches whose endpoint has no registered truth surface are `unattributed` and
  lower the all-returned precision bound rather than being proven false. Partial
  block models occlude as full cubes but are excluded from truth ownership.
* Truth and returned identities deduplicate only keypoints at exactly the same
  pixel; near-duplicate keypoints at neighbouring pixels are not collapsed.
* Only three independent topology groups exist and all splits are `train`;
  leave-one-group-out is exploratory and no generalization claim is made.
* One corpus and one run; the stage ledger recomputes the raw mutual baseline and
  candidate graph (in addition to the unchanged matcher's internal computation) so
  their identities can be scored, which costs extra time but changes nothing in
  production.

## 7. Stop condition

Stopped for lead review. No production edit, no threshold change, no corpus run,
no promotion and no follow-on run was made. No background process/job/workflow,
subagent, shell nesting, junction/reparse point, recursive deletion, generated
executable, install/download, Minecraft/client/capture action, Defender change or
system write was used, and the pre-existing dirty work tree was preserved.

## 8. DS-R5J-R1 errata (as-run artifact audit and scoring-identity correction)

Appended 2026-10-03 after the lead review of the single authorized DS-R5J corpus
run. This errata records the as-run artifact and corrects the evaluation-only
scorer. It does **not** recalculate or rewrite the sealed artifact, which is
preserved byte-for-byte.

### 8.1 As-run artifact (preserved as-run)

The single authorized foreground corpus run of this runner was executed after lead
review (AGENT_COMMS 2026-10-03), exit 0, 84.44 s wall. The reserved output
`.tmp-joint-matcher-stage-20261003.json` is 910,663 bytes, SHA-256
`5a803a5e6cabec57b35e1cf3a349cfa800117be350a07c81c5183123bebce571`, self-seal
`103f536505a40e0ed179920a00f56d4899de8ff9f05771e1c308810608c63822`
(lead-verified), 5 entries / 3 groups / 61 pairs. Section 5's "not run" status is
superseded: the run occurred, is sealed, and is preserved as-run. No second corpus
run is authorized by this task.

### 8.2 Defect: candidate-stage index/pixel hybrid

The sealed artifact correctly flags `consistency.ok = false`: the candidate stage
has 40 `positive_identity_mismatch`. `score_pair_matches` deduplicated the
classification counts and returned/precision denominator by keypoint index pair,
while candidate correspondence reach (`truth_pairs_returned`) intersected unique
pixel pairs. Duplicate keypoint indices at identical pixel coordinates therefore
inflated the candidate positive and per-verdict counts and the candidate returned
denominator, producing `positive > truth_pairs_returned`. **The as-run candidate
positive 385, candidate precision, and candidate per-verdict category counts are
INVALID and must not be quoted.** The mutual and selected stages were internally
consistent (their match rows are canonicalised to one index per pixel before
scoring) and remain valid.

### 8.3 Correction (this revision)

`score_pair_matches` now takes `left_points`/`right_points` and derives every
returned identity from those arrays, deduplicating all classification counts, the
returned/precision denominator and the recall reach by unique pixel pair, including
rows with no truth surface. Index/pixel hybrid fallback is removed, every row index
is validated against both the truth and pixel tables, and a conflicting verdict for
one pixel pair raises. `attribute_pair` threads
`left.features.points`/`right.features.points` through the raw mutual, reciprocal
candidate and selected calls.

The correction changes only evaluation arithmetic, so the corrected candidate
numbers cannot be recovered from the as-run artifact; they require a re-run, which
is **not authorized here** and was not performed.

### 8.4 Stage facts that remain valid (lead-verified from the as-run artifact)

Candidate reach uses unique pixel-pair intersection and is unaffected: 345/1,215
truth pairs (28.40%) in Canal Town; selected reach 52/1,215 (4.28%), losing
293/345 = 84.93% of candidate-reachable truth pairs; mutual baseline 6/1,215
(0.49%). Only Canal Town has a nonzero truth denominator, so the other two groups'
recall is undefined. Selected counts are internally consistent: 52/12,423 precision
(0.42%) and 52/1,215 recall (4.28%), with precision a censored lower bound because
the registered ground-truth region covers part of each screenshot. Joint outcome:
6 solved, 55 refused with selected matches; reasons 31 `cheirality_failed`, 18
`low_spatial_spread`, 6 `insufficient_one_to_one_support`; no refusal produced a
DS-11 `PairPose` (the failures are pre-`solve_pair` geometric gates). Oracle camera
assistance and the three-group/train-only exploratory limits stand. The partial
region `unattributed` caveat and the no-texture-ground-truth claim are unchanged.

### 8.5 Files and checks (this revision)

| Path | sha256 |
| --- | --- |
| `scripts/evaluation/evaluate_joint_matcher_stage.py` | `e52770b9de3adcba159d3743789efa565aa1b34392b062db1820f8d32aa7bd79` |
| `tests/test_joint_matcher_stage_eval.py` | `960641becf739e90bed7ed37d5fea01cb3100174a47d2e2102fce0d5c7c67335` |

Focused tests add a caller-threading check (all three stage scorers receive the
detected point arrays) plus candidate-stage dedupe regressions for duplicate
attributed and unattributed indices:

```
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m pytest -o addopts='' -q tests\test_joint_matcher_stage_eval.py
→ 21 passed in 1.07s

.\.venv\Scripts\python.exe -m pytest -o addopts='' -q tests\test_joint_matcher_stage_eval.py tests\test_evaluation_pair_correspondence.py
→ 56 passed in 1.32s

.\.venv\Scripts\python.exe -m py_compile scripts\evaluation\evaluate_joint_matcher_stage.py tests\test_joint_matcher_stage_eval.py
→ exit 0
```

Sealed artifacts re-hashed read-only after the edit and unchanged:
`.tmp-pair-correspondence-eval-20261003.json`
`8dec1dcd56a8e495958ec44a52759147ec96c75caee485b67f91a3b1c08fe5bd` (95,404
bytes), `.tmp-pair-correspondence-eval-r1-20261003.json`
`7faf44c096b51085fb51477c403149ffdb2dbbf00b29d5a7f948003d2fd2784d` (97,289
bytes), `.tmp-joint-matcher-stage-20261003.json`
`5a803a5e6cabec57b35e1cf3a349cfa800117be350a07c81c5183123bebce571` (910,663
bytes). No evaluator or corpus run was made.
