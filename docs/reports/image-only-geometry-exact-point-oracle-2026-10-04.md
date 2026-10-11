# DS-R5O same-surface world-hit proximity oracle vs mode selection (2026-10-04)

Status: **single pinned evaluation complete; artifact sealed and self-verified.
Evaluation-only truth-assisted diagnostic; no production changes.**

Remediation 2026-10-04 (Luna HOLD, `R5O inference and metric wording`, plus the
Lead reviewer addendum `R5O factual corrections`): the run artifact is unchanged
(no rerun); this report and the script docstrings now label every arm as
**same-surface continuous world-hit proximity at the arm tolerance** instead of
literal feature identity, the earlier causal readings are retracted, and a per-arm
precision/recall table against each arm's own tolerance denominator is added
beside the fixed 1,213 surface-denominator table. The pose-error generalization is
retracted and recomputed from the artifact, the test boundary is corrected to
`0.25 + 1e-6`, and the report states that exact feature identity is not
established. The legacy `exact_point`/`point_oracle_*` schema and arm keys are
preserved as labels only. See section 10.

Task: `agent swarm instantiation template/tasks/active/image-only-geometry-exact-point-oracle-2026-10-04.md`.
Owner: DeepSeek (Sol-equivalent). Luna coordinates; Astra read-only acceptance.

## 1. Question

On the canonical 8-view Canal Town fixture, how does the unchanged production
matcher/solver respond as the injected candidate graph is built from the same
visible surface with a tighter or looser **continuous world-hit proximity** bound,
relative to the DS-R5K/DS-R5N same-surface oracle? Detection, pair selection,
hypothesis generation, selector, solver and every production threshold are held
fixed; only the injected candidate graph differs. The proximity bound is not a
literal feature-identity test, and because candidate count, achieved surface
precision and each arm's own proximity denominator move together, this is a
single-fixture sensitivity measurement rather than a causal experiment (section 7).


## 2. Scope and owned paths

| Path | State | sha256 |
| --- | --- | --- |
| `scripts/evaluation/evaluate_exact_point_oracle.py` | created; docstrings reworded 2026-10-04 | `afe3ebc4533e2c235be21ee213b78b8f13714e9d51b4409594ae239a9664883e` |
| `tests/test_exact_point_oracle.py` | created; unchanged by the remediation | `ebb86af53fad6bf41617a96a07002126aedeb884a2f8214397c8a43915b58daf` |
| `docs/reports/image-only-geometry-exact-point-oracle-2026-10-04.md` | this file | see section 8 |
| `.tmp-exact-point-oracle-20261004.json` | created once; unchanged by the remediation | `03faaefe40ac1496ee6f86800818f070bb3db043984d092025a8dff67c65007d` |
| `agent swarm instantiation template/tasks/active/image-only-geometry-exact-point-oracle-2026-10-04.md` | result section appended | - |
| `AGENT_COMMS.md` | ASCII result appended at byte level | - |

No file under `src/`, no CLI, no canonical schema, no gate and no prior evaluation
script/test was modified. The reviewed sealed artifacts
`.tmp-oracle-candidate-graph-20261003.json` (DS-R5K) and
`.tmp-oracle-support-sensitivity-20261003.json` (DS-R5N) were neither read nor
written; the runner's output guard refuses them case-insensitively.

## 3. Preregistered design

### 3.1 Canonical entry and pinned preflight

The reviewed DS-R5K canonical selection (`select_canonical_entry`, scoped to the
`canal-town` scene prefix, most complete export, tie-broken) and the reviewed
DS-R5N pinned preflight (`verify_r5k_pins`, `verify_canonical_scene_pins`) are
loaded by path and reused verbatim, so this diagnostic cannot drift from the
frozen corpus. Before any image processing `discover_finalized_entries` re-hashes
every listed index scene-file and screenshot, `verify_r5k_pins` requires exactly
the five pinned dataset ids and index digests, `verify_canonical_scene_pins`
requires the pinned scene files and all eight screenshot digests, and
`build_group_manifest` rejects split/fold leakage.

### 3.2 Assisted oracle camera

The matcher's geometric stage consumes `CameraEstimate` records built from the
recorded factory capture metadata (actual pose with planned fallback, FOV and
frame resolution via `intrinsics_for`). Every frame record is labelled
`oracle_factory_capture_metadata`/`assisted = true`. Feature extraction reads only
pixels and the authoritative usable mask. Normal reconstruction remains
image-only; this is not the default path and makes no image-derived calibration
claim.

### 3.3 Six arms, one unchanged solver

Per unordered frame pair the unchanged production
`match_feature_frames` runs under six injected candidate graphs that share
identical detection settings, assisted cameras, gates, ratio test, thresholds and
solver (`JointMatchSettings()` defaults; detection mirrors production:
`max_features = 4000`, `mask_boundary_pad_px = 8.0`, `ratio_test = 0.75`):

1. `production_baseline` - the production reciprocal descriptor graph;
2. `surface_oracle` - the reviewed DS-R5K same-surface graph, i.e. the frozen
   DS-R5I rule (same block coordinate and face slot, raycast world points within
   `SURFACE_TOLERANCE_BLOCKS = 0.0625`);
3. `point_oracle_0.05`, `_0.125`, `_0.25`, `_0.5` - the same-surface graph
   rebuilt under a **fixed, preregistered continuous world-hit separation
   sweep**: both detector keypoints raycast through the frozen actual camera into
   the *same visible surface* (same block coordinate and face slot) and their
   retained continuous hit points are separated by no more than the fixed,
   preregistered tolerance. The `point_oracle_*` arm keys are legacy labels
   retained for artifact continuity; the rule is continuous world-hit proximity,
   not literal feature-point identity. The tolerance sweep is not selected from
   solve success and is never a production threshold.

At the surface tolerance the point rule and the DS-R5K surface rule are the same
construction; the sweep brackets 0.0625 with one stricter value (0.05) and three
looser values. Reusing the reviewed DS-R5K graph builder means the point oracle is
exactly the frozen truth rule `construct_truth_pairs` at the swept tolerance,
mapped to canonical first keypoint indices with each edge's actual L2 descriptor
distance preserved and edges sorted by `(distance, left index, right index)`.

### 3.4 Injection and restoration

`mc_photogrammetry.pipeline.joint_matching.reciprocal_topk_graph` is replaced for
exactly one matcher call per oracle arm and restored in a `finally` block, so it
restores even if the matcher raises. The production baseline runs before any
injection and never sees an injected graph.

### 3.5 Stage accounting and typed absences

Every arm's candidate graph and selected matches are scored with the corrected
unique-pixel-pair evaluator against the **fixed 1,213-pair surface denominator**
(so arms are comparable) and, for the oracle arms, additionally against that arm's
own proximity denominator (the sealed `candidate_stage_arm_tolerance` /
`selected_stage_arm_tolerance` fields). Both readings are reported in section 6.1.
The sealed record separates:

* the strict same-surface truth-pair denominator (1,213) and each same-proximity
  truth-pair denominator (1,068 / 1,780 / 5,037 / 13,545);
* censored points explicitly (`keypoints_with_truth` 5,531 of 25,834; no-truth
  reasons `exceeded_max_distance` 17,180 and `excluded_partial_owner` 3,123;
  1,588 boundary keypoints dropped);
* candidate-graph precision/recall, selected-match precision/recall after the
  unchanged selector, pair solve success, typed refusal reason, selection/pose
  state, support/spatial/cheirality/parallax/singular/ambiguity gates and every
  ranked hypothesis and per-yaw `YawClassScore`;
* the DS-11 pose error against the assisted ground-truth cameras where defined.

This diagnostic stops at the pair matcher/solver. It never runs the production
plane-graph or placement stages, so axis coverage/resultants/residuals and
placed-cell geometry are reported as explicitly **unavailable**
(`pair_level_diagnostic_has_no_plane_graph_record`,
`pair_level_diagnostic_has_no_placement_record`), never as zero, at both pair and
payload level.

### 3.6 Fail-closed context

`build_point_oracle_graph` refuses a non-positive/non-finite separation tolerance,
a truth hit table whose length does not match the detected keypoints, and a truth
pixel with no detected keypoint (typed `ExactPointOracleError`). Empty truth
builds an empty graph, which the unchanged matcher reports as its typed
`candidate_graph_empty` refusal.

## 4. Focused tests

```
.\.venv\Scripts\python.exe -m py_compile scripts\evaluation\evaluate_exact_point_oracle.py tests\test_exact_point_oracle.py
-> exit 0

$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m pytest -o addopts='' -q tests\test_exact_point_oracle.py --no-header
-> 30 passed in 7.03s
```

Coverage: same-surface admission inside/outside each tolerance; the inclusive
boundary (`separation == tolerance` admitted at the exactly representable 0.25,
0.25 + 1e-6 over refused); different-coordinate and different-slot surface rejection at
identical world points; equivalence with the reviewed DS-R5K surface graph at the
surface tolerance; actual descriptor-distance preservation and edge sort order;
duplicate-pixel collapse to one canonical edge; empty-truth empty graph; typed
fail-closed errors for an orphan truth pixel, a short hit table and bad
tolerances; the six ordered arms with the production builder restored and
production `JointMatchSettings` unchanged; per-arm tolerance binding and
denominators; deterministic repeated runs; wrong/occluded surface refusal; the
surface-versus-proximity mutation control (same face 0.1 apart: 2 surface edges
versus 4 edges at 0.125); pair-record denominators, censoring and typed absences;
pooled accounting with separate arms and zero positive-identity mismatch; sealed
payload labelling, JSON safety and self-seal reproduction; case-insensitive sealed
artifact guard, overwrite refusal and failure-without-writing.

## 5. The single pinned run

```
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\evaluation\evaluate_exact_point_oracle.py
```

Run once in the foreground from the repository root. Exit code `0`; wall time
`102.613 s`. Preflight confirmed `.tmp-exact-point-oracle-20261004.json` and its
`.partial` path absent; postflight confirmed the output exists and no `.partial`
was written. No rerun, no overwrite.

Artifact: `.tmp-exact-point-oracle-20261004.json`, 2,855,980 bytes, sha256
`03faaefe40ac1496ee6f86800818f070bb3db043984d092025a8dff67c65007d`. Stored
`seal_sha256 = 5d8a67edbb6aeb1c0dadc998f1ffa2610718e5cd36e84a5a98c11c0ab5e73899`;
recomputing `seal_record` reproduces it exactly. `consistency.ok = true` with zero
`positive_identity_mismatch` for every arm and both scored stages.

Input admission: 5 discovered entries, all index pins matched; 3 topology groups;
`split_leakage_issues = []`, `fold_leakage_issues = []`; canonical
`fresh-c4-canal-town-framed-2026-09-12` (scene
`canal-town-architectural-component-bbox`, group
`74f080aab20105733ddf845ccd1a938b76f01f00edace3a33cb9275a47cf458a`, split
`train`); all 5 canonical scene-file pins and all 8 screenshot pins matched; the
4-frame `fresh-c4-canal-town-2026-09-12` duplicate was a considered candidate and
was not selected. 8 frames, 28 unordered pairs, 1,213 pooled surface truth pairs,
0 empty-truth pairs.

## 6. Results

### 6.1 Pooled per-arm stage metrics

Returned = unique-pixel-pair geometry identities; reach = `truth_pairs_returned`
against the fixed 1,213 surface denominator; `edges` is the summed injected graph
edge count.

| Arm | Separation tol | Solved | Edges | Cand returned | Cand reach | Cand precision | Cand recall | Sel returned | Sel reach | Sel precision | Sel recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| production_baseline | - | 1 | 178,056 | 169,586 | 345 | 0.002034366 | 0.284418796 | 8,863 | 52 | 0.005867088 | 0.042868920 |
| surface_oracle | 0.0625 | 4 | 1,213 | 1,213 | 1,213 | 1.0 | 1.0 | 865 | 865 | 1.0 | 0.713107997 |
| point_oracle_0.05 | 0.05 | 3 | 1,068 | 1,068 | 1,068 | 1.0 | 0.880461665 | 717 | 717 | 1.0 | 0.591096455 |
| point_oracle_0.125 | 0.125 | 7 | 1,780 | 1,780 | 1,213 | 0.681460674 | 1.0 | 1,253 | 965 | 0.770151636 | 0.795548228 |
| point_oracle_0.25 | 0.25 | 2 | 5,037 | 5,037 | 1,213 | 0.240817947 | 1.0 | 2,526 | 1,057 | 0.418448139 | 0.871393240 |
| point_oracle_0.5 | 0.5 | 4 | 13,545 | 13,545 | 1,213 | 0.089553341 | 1.0 | 3,450 | 752 | 0.217971014 | 0.619950536 |

The `production_baseline` and `surface_oracle` rows reproduce the DS-R5K sealed
result exactly (1 solved / 345 candidate reach / 52 selected reach; 4 solved /
1,213 / 865). Every oracle candidate graph is precision 1.0 by construction
against its own rule; the surface-denominator columns are the comparable ones.

Same-surface proximity truth-pair denominators (unique same-surface pixel pairs
within each separation tolerance): `0.05 -> 1,068`, `0.125 -> 1,780`,
`0.25 -> 5,037`, `0.5 -> 13,545`; the 1,213 surface denominator sits between the
first two. Candidate-reach ceilings: production 345, surface 1,213,
same-proximity 1,068 / 1,213 (capped at the surface denominator) / 1,213 / 1,213.

**Per-arm precision/recall against each arm's own tolerance denominator** (read
from the immutable artifact's `summary.arms[*].candidate_stage_arm_tolerance` and
`selected_stage_arm_tolerance` fields; denominator = that arm's own unique
same-surface-within-tolerance truth pairs). This table sits beside the fixed
1,213 surface-denominator table above.

| Arm | Separation tol | Own truth denominator | Cand returned | Cand precision | Cand recall | Sel returned | Sel precision | Sel recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| production_baseline | - | unavailable (`arm_has_no_arm_tolerance_stage`) | - | - | - | - | - | - |
| surface_oracle | 0.0625 | 1,213 | 1,213 | 1.0 | 1.0 | 865 | 1.0 | 0.713107997 |
| point_oracle_0.05 | 0.05 | 1,068 | 1,068 | 1.0 | 1.0 | 717 | 1.0 | 0.671348315 |
| point_oracle_0.125 | 0.125 | 1,780 | 1,780 | 1.0 | 1.0 | 1,253 | 1.0 | 0.703932584 |
| point_oracle_0.25 | 0.25 | 5,037 | 5,037 | 1.0 | 1.0 | 2,526 | 1.0 | 0.501488982 |
| point_oracle_0.5 | 0.5 | 13,545 | 13,545 | 1.0 | 1.0 | 3,450 | 1.0 | 0.254706534 |

In this own-denominator reading the candidate graph *is* the arm's truth-pair list
by construction, so candidate precision and recall are both 1.0 and carry no
information beyond reach; selected precision is likewise 1.0 because selected
matches are a subset of the injected arm-truth edges. The informative column is
selected recall, which varies non-monotonically with the bound: 0.713 (0.0625),
0.671 (0.05), 0.704 (0.125), 0.501 (0.25), 0.255 (0.5). `production_baseline`
has no arm-tolerance stage because its production graph is not built from a
proximity rule; that is reported as typed unavailable, never zero.

### 6.2 Refusals, selection states and pose states (28 pairs)

| Arm | Refusal reasons | Selection states | Pose states |
| --- | --- | --- | --- |
| production_baseline | cheirality_failed 27 | selected_on_refusal 27, selected_solved 1 | pose_not_attempted 27, pose_solved 1 |
| surface_oracle | insufficient_one_to_one_support 19, solve_pair_refused 4, low_spatial_spread 1 | absent_selection 19, selected_on_refusal 5, selected_solved 4 | pose_not_attempted 20, pose_abstained 4, pose_solved 4 |
| point_oracle_0.05 | insufficient_one_to_one_support 20, solve_pair_refused 4, low_spatial_spread 1 | absent_selection 20, selected_on_refusal 5, selected_solved 3 | pose_not_attempted 21, pose_abstained 4, pose_solved 3 |
| point_oracle_0.125 | insufficient_one_to_one_support 17, solve_pair_refused 2, ambiguous_hypothesis 2 | absent_selection 17, selected_on_refusal 4, selected_solved 7 | pose_not_attempted 19, pose_abstained 2, pose_solved 7 |
| point_oracle_0.25 | insufficient_one_to_one_support 13, ambiguous_hypothesis 13 | absent_selection 6, selected_on_refusal 20, selected_solved 2 | pose_not_attempted 26, pose_solved 2 |
| point_oracle_0.5 | ambiguous_hypothesis 13, insufficient_one_to_one_support 9, cheirality_failed 2 | absent_selection 1, selected_on_refusal 23, selected_solved 4 | pose_not_attempted 24, pose_solved 4 |

Transitions relative to the surface oracle:

* `point_oracle_0.05` gained 0 and lost 1 (`v03-v04`): the stricter proximity
  bound reduces reach and refuses on support.
* `point_oracle_0.125` gained 3 (`v02-v04`, `v02-v06`, `v04-v07`) and lost 0.
* `point_oracle_0.25` gained 2 (`v01-v02`, `v03-v05`) and lost all four surface
  solves (`v02-v03`, `v03-v04`, `v04-v06`, `v06-v07`).
* `point_oracle_0.5` gained 3 (`v00-v01`, `v01-v03`, `v02-v05`) and lost 3
  (`v03-v04`, `v04-v06`, `v06-v07`).

### 6.3 Assisted-camera ground-truth pose error of solved arms

| Arm | Solved | Pose defined | Yaw class correct | Median relative rotation error (deg) | Median gauge-aligned baseline error (deg) |
| --- | ---: | ---: | ---: | ---: | ---: |
| production_baseline | 1 | 1 | 0 | 180.0 | 7.746654 |
| surface_oracle | 4 | 4 | 4 | 6.037e-07 | 0.063844 |
| point_oracle_0.05 | 3 | 3 | 3 | 1.207e-06 | 0.079967 |
| point_oracle_0.125 | 7 | 7 | 7 | 1.207e-06 | 0.065730 |
| point_oracle_0.25 | 2 | 2 | 2 | 6.037e-07 | 0.376038 |
| point_oracle_0.5 | 4 | 4 | 4 | 0.0 | 0.344994 |

The single baseline solve chooses the wrong yaw quadrant (180 deg), matching the
R5K/R5N finding that the baseline solve is not a correct correspondence. All 20
solved oracle rows have a defined pose and select the assisted ground-truth yaw
class. Per-solve relative rotation error spans 0 to 1.708e-06 deg across the
oracle arms (per-arm median in the table above); per-solve gauge-aligned baseline
error spans 0.0143 deg (surface, `v02-v03`) to 1.335 deg (0.5, `v02-v03`), so no
single sub-0.4-deg bound holds for every solved row. The earlier wording claiming
a universal sub-microdegree / sub-0.4-deg result for every solve is retracted; the
accurate per-arm and per-solve values above are recomputed from the immutable
artifact's `pose_error` fields.

### 6.4 Per-pair ledger

`vNN` = `view-canal-town-architectural-component-bbox-NN`; `truth` is the surface
denominator; `pt` is the same-proximity denominator at 0.05/0.125/0.25/0.5.

| # | Pair | Truth | pt sweep | Baseline | Surface | p0.05 | p0.125 | p0.25 | p0.5 |
| ---: | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | v00-v01 | 20 | 16/34/82/202 | cheirality_failed | support | support | support | ambiguous | **solved** |
| 2 | v00-v02 | 4 | 4/12/60/194 | cheirality_failed | support | support | support | support | ambiguous |
| 3 | v00-v03 | 1 | 1/4/15/48 | cheirality_failed | support | support | support | support | support |
| 4 | v00-v04 | 2 | 1/7/32/100 | cheirality_failed | support | support | support | support | support |
| 5 | v00-v05 | 6 | 5/15/58/156 | cheirality_failed | support | support | support | support | ambiguous |
| 6 | v00-v06 | 3 | 3/8/45/126 | cheirality_failed | support | support | support | support | support |
| 7 | v00-v07 | 4 | 3/15/37/101 | cheirality_failed | support | support | support | support | ambiguous |
| 8 | v01-v02 | 37 | 31/68/201/594 | cheirality_failed | support | support | support | **solved** | ambiguous |
| 9 | v01-v03 | 11 | 10/25/76/219 | **solved** | support | support | support | support | **solved** |
| 10 | v01-v04 | 7 | 5/19/59/157 | cheirality_failed | support | support | support | support | support |
| 11 | v01-v05 | 18 | 13/26/57/152 | cheirality_failed | support | support | support | support | support |
| 12 | v01-v06 | 4 | 3/8/32/90 | cheirality_failed | support | support | support | support | support |
| 13 | v01-v07 | 9 | 9/15/22/47 | cheirality_failed | support | support | support | support | support |
| 14 | v02-v03 | 91 | 86/113/265/731 | cheirality_failed | **solved** | **solved** | **solved** | ambiguous | **solved** |
| 15 | v02-v04 | 102 | 90/154/433/1186 | cheirality_failed | spatial | spatial | **solved** | ambiguous | ambiguous |
| 16 | v02-v05 | 99 | 86/144/420/1105 | cheirality_failed | solve_refused | solve_refused | solve_refused | ambiguous | **solved** |
| 17 | v02-v06 | 68 | 53/107/302/863 | cheirality_failed | support | support | **solved** | ambiguous | ambiguous |
| 18 | v02-v07 | 17 | 16/32/145/347 | cheirality_failed | support | support | support | ambiguous | ambiguous |
| 19 | v03-v04 | 63 | 62/78/204/504 | cheirality_failed | **solved** | support | **solved** | ambiguous | ambiguous |
| 20 | v03-v05 | 51 | 47/59/159/388 | cheirality_failed | support | support | support | **solved** | ambiguous |
| 21 | v03-v06 | 24 | 22/33/93/226 | cheirality_failed | support | support | support | support | support |
| 22 | v03-v07 | 6 | 6/10/55/110 | cheirality_failed | support | support | support | support | support |
| 23 | v04-v05 | 140 | 123/185/482/1272 | cheirality_failed | solve_refused | solve_refused | solve_refused | ambiguous | cheirality_failed |
| 24 | v04-v06 | 88 | 79/130/381/1106 | cheirality_failed | **solved** | **solved** | **solved** | ambiguous | ambiguous |
| 25 | v04-v07 | 39 | 30/57/172/440 | cheirality_failed | support | support | **solved** | ambiguous | ambiguous |
| 26 | v05-v06 | 159 | 145/210/590/1635 | cheirality_failed | solve_refused | solve_refused | ambiguous | ambiguous | cheirality_failed |
| 27 | v05-v07 | 77 | 63/114/258/675 | cheirality_failed | solve_refused | solve_refused | ambiguous | ambiguous | ambiguous |
| 28 | v06-v07 | 63 | 56/98/302/771 | cheirality_failed | **solved** | **solved** | **solved** | ambiguous | ambiguous |

(`support` = `insufficient_one_to_one_support`, `spatial` =
`low_spatial_spread`, `solve_refused` = `solve_pair_refused`, `ambiguous` =
`ambiguous_hypothesis`.)

## 7. Sensitivity reading (causal claims retracted)

**The unchanged solver still fails at every swept proximity tolerance.** Solve
counts are 4/28 (surface, 0.0625), 3/28 (0.05), 7/28 (0.125), 2/28 (0.25) and
4/28 (0.5), and 21-26 of the 28 pairs refuse at every arm. That is the whole
achieved result.

This is a single-fixture *sensitivity* measurement, not a causal experiment.
Tightening the bound to 0.05 removes support (1,068 of the 1,213 surface pairs),
while loosening it adds same-surface-but-different-continuous-point edges; across
arms the candidate count, the achieved surface precision, the achieved proximity
denominator and the solver's support/spatial/ambiguity gates all move together.
The sweep therefore cannot separate the contribution of within-face correspondence
ambiguity from those co-moving effects, and this evidence **neither confirms nor
rules out** within-face correspondence ambiguity as a cause of the DS-R5K/DS-R5N
failures. It also does not causally localize a single remaining bottleneck.

The only statements this artifact supports are:

* the unchanged selector/solver still fails on this fixture at every swept
  tolerance (2-7 of 28 pairs solve; 21-26 of 28 refuse per arm);
* solve success is non-monotone in the bound (3/28 at 0.05, 7/28 at 0.125, 2/28
  at 0.25, 4/28 at 0.5), so more candidate edges do not translate monotonically
  into more solves;
* all 20 solved assisted-oracle rows have a defined pose and select the
  ground-truth yaw class, with per-solve relative rotation error between 0 and
  1.708e-06 deg and per-solve gauge-aligned baseline error between 0.0143 and
  1.335 deg; the single production-baseline solve chose the wrong quadrant
  (180 deg). No universal sub-microdegree or sub-0.4-deg bound is claimed.

The earlier wording of this section read the sweep causally ("within-face
correspondence ambiguity therefore does not explain the R5K/R5N failures", "the
evidence localizes the remaining bottleneck to ... the support, spatial and
ambiguity gates"). Those readings are **retracted**: they exceeded what a
one-fixture tolerance sweep with jointly changing candidate count, achieved
surface precision and per-arm denominators can establish. The swept quantity is
continuous raycast world-hit proximity on a shared surface; it is not a literal
feature-identity test. This experiment does not establish exact feature identity:
a literal identity claim would require independent common-point labels or a
justified localization-error model, neither of which exists in this diagnostic.

No production threshold, gate, matcher, solver, schema or promotion setting was
changed and none should be inferred from this single fixture.


## 8. Reproduction and hashes

Commands, artifact hash, self-seal and `consistency.ok` are recorded in section 5.
File hashes after the 2026-10-04 wording/sensitivity remediation:

* `scripts/evaluation/evaluate_exact_point_oracle.py`
  `afe3ebc4533e2c235be21ee213b78b8f13714e9d51b4409594ae239a9664883e`
* `tests/test_exact_point_oracle.py` (unchanged; 30 passed)
  `ebb86af53fad6bf41617a96a07002126aedeb884a2f8214397c8a43915b58daf`
* `docs/reports/image-only-geometry-exact-point-oracle-2026-10-04.md`: this file
  cannot carry its own post-edit digest; its final sha256 is recorded in the task
  result and the ASCII `AGENT_COMMS.md` entry.
* `.tmp-exact-point-oracle-20261004.json` (sealed run artifact, unchanged)
  `03faaefe40ac1496ee6f86800818f070bb3db043984d092025a8dff67c65007d`
* reviewed sealed DS-R5K/DS-R5N artifacts: neither read nor written (path-only
  output guard; the runner never opens them).

Reused reviewed sources: `scripts/evaluation/evaluate_oracle_candidate_graph.py`,
`scripts/evaluation/evaluate_support_sensitivity.py`,
`src/mc_photogrammetry/evaluation_pair_correspondence.py`,
`src/mc_photogrammetry/pipeline/joint_matching.py`,
`src/mc_photogrammetry/pipeline/correspondence.py`.

## 9. Limitations

* Evaluation-only truth-assisted diagnostic: not a production result, not a
  model-performance estimate and not a generalization claim.
* One canonical 8-view export, one run. Only three topology groups exist and every
  index split is `train`.
* The two non-production arms deliberately change graph construction while
  detection, cameras, gates, ratio test, selector and solver stay fixed; no
  production or calibration claim follows from this graph-only contrast.
* Single-fixture **sensitivity only**: candidate count, achieved surface
  precision, each arm's own proximity denominator and the resulting gates move
  together across the sweep, so this evidence neither confirms nor rules out
  within-face correspondence ambiguity as a cause and does not causally localize a
  remaining bottleneck.
* The sweep measures continuous world-hit proximity, not feature identity. This
  diagnostic provides no independent common-point labels or justified
  localization-error model, so a literal exact-feature-identity claim is not
  supported; and no universal sub-microdegree or sub-0.4-deg pose-error bound
  holds for every solved row (per-solve gauge-aligned baseline error reaches 1.335
  deg at the 0.5 bound).
* The assisted factory camera is an oracle input; normal reconstruction never sees
  it.
* Candidate/selected precision and recall are scored against the evaluator's
  unique-pixel-pair surface model, not independent feature-point or texture
  labels. Joint hypotheses are not truth correspondences.
* Same-surface edges are not feature identity; the 1.0 candidate/selected
  precision of each oracle arm against its own proximity rule is tautological,
  which is why the fixed 1,213 surface denominator is reported beside the
  own-denominator table.
* Partial block models occlude as full cubes but are excluded from truth
  ownership; 20,303 of 25,834 keypoints carry no registered surface and are
  censored, never scored.
* Truth and returned identities deduplicate only keypoints at exactly one pixel;
  near-duplicate keypoints at neighbouring pixels are not collapsed. Pair
  hypotheses sharing one pixel map to one canonical index, which can only
  under-count oracle reach.
* No plane-graph/axis or placement record exists at this pair-level seam; those
  stages are reported as typed unavailable, never zero.
* 21-26/28 pairs refuse at every arm; failure classes are attributed from the
  sealed per-pair typed reasons and gates, not from a full per-pair causal trace.

## 10. Review remediation (2026-10-04, DeepSeek)

Luna HOLD `R5O inference and metric wording`, extended by the Lead reviewer
addendum `R5O factual corrections`, required the following changes on the existing
sealed run; no replay, rerun or artifact edit was performed:

1. **Wording.** The script docstrings, `oracle_policy` definitions, arm
   descriptions and this report now describe the swept quantity as **same-surface
   continuous world-hit proximity at each tolerance**, explicitly not literal
   feature identity. The sealed schema string, output filename and
   `point_oracle_*` arm keys are unchanged and are documented as legacy labels
   retained for artifact continuity.
2. **Causal claims retracted.** Section 7 is rewritten as a sensitivity reading:
   solves vary non-monotonically with the swept bound, refusals persist at every
   arm, and the evidence neither confirms nor rules out ambiguity as a cause nor
   localizes one remaining bottleneck.
3. **Own-denominator table added.** Section 6.1 now reports, beside the fixed
   1,213 surface-denominator table, the per-arm precision/recall from the
   artifact's immutable `candidate_stage_arm_tolerance` and
   `selected_stage_arm_tolerance` fields, i.e. against each arm's own
   same-surface-within-tolerance denominator. `production_baseline` has no such
   stage and is reported as typed unavailable.
4. **Artifact preserved.** The sealed artifact was not read as input, rewritten or
   re-run. Read-only verification: stored `seal_sha256`
   `5d8a67edbb6aeb1c0dadc998f1ffa2610718e5cd36e84a5a98c11c0ab5e73899` recomputed
   with `evaluation_pair_correspondence.seal_record` matches exactly;
   `consistency.ok = true` with zero positive-identity mismatch; artifact file
   sha256 remains
   `03faaefe40ac1496ee6f86800818f070bb3db043984d092025a8dff67c65007d`.
5. **Boundary correction.** Section 4 now states the tolerance boundary is
   `0.25 + 1e-6` over refused, not "one ulp".
6. **Pose-error generalization retracted and recomputed from the artifact.**
   Section 6.3 and the section 7 bullet no longer claim a universal
   sub-microdegree / sub-0.4-deg result for every solve. Recomputed from the
   immutable per-pair `pose_error` fields: all 20 solved oracle rows have a
   defined pose and the correct yaw class; per-solve relative rotation error spans
   0 to 1.708e-06 deg and per-solve gauge-aligned baseline error spans 0.0143 to
   1.335 deg (max at 0.5, `v02-v03`). The per-arm medians in the section 6.3
   table are unchanged and verified against the artifact.
7. **Identity claim.** Sections 7 and 9 now state explicitly that the sweep does
   not establish exact feature identity; a literal identity claim would require
   independent common-point labels or a justified localization-error model, none
   of which this diagnostic provides.

Verification after remediation: `py_compile` on both owned Python paths exit 0;
`tests/test_exact_point_oracle.py` 30 passed (unchanged test file). No production,
threshold, gate, solver, schema or promotion change; no second experiment.

## 11. Stop condition

Stopped after the focused tests, the single pinned run, artifact hashes and this
report for Luna verification and Astra acceptance. No production, threshold, gate,
solver, schema or promotion change; no second run; no prior sealed artifact read
or written; no recursive deletion, reparse/junction operation, encoded command,
background job, generated executable, installer/download, persistence,
Defender/system change or commit.

