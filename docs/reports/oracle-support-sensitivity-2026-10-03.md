# DS-R5N oracle support-floor sensitivity (2026-10-03)

Status: **implementation and preregistration only; no corpus evaluation run.
Evaluation-only; no production or promotion change.**

DS-R5M forwarded the computed `support_floor` on the no-mode refusal branch. DS-R5N
now preregisters a *separate* evaluation-only diagnostic that keeps the DS-R5K
canonical 8-view Canal Town entry, immutable DS-R5K pins, strict
discovery/leakage preflight and truth-surface oracle graph, and adds one explicit
**oracle-only** control `min_support_absolute = 8, min_support_fraction = 0.0` to
measure how many currently unreachable pairs gain a candidate mode, a one-to-one
selected support set and a DS-11 solve. It does not run the corpus, does not read
or write the sealed DS-R5K artifact, and does not change any threshold, gate,
matcher, solver, project or promotion setting.

## 1. Scope and owned paths

DS-R5K reached all 1,213 Canal Town truth pixel pairs but selected only 52; its
report section 5.2 reconstructed that 19/28 pairs had a whole oracle surface-pair
edge count below the production floor `max(8, ceil(0.02 * min(raw_keypoints)))`
and that the four `solve_pair_refused` pairs had no serialized DS-11 per-yaw
`YawClassScore` evidence. DS-R5N is the smallest evaluation-only follow-up that
(a) removes only the *fractional* half of the floor on the oracle graph and
(b) serializes the missing pose evidence, without touching production.

| Path | State | sha256 | Notes |
| --- | --- | --- | --- |
| `scripts/evaluation/evaluate_support_sensitivity.py` | updated (DS-R5N-R1) | `6dd69e3bca3e3439e1fac02aee7b20e58857491809b386ab8ec7cef5d93dd253` | one-shot three-arm runner; case-insensitive sealed-artifact guard |
| `tests/test_support_sensitivity.py` | updated (DS-R5N-R1) | `5b17192941e843806ae886db7d10fd3bd6558d782fe7a64804adb3554c50ad06` | focused tests; mixed-case guard regression |
| `docs/reports/oracle-support-sensitivity-2026-10-03.md` | this file | — | preregistration + implementation/test record |
| `.tmp-oracle-support-sensitivity-20261003.json` | reserved; not created | — | sealed output path; confirmed absent |
| `.tmp-oracle-candidate-graph-20261003.json` | untouched | `775e006f369aec7d7b5a76545f184f3f138007ffed9537c7f9306c03deb0db84` | sealed DS-R5K artifact; never read/written |

No file under `src/` was modified. Existing scripts, tests, sealed prior
artifacts and the pre-existing dirty work tree were preserved. Current reusable
production hashes: matcher
`src/mc_photogrammetry/pipeline/joint_matching.py`
`9303f56ab194600c58a27d97321fa2f2e4323e4bb78276a2df0372e8ed8ae4e3`; evaluator
`src/mc_photogrammetry/evaluation_pair_correspondence.py`
`18a0b3a60394c7bbc63e3f3d4a1eb24511d5164c35b5d80629d0c9a78cdef6de`;
correspondence `src/mc_photogrammetry/pipeline/correspondence.py`
`192a4851e428d8116e0fbffd28c7ff157d1aedd8035e499c6df7ec2a9d787fa1`; reused DS-R5K
script `scripts/evaluation/evaluate_oracle_candidate_graph.py`
`eabee92286e90481b64ff10e599b67d0dc90a1407d0fe76ad700008d34b60726`.

## 2. Preregistered design

### 2.1 Reused DS-R5K machinery

The new script loads `scripts/evaluation/evaluate_oracle_candidate_graph.py` by
path (never its sealed output) and reuses, unchanged:

* `select_canonical_entry` / `canonical_entry_record` (Canal Town-scoped
  deterministic canonical selection);
* `observe_oracle_frame` / `oracle_camera_estimate` (production joint detection
  plus the assisted factory-camera estimate);
* `build_oracle_graph` (the strict same-world-surface unique-pixel-pair graph
  with real descriptor distances);
* `injected_candidate_graph` (temporary candidate-graph injection restored in a
  `finally` block);
* `detection_settings`, `describe_outcome`, `hypothesis_record`, `frame_record`,
  `camera_record`, `geometry_record` and the `ScoreTotals` helpers.

Canonical selection resolves to `fresh-c4-canal-town-framed-2026-09-12`
(8 frames, 28 unordered pairs) exactly as in DS-R5K.

### 2.2 Immutable DS-R5K pins (preflight)

Before any image processing `main` runs, in order:

1. `discover_finalized_entries(exports_root)` - re-hashes every listed scene-file
   and screenshot for every discovered `index.json`;
2. `verify_r5k_pins(entries)` - the discovered dataset-id set must equal the five
   DS-R5K ids and every index sha256 must equal the recorded R5K index pin
   (`fresh-c4-canal-town-framed-2026-09-12` `0637caba…`, `fresh-c4-canal-town-2026-09-12`
   `d3dc69b0…`, `recon-canal-houses-2026-09-12` `5d0db614…`,
   `pilot-batch-train-station-2026-09-12` `4ef40f8b…`,
   `smoke-train-station-ds20c-2026-09-12` `a2a21eb4…`);
3. `build_group_manifest(entries)` - rejects split leakage and fold leakage;
4. `select_canonical_entry(entries, dataset_id=…)` - the Canal Town-scoped
   canonical choice, with the DS-R5K override guard;
5. `verify_canonical_scene_pins(canonical)` - the canonical export's five
   scene-file pins (`capture.json`, `ground_truth.json`, `plan.json`,
   `recipe.json`, `reference.litematic`) and its eight screenshot pins must match
   the immutable DS-R5K pins exactly.

Any preflight failure prints a diagnostic and exits `1` **without writing any
output**. This is intentionally stricter than DS-R5K: corpus drift now refuses
instead of silently evaluating different inputs.

### 2.3 Three arms, one oracle-only control

Per unordered frame pair `attribute_pair` runs:

| Arm | Candidate graph | Settings |
| --- | --- | --- |
| `production_baseline` | production reciprocal top-k descriptor graph | untouched production defaults |
| `oracle_default` | truth-surface oracle graph | untouched production defaults (`min_support_fraction = 0.02`) |
| `oracle_control` | the *same* truth-surface oracle graph | `min_support_absolute = 8`, `min_support_fraction = 0.0` |

The production arm is complete before any injection, so it can never see the
oracle graph. The two oracle arms share one graph; only the support-floor
settings differ. The control is **oracle-only** and is a measurement instrument,
not a candidate production setting.

`min_support_absolute` stays at the production value of 8 in the control; only
the fractional term is removed. The support floor the matcher computes remains
`max(min_support_absolute, ceil(min_support_fraction * min(raw_keypoints)))`, so
the default arm's floor is unchanged and the control arm's floor is the absolute
8.

### 2.4 Per-pair serialization

Each sealed pair record reports, for every arm: the candidate `graph_edges`
count, `candidate_edges`/`total_graph_edges`/`truncated`, `raw_keypoints`,
computed `support_floor`, one-to-one selected `support` (and its
`selected_support_edges` canonical index pairs), `spatial_bins`, every ranked
joint `hypotheses` entry with its own gate verdict, the full gate diagnostics,
`status`/`reason`/selection/pose states, the resolved `pose_class`, and:

* `pose_classes` - **every field of every DS-11 `solve_pair` `YawClassScore`**
  (`yaw_class`, `passed`, `refusal`, `inliers`, `informative`, `inlier_fraction`,
  `median_rho_degrees`, `parallax_degrees`, `cheirality_fraction`,
  `singular_ratio`, `singular_values`) when the pose stage ran, else empty;
* `pose_error` - the assisted-camera ground-truth yaw/baseline error where
  technically well-defined.

The pair also carries the shared `oracle_graph_edges` (canonical index pairs) and
a `comparison` block: default-vs-control solved flags, refusal reasons,
hypothesis/support/spatial deltas and the booleans
`control_gained_candidate_mode`, `control_gained_selection`,
`control_gained_solve`, `control_lost_solve`. No pixel coordinate, image byte,
screenshot path or selected-match pixel row is serialized.

### 2.5 Assisted-camera pose error (when defined)

The assisted factory cameras are the ground truth, so:

* `true_yaw_class` is the quadrant `q` minimising the rotation error between
  `R_right @ Ry(90 q) @ R_left.T` and the assisted relative rotation
  `R_right @ R_left.T`, mirroring `scripts/diagnose_joint_pairs.py`;
* `relative_rotation_error_degrees` compares the solved DS-11 relative rotation
  with that assisted relative rotation (`solve_pair` carries the class label into
  its rotation), and `yaw_class_correct` compares the solved class with
  `true_yaw_class`;
* `baseline_error_degrees_gauge_aligned` / `_without_gauge` / `_signless` compare
  the solved unit baseline direction with the assisted world baseline
  `C_right - C_left` after removing the four-fold world yaw gauge;
  `gauge_index` and `true_baseline_length_blocks` are recorded.

`defined = true` only when the pose stage solved and produced a relative rotation;
the baseline half is separately flagged by `baseline_error_defined` with a typed
`baseline_undefined_reason`. When the pose stage never ran,
`undefined_reason = "pose_not_attempted"`; when it abstained,
`undefined_reason = "pose_unsolved"`.

### 2.6 Same surface is not the same feature point

> A strict same-world-surface oracle edge means its two endpoints observe the
> same block face within the preregistered tolerance; it does **not** mean they
> are the same feature point. Distinct keypoints on one surface are different
> physical points. Relaxing the support floor therefore admits
> same-surface-but-different-point correspondences, so a control-arm candidate
> mode, selection or solve is **not** evidence of a correct feature
> correspondence and must never be read as a reason to lower the production
> support floor.

This caveat is carried verbatim in the sealed payload
(`oracle_policy.same_surface_not_same_feature_point`,
`summary.same_surface_not_same_feature_point` and the limitations list) and in
these tests. No promotion, production threshold change or project decision may
be made from this diagnostic.

### 2.7 Sealed output and guards

The reserved output is `.tmp-oracle-support-sensitivity-20261003.json`. The
runner refuses to overwrite an existing output (exit `2`), refuses to write to
the sealed DS-R5K artifact path (exit `2`), and on any preflight/processing error
prints and exits `1` without writing. **DS-R5N-R1 remediation:** the sealed
DS-R5K path guard is now Windows-case-insensitive. `output_targets_sealed_r5k_artifact`
compares absolute, lexically collapsed (`abspath` + `normpath`, no filesystem
access) and case-folded path keys, so a mixed-case spelling such as
`.Tmp-OrAcLe-CaNdIdAtE-GrApH-20261003.JsOn` is refused exactly like the canonical
spelling; the previous raw `name`/`endswith` string comparison admitted it on a
case-insensitive filesystem. On success the runner writes one JSON record with
schema `mc-photogrammetry-oracle-support-sensitivity-v1`, camera/oracle/arm
policies, exact default and control parameters, the hash-pinned inputs, the
recorded R5K pin verification, per-frame truth coverage, the per-pair ledger, the
pooled default-vs-control summary, the renderer/geometry policy, an internal
`seal_sha256` self-seal and the limitations list.

## 3. Focused tests and exact commands

Run from the repository root before and independently of any corpus evaluation:

```
.\.venv\Scripts\python.exe -m py_compile scripts\evaluation\evaluate_support_sensitivity.py tests\test_support_sensitivity.py
→ exit 0

$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m pytest -o addopts='' -q tests\test_support_sensitivity.py
→ 20 passed in 6.06s
```

DS-R5N-R1 ran only these two commands (focused file plus `py_compile`), never the
corpus/evaluator, and created no output: the reserved
`.tmp-oracle-support-sensitivity-20261003.json` remains absent and the sealed
`.tmp-oracle-candidate-graph-20261003.json` remains byte-identical at
`775e006f369aec7d7b5a76545f184f3f138007ffed9537c7f9306c03deb0db84`.

Coverage:

* **Default-path / floor-setting isolation** - `control_settings()` is exactly
  `(8, 0.0)`, `default_settings()` is the untouched `(8, 0.02)`, the two objects
  are distinct, running all three arms leaves `JointMatchSettings()` equal to its
  pre-run defaults, and the production `reciprocal_topk_graph` is restored.
* **Control behaviour** - a synthetic pair with 20 exact truth edges and 1,020
  keypoints: the default floor is 21 and refuses with
  `insufficient_one_to_one_support`, while the control floor of 8 produces a
  candidate mode, a selected support and a DS-11 solve. Arm isolation is checked
  by asserting each arm's `candidate_edges` equals its own graph's edge count.
* **Per-pair serialization** - every required arm field is present, raw keypoints
  are correct, `selected_support_edges` matches the support size, spatial bins
  have the expected shape, and the shared oracle graph edges are serialized.
* **Per-yaw pose serialization** - the solved control arm reports
  `pose_classes` for all four classes with exactly the eleven `YawClassScore`
  fields, the resolved `pose_class` matches the pose, and the assisted
  ground-truth yaw class is correct.
* **Assisted-camera pose error** - defined rotation/baseline errors, gauge index
  and true baseline length for a solved pose; a typed
  `pose_not_attempted` record and empty `pose_classes` for a refusal where the
  pose stage never ran.
* **No pixel payload** - a recursive scan finds no pixel/match/image key in the
  assembled payload and the payload is JSON-safe.
* **Deterministic self-seal** - two payload builds produce identical
  `seal_sha256`; `seal_record` reproduces it; tampering with a field changes it.
* **Immutable pins** - the five-entry DS-R5K index pin set is accepted, a
  tampered index hash and a short set are rejected, and the canonical
  scene/screenshot pins are required exactly.
* **Guards** - the sealed DS-R5K artifact path is refused (exit `2`), including a
  **mixed-case** spelling that differs only by case under a `tmp_path` directory
  (DS-R5N-R1 regression; the test never reads or writes the real sealed file), an
  existing output is not overwritten (exit `2`), pin failure and leakage failure
  exit `1` with no output, a successful write is refused on a second run, and a
  missing export root exits `1` with no output.
* **Path reservation** - `DEFAULT_OUTPUT` is
  `.tmp-oracle-support-sensitivity-20261003.json`, distinct from and not
  containing the DS-R5K artifact path.

## 4. Corpus result (single authorized foreground run)

**Completed once; lead review pending.** The lead-authorized run used the exact command below, returned exit code `0` (DeepSeek-reported), and took approximately 70 seconds (DeepSeek-reported; the recorded process/artifact timestamps are rounded to one second). A Windows venv launch showed the `\.venv\Scripts\python.exe` parent and base Python child; a separate harmless probe reproduced that same parent-child launch pattern, so the two process IDs observed during execution were one invocation, not duplicate evaluations.

```
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\evaluation\evaluate_support_sensitivity.py
```

Artifact `.tmp-oracle-support-sensitivity-20261003.json`: 1,695,546 bytes; SHA-256 `99e40063f54129821c9ae3aa53788541d24494bc269c8a50c2154a0c78d2af27`; self-seal `ee36f6eb9561835183f13c8b4f5d7e4f3f36529c07ad546e36ed0bc06923d698` independently recomputes exactly. `consistency.ok = true`; all three stages have zero positive-identity mismatch. It selects `fresh-c4-canal-town-framed-2026-09-12` (8 frames, 28 unordered pairs, 1,213 unique surface pixel-pair truth edges). All five R5K index pins and the canonical scene/screenshot pins report `all_match = true`; split/fold leakage lists are empty. The R5N `.partial` path is absent. The DS-R5K artifact remains byte-identical at SHA-256 `775e006f369aec7d7b5a76545f184f3f138007ffed9537c7f9306c03deb0db84`, and its self-seal remains valid.

| Arm | Selected returned | Surface-proxy recall | Surface-proxy precision | Solved pairs | Refusals |
| --- | ---: | ---: | ---: | ---: | --- |
| Production reciprocal graph + default settings | 8,863 | 52/1,213 (4.29%) | 52/8,863 (0.587%) | 1/28 | 27 `cheirality_failed` |
| Oracle surface graph + default settings | 865 | 865/1,213 (71.31%) | 865/865 (100%, by construction) | 4/28 | 19 `insufficient_one_to_one_support`; 1 `low_spatial_spread`; 4 `solve_pair_refused` |
| Same oracle graph + control floor `(8, 0.0)` | 1,152 | 1,152/1,213 (94.97%) | 1,152/1,152 (100%, by construction) | 0/28 | 9 `insufficient_one_to_one_support`; 7 `low_spatial_spread`; 12 `ambiguous_hypothesis` |

The oracle-only control produced candidate modes and selected support on 10 of the 19 default `insufficient_one_to_one_support` pairs, but **none solved**. It also caused all four pairs solved under `oracle_default` (pairs 14, 19, 24 and 28) to refuse earlier as `ambiguous_hypothesis`; no control pair reached DS-11 `solve_pair`. The relaxed floor therefore increases surface-proxy reach but makes pair-pose output worse on this fixture. Do **not** lower the production support floor from this control.

All four `oracle_default` solves selected assisted-camera yaw class 0 correctly. Their median relative-rotation error is about `0.0000006 degrees`, and median four-fold-yaw-gauge-aligned baseline-direction error is `0.0638 degrees`. This validates the diagnostic's assisted pose-error fields for those four pairs only; it does not establish image-only camera recovery.

**Interpretation limit:** `100%` control/default oracle precision is tautological against the evaluator's same-surface labels. A same-world-surface pair does not guarantee the same texture feature point, so the control's increased selected reach is not exact correspondence precision, not a model result, and not a production accuracy estimate. No production threshold, gate, matcher, solver, project, promotion setting or prior sealed artifact was changed. The complete pair ledger and all per-yaw `YawClassScore` details are in the sealed JSON.

## 5. Limitations

* **Evaluation-only counterfactual.** The oracle candidate graph is a
  truth-assisted upper bound, not a production result and not a
  model-performance estimate.
* **Same surface is not the same feature point** (section 2.6). Every
  control-arm reading is governed by this caveat; the control is never a
  promotion argument.
* **Assisted camera.** The factory capture pose, field of view and frame
  resolution are an evaluation aid; normal reconstruction never sees them and no
  image-derived calibration claim is made.
* **One export, one run.** Only the canonical 8-view Canal Town export is
  evaluated; every index split is `train` and only three topology groups exist,
  so no generalization claim is made.
* **Surface-model metrics.** Precision/recall are scored against the
  evaluator-built unique-pixel-pair surface model, not independent
  feature-point labels; partial block models occlude as full cubes but are
  excluded from truth ownership.
* **No texture identity** exists anywhere in the pipeline;
  `same_state_same_slot_other_block` is only an appearance proxy.
* **Upper-bound candidate reach.** The oracle edges keep real descriptor
  distances, but the descriptor graph itself is replaced.
* **No production change.** No matcher, threshold, gate, solver, project or
  promotion setting was changed; the production graph builder is restored in a
  `finally` block after the oracle calls.

## 6. Stop condition

Stopped for lead review after the one authorized corpus/evaluator run. The sealed output is preserved; no production edit, threshold/gate change or promotion decision was made. No
background process/job/workflow, subagent, shell nesting, junction/reparse point,
recursive deletion, generated executable, install/download, Minecraft/capture
action, Defender change or system write was used, and the pre-existing dirty work
tree was preserved.
