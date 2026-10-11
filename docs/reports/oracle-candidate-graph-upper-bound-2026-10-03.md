# DS-R5K truth-filtered oracle candidate-graph upper bound (2026-10-03)

Status: **single authorized corpus evaluation complete; artifact is sealed and
reviewed. Evaluation-only upper bound; no production changes.**

This revision is the lead-required DS-R5K **override-guard remediation**: the
`--dataset-id` override is now restricted to the same canonical Canal Town
scene-prefix candidate set and must match exactly one of those candidates, so a
non-Canal (Train Station) dataset ID is rejected; `canonical_entry_record` reports
that same scoped candidate set.

Evaluation-only. No production module, threshold, matcher, gate, solver, project or
promotion decision was changed. The single authorized run and its output are recorded
in section 5.1. No capture, Minecraft/client action, installation, job/workflow,
junction/reparse point, recursive deletion, generated executable, background process,
Defender/system change or system write was used.

Sections 1-4 record the preregistration and implementation/test evidence. Section
5.1 records the one authorized foreground corpus run; no retry or overwrite occurred.

## 1. Scope and owned paths

DS-R5J measured a candidate-reach loss in the Canal Town group: the reciprocal
top-k candidate graph reached 345/1,215 truth pairs while the joint selector kept
only 52, losing 293/345 = 84.93% of candidate-reachable truth pairs (lead-verified
stage facts, joined-matcher report section 8.4). DS-R5K isolates that gap with a
deliberately truth-assisted counterfactual on a single canonical Canal Town export:
run the **unchanged** production
`pipeline.joint_matching.match_feature_frames` twice per unordered frame pair, once
on the production reciprocal descriptor graph and once on an oracle graph whose
edges are exactly the strict same-world-surface truth pairs. The oracle graph is an
upper bound, not a production result.

| Path | State | sha256 | Notes |
| --- | --- | --- | --- |
| `scripts/evaluation/evaluate_oracle_candidate_graph.py` | created | `eabee92286e90481b64ff10e599b67d0dc90a1407d0fe76ad700008d34b60726` | one-shot runner + oracle diagnostic |
| `tests/test_oracle_candidate_graph.py` | created | `ac88a3978fe052306a8257f42341d3b0019857deb9e5e3fa66d0826c41b49290` | focused tests |
| `docs/reports/oracle-candidate-graph-upper-bound-2026-10-03.md` | this file | — | preregistration + implementation/test record |
| `.tmp-oracle-candidate-graph-20261003.json` | created once after authorization | `775e006f369aec7d7b5a76545f184f3f138007ffed9537c7f9306c03deb0db84` | sealed evaluation artifact; results in section 5.1 |

For the DS-R5K evaluation run, no file under `src/` and no existing evaluation
script/test was modified. The DS-R5I / DS-R5I-R1 / DS-R5J sealed artifacts remain
untouched.

## 2. Preregistered design

### 2.1 Canonical Canal Town entry

Only one scene export is evaluated. `select_canonical_entry` always scopes the
choice to entries whose `scene_id` starts with the `canal-town` prefix, then takes
the **most complete** export (largest screenshot/frame count), tie-broken by
`(scene_id, directory, dataset_id)`; an optional `--dataset-id` override must name
**exactly one entry within that same Canal Town scoped set** and is rejected for any
non-Canal entry (for example a Train Station export). `canonical_entry_record`
records that same scoped candidate set as `considered_entries`, so an override can
never widen the evaluated scope to a non-Canal export. In the frozen corpus, static
inspection of the two Canal Town `index.json` files (both `scene_id
canal-town-architectural-component-bbox`, group `74f080aab201…`) shows 4 frames for
`fresh-c4-canal-town-2026-09-12` and 8 frames for
`fresh-c4-canal-town-framed-2026-09-12`, so the rule resolves to the 8-frame framed
export. The sealed payload records the rule and every considered candidate, so the
choice is auditable. No index or screenshot hash was re-verified in this session
(no corpus run).

### 2.2 Assisted oracle-camera policy (evaluation-only)

The matcher's geometric stage consumes `CameraEstimate` records; this diagnostic
builds them from the **exact factory capture metadata** (recorded actual pose with
planned fallback, factory FOV and frame resolution via `intrinsics_for`). Every
frame record carries `camera_source = "oracle_factory_capture_metadata"` and
`assisted = true`; the sealed payload carries a top-level `camera_policy` with
`assisted = true` and `default_image_only_reconstruction = false`. Feature
extraction still reads only pixels and the authoritative usable mask; the oracle
cameras reach the matcher's geometric stage alone. Normal reconstruction remains
image-only and this diagnostic makes no image-derived calibration claim.

Joint detection mirrors production `match_and_solve_pair`:
`CorrespondenceSettings(max_features = joint.max_features = 4000,
mask_boundary_pad_px = joint.mask_boundary_pad_px = 8.0)` with
`ratio_test = 0.75`, so the descriptor-footprint boundary drop is the production
one.

### 2.3 Oracle candidate graph (truth-assisted, evaluation-only)

`build_oracle_graph` constructs a `ReciprocalGraph` whose edges are exactly the
unique `(left pixel, right pixel)` strict truth pairs of the frozen DS-R5I rule
(`construct_truth_pairs`: same `(block coordinate, face slot)`, world surface
points within `SURFACE_TOLERANCE_BLOCKS = 0.0625`), mapped to their **canonical
first keypoint index** in each detected point array:

* repeated keypoints at exactly one pixel collapse to one edge; distinct pixels
  stay distinct;
* each edge keeps the **actual L2 descriptor distance** of the mapped keypoint
  pair, so the matcher's descriptor-cost ordering and diagnostics remain real —
  no descriptor or distance is fabricated;
* edges are sorted by `(distance, left index, right index)`, the production
  `reciprocal_topk_graph` ordering convention;
* empty truth yields an empty graph, which the unchanged matcher reports as a
  typed `candidate_graph_empty` refusal (this is validated by tests, not an
  exception).

### 2.4 Injection and restoration

`injected_candidate_graph(graph)` is a context manager that replaces
`mc_photogrammetry.pipeline.joint_matching.reciprocal_topk_graph` for exactly one
matcher call and restores the original function object in a `finally` block, so it
restores even if the matcher raises. The baseline graph is built with the
production builder through this module's own imported reference, so the baseline
run is never affected by the injection. After every `attribute_pair` the production
builder is byte-identical to its pre-call object (asserted by tests).

### 2.5 Dual run and corrected pixel-identity scoring

Per unordered frame pair `attribute_pair`:

1. recomputes the production reciprocal graph (`reciprocal_topk_graph`) so its
   returned identities can be scored — the matcher computes the identical graph
   internally;
2. runs `match_feature_frames(...)` unchanged on that graph (baseline);
3. builds the oracle graph from truth and runs `match_feature_frames(...)`
   unchanged inside the injection context (oracle). All settings, gates and the
   ratio test are byte-identical between the two runs; only the candidate graph
   differs.

Every stage is scored with the **corrected** unique-pixel-pair evaluator
`score_pair_matches(indices, left_hits, right_hits, left_points, right_points,
tolerance=...)` from `evaluation_pair_correspondence`: all returned counts,
precision denominators and recall numerators share the unique
`(left pixel, right pixel)` identity, including rows with no truth surface and
rows whose indices are duplicated at one pixel. `PlaneGraphTrace`/hypothesis counts
are never used as match counts.

### 2.6 Per-pair ledger

Each sealed pair record reports:

* `truth_pairs` (the correspondence recall denominator) and `oracle_graph_empty`;
* `baseline_solved`, `oracle_solved`, and `oracle_available_when_baseline_refused`;
* for both runs: `status`, typed `reason`, `selection_state`, `pose_state`,
  `matches_available`, `pose_available`, graph edge count, candidate/total/truncated
  counts, support and support floor, spatial bins, median selected residual,
  descriptor cost, parallax, cheirality fraction, singular ratio, ambiguity margin,
  the DS-11 pose facts and every ranked hypothesis with its own gate verdict;
* five stage geometries (`mutual_stage`, `baseline_candidate_stage`,
  `oracle_candidate_stage`, `baseline_selected_stage`, `oracle_selected_stage`)
  with returned/positive/appearance-proxy/wrong-surface/unattributed counts,
  precision and recall;
* explicit `denominators` (geometry counts per stage, correspondence truth pairs)
  and `reach` (truth pairs covered by mutual, baseline candidate, oracle candidate,
  baseline selected, oracle selected).

The pooled `summary` aggregates the same split, plus solved counts,
`oracle_solved_where_baseline_refused`, `oracle_lost_where_baseline_solved`, typed
refusal-reason histograms for both runs, selection/pose state histograms, and a
`positive_identity_mismatch` per stage; `consistency.ok` is true only when every
mismatch is zero.

## 3. Sealed output

The runner refuses to overwrite an existing output (exit 2). On success it writes
one sealed JSON record with schema
`mc-photogrammetry-oracle-candidate-graph-upper-bound-v1` containing the camera and
oracle policies, the exact joint/detection parameters, the hash-pinned inputs and
group manifest (all discovered entries are hash-verified before the canonical one
is selected), the canonical-entry decision and candidates, per-frame truth
coverage, the per-pair ledger, the pooled summary, the renderer/geometry policy,
an internal `seal_sha256` self-seal and the limitations list. It makes no
production, model-performance or generalization claim.

## 4. Focused tests and exact commands

Focused suite and `py_compile`, run before and independently of any corpus
evaluation:

```
.\.venv\Scripts\python.exe -m py_compile scripts\evaluation\evaluate_oracle_candidate_graph.py tests\test_oracle_candidate_graph.py
→ exit 0

$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m pytest -o addopts='' -q tests\test_oracle_candidate_graph.py
→ 28 passed in 2.80s

$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m pytest -o addopts='' -q tests\test_oracle_candidate_graph.py tests\test_joint_matcher_stage_eval.py tests\test_evaluation_pair_correspondence.py
→ 84 passed in 3.74s
```

Coverage:

* **Oracle graph construction** — unique truth pixel pairs become exactly one edge
  each; the graph type/endpoint counts/truncation are exact; edges are sorted by
  `(distance, left, right)`.
* **Descriptor-distance alignment** — every edge distance equals the actual L2
  descriptor distance of its mapped keypoint pair and the mean is non-trivial.
* **Duplicate feature coordinates** — two keypoints at one pixel/surface collapse
  to one truth pair and one canonical `(0, 0)` edge whose distance uses the first
  index, not the duplicate.
* **Unmappable truth** — a truth pixel with no detected keypoint raises the typed
  `OracleCandidateGraphError` rather than silently dropping evidence.
* **Empty truth** — an empty graph is built without raising; the unchanged matcher
  refuses with `candidate_graph_empty`, no selection and `oracle_solved = false`.
* **Graph injection/restoration** — the production builder is replaced inside the
  context and restored afterwards; restoration also holds after an exception; an
  end-to-end `attribute_pair` leaves the production builder restored.
* **Exact same synthetic surface** — both runs solve; the oracle candidate graph
  reaches the whole truth denominator (precision 1.0); the unchanged selector keeps
  all of it (selected precision/recall 1.0) and the pair record exposes the
  denominator/reach split.
* **Wrong and occluded synthetic surfaces** — zero truth denominator, zero
  positives, `null` recall and a typed empty-graph refusal.
* **Gate diagnostics** — `result_record` exposes the full gate set and hypothesis
  verdicts.
* **Pooled accounting** — empty-truth/truth pair counts, oracle empty-graph count,
  refusal histograms, reach totals, zero positive-identity mismatch, and
  `oracle_solved_where_baseline_refused` when the baseline run refuses but the
  oracle graph lets the unchanged selector solve.
* **Canonical entry** — the most complete Canal Town export is chosen, the
  `--dataset-id` override selects exactly one *Canal Town* candidate (both known
  Canal Town IDs accepted, zero/multiple rejected), a known Train Station ID is
  rejected even as the only discovered entry, `canonical_entry_record` lists only
  the same Canal Town scoped candidates under an override, and a corpus with no
  Canal Town entry raises.
* **Payload assembly** — `build_payload` seals, labels the assisted oracle camera
  and truth-assisted oracle graph, is JSON-safe, reports the oracle edge count, and
  its self-seal reproduces (scene/features injected; no corpus).
* **Runner guards** — overwrite refusal (exit 2) and failure-without-writing when
  the export root is missing (exit 1, no output).

## 5. Corpus run status

**Completed.** The run was authorized after lead review of the implementation and
tests. It re-verified factory pins and split/fold leakage before image processing.

The exact run was:

```
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\evaluation\evaluate_oracle_candidate_graph.py
```

It ran once in the foreground with no retry. `PYTHONPATH=src` is required because the venv's
`mc_photogrammetry.pth` points at a stale path. Because only the canonical entry
runs, the reported counts are **not** the pooled DS-R5J Canal Town group counts
(which combined both duplicate exports); the run locates the candidate-versus-
selected loss within one export.

### 5.1 Results

**Results (single authorized foreground run, appended 2026-10-03).** Nothing above
or below this added block was edited; no rerun or overwrite was made.

Exact command, run once in the foreground from the repository root:

```
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\evaluation\evaluate_oracle_candidate_graph.py
```

Exit code `0`; wall time `70.816 s` (process stopwatch). Preflight confirmed both
`.tmp-oracle-candidate-graph-20261003.json` and
`.tmp-oracle-candidate-graph-20261003.json.partial` absent; postflight confirmed the
output exists and the `.partial` path is still absent. No retry, no overwrite.

Artifact: `.tmp-oracle-candidate-graph-20261003.json`, 521,306 bytes, sha256
`775e006f369aec7d7b5a76545f184f3f138007ffed9537c7f9306c03deb0db84`. Self-seal field
`7812e8f02cedafb7c45d872de509850de56e6fd0912324642924666176ab0876`; recomputing
`seal_record` over the payload reproduces it exactly (`SEAL_MATCH = True`).
`consistency.ok = true` with zero `positive_identity_mismatch` at every stage.

Input admission (before any image processing): `discover_finalized_entries`
re-hashed every listed scene-file and screenshot pin and `build_group_manifest`
rejected split/fold leakage. All 5 discovered index entries admitted; sealed
`split_leakage_issues = []` and `fold_leakage_issues = []`; 5 index entries, 3
topology groups.

Canonical selection: `fresh-c4-canal-town-framed-2026-09-12`, scene
`canal-town-architectural-component-bbox`, group
`74f080aab20105733ddf845ccd1a938b76f01f00edace3a33cb9275a47cf458a`, split `train`;
**8 frames, 8 choose 2 = 28 unordered pairs**; pooled correspondence truth
denominator 1,213 unique pixel pairs; 0 empty-truth pairs, 0 empty-oracle-graph
pairs. The 4-frame `fresh-c4-canal-town-2026-09-12` duplicate was a considered
candidate and was not selected.

Pooled baseline-versus-oracle stage metrics (returned = geometry identities,
reach = `truth_pairs_returned`, truth = 1,213 unless stated):

| Stage | returned | reach | precision | recall |
| --- | ---: | ---: | ---: | ---: |
| mutual baseline | 580 | 6 | 0.010344828 | 0.004946414 |
| baseline reciprocal candidate graph | 169,586 | 345 | 0.002034366 | 0.284418796 |
| oracle truth candidate graph | 1,213 | 1,213 | 1.0 | 1.0 |
| baseline selected | 8,863 | 52 | 0.005867088 | 0.042868920 |
| oracle selected (unchanged selector) | 865 | 865 | 1.0 | 0.713107997 |

(`baseline_candidate_edges` sums to 178,056 raw graph edges across pairs; the
169,586 stage `returned` is the deduplicated unique-pixel-pair identity count. The
oracle graph contains exactly the 1,213 truth pixel pairs, so its candidate
precision/recall are 1.0 by construction.)

Per-pair ledger (`vNN` = `view-canal-town-architectural-component-bbox-NN`; figures
are `returned / reach / precision / recall`; `P=None` means no returned identities):

| # | pair | truth | baseline status | oracle status | baseline candidate | baseline selected | oracle candidate | oracle selected |
| ---: | --- | ---: | --- | --- | --- | --- | --- | --- |
| 1 | v00–v01 | 20 | refused cheirality_failed | refused insufficient_one_to_one_support | 4920/4/0.000813008/0.20 | 218/0/0.0/0.0 | 20/20/1.0/1.0 | 0/0/None/0.0 |
| 2 | v00–v02 | 4 | refused cheirality_failed | refused insufficient_one_to_one_support | 4775/1/0.000209424/0.25 | 212/0/0.0/0.0 | 4/4/1.0/1.0 | 0/0/None/0.0 |
| 3 | v00–v03 | 1 | refused cheirality_failed | refused insufficient_one_to_one_support | 4757/0/0.0/0.0 | 206/0/0.0/0.0 | 1/1/1.0/1.0 | 0/0/None/0.0 |
| 4 | v00–v04 | 2 | refused cheirality_failed | refused insufficient_one_to_one_support | 4621/0/0.0/0.0 | 233/0/0.0/0.0 | 2/2/1.0/1.0 | 0/0/None/0.0 |
| 5 | v00–v05 | 6 | refused cheirality_failed | refused insufficient_one_to_one_support | 4619/0/0.0/0.0 | 269/0/0.0/0.0 | 6/6/1.0/1.0 | 0/0/None/0.0 |
| 6 | v00–v06 | 3 | refused cheirality_failed | refused insufficient_one_to_one_support | 4939/0/0.0/0.0 | 255/0/0.0/0.0 | 3/3/1.0/1.0 | 0/0/None/0.0 |
| 7 | v00–v07 | 4 | refused cheirality_failed | refused insufficient_one_to_one_support | 4117/0/0.0/0.0 | 201/0/0.0/0.0 | 4/4/1.0/1.0 | 0/0/None/0.0 |
| 8 | v01–v02 | 37 | refused cheirality_failed | refused insufficient_one_to_one_support | 6745/7/0.001037806/0.189189189 | 262/1/0.003816794/0.027027027 | 37/37/1.0/1.0 | 0/0/None/0.0 |
| 9 | v01–v03 | 11 | solved | refused insufficient_one_to_one_support | 7076/1/0.000141323/0.090909091 | 335/0/0.0/0.0 | 11/11/1.0/1.0 | 0/0/None/0.0 |
| 10 | v01–v04 | 7 | refused cheirality_failed | refused insufficient_one_to_one_support | 6395/2/0.000312744/0.285714286 | 266/0/0.0/0.0 | 7/7/1.0/1.0 | 0/0/None/0.0 |
| 11 | v01–v05 | 18 | refused cheirality_failed | refused insufficient_one_to_one_support | 6493/2/0.000308024/0.111111111 | 309/1/0.003236246/0.055555556 | 18/18/1.0/1.0 | 0/0/None/0.0 |
| 12 | v01–v06 | 4 | refused cheirality_failed | refused insufficient_one_to_one_support | 6879/0/0.0/0.0 | 339/0/0.0/0.0 | 4/4/1.0/1.0 | 0/0/None/0.0 |
| 13 | v01–v07 | 9 | refused cheirality_failed | refused insufficient_one_to_one_support | 5482/0/0.0/0.0 | 236/0/0.0/0.0 | 9/9/1.0/1.0 | 0/0/None/0.0 |
| 14 | v02–v03 | 91 | refused cheirality_failed | **solved** | 7052/31/0.004395916/0.340659341 | 331/2/0.006042296/0.021978022 | 91/91/1.0/1.0 | 89/89/1.0/0.978021978 |
| 15 | v02–v04 | 102 | refused cheirality_failed | refused low_spatial_spread | 6635/28/0.004220045/0.274509804 | 328/0/0.0/0.0 | 102/102/1.0/1.0 | 99/99/1.0/0.970588235 |
| 16 | v02–v05 | 99 | refused cheirality_failed | refused solve_pair_refused | 6924/33/0.004766031/0.333333333 | 413/2/0.004842615/0.020202020 | 99/99/1.0/1.0 | 96/96/1.0/0.969696970 |
| 17 | v02–v06 | 68 | refused cheirality_failed | refused insufficient_one_to_one_support | 7792/20/0.002566735/0.294117647 | 515/3/0.005825243/0.044117647 | 68/68/1.0/1.0 | 0/0/None/0.0 |
| 18 | v02–v07 | 17 | refused cheirality_failed | refused insufficient_one_to_one_support | 5726/2/0.000349284/0.117647059 | 290/0/0.0/0.0 | 17/17/1.0/1.0 | 0/0/None/0.0 |
| 19 | v03–v04 | 63 | refused cheirality_failed | **solved** | 6419/21/0.003271538/0.333333333 | 315/1/0.003174603/0.015873016 | 63/63/1.0/1.0 | 63/63/1.0/1.0 |
| 20 | v03–v05 | 51 | refused cheirality_failed | refused insufficient_one_to_one_support | 6597/9/0.001364256/0.176470588 | 324/2/0.006172840/0.039215686 | 51/51/1.0/1.0 | 0/0/None/0.0 |
| 21 | v03–v06 | 24 | refused cheirality_failed | refused insufficient_one_to_one_support | 7176/3/0.000418060/0.125 | 368/0/0.0/0.0 | 24/24/1.0/1.0 | 0/0/None/0.0 |
| 22 | v03–v07 | 6 | refused cheirality_failed | refused insufficient_one_to_one_support | 5419/1/0.000184536/0.166666667 | 258/0/0.0/0.0 | 6/6/1.0/1.0 | 0/0/None/0.0 |
| 23 | v04–v05 | 140 | refused cheirality_failed | refused solve_pair_refused | 6740/46/0.006824926/0.328571429 | 515/21/0.040776699/0.15 | 140/140/1.0/1.0 | 137/137/1.0/0.978571429 |
| 24 | v04–v06 | 88 | refused cheirality_failed | **solved** | 7041/24/0.003408607/0.272727273 | 380/0/0.0/0.0 | 88/88/1.0/1.0 | 85/85/1.0/0.965909091 |
| 25 | v04–v07 | 39 | refused cheirality_failed | refused insufficient_one_to_one_support | 5394/10/0.001853912/0.256410256 | 283/1/0.003533569/0.025641026 | 39/39/1.0/1.0 | 0/0/None/0.0 |
| 26 | v05–v06 | 159 | refused cheirality_failed | refused solve_pair_refused | 7328/54/0.007368996/0.339622642 | 538/17/0.031598513/0.106918239 | 159/159/1.0/1.0 | 158/158/1.0/0.993710692 |
| 27 | v05–v07 | 77 | refused cheirality_failed | refused solve_pair_refused | 5583/20/0.003582303/0.259740260 | 322/1/0.003105590/0.012987013 | 77/77/1.0/1.0 | 75/75/1.0/0.974025974 |
| 28 | v06–v07 | 63 | refused cheirality_failed | **solved** | 5942/26/0.004375631/0.412698413 | 342/0/0.0/0.0 | 63/63/1.0/1.0 | 63/63/1.0/1.0 |

Selected-match status / gate / refusal outcomes:

* baseline: 1/28 solved, 27/28 refused with typed reason `cheirality_failed`
  (selection state `selected_on_refusal` in all 27; pose `pose_not_attempted`
  because the pose stage never ran). Baseline selected cheirality fractions are
  0.0107–0.4324, all below the unchanged `min_cheirality_fraction = 0.75`; the one
  solved pair (9) reached cheirality 0.9914 with support 350 and parallax 73.14°.
* oracle: 4/28 solved (pairs 14, 19, 24, 28), 24/28 refused — typed reasons
  `insufficient_one_to_one_support` 19 (selection `absent_selection`, pose
  `pose_not_attempted`), `solve_pair_refused` 4 (all `pose.reason = ambiguous_yaw_class`; selection
  `selected_on_refusal`, pose `pose_abstained`), `low_spatial_spread` 1 (pair 15, selection
  `selected_on_refusal`, pose `pose_not_attempted`). Oracle selection states:
  `selected_solved` 4, `selected_on_refusal` 5, `absent_selection` 19; oracle pose
  states: `pose_solved` 4, `pose_abstained` 4, `pose_not_attempted` 20.
* pooled: `baseline_solved = 1`, `oracle_solved = 4`,
  `oracle_solved_where_baseline_refused = 4`,
  `oracle_lost_where_baseline_solved = 1` (pair 9: the 11-edge oracle graph is too
  sparse to reach the support floor). Gates are otherwise unchanged: every oracle
  solve passed support (89/75, 63/63, 85/63, 63/49) with spatial bins 16,
  cheirality 1.0, singular ratio 0.0845–0.1573 and parallax 19.06–39.18°.
* **Explicit answer:** the truth-filtered oracle candidate graph does **not**
  uniformly make the existing geometric selector solve. It raises candidate reach
  from 345 to all 1,213 truth pairs and lifts selected reach from 52 to 865 at
  precision 1.0, and it converts 4 refused pairs into solves, but the unchanged
  selector still refuses 24/28 pairs because the truth candidate set is too sparse
  to satisfy the support/spatial gates (19 pairs) or the pose solve still refuses
  (4 pairs); candidate contamination is therefore only one of at least two
  bottlenecks.

Caveats: this is a single truth-assisted upper bound on one 8-frame export, not a
production result, not a model-performance estimate and not a generalization claim
(only three topology groups, every split `train`); the oracle camera is assisted
factory capture metadata; only 5,531/25,834 detected keypoints carry truth
(`exceeded_max_distance` 17,180, `excluded_partial_owner` 3,123); 1,588 boundary
keypoints were dropped; the 1,213 truth pairs are the evaluator's unique-pixel-pair
model, not independent labels; partial models occlude as full cubes but are excluded
from ownership, so returned identities without a registered surface are
`unattributed` (149,496 baseline-candidate, 6,901 baseline-selected, 0 oracle) and
baseline precision is a censored lower bound; the oracle graph replaces the
descriptor graph, so its reach is an upper bound on what the descriptor stage could
supply. No source, test, threshold, gate, capture, sealed prior artifact or
production setting was changed, and the production builder was restored after each
oracle call.

Current hashes after the run (unchanged from before the run except the output):
script `scripts/evaluation/evaluate_oracle_candidate_graph.py`
`eabee92286e90481b64ff10e599b67d0dc90a1407d0fe76ad700008d34b60726`; tests
`tests/test_oracle_candidate_graph.py`
`ac88a3978fe052306a8257f42341d3b0019857deb9e5e3fa66d0826c41b49290`; evaluator
`src/mc_photogrammetry/evaluation_pair_correspondence.py`
`18a0b3a60394c7bbc63e3f3d4a1eb24511d5164c35b5d80629d0c9a78cdef6de`; matcher
`src/mc_photogrammetry/pipeline/joint_matching.py`
`4d856c51fba3cc1855e49ac2325eaf6213b7f23341a369503a0561f5c6a63e7e`;
correspondence `src/mc_photogrammetry/pipeline/correspondence.py`
`192a4851e428d8116e0fbffd28c7ff157d1aedd8035e499c6df7ec2a9d787fa1`. Input index
sha256: `fresh-c4-canal-town-framed-2026-09-12`
`0637caba1a523bbdf7351dedd7c4f93bfdb611f0b549fc54b3a781f3d25ee46b`;
`fresh-c4-canal-town-2026-09-12`
`d3dc69b048bf8b827d75df402aa041445678ae63ba5124e6eefc6586278413dd`;
`recon-canal-houses-2026-09-12`
`5d0db614b1802bd7fcd047c390dd4637eb20dcd88f550d76a624d3bf6ba6fae9`;
`pilot-batch-train-station-2026-09-12`
`4ef40f8be5663108036bd3827e298286c93747a71ed9bbb4dcd2a3266ecf4837`;
`smoke-train-station-ds20c-2026-09-12`
`a2a21eb49d4b7664f603eaaca77bf8a8095defee520a2c64dc2592fb1625a3e0`. Canonical
entry scene-file pins: `capture.json`
`c6c44d43408779006854c81742a68d00cfe5b546f506ffa08ea59ba7b812abe6`,
`ground_truth.json` `36114feebe61d7b358927095a159d56dc5762e02c5a52a52e8a12ae17ce4a35a`,
`plan.json` `70fcb26bd49e37ca241ba89dcdb22128e859d821ad742419d0bcb5e158a593bc`,
`recipe.json` `10d415c85f5f8f7b7ab5f8e9a3c8d78bad3e7a0233b2971bc563f70eefa8ca4a`,
`reference.litematic` `f2601f55aee8156da8208aa7225c7406cdd67b6f5f078b312ed4d8c89c3eaa74`;
screenshot pins `48ba0ff58c77b588acd5aa06a2c630fc2f5103c6d0d67b45b4bc21b81ecf533c`,
`8750bb197329eb50144896264db40eef70a6c44cc3dce60eac3a4f1e9aa79696`,
`2d017c8f804999446f54dbc742fb87d7748af76ad50249c0e2853754c55b6a95`,
`e71bfda3ece7a0b74d2da2fd62834645fcbcf662fb4b5889fae95fa26c9a09e9`,
`8797d70e0e09475d296c3cf18e5892f6b22d41750ed18d6c3364da80ba60a81f`,
`6f844d5e3600e853bf3045fc25b55a271c0b665226053602b6d30c945f0f016b`,
`53c38526d378a0597ab6e00fb5e01a6c7aa0e2dc11619c1b7eaa3cfc0ccf8923`,
`fdd8a3709479795b204f67153b10f44b49e85f2bc0050a46d9fdd0461f025556`.

### 5.2 Lead review erratum: support-floor reachability and pose refusal detail

The sealed artifact records `support_floor = 0` for the 19 `insufficient_one_to_one_support` rows. This is a telemetry omission in `joint_matching.match_feature_frames`: the `if not modes` refusal branch calls `_refused` without forwarding the already-computed support floor. The immutable artifact is not edited. The effective floor can be reconstructed from its raw-keypoint counts and sealed settings as `max(8, ceil(0.02 * min(raw_keypoints)))`. In every one of those 19 pairs, even the entire oracle edge count is below that floor, so none could have reached the configured floor even before endpoint one-to-one conflicts. That establishes a hard support-reachability ceiling for this selector under this fixture; it does not prove the production threshold should be lowered, because same-surface oracle edges are not guaranteed exact feature-point correspondences.

| Pair IDs | Oracle surface-pair edges / reconstructed support floor |
| --- | --- |
| 1-7 | 20/35, 4/35, 1/35, 2/35, 6/35, 3/35, 4/35 |
| 8-13 | 37/75, 11/75, 7/63, 18/72, 4/75, 9/49 |
| 17-18 | 68/75, 17/49 |
| 20-22 | 51/72, 24/75, 6/49 |
| 25 | 39/49 |

The four `solve_pair_refused` pairs are 16, 23, 26 and 27. The artifact records the same typed pose refusal, `ambiguous_yaw_class`, for all four; their selected truth-graph supports/floors are 96/72, 137/63, 158/72 and 75/49. Full DS-11 per-yaw `YawClassScore` evidence is not serialized, so the common class name is verified but the competing residuals are unavailable without a new diagnostic. Do not weaken the ambiguity margin from this artifact alone.

## 6. Limitations

* **Truth-assisted upper bound.** The oracle graph uses the registered
  ground-truth surfaces and the recorded factory capture cameras, which normal
  reconstruction never sees. It is deliberately optimistic and is not a production
  result, not a model-performance estimate and not a generalization claim.
* **Assisted camera.** The oracle factory camera metadata is an evaluation aid, not
  image-derived calibration; this is not the default image-only reconstruction path.
* **One export, one run.** Only the canonical Canal Town entry is evaluated. Only
  three independent topology groups exist and every index split is `train`, so no
  generalization claim is made.
* Pair hypotheses are not truth correspondences; precision/recall are scored
  against an evaluator-built truth model, not independent labels.
* **No texture identity** exists anywhere in the pipeline.
  `same_state_same_slot_other_block` is an appearance proxy only.
* Partial block models occlude as full cubes but are excluded from truth ownership;
  the exported region covers only part of each screenshot, so returned matches
  whose endpoint lacks a registered surface are `unattributed` and the all-returned
  precision is a censored lower bound.
* Truth and returned identities deduplicate only keypoints at exactly the same
  pixel; near-duplicate keypoints at neighbouring pixels are not collapsed. Truth
  pairs sharing one pixel therefore map to one canonical index, which can only
  under-count oracle reach, never inflate it.
* The oracle edges keep the real descriptor distances, but the descriptor graph
  itself is replaced, so oracle candidate reach is an upper bound on what the
  descriptor stage could supply.
* No matcher, threshold, gate, solver, production, project or promotion setting was
  changed; the production reciprocal graph builder is restored after every oracle
  call.

## 7. Stop condition

Stopped for lead review. No production edit, threshold/gate change, promotion, or
additional corpus run was made; the one authorized corpus evaluation is documented
in section 5.1. No background process/job/workflow,
subagent, shell nesting, junction/reparse point, recursive deletion, generated
executable, install/download, Minecraft/client/capture action, Defender change or
system write was used, and the pre-existing dirty work tree was preserved.
