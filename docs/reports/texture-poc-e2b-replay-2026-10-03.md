# E2B evaluator-only E1 -> production trace replay reader

Date: 2026-10-03
Task: `texture-poc-e2b-production-replay-reader-2026-10-03`
Status: implemented, focused tests + `py_compile` pass, stopped for lead review.
This report records the reader design and its synthetic evidence. It is **not** a
corpus result and it makes no promotion recommendation.

**Post-implementation update (same date):** the lead subsequently ran the pinned
default T2/T6 pipelines with schema-2 traces and executed this reader. The
authoritative run metrics and artifacts are in the separate
[pinned E2B results report](texture-poc-e2b-pinned-results-2026-10-03.md).
Section 6 below records the workspace state when this implementation task ended;
its pre-run statement that no schema-2 traces were available is historical.

## 1. Scope and owned files

The task asked for an evaluator-only reader that reconciles a sealed E1
patch-label artifact (schema `mc-photogrammetry-texture-patch-labels-v1`) with a
production stage trace (schema 2, the E2A observation trace). The reader parses
two JSON documents, validates both fail-closed, joins them and returns one
JSON-safe summary. It reads no screenshot, reference litematic, camera manifest,
mask or JAR, runs no benchmark or corpus evaluation, writes no artifact, and
changes no production, threshold, palette or gate.

Owned files (only these were created or edited):

| Path | SHA-256 |
| --- | --- |
| `scripts/evaluation/replay_texture_observation_trace.py` | `1f5c3417ae2921dfd4c355626da1492bbd231f65faab23dbc9a7ab9c40677172` |
| `tests/test_texture_observation_trace_replay.py` | `22f46f1e846117fe80a9a33f24ba234377b75da2da112a22a1cc9620a5bc5b81` |
| `docs/reports/texture-poc-e2b-replay-2026-10-03.md` | this report |

CLI:

```
python scripts/evaluation/replay_texture_observation_trace.py \
    <e1-artifact.json> <stage-trace.json> --role {template-2|template-6}
```

The summary is printed to stdout as JSON (`REPLAY_SCHEMA =
"mc-photogrammetry-texture-observation-replay-v1"`); a refusal exits `2` with a
typed `TraceReplayError` on stderr. Nothing is written to disk by the reader and
there is no optional mode or duplicate-collapse feature.

This revision follows the four lead E2B clarifications posted after assignment:
the `seen_here` source check, "proceed without further design branching", scope
control (no optional duplicate modes, no extra CLI features), and the explicit
"no `--allow-multipart-duplicates`" duplicate-policy confirmation.

A later lead review found a correctness bug and this revision implements the
remediation: the E1 face-patch key intersection, digest parity and
per-observation truth rank are now restricted to observations whose `source` is
`resolved_face`. `centre_patch_fallback` observations are counted separately as
production-only evidence, including any `(frame_id, coordinate, slot)` key
collision with an E1 face key, and are never counted as E1 face joins. Per-cell
outcomes still come from the production coordinate trace (so fallback evidence
remains part of the actual production cell result) and now carry an explicit
denominator. This change is described in sections 2 and 5; the added regression
is in section 7.

A second lead review then raised two narrow fallback-contract cleanup items, both
implemented in this revision. First, the per-cell field named
`fallback_observations_in_labeled_cells` actually counted *distinct E1-labeled
coordinates* carrying fallback evidence, so it is replaced by an accurate pair:
`fallback_observation_rows_in_labeled_coordinates` (fallback observation rows at
E1-labeled coordinates) and `fallback_labeled_coordinates_with_evidence`
(distinct such coordinates). Several fallback observations can sit at one labeled
coordinate - in different slots or frames - so the two numbers are not
interchangeable. Second, `load_trace` now enforces the digest half of the fallback
contract: `observed_patch`, `valid`, `coverage` and `pattern` must be null for a
`centre_patch_fallback` observation, and any non-null value is refused. This
closes the gap where a fallback could previously carry a well-formed digest
through field validation. Both changes are described in sections 3-5 and tested in
section 7.

## 2. Duplicate-identity invariant

Source check confirms that `raycast.reconstruct_scene_surfaces` keeps a
`seen_here: set[(coordinate, slot)]` per frame between `sample_visible_faces`
and the retained `raw` list, so at most one resolved-face observation per
`(frame_id, coordinate, slot)` survives, and the centre-patch fallback only runs
for coordinates no resolved face covered. E1's `deduplicate_samples` keeps one
patch per key. A duplicated `(frame_id, coordinate, slot)` identity is therefore
a source contradiction, and the reader refuses it in `load_trace` rather than
collapsing or silently picking one.

The same source check fixes the join semantics. Only `resolved_face`
observations carry a rectified face patch, so only they participate in the E1
patch-key intersection, digest parity and per-observation truth rank. A
`centre_patch_fallback` observation is production-only evidence even when its
`(frame_id, coordinate, slot)` collides with an E1 face key: it is counted under
`fallback_observation_keys`, `fallback_only_keys` and
`fallback_key_collisions_with_e1`, and never as a joined face patch or a rank
row. Fallback evidence still contributes to the production coordinate trace and
therefore to the per-cell outcomes, whose denominator is stated explicitly.

## 3. Fail-closed input contract

E1 side:

* `schema == mc-photogrammetry-texture-patch-labels-v1`, `evaluation_only is
  True`, `role` equals the requested reviewed role (`template-2`/`template-6`).
* The embedded `artifact_sha256` is recomputed over the canonical JSON of the
  record without that field and must match.
* `(frame_id, coordinate, slot)` keys must be unique; patches for one coordinate
  that disagree on the truth state are refused.
* `patch`/`valid`/`coverage` are decoded only at the exact recorded shape and
  dtype (16x16x3 `<f4`, 16x16 `|b1`, 16x16 `<f4`), with strictly validated,
  bounded base64; non-finite floats are refused.

Trace side:

* `schema == 2` (an `int`; JSON `true`/`2.0` are rejected) and every schema-1 and
  schema-2 key present.
* observation ordinals are the record positions; `observations` and
  `observation_costs` have equal length and agree on
  `frame_id`/`coordinate`/`slot`/`source`; `palette_order` equals
  `final_palette.states`; every cost vector has the palette length and is finite;
  `effective_weight == weight * view_weight`; a production cost vector is
  `min(distance - best, ceiling)` and therefore must carry a zero entry.
* digest fields obey the observation's `source`. A `resolved_face` observation
  must carry `observed_patch`, `valid` and `coverage` (and may carry `pattern`);
  a `centre_patch_fallback` observation must carry `observed_patch`, `valid`,
  `coverage` and `pattern` as `null`. A non-null fallback value is refused, so a
  fallback can never smuggle a patch digest into the join. `rgb` and `descriptor`
  are required for both sources.
* `aggregation` coordinates are unique and carry a finite total of the palette
  length and a finite non-negative weight.
* Replaying `observation_costs` with `effective_weight` (same ordinal order,
  skipping non-positive weights, exact float64 arithmetic) must reproduce every
  `aggregation` total and weight to `rtol=1e-9, atol=1e-12`; otherwise the run is
  refused. The production top-K set is then re-derived independently from the
  replayed mean costs through the same
  `np.argpartition(mean, min(8, size) - 1)[:min(8, size)]` contract and compared
  against each coordinate record's `top_candidates`. That check is a separate
  summary section and never feeds the E1 per-face ranking.

## 4. Digest identity contract

The trace digests `FaceSample.observed_patch`: invalid pixels are replaced by the
coverage-weighted `FaceSample.rgb` in float32, and an all-valid patch is returned
unchanged. E1 stores the raw float32 `patch` plus `valid` and `coverage`. For a
like-for-like comparison the reader reconstructs that production float32 observed
patch from the E1 raw arrays using production's own arithmetic; it does **not**
reuse E1's `_observed_patch`, which deliberately promotes to float64 for its
weight calculation. Both comparisons use the production contract
`sha256(f"{shape}|{dtype.str}|" + C-order bytes)`:

* `patch_observed_reconstruction` — reconstructed production observed patch vs
  trace `observed_patch`. This is the parity claim and the like-for-like check.
* `patch_raw_vs_observed` — E1's stored raw patch vs trace `observed_patch`. It
  differs by construction on partially owned rows and is reported separately
  rather than hidden.

`valid` and `coverage` are compared directly. Mismatch counts are exact and up to
ten example rows are included, so nothing is silently discarded. Digest parity is
evaluated only over keys joined to a `resolved_face` observation; fallback
observations carry no patch digests and are excluded from these denominators.
`load_trace` enforces that input contract: a `centre_patch_fallback` observation
whose `observed_patch`, `valid`, `coverage` or `pattern` is non-null is refused
before any join, so exclusion here is a checked invariant rather than an
assumption.

## 5. Report denominators (all distinct)

| Section | Fields | Denominator |
| --- | --- | --- |
| `join` | `e1_patch_keys`, `production_observation_keys`, `resolved_face_keys`, `fallback_observation_keys`, `intersection_keys`, `e1_only_keys`, `production_only_keys`, `fallback_only_keys`, `fallback_key_collisions_with_e1`, `per_frame`, `per_slot` | resolved-face key sets and per frame/slot; `intersection`/`e1_only`/`production_only` cover resolved faces only and fallback keys are counted separately |
| `digest_equality` | `patch_observed_reconstruction`, `patch_raw_vs_observed`, `valid`, `coverage`, `all_three_observed_contract`, `partial_rows_reconstruction`, `fully_valid_rows_reconstruction`, `mismatches` | joined resolved-face keys, split by E1 `partial` |
| `truth_rank` | `top1`/`top8`/`top16`/`top32`, `truth_in_final_palette`, `truth_out_of_vocabulary`, `examples` | joined resolved-face keys; recall denominator is truth-in-final-palette only |
| `per_cell` | `top8_membership`, `raw_winner_match`, `post_smoothing_match`, `selected_state_match`, `final_state_match` | E1-labeled coordinates whose trace record provides the field (production cell result includes fallback evidence); `candidate_abstentions`, `placement_abstentions`, `coordinates_missing_trace_record`, `truth_out_of_vocabulary`, `fallback_observation_rows_in_labeled_coordinates`, `fallback_labeled_coordinates_with_evidence` counted separately |
| `aggregation_replay` | `weights_equal`, `totals_equal`, max absolute errors | aggregation coordinate set |
| `production_top_k_check` | `checked`, `disagreements` | coordinates with `raw_winner` and a replayed total |

Every rate is serialized as `{numerator, denominator, rate}`; an empty
denominator yields `rate: null`. The reader never infers the visible-cell total
from patch rows: `per_cell` is explicitly the E1-labeled coordinate subset, and
the artifact's own `denominators` mapping is carried through under `e1`. Truth
rank is defined only where the E1 truth state is a member of `final_palette`;
ties are broken by palette position (cheaper first, then earlier palette index).
`fallback_observation_rows_in_labeled_coordinates` counts fallback *observations*
whose coordinate is E1-labeled, while
`fallback_labeled_coordinates_with_evidence` counts the *distinct coordinates*
carrying that evidence; one coordinate may hold several fallback rows, so the pair
is reported instead of a single number that could only be one of the two.

## 6. Evidence: real sealed E1 artifacts and the missing v2 trace

Read-only structural validation with `load_e1_artifact` (no images, no writes):

| Role | Patches | Distinct coords | Partial | File SHA-256 |
| --- | --- | --- | --- | --- |
| `template-2` | 1,168 | 577 | 74 | `fb9fb5f9…b068874` (R4) |
| `template-6` | 4,025 | 1,866 | 201 | `661d537d…efa0d30` (R4) |

Both artifacts' embedded seals recomputed cleanly. Their distinct-coordinate
counts equal the R4 `scorer_usable_cells` denominators (577 / 1,866). The partial
counts are the rows where the raw-vs-observed distinction matters.

Production traces currently present in the workspace are all **schema 1** and
carry no E2A keys, so no real join can be executed yet:

| Trace | schema | v2 keys |
| --- | --- | --- |
| `.tmp-stagewise-pinned-template-2-trace.json` | 1 | absent |
| `.tmp-stagewise-pinned-template-6-trace.json` | 1 | absent |
| `.tmp-stagewise-t6-no-prune-trace.json` | 1 | absent |
| `.tmp-ds-r36-t6-noprune-trace.json` | 1 | absent |

This report therefore contains no real reconciliation numbers. Producing a
schema-2 trace (the E2A writer) is the precondition for the first real E2B run.

## 7. Synthetic golden join

`tests/test_texture_observation_trace_replay.py` builds a four-state palette and
a self-sealed E1 artifact with four patches (one partial, one OOV truth) plus a
five-observation trace (four resolved observations, one of them production-only,
and one centre-patch fallback). The golden summary asserts, among others:

* join: 4 E1 keys, 5 production observations (4 `resolved_face`, 1
  `centre_patch_fallback`), 3 resolved-face intersection, 1 E1-only, 1 resolved
  production-only, 1 fallback-only, 0 fallback key collisions with E1; per-frame
  and per-slot rows carry `resolved_face` and `fallback` separately;
* `all_three_observed_contract` 3/3, `valid` 3/3, `coverage` 3/3,
  `patch_observed_reconstruction` 3/3, `partial_rows_reconstruction` 1/1,
  `fully_valid_rows_reconstruction` 2/2, and `patch_raw_vs_observed` 2/3 because
  the partial row's raw patch cannot equal production's observed patch;
* `truth_rank` 2 in final vocabulary, 1 OOV, top-1 1/2, top-8 2/2, with the
  partial cell ranked 2;
* per-cell: 4 E1-labeled coordinates, 3 with a trace record, 1 missing record,
  2 truth-available, 1 OOV, top-8 2/2, raw winner 1/2, post-smoothing / selected /
  final 2/2, and 0 fallback observation rows in labeled coordinates with 0 labeled
  coordinates carrying evidence (the only fallback sits at coordinate `(5,0,0)`,
  which E1 did not label);
* `aggregation_replay` weights and totals 5/5 and the independent top-K check
  4/4.

Other synthetic cases cover duplicate E1 keys, conflicting coordinate truth,
duplicate production identities, bad base64/shape/dtype, non-finite arrays and
costs, a cost vector without a zero, unknown schemas, tampered E1 seal,
role/marker mismatches, cost-length and palette-order mismatches, ordinal
misalignment, duplicated aggregation coordinates, aggregation total and weight
tampering, OOV separation, clipped-cost ties ranked by palette position,
candidate/placement abstentions, a 10-state explicit cheapest-eight top-K check
and an injected top-K disagreement, a JSON round trip, and the CLI.

The added fallback-collision regression builds one E1 face key whose only
production observation is a `centre_patch_fallback` at the same
`(frame_id, coordinate, slot)`, with no resolved observation, and asserts:

* `e1_patch_keys` 1, `resolved_face_keys` 0, `fallback_observation_keys` 1,
  `intersection_keys` 0, `e1_only_keys` 1, `production_only_keys` 0,
  `fallback_only_keys` 0 and `fallback_key_collisions_with_e1` 1 — the collision
  is counted as fallback evidence, not as a joined patch;
* `digest_equality.joined_key_denominator` and `truth_rank.joined_key_denominator`
  are 0, digest and rank rates are `{0, 0, null}`, the mismatch count is 0 and no
  rank row is produced;
* the fallback still drives the production cell result:
  `fallback_observation_rows_in_labeled_coordinates` 1 and
  `fallback_labeled_coordinates_with_evidence` 1, `raw_winner_match` 1/1 and the
  independent top-K check 1/1.

The fallback-contract cleanup adds three focused cases:

* `test_fallback_observation_and_coordinate_counts_are_separate` places two
  fallback observations (different slots) at one E1-labeled coordinate and
  asserts the row count is 2 while the distinct-coordinate count is 1, with both
  fallback keys counted as production-only evidence (one of them a key collision);
* a parametrized `test_fallback_observation_digest_fields_must_be_null` sets each
  of `observed_patch`, `valid`, `coverage` and `pattern` in turn on a
  `centre_patch_fallback` observation and requires a `must be null for a
  centre_patch_fallback observation` refusal;
* `test_resolved_face_observation_requires_patch_digests` nulls a resolved
  observation's `observed_patch` and requires the `is required for a resolved
  face` refusal, locking the contract as one-sided rather than merely permissive.

## 8. Checks run

Per the task, only `py_compile` and the focused replay tests were run this
revision.

```
.\.venv\Scripts\python.exe -m py_compile scripts\evaluation\replay_texture_observation_trace.py tests\test_texture_observation_trace_replay.py
-> exit 0

$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest -o addopts='' -q tests\test_texture_observation_trace_replay.py
-> 29 passed in 0.47s
```

No fixture/corpus evaluator was run, no image/reference/mask/JAR was read, and no
artifact was generated or overwritten. Only the three owned paths changed.

## 9. Limitations and preconditions for the first real run

* Results are conditional on the supplied cameras and reference-assisted
  geometry the E1 artifact was produced with. This is not an image-only product
  score and must never be reported as one.
* E1 patch rows are the reference-assisted exact-model sampler subset; they are
  never a visible-cell total. The artifact's own `denominators` remain the only
  visible-domain numbers.
* E1 samples through the reference `geometry_palette` while production uses the
  pipeline palette and its own `seen_here` geometry pass, so digest equality
  rates measure a real interface gap; a low rate is a finding, not a reader
  failure.
* Only `resolved_face` observations carry patch digests and only they join E1
  face keys or contribute truth-rank rows. `centre_patch_fallback` observations
  are counted as production-only evidence, including any `(frame_id, coordinate,
  slot)` key collision, and remain part of the production per-cell result. A
  `centre_patch_fallback` observation must carry null `observed_patch`, `valid`,
  `coverage` and `pattern`; the reader refuses a non-null value before any join.
* If a real trace ever contains a duplicated `(frame_id, coordinate, slot)`
  identity, the reader refuses it and the lead must be told: it contradicts the
  `seen_here` invariant and is a source mismatch, not something this reader may
  collapse.
* This reader trains nothing, changes no gate or threshold, and recommends no
  model promotion.
