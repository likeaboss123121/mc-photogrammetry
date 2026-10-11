# M5 appearance/state proposal adapter — implementation report

Date: 2026-09-19
Task: bounded M5 appearance/state proposal adapter (three-new-path scope)
Role: bounded developer (DeepSeek/Sol-equivalent) under the direct-shell policy in
[`docs/deepseek-worker.md`](../../docs/deepseek-worker.md)
Status: implementation complete; Sol's retained-array-invariant and digest-binding
findings repaired in place; required checks pass; **stopped for manager/Astra review**

## Context and scope

This task adds the narrow, bounded M5 seam that turns accepted immutable
`FaceObservation` tuples into typed per-cell material proposals. It consumes only
observations and an explicit `PaletteColor` tuple; it never touches geometry,
occupancy, cameras, scale, bounds, manifests, CLI/UI, persistence, fusion or
export. Exactly three new paths were added, and nothing else was edited:

- `src/mc_photogrammetry/reconstruction/appearance_state_proposals.py` (new)
- `tests/test_appearance_state_proposals.py` (new)
- `docs/reports/M5-appearance-state-proposals-2026-09-19.md` (this file)

`BlockState` appears only as the proposed value inside a proposal. The adapter
does not import `facade`, `raycast`, `occupancy`, `litematic`, `cli`, `ui`,
`store`, `shape_fit` or any volume/export surface; the test suite asserts that by
import AST and by module-namespace inspection. `AGENT_COMMS.md` was not modified.

## Repair pass (Sol findings, 2026-09-19)

Two defects from the first pass were repaired in place. The public contract, the
record shapes, the entry-point signature and the set of three scoped paths are
unchanged.

1. **Retained-array invariants were not re-bound at the adapter boundary.**
   `_validate_observation` checked only the shape, finiteness and range of
   `patch`, `valid` and `coverage`, so a record built past
   `FaceObservation.__post_init__` could carry a writable or reopenable array.
   It now enforces the accepted bridge's exact artifact: `patch` is `float32`,
   `valid` is boolean, `coverage` is `float64`, every retained array has
   `writeable=False`, and every retained array's `base` chain terminates at
   immutable `bytes` (non-reopenable). Nothing is copied or coerced to satisfy
   the check: the exact handed-in arrays are inspected by
   `_has_immutable_bytes_base` and `_require_immutable_retained`, mirroring the
   accepted `face_observation._readonly` retention. Observation validation
   already runs before every budget, so an adversarial writable or reopenable
   record now raises the typed `AppearanceStateProposalError` even when
   `max_observations`, `max_candidate_evaluations` and `max_payload_bytes` are
   all zero, instead of falling through to a budget refusal.

2. **`_observation_digest` bound only array bytes.** A `(2, 1, 3)` and a
   `(1, 2, 3)` patch of the same values, or an `int32` and a `float32` array of
   the same shape, serialize to identical bytes and therefore produced the same
   observation digest. `_retained_array_payload` now binds each of `patch`,
   `valid` and `coverage` as `{dtype, shape, sha256(C-order bytes)}`, and
   `_observation_digest` uses that payload for all three arrays. Reshaped or
   retyped evidence can no longer replay a proposal under the same digest.

Both repairs are covered by new synthetic tests. Same-cell grouping, canonical
state merging, exact scorer binding, typed abstentions, atomic budgets, digest
stability and the no-geometry/camera/scale/manual-metadata/export/CLI/UI
boundary are unchanged; `AGENT_COMMS.md` was not modified.

## Contract

### Inputs

`propose_face_states(observations, palette, *, max_cost, min_margin,
max_observations, max_candidate_evaluations, max_payload_bytes)`:

- `observations` must be an **immutable `tuple`** of `FaceObservation`. Every
  record is re-validated in full at the adapter boundary (cell, slot/normal
  agreement, finite `float32` patch, boolean `valid`, `float64` coverage, array
  shapes and ranges, read-only and immutable-`bytes`-backed retained arrays,
  usable coverage, support frames including `frame_id`, unique provenance,
  finite non-negative weight), so a tampered record built past `__post_init__`
  is rejected rather than scored.
- `palette` must be an **immutable `tuple`** of `PaletteColor` with a
  `BlockState`.
- `max_cost` and `min_margin` are explicit finite non-negative thresholds; the
  three budgets are explicit non-negative integers. Validation of observations
  and palette happens **before** any budget is consulted.

### Scoring delegation

Each usable observation is delegated unchanged to
`appearance_scoring.AppearanceScorer.score`, and the returned
`appearance_scoring.FaceScore` is bound exactly: it must be a `FaceScore`, must
not abstain for an observation that reported itself usable, must score every
palette entry exactly once, must key costs by `BlockState` with finite
non-negative values, and its state set must equal the palette's canonical state
set. Anything else raises the typed `AppearanceStateProposalError` before any
result can exist. A scorer that raises is wrapped in the same typed error.

### Aggregation

Observations are grouped by cell and sorted by `(cell, slot, frame_id,
observation_digest)`; a cell's proposal consumes only that cell's observations.
Candidates are merged by the canonical `BlockState` identity
(`BlockState._identity`), so repeated spellings of one state are one candidate.
For each usable observation the absolute cost of a candidate is
`best_cost + relative_cost`; the aggregate is the equal-weight mean over the
cell's usable observations. Ranking is `(aggregate cost, canonical state key)`,
so it is palette-order independent and deterministic.

### Decision and typed abstention

Precedence is explicit:

1. `no_observation` — the cell has no usable observations (or the call has no
   observations at all, at result level).
2. `high_cost` — aggregate best cost is not finite, or exceeds `max_cost`.
3. `weak_margin` — `runner_up_cost - best_cost` is below `min_margin`.
4. `conflicting_faces` — two usable observations of the **same slot** each
   nominate a different confidently-matched state (each with finite best cost
   `<= max_cost` and own margin `>= min_margin`). The conflict veto is applied
   only after both acceptance gates pass, so the reason is preserved rather than
   collapsing to `weak_margin`.
5. Otherwise the aggregate winner is proposed.

`empty_palette` and `budget` are result-level reasons; a single-candidate
palette has an infinite margin.

### Provenance and digests

Each `FaceStateProposal` retains the sorted support frame ids, the sorted
content digests of every retained observation (usable or not), the sorted union
of exact provenance references, the sorted observed slots, and the full
per-candidate aggregate cost tuple. `evidence_digest` binds the cell, the
palette scoring payload and the ordered per-observation entries.
`observation_digest` binds each retained array's dtype, shape and exact C-order
bytes for the `patch`, `valid` and `coverage` arrays, plus the
identity/provenance fields, coverage and weight. `proposal_digest` is a
deterministic SHA-256 over every proposal field; the result carries a canonical
`payload_sha256` and `result_digest`.

### Budgets and atomicity

- `max_observations` counts every validated observation (usable or not) and is
  enforced before scoring.
- `max_candidate_evaluations` is `usable_count * palette_size` and is enforced
  before any scorer is constructed or called.
- `max_payload_bytes` is the canonical UTF-8 size of the emitted result payload
  (the payload deliberately excludes `max_payload_bytes` itself, so the budget
  is not self-referential) and is enforced after construction.
- Any exhausted budget returns `status="budget"`, `reason="budget"`, the
  refusing budget name, `proposals=()`, and **no partial result**. A payload
  refusal happens after scoring and reports the evaluations performed; the
  pre-scoring refusals report zero.

## Files (post-repair SHA-256, lowercase; line counts)

```
39ecfc70415472a7770a76fe492cc251d42233a43a557ecbf60305ae8846876a  1200  src/mc_photogrammetry/reconstruction/appearance_state_proposals.py
902a62ca9eeb84afdf889947b98828cdca220bcb6af0feeb45c664ab3348a885   825  tests/test_appearance_state_proposals.py
```

`diff-scope`: the repair edited only the two source/test paths above and this
report; no fourth path was created or touched. Every other modified/untracked
path in `git status` (earlier-task M4/M5 work, `AGENT_COMMS.md`,
`AGENT_HANDOFF.md`, the swarm template and the pre-existing dirty baseline) was
not touched. No commit, stage, reset or clean was performed.

## Tests added

`tests/test_appearance_state_proposals.py` — 36 collected synthetic tests, all
built in memory (no fixture, image, manifest, camera or bound is read):

- exact winner with full provenance: proposed state, support frames, provenance,
  observation digests, per-candidate costs and digests;
- single-candidate palette infinite margin; canonical duplicate-state merging;
  palette-order independence of the value and digest;
- `weak_margin`, `high_cost` and `conflicting_faces` typed abstentions, the
  same-slot-only scope of conflict (different slots degrade to `weak_margin`),
  and `no_observation`/`empty_palette` result-level abstentions;
- an unusable observation is retained for provenance but cannot change the
  winner, and a cell of only unusable observations abstains with
  `no_observation` while keeping its digests and frames;
- deterministic ordering/digests: input-order independence, canonical cell
  order, same-cell evidence isolation between cells, digest stability and
  binding to value/evidence/thresholds, and stored-digest/payload agreement;
- budget atomicity: `max_observations` and `max_candidate_evaluations` refuse
  before the scorer is constructed or called (recording double), exact-boundary
  acceptance, `max_payload_bytes` refusal with zero proposals, and validation
  strictly before budget;
- malformed observation cases, malformed palette containers, malformed scorer
  output (non-`FaceScore`, abstained, missing/extra/bogus states, non-finite or
  negative costs, NaN best cost) and wrapped scorer failures;
- retained-array invariant enforcement: writable and reopenable (clear
  `writeable` flag but numpy-owned) `patch`/`valid`/`coverage` records are
  refused with a typed error even when all three budgets are zero; wrong-dtype
  retained arrays are refused; the accepted bridge's real arrays still validate
  and propose, and cannot be reopened;
- digest binding: a byte-identical reshaped pair (with `patch`, `valid` and
  `coverage` all reshaped) and byte-identical same-shape retyped pairs (`int32`
  vs `float32` patch, `uint8` vs `bool` `valid`, `int64` vs `float64`
  `coverage`) now produce different observation digests;
- boundary tests: no export/volume import or namespace symbol, the entry-point
  signature contains only the observations, palette and five explicit limits,
  frozen records, and cost-record consistency.

## Checks run (repair pass)

```
$ .\.venv\Scripts\python.exe -m py_compile `
    src\mc_photogrammetry\reconstruction\appearance_state_proposals.py `
    tests\test_appearance_state_proposals.py                       # exit 0

$ .\.venv\Scripts\python.exe -m pytest tests\test_appearance_state_proposals.py `
    -q -p no:cacheprovider -o addopts=                              # 36 passed

$ .\.venv\Scripts\python.exe -m pytest tests\test_face_observation.py `
    tests\test_appearance_scoring.py tests\test_image_derived_session.py `
    tests\test_project_store.py -q -p no:cacheprovider -o addopts=  # 140 passed (accepted suites)
```

The first pass reported the same three commands green (then 31 in the new suite);
its broader adjacent suites (164 passed) and full repository (2478 passed,
1 skipped) were also green. The repair changed only the two paths above, and a
repository-wide search confirms no module or test other than this pair imports
`appearance_state_proposals`, so the accepted re-run here bounds the blast
radius rather than a repeated full-suite pass.

## Deviations, limits and uncertainties

- **Payload refusal is post-evaluation.** `max_payload_bytes` measures the
  emitted result payload, so the adapter may score the cell set before refusing;
  the refusal is still atomic (no proposals are emitted) and the performed
  evaluation count is reported honestly. A caller wanting to avoid that work
  uses the two pre-scoring budgets.
- **Equal weighting.** Aggregation ignores `FaceObservation.weight`; the field
  is retained (per the accepted observation contract) but documented as a
  diagnostic, not a calibrated fusion weight. Weighted aggregation can be added
  later without changing the record shapes.
- **Strict conflict rule.** Any same-slot confident disagreement abstains, even
  when one state has a numeric majority across observations. This is the
  conservative "preserve the conflict" reading; the aggregate cost/margin gates
  are still evaluated first so the conflict reason is not lost.
- **State-set binding.** The scorer contract is bound by "one cost per palette
  entry and the canonical state set matches the palette", not by palette order;
  a future scorer that reorders its costs remains compatible, while one that
  drops, invents or renames a candidate stops the call.
- **Palette digest scope.** `evidence_digest` binds the palette fields that can
  affect a score under this adapter's fixed settings (state, colours, face map,
  family, descriptors, cube/box geometry); thumbnail/model-face payloads that
  only matter to unused texture/model terms are excluded from the digest.
- **Record validation duplication.** The adapter re-validates every field that
  `FaceObservation.__post_init__` already checked — now including the retained
  array dtypes (`float32`/`bool`/`float64`), the read-only flag and the
  immutable-`bytes` base chain — because a hostile caller can build a record
  past that constructor. Validation reads the handed-in arrays directly and
  never copies or coerces them to pass. This is deliberate defence in depth, not
  a second source of truth: the checks mirror the accepted contract.
- **Observation-digest payload.** The digest payload now carries a nested
  `{dtype, shape, sha256}` object per retained array instead of a bare bytes
  digest. `OBSERVATION_SCHEMA` (`m5-face-observation-1`) is unchanged because it
  names the observation record, not the digest payload; no persisted artifact
  binds the old digest shape.
- No image-only or fixture evidence was produced or scored. No occupancy,
  hidden-block, geometry, camera, scale, bounds or manual metadata is read or
  inferred. M5 production integration and promotion remain open.

Stopping here for manager review and independent verification. No commit, stage,
reset or clean was performed.
