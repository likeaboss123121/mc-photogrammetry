# M5 state-promotion adapter — implementation report

Date: 2026-09-19
Task: bounded M5 state-promotion adapter (three-new-path scope)
Role: bounded developer (DeepSeek/Sol-equivalent) under the direct-shell policy in
[`docs/deepseek-worker.md`](../../docs/deepseek-worker.md)
Status: implementation complete; Sol's result-status, nested-claim-replay and
forged-budget findings repaired in place; all required checks pass; **stopped for
Sol review**

## Context and scope

This task adds the narrow, bounded M5 seam that turns one accepted, immutable,
geometry-only `PlaneVolumeResult` plus one accepted, immutable
`FaceStateProposalResult` into typed, digest-bound promoted block states. It
consumes only those two records and an explicit promotion budget; it never touches
CLI/UI, persistence, cameras, poses, exporters, or any existing production
module. Exactly three new paths were added, and nothing else was created or
edited:

- `src/mc_photogrammetry/reconstruction/state_promotion.py` (new)
- `tests/test_state_promotion.py` (new)
- `docs/reports/M5-state-promotion-2026-09-19.md` (this file)

`AGENT_COMMS.md` was not modified. No commit, stage, reset or clean was performed.
The module imports only `hashlib`, `json`, `math`, `dataclasses`, `typing`,
`mc_photogrammetry.domain.models`, `reconstruction.plane_volume` and
`reconstruction.appearance_state_proposals`; the test suite asserts that by import
AST, by code-symbol inspection and by source-text exporter-symbol inspection.

## Repair pass (Sol findings, 2026-09-19)

Three defects from the first pass were repaired in place, within the same three
scoped paths. The public contract, the record shapes, the entry-point signature,
the closed reason enums and the accepted happy-path, omission-opt-in, atomicity
and isolation behavior are unchanged.

1. **A result-level `review` with nonempty proposed children could promote.**
   `_check_source` admitted both `PROPOSED` and `REVIEW` result statuses, and
   `_check_proposals` inspected only the per-cell status, so a self-consistent
   `FaceStateProposalResult(status="review")` carrying individually valid
   `"proposed"` children reached the block builder and promoted. Because the
   honest producer sets the result status to `PROPOSED` iff at least one child is
   proposed, the new gate fires exactly when the status is not `PROPOSED` *and* a
   proposed child is present, before the cell set is even compared: such a record
   is a typed `review_abstention` with
   `{"field": "result_status", "status", "reason"}`. An honest result-level
   review has only review children, falls through to its exact per-cell
   `{"cell", "reason"}` refusal, and is unchanged. Empty result-level reviews keep
   their `{"field": "result", "reason"}` refusal and a `"budget"` proposal result
   keeps its `source_mismatch`.

2. **Nested `CellClaim`s were not rebuilt before their enclosing cell.** The
   first-pass `_replay_volume` rebuilt each `SupportedCellObservation` from
   `_record_fields(cell)`, which passes the *same* claim objects through, so a
   claim built past `CellClaim.__post_init__` with a forged field (an empty
   `frame`, for example) could be resealed at the claim, cell and volume levels
   and still satisfy every record-level digest check. `_replay_claims` now
   rebuilds every nested claim through `plane_volume.CellClaim.build` before the
   enclosing `SupportedCellObservation` (and `CellConflict`) is rebuilt, so the
   accepted constructor's field validation runs no matter how the record was
   sealed. A resealed forged claim field is now a typed `digest_mismatch` on the
   `volume` before any budget decision, including with all-zero caps.

3. **A constructor-bypassed budget was trusted.** `promote_plane_states` checked
   only `isinstance(budget, PromotionBudget)`, so a budget built past
   `PromotionBudget.__post_init__` could carry forged caps or a stale
   `budget_sha256` and reach the budget comparisons. `_replay_budget` now
   rebuilds the digest from the declared caps before any input or budget work:
   malformed caps are a typed `invalid_input`/`{"field": "budget"}` and a
   mismatched digest is a typed `digest_mismatch`/`{"field": "budget"}`, even for
   an all-zero budget, so a forged budget is named as such rather than reported
   as `budget_exhausted`.

All three repairs are covered by new synthetic tests (43 total, up from 39).
No forbidden import, exporter call, manual metadata or new path was introduced;
`AGENT_COMMS.md` was not modified.

## Contract

### Entry point

`promote_plane_states(volume, proposals, *, budget, allow_omitted_unknown=False)`
returns **either** an immutable `PromotionResult` **or** an immutable
`PromotionRefusal`; it never raises for malformed input and never returns partial
blocks.

### Input discipline (before any budget)

- `volume` must be an accepted `plane_volume.PlaneVolumeResult`; `proposals` must
  be an accepted `appearance_state_proposals.FaceStateProposalResult`.
- The explicit `budget` must be an accepted `PromotionBudget` whose own
  `budget_sha256` replays from its declared caps. This replay runs **before** any
  input or budget work: a constructor-bypassed/forged budget is a typed
  `invalid_input` (malformed caps) or `digest_mismatch` (stale digest) even when
  every cap is zero.
- Both records are re-validated at this boundary: the tuple-valued collections
  (`cells`, `conflicts`, `omitted`, `support_bindings`, `proposals`) must be
  immutable tuples, and every nested record is rebuilt through its accepted
  constructor — every nested `CellClaim` through `CellClaim.build`, each
  `SupportedCellObservation`, each `CellConflict`, each `OmittedCell`, the
  `VolumeWork`, each `FaceStateProposal`, and the enclosing results — so a record
  built past `__post_init__` cannot ride along on a stale nested digest, and a
  forged claim field the enclosing record-level digest cannot see is still a
  typed `digest_mismatch`. The rebuilt records are compared against the stored
  top-level digests (`volume_sha256`, `payload_sha256`, `result_digest`).
- Source/version checks cover the volume schema, producer version, gauge state and
  axes, `export_bounds is None`, `complete is False` and the proposal result
  status (`source_mismatch` for an unsupported version, the wrong gauge, a
  present export bound, a completed volume, or a proposal call that itself
  refused).
- Geometry cells must be **unique and canonically ordered**; no `CellConflict`
  and no conflict-flagged cell may be present.
- Proposal cells must be **exactly** the geometry cells: a missing or extraneous
  proposal, or a duplicated proposal cell, is a typed refusal.
- Every geometry cell must carry a proposal with status `"proposed"` and the
  enclosing result must itself have status `"proposed"`. An honest result-level
  review (only review children) keeps its exact per-cell refusal; a result-level
  review carrying proposed children is a typed `review_abstention` with
  `{"field": "result_status", "status", "reason"}`. Any other per-cell or
  result-level review/abstention is a typed refusal.

### Typed outcomes

`PROMOTION_REVIEW_REASONS = ("review_abstention", "omitted_unknown")` carry
`kind="review"`; `PROMOTION_REFUSAL_REASONS = ("invalid_input",
"source_mismatch", "digest_mismatch", "conflict", "missing_proposal",
"extraneous_proposal", "invalid_state", "budget_exhausted")` carry
`kind="refusal"`. A `PromotionRefusal` is a frozen record with a canonical
`detail`, a `detail_sha256` and a self-replaying `refusal_sha256`; its constructor
refuses any `detail` carrying `cells` or `blocks`, and every budget refusal must
carry exactly `{"cap", "limit", "observed"}`.

### Budgets

`PromotionBudget` binds explicit non-negative `max_cells`, `max_proposals` and
`max_payload_bytes` (zero is a valid, immediately-refusing cap) in a
self-validating `budget_sha256`. The budget digest is replayed from its declared
caps before any input is inspected, so a constructor-bypassed or forged budget is
a typed `invalid_input`/`digest_mismatch` rather than a budget decision, even with
all-zero caps. All input, provenance and nested-digest validation then completes
**before** any budget decision, so a tampered record is a typed
`digest_mismatch`/`conflict`/`review_abstention`/`invalid_state` rather than
`budget_exhausted`. `max_cells` and `max_proposals` are enforced before any block
is built; `max_payload_bytes` is measured as the canonical UTF-8 size of the
emitted payload (which deliberately excludes `max_payload_bytes` itself), and an
excess discards every block. Each refusal carries the exact observed counter.

### Result, eligibility and confidence

`PromotionResult` binds `volume_sha256`, the proposal `result_digest`, every
per-cell `proposal_digest`/`evidence_digest`/`observation_digests`, each state
payload, `frame_ids` from the proposal's support ids, `provenance`, the
`PromotionResult`-level payload digests, the carried `normalization_offset`, and
`export_bounds=None`. `scale_state` and `normalization_state` are hard
`"unresolved"`.

`export_ready` is true exactly when every geometry cell carries a proposed state
and the geometry has no conflicts; `exact_state_eligible` additionally requires
**zero** omitted unknown cells. Because a result is only ever emitted after all
cells are proposed and conflict-free, `export_ready` is validated `True` on every
emitted record, and `exact_state_eligible == (export_ready and omitted_unknown ==
0)` is enforced by the record constructor.

`confidence` is derived deterministically from the proposal's own aggregate
evidence by `promotion_confidence(best_cost, margin)` (policy
`"margin-over-cost-separation-v1"`): `separation` is `margin / (margin +
best_cost)` (`1.0` for the infinite single-candidate margin, `0.0` for a zero
margin) and `quality` is `1 / (1 + best_cost)`; the product is clamped to
`[0, 1]` and is finite for every accepted proposed proposal.

`PromotionResult.to_block_volume()` builds a **fresh** `BlockVolume` on every
call: a fresh `BlockState` from the sealed `name` + canonical tuple of property
pairs, `ObservationKind.OBSERVED`, the deterministic confidence, and a fresh
`frame_ids` list. The record never stores a mutable volume or a live state, and
the adapter never calls an exporter.

## Files (SHA-256, lowercase; line counts)

```
eda22e079162669cbb820da647118ac263ed7fc37d40bba1d41481621e53100b  1039  src/mc_photogrammetry/reconstruction/state_promotion.py
2bcc4fe2a1bf4b99ac1c406c58bc65ad6c6a8042516630e44f5dc3d4c2d47303  1231  tests/test_state_promotion.py
```

`diff-scope`: the two repaired paths above and this report are the only paths
this repair pass touched; no existing source, CLI/UI, persistence, camera, pose,
exporter or test path was edited. Every other modified/untracked path in
`git status` is the pre-existing dirty baseline.

## Tests added

`tests/test_state_promotion.py` — 43 synthetic tests, all built in memory (no
fixture, image, manifest, camera or bound is read). The geometry is a
self-consistent synthetic `PlaneVolumeResult`; the proposal result is produced by
the real `propose_face_states` producer over synthetic `FaceObservation` patches:

- happy path: two cells promote to their exact stone states with frame ids,
  provenance, observation digests, proposal/evidence digests, the derived
  confidence, `export_ready`/`exact_state_eligible` true, carried offset and
  `export_bounds=None`;
- record is frozen, exposes no live volume, and every block/payload/result digest
  replays; inputs are not mutated;
- determinism: repeated calls are byte-identical, block coordinates are
  canonical, the promotion digest binds the volume source, and a resealed
  reversed proposal tuple is canonicalized to the same block order while still
  binding the new `result_digest`;
- missing, extraneous, duplicate and non-canonical geometry/proposal sets are
  typed refusals;
- per-cell review (`weak_margin`), result-level review (`no_observation`),
  proposal-source budget refusal and air/malformed states are typed
  review/refusal outcomes;
- result-status gate: a self-consistent result-level `review` carrying two
  individually `proposed` children is a typed `review_abstention` with the
  `result_status` detail (with and without a reason), refused before an all-zero
  budget, while the honest per-cell review keeps its exact cell/reason detail;
- conflicts: a forged `conflicts` tuple and a conflict-flagged cell both refuse;
- omitted unknown: default typed `review`/`omitted_unknown` with no blocks, and
  the explicit opt-in preserves the count, keeps `export_ready=True`, sets
  `exact_state_eligible=False`, and emits no invented block for the omitted cell;
- tampered nested digests: geometry cell, work record, proposal digest, evidence
  digest and result payload digest are all refused as `digest_mismatch` (the
  nested proposal case is the one the result-level digest deliberately cannot
  see); a claim forged past its constructor (empty `frame`) and resealed at the
  claim, cell and volume levels is refused as `digest_mismatch` on the `volume`
  before an all-zero budget, while a record-only cell rebuild is shown to accept
  it; a resealed stale claim digest is likewise refused;
- forged budget: a cloned budget with a stale `budget_sha256` is a typed
  `digest_mismatch` (including all-zero caps and before the volume is inspected),
  and constructor-bypassed negative/non-integer caps with a recomputed digest are
  a typed `invalid_input`;
- budget atomicity: exact-boundary acceptance and one-under refusal for
  `max_cells`, `max_proposals` and `max_payload_bytes`, with exact observed
  counters and no blocks; validation precedes all-zero budgets for malformed,
  conflicted, review, missing-proposal and air-state inputs;
- fresh `BlockVolume` mutation isolation and a list-backed proposal state
  snapshotted into an immutable tuple;
- isolation boundary: allowed import roots only, no forbidden module/import, no
  `open`/`exec`/`eval`/`compile`/`__import__`, no exporter symbol or `.export(`
  call, no manual/manifest/camera/F3 code symbol, the exact entry-point
  signature, and the resolved budget/scale/normalization constants.

## Checks run

```
$ .\.venv\Scripts\python.exe -m py_compile `
    src\mc_photogrammetry\reconstruction\state_promotion.py `
    tests\test_state_promotion.py                                   # exit 0

$ .\.venv\Scripts\python.exe -m pytest tests\test_state_promotion.py `
    -q -p no:cacheprovider -o addopts=                              # 43 passed

$ .\.venv\Scripts\python.exe -m pytest tests\test_plane_volume.py `
    tests\test_appearance_state_proposals.py tests\test_face_observation.py `
    tests\test_litematic.py -q -p no:cacheprovider -o addopts=      # 144 passed (2:22)
```

The four adjacent suites were re-run together against the repaired source (with an
extended harness timeout); the combined run is green. No failure was observed in
any accepted suite.

## Deviations, limits and uncertainties

- **Omitted-unknown default vs. the eligibility rule.** The contract says an
  unknown geometry omission yields a typed atomic review/refusal *and* that
  omissions make `exact_state_eligible` false. These are both honoured through one
  explicit keyword, `allow_omitted_unknown` (default `False`): the default is a
  typed `review`/`omitted_unknown` refusal carrying no blocks, while the literal
  `True` opt-in emits only the known cells (never an invented block for the
  omitted cell), preserves `omitted_unknown`, keeps `export_ready=True` and sets
  `exact_state_eligible=False`. This keyword is the one API element not spelled
  out verbatim in the task; it is flagged for Sol.
- **`CellConflict` is unreachable from validated records.** As the accepted M4
  bridge documents, a validated `CellClaim` always bounds its own coordinate, so
  `CellConflict` is a reserved type that no valid record can carry. The conflict
  gate is therefore exercised with forged clones (a `conflicts` tuple and a
  flagged cell), and the gate runs *before* the nested-digest replay so a forged
  conflict is reported as `conflict`, not `digest_mismatch`.
- **No upstream source replay.** Unlike `plane_volume`, this adapter cannot rerun
  the upstream geometry/proposal producers: no universe, placement, supports or
  observations are in scope. It validates that every nested digest replays and
  binds the source digests into its own record. A record that is self-consistent
  but produced by a resealed alternative proposal call is indistinguishable here;
  that is outside this seam's trust scope and is bound, not hidden, by
  `proposal_result_digest` and the per-cell proposal digests.
- **Digest-refusal detail.** A rebuild failure reports
  `{"field": "volume"|"proposals", "detail": "<ExceptionType>"}`; tests assert the
  field and the reason, not the exception name. The extra field aids diagnosis and
  does not affect atomicity.
- **`export_ready` is always true on an emitted result.** Since a result is only
  emitted after all cells are proposed and conflict-free, the constructor
  validates `export_ready is True`; the flag exists so a forged record cannot
  claim eligibility it does not have. `exact_state_eligible` is the only flag that
  varies, and only with omitted unknown cells.
- **Zero budgets are legal.** `PromotionBudget` accepts zero caps so the
  "validation strictly before any budget" property is directly testable; a valid
  input with a zero cap is an atomic `budget_exhausted`.
- **Result-status gate is exact, not blanket.** An honest producer sets the
  result status to `PROPOSED` iff at least one child is proposed, so an honest
  result-level review has only review children and keeps its exact per-cell
  `{"cell", "reason"}` evidence. The new gate therefore fires only when the
  status is not `PROPOSED` *and* a proposed child is present — a combination the
  producer cannot emit — and does so before the cell set is compared.
- **Budget replay precedes input inspection.** `_replay_budget` runs immediately
  after the `PromotionBudget` type check and before `allow_omitted_unknown`, the
  volume and the proposal result, so a forged budget is named even when another
  input is unusable. This is intentional precedence, not a change to the
  input-validation-before-budget property (the replay is itself validation).
- **Defensive re-validation.** The adapter re-checks fields the upstream
  constructors already check because a hostile caller can build a record past
  `__post_init__`. This is deliberate defence in depth, not a second source of
  truth. `_replay_claims` and `_replay_budget` exist because record-level digests
  cannot see a forged nested field or a constructor-bypassed budget.
- No image-only or fixture evidence was produced or scored. No occupancy,
  hidden-block, air, terrain, geometry, camera, coordinate, manifest, F3, scale,
  bound or manual metadata is read or inferred. M5 production integration and the
  production gates remain open.

Stopping here for Sol review and independent verification.
