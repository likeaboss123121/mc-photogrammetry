# M4 V2-B runtime accounting implementation, validation repair and source replay (2026-09-19)

Implementation task: `v2b-runtime-accounting-implementation-2026-09-19`
Route: `route-v2b-runtime-accounting-implementation-2026-09-19-0edd5b639cd2`
Repair task: `v2b-runtime-accounting-validation-repair-2026-09-19`
Route: `route-v2b-runtime-accounting-validation-repair-2026-09-19-bd0b4c708132`
Repair task: `v2b-source-replay-and-branch-repair-2026-09-19`
Route: `route-v2b-source-replay-and-branch-repair-2026-09-19-1c5c75250b53`
(`jev-reference-1`, role `developer`, profile `sol`)
Worker: Sol-equivalent CLI worker (DeepSeek), bounded implementation + two repairs
Status: **source-replay and branch-eligibility repair complete; focused,
replay-mutation, reachable-fixture and adjacent checks pass; stopped for
Manager/Astra review.**

Write scope used (exactly three routed paths):

- `src/mc_photogrammetry/reconstruction/plane_phase_anchor.py`
- `tests/test_plane_phase_anchor.py`
- `docs/reports/M4-V2B-runtime-accounting-implementation-2026-09-19.md` (this file)

No other source, test, report, contract, calibration/evaluation artifact,
handoff, comms, CLI/UI, GPU, launcher, schema, security, placement, volume,
scoring or wiring file was created or edited. No git stage/commit/reset/clean.
The 2026-09-19 repair edits only the two routed code/test files and this report.

## Accepted amendment implemented (first pass, 2026-09-19)

Astra's exact amendment (`runs/v2b-runtime-accounting-design-2026-09-19/attempt-01/decision.md`
lines 30-36, routed by the active task) is implemented as written:

1. **New contracts.** `V2BAnchorBudget.max_accepted_frame_supports` (positive
   exact integer, no default) and `V2BAnchorWork.accepted_frame_supports` (exact
   nonnegative integer) were added, both bound into the canonical payload and
   `*_sha256` digests.
2. **Planned upper bound `U`.** `U = sum_w |H_w| * |{frame(o): o in Q_w}|`
   (deterministic plan in `_plan_work`); it is recorded in
   `planned_work.accepted_frame_supports` and is **never** compared to
   `max_accepted_frame_supports` at preflight.
3. **Runtime counter `K`.** `K` is the cardinality of unique
   `(world_axis, hypothesis, frame)` triples whose frozen test accepted at least
   one processed observation. A triple is counted once at insertion, retained
   after later rejections, and the run stops atomically on the first `K > cap`.
4. **Preflight scope.** `_STATIC_WORK_FIELDS` is the fifteen existing static
   counters; `_work_within_cap`/`_first_over_cap` iterate those only.
5. **Truthful snapshots.** `_runtime_work` builds the exact runtime snapshot over
   the observations actually processed; the fabricated `observed = planned + 1`
   path is deleted.
6. **Held-out accounting.** `held_out_samples_per_reading` is the maximum over
   processed readings of the distinct ordered D-coordinate union
   (`_d_coordinate_union_size`).
7. **Precedence.** Invalid input refuses before any cap; planned exhaustion
   carries the exact plan.

## Validation repair (this task, 2026-09-19)

Objective: make the amendment fail closed when records are freshly resealed with
contradictory counters, aliases, snapshots, branch details, or cap types. A
freshly resealed record recomputes its own digests, so every repair below
cross-checks the mutated field against an independently retained record or an
alias rather than against a digest.

### 1. Strict public cap typing (`make_anchor_budget`, `_make_work`)

`make_anchor_budget` previously coerced every cap with `int(...)`, so
`True -> 1`, `"9" -> 9`, and `9.0 -> 9` all passed as valid public caps. It now
requires every cap to already be an exact `int >= 1` and raises
`v2b_input_invalid` **before** any coercion. `_make_work` likewise rejects
non-exact-integer counters instead of coercing them. `V2BAnchorBudget` and
`V2BAnchorWork` constructors already used `type(value) is int`; the digest
binding is unchanged.

### 2. Constructor binding of work snapshots (`V2BAnchorPrediction`)

`V2BAnchorPrediction.__post_init__` now calls `_validate_retained_binding`,
which re-derives all of the following from the retained `axes`, `hypotheses` and
`observations` and requires the two work snapshots to match exactly:

- per-axis and total hypothesis counts (`accepted + rejected` per axis must equal
  the retained hypothesis set, with no duplicate or extra digest);
- `contributor_refs_per_hypothesis` = max `contributor_count` and
  `contributor_refs_total` = sum `contributor_count`;
- `runtime_work.observations == len(observations)`;
- `runtime_work.accepted_frame_supports == K`, recomputed by re-running the
  frozen absolute test (`_observation_accepts`) over the retained observations;
- `planned_work.accepted_frame_supports == U`, recomputed as
  `sum_w |H_w| * |{frame(o): o in Q_w}|`;
- `K <= U` and `K <= runtime observations` (the `K == limit + 1 <= U` branch
  obligation is enforced in the result constructor below);
- every completed-run shared counter is identical between the planned and
  runtime snapshots (`_PREDICTION_SHARED_FIELDS`, including
  `held_out_samples_per_reading`), because a completed run processes every
  planned observation key;
- the held-out maximum lies inside the source-derived per-reading union bounds
  `max_r max_a |D_a(r)| <= held_out <= max_r sum_a |D_a(r)|`, recovered from the
  retained per-axis `held_out_samples` coverage counts.
- every observation maps to a retained `(world_axis, hypothesis_sha256)`.

`V2BAnchorWork.__post_init__` additionally enforces the two exact alias
relations that hold on every branch: `boundary_interior_terms == 2 *
observations` and `replay_crosslinks == 2 * source_records`.

### 3. Branch consistency (`V2BAnchorResult`)

`V2BAnchorResult.__post_init__` now binds the completed branches' outer `work`
to `prediction.runtime_work`, and `_validate_budget_branch` validates every
`budget_exhausted` branch:

- `work` is present and a `V2BAnchorWork` (rejects `work=None`);
- the refusal detail carries only `budget_stage`/`cap`/`limit`/`observed` (and
  the two snapshots on the runtime branch), so no hypothesis, observation, or
  anchor evidence can be smuggled into an evidence-free branch;
- `detail["budget_stage"] == result.budget_stage`, `detail["cap"]` is a known
  budget cap, `limit`/`observed` are exact `int`s with `observed > limit`;
- `observed` equals the work counter named by `cap` (matching stage/cap/observed
  aliases);
- **planned** exhaustion carries the exact planned snapshot as `work` and
  neither `planned_work` nor `runtime_work` in the detail;
- **runtime** exhaustion carries both snapshots in the detail, the detail
  runtime snapshot is byte-equal to `work`, `K <= U`, `K <= observations`,
  `runtime observations <= planned observations`, the runtime held-out maximum
  is `<=` the planned maximum, all shared counters are equal, and for
  `max_accepted_frame_supports` the stop is exact: `K == limit + 1 <= U`.

### 4. Freshly resealed mutation tests

Six new tests replace the single prior reseal test (suite 34 -> 39):

- `test_resealed_work_alias_relations_fail_closed` — stale digest; resealed
  `boundary_interior_terms != 2*observations`; resealed
  `replay_crosslinks != 2*source_records`; negative and bool/float/str counters.
- `test_make_anchor_budget_rejects_non_exact_integer_caps` — `True`, `False`,
  `"9"`, `9.0`, `9.5`, `None`, `(9,)` for both a static cap and the amended cap,
  plus a digest-identical valid rebuild.
- `test_resealed_prediction_k_above_upper_bound_fails_closed` — `K = U + 1`,
  contradicted `hypotheses_total`, contradicted `contributor_refs_total`.
- `test_resealed_prediction_held_out_and_observation_mismatch` — runtime held-out
  disagreed with planned (191 vs 192); both snapshots fabricated above the
  source-derived upper bound; observation count bumped with its boundary-term
  alias kept consistent.
- `test_resealed_outer_work_and_completed_prediction_binding` — outer `work`
  resealed to a different record; outer `work=None`.
- `test_resealed_budget_branch_consistency_fails_closed` — malformed detail with
  `budget_stage=None`; wrong `observed`; forged cap that skipped the exact
  first-over stop; wrong result stage; `work=None`; detail runtime snapshot
  disagreeing with `work`; detail planned upper bound below `K`; evidence key in
  the detail; runtime snapshot added to a planned branch; invalid cap types.
  A control asserts the untouched accepted branch still reseals and validates.

Every mutation that succeeds at the low-level record constructor is shown to be
rejected by the enclosing semantic constructor (`assert resealed...` proves the
attacker rebuilt the digest).

### Preserved behaviour

All 33 non-mutation V2-B tests pass unchanged, including the public five-frame
ring fixture and its exact counters (`U = K = 10`, held-out union 192, runtime
cap 9 exhausts on the tenth insertion, caps 10/11 complete, D-only control
`U = 10, K = 0`), the fallback static cap boundaries, atomic
invalid-before-budget precedence, schema layout, canonical ordering, calibration
binding, and the AST/isolation boundary. No existing record field was added,
removed, or reordered; the repair adds only constructor validation and strict
input typing.

## Source-bound replay and branch eligibility repair (this task, 2026-09-19)

Objective: close the two gaps the previous repair recorded as unresolved (report
§ Unresolved concerns) without changing the V2-B schema amendment or any product
scope. A retained-record constructor can bind a freshly resealed record only to
the evidence it already holds; it cannot recover the exact held-out D-coordinate
union or the exact runtime insertion prefix, and it had no branch-eligibility
rule tying a planned refusal to a cap the preflight can actually evaluate.

### 1. `validate_phase_anchor_replay` (source-bound replay validator)

New public function in the routed module. It accepts the claimed
`V2BAnchorResult`, the original accepted upstream sources (`c2rb_result`,
`vector_result`, `support`, `p1d_result`, `matches`, `phase_evidence`,
`readings`, `signatures`, tolerances, `v2a_result`), the original validated
`split_sources`, the caller's `budget` and the optional `calibration` -- exactly
the context the existing analysis path requires. It re-runs
`analyze_phase_anchor` on those sources and returns
`replay.to_json_bytes() == claimed.to_json_bytes()`, i.e. a canonical
result-byte comparison. A non-`V2BAnchorResult` claim, or an analysis that raises
rather than returning a record, returns `False`.

This closes the two recorded gaps:

- **Held-out snapshot forgery.** A prediction whose planned and runtime
  `held_out_samples_per_reading` were both resealed from 192 to 191 (or to any
  other value inside the retained coverage bounds `[128, 256]`) still satisfies
  every constructor invariant, but the replay recomputes the real union (192)
  from the original split sources and the bytes differ, so it is rejected.
- **Runtime-prefix forgery.** A runtime stop rewritten from the real cap-9 stop
  (`limit = 9`, `K = observed = 10`) to `limit = 8`, `K = observed = 9`, with
  `observations = 10` still satisfies `K == limit + 1 <= U` and every snapshot
  alias, but the replay produces the real stop and the bytes differ.

The validator inherits the analysis path's invalid-input-before-budget
precedence (invalid split sources refuse before any cap is evaluated, so a
forged `budget_exhausted` record over invalid sources never validates) and
consumes no user-supplied geometry, manifest or metadata, so no advanced-only
boundary moved.

### 2. Branch-specific cap eligibility and strict refusal limits

`V2BAnchorResult._validate_budget_branch` now enforces:

- **Planned branches may only name static preflight counters.** A new module
  constant `_PLANNED_EXHAUSTIBLE_CAPS` is exactly the caps of
  `_STATIC_WORK_FIELDS` (every public cap except
  `max_accepted_frame_supports`). `max_accepted_frame_supports` is runtime-only
  -- `_work_within_cap`/`_first_over_cap` deliberately skip it and `_plan_work`
  never compares it -- so a `planned` refusal whose `cap` is
  `max_accepted_frame_supports` is refused as a branch forgery. Genuine planned
  exhaustion by any static cap is unchanged.
- **Refusal limits below 1 are rejected before branch validation.** The bounds
  check is now `limit < 1` (was `limit < 0`) and is evaluated before the cap and
  stage/detail checks, so `limit = 0` or `limit = -1` is refused on both the
  planned and runtime branches. Every public cap is a positive exact integer, so
  no genuine stop can carry such a limit.

### 3. Replay and branch tests (suite 39 -> 46)

Six new tests plus one full-suite expansion:

- `test_source_bound_replay_accepts_unchanged_completed_result` -- positive
  control for the unchanged `partial` result (and a non-record claim is
  rejected, never raised).
- `test_source_bound_replay_accepts_genuine_planned_and_runtime_exhaustion` --
  positive controls for the genuine held-out cap-191 planned stop and the genuine
  cap-9 runtime stop.
- `test_source_bound_replay_rejects_resealed_held_out_snapshots` -- the
  192 -> 191 both-snapshot reseal is shown to build a valid prediction and
  result first, then is rejected by replay.
- `test_source_bound_replay_rejects_forged_runtime_support_stop` -- the
  `limit = 8`/`K = observed = 9`/`observations = 10` rewrite is shown to build a
  valid runtime branch first, then is rejected by replay; the genuine stop
  replays clean.
- `test_planned_branch_cannot_claim_runtime_only_support_cap` -- planned
  `max_accepted_frame_supports` exhaustion is refused; the genuine static planned
  branch still validates.
- `test_budget_refusal_limit_below_one_is_rejected` -- `0`/`-1` limits are
  refused on both branches; the genuine limit-1 planned branch validates.
- `test_source_bound_replay_preserves_invalid_input_before_budget` -- an invalid
  final source with tiny caps refuses before budget and that exact refusal
  replays; a genuine budget branch never validates against the invalid sources.

### 4. Preserved behaviour

All 39 prior tests pass unchanged (33 original + 6 previous resealed-mutation
tests), including the public five-frame ring fixture and its exact counters, the
fallback static cap boundaries, atomic invalid-before-budget precedence, schema
layout, canonical ordering, calibration binding and the AST/isolation boundary.
No record field was added, removed or reordered; no existing constructor alias,
K/U check, strict cap typing or atomic runtime behaviour changed. The only
implementation additions are the public validator, the
`_PLANNED_EXHAUSTIBLE_CAPS` constant, the planned-cap eligibility check and the
`limit < 1` refusal-limit check.

## Independent five-frame ring fixture (unchanged)

| Quantity | Value |
| --- | --- |
| Planned/runtime `accepted_frame_supports` (U = K) | 10 |
| Planned/runtime `observations` | 10 |
| Planned/runtime `boundary_interior_terms` | 20 |
| Planned/runtime `d_profile_terms` | 480 |
| Planned/runtime `payload_bytes` | 4290 |
| Planned/runtime `replay_crosslinks` | 10 |
| Planned/runtime `pixel_samples` | 1280 |
| Planned/runtime `held_out_samples_per_reading` | 192 |
| Planned/runtime `split_reconstructions` / `source_records` | 5 |
| Per-axis retained `held_out_samples` (u, v) / union bounds | 128, 128 / [128, 256] |
| Runtime cap 9 | `budget_exhausted`/`runtime`, limit 9, observed 10 |
| Runtime caps 10 / 11 | `partial`, runtime K = 10 |
| Held-out cap 191 | `budget_exhausted`/`planned`, limit 191, observed 192 |
| Held-out caps 192 / 193 | `partial` |
| D-only impulse removal (structural patch) | U = 10, K = 0, held-out 192 |

Reference digests (unchanged by the repair; default `_anchor_budget` computed by
the test helper): budget
`37ac595f4bb205ed63a82339885ac4a388635dbda9524939caf64f43db8995fa`;
public-fixture planned and runtime `work_sha256`
`26996d69bb8c24bc8946d599ac96801753ea8b90948e753ae11da2a2be0c6e69`;
public-fixture `result_sha256`
`0b85a2c90ff3036ecf8e4e933ed7b9548a19c05e1571b64cceb0e7b59c89ac79`.

## Exact commands and outcomes (final code, all three passes)

All foreground, project root, frozen interpreter `.venv\Scripts\python.exe`,
`-p no:cacheprovider -o addopts=`:

1. `py_compile src\mc_photogrammetry\reconstruction\plane_phase_anchor.py tests\test_plane_phase_anchor.py`
   -> exit 0, no output.
2. `pytest tests\test_plane_phase_anchor.py -q` -> **46 passed in 10.03s**, exit 0
   (33 original + 6 prior resealed-mutation + 7 source-replay/branch tests).
3. `pytest tests\test_plane_phase_anchor.py -q -k "replay or resealed or non_exact_integer or planned_branch or limit_below_one"`
   -> **16 passed, 30 deselected in 3.11s**, exit 0 (replay/replay-mutation
   matrix).
4. `pytest tests\test_plane_phase_anchor.py -q -k "frame_support or runtime_exhaustion or null_fixture or held_out or unequal"`
   -> **8 passed, 38 deselected in 1.63s**, exit 0 (reachable runtime fixture).
5. `pytest tests\test_plane_endpoint_provenance.py tests\test_lattice_planes.py -q`
   -> **102 passed in 44.95s**, exit 0 (adjacent endpoint/lattice).
6. `pytest tests\test_split_phase_producer.py tests\test_split_phase_calibration.py -q`
   -> **72 passed in 84.90s**, exit 0 (adjacent P0/calibration).
7. `git status --porcelain -- <three routed paths>` -> the three routed V2-B
   paths (all still untracked new artifacts); `git status --porcelain` shows no
   newly changed tracked file attributable to this task. `py_compile`/`pytest`
   refreshed standard `__pycache__` caches only.

No warnings, errors, skips or non-zero exits. The prior pass's counts (39/6/6/
102/72) were reproduced before the new repair and the two adjacent suites are
byte-for-byte the same accepted upstream modules (hashes below).

## Hashes

Changed (this source-replay repair run; the report's own hash is recomputed
after its final edit and reported in the run result):

| Path | Lines | SHA-256 |
| --- | --- | --- |
| `src/mc_photogrammetry/reconstruction/plane_phase_anchor.py` | 3255 | `17b169b7ebce339a320b4edf3b34266eccc9422f24386c337e2c4ec3dde0e1fe` |
| `tests/test_plane_phase_anchor.py` | 1487 | `664c8a17d4bd5b8edfbda15463221bc63950f0785df35da448db5e6096bf7903` |

Previous repair-run hashes (this-file history) were source
`344fd4d3759182253dd09549ff948c0133006e50f6ffe7ad07bdf40e3fd19de5` (3180 lines)
and tests `c7a62696360fac34f78a32d6d1ab4005be2b45dc588249e52751585d5e9b6c67`
(1299 lines); the originally accepted V2-B implementation hashes were source
`206e83586119c367398e6d31a5e4cdd08f521021a736191800fab4bc7d269e1b` and tests
`bd2ab100cc2e2d1a89646d4cd801e6b30a60fd76a285a1708eeffb92a0ff30f1`
(`runs/v2b-verification-evidence-2026-09-19/attempt-02/verification.md`); the two
routed V2-B files changed exactly as authorized and no upstream module did.

Unchanged upstream accepted hashes (recomputed, all match the accepted record):

| Path | SHA-256 |
| --- | --- |
| `src/.../lattice_planes.py` | `82ada0ccc0f92b580342aa35f8333d1013910f7cfa91eada0ed7d5e7c95831e7` |
| `src/.../plane_endpoint_provenance.py` | `3cb6aacea4d9e6873d0b10edd4ea7539b1646a1e75475cef2006f81caa3316e7` |
| `src/.../plane_relative_translation.py` | `f703a28d275d55e4a85fad7fde0bfd612928f8d54a5489c0ea097fb42efbcd3e` |
| `src/.../plane_residue.py` | `55ec0cad8fe1088b7b33b1f57c4d58d3d3b912f79765a1f391553ac7b845be0d` |
| `src/.../plane_vector_domain.py` | `72ee29fb936cffd7f8a154c18d6f6bab18876d57ad66f45bc316e39f7d45e7f5` |
| `src/.../plane_axis_support.py` | `18598afb2096b504464411e3a2cbf6a2e960cb2601c4dd2fb8f77d4b46c91f3f` |
| `src/.../plane_graph.py` | `c5245680b469f93ded5525bb80b0ddf065b46a64e1d88d66cc1fa93115522e92` |
| `tests/test_plane_endpoint_provenance.py` | `967559f7b20184442f558c34af803128e85ab7596fd50c2489dd8a16c5270f95` |
| `tests/test_split_phase_producer.py` | `e5e17b9ccc8203394ebe1f9f78601176f33c041f34620f214d1b02457b7a9b31` |
| `tests/test_split_phase_calibration.py` | `c71dd150488baa8336c8d6d8d1e0ff6483832f4acdff2efb2f72b7ca10165d6d` |

`docs/reports/M4-V2B-physical-phase-anchor-contract-2026-09-14.md` (frozen
contract) was **not** edited; the accepted amendment is recorded in the task and
Astra decision records, not by rewriting the frozen contract.

## Deviations

1. **Runtime refusal detail shape (amended branch).** Runtime exhaustion places
   both `planned_work` and `runtime_work` records inside the bounded refusal
   detail so the contract's "retain exact planned and runtime snapshots"
   obligation can be met when `V2BAnchorResult` has a single `work` slot; planned
   exhaustion keeps the exact plan as `work` and carries neither snapshot in the
   detail. The result constructor now forbids any other detail key, so this
   bounded shape is also the only accepted evidence-free branch shape.
2. **Held-out alias relations added to `V2BAnchorWork`.** The two exact formulas
   (`boundary_interior_terms == 2*observations`,
   `replay_crosslinks == 2*source_records`) are now constructor invariants.
   They held on every branch before this repair; they are enforced, not changed.
3. **Two internal budget stops removed (first pass).** `_derive_hypotheses` and
   `_validate_split_sources` no longer raise `_V2BBudgetStop`; the caps are
   enforced by `_plan_work`, matching contract §10 precedence and removing the
   old `observed=0`/`work=None` planned-exhaustion defect. `budget` remains an
   accepted (now unused) parameter of both helpers.
4. **`_exceeds` deleted** as dead code in the first pass.

## Unresolved concerns (honest limits; not blockers)

Closed by the 2026-09-19 source-replay repair, recorded for history:

- **Exact D-coordinate union.** A constructor can bind the held-out maximum only
  to the source-derived bounds `[max_a |D_a|, sum_a |D_a|]` per reading, so a
  reseal that moves **both** snapshots to a different in-bounds value (e.g.
  192 -> 191, bounds [128, 256]) is invisible to it. `validate_phase_anchor_replay`
  now recovers the exact union by replaying the original split sources, so the
  both-snapshot reseal is rejected.
- **Whole-record self-consistent runtime stop.** A forgery that consistently
  rewrites `limit`, `observed`, `K`, `U` and both snapshots passes every
  retained-record alias, but `validate_phase_anchor_replay` recomputes the real
  runtime prefix and cap and rejects it. The two gaps are therefore closed at the
  validation boundary while the constructors keep their bounded,
  source-independent invariants.

Remaining limits (not blockers):

- The runtime checks for `observations`/`boundary_interior_terms` remain in
  `_determine` but are dominated by the planned preflight, so they cannot be
  independently exercised; reachable runtime exhaustion is the amended
  `max_accepted_frame_supports` unit.
- Unequal per-reading D unions cannot be hosted by a single accepted bundle; the
  maximum-union rule is exercised with two real accepted P0 results (16x16 union
  192 and 20x20 union 300) plus the full-path equal-size fixture.
- `test_every_static_cap_boundary_w_minus_one_w_w_plus_one` cannot construct the
  `W-1` cap for `max_hypotheses_per_axis` because its observed value is 1 and
  caps must be positive; the constructor rejection is asserted instead.
- `max_accepted_frame_supports` is enforced only at insertion;
  `_work_within_cap` deliberately ignores it, so a resealed work record with
  `accepted_frame_supports > cap` is constructible as a low-level record, but the
  enclosing prediction and branch constructors now reject it.

## Blocker

None. No failing check, hash mismatch, source-scope violation or architectural
inconsistency was found. Placement, volume construction, product wiring,
evaluation scoring and promotion remain blocked and untouched. Stop for
manager/Astra review.

## Files written

- `src/mc_photogrammetry/reconstruction/plane_phase_anchor.py` (implementation +
  validation repair + source-bound replay validator and branch eligibility)
- `tests/test_plane_phase_anchor.py` (tests + resealed-mutation suite + replay
  and branch-eligibility tests)
- `docs/reports/M4-V2B-runtime-accounting-implementation-2026-09-19.md` (this
  report)

Side-effect note: the required `py_compile`/`pytest` runs refreshed standard
`__pycache__` bytecode caches only.
