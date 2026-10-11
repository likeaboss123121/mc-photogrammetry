# M4 V1 aligned region support implementation — 2026-09-19

Implementation task: `m4-v1-aligned-region-support-implementation-2026-09-19`
Original route: `route-m4-v1-aligned-region-support-implementation-2026-09-19-c36f96e6236e`
Repair task 1: `m4-v1-aligned-region-support-repair-2026-09-19`
Repair route 1: `route-m4-v1-aligned-region-support-implementation-2026-09-19-d97bd3522353`
Repair task 2: `m4-v1-refusal-invariant-repair-2026-09-19`
Repair route 2: `route-m4-v1-refusal-invariant-repair-2026-09-19-349dbb745bdb`
Upstream design (approved): `docs/reports/M4-placement-volume-bridge-contract-2026-09-19.md`
Worker: Sol-equivalent CLI worker (DeepSeek), bounded implementation + two repairs
Status: **V1 repaired against all four Astra findings and the Astra
conditional-review refusal-invariant defect; focused, adjacent provenance/mask,
and broader plane checks pass; stopped for Astra read-only review.**

Write scope used (exactly the three routed paths):

- `src/mc_photogrammetry/reconstruction/plane_region_support.py`
- `tests/test_plane_region_support.py`
- `docs/reports/M4-V1-aligned-region-support-implementation-2026-09-19.md`
  (this file)

No accepted V2-B source, placement, volume, CLI/UI, persistence, litematic
wiring, manual-metadata or promotion-gate file was created or edited. No git
stage/commit/reset/clean. `py_compile`/`pytest` refreshed standard `__pycache__`
bytecode caches only. All written files are UTF-8.

## Repair summary (Astra rejection → bounded fix)

Astra rejected the first V1 delivery on four bounded defects. The repair keeps
the accepted policy (conservative full-area support, no air, no polygon
rasterization, atomic budget refusal) and changes only the four mechanisms.

| # | Finding | Repair | Regression tests |
| --- | --- | --- | --- |
| 1 | Rounded-float projection is non-conservative | `_exact_homography` composes the accepted closed form in exact rational arithmetic from the declared calibration inputs; `rectification_homography` is called only to detect an overflowing projection, and its `float64` coefficients are never fed to the cell geometry. A cell whose exact projected edge lies outside the image/bounds can never be `supported`. | `test_projection_uses_exact_rational_coefficients_not_rounded_float`, `test_exact_negative_projected_edge_is_never_supported`, `test_exact_homography_matches_the_accepted_closed_form` |
| 2 | `ReadingIdentity` not content-bound | `build_aligned_region_support` now takes the canonical `reading_universe` the identity was derived from and recomputes the V2-A identity from the canonical reading, comparing every field; the universe entry must then content-equal the reading (covering rectification/calibration, which V2-A's order key omits). The sealed record adds `reading_binding_sha256`, a cross-link over identity + calibration + retained alternative + source-region mask digest + image digest + image shape, verified by reconstruction. Alternatives are canonicalized before identity/content checks so digests stay insertion-order independent. | `test_changed_reading_content_cannot_reuse_the_old_identity` (10 cases), `test_support_record_rejects_resealed_identity_content_mutations`, `test_reordered_alternatives_share_the_canonical_identity` |
| 3 | Resealed records/refusals accepted contradictions | `AlignedRegionSupport.__post_init__` requires the row/column labels to be exactly the two tangent axes of `reading_identity.reading_axis` (the normal can never be a tangent label). `AlignedSupportRefusal.__post_init__` enforces reason-specific invariants: budget refusals require observed work, exactly `{cap, limit, observed}`, a real support cap, positive `limit`/`observed` with `observed > limit`, `observed` matching the work counter for runtime stops, and a planned `max_cells` stop when no cell was examined; any refusal rejects an embedded `cells` payload; non-budget refusals must not carry work evidence. | `test_support_record_rejects_normal_axis_as_tangent_label`, `test_resealed_budget_refusal_rejects_a_nonexistent_cap`, `test_resealed_budget_refusal_rejects_nonpositive_observed_limits`, `test_resealed_budget_refusal_requires_work_evidence`, `test_resealed_budget_refusal_rejects_embedded_cells`, `test_resealed_non_budget_refusal_rejects_work_evidence`, `test_resealed_budget_refusal_observed_must_match_work_counter` |
| 4 | Invalid projection lost precedence to budget | Homography construction/validation (accepted-float finiteness plus exact-rational degeneracy) now completes before the planned `max_cells` decision, so an overflowing or degenerate homography returns the deterministic `calibration_mismatch` refusal for every budget size, never `budget_exhausted`. | `test_non_finite_homography_precedes_budget`, `test_degenerate_homography_precedes_budget`, `test_valid_homography_still_exhausts_a_small_budget` |

## Final refusal-invariant repair (Astra conditional acceptance)

Astra conditionally accepted the four repairs above and left one bounded defect:
a planned `budget_exhausted` refusal could be freshly resealed with nonzero work
counters (all digests recomputed) and still construct. Two semantic
work/refusal invariants close it. Runtime exhaustion semantics, exact
projection, content-bound identity, validation-before-budget precedence, the
no-air/no-polygon policy and every scope exclusion are unchanged.

| Repair | Mechanism | Regressions |
| --- | --- | --- |
| Planned refusal must carry zero work | `AlignedSupportRefusal.__post_init__` treats `max_cells` as the only ex-ante cap: a `budget_exhausted` whose cap is `max_cells` or whose `cells_examined` is zero must have every one of the seven work counters exactly zero. A runtime stop keeps the existing `observed`/cap-counter binding. | `test_resealed_planned_budget_refusal_rejects_nonzero_work_counters` (6 counters), `test_resealed_max_cells_refusal_rejects_examined_work`, `test_resealed_genuine_planned_and_runtime_budget_refusals_still_construct` |
| Work records enforce semantic relations | `AlignedSupportWork.__post_init__` requires `supported_cells + unknown_cells <= cells_examined` and `frame_clips <= cells_examined` alongside the existing non-negative counters; `AlignedSupportRefusal.__post_init__` and `AlignedRegionSupport.__post_init__` re-check the same relations so a deliberately forged work object cannot be smuggled into a freshly resealed record. Existing cap relations (`observed`/cap-counter binding, grid/state cross-checks) are preserved. | `test_resealed_runtime_budget_refusal_rejects_state_count_overflow`, `test_resealed_runtime_budget_refusal_rejects_frame_clip_overflow`, `test_work_record_rejects_impossible_counter_relations`, `test_resealed_support_record_rejects_forged_work_relation` |

Every new contradiction test recomputes the record's digests from the mutated
fields (work digest independently, refusal digest through `_reseal_refusal`,
support digests through `AlignedRegionSupport.build`) and bypasses the work
constructor with `object.__new__` where needed, so only the semantic invariants
can reject the record. An independent probe reproduced the pre-repair predicate
and confirmed it accepted a resealed planned refusal carrying `frame_clips=1`,
while the repaired constructor raises.

## Scope and boundary

Only the V1 contract of the approved bridge report was implemented: a versioned,
provenance-bound `AlignedRegionSupport` that reprojects the *original retained
region mask*, intersected with the caller's validated usable mask, onto unit
cells of a phase-aligned local block grid and types every cell `supported` or
`unknown`. Face placement (V2), volume emission (V3), CLI/UI, persistence schema
migration, litematic wiring, manual-coordinate inputs and promotion gates are
**not** touched. The new module is not imported by `reconstruction/__init__.py`
or by any product module, so it cannot change any existing CLI/pipeline/export
behaviour. The accepted `PlaneRegionEvidence`, its packed-mask codec and V2-A
`ReadingIdentity` are reused **read-only**.

## Contract implemented

`plane_region_support.AlignedRegionSupport` (schema
`m4-v1-aligned-region-support-1`) binds every one of the fields the task names
plus the repair cross-link:

| Requirement | Bound field |
| --- | --- |
| frame/image digest | `reading_identity.frame` (inside the identity record) + `image_sha256` |
| reading identity | full `v2a.ReadingIdentity` + `reading_identity_sha256` |
| content binding | `reading_binding_sha256` (identity × calibration × alternative × region-mask digest × image digest) |
| source-region ID | `source_region_id` (and `region_mask_sha256`) |
| retained alternative | `RetainedAlternative` (period, distance, support, digest) |
| usable-mask digest | `usable_mask_sha256` |
| calibration | `RegionSupportCalibration` (axis matrix, focal, principal point, patch centre, side, digest) |
| producer version | `producer_version` |
| phase-aligned local-grid origin | exact rational `origin_uv` (fractional allowed) |
| dimensions | `grid_shape` |
| row/column convention | `column_label`/`row_label` (the two tangent labels, `u` first) plus `column_convention="u"`, `row_convention="v"` |
| coverage policy | `coverage_policy="conservative_full_area_v1"` |
| exact work counters | `AlignedSupportWork` (seven exact integer counters) |
| canonical digest | `support_sha256` |

Each `AlignedSupportCell` carries its `state`, typed `evidence`
(`full`/`outside_frame`/`outside_region`/`outside_usable`/`degenerate`) and the
exact reduced-rational `cell_area`, `region_excluded`, `usable_excluded` and
`frame_excluded` diagnostics, plus its own canonical digest.

### Rasterization (no polygon, no interpolation, no air)

- Cell `(row, column)` covers block coordinates
  `[u0+column, u0+column+1] x [v0+row, v0+row+1]`; its four corners are mapped
  through an **exact rational homography** built from the calibration's dyadic
  inputs at the retained alternative's exact `distance_blocks`. The proposal
  polygon is never rasterized and window-relative pixel counts are never
  shifted. The accepted `float64` closed form is consulted only to fail closed on
  an overflowing projection; its rounded coefficients never reach the geometry.
- The projected quad is clipped (Sutherland–Hodgman against the image rectangle
  and against single unit pixel squares) and its exact intersection area with
  the image, the region mask and the usable mask is accumulated. All coordinates
  are exact `fractions.Fraction` values (every IEEE-754 input is dyadic), so
  "fully established inside" is an exact zero excluded area, not a tolerance.
- A cell is `supported` only when `cell_area > 0` and the region, usable and
  frame excluded areas are all exactly zero. Every other cell is `unknown`;
  absent mask pixels never become air and interior mask holes are never filled.
  A projected edge that is exactly outside the image/bounds yields a strictly
  positive exact exclusion and can never be `supported`.
- Each cell's projected quad is convex (all four corners require a positive
  homogeneous denominator), so exact convex clipping is valid.

### Determinism, alignment and budgets

- Projected area scales exactly with the alternative distance (`1/d^2`), and the
  origin is stored as an exact rational pair, so fractional (phase-aligned)
  grids are preserved bit-for-bit.
- An integer origin shift only relabels cells: the shifted state at
  `(row, column)` equals the base state at `(row, column+k)`, verified by test.
- `canonical_alternative_order` and `canonical_support_order` sort by content;
  reading alternatives are canonicalized before identity and content binding, so
  digests are independent of object identity and insertion order.
- `AlignedSupportBudget` has five explicit positive caps and no defaults.
  Planned (`max_cells`) and runtime excess both return an **atomic**
  `budget_exhausted` refusal that carries no cells; the observed counters are
  retained as evidence.

### Validation precedence (deterministic, first failure wins)

`invalid_input` (budget type, producer version, reading type, image/phase
digests) → `missing_provenance` (identity type, region evidence, reading
universe) → region mask re-decode (`invalid_region_mask`) → identity content
rebinding (`provenance_mismatch`) → `invalid_usable_mask` →
`image_shape_mismatch` → `calibration_mismatch` (calibration build) →
`alternative_not_retained` → `invalid_origin` → `invalid_grid` → projection
validation (`calibration_mismatch`: non-finite or degenerate homography) →
`budget_exhausted` (planned) → rasterize (runtime `budget_exhausted`).

The projection step is deliberately before every budget decision, so a
degenerate/overflowing homography is refused identically for a one-cell cap and
for a sufficient cap.

## Tests (81 cases; `tests/test_plane_region_support.py`)

The fixture is a hand-declared calibration whose projected unit cell is exactly
one image pixel, so cell states, excluded areas and counters are computable by
hand and are never read back from the module under test. Every valid
`ReadingIdentity` is now derived from real reading/universe content through the
accepted V2-A helpers; no dummy identity hash validates a real build.

The original 44 accepted cases are preserved. The first repair added 25 cases
and the final refusal-invariant repair adds 12:

- **Exact projection (finding 1):** a finite perturbed `rectification_homography`
  return must not change cells or digest; an exact negative projected edge is
  `unknown` with exact `(1, 2^20)` frame/region/usable exclusion; the exact
  homography agrees with the accepted closed form to `1e-9` relative and is
  non-degenerate.
- **Content-bound identity (finding 2):** ten independent mutation cases
  (alternative period, alternative distance, reading-order content, region
  evidence, rectification, frame, axis, source region, dummy order-key hash,
  dummy universe hash) must refuse `provenance_mismatch`; a sealed record rejects
  resealed calibration/alternative/region-mask/image/identity/binding mutations;
  reordered alternatives share one canonical identity.
- **Sealed contradictions (finding 3):** the normal axis can never be a tangent
  row/column label; freshly resealed budget refusals are rejected for a
  nonexistent cap, non-positive/non-exceeding limits, missing work evidence,
  embedded cells, `observed`/work-counter disagreement, and a non-`max_cells`
  stop with no examined cells; non-budget refusals reject work evidence.
- **Refusal invariants (final repair):** a freshly resealed planned
  `budget_exhausted` with any single nonzero counter (`frame_clips`,
  `region_pixel_tests`, `usable_pixel_tests`, `polygon_vertices`,
  `supported_cells`, `unknown_cells`) is rejected; a `max_cells` refusal with any
  examined work is rejected; runtime refusals with
  `supported_cells + unknown_cells > cells_examined` or
  `frame_clips > cells_examined` are rejected; `AlignedSupportWork` directly
  rejects both relations; a resealed support record with a forged work object
  (`frame_clips > cells_examined`) is rejected; and genuine planned and runtime
  refusals still reseal unchanged.
- **Precedence (finding 4):** non-finite and degenerate homographies return the
  same `calibration_mismatch` record for a one-cell cap and a sufficient cap; a
  valid homography still returns `budget_exhausted` for the small cap.

Plus the original coverage (rectangular/disconnected/L-shaped masks, holes,
usable exclusions, boundary unknowns, fractional alignment, integer shift
invariance, alternative-specific scale, reordered-input determinism, invalid
masks, planned/runtime atomic budget exhaustion, digest-mutation rejection,
canonical ordering, usable-mask digest and AST isolation).

## Exact commands and outcomes

Frozen interpreter `.venv\Scripts\python.exe`, project root, foreground:

1. `py_compile src\mc_photogrammetry\reconstruction\plane_region_support.py tests\test_plane_region_support.py`
   -> exit 0, no output.
2. `pytest tests\test_plane_region_support.py -q` -> **81 passed**, exit 0
   (collected count 81 = 44 prior + 25 first repair + 12 refusal-invariant
   repair).
3. `pytest tests\test_lattice_planes.py tests\test_plane_endpoint_provenance.py -q`
   -> **102 passed**, exit 0 (42 + 60; adjacent packed-mask/region-evidence and
   `ReadingIdentity` provenance suites).
4. `pytest tests\test_plane_phase_anchor.py tests\test_plane_graph.py tests\test_plane_relative_translation.py tests\test_plane_residue.py tests\test_plane_axis_support.py tests\test_plane_shift_domain.py tests\test_plane_vector_domain.py -q`
   -> **288 passed**, exit 0 (76 + 51 + 36 + 43 + 43 + 21 + 18; broader accepted
   V1/V2 provenance, shared-axis and adjacent mask/graph suites).

No warnings, errors, skips or non-zero exits.

Independent discrimination probe (not part of the suite) fed the pre-repair
path `Fraction(float(perturbed rectification_homography))` to `_rasterize` and
confirmed it classifies `['supported', 'unknown', 'supported', 'unknown']`
where the repaired exact path classifies `['supported'] * 4`, so
`test_projection_uses_exact_rational_coefficients_not_rounded_float` fails on
the pre-repair behaviour.

A second independent probe reproduced the pre-repair refusal predicate and
confirmed it accepted a freshly resealed planned `budget_exhausted` whose work
carried `frame_clips=1` with every digest recomputed; the repaired constructor
raises `ValueError` for the same record, while a genuine planned and a genuine
runtime refusal still reseal with identical detail and work.

## Diff scope and hashes

| Path | Lines | SHA-256 |
| --- | --- | --- |
| `src/mc_photogrammetry/reconstruction/plane_region_support.py` | 1394 | `ff4b53dd81f15843f168bfc5e2e5ed425d02cb8785003192cf559d719e7fcef1` |
| `tests/test_plane_region_support.py` | 1212 | `c82a7adc9005445afc33734a4b0f59513d02b43d7beae86b1ae18f0ea664c765` |

`git status --porcelain -- <the three scope paths>` shows all three as untracked
(`??`) new artifacts of this task. `git diff --name-only` lists only pre-existing
unrelated working-tree edits that this session did not touch.

Adjacent accepted modules recomputed unchanged (all match the accepted
M4-V2B/repair record, so no accepted source was touched):

| Path | SHA-256 |
| --- | --- |
| `src/.../plane_endpoint_provenance.py` | `3cb6aacea4d9e6873d0b10edd4ea7539b1646a1e75475cef2006f81caa3316e7` |
| `src/.../plane_phase_anchor.py` | `36d8d24541a856d97476ae440ee37fb43173672cfff18340538ca92b56bdeefc` |
| `src/.../plane_graph.py` | `c5245680b469f93ded5525bb80b0ddf065b46a64e1d88d66cc1fa93115522e92` |
| `src/.../plane_relative_translation.py` | `f703a28d275d55e4a85fad7fde0bfd612928f8d54a5489c0ea097fb42efbcd3e` |
| `src/.../plane_residue.py` | `55ec0cad8fe1088b7b33b1f57c4d58d3d3b912f79765a1f391553ac7b845be0d` |
| `src/.../plane_axis_support.py` | `18598afb2096b504464411e3a2cbf6a2e960cb2601c4dd2fb8f77d4b46c91f3f` |
| `src/.../plane_vector_domain.py` | `72ee29fb936cffd7f8a154c18d6f6bab18876d57ad66f45bc316e39f7d45e7f5` |

Reused upstream region-mask owner, also unchanged by this task: `src/.../lattice_planes.py`
`82ada0ccc0f92b580342aa35f8333d1013910f7cfa91eada0ed7d5e7c95831e7`,
`src/.../plane_shift_domain.py`
`512c6fb1c9d3491e3deb4ea4db11353e7f8167dcf8aaaabe8fd17ea17303d91b`.

The report's own hash is recomputed after its final edit and reported in the run
result.

## Deviations and design decisions

1. **`build_aligned_region_support` now requires `reading_universe`.** A V2-A
   `ReadingIdentity` deliberately does not cover the reading's
   rectification/calibration, so an old identity cannot be bound to a changed
   calibration without the universe the identity was derived from. The builder
   recomputes the canonical identity from `(reading, universe)` and additionally
   requires the universe entry at the identity's canonical index to content-equal
   the reading. Callers must present alternatives in the V1 canonical order in
   both the reading and the universe; the builder canonicalizes defensively.
   Recorded as an integration gate below.
2. **`AlignedSupportRefusal` now rejects non-budget work evidence.** The builder
   only ever attaches work to `budget_exhausted`, so this is a pure
   contradiction guard with no behavioural loss. The runtime/planned work
   cross-link is conditional on `cells_examined`: the planned stop legitimately
   reports the planned cell count while no cell has been examined.
3. **Disconnected/L-shaped/holed masks are not refused; their unsupported cells
   are `unknown`.** A structurally valid retained mask with disconnected
   components, an L shape or a hole is exactly the case the V1 support map must
   describe conservatively, so it produces `unknown` cells (with exact
   excluded-area evidence) rather than a typed refusal. Only an actually
   malformed mask (`invalid_region_mask`, reachable only by a decoder fault since
   `PlaneReading` pre-validates its evidence) refuses.
4. **`ReadingIdentity` is reused from the accepted V2-A module
   `plane_endpoint_provenance`.** This is the identity record named by the
   approved bridge report. It makes V1 depend on an already-accepted, frozen
   module; it is acyclic (no accepted module imports `plane_region_support`).
   Recorded as an integration gate below.
5. **Exact rational arithmetic is mandatory.** A cell is supported only on an
   exact zero excluded area. This is slower than float rasterization but is the
   only way to satisfy "conservatively established" without a tolerance that
   could fabricate support at a boundary. Work is capped by the explicit budget,
   so a pathological projection fails closed instead of running unbounded.
6. **The projected cell area for the `degenerate` case is stored as `(0,1)`**
   (zero) with all excluded areas zero; the cell is `unknown` because
   `cell_area == 0` is never "established inside" anything. A degenerate
   *homography* (zero determinant) is refused earlier as `calibration_mismatch`.
7. **The accepted `float64` homography is still called for validation only.**
   This keeps an overflowing projection observable as a typed refusal without
   letting any rounded coefficient influence a cell boundary.
8. **A planned `budget_exhausted` must report all-zero work.** `max_cells` is the
   only cap decided before any cell is examined, so a refusal that names it (or
   that reports `cells_examined == 0`) is an ex-ante stop and must report zero
   for all seven counters. Runtime stops are unchanged: they report a positive
   `cells_examined` and an `observed` equal to their cap's work counter. The
   `supported_cells + unknown_cells <= cells_examined` and
   `frame_clips <= cells_examined` relations are enforced on
   `AlignedSupportWork` itself and re-checked by the refusal and support
   constructors, so a deliberately forged work object cannot be smuggled through
   a freshly resealed record. This is a pure contradiction guard: the builder
   already produces only all-zero planned work and consistent runtime work.

## Unresolved concerns / integration gates (not blockers)

- **Canonical-universe identity contract.** V1 now requires the caller to pass
  the canonical reading universe that produced the identity, with alternatives
  in canonical content order. A future V2 integration must derive its V2-A
  identities from that same canonical form. This is a deliberate contract
  tightening required by Astra finding 2.
- **V1 -> V2-A identity dependency.** `ReadingIdentity` currently lives in
  `plane_endpoint_provenance` (V2-A), and V1 also reuses the private
  `_ordered_readings`/`_canonical_index_lookup`/`_reading_identity`/
  `_canonical_universe_sha256`/`_reading_order_key_sha256` helpers. If a future
  layering rule requires V1 to precede V2-A, those would need to be extracted to
  a shared identity module; that is an architectural change outside this scope.
- **V2-A's order key does not encode rectification, so the strongest available
  rectification binding is two-layered.** The builder refuses a changed
  rectification when the identity's original universe is presented (the
  realistic provenance path), and the sealed record's `reading_binding_sha256`
  makes any partial reseal with a changed calibration/alternative/region
  digest fail closed. A caller that both mutates the reading *and* rebuilds the
  V2-A universe around the mutation produces an identity V2-A itself cannot
  distinguish; making that distinguishable would require extending
  `ReadingIdentity`, which is accepted frozen V2-A source outside this scope.
- **`PlaneRegionEvidence` re-decode uses the accepted private
  `lattice_planes._decode_region_evidence`.** This is deliberate reuse of the
  validated codec (re-deriving the digest independently would risk divergence).
- **No public caller is wired yet.** V2 placement (the proposed consumer of this
  support map) is not implemented, so the contract is exercised only by tests.
  The support grid origin is supplied by the caller; binding it to a replayed
  V2-B phase anchor is a V2/V4 gate, represented here only by the optional
  `phase_anchor_sha256` provenance field.
- **No replay validator for a completed support record.** Tamper resistance comes
  from the canonical digest recheck and the content cross-link in the record
  constructor; a full source-bound replay function is not part of the declared
  V1 scope.

## Blocker

None. All four Astra findings and the conditional-review refusal-invariant
defect are repaired inside the routed three-file scope; every required check
passes; no accepted hash changed, no scope violation and no architectural
inconsistency was found. Placement, volume emission, CLI/UI, persistence,
litematic wiring and production promotion remain blocked and untouched. **Stop
for Astra read-only review.**

## Files written

- `src/mc_photogrammetry/reconstruction/plane_region_support.py` (exact-rational
  projection, canonical-universe identity rebinding, record/refusal semantic
  invariants, planned-refusal zero-work and work-relation invariants,
  validation-before-budget precedence, versioned `AlignedRegionSupport`,
  canonical digests, atomic budgets, typed refusals).
- `tests/test_plane_region_support.py` (81 focused cases: 44 prior + 25 first
  repair + 12 refusal-invariant repair).
- `docs/reports/M4-V1-aligned-region-support-implementation-2026-09-19.md`
  (this file).
