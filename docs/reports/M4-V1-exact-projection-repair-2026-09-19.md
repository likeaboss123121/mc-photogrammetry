# M4 V1 exact projection repair — 2026-09-19

Task: `m4-v1-exact-projection-repair-2026-09-19`
Dispatched by: `AGENT_COMMS.md` (2026-09-19T18:00:00-05:00, Luna Manager)
Worker: Sol-equivalent CLI worker (DeepSeek), bounded implementation
Upstream module: `src/mc_photogrammetry/reconstruction/plane_region_support.py`
Status: **Repair implemented in the routed three-file scope; `py_compile`, 87
focused region-support tests, 102 adjacent provenance/mask tests, and 376
broader plane tests pass; stopped for Astra read-only review.**

Write scope used (exactly the three routed paths):

- `src/mc_photogrammetry/reconstruction/plane_region_support.py`
- `tests/test_plane_region_support.py`
- `docs/reports/M4-V1-exact-projection-repair-2026-09-19.md` (this file)

No placement, volume, wiring, export, litematic, CLI/UI, persistence, scoring,
schema, launcher, security, manual-metadata or promotion-gate file was created
or edited. No git stage/commit/reset/clean. `py_compile`/`pytest` refreshed
standard `__pycache__` bytecode caches only. All written files are UTF-8.

## Defect and repair

The retained calibration's projection is the accepted closed form
`projection @ basis` with
`basis = column_stack(first, second, distance*side*normal)` and
`normal, first, second = plane_basis(axis, axis_matrix)`.  Every declared input
is an exact dyadic rational.  A composition that forms any coefficient as a
rounded `float64` product and converts it afterwards carries that rounding into
a cell boundary, where it changes a boundary cell's classification.

The repair makes the whole coefficient composition exact, from the retained
calibration/alternative *before* any arithmetic:

| Mechanism | Change |
| --- | --- |
| Exact plane basis | New `_exact_plane_basis(axis, calibration)` reproduces `lattice.plane_basis`'s tangent convention exactly (normal = row `axis`; first/second = remaining rows in ascending label order) but converts each declared `float64` entry once with `Fraction` and returns exact rational vectors.  No float64 axis-matrix round trip remains in the geometry path. |
| Exact coefficient composition | `_exact_homography` builds `projection` and `basis` from the retained calibration and alternative as rationals and composes them with `Fraction` multiplication/addition only.  The retained scale is the exact rational product `side * distance_blocks`.  `lattice_planes.rectification_homography` is still called, but **only** for its finiteness validation; its rounded coefficients never reach a cell boundary. |
| Fail-closed exactness guard | `build_aligned_region_support` refuses `calibration_mismatch` (`"homography is not exact rational"`) if any composed coefficient is not a `Fraction`, so a future rounded coefficient cannot silently reach a cell boundary.  This is a pure contradiction guard: the builder's own composition is exact by construction and the branch is unreachable in this module. |

Everything else is preserved unchanged: the conservative full-area policy
(supported only on an exact zero excluded area, never inferred air), the closed
`supported`/`unknown` cell states with exact `cell_area`/`region_excluded`/
`usable_excluded`/`frame_excluded` diagnostics, the exact rational origin and
grid, the `u`/`v` tangent convention and label ordering, the content-bound
`ReadingIdentity` rebinding and `reading_binding_sha256` cross-link, the atomic
planned/runtime budget refusals and their semantic invariants, and the
deterministic validation precedence (projection validation strictly before any
budget decision).

### Behavioural equivalence of the repaired composition

The new helper is a pure refactor of the previous arithmetic.  Comparing the
repaired `_exact_homography` against the original `lattice.plane_basis`-based
composition over 4 axis matrices x 3 axes x 3 focals x 3 principal points x 3
sides x 3 distances = **972 calibrations** produced **0 mismatching rational
coefficients**.  The source repair therefore cannot change an accepted placement,
provenance or budget result; it removes the float64 round trip from the
definition and guards against its reintroduction.

## Astra counterexample and controls

### Counterexample: distance 0.3, principal 0.3, grid origin -0.09

`test_astra_counterexample_exact_boundary_exposes_the_float_defect` (retained
from the concurrent regression already on the stabilized test revision) pins the
decisive trap.  With `focal=1`, `side=1`, retained `distance_blocks=0.3`,
principal point `(0.3, 4.0)` and grid origin `-Fraction(0.09)`:

- the rounded float64 composition places the single cell's left edge at exactly
  `0`, so it classifies an all-true cell as `full` with every exclusion zero —
  the non-conservative defect Astra used to reject the rounded path;
- the exact rational composition places the same edge at
  `-Fraction(1, 90071992547409920)`, outside the frame;
- the module under test reports `unknown`/`outside_frame` with a strictly
  positive exact frame exclusion and `supported_count == 0`.

`test_counterexample_boundary_cell_stays_conservative_and_exact` adds the
single-retained-pixel mask variant at the same fractional parameters and pins
the consumed translation coefficient to the exact rational product
`principal * (distance * side) * normal`, which differs from the rounded binary64
product even though both print as `0.09`.

### Oblique and non-trivial calibration controls

- `test_oblique_counterexample_control_remains_conservative`: a genuinely
  oblique axis matrix (`(0.9, 0, 0.43589) / (0, 1, 0) / (-0.43589, 0, 0.9)`) at
  the same fractional distance/principal/origin stays exact (coefficient-wise
  equal to an independently recomposed rational homography) and every cell stays
  `unknown`.
- `test_nontrivial_oblique_calibration_matches_closed_form_and_stays_exact`: a
  non-axis-aligned axis matrix with a fractional principal point
  `(320.2, 240.7)`, `focal 720.5`, `side 0.7` and `distance 3.3` composes exactly,
  matches the accepted closed form's binary64 coefficients to the existing `1e-9`
  relative bound, and has a non-zero exact determinant.
- `test_exact_homography_composes_retained_coefficients_in_rationals`: the
  consumed matrix equals an independently written rational composition and every
  coefficient is a `Fraction`.
- `test_exact_homography_is_not_the_rounded_float_composition`: the rounded
  binary64 composition is provably a different rational on the counterexample
  while printing the same values, so exactness is observable, not cosmetic.

### Preserved provenance, precedence and convention

- `test_exact_projection_keeps_invalid_input_precedence_over_budget`: with the
  oblique counterexample and a one-cell budget the invalid-path accounting is
  unchanged: a valid projection still returns the atomic planned
  `budget_exhausted` (`max_cells`, `cells_examined == 0`) rather than a
  projection refusal.
- All prior provenance/identity, refusal-invariant, no-air/no-polygon, budget
  atomicity, digest-mutation and module-isolation tests are retained unchanged,
  including
  `test_projection_uses_exact_rational_coefficients_not_rounded_float`,
  `test_exact_negative_projected_edge_is_never_supported`,
  `test_exact_homography_matches_the_accepted_closed_form`,
  `test_non_finite_homography_precedes_budget` and
  `test_degenerate_homography_precedes_budget`.
- `test_tangent_axis_labels_follow_the_shared_sorted_convention` continues to
  pin the `u`/`v` label ordering for axes 0/1/2.

## Tests (87 cases; `tests/test_plane_region_support.py`)

The 81 cases accepted with the V1 module are preserved; the exact-projection
regression adds the counterexample, the oblique and non-trivial controls, the
independent rational-composition identity, the rounded-vs-exact discrimination
and the precedence control.

## Exact commands and outcomes

Frozen interpreter `.venv\Scripts\python.exe`, project root, foreground:

1. `py_compile src\mc_photogrammetry\reconstruction\plane_region_support.py
   tests\test_plane_region_support.py` -> exit 0, no output.
2. `pytest tests\test_plane_region_support.py -q` -> **87 passed**, exit 0.
3. `pytest tests\test_lattice_planes.py tests\test_plane_endpoint_provenance.py
   -q` -> **102 passed**, exit 0 (adjacent packed-mask/region-evidence and V2-A
   `ReadingIdentity` provenance suites).
4. `pytest tests\test_plane_region_support.py tests\test_lattice_planes.py
   tests\test_plane_endpoint_provenance.py tests\test_plane_phase_anchor.py
   tests\test_plane_graph.py tests\test_plane_relative_translation.py
   tests\test_plane_residue.py tests\test_plane_axis_support.py
   tests\test_plane_shift_domain.py tests\test_plane_vector_domain.py -q`
   -> **478 passed**, exit 0 (87 focused plus 391 adjacent/broader accepted
   V1/V2 provenance, shared-axis, graph and mask suites).

No warnings, errors, skips or non-zero exits in the runs above.

### Pre-existing V2 placement failures (not caused by this repair)

`pytest tests\test_plane_placement.py` currently reports `3 failed, 407 passed,
1 error` (`test_recipe_c2rb_is_unique_exhaustive_single_signed_class`,
`test_recipe_v2a_is_endpoint_unique`,
`test_recipe_v2b_determines_every_axis_with_zero_anchor`, and the setup error
`runtime_count_mismatch` for `test_recipe_v1_supports_replay_with_residue_origins`).
These are V2 placement/V2-B fixture issues in an unaccepted branch already noted
as an open blocker in `AGENT_COMMS.md` (`m4-v2-fixture-reachability` stopped at
the V2-B replay boundary).  They are not caused by this repair: the repaired
composition is bit-for-bit identical to the previous exact composition across the
972-calibration cross-check above, so no V1 input to `plane_placement` changed.
The unrouted `tests\test_plane_placement.py` is outside this task's write scope
and was not modified.

## Diff scope and hashes

| Path | Lines | SHA-256 |
| --- | --- | --- |
| `src/mc_photogrammetry/reconstruction/plane_region_support.py` | 1428 | `45a5b02947e3f95ee5d160334b5172e3a547b16c5893b9c757f964839d8896e3` |
| `tests/test_plane_region_support.py` | 1541 | `544a6337aa37bcfd3923038aa7b4d4e932c286e0b500d576e1d71297385a1dad` |

Both paths are untracked (`??`) new artifacts of the M4 V1 task family; no
accepted hash was changed.  The report hash is recorded in the worker run
result (a self-referential digest cannot be embedded here).
Adjacent accepted modules were reused read-only:
`lattice_planes.py`
`82ada0ccc0f92b580342aa35f8333d1013910f7cfa91eada0ed7d5e7c95831e7`,
`plane_endpoint_provenance.py`
`3cb6aacea4d9e6873d0b10edd4ea7539b1646a1e75475cef2006f81caa3316e7`.

## Deviations and design decisions

1. **The exact composition is now structurally independent of the float64
   helper.**  Previously exactness depended on `lattice.plane_basis` being a pure
   row selector; the rational basis is now derived inside this module with the
   same convention, so a change to that helper's arithmetic cannot silently
   reintroduce rounding here.  `lattice_planes` remains an allowed import for
   `PlaneReading`/`PlaneRegionEvidence`/`DistanceAlternative` types and the
   retained float64 finiteness validation.
2. **Acceptance evidence includes both a coefficient identity and the
   classification trap.**  Whether the rounded composition flips a particular
   boundary cell depends on the last-bit rounding direction of the specific
   input; the counterexample control pins the deterministic facts (rounded left
   edge exactly `0` and `full`; exact left edge strictly outside; module
   conservative and exact), while the independent-recomposition tests pin the
   exact coefficients across oblique calibrations.
3. **No public contract change.**  `_exact_plane_basis` is private and no record
   field, refusal reason, digest payload or public signature changed, so no
   caller, digest or downstream consumer is affected.

## Concurrency note

The focused test file was concurrently rewritten by another writer while this
task ran; the counterexample regression and its `_exact_homography_reference` /
`_float_composed_homography` / `_classify_with_homography` helpers were already
present on the stabilized revision when this repair added its source change and
controls on top.  This task did not revert or weaken that work; the final test
hash above covers both contributions.  Placement, volume, wiring, export and
manual metadata remain untouched.

## Unresolved concerns / integration gates

- **Astra should re-run the original reproducer against the new source hash.**
  The counterexample control confirms the module already follows the exact
  composition while the rounded composition is non-conservative at that
  boundary; this repair makes that exactness structural and fail-closed rather
  than changing an accepted numeric result.
- **V2 placement fixture reachability remains blocked.**  The pre-existing
  `test_plane_placement.py` failures above belong to that open blocker and are
  unaffected by this repair.
- **No public caller is wired.**  `plane_region_support` is still exercised only
  by tests plus the unaccepted `plane_placement` branch; placement, volume,
  production wiring, persistence and litematic promotion remain blocked.

## Blocker

None inside this task's scope.  The bounded exact-projection repair is
implemented and all required checks pass; no accepted hash changed, no scope
violation and no architectural inconsistency was found.  **Stop for Astra
read-only review.**

## Files written

- `src/mc_photogrammetry/reconstruction/plane_region_support.py`
  (`_exact_plane_basis`, fully rational `_exact_homography`, fail-closed
  exactness guard; all other behaviour preserved).
- `tests/test_plane_region_support.py` (exact-projection counterexample, oblique
  and non-trivial controls, rational-composition identity, precedence control,
  plus the preserved 81 accepted cases).
- `docs/reports/M4-V1-exact-projection-repair-2026-09-19.md` (this file).
