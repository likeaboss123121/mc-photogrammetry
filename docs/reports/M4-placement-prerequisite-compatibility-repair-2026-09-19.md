# M4 placement prerequisite compatibility repair (2026-09-19)

Implementation task: `m4-placement-prerequisite-compatibility-implementation-2026-09-19`
Route: `route-m4-placement-prerequisite-compatibility-implementation-2026-09-19-73ed9d5e96e4`
Upstream design (accepted): `docs/reports/M4-placement-volume-bridge-contract-2026-09-19.md`
Prerequisite findings: `docs/reports/DEEPSEEK-M4-V2-placement-prerequisite-audit-2026-09-13.md`
Worker: Sol-equivalent CLI worker (DeepSeek), bounded implementation
Status: **both approved compatibility repairs implemented; focused, all-axis,
permutation/replay and adjacent C2-R-B/P1-B checks pass; stopped for
Manager/Astra review.**

Write scope used (exactly three routed paths):

- `src/mc_photogrammetry/reconstruction/plane_phase_anchor.py`
- `tests/test_plane_phase_anchor.py`
- `docs/reports/M4-placement-prerequisite-compatibility-repair-2026-09-19.md`
  (this file)

No other source, test, report, contract, calibration/evaluation artifact,
handoff, comms, CLI/UI, persistence, `.litematic`, `BlockVolume`, placement,
region-support, volume or wiring file was created or edited. No git
stage/commit/reset/clean. `py_compile`/`pytest` refreshed standard
`__pycache__` bytecode caches only.

## Scope and boundary

Only the two repairs named by the task were implemented. Placement, region
support rasterization, volume emission, CLI/UI, persistence, litematic wiring
and manual metadata inputs were **not** touched. `plane_phase_anchor` is not
imported by any product module (only `tests/test_plane_phase_anchor.py`
imports it), so the repair cannot and does not change CLI, pipeline,
persistence or export behaviour. The accepted V2-B accounting/replay API and
the advanced-only metadata boundary are preserved exactly.

## Repair 1 — full signed dyadic translation parsing

### Defect

C2-R-B's certificate contract (`plane_relative_translation._validate_dyadic`)
admits every **reduced signed dyadic** with a positive power-of-two
denominator: integers such as `3` and negative fractions such as `-1/4` are
valid. V2-B parsed the certificate with `plane_phase_anchor._fraction`, the
strict **modulo-one canonical** parser, which requires `0 <= numerator <
denominator` and an odd numerator. Two code paths were affected:
`_translations_by_frame` (used by `_derive_hypotheses`) and the inline copy in
`_determine`.

Observed on the unmodified source (read-only probe, public five-frame ring
fixture certificate replaced by the accepted C2-R-B sealer with signed values):

```text
File "plane_phase_anchor.py", line 1838, in _translations_by_frame
  translations[frame] = tuple(_fraction(value) for value in xyz)
File "plane_phase_anchor.py", line 204, in _fraction
  raise ValueError("noncanonical_dyadic")
ValueError: noncanonical_dyadic
```

A valid `(3, 1)` translation (and any negative fraction) therefore made
`_derive_hypotheses` raise, which `analyze_phase_anchor` maps to
`refused/v2b_input_invalid` — an upstream-valid acceptance was impossible.

### Change

- New private parser `_signed_dyadic(value)` (placed beside `_frac`) accepts
  **exactly** the C2-R-B certificate domain: a 2-tuple of exact `int`s with a
  positive power-of-two denominator and `gcd(abs(numerator), denominator) == 1`.
  Everything else still raises `ValueError("noncanonical_dyadic")`. Periodicity
  is applied later, at the exact-rational anchor (`_mod1`), never at parse time,
  so valid translations are **not** reduced modulo one during parsing.
- `_translations_by_frame` now parses each certificate entry with
  `_signed_dyadic` and returns the full exact `Fraction` (integers and negative
  fractions preserved).
- `_determine`'s duplicated inline parse was deleted; it now calls the single
  `_translations_by_frame`. The two parse sites can no longer diverge.

### Preserved semantics

- Anchor arithmetic is unchanged: `_mod1(world_sign * value +
  translation[axis])` consumes the full exact fraction and is mathematically
  identical to consuming its modulo-one residue.
- The retained contributor field `frame_translation_dyadic` stays the accepted
  canonical modulo-one dyadic (`_reduce_dyadic`), so every previously accepted
  input keeps its exact record bytes and digests, and contributors are still
  validated by `_validate_dyadic`. Values already in `[0, 1)` have identical
  full-signed and modulo-one encodings, so no digest changes.
- Canonical ordering, hypothesis merging, invalid-input precedence and
  source-bound replay are untouched. A malformed certificate still raises
  `ValueError` inside the same guarded call and still refuses as
  `refused/v2b_input_invalid` before any budget cap.

## Repair 2 — P1-B actual tangent-label mapping

### Defect

The P1-B contributor branch of `_derive_hypotheses` hard-coded the local
`u`/`v` names to the world labels `0`/`1`:

```python
local_axis = record.local_axis
local_label = 0 if local_axis == "u" else 1
```

`PeriodicResidueCandidates` carries only the linear `u`/`v` name, and the
shared convention (used by `plane_vector_domain`, `plane_axis_support`,
`plane_graph.cell_grid_homography` and the V2-A domains) is that the two
tangent labels are **the world labels other than the reading's normal, sorted
ascending, `u` first, `v` second**. The hard-coded index is correct only for a
reading whose normal is world axis `2`.

Observed on the unmodified source across the three reading normals (read-only
probe building the public fixture with each normal):

| Normal | Before fix |
| --- | --- |
| 0 | spurious axis-0 hypothesis `[('u', 0)]`; axis 1 contaminated to `[('u', 1), ('v', 1)]`; axis 2 only `[('v', 2)]`; axes 0/1/2 = `no_seam_evidence`/`determined`/`determined` |
| 1 | axis 0 `[('u', 0)]`; axis 1 spurious `[('v', 1)]`; axis 2 `[('v', 2)]`; axis 1 abstains `no_seam_evidence` |
| 2 | masks the defect: `[('u', 0)]` and `[('v', 1)]` are the real labels |

Because branch 2 built hypotheses on the wrong world axes while the
observation descriptors (which already used `residue.local_axis_label`) built
them on the right axes, the axes could not agree and the anchor abstained or
retained contaminated hypotheses.

### Change

- New private helper `_reading_tangent_label(reading_axis, local_axis)` returns
  the shared sorted tangent label (`range(3)` minus the normal, `u` first).
  Invalid `reading_axis`/`local_axis` values raise `axis_transform_invalid`.
  The helper selects only the **label**; `plane_graph.axis_transform(label,
  quadrant)` remains the single authority for the world axis and sign, exactly
  as in every other contributor branch.
- The P1-B branch now derives the label from the reading/domain:
  `residue = domain.residue_u if local_axis == "u" else domain.residue_v`,
  then requires
  `residue.local_axis == local_axis`,
  `residue.local_axis_label == _reading_tangent_label(record.reading_axis,
  local_axis)` and
  `residue.source_record_sha256 == source_sha`.
  A mismatch records the existing typed `axis_transform_invalid` failure. For
  accepted, replayed V2-A data these three invariants always hold, so no
  accepted input changes.
- Branch 3 (edge references) and branch 1 (reading domains) already used
  `residue.local_axis_label` and were not changed. `_observation_inputs` already
  used `residue.local_axis_label`; it is now consistent with hypothesis
  derivation.

## Tests added (30 new cases; suite 46 -> 76)

All in `tests/test_plane_phase_anchor.py`, all declared from the fixture, never
read back from the module under test.

Full signed dyadic translation domain:

- `test_translation_parser_accepts_full_signed_dyadic_domain` — a certificate
  built through the **accepted C2-R-B sealer** with `(3,1)`, `(-1,4)`, `(5,8)`,
  `(-3,1)`, `(7,2)`, `(-1,2)`, `(1,1)`, `(-5,4)`, `(9,8)` on the three world
  axes parses to the exact `Fraction`s (`3` stays `3`; `-1/4` stays negative).
- `test_translation_parser_admits_exactly_the_c2rb_validated_domain` —
  exhaustive cross-check over numerators `-8..8` and denominators
  `1,2,4,8,16`: accepted iff `plane_relative_translation._validate_dyadic`
  accepts, and then equal to `relative._fraction`.
- `test_translation_parser_rejects_noncanonical_values` (14 cases) —
  non-dyadic `(1,3)`, non-reduced `(2,4)`/`(0,2)`, zero/negative denominators,
  wrong arity, floats, bools, strings and `None` all keep
  `ValueError("noncanonical_dyadic")` and are simultaneously rejected by
  C2-R-B's own validator.
- `test_signed_translation_certificate_replays_identically_to_mod_one` — the
  public five-frame ring certificate is rebuilt twice through the accepted
  sealer: once with signed integers/negative fractions on all three axes and
  once with their exact modulo-one residues. Hypothesis derivation is
  byte-identical for both, differs from the all-zero baseline (the declared
  residues are non-zero, so the equivalence is not degenerate), every
  contributor `frame_translation_dyadic` is still canonical modulo one, and
  reversing the P1-B evidence tuple leaves the hypotheses byte-identical.

P1-B tangent labels across all normal axes:

- `test_tangent_labels_follow_the_shared_sorted_convention` — for all three
  normals, the helper returns the shared sorted tangent labels, and for all
  four quadrants `plane_graph.axis_transform` maps them bijectively onto the
  two non-normal world axes with `+/-1` signs.
- `test_tangent_label_helper_rejects_invalid_inputs` (6 cases) — out-of-range,
  negative, bool, float and unknown-name inputs raise.
- `test_all_normal_axes_derive_actual_p1b_tangent_labels` (3 cases) — the
  public fixture built with reading normal 0, 1 and 2. For each normal: the
  normal axis abstains `no_hypothesis`, hypotheses exist exactly on the two
  tangent axes, each hypothesis has 20 contributors
  (`reading_domain` 5, `phase_candidates` 10, `edge_reference` 5), and every
  contributor's `local_axis_label` equals the real tangent label and replays
  through `plane_graph.axis_transform` to its hypothesis axis/sign. Both
  tangent axes are `determined` with exact `alpha = (0, 1)`.
- `test_all_normal_axes_permutation_and_replay` (3 cases) — for each normal,
  reversing matches, phases, readings, signatures and split sources produces
  byte-identical result JSON, and `validate_phase_anchor_replay` accepts the
  baseline record.

Post-fix all-normal result (declared expectation now asserted by the test):

| Normal | Axis 0 | Axis 1 | Axis 2 | Hypotheses |
| --- | --- | --- | --- | --- |
| 0 | `abstained/no_hypothesis` | `determined` | `determined` | axis 1 `[('u', 1)]`, axis 2 `[('v', 2)]` |
| 1 | `determined` | `abstained/no_hypothesis` | `determined` | axis 0 `[('u', 0)]`, axis 2 `[('v', 2)]` |
| 2 | `determined` | `determined` | `abstained/no_hypothesis` | axis 0 `[('u', 0)]`, axis 1 `[('v', 1)]` |

Normal 2 is byte-identical to the previously accepted fixture (the
hard-coded labels happened to be its real labels).

## Exact commands and outcomes

Frozen interpreter `.venv\Scripts\python.exe`, project root, foreground:

1. `py_compile src\mc_photogrammetry\reconstruction\plane_phase_anchor.py tests\test_plane_phase_anchor.py`
   -> exit 0, no output.
2. `pytest tests\test_plane_phase_anchor.py` -> **76 passed in 12.93s**, exit 0
   (46 pre-existing + 30 new; includes the AST/isolation boundary, exact
   accounting, resealed-mutation and source-bound replay suites).
3. `pytest ... -k "signed or tangent or all_normal"` -> **15 passed, 61
   deselected in 2.29s**, exit 0 (repair-specific selection).
4. `pytest tests\test_plane_relative_translation.py tests\test_plane_residue.py tests\test_plane_graph.py`
   -> **130 passed in 35.51s**, exit 0 (C2-R-B / P1-B / shared-axis-transform
   adjacent suites, collected counts 36 + 43 + 51).
5. `pytest tests\test_plane_endpoint_provenance.py tests\test_plane_axis_support.py tests\test_plane_shift_domain.py tests\test_plane_vector_domain.py tests\test_c2r_vector_domain_proof.py`
   -> **143 passed in 12.39s**, exit 0 (broader upstream P1/C1/A4 evidence:
   60 + 43 + 21 + 18 + 1).

No warnings, errors, skips or non-zero exits.

## Diff scope and hashes

Changed by this task:

| Path | Lines | SHA-256 |
| --- | --- | --- |
| `src/mc_photogrammetry/reconstruction/plane_phase_anchor.py` | 3318 | `36d8d24541a856d97476ae440ee37fb43173672cfff18340538ca92b56bdeefc` |
| `tests/test_plane_phase_anchor.py` | 1715 | `0ad161b0f92bc39f3175d25dea78b1905e5205baf9fc9e076a509549a0afbd59` |

Pre-repair accepted baseline (from
`docs/reports/M4-V2B-runtime-accounting-implementation-2026-09-19.md`): source
`17b169b7ebce339a320b4edf3b34266eccc9422f24386c337e2c4ec3dde0e1fe` (3255
lines), tests
`664c8a17d4bd5b8edfbda15463221bc63950f0785df35da448db5e6096bf7903` (1487
lines). The repair adds 63 source lines and rewrites the two defect sites only.

Unchanged upstream accepted hashes (recomputed; all still match the accepted
record, so no upstream module was touched):

| Path | SHA-256 |
| --- | --- |
| `src/.../plane_relative_translation.py` | `f703a28d275d55e4a85fad7fde0bfd612928f8d54a5489c0ea097fb42efbcd3e` |
| `src/.../plane_residue.py` | `55ec0cad8fe1088b7b33b1f57c4d58d3d3b912f79765a1f391553ac7b845be0d` |
| `src/.../plane_endpoint_provenance.py` | `3cb6aacea4d9e6873d0b10edd4ea7539b1646a1e75475cef2006f81caa3316e7` |
| `src/.../plane_vector_domain.py` | `72ee29fb936cffd7f8a154c18d6f6bab18876d57ad66f45bc316e39f7d45e7f5` |
| `src/.../plane_axis_support.py` | `18598afb2096b504464411e3a2cbf6a2e960cb2601c4dd2fb8f77d4b46c91f3f` |
| `src/.../plane_graph.py` | `c5245680b469f93ded5525bb80b0ddf065b46a64e1d88d66cc1fa93115522e92` |

`plane_graph.py` was in the routed write scope but was **not** modified: the
parser owner is `plane_phase_anchor._translations_by_frame`, and the shared
`plane_graph.axis_transform` is reused unchanged, so the narrow owner test file
is `tests/test_plane_phase_anchor.py`. The report's own hash is recomputed after
its final edit and reported in the run result.

`git status --porcelain` (scoped): the two routed code/test paths and this
report are untracked new artifacts, consistent with their pre-existing
untracked state; no tracked file attributable to this task was modified.

Accepted-record preservation (recomputed on the public normal-2 fixture with
the default `_anchor_budget`): `result_sha256`
`0b85a2c90ff3036ecf8e4e933ed7b9548a19c05e1571b64cceb0e7b59c89ac79` and
`planned_work.work_sha256`
`26996d69bb8c24bc8946d599ac96801753ea8b90948e753ae11da2a2be0c6e69` are
**byte-identical to the previously accepted V2-B records**. The repairs add no
field, reorder no contributor and alter no digest for previously accepted
input; the signed-translation path is exercised only by the new fixture.

## Deviations

1. **Signed-translation end-to-end fixture is hypothesis-level, not a genuine
   solver certificate.** The public accepted ring fixture's C2-R-B certificate
   is the all-zero gauge class, so no genuine solver output with a non-zero
   integer/negative translation is reachable from the existing bundle. The
   test therefore rebuilds a **structurally valid certificate through C2-R-B's
   own sealer** (`_seal_record` + `_certificate_payload`) with signed values on
   all three axes and compares hypothesis derivation against the exact
   modulo-one counterpart and the all-zero baseline. This exercises the real
   changed code path (`_translations_by_frame` -> `_derive_hypotheses` ->
   contributors -> digests) with full exact-rational equality.
2. **P1-B branch adds three consistency checks.** The repaired branch also
   requires the chosen domain residue's local axis, derived label and source
   digest to agree with the P1-B record. These invariants already hold on all
   accepted, replayed inputs (the pre-existing `source_sha in accepted_sources`
   membership is equivalent for valid digests); they only convert an
   inconsistent forged record into the existing typed
   `axis_transform_invalid` refusal instead of a wrong-axis hypothesis.
3. **Path A diagnostic left unchanged.** `diagnose_full_patch_consistency`
   still hard-codes `u/v -> 0/1` and quadrant `0`. It is explicitly
   non-promotable (never emits `determined`, never sets `alpha`), is not the
   "P1-B contributor branch" named by the task, and reads full-patch phase
   evidence without domain labels. Recorded here as a residual observation for
   Manager/Astra; changing it would be a separate, unscoped diagnostic change.

## Unresolved concerns (honest limits; not blockers)

- Branch 2's label derivation is verified through the V2-A domain atoms, whose
  labels are themselves validated against the sorted-tangent convention by
  `plane_vector_domain.LateralAxisFactor`; the new test asserts the same
  convention directly for all normals and quadrants, but a full physical bundle
  with non-zero **relative** quadrant is not constructible from the existing
  fixture, so absolute contributions at quadrants 1/2/3 are covered only by the
  shared-transform table check, not by an end-to-end anchor run.
- The signed-translation test uses a resealed certificate rather than a solver
  acceptance because the accepted fixture's gauge class is zero (Deviation 1).
  The parser itself is additionally cross-checked exhaustively against C2-R-B's
  own `_validate_dyadic` over the signed dyadic lattice used by the repair.

## Blocker

None. Both approved repairs are implemented inside the routed scope; all
required checks pass; no hash mismatch, scope violation or architectural
inconsistency was found. Placement, region-support rasterization, volume
emission, CLI/UI, persistence and production wiring remain blocked and
untouched. Stop for Manager/Astra review.

## Files written

- `src/mc_photogrammetry/reconstruction/plane_phase_anchor.py` (signed-dyadic
  translation parser + single parse site; P1-B actual tangent-label mapping)
- `tests/test_plane_phase_anchor.py` (all-axis signed/permutation/replay and
  P1-B label tests; per-normal fixture parameterization)
- `docs/reports/M4-placement-prerequisite-compatibility-repair-2026-09-19.md`
  (this report)
