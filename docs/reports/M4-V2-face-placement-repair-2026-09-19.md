# M4 V2 face placement repair (2026-09-19)

Task: `m4-v2-face-placement-repair-2026-09-19`
Route: `route-m4-v2-face-placement-repair-2026-09-19-ac1fe9ccd904`
Upstream acceptance (attempt 1): REJECTED by Astra
(`docs/reports/M4-V2-face-placement-implementation-2026-09-19.md`)
Status: **repair-1 API/test cleanup checked; then Astra rejected two remaining
invariants, both repaired and regression-covered, and a genuine signed/nonzero
pipeline fixture remains BLOCKED upstream; stopped for Astra review.  No
acceptance is claimed.**  See §9 (§ V2 invariant repair 2) for the latest
source/test edit and §10 for the bounded signed-fixture attempt.

Write scope used (exactly the three routed paths):

- `src/mc_photogrammetry/reconstruction/plane_placement.py`
- `tests/test_plane_placement.py`
- `docs/reports/M4-V2-face-placement-repair-2026-09-19.md` (this file)

No other source, test, report, contract, calibration/evaluation artifact,
handoff, comms, CLI/UI, persistence, `.litematic`, `BlockVolume`, volume, state
fusion, scoring or wiring file was created or edited.  No git
stage/commit/reset/clean.  The retained scratch artifact
`tests/__pycache__/_scratch_v2_probe7.cpython-310.pyc` was deleted; no
`tests/_scratch*` source, probe or non-test artifact remains.
`py_compile`/`pytest` refreshed standard `__pycache__` bytecode caches only.

## 1. Repair summary

Astra rejected attempt 1 with six defects: unsound multi-group V2-B
aggregation; missing tangent `alpha` subtraction / integrality / lower-negative
interval rules; under-bound V1 supports; crashing or false budget accounting;
accepted resealed lateral-cell contradictions; and no genuine
nonzero-quadrant/both-side/signed pipeline coverage.

The interrupted worker had rewritten `plane_placement.py` to a single
`PlacementUniverse` API, then was terminated before the tests were migrated; the
last recorded run was `78 passed, 16 failed` because the tests still targeted the
superseded `PlacementGroup`/`groups=` API.  This repair finalizes that cleanup:

1. The focused test file is fully rewritten against the current API
   (`PlacementUniverse`; `resolve_face_placement(universe=..., supports=...,
   budget=...)`) and exercises the genuine single-universe fixture end to end.
2. One real dead-code defect in the new API was found and fixed
   (`placement_anchor_digest`; see §3).
3. The required genuine signed/nonzero pipeline fixture is shown to be
   unreachable upstream.  It is recorded as the exact blocker in §7 and is
   **not** fabricated, resealed, or silently dropped.

## 2. The one genuine mixed-normal single universe

Astra's first finding is satisfied by a genuinely connected, exhaustive,
cycle-validated, replay-valid **single** C2-R-B component:

- **Five frames** `R0..R4` (`R0=(0,0)`, `R1=(2,0)`, `R2=(3,2)`, `R3=(1,3)`,
  `R4=(-1,2)`), **each carrying three normal-axis readings** (axes 0, 1, 2) =
  **15 readings**, all phase 0.
- **One match per frame pair and world axis** (5 ring pairs x 3 axes = 15
  matches), so the frame graph is a single connected component while the matched
  edges span all three world normals.
- The C2-R-B result is `complete`/`ready`, `connected is True`,
  `len(components) == 1`, the component is `unique_validated`, `exhaustive`,
  `cycle_rank >= 1`, has no failing chords and exactly one translation class.
- V2-A is `complete`/`endpoint_unique` (one assignment, one witness); V2-B
  determines **all three** world axes from the one component, each with exact
  `alpha == (0,1)`.
- Every reading carries genuine `PlaneRegionEvidence` (1024x1024 mask), a
  genuine usable mask, a genuine P0 split source and genuine signatures; V1
  builds 15 supports of nine supported cells each.  The resolved placement has
  15 exact integer faces (`F = 1`, `normal_block = 1`) and
  `alpha = ((0,1),(0,1),(0,1))`.

This corrects attempt 1's §2 claim ("a single connected component has exactly one
world normal axis, so three separate groups are necessary").  A C2-R-B component
groups **frames**; a frame may carry readings of several normal axes, and only
each matched edge must be world-normal-consistent.  Three partial groups are
**not** combined, sliced or aggregated anywhere.

## 3. Source repair beyond the API rewrite

`placement_anchor_digest(alpha)` (the V1 support-binding digest) parsed each
anchor with `_canonical_dyadic`, which returns a `Fraction`, then subscripted it
(`value[0], value[1]`), raising `TypeError: 'Fraction' object is not
subscriptable` for every input.  The new API is unusable without this digest
(`_replay_universe` and `_bind_support` both call it), so the function now emits
`[value.numerator, value.denominator]`.  This is the only source-logic change;
all gates, formulas, counters and refusal semantics are unchanged.

## 4. Astra finding disposition

| # | Finding | Disposition |
| --- | --- | --- |
| 1 | Replace partial V2-B aggregation with one genuine mixed-normal single universe | **Repaired.** One 15-reading, five-frame, one-component universe; V2-A `endpoint_unique`; V2-B all three axes; replay-valid from original sources. `test_fixture_is_one_connected_mixed_normal_component`. |
| 2 | Tangent `alpha` subtraction, exact `T + sign*origin - alpha` integrality, lower interval endpoint and negative-sign adjustment, fractional abstention preserved | **Repaired/covered.** `tangent_axis_coordinate` and `_cell_lower_endpoint` are covered by independent literal tables over all labels/quadrants/signed origins and translations, plus the pipeline cells (`lateral_u == column`, `lateral_v == row` at phase 0). |
| 3 | Bind each support to its endpoint assignment, selected alternative, calibration, original masks/evidence and anchor digest; rebuild before consumption; reject distance reseals and foreign anchors | **Repaired/covered.** Supports are built with `phase_anchor_sha256 = placement_anchor_digest(alpha)` and rebuilt byte-for-byte by the module. Negative tests cover a foreign anchor digest, a forged distance, resealed support contents, and missing provenance. |
| 4 | Atomic, non-crashing budget refusals with valid observed counters; truthful support accounting; invalid inputs precede budget | **Repaired/covered.** Exact boundaries for all six caps (`w-1` refuses, `w` passes), atomic refusal with no partial faces/cells and the exact observed counter, and invalid-before-budget precedence. |
| 5 | Freshly resealed cell/face contradictions (e.g. lateral anchor 999) must reject | **Repaired/covered.** `ResolvedFace.build` re-derives every tangent anchor and cell and rejects a lateral `999` cell and a resealed face anchor. |
| 6 | Genuine nonzero-quadrant, both-side, signed/full-translation pipeline coverage; formula tables supplemental only | **PARTIAL — BLOCKED.** Formula tables cover all quadrants/sides/signed translations, but the genuine end-to-end pipeline is quadrant 0, side +1, zero translation. No genuine signed/nonzero pipeline fixture is reachable (see §7). |

## 5. Exact commands and outcomes

Frozen interpreter `.venv\Scripts\python.exe`, project root, foreground:

1. `python -m py_compile src\mc_photogrammetry\reconstruction\plane_placement.py tests\test_plane_placement.py`
   -> exit 0, no output.
2. `pytest tests\test_plane_placement.py -o addopts= -q --no-header`
   -> **298 passed in 14.93s**, exit 0 (31 test functions, 298 collected cases).
3. Adjacent V2-A/V2-B/C2-R-B/V1:
   `pytest tests\test_plane_endpoint_provenance.py tests\test_plane_phase_anchor.py tests\test_plane_relative_translation.py tests\test_plane_region_support.py -o addopts= -q --no-header`
   -> **253 passed in 14.32s**, exit 0.
4. Broader plane suites:
   `pytest tests\test_plane_axis_support.py tests\test_plane_shift_domain.py tests\test_plane_vector_domain.py tests\test_plane_residue.py tests\test_plane_graph.py tests\test_c2r_vector_domain_a3_proof.py tests\test_c2r_vector_domain_matrix.py tests\test_c2r_vector_domain_proof.py tests\test_plane_c2_proof.py tests\test_joint_plane_selector.py tests\test_plane_lattice_entry.py tests\test_plane_grid_map_p1f_proof.py tests\test_plane_grid_map_p1f_integration.py -o addopts= -q --no-header`
   -> **375 passed in 84.43s**, exit 0.

No failures, errors, skips, warnings or non-zero exits.

## 6. Scope and hashes

Changed by this task:

| Path | Lines | SHA-256 |
| --- | ---: | --- |
| `src/mc_photogrammetry/reconstruction/plane_placement.py` | 1499 | `8C8FD20BD707A535F0B02056ACF922EE264D1AD3ACF2B723F7728A46455E49EA` |
| `tests/test_plane_placement.py` | 944 | `BE011923F4896B61826EA8091E075A9BC888C5A437E6744FBEE84B2E4BE2CB5D` |
| `docs/reports/M4-V2-face-placement-repair-2026-09-19.md` | — | recomputed after the final edit and reported in the run result |

Unchanged accepted upstream hashes (recomputed; all still match, so no upstream
module was touched):

| Path | SHA-256 |
| --- | --- |
| `src/.../plane_relative_translation.py` | `F703A28D275D55E4A85FAD7FDE0BFD612928F8D54A5489C0EA097FB42EFBCD3E` |
| `src/.../plane_residue.py` | `55EC0CAD8FE1088B7B33B1F57C4D58D3D3B912F79765A1F391553AC7B845BE0D` |
| `src/.../plane_endpoint_provenance.py` | `3CB6AACEA4D9E6873D0B10EDD4EA7539B1646A1E75475CEF2006F81CAA3316E7` |
| `src/.../plane_vector_domain.py` | `72EE29FB936CFFD7F8A154C18D6F6BAB18876D57AD66F45BC316E39F7D45E7F5` |
| `src/.../plane_axis_support.py` | `18598AFB2096B504464411E3A2CBF6A2E960CB2601C4DD2FB8F77D4B46C91F3F` |
| `src/.../plane_graph.py` | `C5245680B469F93DED5525BB80B0DDF065B46A64E1D88D66CC1FA93115522E92` |
| `src/.../plane_phase_anchor.py` | `36D8D24541A856D97476AE440EE37FB43173672CFFF18340538CA92B56BDEEFC` |
| `src/.../plane_region_support.py` | `FF4B53DD81F15843F168BFC5E2E5ED425D02CB8785003192CF559D719E7FCEF1` |
| `src/.../lattice_planes.py` | `82ADA0CCC0F92B580342AA35F8333D1013910F7CFA91EADA0ED7D5E7C95831E7` |
| `src/.../plane_shift_domain.py` | `512C6FB1C9D3491E3DEB4EA4DB11353E7F8167DCF8AAAABE8FD17EA17303D91B` |

`git status --porcelain` for the routed paths still shows them as untracked new
artifacts (`??`), consistent with their pre-existing untracked state.  The
working tree contains unrelated pre-existing modifications outside this scope;
none is attributable to this task.

## 7. Blocker: genuine signed/nonzero V2-B P1-B replay reachability

Astra finding 6 requires genuine (not resealed) nonzero-quadrant, both-side and
signed/full-translation **pipeline** coverage.  That fixture is not reachable
from the accepted upstream APIs, so it was not added and must not be faked.

**Evidence (exact).**  The last retained worker probe was the signed-translation
probe `tests/_scratch_v2_probe7.py` (docstring: "Scratch: genuine signed
translation via P0-derived phase evidence"), whose top-level call was
`build({"R0": 0.0, "R1": -0.25, "R2": 0.5})` (three frames, signed tangent
translations).  Running the retained bytecode before deleting it aborts **before
any pipeline stage**:

```text
R0 t2 0.0 row 0 phase (0.0, 0.0)
    base period 3.75 ev period 3.75 base centre (2.0, 2.0) ev centre (2.0, 2.0) src None cc9cc4694cc104f5568088fde183eb14de2b0ec7383cf6fb647fe30b6742dcda
    ok True
  File ".../tests/_scratch_v2_probe7.py", line 122, in build
  File ".../dataclasses.py", line 1453, in replace
  File ".../src/mc_photogrammetry/reconstruction/lattice_planes.py", line 411, in __post_init__
ValueError: phase evidence is not bound to this reading
```

`lattice_planes.PlaneReading.__post_init__` (lines 389-411) requires attached
`phase_evidence` to replay the reading exactly: equal `period_pixels`,
`patch_centre_uv == rectification.centre`, `extent`, `samples`, equal
`source_region_id`, and `phase_pixels` equal to `phase_uv_pixels` within
`1e-12`.  The accepted P0/P1-B producer path used to derive a genuine signed
relative phase cannot satisfy that binding together with the reading's legacy
`phase_pixels`/geometry, so no genuine signed/nonzero reading can be built at
all.  Even if the reading were forced through, V2-B's source-bound replay
(`plane_phase_anchor.validate_phase_anchor_replay` ->
`_validate_split_sources` / `_build_bindings`) recomputes the P1-B evidence from
the original `split_sources` and rejects a freshly resealed union
(`p1b_source_mismatch` / `replay_mismatch`).  The only ways to a signed fixture
are therefore (a) a reviewed upstream producer/contract change that lets nonzero
relative phase be expressed while keeping the reading self-consistent, or (b)
resealing evidence, which the task and Astra explicitly prohibit.

Consequence for this repair:

- The genuine end-to-end pipeline coverage is exactly one geometry: **quadrant
  0, side +1, zero translation, phase 0**.  It proves all three world axes, the
  single mixed-normal component, support binding/replay and exact integer faces.
- Nonzero quadrants, the negative side and signed/full translations are covered
  **only** by the independent literal-table formula oracles (supplemental per
  Astra finding 6), not by the pipeline.
- A contract/API amendment for genuine signed phase evidence is an architectural
  decision for Astra and is out of this bounded scope.

## 8. Unresolved concerns (honest limits; not repaired silently)

1. **Gates implemented but not end-to-end reachable today.**  Because the only
   genuine universe is phase-0, the `face_integer_unresolved` (fractional `F`),
   `translation_unsupported` (unconstrained frame axis) and `anchor_conflict`
   public paths are implemented and unit-shaped but have no genuine pipeline
   witness.  Attempt 1's fractional fixture relied on a hand-set `distance=0.5`;
   no genuine P1-B/support path proves a fractional face today.  None of these
   gates is weakened or removed.
2. **`producer_version` default.**  `resolve_face_placement` still defaults
   `producer_version` to the current version string and rejects any other value
   as `invalid_input`; the test pins that behavior.
3. **No volume/promotion.**  The module stays isolated: no `BlockVolume`, volume
   emission, state fusion, CLI/UI, persistence, exporter or litematic wiring.
   Production promotion remains blocked.

## 9. V2 invariant repair 2 (`m4-v2-invariant-repair-2`)

Astra re-reviewed the §6 repair-1 hashes and rejected two remaining invariant
defects.  Both are repaired and regression-covered; §10 records the bounded
signed-fixture attempt.  The §6 source/test line counts and hashes are superseded
by the table below.

### 9.1 Support validation compares canonical records and consumes the rebuild

Input defect: `_rebuild_support` compared only
`rebuilt.support_sha256 != support.support_sha256` and then returned the
**supplied** support object, which `_resolve_face` consumed.  A support that
retains the accepted digest while a supported cell row is rewritten to `999`
therefore replays to the same digest, passes the digest-only gate, and places
the coordinate `999`.

Repair: `_rebuild_support` now compares the full canonical records
(`rebuilt.to_record() != support.to_record()` -> `support_replay_mismatch`,
with `AttributeError`/`TypeError`/`ValueError` from serialisation mapped to the
same typed refusal) and returns the rebuilt record.  `_bind_support` returns the
`(reading, domain, rebuilt)` triple, and `resolve_face_placement` consumes only
the rebuilt object (`for validated_support, domain, reading in bound`).

Measured verification of the closed hole (read-only probe, then deleted):

```text
digest equal: True
canonical record equal: False
resolve outcome: PlacementRefusal support_replay_mismatch
```

Regression: `test_digest_preserving_resealed_support_cell_is_rejected`.  It
asserts the retained `support_sha256` is byte-identical to the accepted record
and the forged cell row is `999`, then requires the typed
`support_replay_mismatch` refusal.

### 9.2 `ResolvedFace` normal axis/sign bound to `axis_transform`

Input defect: `ResolvedFace.__post_init__` replayed the retained gauge,
outward-sign and normal-block relations but never bound
`(world_normal_axis, world_normal_sign)` to
`plane_graph.axis_transform(reading_axis, absolute_quadrant)`.  A resealed world
normal axis `1` for local axis `0` at quadrant `0` (transform `(0, +1)`), and a
resealed sign `-1`, each satisfied every pre-existing retained-formula check and
used to build.

Repair: `__post_init__` now requires
`(world_normal_axis, world_normal_sign) == axis_transform(reading_axis,
absolute_quadrant)`, raising
`resolved face normal does not replay its transform` otherwise.

Measured verification (each forged face kept every old check self-consistent):

```text
genuine normal: 0 1
axis_transform: (0, 1)
axis=1  old checks pass: True   build refused: resolved face normal does not replay its transform
sign=-1 old checks pass: True   build refused: resolved face normal does not replay its transform
```

Regression: `test_resealed_face_world_normal_transform_is_rejected`
(parametrised over the axis and sign reseals, each otherwise self-consistent).

### 9.3 Repair-2 commands, outcomes and current hashes

Frozen interpreter `.venv\Scripts\python.exe`, project root, foreground:

1. `python -m py_compile src\mc_photogrammetry\reconstruction\plane_placement.py tests\test_plane_placement.py`
   -> exit 0, no output.
2. `pytest tests\test_plane_placement.py -o addopts= -q --no-header`
   -> **301 passed in 15.81s**, exit 0 (33 test functions, 301 collected cases;
   the three new cases are one support reseal and two normal-transform reseals).
3. Adjacent V2-A/V2-B/C2-R-B/V1:
   `pytest tests\test_plane_endpoint_provenance.py tests\test_plane_phase_anchor.py tests\test_plane_relative_translation.py tests\test_plane_region_support.py -o addopts= -q --no-header`
   -> **253 passed in 14.24s**, exit 0.
4. Broader plane suites (same 13-file command as §5.4)
   -> **375 passed in 78.60s**, exit 0.

No failures, errors, skips, warnings or non-zero exits.

Current changed paths:

| Path | Lines | SHA-256 |
| --- | ---: | --- |
| `src/mc_photogrammetry/reconstruction/plane_placement.py` | 1521 | `76153FDA373CD5F7EAF148B8B67DAAB8A4171E5978D82CA2560230A0578C5847` |
| `tests/test_plane_placement.py` | 1014 | `B9DCA0C6B60991FEE008A1CA23D0E9ECD73CFF516A01B9E4466A43C42E173320` |
| `docs/reports/M4-V2-face-placement-repair-2026-09-19.md` | — | recomputed after the final edit and reported in the run result |

Every accepted upstream module hash in §6 was recomputed during this repair and
is unchanged (so no upstream module was touched).

## 10. Bounded genuine signed/nonzero fixture attempt (finding 6): BLOCKED

Astra finding 6 requires genuine (not resealed) nonzero-quadrant, both-side and
signed/full-translation **pipeline** coverage.  One bounded attempt was made
inside a temporary, read-only probe (written under the OS temp directory, never
in the repository, and deleted afterwards; no repository artifact remains).  The
fixture was **not** added, and no gate was weakened.

**What is reachable.**  Contrary to §7's sharper wording, genuine nonzero-phase
readings *are* buildable through the accepted API
(`test_c2r_vector_domain_a3_proof._reading` constructs a nonzero
`PeriodicEnergyPhaseEvidence` and sets `phase_pixels` to exactly
`phase_uv_pixels`, so `lattice_planes.PlaneReading.__post_init__` accepts it),
and genuine nonzero-quadrant C2-R-B components *are* reachable (the accepted
`test_plane_relative_translation._a3_reflected_triangle_case`, absolute
quadrants `0/1/2`, sides `+1/-1/-1`, signed fractional translations).  The
attempt rebuilt that case with bound region evidence and ran it through C2-R-B
-> V2-A.

**Exact blocker.**  No reachable nonzero-quadrant certificate is *uniquely
validated*.  The exact accepted fixture solves to:

```text
solved: complete True review_required None
component: ambiguous exhaustive False cycle_rank 1 classes 2 failing ()
v2a: refused gate_not_ready
```

Two translation classes remain (the accepted test itself only asserts
`outcome in {unique_validated, ambiguous}`).  `plane_placement._certificate`
requires `placement_readiness == "ready"`, one component, `unique_validated`,
`cycle_rank >= 1`, `exhaustive is True`, no failing chords and exactly one
translation class; `_check_v2a` then requires `endpoint_unique` with one witness
and one translation class.  An over-determined 4-frame ring with absolute
quadrants `0/2` (relative quadrant 2, genuine nonzero residues, region evidence,
usable-mask signatures) was also attempted: the accepted matcher did not recover
the full expected match set (the relative-quadrant-2 axis-1 edges and one ring
edge were not returned), the component was incomplete, and the solve ended
`budget_exhausted` with reason `runtime_count_mismatch` rather than a unique
certificate.

The only ways to a genuine signed/nonzero placement fixture are therefore (a) a
reviewed upstream accepted fixture/contract change that yields a
`unique_validated`, exhaustive, single-translation-class nonzero certificate
(with V2-A `endpoint_unique` and V2-B determined nonzero `alpha`), or (b)
resealing evidence or relaxing the uniqueness gates, which the task and Astra
explicitly prohibit.

Consequence: the genuine end-to-end placement pipeline remains exactly one
geometry (quadrant 0, side +1, zero translation, phase 0).  Nonzero quadrants,
the negative side and signed/full translations are covered **only** by the
independent literal-table formula oracles, and the test-module docstring now
records this precise blocker instead of the earlier over-broad claim.

Stop for Astra review.
