# M4 V2 resolved face placement implementation (2026-09-19)

Implementation task: `m4-v2-face-placement-implementation-2026-09-19`
Route: `route-m4-v2-face-placement-implementation-2026-09-19-e1f548bb4bfe`
Upstream design (accepted): `docs/reports/M4-placement-volume-bridge-contract-2026-09-19.md`
Prerequisite formula authority: `docs/reports/DEEPSEEK-M4-V2-placement-prerequisite-audit-2026-09-13.md`
Worker: Sol-equivalent CLI worker (DeepSeek), bounded implementation
Status: **implementation and all required checks complete; stopped for Astra review.**

Write scope used (exactly the three routed paths):

- `src/mc_photogrammetry/reconstruction/plane_placement.py` (new, 1117 lines)
- `tests/test_plane_placement.py` (new, 685 lines)
- `docs/reports/M4-V2-face-placement-implementation-2026-09-19.md` (this file)

No other source, test, report, contract, calibration/evaluation artifact,
handoff, comms, CLI/UI, persistence, `.litematic`, `BlockVolume`, volume, state
fusion, or wiring file was created or edited. No git stage/commit/reset/clean.
`py_compile`/`pytest` refreshed standard `__pycache__` bytecode caches only.

## 1. Contract interpretation (exact, with authority)

The bridge contract states the formula but names `c` and `d` only symbolically.
The accepted prerequisite audit fixes their meaning (`DEEPSEEK-M4-V2-placement-
prerequisite-audit-2026-09-13`, lines 24-42): for reading label axis `a`,
quadrant `q`, camera-to-plane side `s`, retained alternative distance `d`, and
solved translation `t`,

```text
world_axis, axis_sign = axis_index(a, q), axis_sign(a, q)
face = t[world_axis] + axis_sign * s * d
outward sign = -axis_sign * s
```

so in the bridge-contract notation `(n, sn) = axis_transform(a, q)`,
**`c = side`** and **`d = distance_blocks`**:

```text
face_gauge   = T[n] + sn * c * d
F            = face_gauge - alpha[n]
outward_sign = -sn * c
normal_block = F - 1 if outward_sign > 0 else F
```

`a` is the reading's own (normal) label axis; the sentence "tangent labels are
the sorted axes excluding the normal label and use the same transform" supplies
the two lateral axes `others[0]` (`u`) and `others[1]` (`v`) in ascending label
order, each mapped by the same `axis_transform`. The module implements this
literally in `face_axis_coordinate` and derives `c`/`d` from retained evidence
(`reading.side`, the V1 retained `alternative.distance_blocks`), never from a
caller-supplied world position. `T` is the full signed reduced-dyadic
C2-R-B certificate translation (integers and negative fractions preserved, never
reduced modulo one); `alpha[n]` is the V2-B reduced dyadic seam anchor.

## 2. Architecture finding: one world-normal axis per connected component

`plane_axis_support` refuses `axis_transform_invalid` on any matched edge whose
endpoints disagree on the mapped world normal axis
(`src/.../plane_axis_support.py:974-981`), and C2-R-B builds its components from
those supported edges. A connected component therefore contains readings of
exactly **one** world normal axis, and V2-B hypotheses are built only from
tangential residue labels, so one accepted V2-B result determines at most its
two tangent axes and always abstains `no_hypothesis` on its own normal axis.
The V2-B gate additionally requires `len(c2rb_result.components) == 1`.

Consequently a *single* connected, exhaustive, cycle-validated, replay-valid
bundle can never determine all three world axes. The only genuine construction
is one complete accepted group per world normal axis, and this is what the
fixture builds and the module consumes: three whole five-frame rings (normals 0,
1, 2), each reaching `unique_validated` with `cycle_rank >= 1`, each with an
`endpoint_unique` V2-A result, each V2-B determination replayed from its own
original split sources. No counter is faked, no world position is supplied, and
no partial ring is combined or sliced; each group is a complete accepted
component, and the union determines `alpha[0]`, `alpha[1]`, `alpha[2]`.
`test_one_connected_component_cannot_cover_three_axes` asserts the invariant on
the live fixture, and `test_fixture_groups_are_complete_and_cover_all_three_axes`
asserts completeness and coverage. This reading is the one point flagged for
Astra; see §7.

## 3. Implementation

`plane_placement.py` exposes a versioned, immutable, self-validating API:

- `PlacementBudget` / `PlacementWork` / `PlacementRefusal` — explicit positive
  exact-integer caps with no defaults, exact observed counters, and a closed
  typed refusal enum (`anchor_axis_unresolved`, `anchor_conflict`,
  `translation_unsupported`, `face_integer_unresolved`, `replay_mismatch`,
  `c2rb_gate_not_unique`, `endpoint_gate_not_unique`, `anchor_gate_not_unique`,
  `budget_exhausted`, …). A budget refusal is atomic, carries the observed
  counter, and embeds no faces or cells.
- `PlacementGroup` — one complete accepted world-normal group carrying the
  original C2-R-B/V2-A/V2-B records and their replay tuple.
- `face_axis_coordinate` — the exact signed formula above; returns
  `integer=False` and `face=None` rather than rounding.
- `TangentAxisMapping` / `PlacedCell` / `ResolvedFace` —
  `ResolvedFace` preserves the three-axis translation, the three-axis anchor,
  the exact `face_gauge`, the integer `face`/`normal_block`, the outward sign,
  both tangent mappings with their fractional world anchors, and every
  supported V1 cell with its exact fractional lateral anchors and the V1
  `support_sha256`/`source_region_id` binding. Its constructor re-derives the
  gauge relation, the anchor integrality and the outward block, so a freshly
  resealed face cannot contradict its own inputs.
- `ResolvedFacePlacement` — `gauge_state =
  "unresolved_whole_build_integer_translation"` with `gauge_axes = (0,1,2)`, the
  exact `alpha` vector, canonical per-group bindings, canonically ordered faces,
  and a canonical digest.

Validation precedence is fixed: type/budget validation, then per-group C2-R-B
gate (`complete`/`ready`/`connected`, one component, `unique_validated`,
`exhaustive`, `cycle_rank >= 1`, no failing chords, every cycle residual with
`matching_provenance_count > 0`, exactly one translation class), V2-A
`endpoint_unique` gate, V2-B determined-axis read, then three source-bound
replays — `plane_relative_translation.validate_relative_translation_graph`,
`v2a.validate_endpoint_provenance_result`, and
`v2b.validate_phase_anchor_replay` — before any anchor is consumed. Each
required `(frame, world axis)` must be present in the accepted C1
`frame_support` with `connected_to_global_anchor` and no refusal, so the solver's
least-squares min-norm `0.0` fill is never accepted as evidence. All input,
gate and replay validation completes before any budget decision, and budget
excess is atomic.

## 4. Fixture (fixture-first, genuine end to end)

`tests/test_plane_placement.py` builds, for each world normal 0/1/2, a full
five-frame C2-R-B ring through the public pipeline: `plane_signature` (with the
explicit usable mask required by proposal-region readings), `match_plane_pair`,
`build_axis_support`, `build_shift_domains`, `build_lateral_vector_domains`,
`solve_relative_translation_graph`, `analyze_endpoint_provenance`, and
`analyze_phase_anchor`. Readings carry genuine `PlaneRegionEvidence` (1024x1024
mask), so V1 `build_aligned_region_support` produces genuine supports with nine
supported cells each for the integer groups. The integer groups use extent 2.0
with distance 1.0 (period 3.75), so the resolved face is the exact integer block
`F = 1`, `normal_block = 1`; the fractional group reuses the original accepted
public geometry (extent 4.0, distance 0.5), whose genuine half-block face gauge
is the typed `face_integer_unresolved` case. Expected values are declared by the
fixture and by an independent literal transform table; nothing is read back from
the module under test.

Coverage: all three world axes; all three normal labels, all four quadrants and
both sides against the literal table; negative and full signed translations
(`0`, `3`, `-1/4`, `5/2`); exact integer success; fractional abstention;
ambiguous/disconnected/incomplete-cycle abstention; anchor-axis-unresolved and
anchor-conflict paths; V2-A endpoint-gate and V2-B anchor-gate abstention;
permutation-invariant canonical digest; resealed split-source and resealed
anchor replay rejection; replay-tolerance mismatch; missing support provenance;
atomic budget exhaustion; invalid-before-budget precedence; AST import/symbol
isolation; no upstream mutation.

## 5. Exact commands and outcomes

Frozen interpreter `.venv\Scripts\python.exe`, project root, foreground:

1. `python -m py_compile src\mc_photogrammetry\reconstruction\plane_placement.py tests\test_plane_placement.py`
   → exit 0, no output.
2. `pytest tests\test_plane_placement.py -o addopts= -q --no-header`
   → **94 passed in 6.78s**, exit 0.
3. Adjacent V2-A/V2-B/C2-R-B/V1:
   `pytest tests\test_plane_endpoint_provenance.py tests\test_plane_phase_anchor.py tests\test_plane_relative_translation.py tests\test_plane_region_support.py -o addopts= -q --no-header`
   → **253 passed in 15.34s**, exit 0.
4. Broader plane suites:
   `pytest tests\test_plane_axis_support.py tests\test_plane_shift_domain.py tests\test_plane_vector_domain.py tests\test_plane_residue.py tests\test_plane_graph.py tests\test_c2r_vector_domain_a3_proof.py tests\test_c2r_vector_domain_matrix.py tests\test_c2r_vector_domain_proof.py tests\test_plane_c2_proof.py tests\test_joint_plane_selector.py tests\test_plane_lattice_entry.py tests\test_plane_grid_map_p1f_proof.py tests\test_plane_grid_map_p1f_integration.py -o addopts= -q --no-header`
   → **375 passed in 107.96s**, exit 0.

No failures, errors, skips, warnings or non-zero exits.

## 6. Scope and hashes

Written by this task:

| Path | Lines | SHA-256 |
| --- | ---: | --- |
| `src/mc_photogrammetry/reconstruction/plane_placement.py` | 1117 | `cb3d3cb4adefdf4a448b64979e91917c0e47853d91885b8b98c642d619cd643c` |
| `tests/test_plane_placement.py` | 685 | `762bd40db02459049eb71924b235286cb8e8185390e65122486c1dba7e9ede58` |
| `docs/reports/M4-V2-face-placement-implementation-2026-09-19.md` | — | recomputed after the final edit and reported in the run result |

`git status --porcelain` for the routed code/test paths shows them as untracked
new artifacts (`??`), consistent with their pre-existing untracked state. The
working tree contains unrelated pre-existing modifications outside this scope;
none is attributable to this task.

Unchanged upstream accepted hashes (recomputed; all still match, so no upstream
module was touched):

| Path | SHA-256 |
| --- | --- |
| `src/.../plane_relative_translation.py` | `f703a28d275d55e4a85fad7fde0bfd612928f8d54a5489c0ea097fb42efbcd3e` |
| `src/.../plane_residue.py` | `55ec0cad8fe1088b7b33b1f57c4d58d3d3b912f79765a1f391553ac7b845be0d` |
| `src/.../plane_endpoint_provenance.py` | `3cb6aacea4d9e6873d0b10edd4ea7539b1646a1e75475cef2006f81caa3316e7` |
| `src/.../plane_vector_domain.py` | `72ee29fb936cffd7f8a154c18d6f6bab18876d57ad66f45bc316e39f7d45e7f5` |
| `src/.../plane_axis_support.py` | `18598afb2096b504464411e3a2cbf6a2e960cb2601c4dd2fb8f77d4b46c91f3f` |
| `src/.../plane_graph.py` | `c5245680b469f93ded5525bb80b0ddf065b46a64e1d88d66cc1fa93115522e92` |
| `src/.../plane_phase_anchor.py` | `36d8d24541a856d97476ae440ee37fb43173672cfff18340538ca92b56bdeefc` |
| `src/.../plane_region_support.py` | `ff4b53dd81f15843f168bfc5e2e5ed425d02cb8785003192cf559d719e7fcef1` |

## 7. Deviations and unresolved concerns

1. **Formula symbols.** The bridge contract defines `c`/`d` only by reference;
   the accepted prerequisite audit supplies `c = side`, `d = distance_blocks`.
   The implementation follows that audit literally and documents it. If Astra
   intended a different `c`/`d` source, only `face_axis_coordinate`'s two
   arguments and the two derivation sites in `_resolve_face` need to change.
2. **All-three-axis reading.** As proved in §2, a single connected component has
   one world normal axis, so the task's "one connected … translation class" and
   "V2-B determined on all three axes" can only both hold across one complete
   group per world normal axis. The fixture and module take that reading. This is
   the primary point requiring Astra's confirmation.
3. **Quadrants/sides/signed translations.** The genuine public fixture is
   quadrant 0, side +1, with an all-zero gauge certificate; no genuine solver
   output with a non-zero or negative translation is reachable from the existing
   bundle (the same limitation recorded by the accepted compatibility repair).
   Quadrant/side/signed-translation coverage is therefore provided by the
   independent literal-table oracle over the exact formula, not by an end-to-end
   anchor run. The genuine end-to-end paths cover integer success and fractional
   abstention.
4. **V1 support geometry in the fixture.** The integer groups' supports are built
   by the accepted V1 builder from genuine region evidence and yield nine
   supported cells; the fractional (half-distance) group's supports build
   successfully but some of their cells are `unknown` under V1's conservative
   policy. Only the face integer coordinate is consumed before the abstraction,
   so this does not weaken the fractional refusal.

## 8. Blocker

None at the implementation level. All required checks pass, no upstream accepted
hash changed, and the module stays isolated (no volume, `BlockVolume`, state
fusion, CLI/UI, persistence, or litematic wiring). The open contract question in
§7.2 is referred to Astra; no production promotion is claimed.

Stop for Astra review.

## Files written

- `src/mc_photogrammetry/reconstruction/plane_placement.py`
- `tests/test_plane_placement.py`
- `docs/reports/M4-V2-face-placement-implementation-2026-09-19.md` (this file)
