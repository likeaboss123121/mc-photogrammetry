# DS-IMG-GEOM-PHASE-CAPTURE-R2 - 22-column face-row index schema + side equation

Task ID: `DS-IMG-GEOM-PHASE-CAPTURE-R2`. Owner: DeepSeek (bounded repair + focused tests).
Date: 2026-10-04. Status: **repair complete; focused tests and `py_compile` pass; stopped for
lead review. No pinned PNG, reconstruction or image run in this task.**

This report records the R2 repair of the evaluation-only per-face phase capture requested by the
`2026-10-04T06:20:00-05:00` R2 assignment, the `2026-10-04T06:40:00-05:00` index-schema hold,
and the `2026-10-04T06:55:00-05:00` endpoint-pair binding addendum. It is not a run report, an
accuracy estimate, a gate change, or an authorization to observe the pinned images. The
interrupted first R2 attempt is not treated as accepted.

## 1. Result summary

- The canonical 22-column face-row schema from the hold is now the single schema used by the
  producer, the replay validator and the dedicated tests:
  `[0 match_index, 1 left_frame, 2 left_plane, 3 left_axis, 4 left_b, 5 left_distance,
  6 left_side, 7 left_sign, 8 right_frame, 9 right_plane, 10 right_axis, 11 right_b,
  12 right_distance, 13 right_side, 14 right_sign, 15 relative_quadrant, 16 face_scalar,
  17 own_quadrant, 18 other_quadrant, 19 face_phase, 20 phasor_real, 21 phasor_imag]`.
- The only index defect the hold named was in `_validate_complete_phase_payload`; it is fixed.
  The producer already emitted all 22 columns and the side-aware scalar.
- The face equation is the solver's exact expression `translation + axis_sign * side * distance`
  (`plane_graph._evaluate_assignment`), with both endpoint `side` values serialized and each
  validated as exactly `+/-1`.
- Per the addendum, each match's two rows are now bound as a reversed endpoint pair: one row's
  own tuple (columns 1-7) must equal the other's partner tuple (columns 8-14) and vice versa,
  with a shared relative quadrant and the flipped relative-quadrant relation. A re-sealed
  single-row partner-side tamper is therefore refused.
- An independent fractional-distance, mixed-side fixture now exercises the equation; the
  integer-only parity fixture is explicitly shown to be insufficient on its own.
- Per-axis LSQ constraints, the shared `failed_assignment_sha256` domain, the runner projected
  alias binding, and default production result/digest parity are preserved unchanged.
- No debug print or temporary probe file was added; the accepted 3-frame observer paths are
  byte-unchanged.

## 2. Producer / validator / digest / test index audit

The hold's three named validator confusions, and their corrections:

| Named defect | Was | Now |
|---|---|---|
| `row[16]` used as own quadrant | face_scalar | `row[17]` (own_quadrant) |
| `row[17]` used as other quadrant | own_quadrant | `row[18]` (other_quadrant) |
| `row[15]` used as scalar | relative_quadrant | `row[16]` (face_scalar) |

Corrected call sites in `_validate_complete_phase_payload`: both `plane_graph.axis_transform`
calls (`row[17]`, `row[18]`), the per-frame quadrant map entries (`row[17]`, `row[18]`), and the
scalar read (`row[16]`). The sign cross-check keeps `row[7]`/`row[14]`; the side equation keeps
`row[5]`/`row[6]`/`row[7]` and `translation_by_frame[row[1]]`; the phase/phasor cross-check keeps
`row[19]`/`row[20]`/`row[21]`.

The rest of the audit found the other seams already coherent with the schema:

- **Producer** (`_collect_failed_phase_capture`): writes the 22 columns in canonical order
  (`match_index, own frame/plane/axis/b/distance/side/sign, other frame/plane/axis/b/distance/
  side/sign, relative_quadrant, scalar, own_quadrant, other_quadrant, phase, phasor_real,
  phasor_imag`). The own endpoint is columns 1-7 and the other endpoint 8-14; the row is emitted
  twice per active edge (own/other then other/own), matching the solver's left/right face order.
  A comment now states the full layout.
- **Scalar producer**: `axis_translation[frame][axis] + sign * side * distance`, exactly the
  solver's `axis_translation[match.left] + left_sign * left.side * left.distance_for(...)`.
- **Replay/digest**: `_phase_capture_payload` serializes the rows in order,
  `solver_phase_capture_record_digest` hashes `{domain, schema, payload}`, and
  `validate_solver_phase_capture` strips `record_sha256` before re-validating, so no index is
  applied twice or to the wrong domain. Unchanged in R2.
- **Runner** (`scripts/run_image_derived_graph_stage.py`): uses only the public
  `validate_solver_phase_capture` and the projected alias digest; no face-row index. Unchanged.
- **Tests**: `_world_axis_of` now reads `row[17]`; axis-count and fractional scalar reads use
  `row[16]`; the omitted/flipped-side mutations use columns 6/13 and 16/19/20/21. No test reads
  `row[15]` as a scalar or `row[16]`/`row[17]` as quadrants.

## 3. Endpoint-pair binding (addendum)

The per-row side equation checks only a row's own endpoint (columns 1-7). The serialized partner
endpoint (columns 8-14) is now bound independently:

- rows are grouped by `match_index` and must reduce to exactly two rows;
- the two rows must be reversed mirrors: `tuple(first[1:8]) == tuple(second[8:15])` and
  `tuple(first[8:15]) == tuple(second[1:8])`;
- both rows must carry the same `relative_quadrant` (`row[15]`);
- the reversed relation must hold: `(other_quadrant - own_quadrant) % 4` equals `row[15]` for
  one row and `(-row[15]) % 4` for the reversed row.

A re-sealed record that omits or flips the partner copy of a side in one row only therefore
cannot replay: the row schema refuses a missing side and the mirror binding refuses a flipped
one. The frame-quadrant coverage and assignment checks continue to bind the per-frame quadrants.

## 4. Side equation and refusal evidence

`_strict_face_side` admits only exact `+/-1` (int or integral float) and returns `None` for a
boolean, non-integral, non-finite or out-of-range value, so a malformed side is the typed
`invalid_phase_capture_diagnostic` refusal with field `endpoint_side` rather than a coerced
`+1`. On replay the validator recomputes
`expected_scalar = translation[world_axis] + row[7] * row[6] * row[5]` from the row's own
serialized sign/side/distance and refuses (`phase face scalar contradicts its side equation`)
when it differs.

The dedicated tests prove:

- a fractional fixture where every `distance_blocks` is non-integral and both `side = -1` and
  `side = +1` appear, and where the omitted-side expression `translation + axis_sign * distance`
  differs from the captured scalar on at least one row;
- each row's scalar/phase/phasor equals the solver's exact equation re-derived from the source
  readings and `plane_graph.axis_transform`, never from the captured scalar;
- replaying the omitted-side scalar under a re-sealed record is refused by the side equation;
- a missing own side is refused by the row schema and a flipped own side without a matching
  scalar is refused by the side equation;
- omitting or flipping the partner copy of a side in one row only is refused by the endpoint-pair
  binding (the `match_index` mirror check);
- a tamper that also mirrors the partner copy, follows the face equation, and rewrites the
  recorded axis recomposition is still refused by the solver-metric binding (`phase axis solver
  metrics contradict its recomposition`).

## 5. Checks performed

Run from the repository root with the baseline interpreter; no screenshot display, no image
decode, no reconstruction, no protected input read.

```
.\.venv\Scripts\python.exe -m pytest tests\test_image_derived_graph_phase_capture.py ^
  tests\test_image_derived_graph_attribution.py tests\test_image_derived_graph_stage_harness.py ^
  tests\test_image_derived_graph.py tests\test_plane_graph.py -q -p no:cacheprovider
# exit 0; 297 passed in 77.91 s
#   15 phase + 79 attribution + 35 stage-harness + 58 image-derived-graph + 110 plane_graph

.\.venv\Scripts\python.exe -m py_compile ^
  src\mc_photogrammetry\pipeline\image_derived_graph.py ^
  scripts\run_image_derived_graph_stage.py ^
  tests\test_image_derived_graph_phase_capture.py tests\test_image_derived_graph.py
# exit 0
```

The 297 passing tests include the 15 focused phase tests, the graph attribution and 3-frame
stage-harness suites that carry the accepted observer, the module snapshot field-set test, and
`tests/test_plane_graph.py` (the solver that defines the exact face equation).

Independent digest/binding check (in the focused suite, recomputed from the recorded payload,
not from the capture's own fields):

- `test_phase_capture_assignment_and_alias_digest_bind_same_identity` rebuilds
  `failed_assignment_sha256` with `hashlib` from
  `{domain: "failed_best_assignment", schema: FAILED_ACTIVE_EDGE_ALIASES_SCHEMA, payload:
  [[frame, quadrant], ...]}` and asserts it equals both the phase `source_sha256` and the
  independent alias record's value, and that a changed assignment differs.
- `test_phase_capture_digest_replay_refuses_tampering` validates the serialized record through
  the public replay, then refuses an unsigned tamper and a re-sealed semantic tamper (axis
  solver metric and phase/phasor contradictions).
- `test_runner_binds_phase_capture_to_projected_alias_record` drives the runner's
  `_validate_solver_failure_observation` with the projected alias trace, accepts the bound
  record, and refuses a re-sealed record whose phase `source_sha256` no longer matches the
  projected alias digest.
- `test_phase_capture_replay_refuses_partner_endpoint_side_tamper` asserts the mirror tuples
  directly and refuses a re-sealed single-row partner-side omission/flip.

## 6. Hashes (SHA-256, post-repair)

| Path | SHA-256 |
|---|---|
| `src/mc_photogrammetry/pipeline/image_derived_graph.py` | `169F5B4437E7C59A4062E0E5364F627DC32676C074D1B3450B7B4A0AEE952754` |
| `tests/test_image_derived_graph_phase_capture.py` | `26D6FCC88F7A4945AE636A2235E8B1131F1D14DD3C20747D1FE83C630817F008` |
| `scripts/run_image_derived_graph_stage.py` | `81E0691A951CB1992F560491A046D23F211B005CEC4129484901143B5FF05B3D` (unchanged) |
| `tests/test_image_derived_graph.py` | `58E3E46C1B63D96A9D7B1202A90ECEB7E85BD80028E47862F946CB58291F6B83` (unchanged) |
| `src/mc_photogrammetry/pipeline/image_derived_graph_attribution.py` | `BB4862C0E0D0F15ECAAB633977A1E1D93BBF306A0DE72379210DC933318A9A5A` (unchanged) |
| `tests/test_image_derived_graph_attribution.py` | `3D44C5E03968B365F280D38F67CE8C5AFC186059EAD1255EF5B4A51FD8A7FF45` (unchanged) |
| `tests/test_image_derived_graph_stage_harness.py` | `5042E0859C30282308BE409B48BB60F9990F7E65A674CA9E6950D0C7233EB5B8` (unchanged) |

Task file: `agent swarm instantiation template/tasks/active/ds-img-geom-phase-capture-r2-2026-10-04.md`.

## 7. Limitations

- No pinned PNG, image, reconstruction, truth, camera, bounds, mask or reference input was read
  or run; this report claims no per-axis real-run value.
- The complete-capture and fractional evidence are synthetic solver failures, not the real
  fixture; they exercise the same public `failed_phase_capture`/`SolverPhaseCapture` seam the
  observer uses and the real solver's own returned metrics.
- The endpoint-pair binding is structural: it proves the two captured rows are reversed mirrors
  and satisfy the relative-quadrant relation, not that the mirror geometry is the true scene.
- The runner binding is exercised by construction of the projected failure trace; the real
  artifact replay remains for the separately authorized observation run.
- Phase metrics are solver-internal unitless circular quantities; no accuracy or scale claim.

## 8. Scope observed

Owned paths only: the named phase-observer region of `image_derived_graph.py`, the named phase
test, the named task file, the named report, and an `AGENT_COMMS.md` result entry. No production
gate, matcher, solver, threshold, CLI or persistence semantics change; no truth/camera/bounds/
mask/reference read; no install, background job, shell nesting, recursive deletion, junction/
reparse point, encoded command, Defender/system change, unrelated edit or commit; no debug print
or temporary probe file. The accepted 3-frame observer and its tests are preserved byte-for-byte.

Stop for lead review; no pinned run or gate change is authorized by this task.
