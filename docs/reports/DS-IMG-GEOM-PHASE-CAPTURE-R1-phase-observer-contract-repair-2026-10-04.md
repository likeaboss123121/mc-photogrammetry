# DS-IMG-GEOM-PHASE-CAPTURE-R1 - evaluation-only phase-observer contract repair

Task ID: `DS-IMG-GEOM-PHASE-CAPTURE-R1`. Owner: DeepSeek (bounded repair + focused tests).
Date: 2026-10-04. Status: **repair complete; focused tests and `py_compile` pass; stopped for
lead review. No pinned PNG run in this task.**

This report records exactly the five lead-HOLD payload-schema/binding repairs to the
evaluation-only per-face phase capture plus the dedicated tests the HOLD lists. It is not a
run report, accuracy estimate, gate change, or authorization to observe the pinned images.
The first phase pass was interrupted before tests/route and is not treated as accepted.

## 1. Result summary

- All five HOLD findings are repaired in
  `src/mc_photogrammetry/pipeline/image_derived_graph.py`; the alias record's
  `failed_assignment_sha256` value is byte-identical to the previously accepted value.
- `scripts/run_image_derived_graph_stage.py` now reads the active-edge alias record from the
  projected failure trace that carries it (the raw snapshot never did), so the phase/alias
  binding is genuinely validated rather than silently skipped.
- A new dedicated suite `tests/test_image_derived_graph_phase_capture.py` (10 tests) covers a
  complete failure payload, every face/axis field and index equation, all axis counts,
  assignment/alias binding, digest replay and resealed semantic-tamper refusal, gauge
  invariance, typed refusal on mismatch and partial/malformed data, runner binding, and
  default result/digest parity.
- The accepted 3-frame observer is preserved byte-for-byte:
  `image_derived_graph_attribution.py`, `tests/test_image_derived_graph_attribution.py` and
  `tests/test_image_derived_graph_stage_harness.py` are unchanged (hashes in section 5).
- No pinned PNG, reconstruction, image, truth/camera/bounds/mask/reference input, gate,
  matcher, solver setting, install, background job or commit was touched.

## 2. The five repairs

The phase record's face row is now the 20-column schema
`[match_index, left_frame, left_plane, left_axis, left_b, left_distance, left_sign,
right_frame, right_plane, right_axis, right_b, right_distance, right_sign, quadrant,
face_scalar, own_quadrant, other_quadrant, face_phase, phasor_real, phasor_imag]`. The axis
row is `[axis, solver_resultant, solver_residual, reconstructed_resultant,
reconstructed_residual, observation_count, face_count]`.

1. **Partner frame index (finding 1).** `_validate_complete_phase_payload` mapped the
   per-frame quadrant from `row[8]` (the right *plane*) instead of `row[7]` (the right
   *frame*). The map now uses `row[7]`, so the frame coverage and assignment checks bind the
   real partner frame.
2. **Axis metric comparison (finding 2).** The replay compared the independent recomposition
   against `(row[2], row[3])` (solver residual, reconstructed resultant). It now compares the
   recomposition against the recorded reconstructed pair `(row[3], row[4])`, and additionally
   binds the recorded solver pair `(row[1], row[2])` to that same reconstructed pair at
   `PHASE_CAPTURE_COMPATIBILITY_DECIMALS`, so a resealed axis row cannot carry a solver metric
   that contradicts its own faces.
3. **Observation count (finding 3).** `observation_count` is the solver's own per-axis face
   observation count (`len(face_scalars)`), not `2 * len(scalars)`. The builder now stores
   `len(scalars)` in `row[5]` (and `row[6]`), and the validator requires `row[5] ==
   len(scalars)`, matching `_phase_capture_control` and
   `ComponentAlternative.axis_observations`.
4. **Shared failed-assignment digest (finding 4).** The phase `source_sha256` hashed the
   frame/quadrant assignment under `PHASE_CAPTURE_SCHEMA`, while the alias
   `failed_assignment_sha256` hashed the same assignment under
   `FAILED_ACTIVE_EDGE_ALIASES_SCHEMA`; the runner's equality check therefore compared two
   unlike domains. Both seams now bind through one `_failed_assignment_digest` helper using
   `FAILED_ASSIGNMENT_DIGEST_DOMAIN = "failed_best_assignment"` and
   `FAILED_ASSIGNMENT_DIGEST_SCHEMA = FAILED_ACTIVE_EDGE_ALIASES_SCHEMA`, so the two digests
   are equal by construction. The alias record's stored value is unchanged (same
   domain/schema/payload), so the accepted observer's digests stay byte-identical. The runner
   now takes the alias record from the projected `trace.solver_failure` (the only place that
   serializes it) instead of `raw`, which never carried it; otherwise the check could never
   run.
5. **Serialized modulo-1 phase and unit phasor (finding 5).** Each face row now carries the
   scalar's `face_phase` and the phasor `(phasor_real, phasor_imag)`. The builder computes
   both with shared `_face_modulo_phase`/`_unit_phasor` helpers, and the replay validator
   re-derives them from the scalar and refuses any contradiction, instead of computing them
   privately with no serialized binding.

**Replay-enabling repair inside the same schema/binding scope.** `validate_solver_phase_capture`
previously passed the whole record (including `record_sha256`) to
`_validate_phase_capture_payload`, whose exact-field-set check then rejected every serialized
record. It now strips `record_sha256` before payload validation, exactly like
`validate_failed_active_edge_aliases` and `validate_boundary_saturation_attestation`, so a
serialized complete or refused record can actually replay.

## 3. Dedicated tests

`tests/test_image_derived_graph_phase_capture.py` (new, 10 tests) uses the existing exact
integer-lattice component fixture (`test_plane_graph.lattice_component(decoy=True)`) solved
with a raised `min_resultant` in the test call only, so the returned per-axis metrics are the
solver's own and the independent recomposition must reconcile with them.

| Required coverage | Test |
|---|---|
| complete failure payload | `test_phase_capture_complete_payload_is_bounded_and_replays` |
| every field/index equation (incl. phase/phasor) | `test_phase_capture_face_row_equations_bind_every_index` |
| all axis counts | `test_phase_capture_axis_rows_bind_solver_metrics_and_counts` |
| assignment/alias binding | `test_phase_capture_assignment_and_alias_digest_bind_same_identity`; `test_runner_binds_phase_capture_to_projected_alias_record` |
| digest replay | `test_phase_capture_digest_replay_refuses_tampering` |
| gauge invariance | `test_phase_capture_recomposition_is_gauge_invariant` |
| typed refusal on mismatch / partial data | `test_phase_capture_typed_refusal_on_metric_mismatch`; `test_phase_capture_typed_refusal_on_partial_data` |
| default result/digest parity | `test_phase_capture_default_result_and_digest_parity` |

The coverage asserts: 20-column rows and 7-column axis rows; both endpoints per match;
`row[6] == axis_transform(row[3], row[15]).sign`; `row[12] == axis_transform(row[9],
row[16]).sign`; both endpoints share one world axis; scalar `== translation + sign*distance`;
`row[17] == scalar % 1`; `row[18] == cos(2*pi*phase)`; `row[19] == sin(2*pi*phase)`; unit
magnitude; `row[1], row[2]` equal to the solver resultants/residuals; `row[3], row[4]` equal
to the recomposition; `row[5] == row[6] ==` the per-axis face count; all three axes present
and equal to `alternatives[0].axis_observations`. Refusal tests assert the atomic
`{schema, status, reason, field, record_sha256}` refusal shape, no partial face/axis rows, and
no substituted zero source digest.

The default-parity test spies `failed_phase_capture` and proves the default
`derive_image_derived_graph` path never allocates it (zero calls), that the attribution
observer gets exactly one in-memory snapshot with a `SolverPhaseCapture`, and that the graph
result and `result_sha256` are identical with and without the observer.

The interrupted first pass added `phase_capture` to `SolverFailureSnapshot` but left
`tests/test_image_derived_graph.py`'s exact snapshot field-set assertion stale, so that module's
own focused test failed with an extra `phase_capture` field. The assertion now includes
`phase_capture`, and the test additionally asserts the phase capture refuses atomically for its
injected foreign match. This is the only change to that file; no behavior changed.

## 4. Checks performed

Run from the repository root with the baseline interpreter; no screenshot display, no image
decode, no reconstruction.

```
.\.venv\Scripts\python.exe -m pytest tests\test_image_derived_graph_attribution.py ^
  tests\test_image_derived_graph_stage_harness.py ^
  tests\test_image_derived_graph_phase_capture.py -q -p no:cacheprovider
# exit 0; 124 passed (114 prior + 10 new)

.\.venv\Scripts\python.exe -m pytest tests\test_plane_graph.py ^
  tests\test_synthetic_targeted_capture_preflight.py ^
  tests\test_targeted_capture_quality.py -q -p no:cacheprovider
# exit 0; 126 passed

.\.venv\Scripts\python.exe -m pytest tests\test_image_derived_graph.py ^
  tests\test_image_derived_graph_policy.py tests\test_graph_endpoint_controls.py ^
  tests\test_graph_solver_ablation.py -q -p no:cacheprovider
# exit 0; 43 passed

.\.venv\Scripts\python.exe -m py_compile ^
  src\mc_photogrammetry\pipeline\image_derived_graph.py ^
  scripts\run_image_derived_graph_stage.py ^
  tests\test_image_derived_graph_phase_capture.py tests\test_image_derived_graph.py
# exit 0
```

The runner's own contained harness `ALLOWED_FRAME_COUNTS`, pin checks, manifest binding and
source guard tests are part of the 114 prior tests and remain green.

## 5. Hashes (SHA-256, post-repair)

| Path | SHA-256 |
|---|---|
| `src/mc_photogrammetry/pipeline/image_derived_graph.py` | `375E2528B0CDDBCD429712A3478015E4A87DCF772CF6424D943E69F63FB0D55B` |
| `scripts/run_image_derived_graph_stage.py` | `81E0691A951CB1992F560491A046D23F211B005CEC4129484901143B5FF05B3D` |
| `tests/test_image_derived_graph_phase_capture.py` | `922A4A36300E381A53243A84F3C76C95D6804F3C17D8E7FA3D0FD7520C9D599C` |
| `tests/test_image_derived_graph.py` (stale snapshot field-set assertion) | `58E3E46C1B63D96A9D7B1202A90ECEB7E85BD80028E47862F946CB58291F6B83` |
| `src/mc_photogrammetry/pipeline/image_derived_graph_attribution.py` | `BB4862C0E0D0F15ECAAB633977A1E1D93BBF306A0DE72379210DC933318A9A5A` (unchanged) |
| `tests/test_image_derived_graph_attribution.py` | `3D44C5E03968B365F280D38F67CE8C5AFC186059EAD1255EF5B4A51FD8A7FF45` (unchanged) |
| `tests/test_image_derived_graph_stage_harness.py` | `5042E0859C30282308BE409B48BB60F9990F7E65A674CA9E6950D0C7233EB5B8` (unchanged) |

Task file: `agent swarm instantiation template/tasks/active/ds-img-geom-phase-capture-r1-2026-10-04.md`.

## 6. Residual observation for the lead (not changed; outside the five)

The captured face scalar is `translation + axis_sign * distance`, while the solver's own face
value is `translation + axis_sign * side * distance` (`plane_graph._evaluate_assignment`), and
the row still does not serialize the endpoint `side` that the task's required output lists.
For an endpoint with `side = -1` and a non-integer distance the two modulo-1 phases differ by
`2 * distance`, so `_phase_capture_control` would return the typed
`phase_recomposition_mismatch` refusal on real (fractional-distance) readings instead of a
complete record. This fails closed (no wrong data and no substituted zero), but the observer
would likely emit no complete payload on the pinned run. The focused parity tests use the
integer-distance lattice fixture, where the offset is an even integer and the modulo-1 phase is
unchanged. This was deliberately not fixed because the HOLD authorized exactly five changes.
The lead should decide whether to authorize a follow-up that serializes `side` and includes it
in the scalar before the pinned observation run.

## 7. Limitations

- No pinned PNG, image, reconstruction, truth, camera, bounds, mask or reference input was
  read or run; this report claims no per-axis real-run value.
- The complete-capture evidence is a synthetic integer-lattice failure solve, not the real
  fixture; it exercises the same public `failed_phase_capture`/`SolverPhaseCapture` seam the
  observer uses.
- The runner binding check is exercised by construction of the projected failure trace; the
  real artifact replay remains for the authorized observation run.
- The phase metrics are solver-internal unitless circular quantities; no accuracy or
  scale claim is made.

## 8. Scope observed

Owned paths only: the named phase-observer region of `image_derived_graph.py`, the named
runner, the new named phase test, the named report, the named task file, and an
`AGENT_COMMS.md` result entry. No production gate, matcher, solver, threshold, CLI or
persistence semantics change; no truth/camera/bounds/mask/reference read; no install,
background job, shell nesting, recursive deletion, junction/reparse point, encoded command,
Defender/system change, unrelated edit or commit. The accepted 3-frame observer and its tests
are preserved byte-for-byte.

Stop for lead review; no pinned run or gate change is authorized by this task.
