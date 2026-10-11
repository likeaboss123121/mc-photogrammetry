# DS-IMG-GEOM-DIAG-RUN - evaluation-only 3-frame geometry-failure observer

Task ID: `DS-IMG-GEOM-DIAG-RUN`. Owner: DeepSeek (bounded implementation + one run/report).
Run date: 2026-10-04. Status: **one authorized observation executed; stopped for lead/Luna review.**

This report records the evaluation-only observer implementation, its focused tests, the
preflight hashes, exactly one foreground three-frame Template 2 re-observation, and the
per-axis geometry-failure evidence it captured. It is a diagnostic re-observation of the
pinned `plane_integer_residual` stop, not a promotion run, threshold change, tuning sweep,
or accuracy estimate.

## 1. Result summary

- The pinned three-frame Template 2 stop **reproduces exactly**: terminal
  `plane_integer_residual`, graph status `unavailable`, `component solver abstained`,
  exit `0`, harness wall time `176.875 s` (outer shell `177.687 s`).
- The first failed geometry predicate is **`minimum_resultant`**
  (`predicate_basis=derived_from_returned_solution`, `residual_predicate_evaluated=false`).
  All three matched axes are below the frozen `min_resultant=0.90`; axis 1 also exceeds the
  frozen `integer_tolerance=0.15`, but the resultant predicate is reported first and the
  residual predicate is not evaluated in this projection.
- Axis coverage is complete (`matched_axes=[0,1,2]`), so the stop is **not** an
  axis-coverage/partial-axis refusal; it is a per-axis resultant-quality refusal. No gate
  was lowered, inferred or tuned.
- The raw in-memory `SolverFailureSnapshot` was captured through a new optional
  evaluation-only observer and bound to the input/entry/lattice/signature/graph digests.
- The session digest `3b6d15bf...9fb7` and lattice digest `6f4cba95...3fd4` are
  **identical** to the previously accepted one-run Route B stage-loss artifact, so this
  observer re-observed the same upstream entry/lattice state that stopped at geometry.
- No production threshold, matcher, geometry, truth, mask, camera manifest, CLI normal
  path or persisted project schema was changed. The production graph result and all
  existing digests are unchanged whether or not the observer is attached.

## 2. Per-axis captured values

Frozen settings: `max_shift_cells=32`, `integer_tolerance=0.15`, `min_resultant=0.90`,
`ambiguous_margin=0.02`. Frames `image-0000..image-0002`; returned quadrants `[0, 3, 1]`.

| Axis | Raw resultant | Below `0.90`? | Raw residual | Above `0.15`? | Raw observations |
|---|---|---|---|---|---|
| 0 | `0.7893573588959358` | yes | `0.09306463330502572` | no | 8 |
| 1 | `0.013942164981374504` | yes | `0.24710226548952122` | yes | 6 |
| 2 | `0.8769985576878125` | yes | `0.06725781508366747` | no | 6 |

Projected first-failed record (`trace.solver_failure`, attributed to
`plane_graph.solve_component`):

- `reason=plane_integer_residual`
- `first_failed_predicate=minimum_resultant`
- `resultant_below_min_axes=[0,1,2]`
- `residual_above_tolerance_axes=[1]`
- `residual_predicate_evaluated=false`
- `matched_axes=[0,1,2]`, `predicate_basis=derived_from_returned_solution`

Additional bound failure evidence carried by the same snapshot:

- `input_match_count=38`, `input_match_identity_sha256=28ad4ac48021c727c4b6e7bf283e9965514dc3016c3c16919a18367cf0868105`
- `failed_assignment_sha256=ab4cb09bf5e8fe00e0bca11ca6cf2732cf24fd086d6f717919369f4191525dcd`
- `active_edge_aliases`: complete, 10 edges, axis edge counts `[4,3,3]`, 4 conflicting
  endpoints, `record_sha256=77660882c2a34a046b33e85610893ab0285a26dd8ffe5b17fe1628af80dd2f9c`
- `boundary_saturation`: **refused** atomically (`status=refused`, `reason=budget_exhausted`,
  `field=admitted`, `record_sha256=aa2f7083edb62f7df72d89528efa3d3e5fa098ae4c7a4450164b91b113395778`).
  The DS-R5 boundary-saturation count is therefore unavailable for this run; it did not
  affect the graph result.

## 3. Implementation (evaluation-only, named files only)

- `scripts/run_image_derived_graph_stage.py` (evaluation harness): accepts exactly three or
  four frozen PNGs (`ALLOWED_FRAME_COUNTS=(3,4)`), preserving the four-frame contract; adds
  the optional observer plumbing on the graph call and a digest-bound
  `solver_failure_observation` record; `_validate_completed_artifact` now replays and joins
  that record (schema, self-digest, projected record, input/entry/lattice/signature/graph
  binding) and refuses a tampered binding. No production threshold literal is present in
  the harness (existing source guard still passes).
- `src/mc_photogrammetry/pipeline/image_derived_graph_attribution.py` (evaluation
  attribution seam): adds the optional `solver_failure_observer` callback and the public
  `solver_failure_snapshot_record()` unprojected serializer. The callback receives the exact
  in-memory `SolverFailureSnapshot` from the same graph seam; it is never forwarded into the
  graph derivation, so the result/attribution/digests are unchanged. Partial-axis maps and
  non-finite or non-JSON values are retained as bounded typed tokens, never dropped or
  clamped to zero. This module is the attribution seam used by the harness; the default
  production path (`derive_image_derived_graph`, `image_derived_geometry_orchestration`) is
  untouched.
- Tests: `tests/test_image_derived_graph_stage_harness.py` and
  `tests/test_image_derived_graph_attribution.py` gained focused coverage for 3-frame
  accepted input, 4-frame parity, unpinned frame-count refusal, 3-frame pin/output
  collision fail-closed, partial-axis raw capture despite projection refusal, observer
  output parity, non-finite tagging, digest-bound observation replay, and a source guard
  that neither the runner nor the seam references truth/camera/bounds/manual metadata.

## 4. Focused tests (run before the observation)

```
.\.venv\Scripts\python.exe -m pytest tests/test_image_derived_graph_attribution.py tests/test_image_derived_graph_stage_harness.py -q
# 114 passed

.\.venv\Scripts\python.exe -m pytest tests/test_plane_graph.py tests/test_synthetic_targeted_capture_preflight.py tests/test_targeted_capture_quality.py -q
# 126 passed (adjacent consumers of the attribution seam)
```

## 5. Preflight (before the run)

- Screenshot display: none. Only SHA-256 content comparison.
- The three pinned `templates 2` screenshots hash exactly to the reviewed `template-2`
  pins in `docs/reports/evaluator-fixture-pins.json` (pin-file read for preflight only).
- Reference pin value (not opened): `519034a6a439ab98145aa820792d5afc588b5d697ecad961c14c36cbef2f3416`.
- The diagnostic root `.tmp-geom-diag-observer-20261004` was absent before this task and
  every run target (`manifest.json`, `stage.mcscan`, `observation.json`, `heartbeat.json`,
  `terminal.json`, `stdout.txt`, `stderr.txt`) was absent. `.tmp-routeb-stage-loss-20261004`
  was never a target.
- Preflight record: `.tmp-geom-diag-observer-20261004.preflight.json`.

| Input / code | SHA-256 |
|---|---|
| `templates 2/2026-09-02_13.22.14.png` | `eb22541171e030ff94d6839a6626e7160c69ae3a780f07aec0298322be665d46` |
| `templates 2/2026-09-02_13.24.24.png` | `e2fd9fd6c83bf9005fe0973987da9e46d213a4351ec91393ce578f8e6d3961ae` |
| `templates 2/2026-09-02_13.24.33.png` | `35d9debb206c2505f65d3be29183f4a4f3d2bf3f9c05644ce75570f48381d1fc` |
| `scripts/run_image_derived_graph_stage.py` | `d9edcc5a4191f826388d8a1c3900921625468b293b7cf435fd4e25470c14be83` |
| `src/mc_photogrammetry/pipeline/image_derived_graph_attribution.py` | `bb4862c0e0d0f15ecaab633977a1e1d93bbf306a0de72379210dc933318a9a5a` |
| `tests/test_image_derived_graph_stage_harness.py` | `5042e0859c30282308be409b48bb60f9990f7e65a674ca9e6950d0c7233eb5b8` |
| `tests/test_image_derived_graph_attribution.py` | `3d44c5e03968b365f280d38f67ce8c5afc186059ead1255ef5b4a51fd8a7ff45` |

Post-run re-hash of all four code files and all three screenshots is unchanged. The run
manifest binds the same runner/module hashes and the same three input hashes.

## 6. Exact command and settings

Executed once, foreground, from the repository root, under the reviewed harness containment
(suspended child, 8-GiB Job object, 600 s internal deadline):

```
.\.venv\Scripts\python.exe scripts\run_image_derived_graph_stage.py --contained ^
  --terminal .tmp-geom-diag-observer-20261004\terminal.json ^
  --manifest .tmp-geom-diag-observer-20261004\manifest.json ^
  --stdout   .tmp-geom-diag-observer-20261004\stdout.txt ^
  --stderr   .tmp-geom-diag-observer-20261004\stderr.txt ^
  --png "templates 2\2026-09-02_13.22.14.png" --sha256 eb22541171e030ff94d6839a6626e7160c69ae3a780f07aec0298322be665d46 ^
  --png "templates 2\2026-09-02_13.24.24.png" --sha256 e2fd9fd6c83bf9005fe0973987da9e46d213a4351ec91393ce578f8e6d3961ae ^
  --png "templates 2\2026-09-02_13.24.33.png" --sha256 35d9debb206c2505f65d3be29183f4a4f3d2bf3f9c05644ce75570f48381d1fc ^
  --project  .tmp-geom-diag-observer-20261004\stage.mcscan ^
  --output   .tmp-geom-diag-observer-20261004\observation.json ^
  --telemetry .tmp-geom-diag-observer-20261004\heartbeat.json
```

- Settings are the frozen production defaults `32 / 0.15 / 0.90 / 0.02`; the harness never
  passes them and a source guard test asserts that. No threshold, palette, radius, matcher
  or solver override.
- Inputs are only the three pinned PNGs. No truth, reference, camera, direction,
  coordinate, bounds, mask or evaluator fixture data entered the run; the only manifest is
  the harness's own input/code binding (three PNG paths/hashes plus nine producer source
  hashes), which carries no truth or camera content.
- One run, no retry, no sweep, no second route, no launcher/Minecraft action.
- Timing: outer shell `177.687 s`; harness `elapsed_seconds=176.875`; exit code `0`;
  `stderr.txt` and `stdout.txt` both empty (`0 B`). Stage timings from the heartbeat:
  entry complete `30.750 s`, lattice complete `48.062 s`, signatures complete `74.000 s`,
  graph complete `176.406 s`; peak working set `254,717,952 B`.

## 7. Immutable run artifacts

| Artifact | Bytes | SHA-256 |
|---|---|---|
| `.tmp-geom-diag-observer-20261004/observation.json` | 7,909,163 | `074c9abba1c354b2c541b38a09dca5c63e6e4862a62b99b38a07a6a784a76df1` |
| `.tmp-geom-diag-observer-20261004/stage.mcscan` | 6,332 | `9d6beca406eafaf7310d7964932effbb1d3d357eeeeae9dde3608a34ce70a7b8` |
| `.tmp-geom-diag-observer-20261004/terminal.json` | 612 | `ac67e9b2eb0d6510867194617eaa5e789f3106bf4796fdc4d4e24e224e943335` |
| `.tmp-geom-diag-observer-20261004/heartbeat.json` | 1,674 | `efb41ec619bd8162d2cd31d97d9f399461bf52455146977dbd0fc65b62b21650` |
| `.tmp-geom-diag-observer-20261004/manifest.json` | 5,050 | `bbd71b25673d651731b7e2afce767cd6d5b84ba4a05684f6f61f009417b29f75` |
| `.tmp-geom-diag-observer-20261004/stdout.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `.tmp-geom-diag-observer-20261004/stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

Internal content addresses:

| Field | Value |
|---|---|
| manifest `manifest_sha256` | `978d1e6232aae22029080a180757d8e58b6ec7cd8b213651bfd18c23acc8c86c` |
| observation `artifact_sha256` | `c682306fd61ed5354cf9e8fe91a2ac71fe90c984662778bce586a4348fbc809f` |
| `solver_failure_observation.observation_sha256` | `43445d82411485dd902e5392a74b98b1229facdb467b833a6b215ad68ee19b20` |
| `graph_sha256` (graph result) | `0a64c5e73b62b376f41e337a756c61af880d72cee1a90992001a42d00d9f39cd` |
| `trace_sha256` (attribution) | `738c9dc09dc17cf64a74e91472cb59a45bbc9917de479cd4f33f80551becd736` |
| `entry_sha256` | `50e918aac6c06088ed73e496c88498aa81121f0f4095f5582d143249ccc4093e` |
| `project_sha256` (`stage.mcscan`) | `9d6beca406eafaf7310d7964932effbb1d3d357eeeeae9dde3608a34ce70a7b8` |
| trace source session | `3b6d15bf2a74b672d88d5a7a40f1bf5298d00ede770735309694d97770be9fb7` |
| trace source lattice | `6f4cba955d1ad2fdc1a7a7d059589c09000752ce7edd6cc2ccb85d83448b3fd4` |
| trace source signatures | `1f8ed48037ad85eaea7682a0c8c74cc3ed943f0810d23d4d02d701f7d93a5186` |

The observer binding block joins the raw record to the three input hashes, `entry_sha256`,
the lattice and signature digests, and `graph_sha256`; the parent process replayed and
validated that binding (a tampered binding is a typed refusal in the focused test).

`observation.json` top-level contract: `status=complete`, `execution_complete=true`,
`graph_assembled=false`, `geometry_available=false`, `solution_available=false`,
`signature_identities=85`, `future_context_estimate.allocated=false`. `stage.mcscan` is a
schema-2 checkpoint with three source paths and session digest `3b6d15bf...9fb7`.

## 8. Cross-run binding to the accepted Route B stop

| Digest | This observer run | Accepted Route B artifact |
|---|---|---|
| `session_sha256` | `3b6d15bf2a74b672d88d5a7a40f1bf5298d00ede770735309694d97770be9fb7` | `3b6d15bf2a74b672d88d5a7a40f1bf5298d00ede770735309694d97770be9fb7` |
| `lattice_sha256` | `6f4cba955d1ad2fdc1a7a7d059589c09000752ce7edd6cc2ccb85d83448b3fd4` | `6f4cba955d1ad2fdc1a7a7d059589c09000752ce7edd6cc2ccb85d83448b3fd4` |
| terminal detail | `plane_integer_residual` | `plane_integer_residual` |

Both accepted one-run artifacts (`.tmp-routeb-stage-loss-20261004/attribution.json`
`5cd4f15a...`, `route-b.mcscan` `fd508d9f...`, `stdout.txt` `95a98cc3...`, empty
`stderr.txt`) were re-hashed after this run and are byte-unchanged, so the new diagnostic
root never overwrote the prior route's evidence.

## 9. Verification checks performed

- Focused tests (section 4) passed before the observation.
- Preflight content hashes, pin comparison and absent-root/target checks (section 5).
- Exactly one foreground contained run; exit `0`; both capture files empty (section 6).
- The harness parent replayed the artifact: schema, `artifact_sha256`, project/session
  join, graph terminal/status flags, signature bindings, trace digest/provenance join, and
  the optional observation schema/self-digest/projected/binding join; exit `0` means all of
  those validated (a failure would have produced exit `126`
  `missing_or_invalid_artifact`).
- Post-run re-hash of the runner, module, both tests, and all three PNGs: unchanged.
- Route B artifact non-overwrite re-hash: unchanged.

## 10. Limitations

- One fixture, one three-frame run, one machine. No accuracy, alignment or metric-scale
  claim; the raw resultants are solver-internal unitless correlation quantities.
- The projected `first_failed_predicate=minimum_resultant` is the accepted projection's
  precedence, not an independent re-derivation: axis 1 also exceeds `integer_tolerance`
  (`0.2471 > 0.15`), and `residual_predicate_evaluated=false` because the resultants
  already fail. The raw record preserves both facts so the residual condition is visible
  even though the projection reports the resultant predicate first.
- The real run had complete axis coverage, so the projection did not refuse; the
  partial-axis raw-capture path is established by the focused tests
  (`SolverFailureSnapshot` with axes `{0,1}` under `plane_integer_residual` yields a
  refused projection while the raw record retains both axes), not by this run.
- The DS-R5 `boundary_saturation` attestation is unavailable for this fixture
  (`budget_exhausted` on `admitted`); no boundary count is claimed.
- The harness child prints nothing, so the separate stdout/stderr captures are both empty;
  the evidence is the `observation.json` artifact, not console output.
- The three-image path reconstructs the session/lattice directly from the pinned pixels
  rather than replaying the persisted Route B `.mcscan` (the `.mcscan` stores shape/hash
  metadata, not pixels or lattice objects). The identical session and lattice digests are
  the binding evidence; this is a re-observation, not a replay of retained in-memory state.
- No production gate was exercised beyond its frozen default; this report does not
  authorize a gate change or a second run.

## 11. Scope and prohibitions observed

No production threshold, matcher, geometry, truth, mask, camera manifest, CLI normal path
or persisted project schema was changed; no production source or persistence file was
edited. Edits are limited to the two named evaluation files, the two named test files, this
new report, and the AGENT_COMMS.md task/result entry. No second run or route, no
launcher/Minecraft action, no install/download, no background process, no shell nesting, no
recursive deletion, no junction/reparse operation, no encoded/obfuscated command, no
generated executable, no Defender/system change, no credential access, and no commit.

Stop for lead/Luna review; no further run is authorized by this task.
