# IMGLOSS-A1 route-separated image-only frontier - implementation report

Date: 2026-10-03
Task: `agent swarm instantiation template/tasks/active/image-only-frontier-a1-implementation-2026-10-03.md`
Design: `docs/reports/image-only-loss-attribution-design-2026-10-03.md` (accepted, §5 Tranche A1)
Status: Astra HOLD items addressed; focused tests pass (54); Route-A frozen
normal-mode CLI parity verified exactly; Route B was not rerun.
**Stopped for lead/Astra review. A1 is not marked complete.**

This is frontier/status measurement only. **No image-only accuracy, percentage,
geometry score or exact-state score is claimed anywhere in this report.** A2 owns
every visible denominator.

## 1. Owned files and SHA-256

| Path | SHA-256 |
| --- | --- |
| `src/mc_photogrammetry/evaluation_image_only_frontier.py` | `4595124df8fb450c3a4b788ba6bdfc639a3e521c50d21150efa72273254d0d4c` |
| `scripts/evaluation/run_image_derived_frontier.py` | `92c89f1722517f5cb9c8167cdd358d40185249a60911d3f3e9e5c95754c52059` |
| `scripts/evaluation/run_image_only_route_a.py` | `eb3edaa19e7e166ac7eac1f1a626082c6e3f17b4fa369045435f602132911d4b` |
| `tests/test_evaluation_image_only_frontier.py` | `82d3c2a7755ae2767a0bea8ea7e9e83f0e97c4dfb66852d71d0720b198614c50` |

No file outside these owned paths (and this report) was edited. The pre-existing
unrelated dirty hunks were preserved; `git diff --check` over the four code/test
paths exits 0. Only A1 temporary root artifacts were removed after the processes
stopped (see §7).

## 2. Lead/Astra corrections and how each was resolved

1. **Do not redesign `FrontierCheckpoint`; `reachable` still means the record was
   carried.** No `carried`/`succeeded` fields and no general validation machinery
   were added. `FrontierCheckpoint` is unchanged except that the type-level
   transition arithmetic was removed (item 7).
2. **Terminal and first missing checkpoint are separate concepts.** Route B keeps
   `terminal_policy` (the run's accepted record) plus the existing first-missing
   checkpoint (`ImageOnlyFrontier.frontier_stop`, first row with
   `reachable=False`). A failed geometry record is carried, so `geometry` can be
   the terminal while `output` is the first unattempted checkpoint.
3. **Route B with a supplied result.** `_route_b_result_terminal` sets
   `terminal_policy` `stage`/`status`/`reason` (and `detail`) from the result,
   requires the result stage to be a route-B stage *and* the last carried record,
   and requires any published `{stage}_sha256` to equal that record's replayable
   digest. The frontier's `terminal_status` is the result status verbatim, which
   is checked against the existing production vocabulary
   (`_ROUTE_B_PRODUCTION_STATUSES`, derived from `_ROUTE_B_TERMINAL_STATUS`); the
   stage-local -> production mapping is used only when no result is supplied.
   `frontier_stop` remains the first unattempted stage.
4. **`first_observable_failure` now reports the terminal failure and, separately,
   the first unattempted checkpoint.** It returns `terminal_stage`,
   `terminal_status`, `terminal_reason`, `first_unattempted_checkpoint`
   (or `null`), `evidence_digest` and `metric: null`.
5. **The old test that expected a geometry result to refuse was corrected**, and
   the exact `geometry`/`geometry_unavailable`/`g_unavailable` + `output=None`
   case was added (see §5, `test_route_b_failed_geometry_is_the_terminal_while_output_is_unattempted`).
   A genuine refusal remains for a result whose terminal stage is not the last
   carried record, is not a route-B stage, or does not bind the record digest.
6. **Route A terminal selection uses the actual failed stage.** `route_a_frontier`
   takes the first unattempted checkpoint as its terminal (falling back to the
   planned `terminal` only when every stage ran), and the Route-A runner summary
   reports `frontier.terminal_policy["stage"]` instead of the last planned row.
7. **`export_round_trip_exact` is `null` unless export and decode actually ran.**
   The Route-A runner only compares the facade and decoded volumes when both
   exist; an earlier failure (for example a camera failure) no longer yields two
   empty maps and a false-positive `true`.
8. **`PlacementRefusal` handling no longer invents `entry`.** The runner reads and
   validates the persisted schema-3 envelope when one exists (reporting its
   `stage`/`status`/`reason`/`detail`), otherwise returns a typed
   `frontier_unavailable("placement_refusal", ...)` with a null terminal stage.
   This is tightened by item 11: an envelope is only attributed when it binds the
   returned refusal.
9. **Premature numeric transition validation and net claims removed.** The
   `FrontierTransition` non-negative/reconciliation/partition checks and the
   `net = previous + gained - lost` derivation are gone; A1 transition numeric
   fields stay `null`, and the test asserts that. A2 owns the arithmetic.
10. **`SimpleNamespace` records are explicitly labelled synthetic.** The recipe
    test is renamed and annotated so the real volume/proposal/promotion
    constructors are distinguished from the synthetic route-B envelope stand-ins;
    no claim of a real end-to-end production constructor remains, and no
    `object.__new__` production record is fabricated.
11. **PlacementRefusal attribution is bound to the returned refusal (Astra HOLD 1).**
    `scripts/evaluation/run_image_derived_frontier.py` attributes a
    `PlacementRefusal` to a persisted schema-3 envelope only when
    `envelope.status == "refused"`, `envelope.refusal_sha256` equals the returned
    `outcome.refusal_sha256`, and `envelope.reason` equals the returned
    `outcome.reason`. An absent envelope stays typed unavailable attribution
    (`terminal_stage`/`terminal_status`/`terminal_reason` all `null`, no stage
    invented); an envelope that exists but does not bind this refusal (mismatched
    digest, mismatched reason or a non-refusal status) fails closed with
    `FrontierRefusal` instead of mislabelling a stale or unrelated terminal as the
    refusal's stage. Five synthetic tests cover matching, mismatched digest,
    mismatched reason, non-refusal envelope and absent envelope; none starts
    production.
12. **A1 transitions reject every non-null numeric field (Astra HOLD 2).**
    `FrontierTransition.__post_init__` refuses any of `previous`, `gained`,
    `lost`, `net`, `geometry_lost`, `state_lost` when it is not `None`, including
    `0`. No arithmetic, reconciliation or partition checks were added: the fields
    stay reserved for A2 and a count can never be read as a measured zero. A
    parametrized test exercises all six fields with the values `0` and `1`.
13. **Route-A frozen normal-mode CLI parity completed (Astra HOLD 3).** The
    runner's default `--columns/--rows/--depth-layers` now mirror the normal-mode
    `reconstruct` defaults read from `cli.py:94-101` (32/32/8), so its default
    invocation is the normal user's. The actual CLI and the runner were each run
    once on `templates 2/2026-09-02_13.22.14.png` with temporary outputs; the
    decoded `.litematic` coordinate/state sets were exactly equal after
    normalizing only each exporter's region origin (see §4A).

## 3. Contract as implemented

* `FrontierCheckpoint.reachable` is the only count-like statement and means *the
  run carried this stage's record*. `metric` is always `null`; a missing record is
  typed unavailable and never a measured zero.
* Universes stay labelled and separate: `G` geometry, `O` ownership, `P` pair
  correspondence, `C` candidate/palette, `E` exact state; the camera row is a
  diagnostic with `NONE` universe and no transition.
* Route-A stage order: `admission` -> `camera` -> `facade_grid` -> `palette_label`
  -> `terminal`. Route-B stage order: `entry` -> `lattice` -> `geometry` ->
  `output`.
* `ImageOnlyFrontier.frontier_stop` is the first unattempted checkpoint;
  `terminal_policy` is the run's accepted terminal tuple.

### 3.1 Source-derived expected frontier (textual)

Route B: `entry` reachable; if `graph.status != "ok"` or
`pose_graph.status != "ok"`, the accepted result is `entry`/`geometry_unavailable`
with `entry` carried and `lattice` the first unattempted stage; else `lattice`
reachable and, if not `assembled`, the accepted result stops at `lattice`; else
`geometry` reachable and, on the recorded genuine runs, the accepted result stops
at `geometry`/`geometry_unavailable` with `reason = "g_<status>"` while `output`
is the first unattempted stage. This is a frontier, not a causal loss.

Route A: on a lattice-bearing image `admission` and `camera` are reachable and the
run proceeds to `facade_grid`, `palette_label`, `terminal`; the terminal is the
actual failed stage when a stage fails (for example `camera`/`unavailable` on a
camera solve failure). Unresolved metric scale is an A2 scoring caveat, never a
frontier `unavailable`.

## 4. Frozen Route B replay (exactly one run)

Command (frozen inputs already in the workspace; no JAR, mask, reference,
manifest or camera input):

```
.\.venv\Scripts\python.exe scripts\evaluation\run_image_derived_frontier.py ^
  --sources "templates 2\2026-09-02_13.22.14.png" ^
            "templates 2\2026-09-02_13.24.24.png" ^
            "templates 2\2026-09-02_13.24.33.png" ^
  --report .tmp-a1-route-b-replay.json
```

Exit 0. The runner used a private temporary project target and removed it. Exact
verified values:

| Field | Value |
| --- | --- |
| route / frontier_available | `route_b` / `true` |
| terminal_stage | `geometry` |
| terminal_status | `geometry_unavailable` |
| terminal_reason | `g_unavailable` |
| terminal_detail | `plane_integer_residual` |
| terminal_policy_agrees | `true` |
| reached stages | `entry`, `lattice`, `geometry` |
| `entry.record_state` | `ok` |
| `lattice.record_state` | `assembled` |
| `geometry.record_state` | `unavailable` (carried) |
| `output` | unreachable, `blocked=output_record_absent` |
| first_observable_failure.terminal_stage/status/reason | `geometry` / `geometry_unavailable` / `g_unavailable` |
| first_unattempted_checkpoint.stage | `output` |
| evidence_digest | `0b9680a168dddd24a56e05d5b7ccf832a9ef63c248c69a1070ce2a84c5be4f2d` |
| frontier_sha256 | `e1693e5b4d7201c7ff820ea78a6630b5ba184ff92aa85466c9215e8241abcfdf` |
| entry_sha256 | `e38546aa07e6decad3e3f09c4abe3963caed6c5b8e8e5630f1fc77fd26f270ea` |
| lattice_sha256 | `6f4cba955d1ad2fdc1a7a7d059589c09000752ce7edd6cc2ccb85d83448b3fd4` |
| geometry_sha256 | `0b9680a168dddd24a56e05d5b7ccf832a9ef63c248c69a1070ce2a84c5be4f2d` |
| output_sha256 | `` (empty) |
| session_sha256 | `3b6d15bf2a74b672d88d5a7a40f1bf5298d00ede770735309694d97770be9fb7` |
| result_sha256 | `205bdd20ce9574b6fd7b4c6a94454c69984eb6866f6bf41968eed2fb2fd022bb` |
| envelope_sha256 | `3fa86b7e13d65769a6c16ed6d94d522a57a890088a10d71e6c9d7195292f4c64` |
| transition | `lattice` -> `geometry`, universe `frozen_reference_cells`, all counts `null` |

This is exactly the case the corrections target: the geometry record is carried
and is the accepted terminal failure, the schema-3 envelope agrees, and `output`
is reported separately as the first unattempted checkpoint. No other Route B run,
probe or repeat was performed. This replay is from the prior correction round and
was **not** rerun for the HOLD items; only the runner's `_refusal_summary` path
(not used by a non-refusal Route B run) changed since.

## 4A. Frozen Route-A normal-mode CLI parity (each side run exactly once)

The normal-mode CLI invocation was read from source before running
(`cli.py:88-110`: `reconstruct` defaults to `--columns 32 --rows 32
--depth-layers 8 --mode normal`, and normal mode rejects
`--horizon/--camera-manifest/--scene` at `cli.py:498-514`). The runner defaults
were aligned to those exact normal-mode defaults. Both sides were run once on the
established single raster with temporary outputs; no manifest, camera, mask,
reference, coordinate, bound or JAR was supplied or read.

```
.\.venv\Scripts\python.exe -m mc_photogrammetry reconstruct ^
  "templates 2\2026-09-02_13.22.14.png" ^
  .tmp-a1-route-a-parity-20261003\cli\2026-09-02_13.22.14.litematic ^
  --project .tmp-a1-route-a-parity-20261003\cli\2026-09-02_13.22.14.mcscan

.\.venv\Scripts\python.exe scripts\evaluation\run_image_only_route_a.py ^
  --image "templates 2\2026-09-02_13.22.14.png" ^
  --workdir .tmp-a1-route-a-parity-20261003\runner ^
  --report .tmp-a1-route-a-parity-20261003\runner-report.json
```

CLI: exit 0; `Camera: image-only solve in a local frame`; `Created ... with 690
block(s)`; `Saved project ...`. Runner: exit 0, `camera_solved=true`,
`facade_block_count=690`, `decoded_block_count=690`,
`export_round_trip_exact=true`, terminal tuple `route_a` / `terminal` /
`available` / reason `""` (empty; no stage failed).

Decoded `.litematic` comparison, normalizing **only** the exporter region origin
(`coordinate - snapshot.position` per side; both exporters write `Position
(0,0,0)`):

| Field | Value |
| --- | --- |
| CLI decoded region origin | `(0, 0, 0)` |
| runner decoded region origin | `(0, 0, 0)` |
| CLI decoded blocks | 690 |
| runner decoded blocks | 690 |
| normalized coordinate/state sets equal | `true` |
| only-CLI coordinates | 0 |
| only-runner coordinates | 0 |
| state mismatches on common coordinates | 0 |
| CLI normalized volume SHA-256 | `81e3efa8f55f53b33e23db8f46f89b51f9f2b22accd4c865563fe6951ecec73e` |
| runner normalized volume SHA-256 | `81e3efa8f55f53b33e23db8f46f89b51f9f2b22accd4c865563fe6951ecec73e` |
| runner facade/decoded volume SHA-256 (summary) | `4bad9bbb68caed01054ad18ba9b456b2eb4c62a09eb4fd366c7f78265230ac37` |
| runner frontier_sha256 | `66ab7f835c0d8e6795724adcf7fc9199d9780ca15bf03fe82270bb84d6ec5084` |

The normal-mode CLI and the Route-A runner therefore produce identical decoded
coordinate/state sets on the frozen raster under region-origin normalization.
This is route/status/parity evidence only: it is not a visible-truth number, and
no image pixels were inspected manually.

## 5. Verification commands and results

```
.\.venv\Scripts\python.exe -m py_compile ^
  src\mc_photogrammetry\evaluation_image_only_frontier.py ^
  scripts\evaluation\run_image_derived_frontier.py ^
  scripts\evaluation\run_image_only_route_a.py ^
  tests\test_evaluation_image_only_frontier.py
# exit 0

$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m pytest tests\test_evaluation_image_only_frontier.py ^
  -o addopts='' -q --no-header
# 54 passed in 0.43s   (this HOLD-correction round; 36 passed previously)

.\.venv\Scripts\python.exe -m pytest tests\test_image_derived_output.py ^
  tests\test_image_derived_orchestration.py tests\test_plane_volume.py ^
  tests\test_state_promotion.py tests\test_evaluation_pair_correspondence.py ^
  -o addopts='' -q --no-header
# 236 passed in 345.56s (design §5 A1 check 9 regressions; prior correction round)
```

The design §5 A1 check-9 regression set passed unchanged in the previous
correction round. This HOLD round changed only the pure A1 module (transition
null enforcement), the two evaluation-only runners and the A1 test module, none
of which that regression set imports; Route B was not rerun.

Focused tests added/changed in the previous correction round include:

* `test_route_b_failed_geometry_is_the_terminal_while_output_is_unattempted`
  (the exact `geometry`/`geometry_unavailable`/`g_unavailable` + `output=None`
  case, both the terminal tuple and `first_unattempted_checkpoint`).
* `test_route_b_result_terminal_stage_must_be_the_last_carried_record`
  (absent stage, non-route-B stage, and unbound terminal digest all refuse).
* `test_route_b_terminal_policy_comes_from_the_accepted_invocation_record`.
* `test_route_a_camera_solve_failure_is_a_typed_unavailable_camera` and
  `test_route_a_complete_route_terminates_at_the_planned_terminal` (actual
  failed-stage terminal selection).
* `test_route_a_runner_summary_uses_the_actual_terminal_and_nulls_no_export`
  (synthetic `RouteARun`; `export_round_trip_exact` is `true` only when export and
  decode ran, otherwise `null`).
* `test_tampered_reachability_and_transition_binding_are_refused` (numeric
  transition fields asserted `null`; arithmetic validation removed).
* `test_synthetic_recipe_tail_quotes_real_volume_records_beside_synthetic_envelope`
  (real volume/proposal/promotion constructors, explicitly synthetic route-B
  envelope).

Focused tests added for the Astra HOLD round:

* `test_route_b_placement_refusal_attributes_its_bound_refusal_envelope`
  (refused status + matching `refusal_sha256` + matching reason attribute the
  envelope's stage/status/reason/detail/envelope digest).
* `test_route_b_placement_refusal_mismatched_digest_fails_closed` and
  `test_route_b_placement_refusal_mismatched_reason_fails_closed` (an unbound
  envelope raises `FrontierRefusal`; no stage is mislabelled).
* `test_route_b_placement_refusal_non_refusal_envelope_fails_closed`.
* `test_route_b_placement_refusal_without_envelope_is_typed_unavailable`
  (absent envelope → `null` terminal stage/status/reason, no envelope read).
* `test_a1_transition_refuses_any_non_null_numeric_field` (all six numeric
  fields, values `0` and `1`).
* `test_route_a_runner_defaults_mirror_the_normal_cli_parameters` (32/32/8).

None of these starts production: the refusal tests monkeypatch the runner's
envelope reader and build envelopes in memory through the real record codec.

Route A frozen-input parity (design §5 A1 check 5) is now **completed** in §4A by
running the actual normal-mode CLI and the Route-A runner once each on the
established frozen raster; the decoded coordinate/state sets match exactly under
region-origin normalization. Route B was not rerun, per the HOLD instruction.

## 6. Non-claims

* No visible-truth numerator/denominator, exact-state recovery, geometry
  percentage, completion/refusal accuracy or "image-only accuracy" is reported.
* The Route B stop at `geometry` is a frontier/status fact only, not a causal
  attribution of loss.
* No model was fitted or trained; no dataset was exported; no gate, threshold,
  palette, schema, signature or product behavior changed.
* The synthetic tests are synthetic; the real replays here (one Route B status
  replay and the Route-A CLI/runner parity pair) are single frozen-input
  status/parity observations, not benchmarks.

## 7. Temporary artifacts

The previous round removed only its own root-level temporary artifacts (probe
scripts, the earlier Route-A/Route-B reports and project, the comms-tail scratch
file and the replay report). This HOLD round created one temporary root,
`.tmp-a1-route-a-parity-20261003/` (the CLI `.mcscan`/`.litematic`, the runner
workdir and its report) and removed it after the parity comparison completed. No
unrelated workspace file, fixture, image or user data was touched.
