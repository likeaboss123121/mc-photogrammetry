# Plane residual observer: single graph-stage invocation captured

Date: 2026-10-07. Task: `plane-residual-observer-2026-10-07`, completion-plan feasibility attempt 8/8.
Owner: DeepSeek CLI worker (developer-equivalent, direct-shell Defender-safe profile). Evaluation-only geometry diagnostic; no promotion, C2, or delivery.

## Result

Exactly one corrected foreground graph-stage observer invocation was run from the repository root, with process-local `PYTHONPATH=src` and no persistent environment or `.pth` change. It completed and produced every runner output.

- Runner exit code: `0`.
- Runner terminal: `status=complete`, `terminal=true`, `execution_complete=true`, `last_started_stage=graph`, `last_completed_stage=graph`, `geometry_available=false`, `acceptance=false`, `refusal=null`, `reason=""`, `cleanup.job_empty=true`.
- Wall time: `180.78100000019185` s against a 600 s deadline.
- Peak recorded RSS: `239579136` bytes (`terminal.rss_bytes`; heartbeat `graph complete` `rss_bytes=239579136`) against the 8 GiB aggregate Job Object limit (`8589934592` bytes). Limit not approached.
- `stdout.txt` and `stderr.txt` are both 0 bytes; all stage output is in the diagnostic records.
- No runner-level typed refusal was recorded (`terminal.refusal` is null).
- The solver abstained with typed reason `plane_integer_residual`; graph terminal `geometry_unavailable`; `component_solution=null`, `matches=[]`.

Output directory `exports/diagnostics/plane-residual-observer-2026-10-07/` after the run (SHA256 of each file):

| File | Bytes | SHA256 |
| --- | --- | --- |
| `preflight.json` | 5823 | `53201cbb411c593f5f2c49f89e6a4f9ecfbbbe17173903dd7c8ee8db65851c125` |
| `project.mcscan` | 9337 | `e471dc2cfa37c9db4c5d0944129526de4f3de7b4ad34952443a5d5649f114263` |
| `manifest.json` | 6690 | `3db4ba7bf2f8d72c97295046dc113b74c115daa5c4d2e2d5285bc8ae87415c99` |
| `observation.json` | 5881493 | `5b34aac34bc7c889c48248ccbf52460382519c9d176ae13a77ae6c33076ab1a2` |
| `heartbeat.json` | 1719 | `02c63315dc95a69684e3546855f7a64c3ceec49ea17fd0967dd706a1f54286ef` |
| `terminal.json` | 623 | `7e0cf4c191556cf7d7ea03ee5878298121835f7f67f82f9ae178dce8f044e6a9` |
| `stdout.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

Note on digest fields: `manifest.json` and `terminal.json` both carry the manifest self-digest `01269c0f3642286dd377428696ea7cc9463ede3da4335f68ab7d973b381a3cb5`, which is the SHA256 of the manifest payload excluding that field; the on-disk file hash differs because the field itself is serialized. The artifact self-digest `2bce2af0a0a7b1bd14be427ec0af4a4ebdca47cb0eb889583ab8747d979d2c53` (`observation.json:artifact_sha256`) likewise replays. Both were re-derived and match.

## Preflight gate

- All four pinned train-split PNGs `04`-`07` present; SHA256 re-verified equal to the task pins immediately before launch (04 `8797d70e...60a81f`, 05 `6f844d5e...f0f016b`, 06 `53c38526...c0ccf8923`, 07 `fdd8a370...1f025556`). Manifest binds the same four hashes.
- All seven runner target paths were absent before launch; only the pre-existing `preflight.json` was present. Fresh-output enforcement passed.
- No other images or data were supplied. No camera/F3/directions/bounds entered; no capture, JAR, checkpoint, truth, mask, sidecar, manifest, or litematic produced; no source edits.

## Solver inputs and thresholds

Four frames `image-0000`..`image-0003` (1280x720, 1600x900, 1600x900, 1280x720). 18 canonical readings; `reading_indices` 0..17. Post-admission candidates 6756 -> post-same-axis solver inputs 70; 16 active edges.

Exact effective `solve_component` arguments (`raw.solve_context`, schema `mc-photogrammetry-solver-solve-context-v1`, `record_sha256=f569744c97308f3f878612a2763187f7e1c071a962e3e9025b047760db962475`, status complete):

- `integer_tolerance = 0.15`
- `min_resultant = 0.90`
- `ambiguous_margin = 0.02`
- `quadrant_constraints` / `constraints = []` (empty)
- `frames = ["image-0000","image-0001","image-0002","image-0003"]`

Matcher attribution settings (`trace.effective_settings`, algorithm `separate-mask-halo-v2`): `max_shift_cells=32`, `min_overlap_cells=9`, `min_structural_share=0.15`, plus the three solver thresholds above.

## Quadrant assignment, alternatives, per-axis state

Selected/best assignment (`raw.quadrants`, `best_alternative_quadrants`, `phase_capture.assignments`) — all identical: `image-0000=0, image-0001=3, image-0002=3, image-0003=1`.

Best-alternative axis observation counts (`raw.best_alternative_axis_observations`): `[8, 12, 12]` for axes 0/1/2.

Per-axis projection (`solver_failure_observation.projected.axes`, basis `derived_from_returned_solution`; world axis, resultant, residual, observation count):

| Axis | Resultant | Residual | Obs | min_resultant 0.90 | residual <= 0.15 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.2197028820687929 | 0.21413319374511344 | 8 | FAIL (below) | would FAIL (above) |
| 1 | 0.6154253827952421 | 0.12243923753714483 | 12 | FAIL (below) | within |
| 2 | 0.5567875711217574 | 0.1298944685360216 | 12 | FAIL (below) | within |

- `matched_axes = [0, 1, 2]`: axis coverage sufficient.
- `resultant_below_min_axes = [0, 1, 2]`: all three axes fail `minimum_resultant`.
- `residual_above_tolerance_axes = [0]`; `residual_predicate_evaluated = false`.
- `first_failed_predicate = minimum_resultant`; `predicate_basis = derived_from_returned_solution`; `reason = plane_integer_residual`; `solver_id = plane_graph.solve_component`.

## Active edge aliases (16)

`solver_failure_observation.projected.active_edge_aliases`, schema `mc-photogrammetry-failed-active-edge-aliases-v1`, `record_sha256=6fc2d946f7bcac37819349be62dab01f8f5cef7cc02620e75b8d6093bf3c3a3e`, basis `derived_from_returned_best_assignment`, status complete. Row layout: `[match_index, left_frame, left_plane, left_axis, left_b, left_canonical_b, right_frame, right_plane, right_axis, right_b, right_canonical_b, relative_quadrant, world_axis]`. `active_edge_count=16`, `axis_edge_counts=[4, 6, 6]`.

| Match | Left endpoint (frame p<plane> ax<axis>, b / canonical b) | Right endpoint | Rel. quad | World axis |
| --- | --- | --- | --- | --- |
| 3 | image-0000 p1 ax1 0.125 / 0.125 | image-0001 p3 ax1 2.0 / 2.0 | 3 | 1 |
| 7 | image-0000 p3 ax2 0.5 / 0.5 | image-0001 p1 ax0 0.5 / 0.5 | 3 | 2 |
| 13 | image-0000 p0 ax0 1.0 / 1.0 | image-0002 p2 ax2 2.0 / 2.0 | 3 | 0 |
| 17 | image-0000 p1 ax1 3.0 / 3.0 | image-0002 p1 ax1 1.0 / 1.0 | 3 | 1 |
| 19 | image-0000 p3 ax2 1.0 / 1.0 | image-0002 p0 ax0 1.0 / 1.0 | 3 | 2 |
| 27 | image-0000 p1 ax1 0.125 / 0.125 | image-0003 p3 ax1 0.5 / 0.5 | 1 | 1 |
| 29 | image-0000 p2 ax2 0.25 / 0.25 | image-0003 p1 ax0 1.0 / 1.0 | 1 | 2 |
| 33 | image-0001 p0 ax0 1.0 / 1.0 | image-0002 p0 ax0 0.5 / 0.5 | 0 | 2 |
| 38 | image-0001 p3 ax1 2.0 / 2.0 | image-0002 p1 ax1 1.0 / 1.0 | 0 | 1 |
| 43 | image-0001 p5 ax2 1.0 / 1.0 | image-0002 p2 ax2 2.0 / 2.0 | 0 | 0 |
| 46 | image-0001 p1 ax0 0.5 / 0.5 | image-0003 p0 ax0 0.25 / 0.25 | 2 | 2 |
| 52 | image-0001 p3 ax1 1.0 / 1.0 | image-0003 p3 ax1 1.0 / 1.0 | 2 | 1 |
| 56 | image-0001 p5 ax2 1.0 / 1.0 | image-0003 p4 ax2 0.5 / 0.5 | 2 | 0 |
| 58 | image-0002 p0 ax0 0.5 / 0.5 | image-0003 p0 ax0 0.25 / 0.25 | 2 | 2 |
| 62 | image-0002 p1 ax1 1.0 / 1.0 | image-0003 p2 ax1 0.25 / 0.25 | 2 | 1 |
| 69 | image-0002 p2 ax2 1.0 / 1.0 | image-0003 p4 ax2 1.0 / 1.0 | 2 | 0 |

`conflicting_endpoints` (endpoints pooled to more than one canonical `b`): `(image-0000,1)`, `(image-0000,3)`, `(image-0001,3)`, `(image-0002,0)`, `(image-0002,2)`, `(image-0003,3)`, `(image-0003,4)`.

## Solver input bindings and same-axis capture

`solver_failure_observation.solver_input_bindings`, schema `mc-photogrammetry-solver-input-bindings-v1`, `record_sha256=2c9855bc49e82e3544045b0457a85d6fa74c544ed442d5a8ea291c7629a60e85`, basis `derived_from_post_same_axis_solver_inputs`, status complete: `input_match_count=70`, `active_edge_count=16`, 18 reading rows (each `[reading_index, frame, plane, local_axis, side, alternatives]`), 70 edge rows (`[match_index, left_endpoint, right_endpoint, quadrant, active]`, endpoints `[reading_ref, alternative_index, b, distance, support, side, sign, local_axis]`). 16 active edges; 54 inactive solver inputs.

`same_axis_candidates`, schema `mc-photogrammetry-same-axis-candidate-capture-v2`, `record_sha256=744cc05bbecce192ba9d4bf33c5306b670eebf068c82409218319ae7b11a352d`, basis `derived_from_active_pre_same_axis_candidates`, status complete: `axes=[0,1,2]`, `group_count=16`, `row_count=437`, `input_match_count=70`. 16 candidate rows `kept`; 421 `dropped` with reason `same_axis_mismatch`. Each of the 16 groups binds exactly one survivor whose match index is one of the 16 active edges above.

`solver_failure_observation.projected.boundary_saturation`, schema `mc-photogrammetry-boundary-saturation-v1`, `record_sha256=24bbc2059663a3b4148b993e0482d38fe9812b6dc2c493e1ac6288b98e77bbb5`, status complete: `admitted_match_count=1630`, `same_axis_kept_match_count=70`, `admitted_boundary_counts=[34,29,0]`, `kept_boundary_counts=[0,0,0]`, `max_shift_cells=32`. No kept shift pinned the search-domain bound.

## Translations and phase reconciliation

- Returned solver translations: **not recorded**. `SolverFailureSnapshot` deliberately drops `solution.translations` (and blocks-per-period, scale, and the full alternatives tuple); no returned translation vector exists anywhere in the artifact. This is a typed absence, reported as-is and not recomputed or substituted.
- Observer-refit translations (`phase_capture.translations`, per-axis least-squares with `image-0000` as the gauge anchor):
  - `image-0000 = [0.0, 0.0, 0.0]`
  - `image-0001 = [5.216654346695167, 341.69895008388016, 4.861263098573644]`
  - `image-0002 = [5.082234829043591, -36.366482268453055, 4.789349491593789]`
  - `image-0003 = [3.2144169303261845, 135.76875349698915, 4.518838651401037]`

`phase_capture`, schema `mc-photogrammetry-solver-phase-capture-v1`, `record_sha256=6e4cffe0ccd7ad68765cb77ca714b8f02d1a4cc94ff6eefabe7572823461ad6d`, `source_sha256=4450b351848e894ac0afac7480b74c36d896f38b32962187a114fc740457ecba`, `compatibility_decimals=6`, `mismatch_reason=null`, status complete. 32 face rows (2 endpoints x 16 active edges). Axis rows `[axis, returned_resultant, returned_residual, reconstructed_resultant, reconstructed_residual, observation_count, face_count]`:

| Axis | Returned resultant | Returned residual | Reconstructed resultant | Reconstructed residual | Obs | Faces |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.2197028820687929 | 0.21413319374511344 | 0.2197028820687929 | 0.21413319374511344 | 8 | 8 |
| 1 | 0.6154253827952421 | 0.12243923753714483 | 0.6154253827952421 | 0.12243923753714483 | 12 | 12 |
| 2 | 0.5567875711217574 | 0.1298944685360216 | 0.5567875711217574 | 0.1298944685360216 | 12 | 12 |

The phase record's independent per-face recomposition reproduces the returned aggregate metrics on all three axes at the fixed 6-decimal comparison scale; the phase control recorded no `phase_recomposition_mismatch`. This validates aggregate metric serialization/reconciliation only. It does not establish that the per-edge translation equations close or are mutually consistent, and it does not compare returned and refitted translation vectors (the returned vectors were not retained).

### Artifact-only selected-equation contradiction certificate

Bound to `solver_failure_observation.observation_sha256=5800af0ad3ece319055e9af8642d326384ece78978edaeec1fea7cbc35a1f0be`. Using the captured signed edge deltas for the best assignment, the world-axis-0 cycle through matches 43, 69 and 56 is nonzero:

`Δ43 + Δ69 − Δ56 = 1.1032689990020605 + (−0.6301293820637670) − (−3.2399259330226156) = 3.7130655499609092 blocks`.

The corresponding selected endpoints are match 43 (`image-0001 p5 ax2 b=1` ↔ `image-0002 p2 ax2 b=2`, quadrant 0), match 69 (`image-0002 p2 ax2 b=1` ↔ `image-0003 p4 ax2 b=1`, quadrant 2), and match 56 (`image-0001 p5 ax2 b=1` ↔ `image-0003 p4 ax2 b=0.5`, quadrant 2); the artifact binds all three to world axis 0. A closed translation cycle must sum to zero, so these selected equations are inconsistent. The machine-readable [cycle certificate](../../exports/diagnostics/plane-residual-observer-2026-10-07/selected-equation-cycle-certificate.json) binds the arithmetic to the existing observation file and inner digest.

The same artifact contains a lower-scoring, dropped candidate choice that is endpoint-b-consistent across **all four captured active world-axis-0 edges** (groups 2, 9, 12 and 15): group 2 row 0 / image-0000 p0 b=1 → image-0002 p2 b=0.5 (score 0.47186, Δ=+1.0541045881); group 9 row 1 / image-0001 p5 b=1 → image-0002 p2 b=0.5 (0.45093, Δ=−2.9248612420); group 12 row 1 / image-0001 p5 b=1 → image-0003 p4 b=1 (0.46528, Δ=−2.2122805437); group 15 row 1 / image-0002 p2 b=0.5 → image-0003 p4 b=1 (0.52208, Δ=+0.7125806983). All four rows are marked `dropped/same_axis_mismatch`; each has a lower score than that group's selected survivor. Repeated endpoint b values agree. The only cycle closes: `Δ43 + Δ69 − Δ56 ≈ −4.4e−16 blocks`; with image-0000 as gauge, translations `[image-0000=0, image-0001=3.9789658300, image-0002=1.0541045881, image-0003=1.7666852863]` satisfy these four axis-0 equations to floating precision. This is an artifact-derived **axis-0 feasibility witness**, not a solver replay: it says nothing about other axes, ambiguity gates, which period/plane is correct, or geometry recovery.

Independently, the selected solution's least-squares refit maximum absolute per-axis equation residuals are approximately `[1.237689, 190.958955, 3.201255]` blocks; normal-equation gradients are below `2e-13`, consistent with a fitted optimum over contradictory selected constraints.

## Digests and validation

- `observation.json` `artifact_sha256 = 2bce2af0a0a7b1bd14be427ec0af4a4ebdca47cb0eb889583ab8747d979d2c53` (re-derives).
- `solver_failure_observation.observation_sha256 = 5800af0ad3ece319055e9af8642d326384ece78978edaeec1fea7cbc35a1f0be`.
- `binding`: `entry_sha256=53d160a46f1c1f2d9be9934b459367bfdd8d858fd7fbba6cf721414e0896d524`, `graph_sha256=7a863bf1078000728e35c8e536c4e44908d16b89b3cd6be219c67354af732329`, `lattice_sha256=786c8f46b55b5ec82e93ed4dc2b70c3494339cc564ae9ce836bc1b1f8d5a1f70`, `signatures_sha256=b151b6164380f4b5de4d7ba9d427f1fe56fc18c00f0343f1875c4c87875490c4`, `input_sha256` = the four pinned PNG hashes in order.
- `graph_sha256`/`result_sha256 = 7a863bf1078000728e35c8e536c4e44908d16b89b3cd6be219c67354af732329`; `trace_sha256`/`attribution_sha256 = 1d9606cb7fec00f86387691030a0bb229fa8220fe7c0e700e53c1a700b968e5a`; `project_sha256 = e471dc2cfa37c9db4c5d0944129526de4f3de7b4ad34952443a5d5649f114263`; `session_sha256 = ce23da55adb62e0626300691bd15b175a3e8d5f8f8d91254c24275c9b4a0e037`; `policy_sha256 = 501ada9ea09113788b775c59c773dcf7089ac755a008c53ae93e2eed4616b08a`. 68 `signature_digests`; 70 input match identities.
- Shared failed-assignment digest `4450b351848e894ac0afac7480b74c36d896f38b32962187a114fc740457ecba` is identical in active-edge aliases, phase capture (`source_sha256`), solver input bindings, and same-axis capture. Shared input-match identity `1e392694dc12b3000280742720cbcd14c5dbd8ad7e61ab02dc36e51ad844ad5a` is identical in aliases, bindings, and same-axis capture. `boundary_saturation.input_match_identity_sha256 = d593cf35769050cc2ec3a8bfb9aac77ae9177872e0a4d2c2f6d8fe7f8e4800b3` uses a distinct declared digest domain and is not expected to equal the others.
- `terminal.heartbeat_sha256 = 02c63315dc95a69684e3546855f7a64c3ceec49ea17fd0967dd706a1f54286ef` equals the on-disk `heartbeat.json` hash; `terminal.manifest_sha256` equals the manifest self-digest.
- Independent re-verification (read-only): `artifact_sha256` and the manifest self-digest both re-derived equal; the runner's own `_validate_completed_artifact` (input identities, session/graph/trace/signature digests, solver-failure observation, solve context, phase capture, solver input bindings, same-axis candidates, project hash) replayed and PASSED on the persisted files.
- Manifest producer hashes (9): `scripts/run_image_derived_graph_stage.py 514ca58662d946666e5593beb2c82c5a34bb7f2dcd88b680b7a9aa36da92f2e9`; `image_derived_entry.py 1767b416061b20b1e60cd5c2d3d5b0814f232a3bc56d01328fc294818b20b6fc`; `image_derived_lattice_measure.py c894ef4cae49795e9b17f7cc10f2840fd853c00f3e049a151f0daa4bf02f4703`; `image_derived_signatures.py fa7a0973d701c3986873ad1f19751b1ed6671b3bb69f52928a2f2e13bbe26125`; `image_derived_graph.py 5d42b4feafd282b8a4f15a500120970f0bc69ca0469d885e55d3d8a77c5ab1ac`; `image_derived_graph_attribution.py 8441e91cf158d5403676d249fe6ebdebeeb723bd4a9967fdeb8371beb1c9cb81`; `.tmp-run-axis0-component-preflight-replacement.py e33456d858cf34197a77ceffd515ed054434c6eee91f9117bc6c6939bc1e67b6`; `plane_graph.py d8896f272aaf79850d3ae105dc002ba1e71dddac1d111ce91de29bdba147603b`; `lattice_planes.py 4c6853a0ad24589c244a3d0f472039ea53e9c079095e7069f0af930686a22f10`.

## Interpretation (narrow decision)

The exact recorded solver constraints and state reproduce the typed gate. `plane_graph.solve_component` assigns `plane_integer_residual` when **either** any matched-axis resultant is below `min_resultant=0.90` **or** any residual exceeds `integer_tolerance=0.15`; the resultant test is ordered first. Here all three matched axes have resultants far below the bound (`0.2197`, `0.6154`, `0.5568`), so the combined reason fires. The observer's finer projection records `minimum_resultant` as the first failed predicate and, because of that ordering, `residual_predicate_evaluated=false`; axis 0's residual `0.21413` would also have failed had it been evaluated. The phase capture reproduces the returned aggregate metrics, but this is separate from the nonzero selected-equation cycle above. Returned translations are absent, so vector-level returned/refit reconciliation is unavailable. The combined reason remains diagnostic; no image-level root cause is assigned.

This supports a real inconsistency in the selected evidence (all axes' integer-face resultants are weak) but does **not** prove which camera, plane, or period is wrong. It is diagnostic only.

## Boundary

No retries, captures, threshold changes, truth scoring, masks, sidecars, external uploads, JARs, checkpoints, holdout claims, C2, promotion, litematic export, or source/environment edits. No geometry F1, exact-state score, camera accuracy, or product-success result is measured. The four train views remain unscored and are not build-disjoint, a holdout, normal delivery, or C2.

Per the corrected task instruction, only the runner diagnostic outputs, this report, and the RESULT record were written. The task status header and the pre-existing run packet `agent swarm instantiation template/runs/plane-residual-observer-2026-10-07.json` were intentionally left unmodified; that packet's pre-launch outcome text is superseded by this report and by `RESULT-plane-residual-observer-2026-10-07.json`. `AGENT_COMMS.md` and unrelated task files were not read. Invocation count for this attempt: 1. Stopped after the single invocation.
