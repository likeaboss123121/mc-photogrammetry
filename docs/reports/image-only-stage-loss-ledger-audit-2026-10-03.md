# Image-only stage-loss ledger audit (read-only) — 2026-10-03

Task: `agent swarm instantiation template/tasks/active/image-only-stage-loss-ledger-audit-2026-10-03.md`
(`id: image-only-stage-loss-ledger-audit-2026-10-03`, kind: audit, risk: medium, attempt 1).
Worker: DeepSeek (`deepseek-flash`), read-only audit.
Status: **READ-ONLY AUDIT COMPLETE. Report is the only file written. Stop for lead review.**

No source, test, config, `AGENT_COMMS.md`, fixture, corpus, manifest, mask, reference or JAR was
edited or opened; no benchmark, reconstruction, screenshot, capture, model job, training,
dependency install, save scan, background process, shell nesting, reparse creation, recursive
deletion, encoded command, Defender/system change or commit was performed. The working tree was
left as found. All statements below are static-source readings plus the recorded, already-sealed
report bindings they cite; no hash was recomputed, no fixture was read, and the default pipeline
was not executed.

---

## 1. Scope, sources and the one carry-over warning

Read for this audit: `AGENT_HANDOFF.md` §5 (lines 264-320) and the current image-derived/status
summary (§2-§6, lines 46-334); `docs/reports/visible-state-stage-attribution-2026-10-03.md`;
`docs/reports/image-derived-graph-stage-v6-2026-10-03.md`;
`docs/reports/pair-correspondence-stage-eval-2026-10-03.md`;
`docs/reports/joint-matcher-stage-attribution-2026-10-03.md`;
`docs/reports/oracle-candidate-graph-upper-bound-2026-10-03.md`;
`docs/reports/oracle-support-sensitivity-2026-10-03.md`;
`docs/reports/DS-R4-M4-production-integration-audit-2026-10-03.md` (including its lead erratum);
`agent swarm instantiation template/project/STATUS.md`.

Source inspected (paths under `src/mc_photogrammetry/` unless noted): `cli.py`, `pipeline/image_only.py`, `pipeline/image_derived_entry.py`,
`pipeline/image_derived_session.py`, `pipeline/image_derived_connections.py`,
`pipeline/image_derived_pose_graph.py`, `pipeline/image_derived_lattice_measure.py`,
`pipeline/image_derived_geometry_orchestration.py`, `pipeline/image_derived_output.py`,
`pipeline/image_derived_output_policy.py`, `pipeline/image_derived_production.py`,
`pipeline/image_derived_production_record.py`, `pipeline/contracts.py`,
`reconstruction/facade.py`, `reconstruction/camera.py`, `reconstruction/calibration.py`,
`reconstruction/raycast.py` (trace header and `_emit_stage_trace`),
`reconstruction/plane_placement.py`, `reconstruction/plane_volume.py`,
`reconstruction/face_observation.py`, `reconstruction/plane_face_observation.py`,
`reconstruction/appearance_state_proposals.py`, `reconstruction/state_promotion.py`,
`reconstruction/frozen_image_runner.py`, `evaluation.py`,
`evaluation_pair_correspondence.py`, `export/visibility.py`, `domain/models.py`,
`scripts/benchmark_templates.py`; tests `test_image_only_entry.py`,
`test_image_derived_production.py`, `test_image_derived_output.py`,
`test_image_derived_orchestration.py`. No test was run.

**Carry-over warning (do not silently merge).** The only currently published visible-truth
percentages (T2 480/667 exact 71.96%, geometry F1 91.88%; T6 350/2,007 exact 17.44%, geometry F1
85.39%) are **supplied-camera, reference-assisted, reviewed-mask diagnostics**
(`visible-state-stage-attribution-2026-10-03.md` lines 5-15; `AGENT_HANDOFF.md` lines 302-304).
They come from `reconstruct_scene_surfaces` in `reconstruction/raycast.py`, which requires a
developer-mode camera manifest and scene manifest (`cli.py` lines 642-648). They are **not**
image-only scores and must not be placed on the same funnel as any image-derived number. No
audited artifact currently reports an image-only visible-truth percentage at all.

---

## 2. The two image-only entry paths (what "default image-only" actually runs)

There is no single production orchestrator. The default (mode `normal`) reconstruction surface is
two disconnected paths with different geometry engines and different observability.

**Path A — single-view CLI façade (the current normal `reconstruct`).**
`cli.py` lines 498-599: gate manual inputs to developer (499-514), load one image (517), derive
the GUI mask (518), in normal mode run `solve_image_cameras` (`pipeline/image_only.py:52`) and
adapt with `camera_model_from_estimate(..., position=LOCAL_FRAME_ORIGIN)` (537-547), then call
`reconstruct_facade` (561-571), save a schema-2 `ScanProject` (574-576) and export `.litematic`
(583-588). `reconstruct_facade` (`reconstruction/facade.py:18-130`) builds a fixed
`columns × rows` image-cell grid; with a camera but no scene it assigns each cell a depth of
`4.0 + (1.0 - normalized_depth) * 28.0` blocks and rounds the unprojected point to integer
"world" coordinates (80-118). It is explicitly a local frame: the CLI prints and the export
description states that metric scale and translation are unresolved (577-595). Its only state
decision is nearest-RGB in `DEFAULT_PALETTE` (8 states, `blocks/appearance.py:86-94`) plus an
optional developer palette over-ride; it has no occupancy, no face ownership, no view fusion, no
palette support pruning and no ranking.

**Path B — image-derived production (the real multi-view chain).**
`cli.py` lines 435-496 dispatch `image-derived-output` / `image-derived-deliver` into
`pipeline/image_derived_production.py::derive_image_output_from_paths` (line 215). The chain is
fixed and each stage binds the previous digest:

1. `build_image_derived_entry` (`pipeline/image_derived_entry.py:388`): `discover_media` →
   `load_image` → `build_image_derived_session` → `build_image_derived_connections`
   (`joint_matching.match_and_solve_pair` per unordered pair) → `compose_image_derived_pose_graph`
   → persist a schema-2 session checkpoint. Terminal token is always `geometry_unavailable`
   (150-157).
2. `measure_image_derived_lattice` (`pipeline/image_derived_lattice_measure.py:819`): rebuilds a
   `CalibrationResult` from retained pixels (371-431), runs `propose_image_planes_v2` and
   `frame_plane_readings_from_proposals` per frame (438-466), rebuilds and verifies the exact
   structural-weight patch, and binds geometry provenance. Status `assembled` or typed
   `unavailable` (`image_unavailable` / `pose_unavailable` / `no_lattice_reading` /
   `provenance_incomplete`).
3. `derive_image_derived_geometry` (`pipeline/image_derived_geometry_orchestration.py:208`): the
   eleven accepted stages `f1 → g → h → j → k → l → m2 → n2 → o2 → p2 → q2` (`_STAGES` line 33,
   `_READY` line 34; fixed gates `max_shift_cells=32, integer_tolerance=0.15,
   min_resultant=0.9, ambiguous_margin=0.02` at 249-253). First non-ready stage short-circuits to
   `status="unavailable"` with `reason = "<stage>_<status>"`.
4. `derive_image_derived_output` (`pipeline/image_derived_output.py:296`, `_derive_ready` 232):
   `resolve_face_placement` → `build_plane_volume` → `frame_context_from_readings` /
   `build_plane_face_observations` → `propose_face_states` → `promote_plane_states`. Status is
   one of `promoted | review | refused | budget_exhausted | unavailable`.
5. `image_derived_production._publish` (199-212) writes a **schema-3** `ScanProject` carrying
   `image_derived_session`, the compact terminal envelope and (only when `promoted`) the
   promotion volume; `cli.py::_export_schema3_project` (305-335) routes a promoted envelope
   through the verified delivery seam.

Path B evidence-bearing records exist only in memory during one run. The durable artifact is the
compact envelope in `pipeline/image_derived_production_record.py`: schema, producer,
stage/status/reason/detail, the four stage digests, `session_sha256`, `volume_sha256`,
`refusal_sha256` and (only if promoted) the serialized `PromotionResult` (keys at 107-112;
`ProductionEnvelope.__post_init__` 246-332). It carries **no** per-cell, per-stage or
per-correspondence ledger, and its status enum is only
`promoted | review | refused | geometry_unavailable | budget_exhausted | unavailable`
(97-104).

**Frozen diagnostic Path C (not production).**
`reconstruction/frozen_image_runner.py` writes the only image-only artifact that keeps per-stage
detail: `stage_reasons` for `{image, calibration, proposals, readings, signatures, matching,
solution}` (line 618), per-frame calibration/proposal/reading/signature records, and graph
matches/refusals/solution (585-616). It is a bounded diagnostic runner consumed by
`scripts/diagnose_m4_*.py`, is not wired into `cli.py`, and is not a production entry.

---

## 3. Observability map per stage (default image-only path)

Legend: **A** = Path A single-view; **B** = Path B image-derived; **C** = frozen diagnostic runner.
"Available" means an accepted record exists at that point in the default run — not that it is
persisted.

| # | Stage | Default availability | Existing observable record (source) | Persisted? |
|---|---|---|---|---|
| A1 | Image admission / decode | A, B (C) | `discover_media` stable order + exact-sha256 dedupe; `load_image` RGB+EXIF; frame id/label | session record carries id/label/shape/digest, no pixels (`image_derived_session.py:350`) |
| A2 | Usable-pixel mask | A, B, C | production `~create_gui_mask(h, w)`; evaluator frozen mask is separate (`benchmark_templates.py:280-293`); `detect_segments` consumes `~gui_mask` (`reconstruction/calibration.py:253`) | mask digest only inside face-observation bindings (`reconstruction/plane_face_observation.py:979-993`) |
| A3 | Frame quality | B (declared, unpopulated) | `FrameObservation.quality`/`mask_path`/`feature_cache_key` fields exist (`pipeline/contracts.py:41-50`) and are never constructed in `src/` | no |
| B1 | Image-derived camera (intrinsics + yaw/pitch/roll) | A (single frame), B, C | `CameraEstimate` with `position=None`, `residual_pixels`, `confidence=0.0`, `scale_known=False`, per-frame `unresolved` notes (`reconstruction/calibration.py:297-316`); session refuses a scale/translation claim | session record `cameras`/`unresolved`/`scale_known`; `residual_pixels`/`confidence` are session-record fields |
| B2 | Camera translation | never | `LOCAL_FRAME_ORIGIN` only (`pipeline/image_only.py:39`); pose graph gives a unit-edge gauge embedding, `metric_scale_known=False` (`reconstruction/image_derived_pose_graph.py:233-243,295`) | gauge poses in the session record |
| B3 | Relative rotation / yaw class | B | `RelativeCameraEvidence` per pair: status/reason, rotation, `YawClassDiagnostic`, match/support counters, digests (`image_derived_connections.py`) | graph record inside the entry payload |
| B4 | Plane proposals per frame | B, C | `region_growth.propose_image_planes_v2` result carried on the lattice result; `LatticeAbstention(frame, reason, detail)` | digest only |
| B5 | Lattice readings (period/phase/extent/coverage/structural share) | B, C | `lattice_planes.PlaneReading` per frame/axis/side with alternatives (`blocks_per_period`, `distance_blocks`, `support`) and rectification | digest only |
| B6 | Lattice sign/scale anchor | B | V2-B phase-anchor stage (`p2`), status `determined/partial/abstained` | digest only |
| C1 | Signature footprint per reading/alternative | B, C | `image_derived_signatures` records (per-domain cells/interior/coverage arrays) | digest only |
| D1 | Match hypotheses + graph admission | B, C | `image_derived_graph` `PlaneMatch`/refusals; v6 report's pre-admission 7,476 → post-admission 1,495 → same-axis 54 funnel | digest + optional failure-only diagnostics |
| E1 | Component solving / occupancy coherence | B | `ComponentSolution` from `plane_graph.solve_component`: per-axis resultants/residuals/observations, `scale_known`, `blocks_per_period`, alternatives | digest + `SolverFailureSnapshot`, `FailedActiveEdgeAliases`, `BoundarySaturationAttestation` on the attribution path (`image_derived_graph.py:174,801,998`) |
| F1 | Face placement (occupancy) | B | `ResolvedFacePlacement`/`ResolvedFace`/`PlacedCell` in lattice coordinates; refusals typed | digest; volume/promotion volume persisted only if promoted |
| F2 | Volume merge, conflicts, omissions | B | `PlaneVolumeResult`: cells + claims, `CellConflict`, `OmittedCell` (unknown), gauge state | digest; volume only if promoted |
| G1 | Face ownership + sampled patch | B | `PlaneFaceObservationResult`: per `(cell, slot, frame)` `FaceObservation` with patch/valid/coverage and `usable_coverage`, plus typed `FaceObservationReview`s (`unknown_cell`, `conflict_cell`, `unsupported_slot`, `degenerate_projection`, `no_source_pixels`, `off_frame`, `empty_coverage`; closed enum at `reconstruction/plane_face_observation.py:134-142`) | not persisted (only `result_sha256` inside the output payload) |
| G2 | Palette coverage | A, B | Path A/B use the exact 8-state `DEFAULT_PALETTE` unconditionally (`image_derived_output_policy.py:116-117` asserts exact default); a JAR palette is developer/normal CLI optional for Path A but never enters Path B | policy record carries `palette`/`palette_sha256` |
| G3 | Support pruning | **not in Path B** | Global support pruning is a raycast-only mechanism (`raycast.py` `PALETTE_SUPPORT`, trace `support_rounds`). Path B has no pruning stage; the analogous loss is G1 reviews + F2 omissions | n/a |
| G4 | Per-cell top-k candidate recall + ranking | B | `FaceStateProposal.candidate_costs` (aggregate absolute cost per canonical state, cheapest first), `best_cost`, `runner_up_*`, `margin`, `slots`, `support_frame_ids`, `observation_digests`, `evidence_digest`, `proposal_digest`; result-level `status/reason/budget` (`reconstruction/appearance_state_proposals.py:129-243,245-293`) | not persisted; `result_digest` only |
| H1 | Cross-view aggregation | B | `_aggregate` equal-weight mean of per-observation absolute costs (`reconstruction/appearance_state_proposals.py:813-870`); per-candidate merge keyed by canonical `BlockState` | inside proposals |
| J1 | "Smoothing"/placement/promotion | B | No adjacency smoothing exists in Path B. The analogous decisions are F1 placement, F2 merge/conflict/omission and `promote_plane_states` (`omitted_unknown`, `exact_state_eligible == (export_ready and omitted_unknown == 0)`, `PromotedBlock` with state/confidence/frame ids/provenance/observation digests, `PromotionResult` `reconstruction/state_promotion.py:393-474,543-628`) | `PromotionResult` persisted only on `promoted` |
| K1 | Terminal `.litematic` metrics | A, B | `LitematicExporter`/`validate_litematic`/`compare_litematics`; `evaluation.visible_metrics` computes visible exact/recall/precision/F1 against a caller-supplied reference and visible set (`evaluation.py:77-110`) | exported file + envelope digests |

**Observability verdict.** For Path B the per-cell candidate and ranking epoch (G4/H1) is
*already recorded in memory at full detail* — it is not a missing capability, it is a missing
persistence/join. The front of the funnel (A1-A3, B1-B6, C1, D1, E1) has typed status and
abstention records but only as digests in the durable artifact, and the v6 report explicitly
records that its per-stage counts are "policy outcomes ... not measured image accuracy"
(`image-derived-graph-stage-v6-2026-10-03.md` line 16). No image-only artifact anywhere maps a
stage record onto a reference-visible coordinate, so **no image-only stage currently reports a
truth coverage or exact recovery against a visible denominator.**

**Evaluator-only assets that already exist and can host the truth side.**
`evaluation.py` `visible_metrics`/`best_visible_alignment`/`visible_breakdowns` (reference +
`visible_reference` + one global integer translation and lattice-axis rotation, no scale fit or
reflection, 113-153); `export/visibility.py` `visible_cells`/`theoretical_maximum` (per-block
reachable pixel counts at a `minimum_pixels` threshold, 225-360) with
`export.visibility.ray_directions`; `evaluation_pair_correspondence.py` (strict same
`(coordinate, slot)` surface pairs within `SURFACE_TOLERANCE_BLOCKS = 0.0625`, unique
`(left pixel, right pixel)` identity, typed `undefined_reason` instead of zero, per-pair stage
funnel and pooled/macro group totals); `scripts/benchmark_templates.py`
`_stagewise_visible_loss`/`_candidate_availability` for the raycast trace (1238-1457 and 1124-1235).

---

## 4. Denominators and accounting rules

The handoff already fixes the metric definitions (`AGENT_HANDOFF.md` lines 266-283). The ledger
must keep three numerator/denominator spaces separate; adding them is the failure mode the
scorecard exists to prevent.

1. **Geometry / occupancy denominator** — `V = reference coordinates visible through usable
   image pixels` under an independently frozen mask and a `minimum_pixels` threshold. Used by
   visible geometry recall/precision/F1 and by every occupancy checkpoint; phantoms are
   `C minus R`. Path A local-frame output cannot be placed on this denominator without a
   scale-preserving alignment; a single-view local frame is one of the cases the handoff excludes
   from scale fitting, so Path A is scored only as a labelling/vocabulary ledger, never as
   geometry recovery.
2. **Exact-state denominator** — the intersection of `V` with the canonical palette the run
   actually used. Path B's palette is always the 8-state `DEFAULT_PALETTE`, so the palette
   ceiling is a real, pre-measured vocabulary ceiling on the image-only path (not an assumption);
   Path A may use a JAR palette, so its ceiling must be reported separately.
3. **Pair-correspondence denominator** — jointly visible, strict same-surface truth pixel pairs,
   with unique-pixel-pair identity. Keyed by `(left pixel, right pixel)` and scored against
   `(coordinate, slot)` surface ownership derived from evaluator cameras and world truth. This
   denominator is built only by evaluator code and never enters normal reconstruction.

Each checkpoint reports:
`exact_state_matches / |V|` (common-denominator coverage),
`exact_state_matches / |stage_conditional_denominator|` (incidence),
`geometry_missing` (cell has no candidate at all) versus `matched_wrong_state` (cell present with
the wrong state), the **oracle ceiling** for that stage (the best attainable count given the
stage's own inputs — palette membership, candidate availability at K, the retained candidate set,
or a truth-filtered candidate graph), and item-level gains/losses.

**Non-monotonicity is mandatory in the contract.** Every transition reports `gained`, lost to
geometry, lost to state, and net change, because a later checkpoint is not bounded by an earlier
one (the raycast implementation already does exactly this at
`scripts/benchmark_templates.py:1358-1382`; the visible report records T2 smoothing 66 gains
against 45 losses and the T6 no-prune run gaining 59 net).

**The one unavoidable image-only substitution.** Path B has no adjacency smoothing; its
"gain/loss" epoch is G1 review omissions → G4 candidate availability → G4 raw winner → F2
omission/conflict → J1 promotion. The ledger must name these stages honestly rather than import
raycast stage names, and it must report Path A separately because its stages are different again.

---

## 5. Ledger design specification

A ledger row is keyed by a **frozen visible reference coordinate** (evaluator-only) plus the
**truth state** at that coordinate, and, for the correspondence section only, a frozen
`(left pixel, right pixel)` pair. Per stage the row records:

```
{ stage, stage_kind: "availability" | "label" | "geometry",
  candidate_key: [x,y,z] (aligned), candidate_present: bool,
  stage_state: <state payload> | None,
  stage_conditional_member: bool,
  decides: "geometry" | "state" | "omission" | "abstention",
  evidence_digest: <the accepted record's own digest> }
```

Ledger-level fields: the frozen mask policy and per-frame mask digest, the alignment
(`rotation_quarter_turns`, integer `translation`, no reflection/scale), the reference digest, the
run's stage/status/reason and the accepted stage digests, and a `schema`/`producer` identity so a
ledger cannot be mistaken for a reconstruction result.

Checkpoint set for Path B (order is the pipeline order; kinds in parentheses):

| Checkpoint | Kind | Definition | Ceiling |
|---|---|---|---|
| `admitted` | geometry | reference cell has at least one admitted frame/mask pixel | frozen mask ∩ frame set |
| `camera` | availability | cell is projectable by every camera the run solved for its owning views | solved intrinsics/pose residual |
| `reading` | geometry | cell lies on a plane reading retained for some frame | proposal/reading stage reach |
| `signature` | geometry | the reading/alternative covering the cell has a signature | accepted proposal set |
| `match` | geometry-availability | cell participates in ≥1 admitted component match | truth-filtered candidate graph (oracle) |
| `solved` | geometry-availability | component solution places the cell's plane | oracle translation/quadrant |
| `placed` | geometry | `PlacedCell` exists for the cell | placement universe |
| `volume_cell` | geometry | cell merged with ≥1 compatible claim | volume support |
| `not_omitted` | geometry | cell is not an `OmittedCell` | unknown-cell input |
| `observed_face` | geometry-availability | a `FaceObservation` exists for `(cell, slot, frame)` with effective coverage | face-observation budget |
| `palette` | availability | truth state ∈ run's canonical palette | `DEFAULT_PALETTE` (8) |
| `candidate_k` | availability | truth state ∈ the cell's `candidate_costs[:K]`, K ∈ {1, 2, 4, 8} | cell's own ranked list |
| `raw_winner` | label | `FaceStateProposal.state` == truth | aggregate cost argmin |
| `promoted` | label | `PromotedBlock.state_name/properties` == truth | promotion eligibility |
| `terminal` | label | exported `.litematic` block state == truth | `LitematicExporter` round trip |

Every checkpoint reports its **own** conditional denominator (`|cells that reached the previous
checkpoint and are in the denom|`), its oracle ceiling, and, for label checkpoints, a
`gained`/`lost` split against the previous checkpoint. A checkpoint with no carried record is
reported `status: "unavailable"` with a typed `unavailable_reason` and **contributes no zero** —
this is the same fail-closed rule the benchmark's `_stagewise_unavailable` uses
(`scripts/benchmark_templates.py:1460-1509`).

**Evaluator-only discipline.** Reference coordinates/truth, the visible mask, evaluator cameras
and alignment are read by the ledger scorer only, after a reconstruction has finished, from
already-loaded evaluator inputs. The ledger type lives in the evaluation layer or an
`scripts/evaluation/` runner, never in `pipeline/`, `reconstruction/` or a public production
signature; a production module must not be able to import it and acquire a reference (the same
rule `evaluation.py:1-6` states for the existing metrics module). The default reconstruction
receives none of this input and no new parameter.

---

## 6. Scoreable now vs. requires instrumentation; promotability

**Scoreable from already-accepted artifacts (in memory at the end of a Path B run, with no new
production instrumentation).** The output result already carries the whole tail: G1
(`output.frames`, `output.face_observations` including reviews and `usable_coverage`),
G4/H1 (`output.proposals` with full `candidate_costs`/margins), J1 (`output.promotion` with
`PromotedBlock` evidence) and K1 (the delivered `.litematic`, re-readable by
`export.compare.load_litematic`). The ledger's scoring and transition arithmetic for
`volume_cell`→…→`terminal` can be built and fully unit-tested today against the accepted
synthetic recipe fixtures that `tests/test_image_derived_orchestration.py` already provides
(`recipe` fixture at lines 227-258, `m5_result` at 260-268, built from
`tests/test_plane_placement.py`; it is synthetic and reads no reference or JAR).

**Requires new instrumentation (not present in any accepted persisted record).**
`admitted`, `camera`, `reading`, `signature`, `match`, `solved` and the projection half of
`placed` are emitted only as digests (or as per-frame/typed records on the diagnostic
`frozen_image_runner` path). To score them against `V`, one of these must be added:
(a) a bounded evaluation-only per-stage export from the already-computed records
(`entry.session` cameras, the lattice result's readings/alternatives, the graph result's
admitted matches/solution, the placement universe/supports) into a vector keyed by the run's own
lattice coordinates, plus the alignment search; or (b) a freeze/diagnostic runner extension that
already keeps those records. Both are evaluator-only: no gate, threshold, matcher, solver,
policy or production signature changes are required or permitted, and the vector is produced
after the production result is sealed.

**Can the default image-only path currently emit a promotable output? No — and this is the
decisive promotability fact.**

- Path B cannot reach `promoted`. `derive_image_derived_geometry` returns `status="unavailable"`
  at the first non-ready stage, and every recorded genuine run stops at stage `g`
  (`plane_integer_residual` on the four frozen PNGs with resultants .39146/.24717/.24521 against
  .90; `ambiguous_quadrant` on the 8-view Canal Town freeze)
  (`DS-R4-M4-production-integration-audit-2026-10-03.md` §2 G3, §3; lead erratum line 285
  confirms the persistence mechanism exists but "no current rendered fixture has reached
  promotion"). `image-derived-output` therefore reports a typed refusal/unavailable and
  `_publish` writes an empty volume with no promotion.
- Even if geometry assembled, the delivery gate requires `export_ready`, `exact_state_eligible`
  and `omitted_unknown == 0` (`reconstruction/state_promotion.py:609`,
  `pipeline/image_derived_output_policy.py:185` `allow_omitted_unknown=False`), while the one accepted
  genuine recipe retains 15 unknown omissions (`DS-R4` §2 G2). So the accepted synthetic recipe
  is structurally non-promotable by design, not by a metric shortfall.
- Path A produces a local-frame `.litematic` with unresolved scale and translation, but that file
  is not a promotable reconstruction: it cannot be aligned to a reference under the allowed
  integer-translation/rotation-only alignment, and the handoff forbids scale fitting from hiding
  wrong block counts (`AGENT_HANDOFF.md` lines 271-278). Path A output may be scored only for
  vocabulary/label sanity, never for visible geometry or visible exact-state recovery.
- Consequence for the ledger: today it can measure the tail of the funnel on accepted synthetic
  records and must report every upstream checkpoint `unavailable`. It cannot produce an
  image-only end-to-end loss number until either a genuine run reaches at least assembled
  geometry, or the evaluation-only per-stage export (instrumentation above) is added. Claiming an
  image-only percentage before then would be a category error.

---

## 7. One bounded next implementation task

**ID:** `image-only-stage-loss-ledger-e1` (evaluation-only; no product behaviour change).
**Rationale:** the scoring contract and the tail-stage join are implementable and independently
testable now; the upstream export is a separate, larger decision that should follow only after
the contract and its fail-closed rules are accepted. The user's actual question — where accuracy
is lost *before* tuning — is only answered by the follow-on front-end export
(`image-only-stage-loss-ledger-e2`, not specified here), because today no image-only run reaches
a state decision at all; E1 makes that export's scoring target concrete and testable first.

**Owned paths (write scope):**
- `src/mc_photogrammetry/evaluation_image_only_ledger.py` — new evaluation module.
- `tests/test_evaluation_image_only_ledger.py` — new focused tests.
- `docs/reports/image-only-stage-loss-ledger-e1-2026-10-03.md` — new report.

Nothing else is created or edited. No `pipeline/`, `reconstruction/`, `cli.py`, `scripts/`,
`domain/models.py` or benchmark change; no schema change; no new dependency.

**Module contract (bounded):**
1. Frozen ledger record types: `LedgerStage`, `LedgerRow`, `StageCheckpoint`, `StageTransition`,
   `ImageOnlyStageLedger`, each with a `schema`/`producer` identity, canonical JSON
   `to_record()`, and constructor validation. Any malformed input is a typed error, never a
   partially built ledger.
2. `build_image_only_ledger(output, *, truth, visible_reference, alignment)` — pure function over
   an already-produced `ImageDerivedOutputResult` (or the earlier stage records when a later
   stage is absent). It reads only: `output.proposals.proposals` (candidate costs, winner,
   margins, slots, support frames), `output.face_observations.observations`/`reviews`
   (ownership/coverage), `output.volume.cells`/`conflicts`/`omitted`, `output.promotion.cells`
   and `output.placement`; plus, when supplied, the optional upstream vectors for
   `admitted/camera/reading/signature/match/solved`. It must not import `cli`, `project.store`,
   `export.litematic` or any IO module, and its public signature must not accept a project path,
   a manifest, a reference path, a JAR, coordinates, directions or bounds.
3. `score_image_only_ledger(ledger, *, reference, visible_reference, alignment)` — reference-aware
   scoring with the three separated denominators of §4, per-checkpoint incidence and
   common-denominator coverage, oracle ceilings, `gained`/`lost`/`geometry_lost`/`state_lost` per
   transition, and `metric_value(numerator, denominator, undefined_reason=...)`-style typing so an
   undefined denominator is `null`, never `0`. Reuse `evaluation.py` for the visible metrics and
   the permitted alignment; do not reimplement `best_visible_alignment`.
4. `image_only_ledger_unavailable(reason)` — the fail-closed section for a checkpoint whose record
   was not carried, mirroring the benchmark's typed `unavailable_reason` shape.

**Fixture roles (all synthetic / in-repo; no reference, JAR or screenshot is opened):**
- Primary: the accepted synthetic recipe already built by
  `tests/test_image_derived_orchestration.py::recipe` / `m5_result` / `pipeline` (sourced from
  `tests/test_plane_placement.py`) driven through the real `propose_face_states`,
  `resolve_face_placement`, `build_plane_volume`, `build_plane_face_observations` and
  `promote_plane_states` bodies. This exercises the real tail-stage records with real digests.
- Synthetic truth: a small hand-built `{coordinate: BlockState}` reference plus a
  `visible_reference` subset and a known alignment, so expected gains/losses/oracle ceilings are
  computable by hand in the test. At least one cell must be absent from the palette and at least
  one truth state must be present in `candidate_costs` but lose the argmin, so the
  palette/candidate/raw-winner transitions are exercised with a known non-monotone step.
- Negative controls: a ledger built without any upstream vector must report every upstream
  checkpoint `unavailable` and must not emit a numeric zero for them; a ledger built from a
  tampered record (wrong digest/binding) must raise a typed refusal and produce no ledger.

**Required checks (the implementation task's acceptance):**
1. New module + tests pass focused suite:
   `.\.venv\Scripts\python.exe -m pytest -o addopts='' -q tests\test_evaluation_image_only_ledger.py`
   (expect all pass, exit 0) and `py_compile` on the module and test.
2. Contract controls: exact-state and incidence denominators are distinct and both reported;
   truth-in-palette cells and beyond-palette cells are charged to different denominators;
   `gained` is counted separately from `lost` on every transition; a checkpoint with a missing
   record reports `available=False` with a typed reason and no zero.
3. Oracle controls: a truth-perfect candidate/oracle input raises the corresponding checkpoint to
   its ceiling while leaving the production winner untouched; a truth state outside the palette
   cannot be counted reachable at any K.
4. Isolation controls: `inspect.signature` of the public builders contains no reference,
   manifest, path, camera, coordinate or bound parameter; a forbidden-token scan over the module
   source shows no `manifest`, `f3`, `bounds`, `open(`/`Path` IO, `LitematicExporter`, a call
   into a production stage (`resolve_face_placement`, `build_plane_volume`,
   `propose_face_states`, `promote_plane_states`) or an `image_derived_production` /
   `cli` / `project.store` import; the
   ledger object cannot be fed back into any production function (no `pipeline`/`reconstruction`
   import consumes it).
5. Determinism: two builds over identical accepted records produce byte-identical canonical
   ledger JSON and digest.
6. Regression: `tests\test_image_derived_output.py`, `tests\test_image_derived_orchestration.py`,
   `tests\test_plane_volume.py`, `tests\test_state_promotion.py` and `tests\test_evaluation_pair_correspondence.py`
   still pass unchanged (the ledger must not alter any accepted record or default).
7. Report records: exact commands, pass counts, module/test SHA-256, the synthetic fixture's
   hand-computed expected ledger rows, every checkpoint's availability, and the explicit statement
   that all upstream checkpoints are `unavailable` and no image-only accuracy is claimed.

**Acceptance metrics (all must hold, not a percentage gate):**
- every stage transition is accounting-complete (`gained + geometry_lost + state_lost` reconciles
  with the two checkpoint counts for every row);
- the three denominators are never summed or interchanged anywhere in the ledger or report;
- `promoted`/`terminal` checkpoints on a non-promoted output are `unavailable` (never `0`), which
  is exactly what the current default path must yield;
- the available tail checkpoints are computed from real accepted records with replayed digests,
  with no fabricated per-cell value.

**Explicit non-goals:** no production instrumentation in this tranche; no per-stage export from
the geometry chain; no gate/threshold/matcher/solver/policy change; no new fixture, capture,
benchmark, screenshot, mask, reference or JAR read; no CLI/UI/schema change; no model training or
fitting; no claimed image-only accuracy.

**Stop conditions (stop and report, do not work around):**
- if any required tail record cannot be reached through the accepted public constructors without
  changing production code, stop and report the exact missing seam;
- if the accepted synthetic recipe cannot be driven to a `FaceStateProposalResult` through public
  APIs in a test without touching `src/`, stop and report the seam needed;
- if the permitted alignment cannot be expressed without scale fitting, stop and return the
  question to the lead (do not add a scale fit);
- if a check fails, do not relax the check or the acceptance metric; report the failure with
  evidence.

---

## 8. Model-fitting gate (explicitly deferred)

No local texture model may be fitted on the strength of this audit. `AGENT_HANDOFF.md` lines
309-310 and the E3/E5 reports close the route until: (i) a lead-approved world-grouped
train/dev/test data contract and export exists (`texture-poc-e3-factory-patch-dataset-export-2026-10-03.md`
lines 246-247; every E3 artifact is `evaluation_only: true`, `training_permitted: false`);
(ii) the label-purity gate is cleared (`texture-poc-e5-factory-state-support-2026-10-03.md`
lines 417-419, 438-439) — only 2 independent builds exist, so no leave-one-build-out split has
more than one training build; and (iii) the image-only geometry bottleneck measured here is
understood, because T6's dominant default loss (1,527 visible truth states removed before
candidates) is an appearance/support loss, while the image-only front end currently loses before
any appearance decision is reachable at all. The immediate need is the ledger measurement, not a
model.

---

## 9. Limits of this audit

Static inspection only; no pipeline, benchmark, fixture or test was executed and no hash was
recomputed. Stage line numbers are the working-tree readings at audit time and can drift. The
Path B stage names, statuses and gate values are read from source and cross-checked against the
already-sealed reports cited in §1; the recorded T2/T6 and graph numbers are quoted from those
sealed artifacts, not re-derived. This audit establishes no image-only accuracy, no promotion and
no threshold change. The proposed next tranche needs a lead/Jev routing decision and (because it
touches the evaluation surface and the ledger contract) an Astra review of the truth-input
boundary and the three-denominator rule before any implementation.
