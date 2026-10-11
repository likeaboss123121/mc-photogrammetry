# DS stagewise-loss executable test plan (report-only audit, R1)

- **Task:** `DS-STAGEWISE-LOSS-R1` (AGENT_COMMS.md, 2026-10-04 Lead -> DeepSeek).
- **Owner:** DeepSeek (Sol-equivalent, direct-shell policy in `docs/deepseek-worker.md`).
- **Kind:** read-only audit and plan. **Write scope:** this file plus a completion entry in
  `AGENT_COMMS.md`. Nothing else.
- **Status:** audit complete; **stopped for lead review**. No source, test, capture, dataset,
  artifact, schema or dependency was changed. No product, evaluator, solver or benchmark path
  was run. No screenshot, truth, camera, mask, reference, JAR or scene payload was opened;
  every number below is quoted from an already-sealed report or read from source structure.
- **Revision (R1 correction):** revised in place per the Lead review HOLD of 2026-10-04
  (`AGENT_COMMS.md`) that tightened `DS-STAGEWISE-K1`. K1 is now an oracle-camera
  **sensitivity/upper-bound** arm, not a causal K-versus-geometry attribution; it specifies a fresh
  paired A0/A1 rerun, A0 serialized and hashed before any evaluator read, full
  recall/precision/F1 with exact cell-set deltas, the 90% geometry-F1 gate as descriptive only,
  and A1 as `reference-assisted` and out of promotion. Report-only: this revision ran no source,
  test, capture, evaluator, product or solver path and read no payload.

## 0. Scope, inputs read, and how to read this document

Inputs read for this audit (all prose/source/schema; no payload):

| Input | Role here |
|---|---|
| `AGENT_HANDOFF.md` §5 (lines 264-313), §2-§4 status lines | frozen metric definitions, required scorecard, current measured state |
| `docs/reports/image-only-loss-attribution-design-2026-10-03.md` | accepted A1/A2 design, five labelled spaces, checkpoint tables, model gates |
| `docs/reports/image-only-stage-loss-ledger-audit-2026-10-03.md` | observability map, scoreable-vs-instrumentation split, Path A/B/C |
| `docs/reports/image-only-attribution-a2-2026-10-03.md` | measured Route A rows; Route B typed-unavailable tail; contract facts |
| `docs/reports/image-only-route-b-stage-loss-run-2026-10-04.md` | the one accepted pinned Route B run and its frontier |
| `docs/reports/visible-state-stage-attribution-2026-10-03.md` + `AGENT_HANDOFF.md:76-90` | supplied-camera per-stage counts (separate universe) |
| `docs/reports/texture-poc-data-audit-2026-10-03.md` | label-derivation trace, factory groups/splits, `PaletteIndex` baseline |
| `docs/reports/texture-poc-e1-results-2026-10-03.md`, `-e2f-...`, `-e2e-...` | per-patch scorer rows and production-patch parity |
| `docs/reports/texture-poc-e3-factory-patch-dataset-export-2026-10-03.md` | dataset-factory exporter interfaces and yield |
| `docs/reports/texture-poc-e4-...`, `-e5-...`, `-e6-...`, `-e7-...` | label consistency, support/OOV, acquisition gates, palette scanner |
| Source structure only: `src/mc_photogrammetry/evaluation_image_only_frontier.py`, `evaluation_image_only_attribution.py`, `evaluation.py`, `evaluation_pair_correspondence.py`, `reconstruction/appearance_state_proposals.py`, `dataset/{splits,schemas,batch,plan}.py`, `export/{compare,validation}.py`, `scripts/benchmark_templates.py`, `scripts/evaluation/*.py` | current scorer/adapter interfaces and reusable seams |

Reading rules used throughout:

- `[F]` = a fact read from a sealed report or from source structure.
- `[R]` = a requirement/plan proposed by this audit; nothing here authorizes implementation.
- A ratio is **never** written without its universe. An undefined denominator is `null` with a
  typed reason, never `0`.
- "Typed unavailable" means the record was not carried and the metric is `null` with a typed
  reason. "Error" means a malformed/contradictory record that must raise atomically
  (`FrontierRefusal` / `AttributionRefusal` / the stage's own typed exception) and produce no
  partial record.

## 1. Universes, and the one naming collision that must be fixed before coding

### 1.1 Fixed universes

| Symbol | Universe | Used by |
|---|---|---|
| `V` | frozen visible reference **cells** (reference ∩ usable-mask pixel domain, primary threshold ≥ 1 usable pixel) | K-conditional, G, C, E, placement, X |
| `V_P` | `{v in V : truth(v) in run palette}` (conditional vocabulary subset of the same cell universe) | C, E conditional |
| `F` | frozen evaluator-visible **face observations** `(frame_id, coordinate, slot)` whose truth surface covers ≥ 1 usable pixel | O, P |
| `Q` | frozen **usable frame pixels** (and their per-face rectified 16×16 descendants) | O |
| `R_px` | frozen **strict same-surface truth pixel pairs** with unique `(left pixel, right pixel)` identity | pair correspondence (design-doc `P`) |
| `N_view` | admitted frozen frames (views) | K diagnostic |

`V`, `V_P`, `F`, `Q`, `R_px` are **never summed, divided or differenced across each other**
(design §2.1; `UNIVERSE_OF_SPACE` in `evaluation_image_only_attribution.py:248-255`).

### 1.2 The collision: the Lead's `P` is not the design's `P`

- `[F]` The accepted design (`image-only-loss-attribution-design-2026-10-03.md` §2.1 item 3,
  `evaluation_image_only_attribution.py:234-242`) uses **`P` = pair correspondence**, universe
  `R_px`, emitted by `MetricSpace.PAIR_CORRESPONDENCE` / `PIXEL_PAIRS_UNIVERSE`.
- `[F]` The current assignment names **`P` = production-patch availability**, which is a
  different question (does a frozen evaluator-visible face observation produce a usable
  production patch and is that patch bound to the production sidecar?). Its natural universe is
  `F`, not `R_px`.
- `[R]` This plan therefore names the assignment's stage **`P` (production-patch availability)**
  and renames the design-doc pair stage **`PC` (pair correspondence)** everywhere below.
  Implementation must keep the two as distinct `MetricSpace`s or distinct named sections; a
  single `P` label must never cover both. `PC` is retained as a required diagnostic but is not
  one of the eight stages the assignment asked to specify; it is summarised in §2.9.

## 2. Executable test matrix

Each stage subsection fixes: universe and both denominators, oracle ceiling, loss/transition
counts, typed-unavailable vs error, the **required synthetic contract test**, and the
**required real image-only evaluation**. Commands are indicative and use the repository
convention (`.venv\Scripts\python.exe`; `PYTHONPATH=src` where the runner imports the package).

### 2.1 Stage K — camera / scale

| Field | Definition |
|---|---|
| Quantity | camera diagnostic (kind `camera`; `RecordKind.CAMERA_DIAGNOSTIC`) |
| Fixed end-to-end denominator | `N_view` = admitted frozen frames in the run's input set (1 for Route A; the discovered still-image count for Route B). **No cell denominator.** |
| Conditional denominator | frames that reached a calibration record and did not abstain (`camera` checkpoint reached: `frontier.camera.reachable`) |
| Metric rows | orientation error (pitch, yaw mod 90°, raw yaw reported separately), focal error %, seam displacement (px, normalized by `1920/max(w,h)`), plus typed `unresolved` members (position, scale, yaw quadrant) |
| Oracle ceiling | the evaluator camera for the same frame — **reference-assisted diagnostic only**; it is an upper bound on K, never a production input and never a stage of the image-only funnel |
| Transitions / loss counts | **none.** K carries no cell-loss equation and contributes no term to any transition (design §2.2 B2, §2.3 A1). Loss is reported as per-frame gate pass/fail counts and the count of `unresolved` members |
| Typed unavailable vs error | unavailable: `camera_unavailable`, `pose_unavailable`, `camera_abstained`, plus `unresolved` members `position`, `scale`, `yaw_quadrant`; error: malformed non-finite/missing intrinsics must raise the calibration module's typed exception, never fabricate a pose |
| Required synthetic contract test | (a) an `ImageOnlyAttribution` camera row carries `metric_value=null`, `checkpoint_reached` from the frontier, and no transition; (b) an independent analytic camera test that does **not** generate and score through the same helpers (`tests/test_calibration_independent.py`, already accepted) must be extended to the K→G join helper |
| Required real image-only evaluation | `scripts/calibrate_frames.py` per predetermined set with settings + input hashes + interpreter versions, `--score` only after inference; `scripts/seam_landmarks.py` frozen-landmark protocol. Accepted evidence: 22/22 views within gate, seam 23/23 frames (handoff §2, `M3-seam-gate-2026-09-08.md`/`image-only-attribution-a2` provenance) |
| Reusable now | `evaluation_image_only_frontier.py` `camera` kind; `evaluation_image_only_attribution.py` `MetricSpace.CAMERA_DIAGNOSTIC`; `calibrate_frames.py`, `seam_landmarks.py` |
| Exact gap | no per-frame **K→G join**: the frontier and attribution record K as one diagnostic row, but nothing joins a frame's camera object to the cells it places. No evaluator-camera override exists in `scripts/evaluation/run_image_only_route_a.py` (it calls `solve_image_cameras`; `reconstruct_facade` already accepts a `CameraModel`). §4.2 specifies such a join only as a **sensitivity** join for one evaluation-only arm, not a causal decomposition. |

### 2.2 Stage G — bounds / visible geometry

| Field | Definition |
|---|---|
| Quantity | geometry, `MetricSpace.GEOMETRY`, universe `frozen_reference_cells` |
| Fixed end-to-end denominator | `\|V\|` at the primary ≥ 1 usable-pixel threshold; stricter thresholds (2/5/10) are **separate labelled rows** (`REPORTED_VISIBILITY_MINIMUMS`) and never replace the primary row |
| Conditional denominator | cells that reached the previous G checkpoint in the same universe, e.g. Route B `placed` → `volume_cell`; Route A `facade_grid` cells that fall on the run's own grid |
| Metric rows | visible geometry recall `\|C∩V\|/\|V\|`; precision `\|C∩V\|/(\|C∩V\|+\|C−R\|)`; F1; `geometry_missing` vs `matched_wrong_state`; phantom count; bounds are derived from `C`, never from reference bounds |
| Oracle ceiling | cells on the run's **own** coordinate grid under the permitted alignment (global integer translation + lattice Y-quarter-turn, no reflection, **no scale fit**). Route B additionally uses the truth-filtered candidate graph as the match-stage ceiling. Route A's unresolved metric scale is a score-reducing caveat, never `unavailable` |
| Transitions / loss counts | G→G is **unpartitioned**: `next = previous + gained − lost`; `geometry_lost = state_lost = null` on a G step. Loss counts are `gained`, `lost`, `net` per comparable checkpoint pair |
| Typed unavailable vs error | unavailable: `g_unavailable` (+ detail, e.g. `plane_integer_residual`, `ambiguous_quadrant`), `output_record_absent`, `gauge_unresolved`; a checkpoint never reached contributes no zero. Error: a candidate coordinate outside the frozen domain, a non-integer coordinate under the permitted alignment, or a cross-space row/transition must raise `AttributionRefusal`/`FrontierRefusal` |
| Required synthetic contract test | hand-built synthetic truth + `visible_reference` + known integer alignment; assert recall/precision/F1, a non-identity translation + quarter-turn, scale/reflection/per-cell alignment refusal, and `next = previous + gained − lost` with `geometry_lost=state_lost=null` |
| Required real image-only evaluation | `report_image_only_attribution.py --route route-a` (normal-mode replay) and `--route route-b` (frozen multi-view) on the pinned fixtures with reviewed mask pins; report `V`, the alignment used, per-checkpoint rows and the frontier stop |
| Reusable now | `evaluation.visible_metrics`, `evaluation.best_visible_alignment`, `score_aligned_cells`, `score_aligned_geometry_coordinates`, `align_candidate_coordinates`, `_stagewise_visible_loss` (supplied-camera shape), `evaluation_image_only_frontier` |
| Exact gap | Route B upstream G checkpoints (`admitted`, `camera`, `reading`, `signature`, `match`, `solved`) exist only as digests; no evaluation-only per-stage coordinate vector is exported, so those G rows are typed unavailable. `_stagewise_visible_loss` is the raycast **supplied-camera** funnel only and cannot be pointed at either image-only route |

### 2.3 Stage O — pixel / face ownership

| Field | Definition |
|---|---|
| Quantity | ownership, `MetricSpace.OWNERSHIP`, universes `FRAME_PIXELS_UNIVERSE` and the explicitly defined `(cell, slot, frame)` face-observation set `F` |
| Fixed end-to-end denominator | `\|Q\|` (frozen usable frame pixels) **or** `\|F\|` (evaluator-visible face observations), declared per row; never `\|V\|` |
| Conditional denominator | face observations whose cell/plane reached placement/volume in the run's own frame |
| Metric rows | per-pixel ownership coverage and false-ownership rate; per-face coverage recall; the closed review histogram (`unknown_cell`, `conflict_cell`, `unsupported_slot`, `degenerate_projection`, `no_source_pixels`, `off_frame`, `empty_coverage` — `reconstruction/plane_face_observation.py:134-142`); mask/valid-mask agreement |
| Oracle ceiling | evaluator surface truth per pixel: `evaluation_pair_correspondence.raycast_surface` for cube truth, `export/attribution.render_result` + `export/visibility.visible_cells` for exact/alpha/partial ownership |
| Transitions / loss counts | O is **not** a cell funnel and never takes the `next = previous + gained − lost` cell identity. Report coverage/agreement rates plus per-reason counts; O contributes no term to any G/C/E transition |
| Typed unavailable vs error | unavailable: an explicit empty O section with a typed reason (the accepted A2 shape is `no bound evaluator surface-ownership artifact`); error: an ownership row or transition constructed in another space must raise `AttributionRefusal` |
| Required synthetic contract test | `tests/test_plane_face_observation.py` and `tests/test_face_sampling.py` already cover ownership/coverage/review reasons and weighted-valid sampling; add a contract test that an O row in the C/G/E universe is refused, and that an O section with a missing record is `null` not `0` |
| Required real image-only evaluation | an evaluation-only export of `PlaneFaceObservationResult.observations`/`reviews` for a genuine run, joined to a separately pinned evaluator `(cell, slot, frame)` surface-ownership render at the frozen camera |
| Reusable now | `reconstruction/plane_face_observation.py`, `reconstruction/face_sampling.py`, `export/visibility.py`, `export/attribution.py` |
| Exact gap | **O is not scoreable today.** The frozen A2 truth assets carry no bound `(cell, slot, frame)` surface-ownership artifact, so the accepted Route B attribution reports an explicitly empty O section. A new evaluation-only ownership export + join is required; no production change |

### 2.4 Stage P — production-patch availability

| Field | Definition |
|---|---|
| Quantity | availability, universe `F` (evaluator-visible face observations); a cell-level variant over `V` may be reported **separately and labelled** |
| Fixed end-to-end denominator | `\|F\|` over the frozen frames (each `(frame_id, coordinate, slot)` whose truth surface covers ≥ 1 usable pixel) |
| Conditional denominator | faces whose cell reached the run's placement/volume (a G-conditional subset), and faces whose truth state is in the run palette (a `V_P`-like conditional, reported separately) |
| Metric rows | (a) patch produced? (b) `valid_fraction ≥` the sampler floor? (c) exact-model geometry used (not cube fallback)? (d) patch/valid/coverage **digest** equal to the production sidecar for the same key? (e) numeric RGB/coverage MAE as a diagnostic only. Report `E1-only`, `production-only` and joined counts because the key join is not lossless |
| Oracle ceiling | every evaluator-visible face whose source pixel footprint meets `MINIMUM_FACE_AREA` and `minimum_visible_fraction`; the sampler's own settings are the ceiling, not the reference |
| Transitions / loss counts | counts of faces with no patch, below valid-coverage floor, cube-fallback labelled, or unbound to the sidecar. P is an availability space; it contributes no cell transition |
| Typed unavailable vs error | unavailable: `no_source_pixels`, `off_frame`, `empty_coverage`, `insufficient_valid_coverage`, `model_metadata_unavailable`, `states_unavailable`, `states_incomplete`, and "no production sidecar captured" as a typed section reason; error: a shape/dtype mismatch, non-finite array, or a key that joins two different `(frame_id, coordinate, slot)` identities must raise `TraceReplayError` (E2F contract) |
| Required synthetic contract test | E2F synthetic rows: exact digest parity separated from tolerant MAE; empty both-valid intersection is `null`; shape mismatch refuses; mixed row shapes report `patch_pixels_per_row=null` with `distinct_patch_pixel_counts`. Add the same contracts for the **image-only** sidecar key set |
| Required real image-only evaluation | capture a production observation sidecar on the frozen **image-only** route (Route A and/or Route B), then replay it against the sealed E1/E3 patch labels for the same frozen frames |
| Reusable now | `scripts/evaluation/replay_texture_observation_trace.py` (E2B/E2D/E2F), `scripts/evaluation/evaluate_texture_patch_labels.py` (E1), `export_factory_texture_patch_dataset.py` (E3), `reconstruction/face_sampling.py` |
| Exact gap | the accepted sidecar/parity evidence (E2E) is **supplied-camera only**. There is no image-only production trace, no digest-bound join from an image-only run to E1/E3 labels, and roughly one-third of face keys remain unmatched (E1-only 311 / production-only 301 on T2; 1,361 / 1,449 on T6). Exact production patch digests are `0/857` T2 and `2/2,673` T6 |

### 2.5 Stage C — full-state candidate recall

| Field | Definition |
|---|---|
| Quantity | availability, `MetricSpace.CANDIDATE`, universe `frozen_reference_cells` |
| Fixed end-to-end denominator | `\|V\|` |
| Conditional denominator | `\|V_P\|` (`V ∩ run palette`), reported as its own row; candidate depth `K ∈ {1,2,4,8}` (`DEFAULT_CANDIDATE_DEPTHS`) |
| Metric rows | `candidate_available_kK = \|{v in V_P : truth(v) in ranked(v)[:K]}\| / \|V\|`; palette coverage `\|V_P\|/\|V\|`; the in-palette conditional; a state that is present but loses the argmin is **available** (a ceiling, not a label) |
| Oracle ceiling | truth state ∈ `candidate_costs[:K]`; a truth state outside the palette is unreachable at every `K` |
| Transitions / loss counts | a cell with no candidate record is charged to **geometry**; a beyond-palette state to the **palette**; gains/losses are counted only against comparable C checkpoints. `candidate_available_kK` may never exceed `palette_coverage` |
| Typed unavailable vs error | unavailable: `palette_unknown`, `proposals_unavailable`, `budget_exhausted`; error: a ranked candidate outside the run palette (a vocabulary contradiction) or a `CellScore` with palette evidence but no truth context must raise `AttributionRefusal` |
| Required synthetic contract test | `CellScore` recomputes `V_P` from the frozen truth snapshot and refuses a forged/subset/superset/foreign membership set; a `GOLD` out-of-palette ranked candidate refuses before and below the top-`K` cut; a state in `candidate_costs` that loses the argmin is available; `candidate_available_kK ≤ palette_coverage`; `palette_coverage == conditional_denominator == \|V_P\|` |
| Required real image-only evaluation | Route A currently measures C as `0/398` because the default 8-state palette does not contain T2 truth states; a normal-CLI **JAR palette** arm (a user option, not developer-gated) or a Route B result that reaches output is required before C carries any image-only information |
| Reusable now | `candidate_metric_rows`, `palette_metric_rows`, `palette_coverage`, `_candidate_availability`, `appearance_state_proposals.FaceStateProposal.candidate_costs` |
| Exact gap | Route B proposals are persisted only as `result_digest`; no image-only C row is reachable until G reaches output. The C path has **never been exercised on a genuine run** (the frozen Route B run has no Stage-7 tail) |

### 2.6 Stage E — exact-state ranking

| Field | Definition |
|---|---|
| Quantity | label, `MetricSpace.EXACT_STATE`, universe `frozen_reference_cells` |
| Fixed end-to-end denominator | `\|V\|` (primary; includes cells whose truth state is outside the run palette) |
| Conditional denominator | `\|V_P\|` as a separate vocabulary-diagnostic row |
| Metric rows | `raw_winner` exact state; `promoted` exact state; decoded `terminal` exact state; per-transition `gained`, `lost`, and the disjoint `geometry_lost` / `state_lost` partition |
| Oracle ceiling | aggregate-cost argmin for `raw_winner`; promotion eligibility (`export_ready and omitted_unknown == 0`) for `promoted`; exporter round trip for `terminal` |
| Transitions / loss counts | E→E **partitions** loss: `next = previous + gained − geometry_lost − state_lost`; `geometry_lost` and `state_lost` are disjoint; a cell is counted in exactly one. Never applied to a G, O, P or PC step |
| Typed unavailable vs error | unavailable: `record_absent`, `proposals_<reason>`, `promotion_<reason>`, `export_unavailable`, `empty_volume`; a carried-empty record is `checkpoint_reached=true` beside `reached=false` and a null metric. Error: an off-space partition, or a non-null metric on an unreached row, refuses |
| Required synthetic contract test | a route-level tail carrying both E endpoints with a geometry loss and a state loss that sum exactly to `lost`; a G step still carries no partition; a carried-empty promotion is reach `true` with no metric while an absent promotion is reach `null`; a decoded-terminal round trip scores the **decoded** volume, not the raw file |
| Required real image-only evaluation | Route A `palette_label` + decoded `terminal` (measured `0/398` and `0/398` today, with `V_P=0` so the conditional row is `null`); Route B `raw_winner`/`promoted`/`terminal` once output is reachable |
| Reusable now | `exact_state_metric_rows`, `label_metric_row`, `attribution_transition`/`cell_transition`, `_stagewise_visible_loss` label stages |
| Exact gap | on the default Route A palette, T2 truth states are outside the palette (`V_P = 0/398`), so E is **uninformative rather than wrong**; the decoded `terminal` record is not carried by the runner (only the pure builder is tested with it). No image-only run reaches Route B E |

### 2.7 Stage Placement

| Field | Definition |
|---|---|
| Quantity | geometry (placement/volume), universe `frozen_reference_cells` |
| Fixed end-to-end denominator | `\|V\|` |
| Conditional denominator | cells whose supporting plane was resolved (`plane_from_evidence`/`resolve_face_placement`) and cells merged into the volume |
| Metric rows | `placed` geometry recall; `volume_cell` geometry recall; `CellConflict` and `OmittedCell` counts; `placed → volume_cell` is a **G→G unpartitioned** step |
| Oracle ceiling | the placement universe under `placement_budget`; the volume support under `volume_budget`; **rectangular extent is never treated as occupancy** (holes preserved) |
| Transitions / loss counts | `gained`, `lost`, `net` with `geometry_lost=state_lost=null`; place/volume are checkpoint counts, not monotone ceilings |
| Typed unavailable vs error | unavailable: `placement_<reason>`, `volume_<reason>`; error: out-of-domain or non-integer coordinates under the permitted alignment refuse |
| Required synthetic contract test | drive the accepted synthetic recipe through the real `resolve_face_placement` / `build_plane_volume` public constructors (no fixtures, no JAR); assert placement/volume rows score coordinates only via `GeometryCellEvidence`, never `{coordinate: None}`, and that an occlusion-removed target is not counted as placed |
| Required real image-only evaluation | one genuine image-only Route B session that reaches placement/volume on the pinned fixture (does not exist today: the frozen run carries `placement_cells=0`, `volume_cells=0`, `promoted_cells=0`) |
| Reusable now | `reconstruction/plane_placement.py`, `reconstruction/plane_volume.py`, A2 `GeometryCellEvidence` + `score_aligned_geometry_coordinates`, `run_image_derived_frontier.py` |
| Exact gap | no genuine image-only run reaches placement; the A2 route-B placement/volume adapters are exercised by **synthetic real-constructor records only** |

### 2.8 Stage X — `.litematic` round trip

| Field | Definition |
|---|---|
| Quantity | label (terminal), universe `frozen_reference_cells` |
| Fixed end-to-end denominator | `\|V\|` |
| Conditional denominator | promoted cells (`checkpoint_reached` from the carried promotion record) |
| Metric rows | decoded-export exact-state recovery over `\|V\|`; normalized coordinate/state set equality between the run's decoded export and the same-run CLI export (Route A parity); `promoted → terminal` is an **E→E partitioned** step |
| Oracle ceiling | the exporter round trip itself (write → `load_litematic` → `validate_litematic`); asset-consistent data version |
| Transitions / loss counts | partitioned `geometry_lost`/`state_lost`, plus `empty_volume` / omitted-unknown counts that gate promotion eligibility |
| Typed unavailable vs error | unavailable: `export_unavailable`, `empty_volume`; error: a digest/validation failure, or a decoded volume that does not match the decoded same-run CLI volume, refuses |
| Required synthetic contract test | `export/compare.py` / `export/validation.py` round-trip tests; the A2 CLI-parity regression (`normalized_sets_equal`, 690 blocks, 0 state mismatches) rerun inside the new arm; a schema-3 envelope replay assertion |
| Required real image-only evaluation | Route A parity is accepted on the frozen frame; Route B needs a promoted run before its `.litematic` exists. Route A's export remains a **local-frame, non-promotable** artifact and may be scored only for vocabulary/label sanity |
| Reusable now | `LitematicExporter`, `export.compare.load_litematic`, `export.validation.validate_litematic`, `export.compare.compare_litematics`, A2 terminal adapter |
| Exact gap | Route B never reaches export; the A2 runner carries no decoded terminal record (the pure builder's terminal path is covered by synthetic tests only) |

### 2.9 Retained diagnostic — PC (pair correspondence, design-doc `P`)

`[F]` PC keeps its accepted universe `R_px` and is **not** the assignment's `P`. Accepted
evidence: DS-R5J/Canal Town candidate reach 345/1,213, selected 52/1,213; same-surface oracle
1,213/1,213, selection retains 865/1,213 but only 4/28 pairs solve; R5N raises surface-proxy
selection to 1,152/1,213 but produces 0/28 solves (`AGENT_HANDOFF.md:304,329`). It is assisted-camera
evaluation-only. `[R]` Keep PC as a separately labelled O/PC section in the ledger; never divide
it by `|V|`.

## 3. Ranked measured losses with provenance

Ordering rule: image-only first (product-relevant), supplied-camera separately, and *measured
magnitude* distinguished from *frontier stop*. `[F]` facts only.

### 3.1 Image-only, measured

| Rank | Loss | Magnitude | Provenance |
|---|---|---|---|
| 1 | **Route A visible geometry (G)** | recall **16/398 = 4.02%**; precision **16/687 = 2.33%**; visible exact state **0/398** | `image-only-attribution-a2-2026-10-03.md` §4/§6, artifact `.tmp-a2-route-a-tracked-20261003.json` SHA-256 `2AC71DF4ECF7931AA57098B9B4D80C58B2459C880CBB8512CEE3C712D94C7F87`; `AGENT_HANDOFF.md:303` calls it "the largest measured end-to-end loss so far" |
| 2 | **Route A palette coverage (C) / ranking (E)** | palette `V_P = 0/398`, so C `0/398`, E `0/398`; both conditional rows are `null` | same A2 §4. This is a **vocabulary ceiling**, not a ranking failure; a JAR palette arm has not been measured |
| 3 | **Route B frontier stop (K/G)** | terminal `geometry` / `geometry_unavailable` / `g_unavailable`, detail `plane_integer_residual`; frontier: entry reached, lattice assembled, geometry reached/unavailable, output not reached; completion 3/4, refusal 1/4 | `image-only-route-b-stage-loss-run-2026-10-04.md` §§1,7; artifact `attribution.json` SHA-256 `5CD4F15A62F6FB505DBC6A162643658EA48C3C9B738742D8BD99663D52DCF19B`, `route-b.mcscan` `FD508D9F0E20846314D3E48723617D3771151CC496457F8A1096B828C6EE30EC`. **No G/O/P/C/E magnitude is measured**; every accuracy row is typed unavailable |
| 4 | **Production-patch seam (P) — supplied-camera provenance, image-only unmeasured** | exact raw-patch digest parity **0/857** T2 and **2/2,673** T6; valid-mask digest **712/857 = 83.08%** / **1,989/2,673 = 74.41%**; ~1/3 of face keys unmatched | `texture-poc-e2e-pinned-sidecar-parity-2026-10-03.md` §E1/E2F; `AGENT_HANDOFF.md:308`. Not yet an image-only number |

### 3.2 Supplied-camera diagnostics (must stay separate; never a ceiling for either image-only route)

| Fixture | Visible exact | Geometry F1 | Palette | Support | Candidates | Raw winner | Smoothed | Placed |
|---|---|---|---|---|---|---|---|---|
| T2 default | 480/667 (71.96%) | 91.88% | 599 | 562 | 561 | 459 | 480 | 480 |
| T6 default | 350/2,007 (17.44%) | 85.39% | 1,920 | 393 | 393 | 329 | 350 | 350 |
| T6 no-prune (`palette_support=0.0`, gate-ineligible) | 511/2,007 (25.46%) | 85.39% | 1,920 | 1,920 | 788 | 470 | 519 | 511 |

`[F]` T2's largest state loss is candidate→raw winner (561→459, **102 cells**); smoothing has
66 gains / 45 losses. T6's largest default loss is palette→support (**1,527 cells**); with
support disabled the dominant loss moves to candidate generation. T6 top-8 in-palette
membership is **788/2,007 = 39.26%**; ordered in-palette recall is 793/1,920 (41.30%) at K=8,
850 (44.27%) at K=16, 932 (48.54%) at K=32. Provenance:
`visible-state-stage-attribution-2026-10-03.md` §§Results/Next; `visible-state-candidate-depth-2026-10-03.md`;
`AGENT_HANDOFF.md:80-90`. These are **raycast, supplied-camera, reference-assisted** diagnostics;
`AGENT_HANDOFF.md:76-79` and the design §1 forbid inheriting them into either image-only route.

### 3.3 Local-model corpus state (separate from both)

`[F]` E1 per-face existing-scorer top-1: T2 363/958 (37.89%) in-vocabulary, T6 522/3,813
(13.69%); top-8 76.93% / 26.49%. E3 exported 8,516 exact-state patches over **two usable
source groups** (Canal Town 3,636; Train Station 4,880), all `train`, `evaluation_only: true`,
`training_permitted: false`. E5 found 63 exact states, only **4 shared**, LOBO OOV **55.76% /
78.74%**. E4 audited 5,223 repeated-key pairs as a consistency proxy only. Provenance:
`texture-poc-e1-results-2026-10-03.md`; `texture-poc-e3-...` §7; `texture-poc-e5-...` §; `texture-poc-e4-...`.

## 4. Largest already-measured bottleneck and the exact next bounded experiment

### 4.1 Which bottleneck, and why

`[F]` The largest already-measured **image-only end-to-end** loss is **Route A visible geometry
(G): 16/398 recall, 0/398 exact state** (§3.1 rank 1). The Route B `plane_integer_residual` stop
is a *frontier*, not a magnitude, and the supplied-camera T6 support loss is disqualified from
the image-only funnel.

`[F]` **Existing artifacts alone cannot isolate whether the Route A G loss is dominated by the
image-derived camera object or by the facade engine.** The retained Route A artifact carries one
facade grid produced by one image-derived camera, `scale_known=False`, a fixed depth formula
(`facade.py:95-107`), and one alignment (Y-quarter-turn 1, translation `(18,13,22)`). No carried
record varies the camera object while holding the geometry generator fixed, so the camera and
geometry effects are confounded. `[F]` The A2 report itself states the observed alignment/score is
measured and "does not establish what any wider or unbounded search would find" (§8).

`[R]` Because the existing artifacts cannot isolate that confound, **one paired evaluation-only arm
(A0/A1)** on the same frozen frame is justified to measure oracle-camera headroom. This is **not a
fresh capture** and not a new screenshot; it reuses frozen inputs only. It is a sensitivity
measurement, not a causal attribution (see §4.2).

### 4.2 Exact next bounded experiment — `DS-STAGEWISE-K1` (evaluation-only, oracle-camera sensitivity / upper-bound arm)

**Question (single factor, camera object only):** with the facade generator, input image, mask,
palette, rows/columns/depth and permitted alignment family frozen, how much does substituting the
evaluator camera bundle raise Route A G recall/precision/F1 on the frozen frame? This measures
**oracle-camera headroom**, not a causal attribution: the arm swaps the whole camera object
(position, scale, intrinsics, orientation) at once, so a gain cannot be assigned to camera
estimation as the sole or dominant cause.

**Fixed inputs (existing, hashed):** `templates 2/2026-09-02_13.22.14.png`
(`eb22541171e030ff94d6839a6626e7160c69ae3a780f07aec0298322be665d46`), `templates 2/cameras.json`,
`templates 2/scene.json`, `templates 2/Template 2.litematic`
(`519034a6a439ab98145aa820792d5afc588b5d697ecad961c14c36cbef2f3416`), the tracked mask JSON for
the frame (`docs/evaluation_masks/templates-2/2026-09-02_13.22.14.json`), and
`docs/reports/evaluator-fixture-pins.json`. All paths verified present this revision.

**A0 comparability (verified this revision).** The A2 report does not bind its Route A result to its
implementation. The Route A runner hash *is* transitively recoverable:
`image-only-frontier-a1-2026-10-03.md:20` records
`scripts/evaluation/run_image_only_route_a.py` =
`eb3edaa19e7e166ac7eac1f1a626082c6e3f17b4fa369045435f602132911d4b`, and
`image-only-attribution-a2-2026-10-03.md` §1 states no A1 file was edited and that this hash still
matched. However, **no sealed A1/A2 artifact records the `reconstruction/facade.py` generator hash**
used by the A2 Route A run (`facade.py` is referenced by line, never hashed, in those reports). The
missing facade implementation hash is therefore **not recovered**, so the retained A2 `16/398` is
**not proven comparable** to a newly built A1. K1 must run a **fresh paired A0/A1** on the same frozen
screenshot under the current pinned code/settings, with A0 reproduced by the current code rather than
quoted from A2. The A2 `16/398` is reported only as historical provenance.

**Arms (identical settings, identical `V=398`, identical permitted alignment family; one run each, no
retry):**

| Arm | Camera object | Status |
|---|---|---|
| A0 | image-derived (`solve_image_cameras`), current pinned code/settings, production path | fresh run; no evaluator camera/reference/mask/truth input |
| A1 | **evaluator camera** for the same frame, feeding the same `reconstruct_facade` | fresh run; evaluation-only, labelled `reference-assisted`, never production/promotion |

**Procedure:**

1. Build A0 through the unmodified production Route A path (`solve_image_cameras` →
   `reconstruct_facade`) on the frozen frame, mask, palette and settings.
2. **Fully serialize and hash A0** (facade volume/coordinate-state set, camera row, settings, runner
   and `reconstruct_facade` implementation hashes) **before** loading any evaluator camera,
   reference, mask or truth. Record the production-before-pins-before-truth order sentinel.
3. Only then load the evaluator camera and build A1 through the **same** `reconstruct_facade`
   call with only the `CameraModel` object differing. A1 must be reachable only from the
   evaluation-only path.
4. Score A0 and A1 against the same frozen `V=398` under the same
   `evaluation.best_visible_alignment` family (integer translation + lattice Y-quarter-turn, no
   reflection, **no scale fit**), with identical mask, palette and rows/columns/depth.
5. A1's serialized artifact carries the `reference-assisted` label; A1 output is excluded from
   production and promotion.

**Reported metrics (descriptive; no invented threshold):** G recall, precision and F1 for each arm;
the exact correct/missing/false cell sets (counts plus digests) and the exact A0→A1 delta for each;
per-arm camera diagnostic row, alignment used, `V`, and settings/implementation hashes. The project's
existing **90% visible geometry-F1 gate is cited as a descriptive reference only** — K1 does not
change, pass or replace it and creates no new threshold.

**Predeclared interpretation (record before the run):**

- A large positive A0→A1 delta establishes **substantial oracle-camera headroom on this frame**. It
  does **not** establish that image-derived camera estimation is the sole or dominant cause, because
  the arm swaps the entire camera bundle while holding only the facade generator fixed.
- A near-zero delta is **consistent with** the facade engine limiting this frame, but does not prove
  it dominates: the bundle swap could be absorbed by the alignment search, and only one frame and one
  alignment family are measured.
- If A1 reaches the frozen visible ceiling under the permitted alignment family, report the ceiling
  and stop. Do **not** call K the sole limiter; the ceiling is a property of this frame and alignment
  family.
- No K-versus-G causal decomposition is claimed in any case. The single-factor statement is limited
  to "the camera object was the only object changed".

**Required contracts:** A1 must be reachable **only** from the evaluation runner; the production
adapter signature scan must show no camera/reference/alignment parameter changed; A0 is fully
serialized and hashed before any evaluator camera/reference/mask/truth is opened; the run records
production-before-pins-before-truth ordering (existing A2 runner order); A1's serialized record is
labelled `reference-assisted` and is out of promotion/production; the arm is one run per arm with no
retry.

**Exact code gap to build first (evaluation-only):** the Route A evaluation path drives
`solve_image_cameras` unconditionally (`scripts/evaluation/run_image_only_route_a.py:147`) before
calling `reconstruct_facade` (`:173`; verified to accept `camera: CameraModel | None` at
`reconstruction/facade.py:18-29`), while `scripts/evaluation/report_image_only_attribution.py` owns
the frozen-`V` scoring. Add an evaluation-only `--camera-source {image,evaluator}` seam (or a sibling
`run_image_only_route_a_camera_arm.py`) that builds and serializes A0 first, then constructs the
evaluator `CameraModel` for A1 and passes it through the same `reconstruct_facade` call. No
production function signature changes.

**Required synthetic tests:** the arm builder refuses a malformed/absent evaluator camera typed; the
production path cannot import the arm; A0 is serialized and hashed before any evaluator input object
is constructed; both arms differ only in the camera object (assert the image, mask, palette,
rows/columns/depth and `reconstruct_facade` call arguments are identical objects); an A1 record
without the `reference-assisted` label is refused.

**Stop conditions:** if the arm cannot be built without editing a production module or a production
signature, stop and report the exact seam instead of working around it. If the evaluator camera
cannot be obtained without opening a forbidden payload before A0 is serialized and hashed, stop.

### 4.3 Secondary bounded experiment on the production route — `DS-STAGEWISE-G1`

`[R]` In parallel (separate authorization), instrument the **first-failed geometry predicate**
at the Route B `lattice → geometry` transition on the already-frozen T2 screenshots, reusing the
R7/R8 output-neutral capture contract: per-axis integer resultants/residuals, active edges,
observations, and the exact `_STAGES` predicate that first returned non-ready
(`image_derived_geometry_orchestration.py:33-37`; frozen gates `max_shift_cells=32,
integer_tolerance=0.15, min_resultant=0.9, ambiguous_margin=0.02`). One run; no gate change; no
fresh capture; output digests must be byte-identical to the existing frozen run when capture is
disabled. This turns the Route B frontier into a G attribution without touching K.

## 5. Supplied-camera separation (hard rules)

`[R]` The following must be enforced by test, not convention:

1. Supplied-camera / reference-assisted runs are labelled `reference-assisted` (or
   `supplied-camera`) in every serialized record and report; their rows may never be pooled with
   image-only rows and may never be quoted as a ceiling, baseline or stage for an image-only
   route (`AGENT_HANDOFF.md:76-79`; design §1).
2. No production module (`pipeline/`, `reconstruction/`, `cli.py`, `project/`, `export/`) may
   accept, resolve, import or construct a reference, visible reference, alignment, evaluator
   camera, manifest or frozen mask. Only the evaluation runner may load them, and only after the
   production call returns (A2 `production < pins < truth` sentinel test, already accepted).
3. The raycast six-checkpoint funnel (`available_palette`, `support`, `candidate`,
   `raw_winner`, `post_smoothing`, `placed`) is the **supplied-camera** scorer. Its stage names
   must never be imported into a Route A/B ledger; Route B's honest analogue is
   placement → volume → ownership → palette/candidates/ranking → promotion/export
   (design §2.2 "one deliberate simplification").
4. A K arm that substitutes an evaluator camera is a reference-assisted diagnostic and must be
   labelled as such in its artifact; it is not image-only evidence.

## 6. Local-model entry criteria, baseline, splits, factory role, resources

`[F]` No model may be trained today. E3 artifacts are `evaluation_only: true`,
`training_permitted: false`; T2/T6 are evaluation-only; there is no approved training-data
contract. The assignment authorizes planning only, and this audit changes no dependency.

### 6.1 Entry criteria (all must hold; necessary, not sufficient)

1. ≥ **6 admitted independent builds** (unique `(world_identity, scene_group)`, one structure per
   world); recommended 12 (`texture-poc-e6-...` §§4-5).
2. ≥ **10 exact states**, each with **≥ 50 admitted examples in ≥ 3 independent builds**
   (`valid_fraction ≥ 0.75`); two builds is explicitly insufficient (E6 §13; E5's falsification
   criterion).
3. No evaluated held-out build exceeds **50% OOV** examples under the frozen core; held-out OOV
   is abstained on, never scored as a core-class error (E6 §10, §13).
4. Per-build/per-capture quotas: ≥3 distinct captures per state, ≥1,000 admitted examples per
   build, ≥3 capture sessions per build, ≥16 finalized HUD-hidden frames per build, ≥10 shared
   material families (E6 §6).
5. World-grouped **train/dev/calibration** folds plus an immutable world-held-out **test** set;
   fold unit = independent build; primary protocol = leave-one-build-out; `dev`/calibration in a
   separately sealed artifact (E6 §10.1-10.2, §10.7).
6. **Label purity quantified** (region-clipping/omitted-occluder contamination), which E4's
   consistency proxy explicitly does **not** establish (E5 §; E4 §6).
7. **Production-patch parity**: byte-identical patches to the production `FaceObservation` for
   the same `(cell, slot, frame)` join, by digest on a held-out join set — not mean-pixel
   similarity (design §7.2; E2F/E2E).
8. A separate lead-approved **training-data contract/exporter** exists that admits only
   designated train folds and excludes T2/T6 (E6 §10.7, §11). It does **not** exist today.

### 6.2 Same-patch `PaletteIndex` baseline (the thing any model must beat)

`[F]` Compare on the **identical frozen patch list** `(frame_id, coordinate, slot, patch bytes,
valid, coverage, weight)`, scored once and reused by both arms. Baseline rows (E1, per-face,
full canonical palette, truth-in-vocabulary denominators):

| Fixture | In-vocabulary patches | Top-1 | Top-8 | Top-16 | Top-32 |
|---|---|---|---|---|---|
| T2 | 958/1,168 | 363 (37.89%) | 737 (76.93%) | 745 (77.77%) | 754 (78.71%) |
| T6 | 3,813/4,025 | 522 (13.69%) | 1,010 (26.49%) | 1,068 (28.01%) | 1,078 (28.27%) |

`[R]` Report per-build and per-capture, not one folded number. Compare at the same `K` values
used by production (`_top_candidates` keep=8; `stage_trace_candidate_limit` sweep to 32). A
classifier gain is **not** an end-to-end gain: a ranker cannot recover a cell whose truth state
the pipeline never generated, and cannot fix the geometry frontier. Prefer a zero-able prior
(`PaletteIndex.with_prior`) so the model arm ablates exactly to baseline.

### 6.3 Build-disjoint split requirements

`[F]` Today there are 3 topology groups / 3 source builds over 5 finalized datasets / 26
screenshots, and **every** index split is `train` (recomputed buckets 13/67/34 vs
`TRAIN_CUTOFF = 70`), so no dev/test fold can appear without new structures
(`texture-poc-data-audit-2026-10-03.md` §2). `[R]` Use `build_group_manifest` /
`check_split_leakage` / `check_fold_leakage` (`evaluation_pair_correspondence.py`) as the
leakage gate, plus a new assertion that no `world_identity` appears in two folds (the code keys
only on topology). Freeze folds before any scored capture; never change a fold after a result.
Report LOO as **exploratory** until ≥6/≥12 builds with the §6.1 quotas exist.

### 6.4 Data-generation factory role

`[F]` The factory is a **capture** factory, not a world generator: inventory existing saves
(`prepare_capture_dataset.py --list` / `inventory_saves`) → plan (`plan_capture_batch.py`) →
stage/arm (`prepare_capture_dataset.py`) → capture (`run_capture_batch.py --execute`, pinned
Fabric mod, disposable Prism worker, hidden desktop, muted/no-input) → finalize
(`finalize_capture_dataset.py`) → export labels
(`export_factory_texture_patch_dataset.py`, E3) → audit (`audit_factory_texture_patch_consistency.py`,
E4; E5-style support). Manual authoring is a bounded fallback only for a measured gap, one
structure per world. Region caps and identity rules: `dataset/schemas.py` (`MAX_REGION_EDGE=512`,
`MAX_REGION_VOLUME=250_000`), `dataset/splits.py` (`TRAIN_CUTOFF=70`,
`volume_topology`/`scene_group`), `dataset/batch.py` (`build_identity_for`).
`[R]` The factory produces **evaluation-only** artifacts until the §6.1 training-data contract
exists; it must not be repurposed as a generator, and no threshold/class list/fold may change
after a scored result.

### 6.5 Resource, runtime and abstention metrics (required for any model arm)

`[R]` Report, separately and in this order: (a) per-patch CPU latency (median/p90) and peak
sampled working set for the model arm versus `PaletteIndex` on the identical patch list, with
the current pinned reconstruction time as the end-to-end baseline; (b) abstention/calibration —
coverage-vs-risk, expected calibration error, and the abstain rate mapped onto the **existing**
`ABSTENTION_REASONS` (`no_observation`, `empty_palette`, `weak_margin`, `high_cost`,
`conflicting_faces`, `budget` in `appearance_state_proposals.py:104-111`); (c) memory/transfer
provenance for any GPU path (the GPU integration remains planned and CPU is the correctness
reference); (d) unchanged end-to-end gates: ≥80% visible exact state and ≥90% visible geometry
F1 measured separately, with any palette/support override making a run gate-ineligible.
`[F]` The current environment has NumPy/Pillow but no PyTorch/scikit-learn/ONNX, so any model
option carries an install/dependency decision that is itself a lead approval — out of scope
here.

## 7. Reuse map and exact code/test gaps before implementation

### 7.1 Reuse (no new code needed for the contracts)

| Need | Existing artifact |
|---|---|
| Frontier checkpoints + typed unavailable | `src/mc_photogrammetry/evaluation_image_only_frontier.py` (`ROUTE_A_STAGES`, `ROUTE_B_STAGES`, `frontier_unavailable`, `first_observable_failure`) |
| G/O/PC/C/E rows, universes, transitions, palette ceiling | `src/mc_photogrammetry/evaluation_image_only_attribution.py` (`MetricSpace`, `UNIVERSE_OF_SPACE`, `score_aligned_cells`, `score_aligned_geometry_coordinates`, `candidate_metric_rows`, `palette_metric_rows`, `exact_state_metric_rows`, `pair_metric_rows`, `cell_transition`, `AttributionRefusal`) |
| Route A/B replay + production-before-truth ordering + mask pins | `scripts/evaluation/report_image_only_attribution.py`, `run_image_only_route_a.py`, `run_image_derived_frontier.py` |
| Visible metrics + permitted alignment | `src/mc_photogrammetry/evaluation.py` (`visible_metrics`, `best_visible_alignment`, `visible_breakdowns`) |
| Supplied-camera six-checkpoint funnel (separate) | `scripts/benchmark_templates.py` (`STAGE_TRACE_STAGES`, `_stagewise_visible_loss`, `_candidate_availability`, `_stagewise_unavailable`) |
| Ownership truth / patch labels | `evaluation_pair_correspondence.raycast_surface`, `export/visibility.visible_cells`, `export/attribution.render_result`, `reconstruction/face_sampling.py` |
| Patch parity | `scripts/evaluation/replay_texture_observation_trace.py` (E2B/D/F), `evaluate_texture_patch_labels.py` (E1), `export_factory_texture_patch_dataset.py` (E3) |
| Splits/leakage | `evaluation_pair_correspondence.build_group_manifest`/`check_split_leakage`/`check_fold_leakage`; `dataset/splits.py` |
| Export round trip | `export/compare.py`, `export/validation.py`, `LitematicExporter` |

### 7.2 Exact gaps (each is evaluation-only unless stated)

1. **K→G join** — no per-frame camera-error→cell-placement join; no evaluator-camera arm in the
   Route A runner. New evaluation-only seam (§4.2).
2. **Route B upstream G vectors** — `admitted`/`camera`/`reading`/`signature`/`match`/`solved`
   are digest-only; no per-coordinate export. New evaluation-only exporter or an extension of
   the frozen diagnostic runner (`reconstruction/frozen_image_runner.py`).
3. **O ownership artifact** — no evaluator `(cell, slot, frame)` surface-ownership export, so O
   is an explicitly empty section. New evaluation-only export + join.
4. **Image-only production patch sidecar** — E2C/E2E are supplied-camera; the image-only route
   emits no production observation sidecar, and ~1/3 of face keys are unmatched. New
   output-neutral capture on the image-only route + a digest-bound join to E1/E3 labels.
5. **Route B proposal persistence** — `FaceStateProposal.candidate_costs` are in memory only;
   C/E rows need an evaluation-only carrier (the proposal digest alone is insufficient).
6. **Route A JAR-palette arm** — the default 8-state palette gives `V_P=0/398`; a normal-CLI JAR
   palette arm (a normal user option, not developer-gated) is required before Route A C/E is
   informative.
7. **Decoded terminal record in the runner** — `route_a_terminal_adapter`/X currently rely on
   the pure builder; the runner does not carry the decoded `.litematic` volume.
8. **Synthetic contract tests** — add: O-in-wrong-space refusal; O absent ≠ 0;
   `candidate_available_kK ≤ palette_coverage`; E partition on both E steps; carried-empty vs
   absent promotion reach; K arm single-factor (only the camera differs); P shape/dtype/join
   refusals on the image-only key set; X decoded-vs-raw refusal.
9. **No production change is required for any of 1-8.** If any gap appears to need a production
   signature, schema, gate or threshold change, stop and report the seam (design §5 "Deferred").

## 8. Execution order, ownership and stop conditions

`[R]` Recommended order (smallest blast radius, each with its own authorization and one pinned
run):

1. **Contract-only** (no run): add the §7.2-8 synthetic tests to the existing evaluation test
   modules; prove O/PC/C/E universe separation and typed-null behaviour. Exit: focused tests
   green, `py_compile`, no production diff.
2. **DS-STAGEWISE-K1** (§4.2): the Route A oracle-camera sensitivity arm. Exit: one fresh paired
   A0/A1 run under current pinned code, A0 fully serialized and hashed before any evaluator read,
   both arms scored on the same frozen `V=398`, and recall/precision/F1 with exact cell-set deltas
   recorded. Describes headroom only; no causal or sole-limiter claim.
3. **DS-STAGEWISE-G1** (§4.3): Route B first-failed geometry predicate instrumentation. Exit:
   one output-neutral run, per-axis resultants/residuals, unchanged production digests.
4. **O/P exports** (§7.2-3/4): evaluation-only ownership and image-only sidecar capture. Exit:
   O and P become measured or explicitly typed unavailable with a named missing seam.
5. **Route A JAR-palette C/E arm** (§7.2-6): exit C/E informative.
6. **Local model**: only after §6.1 gates clear and §6.2 baseline is reproduced on the identical
   patch list. Until then, do not train, do not add dependencies.

Every tranche: stage named paths only; `git diff --check`; focused tests + `py_compile`; record
settings/source revision/input hashes; preserve prior artifacts. Stop and report rather than
relax a denominator, invent a coordinate, fit scale, or read evaluator data into inference.

## 9. Non-claims and prohibitions honored

- No image-only accuracy is claimed anywhere in this report; the only image-only numbers are
  the already-sealed Route A `16/398` and the Route B frontier, both quoted with provenance.
- The supplied-camera T2/T6 numbers are labelled as such and are **not** a ceiling, baseline or
  stage for either image-only route.
- Route A's `V_P = 0/398` is a measured vocabulary ceiling, not a ranking result. Route B's
  `plane_integer_residual` is a first-stop frontier, not a percentage loss.
- **K1 is a sensitivity/upper-bound arm, not causal attribution.** Substituting the evaluator camera
  swaps the whole camera object (position, scale, intrinsics, orientation) at once, so no A0→A1 delta
  identifies image-derived camera estimation as the sole or dominant cause. A0 and A1 are re-run fresh
  because no sealed A1/A2 artifact binds `reconstruction/facade.py` to the retained A2 A0 result (the
  Route A runner hash is transitively recoverable from `image-only-frontier-a1-2026-10-03.md:20`).
- **A1 is `reference-assisted` and out of promotion.** No K1 output is a production input, a promotion
  artifact or image-only evidence; the existing 90% visible geometry-F1 gate is cited descriptively
  only and no new threshold is created.
- No fresh capture is proposed; §4.2 reuses frozen inputs and is justified only because §4.1 shows
  the retained artifacts cannot isolate the camera/geometry confound.
- This audit performed no training, no dependency install, no source/test/capture/dataset/schema
  edit, no product/evaluator/solver run, and opened no screenshot, reference, camera, mask, JAR
  or truth payload. Only this report and the `AGENT_COMMS.md` completion entry were written.
- **Stop for lead review.**
