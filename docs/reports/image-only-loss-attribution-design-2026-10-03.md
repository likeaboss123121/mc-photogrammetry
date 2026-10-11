# Image-only loss attribution design (read-only) — 2026-10-03

Task: `agent swarm instantiation template/tasks/active/image-only-loss-attribution-design-2026-10-03.md`
(`id: image-only-loss-attribution-design-2026-10-03`, kind: read-only design audit; owner: DeepSeek;
lead review required).
Status: **READ-ONLY DESIGN REVISED per the lead-review correction addendum (2026-10-03). This report
is the only file written. Stop for lead review.**

Revision note: this pass addresses the addendum's three blockers — (1) route A ownership: A1 now
names route-specific adapters plus a Route-A evaluation runner and A2 scores both routes; (2)
accounting units: every metric names a fixed universe and transitions are restricted to comparable
success sets in one universe; (3) the local-model support gate is replaced with the accepted exact
E6 requirements. The previously accepted corrections (Route A scale error reduces score; all-`V`
primary exact-state denominator; ≥1-pixel primary visibility; corrected Path B order; camera
diagnostics; completion/refusal rates; evaluator-only truth; decoded terminal export) are preserved.

No source, test, config, schema, handoff, comms, task or fixture was edited by the DeepSeek worker;
this report was its sole write target and was then revised by the lead during review. No
reconstruction, benchmark, fixture, capture, scan, test or model job was run; no screenshot,
reference, manifest, mask (file), JAR or save content was opened; nothing was
trained, downloaded or installed; no shell nesting, background work, reparse creation, recursive
deletion, encoded command, Defender/system change or commit was performed. `AGENT_COMMS.md` was not
read (per the task instruction). Every statement below is a static source/`.md` reading or a
recommendation; facts are marked **[F]** and recommendations **[R]**. Line anchors are the
working-tree readings at audit time and can drift.

Read for this design: `docs/reports/image-only-stage-loss-ledger-audit-2026-10-03.md` (the prior
audit; cited as "AUDIT"); `docs/reports/visible-state-stage-attribution-2026-10-03.md`;
`docs/reports/image-derived-graph-stage-v6-2026-10-03.md`;
`docs/reports/DS-R4-M4-production-integration-audit-2026-10-03.md` (including the lead erratum at
281-287); `AGENT_HANDOFF.md` §5 (264-321);
`docs/reports/texture-poc-e3-factory-patch-dataset-export-2026-10-03.md`,
`docs/reports/texture-poc-e4-cross-view-label-consistency-2026-10-03.md`,
`docs/reports/texture-poc-e5-factory-state-support-2026-10-03.md`,
`docs/reports/texture-poc-e6-balanced-build-acquisition-design-2026-10-03.md`,
`docs/reports/texture-poc-e2f-numeric-patch-similarity-2026-10-03.md`. Source inspected under
`src/mc_photogrammetry/`: `cli.py`, `pipeline/image_only.py`, `pipeline/image_derived_production.py`,
`pipeline/image_derived_production_record.py`, `pipeline/image_derived_entry.py`,
`pipeline/image_derived_lattice_measure.py`, `pipeline/image_derived_geometry_orchestration.py`
(head), `pipeline/image_derived_graph.py` (grep), `pipeline/image_derived_output.py`,
`pipeline/image_derived_output_policy.py`, `reconstruction/facade.py`,
`reconstruction/calibration.py` (276-316), `reconstruction/appearance_state_proposals.py`,
`reconstruction/state_promotion.py`, `evaluation.py`, `evaluation_pair_correspondence.py`
(`metric_value` 678-684), `export/compare.py` (names only), `blocks/appearance.py` (86-95),
`scripts/benchmark_templates.py` (97-172, 1124-1510). No hash was recomputed.

---

## 1. Two entry routes: which is the user-facing target, and what is scoreable

**[F] There are exactly two image-only user routes, and they are separate programs with different
geometry engines.** The prior audit established this (AUDIT §2, lines 58-123) and it is unchanged in
the current tree:

* **Route A — single-view `reconstruct` (the normal-mode default).** `cli.py` 498-514 rejects
  `--horizon/--camera-manifest/--scene` in normal mode; 517-518 loads one image and a GUI mask;
  536-551 in normal mode calls `solve_image_cameras` (`pipeline/image_only.py:52`) and adapts with
  `camera_model_from_estimate(..., position=LOCAL_FRAME_ORIGIN)`; 561-571 calls `reconstruct_facade`;
  574-588 saves the project and exports one `.litematic`. `reconstruct_facade` builds a fixed
  `columns × rows` (default 32×32) image grid; with a camera but no scene it assigns
  `4.0 + (1.0 - normalized_depth) * 28.0` blocks and rounds the unprojected point to integers
  (`reconstruction/facade.py:74-118`). `LOCAL_FRAME_ORIGIN = (0,0,0)` and `scale_known=False` are
  documented as the only admissible values (`pipeline/image_only.py:17-25,39-45`;
  `reconstruction/calibration.py:276,312-316`).
* **Route B — multi-view `image-derived-output` / `image-derived-deliver`.** `cli.py` 435-496
  dispatches into `derive_image_output_from_paths` (`pipeline/image_derived_production.py:215`),
  whose stage order is entry → lattice → geometry → output (245-300). The acceptance clause here is
  only that at least one still image was discovered (`pipeline/image_derived_entry.py:437-444`); the
  route is a pair-matching multi-view chain, so it is meaningful only from two frames up. Unlike
  `reconstruct-multiview` (which fails closed in normal mode, `cli.py:601-605`),
  `image-derived-output` and `image-derived-deliver` have no `mode` gate
  (parsers at `cli.py:206-246`; dispatch at 435-496).

**[R] Route B is measured first because it is the intended multiview reconstruction path and
already exposes a typed funnel; Route A remains a required, separately measured normal-mode
diagnostic.** Route A is what a normal user currently runs by default (one screenshot, default
settings), but it emits only a local-frame `.mcscan`/`.litematic` without typed per-stage records.
Route B has typed statuses, digests, and (for a promoted result) a full `PromotionResult`; the
**recorded** genuine runs do not reach promotion (§6), and it needs more than one image. Route A
also chooses and exports states (its palette labels, A3/A4), so it is not a vocabulary-only route
and must not be omitted from the measurement.

**[F] What is presently scoreable per route:**

| Route | Artifact a run emits | Per-stage evidence carried | Scoreable against a visible-truth denominator today |
|---|---|---|---|
| A | `.mcscan` (schema 2) + `.litematic` (`cli.py` 574-588) | none; no typed status, no digest envelope | **Yes, geometry and labels.** The facade emits integer block-lattice cells, so G is scored under one permitted global integer translation + lattice Y-quarter-turn; unresolved metric scale **reduces** the score and is reported, it does not make G unscorable (`cli.py` 577-595). E rows are the run's palette labels; `V ∩ palette` is the conditional diagnostic. |
| B | schema-3 `ScanProject` (`pipeline/image_derived_production.py:199-212`) carrying `image_derived_session` + a compact terminal envelope | stage, status, reason, detail, four stage digests, `session_sha256`, `volume_sha256`, `refusal_sha256`, and (only if promoted) the serialized `PromotionResult` (`pipeline/image_derived_production_record.py:107-112`) | **Not on recorded runs.** End-to-end visible geometry/exact-state is undefined because the recorded genuine runs stop before geometry assembly and promotion (§6); the labelled spaces are defined, but no recorded run reaches them. Policy/status counts are not accuracy. |

**[F] Gate-policy counts must not be read as accuracy.** The closed status enums
(`promoted | review | refused | geometry_unavailable | budget_exhausted | unavailable`,
`pipeline/image_derived_production_record.py:97-104`) and the stage-reason strings
(`"<stage>_<status>"`, `pipeline/image_derived_geometry_orchestration.py:95-97`) are **policy
outcomes**: they say what the pipeline refused or abstained on, not how many visible reference
cells were recovered. The v6 report already states this explicitly for its own per-stage counts
(AUDIT §3, line 159-160).

**[F] The supplied-camera raycast diagnostics must not be joined to either image-only score.**
The only published visible-truth percentages (T2 480/667 exact 71.96%, geometry F1 91.88%; T6
350/2,007 exact 17.44%, geometry F1 85.39%) are supplied-camera, reference-assisted, reviewed-mask
diagnostics (`docs/reports/visible-state-stage-attribution-2026-10-03.md:5,13-14`;
`AGENT_HANDOFF.md:303`). They come from `reconstruct_scene_surfaces` in `reconstruction/raycast.py`,
which requires a developer-mode camera manifest and scene manifest (`cli.py:642-648`; AUDIT §1,
47-54). **Neither route A nor route B may inherit these numbers as a stage, a ceiling or a
baseline.**

---

## 2. Stage checkpoints from admission to export

### 2.1 Five labelled spaces, never summed or interchanged

**[F]** The handoff already fixes the metric definitions (`AGENT_HANDOFF.md:266-283`, especially
267-278) and the scorecard requirements (`295-299`). This design keeps five labelled
numerator/denominator spaces strictly separate, each named explicitly wherever a number appears:
geometry **G**, face/pixel ownership **O**, pair correspondence **P**, candidate/palette **C**, and
exact-state **E**. Adding, interchanging or silently relabelling them is the failure the scorecard
exists to prevent.

**[R] Every metric declares a fixed universe, and no arithmetic crosses universes.** The universe
is the identity set the numerator and denominator are drawn from; a number is meaningless without
it.

| Space | Fixed universe | Common all-`V` coverage? |
|---|---|---|
| G — geometry | **frozen reference cells** `V` (primary ≥1 usable pixel) | yes — this is the cell universe |
| C — candidate/palette | **frozen reference cells** `V` (same cell universe as G and E) | yes — availability over the same cells |
| E — exact state | **frozen reference cells** `V` (primary all-`V`); `V ∩ palette` is a conditional row in the same universe | yes — the primary denominator is all of `V` |
| O — face/pixel ownership | **frozen frame-pixels**, or explicitly defined `(cell, slot, frame)` **face observations** | no — a different, non-cell universe |
| P — pair correspondence | **frozen frame-pair / pixel-pair identities** | no — a different, non-cell universe |
| camera (A1/B2) | none — diagnostic | no — no cell denominator and **no cell-loss equation** |

Common all-`V` coverage therefore applies **only to the defined cell metrics** (G, C, E). An O or
P numerator is never divided by `|V|` and never added to a G/C/E count; a G/C/E count is never
divided by a frame-pixel or pair count. Any cell-level aggregation that is not the primary metric
(e.g. an occupancy-only variant, or the strict-pixel-threshold variant) is **derived** and reported
as its own labelled aggregate over the same cell universe, never as a transition.

1. **G — geometry / occupancy.** `V = reference coordinates visible through usable image pixels`
   under a frozen, independent mask. **The primary visibility threshold is ≥1 usable pixel**
   (`AGENT_HANDOFF.md:267,275-276`); stricter thresholds (e.g. `minimum_pixels > 1`) are reported as
   **separate labelled rows** and never replace the primary ≥1-pixel denominator. Visible geometry
   recall `= |C∩V|/|V|`; precision `= |C∩V|/(|C∩V| + |C−R|)` (`268-271`); phantoms are `C−R`.
   Route A's candidate coordinates are integer block-lattice cells, so G is scored under one
   permitted global integer translation + lattice Y-quarter-turn, and the unresolved metric scale is
   a reported score-reducing caveat, not an `unavailable` case (§2.3).
2. **O — face / pixel ownership.** Which frozen visible pixels the run actually assigned to a face
   or slot, per `(cell, slot, frame)`, scored against evaluator-derived `(coordinate, slot)` surface
   ownership (`AGENT_HANDOFF.md:298`). **Its universe is frozen frame-pixels / explicitly defined
   `(cell, slot, frame)` face observations, not reference cells, so an O row never takes the all-`V`
   denominator.** O is labelled separately from G and P; an ownership row never counts as geometry
   recall, and a P row never counts as ownership.
3. **P — pair correspondence.** Jointly visible, strict same-surface truth pixel pairs with unique
   `(left pixel, right pixel)` identity (`AGENT_HANDOFF.md:298`; `evaluation_pair_correspondence.py`,
   e.g. `metric_value` 678-684). **Its universe is frozen frame-pair / pixel-pair identities, not
   reference cells.** **P never enters reconstruction and is never added to or subtracted from G,
   O, C or E.** On route B the pair stage is the connection step inside entry
   (`joint_matching.match_and_solve_pair`; AUDIT §2, 82-86).
4. **C — candidate / palette availability.** Whether the truth state is reachable at all from the
   run's own inputs: palette membership and candidate depth `K ∈ {1,2,4,8}`
   (`reconstruction/appearance_state_proposals.py:129-243`). **Its universe is the same frozen
   reference cells as G and E**, so its all-`V` coverage is directly comparable to G and E. On route
   B the palette is **always** the exact 8-state `DEFAULT_PALETTE` (`pipeline/image_derived_output_policy.py:116-117,179`;
   states at `blocks/appearance.py:86-95`), so the palette ceiling is a measured vocabulary ceiling,
   not an assumption. On route A the palette is the default 8-state palette **or** a
   normal-CLI-supplied JAR palette (`cli.py:552-560`) — a JAR palette on the normal CLI is an
   ordinary user option and is **not developer-gated** — so route A's C rows are reported per-run.
   C is an availability space, never accuracy.
5. **E — exact-state.** **The primary E denominator is all frozen visible `V`** — every visible
   cell, whether or not its truth state is in the run palette — so E shares the G/C cell universe.
   `V ∩ palette` is only a
   **conditional vocabulary diagnostic**, reported as its own labelled row against the palette
   ceiling; it must never be substituted for the primary all-`V` denominator
   (`AGENT_HANDOFF.md:279-280`). Route A also selects and exports states (A3/A4), so E rows exist on
   both routes.

**[R] Per-checkpoint record contract.** Every checkpoint row reports, for its own labelled space:
the **record/source**, its **fixed universe** (§2.1), the **denominator**, the **oracle ceiling**
(the best attainable count given *that stage's own inputs and policy* — palette membership,
candidate depth `K`, the retained candidate set, a truth-filtered candidate graph), the
**conditional denominator** (units that reached the prior checkpoint **in the same universe** and
are in this denominator), the **common-denominator metric** (for cell metrics, true-state coverage
over `|V|`; O and P report over their own frame-pixel / pixel-pair universes instead and never use
`|V|`), **gains/losses** for comparable transitions in the same universe (`gained`, `lost`, `net`;
for cell-level exact-state transitions only, disjoint `geometry_lost` and `state_lost`), and a
**typed unavailable case**.

**[R] Transitions are defined only between comparable success sets in the same universe.**
`next = previous + gained − lost` is an identity **inside one universe** (the G/C/E reference-cell
sets, the O frame-pixel / face-observation set, or the P frame-pair / pixel-pair set). Nothing is
ever subtracted across G, O, P, C and E, and no ratio is taken across two universes. For the
cell-level exact-state transition the loss term is partitioned into **disjoint** cell sets:
`geometry_lost` (the cell leaves the scored success set because its geometry/placement was lost)
and `state_lost` (geometry survives but the exact state is lost). A cell is counted in exactly one
of them; the sets must not overlap, and a transition whose arithmetic does not reconcile is a typed
error, not a rounding note. **Camera checkpoints (A1/B2) are diagnostics: they have no cell-loss
equation and contribute no term to any transition.** A derived cell aggregation over the same
universe is a separate labelled row, never a transition. A checkpoint with no carried record is
`available: false` with a typed `unavailable_reason` and **contributes no zero** — the same
fail-closed rule `scripts/benchmark_templates.py:_stagewise_unavailable` (1460-1509) already
implements and `:1340` currently violates only at the *rate* level. Per run and per cohort the
ledger additionally reports **completion and refusal rates**: the labelled share of runs that
reached each checkpoint and the share that stopped there — policy rates, never accuracy.

### 2.2 Route B checkpoint table (pipeline order)

**[F]** Record sources are the in-memory accepted records already returned by the existing stage
functions; only the terminal envelope is durable today (AUDIT §3 and §6, 290-301). "Persisted?"
means inside the schema-3 terminal.

| # | Checkpoint | Quantity | Record / source (fact) | Denominator | Oracle ceiling (stage's own inputs) | Typed unavailable case | Persisted? |
|---|---|---|---|---|---|---|---|
| B0 | `entry` | P | `ImageDerivedEntryResult.graph` (`pipeline/image_derived_entry.py:388`; terminal always `geometry_unavailable` 150-157) | **P universe: frozen frame-pair / pixel-pair identities** — jointly visible strict same-surface truth pairs (evaluator) | admitted matches under truth-filtered graph | `entry_refused` / `graph_not_ok` / `pose_graph_not_ok` (`image_derived_production.py:253-256`) | graph digest only |
| B1 | `admitted` | G | per-frame presence from `session.frames` + retained usable masks (`image_derived_lattice_measure.py:371-431`; masks re-derived at `image_derived_output.py:250-261`) | all frozen visible `V` (primary, ≥1 usable pixel); frozen evaluator mask ∩ frame set (evaluator-only) | cells that any admitted frame's mask can support | `no_admitted_frames` / `mask_unavailable` | no |
| B2 | `camera` | camera diagnostic | `CameraEstimate` intrinsics/yaw/pitch/roll/residual; `scale_known=False`, `position=None` (`reconstruction/calibration.py:297-316`) | none — diagnostic; **no cell denominator, no cell-loss equation** | **None.** Camera residual/projectability are diagnostics; they are **not** visible-cell recovery and **not** an oracle ceiling | `pose_unavailable` / `camera_abstained` | session record |
| B3 | `reading` | G | `propose_image_planes_v2` + `LatticeAbstention(frame, reason, detail)`; `PlaneReading` per frame/axis/side (`image_derived_lattice_measure.py:438-466`) | all frozen visible `V` | cells on any retained reading/alternative | `no_lattice_reading` / `provenance_incomplete` (266-270) | digest only |
| B4 | `signature` | G | `image_derived_signatures` per-domain cells/interior/coverage arrays | all frozen visible `V` | cells covered by an accepted signature | `signature_absent` | digest only |
| B5 | `match` | G | `image_derived_graph.PlaneMatch` + refusals; pre→post-admission funnel in the v6 report | all frozen visible `V`, and (evaluator-only) truth-filtered candidate graph as the **oracle** ceiling | admitted matches under the same gates | `graph_unavailable` | digest only |
| B6 | `solved` | G | `plane_graph.solve_component` `ComponentSolution` (resultants/residuals/observations/`scale_known`/`blocks_per_period`) | all frozen visible `V` | placements attainable under the solver's own policy | `component_unavailable` / `solver_abstained` | digest only |
| B7 | `placed` | G | `ResolvedFacePlacement`/`ResolvedFace`/`PlacedCell` (`plane_placement`) | all frozen visible `V` | placement universe under `placement_budget` | `placement_<reason>` (`image_derived_output.py:237-239`) | digest; volume only if promoted |
| B8 | `volume_cell` | G | `PlaneVolumeResult.cells` + `CellConflict` + `OmittedCell` (`reconstruction/plane_volume.py`) | all frozen visible `V` | volume support under `volume_budget` | `volume_<reason>` (243-245) | digest; volume only if promoted |
| B9 | `observed_face` | O (per `(cell, slot, frame)`), with P and E rows reported separately | `PlaneFaceObservationResult.observations`/`reviews` + `usable_coverage`; closed review enum (`reconstruction/plane_face_observation.py:134-142`) | **O universe = frozen frame-pixels / explicitly defined `(cell, slot, frame)` face observations** (never `|V|`); P rows use the pixel-pair universe and E rows the all-`V` cell universe — **reported separately, never combined or subtracted** | face-observation budget | `face_observations_<reason>` (267-270) | digest only |
| B10 | `palette` | C (availability) | the run's palette (`image_derived_output_policy.py:179`) | all frozen visible `V`, with `V ∩ palette` as the conditional diagnostic | truth state ∈ `DEFAULT_PALETTE` (8 states) | `palette_unknown` | policy digest |
| B11 | `candidate_k` | C (availability) | `FaceStateProposal.candidate_costs`, `best_cost`, `runner_up_*`, `margin`, `slots`, `support_frame_ids`, `proposal_digest` (`reconstruction/appearance_state_proposals.py:129-243`) | all frozen visible `V`, with `V ∩ palette` as the conditional diagnostic | truth state ∈ `candidate_costs[:K]`, `K ∈ {1,2,4,8}`; palette-beyond states unreachable at every `K` | `proposals_unavailable` / `budget_exhausted` (277-282) | digest only |
| B12 | `raw_winner` | E (label) | `FaceStateProposal.state` == truth | all frozen visible `V` (primary); `V ∩ palette` reported separately | aggregate-cost argmin | `proposals_review` (`proposals_<reason>`) | digest only |
| B13 | `promoted` | E (label) | `PromotedBlock.state_name/properties` (`reconstruction/state_promotion.py:393-474`); eligibility `export_ready and omitted_unknown == 0` (609) | all frozen visible `V` (primary); `V ∩ palette` reported separately | promotion eligibility under `promotion_budget` and `allow_omitted_unknown=False` (`image_derived_output_policy.py:185`) | `promotion_<reason>` (286-289) | `PromotionResult` on `promoted` |
| B14 | `terminal` | E (label) | **decoded terminal export input:** `export.compare.load_litematic` (103) decodes the exported `.litematic` into a decoded `BlockVolume`, validated by `validate_litematic` (`export/validation.py:10`); those decoded cells are the input scored here | all frozen visible `V` (primary); `V ∩ palette` reported separately | exporter round trip | `export_unavailable` / `empty_volume` | file + digests |

**[R] Non-monotonicity is mandatory, with an exact reconciliation identity.** Every transition
between comparable success sets in one universe reports `gained` and `lost` and satisfies
`next = previous + gained − lost`; a later checkpoint is not bounded by an earlier one. For
cell-level exact-state transitions only, partition `lost` into disjoint `geometry_lost` and
`state_lost` (no cell is counted in both), so
`next = previous + gained − geometry_lost − state_lost`. Never apply that cell-specific equation to
O or P, or subtract across universes. Never divide a G/C/E count by a frame-pixel or pixel-pair
count, and never give a camera diagnostic (B2) a cell-loss equation. The raycast implementation
already does exactly this for its cell-level state transitions
(`scripts/benchmark_templates.py:1358-1382`) and the visible report records T2 smoothing gaining 66
against 45 losses (AUDIT §4, 207-211). Route B has
**no** adjacency smoothing; its honest "gain/loss" epoch follows the pipeline stage order:
placement (B7) → volume/conflicts/omissions (B8) → face ownership/patches (B9) →
palette/candidates/ranking (B10-B12) → promotion/export (B13-B14) (AUDIT §4, 213-216). The first
checkpoint that is not reached is reported as a **frontier stop** (what happened first), never as a
causal attribution of the loss.

**[R] One deliberate simplification.** Route B has no `signature`-visibility pruning stage in the
output path; the analogous losses are B9 reviews and B8 omissions. Do not import raycast stage
names (`available_palette`, `support`, `candidate`, `post_smoothing`, `placed` —
`scripts/benchmark_templates.py:103-114`) into a route-B ledger.

### 2.3 Route A checkpoint table (integer-block geometry + label ledger)

**[F]** Route A's stages are admission → GUI mask → image-only camera → facade grid → palette
label → local-frame export (`cli.py:517-588`; `reconstruction/facade.py:48-130`). The facade emits
integer block-lattice cells (`facade.py:74-118`), so its geometry **is** scoreable under the
permitted alignment.

| # | Checkpoint | Quantity | Record / source | Denominator | Ceiling | Typed unavailable case |
|---|---|---|---|---|---|---|
| A0 | `admission` | — | `load_image` output; `create_gui_mask(h,w)` | **frame universe**: frozen admitted frames (1) | 1 frame | `image_load_failed` |
| A1 | `camera` | camera diagnostic | `CameraEstimate`; yaw/pitch/roll + focal only, position `None`, `scale_known=False` | none — diagnostic; **no cell denominator, no cell-loss equation** | **None.** Residual/projectability are diagnostics, **not** visible-cell recovery and **not** an oracle ceiling | `ImageOnlyCameraError` → CLI `SystemExit` (**untyped**: `cli.py:548-551`) |
| A2 | `facade_grid` | G (geometry) | `BlockVolume` of ≤ `columns×rows` integer cells with the fixed depth formula | **cell universe**: all frozen visible `V` (primary, ≥1 usable pixel) | cells on the facade's own integer grid under the permitted alignment; unknown scale reduces the attainable score | `no_visible_blocks` (573) |
| A3 | `palette_label` | C (palette) → E (label) | nearest-RGB in the run palette (`facade.py:71`) | **same cell universe as A2** — **primary E denominator is all frozen visible `V`**; `V ∩ palette` is the conditional vocabulary diagnostic, reported separately | run palette (8 default or normal-CLI JAR set, not developer-gated) | `palette_unavailable` |
| A4 | `terminal` | E (label) | decoded `.litematic` round trip (`cli.py:583-588`) | **cell universe**: all frozen visible `V` (primary); `V ∩ palette` reported separately | exporter round trip | `export_unavailable` |

**[R] Route A geometry is scored, with the unknown scale reported as a score-reducing caveat, not
as `unavailable`.** The facade's coordinates are integer block-lattice cells, so one permitted
global **integer translation + lattice Y-quarter-turn rotation** aligns them to the reference
(`AGENT_HANDOFF.md:277-278`; `evaluation.py:113-123` "deliberately excludes reflections and scale
fitting"). The camera's `scale_known=False` and the fixed depth formula (`facade.py:95-107`) mean the
run's metric scale is unresolved: that **reduces** the geometry score and is reported verbatim as
the unknown-scale caveat on every G row, but it does **not** make G unscorable and must not be
recorded as `geometry: {available: false}` or `"local_frame_scale_unresolved"`. The alignment
search must never fit scale (`|C|/|V|`, bounding-box ratio or per-axis stretch). Route A also
selects and exports states (its palette labels, A3/A4), so its E rows are real, and the run's
palette is scored per-run.

### 2.4 Does the pipeline ever have to participate? No.

**[R]** The ledger is a pure **post-hoc evaluator over records the pipeline already produced**. It
must not change a gate, threshold, matcher, solver, policy, palette, budget or production
signature, and production must not be able to import it. The existing precedent for a
reference-carrying diagnostic module is `evaluation.py:1-6` ("does not load fixture files, so
production reconstruction code cannot acquire a reference simply by importing an evaluator") and
`evaluation_pair_correspondence.py` (function-pure scorers over already-loaded records).

---

## 3. Evaluator-only discipline and the unknown-scale / gauge rule

**[F] Where the truth side lives today.** Evaluator assets already exist and are sufficient:
`evaluation.visible_metrics` / `best_visible_alignment` / `visible_breakdowns`
(77-183; alignment is one global integer translation + lattice-axis rotation, no reflection, no
scale); `export/visibility.py` `visible_cells` / `theoretical_maximum` with
`export.visibility.ray_directions`; `evaluation_pair_correspondence.py` for strict same-surface
pairs with typed `undefined_reason`; and `scripts/benchmark_templates.py`
`_stagewise_visible_loss` / `_candidate_availability` (1124-1457) as the already-reviewed
attribution *shape*. The prior audit lists these at AUDIT §3, 164-173.

**[R] Boundary contract for the new ledger (must be enforced by test):**

1. The ledger type lives in the evaluation layer (`src/mc_photogrammetry/evaluation_*.py`) or an
   evaluation-only runner under `scripts/evaluation/`. It is never imported by `pipeline/`,
   `reconstruction/`, `cli.py`, `project/` or `export/`.
2. The evaluator layer is **permitted** to accept evaluator-only truth/reference/alignment inputs —
   the truth reference, `visible_reference`/mask, evaluator cameras and the alignment — either as
   already-loaded objects or as evaluator-only paths resolved by the evaluation runner, but only
   after the reconstruction has finished. **No production access is permitted:** no production
   module (`pipeline/`, `reconstruction/`, `cli.py`, `project/`, `export/`) may receive, resolve,
   import or construct any of them.
3. Production receives none of this input and no new parameter. The route-B stage functions keep
   their current signatures (`derive_image_output_from_paths` 215-226;
   `derive_image_derived_output` 296-298; `measure_image_derived_lattice` / `derive_image_derived_geometry`
   as called at 258-273).
4. `reference`, `visible_reference`, `alignment`, `evaluator cameras` and `masks` are
   **evaluator-only**. This matches the standing rule ("fixture cameras/references/masks remain
   evaluator-only and cannot enter normal reconstruction", `AGENT_HANDOFF.md:311`) and the prior
   audit's rule (AUDIT §5, 267-273).
5. **No scale fit, reflection, per-cell alignment or hidden manifest input to production.** Only
   one global integer translation plus lattice-axis rotation per run (`AGENT_HANDOFF.md:277-278`).

**[F] Unknown scale reduces a block-lattice geometry score; it does not make it unscorable.**
Route A: coordinates are integers, so one permitted global **integer translation + lattice
Y-quarter-turn** aligns them to the reference (`pipeline/image_only.py:17-25`, `cli.py:592-595`);
the unresolved metric scale is reported as a score-reducing caveat on the G rows and **never** as
`geometry: {available: false}` or `"geometry_unscorable"`. Route B: the pose graph is a **unit-edge
gauge embedding** with `metric_scale_known=False` (`pipeline/image_derived_pose_graph.py:295,308`),
the session's `scale_known` is hard-`False` (`pipeline/image_derived_session.py:222`), and promotion
keeps `scale_state`/`normalization_state` "unresolved"
(`reconstruction/state_promotion.py:585-587`, constants at `PROMOTION_SCALE_STATE` /
`PROMOTION_NORMALIZATION_STATE`). So even an assembled route-B result has coordinates only up to a
lattice-unit gauge: the ledger may search integer translation and rotation, but it must never
derive a scale from the candidate/reference extent ratio (`|C|/|V|`, bounding-box ratio, or any
per-axis stretch) — that is a scale fit in disguise and is forbidden. If a run's own lattice does
not deliver block-unit coordinates at all, the affected checkpoints are
`unavailable: "gauge_unresolved"`.

**[F] Do not reuse `evaluation.py`'s ratio helper for the ledger's own rates.**
`evaluation._ratio` returns `0.0` on a zero denominator (`evaluation.py:262-263`) and
`best_visible_alignment` inserts `-1` sentinels into its rank tuple (129). The ledger must use
`null`-typed metrics (`metric_value` in `evaluation_pair_correspondence.py:678-684`, already tested
at `tests/test_evaluation_pair_correspondence.py:358-371`) so an undefined denominator is `null`,
never `0`. Reusing `visible_metrics` itself for the G/E numbers is correct and required; it is
only the *rate* wrapper that must stay typed.

---

## 4. Is the proposed E1 tail-only ledger worth doing first?

**[F] What E1 would measure today.** The prior audit's E1 tranche (AUDIT §7, 331-437) builds a
ledger over an accepted `ImageDerivedOutputResult` and reports every upstream checkpoint
`unavailable`; it can only score the tail (`volume_cell` → `terminal`) and only over the accepted
**synthetic** recipe driven through the real `propose_face_states` / `resolve_face_placement` /
`build_plane_volume` / `build_plane_face_observations` / `promote_plane_states` bodies. It
explicitly does not produce an image-only end-to-end number (AUDIT §6, 323-327).

**[R] Assessment: E1 as specified is a contract rehearsal, not the user's answer — do not fund it
as the first tranche on its own. Re-scope it into one measurement-first tranche (§5).**

* The user's question is *where visible geometry and exact-state recovery are lost in the actual
  image-only route*. E1 answers that only for `placed` → `terminal` on an accepted synthetic
  fixture whose promotion is structurally ineligible (15 retained unknown omissions; the policy
  requires `omitted_unknown == 0`, `reconstruction/state_promotion.py:609` and
  `image_derived_output_policy.py:185`; DS-R4 erratum 281-285). At best it reports
  `promoted: unavailable`, `terminal: unavailable`.
* The same module contract, fail-closed rules, five-space separation and transition
  arithmetic can be built **inside** the tranche below at no extra cost, where they immediately
  score at least one real checkpoint on a real input.
* What E1 genuinely buys — an independently testable, non-monotone, fail-closed accounting
  contract that the later upstream export can target — is preserved by folding it in as the tail
  section rather than as a prerequisite.

**[R] Therefore the first tranche is the smallest end-to-end measurement tranche: a frontier-first
ledger.** It answers "what is the first observable failure per route" *today*, with no production
instrumentation and no reference, and then extends to visible-denominator scoring of every
checkpoint whose record is reachable. **The first stop it reports is a frontier indicator of where
the run stopped, not a causal attribution of the loss.** Sequence:

* **A1 (first, smallest, answers part of the question immediately).** A read-only, evaluation-only
  **frontier funnel runner** with **route-specific adapters**, Route B first. It calls the existing
  production functions over frozen inputs in process and normalizes returned records into one
  canonical per-run funnel record giving, per
  checkpoint, `{route, reached: bool, kind, status, typed_reason, universe,
  denominator_available, metric: null}`.
  * **Route B adapters (primary).** `route_b_entry_adapter` / `route_b_lattice_adapter` /
    `route_b_geometry_adapter` / `route_b_output_adapter` consume the in-memory
    `ImageDerivedEntryResult`, the lattice-measure records, the geometry records and the
    `ImageDerivedOutputResult` from the returned `ImageDerivedProductionResult`
    (`image_derived_production.py:68-89,226-300`); no new persistence or schema is needed.
    The named Route-B runner `scripts/evaluation/run_image_derived_frontier.py` calls
    `derive_image_output_from_paths` with a temporary project target and captures the returned
    full result; it compares terminal stage/status/reason with the schema-3 envelope from that
    same invocation. No project is written outside its temporary evaluation directory.
    `scripts/evaluation/run_image_derived_frontier.py` calls the existing public
    `derive_image_output_from_paths` function using a temporary project target, then passes the
    returned full record through these adapters; it compares its terminal stage/status/reason with
    the persisted schema-3 envelope from that same invocation. No project is written outside its
    temporary evaluation directory.
  * **Route A adapters.** `route_a_camera_adapter` (the solved `CameraEstimate`/`CameraModel` from
    `solve_image_cameras` + `camera_model_from_estimate`, `pipeline/image_only.py:52,73`;
    `cli.py:536-551`), `route_a_facade_adapter` (the `BlockVolume` from `reconstruct_facade`,
    `reconstruction/facade.py:18-130`; `cli.py:561-571`), `route_a_palette_adapter` (the palette
    the run actually used — `DEFAULT_PALETTE` or the normal-CLI JAR palette, `cli.py:552-560`),
    and `route_a_terminal_adapter` (the decoded `.litematic` round trip, `cli.py:583-588`). A
    dedicated evaluation-only runner `scripts/evaluation/run_image_only_route_a.py` replays those
    same production functions on frozen inputs so the camera/facade/palette/decoded-export records
    exist without changing `cli.py`.
  * **No per-route coverage claim until both adapter families exist.** A1 is not route-B-only: if it
    ever ships accepting only Route-B records it must state that Route A coverage is absent, not
    claim per-route implementation coverage.
  * Must report the first unreached checkpoint as the **frontier stop** — where the run stopped, not
    a causal attribution of the loss — and the typed stop reason, plus per-run and per-cohort
    **completion and refusal rates** (labelled policy rates, never accuracy). **No reference input
    is used and no visible-truth number is produced in A1.** Acceptance for A1 is exact: on any
    frozen input each route's runner reproduces the same terminal stage/status/reason as the CLI
    would, proves that no checkpoint after the stop is scored as zero, and proves that gate-policy
    counts are never labelled accuracy. This is where E1's fail-closed/typed-`unavailable` record
    types belong.
* **A2 (the measurable attribution, extends A1).** Add the evaluator-only visible-denominator
  section: the evaluator-only builder is **permitted** to accept evaluator-only
  truth/reference/alignment inputs (the frozen truth/visible reference, masks, evaluator cameras),
  computes the permitted alignment with `evaluation.best_visible_alignment`, and scores the
  checkpoints A1 proved reachable **for both routes** — Route A's G row (`facade_grid`) and E rows
  (`palette_label`, decoded `terminal`) under the same permitted alignment, and every reachable
  Route B checkpoint — keyed by the run's own lattice coordinates. Every metric is reported in its
  declared universe (§2.1): cell metrics (G, C, E) over `|V|`, O over frame-pixels / face
  observations, P over pixel-pair identities. Report the labelled
  **G, O, P, C and E** rows separately — geometry, face/pixel ownership, pair correspondence,
  candidate/palette, exact-state — with per-checkpoint conditional denominators, oracle ceilings,
   non-monotone gains/losses reconciled **in-universe** as `next = previous + gained − lost`; for
   cell-level exact-state transitions only, partition the loss into disjoint `geometry_lost` and
   `state_lost`. No arithmetic crosses metric universes. Report `null` for every checkpoint whose
   record was not carried. **Do not wait for A2 to report A1's
  frontier** (the first stop is a frontier indicator, not a causal attribution).
* **A3 (only after A2 + lead approval).** Only if A2 shows the first stop is appearance/ranking
  rather than geometry, consider the local appearance model of §7 — subject to its data gates.

**[R] The one real cost of A2, stated plainly.** Scoring a geometry checkpoint against `V`
requires the run's coordinates in the evaluator's frame. `best_visible_alignment` is exact and
integer (`evaluation.py:113-153`) but its `votes` loop (`133-139`) is `20_000 × 2_000` per rotation
and is calibrated for the raycast tail. For the much larger route-B coordinate set the alignment
must be given a bounded, citable budget (the ledger reports the alignment it used, including
whether it was solved or taken from the run's own gauge), or A2 must first confirm the run's
coordinates are block-lattice-aligned so the search window is small. This is an implementation
bound, not a correctness relaxation, and must be reported, not hidden.

**[R] What A2 must not do:** it must not add persistence to production, must not add a parameter to
a production signature, must not fabricate coordinates for tail checkpoints that never existed,
and must not report a `0` where a record is missing. If A2 cannot reach an event without a
production signature/contract change, it stops and returns the exact missing seam to the lead
(AUDIT §7 stop conditions, 429-437, remain in force).

---

## 5. Implementation sequence with owned paths and acceptance checks

**[R]** All three items are evaluation-only. Nothing below changes `pipeline/`, `reconstruction/`,
`cli.py`, `domain/models.py`, any schema, or any gate.

### Tranche A1 — frontier funnel (smallest end-to-end attribution)

**Owned paths (write scope):**
* `src/mc_photogrammetry/evaluation_image_only_frontier.py` (new evaluation module)
* `scripts/evaluation/run_image_derived_frontier.py` (new, evaluation-only Route B replay runner)
* `scripts/evaluation/run_image_only_route_a.py` (new, evaluation-only Route A replay runner)
* `tests/test_evaluation_image_only_frontier.py` (new tests, both route adapter families)
* `docs/reports/image-only-frontier-a1-2026-10-03.md` (new report)

**Content:**
1. Frozen record types for the funnel: `FrontierCheckpoint`, `FrontierTransition`,
   `ImageOnlyFrontier`, each with `schema`/`producer`, canonical JSON `to_record()`, and constructor
   validation; malformed input is a typed error, never a partially built funnel.
2. Route-specific pure adapters, each over already-returned production records (or absent/`None`
   when a stage was never reached), plus the canonical `ImageOnlyFrontier` record they build:
   * `route_b_frontier(*, entry, lattice=None, geometry=None, output=None)`;
   * `route_a_frontier(*, camera=None, facade=None, palette=None, decoded_export=None)` — the
     inputs are the solved camera, the facade `BlockVolume`, the palette tuple actually used, and
     the decoded export `BlockVolume` (§4 A1), in that production order.
   Both read only fields the accepted records already expose and must not import `cli`,
   `project.store`, `pipeline.image_derived_production`, or any IO module. These A1 adapters take
  **no** truth/reference/alignment/mask input; the A2 evaluator-only builders and the two named
  replay runners are the only places where frozen input paths may appear, and no
   production module may import either module or receive any such input.
   `run_image_only_route_a.py` is an evaluation-only script that calls `load_image`,
   `create_gui_mask`, `solve_image_cameras`, `camera_model_from_estimate`, `reconstruct_facade`,
   `LitematicExporter` and `load_litematic`/`validate_litematic` on frozen inputs in the normal
   CLI order; it never changes `cli.py`.
3. `frontier_unavailable(reason)` typed fail-closed section, mirroring
   `scripts/benchmark_templates.py:_stagewise_unavailable` (1460-1509).
4. `first_observable_failure(frontier)` — returns the first checkpoint that is not reachable, its
   kind, its typed reason, and the evidence digest of the last reachable record.

**Acceptance checks (exact):**
1. `.\.venv\Scripts\python.exe -m pytest -o addopts='' -q tests\test_evaluation_image_only_frontier.py`
   exits 0; `py_compile` passes on the module and the runner. (Command form copied from the E1 spec,
   AUDIT 389; it was **not** run here.)
2. **Contract:** every checkpoint after the first unreachable one is `reached: false` with a typed
   reason and `metric: null`; no zero is ever emitted for a missing record; each row names its
   **fixed universe**; the five labelled spaces (G, O, P, C, E) appear as distinct labelled fields
   and are never summed or interchanged anywhere in the module or its report.
3. **Route B synthetic replay (no reference, no JAR, no screenshot):** drive the accepted synthetic
   recipe from `tests/test_image_derived_orchestration.py::recipe` / `m5_result` / `pipeline`
   (227-268, sourced from `tests/test_plane_placement.py`) through the real public constructors so
   the tail records are real and digest-bearing; assert `route_b_frontier` reproduces the expected
   reachable set.
4. **Route B frozen-input parity:** run `run_image_derived_frontier.py` on a frozen multi-view
   input set and compare its terminal stage/status/reason with the normal CLI command's schema-3
   envelope; any canonical-prefix records must bind the returned `ImageDerivedProductionResult`.
   This check establishes route parity only and produces no visible-truth number.
5. **Route A frozen-input parity (no reference):** with `run_image_only_route_a.py` on a frozen
   input image, the runner's solved camera, facade `BlockVolume` (cells and block states) and
   decoded export equal the normal CLI `reconstruct` route's outputs on the same frozen input — the
   runner's facade is exactly `reconstruct_facade(load_image(...), create_gui_mask(...), ...)` with
   the run palette, and its decoded `.litematic` equals the CLI-exported decoded `BlockVolume`.
   Report the compared digests. This check produces no visible-truth number.
6. **Negative controls:** a malformed/tampered record raises a typed refusal and produces no
   funnel; a Route B frontier built with every optional stage absent reports `entry` only; a Route A
   frontier with every optional record absent reports `admission` only.
7. **Isolation:** `inspect.signature` of the A1 adapters and builders contains no `path`,
   `manifest`, `reference`, `visible_reference`, `truth`, `alignment`, `coordinate` or `bounds`
   parameter. The Route-A adapter is exempt only for its own production records `camera`, `facade`,
   `palette` and `decoded_export` — solved/derived production outputs, not evaluator truth. A
   forbidden-token scan over the A1 module source shows no `open(` / `Path` IO, no
   `LitematicExporter`, no `image_derived_production` / `cli` / `project.store` import, and no call
   into `resolve_face_placement`, `build_plane_volume`, `propose_face_states` or
   `promote_plane_states`; `run_image_only_route_a.py` is the only A1 artifact permitted to read a
   frozen input path or call `LitematicExporter` / `load_litematic`; only the two named replay
   runners may read frozen image paths. This constrains A1 only: A2's
   evaluator-only builders **may** accept evaluator-only truth/reference/alignment inputs, and no
   production module may import or call either module.
8. **Determinism:** two builds over identical records give byte-identical canonical JSON and digest.
9. **Regression:** `tests\test_image_derived_output.py`, `tests\test_image_derived_orchestration.py`,
   `tests\test_plane_volume.py`, `tests\test_state_promotion.py`,
   `tests\test_evaluation_pair_correspondence.py` still pass unchanged.
10. **Report:** exact commands, pass counts, module/test SHA-256, the *textual* frontier expected from
   source for each route (the source-derived expectations are in §6), and the explicit statement
   that **no image-only accuracy is claimed**.

### Tranche A2 — visible-denominator attribution over reachable checkpoints

**Owned paths:** `src/mc_photogrammetry/evaluation_image_only_attribution.py` (new, or extend the A1
module), `tests/test_evaluation_image_only_attribution.py` (new),
`scripts/evaluation/report_image_only_attribution.py` (new, evaluation-only runner),
`docs/reports/image-only-attribution-a2-2026-10-03.md` (new report).

**Content:** the §2 checkpoint table, scored, **for both routes**. Reuse `evaluation.visible_metrics`
for the G and E numbers, `evaluation.best_visible_alignment` for alignment, and
`evaluation_pair_correspondence.metric_value` for every rate. Reuse the
`_stagewise_visible_loss`/`_candidate_availability` semantics (per-stage kinds, per-cell truth
lookup, gains/losses, `captured_limit` honesty) rather than inventing new ones. Route A is scored
through the same A1 `route_a_*` adapters: G from the facade `BlockVolume` and E from the palette
labels / decoded export, under the permitted global integer translation + lattice Y-quarter-turn,
with the unresolved metric scale reported as a score-reducing caveat, **not** `unavailable`. Every
reachable Route B checkpoint is scored as well. The evaluator-only builder may accept evaluator-only
truth/reference/alignment inputs (already-loaded objects or evaluator-only paths); production
receives none.

**Acceptance checks:** A1's checks 1-9 plus: Route A's G row (`facade_grid`) and E rows
(`palette_label`, decoded `terminal`) are scored and distinct from every Route B row; the Route A
facade/export parity check (A1 check 4) is repeated inside A2's frozen recipe; per-checkpoint
incidence denominator and common-denominator coverage are both reported and distinct; each metric is
reported in its declared universe with no cross-universe ratio; a truth state outside the run
palette cannot be counted reachable at any `K`; the primary E denominator is all frozen visible `V`,
with `V ∩ palette` reported only as a separate conditional vocabulary diagnostic; the primary
visibility threshold is ≥1 usable pixel, with stricter thresholds in separate rows; a truth state
present in `candidate_costs` but losing the argmin is exercised as a known non-monotone step;
comparable transitions reconcile in-universe as `next = previous + gained − lost`, and cell-level
exact-state transitions partition loss into disjoint `geometry_lost`/`state_lost` with no O/P term;
the alignment used is reported verbatim and is
integer-translation + rotation only; G, O, P, C and E rows are printed in separate labelled sections
and never combined; camera diagnostics have no cell-loss equation; every checkpoint whose record was
not carried is `null` with a typed reason; completion and refusal rates are reported as labelled
policy rates, never accuracy.

### Deferred (do not start)

* Any production instrumentation, per-stage persistence, schema change, or production signature
  change.
* Any local model training or fitting (see §7).
* Any change to gates, thresholds, budgets, the palette, `allow_omitted_unknown`, or promotion
  policy.

---

## 6. Synthetic contract tests and the frozen evaluation recipe

**[R] Test substrate (no screenshot/reference/JAR/manifest is opened).**
* Primary: the accepted synthetic recipe already built by
  `tests/test_image_derived_orchestration.py` (`recipe` 227-258, `m5_result` 260-268, `pipeline`),
  sourced from `tests/test_plane_placement.py`. It is synthetic and reads no reference or JAR
  (AUDIT §6, 282-288).
* Synthetic truth: a hand-built `{coordinate: BlockState}` reference plus a `visible_reference`
  subset and a known integer alignment, so expected gains/losses/ceilings are computable by hand.
  At least one cell must be outside the palette and at least one truth state must be present in
  `candidate_costs` but lose the argmin.
* Negative controls: an all-`None` frontier; a tampered record (wrong digest/binding) that must
  raise a typed refusal and build nothing.

**[R] Frozen evaluation recipe (record it before any run; do not tune after seeing a number).**
1. **Inputs.** The already-frozen capture inputs and their recorded digests from the sealed reports;
   route B needs ≥2 frames. No new capture, no fixture edit.
2. **Invocation.** Route A: `reconstruct --image <frozen>` (normal mode, default columns/rows/
   layers). Route B: `image-derived-output --sources <frozen...>` / `image-derived-deliver`
   (`cli.py:435-496`) with the default budget flags listed at 441-447.
3. **Policy pin.** Record `IMAGE_DERIVED_OUTPUT_POLICY.policy_sha256` (built at
   `image_derived_output_policy.py:161-191`) and the palette digest; every route-B run must show the
   same `policy_sha256` or the comparison is invalid.
4. **Evaluator inputs (role-separated, evaluator-only).** Frozen evaluator mask spec, evaluator
   cameras, truth reference and the pre-computed `visible_reference` may be accepted by the
   evaluation runner (as already-loaded objects or evaluator-only paths); **no production code path
   may read, resolve or receive any of them**.
5. **Alignment.** One call to `evaluation.best_visible_alignment` over the permitted rotation/
   translation only (global integer translation + lattice Y-quarter-turn); report the returned
   rotation, translation and the coordinates transform string. Never fit scale.
6. **Reported numbers.** Per checkpoint, in its labelled space (G, O, P, C or E) and its declared
   **fixed universe** (§2.1): `numerator`, `denominator`, `value` or `null` with
   `undefined_reason`, the ceiling it is measured against, and `gained/lost/net` per transition,
   reconciled inside one universe as `next = previous + gained − lost`. For cell-level exact-state
   transitions only, partition the loss into disjoint `geometry_lost`/`state_lost` and use
   `next = previous + gained − geometry_lost − state_lost`. No ratio or subtraction crosses G/O/P/C/E,
   and the camera rows carry no transition term. The primary E
   denominator is all frozen visible `V`, and the primary visibility threshold is ≥1 usable pixel;
   `V ∩ palette` and stricter pixel thresholds are separate labelled rows. For the terminal
   checkpoint the scored input is the **decoded terminal export** (the `load_litematic`-decoded
   `BlockVolume`), not the raw file. Both routes are reported: Route A's G/E rows come from the
   runner's facade/palette/decoded-export records, Route B's from its stage records. Per run:
   terminal stage/status/reason and the accepted digests. Per cohort: **counts and completion/
   refusal rates by terminal stage/status only** — labelled policy rates, never a pooled percentage
   across fixtures, and never a single "image-only accuracy".
7. **What is not run.** No benchmark, no raycast stage trace, no supplied-camera diagnostic, no
   mask/reference/JAR read by any production code path (the evaluation runner only), no save scan,
   no training.

**[F] The source-derived expected frontier for a genuine route-B run today** (to be confirmed by
A1's run, not asserted here as measured): `entry` reachable; if `graph.status != "ok"` or
`pose_graph.status != "ok"` the stop is `entry` / `geometry_unavailable`
(`image_derived_production.py:253-257`); else `lattice` reachable and, if not `assembled`, stop at
`lattice` / `lattice_<reason>` (266-270); else `geometry` reachable and, **on the recorded genuine
runs**, stop at `geometry` / `geometry_unavailable` with `reason = "g_<status>"`, the first
non-ready stage being `g` (`image_derived_geometry_orchestration.py:33-37,95-97`; recorded as
`plane_integer_residual` on the frozen PNGs and `ambiguous_quadrant` on the 8-view Canal Town
freeze, `DS-R4` §2 G3 and lead erratum 285). Only the **recorded** route-B runs stop here and did
not reach promotion; this first stop is a **frontier**, not a causal attribution of the loss.
**None of these is an accuracy number.**

**[F] The source-derived expected frontier for a normal route-A run:** `admission` and `camera`
reachable on a lattice-bearing image, `facade_grid` and `terminal` reachable; G is scored under the
permitted global integer translation + Y-quarter-turn with the unresolved metric scale reported as
a score-reducing caveat, **not** as `unavailable` (§2.3). No published image-only visible-truth
percentage exists for either route (AUDIT §1, 47-54; §3, 155-162).

---

## 7. Separate assessment: the local appearance-model proposal

**[F] This is a separate proposal from the ledger; keep the two decisions independent.**
`AGENT_HANDOFF.md:299,304-305,310-311` proposes a compact local vision model as an auxiliary
pre-prune state proposer/ranker, never as a replacement for geometry or state-property inference.
Nothing in this design authorizes training; E3 artifacts remain `evaluation_only: true`,
`training_permitted: false` (`docs/reports/texture-poc-e3-factory-patch-dataset-export-2026-10-03.md:109,200,246-247`),
and T2/T6 remain evaluation-only.

### 7.1 Suitable compact supervised proposer or ranker

**[R]** If a model is ever fitted, the only defensible shapes are:
* **Pre-prune candidate proposer** (cell-local, top-k over `BlockState`) targeting T6's
  candidate-generation loss; and
* **Ranker over the existing candidate list** (re-order `candidate_costs`) targeting T2's
  ranking loss.
A full classifier replacing `AppearanceScorer` is out of scope and would change geometry-affecting
decisions. Keep geometry-affecting state properties deterministic where possible
(`AGENT_HANDOFF.md:299`).

### 7.2 Exact production patch parity

**[F]** This is currently **not** established. The handoff records ~98.5% mean valid-pixel
agreement after sampler shrink, but exact production patch digests match **0/857 T2 and 2/2,673 T6
joins**, with about one-third of production face keys unmatched (`AGENT_HANDOFF.md:306`;
`docs/reports/texture-poc-e2f-numeric-patch-similarity-2026-10-03.md`). "Identity joins and
near-equal crops are not proof of training/evaluation pixel parity."

**[R] Falsifiable parity gate before any training:** a candidate model must consume **byte-identical**
patches to the production `FaceObservation` patch for the same `(cell, slot, frame)` join — proven
by digest equality on a held-out join set, not by mean-pixel similarity. Until digest parity holds,
any accuracy number is about a different input than production sees. This gate is upstream of, and
independent from, the ledger.

### 7.3 Data, label and split gates

**[F]** From the sealed reports and the E6 design:
* E3 exported **8,516** patch examples over **2 independent source groups/builds**; all index
  splits are topology-keyed and it is explicitly "evaluation-only data and not yet a
  train/validation split" (`texture-poc-e3-factory-patch-dataset-export-2026-10-03.md:184-187,224-225,246-249`).
* E5 found **63 exact states but only 4 shared across the two builds**; leave-one-build-out OOV is
  **55.76% / 78.74%** (`AGENT_HANDOFF.md:307`). E5's own falsification criterion: if a held-out
  build still has >~50% OOV examples, or fewer than ~10 exact states with ≥50 examples in at least
  two builds, a closed-set build-held-out experiment is indefensible
  (`texture-poc-e5-factory-state-support-2026-10-03.md:412-416`). **On the recorded numbers the
  first branch fails today, and E5's two-build corpus is not a substitute for the E6 gates below.**
* The E6 acquisition design fixes the binding support requirements
  (`docs/reports/texture-poc-e6-balanced-build-acquisition-design-2026-10-03.md`): a **minimum of 6
  admitted independent builds** (one structure per world; fewer than 6 is "do not claim", §4
  `:140-144`), and the no-go test that acquisition failed if a held-out build still has **>50% OOV**
  (§13 `:429`) or **fewer than 10 exact states have ≥50 admitted examples in at least three
  independent builds** (§5 `:182`; §13 `:430-431`, which states that two builds is not sufficient).
* Label purity is separate and unmet: E4's consistency proxy does not establish correct labels;
  region-clipping and occluder contamination are unquantified
  (`texture-poc-e5-factory-state-support-2026-10-03.md:417-420`).

**[R] Gates that must all be cleared before any fitting** — the accepted E6 support requirements
are **necessary, not sufficient authorization to train**:

1. **≥6 admitted independent builds overall.** An independent build is a unique
   `(world_identity, scene_group)` with one structure per world; recaptures/variants of one
   structure are not builds (E6 §3 `:120`, §4 `:140-144`). E5's two-build evidence is **not** a
   substitute (E6 §13 `:430-431`).
2. **No evaluated held-out build exceeds 50% OOV** examples under the frozen core (E6 §13 `:429`);
   held-out OOV is abstained on, never scored as a core-class error, and reported per build
   (E6 §10 `:348-350`).
3. **≥10 exact core states each have ≥50 admitted examples in ≥3 independent builds**, drawn from
   the Tier A ∪ Tier B core (E6 §5 `:182`, §6 `:210-211`, §13 `:430-431`; two builds is explicitly
   insufficient).
4. **E6 per-build and per-capture requirements preserved:** ≥3 distinct captures per state, ≥50
   admitted examples per core state (target ≥100), ≥1,000 admitted examples per build (target
   ≥2,500), ≥3 capture sessions per build, ≥16 finalized HUD-hidden frames per build, ≥3 builds per
   core state, ≥10 core exact states, ≥10 shared material families; results reported per build and
   per capture, never as one folded number (E6 §6 `:203-212`, §10 `:351-352`; admission uses
   `valid_fraction >= 0.75`, E6 §6 `:200-201`).
5. **World-grouped train/dev/calibration folds plus immutable test roles.** Fold unit = independent
   build = world; all captures and variants of a build stay in one fold; primary protocol is
   leave-one-build-out; `dev`/calibration examples live in their own separately sealed artifact; an
   immutable world-held-out **test** set is reserved and excluded from fitting (E6 §10.1-10.2
   `:341-347`, §10.7 `:357-367`). This requires the separate lead-approved training-data
   contract/exporter that **does not exist yet** (E6 §10.7 `:357-367`, §11 `:407-410`).
6. Grouping by **source build**, not by screenshot/view (`AGENT_HANDOFF.md:316`), with no
   index-split leakage (`check_split_leakage` / `build_group_manifest`, E3 report 129-132).
7. Label purity quantified and accepted (`texture-poc-e5-factory-state-support-2026-10-03.md:417-420`).
8. Exact production-patch digest parity (§7.2).
9. No E3/E4/E5/E6 artifact is used as training input; T2/T6 remain evaluation-only
   (`AGENT_HANDOFF.md:304,310`).
10. **Necessary, not sufficient.** Clearing items 1-9 does not authorize training; fitting stays
    blocked until the separate training-data contract, the frozen world-grouped split manifest and
    the label-purity gate are reviewed and approved (E6 §16 `:488-489`; P3 blocked, E6 §12 `:419`).

### 7.4 Baselines, metrics, runtime

**[F] Runtime:** the current Windows environment has NumPy/Pillow but **no PyTorch, torchvision,
scikit-learn or ONNX Runtime** (`AGENT_HANDOFF.md:310`). Any model option therefore carries an
install/runtime decision that is itself a lead approval, not an implementation detail.

**[R]** Compare any candidate against `PaletteIndex` on **identical production-aligned patches**
(`AGENT_HANDOFF.md:299`). Report, in this order and separately:
material top-1/top-k; full-`BlockState` candidate recall (including beyond-palette truth, which is
unreachable at any `K`); calibration/abstention quality (`ABSTENTION_REASONS`,
`appearance_state_proposals.py:104-111`); then **pinned geometry and exact-state end-to-end metrics
on frozen identical inputs** with the unchanged promotion gates. **Classifier gains are not
end-to-end gains**: a ranker that improves top-1 over the existing candidate list cannot recover a
cell whose truth state the pipeline never generated, and neither can it fix the geometry frontier.

### 7.5 Falsifiable promotion criteria, and the RL recommendation

**[R] Promote a model only if, on frozen identical inputs and with every existing gate unchanged:**
(a) held-out-**build** exact-state top-1 and top-k improve over `PaletteIndex` by a pre-registered
margin with confidence intervals that exclude zero; (b) production-patch digest parity holds for
100% of the scored joins; (c) end-to-end route-B checkpoints (candidate recall, raw winner,
promotion eligibility) improve, not just the classifier; (d) abstention/calibration does not
regress; (e) no geometry-orientation property becomes model-dependent; (f) runtime/memory fits the
declared budget. If (a) holds but (c) does not, the model is a diagnostic, not a promotion.

**[R] Recommend against reinforcement learning.** There is no verified online reward and no
user-confirmed correction-label stream in this project; the only "rewards" available are frozen
offline evaluator metrics, which are exactly the kind of proxy RL over-optimizes. Absent a
lead-approved, user-confirmed correction-label pipeline with an auditable reward signal, RL should
not be attempted.

---

## 8. Acceptance trace

This report satisfies the task's acceptance criteria as follows:

* **"Identifies the first currently observable failure per route without treating gate policy counts
  as accuracy."** Route B: the first observable failure is the typed terminal stage/status/reason —
  the **recorded** genuine runs stop at `geometry`/`geometry_unavailable` with the first non-ready
  stage `g` and did not reach promotion (§6). Route A: `admission`, `facade_grid` and `terminal` are
  reachable; G is scored under the permitted global integer translation + Y-quarter-turn with the
  unresolved metric scale reported as a score-reducing caveat, **not** as `unavailable` (§2.3). The
  first stop is reported as a **frontier**, not a causal attribution, with completion/refusal rates.
  §1 and §6 state explicitly that status/reason distributions are policy outcomes, not accuracy, and
  that the supplied-camera T2/T6 numbers are disqualified from both routes.
* **Route A ownership (addendum 1).** A1 names `route_a_camera_adapter`, `route_a_facade_adapter`,
  `route_a_palette_adapter` and `route_a_terminal_adapter` plus the evaluation-only
  `scripts/evaluation/run_image_only_route_a.py` runner, with frozen-input facade/export parity
  against the normal CLI route; A2 scores Route A G/E as well as every reachable Route B checkpoint
  (§4, §5). No per-route implementation coverage is claimed while a route's adapters are absent.
* **Accounting units (addendum 2).** Every metric declares a fixed universe (reference cells for
  G/C/E, frame-pixels / face observations for O, frame-pair / pixel-pair identities for P);
  transitions are allowed only between comparable success sets in one universe and partition
  cell-level exact-state losses into disjoint `geometry_lost` and `state_lost`; nothing is
  subtracted across G/O/P/C/E and camera diagnostics have no cell-loss equation (§2.1-§2.3, §6).
* **"A reviewer can translate its chosen next tranche into an owned file list and exact acceptance
  checks."** §5 gives owned paths, module content, and numbered acceptance checks for A1 (including
  Route A) and A2.
* **"The texture-model recommendation respects all data gates."** §7 keeps E3/E4/E5/E6 artifacts
  `evaluation_only`/`training_permitted: false`, T2/T6 evaluation-only, and states the exact accepted
  E6 support requirements (≥6 admitted independent builds; no held-out build >50% OOV; ≥10 exact
  states with ≥50 admitted examples in ≥3 independent builds; E6 per-build/per-capture quotas;
  world-grouped train/dev/calibration folds with an immutable test role), explicitly as necessary
  but not sufficient authorization to train. The separate world-grouped contract, split, support,
  label-purity and patch-parity gates are prerequisites rather than assumptions.

**[F] No image-only visible-truth percentage is claimed anywhere in this report**, and none exists
in the audited tree (AUDIT §1, 47-54; §3, 155-162). The only numbers quoted (T2/T6, E3/E5/E6) are
labelled as supplied-camera or evaluation-only-corpus/design facts from sealed reports.

**Limits.** Static reading only; no pipeline, benchmark, fixture, test or model job was executed and
no hash was recomputed. Line anchors can drift. The A1/A2 tranches need a lead/Jev routing decision;
A2 and any model route additionally need Astra review of the truth-input boundary (evaluator-only
inputs permitted, production access forbidden) and the five-space/universe rule before
implementation. This revision applied the lead-review correction addendum (route A ownership,
accounting units, exact E6 support gates) while preserving the earlier accepted corrections;
**Astra re-review is requested for the route A adapter/runner ownership, the fixed-universe
transition rule, and the restored E6 gates. Stop for lead review.**
