# IMGLOSS-A2 visible-denominator attribution - implementation report

Date: 2026-10-03
Task: `AGENT_COMMS.md` 2026-10-03 Lead -> DeepSeek - IMGLOSS-A2 visible-denominator
attribution implementation
Design: `docs/reports/image-only-loss-attribution-design-2026-10-03.md` (accepted, §4-§6 Tranche A2)
Status: implementation + tracked-mask/pin correction + final metric-contract
correction + final P digest-binding correction + checkpoint-reach/metric-availability
separation + Astra follow-up reach-gating/E-partition/substage-reach repair + Route B
run-native candidate/alignment adapter repair + run-native palette-alignment repair
+ runner promotion-presence repair + runner real-record/coordinate-only adapter repair
+ production-before-truth ordering repair complete;
focused tests pass; one corrected Route A and one corrected
Route B frozen run performed (no route rerun in this revision).
**Stopped for lead/Astra review. A2 is not marked complete.**

This revision applies the 2026-10-03 HOLD: the scored mask now comes from the
tracked reviewed JSON specifications under `docs/evaluation_masks/templates-2`
through the accepted evaluator rectangle/polygon semantics, and a supplied
reviewed pin is mandatory - a mask-domain mismatch refuses and builds no
attribution. The earlier ignored `.masks/*.png`-derived number is superseded. The
unsupported "bounded translation search proves a global/theoretical ceiling"
claim is removed. No production file changed.

This revision also applies the 2026-10-04 final metric-contract corrections and the
final P digest-binding HOLD to the remaining independent review findings, in
A2-owned paths only:

1. **The Route B `entry`/B0 P row is digest-bound, never an evaluator alias.**
   `evaluate_scene`/`select_mutual_matches` is a different matcher from the
   entry's `joint_matching.match_and_solve_pair`, so its `0/232` recall and `0/33`
   precision are **not** Route B achieved P metrics. The entry row is only
   scoreable when a rederived joint match-array digest equals the accepted
   `RelativeCameraEvidence.matches_sha256`; `EntryMatchBinding`/`EntryPairEvidence`
   enforce that in the pure module and `pair_metric_rows` refuses any unbound
   aggregate. Because the retained evidence carries no pixel/keypoint-pair
   identities and this revision performed no route rerun, the Route B entry/P row
   is reported `available: false` / `metric: null` with the typed reason
   `entry_match_identities_not_retained_or_digest_unbound`, and the generic
   evaluator aggregate is kept only under a separately named `evaluator_pair_control`
   diagnostic explicitly excluded from the Route B attribution and loss funnel.
2. **A Route B `G` transition is unpartitioned same-universe reconciliation.**
   `placed` and `volume_cell` are both geometry, so their step reconciles
   `matched` cells and carries `geometry_lost = state_lost = null`;
   `geometry_lost`/`state_lost` remains exact-state-`E`-only. A regression carries
   both geometry checkpoints and proves the identity holds without a state
   partition.
3. **Checkpoint reach is serialized separately from metric availability.** The
   Route B `entry`/B0 checkpoint is demonstrably reached, but its P metric is
   unavailable because the accepted entry evidence retains only counts/digests.
   `AttributionRow` previously expressed both facts through `reached`, so the
   reached-but-unscorable entry row serialized `reached: false`, which can be
   misread as the production checkpoint never having run. Every row now also
   carries an explicit `checkpoint_reached` (`true`/`false`, or `null` when the row
   names no production checkpoint, e.g. an aggregate/evaluator-only row),
   populated from the accepted A1 frontier's own `reachable` per checkpoint. The
   entry P row is therefore `checkpoint_reached: true` beside `reached: false`,
   `metric_value: null`, `ceiling: null`, `conditional_denominator: null` and the
   unchanged typed reason `entry_match_identities_not_retained_or_digest_unbound`;
   an entry the run never reached carries `checkpoint_reached: false`. A reached
   checkpoint never implies a measured metric and a null metric never implies the
   checkpoint did not run. No route was rerun for this correction.

This 2026-10-04 follow-up revision applies the three defects in the independent
Astra HOLD (`agent swarm instantiation template/tasks/active/image-only-a2-reach-and-transition-repair-2026-10-04.md`),
in A2-owned paths only, with **no Route A or Route B replay**:

4. **A digest-bound entry aggregate is never scored on an unreached entry.**
   `route_b_attribution` previously tested `entry_pair_evidence.bound` *before*
   `entry.reachable`, so a syntactically digest-bound aggregate could emit numeric
   P rows with `checkpoint_reached: false`. Reach is now checked first: a frontier
   that did not reach `entry`/B0 emits one typed-unavailable row with
   `reason: entry_unreached`, `checkpoint_reached: false`, `reached: false` and
   `metric_value: null` - no numeric metric - even when bound evidence is passed.
   `pair_metric_rows` additionally raises `AttributionRefusal` when called with
   `checkpoint_reached=False`, so the contradiction cannot be constructed directly.
   A new route-level regression builds the entry-absent frontier, supplies the
   existing valid digest-bound aggregate, and asserts no numerator/value is emitted.
5. **Every Route B E-to-E transition partitions its loss again.** The shared
   transition loop passed `partition_loss=False` for both G and E, so the E steps
   `raw_winner -> promoted` and `promoted -> terminal` omitted
   `geometry_lost`/`state_lost`. Only the G-to-G step (`placed -> volume_cell`) is
   now unpartitioned; each E-to-E step passes `partition_loss=True`. A new
   route-level regression carries both E endpoints (`raw_winner`/`promoted` and
   `promoted`/`terminal`) with a geometry loss and a state loss that sum exactly to
   `lost`, and asserts the G step still carries no partition.
6. **Promotion/decoded-export reach is not copied from the coarse A1 `output`
   checkpoint.** `promoted` and `terminal` previously inherited
   `checkpoints["output"].reachable`, which could claim a state promotion or a
   decoded export was reached on a review/refusal output. A1 has no dedicated
   promotion/export checkpoint, so those rows now serialize
   `checkpoint_reached: null` unless their **own carried record** proves the
   substage (`checkpoint_reached: true` when the record is carried). Presence is
   explicit record existence (`record is not None`), **never mapping truthiness**:
   a carried empty mapping proves the substage record exists and serializes reach
   `true` with an unavailable metric, while only a missing record is `null`
   (unattempted/unknown), never a fabricated `false`. Rows whose evidence
   *is* the A1 output-stage record (`volume_cell`, `raw_winner`, candidate/palette
   availability) keep the output reach. New regressions cover the absent (`null`),
   carried (`true`) and carried-empty (`true` beside `reached: false`) cases; this
   limit is documented in the module contract, the builder docstring, a serialized
   caveat and this report.

This 2026-10-04 A2 R3 follow-up revision implements the Route B
candidate-space/alignment adapter (`agent swarm instantiation template/tasks/active/image-only-a2-route-b-candidate-alignment-2026-10-04.md`)
on the four owned paths, with **no Route A or Route B replay**:

7. **Route B builder inputs are run-native; the alignment is applied exactly once.**
   `route_b_attribution` already transforms each cell map through
   `score_aligned_cells`/`align_candidate`, but the runner additionally called
   `align_candidate` on placement, volume, promotion and proposal coordinates before
   calling the builder, so both applied the same non-identity transform. The
   runner's pre-alignment block (and its now-unused `align_candidate` and
   `SimpleNamespace` imports) is removed; the builder receives the run's own
   unaligned maps/proposals, and the measured `alignment` record is retained only
   for scoring/reporting. The builder docstring and the module contract now state
   this run-native/one-transform input rule explicitly.
8. **Proposal coordinates *and* every ranked candidate state align once.** The
   candidate sweep previously met `candidate_metric_rows` through a
   `reachable_by_coordinate` map whose keys were still run-frame proposal cells
   (and whose ranked states were unrotated). A new pure helper,
   `align_candidate_rankings`, passes each proposal cell and **every** ranked
   `BlockState` through the same `align_candidate` semantics (one integer
   translation + lattice Y-quarter-turn, directional properties rotated), preserving
   ranking order and any cost association, so the ranked lists join the C score's
   reference-frame `coordinate_truth` correctly. The old runner's `{cell: None}`
   proposal coordinate mapping would additionally have raised on a non-identity
   quarter-turn; that path is gone.
9. **The C rows get a genuine C-space context.** `palette_metric_rows` and
   `candidate_metric_rows` were fed the G-space `placed`/`volume_cell` `CellScore`,
   which the strict `candidate_metric_rows` C-space guard correctly refused (so a
   Route B result carrying geometry + proposals + a non-null palette could not
   report C at all). The builder now builds a real `MetricSpace.CANDIDATE`
   `CellScore` with `score_aligned_cells` over the same frozen visible-cell universe
   and palette, and never hands the G-space base score to a C helper; the strict
   C-space guard is unchanged. Missing base/proposal records still serialize the
   typed-unavailable rows (`output_record_absent`/`palette_unknown`/
   `proposals_unavailable`), never a measured zero. New synthetic route-level
   regressions drive a non-identity translation + Y-quarter-turn with a directional
   `oak_stairs` state, assert the `candidate_available_k1/k2/k4` counts (2/3/3) and
   the full `placed` geometry recall, and a runner-level regression asserts the
   builder receives the run-native maps and that one transform reproduces the
   reference (the pre-fix double transform scores an overlap of one).

The 2026-10-04 A2 R4 palette-alignment repair (task
`agent swarm instantiation template/tasks/active/image-only-a2-palette-alignment-2026-10-04.md`,
routed by official Jev `fix_a2` at confidence 1.0 after Astra's read-only review)
fixes the last alignment gap, on the palette itself:

10. **The run-native palette is rotated with the candidate.** `score_aligned_cells`
    rotated candidate coordinates and directional candidate states but tested the
    reference truth against the **unrotated** run-native `palette_states`. For a
    directional palette this made the C palette row disagree with the aligned E and
    candidate rows (native `oak_stairs[facing=north]`, quarter-turn 1, truth
    `facing=east`: candidate/E 1/1 while palette coverage was 0/1). A new pure
    helper, `align_palette_states`, maps every palette `BlockState` through the
    **same** Y-quarter-turn state mapping `align_candidate` uses (cardinal property
    keys, `facing`, `axis`, `rotation`, compound `shape`); the alignment
    **translation cannot affect a coordinate-free palette**, a zero quarter-turn is
    the identity, and the result is a deterministic order-independent `frozenset`.
    Both `score_aligned_cells` and the standalone `palette_coverage` helper now use
    it, so palette coverage, the in-palette conditional denominator and the
    candidate/E rows agree, while a truth state deliberately outside the transformed
    palette still leaves the primary `|V|` denominator untouched. The prior
    synthetic adapter test had masked the defect by passing an already
    reference-frame palette; it now passes the run-native palette, and a new
    adversarial route-level regression drives a run-native directional palette,
    placement and proposals under quarter-turn 3 plus translation, asserting
    palette coverage 3/4 with an out-of-palette `GOLD` cell,
    `palette_coverage_in_palette` 3/3, `candidate_available_k1` 3/4 and
    `raw_winner` exact-state 3/4 (and its in-palette conditional 3/3). Helper-level
    regressions prove translation independence, order/iteration independence, the
    zero-turn identity and the forbidden-field refusal; both the corrected and the
    new test were verified to fail when the palette rotation is reverted.

The 2026-10-04 A2 R5 runner promotion-presence repair (task
`agent swarm instantiation template/tasks/active/image-only-a2-promotion-presence-2026-10-04.md`,
routed by official Jev `fix_runner` at confidence 1.0 after independent Astra/Luna
reviews) closes the last adapter-boundary gap in the substage-reach contract of
item 6, with **no Route A or Route B replay**:

11. **The runner preserves promotion-record presence.** The pure builder
    distinguishes an absent promotion record (`checkpoint_reached: null`) from a
    carried record (`checkpoint_reached: true` beside an unavailable metric) by
    explicit `is not None`, never mapping truthiness. The runner instead passed
    `promoted_cells=promoted_states or None`, and `_promoted_states` returns `{}`
    both when `promotion is None` and when a carried promotion has zero cells, so
    `or None` erased the carried-empty reach evidence before the builder could see
    it. The runner now retains the promotion object itself and passes the extracted
    mapping whenever `outcome.output.promotion is not None` (including `{}`), and
    passes `None` only for a genuinely absent record. A carried empty promotion
    therefore serializes `promoted` as `checkpoint_reached: true` beside
    `reached: false`, `metric_value: null` and a typed reason, exactly as the pure
    contract specifies, while an absent promotion serializes
    `checkpoint_reached: null`. Two runner-level regressions drive the real
    `_route_b_once` seam with synthetic production records (no fixture or route
    replay): the carried-empty case asserts the builder receives `{}` and the
    serialized `promoted` row keeps reach `true` with no metric, and the absent
    case asserts the builder receives `None` and the row serializes reach `null`
    with no metric; both assert the metric-availability flag (`reached: false`,
    `metric_value: null`) stays separate from reach. The carried-empty regression
    fails on the pre-fix `promoted_states or None` runner. The runner carries no
    decoded terminal record, so `terminal` remains `checkpoint_reached: null`; a
    carried terminal record stays covered by the pure-builder regressions.

The 2026-10-04 A2 R6 palette-ceiling repair (task
`agent swarm instantiation template/tasks/active/image-only-a2-candidate-palette-ceiling-2026-10-04.md`,
routed by official Jev `attach_palette` at confidence 0.96 after Luna review)
enforces the vocabulary ceiling per cell in the candidate and conditional
exact-state rows, with **no Route A or Route B replay**:

12. **`C_K` and the conditional `E` numerator are bounded by the aligned
    palette membership `V_P`.** `score_aligned_cells` now carries the aligned
    producer palette and its exact visible membership `V_P` (the visible cells
    whose reference truth is a member of the aligned palette) as in-memory-only
    evidence on the `CellScore`; the fields are never serialized and the primary
    `|V|` denominator is untouched. `candidate_metric_rows`'s numerator is now
    `|{v in V_P : truth(v) in ranked(v)[:K]}|`, so `candidate_available_kK` can
    never exceed `palette_coverage`, and the conditional exact-state numerator is
    `E inter V_P` rather than the whole `E` set (the prior code divided all exact
    cells by `|V_P|` and could report a `3/2` rate). Every retained ranked candidate
    is validated against the aligned producer palette **before** the top-`K` cut:
    an out-of-palette candidate contradicts the run's own vocabulary and is a typed
    `AttributionRefusal`, never silently filtered or reordered before `K`. A C
    score with no aligned palette membership is typed unknown and refuses candidate
    scoring rather than assuming an empty palette; a deliberately empty palette
    carries `V_P = {}` and refuses any ranked candidate as a contradiction. The
    per-cell `CellScore` invariant now requires `palette_coverage ==
    conditional_denominator == |V_P|` and `V_P` to be visible truth cells of the
    same score.

Adding the serialized `checkpoint_reached` field re-baselines the canonical A2
attribution record shape, so the Route A `attribution_sha256` recorded in §4
(`901408b7...`) and the Route B retained digest are **pre-revision (field-absent)
provenance**: they are not recomputed in this revision because no Route A or Route B
replay is authorized. The measured Route A numerators/denominators, frozen `V`,
alignment, transitions and policy rates are unchanged by that field, because it is
serialization metadata. No new canonical digest
is claimed here; the lead may re-run a route if a re-baselined digest is required.
The A1/design hashes still match.

The A2 R3 adapter repair and the A2 R4 palette-alignment repair also change what the
canonical serializer emits (the runner now hands the builder run-native records, the
builder emits a genuine C-space context, and every palette state is mapped through
the permitted directional rotation), so the Route A `901408b7...` / Route B retained
digests remain **pre-revision provenance** as well. No Route A or Route B replay was
performed, so no replacement `attribution_sha256` is derived or claimed. The primary
G rows, the primary all-`V` E rows, frozen `V`, the measured alignment and the policy
rates are palette-independent and unchanged. The **palette-dependent** rows
(`palette_coverage`, `palette_coverage_in_palette`, `exact_state_recovery_in_palette`)
are retained measured numbers: the frozen Route A run used the default 8-state
palette, whose eight states carry no directional properties, so the Y-quarter-turn
rotation is the identity for it and its measured `V inter palette = 0` of 398 is
unchanged. Any future run whose palette carries directional states under a
non-identity quarter-turn would now be scored with the rotated palette rather than
the pre-fix unrotated one; no such run is re-scored here. The pre-fix runner/builder
adapter had no effect on the frozen Route B run (its Stage-7 tail is absent, so the
C path was never exercised).

The A2 R5 runner promotion-presence repair changes only which promotion record the
runner hands the builder (`{}` for a carried empty promotion, `None` for an absent
one); it does not change the builder, the canonical record shape or any measured
number. The retained frozen Route B run stopped before an output record
(`g_unavailable`), so it never carried a promotion record and its retained
`promoted`/`terminal` reach remains `null` under both the pre-fix and post-fix
runner. No Route A or Route B replay was performed for R5 either, so the Route A
`901408b7...` / Route B retained digests remain **pre-revision provenance** and no
replacement `attribution_sha256` is claimed.

The A2 R6 palette-ceiling repair changes only the pure module's C/E numerators and
the `CellScore` evidence it carries; the runner is untouched, no canonical field was
added, and the frozen Route B run has no Stage-7 tail, so its C path is unexercised
and its retained attribution is unaffected. The primary G rows, the primary all-`V`
E rows, `V`, the measured alignment and the policy rates are unchanged; a
palette-dependent row can change only where an exact cell's truth was outside the
run palette (the conditional numerator is now correctly bounded) or where a ranked
list contained an out-of-palette state (now a typed refusal instead of a count). The
retained Route A palette-dependent rows (`palette_coverage 0/398` with the default
8-state palette) are unchanged because no exact or ranked cell there was outside
that palette. No Route A or Route B replay was performed, so the Route A
`901408b7...` / Route B retained digests remain **pre-revision provenance** and no
replacement `attribution_sha256` is claimed.

The 2026-10-04 A2 R6 validation addendum (same task file, `validate_vp` addendum,
routed by official Jev at confidence 1.0 after Luna found the `CellScore` invariant
accepted a mere subset) closes the record-level hole: `CellScore.__post_init__` now
**requires** the visible truth context whenever palette evidence is carried and
**recomputes** `V_P = {v in V : truth(v) in aligned_palette}` from `coordinate_truth`
and `aligned_palette`, refusing a missing (or non-mapping) context and any
incomplete/superset/foreign membership set with `AttributionRefusal`. The carried set
is therefore never trusted as a claim, and a manually constructed forged `CellScore`
cannot reach the C ceiling or the conditional-E numerator. The Astra follow-up finding
on the same task (the whole-`E` fallback) is closed in the same invariant: a palette
aggregate (`palette_coverage`/`conditional_denominator`) now requires the complete
context (aligned vocabulary + exact `V_P` + `coordinate_truth`) at record construction,
so an aggregate-only record can no longer exist, and `exact_state_over_palette` has no
`len(exact)` fallback - it always uses `E inter V_P`, and an absent palette yields no
conditional row at all. The three accepted formulas
(`C_K = |{v in V_P : truth(v) in ranked(v)[:K]}| / |V|`, conditional
`E = |E inter V_P| / |V_P|`, primary `E = |E| / |V|`) and the
`palette_coverage == conditional_denominator == |V_P|` check are unchanged; the module
still never serializes palette contents and no canonical field changed. No Route A or
Route B replay was performed, so the Route A `901408b7...` / Route B retained digests
remain **pre-revision provenance**.

The 2026-10-04 A2 R6 immutable-snapshot addendum (same task file, `immutable_copy`
addendum, routed by official Jev at confidence 1.0 after Luna found
`CellScore.coordinate_truth` stayed a mutable dict after the exact-`V_P`
validation) closes the last record-level hole in that evidence chain:

13. **The truth context is frozen before `V_P` is validated.** `CellScore.__post_init__`
    now requires the truth context (whenever one is carried) to be a `Mapping`, then
    replaces it with an immutable snapshot - a `MappingProxyType` read-only view over
    a private `dict` copy - before the exact-`V_P` recomputation reads it. The
    exposed `score.coordinate_truth` therefore rejects item assignment and deletion,
    and mutating the constructor's source mapping after construction can no longer
    change the snapshot, the validated `V_P` membership or the `C`/conditional-`E`
    rows. The accepted formulas, the `palette_coverage == conditional_denominator ==
    |V_P|` check, `to_record()` and the runner are all unchanged; no canonical field
    or production record shape changed, palette contents are still never serialized,
    and no Route A or Route B replay was performed (the `901408b7...` / Route B
    digests remain pre-revision provenance).

The following items document the initial R7 implementation snapshot and are
superseded where noted by the follow-up section at the end of this report.

The initial 2026-10-04 A2 R7 runner-record adapter repair (task
`agent swarm instantiation template/tasks/active/image-only-a2-runner-record-adapters-2026-10-04.md`,
routed by official Jev `repair_evaluation_runner` at confidence 1.0 after Astra's
read-only review, then corrected by the lead's coordinate-only clarification) makes
the runner adapter score synthetic Route B records built through the real public
constructors, with **no Route A or Route B replay**:

14. **The runner reads the real typed records, and geometry-only records are never
    faked as states.** The Route B adapter now takes placement from
    `output.placement` (a `ResolvedFacePlacement`, one world coordinate per
    `ResolvedFace`/`PlacedCell` derived exactly as `build_plane_volume` derives it),
    volume from `output.volume.cells` (`SupportedCellObservation.coordinate`) and
    promotion from `output.promotion` through the real `PromotedBlock.state()` API.
    It no longer reads the nonexistent `geometry.placement_records["axis_support"]`
    field and no longer returns `{}` for every volume. Because placement/volume carry
    coordinates but **no** `BlockState`, they are passed to the builder as a new pure
    `GeometryCellEvidence` coordinate record, not as a `{coordinate: None}` map: the
    accepted `align_candidate` consumes and rotates `BlockState` values, so a `None`
    value would raise under a non-identity Y-quarter-turn. A new pure
    `score_aligned_geometry_coordinates` applies the same one permitted integer
    translation + lattice Y-quarter-turn to the coordinates only and computes the
    same G counts `evaluation.visible_metrics` computes for a state-free candidate
    (`matched`, phantoms, hidden inferred), emitting the existing `CellScore` fields
    with an empty `exact` set; a geometry-backed C context is built the same way so
    the palette/availability rows keep their validated `V_P`. No serialized field,
    metric formula, universe or threshold changed, and no label is invented.
15. **Initial R7 behavior, superseded by the follow-up below.** A non-identity
    production map carries measured alignment evidence.
    The runner measures the alignment from the run's own placement/volume coordinates
    through a new pure coordinate-only search,
    `best_visible_alignment_for_coordinates`, which mirrors
    `evaluation.best_visible_alignment`'s permitted transform family and
    candidate/vote search but ranks by coordinate matches alone (there are no labels
    to rotate). The old runner could substitute an identity record
    (`no_route_b_candidate_cells_identity_alignment`) or use `compute_alignment` with
    a `{coordinate: None}` candidate that raises; both paths are gone, and a run that
    carried no coordinate evidence refused typed
    (`route_b_measured_alignment_absent`) instead of silently scoring an identity
    alignment at that snapshot. The follow-up section replaces this no-candidate
    behavior with a typed-unavailable stage ledger and null alignment.
16. **Production runs before every evaluator file.** `main` now executes the
    production route (Route A or Route B) before it opens the evaluator pin
    manifest, reference, scene, cameras or frozen masks, and a runner-level
    regression with a production-call sentinel, a pin-loader sentinel and a
    truth-loader sentinel proves the order **production call < pin loader < truth
    loader** for both routes. The end-to-end pin-mismatch regression stubs the
    production route so the fail-closed mask-domain refusal is still exercised under
    the corrected order.
17. **Initial regressions.** Five focused checks were added (68 -> 73 passing): the real
    `ResolvedFacePlacement`/`PlaneVolumeResult`/`PromotionResult` constructors drive
    the runner seam and assert the builder receives `GeometryCellEvidence` plus the
    real promoted states (placement/volume/promoted all 3/3 over `V`); a non-identity
    measured alignment over real placement coordinates is applied exactly once (full
    G recall) while the search spy proves it consumed run-native coordinates; a
    carried-empty versus absent promotion keeps reach `true`/`null` separate from a
    null metric; the coordinate-only scorer's counts equal `visible_metrics`; and the
    coordinate-only search recovers a non-identity transform. `py_compile` on the
    three owned Python paths exits 0. No route was replayed, so the Route A
    `901408b7...` / Route B retained digests remain **pre-revision provenance**.

This report contains real, labelled numerators over declared denominators for Route
A on one frozen fixture. **No pooled "image-only accuracy" is claimed**, no
accuracy is derived from a status/policy count, and the first unreached checkpoint
is reported only as a frontier indicator, never as a causal attribution of loss.

## 1. Initial R7 implementation snapshot: owned files and SHA-256

These hashes identify the initial implementation reviewed before the alignment/no-tail
follow-up. They are historical; use the current hashes in the dated follow-up section below.

| Path | SHA-256 |
| --- | --- |
| `src/mc_photogrammetry/evaluation_image_only_attribution.py` | `DAC90748326AC3266149935E6F0C91B3E240AF1D7ABA49AE22CB916AFA84E260` |
| `scripts/evaluation/report_image_only_attribution.py` | `F11B56C1CFCDA649159D8A0F6647E7F77DC372943A9185666B2814CB6ABFA10B` |
| `tests/test_evaluation_image_only_attribution.py` | `316F6852C58B950928AC643A80956913C6DF993A5E0D19F325010A74D6383C60` |

No A1 file was edited. The four accepted A1 hashes still match the A1 report
exactly (`4595124df8...d0d4c`, `92c89f17...52059`, `eb3edaa1...11d4b`,
`82d3c2a7...14c50`). The accepted design is unchanged
(`21B74E9B7BF2DCC0217C4012C964AA72033D4C1963464C9BBEBAD32008CBDC52`). The
pre-existing unrelated dirty hunks in the worktree were preserved; no production
(`pipeline/`, `reconstruction/`, `cli.py`, `project/`, `export/`), schema, gate,
threshold, budget or palette file was touched. The earlier checkpoint-reach revision
did not edit the runner and it serializes `attribution.to_record()` verbatim, so the
`checkpoint_reached` field flows into its report automatically; the A2 R3 adapter
repair, the A2 R5 promotion-presence repair and the A2 R7 real-record/coordinate-only
adapter repair all own and edit the runner (the pre-alignment removal, the explicit
promotion-record presence, the real `output.placement`/`output.volume`/`output.promotion`
adapters, the coordinate-only alignment and the production-before-truth ordering). The
A2 R6 palette-ceiling repair and its validation/immutable-snapshot addenda edit the
pure module and its focused test module only. The hashes above are the post-R7 values
for all three owned Python paths.

The three tracked reviewed mask specifications the corrected runs loaded:

| Specification | File SHA-256 (equals the pinned `usable_masks[].file_sha256`) |
| --- | --- |
| `docs/evaluation_masks/templates-2/2026-09-02_13.22.14.json` | `2B0E18CFC26D225C6BB5CB1043792CDCC252983215B5FB7FD09480E88622850E` |
| `docs/evaluation_masks/templates-2/2026-09-02_13.24.24.json` | `068C7FE623E66D770E77AF98538916EAFE34A7F4E7B0F3D7080ED5FA5D5687C7` |
| `docs/evaluation_masks/templates-2/2026-09-02_13.24.33.json` | `212A5D7E49B66D14EED26A829F75440E15B20B8281B433907C7F4319FD830DD2` |

This report is the A2 write target; the runner wrote only `.tmp-a2-*-tracked-*`
temporary artifacts (retained for verification, see §7).

## 2. Exact scope and evaluator-input confirmation (pre-work check)

**Scope.** Implement accepted Tranche A2 only: an evaluation-only module that scores
both routes' reachable checkpoints against frozen evaluator-only truth, a
Route-A/Route-B replay runner, a focused test module, and this report. Route B first
by routing priority but not exclusively; Route A (the normal-mode default CLI) is
measured as well. The five labelled spaces (G, O, P, C, E) stay separate with fixed
universes, the primary E denominator is all visible `V` at `>= 1` usable pixel, the
permitted alignment is one global integer translation + lattice Y-quarter-turn with
**no scale fit**, and camera diagnostics carry no cell denominator and no
cell-loss equation.

**Required evaluator inputs - availability confirmed before implementation:**

| Input | Source | Status |
| --- | --- | --- |
| Frozen reference litematic | `templates 2/Template 2.litematic` | **available**; SHA-256 `519034a6a439ab98145aa820792d5afc588b5d697ecad961c14c36cbef2f3416` equals the tracked `template-2` pin; 852 blocks |
| Frozen frames | `templates 2/*.png` (3) | **available**; all three content digests equal the tracked pins |
| Evaluator cameras | `templates 2/cameras.json` | **available**; 3 cameras, 2560x1369 frames, fov 110 |
| Evaluator scene transform | `templates 2/scene.json` | **available**; world origin `(-1008, 107, 8014)`, size `(9, 12, 17)` |
| Tracked frozen mask specifications | `docs/evaluation_masks/templates-2/*.json` (3) | **available and pinned**; loaded through the accepted evaluator rectangle/polygon semantics, every generated scored domain matches the reviewed pin exactly (see §6) |
| Local mask PNG derivatives | `.masks/templates-2/*.png` | present but **not scored**: identity-divergent and gitignored; retained only as an explicit fallback when no tracked specification exists |
| Pinned expected identity | `docs/reports/evaluator-fixture-pins.json` `template-2` | reference pin **matches**, frame pins **match**, mask domain pins **match** the tracked specifications |
| Accepted A1 frontier/runners | `evaluation_image_only_frontier.py`, both replay runners | **available**, hashes match the A1 report |
| Raycast evaluator | `export.visibility.visible_cells`, `evaluation.best_visible_alignment`, `evaluation.visible_metrics`, `evaluation_pair_correspondence.metric_value` | **available**; `cv2`/`numpy`/`Pillow` present, no JAR needed for a visibility-domain raycast |
| Evaluator pair-correspondence seam | `evaluation_pair_correspondence.construct_truth_pairs`, `evaluate_scene`, `FinalizedScene`/`FrameInput`, `reconstruction.scene.SceneTransform.to_world` | **available**, but matcher-distinct from the Route B entry; supplies only the labelled `evaluator_pair_control` diagnostic over the pinned reference occupancy (local mapped to world), the frozen frames and the evaluator cameras, never the Route B entry P row |
| Frozen entry admitted matches | `ImageDerivedEntryResult.graph.evidence[].matches_sha256` | **digest only** (no pixel/keypoint-pair identities carried); the Route B entry P row therefore has no digest-bound joint set to score and is `metric: null` with reason `entry_match_identities_not_retained_or_digest_unbound`, while the evaluator's own mutual-match pool is retained only as the separate `evaluator_pair_control` diagnostic |

The Route A normal-mode CLI was confirmed to accept the frozen raster (stdout
`Created ... with 690 block(s)` in the A1 parity run, reproduced here).

## 3. Contract as implemented

* **No production access.** `evaluation_image_only_attribution.py` imports no
  `pipeline`/`cli`/`project.store`/`reconstruction`/`export`/`pathlib`/`os`/
  `tempfile`; it never opens a path, never calls an exporter or reconstruction body,
  and only reads fields the accepted in-memory records expose. The only artifact
  permitted to read frozen evaluator paths is
  `scripts/evaluation/report_image_only_attribution.py`; production cannot import it.
* **Five labelled spaces.** `MetricSpace` = G/O/P/C/E plus the camera diagnostic;
  `UNIVERSE_OF_SPACE` fixes reference cells for G/C/E, frame pixels for O and
  pixel-pair identities for P. `AttributionSection` refuses a row or transition from
  another space, so nothing is summed or interchanged across spaces.
* **Primary E denominator is all visible `V`**; `V inter palette` is a separate
  conditional row in the same universe, and a truth state outside the run palette is
  unreachable at every candidate depth `K`.
* **Typed unavailable, never a measured zero.** An unreached row carries
  `metric_value: null`, `ceiling: null`, `conditional_denominator: null` and a typed
  reason; `AttributionRow` refuses a non-null metric on an unreached row. Every rate
  goes through `evaluation_pair_correspondence.metric_value`, so an undefined
  denominator is `null`, never `0.0`.
* **Checkpoint reach is a separate serialized fact from metric availability.**
  `AttributionRow.reached` keeps its compatibility meaning (a metric record was
  produced). Every row additionally serializes `checkpoint_reached`, which states
  whether the row's *production checkpoint* was reached under the accepted A1
  frontier's own `reachable` semantics, independent of whether the metric could be
  scored. It is `true` for a reached-but-unscorable checkpoint, `false` for a
  checkpoint the run never reached, and `null` when the row names no production
  checkpoint (aggregate/evaluator-only rows). The serializer never infers a metric
  from a reached checkpoint, nor an unreached checkpoint from a null metric: the
  Route B `entry`/B0 P row is `checkpoint_reached: true`, `reached: false`,
  `metric_value: null`. A row naming a Route B **promotion** (`promoted`) or
  **decoded-export** (`terminal`) substage also serializes `null` unless its own
  carried record proves that substage, because A1 has no dedicated promotion/export
  checkpoint: the coarse `output` reach is never copied onto them, and absence is
  never inferred as `false` where it may mean unattempted/unknown. Presence is
  explicit record existence (`record is not None`), never mapping truthiness: a
  carried empty record proves reach `true` beside an unavailable metric.
* **Permitted alignment only.** `alignment_from_record` rejects any `scale`,
  `scale_fit`, `reflection`, `per_cell`, `stretch` field and any transform string
  other than the accepted integer-translation + Y-quarter-turn; `align_candidate`
  maps the **candidate** into the reference frame exactly as
  `evaluation.visible_metrics` does, including the directional state rotation.
* **Reconciled transitions.** `next = previous + gained - lost` inside one universe.
  A `G` transition reconciles the geometry success set (`V inter candidate`) and is
  **unpartitioned**; cell-level exact-state `E` loss partitions disjointly into
  `geometry_lost` and `state_lost`, and an off-space partition is refused. A `G`
  step therefore never carries a `geometry_lost`/`state_lost` pair. On Route B only
  `placed -> volume_cell` is a `G` step; both `raw_winner -> promoted` and
  `promoted -> terminal` are exact-state `E` steps and carry the disjoint partition.
  A camera diagnostic cannot carry a transition.
* **The reachable P entry is digest-bound, never an evaluator alias.** Route B's
  reachable `entry`/B0 checkpoint is scoreable in the frozen frame-pair / pixel-pair
  universe only by `pair_metric_rows` over an `EntryPairEvidence` whose
  `EntryMatchBinding` binds the aggregate to the *same* joint matcher the entry used
  (`joint_matching.match_and_solve_pair`) with a rederived match-array digest equal
  to the accepted `RelativeCameraEvidence.matches_sha256`. `pair_metric_rows` raises
  an `AttributionRefusal` on any unbound aggregate, and `route_b_attribution` emits
  the typed unavailable row with `metric: null` and reason
  `entry_match_identities_not_retained_or_digest_unbound` instead. Entry reach is
  checked **before** the binding, so a frontier that never reached entry fails closed
  to reason `entry_unreached` with `checkpoint_reached: false` and emits no numeric
  metric even if a digest-bound aggregate is supplied; `pair_metric_rows` refuses an
  explicitly unreached checkpoint directly. That typed
  unavailable row on a *reached* entry carries `checkpoint_reached: true` (from the
  accepted frontier's entry reach) beside `reached: false`, so metric unavailability
  is never read as an unreached checkpoint. The primary row
  would be `truth_pairs_returned / truth_pairs` over jointly visible strict
  same-surface truth pixel pairs, with the accepted `positive / returned` precision
  row alongside; a zero truth denominator is `metric_value`'s typed `null`
  (`no_truth_keypoint_pairs`), never a measured zero. The evaluator's own
  `evaluate_scene`/`select_mutual_matches` aggregate is a named `control` diagnostic,
  excluded from the attribution and loss funnel. Entry connectivity is never turned
  into accuracy and P is never divided by `|V|`.
* **Run-native inputs, one transform.** Every cell map, proposal record, ranked
  candidate list and palette a Route B builder receives is in the run's own frame.
  The builder applies the one permitted alignment (`align_candidate` for cell maps,
  `align_candidate_rankings` for ranked candidate lists, `align_palette_states` for
  the coordinate-free palette) exactly once and joins the result against
  reference-frame truth; a caller never pre-aligns those records, and a doubly
  transformed map is never scored. Proposal cells and **every** ranked `BlockState`
  transform together, so a directional candidate is never compared in an unaligned
  orientation, while ranking order and cost association are preserved. Every palette
  `BlockState` receives the same directional Y-quarter-turn rotation (the
  alignment's translation has no coordinate to move), so the C palette-coverage row
  and the E in-palette conditional row can never disagree with the aligned
  candidate/E rows for a directional state; a palette state the transformed run
  palette still lacks leaves the primary `|V|` denominator unchanged.
* **Availability, not accuracy.** Candidate rows reuse the accepted
  `_candidate_availability` semantics: a cell with no candidate record is charged to
  geometry, a beyond-palette truth state to the palette, and a truth state that sits
  in the ranked list but loses the argmin is *available* (a ceiling, not a label).
  The C rows are built from a genuine `MetricSpace.CANDIDATE` `CellScore` over the
  same frozen visible-cell universe as G/E (never from a G-space base score), and a
  missing base/proposal record stays a typed-unavailable row, never a zero; the
  strict `candidate_metric_rows` C-space guard is unchanged.
* **Tracked mask specifications with fail-closed pins.** The scored mask is loaded
  from the reviewed tracked JSON specification for the frame
  (`resolve_mask_path` prefers `<stem>.json`, then an exact-name PNG fallback) with
  the accepted evaluator semantics: all-`True` domain, integer
  `excluded_rectangles` cleared, then `excluded_polygons` filled `False`, exactly as
  `benchmark_templates._load_usable_masks` builds them. When reviewed pins are
  supplied, `require_pins` defaults to on: every frame's observed scored
  `domain_sha256` must equal the pinned value or the run raises a typed refusal and
  builds no attribution. `--allow-pin-mismatch` is the only way to downgrade that
  to a caveat, and a pinned run that checks no pinned domain at all also refuses.
* **Completion/refusal rates** are reported per run as labelled policy rates by
  terminal stage, never pooled and never accuracy.

## 4. Corrected frozen Route A measurement (one run)

Command (normal-mode default; temporary workdir; tracked mask specifications):

```
.\.venv\Scripts\python.exe scripts\evaluation\report_image_only_attribution.py ^
  --route route-a --image "templates 2\2026-09-02_13.22.14.png" ^
  --reference "templates 2\Template 2.litematic" --scene "templates 2\scene.json" ^
  --cameras "templates 2\cameras.json" ^
  --masks "docs\evaluation_masks\templates-2" ^
  --pins docs\reports\evaluator-fixture-pins.json --fixture-role template-2 ^
  --workdir .tmp-a2-route-a-tracked-20261003 --cli-parity ^
  --report .tmp-a2-route-a-tracked-20261003.json
```

Exit 0. `mask_origin = tracked_json_specification`; the single frame resolved to
`docs/evaluation_masks/templates-2/2026-09-02_13.22.14.json`, observed domain
`c1536420ce3f3b7fe9c4cf92a2221313b712542752c20dcec51184dd0ca08245` equals the
reviewed pin (`mask_pins_match = true`, `mask_pinned_domains_checked = 1`, no
mask caveat). Frozen truth: reference 852 cells, frozen visible `V` = **398** cells
at `minimum_pixels = 1` (single frame); the tracked specification excludes the HUD
and overlay regions, so `V` is two cells below the prior `.masks/*.png`-derived
401. Alignment **Y-quarter-turn 1, translation `(18, 13, 22)`**, source
`evaluation.best_visible_alignment`. Attribution digest
`901408b77fc9b3f10c00dbf08cf0517643008610802261d459770bfef342e56e` (pre-revision
serialization; every row below is `checkpoint_reached: true`, because all five
Route A frontier checkpoints were reached, and the added field re-baselines the
record digest as described above).

| Space | Checkpoint | Metric | Numerator | Denominator | Value |
| --- | --- | --- | --- | --- | --- |
| G | `facade_grid` | geometry recall `\|C∩V\|/\|V\|` | 16 | 398 | 0.040201 |
| G | `facade_grid` | geometry precision `\|C∩V\|/(\|C∩V\|+\|C−R\|)` | 16 | 687 | 0.023290 |
| C | `palette_label` | palette coverage `\|V∩palette\|/\|V\|` | 0 | 398 | 0.0 |
| C | `palette_label` | in-palette conditional (same universe) | 0 | 0 | **null** |
| E | `palette_label` | exact-state recovery over all `V` | 0 | 398 | 0.0 |
| E | `palette_label` | `V∩palette` conditional vocabulary row | 0 | 0 | **null** |
| E | `terminal` | exact-state recovery over all `V` | 0 | 398 | 0.0 |
| E | `terminal` | `V∩palette` conditional vocabulary row | 0 | 0 | **null** |
| camera | `camera` | diagnostic (solved) | - | none | null |

Transition (E, `frozen_reference_cells`): `palette_label -> terminal`, previous 0,
gained 0, lost 0, net 0, `geometry_lost` 0, `state_lost` 0. Policy rates: completion
5/5 = 1.0, refusal 0/5 = 0.0, terminal stage `terminal`. O and P are reported as
explicitly empty sections. Caveats recorded verbatim: unresolved `scale_known=False`
on every G row (reported as score-reducing, **not** as unavailable) and the
permitted-alignment statement. No mask-identity caveat remains.

**Route A normal-mode CLI parity (same frozen input, one CLI run):** the runner's
decoded `.litematic` and the normal CLI's decoded `.litematic` have equal
coordinate/state sets under only the exporter region-origin normalization
(`normalized_sets_equal = true`, 690 blocks each, only-CLI 0, only-runner 0, state
mismatches 0, shared normalized SHA-256
`81e3efa8f55f53b33e23db8f46f89b51f9f2b22accd4c865563fe6951ecec73e`). This is
route/status/parity evidence only; it is not a visible-truth number.

## 5. Corrected frozen Route B measurement (one run)

Command:

```
.\.venv\Scripts\python.exe scripts\evaluation\report_image_only_attribution.py ^
  --route route-b --fixture "templates 2" ^
  --sources "templates 2\2026-09-02_13.22.14.png" ^
            "templates 2\2026-09-02_13.24.24.png" ^
            "templates 2\2026-09-02_13.24.33.png" ^
  --reference "templates 2\Template 2.litematic" --scene "templates 2\scene.json" ^
  --cameras "templates 2\cameras.json" ^
  --masks "docs\evaluation_masks\templates-2" ^
  --pins docs\reports\evaluator-fixture-pins.json --fixture-role template-2 ^
  --report .tmp-a2-route-b-tracked-20261003.json
```

Exit 0. `mask_origin = tracked_json_specification`; all three frames resolved to
their tracked specifications and all three observed domains equal the reviewed pins
(`mask_pins_match = true`, `mask_pinned_domains_checked = 3`, no mask caveat).
Frozen visible `V` over all three frames = **621** cells (the tracked
specifications reproduce the same scored domains the pins describe, and the
trajectory still stops before any placement cell is scored; this run is the one
corrected Route B evaluation for this revision). Verified terminal (from the
accepted result and the persisted schema-3 envelope, which replays the returned
result): `geometry` / `geometry_unavailable` / `g_unavailable`, detail
`plane_integer_residual`; `entry_sha256
2a67e20f3eda89b7def2ae30c2f29f96be3e35894002c1b1908f26a3ee1c62d1`,
`lattice_sha256 6f4cba955d1ad2fdc1a7a7d059589c09000752ce7edd6cc2ccb85d83448b3fd4`,
`geometry_sha256 0b9680a168dddd24a56e05d5b7ccf832a9ef63c248c69a1070ce2a84c5be4f2d`,
`output_sha256` empty, session unchanged. The terminal tuple is the same one A1
recorded; `lattice_sha256` and `geometry_sha256` match the A1 report byte-for-byte.

**The Route B `entry`/B0 checkpoint is reached, while its P metric is unavailable
(`metric: null`), not the evaluator's `0/232`/`0/33`.** The accepted frontier
reaches `entry`/B0, but `evaluation_pair_correspondence.evaluate_scene` /
`select_mutual_matches` is a different matcher from the entry's
`joint_matching.match_and_solve_pair`, so its aggregate is not an achieved Route B
P metric. The entry row may only be scored from a digest-bound joint match set: a
rederived match array whose digest equals the accepted
`RelativeCameraEvidence.matches_sha256`. The retained corrected run carries only
`entry_sha256`, `lattice_sha256`, `geometry_sha256` and the evaluator control
totals; the pre-revision runner did not serialize each pair's
`match_count`/`mutual_survivors`/`matches_sha256`, and the Route B project target
was temporary, so no bit-identical joint replay can be bound without a route
rerun, which this hold forbids. This revision therefore separates the two facts at
the row level: the in-`attribution` P row serializes `checkpoint_reached: true`
(and `reached: false` for the metric), so an unavailable P metric can no longer be
read as the entry checkpoint never having run. The current code therefore reports:

| Route B entry/P attribution field | Value |
| --- | --- |
| accepted frontier `entry` checkpoint reach | `true` (the run carried the entry record) |
| in-`attribution` P row `checkpoint_reached` | `true` |
| in-`attribution` P row `reached` (metric availability) | `false` |
| in-`attribution` P row `metric_value` / `ceiling` / `conditional_denominator` | `null` / `null` / `null` |
| in-`attribution` P row `typed_reason` | `entry_match_identities_not_retained_or_digest_unbound` |
| `entry_pair_attribution.available` | `false` |
| `entry_pair_attribution.metric` | `null` |
| `entry_pair_attribution.reason` | `entry_match_identities_not_retained_or_digest_unbound` |
| `entry_pair_attribution.entry_binding.entry_reachable` | `true` |

The runner-level `entry_pair_attribution` summary keeps its own
`entry_binding.entry_reachable` field; the row-level `checkpoint_reached` above is
the canonical serialized statement inside the attribution record. No runner edit
was needed or authorized for this correction, because the runner serializes
`attribution.to_record()` verbatim. An entry the run never reached serializes
`checkpoint_reached: false` with the typed reason `entry_unreached`, and in that
case reach is checked **before** the evidence binding, so a digest-bound aggregate
is never scored and no numeric P metric is emitted.

The generic evaluator aggregate is retained only as a separately named control:

| Control field | Value |
| --- | --- |
| `evaluator_pair_control.role` | `control`, `excluded_from_route_b_attribution: true` |
| `evaluator_pair_control.status` | measured over its own declared denominator |
| evaluator truth pairs | 232 |
| evaluator returned pixel-pair identities | 33 |
| evaluator positive | 0 |
| coverage | 3 frames, 3 unordered pairs, 3773 keypoints, 1973 with a truth surface |

Those control numbers are a **single-fixture evaluator diagnostic over its own
declared denominator**; they are never attached to the entry row and never enter
the Route B attribution or loss funnel. The control denominator uses the accepted
evaluator's own usable-pixel mask (`usable_mask_for`), not the tracked
visible-cell mask, so it is not the same mask domain as the 621-cell `V`, and it
is never compared with `|V|`. `route_b_evaluator_pair_control` keeps its typed
fail-closed branches (`entry_pair_frames_below_two`, `entry_pair_camera_missing`,
`entry_pair_reference_empty`), unit-tested with synthetic truth maps.

The `matches_sha256`/`match_count` values the hold asked to include as context are
**not present in the retained pre-revision artifacts**: the prior runner did not
serialize `ImageDerivedEntryResult.graph.evidence`, and the Route B project was a
temporary target that no longer exists. The corrected runner now records them
under `entry_pair_attribution.retained_match_evidence` (counts and digests only,
`identities_retained: false`) on any future run, but this revision intentionally
performed no route rerun, so no such values are claimed here. The only retained
entry identity is `entry_sha256
2a67e20f3eda89b7def2ae30c2f29f96be3e35894002c1b1908f26a3ee1c62d1`.

Because no Stage-7 placement/volume tail record exists on this genuine run,
**every** Route B cell checkpoint is typed unavailable and contributes no zero:
`placed`/`volume_cell` (`g_unavailable`), `raw_winner`/`promoted`/`terminal`
(`record_absent`), `palette`/`candidate_k` (`palette_unknown`). The `promoted` and
`terminal` rows serialize `checkpoint_reached: null` here (their substage records are
absent), because A1 has no dedicated promotion or decoded-export checkpoint and the
coarse `output` reach is never copied onto them; on a synthetic carried tail those
rows serialize `checkpoint_reached: true` from their own record. The O section is
explicitly empty with its own note, and the P section carries one typed-unavailable
`entry` row with `metric_value: null`. Frontier stop `output`
(`output_record_absent`); policy completion 3/4 = 0.75, refusal 1/4 = 0.25,
terminal stage `geometry`. The retained Route B JSON/stdout pair predates this
contract correction: its `pair_correspondence` block (with the `0/232`/`0/33`
evaluator rows) and its `attribution_sha256`
`eb1d942d9f4eb1e686f3c1106ff4118438fbbc6573da607a11aa065e481bf016` reflect the
superseded measured-P-row shape, not the current digest-bound contract. No route
rerun was performed, so no new Route B `attribution_sha256` is claimed; the
retained file hash `3C48715D5E60E0866FC38BE29895432AE07BE7AE0D7E57A6D22EA045E2BF127E`
is unchanged pre-revision provenance. Route B produced no visible-truth *cell*
number, and none is fabricated; the evaluator control numbers live only in their
own `control` diagnostic and are never a Route B achieved P metric.

**Observed `entry_sha256` discrepancy (observation only).** Frozen Route B runs
report different `entry_sha256` values: the accepted A1 report
`e38546aa07e6decad3e3f09c4abe3963caed6c5b8e8e5630f1fc77fd26f270ea`, the earlier A2
reports `9c156d81...` and `4d93225f...`, and this corrected run
`2a67e20f3eda89b7def2ae30c2f29f96be3e35894002c1b1908f26a3ee1c62d1`. The existing
source contract shows only that `entry_sha256` canonicalizes
`_entry_payload`, which includes `project_path` and `source_paths`
(`image_derived_entry.py:927-955`); this report does **not** claim that those
bindings explain the difference, because no controlled comparison was run. The
digests that gate A2 scoring (`lattice_sha256`, `geometry_sha256`) are unchanged
across the runs, so the corrected mask provenance does not affect the Route B
terminal or its unavailable tail.

## 6. Findings

1. **The tracked mask specifications are the pinned source of truth and now match
   exactly.** All three `docs/evaluation_masks/templates-2/*.json` specifications,
   loaded through the accepted evaluator rectangle/polygon semantics, generate the
   reviewed `usable_masks[].domain_sha256` values
   (`c1536420...`, `dfe16f05...`, `f8260d64...`) and their file SHA-256 values equal
   the pinned `file_sha256` values. The ignored `.masks/*.png` derivatives are
   identity-divergent and are no longer scored; the corrected runs report
   `mask_origin = tracked_json_specification` with no mask-identity caveat. A
   supplied reviewed pin is mandatory: a divergent scored domain, or a pinned run
   that checks no pinned domain, refuses with exit 2 and writes no report.
2. **Route A's geometry is measurable but small, and the observed alignment search
   is not the limiter for this run.** The facade's 690 cells occupy a coordinate
   range far wider than the 9x12x17 reference (x -80..35, y -12..48, z -42..83 vs
   x 0..8, y 0..11, z 1..16). The accepted alignment search returned Y-quarter-turn
   1 and translation `(18, 13, 22)`, and the scored overlap is 16 of 398 visible
   cells. The unresolved metric scale reduces this recall to ~4% rather than merely
   perturbing it, exactly as the design anticipated ("unknown scale reduces score",
   not "unscorable"). **No global or theoretical ceiling is claimed:** the observed
   alignment and score are reported as measured, and this single-fixture run does
   not establish what any wider or unbounded search would find.
3. **Template 2 truth states are outside the default 8-state run palette.**
   `V∩palette = 0` of 398 with `DEFAULT_PALETTE`, so the in-palette conditional
   denominator is legitimately empty (`null`, not 0.0) and all E rows are 0/398.
   This is the measured palette-availability ceiling for this run, not an
   implementation failure; a normal-CLI JAR palette would change the C rows (and is
   not developer-gated) but no JAR was used or read here.
4. **Route B's genuine frontier is a geometry stop, so A2's Route B cell tail is
   unmeasurable on the frozen fixture, and its reachable P entry has no
   digest-bound joint set.** The evaluation-only ledger reports the cell tail as
   typed unavailable rather than scoring fabricated coordinates; the exact missing
   seam is the Stage-7 placement/volume tail the accepted design already
   identified. The `entry`/B0 checkpoint is reached, but the retained
   `RelativeCameraEvidence` keeps only counts and `matches_sha256` - no
   pixel/keypoint-pair identities - so the entry P row is `metric: null` with
   reason `entry_match_identities_not_retained_or_digest_unbound` **and
   `checkpoint_reached: true`**: the row now states entry reach and metric
   unavailability as two separate serialized facts, so the null metric is not read
   as an unreached checkpoint. A synthetic frontier that never reaches entry
   serializes `checkpoint_reached: false`. `V` is 621 and the terminal tuple and
   gating digests are unchanged.
5. **O is not scoreable from the frozen A2 assets; P is controlled, not achieved.**
   The frozen assets carry no evaluator-derived `(cell, slot, frame)`
   surface-ownership map, so the O section is reported explicitly empty with a
   typed reason instead of being substituted by a cell count. The evaluator's own
   `evaluate_scene`/`select_mutual_matches` pool (232 truth pairs, 33 returned) is
   a different matcher from the entry's `joint_matching.match_and_solve_pair`, so
   it is kept only under `evaluator_pair_control`, excluded from the Route B
   attribution and loss funnel; its denominator uses the evaluator's own
   usable-pixel mask and is never compared with `|V|`.
6. **The frozen `entry_sha256` differs across the accepted A1 run, the earlier A2
   runs and this corrected run.** It is reported as an observed discrepancy with the
   contract fact that `entry_sha256` canonicalizes `project_path` and
   `source_paths`; no causal explanation is asserted (§5).
7. **A `G` geometry transition is unpartitioned; every `E` step is partitioned.**
   `placed` and `volume_cell` are both geometry, so the step reconciles `matched`
   cells and carries `geometry_lost = state_lost = null`; a regression carries both
   geometry checkpoints and proves `previous + gained - lost` reconciles without a
   state partition. Route B's two exact-state steps (`raw_winner -> promoted`,
   `promoted -> terminal`) carry the disjoint `geometry_lost`/`state_lost` partition,
   verified by a new route-level regression carrying both E endpoints and both loss
   kinds that sum exactly to `lost`.
8. **Reach is gated before evidence, and coarse stage reach is not copied onto
   uncheckpointed substages.** A digest-bound entry aggregate is never scored when
   the accepted frontier did not reach entry (`entry_unreached`,
   `checkpoint_reached: false`, no numeric metric); and Route B's `promoted` and
   `terminal` rows serialize `checkpoint_reached: null` unless their own carried
   record proves the substage, because A1's `output` checkpoint is not a promotion
   or decoded-export checkpoint. Both are covered by new focused regressions.
9. **The palette is part of the aligned run frame.** The run-native palette is
   rotated by the same permitted Y-quarter-turn state mapping as the candidate
   (`align_palette_states`; the translation cannot affect a coordinate-free
   palette), so a directional truth state is never scored against an unrotated
   palette. Before this repair, a run-native `oak_stairs[facing=north]` palette
   under quarter-turn 1 with a `facing=east` truth produced a C palette row that
   disagreed with the aligned candidate/E rows (2/4 versus 3/4). The corrected and
   new adversarial tests both fail when the rotation is reverted, and the
   out-of-palette `GOLD` cell confirms the primary `|V|` denominator is not reduced
   by the conditional vocabulary.

## 7. Verification commands, results and artifacts

```
.\.venv\Scripts\python.exe -m py_compile ^
  src\mc_photogrammetry\evaluation_image_only_attribution.py ^
  scripts\evaluation\report_image_only_attribution.py ^
  tests\test_evaluation_image_only_attribution.py
# exit 0 (the A2 R7 real-record adapter repair edits the module, the runner and the
# tests; the A2 R6 addenda edited only the module and its tests)

$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m pytest -o addopts='' -q ^
  tests\test_evaluation_image_only_attribution.py --no-header
# 73 passed in 0.51s
```

The A2 R3 adapter repair added three focused regressions (52 -> 55):
`align_candidate_rankings` transforms a cell and every ranked directional state
exactly once in order and refuses a forbidden alignment field; a route-level
palette+proposals builder run over a non-identity translation + Y-quarter-turn emits
measured C rows (`palette_coverage` 3/3, `candidate_available_k1/k2/k4` = 2/3/3)
with full `placed` geometry recall instead of refusing; and a runner-level
regression captures the builder inputs and asserts the runner passes the run-native
maps unmodified and one transform reproduces the reference (the pre-fix
runner+builder double transform scores an overlap of 1, verified against the
simulated pre-fix adapter).

The A2 R4 palette-alignment repair added three focused regressions (55 -> 58):
`align_palette_states` rotates directional states only, ignores the translation,
is iteration-order independent and refuses a forbidden alignment field; the
standalone `palette_coverage` helper rotates a run-native directional palette
(2/2); and a new route-level adversarial regression passes a run-native
`oak_stairs` palette, placement and proposals under quarter-turn 3 plus translation
and asserts palette coverage 3/4, `palette_coverage_in_palette` 3/3,
`candidate_available_k1` 3/4 and `raw_winner` exact-state 3/4 with its in-palette
conditional 3/3, while the deliberately out-of-palette `GOLD` truth cell keeps the
full `|V|` denominator. The prior synthetic adapter test (whose palette was
pre-rotated and therefore masked the defect) now supplies the run-native palette and
is asserted at `palette_coverage` 3/3. Both the corrected test and the new one were
re-verified to fail with `align_palette_states` reverted to the identity, proving
they exercise the palette alignment rather than a no-op.

The A2 R5 runner promotion-presence repair added two focused runner-seam
regressions (58 -> 60): a carried promotion record with zero cells reaches the
builder as `{}` and serializes `promoted` with `checkpoint_reached: true`, `reached:
false` and `metric_value: null` (this regression fails on the pre-fix
`promoted_states or None` runner, confirmed by temporarily reverting that
expression); and an absent promotion record reaches the builder as `None` and
serializes `checkpoint_reached: null` with the metric unavailable. Both assert the
runner-supplied `terminal` substage stays `null` because no decoded terminal record
is carried.

The A2 R6 palette-ceiling repair strengthened two existing regressions and added two
focused ones (60 -> 62), and changed two fixtures that had ranked out-of-palette
states: the conditional exact-state test now asserts `E inter V_P` (`2/2`, value
`1.0`) beside the unchanged primary `3/3`, and the beyond-palette candidate test
asserts the numerator never exceeds `palette_coverage` while `|V|` stays `2`; the
new `test_candidate_ceiling_refuses_an_out_of_palette_ranked_candidate` proves a
`GOLD` truth cell with palette `{STONE}` and a ranked list containing `GOLD` refuses
atomically (including when the out-of-palette entry sits below the top-`K` cut,
proving the full list is validated before slicing), while an all-palette ranked list
is a normal zero row charged to the palette; the new
`test_candidate_ceiling_refuses_missing_palette_membership_evidence` proves a C score
with no palette evidence is typed unknown and refuses (both with and without a
ranked map), and that an empty palette refuses any ranked candidate as a
contradiction. The two out-of-palette fixtures were corrected to in-palette
distractors (`test_candidate_availability_counts_a_state_that_loses_the_argmin`, the
route-level directional adapter test and the runner input-capture test), because a
run can only rank states its own palette holds. All three mutation controls were
verified: reverting the conditional numerator to the whole `E` set fails the
`2/2` assertion (`assert 3 == 2`); removing the ranked-candidate validation and the
`V_P` filter fails the contradiction regression with `DID NOT RAISE`; and removing
the palette-evidence guard fails the missing-evidence regression with a
`TypeError` instead of the typed `AttributionRefusal`. The fix was restored and
re-verified in each case (`62 passed`).

The A2 R6 validation addendum (both the exact-`V_P` follow-up and the Astra whole-`E`
fallback finding) added three focused record-level regressions (62 -> 65):
`test_cell_score_refuses_a_forged_vp_membership_set` builds a valid C score with
palette `{STONE, DIRT}` over visible truth `{(0,0,0):STONE, (1,0,0):DIRT,
(2,0,0):GOLD}`, then forges an incomplete `V_P`, a superset containing the
out-of-palette `GOLD` cell, a set naming a coordinate outside the truth context and a
same-size swap replacing an eligible cell with `GOLD`, each with consistent aggregate
counts so only the exact-membership recomputation can refuse it, while an exactly
recomputed set is still accepted;
`test_cell_score_refuses_palette_evidence_without_truth_context` proves palette
evidence with `coordinate_truth=None` (or a non-mapping context) refuses while a
no-palette score stays typed unknown; and
`test_cell_score_refuses_palette_aggregates_without_complete_context` proves a
replacement or direct constructor carrying `palette_coverage`/`conditional_denominator`
with no vocabulary/`V_P` refuses, that a valid palette score's conditional row is
exactly `2/2` (never the whole-`E` `3/2`), and that an absent palette yields no
conditional row at all. Mutation controls: restoring the pre-fix subset check makes
the two evidence regressions fail with `DID NOT RAISE AttributionRefusal` (`2 failed,
62 deselected`); disabling the aggregate-completeness check and restoring the
`len(exact)` fallback makes the aggregate regression fail with `DID NOT RAISE
AttributionRefusal`; the fix was restored and re-verified each time (`65 passed`).

The A2 R6 immutable-snapshot addendum added three focused record-level regressions
(65 -> 68): `test_cell_score_truth_snapshot_is_read_only_and_caller_isolated` builds a
score through `dataclasses.replace` from a caller-owned source dict and asserts the
carried `coordinate_truth` is a different object that rejects item assignment and
deletion (`TypeError`), and that mutating the source dict afterwards leaves the
snapshot and the validated `V_P` membership unchanged;
`test_candidate_rows_stay_stable_after_source_truth_mutation` rewrites the covered
cell's truth out of the palette in the source dict and asserts the later
`candidate_metric_rows` numerator is still `1`/`2` (it would drop to `0` if the live
dict were read); and
`test_conditional_exact_state_stays_stable_after_source_truth_mutation` asserts the
snapshot is unchanged and the conditional exact-state rate stays `1/1` after the
source mutation. Mutation control: storing the caller's mapping directly instead of
the frozen snapshot fails all three regressions (`assert ... is not ...`,
`assert 0 == 1`, and the snapshot-equality assertion); the fix was restored and
re-verified (`68 passed`).

The A2 R7 real-record adapter repair replaced one runner-proxy regression and added
four focused checks (68 -> 73): `test_runner_reads_real_placement_volume_and_promotion_records`
drives the runner through real `ResolvedFacePlacement`, `PlaneVolumeResult` and
`PromotionResult` constructors and asserts the builder receives
`GeometryCellEvidence` coordinate records (never `{coordinate: None}`) while the
`placed`, `volume_cell` and `promoted` rows score 3/3 over `V`;
`test_runner_measures_alignment_from_run_native_geometry_and_aligns_once` forces a
non-identity measured alignment, asserts the search spy consumed the run-native
coordinate set and that the single builder transform yields full `placed` recall;
`test_runner_runs_production_before_evaluator_pins_and_truth` records a
production-call sentinel, a pin-loader sentinel and a truth-loader sentinel and
asserts, for both routes, the order **production < pins < truth**; `test_coordinate_only_scorer_matches_the_accepted_visible_geometry_counts`
proves the coordinate-only G counts equal `evaluation.visible_metrics` for the same
aligned coordinates with an empty `exact` set;
`test_best_visible_alignment_for_coordinates_recovers_a_nonidentity_transform`
proves the coordinate-only search recovers a transform that maps the run
coordinates exactly onto the reference; and
`test_geometry_cell_evidence_and_coordinate_alignment_refuse_invalid_input` covers
the typed refusals (non-frozen/non-integer coordinates, forbidden alignment field).
The end-to-end pin-mismatch regression now stubs the production route so the
fail-closed mask-domain refusal is still exercised after the ordering repair. All
`py_compile` and pytest controls above were run together and pass at `73 passed`.

The adjacent evaluator/frontier suites (89 passed in 0.61s in the prior revision)
were **not rerun** in this revision, because the adapter repair touches only the A2
module, its runner and its own focused module, and the hold instruction is to run
focused tests plus `py_compile` only.

A2's runner/tests are evaluator-only and unreferenced by any product test, so the
prior run's design §5 A1 check-9 regression set (236 passed) was not rerun and no
A1 check was rerun, per the hold instruction to run focused tests only.

Focused coverage includes: module isolation and forbidden-import/identifier scan;
signature scan (no `path`/`manifest`/`coordinate`/`bounds`/`file`/`directory`
parameter on any A2 builder); the five spaces and fixed universes; a section
refusing a cross-space row/transition; camera-diagnostic typing; scorer agreement
with `evaluation.visible_metrics` for all four rotations; scale/reflection/per-cell
alignment refusal; state-property rotation; run-native palette rotation semantics
(`align_palette_states`: translation independence, iteration-order independence, the
zero-turn identity, the forbidden-field refusal and the standalone
`palette_coverage` helper); the primary all-`V` E denominator vs the
conditional in-palette row; candidate availability including a state that loses the
argmin and a beyond-palette state; non-monotone gain and the disjoint
`geometry_lost`/`state_lost` partition; typed unavailable rows that never emit a
zero; `null` over an empty denominator; deterministic digests; Route A and Route B
builder shapes; and a synthetic route-B tail driven through the real public
`PlaneVolumeResult`/`propose_face_states` constructors. The P digest-binding tests
cover: the entry row measured only from a same-matcher digest-equal
`EntryPairEvidence` (recall and precision, and that P starts no transition); the
typed-unavailable entry row with `metric: null` and reason
`entry_match_identities_not_retained_or_digest_unbound` when no digest-bound joint
set is supplied; the checkpoint-reach separation (the reached-but-unscorable entry
row serializes `checkpoint_reached: true` beside `reached: false` and
`metric_value: null`; a frontier that never reaches entry serializes
`checkpoint_reached: false` with reason `entry_unreached`; a non-boolean reach is
refused; and a row with no production checkpoint serializes reach `null`); the
evaluator-control regression that an
`evaluate_scene`/`select_mutual_matches` aggregate (and a same-named source with a
mismatched digest) is never attached to the entry row and that `pair_metric_rows`
raises on an unbound binding; a zero truth denominator yielding `metric_value`'s
typed `null` rather than `0.0`; the unpartitioned `placed -> volume_cell` `G`
transition carrying both geometry checkpoints; the four new follow-up regressions
(an entry-absent frontier plus a valid digest-bound aggregate still emits no
numeric P row - `pair_metric_rows` refuses `checkpoint_reached=False` directly; a
route-level tail carrying both E endpoints partitions geometry/state loss
disjointly on `raw_winner -> promoted` and `promoted -> terminal` while the `G`
step stays unpartitioned; `promoted`/`terminal` reach is `null` when the record is
absent and `true` only from a carried record, never the coarse A1 `output` reach;
and a carried **empty** substage record is reach `true` with `reached: false` and a
null metric, proving presence is explicit `is not None`, not mapping truthiness);
the runner's
`route_b_evaluator_pair_control` fail-closed branches; and
`entry_match_context` reporting counts/digests with
`identities_retained: false`. The six mask-provenance tests cover:
the three tracked `templates-2` specifications reproducing every reviewed pinned
`domain_sha256`; the A2 JSON loader agreeing byte-for-byte with
`benchmark_templates._load_usable_masks`; PNG fallback and `resolve_mask_path`
preference; refusal of malformed, out-of-bounds and wrong-size specifications;
mandatory reviewed pins failing closed on a divergent domain (with the explicit
diagnostic downgrade still recording a caveat, and a pinned run that checks no
pinned domain refusing); and an end-to-end runner refusal that writes no report and
names the mask pin mismatch. The A2 R3 adapter tests add: the pure
`align_candidate_rankings` order/one-transform contract (including a forbidden-field
refusal); a route-level palette+proposals C-context run over a non-identity
translation + Y-quarter-turn with a directional `oak_stairs` truth, asserting
`palette_coverage` 3/3, `candidate_available_k1/k2/k4` = 2/3/3 and full `placed`
geometry recall; and a runner-level regression that captures the builder inputs,
asserts the runner passes run-native maps and proposals unmodified, and asserts one
transform reproduces the reference while the pre-fix double transform scores an
overlap of one. The A2 R4 palette tests add: the `align_palette_states`
translation/order/identity/refusal contract; the standalone `palette_coverage`
helper rotating a run-native directional palette; and the adversarial route-level
run-native palette+placement+proposals regression under quarter-turn 3 plus
translation that asserts C palette 3/4, conditional 3/3, `candidate_available_k1`
3/4 and `raw_winner` exact-state 3/4 with the out-of-palette `GOLD` cell keeping the
full `|V|` denominator. The prior pre-rotated-palette adapter test was corrected to
the run-native palette. The A2 R5 runner tests add two `_route_b_once` regressions
over synthetic production seams (no fixture or route replay): a carried empty
promotion reaches the builder as `{}` and serializes `promoted` with
`checkpoint_reached: true` beside `reached: false` and `metric_value: null` (the
regression fails on the pre-fix `promoted_states or None` runner), and an absent
promotion reaches the builder as `None` and serializes `checkpoint_reached: null`
with the metric still unavailable; both also assert the runner-supplied `terminal`
substage stays `null` because no decoded terminal record is carried.

Retained verification artifacts (hashes below; the runner otherwise used only a
private Route B temporary target). The Route B JSON/stdout pair is **pre-revision
provenance**: it was produced by the corrected run before this digest-binding
correction and its `pair_correspondence` block still shows the superseded measured
evaluator rows. No artifact was regenerated in this revision, because no Route A or
Route B rerun was permitted:

| Artifact | SHA-256 |
| --- | --- |
| `.tmp-a2-route-a-tracked-20261003.json` | `2AC71DF4ECF7931AA57098B9B4D80C58B2459C880CBB8512CEE3C712D94C7F87` |
| `.tmp-a2-route-b-tracked-20261003.json` (pre-revision) | `3C48715D5E60E0866FC38BE29895432AE07BE7AE0D7E57A6D22EA045E2BF127E` |
| `.tmp-a2-route-a-tracked-20261003.stdout.txt` | `80065C363E1CDFFE657DC00E3FD729C756A25DBF5B21C0F638CB2D6A4E96569B` |
| `.tmp-a2-route-b-tracked-20261003.stdout.txt` (pre-revision) | `A8F5D37DB16A0781F8557BCA663F2F858F0F698C371DF6D49D5932F330611EAC` |
| `.tmp-a2-route-a-tracked-20261003/2026-09-02_13.22.14.mcscan` | `E08111EC2ACD91F5CE41B902E7E8D9389F0328541158558E751D45354B9431E6` |
| `.tmp-a2-route-a-tracked-20261003/2026-09-02_13.22.14.litematic` | `BDC8FD54197463FE95F7FBE6C68B18D400BFFF78E4797FBD7E90C673B81390F6` |
| `.tmp-a2-route-a-tracked-20261003/cli-parity/2026-09-02_13.22.14.mcscan` | `E08111EC2ACD91F5CE41B902E7E8D9389F0328541158558E751D45354B9431E6` |
| `.tmp-a2-route-a-tracked-20261003/cli-parity/2026-09-02_13.22.14.litematic` | `DF5C1B816B44FD9E4E71CDFDB1F9E25133E9C96F275934E2A47729E8467239E2` |

The Route A artifact and its temporary project are the retained, unchanged Route A
verification. Its stored `attribution_sha256` (`901408b7...`) is **pre-revision
serialization provenance**: the added `checkpoint_reached` field re-baselines the
canonical row shape, so that digest is not re-derived in this revision and no
replacement digest is claimed (no Route A/B replay is authorized). The measured
Route A numerators/denominators, `V`, alignment, transitions and policy rates are
unchanged serialization-independent numbers. The Route B JSON/stdout pair
above was *not* regenerated in this revision either: it is the pre-revision retained
artifact, kept for provenance, and its P block is superseded by the current
digest-bound contract (the evaluator numbers now appear only under
`evaluator_pair_control`). No duplicate benchmark loop was run and no unrelated or
A1 check was rerun. The retained files are untracked A2 scratch for the lead's
verification and may be removed after review. No unrelated workspace file, fixture,
image or user data was touched; no recursive deletion, reparse/junction operation,
background job, launcher/client automation, Defender/system change, install,
download, model training or commit was performed. No screenshot was displayed or
inspected: frames were opened only for size, content-digest and evaluator
feature/raycast verification, masks only as the evaluator's scored boolean domain.

## 8. Non-claims

* The Route A numbers are single-fixture, single-run **diagnostic** measurements over
  declared denominators against the tracked, pinned mask specifications; they are
  not a pooled accuracy, not a benchmark, and not a claim about any other fixture or
  route.
* Route A's observed alignment and score are reported as measured. No global,
  theoretical or "search-ceiling" claim is made from this bounded single run.
* The Route B entry P row is `available: false` / `metric: null` with reason
  `entry_match_identities_not_retained_or_digest_unbound`. The evaluator
  `0/232`/`0/33` numbers are only the separately named `evaluator_pair_control`
  diagnostic: a different matcher (`evaluate_scene`/`select_mutual_matches`), on the
  evaluator's own usable-pixel mask, never attached to the entry row, never in the
  loss funnel, not a `|V|` ratio, and not a claim about any other fixture. No Route B
  achieved P metric exists on this retained evidence.
* The Route B `entry`/B0 checkpoint is reached (`checkpoint_reached: true`) while
  its P metric is unavailable (`reached: false`, `metric_value: null`). Checkpoint
  reach is never inferred from metric availability and metric availability is never
  inferred from reach; an unreached entry serializes `checkpoint_reached: false` and
  is never scored from a digest-bound aggregate.
* Route B's `promoted` and `terminal` rows state reach only from their own carried
  record; with no dedicated A1 promotion/export checkpoint, a missing substage
  record serializes `checkpoint_reached: null` and the coarse `output` reach is
  never copied onto them. `null` here is unattempted/unknown, not a measured
  `false`; a carried empty record still proves reach `true` beside a null metric.
* `checkpoint_reached` is serialization metadata only. Adding the field
  re-baselines the canonical A2 row/record digest, so no retained
  `attribution_sha256` is claimed to recompute under the current serializer; no
  Route A or Route B rerun was performed and no new canonical digest is claimed.
* Route B produced no visible-truth *cell* number; its cell checkpoints remain a
  typed frontier/status record over the accepted result and persisted envelope. Its
  `entry_sha256` spread across runs is reported as an observation only.
* Gate/status distributions and completion/refusal rates are policy rates, never
  accuracy; the first stop is a frontier indicator, not a causal attribution.
* A `G` transition is unpartitioned and every exact-state `E` transition partitions
  `geometry_lost`/`state_lost` disjointly; the partition is never emitted on a
  geometry step.
* The Route B candidate/alignment adapter is exercised by synthetic run-native
  records only. No Route A or Route B replay was performed, so the retained frozen
  Route B attribution (whose Stage-7 tail is absent and therefore never entered the
  pre-fix C path) is unaffected, and no new `attribution_sha256` is claimed.
* The A2 R7 runner-record adapter repair is a measurement-harness correction: the
  runner reads the real `output.placement`/`output.volume`/`output.promotion`
  records, passes geometry-only placement/volume as coordinate-only
  `GeometryCellEvidence` to the pure coordinate-only scorer, measures the
  alignment from the run's own coordinates, and executes production before opening
  every evaluator file (pin manifest, reference, scene, cameras, masks; order
  production < pins < truth). It changes no production record, schema, formula,
  threshold/gate or the primary `|V|` denominator, and it is exercised by synthetic
  production seams built from the real public constructors only. The focused A2 test
  module does read the tracked mask JSON pin *specifications* under
  `docs/evaluation_masks/templates-2/` (evaluator-spec metadata used for the
  mask-domain parity checks), but no fixture screenshot, reference litematic,
  camera manifest, JAR or truth artifact was opened and no route was replayed. The
  frozen Route B run's Stage-7 tail remains absent, so its
  `placed`/`volume_cell`/`promoted` rows stay typed unavailable; no re-baselined
  `attribution_sha256` is claimed. O ownership and pair correspondence P remain
  unavailable unless their own evidence contracts are bound.
* The A2 R4 palette alignment is an evaluation-only scorer fix: it changes how a
  run-native palette is mapped into the reference frame, not any production
  palette/vocabulary, production behavior or the primary `|V|` denominator, and it
  is exercised by synthetic records only. The retained palette-dependent Route A
  numbers used the default 8-state palette (no directional properties), whose
  rotation is the identity, so they are unchanged; no route was replayed, so no
  re-baselined palette-dependent number is claimed for any run whose palette carries
  directional states under a non-identity quarter-turn.
* The A2 R5 runner promotion-presence repair is an adapter-boundary fix: it changes
  only whether the runner hands the builder `{}` (carried promotion record) or
  `None` (absent promotion record). It does not change the pure builder, production
  behavior, any threshold/gate or any measured number, and it is exercised by
  synthetic runner seams only. The retained frozen Route B run stopped before an
  output record, so it carried no promotion record and its `promoted`/`terminal`
  reach is `null` under both the pre-fix and post-fix runner.
* The A2 R6 palette-ceiling repair is an evaluation-only scorer-contract fix: it
  bounds the C candidate and conditional exact-state numerators by the aligned
  palette membership `V_P` and refuses an out-of-palette ranked candidate instead of
  counting it. It changes no production palette/vocabulary, production behavior,
  threshold/gate or the primary `|V|` denominator, and it is exercised by synthetic
  records only (the frozen Route B run has no Stage-7 tail, so its C path is
  unexercised). No route was replayed, so no re-baselined `attribution_sha256` is
  claimed.
* The A2 R6 validation addendum is a record-invariant fix inside that same
  evaluation-only scorer contract: `CellScore` recomputes `V_P` exactly from the
  visible truth context and refuses palette evidence without it, requires the
  complete context for any palette aggregate and removes the whole-`E` conditional
  fallback, so a forged membership record cannot reach the C/E rows and an
  aggregate-only record cannot report a rate above its `|V_P|` ceiling. It adds no
  production palette/vocabulary, production behavior, threshold/gate, primary `|V|`
  denominator or canonical field change, and it is exercised by synthetic records
  only. No route was replayed, so no re-baselined `attribution_sha256` is claimed.
* The A2 R6 immutable-snapshot addendum is a record-invariant fix inside the same
  evaluation-only scorer contract: `CellScore` stores its truth context as an
  immutable `MappingProxyType` snapshot over a private copy, so post-construction
  mutation through the exposed mapping or the constructor's source mapping cannot
  stale the validated `V_P` membership or the `C`/conditional-`E` rows. It adds no
  production palette/vocabulary, production behavior, threshold/gate, primary `|V|`
  denominator or canonical field change, and it is exercised by synthetic records
  only. No route was replayed, so no re-baselined `attribution_sha256` is claimed.
* No production behavior, schema, signature, gate, threshold, budget, production
  palette/vocabulary or persistence changed; no model was fitted, trained or
  exported.

### 2026-10-04 R7 alignment/no-tail follow-up

Independent review found that the runner passed the `VisibleReference` wrapper to a coordinate helper that expects visible coordinates. It now passes `visible_reference.visible`. Alignment is measured from placement/volume coordinates, falling back to coordinate keys in carried promoted-state/proposal records. When no coordinate-bearing records exist, alignment is inapplicable (`null`); the evaluator preserves the full typed-unavailable frontier ledger rather than refusing the whole run or fabricating identity. Carried evidence without a measurable alignment remains a typed refusal. Existing metric formulas and denominators are unchanged.

The nonidentity runner regression uses the real coordinate helper; promoted/proposal-key fallback tests verify later state-bearing records. The no-tail regression uses terminal stage `geometry` with unavailable output, forbids alignment search, and asserts envelope/result/frontier provenance plus typed-null G, C, P and E rows, including palette and candidate availability. Test prose discloses tracked mask-spec JSON reads and clarifies that adapters use synthetic records built with real constructors, not a frozen production tail.

Checks: focused A2 module **76 passed**; `py_compile` of runner, pure scorer and focused tests passed. No route replay or fixture-media/reference/camera reads occurred for this evaluator repair. Astra accepted the implementation and expanded assertions; Luna verified the test count, claims and hashes after the historical snapshot was marked superseded. Current code hashes:

| File | SHA-256 |
| --- | --- |
| `scripts/evaluation/report_image_only_attribution.py` | `EE976B6B9E79A8535A412F9430FDF5E0CA8D1C16CFFEA3DF342E130EECF22805` |
| `src/mc_photogrammetry/evaluation_image_only_attribution.py` | `4FED6AA8E22DD881E69063A33B56DF2480B1700BFE62CC7D86D69EE54A8FE2A3` |
| `tests/test_evaluation_image_only_attribution.py` | `3B94D058DF7A896550C5265D4B037700EA310B6DAF21048609ED620CA0611CC5` |
