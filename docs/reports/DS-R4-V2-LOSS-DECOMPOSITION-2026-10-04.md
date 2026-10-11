# DS-R4-V2-LOSS-DECOMPOSITION - R4 complete-support capture: pipeline completion vs reconstruction quality

- Task: `AGENT_COMMS.md` 2026-10-04 LEAD -> DeepSeek `DS-R4-V2-LOSS-DECOMPOSITION`.
- Owner: DeepSeek CLI worker (direct shell).
- Mode: **read-only**, source-grounded analysis of the accepted R4 record plus the source
  needed to interpret it.
- Not done: no PNG opened, no truth/reference/evaluator/litematic/manifest/camera/F3/
  direction/mask/coordinate read, no pipeline/runner/inference/capture/test/evaluator or
  Minecraft execution, no source edit, no tuning, no retry, no new run or capture.
- The row counts in this report were computed by one read-only standard-library `json`
  parse of the persisted `record.json` (no project module imported, no runner or pipeline
  executed). All percentages are exact quotients printed to 4 decimal places.

## 0. Inputs read and pins used

Record and artifact (both accepted by the R4 report, both re-hashed read-only here):

| artifact | bytes | SHA-256 |
|---|---|---|
| `.tmp-routeb-v2-frozen-r4-20261004/record.json` | 16,896,513 | `006E8B6B16A312CDAA61A17B4336CE27C1ABAC590435645908EA9CFEF2E998EE` |
| `.tmp-routeb-v2-frozen-r4-20261004/route_b.mcscan` | 9,319 | `EE7C56BB5BD922D42BD2AD782E0FDFF202AD9DBE61CD331B423087C5A6407057` |

Accepted report this analysis builds on:
`docs/reports/DS-R4-V2-ROUTEB-FROZEN-IMAGE-R4-2026-10-04.md`
(12,732 B, SHA-256 `1419E550BB08442803FF8E3A62FEAE658FE91B1CB2AA84C51CCBC3EE224DD338`).

In-record pins used below: `schema mc-photogrammetry-route-b-diagnostic-record-v1`,
`status complete`, `expected_row_count 1928`, `signature_witness_count 85`,
`whole_universe_identity_sha256 a5ddc8a65c0414e99b6026b5c2d7f4770e942f495ac9ebd8f26e59368d9c0efc`,
`input_match_identity_sha256 28ad4ac48021c727c4b6e7bf283e9965514dc3016c3c16919a18367cf0868105`,
`failed_assignment_sha256 ab4cb09bf5e8fe00e0bca11ca6cf2732cf24fd086d6f717919369f4191525dcd`,
in-record `record_sha256 6eddf7f2bb60c1bdeac053d86e592dcd3c5c1b4bdd91a056c960c6f8db8ba797`,
`artifact_sha256 678475d64a95a2814b5cce6864cd55853cc8f8b5e45cdf8d1b86a7023d5d41af` (all match the
accepted R4 report Section 5).

Source read (only what interprets the record): `scripts/evaluation/run_route_b_diagnostic.py`;
`src/mc_photogrammetry/pipeline/candidate_support_provenance_v2.py`;
`.../candidate_support_provenance.py`; `.../image_derived_graph.py`;
`.../image_derived_graph_attribution.py`; `.../image_derived_entry.py`;
`.../image_derived_signatures.py`; `src/mc_photogrammetry/reconstruction/plane_graph.py`;
`.../lattice_planes.py`; `.../lattice.py`; `src/mc_photogrammetry/domain/models.py`.
Prior text read for context only (no artifact executed): `DS-STAGEWISE-LOSS-TEST-PLAN-2026-10-04.md`,
`visible-state-candidate-depth-2026-10-03.md`.

## 1. Headline: `complete` is a pipeline-completion result, not a reconstruction result

`status complete` is the **support-provenance capture** completing. It is not a geometry or
accuracy result, and the source shows it is affirmative evidence of a solver **abstention**:

- The runner attaches the V2 collector through `complete_support_sink` and requires the sink
  to fire exactly once (`run_route_b_diagnostic.py:480-499`).
- In the graph seam, `complete_support_sink` is called only inside the `if solution.reason:`
  block, after the single `solve_component` call, and the function then returns
  `_unavailable(..., solution.reason, "component solver abstained", ...)`
  (`image_derived_graph.py:5082`, `:5161-5201`). On an assembled solution the sink is never
  called and the function returns `"assembled"` (`:5208-5213`).
- Therefore the published `complete` record, whose row count and identity gauntlet the runner
  enforces, is only producible when the component solver returned a typed abstention.

So R4 proves: entry, lattice, signatures, matcher, same-axis filter and the complete-support
capture all ran to completion; and the component solver did **not** produce a solution. It
does not prove anything about reconstruction quality.

Consistently, `route_b.mcscan` carries no geometry: the entry persists a session-only
`ScanProject` (`image_derived_entry.py:665-694`), whose `volume` defaults to an empty
`BlockVolume` (`domain/models.py:110`, `:185`), and the entry terminal is deliberately
`GEOMETRY_UNAVAILABLE` (`image_derived_entry.py:11`, `:543`). That is why `volume.blocks` is
empty - it is a designed non-claim of the entry seam (docstring `:49-57`), not a measurement
of a failed reconstruction.

## 2. Exact denominators from the persisted 1,928 rows

All 1,928 admitted rows are in `capture.index_pages` (`5` pages, row counts
`469, 467, 465, 466, 61`), `admitted_row_index` contiguous `0..1927`. Disposition,
`support_state` and `support_refusal_reason` are full cross-tabulated below.

| disposition | support_state | support_refusal_reason | rows | % of 1928 |
|---|---|---|---|---|
| `kept` | `present` | (null) | 26 | 1.3485% |
| `kept` | `refused` | `support_payload_over_cap` | 11 | 0.5705% |
| `kept` | `refused` | `source_tap_witness_unavailable` | 1 | 0.0519% |
| `same_axis_mismatch` | `present` | (null) | 945 | 49.0145% |
| `same_axis_mismatch` | `refused` | `support_payload_over_cap` | 833 | 43.2054% |
| `same_axis_mismatch` | `refused` | `source_tap_witness_unavailable` | 112 | 5.8091% |
| **total** | | | **1928** | **100%** |

Marginals:

| marginal | count | % of 1928 |
|---|---|---|
| disposition `kept` | 38 | 1.9710% |
| disposition `same_axis_mismatch` (dropped) | 1890 | 98.0290% |
| support `present` | 971 | 50.3631% |
| support `refused` | 957 | 49.6369% |
| refusal `support_payload_over_cap` | 844 | 43.7759% |
| refusal `source_tap_witness_unavailable` | 113 | 5.8610% |

Conditional views (same numbers, different denominator; still not accuracy):

- Of the 38 kept rows: 26 present (68.4211%), 12 refused (31.5789%).
- Of the 1890 dropped rows: 945 present (50.0000%), 945 refused (50.0000%).
- Of the 957 refused rows: 844 over-cap (88.1923%), 113 witness-unavailable (11.8077%).
- Of the 844 over-cap rows: 833 dropped (98.6967%), 11 kept (1.3033%).
- Of the 113 witness-unavailable rows: 112 dropped (99.1150%), 1 kept (0.8850%).
- No row carries `matcher_witness_unavailable`, `support_payload_unavailable`,
  `witness_digest_mismatch`, `payload_digest_mismatch` or
  `source_tap_witness_shape_mismatch`; every admitted row's matcher witness was present and
  digest-consistent (`_resolve_row_support`, `candidate_support_provenance_v2.py:1931-1964`).

Other persisted denominators:

- `support_payload_sha256`: 971 non-null (exactly the present rows), 957 null.
- `match_index`: 38 non-null (exactly the kept rows); every kept row has one and no dropped
  row claims one (`candidate_support_provenance_v2.py:641-645`).
- sidecar: `67` pages, `971` entries, one entry per present row in admitted order; page sizes
  range `8..28` (mean 14.49); page `first_admitted_row_index` spans `0..1887` with real gaps
  (page 63 starts `1803`, page 64 starts `1839`), the accepted sparse-page behaviour.
- digest pages: `6` sections x `1` page each; every section holds `1928` digests, `0` nulls.
- root: `row_count 1928`, `declared_universe_cap 1928`, `universe_max_rows 1928` - the cap
  equals the exact frozen count, so any 1929th admitted row would be `row_cap_exceeded`
  (`:2039-2040`); this is a frozen-profile cap, not a quality signal.
- groups: `35` distinct `(left,right,left_axis,quadrant_of_right)` groups, `group_index 0..34`,
  group sizes `1..130`; kept rows appear in all 35 groups (max 2 per group).
- signature witnesses: `85` entries = `66` bound (`outcome: bound`) + `19` refused
  (`outcome: refused`). All 19 refusals are `source_tap_witness_budget_exhausted` with
  details `aggregate_rectified_samples` 15, `aggregate_cell_memberships` 3,
  `aggregate_payload_bytes` 1. The 66 bound bind `66` distinct signature digests
  (`image_0000:25`, `image_0001:20`, `image_0002:21`); no bound witness carries a null
  `signature_sha256`.

**None of these is a block-accuracy score.** `38/1928 = 1.9710%` is the same-axis kept share;
`971/1928 = 50.3631%` is the support-payload captured share. Without truth, neither can be
read as correct/total exact-match, visible-state recall, or geometry accuracy, and this report
does not do so.

## 3. What the four named terms mean, per source

### 3.1 `kept` and `same_axis_mismatch` (disposition, a production filter verdict)

`plane_graph.SAME_AXIS_DISPOSITION_KEPT = "kept"` and
`SAME_AXIS_DISPOSITION_DROPPED = "same_axis_mismatch"` (`plane_graph.py:72-76`).
`SameAxisCandidateObservation` holds the matcher's complete admitted set with one parallel
verdict per candidate: `kept` means the candidate survived `_apply_same_axis_filter` into the
solver input, `same_axis_mismatch` means the equality filter dropped it because another
same-axis plane pair implied a different displacement (`:298-315`). The disposition is derived
by object identity against the filtered list (`:1270-1273`) and is published per candidate in
`PreFilterMatcherWitnessRowV1` with a `post_match_index` for kept rows and `None` for dropped
(`:478-502`, `:553-565`). The V2 index row stores that same disposition (`v2:613-645`).
This is a **production gate outcome**: 1890 dropped is the same-axis filter removing
conflicting candidates, not a comparison to any truth.

### 3.2 `support_payload_over_cap` (row-local support refusal)

One of the bounded `SUPPORT_REFUSAL_REASONS` (`v2:174-183`). `_resolve_row_support` emits this
one code from **four distinct cap-producing branches** (`v2:1931-1964`); they are not four
consecutive steps of a fixed gate list, because the witness lookup and the witness shape check
sit between the third and the fourth:

- `_support_payload(row)` raising `SUPPORT_PAYLOAD_OVER_CAP` (`v2:1941-1946`; reason defined at
  `v1:91`, raised at `v1:647`, `v2:759`);
- `_payload_over_declared_bounds(payload)` (shape/element bounds, `v2:309-324`, called at
  `v2:1952-1953`);
- `payload.payload_bytes() > plan.sidecar_entry_max_raw_bytes` (`v2:1954-1955`);
- the serialized `MatcherSupportSidecarEntryV2` exceeding `plan.sidecar_entry_max_bytes`,
  checked **after** the missing-witness return (`v2:1956-1957`) and the witness
  shape-mismatch return (`v2:1958-1959`) (`v2:1960-1963`).

So only the first three cap branches precede the witness lookup; the witness shape check
follows, and the serialized-entry byte cap is last.

`payload_bytes()` is the summed `nbytes` of `logical_pairs` (`(overlap_cells, 4)` `int64`),
`warp_source_rows` and `warp_source_columns` (`v1:308-310`). The R4 plan is
`sidecar_entry_max_raw_bytes = 131,072`, `sidecar_entry_max_bytes = 233,472`
(`v2:1877-1885`). The record persists only the single bounded code, not which branch fired;
that is a real limitation (Section 4). This refusal is about the size of the already-minted
matcher-support arrays, not about geometry correctness.

### 3.3 `source_tap_witness_unavailable` (row-local support refusal)

Returned when either endpoint's signature digest has no bound
`SourceTapWitnessBindingV1` in the signature-witness map (`v2:1956-1957`;
lookup at `:2066-2067`). The map is built from bound sink items only (`:2041-2044`). A binding
is emitted per accepted reading alternative by the ordinary signature pass
(`image_derived_signatures.py:428-460`); a binding is refused when the sampler's own witness was
missing, malformed, or over its planned-work budget, in which case the sink receives a typed
`SourceTapWitnessRefusalV1` with no `signature_sha256` (`:482-517`). The budget refusal comes
from `lattice.WitnessBudgetV1` (`lattice.py:41-49`, `:60-79`) evaluated inside the ordinary
`plane_signature` call (`lattice_planes.py:1747-1772`: `aggregate_rectified_samples`,
`aggregate_bilinear_taps`, `aggregate_cell_memberships`, `aggregate_payload_bytes`).
This is a **diagnostic-evidence availability** verdict, not a reconstruction-quality verdict.

### 3.4 `present` / `refused` (`support_state`)

`SUPPORT_STATE_PRESENT = "present"` means the row's support payload was captured and bound to a
sidecar entry (non-null `support_payload_sha256`); `SUPPORT_STATE_REFUSED = "refused"` means
the row kept its disposition but no payload was persisted, with exactly one bounded reason
(`v2:171-172`, `:650-656`).

## 4. Gate-order interaction: 844 over-cap and 113 witness-unavailable are not independent

Reading the two refusal codes as independent failure counts would be wrong, because the first
three cap branches run **before** the witness-availability check (`v2:1940-1955` precede the
missing-witness return at `v2:1956-1957`; the fourth, serialized-entry cap is after it).
Combining the persisted per-row endpoint signature digests with the 66 bound witness digests:

- The 1,928 rows reference `78` distinct signature digests across their two endpoints.
- `59` of those 78 have a bound witness; `19` do not.
- `7` of the 66 bound witnesses bind signatures that no admitted row references.
- `525` rows (`27.2303%` of 1928) have at least one endpoint whose signature has no bound
  witness.
- Of those 525, `412` are also over-cap and are therefore reported as
  `support_payload_over_cap`; only the remaining `113` reach the witness gate and are reported
  as `source_tap_witness_unavailable` (`412/525 = 78.4762%`;
  `113/525 = 21.5238%`). `113 + 412 = 525` closes exactly. Because only the first three cap
  branches precede the witness lookup, each of these 412 rows necessarily hit an early
  (pre-witness) cap; the exact early sub-gate is not recoverable from the record.

Sub-gate evidence for the 844 over-cap rows:

- `102` of them are **provably** over the raw cap from the `logical_pairs` array alone:
  `32 * overlap_cells > 131072`, i.e. `overlap_cells > 4096` (`12.0853%` of 844, `5.2905%` of
  1928). This is the third (raw-byte) branch above.
- The remaining `742` cannot be sub-gated from the persisted record: the arrays are not
  persisted for refused rows, so it is not recoverable whether they hit the declared element
  bounds, the raw-byte cap through the warp arrays, or the serialized-entry cap. The
  serialized-entry branch is reachable only for a row whose two endpoint signatures both have a
  bound witness; a refused row with an unbound endpoint can only have been produced by one of
  the first three branches.
- The largest **present** payload is `131,040` bytes against the `131,072` raw cap (present
  payloads: min 1,312, median 22,640, mean 37,702, 39 present entries above 120,000 bytes).
  The cap is therefore binding at the margin, which is consistent with - but does not prove -
  the raw cap being the dominant sub-gate. Regression of `overlap_cells` on refusal is stark
  (present max 1,851; refused median 343, max 27,948), but `overlap_cells` is a matcher support
  size, not a truth quantity.

## 5. Persistence audit: what the record does and does not carry

`record.json` top-level keys: `artifact_sha256, capture, expected_row_count,
failed_assignment_sha256, ingress, input_match_identity_sha256, inputs, note, record_sha256,
schema, settings, signature_witness_count, signature_witnesses, stage_digests, status,
whole_universe_identity_sha256`. The capture root carries page bindings, digest sections,
ingress pins, the cap plan and the identity digests.

- **Not persisted:** graph solver `status`, solver `reason`/`detail`, `first_failing_substage`,
  `pair_attribution`, `selected_quadrants`, `selected_scale`, `axis_diagnostics`,
  `solver_failure`, `candidate_counts`, and any placement/volume/block/coordinate geometry.
  `ImageDerivedGraphAttribution.to_record()` builds exactly those fields
  (`image_derived_graph_attribution.py:402-431`) and the runner builds the attribution object
  in memory, but `_stage_digests` persists only six opaque SHA-256 values
  (`run_route_b_diagnostic.py:648-657`): `entry_sha256`, `session_sha256`, `lattice_sha256`,
  `signatures_sha256`, `graph_sha256` (= graph result `result_sha256`), `trace_sha256`
  (= attribution `attribution_sha256`). The CLI prints only status/record/row_count/file hash,
  so the in-memory trace is dropped (`:1104-1116`).
- **Persisted:** support evidence only - 1,928 index rows, their per-row disposition and
  support outcome, the sidecar payloads for present rows, the six digest sections, ordered
  ingress pins, and the identity digests. `failed_assignment_sha256` binds the failed best
  assignment, but it is a digest, not a readable reason.

Source-derived substage (labelled as derived, since it is not persisted): the capture proves
`post_admission_candidates = 1928 > 0` and `post_same_axis_candidates = 38 > 0`
(`plane_graph.py:1249`, `:1277`). The attribution's stage selector then yields
`first_failing_substage = "solver"`, not admission and not same-axis filter
(`image_derived_graph_attribution.py:480-482`). The exact `solution.reason` (for example the
`plane_integer_residual` recorded by the earlier Route B run cited in
`DS-STAGEWISE-LOSS-TEST-PLAN-2026-10-04.md` Section 3.1 rank 3) is **not** recoverable from
R4's own record; asserting it for R4 would be an unsupported transfer.

## 6. First actual reconstruction checkpoint still missing/unverified

The R4 chain reached: entry -> lattice -> signatures -> matcher -> same-axis filter ->
graph/capture. The first reconstruction checkpoint **missing** is the next one: a resolved
plane-graph component solution and its placement/scale/volume. The graph result on this run is
the typed `_unavailable` return (`image_derived_graph.py:5199-5201`), and the entry terminal is
`GEOMETRY_UNAVAILABLE`; no `PlacementUniverse`, aligned region support, resolved face
placement, plane volume or block exists (`image_derived_entry.py:17-23`, `:49-57`). The first
checkpoint **unverified** is one step earlier: the solver abstention itself is bound only by
`trace_sha256`, so the reason and first-failed predicate are not persisted.

Derived context, not an accuracy claim: only `38` of `1928` admitted candidates survived the
same-axis filter into the solver, and that solver abstained. Whether `38` is too few, and
which predicate failed, requires the attribution record (Section 7) and then truth.

## 7. Prioritized next experiment (recommendation only; nothing was run)

### 7.1 Priority 1 - make the existing Route B stop attributable (evaluation-only, no truth)

The runner already holds the answer in memory and throws it away. Recommended bounded
experiment, in the spirit of the accepted `DS-STAGEWISE-G1` plan
(`DS-STAGEWISE-LOSS-TEST-PLAN-2026-10-04.md` Section 4.3):

- Run the frozen three-image Route B chain once with `complete_support_sink` **disabled** and an
  evaluation-only sidecar that serializes the already-built graph attribution record
  (`reason`, `first_failing_substage`, `candidate_counts`, `pairs`, `selected_quadrants`,
  `selected_scale`, `axis_diagnostics`, `solver_failure`) plus the graph result status.
- Assert output-neutrality: `graph_sha256`/`trace_sha256` and every existing artifact digest
  are byte-identical to R4 when only digest capture is compared; no gate, threshold or
  production signature changes.
- This converts "the solver abstained, reason unknown" into a named first-failed predicate and
  per-axis resultants/residuals. It needs no truth, so it is safe to run before any truth arm.

### 7.2 Priority 2 - the stage-by-stage geometry/state loss ladder against existing fixture truth

Once 7.1 names the stop, run one evaluation-only ladder on the same frozen, **image-derived**
production inputs against the existing frozen Template 2 fixture truth, with universes and
denominators kept separate (the existing `evaluation_image_only_attribution` metric spaces and
`evaluation_image_only_frontier` checkpoints):

1. entry/session state (frames, ingress, `scale_known`);
2. lattice readings and signature availability;
3. matcher admission (`pre`/`post`) and the same-axis kept set, versus fixture correspondence;
4. solver outcome and, if assembled, the selected quadrants/scale versus fixture geometry;
5. placement/volume/ownership and, only later, palette/candidate/exact-state loss.

Hard boundaries: manifest/camera/F3/manual coordinates stay **advanced-only and
evaluation-only**; the production call must return and be serialized/hashed **before** any
truth/camera/mask/reference is opened (production-before-truth ordering already accepted for
the image-only attribution runners); reference-assisted rows must be labelled and never pooled
with image-only rows. No gate, threshold or production path changes.

### 7.3 Why this order

R4 measured no geometry at all, so any state-loss number taken now would be undefined, not
small. The cheapest way to make progress measurable is to attribute the existing stop, then
join stage states to truth. This is also the plan's stated image-only priority: the largest
measured image-only end-to-end loss is Route A visible geometry (`16/398`), and Route B's
`plane_integer_residual` stop is a **frontier**, not a magnitude
(`DS-STAGEWISE-LOSS-TEST-PLAN-2026-10-04.md` Sections 3.1, 4.1).

## 8. Narrow assessment of the proposed local texture model

**What measured failure it could address (and only this).** A local texture/material model is
an auxiliary proposer/scorer for the per-patch candidate stage. The failure it could target is
measured per-patch candidate recall/ranking once production patches exist: the existing-scorer
per-face top-1 is T2 `363/958 = 37.89%`, T6 `522/3,813 = 13.69%`, with top-8 `76.93%` /
`26.49%` (`DS-STAGEWISE-LOSS-TEST-PLAN-2026-10-04.md` Section 3.3, Section 6.2). Those are
**supplied-camera, reference-assisted** diagnostics, and they are a baseline a model must beat
on identical patches, not an image-only R4 number.

**What it cannot address.** It cannot repair camera/scale (K), visible geometry (G), pixel/face
ownership (O), pair correspondence (PC) or placement. It cannot recover a cell whose truth
state the pipeline never generated, and it cannot fix a geometry frontier
(`DS-STAGEWISE-LOSS-TEST-PLAN-2026-10-04.md` Sections 5, 6.2; prior routing in
`AGENT_COMMS.md` likewise: "Do not train a texture model yet" and "A texture model cannot
repair K/G/PC"). For R4 specifically, the run stopped at the solver before any candidate/state
output, so C/E appearance loss on this path is **unmeasured**, and no texture-model value can
be inferred from R4.

**Minimum vanilla-only controlled labels (necessary, not sufficient).** Per the accepted plan
(`DS-STAGEWISE-LOSS-TEST-PLAN-2026-10-04.md` Section 6.1): at least 6 admitted independent
builds (recommended 12); at least 10 exact states, each with at least 50 admitted examples in
at least 3 independent builds at `valid_fraction >= 0.75`; held-out OOV at or below 50% and
abstained on; at least 3 captures/state, 1,000 examples/build, 3 sessions/build, 16 finalized
HUD-hidden frames/build, 10 shared material families; world-grouped train/dev/calibration
folds plus an immutable world-held-out test set; quantified label purity; byte-identical
production-patch parity; and an approved training-data contract/exporter. Labels must stay
vanilla-only (the vanilla palette/states) and build-disjoint, with geometric state properties
kept as deterministic decisions.

**Is it justified before geometry attribution?** No. Current corpus state is far from the
entry criteria: three topology groups / three source builds over five datasets and 26
screenshots, every index split `train` (13/67/34 vs `TRAIN_CUTOFF = 70`), E3's 8,516 patches
over two usable source groups, E5's 63 exact states with only 4 shared and LOBO OOV
`55.76%` / `78.74%`, and no approved training-data contract
(`DS-STAGEWISE-LOSS-TEST-PLAN-2026-10-04.md` Sections 3.3, 6.1, 6.3). The narrow position is:
keep the model as a planned auxiliary, defer any training/integration until after the Route B
geometry attribution and the data gates, and then compare it to the same-patch `PaletteIndex`
baseline on held-out builds. No new feature or training-data collection is recommended now.

## 9. Limitations and non-claims

- **No accuracy / exact-match / visible-state / geometry / promotion claim.** No truth,
  reference, evaluator, litematic, scene, camera, F3, direction, mask or coordinate input was
  read. `38/1928` and `971/1928` are diagnostic shares, not accuracy scores, and no exact
  visible loss is inferred.
- **No runtime trace.** The per-stage ordering and the derived `first_failing_substage` are
  source-derived, not observed, for this run; the graph attribution record was never persisted.
- **Sub-gate opacity.** `support_payload_over_cap` aggregates four cap-producing branches (three
  before the witness lookup, the serialized-entry cap after it); only 102 of 844 over-cap rows
  are provably over the raw cap from persisted fields, and the other 742 cannot be sub-gated
  from the record.
- **Single run, single frozen input set.** No determinism, generalisation, or R3-class-absence
  claim; the universe cap equals the exact row count, so the profile is frozen to 1,928.
- **Witness counts are sampler-budget diagnostics.** The 19 refused witness bindings report
  the sampler's planned-work budget (`lattice.WitnessBudgetV1`), not a reconstruction verdict.
- **The proposed experiments were not run.** Section 7 is a recommendation for separate
  authorization; no pipeline, runner, test, evaluator or capture was executed.

## 10. Direct-shell compliance

Per `docs/deepseek-worker.md` and `docs/deepseek-direct-shell.patch.yml`: no junction/reparse
creation or manipulation, no recursive deletion, no encoded/obfuscated commands, no generated
executables or installers, no background or persistent processes, no Defender or system
changes, no external temp inspection, no client automation, and no image/truth/launcher
automation. Only this report and a concise `AGENT_COMMS.md` result were written; no production
source, test, runner, pipeline or artifact was edited.
