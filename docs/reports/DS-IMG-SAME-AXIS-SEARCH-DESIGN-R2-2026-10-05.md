# DS-IMG-SAME-AXIS-SEARCH-DESIGN-R2 — read-only audit and one bounded evaluation-only joint-alternative experiment design

- **Task:** `AGENT_COMMS.md` 2026-10-05 01:32 UTC, Lead -> DeepSeek `DS-IMG-SAME-AXIS-SEARCH-DESIGN-R2`.
- **Owner:** DeepSeek CLI worker (read-only audit; `docs/deepseek-worker.md` read-only discipline).
- **Mode:** read-only, source-grounded. **Nothing was run, edited, captured or solved.** No truth /
  reference / camera / mask / image-payload / evaluator / JAR / Minecraft access. No test, solver,
  pipeline, harness or runner execution. No threshold, filter or gate change. No production source
  or test edit. **Only this report file was written.** No `AGENT_COMMS.md` result was composed or
  written, and no other file in the repository or agent state was created or modified.
- **Purpose:** prepare the next geometry experiment. Trace how plane candidates form, how
  `_apply_same_axis_filter` disposes of them, what `solve_component` actually searches and gates,
  and where a solved component would become placement geometry. Explain why the accepted
  solver-consistent counterfactuals are not geometry proof. Name the identity/footprint/coordinate
  data the sealed artifacts are missing. Propose **one** bounded, evaluation-only, production-path
  experiment that can measure a *usable image-derived geometry output*, not solver acceptance.

Evidence quoted here is taken from already-accepted reports/artifacts and from source structure.
Per the reading rules of `DS-STAGEWISE-LOSS-TEST-PLAN-2026-10-04.md` §0, an undefined denominator
is typed unavailable, never `0`; no ratio is written without its universe.

---

## 0. R2 review corrections folded into this revision

This revision resolves seven findings against the prior R2 draft. It remains report-only.

| # | Correction | Where |
|---|---|---|
| 1 | Canonical geometry orchestration **recomputes** the graph, and downstream stage J **replays the matcher and requires exact `graph_result.matches`**; an observer therefore cannot continue altered `JointCandidate` edges. Specified a separate **eval-only arm validator** that preserves the canonical baseline G/J, binds the complete ordered **pre-filter** admitted pool to the session/lattice/signature digests, requires synthetic parity, and refuses on tamper/missing edge — otherwise the run **stops at the exact seam** (further corrected by addendum §0.1, A1-A3). | §2.5, §5.6, §5.9 |
| 2 | `PlacementUniverse` is **not** coordinates or output. Named `resolve_face_placement` and `build_plane_volume` with exact inputs/budgets/refusals and coordinate/digest extraction; the assembler's reach is **not** a score. | §4.3, §5.5, §5.8 |
| 3 | The selector returns a **Pareto `JointCandidate`** and does **not** accept `quadrant_constraints` or `ambiguous_margin = 0.02`; independent verification must re-run the **unchanged `solve_component`** under frozen constraints and `.90/.15/.02`; never choose silently among `b_unresolved` / `ambiguous_quadrant` frontier. Distinguished **frontier** from **all labels**. | §2.4, §5.4, §5.5 |
| 4 | Ranked by **`JointCandidate.cost`** (not `alternatives[0].cost`); the production solver **minimizes** cost, so no descending ranking. The 40-certificate / 2,520-row selector caps are bounded with the verifier, downstream, byte and time limits as one atomic budget and are **not** treated as justification for any fixed arm count (addendum A9). | §5.4 |
| 5 | Baseline is **geometry-unavailable**: no gained/lost/net and **no improvement claim**. If later authorized, seal the ordinary baseline plus **all** counterfactuals before truth and report **first scoreability** plus **absolute per-arm** placed/volume recall, precision, F1. **No exact-state score.** | §5.7, §5.8 |
| 6 | The R11 accepted substitutions may **not be globally coherent under one `b` per reading**; a coherence control is required, and a negative search result does **not** prove an evidence-quality bottleneck. | §3.5, §5.4, §5.10 |
| 7 | Removed the false claim that the worker wrote an `AGENT_COMMS.md` result; **only this report was written**. | header, §8 |

The no-run / no-truth / no-images constraints and the advanced-only, evaluation-only status of
supplied data are retained unchanged.

### 0.1 Latest review addendum applied in this revision

This addendum supersedes any conflicting wording in §4.3, §5.4, §5.5 and §5.6. It keeps the report
a **conditional design**, authorizes nothing, and requires no image run.

| # | Addendum correction | Where |
|---|---|---|
| A1 | Preserve the **complete ordered pre-filter admitted `PlaneMatch` pool**. The public `graph.matches` is empty on the frozen run and, when present, is the post-`_apply_same_axis_filter` set; a membership rule stated against `graph.matches` therefore blocks exactly the dropped R11 arms the design must reach. The arm universe is the pre-filter admitted pool (R10 `1,928`), not the survivors. | §5.4, §5.5, §5.6 |
| A2 | The baseline canonical **G/J stays unchanged**. A **separate eval-only arm validator** binds the complete ordered pre-filter pool to the session/lattice/signature digests; it does **not** forge a production graph and does **not** weaken J (the canonical matcher replay / exact-match requirement is untouched). | §5.6 |
| A3 | If downstream continuation cannot be **source-grounded** from those digests, the run **stops at the exact seam**. The next separate gate is a **synthetic proof** (no images): (a) a genuinely dropped arm reaches placement + volume, (b) foreign-edge refusal, (c) baseline parity. No image run is authorized. | §5.6, §5.10, §5.11 |
| A4 | `quadrant_constraints` are **frozen from the baseline** solve and are never derived from each candidate; an incompatible label is rejected. | §5.4, §5.6 |
| A5 | Per arm, run the **unchanged `solve_component`** under `.90/.15/.02`, compare its **returned assignment** to the arm label, and explicitly verify **one `b` per reading**. | §5.4, §5.6 |
| A6 | `PlacementUniverse` is **not geometry**: no `universe_sha256`. Use `placement_sha256` / `volume_sha256`, and extract coordinates via `normal_block`, the tangent world axes and `lateral_u` / `lateral_v`. | §4.3, §5.5 |
| A7 | Evaluate bounded **Pareto arms separately in stable identity order**, with **no production nomination**; otherwise type-refuse. | §5.4, §5.11 |
| A8 | Removed the claim that negative score deltas force a score-first rule to discard the entire accepted set. | §5.4, §5.10 |
| A9 | Bound **certificate / row / verifier / downstream / bytes / time** coherently as one atomic budget, and refuse overflow atomically. The M4-G1C `40`-certificate / `2,520`-row values do **not** justify a 64-arm target. | §5.4, §5.10 |
| A10 | Baseline G is **unavailable**: no delta, **absolute per-arm geometry only**, no exact-state result. | §5.9 |
| A11 | The report remains a **conditional design, not an authorized run**. | §5.1, §7 |

---

## 1. Current evidence state (quoted, not re-derived)

| Fact | Source |
|---|---|
| Accepted geometry observation: solver first-failed predicate `minimum_resultant`; all three per-axis resultants `< 0.90` | task text; `AGENT_COMMS.md` 5225-5226, 5754, 5853 |
| Concrete projected values on the frozen image-only three-view run: resultants `0.789357 / 0.013942 / 0.876999`, residuals `0.093065 / 0.247102 / 0.067258`, observations `[8, 6, 6]`, unmatched-axis coverage complete | `AGENT_COMMS.md` 5325, 5754, 5853 |
| Candidate funnel on that same run: pre-admission `9,592` -> admitted `1,928` -> post-same-axis `38` | `AGENT_COMMS.md` 5853; handoff line 407 |
| R10 all-axis prefilter capture: complete pre-filter admitted universe `1,928`, of which the same-axis artifact covers the solver-active subset `448` rows (168 + 240 + 40 axis counts), `10` kept / `438` dropped | handoff 392, 407; `AGENT_COMMS.md` 6681, 6687 |
| R11 exact solver-context capture: 38 inputs, 15 readings, 10 active edges, assignment `[0, 3, 1]` | `AGENT_COMMS.md` 5853 |
| 6,068-arm scan: unchanged solver accepted **25 / 6,068** (0.412%); all accepted arms have **negative** per-axis correlation-score deltas against the survivors they replace | `docs/reports/DS-IMG-R11-AXES02-PAIR-SCAN-2026-10-04.md` §1, §6 |
| Measured-best accepted arm (`A020-308`): resultants `0.998985 / 0.989176 / 1.0`, residuals `0.005399 / 0.020419 / ~0`, assignment `[0, 3, 1]`, `duplicate_b false` | same report §1, §8 |
| Frozen ordinary-image Route B run: `geometry_unavailable` / `plane_integer_residual`, `placement_cells = 0`, `volume_cells = 0`, `promoted_cells = 0`, no output digest | handoff 394, 411; stage plan §3.1 rank 3 |
| Truth/geometry accuracy of any of the 25 accepted arms: **not measured** — by design, the scan was diagnostic-only | pair-scan report §11; handoff 397 |

Two consequences frame everything below.

1. **The solver gate is a consistency predicate, not an identity predicate.** Passing it says a set
   of edges implies mutually consistent integer face offsets; it says nothing about whether those
   edges name the same visible Minecraft plane.
2. **No arm in the existing evidence is attached to a scoreable geometry output.** Every accepted
   arm stopped at solver acceptance; the frozen production run stopped one stage earlier. So the
   quantity the Lead actually needs — *does a joint-alternative search improve usable geometry* — is
   currently **undefined**, not small. Any design must therefore make the output reachable and
   scorable, not merely the gate green.

---

## 2. Candidate formation (source trace, stage C -> M)

### 2.1 From sampled pixels to one `PlaneReading` per frame-axis-side

- `lattice_planes.frame_plane_readings` / `frame_plane_readings_from_proposals`
  (`lattice_planes.py:1227`, `:1568`) measure candidate regions and produce `PlaneReading`
  (`:356-459`). A reading carries `frame`, local `axis`, `side` (which face of the plane),
  `period_pixels`, `phase_pixels`, `extent`, `samples`, `coverage_cells`, `structural_share`,
  `confidence`, the rectification context, and — when the reading came from an exact proposal
  region — `region_evidence` (`PlaneRegionEvidence`, `:148-184`) with `source_region_id`,
  `bbox_xyxy`, packed mask, pixel count and `axis`.
- The distance ambiguity is explicit. `_distance_alternatives` (`:998-1047`) builds
  `DistanceAlternative(b, distance_blocks, support)` for the reference reading `b = 1`, the
  sub-block aliases `b = 1/m` (`m` in 2/4/8/16), and the supra-block aliases `b = k`
  (`k` in 2/3/4/8/16), prunes on `SUPPORT_PRUNE = 0.5 * max(support)`, and orders by support then
  `b` descending. `PlaneReading.distance_for` / `matching_alternatives` (`:418-459`) resolve a
  requested `b` to a distance **only if exactly one retained alternative matches**; missing and
  ambiguous are both a typed `None`.
- The signature is built per `(reading, b)`: `derive_image_derived_signatures`
  (`image_derived_signatures.py:348`) emits one `SignatureEvidence` per accepted reading
  alternative, each carrying a `PlaneSignature` (`lattice_planes.py:462-483`) whose `cells`,
  `interior`, `coverage` and `origin_uv` are a one-cell-per-block grid. `PlaneSignature` may carry
  an alternative-specific `region_footprint` (`PlaneRegionFootprint`, `:187-253`) — inside-region
  and valid sample counts per cell.

**This is the first piece of missing provenance.** A reading's region identity and footprint exist
in memory at this stage, but they are *not* propagated into the graph matcher's captured records
(§4).

### 2.2 From signatures to `PlaneMatch`

`plane_graph.match_plane_graph` (`plane_graph.py:1145-1336`):

1. `_canonical_reading_index` (`:123`) sorts each frame's readings by `reading_order_key` into the
   one canonical `(frame, plane_index) -> PlaneReading` map used by both matcher and solver.
2. Signatures are grouped by frame; the loop is **ordered frame pairs**, then left signatures, then
   right signatures, then `quadrant in range(4)` (`:1205-1212`). Frames are `sorted()`
   (`:1201`), so the enumeration is deterministic.
3. `match_plane_pair` (`:919-1132`) applies, in order: `axis_transform` compatibility for the
   quadrant hypothesis (`:963-969`, `axis_mismatch`); unique selected `b` on both endpoints
   (`:971-977`, `invalid_distance_alternative`); the cell-grid homography (`:978-984`);
   non-degenerate covered maps (`:995-1002`, `no_signal`); whitened phase correlation with support
   (`_correlation_surface_with_support`, `:807-870`) over `[-max_shift_cells, +max_shift_cells]`;
   the overlap floor `MIN_OVERLAP_CELLS = 9` (`:1017-1023`, `insufficient_overlap`); the
   **repetition gate** `min(structural_share) >= MIN_STRUCTURAL_SHARE = 0.15` and `psr >= 3.0`
   (`:1032-1055`, `repetition_unresolved`); then the score
   `0.5*min(psr/PSR_SCALE,1) + 0.3*overlap_fraction + 0.2*clip(margin,0,1) - 0.5*(1-share)`
   (`:1059-1063`).
4. The admitted `PlaneMatch` (`:212-247`) carries the two frame ids, both local axes, the two
   selected `b` values, the relative quadrant, `psr`, `peak`, `second_peak`, `overlap_cells`,
   `score`, the resolved `left_plane`/`right_plane` indices, and `cell_shift_rc`. `integer_residual`
   is `None` until a solution exists.

The trace row `PlanePairTrace` (`:262-287`) additionally records raw-vs-support maxima, overlap
fraction, structural shares and a `selected` flag. Note precisely what a `PlaneMatch` **is not**:
it has no region id, no polygon, no pixel/cell footprint, no occupancy, and no world coordinate.

### 2.3 `_apply_same_axis_filter` — the exact drop rule

`_apply_same_axis_filter` (`plane_graph.py:1410-1468`):

- Groups by `(left frame, right frame, left local axis, relative quadrant)` (`:1434-1437`).
  Mutually exclusive quadrant hypotheses are deliberately *not* compared.
- For each group with `>= 2` members it computes `_delta(match, readings)` (`:1345-1359`):
  `left.side * left.distance_for(left_b) - sigma * right.side * right.distance_for(right_b)`,
  where `(mapped_axis, sigma) = axis_transform(right.axis, quadrant)` and the result is `None`
  unless `mapped_axis == left.axis`. The group's members all describe `t_right - t_left` in the
  left-frame gauge, so agreement in `_delta` means the same measured displacement.
- For every pair of group members: if `|delta_a - delta_b| <= SAME_AXIS_TOLERANCE = 0.15` both
  survive; otherwise the **lower-`score`** member is dropped, and an exact score tie within `1e-12`
  drops **both** (`:1443-1457`). Dropped members get a `PlaneMatchRefusal` with reason
  `same_axis_mismatch` (`:1458-1468`).
- Selection is therefore driven by the **local correlation score**, which is a matcher cue, not a
  geometry measurement. The docstring itself frames this as a cheap equality prefilter.

Measured consequence on the frozen fixture: `1,928 -> 38` (1.97 % kept); on the R10 solver-active
subset `448` rows -> `10` kept / `438` dropped (handoff 392). The filter is not a rare tie-break;
it is the dominant candidate-elimination stage on this input.

### 2.4 The solver's input and what it searches

`plane_graph.solve_component` (`:1503-1597`) receives the **post-filter** tuples plus
`integer_tolerance = 0.15`, `min_resultant = 0.90`, `ambiguous_margin = 0.02`, and the immutable
`quadrant_constraints` tuple. Its search space is *only* the per-frame quadrant assignment:

- up to `4 ** (len(frames) - 1)` labels (`MAX_ASSIGNMENTS = 16,384`, `:1548-1549`), with the
  lexicographically first frame anchored at quadrant `0` — a declared gauge (`:1551-1557`);
- a label is skipped unless it satisfies the connection-graph constraints: the label's relative
  quadrant must equal the solved-seam yaw class for that frame pair (`:1558-1560`);
- for each surviving label, `_evaluate_assignment` (`:1600-1734`) keeps only the matches whose
  relative quadrant equals the label's frame-pair difference (`:1611-1613`), requires both endpoint
  `b` values to resolve uniquely (`:1628-1631`), forms the per-world-axis constraint
  `delta = left_sign*side*distance - right_sign*side*distance` (`:1634-1635`), least-squares solves
  the per-axis frame translations (`:1649-1679`), and scores the axis by the **circular resultant**
  and **mean circular residual** of the face values modulo one block (`:1681-1709`);
- the returned `cost` is an explicit **penalty** (`:1717-1719`):
  `sum((1 - resultant[a]) * observations[a]) + LAMBDA_UNRESOLVED * unresolved
  + MU_MATCH_SCORE * (1 - mean_score)`. Candidates are ranked by **ascending** cost
  (`ranked.sort(key=lambda item: (item[0], item[1].quadrants))`, `:1571`), so the production
  solver **minimizes** cost. `alternatives` keeps the first `MAX_ALTERNATIVES` ranked tuples
  ascending (`:1573`), so `alternatives[0].cost == best_cost`. A ranking rule must therefore not
  read `cost` as a score to maximize.

Gates, in order, after ranking labels by cost (`:1571-1597`): `ambiguous_quadrant`
(`cost` gap `< 0.02` to a rival label whose quadrants differ), `b_unresolved` (`duplicate_b`: two
active edges reuse the same `(left frame, left plane, right frame, right plane, quadrant)` key,
`:1636-1639`), `insufficient_axis_coverage` (`< 3` axes), then
`any(resultant[a] < min_resultant) or any(residual[a] > integer_tolerance)` ->
`plane_integer_residual`, else accepted.

**The decisive structural fact:** the solver never searches `b`/edge identity. Which edges exist,
which `b` each endpoint uses, and which candidates the filter already dropped are all fixed before
the call. Quadrant enumeration and the least-squares fit can only choose *among the constraints it
was handed*. A `b` alias that the filter dropped, or an edge identity the matcher never admitted,
is unreachable to `solve_component` — exactly the gap the R11 substitution arms exploited.

### 2.5 Where a solved component becomes geometry (the output/placement seam)

`image_derived_graph._derive_image_derived_graph` (`image_derived_graph.py:4964-5213`) is the
production G producer. In the ordinary path (no sinks) it: replays the session/lattice/signatures,
binds their digests, derives connection constraints, calls `match_plane_graph` **once**
(`:5060`), calls `solve_component` **once** (`:5075-5079`), and then either

- `solution.reason` non-empty -> `_unavailable(..., solution.reason, "component solver abstained")`
  (`:5199-5201`); the public `derive_image_derived_graph` (`:5216-5227`) passes no sinks at all; or
- accepted -> `ImageDerivedGraphResult(status="assembled", ..., solution)` (`:5208-5213`).

Downstream, `image_derived_geometry_orchestration.derive_image_derived_geometry`
(`image_derived_geometry_orchestration.py:208-307`) runs the fixed stage tuple
`_STAGES = ("f1","g","h","j","k","l","m2","n2","o2","p2","q2")` with per-stage readiness in
`_READY` (`:33-37`). Crucially for this design, it **does not consume an externally supplied graph
result**: the stage `g` call is made *inside* the orchestrator (`:249-253`) with its own frozen
policy arguments (`max_shift_cells=32`, `integer_tolerance=0.15`, `min_resultant=0.9`,
`ambiguous_margin=0.02`), and only `session`, `lattice_result` and an optional `connection_graph`
enter the function. The graph is **recomputed** here.

Stage `j`, `image_derived_graph_policy.bind_image_derived_graph_policy` (`:162-186`), then
**replays the matcher** from the canonical readings:

- `axis_adapter._bind_universe(graph)` (`image_derived_axis_support.py:166-210`) validates that the
  graph's `readings`, `signatures`, `matches` and `component_solution` are mutually consistent and
  that every match endpoint resolves in the canonical reading index;
- `j` then calls `plane_graph.match_plane_graph(readings, raw, max_shift_cells=policy.max_shift_cells)`
  again (`:178-179`) and requires `replayed == graph.matches` (`:183`), else
  `provenance_mismatch` / `graph_result.matches.policy_replay` (`:184-185`).

So the gate into H -> Q2 -> placement is: **one accepted `ComponentSolution` bound to a graph whose
matches are exactly the canonical matcher's output.** An *altered* `JointCandidate` edge set — one
that adds, drops, or re-identifies edges relative to the canonical graph — **cannot be carried
through stages H..Q2**: the `_bind_universe` identity checks and the `j` exact-match replay both
fire. Any experiment that intends to score joint-alternative geometry must therefore address this
seam explicitly; it cannot silently feed replacement matches downstream.

After the stages, `_placement_mapping` (`:80-102`) builds the mandatory keyword bundle
(`axis_support`, `p1d_result`, `vector_result`, `c2rb_result`, `v2a_result`, `v2b_result`, `readings`,
`signatures`, `matches`, `phase_evidence`, `usable_masks`, `split_sources`, `anchor_budget`,
`support_budget`, plus `comparison_tolerance`, `closure_tolerance`) and
`d_mod.assemble_image_derived_placement` (`:293`) binds readings and split sources to the sealed
lattice before calling `assemble_placement_universe(**records)`. What that returns is a
`PlacementUniverse` — an immutable replay tuple, **not** coordinates and **not** a scored output
(§4.3).

Today's frozen run never passes the gateway, which is why the graph payload has empty matches,
`component_solution = null`, and zero placement/volume/promoted cells.

---

## 3. Why solver-consistent alternatives are **not** proof of correct geometry

1. **The gate tests integer consistency modulo one block, not identity.** The resultant is the
   magnitude of the mean unit phasor of `face_values mod 1` (`:1699-1700`); the residual is the mean
   circular deviation (`:1701-1704`). A resultant near `1.0` means "these face values lie on one
   consistent residue lattice", which is equally true of a *wrong* but coherent alias set. Nothing
   in the computation compares a face to a reference surface.
2. **Distance is a derived multiple of an ambiguous period.** `distance_blocks = b / world`, where
   `world = period_pixels * lattice.rectified_pixel_step(extent, samples)` (`:1024`, `:1043`).
   A supra-block alias `b = k` produces `k` times the distance. Because every constraint and the
   circular residual are computed modulo one block, a *consistent* global misuse of the alias can
   still satisfy the gate — the solver cannot distinguish a correct scale from a uniformly
   mis-scaled one.
3. **The signed mapping couples `b` and quadrant.** `_delta` and `_evaluate_assignment` use
   `axis_transform(right.axis, quadrant)` signs (`:1354-1359`, `:1624-1633`). A sign/quadrant
   misassignment can be compensated by a different endpoint alias, so "consistent" is not unique.
4. **The filter's drop decisions are score-driven and local.** `_apply_same_axis_filter` drops the
   lower correlation-score member of a disagreeing pair. Score is a matcher cue built from PSR,
   overlap and peak margin (`:1059-1063`); it is not a truth probability. The accepted arms are
   precisely the ones with **negative** score deltas against the survivors they replace
   (pair-scan §1, §6): the search reached consistency by *accepting worse-scoring* edges.
5. **The R11 substitution arms need not be globally coherent under one `b` per reading.** The scan
   replaced edges one-for-one against the survivors. Independent per-edge substitutions can give a
   single `(frame, plane)` reading two different `b` values across different active edges, or leave
   the surviving edge set globally inconsistent, which the unchanged solver would flag as
   `b_unresolved` (`duplicate_b`) or fail on resultants if it were re-solved as a joint label. The
   pair-scan acceptance therefore establishes per-tuple consistency, **not** that the accepted
   substitutions are jointly coherent under one `b` per reading. Any reuse of those arms requires
   an explicit **coherence control** (§5.4) that re-solves the complete arm under the frozen
   constraints before it can be called coherent.
6. **The accepted alternatives are different edge identities.** Per the handoff (line 390) the two
   measured candidates in the accepted combination use different edge identities from the
   survivors they replaced. Acceptance therefore proves only that *that* set is mutually
   consistent; it carries no statement that the replaced survivor was wrong or that the replacement
   is right.
7. **Abstention is not the only failure mode; a wrong accept is silent.** A gate-passing
   `plane_integer_residual`-free solution feeds H -> Q2 -> placement unchanged, so a wrong-but-
   consistent alias set would produce a confident, fully serialized, *wrong* geometry. This is why
   the next experiment must be scored against reference geometry, not against the gate.
8. **The quadrant gauge and scale are partly unmeasured.** The anchor frame's quadrant is fixed to
   `0` by convention (`:1551`, `:1556`); `scale_known` is set from axis coverage, not from a
   measured metric scale (`:1597`; `ComponentSolution` docstring `:663-681`). So even a "correct"
   solution is anchored to a declared gauge.

---

## 4. The specific identity / footprint / coordinate data that are missing

Read against the actual captured row schemas (R10/R11), not against in-memory types.

### 4.1 What the sealed same-axis capture *does* persist

`SameAxisCandidateCapture` (`image_derived_graph.py:3303-3384`) is deliberately bounded:

- candidate row (19 fields, `_validate_same_axis_candidate_row`, `:3548-3599`):
  `group_index, match_index, row_index, left_ref, left_alt, right_ref, right_alt,
  relative_quadrant, score, psr(+inf tag), peak, second_peak, overlap_cells, shift_row,
  shift_col, delta, disposition, reason`;
- reading table row (`:4347-4348`): `(position, frame id, plane_index, local_axis, side,
  alternative rows)` where each alternative row is `(b, distance_blocks, support)`.

### 4.2 What is therefore missing to bind a candidate to visible support

| Missing datum | Where it exists in memory | Why it matters |
|---|---|---|
| Region identity (`source_region_id`), exact polygon and packed mask | `PlaneRegionEvidence` on `PlaneReading` (`lattice_planes.py:148-184`) | Without it a captured row cannot be joined to a source-image region, so plane-to-truth correspondence cannot be asserted or refuted |
| Alternative-specific footprint (per-cell inside-region / valid sample counts, `origin_uv`, `grid_shape`, `subsample_factor`) | `PlaneRegionFootprint` on `PlaneSignature` (`:187-253`) | This is the only existing bridge from a signature cell to source pixels; it is what would let an evaluator say *which* pixels a `b` alternative actually covers |
| Signature-cell grid geometry and `origin_uv` per retained alternative | `PlaneSignature.origin_uv` (`:481`) | Needed to map `cell_shift_rc` and overlap counts onto real source cells |
| The selected `b`'s world/per-axis face coordinate and frame translation | solver-internal (`_evaluate_assignment`, `:1628-1634`, `:1693-1698`) | The placement stage needs it; the capture keeps only `b`/`distance_blocks`/`delta`, not the solved face coordinate |
| Occupancy / placed-cell / volume coordinates | never produced (pipeline stalls before placement) | No geometry metric can exist until the gateway in §2.5 is passed *and* the coordinate builders in §4.3 succeed |
| Per-frame world-axis mapping and gauge | only the relative-quadrant sequence and `assignments` are captured | Needed to compare a solution to reference geometry under a permitted alignment |
| Source-image identity/pins at the graph row level | session ingress only | A capture row cannot be proven to come from a specific frozen PNG pixel domain, only from a session digest |

### 4.3 `PlacementUniverse` is not coordinates, and assembler reach is not a score

`plane_placement.PlacementUniverse` (`plane_placement.py:592-618`) is an **immutable replay tuple**
of one accepted C2-R-B/V2-A/V2-B component: `c2rb_result`, `vector_result`, `axis_support`,
`p1d_result`, `matches`, `phase_evidence`, `readings`, `signatures`, `v2a_result`, `v2b_result`,
`split_sources`, `anchor_budget`, `support_budget`, `usable_masks`, `comparison_tolerance`,
`closure_tolerance`. It carries upstream records and masks, **not cell coordinates, not a volume,
and not output**. Reaching it means the assembler returned a record; it is not a metric and must
never be reported as if it were.

Output coordinates are produced only by the two builders below, both of which are **replay-bound**
and atomic:

- `plane_placement.resolve_face_placement` (`:1451-1522`). Inputs (keyword-only): `universe`
  (`PlacementUniverse`), `supports` (sequence of `AlignedRegionSupport`, one per reading, unique by
  `reading_identity_sha256`), `budget` (`PlacementBudget`), `producer_version` (must equal the exact
  frozen producer string). It calls `_replay_universe(universe)` to re-derive the component,
  certificate, translations, supported set, `alpha` and binding, binds every support to its unique
  endpoint assignment and retained alternative, rebuilds it from original sources, and only then
  places its supported cells. Exceeding any cap raises `_BudgetStop` and returns a typed
  `PlacementRefusal("budget_exhausted", ...)` that discards every partial face atomically; malformed
  input returns `invalid_input`; a foreign/duplicate support returns `provenance_mismatch`.
  Success returns `ResolvedFacePlacement` whose faces carry `c`, `d`, `translation`, `alpha`,
  `face_gauge`, `face`, `outward_sign`, `normal_block`, `tangent_mappings` and exact integer
  `cells` (`_face_payload`, `:621-643`). **Those integer `cells` are the first extractable
  placement coordinates.**
- `plane_volume.build_plane_volume` (`plane_volume.py:1611-1649`). Inputs (keyword-only): the same
  `universe` and `supports`, the candidate `placement` (`ResolvedFacePlacement`), its
  `placement_budget`, a separate `budget` (`PlaneVolumeBudget`), and an exact `producer_version`. It
  re-runs the accepted V2 producer from the original universe/supports/placement budget and compares
  canonical placement records and every face/support digest **before** any volume-budget decision —
  explicitly so that "a freshly resealed placement that keeps the original bindings (for example a
  `+100` normal translation) is refused" (`:1620-1632`). Each supported face cell emits exactly one
  adjacent block; merged coordinates retain all compatible claims; unknown cells are omitted with
  typed evidence. Any cap or dense bounding-box preflight failure returns a `VolumeRefusal` with no
  cells. **The emitted V3 block cells are the volume coordinates.**

**`PlacementUniverse` is not geometry, and there is no `universe_sha256`.** The assembler record is
an upstream replay tuple and must never be hashed, sealed or reported as a geometry identity.
Digests/identity to bind for scoring are the placement and volume payload digests — recorded here as
**`placement_sha256`** and **`volume_sha256`** (or the builders' own canonical payload digests) —
plus per-face `reading_identity_sha256` / `support_sha256` / `mapping_sha256`. Coordinates are
extracted only through `ResolvedFacePlacement`: `normal_block`, the `tangent_mappings` projected onto
the world axes, and the in-face `lateral_u` / `lateral_v` axes, together with the exact integer face
`cells` and the V3 volume cells. A digest that is not actually recomputed and compared must be typed
unavailable, never asserted.

### 4.4 What the artifacts therefore cannot establish

Per the Lead's own handoff wording (line 394) and the pair-scan report §11: the sealed R10/R11
artifacts capture scalar plane readings and edge ids but **no pixel footprints/polygons and no cell
occupancy**. Consequently neither the accepted exhaustive arm nor the 25 pair-scan arms can be
claimed to have:
plane-to-truth correspondence, correct alias/scale, visible-surface overlap, correct quadrant per
frame, placement coordinates, a geometry recall/precision/F1, exact-state recovery, or a percentage
gain over baseline. Any such claim would require the missing joins above **and** an evaluator-only
reference — which is exactly the separation the proposed experiment must preserve.

---

## 5. Proposed experiment — `DS-IMG-JOINT-ALTERNATIVE-GEOMETRY-PROBE`

### 5.1 One-sentence goal

**Design (not authorize)** one bounded, evaluation-only production-path experiment on the
already-frozen three-view ordinary screenshots that lets the *existing joint selector prototype*
jointly choose the per-frame quadrant **and** one `blocks_per_period` value per touched reading,
then — for any arm that passes the unchanged gates **and** the eval-only arm validator of §5.6 —
drives the unchanged downstream stages to a real placement and volume coordinate set and scores that
geometry against evaluator-only reference geometry, so the result is **usable image-derived geometry
reached and measured**, not solver acceptance. This remains a conditional design: §5.6 item 7 is the
next separately authorized gate, and no image run is authorized here.

### 5.2 What is reused unchanged (the "production path")

- `derive_image_derived_signatures` -> `plane_graph.match_plane_graph` (one matcher pass) ->
  the **complete ordered pre-filter admitted `PlaneMatch` pool** (i.e. before
  `_apply_same_axis_filter`) with its endpoint alternatives; that pool — not the public, filtered
  `graph.matches` — is the arm universe (§5.6).
- The existing internal but unexported prototype
  `src/mc_photogrammetry/reconstruction/_joint_plane_selector.py` (accepted in
  `docs/reports/M4-G1C-joint-selector-2026-09-13.md`): it jointly enumerates an anchored quadrant
  per frame plus one `b` per touched reading; **every complete label induces all compatible pair
  records**, so a caller cannot select or discard edges independently. That structural property is
  what makes it the right search: it preserves the one-`b`-per-reading invariant *by construction*,
  in contrast to the R11 one-for-one substitution approach. It is a **candidate generator**, not a
  truth oracle: it returns `JointCandidate` records plus a Pareto frontier, and it does **not**
  accept `quadrant_constraints` or `ambiguous_margin` (§5.4).
- The unchanged gates `max_shift_cells = 32`, `integer_tolerance = 0.15`, `min_resultant = 0.90`,
  `ambiguous_margin = 0.02`, `MIN_OVERLAP_CELLS = 9`, `MIN_STRUCTURAL_SHARE = 0.15`
  (`plane_graph.py:78-94`; the production call site
  `image_derived_geometry_orchestration.py:249-253`).
- The unchanged downstream stages H -> Q2 and `assemble_image_derived_placement`, then the two
  replay-bound coordinate builders `resolve_face_placement` and `build_plane_volume`, so placement
  and volume coordinates come from the accepted production assemblers, not a bespoke path.
- The unchanged camera/bounds policy: image-derived session/calibration only; no evaluator camera,
  no manifest, no F3, no manual coordinates, no reference bounds inside inference.

**Not reused:** the R11 substitution arms are *not* re-run as arms. The pair scan already measured
them and its artifact is sealed; re-running it would add cost without adding geometry. They enter
only through the §5.4 coherence control, which is a read-only re-solve of the already-captured arm
inputs under the frozen constraints.

### 5.3 Frozen inputs

- The same ordinary pinned three-view screenshot set used by the accepted G1 observation
  (Template 2), identified by its existing session/lattice/signature ingress pins. No new capture,
  no Minecraft, no launcher, no UI. If the pinned input set cannot be replayed from the sealed
  ingress without opening a forbidden payload, the experiment stops rather than substituting
  inputs.
- Freeze and record: interpreter version, module/inventory hashes (including the selector), gate
  values, candidate/solver budgets, and the exact command.

### 5.4 Search, budgets and determinism

Two-level bounded search; the first bound is on **joint labels**, the second on **arms driven
downstream**.

**Joint-label budget (per arm).** `_joint_plane_selector.SearchBudget` has no defaults and limits
seven counters independently (`_joint_plane_selector.py:62-85`): `max_search_nodes`,
`max_compatibility_checks`, `max_objective_rows`, `max_axis_solves`, `max_pareto_comparisons`,
`max_candidate_certificates`, `max_buffered_match_rows`. Use the M4-G1C conservative candidate as
the *starting* values to be re-derived for this 3-frame fixture and frozen **before** the run:
`32,768 / 5,000,000 / 200,000 / 10,000 / 1,560 / 40 / 2,520` (M4-G1C report §"synthetic full-stage
profile"). Any exceeded counter returns `budget_exhausted` with no candidate, no Pareto set and no
certificate — an atomic abstention, never a partial arm.

**Reconciling the selector caps with any arm target.** The M4-G1C
`max_candidate_certificates = 40` and `max_buffered_match_rows = 2,520` are *selector-call* caps
inherited unchanged from the prototype's synthetic profile. They are the only values measured for
the prototype, and they do **not** justify a 64-arm (or any fixed-count) downstream target: a
certificate is an enumerated candidate and a buffered row is one active-match row, neither of which
is a geometry arm, so no arithmetic on 40 or 2,520 yields an arm count. This revision therefore does
**not** claim those values must be re-derived to `>= 64`. The arm count is derived from the fixture
and the coherent budget below, and the run records which single limit bound the result instead of
assuming a target.

**One coherent, atomic budget.** The certificate, active-row, verifier, downstream, encoded-byte and
wall-clock limits are frozen together as one reviewed tuple **before** the run, sized from the
fixture rather than from unrelated caps, and every one of them is enforced atomically:

| Limit | Meaning | Overflow outcome |
|---|---|---|
| `max_candidate_certificates` | gated `JointCandidate`s buffered per selector call (M4-G1C default 40) | `budget_exhausted`; no candidate, no frontier, no certificate |
| `max_buffered_match_rows` | buffered active-match rows per selector call (M4-G1C default 2,520) | `budget_exhausted`, atomic |
| `MAX_VERIFIED_ARMS` | gate-passing arms the independent verifier may accept and drive downstream | typed refusal for each excess arm; none partially driven |
| `MAX_SOLVER_CALLS` | total `solve_component`-equivalent evaluations across baseline, verifier and arms | `budget_exhausted`, atomic |
| `MAX_CAPTURE_BYTES` | encoded bytes of the default-off capture (§5.7) | typed capture refusal with no truncated record; geometry unaffected |
| `MAX_WALL_CLOCK` | total run time, measured rather than inferred | typed timeout refusal; no partial arm, no seal, no metric |

A partial arm, a truncated row set or a partially sealed capture is never produced: every limit
above is an atomic typed refusal. Any value that would truncate the enumeration, the verifier, the
downstream drive, the byte capture or the time box is a declared, reviewed budget decision made
**before** the run — not a silent loosening during it.

**Verified-arm budget.** Drive downstream only as many gate-passing arms as the frozen coherent
budget admits. `MAX_VERIFIED_ARMS` is a declared fixture-derived value, never justified by the
M4-G1C certificate/row caps (A9), and `MAX_SOLVER_CALLS` bounds the total
`solve_component`-equivalent evaluations across the baseline, the independent verifier and every arm
(the selector's internal objective evaluation counts against its own budget; any independent
production re-solve counts here). If the frozen enumeration returns fewer gate-passing arms than the
cap, drive them all. Reasons to keep the downstream cap modest rather than running the full accepted
set: the downstream stages (H -> Q2, placement, volume) are the expensive part, and a modest cap
still tests the *storage/output* claim (does an accepted arm actually produce a placement and volume
coordinate set?) many times over while keeping the run bounded. The accepted-set size itself (25 in
the R11 landscape) is reported as a count, not run to exhaustion here.

**Independent verification (mandatory, not the selector's own verdict).** The selector returns a
`JointCandidate` (and a Pareto frontier) but does **not** accept `quadrant_constraints` and does
**not** apply `ambiguous_margin = 0.02`; its `_make_candidate` gate (`:1123-1128`) checks
`>= 3` axes, `min_resultant` and `integer_tolerance` only, and it has no `duplicate_b` gate at all
(its own comment at `:1120-1122` says the legacy sentinel "cannot be reused here" for one-`b`-per-
reading labels). For every candidate that is to be driven downstream, the experiment must therefore
run the **unchanged `plane_graph.solve_component`** on the arm's edge set under the frozen
`min_resultant = 0.90`, `integer_tolerance = 0.15`, `ambiguous_margin = 0.02`. The
`quadrant_constraints` are **frozen once from the baseline** canonical solve and are **never derived
from each candidate**; a label incompatible with the frozen constraints is refused
(`quadrant_constraint_inconsistent`) rather than having the constraints redesigned around it. The
verifier then (a) compares its **returned assignment** against the arm label's quadrants — any
difference is a refusal — and (b) explicitly verifies **one `b` per reading** across the arm's
active edges (`duplicate_b == false`, and no `(frame, plane_index)` reading carrying two different
`b` values). Only an accepted `ComponentSolution` (empty `reason`) whose returned assignment equals
the label and whose `b` map is single-valued is eligible; `ambiguous_quadrant`, `b_unresolved`,
`quadrant_constraint_inconsistent`, `insufficient_axis_coverage` or `plane_integer_residual` from
that verifier is a typed refusal for that arm, and the selector's own acceptance never substitutes
for it. This verifier is also the **coherence control** for §3.5: re-solving an R11-captured arm
under one `b` per reading is what determines whether its substitutions were globally coherent.

**Frontier vs all labels; no silent choice.** `_finish` (`:1246-1257`) returns `selected` only when
the Pareto frontier has exactly one member; otherwise it returns `ambiguous_quadrant`
(`equivalent_half_turn_labels`) or `b_unresolved` (`multiple_pareto_maxima`) with the full frontier
tuple and **no** `candidate`. The Pareto reduction (`_all_pairs_frontier`, `:1306-1330`) keeps only
nondominated candidates under a deliberately nontransitive dominance relation, so the frontier is a
strict subset of all gate-passing labels. The experiment must:

- record **all labels** the enumeration produced (each with its gate outcome), and separately
  record the **frontier** the selector returned, labelled as such;
- never treat `SelectorResult.candidate` as available when the status is `ambiguous_quadrant` or
  `b_unresolved` — the selector has nominated no candidate, and silently promoting one as a
  production choice is forbidden; frontier members may only be evaluated separately, per the bullet
  below;
- if the selector status is ambiguous (`ambiguous_quadrant` / `b_unresolved`), evaluate the bounded
  Pareto frontier **separately as evaluation-only arms, in stable identity order** (the canonical
  nine-scalar identity list, ascending), with **no production nomination** of any member and no
  silent choice; each member's independent-verifier and downstream outcome is reported on its own,
  and a refused member is a typed refusal for that member only. If even this separate evaluation
  cannot be source-grounded under §5.6, type-refuse the whole frontier and stop at the exact seam.

**Deterministic enumeration and ties.** The R11 landscape reports negative per-axis correlation-score
deltas on some accepted arms, but that diagnostic does **not** establish that a score-first rule
would discard the entire accepted set, and no such claim is relied on here. The solver's `cost` is a
**penalty** and must be read as ascending. Freeze, before the run, an ordering that uses **only** the
unchanged gates, the solver's own returned values, and the independent verifier's verdict:

1. `duplicate_b == false` before `true` (a gate, not a preference);
2. descending `min(resultant[a] - min_resultant)` over matched axes (the unchanged resultant gate
   margin);
3. ascending `max(residual[a] - integer_tolerance)` over matched axes (the unchanged residual gate
   margin);
4. descending sum of matched-axis resultants;
5. **ascending `JointCandidate.cost`** as returned by the selector's objective (the same penalty
   formula as `solve_component`'s `cost`, `:1717-1719`, which the production solver **minimizes**).
   Do **not** use `alternatives[0].cost` as a separate criterion — it is the same best cost as the
   candidate's, and it must not be ranked descending;
6. ascending lexicographic order of the canonical nine-scalar arm identity list.

Ties beyond (6) are impossible because the identity list is exact. Report whether any alternative
ordering would have selected a different arm — no "best arm" is promoted, and the frontier is
reported as a set, following the R11 discipline.

**Duplicate-edge / frame / axis constraints (enforced, not assumed).**

- *Duplicate edge / one `b` per reading:* because a joint label induces a complete edge set, the
  same reading cannot simultaneously carry two different `b` values; the experiment must
  additionally assert that the label's active edge keys
  `(left frame, left plane, right frame, right plane, quadrant)` are unique per reading pair — the
  independent `solve_component` verifier's `duplicate_b` must be `false` — and must refuse
  `invalid_input` on any two edges claiming different `b` for the same `(frame, plane_index)`
  reading within one arm.
- *Frame:* the component's frame universe must equal the session's admitted frames (all three), and
  every active edge's endpoints must resolve inside that universe.
- *Axis:* unchanged axis-coverage gate (all three matched axes), with the baseline assignment
  `[0, 3, 1]` and the baseline-frozen `quadrant_constraints` reused rather than re-searched; a label
  whose quadrants are incompatible with those frozen constraints is refused.
- *Identity:* each captured candidate must resolve through the canonical
  `(frame, plane_index) -> reading` index **and** must be present in the complete ordered pre-filter
  admitted pool bound in §5.6; a foreign reading, an edge absent from that pool, or a mismatched
  axis is `invalid_input`.

### 5.5 Arm output and abstention

For every arm driven downstream, the run records, in order:

1. the arm's joint label (quadrants + per-reading `b`), its nine-scalar edge identities, its gate
   values (resultants, residuals, observations, `duplicate_b`), and the independent
   `solve_component` verifier's returned reason (empty only on acceptance);
2. `component_solution` reached / not reached, and on reach the unchanged H -> Q2 stage statuses;
3. `assemble_image_derived_placement` outcome: the `PlacementUniverse` replay record — **not
   geometry, and with no `universe_sha256`** — or the typed `ImageDerivedPlacementUnavailable`
   (`geometry_unavailable` / `missing_prerequisite`). Reaching a `PlacementUniverse` is recorded as
   **universe-assembled only** — it is neither coordinates nor a score;
4. `resolve_face_placement` outcome: `ResolvedFacePlacement` with its exact integer face `cells`
   and `placement_sha256`, extracting coordinates only via `normal_block`, the tangent world axes
   and `lateral_u` / `lateral_v`, or the typed `PlacementRefusal` (`invalid_input`,
   `provenance_mismatch`, `budget_exhausted`);
5. `build_plane_volume` outcome: V3 block cells with `volume_sha256`, or the typed
   `VolumeRefusal`; and
6. the serialized placement/volume coordinates, if any, **sealed and hashed before any evaluator
   read**.

Typed abstention rules — each produces **no** geometry and **no** metric, and is never replaced by
a fallback or a looser re-run:

- selector `budget_exhausted` (any counter) -> one typed refusal, zero arms downstream;
- selector `ambiguous_quadrant` / `b_unresolved` -> **no** arm is nominated as production; the full
  frontier is reported and may be evaluated only as separate eval-only arms in stable identity order
  (§5.4); if that separate evaluation is not source-groundable, the frontier is type-refused;
- independent verifier `ambiguous_quadrant`, `b_unresolved`, `quadrant_constraint_inconsistent`,
  `insufficient_axis_coverage`, `component_too_large`, `no_matches`, `no_plane_reading`,
  `plane_integer_residual` -> the verifier's own typed reason, reported per arm, no geometry;
- no gate-passing label -> `no_accepted_arm`;
- any downstream stage not `_READY` -> `g_unavailable` with that stage's typed reason;
- `ImageDerivedPlacementUnavailable` -> typed unavailable (`missing_prerequisite` /
  `geometry_unavailable`);
- `PlacementRefusal` / `VolumeRefusal` -> that builder's typed reason, no partial coordinates;
- malformed / capped / non-finite input -> atomic typed refusal, no partial record.

### 5.6 Canonical-seam continuation (eval-only arm validator; otherwise stop at the exact seam)

Because `derive_image_derived_geometry` recomputes the graph and stage `j` requires exact
`graph_result.matches` (§2.5), a joint-alternative arm **cannot** be continued by handing the
orchestrator a modified graph or a modified match tuple. The design therefore specifies a
**separate, evaluation-only arm validator** with the rules below, and otherwise stops at the exact
seam. The validator is not the production graph and does not stand in for it.

1. **Preserve the canonical baseline G/J.** The canonical G producer and stage J run exactly as
   accepted, with their statuses and digests unchanged. The validator never mutates, filters,
   reorders or replaces `graph.matches`, `graph.readings`, `graph.signatures` or their digests, does
   **not forge a production graph result**, and does **not weaken J**: the canonical matcher replay
   and its exact-match requirement stay byte-for-byte as accepted.
2. **Preserve the complete ordered pre-filter admitted pool.** The arm universe is the **complete
   ordered pre-filter admitted `PlaneMatch` pool** produced by `match_plane_graph` **before**
   `_apply_same_axis_filter` (the R10 `1,928`-row admitted universe), not the post-filter survivors.
   This is required because the public `graph.matches` is empty on the frozen run and, when present,
   is the filtered set: a membership rule stated against `graph.matches` would silently exclude
   exactly the dropped arms (including the R11 arms) that the design exists to evaluate.
3. **Digest binding (source grounding).** The validator binds that complete ordered pool to the
   session, lattice and signature digests it was replayed from, and refuses
   (`arm_pool_not_source_grounded`) if the pool cannot be re-derived from those digests in canonical
   order with canonical `(frame, plane_index)` keys and a unique per-endpoint `b`. Every active match
   must resolve its endpoints in the canonical reading index. A foreign or non-canonical edge is a
   typed refusal; the validator never invents, re-admits or synthesizes a missing edge.
4. **Frozen constraints, returned assignment, single-valued `b`.** `quadrant_constraints` come from
   the baseline solve only. For each arm the validator runs the **unchanged** `solve_component` under
   `.90/.15/.02`, compares the **returned assignment** with the arm label, and explicitly verifies
   **one `b` per reading**. A label incompatible with the frozen constraints, or a returned
   assignment that differs from the label, is a typed refusal — constraints are never redesigned per
   candidate.
5. **Downstream reach.** Only after (1)-(4) succeed may the accepted `ComponentSolution` be offered
   to the unchanged H -> Q2 -> placement -> volume stages. Reaching geometry still requires the
   `placement_sha256` / `volume_sha256` coordinate path of §4.3/§5.5; a `PlacementUniverse` alone is
   not geometry.
6. **Stop at the exact seam; no forged graph, no image run.** If the validator cannot source-ground
   the pre-filter pool from the session/lattice/signature digests, or if the downstream stages cannot
   be reached without forging a production graph result or weakening J, then the honest outcome is
   **"stopped at the exact seam; no scoreable geometry"**. No substituted-match continuation, no
   forged graph and no image run is authorized by this report.
7. **Next separate gate (synthetic proof; no images).** Before any image run, the next authorization
   gate is a focused synthetic proof on synthetic inputs only: (a) a **genuinely dropped** arm (one
   `_apply_same_axis_filter` removed) is source-grounded from the ordered pre-filter admitted pool
   and reaches placement **and** volume coordinates under the unchanged downstream stages; (b) a
   foreign/non-canonical edge, an ungrounded pool, and a label incompatible with the frozen
   `quadrant_constraints` are each refused with no geometry; and (c) the baseline path is
   parity-identical in statuses and digests when the validator is disabled. Only that proof can
   promote the validator from design to a candidate for authorization.

The validator is new code and requires separate authorization; this section specifies its contract,
it does not authorize its implementation. If that authorization is not granted, the standing outcome
is the exact-seam stop above, and the synthetic proof is the next gate — not an image run.

### 5.7 Output-neutral capture

The new material is bound only through a **default-off** optional capture on the existing
failure-observer family, so that a default production call allocates nothing new and returns a
byte-identical result and digest:

- the new parameters default to `None`/off, and when off the production call must be
  byte-identical to the current `derive_image_derived_graph` / `derive_image_derived_geometry`
  result and digests (asserted by test, not by convention);
- the capture may record: arm count, per-arm label, per-arm gate values, verifier outcomes, budget
  counters, the selector status and frontier membership flags, per-stage statuses, the
  `placement_sha256` / `volume_sha256` payload digests, and the extracted coordinate digests — never
  a `universe_sha256` — under the declared atomic `MAX_CAPTURE_BYTES` / row cap of §5.4, with an
  atomic typed refusal on overflow;
- no public payload/digest changes, no schema change, no new serialized field on the public graph
  or geometry result;
- the search and any independent production solve must be **additional** passes, not a replacement
  of the existing single call, so the baseline path stays exactly as accepted.

### 5.8 Evaluator separation (hard ordering)

`production < seal < truth`, already the accepted discipline for the image-only attribution
runners:

1. Phase A (image-only, no evaluator): run the baseline arm and every counterfactual arm to be
   reported; serialize and hash the geometry outputs (universe, placement, volume coordinates).
   Nothing in inference may import or receive a reference, visible reference, alignment, evaluator
   camera, manifest, F3 value, direction, manual coordinate or frozen mask
   (`DS-STAGEWISE-LOSS-TEST-PLAN` §5 rules 1-4).
2. **Seal before truth.** Seal the ordinary baseline result **and all** counterfactual arms —
   including the ones that abstained, recorded as typed unavailable — before any truth or reference
   read. No arm may be added, dropped, or re-run after the seal.
3. Phase B (evaluator-only): score the sealed Phase A coordinates against the frozen Template 2
   visible reference using the existing evaluator interfaces
   (`evaluation_image_only_attribution.align_candidate_coordinates:757`,
   `score_aligned_cells:890`, `score_aligned_geometry_coordinates:979`;
   `evaluation.visible_metrics:77`, `best_visible_alignment:113`).
4. **First scoreability** is reported explicitly: which arm first produced a scoreable coordinate
   set, and how many of the sealed arms were scoreable at all (unavailable is a category, not a
   zero).
5. Reference-assisted rows are labelled and never pooled with image-only rows; the evaluator
   camera arm (if any) is explicitly out of promotion.

### 5.9 The measurable geometry metric

| Field | Definition here |
|---|---|
| Quantity | geometry, `MetricSpace.GEOMETRY`, universe `V` = frozen visible reference cells (`DS-STAGEWISE-LOSS-TEST-PLAN` §1.1, §2.2) |
| Primary rows | **absolute per-arm** visible geometry recall `\|C∩V\|/\|V\|`, precision `\|C∩V\|/(\|C∩V\|+\|C−R\|)`, and F1 for that arm's `resolve_face_placement` face-cell set and `build_plane_volume` volume-cell set `C` — each arm scored on its own, never pooled |
| Checkpoints | `placed` geometry recall and `volume_cell` geometry recall (stage plan §2.7) |
| Gained / lost / net | **not computed.** The baseline is geometry-unavailable, so there is no baseline cell set to difference against and no valid denominator for a delta |
| Alignment | the permitted alignment only: global integer translation + lattice Y-quarter-turn, no reflection, no scale fit; the chosen alignment and `V` are reported explicitly |
| Baseline | the frozen ordinary run, which abstains at `plane_integer_residual`: its G row and its geometry are **typed unavailable, not 0**. It is sealed alongside the arms but contributes no metric and no comparison |
| Secondary (never a substitute) | `phantom` / `CellConflict` / `OmittedCell` counts; typed `g_unavailable` reason when no arm reaches placement |
| Explicitly not a metric | solver acceptance, per-axis resultant/residual, candidate counts, selector status, `PlacementUniverse` reach, or score deltas — all are pre-geometry and are reported as gate diagnostics only |

**No exact-state score is reported.** The run does not score `promoted_cells` or any exact-state
recovery claim; the metric is limited to the placed and volume coordinate sets against visible
reference cells.

**What this experiment can and cannot conclude.** It can conclude: *"under the unchanged gates, the
frozen inputs and the eval-only arm validator, `k` of the sealed arms were scoreable, and arm `i`'s
placed/volume coordinate sets scored an absolute visible-geometry recall of `x`, precision `y`, F1
`z` against the reference under the permitted integer alignment."* It **cannot** conclude that the
search causes a geometry improvement, that any arm is better than the baseline, or that the result
is promotable: the baseline G is unavailable, there is no gained/lost/net comparison and **no
delta**, only **absolute per-arm** geometry, there is **no exact-state result**, the metric
universe is one frozen fixture, and the alignment gauge/scale caveat (§3.2, §3.8) still applies.
Any improvement claim would require a separately authorized baseline with its own scoreable
geometry, sealed by the same `production < seal < truth` order.

### 5.10 Required tests (synthetic/focused, no image or solver workload)

Following `docs/deepseek-worker.md` and prior accepted tasks, only focused modules may run:

1. **Default-path output neutrality:** with the new capture off, the graph/geometry result and all
   digests are byte-identical to the current path; no new allocation on the default call.
2. **Joint-label invariants:** duplicate-edge rejection (one reading, two `b` values) is detected
   and refused; an explicit one-`b`-per-`(frame, plane_index)` check is enforced; frame-universe
   mismatch refused; a foreign reading/plane index and an edge absent from the ordered pre-filter
   admitted pool refused; axis-coverage rule unchanged.
3. **Independent verifier parity:** for a synthetic label whose quadrants and `b` match the
   canonical accepted solution, the unchanged `solve_component` under the **baseline-frozen**
   `quadrant_constraints` / `.90` / `.15` / `.02` returns an accepted solution whose **returned
   assignment equals the label** and whose `b` map is single-valued; for a synthetic ambiguous,
   `duplicate_b`, or constraint-incompatible label it returns the typed refusal the selector would
   have missed (the key regression: selector acceptance never substitutes for the verifier, and
   constraints are never re-derived per candidate).
4. **Frontier handling:** a synthetic `b_unresolved` / `ambiguous_quadrant` selector result
   nominates no arm, reports the frontier, and evaluates its members separately in stable identity
   order as eval-only arms; no member is silently promoted; an ungrounded frontier is type-refused.
5. **Eval-only arm validator (§5.6):** synthetic proof on synthetic inputs only — (a) a **genuinely
   dropped** arm (removed by `_apply_same_axis_filter`) is source-grounded from the ordered
   pre-filter admitted pool and reaches placement **and** volume coordinates under the unchanged
   downstream stages; (b) a foreign/non-canonical edge, an ungrounded pool, and a label incompatible
   with the frozen `quadrant_constraints` are each refused with no geometry; (c) the baseline path
   is parity-identical in statuses and digests when the validator is disabled. No image run.
6. **Deterministic enumeration:** a synthetic tie set exercises the frozen 6-step order including
   ascending `JointCandidate.cost` and the exact-identity final key; permutation of input order
   yields the same arm set; a synthetic arm with a negative score delta is ordered by the frozen
   criteria and is not implicitly excluded by any score-based shortcut (no claim is made that a
   score-first rule would discard the entire accepted set).
7. **Budget atomicity:** each of the six coherent limits of §5.4 — certificate, buffered row,
   verifier arm, solver call, capture byte and wall clock — exhausted -> one typed refusal, no
   partial arm, no partial certificate, no partial seal; `MAX_VERIFIED_ARMS` and the global
   `MAX_SOLVER_CALLS` enforced across all calls; over-cap row/byte capture refused atomically.
8. **Abstention matrix:** `no_accepted_arm`, `ambiguous_quadrant`, `b_unresolved`,
   `quadrant_constraint_inconsistent`, `insufficient_axis_coverage`, `component_too_large`,
   `no_matches`, `g_unavailable`, `missing_prerequisite`, `PlacementRefusal`, `VolumeRefusal` each
   produce the typed outcome and no geometry.
9. **Evaluator separation:** an import/contract test that no production module can reach a
   reference/evaluator camera/manifest/mask, and that Phase B refuses to run before the Phase A
   seal exists for the baseline **and** every counterfactual arm.
10. **Placement/volume round trip and refusal:** synthetic accepted arm -> `PlacementUniverse`
    (recorded as a replay tuple, never a `universe_sha256`) -> `resolve_face_placement` ->
    `build_plane_volume` with known integer coordinates, asserting the extracted cell sets via
    `normal_block`, the tangent world axes and `lateral_u`/`lateral_v`, plus `placement_sha256` /
    `volume_sha256`; and the replay-bound refusals (a resealed placement with a `+100` normal
    translation, a duplicate support, a malformed budget) return their typed refusal with no cells.
11. **Output-neutral diagnostic record:** canonical-round-trip and strict-JSON replay of the new
    capture; tamper detection; public payload/digest parity.

### 5.11 Expected outcomes and what each would establish

| Outcome | Interpretation |
|---|---|
| No label passes the gates | The joint search over the same measured alternatives cannot reach the unchanged gates on this fixture; no geometry metric exists. This is a one-fixture negative result and does **not** by itself prove that evidence quality (rather than search or solver design) is the bottleneck. |
| Selector ambiguous / `b_unresolved` frontier | The bounded search found multiple nondominated candidates. Each frontier member is evaluated separately as an eval-only arm in stable identity order with **no production nomination**; members can individually produce typed refusals or, when source-groundable, coordinates — but none is nominated. |
| A label passes but no arm reaches placement | The gateway accepted, but an eval-only-arm-validator check or a downstream stage refused; report each typed stage reason. Geometry still unavailable, not zero. |
| An arm reaches `PlacementUniverse` but not coordinates | The assembler record exists but `resolve_face_placement` or `build_plane_volume` refused; report the typed refusal. This is explicitly **not** a score. |
| An arm produces scoreable coordinates and scores near zero recall | Solver acceptance is shown to be achievable with wrong geometry — the direct empirical answer to "is consistency proof?" |
| An arm produces scoreable coordinates with a nonzero absolute recall/precision/F1 | First scoreable Route B geometry tail with an absolute per-arm measurement; still one-fixture evidence, no baseline comparison, not promotion. |
| Eval-only arm validator cannot source-ground the pre-filter pool, or refuses every arm | The run stops at the **exact seam** and reports exactly that; no substituted-match continuation, no forged production graph, no weakened J and no metric. The synthetic proof (§5.6 item 7) is the next separate gate. |
| Budget exhaustion | One of the six coherent limits bound the result; report which single limit fired and the counters, and do not loosen any bound silently or leave a partial arm/seal. |
| Synthetic proof (next gate) not authorized or fails | No image run occurs; the design stays conditional and the failing synthetic check (dropped-arm placement+volume, foreign-edge refusal, baseline parity) is reported as the blocker. |

---

## 6. Exact source references used

Source (read-only): `src/mc_photogrammetry/reconstruction/plane_graph.py` (`PlaneMatch` `:212-247`,
`match_plane_pair` `:919-1132`, `match_plane_graph` `:1145-1336`, `_match_order` `:1339-1342`,
`_delta` `:1345-1359`, `annotate_integer_residuals` `:1362-1407`, `_apply_same_axis_filter`
`:1410-1468`, `_match_key` `:1471-1473`, `solve_component` `:1503-1597`, `_evaluate_assignment`
`:1600-1734` including the ascending-cost sort `:1571` and the cost formula `:1717-1719`,
`ComponentAlternative` `:655-659`, `ComponentSolution` `:662-681`, constants `:72-94`, `:202-208`);
`src/mc_photogrammetry/reconstruction/lattice_planes.py` (`DistanceAlternative` `:116-128`,
`PlaneRegionEvidence` `:148-184`, `PlaneRegionFootprint` `:187-253`, `PlaneReading` `:356-459`,
`PlaneSignature` `:462-483`, `_distance_alternatives` `:998-1047`);
`src/mc_photogrammetry/reconstruction/plane_placement.py` (`PlacementUniverse` `:592-618`,
`_face_payload` `:621-643`, `resolve_face_placement` `:1451-1522`);
`src/mc_photogrammetry/reconstruction/plane_volume.py` (`build_plane_volume` `:1611-1649`);
`src/mc_photogrammetry/pipeline/image_derived_signatures.py` (`derive_image_derived_signatures`
`:348`); `src/mc_photogrammetry/pipeline/image_derived_graph.py`
(`_derive_image_derived_graph` `:4964-5213`, `derive_image_derived_graph` `:5216-5227`,
`SameAxisCandidateCapture` `:3303-3384`, `_validate_same_axis_candidate_row` `:3548-3599`,
`_collect_same_axis_candidates` `:4114-4405`);
`src/mc_photogrammetry/pipeline/image_derived_graph_attribution.py`
(`ImageDerivedGraphAttribution` `:370-431`, `derive_image_derived_graph_attribution` `:434-500`);
`src/mc_photogrammetry/pipeline/image_derived_graph_policy.py`
(`bind_image_derived_graph_policy` `:162-186`, matcher replay `:178-179`, exact-match requirement
`:183-185`); `src/mc_photogrammetry/pipeline/image_derived_axis_support.py`
(`_bind_universe` `:166-210`, `derive_image_derived_axis_support` `:213-229`);
`src/mc_photogrammetry/pipeline/image_derived_geometry_orchestration.py` (`_STAGES`/`_READY`
`:33-37`, `derive_image_derived_geometry` `:208-307`, graph call `:249-253`, `_placement_mapping`
`:80-102`, placement call `:293`); `src/mc_photogrammetry/pipeline/image_derived_placement.py`
(`assemble_image_derived_placement` `:228-300`);
`src/mc_photogrammetry/reconstruction/_joint_plane_selector.py`
(`SearchBudget` `:62-85`, `SearchCounters` `:88-123`, `JointCandidate` `:126-161`,
`SelectorResult` `:164-171`, `_evaluate_objective` `:838-875`, `_dominates` `:1062-1100`,
`_make_candidate` gates `:1103-1167`, `_candidate_payload` `:1204-1220`, `_result`/`_finish`
`:1223-1257`, `_all_pairs_frontier` `:1306-1330`, `select_joint_labels` `:1539-1557`);
`src/mc_photogrammetry/evaluation.py` (`visible_metrics` `:77`, `best_visible_alignment` `:113`);
`src/mc_photogrammetry/evaluation_image_only_attribution.py`
(`align_candidate_coordinates` `:757`, `score_aligned_cells` `:890`,
`score_aligned_geometry_coordinates` `:979`).

Reports/handoff (quoted, not executed): `docs/reports/DS-IMG-R11-AXES02-PAIR-SCAN-2026-10-04.md`;
`docs/reports/DS-STAGEWISE-LOSS-TEST-PLAN-2026-10-04.md`; `docs/reports/M4-G1C-joint-selector-2026-09-13.md`;
`docs/reports/DS-R4-V2-LOSS-DECOMPOSITION-2026-10-04.md`; `AGENT_HANDOFF.md:389-414`;
`AGENT_COMMS.md` entries 5225-5226, 5325, 5754, 5853, 6681-6695, 7389-7391.

## 7. Boundaries and non-claims

- **No run, edit, test, capture, solver call, threshold/filter change, Minecraft/launcher/UI.**
  This report is a **conditional design, not an authorized run**: every proposed element requires
  separate authorization, including the eval-only arm validator specified (not implemented) in §5.6
  and the synthetic proof of §5.6 item 7. No image run is authorized, and nothing was executed to
  produce this revision.
- **No truth/reference/camera/mask/image-payload/evaluator read while performing this audit.**
  The proposed Phase B is described, not performed.
- **No geometry, accuracy, promotion or causality claim.** The 25/6,068 and `1,928 -> 38` figures
  are gate/candidate diagnostics, not accuracy scores. `geometry_unavailable` stays typed
  unavailable, never `0`. The baseline is not compared against any arm (§5.9).
- **No production filter change is proposed.** The experiment tests whether a joint search is worth
  a later, separately reviewed filter change; it does not change the current one.
- **Supplied camera/F3/directions/manual coordinates remain evaluation-only and advanced-only**,
  and never enter inference.

## 8. Direct-shell / read-only compliance

Per `docs/deepseek-worker.md`: no junction/reparse creation or manipulation, no recursive deletion,
no encoded or obfuscated commands, no installers or generated executables, no background or
persistent processes, no Defender/system changes, no external temp inspection, no subagents, and no
client/Minecraft automation. **Only this report file was written.** No `AGENT_COMMS.md` result, and
no production source, test, runner, pipeline, artifact or schema was edited.

---

## 9. Artifact digest (report only)

- Path: `docs/reports/DS-IMG-SAME-AXIS-SEARCH-DESIGN-R2-2026-10-05.md`
- Encoding as written: UTF-8, LF line endings (the editing environment wrote the literal content;
  a Windows checkout may present CRLF on disk).
- Bytes and SHA-256: **typed unavailable in this revision session.** The editing agent had no shell
  or hashing tool, so neither a byte count nor a digest could be measured; per this file's own
  discipline an unverified byte count or digest must not be pinned as if measured. A later revision
  may pin them once a hashing tool is available; until then any byte count or SHA-256 quoted for
  this file is unverified and must not be used for seal, tamper or equality checks.
- To pin it, run on the file bytes (case-insensitive on Windows shells):

  ```text
  (Get-Item docs/reports/DS-IMG-SAME-AXIS-SEARCH-DESIGN-R2-2026-10-05.md).Length
  Get-FileHash -Algorithm SHA256 docs/reports/DS-IMG-SAME-AXIS-SEARCH-DESIGN-R2-2026-10-05.md
  ```

  and report the two outputs alongside this section. Until that is done, any byte count or SHA-256
  for this report — including one quoted in `AGENT_COMMS.md` — is unverified and must not be used
  for seal, tamper or equality checks.
