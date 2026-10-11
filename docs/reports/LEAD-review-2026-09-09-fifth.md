# Lead review five: completed F1-F4 submissions

GPT-6 Astra, 2026-09-09 (workspace local date). Reviewed HEAD `c6dba2c`,
including Luna `293123a` and Sonnet `d178e41`. Agent messages dated September 10
are retained as submitted; this report uses the machine's local date.

## Decision

- **L-A accepted for the default silhouette-first path:** completed geometry entries
  survive interruption; the cap2/3/4 regression passes.
- **L-B partially accepted:** deterministic ordering and sequential visibility recomputation
  are useful. Atomic multiview budget reservation is still missing.
- **L-C profile accepted as a bounded measurement; reporting package incomplete.**
- **S-A workflow improved, annotation acceptance still withheld.** Remaining errors concern
  semantic plane identity and oversized negative regions, not just coordinate mapping.
- **S-B threshold repair accepted for the tested controls.** General production integration,
  complete saved-affine XYZ verification and M4 acceptance remain open.
- No milestone or production gate is promoted. Current next assignments are in
  [implementation-next.md](../implementation-next.md).

## Independent verification

Full workspace command:
`.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider -o addopts=`
returned **514 passed in 224.84 seconds**. Tests passing does not establish annotation truth.
Lead performed read-only code review, image inspection and in-memory probes; no source edits.
Existing uncommitted changes to three older M5 reports were preserved.

### F2: equal shares can prevent every hypothesis from completing

Reused the setup of `test_multi_cell_budget_round_robin_covers_both_shape_families_and_order`:
two cells `(0,0,0)` and `(2,0,0)`, slab/carpet targets, cube/slab/carpet palette,
two 160x120 front/back views, scale1, texture/edge weight0. Changed only candidate cap.

| Global cap | Actual candidate renders | Completed result | Other renderer work |
|---|---:|---|---:|
| 0 | 0 | no edits; both cells labelled `reached` | 12 |
| 2 | 2 | no completed hypothesis; both cells labelled `reached` | 12 |
| 4 | 4 | both select carpet with `partly_tested` status | 12 |

Cap2 allocates one render to each two-view cell. Both prefixes are discarded even though
the same budget could complete one hypothesis. Rejecting prefixes is correct; spending the
entire budget on them violates L-B's reservation requirement. At cap2 each per-cell result
also reports zero candidate renders while the aggregate correctly reports two.

At cap0 the wrapper still performs background/context/probe work for every cell (four each
in this control). This is not a violation of the candidate-only cap, but it defeats an
efficient stop and scales expensive preparation with cells that cannot be searched.
Stop promptly when no candidate work can occur; keep explicit unvisited/budget status.

The cap12 and overlapping-context controls pass. A reserved sequential architecture is
acceptable; no requirement to replace it merely to obtain a literal round-robin queue.
Reserve whole view sets, handle uneven usable-view costs, and keep comparisons in the current
scene context. Preserve the accepted evidence-retention fix.

### F2 reporting/profile: useful evidence, incomplete conclusions

[Profile JSON](M5-t2-first-cell-profile.json) hashes match the current fitter and runner.
The recorded supplied-camera first-cell experiment uses 6163 states / 38 geometries,
weight40 and cap12: preparation8.641479s, fit50.274413s, traced heap368282341 bytes.
Render40.336777s plus ownership5.511819s dominates; texture lookup is0.009203s.
These are agent-measured profiled timings, hash-audited here, not a new lead timing run.

Twelve candidate renders at three views cover only four representatives, not all38.
The phrase "complete three-view representative pass" must specify that limited coverage.
The selected mud-brick stair is still material-wrong; this is not the earlier v2 spruce result.
Do not compare cap12 profiling time to the prior cap256 run as a measured optimization speedup.

The standalone profile serializes selected properties, but fitter cell diagnostics still
conflate action with search status and omit full state properties. Per-geometry completed
view accounting, dependency versions and actual loaded-cache provenance remain incomplete.
No measured next CPU optimization proposal was supplied. Prioritize duplicate render/buffer
work, subject to context/ownership parity, before texture-classifier tuning or GPU expansion.

### F3: remaining annotation errors reproduced from source imagery

Reviewed all three v3 full composites, the original third frame, and the source-labelled
`f1_dining_area.png` crop in [the audit bundle](M4-F3-annotation-audit-2026-09-09/).
The original two frame1 wall points and frame2 bench boundary were improved.

1. **Frame1 `(1150,740)` is visible wall but resolves negative.** The dining negative rectangle
   begins at `(1075,700)` and includes wall to the left of the furniture. The agent's own
   dining-area crop plainly shows wall at the cited coordinate. A furniture bounding box
   cannot be treated as fully occluded support. Trace occupancy or mark ambiguous margins unknown.
2. **Frame3's positive target joins different planes.** Source points `(600,300)` on the left
   wall and `(1400,300)` on the far wall both resolve positive, as does `(1400,650)`.
   The room corner is visible in the original. `boundary_scoring.py` explicitly defines a
   positive as *this plane's own* visible material; matching wood texture is insufficient.
   Frame2 already excludes its far wall on precisely this basis. Use a single declared plane
   or separate target IDs; never score one plane against a union of room walls.

These label results were independently obtained through `target_masks` with the source-size
domain. The points lie away from HUD exclusions. Hash-frozen tests only prove that the JSON
continues to encode those labels; they do not make the labels correct.
Saved-C2 v3 0/3 remains a provisional annotation-dependent result. The no-candidate frame is
still a search failure, but "failure is not an annotation artifact" overstates what this audit
supports. Preserve all versions and repair semantics before proposing another cue.

### F4: supported improvement and evidence corrections

The exact earlier reproduction uses identity axes, distance1, period1, focal1,
principal point `(2.2,3.2)`, cell `[2.2,3.2] x [2.2,3.2]`, true visible fraction0.55.
Current lower/upper bounds are **0.5379726573 / 0.5612121130**, with no declaration.
Well-resolved controls give lower bounds0.545403/0.585298/0.595796/0.605245/0.994744
for true fractions55/59/60/61/100%; only61/100% declare visible.
Twenty deterministic random convex-polygon intersections (seed19, fractional intrinsics)
also bracketed independently computed convex intersection areas within1e-6. This exercises
oblique edges but is not a universal proof for arbitrary invalid polygons.

The new exact-case test silently retains `_rectangle_cell`'s default distance10, producing
a 0.1px cell despite its 1px docstring. Correct the fixture explicitly. The report's quoted
0.3482 result and 522-test total do not match this review; baseline f180ff3 was499, not513.
Document the rounding/edge-band bound quantitatively and retain translated, mixed-label and
large-tile controls. Four million samples bounds an individual mask allocation; multiple masks
and multiple bands mean neither total bytes nor total processing is capped at that number.

`scripts/verify_boundary_targets_v3.py`, cited in Sonnet's report/annotation prose, is absent.
Use the existing test as the reproducible verification command or deliver the missing utility.
Correct dated reports with explicit errata rather than silently overwriting historical results.
