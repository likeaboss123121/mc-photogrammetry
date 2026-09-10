# Current implementation work packages

Issued by GPT-6 Astra, 2026-09-09. **READY TO START; implementation planned, not complete.**
User has authorized continued foundational research. Sonnet and Luna can start independently
without another permission checkpoint. This file resolves sequencing for the next work cycle;
[review four](reports/LEAD-review-2026-09-09-fourth.md) supplies reproductions and acceptance history.

## Shared scope and working rules

- Immediate goal: a trustworthy vanilla screenshot reconstruction base. Existing gates remain
  80% visible exact-state recovery and 90% visible geometry F1 on the maintained fixture set.
- Shaders, arbitrary resource packs, uploaded pack/JAR assets and generated training datasets
  remain [long-term planned scope](planned-data-and-appearance-scope.md), not this work cycle.
- Production consumes images and assets. Reference geometry, labels, capture poses, F3 values
  and fixture bounds are evaluation-only; manual geometry requires advanced/developer opt-in.
- Read latest comms and inspect the working tree before editing. Preserve others' changes;
  stage named files. Commit only owned changes; lead-maintained coordination files may be uncommitted.
- Post an acknowledgement in AGENT_COMMS.md naming the owned work package. Report discoveries
  there while continuing independent work. Acknowledge delivery separately from lead acceptance.
- No new dependency, broad renderer rewrite, training, scraping or GPU production switch in this
  package. If an implementation needs a shared contract change, post the exact proposed change
  and continue independent work; do not silently edit another agent's owned files.

## Luna: F1 then F2 — material fitting and bounded execution

**Owned paths:** reconstruction/shape_fit.py, tests/test_shape_fit.py,
scripts/m5_matched_t2_diagnostic.py, and new M5 reports/diagnostic artifacts.
Keep existing frozen v2 data immutable. Broader renderer changes are proposals, not assigned coding.

### L-A. Preserve completed evidence when work stops

1. Add the review's stone/carpet cap 2/3/4 regression before changing selection.
2. Maintain one decision entry for every geometry whose representative completed all usable
   views. A completed material refinement updates that entry, preserving distinct shape score,
   selected material and that material's own acceptance score. No duplicate representative entry.
3. A work stop retains all completed entries. A partly rendered hypothesis never replaces them.
   Preserve air/current baselines, explicit deletion and actual supporting-frame provenance.
4. Verify cap 2/3/4 returns stone in the named control; retain grain-disagreement, missing-metadata,
   air, correct-current-state and multiview interruption controls. This does not assert that more
   compute always improves ground-truth accuracy; it asserts completed evidence is not discarded.

### L-B. Share exploration across cells without stale comparisons

1. Use a deterministic queue that visits eligible cells for representative exploration before
   spending the remaining work on material refinement. Round-robin complete-view hypotheses is
   the default design. Keep cap 256 globally; don't reset it per cell or hide ownership/setup work.
2. Reserve enough remaining candidate renders for all usable views before starting a hypothesis.
   If it cannot fit, record unresolved budget and consider another eligible task that can fit.
   Do not accept results from only the most favorable view or claim unvisited cells were tested.
3. During an exploration round, score against a fixed scene snapshot. Commit selected edits in
   deterministic order; invalidate/recheck decisions whose visibility/context changed before
   applying them. Buffer reuse requires matching scene version, camera, mask, shape/state and scale.
   An equivalent versioned sequential design is acceptable if it proves the same consistency.
4. First control: two spatially separate cells, cube/slab/carpet hypotheses, two usable views each,
   cap 12 (2 cells x 3 geometries x 2 views), common material appearance. Both cells receive all
   three representative hypotheses before any refinement; recover their different target shapes.
   Reverse input order and include cube-heavy material variants. Add an overlapping-context control
   proving cached scores are invalidated when an accepted edit affects visibility.
5. If the budget cannot cover even one hypothesis for every cell, report deterministic truncation.
   Fair scheduling does not justify unsupported geometry or claim all 38 real shapes were searched.

### L-C. Reporting and bounded profile

- Record separate search status and action, full block properties, per-cell/per-geometry completed
  view counts, air/current/selected scores and support. Report family-only and exact-state recall
  separately; preserve the measured v2 result: one material-wrong stair, no new phantom.
- Count all render categories and distinguish the candidate cap from total renderer work.
  Record actual source/cache hashes, dependencies and candidates surviving vocabulary filtering.
- Profile the frozen first real cell with at most 12 candidate renders. Keep v2 weight 40 and
  other settings; label profiling/heap overhead. Separate setup, rendering, ownership, rectification
  and texture costs. Report fit-only time separately from any measured end-to-end total.
- Reuse representative buffers only under the validity rule above. Submit measured next CPU
  optimization and parity-check design; do not implement a broad crop/scissor renderer rewrite yet.
- Stop after controls and this bounded profile. No new full 40-cell fixture run this cycle.

## Sonnet: F3 and F4 — trustworthy annotations and coverage

**Owned paths:** reconstruction/footprint_adapter.py, boundary_scoring.py if needed for validation,
associated tests, annotation audit utilities and new versioned M4 annotations/reports/artifacts.
No changes to Luna's fitter or automatic boundary proposal algorithms in this package.

### S-A. Repair the annotation workflow, not individual test anchors

1. Establish source coordinates on original 2560x1369 images. Record every audit crop's origin
   and resize scale; test forward/inverse mapping at corners and interior points before tracing.
2. Trace actual wall/table transitions on all three frames. Validate the known wall points
   (400,600)/(600,600) in frame 1 plus independently chosen points on both sides of each boundary.
   Do not derive expected labels from the mask under test. Uncertain boundaries remain unknown.
3. Deliver a source-coordinate grid, annotated crop contact sheet, full-image composite, anchor
   table with reasons, and versioned JSON with provenance. Check labels against source imagery.
4. Freeze the audited version before scoring. Preserve old artifacts. Optional rescoring uses
   saved C2 polygons only and reports 0/3-style accounting including no-candidate failures.

### S-B. Make threshold decisions mathematically defensible

1. Reproduce the true-55%-coverage case that currently reports 61.23% and exports at threshold 60%.
2. Choose consistent exact area integration or conservative lower/upper area bounds. For an
   approximation, classify only when the lower coverage bound meets the threshold; ambiguous
   near-threshold cells stay unknown. Raw supersample estimates are not confidence bounds.
3. Maintain a common area measure for cell, in-frame portion and disjoint visible/opening/
   occlusion/conflict regions. Fractions and disjoint sums stay in [0,1] within floating-point
   arithmetic tolerance. Do not conceal error by independently clamping each label.
4. Tests: 55%/59% reject, 61%/100% accept for well-resolved analytic controls; exact 60% follows
   a documented >= rule for exact integration, or abstains if an approximation cannot certify it.
   Repeat under translation, fractional scale, clipping, mixed labels and subpixel uncertainty.
5. Bound allocation and work even when an image crop alone exceeds four million pixels. Process
   bounded tiles or stop with explicit unresolved error; never allocate the entire offscreen extent.
   Do not loosen existing thresholds/tolerances to accommodate a failing decision algorithm.

Stop after the annotation audit, coverage controls and updated B diagnostic bundles. No C3,
new cue, threshold sweep or SfM restart. The accepted depth guard stays; general saved-affine
XYZ verification is still backlog and does not interrupt these two tasks.

## Submission and lead review

- Each agent runs relevant focused checks, then the full `.venv` regression once their owned
  changes are stable. Coordinate full-suite timing in comms to avoid competing expensive runs.
- Submit named commits/changed paths, commands/settings, control outputs, artifact links and
  remaining limitations. Preserve failures rather than replacing them with a selected success.
- Lead reviews F1 selection, F2 fairness/context/profile, F3 image-space annotation integrity and
  F4 threshold correctness. Then decide on the next bounded reconstruction experiment.
- Latest verified baseline is 499 passing tests at f180ff3; this document makes no new test claim.
  No source code was changed in preparing these assignments.
