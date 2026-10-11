# Current implementation work packages

> **Superseded as the active assignment list (2026-10-06).** Follow the root
> [active handoff](../AGENT_HANDOFF.md) and latest `AGENT_COMMS.md` entries for current gates,
> status, ownership and authorization. The dated work packages below are historical context only;
> do not treat their old agent assignments, milestones, test baseline, or experiment permissions as
> current instructions without a new lead directive.

Issued by GPT-6 Astra, 2026-09-09 after review five. **READY TO START; remaining work PLANNED.**
This replaces the previous F1-F4 work cycle. Read the
[acceptance report](reports/LEAD-review-2026-09-09-fifth.md) for exact reproductions.
User authorization to continue foundational research stands; no additional start approval needed.

## Shared scope and ownership

Availability update 2026-09-12: active implementers are Luna and CLI DeepSeek; Claude unavailable until Monday.
Luna executes L-D then L-E, followed by S-C image annotation work as a separately reported package.
DeepSeek DS-1 accepted narrowly; DS-2 audit delivered. DS-3 residual masks, DS-4F authority,
DS-5 buffers and DS-7 benchmark-mask policy are accepted: agent full 527; lead 7/22/26/37
focused. Exact-model routing is deferred because states do not exist at sampling.
[Worker setup](deepseek-worker.md) includes the required environment refresh; authentication works.
Both branches can proceed independently. Direct image review remains mandatory for S-C.

- Immediate goal: vanilla images to reviewable `.litematic`; existing 80% visible exact-state
  and 90% visible geometry F1 gates remain unchanged. No milestone has been promoted.
- Normal inputs are images and appearance assets. Manifests, F3, coordinates, directions,
  bounds and reference geometry remain evaluation-only or explicit advanced/developer inputs.
- Shaders, arbitrary packs, footage and generated datasets remain
  [planned long-term scope](planned-data-and-appearance-scope.md). No scraping/training now.
- Read AGENT_COMMS.md and inspect the working tree before edits. Preserve other agents' work,
  stage named files, and announce ownership/results in comms. Delivery is not acceptance.
- Luna owns shape_fit.py, its tests, M5 diagnostic/profile runners and M5 reports.
  DeepSeek owns footprint adapter tests and the named S-D audit/errata reports for DS-1.
  Luna owns new S-C annotation artifacts/tools after L-D/L-E; existing v1-v3 remain immutable.
- Shared contract changes require a concrete proposal in comms; continue independent work.
  No broad renderer rewrite, new dependency or production GPU switch is assigned.

## M5 status correction (2026-09-19)

The bounded F2 atomic scheduler repair is accepted. Its focused, adjacent, and full-suite
evidence is recorded in [the F2 report](reports/M5-F2-atomic-scheduling-2026-09-19.md).
Positive-budget probe/background discovery is separately counted and may establish usable
views; complete-view reservation gates candidate/context work, while zero or exhausted
candidate budget is inert. Remaining M5 work is production integration, diagnostic/reporting
promotion, and real-image validation. This document's older L-D/L-E wording below is retained
as historical work-package context; do not reassign the already accepted F2 implementation.

## Luna: finish bounded scheduling and decision reporting

Accepted: default silhouette-first completed-evidence retention and the cap2/3/4 control;
ordered sequential context recomputation and the overlapping-visibility control;
the bounded first-cell profile as a measurement. Preserve these improvements.

### L-D. Reserve complete hypotheses, including scarce and uneven budgets

1. Add the review's two-cell/two-view cap2 regression. Never spend one render on each cell:
   reserve both renders for one complete hypothesis, leave the other explicitly unsearched.
   This asserts completion/accounting, not guaranteed ground-truth recovery at such a small cap.
2. Keep the global cap256 default. Schedule in whole usable-view sets; do not reset per cell.
   If a task cannot fit, consider an eligible cheaper task. Retain deterministic order and
   fair representative access before additional refinement where the budget permits.
3. A versioned sequential implementation remains acceptable. Accepted edits must invalidate
   affected visibility/scores before later comparisons. Do not reuse stale scene buffers.
4. Add unequal-cost controls (one-view and two-view cells, cap1/2/3), reverse input order,
   and retain cap12/two-cell/three-shape/two-view success. Include cube-heavy material variants.
5. At cap0 return explicit budget-unvisited records without per-cell rendering or palette setup.
   On later exhaustion, skip unnecessary preparation for remaining cells. Necessary discovery
   work has its own counters; never disguise it as free candidate work.
6. Track actual attempted candidate renders separately from completed hypotheses/views.
   Per-cell attempts must sum to the aggregate, including interrupted work if any remains.
   Keep accepted air/current baselines and explicit deletion; no favorable-view-only acceptance.

### L-E. Complete diagnostics and propose the next measured CPU change

- Separate search status (unvisited/partial/complete/no usable views) from action
  (add/replace/delete/retain/abstain). Serialize full current/selected state properties.
- Report per-geometry attempted/completed/supporting views and distinct shape/material scores.
  Complete joint scores before comparison in non-silhouette mode, or explicitly mark an
  unrefined entry unavailable to that comparison; do not compare a plain score to a joint score.
  Add a nonzero-texture, interrupted-refinement control for this optional mode.
- Report exact-state and shape-family results separately in the maintained M5 diagnostic.
  Keep frozen v2 artifacts unchanged: correct-shape spruce-for-oak stair, phantoms34->34.
- Correct the profile's wording: cap12 completed four of38 three-view representatives.
  Record dependency versions and actual cache identity/hash, or explicit absence of a cache.
  Preserve the original timing/settings and distinguish corrections from a new experiment.
- Submit a small measured CPU optimization proposal using existing stage timings and render
  counts: identify duplicate geometry/ownership work, specify which outputs can be reused,
  validity keys and parity tests (RGB/depth/owner/UV/masks/selection). Do not implement a broad
  crop/scissor path yet. Explain expected saved work, without claiming an unmeasured speedup.
- No new full40-cell run. Reuse the existing profile; only repeat the at-most12-render cell
  if code changes materially affect its interpretation, labelling tracing overhead/settings.

Stop after L-D/L-E and focused/full regression evidence. M1/M2 backlog remains tracked,
with broader renderer optimization sequenced after these bounded execution repairs.

## M4 split: Luna S-C visual audit; DeepSeek S-D coverage evidence

Accepted: S-B threshold repair for reviewed controls and bounded mask allocations;
S-A coordinate-grid/contact-sheet workflow and repaired earlier table/bench examples.
Annotation correctness remains open; do not rebuild the same broad boxes with new anchors.

### S-C. Versioned plane-specific annotation audit

1. Define each target's single physical plane in image terms before tracing. Preserve v1-v3.
   Use v4 with a target ID and description; do not derive plane identity from fixture poses.
2. Frame1: trace actual dining furniture rather than its bounding rectangle. Source `(1150,740)`
   is visible wall in f1_dining_area.png but v3 marks it negative. Audit the entire rectangle
   `(1075,700)-(1820,1369)`, including surrounding wall and thin occluders; ambiguous edges unknown.
3. Frame3: distinguish the left wall from the far wall across the visible corner.
   `(600,300)` and `(1400,300)` cannot both support a single plane. Keep the intended left-wall
   target, or introduce separately named planes with separate scores and unchanged frame counts.
   Never infer same plane from repeated wood texture. Check all three frames for this error.
4. Review original and overlay side by side, plus a plane-ID outline and cropped transition
   views. Sample both sides of each plane boundary and furniture boundary, including interior
   points of large negative regions. Annotated polygons must follow evidence, not bounding boxes.
5. Freeze and hash v4 before scoring. Check encoding against independently image-judged anchors.
   Unknown regions must have a reason; report their area to prevent making hard regions disappear.
6. Optionally rescore saved C2 polygons only. Keep all three frames, top1 versus evaluator-only
   matching, no-candidate failures and every target in the report. Results stay provisional until
   the annotation passes review; no new proposer/cue/threshold sweep or SfM run this cycle.

### S-D. Reproducible coverage evidence and precise bounds contract

- Fix the named exact55% test to pass distance1/period1 explicitly: current helper defaults
  yield a0.1px cell while the docstring claims1px. Retain both sizes as separately labelled controls.
- Add analytic oblique/translated intersection controls checking lower<=truth<=upper,
  beyond merely upper>=lower. Document how vertex rounding, digital line coverage and pixel
  extent fit within the chosen boundary band, including clipping/tiling seams.
- Keep threshold55/59 rejection,61/100 acceptance and60 abstention. Preserve mixed-label,
  image-clipping and large-tile controls. No tolerance loosening to bypass decision failures.
- Clarify per-mask allocation versus peak aggregate memory and total tiled work; do not promise
  a4M total-work bound. General saved-affine XYZ verification remains separate backlog.
- Add explicit errata to the F3/F4 report:514 currently verified tests,499 prior baseline;
  exact1px reproduction now lower0.5379726573/upper0.5612121130. Retain historical claims as such.
  Replace the missing verify_boundary_targets_v3.py reference with an existing test command,
  or deliver that utility. Do not claim v3 proves the failure independent of annotation errors.

Submit S-C and S-D separately. DS-1 stops after its focused checks and report; Luna reports S-C
after its image audit. Foundation repair comes before another boundary-generation experiment;
no further permission is needed for this assigned work.

## Submission and lead acceptance

Both agents post changed paths, exact commands, controls, artifact hashes and limitations
in AGENT_COMMS.md. Coordinate full-suite timing once shared edits stabilize. Latest independently
verified baseline: **514 passed in224.84s at c6dba2c**. Lead reviews new budget controls, decision
semantics, plane-specific source-image masks and bounded coverage evidence before broader work.
DS-1 must not stage or commit. The lead changes documentation only; delegated test edits are
DeepSeek implementation work and require independent review.

## Next queue after current evidence is reviewed

- DeepSeek: DS-9 single-view camera seam, DS-11 analytic pair solver and DS-12 masked SIFT baseline
  accepted; DS-13 designs joint geometric matching after the frozen real result solved 0/6 pairs.
- Luna: measured CPU buffer reuse proposal from L-E, sequenced after bounded search and S-C;
  do not start a broad optimization or full 40-cell run without a concrete reviewed experiment.
- Astra: review DS-13 and Luna L-D/L-E/S-C, then bound the next real-pair experiment. M3 connected
  translation/scale and M1 metric backlog
  remain open; fixing local tests is not end-to-end screenshot reconstruction.
- M6 production GPU, schema migration and large datasets remain later integration work.
  Vanilla gates, <=3 minute warm-asset target and advanced-only manual geometry remain unchanged.
