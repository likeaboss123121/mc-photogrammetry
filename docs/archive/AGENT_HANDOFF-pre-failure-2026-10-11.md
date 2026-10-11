# Agent Handoff - MC Photogrammetry

Updated: 2026-10-10 after Astra reviewed the refreshed disabled camera-evaluation packet; no evaluator run. Goal tracker: last reported about 235.8 hours; no authoritative counter was found to refresh it, so remaining time is unknown.

Stop gate: at 240 tracked hours, continue only with meaningful visible reconstruction progress or an independently reconciled, image-specific implementation gap that is reasonably solvable. Synthetic-only work and diagnostics alone do not count. Product gates remain 90% visible geometry F1 and 80% exact visible-state recovery from ordinary image-derived input.

Current status: no scoreable image-only geometry, `.litematic`, or visible-accuracy gain (0 pp). AB-05 still has zero correct camera edges, one false accepted edge per arm, six disconnected frames, and no geometry F1/state/export. Camera-basin attempt01 was consumed but failed at candidate-table validation before candidate freeze; no availability result and no retry. The default image-only camera path zero-fills existing GUI-mask pixels before calibration; synthetic input-authority tests pass (17 focused tests independently rerun). The separate post-mask camera evaluator now requires source/runtime closure digests, freezes predictions before parsing evaluation truth, and passed 102 focused tests plus Astra read-only acceptance. This is still no image-scored result. Evaluator has not run. A disabled packet is prepared with nine exact pre/post-sealed inputs, 198 verified source files, and 21,130 verified runtime files; Astra approved disabled-packet readiness only; exact packet/source/runtime pins and the wrapper passed review. `tracked_hours` is still unknown and packet authorization flags remain false. No evaluator run, match gain, geometry, or `.litematic` was produced. Latest review (2026-10-10 local): successor source closure refreshed for the stale packet-task pin; its verifier passed 198/198 source pins and Astra approved disabled readiness only. Hours remain unreconciled; all successor launch flags remain false; no evaluation or reconstruction gain.

**Matcher source correction:** `SOL-JOINT-PHYSICAL-SUPPORT-01` now identifies exact pixel endpoints before support-floor calculation, RANSAC sampling, greedy selection/refit, and candidate-mode shortlist scoring. This stops SIFT orientation rows at identical pixels from inflating geometric support or hiding competitors. Configured numeric thresholds and geometric gates are unchanged. Astra approved offline02; five focused suites report 88 passing tests. The synthetic planar false-pose control now refuses as ambiguous, but that is abstention, not correct-pose recovery. Full review/receipt: `agent swarm instantiation template/runs/SOL-JOINT-PHYSICAL-SUPPORT-01.offline02.receipt.json`.

**AB-04/05 current-source fixture result (both spent; no retry):** same sealed six-image fixture, 15 pairs per detector arm. AB-04 exposed an image-0003 camera failure: pitch error -102.597 degrees (estimated -86.3 vs recorded +16.3), SO(3) error 102.679 degrees, focal error +47.35%; other five camera SO(3) errors are 0.29-2.78 degrees. The only accepted pair was image-0003/image-0004 in both arms, false by 101.770 degrees; zero correct edges and all six disconnected. AB-05 applied only the fixed-anchor calibration correction; image-0003 pitch/rotation stayed wrong and focal error worsened to +133.98% (1202.95 px vs 757.57 px); other cameras unchanged. The same pair remained false in both arms, support 133/122, still zero correct edges. Both runs were fully supervised with all six input hashes matching pre/post. No geometry F1/state/export. Reports: `docs/reports/spatial-sift-matcher-ab-04-2026-10-11/pair-ab.scalar.json` and `docs/reports/spatial-sift-matcher-ab-05-2026-10-11/pair-ab.scalar.json`; packets one-shot spent; no retries.

Cross-view plane-continuity diagnostic: repaired same-session join and coordinate mapping passed focused tests (347); collector/continuity tests pass (75). Run01 (spent, never retry) stopped at incomplete correspondences before modeling; Run02 resolved the collector cause on the sealed six-image fixture: all 15 pairs refused (13 `low_spatial_spread`, 2 `cheirality_failed`); three refused pairs had >64 rows and were correctly census-counted as truncated, while 12 captures were complete. Thus the >64 cap caused run01's incomplete report but did not cause the geometry failure. Run02's result is superseded for current matcher behavior by AB-03 above. See `docs/reports/cross-view-plane-consistency-02-2026-10-10/{worker.scalar.json,ingress.scalar.json,supervisor.scalar.json}` and `agent swarm instantiation template/runs/SOL-CROSSVIEW-PLANE-DIAGNOSTIC-RUN-02.*`; its packet is spent and never retry.

Latest partial-search run: 18 readings, 4,086 candidates, 11 eligible assignments/1,750 branches, then 524,288 relation visits exhausted; zero physical assignments/evaluations, C1 not reached, no geometry/export. Partial observer retained 4 of 128 exclusions (124 omitted); capped examples cannot prove factor concentration. Seed census has 19/36 available slots and 17 absent. Correct first-fail attribution for absent slots: 12 normal-score, 4 side-match, 1 tangent-support, 0 margin. Nine of 12 normal-score misses have no same-atom tangent co-occurrence; three have weak positive scores. These are diagnostic classifications, not proof a visible plane was missed. Sol found no proven predicate defect; 17 focused tests passed. Do not claim match gain.

A separate G7 support-score invocation already ran once on Oct 10 05:11-05:19 UTC, despite stale task wording. Preserve it as historical/consumed: 18 readings accepted; 19/36 frame-axis-side seed slots available; graph returned no eligible assignment (113 kept, 3,973 same-axis mismatches), `g_unavailable`, no `.litematic`. Four of 108 runtime pins differ from today's accepted checkout, so it is not current-source validation. Its output monitor SHA is `C659B7FD20D390A8838B15736632A4F089163DE350F0A8FF7A1C6B1D6A960195`.

Latest calibration falsifier (2026-10-10): exact duplicate-line removal fixed a synthetic multiplicity ranking failure, but changed the existing roll recovery and was rejected. A bounded 162-call orientation refinement restored +3-degree roll while failing mirrored -3-degree controls; Sol reverted it. Exact AB-05 source/test pins were independently reverified after restoration. No AB-06/image run, no real-image match gain, and no source promotion. AB-04/05 still show image-0003 camera failure and zero correct matcher edges; the remaining problem is not yet demonstrated as a reasonably solvable image-specific gap. Reassess the tracked-hour counter before any further experiment. At 240 hours, stop unless visible reconstruction improves or new evidence isolates a concrete, reasonably solvable image-specific implementation gap. Preserve image-derived default mode, advanced-only manual metadata, and all matcher/promotion gates.

Keep detailed chronology in linked reports and append-only `AGENT_COMMS.md`; update this current block instead of adding more chronology. Luna routes/verifies, Sol implements, Astra reviews, and DeepSeek handles explicitly scoped grunt work.
## 1. Purpose and binding user decisions

- Recover Minecraft builds as reviewable `.litematic` files; immediate proof of concept uses vanilla screenshots.
- Production must estimate cameras, FOV, relative block scale, and scene extent from image evidence.
  Supplied manifests, F3/debug values, coordinates, directions, bounds and references are refinement/evaluation aids only.
  Normal mode must neither require nor accept manual camera/geometry metadata, including F3 OCR or sidecar auto-loading.
  Gate such inputs behind explicit advanced/developer opt-in across UI, CLI and project loading; label those runs assisted.
- Support buildings and rooms first, then irregular terrain and geodes. Keep single-view input useful.
- Long-term scope includes shaders and arbitrary resource packs in screenshots/footage; vanilla crops/HUD are the immediate proof of concept.
  Support known pack/JAR upload or uncertain closest-vanilla matching; see [planned scope/data pipeline](docs/planned-data-and-appearance-scope.md).
- Use a best estimate with visible uncertainty, masks and block/state review; manual pose/alignment, length and plane constraints require advanced/developer mode.
- Prioritize AMD RX 9070 XT acceleration; use CPU for correctness tests and PyCOLMAP, which cannot use this GPU through CUDA.
- First production gate: **80% visible exact-state recovery and 90% visible geometry F1**,
  independently on Template 2, Template 6, and frozen small-view Template 3 subsets (section 5).
- Preserve calibrated diagnostics and existing litematic/world ZIP exports; further world-export work is not a prerequisite.
- Do not fill unsupported hidden volumes. Distinguish observed blocks from inferred completions.
- Reference-derived palettes/calibration stay evaluation-only; user pack/JAR appearance assets are normal inputs, not manual camera constraints.

### Superseded claims and their replacements

| Historical claim or instruction | Current rule |
|---|---|
| Camera recovery is a later phase; manifests are required inputs | Image-derived cameras and bounds are core requirements. |
| Free-space carving is the next strategy | Fit lattice-aligned surfaces; use local stereo later for irregular surfaces. |
| A second view establishes absolute scale | Scale requires identifiable block/texture spacing or a user constraint. |
| Conditional Template 2 accuracy is the production gate | Use the visible metrics and multi-fixture gates in section 5. |
| Geometry works outside Template 4 | Shell generation is a general architectural defect. |
| Lighting correction is solved | Revalidate lighting on corrected geometry and supported evidence. |
| Partial blocks are absent or unimplemented | Models and opt-in fitting exist; proposal and rendering limitations remain. |
| World export is missing or blocked | Both litematic and world ZIP exporters are implemented. |
| Projection and occlusion are fixed everywhere | SfM, sampling, and residual rendering retain specific defects. |
| New Python/GPU dependencies are installed | The existing environment and the proposed replacement are separate below. |

## 2. Verified current state
**Current accepted components**

- Normal still-image entry, schema-2 session persistence, image-derived lattice/provenance, geometry orchestration, promotion/output, and `.litematic` delivery have accepted seams; R3 wires the image-derived path.
- Litematic and world ZIP exporters already exist. The `.litematic` delivery seam only emits accepted exact-state eligible geometry; it does not solve geometry or appearance.
- Dense-upright SIFT experiment retained features but zero queries passed strict ratio0.75; the spatial-context A/B work is still synthetic-only.
- The 90% visible geometry F1 and 80% exact visible-state gates are unchanged. No current full-suite result or image-only score is claimed.

**Remaining architectural defects**

- The post-index G7 revalidation attempt (2026-10-10) launched once but the supervisor stopped at process_snapshot_failed after 142 samples/143.166s (CimException; native/status code 1), before child output. Drain/cleanup completed; accounting was incomplete; six input path/size/SHA records matched before/after; stdout/stderr empty; no diagnostic, mcscan, or litematic. Its one-shot authorization is spent; no retry. Monitor SHA994DD1D09527E4CDE803E6F0CAAD873AAC10C07DE4FEFA4818F1800E28E50FF8. This failed attempt is superseded by the complete post-index selector and seed-predicate reports in the latest checkpoint below; do not use it as current selector status.
- **Native CIM synthetic exercise (2026-10-10):** one authorized invocation stopped during its first sample: 1 attempted/0 completed, 1.497s, no sampled memory, child exit -1, cleanup/drain complete, accounting incomplete, no sample rows or stream bytes. Terminal SHA602D1D441706AAD19EC9F375D9D0C24CF61837DB09766428E0A52ED3735E8C72; watchdog SHA E7AEA108FD927D5399B984DA036BA40E52FC5ECB7F438478F30442FE8B49CE53 reported controller exit/terminal commit in 0.944s. HRESULT/native/status null, so no native CIM result or reliability conclusion. One-shot spent; no retry. Independent read-only reviews agree the stop was between attempt increment and first recorded memory sample. Candidate path: imported sampler/snapshot processing, row/identity accounting, or initial memory observation. Missing evidence: phase/type telemetry and a fake integration control combining the actual lifecycle with the imported sampler and snapshot wrapper; no edits, native query, process inspection, or retry authorized by that exercise.
- **Historical POST-INDEX-02 launch preparation (superseded):** Astra found and Sol corrected a monitor-only legacy supervisor-slot mapping to the spent POST-INDEX-01 root; amendment01 bound the fresh outputs and disjoint root. That conditional launch later led to the consumed post-index diagnostic and then the accepted reading-attrition and seed-predicate runs recorded in the latest checkpoint. The old `No run yet` and pending Astra language applied only before those runs and grants no current authorization. Its one-run/no-retry rule remains historical; the seed-predicate authorization is spent.
- **G7 diagnostic test pin reconciled (2026-10-09):** the test retained the pre-index endpoint source hash while the diagnostic and current source were post-index. Sol changed only the 64-byte SHA literal in `tests/test_diagnose_g7_selection_failure.py`; all 90 focused tests passed (5.29s). Astra accepted the diff, receipt, and pins. No production source or image behavior changed; this restores the offline harness gate but does not establish post-index performance on the six-image fixture. Artifacts: `agent swarm instantiation template/runs/SOL-G7-TEST-PIN-RECONCILIATION-01.*` (test SHA `11E05358DE943CFE706B2DC4FD452FAE3C2F68153139E32C042DA14F67BC75F0`, receipt SHA `80F70371981FEDEF698825DC38F5DC41CDE9406FAB038637B9D6B9BD5F90A9CC`).
- Existing unit-camera pose-graph translations have unit-edge gauge, not metric camera centers. Scale, physical association, and endpoint consistency remain unresolved from ordinary images.
- `supported._edge_profiles` has a separate unsigned-difference defect; do not assume adjacency evidence is correct until its scoped repair is accepted.

**Environment and execution**

- Workspace is Windows/PowerShell; verify interpreter/package/GPU state from the dated environment reports before experiments. The September dependency inventory is historical.
- RX9070XT acceleration is a priority, but this does not prove an active end-to-end GPU path. WSL/Prism/Minecraft runtime and capture remain separately gated; no client launch is authorized by current tasks.
- Normal image ingress defaults to12 and caps at32 frames. The global8/8 attempt quota is revoked; retain task-local work, memory, evidence, and authorization limits.
## 3. Approved implementation plan

**Historical plan snapshot:** the M1-M7 sequencing below is the original approved plan, not a current milestone completion ledger. Later architecture work is recorded in `agent swarm instantiation template/project/STATUS.md`, active tasks and their run receipts. Keep the user's production scope and gates above unchanged.

### M1. Measurements and contracts — Luna inherits backlog; **acceptance gaps open**

- Snapshot current source/environment and establish excluded local version control before source changes.
- Upgrade the maintained benchmark: explicit reference selection, visible metrics, per-state errors,
  observed/inferred counts, phase timing, peak memory, input hashes, dependencies, JAR/cache provenance,
  and complete effective settings. No arbitrary `next(glob("*.litematic"))` reference selection.
- Separate image-only, supplied-camera diagnostic, and reference-assisted diagnostic modes.
  Only image-only input qualifies for production; report user-assisted runs separately.
- Freeze corrected-metric results; retain original historical denominators. Denominator corrections are not algorithm improvements.
- Freeze section 4 contracts before parallel integration. Keep calibrated commands as regression tools.
- Deliverable: reproducible baseline reports, metric definitions, and agreed interfaces. Evidence:
  [M1 baseline](docs/m1-baseline.md), reports under `exports/baselines/`, historical 252-test snapshot.
- Current closure: review B3/B4 — valid masks, full-state alignment/breakdowns, complete provenance,
  and matched baseline/current runs. Existing masked/unmasked diagnostics remain provisional.

### M2. Consistent projection and rendering — Luna inherits backlog; **REOPENED after lead review**

- Share camera transforms and projection utilities. Fix SfM handedness and horizontal/vertical FOV drift.
- Represent intrinsics, principal point, and image dimensions explicitly; transform them with crop/resize.
- Build a shared reconstruction renderer with clipped model polygons, per-pixel depth, exact-state
  geometry, UVs, pixel ownership, alpha masks, and element-local rotations. Boxes are acceleration data.
- Cull against actual partial geometry, not occupied cells. Route sampling, residuals, and shape scoring
  through the renderer; sample surviving pixels instead of discarding faces at HUD/image boundaries.
- Preserve independent analytic ray-intersection tests to detect bugs shared by renderer consumers.
- Deliverable: trustworthy face-to-pixel correspondence and geometry-hypothesis rendering. Earlier crash,
  culling and optics fixes are accepted. Current B1/B2 block closure: weighted valid samples and a fixed
  candidate-independent scoring region, plus exact-model sampling. Add independent reproductions.
- Acceptance: matched T2/T6 baseline/candidate metrics and relevant tests. Dependency: M1 contracts; correctness precedes GPU work.

### M3. Image-derived cameras and lattice — Sonnet; **RESEARCH DELIVERED; recovery incomplete**

- Approved scope remains image-derived orientation/intrinsics, lattice period/phase and competing scale
  hypotheses; identifiable block/texture evidence or a user length resolves scale. Do not assume scale.
- Bounded Template 3 experiment, multiscale vanishing directions, optical refinement and contrast cues
  are implemented. Contrast weighting did not improve calibration and remains off by default.
- Camera adapter/roll fixes passed lead replay. Session fitting preserves unknown positions and scale;
  fit inlier fraction is a diagnostic, not correctness confidence. Mixed crop/session handling stays explicit.
- Delivered maintained calibration covers all Template 2/6 views and `t3-sparse-8` (plus Template 4).
  Complete maintained `t3-local-3`/`t3-local-5` evidence; no selection by reconstruction success.
- Lattice period/phase works in some oracle-isolated regions. Known-length plane distance is implemented;
  half/double-distance aliases and inconsistent constraints remain review items. Preserve alternatives.
- CPU PyCOLMAP extraction/matching/two-view experiments were evaluated and stopped after the declared
  configurations failed. That stopping decision is accepted; generic matching is not to be blindly retuned.
- Connected pose recovery, incremental mapping and bundle adjustment remain unfulfilled production
  requirements. A failed pairwise experiment does not supply them. Never replace them with fixture poses.
- Gate remains orientation <=0.5°, focal <=2%, independently measured median seam displacement <=2 px
  at a 1920-pixel long edge. Reproduced uncropped diagnostics use reference translation explicitly.
- B5-B7 fixes are agent-reported: COLMAP pose scoring, cropped seam integrity and invalid length constraints.
  Lead acceptance and complete reproducible input provenance remain required; M4 is Sonnet's immediate assignment.
- Entry points: `scripts/calibrate_frames.py`, `relative_pose.py`, `overlay_mask.py`, `seam_landmarks.py`.
  [Pose report](docs/reports/M3-relative-pose-2026-09-08.md), [seam report](docs/reports/M3-seam-gate-2026-09-08.md).
- M4 may build useful supported planes in a local frame from image evidence and optional stated length;
  unresolved relative placement between views must remain explicit. M1 types/M2 rendering are dependencies.

### M4. Supported surfaces replace shells — Sonnet; **C2 DELIVERED; MASK/COVERAGE CORRECTIONS OPEN**

- Represent each plane by axis, integer position, supported cell footprint, observations, and uncertainty.
  Footprints support holes and disconnected patches; rectangular extent is not proof of occupancy.
- Jointly fit translations, lattice phase, plane positions/extents using seams, reprojection, and occlusion edges.
  Refinement requires independent evidence; the current one-distance-per-view link model only propagates offsets.
- Construct cells from supported surfaces. Derive export bounds from reconstruction, never reference bounds.
- Alternate continuous camera refinement and discrete plane/lattice proposals; evaluate coarse hypotheses,
  then refine promising ones at higher resolution. Retaining the current scene is always a candidate.
- Split, extend, shift, or remove surfaces where spatially consistent residuals support geometry changes.
- Preserve alternatives for pure rotation, repeated textures, or disconnected views; do not invent placement.
  Single-view planar estimates remain useful, with unsupported depth relationships marked for review.
- Deliverable: image-only three-view Template 2 reconstruction, then Template 6 and small Template 3 subsets.
- [Distance/support experiments](docs/reports/M4-supported-plane-2026-09-08.md) retain historical measurements.
  Agent reports tie-corrected AUCs 0.528/0.502; geometry labels still need validation. 168/169 judged cells came from one view.
- [Registration](docs/reports/M4-registration-2026-09-08.md) propagates simulated length/identity/quadrant inputs
  within 0.004 blocks on resolved axes; one camera's Z remains unknown. This is assisted diagnostic evidence.
- [Lead recovery directive](docs/reports/M4-LEAD-direction-2026-09-08.md) supersedes the hold and impossibility
  claims in [the status report](docs/reports/M4-status-2026-09-08.md): weak seams do not prove holes;
  global compass orientation is a gauge choice, while relative quarter-turns may remain competing hypotheses.
- [Latest review](docs/reports/LEAD-review-2026-09-09-fourth.md): negative composition, large support gaps and depth rejection improved.
  F3: table annotation still removes visible wall; audit crop/resize-to-source mapping and every boundary before another experiment.
- F4: true 55% coverage reports 61.2% and exports at threshold 60%; use consistent area or conservative error-bounded decisions.
  Full saved-transform XYZ reprojection remains backlog beyond the accepted depth guard. No C3 assigned; preserve C/C2 artifacts.
- No extra approval for F3/F4 controls. Keep all three frames in failure counts and top-1 separate from evaluator-only matching.
  >=90% region precision/>=50% recall on two targets remains local only; production gates and advanced-mode boundaries are unchanged.

### M5. Materials and partial blocks — Luna; **OPEN — integration corrections required**

- Apply face-aware texture scoring and multiview fusion to recovered surfaces.
- Fit lighting/exposure only from supported visible evidence; centre-pixel fallback cannot establish occupancy
  or dominate palette pruning. Revalidate existing lighting experiments on the corrected evidence.
- Preserve alternatives for rare materials. Scene-level support may rank but must not simply eliminate them.
- Propose local geometry from silhouettes, depth boundaries, residuals, and pixels unexplained by any surface.
- Consider air, cubes, slabs, stairs, carpets, fences, walls, panes, bars, and torches where relevant.
- Decide shape from geometry/visibility and material from rectified textures; keep their verdicts distinct.
- Recompute occlusion after accepted edits. Use block-specific neighbour rules for derived state properties;
  properties such as `open` are not universally derived. Mark visually unknowable properties as uncertain.
- Deliver increased visible exact matches and partial recall without hiding lost cells; depend on M4/M2, preserving isolated diagnostics.
- [Latest M5 review](docs/reports/LEAD-review-2026-09-09-fourth.md): metadata/quantile/harness fixes accepted; complete-palette v2 delivered.
  One correct-shape spruce-for-oak stair, family 0->1/69, exact unchanged, phantoms 34->34. F1 cap drops scored shapes; F2 scheduling/profile next.

### M6. GPU acceleration and integration — Luna; Sonnet coordinates stages; **GROUNDWORK REPAIRS ACCEPTED; integration planned**

- Isolated wgpu environment/scoring/transfer probes exist; production D3D12/Vulkan rendering/batched scoring remains planned, with CPU reference.
- Cache replacement accounting, snapshot ownership and dispatch corrections accepted in [follow-up review](docs/reports/LEAD-review-2026-09-09-followup.md); transfer parity is not renderer correctness.
- Cache pyramids/features/geometry/descriptors/accepted buffers; use affected-tile rendering later. M6 production wiring awaits M2/M4/M5 acceptance.
- Bound work by observed surfaces/active hypotheses. Avoid frame-by-volume enumeration and loading every
  full-resolution image unconditionally; record memory use and actual adapter/backend.
- Make CLI and desktop call one pipeline on all selected images. Identify duplicates without dropping useful
  viewpoints merely because repetitive Minecraft textures yield similar hashes.
- Provide overlays for cameras, lattice, uncertain cells, excluded frames, and unresolved components.
  Support masks/block-state review; expose manual lengths, poses/directions and geometric alignment only in advanced/developer mode.
- Save `.mcscan` before export; preserve cancellation/resume/partial results. Round-trip litematics with asset-consistent data versions.
- Target: image-only Template 2 <=3 minutes on this machine with warm assets; report cold-start separately.
  This is a provisional performance target, not a measured result; quality gates take precedence.
- Dependency: M1 contracts and M2 correctness for acceleration; M4/M5 for production integration.

### M7. Irregular surfaces — Sonnet; Luna supplies acceleration/evaluation; **PLANNED**

- Extend the lattice to smaller connected patches on geodes and terrain after the buildings milestone.
- Where planar evidence is weak, use confidence-filtered stereo depth to propose local lattice faces and
  verify them with the shared renderer. Preserve ambiguity when limited views cannot establish depth.
- Do not restart rejected global photo-consistency carving. Add cutouts, biome tinting, and transparency
  handling relevant to these scenes; extend material evidence without changing production/reference isolation.
- Promote terrain support only after Templates 4 and 5 meet the same quality gates.
  Use Template 1 for large-scene scaling validation.
- Dependency: buildings promotion through M6; deliverable is separately validated irregular-scene support.

## 4. Shared interfaces and persistence — initial contracts implemented; integration/migration planned

| Contract | Required responsibility |
|---|---|
| `FrameObservation` | Stable frame ID, dimensions, crop/resize transform, mask, quality, cached features. |
| `CameraSolution` | Per-frame intrinsics/pose, components, residuals, confidence, unresolved calibration/scale. |
| `LatticeScene` | Coordinate frame, scale evidence, plane footprints, candidates, alternatives, observations. |
| `RenderResult` | Colour, depth, exact-state/face ownership, UVs, valid-pixel coverage. |
| `ReconstructionResult` | Block volume, alternatives, diagnostics, review items, timings, completion status. |

- `CameraSolver.solve` now returns `CameraSolution`; `CameraPose.position` can be `None`.
  Sonnet preserves measured orientation/roll/principal point in adapters; Luna owns shared camera/types.
- Freeze eye/feet, unresolved scale/components, typed plane alternatives, face/state ownership and usable-pixel weights (R5/R6).
- One production entry point accepts frames, assets, settings, review edits, progress/cancellation; gate explicit geometric constraints by mode.
  References/fixture manifests stay in evaluation adapters; default mode cannot silently consume diagnostic metadata or saved advanced constraints.
- Advance `.mcscan` to schema 2 for cameras, lattice, constraints, settings, evidence, and review status.
  Load schema 1 with explicit unavailable reconstruction metadata; preserve existing volumes.
- Reuse observed/inferred `BlockObservation` kinds. Confidence is uncalibrated until validation establishes
  reliability; keep fit statistics distinct. Persist alternatives without inserting every alternative into export.
- Export the best supported estimate without hidden filling; preserve asset/project/data-version/state consistency end to end.

## 5. Evaluation, tests, and promotion

### Metric definitions

Let `R` be all reference coordinates, `V` reference coordinates visible through usable image pixels,
and `C` exported coordinates, after the allowed global evaluation alignment.

- Visible exact-state recovery = correctly positioned and labelled reference blocks in `V` divided by `|V|`.
- Visible geometry recall = `|C intersect V| / |V|`.
- Visible geometry precision = `|C intersect V| / (|C intersect V| + |C minus R|)`.
- Visible geometry F1 = harmonic mean of this precision and recall. Correct hidden blocks receive separate
  inference credit; they neither inflate visible recall nor count as phantoms.
- Also report conditional common-coordinate state accuracy, whole-reference exact recovery, phantom counts,
  per-state errors, observed/inferred counts, and accuracy by projected pixel area.
- Primary visibility includes at least one pixel; report larger thresholds separately rather than dropping
  hard cells. Freeze evaluation masks independently so reconstruction cannot mask away its own errors.
- Permit one global integer translation and lattice-axis rotation; transform directional states consistently.
  Report alignment and direct results explicitly. No reflection or scale fitting may hide wrong block counts.
- Exact state uses the existing normalized `BlockState` equality. Ambiguous or invisible properties are still
  counted honestly in raw exact-state metrics, with their uncertainty reported separately.
- Seam gate: use held-out, independently fixed landmarks/segments and actual displacement normalized by
  `1920/max(width,height)`. Fit-line residuals and long-segment strata are separate diagnostics. The 2-pixel
  gate is unchanged; orientation-only evaluation may hold reference translation fixed with explicit labelling.

### Required tests and scenarios

- Projection/triangulation agreement across aspect ratios, crops, principal-point offsets, yaw, pitch, roll.
- Near-plane crossing, depth ordering, different stair/slab states, partial-neighbour occlusion,
  alpha cutouts, rotated elements, and image-edge clipping.
- Single-view ambiguity, repetitive textures, disconnected views, pure rotation, missing seams,
  unknown FOV, and inconsistent user constraints.
- Test HUD removal without losing unobscured pixels, mixed resolution/exposure, and independent analytic GPU/CPU agreement.
- Test default-mode rejection of manual/imported camera metadata, no F3/OCR dependence, explicit advanced opt-in, and fixture/reference isolation.
- Schema migration, cancellation/resume, partial-result persistence, exact litematic round trips.

### Promotion rules

- Require >=80% visible exact-state recovery and >=90% visible geometry F1 separately on Template 2,
  Template 6, and frozen small-view Template 3 subsets. Report automatic and user-assisted runs separately.
- Plan community-build/world capture datasets after stable ingestion/evaluation: varied cameras, gamma, shaders/packs; split by build before variants.
- Every promotion needs settings, before/after artifacts, overlays, error attribution, and relevant passing tests.
- Re-evaluate baseline and candidate together after renderer or metric changes. Preserve calibrated-path
  regression reports under identical settings; investigate regressions rather than replacing their denominator.
- New terrain promotion applies the same gates to Templates 4/5; Template 1 measures scaling.
- Purely historical results below do not demonstrate that any new production gate has passed.

## 6. Immediate assignments and communication

| Agent | Responsibility | Next directive — assigned; acknowledge in comms |
|---|---|---|
| Luna | Project manager; user communication, task decomposition, routing, scope locks, handoff and independent verification. | Read latest comms; keep normal-mode metadata image-derived; do not implement product code. |
| Sol | Developer; bounded implementation under explicit routed tasks. | Native-CIM first-sample diagnosis complete (read-only). Astra accepted the revised fake-only test design and its source pins. Sol added four fake-only lifecycle/sampler integration cases; Astra reviewed the exact diff and authorized one isolated run. It passed 4/4 (83 existing cases deselected) in 4.58s; Astra accepted the exact execution receipt/result. The fake valid path produced 300 durable samples; empty/mismatch/throw paths preserved zero durable acknowledgements and typed fail-closed outcomes. No native CIM queries, real process inspections, production source changes, or image runs occurred. This validates harness composition only, not native CIM reliability or reconstruction. It must compose actual lifecycle + imported sampler/snapshot in a test copy, replace exactly one GetProcessById binding, override Stop-PacketTree after import, and keep all process factories fake. Verify valid (300 durable samples, 240s fake natural exit + tail), empty/mismatched rows (1 in-memory, 0 durable), and throwing provider (0 samples, original error preserved); no production code, native/process/CIM execution, retries, or image packet is authorized. See AGENT_COMMS.md. |
| Astra | Consultant; read-only architecture and high-risk acceptance. | Review frozen pins and resolve architecture blockers; no implementation. |
| DeepSeek | Scoped low-cost developer-equivalent or read-only auditor, as routed. | Follow `docs/deepseek-worker.md`; no broad refactors or unscoped runtime work. |

- [Ready-to-start work packages](docs/implementation-next.md) define implementation order/checks; [review four](docs/reports/LEAD-review-2026-09-09-fourth.md) records evidence/acceptance.
  [B1-B7 review](docs/reports/LEAD-review-2026-09-08.md) remains the acceptance backlog; it does not interrupt Luna's M5.
- Sonnet owns M4 modules/scripts; Luna owns M5 palette/renderer/shape-fitting integration. Propose shared
  type changes in comms before edits. GPU implementation awaits M2 acceptance; stage named paths, never `git add -A`.
- Use [AGENT_COMMS.md](AGENT_COMMS.md) with model, timestamp/timezone, recipient, task and evidence/blocker; read before work and at milestones.
- Record evidence-backed status here; keep detailed experiments, settings, results and stopping reasons in linked reports.
  Failed attempts retain evidence; do not silently convert a rejected approach into a default.

## 7. Essential history, artifacts, and navigation

### Lessons to retain

- Six global carving/consistency approaches failed to transfer; repetitive textures and few views are central.
- Texture-only or global-colour tuning before geometric correspondence is correct has repeatedly plateaued.
- Dropping fallback cells increased conditional accuracy while losing correct blocks; report counts and recall.
- Shape and material need different evidence. More partial candidates or more views alone did not solve shape.
- Palette pruning can erase the correct state; more confident wrong evidence is not an improvement.
- Lighting percentile/damping wins on crops did not reliably transfer to whole scenes; revalidate after geometry.
- Shared-renderer self-comparison can hide shared defects. Independent occlusion/projection checks matter.
- Name-token material bonuses, blind face-shrink sweeps, stronger adjacency, and global-light retuning require
  a new falsifiable hypothesis before repetition. Preserve the useful code, not obsolete tuning instructions.

### Historical measurements — retain original labels

- [Archived Template 2 report](exports/baselines/t2-final.json): 814 candidate blocks, 780 common coordinates,
  668 exact matches; conditional state accuracy 85.64%, whole-reference exact recovery 78.40%,
  coordinate F1 93.64%, archived surface F1 80.44%, reconstruction 19.24 seconds, palette size 726.
  These are that artifact's metrics/settings, not the new camera-visible production scores.
- [Visibility summary](theoretical_max/summary.json) records 669 visible of 852 reference blocks for Template 2,
  570/6,225 for Template 4, 2,031/2,376 for Template 6, 29,406/31,862 for Template 5,
  16,356/66,414 for Template 3, and 14,191/566,680 for Template 1.
  These archived visibility counts precede the planned independently frozen usable-pixel masks.
- A separate historical analysis attributed 535 visible exact matches out of 669 to the Template 2 run,
  plus 133 correct inferred/unseen blocks. Treat this as an archived claim pending maintained-harness reproduction.
- Re-score historical visibility ceilings with corrected alpha/models; they are neither production inputs nor current denominators.

### Maintained entry points and source map

- Current experiments: [M3 calibration](docs/reports/M3-camera-calibration-2026-09-07.md),
  [M3 generalisation/lattice](docs/reports/M3-generalisation-and-lattice-2026-09-07.md),
  [frozen subsets](docs/reports/M3-frozen-subsets.json), [M4 cues](docs/reports/M4-planar-region-proposal-2026-09-07.md).
  These are dated measurements; lead-review corrections and current status above take precedence.
- [Benchmark](scripts/benchmark_templates.py), [CLI](src/mc_photogrammetry/cli.py),
  [desktop workflow](src/mc_photogrammetry/ui/main_window.py).
- [Camera](src/mc_photogrammetry/reconstruction/camera.py), [calibrated SfM](src/mc_photogrammetry/reconstruction/sfm.py),
  [shell/classifier pipeline](src/mc_photogrammetry/reconstruction/raycast.py), [face sampling](src/mc_photogrammetry/reconstruction/face_sampling.py).
- [Model atlas](src/mc_photogrammetry/blocks/model_atlas.py), [texture provider](src/mc_photogrammetry/blocks/textures.py),
  [residuals](src/mc_photogrammetry/reconstruction/residual.py), [shape fitting](src/mc_photogrammetry/reconstruction/shape_fit.py).
- [Visibility evaluator](src/mc_photogrammetry/export/visibility.py), [attribution renderer](src/mc_photogrammetry/export/attribution.py),
  [domain models](src/mc_photogrammetry/domain/models.py), [contracts](src/mc_photogrammetry/pipeline/contracts.py).
- [Project storage](src/mc_photogrammetry/project/store.py), [litematic export](src/mc_photogrammetry/export/litematic.py),
  [world ZIP export](src/mc_photogrammetry/export/anvil.py), [tests](tests), [project configuration](pyproject.toml).
- [README](README.md) has historical status prose; use this handoff for current priorities and capabilities.
- [Historical handoff](docs/archive/AGENT_HANDOFF-2026-09-07.md) preserves experiments; its scratchpad tools are unavailable.


## Most recent image-scored checkpoint: seed-predicate attrition diagnostic (2026-10-10)

- Astra accepted the exact-pinned postrun evidence. Final packet `agent swarm instantiation template/runs/SOL-G7-SEED-PREDICATE-DIAGNOSTIC-01.amendment02.packet.json` SHA256 `52B7FCCB667581A3DB0E4EA07B13EB130D8360A7E8D92F084CE7D3334DAB74E1`; freeze `204103CD2C8EE42689072DE63917A4D6D4AF7D3CD145212CE0C1DEAE603B6CC2`. Exactly one supervised launch completed exit0 in 433.1537s; monitor SHA256 `2323E534052D2F91118DD5310EA236A78E17371046D80E6E710534767E617912`, 434 samples, peak 598,532,096 B, complete accounting/drain, zero overshoot, and all six ordered input seals unchanged. No retry; authorization spent.
- Report `exports/diagnostics/g7-post-index-train-station-02-seed-predicate-01/g7-seed-predicate-01.selection-diagnostic.json` SHA256 `A6128386948F59DD8F01D4FCEFC2E861B06DA77ED98FF8E1CBC1C677B939B082`; report digest `e8ba6c16751c2032e9eb166ce010a29f45c7a9d55bd6170c64ef0c9f6863e33e`. Observer schema-v2 complete, no omissions, 36/36 rows reconciled, all ten checks true. Totals: 19 available / 17 absent seeds; 19 proposals (18 returned, 1 area rejected); 18 readings accepted.
- Across 36 frame/axis/side rows, sequential seed memberships: side-matched 22,512 -> score 183 -> margin 174 -> support 164. These are repeated memberships across hypotheses, not unique atoms. For image-0000, axis0 sides have 1,223/0 side matches and zero score passes; axis2 sides have 560/663 side matches and zero score passes. Maximum normal scores are 0.151583, null, 0, and 0.188992; the unchanged threshold is 0.26. Margin/support cannot explain those four absent rows because none reach them. This establishes the immediate rejection stage only, not the visual cause, physical infeasibility, or general accuracy.
- The same run confirms 4,086 candidates, 18 readings, 81 signatures, zero selected candidates; 1,024/1,024 selector assignments first fail `axis_0_disconnected`, 251,659/262,144 relation visits, zero geometry evaluations. Production is `geometry_unavailable/g_unavailable`; no `.litematic`. `.mcscan` was not inspected.
- Follow-up 2026-10-10: Sol added `test_authored_pixels_real_detector_to_support_normal_side_and_seed` covering all six axis/side gates through real LSD detection, support allocation, score, and growth; local focused verification passed (7 tests). No axis/side/score/seed wiring defect found; this synthetic control does not establish image sufficiency. Existing `G7-UNIT-WEIGHT-SPARSE-CF-01` already recorded a complete same-pass structural-vs-unit sparse-support ledger on the six-image fixture, but only scalar aggregates; its graph had zero eligible assignments, zero geometry evaluations, and no litematic. The separate unit full-delivery run also refused with no geometry and used different frozen source pins. Neither artifact isolates a reasonably solvable image-specific gap. Astra/Luna review: NO-GO under the 240-hour gate. DeepSeek inventoried all nine `exports/diagnostics/g7*/**/*.selection-diagnostic.json` reports: none stores replayable segment IDs/endpoints/lengths/weights/associations or per-atom support arrays. `paired_segment_support`, `support_score_attrition`, and `unit_weight_sparse_support` persist 36 frame-axis-side aggregate rows, hashes and counts only; candidate rows are digest-bound but omitted. All nine have zero geometry evaluations and no litematic. Thus no already-captured, source-bound replay record exists in this diagnostic family. Do not infer a production fix from aggregate support sensitivity, inspect `.mcscan` payloads, or run another image capture without new evidence/review. Tracked-hour counter remains last reported at 235.8; not independently refreshed.
