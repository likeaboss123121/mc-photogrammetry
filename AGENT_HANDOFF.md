# Agent Handoff — MC Photogrammetry

Updated: 2026-09-08 by GPT-6 Astra, project lead.
Status: M1/M2 **OPEN — review fixes required**; M3 research delivered, production recovery **INCOMPLETE**.
M5-M7 remain **PLANNED**, except isolated CPU-SfM environment setup is pulled forward to unblock M3.
This file is the current working specification. Update its sections instead of appending a chronology.
The lead reviews and plans; Claude Opus and GPT-5.6 Terra perform implementation coding.
Historical recommendations in [the archived handoff](docs/archive/AGENT_HANDOFF-2026-09-07.md)
and the former `PLAN.md` (no longer present) are superseded by this specification.

## 1. Purpose and binding user decisions

- Recover Minecraft builds from ordinary vanilla screenshots and export a reviewable `.litematic`.
- Production must estimate cameras, FOV, relative block scale, and scene extent from image evidence.
  Camera manifests, scene bounds, reference litematics, and reference palettes are evaluation scaffolding.
- Support buildings and rooms first, then irregular terrain and geodes. Keep single-view input useful.
- Initial image scope is vanilla textures, including crops and HUD overlays. Resource packs, shaders,
  photographs of monitors, and expansion of the video workflow are outside the first release.
- Use a best estimate with visible uncertainty and user review. Allow masks, alignment corrections,
  a known wall/block length, and block/state corrections when the images are ambiguous.
- Prioritize GPU acceleration on the user's AMD Radeon RX 9070 XT; use CPU paths for correctness tests
  and tasks such as PyCOLMAP pose recovery that cannot use this GPU through CUDA.
- First production gate: **80% visible exact-state recovery and 90% visible geometry F1**,
  independently on Template 2, Template 6, and frozen small-view Template 3 subsets (section 5).
- Preserve existing exports and calibrated diagnostic commands. World ZIP export already exists;
  further world-export work is not a prerequisite for the reconstruction milestones.
- Do not fill unsupported hidden volumes. Distinguish observed blocks from inferred completions.
- Reference-based appearance calibration is evaluation-only and must never reach production.

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

Lead review of `e8dcd65` on 2026-09-08 established this state; see
[current findings and directives](docs/reports/LEAD-review-2026-09-08.md). Earlier review findings are historical.

- **345 tests passed** with `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider` in lead review.
  The pre-milestone baseline had 238 passing tests; M1's recorded snapshot had 252.
- M1 created local Git baseline `65d2a16`; fixture media, reference litematics, local assets,
  environments, generated output, and scratch artifacts are excluded. See [M1 baseline](docs/m1-baseline.md).
- The maintained benchmark now requires explicit reference/camera/scene paths, labels its mode,
  records hashes, dependencies, cache schema, effective settings, timing, peak Python heap, observed/
  inferred counts, and visible metrics. It rejects unsupported image-only scoring rather than passing
  supplied camera data as production input. Template 2 and 6 supplied-camera diagnostics are recorded
  in the M1 report and do not qualify for promotion.
- Working foundations: media discovery/image loading/video decoding, masks, local-JAR texture loading,
  model/partial-geometry extraction, texture classification, multiview evidence fusion, sparse volumes,
  project persistence, litematic validation/comparison, and both litematic and world ZIP export.
- Partial-state fitting and residual proposals are implemented but opt-in; they do not solve general geometry.
- `reconstruct_scene_surfaces` enumerates supplied scene bounds, projects cell centres, and selects a shell.
  High scores on box-filling fixtures do not establish correct recovery of interior geometry.
- `reconstruct_from_calibrated_views` requires known poses; it is not a camera solver. It now uses the
  shared camera projection convention, including explicit intrinsics and transformed crop/resize frames.
- Desktop `_reconstruct_source` uses the first screenshot alone when camera/scene manifests are absent.
  CLI, GUI, and benchmark also differ in entry points and palette selection.
- The previous near-plane alpha crash, opposing-slab culling and calibration roll/adapter defects are fixed.
- Sampling preserves some clipped/HUD evidence, but invalid pixels contaminate downsampled RGB and
  its separate cube-quad path lacks exact model geometry. Candidate scoring ignores missing-surface pixels.
- Visibility now supports shared intrinsics/model rendering. The frozen mask excludes 39% of frames,
  including unobscured build pixels; review requires per-frame mask validation before acceptance.
- Model appearance preserves alpha masks and element-local rotations in the cache. Cache schema is 15.
- Benchmark provenance/masks/breakdowns improved; full-state rotation, complete settings and matched
  pre-M2/current evaluation remain open. Latest lighting diagnostics are unmasked: T2 78.74% exact /
  92.66% geometry F1; T6 30.39% / 85.86%. Neither is an image-only result or matched baseline proof.
- `CameraSolver.solve` returns `CameraSolution`; calibration preserves measured optics and unknown position.
- M3's 15-frame seam diagnostic reproduced 1.01/0.71/0.65 px pooled medians for T2/T6/T3-sparse-8,
  with reference translation. Its harness still mishandles cropped intrinsics and incomplete coverage.
- PyCOLMAP's bounded experiment remains unsuccessful after correcting rotation scoring (0/41 within 5°).
  Known-length plane scale exists; inconsistent constraints can return a distance. Positions remain unknown.
- `.mcscan` remains schema 1 without camera/lattice persistence; schema migration is planned.

### Existing environment versus planned environment

- Workspace: `C:\Users\Likea\Documents\mc photogrammetry project`; Windows/PowerShell.
- Baseline interpreter: `.venv\Scripts\python.exe`, Python 3.10.11; retained for regression reproduction.
- Git baseline is established; the local branch is `main`. New environments must be ignored explicitly.
- GPU: AMD Radeon RX 9070 XT, plus integrated AMD graphics; approximately 62 GiB system RAM.
- `pycolmap`, `wgpu`, and `scipy` remain absent in the current `.venv`; DirectML was absent in initial review.
- Current texture cache schema is **15**; M1's frozen reports used **13**. Avoid copying defaults tables;
  capture the actual settings, including module-qualified names and structured values, in each run report.
- Fixture JAR: `%APPDATA%\.minecraft\versions\1.21.5\1.21.5.jar`; validate at runtime, never redistribute.
- Delivered CPU environment: ignored `.venv-sfm\Scripts\python.exe`, Python 3.12.10, PyCOLMAP 4.2.0;
  [environment and smoke report](docs/reports/CPU-SfM-environment-2026-09-08.md). GPU/wgpu remains planned.
- wgpu requires Python 3.11+; target D3D12/Vulkan on discrete AMD. PyCOLMAP uses the separate CPU setup.
- Dependency docs: [wgpu](https://wgpu-py.readthedocs.io/en/stable/start.html), [PyCOLMAP](https://colmap.github.io/pycolmap/index.html); record installed versions.
- Use PowerShell-compatible commands and explicit UTF-8; verify artifacts instead of trusting stale logs.

## 3. Approved implementation plan

### M1. Measurements and contracts — Terra; **BASELINE DELIVERED; acceptance gaps open**

- Snapshot current source/environment and establish excluded local version control before source changes.
- Upgrade the maintained benchmark: explicit reference selection, visible metrics, per-state errors,
  observed/inferred counts, phase timing, peak memory, input hashes, dependencies, JAR/cache provenance,
  and complete effective settings. No arbitrary `next(glob("*.litematic"))` reference selection.
- Separate image-only, supplied-camera diagnostic, and reference-assisted diagnostic modes.
  Only image-only input qualifies for production; report user-assisted runs separately.
- Freeze current results using corrected metrics; retain old percentages with their original denominators.
  A denominator correction is not an algorithm improvement.
- Freeze section 4 contracts before parallel integration. Keep calibrated commands as regression tools.
- Deliverable: reproducible baseline reports, metric definitions, and agreed interfaces. Evidence:
  [M1 baseline](docs/m1-baseline.md), reports under `exports/baselines/`, historical 252-test snapshot.
- Current closure: review B3/B4 — valid masks, full-state alignment/breakdowns, complete provenance,
  and matched baseline/current runs. Existing masked/unmasked diagnostics remain provisional.

### M2. Consistent projection and rendering — Terra; **REOPENED after lead review**

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
- Acceptance needs matched Template 2/6 baseline/candidate metrics and relevant passing tests, not tests alone.
- Dependency: M1 contracts; software correctness precedes GPU optimization.

### M3. Image-derived cameras and lattice — Claude; **RESEARCH DELIVERED; recovery incomplete**

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
- Current B5-B7: correct COLMAP pose scoring/validation, cropped seam intrinsics and coverage/integrity,
  and invalid known-length constraints; add independent tests and complete reproducible input provenance.
- Entry points: `scripts/calibrate_frames.py`, `relative_pose.py`, `overlay_mask.py`, `seam_landmarks.py`.
  [Pose report](docs/reports/M3-relative-pose-2026-09-08.md), [seam report](docs/reports/M3-seam-gate-2026-09-08.md).
- M4 may build useful supported planes in a local frame from image evidence and optional stated length;
  unresolved relative placement between views must remain explicit. M1 types/M2 rendering are dependencies.

### M4. Supported surfaces replace shells — Claude; **EXPERIMENTS IN PROGRESS**, geometry unsolved

- Represent each plane by axis, integer position, supported cell footprint, observations, and uncertainty.
  Footprints support holes and disconnected patches; rectangular extent is not proof of occupancy.
- Jointly fit translations, lattice phase, plane positions/extents using seams, reprojection, and occlusion edges.
- Construct cells from supported surfaces. Derive export bounds from reconstruction, never reference bounds.
- Alternate continuous camera refinement and discrete plane/lattice proposals; evaluate coarse hypotheses,
  then refine promising ones at higher resolution. Retaining the current scene is always a candidate.
- Split, extend, shift, or remove surfaces where spatially consistent residuals support geometry changes.
- Preserve alternatives for pure rotation, repeated textures, or disconnected views; do not invent placement.
  Single-view planar estimates remain useful, with unsupported depth relationships marked for review.
- Deliverable: image-only three-view Template 2 reconstruction, then Template 6 and small Template 3 subsets.
- Next bounded prototype: each Template 2 view separately, supported local plane cells and optional length,
  then join only with geometric evidence. Record automatic/assisted results; no default brightness cascade. ✔️
  Built and measured 2026-09-08: [report](docs/reports/M4-supported-plane-2026-09-08.md),
  [artifact](docs/reports/M4-supported-plane-templates-2.json). **Distance works, per-cell support does not.**
  A stated length places the plane to 0.4%/0.3%/0.1% across the three views; the automatic reading is
  2.1%/0.1%/1.6% on far planes and 196%/269% on a floor 1.6 blocks away, where the fixed-angle window holds
  almost no surface. Phase agreement then judges cells at chance (median 0.357-0.375 against a measured chance
  level of 0.378), including on the view where the distance is right to 0.1% and 168 cells are in frame. Scored
  at every period from block to texel: chance at all of them, so the scale is not the problem and the cue is.
  Third image-only cue measured and refuted, after planar region proposal and period strength. Not tuning it.

### M5. Materials and partial blocks — Claude; Terra owns renderer/palette support

- Apply existing face-aware texture scoring and multiview fusion to recovered surfaces.
- Fit lighting/exposure only from supported visible evidence; centre-pixel fallback cannot establish occupancy
  or dominate palette pruning. Revalidate existing lighting experiments on the corrected evidence.
- Preserve alternatives for rare materials. Scene-level support may rank but must not simply eliminate them.
- Propose local geometry from silhouettes, depth boundaries, residuals, and pixels unexplained by any surface.
- Consider air, cubes, slabs, stairs, carpets, fences, walls, panes, bars, and torches where relevant.
- Decide shape from geometry/visibility and material from rectified textures; keep their verdicts distinct.
- Recompute occlusion after accepted edits. Use block-specific neighbour rules for derived state properties;
  properties such as `open` are not universally derived. Mark visually unknowable properties as uncertain.
- Deliverable: increased visible exact matches and partial-block recall without hiding lost cells in a rate.
- Dependency: M4 supported geometry and M2 rendering; preserve isolated material/shape diagnostics.

### M6. GPU acceleration and production integration — Terra; Claude integrates stages

- Create the isolated environment described in section 2. Use wgpu on the discrete GPU, D3D12 first,
  Vulkan alternate, for rendering and batched image/texture scoring; keep a small CPU correctness path.
- Cache pyramids, features, geometry, descriptors, and accepted buffers. Re-render affected tiles for edits.
- Bound work by observed surfaces/active hypotheses. Avoid frame-by-volume enumeration and loading every
  full-resolution image unconditionally; record memory use and actual adapter/backend.
- Make CLI and desktop call one pipeline on all selected images. Identify duplicates without dropping useful
  viewpoints merely because repetitive Minecraft textures yield similar hashes.
- Provide overlays for cameras, lattice, uncertain cells, excluded frames, and unresolved components.
  Support mask edits, known-length constraints, alignment review, and block/state corrections.
- Save `.mcscan` before export. Preserve cancellation, resume information, and useful partial results.
- Validate and round-trip the litematic. Use an export Minecraft data version consistent with the asset version.
- Target: image-only Template 2 <=3 minutes on this machine with warm assets; report cold-start separately.
  This is a provisional performance target, not a measured result; quality gates take precedence.
- Dependency: M1 contracts and M2 correctness for acceleration; M4/M5 for production integration.

### M7. Irregular surfaces — Claude; Terra supplies acceleration/evaluation

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
  Claude must preserve measured orientation/roll/principal point in adapters; Terra owns shared camera/types.
- Freeze remaining semantics before integration: eye versus feet position, unresolved scale/components,
  typed plane alternatives, face/state ownership lookup and usable-pixel weights (review R5/R6).
- One production entry point accepts frames, assets, settings, optional user constraints, progress, cancellation.
  Ground-truth references and fixture manifests belong exclusively to evaluation adapters.
- Advance `.mcscan` to schema 2 for cameras, lattice, constraints, settings, evidence, and review status.
  Load schema 1 with explicit unavailable reconstruction metadata; preserve existing volumes.
- Reuse observed/inferred `BlockObservation` kinds. Confidence is uncalibrated until validation establishes
  reliability; keep fit statistics distinct. Persist alternatives without inserting every alternative into export.
- Default export selects the best estimate within reconstructed surfaces; exclude unsupported hidden filling.
- Assets, project version, exported data version, and state properties must remain consistent end to end.

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
- HUD removal while retaining unobscured pixels; mixed resolutions and exposure levels.
- GPU/CPU agreement away from raster boundaries, with independent analytic correctness checks.
- Production isolation tests fail if fixture manifests or reference litematics are accessed.
- Schema migration, cancellation/resume, partial-result persistence, exact litematic round trips.

### Promotion rules

- Require >=80% visible exact-state recovery and >=90% visible geometry F1 separately on Template 2,
  Template 6, and frozen small-view Template 3 subsets. Report automatic and user-assisted runs separately.
- Add procedural holdouts to reduce fixture-specific tuning; freeze data and evaluation before tuning.
- Every promotion needs settings, before/after artifacts, overlays, error attribution, and relevant passing tests.
- Re-evaluate baseline and candidate together after renderer or metric changes. Preserve calibrated-path
  regression reports under identical settings; investigate regressions rather than replacing their denominator.
- New terrain promotion applies the same gates to Templates 4/5; Template 1 measures scaling.
- Purely historical results below do not demonstrate that any new production gate has passed.

## 6. Immediate assignments and communication

| Agent | Responsibility | Next directive — assigned; acknowledge in comms |
|---|---|---|
| GPT-6 Astra | Lead review, priorities, interfaces, acceptance; no implementation coding. | Review B1-B7 closure and supported-plane prototype. |
| Claude Opus | Cameras, SfM, lattice/surfaces, materials/shapes, terrain. | B5-B7 evaluator/constraint fixes, then bounded local-plane reconstruction. |
| GPT-5.6 Terra | Benchmark, renderer, environment, GPU, persistence, UI/export. | B1/B2 sampling/scoring, then B3/B4 masks/alignment and matched baselines. |

- [Current lead review](docs/reports/LEAD-review-2026-09-08.md) defines reproductions and next deliverables.
  CPU-SfM setup is delivered; GPU implementation awaits M2 acceptance. Stage named paths, never `git add -A`.
- Claude's calibration/multiview work and Terra's measurement/renderer work proceed in parallel.
  Claude consumes Terra-owned camera utilities; do not concurrently edit shared projection implementation.
- All inter-agent messages go in [AGENT_COMMS.md](AGENT_COMMS.md), with model, timestamp including
  timezone, recipient, task, and evidence or blocker. Read it before work and at milestone boundaries.
- Record current ownership and completed milestones here only when evidence supports completion.
- Keep detailed experiments in linked reports with settings, results, and stopping reasons.
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
- Visibility references are evidence ceilings, not production inputs. Re-score archived box-based ceilings
  with the corrected alpha/model evaluator; historical counts are not current acceptance denominators.

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
