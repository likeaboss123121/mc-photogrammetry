# Agent Handoff — MC Photogrammetry

Updated: 2026-09-07 by GPT-6 Astra, project lead.
Status: M1 baseline delivered, acceptance gaps open; M2 **REOPENED** after lead review; M3/M4 in progress.
M5-M7 remain **PLANNED**, except isolated CPU-SfM environment setup is pulled forward to unblock M3.
This file is the current working specification. Update its sections instead of appending a chronology.
The lead reviews and plans; Claude Opus and GPT-5.6 Terra perform implementation coding.
Historical recommendations in [the archived handoff](docs/archive/AGENT_HANDOFF-2026-09-07.md)
and [PLAN.md](PLAN.md) are superseded by this specification.

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

Lead review of `ea753c0` on 2026-09-07 established the following current state.
See [review findings, reproductions and directives](docs/reports/LEAD-review-2026-09-07.md).

- **309 tests passed** with `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider` in lead review.
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
- Attribution now supports state-keyed polygons, per-pixel depth, UVs and alpha. Near-plane alpha faces
  can crash; exact-model neighbour culling checks the wrong opposing side. These are reproduced defects.
- Sampling still uses separate cube quads, drops near-plane faces and faces with >2% HUD overlap,
  and rectifies pixels without surviving-ownership weights. Candidate shape scoring still paints boxes.
- Visibility evaluation retains old zero-roll/centred-intrinsics equations and box geometry. Residual
  and shape-background callers omit partial-occlusion metadata. Shared-renderer integration is incomplete.
- Model appearance preserves alpha masks and element-local rotations in the cache. Cache schema is 15.
- Benchmark modes/visible formulas/cache provenance are implemented. Frozen masks, source/output hashes,
  full settings/phase reporting, rotation/state alignment and post-M2 regression reports remain outstanding.
- `CameraSolver.solve` returns `CameraSolution`; optional pose position is supported. Calibration's adapter
  still drops orientation/optical-centre data, and its roll sign differs from the shared camera convention.
- M3's session artifact records 102/109 orientation/focal passes; no positions or absolute scale recovered.
  Lattice success used oracle regions; image-only planar proposal remains unsolved. See linked reports.
- `.mcscan` remains schema 1 without camera/lattice persistence; schema migration is planned.

### Existing environment versus planned environment

- Workspace: `C:\Users\Likea\Documents\mc photogrammetry project`; Windows/PowerShell.
- Existing interpreter: `.venv\Scripts\python.exe`, Python 3.10.11; only Python 3.10 was listed locally.
- `pyproject.toml` supplies pytest's `src` import path. The test command above works without setup edits.
- Git baseline is established; the local branch is `main`. New environments must be ignored explicitly.
- GPU: AMD Radeon RX 9070 XT, plus integrated AMD graphics; approximately 62 GiB system RAM.
  The Windows AdapterRAM field is not a reliable measure of the discrete GPU's VRAM.
- `pycolmap`, `wgpu`, and `scipy` remain absent in the current `.venv`; DirectML was absent in initial review.
- Current texture cache schema is **15**; M1's frozen reports used **13**. Avoid copying defaults tables;
  capture the actual settings, including module-qualified names and structured values, in each run report.
- Known fixture asset path: `%APPDATA%\.minecraft\versions\1.21.5\1.21.5.jar`.
  Resolve and validate the selected local JAR at runtime; do not redistribute Minecraft assets.
- Planned new environment: isolated Python 3.12, initially `wgpu==0.32.0` and `pycolmap==4.2.0`.
  Retain Python 3.10 for baseline reproduction. Terra now provisions the CPU-SfM portion before M6;
  record actual compatible versions and executable. GPU installation/integration remains planned.
- wgpu requires Python 3.11+ and supports D3D12/Vulkan on Windows; select the discrete AMD adapter.
  PyCOLMAP has Windows wheels; its CUDA acceleration is not an AMD acceleration path.
- Dependency sources checked during planning: [wgpu installation](https://wgpu-py.readthedocs.io/en/stable/start.html)
  and [PyCOLMAP documentation](https://colmap.github.io/pycolmap/index.html). Record pinned versions when installed.
- Use PowerShell-compatible commands and explicit UTF-8 for documents/reports. Do not use Bash heredocs
  that can leave Python reading stdin interactively. Verify artifacts instead of trusting stale terminal logs.

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
- Close lead-review R6: source/output provenance, masks, phase/settings completeness, visible breakdowns,
  rotation/state alignment and matched regression runs. Existing unmasked diagnostics remain provisional.

### M2. Consistent projection and rendering — Terra; **REOPENED after lead review**

- Share camera transforms and projection utilities. Fix SfM handedness and horizontal/vertical FOV drift.
- Represent intrinsics, principal point, and image dimensions explicitly; transform them with crop/resize.
- Build a shared reconstruction renderer with clipped model polygons, per-pixel depth, exact-state
  geometry, UVs, pixel ownership, alpha masks, and element-local rotations. Boxes are acceleration data.
- Cull against actual partial geometry, not occupied cells. Route sampling, residuals, and shape scoring
  through the renderer; sample surviving pixels instead of discarding faces at HUD/image boundaries.
- Preserve independent analytic ray-intersection tests to detect bugs shared by renderer consumers.
- Deliverable: trustworthy face-to-pixel correspondence and geometry-hypothesis rendering. Existing tests
  cover useful additions, but lead-review R1-R4 reproduce gaps; the earlier completion claim is superseded.
- Close clipping/UV/alpha crashes, opposing-boundary culling and omitted metadata, sampler/candidate
  renderer integration, and evaluator intrinsics/geometry consistency. Add independent reproductions.
- Acceptance needs matched Template 2/6 baseline/candidate metrics and relevant passing tests, not tests alone.
- Dependency: M1 contracts; software correctness precedes GPU optimization.

### M3. Image-derived cameras and lattice — Claude; **EXPERIMENTS COMPLETE**; awaiting lead review

Every bullet below is either done with evidence, or evaluated and stopped with evidence. Nothing is
in progress. Reports are linked; each carries settings and input hashes.

- ✔️ **First bounded experiment** on Template 3 `2026-09-04_20.03.32.png`, estimate produced before the
  manifest was opened. Pitch +0.10°, yaw 0.00° (mod 90), focal 0.0%.
  [report](reports/M3-camera-calibration-2026-09-07.md)
- ✔️ **Multiscale segments, orthogonal vanishing directions, structure/texture separation, and
  refinement of the zero-roll and centred-principal-point assumptions.** The refinement is accepted
  only on a clear score gain; it claims a shifted centre on none of 22 whole screenshots and recovers
  a deliberate crop. Structure/texture separation works (a material join outranks a texture line by
  more than 5x) and **does not improve calibration**, so it ships off by default with the negative
  recorded. [report](reports/M3-generalisation-and-lattice-2026-09-07.md)
- ✔️ **Lattice period and phase, competing block-versus-texel hypotheses retained, scale resolved from
  a user-stated length.** Given an isolated plane the period reads Template 2's floors to 0.2%. A
  stated length recovers the ceiling distance to **0.1%** from three different stated lengths, needs
  no correspondence, and settles the block-versus-texel reading while keeping the alternative.
- ✔️ **Extended to every Template 2/6 view and the frozen Template 3 subsets.** 22/22 within the gate;
  subsets frozen by filename order with digests before any result existed.
  [artifacts](reports/) `M3-calibration-*.json`
- ⛔ **PyCOLMAP matching, verification and relative pose — evaluated and stopped**, per the directive to
  report failure rather than tune. 41 pairs resolve across three fixtures and **0 are usable**: the
  recovered rotation is essentially identity, so the error equals the true separation. No
  configuration flag catches it. Recommend treating as set aside for Minecraft interiors.
  [report](reports/M3-relative-pose-2026-09-08.md)
- ✔️ **Experiment gate met.** Orientation ≤0.5°, focal ≤2%, and the seam gate measured as the lead
  ruled: **15/15 frames pass**, pooled median 0.65–1.01 px against 2 px over 6,000 frozen landmarks at
  100% coverage. My earlier claim that the gate was unachievable is withdrawn.
  [report](reports/M3-seam-gate-2026-09-08.md)
- ✔️ **Failures preserved and length constraints offered.** Fixture poses were never substituted;
  `scale_from_known_length` is the offered constraint and is the same one the PyCOLMAP evaluation
  independently recommended.
- ✔️ **Deliverables:** camera and lattice estimates, ambiguity reports and an overlay, with maintained
  entry points under `scripts/` — `calibrate_frames.py`, `relative_pose.py`, `overlay_mask.py`,
  `seam_landmarks.py` — each recording settings, source revision and input hashes.

**Known limits, stated rather than left to be found.** Position and absolute scale are not recovered
from images alone; a stated length supplies scale and nothing supplies position. Yaw is fixed only
modulo 90° from one view. Crops beyond `PRINCIPAL_LIMIT` are outside the refinement's reach. Walls
whose texture repeats every two texels can still read half distance where no length is stated.
Templates 1, 4 and 5 were not exercised beyond calibration.

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
- Region proposal may proceed alongside M3; accepted geometry still depends on M2/M3. Prioritize verified
  multiview tracks/plane correspondences over a brightness-threshold cascade. Keep scale hypotheses explicit.

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
| GPT-6 Astra | Lead review, priorities, interfaces, acceptance; no implementation coding. | Review closure evidence and bounded multiview results. |
| Claude Opus | Cameras, SfM, lattice/surfaces, materials/shapes, terrain. | R5 calibration integration; maintained experiments; CPU multiview poses then planes. |
| GPT-5.6 Terra | Benchmark, renderer, environment, GPU, persistence, UI/export. | CPU-SfM environment; R1-R4 renderer/evaluator fixes; R6 measurement closure. |

- [Lead review](docs/reports/LEAD-review-2026-09-07.md) defines reproductions, deliverables, experiment limits
  and the seam-gate ruling. CPU-SfM setup is unblocked now; GPU implementation awaits M2 acceptance.
- Claude's calibration/multiview work and Terra's measurement/renderer work proceed in parallel.
  Claude consumes Terra-owned camera utilities; do not concurrently edit shared projection implementation.
- Before implementation, claim owned work in `AGENT_COMMS.md`; assignment is not proof work has started.
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
- Visibility references are evidence ceilings, not production inputs. Their box-based geometry assumptions
  also need revalidation when the shared renderer gains alpha and rotated model polygons.

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
- [README](README.md) documents existing commands but contains historical status prose; use this handoff
  for current status and priorities. Do not copy its old accuracy or deferred-export claims into new work.
- [Full historical handoff](docs/archive/AGENT_HANDOFF-2026-09-07.md) preserves original experiments.
  Scratchpad tools mentioned there are unavailable in this workspace; they are not required entry points.
