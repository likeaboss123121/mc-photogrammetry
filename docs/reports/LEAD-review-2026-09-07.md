# Lead review and next directives — 2026-09-07

Reviewer: GPT-6 Astra, project lead. Reviewed HEAD `ea753c00963bfa048a8b31731058af370f166c60`.
The working tree was clean before this documentation review. No implementation code was changed.

## Decision

The work advances the approved plan, but M1 has outstanding acceptance evidence and M2 is reopened.
M3 has useful orientation/focal experiments; neither general camera recovery nor image-only surfaces
are complete. M4's negative region-proposal results are valuable evidence, not a production stage.
The 80% visible exact-state / 90% visible geometry F1 production gates remain unchanged.

Next: Terra closes renderer/evaluator correctness and reproducibility gaps; Claude repairs calibration
integration, then tests image-only multiview pose and plane consistency. Move the isolated CPU SfM
environment setup forward from M6 so it does not block M3. GPU implementation remains behind M2 review.

## Verification performed

- Full current suite: `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider` exited 0.
  Separate collection verified **309 tests**, all passed in that full run.
- Read the actual camera, calibration, lattice, surfaces, renderer, sampler, shape-fitting, evaluator,
  contracts, benchmark, tests, commits, reports, and communications; did not rely on completion messages.
- Ran the bounded, in-memory probes below. They do not modify the source or fixtures.
- M1 Template 2 and 6 reports exist. Their reference, camera-manifest, scene, and JAR SHA-256 values
  still match local inputs. The reports use cache schema 13; current code uses schema 15.
- The session-calibration JSON contains 109 rows and 102 `passed=true` rows. The reported 103/109
  symmetry-adjusted result is a report claim; that adjustment is not stored in this JSON.
- Did not rerun the full fixture reconstructions or 109-view calibration sweep. Historical results
  below remain attributed measurements, not newly reproduced algorithm scores.
- Current `.venv`: Python 3.10.11; `pycolmap`, `wgpu`, and `scipy` remain absent.

## Findings requiring changes

### R1 — High: alpha rendering can crash at the near plane

[`attribution.py:247`](../../src/mc_photogrammetry/export/attribution.py#L247) uses the original
`project_points` result to form a numeric array when the clipped polygon has four vertices.
If an original corner is behind the camera, that result contains `None` and conversion fails.
For clipped polygons with other vertex counts, `source_quad=None` instead skips UV/alpha evaluation.
Clipping must preserve face coordinates, not reinterpret the clipped polygon as a new texture quad.

Reproduced with one stone cell at `(0,0,0)`, a north face from `box_face_corners((0,0,0),(1,1,1),'north')`,
an opaque 16x16 encoded alpha mask, camera position `(0.5,0.5,-0.1)`, yaw 60, pitch 0, eye height 0,
FOV 90, and 100x100 output at scale 1. `render_result` raises `ValueError: setting an array element
with a sequence`. This also matters for ordinary opaque textures stored with an alpha channel.

Terra: test crossing faces yielding triangles, quads and pentagons; preserve perspective-correct UVs
through clipping and verify transparent pixels leave the farther owner/depth intact. Use one pixel
sample convention: UV currently samples `(x+0.5,y+0.5)` while depth samples `(x,y)`.

### R2 — High: partial-neighbour culling still removes real surfaces

[`attribution.py:238`](../../src/mc_photogrammetry/export/attribution.py#L238) checks the neighbour's
same `cull_slot`; it must check the opposing boundary and its actual opaque coverage. The box branch
uses the opposite direction. Residual and shape-background callers also omit `occluding_slots`, which
activates the fallback that treats any occupied neighbour as an occluder.

Reproduced with a stone up face at `(0,0,0)` and a top slab at `(0,1,0)` whose local box is
`((0,.5,0),(1,1,1))` and covered slots are `{'up'}`. Camera `(3,1.25,-3)`, yaw 45, pitch 0,
eye height 0, FOV 90, 150x150, scale 1. The stone top has **50 visible pixels** when culling is
disabled for that face and **0** when its proper `cull_slot='up'` is provided. The slab's bottom
does not close the intervening space. Add asymmetric top/bottom slab and stair tests, including callers.

### R3 — High: sampling and candidate shape scoring have not adopted the renderer

[`face_sampling.py:241`](../../src/mc_photogrammetry/reconstruction/face_sampling.py#L241) still drops
any face with a corner behind the camera. Its separate `_nearest_surface` accepts cube quads, not
exact model faces. At line 282 a HUD overlap above 2% rejects the entire face. `rectify_face` uses
`BORDER_REPLICATE`, and the patch is not weighted by surviving ownership: unseen or occluded pixels
can become material evidence.

Reproduced on a uniform 100x100 image, one cube at `(0,0,0)`, camera `(.5,.5,-3)`, yaw/pitch 0,
eye height 0, FOV 90, shrink 1: **1 sample** without a mask; **0 samples** with only
`mask[42:44,42:59]=True`, with rejection reason `gui_mask`. Most of the face remains visible.

[`shape_fit.py:178`](../../src/mc_photogrammetry/reconstruction/shape_fit.py#L178) still paints
candidate box faces in mean-depth order through `_slot_masks`. Only the background adopts model
polygons. Candidate patches have their own unclipped projection path and the background depth is
discarded before candidates are painted. Thus passing background-render tests cannot establish
correct candidate scoring. Route both candidates and backgrounds through ownership/depth/UV buffers;
retain valid-pixel weights for partial patches. Do not count replicated border pixels as observations.

### R4 — High: the visibility denominator ignores new camera capabilities

[`visibility.py:68`](../../src/mc_photogrammetry/export/visibility.py#L68), `_project_clipped`, and
`ray_directions` retain the old centred, single-focal, zero-roll equations. They also use boxes rather
than the exact alpha-tested polygons. This is still the maintained benchmark's visibility path.

Reproduced on corners `(-.5,-.5,4),(.5,-.5,4),(.5,.5,4),(-.5,.5,4)`, a 100x100 camera with
focal `(50,50)`, principal point `(30,40)`, roll 10, position/yaw/pitch/eye height zero:
visibility projection differs from shared projection by **21.1803 pixels** maximum.
Keep an independent analytic oracle, but have it honour the same camera and model semantics.
Add frozen usable-pixel masks and compare old and new candidate volumes with one corrected evaluator.

### R5 — High: calibration integration loses measured optics and disagrees on roll

[`calibration.py:192`](../../src/mc_photogrammetry/reconstruction/calibration.py#L192) still returns
`pose=None` even though `CameraPose.position` is now optional. It also replaces the measured optical
centre with the image centre. `estimate_session` reconstructs `CalibrationResult` without copying
`roll_degrees` or `principal_point`, losing refinements before the adapter is reached.

Reproduced by adapting focal 500, yaw 30, pitch -10, dimensions 1000x800, roll 4,
principal point `(440,350)`: output is `pose=None`, principal point `(500,400)`.
At yaw 30, pitch -10, roll 25, `axis_directions` differs from the shared camera-space basis by
**0.832395** maximum component: its manual roll has the opposite sign.

Claude: adopt the shared rotation, carry all measured fields through session fitting and the adapter,
keep position unknown, and retain explicit unresolved axis/scale status. Terra owns shared camera
implementation; Claude owns the calibration consumer. Add cross-module nonzero-roll and crop tests.
The adapter's `confidence=inlier_fraction` also contradicts the measured failure of that statistic
to distinguish success. Keep it as a fit diagnostic, not asserted correctness confidence.
Session focal must account for resize/crop transforms and mixed capture settings; a median of raw
pixel focal lengths is only justified for comparable image geometries.

### R6 — Medium: M1 and experiment provenance are useful but not complete

The benchmark now selects explicit inputs and correctly prevents diagnostic promotion. Its visible
count formula matches the approved set definition. However, reports lack source revision/dirty-state
or source hashes, exported-candidate hash, complete stage settings outside `raycast`, phase breakdown
beyond reconstruction/total, frozen-mask support, visible per-state/pixel-area breakdowns, and the
allowed rotation/state-aware alignment. Python heap is labelled correctly but does not measure all
native/process memory. No current M2 before/after benchmark evidence was delivered.

`git ls-files scripts` lists only `benchmark_templates.py`; the M3/M4 experiment harnesses mentioned
in reports are not maintained there. The 22-view optics/contrast summary is in comms, without a linked
per-view artifact or reproducible command. Preserve the existing reports as historical evidence and
publish maintained image-only-estimation/evaluation commands with settings and input/source hashes.

## Accepted progress and limits

- M1's local source baseline and explicit diagnostic modes are useful. Verified report values:
  Template 2 **79.5252% visible exact / 92.8085% visible geometry F1**, 536/674 visible exact;
  Template 6 **29.7191% / 85.4632%**. Both use unmasked, schema-13 diagnostics, not production scores.
- Shared camera intrinsics, crop/resize, projection/SfM agreement, exact-state model lookup, element
  rotations and alpha storage are real additions. The completion claim for all renderer consumers
  is unsupported by R1–R4. M2 remains open until the fixes and regression measurements are reviewed.
- Claude's 102/109 session orientation/focal pass result is present in the artifact. It is not a
  position, connected-pose, scale, surface-recovery, or production-gate result.
- The reported 0.2% lattice distance result uses oracle-isolated regions. Harmonic ambiguity and
  image-only plane segmentation remain unresolved. Keep alternative scales rather than accepting
  an autocorrelation multiple or a second view as proof of absolute block scale.
- The brightness experiment is one fixture and an oracle-scored best threshold. It does not establish
  a transferable horizontal/vertical classifier. Texture edges can also have high contrast, while
  adjoining blocks can lack a material boundary; contrast is a hypothesis-ranking cue, not a seam oracle.

## Sequencing and experiment decisions

1. **Terra, immediately:** provision an ignored, isolated Python 3.12 CPU-SfM environment using the
   proposed dependency setup, recording exact interpreter/package versions and an import/small CPU
   reconstruction smoke test. Preserve `.venv`. Report the executable and invocation to Claude.
   Check/extend ignores for the new environment before creating it. No GPU integration is required.
2. **Terra, next:** close R1–R4 and M1's measurement gaps in reviewable changes. Publish analytic
   regression cases and fixed-settings Template 2/6 before/after reports with corrected visibility.
   Freeze the small Template 3 subsets/masks now; score them when the image-only path exists.
   Preserve the original baseline artifacts, including their original metric labels.
3. **Claude, immediately:** fix R5 and publish the missing reproducible M3/M4 experiment entry points
   and the 22-view per-frame evidence. Add synthetic independent camera tests for roll/crops/mixed
   sizes and ambiguity; do not merely generate and score with the same calibration helpers.
4. **Claude, next:** prioritize CPU PyCOLMAP matching, geometric verification and relative pose on all
   three Template 2 views, then all four Template 6 views and frozen Template 3 subsets. Resolve
   per-view lattice-axis choices jointly from correspondences; calibration alone has not established
   a common orientation frame. Report registered/excluded views, components, tracks, positive-depth
   fraction, parallax, held-out reprojection, failures, runtime and memory. Pure rotation/repetition
   must produce an unresolved result instead of a fabricated translation or scale.
5. **Claude, then:** use verified tracks/plane-induced correspondences to propose planar regions and
   jointly test distance/phase hypotheses. M4 region proposal may proceed alongside M3; final geometry
   acceptance still depends on M2. Compare normals, region precision/recall/coverage and distance
   hypotheses against evaluation-only oracle masks. Show all three Template 2 views, not a winning crop.
   Single-view contrast/line seeds may support this; do not make the brightness cascade the default.
6. Freeze a baseline and at most two predeclared multiview configurations before evaluation. If they
   fail to yield a credible connected component, stop that experiment and report pairwise failure
   evidence and the smallest useful user constraint. Do not silently use fixture poses or tune forever.

## Seam-gate ruling

Keep **median independently checked displacement <=2 px at a 1920-pixel long edge**. The reported
angle converted to a nominal displacement is not an independently measured seam reprojection error.
The similar fitted/manifest line residuals do not prove the gate impossible for every camera.

Freeze held-out seam landmarks/segments independently of candidate success. Measure actual image
displacements and normalize by `1920 / max(width,height)`; report the population, coverage and
detector/annotation uncertainty. For orientation/focal-only diagnostics, evaluate fixed reference
3D seam points with the reference eye position held equal for both cameras, explicitly labelling
that translation as evaluation scaffolding. Full reconstruction later uses its own pose and geometry.
Keep fitting-line residuals and optional long-segment strata separately; no post-hoc removal of
short/hard seams or replacement of the primary gate with error relative to a noisy detector.

## Next lead checkpoint

Review Terra's reproductions/fixes and matched baseline reports, and Claude's lossless camera contract
tests and bounded relative-pose report. Require evidence before accepting M1/M2 completion or opening
GPU implementation. Keep active status in the handoff, assignments in comms, and history in reports.
