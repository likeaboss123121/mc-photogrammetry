# DS-2 render-sampling audit — 2026-09-12

Read-only audit of the M2 face-sampling / residual / shape-fit call paths, requested in
`AGENT_COMMS.md` (DS-2). Inspected at working tree HEAD `c6dba2c` with only documentation and
`tests/test_footprint_adapter.py` dirty. No source or test file was edited, no benchmark was run,
nothing was staged or committed, no package was installed, and no worker was started. Findings
below are static line-referenced readings plus in-memory analytic probes run with
`.venv\Scripts\python.exe` that wrote nothing to disk.

Scope read: `src/mc_photogrammetry/reconstruction/face_sampling.py`,
`.../residual.py`, `.../shape_fit.py`, their call sites in `.../raycast.py`,
`.../export/attribution.py`, `.../export/visibility.py`, `src/mc_photogrammetry/cli.py`,
`scripts/benchmark_templates.py`, and the relevant tests
(`tests/test_face_sampling.py`, `tests/test_shape_fit.py`). The two named historical claims are
stated as they appear in `docs/reports/LEAD-review-2026-09-08.md` §B1/§B2.

## Verdict on the two named historical claims

**B1 (masked pixels contaminate valid samples) — no longer a current defect.** The review's exact
reproduction now returns 160.0, not 20.0. `rectify_face_valid` (`face_sampling.py:141-181`)
multiplies RGB by source ownership before warping (`:166`), warps ownership through the same
transform (`:171-174`), area-resizes colour and coverage together (`:175-177`), and divides
(`:179`). Invalid-source colour is annihilated by the zero coverage weight, so it cannot enter the
valid average. Analytic probes (below) confirm the mean is invariant to the invalid pixels' colour,
including a red-255 HUD fill, at an image edge, and with a single surviving source column. The
per-pixel `valid = coverage > 0.01` gate at `:178` keeps partially covered pixels with their true
area weight, which is the correct treatment. One qualification: `observed_patch`
(`face_sampling.py:76-82`) neutralises invalid patch pixels to `self.rgb`, which is the
coverage-weighted mean, so neutralised texture is not photo evidence — this is deliberate and
therefore not contamination, but downstream uniformity/pattern statistics must not be described as
"observed" photo texture for those pixels (`raycast.py:720-721,751-752`).

**B2 (smaller candidates escape missing-surface penalties) — no longer a current defect, with one
narrow remainder.** The scoring region is now the single-cell silhouette of a material-independent
full-cube probe (`shape_fit.py:634-653`), shared by air, the current state and every hypothesis
(`:683-711,743-776,882-909`). Air is an explicit ranked candidate (`:783-785`), so a carpet that
leaves an observed slab uncovered is compared against the empty render over the same pixels.
Patch-by-patch coverage: `tests/test_shape_fit.py:274` (slab beats carpet in both palette orders)
and `tests/test_shape_fit.py:300` (occluded cell abstains). The remainder is that the **texture**
sub-score still gathers grain through candidate-box quads (`shape_fit.py:392-447`,
`:817-821`) and masks each patch with the *representative* material's exact-model ownership
(`:810-813`, `:431-439`), so the grain evidence domain is a function of the representative's
geometry rather than the candidate's own rendered face. With `TEXTURE_WEIGHT = 64.0`
(`shape_fit.py:84`) the grain decides most material choices, so this remains a live
candidate-dependent domain, but it no longer decides the shape silhouette.

## Current defects, with references

Severity is relative to a future run with the thin-block chain enabled. Only D1 is reachable today,
and only through the CLI: `FIT_PARTIAL_SHAPES = False` (`raycast.py:99`), no caller in `src/` or
`scripts/` passes `fit_partial_shapes=True` or `shape_palette`, and
`scripts/benchmark_templates.py:130-135` never enables it, so D2–D7 are latent in every stored
benchmark artifact.

### D1 — High — the residual pass is the one render path with no usable-pixel authority

`residual.py:56-105` accepts `scale` and `minimum_pixels` only. There is no mask parameter, and
`raycast.py:1242-1243` calls `residual_proposals` without `usable_masks` even though
`reconstruct_scene_surfaces` already holds them (`raycast.py:544`) and forwards them to
`fit_states` two lines later (`:1246`). Consequences, all at `residual.py:91-99`:

- `photo = cv2.resize(image, ...)` (`:91-92`) is an unmasked area downsample, so HUD/held-item and
  frozen-mask-excluded pixels are mixed into every render-resolution photo pixel. Unlike
  `shape_fit._frame_masks` (`shape_fit.py:209-218`), there is no "fully usable footprint" rule.
- `gain` (`:93-94`) and `typical` (`:96`) are medians over `seen = owner >= 0` only (`:88`), so
  excluded regions steer the normalisation that every face's score is divided by.
- `relative` (`:97`) then averages over every owned pixel with no candidate-independent domain,
  and `minimum_pixels = 8` (`:52`) is met by a handful of pixels, so a face that renders small can
  clear the `RESIDUAL_THRESHOLD = 1.0` bar (`:49`) on non-photographic pixels.

Measured compression from a synthetic 120x160 frame with one 40x60 HUD block (~12.5% of the
frame, versus the 39% exclusion the review measured for the real frozen mask): counting the HUD
pixels changes the frame gain from 1.1042 to 1.3750 and inflates `typical` from 18.04 to 56.29
(3.1x). Because `relative = difference / typical`, every face residual is divided by ~3, so faces
that should cross `threshold` do not, and `residual_proposals` silently proposes fewer cells.
This is an ordering/normalisation defect, not a quality difference: the same scene with differently
coloured HUD pixels yields different proposals.

### D2 — High — the production sampler cannot use exact-model geometry; proposal and sampling stages disagree

`sample_visible_faces` switches to exact alpha-tested ownership only when `blocks` and `palette`
are both supplied (`face_sampling.py:289-291,302-305,362-376`). Every production call omits them:
`raycast.py:686-691` and `:700-705` pass only `surface`/`proposed | surface` coordinates and a GUI
mask. The only callers that pass `blocks` and `palette` are tests (`tests/test_face_sampling.py:192-196`).
So the runtime sampler always takes the cube-quad `_nearest_surface` fallback (`:378`), while
`residual.py:84-86` renders the same reconstructed volume with exact `model_faces`. A slab, stair,
fence or cut-out is proposed against its true geometry and then sampled as a full cube. The
module docstring's "finish exact-model sampling" objective is implemented but unreachable from the
pipeline; the routing flag is absent, so this cannot be detected from diagnostics either.

Concrete secondary divergence in the same area: `attribution.render_volume` sizes the buffer with
`int(height * scale), int(width * scale)` (`attribution.py:217`), while `_nearest_surface` uses
`round` (`face_sampling.py:217-218`). At an odd width and `scale = 0.5` (e.g. width 101) the two
paths produce 50 versus 51 columns, so even the same nominal scale gives different visibility
buffers, ownership masks and `visible` values.

### D3 — Medium — the sampler and the residual do not render at the same resolution, contrary to the stated contract

`face_sampling.DEPTH_SCALE = 1.0` (`face_sampling.py:22`) carries the docstring "Sampling and
residual rendering must agree on which source pixels survive. The former quarter-resolution
default ... could both reject a visible face and disagree with the renderer" (`:23-29`).
`residual.RENDER_SCALE = 0.35` (`residual.py:38`) is unchanged, and `shape_fit.RENDER_SCALE = 0.5`
(`shape_fit.py:70`) is a third value. The CLI path never overrides any of them
(`cli.py:382-397` passes no `depth_scale` or `scale`). Acceptance decisions in
`sample_visible_faces` therefore use full-resolution ownership while proposal scoring and the
`visible_fraction` gate consumers compare against 0.35-resolution renders. At reduced scale the
buffer also quantises visibility coarsely: for one identical half-occluded face, the analytic probe
measured `visible` 0.96 at scale 1.0 versus 1.00 at 0.5 and 0.98 at 0.35. This is a low-magnitude
inconsistency at these scales, but it means the docstring's agreement claim is false as written and
a future sub-1.0 default would change which faces survive `minimum_visible_fraction`
(`face_sampling.py:408-410`).

### D4 — Medium — `fit_states` falls back to the mutable GUI-mask heuristic when no frozen mask is supplied

`shape_fit._frame_masks` calls `create_gui_mask` whenever `usable_masks is None`
(`shape_fit.py:197-200`), and for a frame absent from a supplied mapping it marks the whole frame
unusable (`:202-204`). The CLI thin-block path passes neither a frozen mask nor a usable-mask set
(`cli.py:382-397`), so a `--fit-partial-shapes` CLI run silently grades evidence against the
mutable heuristic while `scripts/benchmark_templates.py:183,196-198` reports
`usable_mask_policy = "all decoded image pixels; no frozen mask was supplied"`. The report and the
scored pixels would disagree. Conversely, `sample_visible_faces` (`face_sampling.py:416-417`) and
`residual.py` apply no frozen mask at all, so with a benchmark that does supply masks three
stages use three different pixel authorities for the same frame.

### D5 — Low/Medium — texture evidence domain is taken from the representative material's render

As noted under B2: `shape_fit.py:810-813` renders ownership for `representative` once per shape and
`shape_fit.py:817-821` restricts every material's rectified patch to `owner_masks[slot]`
(`:431-439`). When a geometry group contains several materials, the grain that ranks those
materials is read only inside the representative's own painted pixels. The representative is
chosen by an RGB prior (`:279-311`), not by evidence, so a poor prior choice shrinks or shifts the
texture domain for every candidate in that group. The `_slot_masks` helper (`shape_fit.py:352-389`)
is consequently entirely uncalled (no reference in `src/` or `scripts/`), and `_window`'s bbox and
`inside` mask (`:326-338`) are discarded, with only its `None` check at `:641-642` acting on the
result.

### D6 — Low — reduced-scale sampling quantises validity and the fractional-coverage weight

`_nearest_surface` rasterises at `width * depth_scale` (`face_sampling.py:217-249`), then
`sample_visible_faces` upsamples with `cv2.INTER_NEAREST` (`:412-415`) and divides the surviving
count by `depth_scale * depth_scale` (`:398`). One buffer pixel represents roughly
`1 / depth_scale^2` full-resolution pixels, so the `visible` quantity is a quantised estimate of
full-resolution pixels rather than a measured one; the probe above shows the half-occluded face's
`visible` moving between 0.96 and 1.00 purely with scale. `coverage` is the coverage field of
`rectify_face_valid`, which after area downsampling is a genuine area fraction, so the weighting
in `FaceSample.rgb`/`weights` (`face_sampling.py:64-89`) and `raycast.py:711-752` is sound; the
caveat is only that at sub-1.0 scale both the mask and the coverage start quantised.

### D7 — Low — `minimum_area` gates on an analytic corner-quad area, not the screen footprint

`sample_visible_faces` computes `area = quad_area(quad)` from the unclipped corner projection
(`face_sampling.py:331-335,382-384`) and compares it to `MINIMUM_FACE_AREA = 24.0` (`:37`), while
the visibility test two blocks later uses the in-frame footprint (`:402-410`). For a face near the
lens, `project_clipped_polygon_uv` returns the clipped polygon but the texture quad is rebuilt from
the original corners (`:345-348`), so a face straddling the near plane can have the analytic area of
its full quad counted for admission. This is not reachable as a false accept on its own because
`:408-410` and the renderer ownership check follow, but the two denominators measure different
things and should not both be called "area".

### Not defects worth repair, recorded to avoid re-litigating them

- `shape_fit._score_image` (`:221-245`) does compute a valid-only mean and erodes the domain for
  the gradient term (`:236-240`). An analytic probe with a synthetic HUD block found **zero**
  difference in `_gradient(photo)` on any pixel the safe domain admits (max |dEdge| = 0.0 over 3,344
  safe pixels). The masked area resample at `shape_fit.py:577-589` genuinely prevents bleed into
  fully-usable output pixels.
- The `render_result` ownership contract (`attribution.py:358-370`) and the `owner >= 0` use in
  `visibility.py:275-280` are consistent; `usable_masks` is correctly applied *after* ownership
  there, so the evaluator denominator is candidate-independent.
- `box_face_corners`/`_slot_masks` geometry and `project_points` scaling in `shape_fit._window`
  (`:314-338`) agree with `render_result` when `scale` divides evenly; no defect found.

## Smallest independent analytic controls

Each control uses an exact synthetic scene and asserts an invariant, not a number copied from the
implementation. None requires the adapter or production code as its oracle.

1. **C1 (D1): mask-invariance of a frame's residuals.** One 120x160 frame, one flat background
   face, a 40x60 rectangle marked unusable. Call `face_residuals` twice with the masked pixels set
   to two different constants (0 and 255). Assert the returned face score is identical and that
   the frame gain and `typical` are identical. This fails today with gain 1.3750 versus a control
   computed over valid pixels only (1.1042) at 3.1x inflated `typical`.
2. **C2 (D1): excluded pixels cannot promote a proposal.** Same frame, but make the excluded
   rectangle the only disagreeing region. Assert `residual_proposals` returns `{}` with the mask
   supplied, and that the proposal set is unchanged when the excluded rectangle's content changes.
3. **C3 (D2): fallback/exact agreement for a cut-out.** One cut-out state (the alpha-masked
   fixture already in `tests/test_face_sampling.py:170-202`), sampled twice: once through the
   coordinate-only fallback and once with `blocks`+`palette`. Assert the returned valid-pixel
   masks differ only inside the alpha cut-out, and record the exact-model mask as the expected one.
   This makes the "production never passes `blocks`/`palette`" gap fail loudly.
4. **C4 (D2): buffer-size agreement across ownership paths.** For images whose scaled dimension is
   fractional (width 101, `scale = 0.5`), assert the ownership array shapes from
   `_nearest_surface` and `render_result` are equal. Fails today (50 vs 51 columns).
5. **C5 (D3): scale-invariance of the accept/reject decision.** One partially occluded face, the
   same scene evaluated at `depth_scale` 1.0, 0.5 and 0.35; assert the set of accepted
   `(coordinate, slot)` is identical and the `visible` values agree within one buffer pixel of
   `1/depth_scale^2`. Fails today for the half-occluded face (0.96 vs 1.00 vs 0.98).
6. **C6 (D5): texture domain is the candidate's own geometry.** Two materials sharing one shape,
   with the representative's shrunken box and a second material's true box differing by one voxel
   quadrant. Assert the rectified texture pixels used for the second material lie inside its own
   exact-model ownership, not the representative's. Fails today by construction of `:810-821`.
7. **C7 (D6/D7): full-resolution equivalence of the cheap path.** With `depth_scale = 1.0`,
   `shrink = 1.0`, assert `sample_visible_faces` returns the same `(coordinate, slot, valid)` set as
   an all-pixels-valid ownership render, and that `area` equals the clipped screen footprint area
   within one pixel. Currently true at the default scale; the control pins the invariant that D6/D7
   would break if the default were lowered.

## Bounded repair order

1. **R1 (D1).** Add `usable_masks`/`valid_pixels` to `face_residuals` and `residual_proposals`,
   thread `usable_masks` from `reconstruct_scene_surfaces` into the `raycast.py:1242` call, mask the
   photo before the area downsample, and compute `gain` and `typical` only over
   `seen & usable`. Land C1 and C2 with it. This is the only defect that changes current CLI
   behaviour and it is self-contained in `residual.py` plus one call site.
2. **R2 (D2, D3).** Make the ownership buffer dimensions single-sourced from `render_result`
   (or assert equality) and route the exact-model sampler from the pipeline, with the cube fallback
   kept explicitly labelled for callers that still lack a palette. Land C3 and C4.
3. **R3 (D4).** Decide one mask authority per call: either require `usable_masks` whenever
   `fit_partial_shapes` is on, or pass the same frozen masks to `sample_visible_faces` and
   `residual.py`. Do not keep the `create_gui_mask` fallback silently active in a run whose report
   claims "all decoded pixels".
4. **R4 (D5).** Move the texture patch's ownership mask from the representative's render to the
   candidate's own exact-model render, or make the material prior a documented, bounded
   tie-breaker that cannot narrow the domain. Land C6 and compare default scoring before/after.
5. **R5 (D6, D7).** Pin the full-resolution equivalence with C7 before any change to
   `DEPTH_SCALE`, and record the buffer-size and `visible` quantisation in diagnostics so a
   future scale change is visible rather than inferred.

No step requires touching `shape_fit`'s shape-selection logic, the palette, camera recovery, the
M4 footprint code, or any annotation path. R1 is independent of R2–R5 and can be reviewed alone.

## Provenance and limits

- Files read are listed above; no file was modified by this audit. The report itself is the only
  write.
- The numeric probes (B1 reproduction mean 160.0; invariance to invalid colour; HUD-driven gain
  1.3750 vs 1.1042 and typical 56.29 vs 18.04; zero edge leakage into the shape-fit safe domain;
  face `visible` 0.96/1.00/0.98 across scales) were computed in memory with the workspace
  `.venv` (Python 3.10.11, OpenCV 5.0.0, NumPy 2.2.6) and printed to stdout only.
- The B2 judgement rests on static reading of `shape_fit.py:634-653,683-711,743-776,882-909` plus
  the two existing regression tests; no end-to-end fit was run, so the claim that silhouette
  selection is palette-order independent at scale is the tests' result, not a fresh one.
- `FIT_PARTIAL_SHAPES = False` (`raycast.py:99`) and no caller anywhere in `src/` or `scripts/`
  passes `fit_partial_shapes=True` or `shape_palette`, so D2, D4, D5 and the residual defects are
  not exercised by any stored benchmark artifact in this tree; only a hand-written CLI invocation
  with `--fit-partial-shapes` reaches them. This audit did not run the fixtures or the test suite, so
  the last independently verified full-suite baseline remains the 514-passing result recorded by
  the lead, not a result from this call.
