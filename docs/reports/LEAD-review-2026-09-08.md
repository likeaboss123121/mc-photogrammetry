# M1–M3 acceptance review — 2026-09-08

Reviewer: GPT-6 Astra, project lead. Reviewed `e8dcd6522acee043ae1e780b4cd49718f385849c`.
This review changes documentation only. A concurrent `.gitignore` edit was left untouched.

## Acceptance decision

| Milestone | Decision | Reason |
|---|---|---|
| M1 | Substantial implementation accepted; milestone remains open | Frozen-mask validity, state alignment and matched baseline evidence remain incomplete. |
| M2 | Earlier fixes accepted; milestone remains open | Validity-weighted sampling and candidate-independent shape scoring still fail. |
| M3 | Bounded research results accepted; production milestone remains open | Orientation/focal recovery and assisted plane scale exist; connected camera positions do not. Evaluation/constraint defects remain. |

The approved 80% visible exact-state and 90% visible geometry F1 gates are unchanged. A failed,
properly stopped experiment can be complete as research without fulfilling the camera-recovery
requirement. No current result demonstrates the intended image-only reconstruction pipeline.

## Verification

- Full suite: `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider` exited 0;
  **345 tests passed**, confirmed by separate collection. No implementation tests were added by the lead.
- Replayed the previous near-plane alpha crash: it now renders 6,700 pixels without crashing.
  Replayed the opposing-slab culling case: the floor retains 50 pixels both with and without culling.
- Shared/calibration roll basis difference is now zero on the previous nonzero-roll probe.
  Camera adaptation preserves yaw 30, pitch -10, roll 4, principal point `(440,350)`, and unknown position.
- Reran maintained seam scoring on the existing Template 2, Template 6 and `t3-sparse-8` artifacts:
  all 15 reported frames pass; pooled normalized medians reproduce **1.01, 0.71, 0.65 px** respectively.
  This reuses stored camera estimates and reference translation; it does not rerun image calibration.
- Recomputed all 41 stored PyCOLMAP pair rotation errors in the corrected camera convention:
  **0/41 within 5 degrees** still holds. No new feature-matching configuration was run.
- Verified the lighting report candidate files match their recorded hashes. Their unmasked visible
  metrics are Template 2 **526/668 = 78.7425%, geometry F1 92.6606%** and Template 6
  **610/2,007 = 30.3936%, geometry F1 85.8576%**. These are stored supplied-camera measurements.
- Confirmed `.venv-sfm` exposes PyCOLMAP 4.2.0 and its two-view APIs; the CPU environment is delivered.
- Did not rerun full fixture reconstructions or the 109-view calibration sweep. The probes below are
  in-memory review checks; results are separated from historical fixture measurements.

## Findings and required closure

### B1 — High, Terra: masked pixels contaminate samples advertised as valid

[`face_sampling.py:129`](../../src/mc_photogrammetry/reconstruction/face_sampling.py#L129)
warps/resizes RGB independently of validity, then accepts a reduced pixel if its valid fraction exceeds
0.01. Masked-out RGB remains in the average. Neutralising rejected output pixels afterwards cannot fix it.

Reproduction: 128x128 RGB image, valid columns `[:,::8]` with RGB 160, all invalid pixels RGB 0;
quad `(0,0),(128,0),(128,128),(0,128)` and default 16x16 output. `rectify_face_valid` declares
**all 256 pixels valid with mean RGB 20**, although every observed source pixel is RGB 160.

Warp/resample weighted colour and coverage together; divide by surviving weight and retain coverage
weights for descriptors. Invalid-source colour must not affect a valid sample. Test HUD boundaries,
occlusion boundaries and image edges, including subpixel interpolation. Sampling still has a separate
cube-quad `_nearest_surface` path and no exact-state model input; complete the planned renderer integration.

### B2 — High, Terra: smaller candidate shapes escape missing-surface penalties

[`shape_fit.py:374`](../../src/mc_photogrammetry/reconstruction/shape_fit.py#L374) averages residuals
only over pixels owned by that candidate. Where a carpet leaves an observed slab unfilled, those pixels
are excluded. `_window_data` is calculated but ignored. Texture evidence still uses separate box patches.

Reproduction: one quartz slab at `(0,0,0)`, camera `(.5,.25,-3)`, yaw/pitch 0, FOV 90, eye height 0,
120x120; same RGB 160 for a bottom slab and white carpet. Render the slab photograph at scale 1 with
ambient shading, then call `fit_states` with scale 1 and edge/texture weights 0. Carpet-first palette
selects **carpet**; reversing the palette selects **slab**. The omitted slab pixels never break the tie.

Score every hypothesis over one candidate-independent usable region, including pixels revealed by
removing geometry and unexplained image evidence. Respect scene depth/occlusion. Add a meaningful
missing-surface test whose answer is independent of palette order, then compare default scoring too.
Do not solve this by requiring a larger shape; valid thin shapes must still win on thin-shape images.

### B3 — High, Terra: the frozen mask removes substantial genuine build evidence

[`vanilla-hud-2560x1369-v1.json`](../evaluation_masks/vanilla-hud-2560x1369-v1.json)
excludes **39.0018%** of every frame. On visual inspection of Template 2 `2026-09-02_13.22.14.png`,
the supposed chat/status/held-item rectangles include unobscured table, floor and wall pixels.
For example `(400,1000)` is table evidence, but the chat rectangle excludes it.

Candidate independence alone does not make this the correct usable-pixel denominator. Produce reviewed
per-frame overlay exclusions, with compact masks/polygons and coverage overlays. Freeze them before
the next scored run, preserve v1 as historical, and report sensitivity to mask changes. Do not reuse
`overlay_mask.py`'s static-pixel heuristic as evaluation truth: unchanged intensity is not proof of HUD.

### B4 — Medium, Terra: state-aware alignment and baseline closure are incomplete

[`evaluation.py:182`](../../src/mc_photogrammetry/evaluation.py#L182) rotates `facing`, `axis` and
`rotation` but leaves directional property keys unchanged. A north-connected fence rotated one quarter
turn remains `north=true`; it should become east-connected. Rails and other orientation-bearing states
also need block-specific rules or explicit unsupported status. Only four Y rotations are supported;
do not describe that as all allowed lattice-axis rotations without declaring the gravity restriction.
`visible_breakdowns` groups by name, so different exact states are merged under a misleading label.

Publish full-state breakdowns and direct versus aligned visible metrics. Close remaining provenance
gaps: effective settings outside `raycast`, alignment timing, and process/native versus Python heap use.
Most importantly, the reports still do not score the preserved pre-M2 and current volumes under one
corrected evaluator/mask policy. The lighting correction runs are unmasked; the R6 runs are masked and
predate that correction. They cannot be combined into a matched quality comparison.

### B5 — High for trusting M3 evidence, Claude: pose evaluation uses incompatible conventions

[`relative_pose.py:330`](../../scripts/relative_pose.py#L330) compares COLMAP's image-down camera
rotation to the project's image-up camera rotation. With `D=diag(1,-1,1)`, the reference relative
rotation is `D @ (R2 @ R1.T) @ D`. A perfect COLMAP rotation for yaw/pitch `(0,0)` to `(30,10)`
receives **20 degrees error** in the current scorer.

Translation is also compared in the wrong frame. For `cam2_from_cam1`, expected translation is
proportional to `R2_cv @ (C1-C2)`, not `R1_up @ (C2-C1)`. Missing metric scale does not by itself
make the sign unobservable after cheirality selects a solution. Verify both directions with independent
synthetic correspondences and the actual installed API, and republish the stored-pair scores.

The corrected stored rotation errors still give 0/41 within 5 degrees (fixture minima 23.49, 25.78,
31.24 degrees for Templates 2, 6, 3). I accept stopping these three generic matching configurations.
That evidence does not rule out orientation-constrained or structural matching. Current artifacts still
mark bad pairs `resolved=True` and graph-connected views `registered`; label library hypotheses versus
validated poses distinctly and add an image-derived consistency/abstention check before consumption.

### B6 — Medium, Claude: seam scoring is not yet a general acceptance harness

[`seam_landmarks.py:200`](../../scripts/seam_landmarks.py#L200) rebuilds a candidate from FOV and
ignores its measured principal point. A perfect 100x100 cropped camera, focal 50 and centre `(30,50)`,
on two points at depth 4 is scored **20 raw pixels / 384 normalized pixels wrong**. Carry all intrinsics.

The freeze uses `next(glob('*.litematic'))`, unions visible cells across frames, and samples all eight
cell corners without checking whether each particular corner is a visible seam in that frame. Scoring
does not verify the stored digest; it skips missing cameras/estimates and nonprojecting points, and
can pass on the surviving population without a coverage requirement. Add explicit reference selection,
per-frame landmark visibility, immutable input/calibration/source hashes, expected frame membership,
digest verification, and failed-projection/missing-frame handling that cannot improve a score.

The reproduced 15-frame diagnostic remains valid evidence of small uncropped-camera reprojection
differences with supplied translation, not proof that these harness gaps are harmless in general.
Complete the frozen `t3-local-3` and `t3-local-5` coverage too; the maintained result covers `t3-sparse-8`.

### B7 — Medium, Claude: an impossible known-length constraint can return a plausible distance

[`lattice.py:154`](../../src/mc_photogrammetry/reconstruction/lattice.py#L154) accepts opposite signs
of ray/plane incidence using `abs(denominator)`. Points on opposite sides of that plane's vanishing
line cannot both hit the same plane in front of the camera.

Reproduction: yaw/pitch 0 axis basis, normal axis 0, focal 500, 800x800 image, points `(300,400)` and
`(500,400)`, length 2: `scale_from_known_length` returns **0.2**, although one intersection is behind
the eye. Require finite inputs, a consistent forward-facing plane side and adequate conditioning.
Carry inconsistent-constraint status instead of a distance. The real ceiling's stated-length accuracy
needs a maintained invocation and point/input provenance; the current entry points do not expose it.

## Next directives and milestone boundaries

**Terra:** first fix B1/B2 and finish exact-model sampling; do not compensate for invalid evidence with
lighting thresholds. Next close B3/B4 and publish a matched baseline table for Template 2/6 with one
reviewed mask/evaluator and explicit light-cast ablations. Include the baseline's 13 lost Template 2
exact matches in the investigation. The existing correction restores Template 6's direct count from
0 to 613, but that is not root-cause isolation or a transferable lighting validation. Keep its toggle
and measure the effect after evidence sampling is corrected. M2 acceptance still precedes GPU work.

**Claude:** first correct B5–B7 with independent synthetic tests and maintained, hashed diagnostics.
No fourth generic PyCOLMAP configuration is requested. Then proceed with a bounded M4 prototype:
one image-derived plane region, image-derived orientation, and an optional user-stated length, producing
supported cells in a local plane frame. Report automatic versus assisted runs separately. Preserve
holes, unresolved depth/axis choices and alternatives; derive bounds from support. A local frame can
fix gauge arbitrarily, but relative placement of independent views cannot be invented.

Start on all three Template 2 images individually, with identical settings and predeclared criteria;
evaluate region precision/recall, lattice distance/phase and visible supported-cell counts. Use reference
regions only for scoring. Test same-material seams, a known-length case, an inconsistent constraint
and a genuinely ambiguous view. Join planes/views only after geometric support establishes their
relative placement. This moves useful reconstruction forward without pretending pose recovery exists.

**Both agents:** claim exact owned files in comms. Terra owns shared projection/types, renderer,
sampler and measurement; Claude owns calibration/lattice/surfaces and pose/seam experiments. Coordinate
shape-fitting edits before sharing that file. Stage named paths only; do not use `git add -A` in this
shared tree. Keep the handoff concise, link dated reports, and reserve milestone acceptance for review.
