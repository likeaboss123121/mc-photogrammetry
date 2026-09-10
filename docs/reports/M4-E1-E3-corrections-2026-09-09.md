# M4 B/C, third pass: E1-E3 corrections and a re-scored C2 audit

> Lead review: negative composition and the depth guard improve correctness. Annotation still removes visible wall, and approximate coverage turns true 55% coverage into a passing 61.2%; F3/F4 remain open. Full independent XYZ reprojection remains backlog. See [current review](LEAD-review-2026-09-09-fourth.md).

**Claude Sonnet 5, 2026-09-09.** Implements Sonnet's assignment in
[`LEAD-review-2026-09-09-third.md`](LEAD-review-2026-09-09-third.md) (E1 High, E2 Medium,
E3 Medium, plus the finite-support correction named under E3), then re-scores the already-saved
C2 candidate polygons against the corrected annotation - not a new proposer run, per the review's
explicit instruction.

Full workspace `.venv` suite: **499 tests, green** (this run also includes Luna's concurrent M5
work in the same tree; the M4 controls this report adds or changes are itemized per section below).

## E1 — the annotation now traces the wall's actual extent, and negatives are actually subtracted

Two distinct bugs, both reproduced exactly:

1. **`boundary_scoring.target_masks` never read `negative_polygons` at all.** They were recorded
   for documentation but had no effect on `positive_mask` - a declared occluder or opening
   overlapping the broad positive extent stayed classified positive. With the frozen v1 files this
   made 18,135 / 28,622 / 9,331 pixels the annotation itself called negative still score as
   positive across the three frames.
2. **Frame 1's `positive_polygon` still omitted real wall material.** The review's own example:
   source pixel (400,400) is visibly wall (verified directly against the pixel data) but fell
   outside the v1 rectangle, entirely because that rectangle was chosen for being easy to trace,
   not because the wall stops there.

Fixed both. `target_masks` now composes explicitly: start from `positive_polygon`, remove every
polygon in `negative_polygons` regardless of sub-category (a specific claim overrides the broad
one where they overlap - the same "more specific wins" rule `footprint_adapter`'s own region
composition already uses), then remove `uncertain_polygons`. A pixel a negative polygon and an
uncertain polygon both cover becomes unknown, not negative - an unresolved disagreement between
two specific annotations, not silently decided either way.

New annotation file **`M4-C2-boundary-targets-v2-2026-09-09.json`** (hash `19308640...fa475cd574`;
the v1 file is preserved untouched). `positive_polygon` on all three frames is now the wall's broad
silhouette from the frame's left edge, not a rectangle chosen to be easy to trace; frame 1's
foreground table is now removed as a precisely-traced negative polygon (its own diagonal edge,
read directly off a source-resolution crop) instead of shrinking the positive claim to avoid it.
Every frame carries `wall_interior_anchors` and `negative_anchors` - specific pixel coordinates
(including the review's own (400,400) example) independently checked against the composed masks
by `tests/test_boundary_scoring.py`, not just asserted in the file. A rendered mask overlay
(positive/negative/unknown, with anchors marked) was inspected before freezing; two anchor points
and one frame-2 negative polygon that turned out to clip real wall material on first attempt were
caught by that check and corrected before the hash was taken.

**Controls** (`tests/test_boundary_scoring.py`, 3 new): a negative polygon actually subtracted
from a broad positive extent (precision drops accordingly - checked numerically, not just via
mask equality), a negative/uncertain overlap resolving to unknown rather than negative, and an
absent `negative_polygons` key behaving identically to an explicit empty one.

**Re-scoring, not re-running:** per the review's instruction, `scripts/rescore_c2_v2.py` reads the
candidate polygons `boundary_pilot_c2.py` already saved to
`M4-C2-boundary-pilot-2026-09-09/results.json` and re-computes `region_metrics`/`boundary_error`
for each against the v2 target - it never calls `find_chains` or `propose_boundary_candidates`.
Result: **unchanged from C2 - 0 of 2 scorable frames pass, and `best_matched_evaluator_only` is
still exactly 0 on both.** The corrected, far more generous annotation does not change the
outcome, because the saved candidates never came close to the wall at all (they land on the
dining table/bench edges, entirely outside even the broad silhouette) - confirmed visually in
the [re-scored overlays](M4-C2-rescored-v2-2026-09-09/). This is a real confirmation, not a
retread: it rules out "the annotation was the problem" as an explanation for C2's result.

## E2 — coverage now converges to the true continuous area regardless of pixel-grid alignment

The review's exact counter-examples to the previous half-pixel coordinate shift: an identity-axes
cell of true area 1, at four different focal/principal-point combinations, none integer-aligned.
Reproduced exactly - `4.0`, `4.1667`, `1.5` for cells whose true area is 1 (the fourth,
grid-aligned case, correctly read `1.0`). The shift only reconciled raster and continuous area
for the one case where a cell's corners happen to land on integer pixel-grid coordinates; general
sub-pixel positioning was never actually integrated.

`footprint_adapter.py`'s per-cell rasterization is rewritten around local supersampling: within
one cell's own bounded crop (never the whole image, and never a declared region's own potentially
enormous extent - `_clip_polygon_to_rect`, Sutherland-Hodgman, bounds every region polygon to the
crop before it is rasterized), each pixel is subdivided into `supersample^2` samples
(`_supersample_factor`, adaptive: finer for a small typical cell, coarser for one that is already
most of the image, capped so the total sample count for one cell never exceeds a fixed budget
regardless of crop size). Counting sub-pixel hits and dividing by `supersample^2` converges to the
true continuous intersection area independent of alignment; `full_area` stays the exact shoelace
value. `_label_masks` (whole-image, single-sample) is replaced by `_local_supersampled_masks`
(per-cell, local, supersampled) - the global precomputation this replaces was itself a source of
imprecision the review's examples exposed.

Reproduced fixed: the four cases now read `1.063`, `1.063`, `1.058`, `1.031` - all within about 6%
of the true value 1.0, all correctly declared `visible`, regardless of alignment. This is now an
approximation (converging with the supersample factor), not an exact match even for the
pixel-grid-aligned case - the existing exact-reproduction test's tolerance was loosened
accordingly, honestly reflecting what the fix actually provides.

**Controls** (`tests/test_footprint_adapter.py`, 2 new): the review's exact four counter-examples
in one parametrized control, and a cell translated by a non-integer amount in both axes
throughout (not just offset from one integer origin). The existing mixed-label, clipped, large-
cell, sub-pixel and extreme-grazing controls from the previous two passes were re-run unmodified
and still pass, including the grazing cell's sub-2-second timing bound - the adaptive supersample
factor falls back to no subdivision for a crop that size, keeping the original bound intact.

## E3 — read-back verification now checks the block's own depth, not just its lateral position

`reproject_exported_volume` read every block the file actually contains, but immediately passed
it through `block_to_cell`, which - correctly, for its own purpose - keeps only the two lateral
coordinates and discards the normal-axis one. A block at the right lateral position but the wrong
depth (a corrupted file, or a nonplanar extrusion) would still produce a valid-looking cell,
since nothing ever checked the coordinate that would have caught it. The review's own control:
loading a real exported/read-back snapshot, then changing one block's normal-axis coordinate
after loading - accepted with an identical result before this fix.

Fixed: every read-back block is now checked against the plane's own declared normal-axis
placement (`cell_to_block`'s own `axis_value` formula: `face_coordinate - 1` if `side > 0` else
`face_coordinate`) *before* it is reduced to a 2D cell at all. A block that fails this is not a
supported cell with a slightly-off corner - it is not on the declared plane, and
`reproject_exported_volume` now raises rather than silently accepting it. Also addressed: an
immutable export transform should be sufficient for a fresh process to re-verify without
recomputing anything from the pre-export footprint. `reproject_exported_volume` now accepts a
`minimum_shift` parameter; when given, nothing is recomputed from `footprint`'s expected cells.
`footprint_pilot.py reload` (already a separate-process command) now exercises this directly:
it reads `minimum_shift` back out of the persisted `export_transform.json` and re-verifies
`control.litematic` with it, rather than only checking `transform.json`/`regions.json` as before.

**Finite-support correction** (named under E3, closing a gap from the S1 pass): `find_chains`
bridged every gap between a chain's member segments by taking the pooled min/max of all their
endpoint projections, so `_corner_supported` treated a real gap between two detections as if it
were itself observed. `Chain.span` is now the union of each segment's own interval, merged only
where they actually touch or overlap (within `chain_gap`) - a chain built from two segments 100px
apart along the line now carries two separate intervals, and a corner falling in the gap between
them is correctly rejected as unsupported. `_corner_supported`'s margin is now a flat tolerance
per interval rather than scaled by that interval's own length, which had been quietly wide enough
to bridge exactly this kind of gap on its own.

**Controls** (`tests/test_boundary_candidates.py`, 2 new; 1 existing test corrected to check the
new split-interval behaviour instead of a single bridged envelope): a corner sitting squarely in
the gap between two real observed intervals is rejected while corners inside either interval are
accepted, and the broken-edge-with-a-gap control now asserts two separate intervals rather than
one envelope spanning the gap. (`tests/test_footprint_adapter.py`, 2 new): the review's exact
loader-injection reproduction (a genuine file verifies cleanly; the same file with one block's
depth shifted after loading is rejected with a clear error naming the mismatch), and a
persisted-`minimum_shift` control confirming a second call needs nothing recomputed from the
footprint.

## What this does and does not establish

**Does:** the annotation now traces what the review directly demonstrated was missing, verified by
anchor points a test checks rather than only asserted in prose. The negative-polygon composition
bug is fixed and the fix is shown to matter (a numeric precision drop, not just a changed mask).
Coverage measurement now converges to the true continuous area regardless of pixel-grid alignment,
verified against all four of the review's counter-examples. Read-back verification now genuinely
checks 3D placement, not just a 2D cell identity a corrupted depth could not affect. The C2 result
is now confirmed against a materially more correct and more generous target, by re-scoring the
same saved candidates rather than re-running anything - ruling out "the annotation was too small"
as an explanation for the 0/3 result.

**Does not:** run a new proposer configuration. Per the review's explicit instruction, this turn
re-scores what C2 already produced; whether a corrected proposer (a C3, seeded or cued
differently) could do better is not addressed here and no such experiment is authorized by this
report.

## Next

Returning this for review.
