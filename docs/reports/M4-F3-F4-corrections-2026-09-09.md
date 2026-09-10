# M4 B/C, fourth pass: F3 annotation-workflow repair and F4 certified coverage bounds

**Claude Sonnet 5, 2026-09-09.** Implements Sonnet's S-A/S-B assignment in
[`docs/implementation-next.md`](../implementation-next.md), specified further by
[`LEAD-review-2026-09-09-fourth.md`](LEAD-review-2026-09-09-fourth.md)'s F3/F4 findings.

Full workspace `.venv` suite: **522 tests, green** (`python -m pytest tests/ -q -p no:cacheprovider
-o addopts=`, 513 baseline + 9 new F4 controls; the F3 anchor test is counted in
`test_boundary_scoring.py`'s total separately below).

## F3 — the annotation was rebuilt from a labelled source-coordinate grid, not patched at two points

The review's finding was explicit that moving the two cited vertices would not be enough: "Change
the annotation workflow rather than moving a few vertices to pass these two new anchors." This pass
did not touch the v2 JSON at all. Instead:

1. **Source-coordinate grid, not a display crop.** Every crop used to trace a boundary
   (`docs/reports/M4-F3-annotation-audit-2026-09-09/*.png`) has its grid lines and coordinate labels
   drawn directly onto the *original* 2560x1369 frame, in that frame's own pixel coordinates,
   *before* any resize - `scripts` in the audit directory upscale only after labelling, so an
   upscale can never shift what coordinate a line is labelled with. Forward mapping was checked at
   the image's own top-left and bottom-right corners (`f1_corner_check.png`,
   `f1_corner_check_br.png`) before any tracing began.
2. **Traced from the image, not from a target.** Every boundary below was read directly off these
   grids - at 50px, then 25px, then 10px spacing where a vertex needed sub-25px localization - and
   the resulting polygon was checked against the composed mask only afterward, as a bug check on the
   JSON encoding, never as a way to choose where a vertex goes. `docs/reports/M4-F3-annotation-audit-2026-09-09/`
   keeps every crop plus a `contact_sheet_frame{1,2,3}.png` tiling them, a `v3_composite_*.png` full
   compositing of the frozen result with every anchor drawn on it, and the exact crop provenance
   (source bounds, grid step, upscale, purpose) recorded per-crop inside the new JSON itself.
3. **All three frames audited, not just the named example.** The review's own citation was frame 1's
   table; re-tracing frame 2 the same way found the *same class* of bug independently - v1/v2's
   furniture boundary there started at y=650 for x:600-980, but the real wall/bench transition runs
   to y:825-970 depending on x (`f2_transition_fine.png`, `f2_left_transition.png`), wrongly removing
   a comparable amount of real wall. Frame 3 had the opposite-direction problem: its positive claim
   stopped around y:600-750 even though this steep, close-up downward-pitched view genuinely shows
   wall filling the frame to y=1369 for x<1080 - confirmed with a gamma-boosted re-crop
   (`f3_lower_bright.png`, gamma 0.4) after the raw screenshot was too dark to distinguish wall
   texture from shadowed floor by eye alone.

New file **`M4-C2-boundary-targets-v3-2026-09-09.json`** (sha256
`082ec648845be2354a5324821e6484291ca867563948e0396cecb496e5adf56a`; v1 and v2 are preserved
untouched). Per-frame changes:

- **Frame 1**: table negative retraced to its true corner - approximately
  `(0,800) -> (788,790) -> (850,1040) -> (862,1369)` - instead of the v2 rectangle that started at
  `(0,435)`. The review's own two pixels, (400,600) and (600,600), are now positive as a
  *consequence* of this trace, not a target it was built to hit; six wall-interior and seven negative
  anchors are placed on both sides of every retraced boundary (table, dining set, two candle
  sconces, roof-gap diamond), each with a written reason. Two anchor candidates that landed in the
  pre-existing corner-uncertainty band (a specific-negative-and-specific-uncertain overlap correctly
  composing to unknown, not a bug) were swapped for cleaner single-label demonstration points.
- **Frame 2**: positive silhouette's lower boundary re-traced from
  `(0,955) -> (300,970) -> (500,930) -> (700,905) -> (850,850) -> (980,825)` (was a flat cut around
  y:650-750). `x>980` was confirmed to be a *different* surface - the room's far gable-end wall with
  its own roof peak and dining table - and deliberately kept out of the positive claim on that basis
  rather than merely being distant from camera. A wall/negative anchor pair sits at the same x (500
  and 700) on both sides of the corrected boundary, directly demonstrating the fix: under the old v2
  boundary, `(700,700)` would have read negative; it is now confirmed positive.
- **Frame 3**: positive extent widened from a `y:600-750` cutoff to the frame's true extent - wall
  fills to `y=1369` for `x<1080` at this angle, confirmed by direct pixel inspection after brightness
  correction, not assumed from the pitch angle alone. The furniture in the bottom-right corner and a
  genuinely ambiguous ceiling strip at the top (`y<100` for `x>650` - not a roof opening; none is
  visible from this angle, matching the original description) are the only exclusions.

**Controls**: `tests/test_boundary_scoring.py::test_f3_v3_annotation_anchors_match_the_composed_masks`
freezes the v3 file's hash and checks every `wall_interior_anchors`/`negative_anchors` point across
all three frames resolves to the composed label it claims, plus the review's exact `(400,600)`/
`(600,600)` pixels by name (not just position in the anchor list). `scripts/rescore_c2_v3.py`
re-scores C2's already-saved candidates against v3 (never re-running `find_chains`/
`propose_boundary_candidates`) and keeps the three-frame denominator per the review's instruction
("one no-candidate frame is a failure, not omitted evidence"): **0 of 3 - unchanged**. Frame 2's
saved candidate now measures non-zero precision (0.668, was 0.000) against the corrected, much larger
true wall area, but recall stays at 0.018 - the candidate still covers only a sliver of the real
wall - so the gate result does not change. Frame 3 still has zero saved candidates. This confirms
again, against a materially different (not just more generous) annotation, that C2's failure is not
an annotation artifact.

**Does not**: run a new proposer, cue, or SfM pass - none was authorized this turn. `M4-C2-rescored-v3-2026-09-09/`
holds the re-scored results and overlays; `M4-C2-rescored-v2-2026-09-09/` from the prior pass is
untouched.

## F4 — coverage decisions are now certified lower/upper bounds, not a raw estimate

The review's own exact counter-example, reproduced first: identity-axis Z plane,
side/distance/period=1, focal=1, principal point (2.2,3.2), cell (0,0) - projected cell
[2.2,3.2]x[2.2,3.2]. A VISIBLE rectangle covering only 55% of its width read **0.6123046875** under
the previous supersampled point estimate and declared the cell visible at the 0.6 threshold, despite
true coverage (55%) being genuinely below it - not a cosmetic discrepancy, an unsupported cell
exported. The review named the root cause precisely: "Raw supersample estimates are not confidence
bounds."

`footprint_adapter.py`'s per-cell measurement is rewritten around a rigorous conservative interval,
not a finer point estimate:

- **`_fill_and_boundary`** draws each polygon's ordinary point-sampled fill (`cv2.fillPoly`, as
  before) *and* a `BOUNDARY_BAND_SUBCELLS`-wide band around every one of its edges (`cv2.polylines`).
  Away from that band, a straight edge cannot pass through a sub-pixel without crossing the band
  around it - by the Jordan curve theorem, a sub-pixel disjoint from every edge is either entirely
  inside the polygon or entirely outside it, regardless of concavity, so its sample there is not
  probably right, it is *certified* right. `fill & ~boundary` is a certified lower-bound set;
  `fill | boundary` is a certified upper-bound set. This holds for a union of several polygons per
  label too (a sub-pixel away from every one of their bands has an unambiguous status for each).
- **`_tile_area_sums`** combines the cell's own certified cell/label sets through the same
  specific-overrides-general exclusivity `_local_supersampled_masks` used before (VISIBLE minus
  OPENING minus FOREGROUND_OCCLUSION, conflict where OPENING and FOREGROUND_OCCLUSION both certainly
  or possibly overlap), now propagating certain/possible through that composition rather than a
  single mask.
- **`rasterize_cells`** sets `coverage_by_label` to the certified *lower* bound and the new
  `coverage_upper_by_label` field to the certified upper bound. `declared` is only ever set from the
  lower bound clearing `VISIBLE_COVERAGE_THRESHOLD` - a declared label's true coverage is therefore
  *guaranteed* to clear the threshold. A cell whose true coverage sits close enough to the threshold
  that this cannot be certified either way - including exactly at the threshold, where a conservative
  lower bound can in principle never clear its own cutoff - stays undeclared, per the review's own
  "abstain if an approximation cannot certify it."
- **Bounded allocation, including when a crop alone exceeds the budget.** The review named a second,
  narrower gap directly: "the current factor>=1 fallback does not cap a crop already larger than
  four million samples." `_supersample_factor` now returns 1 once a crop's own unsupersampled pixel
  count already reaches `MAX_SUPERSAMPLED_CELLS`, and `rasterize_cells` processes that case in
  bounded row-bands (`_tile_row_bounds`), accumulating each band's own area sums - the full crop's
  mask is never one array, however large the crop is. A crop whose own *width* alone would still
  exceed the budget in a single row leaves the cell explicitly unresolved (`declared is None`, no
  coverage claimed) rather than allocating past the bound.
- **`MAX_SUPERSAMPLE_FACTOR` raised from 32 to 2000** (a safety ceiling now, not a target): finer
  subdivision directly tightens the certified interval under this scheme (unlike the old point
  estimate, where the review's own reasoning for capping it at 32 - "diminishing returns" - no longer
  applies), and `_supersample_factor`'s own budget formula already bounds total cost regardless of
  the ceiling.
- **`in_frame`/`outside_image_fraction`** now come from exactly clipping the cell's own (convex)
  corners against the image rectangle (`_clip_polygon_to_rect` + shoelace, both convex, so this is
  exact, not a bound) - decoupled from the certified-coverage machinery, which would otherwise never
  report a cell as *exactly* fully in-frame even when it plainly is, since its own boundary margin
  applies uniformly regardless of proximity to the image edge.

Reproduced fixed: the review's 55%-width case now reads a certified lower bound of **0.3482** (at the
tiny, toy 1-pixel cell size the review's own example uses - a case this small cannot get a tight
bound with any bounded method, but the true value, 0.55, is correctly never exceeded and the bound
correctly stays below the 0.6 threshold either way) and the cell correctly stays undeclared.
At a well-resolved, ordinary cell size (~100px), the same 55% case reads **0.5452**, within 0.5% of
true.

**Controls** (`tests/test_footprint_adapter.py`, 9 new): the review's exact 55%-width/1-pixel-cell
reproduction, confirming it never certifies visible; 55/59/60/61/100% coverage at a well-resolved
cell size, confirming 55/59% correctly stay undeclared, 61/100% correctly declare visible, and
**exactly 60% also stays undeclared** (an approximation cannot certify a value sitting exactly on its
own threshold, per the review's own wording); upper-bound-never-undershoots-lower-bound soundness
across seven coverage fractions; the same 59/61% reject/accept check at a translated, fractional
(non-pixel-grid) principal point and focal; mixed-label lower/upper bounds staying within [0,1] for a
VISIBLE/OPENING split cell; a crop exceeding `MAX_SUPERSAMPLED_CELLS` before any subdivision (9M
pixels, a near-full-frame cell) completing in bounded time via row-band tiling rather than allocating
the whole crop, with the full-coverage case still reading within 1% of 1.0; and the pathological
width-alone-exceeds-budget case (exercised by lowering the module's own budget) leaving the cell
explicitly unresolved rather than allocating past it. Four previously-loosened tests
(`test_s3_pixel_aligned_unit_cell...`, `test_e2_area_units_stay_consistent...`, `test_e2_a_translated_fractional_cell...`,
`test_g1_a_cell_mostly_outside_the_image...`) needed their tolerances revisited again - not loosened
to paper over a failure, but corrected to state the new, different, and now honestly weaker guarantee
a 1-pixel cell's certified bound can give (a tiny cell's lower bound legitimately sits further from
1.0 than a well-resolved cell's does, because the same fixed boundary margin is a much larger share
of a 1-pixel cell's own area) - each documents why in its own docstring rather than silently widening
`abs=`.

**Does not**: change the accepted depth guard (E3) or claim a complete 3D export transform - both
remain as the third pass left them, per the review's own note that general saved-affine XYZ
verification is backlog and does not interrupt this repair.

## Next

Returning this for review.
