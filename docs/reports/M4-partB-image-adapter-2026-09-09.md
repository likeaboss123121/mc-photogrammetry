# M4 B, complete: image-polygon adapter, reviewable bundle, real-image pilot

**Claude Sonnet 5, 2026-09-09.** Completes Astra's kernel-review assignment
(`docs/reports/M4-B-kernel-review-2026-09-09.md`), steps 1-6. Steps 1-5
(the adapter itself, the synthetic loop closure, the reviewable bundle) were
committed at `c27747b`; this report covers a bug found after that commit and
the real-image pilot (step 6).

Full suite: 428 tests, green.

## A real bug found before it reached the real pilot

While preparing the real-image annotation - a broad visible-extent polygon
for the whole wall, with a smaller occlusion polygon for the foreground
object inside it - `rasterize_cells`'s label selection turned out to depend
on which order `regions` happened to be listed in. A cell where both a broad
`VISIBLE` region and a smaller, more specific `OPENING`/`FOREGROUND_OCCLUSION`
region both cleared the 60% coverage threshold picked whichever label the
caller listed first, not the more specific claim. Fixed: `OPENING` and
`FOREGROUND_OCCLUSION` are now checked before `VISIBLE`, since a small region
drawn inside a larger one is a person overriding the broad claim for that
specific area, not a coin flip. New test
(`test_occlusion_wins_over_a_broader_overlapping_visible_region_regardless_of_order`)
checks both region orderings give the same answer.

> Lead follow-up: image adapter/bundle delivered; coverage, exported axis mapping and assumed-phase
> provenance require corrections. See [current acceptance and conditional C directive](LEAD-review-2026-09-09.md).

## The real-image pilot

Annotation: [`M4-partB-pilot-annotation-2026-09-09.json`](M4-partB-pilot-annotation-2026-09-09.json),
frozen at `9621bc5a07b1c69f649a8e3792128708157c16c50ef9c09135a5d87078248978`
before scoring. Corrections from the superseded annotation, per review:

- `visible_extent_polygon` now starts at y=130 (original px) - clear of the
  F3 debug text block (roughly y=0-122), which the old polygon overlapped.
- One frame, one plane - not three frames of best-effort guesses.
- The stated length keeps the same landmark method (the wall's repeating
  decorative motif, one motif assumed per block) but is recorded with an
  explicit **low confidence** flag rather than asserted as precise - the
  prior attempt at this exact landmark, once overlaid, visibly spanned about
  one motif rather than the intended two-to-three, and I cannot verify
  pixel-perfect motif edges from a static, non-zoomable image view. Per the
  assignment's own permission: report the ambiguity rather than invent a
  falsely precise number.

Reproduce: `python scripts/footprint_pilot_real.py --out-dir <dir>`.

**Result:** axis 2, side +1, distance 5.750 blocks (low confidence). 14x10
candidate grid, **53 supported, 87 unknown, 0 opening**. Independently
re-verified after the fact (not just trusted from the run): every one of the
53 supported cells' projected pixel centre falls inside the declared
extent-polygon mask, and none falls inside the occlusion mask - checked
programmatically against the actual rasterized regions, not eyeballed from
the overlay alone.
[`overlay.png`](M4-partB-pilot-templates-2-2026-09-09/overlay.png) shows the
candidate grid, extent/occlusion regions, and the stated-length line labelled
with its confidence, drawn directly on the photograph.
`pilot.litematic` (gitignored, reproducible) holds the 53 solid placeholders;
`result.json` records the axis/side/distance, verdict totals, and cell
coordinates in the transform's own coordinate frame (not `footprint.py`'s
internal 0-indexed one - an earlier ad hoc verification script tripped on
exactly that distinction, which is why `result.json` now states it plainly
and gives cells already converted).

## What this does and does not establish

**Does:** the full geometric chain from an image-drawn polygon on a real
photograph to placed, exported geometry, using image-derived camera optics
throughout and never reading the manifest until this report's own
description of the result. The occlusion/visible priority fix and the
result.json coordinate-frame fix were both found by independently
re-verifying the pilot's own output against the source annotation, not by
trusting the run.

**Does not:** establish that 53 is the *right* number of supported cells, or
that the distance is accurate. The stated length is explicitly low-confidence
and the underlying per-cell texture cue that would normally cross-check this
kind of result was deliberately not consulted here (per the assignment: "A1's
seam-to-geometry verdict defect remains documented and quarantined from this
adapter"). This pilot demonstrates the *mechanism*, not a validated
reconstruction of this wall.

## Everything from the assignment, accounted for

1. Image-polygon-to-footprint adapter: `footprint_adapter.py`, reusing
   `plane_basis` and `footprint.py`'s placement, no seam/frequency voting.
2. Explicit, serializable geometric mapping: `LatticeTransform`, saved to
   `transform.json` with a content hash, reloaded and verified in a separate
   process (`footprint_pilot.py reload`).
3. Region rasterization with coverage and priority, not a boolean mask:
   `rasterize_cells`, `CellEvidence.coverage_by_label`.
4. Synthetic loop closed: the 4x3 control recovered from its own projected
   image regions, corner reprojection, fractional origin, both face signs,
   nonzero principal point, image clipping, and (structurally, not
   spot-checked) texture invariance - all in `test_footprint_adapter.py`.
5. Reviewable bundle: `footprint_pilot.py build`/`reload`, plus the 11-vs-10
   litematic metadata fix (`footprint_to_volume`'s
   `include_confirmed_openings`, default now `False`).
6. One real-image pilot, corrected per review, reported above.
7. Stopping here, as instructed - no fixture sweep, no detector tuning. C
   follows lead review of this complete path.
