# M4 B/C, second pass: S1-S4 corrections and a corrected C2 run

**Claude Sonnet 5, 2026-09-09.** Implements Sonnet's assignment in
[`LEAD-review-2026-09-09-followup.md`](LEAD-review-2026-09-09-followup.md) (S1 High, S2 High,
S3 Medium, S4 Medium), then runs the corrected C2 boundary experiment the review pre-authorized
once S1/S2's own controls pass.

Full suite: **489 tests, green** (up from 465 at the reviewed commit `e842c0e`; 24 new
regression tests across `boundary_candidates.py`, `boundary_scoring.py` and `footprint_adapter.py`).

## S1 — the boundary proposer now pairs opposite edges within one axis, not by image centre

The review's exact reproduction: a 100x100 frame, vertical chains at x=10 and x=30 (both left of
centre), horizontal chains at y=20 and y=80. `propose_boundary_candidates` split chains into
"left"/"right" buckets by whether their position was left or right of the image *centre* - both
verticals landed in "left," and "right" silently fell back to the image's own edge (x=100),
producing candidates spanning x=10..100 and x=30..100 instead of the true (10,20)-(30,80)
rectangle. A second, related defect: candidates could pair one line from one calibrated axis
against a line from a *different* axis as if they were a plane's two opposite sides, which they
are not - two different world-axis pencils are not parallel to each other in general, even when
they look similarly oriented in one image.

`src/mc_photogrammetry/reconstruction/boundary_candidates.py` is rewritten around this: a
candidate's two pairs of opposite edges each come from **one calibrated axis's own pencil** of
chains (`_edge_pairs`, all `C(n,2)` distinct pairs within that axis, ordered along the pencil, not
role-assigned by position), combined across every pair of distinct axes present. `Chain` gained a
genuine along-line `span` (endpoint projections onto its own direction, not the offset-clustering
spread the field previously held) and an `offset` field for ordering pairs within a pencil.
`_build_quadrilateral` rejects rather than silently patches: a degenerate/parallel intersection,
a self-intersecting corner order (`_is_simple_quad`), a corner sitting outside a generous frame
margin, or a corner requiring extrapolation far past a real chain's own observed extent
(`_corner_supported`, replacing the old "substitute (0,0)" and "clamp each vertex independently"
behaviour) all reject the whole combination instead of returning a distorted one. A valid
combination is geometrically clipped to the image rectangle (Sutherland-Hodgman) only after
passing every validity check, not before. Missing a second chain for one axis still falls back
to that side of the image frame, but the resulting `BoundaryCandidate.complete` is `False` and
`edges[...]["low"/"high"]` is `None` - an explicit, visible fallback, not indistinguishable from
a real detection. Both `find_chains`' chain cap and `propose_boundary_candidates`' candidate cap
are now distributed across axis and a coarse 3x3 image-region bucket (`_distribute`) rather than
a pure global top-N, so one cluttered region or axis cannot use up every slot.

`propose_boundary_candidates` also accepts an optional `stats` dict, filled with
`combinations_tried`/`accepted`/`rejected` (by reason) - diagnostics S2's report below uses,
never consulted by the ranking itself.

**Controls** (`tests/test_boundary_candidates.py`, 15 tests): the review's exact off-centre
reproduction, off-centre rectangles in all four quadrants, a rolled-perspective variant of the
synthetic-rectangle control, two lines from different axes never pairing as opposite edges,
one axis alone yielding no candidates rather than a guess, near-parallel cross-axis lines
rejected instead of intersected far away, a corner whose supporting chain's real extent is too
short getting rejected, a broken edge with an along-line gap still merging into one chain, and
two separate synthetic planes both recoverable among the candidate set.

## S2 — the C evaluation target and scoring are corrected

The frozen C1 target (`M4-C-boundary-targets-2026-09-09.json`) described a small, deliberately
safe strip well inside the visible wall - not the wall's actual boundary. Pixels outside that
strip were neither confirmed positive nor negative, but `_region_metrics` counted all of them as
false-positive area against any candidate that covered them, and the reported boundary error
compared the original declared extent polygon rather than the domain-clipped region actually
scored.

New target file **`M4-C2-boundary-targets-2026-09-09.json`** (hash `2b116093...9ea982306`, frozen
before scoring; preserves C1's file untouched as, in the review's words, "an invalidly targeted
historical diagnostic"): each frame's wall is re-annotated at source resolution (cropped zooms,
not the resized preview) with three explicit categories - `positive_polygon` (confidently this
wall's own material, now the wall's actual extent rather than a safe interior strip),
`negative_polygons` (confidently not this wall: occluders, an opening, kept for documentation/
overlay clarity though they do not change the score, since anything non-positive/non-unknown in
the domain is already a negative by construction), and `uncertain_polygons` (genuinely ambiguous
transitions - a possible corner/support-beam junction I could not resolve confidently even at
higher zoom on frame 1, the roof-gap junction on frames 1/2 - excluded from scoring entirely,
never guessed either way).

New module **`boundary_scoring.py`** (separate from the proposal path on purpose, mirroring the
review's "target geometry... remain unavailable to the proposer"): `usable_domain` (HUD/hotbar
exclusion), `target_masks` (positive/unknown, both domain-clipped), `region_metrics`
(precision/recall/IoU with unknown pixels excluded from both sides), `mask_boundary` +
`boundary_error` (symmetric mean nearest-boundary distance via `cv2.distanceTransform`, L2,
normalized by the image diagonal - computed from the actual *scored* region masks, domain-clipped
and unknown-excluded, not the raw declared polygons). Five analytic controls in
`test_boundary_scoring.py`: a perfect candidate scores 1/1/1, a disjoint candidate scores 0 without
error, the HUD/hotbar bands are excluded from both sides, an uncertain region is excluded from
both sides (checked against what a naive negative-by-default score would have been, to confirm
the exclusion does something), and a partially-occluded positive region still scores a covering
candidate correctly (precision drops for a candidate that instead covers the old, larger bounding
box).

**`scripts/boundary_pilot_c2.py`** derives its frame list from the calibration artifact and each
image's own `width`/`height` from `cv2.imread` - never from the target file, which stays unopened
until scoring. It crops to the usable domain *before* calling `find_chains` (the review named this
directly: masking the HUD band in place, as C1 did, draws the mask's own straight edge as a
strong, fully artificial line for the detector to find; cropping removes the pixels instead of
drawing a border around them) and shifts the principal point and every detected chain back by the
crop offset before scoring. It reports the algorithm's own top-strength candidate (`top1`)
separately from `best_matched_evaluator_only` (the highest-IoU candidate anywhere in the returned
set - an honest "does a correct answer exist among the alternatives" diagnostic, explicitly never
used to pick the output), and the local continuation gate is evaluated on `top1` alone.

## C2 result

Run: `python scripts/boundary_pilot_c2.py --out-dir docs/reports/M4-C2-boundary-pilot-2026-09-09`.
One configuration, run once, on all three T2 frames - same caps as C1 (12 chains, 8 candidates),
no threshold sweep.

| Frame | Chains | Combos tried | Accepted | top1 P/R/IoU | Boundary err | best_matched IoU | Gate |
|---|---|---|---|---|---|---|---|
| `13.22.14` | 12 | 87 | 1 | 0.000 / 0.000 / 0.000 | 0.099 | 0.000 | fail |
| `13.24.24` | 12 | 108 | 10 (8 returned) | 0.000 / 0.000 / 0.000 | 0.124 | 0.000 | fail |
| `13.24.33` | 12 | 108 | 0 | - (no candidates) | - | - | fail |

**The local continuation condition is not met: 0 of 3 (2 scorable) frames pass.** Unlike C1, this
is not a phantom result papering over a pairing bug - `best_matched_evaluator_only` (the best IoU
anywhere among all returned candidates, not just top1) is also exactly 0 on both frames that
produced candidates at all. No candidate this configuration proposed, ranked first or not, lands
on the annotated wall.

**Why:** the overlays
([`13.22.14`](M4-C2-boundary-pilot-2026-09-09/overlay_2026-09-02_13.22.14.png),
[`13.24.24`](M4-C2-boundary-pilot-2026-09-09/overlay_2026-09-02_13.24.24.png)) show every returned
candidate landing on the room's dining table/bench edges, not the wall panelling. This is a
genuine Manhattan-world interior with many strong, valid structural lines that are not this wall's
boundary - the wood-grain table edges are longer and higher-contrast than the wall's subtler
panel seams, so `strength` (contrast-weighted length, unchanged by this fix) ranks them first.
Corrected pairing (S1) means these candidates are now geometrically *valid* quadrilaterals rather
than degenerate slivers - the rejection breakdown (`self_intersecting`/`corner_unsupported`/
`out_of_frame` counts in `results.json`) shows the mechanism actively discarding many genuinely
invalid combinations - but validity is not the same claim as correctness, and among the valid
ones, furniture consistently out-scores wall.

Per the review's explicit instruction, this stops here: no threshold sweep, no region-of-interest
seed (which would read the target - the thing S2 exists to prevent), no texture-occupancy cue, no
SfM restart. The mechanism (S1) and the evaluation (S2) are both independently correct now,
verified by their own controls; the outcome they jointly and honestly produce on this fixture,
with this one configuration, is a fail.

## S3 — full-cell coverage no longer mixes continuous and raster area units

The review's exact reproduction: identity axes, a cell projecting to the pixel-aligned unit square
(2,2)-(3,3) (true continuous area 1), fully covered by a matching `VISIBLE` polygon -
`coverage_by_label` returned `4.0`. Cause: `_rasterize_polygon` fed continuous corner coordinates
straight to `cv2.fillPoly`, which treats an integer vertex as a pixel's own centre and includes
boundary pixels - a unit square with corners at continuous `(2,2)` and `(3,3)` filled pixel-centres
2 through 3 inclusive in both axes, a 2x2, four-pixel block, while the shoelace-based denominator
(introduced for G1) correctly measured 1.

Fixed by shifting every coordinate by half a pixel before rounding (`round(x - 0.5)`), aligning
the two conventions: continuous corner `2` becomes centre-grid `1.5`, correctly falling between
pixel centres 1 and 2 instead of landing on one. Applied once, inside `_rasterize_polygon`, so
both the cell mask and every label mask use the same corrected convention. Separately, a cell
whose true area is below one pixel cannot be measured by counting whole raster pixels at all - any
non-zero raster count is already several times its true area - so `rasterize_cells` now never
declares a cell whose `full_area < RESOLVABLE_AREA_PIXELS` (1.0), regardless of its raw ratio:
"retain uncertainty below resolvable area," per the review, rather than report a number
rasterization at that scale cannot back up.

**Controls** (`tests/test_footprint_adapter.py`, 5 new): the review's exact pixel-aligned
reproduction (now exactly 1.0), a genuinely sub-pixel cell (never declared, regardless of its raw
ratio), a large (~2500px) cell (still ~1.0, confirming the fix holds at ordinary scale, not just
the 1px toy case), two disjoint labels within one cell summing to no more than the cell's own area
(the property this fix exists for, checked directly rather than only via a full-coverage case),
and a milder 50%-clipped cell alongside the existing 90%-clipped G1 control.

## S4 — read-back verification now reprojects the file's own blocks, not the pre-export footprint

An earlier version of this check (from the G1/G2/G3 pass) read the exported litematic back only
to confirm *which* cells came back (a set-equality check against the pre-export footprint), then
recomputed pixel corners by round-tripping `cell_to_block`/`block_to_cell` on the *pre-export* cell
set again - the file's content never drove the actual corner computation, so a bug that happened
to leave the recovered cell *set* correct while still misplacing individual corners would not have
been caught, and the overlay still drew every cell (including "confirmed" ones) from the in-memory
footprint's own iteration, just with a thicker line for set members.

New shared function **`footprint_adapter.reproject_exported_volume`**: reads a litematic, takes
every non-air block *straight out of the loaded snapshot*, shifts it by the exported volume's own
minimum corner (a single-region litematic always stores `Position (0,0,0)` and its shape local to
its own minimum - a tested format invariant, not a bug routed around here), converts *that* block
to a cell with `block_to_cell`, and only then calls `cell_corners_pixels` - every step after the
file read is driven by the file's own block values. Both `footprint_pilot.py` and
`footprint_pilot_real.py` now call this one function instead of their own duplicated inline
logic, and their overlays draw `SUPPORTED` cells' polygons from its output directly (thicker
outline, "(file)" label) rather than iterating the full candidate grid and checking set
membership for line thickness alone. Both scripts now persist the export-to-reconstruction
transform explicitly (`export_transform.json`: `minimum_shift`, `axis`/`side`/`face_coordinate`,
`lateral_origin`/`period`, and for the real pilot the additional `row_offset`/`column_offset`
between `footprint.py`'s local cell indexing and `transform.json`'s world lattice) rather than
only computing it transiently in memory.

**Control** (`tests/test_footprint_adapter.py`): an asymmetric (non-square, non row/column-
symmetric - exactly the shape G2's actual transposition bug would have misplaced) 3-cell footprint,
exported to a real litematic and reprojected via `reproject_exported_volume`, across all three
world axes and both face signs (six configurations, each with a camera pose found visible for that
specific axis/sign), checked against `cell_corners_pixels` called directly on the cell identity -
an independent formula path from the export/import round trip itself.

Both regenerated bundles (`M4-footprint-synthetic-pilot-2026-09-09/`,
`M4-partB-pilot-templates-2-2026-09-09/`) now include `export_transform.json` and an overlay
labelled with the file-reprojection provenance; their substantive results (10/1/1 synthetic,
53 supported real) are unchanged - this closes a verification gap, not a placement bug.

## What this does and does not establish

**Does:** S1's pairing/validity fix and S2's evaluation fix are each independently verified by
their own controls, decoupled from real-image noise. C2 confirms the corrected mechanism runs
end-to-end on three real photographs and produces geometrically valid (if not target-matching)
candidates, with an honest, fully-instrumented accounting of what was tried and why each rejected
combination failed. S3 and S4 close two bounded B correctness/verification gaps with their own
controls, independent of whether C succeeds.

**Does not:** establish that automatic 2D boundary recovery works on this fixture with this one
strength-ranked configuration - the opposite. Does not add or imply permission for a
texture/seam cue, a threshold sweep, or a target-seeded region of interest, all explicitly ruled
out by the review. Does not touch M5/M6 (Luna's assignment) or promote any production gate.

## Next

Returning this for review, as instructed, regardless of C2's outcome.
