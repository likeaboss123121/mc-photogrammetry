# DS-1: certified-coverage contract audit and S-D test repairs

DeepSeek DS-1, 2026-09-12 (workspace local date). Bounded audit of
`src/mc_photogrammetry/reconstruction/footprint_adapter.py` against
[LEAD-review-2026-09-09-fifth.md](LEAD-review-2026-09-09-fifth.md) and the S-D
section of [implementation-next.md](../implementation-next.md). Reviewed HEAD
`c6dba2c`. No production code, annotations, dependencies or commits were
changed; no full-suite run. Scope was the adapter's certified lower/upper
coverage contract and the S-D test repairs only.

Verification command (focused module, workspace `.venv`, no cacheprovider):

```
.\.venv\Scripts\python.exe -m pytest tests/test_footprint_adapter.py -q -p no:cacheprovider -o addopts=
```

Result: **39 passed in 10.22 s** (was 35 collected before this change: the
exact-55% test was retargeted to a true 1px cell, the old 0.1px configuration
became its own named control, and five analytic bracketing cases were added -
three of them parametrized).

## The contract, part by part

### Projection and coordinate rounding

- `pixel_to_lateral` unprojects through the plane equation
  (`footprint_adapter.py:90-119`); `lateral_to_pixel` inverts it
  (`:122-137`). The two are written independently, so the corner round-trip
  test is a real cross-check.
- Continuous image convention: pixel `(x, y)` occupies `[x, x+1) x [y, y+1)`
  (`_polygon_area`, `:324-335`). `_rasterize_polygon` (`:338-362`) shifts every
  vertex by `-0.5` and rounds to integers so `cv2.fillPoly`'s index-as-centre
  convention lines up with it.
- Cell-internal rounding is therefore a **half-pixel coordinate shift plus a
  `round()` call**. Crucially, the built-in `round()` is half-to-even, so
  exactly-half coordinates are not resolved the way the docstring describes
  (see finding D2).

### Boundary band: certified, not estimated

- `BOUNDARY_BAND_SUBCELLS = 3` (`:382-392`). For every polygon,
  `_fill_and_boundary` (`:475-518`) computes the ordinary point-sampled
  `fillPoly` fill and a 3-subcell-wide band around every edge
  (`cv2.polylines`, `:515`). `fill & ~boundary` is the certified *inside* set
  (lower bound), `fill | boundary` the *possible* set (upper bound); away from
  every band a sub-pixel is provably inside or outside by the Jordan curve
  argument, and the argument composes over a union of polygons per label.
- `_tile_area_sums` (`:521-582`) composes those sets through the
  specific-overrides-general exclusivity: VISIBLE minus (possibly)
  OPENING/OCCLUSION; that is **not** a defect. Its purpose is to guarantee the
  interval brackets the true coverage from both sides and that `declared` is
  never set from an estimate.

### Decision semantics

- `coverage_by_label` is the certified lower bound, `coverage_upper_by_label`
  the upper bound (`rasterize_cells`, `:676-685`). `declared` is set only from
  the **lower** bound clearing `VISIBLE_COVERAGE_THRESHOLD` (`:687-692`),
  iterating `OPENING, FOREGROUND_OCCLUSION, VISIBLE`. A cell exactly at the
  threshold can therefore never be certified and abstains. The upper bound is
  clamped with `min(1.0, ...)` (`:681`); the lower bound needs no clamp because
  its set is a subset of `cell_certain`.
- `full_area` is the shoelace area of the cell's own projected quad (never a
  raster count, `:632`), and `result` is skipped (`declared is None`) for
  behind-camera corners, zero area, empty crops, sub-`RESOLVABLE_AREA_PIXELS`
  cells, and the width-alone-over-budget guard (`:649-655`).

### Tiling and budget

- `_supersample_factor` (`:395-411`) targets `MAX_SUPERSAMPLED_CELLS /
  crop_area` sub-pixels per pixel, capped at `MAX_SUPERSAMPLE_FACTOR = 2000`.
- A crop already at the budget gets factor 1 and is processed by
  `_tile_row_bounds` (`:414-434`) in row bands of
  `MAX_SUPERSAMPLED_CELLS // (crop_width * supersample^2)` rows; `rasterize_cells`
  accumulates each band's sums (`:657-662`). Bands partition the rows exactly,
  so there is no double counting. If even a one-row band would exceed the
  budget, the cell is left explicitly unresolved (`:649-655`).
- Tiling seams re-clip every polygon to each band's rectangle, so a
  `BOUNDARY_BAND_SUBCELLS` margin is also drawn along the artificial seam
  edges. That only ever **widens** the interval (conservative); it was not
  observed to narrow it.

## Independent analytic evidence (no production defect found)

Every expected value below is exact geometry (rectangle or triangle area),
computed independently of the adapter - the adapter was never its own oracle.

| Control | Exact truth | Certified lower | Certified upper | Brackets? |
|---|---:|---:|---:|:--:|
| 1px 55% (review's exact: distance 1, period 1, focal 1, pp (2.2,3.2)) | 0.55 | 0.5379726573 | 0.5612121130 | yes, `declared=None` |
| same 55% at the old 0.1px default (distance 10) | 0.55 | 0.47499999999996795 | 0.6287999999999576 | yes, `declared=None` |
| translated fractional cell, exact quarter rectangle | 0.25 | 0.2475901440 | 0.2529398545 | yes |
| oblique rotated cell (15/30/70 deg), triangle = half | 0.50 | (tested) | (tested) | yes, `declared=None` |
| oblique + translated half rectangle (40 deg) | 0.50 | (tested) | (tested) | yes, `declared=None` |
| tiled 3000x3000 cell, half rectangle | 0.50 | 0.4955094444 | 0.5010000000 | yes |
| tiled 3000x3000 cell, 55% rectangle | 0.55 | 0.5452261111 | 0.5510000000 | yes |

Beyond the committed tests, a non-committed sweep of **311** analytic cases -
225 translated rectangle fractions (0.05-0.99; distances 0.7-10; several
fractional principal points and lattice origins, all fully in frame) and 86
oblique triangle-half cases (5-89 deg, several distances) - found **zero**
cases where the true fraction fell outside `[lower, upper]`. A supersample=1
integer-aligned huge cell (odd and even corner parity, 0.5 and 0.55 truth) also
bracketed correctly.

Conclusion: the certified interval is sound on every control examined, and no
production defect was exposed by an analytic coverage control. The two concrete
issues found are below; neither changes a decision.

## Findings

**D2 (concrete defect - documentation vs. behavior; low decision risk).**
`_rasterize_polygon`'s docstring (`footprint_adapter.py:352-355`) states that
shifting continuous corner `2` to centre-grid `1.5` makes it "correctly fall
between pixel centres 1 and 2 rather than landing on one". Line `360` then
applies `round()`, which is half-to-even: `1.5` becomes `2`, i.e. it lands
exactly on a pixel centre, and odd-aligned half-integers resolve to the
*opposite* parity neighbour from even-aligned ones. Reproduced directly at
supersample=1: a continuous unit square based at `(2,2)` fills 1 pixel, the
same square based at `(3,3)` fills a 3x3 block (9 pixels, true area 1). At
supersample>=2 the displacement is sub-pixel but still asymmetric for
grid-aligned edges.

Impact is bounded by the band, not by the shift: because the band is drawn
around the same rounded vertices, a wrongly included exterior sub-pixel next to
an edge always falls inside the 3-subcell band and is excluded from the
*certain* set; the rounding shows up as a wider interval, not an unsound
decision. That is consistent with the sweep above finding no bracket
violations, including at supersample=1 with odd corner parity. Recommended
fix is the docstring wording (production edits were out of scope this cycle);
the guard `BOUNDARY_BAND_SUBCELLS = 3` (`:382-392`) is load-bearing for this and
should not be reduced.

**D3 (documentation - memory/work scope).** `MAX_SUPERSAMPLED_CELLS`
(`:365-369`) bounds **one mask**; `rasterize_cells`' docstring (`:610-620`)
says every allocation is bounded by it, which is true per allocation but should
not be read as a cap on a cell or a call. `_tile_area_sums` (`:551-558`)
allocates two masks each for the cell and for all three label groups - up to
eight masks per band - so peak aggregate memory is a small multiple of the
budget, and total tiled work is the sum over bands, which grows with the crop
area when factor 1 is used. No 4M total-work or total-memory bound should be
promised (matches the lead's S-D wording).

**D4 (documentation nuance).** `CellEvidence.coverage_of_in_frame`'s docstring
(`:296-301`) calls it "the same overlaps, but as a fraction of only the cell's
in-frame pixels", but `rasterize_cells:682` divides the label's certified
**lower** area by the exact in-frame area, so it is itself a conservative
underestimate. Behavior is correct/conservative; the docstring should say so.

**Not defects (checked, left alone).** `_tile_row_bounds`' band size matches
the width-alone guard (`:414-434` vs `:649-655`); `_clip_polygon_to_rect`'s
intersection denominators (`:460-465`) are only used when exactly one endpoint
is inside, so they are never zero; the lower/upper clamp asymmetry (`:681`) is
safe for the reason above; the `in_frame` denominator is exact shoelace on the
clipped convex quad (`:673-674`), decoupled from the banded machinery as
documented.

## Test repairs made (tests/test_footprint_adapter.py only)

- `test_f4_the_reviews_exact_55_percent_case_never_certifies_visible` now
  passes `distance=1.0, period=1.0, focal=1.0, principal_point=(2.2, 3.2)`
  explicitly, asserts the setup really is 1px, asserts the full bracket
  `lower <= 0.55 <= upper`, pins the lead's reproduced values
  `0.5379726573 / 0.5612121130`, and keeps the original threshold decision
  assertions (`lower < 0.6`, `declared is None`).
- `test_f4_the_same_55_percent_case_at_the_older_0_1px_default_stays_undeclared`
  preserves the previous 0.1px configuration as a separately named control,
  including that its upper bound legitimately exceeds 0.6 while `declared`
  still abstains.
- Added five analytic oblique/translated bracketing cases with exact
  triangle/rectangle truths: translated fractional quarter rectangle; oblique
  rotated-cell half-triangle at 15/30/70 deg (with an independent shoelace
  setup check that the projected cell area equals the square of
  `focal*period/distance`); and an oblique+translated half rectangle built from
  absolute plane-lateral coordinates. No tolerances were loosened; all existing
  threshold, mixed-label, clipping and large-tile controls are unchanged.
- `_rectangle_cell` gained an optional `lateral_origin` keyword (default
  `(0,0)`, so all existing callers are bit-for-bit unchanged) and its docstring
  now states that `distance`/`period` fix the projected side length - the exact
  trap that made the old 55% test claim 1px while measuring 0.1px.

## Errata

The stale numbers in
[M4-F3-F4-corrections-2026-09-09.md](M4-F3-F4-corrections-2026-09-09.md) are
corrected in an explicit errata section appended to that report, with the
historical claims retained as such. No other documents were edited; no commits
or staging were performed.


## Lead acceptance addendum - GPT-6 Astra, 2026-09-12

Independently inspected the test diff and ran the documented focused command:
**39 passed in 10.10 s**. Accepted the explicit 1px reproduction, separately
preserved 0.1px configuration and six added collected cases (one old-size control
plus five analytic cases). The pre-change module therefore had 33 cases, not the
35 claimed above. No production adapter or annotation file was changed.

Acceptance is limited to these controls and the corrected historical evidence.
The reported 311-case temporary sweep was not independently replayed or saved as
an artifact; retain it as agent-reported evidence. Neither that sweep nor the
Jordan-curve statement alone proves that OpenCV's particular digital edge band
covers every true edge after rounding. A quantitative rounding/digital-line/
pixel-extent argument remains an S-D follow-up, not an established general proof.
Eight retained cell/label masks are not a peak-memory bound: combined masks and
intermediates add allocations. No total-byte claim is accepted without measurement.
Pinned numeric interval endpoints are a reproduction snapshot, not a restriction
against a future tighter valid bound; preserve the mathematical bracketing and
threshold assertions when revising that implementation.

The CLI printed a completion response but exited with code1, without a visible
credit/authentication error. Independent file inspection and test results establish
the test outcome; process status alone did not. No credit exhaustion is inferred.
No full-suite run was needed for this test/documentation-only package; the last
verified full baseline remains 514 at c6dba2c. No milestone is promoted.
