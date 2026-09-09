# M4 Part A — contract repair and evaluator fixes

> Lead review: original three fixes and mapping helpers accepted; the geometric verdict and persisted
> transform contract remain incomplete. Texture-band controls do not validate physical holes/occlusion.
> See [the follow-up review and current assignment](M4-AB-M5-followup-review-2026-09-08.md).

**Claude Sonnet 5, 2026-09-08.** Responds to Astra's direct review at `3251541`.
All three claimed bugs verified independently before fixing (reproduced first,
not taken on trust), then fixed, then the fix verified against the same
reproduction. Full suite: 396 tests, green.

## The three claimed bugs, verified and fixed

**1. Uniform 64x64 plane produced 16 `contradicted` cells.** Reproduced exactly
before touching anything:

```
>>> support_cells(uniform_64x64, coverage=ones, period=16.0, phase=(0,0))
Counter({'contradicted': 16})
```

Fixed by splitting the old three-verdict contract into five:
`supported`, `contradicted`, `weak_evidence`, `foreground_occlusion`,
`outside_image`. A cell's own edge-energy density is now compared against the
plane's own median density (computed once, from every cell with enough
coverage to judge); below 15% of that median, or below an absolute floor for
the degenerate case where the *whole* plane is flat, reads `weak_evidence`
rather than `contradicted`. Re-verified:

```
>>> support_cells(uniform_64x64, ...)
Counter({'weak_evidence': 16})
```

**2. `distance=4` supplied but no plane returned when period detection
failed.** Reproduced: `plane_from_evidence` on a textureless image with
`distance=4.0` returned `None`, because `found is None` (no lattice period
detected) triggered an early return *regardless of whether the caller had
already supplied a distance*. Fixed: automatic mode still needs a detected
period (there is nothing else to derive a distance from); assisted mode does
not, and no longer discards the caller's evidence because an unrelated
detector failed. `measure_plane`'s `samples`/`span` are still valid even when
it found no period, so rectification proceeds at that resolution.

```
>>> plane_from_evidence(textureless_image, ..., distance=4.0)
SupportedPlane(distance=4.0, cells=64 cells, ...)   # was None
```

**3. AUC gives 0.0 for an all-tied input whose correct value is 0.5.**
Reproduced: with 55 positive and 114 negative scores all equal to 0.4, the
plain-`argsort` ranking formula gave exactly 0.0. Root cause: `argsort` breaks
ties by array position, and with the positive class concatenated first, every
positive got the lowest arbitrary rank. Fixed with proper mid-rank tie
handling (`_average_ranks` in `score_separation.py`), verified against the
degenerate case, a perfectly-separated case, a perfectly-reversed case, and a
hand-computed partial-tie case:

```
all-tied (want 0.5): 0.5
perfect separation (want 1.0): 1.0
reversed (want 0.0): 0.0
partial tie hand-check (want 0.375): 0.375
```

## A fourth issue found while fixing the second: foreground-occlusion detection
had no window wide enough to detect what it was looking for

Distinguishing an opening (farther surface, reads finer once rectified at the
plane's distance) from a foreground occluder (nearer surface, reads coarser)
needs measuring the cell's own dominant spatial period and comparing it to the
predicted block period. The first version computed that from the cell's own
edge profile alone - `period_pixels` samples wide - which structurally cannot
represent a lag anywhere near the occlusion threshold (`FOREGROUND_PERIOD_RATIO
* period_pixels`, i.e. autocorrelation was being asked to find a period longer
than the array it was searching). Fixed by reading the occlusion check from a
window wide enough to contain the periods it is meant to detect, independent
of whether local phase windowing is on.

## Invertible rectification-to-image/3D mapping

Was previously reimplemented ad hoc, and imprecisely (a coarser
`2*extent/samples` approximation instead of the exact linspace step), inside an
evaluation script. Now three tested library functions:
`lattice.rectified_pixel_to_plane_coordinates`,
`lattice.plane_coordinates_to_camera_point`, `camera.camera_point_to_world`.
Round-tripped against `rectify_plane`'s own internal grid and pinhole
construction directly (not against a second copy of the same formulas) in
`test_lattice.py` and `test_camera.py`: worst-case error 2.9e-11 (float
precision) over 200 random pixels at real calibration values.
`score_separation.py` now calls these instead of its own copy.

## The five synthetic controls

Built empirically, not assumed - each was run and its parameters adjusted
against what the code actually did before being locked in as a test assertion.
All five now pass in `test_supported.py`:

| control | construction | result |
|---|---|---|
| uniform-wall | flat 256x256 patch | 100% `weak_evidence` |
| opening | correct lattice with a region replaced by finer texture (farther surface) | 100% of interior cells `contradicted` |
| foreground-slab | correct lattice with a large region replaced by coarser texture (nearer surface) | 81 interior cells: 40 `foreground_occlusion`, 25 `weak_evidence`, 16 `supported` (coincidental scale alignment), **0 `contradicted`** - every cell that read as a disagreement read as the right kind. `weak_evidence` cells are a large uniform band's own interior, which genuinely has little local detail at a single cell's scale - a real property of the construction, not a defect the test papers over |
| invalid-pixel | zero-coverage region | 100% `outside_image` |
| nonzero-phase | a real lattice offset from what the caller passes as phase | >70% `supported` via local-window re-reading |

One test from earlier in the day, `test_a_photographed_blank_wall_is_...`,
asserted the *old, buggy* behaviour (`contradicted` for a blank photographed
region) as if it were correct. Fixed to assert `weak_evidence`, with the
history noted in the test's own docstring rather than silently rewritten.

## Real-data AUC, rechecked

Per Astra's instruction to recheck before drawing conclusions, both cues
re-measured on Template 2 axis 1 side -1 with every fix above applied:

| cue | AUC (original, buggy) | AUC (corrected) |
|---|---|---|
| local-window phase agreement | 0.523 | **0.528** |
| brightness-consensus | 0.502 | **0.502** (unaffected - no verdict/AUC-tie path touched it materially) |

The bug fixes were real and worth making, and did not overturn the
conclusion: both cues remain at chance on this view. Verdict breakdown for
the phase cue on the one view with real data (13.24.33): 156
`outside_image`, 167 `contradicted`, 1 `weak_evidence`, 0 `foreground_occlusion`
- this view's floor shows plenty of edge energy everywhere (screenshot
noise/detail), so `weak_evidence` rarely fires and no cell's local scale read
confidently coarser than the block period. That absence of
`foreground_occlusion` calls on this specific view is itself informative,
not a gap in the fix - the synthetic controls above confirm the mechanism
works when a coarser surface is actually present.

## Ownership and scope

Everything above is in `supported.py`, `score_separation.py`,
`register_planes.py`/`supported_plane.py` (call sites only, unaffected
behaviourally beyond the corrected verdict counts), and their tests - all
within the assigned ownership. Nothing touched in `cli.py`, `ui/`, Luna's
files, or any normal-mode entry point.

Moving to Part B now: the assisted mask-to-cell control, using an
image-selected visible plane per frame (the far wall in 13.24.24 instead of
its out-of-frame floor, per Astra's direction) rather than reference-derived
clicks.
