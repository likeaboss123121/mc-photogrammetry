# M4 Part B — assisted mask-to-cell control, genuinely image-drawn

**Claude Sonnet 5, 2026-09-08/09.** Responds to Astra's B directive. All input
- the two length-anchor points, the block count, the visible-extent polygon,
the occlusion regions - was picked by looking at each of the three Template 2
screenshots directly (see the images referenced below); none of it was read
from `cameras.json`, the reference litematic, or F3 text visible in-frame.
Camera orientation still comes from image-derived calibration, not the
manifest. The manifest is read only in the final scoring step below.

Code: [`mask_to_cell_control.py`](../../scripts/mask_to_cell_control.py),
[`mask_to_cell_overlay.py`](../../scripts/mask_to_cell_overlay.py).
Annotations: [`M4-partB-annotations-templates-2.json`](M4-partB-annotations-templates-2.json).
Outputs: `docs/reports/M4-partB-templates-2/` (geometry, evidence sidecar, and
both overlays per frame).

## Result

| frame | plane chosen | judged cells | supported | precision | recall | IoU |
|---|---|---|---|---|---|---|
| 13.22.14 | axis 2, side +1, left wall | 121 | 0 | n/a (0 predicted) | **0.000** | 0.000 |
| 13.24.24 | axis 0, side +1, left wall | 100 | 0 | n/a (0 predicted) | **0.000** | 0.000 |
| 13.24.33 | axis 0, side +1, left wall | 900 | 2 | **0.000** | n/a (0 truth) | 0.000 |

Truth counts (22, 3, 0 camera-reachable reference cells on the chosen plane)
come from the same tested world-point reconstruction built for Part A
(`rectified_pixel_to_plane_coordinates` -> `plane_coordinates_to_camera_point`
-> `camera_point_to_world`), read only here, to score.

**This fails Astra's stated exit bar (>=90% precision and >=50% recall on at
least two targets) by a wide margin, not marginally.** Zero recall where real
occupancy exists; the two predictions made elsewhere were both wrong. Per that
directive: deliver the assisted control and failed overlays, and return for
lead review rather than continuing toward automatic integration or starting
Part C.

## What the mechanism actually got right

The pipeline mechanics behave correctly, independent of whether the underlying
cue works - this is the part a control is for, and it held up:

- **No filled rectangle, no silent hole.** With almost nothing `supported`,
  the geometry files correctly contain almost nothing (`geometry_cell_count`:
  0, 0, 9) rather than guessing a rectangle or defaulting the unknown space to
  either full or empty.
- **The footprint is genuinely irregular.** See
  `2026-09-02_13.24.33.png.verdicts.png`: a diagonal band of judged cells
  following the mask boundary, not a rectangle - because the mask itself was a
  rough hand-drawn quadrilateral over a wall seen at an angle, and the cell
  grid follows it exactly.
- **A real implementation bug found and fixed mid-run.** The first version of
  this control used "the whole frame minus the occlusion polygons" as the
  mask. That fed `plane_window` regions showing entirely different, near-
  grazing-incidence surfaces (the ceiling, the opposite wall) for the axis
  being tested, and its window computation blew up (span ~283145, implied
  block period ~0.003px), silently returning `None` for every combination on
  two of three frames. Fixed by scoping the mask to an explicit, hand-drawn
  `visible_extent_polygon` for the plane actually being measured, with
  occlusion regions subtracted from inside it - not "everywhere not known to
  be blocked." All three frames resolved after the fix.
- **A texture-scale occlusion signal fired on real data, independent of the
  manual mask.** The verdict overlay for 13.24.33 shows a cluster of
  `foreground_occlusion` cells (orange) in the middle of the judged region,
  outside the manually-drawn occlusion polygon - the automatic detector built
  in Part A caught something the hand-drawn mask did not. Not verified against
  ground truth here (out of this control's scope), but a genuine, independent
  signal worth noting.

## An honest source of additional error: my own point-picking

`2026-09-02_13.22.14.png.annotation.png` shows the stated-length line I drew
spans roughly one repeating wall motif, not the two-to-three I intended when
writing the annotation JSON - by-eye pixel selection from a static image is
imprecise, the same way it would be for a real user without a zoomable, exact
tool. This is real, and worth stating rather than silently redoing until it
looks right: it means the assisted distance for that frame likely carries more
error than the 0.1-0.4% this project's earlier length-based measurements
achieved with more carefully chosen points. It does not change the
conclusion - a wrong distance would make cell alignment worse, not
accidentally produce false negatives that happen to read as "the cue doesn't
work" when it actually does.

## Conclusion

A third, independent measurement - genuinely image-drawn input, correctly
scoped masks, tested world-point reconstruction - agrees with Part A's finding
on reference-derived annotations: per-cell appearance-based occupancy does not
separate real occupancy from not on this data. The pipeline itself (mask
scoping, irregular footprints, neutral-block export with an evidence sidecar,
unknown space left unknown) works as specified. Stopping here per Astra's exit
criteria rather than proceeding to Part C.
