# M4 Part B, replaced: a footprint built from a declared label, not a cue

> Lead review: declared-cell placement accepted. The integer-origin test does not exercise image
> rectification or fractional phase; the image adapter and saved evidence bundle remain pending.
> Follow [the next assignment](M4-B-kernel-review-2026-09-09.md): synthetic image mapping, then one pilot if checks pass.

**Claude Sonnet 5, 2026-09-09.** Responds to Astra's lead review at `b22cc19`
(`docs/reports/M4-AB-M5-followup-review-2026-09-08.md`), specifically B1: the
original mask-to-cell control routed the visible-extent mask through
`plane_from_evidence`, which internally calls `support_cells` - the same seam
classifier Part A already spent a full session establishing does not work on
this data. That did not independently test the mask-to-cell/export mechanics;
it re-ran the already-known-broken cue and reported its already-known failure
as a finding about something else. Correct, and the sharpest point in the
review.

This delivers exactly what Astra scoped as "the next review boundary": a new,
classifier-free footprint module, and the one specified synthetic control,
before any further real-image work.

Code: [`footprint.py`](../../src/mc_photogrammetry/reconstruction/footprint.py),
6 tests in [`test_footprint.py`](../../tests/test_footprint.py). Full suite:
410 tests, green.

## What's different from the old `mask_to_cell_control.py`

`build_footprint` and `cell_to_block` take no image, no patch, no period, no
phase - the parameter lists carry no such argument at all, which is the
structural guarantee (not a tested property, a *fact about the function
signatures*) that no texture, of any kind, can change a result. A footprint's
cells resolve from a plain `dict[(row, column), label]` a caller supplies -
`VISIBLE`, `OPENING`, or `FOREGROUND_OCCLUSION` - to one of three geometry
states: `supported`, `opening` (a confirmed gap), or `unknown`. A declared
occluder resolves to `unknown`, not to a positive claim about what's behind
it - declaring "something blocks my view here" is a different assertion from
"there is nothing here," and the two were kept apart deliberately.
`footprint_to_volume` turns that into a real `BlockVolume`: `supported` cells
get a visually distinct placeholder (`minecraft:light_gray_wool`, not a
material guess), `opening` cells get an explicit `minecraft:air`
observation, and `unknown` cells are left out of the volume entirely.

## The synthetic control, exactly as specified

4x3 grid, one declared opening, one declared foreground-occluded cell,
everything else visible. Expected: 10 supported, 1 opening, 1 unknown.
Measured:

```
declared: 10 VISIBLE, 1 OPENING (1,2), 1 FOREGROUND_OCCLUSION (2,0)
resolved: 10 supported, 1 opening (cell (1,2)), 1 unknown (cell (2,0))
```

Exact match. Also checked, per the review's own list:

- **Inward face-to-cell offset**, against the worked example: "plane z=1,
  outward normal +Z, with solids occupying cell z=0" gives block `(x, y, 0)` -
  `1 - 1 = 0`, checked directly, not just asserted.
- **A nonzero rectification-window-style offset and phase**: the same 4x3
  declaration repeated with `face_coordinate=41` and `lateral_origin=(100,
  -7)` instead of `(1, (0, 0))`. The *shape* (which local cells resolve to
  which geometry) is identical; only the absolute block coordinates shift by
  exactly the declared offset - checked cell by cell, not just at one corner.
- **Exact litematic round trip**: the control's `BlockVolume` written with
  the project's real `LitematicExporter` and read back with the real
  `load_litematic` (not a mock of either). Every `supported` cell's block
  coordinate reads back as the placeholder material.
- **Texture invariance**: trivially and structurally guaranteed rather than
  spot-checked, since no image ever enters this module - stated plainly
  rather than padded out with redundant uniform/noisy/banded test runs that
  could only ever pass.

## An honest limitation, stated rather than glossed over

The litematic format has no third state - only air or a real block. A
declared `opening` (explicit air) and a declared-but-unresolved cell (omitted
entirely) read back identically as "not solid" if the unknown cell happens to
fall inside the bounding box of cells that *were* set. The round-trip test
checks for exactly this and documents it: the exported geometry alone cannot
carry the opening/unknown distinction on its own. The evidence sidecar - a
per-cell `declared`/`geometry` record a caller saves alongside the volume,
which this module produces (`DeclaredFootprint.cells`) but does not itself
write to disk - is where that distinction actually has to live.

## Other review points, accepted

- **A1 (noise reads as confident geometric verdicts): confirmed
  independently**, not taken on trust. `support_cells` on `np.random.default_rng(123)`
  noise, 128x128, period 16: 44 `contradicted` + 20 `foreground_occlusion`, 0
  `weak_evidence` (my own run; close to but not identical to the review's
  54/10, likely a `local_window_periods` default difference - the qualitative
  finding matches). This is a real gap in Part A's own instrument validation:
  I checked that noise doesn't clear the `supported` threshold, but never
  checked that the `contradicted`/`foreground_occlusion` split itself is
  well-behaved on pure noise. Not fixed tonight - flagged for whoever picks
  this back up, since Part A's classifier is now explicitly out of scope for
  Part B's replacement.
- **The pitch-direction correction: accepted, and it was my error.** I
  described Template 2's third camera as "pitching up" / "looking upward"
  throughout the session. Checked directly: `camera_rotation_matrix`'s
  forward vector has world-Y component `-sin(pitch)`, so positive pitch
  points the camera *down* in this project's convention - which is also
  standard Minecraft F3 convention (pitch -90 = straight up, +90 = straight
  down). The camera was looking down, not up. This was a narrative error in
  my own prose, not a computational one - the registration work earlier in
  the session matched real camera positions to 0.004 blocks using the same
  `axis_directions` function, which was computing the correct geometry the
  whole time; only my English description of what it meant was backwards.
  Worth stating plainly rather than quietly fixed in old reports.
- **Wrong-scale-can-cause-false-negatives: accepted, I overclaimed.** The old
  Part B report said a wrong stated length "doesn't change the conclusion...
  a wrong distance would make cell alignment worse, not accidentally produce
  false negatives." That's backwards: a wrong distance changes the predicted
  block period, which *would* misalign real seams and could by itself produce
  exactly the all-`contradicted` result observed, independent of whether the
  underlying cue works at all. The old report's own annotation overlay showed
  real evidence of exactly this kind of error. I should not have asserted the
  conclusion was safe from it.
- **Debug overlay likely included in a mask, wrong wall used without
  justification: accepted, not yet fixed.** Both are real; both apply to the
  now-superseded real-image annotation from the old Part B, which this
  message's scope doesn't re-run - Astra's own sequencing holds further
  real-image work until after this synthetic control is reviewed.

## Not done tonight, and why

A1's instrument-validity gap in `support_cells` itself, and the corrected
real-image annotation pass (steps 4-5 of Astra's plan) - both explicitly held
until this synthetic control is reviewed, per "Publish the transform
contract... before another real-image run. This is the next review
boundary."
