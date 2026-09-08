# Image-only planar region proposal — three cues measured, one path forward

- **Agent:** Claude Opus 5 · **Date:** 2026-09-07T19:05-05:00
- **Why:** M3's lattice reads a plane's distance to 0.2%, but only when told which pixels are one
  plane. That mask came from the reference litematic, which makes it a demonstration rather than a
  pipeline stage. This is the attempt to find the regions from the picture.
- **Outcome:** **not solved.** Three cues characterised, two of them refuted on measurement and one
  partially successful. The combination they point to is described but not built.
- **Code:** [`reconstruction/surfaces.py`](../../src/mc_photogrammetry/reconstruction/surfaces.py),
  [`tests/test_surfaces.py`](../../tests/test_surfaces.py). 295 tests pass.
- **Fixture:** Template 2, `2026-09-02_13.22.14.png`, scored against a per-pixel oracle rendered from
  the reference. The oracle scores; it never reaches the proposer.

## Cue 1 — the normal is the direction that is absent

A face perpendicular to a world axis is crossed by edges running along the *other two* axes and
contains none running along its own: an edge along the normal points away from the surface, so it can
only be a silhouette. Since `calibration` already sorts every segment into one of three vanishing
directions, the normal of a patch should be the direction missing from it.

This is the right *shape* of cue, and it beats asking which directions are present — an edge running
east lies in the north-facing plane and equally in the up-facing one, so presence cannot separate
those two and absence can.

**Measured, it decides too little.** Among tiles carrying enough segment length, the median margin
between the least- and second-least-represented direction is **0.00**: the typical tile has *two*
directions absent, not one, because a 48-pixel patch of wall usually catches seams from a single
family. Coverage of the frame:

| tile | tiles with evidence | decidable at margin ≥0.15 | ≥0.35 |
|---|---|---|---|
| 48 px | 1276 / 1566 (81%) | 209 (13.3%) | 84 (5.4%) |
| 96 px | 383 / 405 (95%) | 79 (19.5%) | 26 (6.4%) |
| 160 px | 143 / 144 (99%) | 32 (22.2%) | 12 (8.3%) |

Larger tiles barely help — the median margin only reaches 0.03. On the target frame this yields four
regions covering 27 of 1566 tiles, 1.7% of the image. Their **normals are right 3 times in 4**, so
the cue is genuine where it fires; it simply almost never fires.

Not tuned around. Lowering the margin admits tiles whose three counts are noise, and raising it just
selects the few percent where the cue happens to work.

## Cue 2 — rectify under each candidate normal and keep the most legible

The pattern that has worked elsewhere in this project: hypothesise and verify. Only the correct
normal turns a perspective-skewed grid into an axis-aligned one, so its rectified patch should show
the strongest periodicity.

**It does not discriminate at all.** Over 140 tiles whose true orientation is unambiguous:

```
normal correct: 47/140 (34%)      chance is 33%
```

A skewed grid remains periodic along whatever directions it is sampled on, so period strength
survives a wrong rectification. This cue is dead as formulated; salvaging it would need a measure of
*rectangularity* rather than of periodicity.

## Cue 3 — vanilla shades faces by orientation

Minecraft multiplies face brightness by orientation: 1.0 up, 0.5 down, 0.8 north/south, 0.6 east/west.
That is a Minecraft-specific signal neither cue above uses, and it does not depend on detecting a
single segment. Per-pixel brightness against the oracle:

| true axis | shading | mean | median |
|---|---|---|---|
| X (east/west) | 0.6 | 26.3 | 24 |
| Y (up/down) | 1.0 or 0.5 | **110.3** | **121** |
| Z (north/south) | 0.8 | 31.5 | 28 |

Best single-threshold accuracy, against the majority-class baseline that a useless classifier achieves:

| split | accuracy | baseline |
|---|---|---|
| **Y vs X+Z** (horizontal vs vertical) | **86.5%** | 64.5% |
| X vs Y | 87.8% | 57.2% |
| Y vs Z | 80.9% | 67.8% |
| **X vs Z** | **73.8%** | **73.8%** |

So brightness solves the horizontal/vertical question well and carries **exactly no information**
about east/west versus north/south — the X vs Z accuracy equals its baseline to a tenth of a percent.
That is what the shading constants predict: 0.6 against 0.8 is a smaller difference than the variation
between one block's texture and another's.

## What this points to, and what it does not

The two surviving cues are complementary rather than competing, and they fail in opposite places:

* Brightness answers **horizontal or vertical** across the whole frame, at 86.5%, and cannot go
  further.
* Absence answers **which vertical axis**, correctly where it fires, but fires on 5–13% of tiles.

Cascading them — brightness first to split the frame, absence second to choose between X and Z, and
region growing to carry a decided tile's answer into its undecided neighbours — is the obvious next
attempt. It is *not* implemented and nothing here shows it will work; the two error rates may well
compound.

Also untried, and worth ranking against it: multi-view consistency. A region's normal and distance
have to agree across views of the same surface, which is evidence no single frame offers, and the
calibration already places the views in a common orientation.

## Incidental fixes to `lattice`

Two defects surfaced only once real regions were fed in, both fixed and covered by tests:

* **The rectification window sat at the plane's centre.** A proposed region is rarely there, so the
  window missed it — every region on Template 2 returned no period. `plane_window` now sizes and
  centres the window on the region, and `measure_plane` returns the extent it settled on, since it is
  no longer the one requested.
* **Planes were only ever placed along the positive normal.** A normal names an axis, not a direction
  along it, so a floor below the eye had every ray pointing away from its plane and produced no window
  at all. The earlier oracle result survived this only because the surface it happened to test was a
  ceiling. Both sides are now handled.
