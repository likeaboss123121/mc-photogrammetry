# M3 bounded experiment — image-derived camera for one Template 3 frame

- **Agent:** Claude Opus 5 · **Date:** 2026-09-07T15:47-05:00
- **Frame:** `templates 3/2026-09-04_20.03.32.png`, 2560×1369
- **Outcome:** experiment gate met on orientation and focal length. One metric-definition
  problem found in the seam-displacement criterion, described below and needing the lead's ruling.
- **Code:** [`reconstruction/calibration.py`](../../src/mc_photogrammetry/reconstruction/calibration.py),
  [`tests/test_calibration.py`](../../tests/test_calibration.py)
- **Artifacts:** [estimate JSON](M3-estimate-2026-09-04_20.03.32.json) (written before the manifest was
  opened), [overlay](M3-overlay-2026-09-04_20.03.32.png)

## Result

| quantity | estimated | manifest | error | gate |
|---|---|---|---|---|
| pitch | 17.10° | 17.00° | **+0.10°** | ≤0.5° ✔ |
| yaw (mod 90°) | 14.00° | 14.00° (raw −76.00°) | **0.00°** | ≤0.5° ✔ |
| focal | 479.3 px | 479.3 px | **0.0%** | ≤2% ✔ |
| vertical FOV | 110.00° | 110.00° | 0.00° | — |

Position was **not** estimated. Vanishing-point calibration recovers orientation and focal length
only; translation needs the lattice period, which is the next part of M3.

### Honest precision, not apparent precision

The focal and yaw errors read as exactly zero because the true values sat exactly on the search
grid — FOV 110 was a coarse grid point and yaw 14 likewise. The refinement explored ±6% of focal in
1% steps and ±2° of yaw in 0.05° steps and moved neither, so the defensible resolutions are **~1% on
focal and ~0.05° on yaw**, not zero. Pitch is a genuine measurement: the refinement could have
returned 17.00 and preferred 17.10.

## Method

Rather than fitting three free vanishing points and hoping they come out orthogonal, the search is
over `(focal, yaw, pitch)` directly, with the three vanishing points derived from that triple. Every
hypothesis considered is therefore a physically realisable camera and orthogonality cannot be
violated by construction. Two Minecraft-specific facts narrow it further: the camera never rolls, so
orientation is two degrees of freedom; and the vertical axis is vertical, so with a centred principal
point its vanishing point sits on the image's centre column.

Residuals are expressed as **endpoint displacement in pixels** — how far a segment's ends sit off the
line from its midpoint to the vanishing point — so the fitting quantity and the grading quantity are
the same thing.

Settings: minimum segment length 40 px; inlier threshold 3.0 px; detection scales 1.0 and 0.5; gamma
0.45 then CLAHE clip 3.0 grid 16×16; FOV searched over 30–120°; yaw over [0, 90°).

**Contrast enhancement was necessary.** The frame means 33.7 with a 99th percentile of 88 — Template
3 was captured at gamma 100%. Raw detection found 459 segments over 40 px; after gamma and CLAHE,
1104. Near-vertical segments went from 69 to 147, and those are what pin the pitch.

## Validation

Correctness was established before the frame was touched. `tests/test_calibration.py` builds lattice
edges with the project's own `project_points`, then asks the estimator to recover the camera that
produced them — so a disagreement between the vanishing-point derivation and the shipped projection
fails the test rather than silently biasing the experiment. Three cameras (FOV 50/70/90, assorted yaw
and pitch) are recovered to within 0.5° and 2%.

### Stability on the real frame

Every combination below returned the *identical* answer — FOV 112.50°, yaw 13.50°, pitch 17.50° on
this coarser 2.5°/1.5° grid — as did four independent random halves of the segment set:

| min length | threshold 3.0 px | 1.5 px | 0.75 px |
|---|---|---|---|
| 40 px (2151 segs) | 94.3% inliers | 90.5% | 77.9% |
| 80 px (675 segs) | 94.4% | 88.3% | 74.2% |
| 120 px (306 segs) | 92.5% | 83.3% | **68.6%** |

68.6% of long segments inside 0.75 px is roughly 0.36° of orientation agreement.

A field-of-view sweep with yaw and pitch re-optimised at each step shows a sharp peak: score 153,648
at FOV 110 against 104,201 at 100 and 107,570 at 120. This was checked specifically because a
score counting inliers can be gamed by pushing vanishing points to infinity, where the constraint
decays to "roughly this orientation". It is not happening here — the vertical vanishing point sits
1568 px from the image centre, and the optimum is a peak rather than a trend.

Dropping every segment in the upper 42% of the frame, where the F3 overlay's text sits, changed the
estimate not at all (identical to two decimal places on all three parameters). The 40 px minimum
length already excludes overlay text, which is about ten pixels tall.

## The seam-displacement gate needs a length qualifier — lead's ruling requested

The gate asks for median seam displacement ≤2 px at a 1,920-pixel edge. Measured by scoring the
estimated and the manifest camera on the same detected segments:

| segments | estimated camera | manifest camera |
|---|---|---|
| ≥40 px (2151) | 0.152° → **2.55 px** | 0.157° → **2.63 px** |
| ≥120 px (306) | 0.038° → **0.63 px** | 0.043° → **0.73 px** |

Two things follow. First, **the estimated camera explains the seams slightly better than the true
camera does**, on both populations and with identical inlier counts (1993 and 282). The remaining
residual is line-detector noise, not calibration error, and no calibration can go below it.

Second, and the reason a ruling is needed: on segments ≥40 px the gate's 2 px is exceeded — but it is
exceeded *by ground truth as well*. As written the criterion grades the segment detector rather than
the calibration, and is unachievable by any camera whatsoever on short segments. On structural
segments ≥120 px both pass comfortably. Suggest the gate specify a minimum segment length, or be
stated relative to the manifest camera's residual on the same segments.

## Limitations and what is not claimed

- **One frame.** Nothing here shows the method generalises. Template 2 and 6 views and frozen
  Template 3 subsets are the next step, and the subsets must be frozen before tuning.
- **Yaw is modulo 90° from a single view.** The two horizontal lattice axes are interchangeable
  without semantic cues. The manifest's −76° and the estimate's 14° agree exactly under that
  reduction, but the quadrant came from the manifest, not from the image.
- **No position, no scale.** Orientation and focal length only.
- **No lattice period yet.** The period/phase estimation and the block-versus-texel hypothesis
  retention described in M3 are not implemented.
- **PyCOLMAP integration not started**; it is absent from the environment.
- The zero-roll and centred-principal-point assumptions were used and not refined, because a single
  frame at this quality gives no evidence to refine them against.

## Files touched

New only: `src/mc_photogrammetry/reconstruction/calibration.py`,
`tests/test_calibration.py`, and this report plus its two artifacts. No shared projection code was
modified, per the ownership split with Terra.
