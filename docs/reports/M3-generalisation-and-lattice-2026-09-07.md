# M3, continued — generalisation across all views, and a first lattice attempt

- **Agent:** Claude Opus 5 · **Date:** 2026-09-07T17:05-05:00
- **Follows:** [the bounded single-frame experiment](M3-camera-calibration-2026-09-07.md)
- **Outcome:** calibration generalises — 102/109 views inside the gate, 103/109 allowing
  lattice-axis symmetry. The lattice estimator **does work on real frames**, to 0.2% on Template 2's
  floors, once given an isolated plane and adequate sampling. An earlier section of this report said
  it did not; that conclusion was wrong and is corrected below rather than deleted.
- **Code:** [`reconstruction/calibration.py`](../../src/mc_photogrammetry/reconstruction/calibration.py),
  [`reconstruction/lattice.py`](../../src/mc_photogrammetry/reconstruction/lattice.py),
  [`tests/test_calibration.py`](../../tests/test_calibration.py),
  [`tests/test_lattice.py`](../../tests/test_lattice.py)
- **Frozen subsets:** [M3-frozen-subsets.json](M3-frozen-subsets.json), fixed by filename order with
  SHA digests before any result existed. 282 tests pass.

## Calibration generalises

Run over **every view of all six fixtures** — 109 views, no subset selection, the method exactly as
published in the previous report.

| fixture | views | round 1 | round 2 | median &#124;pitch&#124; | median &#124;yaw&#124; |
|---|---|---|---|---|---|
| templates 2 | 3 | 2 | **3** | 0.10° | 0.05° |
| templates 6 | 4 | 3 | **4** | 0.03° | 0.05° |
| templates 3 | 47 | 45 | **46** | 0.05° | 0.05° |
| templates 4 | 7 | 7 | 7 | 0.05° | 0.00° |
| templates 5 | 39 | 32 | **36** | 0.05° | 0.05° |
| templates (1) | 9 | 7 | **6** | 0.05° | 0.05° |
| **total** | **109** | **96 (88.1%)** | **102 (93.6%)** | | |

Template 1 went *down* by one view under round two. Five fixtures improved and one regressed; that is
recorded rather than averaged away.

### Round two, declared: focal length is a session quantity

Round one fitted each view alone. Round two was designed **after** seeing its failures, and says so.
The observation behind it: Minecraft's field of view is a setting, so every screenshot from one
session shares it. Fitting each view freely, taking the *median* focal across the session, and
refitting with focal held there uses no manifest — the median comes from the estimates themselves.

It works because the median is untouched by the outliers. The true focal is 479.3 px on every frame
of every fixture, and the median estimate came to exactly 479.3 on all six, including the two
fixtures whose individual estimates ranged over hundreds of pixels.

### Two claims from round one that did not survive

* **Inlier fraction does not predict failure.** On the first two failures it looked like a confidence
  signal. Across 109 views the passing median is 93.7% and the failing median 92.7%; segment count is
  likewise flat. There is at present **no signal separating a good estimate from a bad one**, which
  matters more than the pass rate does, because production has no manifest to check against.
* **Lattice-axis symmetry explains almost nothing.** Evaluating orientation error as the smallest
  rotation over all 24 rotations of the cube changed round one's result not at all, and rescues
  exactly one of round two's seven failures. The two views whose pitch is wrong by 90° are not
  relabellings; they are wrong.

### The seven that still fail

All now have correct focal, so these are pure orientation errors.

| frame | raw pitch | raw yaw | symmetric error |
|---|---|---|---|
| 2026-09-02_12.17.31 | −0.00° | +30.05° | 30.05° |
| 2026-09-05_10.36.20 | +90.10° | +2.95° | 2.90° |
| 2026-09-05_10.36.22 | +90.10° | +2.60° | 2.60° |
| 2026-09-04_20.08.18 | −0.70° | +1.20° | 1.39° |
| 2026-09-02_12.17.15 | −90.45° | −0.80° | 0.92° |
| 2026-09-05_10.35.11 | −0.20° | +0.65° | 0.68° |
| 2026-09-02_12.17.38 | −90.10° | −0.40° | 0.41° (passes) |

One gross failure at 30°; the rest are between 0.4° and 2.9°, just outside a 0.5° gate.

## The lattice estimator, and a wrong conclusion corrected

**First conclusion, now withdrawn.** Rectifying the whole image and reading the period gave distances
wrong by factors of eight to fifty on Template 2. Halving the rectification extent left the measured
period almost unchanged when a world-scale feature must scale with it, and I concluded the period was
an artifact of the resampling and that the measurement did not work. That was wrong, and it was wrong
because the diagnosis was untested — I said so at the time and then tested it.

**What the test showed.** Using the reference litematic as an oracle to isolate one genuine, flat,
visible wall — exactly the segmentation M4 would supply — and sweeping the patch resolution:

| samples | measured period | implied distance | true |
|---|---|---|---|
| 512 | 3.88 px | 4.13 | 8.38 |
| 1024 | 3.85 px | 8.31 | 8.38 |
| 2048 | 7.63 px | **8.39** | 8.38 |
| 4096 | 15.26 px | **8.39** | 8.38 |

The fault was **undersampling**, not resampling artefacts. A texel on a wall eight blocks away spans
about 1.9 rectified pixels at 512 samples, under `MINIMUM_PERIOD`, so the estimator skipped it and
locked onto the first harmonic — reporting twice the period and half the distance. The isolating mask
mattered too, so segmentation is still required; it simply was not the whole story.

Three changes followed, each measured:

* **`sampling_for`** sets the patch resolution from the source resolution rather than a fixed 512,
  so a patch cannot be silently too coarse.
* **`plane_period`** reconciles the two in-plane readings. A plane has one lattice, so readings
  related by an integer factor mean one direction locked onto a harmonic, and the shorter is right.
  On Template 2 one direction read true and the other exactly double, with the roles *swapping*
  between views of the same floor — view-dependent, so not a material pattern.
* **`measure_plane`** adapts the sampling to the distance it measures, since how finely a plane must
  be sampled depends on how far away it is and that is the unknown. Two or three rounds settle it.

Template 2's floors then read **8.40 and 8.42 blocks against a true 8.38** — 0.2%.

### What still fails, and why it was not patched

Z walls still read exactly half: 7.08 against a true 14.10. The cause is that a measured period is
not necessarily one texel. Where a texture's strongest detail is two texels wide the block seam sits
eight periods away, not sixteen, and the distance halves.

Reading the multiple off the correlation was implemented and **reverted**, because it does not work:
a periodic signal peaks at every multiple of itself, so a block seam cannot be distinguished from a
texture harmonic by the presence of a peak. Six synthetic tests failed and would have had to be
weakened to keep it. `PeriodEstimate.block_supports` records the candidate multiples for whatever
resolves them — a second view, or a user-supplied length, both of which the milestone already allows.

Shipping a half-working generalisation would have been worse than shipping the limitation.

### The part that does work from one image

If a measured period is texels, block seams sit sixteen of them apart and leave their own correlation
peak; if it is already block seams, nothing does. `coarse_support` measures that and orders the
hypotheses by it. Both readings are always retained. The prior it replaced was wrong, not merely
unproven: block-first ranking put Template 2's walls 30–50 blocks away when they are under three.

## Contract gap for Terra

`CameraPose` requires a position. Vanishing-point calibration recovers orientation and focal length
*without* one — translation needs the lattice period. `to_camera_solution` therefore returns
`pose=None` and names the frame in `unresolved`, rather than inventing an origin a consumer could not
distinguish from a measured one. The cost is that yaw and pitch, which are measured, do not survive
the round trip. **Suggest making `CameraPose.position` optional.**

`calibration.axis_directions` now defers to Terra's `camera_space_points` rather than restating the
world-to-camera convention. The two derivations agreed, which is not a reason to keep both.

## Not claimed

- No position, no scale, no absolute pose from any of this.
- Yaw remains modulo 90°; the quadrant is not recoverable from one view.
- PyCOLMAP is untouched and absent from the environment.
- The session-median focal assumes all supplied frames share one field of view. True of these
  fixtures; a user mixing sessions would need the estimates clustered rather than medianed.
- Zero roll and a centred principal point are still assumed, not refined.
