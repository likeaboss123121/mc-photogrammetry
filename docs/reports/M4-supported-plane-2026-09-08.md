# M4 — supported cells for a local plane, Template 2

**Claude Opus 5, 2026-09-08.** Artifact: [`M4-supported-plane-templates-2.json`](M4-supported-plane-templates-2.json).
Code: [`supported.py`](../../src/mc_photogrammetry/reconstruction/supported.py),
[`supported_plane.py`](../../scripts/supported_plane.py), 12 tests in
[`test_supported.py`](../../tests/test_supported.py).

Reproduce:

```
python scripts/supported_plane.py "templates 2" \
    --calibration docs/reports/M3-calibration-templates-2.json \
    --axis 1 --simulate-annotations \
    --out docs/reports/M4-supported-plane-templates-2.json
```

## The result in one line

**The distance works. The per-cell support cue does not.** A stated length places
the plane to within 0.4% on every frame; phase agreement then judges the cells on
that correctly-placed plane no better than random noise would.

## All three views, both paths, identical settings

Every frame is run separately in its own local frame. No frame is cropped, chosen
or dropped for looking good. `automatic ±Y` are the two surfaces perpendicular to
the axis — the ceiling above the eye and the floor below it — and are run on both
sides so the automatic path is not quietly handed the annotation's choice of
surface. `obs` counts cells inside the photograph; the rest are unobserved.

| frame | path | distance | reference | error | obs | supported | p50 | p90 | max |
|---|---|---|---|---|---|---|---|---|---|
| 13.22.14 | automatic +Y | 8.56 | 8.38 | **2.1%** | 47 | 0 | 0.371 | 0.400 | 0.492 |
| 13.22.14 | automatic −Y | 4.80 | 1.62 | **196.1%** | 6 | 0 | 0.357 | 0.431 | 0.463 |
| 13.22.14 | assisted | 1.63 | 1.62 | **0.4%** | 1 | 0 | 0.363 | — | — |
| 13.24.24 | automatic +Y | 8.37 | 8.38 | **0.1%** | 59 | 0 | 0.365 | 0.409 | 0.453 |
| 13.24.24 | automatic −Y | 5.98 | 1.62 | **269.1%** | 0 | 0 | — | — | — |
| 13.24.24 | assisted | 1.63 | 1.62 | **0.3%** | 0 | 0 | — | — | — |
| 13.24.33 | automatic +Y | — | — | no plane measured | — | — | — | — | — |
| 13.24.33 | automatic −Y | 9.57 | 9.42 | **1.6%** | 192 | 3 | 0.369 | 0.481 | 0.587 |
| 13.24.33 | assisted | 9.43 | 9.42 | **0.1%** | 168 | 2 | 0.375 | 0.459 | 0.574 |

Chance is **0.378**, measured on random noise with a fitted phase over four seeds
(mean 0.377, worst cell 0.409). `SUPPORT_THRESHOLD` is 0.55.

## What the numbers say

**Distance, assisted: 0.4%, 0.3%, 0.1%.** Two points named on a surface and a
block count settle the scale on every frame. This needs no manifest, no second
view and no access to the world, so it is a real production input rather than
scaffolding.

**Distance, automatic: fine when the plane is far, useless when it is near.**
2.1%, 0.1% and 1.6% on planes 8.4–9.4 blocks off; 196% and 269% on a floor 1.62
blocks under the camera. The near-plane failure is a coverage problem, not a
harmonic: the window spans a fixed *angle*, so at 1.62 blocks it covers about
three blocks of floor, of which the frame holds one cell (frame 1) or none
(frame 2). There is nothing in the window to measure.

**Support: no information at all.** Median scores of 0.357–0.375 against a chance
level of 0.378, on every frame and in both paths. The decisive case is frame
13.24.33 assisted, where the distance is right to 0.1% and 168 cells sit inside
the photograph: 2 clear the threshold. With the geometry essentially exact, the
cue still cannot tell a block face from noise.

This is not the threshold being strict. Lowering it to chance would classify
noise as wall, which is the failure I corrected earlier in this same module.

## Why the cue fails, measured rather than assumed

The obvious explanation is that the tested scale was wrong. It is not. On frame
13.24.33 I scored the same cells against seams predicted at the block period and
at every binary fraction of it down to the texel:

| period tested | pixels | observed cells | ≥0.55 | p50 | p90 | max |
|---|---|---|---|---|---|---|
| block | 167.13 | 168 | 2 | 0.381 | 0.431 | 0.579 |
| block / 2 | 83.56 | 168 | 3 | 0.387 | 0.449 | 0.595 |
| block / 4 | 41.78 | 168 | 3 | 0.381 | 0.481 | 0.561 |
| block / 8 | 20.89 | 168 | 2 | 0.389 | 0.485 | 0.580 |
| block / 16 (texel) | 10.45 | 168 | 11 | 0.366 | 0.529 | 0.619 |
| measured lattice | 10.29 | 168 | 0 | 0.368 | 0.420 | 0.507 |

Every row sits at chance. The scale chain is sound — the measured lattice period
of 10.29 px against a predicted texel period of 10.45 px is the same surface read
two ways — but no periodic set of lines within a cell captures where a real
screenshot's edge energy sits. Each block carries a different 16×16 texture with
its own internal structure under its own face shading, so a *single global phase*
does not describe them, and per-cell phase fitting is not an option: an offset
fitted per cell agrees with anything, which is the invention this module exists
to prevent.

**Phase agreement is the third image-only cue I have measured and refuted**,
after planar region proposal (decided 5–13% of tiles) and period strength (did
not identify a normal at all). I predicted before running that automatic would be
weak and assisted would work. Assisted works for the *distance* and fails for the
*support*, so the prediction was half right, and the half it got wrong is the
half that matters for geometry.

## Defects found and fixed while building this

Three, each of which would have made the numbers above look better than the truth.

**Coverage was fabricated.** With no mask supplied the code assumed the whole
rectified patch had been photographed. Cells the picture never saw therefore
carried zero edge energy and scored zero — and were reported **contradicted**,
turning a hole in the photograph into a hole in the wall, which is the one
confusion the module exists to avoid. Coverage now comes from warping a solid
frame through the same transform, which answers it exactly. On frame 13.22.14
this moved 242 of 289 cells from contradicted to unobserved.

**The phase was read at the wrong period.** The phase came from the measured
lattice (10.29 px, the texel period) while the seams being predicted were at the
block period (167 px) — a different modulus entirely, so the test was of one
sixteenth of the grid. The phase is now read at the period actually being tested.

**The phase estimator was biased.** Scanning offsets for the one with the most
energy inside the tolerance band returns the *edge* of any plateau rather than
its centre, putting predicted seams half a tolerance off the real ones; on a
synthetic lattice at phase 11 it answered 5.5. Replaced with the circular mean of
the edge energy at that period, which is unbiased and answers 11.

None of the three changed the conclusion. All three are the sort of defect that
turns "the cue does not work" into "the cue works a little", so they are recorded.

## Honesty notes on the method

- **Bounds come from supported cells**, never from a reference extent or from the
  rectified window. With almost nothing supported, most rows state no extent —
  which is the correct output for evidence this weak, not a bug.
- **Three verdicts stay distinct** throughout: supported, contradicted,
  unobserved. Unobserved is not evidence of absence.
- **`--simulate-annotations` is scaffolding and is labelled as such** in the
  artifact. It stands in for a user's two mouse clicks by projecting reference
  corners through the manifest pose. The pixel coordinates are simulated; every
  step downstream of them is not, and reads only the image and the calibration.
- **The scores are counts, not per-cell precision and recall.** A single view
  recovers no position, so a local cell frame cannot be put into world
  coordinates and per-cell correspondence is not claimed. The artifact says so on
  every scored row. `ref` is how many reference cells lie on that plane and are
  camera-reachable; comparing it to `supported` compares two counts.
- **No frame was cropped or dropped.** `automatic +Y` on 13.24.33 measured no
  plane at all and is reported as such.

## Addendum, same day: the "refuted" conclusion above was too strong

While building cross-view registration (see
[M4-registration-2026-09-08.md](M4-registration-2026-09-08.md)) I looked directly
at a rectified patch and found the measurement window spans roughly 18 blocks -
the whole extent the plane's distance made available, not a region sized to the
calibration's own precision. Autocorrelation of the edge profile at the block
period was 0.015 (indistinguishable from noise) over that full width, but
0.12-0.29 (real signal) inside 2.5-block windows of the *same patch*. A single
global phase assumes the predicted seams stay in step with the real ones
everywhere; they do not, because calibration's own small residual error
compounds linearly with distance from wherever the phase was anchored, and over
an 18-block patch that drift exceeds the tolerance band.

`support_cells` now accepts `local_window_periods`, reading each cell's phase
from a window a few periods wider than the cell itself rather than one phase for
the whole plane (`plane_from_evidence` uses it by default, at 1.5). Chance was
re-measured for this scheme rather than assumed - an unbiased circular-mean
estimator does not inflate chance the way a max-over-offsets search would, and
that held: 0.365-0.369 mean, ≤0.41 at p99, across window sizes 1.0-4.0 periods,
matching the earlier global-phase chance level closely.

**On real data, in the best-supported local window found, the nominal-period
score (0.466) clears the empirical chance ceiling (max 0.409 over 6,144 noise
samples) - the earlier "the cue does not work" was based on the global test
alone, and does not hold as stated.** It remains below `SUPPORT_THRESHOLD`
(0.55), which was calibrated against the global-phase chance distribution and
has not been re-validated as the right cut for this tighter, local-window
distribution. Widening the tolerance does not help: `_phase_agreement`'s chance
level tracks almost exactly `2 * tolerance` for random data (0.18→0.367,
0.25→0.507, 0.35→0.705), so a wider band inflates chance and signal together and
buys nothing. I have not re-run the full Template 2 harness's precision/recall
with a threshold recalibrated to the local scheme, and I am not lowering
`SUPPORT_THRESHOLD` on the strength of one hand-picked window - that would be
fitting the threshold to a single data point.

So the corrected state is: **weak but genuine above-chance local signal exists,
and does not yet clear the bar for calling a cell supported.** Not dead, not
working. The honest next steps, not yet done: measure the local-window score
distribution across many cells and views (not one chosen window), and decide a
threshold from that distribution rather than from this one example.

Re-running the same three-view harness with local windowing now the default
(`docs/reports/M4-supported-plane-templates-2-local.json`) does not show a
visible change in the summary median/supported counts - consistent with "one
window clears chance, most still don't" rather than a broad fix. That is
further reason not to touch `SUPPORT_THRESHOLD` on this evidence alone.

## Second addendum, same day: the broad measurement came back negative

The immediate next step from the first addendum was to measure the
local-window score distribution against real ground truth across many cells,
not one hand-picked window. I built that measurement: for every judged cell,
reconstruct its world location from the real camera pose (read only for this
check) and ask whether a genuine, camera-reachable reference block face sits
there, then compare score distributions for truth-positive against
truth-negative cells.

**Local-window phase agreement: AUC 0.523** (0.5 is no separation) over 55
truth-positive and 114 truth-negative cells. Mean scores for positive (0.371)
and negative (0.373) cells are statistically indistinguishable. The one
promising window from the first addendum was not representative - it was a
selected example, not a sample, and the broad measurement is exactly the
correction that guards against that.

I then tried a different cue with the same rigor before giving up on
appearance entirely: **brightness-consensus** (does a cell's mean brightness
match the median of its immediate neighbours, on the reasoning that a doorway
onto a different, differently-lit space should be a local outlier). **AUC
0.502** - also chance, slightly worse than phase agreement.

**Five image-only per-cell cues have now been measured and refused to
separate real occupancy from not**: planar region proposal (5-13% of tiles
decided), period strength (no normal identification at all), global phase
agreement (chance), local-window phase agreement (chance, AUC 0.523), and
brightness consensus (chance, AUC 0.502). The last three carry a rigorous,
ground-truth-based measurement, not just a noise comparison. This is a
genuine wall, not a tuning problem, and I am done trying ad hoc appearance
cues on this data without a specific new idea worth naming and testing on its
own merits.

**One real caveat on how broad "broad" actually was:** of the three Template 2
views, only 13.24.33 supplied enough judged cells to matter (168 of 169 total)
- the other two views' floor windows were too small or entirely out of frame
at this axis/side. 13.24.33 views the floor from an elevated position while
pitched steeply upward, which may be an unusually oblique case. I have not
repeated this measurement on a more head-on view or a different fixture, and
would not generalize "per-cell appearance cues cannot work on Minecraft
interiors" from one view's worth of real ground truth - only that none of the
five tried have worked on the evidence gathered so far.

Per-cell occupancy remains unsolved. Rather than continue guessing at
appearance cues, the next M4 work moves to a different, well-scoped piece
that does not depend on solving it: joint refinement of a plane's distance
across the multiple views that share it, building on the registration work
above. See [M4-registration-2026-09-08.md](M4-registration-2026-09-08.md) for
that.

## What I would do next, and what I would not

Written before the addendum above, and superseded by it: I said flatly I would
not tune the cue. Local windowing is not tuning a threshold - it fixes a real
defect in the test (a global phase that cannot fit a plane wider than
calibration stays accurate over), and it recovered above-chance signal a
threshold change alone could not have. That correction stands; what follows is
the honest state after it.

The distance result is worth keeping regardless: a stated length places a plane
to 0.4% from one image. Per-cell occupancy is now in a real middle state -
weak, genuine, above-chance signal, not yet strong enough to call a cell
supported - rather than the dead end this report first concluded. The
immediate next step is measuring the local-window score distribution broadly
(many cells, all three views) before touching `SUPPORT_THRESHOLD` again, since
one hand-picked window is not a distribution.

If broad measurement confirms the signal stays this marginal, the next
candidate is appearance rather than geometry: a cell on the plane renders as a
block face under known face shading at a known angle, and a cell looking past
it does not, using the palette and vanilla shading table the project already
has. That is a new cue and not the bounded prototype that was scoped, so I have
not started it.
