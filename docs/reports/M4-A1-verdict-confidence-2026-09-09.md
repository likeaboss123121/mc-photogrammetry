# M4 A1 — noise no longer earns a confident geometric verdict

**Claude Sonnet 5, 2026-09-09.** Self-directed while M4 B waits for lead
review: Astra's kernel review confirmed a real defect in `support_cells`
(A1) and explicitly noted fixing it was not a prerequisite for the B
assignment. It's in my own ownership and didn't need review to be worth
doing, so I fixed it properly rather than leaving it flagged.

Code: [`supported.py`](../../src/mc_photogrammetry/reconstruction/supported.py).
2 new tests in [`test_supported.py`](../../tests/test_supported.py) (22 total).
Full suite: 445 tests, green.

## The defect, confirmed independently before fixing

Pure noise (128x128, period 16, full coverage - a plane with nothing on it)
produced **44 `contradicted` and 20 `foreground_occlusion` verdicts, zero
`weak_evidence`**. Root cause: `contradicted` was the *unguarded default* -
whatever a cell's score, if `foreground_occlusion`'s coarser-period check
didn't confidently fire, the cell fell straight through to `contradicted`.
Noise clears "not confidently coarser" exactly as easily as a genuine
mismatch does, so the classifier was reporting noise as active geometric
evidence with full confidence.

A second, compounding defect: `_dominant_period`'s own gate for "is there a
periodic scale here at all" was `0.1` - essentially no gate. Measured against
noise's actual edge-gradient autocorrelation (the real input this function
receives, not raw pixel values): **noise finds *some* peak clearing 0.1 on
100% of trials**, with ratios spread from 0.19x to 2.94x the plane's period -
comfortably inside both the `foreground_occlusion` (>=1.6x) and a symmetric
`contradicted` (<=0.625x) band a fair fraction of the time.

## The fix, measured before being chosen

**A properly calibrated peak-strength gate.** Measured noise's edge-gradient
autocorrelation peak strength at the exact window size `support_cells` uses,
over 2,000 seeds: mean 0.195, p99 0.336, p99.9 0.398, max 0.428 - a stable
tail, not still climbing. A modest genuine periodic signal (a plain two-tone
band pattern, well short of real block texture's usual contrast) reached
0.50-0.94 in the same test. `DOMINANT_PERIOD_STRENGTH = 0.45` sits with real
margin above the measured noise ceiling and below the weak end of genuine
signal - not at either boundary.

**A symmetric confidence bar for `contradicted`.** It now needs the same kind
of positive evidence `foreground_occlusion` already required: a confidently
*finer* period (ratio `<= 1/FOREGROUND_PERIOD_RATIO`), not merely the absence
of a confidently coarser one. A cell with no confident period at all, or one
too close to the plane's own period to call farther or nearer, is
`weak_evidence` - real energy present, but this cell does not establish which
kind of disagreement it is.

## Verified, not just asserted

```
seed 123, 128x128 noise, period 16: {'weak_evidence': 64}   # was 44 contradicted / 20 occlusion
20 seeds combined (1,280 cells):     {'weak_evidence': 1280}  # zero false positives
```

The two existing synthetic controls this fix could plausibly have weakened -
`opening` and `foreground-slab`, both built on exactly the mechanism changed
here - were re-measured and are **numerically identical** to before the fix:

```
opening control interior:         {'contradicted': 81}                          # unchanged
foreground-slab control interior: {'foreground_occlusion': 40, 'weak_evidence': 25, 'supported': 16}  # unchanged
```

The stricter bar eliminated the false positives without costing any of the
already-validated true positives.

## What this does and does not change

**Does:** make `support_cells`'s `contradicted`/`foreground_occlusion`
verdicts mean what their names claim - confident, positive evidence of a
specific kind of disagreement - rather than "didn't clear the supported
threshold and wasn't obviously coarser."

**Does not:** make `support_cells` a working per-cell occupancy cue. That
question was answered earlier (AUC ~0.50-0.53 against real ground truth,
chance) and this fix doesn't reopen it - A1 remains quarantined from the M4 B
image-adapter path exactly as Astra's review specified, and this report
doesn't propose otherwise. What it does establish: the diagnostic's own
verdicts are no longer confidently wrong on inputs that carry no evidence at
all, which matters for anyone reading its output regardless of whether the
underlying cue ever becomes reliable.
