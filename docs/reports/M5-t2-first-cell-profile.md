# M5 Template 2 first-cell profile — 2026-09-09

This is the bounded L-C control requested in Astra’s handoff. It profiles only
the first frozen v2 proposal, `(1, 1, 12)`, using the supplied cameras, frozen
evaluation masks, complete local 1.21.5 palette, and texture weight `40`. The
candidate-render cap is `12`; no full diagnostic pass was started.

Preparation took **8.64 s**. The fit itself took **50.27 s** and used exactly
12 candidate renders, reaching all three usable views before the cap stopped
material refinement. Peak traced Python heap was **368,282,341 bytes**.

The fit-only stage timings were:

| stage | seconds |
| --- | ---: |
| render | 40.336777 |
| ownership | 5.511819 |
| rectification | 0.736879 |
| texture | 0.009203 |
| fitter setup | 3.595251 |

The cell was marked `partly_tested` because the cap ended after the complete
three-view representative pass and before all material refinements. Its
representative selection was `minecraft:mud_brick_stairs`; this is a diagnostic
measurement, not a promotion decision. The machine-readable provenance and
diagnostics are in [`M5-t2-first-cell-profile.json`](M5-t2-first-cell-profile.json),
which also records SHA-256 hashes for the executed fitter and profile runner.
