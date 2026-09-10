# M5 matched Template 2 diagnostic v2 ? 2026-09-09

This is the single corrected diagnostic required by Astra's D1?D4 review. It is a supplied-camera evaluation run, not a production promotion. The settled baseline and 40-cell proposal inventory are frozen from the earlier diagnostic; because the original processing order was not retained, v2 declares lexicographic `(x,y,z)` order.

## Preflight and reproducibility

- Asset-derived `complete_palette`: **6163** entries, **5415** non-full states, **38** geometry groups.
- Eligible partial families: {'bars': 16, 'carpet': 67, 'fence': 400, 'pane': 272, 'slab': 114, 'stairs': 1296, 'torch': 23, 'wall': 1434}.
- Candidate render cap: **256**; shape scale 0.5; edge weight 0.25; texture weight 40.0.
- Frozen proposal processing order: lexicographic unique v2 inventory, recorded in the JSON report.
- Input, palette, JAR/cache, source revision/status, baseline/output hashes, effective settings, masks, cameras and frames are recorded in [M5-t2-matched-v2.json](M5-t2-matched-v2.json).

## Result

| Measure | Before | After |
| --- | ---: | ---: |
| Candidate blocks | 814 | 815 |
| Exact reference cells | 678 | 678 |
| Partial-family cells | 69 | 69 |
| Exact partial-family cells | 0 | 0 |
| Partial-family recall | 0.0% | 0.0% |
| Added / deleted / state-changed cells | ? | 1 / 0 / 0 |
| Lost exact cells | ? | 0 |
| Phantom cells | 34 | 34 |

Only `(1,1,12)` was reached before the global cap and it selected `minecraft:spruce_stairs`; the other 39 proposals were explicitly recorded as `not_reached`. The reached cell was marked `partly_tested` because the cap stopped during material refinement. It was an addition, not an exact reference state.

Shape fitting performed **322** total render work, including **256** candidate renders, and took **513.11 s** with a peak Python heap of **368,476,514 bytes**. Cell decision counts are {'not_reached': 39, 'partly_tested': 1}.

The complete palette makes partial recovery eligible for evaluation, but this run still achieved 0/69 exact partial-family recall and added one phantom. It is evidence about the bounded fitter and harness only; no production quality promotion follows.
