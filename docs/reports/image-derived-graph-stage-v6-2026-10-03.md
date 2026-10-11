# Image-derived graph-stage attribution — 2026-10-03

## Scope and result

One evaluation-only, contained run used the four frozen PNGs from the DS-R5D package. The normal image-derived path completed entry, lattice and signatures, then graph attribution. It produced 72 signatures and completed the graph stage, but geometry remained unavailable (`plane_integer_residual`). No thresholds, production outputs, images, cameras, or inputs were changed. This is one fixture run and does not establish visible-state accuracy or truth correspondence recall.

## Measured internal funnel

| Trace stage | Count | Meaning |
|---|---:|---|
| Pre-admission | 7,476 | Pair hypotheses emitted by the matcher across signature pairs/quadrants; not independent correspondences or truth labels. |
| Post-admission | 1,495 (20.0%) | Hypotheses passing the graph admission policy. |
| Post same-axis filter | 54 (3.61% of admitted; 0.72% of pre-admission) | Matches passed to component solving after same-axis filtering. |
| Component solution | unavailable | Best assignment failed the fixed resultant gate on all three axes. |

The 7,476 pair-trace reason values were 4,982 `axis_mismatch`, 864 `no_signal`, 95 `insufficient_overlap`, 40 `repetition_unresolved`, and 1,495 admitted. These are policy outcomes over generated hypotheses, not measured image accuracy. The highest numeric rejection count (`axis_mismatch`) is expected filtering and is not by itself evidence of the dominant product loss.

The solver reported axis `(resultant, residual, observations)` as `(0, .3915, .1740, 12)`, `(1, .2472, .2017, 6)`, `(2, .2452, .1872, 10)`. All resultants are below `.90`; all residuals also exceed `.15`, but the failure diagnostic reports `first_failed_predicate=minimum_resultant` and `residual_predicate_evaluated=false`. Fixed settings were `max_shift_cells=32`, `min_overlap_cells=9`, `min_structural_share=.15`, `integer_tolerance=.15`, `min_resultant=.90`, `ambiguous_margin=.02`.

The failure-only active-edge alias record has 14 edges across axes `[6,3,5]` and 8 conflicting endpoints. This is a real inconsistency signal in the solver's best assignment, but the single trace does not establish whether alias ambiguity caused low coherence or is a consequence of false/misaligned matches. Boundary saturation is not a leading explanation: 38/1,495 admitted rows and 2/54 solver-input rows touched the ±32 boundary. Component-incidence counts are `[19,19,0]` admitted and `[1,1,0]` kept; these are per-axis incidences, not row counts. Widening the radius is unsupported by this evidence.

## Next discriminating work

Use the retained project/trace for a bounded evaluation-only re-solve/ablation, if the existing typed seams permit it: compare baseline solver output with a counterfactual that removes only boundary-touching matches, then separately removes active-edge-alias-conflicting endpoints. Record per-axis resultants/residuals and preserve the production result. If the current persisted artifacts do not contain enough replayable `PlaneMatch`/reading state, stop and specify the smallest evaluator seam needed; do not reconstruct undocumented state from pair counts. A further matcher change or threshold relaxation needs evidence that a specific class of accepted correspondences is wrong or missing.

This graph-stage measurement and the supplied-camera T2/T6 texture/state attribution measure different parts of the system. The latter remains the only current report with visible-truth percentages. T6's largest measured default loss is palette support pruning; graph-stage counts must not be added to that percentage funnel.

## Provenance

- Request/argv SHA-256: `aa79676488157e416477af86b3b0b239b5758e9c20a51afc9f2b5107e08e3386` (2,085 UTF-8 bytes).
- Manifest digest: `ebbc06670accab1d7ce98fdfd7718613605b62ca0f8b94b2a43996309d1973ca`.
- Runtime: 142.234 s; terminal exit 0; child Job empty after cleanup; peak reported RSS 227,651,584 bytes.
- Stage artifact `.tmp-axis0-graph-stage-attribution-v6-20261003.json`: SHA-256 `7688538f4362dad87ef8c0013717b6b83230682457680fc4fb55cbd22e8656c1`.
- Project `.tmp-axis0-graph-stage-attribution-v6-20261003.mcscan`: SHA-256 `a85fe65babfd4d07e257b3451dacdb58e6b5f527c078313fd86d58cbeb727f87`.
- Trace SHA-256 `eb2fabfa1e2acc907e1397e61430cbbb9cb2a2e785ce516b8c66b8d60b2ac176`; graph result SHA-256 `83eb7fb3b995b9d89bdccedef7480f40daf0793564191386d9f4024c81226ae4`.
- Terminal `.tmp-axis0-graph-stage-attribution-v6-20261003.terminal.json`: SHA-256 `b29e444abbeaccf9d99bff3ca9259bce43bced24ac7b0ae84edfa416a1fab4ca`.
- All 9 producer/helper and 4 ordered input PNG hashes matched the request at preflight. The contained runner used its 8-GiB process/Job limits and 600-second deadline. `acceptance=false`; geometry unavailable.
