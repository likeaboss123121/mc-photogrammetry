# R5 P1-E image-only frozen-run results

Date: 2026-09-13  
Status: exactly one preregistered execution completed; awaiting lead review  
Preregistration: [M4-R5-P1E-image-only-prereg-2026-09-13.md](M4-R5-P1E-image-only-prereg-2026-09-13.md)

The exact ordered eight-image preflight passed. The prescribed `run_frozen_image_pipeline` call ran once with current defaults, completed in 250.158 s, and strict-load/sidecar verification passed. No retry, tuning, evaluation, selector probe, or image-based subset selection occurred. Peak process heap was unavailable.

- R5 artifact ID `3c1c88200c15f32f8305314e5f51d928fc414b026eb6aa039ad3fb8cd45d5cb6`, schema `1`, `labels_present=false`.
- Freeze `exports/evaluation/frozen-image-run-r5-p1e-canal-town-2026-09-13/freeze.json`: 19,426,263 bytes, SHA-256 `8dfc8b41bad1dff018fa08db42411c5f91572233cb1e7718cc17a082ab0c218b`.
- Sidecar `freeze.json.087d379673be1ff291f807753cdb00206ad02a64aa26f91104d24abc36a8db5b.masks.bin`: 89,608,308 bytes, SHA-256 `087d379673be1ff291f807753cdb00206ad02a64aa26f91104d24abc36a8db5b`.
- Exhaustive deterministic diff: [M4-R5-P1E-exhaustive-diff-2026-09-13.json](M4-R5-P1E-exhaustive-diff-2026-09-13.json), 5,285,239 bytes, SHA-256 `9731923b092d28a0d52453a44fadcb8875dd94974c2b4b80339c7cbbe5653089`. It enumerates every upstream proposal/read/signature record, exact match identity delta, full refusal record multiset delta, source hashes, and plane-index audit.

R5 frame stages: 8/8 calibration/image/proposal/readings/signatures ok; 35 proposals, 33 readings, 203 signatures; matching and solution each completed once. Per-frame counts and protected image hashes are in the JSON diff. Proposal mask contents match R4 for all 35 proposals; only sidecar storage offsets differ. Reading identities match but all 33 rows have phase and alternative-distance changes. Signature identities match for all 203 rows; all cells/interior hashes, 107 coverage hashes, 54 origins, and 11 shapes differ. These upstream differences cannot be attributed to P1-E alone.

R4/R5 graph comparison: R4 200 matches, 71,216 refusals, terminal `component_too_large`; R5 69 matches, 71,347 refusals, terminal reason empty (solved), `scale_known=true`. Match identity intersection is zero (200 R4-only, 69 R5-only). Therefore no common-match metric deltas exist; frozen schema v1 does not serialize `cell_shift_rc`. Exact match/refusal lists, quadrant and selected-b strata, and refusal digests are in the diff artifact.

Plane-index audit found 0 signatureless readings, 0 matcher/solver index-map divergences, and 0 match endpoint index/axis/selected-b/signature mismatches across both runs (538 endpoint references total). In the selected R5 assignment, active world-axis edge/cycle counts are axis 0: 4/0, axis 1: 3/0, axis 2: 2/0; no active match maps to different world axes at its endpoints.

R5 selected quadrants in frame order 00..07: `[0, 2, 2, 2, 3, 2, 3, 0]`. Axis resultants: `{'0': 0.96220109759987, '1': 0.9080659485240001, '2': 0.9935638206940539}`; residuals: `{'0': 0.03782358885451724, '1': 0.06245987171234511, '2': 0.018066848601622873}`. The eight alternatives, costs, observations, and resultants are recorded in the diff artifact.

Protected R4 freeze/sidecar posthashes match preflight; all eight capture posthashes match the preregistered input hashes. R5 source hashes are recorded in the diff; the R4 snapshot has older runner, `lattice_planes.py`, and plane-graph hashes. This remains label-blind diagnostic evidence and does not evaluate reconstruction quality.

