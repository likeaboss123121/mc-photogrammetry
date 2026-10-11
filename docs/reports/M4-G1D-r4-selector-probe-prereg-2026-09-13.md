# M4 G1D preregistration: one low-budget R4 selector-only probe

Date: 2026-09-13

## Frozen inputs and scope

The only evaluation input is `exports/evaluation/frozen-image-run-r4-l1b-canal-town-2026-09-13/freeze.json`, pinned at SHA-256 `96493dd9f42f1d20c932a8b09f267d7f86da38b7e5ac588dec1330ad9df81c9d`, and its content-addressed mask sidecar `freeze.json.f28cc68b2fc5086473ba3b73c4b277e0d1338c10639c0c6a4b64e94585be0bac.masks.bin`, pinned at SHA-256 `f28cc68b2fc5086473ba3b73c4b277e0d1338c10639c0c6a4b64e94585be0bac`. The G1C selector hash is recorded in the dated amendment below; its original accepted hash was `30338b9ed5851526277a5d65d49a592583ac51f10756904cc79a33f82b4d3bba`.

The probe will deserialize only the already-frozen R4 `PlaneReading` and `PlaneMatch` records and make one call to isolated G1C `select_joint_labels`. It will not load any PNG, GT, evaluator, capture manifest, or sidecar payload; the sidecar is hash-verified only. R4/G1A/G1B/G1C and all E artifacts are read-only. No optimization edits, production wiring, default budget, schema migration, or R5 are in scope.

## Explicit diagnostic budget and stopping behavior

One invocation will use:

| Counter | Limit |
|---|---:|
| search nodes | 4,096 |
| compatibility checks | 1,000,000 |
| objective rows | 20,000 |
| axis solves | 512 |
| Pareto comparisons | 64 |
| buffered candidate certificates | 16 |
| buffered match rows | 1,024 |

The bounded call must return either a complete ordinary result within these ceilings or `budget_exhausted` with no candidate, Pareto entries, or certificate hash. No retry or larger budget is permitted. The output path `exports/evaluation/m4-g1d-r4-selector-probe.json` must be absent before the call; the probe refuses to overwrite it. The result records exact input/source hashes, dimensions of the solver input only (frame/reading/match counts), elapsed time, every counter, terminal status/reason, and whether any partial result escaped.

After the one call, G1D is design-only. Analyze counters to propose behavior-equivalent acceleration candidates and proof obligations against G1C's Cartesian oracle. Preserve incomparable alternatives and q+2 ambiguity. Do not edit or benchmark optimization code. Stop for lead review.

## Pre-call correction: optional null diagnostics

The first probe-script attempt stopped during metadata preflight, before `select_joint_labels` was called; no selector call occurred and neither the output nor temporary output was created. R4 contains `psr=null` in 6/200 matches and `integer_residual=null` in 200/200 matches. The G1C selector converted `psr` to float during preparation, so it would reject the frozen records before the intended search. The lead authorized this narrow isolated-selector input-compatibility fix before the one probe: retain null `psr` as an explicit tagged optional diagnostic in deterministic sort/dedup, while keeping score, axes, and both `b` values required finite; do not replace null with any numeric sentinel. Add bounded/oracle/permutation coverage for null `psr`, rerun the focused G1C regression suite, then pin the corrected selector and this preregistration's final SHA-256 below before making the single selector call. No production wiring or optimization is authorized.

Corrected selector SHA-256: `367329861d3838ed3df360bde9b2735242f611ce0e25bfecb68cf26ac2f0298b`.
Null-PSR regression source SHA-256: `62eb03d5d252163663ab26fd7370de511fcb6dc4feb4ecc3fb45427d1f408933`.
The selector-specific regression passed 20/20 using `.venv\Scripts\python.exe -m pytest tests\test_joint_plane_selector.py -q`; the full G1C regression suite is recorded in the dated result report after it completes.
The byte-exact preregistration SHA-256 is pinned by the probe script and repeated in its output/report.
