# M4 R5 preregistration: P1-E corrected image-only graph

Date: 2026-09-13  
Status: approved for exactly one execution  
Dependency: P1-E accepted after 346 independent checks

## Question

Measure how the corrected one-cell-per-block grid map changes image-only plane matching
and graph status on the exact immutable R4 PNG inventory. R5 is label-blind diagnostic
evidence. It is not reconstruction-quality evaluation or production promotion.

## Inputs

Use only these eight PNGs, in this order, under:

```text
exports/datasets/fresh-c4-canal-town-framed-2026-09-12/
  scenes/canal-town-architectural-component-bbox/captures/
```

| file | size | SHA-256 |
|---|---:|---|
| `view-canal-town-architectural-component-bbox-00.png` | 1280x720 | `48ba0ff58c77b588acd5aa06a2c630fc2f5103c6d0d67b45b4bc21b81ecf533c` |
| `view-canal-town-architectural-component-bbox-01.png` | 1600x900 | `8750bb197329eb50144896264db40eef70a6c44cc3dce60eac3a4f1e9aa79696` |
| `view-canal-town-architectural-component-bbox-02.png` | 1920x1080 | `2d017c8f804999446f54dbc742fb87d7748af76ad50249c0e2853754c55b6a95` |
| `view-canal-town-architectural-component-bbox-03.png` | 1920x1080 | `e71bfda3ece7a0b74d2da2fd62834645fcbcf662fb4b5889fae95fa26c9a09e9` |
| `view-canal-town-architectural-component-bbox-04.png` | 1280x720 | `8797d70e0e09475d296c3cf18e5892f6b22d41750ed18d6c3364da80ba60a81f` |
| `view-canal-town-architectural-component-bbox-05.png` | 1600x900 | `6f844d5e3600e853bf3045fc25b55a271c0b665226053602b6d30c945f0f016b` |
| `view-canal-town-architectural-component-bbox-06.png` | 1600x900 | `53c38526d378a0597ab6e00fb5e01a6c7aa0e2dc11619c1b7eaa3cfc0ccf8923` |
| `view-canal-town-architectural-component-bbox-07.png` | 1280x720 | `fdd8a3709479795b204f67153b10f44b49e85f2bc0050a46d9fdd0461f025556` |

Preflight all names, dimensions, hashes, order, and absence of the output directory.
Do not open neighboring capture metadata, plan files, manifests, world coordinates,
references, schematics, context/target GT, masks, or evaluation artifacts as inputs.

## Frozen invocation

Call only:

```text
run_frozen_image_pipeline(ordered_png_paths,
  exports/evaluation/frozen-image-run-r5-p1e-canal-town-2026-09-13/freeze.json)
```

Use the runner's current fixed schema-v1/default configuration. No parameter override,
threshold change, retry, tuning, alternate order, second output, or diagnostic mutation
is permitted. The output directory and both final/temp freeze targets must be absent at
preflight. If input/source preflight fails, do not run.

Code snapshot before execution must include exact hashes for runner, GUI mask,
calibration, region growth, lattice, lattice planes, plane graph, and any other source
the runner itself records. P1-E's accepted `plane_graph.py` hash is:
`c0326d9a300e77aee68ed835e6fc035e397560b7b0d95709edb5f8b65b687ce6`.

## R4 comparison frozen before R5

R4 remains immutable at:

```text
exports/evaluation/frozen-image-run-r4-l1b-canal-town-2026-09-13/freeze.json
```

- freeze SHA-256: `96493dd9f42f1d20c932a8b09f267d7f86da38b7e5ac588dec1330ad9df81c9d`
- artifact ID: `7d0a08f4a4eb09ae8b5e4568356b515a1f186194e2d51eda2b037a244b9f1ec0`
- stages: 8 calibrations, 35 proposals, 33 readings, 203 signatures
- graph: 200 matches, 71,216 refusals, terminal `component_too_large`
- saved legacy plane-graph hash:
  `3e330b5358542d618a42b7c35b105536b858f38791ab48350f45837afae2688a`

Hash R4 JSON/sidecar and all eight PNGs immediately before and after R5. Any mutation
invalidates R5. Never rewrite R4 or reuse its output directory.

## Required report

Strictly load and verify the new freeze, then report:

- runtime, output sizes/hashes, artifact ID, sidecar identity, `labels_present`;
- per-frame calibration/proposal/reading/signature status and aggregate counts;
- whether upstream proposal/readings/signature arrays match R4 by exact identities and
  content hashes, with every difference enumerated rather than assumed to be P1-E;
- R4/R5 total matches and refusals;
- exact match-identity intersection, additions, removals, and for common identities all
  changes in cell shift, peak, second peak, PSR, overlap, score, or refusal;
- counts stratified by relative quadrant and selected left/right `b` pair;
- graph solution reason, frames, scale status, axis resultants/residuals, and alternatives;
- peak process heap if available without changing the runner, otherwise state absent;
- exact source hashes and pre/post protected hashes.

This comparison may read the two verified freeze records only after R5 publication. It
must not read their sidecar arrays except through the strict loader and only for exact
content identity checks. It must not inspect images to choose a subset or configuration.

## Interpretation fixed before execution

- Upstream changes are evidence of another source/dataflow change and must be isolated;
  they cannot be attributed to P1-E automatically.
- Match-count increase is not quality proof. Changed shifts/PSR/scores describe the
  corrected coordinate convention but have no labels.
- `component_too_large`, ambiguity, no matches, or another abstention is an honest result.
- A solved graph is still label-blind and cannot promote reconstruction quality.
- Do not evaluate, tune, rerun, select a better subset, or change thresholds after seeing
  R5. Any follow-up receives a new name and preregistration.

## Exit

Publish one report and append concise results to `AGENT_COMMS.md`; stop for lead review.
Do not implement C2-R/V2, run a selector probe, evaluate against GT, wire CLI/renderer/
export, or capture new images inside R5.
