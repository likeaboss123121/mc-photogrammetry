# M5 matched Template 2 diagnostic — 2026-09-09

This is the single matched before/after diagnostic authorized by Astra after the L1/L2 controls. It is an evaluation-only, supplied-camera run. It does not promote the partial-shape stage or qualify image-only production.

## Frozen configuration

Both states used the same three decoded frames, supplied camera manifest, scene bounds, frozen evaluator masks, Minecraft 1.21.5 jar palette, light-cast setting, proposal limit (40), and partial-fit candidate cap (256). The baseline first reconstructed with `fit_partial_shapes=False`. The after state ran `residual_proposals` once on that settled baseline and applied `fit_states` only to the resulting frozen 40-cell proposal set. No reference data was used to choose proposals.

- Reference: `templates 2/Template 2.litematic` (852 non-air cells)
- Baseline settled cells: 814
- Frozen residual proposals: 40
- Before reconstruction time: 108.28 s
- After reconstruction plus shape-fit time: 566.75 s
- Shape-fit time: 458.47 s
- Shape-fit render work: 416 total (256 candidate renders; cap reached), 96 ownership renders, 28 probe renders, 33 context renders, 3 background renders

## Matched result

| Measure | Before | After |
| --- | ---: | ---: |
| Candidate blocks | 814 | 818 |
| Exact reference cells | 678 | 678 |
| Partial-family reference cells | 69 | 69 |
| Exact partial-family cells | 0 | 0 |
| Partial-family recall | 0.0% | 0.0% |
| Lost correct cells | — | 0 |
| Added cells | — | 4 |
| Deleted cells | — | 0 |

The four additions were `(2,1,1) slime_block`, `(4,1,14) slime_block`, `(5,1,14) loom[facing=west]`, and `(6,2,1) slime_block`. No previously correct cell was lost. Family recalls remained 0/23 stairs, 0/24 carpets, 0/5 slabs, 0/13 fences, and 0/4 torches.

The stage therefore produced bounded, non-destructive changes on this frozen run, but it did not recover a correct partial-family state. This is evidence for the implementation and budget accounting only; it is not a quality promotion.

The machine-readable details, proposal coordinates, diagnostics, and exact settings are in [M5-t2-matched-diagnostic.json](M5-t2-matched-diagnostic.json). The exported before/after fixtures are `M5-t2-before.litematic` and `M5-t2-after.litematic` in this report directory.
