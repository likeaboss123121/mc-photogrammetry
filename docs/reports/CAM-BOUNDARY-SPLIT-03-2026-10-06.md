# CAM-BOUNDARY-SPLIT-03 evaluation-only A/B study

CAM_SUPPORT_EXECUTION_STATE: COMPLETE

Preregistration SHA-256: `7c178366e8f9398afe056dc267abbd575dce3694e8e4fc3fc66817ee8371c547`.

A is the unchanged R2 segment splitter. B changes only its exclusion mask: for each in-image horizontal or vertical adjacent pair with differing automatic-mask values, mark both pixels; dilate seeds with a fixed 7x7 square (radius-three Chebyshev) and constant-zero exterior; union the band with the original automatic mask. Genuine edge transitions remain seeds; no outside-image transition is invented. The expanded mask is passed only to segment splitting and never to the zero-fill, enhancement, or LSD path. The band implementation and report are pinned in JSON.

The cohort contains six ordinary and six stress cases (two GUI-boundary, two crop-boundary, two texture/geometry), with 12 unique RGB hashes disjoint from all R0/R1/R1A/R2 registry entries. Base-family labels, order, seeds, recipes, source/runtime pins, RGB bytes and truth references are frozen in JSON. The whole cohort uses one deterministic parent-seed draw for family camera values within R2 ranges; paired stress cases inherit their base camera, while R2's singleton eye value is preserved. The original automatic mask remains the only mask supplied upstream.

The accepted bounded gzip-member store and truth-isolated one-shot runner are reused. The unchanged five gates compare B to A; refusals are not accuracy recoveries. The single authorized synthetic inference/scoring run completed and is awaiting Astra final review. No production integration or product accuracy claim is authorized.


## Paired outcomes

| Case | Stratum | A | B | Raw segments | Retained | Samples |
|---|---|---:|---:|---:|---:|---:|
| boundary-r3-ordinary-planar-a | ordinary | wrong | refusal | 22 | 5 | 3993 |
| boundary-r3-ordinary-planar-b | ordinary | wrong | refusal | 21 | 6 | 3795 |
| boundary-r3-ordinary-stepped-a | ordinary | refusal | refusal | 13 | 1 | 3546 |
| boundary-r3-ordinary-stepped-b | ordinary | wrong | refusal | 22 | 9 | 4485 |
| boundary-r3-ordinary-occluding-a | ordinary | wrong | wrong | 29 | 15 | 5250 |
| boundary-r3-ordinary-occluding-b | ordinary | wrong | refusal | 21 | 8 | 4058 |
| boundary-r3-stress-gui-1 | gui_contamination | wrong | refusal | 22 | 5 | 3993 |
| boundary-r3-stress-gui-5 | gui_contamination | wrong | wrong | 29 | 15 | 5250 |
| boundary-r3-stress-crop-2 | crop_boundary | wrong | refusal | 21 | 7 | 3417 |
| boundary-r3-stress-crop-4 | crop_boundary | wrong | refusal | 18 | 4 | 3602 |
| boundary-r3-stress-low-texture | texture_geometry | wrong | refusal | 17 | 4 | 4995 |
| boundary-r3-stress-multidepth | texture_geometry | wrong | refusal | 21 | 7 | 4046 |


## Frozen gates

A passes: 0/12; B passes: 0/12; net gain: 0; lost A passes: 0.
Stress wrong cases: A 6/6, B 1/6, newly wrong 0.
Promotion gate: FAIL.
Synthetic camera diagnostics only; no product geometry/state accuracy or production integration claim is made.
Full traces are in the adjacent `.payloads.gz`; the JSON contains member descriptors, evaluations, runtime/source pins and per-gate values.
