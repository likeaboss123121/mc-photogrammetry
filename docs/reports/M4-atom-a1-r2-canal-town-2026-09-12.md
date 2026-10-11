# M4 Atom A1 — R2 image-only atomization result

A1 replaces tile-isolated atom assignment with one deterministic overlapping SLIC-style pass and a bounded 4-connected cleanup. Every seed compares inside a fixed 2S-wide image window against global best-distance/label buffers. Seeds are visited in ascending ID order; strict `<` makes lower IDs win exact ties. Updated color/XY centers use only usable assigned pixels. Initial disconnected components are split with 4-connectivity; components smaller than 64 pixels (`ceil(16² * 0.25)`) merge into the adjacent component with the greatest shared boundary, with stable component ID for ties. Isolated usable islands remain separate. Merges follow an existing shared edge, preserving connectivity; original and final usable masks are compared. If the final connected-atom count exceeds the unchanged 20,000 cap, the proposer returns the specific `atom_connectivity_cap` abstention without dropping usable pixels or merging large components to fit.

The work is bounded by per-seed local windows and the component adjacency graph: O(analysis pixels + seed count + local-window area) work per fixed iteration/cleanup pass and O(analysis pixels + seed/component count) memory. The analyzer remains image-only; no manifest, camera metadata, GT, labels, evaluator, or adjacent metadata was supplied to the run.

## Preregistered validation

Before R2, the preregistration at `docs/reports/M4-atom-a1-prereg-2026-09-12.md` froze the test/config contract. It records the default config digest `dc7f92521f2ec5aaeb8bbbd99496855bd6603fbfb85a1f3867334c8c4d3bf682`, region-growth source SHA-256 `5375bc247defa370baee530434ef9751d9aec8a477ad143599d8fadc5b007e8a`, test SHA-256 `6b00998996781813740049b18dd09cb7d9826f4bfde67096efe5af27a618da12`, and unchanged runner SHA-256 `17408ac780bf27bd25bd8819a02fcf534e77e940b741a075c0a3c7f5bcdab60e`. No seed, plane-growth, or edge threshold was changed.

Focused C4/B1/R1 regressions:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_region_growth.py tests\test_lattice_planes.py tests\test_plane_graph.py tests\test_frozen_image_runner.py tests\test_calibration.py -o addopts= -p no:cacheprovider -q
```

Result: **86 passed in 118.08s**. The suite includes the new checker/stripe, GUI-hole, 4-connectivity, diagonal-island, shared-boundary/tie, exact-seed-cap, unmergeable-island abstention, label-hash determinism, channel-sensitive determinism, and resource/runtime controls. `py_compile` and targeted `git diff --check` passed.

## One R2 pixels-only run

The single preregistered `run_frozen_image_pipeline` call used the eight ordered Canal Town PNG paths listed in the preregistration. Observed command wait time was about 224 seconds. After the runner returned and published the freeze, the small wrapper's summary formatting raised `KeyError: 'stage_counts'` because the schema field is named `stage_reasons`; I read and strictly verified the published artifact without rerunning the pipeline. It is label-free (`labels_present=false`) and binds the expected R2 source hash.

All 8 images calibrated and all 8 proposal stages returned `image_edge_graph_growth`; all 8 reading and signature stages completed. The run produced 35 proposals, 29 readings, and 232 signatures. The graph stage returned normally; component solving reported `component_too_large` (no reconstruction-quality claim is made). Each frame had 2,040 seeds, preserved every usable pixel, and finished below the 20,000 atom cap:

| Frame | 4-connected components before | Cleanup merges | Final atoms | Proposals | Readings | Signatures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 00 | 11,257 | 9,876 | 1,381 | 3 | 2 | 16 |
| 01 | 18,649 | 17,326 | 1,323 | 5 | 5 | 30 |
| 02 | 35,166 | 33,685 | 1,481 | 2 | 2 | 19 |
| 03 | 31,167 | 29,735 | 1,432 | 6 | 5 | 38 |
| 04 | 27,665 | 26,297 | 1,368 | 4 | 4 | 35 |
| 05 | 33,123 | 31,785 | 1,338 | 7 | 6 | 52 |
| 06 | 26,266 | 24,921 | 1,345 | 4 | 2 | 19 |
| 07 | 27,577 | 26,265 | 1,312 | 4 | 3 | 23 |

Totals: 210,870 pre-cleanup components, 199,890 merges, and 10,980 final atoms. All frames have `connectivity_mask_preserved=true`; final IDs are 4-connected by construction and the synthetic suite independently checks connectivity. No frame abstained at atomization.

R2 freeze: `exports/evaluation/frozen-image-run-r2-atom-a1-canal-town-2026-09-12/freeze.json`, 25,083,593 bytes, SHA-256 `3c4e8e1b63f73748bde1f072d97dbfb4276ebc90c957889123760edcc6a1c90f`, artifact ID `74e0435188ba83f4a1a58d50b641c10c887d5887fa0672814394cb7bb4cb3d6c`. Its content-addressed sidecar is `freeze.json.68277b112b4b7ad755878501c8c04617119477f5ceff76be9dff2c0403bc74db.masks.bin`, 127,087,308 bytes, SHA-256 `68277b112b4b7ad755878501c8c04617119477f5ceff76be9dff2c0403bc74db`. The artifact strict-load verified; embedded region-growth hash matches the preregistered source. R1 and E1 artifacts remain unchanged. No evaluator or threshold changes, capture, CLI wiring, staging, or commit.
