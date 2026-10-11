# M4 Atom A1 preregistration — 2026-09-12

This file freezes the synthetic controls and atomization configuration before R2 reads any Canal Town pixels. No ground truth, capture metadata, evaluator, labels, or neighboring metadata were consulted to define these choices.

## Frozen implementation and tests

- `src/mc_photogrammetry/reconstruction/region_growth.py` SHA-256: `5375bc247defa370baee530434ef9751d9aec8a477ad143599d8fadc5b007e8a`
- `tests/test_region_growth.py` SHA-256: `6b00998996781813740049b18dd09cb7d9826f4bfde67096efe5af27a618da12`
- unchanged R1 runner `src/mc_photogrammetry/reconstruction/frozen_image_runner.py` SHA-256: `17408ac780bf27bd25bd8819a02fcf534e77e940b741a075c0a3c7f5bcdab60e`
- canonical default `RegionGrowthConfig` SHA-256: `dc7f92521f2ec5aaeb8bbbd99496855bd6603fbfb85a1f3867334c8c4d3bf682`
- cleanup minimum component fraction: 0.25 of `atom_spacing²`; at spacing 16 this is 64 pixels. This is an atom-connectivity cleanup setting, not a seed, plane-growth, or edge threshold. All existing seed/growth/edge values remain unchanged.

The exact default proposal configuration is:

```json
{"analysis_max_side":960,"atom_spacing":16,"bridge_edge_barrier":15.0,"connectivity_min_component_fraction":0.25,"dedup_iou":0.7,"hard_edge_barrier":48.0,"max_atoms":20000,"max_bridge_lab_delta":22.0,"max_candidates":32,"max_input_pixels":6000000,"max_lab_delta":40.0,"max_segment_samples":128,"max_segments":5000,"min_boundary_axis_alignment":0.9,"min_bridge_support":18.0,"min_calibration_inliers":12,"min_continuation_support":5.0,"min_growth_score":0.055,"min_region_fraction":0.004,"min_seed_margin":0.035,"min_seed_score":0.26,"min_seed_support":12.0,"min_tangent_dominance":0.84,"min_usable_fraction":0.2,"slic_compactness":10.0,"slic_iterations":3}
```

Frozen synthetic suite:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_region_growth.py tests\test_lattice_planes.py tests\test_plane_graph.py tests\test_frozen_image_runner.py tests\test_calibration.py -o addopts= -p no:cacheprovider -q
```

Result before R2: **86 passed in 118.08s**. Atom-specific controls include checker/stripe texture, GUI-shaped and interior mask holes, diagonal and tiny isolated usable islands, deterministic lowest-seed ties, greatest shared-boundary and stable-ID merging, exact mask preservation, exact seed/cap boundary, typed abstention for unmergeable islands, repeated-run label hashes, RGB channel-sensitive determinism, and resource/runtime caps.

## Frozen R2 inputs and output

The only runtime inputs are these eight ordered PNG paths:

```text
exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox/captures/view-canal-town-architectural-component-bbox-00.png
exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox/captures/view-canal-town-architectural-component-bbox-01.png
exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox/captures/view-canal-town-architectural-component-bbox-02.png
exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox/captures/view-canal-town-architectural-component-bbox-03.png
exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox/captures/view-canal-town-architectural-component-bbox-04.png
exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox/captures/view-canal-town-architectural-component-bbox-05.png
exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox/captures/view-canal-town-architectural-component-bbox-06.png
exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox/captures/view-canal-town-architectural-component-bbox-07.png
```

The only planned output is the new path-only freeze `exports/evaluation/frozen-image-run-r2-atom-a1-canal-town-2026-09-12/freeze.json` and its content-addressed mask sidecar. The output root was absent at preregistration. The workspace interpreter will call `run_frozen_image_pipeline` directly with the ordered paths above. No metadata or labels are runtime inputs.
