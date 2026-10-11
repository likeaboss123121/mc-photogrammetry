# M4 P0 â€” endpoint-inclusive rectified spacing

**Date:** 2026-09-13  
**Scope:** one exact pixel-step contract across image-derived lattice consumers. No thresholds, distance domains, graph objectives, selector semantics, schemas, or CLI modes changed.

`rectify_plane` samples each axis with `numpy.linspace(-extent, +extent, samples)`. Its exact spacing is therefore `2*extent/(samples-1)`. The former `/samples` arithmetic overstated inferred plane distance by the reciprocal endpoint factor; the corrected distance is smaller by `1/samples` of the old value. A validated public `lattice.rectified_pixel_step(extent, samples)` now defines this spacing and rejects non-finite/non-positive extents, booleans/non-integer counts, and fewer than two samples.

The helper is used by `rectified_pixel_to_plane_coordinates`, `period_hypotheses`, `_distance_alternatives`, signature scale validation, `supported.plane_from_evidence`, and `scripts/score_separation.py`. `plane_signature` resolves its requested `b` to exactly one retained `DistanceAlternative`, canonicalizes to that alternative, and uses its stored distance, including for legacy/manual readings. `PlaneReading.distance_for` now returns `None` for missing, invalid, or ambiguous alternatives. `_cell_homography` uses those retained distances directly; image matching records the typed `invalid_distance_alternative` refusal rather than choosing a duplicate or independently recalculating scale. `plane_window`'s return annotation now reflects its four values.

For fixed extent and pixel period, `new_step/old_step = N/(N-1)` and `new_distance/old_distance = (N-1)/N`. For `extent=2`, period `32 px`, the inferred one-block distance changes as follows:

| Samples | Previous distance | Endpoint-inclusive distance | Distance change |
| ---: | ---: | ---: | ---: |
| 64 | 0.500000 | 0.492188 | -1.5625% |
| 128 | 1.000000 | 0.992188 | -0.78125% |
| 512 | 4.000000 | 3.992188 | -0.1953125% |

Tests analytically compare the helper to `linspace` endpoints across multiple counts; verify period and retained-distance conversions; verify signature grid sizing and graph homography/warp parity; reject absent or duplicate alternatives; and exercise the supported-plane minimum-period boundary where the old formula accepted exactly 2 px but the actual endpoint-inclusive grid is below 2 px. The evaluation script was source-updated for consistency but not run.

Validation: `pytest tests/test_lattice.py tests/test_lattice_planes.py tests/test_plane_graph.py tests/test_supported.py` â€” **132 passed**; `pytest tests/test_joint_plane_selector.py tests/test_plane_lattice_entry.py` â€” **66 passed**; the new return-annotation control â€” **1 passed**; focused `py_compile` across the changed modules/scripts/tests passed. The accepted R4 L1B freeze was strict-loaded only for compatibility; artifact ID `7d0a08f4a4eb09ae8b5e4568356b515a1f186194e2d51eda2b037a244b9f1ec0`, freeze SHA-256 `96493dd9f42f1d20c932a8b09f267d7f86da38b7e5ac588dec1330ad9df81c9d`, and mask-sidecar SHA-256 `f28cc68b2fc5086473ba3b73c4b277e0d1338c10639c0c6a4b64e94585be0bac` matched their pinned values. No score/replay was run and the freeze/sidecar bytes were not modified.

Whole-file SHA-256 values at closeout (some files also contain previously accepted work):

| File | SHA-256 |
| --- | --- |
| `src/mc_photogrammetry/reconstruction/lattice.py` | `6fcde3a7e8d451e207ec0eed70c26a3f1746ab89f9ac8cf7bba5bcb8fd7bb17e` |
| `src/mc_photogrammetry/reconstruction/lattice_planes.py` | `a1ec9c378edab36bb1382333c64db844867299a4c39ed8804e42783adc37e6af` |
| `src/mc_photogrammetry/reconstruction/plane_graph.py` | `947c5a18e47c6cd96b0b798946f2a23b2a3ad45a3700bac4086a5c9d1b474492` |
| `src/mc_photogrammetry/reconstruction/supported.py` | `c3afaed0c52bb38275bd241c40e86fc4981b0f9ff43de03e3e0c5136a53d3ab5` |
| `src/mc_photogrammetry/reconstruction/__init__.py` | `4e156c8be9374ceb62739a102557b7d47f2845a3bd51f864e615c05f9f12eeae` |
| `scripts/score_separation.py` | `e45899178d6d2da26d452c83a2110b2c48629fd9f6d53ebe8555da578e29e774` |
| `tests/test_lattice.py` | `cf73c67f06b2c401fc297e5fc1c114e6e371f6c7086d9728d6b84b52ec79a882` |
| `tests/test_lattice_planes.py` | `0c07c3019f3a7ad673d648ab112f285b573c8d33d724f8678496015ad0d7f514` |
| `tests/test_plane_graph.py` | `82d256936a8d6f26085641543507df3cf2d60c1e8e3ec2b57bfa6148ee94f586` |
| `tests/test_supported.py` | `eefa6ee32a4e3c50c23002f16d86f93c440ae18b2b08c6270672db5fd73fa3a2` |
