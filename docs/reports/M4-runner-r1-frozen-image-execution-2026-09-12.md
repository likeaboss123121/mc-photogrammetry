# M4 Runner R1 — Frozen Image-Only Execution

Date: 2026-09-12

## Contract and implementation

Added `reconstruction.frozen_image_runner.run_frozen_image_pipeline(image_paths, freeze_path)`. Its only data inputs are an ordered list of RGB image paths and one JSON freeze path. It decodes those image bytes, derives the default GUI usable-pixel mask, invokes independent per-frame calibration, C4 region growth, the B1 proposal-to-reading adapter, signatures for every retained distance alternative, graph matching/refusal, and component solving. It has no dataset/capture/ground-truth/reference/manual-mask input surface, is not connected to the normal CLI, and its package does not contain an evaluator.

`calibration.estimate_image_calibrations` provides the image-only calibration seam. It zeros default-GUI-excluded pixels before the existing camera fitter and estimates frames independently, including equal-resolution frames. The current masked-fill boundary segments remain part of that fitter's existing behavior; a poison control confirms excluded source pixels cannot change its input. Shared-session focal pooling remains outside this runner because capture settings are not supplied to it.

Schema v1 records ordered input basenames, dimensions and byte SHA-256 values; exact GUI/calibration/proposal/reading/signature/matching/solver configs; relevant source hashes; typed per-stage reasons; calibration/proposals/readings/signatures/matches/refusals/solution; and `labels_present=false`. Proposal masks are intersected with the derived usable mask into new arrays, packed into a content-addressed binary sidecar, and never alter producer masks. Signature arrays share that sidecar as bounded little-endian float32. The strict loader checks duplicate/unknown/forbidden keys, JSON non-finite values, content IDs, sidecar and array digests, shape/resource limits, ranges, and frame/input consistency. The producer runs the same schema validator before publication. Sidecar and JSON use sibling-temp atomic replacement, with the JSON published last; content-addressed sidecars preserve a prior valid freeze if JSON replacement fails.

The artifact ID is canonical over the caller-ordered input inventory, configs, code hashes, and outputs; it omits timestamps and absolute paths. Repeating the same ordered basenames/bytes at different source and output paths reproduced the same ID. Reversing the explicit input sequence changes the ID, as the order is part of the runner contract.

## Synthetic verification

`tests/test_frozen_image_runner.py`: **12 passed**. Controls cover full synthetic pipeline serialization, path/output-independent IDs and explicit order, tampered JSON and sidecar, unknown/forbidden keys, NaN/Infinity rejection, oversized descriptor shapes, content-addressed atomic publication and old-pair preservation, producer-forbidden-stat abstention, blank-image typed abstention, equal-size independent calibration, GUI-mask poison invariance, and the path-only public API/import surface.

Focused regressions for calibration, lattice planes, plane graph, and region growth were run with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -o addopts='' tests/test_calibration.py tests/test_lattice_planes.py tests/test_plane_graph.py tests/test_region_growth.py
```

Result: **67 passed**. The R1 suite command `.\.venv\Scripts\python.exe -m pytest -q -o addopts='' tests/test_frozen_image_runner.py` passed **12**. Together, all 79 focused tests across the runner and its consumed calibration/proposal/reading/graph modules passed. The real-image run was performed exactly once.

## One-shot pixel-only run

Input paths were the eight explicit files `view-canal-town-architectural-component-bbox-00.png` through `-07.png` under `exports/datasets/fresh-c4-canal-town-framed-2026-09-12/scenes/canal-town-architectural-component-bbox/captures/`. The runner received only those PNG paths and the output path below; no neighboring capture, plan, ground-truth, or reference file was opened. Runtime was 77.252 seconds.

| Frame | Dimensions | PNG SHA-256 | Calibration | Proposal result | Readings | Signatures |
|---|---:|---|---|---|---:|---:|
| 00 | 1280×720 | `48ba0ff58c77b588acd5aa06a2c630fc2f5103c6d0d67b45b4bc21b81ecf533c` | ok | `image_edge_graph_growth`, 3 | 2 | 16 |
| 01 | 1600×900 | `8750bb197329eb50144896264db40eef70a6c44cc3dce60eac3a4f1e9aa79696` | ok | `insufficient_atoms` | 0 | 0 |
| 02 | 1920×1080 | `2d017c8f804999446f54dbc742fb87d7748af76ad50249c0e2853754c55b6a95` | ok | `insufficient_atoms` | 0 | 0 |
| 03 | 1920×1080 | `e71bfda3ece7a0b74d2da2fd62834645fcbcf662fb4b5889fae95fa26c9a09e9` | ok | `insufficient_atoms` | 0 | 0 |
| 04 | 1280×720 | `8797d70e0e09475d296c3cf18e5892f6b22d41750ed18d6c3364da80ba60a81f` | ok | `insufficient_atoms` | 0 | 0 |
| 05 | 1600×900 | `6f844d5e3600e853bf3045fc25b55a271c0b665226053602b6d30c945f0f016b` | ok | `insufficient_atoms` | 0 | 0 |
| 06 | 1600×900 | `53c38526d378a0597ab6e00fb5e01a6c7aa0e2dc11619c1b7eaa3cfc0ccf8923` | ok | `insufficient_atoms` | 0 | 0 |
| 07 | 1280×720 | `fdd8a3709479795b204f67153b10f44b49e85f2bc0050a46d9fdd0461f025556` | ok | `insufficient_atoms` | 0 | 0 |

Aggregate stage counts: 8/8 image decodes and calibrations; proposal growth returned candidates for 1/8 and abstained `insufficient_atoms` for 7/8; the one successful frame produced 2 readings and 16 signatures; graph matching returned 0 matches and 0 refusals; the component solver returned typed `no_matches`. This records pixels-only pipeline behavior and makes no claim about labels or reconstruction accuracy.

Freeze: `exports/evaluation/frozen-image-run-r1-canal-town-2026-09-12/freeze.json`, SHA-256 `60eb97973473596ca7ba44c9c305e386615043ff336d9bd81d30e2de28744231`, artifact ID `193536b469d60fec210d7571c8f2537816ac3a6c3102c0889a4040c3eee1734e`. Sidecar `freeze.json.e55383ac4279a64137497823ff4f7ca07199fc56fff980e577a9dcb36767115c.masks.bin`, SHA-256 `e55383ac4279a64137497823ff4f7ca07199fc56fff980e577a9dcb36767115c`, 5,397,420 bytes. The strict loader verified the written pair.

Code hashes frozen in the artifact: runner `17408ac780bf27bd25bd8819a02fcf534e77e940b741a075c0a3c7f5bcdab60e`; calibration `2ee1690a817a59f8b6ffd7858b65ce9d36bb01a6506577261b267c79e8912bfc`; region growth `c3f35babf8128e41c090ad0415314b3653e2377010f26249ae651db9881c2734`; lattice planes `9e79617060e184f470c0af659d23886486b87bc149a329729091d576e2519ad7`; plane graph `3e330b5358542d618a42b7c35b105536b858f38791ab48350f45837afae2688a`; GUI mask `1d8a6d9ed9537a2e5dd98c794aa9c45b07c748090e1a17f9db3d230e5e74cda9`.
