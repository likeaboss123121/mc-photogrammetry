# DS-STAGEWISE-K1: paired Route A image-camera / evaluator-camera arms

- **Task:** `DS-STAGEWISE-K1` (lead authorization 2026-10-04, `AGENT_COMMS.md`), defined by
  `docs/reports/DS-STAGEWISE-LOSS-TEST-PLAN-2026-10-04.md` §4.2.
- **Owner:** DeepSeek (Sol-equivalent, direct-shell policy in `docs/deepseek-worker.md`).
- **Kind:** evaluation-only **oracle-camera sensitivity / upper-bound** measurement. **Not** causal
  attribution, **not** production or promotion evidence, **not** image-only accuracy.
- **Status:** one paired run completed; **stopped for lead review**. No retry was performed.
- **Write scope:** one sibling evaluation harness
  (`scripts/evaluation/run_image_only_route_a_camera_arm.py`), one focused synthetic test module
  (`tests/test_image_only_route_a_camera_arm.py`), this report, one result entry in
  `AGENT_COMMS.md`, and the run's scratch artifacts under `.tmp-k1-route-a-camera-arm/`. No
  production module, production signature, CLI, gate, template, mask, capture, dataset, schema or
  frozen artifact was edited.

## 1. Question and predeclared interpretation

Single factor: with the facade generator, decoded screenshot, GUI mask, palette, rows/columns/depth
and permitted alignment family frozen, how much does substituting the **evaluator camera object**
raise Route A visible-geometry (G) recall/precision/F1 on the frozen frame?

Predeclared interpretation (recorded in the harness before the run, `PREDECLARED_INTERPRETATION`):

- any A0→A1 delta is **oracle-camera headroom on this frame only**; the arm swaps the whole camera
  bundle (position, scale, intrinsics, orientation) at once, so no camera-estimation cause is
  identified;
- a near-zero delta is **consistent with** the facade generation path limiting this frame, but does
  not prove it dominates: the bundle swap could be absorbed by the alignment search, and only one
  frame and one alignment family are measured;
- if A1 reached the frozen visible ceiling under the permitted family, report the ceiling and stop;
  never call camera recovery the sole limiter;
- no K-versus-G causal decomposition is claimed; the single-factor statement is limited to "the
  camera object was the only object changed".

The project's existing **90% visible geometry-F1 gate is cited descriptively only**. K1 creates,
changes and passes no threshold.

## 2. Frozen inputs (evaluator reads only after the A0 seal)

| Input | SHA-256 / identity |
|---|---|
| `templates 2/2026-09-02_13.22.14.png` | `EB22541171E030FF94D6839A6626E7160C69AE3A780F07AEC0298322BE665D46` (matches pinned frame), 2560×1369 |
| `templates 2/Template 2.litematic` | `519034A6A439AB98145AA820792D5AFC588B5D697ECAD961C14C36CBEF2F3416` (matches pinned reference) |
| `docs/evaluation_masks/templates-2/2026-09-02_13.22.14.json` | file `2B0E18CFC26D225C6BB5CB1043792CDCC252983215B5FB7FD09480E88622850E`; scored domain `c1536420ce3f3b7fe9c4cf92a2221313b712542752c20dcec51184dd0ca08245` (matches pin) |
| `templates 2/cameras.json` | `4F743E4FC478430F608E0E879F73CD4A3DC9CE50C726841D2DA897BE78BE6126` |
| `templates 2/scene.json` | `9620608A2544E9460240C2EFD0FF83DA354D3FFCBEE8C01E61E2D9A05B627881` |
| `docs/reports/evaluator-fixture-pins.json` | `C564743A7AB63009E2105BAAE3D591338D6D3B8A694B0633DA3B0EBA6F019FD1` |

Frozen truth: reference cells `852`, canonical visible denominator **`V = 398`** (primary minimum
1 usable pixel), mask origin `tracked_json_specification`, `reference_pin_matches = true`,
`mask_pins_match = true`, no truth caveats. The observed denominator equals the fixed `V = 398`
required by the lead entry; the harness refuses to score on any other denominator.

## 3. Procedure and enforced evaluator order

Harness: `scripts/evaluation/run_image_only_route_a_camera_arm.py`, SHA-256
`71E9061EE972AF1969DD56F0FBE240036ACA58991E73FBBC617B6B781CB9FF44`.

1. **A0 (image-only).** Calls the ordinary Route A replay runner
   (`scripts/evaluation/run_image_only_route_a.py`,
   `eb3edaa19e7e166ac7eac1f1a626082c6e3f17b4fa369045435f602132911d4b`) on the frozen frame. The
   production path runs `solve_image_cameras` → `camera_model_from_estimate(position=LOCAL_FRAME_ORIGIN)`
   → `reconstruct_facade` → `save_project` → `LitematicExporter` → `load_litematic`. No evaluator
   camera, reference, mask or truth is reachable.
2. **A0 seal.** The camera record, facade/decoded volume digests, image/GUI-mask/palette digests,
   settings, code hashes, export parity and frontier digest are serialized and hashed to
   `a0-seal.json` **before any evaluator file is opened** (`a0_seal_sha256 = 929846F289FDD60C00D27B477EE23745BEE3800BF756D35517DF2E4B2E157809`,
   `evaluator_reads_before_seal = 0`).
3. **Evaluator camera → A1.** Only after the seal, the evaluator camera for
   `2026-09-02_13.22.14.png` is loaded from `templates 2/cameras.json` and A1 is rebuilt through the
   **same** `reconstruct_facade` function with the identical already-decoded screenshot object, GUI
   mask object, palette object, `columns=32`, `rows=32`, `depth_layers=8`, `horizon=None`,
   `scene_transform=None`. Only the `CameraModel` object differs. A1 is labelled
   `reference-assisted`, `out_of_promotion = true`, and is reachable only from this evaluation
   runner.
4. **Freeze both arms, then truth.** Both arm digests are written to `arms-frozen.json`
   (`camera_is_the_only_changed_object = true`) before the visible-reference litematic, frozen mask
   specification, scene transform and evaluator raycast are opened. The harness's `ArmOrderingGuard`
   refuses camera access before the seal and truth access before both arms are frozen.
5. **Score both arms** through the accepted evaluator: `evaluation.best_visible_alignment` (integer
   translation + lattice Y-quarter-turn; no scale fit, no reflection, no per-cell transform) and
   `evaluation_image_only_attribution.score_aligned_cells` (G, universe `frozen_reference_cells`,
   checkpoint `facade_grid`, primary `minimum_pixels = 1`).

Focused evidence that the order/single-factor contracts hold synthetically:
`tests/test_image_only_route_a_camera_arm.py`,
SHA-256 `30B012D38DCB36BD12B7E1DB7870982165BB372315EEB4AC37FFAFE2E320F750` — 13 passed (ordering
with injected loaders, seal-before-evaluator access, malformed/absent camera typed refusal,
reference-assisted label refusal, identical non-camera objects via a reconstruct spy, exact
hash/seal stability, fixed-`V` refusal, production isolation, unchanged `reconstruct_facade`
signature).

## 4. Command and result

```
.\.venv\Scripts\python.exe scripts\evaluation\run_image_only_route_a_camera_arm.py ^
    --image "templates 2\2026-09-02_13.22.14.png" ^
    --fixture "templates 2" ^
    --reference "templates 2\Template 2.litematic" ^
    --scene "templates 2\scene.json" --cameras "templates 2\cameras.json" ^
    --masks docs\evaluation_masks\templates-2 ^
    --pins docs\reports\evaluator-fixture-pins.json --fixture-role template-2 ^
    --workdir .tmp-k1-route-a-camera-arm ^
    --report .tmp-k1-route-a-camera-arm\k1-result.json
```

- **exit code 0**, stderr empty, elapsed **12.5 s**, one paired invocation, no retry.
- `py_compile` of the harness and the focused test module: exit 0.
- Focused tests: `python -m pytest -o addopts= -q tests\test_image_only_route_a_camera_arm.py`
  → **13 passed**.
- Result record: `result_sha256 = 4A85FD57CD6143C9827EE799B5835787865EE622BA251675CEFF34184022AF08`;
  file `k1-result.json` SHA-256
  `41B81AEDFF31DB6B17BCF5E0CB6A5CDBFA5CDC7B3557907A4133A45BBB5BE40B`.
- A0 reproduced the retained A2 Route A numbers exactly (visible correct 16, precision denominator
  687, alignment quarter-turn 1, translation `(18,13,22)`), so the fresh A0 is directly comparable
  to the historical `16/398` while now binding `reconstruction/facade.py` by hash.

## 5. Metrics — same frozen `V = 398`

| Arm | Camera object | Aligned cells | Correct (V∩C) | Missing (V−C) | False (C−R) | Exact state | Recall | Precision | F1 |
|---|---|---|---|---|---|---|---|---|---|
| **A0** image-only | image-derived (`solve_image_cameras`) | 690 | **16** | 382 | 671 | 0 | **4.0201%** (16/398) | **2.3290%** (16/687) | **2.9493%** |
| **A1** reference-assisted | evaluator camera | 690 | **17** | 381 | 671 | 0 | **4.2714%** (17/398) | **2.4709%** (17/688) | **3.1308%** |
| **A1 − A0** | — | 0 | **+1** | **−1** | **0** | **0** | **+0.2513 pp** | **+0.1420 pp** | **+0.1814 pp** |

Hidden-inferred cells (reference minus visible, matched by the aligned candidate): A0 = 3, A1 = 2.

Alignment chosen per arm inside the permitted family:

- A0: `rotation_quarter_turns = 1`, `translation = (18, 13, 22)`, source
  `evaluation.best_visible_alignment`.
- A1: `rotation_quarter_turns = 0`, `translation = (994, −97, −8018)`, same source. The large
  translation is the world→local frame offset (scene world origin `(−1008, 107, 8014)`); the
  evaluator camera is a world-frame pose and A1 ran with `scene_transform=None`, exactly as A0 did.

Descriptive reference only: the project's 90% visible geometry-F1 gate; neither arm approaches it,
and no threshold is passed or changed.

## 6. Exact cell sets (counts + sealed digests)

Sealed sha256 of the sorted coordinate sets (full coordinate lists are carried in
`k1-result.json`):

| Set | A0 count | A0 sha256 | A1 count | A1 sha256 |
|---|---|---|---|---|
| correct (V∩C) | 16 | `72e91743234142d4b0ba9295bc2672734a637f0c23dd13ac40d9c122363560c8` | 17 | `df4ece021193e2172a465da35b87a79cf02cbcf646ddae103b1e31ef91523bda` |
| missing (V−C) | 382 | `8434d141abad705efe81a68c04469b7ed73d8a68d9d3b9a8837bad11c541a9d8` | 381 | `ac6b1c96c3dd4f03426f4fbc65772b544f728fb4c2e5d4491e592cc19e501377` |
| false (C−R) | 671 | `fa0e4c8b0efd8a455acd8d2a0ea7a2cf95984551b65ae0b8e6a4b167edb41da0` | 671 | `27db77adc114192df84fbbd4e221eb7f29aa3644b830ff988aefe509313439db` |

A0 correct cells (16): `(0,1,16) (0,2,16) (0,4,16) (0,6,16) (0,7,10) (0,7,16) (0,8,10) (3,7,8)
(6,3,16) (6,7,16) (8,4,9) (8,6,9) (8,7,3) (8,7,9) (8,8,3) (8,9,4)`.

A1 correct cells (17): `(0,1,6) (0,2,11) (0,4,11) (0,5,6) (0,5,11) (1,5,16) (1,6,16) (3,1,3)
(4,0,8) (4,5,8) (6,0,6) (6,7,16) (8,2,4) (8,4,4) (8,5,9) (8,6,9) (8,7,9)`.

Set relation: the two arms share only **3** correct cells (`(6,7,16) (8,6,9) (8,7,9)`); A0 has 13
correct cells that A1 does not, and A1 has 14 that A0 does not. The missing and false sets also
differ even though the false counts tie at 671. The net count delta is therefore **+1 cell under
substantial composition churn**, not a like-for-like improvement of the same footprint.

## 7. Camera records and single-factor evidence

| | A0 (image-only) | A1 (reference-assisted) |
|---|---|---|
| provenance | `image-only`, `out_of_promotion = false` | `reference-assisted`, `out_of_promotion = true` |
| position | `(0.0, 0.0, 0.0)` (LOCAL_FRAME_ORIGIN) | `(−1006.1, 108.0, 8017.8)` (world) |
| yaw / pitch | `47.35` / `−4.5` | `−42.7` / `−4.6` |
| fov | `110.0` | `110.0` |
| intrinsics | `2560×1369`, focal `(479.29206, 479.29206)`, principal `(1280.0, 684.5)` | none stored (derived from fov/height) |
| roll / eye height | `0.0` / `1.62` | `0.0` / `1.62` |
| camera record sha256 | `C368A0FA7ACB771AC9A7FDFA49C91FBBF79CB18EED62CA72AA5055C8178DF3AA` | `1EFC2062CEB0DA5B5D06A418999FD905E11B4C56D9774AC4991180C31433772E` |

Shared, byte-identical non-camera factor record (`factor_record_sha256 =
C16BEF6D398B957FCC529D063A25CE997939D9C457B4FE77AB173A83ACC09D37` for both arms):

- decoded screenshot array sha256 `119fe8052f4aec39a7bb276874b7e33fa73cef46a421a45e83e69a646f950e33`
  (the on-disk frame file is the pinned `EB225411…`), shape `[1369, 2560, 3]`, uint8;
- production GUI mask domain sha256 `b541756fb95ee610c893013a4f6d2a3e0c63837993cbbd94fb6b0a6e53a14a88`;
- palette = the default 8-state `DEFAULT_PALETTE`, states sha256
  `0650cb5055520fcae5e37364db7400317373376e70a62d4d3bfaae1761cb962f`;
- settings `columns=32, rows=32, depth_layers=8, horizon=null, scene_transform=null,
  palette_jar=null`.

A0 production records: camera solved `true`, `camera_error = ""`, facade blocks `690`, decoded
blocks `690`, facade→export round trip exact `true`, frontier terminal `facade_grid`/`available`.
A1 production record: evaluator camera `690` facade blocks.

Observation, not a causal claim: the image-derived yaw (`47.35°`) differs from the evaluator yaw
(`−42.7°`) by ≈90°, i.e. the lattice yaw quadrant that one view leaves unresolved; A0's chosen
alignment absorbs a quarter-turn (`turns = 1`) while A1's does not (`turns = 0`). This is exactly
the predeclared caveat that the alignment search can absorb part of the bundle swap; it is
descriptive and identifies no cause.

## 8. Implementation and artifact hashes

Code hashes sealed in A0 before any evaluator read:

| Path | SHA-256 |
|---|---|
| `src/mc_photogrammetry/reconstruction/facade.py` | `c324946e27c604f62f3397af7d8f331febfdd43fa8fa8a757a0277e64344a6cf` |
| `src/mc_photogrammetry/pipeline/image_only.py` | `5a6e99b66e7d580d9082f00347514d286937e9a6917dd4a36b8bcc9f5591d391` |
| `src/mc_photogrammetry/reconstruction/camera.py` | `0515c5bd82f01d3dc174ced6cb54405adef0a755b94c577dae513fc53c4fe680` |
| `src/mc_photogrammetry/evaluation.py` | `7dee55c574fa24c6d1f60f35de5dc10573b1698f3ce4f34bae879366cddf4327` |
| `src/mc_photogrammetry/evaluation_image_only_attribution.py` | `4fed6aa8e22dd881e69063a33b56df2480b1700bfe62cc7d86d69ee54a8fe2a3` |
| `scripts/evaluation/run_image_only_route_a.py` | `eb3edaa19e7e166ac7eac1f1a626082c6e3f17b4fa369045435f602132911d4b` |
| `scripts/evaluation/report_image_only_attribution.py` | `ee976b6b9e79a8535a412f9430fdf5e0ca8d1c16cffea3df342e130eecf22805` |
| `scripts/evaluation/run_image_only_route_a_camera_arm.py` | `71e9061ee972af1969dd56f0fbe240036aca58991e73fbbc617b6b781cb9ff44` |

Run artifacts (scratch, `.tmp-k1-route-a-camera-arm/`):

| Artifact | SHA-256 |
|---|---|
| `a0-seal.json` | `89F918157126A6C71EF737601BB8F119BB621C30F5779A23F8379D81183C8FDF` (seal digest `929846F289FDD60C00D27B477EE23745BEE3800BF756D35517DF2E4B2E157809`) |
| `arms-frozen.json` | `1C1268AC4C7CBA4BEBFBFC2CC86FF0FDDAF4844D27C4E9A5BDD7E82CDD3498C4` |
| `k1-result.json` | `41B81AEDFF31DB6B17BCF5E0CB6A5CDBFA5CDC7B3557907A4133A45BBB5BE40B` (result digest `4A85FD57CD6143C9827EE799B5835787865EE622BA251675CEFF34184022AF08`) |
| `2026-09-02_13.22.14.mcscan` / `.litematic` | A0 production export from the replay (local frame) |

Repository revision at run: `c6dba2cf4610876db7d24361eb13fc9ea3ceca0c`. The tree already carried
unrelated pre-existing modifications; K1 added only the harness, the focused test module, this
report, the `AGENT_COMMS.md` entry and the scratch run directory.

## 9. Interpretation (headroom only)

On this one frozen frame and this one alignment family, substituting the evaluator camera changed
the G count by **+1 cell** (`16 → 17`) and F1 by **+0.18 pp**. Per the predeclared rule this
near-zero delta is **consistent with** the facade generation/placement path (not camera
estimation) limiting this frame, but it does **not** prove that path dominates: the whole camera
bundle changed at once, only 3 correct cells are shared between the arms (13 lost, 14 gained), and
the quarter-turn alignment differs between arms, so the alignment search can absorb camera
differences. The correct statement is limited to "the camera object was the only object changed",
and there is no causal K-versus-G decomposition. No claim is made that camera recovery is the sole
or dominant limiter.

## 10. Limitations and non-claims

- One frozen frame, one paired run per arm, no retry, one permitted alignment family.
- The arm swaps the entire camera bundle (position/scale/intrinsics/orientation); orientation,
  focal, position and scale effects are not separately identified.
- A1 is `reference-assisted`, out of production and promotion, and is never image-only evidence; it
  is excluded from the supplied-camera funnel.
- Route A's metric scale is unresolved; each facade score carries that caveat.
- The 90% visible geometry-F1 gate is descriptive only; K1 passes no gate.
- No model training, dependency change, new screenshot/capture, fixture modification, Minecraft/
  launcher/UI involvement, or production/CLI/schema/gate change occurred.
- This is not production evidence and creates no promotion state.

## 11. Stop

Stopped for lead review with the exact command, exit code, hashes, metrics and digests above.
