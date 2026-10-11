# M4 F3 plane-specific annotation audit v4

Status: **submitted for lead review; annotation-dependent scores remain provisional.** This package closes Luna's bounded S-C visual audit only. It does not promote a reconstruction result or change the production gate.

The new evaluation annotation is [M4-C2-boundary-targets-v4-2026-09-12.json](M4-C2-boundary-targets-v4-2026-09-12.json), SHA-256 `f4a0f86938561b3fc81b096fc614d1ae404c1372e65c925811c1368ce56a0da5`. It names one target plane per frame (`t2_f1_left_wall`, `t2_f2_left_wall`, `t2_f3_left_wall`). The three source screenshots were reviewed at native 2560x1369 resolution. Plane identity follows visible room corners and beam transitions; repeated wood appearance and fixture poses were not used. v1-v3 remain untouched, and the existing v3 file still hashes to `082ec648845be2354a5324821e6484291ca867563948e0396cecb496e5adf56a`.

Source screenshot hashes stored in v4 are frame 1 `eb22541171e030ff94d6839a6626e7160c69ae3a780f07aec0298322be665d46`, frame 2 `e2fd9fd6c83bf9005fe0973987da9e46d213a4351ec91393ce578f8e6d3961ae`, and frame 3 `35d9debb206c2505f65d3be29183f4a4f3d2bf3f9c05644ce75570f48381d1fc`.

Frame 1's source rectangle `(1075,700)-(1820,1369)` was inspected in full. The v3 dining-area rectangle is removed. The v4 target preserves the wall at `(1150,740)`, traces the tabletop, apron/legs, flower stem and separate end benches, and leaves a narrow tabletop-edge ribbon unknown. The existing broad claim and other reviewed foreground exclusions remain. Image-judged anchors are `(1150,740)` wall, `(1200,748)` uncertain table edge, `(1200,775)` table interior, `(1460,720)` flower/stem, and `(1750,805)` right bench. The first point was negative in v3 and positive in v4.

Frame 2's existing left-wall extent and corner uncertainty band were checked against the source and retained. `(500,300)` and `(700,700)` remain on the left wall; `(1000,300)` stays unknown at the corner; `(1200,250)` and `(1200,850)` are outside the left-wall plane.

Frame 3's positive region is narrowed to the left-wall side of the visible corner. The band `x=960..1060, y=100..780` is unknown. The source supports left-wall pixels at `(600,300)` and `(600,1200)`; `(1400,300)` lies on a distinct farther wall and is negative for this target despite its matching wood texture. This removes the v3 union of adjacent planes while retaining the lower-left wall visible in the gamma-boosted source crop. There remains one target per frame and the frame denominator stays three.

Unknown regions and composed mask areas (before HUD/domain exclusions) are recorded so hard image regions remain visible in the audit:

| Frame | v3 positive px | v4 positive px | v3 unknown px (% frame) | v4 unknown px (% frame) |
|---|---:|---:|---:|---:|
| 1 | 905,449 | 912,792 | 240,926 (6.874%) | 243,006 (6.934%) |
| 2 | 892,689 | 892,689 | 41,796 (1.193%) | 41,796 (1.193%) |
| 3 | 1,830,309 | 1,283,518 | 126,351 (3.605%) | 195,031 (5.565%) |

The additional frame 3 unknown area is the visible corner band plus the pre-existing low-contrast ceiling transition. Frame 1's unknown area includes its pre-existing roof/corner regions and the new narrow table-edge strip. Each uncertain polygon now has a written reason in v4.

The review bundle contains original/overlay pairs for all three frames in [all_frames_side_by_side.png](M4-F3-plane-mask-audit-2026-09-12/all_frames_side_by_side.png), plus the native-coordinate [frame 1 dining transition](M4-F3-plane-mask-audit-2026-09-12/frame1_dining_transition.png) and [frame 3 corner transition](M4-F3-plane-mask-audit-2026-09-12/frame3_corner_transition.png). Per-frame overlays outline the positive plane in cyan, explicit furniture/occluder polygons in orange, unknown edges in yellow, and independently image-judged anchors by label. [build_v4_artifacts.py](M4-F3-plane-mask-audit-2026-09-12/build_v4_artifacts.py) reproduces the annotation and visual bundle from v3 plus the source screenshots.

The v4 anchor test checks the frozen file hash, immutability of v3, target IDs, uncertainty reasons, and all listed labels after mask composition. Manual inspection of the original and overlay images establishes the source-image judgment; the test only checks that the JSON still encodes that judgment.

## Saved C2 diagnostic

After freezing v4, [rescore_saved_c2_v4.py](M4-F3-plane-mask-audit-2026-09-12/rescore_saved_c2_v4.py) re-scored only the candidate polygons already saved in `M4-C2-boundary-pilot-2026-09-09/results.json`. It never called `find_chains`, `propose_boundary_candidates`, or any SfM path. The local diagnostic thresholds remain precision `>=0.90` and recall `>=0.50`; they are not production gates.

| Frame / target | Saved candidates | Top-1 P / R / IoU | Best evaluator-only rank, P / R / IoU | Top-1 local gate |
|---|---:|---|---|---|
| 1 / `t2_f1_left_wall` | 1 | 0.000 / 0.000 / 0.000 | rank 0: 0.000 / 0.000 / 0.000 | fail |
| 2 / `t2_f2_left_wall` | 8 | 0.668 / 0.018 / 0.018 | rank 1: 0.327 / 0.125 / 0.099 | fail |
| 3 / `t2_f3_left_wall` | 0 | no candidate | no candidate | fail |

Result: **0/3 frames pass; the local continuation condition is not met.** The no-candidate frame remains a failure. The annotation narrows target semantics, but this diagnostic does not establish an image-only production result or change the candidate search. Full values and overlays are in [rescored_c2/results.json](M4-F3-plane-mask-audit-2026-09-12/rescored_c2/results.json) and that directory.

## Reproduction

From the workspace root, the artifact build and saved-C2 rescore were:

```powershell
.\.venv\Scripts\python.exe docs\reports\M4-F3-plane-mask-audit-2026-09-12\build_v4_artifacts.py
.\.venv\Scripts\python.exe docs\reports\M4-F3-plane-mask-audit-2026-09-12\rescore_saved_c2_v4.py
```

Focused regression evidence:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_boundary_scoring.py -q -p no:cacheprovider -o addopts=
```

No proposer, threshold sweep, camera fit, coordinate/pose metadata, production path, or dataset-factory path was changed. Annotation acceptance remains with the lead.
