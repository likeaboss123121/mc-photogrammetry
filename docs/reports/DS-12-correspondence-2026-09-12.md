# DS-12 — SIFT correspondence extraction and a frozen real-pair diagnostic

DeepSeek CLI DS-12, 2026-09-12. New paths only:
`src/mc_photogrammetry/pipeline/correspondence.py`,
`tests/test_correspondence.py`, `scripts/diagnose_connected_pairs.py`,
generated `exports/diagnostics/ds12/{estimates,evaluation}.{json,csv}`, and this
report. DS-11 source/tests, `residual.py`, `raycast.py`, shape fitting,
benchmark, annotations, handoff/comms and every Luna path are untouched. No
dependency was added or installed. Nothing was staged or committed. Revision:
working tree at `c6dba2c` plus the accepted DS-3/4F/5/7/9/10/11 edits
(`git status` dirty, as before).

## 1. What was built

One module and one public entry point. `extract_pair_matches(left_id, left_rgb,
right_id, right_rgb, *, usable_masks=None, settings=None)` returns a
`PairMatches`: SIFT features per frame, a filtered `N x 4` correspondence array
`(left_x, left_y, right_x, right_y)`, the funnel counts, and a typed `status`
plus `reason`. `feedable` is true only for `status == "ok"`; only those pairs are
handed to DS-11's `solve_pair`.

`select_mutual_matches(left_points, right_points, left_descriptors,
right_descriptors, ratio_test=...)` is the pure, image-free filter: both
directions take the two nearest L2 neighbours, both apply the ratio test, and a
pair survives only when each side is the other's ratio-surviving nearest
neighbour. It is the same code path real SIFT descriptors use, so the synthetic
controls exercise production logic.

Extraction reads pixels, an optional usable-pixel mapping, and nothing else: no
camera, scene, lattice, manifest, coordinate, direction or asserted geometry is
reachable from this module, so a correspondence set can be frozen before any
evaluation code looks at a fixture.

## 2. Extraction/evaluation separation (recorded)

The diagnostic is two phases in one script:

* `--extract-only` runs `run_extraction`, which necessarily runs before any
  manifest is opened, and writes `estimates.json` + `estimates.csv`. Those bytes
  carry `"phase": "extraction_and_estimation_blind"` and the note that no
  `cameras.json`, reference litematic, lattice or asserted geometry was opened.
  `load_camera_manifest` is imported *inside* `run_evaluation`, never at module
  scope, so the blind phase cannot see it.
* `--evaluate-only` loads the already-written `estimates.json` from disk,
  records its SHA-256 in `evaluation.json`, and only then opens the three
  fixture manifests. `evaluation.json` carries `"phase":
  "reference_evaluation"` and states that no extraction or solver parameter was
  changed after the comparison.

The run order actually used was:
`--extract-only` (wrote `estimates.json`, sha256 `290204c6…31afdf`) then
`--evaluate-only` (recorded that digest, wrote `evaluation.json`, sha256
`53981138…17b03`). These are the artifacts of record; the settings below were
declared in the module before the first extraction run and were not changed
afterward. No threshold was tuned after reference inspection.

## 3. Extraction design

**Usable-pixel authority** follows DS-4 exactly (`raycast._pixel_authority`
semantics, re-expressed locally): `None` derives `~create_gui_mask(h, w)` per
frame; a supplied mapping is authoritative, a missing frame is all-false, a
wrong-shaped mask raises `ValueError` before any feature is detected. Excluded
pixels are zeroed *before* `SIFT.detectAndCompute` and are also excluded by the
detector mask, so RGB outside the domain cannot reach a keypoint or descriptor —
this is what makes the excluded-RGB invariance control meaningful.

**Statuses** (`MATCH_STATUSES`), first failure wins: `unusable_frame` (a frame
has no usable pixels), `insufficient_keypoints` (raw keypoints below
`min_keypoints_per_frame`), `insufficient_matches` (mutual matches below
`min_matches`), `repetition_risk` (spatial-bin coverage below
`min_spatial_coverage`, or mutual fraction below `min_mutual_fraction`), else
`ok`. Every status that is not `ok` is reported as a refusal; nothing is padded.

**Declared settings** (`CorrespondenceSettings`), echoed into the artifact:
`max_features=4000`, `ratio_test=0.75`, `min_matches=12`,
`min_keypoints_per_frame=24`, `spatial_grid=(4, 4)`,
`min_spatial_coverage=0.2`, `min_mutual_fraction=0.02`. The artifact also
records DS-11's `PairGates` defaults unchanged (`min_inliers=8`,
`inlier_residual_degrees=1.0`, `min_inlier_fraction=0.5`,
`min_parallax_degrees=3.0`, `min_cheirality_fraction=0.75`, `ransac_seed=0`, …).

**Reported per pair**: raw keypoints both sides, usable pixels, directed ratio
survivors, mutual survivors, mutual fraction, non-finite/duplicate discards,
spatial bins covered on each frame plus coverage, descriptor-distance
min/median/mean/max, and the raw match list. Extraction is deterministic
(seeded nothing; SIFT and the pure filter are both deterministic).

## 4. Controls (`tests/test_correspondence.py`, 8 tests)

| Control | What it pins |
| --- | --- |
| `test_mutual_selection_keeps_reciprocal_ratio_survivors` | 5×4 synthetic descriptor table: ratio survivors 4/3, mutual 3, exact surviving rows; a near-duplicate left row is not reciprocal and an ambiguous row fails the ratio in both directions |
| `test_mutual_selection_is_deterministic_and_one_to_one` | two runs bit-identical; no duplicate left/right index; a two-lefts-one-right tie yields `mutual == 0` |
| `test_non_finite_and_duplicate_pixel_pairs_are_removed` | NaN pixel row dropped (`discarded_non_finite == 1`); identical pixel pair deduped (`discarded_duplicate == 1`) |
| `test_none_derives_the_production_gui_mask_per_frame` | `usable_mask_for(..., None)` equals `~create_gui_mask`; an explicit all-true mask yields strictly more usable pixels |
| `test_missing_frame_is_unusable_and_wrong_shape_rejects` | missing frame ⇒ `unusable_frame`, no matches; wrong-shaped mask ⇒ `ValueError` |
| `test_excluded_rgb_cannot_change_extracted_matches` | 220×180 noise pair, rectangular usable domain: mutating both frames to 255/0 outside the domain leaves the match array and counters identical (non-vacuous: baseline has matches) |
| `test_unique_geometric_marks_yield_a_feedable_pair` | 380×300 image of 30 unique marks, second view affine-rotated 3° and shifted: `ok`, >12 matches, one-to-one, repeat run identical |
| `test_repeated_grid_is_flagged_instead_of_confidently_matched` | identical 40-px tile over 320×240, second view translated: `insufficient_matches` (2 ratio survivors) — never `ok` |

Measured synthetic outcomes (development probe over the same public result):
marks `ok`, raw 247/258, ratio 142/136, mutual 115, matches 83, coverage 0.81;
repeated grid raw 815/690, ratio 2/2, mutual 2, `insufficient_matches`.

## 5. Frozen real-pair diagnostic — provenance and result (phase 1)

**Frozen pair rule** (stated before any run, outcome-blind): for each fixture's
image-derived M3 calibration artifact, take the first two adjacent frame pairs in
sorted-filename order — Template 2 (3 frames), Template 6 (4 frames), and the
frozen `t3-sparse-8` subset. Cameras come from the image-derived M3 artifacts
(`M3-calibration-templates-2/-6.json`, `M3-calibration-templates-3-t3-sparse-8.json`);
usable masks are `None` (derived production GUI domain). Extraction is
manifest-free; `estimates.json` records the calibration-artifact digests
(`435c19bb…`, `a24751ab…`, `bbbd3a0d…`), frame digests and sizes, Python 3.10.11,
NumPy 2.2.6, OpenCV 5.0.0, and the git revision (`c6dba2c`, dirty).

All 9 frames are 2560×1369. Extraction time per pair was 0.35–0.51 s (frame
loading excluded; solve ≤0.07 s). Result:

| fixture | pair (suffix) | status | reason | raw L/R | ratio L/R | mutual | matches | cov |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T2 | 13.22.14 → 13.24.24 | `repetition_risk` | mutual fraction 0.0196 < 0.0200 | 1286/919 | 60/83 | 18 | 17 | 0.31 |
| T2 | 13.24.24 → 13.24.33 | `insufficient_matches` | 9 mutual < 12 | 919/1568 | 20/25 | 9 | 9 | 0.19 |
| T6 | 18.01.09 → 18.01.14 | `repetition_risk` | coverage 0.06 < 0.20 | 1799/2560 | 21/32 | 14 | 12 | 0.06 |
| T6 | 18.01.14 → 18.01.18 | `repetition_risk` | coverage 0.19 < 0.20 | 2560/1810 | 110/93 | 43 | 40 | 0.19 |
| T3-sparse-8 | 20.03.32 → 20.03.55 | `ok` | — | 973/320 | 66/43 | 21 | 20 | 0.25 |
| T3-sparse-8 | 20.03.55 → 20.04.18 | `insufficient_matches` | 8 mutual < 12 | 320/668 | 26/18 | 8 | 8 | 0.19 |

**5 of 6 pairs are refused by the correspondence stage** (three on spatial
coverage or mutual fraction — the repetition guard — and two on too few mutual
matches). Only the T3-sparse-8 pair `20.03.32 → 20.03.55` was feedable, with 20
mutual matches (raw 973/320, ratio 66/43, coverage 0.25, mutual fraction 0.066).

## 6. Reference evaluation (phase 2)

Manifests opened only here: `templates 2/cameras.json` (`4f743e4f…`),
`templates 6/cameras.json` (`836ad62f…`), `templates 3/cameras.json`
(`92dc88ec…`). The one feedable pair is refused by DS-11 with
`insufficient_inliers`: 20 informative correspondences but only 5/7/5/8 inliers
across quadrants 0/1/2/3 (needing 8), so no class passes. Per-quadrant medians
are 19.7/11.0/5.1/6.5 deg, i.e. the robust fit never converges — the matches are
not a consistent two-view geometry. Refusal reasons are exactly DS-11's typed
tokens.

**Solved pairs: 0/6. Angular errors: none measurable.** The solved-yaw-class and
baseline-direction comparison therefore has no real-data row. This is an honest
refusal result, not a fabricated one: DS-12 does not report an error it did not
measure, and no setting was relaxed to manufacture a solve. It is consistent
with the recorded M3 relative-pose evidence (`scripts/relative_pose.py`, CPU
PyCOLMAP: hypotheses produced but 0/41 within 5° after the rotation-scoring fix,
handoff §2): on these repetitive frames, correspondence-stage refusals are the
expected outcome, not a regression introduced here.

What phase 2 did establish, on all 6 pairs, is the reference machinery itself.
The manifest reference class (argmin over quadrants of the public
`camera_rotation_matrix`/`angle_between` construction) equals the independent
`floor(yaw_r/90) − floor(yaw_l/90) mod 4` formula on every pair:
T2 3/3, T2 2/2, T6 2/2, T6 1/1, T3 0/0, T3 3/3. That confirms the DS-11/DS-10
C2 gauge algebra the evaluation relies on: the relative class is gauge-free
while the left view's own quadrant is a gauge.

The evaluation path's baseline branch was separately validated on a synthetic
solved pair (DS-11's own exact scene, left/right yaw 85°/95°, positions
`(0,3,0)`/`(2.5,3,1)`): `solve_pair` returns class 1, `_reference_class` returns
1, rotation error 0.000°, and the gauge search returns index 0 (=
`floor(85/90) mod 4`) with 0.000° baseline error. That control is throwaway
evidence here, not a new test artifact.

## 7. Limits and non-claims

1. **No real pair solved.** The diagnostic measures correspondence quality and
   DS-11's refusal behaviour on real frames; it does not measure relative-pose
   accuracy on real data. The M3 gate is untouched.
2. **The mutual-fraction gate is a declared default, and one pair is borderline.**
   T2 `13.22.14 → 13.24.24` has 18 mutual matches over 31% of bins but a mutual
   fraction of 1.96%, one basis point under the pre-registered 2.00% guard. It
   was refused, and it stays refused: lowering the guard after seeing the
   extraction funnel would be outcome-driven tuning. A reviewer who wants that
   pair attempted should change the declared default in a separate, stated run.
3. **Boundary features.** Zeroing the excluded domain is what guarantees
   excluded-RGB invariance, but it leaves a deterministic mask edge inside each
   frame; SIFT may place features there. Those descriptors are invariant to
   excluded RGB yet are not scene correspondence and are part of why DS-11 sees
   few inliers. A small detection-mask erosion would trade coverage for purity
   and is deliberately not done here.
4. **Adjacent pairs only.** The frozen rule uses adjacent sorted frames; nothing
   about the baseline or overlap was optimised. Larger-baseline pairs were not
   tried, and no pair was selected by outcome.
5. **Raw matches are not validated correspondences.** The artifact stores the
   raw `N x 4` list so a later solver can re-read it; it is evidence of what
   extraction produced, not a correspondence-truth claim.
6. **Runtime is not benchmark provenance.** Extraction seconds are single-run
   wall-clock with frame loading outside the timer, on a dirty tree; they are
   not an M6 performance measurement.
7. This package adds a correspondence primitive and a refusal diagnostic. It
   establishes no components, no positions, no scale, no `scale_known`, no
   schema/CLI/desktop wiring and no production promotion.

## 8. Evidence

```
.venv\Scripts\python.exe -m pytest tests/test_correspondence.py \
  tests/test_connected_camera.py tests/test_camera.py -p no:cacheprovider
39 passed in 6.04s
```
(8 new correspondence controls; 31 DS-11+`test_camera.py` controls. No full
suite was run, per the assignment.)

Artifacts:
`exports/diagnostics/ds12/estimates.json` (`290204c69f9af33200c774873116cfdbe9279d2dc1c1d9afbfa42f17a731afdf`),
`estimates.csv`, `evaluation.json`
(`53981138ce4e4e69798f76490e1ffa034642312e8cb793bb0b68d6b295517b03`),
`evaluation.csv`. The next bounded package would be a repetition-aware matcher
(or a lattice-constrained one) plus a solver edge only over correspondence sets
that pass this funnel; DS-11's graph/component layer remains unimplemented.
