# M5-S1 — neutral appearance scoring and face-observation contracts

Date: 2026-09-19
Task: `m5-s1-neutral-appearance-2026-09-19` (routed repair, attempt 02)
Role: bounded developer (DeepSeek/Sol-equivalent) under the direct-shell policy in
[`docs/deepseek-worker.md`](../../docs/deepseek-worker.md)
Status: attempt-02 repair complete, required checks pass, **stopped for manager/Astra review**

## Context

Attempt 01 extracted the neutral M5 scorer and the immutable `FaceObservation`
contract and passed focused/adjacent/F2/full checks, but Astra's read-only review
rejected it (`AGENT_COMMS.md` 2026-09-19 "Astra -> Luna - M5-S1 review rejected")
after reproducing five defects. Jev route
`route-m5-s1-neutral-appearance-2026-09-19-aa5f09b114e3` authorized one fresh,
bounded repair attempt with regression tests. This report covers that repair.
M4 placement/wiring, V2-B, CLI/UI, schema, evaluator, scheduler, launcher,
security settings and git state remain untouched and blocked as before.

## Repaired defects

Each defect was reproduced against the attempt-01 source before the fix, then
re-verified after it.

1. **All-zero effective masks abstained on the summary instead of the pixels.**
   An observation with `usable_coverage=1.0` and either `valid` all-`False`, or a
   `coverage` array that is all zero, or `valid`/`coverage` masks that never
   overlap, was scored rather than abstaining. `FaceObservation` now exposes
   `has_effective_coverage` (any effective weight `> 0`) and
   `abstention_reason` checks it after the valid-mask check. All three cases now
   abstain (`zero_coverage`, or `no_valid_pixels` when the valid mask itself is
   empty, preserving the accepted reason precedence) with empty costs.

2. **Coverage-zero and invalid pixels could move the descriptor.**
   `patch_descriptor`'s fixed-shape gradient term is unweighted, so a
   `coverage==0` pixel (still `valid`) changed the descriptor even though the
   weighted terms ignored it. `FaceObservation.descriptor` now neutralises every
   pixel whose effective coverage is `<= 0` to the mean of the retained owned
   pixels before calling `patch_descriptor`, and keeps the effective-coverage
   weights. Changing an excluded pixel's value can no longer change the
   descriptor. The shared `patch_descriptor` is unchanged, so the legacy
   `raycast` descriptor path (`raycast._patch_descriptor is
   blocks.appearance.patch_descriptor`) is byte-for-byte compatible.

3. **Retained arrays could be reopened with `setflags(write=True)`.**
   A private copy that owns its data can always be made writeable again.
   `_readonly` now backs the retained copy with immutable `bytes`
   (`np.frombuffer(copy.tobytes()).reshape(...)`). `patch`, `valid` and
   `coverage` (and their bases) raise `ValueError` on `setflags(write=True)` and
   on direct item assignment.

4. **`face_observation_from_sample` silently coerced lossy coordinates.**
   `int(1.9) == 1` attributed evidence to the wrong cell. The adapter now reads
   the coordinate through `_sample_cell`, accepting non-bool integers, integral
   floats/`np.integer`s, and rejecting fractional, `NaN`, infinite, boolean,
   string, wrong-arity and non-iterable coordinates with `ValueError`. It also
   validates `slot`, `patch` and `visible` presence/type instead of coercing
   them (`float(True)`, `str(None)`, missing attributes) and passes `visible`
   through raw so the record's own range checks apply.

5. **`FaceScore.ranked()` ties followed palette order.**
   Python's stable sort left equal-cost candidates in palette order, so two
   palettes with the same candidates in a different order nominated different
   winners. `ranked()` now breaks ties with `_candidate_sort_key`, the
   normalised `BlockState` identity (`BlockState._identity`) that value equality
   already uses, making the ranking deterministic and palette-order independent.

## Regression tests added

`tests/test_face_observation.py` (+17 collected cases, now 49):

- `test_retained_arrays_cannot_be_reopened_for_writing` (flag flip, base flip,
  item assignment all rejected).
- `test_all_zero_coverage_abstains_despite_positive_usable_coverage`,
  `test_all_zero_valid_abstains_despite_positive_usable_coverage`,
  `test_masks_that_never_overlap_leave_no_effective_coverage`.
- `test_coverage_zero_pixels_do_not_change_the_descriptor`,
  `test_invalid_pixels_do_not_change_the_descriptor`.
- `test_face_observation_from_sample_accepts_integral_float_coordinates`.
- `test_face_observation_from_sample_rejects_lossy_coordinates` (parametrised
  over fractional, epsilon-fractional, `NaN`, `inf`, boolean, short, long,
  string and non-iterable coordinates).
- `test_face_observation_from_sample_rejects_invalid_or_missing_fields`.

`tests/test_appearance_scoring.py` (+3 collected tests, now 16):

- `test_zero_effective_coverage_abstains_even_with_positive_usable_coverage`
  (all-zero coverage and disjoint valid/coverage masks at the scoring seam).
- `test_coverage_zero_pixels_cannot_affect_a_face_score`.
- `test_ranked_ties_are_deterministic_regardless_of_palette_order`.

## Preserved behaviour and boundaries

- Compatibility surface is unchanged: every legacy name in `facade`/`raycast`
  still resolves to the neutral objects (`raycast.PaletteIndex is
  appearance_scoring.PaletteIndex`, `_patch_descriptor`,
  `_observation_costs`, …).
- `PaletteColor` fields/defaults/equality, `DEFAULT_PALETTE`, relative/capped
  state-keyed costs, missing-metadata fallback, partial-shape `slot_coverage`,
  texture neutrality and the accepted F2 atomic-scheduling behaviour are
  untouched. No shared primitive's numeric output changed.
- No advanced-only metadata boundary moved; no M4 placement, V2-B, CLI/UI,
  schema, evaluator, GPU, launcher, security or git change was made. `HEAD`
  remains `c6dba2cf4610876db7d24361eb13fc9ea3ceca0c` on `main`, nothing staged,
  no commit/stage/reset/clean.

## Files changed (SHA-256, lowercase; line counts)

```
2380f1676e713d340432dfa28f2146b924206157bbb41f79f8c39ed897c621b8   332  src/mc_photogrammetry/blocks/appearance.py            (unchanged from attempt 01)
84f767fb781bebb08c39cb45fa0ec41990655b3562dcfa090ff8f4defda9ed02   479  src/mc_photogrammetry/reconstruction/appearance_scoring.py  (repair 5)
03710aaec58d1e2fffb769174db34f826c7d8dd830bedb8d9074bd231bda8c1e   351  src/mc_photogrammetry/reconstruction/face_observation.py   (repairs 1-4)
c324946e27c604f62f3397af7d8f331febfdd43fa8fa8a757a0277e64344a6cf   228  src/mc_photogrammetry/reconstruction/facade.py        (unchanged from attempt 01)
7daa9e43793a91e7b49c37bc2781f33e08ebce2982127a0abe552a675f87182e  2329  src/mc_photogrammetry/reconstruction/raycast.py       (unchanged from attempt 01)
cb08402f5b0230a615fab70b57982b539b3f361c9269f4197554a479b8ad0b97  1276  src/mc_photogrammetry/reconstruction/shape_fit.py     (unchanged from attempt 01)
f735ae866d949bdcf6f24c90b2a604b4c2b8a0b2d5661561aa9f4ff47c37cc45   923  src/mc_photogrammetry/blocks/textures.py              (unchanged from attempt 01)
bf71dcb0db9a5daeb0fd9355f90dfd71da7076ea65c2e86ac0f05735104d6fdb   349  tests/test_appearance_scoring.py                      (repairs 1, 2, 5)
bde8f159020c320ca61fb01268d4ef9582c76ea78303f10304db39a630f1c94b   329  tests/test_face_observation.py                        (repairs 1-4)
```

Attempt-02 edited only the four in-scope repair targets above and this report.
`appearance.py`, `facade.py`, `raycast.py`, `shape_fit.py` and `textures.py`
hash-match attempt 01 exactly, so the compatibility/F2 surface did not move.

`diff-scope`: attempt 02 changed only the four in-scope repair targets and this
report. Every other modified/untracked path in `git status` (including the
attempt-01 `facade.py`/`raycast.py`/`shape_fit.py`/`textures.py` surface and the
pre-existing F2 `tests/test_shape_fit.py`) is earlier-task work carried in the
baseline and was not touched. `git diff --cached` is empty; `HEAD`/branch are
unchanged from attempt 01. No forbidden path (V2-B, M4 placement, CLI/UI,
schema, evaluator, GPU, launcher, security, git) appears.

## Checks run

```
$ .\.venv\Scripts\python.exe -m py_compile `
    src\mc_photogrammetry\blocks\appearance.py `
    src\mc_photogrammetry\reconstruction\appearance_scoring.py `
    src\mc_photogrammetry\reconstruction\face_observation.py `
    src\mc_photogrammetry\reconstruction\facade.py `
    src\mc_photogrammetry\reconstruction\raycast.py `
    src\mc_photogrammetry\reconstruction\shape_fit.py `
    src\mc_photogrammetry\blocks\textures.py `
    tests\test_appearance_scoring.py tests\test_face_observation.py   # exit 0

$ .\.venv\Scripts\python.exe -m pytest tests\test_face_observation.py tests\test_appearance_scoring.py `
    -q -p no:cacheprovider -o addopts=              # 65 passed (counterexample/regression tests)

$ .\.venv\Scripts\python.exe -m pytest tests\test_shape_fit.py tests\test_raycast.py `
    tests\test_face_sampling.py tests\test_complete_palette.py tests\test_textures.py `
    tests\test_texture_cache.py tests\test_facade.py tests\test_reconstruction.py `
    -q -p no:cacheprovider -o addopts=              # 107 passed (required adjacent)

$ .\.venv\Scripts\python.exe -m pytest tests\test_adjacency.py tests\test_attribution.py `
    tests\test_boundary_candidates.py tests\test_capture_quality.py tests\test_cli.py `
    tests\test_exposure_reconciliation.py tests\test_frozen_plane_evaluation.py `
    tests\test_light_cast.py tests\test_model_geometry.py tests\test_pixel_authority.py `
    tests\test_raycast_contrast.py tests\test_residual.py tests\test_shape_persistence.py `
    tests\test_stereo.py tests\test_surface_pruning.py tests\test_visibility.py `
    -q -p no:cacheprovider -o addopts=              # 219 passed (broader M5 adjacent)

$ .\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider -o addopts=
                                                    # 1764 passed, 1 skipped in 461.47s
```

F2 independent replay (the five required cases, six named tests) — all passed:

```
tests/test_shape_fit.py::test_zero_cap_reports_every_cell_budget_unvisited_without_rendering
tests/test_shape_fit.py::test_candidate_render_cap_below_view_count_reserves_no_partial_hypothesis
tests/test_shape_fit.py::test_atomic_view_reservation_below_equal_and_above_cap
tests/test_shape_fit.py::test_two_view_then_three_view_cap_two_does_not_starve_the_budget
tests/test_shape_fit.py::test_two_cell_cap_seven_reports_the_shared_cap_as_reached
tests/test_shape_fit.py::test_completed_geometry_decisions_survive_a_later_refinement_cap
```

The one skip is the pre-existing Windows external-symlink case. The full-suite
count rose from the attempt-01 1744 to 1764 exactly by the 20 new collected
regression cases; no previously passing test changed status.

## Deviations, limits and uncertainties

- `has_effective_coverage` treats any effectively positive weight as coverage;
  a numerically negligible but positive coverage therefore still counts as
  observed. That matches the per-pixel weighting the scorer already applies.
- Descriptor neutralisation keeps the mask geometry: two observations with the
  same owned pixels but different excluded-pixel *positions* still differ, by
  design, because the mask is part of the evidence. Only excluded pixel
  *values* are now inert.
- `_readonly` stores a `bytes`-backed view, so retained arrays are not
  writeable through NumPy's flag/assignment APIs or their `.base` chain. A
  hostile caller using raw `ctypes`/`numpy.ctypeslib` against the process is out
  of scope for this contract.
- `FaceScore.ranked()` breaks ties on normalised block-state identity; two
  palette entries that are the *same* state remain order-dependent, but they are
  the same candidate, so no winner changes.
- `shape_fit.py` still resolves `PaletteIndex` through `raycast` (required by
  the accepted M5 monkeypatch test); unchanged in this attempt.
- No image-only or fixture evidence was produced or scored; this task remains
  contracts and extraction only. Promotion gating is unchanged.

Stopping here for manager review and independent verification. No commit, stage,
reset, or clean was performed.
