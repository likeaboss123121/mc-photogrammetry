# DS-3 R1 — residual usable-mask propagation — 2026-09-12

Implementation of the single repair authorized in `AGENT_COMMS.md`
(2026-09-12T09:31:16, "DS-3 R1"): current defect **D1** of
`docs/reports/DS-2-render-sampling-audit-2026-09-12.md`, the residual pass being
the one render path with no usable-pixel authority. Only the four owned paths were
touched: `src/mc_photogrammetry/reconstruction/residual.py`,
`src/mc_photogrammetry/reconstruction/raycast.py`, new `tests/test_residual.py`
and this report. No shape-fitting, footprint, annotation, M5, handoff or comms file
was edited; nothing was staged or committed, and no package, renderer or threshold
was changed. R2–R5 are untouched.

## What changed

`residual.py`:

- `face_residuals` gained the keyword-only `usable_masks: Mapping[str, np.ndarray] | None`
  (`residual.py:69`) and documents the contract in its docstring (`:76-89`).
- The render-scale usable domain is taken from `shape_fit._frame_masks`, imported
  once at `residual.py:37-40`, rather than restating the footprint rule. The two
  scoring paths therefore cannot drift; `shape_fit` itself is unchanged.
- A frame missing from a supplied mapping is skipped before its render
  (`residual.py:103-105`). A wrong-shaped mask still raises `ValueError` through
  `_frame_masks` for every frame that is rendered.
- `None` keeps the historical path byte-for-byte: `valid = seen`, a plain whole-frame
  `INTER_AREA` resize, and gain/typical/score over `seen` (`residual.py:112-117,139-145`).
- With a mapping, the photo is a masked area downsample divided by its own weights
  (`residual.py:129-138`, the same construction as `shape_fit.py:577-589`), and
  `valid = seen & usable`, so only render pixels whose whole source footprint is
  usable reach the gain, the typical residual, the per-face mean and the
  `len(values) >= minimum_pixels` gate (`residual.py:139-151`).
- `residual_proposals` forwards `usable_masks` to `face_residuals`
  (`residual.py:164,171-175`).

`raycast.py`: the `reconstruct_scene_surfaces` thin-block path forwards the
`usable_masks` it already holds to `residual_proposals`
(`raycast.py:1242-1244`), matching the `fit_states` call two lines below.

## Tests (`tests/test_residual.py`, 7 cases)

Synthetic scenes and analytic invariants only; no expected value is copied out of
the implementation. A frame built by `render_result` is reproduced as a uint8 photo
at scale 1.0, so any disagreement is one the test put there.

1. `test_none_keeps_the_historical_residuals_and_proposals` — default and explicit
   `usable_masks=None` return identical residuals and proposals (both non-empty).
2. `test_a_mask_missing_a_frame_makes_that_frame_unusable` — a destructive second
   frame changes nothing when absent from the mapping, and does change the unmasked
   result, so the control is not vacuous.
3. `test_a_wrong_shaped_mask_is_rejected` — `ValueError` from both residual APIs.
4. `test_excluded_pixel_colours_cannot_reach_residuals_or_proposals` — painting an
   excluded rectangle 0 versus 255 leaves residuals and proposals identical, leaves
   every usable face at exactly 0.0 (the photo agrees with the render there) and
   proposes nothing; the same rectangle unmasked changes the scores.
5. `test_a_partly_usable_render_pixel_is_not_evidence` — at scale 0.5 each render
   pixel is exactly one 2x2 source block; leaving 1 of 4 source pixels usable scores
   identically to leaving none, because a partly covered render pixel carries no
   evidence. A partial-coverage acceptance rule would separate the two masks.
6. `test_minimum_pixel_counts_use_only_usable_pixels` — a face with exactly
   `minimum_pixels` owned pixels is present; keeping only half of them usable drops
   it at the same minimum.
7. `test_reconstruct_scene_surfaces_forwards_usable_masks` — a monkeypatched spy on
   `residual_proposals` shows the call site passes the same mapping object through.

## Results

- `pytest tests/test_residual.py tests/test_shape_fit.py -p no:cacheprovider -q`
  (workspace `.venv`, Python 3.10.11, OpenCV 5.0.0, NumPy 2.2.6): **46 passed**
  (7 new + 39 shape-fit).
- Full suite, run after the focused modules were green and with no other worker
  process or recent non-DS-3 edit present: **527 passed in 170.85s**, exit 0
  (`pytest -p no:cacheprovider`). 527 is also the collected count, so nothing was
  deselected or skipped in that total.
- Lint: `ruff` is not installed in the workspace `.venv` and no package was
  installed. Added lines are within the 100-character convention except one
  pre-existing 105-character line at `residual.py:95` and none in the new test
  module.

## Limits and non-goals

- `None` deliberately does **not** adopt `shape_fit`'s `~create_gui_mask` fallback:
  the DS-3 contract requires `None` to preserve behaviour, and the residual pass
  historically scored the whole owned frame. R3 (one mask authority per call) is
  still open, so a caller that supplies masks to `fit_states` but not to the residual
  will still get two pixel authorities.
- D3 (residual renders at `RENDER_SCALE = 0.35` while the sampler uses
  `DEPTH_SCALE = 1.0`) is unchanged; R1 makes the residual obey a mask, not agree
  with the sampler's resolution.
- No caller reaches the masked branch today. `cli.py:382-397` never passes
  `usable_masks` (so the parameter is `None`), and `scripts/benchmark_templates.py:130-135`
  passes the loaded masks but never enables `fit_partial_shapes`, so the residual
  call is not made. An empty mapping is not `None`: it now means "no frame has a
  usable mask", which is exactly what `shape_fit` already does with the same
  input. The shipped CLI and every stored benchmark artifact therefore behave as
  before.
- No end-to-end benchmark, template run, threshold sweep or full40-cell run was
  performed, and no claim is made about changed proposal quality on real captures;
  the evidence is the focused synthetic controls above.
