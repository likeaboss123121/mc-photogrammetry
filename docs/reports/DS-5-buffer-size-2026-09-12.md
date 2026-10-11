# DS-5 - fractional scaled ownership dimensions

Date: 2026-09-12. Scope: DS-5 only. Owned paths touched:
`src/mc_photogrammetry/reconstruction/face_sampling.py`,
`tests/test_face_sampling.py`, this report.

## Defect

Sampling and the shared renderer sized their ownership buffers with different
rules whenever `width * scale` or `height * scale` was fractional:

- `_nearest_surface` (cube fallback) used `max(1, int(round(d * scale)))`.
- `render_volume` / `render_result` (exact path) uses `int(d * scale)`
  (`export/attribution.py:217`).
- the exact path's missing-owner fallback array (`default_owner`) recomputed the
  shape with `max(1, int(round(...)))` instead of taking the rendered shape.

At 101x99 with `scale=0.5`, the fallback allocated 50 rows while the renderer
produced 49; at `scale=0.4` it allocated 40 rows against the renderer's 39.
`source_owner` then described a different pixel grid than
`rendered.ownership`, so the visible-fraction count, the ownership mask handed
to `rectify_face_valid`, and the renderer's own ownership disagreed.

## Changes (production, `face_sampling.py`)

1. `_nearest_surface` (lines 224-225) now truncates with a one-pixel floor:
   `columns = max(1, int(width * scale))`, `rows = max(1, int(height * scale))`,
   matching the renderer's ownership buffer for every non-degenerate frame.
   Docstring states the rule and why rounding is wrong.
2. `sample_visible_faces` (lines 297-299) rejects an out-of-range depth scale:
   `if not 0.0 < depth_scale <= 1.0: raise ValueError(...)`. Both paths need it:
   the fallback buffer indexes itself by scale and `render_volume` already
   refuses a scale outside `(0, 1]`. No caller in `src`, `tests` or `scripts`
   passes a non-default `depth_scale`, so this is a guard, not a behaviour flip
   for existing runs.
3. Exact path (lines 373, 387, 390, 403-404): the rendered ownership shape is
   captured once as `exact_owner_shape = rendered.ownership.shape` and used both
   for missing keys in `exact_owners` and for `default_owner`, replacing the
   rounded recomputation. The owner arrays are the size the renderer actually
   produced.

No other file, scale default, threshold, renderer, raycast/residual/shape-fit or
benchmark code was touched; nothing committed or staged.

## Tests added (`tests/test_face_sampling.py`)

- `test_fallback_ownership_buffer_truncates_like_the_renderer` (parametrized
  `scale`/expected): 101x99 fallback buffer is `(99,101)`, `(74,75)`, `(49,50)`,
  `(39,40)` at scales 1.0, 0.75, 0.5, 0.4, and equals `render_result(...).ownership.shape`
  for the same frame and scale. Pre-fix it produced 50 rows at 0.5 and 40 at 0.4.
- `test_fractional_scale_keeps_fallback_and_exact_face_sets_in_step` (scales
  1.0, 0.5, 0.4): the cube fallback and the exact renderer accept exactly
  `{((0,0,0), "north")}` on a 101x99 frame. Preserves default scale-1 behaviour
  and pins cross-path agreement at fractional scales.
- `test_exact_missing_owner_fallback_uses_the_rendered_ownership_shape`: two
  adjacent blocks with the camera off the shared wall; the renderer culls that
  face while sampling still enumerates it and falls back to an all-false owner.
  `cv2.resize` is spied (actual output, not a mock of our own code) and every
  ownership array resized up to the 79x99 frame must have the rendered source
  shape `(49, 39)`. Pre-fix a `(50, 40)` source appeared, so this control fails
  on the old code.
- `test_sample_visible_faces_rejects_an_out_of_range_depth_scale` (0.0, -0.25,
  1.5): both the fallback call and the exact `blocks`/`palette` call raise
  `ValueError`.

## Verification

- `pytest tests/test_face_sampling.py` -> 26 passed.
- `pytest tests/test_face_sampling.py tests/test_attribution.py
  tests/test_visibility.py tests/test_raycast.py tests/test_raycast_contrast.py
  tests/test_pixel_authority.py tests/test_residual.py` -> 84 passed in 1.38s.
- `pytest tests/test_reconstruction.py` -> 4 passed.
- Pre-fix behaviour of the two new controls was reproduced in-process by
  executing a copy of the module source with the old expressions restored:
  fallback shape `(50, 40)` reached `cv2.resize` as an ownership source at 79x99
  scale 0.5. No full suite was run (DS-5 forbids it; DS-3/DS-4F suite remains
  527 green).

## Limits

- `render_volume` still has no `max(1, ...)` floor (`attribution.py:217`), so a
  frame/scale combination whose truncated product is zero would give the
  renderer a zero-sized buffer while the fallback floors at one pixel. Both
  then disagree only in that degenerate case; the renderer is outside DS-5
  ownership, so it is reported rather than changed.
- `visible` still divides the scaled pixel count by `depth_scale**2`
  (`face_sampling.py:409`). With truncation the buffer holds slightly fewer than
  `scale^2 * width * height` pixels, so the full-resolution equivalent is a
  fraction of a percent low at fractional scales. The DS-5 mandate was buffer
  dimensions, not this conversion; flagged for the lead rather than changed.
- The exact path still enumerates cube-fallback faces for palette states that
  have `shape_boxes` but no `model_faces`, while the renderer paints the shaped
  boxes (pre-existing divergence, M1 backlog; not touched).
- `_nearest_surface` and `render_result` were compared for a single cube and a
  two-block wall; deeper scenes were not swept.

Stopped as instructed: no commit, staging, benchmark, full suite or chaining.
