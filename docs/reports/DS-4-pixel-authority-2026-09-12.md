# DS-4: one internal pixel authority in `reconstruct_scene_surfaces` (R3 + DS-4F)

2026-09-12. Agent: DeepSeek. Owner-verified against the DS-4 assignment
(2026-09-12T09:42) and the DS-4F correction (2026-09-12T09:49) in
`AGENT_COMMS.md`. No commit, no staging, no chaining.

## Scope and change

Owned paths touched:

* `src/mc_photogrammetry/reconstruction/raycast.py`
* `tests/test_pixel_authority.py`
* `scripts/benchmark_templates.py` (DS-4F)
* `tests/test_benchmark_templates.py` (DS-4F)
* `docs/reports/DS-4-pixel-authority-2026-09-12.md` (this file)

Nothing else was edited; DS-3's `residual.py` and `tests/test_residual.py`
changes were left exactly as they were.

`raycast._pixel_authority(frames, usable_masks)` normalises the usable-pixel
domain exactly once per `reconstruct_scene_surfaces` call, after argument
validation and before pass one. It returns `(usable, excluded)`, both keyed by
every frame, both in source-image coordinates:

* `usable_masks is None` derives `usable[f] = ~create_gui_mask(h, w)` per frame
  and `excluded[f] = ~usable[f]`; the historical sampling domain is preserved.
* A supplied mapping is authoritative: `usable[f]` is the caller's mask cast to
  bool (the same object when it is already a bool array), a frame the mapping
  omits is all-false, and a mask whose shape is not the frame's `(h, w)` raises
  `ValueError` at this point - before any sampling, scoring or scoring-adjacent
  work, and regardless of `fit_partial_shapes`.

The one authority is handed to every pixel consumer:

| consumer | call | receives |
| --- | --- | --- |
| face sampling, pass two (both calls) | `sample_visible_faces(..., gui_mask=...)` | `excluded[f]` |
| occupancy proposal sampling (`propose_colour_gap > 0`) | `proposed_cells(..., gui_masks=...)` | `excluded` |
| centre-patch fallback, pass three | `frame_usable = usable_pixels[frame_id]` | `usable[f]` |
| residual proposals | `residual_proposals(..., usable_masks=...)` | `usable` (the map object itself) |
| partial shape fitting | `fit_states(..., usable_masks=...)` | `usable` (the same map object) |

The occupancy path was already called with `create_gui_mask` per frame; it is the
same sampling authority, so it was moved onto `excluded` too. For
`usable_masks is None` its content is byte-identical to the previous behaviour.
`residual.py` was not touched, so its standalone `None` compatibility is
unaffected; `reconstruct_scene_surfaces` now never forwards `None` to it, which
is the point of the task - the thin-block CLI can no longer sample under the GUI
mask and then score residuals over unmasked pixels.

## DS-4F: pass-three fallback and benchmark caller

**Pass-three fallback.** The centre patch `image[y - r : y + r + 1,
x - r : x + r + 1]` is now admitted only when the whole matching region of
`usable[f]` is true:

```python
if not frame_usable[
    y - sample_radius : y + sample_radius + 1,
    x - sample_radius : x + sample_radius + 1,
].all():
    continue
```

The check runs before the RGB median and the descriptor are computed, so an
unusable source pixel cannot reach the palette classifier, `fallback_share` or
the volume. An all-false or missing authority therefore produces zero fallback
evidence, and the claimed single authority now covers all four consumers.

**Benchmark caller.** `scripts/benchmark_templates._load_usable_masks` returns
`None` when no mask paths were given, and `run` forwards that value
unconditionally. `{}` is no longer manufactured for "no masks", so an unfrozen
run keeps the reconstructor's derived (`~create_gui_mask`) domain instead of
becoming an authoritative mapping with every frame missing. The report's
`usable_pixels` comprehension now guards the optional mapping
(`sorted((usable_masks or {}).items())`); the policy strings and `visible_cells`
already treat falsy/`None` as "all decoded pixels". Explicitly passing `{}` to
`reconstruct_scene_surfaces` still means "supplied authority, every frame
missing" - that semantic lives in `_pixel_authority` and is pinned by
`test_an_empty_mapping_is_not_the_same_as_none`.

## Focused checks

`tests/test_pixel_authority.py` has 9 tests: the 6 spy tests from R3 plus three
actual-output controls that exercise the production path with no spies.

Spy controls (unchanged): both sampling passes get the same `~mask` object, the
residual and fitting calls receive the identical authority map by identity, a
missing frame is all-false, `{}` is not `None`, wrong shape raises before any
consumer, `None` derives `~create_gui_mask` for every consumer, and
`fit_partial_shapes=False` leaves the downstream spies untouched.

Actual-output controls (DS-4F), bounded 128x128 synthetic frame, 8x8x8 scene,
single camera, `black_concrete`/`stone`/`white_concrete` palette:

* `test_masked_fallback_pixels_cannot_change_the_output`: with the left half
  excluded, mutating only the excluded half from RGB 5 to RGB 250 leaves the
  `fallback_samples` count and the whole volume signature unchanged. The control
  asserts positive fallback evidence so the equality is not vacuous.
* `test_no_usable_pixels_yields_no_image_derived_evidence` (`missing` and
  `all-false`): the volume is empty and `resolved_face_samples`,
  `fallback_samples` and `blocks_from_fallback_only` are all zero, including
  with `fit_partial_shapes=True`.

`tests/test_benchmark_templates.py` adds a caller test at the `run` boundary:
with no mask paths the value handed to `reconstruct_scene_surfaces` is `None`
(both `None` and `{}` inputs), and with a JSON mask path it is the loaded
mapping. `_load_usable_masks({}, frames)` is also pinned to `None`.

```
tests/test_pixel_authority.py tests/test_benchmark_templates.py
tests/test_residual.py tests/test_raycast.py
22 passed in 1.27s

plus test_raycast_contrast.py test_shape_fit.py test_face_sampling.py
test_gui_mask.py test_visibility.py
102 passed in 2.21s
```

Run with the workspace `.venv`, `-p no:cacheprovider`. No full suite was run
(the task forbids it). `ruff` is not installed in this `.venv`, so no lint run
was made and none was installed. Nothing was staged or committed.

## Limits and findings for the lead

1. **Pass-three fallback is now inside the authority.** Resolved in DS-4F: the
   fallback requires its entire source patch usable, and all-false/missing
   authority yields zero fallback evidence. This changes the measured classifier
   path for masked runs (fallback/upward shares) relative to the DS-4 R3 build;
   that is the intended correction, not a threshold change.
2. **Benchmark regression is closed.** Resolved in DS-4F: an unfrozen benchmark
   run forwards `None` and reconstructs under the derived GUI domain again,
   instead of the `{}` all-unusable mapping that lost every rectified face.
3. **Pre-existing visibility-policy mismatch remains (M1 backlog).**
   `export.visibility.visible_cells` treats a key missing from `usable_masks` as
   "no mask" (unmasked), the opposite of the DS-3/DS-4 reconstruction
   authority, and `None` as "all pixels". This is a cross-policy redesign and was
   deliberately not broadened here; flagged as M1 backlog.

No production algorithm, threshold, rendering path, exact-model routing, scale,
annotation or benchmark metric was changed. The S-D/M5/footprint artifacts are
untouched.
