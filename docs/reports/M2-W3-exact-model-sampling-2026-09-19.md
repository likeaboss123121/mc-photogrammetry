# M2-W3 — exact-model production sampling

Date: 2026-09-19
Task: `m2-w3-exact-model-sampling-2026-09-19` (routed implementation, attempt 02, Astra contract amendment)
Role: bounded developer (DeepSeek/Sol-equivalent) under the direct-shell policy in
[`docs/deepseek-worker.md`](../../docs/deepseek-worker.md)
Status: attempt-02 complete; diagnostics-serialization repair applied, required checks pass,
**stopped for manager/Astra review**

## Context

Attempt 01 was blocked by sequencing: both production `sample_visible_faces` calls in
`reconstruct_scene_surfaces` execute before per-cell `BlockState` assignment, so no
authoritative state input existed and inventing one was correctly refused
(`runs/m2-w3-exact-model-sampling-2026-09-19/attempt-01/`). Astra's amendment authorizes
an **existing-hypothesis snapshot** parameter and a fixed decision contract. This report
covers that implementation. Jev route
`route-m2-w3-exact-model-sampling-2026-09-19-3d0d3d753efa` (`sol`, bounded, write-scoped)
was used unchanged.

## Contract implementation

### `face_sampling.py`

- `SamplingGeometryDecision` — a frozen, slotted record with `pass_name`, `mode`,
  `target_count`, `missing_state_count`, `missing_model_count` and `fallback_reason`.
  Modes are `EXACT_MODEL` / `CUBE_GEOMETRY`; reasons are `STATES_UNAVAILABLE`,
  `STATES_INCOMPLETE`, `MODEL_METADATA_UNAVAILABLE` (or `None` for exact).
  Pass names are `surface` and `surface_proposed`.
- `validate_sampling_states(...)` — `None` stays `None`; a non-mapping raises
  `TypeError`; a malformed coordinate (wrong arity, non-integer, `bool`, string, `None`)
  raises `ValueError`; a non-`BlockState` value raises `TypeError`. Returns a fresh
  `dict` with Python-integer keys, so the snapshot cannot be mutated underneath a
  resolved decision.
- `exact_model_states(palette)` — palette states with non-empty `model_faces`, matched by
  exact state identity only. The name-keyed alias in `attribution.face_model_faces` is
  deliberately **not** consulted, so a sibling state sharing a name cannot lend geometry.
- `resolve_sampling_geometry(targets, sampling_states, palette, pass_name)` — decides one
  call's geometry over exactly that call's target set:
  - `sampling_states is None` → cube, `states_unavailable`,
    `missing_state_count == target_count`;
  - a target with no supplied state → cube, `states_incomplete`, counted;
  - every target has a state but some state lacks palette model metadata → cube,
    `model_metadata_unavailable`, counted;
  - otherwise → exact model, `fallback_reason is None`.
  Snapshot entries outside the target set are ignored, not counted, and cannot become
  targets.
- `exact_sampling_blocks(decision, targets, sampling_states)` — `None` for a cube
  decision (so the sampler keeps the legacy ownership buffer untouched); otherwise a
  mapping covering **every** target.

### `raycast.py`

- `reconstruct_scene_surfaces` gains keyword-only
  `sampling_states: Mapping[BlockCoordinate, BlockState] | None = None`, appended after
  `usable_masks`, so positional behavior is unchanged.
- The snapshot is validated once near the top, before `_pixel_authority` and before any
  sampling/scoring. It never reaches `raw`, `PaletteIndex`, `_supported`,
  `_smooth_adjacent_states`, exports, or any CLI/UI input; it cannot add a coordinate to
  `surface`/`proposed`.
- Geometry is resolved **once per call**, before the per-frame sampling loop (the target
  set is frame-independent). Pass one resolves `surface`; pass two, when it runs,
  resolves `surface | proposed`. Each call is entirely one mode; a call whose targets are
  not fully covered keeps the whole call on cube geometry, so a single depth buffer can
  never mix exact and cube ownership.
- `diagnostics["sampling_geometry"]` receives a tuple of the executed decisions, in pass
  order (one entry when no proposals exist, two when the second pass runs).
- The two calls, their target sets, the `seen_here` duplicate suppression, the GUI mask,
  `rejections`, `depth_scale`, and the centre-patch fallback are unchanged. With
  `sampling_states=None` neither call receives `blocks`/`palette`, so the legacy path is
  byte-for-byte the pre-change behavior.

Contract prohibitions honored: no name-only fallback, no fabricated state, no
first-palette choice, no mixed exact/cube depth buffer within a call, no target loss.

## Tests added

`tests/test_face_sampling.py` (26 → 51 collected, +25): validation of `None`, non-mapping,
malformed-coordinate and non-state inputs; the empty-snapshot vs no-snapshot distinction;
each fallback reason and its counts; out-of-target entries ignored; name-only model match
refused; the decision's frozen/typed nature; `exact_sampling_blocks` completeness; and two
real-renderer cut-out cases (transparent-pixel RGB/coverage invariance, reduced ownership
vs cube, and masked-pixel marking of the owned domain).

`tests/test_raycast.py` (2 → 8 collected, +6):

- `test_exact_model_cutout_sampling_end_to_end` — real `reconstruct_scene_surfaces`
  through the actual sampler/renderer. A front-facing transparent cut-out over a
  contrasting background: exact resolves the owned face and its reconstruction is
  invariant to the background RGB (`cutout` both ways), while the cube fallback fails
  the uniformity gate, uses its centre patch and changes with the background
  (`dark` vs `mid`). It also pins the recorded decision mode/reason/counts.
- `test_exact_sampling_records_both_passes_with_complete_states` — both passes exercised
  via `extra_proposals`, recorded as `surface`/`surface_proposed`, target counts `1`/`2`.
- `test_second_pass_falls_back_whole_call_when_a_proposal_is_uncovered` — first pass
  exact, second pass whole-call cube for `states_incomplete` and
  `model_metadata_unavailable`.
- `test_sampling_states_cannot_create_coordinates_or_seed_scores` — an out-of-target
  entry leaves the volume and decision identical and creates no coordinate.
- `test_cube_geometry_passes_no_blocks_and_exact_threads_the_target_set` — a spy pins
  that the cube path passes neither `blocks` nor `palette`; the exact path passes a
  mapping exactly equal to each call's target set (including the full
  `surface | proposed` set on the second call) keyed to the same palette object.
- `test_invalid_sampling_states_raise_before_any_sampling` — non-mapping, malformed
  coordinate and non-state value all raise with zero sampler calls.

## Preserved behavior and boundaries

- Existing face-sampling tests retained: partial-model/exact-ownership, masked-pixel
  overlap, fractional-scale fallback/exact parity, renderer-buffer-shape fallback.
- No M4 placement/V2-B, M5 scheduler/scorer, CLI/UI wiring, schema, evaluator gate, GPU,
  launcher, security or frozen artifact was touched. No advanced-only metadata boundary
  moved; `sampling_states` is an in-memory existing-hypothesis snapshot only, and normal
  image-derived reconstruction is unchanged when it is absent.
- `HEAD` remains `c6dba2cf4610876db7d24361eb13fc9ea3ceca0c` on `main`; nothing staged
  (`git diff --cached` empty); no commit/stage/reset/clean was performed.

## Files changed (SHA-256, lowercase; line counts)

```
4af36c78a5d85d5f92b5b6c2cf228de13f415f41486e442ea63010edc247ffd3  2371  src/mc_photogrammetry/reconstruction/raycast.py
50ccee2e5a800d1807d40fb4eaac9011210406fc7799594eda805955ce04600b   610  src/mc_photogrammetry/reconstruction/face_sampling.py
5817e8f25febeba3a98af7aa34063ac40c850ee42683f18449e65e2c645b4c9e   333  tests/test_raycast.py
74814842ce0637333d6b2ac1eb58660b7e694e381e11d2ccc15ce5a93b17e725   616  tests/test_face_sampling.py
```

Attempt 02 edited only these four routed paths and this report. `raycast.py` and
`face_sampling.py` were already modified in the working-tree baseline by earlier tasks;
this attempt's hunks are the M2-W3 additions described above.

## Checks run

```
$ .\.venv\Scripts\python.exe -m py_compile `
    src\mc_photogrammetry\reconstruction\raycast.py `
    src\mc_photogrammetry\reconstruction\face_sampling.py `
    tests\test_raycast.py tests\test_face_sampling.py                 # exit 0

$ .\.venv\Scripts\python.exe -m pytest tests\test_face_sampling.py tests\test_raycast.py `
    tests\test_pixel_authority.py tests\test_residual.py tests\test_shape_fit.py `
    -q -p no:cacheprovider -o addopts=                                # 120 passed

$ .\.venv\Scripts\python.exe -m pytest tests\test_attribution.py tests\test_visibility.py `
    tests\test_model_geometry.py tests\test_raycast_contrast.py tests\test_surface_pruning.py `
    tests\test_stereo.py tests\test_adjacency.py tests\test_complete_palette.py `
    tests\test_texture_cache.py tests\test_facade.py tests\test_reconstruction.py `
    tests\test_exposure_reconciliation.py tests\test_light_cast.py tests\test_shape_persistence.py `
    tests\test_appearance_scoring.py tests\test_face_observation.py `
    -q -p no:cacheprovider -o addopts=                                # 193 passed

$ .\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider -o addopts=
                                                    # 1795 passed, 1 skipped in 460.14s
```

The full-suite count rose from 1764 to 1795 exactly by the 31 new collected cases
(+25 face-sampling, +6 raycast); no previously passing test changed status. The one skip
is the pre-existing Windows external-symlink case.

`diff-scope`: only the four routed paths plus this report differ from the attempt-02
starting state; no forbidden path (M4 placement/V2-B, M5 scheduler/scorer, CLI/UI,
schema, evaluator, GPU, launcher, security, artifacts) appears.

## Deviations, limits and uncertainties

- **Validation is global, relevance is per-target.** Every snapshot key/value is
  type-checked at entry, including entries outside a call's target set; only relevance
  (counting and block selection) ignores out-of-target entries. "Invalid mapping
  types/coordinates raise" is read as unconditional.
- **An empty target set resolves to exact with `target_count == 0`** (the "every target"
  condition is vacuously satisfied). This is behaviorally identical to the cube path
  (`sample_visible_faces` returns `[]` either way), and no fallback reason among the three
  named applies.
- **`missing_model_count` is reported only once every target has a state.** When a target
  state is missing, the decision returns `states_incomplete` with `missing_model_count`
  zero, because model metadata cannot be assessed for a state that was not supplied.
- **A decision is recorded only for a pass that actually runs.** With no proposals the
  second pass is skipped as before, so `sampling_geometry` has one entry rather than a
  synthetic record for a call that did not happen.
- No image-only or real-screenshot evidence was scored; the end-to-end regression is a
  synthetic cut-out fixture exercising the actual sampler and renderer. Promotion gating
  and normal-mode image-derived behavior are unchanged.

Stopping here for manager review and independent Astra verification. No commit, stage,
reset, or clean was performed.

## Attempt 03 — JSON-safe `sampling_geometry` diagnostics

Task: `m2-w3-diagnostics-serialization-2026-09-19` (routed repair; Astra follow-up to
attempt 02). Jev route
`route-m2-w3-diagnostics-serialization-2026-09-19-4d45ab8d4730` (`sol`, bounded,
write-scoped). Sampling behavior is not changed.

### Why

Attempt 02 stored the live frozen `SamplingGeometryDecision` objects in
`diagnostics["sampling_geometry"]`. `json.dumps` cannot encode a dataclass, so any caller
that serialized reconstruction diagnostics failed on that key. The typed decision must
stay the internal contract; only its projection into diagnostics needs to change.

### Changes

- `SamplingGeometryDecision.to_record()` (`face_sampling.py`) — returns exactly six
  primitive fields: `pass_name`, `mode`, `target_count`, `missing_state_count`,
  `missing_model_count`, `fallback_reason`. Counts are coerced with `int()` and the reason
  with `str()`/`None`, so a numpy integer, a `bool` or any subclass cannot leak into a
  record and re-decode as a different type. `fallback_reason` is `None` exactly when the
  call was exact. `resolve_sampling_geometry` still returns the frozen typed decision and
  `exact_sampling_blocks` still consumes it — nothing internal changed.
- `raycast.reconstruct_scene_surfaces` now stores
  `diagnostics["sampling_geometry"] = tuple(decision.to_record() for decision in executed)`.
  Pass order and cardinality are unchanged: one record when the proposal pass did not run,
  two when it did, and no synthetic record for a call that never happened.

### Tests added

- `tests/test_face_sampling.py::test_decision_to_record_is_primitive_and_json_round_trips`
  — all four outcomes (`states_unavailable`, `states_incomplete`,
  `model_metadata_unavailable`, exact), asserting the exact key set, that every value is
  `str`/`int`/`None`, and a real `json.dumps`/`json.loads` round trip.
- `tests/test_raycast.py::test_sampling_geometry_diagnostics_json_round_trip_over_both_passes`
  — drives the real two-pass reconstruction for an exact call and both whole-call
  fallbacks, encodes/decodes the recorded tuple and compares it field for field, and pins
  the single-entry case when no proposal pass runs.
- The four existing raycast diagnostics assertions were changed from attribute access to
  record-key access; their expected modes, reasons and counts are unchanged.

### Files changed this attempt (SHA-256, lowercase; line counts)

```
8a4c050f6017c28ad4b6aea92e82f8eabc130b2674b600b46acbac91080d2fa5  632  src/mc_photogrammetry/reconstruction/face_sampling.py
60ad571c5a88d85ce3b51e479fc299181ed58f18488e01e8a5ba5b33e0bd1b1c 2378  src/mc_photogrammetry/reconstruction/raycast.py
0af072ee874a90269a8aaee3592d9c4882da73ee472eb4f1efc0ab90d1fd3844  666  tests/test_face_sampling.py
ba2e1e20960b9ace983daadf323ee1d3ac8039668e3d46d0003c9fa4721e2761  431  tests/test_raycast.py
```

The hash table in the attempt-02 section above is that attempt's snapshot and no longer
matches the working tree; this table is the post-repair state. Only these four files plus
this report were written this attempt.

### Checks run

```
$ .\.venv\Scripts\python.exe -m py_compile `
    src\mc_photogrammetry\reconstruction\raycast.py `
    src\mc_photogrammetry\reconstruction\face_sampling.py `
    tests\test_raycast.py tests\test_face_sampling.py                 # exit 0

$ .\.venv\Scripts\python.exe -m pytest tests\test_face_sampling.py tests\test_raycast.py `
    -q -p no:cacheprovider -o addopts=                                # 61 passed

$ .\.venv\Scripts\python.exe -m pytest `
    "tests\test_raycast.py::test_sampling_geometry_diagnostics_json_round_trip_over_both_passes" `
    "tests\test_face_sampling.py::test_decision_to_record_is_primitive_and_json_round_trips" `
    -q -p no:cacheprovider -o addopts=                                # 2 passed

$ .\.venv\Scripts\python.exe -m pytest tests\test_face_sampling.py tests\test_raycast.py `
    tests\test_pixel_authority.py tests\test_residual.py tests\test_shape_fit.py `
    tests\test_attribution.py tests\test_visibility.py tests\test_model_geometry.py `
    tests\test_raycast_contrast.py tests\test_surface_pruning.py `
    -q -p no:cacheprovider -o addopts=                                # 175 passed

$ .\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider -o addopts=
                                                    # 1797 passed, 1 skipped in 461.22s
```

The full-suite count rose from attempt 02's 1795 to 1797 exactly by the two new
regressions; no previously passing test changed status. The one skip is the pre-existing
Windows external-symlink case.

`diff-scope`: only the four routed code/test paths plus this report were written this
attempt. Across `src/`, `tests/`, `docs/` and `scripts/`, the only files whose modification
time is at or after the attempt's start are the five routed paths; no forbidden path (M4
placement/V2-B, M5 scheduler/scorer, CLI/UI, schema, evaluator, GPU, launcher, security,
artifacts) was touched. `git diff --cached` is empty and `HEAD` is still
`c6dba2cf4610876db7d24361eb13fc9ea3ceca0c` on `main`; no commit, stage, reset or clean was
performed. Positional APIs, masks, state validation, exact-model routing and fallback
reasons are unchanged.

Stopping here for manager review and independent Astra verification.
