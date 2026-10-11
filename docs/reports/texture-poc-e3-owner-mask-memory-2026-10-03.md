# E3 bounded exact-owner memory scaling

Date: 2026-10-03
Task: `texture-poc-e3-owner-mask-memory-2026-10-03`
Status: implemented, `py_compile` and the focused face-sampling suite pass,
stopped for lead review. This is a bounded, semantics-preserving memory fix in
one production sampler plus synthetic regressions. It runs no real
corpus/image/reference/JAR access, no E3 exporter rerun and no benchmark, and it
makes no acceptance or promotion claim.

## 1. Scope and owned files

The task authorized exactly three paths:

| Path | Change |
| --- | --- |
| `src/mc_photogrammetry/reconstruction/face_sampling.py` | sparse owner-pixel index + one reusable owner scratch + one reusable footprint scratch |
| `tests/test_face_sampling.py` | five synthetic regressions (duplicate IDs, full-vs-sparse parity, background, end-to-end aggregation, scratch hygiene) |
| `docs/reports/texture-poc-e3-owner-mask-memory-2026-10-03.md` | this report (new) |

The working tree already carried unrelated uncommitted hunks in
`face_sampling.py` and `test_face_sampling.py`. Those hunks were preserved
verbatim; only the regions described in section 3 were edited, and no
pre-existing hunk was moved, reverted or rewritten.

## 2. Root cause

`sample_visible_faces` renders with the shared exact-model renderer
(`render_result`) and then used to build, for every rendered face identifier, a
full `H x W` boolean owner mask:

```python
for identifier, owned in enumerate(face_index):
    exact_owners[owned] = np.logical_or(
        exact_owners.get(owned, np.zeros(exact_owner_shape, dtype=bool)),
        rendered.ownership == identifier,
    )
```

Every mask was retained for the whole call, so retained ownership memory scaled
as **rendered faces x frame pixels**. On the finalized Train Station dataset
(139,425 reference blocks, 6 frames) that is the dominant ~20 GB RSS the lead
observed before artifact emission; the `quads` list and one transient footprint
per quad are linear in faces and do not explain that ceiling.

The same loop also allocated a fresh `np.zeros((height, width), np.uint8)`
footprint per enumerated quad, and reallocated an all-false `default_owner`
per quad for owners with no rendered pixels.

## 3. Change

### 3.1 Sparse owner-pixel index (`_owner_pixels_by_face`, new module function)

`render_result` gives every drawn pixel exactly one sequential face identifier
(`owner[chosen] = len(index)`), while `face_index[identifier]` is the
`(coordinate, slot)` key. The new helper returns, per key, the flat pixel
positions owned by that key:

* background (`ownership < 0`) is dropped;
* identifiers are stably sorted so each identifier's pixels are one
  contiguous slice;
* identifiers that share a key are concatenated and deduplicated with
  `np.unique` — the exact union the old `logical_or` chain produced.

Stored size is one integer per owned pixel, i.e. `O(owned pixels)` and bounded
by the renderer's own ownership buffer, instead of one complete boolean frame
retained per rendered face.

### 3.2 One reusable owner scratch

The loop now keeps a single `np.zeros(exact_owner_shape, dtype=bool)` scratch
and its flat view. Per owner it clears only the pixels the previous face set,
sets the current key's pixels, and reads `visible_pixels = owner_pixels.size`.
A key with no rendered pixels leaves the scratch all-false without allocating.

Clearing only the previously set pixels (rather than the whole frame) keeps the
per-face work proportional to that face's own footprint, so the fix does not
trade the memory term for a new `O(faces x frame)` time term. The invariant is
that the scratch's true pixels are always a subset of the previously set pixel
list, so clearing that list always restores an all-false mask even when an
earlier `gui_mask` cleared some of them in place.

### 3.3 One reusable footprint scratch

`footprint` is allocated once per call and zeroed with `footprint[:] = 0`
before each `cv2.fillConvexPoly`. The fill call itself is byte-for-byte the old
call (same buffer shape, same `quad.astype(np.int32)`), so the in-frame area is
identical; only the per-quad allocation is removed.

## 4. Semantics preservation

* `visible = visible_pixels / (depth_scale * depth_scale)` equals the previous
  `float(source_owner.sum()) / (depth_scale * depth_scale)`, because the
  reconstructed mask's true pixels are exactly the unique owned pixels.
* A face whose key the renderer never wrote (culled or fully hidden) reads an
  all-false mask and count 0, exactly as the old `default_owner` did.
* Duplicate `(coordinate, slot)` identifiers aggregate to their union.
* Face enumeration order, area/visibility/HUD gates, `reject` labels and
  counts, the cube fallback path (`buffer == index`, `_nearest_surface`),
  patch/valid/coverage construction and `FaceSample` fields are unchanged.
* The owner mask handed to `rectify_face_valid` keeps the renderer-produced
  shape (`rendered.ownership.shape`), so the existing
  shape-mismatch resize and GUI-mask intersection still apply.

## 5. Complexity after the change

| Term | Before | After |
| --- | --- | --- |
| Retained ownership | `O(rendered faces x H*W)` booleans | `O(owned pixels)` ints + one `H*W` bool scratch |
| Per-quad ownership work | `O(H*W)` compare + `O(H*W)` or | `O(this face's pixels)` |
| Per-quad footprint | one `H*W` uint8 allocation + fill | one shared `H*W` uint8 buffer + fill |
| Renderer buffers | unchanged `O(H*W)` | unchanged `O(H*W)` |

The retained term that produced the ~20 GB ceiling is removed. The remaining
cost is linear in the number of faces, not faces x frame.

## 6. Tests and exact results

Added to `tests/test_face_sampling.py`:

1. `test_owner_pixels_by_face_aggregates_duplicate_face_ids` — two identifiers
   sharing one `(coordinate, slot)` union to the expected four flat pixels.
2. `test_owner_pixels_by_face_matches_full_frame_logical_or_masks` — for a
   fixed buffer with background and a three-identifier key, the sparse
   reconstruction is `array_equal` to the dense `logical_or` mask for every key,
   and the split key is a genuine union.
3. `test_owner_pixels_by_face_drops_background_pixels` — an all-`-1` buffer
   yields no owners.
4. `test_exact_sampling_aggregates_duplicate_owner_ids_into_one_mask` — a real
   exact-model palette with two north faces on one cell; the renderer assigns
   two distinct identifiers, both splits own pixels, and both captured
   `source_valid` masks equal the aggregated dense union.
5. `test_exact_owner_scratch_does_not_bleed_across_missing_owners` — a
   two-block scene where the renderer culls the shared wall while sampling still
   enumerates it; every mask reaching the rectifier is either one owner's exact
   dense mask or empty, and at least one face is empty.

Commands and results (foreground, ordinary workspace commands):

* `python -m py_compile src/mc_photogrammetry/reconstruction/face_sampling.py tests/test_face_sampling.py` -> exit 0.
* `pytest tests/test_face_sampling.py -o addopts='' -q --no-header` -> **57 passed in 0.32s**.
* Dependent consumer check `pytest tests/test_export_factory_texture_patch_dataset.py -o addopts='' -q --no-header` -> **41 passed in 0.88s** (real exact sampler with multiple blocks).
* `git diff --check` on both scoped files -> clean.

No corpus, image, reference, mask, camera, scene or JAR content was read; no
exporter or benchmark was rerun.

## 7. Scoped hashes (SHA-256)

| Path | SHA-256 | Bytes |
| --- | --- | --- |
| `src/mc_photogrammetry/reconstruction/face_sampling.py` | `14be0f406364c2175db84aa4909825a2d059511b6d39c0c33debf4b3d2ecba6e` | 31,238 |
| `tests/test_face_sampling.py` | `a4d316bd82cc801b8a3d526d02c40675a979ab3dca6e9bbef3f6550dcd7d1b8c` | 33,301 |

The report's own post-freeze hash is recorded in `AGENT_COMMS.md` after this
file is written.

## 8. Residual complexity and uncertainties

1. **`quads` is still materialized up front.** One tuple per enumerated face
   (projected quad, per-vertex depths, texture quad, key, facing) is retained
   for the whole call before any gate. For Train Station that is roughly
   0.8M entries of Python/numpy object overhead. This is linear and bounded, but
   it is the next-largest retained term and was not part of the authorized fix.
2. **`texture_quad` is computed for every quad** even though it is only needed
   after the area and visibility gates. Deferring it is a further reduction but
   would reorder work and was out of scope.
3. **The footprint is still zeroed and summed over the full frame per quad.**
   The allocation is gone, but the scan remains `O(faces x H*W)` time. A
   bbox-local fill would change the exact `fillConvexPoly` rasterization
   contract, so it was deliberately not attempted.
4. **The cube fallback** still forms `buffer == index` per quad (one transient
   full-frame bool at a time); it was already linear and is unchanged.
5. **No real-run RSS was measured.** The task prohibited corpus/exporter reruns,
   so the improvement is argued from the removed `faces x frame` term and the
   synthetic parity regressions, not from a live Train Station RSS reading. A
   lead-authorized exporter rerun is the natural confirmation.
6. The sparse grouping adds a one-off `argsort` over owned pixels per call
   (`O(P log P)` time, `O(P)` transient memory, `P <= ownership buffer size`).

## 9. Prohibitions honored

No junction/reparse-point operations, recursive deletion, encoded or obfuscated
commands, generated executables, installers, background processes, jobs,
workflows, subagents, persistence, Defender changes/exclusions,
system-directory writes, live Minecraft/client automation, dependency changes,
unrelated edits or commit. Only the three owned paths were touched; unrelated
dirty work was preserved.


## Lead verification after worker handoff

The lead independently reran `py_compile`, `tests/test_face_sampling.py`, and the
dependent synthetic E3 exporter suite: **98 tests passed** together. The scoped
source/test hashes still match the worker report. A finalized Train Station E3
export then completed: 2,507 patch examples over the 3 HUD-hidden frames. During
foreground polling its sampled working set stayed near 0.58 GB; the earlier
pre-fix attempt on the same dataset reached about 20 GB and was stopped before
output. This is a sampled working-set comparison, not a controlled peak-RSS
benchmark; the observed memory reduction is approximately 97%. The E3 corpus
results and limitations are in [the exporter report](texture-poc-e3-factory-patch-dataset-export-2026-10-03.md).
