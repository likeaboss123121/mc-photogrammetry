# M4 image-derived session producer/deserializer implementation (2026-09-19)

Task: bounded M4 production seam requested 2026-09-19
Repair 1: M4 session-integrity blocker (reopenable frame buffer) fixed 2026-09-19
Repair 2: unresolved-fact substring bypass (resolved-wording note) fixed 2026-09-19
Worker: Sol-equivalent CLI worker (deepseek-flash), bounded implementation
Status: **implementation complete; the reopenable-buffer integrity blocker and the
unresolved-fact substring bypass are repaired; the focused suite (45) and the
required adjacent suites pass; stopped for Manager/Astra review.**

Write scope used (exactly three routed paths):

- `src/mc_photogrammetry/pipeline/image_derived_session.py` (new)
- `tests/test_image_derived_session.py` (new)
- `docs/reports/M4-image-derived-session-2026-09-19.md` (this file, new)

No other source, test, report, contract, schema, CLI/UI, export, `BlockVolume`,
state-fusion, evaluation, calibration, handoff, or wiring file was created or
edited. `AGENT_COMMS.md` was **not** touched (its repository modification
predates this task). No git stage/commit/reset/clean was run. The only
side effect of the required `py_compile`/`pytest` runs is refreshed standard
`__pycache__` bytecode caches.

## Objective

Build a typed image-derived session producer/deserializer that:

- consumes only an ordered mapping `frame_id -> uint8` grayscale/RGB numpy image,
  plus optional `source_paths` labels;
- delegates camera recovery to the existing
  `pipeline.image_only.solve_image_cameras` (no calibration algorithm is
  reimplemented);
- adapts every solved estimate with
  `pipeline.image_only.camera_model_from_estimate` at
  `LOCAL_FRAME_ORIGIN`;
- preserves the unresolved scale / translation / yaw-quadrant facts;
- never accepts a camera manifest, F3/manual coordinate, direction, bound, scene
  transform, annotation, or other metadata;
- captures every frame into a private, `bytes`-backed read-only array that cannot
  be reopened, records a per-frame SHA-256, computes a deterministic session
  digest, declares an explicit schema/version, and refuses malformed input with
  typed reasons;
- serializes only frame ids, source labels, image shapes/digests, camera solution
  fields, unresolved flags, and local-frame camera data, and deserializes by
  verifying record/digest integrity without fabricating pixels or claiming
  scale/translation.

No CLI/UI/export/`BlockVolume`/state-fusion edits were made.

## Repair 1: non-reopenable frame capture

The original producer captured each frame with `np.array(image, copy=True)` and
then `pixels.setflags(write=False)`. Because numpy owned that freshly allocated
writable memory, a later `setflags(write=True)` reopened it — including inside
the delegated solver, which receives the captured mapping. The M4 integrity
claim ("later caller mutation cannot change the session") therefore rested on a
flag a writer could clear rather than on an immutable buffer.

The repair centralizes frame capture in `_immutable_pixels(image)`: the exact
C-order bytes are buffered with `ndarray.tobytes(order="C")` and wrapped by
`np.frombuffer(..., dtype=np.uint8).reshape(image.shape)`. The resulting array is
read-only and its `base` chain terminates at an immutable `bytes` object, so
numpy refuses `setflags(write=True)` and every in-place write. Shape, dtype,
per-frame digest, session digest, serialization, and the deserializer are
unchanged. `_validated_image(..., require_immutable=True)` now also refuses any
frame whose base chain is numpy-owned or a view into another array, so
`ImageDerivedFrame` cannot be constructed around a reopenable read-only buffer.

Regression tests pin three properties: `setflags(write=True)` on a produced frame
raises `ValueError`; a hostile delegated solver cannot reopen or write the frame
and the captured bytes/digest are unchanged; and post-build frame/session digests
plus the record round trip stay valid. A fourth test proves the frame contract
refuses a reopenable numpy-owned read-only copy. The normal-mode surface is
untouched: the producer signature is still exactly `(frames, *, source_paths=None)`
with no metadata parameter and no `**kwargs`.

## Repair 2: exact canonical unresolved-note validation

`_require_unresolved_facts` matched each unresolved fact with
`note.startswith(f"{frame_id}: ") and fact in note`. A resealed record whose notes
were only the resolved wording (`"frame.png: position"`, `"frame.png: scale"`,
`"frame.png: yaw quadrant"`) therefore satisfied the check: the frame-id prefix
matched and the fact noun was a substring of the note, so the record could claim
the facts were still recorded while actually naming them as resolved.

The repair replaces substring matching with the exact canonical per-frame notes
emitted by `reconstruction/calibration.py` lines 312-314:
`"{id}: position (vanishing points fix no translation)"`,
`"{id}: scale (needs the lattice period)"`, and
`"{id}: yaw quadrant (recovered modulo 90 degrees)"`. For every frame id, each
canonical note must appear verbatim exactly once, and any note prefixed by the
frame id that is not one of the three canonical notes is refused as
`SESSION_UNRESOLVED_FACT_MISSING`. This rejects resolved-only wording, near-miss
wording, duplicates, and extra prefixed notes that could be read as a resolution
claim. The canonical producer path is unchanged: `to_camera_solution` already
emits exactly these strings, so every honest record still round-trips. The
bytes-backed immutability capture (Repair 1) and all other behavior are
untouched.

## Public API (small and documented)

Module `mc_photogrammetry.pipeline.image_derived_session`:

- `IMAGE_DERIVED_SESSION_SCHEMA = "mc-photogrammetry-image-derived-session"`
- `IMAGE_DERIVED_SESSION_VERSION = 1`
- reason constants `SESSION_INPUT_INVALID`, `SESSION_FRAME_INVALID`,
  `SESSION_CAMERA_UNRESOLVED`, `SESSION_RECORD_INVALID`,
  `SESSION_SCHEMA_UNSUPPORTED`, `SESSION_DIGEST_MISMATCH`,
  `SESSION_UNRESOLVED_FACT_MISSING`, `SESSION_FORBIDDEN_CLAIM`
- `ImageDerivedSessionError(ValueError)` with a machine-readable `.reason` and a
  human `.detail`
- `ImageDerivedFrame` (frozen dataclass: frame id, label, read-only pixels or
  `None`, shape, digest; `has_pixels`)
- `ImageDerivedSession` (frozen dataclass: schema, version, frames, cameras,
  components, unresolved, `scale_known`, local cameras, digest; `has_pixels`)
- `image_sha256(image)` — shape-framed, dtype-framed C-order SHA-256
- `build_image_derived_session(frames, *, source_paths=None)`
- `image_derived_session_to_record(session)`
- `image_derived_session_from_record(record)`

The producer signature is exactly `(frames, *, source_paths=None)`; there is no
`**kwargs` and no metadata parameter.

## Exact source contracts bound

| Contract | Use |
| --- | --- |
| `pipeline.image_only.solve_image_cameras` | sole camera-recovery delegate |
| `pipeline.image_only.camera_model_from_estimate` | sole estimate->camera adapter, called with `position=LOCAL_FRAME_ORIGIN` |
| `pipeline.image_only.LOCAL_FRAME_ORIGIN` | the only accepted local camera position |
| `pipeline.image_only.ImageOnlyCameraError` | re-typed into `SESSION_CAMERA_UNRESOLVED` |
| `pipeline.contracts.CameraSolution` | `.cameras`, `.components`, `.unresolved`, `.scale_known` |
| `pipeline.contracts.CameraEstimate` | `.intrinsics`, `.pose`, `.residual_pixels`, `.confidence` |
| `pipeline.contracts.CameraIntrinsics` | `.image_size`, `.focal_length_pixels`, `.principal_point_pixels` |
| `pipeline.contracts.CameraPose` | `.position`, `.yaw_degrees`, `.pitch_degrees`, `.roll_degrees` |
| `reconstruction.camera.CameraModel` | `.position`, `.yaw_degrees`, `.pitch_degrees`, `.fov_degrees`, `.eye_height`, `.intrinsics`, `.roll_degrees` |

Every required field bound; no field was missing and no substitute was invented.

## Guarantees

1. **Immutable, non-reopenable pixels.** Each input frame's exact C-order bytes
   are buffered into an immutable `bytes` object and wrapped with
   `np.frombuffer(...).reshape(shape)`, so the captured array is read-only *and*
   its base is immutable bytes. Unlike the earlier `np.array(copy=True)` +
   `setflags(write=False)` capture, `setflags(write=True)` is refused by numpy
   and a delegated solver cannot mutate or reopen the frame. Later caller
   mutation cannot change the session or any digest. `ImageDerivedFrame`
   re-validates rank/dtype/positivity, refuses any reopenable (numpy-owned or
   array-view) base, and re-derives the digest of a produced frame.
2. **Per-frame SHA-256.** The frame rank, every dimension, the dtype string, and
   the C-order bytes are hashed together, so `(2, 4)` and `(4, 2)` frames with
   identical bytes cannot share a digest.
3. **Deterministic session digest.** Canonical JSON (sorted keys, compact
   separators, `allow_nan=False`) over the record minus `session_sha256`; the
   constructor recomputes and verifies it, or computes it when omitted. Frame
   order is significant, and camera/local-camera maps must key every frame
   exactly once in frame order.
4. **Explicit schema/version.** A missing, non-exact-integer, or non-`1` version
   is refused as `SESSION_SCHEMA_UNSUPPORTED`, never guessed.
5. **Preserved unresolved facts (exact canonical notes).** `scale_known is
   False`, every `CameraPose.position is None`, and every frame must carry the
   three verbatim `to_camera_solution` notes exactly once each, with no other
   frame-prefixed note; renamed or resealed wording is refused as
   `SESSION_UNRESOLVED_FACT_MISSING`.
6. **Forbidden claims refused.** A record/session claiming scale, a camera
   translation, or a local camera off `LOCAL_FRAME_ORIGIN` is refused as
   `SESSION_FORBIDDEN_CLAIM`; a local camera that is not exactly
   `camera_model_from_estimate(estimate, position=LOCAL_FRAME_ORIGIN)` is
   refused as `SESSION_RECORD_INVALID`.
7. **No partial sessions.** A frame with no solved estimate, an estimate that
   cannot become a camera, an empty mapping, or a non-`CameraSolution` return
   stops the producer atomically.

## Serialization shape

The record's top-level keys are exactly `schema`, `version`, `frames`,
`cameras`, `components`, `unresolved`, `scale_known`, `local_cameras`,
`session_sha256`. Frame records carry exactly `frame_id`, `source_label`,
`image_shape`, `image_sha256`. Camera records carry `frame_id`, `intrinsics`,
`pose` (`position` always `null`), `residual_pixels`, `confidence`. Local-camera
records carry `frame_id`, `position`, `yaw_degrees`, `pitch_degrees`,
`fov_degrees`, `eye_height`, `roll_degrees`, `intrinsics`. No pixel payload of
any kind travels: the record contains no `image`/`pixels`/`array` key and no
numpy array (pinned by a recursive test).

## Deserialization integrity and honest limits

The deserializer checks the exact key sets, schema/version, frame/camera shapes,
sample digests, forbidden claims, retained unresolved facts, local-camera
adaptation coherence, and the deterministic session digest; it rebuilds
`CameraEstimate`/`CameraModel` values from record fields. It **cannot** recompute
an image digest, because pixels are intentionally absent, so each rebuilt frame
keeps `image=None` and `session.has_pixels is False`; nothing is fabricated.

Honest limit: the session digest is self-describing, so a fully resealed record
that recomputes a matching digest is indistinguishable from an original without
an externally retained digest. The semantic invariants (no scale, no
translation, local-frame adaptation coherence, exact-canonical retained
unresolved facts) are enforced independently of the digest and therefore survive
a reseal. In particular, resealing a record with resolved-only or otherwise
non-canonical prefixed notes still fails `SESSION_UNRESOLVED_FACT_MISSING`.

## Tests (45)

`tests/test_image_derived_session.py` (45 cases):

- immutable private copy, digest stability under later mutation, digest framing;
- **repair regressions:** a produced frame refuses `setflags(write=True)` and
  in-place writes; a hostile delegated solver cannot reopen or mutate the frame
  it is handed; post-build frame/session digests, the record digest, and the
  round trip stay valid after caller mutation; a frame refuses a reopenable
  numpy-owned read-only copy;
- RGB and grayscale admission; read-only copies in frame order reach the solver;
- delegation to `solve_image_cameras` and adaptation with an observed
  `position=LOCAL_FRAME_ORIGIN`;
- preserved optics and exact canonical unresolved facts;
- producer refusals: missing per-frame estimate, `scale_known=True`, intrinsics-
  or pose-less estimate, empty mapping, non-mapping, float/4-channel/1-channel/
  zero-size/non-array frames, empty frame id, unknown source label, absent label
  -> `null`, `pathlib.Path` label;
- API surface: only `frames`/`source_paths`, no `**kwargs`; manifest and scene
  loaders are never consulted;
- record key set, no pixel payload, null pose position;
- round trip preserves digests/labels/shapes/cameras/local cameras and reports
  `image=None` / `has_pixels=False`;
- deterministic, order-sensitive digest; `NaN` residual recorded as `null`;
- deserializer refusals (typed): tampered frame digest, resealed `scale_known`,
  claimed pose position, resealed missing unresolved fact, resealed
  resolved-only wording (`frame.png: position`/`scale`/`yaw quadrant`), resealed
  extra / near-miss / duplicate prefixed unresolved note, forged local-camera
  `fov`, forged local-camera position, extra/missing keys, wrong schema/version,
  empty/duplicate frames, non-mapping record, direct-construction immutability
  and digest mismatch, deserialized frame has no pixels.

## Exact commands and outcomes (final code)

All foreground, project root, frozen interpreter `.venv\Scripts\python.exe`,
`-p no:cacheprovider -o addopts=`:

1. `py_compile src\mc_photogrammetry\pipeline\image_derived_session.py tests\test_image_derived_session.py`
   -> exit 0, no output.
2. `pytest tests\test_image_derived_session.py -q` -> **45 passed in 0.24s**,
   exit 0.
3. `pytest tests\test_image_only_entry.py tests\test_pipeline_contracts.py -q`
   -> **13 passed in 0.21s**, exit 0 (11 image-only-entry + 2 pipeline-contract
   cases, all unchanged).
4. Independent adversarial probe (non-contiguous RGB view, a hostile solver that
   attempts `setflags(write=True)` and an in-place write, a numpy-owned read-only
   frame, and a session digest/round-trip recheck) -> `PROBE OK`, exit 0.
5. Independent Repair-2 probe: a canonical record round-trips, while a resealed
   record whose only notes are `frame.png: position`, `frame.png: scale`, and
   `frame.png: yaw quadrant` is refused with `session_unresolved_fact_missing`
   -> `canonical round trip ok: True` / `resealed resolved-only refused: True`,
   exit 0.

No warnings, errors, skips, or non-zero exits.

`grep` of `image_derived_session.py` for substring matching
(`fact in note` / `in note`) returns no match: the only `startswith` is the
frame-id prefix used to scope a note, and it is combined with exact membership in
the canonical-note list, never a substring test.

## Hashes

| Path | Lines | SHA-256 |
| --- | --- | --- |
| `src/mc_photogrammetry/pipeline/image_derived_session.py` | 880 | `4e842cf3b49c38cd6334254c48ce886f8129a43ec5f2edf87f8676876c3a772a` |
| `tests/test_image_derived_session.py` | 725 | `b917609c41c642ecb964242150430939579371a3052d2d9e8f317c6ff037e971` |

The report's own SHA-256 is recomputed after its final edit and returned in the
run result. `AGENT_COMMS.md` is unmodified by this task.

## Deviations

1. **Source labels are stored as plain strings.** `pathlib.Path` labels are
   normalized with `str(...)` because a serialized label must be a JSON string;
   no path resolution, existence check, or geometry read is performed.
2. **`residual_pixels` non-finite values serialize as `null`.** `median_residual`
   is legitimately `NaN` when a fit had no inliers; `allow_nan=False` forbids
   `NaN`, so the diagnostic is recorded as `null` and round-trips as `None`. No
   measured camera field is affected.
3. **Local-camera adaptation coherence is enforced at construction.** The
   session re-derives each local camera from its estimate rather than trusting
   the carried copy; this is strictly stronger than "carry local camera data"
   and closes a resealed forged-focal gap.
4. **`SESSION_*` reason constants are public.** They are the typed refusal
   interface; the exception body stays a `ValueError` so existing callers keep
   working.
5. **Frame capture is bytes-backed and frame construction refuses reopenable
   buffers.** `_immutable_pixels` replaces the copy + `setflags` capture, and
   `ImageDerivedFrame` rejects a read-only array whose base is numpy-owned (an
   array view or an `owndata` buffer). This is strictly stronger than the
   original "writeable flag is False" check; test fixtures that need a direct
   frame now seal the buffer with the same bytes-backed construction.
6. **Unresolved facts are validated as exact canonical strings.** Each frame must
   carry the three verbatim `to_camera_solution` notes exactly once, and any
   frame-prefixed note that is not canonical (including a resolved-only
   `"<id>: position"`) is refused. This is strictly stronger than the original
   substring check and is required so a resealed record cannot silently drop the
   unresolved wording.

## Unresolved concerns (honest limits; not blockers)

- A fully resealed record whose digest matches is indistinguishable from the
  original without an externally retained digest (documented above).
- Image digests cannot be re-verified after deserialization because pixels are
  not serialized; they are identity labels bound into the session digest.
- The serializer writes a `Mapping` only; it does not persist to disk, so no
  file-format or atomic-write contract is claimed at this seam.
- The producer's real-solver path (unpatched `solve_image_cameras`) is exercised
  by the existing `tests/test_image_only_entry.py`; this seam's own suite
  monkeypatches the solver, as the adjacent DS-9 suite does.

## Blocker

None. No failing check, hash mismatch, source-scope violation, or architectural
inconsistency was found. CLI/UI wiring, export, `BlockVolume` construction, and
state fusion remain untouched and unauthorized here. Stop for manager/Astra
review.

## Files written

- `src/mc_photogrammetry/pipeline/image_derived_session.py`
- `tests/test_image_derived_session.py`
- `docs/reports/M4-image-derived-session-2026-09-19.md` (this file)
