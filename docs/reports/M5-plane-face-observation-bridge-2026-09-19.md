# M5 plane-face observation bridge — 2026-09-19

Task: `m5-plane-face-observation-bridge-2026-09-19` (resumed under the explicit
plane-homography contract)  
Role: bounded developer (DeepSeek/Sol-equivalent) under the direct-shell policy in
[`docs/deepseek-worker.md`](../../docs/deepseek-worker.md)  
Status: **IMPLEMENTED — three files written, all required checks green, stopped
for Astra/Sol review; production wiring remains blocked**

Follow-up (2026-09-19, post-acceptance cleanup): the lone nonblocking item Sol
recorded — the declared-but-unreachable `slot_mismatch` refusal — is resolved by
removal in this same three-file scope. See **Nonblocking cleanup** below; plane
homography behavior and every accepted-input hash are unchanged.

## Why this attempt could proceed

The first attempt stopped on rectification-contract ambiguity and wrote no source
or test file (report SHA-256 `67C1541315CD863E2084589ECE7C268E7E815F710A56F68537E8EF7449D44AFD`,
matching the `AGENT_COMMS.md` record). Sol then confirmed the stop and named the
contract in `AGENT_COMMS.md` (2026-09-19 — *Sol confirms M5 bridge blocker and
recommends unblock*): plane-homography route; require accepted V3 plus the
original `PlacementUniverse`, `AlignedRegionSupport` records, immutable frame
images and usable masks; rebuild the exact V1 image quads from
support calibration/`origin_uv`/`row`/`column`; coverage is the rectified
intersection of source-region/usable/in-frame pixels; no cube
occupancy/occlusion; one observation per `(cell, slot, frame)`;
`support_frames` are same-face claims only; explicit caps for claims, frames,
rectified pixels, payload bytes and replay checks with typed refusals.

This attempt implements exactly that contract. No source field had to be
fabricated; the one field that cannot be recomputed is called out below and is
bound as rigorously as the records allow.

## What was implemented

### Scope

Only the three permitted paths were written:

* `src/mc_photogrammetry/reconstruction/plane_face_observation.py` (new)
* `tests/test_plane_face_observation.py` (new)
* this report

No V1/V2/V3/domain/models/store/export/CLI/UI/manifest file was edited. Accepted
module and test hashes are byte-identical to the ones recorded at the previous
stop (see **Files**).

### Public API

```python
build_plane_face_observations(
    result,                       # accepted V3 PlaneVolumeResult
    *,
    universe,                     # original PlacementUniverse
    supports,                     # original AlignedRegionSupport records
    placement,                    # original ResolvedFacePlacement
    placement_budget,             # original PlacementBudget
    volume_budget,                # original PlaneVolumeBudget
    frames,                       # FrameRectificationContext per frame
    budget,                       # PlaneFaceObservationBudget
    producer_version=...,
) -> PlaneFaceObservationResult | PlaneFaceObservationRefusal
```

Supporting public records: `FrameRectificationContext`,
`ReadingRectification`, `FaceObservationBinding`, `FaceObservationReview`,
`PlaneFaceObservationBudget`, `PlaneFaceObservationWork`,
`PlaneFaceObservationRefusal`, `PlaneFaceObservationResult`,
`frame_context_from_readings`, `slot_of_normal`, `face_observation_sha256`,
`face_observation_sort_key`, `canonical_face_observation_order`.

`FrameRectificationContext` retains a read-only, bytes-backed copy of the frame
image (canonicalised to `float32`) and of the usable mask, the upstream image
digest the supports declare, a recomputed pixel digest, a recomputed
`plane_region_support.usable_mask_sha256`, and one typed
`ReadingRectification` per reading axis. `frame_context_from_readings` is the
sanctioned derivation path: it copies each original `PlaneReading`'s own
`PlaneRectification`, never a re-measured or invented one. The bridge re-derives
each rectification from the original universe anyway, so a hand-built context
that merely claims this provenance is refused (`frame_context_mismatch`).

A frame context must cover **exactly** the reading axes the accepted result uses:
an extra rectification is `frame_context_mismatch`
(`unreferenced_rectification`) and a missing one is `missing_rectification`.

### Source-bound V3 replay

`build_plane_face_observations` reruns `plane_volume.build_plane_volume` from the
original universe, supports, placement and both budgets, then requires the
supplied result to match the rebuilt result by `volume_sha256` **and** canonical
record, before any bridge budget decision. A non-V3 object is
`non_volume_input`; a builder refusal is `volume_replay_refused`; a resealed,
phantom or stale-digest result is `replay_mismatch`. The bridge works only from
the rebuilt authoritative result.

### Plane homography, quad and coverage

For each claim, the image quad is rebuilt from the bound support's exact-rational
`origin_uv`, the support calibration (which must equal
`plane_region_support._calibration_from_reading(original_reading)`), the claim's
`row`/`column` and the retained `alternative.distance_blocks`, using V1's own
`_exact_homography` / `_project`:

```
u_start = origin_u + column,  v_start = origin_v + row
corners = (u,v), (u+1,v), (u+1,v+1), (u,v+1)   projected by the exact homography
```

Rectification uses the accepted `face_sampling.rectify_face_valid` helper (and
`face_sampling.quad_area` for the degenerate check) at
`face_sampling.PATCH_SIZE`. **No cube occupancy, no
`sample_visible_faces`, no depth buffer and no occluder set is consulted.**

Coverage is exactly the rectified intersection of the decoded source-region mask,
the frame's usable mask and the in-frame pixels:

* source-space valid = `region_mask & usable_mask` (both bound to the original
  reading/support); out-of-frame source pixels are outside the array, and the
  accepted rectifier fills them with zero coverage;
* `FaceObservation.valid` / `coverage` are the rectifier's own per-pixel masks
  (coverage clamped to `[0, 1]`);
* `FaceObservation.usable_coverage` is the mean rectified fractional coverage.

### Six slots and slot/normal validation

The outward world normal of a claimed face is `outward_sign * e_{world_normal_axis}`
(consistent with the claim's own `face`/`normal_block` binding). That maps through
a literal six-entry table to the canonical `NORMAL_OF_SLOT` slot
(`(0,+1) east, (0,-1) west, (1,+1) up, (1,-1) down, (2,+1) south, (2,-1) north`).
The table is asserted total and injective at import, enforced per claim when
grouping, re-validated in `FaceObservationBinding.__post_init__` (a hand-built
binding whose slot disagrees with its outward geometry is rejected there with
`ValueError`), and is present in `FaceObservation.normal`.

### Nonblocking cleanup: unreachable `slot_mismatch` removed

Sol's acceptance recorded one nonblocking follow-up: `slot_mismatch` was declared
in the closed refusal vocabulary and promised in the module docstring, but was
effectively unreachable after the source-bound V3 replay. This cleanup resolves
it by **removal**, which is the contract-correct choice:

* the accepted plane-homography contract has no independent slot input — the
  single observation slot is a pure total function of the claim's validated
  `world_normal_axis`/`outward_sign`;
* the source-bound V3 replay rebuilds and byte-compares the authoritative
  `PlaneVolumeResult` before any bridge decision, and `CellClaim.__post_init__`
  already enforces `world_normal_axis in (0,1,2)` and `outward_sign in (-1,1)`, so
  the six-entry table can never miss;
* there is therefore no reachable request that could emit `slot_mismatch`, and
  inventing a comparison input to make it reachable would widen the contract
  rather than test it.

Removed consistently in all three files: the `slot_mismatch` member of
`M5_FACE_OBSERVATION_REFUSAL_REASONS`, the docstring promise, and the report
wording above. The geometry guard itself is preserved: `slot_of_normal` stays
total/injective and `FaceObservationBinding.__post_init__` still rejects a
contradictory slot. A direct regression test
(`test_slot_mismatch_refusal_is_removed_and_slot_geometry_stays_total`) pins the
removal, the six-entry bijection, and the surviving `ValueError` guard. No
emitted observation, refusal reason, review reason, digest or count changed.

### Emitted records

* Exactly one immutable `FaceObservation` per `(cell, physical slot, frame
  claim)`; `support_frames` is exactly the set of frames that hold a claim on
  that same `(cell, slot)`, which the result constructor independently verifies.
* `FaceObservationBinding` per observation carries the V3 claim/reading/support/
  face digests, the slot and outward geometry, the frame-context, rectification,
  pixel and mask digests, and the observation digest.
* Typed `FaceObservationReview` records omit everything else:
  * `unknown_cell` — each V3 omitted unknown cell (15 in the recipe);
  * `conflict_cell` — a V3 cell with an explicit conflict kind;
  * `unsupported_slot` — a `(cell, slot)` with no claim (282 in the recipe);
  * `degenerate_projection`, `no_source_pixels`, `off_frame`, `empty_coverage` —
    per-claim rectification failures (none in the recipe; constructible only
    through non-recipe geometry because V1 only marks a cell `supported` when the
    whole projected cell is inside region, usable and frame).
* No unknown/conflicting/unclaimed cell or slot is silently dropped, and no
  observation with zero usable evidence can be constructed inside the result
  (`usable_coverage > 0`, valid pixels and effective coverage are enforced).

### Budgets, invalid-before-budget, atomicity

`PlaneFaceObservationBudget` has five explicit positive caps:
`max_claims`, `max_frames`, `max_rectified_pixels`, `max_payload_bytes`,
`max_replay_checks`. The work record carries `frames, claims, observations,
reviews, rectified_pixels, payload_bytes, replay_checks`, all of which are
re-derived from the emitted records by `PlaneFaceObservationResult.__post_init__`.
`payload_bytes` is the retained byte count of each observation (float32 patch +
bool valid + float64 coverage, i.e. FaceObservation's own retention);
`replay_checks` is `frames + reading axes + distinct supports`.

Every type, provenance, source-replay, context and slot validation completes
before any budget decision: the authoritative records are derived once with an
unlimited recorder and only then are the recorded work events replayed through
the caller's caps. A cap excess returns a `budget_exhausted` refusal whose detail
is exactly `{cap, limit, observed}`, whose work counter equals `observed`, and
which carries no partial observations/cells/reviews.

## The one non-recomputable source field (no fabrication)

`AlignedRegionSupport.image_sha256` is an **upstream opaque digest**: the bridge
cannot recompute it from pixels, and the genuine recipe's value is itself the
literal `sha256(b"placement-fixture-image")` placeholder, so no pixel-level
binding exists even in principle. Rather than fabricate one, the bridge:

1. requires the context's declared `image_sha256` to equal the digest every bound
   support on that frame carries (cross-record agreement), and
2. retains the actual `float32` pixels and records a recomputed
   `pixels_sha256` in the frame binding, the observation provenance, every
   binding and the result digest, so the exact consumed bytes can never be
   swapped without changing the emitted digests.

Everything else used by the homography is bound exactly and independently:
`origin_uv` (support rationals), `row`/`column` (claim), the distance (support
alternative, cross-checked by V3 against the face), the calibration
(`_calibration_from_reading` of the original reading) and the rectification (the
original reading's own record). The region mask is re-decoded from the original
reading's `PlaneRegionEvidence` (which re-verifies its own mask digest and
source-region id) and compared with the support's `region_mask_sha256` /
`source_region_id`. The original `PlacementUniverse.usable_masks` entry for each
reading is additionally required to digest to the support's own
`usable_mask_sha256`, so the universe's masks are load-bearing at the bridge
boundary as well as through the accepted V2 support rebuild
(`test_tampered_universe_usable_masks_are_refused`).

## Deterministic test fixture provenance

The genuine recipe has no retained screenshot. The tests therefore reconstruct
**only** what the recipe itself builds and discards:

* the frame buffer is exactly `np.random.default_rng(992).random((1024, 1024, 3),
  dtype=np.float32) + 0.1` — the literal array
  `tests/test_plane_placement.py::_signature` creates for every reading and never
  returns;
* the usable mask is exactly `tests/test_plane_placement.py::_mask_for(axis)`,
  the mask the recipe passes to `build_aligned_region_support` and stores in
  `PlacementUniverse.usable_masks`;
* `image_sha256` is the recipe's literal `IMAGE_SHA`.

`test_fixture_image_reproduces_the_recipe_signature` proves the reconstruction:
re-running `lattice_planes.plane_signature` on the rebuilt buffer reproduces the
recipe's own retained `PlaneSignature.interior` and the cells the recipe did not
overwrite with world texture. No camera, screenshot, occupancy model or
occlusion is invented anywhere; `frame_context_from_readings` derives the
rectification context from the original `PlaneReading` records.

## Boundaries

The module imports only `hashlib`, `dataclasses`, `fractions`, `typing`,
`numpy`, `blocks.model_atlas`, `reconstruction.face_sampling`,
`reconstruction.lattice_planes`, `reconstruction.plane_placement`,
`reconstruction.plane_region_support`, `reconstruction.plane_volume` and
`reconstruction.face_observation`. It defines no `BlockVolume`, no `BlockState`,
no state fusion, no export/exporter, no CLI/UI, no manual metadata and no
persistence, and performs no IO. Tests enforce this by AST: allowed imports only,
no `BlockVolume`/`BlockState`/`ObservationKind`/`sample_visible_faces`/`occupancy`
references, no export/CLI/UI import path, and no `open`/`exec`/`eval`/`compile`/
`__import__` calls.

## Checks run

`py_compile` of the new module, the new test and the six accepted modules —
exit 0:

```
.\.venv\Scripts\python.exe -m py_compile `
  src\mc_photogrammetry\reconstruction\plane_face_observation.py `
  src\mc_photogrammetry\reconstruction\plane_volume.py `
  src\mc_photogrammetry\reconstruction\plane_placement.py `
  src\mc_photogrammetry\reconstruction\plane_region_support.py `
  src\mc_photogrammetry\reconstruction\face_observation.py `
  src\mc_photogrammetry\reconstruction\appearance_scoring.py `
  src\mc_photogrammetry\reconstruction\face_sampling.py `
  tests\test_plane_face_observation.py          # exit 0
```

Bridge suite alone (re-run after the `slot_mismatch` cleanup):

```
.\.venv\Scripts\python.exe -m pytest tests\test_plane_face_observation.py `
  -q -p no:cacheprovider -o addopts=
# 24 passed in 77.34s        (was 23 passed in 80.17s before the cleanup test)
```

Post-cleanup `py_compile` of the same eight paths above: exit 0. The one added
test is
`test_slot_mismatch_refusal_is_removed_and_slot_geometry_stays_total`, which pins
the removed reason, the six-entry slot bijection, and the surviving
`FaceObservationBinding` `ValueError` guard. Its emitted recipe record is
byte-identical to the pre-cleanup run (`result_sha256` and `work_sha256` below
are unchanged), so no accepted plane-homography behavior moved.

Required focused gate (bridge + V3 + placement + region + face observation +
appearance scoring), run on the pre-cleanup final artifacts and preserved here as
the accepted baseline (not re-run for this documentation-only cleanup):

```
.\.venv\Scripts\python.exe -m pytest `
  tests\test_plane_face_observation.py tests\test_plane_volume.py `
  tests\test_plane_placement.py tests\test_plane_region_support.py `
  tests\test_face_observation.py tests\test_appearance_scoring.py `
  -q -p no:cacheprovider -o addopts=
# 537 passed in 250.22s
```

The 514 accepted tests are unchanged and green (V3 51, placement 310, region 88,
face observation 49, appearance scoring 16); the 24 bridge tests (23 accepted plus
the cleanup regression) are the remaining total, and the post-cleanup bridge suite
reports 24 passed. V3 recipe invariants therefore still hold: 66 contributions,
58 occupied coordinates, 15 unknown omissions, dense 224, normalization
`(-4, 0, -4)`, `complete=False`.

The new suite covers, independently of the bridge's own records: the literal
`(cell, slot, frame)` identity of all 66 claims; the six-slot/outward-sign
geometry (including 22 reflected-normal claims and 31 negative coordinates);
per-observation coverage recomputed from the original reading/support records
through `_exact_homography`/`_project`/`rectify_face_valid`; immutability of the
retained patch/valid/coverage snapshots; determinism and digest replay under
reversed frame and support order; typed refusals for missing frame context,
unreferenced frame, missing rectification, rectification/mask/image-digest
mismatch, a forged context with a stale digest, non-V3 input, a stale-digest
result and a tampered universe; exact-and-atomic budget boundaries for all five
caps; `budget_exhausted` never masking an invalid request; and
appearance-scoring consumption of the emitted observations.

## Emitted recipe evidence (deterministic)

```
volume_sha256    151af3cb2a77e20ab1f4362d1136b451e67086c3bb4f67e8f99e9c61239d57cc
placement_sha256 3339b16f2cfa70d28fefe1a538757eb60976d7431a237d1cbc68ad3665c84daa
result_sha256    6939b1061c0a83b65ff3a2ea771e4e0f958b964a4637216e52414af37b8ac89e
work_sha256      550da79893f3f2e609c848e23a9962b6eb9c60b17dbc96b5ddb39f6351f53875
work             frames 3, claims 66, observations 66, reviews 297,
                 rectified_pixels 16896, payload_bytes 354816, replay_checks 21
reviews          282 unsupported_slot + 15 unknown_cell, 0 other
slots            all six present; outward signs both present
coverage         min = max = 1.0 for every emitted observation
```

## Files

Created/rewritten (SHA-256, uppercase). Current post-cleanup hash first, with the
accepted pre-cleanup hash in parentheses:

```
D86E64642B8CA6CCB3173D295B247BC06E56D443F345758F29B3FDB2D7364168  src/mc_photogrammetry/reconstruction/plane_face_observation.py
4353B3655ABCE9700A8A0A2A7D4C1877272372A2DFCB8BCE1C1514760495C269  tests/test_plane_face_observation.py
```

Pre-cleanup accepted hashes (recorded in `AGENT_COMMS.md`, superseded by the
`slot_mismatch` removal above):

```
69817BEBA0E0437618974E441B7103AB1E2A5FAEF4514FEACB9F4A61434CEE7C  src/mc_photogrammetry/reconstruction/plane_face_observation.py
881ED76BDB62080BD91E20FD329776E0332FB93E0DD81EDA663190EAFC704D89  tests/test_plane_face_observation.py
```

This report supersedes the earlier stopped version
(`67C1541315CD863E2084589ECE7C268E7E815F710A56F68537E8EF7449D44AFD`); its stop
analysis is preserved in `AGENT_COMMS.md`.

Accepted inputs re-verified byte-identical (unchanged by this task):

```
872B6E5317F7CF0F2C729CCB57F0AD8C52368290397FACB0EB34B37EED25A7BD  src/mc_photogrammetry/reconstruction/plane_volume.py
76153FDA373CD5F7EAF148B8B67DAAB8A4171E5978D82CA2560230A0578C5847  src/mc_photogrammetry/reconstruction/plane_placement.py
45A5B02947E3F95EE5D160334B5172E3A547B16C5893B9C757F964839D8896E3  src/mc_photogrammetry/reconstruction/plane_region_support.py
03710AAEC58D1E2FFFB769174DB34F826C7D8DD830BEDB8D9074BD231BDA8C1E  src/mc_photogrammetry/reconstruction/face_observation.py
84F767FB781BEBB08C39CB45FA0EC41990655B3562DCFA090FF8F4DEFDA9ED02  src/mc_photogrammetry/reconstruction/appearance_scoring.py
8A4C050F6017C28AD4B6AEA92E82F8EABC130B2674B600B46ACBAC91080D2FA5  src/mc_photogrammetry/reconstruction/face_sampling.py
449237002E09534AFCC185C98A94550BA0CB53D96018333C8CA0EFE1AE26F604  tests/test_plane_volume.py
62D8EDFF0A95C2540FF0B66D4C8FD147D559C7B8CCB4060040AAB713521AD05E  tests/test_plane_placement.py
544A6337AA37BCFD3923038AA7B4D4E932C286E0B500D576E1D71297385A1DAD  tests/test_plane_region_support.py
BDE8F159020C320CA61FB01268D4EF9582C76EA78303F10304DB39A630F1C94B  tests/test_face_observation.py
BF71DCB0DB9A5DAEB0FD9355F90DFD71DA7076EA65C2E86AC0F05735104D6FDB  tests/test_appearance_scoring.py
```

## Decisions a reviewer should confirm

1. `unsupported_slot` reviews are emitted for every `(cell, slot)` of an emitted
   V3 cell that has no claim (282 for the recipe), so "unsupported" coverage is
   explicit rather than absent from the record.
2. A frame context must carry exactly the reading axes the accepted result uses.
   Contexts for readings the accepted placement never consumed are refused
   rather than silently bound.
3. `usable_coverage` is the mean rectified fractional coverage; `valid` uses the
   accepted rectifier's `coverage > 0.01` threshold.
4. `payload_bytes` follows `FaceObservation`'s own retention (float32 patch,
   bool valid, float64 coverage).
5. `replay_checks = frames + reading axes + distinct supports`; it is an observed
   counter and also re-derives from the result records.
6. The unreachable `slot_mismatch` refusal is removed rather than tested: the
   accepted contract supplies no independent slot input, V3 replay already
   validates the claim geometry the slot is derived from, and the surviving
   geometry guard is the binding constructor's `ValueError`. Re-adding the reason
   would need a new contract input, not a new test.

## Stop

Stopping for Astra/Sol review. No commit, stage, reset or clean was performed.
No `BlockVolume`, `BlockState`, state fusion, export, CLI/UI, persistence or
manual metadata exists anywhere in the bridge, and production wiring remains
blocked pending acceptance.
