# M4 V3 supported-volume bridge — source-bound repair + final adapter hardening (2026-09-19)

Implementation task: `m4-v3-supported-volume-bridge-2026-09-19` (source-bound
repair after Astra rejection, then final adapter hardening after Astra's second
review).  Worker: Sol-equivalent CLI worker (DeepSeek), bounded implementation.

Upstream design (accepted): `docs/reports/M4-placement-volume-bridge-contract-2026-09-19.md`.
Upstream prerequisites: accepted V1 (`plane_region_support.py`) and accepted V2
(`plane_placement.py`).  Both are consumed read-only and their hashes are
unchanged (see §5).

Status: **source-bound repair + final adapter hardening complete, all required
checks pass, stopped for Astra review.**

Write scope used (exactly the three routed paths):

- `src/mc_photogrammetry/reconstruction/plane_volume.py`
- `tests/test_plane_volume.py`
- `docs/reports/M4-V3-supported-volume-bridge-2026-09-19.md` (this file)

No other source, test, report, model, store, exporter, V1/V2, manifest,
handoff, comms, CLI/UI, persistence or `.litematic` file was created or edited.
No git stage/commit/reset/clean.  `py_compile`/`pytest` refreshed standard
`__pycache__` bytecode caches only.

## 1. Astra rejection repaired

Astra rejected the prior V3 implementation and required a source-bound replay
rather than a self-consistent structural replay.  The repair addresses every
item:

1. **`build_plane_volume` requires the original replay context.**  The signature
   is now

   ```text
   build_plane_volume(*, universe, supports, placement, placement_budget,
                      budget, producer_version=...)
   ```

   It reruns `plane_placement.resolve_face_placement(universe=original,
   supports=original, budget=placement_budget)` and compares the canonical
   placement records and digests, plus every face and support record/digest,
   **before** any volume budget decision.  A freshly resealed `+100` normal
   translation that keeps the original bindings replays its own fields but not
   the source records, so it is refused (`replay_mismatch`, no work, no cells).
   A producer refusal (for example a placement budget exhaustion) is surfaced as
   the new typed `placement_replay_refused` refusal, still before any volume
   budget decision.

2. **Each face and support is cross-checked against original source records and
   digests.**  `_cross_check_sources` re-binds every candidate face to its
   source-derived face (`face_sha256` and full `to_record()`), re-binds every
   support digest to the candidate face and the source face, and re-checks the
   support's reading identity, original region-mask/region-id/image-shape
   evidence, accepted phase-anchor digest, and usable-mask presence.  Missing
   and unreferenced supports are refused on provenance.  The candidate supports
   are additionally replayed structurally (`_replay_support`), and the rerun
   itself rebuilds each support byte-for-byte from the original reading,
   alternative, calibration, usable mask and support budget.

3. **`to_geometry_only_preview` requires the same source replay context.**  It
   now takes `universe`, `supports`, `placement`, `placement_budget` and
   `budget`, reruns the accepted source replay, compares the supplied result with
   the authoritative full rebuild, and derives and verifies the cell bounds,
   normalization offset and dense-size limit.  A modified result with a phantom
   coordinate, a stale digest or a resealed placement is refused before any
   preview record is built.  The exact validation/budget ordering was tightened
   again in §1a item 8.

4. **Conflict rule replaced.**  Compatible perpendicular corner claims and the
   two opposite faces of one cube now merge with **no** `CellConflict`.  The
   recipe produces an empty, valid conflict tuple.  `CellConflict` is preserved
   as a validated, exported type but is reserved for explicitly incompatible
   evidence: its constructor refuses to record a conflict when every claim
   bounds the merged cell, so a conflict cannot be fabricated from compatible
   geometry.  No incompatible recipe conflict was invented.

5. **Preview is immutable and exposes no mutable `BlockVolume`.**  The preview
   stores an immutable tuple of typed `PreviewBlock` records; it has no `volume`
   field.  `GeometryOnlyPreview.to_block_volume()` returns a fresh independent
   mutable copy for an exporter, so mutating that copy cannot change the preview
   record or its digest.  `exact_state_eligible` stays hard `False` and
   `complete` stays `False`.

### 1a. Final adapter hardening (second Astra review)

Astra accepted the source-bound replay and conflict repairs but found three
remaining adapter defects, all fixed within the same three files:

6. **`preview_state` was retained by reference.**  A list-backed
   `BlockState.properties` (for example
   `BlockState("minecraft:stone", [["facing", "north"]])`) aliased the sealed
   preview and its exported observations, so mutating the caller's list changed
   `to_record()`/`to_block_volume()` while `preview_sha256` stayed unchanged.
   `to_geometry_only_preview` now canonicalizes the state through
   `_canonical_preview_state` into an immutable `preview_state_name` plus a tuple
   of property pairs, and `GeometryOnlyPreview` stores only that snapshot.  A
   genuinely malformed nested shape (`properties=None`, a bare string, a
   one-element pair, an empty or non-string key/value, a namespaced-less name)
   is refused as `preview_state_invalid`.  `to_block_volume()` rebuilds a fresh
   `BlockState` from the snapshot on every call.
7. **Malformed result data could raise `AttributeError`.**  A clone such as
   `cells=(None,)` reached `result.to_record()` and escaped the API as an
   `AttributeError`.  The preview now validates the supplied result's nested
   records against the authoritative rebuild inside a typed guard; any
   serialization or comparison failure becomes a typed `replay_mismatch`
   (`work is None`, no cells), never an exception.
8. **A tiny volume budget pre-empted replay validation.**  `to_geometry_only_preview`
   previously called `build_plane_volume` first, so a phantom-coordinate result
   plus a tiny budget returned `budget_exhausted` before the result was checked.
   The preview now derives the authoritative *full* record set once with an
   unlimited `_RecordingCounter` (no budget decision), validates the source
   replay and the nested result data, and only then replays the recorded work
   events through a real `_Counter` for the caller's budget.  A fully
   self-consistent phantom result plus a tiny budget is therefore a typed
   `replay_mismatch`; a valid result plus a tiny budget still returns the exact
   builder budget refusal (same reason, detail, work record and refusal digest).

## 2. Load-bearing semantics kept

- **One adjacent block per supported face cell.**  For every V1 `supported` cell
  of a resolved face, exactly one block is emitted at the face's `normal_block`
  and the cell's exact integer lower tangent endpoints.  No extrusion, shell,
  gap or bounding-box fill; an `unknown` cell is omitted as a typed
  `OmittedCell`, never as air.
- **Compatible deterministic merge.**  Contributions on the same coordinate merge
  into one `SupportedCellObservation` retaining every claim, real frame ID,
  reading identity, support digest and face digest.
- **Exact geometry.**  The recipe keeps **66 contributions over 58 occupied
  coordinates**, **15 unknown omissions**, bounds `(-4,0,-4)..(2,3,3)`, dense
  size 224, both outward signs, six reflected tangents, all three world axes,
  signed/fractional translations and negative coordinates.
- **Truthful atomic budgets.**  Every input, provenance and replay check finishes
  before any counter runs; a cap excess or the dense preflight discards every
  partial cell and carries only the exact observed counter.  A tampered record
  can never be reported as `budget_exhausted`.
- **Determinism.**  Support permutation leaves the canonical digest and record
  byte-identical.
- **Isolation.**  The module imports only `__future__`, `dataclasses`,
  `fractions`, `hashlib`, `json`, `math`, `typing`,
  `mc_photogrammetry.domain.models` and the two accepted reconstruction modules;
  no export, CLI, UI, project-store, manual-metadata, filesystem, process or
  dynamic-import facility.
- `complete=False`, `export_bounds=None`, `exact_state_eligible=False`, whole
  build integer translation gauge unresolved and carried through.

## 3. Fixture and evidence

`tests/test_plane_volume.py` reuses the genuine accepted recipe fixture from
`tests/test_plane_placement.py` through those public/test APIs only
(`_build_recipe_fixture`, `_recipe_universe`, `_recipe_supports`, `_budget`, and
the literal `_literal_recipe_*` helpers).  It builds no certificate and reseals
no genuine record.  Expected counts, coordinates and bounds are declared from the
recipe construction, never read back from `plane_volume`.

Coverage added or strengthened by this repair and the final adapter hardening:

- the literal 58 coordinates / 66 contributions / 15 omissions are emitted;
- the seven merged coordinates (`2,2,2,3,2,2,2` claims) are all compatible
  perpendicular corner claims with **zero** conflicts, and every claim is
  checked with `_claim_bounds_cell`;
- record-level controls prove perpendicular corner claims and the two opposite
  faces of one cube are compatible evidence that merge with no conflict, and
  that `CellConflict` refuses to record a conflict for either;
- the recipe's minimal conflict cap (`max_conflicts=1`) is valid with an empty
  conflict tuple;
- a self-consistent freshly resealed `+100` normal translation (with its own
  recomputed placement digest) is refused with `replay_mismatch`, no work and no
  cells; the same reseal under the original stale placement digest is refused;
- a resealed support cell, resealed face cell and resealed placement record are
  refused; missing and duplicate supports are refused on provenance; a placement
  budget exhaustion is refused as `placement_replay_refused` before the volume
  budget;
- every work cap is exact at its boundary and atomic one below it;
  `max_dense_bounds_cells=223` yields the typed `dense_bounds_exceeded` refusal
  with observed 224 and no cells;
- the preview refuses a stale result digest and a freshly resealed phantom
  coordinate, refuses a resealed placement, requires the literal
  `geometry_only=True` flag, refuses non-states and every air state, exposes no
  `volume` field, keeps its record stable across mutation of a returned copy,
  and marks every produced observation inferred at zero confidence;
- **final adapter hardening**: a list-backed `BlockState` is snapshotted so
  mutating the caller's list changes neither `preview.to_record()` nor any
  exported observation state; nine malformed nested-state shapes (including
  list-backed, wrong arity, empty/non-string pairs and a namespaced-less name)
  return `preview_state_invalid`, and `GeometryOnlyPreview.build` itself rejects
  list-backed properties;
- a **fully self-consistent** phantom result (all work counters, bounds, dense
  size, normalization and binding counts recomputed so it passes its own
  constructor) is refused as `replay_mismatch` under tiny `max_faces`,
  `max_contributions` and `max_dense_bounds_cells` budgets, with no work and no
  cells — never `budget_exhausted`;
- malformed nested result data (`cells=(None,)`, a non-record cell, an appended
  `None`) is a typed `replay_mismatch` rather than an escaped `AttributeError`;
- a valid result under each tiny budget returns a preview refusal whose reason,
  detail, work record and full record are byte-identical to
  `build_plane_volume`'s refusal;
- the real `LitematicExporter` exports the preview's fresh volume with
  `bounds=None` to the derived `(7,4,8)` region and 58 total blocks;
- an AST scan proves the source imports stay inside the boundary and contain no
  export/CLI/manual symbol.

## 4. Exact commands and outcomes

Frozen interpreter `.venv\Scripts\python.exe`, project root, foreground/recorded:

1. `python -m py_compile src\mc_photogrammetry\reconstruction\plane_volume.py tests\test_plane_volume.py`
   → exit 0, no output.
2. `pytest tests\test_plane_volume.py -o addopts= -q --no-header -p no:cacheprovider`
   → **51 passed in 115.11s**, exit 0.
3. Focused V3 + placement + region + litematic + store:
   `pytest tests\test_plane_volume.py tests\test_plane_placement.py tests\test_plane_region_support.py tests\test_litematic.py tests\test_project_store.py -o addopts= -q --no-header -p no:cacheprovider`
   → **458 passed in 138.74s**, exit 0.

No failures, errors or non-zero exits.

## 5. Scope and hashes

Written by this task:

| Path | Lines | SHA-256 |
| --- | ---: | --- |
| `src/mc_photogrammetry/reconstruction/plane_volume.py` | 1992 | `872b6e5317f7cf0f2c729ccb57f0ad8c52368290397facb0eb34b37eed25a7bd` |
| `tests/test_plane_volume.py` | 1045 | `449237002e09534afcc185c98a94550ba0cb53d96018333c8ca0efe1ae26f604` |
| `docs/reports/M4-V3-supported-volume-bridge-2026-09-19.md` | — | recomputed after the final edit and reported in the run result |

Upstream records consumed unchanged (recomputed; all still match, so no upstream
module was touched):

| Path | SHA-256 |
| --- | --- |
| `src/.../plane_placement.py` | `76153fda373cd5f7eaf148b8b67daab8a4171e5978d82ca2560230a0578c5847` |
| `src/.../plane_region_support.py` | `45a5b02947e3f95ee5d160334b5172e3a547b16c5893b9c757f964839d8896e3` |
| `tests/test_plane_placement.py` | `62d8edff0a95c2540ff0b66d4c8fd147d559c7b8ccb4060040aab713521ad05e` |
| `src/.../domain/models.py` | `0450a40b9df7f26066fc78d985c07f7c65bb0e37ae00eee38448a15fa83b6740` |
| `src/.../export/litematic.py` | `1917451e46ccb451dcbf2e5194161edfbea0862dc55e62489fcdeeb14f3db794` |

## 6. Deviations and honest limits

1. **`CellConflict` is intentionally unreachable from replayed geometry.**  A
   validated `CellClaim` always bounds its emitted cell, so no group of valid
   replayed claims can be explicitly incompatible.  The type is preserved,
   exported and validated, but its constructor refuses compatible claims, so the
   recipe's conflict tuple is a valid empty tuple and no incompatible conflict
   was fabricated.  A future explicit-incompatibility evidence source would be
   the only producer of a non-empty conflict tuple.
2. **Cost.**  The required rerun of `resolve_face_placement` makes each
   `build_plane_volume` call revalidate the whole accepted V2/V1 evidence graph;
   the V3 suite is correspondingly slower (51 tests / ~116 s).  This is the price
   of the source-bound replay and is not cached, precisely so an in-place
   reseal cannot be served from a stale result.
3. **Preview state source.**  The adapter still takes an explicit caller
   `preview_state`; it is canonicalized into an immutable name+tuple-properties
   snapshot, every produced observation is forced to inferred, zero-confidence
   and `exact_state_eligible=False`, and air states are refused.
4. **Preview budget independence.**  To validate the supplied result before any
   budget decision, `to_geometry_only_preview` derives the authoritative *full*
   record set once with an unlimited `_RecordingCounter` and then replays the
   recorded work events through the caller's `_Counter`.  A preview call
   therefore performs the whole cell derivation even when the caller's budget is
   tiny.  That work is bounded by the same source replay
   (`_cross_check_sources` already visits every face and support cell) and by the
   caller's own `placement_budget`; it is a deliberate ordering choice, not a
   relaxation of `build_plane_volume`, which still stops on the first cap excess.

## 7. Blocker

None at the implementation level.  All required checks pass, no upstream
accepted hash changed, and the module stays isolated (no CLI/UI, manual
metadata, persistence, export or litematic wiring).  The three second-review
adapter defects (aliased list-backed preview state, escaping `AttributeError`
on malformed result data, and `budget_exhausted` pre-empting replay validation)
are all closed with typed refusals and new regressions.  Production promotion
and M5 material fusion remain separately gated.

Stop for Astra review.

## Files written

- `src/mc_photogrammetry/reconstruction/plane_volume.py`
- `tests/test_plane_volume.py`
- `docs/reports/M4-V3-supported-volume-bridge-2026-09-19.md` (this file)
