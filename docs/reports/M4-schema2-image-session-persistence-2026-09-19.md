# M4 schema-2 image-derived session persistence (2026-09-19)

Task: bounded production seam requested 2026-09-19 - persist an accepted
image-derived session in `.mcscan` schema 2.
Worker: `deepseek-flash`, bounded implementation.
Status: **implementation complete; focused, adjacent, and full suites pass; stopped
for Manager/Astra review.**

Write scope used (exactly four routed paths):

- `src/mc_photogrammetry/domain/models.py` (edited)
- `src/mc_photogrammetry/project/store.py` (edited)
- `tests/test_project_store.py` (edited)
- `docs/reports/M4-schema2-image-session-persistence-2026-09-19.md` (this file, new)

No CLI/UI/export, reconstruction, `pipeline`, dataset, evaluation, handoff, or
`AGENT_COMMS.md` file was created or edited. `AGENT_COMMS.md` was **read only**
(its repository modification predates this task). No git stage/commit/reset/clean
was run. The only side effect of the required `py_compile`/`pytest` runs is
refreshed standard `__pycache__` bytecode caches.

## Objective

Advance `ScanProject` `.mcscan` persistence to schema 2 while:

- preserving schema-1 load compatibility and re-save behavior;
- making schema 2 explicit in the payload and the default for newly saved
  projects;
- adding one optional image-derived session record field;
- serializing that record only through
  `pipeline.image_derived_session.image_derived_session_to_record` and
  `..._from_record`, with strict validation and typed `ProjectStoreError`
  wrapping;
- never persisting pixels, camera manifests, F3/manual coordinates, directions,
  bounds, scene metadata, or fabricated scale/translation;
- keeping `BlockVolume` round-trip behavior unchanged.

## Layering decision (resolved, not ambiguous)

`pipeline/contracts.py` imports `BlockVolume`/`ScanProject` from
`domain.models`, and `pipeline.image_derived_session` transitively imports
`pipeline.contracts`. A module-level `domain.models -> pipeline.image_derived_session`
import therefore creates a real cycle:

- importing `mc_photogrammetry.pipeline` first would run
  `pipeline.__init__ -> contracts -> domain.models -> image_derived_session ->
  image_only -> contracts` while `contracts` is only partially initialized and
  has not yet defined `CameraEstimate`/`CameraSolution`;
- importing `domain.models` first and adding the same import at module top or
  bottom breaks the other order symmetrically.

Rather than invert the architecture with a lazy/function-local import, the
domain model stays dependency-free:

- `ScanProject.image_derived_session` holds the **canonical JSON record** (the
  output of `image_derived_session_to_record`), or `None`. It is a plain mapping,
  carrying no typed pipeline object and no pixels.
- `project/store.py` is the composition boundary. On **both** save and load it
  calls `image_derived_session_from_record` (the sole validator) followed by
  `image_derived_session_to_record` (canonical re-serialization), and wraps every
  `ImageDerivedSessionError` (a `ValueError`) in `ProjectStoreError`.

This keeps `domain` as the base layer, so no import cycle exists in any order.
It was verified in fresh interpreters for five import orders (see checks).

## Schema shape

Schema-1 payload (unchanged, still emitted when `schema_version == 1`):

```json
{
  "schema_version": 1,
  "name": "...",
  "minecraft_version": "...",
  "source_paths": [...],
  "volume": {"blocks": [...]}
}
```

Schema-2 payload (the default for newly constructed `ScanProject`):

```json
{
  "schema_version": 2,
  "name": "...",
  "minecraft_version": "...",
  "source_paths": [...],
  "volume": {"blocks": [...]},
  "image_derived_session": null
}
```

`image_derived_session`, when present, is exactly the record documented and
produced by `pipeline.image_derived_session` (top-level keys `schema`, `version`,
`frames`, `cameras`, `components`, `unresolved`, `scale_known`, `local_cameras`,
`session_sha256`). No other project key is accepted.

## Validation and refusals

`ScanProject.from_dict`:

- rejects a non-mapping root, a non-integer `schema_version`, and any value other
  than 1 or 2 (`unsupported scan project schema`);
- schema 1: rejects `image_derived_session` explicitly (a legacy payload cannot
  smuggle a session) and rejects any other key outside the five legacy keys;
- schema 2: rejects every key outside the six allowed keys, including the
  forbidden metadata names (`pixels`, `image(s)`, `cameras`, `camera_manifest`,
  `f3`, `coordinates`, `directions`, `bounds`, `scene`, `scene_transform`,
  `scale`, `translation`, `metadata`, `annotations`);
- requires `image_derived_session` to be a mapping or `null`.

`project.store`:

- `save_project` builds the payload with `project.to_dict()`, validates and
  canonicalizes any session record, then writes atomically as before. A refusal
  or a serialization error leaves no file behind.
- `load_project` parses JSON, delegates to `ScanProject.from_dict`, then
  validates and canonicalizes the session record.
- Every project-level or session-level refusal is raised as `ProjectStoreError`
  (a `ValueError` subclass) with the original typed `ImageDerivedSessionError`
  reason chained as `__cause__`.

The session record validator itself is unchanged: it still enforces the exact
record key set, schema/version, per-frame digests, the deterministic session
digest, forbidden scale/translation claims, local-frame adaptation coherence,
and the exact canonical unresolved-fact notes. Nothing was weakened.

## Determinism

For a schema-2 project carrying a canonical record, `save -> load -> save`
produces byte-identical files. `image_derived_session_to_record(from_record(r))`
is canonical, so loading normalizes a valid-but-unnormalized record (for example
JSON-parsed number types) to the same bytes a fresh save would write.

## Tests

`tests/test_project_store.py` now has **30 tests** (the original sparse-volume
round trip is preserved). New coverage:

- schema-2 is the constructor default and is explicit in the payload; a
  session-less schema-2 project round-trips;
- a valid session record round-trips byte-for-byte and equals `to_dict()`;
- the saved record carries the session schema/version, `scale_known is False`,
  a null pose position, no pixel payload of any kind, and serializes with
  `allow_nan=False`;
- legacy schema-1 payloads load with `image_derived_session is None`, and
  re-saving keeps schema 1 with no session key;
- a schema-1 payload or project cannot smuggle a session;
- forbidden project metadata (`pixels`, `cameras`, `camera_manifest`, `f3`,
  `coordinates`, `directions`, `bounds`, `scene`, `scene_transform`, `scale`,
  `translation`) is refused; unknown project keys are refused; a non-mapping
  session field is refused;
- unsupported project schema (3) and unsupported session version (2) are
  refused;
- a tampered session digest is refused on both load and save, and save leaves no
  file; a session claiming scale or a camera translation is refused; a session
  record with unknown nested keys is refused;
- store refusals are typed `ProjectStoreError` (and remain `ValueError`s).

## Exact commands and outcomes (final code)

All foreground, project root, frozen interpreter `.venv\Scripts\python.exe`,
`-p no:cacheprovider -o addopts=`:

1. `py_compile src\mc_photogrammetry\domain\models.py src\mc_photogrammetry\project\store.py tests\test_project_store.py`
   -> exit 0, no output.
2. `pytest tests\test_project_store.py -q` -> **30 passed in 0.34s**, exit 0.
3. `pytest tests\test_domain_models.py tests\test_cli.py tests\test_pipeline_contracts.py tests\test_image_only_entry.py tests\test_image_derived_session.py -q`
   -> **77 passed in 0.45s**, exit 0 (domain, CLI, pipeline-contract, image-only
   entry, and the accepted M4 session suite are all unchanged and green).
4. Full suite `pytest tests -q` -> **2400 passed, 1 skipped in 733.80s
   (0:12:13)**, exit 0.
5. Import-order probe in five fresh interpreters (`pipeline`-first,
   `domain.models`-first, `project.store`-first, `pipeline.contracts`-first, and
   `project.store` import-only) -> `TOTAL-FAILURES 0`, exit 0. No circular import.

No warnings, errors, or skips attributable to this change.

## Hashes

| Path | Lines | SHA-256 |
| --- | --- | --- |
| `src/mc_photogrammetry/domain/models.py` | 261 | `91AC20A6C5687F702373EFC1ED0F3A9B8CD5D453FBBE9EAA991EB7206B3D3B59` |
| `src/mc_photogrammetry/project/store.py` | 77 | `CA3C1DAEFA7A1C4898E45824873E274FBC986A94EC2DC09285394D70E03FE339` |
| `tests/test_project_store.py` | 416 | `03603AB64BCF7617DD2F8180563D067621FDD91EE0EBA88172A01A57AE505C06` |

The report's own SHA-256 is recomputed after its final edit and returned in the
run result. `AGENT_COMMS.md` is unmodified by this task.

## Deviations

1. **The session field holds the canonical record, not an `ImageDerivedSession`
   object.** This is forced by the `pipeline.contracts -> domain.models` import
   direction: storing the typed object would require `domain` to import
   `pipeline`, which is a genuine cycle in at least one import order. The record
   is still produced and validated exclusively through the two mandated
   functions, in `project.store`.
2. **Schema 1 now rejects unknown top-level keys and an `image_derived_session`
   key.** The canonical legacy payload is unchanged and still loads/re-saves.
   This is stricter, not weaker; it prevents a legacy-schema payload from
   smuggling a session.
3. **`save_project` validates before creating the temporary file.** A refused
   session therefore leaves neither the temporary nor the destination file.
4. **`to_dict`/`from_dict` raise `ValueError` for schema/field refusals; the
   store wraps them.** `ScanProject` itself stays free of the persistence error
   type so it can be used without the store.

## Unresolved concerns (honest limits; not blockers)

- The session record is self-describing, so a fully resealed record with a
  matching digest is still indistinguishable from an original without an
  externally retained digest. The session module's semantic invariants (no scale,
  no translation, local-frame adaptation coherence, exact canonical unresolved
  notes) are enforced independently of the digest and survive a reseal, so
  resealing cannot manufacture a forbidden claim.
- Loading a schema-1 project and re-saving it keeps schema 1; this seam does not
  silently migrate legacy files. Migration is out of scope here.
- This seam persists a session record that already exists; it does not wire the
  producer into the normal `reconstruct` entry point, nor does it add CLI/UI
  flags to supply or consume a session.

## Blocker

None. No failing check, hash mismatch, source-scope violation, or architectural
inconsistency was found. CLI/UI wiring, export, state fusion, and litematic
promotion remain untouched and unauthorized here. Stop for Manager/Astra review.

## Files written

- `src/mc_photogrammetry/domain/models.py`
- `src/mc_photogrammetry/project/store.py`
- `tests/test_project_store.py`
- `docs/reports/M4-schema2-image-session-persistence-2026-09-19.md` (this file)
