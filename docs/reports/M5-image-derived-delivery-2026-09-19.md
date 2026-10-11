# M5 image-derived litematic delivery adapter — implementation report

Date: 2026-09-19
Task: bounded image-derived litematic delivery adapter (three-new-path scope)
Role: bounded developer (DeepSeek/Sol-equivalent) under the direct-shell policy in
[`docs/deepseek-worker.md`](../../docs/deepseek-worker.md)
Status: implementation + DS-M5 delivery repair complete; all required checks pass;
**stopped for Sol review**

## Context and scope

This task adds the narrow, bounded, final seam of the image-derived M5 chain: it
turns one accepted, immutable, digest-bound `PromotionResult` into one verified
litematic artifact. It consumes only that record plus an explicit destination
`Path`, an explicit `name`, an explicit `description`, an explicit `overwrite`
flag and an explicit exporter. It never touches CLI/UI, project persistence,
cameras, poses, images, manifests, F3 references or manual metadata, and it never
edits an upstream producer. Exactly three new paths were added, and nothing else
was created or edited:

- `src/mc_photogrammetry/pipeline/image_derived_delivery.py` (new)
- `tests/test_image_derived_delivery.py` (new)
- `docs/reports/M5-image-derived-delivery-2026-09-19.md` (this file)

`AGENT_COMMS.md` was not modified. No commit, stage, reset or clean was performed.
Unlike the upstream M5 promotion adapter, this seam is *allowed* to import the
accepted exporter: it imports only `dataclasses`, `hashlib`, `json`, `math`, `os`,
`re`, `tempfile`, `pathlib`, `typing`, `mc_photogrammetry.export.litematic`,
`mc_photogrammetry.reconstruction` (for `state_promotion`) and a deferred
`nbtlib` inside the artifact reader. The test suite asserts that boundary by
import AST, by code-symbol inspection and by source-text inspection.

## Contract

### Entry point

`deliver_image_derived_litematic(promotion_result, destination, *,
name, description, overwrite=False, exporter=None)` returns **either** an
immutable `DeliveryResult` **or** an immutable `DeliveryRefusal`; it never raises
for malformed input and never returns a partial artifact.

A second public function, `verify_image_derived_litematic(promotion_result,
artifact)`, re-validates an already-written artifact against the promotion that
produced it and returns `(artifact_sha256, block_count)` or a typed refusal. It is
the same verification the delivery uses internally, exposed so an independent
reviewer can re-check a delivered file without re-exporting it.

### Input discipline (before any filesystem work)

Every check below runs before the destination directory is created and before the
temporary file exists:

- `promotion_result` must be an accepted `state_promotion.PromotionResult` - a
  plain `PromotionRefusal` (a review, an omitted-unknown default refusal, an
  exhausted budget), a mapping or any other object is a typed `invalid_input`.
- The schema and producer identity must be the accepted literals, the gauge state
  and axes must be the accepted M4 gauge, `scale_state` and `normalization_state`
  must still be the explicit `"unresolved"` literals, and `export_bounds` must
  still be `None`.
- `export_ready` must be literally `True`, `exact_state_eligible` must be
  literally `True`, `omitted_unknown` must be exactly `0`, and the cell tuple must
  be nonempty.
- Every nested block is rebuilt through `PromotedBlock.build`, its
  `block_sha256`/`to_record` must replay, and the cells must be unique and
  canonically ordered. The enclosing payload is then replayed through
  `state_promotion._result_payload` and the record is rebuilt through
  `PromotionResult.build`, so a constructor-bypassed or resealed record (a forged
  nested field, a swapped block under a stale top-level digest, a non-canonical
  order) is a typed `invalid_promotion` rather than an export.
- The exporter's dense region is preflighted against `DELIVERY_MAX_DENSE_CELLS`
  (4,194,304 cells), so an astronomically sparse promotion is refused *before*
  the exporter would allocate its dense palette array.

### Destination, name and description

`destination` must be a `pathlib.Path` with a name, a parent that is a directory
(or absent), and must not be an existing directory. An existing file at the
destination is a typed `destination_exists` unless `overwrite` is the literal
`True`. `overwrite` must be a literal `bool`. `name` and `description` must be
non-empty, bounded printable-ASCII `str` values (128 and 512 characters).

The promotion is validated **before** the destination and metadata, so a tampered
record is named as such even when the destination also conflicts; this precedence
is deliberate and covered by a test.

### Typed outcomes

`DELIVERY_REFUSAL_REASONS = ("invalid_input", "invalid_promotion",
"promotion_not_export_ready", "promotion_not_exact_state_eligible",
"promotion_omitted_unknown", "promotion_empty", "invalid_destination",
"invalid_name", "invalid_description", "destination_exists", "export_failed",
"verify_failed")`. A `DeliveryRefusal` is a frozen record with a canonical
`detail`, a `detail_sha256` and a self-replaying `refusal_sha256`; its constructor
refuses any `detail` carrying `cells`, `blocks` or `artifact`, so a refusal can
never smuggle a partial artifact.

### Atomicity and cleanup

The accepted `LitematicExporter.export` is called exactly once, with
`bounds=None` and a **fresh** `promotion_result.to_block_volume()` volume, writing
to a same-directory temporary file created by `tempfile.mkstemp`. The temporary is
loaded back and verified; only then is it moved onto the destination with
`os.replace` (atomic on Windows and POSIX). A refusal at any point - exporter
error, unreadable artifact, block/state mismatch, empty artifact - unlinks the
temporary in a `finally` branch and leaves the destination exactly as it was
(including a pre-existing file that the caller opted to overwrite). A test asserts
that no temporary survives any refusal.

Every **filesystem placement** step is also a typed refusal with the same
cleanup guarantee, so no `OSError` can escape the seam (repair below):
`destination.parent.mkdir(parents=True, exist_ok=True)` is wrapped and returns an
`export_failed` refusal (`field="destination_parent"`); `tempfile.mkstemp` /
`os.close` failures return `export_failed` (`field="temporary"`) and the
partially-created temporary is removed if the descriptor close fails; the
`temporary.stat().st_size` read stays a `verify_failed` refusal; and
`os.replace` is wrapped so a locked/inaccessible destination returns
`export_failed` (`field="placement"`) while the `finally` branch removes the
temporary. Because the swap itself failed, a pre-existing destination is never
truncated or unlinked, so `overwrite=True` preserves the original bytes exactly
when the new artifact cannot be placed. The top-level entry point also carries an
`OSError` backstop that converts any unexpected escaped filesystem error into the
same typed `export_failed` (`field="placement"`) refusal instead of raising.

### Result record and delivery digest

`DeliveryResult` is a frozen slotted record whose only fields are
`promotion_sha256`, `artifact_sha256`, `block_count`, `bytes_written` and
`output_path`; `output_path` is the single runtime-only field (excluded from
`repr` and equality) and `to_record()` omits it entirely, so the serializable
record is destination-independent by construction. The record retains no
reference to the promotion, any `PromotedBlock`, the exporter or a mutable
`BlockVolume`; a test walks every field and asserts that.

`artifact_sha256` is **not** the file's byte hash and **not** a hash of the gzip
container. After the artifact is loaded back, its tight region size and exact
coordinate/state pairs are compared against the promotion's fresh
`to_block_volume()` blocks; only then is the digest computed over
`{promotion_sha256, sorted blocks, block_count}`. The exporter's wall-clock
`TimeCreated`/`TimeModified` metadata and the destination path are excluded, so
two exports of one promotion to two destinations have the same logical digest even
though their bytes differ. `bytes_written` is the one record field that is
filesystem- and timestamp-sensitive; it is the gzip container size and is
documented as not part of the delivery identity.

## DS-M5 delivery repair (Sol findings)

Sol's review found two defects in this seam. Both were reproduced against the
pre-repair source before fixing and are now covered by adversarial regressions; no
behavior outside the three scoped paths changed.

1. **Placement `OSError` escaped the typed refusal and could leave a temporary.**
   `destination_path.parent.mkdir(parents=True, exist_ok=True)` and
   `os.replace(temporary, destination_path)` were called outside the `except
   OSError` handlers, and `OSError` is not in the module's `_ERRORS` tuple, so a
   denied parent creation or a locked destination propagated out of
   `deliver_image_derived_litematic` as a raw exception instead of an atomic
   `DeliveryRefusal`. The `mkstemp`-then-`os.close` path had the same gap: a
   failing close leaked the already-created temporary because the cleanup
   `finally` only covered the post-`mkstemp` region. Fix: a typed `export_failed`
   refusal now wraps parent `mkdir` (`field="destination_parent"`), the whole
   `_temporary_path` helper (removing the created temporary before re-raising
   `os.close` failures), and `os.replace` (`field="placement"`), with the
   pre-existing `finally` unlinking the temporary on every unplaced refusal. The
   placement swap is atomic, so a failed `overwrite=True` placement preserves the
   original destination bytes exactly. A top-level `OSError` backstop guarantees
   the "never raises for malformed or failed input" contract even for an
   unforeseen filesystem error.
2. **A corrupted `BlockStatePalette[0]` could hide non-air cells.** The artifact
   reader treated every palette index 0 cell as air and skipped it, without
   checking that index 0 actually means `minecraft:air`. A hand-written artifact
   with `palette[0] = minecraft:diamond_block` therefore hid a real block in a
   gap cell: the packed `BlockStates` were byte-identical to a genuine delivery,
   the hidden cell contributed nothing to the loaded block map, and the block
   count and every remaining coordinate/state matched, so verification accepted
   it. Fix: `_read_artifact` now rejects a non-empty palette whose index 0 is not
   literally `minecraft:air` (and an empty palette) with a `ValueError`, which
   `verify_image_derived_litematic` converts to a `verify_failed`
   (`field="artifact"`) refusal **before** any exact logical-block comparison.
   The check is deliberately before the cell loop, not after it, so the implicit
   air convention can never mask a hidden block.

Adversarial regressions added (`tests/test_image_derived_delivery.py`):
`test_a_parent_mkdir_failure_is_a_typed_refusal_with_no_temporary`,
`test_a_temporary_creation_failure_is_a_typed_refusal`,
`test_a_temporary_close_failure_removes_the_created_temporary`,
`test_a_replace_failure_is_a_typed_refusal_that_removes_the_temporary`,
`test_a_replace_failure_preserves_an_overwritten_destination`, and
`test_a_non_air_palette_index_zero_cannot_hide_a_gap_cell`. The palette control
writes two artifacts with byte-identical packed `BlockStates` and only the
index-zero name changed: the `minecraft:air` version verifies and the
`minecraft:diamond_block` version refuses. Each regression was confirmed to fail
against the pre-repair source (the palette control accepted the corrupted artifact
and the placement tests raised `PermissionError` or leaked a `.tmp`).

## Files (SHA-256, lowercase; line counts)

```
3ec50b8b934340c4a36eb3f8ebd2b8218ccd56b3471733145bb2f07cc21abfa4   759  src/mc_photogrammetry/pipeline/image_derived_delivery.py
88a64f3a67a0a8ddb317351c024865efc7170c3e100f7f8ed5b41f2a100e046e  1348  tests/test_image_derived_delivery.py
```
`diff-scope`: `git status --porcelain --untracked-files=all` lists exactly the two
new paths above plus this report as new, and
`git diff --stat -- src/mc_photogrammetry/pipeline` is empty, so no existing
source, test, CLI/UI, persistence, camera, pose or exporter path was edited. Every
other modified/untracked path in `git status` is the pre-existing dirty baseline
(for example `AGENT_COMMS.md`, `AGENT_HANDOFF.md`, `cli.py`, `project/store.py`).
`AGENT_COMMS.md` was not modified by this task.

## Tests added

`tests/test_image_derived_delivery.py` — **81 synthetic tests**, all inputs built
in memory. The promotion is a real `promote_plane_states` result over a synthetic
`PlaneVolumeResult` and a synthetic `propose_face_states` result; the only
filesystem objects are pytest's `tmp_path` outputs, which are re-read through the
accepted `nbtlib`/`validation`/`compare` readers.

- happy path: a two-cell promotion exports, exists, is non-empty, loads through
  `validate_litematic` and `load_litematic`, and its blocks equal a fresh
  `to_block_volume()` block-for-block;
- the explicit `name`/`description` reach the artifact metadata, and the accepted
  exporter/author identity is preserved;
- state properties (a log `axis`) and a tight region whose lower bound is not the
  world origin both round trip, with the negative/positive coordinate recovered by
  the verification reader;
- record contract: 5 fields exactly, frozen, `to_record()` has 4 keys and no
  destination string, no source/exporter/volume reference is retained, and the
  field validator rejects malformed digests/counts/paths;
- overwrite: an existing destination without `overwrite` is a typed refusal that
  leaves both the file bytes and the directory untouched; the opt-in replaces it;
  a directory destination is refused with its contents intact;
- eligibility gates: `export_ready=False`, `exact_state_eligible=False`,
  `omitted_unknown>0`, an empty cell tuple, a present `export_bounds`, a resolved
  `scale_state`, a resolved `normalization_state`, a non-canonical cell order, a
  sparse span over the cap, and a real upstream `allow_omitted_unknown` promotion
  are each a typed refusal with an empty destination directory;
- tampered records: a forged block digest, a nested forged property order resealed
  at the block level, a swapped block under a stale top-level digest, and the
  upstream `PromotionRefusal` review outcome are all refused before any filesystem
  work;
- invalid input: non-promotion values, non-`Path` destinations, a file parent, an
  empty/overlong/non-`str`/control-character name, a malformed description, a
  non-bool `overwrite`, a non-exporter, and a nameless destination;
- exporter failure cleanup: a raising exporter (`LitematicExportError` and an
  unexpected `RuntimeError`), an exporter that writes a non-litematic file, and an
  exporter that writes an empty file all yield typed refusals with no destination
  file and no leftover temporary, and a failed overwrite leaves the original
  bytes intact;
- filesystem placement cleanup (repair): a denied parent `mkdir`, a denied
  `mkstemp`, a failing `os.close` and a failing `os.replace` each yield a typed
  `export_failed` refusal with no leftover temporary, and a failed replace under
  `overwrite=True` preserves the original destination bytes exactly;
- palette index 0 (repair): two artifacts with byte-identical packed
  `BlockStates` and only the index-zero name changed - `minecraft:air` verifies,
  while a non-air `minecraft:diamond_block` gap cell is refused as `verify_failed`
  before any exact block comparison;
- determinism: with `time.time` monkeypatched to two different epochs the artifact
  bytes differ but `artifact_sha256` and `promotion_sha256` match; two different
  destination directories yield the same digest; `verify_image_derived_litematic`
  is repeatable and matches the delivery; the digest changes with the promotion
  digest or the block set; a hand-written artifact with the right shape but the
  wrong state is refused as `verify_failed`;
- isolation boundary: allowed import roots only, no CLI/UI/project-store symbol,
  no `AGENT_COMMS`/`AGENT_HANDOFF`/`argparse`/`sys.argv` text, the resolved
  scale/normalization and `bounds=None` constants, the accepted exporter is
  actually reused, and the exact both entry-point signatures.

## Checks run

```
$ .\.venv\Scripts\python.exe -m py_compile `
    src\mc_photogrammetry\pipeline\image_derived_delivery.py              # exit 0
$ .\.venv\Scripts\python.exe -m py_compile `
    tests\test_image_derived_delivery.py                                  # exit 0

$ .\.venv\Scripts\python.exe -m pytest tests\test_image_derived_delivery.py `
    -q -p no:cacheprovider -o addopts=                                    # 81 passed (0.60s)

$ .\.venv\Scripts\python.exe -m pytest tests\test_state_promotion.py `
    tests\test_plane_volume.py tests\test_appearance_state_proposals.py `
    tests\test_litematic.py -q -p no:cacheprovider -o addopts=            # 138 passed (2:24)

$ .\.venv\Scripts\python.exe -m pytest tests\test_state_promotion.py `
    tests\test_plane_volume.py tests\test_appearance_state_proposals.py `
    tests\test_litematic.py tests\test_image_derived_delivery.py `
    -q -p no:cacheprovider -o addopts=                                    # 219 passed (2:24)
```

The pre-requirement `state_promotion`/`plane_volume`/`appearance`/`litematic`
run (138 passed) and the repaired delivery suite (81 passed) are both green, and
the combined five-suite run is green. No failure was observed in any accepted
suite. The repair was also validated by temporarily reverting each fix: the three
placement regressions then raised `PermissionError` (or, with the backstop,
reported the wrong refusal field), the close-failure regression left a `.tmp`
behind, and the palette control accepted the corrupted artifact; restoring the
fixes returned all 81 tests to green.

## Deviations, limits and uncertainties

- **The exporter is imported here, and only here.** The task explicitly requires
  calling the accepted `LitematicExporter.export`, so this seam is the one M5 path
  that may import `mc_photogrammetry.export.litematic`. It imports nothing else
  from `export` (no `compare`, `validation`, `anvil` or `attribution`) and never
  edits the exporter.
- **An injectable exporter for tests.** `exporter=None` means "use the accepted
  `LitematicExporter()"`; a caller may pass an `LitematicExporter` subclass to
  simulate a failing exporter. A non-`LitematicExporter` value is a typed
  `invalid_input`. This keyword is the one API element not spelled out verbatim in
  the task; it is flagged for Sol.
- **`DELIVERY_MAX_DENSE_CELLS` is an added explicit cap.** The accepted exporter
  materializes one palette index per cell of the tight region, so a promotion with
  a 10^6 block gap would otherwise abort inside `pack_palette_indices` after the
  destination directory existed. Refusing it as an `invalid_promotion`
  `dense_bounds` refusal is what preserves the required "no filesystem work before
  validation" property for *every* input; a test pins both the over-cap refusal and
  the at-cap delivery.
- **`artifact_sha256` is a logical digest, not a byte digest.** This is required by
  the task ("avoid timestamps causing digest instability") and matches the observed
  exporter behavior (`TimeCreated`/`TimeModified` are wall-clock). Consequently
  `bytes_written` is *not* destination-independent, because gzip output length
  varies with the compressed timestamp; only `bytes_written` varies, and it is
  excluded from the serializable identity by `to_record()`. If Sol prefers a
  byte-level digest as a second field, that is an additive change, not a repair.
- **Verification re-validates the promotion.** `verify_image_derived_litematic`
  re-runs the full promotion validation before reading the artifact, so a caller
  cannot verify against a forged source. This is defence in depth, not a second
  source of truth.
- **No upstream source replay.** As with the state-promotion adapter, this seam
  cannot rerun the geometry/proposal producers: no universe, placement, supports or
  observations are in scope. It binds `promotion_sha256` and re-validates every
  nested digest; a self-consistent record produced by a resealed alternative
  promotion call is outside this seam's trust scope and is bound, not hidden.
- **Litematic coordinates are region-local.** The accepted exporter pins every
  region at `Position=(0,0,0)` and `load_litematic` reports region-local
  coordinates; the verification reader therefore adds the tight lower bound back
  before comparing with the promotion's world coordinates. A test pins this with a
  `(-4, 61, 7)` block.
- **No image-only or fixture evidence was produced or scored.** No occupancy,
  hidden-block, air, terrain, geometry, camera, coordinate, manifest, F3, scale,
  bound or manual metadata is read or inferred. The production M5 pipeline
  integration and its gates remain open; this seam delivers only an explicitly
  supplied, already-accepted promotion.

Stopping here for Sol review and independent verification.
