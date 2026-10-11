# DS-27 - launch-free bulk capture queue

Date: 2026-09-12. Worker: DeepSeek DS-27. Contract source: the DS-27 assignment in
`AGENT_COMMS.md` and the accepted DS-17/20C/21/22 factory.
Scope: the bounded bulk queue described by the assignment. No reconstruction,
Java, evaluator, threshold, existing report/artifact, or git index was touched;
no Minecraft launch, install, arm, or world staging was performed.

## Deliverables

| Path | Purpose |
| --- | --- |
| `src/mc_photogrammetry/dataset/batch.py` (new) | Batch spec, source identity, dedupe/leakage, containment, content-addressed manifest, statuses, atomic resume, command generation |
| `scripts/plan_capture_batch.py` (new) | CLI: dry-run by default, `--materialize`, `--status`/`--commands`/`--mark`/`--resume` |
| `tests/test_capture_batch.py` (new) | 33 focused tests for the required categories |
| `src/mc_photogrammetry/dataset/__init__.py` | Minimal public exports for the batch API |

## Behaviour

**Spec.** A declarative JSON file (`schema_version: 1`) names the read-only
`game_dir`, the output `dataset`, an optional `staging_game_dir` (default
`<dataset>/staging-game`) and `queue_dir` (default `<dataset>/queue`), a default
`seed`/`views`, and a non-empty `tasks` list. Each task names one `save`, a
`declared_split` in train/dev/test, and either a `region` (`name`/`min`/`max`) or a
`landscape` block, with optional `variants` (`variant_id`, `seed`, `views`,
`settings`, `resource_pack`, `shader_pack`). `region` is the only geometry the
queue accepts; `camera`, `f3`, `coords`, `coordinates`, `pose`, `position`,
`target`, `yaw`, `pitch`, `roll` and `rotation` are refused recursively at spec
load and again at plan time (`FORBIDDEN_SPEC_KEYS`). The whole package is
advanced developer data generation and is labelled `advanced_only` with the
usage note "never a normal reconstruction input".

**Identity before variants.** Every selected save is read-only fingerprinted
with `hashing.staging_fingerprint` (SHA-256 over file contents) to get a
`source_identity`, and a salted `build_id` is derived from it. This happens
before any variant is expanded into a scene, so dedupe keys never depend on
camera/seed choices.

**Split deferral.** `declared_split` is a queue grouping only. The exact topology
does not exist until the mod exports ground truth, so the queue does not invent a
split: `manifest_payload["split_authority"]` records that the authoritative
train/dev/test assignment remains the topology-keyed split computed by the
existing `finalize_scene`. The queue refuses the same source content under two
different declared splits, and refuses a source already finalized in
`<dataset>/index.json` under a different split, so one build cannot be requested
across train and evaluation.

**Duplicate/leakage refusal.** `plan_batch` refuses duplicate `scene_id`s,
identical `(source identity, region bounds, variant seed/views/settings)` work,
one build straddling two declared splits, and a `scene_id` that already exists in
the dataset index.

**Path containment.** `dataset`, `staging_game_dir` and `queue_dir` are resolved
and refused if any of them overlaps (equals, contains, or sits under) the game
directory or any selected source save; dataset and staging must differ; staging
must not sit inside `<dataset>/scenes`. Nothing in the module opens a source save
for writing.

**Content-addressed immutable manifest.** `manifest_id` is the digest of the
immutable plan payload (paths, seed, per-scene region/variant/pack records and
the exact prepare/finalize argv). Statuses and attempt evidence live in the same
`manifest.json` but are stripped before hashing, so re-planning the same spec
yields the same id, re-materializing is byte-idempotent, and an existing manifest
with a different id is refused rather than replaced. `load_queue` re-hashes the
immutable payload and refuses a tampered/corrupted manifest, a wrong schema
version, an unknown status, malformed argv, or a repeated scene.

**Commands.** Each scene carries `prepare_argv`
(`prepare_capture_dataset.py --game-dir ... --save ... --dataset ... --dataset-id
... --staging-game-dir <root>/<scene_id> --regions <queue>/regions/<scene_id>.json
--seed ... --views ... --allow-overwrite`) and `finalize_argv`
(`finalize_capture_dataset.py --dataset ... --staging-game-dir <root>/<scene_id>
--scene <scene_id> --allow-overwrite`). Each scene gets its own staging game dir
so scenes of one save cannot collide on a staged copy, and `--allow-overwrite`
makes a retried failed scene idempotent. `scene_id_for` reproduces the factory
script's `_scene_id` exactly and additionally refuses an id the plan frame-id
contract cannot represent.

**Statuses and resume.** Allowed transitions are
`pending -> running -> {complete, failed}` and `failed -> running`; complete is
terminal, unknown statuses and same-status writes are refused, and every
transition appends an attempt record (`attempt`, `status`, `evidence`). All
writes are a temp-file + `os.replace` atomic swap. `resume_queue` converts a
stale `running` scene (only an interrupted process can leave one) into a `failed`
attempt with the reason recorded, reports `actionable`/`complete`, never touches
`complete`, and is a no-op when repeated.

**Packs.** `resource_pack`/`shader_pack` names are recorded as
`*_requested` + `settings_requested`, with `resource_pack_applied` and
`shader_pack_applied` fixed `False` and a `pack_note` saying no runtime switch is
performed and no pack is claimed active.

**CLI.** `--spec` is required. Without `--materialize` the run is a dry run that
writes nothing. `--materialize` writes only the queue directory, `manifest.json`
and the per-scene `regions/*.json` inputs the existing prepare command consumes -
no staged world, no armed plan, no dataset scene, no launch. `--status`,
`--commands`, `--mark SCENE STATUS [--evidence JSON]` and `--resume` read/update
the materialized queue; `--json` emits the raw payload. Console output goes
through the DS-26 encoding-safe writer.

## Evidence

```
.\.venv\Scripts\python.exe -m pytest tests/test_capture_batch.py -o addopts= -p no:cacheprovider -q
-> exit 0, 36 passed in 1.68s   (33 before the narrow review fixes)

.\.venv\Scripts\python.exe -m pytest tests/test_capture_batch.py tests/test_capture_dataset.py \
    tests/test_capture_options.py tests/test_capture_console.py tests/test_litematic.py \
    tests/test_camera.py tests/test_project_store.py -o addopts= -p no:cacheprovider -q
-> exit 0, 147 passed, 1 skipped in 8.28s   (144 before the narrow review fixes)
   (the skip is the pre-existing external-symlink staging case)
```

The 36 new cases cover every required category:

* deterministic idempotence - same spec gives the same `manifest_id` and
  byte-identical re-materialization; changed save content changes the identity;
  a different manifest in the same queue dir is refused;
* corrupted-state refusal - invalid JSON, wrong schema version, unknown status,
  a mutated scene field that no longer hashes to `manifest_id`, a tampered region
  file, and a missing manifest are all refused;
* path containment - dataset inside the game dir, staging inside a source save,
  dataset equal to a save, staging equal to the game dir, and staging inside
  `<dataset>/scenes` are refused;
* duplicate/leakage - duplicate scene ids, identical work under another region
  name, one source under two declared splits, an existing index scene id, and an
  existing index cross-split source are refused;
* status transitions - the legal ladder succeeds with evidence, and illegal
  transitions, unknown statuses, missing scenes, same-status writes and
  non-JSON evidence are refused without a partial write; resume resets stale
  `running` and is repeat-safe;
* Unicode - a CJK `level.dat` `LevelName` round-trips through the manifest and
  renders escaped on a strict cp1252 stream (including one real subprocess CLI
  run with `PYTHONIOENCODING=cp1252`); a CJK save directory is refused with the
  frame-id reason;
* command arguments - generated argv values, per-scene staging dir, absence of
  `--landscape`/camera flags, and a regions file that the existing
  `load_region_requests` parses back to the exact bounds/name/views;
* multi-variant identity - a two-variant explicit-region task materializes two
  distinct scene ids, region files, staging dirs and command sets while sharing
  one build id, identical bounds and one declared split, and the shared build is
  still refused across two declared splits; landscape multi-variant names are
  distinct in the same way;
* mutable attempt history - because `status`/`attempts` are stripped from the
  digest, `load_queue` re-validates ordinals exactly `1..N`, the transition ladder
  rooted at `pending`, and agreement between the scene status and the final
  attempt; gaps, restarts, boolean ordinals, a skipped first transition,
  same-status repeats, an empty non-pending history and status disagreement are
  all refused while the pristine `pending`/zero-attempt and a full legal ladder
  still load.

Additionally: a dry run creates nothing and leaves the source save bytes, mtime
and file set unchanged; a landscape task derives its bounds from the save spawn
(and refuses a save with no `SpawnY`); `--allow-overwrite` retries; region caps
are enforced by the existing `RegionBounds`.

## Narrow review fixes (2026-09-12, second pass)

Three findings from the lead review were fixed in the owned DS-27 paths only.

1. **Multi-variant explicit-region collision.** `_expand_task` only suffixed the
   variant id onto *landscape* region names; two variants of one explicit-region
   task reused the same `region_name`, hence the same `scene_id`, region file and
   staging dir, so a desired camera/settings variant pair was refused as a
   duplicate id. The suffix is now applied to both region sources whenever a task
   declares more than one variant, so every variant gets a deterministic distinct
   `scene_id`/region file/command while the bounds, `source_identity`, `build_id`
   and `declared_split` stay identical for split-leakage grouping. Duplicate
   `variant_id`s inside one task are now refused explicitly (they would otherwise
   collide). Two variants that carry genuinely identical work are still refused.
2. **Unvalidated mutable attempt history.** `status` and `attempts` are stripped
   from the content-addressed digest, so the digest cannot protect them. The new
   `_validate_attempt_history` is now the authority: ordinals must be exactly
   `1..N`, statuses must walk `_ALLOWED_TRANSITIONS` from `pending`, and the
   recorded scene status must equal the final attempt status (the only empty
   history allowed is a pristine `pending` scene). Corrupted sequences and
   status/history disagreement are refused. Attempt ordinals also reject `bool`
   (which is an `int` subclass) instead of silently accepting `true` as `1`.
3. **Non-ASCII docstring.** The only non-ASCII phrase in
   `dataset/console.py` - the docstring example that embedded the character
   between backticks - was replaced with the ASCII phrase `the e-acute
   character`. `console.py` is now byte-pure ASCII; no code or behavior changed.

## Honest limits

* `materialize_queue` writes queue bookkeeping only: `manifest.json` and the
  per-scene `regions/*.json` the manifest's prepare command consumes. Those
  region files are manifest inputs, not dataset artifacts; no world is staged, no
  plan is armed, and no scene is finalized.
* The queue never executes a command it prints. A runner (the lead) advances
  statuses with `--mark`/`--resume`; nothing here can observe a real capture.
* `source_identity` hashes whole save contents (bounded by the existing staging
  caps), so planning a batch of very large saves is I/O-heavy. That cost is the
  price of refusing duplicate/leaking builds before any capture.
* `declared_split` is not the final split and is never written as such; the
  authoritative split still comes from the topology-keyed finalizer.
* A save directory name that cannot form an ASCII plan frame id (for example a
  CJK directory name) is refused at plan time rather than queued into a
  guaranteed `build_plan` failure. A Unicode `LevelName` inside an ASCII-named
  save is preserved as `world_identity`.
* Resource-pack and shader-pack switching remains unimplemented; the fields are
  requested/provenance only.
* No launch, staging, arming, capture, or finalization was performed for this
  package, so every claim above is unit-level only.

## State left behind

Only the owned paths were added or amended: `dataset/batch.py` (new),
`dataset/__init__.py` (exports), `scripts/plan_capture_batch.py` (new),
`tests/test_capture_batch.py` (new), `dataset/console.py` (the single ASCII
docstring phrase in the second pass), this report, and the byte-safe
`AGENT_COMMS.md` completion note. No source save, live `.minecraft`, staging
world, dataset scene, or build artifact was created or modified; no
git stage/commit was made. No self-chain was started.
