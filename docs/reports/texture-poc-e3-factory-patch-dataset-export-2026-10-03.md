# Texture POC E3 — factory production-aligned patch-dataset exporter (2026-10-03)

Task: `texture-poc-e3-factory-patch-dataset-export-2026-10-03` (assigned in
`AGENT_COMMS.md`, lead → DeepSeek Sol-equivalent worker).

Kind: bounded evaluation-only exporter implementation plus synthetic tests.
Write scope: exactly the three paths below. No commit.

**Worker status:** implementation and synthetic verification completed. No real corpus,
image, reference, camera, scene or JAR was read by the worker. Lead review and the
first bounded corpus export are recorded below.

Owned and written this attempt:

| Path | Role |
|---|---|
| `scripts/evaluation/export_factory_texture_patch_dataset.py` | the evaluation-only E3 exporter |
| `tests/test_export_factory_texture_patch_dataset.py` | focused synthetic-fixture tests |
| `docs/reports/texture-poc-e3-factory-patch-dataset-export-2026-10-03.md` | this report |

The exporter is the dataset-side counterpart to the reviewed E1 patch-label
harness (`scripts/evaluation/evaluate_texture_patch_labels.py`). It reads only
*finalized* factory datasets, so the pinned T2/T6 evaluation fixtures are not an
input and are additionally refused by path.

## 1. Inputs and admission

A dataset root must hold `index.json` plus `scenes/<id>/{capture,ground_truth,
recipe,reference}.json/litematic`. Loading reuses the reviewed factory helper
`mc_photogrammetry.evaluation_pair_correspondence.discover_finalized_entries`,
The E3 preflight validates index, scene, hash-file and screenshot paths before
calling discovery; discovery then recomputes every pinned scene-file and
screenshot digest before the exporter looks at geometry. On top of that the exporter enforces:

* **Finalized only.** Every `scenes/*` directory must be a finalized index entry
  carrying `capture.json`, `ground_truth.json` and `reference.litematic`; an
  unindexed or incomplete (staging) scene directory is a hard refusal, as is a
  missing index, a non-`advanced_only` index, a hash mismatch, or an index entry
  missing from its own payload.
* **Safe paths.** A scene `directory` must be exactly `scenes/<id>` and stay
  inside the dataset root; absolute, drive-prefixed and `..` directories are
  refused. Screenshot filenames must match `^[A-Za-z0-9][A-Za-z0-9._-]*\.png$`.
  The output path is refused if it contains `..`, is an existing directory, or
  resolves inside the dataset root.
* **Hash before decode.** The index/screenshot digests are verified up front,
  and each admitted screenshot is re-hashed immediately before its first decode.
* **Actual pose required.** The camera is rebuilt from the frame's recorded
  `actual.eye_position` / `yaw_degrees` / `pitch_degrees` / `fov_degrees`, with
  `eye_height=0.0`. A frame without a complete, finite actual pose is refused; the
  planned `capture.json.cameras[]` pose is never substituted (a dedicated test
  keeps a valid planned camera present while the frame is refused).
* **HUD hidden required.** `actual.hud_hidden` must be exactly `true`.
* **Vanilla only (E3 ID note).** Admission requires an explicitly reported
  `active_resource_packs == ["vanilla"]`, `resource_pack_applied == false`, and
  `shader_pack_applied == false`. Requested pack fields are provenance only. A
  requested-but-unapplied resource-pack name stays admissible as provenance
  (matching the audit's reading of `pack_note`); non-vanilla actives are
  refused, and any missing/unknown pack or shader state is refused rather than
  treated as vanilla.

### Capture refusal is per frame and recorded

The audit requires the factory's HUD-visible frames to be dropped and its hidden
frames kept, and E2's leave-one-group-out over three factory groups needs the
train-station group, whose scenes contain HUD-visible frames. Capture admission
is therefore **per frame**: a refused capture is excluded and recorded with a
typed reason (`admission.frames[].reason`, `admission.scenes[].refused_frames`,
and the pooled `admission.refusals` histogram) while the scene's other captures
may still export. A scene that yields no admitted capture is recorded as
`admitted: false`, and an export in which *no* capture passes admission is
refused. Scene-level structural problems (staging/incomplete/unverified/unsafe)
remain hard refusals that raise before any image is decoded.

Typed refusal reasons: `capture_failed`, `invalid_screenshot_filename`,
`missing_screenshot_digest`, `missing_actual_metadata`, `missing_actual_pose`,
`hud_not_hidden`, `non_vanilla_pack_metadata`, `unknown_pack_metadata`.

## 2. Exact-state label derivation

Per admitted frame the exporter mirrors the reviewed E1 derivation:

* The reference geometry is the hash-pinned `reference.litematic`, loaded through
  the reviewed `load_litematic`; its blocks are local coordinates and the
  audited factory transform is derived as `SceneTransform(region.min,
  region.size)`, so a label's world coordinate is `local + region.min`. No
  runtime coordinate is inferred from a label: the pose is the recorded one and
  the geometry is the exported reference.
* The geometry palette is
  `provider.palette_for_states(reference_states, include_partial=True)`. Only
  states carrying exact baked model polygons are labelled; a state without them
  is counted as `model_metadata_unavailable` and is never sampled through the
  cube fallback.
* Sampling is the reviewed `reconstruction.face_sampling.sample_visible_faces`
  with `blocks` + `palette` (exact-model alpha-tested ownership) at production
  `shrink = raycast.FACE_SHRINK` (**1.0**), `minimum_area = 24.0`,
  `minimum_visible_fraction = raycast.MINIMUM_VISIBLE_FRACTION` (0.6),
  `depth_scale = 1.0`, and `gui_mask = None` (HUD-hidden is an admission
  requirement, and factory scenes have no frozen mask). Patches are rectified to
  16x16x3 with per-pixel `valid` (ownership) and fractional `coverage`.
* A face with `valid_fraction < 0.5` is refused (`insufficient_valid_coverage`);
  a fully unowned face is refused (`no_valid_pixels`). Duplicate polygons on one
  cell face collapse deterministically to the most fully observed sample.
* The recorded observation `weight` is the exact production formula, including
  the `_face_uniformity` divisor.

## 3. Artifact

Schema `mc-photogrammetry-factory-texture-patch-dataset-v1`, engine `e3-1`,
`evaluation_only: true`, `training_permitted: false`, `t2_t6_excluded: true`.
Each example stores `scene_id`, `dataset_id`, `scene_group`, `fold`,
`frame_id`/`filename`, local `coordinate` and derived `world_coordinate`, `slot`,
the exact `label` state, observation metrics, the `patch`/`valid`/`coverage`
arrays (self-describing dtype/shape/base64 + SHA-256) and per-example
`patch_sha256`/`valid_sha256`/`coverage_sha256`/`example_sha256`. Build
provenance is `staging_token` + `world_identity` from the index (the batch queue
`build_id` is deliberately **not** required, per the E3 ID note).

The artifact also carries the verified inputs (index digests, per-scene file and
screenshot digests, JAR digest when supplied), the admission policy and every
refusal, the sampler parameters, the grouping record, a coverage summary, and a
self-seal `artifact_sha256` over canonical JSON with that field removed. The
artifact is bounded (`MAX_EXAMPLES = 200_000`), contains no timestamps, sorts
examples by `(scene_group, scene_id, frame_id, coordinate, slot)`, refuses to
overwrite an existing path (checked before work and again before the atomic
replace), refuses a non-frozen patch domain, and refuses an output inside the
dataset root.

Grouping: `scene_group` is the authoritative conservative leakage group. Each
example's `fold` is its group's index split (the finalizer already keys splits by
topology), and the reviewed `build_group_manifest` leave-one-group-out record is
embedded, so all captures of one `scene_group` stay in one fold. Split/fold
leakage is refused through `check_split_leakage` / `build_group_manifest`.

## 4. Checks run

All commands run from the repository root, foreground only:

```powershell
.venv\Scripts\python.exe -m py_compile scripts/evaluation/export_factory_texture_patch_dataset.py tests/test_export_factory_texture_patch_dataset.py
# exit 0

.venv\Scripts\python.exe -m pytest tests/test_export_factory_texture_patch_dataset.py -o addopts= -q --no-header
# 36 passed in 0.76s
```

Scoped whitespace scan of the two owned Python files: no trailing whitespace, no
tab characters, LF line endings only. Scoped `git status` shows only the three
owned paths as new; all pre-existing dirty hunks were left untouched.

Synthetic tests cover: schema/integrity (sealed self-digest, evaluation-only and
T2/T6 markers, deterministic byte-identical reruns, overwrite refusal, output
inside dataset root, unsafe output path, array round-trip/malformed refusal,
frozen patch-domain enforcement); finalized admission (staging/unindexed scene,
incomplete scene, tampered scene-file and screenshot hashes, scene-directory
escape, missing index, non-advanced index, unknown split, missing staging token,
T2/T6 fixture root); actual-pose/HUD/vanilla admission (HUD-visible refusal,
missing actual pose with a valid planned pose present, missing actual metadata,
non-vanilla actives, requested-but-unapplied pack provenance accepted, missing
shader-applied state and unknown/empty active-pack state refused, a scene whose
only captures are refused, an all-refused export); group isolation
(same group one fold and one LOO held-out set, split leakage across one group
refused); bounds (reference block outside the region, label cap, local→world
coordinate transform inside region bounds); and exact-model labelling (unmodelled
state refused, not cube-labelled; empty exact-model palette refused). No test
loads a real corpus frame, reference, camera, scene or JAR, decodes a real image,
or writes outside `tmp_path`.

## 5. Hashes (SHA-256)

```
scripts/evaluation/export_factory_texture_patch_dataset.py
  290720b253b7da4d4c7e77bc9431b0b6ecd44c40ed6e5cb17a73293e301a8c94
tests/test_export_factory_texture_patch_dataset.py
  3ed6862ff85f3cdc55ba129d870ebe689aacbff6159c9561a459e08193bd9a45
```

## 6. Non-actions, caveats and handoff

* These statements describe the worker's implementation-only pass: no real corpus
  was read by that worker. After acceptance, the lead ran four finalized dataset
  roots with the pinned local 1.21.5 JAR and wrote ignored artifacts under
  `exports/texture-poc-e3/`. Each artifact seal, every example seal, and all
  patch/valid/coverage byte digests were independently checked.
* The four artifacts contain 8,516 patch examples over two independent
  `scene_group`s: Canal Town 3,636 (two capture datasets of one group) and Train
  Station 4,880 (two capture datasets of one group). All groups remain `train`;
  this is evaluation-only data and not yet a train/validation split.
* `recon-canal-houses-2026-09-12` passed metadata admission but emitted no exact
  visible face samples, so no artifact was written. One inspected capture visibly
  shows sky rather than the reference build. Treat this as a factory framing/
  visibility failure and exclude it from model data pending recapture/quality gate.
* The initial Train Station export reached approximately 20 GB working set and
  was stopped. After the sparse owner-mask sampler change in the linked memory
  report, the same root completed at approximately 0.58 GB sampled working set
  and emitted 2,507 examples. This is a sampled observation, not a peak-RSS
  benchmark; the reduction is roughly 97% at the observed point.
* Remaining pre-training gate: region-clipping label noise from omitted occluders
  is still unmeasured. The factory contains only two groups with usable labels,
  so LOO evaluation is exploratory and too small for a generalization claim.
  Artifacts assert `evaluation_only: true` and `training_permitted: false`; model
  training remains unapproved until label quality and split design are accepted.
* No install/download, background process, junction/reparse operation, recursive
  deletion, Defender/system change or commit occurred.


## 7. Lead acceptance and first finalized-corpus yield

Luna independently audited E3 admission, grouping and label derivation and
accepted the implementation after the lead's path-preflight and no-overwrite
review fixes. Lead reran py_compile and both focused suites: **98 passed**
(face sampler + E3 exporter). The dataset export suite verifies synthetic paths
only; no model training has occurred.

Four finalized roots produced sealed, evaluation-only artifacts, all using the
pinned JAR SHA-256 `522672ad20b460c02c2e39b6c5035ef6a849af28eb11ab2ac7eb293d395a8c11`:

| Artifact | Patch examples | Source group | Artifact SHA-256 |
|---|---:|---|---|
| `fresh-c4-canal-town-2026-09-12.json` | 1,303 | Canal Town | `fb737108a659ed377ea201c66f69fcdf3ae46d224c99f651c9bfbc7524135fa7` |
| `fresh-c4-canal-town-framed-2026-09-12.json` | 2,333 | Canal Town (same build group) | `97a3a64c06cf811e85ea9e0796de3a6348c10f5eb3324db17dc953af047537b0` |
| `pilot-batch-train-station-2026-09-12.json` | 2,507 | Train Station | `1a469f6913b2fe3850bca0ea13fdf627cd367e952f4b4154c8667ca239694965` |
| `smoke-train-station-ds20c-2026-09-12.json` | 2,373 | Train Station (same build group) | `fae14c754b1e3523c5838a3f6f6c7403a7d4cb0bae08a7aa7cff2d68329a4aaa` |

Total: **8,516 examples, 17 metadata-admitted frames and 5 HUD-visible frame
refusals, across only 2 independent source groups**. All index splits are
`train`; keeping repeated captures of each `scene_group` together leaves no
designated dev/test group, so the corpus is not ready for a trustworthy
generalization evaluation.

The fifth root, `recon-canal-houses-2026-09-12`, passed pose/pack/HUD admission
but produced no eligible exact-model face samples from any of its 4 frames. The
export now distinguishes that condition (`empty label export`) from metadata
refusal. One inspected capture shows sky rather than the reference build. It is
a capture/framing failure, not a texture-classifier error; exclude it from model
data until the camera/region visibility is repaired.

The first Train Station export attempt before sparse owner indexing reached
approximately 20 GB working set and was stopped before output. After that
semantics-preserving sampler change, the same root completed with approximately
0.58 GB working set sampled during execution (roughly 97% lower at the sampled
point), producing 2,507 examples. This is not a controlled peak-RSS benchmark;
see [owner-mask scaling report](texture-poc-e3-owner-mask-memory-2026-10-03.md).

Lead independently verified each artifact seal, all example seals, and every
serialized patch/valid/coverage byte digest. Artifacts live under ignored
`exports/texture-poc-e3/`; each declares `evaluation_only: true` and
`training_permitted: false`. Remaining gate: measure label contamination from
reference-region clipping/omitted occluders and obtain more independent usable
build groups before any model training or promotion.
