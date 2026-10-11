# Texture POC data audit — 2026-10-03

Task: `agent swarm instantiation template/tasks/active/texture-poc-data-audit-2026-10-03.md`.
Kind: read-only source/artifact audit. Write scope: this file only.

Method note: this audit was performed with file reads only, under the read-only/no-shell
profile of `docs/deepseek-worker.md` (no PowerShell/bash, no hashing commands, no Python
execution, no JAR load, no training, no capture). Every number below is either read from a
tracked artifact or is arithmetic over values read from one; no percentage or count is
invented. SHA-256 values named in the two prior reports were checked for **existence** of
their files, not recomputed — that residual check belongs to the lead, who can hash.

## Verdict

- Exact visible pixel/face/patch labels **can be derived today** from current scene truth and
  captures. The reference geometry, the per-frame ground-truth pose, the world region, and a
  per-pixel truth-surface resolver all exist and are already used by the factory-dataset
  evaluators. What does **not** exist is any stored label artifact, any dataset-side label
  derivation, or any frozen per-frame usable mask for factory captures.
- The current factory has **3 independent build/topology groups**, **5 finalized scene
  entries**, **26 screenshots**, and **every** index split is `train` (`dev = test = 0`). The
  absence of a held-out fold is a consequence of the deterministic split hash on the three
  existing groups, not of missing captures.
- Source-build held-out evaluation **is possible now only as an exploratory
  leave-one-group-out over three groups**, plus T2/T6 as additional unseen builds. A strong
  generalization claim is not possible, and the repository already says so in
  `scripts/evaluation/evaluate_oracle_candidate_graph.py` (`LIMITATIONS`) and in
  `evaluation_pair_correspondence.build_group_manifest`.
- The T6 bottleneck is consistent with a **candidate-generation/recall** problem
  (39.26% top-8 membership, 48.54% ordered recall at K=32, truth in the canonical palette),
  with a secondary palette-coverage problem (87/2,007 references outside the palette). A local
  patch model is a plausible auxiliary proposer/ranker, but the audit's smallest next step is
  **label derivation first**, not training.

## 1. Can exact visible pixel/patch labels be derived? (trace)

### 1.1 What current truth and captures contain

| Fact needed for a label | Stored where | Schema / function |
|---|---|---|
| Exact block states over the region | `scenes/<id>/ground_truth.json` | `dataset/schemas.py:GroundTruthExport`, `BlockRecord` (`pos`, `name`, namespaced, `properties`) |
| Region bounds (world) | `ground_truth.json.region`, `capture.json.region` | `GroundTruthExport.region`, `RegionBounds` (inclusive `min`/`max`) |
| Region size (local) | `scenes/<id>/recipe.json.dimensions` | `SceneRecipe.dimensions` |
| Per-frame ground-truth pose | `capture.json.cameras[]` (planned) and `capture.json.frames[].actual` (recorded) | `CameraModel`; `evaluation_pair_correspondence._actual_camera` prefers actual, falls back to planned |
| Per-frame resolution | `capture.json.resolutions[]`, `frames[].actual.framebuffer_width/height` | `_frame_resolution` |
| Screenshot pixels + digest | `scenes/<id>/captures/<file>.png`, `capture.json.screenshots[]` | `finalize_scene` copies and SHA-256-checks each frame |
| Exact reference litematic | `scenes/<id>/reference.litematic` | `dataset/blocks.py:write_reference_litematic`, pinned to `bounds=(region.min, region.max)` so it stays coordinate-comparable to the camera poses |
| Requested vs actual client settings | `capture.json.frames[].actual` / `.requested` | `finalize.validate_settings` (gamma, FOV, HUD, time, weather, framebuffer) |

### 1.2 The derivation chain

1. **Reference volume.** `dataset/blocks.py:volume_from_ground_truth` → `BlockVolume` of exact
   `BlockState`s (air dropped; `AIR_BLOCKS`). `reference.litematic` carries the same states.
2. **World↔local transform.** `reconstruction/scene.py:SceneTransform(world_origin, size)`.
   `load_scene_transform` reads a *template* `scene.json`
   (`reference_region.world_min` / `world_max_exclusive`). Factory scenes have **no
   `scene.json`**; the equivalent is directly derivable as
   `world_origin = capture.json.region.min`, `size = region.max - region.min + 1`, which
   equals `recipe.json.dimensions` (checked: train-station 65×33×65; canal-houses 40×57×40;
   canal-town 17×23×16). This is a small adapter, not a blocker, and the existing evaluator
   sidesteps it by raycasting in world coordinates.
3. **Per-pixel truth surface, two independent implementations.**
   - Evaluator cube-truth: `evaluation_pair_correspondence.raycast_surface(frame_id, camera,
     width, height, occupancy, x, y)` → `_traverse` front-to-back voxel walk over
     `FinalizedScene.occupancy`, returning
     `SurfaceHit(frame_id, coordinate, slot, world_point, camera_depth, state, pixel)`. Entry
     face comes from `_ENTRY_SLOTS`; partial-model cells occlude but return
     `excluded_partial_owner` instead of a surface. This already resolves **any** pixel and is
     already used on factory data by `observe_frame` / `observe_oracle_frame`.
   - Renderer ownership: `export/attribution.py:render_result` → `render_volume` returns
     `(canvas, ownership, index, depth, uv)` with `index[i] = (coordinate, slot)`, per-face
     alpha cut-out at `alpha_cutoff = 0.5`, and a UV buffer; `export/visibility.py:visible_cells`
     aggregates per-cell pixel counts at `scale = 1.0` and applies `usable_masks` after
     rendering. This path is exact for partial models and cut-outs but needs a palette carrying
     `model_faces` / `shape_boxes` / `covered_slots` (i.e. `VanillaJarTextureProvider` +
     `palette_for_states(reference_states)`).
4. **Per-patch labels identical to production patches.**
   `reconstruction/face_sampling.py:sample_visible_faces(coordinates, camera, scene, image,
   gui_mask=..., minimum_area=24.0, shrink=0.35, minimum_visible_fraction=0.6,
   blocks=<reference states>, palette=<reference palette>)` returns
   `FaceSample(coordinate, slot, patch(16,16,3), area, depth, visible, facing, valid,
   coverage)`. With `blocks`+`palette` supplied, `resolve_sampling_geometry` /
   `exact_sampling_blocks` select `EXACT_MODEL`, ownership comes from `render_result` on the
   **reference** geometry, and `(coordinate, slot)` is therefore an exact reference-derived
   label for the patch. `rectify_face_valid` yields per-pixel `valid`/`coverage`, so the label
   is per-pixel and per-patch, not only per-cell. This is the same call the production pass uses
   (`reconstruction/raycast.py` loop at ~lines 804–883), which builds `raw` tuples
   `(frame_id, coordinate, rgb, descriptor, slot, weight, pattern, area)` and **never persists
   them**.

### 1.3 Missing fields / artifacts (concrete)

1. **No label artifact and no dataset-side derivation.** A content search of
   `src/mc_photogrammetry/dataset/` for `visible|label|occlu|pixel|face` finds only HUD/quality
   tags and a projected-pixel envelope; nothing computes visibility, pixel ownership, face
   ownership, or patch labels. Finalization (`dataset/finalize.py:finalize_scene`) writes
   `capture.json`, `ground_truth.json`, `reference.litematic` and the index only. The existing
   machinery is evaluator-side: `evaluation_pair_correspondence.py` (keypoint-level truth
   surfaces) and `export/visibility.py` (cell-level pixel counts). Neither emits training
   patches.
2. **No frozen usable mask for factory frames.** `capture.json` records `hud_hidden`
   per frame only. Factory captures include HUD-visible frames:
   `smoke-train-station-ds20c`: 2 of 4 hidden; `pilot-batch-train-station` orbit-a: 3 of 6
   hidden; the other three scenes: all hidden. Both evaluators fall back to the production
   heuristic (`create_gui_mask` / `usable_mask_for(..., None)`), which
   `benchmark_templates.py` explicitly calls provisional and gate-ineligible; only
   `templates 2` (3 frames) and `templates 6` (4 frames) have reviewed frozen masks in
   `docs/evaluation_masks/`.
3. **No `scene.json` in factory scenes** (world↔local transform must be reconstructed as in
   §1.2 step 2).
4. **No lighting/light-level export.** Only `settings.time_of_day`/`weather` are recorded.
   Material appearance in a screenshot is texture × block light × exposure; the pipeline
   re-estimates this (`reconstruction/lighting.py`, `MINIMUM_LIGHT_FACTOR = 0.15`,
   `MINIMUM_LIGHT_EFFECT`). A label generator must reuse that inversion or record brightness as
   a label, otherwise bright/dark patches of the same state are inconsistently labelled.
5. **Requested packs are provenance only.** `active_resource_packs` is recorded (factory
   scenes show `["vanilla"]`), while `queue/manifest.json.pack_note` states pack names are
   requested provenance and `resource_pack_applied` / `shader_pack_applied` are `false`. This is
   sufficient for the vanilla POC and is not proof any non-vanilla pack was active.

### 1.4 Occlusion and face-ownership limitations

- **Region clipping is the sharpest risk.** `GroundTruthExport.__post_init__` refuses blocks
  outside the declared region, so unexported geometry cannot occlude in the truth model. In
  every factory scene checked, cameras stand **outside** the exported region: train-station
  cameras at y 59.7–71.3 / x up to 286 / z −174 against region y≤56, x≤240, z≥−128;
  canal-town camera at y 90.0, z 389.5 against region y≤88, z≤376; canal-houses camera at
  z 61.3 against region z≤5. Any unexported occluder between camera and region yields a pixel
  that the truth model still attributes to a region face. This makes per-pixel labels
  *derivable* but not automatically *clean*; a photometric/consistency gate is required before
  training (see §6). This audit did not render anything and does not claim the mislabel rate.
- **Cube vs exact geometry.** The factory-data path that exists today (`raycast_surface`) uses
  the name heuristic `is_full_cube_truth_state`; partial families (slabs, stairs, fences,
  panes, plants, …) occlude as full cubes and are excluded from ownership. Exact partial
  geometry needs the local 1.21.5 JAR + `palette_for_states`. Production sampling itself falls
  back to `CUBE_GEOMETRY` whenever any target lacks a state or exact model
  (`STATES_UNAVAILABLE` / `STATES_INCOMPLETE` / `MODEL_METADATA_UNAVAILABLE`), so a label set
  built from a reconstruction's own geometry would be cube-approximate for exactly the cells
  most likely to be mislabelled.
- **Two different "visible" denominators.** The pinned denominator is
  `visible_cells(..., minimum_pixels=1)` (T6: 2,007; T2: 667). The production sampler then
  drops faces below `MINIMUM_FACE_AREA = 24 px`, below `minimum_visible_fraction = 0.6` of the
  in-frame footprint, with `shrink = 0.35`, and the reconstruction loop adds
  `MAX_FACE_VARIATION = 12.0` and `min_surface_contrast` gates. A labelled subset is therefore
  **scored-visible**, not all-visible; the two denominators must be reported separately (the
  prior reports already keep 2,007 visible distinct from 393 support-candidate cells).
- **Slot-level vs UV-level ownership.** `_traverse` reports the entered axis face and no UV;
  the renderer path has a UV buffer and per-face alpha, so only it can label sub-face pixels of
  a partial model exactly. Grazing-angle contamination is why production weights by `facing`,
  `visible`, `area**0.5` and uniformity; a label generator should reuse those weights or state
  explicitly that its denominator differs.
- **Transparency/cut-outs.** Alpha masks exist in the JAR palette (`_encode_alpha_mask`,
  `face_model_faces`; renderer honours `alpha_cutoff = 0.5`). The cube-truth path would treat
  leaves/glass as opaque cubes. Any POC label set must exclude or handle these explicitly.

## 2. Finalized build groups and split distribution (verified)

Verified by reading all five `exports/datasets/*/index.json` and listing each scene directory.

| Dataset (`index.json`) | Scene id | Frames | `scene_group` (topology) | `topology_digest` | Split | `provenance.staging_token` (build) |
|---|---|---:|---|---|---|---|
| `smoke-train-station-ds20c-2026-09-12` | `train-station-spawn-208-96` | 4 | `1757a5ad…` | `b50cf72b…` | train | `4437817610…` |
| `pilot-batch-train-station-2026-09-12` | `train-station-spawn-208-96-orbit-a` | 6 | `1757a5ad…` | `b50cf72b…` | train | `4437817610…` |
| `recon-canal-houses-2026-09-12` | `canal-houses-test-recon-player-spawn` | 4 | `3e9dfb43…` | `59402f6d…` | train | `0155973001…` |
| `fresh-c4-canal-town-2026-09-12` | `canal-town-architectural-component-bbox` | 4 | `74f080aa…` | `075f77bf…` | train | `b4aeeb8440…` |
| `fresh-c4-canal-town-framed-2026-09-12` | `canal-town-architectural-component-bbox` | 8 | `74f080aa…` | `075f77bf…` | train | `b4aeeb8440…` |

Counts (each independently checked):

- 5 finalized datasets / 5 index scene entries / **26 screenshots**, all present at
  `scenes/<id>/captures/` (4 + 6 + 4 + 4 + 8) and all 26 screenshot digests distinct.
- **3 topology groups** and **3 source builds**. `scene_group` is the topology key
  (`dataset/splits.py:scene_group` / `volume_topology`); the build key is the staged-save
  content fingerprint (`dataset/batch.py:build_identity_for`, `source_identity =
  staging_fingerprint`), which the index preserves as `provenance.staging_token`. The two keys
  coincide here (train-station: 2 scenes / 2 datasets / 1 group; canal-town: 2 datasets /
  1 scene id / 1 group; canal-houses: 1).
- **Distinct artifacts:** `ground_truth.json` hashes — 4 distinct (the two canal-town exports
  share `36114fee…`; the two train-station exports differ, `027ac042…` vs `ea2cdd51…`);
  `reference.litematic` — 5 distinct (identical topology can still have different bytes because
  the exporter records the scene name). Group identity therefore comes from `topology_digest`,
  never from an artifact hash.
- **Split distribution:** every index reports `by_split = {train: 1, dev: 0, test: 0}`.
  Recomputed buckets `int(scene_group[:8], 16) % 100` = **13, 67, 34** against
  `TRAIN_CUTOFF = 70` — all three land in `train`, so no dev/test fold can appear without new
  structures. This is the concrete reason the candidate-depth report's "no held-out fold"
  statement holds.
- **Index grouping mechanics:** `build_group_manifest` groups strictly by `scene_group`
  (duplicate exports of one structure cannot be separated) and already emits exploratory
  leave-one-group-out folds with `check_split_leakage` / `check_fold_leakage`; the
  train-station pair is the clean "same topology, different capture seed/views" case, and the
  fresh-c4 pair is the clean "same ground truth, 4 vs 8 views" case.
- **Unfinalized material:** `exports/datasets/smoke-train-station-2026-09-12` has a scene
  `plan.json`/`recipe.json`, a successful run (`run_failed: false`,
  `ground_truth_exported: true`), 4 staging screenshots, and **no** `capture.json` /
  `ground_truth.json` / `reference.litematic` / `index.json`.
  `exports/datasets/smoke-frozen-wasteland-2026-09-12` has plan/recipe only and a failed run
  (`run_failed: true`, "the plan aborted before a ground-truth export: no integrated world was
  entered before the wait limit"). Neither is dataset material today.
- **Reported artifacts:** the files named in
  `docs/reports/visible-state-stage-attribution-2026-10-03.md` (`.tmp-stagewise-pinned-template-2-*`,
  `-6-*`, `-t6-no-prune-*`) and in `visible-state-candidate-depth-2026-10-03.md`
  (`.tmp-ds-r36-t6-noprune-*`) all exist. The fixture mask counts match
  `docs/evaluation_masks/` exactly (templates-2: 3 masks; templates-6: 4 masks) and
  `docs/reports/evaluator-fixture-pins.json` lists `reviewed: true` for both roles. Digests were
  **not** recomputed (read-only/no-shell audit).

## 3. The current scorer baseline, on identical patches

The baseline the POC must beat is the existing per-cell appearance cost and its top-K selection:

- Cost vector: `reconstruction/appearance_scoring.py:PaletteIndex.costs` — expected slot colours
  from `blocks/appearance.py:palette_face_colors` (face map × `FACE_SHADE`), then
  `appearance_distance` (`CHROMA_WEIGHT = 0.75`, fitted gain, floor `MINIMUM_LIGHT = 1.0`),
  plus a `patch_descriptor` histogram/mean/σ/gradient term (weight 80), an optional texture
  pattern term (`texture_weight`; benchmark default `raycast.TEXTURE_WEIGHT = 40.0`), a
  `simplicity` term, and a `shape_weight`/`slot_coverage` term; every candidate is finally
  `min(distance - best, OUTLIER_CEILING = 24.0)`.
- Per-cell aggregation: `raycast.py` `raw` observations weighted by
  `area**AREA_POWER(0.5)`, depth, `visible`, `facing` and uniformity
  (`_accumulate_costs`), then `_top_candidates(mean_costs, index, keep=8)` via `np.argpartition`,
  then global support pruning (`PALETTE_SUPPORT = 0.02`, `PALETTE_MINIMUM = 6`,
  `palette_prune_rounds = 2`, `LOCAL_SUPPORT_KEEP = 0`), `_smooth_adjacent_states`, `_decide`.
- Palette: `VanillaJarTextureProvider.canonical_palette()` for the pinned runs — the
  appearance+pattern de-duplicated **726-state** full-cube palette; reference-assisted mode uses
  `palette_for_states(..., include_partial=False)`.
- **"Identical patches"** must mean: freeze the `raw` observation list
  `(frame_id, coordinate, slot, patch bytes, valid, coverage, weight)` once, then score the
  current scorer and any learned proposer/ranker over that frozen list. The existing
  `stage_trace` (`raycast._emit_stage_trace`, `stage_trace_candidate_limit`) is **not** a patch
  store — it records `top_candidates` and per-cell decisions and is diagnostic-only, never read
  back.
- Measured baseline bottlenecks (from the two prior reports, reproduced here as evidence, not
  recomputed): T6 default 17.44% visible exact; palette-to-support removes 1,527 occurrences;
  no-prune ablation 25.46%; top-8 membership 39.26% (788/2,007); ordered in-palette recall
  41.30% / 44.27% / 48.54% at K = 8/16/32; support→candidate still loses 1,132 and
  candidate→winner 318. T2 is different: 71.96% visible exact with truth often already in the
  candidate set (raw winner 459/667) and 68 visible references outside the palette.

## 4. Local CNN / vision encoder as an auxiliary proposer/ranker

Formulated strictly against the above baseline. It is an auxiliary in-vocabulary proposer and
reranker, not a replacement for geometry or for the pipeline's gates.

- **Role.** Given one rectified patch and its world face slot, output either a distribution over
  the run's palette or a reordering/prior over it. Two workable shapes: (a) direct full-palette
  classifier + abstain; (b) material/family head whose output conditions deterministic property
  assignment. Prefer injecting the result as an additive prior (`PaletteIndex.with_prior`) so
  `_top_candidates`, support pruning, smoothing and placement stay byte-identical when the prior
  is zeroed.
- **Training labels.** Truth `(coordinate, slot, state)` from §1.2 step 4 — reference-litematic
  geometry + recorded actual pose, patch pixels from the real screenshot. Never train on the
  T2/T6 frames; they are the evaluation builds.
- **Class granularity.** Start at exact `BlockState` over the palette the run loads, and report
  two granularities: block-name level and canonical-palette-class level. The canonical palette
  de-duplicates states that are visually identical by construction, so for some states a
  name-level label is unrecoverable; report the palette's label-collision count so the ceiling
  is explicit. Partial blocks need the `palette_for_states(..., include_partial=True)` vocabulary
  and exact model geometry; if they are excluded, say so and score them separately.
- **Augmentation.** Photometric only (brightness/gamma mapping consistent with the light
  inversion, mild blur/noise/compression, colour cast within the observed `LIGHT_CAST`) plus
  small pose jitter bounded by the recorded tolerances (`position_tolerance_blocks = 0.25`,
  `HEADING_TOLERANCE_DEGREES`) that shifts the patch a fraction of a texel. No horizontal flips
  or 90° rotations — they permute face slots and silently relabel east/west; no brightening past
  the "texture is only darkened" assumption; no augmentation drawn from the evaluation fixture.
- **Build-held-out evaluation.** Provide the model one `scene_group` fold at a time using the
  existing `build_group_manifest` folds (leakage-checked). Report each fold separately and pool
  only as "exploratory", because three groups cannot support a generalization claim. T2/T6 are
  then reported as additional **unseen builds** for transfer.
- **Top-k state recall.** Report exact-state top-1 and top-k at k = 8/16/32 (matching
  `_top_candidates` and `stage_trace_candidate_limit`) on the frozen patches, and always
  alongside the current scorer's rows on the same patches, so the delta is causal. Keep the
  conditional denominators explicit: (i) all visible reference cells, (ii) truth inside the
  palette, (iii) truth inside the model's vocabulary. The T6 no-prune ablation is the right
  harness because it removes the global-pruning confound while remaining gate-ineligible.
- **Calibration / abstention.** Fit a temperature/isotonic map on a held-out group; abstain on
  low confidence and route the cell into the **existing** review vocabulary
  (`appearance_state_proposals.ABSTENTION_REASONS`: `no_observation`, `empty_palette`,
  `weak_margin`, `high_cost`, `conflicting_faces`, `budget`) rather than inventing a status.
  Report coverage-vs-risk and calibration error; a learned score that cannot be mapped onto the
  same relative capped cost scale (`OUTLIER_CEILING`, `max_cost`, `min_margin`) must be carried
  as an explicit prior instead.
- **End-to-end gates.** Unchanged `scripts/benchmark_templates.py` runs on the pinned fixtures
  with frozen evaluator masks, using the stage trace:
  acceptance is ≥ `ACCEPTANCE_MINIMUM_VISIBLE_EXACT_STATE_RECOVERY = 0.80` visible exact-state and
  ≥ `ACCEPTANCE_MINIMUM_VISIBLE_GEOMETRY_F1 = 0.90` visible geometry F1, with any
  `palette_support_override` making the run ineligible by design. Report
  `_stagewise_visible_loss` so a gain is attributed to support, candidate generation, winner,
  smoothing or placement rather than to the aggregate.

## 5. Is source-build held-out evaluation possible now?

- **Yes, exploratory only.** Three independent groups exist, all held in `train`, and the
  leakage-checked LOO folds are already generated by
  `evaluation_pair_correspondence.build_group_manifest`. T2/T6 add two further unseen builds
  that are not in the factory groups. Train-station and canal-town each have two capture
  variants, which additionally supports a *within-group* view-held-out check but **not** a
  source-build holdout.
- **No for a strong claim.** With three groups, one structure each and no dev/test fold, a
  model evaluated this way cannot be reported as a generalizing recognizer. The repository's own
  limitations text agrees
  (`evaluate_oracle_candidate_graph.LIMITATIONS`: "Only three independent topology groups exist
  and every index split is 'train', so no generalization claim is made"). Expanding grouped
  capture data (more independent saves/structures) is a prerequisite for any stronger claim.
- The pinned acceptance fixtures remain `template-2` and `template-6`; only those two roles are
  `reviewed: true` in `docs/reports/evaluator-fixture-pins.json`.

## 6. Smallest causal experiment

Two stages. E1 has no training and is a prerequisite for E2; E2 is the smallest experiment that
actually tests the model.

### E1 — Label derivation and identical-patch oracle-candidate audit (no training)

- **Inputs:** the T6 and T2 fixtures (reference litematic, `cameras.json`, `scene.json`),
  their reviewed mask specs under `docs/evaluation_masks/`, and the local 1.21.5 JAR.
- **Procedure:** build the reference volume and a JAR-backed palette
  (`palette_for_states(reference states)`); for each frame call `sample_visible_faces` with
  `blocks` + `palette` and the frozen mask as `gui_mask`; persist
  `(frame_id, coordinate, slot, patch, valid, coverage, truth_state)` with a content digest.
  In parallel compute the broader `visible_cells(minimum_pixels=1)` denominator.
- **Measure:** (i) label coverage — how many of the 2,007 T6 visible cells yield ≥1 labelled
  patch, and label purity under a photometric consistency gate; (ii) the current scorer's
  identical-patch top-k and winner accuracy at k = 1…32; (iii) per-slot and partial-block
  breakdown; (iv) how much of the 1,131 support→candidate and 318 candidate→winner losses is
  "no usable patch" versus "patch present, truth outside K".
- **Success criterion:** the artifact is byte-reproducible on a rerun and reconciles with the
  published 39.26% / 41.30% / 44.27% / 48.54% rows on the same denominator.
- **Failure criterion:** it cannot reach those rows, which means the patch denominator differs
  from the trace denominator — reconcile denominators before any training, and do not train on a
  label set whose coverage is unexplained.
- **Compute:** CPU only; T6 is a 7,728-cell region with 4 frames and T2 a 1,836-cell region with
  3 frames, so this is small and fast relative to the 419.8 s default T6 reconstruction.
- **Establishes:** whether the bottleneck is patch/label quality or ranking, and the exact frozen
  patch set every later model must be compared against. This is the cheapest step most likely to
  change what we believe about the T6 bottleneck.

### E2 — Frozen-encoder linear probe over the canonical palette (smallest model test)

- **Train data:** E1-identical labels from the three factory build groups, LOO by
  `scene_group`; never T2/T6. Volume is bounded by the exports: train-station 139,425 cells × 10
  frames (4 + 6), canal-houses 91,200 × 4, canal-town 6,256 × 12 (4 + 8) before occlusion and
  sampler rejection — the exact label count is an E1 output and is not yet measured. T6
  evaluation is ≤ 2,007 visible cells × 4 frames.
- **Model:** one compact encoder (small CNN, or a frozen pretrained patch encoder) plus a linear
  head over the palette and an abstain output; optionally a second material/family head.
- **Bound compute:** CPU-only probe is sufficient at 16×16×3 patches; keep the current pinned T6
  reconstruction time as the end-to-end cost baseline and require no material runtime
  regression beyond the added scoring call.
- **Success (against the baseline, not "AI"):** identical-patch top-8 exact-state recall
  strictly above the current 39.26% set membership and above the 41.30% ordered-K8 row on T6,
  with calibrated abstention that does not reduce in-palette coverage, and no change to visible
  geometry F1 (85.39%) or to the 0.80/0.90 gate definition.
- **Failure:** no K=8 recall gain, or a gain only where truth is outside the canonical palette
  (the model is then not the bottleneck; the lever is palette/geometry coverage, e.g. the 87
  out-of-palette T6 references).
- **Establishes:** whether the T6 recall bottleneck is a representation limit of the colour +
  pattern scorer that a small learned patch model can close. Every other stage and every patch
  is frozen, so the comparison is causal.

## 7. Determinism to preserve

The model may only propose and rank **material states**. Keep deterministic: occupancy and
placement (`_remove_enclosed_blocks`, placement passes), adjacency smoothing with fixed weights
(`_smooth_adjacent_states`), shape gating (`slot_coverage`, `shape_weight`), directional and
state properties inferred from geometry/occlusion rather than appearance, and the existing
`argpartition` tie behaviour (the K=8 membership delta at exactly `OUTLIER_CEILING = 24.0` is a
known tie artefact, not a modelling signal — do not "fix" it inside a model experiment).
Prefer a zero-able prior injection over replacing the cost vector so a model arm can be ablated
to the exact baseline.

## 8. What was verified, and what was not

- Verified by reading: all five indexes and scene directories; 26 screenshots present and
  distinct; 3 topology groups; 3 build (staging-token) groups; every split `train` with
  `dev = test = 0`; bucket arithmetic 13/67/34 vs cutoff 70; two plan-only unfinalized scene
  directories; the frozen-wasteland failure record and the smoke-train-station success-without-
  finalization record; the fixture mask counts and reviewed pins; existence of every `.tmp`
  report artifact named in the two prior reports.
- Not verified (out of scope for a read-only/no-shell audit, or requiring execution): SHA-256
  values named in the prior reports; any rendered label, patch or model result; whether a CNN
  actually helps; the unexported-occluder mislabel rate. No training, dependency installation,
  download, launcher/Minecraft run, screenshot capture, palette/gate/threshold change, or use of
  evaluator labels as model inputs occurred. Only this report was written.
- Residual risk to hand to the lead: quantify region-clipping label noise in E1 before any
  training; freeze usable masks (or restrict to `hud_hidden: true` frames) for any factory frame
  used as training data; and do not let the pinned T2/T6 fixtures enter training.
