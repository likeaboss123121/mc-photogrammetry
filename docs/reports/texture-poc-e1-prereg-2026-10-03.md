# Texture POC E1 — visible-face label and production-patch preregistration (2026-10-03)

**Execution status (2026-10-03):** after independent harness review and pin
verification, the lead authorized and completed T6 and T2 E1 runs. Results,
denominators, artifact hashes, and the R3-to-R4 visibility-palette correction are
in the [E1 results report](texture-poc-e1-results-2026-10-03.md). The preregistration
below is retained as the plan and historical run contract; its original "not yet
run" statements describe its creation state only.

Task: `agent swarm instantiation template/tasks/active/texture-poc-e1-label-patch-harness-2026-10-03.md`
(route `jev-texture-poc-e1-2026-10-03`, Sol-equivalent developer attempt 01).

Harness review corrections were accepted before evaluation: R1 separated the
canonical scoring palette from reference geometry, pinned camera/scene/JAR
inputs, matched the production uniformity weight, and corrected denominators;
R2 explicitly enabled the scorer's texture pattern term; R3 fixed same-stem
mask-spec lookup for pinned PNG frames. Focused verification now passes 39 tests.

At the time this preregistration was created, only the harness and tests existed.
The later E1 run generated the two sealed, evaluation-only artifacts linked above;
no training or ML dependency was added, and no production module, palette, gate,
threshold or frozen input was modified.

## 1. Scope and write paths

Owned and written this attempt (exactly these three paths):

| Path | Role |
|---|---|
| `scripts/evaluation/evaluate_texture_patch_labels.py` | the evaluation-only E1 harness |
| `tests/test_texture_patch_labels.py` | focused synthetic-fixture tests |
| `docs/reports/texture-poc-e1-prereg-2026-10-03.md` | this preregistration |

The following reserved E1 artifacts were not created by the original harness
implementation attempt. They were produced by the later lead-authorized runs and
are retained as evaluation-only evidence:

| Reserved artifact | Contents |
|---|---|
| `.tmp-texture-poc-e1-patch-labels-20261003-template-6.json` | sealed T6 frozen patch labels |
| `.tmp-texture-poc-e1-patch-labels-20261003-template-2.json` | sealed T2 frozen patch labels |

They are `.tmp` diagnostic artifacts in the same ignored class as the sealed
`.tmp-stagewise-*` and `.tmp-ds-r36-*` files named in the two prior reports.
They must never be treated as training data or promoted as production output.

## 2. Admitted inputs and pin verification

The harness admits only:

* the two reviewed roles `template-2` and `template-6` in
  `docs/reports/evaluator-fixture-pins.json` (`reviewed: true`); an unknown or
  unreviewed role is refused outright;
* each role's exact reference Litematic, `cameras.json`, `scene.json` and PNG
  frames under `templates 2` / `templates 6`;
* the tracked reviewed per-frame mask specs under
  `docs/evaluation_masks/templates-{2,6}/`;
* the local vanilla 1.21.5 JAR
  (`%APPDATA%\.minecraft\versions\1.21.5\1.21.5.jar`).

Before a single screenshot is decoded, `preflight` verifies the reviewed
reference/frame/mask pins **and** the immutable supplied-camera asset pins below:
the reference SHA-256, the ordered frame-name + content digests, and both the
exact scored boolean mask **domain** digest and the mask-spec file digest for
every frame. A missing, malformed, unreviewed or drifting pin field (`reviewed`,
`fixture`, `reference_sha256`, `frames`, `usable_masks`) is a refusal; on refusal
the harness raises and never creates the output path. `manifest_schema` must be
an exact `int` equal to `1` (`true` and `1.0` are rejected).

### Immutable camera / scene / JAR asset pins

The camera manifest, scene definition and local vanilla 1.21.5 JAR are pinned by
immutable SHA-256 in the harness itself (`EXPECTED_ASSET_PINS`), not in the
tracked reviewed-pin manifest. They are verified, fail-closed, beside the
reference/frame/mask pins and still before any mask is parsed, any frame digest
is computed or any screenshot is decoded. A role or asset with no declared pin is
itself a refusal. Expected SHA-256:

| Role | Asset | SHA-256 |
|---|---|---|
| `template-2` | `templates 2/cameras.json` | `4f743e4fc478430f608e0e879f73cd4a3dc9ce50c726841d2da897be78be6126` |
| `template-2` | `templates 2/scene.json` | `9620608a2544e9460240c2efd0ff83da354d3ffcbee8c01e61e2d9a05b627881` |
| `template-6` | `templates 6/cameras.json` | `836ad62f99c5d960cd98c10cf8f850372c5b571be1edd14ff5935fd89027f70f` |
| `template-6` | `templates 6/scene.json` | `0a5528187e2ed47d1756d1a93dd28feb68c8c695e72aa210f067d07c45704563` |
| (both) | `%APPDATA%\.minecraft\versions\1.21.5\1.21.5.jar` | `522672ad20b460c02c2e39b6c5035ef6a849af28eb11ab2ac7eb293d395a8c11` |

The observed values were confirmed against the four on-disk files and the local
JAR while writing this revision (all five matched). `docs/reports/evaluator-fixture-pins.json`
is **not** edited: it remains the reviewed reference/frame/mask identity record.

The reported mode is always `assisted/supplied-camera evaluation`: supplied
cameras and reference-assisted geometry, not image-only and not production.

## 3. Label derivation contract

Per frame, per reference cell, the harness partitions states by exact model
metadata:

* **exact**: the state carries baked `model_faces` polygons in the **geometry
  palette** (`palette_for_states(reference_states, include_partial=True)`), so
  ownership comes from the same alpha-tested model renderer the production
  sampler uses;
* **refused**: the state has no exact polygons in the geometry palette and is
  counted as `model_metadata_unavailable`. It is never sampled through the cube
  fallback and never quietly labelled with a full-cube face. The full reference is
  still rendered for ownership, so a refused cell occludes as its conservative
  cube rather than being dropped; dropping it would silently label exact pixels
  behind it. Refused-cell faces enumerated only for occlusion are counted
  separately (`refused_state_faces`).

### Two deliberately separate palettes

These are distinct vocabularies and are never interchanged:

* **geometry palette** — `provider.palette_for_states(reference_states,
  include_partial=True)`. Truth-owned exact geometry, rendering, occlusion and
  the `visible_cells` support. Presence here says nothing about scoring.
* **canonical scoring palette** — `provider.canonical_palette()`, the same
  full-cube production vocabulary the pinned benchmark scores against. Patches
  are scored only against this. A truth state with exact geometry but no
  canonical membership is *geometry-labelable* yet *out-of-vocabulary* for
  scoring: it is counted in `truth_out_of_vocabulary_patches` and can never be an
  oracle candidate or enter `truth_in_vocabulary_patches`.

### Frozen observation weight (exact)

For every accepted face the frozen weight is byte-for-byte the
`raycast.reconstruct_scene_surfaces` production weight, **including** the
quadrant-uniformity divisor that attempt 01 omitted:

```
weight = area**raycast.AREA_POWER / (1 + depth/16)
if raycast.VISIBILITY_POWER:  weight *= visible**raycast.VISIBILITY_POWER
if raycast.OBLIQUITY_POWER:   weight *= facing**raycast.OBLIQUITY_POWER
weight /= 1 + raycast._face_uniformity(observed_patch, coverage) / raycast.UNIFORMITY_SCALE
```

`observed_patch` is the production one: unowned pixels replaced by the face's
coverage-weighted mean colour before the quadrant gap is measured. Current
production values: `AREA_POWER = 0.5`, `VISIBILITY_POWER = 0.0`,
`OBLIQUITY_POWER = 0.0`, `UNIFORMITY_SCALE = 1e9`, `TEXTURE_WEIGHT = 40.0`. The
applied `uniformity_scale` and the full formula string are recorded in the
artifact; the default appearance texture weight is `raycast.TEXTURE_WEIGHT`
(40.0).

Exact uniformity value for the synthetic parity fixture used in the tests
(two-by-two quadrant patch `100 / 120 / 140 / 160`, full coverage):
`_face_uniformity = 25.98076211353316`, so the divisor is
`1 + 25.98076211353316 / 1e9 = 1.0000000259807622` and the exact weight is
`8.421052412793587` against the scale-omitted `8.421052631578947`
(relative difference `2.598e-08`). The divisor is real but currently tiny under
the production `UNIFORMITY_SCALE`; it is applied exactly either way, and a
future scale change is recorded from `raycast` rather than assumed.

For every sampled face the pixel authority is the renderer's owned, GUI-masked
domain. The frozen record keeps the rectified `patch` (16x16x3 float32), the
boolean `valid` mask, the fractional `coverage`, the truth `BlockState`, the
frame/coordinate/slot key and the observation weight above.

Transparent, cut-out, partially occluded or HUD-masked pixels are **excluded
per pixel** by `valid`/`coverage`, and a face with any excluded pixel is marked
`partial`. A face with `valid_fraction < 0.5` is refused with
`insufficient_valid_coverage` rather than frozen as a mostly-unobserved patch; a
fully masked face is refused with `gui_mask` / `no_valid_pixels`. Every
abstention reason, per frame and pooled, is serialized.

No source screenshot pixels, camera image, or unrelated frame content are
serialized: the frozen domain is exactly the rectified patch plus its masks, and
the harness refuses any other shape.

## 4. Frozen artifact, determinism and sealing

The sealed artifact (`mc-photogrammetry-texture-patch-labels-v1`) carries:
schema/role/mode; the full reviewed-pin verdict and observed digests; input
hashes (reference, camera manifest, scene, JAR, frames, masks) with their
`pinned_sha256` values and `asset_pins_verified = true`; the sampler parameters
and exact weight policy; the four denominators below; mask/visibility/patch
coverage and every abstention reason; the deterministically ordered frozen
patches; the existing-scorer read; the reconciliation record; and a self-seal
`artifact_sha256` computed over the canonical JSON with that field removed.

Order is stable: patches are sorted by `(frame_id, coordinate, slot)` and a
duplicate polygon on the same cell face is collapsed deterministically to the
most fully observed sample, with the duplicate count recorded. The artifact
contains no timestamps, so two runs over identical inputs must be byte-identical.

The harness refuses to overwrite an existing artifact (checked before preflight
and again immediately before the atomic replace), and refuses to seal a patch
domain that is not exactly the frozen 16x16x3 + mask shapes.

## 5. Metric definitions and distinct denominators

Four denominators are computed and reported separately; they are never
interchanged:

* `all_visible_cells` — reference cells reached by at least one usable pixel
  across any frame under `visible_cells(..., minimum_pixels=1)` with the reviewed
  masks. This is the pinned all-visible domain (T6 2,007; T2 667).
* `scorer_usable_cells` — visible cells with at least one frozen patch on which
  the existing typed scorer did not abstain.
* `geometry_labelable_cells` — the subset of `all_visible_cells` whose truth state
  carries exact baked model polygons in the **geometry palette**. This is the
  geometry-metadata denominator, renamed from attempt 01's misleading
  `model_vocabulary_cells`; it is *not* a scoring-vocabulary count.
* `model_vocabulary_cells` — the subset of `all_visible_cells` whose truth state
  is a member of the **canonical scoring palette** (`canonical_palette()`). This
  is the corrected scoring-vocabulary denominator, matching the pinned
  benchmark's truth-in-canonical-palette domain.

Scorer rows: `scored_patches` (abstention-free frozen patches),
`truth_in_vocabulary_patches` and `truth_out_of_vocabulary_patches` are reported
separately; `ordered_top_k` at K = 1/8/16/32 divides top-k membership only by
`truth_in_vocabulary_patches` and is explicitly an **ordered recall** of this
frozen reference-only sample. It is not the production `np.argpartition`
set-membership trace, and `production_set_membership_claimed` is always `false`.

Patch accounting: `frozen_patches`, per-frame and per-slot and per-state counts,
`labeled_cells`, `partial_observations`, duplicate-face count, and the pooled
abstention histogram (`model_metadata_unavailable`, `gui_mask`,
`visible_fraction`, `minimum_area`, `off_frame`, `behind_camera`,
`insufficient_valid_coverage`, `no_valid_pixels`, …).

## 6. Historical-row reconciliation gap (explicit stop)

The published rows this POC must eventually compare against are:

| Row | Value | Source metric | Denominator |
|---|---|---|---|
| T6 production top-8 set membership | 788/2007 (39.26%) | `argpartition` set membership | all-visible 2007 |
| T6 ordered K=8 | 793/1920 (41.30%) | ordered recall | truth-in-canonical-palette 1920 |
| T6 ordered K=16 | 850/1920 (44.27%) | ordered recall | truth-in-canonical-palette 1920 |
| T6 ordered K=32 | 932/1920 (48.54%) | ordered recall | truth-in-canonical-palette 1920 |

Those rows are read post hoc from a `reconstruct_scene_surfaces`
reference-assisted stage trace. The frozen sample here is the reference-assisted
exact-model sampler subset with independently computed denominators. Sample
identity and denominator equality with that trace are **unproven from these
exact patches**, so the harness records a typed denominator/interface gap
(`reconciliation.status = denominator_interface_gap_recorded`,
`claimed_equal = false`) and claims no reconciliation.

The precise remaining non-equivalence to the production per-cell pipeline is:

* **Ranking granularity.** This harness reports a **per-patch ordered ranking**
  (each frozen patch ranked independently, its truth's position read off
  cheapest-first), not the production per-cell aggregation. Production sums each
  observation's relative cost vector into a weighted per-coordinate total
  (`_accumulate_costs`) and decides one state per cell.
* **Pruning.** Production prunes the palette by support and locality
  (`palette_support`, `palette_prune_rounds`, `local_support_*`) and prunes
  candidates per cell; this harness scores against the whole canonical palette.
* **Selection rule.** Production top-K uses `np.argpartition` set membership
  (`_top_candidates(..., keep=8)`); this harness uses an ordered sort and
  `ordered_top_k`. Ties at `OUTLIER_CEILING` therefore resolve differently.
* **Smoothing and placement.** Production applies adjacency smoothing
  (`_smooth_adjacent_states`) and placement (`_remove_enclosed_blocks`, placement
  passes) after scoring; this harness performs neither.
* **Scoring inputs.** The typed per-observation scorer read here applies the
  canonical palette with `texture_weight = raycast.TEXTURE_WEIGHT` (40.0) but
  still does not reproduce production's per-view calibration/light inversion,
  self-calibration, simplicity, shape weights or per-view weighting.

**Harness correction (2026-10-03; before any corpus run).** Lead review found
that `score_patches()` omitted `use_pattern=True`, so the appearance scorer's
default silently disabled its texture-pattern term despite the nonzero weight.
The harness now explicitly enables that term and records `use_pattern: true` in
its scorer result. A synthetic regression verifies that the pattern changes the
rank of a same-colour pair. This repairs the scorer invocation; it does not make
the patch-level read equivalent to production's per-cell scoring pipeline, and
no T2/T6 corpus result is implied. Focused verification: 37 tests passed; both
owned Python files compile.

So the two metrics stay distinct: `ordered_top_k` is a deterministic diagnostic
ranking of this frozen reference-only sample, while production membership follows
per-cell aggregation plus `argpartition`.

**Stop condition:** if a future E1 run cannot show that these exact frozen
patches reach the historical rows on the same denominator, record the specific
denominator/interface gap and stop. Do not train, do not alter source modules to
force agreement, and do not treat the frozen reference-only sample as the
production top-K trace. A lead-authorized E1 corpus run is required first.

## 7. Caveats

* **Region clipping.** Export ground truth omits geometry outside the declared
  region, and every factory camera stands outside it, so an unexported occluder
  between camera and region can still attribute a pixel to a region face. The
  harness cannot detect this; the E1 run must quantify it (audit §1.4) before any
  training, and per-pixel `valid`/`coverage` is the only documented mitigation.
* **Partial and cut-out geometry** are refused unless exact polygons exist, so
  the labelled subset is exact-model-only, not all-visible.
* **Assisted mode.** These are supplied-camera, reference-assisted diagnostics.
  Nothing here is image-only performance, a production qualification, or a
  generalization claim.
* **No training data.** T2/T6 are evaluation builds; the frozen patches must not
  be used as model training data.
* **Fixture scope.** Only Template 2 and Template 6 have `reviewed: true` pins;
  `frozen-small-template-3` remains unpinned and inadmissible.

## 8. Commands

Required checks for this attempt (all run from the repository root):

```powershell
.venv\Scripts\python.exe -m py_compile scripts/evaluation/evaluate_texture_patch_labels.py tests/test_texture_patch_labels.py
.venv\Scripts\python.exe -m pytest tests/test_texture_patch_labels.py -v --no-header
git status --porcelain=v1 --untracked-files=all   # diff-scope: only the three owned paths
```

Lead-authorized E1 commands, executed successfully after R3. The first T6 attempt
before R3 refused on the `.png.json` mask-path defect before image decode/output;
the corrected runs were:

```powershell
$env:PYTHONPATH = 'src'
.venv\Scripts\python.exe scripts/evaluation/evaluate_texture_patch_labels.py template-6 `
  .tmp-texture-poc-e1-patch-labels-20261003-template-6.json `
  --minecraft-jar "$env:APPDATA\.minecraft\versions\1.21.5\1.21.5.jar"
.venv\Scripts\python.exe scripts/evaluation/evaluate_texture_patch_labels.py template-2 `
  .tmp-texture-poc-e1-patch-labels-20261003-template-2.json `
  --minecraft-jar "$env:APPDATA\.minecraft\versions\1.21.5\1.21.5.jar"
```

## 9. Focused verification performed

* `py_compile` of both Python paths: passed.
* Original harness attempt: `pytest tests/test_texture_patch_labels.py`: **34 passed**. Coverage:
  exact state/face label derivation; mask (HUD) abstention; partial-geometry
  refusal without cube labelling; conservative occlusion by a refused cell so an
  occluded exact face is not labelled; cut-out partial observation with excluded
  pixels; sparse-ownership refusal; unrelated-image-pixel rejection; exact
  production observation weight including the uniformity divisor and the
  production-pipeline parity transcription; canonical scoring palette defaults to
  `raycast.TEXTURE_WEIGHT` and is overridable; a truth state absent from the
  canonical scoring palette is counted out-of-vocabulary and can never be an
  oracle candidate; geometry-presence outside the scoring vocabulary is not
  promoted; existing-scorer read with separate denominators; deterministic and
  order-independent seal; tamper detection; frozen-pixel-domain enforcement;
  overwrite refusal; pin-manifest schema and mismatch/`reviewed` fail-closed
  checks; matching camera/scene/JAR asset pins accepted; camera, scene and JAR
  asset-pin mismatches each refused; a role with no declared asset pin refused;
  no output left behind on a camera/scene/JAR asset-pin mismatch; mask-spec
  domain and bounds; unknown-role refusal; no-output-on-preflight-failure;
  existing-output refusal.
* R2/R3 lead verification: `py_compile` passed; `pytest tests/test_texture_patch_labels.py`: **39 passed**.
  Synthetic tests confirm the pattern term changes same-colour state ranking; the
  mask mapping handles timestamp dots and missing specs still refuse.
* Corpus outcome: both reviewed roles passed preflight and completed. Their sealed
  artifacts and exact metrics are recorded in the linked E1 results report.
* `diff-scope`: only the three owned paths were edited by this attempt; all
  pre-existing dirty paths (including `src/` and unrelated untracked build
  output) were preserved untouched.

Code hashes (SHA-256):

* `scripts/evaluation/evaluate_texture_patch_labels.py` —
  `EE06B23D054775EDE76DECCC65FD6A6770B30D737D38AFD1097CB6E331CB41A1`
* `tests/test_texture_patch_labels.py` —
  `636F15436954C90C6CB253C1DE827092DEAFD4D0AE16A3B82D36F541922B6DD2`

## 10. Handoff

E1 harness and the two authorized R4 runs are complete. The visible-cell domain
now matches the production denominator; patch-level rows still do not reconcile
to production per-cell scorer rows. Do not train on these evaluation fixtures.
Follow the production-observation replay and stop conditions in the
[results report](texture-poc-e1-results-2026-10-03.md).
