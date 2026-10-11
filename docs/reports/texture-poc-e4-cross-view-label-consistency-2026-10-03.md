# E4 cross-view factory patch-label consistency audit (evaluation-only)

- Task ID: `texture-poc-e4-cross-view-label-consistency-2026-10-03`
- Worker: DeepSeek (Sol-equivalent) under the direct-shell Defender-safe profile
  in [docs/deepseek-worker.md](../../docs/deepseek-worker.md)
- Status: implemented and analysis-only read complete; lead-requested two-pass
  admission remediation applied; **stopped for lead review**
- Verdict: **no quality or correctness claim is made.** This report measures
  repeated-key patch agreement (a consistency proxy) and nothing else.

## 1. Scope and owned paths

Owned and edited paths (only these were written):

| Path | Role |
|---|---|
| `scripts/evaluation/audit_factory_texture_patch_consistency.py` | auditor |
| `tests/test_audit_factory_texture_patch_consistency.py` | focused synthetic tests |
| `docs/reports/texture-poc-e4-cross-view-label-consistency-2026-10-03.md` | this report |

One ignored analysis output was written under `exports/*` (git-ignored):
`exports/texture-poc-e4/cross-view-label-consistency-2026-10-03.json`.

The auditor reads **only** the four accepted E3 artifacts named by the task. No
raw screenshot, reference Litematic, camera manifest, scene definition, JAR,
T2/T6 fixture or any other input was read or hashed. No production or runtime
source, exporter or test outside the two owned paths was touched; the exporter
was not rerun; no model was trained, no dependency installed, no client or
Minecraft automation started, and no file was committed.

## 2. Method

### 2.1 Seal verification before any array decode

Loading is a strict two-pass admission, not a single verify-and-decode loop:

**Pass 1 — admission (no NumPy array is built).**

1. the artifact-level self-seal `artifact_sha256` is recomputed over the
   canonical JSON of the record without that field;
2. `schema`, `engine_version`, `artifact_id`, `evaluation_only == true`,
   `training_permitted == false` and `t2_t6_excluded == true` are required;
3. the declared grouping is cross-checked against the artifact's own examples
   (`grouping_key`, `fold_key`, per-`scene_group` fold and `example_count`);
4. every example record is validated: the per-example seal
   `digests.example_sha256` is recomputed, all three array records (`patch`,
   `valid`, `coverage`) are checked for their exact frozen dtype/shape, byte
   length, own `sha256` and matching per-example digest, and all non-array
   metadata (build/provenance digests, slot, normalizable exact state,
   coordinate, `scene_group`, fold, ids) is validated by
   `validate_example_fields` — which touches no array.

**Pass 2 — decode.** Only after Pass 1 has succeeded for *every* example in the
artifact does `_observation_from_example` decode `patch`/`valid`/`coverage` into
typed arrays. A tampered late example therefore refuses before the first NumPy
array is constructed; `test_admission_verifies_every_seal_before_decoding_any_array`
pins this by monkeypatching `_decode_array` to record calls and asserting the
list stays empty on a refusal.

Any mismatch is a typed refusal (`TexturePatchConsistencyError`); the audit
produced zero seal failures on the four accepted artifacts (section 4).

### 2.2 Repeated exact-face key

One *observation* is
`(scene_group, world_identity, world_coordinate, slot, exact BlockState)`.
Exact state identity is the normalized `(name, properties)` record; property
order is canonicalized, so two spellings of the same state are one key. Only
keys with at least two observations are compared; keys seen once are counted but
never paired.

### 2.3 Duplicate-image suppression and pair strata

Inside one key, observations are ordered deterministically
(`frame_key`, `screenshot_sha256`, artifact, `frame_id`, coordinate) and reduced
to one representative per frame and then one per distinct `screenshot_sha256`.
Raw unordered pairs are stratified **before** suppression:

- `same_frame_pairs` — the same frame identity twice;
- `duplicate_image_pairs` — different frames carrying identical screenshot bytes;
- `distinct_capture_pairs_raw` — everything else.

Only pairs among the deduplicated representatives are measured, so a captured
image repeated in a second artifact can never be counted as an independent
sample, and `suppressed_duplicate_pair_equivalents` records exactly how many raw
distinct-capture pair equivalents the dedup removed. Captures are never split
independently.

### 2.4 Pair metric

A pair is measured on the **shared valid pixels** — the intersection of the two
frozen ownership masks — so a pixel one capture does not own cannot enter the
comparison. A pair with fewer than `MINIMUM_SHARED_VALID_PIXELS = 16` jointly
owned pixels (one sixteenth of the 16x16 patch; a fixed coverage floor, not a
fitted threshold) is recorded as `insufficient_shared_valid_pixels` and
contributes no error value. Two metrics are reported in native `0-255` float32
patch units:

- **raw RGB MAE** — `mean |A - B|` over the shared valid pixels and channels;
- **brightness-normalized RGB MAE** — both patches are rescaled so their
  shared-support mean RGB brightness equals the pair mean, then the same MAE is
  computed. A pair whose shared support is black in either patch
  (`brightness <= 1e-6`) keeps its raw value and is recorded as
  `zero_brightness` for the normalized metric.

No threshold is fitted and no pass/fail bar is defined.

## 3. Inputs (as read)

All four artifact self-seals and all `8516` example seals and `25548` array
seals verified.

| Artifact file | file SHA-256 | internal `artifact_sha256` | examples |
|---|---|---|---|
| `exports/texture-poc-e3/fresh-c4-canal-town-2026-09-12.json` | `9ec453beb82ee5a64e07074275e9142b3cd43e0e0fd994848ee9bd79d61cfef4` | `fb737108a659ed377ea201c66f69fcdf3ae46d224c99f651c9bfbc7524135fa7` | 1,303 |
| `exports/texture-poc-e3/fresh-c4-canal-town-framed-2026-09-12.json` | `206a92219923ccefbf2b3286d5282372bd0680b658b4df1f2979cf28fa04986a` | `97a3a64c06cf811e85ea9e0796de3a6348c10f5eb3324db17dc953af047537b0` | 2,333 |
| `exports/texture-poc-e3/pilot-batch-train-station-2026-09-12.json` | `a04b2db2dbca3503be55e6ff62740c22152c41f062e235f9c83acdb751f32114` | `1a469f6913b2fe3850bca0ea13fdf627cd367e952f4b4154c8667ca239694965` | 2,507 |
| `exports/texture-poc-e3/smoke-train-station-ds20c-2026-09-12.json` | `f71cce99ae0a819edf62ce3f0c9611925be8a9bb5a1d60cc5197c95d49313e44` | `fae14c754b1e3523c5838a3f6f6c7403a7d4cb0bae08a7aa7cff2d68329a4aaa` | 2,373 |

Builds and folds: two `scene_group`/build pairs (Canal Town `74f080aa…`, Train
Station `1757a5ad…`); every scene is `train`. The four artifacts are the four
accepted E3 artifacts; nothing else under `exports/texture-poc-e3/` was globbed
or read.

## 4. Deterministic denominators

| Quantity | Value |
|---|---|
| artifacts / examples / builds | 4 / 8,516 / 2 |
| unique frames / unique screenshots | 17 / 17 |
| exact keys | 5,307 |
| keys seen once (never paired) | 3,298 |
| repeated keys (≥2 observations) | 2,009 |
| keys with ≥2 distinct captures | 2,009 |
| observations inside repeated keys | 5,218 |
| raw unordered pairs | 5,223 |
| same-frame pairs | 0 |
| duplicate-image pairs | 0 |
| distinct-capture pairs before suppression | 5,223 |
| **analyzed distinct-capture pairs** | **5,223** |
| suppressed duplicate pair equivalents | 0 |
| pairs below the 16-pixel floor | 0 |
| `zero_brightness` normalized exclusions | 0 |
| shared valid pixels (min / median / mean / max) | 112 / 240 / 231.56 / 256 |
| per-observation patch valid fraction (min / median / mean) | 0.512 / 0.977 / 0.950 |

Every one of the 5,223 pairs is a genuine distinct-capture pair inside one
build; no pair mixes builds or exact keys. The real corpus contains **no**
duplicate screenshots and no repeated frame identities inside a key, so the
dedup path is exercised only by the synthetic tests; that is stated rather than
implied.

## 5. Results

### 5.1 Overall (5,223 analyzed pairs)

| Metric | min | median | mean | p90 | max |
|---|---|---|---|---|---|
| raw RGB MAE | 0.515 | 9.071 | 15.556 | 38.302 | 194.308 |
| brightness-normalized RGB MAE | 0.314 | 8.630 | 11.140 | 21.576 | 102.335 |

Brightness normalization lowers the pooled mean by ~28% and the p90 by ~44% but
barely moves the median (9.07 → 8.63). Read descriptively: part of the raw
disagreement is a global exposure/lighting scalar, and part is not.

### 5.2 Per group

| Group (build) | examples | keys | repeated keys | obs in repeated keys | analyzed pairs |
|---|---|---|---|---|---|
| Canal Town `74f080aa…` | 3,636 | 1,409 | 1,027 | 3,254 | 4,241 |
| Train station `1757a5ad…` | 4,880 | 3,898 | 982 | 1,964 | 982 |

| Group | raw MAE median / p90 / max | normalized MAE median / p90 / max | shared px min / median |
|---|---|---|---|
| Canal Town | 9.242 / 26.446 / 194.308 | 9.003 / 21.690 / 102.335 | 112 / 240 |
| Train station | 5.714 / 50.310 / 84.729 | 3.357 / 21.339 / 50.228 | 162 / 239 |

Canal Town's repeated keys are genuinely multi-view (up to 9 observations per
key, 357 keys with 2 … 1 key with 9), while Train station's 982 repeated keys
are exactly 2-observation pairs in the two accepted captures. The two groups
therefore answer different questions: Canal Town gives a broad multi-view
spread, Train station a narrower two-capture comparison. Neither is a
generalization claim.

### 5.3 Slot strata (analyzed pairs, raw MAE median)

| Slot | all pairs | all raw median | Canal Town pairs / median | Train station pairs / median |
|---|---|---|---|---|
| `up` | 2,383 | 10.152 | 2,073 / 9.717 | 310 / 36.683 |
| `east` | 1,277 | 5.237 | 821 / 7.409 | 456 / 2.078 |
| `south` | 621 | 13.005 | 579 / 11.974 | 42 / 34.020 |
| `west` | 552 | 9.705 | 378 / 7.898 | 174 / 28.873 |
| `north` | 390 | 8.555 | 390 / 8.555 | — |
| `down` | 0 | — | — | — |

### 5.4 State strata (largest contributors, raw MAE median per exact state)

Canal Town: `cobbled_deepslate` 2,001 pairs / 7.850, `cobblestone` 767 / 11.577,
`smooth_quartz_slab` 336 / 13.900 plus 11 / 2.381, `smooth_quartz` 114 / 3.793,
`cobblestone_stairs` 199 / 18.169 and 104 / 14.512, `cobbled_deepslate_stairs`
67 / 12.508, 49 / 11.199, 43 / 9.394 and 36 / 10.217, `white_concrete` 34 /
37.049, `stone_brick_slab` 4 / 43.617.

Train station: `dirt` 416 / 2.569, `grass_block` 112 / 7.757, `gravel` 110 /
43.218, `bricks` 68 / 14.576, `green_concrete_powder` 42 / 26.147,
`dead_brain_coral_block` 15 / 35.117, `dead_horn_coral_block` 24 / 37.420,
`dead_tube_coral_block` 29 / 37.645, `dead_fire_coral_block` 1 / 3.407,
`cobblestone` 28 / 45.977.

State strata are keyed by the **exact** state (including properties), so one
block name can appear once per property set (for example the stair facing
variants above); the counts above are per exact-state key, not summed by name.
The full per-state and per-state-and-slot tables, and every individual pair
record, are in the ignored JSON output.

## 6. Interpretation and limitations (no verdict)

- **Consistency is not correctness.** Two captures of a consistently occluded or
  mislabelled face agree by construction and produce a low MAE. A low MAE cannot
  distinguish "both captures saw the same true face" from "both captures saw the
  same wrong region".
- **Region-clipping / occluder contamination is invisible here.** The shared
  valid mask is the intersection of the frozen ownership masks; if both captures
  clip the same neighbour, the clipped pixels are excluded from both and the
  remaining overlap can still match. This audit cannot detect that class.
- The raw median of ~9/255 and normalized median of ~8.6/255 describe how much
  repeated exact-face pixels differ under the production shrink=1.0 sampler and
  the recorded factory captures. They are **not** an accuracy, precision or
  quality score, and no threshold was chosen to separate "good" from "bad".
- Only two builds exist and every scene is `train`; no dev/test capture group
  exists. Nothing here supports a generalization claim, and no capture was split
  independently.
- 3,298 exact keys (62% of keys) are seen by a single capture and are outside the
  comparison domain entirely.
- The metric is computed at the E3 patch domain (16×16×3) and inherits every E3
  seal/refusal semantic; no source screenshot pixels were re-read.

## 7. Checks run

Remediation pass (two-pass admission), all on the three owned paths:

- `.venv\Scripts\python.exe -m py_compile scripts/evaluation/audit_factory_texture_patch_consistency.py tests/test_audit_factory_texture_patch_consistency.py`
  → exit 0.
- `.venv\Scripts\python.exe -m pytest tests/test_audit_factory_texture_patch_consistency.py -o addopts= -q --no-header`
  → **24 passed** (23 before, +1 two-pass regression).
- Deterministic metrics output: re-ran the auditor over the four named E3
  artifacts to the existing ignored path
  `exports/texture-poc-e4/cross-view-label-consistency-2026-10-03.json`. The
  output is byte-identical to the pre-remediation run —
  **`C1681C86D432F0BCB6AAFA2CCF92B3CCCE7C2FD7007624CD39B5ADF2F26FBCF0`** — and
  the stdout summary is unchanged: 4 artifacts, 8,516 examples, 2 builds, 5,307
  keys, 2,009 repeated keys, 5,223 analyzed pairs, 0 seal failures. This
  confirms the refactor is output-neutral for the audit payload.
- Trailing-whitespace scan on both new files: clean.

Original build pass (retained for provenance; not rerun during remediation):

- Regression on the untouched exporter suite
  `pytest tests/test_export_factory_texture_patch_dataset.py -o addopts= -q --no-header`
  → **41 passed**.
- Determinism: two consecutive CLI runs to distinct ignored paths produced
  byte-identical JSON (`C1681C86D432F0BCB6AAFA2CCF92B3CCCE7C2FD7007624CD39B5ADF2F26FBCF0`).
- `git status` shows only the two new untracked files; the ignored JSON output is
  not tracked.

## 8. Hashes

| Artifact | SHA-256 |
|---|---|
| `scripts/evaluation/audit_factory_texture_patch_consistency.py` | `CBA5195FE4F22C15667A74CF56B79AD839316F89B9910E3C499F90CE7668CF3F` |
| `tests/test_audit_factory_texture_patch_consistency.py` | `63C33B4AB3181CBFF3EE4B6A714A6EF5053F118EB76DD8CE89C6E3E7BD6ACF25` |
| ignored audit JSON (unchanged) | `C1681C86D432F0BCB6AAFA2CCF92B3CCCE7C2FD7007624CD39B5ADF2F26FBCF0` |

## 9. Residual complexity / for lead review

1. `pair_records` serializes every analyzed pair (~5,200 records, ~7 MB JSON) so
   the distributions are independently recomputable; if the lead prefers a
   compact payload, the per-pair array can be dropped behind a flag without
   changing any aggregate.
2. The brightness normalization uses the pair's arithmetic-mean shared support
   brightness as the common target. It is documented and deterministic; a
   different scalar definition would change the normalized numbers.
3. The 16-pixel coverage floor is a fixed geometric floor, not a tuned value; on
   this corpus it excludes nothing (minimum observed overlap is 112 pixels). It
   is configurable only to make the refusal testable.
4. This audit measures label/pixel consistency only. It does **not** establish
   that the repeated exact faces are correct, and it should not be used as a
   training-data quality gate. No next experiment is proposed here; the lead
   owns any follow-up.
5. Stop condition reached: no further action taken. No self-acceptance.
