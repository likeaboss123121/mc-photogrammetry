# M4 V2-B — split-phase calibration readiness inventory (2026-09-14)

Status: **documentation-only readiness inventory; stopped for lead review.** No P0 or
V2-B implementation, no calibration fit, no control generation, no label creation or
scoring, no placement/wiring, and no code/test/contract/handoff/comms edit was made.
Owned path: this file only.

This inventory answers the standing assignment in `AGENT_HANDOFF.md` §6 ("Inventory
and execute the frozen split-phase calibration gate using evaluation-only
fixtures/tools; do not implement V2-B or product wiring"). It stops at the inventory;
execution of the gate is a separate, explicitly authorized work package (§9).

## 0. What was read and what was run

Read (all paths verified to exist in §1 and §11): the two frozen contracts
(`docs/reports/M4-V2B-P0-split-phase-producer-contract-2026-09-14.md`,
`docs/reports/M4-V2B-physical-phase-anchor-contract-2026-09-14.md`), the accepted P0
implementation report and source/tests, the P1-A/P1-B replay surface, the dataset
factory, the frozen-image runner and frozen-plane evaluator, and the current dataset
factory output directory `exports/datasets/`.

Run (label-free only, no label was generated, opened, or scored):

| Command | Result |
| --- | --- |
| `.venv\Scripts\python.exe -m pytest tests\test_split_phase_producer.py -q -p no:cacheprovider -o addopts=` | **23 passed** in 0.26 s (reproduces the accepted implementation report §4) |
| a read-only probe timing `produce_split_phase_evidence` + `validate_split_phase_evidence` on synthetic `float64` patches (no labels) | table in §8.2 |

Not run: the accepted broader suites (197 / 179 passes), any calibration, any control
synthesis, any B/I computation, any label comparison, any dataset capture or
finalization. The broader-suite numbers in §8.1 are the accepted report's, not
re-measured here.

Created: this file only.

## 1. Authority and hash basis (all verified in place)

| Artifact | SHA-256 (lowercase) | Lines | Role |
| --- | --- | --- | --- |
| `docs/reports/M4-V2B-P0-split-phase-producer-contract-2026-09-14.md` | `f7fa9295196eb02b284416934c50c4b895438d8eadb0bdb5353863e5e8f79172` | 171 | frozen P0 contract (calibration §7, gate §9, tests §10) |
| `docs/reports/M4-V2B-physical-phase-anchor-contract-2026-09-14.md` | `f7c336e848bd3c082c5a1be914c91b222f1fb611b658269518497f275426787e` | 214 | frozen V2-B contract (B/I §7, calibration prerequisite §11, controls §13) |
| `docs/reports/M4-V2B-P0-split-phase-producer-implementation-2026-09-14.md` | `64d5b4136b62d5ef1eb2e35d2799432be9bdde7db8b1971d27db088fdadc117d` | 228 | accepted P0 implementation/report (revision 2) |
| `docs/reports/M4-P0-endpoint-spacing-2026-09-13.md` | `7bb4f730d8094264a6e49a4878fea7d09ca8bb350b153b6e1b92bba2bab02041` | 26 | accepted endpoint-inclusive spacing (distinct, not superseded) |
| `src/mc_photogrammetry/reconstruction/lattice_planes.py` | `82ada0ccc0f92b580342aa35f8333d1013910f7cfa91eada0ed7d5e7c95831e7` | 3807 | P0 producer host + calibration data contracts |
| `tests/test_split_phase_producer.py` | `e5e17b9ccc8203394ebe1f9f78601176f33c041f34620f214d1b02457b7a9b31` | 630 | accepted P0 production controls |

Accepted-authority table in the V2-B contract §0 re-verified unchanged (full digests):

| Authority | SHA-256 (lowercase) |
| --- | --- |
| `docs/reports/M4-V2-cell-placement-contract-2026-09-13.md` | `9e1d636d73063a0642ec6950dcee249906ed9f79373ddf5bb4e16b2ec1cbd822` |
| `docs/reports/M4-V2A-public-multicandidate-gate-2026-09-14.md` | `5e118ebfce0954eb1ec961e8d7aaed4348a31d2587d462ecc6b105a3104d8b65` |
| `docs/reports/M4-P1-C2R-B-cycle-closure-contract-2026-09-13.md` | `8929b19c3be1f729c20bc2be5849df3f7ca54dfc25fd5fa528ef96bc19492ce5` |
| `docs/reports/M4-P1-C1-axis-and-support-contract-2026-09-13.md` | `7b6c2fe99322808edbbe9f4cfc3007eda346025b6e559830644b1905c5dc3e54` |
| `docs/reports/M4-P1-B-relative-residue-candidates-2026-09-13.md` | `9bec52b3f986bd7a08943d3eacf32ee73243534845edfdfc3814a5c1337061fe` |
| `docs/reports/M4-P1-A-periodic-phase-and-cell-shift-2026-09-13.md` | `ab084df0b1b140204ec011db5e279937d7fa7be5243dcd53e98465e2ca91bec2` |
| `docs/reports/M4-V1-region-provenance-footprints-2026-09-13.md` | `09497c2600ddad1355ca9aefbd3c6276fc6f3a2c20c3ff46205c8e185df9a0a7` |
| `docs/reports/DEEPSEEK-M4-V2-placement-prerequisite-audit-2026-09-13.md` | `e2b2205863d283f946c930f32485b821b4ff210f2d10ac9fcf253e79852ed602` |

Frozen constants re-derived from the source (not quoted from a report): built-in
`SPLIT_PHASE_PARTITION_CONFIG_SHA256 = b2225d8f2ca16bf01189826870511e025e80c108ac3c1122c57ed4f7ba6b214f`
(matches the P0 implementation report §5), `SPLIT_SEAM_PROFILE_PAYLOAD_VERSION` is
21 bytes, `SPLIT_PHASE_REFUSAL_REASONS` has exactly 25 members, `_SPLIT_BUDGET_FIELDS`
has exactly 11 caps.

## 2. What calibration machinery already exists

### 2.1 P0 data contracts (implemented; schema-only fitting)

`lattice_planes.SplitPhaseCalibrationTuple`, `SplitPhaseCalibrationGrid` and
`SplitPhaseCalibration` exist as frozen slotted dataclasses with canonical
`to_record()` and self digests (`_split_digest`). The P0 implementation report §6 is
explicit that the empirical fitting/promotion run was **not** performed, and
`produce_split_phase_evidence` has no calibration, grid, tau, hypothesis or label
parameter (verified from the signature at `lattice_planes.py:3740`). The run-time
producer cannot see calibration by construction. This is the correct starting state
for the gate.

### 2.2 P0 split evidence a calibration harness can obtain today

From a `complete` `SplitPhaseProducerResult` (built-in config, `g = 0`, `L = 4` lane
sets):

- `generation_evidence[0]`: the 20-field G-only record — `phase_uv_pixels`,
  `response_magnitude_uv`, `profile_mass_uv`, `concentration_uv`,
  `unavailable_reason_uv`, `g_profile_sha256`, `partition_sha256`,
  `G_u_sha256`/`G_v_sha256`, `G_u_sample_count`/`G_v_sample_count`,
  `generation_evidence_sha256`;
- `seam_profile_evidence` (ordered `u`, then `v`): the 16-field held-out `D` record —
  `profile_samples_float64_le_base64` (`E`), `profile_weights_float64_le_base64`
  (`C`), `axis_bin_indices_int64_le_base64` (`0..n-1`),
  `sample_count == weight_count == index_count == n` (`n = W` for `u`, `n = H` for
  `v`), `coverage {total, valid, per_bin}`, `payload_sha256`, `d_profile_sha256`,
  `D_a_sha256`, `seam_evidence_sha256`;
- the coordinate-set digests `G_a_sha256`/`D_a_sha256`/`X_a_sha256`, the six-input
  `producer_input_sha256`, `planned_work`/`runtime_work` (11 counters each) and
  `result_sha256`.

The P0 result **does not** carry boundary/interior energy, `B(h)`/`I(h)`, `tau_band`,
`tau_margin`, `tau_energy`, a hypothesis, a translation, the P1-B
`leading_cell_coordinate` (`L`), or the downstream `selected_b`. Those are V2-B /
provenance values (§7 of this report).

### 2.3 P1-A / P1-B replay surface (read-only; exists but is not wired to P0)

- `lattice_planes._periodic_energy_axis`, `_periodic_energy_measurements`,
  `_periodic_energy_phase_evidence`, `PeriodicEnergyPhaseEvidence` (P1-A);
- `plane_residue.PeriodicResidueCandidates.leading_cell_coordinate`,
  `plane_residue.HypothesisPhaseShiftEvidence`, `hypothesis_phase_shift_evidence`
  (P1-B);
- `plane_graph._AXIS_TRANSFORMS` / `axis_transform(local_axis, quadrant)` (world
  axis/sign table);
- `plane_endpoint_provenance.ResidueAxisDomain`/`candidate_values` (V2-A) retains
  `selected_b`.

None of this is an independent forward oracle and none of it computes B/I.

### 2.4 Synthetic fixtures that exist today (test-local only)

| Fixture | What it supplies | What it does not supply |
| --- | --- | --- |
| `tests/test_split_phase_producer.py` `_patch(seed=7)` | i.i.d. half-normal `float64` `12×10` patch; textureless zeros patch; per-axis and `D_u ∩ D_v` leak controls; budget boundary matrix | no declared seam, no `L`/`selected_b`, no hypothesis set, no label |
| `tests/test_plane_residue.py` `_phase_evidence`/`_reading`/`_match` | synthetic P1-A phase evidence, `PlaneReading`, `PlaneMatch`, complete rational candidate sets | no split patch, no P0 producer call, no D profile, no declared world seam |
| `tests/test_lattice_planes.py` (`test_periodic_energy_phase_evidence_uses_u_columns_v_rows_and_zero_is_measured`, `…explicit_unavailable_reasons…`) | synthetic periodic-energy comb patch and P1-A boundary cases | exercises P1-A only, not the split producer |
| `tests/test_plane_endpoint_provenance.py` `synthetic kernel builder` | hand-declared endpoint chains and independent zero/one/multiple oracles (V2-A) | endpoint domain only; no split-phase patch or D profile |

**There is no declared-seam split-phase forward oracle, no calibration control
corpus, and no label fixture anywhere in the repository.**

### 2.5 Dataset-factory artifacts (real captures, not synthetic controls)

`exports/datasets/` holds five finalized real-Minecraft scene directories (four
distinct `scene_id`s; the two canal-town directories share one topology group) plus
two unfinalized directories. Each finalized directory has a timestamp-free `index.json`
with per-file hashes (`plan.json`, `recipe.json`, `capture.json`, `ground_truth.json`,
`reference.litematic`), per-screenshot `sha256`/`source_sha256`, a `topology_digest`,
a `scene_group`, a `split`, and `"usage": "developer/evaluation ground truth only;
never normal reconstruction input"`.

| Dataset | scenes | screenshots | splits | scene_group |
| --- | ---: | ---: | --- | --- |
| `fresh-c4-canal-town-2026-09-12` | 1 | 4 | train=1 | `74f080aa…cf458a` |
| `fresh-c4-canal-town-framed-2026-09-12` | 1 | 8 | train=1 | `74f080aa…cf458a` |
| `pilot-batch-train-station-2026-09-12` | 1 | 6 | train=1 | `1757a5ad…c84f4a4f` |
| `recon-canal-houses-2026-09-12` | 1 | 4 | train=1 | `3e9dfb43…68766edb` |
| `smoke-train-station-ds20c-2026-09-12` | 1 | 4 | train=1 | `1757a5ad…c84f4a4f` |

The factory is a **real-save** pipeline: `prepare_capture_dataset.py` inventories and
stages a live Minecraft save and the Fabric mod writes screenshots and
`ground_truth.json`. It has no synthetic-scene generator, and `build_plan` requires a
real staging token cut from a staged world (`dataset/plan.py`). The factory split
(`dataset/splits.py`) is keyed to the block topology digest alone and refuses the same
`scene_group` in two splits (`split_report`); `batch.SPLIT_AUTHORITY_NOTE` records
that a queue `declared_split` is only a leakage guard and the authoritative split is
computed at finalization.

### 2.6 Evaluation tooling conventions that the calibration harness should follow

- `frozen_image_runner` writes a schema-v1 freeze with `labels_present: False`, a
  strict top-key allow-list, a `_FORBIDDEN_KEYS` set (`gt`, `ground_truth`,
  `reference`, `manifest`, `capture`, `annotation`, `manual_mask`, `camera`, `pose`,
  `bounds`, `labels`, `label_data`) and a binary mask sidecar; `load_and_verify_frozen_image`
  refuses a freeze whose `labels_present` is not exactly `False`. This is the
  established prediction-freeze pattern.
- `evaluation_frozen_planes` verifies the immutable freeze before opening any
  capture/GT, and `scripts/evaluate_frozen_planes.py` is a post-freeze evaluator
  explicitly outside the normal CLI.
- `dataset/hashing.py` supplies `digest_json`/`canonical_json`/`sha256_file`
  (canonical JSON with `sort_keys=True`, compact separators, `allow_nan=False`), which
  matches the P0 canonical-JSON definition.

## 3. Calibration / evaluation / null / seam control availability

Contract §7.3 requires zero false accepts on every null and the fixed target power on
synthetic seam controls, naming three null classes: same-block/invisible-seam,
textureless, synthetic seam-null. V2-B §13 requires an independent forward oracle
that declares seams before synthesis and recovers them exactly.

| Control | Contract requirement | Existing artifact | Available now? |
| --- | --- | --- | --- |
| textureless null | zero false accepts | zeros patch in `test_split_phase_producer.py` yields `empty_profile` on both axes | **partial** — the producer abstention is exercised, but no calibration label/class or hypothesis set exists |
| same-block / invisible-seam null | zero false accepts | none | **missing** |
| synthetic seam-null | zero false accepts | none | **missing** |
| synthetic seam controls | target power with exact `alpha` recovery | none | **missing** |
| declared-seam forward oracle | declares `alpha` before synthesis, recovers exactly | none for split-phase | **missing** |
| label set (class + declared `alpha`) | labels only after prediction freeze | none | **missing** |
| calibration corpus identity + per-sample hashes | §7.5 disjointness | none for split-phase | **missing** |
| G-only replay unit | whole `HypothesisPhaseShiftEvidence` | P1-B surface exists; P0 G evidence exists; not wired | **present but unassembled** |
| D profile `E`/`C`/bins | frozen P0 output | `SplitSeamProfileEvidence` complete | **available** |
| `x_i` bin map and `B`/`I` classifier | V2-B §7 literal | **no implementation anywhere** (`tau_band`, `boundary_energy`, `interior_energy`, `zero_coverage` appear only in contract/derivation text or the calibration schema) | **missing** |

## 4. Can labels and P0 split evidence be produced without prohibited inputs?

P0 §1 allowed inputs are exactly `P`, the frozen `SplitPhaseGeometry`, the frozen
built-in `SplitPhasePartitionConfig`, the frozen `SplitPhaseProducerBudget`, and the
already-accepted reading/match identity. P0 §1 prohibitions are the source image,
camera manifest, F3 data, supplied coordinates/directions/bounds, reference
schematics, ground truth, labels/masks/texture labels, manual phase, caller seed,
residual/PSR winner selection, filesystem, network, process, environment, `open`,
`exec`, `eval`, `compile`, dynamic import. V2-B §2 adds the same list for production
inputs.

- **P0 split evidence:** yes. A synthetic `float64` patch `P` (finite, `>= 0`) plus a
  synthetic `SplitPhaseGeometry` is sufficient; no image, manifest, GT or label is
  read by the producer. The accepted test file already builds such patches.
- **Labels:** yes, from a test-local forward oracle that declares the control class
  and `alpha` **before** synthesizing `P`. A synthetic declaration is not a prohibited
  production input; it never reaches the producer, and the prediction pass can be
  frozen before the label file is opened (the `frozen_image_runner` `labels_present:
  False` pattern).
- **Hypotheses / provenance:** the P0 record deliberately omits `L`, `selected_b`,
  `H_w`, frame translations, and world sign. For synthetic controls these must be
  declared by the oracle just as it declares `alpha`; the harness must not import the
  full V2-A/C2-R-B/P1-B chain to obtain them. Calibration therefore exercises the P0
  producer + the V2-B §7 boundary/interior classifier **in isolation**; it does not
  exercise the V2-B §2/§8 input gate, which remains V2-B's own (unauthorized)
  responsibility.
- **Real-capture GT:** the dataset factory's ground truth and screenshots are
  evaluation-only scaffold (`advanced_only: true`) and are **not needed** for the
  synthetic null/seam controls the contract names. If a later evaluation split uses
  real captures, its labels are admissible only after the evaluation predictions are
  frozen, and its capture inputs still must never reach the producer.

**Conclusion:** the calibration controls and their labels are producible from
synthetic, evaluation-only inputs with no prohibited production input. The missing
work is the generator/oracle/classifier, not an input-permission problem.

## 5. Corpus disjointness and hash provenance

Contract requirements (P0 §7.5, V2-B §11): calibration and evaluation corpus identity
hashes and per-sample hashes are disjoint; calibration samples are disjoint from
hypothesis-generation (`G`) samples and from evaluation samples; disjointness is
digest-bound and re-checked at load; labels are used only after prediction bytes are
frozen (SHA-256 + timestamp) in a separate non-writing pass.
`SplitPhaseCalibration` enforces only `set(calibration_sample_sha256s) &
set(evaluation_sample_sha256s) == ∅` and 64-hex corpus/grid digests; it does **not**
enforce non-empty sample lists, `G`-disjointness, control-id disjointness, or grid
membership of the accepted tuple.

What exists:

- The dataset factory's topology-keyed split and `split_report` group-leak refusal,
  plus `digest_json`/`sha256_file` for per-file and per-screenshot hashes. This is a
  working precedent for corpus identity, but it is keyed to real Minecraft topologies.
- P0 already guarantees `G_a ∩ D_a = G_a ∩ X_a = D_a ∩ X_a = ∅` **within** a patch and
  a coordinate-set-only `D_a_sha256`, so per-patch G/D disjointness is digest-bound
  inside P0.

What is missing:

- `calibration_corpus_sha256`/`evaluation_corpus_sha256` and the per-sample
  `calibration_sample_sha256s`/`evaluation_sample_sha256s` for split-phase controls;
- a sealed label binding (label file digest recorded in the frozen prediction
  artifact) and a load-time re-check;
- a cross-corpus disjointness check that calibration controls are disjoint from the
  controls whose G evidence generates the hypotheses (there is no such control
  partition yet);
- all five existing finalized real-capture scene directories are `train`; the real
  corpus has no `dev`/`test` split and no split-phase labels.

The smallest defensible provenance unit is a corpus manifest that binds, per control:
`control_id` (64-hex), `P` bytes digest (the P0
`SHA256(b"mc-photogrammetry-split-patch-v1" || uint64_le(rows) || uint64_le(columns)
|| uint64_le(byte_length) || P.tobytes("C","<f8"))`), the `SplitPhaseGeometry` record
digest, the synthetic provenance digest (`L`, `selected_b`, frame translation, world
axis/sign, `H_w`), the control class, and the sealed label digest. The manifest itself
is then the corpus identity. This does not exist yet.

## 6. Finite lexicographically ordered dyadic grid and fixed target power

The contract justifies the **shape** of the grid but freezes **no values**; no grid
literal, target power, or control-id set exists anywhere in the repository (only the
schema and a round-trip test using placeholder ids `"a"*64`/`"b"*64`).

Exact schema constraints (P0 §7, verified against the implementation):

- `grid_version == "v2b-p0-calibration-grid-1"`, `evaluation_only is True`;
- `target_power` a reduced dyadic with `0 < value <= 1`;
- `tuples` non-empty, at most 4096, no duplicates, strictly increasing under
  `_split_calibration_tuple_lt` (exact cross-multiplied dyadic compare on the first
  three components, then `(min_held_out_samples, min_distinct_frames_per_axis)`);
- each tuple `(tau_band, tau_margin, tau_energy, min_held_out_samples,
  min_distinct_frames_per_axis)` with `tau_band` a reduced dyadic in
  `{0} ∪ (0,1]`, `tau_margin` a reduced dyadic in `(0,1]`, `tau_energy` a reduced
  dyadic in `{0} ∪ (0,1]`, `min_held_out_samples` an exact `int >= 1`,
  `min_distinct_frames_per_axis` an exact `int >= 2`;
- `null_control_ids`/`seam_control_ids` non-empty tuples of 64-lowercase-hex ids.

Meaningful-domain derivation (from the V2-B §7 metric, not a threshold guess):
`B` and `I` are coverage-normalized sums of `E` over bins within the **inclusive**
circular band `d1(x_i, ·) <= tau_band`, where `d1` is the exact circular metric on
`[0,1)` with maximum `1/2`. A radius `> 1/2` admits every bin for both `B` and `I`, so
`B - I == 0` and no tuple with `tau_margin > 0` can accept; `tau_band = 1/2` is the
largest non-degenerate band. `tau_band = 0` is schema-legal and means the exact-bin
band. So the useful band domain is `{0} ∪ (0, 1/2]`.

Proposed preregistration candidate (the lead must freeze the literal; this is the
recommended minimal default, not a contract value):

```
tau_band                     in {0, 1/8, 1/4, 3/8, 1/2}        (5)
tau_margin                   in {1/8, 1/4, 3/8, 1/2, 1}        (5)
tau_energy                   in {0, 1/8, 1/4, 3/8, 1/2}        (5)
min_held_out_samples         in {1, 2, 4, 8}                    (4)
min_distinct_frames_per_axis in {2, 3}                          (2)
target_power                 = (1, 2)
```

Nesting the loops in that exact order (each component ascending) yields 1000 strictly
increasing, duplicate-free tuples, already in the frozen lexicographic order; the
"lexicographically first feasible tuple" is then simply the first feasible element of
the nested scan. A compact alternative is
`{0,1/4,1/2} × {1/4,1/2} × {0,1/4} × {1,4} × {2,3}` = 48 tuples.

`target_power = (1,2)` is justified as the smallest non-degenerate rational power in
`(0,1]` with an exact integer test (`2 * matched >= total`); `(1,1)` (all seam
controls) is the stricter legal alternative. The exact definition of "power" and
"false accept" is not frozen by the contract and must be preregistered (§7.4).

## 7. Exact two-pass prediction-freeze / label-score procedure

Pass 0 — preregistration (lead-owned). Freeze: the grid literal (§6), `target_power`,
the control-corpus manifest (counts and per-control recipes), the sealed-label format,
the definitions of `held_out_samples`, "false accept", and "power", and the exact
commands. The grid is evaluation-only and never a producer input.

Pass 1a — corpus synthesis and sealing (label-bearing, once). For each control,
generate `P`, `SplitPhaseGeometry`, and the synthetic provenance (`L`, `selected_b`,
frame translation `T_f[w]`, world axis `w`, sign `s`, hypothesis set `H_w`), plus the
sealed label (class null/seam; declared `alpha` for seams). Write the corpus manifest
with per-control digests and the sealed-label file digest. The prediction pass is
given the manifest **without** the label file.

Pass 1b — label-blind prediction freeze. For each control: call
`produce_split_phase_evidence(P, geometry=…, frame=…, plane_index=…, reading_axis=…,
source_region_id=…, partition_config=SPLIT_PHASE_PARTITION_CONFIG, budget=…)` once and
require `status == "complete"` and `validate_split_phase_evidence(...) is True`. Then
for **every** grid tuple, for every axis `w`, every declared `h in H_w`, and every
declared observation `o`, compute the exact `B`/`I` and the accept/reject matrix using
only frozen evidence and declared provenance (block below). Freeze the complete
per-(control, tuple) determination status/`alpha` record as canonical JSON, compute
its SHA-256, and timestamp it before opening the label file. No label, no
`SplitPhaseCalibration`, and no grid value reaches the producer.

Pass 2 — label score and selection (separate non-writing comparison). Verify the
frozen prediction digest and timestamp; open the sealed labels and verify their
digest. For each tuple, evaluate the frozen predictions: zero false accepts on every
null, and power on seam controls at least `target_power`. Select the lexicographically
first feasible tuple; if none, `no_feasible_calibration`. Build
`SplitPhaseCalibration` (version, built-in `partition_config_sha256`, the accepted
tuple, `tie_break="no_tie_break_abstain"`, `target_power`, both corpus identity
hashes, both disjoint sample-hash tuples, both control-id tuples, the grid digest,
and `calibration_sha256`), validate it, and install it as the versioned built-in
record. A revision of any frozen parameter is a contract amendment and invalidates the
experiment.

Exact per-observation math (literal V2-B §7; no float decides a classification):

```
x_i      = (L + selected_b * Fraction(i) / Fraction(period_pixels)) mod 1   # i = axis bin index
ell(h,o) = (s * (h - T_f[w])) mod 1
B(h,o)   = sum(E[i] for d1(x_i, ell)        <= tau_band) / sum(C[i] for those bins)
I(h,o)   = sum(E[i] for d1(x_i, (ell+1/2)%1) <= tau_band) / sum(C[i] for those bins)
accept(o,h) iff B - I >= tau_margin and B >= tau_energy
               and held_out_samples(o) >= min_held_out_samples
```

`(h, L, selected_b, T_f[w], s, period_pixels)` are exact rationals converted once with
`Fraction(float.as_integer_ratio())`; `d1(x,y) = min(|x-y|, 1-|x-y|)`; a zero
denominator refuses that observation as `zero_coverage` (never a zero, never silently
dropped) and excludes it from the qualifying set. Axis `w` is determined iff exactly
one `h in H_w` is accepted by every qualifying observation and supported by at least
`min_distinct_frames_per_axis` distinct frames; a tie abstains. `E`/`C`/`axis_bin_indices`
come only from the P0 `SplitSeamProfileEvidence`; the classifier must decode the
base64 little-endian arrays and verify `payload_sha256`/`d_profile_sha256`/`D_a_sha256`
before use.

Definitions the preregistration must pin down (contract gaps): (a) `held_out_samples`
is not defined beyond the P0 `coverage {total, valid, per_bin}` record — the natural
scalar is `coverage["valid"]` (for the built-in profile it equals
`(d_stop-d_start)*E`); (b) "false accept" is stated at the null-control level, while
the §7 accept rule is per `(observation, hypothesis)` — I recommend defining a null
false accept as "any axis reported `determined`", and recording the per-hypothesis
accepts too; (c) "power" is stated for "synthetic seam controls" while V2-B §15 speaks
of `alpha` "on determined axes" — I recommend per-control success = `overall
determined` with `alpha` exactly equal to the declared vector (`dTorus == 0`), with
per-axis matches also recorded.

Evaluation pass (V2-B §15, separate corpus): repeat Pass 1b/Pass 2 on the disjoint
evaluation split with the frozen calibration, freezing evaluation predictions before
opening evaluation labels, and check the §15 promotion criteria. This is not part of
the calibration fit and cannot be run before the fit is accepted.

## 8. Commands and runtime estimate

### 8.1 Verification commands (exist now; label-free)

```powershell
.\.venv\Scripts\python.exe -m py_compile src\mc_photogrammetry\reconstruction\lattice_planes.py tests\test_split_phase_producer.py
.\.venv\Scripts\python.exe -m pytest tests\test_split_phase_producer.py -q -p no:cacheprovider -o addopts=
```

Re-measured here: `py_compile` clean and **23 passed in 0.26 s**. The accepted report
also records `197 passed` in 56.84 s and `179 passed` in 40.85 s for the broader
focused sets (not re-run here). Interpreter: `.venv\Scripts\python.exe` (CPython
3.10.11).

### 8.2 Producer cost (measured, label-free)

| `P` shape | produce | validate | `patch_samples` | `d_profile_terms` | `payload_bytes` | `replay_checks` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32×32 | 2.90 ms | 1.33 ms | 1 024 | 192 | 1 626 | 8 |
| 64×64 | 5.69 ms | 2.63 ms | 4 096 | 384 | 3 162 | 8 |
| 128×128 | 20.02 ms | 10.04 ms | 16 384 | 768 | 6 234 | 8 |
| 256×256 | 77.86 ms | 38.07 ms | 65 536 | 1 536 | 12 378 | 8 |

### 8.3 Calibration runtime estimate (estimate only; no calibration was run)

The producer/validator runs **once per control** and is independent of the grid. The
per-tuple cost is a pure-Python `O(T · Σ_controls |H_w| · 2n)` scan over the frozen
`E`/`C` arrays plus exact `Fraction` bin math, with `n = W_c` for `u` and `n = H_c`
for `v`. The frozen grid caps `T` at 4096. Representative assumptions (must be
replaced by a measured run when the harness exists): 64 controls, square 64×64
patches (`2n = 128`), `|H_w| = 4`, `T = 1000` → ~3.3×10^7 band comparisons plus
64 producer+validate calls (~0.5 s, from §8.2); a Python implementation of this size is
expected in the low tens of seconds, not minutes. The estimate is dominated by the
classifier scan, not the producer, and should be measured once the harness exists; no
calibration runtime is claimed here.

## 9. Smallest source/test work package needed to run the gate

The accepted P0 gate (§9) authorized only the narrow `lattice_planes.py` producer plus
isolated tests; it does **not** authorize calibration. The smallest package that runs
the gate without touching the accepted P0/V2-B/P1 modules or any product path is:

| New path | Responsibility |
| --- | --- |
| `src/mc_photogrammetry/reconstruction/split_phase_calibration.py` | evaluation-only: the frozen grid literal + validator; the synthetic control generator and independent forward oracle (declares class/`alpha`/provenance before synthesis); the literal §7 `x_i`/`B`/`I`/accept classifier; the lexicographic scan and `SplitPhaseCalibration`/`SplitPhaseCalibrationGrid` construction; the corpus manifest + disjointness re-check. Imports `lattice_planes` read-only and `dataset.hashing` for digests; never imports `plane_residue`, capture, renderer or product paths; no label ever flows into `lattice_planes`. |
| `tests/test_split_phase_calibration.py` | frozen-schema round trips, oracle exact recovery, zero-false-accept mutants, target-power boundary, lexicographic-order/selection determinism, dyadic/duplicate/disjointness refusals, payload decode/re-hash, and an AST isolation gate. |
| `scripts/fit_split_phase_calibration.py` | the only IO: `--stage predict` (writes the label-blind prediction freeze + SHA-256 + timestamp) and `--stage score` (verifies the freeze, opens the sealed labels, writes the accepted `SplitPhaseCalibration` artifact); evaluation-only, not wired into the CLI. |
| `docs/reports/M4-V2B-split-phase-calibration-prereg-2026-09-14.md` (lead-owned) | freezes the grid literal, target power, control counts/recipes, label format, and the three definitions in §7.4. |

Forbidden in that package: editing `lattice_planes.py`, `plane_residue.py`,
`supported.py`, `lattice.py`, `plane_graph.py`, `plane_axis_support.py`, the V2-A or
P1 modules; creating any V2-B module; placement/wiring; runner/schema/selector/CLI/
evaluator/renderer/export/product changes; any source-image, camera-manifest, F3, GT
or label access in the producer; any expectation derived from the code under test.

Open decisions for the lead before the package is authorized:

1. Freeze the grid literal and `target_power` (§6) and the three definitions in §7.4.
2. Decide whether the calibration classifier is a fresh independent implementation of
   the §7 math (recommended; matches V2-B §13.1) or a shared evaluation-only module
   that V2-B later imports — the latter would create a V2-B dependency now and is not
   recommended.
3. Confirm that synthetic controls may supply `L`/`selected_b`/`H_w` directly, so
   calibration does not require the unimplemented V2-A/C2-R-B/P1-B chain.
4. Confirm corpus sizes and the corpus/evaluation disjointness evidence required for
   the `SplitPhaseCalibration` sample-hash tuples.
5. Note the schema gaps that currently depend on the evaluator to enforce: accepted
   tuple membership in the grid, control-id disjointness, non-empty sample hashes, and
   `calibration_grid_sha256 == SplitPhaseCalibrationGrid.grid_sha256`.

## 10. Readiness verdict

- **Calibration data contracts:** ready (implemented, validated, digest-bound).
- **Calibration inputs (controls, labels, corpus, grid, target power):** not ready —
  none exist; this is the whole gap.
- **Classifier (§7 `x_i`/`B`/`I`):** not implemented anywhere.
- **Split evidence:** ready to produce from synthetic `P` with no prohibited input.
- **Labels:** producible synthetically with no prohibited production input, but must
  be sealed and scored only after the prediction freeze.
- **Corpus disjointness provenance:** methodology exists (dataset factory), split-phase
  corpus identities do not.
- **Gate executable today:** no. The smallest work package in §9 is the prerequisite.

## 11. Verification appendix — referenced paths and hashes

Every path below was confirmed to exist; every quoted digest was recomputed here.

| Path | SHA-256 (lowercase) |
| --- | --- |
| `src/mc_photogrammetry/reconstruction/lattice_planes.py` | `82ada0ccc0f92b580342aa35f8333d1013910f7cfa91eada0ed7d5e7c95831e7` |
| `src/mc_photogrammetry/reconstruction/lattice.py` | `6fcde3a7e8d451e207ec0eed70c26a3f1746ab89f9ac8cf7bba5bcb8fd7bb17e` |
| `src/mc_photogrammetry/reconstruction/supported.py` | `4c94564d3842f1c3a03883f6f95a78a4ac08260150eb9fc12251bf708e35f29c` |
| `src/mc_photogrammetry/reconstruction/plane_residue.py` | `55ec0cad8fe1088b7b33b1f57c4d58d3d3b912f79765a1f391553ac7b845be0d` |
| `src/mc_photogrammetry/reconstruction/plane_graph.py` | `c5245680b469f93ded5525bb80b0ddf065b46a64e1d88d66cc1fa93115522e92` |
| `src/mc_photogrammetry/reconstruction/plane_endpoint_provenance.py` | `3cb6aacea4d9e6873d0b10edd4ea7539b1646a1e75475cef2006f81caa3316e7` |
| `src/mc_photogrammetry/reconstruction/plane_vector_domain.py` | `72ee29fb936cffd7f8a154c18d6f6bab18876d57ad66f45bc316e39f7d45e7f5` |
| `src/mc_photogrammetry/reconstruction/plane_axis_support.py` | `18598afb2096b504464411e3a2cbf6a2e960cb2601c4dd2fb8f77d4b46c91f3f` |
| `src/mc_photogrammetry/reconstruction/plane_relative_translation.py` | `f703a28d275d55e4a85fad7fde0bfd612928f8d54a5489c0ea097fb42efbcd3e` |
| `src/mc_photogrammetry/reconstruction/frozen_image_runner.py` | `8221c52b3e89e2c136a0bb779a01b45121677b01bf52abf4b26114e1ce668b61` |
| `src/mc_photogrammetry/evaluation_frozen_planes.py` | `3918388075dcfab7b9cde0788bbe930df7ef295041a6871bfd045a0594aee91c` |
| `src/mc_photogrammetry/evaluation_context_export.py` | `d0aa5eba518d35ae33fd34d3d32f68aecae161b38869123ea3fb8c38d0652b87` |
| `src/mc_photogrammetry/dataset/splits.py` | `bb7398b9546c11df73f2bae6758cd6d87ac44e9d29cddba362491955863a02d9` |
| `src/mc_photogrammetry/dataset/hashing.py` | `265deed3be4095e66e6a3a4a6d0632d8cbfb8ee8a856a9b69408b8363b979611` |
| `src/mc_photogrammetry/dataset/finalize.py` | `d16798384e84a1898fe58fd94947c9c5c8ea9ce6ffab7f83b9e919edb89de033` |
| `src/mc_photogrammetry/dataset/plan.py` | `422c14743782d952996f3d3198d13659305a0c7b45ee9c4761d873d0cda335ac` |
| `src/mc_photogrammetry/dataset/schemas.py` | `aa89ba78ff5cba4c276d3c0404ffe3739eadf5b7a4e81fe0dc56359b270098a3` |
| `src/mc_photogrammetry/dataset/batch.py` | `f61f75212913d3c926293cc8e35b82ed8141475d433ec796da67f8923db04900` |
| `scripts/evaluate_frozen_planes.py` | `3c2311c655074b975f3c7c2fc06f256d451ca9d9ea692a556e7878a85b572270` |
| `scripts/prepare_capture_dataset.py` | `83b8f587f9865f6139af59450ff9561a72d51a7ae4cd0776110241a506aa5c31` |
| `scripts/finalize_capture_dataset.py` | `8356821199143d1eafb94b1b25811f803c413fe3b0eaed7822e6589a1049b924` |
| `tests/test_split_phase_producer.py` | `e5e17b9ccc8203394ebe1f9f78601176f33c041f34620f214d1b02457b7a9b31` |
| `tests/test_plane_residue.py` | `4d7d00cd1b4d31ec70d964ee153bf13e648f8325e8b284f0e46a130eec43a36f` |
| `tests/test_lattice_planes.py` | `76351e29dd63ab1231d4655950ef2f0f4123663165749b5c4d35cfa64da3dcf1` |
| `tests/test_plane_endpoint_provenance.py` | `967559f7b20184442f558c34af803128e85ab7596fd50c2489dd8a16c5270f95` |
| `tests/test_frozen_plane_evaluation.py` | `7407c1d43d30e0537d12aa7819f8d131948c3839cb974060026abdaee04c02b0` |
| `exports/datasets/fresh-c4-canal-town-2026-09-12/index.json` | `d3dc69b048bf8b827d75df402aa041445678ae63ba5124e6eefc6586278413dd` |
| `exports/datasets/fresh-c4-canal-town-framed-2026-09-12/index.json` | `0637caba1a523bbdf7351dedd7c4f93bfdb611f0b549fc54b3a781f3d25ee46b` |
| `exports/datasets/pilot-batch-train-station-2026-09-12/index.json` | `4ef40f8be5663108036bd3827e298286c93747a71ed9bbb4dcd2a3266ecf4837` |
| `exports/datasets/recon-canal-houses-2026-09-12/index.json` | `5d0db614b1802bd7fcd047c390dd4637eb20dcd88f550d76a624d3bf6ba6fae9` |
| `exports/datasets/smoke-train-station-ds20c-2026-09-12/index.json` | `a2a21eb49d4b7664f603eaaca77bf8a8095defee520a2c64dc2592fb1625a3e0` |
| `docs/reports/DS-17-real-capture-factory-2026-09-12.md` | `189ed75e917a4c41950137961b73823066d559ef2fadf9fde83744a0b2de63b1` |
| `docs/reports/M4-evaluator-e1-frozen-plane-score-2026-09-12.md` | `a4010caf625a8ca2b3882f24bad9b3697c7f90904018807be693c6a8064bcfea` |
| `docs/deepseek-worker.md` | `dac0b83d4e516bf661c31361b4f213e6c0fa289692f7f12d599ec6f887818231` |

Stale-hash note: the accepted `M4-P0-endpoint-spacing-2026-09-13.md` table lists test
hashes that later work has since changed (`tests/test_lattice_planes.py`,
`tests/test_plane_graph.py`, `tests/test_supported.py`); those are historical rows and
are not reasserted here. The frozen contract hashes in §1 are unchanged.

## 12. Stop

This report is the only artifact of this task: no source, test, schema, fixture,
image, label, calibration, runner, CLI, evaluator, renderer, export, contract,
handoff, or comms file was created or modified; no calibration or label scoring was
run; no V2-B, placement, volume, wiring, launcher/client automation, background
process, Defender/security change, system-directory write, junction/reparse
operation, recursive delete, encoded/obfuscated command, or git operation was
performed. Stop for lead review.
