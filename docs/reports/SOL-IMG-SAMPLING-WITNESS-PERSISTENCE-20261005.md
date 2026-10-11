# SOL-IMG-SAMPLING-WITNESS-PERSISTENCE-20261005

Task `DS-IMG-V2-SAMPLING-WITNESS-PERSISTENCE` (AGENT_COMMS 2026-10-05),
implemented by DeepSeek under Sol 6.1's read-only architecture decision. This is
an **evaluation-only** persistence slice: it adds an independently versioned,
strictly capped section that records the ordinary signature sampler's own
`SignatureSamplingWitnessV1` tap tables into a new V2 Route B diagnostic record,
and teaches the ownership preflight to validate that section while still
withholding ownership authorization.

No image was rerun, no evaluator/inference/matcher/solver/signature produced a
new payload, no production capture was performed, and no historical R4 artifact
was recreated or migrated.

## 1. Owned paths

| Path | Change |
| --- | --- |
| `src/mc_photogrammetry/pipeline/signature_sampling_persistence.py` | **new** section serializer/deserializer + bounded stable-file reader |
| `scripts/evaluation/run_route_b_diagnostic.py` | V2 record schema, persisted section, bounded/prescan replay validation |
| `src/mc_photogrammetry/evaluation_same_axis_ownership_preflight.py` | V1 legacy + V2 sampling availability validation, ownership still withheld |
| `tests/test_signature_sampling_persistence.py` | **new** focused synthetic suite |
| `tests/test_route_b_diagnostic_runner.py` | V2 persistence/cross-bind/bounded-replay tests |
| `tests/test_evaluation_same_axis_ownership_preflight.py` | frozen-JSON byte-identity + V2/V1 preflight tests |
| `AGENT_COMMS.md` | authorized result append only |

No production module, frozen artifact, pinned JSON, schema wrapper or matcher
sidecar was modified.

## 2. The loss seam being closed

Sol's review found that `derive_image_derived_signatures` retains the full
`SourceTapWitnessBindingV1` (including the sampler's `SignatureSamplingWitnessV1`),
but `run_route_b_diagnostic.py::_record_payload` wrote only `item.to_record()`
— identity plus `witness_sha256`. The arrays that
`project_ownership_into_source_space` consumes (`cell_rectangles`,
`cell_source_indices`, `cell_source_ptr` and the rectification map/tap tables)
were never persisted, and cannot be reconstructed from their hashes.

This slice persists them. It does **not** fabricate or migrate historical arrays:
a V1 record stays legacy with the sampling payload typed unavailable, and a real
future record requires a separately authorized ordinary source-image pass. The
implementation and all tests are synthetic.

## 3. Section contract

Outer schema `mc-photogrammetry-route-b-diagnostic-record-v2` adds one key,
`sampling_persistence`, holding
`mc-photogrammetry-signature-sampling-persistence-v1`:

```
schema, status ("complete"|"refused"), reason, field, detail,
policy (the exact cap profile), entry_count, entries,
payload_bytes, metadata_bytes, work_units,
section_sha256 = SHA256(domain + section without section_sha256)
```

Each ordered entry preserves its ordinal and identity (`frame_id`,
`canonical_index`, `alternative_index`, `signature_sha256`, `ingress_sha256`,
`mask_sha256`, `masks_sha256`, `max_cells`, `max_rectified_side`,
`pixels_per_cell`, `witness_sha256`) with outcome:

* `bound` — the full `SignatureSamplingWitnessV1` payload;
* `refused` — the original `SourceTapWitnessRefusalV1` reason/detail, no payload;
* `persistence_refused` — the witness existed but a cap refused its persistence;
  the original bound witness and matcher disposition are unchanged.

The persisted witness carries sampling metadata/intermediate hashes
(`grayscale_sha256`, `gradient_y_sha256`, `gradient_x_sha256`, `energy_sha256`,
`coverage_patch_sha256`, `gradient_edge_order`, `subsample_factor`, `seam_band`),
the full RGB/coverage rectification witnesses and the optional region
rectification witness (backend pins `opencv_version`/`opencv_build_sha256`,
`interpolation`/`border`/`inter_bits`/`weight_denominator`, `source_shape`,
maps/validity/tap pointers/source indices/weights), plus `cell_rectangles`,
`interior_rectangles`, `cell_source_ptr` and `cell_source_indices`.

Arrays are canonical base64 of little-endian raw bytes with an **exact**
per-field dtype — only `<i8`, `<f4`, `<u2`, `|b1` — plus declared shape, byte
length and raw SHA-256. NumPy aliases (`i8`, `int64`, `=i8`, `>i8`, `float32`,
`uint16`, `bool`, …) are deliberately not broadened. Decoding rebuilds immutable
arrays and the existing typed `RectificationWitnessV1` /
`SignatureSamplingWitnessV1` / `SourceTapWitnessBindingV1` values; every nested
`witness_sha256` is recomputed and must match, and a supplied ingress pin must
match the persisted `ingress_sha256`.

Cap profile (validated as a configuration), as revised by
`DS-IMG-V2-SAMPLING-WITNESS-REVISE` (`max_source_side=4096`,
`max_source_pixels=16,777,216` split the source-image bound from the
rectified-output bound; the known 2560x1369 T2 ingress therefore persists):

* 85 entries; 2 MiB raw payload per entry / 8 MiB aggregate;
* 64 KiB metadata per entry / 512 KiB aggregate (identity and refusal
  reason/detail included);
* 32 MiB complete serialized record; depth 16; 6,000,000 JSON tokens;
* 20,000,000 work units; 256 cells per side; 2,048 rectified side;
* 4,096 source side / 16,777,216 source pixels.

All caps are measured on plain declared sizes before any `tolist`, base64
encode/decode, JSON serialization of arrays or NumPy allocation. Cells, samples,
taps and memberships are bounded before traversal; pointer monotonicity,
index/sentinel bounds and the producer's own per-sample bilinear contract
(each `tap_ptr` span holds 1–4 taps summing to the weight denominator, and every
depth-invalid sample is exactly one weight-1024 tap at the `-1` border sentinel)
are validated. Section digest verification runs **after** every scalar, length
and declared-budget check, so an over-cap section is never serialized to hash it.

Refusal semantics: budget exhaustion yields a small typed
`persistence_refused` entry (or a typed refused section when the whole section or
work budget is exhausted). Nothing is truncated, silently omitted or
substituted; a missing payload stays UNKNOWN.

## 4. Runner changes

* New captures write schema **v2** with the section; V1 stays readable as
  legacy.
* The section is persisted from the same single signature pass's
  `signature_witnesses`, and the top-level witness summary and support capture
  are unchanged.
* Before publication the runner re-validates the section and its ordered
  cross-bind against the typed ingress pins.
* `_write_record` refuses a record over 32 MiB; if the additive section alone
  would push the record over the cap it degrades to a typed refused section
  (support capture and dispositions unchanged) rather than truncating.
* `validate_record` no longer uses unbounded `read_text()` or record-controlled
  NumPy conversion: it reads one bounded, stable, non-link snapshot (chunked at
  `max_serialized_bytes + 1`, `lstat`/`fstat` identity, growth/replacement
  refused), scans depth/tokens on the raw bytes before `json.loads`, prescans
  every existing support array and the new sampling payloads before allocation,
  then rebuilds and replays the capture and the sampling section. The post-write
  replay uses the same bounded reader.

## 5. Preflight changes

* A V1 record is read exactly as before: `sampling_witness_state=unavailable`,
  the `SignatureSamplingWitnessV1` tap-table seam, ownership withheld. Its
  result JSON is **byte-identical**.
* A V2 record must carry a caps-valid, digest-sealed section. The prescan
  validates it on the plain document before allocation; the orchestration
  cross-binds the ordered outcomes to the record's own witness summary and
  decodes them against the typed ingress pins. **Root-pinned ingress is required
  for every bound witness before `available`**: a bound frame with no
  capture-root pin is a typed refusal, and the binding constructor re-checks each
  rectification `source_shape` against the pinned frame shape.
* When every bound witness decodes against a root pin the state becomes
  `available` and the `sampling_witness_tap_tables_not_persisted` blocker is
  dropped. Ownership authorization remains **withheld** for the five absent
  alignment pins (`camera`, `intrinsics`, `pose`, `occlusion_convention`,
  `evaluator_domain`), and the result reports ingress-bound/unbound counts
  explicitly.
* No ownership/correctness percentage is emitted anywhere, and no manual
  metadata is exposed.

## 6. Acceptance gates and evidence

| Gate | Evidence |
| --- | --- |
| 1. Ordinary sampler roundtrip reproduces every array and nested/witness hash, incl. region data and border sentinels | `test_ordinary_sampler_roundtrip_reproduces_every_array_and_digest`, `test_valid_border_sentinel_tap_roundtrips_exactly`, `test_optional_region_rectification_roundtrips` |
| 2. Runner fake-API integration: one pass, exact witnesses persisted, support capture/dispositions unchanged | `test_v2_record_persists_the_exact_sampler_witnesses_without_changing_support` (+ existing single-pass tests) |
| 3. Allocation traps: oversized/aliased dtype, false shape, malformed/oversized base64, byte/work caps, invalid pointers/weights/indices, out-of-range integers, metadata expansion | `test_oversized_string_dtype_*`, `test_aliased_or_foreign_dtype_*`, `test_false_shape_*`, `test_declared_byte_length_mismatch_*`, `test_malformed_and_oversized_base64_*`, `test_decode_refuses_a_corrupt_base64_body_*`, `test_invalid_tap_pointer_index_and_weight_*`, `test_out_of_range_rectangle_integer_*`, `test_rectangle_count_expansion_*`, `test_metadata_expansion_*`, `test_byte_and_work_caps_*` |
| 4. Re-signed tampering: foreign/swapped identities, missing/duplicate entries, modified arrays, nested backend/hash mismatch | `test_modified_array_is_refused_even_when_resigned`, `test_foreign_or_swapped_identity_*`, `test_swapped_witness_between_entries_*`, `test_swapped_identity_between_entries_*`, `test_missing_and_duplicate_entries_*`, `test_nested_backend_and_hash_mismatch_*`, `test_tampered_section_digest_and_policy_*`, `test_ingress_pin_mismatch_*`, `test_resigned_sampling_array_tamper_refuses_on_replay` |
| 5. Exact cap boundaries, persistence refusals, immutable arrays, stable-read growth/replacement, V1 legacy | `test_exact_entry_payload_cap_boundary_*`, `test_decoded_arrays_are_immutable`, `test_cap_configuration_validation`, `test_default_reader_never_uses_unbounded_read_bytes`, `test_bounded_stream_read_*`, `test_reader_refuses_simulated_growth_*`, `test_reader_refuses_simulated_replacement`, `test_reader_depth_and_token_caps_*`, `test_v1_legacy_record_validates_without_a_sampling_section`, `test_frozen_v1_record_reports_the_legacy_sampling_seam_unchanged` |
| 6. Frozen accounting/JSON unchanged; V2 availability validated; no evaluator reads/scores/manual metadata | `test_frozen_result_record_is_byte_identical_to_the_pinned_json`, `test_v2_sampling_section_decodes_available_against_typed_ingress`, `test_v2_refused_sampling_section_full_preflight_is_typed_unavailable`, `test_v2_available_wiring_closes_the_tap_blocker_but_withholds_ownership`, `test_v2_missing_sampling_section_is_refused`, `test_v1_schema_with_a_sampling_section_is_refused` |

## 7. Exact commands and results (foreground)

```
.venv\Scripts\python.exe -m pytest tests/test_signature_sampling_persistence.py -o addopts= -p no:cacheprovider -q
  -> 44 passed in 0.80s (exit 0)

.venv\Scripts\python.exe -m pytest tests/test_route_b_diagnostic_runner.py -o addopts= -p no:cacheprovider -q
  -> 41 passed in 17.65s (exit 0)

.venv\Scripts\python.exe -m pytest tests/test_evaluation_same_axis_ownership_preflight.py -o addopts= -p no:cacheprovider -q
  -> 66 passed in 114.77s (exit 0)

.venv\Scripts\python.exe -m pytest tests/test_candidate_support_provenance.py tests/test_candidate_support_provenance_v2_preflight.py tests/test_signature_sampling_witness.py tests/test_candidate_support_ownership_harness.py -o addopts= -p no:cacheprovider -q
  -> 125 passed in 7.96s (exit 0)
```

## 8. Frozen pins preserved

* `.tmp-routeb-v2-frozen-r4-20261004/record.json` — 16,896,513 B,
  `006e8b6b16a312cdaa61a17b4336ce27c1abac590435645908ea9cfef2e998ee` (not read
  for this implementation; unchanged).
* `docs/reports/SOL-IMG-SAME-AXIS-OWNERSHIP-PREFLIGHT-20261005.json` — 5,804 B,
  `448c6debb9f6b36160b9a08ea1050fbf925813b378d688712b3dd55bb614201d`
  (regeneration is byte-identical; asserted by test).
* `candidate_support_provenance_v2.py` `F5CEE3CF…CCD2`,
  `image_derived_graph.py` `32747546…4188`,
  `image_derived_signatures.py` `8F7D24C0…7555`,
  `image_derived_graph_attribution.py` `8441E91C…CB81` — all unchanged.

## 9. Changed-file SHA-256 (lowercase)

| Path | Bytes | SHA-256 |
| --- | --- | --- |
| `src/mc_photogrammetry/pipeline/signature_sampling_persistence.py` | 88,048 | `948d6548414ff593dbea69f11f18d849660e6698db514dec7b2ddee01f75dc4f` |
| `scripts/evaluation/run_route_b_diagnostic.py` | 63,553 | `f4fdb722ce698fb4f75571e14a80e050defe7f992aa6f4b01811c413b156b922` |
| `src/mc_photogrammetry/evaluation_same_axis_ownership_preflight.py` | 98,607 | `a673ff7d8a28093697fa0290a035874eacf51dfac7f0b7d0c3c3cefecc9c2784` |
| `tests/test_signature_sampling_persistence.py` | 34,345 | `2c9bd49de973372f02570bba958ba1da617cbfc86b60cfc042fd4162f77f2dbb` |
| `tests/test_route_b_diagnostic_runner.py` | 56,767 | `1aa1ee3f03dc88249bf6bf910c0852563ad1cc2edeed5bdf7d91b4ce80501cf9` |
| `tests/test_evaluation_same_axis_ownership_preflight.py` | 37,632 | `388ed6e58f722fe96f0789c90c2c0baaada1ac3b87e7103b0f0bac54581ffa8f` |

## 10. Limitations and what is not claimed

* No ownership, alignment, geometry, correctness, recall or accuracy claim is
  made; no ownership/accuracy percentage exists anywhere in the output.
* The persisted sampling section is a *source-space dependency record*, not
  proof of camera-to-screenshot alignment. The preflight still types camera,
  intrinsics, pose, occlusion-convention and evaluator-domain pins unavailable
  and withholds ownership.
* No real V2 record exists yet: a real future capture requires a separately
  authorized ordinary source-image pass. Historical R4 cannot reconstruct the
  arrays from hashes and was not migrated.
* The preflight reads a V2 record only when it carries a valid section;
  availability means "the persisted tap tables decoded against the reviewed caps
  and the record's own witness summary", not that any ownership join ran.
* The persistence reader proves one stable regular-file read on this platform
  under an `lstat`/`fstat` `(dev, ino, size, mtime_ns)` identity gate; it is
  **not** an atomic snapshot against a hostile same-size in-place writer that
  also restores the recorded mtime (see section 12). The trusted injected
  `reader` seam is expected to return exact bytes.
* The preflight's V2 integration tests use the frozen capture for the
  availability decode path and the frozen-derived refused/missing/mixed paths.
  As of `DS-IMG-V2-SAMPLING-WITNESS-REVISE` the full preflight availability path
  is also closed end-to-end **without any stubbed decoder**, over a section
  built from the frozen record's own root ingress pins; the stubbed-decoder
  wiring test remains only as an additional wiring control.
* Single frozen run with the universe cap equal to the row count still applies.

## 11. What was not run

No source screenshots/images, truth payloads, camera manifests, masks,
scene/world/Litematic/JAR inputs, client/Prism, evaluator execution, inference,
replay, matcher, solver, signature run or network was used. No production
capture, no historical R4 recreation. Synthetic tests only. No junction/reparse
creation, recursive deletion, encoded/obfuscated command, generated
executable/installer, background/persistent process, Defender/system change or
system-directory write was used.

## 12. Revision `DS-IMG-V2-SAMPLING-WITNESS-REVISE` (reviewer P1/P2 corrections)

Implemented only inside the same seven owned paths plus the authorized
`AGENT_COMMS.md` append. Sections above are updated in place where the revision
changed a claim; this section records each correction and its evidence.

### P1 corrections

1. **Source shape decoupled from the rectified-output cap.** New
   `max_source_side=4096` / `max_source_pixels=16,777,216` limit fields are
   validated as part of the cap profile and applied to `source_shape` in
   `_preflight_rectification` and `_validate_rectification_plain`;
   `max_rectified_side` now bounds only the rectified map/rectangle grid. The
   binding constructor (and therefore `decode_section`) additionally requires
   every rectification `source_shape` to equal the pinned root ingress shape.
   Evidence: `test_known_t2_source_shape_roundtrips_with_a_tiny_rectification`
   (real 1369x2560 ingress + real sampler roundtrip; 2560 > 2048),
   `test_source_shape_cap_boundaries_are_independent_of_the_rectified_side`,
   `test_witness_source_shape_is_bound_to_the_root_pin_on_decode`.
2. **Allocation/traversal/serialization ordering.** `_preflight_contract_array`
   reads dtype/rank/size/bytes from the original object and requires the exact
   contract dtype *before* `np.ascontiguousarray`; `_preflight_rectification`
   charges every declared array byte before any value scan;
   `_account_witness` applies the payload, metadata and work caps before
   `np.diff`/min/max/per-sample scans or metadata materialization;
   `_declared_witness_metadata_bytes` measures the metadata (including rectangle
   cardinalities via `_rectangles_json_size`) without building the per-cell
   lists; `canonical_json_size` gives the exact serialized size before any
   serialization (used by `persist_signature_witnesses`,
   `_fit_sampling_section` and `_write_record`). Evidence:
   `test_oversized_noncontiguous_array_is_refused_before_any_copy`,
   `test_rectangle_expansion_refuses_before_metadata_materialization`,
   `test_record_cap_is_decided_before_the_complete_record_is_serialized`.
3. **Decoder digest last.** `validate_section_plain` now validates policy,
   schema/status, depth, key sets, entry count/order, scalar types, declared
   payload/metadata/work budgets and refusal identity/detail sizes first, and
   only then calls `_verify_section_digest`. Evidence:
   `test_over_cap_refusal_detail_is_refused_before_any_serialization`,
   `test_over_cap_rectangle_count_is_refused_before_any_serialization`
   (both trap `section_digest` and `_canonical_bytes`), plus the unchanged
   `test_tampered_section_digest_and_policy_are_refused`.
4. **Refusal metadata is budgeted.** Identity plus refusal reason/detail are part
   of the per-entry and aggregate metadata caps on both the producer and the
   validator; an over-cap refusal detail degrades to a small typed
   `persistence_refused` record (`persistence_entry_metadata_over_cap`, empty
   detail) that still cross-binds to the record's original refused summary.
   Evidence: `test_over_cap_refusal_detail_degrades_to_a_small_typed_record`,
   `test_aggregate_metadata_exhaustion_stays_a_small_typed_refusal`,
   `test_refused_identity_metadata_is_counted_in_the_section_total`.
5. **Root-pinned ingress required for `available`.** `decode_section` treats an
   omitted ingress map as "no pins" and refuses every bound entry when
   `require_ingress` is true (the default); `SignatureSamplingPersistenceSectionV1.state`
   requires every bound entry to carry a binding. The preflight's
   `_decode_sampling_section` now passes `require_ingress=True`, and the runner
   supplies its typed capture-root pins (`require_ingress=capture is not None`);
   only the post-write structural replay, which has no typed pins and never
   computes availability, stays permissive. The runner test fakes now build
   witnesses over the entry session's own pins so records are internally
   consistent. Evidence:
   `test_availability_requires_root_pinned_ingress_for_every_bound_entry`,
   `test_v2_missing_root_pin_is_a_typed_refusal`,
   `test_v2_foreign_root_pin_digest_is_a_typed_refusal`,
   `test_v2_root_pinned_sampling_section_full_preflight_is_available`
   (full preflight availability without any stubbed decoder).
6. **Identity scalars before dedup/hash.** `_validate_identity_scalars` runs
   before the seen-set key is built for every outcome, so `frame_id=[]` and
   friends are typed `persistence_record_malformed` refusals instead of raw
   `TypeError`. Evidence:
   `test_malformed_identity_is_a_typed_refusal_not_a_type_error` (bound and
   refused, 6 parametrizations),
   `test_malformed_persistence_refused_identity_is_typed`.

### P2 corrections

7. **Work is precharged and includes scans/encoding.** `work_units` is charged
   on declared payload bytes + metadata bytes + 4 per rectangle + 2 per array
   record, computed identically by `_account_witness` and
   `_validate_witness_plain`, and checked before any pointer/index/weight/
   per-sample scan or encode. Evidence:
   `test_work_cap_is_refused_before_any_value_traversal` (traps every value
   validator and `_encode_witness`), plus the existing
   `test_byte_and_work_caps_are_typed_persistence_refusals`.
8. **Per-sample bilinear producer contract.** `_validate_bilinear_tap_semantics`
   requires every sample's `tap_ptr` span to hold 1–4 taps summing exactly to
   `weight_denominator` (1024), and every depth-invalid sample to be exactly one
   weight-1024 tap at the `-1` border sentinel; it runs on the producer account
   path and on decode. Evidence:
   `test_individually_valid_but_wrong_per_sample_weight_total_is_refused`,
   `test_invalid_sample_tap_span_semantics_are_enforced` (all real sampler
   witnesses, including the 4x4 invalid-sample case).
9. **Ownership-aware rollback.** `_write_record` now wraps the whole
   post-publication verification (bounded re-read, capture/digest replay and the
   second sampling replay) in one rollback that unlinks exactly the artifact this
   call created. Evidence:
   `test_post_write_sampling_replay_failure_rolls_back_the_owned_record`
   (forced second-replay failure leaves no record, partial, project or staging).
10. **Narrowed reader claim.** `_read_bounded_file_snapshot` and the preflight
    reader now state the residual limitation explicitly: the
    `(st_dev, st_ino, st_size, st_mtime_ns)` path/descriptor identity gate
    detects growth, truncation and replacement but cannot prove an atomic
    snapshot against a same-size in-place writer that restores the recorded
    mtime. No descriptor-level snapshot primitive exists in this process, so the
    record is validated as one stable regular-file read under a best-effort
    identity gate, not as a lock. (Option explicitly permitted by the review;
    no code claim is made beyond what is enforced.)

### Focused foreground commands and exact results (revision run)

```
.venv\Scripts\python.exe -m pytest tests/test_signature_sampling_persistence.py -o addopts= -p no:cacheprovider -q
  -> 67 passed in 1.30s (exit 0)

.venv\Scripts\python.exe -m pytest tests/test_route_b_diagnostic_runner.py -o addopts= -p no:cacheprovider -q
  -> 43 passed in 21.31s (exit 0)

.venv\Scripts\python.exe -m pytest tests/test_evaluation_same_axis_ownership_preflight.py -o addopts= -p no:cacheprovider -q
  -> 69 passed in 129.33s (exit 0)

.venv\Scripts\python.exe -m pytest tests/test_candidate_support_provenance.py tests/test_candidate_support_provenance_v2_preflight.py tests/test_signature_sampling_witness.py tests/test_candidate_support_ownership_harness.py -o addopts= -p no:cacheprovider -q
  -> 125 passed in 8.15s (exit 0)
```

### Frozen pins re-verified after the revision

* `.tmp-routeb-v2-frozen-r4-20261004/record.json` — 16,896,513 B,
  `006e8b6b16a312cdaa61a17b4336ce27c1abac590435645908ea9cfef2e998ee`
  (unchanged).
* `docs/reports/SOL-IMG-SAME-AXIS-OWNERSHIP-PREFLIGHT-20261005.json` — 5,804 B,
  `448c6debb9f6b36160b9a08ea1050fbf925813b378d688712b3dd55bb614201d`
  (unchanged; byte-identity asserted by
  `test_frozen_result_record_is_byte_identical_to_the_pinned_json`).
* Production pins asserted unchanged by
  `test_production_default_and_public_sources_stay_byte_identical`.

### Revision changed-file SHA-256 (lowercase; the report's own digest is recorded in the `AGENT_COMMS.md` append)

| Path | Bytes | SHA-256 |
| --- | --- | --- |
| `src/mc_photogrammetry/pipeline/signature_sampling_persistence.py` | 110,329 | `0d8fd306a070173cfaa0b6430e78d25beea277f04f2af255b76b837f818c1704` |
| `scripts/evaluation/run_route_b_diagnostic.py` | 64,689 | `f56357ad20d026be6d08c247491d1c5506ecbb52760037af89bcd9f03f6fc337` |
| `src/mc_photogrammetry/evaluation_same_axis_ownership_preflight.py` | 99,048 | `ab563e9d7a62e205b00d725fefb911730032eadfd5cdc31c1bca7d9ffb0f6bb6` |
| `tests/test_signature_sampling_persistence.py` | 61,507 | `a626433a8ead921d5a74b8b3dbd253c4525bac19df48ad05916a1f05fe8687b5` |
| `tests/test_route_b_diagnostic_runner.py` | 61,777 | `4b049ce22269b5f9961ef773786dd5dc366e5ff07f00aa335c5b3bf5d592c0b4` |
| `tests/test_evaluation_same_axis_ownership_preflight.py` | 42,884 | `11ff2f6a85b3c56a53040d76be6d2e9efa0fb847f1c73f806386f71fb0fb2e17` |

### Residual limitations after the revision

* No ownership/alignment/geometry/correctness claim and no accuracy percentage
  anywhere; ownership authorization remains withheld.
* No real V2 record exists yet; historical R4 was neither migrated nor recreated.
  Real future records still need a separately authorized ordinary source-image
  pass.
* `work_units` is a declared **byte-plus-structure** traversal/encoding proxy
  (payload bytes + metadata bytes + per-rectangle and per-array charges), not a
  measured instruction count; it is recomputed identically by the producer and
  the plain validator, and it deliberately retains the reviewed 20,000,000
  backstop (≈ 2.3x the aggregate payload cap plus metadata cap).
* The reader residual limitation in P2#10 above stands: a same-size in-place
  writer restoring mtime is not detected; the claim was narrowed, not the
  mechanism strengthened.
* The runner's post-write structural replay has no typed pins and therefore
  decodes with `require_ingress=False`; it never reports sampling availability.
  Availability is only ever granted by the preflight, which requires
  root-pinned ingress for every bound witness.
* Per-sample bilinear semantics are validated on typed/decoded arrays (producer
  account and decode), not on the plain declared pass, which cannot see weight
  values without decoding.
* The full-preflight availability test is gated on the optional frozen record
  fixture being present (`@requires_frozen`), like the existing frozen tests.
* Single frozen run with the universe cap equal to the row count still applies;
  no image/truth/camera/world/JAR/client/evaluator payload was opened and no
  inference/replay/matcher/solver/signature/network/production capture ran.

## REVISE-2 corrections (2026-10-05)

Task `DS-IMG-V2-SAMPLING-WITNESS-REVISE-2`. Only the four independently
reproduced defect groups were changed, inside the same seven owned paths; no
module outside the ownership list, no frozen artifact and no pinned JSON was
edited, and no image/evaluator/inference/replay/matcher/solver/signature/capture/
client/network/world/JAR action ran.

1. **Exact escaped per-entry metadata cap for refused and persistence-refused
   entries.** `validate_section_plain` now checks the exact escaped canonical
   JSON size (`_non_bound_entry_metadata_bytes` → `canonical_json_size`, which
   accounts for identities, ordinal, outcome, reason and detail) against
   `max_entry_metadata_bytes` for *both* the `refused` and the
   `persistence_refused` branches, before the section digest. 20,000 U+00E9
   characters are 20,000 raw characters but 120,000 escaped JSON bytes, so a
   raw-length gate alone cannot see the over-cap case; the new tests assert the
   typed refusal with `section_digest`/`_canonical_bytes` trapped (never called).
   The producer already degraded such an entry; `charge_refusal` additionally
   refuses the whole section when even the minimal degraded record would exceed
   the per-entry cap, so the producer can never emit a complete section whose own
   validator rejects it. Evidence:
   `test_escaped_refusal_detail_is_refused_before_the_section_digest`,
   `test_escaped_refusal_detail_degrades_to_a_small_typed_record`,
   `test_escaped_persistence_refused_detail_is_refused_before_digest`,
   `test_refusal_entries_never_exceed_the_per_entry_metadata_cap`.
2. **Section serialized cap before the digest, and the runner record cap before
   the artifact digest.** `validate_section_plain` now runs
   `_check_section_serialized_cap` (exact `canonical_json_size`, no string built)
   immediately before `_verify_section_digest` for both section statuses, so an
   over-cap section is refused before it is canonically serialized. The runner's
   `_write_record` now calls `_fit_sampling_section`, probes the complete record
   with a placeholder 64-hex digest and rejects `record_over_cap` *before* the
   `artifact_sha256` serialization. Evidence:
   `test_over_cap_complete_section_is_refused_before_any_serialization`,
   `test_section_serialized_cap_rejects_a_producer_oversized_record`,
   `test_write_record_rejects_over_cap_before_the_artifact_digest` (direct
   `_write_record` trap: runner `_canonical_bytes`/`_digest` never called).
3. **Declaration-before-traversal producer preflight.** `_preflight_contract_array`
   became `_declare_contract_array` (original-object dtype/rank/shape/nbytes read,
   no `np.ascontiguousarray`/`astype`/copy) and `_account_witness` no longer
   scans, materializes metadata lists, encodes or hashes: it only declares payload
   bytes, exact metadata size from cardinalities and precharged work, then applies
   the per-entry payload/metadata/work caps. The new `_validate_witness_values`
   performs the pointer/index/weight/bilinear scans and `_encode_witness` makes
   the contiguous copies, and the producer calls both only after the aggregate
   payload, entry-metadata, aggregate-metadata, serialized and aggregate-work
   budgets have passed. Every refusal branch now charges its declared work unit
   through `charge_refusal`, which enforces the aggregate work cap (plus the
   aggregate/per-entry metadata caps) before appending, so no refusal can bypass
   the remaining budget. Evidence: `test_work_refusal_makes_no_array_copy_or_value_scan`
   (traps `np.ascontiguousarray` and every value scanner at `max_work_units=1`:
   zero copies), `test_two_refusals_cannot_produce_a_complete_section_over_the_work_cap`
   (two refusals at `max_work_units=1` yield the typed work refusal; one refusal
   yields a complete section its validator accepts),
   `test_aggregate_payload_refusal_precedes_value_scan_and_encoding` (only the
   in-budget witness is encoded). `test_oversized_noncontiguous_array_is_refused_before_any_copy`,
   `test_work_cap_is_refused_before_any_value_traversal` and
   `test_rectangle_expansion_refuses_before_metadata_materialization` still pass.
4. **Exact primitive types before equality/dedup/digest.** The entry `ordinal`
   is required to be an exact `int` before the ordinal comparison, so `0.0` (and
   `False`) no longer satisfies ordinal `0`; `entry_count`, `payload_bytes`,
   `metadata_bytes` and `work_units` require exact `int` types before their
   equality/accounting comparisons; `outcome`, section `reason`/`field`/`detail`
   and refusal `reason` values require exact `str` types; the witness
   `gradient_edge_order`, rectification `inter_bits` and `weight_denominator`
   require exact `int` types. `interior_rectangles` (and `cell_rectangles`) are
   type-validated before `len()`/iteration, so `interior_rectangles=null` is a
   typed `persistence_rectangle_invalid` refusal rather than a raw `TypeError`.
   Evidence: `test_non_integer_ordinal_never_satisfies_an_integer_ordinal`,
   `test_accounting_scalars_require_exact_integers`,
   `test_null_rectangle_sequence_is_a_typed_refusal_not_a_type_error`,
   `test_float_backend_scalars_are_refused_as_schema_invalid` (all re-signed
   tamper tests).
5. **Prior fixes preserved, not weakened:** the source-image shape cap remains
   separate from the rectified-side cap and is re-bound to the pinned root
   ingress on decode; `decode_section` still requires root-pinned ingress for
   every bound witness before `available`; the per-sample bilinear producer
   semantics, the ownership-aware rollback and the explicitly narrowed
   stable-reader guarantee are unchanged.

### REVISE-2 focused foreground commands and exact results

```
.venv\Scripts\python.exe -m pytest tests/test_signature_sampling_persistence.py -o addopts= -p no:cacheprovider -q
  -> 87 passed in 1.61s (exit 0)

.venv\Scripts\python.exe -m pytest tests/test_route_b_diagnostic_runner.py -o addopts= -p no:cacheprovider -q
  -> 44 passed in 21.66s (exit 0)

.venv\Scripts\python.exe -m pytest tests/test_evaluation_same_axis_ownership_preflight.py -o addopts= -p no:cacheprovider -q
  -> 69 passed in 119.87s (exit 0)

.venv\Scripts\python.exe -m pytest tests/test_candidate_support_provenance.py tests/test_candidate_support_provenance_v2_preflight.py tests/test_signature_sampling_witness.py tests/test_candidate_support_ownership_harness.py -o addopts= -p no:cacheprovider -q
  -> 125 passed in 7.58s (exit 0)
```

### REVISE-2 changed-file SHA-256 (lowercase)

| Path | Bytes | SHA-256 |
| --- | --- | --- |
| `src/mc_photogrammetry/pipeline/signature_sampling_persistence.py` | 115,435 | `4f1c93ac80b57b3784144a02d3ba438d0940c77cec28f22e495ee175337450a1` |
| `scripts/evaluation/run_route_b_diagnostic.py` | 65,142 | `871c6739f8f061e82686b8fffe120c9919c07cf5daba7fd24aa2845a209ab90f` |
| `tests/test_signature_sampling_persistence.py` | 75,024 | `21f71e7f54ad0e56b78eaf30b5d72a4e28ab6cbcfab6bf84c88e6283edd182f3` |
| `tests/test_route_b_diagnostic_runner.py` | 63,357 | `551029800d84c92fe6418e4262e0403cd675fff3833912150110b8874dc0b267` |

`src/mc_photogrammetry/evaluation_same_axis_ownership_preflight.py` (99,048 B,
`ab563e9d7a62e205b00d725fefb911730032eadfd5cdc31c1bca7d9ffb0f6bb6`) and
`tests/test_evaluation_same_axis_ownership_preflight.py` (42,884 B,
`11ff2f6a85b3c56a53040d76be6d2e9efa0fb847f1c73f806386f71fb0fb2e17`) are
byte-identical to the REVISE run: no defect in the preflight path itself was
reproduced, so it was not edited. This report's own digest after the append is
recorded in the `AGENT_COMMS.md` entry.

### Frozen pins re-verified after REVISE-2 (unchanged)

* `.tmp-routeb-v2-frozen-r4-20261004/record.json` — 16,896,513 B,
  `006e8b6b16a312cdaa61a17b4336ce27c1abac590435645908ea9cfef2e998ee`.
* `docs/reports/SOL-IMG-SAME-AXIS-OWNERSHIP-PREFLIGHT-20261005.json` — 5,804 B,
  `448c6debb9f6b36160b9a08ea1050fbf925813b378d688712b3dd55bb614201d`
  (byte-identity still asserted by the preflight suite).

### Residual limitations after REVISE-2

* No ownership/alignment/geometry/correctness claim and no accuracy percentage
  anywhere; ownership authorization remains withheld for
  camera/intrinsics/pose/occlusion_convention/evaluator_domain.
* The section serialized cap is enforced as an exact *declared* size walk
  (`canonical_json_size`) before the digest; the digest itself still serializes
  the section once, now only for a section already known to fit.
* `work_units` remains a declared byte-plus-structure traversal/encoding proxy
  (payload bytes + metadata bytes + per-rectangle/per-array charges), recomputed
  identically by producer and plain validator, retaining the reviewed
  20,000,000 backstop. It is not a measured instruction count.
* Refusal outcomes are charged one declared work unit and are now checked
  against the aggregate and per-entry caps; a per-entry cap below even the
  minimal degraded refusal record yields a whole-section typed refusal rather
  than a complete section (documented above).
* The reader is still a best-effort stable-read identity gate, not an atomic
  snapshot against a same-size in-place writer restoring mtime; the
  post-write/structural replays still do not compute availability (only the
  preflight does, with root-pinned ingress for every bound witness); the
  full-preflight availability test remains gated on the optional frozen record
  fixture, and the injected reader seam is still trusted to return exact bytes.
* No real V2 record exists yet; historical R4 was neither migrated nor recreated;
  a real future capture still needs a separately authorized ordinary
  source-image pass. No image/truth/camera/mask/scene/world/Litematic/JAR/
  client/evaluator payload was opened, and no inference/replay/matcher/solver/
  signature/network/production capture ran.
* No junction/reparse creation, recursive deletion, encoded/obfuscated command,
  generated executable/installer, background/persistent process, Defender/system
  change, system-directory write or client/Prism launch was used.

## REVISE-3 corrections (2026-10-05)

Task `DS-IMG-V2-SAMPLING-WITNESS-REVISE-3`. Only the four remaining Sol/Astra
REVISE-2 findings were fixed, inside the same seven owned paths plus the
authorized `AGENT_COMMS.md` append. No module outside the ownership list, no
frozen artifact and no pinned JSON was edited, and no image/evaluator/inference/
replay/matcher/solver/signature/capture/client/network/world/JAR action ran.

1. **Reader ordering: canonical record cap and structural prescans precede the
   artifact digest.** `validate_record` now walks the exact declared canonical
   size (`persistence.canonical_json_size`, no string built) and refuses a record
   whose declared size plus newline exceeds `MAX_RECORD_BYTES` *before* `_digest`
   serializes the complete document, then runs `_prescan_record_document`
   (support-array prescan plus sampling section caps) and only then the artifact
   digest. A bounded raw read can still expand canonically because
   `ensure_ascii=True` escapes each non-ASCII code point as `\uXXXX`; the
   regression uses 6,000,000 U+00E9 code points (exactly 12,000,174 raw UTF-8
   bytes, 36,000,174 escaped JSON bytes) behind a mocked bounded-reader snapshot
   with a `_digest` trap, and proves the typed
   `persistence_serialized_over_cap`/`record` refusal happens with zero digest
   calls. The bounded read, token and depth caps are retained unchanged.
   Evidence: `test_validate_record_caps_canonical_size_before_the_artifact_digest`.
2. **Declared witness work is reserved before the value scans and never refunded
   on a semantic failure.** Producer work accounting now uses two counters:
   `total_work` is the serialized accounting written to the section and recomputed
   by `validate_section_plain`; `reserved_work` is budget-only and holds the
   declared work of a witness whose value scans were attempted but refused
   semantically. Every work check uses `total_work + reserved_work`, and the
   reservation is made *before* `_validate_witness_values`/`_encode_witness` and
   transferred into `total_work` only on success. The regression builds two
   structurally valid bindings whose last bilinear weight is changed (rebuilt via
   `dataclasses.replace`, `witness_sha256=""`), confirms the tamper fails only the
   per-sample weight contract, caps the work budget at one witness's declared work
   plus one, and proves with a scan counter that the second witness is refused by
   the aggregate work cap before any value scan. Evidence:
   `test_declared_witness_work_is_reserved_before_scans_and_never_refunded`.
3. **Non-bound refusal metadata traversal/encoding is charged as work.** A
   `refused`/`persistence_refused` outcome now charges one declared work unit
   *plus* its exact escaped metadata bytes, identically in `charge_refusal` and in
   both plain-validator non-bound branches. A refusal whose metadata exceeds the
   work budget can therefore no longer validate complete; the producer emits a
   whole-section typed `persistence_work_over_cap` refusal and the plain validator
   refuses the equivalent one-unit complete section before the section digest.
   Evidence: `test_non_bound_refusal_metadata_work_is_charged` (producer and
   validator agreement plus the `section_digest`/`_canonical_bytes` digest-order
   trap); the updated
   `test_two_refusals_cannot_produce_a_complete_section_over_the_work_cap` now
   records that even one small non-bound refusal cannot fit a one-unit work cap.
4. **No rectangle list materialization before the limits.** The producer no
   longer calls `_witness_metadata(..., materialize_rectangles=True)` to size the
   entry metadata. New `_declared_entry_metadata_bytes` measures the bound entry
   with the rectangle sequences empty (no per-cell list built) and adds back the
   exact `_rectangles_json_size` of each declared sequence, so the per-entry and
   aggregate metadata, serialized and work budgets are all applied before any
   rectangle list exists; `_encode_witness` builds the lists only after every
   budget passed. The regression caps the entry metadata between the witness
   metadata and the bound entry metadata, traps every
   `_witness_metadata(materialize_rectangles=True)` call, and proves the over-cap
   witness becomes a typed `persistence_entry_metadata_over_cap`
   `persistence_refused` entry with zero materializations. Evidence:
   `test_rectangle_lists_are_not_materialized_before_the_metadata_cap`; the
   earlier `test_rectangle_expansion_refuses_before_metadata_materialization`
   still passes.
5. **Prior fixes preserved:** the escaped per-entry metadata cap, the section
   serialized cap before the digest, the declaration-before-traversal producer
   preflight, exact primitive typing before equality/dedup/digest, the source
   shape/rectified-side split, the root-pinned ingress requirement, the per-sample
   bilinear producer contract, the ownership-aware rollback and the narrowed
   reader claim are unchanged.

### REVISE-3 focused foreground commands and exact results

```
.venv\Scripts\python.exe -m py_compile src/mc_photogrammetry/pipeline/signature_sampling_persistence.py scripts/evaluation/run_route_b_diagnostic.py tests/test_signature_sampling_persistence.py tests/test_route_b_diagnostic_runner.py
  -> exit 0

.venv\Scripts\python.exe -m pytest tests/test_signature_sampling_persistence.py -o addopts= -p no:cacheprovider -q
  -> 90 passed in 1.66s (exit 0)

.venv\Scripts\python.exe -m pytest tests/test_route_b_diagnostic_runner.py -o addopts= -p no:cacheprovider -q
  -> 45 passed in 22.84s (exit 0)

.venv\Scripts\python.exe -m pytest tests/test_evaluation_same_axis_ownership_preflight.py -o addopts= -p no:cacheprovider -q
  -> 69 passed in 118.22s (exit 0)

.venv\Scripts\python.exe -m pytest tests/test_candidate_support_provenance.py tests/test_candidate_support_provenance_v2_preflight.py tests/test_signature_sampling_witness.py tests/test_candidate_support_ownership_harness.py -o addopts= -p no:cacheprovider -q
  -> 125 passed in 7.44s (exit 0)
```

### REVISE-3 changed-file SHA-256 (lowercase)

| Path | Bytes | SHA-256 |
| --- | --- | --- |
| `src/mc_photogrammetry/pipeline/signature_sampling_persistence.py` | 119,596 | `4c289a060f25b35a7008328a9fea34549d6879d7d46f9ab489c3b355f7203820` |
| `scripts/evaluation/run_route_b_diagnostic.py` | 65,981 | `3936ec651af9c157afedb845c6312735fa6700f07aa0257283d2db162ef790b7` |
| `tests/test_signature_sampling_persistence.py` | 82,432 | `0c1ae1f4c75ec880a13ab45a3d3b5c4bbc46f233a1e27568fbad6918d4dc8e6e` |
| `tests/test_route_b_diagnostic_runner.py` | 65,153 | `c48c3851934b2d0c6f5cea7bd4329dbab6912e6b44691b5ebac4865dc3976c0d` |

`src/mc_photogrammetry/evaluation_same_axis_ownership_preflight.py` (99,048 B,
`ab563e9d7a62e205b00d725fefb911730032eadfd5cdc31c1bca7d9ffb0f6bb6`) and
`tests/test_evaluation_same_axis_ownership_preflight.py` (42,884 B,
`11ff2f6a85b3c56a53040d76be6d2e9efa0fb847f1c73f806386f71fb0fb2e17`) remain
byte-identical to the REVISE-2 run: no defect in the preflight path itself was
reproduced, so it was not edited. This report's own digest after the append is
recorded in the `AGENT_COMMS.md` entry.

### Frozen pins re-verified after REVISE-3 (unchanged)

* `.tmp-routeb-v2-frozen-r4-20261004/record.json` — 16,896,513 B,
  `006e8b6b16a312cdaa61a17b4336ce27c1abac590435645908ea9cfef2e998ee`.
* `docs/reports/SOL-IMG-SAME-AXIS-OWNERSHIP-PREFLIGHT-20261005.json` — 5,804 B,
  `448c6debb9f6b36160b9a08ea1050fbf925813b378d688712b3dd55bb614201d`
  (byte-identity still asserted by the preflight suite).

### Residual limitations after REVISE-3

* No ownership/alignment/geometry/correctness claim and no accuracy percentage
  anywhere; ownership authorization remains withheld for
  camera/intrinsics/pose/occlusion_convention/evaluator_domain.
* `work_units` remains a declared byte-plus-structure traversal/encoding proxy.
  For a bound entry it is payload bytes + metadata bytes + per-rectangle/per-array
  charges; for a non-bound outcome it is one unit plus exact metadata bytes. It is
  not a measured instruction count.
* The producer work *budget* (`total_work + reserved_work`) can exceed the
  serialized `work_units` after a semantic refusal: the retained declared work is
  deliberately not written to the record (the validator cannot recompute an
  absent witness), so the budget is a strict superset of the reported charge. It
  never lets a complete section exceed the cap.
* The canonical-size walk in `validate_record` is an extra full traversal of the
  parsed record before the digest; it visits only JSON scalars and allocates no
  serialized string.
* The reader is still a best-effort stable-read identity gate, not an atomic
  snapshot against a same-size in-place writer restoring mtime; the
  post-write/structural replays still do not compute availability (only the
  preflight does, with root-pinned ingress for every bound witness); the
  full-preflight availability test remains gated on the optional frozen record
  fixture, and the injected reader seam is still trusted to return exact bytes.
* No real V2 record exists yet; historical R4 was neither migrated nor recreated;
  a real future capture still needs a separately authorized ordinary source-image
  pass. No image/truth/camera/mask/scene/world/Litematic/JAR/client/evaluator
  payload was opened, and no inference/replay/matcher/solver/signature/network/
  production capture ran.
* No junction/reparse creation, recursive deletion, encoded/obfuscated command,
  generated executable/installer, background/persistent process, Defender/system
  change, system-directory write or client/Prism launch was used.

## REVISE-4 correction (2026-10-05)

Task `DS-IMG-V2-SAMPLING-WITNESS-REVISE-4`. Only Sol's exact serialized-size
boundary finding was fixed, inside the same seven owned paths plus the authorized
`AGENT_COMMS.md` append. No module outside the ownership list, no frozen artifact
and no pinned JSON was edited, and no image/evaluator/inference/replay/matcher/
solver/signature/capture/client/network/world/JAR action ran.

1. **Exact `section_sha256` reservation before sealing.** The producer's pre-seal
   gate in `persist_signature_witnesses` reserved an ad-hoc 80 bytes for the
   digest member that `_seal_section` appends, but that member
   (`"section_sha256":"<64 lowercase hex>"`) contributes exactly 84 canonical
   bytes: one comma + the 16-byte quoted key + a colon + the 66-byte quoted
   digest. At the reported `max_serialized_bytes=1084`, a refused-entry fixture
   whose sealed complete section is exactly 1,085 bytes passed the 80-byte gate
   (unsealed 1,001 + 80 = 1,081) and was sealed and canonicalized before the
   defensive byte check degraded it. The gate is now the exact allocation-free
   probe `_sealed_section_size(section)`: `canonical_json_size(section)` plus the
   digest member measured with a 64-character placeholder digest
   (`canonical_json_size("section_sha256")` and
   `canonical_json_size(_SECTION_DIGEST_PLACEHOLDER)`). The 84 bytes are derived,
   not hard-coded, and no JSON string or second mapping is built, so an over-cap
   complete section is refused *before* `_seal_section` or `_canonical_bytes`
   serializes anything. The trailing `len(_canonical_bytes(sealed))` check is
   retained as a defensive backstop. Evidence:
   `test_sealed_section_size_probe_is_the_exact_digest_member` and
   `test_over_cap_complete_section_is_refused_before_the_canonical_walk`. The
   second builds the reported fixture (one `SourceTapWitnessRefusalV1` with a
   300-character detail under `max_serialized_bytes=1084`,
   `max_entry_metadata_bytes=1084`, `max_total_metadata_bytes=1084`), proves the
   unsealed payload is exactly 1,001 canonical bytes and the sealed complete
   section exactly 1,085 bytes, traps `_canonical_bytes` for every would-be
   complete payload, and asserts zero complete canonicalizations, the typed
   `persistence_serialized_over_cap`/`section` refusal with no entries, and that
   the returned refusal still validates. The pre-fix behavior was re-simulated in
   the foreground with the old `canonical_json_size(section) + 80` probe; the trap
   fires exactly once, confirming the regression catches the defect.
2. **Prior fixes preserved:** the reader canonical-cap ordering, the reserved
   declared witness work, the charged non-bound refusal metadata, the
   declaration-only rectangle sizing, the escaped per-entry metadata cap, the
   section serialized cap before the digest, the declaration-before-traversal
   producer preflight, exact primitive typing, the source shape/rectified-side
   split, the root-pinned ingress requirement, the per-sample bilinear contract,
   the ownership-aware rollback and the narrowed reader claim are unchanged.

### REVISE-4 focused foreground commands and exact results

```
.venv\Scripts\python.exe -m py_compile src/mc_photogrammetry/pipeline/signature_sampling_persistence.py tests/test_signature_sampling_persistence.py
  -> exit 0

.venv\Scripts\python.exe -m pytest tests/test_signature_sampling_persistence.py -o addopts= -p no:cacheprovider -q
  -> 92 passed in 1.55s (exit 0)

.venv\Scripts\python.exe -m pytest tests/test_route_b_diagnostic_runner.py -o addopts= -p no:cacheprovider -q
  -> 45 passed in 22.63s (exit 0)

.venv\Scripts\python.exe -m pytest tests/test_evaluation_same_axis_ownership_preflight.py -o addopts= -p no:cacheprovider -q
  -> 69 passed in 118.68s (exit 0)

.venv\Scripts\python.exe -m pytest tests/test_candidate_support_provenance.py tests/test_candidate_support_provenance_v2_preflight.py tests/test_signature_sampling_witness.py tests/test_candidate_support_ownership_harness.py -o addopts= -p no:cacheprovider -q
  -> 125 passed in 7.51s (exit 0)
```

### REVISE-4 changed-file SHA-256 (lowercase)

| Path | Bytes | SHA-256 |
| --- | --- | --- |
| `src/mc_photogrammetry/pipeline/signature_sampling_persistence.py` | 121,243 | `d4164d8e0435f6e537a9c4fecd9d0d1fb36ed8b90ae9fea037d5d79c032a08d9` |
| `tests/test_signature_sampling_persistence.py` | 86,420 | `22528ad7641287cc3f0b04243f37c15ec218f3e6611024da8793edcd581501bd` |
| `scripts/evaluation/run_route_b_diagnostic.py` | 65,981 | `3936ec651af9c157afedb845c6312735fa6700f07aa0257283d2db162ef790b7` |
| `tests/test_route_b_diagnostic_runner.py` | 65,153 | `c48c3851934b2d0c6f5cea7bd4329dbab6912e6b44691b5ebac4865dc3976c0d` |

`src/mc_photogrammetry/evaluation_same_axis_ownership_preflight.py` (99,048 B,
`ab563e9d7a62e205b00d725fefb911730032eadfd5cdc31c1bca7d9ffb0f6bb6`) and
`tests/test_evaluation_same_axis_ownership_preflight.py` (42,884 B,
`11ff2f6a85b3c56a53040d76be6d2e9efa0fb847f1c73f806386f71fb0fb2e17`) remain
byte-identical to the REVISE-3 run: the correction is confined to the producer
gate and its regression, so the runner, preflight module and their tests were not
edited. This report's own digest after the append is recorded in the
`AGENT_COMMS.md` entry.

### Frozen pins re-verified after REVISE-4 (unchanged)

* `.tmp-routeb-v2-frozen-r4-20261004/record.json` — 16,896,513 B,
  `006e8b6b16a312cdaa61a17b4336ce27c1abac590435645908ea9cfef2e998ee`.
* `docs/reports/SOL-IMG-SAME-AXIS-OWNERSHIP-PREFLIGHT-20261005.json` — 5,804 B,
  `448c6debb9f6b36160b9a08ea1050fbf925813b378d688712b3dd55bb614201d`
  (byte-identity still asserted by the preflight suite).

### Residual limitations after REVISE-4

* No ownership/alignment/geometry/correctness claim and no accuracy percentage
  anywhere; ownership authorization remains withheld for
  camera/intrinsics/pose/occlusion_convention/evaluator_domain.
* The section serialized cap is now an exact declared-size walk that includes the
  pending 64-character `section_sha256` member before the digest, so the digest
  and the final serialization run only for a section already known to fit; the
  surviving `len(_canonical_bytes(sealed))` check is a defensive backstop that the
  exact probe should make unreachable for an over-cap section.
* `work_units` remains a declared byte-plus-structure traversal/encoding proxy
  (payload bytes + metadata bytes + per-rectangle/per-array charges; one unit plus
  exact metadata bytes for a non-bound outcome), recomputed identically by producer
  and validator, retaining the reviewed 20,000,000 backstop; the producer work
  budget (`total_work + reserved_work`) can exceed the serialized `work_units`
  after a semantic refusal and never lets a complete section exceed the cap.
* The canonical-size walk in `validate_record` remains an extra allocation-free
  traversal of the parsed record before the digest.
* The reader is still a best-effort stable-read identity gate, not an atomic
  snapshot against a same-size in-place writer restoring mtime; the
  post-write/structural replays still do not compute availability (only the
  preflight does, with root-pinned ingress for every bound witness); the
  full-preflight availability test remains gated on the optional frozen record
  fixture, and the injected reader seam is still trusted to return exact bytes.
* No real V2 record exists yet; historical R4 was neither migrated nor recreated;
  a real future capture still needs a separately authorized ordinary source-image
  pass. No image/truth/camera/mask/scene/world/Litematic/JAR/client/evaluator
  payload was opened, and no inference/replay/matcher/solver/signature/network/
  production capture ran.
* No junction/reparse creation, recursive deletion, encoded/obfuscated command,
  generated executable/installer, background/persistent process, Defender/system
  change, system-directory write or client/Prism launch was used.
