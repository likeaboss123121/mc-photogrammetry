# SOL-IMG-SAME-AXIS-OWNERSHIP-PREFLIGHT - evaluation-only V2 saved-support / ownership-alignment preflight

- Task: `AGENT_COMMS.md` 2026-10-05 Lead -> DeepSeek `DS-IMG-V2-SUPPORT-PREFLIGHT`
  (implementation of Sol's immediately preceding bounded preflight design).
- Follow-up: `AGENT_COMMS.md` 2026-10-05 Lead -> DeepSeek
  `DS-IMG-V2-PREFLIGHT-CAP-FIX` (fixes the two cap bypasses from Luna's
  immediately preceding read-only `REVISE`).
- Follow-up: `AGENT_COMMS.md` 2026-10-05 Sol REVISE + Lead -> DeepSeek
  `DS-IMG-V2-DTYPE-ALLOCATION-FIX` (pre-allocation dtype and scalar-leaf
  validation, per-array and aggregate payload-byte caps).
- Owner: DeepSeek CLI worker (direct shell, foreground commands only).
- Mode: **read-only over the ONE accepted frozen record** plus pure evaluation
  validation. No inference, capture, replay, screenshot, matcher, signature or
  solver execution.
- Owned paths written: this report, the focused test file, the preflight module,
  and one compact JSON result. No existing source, test, runner, pipeline,
  artifact or frozen input was modified.

## 0. Inputs read and pins

| input | bytes | SHA-256 |
|---|---|---|
| `.tmp-routeb-v2-frozen-r4-20261004/record.json` | 16,896,513 | `006e8b6b16a312cdaa61a17b4336ce27c1abac590435645908ea9cfef2e998ee` |

The named V2 index / digest / support sidecar sections are embedded in that one
document (`capture.index_pages`, `capture.digest_pages`,
`capture.sidecar_pages`). `.tmp-routeb-v2-frozen-r4-20261004/route_b.mcscan` was
**not** read.

In-record pins re-verified: `schema
mc-photogrammetry-route-b-diagnostic-record-v1`, `status complete`,
`expected_row_count 1,928`, `signature_witness_count 85`,
`whole_universe_identity_sha256 a5ddc8a6...9c0efc`,
`input_match_identity_sha256 28ad4ac4...0868105`,
`failed_assignment_sha256 ab4cb09b...1525dcd`,
`record_sha256 6eddf7f2...8ba797`,
`artifact_sha256 678475d6...3d5d41af`; all replay exactly.

No image, camera, intrinsics, pose, mask, truth, evaluator, scene, world,
Litematic, JAR, asset or network input was opened. The preflight imports only
pure V1/V2 contract modules.

## 1. Exact commands and result

```text
.venv\Scripts\python.exe -c "from mc_photogrammetry import evaluation_same_axis_ownership_preflight as pf; r = pf.run_same_axis_ownership_preflight(); print(r.status, r.reason)"
-> complete (no refusal); wall time ~3.9 s
```

The compact JSON result was produced from the same frozen record:

```text
.venv\Scripts\python.exe -c "... run_same_axis_ownership_preflight() ... write docs/reports/SOL-IMG-SAME-AXIS-OWNERSHIP-PREFLIGHT-20261005.json"
docs/reports/SOL-IMG-SAME-AXIS-OWNERSHIP-PREFLIGHT-20261005.json
  bytes 5,804 SHA-256 448c6debb9f6b36160b9a08ea1050fbf925813b378d688712b3dd55bb614201d
```

Caps applied: input 32 MiB, JSON depth 16, JSON tokens 6,000,000, rows 1,928,
witnesses 85, sidecar pages 67, payload elements 6,000,000, aggregate payload
bytes 48,000,000, per-array payload bytes 48,000,000, wall clock 60 s. The
frozen record measured `max_json_depth 10`, `json_tokens 4,987,420`,
`payload elements 4,576,118` / `payload bytes 36,608,944` (largest single array
59,232 bytes), so it fits with headroom.

## 2. Reproduced 1,928-row disposition / refusal cross-tabs

| disposition | support_state | support_refusal_reason | rows |
|---|---|---|---|
| `kept` | `present` | (null) | 26 |
| `kept` | `refused` | `support_payload_over_cap` | 11 |
| `kept` | `refused` | `source_tap_witness_unavailable` | 1 |
| `same_axis_mismatch` | `present` | (null) | 945 |
| `same_axis_mismatch` | `refused` | `support_payload_over_cap` | 833 |
| `same_axis_mismatch` | `refused` | `source_tap_witness_unavailable` | 112 |

Marginals reproduced exactly: kept 38, dropped 1,890, present 971, refused 957
(the 957 UNKNOWN rows), over-cap 844, witness-unavailable 113; 35 groups;
5 index pages; 6 digest pages (one per section, 1,928 digests each, 0 nulls);
67 sidecar pages covering 971 entries; `match_index` non-null exactly on the 38
kept rows; `support_payload_sha256` non-null exactly on the 971 present rows.
Every refused row is carried as UNKNOWN, never as incorrect.

Missing-witness / cap intersections reproduced exactly:

- 78 distinct signature digests are referenced across the two endpoints;
  59 have a bound witness, 19 do not; 7 bound witnesses reference no admitted row.
- 525 rows have at least one endpoint without a bound witness; 412 of those are
  reported `support_payload_over_cap` and 113 are reported
  `source_tap_witness_unavailable` (412 + 113 = 525 closes exactly).
- 102 of the 844 over-cap rows are provably over the raw cap from the persisted
  `logical_pairs` size alone (`32 * overlap_cells > 131,072`).
- Witness outcomes: 66 bound, 19 refused; the 19 refusals are
  `source_tap_witness_budget_exhausted` (15 `aggregate_rectified_samples`,
  3 `aggregate_cell_memberships`, 1 `aggregate_payload_bytes`).

## 3. Joins verified from the saved record

- **Ordered row identity**: `capture.index_pages` flatten to contiguous
  `admitted_row_index 0..1927`; each page digest, every digest section binding,
  the per-row `input_match_identity_sha256` section, the kept-subset identity
  digest and the `whole_universe_identity_sha256` all replay through the
  existing V2 `validate_complete_support_capture` (plus the independent
  plain-document prescan, which agrees on every count).
- **Envelope / support binding**: exactly one sidecar entry per present row, in
  admitted order, with `payload_sha256 == support_payload_sha256`,
  `overlap_cells == len(logical_pairs)` and entry shapes equal to payload shapes.
- **Endpoint-alternative**: each row's left signature digest binds a witness
  whose `frame_id == row.left` and `alternative_index == row.left_alternative_index`,
  and likewise on the right; the asymmetric (swapped source/target) case is
  refused.
- **Signature**: 85 distinct witnesses with no duplicate `signature_sha256` and
  no duplicate `(frame_id, canonical_index, alternative_index)`; all 19
  refusals carry a bounded reason code.
- **Ingress**: the entry-side pins, `capture.root.ingress`, the top-level
  `ingress` records and the positional `inputs` rows agree exactly; each bound
  witness `ingress_sha256` resolves to one ingress pin with a matching
  `frame_id`; every per-frame `transform.resolved_output_size` equals its
  decoded shape's `(width, height)`.
- **Source shape / transform / index / sentinel**: every present payload's
  `source_shape`/`target_shape` are internally consistent; `logical_pairs` are
  non-negative and within source/target bounds; the `warp_source_rows` /
  `warp_source_columns` flat source indices accept exactly the documented `-1`
  border sentinel and reject `< -1` or `>= source_shape[0] * source_shape[1]`.

## 4. Required per-frame alignment-pin inventory (typed)

Three frames: `image-0000`, `image-0001`, `image-0002`. For each frame:

| pin | state | source / typed reason |
|---|---|---|
| `source_transform` | available | `record.ingress.transform` (`source_size [2560,1369]`, `crop_origin [0,0]`, `crop_size null`, `output_size null`, `resolved_output_size [2560,1369]`) |
| `camera` | unavailable | `camera_pin_absent_from_frozen_record` |
| `intrinsics` | unavailable | `intrinsics_pin_absent_from_frozen_record` |
| `pose` | unavailable | `pose_pin_absent_from_frozen_record` |
| `occlusion_convention` | unavailable | `occlusion_convention_absent_from_frozen_record` |
| `evaluator_domain` | unavailable | `evaluator_domain_absent_from_frozen_record` |

## 5. Exact seam: the per-cell source-tap tables are not persisted

The bounded **support/tap surface** that the record does carry (the sidecar
`logical_pairs` plus the `warp_source_rows`/`warp_source_columns` flat
source-tap maps, with `-1` border semantics) reconstructs cleanly for all 971
present rows. What the record does **not** carry is the
`SignatureSamplingWitnessV1` payload that each bound source-tap binding only
hashes:

```text
signature_witnesses[].witness_sha256 has no persisted SignatureSamplingWitnessV1
payload (cell_rectangles/cell_source_indices/rectification tap tables)
```

`project_ownership_into_source_space` consumes exactly those per-cell tap tables
from the live `SourceTapWitnessBindingV1.witness`, and the four ingress pins are
frame-ingress only. The preflight therefore reports
`sampling_witness_state = unavailable`,
`ownership_authorization = withheld` with blockers
`sampling_witness_tap_tables_not_persisted`, `camera_pin_absent`,
`intrinsics_pin_absent`, `pose_pin_absent`, `occlusion_convention_absent`,
`evaluator_domain_absent`, and **stops at this seam** instead of re-running
screenshots. A missing or changed present-row sidecar entry is refused as
`source_tap_reconstruction_unavailable` naming the exact
`sidecar.entry.<admitted_row_index>` seam.

## 6. No ownership or correctness percentage

The preflight emits **no** ownership, correctness, recall or accuracy
percentage, and reads no evaluator payload. `38` kept rows, `971` present
payloads and `957` UNKNOWN rows are diagnostic support counts, not accuracy
scores; the withheld ownership authorization is a prerequisite refusal, not a
low score.

## 7. Tests (foreground)

```text
.venv\Scripts\python.exe -m pytest tests/test_evaluation_same_axis_ownership_preflight.py -o addopts= -p no:cacheprovider -q
-> 58 passed in 79.77s (exit 0)
```

Coverage: the positive frozen end-to-end preflight (all six cross-tabs, the
957 UNKNOWN rows, the 525/412/113 and 78/59/19/7 intersections, source
shape/index and `-1` sentinel, typed pin inventory, no-percentage check);
asymmetric left-source/right-target orientation (both a self-consistent unit
refusal and a file-level orientation-field tamper); tampered index row;
tampered signature digest section; tampered ingress (artifact re-signed, so the
ingress cross-bind is what refuses); duplicate admitted row; missing sidecar
payload at the exact seam; negative sentinel; cap-before-allocation (byte cap
with the reader proven not to run, depth/work scans, row and payload-element
caps with `payload_elements_allocated == 0`); and the evaluator-access gate
refusing before the frozen inference digest is sealed.

Cap-bypass regressions added by `DS-IMG-V2-PREFLIGHT-CAP-FIX`:

- `test_bounded_stream_read_is_capped_at_max_bytes_plus_one` - the chunked read
  loop returns exactly `max_bytes + 1` at most, and all bytes when under cap.
- `test_default_reader_never_uses_unbounded_read_bytes` - with
  `Path.read_bytes` replaced by a raising trap, `read_bounded_record_bytes`
  still returns the exact opened-file bytes.
- `test_linklike_check_is_fail_closed` - a regular file is accepted, an
  `lstat`-failing path is refused (no symlink/reparse point is created).
- `test_record_reader_refuses_simulated_growth_without_retaining_bytes` - the
  file is appended past the cap mid-read through the patched `os.read`; the
  refusal is `input_over_cap`, the refused result carries `input_bytes == 0` and
  `input_sha256 == ""`, and the loop never pulled more than `cap + 1` bytes.
- `test_record_reader_refuses_simulated_replacement` - the opened descriptor's
  `fstat` identity changes between the before/after checks; refused as
  `input_size_drift`.
- four `np.asarray`-trapped regressions: a declared shape smaller than the
  nested `values`, a short `values` row count, a nested-list leaf where a 2-D
  scalar is required, and an over-cap payload-element limit. Each is refused in
  the plain-document prescan with `payload_elements_allocated == 0` and the
  NumPy call is never reached.

Pre-allocation dtype/byte-cap/scalar-leaf regressions added by
`DS-IMG-V2-DTYPE-ALLOCATION-FIX` (32 new cases, both `np.asarray` and `np.dtype`
trapped; each asserts `payload_elements_allocated == 0`):

- `test_oversized_string_dtype_is_refused_before_np_dtype_or_asarray` - a
  tampered `U1048576` dtype on the valid `[147, 4]` array (which would request
  ~2.47 GB) is refused as `payload_dtype_invalid` with neither `np.dtype` nor
  `np.asarray` called.
- `test_any_non_contract_dtype_is_refused_before_numpy` (18 parametrisations) -
  `int64`, `i8`, `=i8`, `>i8`, `<i4`, `int32`, `<u8`, `uint8`, `float64`,
  `|u1`, `<U8`, `U8`, `S8`, `|b1`, `bool`, `None`, `8` and a list are all
  refused; no incidental NumPy alias is broadened into the contract.
- `test_per_array_payload_byte_cap_refuses_before_numpy` - a 4,703-byte
  per-array cap refuses the 4,704-byte first array as `payload_over_cap` at
  `payload.logical_pairs.bytes`.
- `test_aggregate_payload_byte_cap_refuses_across_arrays_before_numpy` - a
  4,704-byte aggregate cap lets the first array through and refuses the second
  (4,608 bytes) at `payload_bytes`, proving the aggregate budget is charged
  across arrays.
- `test_non_integer_leaf_is_refused_before_numpy` (6 parametrisations) - float
  (`1.0`, `1.5`), `bool` (`True`, `False`), string and `null` leaves are refused
  as `payload_value_invalid` rather than silently truncated by the `<i8`
  conversion.
- `test_out_of_range_integer_leaf_is_refused_before_numpy` - a leaf above
  `PAYLOAD_MAX_ABS_ELEMENT` is refused as `tap_index_invalid`.
- `test_logical_pairs_negative_sentinel_is_refused_before_numpy` - `-1` in
  `logical_pairs` is refused as `tap_index_invalid`.
- `test_minus_one_is_preserved_only_in_border_sentinel_arrays` - `-1` is
  accepted and returned by `_array_from_record` for a border-sentinel array and
  refused for a non-sentinel array.
- `test_conversion_overflow_is_mapped_to_a_typed_refusal` - a raised
  `OverflowError` from the conversion is mapped to a typed
  `sidecar_replay_mismatch` refusal instead of escaping.
- `test_contract_dtype_and_payload_byte_caps_are_exact` pins
  `PAYLOAD_CONTRACT_DTYPE == "<i8"`, `PAYLOAD_CONTRACT_ITEM_BYTES == 8` and
  `MAX_PAYLOAD_BYTES == 48,000,000`.

Targeted regression (existing suites, unchanged):

```text
.venv\Scripts\python.exe -m pytest tests/test_candidate_support_provenance_v2_preflight.py tests/test_candidate_support_provenance.py -o addopts= -p no:cacheprovider -q
-> 112 passed in 7.58s (exit 0)
.venv\Scripts\python.exe -m pytest tests/test_route_b_diagnostic_runner.py -o addopts= -p no:cacheprovider -q
-> 37 passed in 14.16s (exit 0)
```

## 8. Changed-file hashes

| file | bytes | SHA-256 |
|---|---|---|
| `src/mc_photogrammetry/evaluation_same_axis_ownership_preflight.py` | 91,957 | `3207c8a53063e5b76dbf312ee006fb2efee4d8365928a4192253b27eeaf072f8` |
| `tests/test_evaluation_same_axis_ownership_preflight.py` | 29,838 | `5298b8539830c50c34c4f5669548fa877769874ce87869ec0c5f3cfbb2d4dd9a` |
| `docs/reports/SOL-IMG-SAME-AXIS-OWNERSHIP-PREFLIGHT-20261005.json` | 5,804 | `448c6debb9f6b36160b9a08ea1050fbf925813b378d688712b3dd55bb614201d` (unchanged) |
| `docs/reports/SOL-IMG-SAME-AXIS-OWNERSHIP-PREFLIGHT-20261005.md` | (this file) | reported in the AGENT_COMMS result |

## 9. Limitations and non-claims

- **No ownership/alignment result.** The preflight proves the saved support and
  tap surface replay and inventories what is missing; it makes no camera,
  registration, occlusion, ownership, geometry or correctness claim.
- **Persisted-digest limits.** `match_sha256`, `envelope_sha256` and
  `matcher_witness_sha256` are persisted digests whose preimages are not in the
  record; the preflight verifies their ordered presence, lengths and the
  sidecar/support binding, not a recomputation of those three sections.
- **Sub-gate opacity preserved.** `support_payload_over_cap` still aggregates
  the four V2 cap branches; only the 102 raw-cap-provable rows are recoverable
  from the record.
- **Single frozen run.** The universe cap equals the exact row count, so the
  profile is frozen to 1,928 rows; no determinism or generalisation claim.
- **No screenshot rerun.** Because the sampling-witness tap tables are absent,
  the preflight reports the seam and stops; it does not re-capture or replay
  signatures to manufacture availability.
- **Stable-read scope.** The bounded opened-file reader proves a single stable
  regular-file snapshot on this platform: it rejects symlinks/reparse points by
  `lstat`, bounds every read at `max_input_bytes + 1`, and refuses size/identity
  drift. It does not claim protection against an adversarial same-identity
  writer that restores size/mtime between checks, and the injected `reader`
  seam (used only by tests and embedded callers) is trusted to return `bytes`.
- **Declared-shape vs actual values.** The prescan verifies exact nested
  `values` row/element counts against the declared `shape`, the remaining
  payload-element and payload-byte budgets, exact JSON-integer leaves (float and
  `bool` excluded), the contract numeric range and the exact serialized contract
  dtype `"<i8"` before NumPy is called. `_array_from_record` repeats the dtype,
  byte-cap and leaf/range validation immediately before `np.dtype`/`np.asarray`
  as defense in depth; the post-allocation `min`/`max` checks are retained but
  are now unreachable for an accepted array.

## 10. Direct-shell compliance

Per `docs/deepseek-worker.md` and `docs/deepseek-direct-shell.patch.yml`: no
junction/reparse-point operation, no recursive deletion, no encoded/obfuscated
command, no generated executable/installer, no background process or
persistence, no Defender or system change, no system-directory write, no
client/Prism launch, no network, and no other agent path edited concurrently.
Only the four owned paths were written, plus the authorized `AGENT_COMMS.md`
result append. Stop for Luna read-only review.

## 11. Cap-bypass correction (`DS-IMG-V2-PREFLIGHT-CAP-FIX`)

Luna's immediately preceding `REVISE` identified two ways an over-cap input
could still be allocated before the cap check. Both are corrected inside the
same three owned paths, with the frozen record pins/counts/joins, the 957
UNKNOWN rows, the no-score behavior, the report-only scope and the evaluator
gate all unchanged (the regenerated result JSON is byte-identical, SHA-256
`448c6debb9f6b36160b9a08ea1050fbf925813b378d688712b3dd55bb614201d`).

1. **Stat-then-read growth/replacement bypass.** `read_bounded_record_bytes`
   used `os.stat(path).st_size` followed by `Path(path).read_bytes()`, so a file
   replaced or grown after the stat could be read at unbounded size. It now uses
   `_read_bounded_file_snapshot`: reject symlinks/reparse points by `lstat`,
   require a regular file within `max_input_bytes`, open one descriptor
   (`O_RDONLY`, plus `O_BINARY`/`O_NOFOLLOW` where the platform defines them),
   compare the pre-open `lstat` to the descriptor `fstat`, read in fixed
   `1 MiB` chunks that never exceed `max_input_bytes + 1`, then re-check the
   descriptor `fstat` and post-read `lstat` identity/size. Growth is refused as
   `input_over_cap`; replacement/identity or size drift is refused as
   `input_size_drift`. An injected `reader` still gets the metadata pre-cap and
   the post-read type/size caps. The reviewed
   `dataset/runtime_render_provenance.read_bounded_file_snapshot` helper was
   evaluated but not called: its fixed `ProvenanceError` taxonomy, fixed
   `MAX_PACK_SCAN_SECONDS` deadline and lack of the injected `clock`/`reader`
   seams cannot express this module's typed refusals, so the same contract is
   mirrored locally (`_linklike_path`, `_same_file_identity`,
   `_read_stream_bounded`).
2. **Declared-shape vs nested-values bypass.** `_prescan_structure` only summed
   declared `shape` products, and `_array_from_record` called `np.asarray` before
   reshaping. A small declared shape with a much larger nested `values` list
   could therefore allocate an over-cap NumPy array before the reshape failed.
   The shared `_validate_nested_values` now walks the plain nested `values`
   level by level before NumPy: every level must be a list whose length equals
   the declared dimension, leaves must be scalars, the leaf count must equal the
   declared product, and counting stops the moment the remaining
   `max_payload_elements` budget is exceeded. Both the prescan and
   `_array_from_record` enforce it, so `np.asarray` is never reached on a
   mismatched or over-cap nesting.

No image, truth, camera, world, JAR, client, evaluator payload, network or
inference path was accessed; no other source path was edited.

## 12. Pre-allocation dtype / byte-cap / scalar-leaf correction (`DS-IMG-V2-DTYPE-ALLOCATION-FIX`)

Sol's review of `DS-IMG-V2-PREFLIGHT-CAP-FIX` found two remaining
pre-allocation failures. Both are fixed inside the same three owned paths; the
frozen record pins/counts/joins, the 957 UNKNOWN rows, the no-score behavior,
the report-only scope and the evaluator gate are all unchanged, and the
regenerated result JSON is byte-identical (SHA-256
`448c6debb9f6b36160b9a08ea1050fbf925813b378d688712b3dd55bb614201d`, 5,804
bytes, `json.dumps(..., indent=1, sort_keys=True)` plus a trailing newline,
compared in memory against the persisted file).

1. **P1 - dtype allocation.** `_prescan_structure` and `_array_from_record` both
   ignored `dtype` before `np.asarray`, so a tampered `U1048576` on a valid
   `[147, 4]` array requested ~2.47 GB. The V1/V2 contracts require `np.int64`
   (`dtype != np.int64` in `candidate_support_provenance` /
   `candidate_support_provenance_v2`), whose `ndarray.dtype.str` is `"<i8"` on
   this little-endian platform, and all 2,913 payload arrays in the frozen
   record persist exactly `"<i8"`. The new `PAYLOAD_CONTRACT_DTYPE = "<i8"` is
   required as an exact string in both passes before `np.dtype`/`np.asarray`;
   NumPy aliases (`int64`, `i8`, `=i8`, `>i8`, ...) are deliberately not
   accepted. `PAYLOAD_CONTRACT_ITEM_BYTES = 8` and explicit
   `max_payload_bytes` / `max_payload_array_bytes` limits (default
   `MAX_PAYLOAD_BYTES = 48,000,000` = 6,000,000 elements x 8 bytes) are charged
   per array and cumulatively during the plain-document prescan, with a
   defensive repeat in `_array_from_record`.
2. **P2 - scalar conversion.** `_validate_nested_values` now requires every leaf
   to be an exact JSON integer (`type(item) is int`, so `bool` is rejected) and
   enforces the contract numeric range `[-PAYLOAD_MAX_ABS_ELEMENT,
   PAYLOAD_MAX_ABS_ELEMENT]` plus the `-1` border-sentinel rule before NumPy:
   `-1` is accepted only in `warp_source_rows`/`warp_source_columns`, never in
   `logical_pairs`. The NumPy/payload conversion paths now map `OverflowError`;
   the capture-construction wrappers still catch only their listed errors and
   have no demonstrated input-reachable `OverflowError` path.
   New typed reasons: `payload_dtype_invalid`, `payload_value_invalid` (both
   added to `REFUSAL_REASONS`).

Measured frozen behavior after the change: `complete`, no reason; 1,928 rows /
971 present / 957 UNKNOWN / 38 kept / 1,890 dropped; 4,576,118 payload elements
allocated; 36,608,944 contract payload bytes under the 48,000,000-byte caps;
971 sidecar entries; 85 witnesses; six cross-tabs identical to the frozen
tuple; gate sealed; ownership authorization withheld; no ownership/accuracy
percentage. No image, truth, camera, world, JAR, client, evaluator payload,
network or inference path was accessed; no other path was edited.
