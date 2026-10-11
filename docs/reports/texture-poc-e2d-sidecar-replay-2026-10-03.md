# E2D optional E2C sidecar validation and E1 pixel-parity replay

Date: 2026-10-03
Task: `texture-poc-e2d-sidecar-replay-reader-2026-10-03`
Status: implemented, focused `py_compile` + tests pass, stopped for lead review.
This report records the extension design and its synthetic evidence. It is **not**
a corpus result, it runs no T2/T6 benchmark, and it makes no promotion
recommendation. The task was not marked complete.

## 1. Scope and owned files

The task extends the evaluator-only E2B reader
(`scripts/evaluation/replay_texture_observation_trace.py`) with an **optional**
E2C production patch sidecar input. The sidecar
(`mc-photogrammetry-texture-patch-sidecar-v1`, written by
`raycast.reconstruct_scene_surfaces` on the opt-in `stage_sidecar` path) carries
the actual resolved-face sampler payloads bound to the E2A schema-2 observation
ordinals. The reader now validates that document against the exact trace bytes,
the trace's ordinal/identity/digest contract, and the fixed E2C bounds, then
joins it to the sealed E1 artifact by exact key so raw-patch, observed-patch,
valid-mask and coverage parity each get their own denominator. With no
`--sidecar`, the E2B output is unchanged.

Owned files (only these three were edited or created):

| Path | SHA-256 |
| --- | --- |
| `scripts/evaluation/replay_texture_observation_trace.py` | see section 7 |
| `tests/test_texture_observation_trace_replay.py` | see section 7 |
| `docs/reports/texture-poc-e2d-sidecar-replay-2026-10-03.md` | this report |

No production source, test, threshold, palette, gate or E1 fixture was touched.
The reader still reads no screenshot, reference litematic, camera manifest, mask
or JAR, runs no benchmark or corpus evaluation, writes no artifact, and uses no
reference label in reconstruction.

CLI (unchanged when `--sidecar` is omitted):

```
python scripts/evaluation/replay_texture_observation_trace.py \
    <e1-artifact.json> <stage-trace.json> --role {template-2|template-6} \
    [--sidecar <production-patch-sidecar.json>]
```

A refusal still exits `2` with a typed `TraceReplayError` on stderr and writes
nothing.

## 2. Sidecar fail-closed contract

`load_sidecar(path, *, trace)` validates the sidecar against one already-loaded
schema-2 trace before any join:

* the file exists, and its byte size is at or below `SIDECAR_MAX_BYTES`
  (64 MiB) **before** it is parsed;
* `schema == mc-photogrammetry-texture-patch-sidecar-v1` and
  `trace_schema == 2` (an `int`; `true`/`2.0` are rejected);
* `trace_sha256` is a SHA-256 and `trace_bytes` a non-negative integer that
  exactly equal the supplied trace file's own SHA-256 and byte count, so the
  sidecar must describe the exact E2A trace bytes it is joined to;
* the three header counts are integers and reconcile exactly:
  `observation_count == len(trace.observations)`,
  `excluded_fallback_observations ==` the trace's `centre_patch_fallback` count,
  `resolved_face_records == len(records)`;
* `records` is a list and its length is at or below `SIDECAR_MAX_RECORDS`
  (10,000), checked **before** any payload array is decoded;
* `len(records) ==` the trace's `resolved_face` observation count: exactly one
  sidecar row per resolved-face observation, with fallbacks excluded and
  count-reconciled;
* each row's `ordinal` is unique, maps to a `resolved_face` trace observation,
  and its `(frame_id, coordinate, slot)` identity and `source` agree with the
  trace observation at that ordinal; duplicate ordinals and duplicate identities
  are refused before any join;
* `raw_patch`/`observed_patch` are decoded only at 16x16x3 `<f4`,
  `valid` at 16x16 `|b1` and `coverage` at 16x16 `<f4`, with bounded, strictly
  validated base64; short/over-long payloads, wrong dtype/shape and non-finite
  floats are refused;
* every digest in the sidecar's `digests` block is recomputed from the decoded
  bytes and compared to the stored value (the stored digest is never trusted),
  and the recomputed `observed_patch`/`valid`/`coverage` digests must equal the
  trace observation's digests at the same ordinal. `raw_patch` has no trace
  counterpart by design.

The ordinal set is re-checked against the resolved-face trace ordinals at the
end, so a missing, extra or misplaced row is refused rather than partially
joined.

## 3. Join and parity denominators

`_reconcile_sidecar` joins the validated sidecar to E1 by the exact
`(frame_id, coordinate, slot)` key. It reports identity and pixel evidence
separately, so an identity match is never read as a pixel match:

| Field | Meaning | Denominator |
| --- | --- | --- |
| `counts.sidecar_resolved_rows` | validated sidecar payload rows | sidecar document |
| `counts.trace_resolved_face_rows` | `resolved_face` trace observations | trace |
| `counts.e1_patch_keys` | sealed E1 patch rows | E1 artifact |
| `counts.identity_joins` | exact sidecar ∩ E1 key set | exact-key join |
| `counts.e1_only_keys` | E1 keys with no sidecar row | E1 keys |
| `counts.sidecar_only_keys` | sidecar rows with no E1 key | sidecar keys |
| `parity.raw_patch` | decoded sidecar `raw_patch` vs E1 stored raw patch | joined keys |
| `parity.observed_patch` | decoded sidecar `observed_patch` vs E1's reconstructed production float32 observed input | joined keys |
| `parity.valid` | decoded sidecar valid mask vs E1 valid mask | joined keys |
| `parity.coverage` | decoded sidecar coverage vs E1 coverage | joined keys |
| `parity.all_four` | all four above at once | joined keys |
| `parity.mismatches` | exact mismatch count plus up to ten example rows | joined keys |

`ordinal_identity_reconciliation` records the reconciled row count, the
resolved-face trace count and the excluded fallback count. Every rate is
`{numerator, denominator, rate}` with `rate: null` on an empty denominator. The
existing E2B `join`, `digest_equality`, `truth_rank`, `per_cell`,
`aggregation_replay` and `production_top_k_check` sections are untouched and are
not fed by the sidecar parity. The E1 per-face truth rank and the production
per-cell outcomes remain separate.

## 4. Synthetic evidence

`tests/test_texture_observation_trace_replay.py` mirrors the E2B golden join
(four E1 patches, four resolved trace observations, one `centre_patch_fallback`)
and adds a fully parity-matching E2C sidecar for the four resolved rows. The
golden test asserts: 4 sidecar rows, 5 trace observations, 4 resolved faces, 1
excluded fallback; E1 4 keys; 3 identity joins; 1 E1-only key (the sampler
domain E1 saw but production did not); 1 sidecar-only key (production sampled it,
E1 did not); and `raw_patch`/`observed_patch`/`valid`/`coverage`/`all_four` all
`3/3`. Matching the trace bytes is asserted directly via the reported
`trace_sha256_bound`/`trace_bytes_bound`.

One-field mutants: for each of `raw_patch`, `observed_patch`, `valid` and
`coverage`, exactly one joined row's payload bytes (and, where the trace carries
that field, the trace observation digest) are changed so the sidecar stays
internally consistent and trace-consistent. Each mutant is **reported**, not
refused: the targeted parity field drops to `2/3`, the other three stay `3/3`,
`all_four` is `2/3`, one mismatch example is emitted, and the identity join is
still `3` — the explicit "identity match does not imply pixel match" case.

Fallback handling: a sidecar with zero records for a trace whose only
observation is a `centre_patch_fallback` that collides with an E1 key is
accepted; the fallback is excluded and count-reconciled, the sidecar parity
denominator is `0` (`rate: null`), and the E2B fallback-collision semantics are
unchanged. Unhandled cases are refused: a missing resolved row, a row at a
fallback ordinal, duplicate ordinals, duplicate identities, identity/source/trace
digest disagreement, a stored digest that does not match its decoded bytes, bad
base64/shape/dtype/payload length/non-finite arrays, bad header counts, a bad
`trace_schema`, a mismatched trace SHA-256 or byte count, both fixed bounds,
a bad sidecar schema, and the preserved E1 schema/seal/role safeguards.

## 5. Checks performed

Per the task only the two owned Python files were compiled and only the focused
replay test module was run, in the foreground:

```
.venv\Scripts\python.exe -m py_compile \
  scripts\evaluation\replay_texture_observation_trace.py \
  tests\test_texture_observation_trace_replay.py
-> exit 0

$env:PYTHONPATH='src'; .venv\Scripts\python.exe -m pytest \
  tests\test_texture_observation_trace_replay.py -o addopts='' -q --no-header
-> 46 passed
```

The 29 prior E2B regressions still pass; 17 new E2D cases were added (14 test
functions, one parametrized four ways).

## 6. Deviations, limitations and prohibitions observed

* The sidecar is strictly optional: with no `--sidecar` the summary has no
  `sidecar` key and no sidecar limitation line, so the existing E2B output is
  unchanged.
* The 10,000-record / 64 MiB bounds are re-checked independently in the reader
  against the serialized document; they mirror E2C's `STAGE_SIDECAR_MAX_RECORDS`
  and `STAGE_SIDECAR_MAX_BYTES` rather than importing production code.
* Raw sidecar bytes are compared to E1's stored raw patch and sidecar observed
  bytes to E1's reconstructed production float32 observed input, so a low parity
  rate is a measured sampler/geometry interface gap, not a reader failure.
* Identity joins are reported separately from every parity rate; no label is used
  in reconstruction, no threshold or gate is changed, and nothing is promoted.
* No corpus/T2/T6/benchmark run, dataset factory, Minecraft client/capture or
  model training. No install/download, shell nesting, job/workflow/subagent,
  junction/reparse point, recursive deletion, encoded command, generated
  executable, background process, persistence, Defender/system change or
  launcher automation.
* No real E1 artifact, trace or sidecar was read for this task; all evidence is
  synthetic and self-contained in `tmp_path`.
* Unresolved precondition for the first real run: a pinned T2/T6 E2A schema-2
  trace plus its E2C sidecar must be produced by the lead at fresh immutable
  paths. This reader does not produce either.

## 7. Files and hashes

Recorded from the final working-tree content (SHA-256):

| Path | SHA-256 |
| --- | --- |
| `scripts/evaluation/replay_texture_observation_trace.py` | `a34ed6d02e6fe578b9fd19adbaa3ff0ad72c2a82aec4be3bb81aa3bdbbab4baa` |
| `tests/test_texture_observation_trace_replay.py` | `46be9c72bab5c60d74eda035eff3836a770a1564c8de614f92e100c03cdf235f` |
| `docs/reports/texture-poc-e2d-sidecar-replay-2026-10-03.md` | this report (hash reported with the handoff, not embedded) |

Only the three owned paths changed; pre-existing dirty hunks elsewhere were
preserved.

Stopped for lead review. The task is not marked complete, and no corpus or
promotion claim is made.
