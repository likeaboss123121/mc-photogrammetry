# E2C bounded, opt-in production resolved-face patch sidecar

Date: 2026-10-03
Task: `texture-poc-e2c-production-patch-sidecar-2026-10-03`
Status: implemented, lead-review remediations applied (sections 7 and 8),
focused tests + `py_compile` pass, stopped for lead review. This report records
the sidecar contract and its synthetic evidence. It is **not** a corpus result, it
runs no T2/T6 benchmark, and it makes no promotion recommendation. The task was
not marked complete. Section 7 records the three lead-review contract gaps and
the first remediation; section 8 records the data-parity and write-ordering
remediation.

## 1. Scope and owned files

The task asked for a separate, opt-in evaluation sidecar carrying production's
*actual* sampled pixels for resolved face observations, so an evaluator can
compare them with E1 labels and prove sample parity. The sidecar is keyed to the
E2A schema-v2 observation ordinals, carries the sampler's own `observed_patch`,
`valid` and `coverage` arrays, is refused rather than truncated past its fixed
record/byte bounds, and is read back by no reconstruction decision. Default
reconstruction, the existing schema-v2 trace and every diagnostic are unchanged.

Owned files (only these five were created or edited):

| Path | Pre-edit SHA-256 | Reviewed snapshot SHA-256 | Post-remediation SHA-256 | Post-parity-remediation SHA-256 |
| --- | --- | --- | --- | --- |
| `src/mc_photogrammetry/reconstruction/raycast.py` | `5e859e9c3d02b1914e8444c6c452c091143a02d3e2dae02f7b0776741cbc8d1b` | `9c5b35c5c9a8f7453fde75f547da45231278f253fca7a3e463aa1a227d56c06b` | `ae16325c885fedc9a2cbe77fa8dd2c47b1b3b1370af2074c05d8fdbf7fae7e99` | `44864cbeb56ae1d171acb9cb4c95256c9c456896fe4127f676010a550eb88ceb` |
| `tests/test_raycast.py` | `27b1f14cbf472f322a5882550ece4c5466f94daeda6ba4412901d7dce7d13a46` | `6cd6c11aed8306dc613f2536d2d0d0b95a5532d16770e682837696ae8f693ce7` | `39d9857533e902b804c4789e8c02335688763d4856f7cde2d116e108443fc37d` | `5cefedbe973024394b3b6315f40f003db4becbda4b5ce634649c9bb4d6d65e45` |
| `scripts/benchmark_templates.py` | `e58f3866513ed6656e8771e004f20ef0fb1e85017794b30e22fc626fb97965af` | `d3872ad52a36f9f1b83af54cba086e388e4badf78f634decac0598a634bf153a` | `cc789690beeeb1142955f6888ead08c5f2253345ff7bf10a86fb27df763d2f82` | `a960171a1c19a99d21d51d3f9d27f546a37003e7088fae6295277595806926fd` |
| `tests/test_benchmark_templates.py` | `1cbbd8051f466272b8907398f8a42782b13b64b8c31e85535f8a56743efe0f44` | `3cd9aeec8d351c54b19fc000a258e8b71f4cdc4f70fd9ddefa463d431214cee1` | `a49eb2527445c0cc6ffcbbd44be83ec9c2ba043ae5b79366958a49754938c5b2` | `e4e88f9b958164eeeec403172f8934fd4113c75aa6a11ac365cfd06520d331e1` |
| `docs/reports/texture-poc-e2c-production-patch-sidecar-2026-10-03.md` | (new) | this report | this report | this report |

*Post-parity-remediation* is the current content after section 8; the hashes are
re-derived after the final edit of this report's body and recorded there.

The *reviewed snapshot* is the exact content the lead reviewed and found the
three contract gaps in; *post-remediation* is the current content after section
7. The pre-edit hashes are the working-tree content before the original E2C task
edits (which already contained the uncommitted E2A trace work). An isolated diff of each owned
file against those baselines shows **additions only** in `raycast.py`,
`test_raycast.py` and `test_benchmark_templates.py`, and additions plus the small
trace-write/call-site refactor described in section 4 in
`benchmark_templates.py`; no pre-existing dirty hunk was moved, reverted or
rewritten.

## 2. Reconstruction-side contract (`raycast.py`)

New constants:

- `STAGE_SIDECAR_SCHEMA = "mc-photogrammetry-texture-patch-sidecar-v1"`.
- `STAGE_SIDECAR_MAX_RECORDS = 10_000`.
- `STAGE_SIDECAR_MAX_BYTES = 64 * 1024 * 1024` (64 MiB).

New optional keyword `stage_sidecar: dict[str, object] | None = None` on
`reconstruct_scene_surfaces`. It is refused before any work when supplied without
`stage_trace`:

```
stage_sidecar requires stage_trace; the sidecar is keyed to the schema-2
trace's observation ordinals and digests
```

Capture happens only inside the existing resolved-face branch of pass two, at the
production sampler call site. `sidecar_records` is `None` unless the caller opted
in (an opt-in allocates the list; the default path allocates no container at all),
so on the ordinary traced and untraced paths this branch and `_sidecar_record`
are unreachable. Within the opt-in branch the fixed record bound is enforced
*before* the extra record's arrays are copied, digested or encoded: reaching
`STAGE_SIDECAR_MAX_RECORDS` raises and refuses the run rather than accumulating a
payload the serializer would only reject later. The new `_sidecar_record`
carries, for each final retained `resolved_face` observation:

- `ordinal` (the raw-list ordinal, index-aligned with the E2A observation),
  `frame_id`, `coordinate`, `slot`, `source = "resolved_face"`;
- the sampler scalars `area`, `depth`, `visible`, `facing`, `weight`;
- `raw_patch`, `observed_patch`, `valid`, `coverage` as self-describing
  `{dtype, shape, base64}` records (`_array_payload`), lossless C-order bytes;
- a `digests` block (`raw_patch`, `observed_patch`, `valid`, `coverage`) computed
  by the same `_array_digest` shape/dtype-bound SHA-256 the E2A trace uses, so the
  `observed_patch`/`valid`/`coverage` digests equal the trace observation's digests
  at the same ordinal. The `raw_patch` digest has no trace counterpart by design.

`raw_patch` is the sampler's own `FaceSample.patch`, the array E1 retains and the
only authority for exact raw sampler-patch parity. `observed_patch` is
`FaceSample.observed_patch`, i.e. the scorer-facing array the E2A trace digests
(invalid pixels replaced by the coverage-weighted `rgb`); `valid` and `coverage`
are the sampler's own masks. Both patches are carried because they are not the
same array: for a partial mask the trace digest describes the neutralised array,
so a sidecar holding only `observed_patch` could never prove raw-patch parity. For
an all-valid patch the two arrays (and their digests) coincide. No array is
mutated (only `np.ascontiguousarray` plus `tobytes`), and no reference label,
evaluator truth or reconstruction result is read or recorded.

`_emit_stage_sidecar` fills the caller-owned mapping once, at the end of the same
`trace_enabled` block that emits the trace:

```json
{
  "schema": "mc-photogrammetry-texture-patch-sidecar-v1",
  "trace_schema": 2,
  "observation_count": <all retained raw observations>,
  "resolved_face_records": <payload record count>,
  "excluded_fallback_observations": <centre_patch_fallback count>,
  "records": [ ... ]
}
```

A `centre_patch_fallback` observation has no rectified face, gets no payload and
is excluded from `records`; it is counted explicitly in
`excluded_fallback_observations` so the reader can reconcile the sidecar against
the trace's full observation list. When `stage_sidecar is None`, nothing is
copied, encoded or digested for the sidecar.

## 3. Benchmark API and CLI (`benchmark_templates.py`)

`run(..., stage_sidecar_path: Path | None = None)` is the new optional path; the
CLI flag is `--stage-sidecar PATH`. Both default to disabled and the report's
`effective_settings` records `stage_sidecar: false` (plus `stage_sidecar_records:
null`) for an ordinary invocation. The setting changes no inference input, metric,
decision or gate.

Checked before any fixture input is read:

1. `stage_sidecar_path` without `stage_trace_path` is refused.
2. A sidecar path that resolves to the trace path is refused. `_paths_alias`
   reduces both arguments to `os.path.realpath(...)` (so a symlinked/reparse
   parent, `.`/`..` or a lexical alias that reaches the same file is caught) and
   then applies `os.path.normcase`, preserving Windows case-insensitivity and
   POSIX case-sensitivity.
3. A sidecar destination that already exists is refused with `FileExistsError`;
   nothing is written and the existing file is untouched.

Write stage (inside the existing trace-write block; ordering corrected in
section 8):

- the exact trace output bytes are computed before anything is written.
  `_trace_output_bytes` replicates the platform newline translation the previous
  `Path.write_text` writer applied, and the trace is written with `write_bytes`
  from those same bytes, so the trace file content is unchanged byte-for-byte
  while becoming available to preflight;
- when a sidecar is requested, those exact bytes are bound into the header as
  `trace_sha256` (SHA-256) and `trace_bytes` *before* either destination exists;
- `_stage_sidecar_payload` serializes `json.dumps(_json_safe(document),
  indent=2, sort_keys=True) + "\n"` deterministically to UTF-8 and refuses, before
  any file is opened, when the record count exceeds 10,000 or the serialized byte
  length exceeds 64 MiB. This is the second line of defence: the record bound is
  already enforced at capture time inside `raycast.py` (section 2), so an
  over-limit run refuses during reconstruction and never reaches this serializer.
  The byte bound is checked here too.
- only after the sidecar has serialized and validated are the trace destination
  and then the sidecar written;
- `_write_stage_sidecar` writes the payload to a uniquely named, task-owned
  temporary file in the destination directory and links it to the destination
  with `os.link`, which fails atomically if the destination exists (so a race
  cannot overwrite). On any failure the temporary file is removed; the
  destination exists only when the full payload landed. It re-checks the alias
  and existing-destination conditions before writing.

Refusal order note: the record bound is enforced at capture time, so an
over-the-record-bound run refuses during reconstruction. The serialized *byte*
bound is now measured during the preflight, before the trace is written, so a
byte-bound refusal leaves neither the trace nor the sidecar destination nor any
temporary file. The up-front checks in items 1-3 refuse before any fixture input,
report or trace is produced.

No trace-writing or trace-identity validation was weakened: the trace serializer
was only factored into a local `trace_text` variable and its output bytes are
reproduced exactly by `_trace_output_bytes`, and the sidecar is a separate file
that can never alias the trace.

## 4. Checks performed

`py_compile` (all four owned Python files, exit 0):

```
.venv/Scripts/python.exe -m py_compile \
  src/mc_photogrammetry/reconstruction/raycast.py tests/test_raycast.py \
  scripts/benchmark_templates.py tests/test_benchmark_templates.py
```

Focused tests after the section 8 remediation (`addopts` overridden to show the
summary; exit 0):

```
.venv/Scripts/python.exe -m pytest tests/test_raycast.py -o addopts="" -q --no-header
35 passed in 1.37s

.venv/Scripts/python.exe -m pytest tests/test_benchmark_templates.py -o addopts="" -q --no-header
110 passed in 0.60s

.venv/Scripts/python.exe -m pytest tests/test_raycast.py tests/test_benchmark_templates.py -o addopts="" -q --no-header
145 passed in 1.76s
```

New raycast tests (9):

- `test_stage_sidecar_requires_the_e2a_trace` - `stage_sidecar` without
  `stage_trace` is refused.
- `test_stage_sidecar_carries_only_resolved_faces_and_mirrors_the_trace` -
  record count equals the trace's resolved-face count, fallback observations are
  excluded and counted, each record's ordinal/identity and
  `observed_patch`/`valid`/`coverage` digests equal the E2A entry, and the added
  `raw_patch` payload/digest has the same shape/dtype as the observed patch.
- `test_stage_sidecar_round_trips_exact_arrays_and_digest_parity` - the document
  survives `json.dumps`/`json.loads` unchanged, the decoded raw and observed
  arrays are exactly `float32`/`float32`/`bool`/`float32` at the frozen shapes,
  re-derived digests equal both the sidecar and trace digests, and no
  label/truth key is present.
- `test_stage_sidecar_raw_and_observed_patch_come_from_the_same_sample`
  (section 8 gap 1) - a synthetic `FaceSample` with two invalid pixels gives a
  deterministic raw-vs-observed difference; the decoded `raw_patch` equals
  `sample.patch` exactly, the decoded `observed_patch` equals the
  neutralised array exactly, both digests are re-derived, and the record proves
  the two arrays differ (so storing one twice would fail).
- `test_stage_sidecar_raw_patch_matches_the_real_sampler_iteration`
  (section 8 gap 1) - the real `sample_visible_faces` is wrapped (not replaced)
  and every captured sample is paired with its sidecar record by
  `(coordinate, slot)`; each decoded `raw_patch` equals `sample.patch` exactly and
  each observed digest equals the sample's own `observed_patch` digest.
- `test_stage_sidecar_leaves_output_diagnostics_and_trace_identical` -
  sidecar-off and sidecar-on runs have identical volume signature, diagnostics
  and full schema-v2 trace content.
- `test_stage_sidecar_is_not_requested_by_default` - an ordinary traced run
  produces the unchanged v2 trace and carries no sidecar data.
- `test_default_path_never_runs_the_sidecar_record_helper` (remediation gap 1) -
  `_sidecar_record` is replaced by an exploding helper, the default traced run
  still succeeds (so it was never called), and a positive control proves the same
  replacement is reachable once a sidecar is opted in.
- `test_stage_sidecar_enforces_the_record_bound_at_capture` (remediation gap 3) -
  the cap is lowered one below the fixture's resolved-face count; the run refuses
  with a record-bound error, `_sidecar_record` runs exactly `cap` times (no record
  is constructed past the bound), and the caller-owned sidecar mapping is empty
  (no truncated sidecar).

New benchmark tests (15):

- CLI parses `--stage-sidecar` (default `None`).
- `stage_sidecar` without a trace path is refused and writes nothing.
- A sidecar path that names the trace is refused (plus a case-insensitivity unit
  check for `_paths_alias`).
- `test_paths_alias_resolves_a_linked_parent_canonically` (remediation gap 2) -
  with a deterministic `realpath` mapping, two lexically distinct paths through
  linked/real parents compare canonically equal, which the old `abspath` +
  `normcase` check missed. No symlink/reparse point is created.
- `test_stage_sidecar_refuses_a_canonically_aliased_trace` (remediation gap 2) -
  `run` refuses two spellings that resolve to the same linked file and writes no
  trace, sidecar or output.
- An existing destination is refused without touching it or writing the trace.
- `effective_settings` reports `stage_sidecar: false` / `stage_sidecar_records:
  null` by default.
- The enabled path writes a deterministic, atomic sidecar whose `trace_sha256`
  equals the SHA-256 of the trace file's actual bytes and whose `trace_bytes`
  matches, with the opt-in buffer reaching reconstruction and the setting
  reported; the sample record carries the `raw_patch` payload beside the observed
  one.
- `test_trace_output_bytes_match_the_previous_text_writer` (section 8 gap 2) -
  `_trace_output_bytes` is byte-identical to what the previous text-mode writer
  produced (including the platform newline translation) and hashes the same, so
  preflighting did not change trace output bytes.
- `test_stage_sidecar_byte_bound_refusal_creates_neither_trace_nor_sidecar`
  (section 8 gap 2) - an end-to-end mocked `run()` with a zero byte bound refuses
  after binding the exact trace digest and leaves no trace, no sidecar, no
  temporary artifact and no report.
- Over the 10,000-record bound and over the byte bound both refuse before writing;
  a malformed document is refused.
- A simulated `os.link` failure leaves no destination and no temporary artifact.

## 5. Prohibitions observed

No corpus/T2/T6/benchmark run, dataset factory, Minecraft/client/capture or model
training. No threshold/palette/matcher/pipeline change, no dependency or
download, no shell nesting, jobs/workflows/subagents, junction/reparse point,
recursive deletion, encoded/obfuscated command, generated executable, background
process, persistence, Defender/system change or launcher automation. Only
foreground `py_compile` and the two focused test modules were run.

## 6. Deviations and unresolved concerns

- The fixed record/byte bounds are module constants, not CLI-configurable; the
  task specified fixed limits, so no bound-override argument was added.
- The record bound is enforced during reconstruction and the serialized byte
  bound is now enforced during the preflight, before either destination is
  written (section 8, gap 2), so a byte-bound refusal leaves no trace and no
  sidecar. The candidate `.litematic` output is produced earlier by the export
  stage and is not part of the sidecar contract.
- `stage_sidecar` is populated in place by `reconstruct_scene_surfaces` and the
  benchmark adds the trace-identity fields after reconstruction, matching how the
  existing `stage_trace` mapping is filled in place.

## 7. Lead-review remediation (2026-10-03)

The lead reviewed the snapshot hashes in section 1 and requested exactly three
contract fixes. Only `raycast.py`, `benchmark_templates.py`, `test_raycast.py`,
`test_benchmark_templates.py` and this report were touched; no other dirty hunk
was moved or rewritten and no E2C behavior changed.

1. **No sidecar allocation while disabled.** `sidecar_records` is now
   `list | None`, initialised to `None` unless `stage_sidecar` was supplied, so
   the ordinary traced and untraced paths allocate no payload container. The
   capture branch is keyed on `sidecar_records is not None` and the emission
   guard requires both the mapping and the list, so `_sidecar_record` and
   `_emit_stage_sidecar` are unreachable without an opt-in.
   `test_default_path_never_runs_the_sidecar_record_helper` pins this.
2. **Canonical path aliasing.** `_paths_alias` now compares
   `os.path.normcase(os.path.realpath(path))` through the new `_canonical_path`,
   so two lexically distinct arguments whose parents resolve to the same
   symlinked/reparse target are refused. Windows case-insensitivity and POSIX
   case-sensitivity are preserved.
   `test_paths_alias_resolves_a_linked_parent_canonically` (deterministic
   `realpath` mock, no symlink/reparse creation) and
   `test_stage_sidecar_refuses_a_canonically_aliased_trace` pin this.
3. **Capture-time record bound.** Inside the opt-in capture branch the bound is
   checked with `len(sidecar_records) >= STAGE_SIDECAR_MAX_RECORDS` *before*
   `_sidecar_record` runs, so an over-limit run refuses without constructing,
   copying, digesting or encoding the offending record. The serializer-side
   `_stage_sidecar_payload` bound check is retained as a second line of defence.
   `test_stage_sidecar_enforces_the_record_bound_at_capture` pins the refusal,
   the exact construction count and the absence of any truncated sidecar.

The remediation's reviewed-snapshot and post-remediation hashes are in section 1.
Its focused checks are the `py_compile` and 33 + 108 + 141-passed runs printed in
section 4.

## 8. Data-parity and preflight remediation (2026-10-03)

The lead reviewed the section 7 snapshot, accepted the three fixes above and
requested exactly two further changes on the same five owned paths. Jev's official
TypeSafe review recommended HOLD on the same two points; no other dirty hunk was
moved or rewritten and no E2C behavior outside these two contracts changed.

1. **Raw sampler patch carried beside the observed patch.** `_sidecar_record` now
   takes both `FaceSample.patch` (the raw rectified array E1 retains) and
   `FaceSample.observed_patch` (the scorer-facing array the E2A trace digests) and
   serializes each as its own lossless `{dtype, shape, base64}` payload, with a
   `raw_patch` digest added to the `digests` block. `observed_patch`/`valid`/
   `coverage` remain tied to the E2A observation digests at the same ordinal; the
   `raw_patch` digest is sidecar-only because the trace never carries raw pixels.
   Both arrays are read-only: capture passes `sample.patch` and the existing
   `observed_patch`, and `_array_payload` only copies a non-contiguous view to read
   its bytes, so no array is mutated and nothing new reaches the trace.
   `test_stage_sidecar_raw_and_observed_patch_come_from_the_same_sample` uses a
   synthetic `FaceSample` with two invalid pixels to force a deterministic
   raw-vs-observed difference and asserts the exact decoded arrays (an
   implementation that stored one array twice would fail);
   `test_stage_sidecar_raw_patch_matches_the_real_sampler_iteration` wraps the real
   `sample_visible_faces`, pairs each captured sample with its sidecar record and
   asserts decoded-array equality; the round-trip test now decodes and re-digests
   the raw array as well. E1 raw-patch parity is therefore establishable, while
   `test_stage_sidecar_leaves_output_diagnostics_and_trace_identical` still pins
   that the E2A trace content is unchanged.
2. **Preflight before writing either destination.** The write block now computes
   the exact trace output bytes first (`_trace_output_bytes`), binds
   `trace_sha256`/`trace_bytes` from those bytes, serializes and bound-checks the
   sidecar, and only then writes the trace followed by the sidecar. The trace is
   written with `write_bytes` from those same preflighted bytes, which
   `test_trace_output_bytes_match_the_previous_text_writer` proves are
   byte-identical to the previous `Path.write_text` output (including the platform
   newline translation), so the trace content is unchanged. The sidecar remains
   atomic and no-overwrite through `os.link`.
   `test_stage_sidecar_byte_bound_refusal_creates_neither_trace_nor_sidecar` is an
   end-to-end mocked `run()` whose serialized sidecar exceeds a zero byte bound; it
   refuses and leaves no trace path, no sidecar path, no `.tmp` artifact and no
   report. The existing trace file-content digest test is retained and still
   passes, and the record bound remains enforced at capture time inside
   `raycast.py`.

Section 8 hashes (recomputed after the final body edit): `raycast.py`
`44864cbeb56ae1d171acb9cb4c95256c9c456896fe4127f676010a550eb88ceb`; `test_raycast.py`
`5cefedbe973024394b3b6315f40f003db4becbda4b5ce634649c9bb4d6d65e45`;
`benchmark_templates.py`
`a960171a1c19a99d21d51d3f9d27f546a37003e7088fae6295277595806926fd`;
`test_benchmark_templates.py`
`e4e88f9b958164eeeec403172f8934fd4113c75aa6a11ac365cfd06520d331e1`.
Its focused checks are the `py_compile` exit 0 and the 35 + 110 + 145-passed runs
in section 4.

Stopped for lead review. The task is not marked complete, and no end-to-end or
promotion claim is made.
