# E2F bounded numeric patch similarity for the optional E2C sidecar replay

Date: 2026-10-03
Task: `texture-poc-e2f-numeric-patch-similarity-2026-10-03`
Status: implemented and remediated (R1 denominator consistency), focused
`py_compile` + tests pass, stopped for lead review.
This report records the extension design and its synthetic evidence. It is **not**
a corpus result, it runs no T2/T6 benchmark, reads no image/reference/mask/camera
or JAR, and makes no promotion recommendation. The task was not marked complete.

## 1. Scope and owned files

The task extends the evaluator-only E2B/E2D reader
(`scripts/evaluation/replay_texture_observation_trace.py`) with a **bounded,
deterministic numeric similarity** section beside the existing exact digest
parity. When `--sidecar` is supplied, the same validated, decoded E2C payload
arrays that feed parity are compared per joined resolved-face row. With no
`--sidecar` the output is unchanged.

Owned files (only these three were edited or created):

| Path | Role |
| --- | --- |
| `scripts/evaluation/replay_texture_observation_trace.py` | reader implementation |
| `tests/test_texture_observation_trace_replay.py` | synthetic-only focused tests |
| `docs/reports/texture-poc-e2f-numeric-patch-similarity-2026-10-03.md` | this report |

No production source, threshold, palette, gate, E1 fixture, camera, image,
reference, mask or JAR was touched. The reader still reads no screenshot,
reference litematic, camera manifest, mask or JAR, runs no benchmark or corpus
evaluation, writes no artifact, and uses no reference label in reconstruction.
Tolerant RGB similarity is **not** semantic parity and never substitutes for the
exact digest parity, the E1 per-face truth rank or any gate.

## 2. Numeric per-row contract

`numeric_patch_similarity(e1_arrays, sidecar_arrays, *, label)` compares the
decoded arrays of one joined `(frame_id, coordinate, slot)` pair. Shape is part
of the contract: patch and observed patch must be `(H, W, 3)` RGB, `valid` and
`coverage` must be `(H, W)`, and every value must be finite. Any shape mismatch
or non-finite value raises `TraceReplayError` before a number is produced.

Per-row fields (RGB error is in float32 channel-value units, accumulated in
float64, over the row's own measured `H*W` pixels — 256 for every fixed
`16x16` production resolved-face row, and the measured count for any smaller
synthetic row):

| Field | Meaning |
| --- | --- |
| `raw_rgb_mae_all_pixels` | mean abs difference of decoded sidecar `raw_patch` vs E1 stored raw `patch` over all pixels |
| `raw_rgb_mae_both_valid` | same, over the intersection valid in **both** masks; `null` when the intersection is empty |
| `observed_rgb_mae_all_pixels` | mean abs difference of decoded sidecar `observed_patch` vs E1's reconstructed production float32 observed input, over all pixels |
| `observed_rgb_mae_both_valid` | same, over the both-valid intersection; `null` when empty |
| `coverage_mae` | mean abs difference of sidecar `coverage` vs E1 `coverage` over all pixels |
| `valid_mask_agreement` | fraction of pixels whose valid masks agree (1.0 = identical mask) |
| `patch_pixel_count` | per-row measured patch pixel denominator (256 for a fixed 16x16 production row, 4 for the 2x2 synthetic rows) |
| `both_valid_pixel_count` | per-row both-valid intersection denominator (0 to the row's patch pixel count) |
| `key` | `{frame_id, coordinate, slot}` the row was joined on |

A real `0.0` MAE and an empty intersection are different outcomes: the empty
intersection is counted and its intersection MAE is serialized as `null`, never
fabricated as zero.

## 3. Aggregates and explicit denominators

`summarize_numeric_rows(rows)` builds the `sidecar.numeric_similarity` section:

* `joined_key_denominator` — joined resolved-face rows the section covers.
* `rows_with_shape_mismatch` — `0` by contract: any mismatched shape is refused
  by `numeric_patch_similarity` before aggregation, so nothing is silently
  dropped.
* `rows_with_empty_both_valid_intersection` — rows whose both-valid intersection
  is empty.
* `pixel_denominators` — `patch_pixels_per_row` **derived from the measured
  `patch_pixel_count` of the joined rows**, plus
  `patch_pixels_per_row_uniform`, `distinct_patch_pixel_counts`,
  `patch_pixel_comparisons`, `coverage_pixel_comparisons`,
  `valid_mask_pixel_comparisons` and `both_valid_pixel_comparisons`, each summed
  over the measured rows. The reported per-row size and the summed totals are
  therefore always mutually consistent: uniform rows report their one measured
  size (`256` for the fixed `16x16` production rows, `4` for the `2x2` synthetic
  rows), and rows that genuinely mix shapes report `patch_pixels_per_row: null`
  with `patch_pixels_per_row_uniform: false` and every distinct measured size in
  `distinct_patch_pixel_counts`. An empty row set reports `null` and `[]` rather
  than a fabricated `256`, so no summary silently assumes the production shape.
* `per_row_metrics` — for each of the six per-row metrics a deterministic
  `{row_denominator, min, median, p90, p99, max}`. `row_denominator` is the count
  of rows actually contributing a non-null value, so a both-valid metric can have
  a smaller denominator than `joined_key_denominator` when intersections are
  empty. With no contributing row every statistic is `null`.
* `rows` — one entry per joined row, bounded by the existing fixed E2C
  `SIDECAR_MAX_RECORDS` (10,000) record bound, so per-row reporting cannot grow
  without an input that E2C itself would refuse.
* `shape_contract` and `note` — the explicit per-row shape rule (each row measures
  its own `(H, W, 3)` patch, production rows are fixed `16x16x3` and measure 256
  pixels per row, mixed shapes are listed rather than averaged) and the statement
  that tolerant RGB similarity is not semantic parity.

The section is nested under `sidecar` and is separate from `sidecar.parity`; the
existing E2B sections (`join`, `digest_equality`, `truth_rank`, `per_cell`,
`aggregation_replay`, `production_top_k_check`) and the E2D fields (`parity`,
`counts`, `ordinal_identity_reconciliation`) are unchanged and are not fed by the
numeric metrics. `replay` gains one extra limitation line only when a sidecar is
present.

## 4. Synthetic evidence

All evidence is synthetic and self-contained in `tmp_path`; no real E1 artifact,
trace or sidecar was read. Seven focused test functions cover the numeric
section (53 total in the module, up from 46; R1 added the mixed-shape
regression):

* `test_numeric_patch_similarity_arithmetic_all_pixels_and_intersection` — hand-
  computed 2x2x3 arrays assert raw MAE 2.5 all-pixels / 1.0 both-valid, observed
  MAE 5.0 / 2.0, coverage MAE 1.0, mask agreement 0.5, pixel denominator 4 and
  intersection denominator 1.
* `test_numeric_patch_similarity_empty_intersection_is_counted_and_null` —
  disjoint masks give `both_valid_pixel_count == 0`, `raw/observed_..._both_valid
  is None`, all-pixel MAE still computed, mask agreement 0.0, and the aggregate
  reports one empty-intersection row with a null `row_denominator: 0` distribution.
  Its summary also asserts the **measured** `patch_pixels_per_row == 4`,
  `patch_pixels_per_row_uniform is True` and `distinct_patch_pixel_counts == [4]`,
  proving the reported row size matches the 2x2 row's own `patch_pixel_count`
  instead of the hardcoded `16x16`.
* `test_numeric_patch_similarity_refuses_shape_mismatch_missing_and_nonfinite` —
  patch `(2,2,2)`, observed `(2,2,1)`, `valid (2,3)`, a missing array and a NaN
  patch each raise `TraceReplayError` (`shape mismatch` / `missing the ... array`
  / `non-finite`).
* `test_numeric_summary_handles_mixed_row_shapes_explicitly` (R1) — one measured
  2x2 row and one measured 16x16 row aggregate to
  `patch_pixels_per_row is None`, `patch_pixels_per_row_uniform is False`,
  `distinct_patch_pixel_counts == [4, 256]`, and 260 for each of the
  patch/coverage/valid-mask/both-valid totals; the `shape_contract` text names the
  distinct-count rule and the fixed 16x16 production shape; an empty row set
  reports `null`/`False`/`[]` and zero totals rather than a fabricated 256.
* `test_sidecar_numeric_similarity_reports_every_row_and_quantiles` — two of three
  joined sidecar raw patches are shifted by +2 and +4; per-row MAE is 2/4/0, the
  partial row's intersection denominator is 254 (vs 256), aggregate raw
  min/median/p90/p99/max is 0 / 2 / 3.6 / 3.96 / 4, pixel denominators are
  768 patch / 766 both-valid, the uniform production row size is asserted as
  `patch_pixels_per_row == 256` with `distinct_patch_pixel_counts == [256]`, and
  **the exact digest parity section still reports the byte mismatch
  independently** (`raw_patch` 1/3, `all_four` 1/3).
* `test_sidecar_numeric_similarity_counts_empty_both_valid_intersection` — an
  all-invalid production mask on an all-valid E1 patch yields one empty
  intersection, null both-valid MAE and a zero both-valid pixel denominator.
* `test_sidecar_numeric_similarity_is_finite_json_and_nested` — `json.dumps(...,
  allow_nan=False)` succeeds, no `NaN`/`Infinity` text appears, every per-row and
  aggregate value is `None` or finite, the section exists only under `sidecar`,
  and the E2D identity-join and parity counts are unchanged.

## 5. Checks performed

Only the two owned Python files were compiled and only the focused replay test
module was run, in the foreground:

```
.venv\Scripts\python.exe -m py_compile \
  scripts\evaluation\replay_texture_observation_trace.py \
  tests\test_texture_observation_trace_replay.py
-> exit 0

$env:PYTHONPATH='src'; .venv\Scripts\python.exe -m pytest \
  tests\test_texture_observation_trace_replay.py -o addopts='' -q --no-header
-> 53 passed
```

The scoped whitespace check
`git diff --check -- scripts/evaluation/replay_texture_observation_trace.py tests/test_texture_observation_trace_replay.py docs/reports/texture-poc-e2f-numeric-patch-similarity-2026-10-03.md`
was clean (no output, exit 0) on the three owned paths. Because these three
E2F paths are new/untracked (`??` in `git status`) and therefore invisible to
`git diff`, each owned file was also scanned directly for trailing spaces/tabs:
0 matching lines in all three. The unscoped `git diff --check` (exit 2) reports
trailing whitespace **only** in the pre-existing dirty, unowned `AGENT_COMMS.md`
working-tree hunk; no owned path appears in that report. No other tracked file
was edited by this task.

## 6. Deviations, limitations and prohibitions observed

* The three owned paths are the only files changed; all unrelated dirty hunks
  were preserved. No commit was made.
* The reader remains byte-for-byte unchanged for the E2B path without
  `--sidecar`: the new limitation line and `numeric_similarity` key appear only
  when a sidecar is supplied.
* A shape mismatch is refused at the function boundary. Because `load_sidecar`
  and `load_e1_artifact` already pin every array to its frozen shape/dtype, no
  well-formed input can reach the join with a mismatch; the aggregate therefore
  reports `rows_with_shape_mismatch: 0` as an explicit invariant, and the refusal
  itself is proven by the direct function test.
* R1 denominator consistency: `patch_pixels_per_row` is the measured uniform row
  size, not a fixed constant. In production every `load_sidecar`/`load_e1_artifact`
  row is pinned to `16x16`, so it is always `256`; the helper's shape-generic
  synthetic rows report their own measured size, and genuinely mixed rows report
  `null` plus `distinct_patch_pixel_counts`. Summed pixel totals always equal the
  sum of the measured per-row counts, so per-row size, totals and the
  `shape_contract` text agree. The numeric comparison remains per-row and never
  mixes pixels across rows.
* Numeric similarity is tolerant and diagnostic. It does not replace, relax or
  reinterpret the exact digest parity, the E1 per-face truth rank, the production
  per-cell outcomes or any threshold/gate.
* No production files, images, references, masks, cameras, scene, JAR, corpus
  benchmark, Minecraft client/capture or model training were used. No install,
  download, shell nesting, job/workflow/subagent, junction/reparse point,
  recursive deletion, encoded command, generated executable, background process,
  persistence, Defender/system change or launcher automation was performed.
* Only synthetic arrays built in `tmp_path` were used; no real E1/trace/sidecar
  or fixture was read.

## 7. Files and hashes

Recorded from the final working-tree content (SHA-256, uppercase):

| Path | SHA-256 |
| --- | --- |
| `scripts/evaluation/replay_texture_observation_trace.py` | `26323DA2E881E46CAB53C22B06A203DE859F9B3F71FDE28F56003975878A2674` |
| `tests/test_texture_observation_trace_replay.py` | `6C59D5CBCC7EB4DEE082C60744BF5E162F75A5D4D03F0CABA00AD92ACAFA8C99` |
| `docs/reports/texture-poc-e2f-numeric-patch-similarity-2026-10-03.md` | this report (hash reported with the handoff, not embedded) |

Stopped for lead review. The task is not marked complete, and no corpus,
tolerance or promotion claim is made.
