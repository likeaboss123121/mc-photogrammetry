# DS-R4-V2-SPARSE-SIDECAR-PAGE-INDEX + OVERFLOW-GUARD (2026-10-04)

Tasks: `AGENT_COMMS.md:7220-7227` (sparse sidecar page-index repair) and
`AGENT_COMMS.md:7248-7253` (overflow guard). Repairs and focused regressions for the sparse sidecar
page contract in `src/mc_photogrammetry/pipeline/candidate_support_provenance_v2.py`, with tests in
`tests/test_candidate_support_provenance_v2_preflight.py`.

Change history recorded by this report:

- Sparse page-index repair (producer `_pack_sidecar_pages` + validator + sparse tests): implemented
  and independently verified by the lead; summarized in sections 2 and 3.
- Overflow guard: an explicit bound check before the `present[len(entries)]` lookup plus a focused
  malformed extra-page regression; summarized in section 2 (guard) and section 3 (test).

## 1. Final source/test identity (after the overflow guard)

| Artifact | Path | SHA-256 | Bytes |
| --- | --- | --- | --- |
| Source | `src/mc_photogrammetry/pipeline/candidate_support_provenance_v2.py` | `F5CEE3CFE9D0247069637992FB29D1F9A8533ABC9383AF841CFC9A726313CCD2` | 122,069 |
| Tests | `tests/test_candidate_support_provenance_v2_preflight.py` | `F1895E5C40148C49C77247CC13E488F7691CCAEC36003619401DF5101A8DBD96` | 114,420 |

These are the final pins recomputed from the working tree after the overflow-guard edit. The
pre-overflow-guard pins recorded for the page-index task were source
`7317102F3C2F5D90B858D4B87690D0B27C30FF6F6898CD52C127BCA6945A49C4` (121,631 B) and tests
`1FE13924DDA40D7834A7BBE90434BC630E6C28D589C4723550FE9A1238D89940` (113,494 B).

## 2. Producer and validator contract (source-grounded, final line numbers)

`_pack_sidecar_pages` (`:1579-1607`) packs the **present-row subsequence**, whose tuple ordinals are
not admitted-row indices. Its docstring (`:1581-1588`) states the corrected rule: a page's
`first_admitted_row_index` is the admitted index of that page's own first entry
(`entries[start].admitted_row_index`, `:1593`), never the slice ordinal `start`; every entry keeps
its original admitted index, so same-page and cross-page gaps are expected and are not renumbered.
The value flows into `_make_sidecar_page` (`:1570-1576`) and into `_sidecar_page_payload`'s
`first_admitted_row_index` field (`:786-797`), so it participates in `_sidecar_page_content_digest`
(`:1213-1216`) and in the page binding record.

Consumer/validator sites of the field:

- `MatcherSupportSidecarPageV2.__post_init__` (`:811-831`): requires the declared first index to be
  a non-negative `int` equal to `entries[0].admitted_row_index` (`:812-820`), then requires strictly
  increasing entry indices (no duplicate/reorder, `:821-825`), then replays the page digest
  (`:827-831`) - the line-820 `ValueError` observed in R3.
- `_sidecar_page_size` / `_make_sidecar_page` (`:1552-1576`): size and digest are computed with the
  page's actual first entry index.
- `validate_complete_support_capture` (`:1277` onward): replays page binding, page digest and
  `entry_count` (`:1371-1380`); requires sidecar pages to remain in increasing admitted order
  (`:1387-1389`); applies the overflow bound check
  (`:1390-1396`, `raise ValueError("complete support sidecar page has no present row to bind")`)
  before the lookup; requires each page's declared first index to equal
  `present[len(entries)].admitted_row_index`, i.e. the next present admitted row the capture must
  cover (`:1397-1400`); then requires the flattened entries to be an exact one-to-one ordered cover
  of the present index rows including payload digest (`:1402-1411`). The inline comment at
  `:1381-1386` records that sparse present rows make sidecar pages **non-consecutive** in
  admitted-row space, so the check binds to present-row positions rather than requiring consecutive
  admitted indices.

### 2.1 Overflow guard

A well-formed sidecar has at most one entry per present row. Before the overflow guard, once the
present rows were fully covered an additional self-consistent sidecar page still passed the
admitted-order check (its first index is greater than the last flattened entry index) and reached
`present[len(entries)]`, which raised `IndexError` instead of the validator's usual `ValueError`.
The guard at `:1390-1396` checks `len(entries) >= len(present)` before indexing, so a malformed
extra page fails closed as `ValueError`. Sparse exact binding, page ordering, digests, caps and
empty-sidecar handling are unchanged:

- Ordering is still enforced at `:1387-1389` (a reordered/duplicate page is refused there).
- The exact present-row/payload cover is still enforced at `:1402-1411`.
- Index pages remain consecutive in admitted order (`:1358-1366`), and digest pages remain
  consecutive ordered slots (`:1419-1429`).
- Empty-sidecar complete captures still take the all-row-local-refusal path (`present` empty means
  any sidecar page is refused by the same bound check).

## 3. Tests (source-grounded)

The focused file contains **66** test functions (65 before the overflow guard plus the new
regression). Sparse-specific coverage:

- `test_sparse_present_rows_pack_and_replay_in_and_across_pages` (`:447-473`): admitted present
  indices `0, 3, 8` with a two-entry page cap; asserts two pages, flattened entries `[0, 3, 8]`,
  each `page.first_admitted_row_index == page.entries[0].admitted_row_index`, page first indices
  `[0, 8]`, and an exact one-to-one present-row/payload binding.
- `test_sparse_single_page_gap_and_adjacent_present_rows_pack_and_replay` (`:476-486`): present sets
  `(2, 5)`, `(1, 2, 3)`, `(0, 1, 3, 9)` - same-page gaps and a first page not starting at index 0.
- `test_sparse_sidecar_reorder_and_misbinding_refuse_at_replay` (`:489-516`): hand-built,
  self-consistent pages (digest recomputed via `_synthetic_sidecar_page`, `:382-394`) that only
  capture-level replay can reject: cross-page reorder and a page misbound to a later present row.
- `test_sparse_sidecar_duplicate_and_misbound_first_index_cannot_be_built` (`:519-540`): constructor
  rejects a wrong declared first index, reversed entries and duplicate entries; a correctly bound
  three-page sparse decomposition then validates.
- `test_sparse_sidecar_overflow_page_refuses_with_value_error` (`:543-558`): present rows `3, 5, 7`
  are fully covered by page 0, then a second self-consistent page starts at admitted index `9`
  (strictly greater than the last flattened entry index `7`, so the ordering guard does not fire).
  `validate_complete_support_capture` must raise `ValueError` from the bound check; without the
  guard this shape reaches `present[3]` on a three-element present list and raises `IndexError`.
  The capture is built with the existing `_sparse_capture_with_pages` seam
  (`:397-444`), which bypasses `CompleteSupportCaptureV2.__post_init__` and recomputes the page
  digest, so only capture-level replay can refuse it.

Supporting helpers: `_sparse_universe` (`:330-368`) builds a dense admitted universe where only the
requested rows are `SUPPORT_STATE_PRESENT`, and `_sparse_capture_with_pages` (`:397-444`) bypasses
`CompleteSupportCaptureV2.__post_init__` so only `validate_complete_support_capture` can refuse,
with real index/digest/root commitments. `_synthetic_sidecar_page` (`:382-394`) is the tamper seam
that rebuilds a page digest. Existing non-sparse regressions (page-binding tamper, 1,928-row round
trip, atomic over-cap refusal, R3 collector localization) are retained in the same file.

## 4. Checks performed (this task)

- `py_compile` on the two edited files: exit `0`.
- `tests/test_candidate_support_provenance_v2_preflight.py`: `66 passed in 7.81s`, exit `0`.

No other file, test, runner, image, capture, evaluator, truth, reference, litematic, Minecraft or
launcher command was executed.

## 5. Scope and limits

- No screenshot, visual, runner, inference, capture, evaluator/truth/reference/litematic,
  Minecraft/launcher, tuning, accuracy scoring or full-suite execution was performed or is claimed.
- No accuracy, reconstruction-quality or acceptance claim is made.
- This report does not assert a verified real-image root cause for the R3 `__post_init__:820`
  refusal; the sparse ordinal-vs-admitted-index mismatch remains the source-supported residual
  candidate, and the guard at `:812-820` is disjunctive, so the trace alone cannot discriminate
  which OR-term fired.
- Stopped for Luna review.
