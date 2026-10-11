# DS-R4-BYTE-CAP-AUDIT-2026-10-04

Task ID: `DS-R4-BYTE-CAP-AUDIT`. Owner: DeepSeek (read-only audit, no source/test
edit, no reconstruction). Date: 2026-10-04.

Status: **PART B — the complete pre-filter admitted universe for the frozen
Template-2 fixture is 1,928 candidates, not 448. The complete candidate-support
record therefore refuses at the first cap it can hit, the row cap, before any byte
is measured (`budget_exhausted` / `row_count`). The 448-row same-axis table is a
proper subset; a complete record cannot fit the 1,024-row cap. Whether the byte caps
would bind after raising the row cap is **unknown**: the sealed artifacts carry no
matcher support arrays and do not prove per-candidate support presence, so no byte-fit
claim is made in either direction. The one retained byte fact is conditional —
7,139 x 4 x 8 = 228,448 raw bytes for a single `logical_pairs` array exceeds the
131,072-byte per-payload cap **only if that candidate actually carries support**. A
default-off complete-record design preserving all 1,928 dispositions (of which 448
are the published same-axis rows) is offered below as an **unvalidated** option set
for lead decision. No fit is claimed and no source, test, schema or cap was changed.**

Revision 2026-10-04b: applies the Lead 19:41 UTC correction. The proven 1,928-vs-1,024
row-cap refusal is unchanged. Complete-record byte fit is now stated as **unknown**;
the 7,139 x 4 x 8 = 228,448 conditional per-payload fact is clarified; the
per-row-minimum x all-rows byte bound and the previously asserted unverified compact-index
per-row sizing are removed; and the index/sidecar and paging designs are labelled
unvalidated.

Scope: source inspection plus the already-sealed `.tmp-geom-prefilter-capture-r10-20261004/
observation.json` and the prior R9/R11 findings. No evaluator/truth/camera/mask/image
payload was read; no image, matcher, solver or reconstruction was rerun; no
PowerShell/bash/jobs/subagent/workflow was used and no file other than this report was
created.

## 1. Exact default budgets and cap semantics (source)

- `src/mc_photogrammetry/pipeline/candidate_support_provenance.py:106-108` —
  `CANDIDATE_SUPPORT_MAX_ROWS = 1_024`, `CANDIDATE_SUPPORT_MAX_RECORD_BYTES = 262_144`,
  `CANDIDATE_SUPPORT_MAX_SUPPORT_BYTES = 131_072`; `:109` tolerance `0.15`.
- Collection order and cap points, `failed_candidate_support_provenance` →
  `_collect_candidate_support` (`:552-698`):
  - `:609-610` complete pre-filter admitted universe `len(observations.candidates) >
    max_rows` → `budget_exhausted` / `row_count`, checked **before** any row is built;
  - `:646-647` **per row**, `support.payload_bytes() > max_support_bytes` →
    `support_payload_over_cap` / `support_bytes`;
  - `:693-697` **once, after all rows are built**, the canonical encoded payload
    (`json.dumps(..., sort_keys=True, separators=(",",":"), ensure_ascii=True,
    allow_nan=False).encode("utf-8")`) `> max_record_bytes` → `budget_exhausted` /
    `record_bytes`.
- Every failure path returns `_refused(...)` (`:462-466`, `:582-586`): a refusal carries
  only schema/status/reason/field plus its digest, `row_count = 0`, no rows and no ingress
  (`:475-480`). There is **no partial/sampled record path** in the current builder.
- The support payload is not a hash-only stub: `MatcherSupportPayloadV1` (`:236-310`) owns
  three `int64` arrays (`logical_pairs`, `warp_source_rows`, `warp_source_columns`) and
  `_payload` (`:284-299`) serializes each one including `"values": value.tolist()`;
  `to_record()` defaults to `include_arrays=True` (`:301-306`). `payload_bytes()`
  (`:308-310`) is the raw `nbytes` sum of the three arrays.
- Therefore a support payload under the 131,072-byte raw cap can encode to far more than
  131,072 bytes of canonical JSON, and the record cap is checked on that JSON. The shipped
  record carries the full array values inline; there is no array-omitting record mode on the
  collector path.

## 2. What the 448-row same-axis table is — and what the collector actually gets

The frozen R10 all-axis prefilter capture
`.tmp-geom-prefilter-capture-r10-20261004/observation.json` (file SHA-256
`BC957B684152AEEBB5C055EB12816643F91A506AB01543761D6461FEE4C624B4`, record
`record_sha256 c23c48ee6af01c11b0d9cd576c504e88c1bf371f762b09a26500d7e81bbe7c62`,
schema `mc-photogrammetry-same-axis-candidate-capture-v2`, status `complete`,
`row_count 448`, `group_count 9`, axes `[0,1,2]`) is a *bounded subset* of the matcher's
admitted universe: `_collect_same_axis_candidates` selects only candidates whose group key
is solver-active (`image_derived_graph.py:4261-4273`), i.e. 9 active groups out of the
matcher's full group set.

The same artifact's own matcher trace carries the complete-universe counters
(`observation.json:13713-13721`): `pre_admission 9592`, **`post_admission 1928`**,
`post_same_axis 38`. `post_admission_candidates` increments once for every admitted
`PlaneMatch` (`plane_graph.py:1247-1249`), i.e. the full **pre-filter** admitted universe
that `SameAxisCandidateObservation.candidates` holds and that
`failed_candidate_support_provenance` consumes (`candidate_support_provenance.py:602-608,
622-629`). `post_same_axis 38` is `len(filtered)` (`plane_graph.py:1276-1277`), matching
the retained `matches`/`input_match_count 38` of the sibling solver records.

So for this fixture:

| Universe | Count | Provenance |
|---|---|---|
| pre-admission quadrant attempts | 9,592 | trace `pre_admission` |
| **pre-filter admitted candidates = candidate-support rows** | **1,928** | trace `post_admission`; `PlaneGraphTrace.post_admission_candidates` |
| published same-axis rows (9 solver-active groups) | 448 | R10 v2 record `row_count` |
| post-filter solver input matches | 38 | trace `post_same_axis`; `input_match_count 38` |

The 448 rows are therefore not "the complete 448-row universe"; they are the subset of the
1,928 admitted candidates that fall inside the 9 solver-active same-axis groups. The R10
artifact serializes only these 448 rows; the other 1,480 candidates are not present
row-by-row in that artifact. At source level, the same-axis serializer's 19-element row is
defined as
`[group_index, match_index, row_index, left_ref, left_alt, right_ref, right_alt,
relative_quadrant, score, psr_value, psr_infinite, peak, second_peak, overlap_cells,
shift_row, shift_col, delta, disposition, reason]`
(`src/mc_photogrammetry/pipeline/image_derived_graph.py:3547-3598`), supplying the scalar
part of candidate-support rows for serialized active-group candidates: `left_ref`/`right_ref` are the frame
ids, `left_alt`/`right_alt` the endpoint alternative indices, `relative_quadrant`,
`score`/`psr`/`peak`/`second_peak`, `overlap_cells`, `shift_row`/`shift_col`, `delta`,
`disposition` and `match_index`. `_indexed_match_payload`
(`src/mc_photogrammetry/reconstruction/plane_graph.py:640-649`) confirms the same scalar
projection used to build the row identity. The fields *not* in the sealed table are
exactly the digest strings and the support payload, which exist only in-process from the
matcher witness rows and the signature bindings.

Key scalar facts read from the sealed 448-row table (not from any image or evaluator
payload): `overlap_cells` minimum is **9** (also the trace's `min_overlap_cells` gate);
the largest value in the candidate table is **7,139**. Arithmetic: **if** that candidate
carries a `logical_pairs` support array of 7,139 rows, that array alone is
7,139 x 4 int64 x 8 bytes = **228,448 raw bytes**, which is greater than the
131,072-byte per-payload cap, so such a support-bearing row would refuse with
`support_payload_over_cap` / `support_bytes`. This is **conditional**: the sealed
artifact contains no matcher support arrays and does not prove that this candidate, or
any candidate, carries support at all. It is therefore evidence about what a
support-bearing row of that size *would* do — not evidence that the per-payload cap
does or does not bind for this fixture, and not evidence about the record cap. Actual
in-process support presence and per-row support sizes remain unknown (section 5), so no
byte-fit conclusion follows from it.

Measured calibration point (already accepted, AGENT_COMMS 2026-10-04): the lean
*same-axis* v2 record is **79,668 canonical bytes** (`record_sha256 c23c48ee…`), a 19-element
row schema with no per-row digests, no support arrays and a shared reading/group table
(~178 bytes/row). The candidate-support row schema adds six 64-hex digests, 32 named keys
and, when a payload is present, inline support arrays, so a support-bearing row is
structurally larger than a same-axis row. Because per-candidate support presence is
unproven, this calibration does **not** translate into a complete-record byte total.

## 3. First cap and refusal (state the exact one)

Collection checks the complete-universe row count first (`:609-610`), so for this fixture
the deterministic refusal is:

```
status = "refused"
reason = "budget_exhausted"
field  = "row_count"
row_count = 0            # atomic: no rows, no ingress
```

because `len(observations.candidates) = 1,928 > CANDIDATE_SUPPORT_MAX_ROWS = 1,024`. The
byte caps are never reached in the shipped order, and the 448 published rows are fewer than
the cap while the complete 1,928 are not.

| Cap | Default | Binds for the complete 1,928-row capture? | Evidence |
|---|---|---|---|
| row cap | 1,024 | **Yes — first and only reached cap in shipped order** (1,928 > 1,024) | trace `post_admission 1928`; `:609-610` |
| support cap | 131,072 **per payload** | Unknown; would bind only for a candidate that carries a support payload whose raw arrays exceed 131,072 bytes. The 7,139-pair maximum *would* exceed it **if** that candidate carries support. | `:646-647`; R10 max `overlap_cells` 7,139; per-candidate support presence is absent from the sealed artifact |
| record cap | 262,144 canonical bytes | Unknown; not reached in shipped order and not provable either way without the matcher arrays/presence evidence | `:693-697`; section 4 |

Interpretation for the requested "complete 448-row record": there is no 448-row *complete*
record to fit. The complete pre-filter universe is 1,928 rows; the 448-row table is the
solver-active subset. The accepted atomic-refusal discipline is already correct (a refusal
carries zero rows and never labels a partial set complete), and the current behavior on an
authorized capture would be `budget_exhausted`/`row_count`.

## 4. Byte fit is unknown — no valid complete-universe bound exists on sealed evidence

The earlier revision of this report claimed a conservative lower bound by multiplying a
support-bearing-row minimum by all 1,928 candidates. That multiplication is **not sound**:
it assumes every admitted candidate (a) carries a support payload, (b) carries all three
int64 arrays, and (c) carries all six 64-hex digest fields. The sealed R10/R11 artifacts
prove none of those. R10 stores only the 19-scalar candidate table (448 active-group rows),
plus readings/groups/identities and sibling solver records; its trace stores counters and
pair rows only. The matcher support arrays and endpoint signature digests exist only
in-process (`MatcherSupportPayloadV1`, `PreFilterMatcherWitnessRowV1.payload`,
`signature_sha256_by_identity`). No sealed artifact exposes per-candidate payload presence
or per-row array lengths beyond the scalar candidate values. Therefore the complete-record
byte fit after raising the row cap is **unknown**:

- The shipped record encodes inline array values via `tolist()` and is checked once, after
  all rows are built, on the canonical JSON (`:693-697`). The 262,144-byte record cap
  *could* bind if enough candidates carry large arrays, but this audit cannot say whether it
  does or at which row, because per-candidate support presence and array lengths are not
  sealed.
- The 131,072-byte cap is per payload, not an aggregate. It *would* refuse an individual
  7,139-pair row **if** that candidate carries support (section 2), but whether any given
  candidate carries support is unproven. It is not evidence that the cap binds, or does
  not bind, for the fixture as a whole.
- The one measured calibration — 79,668 canonical bytes for the lean, no-support,
  no-digest 448-row same-axis record — bounds only a non-support record. It says a support
  row is structurally larger (extra digests, keys and, when present, arrays), which is a
  qualitative expectation, not a proven size, and it does not bound the complete record.

Establishing any byte fit requires either the in-process matcher witness arrays /
payload-presence evidence or a bounded synthetic worst-case capture under an authorized
task. Neither was available, permitted, or performed here, so no byte-fit claim is made in
either direction.

## 5. What cannot be established from sealed artifacts

The candidate-support record's support payloads are built from the in-process
`PreFilterMatcherWitnessRowV1.payload` (`plane_graph.py:479-616,1256-1335`) and the
`signature_sha256_by_identity` bindings. Neither the matcher pair witness arrays
(`logical_pairs`, `warp_source_rows`, `warp_source_columns`) nor the endpoint signature
digests are serialized in the R10/R11 observations: R10 stores the 19-scalar candidate
table (448 active-group rows) plus readings/groups/identities and the sibling solver
records, and its trace stores counters and pair rows only. Luna's review (AGENT_COMMS
2026-10-04) states the 256 KiB record and 128 KiB support-byte caps "remain unproven for
this capture"; this audit adds that the first reached cap is the row cap and therefore that
the byte fit is **unknown**, not refuted. The exact per-candidate support presence and
per-row support sizes cannot be reconstructed offline, and no exact first row at which the
record cap would be exceeded can be named for that reason (it is moot in the shipped order,
since the row cap refuses first).

## 6. Default-off complete-record design options (unvalidated; lead decision required; no edit made)

All options below are **unvalidated design sketches**, not size-verified proposals: none is
claimed to fit any cap. They keep the collector default-off, keep every production
decision/digest unchanged, keep the per-payload support cap at 131,072 bytes for any payload
that is still carried, and refuse atomically on any cap or count/digest mismatch. Because the
complete universe is 1,928 rows (not 448), any "complete record" design must either raise the
row cap or split the universe; the 1,024-row cap cannot hold it.

**Option A — disposition index + content-bound support sidecar (unvalidated; recommended for
further study).** Add one versioned record with two parts, each independently capped:
1. *Index record*: one row per admitted candidate with the complete disposition set —
   `admitted_row_index`, `group_index`, `left`/`right`, `left_axis`/`right_axis`,
   `quadrant_of_right`, `left_plane`/`right_plane`, `left/right_alternative_index`,
   `left/right_b`, `disposition`, `match_index`, `overlap_cells`, `cell_shift_rc`, and the
   typed refusal reason when support is absent. The intent is to keep the index compact by
   omitting support arrays and per-row digest strings, but **no per-row byte size is claimed
   and whether a 1,928-row index fits 262,144 canonical bytes is unverified**. That must be
   established by measurement or a synthetic worst-case test before the option is accepted.
   Bind `row_count` and the ordered universe to the frozen input-identity digest, exactly as
   Luna proposed. The row cap must be raised (e.g. to a declared universe cap) or the index
   paginated so 1,928 compact rows are admissible.
2. *Support sidecar* (or paged companion record): the full `MatcherSupportPayloadV1` per
   admitted index, bound to the index row by `payload_sha256` plus the three array
   `sha256` values, with the existing per-payload 131,072-byte cap and its own aggregate
   cap/refusal. Persist it separately so the index record's 262,144-byte cap is never
   consumed by inline values.
Every index row's `support_refusal_reason`/`payload_sha256` preserves the complete
disposition set even when a sidecar page refuses; no row is dropped or sampled. The index
and sidecar caps and any aggregate sidecar cap are design choices that require their own
justification; none is claimed to fit here.

**Option B — canonical record pages with a bound root index (unvalidated).** Keep the current
row shape but split the sorted 1,928-row universe into page records, each independently
checked against 262,144 canonical bytes, plus a root record binding page order, per-page
counts/digests and the full input-identity digest. All 1,928 dispositions survive; a
refusing page refuses the whole capture atomically. This preserves the shipped row schema
(only the page envelope is new) but needs a new multi-page producer/consumer contract.
Whether a support-bearing page can fit 262,144 bytes is untested and unknown.

**Option C — raising the record cap alone (not recommended).** Under inline-array
semantics the required ceiling for a support-bearing complete record is unknown and could be
large; the 131,072-byte per-payload cap would remain a separate, non-aggregate limit. It also
contradicts the accepted atomic-refusal discipline for R4/R7/R9/R10.

Option A is the design that appears to preserve every admitted disposition within a bounded
record shape, and it matches the R4 ingress guidance never to retain unbounded buffers. It
remains **unvalidated**: no option here is claimed to fit until a synthetic or measured
preflight establishes the index and sidecar sizes. Recommend a follow-up bounded
implementation task for Option A with synthetic tests for: complete 1,928-row index
admission, an explicit declared row/universe cap or pagination, exact count/identity-digest
binding to the frozen input identity, per-payload sidecar cap refusal, aggregate sidecar
refusal, index-record byte refusal, and zero partial rows on every refusal. No new capture
should be authorized until that design is accepted and the preflight can state the expected
index/sidecar sizes.

## 7. Provenance and limits

- Source: `candidate_support_provenance.py` (863 lines, SHA-256
  `A8BD3102652720513ED85E2D3B7E2C81A79A01A7621C41238B09F6E741604562`, as recorded in
  AGENT_COMMS for the accepted DS-R4-TAP-SENTINEL state), `plane_graph.py`
  (`PreFilterMatcherWitnessRowV1`, `PlaneGraphTrace.candidate_counts`,
  `MatcherPairWitnessV1`, `GraphMatcherWitnessPayloadV1`), `image_derived_graph.py`
  (`_same_axis_candidates_payload`, `_validate_same_axis_candidate_row`,
  `_input_match_identity`, `_collect_same_axis_candidates`, candidate-support sink call).
- Sealed artifact: `.tmp-geom-prefilter-capture-r10-20261004/observation.json`, file
  SHA-256 `BC957B684152AEEBB5C055EB12816643F91A506AB01543761D6461FEE4C624B4`, which the
  R11 pair-scan report independently pinned (section 3 of
  `docs/reports/DS-IMG-R11-AXES02-PAIR-SCAN-2026-10-04.md`). R11 solver-context artifact
  file SHA-256 `924D388D5ED41287EB88A6B0B709B8228C2EAD8422F9957946EFC6A734F75A92`; no R11
  payload beyond its pin was needed.
- Section 4 does not assert a complete-universe byte bound: the sealed artifacts lack
  per-candidate support presence and the matcher arrays, so the byte fit is **unknown**. The
  only byte arithmetic retained is the conditional single-array fact in section 2. No
  reconstruction, image read, truth/evaluator/camera/mask access, sampling, truncation,
  code/test/schema/cap edit, or fit claim was made.
- Stopped for lead review. No R1 rerun is authorized or performed.
