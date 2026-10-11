# DS-R4-COMPLETE-SUPPORT-DESIGN-2026-10-04

Task ID: `DS-R4-COMPLETE-SUPPORT-CONTRACT-DESIGN`. Owner: DeepSeek (read-only design,
no source/test/schema edit, no reconstruction, no capture). Date: 2026-10-04.
Supersedes nothing: `DS-R4-BYTE-CAP-AUDIT-2026-10-04.md` remains the accepted audit and
this report is its design follow-up.

Status: **DESIGN ONLY — for Luna/lead review. Nothing here is implemented, measured or
authorized.** It is a bounded, default-off replacement design that preserves all **1,928**
pre-filter admitted candidate dispositions with no sampling, no truncation, no inference or
output change, and no partial-complete labeling. No source, test or schema file was edited;
no reconstruction, evaluator/truth/camera/mask/image payload read, or capture was performed.
Per-row byte sizes and any complete-universe byte fit remain **unknown** exactly as the
accepted audit states; this design names the preflight that must measure them before any
capture is considered.

**Revision note (2026-10-04, read-only patch per Lead hold 19:53 UTC).** This amendment
resolves the two contract conflicts the Lead raised and keeps every size claim unverified until
the synthetic preflight. (1) The six ordered 1,928-element digest sequences are moved out of
`CandidateSupportCaptureRootV2` into bounded, content-hashed `CandidateSupportDigestPageV2`
records; the root now binds only page counts, page order, page/section digests and the
whole-universe identity digest. §4.3 states the structural reason the root could never hold
them and §6.2 makes the root/digest/index/sidecar/aggregate bounds a preflight obligation.
(2) Row-local `support_state=refused` — an assembled capture outcome that retains its index row
and disposition — is now explicitly separated from capture-level atomic refusal
(`status=refused`, `row_count=0`); §6.3 and the new §6.4 classify every cap failure as
row-local or capture-global, and §5, §8 are made consistent with that split. No source, test or
schema file was edited; no reconstruction, evaluator/truth/camera/mask/image payload read, or
capture was performed.

Scope read (only): `src/mc_photogrammetry/pipeline/candidate_support_provenance.py`,
`src/mc_photogrammetry/reconstruction/plane_graph.py` (candidate/witness types and
`match_plane_graph`), `src/mc_photogrammetry/pipeline/image_derived_graph.py` (consumer and
sibling capture contracts), `docs/reports/DS-R4-BYTE-CAP-AUDIT-2026-10-04.md`,
`docs/deepseek-worker.md`. No other path was read or modified.

## 1. Problem statement (source-grounded)

The shipped collector `failed_candidate_support_provenance` →
`_collect_candidate_support` (`candidate_support_provenance.py:552-698`) refuses the complete
frozen universe atomically at the **first** cap it can reach:

- `:609-610` `len(observations.candidates) > max_rows` with `CANDIDATE_SUPPORT_MAX_ROWS =
  1024` refuses `budget_exhausted` / `row_count` **before any row is built**. The accepted
  audit proves the frozen Template-2 universe is **1,928** (`post_admission 1928`), so the
  shipped record can never hold it.
- Support presence and per-row array lengths are not sealed, so the 262,144-byte canonical
  record cap and the 131,072-byte per-payload support cap are **unproven for this fixture**
  (audit §4-5). This report does not reverse that: it treats both as unknown and requires a
  preflight to measure them.
- Every failure path already returns `_refused` (`:462-466`, `:582-586`): schema/status/
  reason/field plus digest, `row_count = 0`, no rows, no ingress. There is no partial or
  sampled path today, and there must not be one in the replacement.

The replacement must therefore (a) admit the whole 1,928-row universe as one logical,
atomic capture; (b) keep every disposition — kept or dropped — and every per-row typed
support refusal; (c) keep the encoder, caps and failure semantics bounded and fail-closed;
and (d) stay default-off and output-neutral.

### 1.1 Frozen universe sources and canonical order (must be reused, not re-derived)

| Quantity | Source | Count |
|---|---|---|
| `observations.candidates` + parallel `observations.dispositions` | `plane_graph.SameAxisCandidateObservation` (`plane_graph.py:299-331`), collected at `:1264-1275` over the sorted `matches` list | 1,928 |
| `matcher_witness_rows[position]` | `PreFilterMatcherWitnessRowV1` (`plane_graph.py:479-545`), published at `:1323-1335` with `admitted_row_index=enumerate(matches)` | 1,928 |
| `signature_witnesses` | `SourceTapWitnessBindingV1` captured by the ordinary signature call | per signature |
| `ingress` | `FrameIngressProvenanceV1` tuple | per frame |
| canonical sort key | `_match_order` (`plane_graph.py:1339-1342`): `(left, right, left_axis, right_axis, left_plane, right_plane, left_b, right_b, quadrant_of_right, -score)` | — |

`admitted_row_index == position` is the order binding, and the parallel `dispositions` tuple
must agree element-wise with `matcher_witness_rows[position].disposition` (the collector
already enforces this at `:627-629`). The replacement must not re-match, re-signature,
re-solve, re-sort, or open a frame.

### 1.2 Normal-mode digest caveat

The frozen 1,928 is a **generic normal-mode** universe. When `plane_graph` runs with a
manifest/camera replacement solver (the `_patch_camera_solver` test path), `SameAxisCandidateObservation`
can be empty for the same code path. Any admission guard must therefore refuse a genuinely
empty/absent observation (`candidates_unavailable`) rather than assert 1,928 rows for every
mode; 1,928 is the frozen-fixture expectation, not a global constant.

## 2. Option comparison

Two families preserve all 1,928 dispositions. Both keep the shipped v1 record and constants
untouched (its 1,024-row cap stays, frozen), and both are default-off.

### Option A — compact disposition index + content-bound support sidecar (recommended)

- **Index**: one row per admitted candidate carrying the complete disposition, compact
  scalars and the typed support outcome code — no inline arrays and no per-row digest strings.
- **Sidecar**: the full `MatcherSupportPayloadV1` material per support-bearing admitted index,
  bound to its index row by content digests; physically separate pages so the index budget is
  never consumed by array values.
- **Root**: one small record that binds row count, order, identity digests, index/sidecar/digest
  page order and page digests, and declares every cap. The six ordered 1,928-element digest
  sequences do **not** live in the root; they are paginated into bounded, content-hashed
  `CandidateSupportDigestPageV2` records (§4.3), so the root stays structurally small.
- Strengths: complete disposition set on the hot path; support arrays isolated and
  independently capped/refusable; per-row refusal never drops a disposition; index pages can
  pack many compact rows; the shipped row semantics are preserved field-for-field; no single
  section of the contract carries an unbounded digest sequence.
- Costs: a new multi-artifact contract (root + index pages + digest pages + sidecar pages) and a
  new aggregate sidecar ceiling that must be declared and measured; the digest sections gain
  their own page cap and page count that the root must bind and the preflight must bound.
- Risk (must be preflighted): "compact" is a structural claim only. Per-row bytes, the 1,928-row
  index page size, the digests-per-digest-page count, the derived digest page count and the
  root's page-binding size are **unknown** until measured; a declared universe cap must be
  proven to hold 1,928 before acceptance (audit §6 requirement).

### Option B — canonical record pages with a bound root index

- Keep the current full row shape (all six 64-hex digests plus inline support arrays) and split
  the sorted 1,928-row universe into page records, each independently checked against
  262,144 canonical bytes, plus a root binding page order, per-page counts/digests and the
  input-identity digest.
- Strengths: preserves the shipped row schema almost verbatim; only a page envelope and root
  are new.
- Costs: support-bearing pages are structurally larger, so page packing is dominated by the
  inline arrays and the number of pages is unknown; a single oversized row (audit's
  conditional 7,139×4×8 = 228,448 raw bytes) can force a very small page or, under the
  per-payload cap, a row-level refusal inside a "page of a complete record". Page boundaries
  also make the order binding subtler (a dropped page must refuse the whole capture).
- Verdict: workable but strictly worse on budget isolation and on keeping the disposition
  index small; it duplicates the uncertainty the replaced design already has. It is simpler in
  one respect only — it keeps the six digests inline in rows, so it needs no digest pages — and
  that is not enough to offset the inline-array budget problem it inherits.

**Recommendation: Option A.** It is the only family that isolates the unknown-sized arrays
from the complete disposition set, so a support-size problem cannot threaten the record of
all 1,928 dispositions. The selection is a design recommendation, not a fit claim.

## 3. Recommended design (Option A)

Six additive, versioned, evaluation-only record kinds plus a declared cap table. Producer
runs only when an explicit evaluation sink is attached; the default production path allocates
and serializes nothing and the returned graph result and digests stay byte-for-byte unchanged.

### 3.1 Records

**`CandidateSupportIndexRowV2`** (one per admitted candidate; **exactly 1,928 rows**). Carries
the shipped compact scalar set: `admitted_row_index`, `group_index`, `left`, `right`,
`left_axis`, `right_axis`, `quadrant_of_right`, `left_plane`, `right_plane`,
`left_alternative_index`, `right_alternative_index`, `left_b`, `right_b`, `delta`, `tolerance`,
`disposition`, `match_index`, `overlap_cells`, `cell_shift_rc`, plus
`support_state ∈ {present, refused}` and, when refused, `support_refusal_reason` from the fixed
per-row code set of §5. Per-row digest strings (`match_sha256`, `envelope_sha256`,
`matcher_witness_sha256`, `left/right_signature_sha256`) are **not** duplicated here; their
one-to-one ordered sequences are bound by the bounded digest pages of §4.3 and by the root's
page bindings, not by 1,928 rows of six keys. The row keeps the
`left-source-right-target` orientation.

**`CandidateSupportIndexPageV2`** (bounded rows). `schema`, `page_index`, `row_count`,
`first_admitted_row_index`, `page_sha256`, rows. Pages are consecutive slices of the single
canonical admitted order; `page_index` is the root-declared order; canonical encoding of each
page is checked against `INDEX_PAGE_MAX_BYTES`.

**`MatcherSupportSidecarEntryV2`** (one per support-bearing admitted index). `admitted_row_index`,
`payload_sha256` (canonical payload digest), `logical_pairs_sha256`,
`warp_source_rows_sha256`, `warp_source_columns_sha256`, `source_shape`, `target_shape`,
`overlap_cells`, plus the array values under the shipped `MatcherSupportPayloadV1` semantics.
Refused rows have **no** sidecar entry; their index row already carries the typed refusal.

**`MatcherSupportSidecarPageV2`** (bounded entries). `schema`, `page_index`, `entry_count`,
`first_admitted_row_index`, `page_sha256`, entries. Entries preserve the canonical admitted
order, and a single entry is atomic (never split across pages).

**`CandidateSupportDigestPageV2`** (bounded digest run, one section per page run). `schema`,
`section` (one of the six digest kinds of §4.3), `page_index`, `first_admitted_row_index`,
`digest_count`, `digests` (ordered 64-hex strings, or an explicit `null` at a position whose
digest material is unavailable), `page_sha256`. Pages are consecutive slices of one section in
admitted order. A section's `digest_count` values across its pages sum to `row_count` exactly,
and `null` positions are never collapsed, so a page boundary cannot shift the global order
binding; a page never mixes sections. Canonical encoding of each page is checked against
`DIGEST_PAGE_MAX_BYTES`.

**`CandidateSupportCaptureRootV2`** (one small record). `schema`, `status`, `basis`,
`failed_assignment_sha256`, `whole_universe_identity_sha256` (§4.3),
`input_match_identity_sha256`, `row_count`, `declared_universe_cap`, `index_page_count`,
`sidecar_page_count`, `digest_page_count`, ordered `index_pages`
(`page_index`, `sha256`, `row_count`), ordered `sidecar_pages` (`page_index`, `sha256`,
`entry_count`) and ordered `digest_sections` (`section`, `page_count`, `section_sha256`, pages
each `page_index`, `first_admitted_row_index`, `digest_count`, `sha256`), `ingress`,
`record_sha256`. The root never carries per-row arrays and never carries a digest *sequence* —
only counts, order, page/section digests and the whole-universe identity digest. Its size is
structural (six section bindings plus a bounded page list), but `digest_page_count` and the page
lists are not yet measured, so the root bound is a §6.2 preflight obligation rather than an
assumption.

### 3.2 Why the index is compact and what is deliberately omitted

Omitted from index pages: inline support arrays (moved to sidecar); the six per-row 64-hex
digest strings (moved to bounded digest pages, §4.3); derived values that are exactly
replayable from the ordered identity tuple and the frozen readings. The shipped v1 row and
record remain untouched for default and legacy callers.

## 4. Identity, count and order binding

### 4.1 Count binding (equality, not inequality)

The root records the exact `row_count`. Each index page's `row_count` sums to it; each sidecar
page's `entry_count` sums to the number of `present` index rows; each digest section's summed
`digest_count` equals it across all six sections; and `row_count ≤ declared_universe_cap`. The
collector must assert, before building any page:

- `len(observations.candidates) == len(observations.dispositions)`
- `len(matcher_witness_rows) == len(observations.candidates)`
- the sum of index-page `row_count` values equals the root `row_count`
- the sum of each digest section's `digest_count` values equals the root `row_count`
- `row_count == 1_928` for the frozen fixture, compared against a declared expected count
  supplied by the evaluation task; a count mismatch refuses atomically with
  `universe_count_mismatch`, and `row_count > declared_universe_cap` refuses with
  `row_cap_exceeded`.

No cap may silently truncate: the row cap is a *declared admission ceiling*, and exceeding it
refuses, it never samples.

### 4.2 Order binding

- Index pages are consecutive slices of the canonical admitted order; page N's first row index
  is exactly page N-1's last row index + 1.
- Within a page, `row.admitted_row_index` equals the page-local position plus the page's
  `first_admitted_row_index`, and the global sequence is strictly increasing by one.
- Sidecar entries are a strictly increasing subsequence of the admitted indices and follow the
  same global order across pages, so `present`/`refused` never reorders a row.
- Digest pages are consecutive slices of one section: the first page starts at admitted row 0,
  the last page ends at `row_count - 1`, page N's `first_admitted_row_index` is exactly page N-1's
  first index plus its `digest_count`, and a `null` position still occupies its slot, so digest
  paging cannot renumber the admitted order.
- A row may not be duplicated, dropped or reordered: any violation refuses with
  `universe_order_mismatch`, and the atomic semantics of §6 mean nothing is partly written.

### 4.3 Identity digests (reuse existing domains; add new domains explicitly)

- **Whole-universe order digest (new, `complete_support_universe` domain):** ordered digest
  over all 1,928 `(admitted_row_index, identity)` pairs, not only the kept subset. This closes
  the gap that today's `input_match_identity_sha256` covers kept rows only.
- **`input_match_identity_sha256`:** recomputed exactly as the shipped record does — the
  `INPUT_MATCH_IDENTITY_DOMAIN` / `INPUT_MATCH_IDENTITY_SCHEMA` digest over the ordered
  `[left, left_plane, left_axis, right, right_plane, right_axis, quadrant_of_right, left_b,
  right_b]` identities of the **kept** rows in admitted order
  (`candidate_support_provenance.py:534-537`, `:683-684`). Cross-checked against the R7/R9
  shared identity so all three records provably describe the same ordered match sequence.
- **`failed_assignment_sha256`:** carried verbatim, same domain as today.
- **Ordered digest pages (new record kind, never root fields):** six parallel 64-hex sequences
  in admitted order — `match_sha256`, `envelope_sha256`, `left_signature_sha256`,
  `right_signature_sha256`, `matcher_witness_sha256` and the per-row identity digest — are
  paginated into `CandidateSupportDigestPageV2` records under `DIGEST_PAGE_MAX_BYTES`. The root
  binds, per section, the ordered page indices, first admitted indices, digest counts, page
  SHA-256 values and a derived `section_sha256` over the ordered page hashes.
  **Why the root cannot carry them (structural, not a size measurement):** as canonical JSON
  string arrays the six sequences need at least 6 × 1,928 × 66 = **763,488** ASCII bytes for the
  quoted 64-hex values alone (64 hex + 2 quotes each; before commas, keys, brackets, section
  digests or any other root field), against the 262,144-byte `ROOT_MAX_BYTES`. The digest
  sequences therefore cannot fit in the root under any honest reading, so paging is a correction
  of a proven impossibility, not a size optimization. Whether any single digest page, the
  derived page count or the root's page bindings fit their caps remains **unknown** until the
  §6.2 preflight.
- **Digest availability:** a row whose digest material is unavailable records an explicit
  `null` at its position in that section instead of being skipped, so every section holds
  exactly `row_count` positions and a page boundary cannot renumber the admitted order. Digest
  absence is a property of the section value only; it is independent of row-local support
  refusal (§5) and never a capture-level failure (§6.4).
- **Page/root digests and mismatch scope:** every page hash covers its full canonical payload;
  the root `record_sha256` covers the root payload including every index, digest and sidecar
  page hash and the section digests. A mismatch between a sidecar entry and the index row's
  `payload_sha256`, or a witness/payload digest mismatch confined to one row's support material,
  is **row-local** and marks that row `support_state=refused` (§5). A mismatch in a whole-universe
  or kept-subset identity digest, in any page hash, or in the root's page/section binding is
  **capture-global** and refuses atomically with `universe_identity_mismatch` (§6.4).

## 5. Per-row typed support refusal

A support outcome is **per row**, and a refusal never removes, reorders or relabels a row.

- **Row-local scope (contract):** every code in this section is a *support* outcome. It leaves
  the capture `status=complete`, `row_count` unchanged, and every index/digest/sidecar page and
  root binding intact; it never triggers the capture-level atomic refusal of §6.3. This is the
  distinction the Lead hold required: a row may be `support_state=refused` inside an assembled,
  complete capture, while `status=refused`/`row_count=0` is reserved for an integrity or
  capture-global cap failure (§6.4).

- Fixed reason codes (bounded enum, no free text): `matcher_witness_unavailable`,
  `support_payload_unavailable`, `support_payload_over_cap`, `source_tap_witness_unavailable`,
  `source_tap_witness_shape_mismatch`, `witness_digest_mismatch`, `payload_digest_mismatch`.
- Exactly one support outcome per row: `present` (with a sidecar entry) or `refused` (with one
  bounded code). A `present` row may not carry a refusal code; a `refused` row may not carry a
  sidecar entry. This mirrors the shipped exactly-one-outcome invariant
  (`candidate_support_provenance.py:374-375`).
- Disposition independence: `disposition` and `match_index` are computed identically whether
  support is present or refused. A dropped candidate with a refused support still records
  `same_axis_mismatch` and `match_index=None`.
- **Design decision flagged for the lead:** v1 refuses the *whole* record when a support-bearing
  row lacks its signature witness (`:649-656`). The complete capture instead keeps that row's
  disposition and marks its support refused. Rationale: the disposition set is the artifact's
  primary product and refusing the whole capture to avoid one absent witness would defeat the
  1,928-row completeness goal; refusal codes keep the absence explicit rather than silent.
  If the lead prefers v1's whole-capture refusal, that is the only behavioral branch to change.
- The sidecar is sparse **only** in the sense that refused rows have no entry; every one of the
  1,928 rows is present in the index always.

## 6. Byte budgets, preflight and atomic failure

### 6.1 Declared caps

| Cap | Role | Scope | Default |
|---|---|---|---|
| `INDEX_PAGE_MAX_BYTES` | per index page, canonical UTF-8 JSON | capture-global | 262,144 |
| `DIGEST_PAGE_MAX_BYTES` | **new**: per digest page, canonical UTF-8 JSON | capture-global | 262,144 |
| `SIDECAR_PAGE_MAX_BYTES` | per sidecar page, canonical UTF-8 JSON | capture-global | 262,144 |
| `SIDECAR_ENTRY_MAX_RAW_BYTES` | per entry, `payload_bytes()` raw `nbytes`; a cheap pre-gate, never the encoded authority | **row-local** | 131,072 |
| `SIDECAR_ENTRY_MAX_BYTES` | **new**: per entry, canonical encoded UTF-8 bytes; the encoded authority for the row-local `support_payload_over_cap` refusal | **row-local** | task-declared; preflight-set |
| `SIDECAR_AGGREGATE_MAX_BYTES` | **new**: total canonical bytes across all sidecar pages | capture-global | task-declared; preflight must measure and the lead must set it — **not** inherited from the record cap |
| `ROOT_MAX_BYTES` | single root record | capture-global | 262,144 |
| `UNIVERSE_MAX_ROWS` | declared admission ceiling; must be ≥ 1,928 with headroom for the frozen fixture | capture-global | proposed 4,096, to be justified by preflight |

The per-payload raw cap is **not** a proxy for encoded size. The audit's conditional
7,139×4×8 = 228,448 raw bytes shows a legal-looking raw payload can exceed 131,072 raw bytes;
conversely, a payload just *under* the raw cap encodes via `tolist()` to roughly five ASCII
bytes per integer, so `SIDECAR_ENTRY_MAX_BYTES`, `SIDECAR_PAGE_MAX_BYTES` and
`SIDECAR_AGGREGATE_MAX_BYTES` must be checked on the **canonical encoding**, never on
`payload_bytes()`. This report makes no claim that any of these caps fit the complete universe.

**Cap-configuration invariant (proven by the §6.2 preflight, refused before Phase A):** the
declared caps must satisfy
`SIDECAR_ENTRY_MAX_BYTES ≤ SIDECAR_PAGE_MAX_BYTES − measured_page_envelope_max`, where
`measured_page_envelope_max` is the synthetic canonical overhead of a sidecar page frame
(`schema`, `page_index`, `entry_count`, `first_admitted_row_index`, `page_sha256` and brackets).
With that invariant a row-local-legal entry always fits a fresh page, so whether a page overflows
never depends on how many entries share it and cannot be a per-row accident. The
preflight also derives `max_digests_per_digest_page` and checks that
`UNIVERSE_MAX_ROWS` positions are representable across the six sections; if either check fails
it refuses with `cap_configuration_invalid` before any capture is attempted.

### 6.2 Default-off synthetic preflight (before any inference read)

The preflight runs before Phase A reads any evaluator/truth/camera/mask/image payload. It reads
no candidate, witness or payload material: it builds **synthetic worst-case shapes only** at the
declared `UNIVERSE_MAX_ROWS` ceiling and measures their canonical encodings. Its outcome is
`preflight_status ∈ {ok, refused}` with `preflight_refusal_reason ∈ {cap_configuration_invalid,
synthetic_size_over_cap, aggregate_unbounded}`; Phase A may run only when it is `ok`. The
preflight is the proof that root, page and aggregate bounds hold — it does not merely report
sizes.

1. **Cap-configuration check.** Verify the §6.1 invariant and the digest-page derivation. A
   violation refuses with `cap_configuration_invalid` and no capture is attempted.
2. **Synthetic worst-case shapes.** Maximum field widths for every scalar; maximum structurally
   legal array shape for one sidecar entry; all-present 64-hex digest sequences for every digest
   section (plus the `null`-at-position case for order binding); the declared
   `UNIVERSE_MAX_ROWS` count.
3. **Derive packing and page counts.** Measure the largest index page, digest page and sidecar
   page that fit their caps; derive `max_rows_per_index_page`, `max_digests_per_digest_page` and
   `max_entries_per_sidecar_page`; from the declared ceiling derive the worst-case
   `index_page_count`, per-section and total `digest_page_count`, and `sidecar_page_count`.
4. **Measure the root against its own derived bindings.** Build the synthetic root with the
   worst-case page lists from step 3 — all six section bindings and every page entry present —
   and measure its canonical bytes against `ROOT_MAX_BYTES`. Because the root now binds page
   lists rather than digest sequences, its size depends on the page counts derived in step 3, so
   this step must run after them; the root bound is therefore a measured proof obligation, not
   an assumption about a "small" record.
5. **Measure each page kind and the aggregate.** Canonical bytes for the worst-case index page
   against `INDEX_PAGE_MAX_BYTES`, digest page against `DIGEST_PAGE_MAX_BYTES`, sidecar page
   against `SIDECAR_PAGE_MAX_BYTES`, and the projected total canonical bytes across all sidecar
   pages against `SIDECAR_AGGREGATE_MAX_BYTES`.
6. **Record and gate.** Record every measured byte value and every derived page count in the
   preflight/capture report. If any measurement exceeds its cap, or the aggregate cannot be
   bounded below `SIDECAR_AGGREGATE_MAX_BYTES`, refuse with `synthetic_size_over_cap` /
   `aggregate_unbounded` and stop. Selecting a larger cap, paginating further, or accepting a
   smaller universe is a lead decision, never an automatic runtime adjustment.

Until this preflight states the index, digest, sidecar and root bounds, no capture is authorized
(audit §6), and no size in this report — including the 763,488-byte structural digest lower
bound and the proposed `UNIVERSE_MAX_ROWS = 4,096` — is a fit claim.

### 6.3 Capture-level atomic failure semantics

- A **capture-level** failure (only) returns one typed refusal: `status="refused"`, `reason`,
  bounded `field`, `row_count=0`, no index page, no digest page, no sidecar content, no root,
  nothing persisted. This preserves the shipped `_refused` shape (`:462-466`) and the
  "capture refusal carries zero rows" invariant.
- `status` is a **capture** status with exactly two values: `complete` — all `row_count` index
  rows plus every index, digest and sidecar page assembled and validated and the root validated
  (row-local support refusals may still be present and do not change it) — and `refused`, the
  atomic zero-row failure above. There is no `partial`, no sampled and no truncated status.
- A capture-level refusal never coexists with a partial page set. Collect-all-then-validate-then-
  persist: pages are staged in memory, every index/digest/sidecar page and the root validate,
  and only then is the capture `complete`. A late capture-level refusal discards all staged
  pages.
- Capture-level failure codes: `universe_count_mismatch`, `universe_order_mismatch`,
  `universe_identity_mismatch`, `row_cap_exceeded`, `index_page_over_cap`, `digest_page_over_cap`,
  `sidecar_page_over_cap`, `sidecar_aggregate_over_cap`, `root_over_cap`, `record_bytes`.
- Row-local outcomes are **not** in that list and never produce this refusal: they are the
  bounded `support_refusal_reason` codes of §5, which leave the capture `complete` with
  `row_count` unchanged. The previous draft wrongly listed `sidecar_entry_over_cap`,
  `witness_digest_mismatch` and `payload_digest_mismatch` as capture failures; §6.4 assigns them
  to the row-local set.

### 6.4 Row-local versus capture-global classification

| Trigger | Scope | Outcome |
|---|---|---|
| One row's matcher witness / support payload / source-tap witness unavailable, or its witness/payload shape or digest mismatched | **row-local** | index row and `disposition` survive; `support_state=refused` with one §5 code; capture stays `complete` |
| One row's support material exceeds `SIDECAR_ENTRY_MAX_RAW_BYTES` or its canonical entry exceeds `SIDECAR_ENTRY_MAX_BYTES` | **row-local** | that row records `support_payload_over_cap`; no sidecar entry is written for it; capture stays `complete` |
| A row's digest is unavailable for a section | **row-local** | explicit `null` at that position; capture stays `complete` |
| `len(candidates)` vs `len(dispositions)` vs `len(matcher_witness_rows)` mismatch, or observed `row_count` ≠ the declared expected count | **capture-global** | `universe_count_mismatch`, `row_count=0` |
| Duplicated, dropped or reordered admitted index; broken page adjacency or overlap | **capture-global** | `universe_order_mismatch`, `row_count=0` |
| Whole-universe or kept-subset identity digest mismatch; any index/digest/sidecar page hash mismatch; a digest page that does not match its section/count/index binding | **capture-global** | `universe_identity_mismatch`, `row_count=0` |
| `len(candidates) > UNIVERSE_MAX_ROWS` | **capture-global** | `row_cap_exceeded`, `row_count=0` |
| A canonical index, digest or sidecar page exceeds its page cap | **capture-global** | `index_page_over_cap` / `digest_page_over_cap` / `sidecar_page_over_cap`, `row_count=0` |
| Canonical total across sidecar pages exceeds `SIDECAR_AGGREGATE_MAX_BYTES` | **capture-global** | `sidecar_aggregate_over_cap`, `row_count=0` |
| Root canonical bytes exceed `ROOT_MAX_BYTES`, or its page bindings do not sum to the page counts | **capture-global** | `root_over_cap`, `row_count=0` |

Rationale for the split: anything that identifies *which* candidate's support material is absent
or oversized is decidable per row and must not discard the other 1,927 dispositions; anything
that casts doubt on *which* rows exist, in what order, or whether the bounded artifact can exist
at all is a property of the whole capture and must refuse atomically. Page caps are
capture-global even though they bound per-entry material, because a page is a packing of many
entries and the §6.1 invariant already guarantees that a row-local-legal entry fits a fresh page:
an overflowing page is therefore an envelope/integrity defect, not a row outcome.

## 7. Phase A freeze, then evaluator-only projection

1. **Phase A (inference path, evaluator-blind):** capture the frozen universe from the existing
   sinks — `SameAxisCandidateObservation`, `PreFilterMatcherWitnessRowV1` rows,
   `SourceTapWitnessBindingV1`, ingress — and freeze `CandidateSupportCaptureRootV2` with all
   index, digest and sidecar page hashes and the derived section digests. Phase A stores **no**
   evaluator ownership, truth, camera, mask or RGB material; the index/sidecar/digest records
   carry only source-space indices, digest values and their bindings.
2. **Freeze barrier:** once the root digest exists the capture is immutable; nothing later
   mutates a row, page, digest or cap.
3. **Phase B (evaluation-only):** run the shipped
   `project_ownership_into_source_space` (`:787-863`) against the frozen
   `SourceTapWitnessBindingV1` used at capture time and a separately pinned
   `ProjectedEvaluatorOwnershipV1`. The join key is the saved witness index and its digests —
   no `plane_signature` replay on evaluator pixels, no evaluator RGB read, no feedback into
   inference. Projection output is a separate `OwnershipProjectionV1` /
   `OwnershipProjectionAbstentionV1` record and never alters Phase A material.

## 8. Synthetic test specification (proposed; not written)

Tests mirror the existing harness style in `tests/test_candidate_support_provenance.py`
(`_match`, `_pair_witness`, `_payload_row`, `_refusal_row`, `_observation`, `_collect`).

1. **Exactly 1,928 rows:** a synthetic universe of exactly 1,928 admitted candidates (mixed
   kept/dropped) assembles; root `row_count == 1_928`; every index/sidecar page count sums
   exactly and each of the six digest sections holds exactly 1,928 positions; all 1,928
   dispositions and all `admitted_row_index` values round-trip with no sampling or truncation.
2. **1,929 mismatch:** one extra candidate refuses atomically with
   `universe_count_mismatch`; `row_count == 0`, no pages, no root, nothing persisted.
3. **Order violations:** a shuffled page, a duplicated index, a missing middle index and a
   cross-page gap each refuse with `universe_order_mismatch`, zero partial output.
4. **Identity binding and digest pages:** the whole-universe digest and the kept-row
   `input_match_identity_sha256` replay; the six sections page to exactly `row_count` positions
   including `null`s; one flipped digest value, a dropped digest page, a reordered digest page, a
   mismatched `section_sha256` and a digest page whose count disagrees with its root binding each
   refuse capture-globally; a record whose kept-subset digest is correct but whose full-universe
   digest is wrong still refuses.
5. **Per-row typed support refusal is row-local:** a universe where some rows carry support and
   others refuse keeps all 1,928 index rows, marks `present`/`refused` consistently, never drops
   a refused row, and assembles as `status=complete` with `row_count == 1_928` and no
   capture-level refusal code.
6. **Row-local entry caps:** a row whose raw arrays exceed `SIDECAR_ENTRY_MAX_RAW_BYTES`, and
   separately a row whose canonical entry exceeds `SIDECAR_ENTRY_MAX_BYTES`, records
   `support_payload_over_cap` on that index row only; its disposition survives and the capture
   still assembles `complete`.
7. **Canonical-vs-raw edge, both scopes:** a payload under the raw cap whose canonical *entry*
   exceeds `SIDECAR_ENTRY_MAX_BYTES` refuses that row only (row-local); a case in which the
   canonical *page* total then exceeds `SIDECAR_PAGE_MAX_BYTES` refuses the whole capture
   (`sidecar_page_over_cap`, capture-global, `row_count=0`).
8. **Aggregate sidecar cap:** a declared aggregate that the measured total exceeds refuses the
   whole capture atomically (`sidecar_aggregate_over_cap`).
9. **Root/page/digest cap edge:** an oversized index page, digest page, sidecar page and root
   each refuse with their own bounded code, zero pages persisted.
10. **Capture status versus row support:** a universe with some `support_state=refused` rows
    yields `status=complete` and `row_count == 1_928`; injecting one order or identity violation
    into that same universe yields `status=refused`, `row_count == 0`, no pages, no root —
    proving the two outcomes are distinct and never conflated.
11. **Atomic-failure invariant:** every capture-level refusal yields `row_count == 0`, no pages,
    no root, no sidecar and no "partial"/"complete" relabeling; a row-local refusal never yields
    any of those.
12. **Default-off parity:** with no evaluation sink attached, the existing graph result, its
    digests and all shipped v1 behavior are byte-for-byte unchanged; the new capture allocates
    nothing.
13. **Synthetic preflight bounds (default-off, no inference read):** the preflight derives the
    index/digest/sidecar page counts at the declared ceiling, measures root/digest/index/sidecar/
    aggregate canonical bytes against their caps, refuses `cap_configuration_invalid` when the
    §6.1 invariant is violated, and refuses `synthetic_size_over_cap` / `aggregate_unbounded`
    when any measurement exceeds its cap; an `ok` preflight emits only counts and byte values and
    reads no candidate, witness or payload material.
14. **Phase A evaluator-blindness:** a Phase A record contains no ownership/truth/camera/mask
    material and no digest sequence outside its digest pages; Phase B given identical frozen
    bindings reproduces the shipped projection result and does not re-match or re-signature.
15. **Determinism:** repeated captures of the same frozen universe produce identical page and
    root digests.
16. **No extra work:** one matcher pass and one signature pass; no additional frame or image
    read during capture validation.

## 9. Open items for the lead

1. Approve per-row refusal over v1's whole-capture refusal for a missing signature witness (§5).
2. Set `SIDECAR_AGGREGATE_MAX_BYTES`, `SIDECAR_ENTRY_MAX_BYTES`, `DIGEST_PAGE_MAX_BYTES` and
   `UNIVERSE_MAX_ROWS` after the synthetic preflight reports measured sizes; none is claimed to
   fit here.
3. Authorize (or not) a bounded implementation task for the chosen option plus the preflight
   measurement. No capture is authorized by this report.
4. Decide whether the whole-universe digest domain should be shared with R7/R9 (as
   `input_match_identity` is) or new (`complete_support_universe`).
5. Confirm that an unavailable per-row digest is recorded as an explicit `null` at its position
   (chosen here so every section keeps `row_count` positions and page boundaries cannot renumber
   the admitted order) rather than as a shortened section or a separate availability bitmap.
6. Confirm the two-value capture status naming — `complete` may contain row-local
   `support_state=refused` rows; `refused` means the atomic zero-row capture failure of §6.3 —
   so "complete" is never read as a support-completeness claim.

## 10. Provenance and limits

- Source read: `candidate_support_provenance.py` (863 lines, SHA-256
  `A8BD3102652720513ED85E2D3B7E2C81A79A01A7621C41238B09F6E741604562`, per the accepted
  DS-R4-TAP-SENTINEL state); `plane_graph.py` (`SameAxisCandidateObservation`,
  `PreFilterMatcherWitnessRowV1`, `MatcherPairWitnessV1`, `GraphMatcherWitnessPayloadV1`,
  `_match_order`, `_indexed_match_payload`, `match_plane_graph`); `image_derived_graph.py`
  (candidate-support sink call, same-axis capture contracts); `DS-R4-BYTE-CAP-AUDIT-2026-10-04.md`.
- The 1,928 universe, the 448-row active-group subset, the 38 post-filter inputs and the
  conditional 7,139×4×8 raw-byte fact are taken from the accepted audit; no artifact payload
  was reopened here.
- Revision provenance: this amendment applies only the Lead hold of 2026-10-04 19:53 UTC
  (`AGENT_COMMS.md`). It moves the six ordered digest sequences into bounded content-hashed
  digest pages, makes the root bind page counts/order/digests plus whole-universe identity,
  makes the synthetic preflight prove the root/digest/index/sidecar/aggregate bounds, and
  separates row-local `support_state=refused` from capture-level `status=refused`,
  `row_count=0`. The 763,488-byte digest lower bound is arithmetic on the declared field width
  (6 × 1,928 × 66), not a measured artifact size; every cap fit remains unverified.
- No byte fit is claimed in either direction. No source, test or schema file was edited; no
  reconstruction, evaluator/truth/camera/mask/image payload read, sampling, truncation,
  capture or authorization was performed; no PowerShell/bash/jobs/subagent/workflow/UI/
  Minecraft/launcher/system operation was used.
- Stopped for Luna/lead review. No R1 rerun is authorized or performed.
