# DS-R4-V2-COLLECTOR-VALUEERROR-TRACE — read-only source audit of R2's refused V2 capture

- Date: 2026-10-04
- Worker: DeepSeek CLI worker (read-only source audit)
- Assignment: `AGENT_COMMS.md` "2026-10-04 LEAD -> DeepSeek - DS-R4-V2-COLLECTOR-VALUEERROR-TRACE"
- Policy: `docs/deepseek-readonly-no-shell.patch.yml` (no shell, no jobs, no workflow, no subagent)
- Sources read only: `src/mc_photogrammetry/pipeline/candidate_support_provenance_v2.py`,
  `src/mc_photogrammetry/pipeline/candidate_support_provenance.py`,
  `src/mc_photogrammetry/reconstruction/plane_graph.py`,
  `src/mc_photogrammetry/pipeline/image_derived_graph.py`,
  `scripts/evaluation/run_route_b_diagnostic.py`,
  `tests/test_candidate_support_provenance_v2_preflight.py`,
  `tests/test_candidate_support_provenance.py`,
  `docs/reports/DS-R4-COMPLETE-SUPPORT-DESIGN-2026-10-04.md`,
  `docs/reports/DS-R4-V2-ROUTEB-FROZEN-IMAGE-R2-2026-10-04.md`,
  `.tmp-routeb-v2-frozen-r2-20261004/refusal.json`

## 0. Scope and disposition

**READ-ONLY AUDIT. No code edit, no test edit, no Python execution, no test/runner/capture
execution, no image/truth/reference/litematic payload opened, no shell/tools/workflows/subagent,
no reparse/recursive deletion/Defender/system change, no background work, no client automation.**
The only file written by this task is this report. The one data file inspected outside the report
set is R2's own persisted `refusal.json` (already public in R2's report).

Disposition: the R2 raw `ValueError` **cannot** be localized to an operation, field or row from the
persisted refusal. The broad catch region is much wider than the single "identity" invariant it is
named after, and it deliberately discards everything except `type(error).__name__`. A bounded,
source-only operation/row tag is the smallest seam that recovers the failing operation without
opening a data channel; a synthetic regression can pin it without any image payload.

## 1. Provenance chain traced (all read-only, line references exact)

1. `image_derived_graph.py:5183-5198` — only on the solver-abstention branch, when
   `complete_support_sink is not None`, the graph calls
   `complete_support_mod.collect_route_b_diagnostic_support(readings, matches,
   signature_witnesses or (), pre_filter_rows or (), raw_observations[0] if raw_observations else
   None, ingress=ingress_records, failed_assignment_sha256=...)`.
2. `candidate_support_provenance_v2.py:1960-1993` — `collect_route_b_diagnostic_support` first
   guards `expected_input_match_identity_sha256 = _match_order_identity(matches)` in
   `except (AttributeError, TypeError, ValueError, IndexError, KeyError)` and, on any such failure,
   silently substitutes `None` (`:1981-1983`). It then delegates every other contract to
   `collect_complete_candidate_support` with the frozen plan/expected-row profile (`:1984-1993`).
3. `candidate_support_provenance_v2.py:1800-1842` — `collect_complete_candidate_support` resolves
   the plan (`:1820-1823`, itself guarded by `_resolve_plan`), then calls
   `_collect_complete_support` (`:1825-1834`) inside the **`try` whose broad arm is the reported
   mint** (`:1838-1842`).
4. `candidate_support_provenance_v2.py:1845-1938` — `_collect_complete_support` performs the
   pre-checks and the full admitted-row walk, then calls
   `_assemble_complete_support` (`:1928-1938`) **still inside the same try region**, because the
   `return` of the collector is the call that assembles.

Therefore the guarded region is **not** "identity checking". It is the entire collector body,
every per-row conversion/construction, the page packing, the root construction, and the full
`CompleteSupportCaptureV2.__post_init__` → `validate_complete_support_capture` replay
(`:993-994`, `:1139-1320`). Any raw `ValueError` (or `TypeError`/`IndexError`/`KeyError`/
`OverflowError`/`ZeroDivisionError`) anywhere in that span is coerced to
`refused_complete_support("universe_identity_mismatch", type(error).__name__, plan=...)`
(`:1840-1842`). `refused_complete_support` keeps the reason (it is already in
`CAPTURE_REFUSAL_REASONS`, `:186-199`) and bounds/keeps the field
(`:1028-1044`), so the reported pair is `universe_identity_mismatch` / `ValueError`.

The identical broad arm also exists at `:1755-1759` for the sibling
`build_complete_support_capture` (the pure pre-built packer); it is the same shape but R2 did not
use it (R2 entered through the collector).

### 1.1 Why the reported field proves the escaping error was **raw**, not typed

`CandidateSupportError` is a `ValueError` subclass (`candidate_support_provenance.py:112-119`) and
its `str()` is the reason (optionally `reason: detail`). It is caught **first**
(`:1835-1837`) and would produce the typed `reason`/`field` pair, never the literal string
`"ValueError"`. The persisted field is exactly `type(error).__name__` from the second arm, so the
escaping error was an untyped `ValueError` (the audit agrees with R2's report on this point).

### 1.2 A second, earlier masking site in the same call chain

`_match_order_identity(matches)` (`:1941-1957`, called at `:1981`) has its own broad
`except` and substitutes `None` on failure. Because `_assemble_complete_support` only enforces the
kept-subset identity when the expected value is truthy (`:1564-1567`), an exception there silently
**drops the kept-subset identity binding for the whole run** rather than refusing. This is a
separate diagnostic blind spot in the same chain: if `_match_order_identity` itself raised on R2's
solver tuple, the run would continue with `expected_input_match_identity_sha256=None` and the
identity gate would silently not fire. It is not the reported refusal (the reported `field` came
from the collector's arm), but it is a second place where an exception is swallowed without any
persisted trace, and it belongs in the same minimal seam review.

## 2. Exact operations inside the broad catch region

The region is `collect_complete_candidate_support`'s `try: ... except CandidateSupportError ...
except (broad)` (`:1824-1842`). The operations executed in it are listed below in execution order.
"Typed" means the operation's own validation raises `CandidateSupportError` (captured by the first
arm with its `reason`/`field`); "raw" means the operation can raise a bare built-in exception that
only the broad arm sees. `CandidateSupportIndexRowV2.__post_init__` and the other record
constructors ship **both** kinds, so the same constructor can produce either a precise typed
refusal or a mask, depending on which of its internal checks trips.

### 2.1 Pre-checks (typed only; `:1851-1867`)

| # | Operation | Raised exception |
|---|---|---|
| 1 | `isinstance(same_axis_observations, plane_graph.SameAxisCandidateObservation)` | typed `universe_count_mismatch` / `same_axis_observation` |
| 2 | `type(expected_row_count) is not int or <= 0` | typed / `expected_row_count` |
| 3 | `isinstance(matcher_witness_rows, (list, tuple))` | typed / `matcher_witness_rows` |
| 4-6 | `len(observations.dispositions) != len(observations.candidates)`, witness-row length, non-empty candidates | typed / `dispositions`, `matcher_witness_rows`, `candidates` |
| 7-8 | `len(candidates) != expected_row_count`, `> plan.universe_max_rows` | typed / `row_count` (or `row_cap_exceeded`) |

### 2.2 Per-admitted-row walk (`:1868-1927`) — the widest raw surface

| # | Operation (line) | Can raise (raw) | Notes |
|---|---|---|---|
| 9 | `witness_by_signature[item.signature_sha256] = item` (`:1869-1871`) | `TypeError` (unhashable signature digest) | `SourceTapWitnessBindingV1` is type-checked by `isinstance`, but a tampered/unhashable attribute still reaches the dict |
| 10 | `kept_positions = {id(match): index for ...}` (`:1872`) | `TypeError` (`matches` not iterable) | plain dict/`enumerate` |
| 11 | `groups.setdefault(group_key, len(groups))` (`:1885-1887`) | `TypeError` (unhashable `group_key` item) | key is `(candidate.left, candidate.right, candidate.left_axis, candidate.quadrant_of_right)` |
| 12 | `matcher_witness_rows[position]` (`:1879`) | `IndexError` | pre-checks pin lengths, so only a tampered/re-bound pair reaches it |
| 13 | `isinstance(row, PreFilterMatcherWitnessRowV1)` (`:1880-1881`) | typed / `matcher_witness_row` | |
| 14 | `row.admitted_row_index != position or row.disposition != disposition` (`:1882-1884`) | typed / `row_disposition_binding` | |
| 15 | `witness_by_signature.get(row.left_signature_sha256 or "")` (`:1891-1892`) | `TypeError` (unhashable `left_signature_sha256`) | |
| 16 | `_resolve_row_support(row, left_witness, right_witness, plan)` (`:1893-1894`) | **row-local only** | its own body catches `CandidateSupportError` and `(AttributeError, TypeError, ValueError, IndexError, KeyError)` (`:1773-1782`); except for the two calls below it cannot escape |
| 16a | └ `_alternative_index(...)` in `_resolve_row_support` (`:1902-1905` → `candidate_support_provenance.py:701-709`) | **`ValueError` (raw)** via `plane_graph._canonical_reading_index` (`plane_graph.py:123-149`: non-sequence readings, non-`PlaneReading` element, empty/non-string frame, non-canonical axis, repeated reading object, malformed reading order key) | this call is **not** inside a local try in `_collect_complete_support`; the local `_resolve_row_support` try covers `_support_payload`, not this helper |
| 16b | └ `_delta(candidate, readings)` (`:1907` → `candidate_support_provenance.py:712-717`) | same raw `ValueError` from `_canonical_reading_index`; also raw from `plane_graph._delta` (`plane_graph.py:1345-1359`: `left.distance_for(...)`, `axis_transform(...)`) | |
| 17 | `CandidateSupportIndexRowV2(...)` (`:1895-1916`) | **`ValueError` (raw)** from its own `__post_init__` (`:484-521`): malformed row fields (`:502`), kept row without match index (`:505`), dropped row with match index (`:507`), malformed cell shift (`:511`), present row without payload digest (`:515`), refused row without exactly one bounded code (`:518`) | the same constructor raises **typed** `index_row_over_cap` / `<field>` at `:519-521`, so this one constructor straddles both arms |
| 17a | └ `int(candidate.left_axis)`, `int(candidate.right_axis)`, `int(candidate.quadrant_of_right)`, `int(candidate.left_plane)`, `int(candidate.right_plane)`, `int(candidate.overlap_cells)`, `int(candidate.cell_shift_rc[i])` (`:1898-1912`) | `TypeError` (e.g. `None`); `ValueError` (e.g. non-numeric string) | these conversions run **before** the constructor's checks |
| 17b | └ `float(candidate.left_b)`, `float(candidate.right_b)` (`:1906`) | `TypeError`/`ValueError` | |
| 17c | └ `_alternative_index(readings, ...)`, `_delta(candidate, readings)` inside the constructor arguments (`:1902-1907`) | raw `ValueError` as rows 16a/16b | |
| 18 | `MatcherSupportSidecarEntryV2(admitted_row_index=position, payload=support)` (`:1918-1919`) | `ValueError` (raw, `:613-618`) **or** typed `support_payload_over_cap` (`:619-621`) | same constructor straddles both arms |
| 19 | `_canonical_bytes(entry.to_record())` inside `_resolve_row_support`'s cap check (`:1795`) | `TypeError`/`ValueError` (non-serializable / non-finite payload value) | the `:1773-1782` try surrounds `_support_payload`, not this later serialization |
| 20 | `_identity_digest([_row_identity(index_row)])` (`:1926-1927`) | `TypeError`/`ValueError` (`float()` in `_row_identity`, `:433-436`); also JSON `allow_nan=False` in `_sha` | rows were already finite-checked by `_row_field_over_cap`, so this is a tamper path |
| 21 | `sections[...].append(...)` (`:1920-1925`) | `KeyError` (only if `DIGEST_SECTIONS`/`sections` were tampered) | normally impossible |

### 2.3 Assembly (`_assemble_complete_support`, `:1502-1607`; reached from `:1928-1938`) — still inside the same try

| # | Operation (line) | Can raise (raw) | Notes |
|---|---|---|---|
| 22 | typed pre-checks (`:1514-1578`) | typed `CandidateSupportError` (`universe_count_mismatch`, `universe_order_mismatch`, `index_row_over_cap`, `universe_identity_mismatch`, `ingress_over_cap`) | these are the "named" refusals |
| 23 | `complete_support_universe_digest(rows)` (`:1556` → `:439-445`) | `TypeError`/`ValueError` via `_sha` JSON (`allow_nan=False`), `_row_identity` `float()` | finite/int checks precede it, so tamper-only |
| 24 | `_pack_index_pages` (`:1579` → `:1377-1394`) | `CandidateSupportIndexPageV2` raw `ValueError`s (`:571-589`: not consecutive, digest mismatch, malformed) | typed `index_page_over_cap` is also here (`:1389-1391`) |
| 25 | `_pack_digest_pages` (`:1580` → `:1474-1497`) | `CandidateSupportDigestPageV2` raw `ValueError`s (`:726-741`) | typed `digest_page_over_cap` also here |
| 26 | `_pack_sidecar_pages` (`:1581` → `:1424-1442`) | `MatcherSupportSidecarPageV2` raw `ValueError`s (`:673-693`: entries not ordered, page digest mismatch, malformed) | typed `sidecar_page_over_cap` also here |
| 27 | `sum(len(_canonical_bytes(page.to_record())) ...)` (`:1582`) | `TypeError`/`ValueError` (non-serializable page) | |
| 28 | `CandidateSupportIndexPageBindingV2` (`:1585-1587`), `CandidateSupportSidecarPageBindingV2` (`:1588-1590`), `CandidateSupportDigestPageBindingV2` + `CandidateSupportDigestSectionV2` (`:1591-1594`, `:1047-1060`) | raw `ValueError` (`:753-757`, `:770-774`, `:788-794`, `:809-816`); typed `universe_identity_mismatch` / `section.<name>` for an empty section (`:1050-1051`) | |
| 29 | `CandidateSupportCaptureRootV2(...)` (`:1595-1602`) | raw `ValueError`s (`:878-964`) | typed `ingress_over_cap` also here (`:955-959`) |
| 30 | `_canonical_bytes(root.to_record())` (`:1603`) | `TypeError`/`ValueError` | |
| 31 | `CompleteSupportCaptureV2(...)` → `validate_complete_support_capture` (`:1605-1607` → `:993-994`, `:1139-1320`) | **`ValueError` (raw)** at ~40 independent checks: root digest replay (`:1150`), refused-shape checks (`:1153-1187`), plan replay (`:1190-1201`), page counts/bindings/consecutiveness (`:1202-1244`), sidecar coverage/replay/binding (`:1245-1256`), digest sections (`:1257-1283`), identity digest sections (`:1284-1295`), and independent cap enforcement (`:1299-1320`) | this is the largest single raw-`ValueError` surface in the region, and it is *inside* the same `try` as everything above |

### 2.4 Exceptions the broad arm names but that are not `ValueError` in practice

The arm catches `AttributeError, TypeError, ValueError, IndexError, KeyError, OverflowError,
ZeroDivisionError` and reports **only the type name**. A `TypeError` from `float(None)` or an
`IndexError` from a re-bound row tuple therefore reaches disk with the same "no site" property as
the raw `ValueError`. Any seam that keys only on `ValueError` would leave those sites silent; the
recommendation below covers the whole arm.

## 3. Which raw sites can and cannot be localized from R2's persisted refusal

R2's persisted artifact is `.tmp-routeb-v2-frozen-r2-20261004/refusal.json` (494 B). Its exact
content (read directly for this audit) carries only:

```json
{"detail":"","field":"ValueError","inputs":[...three paths...],"reason":"universe_identity_mismatch",
 "schema":"mc-photogrammetry-route-b-diagnostic-runner-refusal-v1",
 "settings":{...},"status":"refused"}
```

`detail` is empty because the runner's `_write_refusal`/`_refusal_record`
(`scripts/evaluation/run_route_b_diagnostic.py:718-733`, `:788-795`) persists `reason`, `field`,
`detail` from the raised `DiagnosticRefusal`, and `replay_capture` raises with only the capture
root's `reason`/`field` (`:534-536`). No Python frame, no operation, no row index, no exception
message and no traceback is persisted; the zero-row refused capture
(`:1037-1044`) carries no partial universe; and R2's run produced no `record.json`, no
`route_b.mcscan`, no staging content and no `stage_digests`.

**Can be localized:** only the *class* of failure — a raw (untyped) built-in exception escaped
`_collect_complete_support`; if the type is `ValueError`, it is a raw `ValueError`, not a typed
`CandidateSupportError`; the capture was refused atomically with zero rows; and every stage before
the graph completed (entry/lattice/signatures/graph and exactly-once sink firing are already
attested by R2's report and the `refusal.json` shape).

**Cannot be localized:** which of rows 9-31 fired; whether it was the per-row walk (rows 9-21) or
assembly/validation (rows 22-31); which admitted row/position; which field; which record
constructor or validator check; whether `_match_order_identity` (row 1.2) also failed; and whether
the input reading universe supplied to the graph sink was itself malformed. The reported
`reason` ("universe_identity_mismatch") is *not* evidence of an identity mismatch: it is the fixed
coercion label for **every** raw built-in exception in the region, which is exactly what makes it
non-localizing. No source-only reading of the persisted artifact can distinguish these cases; a
re-run without an added seam would only re-emit the same four-field refusal.

## 4. Recommended smallest safe source-only diagnostic seam

The goal is to retain the failing **operation** (and, for per-row operations, the deterministic
admitted position) in the persisted refusal, with no image/pixel/payload data and no change to the
canonical record schema. Three properties make the smallest seam possible:

- The runner persists `field` verbatim (bounded at `REFUSAL_FIELD_MAX_BYTES`,
  `_bounded_refusal_field`, `:390-405`), so `field` is already the one existing, bounded,
  data-safe diagnostic channel.
- `refused_complete_support` keeps any reason that is in `CAPTURE_REFUSAL_REASONS`
  (`:1034-1035`) and does not require `field` to be a reason code, so the operation tag can live in
  `field` without changing `reason`, `status`, `row_count`, page shape or the root schema.
- Every operation code and admitted position is a source-derived integer/short ASCII token, not
  image material, so it cannot leak pixels, paths, masks or reference content.

**Seam (source-only, one collector, one refusal field):** in
`collect_complete_candidate_support`'s broad arm (`:1838-1842`), replace the constant
`type(error).__name__` string with a bounded composite of the exception type plus a fixed-allowlist
operation code, for example:

- introduce a module-private bounded code table (a `str` → short token map) covering the
  operations in §2, e.g. `"row-constructor"`, `"sidecar-entry"`, `"reading-index"`,
  `"delta"`, `"row-identity"`, `"universe-digest"`, `"index-page"`, `"digest-page"`,
  `"sidecar-page"`, `"page-binding"`, `"root-constructor"`, `"capture-validate"`,
  `"entry-construct"` (new entries require a source edit and a test, so the allowlist cannot
  silently grow);
- set a module-level "current operation" marker before each operation, or wrap only the small
  operation helpers with a targeted `try/except` that re-raises the original exception with the
  operation attribute attached, then read that attribute in the broad arm. The wrapper variant
  keeps the change local and does not restructure the collector;
- build `field` as `_bounded_refusal_field(f"{type(error).__name__}:{op_code}" + (f"#{position}" for
  per-row operations))`. The type name is a fixed builtin name, `op_code` is a fixed source
  constant and `position` is an admitted integer index — all bounded, all data-safe, and the
  existing `_bounded_refusal_field` already truncates deterministically. A larger position is
  impossible (`plan.universe_max_rows == 1_928`).
- leave `reason` as `universe_identity_mismatch` and every typed path untouched, so the accepted
  design's §6.3/§6.4 refusal contract, the zero-row invariant and the canonical root schema do not
  move.

Do **not** put `str(error)` or a traceback into the refusal: `KeyError`/`AttributeError` messages
embed arbitrary data-derived text, and a traceback embeds paths. The operation code plus row index
is sufficient to localize the site, and it is the only content the persisted channel can carry
safely today.

**Optional second half of the same seam:** give `_match_order_identity`'s guard (`:1980-1983`) the
same treatment instead of silently substituting `None`, because a swallowed failure there (a)
leaves the kept-subset identity binding unenforced and (b) is invisible in the persisted refusal.
The minimal form is to record the same bounded `identity` operation code when the guard fires; the
larger form (refuse instead of continue) is a contract decision for the lead, not a diagnostic
edit, and is called out here only so the lead can accept or reject it explicitly.

**Not proposed:** any change to the canonical `CompleteSupportCaptureV2`/root schema, any new
persisted sidecar channel, any production default, any real re-run, and any image/truth/reference
payload. This audit proposes no re-run at all; the seam is the prerequisite for a future,
separately authorized diagnostic that would finally localize R2's raw `ValueError`.

## 5. Recommended synthetic regression

Add to `tests/test_candidate_support_provenance_v2_preflight.py` (synthetic in-memory material
only, no image/truth/evaluator/litematic payload), reusing `_collector_material` /
`_route_b_universe` / `_observation` / `_readings_for`:

1. **Per-operation localization, one case per §2 raw operation.** For each operation, monkeypatch
   that one helper/constructor to raise a raw `ValueError` (and, in a second pass, a `TypeError`)
   at a known admitted position, call `collect_complete_candidate_support`, and assert:
   `capture.status == "refused"`, `capture.root.reason == "universe_identity_mismatch"`,
   `capture.root.field == "<TypeName>:<op-code>#<position>"` (exact string), and
   `capture.row_count == 0`, `capture.index_pages == ()`, `capture.digest_pages == ()`,
   `capture.sidecar_pages == ()`. Before the seam the assertion is exactly `field == "ValueError"`;
   after the seam it names the operation and row, so the test is a true failing-before/passing-after
   control for the localization, not just for the refusal.
2. **Different operations must produce different fields.** Two cases (e.g. a per-row
   `CandidateSupportIndexRowV2` `ValueError` at position 1 and a `_canonical_reading_index`
   `ValueError` from `_alternative_index`) must yield two distinct fields, proving the tag family
   does not collapse back to one constant.
3. **No data leak and determinism.** Assert the resulting `field` is drawn only from
   `[A-Za-z0-9_:.#-]` and is at most `REFUSAL_FIELD_MAX_BYTES` serialized bytes; assert the
   refusal's canonical bytes for the same injected fault are identical across two runs (the tag is
   deterministic and carries no witness/pixel/array material). Optionally assert the field string
   does not contain any admitted frame id or the payload digest fed to the fixture, which is the
   concrete "no image/payload data" control.
4. **Typed paths unchanged.** Keep one case per typed reason (`universe_count_mismatch`,
   `index_row_over_cap`, `universe_identity_mismatch`, the cap reasons) asserting the typed
   `reason`/`field` pair is still produced, so the seam cannot swallow a typed refusal into the new
   tag.
5. **`_match_order_identity` guard (if the lead accepts the optional half).** A solver tuple whose
   `float(match.left_b)` raises must (a) no longer silently drop the binding, and (b) surface the
   bounded identity operation tag in the refusal.

The regression needs no real screenshot, no capture, no runner invocation, no evaluator/truth
input and no production default change; it is entirely the existing synthetic fixture family.

## 6. Checks performed and constraints honored

- Read-only source inspection with line-numbered reads and content greps only. No Python
  interpreter, `pytest`, `py_compile`, runner, capture, eval or image decode was executed.
- No file was edited except this report. No `AGENT_COMMS.md` edit is included in this report step;
  the task's concise result is appended by the separate comms step.
- The only inspected data artifact is R2's own 494 B `refusal.json`, which R2's report already
  published. No truth/evaluator/reference/litematic/scene/camera/F3/direction/mask/coordinate
  payload was opened, and no image file was read.
- No shell/tools/workflows/subagent, no reparse or recursive deletion, no Defender or system
  change, no background/persistent process and no Minecraft/launcher or client automation was used.

## 7. Boundary

This audit localizes a **source-supported candidate set** and a **seam design**, not the actual R2
failure. It makes no accuracy claim, no claim that the frozen images pass downstream stages, and no
claim about the 1,928-row universe beyond R2's own refusal. A real re-run remains a separate,
separately authorized step, and the recommended seam is the smallest prerequisite that would let
such a run name the failing operation without leaking image data.
