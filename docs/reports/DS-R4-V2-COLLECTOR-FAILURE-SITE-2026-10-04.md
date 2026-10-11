# DS-R4-V2-COLLECTOR-FAILURE-SITE-2026-10-04

Task ID: `DS-R4-V2-COLLECTOR-FAILURE-SITE`, including the lead clarification at
`AGENT_COMMS.md:7155-7156`, and the remediation `DS-R4-V2-COLLECTOR-FAILURE-SITE-REMEDIATION`
at `AGENT_COMMS.md:7175-7176` (Luna HOLD at `AGENT_COMMS.md:7170-7173`). Owner: DeepSeek CLI
worker (Sol-equivalent, bounded implementation, direct-shell profile). Date: 2026-10-04.

Status: **IMPLEMENTED — remediation applied; for Luna re-review. Stopped as instructed.** Only
`src/mc_photogrammetry/pipeline/candidate_support_provenance_v2.py`, its focused test module
`tests/test_candidate_support_provenance_v2_preflight.py`, and this report were written; a
concise result was appended to `AGENT_COMMS.md`. No production runner/inference/capture, no
image/screenshot/truth/evaluator/reference/litematic payload, no Minecraft/launcher, no
performance tuning, no unrelated dirty file, and no real image run was performed.

## 1. Problem addressed

`DS-R4-V2-ROUTEB-FROZEN-IMAGE-R2` ran the repaired Route B diagnostic once: all four stages ran
(`entry -> lattice -> signatures -> graph`), the V2 sink fired exactly once, and the runner then
refused the non-complete capture with `reason="universe_identity_mismatch"`,
`field="ValueError"`, empty `detail`. `DS-R4-V2-COLLECTOR-VALUEERROR-TRACE` traced that field to
the broad raw-exception arm of `collect_complete_candidate_support` (then `:1838-1842`), which
coerced every raw built-in exception to the fixed `universe_identity_mismatch` /
`type(error).__name__` pair. The coercion carried no information about which operation, source
line or admitted row raised the raw exception, so the failing site could not be localized from
the persisted refusal.

Luna's review of that trace is **HOLD** and noted that a partial operation allowlist would miss
uncaught operations inside the broad try (for example `tuple(ingress)` at the call boundary and
`MatcherSupportSidecarEntryV2` construction/serialization after `_resolve_row_support`). This
implementation therefore uses **trusted traceback static metadata** rather than a partial
operation list, so coverage is total and derived from the interpreter.

This is a diagnostic-only source change with synthetic tests only: no real capture is invoked,
and this report makes no claim about what the frozen R2 images do downstream.

## 2. Change (source)

All edits are inside `src/mc_photogrammetry/pipeline/candidate_support_provenance_v2.py`.

1. **New bounded field builder** `_raw_failure_field(error, position=None)` (`:526-543`): returns
   `"<Category>:<module>.<operation>[:<line>][@row<position>]"`, where:
   - `<Category>` is a fixed canonical built-in category from
     `_raw_failure_exception_category` (`:450-462`), selected by `isinstance` against the declared
     caught bases (`:444-446`), applied identically to every raw class in the catch tuple (not a
     `ValueError` special case). It is **never** `type(error).__name__`, so a custom subclass name
     (caller-controlled text that can embed a path) cannot enter the field;
   - `<module>` and `<operation>` are the innermost frame's module `__name__` and `co_name`,
     sanitized by `_raw_failure_token` (`:419-434`) to ASCII `[A-Za-z0-9_.]` / `[A-Za-z0-9_]` and
     capped at 64 characters each, falling back to the fixed identifier `unknown`
     (`:408-416`);
   - `<line>` is the innermost frame's `tb_lineno`, included only when it is an exact positive
     `int`;
   - `@row<position>` is appended only when `position` is an exact non-negative `int`.
   The result passes through the existing `_bounded_refusal_field`, so it always fits the
   declared `REFUSAL_FIELD_MAX_BYTES` (256) bound.
2. **New `_raw_failure_site_metadata(error)`** (`:488-523`): reads only three data-free
   interpreter values of the innermost frame — the frame module's `__name__` (a static import
   identity such as `mc_photogrammetry.reconstruction.plane_graph`), `co_name` and `tb_lineno`.
   It never reads `co_filename`, `str(error)`, rendered traceback text, frame locals, any other
   global or any object repr, so no path, payload, image-derived value or exception message can
   enter the field.
3. **New `_CollectorFailureSite`** (`:465-485`): the one piece of collector context a traceback
   cannot supply. It retains only the currently admitted row and only when that value is an exact
   non-negative `int`; `leave_row()` clears it so a non-row phase cannot be mislabeled.
4. **`collect_complete_candidate_support`** (`:1940-1988`): creates the site, evaluates
   `tuple(ingress)` and the collector call inside the same `try`, and the broad raw-exception arm
   (`:1979-1988`) now builds its field with `_raw_failure_field(error, failure_site.position)`.
   The arm still catches the same classes (`AttributeError, TypeError, ValueError, IndexError,
   KeyError, OverflowError, ZeroDivisionError`) and still returns
   `reason="universe_identity_mismatch"`. The typed `CandidateSupportError` arm (`:1976-1978`) is
   unchanged, so every typed reason/field pair is preserved.
5. **`_collect_complete_support`** (`:1991-2089`): accepts the optional `failure_site`, calls
   `enter_row(position)` at the top of the admitted-row walk (`:2026`; `position` is the exact
   `enumerate` index, already required to equal `row.admitted_row_index`) and `leave_row()`
   immediately before assembly (`:2077-2078`).

### 2.1 Exact coverage of the broad try region

The `try` at `:1965` covers, with no exceptions to the listing:

- `tuple(ingress)` at the call boundary (`:1968`), before the collector is entered;
- the whole `_collect_complete_support` call (`:1966-1975` -> `:1991-2089`):
  - all typed pre-checks (`:1997-2013`);
  - the witness map, `kept_positions` dict comprehension and `groups`/`rows`/`entries`/`sections`
    initialization (`:2014-2022`);
  - the full per-row walk (`:2023-2076`): row fetch and type check, row/disposition binding, group
    key and `setdefault`, kept-match index, left/right witness lookups, `_resolve_row_support`
    (its own local raw catch turns `_support_payload` raw failures into row-local refusals and is
    unchanged), the `CandidateSupportIndexRowV2(...)` construction expression including
    `_alternative_index`, `_delta`, the `int()`/`float()` conversions and the cell-shift tuple,
    the `MatcherSupportSidecarEntryV2(...)` sidecar construction/serialization inside
    `_resolve_row_support`, `entries.append(...)`, and the six digest-section appends including
    `_identity_digest([_row_identity(index_row)])`;
  - `_assemble_complete_support(...)` (`:2078-2088`): index/digest/sidecar page packing, the
    page-binding/section/root construction and `CompleteSupportCaptureV2.__post_init__` ->
    `validate_complete_support_capture`.

Because the field is derived from the **actual raise site's** traceback frame, every raw built-in
exception escaping any of those sites is localized; the mechanism does not depend on an operation
allowlist and cannot omit a site. The only collector-specific fact the traceback cannot supply is
the admitted row, which `_CollectorFailureSite` carries: a `tuple(ingress)` failure reports no
row, a per-row failure reports the exact row, and an assembly/validation failure reports no row
(cleared at `:2077`).

The sibling broad arm in `build_complete_support_capture` (`:1893-1899`) now applies the same
`_raw_failure_exception_category` canonicalization as the remediation (§2.2). Its field stays the
bare category name (for example `ValueError`) and is unchanged for plain built-in exceptions; only
a custom subclass name is replaced by its declared base category. No capture schema, status,
digest, plan, cap, default path or public API changed.

### 2.2 Remediation (`DS-R4-V2-COLLECTOR-FAILURE-SITE-REMEDIATION`)

Luna's HOLD (`AGENT_COMMS.md:7170-7173`) found that `_raw_failure_field` inserted
`type(error).__name__` verbatim into the persisted field. Because the broad arm catches
**subclasses** of the seven built-ins, a custom class such as
`type("C:/sentinel", (ValueError,), {})` could carry path-like text into the refusal field;
`_bounded_refusal_field` only truncates by byte count and does not sanitize it.

The repair:

- Adds `_RAW_FAILURE_EXCEPTION_CATEGORIES` (`:444-446`), the explicit caught bases in the declared
  match order (`AttributeError, TypeError, ValueError, IndexError, KeyError, OverflowError,
  ZeroDivisionError`), and `_raw_failure_exception_category(error)` (`:450-462`), which returns
  `category.__name__` for the first base that matches by `isinstance`, else the fixed
  `_RAW_FAILURE_UNKNOWN_EXCEPTION` (`"unknown"`). No base in the tuple is a subclass of another,
  so the order is deterministic with no shadowing; a custom class inheriting from several caught
  bases still resolves to a single declared category.
- Uses that category in `_raw_failure_field` (`:538`) instead of `type(error).__name__`.
- Applies the same canonicalization to the sibling raw arm of `build_complete_support_capture`
  (`:1893-1899`), which had the identical `type(error).__name__` leak on its own path (lead
  amendment `AGENT_COMMS.md:7178-7179`). For a plain built-in this is behavior-preserving (the
  category name equals the class name); only custom subclass names change, to their declared base
  category.
- Never emits a custom class name, `str(error)`, a path, traceback text, locals or a repr; the
  module/function/line/row localization and the byte bound are unchanged.

The pre-repair promise that the field "can never carry a path" is now unconditionally true for
every exception the catch accepts at **both** emission points.

## 3. Change (tests)

All edits are inside `tests/test_candidate_support_provenance_v2_preflight.py`, synthetic
in-memory material only.

- Module-level injected raisers with distinct names and distinct classes —
  `_row_site_raw_raiser` (raw `ValueError` at a row-construction argument, `:870-872`),
  `_assembly_site_raw_raiser` (raw `TypeError` at a capture page-packing site, `:875-877`), and
  `_sidecar_site_raw_raiser` (raw `ValueError` at the row-local sidecar construction after
  `_resolve_row_support`'s local catch, `:880-882`) — plus the sentinel `_RAW_FAILURE_SENTINEL`
  (`:867`) and `_raw_failure_refusal` (`:885-903`), which asserts the refusal stays atomic (zero
  rows, no pages) and still validates before returning the field.
- `test_collector_raw_row_failure_field_localizes_operation_and_position` (`:906-923`): injects at
  `_alternative_index` and asserts the field names the class, module, operation and a positive
  source line, and appends `@row0` (the exact admitted position).
- `test_collector_raw_assembly_failure_field_localizes_without_position` (`:926-939`): injects at
  `_pack_index_pages` and asserts the class/module/operation/line are named while **no** `@row`
  position is appended.
- `test_collector_raw_sidecar_failure_field_localizes_row_construction` (`:942-955`): injects at
  `MatcherSupportSidecarEntryV2` construction (the site Luna flagged as escaping after
  `_resolve_row_support`) and asserts the same localization with `@row0`.
- `test_collector_raw_failure_fields_distinguish_row_and_assembly_sites` (`:958-968`): runs the
  row and assembly injections and asserts the two fields are **distinct** and neither is the bare
  class name.
- `test_collector_typed_candidate_support_error_field_is_unchanged` (`:970-987`): injects a typed
  `CandidateSupportError("index_row_over_cap", "typed_control_field")` at the same raw site and
  asserts the typed reason/field pair is preserved verbatim.
- **Remediation regression** `test_collector_raw_failure_field_canonicalizes_custom_exception_subclass`
  (`:990-1014`): builds `leaky_error = type("C:/sentinel", (ValueError,), {})` — a custom
  `ValueError` subclass whose class name is path-like — and injects it at the raw row site
  `_alternative_index`. It asserts the field **starts with the canonical `ValueError:`**, contains
  no `C:/sentinel`, no `"/"` or `"\\"`, while retaining the module/function (`._leaky_row_site`),
  positive source line and `@row0` admission, and stays within the byte bound. Pre-repair this
  test fails: the field began `C:/sentinel:` and `field.split(":")` did not yield the three
  expected components.
- **Sibling-arm regression** `test_build_capture_raw_field_canonicalizes_custom_exception_subclass`
  (`:1016-1038`): injects the same path-named custom `ValueError` into `_assemble_complete_support`
  reached through `build_complete_support_capture` and asserts its refusal field is exactly
  `"ValueError"` with no path sentinel, keeping the refusal atomic and valid.
- Every raw-field test asserts the sentinel (`str(error)`) and any `"/"` or `"\\"` path separator
  is absent and that `0 < _serialized_string_bytes(field) <= REFUSAL_FIELD_MAX_BYTES`, which is
  the concrete no-data-leak and boundedness control.

## 4. Checks run (focused only)

| Check | Command | Result |
|---|---|---|
| Byte-compile | `.venv\Scripts\python.exe -m py_compile src/mc_photogrammetry/pipeline/candidate_support_provenance_v2.py tests/test_candidate_support_provenance_v2_preflight.py` | exit 0 |
| Focused suite | `.venv\Scripts\python.exe -m pytest tests/test_candidate_support_provenance_v2_preflight.py -o addopts='' -q --no-header` | **61 passed**, exit 0 (54 pre-existing + 5 original + 2 remediation) |
| Raw-failure regressions only | `.venv\Scripts\python.exe -m pytest tests/test_candidate_support_provenance_v2_preflight.py -o addopts='' -q --no-header -k "collector_raw or typed_candidate or canonicalizes_custom"` | **7 passed, 54 deselected**, exit 0 |

`addopts=''` only disables the repository's default `-q` so the summary line is visible; no
existing test was modified or deselected. No full suite, runner invocation, capture, accuracy
scoring or performance measurement was performed.

## 5. Hashes

| File | SHA-256 | Bytes |
|---|---|---|
| `src/mc_photogrammetry/pipeline/candidate_support_provenance_v2.py` | `56A5D3B949028A9C0E363E449805B55368217280EE8A57DFC12A57804B5F28F7` | 120,413 |
| `tests/test_candidate_support_provenance_v2_preflight.py` | `1670CE39A77613CE67F231EA35F74169FFB5D479CF0B400C24B4D485E646E75F` | 102,475 |

These are the post-remediation hashes and supersede the pre-remediation pins
(`D71D5FEC…` source / `54A627A9…` tests) that Luna reviewed in the HOLD. Both files are untracked
in the working tree (`??`); no unrelated path was modified or inspected. The pinned V1 module
(`candidate_support_provenance.py`) and the graph module (`image_derived_graph.py`) were not
touched, so the existing `test_v1_source_is_unchanged_and_graph_is_the_pinned_additive_seam` pin
still holds (it passed in the suite above). This report's own hash is not self-asserted here; an
operator can pin it independently.

## 6. Scope limits, retained HOLD and disposition

- **Diagnostic-only.** The change alters no geometry, matcher, solver, capture schema/status or
  digest. It only enriches the refusal `field` for a caught raw exception; `reason` stays
  `universe_identity_mismatch`, the capture stays atomic (zero rows, no pages) and typed
  `CandidateSupportError` behavior is unchanged.
- **Luna HOLD addressed.** The exception-name leak is closed at both emission points: the field
  now carries a fixed canonical built-in category selected by `isinstance` against the explicit
  caught bases, never `type(error).__name__`. `test_collector_raw_failure_field_canonicalizes_custom_exception_subclass`
  proves a path-named custom `ValueError` subclass cannot reach the field while localization and
  the byte bound are retained, and the sibling-arm regression covers the other emission point.
- **Source-audit HOLD retained.** `DS-R4-V2-COLLECTOR-VALUEERROR-TRACE` remains on HOLD per Luna's
  review. This task implements the localization seam but does **not** supply the corrected
  operation-to-row map that review asked for; that corrected map still requires separate review,
  and this report does not claim to close the HOLD.
- **Separate follow-up (not modified here, per task):** `collect_route_b_diagnostic_support`
  swallows a `_match_order_identity` failure and substitutes `None`
  (`candidate_support_provenance_v2.py:2131-2133`), silently dropping the kept-subset identity
  binding. Its semantics are untouched by this task; the existing trace report already recommends
  recording a bounded identity operation code (or refusing instead of continuing) as a separate
  lead contract decision.
- **No rerun proposed or performed.** Localizing the real R2 failure site still requires the
  separate lead-authorized real invocation. A localized field from such a run would read, for
  example, `ValueError:mc_photogrammetry.reconstruction.plane_graph._canonical_reading_index:<line>@row<position>`
  for the traced `_alternative_index`/`_delta` path (mechanism expectation, not a measured value).
- **Stopped for Luna re-review.** No accuracy, downstream-certification or R2-root-cause claim is
  made.
