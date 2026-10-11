# DS-R4-V2-PREFLIGHT-IMPL-2026-10-04

Task IDs: `DS-R4-V2-PREFLIGHT-IMPL` (original implementation),
`DS-R4-V2-BOUNDS-REMEDIATION` (first Luna HOLD),
`DS-R4-V2-ROOT-CAPS-REMEDIATION` (second Luna HOLD),
`DS-R4-V2-CONTENT-REPLAY` (third Luna HOLD),
`DS-R4-V2-PLAN-VALIDATION` (fourth Luna HOLD),
`DS-R4-V2-PLAN-LIVE-FIELD-VALIDATION` (fifth Luna HOLD, ACCEPT) and
`DS-R4-V2-ROUTEB-DIAGNOSTIC-WIRING` (this revision). Owner:
DeepSeek (Sol-equivalent CLI worker). Date: 2026-10-04.

Status: **Fifth Luna review ACCEPT; the lead then accepted a fixed-run
1,928-row Route B cap profile and this revision adds only the additive,
default-off diagnostic wiring that collects the complete V2 record on that
path. Stopped for Luna/lead review before any real capture.** This report
records evidence only. It does **not** approve a general production cap,
authorize a real capture, or claim any capture/evaluator/truth result.

> **HOLD notice.** Luna reviewed the V2 preflight five times and returned HOLD
> each time. Round 1: the synthetic ceiling was not a genuine upper bound because
> index/root/ingress fields were unbounded and the sidecar maximum was a finite
> shape sample. Round 2: the transform size tuples were only int64-bounded (not
> dimension-bounded) and their arity was unchecked, refused-root `reason`/`field`
> could be unbounded, the synthetic sidecar entry used admitted index 0 instead of
> the maximum admitted index, and a directly constructed capture could bypass the
> page-byte caps because the root did not bind the byte-cap plan and
> `validate_complete_support_capture` had no declared caps to enforce. Round 3:
> `validate_complete_support_capture` compared cached page hashes to the root
> bindings but never recomputed a page digest from current content, and it never
> replayed a sidecar entry's payload/array digests from the current arrays, so an
> `object.__setattr__` tamper could ride stale hashes; refusals also silently
> dropped the producing cap plan. Round 4: the cap plan's own
> eight values were never re-validated at replay time, so a frozen plan mutated
> with `object.__setattr__` (for example the four sidecar caps set to 0) and
> re-signed with fresh plan/root digests still validated; and a **planless**
> refusal was accepted for any capture-refusal reason, so
> `refused_complete_support("row_cap_exceeded", "row_count")` minted an
> unattributable planless refusal. Round 5 (this revision):
> `_cap_plan_field_over_cap` validated only the **overridable `as_record()`
> mapping**, while runtime enforcement and the cap-plan digest read the **live
> attributes**, so a `CompleteSupportCapPlanV1` subclass whose `as_record()`
> returned a default-valid record while its real sidecar caps were zero passed
> construction, `_resolve_plan` and a fully re-signed empty-sidecar capture
> replay. **All measurements and proposed caps from
> every earlier revision are withdrawn as bounds.** They are not accepted and
> must not be used to approve a cap or to authorize a capture. The measurements in
> section 6 below are a new bounded-model preflight; they too are synthetic,
> unapproved and subject to the same lead/Luna review.

> **Revision 2026-10-04c (post-acceptance, `DS-R4-V2-ROUTEB-DIAGNOSTIC-WIRING`).**
> Luna ACCEPTed round 5, and the lead then accepted the exact fixed-run profile
> in section 10 for the frozen 1,928-row ordinary-image Route B diagnostic only.
> That profile is **not** a general production cap and changes no shipped
> default; the section 6-7 measurements remain synthetic ceiling evidence. This
> revision adds only the additive, default-off `complete_support_sink` graph seam
> and `collect_route_b_diagnostic_support`, and authorizes no capture on its own.

## 0. Scope, authority and limits

- Authority: the accepted design
  `docs/reports/DS-R4-COMPLETE-SUPPORT-DESIGN-2026-10-04.md` (Luna ACCEPT for
  implementation preflight only), the Lead assignment `DS-R4-V2-PREFLIGHT-IMPL`,
  the Lead hold `DS-R4-V2-BOUNDS-REMEDIATION`, the Lead hold #2
  `DS-R4-V2-ROOT-CAPS-REMEDIATION`, the Lead assignment/hold #3
  `DS-R4-V2-CONTENT-REPLAY`, the Lead hold #4 `DS-R4-V2-PLAN-VALIDATION`, the
  Lead hold #5 `DS-R4-V2-PLAN-LIVE-FIELD-VALIDATION`, and the Lead assignment
  `DS-R4-V2-ROUTEB-DIAGNOSTIC-WIRING`.
- Owned paths only: `src/mc_photogrammetry/pipeline/candidate_support_provenance_v2.py`,
  `src/mc_photogrammetry/pipeline/image_derived_graph.py` (the single additive,
  default-off `complete_support_sink` seam), `tests/test_candidate_support_provenance_v2_preflight.py`,
  this report, and `AGENT_COMMS.md`. V1 (`candidate_support_provenance.py`),
  evaluator, fixtures, schemas, handoff and all production decisions are
  untouched; V1 remains byte-for-byte pinned.
- The implementation is **additive and default-off**. It is one new sibling
  module plus one default-`None` keyword on the internal graph deriver; the
  public producer and the production orchestration entry never expose the sink,
  so the default graph path allocates and collects nothing. There is no capture,
  no evaluator/truth/camera/mask/image read. The preflight is pure synthetic
  arithmetic over declared synthetic shapes.
- All byte numbers and proposed caps below are **synthetic measurements** at
  **declared contract bounds**, not measurements of the real 1,928-candidate
  fixture. They are **unapproved**. No real capture is authorized and no cap fit
  on real data is claimed in either direction.

## 1. Withdrawn claims (why each HOLD was placed)

### 1a. Round 1 (Luna, `DS-R4-V2-BOUNDS-REMEDIATION`)

The original preflight failed to bound runtime-legal records; the following are
**withdrawn**:

1. `index_row_worst_case_bytes = 1,063` was **not a bound**: `left`/`right` and
   the numeric fields had no length/range limits.
2. `root_worst_case_bytes = 233,414` was **not a bound**: the preflight used
   `ingress=()` while runtime accepted an arbitrary ingress tuple.
3. `sidecar_entry_worst_case_bytes = 115,818` was **not a bound**: the sidecar
   maximization sampled a fixed finite set of grid shapes while `source_shape`
   dimensions are serialized outside the raw-array accounting.
4. The page counts and cap proposals derived from (1)-(3) are withdrawn with them.
5. `expected_row_count` was optional, so 1,929 rows could pass the 4,096 ceiling.
6. Candidates/dispositions parallelism was not re-validated before `zip`.

### 1b. Round 2 (Luna, `DS-R4-V2-ROOT-CAPS-REMEDIATION`)

The bounded-model revision still did not prove every serialized path bounded;
the following measured claims are **withdrawn**:

1. `index_page_worst_case_bytes = 262,115` and `root_worst_case_bytes = 508,413`
   did **not** bound a legal pin whose `FrameTransform.source_size`,
   `crop_origin` or `crop_size` is an arbitrarily long tuple of individually
   in-bound integers. `FrameTransform` is a plain dataclass and
   `_ingress_record_over_cap` only bounded each integer, not the tuple arity; with
   an explicit two-value `output_size` a pin with a long `source_size`,
   `crop_origin` or `crop_size` still constructs. Source/crop/output dimensions
   were additionally allowed up to `INGRESS_MAX_ABS_INT` (19 digits) while the
   synthetic record used only `INGRESS_MAX_SHAPE_DIM` (10 digits), so even a
   two-value size tuple could serialize wider than measured.
2. `root_worst_case_bytes` did **not** bound a refused root: refused
   `reason`/`field` were unbounded strings, and no maximal refused-root size was
   measured.
3. `sidecar_entry_worst_case_bytes = 230,536` did **not** bound a real sidecar
   entry: the synthetic entry used `admitted_row_index = 0`, so the longest
   admitted index (`UNIVERSE_MAX_ROWS - 1 = 4,095`, four digits instead of one)
   was not included in the entry, page or aggregate measurement.
4. The measured page/entry/aggregate/root sizes were **not enforced by
   validation**: the page dataclasses and `CandidateSupportCaptureRootV2` did not
   carry or enforce the byte-cap plan, so a directly constructed
   `CompleteSupportCaptureV2` or a tampered root could bypass every byte cap.
5. The page counts and cap proposals derived from (1)-(4) are withdrawn with them.

## 2. Root/caps remediation implemented this round (`DS-R4-V2-ROOT-CAPS-REMEDIATION`)

All changes are confined to the V2 sibling module and its focused test module.

1. **Transform tuple arity and dimension bounds.**
   `_is_dimension_pair` requires `FrameTransform.source_size`, `crop_size` and
   `output_size` to be exactly two positive integers in
   `[1, INGRESS_MAX_SHAPE_DIM]`; `_is_signed_pair` requires `crop_origin` to be
   exactly two integers in `[-INGRESS_MAX_ABS_INT, INGRESS_MAX_ABS_INT]`. A
   `decoded_shape` check (`_decoded_shape_over_cap`) mirrors the V1 `_shape`
   contract (two or three positive dimensions, third fixed at 3). A violation is
   an atomic `ingress_over_cap` refusal. The synthetic ingress record is now built
   at the exact enforced maximum: the three-dimensional decoded shape, all size
   tuples at `INGRESS_MAX_SHAPE_DIM`, and `crop_origin` at `-INGRESS_MAX_ABS_INT`.
2. **Bounded refusal root.** `REFUSAL_FIELD_MAX_BYTES = 256`; a refused root now
   requires `reason in CAPTURE_REFUSAL_REASONS` and a nonempty field whose
   canonical (ensure_ascii) width is at most that bound (and `cap_plan`/`cap_plan_sha256`
   absent). `refused_complete_support` sanitizes an unknown reason to
   `universe_identity_mismatch` and a missing/non-string/over-long field to a
   deterministic bounded value, so an internal refusal stays atomic. The preflight
   measures `refused_root_worst_case_bytes` (longest reason plus a
   bound-width field) and gates it against the root cap.
3. **Maximal sidecar entry index.** `_synthetic_sidecar_entry(raw_cap,
   admitted_row_index)` now takes the index and the preflight builds it at
   `universe_max_rows - 1`, so entry, page and aggregate measurements include the
   longest admitted-index decimal form.
4. **The declared cap plan is persisted, hashed and enforced.**
   `CandidateSupportCaptureRootV2` carries `cap_plan: CompleteSupportCapPlanV1`
   (all eight caps: index/digest/sidecar page caps, raw and encoded entry caps,
   aggregate cap, root cap, universe cap) plus `cap_plan_sha256`
   (`CAP_PLAN_DIGEST_DOMAIN`); both are inside the root `record_sha256` payload.
   `_assemble_complete_support` writes exactly the resolved plan into the root.
   `validate_complete_support_capture` independently requires a complete root to
   carry a plan, replays its digest, checks `declared_universe_cap ==
   plan.universe_max_rows`, `row_count <= plan.universe_max_rows`, the root's own
   canonical bytes against `plan.root_max_bytes`, every index/digest/sidecar page
   against its declared page cap, every sidecar entry against both the raw and
   encoded entry caps, and the sidecar aggregate against the aggregate cap. The
   synthetic root now binds the full cap table, so `root_worst_case_bytes`
   includes it.

Row-local sidecar overflow remains row-local (`support_payload_over_cap` keeps the
capture `complete`); index-row, ingress, page, aggregate and root failures remain
atomic zero-row refusals with no pages.

## 2A. Content replay and refusal semantics (`DS-R4-V2-CONTENT-REPLAY`)

This revision answers Luna's third HOLD (round 3 in the notice above). All source
changes are confined to the same two owned files; this report and `AGENT_COMMS.md`
are updated with the result as the assignment requires.

1. **Every page digest is recomputed from current content.**
   `validate_complete_support_capture` no longer trusts the cached page hash. For
   every index page it recomputes
   `_digest(INDEX_PAGE_DIGEST_DOMAIN, _index_page_payload(page, include_digest=False))`
   from the page's current rows, for every digest page it recomputes
   `DIGEST_PAGE_DIGEST_DOMAIN` from the current ordered values, and for every
   sidecar page it recomputes `SIDECAR_PAGE_DIGEST_DOMAIN` from the current
   entries. Each recomputed value must equal both the page's cached
   `page_sha256` and the root binding, so a tampered row field, a replaced row
   tuple, a changed digest value or a rewritten cached page hash fails closed
   (`_index_page_content_digest`, `_digest_page_content_digest`,
   `_sidecar_page_content_digest`). Cached `digest_count`/`entry_count` values are
   also re-bound to `len(...)`.
2. **Every sidecar entry is replayed from its current arrays.**
   `_sidecar_entry_content_mismatch` re-checks the payload type, the
   `source_shape`/`target_shape` arity, the logical-pair shape/dtype/read-only
   contract, the warp arrays' dtype/read-only contract and their shape equality
   with `target_shape`, the declared shape/element bounds, `overlap_cells ==
   len(logical_pairs)`, each component array's SHA-256 recomputed from
   `tobytes(order="C")`, and the payload's canonical digest recomputed from a fresh
   `payload.to_record()` with the cached digest removed. The recomputed payload
   digest must match both the entry's cached `payload_sha256` and the payload's own
   cached digest, and the existing entry/index-row binding then carries it to
   `row.support_payload_sha256`. A same-shape read-only array substituted with
   `object.__setattr__`, or a stale cached component digest, therefore cannot pass.
   Frozen dataclasses and cached hashes are explicitly **not** treated as tamper
   protection.
3. **Refusals name the plan that produced them.**
   `CandidateSupportCaptureRootV2` now permits a refused root to carry
   `cap_plan` + `cap_plan_sha256` (`CAP_PLAN_DIGEST_DOMAIN`), both serialized in
   the refused payload. `refused_complete_support(reason, field, plan)` preserves a
   resolved plan; `_resolve_plan` returns the caller's plan, the default plan, or
   `None`; `build_complete_support_capture` and `collect_complete_candidate_support`
   pass the resolved plan to every refusal they produce. A planned refusal is
   bounded by the fixed `ROOT_MAX_BYTES` refusal bound; a planless refusal (plan
   resolution itself failed, or a non-plan object was supplied) is separately
   bounded by `PLANLESS_REFUSAL_ROOT_MAX_BYTES = 2,048` and uses the explicit new
   capture-level reason `cap_configuration_invalid` with field `cap_plan`. The
   validator replays the refused plan digest and re-measures the refused record
   against the matching bound for both branches.
4. **The cap plan's own values are bounded.**
   `CompleteSupportCapPlanV1` now rejects any cap outside
   `[1, CAP_PLAN_MAX_ABS_INT]` (`2^63 - 1`). That makes the plan's canonical width
   finite, so the maximal planned-refusal root has a measurable ceiling and a
   directly constructed plan cannot smuggle an unbounded decimal into a refusal.
5. **Preflight measures both refusal branches.**
   `refused_root_worst_case_bytes` is now the maximal **planned** refusal (longest
   reason, a bound-width field and a plan at `CAP_PLAN_MAX_ABS_INT` in every
   field), gated against `ROOT_MAX_BYTES`; the new
   `planless_refused_root_worst_case_bytes` is the maximal planless refusal, gated
   against `PLANLESS_REFUSAL_ROOT_MAX_BYTES`. The `ROOT_MAX_BYTES` proposal
   considers both.

## 2B. Plan-value replay and the fixed planless pair (`DS-R4-V2-PLAN-VALIDATION`)

This revision answers Luna's fourth HOLD (round 4 in the notice above). All source
changes are confined to the same two owned files; this report and `AGENT_COMMS.md`
are updated with the result as the assignment requires.

1. **The cap plan's eight values are re-validated at replay time.**
   `_cap_plan_field_over_cap(plan)` returns the first invalid plan field (or `""`),
   independently of construction. It requires an instance of
   `CompleteSupportCapPlanV1` whose canonical record carries **exactly**
   `CAP_PLAN_FIELDS` (no missing or extra serialized field) and whose every value
   is an exact positive `int` at most `CAP_PLAN_MAX_ABS_INT`. The check is called
   from `CompleteSupportCapPlanV1.__post_init__` (deduplicated through the shared
   `CAP_PLAN_FIELDS` tuple), from `CandidateSupportCaptureRootV2.__post_init__`
   for both complete and planned-refused roots, from `_resolve_plan`, and from
   `validate_complete_support_capture` for both branches. A frozen plan mutated
   with `object.__setattr__` and re-signed with fresh `cap_plan_sha256` **and**
   `record_sha256` is therefore rejected, because the validator re-checks the
   values rather than trusting the hashes it just replayed.
2. **A planless refusal is exactly the fixed configuration-invalid pair.**
   `PLANLESS_REFUSAL_REASON = "cap_configuration_invalid"` and
   `PLANLESS_REFUSAL_FIELD = "cap_plan"` are now declared constants. A refused root
   with `cap_plan is None` is rejected at construction and at validation unless its
   reason/field are exactly that pair, even when its own `record_sha256` is
   internally consistent. `refused_complete_support` therefore coerces any
   planless request (no plan, or a value-invalid plan) to that fixed pair, so an
   ordinary refusal code can never be emitted without the plan that produced it;
   the coercion keeps an internal refusal atomic rather than raising out of the
   builder. Planned refusals are unchanged: they still carry and hash their
   resolved plan, keep the longest legal reason and a bounded field, and validate.
   `_resolve_plan` likewise treats a value-invalid frozen plan as a
   plan-resolution failure and returns `None`, so
   `build_complete_support_capture`/`collect_complete_candidate_support` refuse
   planless instead of assembling under a tampered plan.
3. **Preflight measures the legal planless maximum.**
   `planless_refused_root_worst_case_bytes` now builds the maximal **legal**
   planless root, i.e. the fixed `cap_configuration_invalid`/"cap_plan" pair, so
   the measured bound is the true maximum of a legal planless refusal rather than
   an over-estimate built from an ordinary reason code. The planned-refusal bound
   is unchanged (longest reason, bound-width field, maximal plan).

Row-local sidecar overflow remains row-local; index-row, ingress, page, aggregate
and root failures remain atomic zero-row refusals with no pages; a planless
refusal is reserved for plan-resolution failure only.

## 2C. Live cap-attribute validation and the exact plan class (`DS-R4-V2-PLAN-LIVE-FIELD-VALIDATION`)

This revision answers Luna's fifth HOLD (round 5 in the notice above). All source
changes are confined to the same two owned files; this report and `AGENT_COMMS.md`
are updated with the result as the assignment requires.

1. **The live attributes are validated, not the overridable `as_record()` map.**
   `_cap_plan_field_over_cap` now requires `type(plan) is CompleteSupportCapPlanV1`
   (an exact class check, not `isinstance`) and reads every cap with
   `getattr(plan, name)`, applying the same exact-positive-`int`/`<=
   CAP_PLAN_MAX_ABS_INT` rule to that live value. This is the value runtime
   enforcement uses, so a subclass that overrides `as_record()` to return a
   default-valid mapping while its real sidecar caps are zero is now rejected
   instead of passing construction, `_resolve_plan` and replay.
2. **The serialized record must still agree with the live attributes.**
   After the live check, the canonical record must be a `dict` carrying exactly
   `CAP_PLAN_FIELDS` (no missing or extra serialized field) and every
   `record[name]` must equal `getattr(plan, name)`. The cap-plan digest is derived
   from `as_record()`, so this agreement check keeps the digest bound to the same
   values the predicate and consumers enforce; a valid-looking but disagreeing
   mapping now fails closed.
3. **A fully re-signed spoof cannot validate.** The new control reproduces Luna's
   repro: an empty-sidecar complete capture whose root `cap_plan` is replaced by
   the subclass, with `cap_plan_sha256` **and** `record_sha256` recomputed from the
   spoofed mapping, is rejected; the same spoof on a planned refusal is rejected;
   `_cap_plan_field_over_cap` returns `"plan"` and `_resolve_plan` returns `None`;
   a request carrying the spoof refuses planless and a refused root cannot carry
   the spoof directly; and untouched builder-produced captures still validate.

The change is contract-level and additive: no serialized schema, field set, record
shape or cap value changed, and no cap is approved by it.

## 3. Added/changed files and hashes

| # | Path | Bytes | SHA-256 |
|---|------|-------|---------|
| 1 | `src/mc_photogrammetry/pipeline/candidate_support_provenance_v2.py` | 109,414 | `6E29F4D60008E550EDBC0B6F20F71729E4F3220BA1A791DB52EECC8A9244C06D` |
| 2 | `tests/test_candidate_support_provenance_v2_preflight.py` | 79,401 | `A6943911005C8A2E2DF9EF210EA6502F380EAD5639BF984E0C502A0BCCD875C8` |

Both files are currently untracked (`git status` -> `??`). The V1 and graph
modules were re-hashed to confirm the additive sibling left them byte-for-byte
unchanged, matching the pins asserted in the test module:

| Preserved source | Bytes | SHA-256 |
|------------------|-------|---------|
| `src/mc_photogrammetry/pipeline/candidate_support_provenance.py` (V1) | 42,223 | `A8BD3102652720513ED85E2D3B7E2C81A79A01A7621C41238B09F6E741604562` |
| `src/mc_photogrammetry/pipeline/image_derived_graph.py` | 266,928 | `2F94B9E3125F16E493C11647A91882BF1C61E1176CA7A9B93BD3AB28F511C5E2` |

## 4. Declared bounds enforced at runtime and measured by the preflight

| Declared bound | Value | Applied to |
|----------------|-------|-----------|
| `INDEX_ROW_MAX_FRAME_ID_BYTES` | 1,024 | canonical byte width of `left`/`right` |
| `INDEX_ROW_MAX_ABS_INT` | 9,223,372,036,854,775,807 | every index-row integer field |
| `INGRESS_MAX_RECORDS` | 16 | `CandidateSupportCaptureRootV2.ingress` length |
| `INGRESS_MAX_FRAME_ID_BYTES` | 1,024 | ingress `frame_id` canonical width |
| `INGRESS_MAX_SOURCE_LABEL_BYTES` | 1,024 | ingress `source_label` canonical width |
| `INGRESS_MAX_ABS_INT` | 9,223,372,036,854,775,807 | `byte_length` and signed `crop_origin` coordinates |
| `INGRESS_MAX_SHAPE_DIM` | 2,147,483,647 | `decoded_shape` dimensions and `source_size`/`crop_size`/`output_size` dimensions |
| `PAYLOAD_MAX_SHAPE_DIM` | 2,147,483,647 | `source_shape`/`target_shape` dimensions |
| `PAYLOAD_MAX_ABS_ELEMENT` | 2,147,483,647 | every serialized array element |
| `REFUSAL_FIELD_MAX_BYTES` | 256 | canonical width of a refused root's `field`; `reason` is the bounded enum |
| `CAP_PLAN_MAX_ABS_INT` | 9,223,372,036,854,775,807 | every `CompleteSupportCapPlanV1` value, so a plan and a planned refusal have a finite canonical width |
| `PLANLESS_REFUSAL_ROOT_MAX_BYTES` | 2,048 | canonical width of a planless refused root (plan resolution failed; no `cap_plan`) |

A legal pin therefore carries exactly two positive size dimensions (at most
10 digits each), exactly two signed `crop_origin` coordinates (the longest
canonical value is the 20-character `-9223372036854775807`), and a two- or
three-dimensional decoded shape. These are declared contract limits, not approved
caps and not measured fixture properties. A real fixture record that exceeds one
of them fails the capture closed and requires the lead to raise the bound
explicitly.

## 5. Verification commands and results

Environment: Python 3.10.11 (`.venv`), pytest 9.1.1.

```
.venv\Scripts\python.exe -m py_compile ^
  src/mc_photogrammetry/pipeline/candidate_support_provenance_v2.py ^
  tests/test_candidate_support_provenance_v2_preflight.py
```
Result: exit 0.

```
.venv\Scripts\python.exe -m pytest ^
  tests/test_candidate_support_provenance.py ^
  tests/test_candidate_support_provenance_v2_preflight.py ^
  tests/test_image_derived_graph_same_axis_candidate_capture.py
```
Result: **113 passed in 5.35 s** (exit 0), broken down as:

| Module | Passed |
|--------|--------|
| `tests/test_candidate_support_provenance.py` (V1) | 46 |
| `tests/test_candidate_support_provenance_v2_preflight.py` (V2) | 48 |
| `tests/test_image_derived_graph_same_axis_candidate_capture.py` (same-axis) | 19 |

Round 3 (`DS-R4-V2-CONTENT-REPLAY`) added four adversarial controls to the V2
module (41 -> 45) and rewrote two:

1. `test_content_replay_rejects_index_and_digest_page_mutation` mutates an index
   row field behind the frozen cached page hash, reorders an index page's rows,
   mutates a digest value, rewrites a cached digest-page hash, and confirms an
   untouched builder-produced capture still replays;
2. `test_content_replay_rejects_sidecar_array_and_entry_mutation` swaps a
   same-shape read-only component array with `object.__setattr__`, stales a
   component digest, stales the payload digest, tampers the recorded
   `source_shape`, tampers `overlap_cells`, substitutes an entire same-shape
   payload, and confirms the untouched capture still replays;
3. `test_refusal_preserves_the_resolved_cap_plan_and_stays_bounded` proves a
   planned refusal carries and hashes its plan, stays inside `ROOT_MAX_BYTES`, and
   rejects a substituted plan or a stale plan digest;
4. `test_invalid_plan_resolution_yields_the_planless_refusal` proves a non-plan
   object yields the bounded planless `cap_configuration_invalid`/"cap_plan"
   refusal, while a resolved plan that merely fails a cap keeps its plan.

Round 4 (`DS-R4-V2-PLAN-VALIDATION`) added two adversarial controls
(45 -> 47) and rewrote two:

1. `test_cap_plan_fields_are_replay_validated_after_tampering` re-signs an
   otherwise-valid capture (with **no** sidecar entries) after setting each of the
   eight plan fields, in turn, to `0`, `-1`, `1.5`, `"1"` and
   `CAP_PLAN_MAX_ABS_INT + 1`, recomputes both `cap_plan_sha256` and
   `record_sha256`, asserts both are internally consistent, and requires
   validation to reject; it also re-signs an invalid plan on a planned refusal,
   proves `_resolve_plan` refuses it, and proves missing/extra serialized plan
   fields (via a subclass `as_record`) cannot be resolved or carried in a root.
2. `test_planless_refusal_is_restricted_to_the_fixed_configuration_pair` mutates a
   valid planless root to `row_cap_exceeded` and to an ordinary field, re-signs
   `record_sha256`, and requires rejection for every capture-level reason except
   the fixed pair, while an ordinary planned refusal with the same reason still
   validates.
   The rewritten `test_refused_root_reason_and_field_are_bounded` now checks the
   planned-refusal field bound, demonstrates that a planless request is coerced to
   the fixed pair, and rejects every other planless pair at construction; the
   rewritten `test_preflight_measures_the_maximal_refused_roots` measures the legal
   planless maximum.

Round 5 (`DS-R4-V2-PLAN-LIVE-FIELD-VALIDATION`, this revision) adds one adversarial
control (47 -> 48), `test_subclass_plan_as_record_spoof_cannot_hide_invalid_live_caps`:
it builds a `CompleteSupportCapPlanV1` subclass whose `as_record()` always returns
the default-valid mapping, sets the four live sidecar caps to `0` with
`object.__setattr__`, and proves the spoofed record looks valid while the live caps
do not; `_cap_plan_field_over_cap` returns `"plan"`, `_resolve_plan` returns `None`,
a request carrying the spoof refuses planless, a refused root cannot carry it
directly, a fully re-signed empty-sidecar complete capture and a re-signed planned
refusal are both rejected, and untouched builder captures still validate. A
separate in-process mutation control (old `as_record`-only predicate monkeypatched
back in, no file change) reproduced the fail-open: `_resolve_plan` accepted the
spoof and `validate_complete_support_capture` accepted the re-signed capture.

The nine round-2 adversarial controls (transform arity/dimensions, decoded shape,
measured ingress maximum, refused reason/field bounds, maximal refused root,
maximal sidecar index, plan persistence/hash, tampered plans and manually rebuilt
over-cap captures) remain, and the earlier controls (overlong IDs, huge
numeric/shape fields, excessive ingress, missing expected count,
candidates/dispositions mismatch, row-local oversized support, the `(64, 128)`
sidecar falsifier, longest digest name, maximal ingress plus full root bindings)
also remain.

No capture, inference, evaluator/truth/camera/mask/image read, or reconstruction
was performed.

## 6. Preflight measurements at the declared bounds (synthetic, unapproved)

`preflight_complete_support()` with the **source default** plan
(`CompleteSupportCapPlanV1()`), universe ceiling `UNIVERSE_MAX_ROWS = 4,096`:

Measured worst-case canonical sizes (bytes):

| Measurement | Value |
|-------------|-------|
| `index_row_worst_case_bytes` | 2,877 |
| `index_page_envelope_bytes` | 217 |
| `index_page_worst_case_bytes` | 262,115 |
| `digest_page_envelope_bytes` | 264 |
| `digest_page_worst_case_bytes` | 262,114 |
| `sidecar_entry_worst_case_bytes` | 230,539 |
| `sidecar_page_envelope_bytes` | 222 |
| `sidecar_page_worst_case_bytes` | 230,761 |
| `sidecar_aggregate_worst_case_bytes` | 945,197,056 |
| `ingress_record_worst_case_bytes` | 2,656 |
| `ingress_tuple_worst_case_bytes` | 42,513 |
| `refused_root_worst_case_bytes` | 841 |
| `planless_refused_root_worst_case_bytes` | 176 |
| `root_worst_case_bytes` | 508,793 |

Derived counts at the ceiling:

| Derived count | Value |
|---------------|-------|
| `max_rows_per_index_page` | 91 |
| `index_page_count` | 46 |
| `max_digests_per_digest_page` | 3,688 |
| `digest_page_count_per_section` | 2 |
| `digest_page_count` | 12 |
| `max_entries_per_sidecar_page` | 1 |
| `sidecar_page_count` | 4,096 |

Default preflight outcome:

- `preflight_status = refused`
- `preflight_refusal_reason = synthetic_size_over_cap`
- `detail = exceeded=sidecar_entry,root,sidecar_aggregate`
- `cap_invariant_ok = true`

i.e. under the source defaults the measured sidecar entry exceeds the placeholder
`SIDECAR_ENTRY_MAX_BYTES = 131,072`, the root (now carrying the cap plan) exceeds
the placeholder `ROOT_MAX_BYTES = 262,144`, and the aggregate exceeds the
placeholder `SIDECAR_AGGREGATE_MAX_BYTES = 8,388,608`. Both refusal branches are
inside their fixed bounds: the maximal planned refusal (841 B) is well inside
`ROOT_MAX_BYTES`, and the maximal planless refusal (176 B, now the fixed
`cap_configuration_invalid`/"cap_plan" pair rather than an ordinary reason code)
is well inside `PLANLESS_REFUSAL_ROOT_MAX_BYTES = 2,048`. Page and entry layouts
remain internally consistent (`cap_invariant_ok`).

## 7. Proposed caps from the bounded preflight (synthetic, unapproved)

| Proposed cap | Value | Basis |
|--------------|-------|-------|
| `INDEX_PAGE_MAX_BYTES` | 262,144 | ceiling of 262,115 to 4 KiB |
| `DIGEST_PAGE_MAX_BYTES` | 262,144 | ceiling of 262,114 to 4 KiB |
| `SIDECAR_PAGE_MAX_BYTES` | 237,568 | ceiling of max(230,761, 233,472 + 222) to 4 KiB |
| `SIDECAR_ENTRY_MAX_RAW_BYTES` | 131,072 | unchanged input gate |
| `SIDECAR_ENTRY_MAX_BYTES` | 233,472 | ceiling of 230,539 to 4 KiB |
| `SIDECAR_AGGREGATE_MAX_BYTES` | 945,815,552 | ceiling of 945,197,056 to 1 MiB |
| `ROOT_MAX_BYTES` | 512,000 | ceiling of max(508,793, 841, 176, 262,144) to 4 KiB |
| `UNIVERSE_MAX_ROWS` | 4,096 | declared ceiling |

Feeding these proposed caps back as the plan makes the synthetic preflight return
`preflight_status = ok`, `preflight_refusal_reason = ""` and `cap_invariant_ok =
true` (`root_worst_case_bytes = 508,795` against the proposed 512,000-byte root
cap; covered by the focused tests). This is a synthetic self-consistency check
only.

**These values are synthetic and unapproved.** They are proposals for the lead,
not certified caps, and they say nothing yet about the real 1,928-candidate
fixture. In particular the 945 MB aggregate is a declared-worst-case ceiling in
which all 4,096 rows carry a maximal support payload; the frozen fixture has
1,928 rows and its real support presence is unknown.

## 8. Remaining lead review

- Confirm the declared bounds in section 4 are acceptable for real frame IDs,
  ingress records (transform arity/dimensions) and payload shapes; raising any
  bound requires re-running the preflight, since the ceiling is derived from them.
- Confirm the additive capture-level reason `cap_configuration_invalid` is
  reserved **exactly** for the planless refusal branch, whose field must be
  `cap_plan` and whose `cap_plan`/`cap_plan_sha256` must be absent; every other
  refusal must carry and hash the plan that produced it. Confirm the two fixed
  refusal bounds (`ROOT_MAX_BYTES` for a planned refusal,
  `PLANLESS_REFUSAL_ROOT_MAX_BYTES` for a planless one) are accepted.
  `CAP_PLAN_MAX_ABS_INT` bounds every plan value, so a real plan above it now
  fails closed and needs the lead to raise the bound; the plan's eight values are
  also re-validated at replay time, so a frozen plan mutated afterwards (and
  re-signed) is rejected.
- Validate and, if accepted, set the caps in section 7.
- Decide whether the declared-worst-case aggregate (about 945 MB at 4,096 rows)
  is acceptable or whether the raw-element/dimension bounds should be tightened.
- No real capture, inference/evaluator read, R1 rerun, or production enablement
  is authorized until the lead accepts the bounded preflight and the caps.

## 9. Provenance and limitations

- The preflight reads no candidate, witness, payload, image, truth, camera, mask
  or evaluator material; it measures canonical encodings of declared synthetic
  shapes only.
- The measured numbers in sections 6-7 are re-derived from the bounded model in
  this revision and are not measured against any fixture.
- The declared bounds and the sidecar maximization argument are contract-level
  claims about this module's own records; they are not claims about the frozen
  fixture's real field widths.
- The worst-case sidecar assumes a payload may carry the most-negative
  declared-bounded element in every warp cell, now at the last admitted index;
  this is a conservative type-level bound, not an observed fixture value.
- All earlier revision numbers, including the round-1 `(1, cells)` sidecar
  maximum and the round-2 transform/refusal/index-0 measurements, remain
  withdrawn and are not bounds.
- `validate_complete_support_capture` now replays every index/digest/sidecar page
  digest and every sidecar entry payload/component-array digest from the current
  in-memory content, so a tampered or manually reconstructed capture cannot ride a
  stale cached hash; frozen dataclasses and cached digests are not tamper
  protection.
- A refusal produced after a plan was resolved preserves and hashes that plan
  inside the refused root; only a plan that could not be resolved at all (missing,
  non-plan or value-invalid) yields the separately bounded planless refusal, and
  that refusal is forced to the fixed `cap_configuration_invalid`/"cap_plan" pair.
  No ordinary refusal silently omits its producing plan.
- The cap plan's eight values are re-validated from the current object at
  construction, at resolution and at capture replay (exact positive bounded ints,
  exactly the declared serialized field set), so a frozen plan mutated with
  `object.__setattr__` and re-signed with fresh plan/root digests is rejected; the
  plan digest alone is not treated as tamper protection.
- The plan check validates the exact plan class and its **live attributes**, then
  requires the serialized record to agree with them, because runtime enforcement
  and the cap-plan digest use those attributes rather than the overridable
  `as_record()` mapping. A subclass that returns a valid-looking record while
  carrying invalid live caps is therefore rejected at construction, resolution and
  replay, even after both digests are recomputed from the spoofed mapping.
- The cap plan is bound inside the root record digest and replayed by
  `validate_complete_support_capture`, which also re-measures every persisted
  page and entry against it; no cap approval, report approval, capture
  authorization, or production enablement is claimed.
- Stopped for lead/Luna review.

## 10. Route B diagnostic wiring (`DS-R4-V2-ROUTEB-DIAGNOSTIC-WIRING`)

### 10.1 Accepted fixed-run profile

The lead accepted exactly this single-run profile (2026-10-04) for the frozen
1,928-row ordinary-image Route B diagnostic universe. It is exposed as
`ROUTE_B_DIAGNOSTIC_CAP_PLAN` / `ROUTE_B_DIAGNOSTIC_EXPECTED_ROW_COUNT`; it is a
fixed-run diagnostic profile, not a general production cap, and it changes no
shipped default or constant:

| Cap | Value |
|---|---|
| `index_page_max_bytes` | 262,144 |
| `digest_page_max_bytes` | 262,144 |
| `sidecar_page_max_bytes` | 237,568 |
| `sidecar_entry_max_raw_bytes` | 131,072 |
| `sidecar_entry_max_bytes` | 233,472 |
| `sidecar_aggregate_max_bytes` | 536,870,912 (512 MiB) |
| `root_max_bytes` | 512,000 |
| `universe_max_rows` | 1,928 |

`preflight_complete_support(ROUTE_B_DIAGNOSTIC_CAP_PLAN)` returns
`preflight_status = ok`, `preflight_refusal_reason = ""`, `cap_invariant_ok =
true`. Measured synthetic worst cases at this ceiling: index page 262,115 B,
digest page 137,154 B, sidecar entry 230,539 B, sidecar page 230,761 B, sidecar
aggregate 444,907,208 B (91,963,704 B below the 512 MiB cap), root 262,492 B
(below 512,000 B), maximal planned refusal 841 B and planless refusal 176 B.

### 10.2 Default-off graph seam

`image_derived_graph._derive_image_derived_graph` gains exactly one additive
keyword, `complete_support_sink=None`. When it is `None` - every default call,
the public producer and the production Route B orchestration included - nothing
is allocated or collected and the returned result/digest are byte-for-byte
unchanged. When it is attached and the component solver abstains, the seam calls
once, from material the ordinary path already produced (no second match, filter,
signature pass or solve):

`candidate_support_provenance_v2.collect_route_b_diagnostic_support(readings,
matches, signature_witnesses or (), pre_filter_rows or (),
raw_observations[0], ingress=<accepted-session ingress pins>,
failed_assignment_sha256=<same failed assignment the V1 seam carries>)`

The public `derive_image_derived_graph` wrapper is unchanged and never exposes
the sink.

### 10.3 Binding and atomic refusal

`collect_route_b_diagnostic_support` fixes the profile above and the exact
`expected_row_count = 1,928`. It additionally derives the kept-subset
`input_match_identity_sha256` independently from the solver's returned
post-filter match order, so a foreign, dropped or reordered kept match becomes an
atomic `universe_identity_mismatch` instead of a silently different digest. All
1,928 admitted rows are always in the index; a row without usable support is a
typed row-local `support_state=refused` inside a `complete` capture, never a
dropped row. Every count, order, identity or byte-cap failure refuses the whole
capture with `row_count = 0` and no pages, and a planned refusal preserves and
hashes the plan that produced it.

### 10.4 Focused checks (once after edits)

- `py_compile` on both changed Python files and the test module: exit 0.
- `pytest tests/test_candidate_support_provenance.py
  tests/test_candidate_support_provenance_v2_preflight.py
  tests/test_image_derived_graph_same_axis_candidate_capture.py`: **119 passed**
  (V1 46, V2 54, same-axis 19).
- `pytest tests/test_image_derived_graph.py
  tests/test_image_derived_graph_attribution.py
  tests/test_image_derived_graph_phase_capture.py
  tests/test_image_derived_graph_solver_input_bindings.py
  tests/test_image_derived_graph_solve_context.py
  tests/test_image_derived_graph_policy.py
  tests/test_candidate_support_ownership_harness.py`: **216 passed**.
- `pytest tests/test_image_derived_graph_stage_harness.py`: **44 passed**.
- V1 source SHA-256 is still
  `A8BD3102652720513ED85E2D3B7E2C81A79A01A7621C41238B09F6E741604562`
  (byte-for-byte unchanged). The graph module changes only by this additive
  seam; its new SHA-256 is pinned by the focused test alongside the V1 pin.
- New focused controls: the accepted profile is exact and preflights `ok`; the
  collector captures the full ordered 1,928-row universe with mixed
  present/refused rows and round-trips through canonical serialization and
  validation; short and over-cap universes and a foreign whole-universe identity
  refuse atomically; every profile byte cap (index/digest/sidecar page,
  aggregate, root) refuses atomically with zero pages; and the graph seam is
  default-off, output-neutral, carries the accepted session ingress pins, and
  delivers either the complete capture or an atomic refusal.

### 10.5 Hashes (SHA-256)

| Artifact | Bytes | SHA-256 |
|---|---|---|
| `candidate_support_provenance_v2.py` | 113,404 | `74040691ABC594FC72F010F2F4C700EBCDA719E69FBB4F83647570E54419D79E` |
| `image_derived_graph.py` | 269,254 | `32747546C934F8C218E43646CC995F310FF383E522C8A10F878E691F54954188` |
| `test_candidate_support_provenance_v2_preflight.py` | 94,405 | `2E732A6D9D0BF3B280B35D7E5642B5DCEC3C82114525FBEEECDF61EE81887336` |
| `candidate_support_provenance.py` (V1, unchanged) | 42,223 | `A8BD3102652720513ED85E2D3B7E2C81A79A01A7621C41238B09F6E741604562` |

### 10.6 Limitations

- All numbers remain synthetic; no candidate, witness, image, truth, camera,
  mask or evaluator material was read and no reconstruction or capture was run.
- The seam is reachable only through the internal
  `_derive_image_derived_graph`; the public producer and the Route B production
  orchestration do not expose it, so a real capture still requires a separate,
  reviewed diagnostic runner (not authorized here).
- The accepted profile is fixed to the frozen 1,928-row diagnostic universe
  only; it is not a general production cap and changes no default.
- Stopped for Luna/lead review before any real capture.
