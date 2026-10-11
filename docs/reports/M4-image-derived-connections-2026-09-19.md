# M4 image-derived connected-camera evidence adapter (2026-09-19)

Task: bounded production seam requested 2026-09-19 - consume an accepted
image-derived session and emit typed, digest-bound relative-camera evidence for
every unordered frame pair through the already-frozen joint matcher.
Worker: `deepseek-flash`, bounded implementation.
Status: **Second Sol-blocker repair complete (strict optional-finite matcher
diagnostics); py_compile, the repaired adapter suite (24 tests), the accepted M4
session suite, the project-store suite and the joint-matching suite all pass;
stopped for Sol/Astra re-review.**

Write scope used (exactly three routed paths):

- `src/mc_photogrammetry/pipeline/image_derived_connections.py`
- `tests/test_image_derived_connections.py`
- `docs/reports/M4-image-derived-connections-2026-09-19.md` (this file)

This repair round edited only those three files. No other module, test, script,
`AGENT_COMMS.md` or `AGENT_HANDOFF.md` file was created, edited or deleted.
`AGENT_COMMS.md` was **read only**. No git stage/commit/reset/clean was run. The
only side effect of the required `py_compile`/`pytest` runs is refreshed standard
`__pycache__` bytecode caches.

Artifacts:

| path | lines | SHA-256 |
| --- | --- | --- |
| `src/mc_photogrammetry/pipeline/image_derived_connections.py` | 1026 | `A2D166BF711D6CD77C169F044DA5CA28D00073E66E60B9BFF992572FB56BA211` |
| `tests/test_image_derived_connections.py` | 753 | `2CE0C2CF3F64A1FF9819C8397C0B6671709E79C7F7C7CFB8D811783A7B40CF6B` |

The report's own digest is not self-referential and is recorded by the lead in
the run record.

## 1. Objective and boundary

The accepted `pipeline.image_derived_session` seam establishes *per-view* facts
(yaw-modulo-90 pose, explicit intrinsics, immutable pixels, and the explicitly
unresolved position/scale/yaw-quadrant notes) and nothing between views. This
adapter is the narrow bridge from that session to the frozen DS-14 matcher: for
each unordered pair of the session's frames, delegate once to
`joint_matching.match_and_solve_pair` with the session's own `CameraEstimate`
values and immutable RGB/gray arrays, and report the matcher's answer as typed,
immutable, digest-bound evidence plus a deterministic graph.

The adapter **does not** compose absolute camera translations, infer or export
metric scale, produce bounds, scene transforms, block volumes or face
observations, export anything, expose CLI/UI, or accept manual metadata. Its only
imports are the accepted session module, the frozen matcher, `numpy`, and the
standard library; there is no reconstruction, export, CLI, dataset or project
dependency.

## 2. Matcher binding check (bind exactly, or stop)

`joint_matching.match_and_solve_pair(left_id, left_rgb, right_id, right_rgb, *,
left_camera, right_camera, usable_masks=None, settings=None, matcher=None,
gates=None)` was inspected and exercised before implementation:

- it accepts a `uint8` grayscale `(h, w)` **or** RGB `(h, w, 3)` frame
  (`correspondence._validate_frame`);
- it consumes `CameraEstimate` values carrying an explicit pose and intrinsics,
  which is exactly what the session stores;
- it returns a typed `JointPairResult` with `status`/`reason`, `solved`, `pose`
  (`baseline_direction`, `relative_rotation`), `right_yaw_class`, `matches`,
  `support`, `support_floor`, the keypoint/edge counters and `hypotheses`;
- it does not mutate its inputs (`detect_features` copies grayscale and
  `cv2.cvtColor` allocates for RGB).

A pre-implementation probe built a real session and called the matcher on its
bytes-backed read-only arrays: it returned a `JointPairResult`, and both session
arrays were still `writeable=False` afterwards. The contract binds exactly; no
stop condition was hit.

## 3. Public contract

### 3.1 `RelativeCameraEvidence` (immutable)

One record per unordered pair, in canonical order. It carries:

- `left_id`, `right_id` - the pair ids;
- `status` (`"solved"`/`"refused"`) and `reason` (`""` on a solve);
- `baseline_direction` - the matcher's **unit** world baseline direction, only
  when the matcher solved, stored verbatim (validated finite and unit within
  `1e-6`, never re-derived or re-scaled);
- `relative_rotation` (3x3 nested tuple) and `right_yaw_class`, only on a solve;
- `match_count`, `support`, `support_floor`, `raw_keypoints`, `candidate_edges`,
  `mutual_survivors` and the geometric diagnostics
  (`median_selected_residual_degrees`, `parallax_degrees`,
  `cheirality_fraction`, `singular_ratio`, `ambiguity_margin`);
- `hypotheses`: the matcher's ranked yaw-class hypotheses reduced to immutable
  `YawClassDiagnostic` records;
- `matches_sha256` - domain-separated SHA-256 over the selected one-to-one match
  array's rank, shape, dtype and C-order bytes;
- `evidence_sha256` - self-computing/verifying SHA-256 over the record;
- `session_sha256` - provenance binding to the accepted session.

Refusals pass the matcher's typed reason through verbatim (never renamed). A
solved result whose baseline is not a unit 3-vector is downgraded to a typed
`invalid_baseline_direction` refusal with no direction, rotation or yaw recorded.

### 3.2 `ImageDerivedConnectionGraph` (immutable)

- `status` `"ok"` with exactly one evidence record per unordered pair in the
  session's declared frame order, and the solved connected components over all
  frames (isolated frames as singletons); or
- `status` `"refused"` with a typed `reason` and **no** evidence and **no**
  components - never a partial result;
- `pair_count == C(frames, 2)`, `frame_ids`, `budgets`, `unresolved` (copied from
  the session), `scale_known` hard-frozen `False`, `graph_sha256` binding all of
  the above.

### 3.3 `ConnectionBudgets` (explicit, self-validating)

- `max_pairs` - the **pair budget**: the number of unordered pairs the adapter
  may evaluate, checked against `C(n, 2)` before any matcher call;
- `max_total_support` - the **total budget**: the sum of every evaluated pair's
  selected `support`, enforced while the evidence list is still private;
- a self-computed `budget_sha256`. `make_connection_budgets` requires both caps
  explicitly (no hidden defaults); `0` is legal and deliberately strict.

### 3.4 Typed refusals

Graph reasons (`CONNECTIONS_REFUSAL_REASONS`):

- `connections_no_pixels` - at least one frame carries no pixels (e.g. a session
  rebuilt from a record, where pixels are intentionally absent);
- `connections_session_mismatch` - a source binding or pixel digest does not
  replay;
- `connections_budget_exhausted` - the pair or total budget was exceeded.

Caller/contract errors raise `ImageDerivedConnectionError` with
`connections_input_invalid` (wrong type, non-callable matcher) or
`connections_matcher_mismatch` (a matcher return that violates its own
contract).

### 3.5 Repaired Sol blockers

Sol reproduced adapter defects across two repair rounds; all are repaired inside
this seam only.

1. **Reopenable frame buffers.** `_frame_pixel_problem` previously accepted any
   read-only `uint8` array whose shape and pixel digest bound, so a swapped
   numpy-owned array with identical bytes could pass every field check and still
   be reopened with `setflags(write=True)` and mutated under the matcher. It now
   also requires the accepted session module's own non-reopenable-base invariant
   (`_has_immutable_bytes_base`, imported rather than restated): the buffer's
   base chain must terminate at immutable `bytes`. A swapped numpy-owned buffer
   is refused as a typed `connections_session_mismatch` during source validation,
   before the pair budget and before any matcher call.
2. **Malformed hypothesis diagnostics.** Every matcher-derived hypothesis field
   is validated exactly instead of coerced: `yaw_class` must be an exact integer
   in `0..3`, `support`/`spatial_bins` non-negative integers, `passed` an exact
   boolean and `refusal` a string, and `hypotheses` must be a sequence. Any
   remaining construction/validation error around `_evidence_from_result` (the
   record/diagnostic constructors or numpy conversion of a matcher array) is
   caught and re-raised as
   `ImageDerivedConnectionError(CONNECTIONS_MATCHER_MISMATCH)`. No raw
   `ValueError` and no silent `bool`/`str` coercion can escape.
3. **Laundered matcher numeric diagnostics (second round).** `_finite_or_none`
   reduced a malformed matcher value - a string, a boolean or a `NaN`/`inf` - to
   `None`, so a non-conforming matcher could record fabricated "unmeasured"
   evidence instead of being refused. It is replaced by
   `_optional_finite_matcher_field`: `None` is preserved as the matcher's
   legitimate "not measured" value, otherwise the value must be an exact real
   numeric (Python `int`/`float` or a NumPy integer/floating scalar, never a
   `bool`/`np.bool_`) and finite; a string, a boolean, a non-finite value or any
   other object raises `ImageDerivedConnectionError(CONNECTIONS_MATCHER_MISMATCH)`
   naming the field, so the graph builder refuses the pair. It is applied to all
   four top-level `RelativeCameraEvidence` diagnostics
   (`median_selected_residual_degrees`, `parallax_degrees`,
   `cheirality_fraction`, `singular_ratio`) and to the same four fields of every
   `YawClassDiagnostic`. The `_require_finite_or_none` dataclass invariant (which
   already raises rather than laundering) remains as the record-level backstop.
   The old `_finite_or_none` symbol no longer exists: a word-boundary search for
   `\b_finite_or_none\b` in `src/` finds nothing.

## 4. Atomic invalid-before-budget behavior

`build_image_derived_connections` orders its decisions so a malformed input can
never be reported as a budget stop:

1. type checks on the session, budgets and matcher (raised);
2. **all source bindings**: the session is re-serialized and rebuilt through the
   accepted `image_derived_session_from_record`, so its exact key sets,
   schema/version, forbidden position/scale claims, retained unresolved facts and
   digest are re-checked by the module that owns them; then the camera
   key order and every frame's immutable, non-reopenable `uint8` array (bytes
   base chain required), shape and pixel digest are validated directly;
3. the **pair** budget against `C(n, 2)`, before any matcher call;
4. the delegated matcher, one pair at a time, with every derived diagnostic
   sealed as a typed contract mismatch if malformed;
5. the **total** support budget, while the evidence list is private.

A tampered digest, a dropped unresolved fact, a pixel-less record, a
reopenable-buffer swap or a pixel-digest mismatch therefore refuses as its own
reason even when the budget is `0`. Exceeding the total budget returns a refusal
with empty `evidence` and empty `components`, so no prefix of a graph is ever
emitted. The session's unresolved position/scale/yaw-quadrant notes and
`scale_known=False` are copied into every result.

## 5. Determinism

- Pairs are enumerated `(i, j)` with `i < j` over the session's declared frame
  order; evidence order is the same canonical sequence.
- Components are union-find groups ordered by first frame, with singletons for
  isolated frames.
- `matches_sha256`, `evidence_sha256`, `budget_sha256` and `graph_sha256` are
  SHA-256 over canonical sorted-key JSON (or shape-framed array bytes), so two
  runs over the same session and matcher produce byte-identical digests. This was
  verified both by rebuilding the same session twice and by re-running the
  adapter on one session.

## 6. Controls (`tests/test_image_derived_connections.py`, 24 tests)

Synthetic tests inject the adapter's declared `matcher` seam; no SIFT is run and
no pixels are read by the matcher. Sessions are produced by the accepted
`build_image_derived_session` with only `image_only.solve_image_cameras`
monkeypatched.

- **Pair ordering / delegation** - a four-frame session delegates exactly the six
  pairs `(a,b) (a,c) (a,d) (b,c) (b,d) (c,d)`, once each, with the session's own
  read-only image objects (identity-checked), and the evidence order matches.
- **Immutability** - a hostile matcher cannot mutate session pixels; digests are
  unchanged.
- **Swapped reopenable buffer (adversarial)** - replacing a frame's bytes-backed
  image with a byte-identical, same-shape, `writeable=False` numpy-owned buffer
  (same digest) refuses as `connections_session_mismatch` at a zero pair budget
  with **zero** matcher calls; the test then reopens the swapped buffer with
  `setflags(write=True)`, mutates it and shows the digest moves, proving the
  refusal was the only thing standing between the adversary and the matcher.
- **Malformed hypothesis (adversarial)** - out-of-range/non-integer `yaw_class`,
  negative `support`/`spatial_bins`, non-boolean `passed`, non-string `refusal`,
  a string/plain-boolean/NumPy-boolean/non-finite/non-scalar value in any of the
  four optional numeric diagnostics, a non-diagnostic entry and a non-sequence
  `hypotheses` each raise `ImageDerivedConnectionError` with
  `connections_matcher_mismatch`; no raw `ValueError`, no silent coercion and no
  laundering to `None` escapes.
- **Malformed top-level diagnostics (adversarial)** - a string, plain/NumPy
  boolean, `NaN`/`inf` (plain/NumPy) or non-scalar injected into
  `median_selected_residual_degrees`, `parallax_degrees`,
  `cheirality_fraction` or `singular_ratio` of a refused **or** a solved matcher
  result raises the typed mismatch naming that field, so the pair is refused
  rather than recorded with fabricated `None` evidence.
- **Legitimate `None` and exact finite preserved** - a refused result's real
  `None` diagnostics survive unchanged and a solved result's finite Python/NumPy
  scalars are stored as exact floats (real hypothesis `None` values too), proving
  the strict parser does not break the matcher's honest answers.
- **Unit-vector acceptance** - an injected matcher that delegates to the public
  `match_feature_frames` on a synthetic projected scene yields a solved record
  whose unit baseline and 3x3 rotation are byte-for-byte the matcher's, with the
  correct yaw class and support.
- **Typed abstentions** - a real refused `match_feature_frames` result and a
  synthetic refusal both pass their `JOINT_REFUSAL_REASONS` token through
  unchanged, with no vector fields.
- **Digest determinism** - two independent sessions with identical bytes and
  estimates give equal `graph_sha256` and equal per-record `evidence_sha256`; a
  re-run on one session is identical.
- **Pair budget before work** - `max_pairs` below `C(n, 2)` refuses with
  `connections_budget_exhausted`, no evidence/components, and **zero** matcher
  calls.
- **Total budget before partial** - a first pair inside the total budget and a
  second pair crossing it refuses with no evidence/components even though one
  record was already built.
- **Session without pixels** - a session rebuilt via
  `image_derived_session_from_record` (`has_pixels is False`) refuses with
  `connections_no_pixels` and zero matcher calls.
- **Session mismatch before budget** - a tampered session digest and a dropped
  unresolved fact both refuse with `connections_session_mismatch` even at a zero
  budget.
- **Components** - solving `(a,b)` and `(b,c)` yields `((a,b,c),(d))` and a mix of
  solved/refused records.
- **Unit guard / contract errors** - a non-unit solved baseline becomes a typed
  `invalid_baseline_direction` pair refusal; a mismatched matcher pair, an untyped
  matcher reason and a non-session input raise the typed contract error.
- **No smuggled geometry** - the graph record's recursive keys contain no
  `image`/`pixels`/`position`/`translation`/`scale`/`bounds`/`block`/`transform`
  keys, and the record binds `session_sha256` and `scale_known=False`.

## 7. Evidence (exact commands and results)

Workspace interpreter `.venv\Scripts\python.exe`, Python 3.10.11, NumPy 2.2.6,
OpenCV 5.0.0:

```
.venv\Scripts\python.exe -m py_compile src\mc_photogrammetry\pipeline\image_derived_connections.py
    -> exit 0
.venv\Scripts\python.exe -m py_compile tests\test_image_derived_connections.py
    -> exit 0
.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_image_derived_connections.py
    -> 24 passed
.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_image_derived_session.py
    -> 45 passed
.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_project_store.py
    -> 30 passed
.venv\Scripts\python.exe -m pytest -o addopts= -q tests/test_joint_matching.py
    -> 10 passed (unmodified matcher sanity)
```

Combined required run: **109 passed** (adapter 24 + session 45 + store 30 +
joint 10).

Boundary grep proving no silent path remains (ripgrep word-boundary `\b`, so the
strict dataclass backstop `_require_finite_or_none` is not a false hit):

```
rg "\b_finite_or_none\b" src
    -> no matches
rg "\b_finite_or_none\b" .
    -> 1 match, tests/test_image_derived_connections.py:644, a docstring that
       narrates the removed defect (not a code path)
rg "_optional_finite_matcher_field" src
    -> 1 definition + 8 call sites (4 top-level + 4 hypothesis)
```

Ruff is not installed in the workspace interpreter, so no lint run was
claimed.

## 8. Limits and non-claims

1. This is a bounded evidence adapter, not a reconstruction. It computes no
   positions, no metric scale and no scene transform; `scale_known` stays `False`
   and the session's unresolved facts are preserved.
2. A solved edge is the matcher's claim, not a validated absolute pose. The
   adapter records it verbatim with its typed refusal history; it does not
   re-adjudicate the matcher's gates, thresholds or ambiguity margins.
3. The pair budget is a count of pairs, and the total budget is a sum of selected
   support; they bound work and admitted evidence, not accuracy.
4. The adapter does not read pixels itself and never serializes pixels; evidence
   digests cover the match arrays the matcher returned.
5. Production wiring (normal-mode orchestration, state fusion, `.litematic`
   promotion) remains out of scope and blocked pending separate acceptance.
