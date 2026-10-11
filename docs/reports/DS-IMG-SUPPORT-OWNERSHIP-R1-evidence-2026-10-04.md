# DS-IMG-SUPPORT-OWNERSHIP-R1 — evidence report: ingress replay seam

Date: 2026-10-04
Author: DeepSeek (CLI worker, report-only)
Assignment: latest stop directive in `AGENT_COMMS.md` (2026-10-04 Lead: "DS-IMG-SUPPORT-OWNERSHIP-R1 run
stopped at ingress replay seam"), which instructs: stop implementation/run work for R1; prepare a concise
evidence report from existing logs, source and artifacts only; include the attempt chronology, the
`provenance_mismatch` seam, source lines/digests, existing focused test results, the exact output digest and
typed-unavailable downstream stages. No rerun, no screenshot change, no lattice production-code change.

**Scope discipline.** Report-only. No reconstruction, no test, no `py_compile`, and no product/evaluator/solver
path was executed while producing this report. No screenshot, reference camera, mask, ownership payload, JAR or
truth payload was opened. No file other than this new report and the appended `AGENT_COMMS.md` result entry was
written. The only inputs read were tracked/untracked **source and test text**, the previously produced R1 run
artifact, and the existing coordination documents.

Prohibitions observed from `docs/deepseek-worker.md` (direct-shell policy): no junction/reparse-point
operations, no recursive deletion, no encoded/obfuscated commands, no generated executables, no installers, no
background processes, no persistence, no Defender changes/exclusions, no system-directory writes, and no live
launcher/client automation. Read-only source inspection only.

---

## 1. Directive under report

- Latest heading: `AGENT_COMMS.md:6561` — "2026-10-04 Lead: DS-IMG-SUPPORT-OWNERSHIP-R1 run stopped at
  ingress replay seam".
- Stop instruction: do not retry; do not run another reconstruction for R1; report-only evidence.
- `AGENT_COMMS.md` pre-append SHA-256 (this report's only later change is the appended result entry):
  `7D72D48D0758608216210699CA6883992B85E94ACE494A7F1F910C40C4841502` (6,569 lines at read time).

## 2. Attempt chronology (existing records only, no rerun)

| # | Event | Outcome |
|---|-------|---------|
| 1 | First harness attempt before the run | Failed only at pinned-input **hash case normalization**, before any inference. No reconstruction run. |
| 2 | Single actual R1 capture-enabled Route B inference attempt | Reached `build_image_derived_entry` / session creation and wrote `.tmp-support-ownership-r1-20261004/phase_a.mcscan`. Stopped at `measure_image_derived_lattice` with `provenance_mismatch`. |
| 3 | Downstream Phase A stages | Not reached. No signatures, no matcher rows, no graph/trace/solve output, no candidate-support capture, no `phase_a.json`, no `report.json`. |
| 4 | Phase B evaluator ownership join | Not reached. No evaluator payload opened; no ownership statistic exists. |

Total actual R1 reconstruction runs: **one** (the run at step 2). No retry was performed after that.

The step-2 harness order is fixed in source: `run()` (1) refuses a non-fresh root, (2) resolves and
re-verifies pinned image bytes, (3) calls the chain that builds the entry and measures the lattice,
(4) only then computes the Phase A payload, freezes it and runs Phase B. The failure at the lattice stage
therefore precedes and prevents the Phase A freeze. The lone `phase_a.mcscan` is the entry/session
`.mcscan` written by `build_image_derived_entry`, not a sealed Phase A payload.

Relevant source lines (`scripts/evaluation/run_candidate_support_ownership.py`):
- `:194-196` `build_image_derived_entry(..., capture_ingress=True)`
- `:201-204` `measure_image_derived_lattice(...)`; `PlacementRefusal` → `HarnessError(f"lattice refusal: ...")`
- `:648-650` non-fresh-root refusal; `:651-658` pinned input verification then chain run
- `:659-669` Phase A payload, freeze, then Phase B
- `:557-587` Phase B typed-unavailable branches (no payload; candidate support not assembled; camera/occlusion unproven)

## 3. Confirmed source seam

The seam is a one-field canon mismatch in the accepted-record replay of the **lattice measurement** stage.

Producer side — the ingress pin really is bound to each session frame:
- `pipeline/image_derived_session.py:300-339` `build_image_derived_session(..., ingress=...)` validates the
  mapping and constructs each frame with `ingress=ingress_by_frame.get(frame_id)` (`:338`).
- `pipeline/image_derived_session.py:741-750` `_frame_payload(frame)` adds `payload["ingress"]` **when**
  `frame.ingress is not None` (`:748-749`). This is the canonical, session-digested payload used for
  `session_sha256`.
- `pipeline/image_derived_session.py:601-611` the record round-trip rebuilds `ingress` from the record via
  `FrameIngressProvenanceV1.from_record` (`:604`) and passes it to the rebuilt `ImageDerivedFrame`.

Replay side — the lattice stage's frame replay drops that field:
- `pipeline/image_derived_lattice_measure.py:227-257` `_replay_frame(frame, record_frame)`:
  - `:245-248` rebuilds `ImageDerivedFrame(frame_id=..., source_label=..., image=..., image_shape=...,
    image_sha256=...)` — **no `ingress=` argument**, so `replayed.ingress is None`.
  - `:254-256` then compares `session_mod._frame_payload(replayed) != session_mod._frame_payload(record_frame)`.
  - `:257` on mismatch: `_refuse("provenance_mismatch", "session.frame")`.
- Because `record_frame` is the record-rebuilt frame (ingress present) and `replayed` is the
  constructor-rebuilt frame (ingress absent), the two payloads differ by exactly the `"ingress"` key whenever
  the session was produced with `capture_ingress=True`. With `ingress=None` on both sides the payloads are
  equal, which is why the default path is unaffected.
- Call chain: `pipeline/image_derived_lattice_measure.py:852-856` `_assemble` → `:855` `_replay_session` →
  `:277-278` per-frame `_replay_frame`; the `_InputRefusal` is converted back to a typed
  `PlacementRefusal` at `:843-844`.

Interpretation: the accepted R4 session-bound ingress path is captured and sealed correctly by the entry
stage, but the next pipeline stage (lattice measurement) cannot replay a session that carries the pin. This is
a producer/consumer contract gap inside the lattice replay, **not** an ingress-capture failure and **not** an
ownership/alignment problem.

### 3.1 The entry stage already handles the same field correctly

`pipeline/image_derived_entry.py:749-785` implements the same accepted-record frame replay and already carries
the identity-bearing metadata explicitly:

- `:765-776` builds the replay frame with `ingress=record_frame.ingress` and a comment stating the pin is
  identity-bearing metadata that must be carried for the payload comparison to be exact;
- `:780-785` compares `_frame_payload` and refuses if it differs.

This is direct in-repo evidence that the intended replay semantics for an ingress-bearing frame are "carry the
record's own pin", and that the lattice-measure replay is the one place missing it. A sibling replay in
`pipeline/image_derived_signatures.py:306-320` avoids the issue differently: it round-trips the record, checks
only `session_sha256` and the frame-id sequence, and returns the **original** retained frames (`:320`), so its
`:426-427` ingress lookup sees the real pins and its accepted R4 source-tap binding stays available. Its
`:312-313` `dataclasses.replace(frame)` validation does not compare `_frame_payload`.

## 4. Captured artifact evidence (exact digests)

Run directory: `.tmp-support-ownership-r1-20261004/`
Contents: exactly one file, `phase_a.mcscan` (9,319 bytes, last written 2026-10-04 13:56:02).

- `phase_a.mcscan` SHA-256 = `4E25D7E3E6C53640A65D257848331F7B7ED3879F2E863D7B55436818F435B8FF`
  — matches the digest recorded in the stop directive exactly.
- Parcel structure: top-level `schema_version`, `name`, `minecraft_version`, `source_paths`, `volume`,
  `image_derived_session`.
- `image_derived_session`: `schema`, `version`, `frames` (3), `cameras`, `components`, `unresolved`,
  `scale_known`, `local_cameras`, `session_sha256`.
- Frame `image-0000`: `source_label` = the T2 frozen screenshot
  `templates 2/2026-09-02_13.22.14.png`, `image_shape` `[1369, 2560, 3]`, `image_sha256`
  `209064e8cc5b95cd8403fddafa6e64758ffd8e1cf6be41e58118ac73e386b9ff`, and an `ingress` record with
  `byte_sha256` `eb22541171e030ff94d6839a6626e7160c69ae3a780f07aec0298322be665d46`, `byte_length` 1609624,
  `decoded_shape` `[1369, 2560, 3]`, `decoded_dtype` `|u1`, `decoded_pixel_sha256` equal to the frame digest,
  `transform.source_size` `[2560, 1369]`, `crop_origin` `[0, 0]`, `crop_size`/`output_size` `null`, and
  `decoder` `input.image.load_image:exif_transpose+convert('RGB')`.
- All three frames `image-0000`, `image-0001`, `image-0002` carry pins; there is no partial-capture case.

**No downstream artifacts exist.** `phase_a.json` and `report.json` are absent, and no `report.json` for the
support-ownership root exists anywhere in the tree. The failed run produced no `HarnessError` record file and no
separate stdout/stderr log; the seam evidence is the artifact above plus the source path and the comms record.

## 5. Unaccepted side-channel hooks: absent

The Lead amendment required removal of `ingress_observer`, `ingress_by_frame` and `candidate_support_ingress`.
Source scan (ripgrep over `*.py`) finds no such parameter or branch in `image_derived_entry.py`,
`image_derived_signatures.py` or `image_derived_graph.py`. The only occurrences are negative assertions that the
hooks do **not** reappear:
- `tests/test_candidate_support_provenance.py:1344-1346`
- `tests/test_candidate_support_ownership_harness.py:83-85`

(`ingress_by_frame` also appears in `pipeline/image_derived_session.py:326/338` and
`pipeline/candidate_support_provenance.py:487-508` as ordinary local variable names for keyed lookup maps; they
are not the removed side-channel parameter.) The accepted session-bound path remains:
`capture_ingress: bool = False` (`image_derived_entry.py:409`), `signature_witness_sink: list | None = None`
(`image_derived_signatures.py:350`), `candidate_support_sink=None` (`image_derived_graph.py:4974`), with the
graph record sourcing pins only from `accepted_session.frames` (`image_derived_graph.py:5155-5157`).

## 6. Existing focused test results (reported, not rerun)

No test was executed for this report, so the following are cited from accepted prior records, not re-measured
here:
- Accepted R4 focused set (the seven authorized R4 modules): **40 passed** for
  `tests/test_candidate_support_provenance.py`; combined R4 set **286 passed** as reported in the accepted R4
  result and its independent review.
- `tests/test_candidate_support_ownership_harness.py` is a focused synthetic module (233 lines) covering: the
  harness uses only the accepted R4 path; Phase B typed-unavailable without a real payload; refusal before
  alignment when candidate support is refused; unpinned camera/occlusion refusal without reading; malformed and
  tampered payload refusals; and the freeze-before-evaluator-read ordering. These are synthetic and read no
  image/camera/mask/reference.

**Test-coverage gap around the seam (static observation).** `tests/test_image_derived_lattice_measure.py`
builds its session via `_build_session` (`:69`) with no `ingress`, and the module's `provenance_mismatch`
controls (`:337-375`) cover a stale session digest, a writable constructor-bypassed frame, and foreign pose
graphs — never an ingress-bearing session. No test in the lattice-measure, entry or signatures modules
exercises `_replay_frame` with `frame.ingress is not None`. That is why the seam passed the accepted focused
suites while failing the first real capture-enabled run. A fix will need a focused regression that replays an
ingress-bearing session (equal payload when the pin is carried; refused when the carried pin contradicts the
record) plus a default-path neutrality check (`ingress=None` behavior unchanged).

## 7. Typed-unavailable downstream stages

The run stopped at lattice replay, so every later stage was never evaluated. The following are the typed
abstentions/unavailables that apply — they must **not** be read as measured zeros:

- `measure_image_derived_lattice` returned a `PlacementRefusal` with `reason = "provenance_mismatch"` and
  `detail.field = "session.frame"` (`image_derived_lattice_measure.py:257`, surfaced at `:843-844`). The
  harness maps it to `HarnessError("lattice refusal: provenance_mismatch")`.
- Signatures: not run (`derive_image_derived_signatures` never reached). Had they run with an unrepairable
  session they would refuse `provenance_mismatch` at `image_derived_signatures.py:315/319`; with the accepted
  session they would bind the real pins at `:426-427/454-457`.
- Source-tap witness coverage, candidate-support record (`status`, `reason`, `row_count`, `record_sha256`),
  same-axis capture, solver snapshots, graph/trace/solve digests and the R10/R11 comparison: **not produced**.
  The harness serializes these only inside the Phase A payload after a successful chain.
- Geometry orchestration: not reached. For reference, its session binding replay would refuse identically
  (`pipeline/image_derived_geometry_orchestration.py:212-214` → `_replay_session`), and its per-stage abstention
  mapping (`:229-243`) yields `reason = "<stage>_unavailable"`.
- Phase B ownership join: `status = "unavailable"` with `reason = "ownership_payload_unavailable"`
  (`run_candidate_support_ownership.py:557-564`); the unassembled-candidate-support branch
  (`:571-578`, `candidate_support_unavailable`) and the camera/occlusion branches (`:579-587`,
  `evaluator_camera_misalignment` / `occlusion_unavailable`) would also apply. No projected ownership
  statistic, precision/recall or kept/dropped overlap exists, and none was computed.
- Product output, placement, volume, exact states, `.litematic` round-trip: not reached; typed unavailable,
  not zero.

The captured `phase_a.mcscan` therefore contains only pre-lattice session data. It is evidence that capture
succeeded, not evidence about candidate support, same-axis filtering or ownership.

## 8. What is proven / not claimed

Proven by inspection of current source plus the sealed artifact:
1. The accepted R4 capture path works at the entry stage: the produced `.mcscan` binds a well-formed ingress
   pin to all three frames, and the entry replay already carries pins correctly.
2. The lattice measurement stage drops `ingress` when rebuilding each retained frame and then compares the
   canonical payload including `ingress`, so any capture-enabled session refuses with
   `provenance_mismatch`/`session.frame`.
3. The seam is deterministic and independent of evaluator data: no camera, reference, mask, ownership or truth
   payload was needed to trigger it, and none was read.
4. The three removed unaccepted hooks are absent from the runtime modules.

Not claimed:
- No causal or accuracy claim: no geometry, candidate-support, ownership, precision/recall or state metric was
  measured in R1.
- No claim that the pin is otherwise unverifiable, or that the lattice result would be valid; only that replay
  cannot reach the measurement.
- No production/test/capture state was modified; no screenshot was created or altered; no threshold, gate,
  selection or promotion state changed.
- No claim that the fix is trivial in all cases: an ingress-aware replay must preserve the accepted
  fail-closed checks (frame identity, shape, pixel digest, and pin/record agreement) rather than blanket-copying
  untrusted metadata. A production task should define that behavior with default-path neutrality tests.

## 9. Proposed next bounded step (for lead decision, not executed)

A separate production task after lead review: make the lattice-measure accepted-record frame replay
ingress-aware (carry the record's own pin and keep the payload comparison exact, following the existing
`image_derived_entry.py:765-785` precedent), add focused regressions for an ingress-bearing session plus
default-path neutrality, and only then authorize **one** fresh R1 capture-enabled run. This report performs
none of that and stops here.

## 10. Verification appendix (read-only checks performed for this report)

| Check | Result |
|-------|--------|
| `phase_a.mcscan` exists and digest | 9,319 bytes; `4E25D7E3E6C53640A65D257848331F7B7ED3879F2E863D7B55436818F435B8FF` (matches directive) |
| Run directory inventory | exactly one file; no `phase_a.json`, no `report.json` |
| Hook scan (`ingress_observer`/`ingress_by_frame`/`candidate_support_ingress`) | runtime modules clean; only negative test assertions |
| `_replay_frame` source read | `image_derived_lattice_measure.py:227-257`; `ingress` argument absent |
| Entry precedent read | `image_derived_entry.py:749-785`; `ingress=record_frame.ingress` present |
| Signatures replay read | `image_derived_signatures.py:306-320`, `:426-427`, `:454-457`; returns original frames |
| Session payload read | `image_derived_session.py:741-750`; conditional `ingress` key |
| Test-coverage scan | `test_image_derived_lattice_measure.py:69` session has no `ingress` |
| Executions performed for this report | none (no pytest, no `py_compile`, no reconstruction) |

File SHA-256 at report time (source of record; note these files are untracked in this working tree, so hashes —
not a commit — pin them):

| File | SHA-256 |
|------|---------|
| `src/mc_photogrammetry/pipeline/image_derived_lattice_measure.py` | `C782EDBF60348F9307D387A61FCAD5674394F8657D2147C0A03DA5E49B16E673` |
| `src/mc_photogrammetry/pipeline/image_derived_session.py` | `A97D5CE04EF91DBA6501DFF9D0436D5A09E2F74539C411F5CDAFD05E85720060` |
| `src/mc_photogrammetry/pipeline/image_derived_entry.py` | `FC794F891C83B60EE823482EA902A167775C2A3C5CD40DD5F92286A3B0DD0739` |
| `src/mc_photogrammetry/pipeline/image_derived_signatures.py` | `8F7D24C018395A32A1D2A3CDA616E2E7A7EED1873DCEFC74FE3D673911CF7555` |
| `src/mc_photogrammetry/pipeline/image_derived_graph.py` | `2F94B9E3125F16E493C11647A91882BF1C61E1176CA7A9B93BD3AB28F511C5E2` |
| `src/mc_photogrammetry/pipeline/candidate_support_provenance.py` | `5B692400F9B8CCCFECFB2727D948FFF353BD790462F1664395F45B43C96630EA` |
| `src/mc_photogrammetry/pipeline/frame_ingress.py` | `3411B4067281B19A6AC2E530ECE5DFC2E53E0F63B978E83E09FFADB27F715F0A` |
| `scripts/evaluation/run_candidate_support_ownership.py` | `1D022AF3FE618C98D05ABA141D1892FA6739727597CC377700D41584052866DD` |
| `tests/test_candidate_support_provenance.py` | `441035CD2B691BC37D1DFCC5187A4C6DD55362270FA6A7D4BF4F83185B0DCD83` |
| `tests/test_candidate_support_ownership_harness.py` | `B3C4D2E9853D0B81C93A0C8C7B1618B45C75160770F7CCE2113096FEA333D314` |

Repo revision at report time: `c6dba2cf4610876db7d24361eb13fc9ea3ceca0c`.

Stopped for lead review. No retry, no run, no source/test/capture change.
