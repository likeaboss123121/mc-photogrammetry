# DS-R4-V2-ENTRY-ADAPTER-FIX-2026-10-04

Task ID: `DS-R4-V2-ENTRY-ADAPTER-FIX`. Owner: DeepSeek CLI worker (Sol-equivalent,
bounded implementation, direct-shell profile). Date: 2026-10-04.

Status: **IMPLEMENTED — for Luna review. Stopped as instructed.** Only
`scripts/evaluation/run_route_b_diagnostic.py`, `tests/test_route_b_diagnostic_runner.py`
and this report were written; a concise result was appended to `AGENT_COMMS.md`. No
screenshot/image payload, evaluator/truth/reference/litematic/scene/camera/F3/direction/
mask/coordinate input, actual runner invocation, capture, tuning, Minecraft/launcher,
production source/default edit or unrelated dirty file was touched.

## 1. Root cause

`_run_single_pass` tested the entry with `_status_refusal("entry", entry)`
(`run_route_b_diagnostic.py:415` before this change). `ImageDerivedEntryResult`
intentionally has **no** `status` field (`image_derived_entry.py:301-313`); its normal
successful terminal is always `GEOMETRY_UNAVAILABLE` with an empty `reason` and the
informational `detail`
(`image_derived_entry._terminal_reason:966-975`). `getattr(entry, "status", None)` was
therefore `None != "assembled"`, so the runner refused **every** successful
graph+pose entry. This is exactly the R1 refusal (`reason="unavailable"`, `field="entry"`)
reported in `DS-R4-V2-ROUTEB-FROZEN-IMAGE-R1` and traced in
`DS-R4-V2-ENTRY-REFUSAL-TRACE`.

## 2. Change (runner)

Added `_entry_refusal(entry)` (`run_route_b_diagnostic.py:408-444`) and used it at the entry
gate (`:458-463`, replacing the invalid `_status_refusal("entry", entry)` call plus its
redundant inline graph/pose check). The adapter mirrors the production readiness contract:

1. **Typed handling preserved first.** A `placement.PlacementRefusal` entry is carried
   through as `DiagnosticRefusal(entry.reason, "entry", entry.detail)`, exactly like the
   other stages.
2. **Accept only when ready.** The entry passes iff
   `entry.graph.status == "ok"` **and** `entry.pose_graph.status == "ok"`.
3. **Geometry-unavailable terminal is informational.** When both graphs are `ok`, the
   status-less, empty-`reason` terminal proceeds unchanged to the ordinary image-only
   `lattice -> signatures -> graph` chain. The empty reason and informational detail are
   not re-read as a refusal.
4. **Otherwise refuse typed.** A graph/pose-not-`ok` entry raises
   `DiagnosticRefusal(entry.reason or "entry_unavailable", "entry", entry.detail)`; when the
   entry carries no detail, a deterministic per-stage explanation
   (`entry connection graph (<status>) and pose graph (<status>) is unavailable`) is used.
   Type reason/detail from the real derived terminal are preserved verbatim.

`_status_refusal` is unchanged and still governs the `lattice`/`signatures` stages (their
results do carry `status`). No production module, default path, `_default_api`, CLI surface,
threshold or digest was changed. The module docstring call-chain item 1 now records the entry
acceptance contract.

## 3. Change (tests)

- The shared `_fake_api` entry no longer fabricates `status="assembled"`; it returns a
  status-less, real-shaped terminal (new `_entry_terminal` helper, `tests:216-232`) with
  `terminal="geometry_unavailable"`, `reason=""`, the informational
  `_GEOMETRY_UNAVAILABLE_DETAIL` (`tests:92-94`), and `graph`/`pose_graph` `ok`. Every
  existing pass-path test therefore now exercises the corrected gate against the real
  status-less contract.
- New helpers: `_session_for` (`tests:207-213`) and `_entry_failure_api`
  (`tests:288-314`), whose later stages fail loudly if a wrongly accepted entry reached them.
- New focused tests:
  - `test_status_less_geometry_unavailable_entry_passes_gate_and_calls_lattice`
    (`tests:369-387`): asserts the terminal has **no** `status` attribute, is
    `geometry_unavailable` with empty reason and the informational detail, and that the gate
    passes with `calls == ["entry", "lattice", "signatures", "graph"]` and a complete report.
  - `test_entry_graph_failure_refuses_before_lattice_and_preserves_reason_detail`
    (`tests:394-410`): refused `graph`, reason `entry_connection_graph_failed`; refusal
    happens with `calls == ["entry"]` (lattice/signatures/graph never run), typed
    reason/field/detail are preserved, no record/project/partial/staging artifact remains, and
    the written `refusal.json` carries the same reason.
  - `test_entry_pose_failure_refuses_before_lattice_and_preserves_reason_detail`
    (`tests:417-433`): the same control for a refused `pose_graph` with `graph == ok`, reason
    `entry_pose_graph_failed`.

The test-module docstring pinned invariants were extended with the entry-gate contract.

## 4. Checks run (focused only)

| Check | Command | Result |
|---|---|---|
| Byte-compile | `.venv\Scripts\python.exe -m py_compile scripts/evaluation/run_route_b_diagnostic.py tests/test_route_b_diagnostic_runner.py` | exit 0 |
| Focused runner suite | `.venv\Scripts\python.exe -m pytest tests/test_route_b_diagnostic_runner.py -q` | **37 passed**, exit 0 (34 pre-existing + 3 new; collection confirms 37) |

This includes `test_production_default_and_public_sources_stay_byte_identical`, which still
passes: the pinned production module hashes are unchanged and the runner still makes exactly
one `derive_image_derived_signatures(` / `derive_image_derived_graph_attribution(` call. No
full-suite, runner invocation or capture was performed.

## 5. Hashes

| File | SHA-256 | Bytes |
|---|---|---|
| `scripts/evaluation/run_route_b_diagnostic.py` | `050AC0123386333B55E7321FE9D121DCF9CAC3E2635097931D2869BA9D0E8EC4` | 50,755 |
| `tests/test_route_b_diagnostic_runner.py` | `D9A595EA84886B645C95182472D6B2E34EC057C410F9AE47195FFA17B54F04F3` | 51,487 |

Previous accepted (pre-fix) values for comparison: runner
`331D0557EA2AEBA436BAA25D364386F63F90C9C5383066F5CD55CCF2F4CAE16A` (48,505 B), tests
`7D7AFD62AF0BC6C16A7DDC310C2E85D786595AB55D08F2835D79001D28BDB618` (45,302 B). Both files are
untracked in the working tree (`??`); no unrelated path was modified or inspected.

## 6. Disposition

- This is a **runner/entry adapter correction only**. It removes the unconditional entry
  refusal so a normal image-only entry can proceed; it makes **no** claim that the three
  frozen PNGs assemble a lattice/signature/match or produce the exact 1,928-row universe.
- Downstream real gates are unchanged and independently owned: lattice `assembled`,
  signatures `assembled`, graph matcher match, exactly-once solver abstention firing the V2
  sink (solver success still yields `capture_unavailable`), and the sealed 1,928-row
  replay/count/identity gate.
- **Stopped for Luna review.** No re-run of the frozen R1 invocation was performed or
  authorized here; a real capture would still require separate lead authorization.
