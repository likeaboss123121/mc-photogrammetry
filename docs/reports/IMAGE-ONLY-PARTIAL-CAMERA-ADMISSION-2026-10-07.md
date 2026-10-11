# Partial camera admission - ready for lead review

Final status (2026-10-07 00:03 CDT): **275 focused tests passed in 65.64s; pytest and shell exit 0; 18 final Python files compiled, exit 0. STOP for independent lead review.** Same completion attempt 6/8. This final handoff supersedes the initial verification status preserved below.

## Remediation and final checks

Lead's independently captured rerun reported **262 passed / 5 failed in 68.72s, exit 1**. Three entry tests accepted hostile active-frame pixels (writable, reopenable and swapped identities); lattice accepted a constructor-bypassed writable active frame; the disconnected-gauge fixture omitted two required tolerance args. Full lead trace remains in the lead tool session, not a repository log file. This count is lead evidence and does not reconstruct the initial lost developer terminal.

Lead explicitly authorized bounded same-attempt remediation. Entry and lattice now replay BOTH the full input inventory and the active/admitted objects against canonical identity and retained immutable pixels. Each active object must be its corresponding inventory object; this covers v1 and v2. Retained ingress pins are validated, rather than substituted with canonical pins. The fixture supplies the required rotation/direction tolerances. Eight new synthetic v2 hostile controls cover writable/reopenable pixels, changed or cross-identity pixels and detached identities; existing v1 controls remain.

The first authorized remediation run retained session **90295**: **274 passed / 1 failed in 65.71s; pytest/shell exit 1**, with 18-file compilation exit 0. The remaining ingress-mismatch test still received a correct `provenance_mismatch` refusal, but diagnostic `detail.field` was `session.frame` instead of the established `session.frame.pixels`. Lead then explicitly authorized only that diagnostic correction and one final identical ten-module check. No assertion was weakened and no model/image work was authorized.

The final run retained session **48763**: **275 passed in 65.64s; pytest/shell exit 0**, compilation **18 files, exit 0**. All checks used final source; no Python source/test edits followed the passing run. Source guard, immutable pixels, full source paths, v1/v2 records, cardinality and disclosure checks in the focused set passed. No full-suite or real-image validation is claimed.

### Exact command for the two authorized remediation runs

Repository working directory and shell were unchanged from the initial record below. `login=false`.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.venv\Scripts\python.exe -B -c "import py_compile; from unittest.mock import patch; paths=['src/mc_photogrammetry/pipeline/image_derived_session.py','src/mc_photogrammetry/pipeline/image_derived_entry.py','src/mc_photogrammetry/pipeline/image_only.py','src/mc_photogrammetry/pipeline/image_derived_connections.py','src/mc_photogrammetry/pipeline/image_derived_lattice_measure.py','src/mc_photogrammetry/pipeline/image_derived_production.py','src/mc_photogrammetry/project/store.py','src/mc_photogrammetry/cli.py','src/mc_photogrammetry/ui/main_window.py','tests/test_image_derived_session.py','tests/test_image_derived_entry.py','tests/test_image_derived_connections.py','tests/test_image_derived_pose_graph.py','tests/test_image_derived_lattice_measure.py','tests/test_project_store.py','tests/test_image_derived_production_cli.py','tests/test_image_derived_delivery_cli.py','tests/test_desktop_image_derived_route.py']; writes=[]; context=patch('importlib._bootstrap_external._write_atomic',side_effect=lambda p,d,m:writes.append((p,len(d)))); context.start(); [py_compile.compile(p,cfile=p,doraise=True) for p in paths]; context.stop(); print('py_compile passed:',len(writes),'files; bytecode writes intercepted')"
$compileExit = $LASTEXITCODE
Write-Output ('PY_COMPILE_EXIT=' + $compileExit)
if ($compileExit -ne 0) { exit $compileExit }
.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider tests/test_image_derived_session.py tests/test_image_derived_entry.py tests/test_image_derived_connections.py tests/test_image_derived_pose_graph.py tests/test_image_derived_lattice_measure.py tests/test_project_store.py tests/test_image_derived_production_delivery.py tests/test_image_derived_production_cli.py tests/test_image_derived_delivery_cli.py tests/test_desktop_image_derived_route.py
$pytestExit = $LASTEXITCODE
Write-Output ('PYTEST_EXIT=' + $pytestExit)
exit $pytestExit
```

Compiler writes were intercepted; no bytecode files were written. Hash collection used `Get-FileHash -LiteralPath <each path> -Algorithm SHA256`, exit 0.

### Captured remediation failure before diagnostic correction

```text
.... [ 78%]
...........................................................              [100%]
================================== FAILURES ===================================
____________ test_a_mismatched_record_frame_ingress_pin_is_refused ____________

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x000001BDA621B580>

    def test_a_mismatched_record_frame_ingress_pin_is_refused(monkeypatch):
        """The carried record pin is re-validated against the retained pixels."""
        session = _build_session(monkeypatch, ingress=True)
        rebuilt = session_mod.image_derived_session_from_record(
            session_mod.image_derived_session_to_record(session))
        frame = session.frames[0]
        wrong_pixels = np.array(frame.image, copy=True)
        flat = wrong_pixels.reshape(-1)
        flat[0] = (int(flat[0]) + 1) % 256
        assert session_mod.image_sha256(wrong_pixels) != frame.image_sha256
        mismatched = _ingress_pin(frame.frame_id, wrong_pixels)
        forged_record = _forge(rebuilt.frames[0], ingress=mismatched)
        with pytest.raises(m9c._InputRefusal) as caught:
            m9c._replay_frame(frame, forged_record)
        assert caught.value.reason == "provenance_mismatch"
>       assert caught.value.detail["field"] == "session.frame.pixels"
E       AssertionError: assert 'session.frame' == 'session.frame.pixels'
E         
E         - session.frame.pixels
E         ?              -------
E         + session.frame

tests\test_image_derived_lattice_measure.py:486: AssertionError
=========================== short test summary info ===========================
FAILED tests/test_image_derived_lattice_measure.py::test_a_mismatched_record_frame_ingress_pin_is_refused
1 failed, 274 passed in 65.71s (0:01:05)
PYTEST_EXIT=1
```

### Complete captured final output

```text
py_compile passed: 18 files; bytecode writes intercepted
PY_COMPILE_EXIT=0
........................................................................ [ 26%]
........................................................................ [ 52%]
........................................................................ [ 78%]
...........................................................              [100%]
275 passed in 65.64s (0:01:05)
PYTEST_EXIT=0
```

## Final Python hashes

| Path | SHA256 |
| --- | --- |
| `src/mc_photogrammetry/pipeline/image_derived_session.py` | `e48414bf9eea6a7da7a7cc7d8c19865e0494953ac8796d9f3cd2e57e81b6248e` |
| `src/mc_photogrammetry/pipeline/image_derived_entry.py` | `1767b416061b20b1e60cd5c2d3d5b0814f232a3bc56d01328fc294818b20b6fc` |
| `src/mc_photogrammetry/pipeline/image_only.py` | `ca28afd0e9dc36d2b827a28807b736d635374acb2787506c2d2f3d07c929c93c` |
| `src/mc_photogrammetry/pipeline/image_derived_connections.py` | `b46d51a17f45a292666ef089199dca120cc11b91125d028e5114d0c6a010aedd` |
| `src/mc_photogrammetry/pipeline/image_derived_lattice_measure.py` | `c894ef4cae49795e9b17f7cc10f2840fd853c00f3e049a151f0daa4bf02f4703` |
| `src/mc_photogrammetry/pipeline/image_derived_production.py` | `6f722046536fcb0040f689caa3c6c1ea7ffaef3cf37ee4d0c064b709cc31875e` |
| `src/mc_photogrammetry/project/store.py` | `b646084f746df0e7c213d5723f01f156992b23997400476796d7ee0187002b41` |
| `src/mc_photogrammetry/cli.py` | `e2e584fed754e48757ed5b3916fbd365da0e9f1d47f8e65ffc9a5de118472377` |
| `src/mc_photogrammetry/ui/main_window.py` | `84896cda500cb9f5429590be3bf72bca17169968895759a2312c859f226a0ff9` |
| `tests/test_image_derived_session.py` | `3bf77f39b14f7e47c5b14bfcf31c3cd783a08f713c0c2bde255bd6799baee8b5` |
| `tests/test_image_derived_entry.py` | `3705d978287fc6ef76dc2c8f624736f7ae2cd7c4e2baf7fbc760daae821210ea` |
| `tests/test_image_derived_connections.py` | `974ca9c6d4c4186d97963eb1246e86630d1108b777b832aa09f3b6717658e30f` |
| `tests/test_image_derived_pose_graph.py` | `19ebab1811e4e829594d58c440dcef3dae7c40326f3cff603c828795a3619d3f` |
| `tests/test_image_derived_lattice_measure.py` | `fc06c3671952acf727485013d68115b4c021fae8e8e7da41a1a8f7be0046bd38` |
| `tests/test_project_store.py` | `ce5841651c29203ee74e4feeeb12568ec3cb5585dc65e9ec74fe83179480a285` |
| `tests/test_image_derived_production_cli.py` | `a80dbbb79a976e35e8cb5b011465f69039639eab7a3e2bdc8157a5491761faaa` |
| `tests/test_image_derived_delivery_cli.py` | `c1961001f0babb557e6df4ec1da8cb45fca117b5768bba160914b8879a36aff8` |
| `tests/test_desktop_image_derived_route.py` | `874f8bf549036bd36b1e96656e32253b3efcd41fc87407b0f32c8a227f8f21f3` |

## Handoff and scope

Task status updated to `ready_for_review` under the lead's explicit instruction; local task attempts set to 1, representing the same completion attempt 6/8. No route/run records changed. Writes remain the nine source files, nine test files, report, append-only Comms and authorized task status.

Approximately 27 Sol agent-minutes from assignment through this handoff, with no delegated agents. Actual remediation took about eight additional minutes, below its 15-25 minute estimate and within the original two aggregate agent-hour cap; lead review/check evidence is recorded separately in Comms.

No real screenshot bytes, metadata/sidecars/truth/JAR/checkpoint or C1 payload access, inference/retry, capture, Minecraft/Prism/UI launch, mouse/input/audio activity, downloads/installs or scope expansion. Solver, estimator/calibration, renderer, catalogue and export algorithms unchanged. Synthetic temporary project persistence is not product output. C1 consumed/blocked and C2 NO-GO preserved.

Focused synthetic checks pass; independent lead acceptance remains pending. No real-image accuracy, useful geometry for the ordinary scene, C2/release readiness or project-completion claim. STOP; no further runtime or check under this assignment.

---

## Initial implementation record (historical; superseded by handoff above)

Task: `image-only-partial-camera-admission-2026-10-07`, completion attempt 6/8.
Repository: `C:\Users\Likea\Documents\vscode projects\mc photogrammetry project`.
Authority: scoped task and Astra's 2026-10-06 23:34 CDT accepted design.
Status at 2026-10-06 23:55 CDT: **implementation unverified; acceptance not met; STOP for lead review.**

## Check outcome and evidence limitation

Compilation reported `py_compile passed: 18 files; bytecode writes intercepted`.
The sole focused pytest invocation emitted the following retained output:

```text
........................................................................ [ 26%]
............................FF...F...................................... [ 53%]
........F.................................
```

Four failure markers were observed in partial stdout. The final failure total, passing total, traceback, and terminal exit code are unknown. The orchestration call forwarded only stdout and lost the returned running session ID. A subsequent attempt to poll the observed shell PID returned `Unknown process id 43232`; that PID was not a valid retained tool session handle. A read-only process check at 23:50:41 CDT found the observed pytest Python processes had finished. Available tooling could not recover their terminal buffer or exit code. Failure causes or affected test identities are not inferred from progress positions.

The lead explicitly directed that this exact process finish, that no second test invocation occur under the task's one-run constraint, and that only observed evidence be reported. No rerun or post-failure code diagnosis was performed. The lead separately requested correction of contradictory entry-module documentation; only that module docstring was changed after the checks. Its initial-report hash below differs from the compiled/tested entry hash `0a553e627acc5ae141e5c9929451d95decb6bc45ec673ff08edaec8ead31a701`. No repeat compilation was performed after that documentation-only change.

## Implemented contract; not accepted by tests

Nine source files now implement the following intended behavior:

- Session v2 records one ordered full input inventory, typed admission outcomes, original active IDs and a pooled-focal caveat. The digest binds all inputs and outcomes; downstream `frames`, cameras and local cameras expose only the admitted subsequence. All-solved sessions continue to serialize as v1, with separate backward payload/digest/read checks added.
- Only absent estimates become `camera_estimate_unavailable` exclusions. Foreign IDs, malformed/present-null estimates, nonfinite or forbidden camera claims, pixel failures and runtime errors remain hard typed failures. Legitimate unusable partial estimates are retained in in-memory error review evidence.
- Full-input source paths and pixel identity are replayed at entry/store and applicable consumer joins. Matchers, pose stages and lattice measurement retain admitted IDs. The estimator is invoked once on the full inventory; no reduced-input rerun is introduced.
- Zero admitted cameras has a `camera_unavailable` terminal and skips graph/pose/lattice/promotion. One admitted camera retains local review and no pair match or geometry promotion. Existing downstream algorithms and disconnected-component handling are retained for larger admitted sets.
- CLI and desktop summaries expose original/admitted/excluded counts and every exclusion before delivery descriptions.

The outer project schema and `domain/models.py` were not changed. Nested v2 round-trip and source-binding tests were added, but are **not claimed passing**. `tests/test_image_derived_production_delivery.py` was unchanged and included in the focused invocation.

Nine test files were changed to cover solved/missing/solved original-ID gaps, active-only matching, full source inventory, zero/one non-promotion, disconnected components, hard malformed failures, excluded-pixel/reason/partition tampering, legacy v1 payload/digest/read compatibility, v2 store round-trip, and CLI/desktop disclosure. These are synthetic/in-memory fixtures, with temporary synthetic project persistence where needed. Added test presence is not acceptance evidence.

## Commands

Executed once, with `PYTHONDONTWRITEBYTECODE=1`, explicit repository working directory and Windows PowerShell:

```powershell
.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider tests/test_image_derived_session.py tests/test_image_derived_entry.py tests/test_image_derived_connections.py tests/test_image_derived_pose_graph.py tests/test_image_derived_lattice_measure.py tests/test_project_store.py tests/test_image_derived_production_delivery.py tests/test_image_derived_production_cli.py tests/test_image_derived_delivery_cli.py tests/test_desktop_image_derived_route.py
```

Final pytest and containing shell exit codes were not captured.

Immediately before pytest, `.venv\Scripts\python.exe -B -c` invoked `py_compile.compile(p, cfile=p, doraise=True)` for the 18 paths below. The executed Python expression imported `py_compile` and `unittest.mock.patch`, patched `importlib._bootstrap_external._write_atomic` to collect writes instead of writing bytecode, compiled each path and printed the retained 18-file success message. No `.pyc` was written by that compilation. The orchestration did not retain a separately reported compiler exit code. The full shell command text was not retained independently; this paragraph records its observed executable, options, operations and output rather than claiming an exact recovered command string.

Read-only SHA256 evidence was generated using `Get-FileHash -LiteralPath <each path> -Algorithm SHA256`; that command exited 0.

## Initial changed Python file hashes

| Path | SHA256 |
| --- | --- |
| `src/mc_photogrammetry/pipeline/image_derived_session.py` | `e48414bf9eea6a7da7a7cc7d8c19865e0494953ac8796d9f3cd2e57e81b6248e` |
| `src/mc_photogrammetry/pipeline/image_derived_entry.py` | `e22670bd35c48b255650de59da966741b30a17ebf3e27b03a77c9530971316e9` |
| `src/mc_photogrammetry/pipeline/image_only.py` | `ca28afd0e9dc36d2b827a28807b736d635374acb2787506c2d2f3d07c929c93c` |
| `src/mc_photogrammetry/pipeline/image_derived_connections.py` | `b46d51a17f45a292666ef089199dca120cc11b91125d028e5114d0c6a010aedd` |
| `src/mc_photogrammetry/pipeline/image_derived_lattice_measure.py` | `109367b5579374b3c159e0973d43c7350a11d6b080d5f88869cfcfd423da4032` |
| `src/mc_photogrammetry/pipeline/image_derived_production.py` | `6f722046536fcb0040f689caa3c6c1ea7ffaef3cf37ee4d0c064b709cc31875e` |
| `src/mc_photogrammetry/project/store.py` | `b646084f746df0e7c213d5723f01f156992b23997400476796d7ee0187002b41` |
| `src/mc_photogrammetry/cli.py` | `e2e584fed754e48757ed5b3916fbd365da0e9f1d47f8e65ffc9a5de118472377` |
| `src/mc_photogrammetry/ui/main_window.py` | `84896cda500cb9f5429590be3bf72bca17169968895759a2312c859f226a0ff9` |
| `tests/test_image_derived_session.py` | `3bf77f39b14f7e47c5b14bfcf31c3cd783a08f713c0c2bde255bd6799baee8b5` |
| `tests/test_image_derived_entry.py` | `d1a65e964fb807456aba94e9f73a1e2b1cb5602f4f29d7d8d23fd7e36b48cc05` |
| `tests/test_image_derived_connections.py` | `974ca9c6d4c4186d97963eb1246e86630d1108b777b832aa09f3b6717658e30f` |
| `tests/test_image_derived_pose_graph.py` | `fef2643aaceb55e7eab3e996c6ddce422412aa9e93c31d8ffd9229be22c28894` |
| `tests/test_image_derived_lattice_measure.py` | `42f0286d5eb5c529c2c1e64cc730ddb0f149c74a09120ff24538cbbc3c70782c` |
| `tests/test_project_store.py` | `ce5841651c29203ee74e4feeeb12568ec3cb5585dc65e9ec74fe83179480a285` |
| `tests/test_image_derived_production_cli.py` | `a80dbbb79a976e35e8cb5b011465f69039639eab7a3e2bdc8157a5491761faaa` |
| `tests/test_image_derived_delivery_cli.py` | `c1961001f0babb557e6df4ec1da8cb45fca117b5768bba160914b8879a36aff8` |
| `tests/test_desktop_image_derived_route.py` | `874f8bf549036bd36b1e96656e32253b3efcd41fc87407b0f32c8a227f8f21f3` |

Additional owned writes: this report and an append to `AGENT_COMMS.md`. The task record and run records were not modified because they are outside this assignment's write scope.

## Scope, effort and stop

Approximately 19 agent-minutes from 23:36 assignment to this report, with no delegated agents; below the two aggregate agent-hour cap. This is one implementation attempt and one focused test invocation. No model/inference/retry, real screenshot bytes, metadata/sidecars/truth/JAR/checkpoint reads, original or fresh C1 artifact access, sealed benchmark fixture use, capture, Minecraft/Prism/UI launch, mouse/input/audio changes, installs or downloads occurred. Solver, estimator/calibration, renderer, catalogue and export algorithms were not changed. Synthetic temporary test outputs are not product output or a reconstruction result.

No real-image accuracy, C2 readiness, release acceptance or completion claim follows. Lead review must decide whether a separately bounded remediation/check run is justified; this task run is stopped.

