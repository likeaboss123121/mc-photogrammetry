# DS-41 - Hidden-desktop window move race

Date: 2026-09-12. Scope: executor-owned virtual-desktop movement and its fake-runner contract tests. No Minecraft/Prism launch, DSH operation, Defender change, or git commit.

## Change

`PrismCaptureLauncher` now passes the spawned Prism process PID to the desktop controller immediately after `Popen` and at every poll. The PowerShell helper starts from that PID, constructs its descendant PID set from the Windows process table, and captures nonzero `MainWindowHandle` values only for those PIDs. It records each process start time and rechecks both process identity and the same handle immediately before `Move-Window`, preventing a reused PID or replaced window handle from being treated as the originally captured window.

Only HRESULT `0x8002802B` (`TYPE_E_ELEMENTNOTFOUND`) is eligible for race suppression, and only when post-error revalidation confirms that the process vanished/replaced or its handle became stale. A live process with the same start time and handle keeps that error fatal; all other HRESULTs, PowerShell/module errors, and command errors remain fail-closed. The helper reports both its pre-move and post-move desktop indices. The controller requires both to equal the index saved at preflight, so movement fails if the operator changes desktops before or during a poll. No active-desktop switch, activation, focus, or input call was added.

Tests inspect the emitted script contract and use an injected fake command runner; they do not execute PowerShell or create windows. Launcher tests verify the exact spawned PID is passed on the immediate move and each poll, and existing failure teardown/lifecycle paths remain covered.

## Validation

Command:

```text
.\.venv\Scripts\python.exe -m pytest tests\test_capture_batch_executor.py -k "desktop or launcher_moves_windows_after_popen or launcher_fails_and_tears_down_when_a_poll_move_fails or lifecycle_order_uses_a_distinct_worker_per_scene" -o addopts= -p no:cacheprovider -q
```

Result: **19 passed, 116 deselected**. Coverage includes PID/descendant scoping, zero-handle exclusion, exact stale-race condition, fatal generic move errors, changed desktop index before/during movement, launcher PID propagation, forbidden focus/input tokens, and lifecycle teardown. PowerShell syntax/runtime and a real virtual desktop were intentionally not exercised.

The current DS-40 jar identity was also verified locally and corrected in the DS-40 report: `mc-photo-capture-0.1.0.jar`, 71,226 bytes, SHA-256 `ff65f4110e8f8422af60c1bbfb00ca0eb0d44fb4f30b4e7cacef3fd5dc3f1419`. This supersedes the stale artifact identity previously recorded there; no jar was copied to a worker.
