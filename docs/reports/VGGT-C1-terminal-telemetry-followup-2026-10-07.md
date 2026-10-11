# C1 terminal telemetry follow-up

Implemented the narrow residual from Astra's R1–R4 ACCEPT under `completion-c1-terminal-telemetry-2026-10-07.md`, completion-plan developer attempt 4/8. Completed on 2026-10-06 approximately 21:45–21:48 CDT, about three aggregate agent-minutes within the one-hour cap. Ready for lead verification; STOP. C1 remains **BLOCKED/consumed**, with no inference/retry, reconstruction accuracy, C2, capture, integration, or promotion claim.

`ReceiptPublisher.finalize_telemetry` shares the existing publisher lock and permits only `wall_seconds`, `cpu_seconds`, `peak_rss_bytes`, and `new_disk_bytes`. Values must be finite, nonnegative numbers. It merges counters into the publisher's existing snapshot without changing terminal priority or accepting caller status, failure, detail, stage, frame, or arbitrary fields. Strict JSON serialization precedes publication; the persisted snapshot is returned after the writer succeeds. Main first submits its final status through the existing sticky publication path, then updates its in-memory state from telemetry finalization's returned snapshot. Prior R1–R4 behavior and all thresholds remain unchanged.

Added exactly two parameterized synthetic regression cases: numeric BLOCKED with zero initial counters and resource BLOCKED. Both prove nonzero final counters are persisted while original status/failure/detail/stage/frame and terminal priority remain unchanged. Arbitrary and stale terminal fields are ignored, subsequent COMPLETE/progress cannot overwrite BLOCKED, and nonfinite final telemetry is rejected without a write. All writers and artifacts are in memory; runner main and model are never invoked.

Verification:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& '.\.venv-vggt\Scripts\python.exe' -B -m pytest -p no:cacheprovider tests/test_learned_geometry_adapter.py
```

**58 passed in 0.72 seconds.** `py_compile.compile(..., doraise=True)` passed for both changed Python files with `_write_atomic` intercepted and source-path `cfile`, producing no bytecode files or cache directories. No further test expansion or broad suite was run.

Exact write scope and final hashes:

| File | SHA256 |
| --- | --- |
| `scripts/evaluation/run_learned_geometry_dev.py` | `267663c9d17842dc6583e3226c6d061718e79ae4141f0ff64e0c030716f9797a` |
| `tests/test_learned_geometry_adapter.py` | `0c0c134fee112ee6fafcec71d2a1083967c6c77e11d19640fd056c6a0dc88c8a` |
| `docs/reports/VGGT-C1-terminal-telemetry-followup-2026-10-07.md` | Supplied separately in the lead handoff to avoid self-reference. |

Only these three files were mutated. Existing unrelated dirty state was preserved. Task, run, and AGENT_COMMS updates are outside this exact write scope and remain for the lead. No consumed C1 artifact/log/report, prior correction report, checkpoint, screenshot, dataset metadata/truth/mask/manifest, or evaluator payload was accessed or changed. No runner-main/model/inference/retry, package/environment/cache mutation, install/download, capture, Minecraft/Prism/Java/UI/input/audio, driver/system action, cleanup, or delegation occurred. No fresh immutable-artifact verification is claimed.
