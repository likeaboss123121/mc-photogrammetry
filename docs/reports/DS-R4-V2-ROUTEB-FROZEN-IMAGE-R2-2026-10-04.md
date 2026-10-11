# DS-R4-V2-ROUTEB-FROZEN-IMAGE-R2 — repaired frozen-image Route B diagnostic run record

- Date: 2026-10-04
- Worker: DeepSeek CLI worker (operator-run, foreground)
- Assignment: `AGENT_COMMS.md` "2026-10-04 LEAD -> DeepSeek - DS-R4-V2-ROUTEB-FROZEN-IMAGE-R2"
- Runner: `scripts/evaluation/run_route_b_diagnostic.py`
  SHA-256 `050AC0123386333B55E7321FE9D121DCF9CAC3E2635097931D2869BA9D0E8EC4` (accepted by Luna)
- Focused tests: `tests/test_route_b_diagnostic_runner.py`
  SHA-256 `D9A595EA84886B645C95182472D6B2E34EC057C410F9AE47195FFA17B54F04F3`

## Disposition

**REFUSED at the capture-status replay gate. Exit code `1`.** One diagnostic execution; no retry,
no settings tuning, no production/default edit, no accuracy or truth scoring, no
Minecraft/launcher. Per the assignment the diagnostic is closed here even though it abstained.

One **pre-diagnostic environment abort** preceded it and is recorded for transparency (see
"Invocation"): a bare `python` invocation failed at `_default_api()` import (`ModuleNotFoundError:
numpy`) before any stage was entered, produced no diagnostic result and left no artifact. The
diagnostic itself was then executed exactly once with the project interpreter.

## Preflight (operator-side, never passed to the runner)

Verified before invocation; all pins matched, and all were re-verified unchanged after the run.

| item | SHA-256 |
|---|---|
| runner | `050AC0123386333B55E7321FE9D121DCF9CAC3E2635097931D2869BA9D0E8EC4` |
| focused test | `D9A595EA84886B645C95182472D6B2E34EC057C410F9AE47195FFA17B54F04F3` |
| `templates 2/2026-09-02_13.22.14.png` | `EB22541171E030FF94D6839A6626E7160C69AE3A780F07AEC0298322BE665D46` |
| `templates 2/2026-09-02_13.24.24.png` | `E2FD9FD6C83BF9005FE0973987DA9E46D213A4351EC91393CE578F8E6D3961AE` |
| `templates 2/2026-09-02_13.24.33.png` | `35D9DEBB206C2505F65D3BE29183F4A4F3D2BF3F9C05644CE75570F48381D1FC` |

Input sizes: 1,609,624 / 1,461,426 / 1,336,191 bytes. These hashes were computed by the operator
for provenance only; the runner's CLI exposes no hash/identity argument and derives source
identity internally.

## Invocation

Pre-diagnostic environment abort (no stage entered, no result, no artifact):

```
python scripts/evaluation/run_route_b_diagnostic.py --image ... --root .tmp-routeb-v2-frozen-r2-20261004
# exit 1; ModuleNotFoundError: No module named 'numpy' raised in _default_api() import
# (runner :313-324, reached from _resolve_api :327-329) -> no entry call, empty root left behind
```

The lead then appended "2026-10-04 LEAD correction - DS-R4-V2-ROUTEB-FROZEN-IMAGE-R2"
(`AGENT_COMMS.md:7114-7115`), which classifies that abort as an operator setup failure, supersedes
the no-retry clause **only** for the pre-pipeline environment abort, authorizes exactly one actual
pipeline invocation with `.venv\Scripts\python.exe` after re-confirming the pinned hashes and an
empty assigned root, and forbids any retry after that one invocation. This record follows that
correction: the same pins were re-verified, the assigned root was still empty, the single
venv-backed invocation below is the only diagnostic execution, and no retry followed it.

Single diagnostic execution (foreground, project root):

```
.venv\Scripts\python.exe scripts/evaluation/run_route_b_diagnostic.py ^
  --image "templates 2/2026-09-02_13.22.14.png" ^
  --image "templates 2/2026-09-02_13.24.24.png" ^
  --image "templates 2/2026-09-02_13.24.33.png" ^
  --root .tmp-routeb-v2-frozen-r2-20261004
```

Only three ordinary screenshot paths, in the assigned order, and the fresh output root were passed.
No hash, settings, reference, Litematic, scene, camera, F3, direction, mask, coordinate, evaluator
or other metadata was passed to the runner. The assignment root was created empty by the aborted
attempt and remained empty; the runner's own freshness pre-check (`_prepare_root`) accepted it,
and the diagnostic run created the only artifact in it.

## Result

- Exit code: `1` (typed `DiagnosticRefusal`, JSON printed to stderr; stdout empty).
- stderr JSON (verbatim, PowerShell `NativeCommandError` wrapper text omitted):

  ```json
  {"detail": "", "field": "ValueError", "reason": "universe_identity_mismatch", "status": "refused"}
  ```

- **Stage-call sequence: `entry` -> `lattice` -> `signatures` -> `graph`.** All four stages were
  invoked. The refusal was raised only afterwards, in `replay_capture`, so the chain completed
  before the identity/replay gate. No stage was re-entered and no second pass occurred.
- Per-stage outcome:
  - `entry`: **accepted** by the repaired `_entry_refusal` adapter (graph and pose-graph status
    `ok`; the informational `geometry_unavailable` terminal proceeded). Not refused.
  - `lattice`: **assembled** (`_status_refusal("lattice", ...)` did not fire; else the refusal
    field would have been `lattice`).
  - `signatures`: **assembled** (same argument; field would have been `signatures`).
  - `graph`: **solver abstained** (the abstention branch at `image_derived_graph.py:5199-5201`
    was reached) and the attached complete-support sink fired **exactly once** (otherwise the
    runner would have raised `capture_unavailable`, and it did not). The sink returned a
    **refused V2 capture**, which is the refusal that was finally reported.
- Refusal provenance (source): `replay_capture` refuses a non-`complete` capture at
  `scripts/evaluation/run_route_b_diagnostic.py:534-536` with the capture root's own
  `reason`/`field`. The capture root's pair
  `reason="universe_identity_mismatch"`, `field="ValueError"` is minted by
  `refused_complete_support` at `src/mc_photogrammetry/pipeline/candidate_support_provenance_v2.py:1835-1842`,
  the broad `except (AttributeError, TypeError, ValueError, ...)` arm of
  `collect_complete_candidate_support`, which coerces a **raw** `ValueError` escaping
  `_collect_complete_support` into `universe_identity_mismatch` / `type(error).__name__`
  (`"ValueError"`). The typed arm (`except CandidateSupportError`, `:1835-1837`) is listed first,
  and `CandidateSupportError` is itself a `ValueError` subclass
  (`candidate_support_provenance.py:112`); the observed field is nonetheless the literal
  `"ValueError"`, so the escaping error was a raw `ValueError`, not a typed
  `CandidateSupportError`. The sink was called from the abstention branch at
  `image_derived_graph.py:5183-5198` via
  `collect_route_b_diagnostic_support` (`candidate_support_provenance_v2.py:1960-1993`).
- Ingress pins/order: **none persisted.** The typed refusal record's `inputs[]` carries only the
  three ordered paths with empty `frame_id`/`sha256` (no `byte_sha256` pins), because the success
  record was never published. Order is exactly as assigned.
- Row count / digests: **not produced.** No `record.json` and no `route_b.mcscan` exist, so there is
  no `expected_row_count`/`row_count`, no `whole_universe_identity_sha256`, no
  `input_match_identity_sha256`, no `failed_assignment_sha256`, no `record_sha256`, no
  `stage_digests`, no signature-witness content and no `record_file_sha256`. The refused capture is
  a zero-row refusal (`candidate_support_provenance_v2.py:1037-1044`), so it carries no admitted
  universe to report.

## Refusal artifact

- `.tmp-routeb-v2-frozen-r2-20261004/refusal.json` — 494 B —
  SHA-256 `70CFE6B579366694BEB43F76D22D3EA3A40CE0C329C70168756788B997355D0B`
- Content: schema `mc-photogrammetry-route-b-diagnostic-runner-refusal-v1`, status `refused`,
  reason `universe_identity_mismatch`, field `ValueError`, detail `""`, `settings` = frozen
  production defaults (`max_shift_cells` 32, `integer_tolerance` 0.15, `min_resultant` 0.9,
  `ambiguous_margin` 0.02), and the three ordered input paths with empty `frame_id`/`sha256`.

## Cleanup / atomicity

The output root contains **exactly one file** (`refusal.json`). Verified absent after the run:
`record.json`, `route_b.mcscan`, `route_b_staging/`, `record.json.partial`,
`refusal.json.partial`, `route_b.mcscan.tmp`. No staged project, no `.partial` and no staging
directory was left behind, so the runner's nonrecursive cleanup and failure atomicity held on this
capture-level refusal.

## Timings

- Diagnostic wall clock (operator, foreground, includes interpreter startup): **546.028 s**.
- Pre-diagnostic environment abort: 0.102 s.
- Runner `elapsed_seconds` unavailable (only returned on a completed capture); the refusal record
  does not carry it.

## Interpretation and boundary

This is the first frozen-image run to pass the repaired entry adapter and traverse the full
ordinary image-only chain: entry accepted, lattice assembled, signatures assembled, graph reached,
solver abstained exactly once and the V2 complete-support sink fired exactly once. The chain then
stopped at the next gate the earlier read-only audit flagged as uncertified: the assembled V2
capture is itself a typed refusal produced by a **raw `ValueError`** escaping
`_collect_complete_support` and coerced to `universe_identity_mismatch`/`ValueError`. That coercion
masks the true failing field/row, so this run alone cannot name the offending row or invariant; no
re-run or instrumentation was performed to localize it. The result is a diagnostic abstention, not
an accuracy or truth result: no truth/evaluator/reference/litematic/scene/camera/F3/direction/mask/
coordinate input, no settings tuning, no retry, no production/default source change, and no
Minecraft/launcher or client automation was used.
