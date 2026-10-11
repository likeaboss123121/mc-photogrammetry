# DS-R4-V2-ROUTEB-FROZEN-IMAGE-R1 — frozen-image Route B diagnostic run record

- Date: 2026-10-04
- Worker: DeepSeek CLI worker (operator-run, foreground)
- Assignment: `AGENT_COMMS.md` "2026-10-04 LEAD -> DeepSeek - DS-R4-V2-ROUTEB-FROZEN-IMAGE-R1"
  (entry immediately preceding the LEAD report correction; runner accepted by Luna at hash
  `331D0557...CAE16A` in the final re-review)

## Disposition

**REFUSED / ABSTAINED at the runner's entry gate.** Exactly one foreground invocation; no retry,
no settings tuning, no production/default change, no truth or accuracy scoring. Exit code `1`.
Per the assignment ("Stop and report even if it abstains") the diagnostic is closed here.

## Invocation

```
.venv\Scripts\python.exe scripts/evaluation/run_route_b_diagnostic.py ^
  --image "templates 2/2026-09-02_13.22.14.png" ^
  --image "templates 2/2026-09-02_13.24.24.png" ^
  --image "templates 2/2026-09-02_13.24.33.png" ^
  --root .tmp-routeb-v2-frozen-r1-20261004
```

- Single run, foreground, run from the project root.
- Only ordinary screenshot paths (three, in the assignment's exact order) and the fresh output
  root were passed. No hashes, reference, Litematic, scene, cameras, F3, directions, masks,
  coordinates, evaluator inputs, camera metadata or hints were passed to the runner.
- Runner under test: `scripts/evaluation/run_route_b_diagnostic.py`
  SHA-256 `331D0557EA2AEBA436BAA25D364386F63F90C9C5383066F5CD55CCF2F4CAE16A` (48,505 B),
  matching the accepted runner hash before and after the run (unchanged).

## Inputs (operator-side provenance check, not passed to the runner)

Order exactly as assigned; all three matched the assignment hashes and were unchanged after the
run.

| order | path | bytes | SHA-256 |
|---|---|---|---|
| 1 | `templates 2/2026-09-02_13.22.14.png` | 1,609,624 | `EB22541171E030FF94D6839A6626E7160C69AE3A780F07AEC0298322BE665D46` |
| 2 | `templates 2/2026-09-02_13.24.24.png` | 1,461,426 | `E2FD9FD6C83BF9005FE0973987DA9E46D213A4351EC91393CE578F8E6D3961AE` |
| 3 | `templates 2/2026-09-02_13.24.33.png` | 1,336,191 | `35D9DEBB206C2505F65D3BE29183F4A4F3D2BF3F9C05644CE75570F48381D1FC` |

These hashes were computed by the operator for provenance only; the runner never received them
(its CLI exposes no hash/identity argument and it derives source identity internally).

## Result

- Exit code: `1` (typed `DiagnosticRefusal`, JSON printed to stderr).
- stderr JSON (verbatim, PowerShell `NativeCommandError` wrapper text omitted):

  ```json
  {"detail": "image-derived session and relative camera evidence are established; V1/V2/V3/M5 geometry is unavailable from images alone in normal mode", "field": "entry", "reason": "unavailable", "status": "refused"}
  ```

- stdout: empty.
- Stage-call sequence: **`entry` only.** The refusal was raised by the entry status gate
  (`_status_refusal("entry", entry)`) after `build_image_derived_entry`, so lattice, signatures
  and graph attribution were never invoked. No signature-witness pass and no complete-support
  capture exist for this run.
- Ingress pins: **none recorded.** The typed refusal record's `inputs[]` carries only the three
  ordered paths with empty `frame_id`/`sha256`.
- Row count / digests: **not produced.** No V2 complete-support capture, therefore no
  `expected_row_count`, `whole_universe_identity_sha256`, `input_match_identity_sha256`,
  `failed_assignment_sha256`, `record_sha256` or `record.json`.

## Refusal artifact

- `.tmp-routeb-v2-frozen-r1-20261004/refusal.json` — 610 B —
  SHA-256 `791B5F53572C6BF7C2E32CACCAB1CB01ACEF4102DAE5AB3013F5FF38E082FAE6`
- Content: schema `mc-photogrammetry-route-b-diagnostic-runner-refusal-v1`, status `refused`,
  reason `unavailable`, field `entry`, `settings` = frozen production defaults
  (`max_shift_cells` 32, `integer_tolerance` 0.15, `min_resultant` 0.9, `ambiguous_margin` 0.02),
  and the three ordered input paths with empty `frame_id`/`sha256`.

## Cleanup / atomicity

The output root contains exactly one file (`refusal.json`). Verified absent after the run:
`record.json`, `route_b.mcscan`, `route_b_staging/`, `record.json.partial`,
`refusal.json.partial`. No staged project, no `.partial`, no staging directory was left behind,
so the runner's nonrecursive cleanup and failure atomicity held on this refusal.

## Timings

- Wall clock (operator, foreground, includes interpreter startup): **29.865 s**.
- Runner `elapsed_seconds` unavailable (only reported on a completed capture).

## Interpretation and boundary

The ordinary image-only Route B entry establishes session and relative camera evidence but
reports V1/V2/V3/M5 geometry unavailable from images alone in normal mode, so the accepted
runner refuses at its entry gate and produces no V2 complete-support capture. This is a
refinement-only diagnostic abstention: no accuracy or truth scoring, no settings tuning, no
retry, no production/default source change, and no Minecraft/launcher or
evaluator/truth/camera/mask/reference input was used.
