# DS-R4-V2-ROUTEB-FROZEN-IMAGE-R3 - trace-enabled Route B diagnostic (executed once)

Task: `AGENT_COMMS.md` 2026-10-04 LEAD -> DeepSeek `DS-R4-V2-ROUTEB-FROZEN-IMAGE-R3`.
Status: **EXECUTED once, foreground, refused (exit 1); no retry.** No accuracy score,
row count or completion claim (no complete capture was produced).

## 1. Constraints honored

- Exactly **one** diagnostic invocation, foreground, with `.venv\Scripts\python.exe`
  directly. **No bare-`python` attempt**, no retry, no settings/tuning.
- Passed only the three ordered `--image` paths and `--root`. No operator hashes,
  settings, reference/evaluator/truth, litematic/scene/camera/F3/direction/mask or
  coordinates; no caller metadata. The runner parser exposes only `--image`/`--root`
  (`scripts/evaluation/run_route_b_diagnostic.py:1083-1095`).
- No production edit, no Minecraft/launcher, no accuracy scoring.
- Direct-shell restrictions honored: no junction/reparse creation or manipulation, no
  recursive deletion, no encoded/obfuscated commands, no generated
  executables/installers, no background/persistent processes, no Defender/system
  changes, no external temp inspection, no client automation.

## 2. Preflight hash pins (all matched before invocation, and were unchanged after)

| Artifact | SHA-256 | Result |
| --- | --- | --- |
| runner `scripts/evaluation/run_route_b_diagnostic.py` | `050AC0123386333B55E7321FE9D121DCF9CAC3E2635097931D2869BA9D0E8EC4` (50,755 B) | MATCH |
| focused runner tests `tests/test_route_b_diagnostic_runner.py` | `D9A595EA84886B645C95182472D6B2E34EC057C410F9AE47195FFA17B54F04F3` (51,487 B) | MATCH |
| V2 collector `src/mc_photogrammetry/pipeline/candidate_support_provenance_v2.py` | `56A5D3B949028A9C0E363E449805B55368217280EE8A57DFC12A57804B5F28F7` (120,413 B) | MATCH |
| focused V2 tests `tests/test_candidate_support_provenance_v2_preflight.py` | `1670CE39A77613CE67F231EA35F74169FFB5D479CF0B400C24B4D485E646E75F` (102,475 B) | MATCH |
| `templates 2/2026-09-02_13.22.14.png` | `EB22541171E030FF94D6839A6626E7160C69AE3A780F07AEC0298322BE665D46` (1,609,624 B) | MATCH |
| `templates 2/2026-09-02_13.24.24.png` | `E2FD9FD6C83BF9005FE0973987DA9E46D213A4351EC91393CE578F8E6D3961AE` (1,461,426 B) | MATCH |
| `templates 2/2026-09-02_13.24.33.png` | `35D9DEBB206C2505F65D3BE29183F4A4F3D2BF3F9C05644CE75570F48381D1FC` (1,336,191 B) | MATCH |

Root freshness: `.tmp-routeb-v2-frozen-r3-20261004` was **absent** immediately before
invocation (fresh, not reused). No other root was used.

## 3. Invocation (single, foreground)

```
.venv\Scripts\python.exe scripts/evaluation/run_route_b_diagnostic.py ^
  --image "templates 2\2026-09-02_13.22.14.png" ^
  --image "templates 2\2026-09-02_13.24.24.png" ^
  --image "templates 2\2026-09-02_13.24.33.png" ^
  --root .tmp-routeb-v2-frozen-r3-20261004
```

Exit code **1**. stdout **empty** (0 B). stderr **177 B**, exactly one line:

```json
{"detail": "", "field": "ValueError:mc_photogrammetry.pipeline.candidate_support_provenance_v2.__post_init__:820", "reason": "universe_identity_mismatch", "status": "refused"}
```

## 4. Refusal field, including operation/line/row

- Full field: `ValueError:mc_photogrammetry.pipeline.candidate_support_provenance_v2.__post_init__:820`
- Decomposition (format `_raw_failure_field`, `candidate_support_provenance_v2.py:526-543`):
  - canonical exception category `ValueError` - selected by `isinstance` against the
    fixed caught-base tuple (`:444-462`), never a custom class name;
  - module `mc_photogrammetry.pipeline.candidate_support_provenance_v2`
    (56 chars, within the 64-char token cap);
  - operation `__post_init__`; source line `820`;
  - **no `@row` suffix**, i.e. the admitted-row holder was `None`
    (`failure_site.leave_row()`, `:2077`) - this is an **assembly-phase** failure,
    not a row-local one.
- Source attribution: line 820 is
  `raise ValueError("matcher support V2 sidecar page is malformed")` inside
  `MatcherSupportSidecarPageV2.__post_init__` (`:800-834`). This constructor is reached
  from `_assemble_complete_support` via `_pack_sidecar_pages` (`:1719`, `:1562-1580`) ->
  `_make_sidecar_page` (`:1553-1559`).
- The guard at `:812-819` is a disjunction (schema; non-negative int `page_index` and
  `first_admitted_row_index`; non-empty `entries` tuple of `MatcherSupportSidecarEntryV2`;
  first entry `admitted_row_index` equal to the page's first index). The traceback line is
  the `raise`, so the field does **not** discriminate which OR-term fired.
  Residual candidate (inference, not measured): the caller passes `start` as the page's
  first index and a non-empty validated entry slice, and the entry types plus upstream
  `:1709-1713` checks pin the other terms, leaving the
  `entries[0].admitted_row_index == first_admitted_row_index` term. No measured conclusion
  is claimed here.
- `reason` remains the fixed compatibility code `universe_identity_mismatch`
  (broad raw-exception arm `:1979-1988`), not identity evidence. `detail` is empty.

## 5. Stage sequence

Recorded `calls` are not persisted on refusal, so the sequence below is derived, not
emitted by the run:

1. `entry` (`_run_single_pass:458-463`) - accepted (`_entry_refusal` returns `None`).
2. `lattice` (`:465-469`) - status gate passed.
3. `signatures` (`:472-478`) - status gate passed.
4. `graph` (`:481-486`) - returned a non-refusal graph with a trace and attached the
   complete-support sink.

The capture sink fired exactly once (`len(captured) == 1`, `:492-496`), the capture was a
`CompleteSupportCaptureV2` (`:498-499`), and the runner then refused in `replay_capture`
at the capture-status gate (`:534-536`) using the capture root's own `reason`/`field`.
All four stages must therefore have completed for this refusal to be reached.

## 6. Input order

`refusal.json` records the three assigned paths, in the assigned order, with empty
`frame_id`/`sha256` (the runner never pre-hashes inputs):

1. `templates 2\2026-09-02_13.22.14.png`
2. `templates 2\2026-09-02_13.24.24.png`
3. `templates 2\2026-09-02_13.24.33.png`

Frozen settings echoed by the artifact: `ambiguous_margin=0.02`,
`integer_tolerance=0.15`, `max_shift_cells=32`, `min_resultant=0.9`.

## 7. Artifacts, timing, cleanup

- `.tmp-routeb-v2-frozen-r3-20261004/refusal.json` - **571 B**, SHA-256
  `8D0898F2EC1C74C2D0048399D29D9500786AD21587E166953226236A83CA0FFF`,
  mtime `2026-10-04T19:29:46.9727826-05:00`, schema
  `mc-photogrammetry-route-b-diagnostic-runner-refusal-v1`.
- Wall clock (foreground, operator-measured): **558.945 s**. The runner's internal
  `elapsed_seconds` is not persisted on refusal.
- Cleanup: the root holds **exactly one file** (`refusal.json`). Verified absent:
  `record.json`, `route_b.mcscan`, `route_b_staging/`, `record.json.partial`,
  `refusal.json.partial`, `route_b.mcscan.tmp`.
- Not produced (so no score/count, no digests): no complete capture, no
  `expected_row_count`/`row_count`, no whole-universe/input-match/record digests, no
  `stage_digests`, no witness content.

## 8. Hash ledger (this report's inputs)

Post-run re-verification of the four code pins and three PNG pins: all seven MATCH
(unchanged). This report was the only file written by the task before the comms result.

Stopped for lead/Luna review; no accuracy or downstream-certification claim.
