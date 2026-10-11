# DS-23 — Label-blind pose-policy audit (DS-23E mechanical finish)

**Agent:** DeepSeek DS-23E · **Date:** 2026-09-12 · **Status:** complete, boundary verified
**Directive:** `AGENT_COMMS.md` 2026-09-12 13:38 CDT (DS-23)
**Artifacts:** `exports/diagnostics/ds23/blind_pose_policy.json`,
`exports/diagnostics/ds23/blind_tape.json`, `exports/diagnostics/ds23/blind_tape.npz`,
`exports/diagnostics/ds23/blind_pose_pairs.csv`,
`exports/diagnostics/ds23/blind_pose_run.log`, `exports/diagnostics/ds23/run_console.txt`
**Script:** `scripts/diagnose_blind_pose_policy.py`
**Selector:** `scripts/blind_pose_policy.py`
**Test:** `tests/test_blind_pose_policy.py` (16 passed)

Mechanical finish only. No production source, artifact, schema, CLI or desktop file was
modified. The diagnostic was not rerun or re-thresholded, and no commit or stage was made.

---

## 1. Scope of this finish

* Removed the source-audit test `test_selector_source_has_no_forbidden_symbols_or_imports`
  from `tests/test_blind_pose_policy.py`. The correspondence-permutation test
  (`test_correspondence_order_permutation_does_not_change_the_decision`) and the
  pair-relabelling test (`test_relabelling_the_pair_identity_cannot_change_the_decision`)
  are retained.
* Ran the focused file with the workspace interpreter and the repository pytest defaults
  disabled. No failing case remained, so no further edit was required.
* Verified the frozen DS-23 diagnostic payload, which is the authority for this milestone.

## 2. Focused test evidence

```
.\.venv\Scripts\python.exe -m pytest tests\test_blind_pose_policy.py -q -p no:cacheprovider -o addopts=
................                                                         [100%]
16 passed in 0.47s
```

`-p no:cacheprovider -o addopts=` disables the repository pytest defaults from
`[tool.pytest.ini_options]`. Exit code 0.

## 3. Frozen DS-23 result: 9 of 9

Source: `exports/diagnostics/ds23/blind_pose_policy.json`
(`schema = ds23-blind-pose-policy-1`, `status = complete`) and
`exports/diagnostics/ds23/blind_pose_run.log`.

`blind_selection`:

| field | value |
|---|---|
| usable | **9** |
| class_correct | **9** |
| of | **9** |
| ds19_9_of_9_survives | **true** |

`threshold` (`all_passed = true`, `status = blind_policy_meets_preregistered_threshold`):

| Preregistered gate | Required | Measured | Pass |
|---|---|---|---|
| reproduce frozen DS-11 oracle baseline | 6/9 | 6/9 | yes |
| blind selection usable | >= 8/9 | **9/9** | yes |
| blind selection yaw class correct | >= 8/9 | **9/9** | yes |
| gauge-aligned baseline error <= 5 deg | >= 8/9 | **9/9** | yes |
| zero regressions among the original six | 0 | **0** | yes |
| API-boundary unexpected reads | 0 | **0** | yes |
| DS-19 equivalence | 9/9 | **9/9** | yes |
| synthetic degeneracy refusals | required | valid 24/24 and parallel-axes 24/24 solved; pure rotation 24/24 refused (`cheirality_failed`) | yes |

Run-tail evidence (`blind_pose_run.log`):

```
[equiv ] clean-room selector vs DS-19 adjudicator on the same geometry: 9/9 identical (<= 1e-09)
[basel ] reproduced the frozen DS-11 oracle baseline (6/9 solved, 6/9 class-correct)
[thresh] blind_policy_meets_preregistered_threshold: baseline 6/9, blind usable 9/9, class-correct 9/9, regressions 0, DS-19 9/9 survives: True
[done  ] baseline 6/9; blind usable 9/9; threshold blind_policy_meets_preregistered_threshold; runtime 14.8s; peak RSS 368.8 MiB
```

The clean-room selector read 12 of 12 whitelisted geometric attributes on each of the nine
pairs with `trace_unexpected_reads = []`, and the frozen tape is digest-pinned before
selection (`blind_tape.json` sha256 `72a1259d…e8a23`, `blind_tape.npz` sha256
`763933ef…603a5`). All nine decisions were frozen before any reference label was read;
the frozen evaluator was used only post-selection.

## 4. Boundary

Production and prior-milestone code are untouched: the selector and the diagnostic are
audit artifacts, and the DS-19 adjudicator remains the frozen comparison. The DS-23
payload is the only authority for the 9-of-9 claim recorded here; this finish read that
payload and did not recompute, rerun or tune it.

## 5. Exact commands

```
.\.venv\Scripts\python.exe -m pytest tests\test_blind_pose_policy.py -q -p no:cacheprovider -o addopts=
```

No full-suite run and no diagnostic run were performed.
