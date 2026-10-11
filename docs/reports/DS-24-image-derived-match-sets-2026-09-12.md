# DS-24 — Image-derived match sets under the frozen DS-23 blind selector

**Agent:** DeepSeek (DS-24 plumbing repair / unchanged rerun) · **Date:** 2026-09-12
**Status:** diagnostic complete; preregistered promotion **failed** (retained diagnostic-only)
**Directive:** resume the paused DS-24 diagnostic: fix the saved script's payload plumbing without
tuning after labels, rerun unchanged, record the reproduced gate.
**Artifacts:** `exports/diagnostics/ds24/evaluation.json`,
`exports/diagnostics/ds24/match_sets.json`, `exports/diagnostics/ds24/match_sets.npz`,
`exports/diagnostics/ds24/decisions.json`, `exports/diagnostics/ds24/freeze.json`,
`exports/diagnostics/ds24/pairs.csv`, `exports/diagnostics/ds24/run.log`
**Script:** `scripts/diagnose_ds24_image_derived_pose.py`
**Test:** `tests/test_ds24_image_derived_pose.py` (2 cases)

No production source, evaluator, threshold or policy was modified. No commit or stage was made.

---

## 1. Plumbing bug fixed

The payload's `per_variant` block serialized each variant's rows from a leaked name. The earlier
CSV loop

```python
for variant, rows in variant_eval["rows"].items():
    ...
```

leaves `rows` bound to the **last** variant (`C_bestof_ratio_epipolar`) after the loop. The payload
comprehension then iterated `for row in rows` for every `variant`, so all three variants serialized
C's rows: same selected counts, same decisions, same evaluation.

Fix (only that input changed): a small payload-row helper reads the rows **by name**:

```python
def payload_variant_rows(variant, variant_eval, references, fraction_by_slot):
    prefix = variant[0]
    return [ ... for row in variant_eval["rows"][variant]]
```

and the payload calls `payload_variant_rows(variant, variant_eval, references, fraction_by_slot)`.
No threshold, gate, selector, evaluator or Phase A/B/C behavior was touched.

## 2. Sentinel test

`tests/test_ds24_image_derived_pose.py::test_payload_serializes_each_variants_own_rows_with_distinct_counts`
builds a synthetic scorecard with deliberately unequal row counts (A = 3, B = 2, C = 1). The old
leaked-`rows` payload would serialize count 1 for all three; the fixed helper serializes
`{A: 3, B: 2, C: 1}` with each variant's own `selected` values. Distinct row counts are the
sentinel.

Against the **stale** payload written by the paused run, the real-artifact case failed on the bug:

```
assert [22, 20, 15, 22, 8, ...] == [106, 176, 14...152, 235, ...]
At index 0 diff: 22 != 106
```

i.e. A/B carried C's selected counts. After the unchanged rerun the case passes.

`test_real_evaluation_selected_counts_match_match_sets_metadata` reads the regenerated
`evaluation.json` and `match_sets.json` and asserts, per variant and slot,
`per_variant[variant].rows.selected == variants[variant].selected_correspondences` (and that the
three frozen vectors are distinct). Corrected payload:

| variant | per-variant `selected` (slots 0..8) | equals tape |
|---|---|---|
| A_reciprocal_ds14 | 106, 176, 142, 210, 152, 235, 243, 248, 216 | yes |
| B_directed_ds18 | 192, 274, 200, 316, 258, 320, 336, 258, 258 | yes |
| C_bestof_ratio_epipolar | 22, 20, 15, 22, 8, 32, 56, 36, 14 | yes |

## 3. Unchanged rerun and digest verification

Run once with the workspace interpreter, no edits after the run. The match-set arrays digest is
**unchanged**:

```
[tape  ] wrote match_sets.json (43044 B) and match_sets.npz (774753 B); arrays digest 7bd6df7223e79a4e...
[tape  ] re-read frozen match_sets.json + match_sets.npz; array digest 7bd6df7223e79a4e... verified
```

`freeze.json` is byte-identical to the paused run (`bc0c1c2a17033a99…`, file sha256
`bc0c1c2a17033a99364c8b93fd1600f217ee058a54c1e4d2a927a5f86cf62169`), and the frozen decisions
digest is also unchanged (`088870d616df8963…`). Only `evaluation.json` and `pairs.csv` timestamps
moved; the CSV content path was already correct.

## 4. Reproduced preregistered result

From `evaluation.json` + `run.log` (schema `ds24-image-derived-pose-policy-1`, `status = complete`):

| Preregistered gate | Required | A | B | C |
|---|---|---|---|---|
| usable | >= 8/9 | 1 | 1 | 0 |
| yaw class correct | >= 8/9 | 4 | 1 | 0 |
| false solves | <= 1 | 4 | 3 | 1 |
| regressions among DS-14's class-correct pairs | 0 | 3 | 3 | 4 |
| synthetic degeneracy controls | required | ok | ok | ok |
| **passed** | | **no** | **no** | **no** |

`threshold.status = no_variant_meets_preregistered_promotion_diagnostic_only`,
`promoted_variants = []`. Reproduction checks pass: DS-14 baseline 5/9 solved, 4/9 class-correct
(`reproduced = true`); DS-23 oracle tape DS-11 6/9 solved / 6/9 class-correct and blind selector
9/9 usable (`reproduced = true`); reference class floor/direct disagreements 0/9. Synthetic controls:
valid baseline 24/24 solved, parallel axes 24/24 solved, pure rotation 24/24 refused
(`cheirality_failed`). Runtime 29.2 s, peak RSS 433.9 MiB (limit 1.5 GiB). 27 decisions, zero
unexpected trace reads. Truth fraction available, alignment median 1.146 px.

Consistent with the paused-run probe: DS-23's 9/9 does **not** transfer to image-derived
correspondences. The repaired payload now reports each variant's own numbers rather than a
C-only copy; the failed gate is unchanged and is not production-promoted.

## 5. Boundary

The diagnostic and its payload are audit artifacts. The evaluator (`diagnose_match_attribution`,
`diagnose_connected_pairs`) and the selector (`blind_pose_policy`) were digest-verified and
untouched; their preregistered digests still match. No label entered Phase A or B (trace reads are
exactly the 12 whitelisted geometric attributes). No tuning occurred after reference access.

## 6. Tests and exact commands

```
.\.venv\Scripts\python.exe -m pytest tests\test_ds24_image_derived_pose.py tests\test_blind_pose_policy.py -q -p no:cacheprovider -o addopts=
18 passed
```

```
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider -o addopts=
715 passed, 1 skipped in 175.59s
```

`-p no:cacheprovider -o addopts=` disables the repository pytest defaults; the single skip is the
pre-existing external-symlink environment case. Exit code 0 for both runs.

```
.\.venv\Scripts\python.exe scripts\diagnose_ds24_image_derived_pose.py
exit 0; arrays digest 7bd6df7223e79a4e...
```
