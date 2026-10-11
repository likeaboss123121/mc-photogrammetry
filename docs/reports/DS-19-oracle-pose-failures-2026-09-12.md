# DS-19 — Oracle-edge pose-adjudication failures

**Agent:** DeepSeek DS-19B · **Date:** 2026-09-12 · **Status:** complete, criterion met
**Directive:** `AGENT_COMMS.md` 2026-09-12 13:04 CDT (assignment) + 2026-09-12 13:16 CDT
(correction after lead interrupt)
**Artifacts:** `exports/diagnostics/ds19/oracle_pose_failures.json`,
`exports/diagnostics/ds19/oracle_pose_pairs.csv`,
`exports/diagnostics/ds19/oracle_pose_run.log`
**Script:** `scripts/diagnose_oracle_pose_failures.py`
**Test:** `tests/test_oracle_pose_diagnostic.py` (8 passed)

Evaluation-only. No production, DS-11..DS-18, descriptor, schema, CLI or desktop source,
artifact or threshold was modified. No delegation, stage, commit or self-chain.

---

## 1. Headline

The nine DS-16B oracle-edge failures were **not** a baseline-conditioning, cheirality,
parallax or correspondence-geometry problem. They are **yaw-class scoring**: in every
repaired pair the true yaw class passes every frozen DS-11 gate, and the frozen
residual-margin rule (`ambiguous_yaw_class`) discards it anyway.

A physically justified adjudicator that changes only the class-selection rule, the
consensus-subset baseline refit and the baseline sign rule — every DS-11 gate degree
untouched — now reads **9/9 usable poses, 9/9 yaw class correct, 0 regressions among the
original six, synthetic degeneracy refusal intact**, as scored exclusively by the frozen
DS-16B evaluator.

| Preregistered criterion | Required | Measured | Pass |
|---|---|---|---|
| usable poses | ≥ 8/9 | **9/9** | yes |
| yaw class correct | ≥ 8/9 | **9/9** | yes |
| gauge-aligned baseline error | ≤ 5° | **0.066°–0.637°** (all 9) | yes |
| regressions among the original six | 0 | **0** | yes |
| synthetic degeneracy refusal | required | valid 24/24, parallel-axes 24/24, pure-rotation 24/24 refused | yes |

Reproduction: the frozen 6/9 DS-11 oracle read-out is reproduced exactly
(best-of 6 solved / 6 class-correct, full 6 solved, reference 6 class-correct), and the
script **aborts** if it is not (frozen guard, unchanged).

`status = adjudicator_meets_preregistered_threshold`.

---

## 2. The correction and the boundary it draws

The 13:16 CDT correction is binding and is implemented literally:

* **The DS-16B evaluator is frozen.** The diagnostic no longer scores the adjudicated
  pose with any local formula. It builds a `PairPose`-shaped hand-over and calls
  `diagnose_match_attribution._evaluate_pose` — the exact function DS-16/DS-16B used —
  through the module object (`_frozen_evaluator._evaluate_pose`), then copies its
  read-out verbatim. A test monkeypatches that module attribute and asserts the values
  pass through field-for-field, so re-deriving the metric locally cannot creep back in.
* **The reference yaw class is frozen.** `diagnose_connected_pairs._reference_class`
  is used unchanged: `direct` = the quadrant minimising the matrix rotation error,
  `floors` = the manifest-floor difference.
* **A floor/reference disagreement is reported, never accepted.** When
  `direct != floors` the pair is diagnosed `evaluator_ambiguity`; the disagreement is not
  promoted into an alternate correct label and cannot satisfy promotion. On these nine
  pairs `direct == floors` for all nine, so **no evaluator-ambiguity disagreement occurred**
  (log line `[evalu ] floor/reference-class disagreements: 0/9`; the per-pair `direct` and
  `floors` values are recorded in the payload's `evaluator_class_check` rather than
  silently omitted).
* **The only addition is the preregistered conjunction**
  `usable = class_equals_expected ∧ within_5deg ∧ baseline_error_degrees_gauge_aligned ≤ 5°`,
  built from the frozen evaluator's own outputs. No gate threshold moved.

### 2.1 The bug this exposed in the interrupted partial run

The partial run left in `exports/diagnostics/ds19/` (13:14) read `adj SOLVED (cur 2)` for
all nine pairs but `usable 0/9` with **rotation errors of exactly 90°/180°**. Root cause,
reproduced before the rewrite with a read-only probe and now pinned by certificate:

* the adjudicator's `baseline_direction` is a **right-frame** vector, exactly like
  `solve_pair`'s `PairPose.baseline_direction`, while the frozen evaluator's
  `baseline_true` is the **world** camera-to-camera vector;
* the partial offline scorer re-derived the rotation as
  `R_right_model @ Ry90(class) @ R_left_model.T`, which lands in a different gauge from
  the production pose and turns a 0.25° pose into a 180° "error".

Both are fixed by using the frozen evaluator and the production pose conventions
(`R_right_est @ Ry90(class) @ R_left_est.T` for the relative rotation; the adjudicator's
own `-(R_right_est @ Ry90(class).T @ t)` baseline vector). The in-run
**gauge hand-over self-check** now compares the adjudicated and frozen DS-11 rotation
errors whenever both solve on the same class: **6 comparable, 6 agree (gap 0.0000°),
0 disagree**; the remaining three pairs are the ones the production best-of reader never
solved, so no comparison exists. That check is reported but is *not* part of the promotion
criterion.

Certificate recorded in the payload (`artifacts`):

| Artifact | bytes | sha256 |
|---|---|---|
| `oracle_pose_failures.json` | 373965 | `0664facd955676a25ba302c9361fbcc57746d56595e94afdc97c2c9204002e79` |
| `oracle_pose_pairs.csv` | 6183 | `c54fb7a1fd908654ef5c330d44921016ce732d59d352eb6bfdd89fbbebfd14d6` |
| `oracle_pose_run.log` | 2735 | `cb36fc83bd8c0292b83efa0cc13ede969209a5c560f674aeef51d04731c1f8f4` |

Frozen inputs (all four verified before any label; mismatch aborts):

| Input | sha256 | ok |
|---|---|---|
| `exports/diagnostics/ds16/match_candidates.json` | `e062b985…e02d93` | yes |
| `exports/diagnostics/ds16/match_candidates.npz` | `3f7b4555…2a8ed4` | yes |
| `exports/diagnostics/ds16/match_attribution.json` | `72da240a…a7a038` | yes |
| `exports/diagnostics/ds16b/match_attribution_sweep.json` | `39729450…9f1cf0` | yes |

Reference inputs (litematic / manifest / scene digests) are recorded per fixture in the
payload's `reference_inputs`.

---

## 3. Attribution

Cell, rule and pair set are frozen: `d0.5_n30` (distance ≤ 0.5 blocks, face-normal
cos ≥ cos 30°, render scale 0.35), the exact nine T2/T6 pairs. Alignment gate:
median frame re-projection error **1.1458 px** (≤ 10.0). Oracle truth counts reproduce
DS-16 and DS-16B exactly on every pair (guarded).

Production failure tally (why the frozen DS-11 best-of reader did not class-match):

| Production failure | Pairs |
|---|---|
| `yaw_class_scoring` | 3 |
| none (already class-correct) | 6 |

Adjudicated tally: `adjudicated_solved` 6, `yaw_class_scoring` 3 — i.e. the six pairs the
production reader already had stay usable, and the three it refused are repaired.

The three attributed failures are all the same mode:

| Pair | true class | true-class gate read-out | frozen refusal | adjudicated |
|---|---|---|---|---|
| 13.22.14→13.24.24 | 3 | passes every gate, 33 inliers, median ρ 0.274° | `ambiguous_yaw_class` | class 3, usable |
| 13.22.14→13.24.33 | 1 | passes every gate, 707 inliers, median ρ 0.288° | `ambiguous_yaw_class` (full edges: `cheirality_failed`) | class 1, usable |
| 18.01.09→18.01.14 | 2 | passes every gate, 264 inliers, median ρ 0.248° | `ambiguous_yaw_class` | class 2, usable |

No failure was attributed to baseline null-space conditioning, cheirality, parallax,
correspondence geometry or evaluator ambiguity. The `oracle_full_current` cheirality
refusal on 13.22.14→13.24.33 is a *different* read-out on the 19 617-edge many-to-many
set, not the best-of set the adjudicator consumes; the best-of true class passes.

---

## 4. Per-pair table

All pose numbers below are the **frozen evaluator's** output on the adjudicated pose.

| # | pair (L→R) | set | ref class / floors | truths | best-of edges | prod best-of | prod full | adj class | inliers | ρ̄ (°) | parallax (°) | cheir. | σ₂/σ₁ | rot err (°) | base err (°) | signless (°) | gauge | usable | production failure |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 13.22.14→13.24.24 | T2 | 3/3 | 408 | 34 | `ambiguous_yaw_class` | `ambiguous_yaw_class` | 3 | 33 | 0.2984 | 63.199 | 1.000 | 0.3079 | 0.2222 | 0.2415 | 89.979 | 3 | **yes** | yaw_class_scoring |
| 2 | 13.22.14→13.24.33 | T2 | 1/1 | 19617 | 794 | `ambiguous_yaw_class` | `cheirality_failed` | 1 | 707 | 0.2743 | 68.939 | 1.000 | 0.2090 | 0.0859 | 0.2563 | 45.376 | 3 | **yes** | yaw_class_scoring |
| 3 | 13.24.24→13.24.33 | T2 | 2/2 | 820 | 190 | solved class 2 | solved | 2 | 142 | 0.3914 | 113.285 | 1.000 | 0.3680 | 0.2498 | 0.4315 | 76.928 | 2 | **yes** | — |
| 4 | 18.01.09→18.01.14 | T6 | 2/2 | 1653 | 264 | `ambiguous_yaw_class` | solved | 2 | 264 | 0.2476 | 102.685 | 1.000 | 0.1389 | 0.1800 | 0.3640 | 77.182 | 3 | **yes** | yaw_class_scoring |
| 5 | 18.01.09→18.01.18 | T6 | 3/3 | 6063 | 405 | solved class 3 | solved | 3 | 377 | 0.1938 | 69.938 | 1.000 | 0.3542 | 0.1581 | 0.4202 | 86.531 | 3 | **yes** | — |
| 6 | 18.01.09→18.01.22 | T6 | 1/1 | 7560 | 673 | solved class 1 | `ambiguous_yaw_class` | 1 | 621 | 0.1121 | 46.534 | 1.000 | 0.3819 | 0.2062 | 0.2921 | 67.204 | 3 | **yes** | — |
| 7 | 18.01.14→18.01.18 | T6 | 1/1 | 6589 | 958 | solved class 1 | solved | 1 | 849 | 0.2700 | 40.327 | 1.000 | 0.2703 | 0.1118 | 0.6371 | 76.559 | 1 | **yes** | — |
| 8 | 18.01.14→18.01.22 | T6 | 3/3 | 8839 | 661 | solved class 3 | solved | 3 | 604 | 0.2378 | 62.290 | 1.000 | 0.5044 | 0.1414 | 0.0657 | 88.867 | 1 | **yes** | — |
| 9 | 18.01.18→18.01.22 | T6 | 2/2 | 8451 | 533 | solved class 2 | solved | 2 | 513 | 0.2008 | 101.001 | 1.000 | 0.5143 | 0.0500 | 0.0750 | 27.401 | 2 | **yes** | — |

Reference-camera control (adjudicator re-run on the manifest estimates, class 0
expected): rotation error **0.0000°** on all nine, gauge-aligned baseline error
0.052°–0.520°. The adjudicated classes are therefore gauge-consistent with the reference
class under both camera regimes, not just the estimate one.

### 4.1 Reading the `signless` column

The frozen evaluator searches four 90°-about-`y` gauge rotations and reports the best
(`gauge_index`); `baseline_error_degrees_signless` keeps the raw disagreement after only
the ± sign ambiguity is removed. Large `signless` values with small gauge-aligned values
mean the pose's baseline is close to a **gauge-rotated** copy of the manifest baseline —
the frozen evaluator's own stated gauge freedom, which the `gauge_index` column exposes.
This is evaluator behaviour, not a defect in the pose, and it is **not** used here to
accept an alternate answer: `usable` never consults `signless`. The wrong-direction
control in `tests/test_oracle_pose_diagnostic.py` shows the search cannot rescue a
direction that is genuinely off the gauge orbit (world-up stays > 45° through all four).

---

## 5. Adjudicator

Name: `consensus_yaw_with_stratum_stability`. Preregistered before the run.

Changed (only these three):

1. **Yaw-class selection** — consensus-edge count with an explicit margin over the
   runner-up (`max(3, 0.25 × winner)`), replacing the frozen residual-margin rule.
2. **Baseline fit** — the bounded Gram/SVD null vector of the recomputed
   residual-consensus subset (3×3 `eigh`, no subsample, no RANSAC support).
3. **Baseline sign** — maximum depth positivity.

Unchanged: every DS-11 gate degree — 1.0° residual, 8 inliers, 0.5 inlier fraction,
3.0° parallax, 0.75 cheirality, 1e-3 singular ratio — the frozen oracle edges, costs and
pair set, and (after the correction) the DS-16B evaluator itself.

Diagnostic only, never a refusal condition: spatial-stratum direction consistency of the
winner (`stability_degrees`, `STRATUM_MIN_EDGES=8`, 4×4 grid) and the
`class_margin`. Determinism: fixed seeds, deterministic stride bound (`SOLVER_ROW_CAP`
4096, unused here — max best-of set is 958 rows), no randomness in the real read-out.

Synthetic controls (24 repeats each, ideal correspondences, seed 19): `valid_baseline`
24/24 usable, `parallel_axes` 24/24 usable, `pure_rotation` 0/24 usable (24× refused,
`cheirality_failed`). Pure-rotation is refused because both eyes share one position, so
no baseline exists; the control asserts refusal, and the refusing gate is reported rather
than tuned.

Boundedness: peak RSS **587.6 MiB** (limit 1500 MiB), runtime **30.5 s**, Python 3.10.11,
NumPy 2.2.6, Windows 10.0.26200.

---

## 6. Blindness boundary

Blind (never re-detected or re-matched here): the digest-verified DS-16 phase-1 keypoints
and descriptors, the frozen DS-14 selected matches, and the adjacent image-derived (M3)
camera estimates used as the "current" regime.

Reference-assisted, evaluation-only, allowed solely for this diagnostic: fixture
manifests/litematics/scene transforms (render ownership and depth labels), the manifest
cameras as the reference regime and for the pose error score, and the frozen DS-16/DS-16B
artifacts as frozen count cross-checks. None of it enters a production path, and the
script refuses to run if any frozen digest has moved.

---

## 7. Exact commands

```
.venv\Scripts\python.exe scripts\diagnose_oracle_pose_failures.py
.venv\Scripts\python.exe -m pytest tests\test_oracle_pose_diagnostic.py -o addopts='' -q
Get-FileHash exports\diagnostics\ds19\oracle_pose_failures.json,exports\diagnostics\ds19\oracle_pose_pairs.csv,exports\diagnostics\ds19\oracle_pose_run.log -Algorithm SHA256
```

Focused suite: **8 passed** (no full suite was run, per scope). The artifacts above are
from a single run of the script at its final state; no run was repeated or re-thresholded
after seeing its outcome.

---

## 8. Next hypothesis

The repaired failure mode is class *scoring*, not class *geometry*: the true class's
support is healthy (33 / 707 / 264 consensus edges) and its median residual is
0.25°–0.29°, yet the frozen 0.25° residual-margin rule declares it ambiguous. The
next question is whether that rule is doing useful work on the blind candidate edges
(where a spurious quadrant can shrink its own median by dropping matches) or only on
oracle edges. The DS-19 result argues for a **support-based tie-break with a residual
guard**: keep the 0.25° rule for near-ties, but let a class with a large consensus-count
margin and a sub-1° median win outright. That is a DS-11 gate-policy question, so it
needs a directive before any change — no production edit is proposed or made here.
