# DS-IMG-R11-AXES02-PAIR-SCAN-2026-10-04

Task ID: `DS-IMG-R11-AXES02-PAIR-SCAN-20261004`. Owner: DeepSeek (bounded
evaluation harness + focused tests + one exact run). Date: 2026-10-04.
Status: **harness/tests complete, exactly 6,068 context-pinned solver calls
executed once, sealed artifact produced, stopped for Lead review. Acceptance
wording erratum (`DS-IMG-R11-PAIR-SCAN-ACCEPTANCE-ERRATUM`) applied 2026-10-04:
the frozen 8.75 MB artifact is unchanged and was not rerun; only the harness
report strings, this report and the contract assertions changed. No promotion,
no production edit, no threshold change, no truth/reference/camera/mask/
evaluator/Minecraft read or write.**

This report resumes the incomplete pair-scan work from the existing working tree
and follows the Lead restart note plus the definitive exact 6-D Pareto contract
that supersedes both the original weighted/bottleneck ranking line and the 2-D
pooled constants that were present in source. It is a single-fixture diagnostic
of serialized provenance. It is not accuracy, correspondence truth, camera or
geometry evidence, and it changes no production behaviour or decision. A later
Lead erratum corrected the acceptance-definition wording in sections 1, 2, 7 and
9 without touching the frozen artifact; see section 12.

## 1. Result summary

- **Primary result (solver-accepted):** **25 of 6,068 pairs accepted** (fraction
  `0.004119973632168754`, i.e. `0.4120%`). "Accepted" is the binding definition:
  the unchanged solver returned its normal accepted outcome on the whole arm
  (`reason == ""` / `accepted true`), including the sealed axis-1 C3 row. The
  separate moved-axis `.90` resultant / `.15` residual gate diagnostic also
  passes on exactly the same 25 pairs, so the count is unchanged either way.
- **Exact 6-D Pareto frontier over all 6,068 arms: 24 members** (fraction
  `0.003955174686882004`); **6044 pairs are exactly dominated**.
- **Named accepted-arm frontier (`frontier_accepted_solver`): 4 members** —
  `A020-300`, `A020-308`, `A065-300`, `A065-308` (fraction
  `0.0006591957811470006`). It is a strict subset of the all-arm frontier. The
  gate-accepted diagnostic frontier (`frontier_accepted_gate`) has the same four
  members on this fixture and is reported separately. No scalar winner is
  selected or promoted.
- **Globally highest-correlation-score pair in-scan:** `A050-304`
  (axis-0 row 50 score `0.7718155451690802` + axis-2 row 304 score
  `0.7357697526812058` = `1.507585297850286`). It is **not** accepted: reason
  `plane_integer_residual`, `resultant0 = 0.480063`, `residual0 = 0.143814`
  (axis 0 gate fails on resultant).
- **Already accepted `A020-308`** is present in-scan, is gate-accepted and
  solver-accepted, and its measured outcome replays the sealed exhaustive final
  arm **field-for-field** (`reason ""`, assignment `[0, 3, 1]`,
  resultants `0.9989846816948525 / 0.9891764547260209 / 1.0`, residuals
  `0.005399373549514589 / 0.02041936055568005 / 2.0379109625673642e-15`,
  observations `[8, 6, 6]`, active count 10, `duplicate_b false`).
- **Axis 1 is the sealed C3 substitution and constant** across all 6,068
  outcomes: resultant `0.9891764547260209`, residual `0.02041936055568005`
  (gate pass). It is reported separately throughout.
- **All 25 accepted pairs have both per-axis score deltas negative** (i.e. lower
  correlation score than the exact survivors they replace). The accepted set is
  the measured score/consistency tradeoff: consistency on the two moved axes is
  bought at a `-0.013` to `-0.497` axis-0 and `-0.031` to `-0.243` axis-2 score
  delta. Score delta is only a correlation-score proxy, never truth probability.
- **No duplicate active edge set** survived at runtime: `duplicate_pair_count =
  0`, and every one of the 6,068 combined active edge sets was proven
  duplicate-free solver-free before the first call.
- **Artifact:** one file, `8,751,913` bytes, no `.partial`, fresh root only:
  `.tmp-geom-r11-axes02-pair-scan-20261004/counterfactual_replay.json`,
  seal `dd2c63c97643d8b5d11d92fc686643c4eb782ce3266955545c9f4f016216a86a`,
  file SHA-256 `0EFFDF865CCDCC1ACEC8B9663502626ED2B966382BD8F24D22FA6277615DC5BD`.
- **Frozen artifact label (erratum note):** the artifact was sealed before this
  erratum and therefore still embeds the earlier gate-primary
  `summary.accepted_definition` / `pareto.rule` text. It was **not** modified or
  resealed and no solver scan was rerun; because the solver-accepted and
  moved-axis gate sets coincide at 25 pairs and their 4-member frontiers
  coincide, every count and frontier recorded in it is correct under the
  corrected definition. See section 12.

## 2. Definitive contract implemented

The restart note's exact six-dimensional per-axis rule replaces the earlier
proposals. For each of the 6,068 arms the maximised objective vector is

```
( score_delta_axis0, score_delta_axis2, resultant0, resultant2,
  -residual0, -residual2 )
```

i.e. maximise both per-axis score deltas against the **exact replaced
survivors** (`candidate_score - replaced_survivor_score`, a correlation-score
proxy only), maximise `resultant0`/`resultant2`, minimise
`residual0`/`residual2`. This is equivalent to maximising the unchanged gate
margins `resultant - 0.90` and `0.15 - residual` on each moved axis. Dominance
is exact componentwise (`>=` in all six, `>` in at least one), with no tolerance;
equal vectors do not dominate each other. There is **no pooled score, no
weighted scalar and no newly selected winner**.

Acceptance is recorded two ways and both are reported:

1. **solver acceptance (primary, the binding definition):** the unchanged
   `solve_component` returned its normal accepted outcome on the whole arm
   (`reason == ""` / `accepted true`), including the sealed axis-1 C3 row;
2. **moved-axis gate acceptance (separate diagnostic):** both moved axes pass the
   unchanged per-axis gate (`resultant >= min_resultant` and
   `residual <= integer_tolerance`). A per-moved-axis-only gate pass is never
   labelled solver-accepted; the per-axis indicators are reported so failures
   stay attributable.

On this fixture the two sets coincide (25 pairs each), and their non-empty
4-member frontiers coincide as well. `frontier_accepted_solver` is the named
accepted-arm frontier; `frontier_accepted_gate` is preserved separately as the
gate-accepted diagnostic. The superseded
`choose accepted arms first; rank by minimum resultant descending, maximum
residual ascending, sum of candidate scores descending, row indices ascending`
line is retained in the artifact only for provenance and is not applied. The
superseded 2-D `PAIR_SCAN_POOLED_OBJECTIVES` constant was removed from source.

## 3. Consumed pins (file SHA-256 and seal)

All six sealed inputs were pinned by file digest and, where they carry a seal,
by replayed `seal_sha256`; none was rebuilt or rerun.

| Input | File SHA-256 | Seal SHA-256 |
|---|---|---|
| R10 capture `.tmp-geom-prefilter-capture-r10-20261004/observation.json` | `BC957B684152AEEBB5C055EB12816643F91A506AB01543761D6461FEE4C624B4` | (record `record_sha256` bound) |
| R11 capture `.tmp-geom-solver-context-r11r2-20261004/observation.json` | `924D388D5ED41287EB88A6B0B709B8228C2EAD8422F9957946EFC6A734F75A92` | context `7e18498e82cd76d589e5673dca75e463d4efd6e0bd39a29cc8f1a8427aa24439` |
| R11 baseline control `.tmp-geom-counterfactual-r11-20261004/counterfactual_replay.json` | `E0E7F9C8F46FF4B97E94E5CA8F2E612C46D926D773B006AD605E5BED27A7DD5C` | `fc94cd3700c0106d3769f928a3461fea5fe8716054b68abfe333f697ef0a5b4f` |
| R11 C1-C5 arms `.tmp-geom-counterfactual-r11-arms-20261004/counterfactual_replay.json` | `CFC12BF5A6CBD36374175F2B0A28EE254519D119A094C15A9347A2D7583E87DC` | `a7a7ba4363f56dcf4e9f30c956ae20d1fecf3bb4761f629ab2387fef5fc85556` |
| R11 axes-0/2 + C3 `.tmp-geom-counterfactual-r11-axes02-20261004/counterfactual_replay.json` | `0ED1FF3328F8059FA041F0EDB3D8ABC539E67FDE630E347539F8103D95547149` | `661257618e6635cdead5696259896af9120f04848757374162b30130610ce249` |
| R11 exhaustive axes-0/2 `.tmp-geom-counterfactual-r11-axes02-exhaustive-20261004/counterfactual_replay.json` | `2FE78B7ABF5AC25831F84624C91A31291B59EA09023DFB06BC1D314B4555B27A` | `c274d77ecd0475809423523e7d76c17c24d2c0319e2e089e3718781ea8c9c2ac` |

The R11 exact captured call replayed through the production validator with
`integer_tolerance = 0.15`, `min_resultant = 0.90`, `ambiguous_margin = 0.02`;
the R10/R11/baseline reconciliation, the sealed C3 provenance, and the
C1-C5/axes-0-2/exhaustive cross-binds all replayed before the first solver call.

## 4. Solver-free prevalidation (before any of the 6,068 calls)

- Eligible substitutions re-derived exactly: **164 world-axis-0 + 37
  world-axis-2 = 201**, per-group/axis `{0:81, 3:53, 8:30, 5:26, 6:11}` and
  per-group/target `{0:4→81, 3:16→25, 3:17→28, 5:23→26, 6:26→11, 8:37→30}`.
- **6,068 unique axis-0 × axis-2 pair keys**, each adding its two one-for-one
  replacements on top of the sealed C3 rows (203→19/group 4, 383→33/group 7),
  enumerated solver-free.
- **All 6,068 combined active edge sets duplicate-free** under the sealed
  `[0, 3, 1]` assignment; `duplicate_free_all = true`, `pair_keys_unique = true`,
  `validated_before_any_solver_call = true`.
- Pair-key digest independently re-derived from the sealed data:
  `8cdc32850ad53bc3388ba68e2e4170fe204bf7256ff6570a4b02a36b7f63fd55`,
  equal to **both** the pinned constant and the accepted exhaustive artifact's
  own recorded `valid_pairs_sha256`. A tampered pin stops the plan before any
  call (focused-test refusal).

## 5. Exact run evidence

- **6,068** solver calls, one per prevalidated pair, once each, no retries
  (`retries = 0`), all context-pinned to the R11 exact call
  (`record_sha256 = 7e18…4439`, threshold/constraint binding replayed).
- **No rerun** of the accepted baseline, C1-C5, C3, axes-0/2 or exhaustive arms
  (`baseline_rerun/c3_rerun/axes02_rerun/exhaustive_rerun/c1_c5_rerun = false`).
- Output root `.tmp-geom-r11-axes02-pair-scan-20261004` was fresh; it now
  contains exactly one file (`counterfactual_replay.json`) and no `.partial`.
- All 6,068 outcomes returned assignment `[0, 3, 1]`; `observations = [8, 6, 6]`
  for every pair; `duplicate_b = false` for every pair; reason counts
  `{"": 25, "plane_integer_residual": 6043}`.

## 6. Primary result: accepted-pair count and tradeoff

Accepted pairs are exactly the measured 5 × 5 cross product of axis-0 rows
`{20, 33, 34, 35, 65}` with axis-2 rows `{273, 290, 293, 300, 308}`
(25 pairs). On the accepted set the moved-axis diagnostics separate cleanly by
moved-axis row:

| axis-0 row | score delta (d0) | resultant0 | residual0 |
|---|---|---|---|
| 20 | -0.2700765894291830 | 0.9989846816948525 | 0.005399373549514589 |
| 33 | -0.4974732932351568 | 0.9219813790673266 | 0.050634… |
| 34 | -0.449763… | 0.9288223513806149 | 0.052949… |
| 35 | -0.203437… | 0.9130136677927988 | 0.055314… |
| 65 | -0.1315546428384300 | 0.9989846816948521 | 0.005399373549517… |

| axis-2 row | score delta (d2) | resultant2 | residual2 |
|---|---|---|---|
| 273 | -0.2431514109381222 | 0.9595856130589900 | 0.039917… |
| 290 | -0.170424… | 0.9671412189073513 | 0.035489… |
| 293 | -0.092233… | 0.9741076761722617 | 0.032798… |
| 300 | -0.0310217192840827 | 0.9946487011376160 | 0.014146… |
| 308 | -0.0687024154695097 | 1.0000000000000000 | 2.0379e-15 |

Every accepted pair has **negative** score deltas on both moved axes; no
accepted pair improves on the survivor it replaces in correlation score. Within
the observed 6,068-arm landscape, resultant/residual consistency on the two
moved axes is only reached by accepting a correlation-score deficit. This is the
measured score/consistency tradeoff; it is a single-fixture diagnostic and is
not a general filter law, accuracy, or promotion criterion.

## 7. Pareto frontiers (exact 6-D, no scalar)

### 7.1 All-arm frontier (24 of 6,068; 6044 dominated)

Frontier members factor exactly as axis-0 rows `{7, 20, 50, 65, 150, 151, 169,
183}` × axis-2 rows `{295, 300, 308}`. Every listed pair is exactly
non-dominated even though several differ only at machine precision on
resultants/residuals; that is the exact-dominance contract, not a rounding
artefact choice.

| Pair | d0 | d2 | resultant0 | residual0 | resultant2 | residual2 | accepted |
|---|---|---|---|---|---|---|---|
| A007-295 | -0.054623 | -0.024637 | 0.782423 | 0.094684 | 0.700226 | 0.116160 | no |
| A007-300 | -0.054623 | -0.031022 | 0.782423 | 0.094684 | 0.994649 | 0.014146 | no |
| A007-308 | -0.054623 | -0.068702 | 0.782423 | 0.094684 | 1.000000 | 0.000000 | no |
| A020-295 | -0.270077 | -0.024637 | 0.998985 | 0.005399 | 0.700226 | 0.116160 | no |
| A020-300 | -0.270077 | -0.031022 | 0.998985 | 0.005399 | 0.994649 | 0.014146 | **yes** |
| A020-308 | -0.270077 | -0.068702 | 0.998985 | 0.005399 | 1.000000 | 0.000000 | **yes** |
| A050-295 | -0.017896 | -0.024637 | 0.480063 | 0.143814 | 0.700226 | 0.116160 | no |
| A050-300 | -0.017896 | -0.031022 | 0.480063 | 0.143814 | 0.994649 | 0.014146 | no |
| A050-308 | -0.017896 | -0.068702 | 0.480063 | 0.143814 | 1.000000 | 0.000000 | no |
| A065-295 | -0.131555 | -0.024637 | 0.998985 | 0.005399 | 0.700226 | 0.116160 | no |
| A065-300 | -0.131555 | -0.031022 | 0.998985 | 0.005399 | 0.994649 | 0.014146 | **yes** |
| A065-308 | -0.131555 | -0.068702 | 0.998985 | 0.005399 | 1.000000 | 0.000000 | **yes** |
| A150-295 | -0.018076 | -0.024637 | 0.486355 | 0.152780 | 0.700226 | 0.116160 | no |
| A150-300 | -0.018076 | -0.031022 | 0.486355 | 0.152780 | 0.994649 | 0.014146 | no |
| A150-308 | -0.018076 | -0.068702 | 0.486355 | 0.152780 | 1.000000 | 0.000000 | no |
| A151-295 | -0.013037 | -0.024637 | 0.225110 | 0.192938 | 0.700226 | 0.116160 | no |
| A151-300 | -0.013037 | -0.031022 | 0.225110 | 0.192938 | 0.994649 | 0.014146 | no |
| A151-308 | -0.013037 | -0.068702 | 0.225110 | 0.192938 | 1.000000 | 0.000000 | no |
| A169-295 | -0.035869 | -0.024637 | 0.706950 | 0.109998 | 0.700226 | 0.116160 | no |
| A169-300 | -0.035869 | -0.031022 | 0.706950 | 0.109998 | 0.994649 | 0.014146 | no |
| A169-308 | -0.035869 | -0.068702 | 0.706950 | 0.109998 | 1.000000 | 0.000000 | no |
| A183-295 | -0.077013 | -0.024637 | 0.840664 | 0.071489 | 0.700226 | 0.116160 | no |
| A183-300 | -0.077013 | -0.031022 | 0.840664 | 0.071489 | 0.994649 | 0.014146 | no |
| A183-308 | -0.077013 | -0.068702 | 0.840664 | 0.071489 | 1.000000 | 0.000000 | no |

The tradeoff is explicit: accepting a lower score delta buys resultant/residual
margin (e.g. `A151-*` has the best axis-0 score delta but the worst resultant0 at
`0.225110`; `A020-*`/`A065-*` have the best resultant0 at `0.998985` but a
`-0.270`/`-0.132` axis-0 score delta).

### 7.2 Named accepted-arm frontier `frontier_accepted_solver` (4 members, non-empty)

This is the named accepted-arm frontier: the exact 6-D nondominated set over the
25 pairs for which the unchanged solver returned `reason == ""` / `accepted true`
on the whole arm.

| Pair | d0 | d2 | resultant0 | residual0 | resultant2 | residual2 |
|---|---|---|---|---|---|---|
| A020-300 | -0.2700765894291830 | -0.0310217192840827 | 0.9989846816948525 | 0.005399373549514589 | 0.9946487011376160 | 0.014145972151965782 |
| A020-308 | -0.2700765894291830 | -0.0687024154695097 | 0.9989846816948525 | 0.005399373549514589 | 1.0000000000000000 | 2.0379109625673642e-15 |
| A065-300 | -0.1315546428384300 | -0.0310217192840827 | 0.9989846816948521 | 0.005399373549517470 | 0.9946487011376160 | 0.014145972151965782 |
| A065-308 | -0.1315546428384300 | -0.0687024154695097 | 0.9989846816948521 | 0.005399373549517470 | 1.0000000000000000 | 2.0379109625673642e-15 |

`A065-*` is better on axis-0 score delta and `A020-*` is better on `resultant0`
by `4.4e-16` (and on `residual0` by `2.9e-15`), so under the mandated exact
dominance **neither dominates the other** and both remain on the accepted
frontier. This machine-precision-boundary sensitivity is reported verbatim rather
than rationalised; no scalar tie-break is applied, and no member is promoted.

The separate gate-accepted diagnostic frontier `frontier_accepted_gate` (exact
6-D nondominated set over the 25 moved-axis gate-passing pairs) has the same four
members on this fixture; it is preserved as a diagnostic, not as the named
accepted-arm frontier.

## 8. Highest-score pair and the accepted reference pair

**Highest total captured candidate score:** `A050-304`
(axis-0 row 50 score `0.7718155451690802` + axis-2 row 304 score
`0.7357697526812058` = `1.507585297850286`; tie-break by axis-0 score, axis-2
score, pair id). Its measured result: reason `plane_integer_residual`,
assignment `[0, 3, 1]`, resultants
`{0: 0.48006288517085793, 1: 0.9891764547260209, 2: 0.6357413103666135}`,
residuals `{0: 0.14381448399286942, 1: 0.02041936055568005,
2: 0.12702037579149505}`, observations `[8, 6, 6]`, `duplicate_b false`,
moved-axis gate fail (axis-0 resultant `0.480063 < 0.90`). It is **not** on the
all-arm frontier and is not accepted. Score alone is demonstrated insufficient.

**Accepted `A020-308` in-scan:** axis-0 row 20 / group 0 / target 4 / score
`0.5196353914813745` / d0 `-0.270076589429183`; axis-2 row 308 / group 6 /
target 26 / score `0.6956890426498159` / d2 `-0.0687024154695097`; reason `""`,
accepted `true`, gate accepted. Its recorded outcome is exactly equal to the
sealed exhaustive `final_arm` outcome on `reason`, `assignment`, `resultants`,
`residuals`, `observations`, `active_match_count` and `duplicate_b`
(cross-check enforced before the artifact was sealed).

## 9. Independent verification performed

- Solver-free prevalidation re-run before the recorded run: 201 substitutions,
  6,068 pairs, all duplicate-free, pair-key digest matched pin and exhaustive.
- Focused test module `tests/test_same_axis_counterfactual_replay.py`:
  **67 passed**, including new pair-scan tests for the 6-D constant contract,
  the 6,068-pair/digest prevalidation, an exactly-6,068-call budget test (stubbed
  solver, no real call duplication), pin refusals before any call, exact-6-D
  dominance/oracle checks, the single real `A020-308` cross-check against the
  sealed exhaustive final arm, and exhaustive-payload tamper refusals.
- `py_compile` passed for both the harness and the test module.
- Independent post-run verification: seal replays; 6,068 unique rows; zero
  internal-consistency violations; no NaN/Inf objectives; and an independently
  written NumPy brute-force dominance pass reproduced **both** stored frontiers
  exactly (24 all-arm; and the two coincident 4-member accepted frontiers — 4 for
  the named solver-accepted `frontier_accepted_solver` and 4 for the separate
  gate-accepted diagnostic `frontier_accepted_gate`). An initial brute-force
  helper with reversed comparison orientation was corrected before acceptance;
  the corrected oracle agrees with the harness.
- Erratum verification (2026-10-04, after the acceptance-wording correction):
  `py_compile` passed again for both the harness and the test module, and the
  focused module passed again (**67 passed**) with added contract assertions
  pinning the solver-accepted primary definition and the
  `frontier_accepted_solver` / `frontier_accepted_gate` naming. No solver scan
  was rerun and the frozen artifact was not modified or resealed.
- Tests may exercise solver calls in-process (permitted); the recorded run above
  is the only full 6,068-call execution, and no call was retried.

## 10. Hashes (recorded)

| Artifact | SHA-256 |
|---|---|
| Harness at the recorded run (run-time code; equals the pre-erratum file) | `1EC482B697EB7985885B32A3CC7C4870F39606B199E102D73224ABD44F551F24` |
| Harness after erratum (current) | `0B70BF33FDD1659D4F02D18B5604BC335770DE0ABA7F7411646E09A4D9CC0881` |
| Test module at the recorded run (pre-erratum) | `2AB3CC9E25C6D0F4A2DA0479CE8432E904B60AA1BC1B4C79F9B0EC559A74F54A` |
| Test module after erratum (current) | `FDDD35765BABF62C3ED8592FECE42835714ABF8708AEAFCD450DF4B6EEF0EB38` |
| `.tmp-geom-r11-axes02-pair-scan-20261004/counterfactual_replay.json` (frozen, unchanged) | `0EFFDF865CCDCC1ACEC8B9663502626ED2B966382BD8F24D22FA6277615DC5BD` |
| Artifact seal (`seal_sha256`, frozen, unchanged) | `dd2c63c97643d8b5d11d92fc686643c4eb782ce3266955545c9f4f016216a86a` |

Consumed-input hashes are listed in section 3. All pre-existing artifacts and
hashes listed there were re-hashed unchanged before and after this task. The
frozen pair-scan artifact and its seal were re-hashed after the erratum and are
byte-identical to the run-time values above.

## 11. Boundaries and disposition

Diagnostic only: no truth, reference, camera, mask, evaluator or Minecraft read
or write; no production edit; no threshold change; no promotion; no junction/
reparse operation, recursive deletion, encoded command, installer, generated
executable, background process, persistence or subagent. Existing artifacts were
preserved; the only new outputs are the harness/test changes above, the single
sealed artifact, and this report.

**Stopped for Lead re-review.** The primary reported quantity is the
solver-accepted pair count/fraction `25 / 6,068`; the all-arm frontier, the named
`frontier_accepted_solver` accepted-arm frontier, the separate
`frontier_accepted_gate` diagnostic and the highest-score pair are reported with
no scalar ranking and no newly selected winner. The frozen artifact was preserved
unchanged and no solver scan was rerun for the erratum.

## 12. Erratum: acceptance-definition wording (DS-IMG-R11-PAIR-SCAN-ACCEPTANCE-ERRATUM)

Lead review of the completed pair scan identified one semantic mismatch: the
source constants and section 2 described **gate** acceptance as primary, while
the later binding Lead clarification defines the primary accepted pair as the
unchanged solver's own `reason == ""` / `accepted true` outcome, with the
moved-axis gate reported independently. This erratum corrects that wording only.

- `PAIR_SCAN_ACCEPTED_DEFINITION` now states the solver-accepted primary
  definition (whole arm, including the sealed axis-1 C3 row) and marks the
  moved-axis unchanged per-axis gate as a separate diagnostic that is never
  relabelled solver-accepted.
- `PAIR_SCAN_PARETO_RULE` now names `frontier_accepted_solver` as the named
  accepted-arm frontier and keeps `frontier_accepted_gate` as a separate
  gate-accepted diagnostic. The harness pair-scan docstring and payload note were
  aligned; report sections 1, 2, 7 and 9 were corrected.
- **Frozen artifact preserved:** `.tmp-geom-r11-axes02-pair-scan-20261004/
  counterfactual_replay.json` is unchanged (8,751,913 bytes, file SHA-256
  `0EFFDF865CCDCC1ACEC8B9663502626ED2B966382BD8F24D22FA6277615DC5BD`, seal
  `dd2c63c97643d8b5d11d92fc686643c4eb782ce3266955545c9f4f016216a86a`). It was
  sealed before the erratum and therefore still embeds the old gate-primary
  `summary.accepted_definition` and `pareto.rule` strings; that older label is
  retained exactly as sealed and no reseal was performed.
- **Equivalence:** both definitions select the same 25 pairs on this fixture, and
  `frontier_accepted_solver` and `frontier_accepted_gate` are the same four pairs
  (`A020-300`, `A020-308`, `A065-300`, `A065-308`). No metric changed, so the
  frozen artifact's recorded counts and frontiers are correct under the corrected
  definition.
- **Verification:** `py_compile` passed for both modules and the focused module
  `tests/test_same_axis_counterfactual_replay.py` passed again (**67 passed**,
  5.01 s), now including assertions that pin the solver-accepted primary
  definition and the `frontier_accepted_solver` / `frontier_accepted_gate`
  naming. **No solver scan was rerun**; the single recorded 6,068-call execution
  is untouched.
- **Hashes:** run-time harness
  `1EC482B697EB7985885B32A3CC7C4870F39606B199E102D73224ABD44F551F24` (equals the
  pre-erratum file) vs corrected harness
  `0B70BF33FDD1659D4F02D18B5604BC335770DE0ABA7F7411646E09A4D9CC0881`; run-time
  test module `2AB3CC9E25C6D0F4A2DA0479CE8432E904B60AA1BC1B4C79F9B0EC559A74F54A`
  vs corrected test module
  `FDDD35765BABF62C3ED8592FECE42835714ABF8708AEAFCD450DF4B6EEF0EB38`. The
  frozen artifact and seal hashes are unchanged.
