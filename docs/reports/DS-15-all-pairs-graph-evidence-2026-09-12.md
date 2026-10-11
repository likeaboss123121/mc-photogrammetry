# DS-15 — All-pairs blind graph evidence over the frozen calibrated frame sets

DeepSeek CLI DS-15, 2026-09-12. Diagnostic only. Owned paths: new
`scripts/diagnose_joint_all_pairs.py`, new
`docs/reports/DS-15-all-pairs-graph-evidence-2026-09-12.md`, generated
`exports/diagnostics/ds15/` (including `repeat_check/`). The DS-14 matcher
(`joint_matching.py`), DS-11/12 sources, every DS-12/14 default and frozen
artifact, all thresholds and every test file are untouched. No dependency added;
nothing staged or committed; no test suite, no benchmark, no graph-production,
component, scale, schema, CLI/desktop, raycast or Luna path was run or edited.
Revision: HEAD `c6dba2cf4610876db7d24361eb13fc9ea3ceca0c` plus the accepted
working-tree edits (dirty, as before). This report is the stop point.

## 1. Question and rule

DS-14 froze six adjacent pairs and solved three, only two with the correct yaw
class. DS-15 asks what the *whole* calibrated frame set looks like: for each
frozen set — T2 (3 frames), T6 (4 frames), `t3-sparse-8` (8 frames) — every
unordered pair is enumerated, so the pair list is fixed by `C(n,2)` on the
artifact's frame inventory in sorted-filename order. No pair was added, dropped
or reordered after seeing a result; there is no outcome filter anywhere. The
matcher, its `JointMatchSettings()` defaults, `PairGates()` defaults and
`usable_masks=None` (per-frame `~create_gui_mask`) are exactly DS-14's.

Phase separation is unchanged: phase 1 reads only frames and the image-derived
M3 calibration artifacts and writes the frozen JSON/CSV; `load_camera_manifest`
is imported inside the reference function only. The manifest-free graph
diagnostics are computed from the already-frozen pair outputs and written
*before* any manifest is opened (recorded as a separate artifact with its own
SHA-256). Reference comparison happens only in phase 2.

## 2. Frozen inventory and provenance

| set | fixture | image-derived calibration artifact (SHA-256 prefix) | frames | unordered pairs |
| --- | --- | --- | --- | --- |
| T2 | `templates 2` | `435c19bba85c` `M3-calibration-templates-2.json` | 3 | 3 |
| T6 | `templates 6` | `a24751ab56cf` `M3-calibration-templates-6.json` | 4 | 6 |
| t3-sparse-8 | `templates 3` | `bbbd3a0d938a` `M3-calibration-templates-3-t3-sparse-8.json` | 8 | 28 |
| **total** | | | **15** | **37** |

Calibration hashes are the same artifacts DS-14 used; manifests (opened in
phase 2 only) are `4f743e4fc478`, `836ad62f99c5`, `92dc88ec990f`. Environment:
Python 3.10.11, NumPy 2.2.6, OpenCV 5.0.0. All frames are 2560×1369.

**Reproducibility of DS-14's six pairs.** For all six `PAIR_SPECS` records, the
DS-15 phase-1 `joint` record and both `_Estimate` records compare byte-equal to
DS-14's frozen `joint_estimates.json` (0 mismatches): same statuses, reasons,
support and support-floor values, classes, edge counts and vectors. So the
all-pairs run did not perturb the accepted baseline; it is a superset.

**Determinism.** Two independent phase-1 runs (main + `repeat_check/`) produced
the identical canonical digest
`5351af6ad90c494b3cae583b805e28831cb81e8161f82f0439500e132a461d61`
(outcome payload, wall-clock stripped); the file SHA-256 differs only through the
`generated` timestamp and the recipe path.

## 3. Phase 1 — blind all-pairs result

`solved 8/37`, `refused 29/37`. Every pair below cleared the DS-14 support floor,
the 4-of-16-bin spread on both frames and conditioning; what is shown is the
manifest-free outcome only. All 37 pairs had `truncated == 0`; candidate graphs
run from 268 to 4940 edges. Total phase-1 wall clock 27.28 s (mean 0.737 s/pair,
max 1.819 s); per set T2 2.24 s, T6 8.31 s, `t3-sparse-8` 16.72 s.

| set | left → right (suffix) | support/floor | edges | med ρ | parallax | cheir | margin | class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T2 | 13.22.14 → 13.24.24 | 106/18 | 1434 | 0.35° | 120.1° | 1.00 | 3 | 3 |
| T2 | 13.24.24 → 13.24.33 | 142/18 | 1729 | 0.42° | 118.6° | 0.86 | 28 | 2 |
| T6 | 18.01.09 → 18.01.18 | 152/35 | 2974 | 0.38° | 123.5° | 0.87 | 3 | 3 |
| T6 | 18.01.14 → 18.01.18 | 243/35 | 4186 | 0.34° | 145.7° | 0.94 | 7 | 3 |
| T6 | 18.01.14 → 18.01.22 | 248/50 | 4940 | 0.38° | 98.1° | 0.99 | 37 | 3 |
| T3 | 20.03.32 → 20.06.52 | 108/16 | 1732 | 0.40° | 130.3° | 0.79 | 7 | 2 |
| T3 | 20.03.55 → 20.05.23 | 56/8 | 886 | 0.38° | 86.3° | 1.00 | 3 | 3 |
| T3 | 20.05.42 → 20.06.32 | 23/8 | 268 | 0.34° | 110.4° | 0.87 | 2 | 3 |

Refusals by typed reason: `cheirality_failed` 23, `low_spatial_spread` 4,
`ambiguous_hypothesis` 2 (T2 1/0/0, T6 3/0/0, T3 19/4/2). Cheirality dominates
every set: 23 of 29 refusals had a geometrically ranked winner that could not
place its support in front of both cameras. Two T3 pairs abstained on the
declared ≥2-edge ambiguity margin (`20.03.32→20.06.32`, `20.04.18→20.06.32`) and
four T3 pairs failed only the 4-of-16-bin spread, including pairs whose sparse
frame has 83–298 raw keypoints.

## 4. Manifest-free graph diagnostics (computed after freeze, before manifests)

Written to `joint_graph_diagnostics.json` + `joint_triangles.csv`; the reference
phase records the graph file's SHA-256 and consumes its summary without
recomputing any pair.

| set | solved edges | components (sizes) | isolated | cycle rank | triangles: solved-edge histogram 0/1/2/3 |
| --- | --- | --- | --- | --- | --- |
| T2 | 2/3 | 1 (3) | — | 0 | 0 / 0 / 1 / 0 |
| T6 | 3/6 | 1 (4) | — | 0 | 0 / 2 / 2 / 0 |
| t3-sparse-8 | 3/28 | 5 (2,2,1,1,2) | 20.04.18, 20.04.53 | 0 | 38 / 18 / 0 / 0 |
| **total** | **8/37** | 7 components | 2 | **0** | **38 / 20 / 3 / 0** |

T2 and T6 solve enough edges to connect their full frame sets (chains); the T3
solved edges are three disjoint 2-frame components plus two isolated frames, so
the solved graph is fragmented. **Every set's solved-edge graph is a forest**
(cycle rank `E − V + C = 0`), and across all 61 triangles at most two of three
edges is ever solved. Consequently **zero triangles are eligible** for the
class-cycle or direction-closure test: there is no cycle of solved pair outputs
to close. That is a graph fact, not a threshold decision — it follows from the
29 refusals above and is reported before any manifest is read.

**Closest available class-cycle evidence.** Three triangles have exactly two
solved edges (T2 one, T6 two). The exact mod-4 closure requirement
`c(i,j)+c(j,k)−c(i,k) ≡ 0` then *implies* a class on the missing edge. All three
missing edges were refused `cheirality_failed`.

| set | triangle (suffixes) | solved classes | missing edge | missing reason | implied class |
| --- | --- | --- | --- | --- | --- |
| T2 | 13.22.14, 13.24.24, 13.24.33 | c(i,j)=3, c(j,k)=2 | i–k | `cheirality_failed` | 1 |
| T6 | 18.01.09, 18.01.14, 18.01.18 | c(j,k)=3, c(i,k)=3 | i–j | `cheirality_failed` | 0 |
| T6 | 18.01.14, 18.01.18, 18.01.22 | c(i,j)=3, c(i,k)=3 | j–k | `cheirality_failed` | 0 |

These implied classes are derived requirements on refused pairs; they are
recorded as evidence and never fed into a solve. Each T6 two-edge triangle
carries two class-3 edges and therefore implies class 0 on its refused third
edge; whether that edge would have taken the implied class is exactly what the
refusal prevents observing.

## 5. Class-cycle and direction-closure definitions (unexercised this run)

For a fully solved triangle `i<j<k`, DS-15 would report, all from the frozen
pair outputs:

* **Relative-class cycle**: `(c(i,j) + c(j,k) − c(i,k)) mod 4`, required exactly
  `0`; and the continuous companion `angle(R(j,k)·R(i,j), R(i,k))`, which must be
  `0` if the reported relative rotations compose. Both are undefined when any
  edge is refused, and no triangle qualifies.
* **Triangle direction closure**: with all three unit baseline directions
  transported into anchor frame `i` (`R(i,j)^T·b(j,k)` moves the j-anchored
  direction back to frame `i`; the third side is `−b(i,k)`), stack
  `M = [u₁ u₂ u₃]` and solve `min ‖M x‖` over `x ≥ 0` with the single scale
  gauge `‖x‖₂ = 1`. The solver enumerates all seven nonempty coordinate subsets
  exactly (the interior optimum of each face is a right singular vector, so the
  global optimum is the smallest singular value over subsets whose null vector
  can be sign-fixed nonnegative). Reported: `σ_min/σ_max` (normalized residual),
  the unconstrained `σ_min`, the nonnegativity penalty, the active subset, unit
  and sum-one lengths, and the residual vector. Declared reporting tolerance
  `0.05` on the normalized residual; raw values are always emitted.

Neither diagnostic modified a pair output; both consume only the frozen phase-1
JSON. They are implemented and ran, but the eligibility set was empty, so the
manifest-free summary is `0/0` triangles for class cycles and `0/0` for closure.

## 6. Phase 2 — reference evaluation (manifests opened only here)

`solved 8/37`, `class_correct 4/8`, `precision among solves 0.50`,
`false solves 4`. The reference class is DS-14's: the best public
`camera_rotation_matrix` match over the four quadrant offsets
(`_reference_class`), reused verbatim. The class criterion and the geometric
criterion agree perfectly in this run: the same four solves are
`within 5° rotation` (0.14–0.25°) and the same four are ≥ 90° wrong.

| set | left → right (suffix) | solved class | ref class | class ok | rotation err | baseline gauge err | baseline signless err | support/floor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T2 | 13.22.14 → 13.24.24 | 3 | 3 | **yes** | 0.222° | 9.586° | 81.094° | 106/18 |
| T2 | 13.24.24 → 13.24.33 | 2 | 2 | **yes** | 0.250° | 14.060° | 89.019° | 142/18 |
| T6 | 18.01.09 → 18.01.18 | 3 | 3 | **yes** | 0.158° | 4.352° | 81.798° | 152/35 |
| T6 | 18.01.14 → 18.01.18 | 3 | 1 | **no** | 179.950° | 18.342° | 18.342° | 243/35 |
| T6 | 18.01.14 → 18.01.22 | 3 | 3 | **yes** | 0.141° | 8.610° | 86.562° | 248/50 |
| T3 | 20.03.32 → 20.06.52 | 2 | 1 | **no** | 90.050° | 29.339° | 63.202° | 108/16 |
| T3 | 20.03.55 → 20.05.23 | 3 | 2 | **no** | 90.050° | 45.017° | 46.503° | 56/8 |
| T3 | 20.05.42 → 20.06.32 | 3 | 0 | **no** | 90.000° | 53.851° | 86.836° | 23/8 |

Per-set precision among solves: T2 `2/2 = 1.00` (0 false solves), T6
`2/3 = 0.667` (1), `t3-sparse-8` `0/3 = 0.00` (3). Median relative-rotation
error over solves 45.125°; median gauge-aligned baseline error 16.201°; median
signless baseline error 81.446°. The four class-correct solves still miss the
true baseline by 4.4–14.1° after gauge removal (81–89° signless, 98–124°
parallax): class agreement is not direction accuracy. The four false solves are
one T6 180° flip (the DS-14 repetition trap, 243 selected edges over 4186) and
three T3 90° flips. The T3 winners are the two smallest supports in the run (23
and 56; 108 for the third, against 106–248 elsewhere); in each of the three the
reference-class hypothesis's best-ranked mode failed a gate — cheirality in all
three (supports 87, 53, 20 against winners 108, 56, 23) — so a competing class
supplied the top passing mode.

## 7. Reading, without retuning

1. **All-pairs does not fix the class problem; it mostly confirms abstention.**
   29/37 pairs refuse, 23 of them on cheirality alone, and the graph that
   survives is a forest. There is no redundancy to triangulate or cross-check.
2. **Two new T6 solves are class-correct** (`09→18`, 152/2974, margin 3;
   `14→22`, 248/4940, margin 37 — the widest-support edge in the run), while the
   DS-14 180° false solve (`14→18`, 243/4186, margin 7) reproduces exactly.
   Support count alone still does not separate them; the DS-14 headline stands on
   the larger set.
3. **Every T3 solve is a quadrant flip.** The T3 winners carry the two smallest
   supports in the run (23 and 56; 108 for the third) and gauge-aligned baseline
   errors of 29–54°; in all three the reference class's best mode failed
   cheirality, so the matcher's floor-meeting passing modes were wrong-class. T3
   precision among solves is 0.0: the sparse set contributes no usable relative
   class, let alone a cycle.
4. **Class cycles and direction closure are unobservable at this sparsity.**
   Reporting them as pass/fail would be vacuous; the honest output is the
   eligibility count (0 triangles) plus the three 2-edge implied-class
   requirements, all on refused `cheirality_failed` edges.
5. **Cheirality is the binding gate, not support.** The 23 `cheirality_failed`
   refusals all cleared the support floor, spatial spread and conditioning before
   failing positive-depth support (cheirality < 0.75); 4 T3 pairs failed only
   spatial spread. Changing either would be a new declared threshold, which this
   assignment forbids and this run did not do.

## 8. Limits and non-claims

1. No production promotion, no accuracy claim, no M3 gate. The six-pair DS-14
   result is unchanged and reproduced exactly; all-pairs adds breadth, not
   authority.
2. `usable_masks=None` (derived `~create_gui_mask`), one settings set, one
   process, single-run wall clock; not a benchmark and not an M6 measurement.
3. The graph diagnostics are descriptive. They are not a reconstruction, not a
   connected-component solver product, and they compute no positions, no scale
   and no `scale_known`.
4. The direction-closure solver is exact for three columns and deterministic,
   but it was not exercised on real data because no eligible triangle exists;
   its correctness is only asserted by construction here, not by a fixture.
5. The 2-edge implied-class projection is a derived requirement on refused
   pairs; it is not evidence that the missing edge's solver would have agreed.
6. T3's 8-frame sparse subset is one frame inventory; the other 24 frames in
   `templates 3` were never enumerated. No claim about them.

## 9. Evidence

Commands run (workspace interpreter), in this order:

```
.venv\Scripts\python.exe scripts/diagnose_joint_all_pairs.py --extract-only
.venv\Scripts\python.exe scripts/diagnose_joint_all_pairs.py --graph-only
.venv\Scripts\python.exe scripts/diagnose_joint_all_pairs.py --evaluate-only
.venv\Scripts\python.exe scripts/diagnose_joint_all_pairs.py --extract-only \
    --out-dir exports\diagnostics\ds15\repeat_check
```

Frozen phase-1 bytes of record: `joint_estimates.json` SHA-256
`40fe2880c90f9afc2dda8a9d0d935ad97b7bda8078e5a6ae17644991679ed018`,
`joint_estimates.csv` `bfa1246938a661d664a0a2846130270334df94f19e867e4f2ea977f9b85a6374`,
canonical digest `5351af6a…461d61` (identical in `repeat_check/`, whose JSON is
`a969cc50ed7008092defef3f37ca88c3e839f4b201f794e98d5cc8a68e7dc808`).
Manifest-free graph artifact `joint_graph_diagnostics.json`
`2ba20f790b9ce0904aa479a905cc2d15e80c4175510484571e25fc3c02f7212a`
(56,588 bytes, includes the full 61-triangle list); `joint_triangles.csv`
`84985f15300bae5971dde0408cffcef6b6d78233145a2bda79803e7b32795de2`.
Reference `joint_evaluation.json`
`5b1f053184c4fdb7e31c0c4b1abcf7b501dd43a974d45757656f3595e16e0c54`,
which records the frozen estimate digest, the graph digest and the manifest
digests before/around the comparison; `joint_evaluation.csv`
`953018b69dc14246517014d0973c2cce6b7f296c3d66d908c006b231b70dfc22` (no
timestamps, so stable across reruns). No test file was run; no DS-14 threshold,
source or artifact was modified; nothing was staged or committed. Stop.
