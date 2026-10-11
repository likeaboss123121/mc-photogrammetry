# R5P: per-pair failure-mode audit of the immutable R5O traces (2026-10-04)

Status: **read-only descriptive audit complete. No rerun, no reconstruction,
matcher or solver execution, no source/test/product/sealed-artifact edit.**

Task: `agent swarm instantiation template/tasks/active/image-only-pair-failure-mode-audit-2026-10-04.md`.
Owner: DeepSeek (Sol-equivalent read-only audit), Luna coordinates, Astra reviews
conclusions. The 2026-10-04 Lead override authorizes a foreground, read-only
Python parser over the single R5O JSON artifact plus the report/task/comms writes.

## 1. Question and method

Among the 28 pairs and six graph arms in the immutable R5O artifact, where does
each pair first stop (candidate availability, selector/support, spatial coverage,
solve/ambiguity), and which controlled next test would best separate the observed
failure classes? This is **descriptive attribution, not causal proof**.

Only the sealed artifact `.tmp-exact-point-oracle-20261004.json` was parsed,
together with the accepted R5O report, the R5N sensitivity report and the source
contracts needed to interpret the field names
(`src/mc_photogrammetry/pipeline/joint_matching.py`,
`src/mc_photogrammetry/evaluation_pair_correspondence.py`, the R5O runner). No
raw capture, manifest, reference, mask, JAR, saved world or R5K/R5N sealed
artifact was read; no matcher/solver/reconstruction was executed. The parser ran
as an inline `python.exe -c` foreground program (no temp files); all output is
reproduced in this report.

## 2. Input verification

| Check | Result |
| --- | --- |
| Artifact | `.tmp-exact-point-oracle-20261004.json`, 2,855,980 bytes |
| File sha256 | `03faaefe40ac1496ee6f86800818f070bb3db043984d092025a8dff67c65007d` (matches R5O report) |
| Stored `seal_sha256` | `5d8a67edbb6aeb1c0dadc998f1ffa2610718e5cd36e84a5a98c11c0ab5e73899` |
| Recomputed `evaluation_pair_correspondence.seal_record` | identical -> **seal match** |
| `consistency.ok` | `true`; zero `positive_identity_mismatch` for all six arms and both scored stages |
| `status` / `evaluation_only` / `promotion_or_production_change` | `complete` / `true` / `false` |
| Pair count | 28 listed = `scene.pair_count` = `summary.pair_count` |
| Fixed surface denominator | 1,213 (`pairs_with_surface_truth` 28, `pairs_with_empty_surface_truth` 0) |
| Tolerance denominators | `0.05 -> 1,068`; `0.125 -> 1,780`; `0.25 -> 5,037`; `0.5 -> 13,545` |
| Per-arm tolerance denominators | per-pair `candidate_stage_arm_tolerance.truth_pairs` sums reproduce 1,213 / 1,068 / 1,780 / 5,037 / 13,545 exactly |
| `stage_availability` | `pair_solve` available; `plane_graph_axis_stage` and `placed_cell_geometry` typed unavailable (`pair_level_diagnostic_has_no_plane_graph_record`, `pair_level_diagnostic_has_no_placement_record`) |

Reconciliation used by this audit: for every one of the 168 arm-pair records the
ordered-gate classifier below reproduced the sealed `reason` field **exactly (0
mismatches)**. That is the strongest available check that the per-pair reasons are
the first failing gate and not a post-hoc label.

## 3. First-stop ladder and field semantics

The unchanged matcher/solver evaluates, in order (`joint_matching._winner_refusal`
plus `_class_modes`):

1. `candidate_graph_empty` — injected candidate graph has zero edges;
2. `selection_absent` — no candidate mode reaches `support_floor`, so no best mode
   exists; the sealed record carries `support = 0`, `matches_available = false`,
   `selection_state = absent_selection`, and all six gate fields `null`;
3. `support_below_floor` — a best mode exists but its refit support is below
   `support_floor`;
4. `low_spatial_spread` — `min(spatial_bins_left, spatial_bins_right) < 4`;
5. `rank_deficient` — `singular_ratio < 0.001`;
6. `low_parallax` — `parallax_degrees < 3.0`;
7. `cheirality_failed` — `cheirality_fraction < 0.75`;
8. `ambiguous_hypothesis` — a rival mode also reaches the floor with
   `best.support - competitor_support < 2`;
9. `solve_pair_refused` — all gates passed, `solve_pair` refused;
10. `solved`.

`support_floor = max(8, ceil(0.02 * min(raw_left, raw_right)))` is computed from
the **pre-injection detected keypoint counts** and is therefore identical across
the six arms for a given pair; it ranges 35..75 over these 28 pairs. Only gate
ladder steps 3-8 are candidate-graph dependent. Steps 1-2 are the support system
of the selector; 9-10 are the DS-11 pose solver. All `point_oracle_*` keys are
treated as same-surface continuous world-hit proximity arms, never as feature
identity; refusal reasons are policy outcomes, never accuracy.

## 4. Per-arm first typed stop over the 28 pairs

First-stop counts (columns sum to 28); "refused" is the sealed `status`:

| Arm | cand empty | selection_absent | support_below_floor | low_spatial | rank/parallax | cheirality_failed | ambiguous | solve_pair_refused | solved | refused |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| production_baseline | 0 | 0 | 0 | 0 | 0 | 27 | 0 | 0 | 1 | 27 |
| surface_oracle | 0 | 19 | 0 | 1 | 0 | 0 | 0 | 4 | 4 | 24 |
| point_oracle_0.05 | 0 | 20 | 0 | 1 | 0 | 0 | 0 | 4 | 3 | 25 |
| point_oracle_0.125 | 0 | 17 | 0 | 0 | 0 | 0 | 2 | 2 | 7 | 21 |
| point_oracle_0.25 | 0 | 6 | 7 | 0 | 0 | 0 | 13 | 0 | 2 | 26 |
| point_oracle_0.5 | 0 | 1 | 8 | 0 | 0 | 2 | 13 | 0 | 4 | 24 |

Stage-reach funnel and pooled denominators. "best mode formed" =
`matches_available`; "support >= floor" is a subset of it; "solver reached" =
`pose_state in {pose_abstained, pose_solved}`. Reach columns are unique-pixel-pair
truth identities against the fixed 1,213 surface denominator.

| Arm | cand graph nonempty | best mode formed | support >= floor | solver reached | solved | cand truth reach | sel truth reach |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| production_baseline | 28 | 28 | 28 | 1 | 1 | 345 / 1,213 | 52 / 1,213 |
| surface_oracle | 28 | 9 | 9 | 8 | 4 | 1,213 / 1,213 | 865 / 1,213 |
| point_oracle_0.05 | 28 | 8 | 8 | 7 | 3 | 1,068 / 1,213 | 717 / 1,213 |
| point_oracle_0.125 | 28 | 11 | 11 | 9 | 7 | 1,213 / 1,213 | 965 / 1,213 |
| point_oracle_0.25 | 28 | 22 | 15 | 2 | 2 | 1,213 / 1,213 | 1,057 / 1,213 |
| point_oracle_0.5 | 28 | 27 | 19 | 4 | 4 | 1,213 / 1,213 | 752 / 1,213 |

Pooled candidate graph sizes (`candidate_edges`): 178,056 / 1,213 / 1,068 / 1,780
/ 5,037 / 13,545. The candidate graph is never empty in any arm, so
`candidate_graph_empty` is not an observed failure class here.

## 5. Cross-tabs and evidence

### 5.1 Support floor versus achieved support

`support` is the ranked winner's refit support; `support_floor` is the raw-keypoint
derived floor. Only best-mode pairs carry a non-zero support.

| Arm | best-mode pairs | support min..max | floor min..max | margin min..max | support < floor |
| --- | ---: | ---: | ---: | ---: | ---: |
| production_baseline | 28 | 206 .. 563 | 35 .. 75 | +171 .. +491 | 0 |
| surface_oracle | 9 | 63 .. 158 | 49 .. 75 | 0 .. +86 | 0 |
| point_oracle_0.05 | 8 | 56 .. 144 | 49 .. 75 | +7 .. +72 | 0 |
| point_oracle_0.125 | 11 | 53 .. 197 | 49 .. 75 | +4 .. +125 | 0 |
| point_oracle_0.25 | 22 | 23 .. 307 | 35 .. 75 | -25 .. +235 | 7 |
| point_oracle_0.5 | 27 | 19 .. 368 | 35 .. 75 | -48 .. +296 | 8 |

The baseline clears the support gate by 171+ on every pair and still refuses 27/28
on cheirality. The oracle arms sit near the floor: at 0.25 and 0.5 the refit
support falls below it on 7 and 8 pairs respectively.

### 5.2 Truth-correspondence count versus first stop

The fixed surface truth per pair ranges 1..159 (total 1,213); the same-surface
within-tolerance denominators per pair are the `pt` sweep in section 6. For the
five oracle arms the injected candidate graph *is* that arm's truth-edge list, so
`candidate_edges` equals the per-pair arm truth count.

**All 63 `selection_absent` records have `candidate_edges < support_floor`
(0 exceptions).** Examples: `surface_oracle` `00-01` 20 < 35; `surface_oracle`
`01-02` 37 < 75; `point_oracle_0.05` `02-06` 53 < 75; `point_oracle_0.25`
`01-07` 22 < 49; `point_oracle_0.5` `01-07` 47 < 49. Conversely no pair with
`candidate_edges >= support_floor` is `selection_absent`. This is an arithmetic
support-count fact implied by the unchanged selector, not a causal finding: when
the whole candidate graph holds fewer edges than the support floor, no mode can
reach the floor.

### 5.3 Spatial-spread evidence

`min_spatial_bins = 4`; `spatial_bins_total` is always 16 (4x4 grid). For
best-mode pairs the minimum of left/right occupied bins is:

| Arm | min-bins distribution (best-mode pairs) | `low_spatial_spread` |
| --- | --- | ---: |
| production_baseline | 8:1, 9:4, 10:1, 11:2, 12:7, 13:10, 14:3 | 0 |
| surface_oracle | 4:8, 3:1 | 1 |
| point_oracle_0.05 | 4:7, 3:1 | 1 |
| point_oracle_0.125 | 4:11 | 0 |
| point_oracle_0.25 | 4:18, 3:3, 2:1 | 0 |
| point_oracle_0.5 | 4:20, 3:7 | 0 |

The one `low_spatial_spread` pair in both `surface_oracle` and
`point_oracle_0.05` is `02-04` with bins 4/3. The oracle arms generally clear the
spatial gate exactly at the 4-bin threshold; the baseline has 8-14 bins.

### 5.4 Candidate graph, modes and hypotheses

`hypothesis_count` is the number of retained per-mode diagnostics
(`max_modes_per_class = 3` over 4 yaw classes -> ceiling 12). The pre-injection
descriptor counters `raw_keypoints`, `ratio_survivors` and `mutual_survivors` are
identical across the six arms for every pair (pooled `mutual_survivors` = 618 for
all arms), confirming that the arms differ only in the injected candidate graph.

| Arm | hypothesis_count distribution | sum | candidate_edges sum |
| --- | --- | ---: | ---: |
| production_baseline | 12 on all 28 pairs | 336 | 178,056 |
| surface_oracle | 0:19, 1:3, 2:6 | 15 | 1,213 |
| point_oracle_0.05 | 0:20, 1:2, 2:6 | 14 | 1,068 |
| point_oracle_0.125 | 0:17, 1:3, 2:6, 4:2 | 23 | 1,780 |
| point_oracle_0.25 | 0:6, 1:6, 2:3, 3:1, 4:2, 5:1, 7:1, 8:2, 9:1, 10:4, 11:1 | 111 | 5,037 |
| point_oracle_0.5 | 0:1, 1:2, 2:1, 4:2, 5:1, 6:2, 10:1, 11:2, 12:16 | 253 | 13,545 |

A `hypothesis_count` of 0 is exactly a `selection_absent` record. The baseline's
constant 12 reflects its dense 178k-edge graph; the oracle arms only reach double
digit hypothesis counts once the looser 0.25/0.5 graphs supply enough modes.

### 5.5 Ambiguity refusals

Every one of the 28 `ambiguous_hypothesis` refusals (2 at 0.125, 13 at 0.25, 13 at
0.5) has `ambiguity_support_margin` of 0 or 1 against the required 2. The retained
per-hypothesis diagnostics show that in every case at least one rival mode also
reaches `support_floor`, and the rival's yaw class is most often 0 (the winner's
class) — i.e. the near-tie is between angularly distinct modes of the same yaw
class. Examples: `point_oracle_0.25` `02-03` winner support 149 vs rival 149
(margin 0, 3 same-class modes above floor); `point_oracle_0.5` `05-07` winner 202
vs rival 202 with 9 modes at/above floor. This is a description of the retained
records, not a causal claim.

### 5.6 Explicitly missing fields

* For every `selection_absent` record all six gate fields
  (`cheirality_fraction`, `parallax_degrees`, `singular_ratio`,
  `median_selected_residual_degrees`, `descriptor_cost`, `ambiguity_margin`) are
  `null`, and `spatial_bins_left/right` are 0. Those gates were never evaluated;
  they are reported as missing, not as failing values.
* `ambiguity_margin` is `null` whenever no rival mode reaches the floor (including
  all `solve_pair_refused` and `solved` records).
* `pose_error` / `pose` fields are undefined except for the 21 solved records:
  of the 31 records that reach the solver, 21 are `pose_solved` and carry a
  defined pose error, while 10 are `pose_abstained` and carry none.
* `axis_stage` and `placed_cell_geometry` are typed unavailable at this pair-level
  seam (section 2).

## 6. Pair-by-pair comparison across tolerance arms

`vNN` = `view-canal-town-architectural-component-bbox-NN`. `T` is the fixed surface
denominator; `pt` is the same-proximity denominator at 0.05 / 0.125 / 0.25 / 0.5.
Stage codes: `ABS` = selection_absent, `SUP` = support_below_floor, `SPA` =
low_spatial_spread, `CHE` = cheirality_failed, `AMB` = ambiguous_hypothesis,
`SREF` = solve_pair_refused, `SOLVED`.

| # | Pair | T | pt (0.05/0.125/0.25/0.5) | base | surf | p05 | p125 | p25 | p5 |
| ---: | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | v00-v01 | 20 | 16/34/82/202 | CHE | ABS | ABS | ABS | AMB | SOLVED |
| 2 | v00-v02 | 4 | 4/12/60/194 | CHE | ABS | ABS | ABS | SUP | AMB |
| 3 | v00-v03 | 1 | 1/4/15/48 | CHE | ABS | ABS | ABS | ABS | SUP |
| 4 | v00-v04 | 2 | 1/7/32/100 | CHE | ABS | ABS | ABS | ABS | SUP |
| 5 | v00-v05 | 6 | 5/15/58/156 | CHE | ABS | ABS | ABS | SUP | AMB |
| 6 | v00-v06 | 3 | 3/8/45/126 | CHE | ABS | ABS | ABS | SUP | SUP |
| 7 | v00-v07 | 4 | 3/15/37/101 | CHE | ABS | ABS | ABS | SUP | AMB |
| 8 | v01-v02 | 37 | 31/68/201/594 | CHE | ABS | ABS | ABS | SOLVED | AMB |
| 9 | v01-v03 | 11 | 10/25/76/219 | SOLVED | ABS | ABS | ABS | SUP | SOLVED |
| 10 | v01-v04 | 7 | 5/19/59/157 | CHE | ABS | ABS | ABS | ABS | SUP |
| 11 | v01-v05 | 18 | 13/26/57/152 | CHE | ABS | ABS | ABS | ABS | SUP |
| 12 | v01-v06 | 4 | 3/8/32/90 | CHE | ABS | ABS | ABS | ABS | SUP |
| 13 | v01-v07 | 9 | 9/15/22/47 | CHE | ABS | ABS | ABS | ABS | ABS |
| 14 | v02-v03 | 91 | 86/113/265/731 | CHE | SOLVED | SOLVED | SOLVED | AMB | SOLVED |
| 15 | v02-v04 | 102 | 90/154/433/1186 | CHE | SPA | SPA | SOLVED | AMB | AMB |
| 16 | v02-v05 | 99 | 86/144/420/1105 | CHE | SREF | SREF | SREF | AMB | SOLVED |
| 17 | v02-v06 | 68 | 53/107/302/863 | CHE | ABS | ABS | SOLVED | AMB | AMB |
| 18 | v02-v07 | 17 | 16/32/145/347 | CHE | ABS | ABS | ABS | AMB | AMB |
| 19 | v03-v04 | 63 | 62/78/204/504 | CHE | SOLVED | ABS | SOLVED | AMB | AMB |
| 20 | v03-v05 | 51 | 47/59/159/388 | CHE | ABS | ABS | ABS | SOLVED | AMB |
| 21 | v03-v06 | 24 | 22/33/93/226 | CHE | ABS | ABS | ABS | SUP | SUP |
| 22 | v03-v07 | 6 | 6/10/55/110 | CHE | ABS | ABS | ABS | SUP | SUP |
| 23 | v04-v05 | 140 | 123/185/482/1272 | CHE | SREF | SREF | SREF | AMB | CHE |
| 24 | v04-v06 | 88 | 79/130/381/1106 | CHE | SOLVED | SOLVED | SOLVED | AMB | AMB |
| 25 | v04-v07 | 39 | 30/57/172/440 | CHE | ABS | ABS | SOLVED | AMB | AMB |
| 26 | v05-v06 | 159 | 145/210/590/1635 | CHE | SREF | SREF | AMB | AMB | CHE |
| 27 | v05-v07 | 77 | 63/114/258/675 | CHE | SREF | SREF | AMB | AMB | AMB |
| 28 | v06-v07 | 63 | 56/98/302/771 | CHE | SOLVED | SOLVED | SOLVED | AMB | AMB |

Fixed-surface reach per pair. Each cell is `candidate truth returned / selected
truth returned` against that pair's own surface truth `T` (so the candidate column
is capped at `T`). The tolerance denominator moves independently of this fixed
denominator.

| # | Pair | base | surf | p05 | p125 | p25 | p5 |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | v00-v01 | 4/0 | 20/0 | 16/0 | 20/0 | 20/17 | 20/15 |
| 2 | v00-v02 | 1/0 | 4/0 | 4/0 | 4/0 | 4/3 | 4/4 |
| 3 | v00-v03 | 0/0 | 1/0 | 1/0 | 1/0 | 1/0 | 1/1 |
| 4 | v00-v04 | 0/0 | 2/0 | 1/0 | 2/0 | 2/0 | 2/1 |
| 5 | v00-v05 | 0/0 | 6/0 | 5/0 | 6/0 | 6/6 | 6/1 |
| 6 | v00-v06 | 0/0 | 3/0 | 3/0 | 3/0 | 3/2 | 3/2 |
| 7 | v00-v07 | 0/0 | 4/0 | 3/0 | 4/0 | 4/3 | 4/1 |
| 8 | v01-v02 | 7/1 | 37/0 | 31/0 | 37/0 | 37/34 | 37/31 |
| 9 | v01-v03 | 1/0 | 11/0 | 10/0 | 11/0 | 11/10 | 11/8 |
| 10 | v01-v04 | 2/0 | 7/0 | 5/0 | 7/0 | 7/0 | 7/5 |
| 11 | v01-v05 | 2/1 | 18/0 | 13/0 | 18/0 | 18/0 | 18/16 |
| 12 | v01-v06 | 0/0 | 4/0 | 3/0 | 4/0 | 4/0 | 4/2 |
| 13 | v01-v07 | 0/0 | 9/0 | 9/0 | 9/0 | 9/0 | 9/0 |
| 14 | v02-v03 | 31/2 | 91/89 | 86/85 | 91/89 | 91/86 | 91/55 |
| 15 | v02-v04 | 28/0 | 102/99 | 90/88 | 102/99 | 102/88 | 102/80 |
| 16 | v02-v05 | 33/2 | 99/96 | 86/84 | 99/95 | 99/86 | 99/68 |
| 17 | v02-v06 | 20/3 | 68/0 | 53/0 | 68/66 | 68/57 | 68/53 |
| 18 | v02-v07 | 2/0 | 17/0 | 16/0 | 17/0 | 17/12 | 17/9 |
| 19 | v03-v04 | 21/1 | 63/63 | 62/0 | 63/62 | 63/59 | 63/55 |
| 20 | v03-v05 | 9/2 | 51/0 | 47/0 | 51/0 | 51/44 | 51/44 |
| 21 | v03-v06 | 3/0 | 24/0 | 22/0 | 24/0 | 24/21 | 24/15 |
| 22 | v03-v07 | 1/0 | 6/0 | 6/0 | 6/0 | 6/3 | 6/2 |
| 23 | v04-v05 | 46/21 | 140/137 | 123/121 | 140/136 | 140/132 | 140/41 |
| 24 | v04-v06 | 24/0 | 88/85 | 79/77 | 88/85 | 88/80 | 88/56 |
| 25 | v04-v07 | 10/1 | 39/0 | 30/0 | 39/39 | 39/37 | 39/36 |
| 26 | v05-v06 | 54/17 | 159/158 | 145/144 | 159/157 | 159/146 | 159/45 |
| 27 | v05-v07 | 20/1 | 77/75 | 63/62 | 77/74 | 77/74 | 77/63 |
| 28 | v06-v07 | 26/0 | 63/63 | 56/56 | 63/63 | 63/57 | 63/43 |

### 6.1 What moves across the sweep

* **Tolerance-specific denominator** grows with the bound per pair (e.g. `v02-v05`
  86 / 144 / 420 / 1,105 while `T = 99`); the fixed 1,213 surface denominator and
  each pair's `T` are unchanged by the sweep.
* **Fixed-surface reach** changes only as the candidate graph changes. The
  candidate column is capped at `T` once the bound reaches 0.125; at 0.05 it is
  below `T` on 23 of 28 pairs (only `v00-v02`, `v00-v03`, `v00-v06`, `v01-v07`
  and `v03-v07` still match `T`).
* **Selector retention** (pooled, unique-pixel-pair returned; and fixed-surface
  truth retention): baseline 8,863/169,586 = 5.23% (truth 52/345 = 15.07%);
  surface 865/1,213 = 71.31% (71.31%); 0.05 717/1,068 = 67.13% (67.13%);
  0.125 1,253/1,780 = 70.39% (truth 965/1,213 = 79.55%); 0.25 2,526/5,037 =
  50.15% (truth 1,057/1,213 = 87.14%); 0.5 3,450/13,545 = 25.47% (truth
  752/1,213 = 62.00%). Retention is non-monotone; the loosest arms retain the most
  truth pairs but the smallest fraction of their own larger graph.
* **Solve outcomes** per arm: 1, 4, 3, 7, 2, 4 of 28 (baseline, surface, 0.05,
  0.125, 0.25, 0.5) -- 21 solved records in total; the 31 records that reach the
  solver comprise these 21 `pose_solved` and 10 `pose_abstained` refusals. Solved
  sets: baseline `{v01-v03}`; surface
  `{v02-v03, v03-v04, v04-v06, v06-v07}`; 0.05 `{v02-v03, v04-v06, v06-v07}`;
  0.125 `{v02-v03, v02-v04, v02-v06, v03-v04, v04-v06, v04-v07, v06-v07}`; 0.25
  `{v01-v02, v03-v05}`; 0.5 `{v00-v01, v01-v03, v02-v03, v02-v05}`.
* **First-stop stage transitions relative to `surface_oracle`** (unchanged /
  changed): 0.05 27/1 (only `v03-v04` `SOLVED -> ABS`); 0.125 23/5
  (`v02-v04` `SPA -> SOLVED`, `v02-v06` `ABS -> SOLVED`, `v04-v07` `ABS ->
  SOLVED`, `v05-v06` and `v05-v07` `SREF -> AMB`); 0.25 6/22 (most `ABS` pairs
  become `SUP` or `AMB`, and all four surface solves become `AMB`); 0.5 2/26
  (only `v01-v07` and `v02-v03` keep their surface stage). The baseline changes
  the stage of all 28 pairs relative to the surface oracle, 27 of them to
  `CHE`.
* The two `cheirality_failed` records at 0.5 (`v04-v05`, `v05-v06`) have
  `cheirality_fraction = 0.0` despite support 314 and 368. The baseline's 27
  cheirality failures span `0.0107 .. < 0.75` with a failed-only median of
  0.131068; the median across all 28 baseline pairs is 0.144345 (only
  `v01-v03` reaches 0.9914).

## 7. What the current evidence supports, and what remains unidentified

**Supported (descriptive):**

* Every arm's per-pair `reason` is the first gate the ranked winner fails; the
  ordered classifier reproduces all 168 sealed reasons with 0 mismatches.
* The baseline's dominant first stop is `cheirality_failed` (27/28) with support
  171+ above the floor and 8-14 spatial bins (failed-only cheirality median
  0.131068; median across all 28 baseline pairs 0.144345): on this fixture the
  production graph fails at the depth-sign (cheirality) gate, not at candidate
  availability or support.
* Within the oracle arms the dominant first stop is `selection_absent` (19/20/17
  at 0.0625/0.05/0.125), and **every** such record has fewer candidate edges than
  the raw-keypoint-derived `support_floor` (63/63).
* Once the looser 0.25/0.5 graphs clear support, the dominant first stop becomes
  `ambiguous_hypothesis` (13/28 each), always with `ambiguity_margin` 0 or 1 and a
  rival same-yaw-class mode at or above the floor.
* Solve counts are non-monotone in the bound (1, 4, 3, 7, 2, 4 of 28), and the
  single baseline solve is the wrong yaw quadrant (R5O section 6.3).

**Not identified by this audit:**

* Whether the cheirality failures of the dense production graph are caused by the
  production candidate edges, by the assisted-camera precondition, or by the
  DS-11 depth convention cannot be separated from these records; the audit only
  reports that the gate is the first failure.
* Whether the `selection_absent` refusals are "too few truth edges" or "support
  floor too high relative to this fixture's detection density" cannot be separated
  without changing one of them; the two quantities are arithmetically tied here.
* Within-face correspondence ambiguity is **not** established as a cause: the
  sweep moves candidate count, achieved surface precision, the per-arm
  denominator and the ambiguity gate together.
* No causal ranking of support, spatial, cheirality and ambiguity is claimed from
  correlation alone; no plane-graph/axis or placement record exists at this seam.

## 8. Next evaluation-only experiment (one factor)

Do **not** repeat the fixed-floor-8 control. DS-R5N already ran the same surface
oracle with `min_support_absolute=8` and `min_support_fraction=0`: selected reach
rose to 1,152/1,213, but solves fell to 0/28 as ambiguity increased. This directly
rules out another floor-only pass on the same arm as the next useful test.

The solved-pair adjacency derived from this artifact is also incomplete for every
arm. Largest connected components over the eight frame IDs are: production
baseline 2/8, surface oracle 5/8, 0.05 3/8, 0.125 5/8, 0.25 2/8, and 0.5 5/8.
This is graph connectivity only; it does not claim pose-graph consistency or
camera recovery. Even the assisted-oracle pair solves do not supply a connected
edge set spanning all eight views in this fixture.

Recommended next experiment: a **single-factor detector-budget sensitivity** on
the same pinned eight views, using the 4,000-feature baseline and one 8,000-feature
arm. Hold cameras/masks, SIFT implementation and other detection settings, ratio
test, truth rule (use the fixed 0.0625 same-surface rule for the primary comparison),
support-floor formula, spatial/rank/parallax/cheirality/ambiguity gates, and DS-11
solver fixed. Rebuild only detected keypoints/descriptors under the changed
`max_features`; report P candidate/selected metrics, support and refusal stages,
and solved-pair component sizes against the same eight frames. This tests whether
additional detector coverage supplies missing bridge pairs or only adds
ambiguous/cheirality failures. Keep it evaluator-only; no production threshold
recommendation follows from this single scene.

## 9. Limitations

* Descriptive audit of one sealed run (one 8-view Canal Town export, 28 pairs,
  six arms); no generalization, no accuracy or model-performance claim.
* The artifact stops at the pair matcher/solver; plane-graph/axis and placed-cell
  records do not exist and are not inferred.
* `point_oracle_*` arms are same-surface continuous world-hit proximity, not
  feature identity; refusal reasons are policy outcomes, never correctness.
* Per-hypothesis `support` and `ambiguity_margin` describe retained modes; a mode
  is not a truth correspondence. Ambiguity margin 0/1 is a gate statistic.
* Gate fields are absent (not zero) whenever no best mode exists; this audit
  reports them as missing.
* The assisted factory camera is an oracle input never used by normal
  reconstruction; the baseline's cheirality failures occur under that assisted
  precondition.
* No causal claim is made; all cross-tabs are associations within a single
  fixture.

## 10. Reproduction and hashes

Method (read-only; no temp files, no other artifact/tool):

```
Get-FileHash .tmp-exact-point-oracle-20261004.json -Algorithm SHA256
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -c "<inline read-only parser over the single R5O JSON>"
```

The inline parser only: loaded the artifact; recomputed
`evaluation_pair_correspondence.seal_record`; classified each arm-pair through the
ordered gate ladder; and aggregated the tables above. It wrote nothing and spawned
nothing.

* Sealed artifact: `.tmp-exact-point-oracle-20261004.json`, 2,855,980 bytes,
  sha256 `03faaefe40ac1496ee6f86800818f070bb3db043984d092025a8dff67c65007d`,
  seal `5d8a67edbb6aeb1c0dadc998f1ffa2610718e5cd36e84a5a98c11c0ab5e73899`
  (recomputed match), `consistency.ok = true`.
* Accepted inputs: `docs/reports/image-only-geometry-exact-point-oracle-2026-10-04.md`
  (R5O), `docs/reports/oracle-support-sensitivity-2026-10-03.md` (R5N).
* This report cannot carry its own post-edit digest; its final sha256 is recorded
  in the task result and the ASCII `AGENT_COMMS.md` entry.

## 11. Stop condition

Stopped after the read-only parse and this report for Luna verification and Astra
read-only acceptance. No rerun, reconstruction, matcher/solver execution, source
or test edit, protected/fixture/R5K/R5N read, artifact write, temp file, install,
background job, recursive deletion, reparse/junction operation, encoded command,
Defender/system change or commit.
