# DS-16B — Preregistered Oracle-Tolerance Sweep of the Frozen DS-16 Candidates

- Date: 2026-09-12 · Author: DeepSeek DS-16BR · Status: evaluation-only and reference-assisted; every label below comes from fixture manifests/litematics and must never enter a production path.
- Directives: `AGENT_COMMS.md` 12:25 (DS-16B grid + stop rule) and 12:38 (DS-16BR: correct boundedness; preserve the aborted log; report both stop-rule predicates separately; state the original combined conclusion is rejected by lead review; run all 12 fixed cells once).
- Owned paths only: `scripts/diagnose_match_attribution_sweep.py`, `exports/diagnostics/ds16b/`, this report. No production/source/test/annotation edit, no re-detection, no threshold or gate change, no cell selection/tuning, no dependency, no delegation, no stage/commit/self-chain.

## 1. Boundedness correction (cause, fix, evidence)

The first DS-16B run was terminated at 3.8 GiB RSS. Its partial console log is preserved verbatim as `sweep_run_aborted_3.8GiB_rss.log` (2158 bytes, 32 lines, 12:36:00): it completed `d0.25_n30/n45/n60` and `d0.5_n30` and died at the start of `d0.5_n45`.

Two independent, additive causes — both fixed without touching any threshold, gate, cell or outcome:

1. **DS-11's full-matrix SVD (production code, deliberately unmodified).** `_robust_baseline`/`_null_vector` call `np.linalg.svd(rows, full_matrices=True)`. For an `(N, 3)` row matrix that returns U of shape `N x N`: `8*N**2` bytes, so 19617 rows (the `d0.5` many-to-many oracle cell) requested 3.08 GiB in one allocation. Local probe: N=2048 → 37.1 ms, 90.6 MiB peak RSS; N=4096 → 253.2 ms, 282.7 MiB peak RSS. Every DS-11 input is now deterministically bounded to `SOLVER_ROW_CAP = 4096` rows (8·4096² = 134 MiB), the smallest power of two above the largest frozen frame's keypoint count (3837), so the **complete per-left-keypoint best-of oracle cell is never subsampled**. Declared rule: stable stride subsample of the lexicographically sorted unique edge set, `step = ceil(rows/cap)`, keep `edges[::step]`, no RNG, identical for every cell and camera regime. Raw oracle counts and both recall denominators remain **exact**; each cell reports `rows_full`, `rows_used` and `sampled`.
2. **Unbounded pre-dedup oracle search.** The old search accumulated every hit of a whole cell (27 spatial-hash shifts × all left keypoints) before one `np.unique(np.vstack(...))`. It now de-duplicates inside a fixed 64-left-keypoint chunk and yields disjoint, sorted chunk sets; concatenating them is exactly the global unique set. Per-pair truth is released before the next pair and no edge payload survives a cell. A `TRUTH_MATERIALIZE_MAX_ROWS = 5e6` guard and a process-wide RSS watchdog (`RSS_LIMIT_BYTES = 1.5 GiB`, exact OS peak working set read at every checkpoint) abort to partial evidence instead of continuing.

Controls for the refactor: the chunked search is exactly equal to the frozen DS-16 `_truth_edges` on 192 synthetic label pairs (axis-aligned and arbitrary normals × 4 tolerances × 3 normal thresholds × random chunking), including a chunk-size-1 boundary stress and chunk disjointness/sortedness — 0 mismatches. In-run, the base cell `d0.5_n30` reproduces all 9 frozen DS-16 pairs exactly (60000 truths, 156 geometric-true, identical hit counts, base-cell cross-check `mismatch_count = 0`).

Result: **peak RSS 618.1 MiB** (from 3.8 GiB, budget 1.5 GiB), total runtime 215.8 s (render/label 9.6 s, oracle search 2.9 s, DS-11 solves 197.2 s). One run, 12 cells, exit 0. Output digest verified with `Get-FileHash` after the run.

## 2. Frozen inputs, grid, and the inert normal axis

Frozen DS-16 digests re-verified before any label: `match_candidates.json` `e062b985…`, `match_candidates.npz` `3f7b4555…`, `match_attribution.json` `72da240a…`. Nine T2/T6 pairs (the DS-16 `oracle_truths >= 8` set); all 28 `t3-sparse-8` pairs excluded from every conclusion. Render scale 0.35, alignment median 1.1458 px (≤ 10), reference-input digests recorded in the JSON.

**The face-normal axis is inert on this fixture family.** cos 30° / 45° / 60° give identical sweep results in all four distance bands, because block faces are axis-aligned: a same-face pair has normal dot 1 and any perpendicular pair has dot 0, so every cosine threshold in (0, 1) separates them identically (only the recorded peak-RSS field differs). The 12 preregistered cells therefore collapse to 4 distinct tolerance results; no threshold was changed.

## 3. Grid results (each distance row covers its three normal cells; T2/T6, 9 pairs)

| distance (blocks) | oracle truths | left-truth kp | geometric true (prevalence) | bo directed / reciprocal / selected | mm directed / reciprocal / selected | oracle best-of solved cur · ref (within 5°) | oracle full solved cur · ref (pairs sampled; rows used/full) | intersection solved cur · ref (evaluable, within 5°) |
|---|---|---|---|---|---|---|---|---|
| 0.25 | 16034 | 3252 | 106 (0.366%) | 1.000 / 0.690 / 0.141 | 0.01179 / 0.00661 / 0.00118 | 6 · 6 (6 · 6) | 5 · 5 (1 pair; 13513/16034) | 2 · 2 (5, 2 · 2) |
| 0.5 | 60000 | 4512 | 156 (0.538%) | 1.000 / 0.705 / 0.137 | 0.00542 / 0.00260 / 0.00045 | 6 · 6 (6 · 6) | 6 · 6 (6 pairs; 22676/60000) | 3 · 3 (6, 3 · 3) |
| 1.0 | 224056 | 6286 | 397 (1.369%) | 1.000 / 0.707 / 0.131 | 0.00432 / 0.00177 / 0.00029 | 5 · 6 (5 · 6) | 1 · 1 (7 pairs; 30817/224056) | 2 · 2 (7, 2 · 2) |
| 2.0 | 856320 | 8557 | 1265 (4.362%) | 1.000 / 0.698 / 0.122 | 0.00373 / 0.00148 / 0.00019 | 6 · 6 (5 · 5) | 1 · 1 (9 pairs; 34053/856320) | 3 · 3 (8, 2 · 2) |

- `bo` = per-left-keypoint best-of recall (the new read-out); `mm` = DS-16's many-to-many recall. The `d0.5` many-to-many values reproduce DS-16's reported aggregate to its printed precision (0.005417 / 0.002600 / 0.000450); the base-cell cross-check inside the run proves the counts exactly.
- **Oracle refusals are typed adjudication failures, not data failures.** With the complete per-left best-of set (3252–8557 rows, never subsampled), 3 of 9 pairs are refused at every cell: `ambiguous_yaw_class` ×3 at d0.25/d0.5/d1; `ambiguous_yaw_class` ×2 + `low_inlier_fraction` ×1 at d2. Over the grid 4 distinct pairs ever fail this complete oracle read-out: T2 `13.22.14→13.24.24`, T2 `13.22.14→13.24.33` (all 12 cells), T2 `13.24.24→13.24.33` (d0.25 and d2), T6 `18.01.09→18.01.14` (d0.5 and d1). Median rotation error for the complete best-of oracle solves stays ≤ 0.15° (current cameras; ≤ 0.16° including the intersection cells) and 0.0° (reference), with median gauge-aligned baseline error ≤ 0.36° (current best-of) / 0.32° (reference best-of; ≤ 0.99° including the intersection cells); the sampled many-to-many cell at d2 is the sole exception (its one "solve" is a 90° quadrant with 26.9° baseline error, inside the `low_inlier_fraction` regime described next).
- **The many-to-many oracle cell stops being an oracle as tolerance widens.** At d1/d2 it is sampled (7/9 and 9/9 pairs) and collapses to 1 solved with `low_inlier_fraction` dominating (5 and 8 pairs): the loosest "truth" set is dominated by redundant far-apart keypoint pairs whose epipolar rows do not concentrate, so it fails the 0.5 inlier-fraction gate. This is a property of the tolerance definition, **not** evidence about the matcher, and the tolerance-invariant complete best-of ceiling (6/9) is the oracle read-out to use.
- **The reciprocal/one-to-one graph is the large measured loss.** The intersection of the reciprocal candidate graph with the oracle truth set is only 106 / 156 / 397 / 1265 edges (0.66% → 0.15% of the truth set), evaluable by DS-11 for only 5–8 of 9 pairs, and solves 2–3. Meanwhile best-of directed recall is 1.000 in every cell — every left keypoint that has any oracle-consistent partner has at least one such partner inside its directed top-5 — and reciprocity keeps ~70%. The drop is then inside DS-14's one-to-one selection: selected best-of recall is 0.122–0.141. The graded ladder (1.00 → ~0.70 → ~0.13) is descriptive; no threshold is tuned and no cell is chosen.

## 4. Corrected stop rule — the two predicates reported separately

The DS-16B directive's written rule ("oracle ceiling ≤ 8/9 **OR** best-of directed recall < 5% in every cell ⇒ the correspondence representation is binding") is reported by the script as `original_combined_conclusion.status = "rejected_by_lead_review"`, with its mechanically computed `as_written_verdict` (`correspondence_representation_binding`) retained only for transparency. Reason recorded: a low oracle ceiling is evidence of **pose-adjudication** failure, not of correspondence representation, so the disjunction conflated two independent mechanisms. The predicates are now reported separately, with the preregistered decomposition:

| predicate | result | reading |
|---|---|---|
| **P1** oracle solved ceiling ≤ 8/9 | **holds** — ceiling **6/9** (same 6/9 for the complete best-of cells alone) | **pose adjudication**: ideal-edged, per-left-complete oracle sets still fail typed DS-11 gates under both current and reference cameras |
| **P2** aggregate best-of directed recall < 5% in *every* cell | **does not hold** — 1.000 in all 12 cells (min = max = 1.000) | **detection/candidate representation is not the binding bottleneck** in this grid; P2's "almost no directed support" reading does not apply here |

Preregistered decomposition tallies (per pair × cell): pose adjudication **3 pairs/cell, 4-pair union** (complete ideal-edge oracle unsolved); detection/candidate representation **0** (no cell/pair has zero best-of directed support); graph filtering/selection **0** under the rule's strict "total absence" form (directed > 0 but reciprocal and selected both 0). The 0 for filtering is a threshold artifact of the boolean rule, not a null result: the graded ladder in §3 (1.00 → 0.70 → 0.13) shows the one-to-one/reciprocal stage discards roughly five-sixths of the surviving true support. Both statements are reported; neither is tuned.

**Net correction.** In the 9-pair T2/T6 universe, increasing the oracle tolerance from 0.25 to 2.0 blocks does not raise the complete oracle ceiling above 6/9 and never removes directed candidate support, so the earlier combined conclusion cannot be read as "the descriptor/ratio/top-k representation is binding". The sweep's positive findings are: (a) 3 pairs are un-adjudicable even with ideal correspondences (`ambiguous_yaw_class` dominant); (b) the remaining loss is concentrated after the directed stage — reciprocity and DS-14 one-to-one selection — not in directed top-k candidate recall. Per the fixed stop rule's action clause, **no matcher gate was tuned**, and DS-14 remains research-only.

## 5. Limitations

- Nine pairs from two fixtures (T2, T6) at render scale 0.35; t3-sparse-8 excluded from every conclusion, so nothing here explains the 28 zero-oracle pairs.
- `oracle_full` at d1/d2 is a declared deterministic stride subsample (7/9 and 9/9 pairs) and is reported as such; only the complete best-of oracle read-out is gate-independent across the grid.
- Recall denominators are per-left-keypoint oracle sets that are themselves tolerance-conditional (3252 → 8557 left keypoints); the mm/bo contrast is the point, and neither is comparable across distance rows without the row label.
- Renders and labels are self-consistent with the manifest cameras (median 1.146 px round-trip) but that check is tautological with respect to photo-vs-camera registration, as DS-16 already notes.
- No random sampling, no RNG anywhere in this sweep; the `ambiguous_yaw_class`/`cheirality`/`inlier-fraction` gates are DS-11 defaults, unchanged.

## 6. Artifacts (this run; one run, no reruns)

| path | bytes | sha256 |
|---|---|---|
| `exports/diagnostics/ds16b/match_attribution_sweep.json` | 882302 | `397294502f902f84d54abee047bc1a52939f843fd8016147577e5020289f1cf0` |
| `exports/diagnostics/ds16b/oracle_sweep_cells.csv` | 37936 | `f8e54a3d2568174b989c8a9fffd238665dfeb91dec3d8bf10b3b72d70923f875` |
| `exports/diagnostics/ds16b/oracle_sweep_aggregate.csv` | 3531 | `a1183010c0cffe6c78ae5c359ab32a23712edce96476a96306b98fc5fe9fec60` |
| `exports/diagnostics/ds16b/sweep_run.log` | 11214 | `5af0a1a2c61c94f55a2f2233693e7951303a42056da4d5ec8210d580aa05bead` |
| `exports/diagnostics/ds16b/sweep_run_aborted_3.8GiB_rss.log` (preserved first-run evidence) | 2158 | `fe18332296418d002a709361565de1c2f17bed06451cd69345bf8c396d36a404` |

Recipe: `.venv\Scripts\python.exe scripts/diagnose_match_attribution_sweep.py` (exit 0). Peak RSS 618.1 MiB; frozen-input, base-cell and digest verifications all passed; no repository file outside the DS-16B paths was written (the chunk-equivalence probe ran from the platform temp directory only).
