# DS-18 — Directed-Candidate Research Matcher and Blind 37-Pair Diagnostic

- Date: 2026-09-12 · Author: DeepSeek DS-18 · Status: research-only; **not** production, **not** promoted, no gate tuned.
- Directives: `AGENT_COMMS.md` 12:51 (allocate DS-18: directed-candidate matcher, no reciprocity; preregistered promotion criterion) and the DS-16B report's accepted reading that the remaining loss sits after the directed stage.
- Owned paths only: new `src/mc_photogrammetry/pipeline/directed_matching.py`, new `tests/test_directed_matching.py`, new `scripts/diagnose_directed_pairs.py`, generated `exports/diagnostics/ds18/`, this report. No existing source/test/threshold edit, no re-detection, no DS-12/14/15 artifact mutation, no production integration, no delegation, no stage/commit/self-chain.

## 1. What was added, and what was deliberately not

| path | role |
|---|---|
| `src/mc_photogrammetry/pipeline/directed_matching.py` | directed left→right top-k graph + per-left-first geometric matcher (research-only) |
| `tests/test_directed_matching.py` | 15 public-output synthetic controls |
| `scripts/diagnose_directed_pairs.py` | two-phase blind diagnostic over the exact 37 DS-15 pairs |
| `exports/diagnostics/ds18/` | frozen phase-1 estimates and phase-2 evaluation (+ logged run) |

`directed_matching` reuses DS-12/DS-14 primitives unchanged: `correspondence.detect_features`, `usable_mask_for`, the descriptor-footprint mask, the blocked Gram distance kernel, and DS-14's `_residuals`, `_ry90`, `_world_to_camera`, `_cameras_ok`, `_camera_rays`, `_sample_rank_ok`, `_singular_values`, `_median_angle_degrees`, `_vector_angle_degrees`, `_positive_depth_fraction`, `_bins_covered` and `_winner_refusal`. `DirectedMatchSettings` **is** `JointMatchSettings` (`directed_matching.DirectedMatchSettings is JointMatchSettings`; asserted in the tests), so no bound or gate is re-declared. `solve_pair` is the public DS-11 solver.

## 2. Matcher design (differences from DS-14)

1. **Directed candidate graph, no reciprocity.** `directed_topk_graph` keeps each left keypoint's `k = 5` nearest right descriptors whatever the right side thinks; sorted by `(distance, left, right)`, capped at 8192, truncation reported. `candidate_edges` for the 37 pairs totals **173152** edges versus the DS-14 reciprocal graph's **57832** (**2.99×** denser); 6/37 pairs hit the cap.
2. **Hypotheses unchanged.** Per yaw class, deterministic seeded RANSAC draws 2 endpoint-distinct edges, rejects rank-deficient samples and solves the unit baseline. The all-rows candidate uses a Gram-matrix null vector (`AᵀA`, `3×3`) instead of DS-14's `np.linalg.svd(rows, full_matrices=True)`; the latter materialises an `8·M²` left factor (the allocation DS-16B had to cap at 4096 rows). The direction is identical up to sign, and every consumer uses it only through `|aᵀt|` residuals and an absolute-dot comparison.
3. **Per-left-first selection.** For a hypothesis, each left endpoint keeps its single best residual-qualified edge; only then are right collisions resolved in a deterministic `(residual, distance, left, right)` order into a one-to-one set. No directed truth is dropped before geometry sees it.
4. **Mode pruning and ranking.** Proposals are pruned and ranked by the resolved one-to-one support, with the pre-collision unique-left and unique-right supports as the next keys, then spatial spread, median selected residual and descriptor cost. Two implementation notes, fixed **before** any reference evaluation and driven only by synthetic controls:
   - pruning by the many-to-many edge count was rejected: for these near-forward pairs the DS-14 residual `|aᵀt|/‖a‖` is the angle between `t` and each edge's epipolar plane, and planes for near-parallel rays are ill-conditioned, so a degenerate baseline fits hundreds of false edges (the winner's all-edge median residual is only **8.34°** across the 37 pairs);
   - ranking on the *pre-collision* unique-left support first was rejected as well, because a mode whose qualified edges mass-collide (high unique-left, low unique-right) outranked a genuinely one-to-one mode. After collision resolution unique-left and unique-right support are equal, so the resolved support **is** both listed statistics.
5. **Gates.** DS-14's support floor `max(8, ⌈0.02·min(raw keypoints)⌉)`, spatial spread (≥4/16), conditioning, parallax (≥3°), cheirality (≥0.75) and the distinct-hypothesis margin are applied to the winner unchanged; the winner's one-to-one set is handed to DS-11 `solve_pair` and the classes must agree. The matcher abstains rather than relaxing any gate.

## 3. Synthetic controls (`tests/test_directed_matching.py`, 15 tests)

All pass. The controls are public-output only (projected 3D points through the shipped camera API, injected `FrameFeatures`), one-to-one checked by pixel:

- **directed truth fails reciprocity yet solves** — one exact-descriptor decoy per true point with `neighbors=1`: 20/20 true edges in the directed graph, **0/20** in the reciprocal graph, and the directed matcher solves class 1 with a ≤1° baseline;
- **no reciprocity requirement / sorted / deterministic** on a pure descriptor table; the reciprocal graph keeps only its own best edge;
- **cap truncation** at graph and matcher level (`total`, `truncated` reported; deterministic tie-break);
- **exact scene** solves through the public DS-11 pose (class 1, ≤1° rotation and baseline, support 20 = unique-left = unique-right);
- **right collisions** — four left endpoints (one true, three pixel-shifted duplicates of the same descriptor) compete for one right endpoint: `left_support = 23 > support = 20 = right_support`, final set one-to-one;
- **repeated descriptors** (declared 12-point repetition mass) plus the unique consistent minority solves class 1; larger masses refuse (see §5);
- **competing baselines** refuse `ambiguous_hypothesis`; **identity self-match** and **few-point support** refuse with typed reasons;
- **determinism**, **unequal explicit intrinsics**, **descriptor-footprint boundary filtering with excluded-RGB invariance**, **frozen DS-12 default boundary behaviour**, and **`DirectedMatchSettings is JointMatchSettings`**.

Focused regression: `tests/test_directed_matching.py tests/test_connected_camera.py tests/test_correspondence.py tests/test_joint_matching.py tests/test_camera.py` → **67 passed in 11.97 s**.

## 4. Blind 37-pair diagnostic (one invocation; phase 1 frozen and digested before phase 2)

Phase 1 opened only the 15 frozen screenshots and the three image-derived M3 calibration artifacts (digests recorded in `directed_estimates.json`); `load_camera_manifest` is imported inside the phase-2 function only. The pair rule is DS-15's `frozen_frame_sets()` verbatim (T2 3 frames, T6 4, `t3-sparse-8` 8 → 37 unordered pairs), `usable_masks=None`, matcher/gate defaults. `directed_estimates.json` was written and digested (`cec240d8…` canonical) before phase 2 opened `cameras.json` and the DS-15 comparator artifact.

| metric | directed (DS-18) | DS-14 comparator (frozen DS-15) |
|---|---|---|
| pairs | 37 | 37 |
| solved | **5** | 8 |
| class correct | **2** | 4 |
| false solves | 3 | 4 |
| precision among solves | **0.40** | 0.50 |
| rotation ≤5° | 2 | 4 |
| median relative-rotation error | 89.95° | 45.12° |
| median gauge-aligned baseline error | 33.53° | 16.20° |
| refusal reasons | `cheirality_failed` 30, `low_spatial_spread` 2 | `cheirality_failed` 23, `ambiguous_hypothesis` 2, `low_spatial_spread` 4 |
| T2 solved / correct / false | 2 / **0** / 2 | 2 / 2 / 0 |
| T6 solved / correct / false | 2 / **1** / 1 | 3 / 2 / 1 |
| t3-sparse-8 solved / correct / false | 1 / **1** / 0 | 3 / 0 / 3 |

Every solve is listed below (rotation error is against the manifest relative rotation; baseline error is gauge-aligned over the four-fold yaw gauge).

| set | pair | class / reference | correct | rot err | base err | support (L/R) | edges |
|---|---|---|---|---|---|---|---|
| T2 | 13.22.14 → 13.24.24 | 2 / 3 | no | 90.00° | 38.42° | 192 (367/192) | 5945 |
| T2 | 13.24.24 → 13.24.33 | 0 / 2 | no | 180.00° | 45.61° | 200 (255/200) | 4340 |
| T6 | 18.01.14 → 18.01.22 | 3 / 3 | **yes** | 0.14° | 1.34° | 258 (357/258) | 8192 |
| T6 | 18.01.18 → 18.01.22 | 3 / 2 | no | 89.95° | 33.11° | 258 (332/258) | 8192 |
| t3-sparse-8 | 20.05.42 → 20.06.52 | 1 / 1 | **yes** | 0.05° | 33.53° | 104 (172/104) | 1490 |

Only one pair (T6 `18.01.14→18.01.22`) is solved correctly by both matchers. DS-18 answers the T2 pairs falsely that DS-14 answered correctly, and answers `20.05.42→20.06.52` correctly where DS-14 was wrong — but that t3 solve still has 33.5° baseline error, so "correct class" is not a usable pose.

Preregistered criterion (`promotion_criterion.declared`), all five predicates required:

| predicate | result |
|---|---|
| class correct ≥ 6/37 | **false** (2) |
| precision among solves ≥ 0.75 | **false** (0.40) |
| T2+T6 correct ≥ 5/9 | **false** (1) |
| T2+T6 false ≤ 1 | **false** (3) |
| t3-sparse-8 zero false solves | **true** (0) |

Verdict: **`retain_research_only`**. No threshold, cell or gate was adjusted after this evaluation, and no matcher code inspects manifests or the promotion block.

## 5. Why the directed path loses (mechanism, not tuning)

- **Density without reciprocity admits chance edges.** Removing the reciprocity requirement makes the graph ~3× denser. On these near-forward-orbit pairs the epipolar residual is near-degenerate (the winner's all-edge median residual is 8.34°, min 2.36°), so the per-left pass finds a qualified best edge for most left endpoints even when the pair is geometrically unrelated. Support inflates to a median of **139** (min 34, max 336) versus DS-14's median **88** (min 23, max 248), and the winning hypothesis' cheirality has median **0.289** (min 0.00, max 0.99).
- **The gates then refuse the inflated winner.** The one-to-one set is dominated by chance edges that do not triangulate in front of both cameras, so `cheirality_failed` accounts for **30 of 32** refusals. The matcher does not fall back to a lower-support mode that would pass, because it — like DS-14 — gates the ranked winner only. This is the intended "abstain rather than relax" behaviour, but it converts the abundance of false support into refusals rather than into solves.
- **Synthetic fragility corroborates it.** In the repeated-descriptor control, a 12-point repetition mass solves with a clean support of 20, but declared masses of 16/20 fail `selection_unstable` and 40 fails `cheirality_failed`; the outcome is governed by how many chance edges the per-left pass admits, not by the true-edge count.
- **Four of 37 pairs have zero DS-12 mutual survivors**, confirming (with DS-12/DS-16B) that a reciprocal/one-to-one filter removes all support on part of this fixture family; the directed graph does reach those pairs but does not convert the extra edges into adjudicable geometry.

## 6. Limitations

- 37 pairs from three fixtures at the frozen render scale; `t3-sparse-8` contributes 28 pairs but only one solve, so the T2/T6 reading (4 solves, 1 correct, 3 false) is thin.
- The diagnostic measures *class* correctness and pose error; "class correct" with a 33.5° baseline error is not a usable pose, as the t3 solve shows.
- The mode-pruning and ranking choices in §2.4 are design decisions made before the reference read but after synthetic failures; they are stated here rather than hidden. A literal pre-collision left-first ranking was measured to select mass-colliding degenerate modes.
- No random sampling anywhere: RANSAC is seeded from the DS-14 defaults and the graph ordering is index-deterministic.

## 7. Artifacts (one run, exit 0)

| path | bytes | sha256 |
|---|---|---|
| `exports/diagnostics/ds18/directed_estimates.json` | 1328575 | `bd5a34214bf1992283bda817eba41a781d2939945c429bb8cc5e9c77b611cc53` |
| `exports/diagnostics/ds18/directed_estimates.csv` | 8843 | `6088263b1175ad55068d1202f83bbc5e8e9ca477948bb38af5361bebffb4085e` |
| `exports/diagnostics/ds18/directed_evaluation.json` | 45432 | `d8f3d4b0b4e0b0afa405ac6c879a52c4c943db242eef5e2c524b911d2e18e216` |
| `exports/diagnostics/ds18/directed_evaluation.csv` | 9185 | `5b26f661b9449e55b990f71b1a46fb6164625146f7f8a8d9b928e123cec6af47` |
| `exports/diagnostics/ds18/directed_run.log` | 11942 | `506f0578ec98dbf8006075fb701bd8e655040bf55f7c48459e5fc63b95a7a59a` |

Canonical pairs digest `cec240d87bf6bcc7014fd287a2ba3d7449f4a7064188ad30b1d44498d50c0988`. Calibration artifacts: `M3-calibration-templates-2.json` `435c19bb…`, `M3-calibration-templates-6.json` `a24751ab…`, `M3-calibration-templates-3-t3-sparse-8.json` `bbbd3a0d…`. DS-15 comparator artifact `exports/diagnostics/ds15/joint_evaluation.json` `5b1f0531…` (8/4/4 confirmed from the artifact, not hard-coded). Runtime: **35.29 s** total (mean 0.95 s/pair, max 1.63 s), T2 3.16 s, T6 8.33 s, t3-sparse-8 23.80 s; process exit 0.

Recipe: `.venv\Scripts\python.exe scripts/diagnose_directed_pairs.py`. Focused tests: `67 passed in 11.97 s`. DS-14 remains the accepted research baseline; DS-18 is retained as negative evidence that removing reciprocity without a repetition-aware one-to-one adjudicator increases false support rather than correct solves, and must not enter production.
