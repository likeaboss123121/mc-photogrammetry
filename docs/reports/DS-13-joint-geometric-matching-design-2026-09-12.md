# DS-13 — Joint geometric matching: design audit and one bounded package

DeepSeek CLI DS-13, 2026-09-12. **Read-only audit.** Owned write: this file only. No source, test,
script, artifact, mask, threshold or fixture was modified; no test, benchmark or fixture manifest
was opened; nothing was staged, committed, installed or chained. Revision: HEAD `c6dba2c` plus the
accepted DS-3/4F/5/7/9/10/11/12 working-tree edits (tree dirty, as before). Evidence read: DS-11 `pipeline/connected_camera.py`; DS-12 `pipeline/correspondence.py`, `scripts/diagnose_connected_pairs.py`,
`exports/diagnostics/ds12/{estimates,evaluation}.{json,csv}` (`estimates.json` sha256 `290204c6…`); `docs/reports/M3-relative-pose-2026-09-08.md`; DS-10 §7 corrections C1–C4;
`lattice.rectify_plane` / `measure_plane` / `period_hypotheses`; OpenCV 5.0.0 primitive inventory (§4); and the bounded probe in §1.

## 1. Bounded manifest-free probe (candidate counts only)

Nine frozen frames, `detect_features(..., usable_mask=None, CorrespondenceSettings())` (DS-12 production
path), then chunked NumPy k-NN (k=6) in both directions; counts printed to stdout only, nothing written.
The probe reproduces DS-12's frozen `ratio_0.75` and `mutual_0.75` columns **exactly on all six rows**,
so it measures the same extraction path. No `cameras.json`, reference litematic or lattice was opened.

| pair (fixture, suffixes) | raw L/R | ratio.75 L/R | mutual.75 | recip k=2 | k=3 | k=5 | k=5 d₁ median | greedy 1-to-1 | bins L/R (16) |
|---|---|---|---|---|---|---|---|---|---|
| T2 13.22.14→13.24.24 | 1286/919 | 60/83 | 18 | 506 | 841 | 1538 | 196.0 | 475 | 9/8 |
| T2 13.24.24→13.24.33 | 919/1568 | 20/25 | 9 | 558 | 955 | 1788 | 227.8 | 554 | 8/8 |
| T6 18.01.09→18.01.14 | 1799/2560 | 21/32 | 14 | 1114 | 1863 | 3607 | 294.6 | 1078 | 8/10 |
| T6 18.01.14→18.01.18 | 2560/1810 | 110/93 | 43 | 1393 | 2325 | 4247 | 228.8 | 1247 | 10/11 |
| T3 20.03.32→20.03.55 | 973/320 | 66/43 | 21 | 350 | 549 | 998 | 230.2 | 273 | 10/12 |
| T3 20.03.55→20.04.18 | 320/668 | 26/18 | 8 | 264 | 447 | 833 | 260.3 | 245 | 12/9 |

Boundary-risk probe (same frames, `distanceTransform(usable)`): keypoints within 4 px of the excluded domain 19–47 (1.4–8.8 %); within 8 px 38–72 (2.9–11.5 %); within 16 px up to 111;
median boundary distance 90–214 px. All detected keypoints are inside `usable` by construction (min distance 0.95 px).

## 2. What the funnel actually loses

DS-12's refusal is not caused by the ratio threshold being too tight. `ratio_0.90` already yields
307/229 directed survivors on T2a, and `mutual.75` is 18: the binding constraint is **1-to-1 nearest
reciprocity**, not `d₁ < 0.75·d₂`. Widening reciprocity from k=1 to k=5 *among edges that already pass
both 0.75 ratios* recovers 43/9/16/67/39/11 edges (vs mutual 18/9/14/43/21/8) — a precise gain,
but small; the remaining thousands of k=5 edges are exactly the edges the descriptor calls ambiguous
(d₁ ≈ d₂), i.e. the repetition mass the historical trap is made of. Two consequences drive the design:
(a) a relaxation that only loosens the ratio adds ambiguity without adding support (alternative A1);
(b) the candidate graph is cheap to build but must be adjudicated by **geometry**, never by descriptor rank, and the adjudication must not manufacture support by discarding rows.
Spatial spread flips too: mutual.75 matches cover 0.06–0.31 of the 4×4 bins in DS-12, while the greedy
one-to-one k=5 sets cover 0.50–0.625 (`84e`-style spread gate in DS-12 was never met by the mutual set).

## 3. Three alternatives

**A1 — unilateral relaxed ratio (reject).** Keep one direction, raise `r` (0.85–0.95), drop or weaken
mutual reciprocity. Math: directed sets grow ~5–11× (`ratio_0.95` 616/418 on T2a), the reciprocal
nearest intersection does not; many-to-one edges have no one-to-one assignment, so a greedy dedupe
must be added anyway, at which point A1 *is* A2 with k=1 and a looser `r`. Complexity unchanged
(O(N_l·N_r)). Controls cannot distinguish repetition from correspondence: the extra edges are the
near-tie look-alikes. Verdict: strictly dominated by A2; adopting it alone would re-create the M3
failure mode with more confident-looking rows.

**A2 — reciprocal top-k candidate graph + orientation-conditioned robust joint fit (select).**
Build the symmetric k-NN graph; fit the DS-11 baseline hypothesis *per yaw class* on candidate edges
(`a = (R_k u_l) × u_r` is already orientation-conditioned through the calibrated rotation hypothesis
`R_k = R_r Ry(90k) R_lᵀ`); select a one-to-one support; choose the class on an all-candidate score and
require a margin. Math and algorithm in §4. Complexity: graph O(N_l·N_r) distance work chunked,
O(k·(N_l+N_r)) memory, E ≤ 8192; fit O(4·B·E + 4·B·s³) with B=512, s=5; greedy O(E log E). No solver
change: it reuses DS-11's `_robust_baseline` math, `PairGates` and `solve_pair` verbatim.

**A3 — rectified-plane / lattice matching (defer).** Use `lattice.rectify_plane` +
`period_hypotheses` per view to fronto-rectify a dominant plane, then match in the rectified domain and
index edges by lattice phase (`round` to a period). Math: per-view plane homography `H = K R_planeᵀ`;
matching becomes translation + phase; block index `b ∈ {1,2,3,4,8,16}` remains ambiguous until scale
and lattice frame are resolved (DS-10 C3), so it cannot gate a pair here. Rectification *aligns*
repeated tiles, converting a false block identity into a lattice-phase identity — it amplifies the
ambiguity instead of removing it, and a plane-only model is degenerate whenever the plane contains the
baseline (DS-10 C4; keep conditioning/parallax, not a planarity label). Cost: plane proposal O(WH) + resample O(WH) per plane/view + period estimation O(S log S) + rectified matching, over M4/M5/Luna-owned surfaces (`supported.py`, `lattice.py`, shape fitting). Verdict: not the smallest package; revisit only
after a plane layer exists, as a plane-conditioned *prefilter* on additional edges, never as the pair gate.

**OpenCV inventory (5.0.0, present but deliberately unused):** `findEssentialMat`/`recoverPose`,
`findHomography`, `estimateAffine2D`, `USAC_*`, `stereoRectifyUncalibrated`, `BFMatcher.knnMatch`,
`erode`, `distanceTransform`. `recoverPose` is the exact historical trap; `USAC_*` seeds a global RNG;
both violate the determinism contract. `distanceTransform` is used only for the boundary rule (§5), and
the deterministic k-NN is the existing chunked NumPy code generalized from 2 to k neighbours.

## 4. Selected package A2 — objective, sampling, complexity, outputs

**Objective (exact).** With `E` the candidate edge set, `u_l/u_r` normalized rays (image-up convention,
each view's own K, `connected_camera._camera_rays`), and per class `k∈{0..3}`:
`R_k = R_r Ry(90k) R_lᵀ`, `a_e = (R_k u_l) × u_r`, `ρ_e(k) = degrees(asin(|a_e·t_k| / ‖a_e‖))`,
`t_k` = unit null vector of the stacked `A_k`. Class score is the median over **all** `E` (DS-11's
anti-gaming rule — a class may not shrink its median by discarding rows), tie-broken by larger
one-to-one support then lower `k`. Winner `k*` is accepted only if: all-`E` median ≤ 1.0°;
runner-up margin ≥ 0.25°; one-to-one support `|S_k*| ≥ max(8, ceil(0.02·|E|))` (DS-12's pre-registered
2 % repetition guard reused as a candidate-support floor); spatial spread of `S_k*` ≥ 4 of 16 bins on
**both** frames (DS-12's pre-registered 0.2 coverage); median parallax ≥ 3.0°; cheirality ≥ 0.75;
conditioning `s1/s0 ≥ 1e-3`; and `solve_pair(S_k*)` returns solved with the same class. Otherwise a
typed refusal, never a vector.

**Support selection, sampling, refit.** `S_k` = greedy one-to-one set: sort `{e ∈ E : ρ_e(k) ≤ 1.0°}`
by `(ρ_e, left index, right index)`, accept an edge only if its left and right keypoints are unused —
this is the only place the assignment is enforced. RANSAC is deterministic and seeded
(`np.random.default_rng(ransac_seed)`, reset identically for every class so classes are compared under
identical sampling): candidate 0 is the all-`E` SVD (DS-11's rule), then `B=512` minimal 5-edge samples;
score by inlier count within the 1.0° band with DS-11's tie-break; refit ≤10 SVD iterations to a fixed
point; select the sign by cheirality. No refit after `S_k` is chosen, so `S_k` is not re-optimised to
inflate itself. A deterministic PROSAC-style distance-ordered schedule is the declared fallback if
uniform sampling proves support-poor; it is **not** part of this package.

**Complexity cap (hard).** `k = 5`; `|E| ≤ max_candidate_edges = 8192` (kept by `(d₁, i, j)` rank,
truncation reported, never silent); `B = 512`; refit ≤10; sample size 5. Work per pair:
O(4·B·|E|) ≈ 1.7·10⁷ vectorized residual evaluations worst case, plus the chunked k-NN pass and
O(|E| log |E|) sort — sub-second NumPy, bounded independently of the 4000-feature detector cap.

**Typed outputs / API.** New `pipeline/joint_matching.py`: `match_and_solve_pair(left_id, left_rgb, right_id, right_rgb, *, usable_masks=None,
settings=None, matcher=None, gates=None) -> JointPairResult`, where `matcher: JointMatchSettings`
carries `neighbors=5`, `max_candidate_edges=8192`, `mask_boundary_pad_px=8.0`, `graph_distance_ceiling=None`,
`ransac_seed=0`; `JointPairResult` carries `status` (`"solved"`/`"refused"`), `reason`, `pose: PairPose | None`
(DS-11's own type, produced by `solve_pair(S_k*)`), `matches` (the selected `N×4`, one-to-one),
`graph` counters (`raw_keypoints`, `candidate_edges`, `truncated`, `boundary_dropped`,
`ratio_survivors`, `mutual_survivors`) and per-class diagnostics. Matcher reasons are module-local
(never appended to DS-11's `REFUSAL_REASONS`): `candidate_graph_empty`, `insufficient_one_to_one_support`,
`low_spatial_spread`, `selection_unstable`. The matcher module reads pixels and the mask only — no
camera, scene, lattice, coordinate or manifest is reachable from it; the diagnostic script holds the
image-derived cameras, exactly as DS-12's two-phase split does. Additive DS-12 edits only:
`FrameFeatures.sizes` (default `None`, preserves all constructors) and
`CorrespondenceSettings.mask_boundary_pad_px: float = 0.0` (0 = current frozen behaviour).

## 5. Mask-boundary descriptors vs pixel authority

DS-12 limit 3 is real: excluded RGB is zeroed, so the mask edge is a synthetic high-contrast contour and
SIFT places features on it (probe: 19–47 keypoints within 4 px per frame). Rule: with `U` the
authoritative usable mask from `usable_mask_for` (semantics unchanged), the detection domain is
`D = {p : dist(p, ~U) > pad + 0.5·kp.size}`, i.e. reject a keypoint whose descriptor footprint touches
the excluded domain; `pad = 8.0 px` declared here (≤11.5 % of detections on these frames, no interior
keypoint lost), with the exact footprint factor pinned by a synthetic control rather than by fixtures.
Authority is preserved by construction: `D ⊆ U` (the filter only shrinks), the mask remains the single
authority for detection, matching and consumption, excluded-RGB zeroing stays as DS-12 wrote it, and
supplied masks keep their authoritative/missing-frame/wrong-shape semantics verbatim. Erosion is a
deterministic function of `U` and the keypoint list; it introduces no second authority and no manifest read.

## 6. Why this cannot repeat the M3 identity-rotation trap

M3 measured `recoverPose` returning ΔR ≈ I with 100–300 inliers on views 66–176° apart, error ≈
separation, with no library flag distinguishing it (M3 §2). A2 is structurally immune in five ways.
(i) **Identity is not in the hypothesis set**: candidates are the four discrete `R_k` built from
independently *image-derived* per-view yaws; ΔR ≈ I is representable only if those calibrated yaws
differ by a multiple of 90°, a property of the calibration, not of the matches. A repetition-driven
match set cannot rotate the hypothesis; it can only fail the gates. (ii) The historical support was
look-alikes at ~identical pixel positions: those give near-zero parallax and are refused by the 3°
parallax floor. (iii) Class comparison uses the all-candidate median, so a wrong class cannot buy a
lower median by discarding rows, and `S_k` is chosen only *after* the class is fixed. (iv) `solve_pair`
must reproduce the joint class on `S_k*` (`selection_unstable` otherwise), so the reported pose is
DS-11's own gated verdict, not a matcher assertion. (v) No lattice IDs, rounded block indices or
asserted metadata are used (C3), and the boundary rule removes the synthetic mask-contour features that
M3's overlay watermark also produced.

## 7. Synthetic controls (`tests/test_joint_matching.py`, new)

1. Top-k graph on a supplied synthetic descriptor table: exact reciprocal edge set, cap/truncation
   reporting, bit-identical across two runs.
2. Tiled 40 px wall, two views ~90° apart, exact intrinsics: the candidate graph is non-empty but the
   result refuses (`insufficient_one_to_one_support` / `low_spatial_spread`) — never a solved identity.
3. Same wall plus ~30 unique 3D landmarks: solves, correct right-yaw class, `matches` are one-to-one
   (no repeated left/right endpoint), two runs identical.
4. Look-alike-at-same-pixel construction (the M3 trap): graph large, result refuses, and no returned
   ΔR is within 5° of I while the calibrated yaws differ.
5. Boundary control: a keypoint whose descriptor footprint touches `~U` is absent, `boundary_dropped > 0`,
   and mutating RGB outside `U` to 0/255 leaves selected matches and every counter identical.
6. Class-rule control: a scene where a wrong class could shrink its own inlier median still loses on the
   all-candidate median; `<8`-edge and non-reciprocal-graph cases return typed refusals, not classes.

## 8. Bounded implementation/experiment package and acceptance

One package: new `src/mc_photogrammetry/pipeline/joint_matching.py`, new `tests/test_joint_matching.py`,
new `scripts/diagnose_joint_pairs.py` (DS-12's phase split replicated, writing
`exports/diagnostics/ds13/{joint_estimates,joint_evaluation}.{json,csv}`), additive DS-12 fields of §4,
and a short DS-13 results report. No change to `connected_camera.py`, DS-12 defaults, masks, schema, CLI,
desktop, benchmark, raycast/residual or any Luna path; no new dependency; no manifest importable from
the matcher. Experiment: the six pairs already frozen by DS-12's stated rule (no pair selection by
outcome), phase 1 blind with a digest frozen before phase 2 opens manifests. **Acceptance is honesty,
not solve count:** all controls pass with no existing test modified; artifacts manifest-free with
recorded digests; two phase-1 runs bit-identical; every refusal carries a typed reason; calibrated
yaws and the all-candidate medians are reported per class; and any solved pair whose ΔR is within 5° of
I while the two image-derived yaw estimates are not is reported as a failure of the joint stage. A
solved-on-real-data claim requires the separate phase-2 comparison; no threshold is changed after it.
**Non-claims:** not implemented, not run, no accuracy/runtime measured, no component/position/scale/
`scale_known` claim, no M3 gate promotion, and no assertion that A2 solves even one frozen pair.

## 9. Lead correction (pre-implementation, 2026-09-12T11:13:50-05:00)

Astra accepted the DS-13 direction (A2, §3-§5, the mask-boundary rule and the §7 controls) and
corrected the **objective** before any implementation. This section supersedes the affected
sentences of §4 only; nothing else in this document is withdrawn. DS-14 implements the corrected
objective.

1. **The full candidate-graph median gate is invalid and is removed.** §4's "class score is the
   median over all `E`" and "all-`E` median ≤ 1.0°" are struck. The top-k graph *intentionally*
   contains mostly false edges (DS-12 §2), so an all-edge residual median is dominated by the
   ambiguity mass the graph was built to expose; gating on it would refuse every real pair for a
   reason that carries no evidence. Class selection must be driven by **selected one-to-one
   support**, never by an all-edge statistic. All-edge residual medians may still be *reported* as
   diagnostics, but they must not gate.
2. **DS-12's 2 % denominator is `min(raw keypoints)`, not `|E|`.** §4's support floor
   `max(8, ceil(0.02·|E|))` is wrong: DS-12's pre-registered repetition guard divides by the
   smaller frame's **raw keypoint** count. The corrected floor is
   `support ≥ max(8, ceil(0.02 · min(raw_keypoints_left, raw_keypoints_right)))`. Reusing `|E|`
   would silently tighten the guard as `k` grows, which is not what was declared.
3. **RANSAC sample size 5 is too large for low-support recovery.** Two independent epipolar rows
   already determine `t` up to sign, and with a mostly-false edge set a 5-edge sample almost never
   lands all-inlier. The corrected sampler draws **2 endpoint-distinct edges** (distinct left and
   distinct right indices), **rejects rank-deficient samples** (the two rows must be independent:
   `s1/s0 ≥ min_singular_ratio`), solves `t` as the sample's unit null vector, and then scores
   **every** candidate edge by the ≤1.0° residual. A 3-edge sample is an acceptable stability
   variant; 5 is not required.
4. **Hypothesis ranking and the ambiguity margin are explicit.** Rank hypotheses by **support count
   first**, then spatial bins, then median **selected** residual, then descriptor cost (then class /
   mode index). Accept only a winner that clears the support floor, the spatial-spread floor, the
   parallax/cheirality/conditioning gates, and a **declared margin over the best distinct competing
   class or baseline**. Otherwise abstain with a typed reason. §4's "refit is not re-optimised after
   `S_k` is chosen" is retained: the refit happens on the selected support to a fixed point *before*
   the winner is fixed, and the public DS-11 `solve_pair` is the compatibility check on the final
   selected set.
5. **Unchanged from §4-§7:** reciprocal top-k graph (`k=5`, `|E| ≤ 8192`, truncation reported),
   deterministic seeded sampling, greedy one-to-one selection, the descriptor-footprint boundary
   rule (§5, pad declared by a synthetic control), image-only extraction with the two-phase frozen
   experiment (§8), and every synthetic control in §7. DS-14's owned paths and stop condition are
   set by the assignment, not by this correction.
