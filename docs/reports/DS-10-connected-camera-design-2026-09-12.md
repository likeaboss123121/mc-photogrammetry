# DS-10 — Production image-only connected-camera solver: read-only design audit

DeepSeek CLI DS-10, 2026-09-12. Owns this file and `DS-9-single-view-camera-seam-2026-09-12.md` only.
Read-only: no source/test/benchmark edit, no test run, no schema/desktop/Luna path, no commit.
Revision: HEAD `c6dba2c` + dirty tree in `git status` (DS-3/4F/5/7 accepted edits, DS-9 seam).
Method: targeted `read`/`rg` over `calibration.py`, `camera.py`, `sfm.py`, `registration.py`,
`lattice.py`, `surfaces.py`, `contracts.py`, `scripts/relative_pose.py`, `scripts/register_planes.py`,
`scripts/calibrate_frames.py` and the tests named below. **No algorithm was executed: every number
is a code constant, a recorded prior measurement, or a proposed tolerance — labelled as such.**
## 1. State of the art (exact anchors)

| Link | Status | Anchor |
| --- | --- | --- |
| Per-view focal/yaw/pitch/roll + principal point | implemented, image-only | `calibration.py:580-690`, `:497-577` |
| Session shared focal (median per image geometry) | implemented | `calibration.py:130-213`; `test_calibration.py:258-319` |
| `CameraEstimate`/`CameraSolution`, unresolved notes | implemented | `calibration.py:216-269`; `contracts.py:87-94` |
| Single-view production seam (DS-9, accepted 28/28) | implemented | `pipeline/image_only.py`; `cli.py` `reconstruct` |
| Translation + yaw quadrant explicitly unresolved | by design | `calibration.py:25-29,265-267` |
| Pairwise two-view geometry (PyCOLMAP) | experiment, **failed 0/41 within 5°** | `scripts/relative_pose.py`; `M3-relative-pose-2026-09-08.md` |
| Union-find over resolved pairs | report field only | `relative_pose.py:246-277` |
| Translation from *stated* shared-plane links | implemented, advanced-only | `registration.py:90-157`; `test_registration.py` |
| Lattice period/phase, block/texel hypotheses | implemented, per plane per view | `lattice.py:720-765` |
| Scale from a *stated* length; plane proposals | implemented, advanced-only / per view | `lattice.py:196-246`; `surfaces.py:89-219` |
| Supplied-pose multiview triangulation | implemented, developer-only | `sfm.py:23-142`; `test_sfm.py` (one guard test) |
| Connected image-only camera solve | **missing** | no `CameraSolution` consumer in `src/` beyond the CLI seam |

**Verdict.** The tree recovers each view's orientation and focal length from pixels and turns an
asserted cross-view fact into a translation on one axis. It has never recovered a relative
translation *direction* from image evidence in a way that survived Minecraft repetition: every
resolved PyCOLMAP pair was wrong, with error equal to the true separation (identity rotation) — the
signature of repeated-texture self-matching (`M3-relative-pose:32-54`). The solver is therefore not
"wire up `sfm.py`": it is a missing estimator plus a discrete-ambiguity resolution.
## 2. Mathematics: three distinct unknowns, three resolutions

**Frames.** World lattice frame is right-handed, `+Y` up (gravity). A view's pose is `C_i ∈ R³` (eye,
blocks) and `R_i = R_z(roll_i)·R_x(pitch_i)·R_y(yaw_i)`, the world→camera rotation of
`camera.py:126-140`; pixels use `project_points` (`camera.py:255-280`) with `K_i = diag(f_i, −f_i)`
about `(c_x, c_y)` in the image-up frame of `camera_space_points` (`:147-160`), and
`camera_projection_matrix` (`:163-177`) is exactly `K_i [R_i | −R_i C_i]`. COLMAP's image-down frame
is the fixed conjugation `D = diag(1,−1,1)`, `D R D` (`relative_pose.py:125-137`).

**(A) Bearing-only gauge.** Vanishing points `v(f, yaw, pitch)` (`calibration.py:429-453`) are
invariant under `C_i ↦ C_i + λ·direction` and under `f ↦ αf, C ↦ αC`; one view fixes
`(f_i, yaw_i mod 90°, pitch_i, roll_i)` and nothing metric.

**(B) Discrete quarter-turn.** The estimator scores each segment against the *set* `{d_x,d_y,d_z}`
of world axes (`calibration.py:486-494`, `min(axis=1)`), invariant under every orthogonal integer
matrix permuting `{±e_x,±e_y,±e_z}`: the 24-element rotation group of the cube. The
`yaw ∈ [0°,90°)` search (`:632`) is a fundamental domain for `{id, R_y(90°), R_y(180°), R_y(270°)}`,
so `(f, yaw mod 90, pitch)` determines a **coset of 4** gravity-aligned orientations: with `R(φ,θ)`
the zero-roll gravity-aligned camera, `score(R(φ,θ)) = score(R(φ+90k°,θ))`, `k ∈ Z/4`, because a
segment's residual is the minimum over the three vanishing points and those are permuted. **Yaw is
recovered modulo 90°, and exactly four physically distinct cameras about the vertical reproduce one
image** — a real unknown, not a labelling convention: they differ in where the lattice axes project,
and only one matches the build.

**(C) Global origin and absolute yaw are gauge.** If the solver recovers every `R_i`, `C_i` in one
lattice frame, then `R_i ↦ G R_i`, `C_i ↦ C_i + b` for any `G = R_y(ψ)`, `b ∈ R³` leaves every image
and inter-camera angle unchanged: the component is fixed only *relative to an anchor*. Forcing
`R_anchor` and `C_anchor = 0` (DS-9's `LOCAL_FRAME_ORIGIN`) is a declared gauge, not a measurement.
The vertical needs no gauge — `+Y` is global — which is why `register_planes.py:9-16` asserts
quadrants for the horizontal axes only.

**(D) Metric scale is separate and image-addressable.** The reconstruction is determined up to
`C_i ↦ αC_i`, `X ↦ αX`, `α > 0`; from `m ≥ 2` views the *direction* `(C_j−C_i)/‖C_j−C_i‖` is
observable but never `α`, which needs one image-measured block/texture spacing (Section 4).
`scale_known` (`contracts.py:94`) must stay `False` until that succeeds, and connectivity must not
flip it. **Relative rotation (well-posed):** with the world frame common, `ΔR_ij = R_j R_iᵀ` is
independent of the (B)/(C) gauges and — because `R_i` comes from calibration, not matching — is
available *before* any correspondence is trusted.

**Relative translation direction, rotation-first.** Write the baseline in the left camera's frame,
`t_ij = R_i (C_j − C_i)`. For a correspondence with unit camera-frame rays `u`, `u'`
(`camera.py:283-298`), `E_ij = [t_ij]_× ΔR_ij`, and coplanarity gives `u'ᵀ E_ij u = 0`. Since
`E u = [t]_×(ΔR u) = (ΔR u) × t`, this is the **linear** constraint

    a(u,u')ᵀ t_ij = 0,    a = (ΔR u) × u'.        (1)

`t_ij` is the unit right-singular vector of the stacked `A ∈ R^{n×3}` of `a` rows; two non-parallel
rows suffice, use ≥ 8 with SVD and Huber reweighting for conditioning. This is a rank-2 homogeneous
least squares, not the 5-/8-point essential solve: knowing `ΔR` removes the 4-fold decomposition and
yields `t̂` up to sign in one closed-form SVD. The sign is fixed by cheirality — triangulate the
matches and require positive depth in both cameras (`sfm.py:145-158`). Report the per-pair residual
`ρ = |aᵀt̂|/‖a‖` (radians) and its median.

**(E) One scale per component.** For `n ≥ 3` views the baselines are dependent:
`C_i + λ_ij t̂_ij = C_j` must close around every cycle, fixing ratios `λ_ij` (law of sines per triple)
and hence all `C_i` up to one `α`. Solve positions by gauge-fixed nonlinear least squares
(`Σ‖(C_j−C_i) − λ_ij t̂_ij‖²`, chordal rotation averaging for `R_i`), report the cycle residual, and
give each component its **own** `α` (`registration.py:22-30` makes the same point for asserted
links): internally consistent, not comparable to other components, not metric.
## 3. Components, quadrant search, rejection rules

**Graph.** Vertex = frame; edge = a pair passing all gates that yields `ΔR_ij`, `t̂_ij`. Union–find
exactly as `relative_pose.py:246-277`, over *verified* edges rather than library configurations.
**Quadrant resolution by bounded enumeration (exact).** Each view has copies
`R_i^(k) = R_y(90k°) R_i`, `k ∈ {0,1,2,3}`. Enumerate assignments per component with the anchor at
`k=0`: `4^(n−1) ≤ 64` for four-frame fixtures, scored by
`Σ_edges angle(ΔR_ij, R_j R_iᵀ) + Σ_edges median ρ_ij`; accept only when every edge residual is below
tolerance. A view with no surviving edge has no evidence to resolve its quadrant and must stay
unresolved (per-frame `"yaw quadrant"` note, `calibration.py:267`). Cyclic consistency across ≥ 3
views (`ΔR_ik = ΔR_jk ΔR_ij`) makes the assignment identifiable; it is checked, not assumed.
**Degeneracy / outlier / repetition rejection** (proposed gates), per candidate edge:
1. Calibrated `inliers ≥ 15` and `inlier_ratio ≥ 0.5`.
2. Median coplanarity residual `ρ ≤ 1.0°`, with bounded `median ρ / IQR(ρ)` outlier spread.
3. **Parallax:** median triangulation angle between matched rays `≥ 3°`. Pure rotation (panoramic)
   has zero parallax and `t̂` is unobservable; keep `relative_pose.py:209-217`'s `PANORAMIC` refusal
   and add the numeric floor, since fixtures include 176°-apart non-overlapping views
   (`M3-relative-pose:82-84`).
4. **Planarity:** if triangulated inliers fit one plane, `t̂` is weakly observable;
   `relative_pose.py:84-85` lists `PLANAR` as unresolved. Detect from the singular-value spectrum of
   the centred point cloud, not a library label.
5. **Repetition audit (the decisive new test).** The world is a unit lattice, so a verified match
   must have `round(X)` (block id) reproduce the same pixel neighbourhood in *both* frames within a
   tolerance derived from `period_pixels` and `f` (`δ ≤ 0.25/period_world`, blocks not pixels).
   Self-matched repeated textures satisfy epipolar geometry by construction
   (`M3-relative-pose:46-49`) but do not produce a lattice-consistent, parallax-separated 3D point.
   Reject an edge whose repetition-consistent inlier fraction is below threshold.
6. Bootstrap with RANSAC (seed 5 pairs, resample), since a majority of matches may be look-alikes.

**Correspondence source (no new dependency).** The existing path is `cv2.SIFT_create` + `BFMatcher`
with a HUD mask (`sfm.py:48-88`), already in `.venv`. Feature matching stays a weak link, so the
package must not *depend* on a dense match set: it accepts `matches` as input and reports
`unresolved` when the edge count cannot connect the required views. Lattice-constrained matches
(vanishing-line intersections, block corners rectified by `lattice.rectify_plane`) are a later
refinement, outside this package.
## 4. Scale from image-derived block/texture spacing only

`lattice.period_hypotheses` (`:720-765`) measures a plane's period in rectified pixels and emits two
readings — one block per period (`d`) and one texel per period (`d/16`), ordered by `coarse_support`
— while `block_supports` records harmonic candidates (`2,3,4,…` periods per seam). The constraint
system:
- View `v` places a plane at distance `d_v` in component units and measures period `P_v`
  (world-units-at-unit-distance, `:750-758`); its true distance is `α d_v`. A span of `b` blocks per
  period implies `α d_v P_v = b`.
- For two views of the *same* physical plane, `α` is common and `P_a d_a = P_b d_b` must hold within
  tolerance once the lattice frame is shared; a mismatch falsifies that plane identity or period
  reading. Pure image check, no manifest.
- Candidates `b ∈ {1,2,3,4,8,16}` (texel hypothesis plus `block_supports`) give candidate `α`. Keep
  those agreeing across independent planes/views within tolerance; require **≥ 2 independent planes
  or ≥ 2 agreeing views**. Only then set `scale_known = True`, record `b`, `α`, supporting
  frames/planes, and mark `"scale resolved from lattice period"`. Otherwise keep
  `scale_known = False` with the explicit note; never convert `b` into a user length.
- `lattice.scale_from_known_length` (`:196-246`) stays advanced-only (it consumes a *stated* length)
  and must not be called from the normal path. One block = one world unit here, so "metric lattice
  scale" means fixing `α` until reconstructed cell size is 1.0, not inventing a unit; `scale_known`
  stays `False` until resolution succeeds.
## 5. Smallest bounded implementation/experiment package (one only)

**Proposed new module `pipeline/connected_camera.py`** (DeepSeek-owned after a lead start prompt):
pure analytic geometry consuming DS-9's single-view estimates. No edit to `calibration.py`, `sfm.py`,
`raycast.py`, schema, desktop, or any Luna path.
```python
@dataclass(frozen=True)
class PairMeasurement:
    left: str; right: str; inliers: int
    relative_rotation: np.ndarray           # 3x3, R_right @ R_left.T
    translation_direction: tuple[float, float, float]   # unit, R_left frame
    median_rho_deg: float; parallax_deg: float; repetition_inlier_fraction: float

def relative_rotation(left: CameraEstimate, right: CameraEstimate) -> np.ndarray
def translation_direction(left, right, matches, *, ransac_seed=0) -> tuple[np.ndarray, dict]
    # eq. (1): SVD of the stacked a-rows, cheirality sign, residuals in the dict
def resolve_quadrants(estimates, measurements) -> dict[str, int]   # 4^(n-1) enumeration
def connect(estimates, matches_by_pair, *, gates=Gates()) -> CameraSolution
    # union-find components; per-component anchor gauge C=0 and own alpha;
    # per-frame unresolved notes (position / component / scale / yaw quadrant) preserved
```
**Synthetic controls (new `tests/test_connected_camera.py`; no real solver, no fixture run).** Build a
ground-truth global frame `(C_i, yaw_i, pitch_i, f)` and project lattice edges/corners with the
shipped `camera.py` projection (pattern of `test_calibration.py:33-58`). 1. **Exact recovery.** 3
views, generous parallax, exact matches: `angle(ΔR̂,ΔR) ≤ 0.5°`, `angle(t̂,t) ≤ 1.0°`, small cycle
residual, `scale_known is False`. 2. **Quadrant gauge.** Yaw in `[0,360)`: the enumeration selects
the true copy; a two-view component resolves only relative to its anchor (global yaw free); a
triangle resolves all four copies. 3. **Sign/cheirality.** Flip the SVD sign; cheirality recovers the
correct sign. 4. **Degeneracy refusals.** No-parallax (pure rotation) pair, single-plane pair,
`< 15` inliers, mismatched image sizes → each returns `unresolved`, never a fabricated `t̂`.
5. **Repetition trap.** Self-matches at identical image positions (identity-rotation trap): the
repetition audit rejects the edge instead of returning `ΔR = I`. 6. **Scale.** Two views of one plane
with a synthetic one-block vs. sixteen-texel period: only the one-block reading survives
`P_a d_a = P_b d_b` and sets `scale_known=True`; a two-periods-per-seam harmonic must select `b=2`.
7. **Limits invariant.** A disconnected 4-view set returns ≥ 2 components with per-component `α` and
no cross-component position claim.
**Acceptance.** All controls pass; no existing test modified; `CameraSolution.unresolved` still names
position, scale and quadrant per unresolved frame; `scale_known=True` only in control 6; the module
imports neither `load_camera_manifest` nor `load_scene_transform` and reads no manifest. Focused run:
new module + `test_calibration.py` + `test_camera.py` + `test_sfm.py`. Out of scope: schema 2, desktop
multi-frame, `raycast`/`sfm` integration, real-fixture accuracy, new dependencies, matching tuning.
## 6. Conflicts, unknowns, non-claims

**Luna adjacency.** `shape_fit.py` and M5 diagnostics are Luna-owned; `raycast.py`, `residual.py` and
`face_sampling.py` carry accepted DS-3/4F/5 authority. The package touches none. `camera.py` and
`contracts.py` are shared: only *read* here; new dataclasses live in the new module, and any
contract/schema change needs a comms proposal first. **Recorded negative evidence stands:**
`M3-relative-pose-2026-09-08.md` remains the result for feature matching on these fixtures; this
package adds the rotation-first estimator, the discrete gauge solution, and the repetition audit. A
real fixture yielding no component is a reportable negative, not a reason to retune matching.
**Not claimed:** that connected cameras improve any production metric; that `sfm.py`'s pipeline
becomes usable; that `α` is recoverable on these fixtures; or that the M3 gate (orientation ≤ 0.5°,
focal ≤ 2%, seam ≤ 2 px) is met by new code. Accuracy, runtime and fixture behaviour are unmeasured.

## 7. Lead corrections (2026-09-12T10:47:05-05:00) and their consequences

The lead retained this audit but rejected the Section 5 package as written. Four errors are corrected
here; DS-11 implements only the analytic pair solver with the corrected conventions.

**C1 — the baseline is not a left-camera vector.** Section 2 wrote `t_ij = R_i (C_j − C_i)`. That is
the world baseline rotated into the *left* camera, and it is the wrong right-hand side for the
canonical pair. With `x_l = R_l (X − C_l)`, `x_r = R_r (X − C_r)`, and `P_l = [I | 0]`,
`P_r = [R | t]`:

    x_r = R_r R_lᵀ x_l + R_r (C_l − C_r),  so  R = R_r R_lᵀ,  t = R_r (C_l − C_r) = −R_r (C_r − C_l).

`t` is the baseline expressed in the **right** camera frame and points from the right eye towards the
left eye. The world direction from the left eye to the right eye — the only frame-independent
quantity a caller can use — is `b = C_r − C_l = −R_rᵀ t`. Section 2's formula points the opposite way
and is expressed in the wrong frame.

**C2 — the yaw class is candidate-dependent, and the copy is a right multiplication.** Section 3
wrote `R_i^(k) = R_y(90k°) R_i` and proposed scoring edges by `angle(ΔR_ij, R_j R_iᵀ)` before the
quadrants are chosen. Both are wrong. First, `R = R_r R_lᵀ` depends on the right view's chosen class,
so a single derived `ΔR_ij` cannot be scored against a fixed rotation — the class and `t` must be
solved jointly, and the anchor's class is a gauge, not a measurement. Second, in this repository
`camera_rotation_matrix` returns `R(yaw) = Rz(roll) D Rx(−pitch) Ry(yaw)` with `D = diag(−1,1,1)`
(verified against `camera.py:126-140`), so the four gravity-aligned copies are
`R(yaw + 90k) = R(yaw) Ry(90k)` — a **right** multiplication. Left multiplication `Ry(90k) R` equals
`R(yaw − 90k)`, a different and incorrect candidate set. The pair solver enumerates
`R_right(k) = R_right Ry(90k)`, `k ∈ {0,1,2,3}`, with the left view anchored at `k = 0`.

**C3 — the repetition audit cannot run before scale and frame recovery.** Section 3 gate 5 required
`round(X)` block IDs to reproduce a pixel neighbourhood. Rounded block IDs are a function of the
lattice origin and period, which are exactly what is unresolved at this stage; the audit is
circular and must not gate a pair. The pair solver therefore uses only rank/conditioning, parallax,
residual and cheirality evidence. It reads no lattice, manifest, or asserted metadata.

**C4 — single-plane structure is not automatically degenerate.** Section 3 gate 4 rejected any
planar point cloud. With `R` known from calibration, a plane is degenerate only when it contains the
baseline: then every `a` row is parallel to the plane normal and `A` collapses to rank 1, so `t` is
unobservable. A plane that does not contain the baseline leaves `A` at rank 2 and `t` observable. The
correct test is the singular-value conditioning of the stacked `A` plus the parallax floor, not a
planarity label.

**Consequence.** Section 5's module (graph, components, quadrant scoring, repetition gate, scale) is
withdrawn as written. DS-11 implements the bounded analytic pair solver alone with C1-C4 applied:
`solve_pair(left, right, matches)` returns `solved=True` with the right yaw class, `R = R_r R_lᵀ` and
the unit world baseline `b = C_r − C_l`, or `solved=False` with a typed reason and no vector.
