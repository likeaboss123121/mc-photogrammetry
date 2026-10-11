# DS-29 — Plane/lattice camera-bootstrap package (first bounded subset)

**Agent:** DeepSeek CLI DS-29 · **Date:** 2026-09-12 · **Mode:** implementation + focused tests
**Directive:** DS-28 (recovered at `%TEMP%\ds28-assistant-messages.txt` lines 78-357)
plus the DS-28 addendum (A1-A3).
**Status:** first bounded package delivered; component/reading contracts and synthetic
controls pass; the full frozen T2/T6 diagnostic (`scripts/diagnose_plane_lattice_pose.py`)
is deliberately **not** implemented in this package.

Owned paths written (nothing else):

| path | role |
|---|---|
| `src/mc_photogrammetry/reconstruction/lattice_planes.py` | readings, distance alternatives, signatures, homography |
| `src/mc_photogrammetry/reconstruction/plane_graph.py` | identity matching, quadrant/translation solve, class-constrained best-of |
| `tests/test_lattice_planes.py` | 11 image-path and contract controls |
| `tests/test_plane_graph.py` | 12 graph/solver/repetition/class controls |
| `tests/test_plane_lattice_entry.py` | 8 source/import/poison/arity audits |
| `src/mc_photogrammetry/reconstruction/__init__.py` | minimal additive exports |
| `docs/reports/DS-29-plane-lattice-bootstrap-2026-09-12.md` | this report |
| `AGENT_COMMS.md` | byte-safe append |

No existing module, threshold, schema, CLI, desktop, raycast/residual/shape_fit,
benchmark, dataset, DS-25 artifact or git state was modified.

---

## 1. Implemented subset (precise)

### 1.1 `lattice_planes.py`

Frozen dataclasses: `DistanceAlternative`, `PlaneRectification` (added context, §2),
`PlaneReading`, `PlaneSignature`, `ReadingAbstention`.

Public functions:

- `rectification_homography(axis, axis_matrix, focal, principal_point, side, distance)`
  — the closed-form 3x3 map from block coordinates on a plane to pixels;
  mirrors `lattice.rectify_plane`'s grid math (`lattice.py:383-399`). No
  `cv2.findHomography`.
- `frame_plane_readings(image, usable, calibration_result, *, frame="", max_readings=6,
  min_structural_share=0.15, max_rectified_side=2048, diagnostics=None)`.
- `plane_signature(image, reading, blocks_per_period, *, usable=None, max_cells=256)`.
- `warp_signature(signature, homography, target_shape)`.
- `reading_order_key(reading)` — content-only per-frame ordering used by both modules.

Algorithm, in the directive's order: masked greyscale → one `detect_segments` pass →
one `structural_weights` pass → `surfaces.planar_regions` **and**
`boundary_candidates.find_chains`/`propose_boundary_candidates` merged by
`(axis, overlap >= 0.5)` keeping the higher `confidence x strength` → `lattice.plane_window`
(evidence decides the side) → a **bounded** three-round measure mirroring
`lattice.measure_plane` but clamping every request to `MAX_RECTIFIED_SIDE` →
the full distance-alternative set `B = {1/16, 1/8, 1/4, 1/2, 1, 2, 3, 4, 8, 16}`,
pruned **only** by `support >= 0.5 * max(support)` → `structural_share` → cap.

Distance-support provenance (addendum A1): `b = 1` carries support 1.0; `b = 1/m`
for `m` in `estimate_period.block_supports` (2/4/8/16) carries that correlation; the
supra-block `b = k` for `k` in 2/3/4/8/16 carries the normalised autocorrelation at
lag `k * period`. That is the half `estimate_period` deliberately suppresses via
`HARMONIC_SHARE`, and it is now explicit.

Resource caps implemented and asserted: `MAX_READINGS = 6`,
`MAX_READINGS_PER_AXIS_SIDE = 2`, `MAX_RECTIFIED_SIDE = 2048`,
`MAX_SIGNATURE_CELLS = 256`, `SIGNATURE_PIXELS_PER_CELL = 8`.

### 1.2 `plane_graph.py`

Frozen dataclasses: `PlaneMatch`, `PlaneMatchRefusal`, `ComponentAlternative`,
`ComponentSolution`, `ClassConstrainedFeatures`.

Public functions:

- `match_plane_pair(left, right, *, quadrant, max_shift_cells=32, refusals=None)`.
- `match_plane_graph(signatures, *, max_shift_cells=32, max_structural_share=None,
  refusals=None)`.
- `solve_component(readings, matches, *, integer_tolerance=0.15, min_resultant=0.90,
  ambiguous_margin=0.02)`.
- `resolved_class(solution, frame, calibration_result)`.
- `class_constrained_bestof(features, neighbors, quadrant, *, residual_degrees=1.0)`.
- `annotate_integer_residuals(solution, readings, matches)`.

The solve enumerates quadrant assignments with the anchor at `k = 0` (a gauge),
keeps same-world-axis matches, solves per-axis translations by least squares in
`t_right[a] - t_left[a] = sign_left*s_left*d_left - sign_right*s_right*d_right`
(the exact `registration.register_axis` semantics, generalised to more than one
plane per axis — `register_axis` raises on that case), and scores the integer-face
resultant `R_a = |mean(exp(2*pi*i*frac(t + sign*s*d)))|`. Typed abstentions:
`no_plane_reading`, `no_matches`, `ambiguous_quadrant`, `b_unresolved`,
`insufficient_axis_coverage`, `plane_integer_residual`, `component_too_large`.
`scale_known` is set only by the integer consistency, never by connectivity.

`match_plane_pair` warps the left cell grid into the right grid through the two
plane homographies, runs whitened phase correlation over integer shifts
`|d| <= 32`, and scores
`0.5*min(psr/6,1) + 0.3*overlap_fraction + 0.2*second_peak_margin - 0.5*(1 - share)`.
The correlation is a **scored cue**; a uniform repeating plane is refused
(`repetition_unresolved`) rather than matched.

No RNG anywhere: `class_constrained_bestof`'s robust baseline direction uses a
fixed arithmetic schedule of endpoint-distinct pairs scored by residual-qualified
support and refined to a fixed point, so it is bit-identical across runs.

---

## 2. Deviations from DS-28, each with its reason

1. **`PlaneReading` carries `rectification: PlaneRectification`** (`axis_matrix`,
   `focal`, `principal_point`, `centre`). `plane_signature(image, reading, b)` as
   specified cannot build a homography without the calibration and the plane-window
   centre, and `match_plane_graph(readings)` has no image to recover them from.
   The extra fields are all image-derived; none is metadata.
2. **`frame_plane_readings` takes `frame: str = ""`.** `CalibrationResult` records no
   frame identity, so the caller must supply the id the readings carry.
3. **`match_plane_graph` consumes `PlaneSignature` values**, each carrying `.reading`;
   a signature needs the image, and the directive's `readings` parameter cannot
   supply it. The parameter is named `signatures`.
4. **Typed match refusals are an optional out-list** (`refusals=[...]`) of
   `PlaneMatchRefusal`, preserving the directive's `PlaneMatch | None` return type
   while exposing `axis_mismatch`, `insufficient_overlap`, `no_signal`,
   `repetition_unresolved`, `same_axis_mismatch`.
5. **The rectification cap is enforced by a bounded measure, not a tiling helper.**
   `_bounded_measure` mirrors `measure_plane`'s three rounds exactly and clamps every
   sampling request to 2048; the tiling helper for larger planes is deferred.
6. **`structural_share` is "1 − the best folded-lattice concentration"**, with the
   perpendicular-seam baseline removed first, rather than the directive's
   `covered_weight / total_edge_energy`. The directive's recipe is a ratio of two
   differently-resampled sums and does not separate a uniform wall from a structured
   one in practice; the fold measure does, and it is the direct test of "is this
   edge energy explained by one lattice phase".
7. **The repetition gate is `share < 0.15` OR `psr < 3.0`**, not the directive's
   conjunction. Measured on synthetic controls, two views of a uniform repeating
   wall correlate *sharply* precisely because they are the same pattern, so the
   conjunction never fires; refusing on the absence of non-repeating evidence is the
   honest treatment of the repetition problem. The correlation remains a scored cue
   and its own refusal when structure is present but the peak is not unique.
8. **The quadrant map carries a world-axis sign**, not only the axis index. A frame
   rotated by 180° keeps the same label indices but reconstructs each horizontal axis
   pointing the other way; without the sign `k` and `k + 2` are arithmetically
   identical, and an all-vertical match set stays genuinely unresolved (control S5)
   while a horizontal one does not. This strengthens DS-28 §5.2 rather than changing it.
9. **Same-axis equality is a post-correlation consistency filter** (deterministic,
   greedy, ties drop both) rather than a pre-correlation enumeration. The candidate
   identities are already enumerated at cell resolution, so the arithmetic is applied
   to the candidates; the pre-correlation form is deferred.
10. **`plane_signature(..., usable=None)`** is an optional mask so a signature can be
    built pure outside the derived GUI mask.
11. **Two LSD passes per frame** (one for `surfaces.planar_regions`, one inside
    `boundary_candidates.find_chains`, which accepts no precomputed segments). The
    directive asked for one; extending `find_chains` would have modified an existing
    module, which this package must not do.

---

## 3. Synthetic controls

| # | Control | Where | Result |
|---|---|---|---|
| S1 | 3-view integer lattice room, exact matches | `test_s1_three_view_component_recovers_quadrants_and_translations` | quadrants `(0,1,0)` exact, translations exact (≤0.25 block), all `R_a = 1`, `scale_known=True` |
| S1' | image-level 3-view room | `test_proposal_pipeline_either_solves_or_abstains_with_a_typed_reason` | typed abstention `ambiguous_quadrant` (§4) |
| S2 | block-vs-texel, both retained, integer test picks | `test_distance_alternatives_build_all_sub_block_readings_from_block_supports`, `test_distance_alternatives_prune_only_by_the_declared_support_floor`, `test_wrong_b_is_rejected_by_the_integer_test` | full `{1,1/2,1/4,1/8,1/16}` retained from `block_supports`; pruning is exactly the 0.5 floor; the texel reading abstains with a strictly worse assignment cost |
| S3 | uniform repeating wall | `test_repetition_gate_refuses_a_uniform_periodic_wall` | no match; refusal reason `repetition_unresolved` |
| S4 | single-plane pair | `test_s4_single_plane_pair_resolves_only_its_normal_axis` | `insufficient_axis_coverage`; normal-axis `R = 1`; other two axes explicitly absent; `scale_known=False` |
| S5 | two views sharing only a floor | `test_s5_two_views_sharing_only_a_floor_leave_the_quadrant_unresolved` | `ambiguous_quadrant` (k and k+2 both valid) |
| S6 | 40 % impostor edges + consistent minority | `test_s6_class_constrained_bestof_selects_the_true_minority_and_solves`, `test_class_constrained_bestof_is_deterministic_and_one_to_one` | one-to-one true set ≥8; `solve_pair` class 1, inlier fraction ≥0.5; two runs bit-identical |
| S7 | frame relabelling + input-order permutation | `test_s7_frame_relabelling_and_input_order_do_not_change_the_solution` | identical quadrants and translations |
| S8 | mutate RGB outside the usable mask | `test_frame_plane_readings_are_pure_outside_the_usable_mask` | identical readings |
| S9 | declared caps | `test_frame_plane_signature_respects_the_declared_caps_and_is_deterministic`, `test_frame_plane_readings_are_deterministic_and_bounded` | ≤6 readings, ≤2 per `(axis, side)`, rectified side ≤2048, cells ≤256², all deterministic |

Additional controls: `rectification_homography` round-trips
`plane_coordinates_to_camera_point`/`camera_point_to_pixel` to <1e-6 px on all
axes/sides; `warp_signature` identity returns the same cells; a known 2-D shift is
recovered by phase correlation; `structural_share` orders an aperiodic patch above a
uniform one; `resolved_class` margins and the abstention vocabularies are exact.

---

## 4. Image-level proposal recall (honest negative, no threshold weakened)

On the rendered multi-material room (720x480, M3-style calibration), the existing
proposal sources yielded, per view:

```
v0: axis 1 (share 0.241, 3 alternatives)
v1: axis 0 (0.211, 2), axis 1 (0.531, 3), axis 1 (0.708, 1)
v2: axis 1 (0.524, 3), axis 1 (0.590, 4)
```

Only the vertical axis is proposed by every view, so the component abstains with
`reason = "ambiguous_quadrant"` (312 typed match refusals: 124
`insufficient_overlap`, 108 `same_axis_mismatch`, 80 `axis_mismatch`). This is
exactly the risk DS-28 §13.1 named: plane-proposal recall, not the integer
arithmetic. No threshold (`minimum_tiles`, `minimum_weight`, `minimum_margin`,
`MIN_STRUCTURAL_SHARE`, `min_resultant`) was changed; the pipeline test accepts a
typed abstention as the honest outcome, and the component arithmetic is exercised by
the exact-match controls above.

---

## 5. Source/import audit

`tests/test_plane_lattice_entry.py` proves, over the two new module files:

- no forbidden identifier appears anywhere in the AST (`load_camera_manifest`,
  `load_scene_transform`, `litematic`, `luna_annotation`, `attribution`, `visibility`,
  `oracle`, `truth`, `reference_class`, `expected_class`, `mask_debug_area`,
  `usable_mask_for`, `render_volume`, `_evaluate_pose`, `_reference_class`);
- no string constant embeds `.masks`, `cameras.json`, `scene.json`,
  `M4-C2-boundary-targets-v4`, `.litematic` or `reference.litematic`;
- no `numpy.random`, `default_rng`, `np.random` or `import random`;
- a clean subprocess importing both modules loads none of the diagnostic/evaluator
  scripts (`diagnose_match_attribution`, `diagnose_connected_pairs`,
  `diagnose_oracle_pose_failures`, `diagnose_ds25`, `blind_pose_policy`, ...);
- with poisoned `load_camera_manifest`, `load_scene_transform` and `Path.read_text`/
  `builtins.open` guards on forbidden paths installed, the component solve is
  unchanged.

---

## 6. Test evidence

```
.\.venv\Scripts\python.exe -m pytest tests\test_lattice_planes.py tests\test_plane_graph.py tests\test_plane_lattice_entry.py -q -p no:cacheprovider -o addopts=
31 passed
```

```
.\.venv\Scripts\python.exe -m pytest tests\test_lattice_planes.py tests\test_plane_graph.py tests\test_plane_lattice_entry.py tests\test_calibration.py tests\test_camera.py tests\test_lattice.py tests\test_supported.py tests\test_boundary_candidates.py tests\test_boundary_scoring.py tests\test_connected_camera.py tests\test_blind_pose_policy.py -q -p no:cacheprovider -o addopts=
189 passed in 100.13s
```

`-o addopts=` disables the repository pytest defaults; no existing test was modified.

```
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider -o addopts=
848 passed, 1 skipped in 214.88s
```

The single skip is the pre-existing external-symlink environment case.

---

## 7. Blockers and honest limitations

- **Proposal recall** (§4) is the binding limitation and is now measured, not argued
  away. A real T2/T6 diagnostic will abstain `no_plane_reading` /
  `ambiguous_quadrant` unless the proposal sources produce ≥2 same-axis planes per
  view pair; that is reportable in one run, per DS-28 §13.1.
- **The pre-correlation same-axis enumeration** and **the tiling rectification helper**
  are deferred; the implemented substitutes are documented in §2.
- **`structural_share` is entangled with the measured harmonic.** The fold removes the
  perpendicular baseline, but a region window that spans fewer than two block periods
  cannot distinguish a uniform wall from a structured one by this statistic alone;
  the phase-correlation refusal covers that case.
- **A concurrent worker modified `boundary_candidates.py` during this package**
  (an intermediate syntax error was observed at 14:44:39 and resolved by 14:45). The
  new modules import it at module scope, so a mid-write file breaks their import; no
  action was taken on that file.
- **Not implemented in this package:** `scripts/diagnose_plane_lattice_pose.py` and
  `scripts/plane_lattice_synthetic.py`, the frozen T2/T6 Phase A/B/C run, and the
  `class_constrained_bestof -> solve_pair -> select_blind_pose` end-to-end wiring.
  The directive's reworded DS-25 plumbing test was not touched.

## 8. Non-claims

No promoted gate, no production entry-point change, no accuracy or runtime claim on
real fixtures, no `scale_known` claim outside the synthetic integer control, no
manifest/F3/reference/annotation input, and no change to `SUPPORT_THRESHOLD`,
`PHASE_TOLERANCE`, DS-25 or any Luna path.
