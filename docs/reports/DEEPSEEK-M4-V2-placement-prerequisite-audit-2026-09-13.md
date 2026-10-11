# M4 V2 Placement Prerequisite Audit

Date: 2026-09-13
Role: DeepSeek read-only/no-shell audit, verified by lead source review
Status: design evidence; no implementation or evaluation performed

## Finding

V2 cannot yet emit absolute lateral block coordinates. Normal-axis face placement is derivable from
the accepted graph solution, but the current signature grid is not anchored to observed block-seam
phase.

- `plane_signature` derives `origin_uv` by flooring a region-window centre and extent
  (`lattice_planes.py:1403-1409`). The centre originates in `plane_window`
  (`lattice.py:408-415`), so this is window-relative rather than a physical seam anchor.
- `PlaneReading.phase_pixels` is measured (`lattice_planes.py:876-884`) but is not consumed by
  `plane_signature`; it currently contributes only to reading ordering.
- `_peak_statistics` returns the best relative cell shift (`plane_graph.py:294-295`), but
  `match_plane_pair` binds it as `_shift` and discards it (`plane_graph.py:347`). `PlaneMatch`
  has no shift field (`plane_graph.py:92-118`).
- Treating `origin_uv` as a world `lateral_origin` would repeat the phase assumption rejected in
  `LEAD-review-2026-09-09.md:74-88`.

## Existing signed geometry

For reading label axis `a`, quadrant `q`, camera-to-plane side `s`, retained alternative distance
`d`, and solved translation `t`:

- `world_axis = axis_index(a, q)` using `plane_graph.py:184-198`.
- `axis_sign = axis_sign(a, q)` using `plane_graph.py:201-215`.
- `face = t[world_axis] + axis_sign * s * d`, identical to the graph's face arithmetic
  (`plane_graph.py:528-533,736-738,792-797`).
- The visible face's outward sign is `-axis_sign * s`. `PlaneReading.side` is the direction from
  camera to plane and must not be passed directly as `DeclaredFootprint.side`.
- `others = [i for i in (0,1,2) if i != a]` in ascending label-axis order
  (`lattice.py:418-427`). Local `u` maps through `others[0]`, local `v` through `others[1]`.
- Cell `(row, column)` maps to local `(u,v) = (origin_u + column, origin_v + row)`, consistent
  with `footprint.py:156-178` and `footprint_adapter.py:148-155`.

The corresponding anchor-frame point before integer phase resolution is:

`t + axis_sign(a,q)*s*d*e_world_axis + axis_sign(others[0],q)*u*e_u_world + axis_sign(others[1],q)*v*e_v_world`.

This is a fractional, gauge-relative point. It is not proof of an absolute Minecraft block cell.

## Additional translation hazard

An abstained solution may still carry translations. Least-squares also min-norm-fills an
unconstrained frame/axis with `0.0` (`plane_graph.py:747-774`), while component acceptance only
requires that all three axes appear somewhere. Placement must therefore require an accepted solution
and explicit per-frame/per-world-axis support; field presence is insufficient.

## Revised sequence

### P1: phase and relative-shift evidence

1. Propagate measured phase into a typed, finite, canonical per-reading/per-axis phase anchor with
   source-region provenance. Converting pixel phase into block phase must use the retained period and
   document its modulo convention.
2. Retain the correlation peak's row/column shift in `PlaneMatch`, with the same PSR, peak,
   second-peak and overlap evidence that admitted the match.
3. Expose one public quadrant transform returning world-axis index and sign. The graph, selector and
   placement code must call the same implementation.
4. Retain per-frame/per-world-axis translation support from the accepted match set.
5. Solve only relative lattice offsets supported by accepted, non-repetitive matches. Preserve the
   unavoidable whole-build integer translation gauge explicitly.

### V2: solved-reading world placement

After P1, emit a frozen placement record containing frame, reading/source identity, canonical retained
`b` and distance, quadrant, world normal axis, camera side, outward sign, face coordinate, lateral
world axes/signs, phase-anchored lateral origin, and per-axis support evidence. It must consume an
accepted solution and image-derived P1 evidence only.

Typed abstentions include: solution abstained, frame absent, quadrant invalid, axis unsolved,
frame-axis unconstrained, alternative not retained, non-finite geometry, face integer unresolved,
phase unavailable, relative shift ambiguous, and repetition unresolved. No missing phase or
translation may default to zero.

## Decisive tests

- All three normal axes, four quadrants and both camera sides against independent axis/sign tables.
- Horizontal `q` versus `q+2` sign reversal and vertical invariance.
- Non-square/non-symmetric grids proving column-to-u and row-to-v without transpose.
- Known fractional seam phase: invariant under integer window movement, changed by fractional seam
  movement, and distinct from `origin_uv`.
- Two views recover the known relative row/column shift; repeating texture abstains.
- Accepted component with one unconstrained frame axis refuses that reading instead of using `0.0`.
- Abstained solutions with populated translations always refuse placement.
- Retained-`b` canonicality, global translation-gauge equivariance and deterministic ordering.
- Independently constructed synthetic world coordinates round-trip through the placement and existing
  `cell_to_block` convention.
- Import/signature isolation forbids dataset, evaluator, manifest, reference, F3, manual camera,
  manual bounds and manual lateral-origin inputs in default mode.

## Boundary

P1/V2 use only in-memory image-derived readings, signatures, matches and accepted graph solutions.
Manual coordinates, directions, F3 values, camera manifests, bounds, reference schematics and fixture
poses remain advanced/developer/evaluation-only. Frozen schema-v1 artifacts remain unchanged; their
omitted provenance/phase data means loaded legacy records must abstain from the new placement path.

## P1 feasibility correction

A second read-only audit of `_phase_pixels`, `phase_at_period`, and the correlation convention narrows
what P1 may claim:

- `phase_pixels=(phase_u, phase_v)` is the circular phase of summed structural edge energy along
  columns/u and rows/v (`lattice_planes.py:876-884`; `supported.py:193-221`). It can describe a
  texture repeat or a multi-block pattern; it is not intrinsically a block seam.
- For retained `b`, the tempting conversion `(phase_pixels * b / period_pixels) mod 1` is independent
  of the unknown integer period index only when `b` is integer. For fractional `b`, adding one
  indistinguishable measured period changes the result by fractional `b`; choosing one residue would
  fabricate which sub-period is the block boundary.
- The current zero return conflates no usable phase signal with a legitimate zero phase. P1 therefore
  needs an explicit availability/quality state rather than treating `(0,0)` as measured evidence.
- The FFT peak convention is `(row_shift, column_shift)=(delta_v,delta_u)` and satisfies
  `warped_left[x + shift] ~= right[x]`. Registering right onto warped-left applies the opposite index
  shift. The present test accepts either sign, so P1 must first pin the sign with an asymmetric
  non-square control.
- The shift is residual after both homographies and both `origin_uv` translations
  (`plane_graph.py:218-241`). It is confounded by calibration, scale and window errors and is safe only
  as match-specific relative evidence with its quadrant, b-pair, PSR, peak, second peak and overlap.

Revised P1 output is relative, hypothesis-scoped evidence. It may expose a canonical phase for an
integer `b` only with measured availability and quality, and must still label it periodic-energy phase
unless a separate test identifies physical seams. Fractional-`b`, absent-signal, repetitive, refused,
unsupported-frame-axis and abstained-solution cases remain typed abstentions. Absolute lateral block
placement stays closed until these ambiguities are resolved rather than rounded away.

## P0 prerequisite

`rectify_plane` samples both endpoints, so its exact coordinate step is
`2*extent/(samples-1)` (`lattice.py:466-468,493-505`). Current period-to-world conversions instead
use `2*extent/samples` in `lattice.py:849-850`, `lattice_planes.py:795,1401`, and
`plane_graph.py:222-225`. The 2026-09-08 lead directive already required a single consistent
convention. P0 must unify this arithmetic before P1 phase formulas or V2 placement are implemented;
otherwise the phase, stored distance, signature homography and graph homography describe different
sample grids.
