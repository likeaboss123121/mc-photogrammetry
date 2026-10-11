# M4 supported-plane to BlockVolume bridge audit

Date: 2026-09-13  
Role: DeepSeek read-only/no-shell source audit, independently checked by GPT-6 Astra  
Status: design evidence; no implementation or promotion

## Result

The current M4 output cannot be converted honestly to visible blocks yet. The
first required package is reading-to-region provenance and an image-supported
cell footprint. `PlaneReading` retains a bounding polygon and scalar coverage,
but discards the exact proposal mask that produced the reading. For tile-grown
regions the polygon is only a bounding rectangle. `PlaneSignature.coverage`
rectifies the whole usable-image mask, so it proves that pixels are usable, not
that the candidate plane owns them. Rasterizing either field would fill gaps
and invent geometry.

This prerequisite is independent of G1F selector performance and can follow its
current diagnostic work. It is also necessary before any M4-to-volume bridge,
normal CLI/UI wiring, persistence migration, or geometry-gate claim.

## Verified source facts

- `PlaneReading` contains frame, axis, side, scale alternatives, phase, extent,
  polygon, scalar `coverage_cells`, confidence, and rectification, but no source
  proposal ID or support mask (`lattice_planes.py:145-175`).
- `_measure_plane_candidates` receives `region_mask` and uses it for measurement,
  then omits it when constructing the reading (`lattice_planes.py:680-774`).
- `_proposal_mask_candidates` computes a stable mask digest for sorting but
  returns only axis, mask, polygon, and strength; the digest/source identity is
  discarded (`lattice_planes.py:802-911`).
- Tile-region `_region_polygon` is an axis-aligned bounding rectangle
  (`lattice_planes.py:267-274`).
- `plane_signature` builds cell `coverage` from the whole supplied usable mask
  (`lattice_planes.py:1069-1096`).
- World-axis and sign resolution exist only as private `_axis_index` and
  `_axis_sign`; solved face positions use translation plus signed image-derived
  distance (`plane_graph.py:184-215, 736-745, 785-797`).
- A `ComponentSolution` abstention can still contain quadrants and translations;
  consumers must require an accepted selector result, not merely present fields
  (`plane_graph.py:680-702`).
- `BlockVolume` and the litematic exporter exist, but there is no supported-plane
  adapter. Normal CLI and desktop still call `reconstruct_facade`; desktop uses
  only the first image (`cli.py:318`, `ui/main_window.py:76`).
- `.mcscan` schema 1 persists no cameras, lattice, alternatives, or review state
  (`domain/models.py:141-171`).

## Required contracts

1. Preserve a stable source-region identity for every reading. The identity must
   survive deterministic proposal ordering and must not expose evaluation labels.
2. At the selected blocks-per-period alternative, rectify the reading's own
   region mask intersected with the usable mask into a one-cell-per-block support
   footprint. Preserve coverage values and grid origin; do not threshold texture
   energy into occupancy.
3. Expose one tested placement helper mapping a solved reading to world normal,
   face coordinate, outward sign, lateral axes, and lateral signs. It must reuse
   the graph's quadrant convention rather than duplicating it in the bridge.
4. Convert only supported footprint cells into one block-thick observations.
   Never extrude behind a face or fill a polygon/bounding box. Derive output
   bounds from emitted cells.
5. Preserve unresolved scale/quadrant and multi-view conflicts as typed review
   alternatives. Do not choose with confidence; current confidence is not a
   calibrated probability.
6. Keep supplied cameras, F3 values, manifests, scene transforms, bounds, JARs,
   and reference structures outside the normal bridge. They remain developer or
   evaluation inputs and must label a run assisted.

## Implementation order

### V1: region provenance and footprint

- Extend the proposal-to-reading path with an immutable source ID and retained
  mask provenance without changing measurement results.
- Add alternative-specific region-support rasterization using the same lattice
  rectification geometry as `plane_signature`.
- Test disconnected/L-shaped masks and openings so their bounding rectangles are
  never filled; test usable-mask intersection, alternative scale, ordering, and
  no forbidden metadata imports.

### V2: solved-reading placement

- Add a public frozen placement value and helper around the graph's quadrant,
  axis, sign, translation, distance, and integer-face conventions.
- Exhaust all four quadrants in synthetic rooms and round-trip cell corners to
  known block coordinates. Reject abstained solutions even when they carry
  translations.

### V3: diagnostic supported-volume bridge

- Produce `ReconstructionResult`/`BlockVolume` from accepted solved readings and
  their own support footprints.
- Emit exactly one surface layer per supported cell, real frame IDs, deterministic
  merges, and typed conflicts/alternatives. Use an explicitly geometry-only state
  until M5 face-material fusion is connected.
- Test no phantom shell, no unsupported fill, reordering determinism, ambiguity
  abstention, and absence of dataset/evaluator/reference dependencies.

### V4: production entry and persistence

- Build one multi-image production entry using independent per-frame image
  calibration, proposals, readings, exact selector, supported-volume bridge,
  material fusion, and review output.
- Replace normal CLI and desktop facade calls; retain manual metadata solely in
  explicit advanced/developer mode.
- Migrate `.mcscan` with backward schema-1 loading, then export through the
  existing litematic exporter with derived bounds.

## Remaining blockers after the bridge

- The exact selector must yield accepted/reviewable alternatives within a bounded
  production budget; G1F-B is measuring its rejection and repeated-state shape.
- M5 must classify and fuse face states before the 80% exact-state gate can pass.
- Production visibility remains image-support-based rather than occluder-complete;
  evaluation must report that scope precisely.
- T2, T6, and frozen small-view T3 must each pass 0.90 visible geometry F1 and
  0.80 visible exact-state recovery before promotion.

No source, test, image, ground-truth, evaluator, or frozen artifact was modified
or executed for this audit.
