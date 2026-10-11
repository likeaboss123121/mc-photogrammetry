# DS-R3-AS: Exact sampling and ownership witness seams

**Date:** 2026-09-20
**Status:** accepted additive evaluation contract; implementation planned

DS-R3-AR correctly refuses surface-local support because current records discard exact source dependencies and physical subface identity. Add three independent, output-neutral witnesses; production defaults remain inert and legacy serialized records unchanged.

## 1. Rectification/signature witness
In `lattice.rectify_plane` / `plane_signature`, optionally retain source shape, exact `map_x/map_y` digest, OpenCV version, interpolation/border IDs, and CSR per rectified sample of nonzero source linear indices with effective OpenCV linear weights. Derive exact weights with `cv2.convertMaps(..., CV_16SC2)` and decode the 5-bit interpolation-table index (1024 denominator), including border taps. Record gradient stencil (`edge_order=1`), cell/subsample/band geometry, and a sorted CSR union of source pixels contributing through gradient neighbours to each aggregate cell. Bind RGB/usable/map/tap/cell-footprint hashes. Add oracle tests against `cv2.remap` for integer, fractional, and border coordinates; pin OpenCV version/interpolation IDs.

## 2. Renderer ownership witness
Add an optional evaluation-only render sink/API; preserve legacy tuple behavior. For each emitted/rasterized owner retain cell coordinate, block-state/palette identity, slot, model-face or box ordinal, exact local corners, alpha identity, axis/sign, and exact dyadic plane coordinate. Repeated same-slot subfaces remain distinct. Unresolved or non-dyadic geometry is `unknown`.

## 3. Matcher pair witness
Add an optional nonserialized sink/shared helper emitting selected support pairs in row-major order with left/right logical cell coordinates, canvas origin/shape, selected shift, coverage admission, algorithm/radius, and a digest. Assert pair count equals `PlaneMatch.overlap_cells`; do not reimplement shift/origin conventions.

## Acceptance and boundaries
Each seam gets focused oracle/tamper/budget tests and preserves normal behavior when no sink is supplied. No production thresholds, decisions, graph fields, camera plans, launcher changes, or metadata coupling. Integrate AR only after all three witnesses independently pass review.