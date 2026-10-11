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
## Accepted implementation details

Compatibility requires optional `witness_sink=None`, no witness allocation when absent, unchanged return types/records/serialization/algorithm IDs/numerical operations, and no production imports of evaluation consumers. Each record binds schema, producer revision, runtime/build identifiers, input/parameter hashes, exact float metadata, dtype/shape/bytes array hashes, and record digest.

`RectificationWitnessV1` captures actual float32 maps, depth-validity, OpenCV version/build, interpolation/border policy, and fixed-point `INTER_BITS=5` tap indices/weights from `cv2.convertMaps`; weights use denominator 1024 and out-of-image constant-border taps are explicit. Unsupported equivalence refuses without changing remap. `SignatureSamplingWitnessV1` captures RGB/usable/region rectification witnesses, float32 channel mean, exact gradient stencils, cell/interior rectangles, energy and coverage rules. `MatcherPairWitnessV1` captures actual nearest-neighbour warp indices/validity, `np.rint`/clipping, selected shift and deterministic right-grid row-major pairs; pair count must equal `PlaneMatch.overlap_cells`. `RendererOwnershipWitnessV1` records owner rows at the existing owner-index append point: block/state, source kind, model-face/box ordinal, slot, unclipped vertices/normal, geometry hashes, scale/alpha/depth/draw policy; only certified axis-aligned geometry receives a dyadic physical-plane key.

Collector-wide caps are required for rectified samples, bilinear taps, gradient terms, cell memberships, pair entries, owner faces/vertices and payload bytes. Preflight bounds before allocation; commit only complete records; cap failure is atomic `budget_exhausted`. Required controls cover interpolation boundaries/zero weights/borders/depth/dtypes, gradient edge and cross-cell dependencies, unequal grids/origins/quadrants/ties/reverse and ±32 shifts, distinct subfaces/culling/alpha/depth/non-axis geometry, deterministic repeats, tamper/version/backend refusal, and cap boundaries.