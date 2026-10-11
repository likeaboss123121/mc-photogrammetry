# M4 placement and volume bridge contract — 2026-09-19

Status: Astra design proposed; implementation and production wiring remain blocked.
Route: `route-m4-placement-volume-bridge-contract-2026-09-19-ab52e105ea37`.

The accepted V2-B accounting result is still an isolated prerequisite. Its public
fixture is `partial`, so it cannot authorize placement. Two compatibility repairs
must precede V2 placement:

1. C2-R-B permits full signed dyadic translations, while V2-B
   `_translations_by_frame` currently accepts only modulo-one values. Valid `3`
   and `-1/4` translations are rejected.
2. The P1-B contributor branch hard-codes tangent labels `u/v` to world labels
   `0/1`; it must use each reading's actual tangent-axis labels.

## V1 — provenance and phase-aligned support

Reuse validated `PlaneRegionEvidence`, packed masks, and `ReadingIdentity`. Add a
versioned immutable `AlignedRegionSupport` containing source-region and alternative
digests, phase-aligned origin/dimensions, row/column convention, per-cell
region/usable/in-frame evidence, typed supported/unknown state, policy, counters,
source bindings, and canonical digest. Reproject the original retained mask
intersected with the validated usable mask onto unit cells; never rasterize a
bounding polygon or infer air from absent support. Initial diagnostic policy emits
only cells whose projected area is conservatively established inside region,
usable pixels, and the image; boundary cells remain unknown.

## V2 — solved face placement

Add a frozen `ResolvedFacePlacement` outcome using replay-validated C2-R-B,
V2-A, and V2-B evidence. Require one connected exhaustive cycle-validated
translation class, `endpoint_unique`, V2-B determined on all three axes, and
supported translation for every required frame/world axis. Replay with
`validate_phase_anchor_replay` before consuming the anchor.

Use full certificate translations `T`, seam offsets `alpha`, and the shared
`axis_transform`:

```text
(n, sn) = axis_transform(a, q)
face_gauge = T[n] + sn*c*d
F = face_gauge - alpha[n]
outward_sign = -sn*c
normal_block = F - 1 if outward_sign > 0 else F
```

Require exact integrality under rational arithmetic; otherwise return
`face_integer_unresolved`. Tangent labels are the sorted axes excluding the normal
label and use the same transform. Preserve full translations, fractional anchors,
integer face, tangent mappings, support references, and the unresolved whole-build
integer gauge. Never substitute a supplied world position.

## V3 — diagnostic supported-volume bridge

Add `SupportedCellObservation`, `CellConflict`, and canonical `PlaneVolumeResult`.
Emit exactly one adjacent block for each supported face cell; do not extrude,
close shells, fill gaps, or derive coordinates from output bounds. Merge identical
contributions deterministically, retain real frame IDs, preserve alternatives and
typed conflicts, and omit unresolved cells. Adapt to existing `BlockVolume` and
exporter without schema changes. A fixed display state is permitted only for an
explicit geometry-only preview; it cannot feed M5 or exact-state scoring. Keep
`complete=False` until material and production gates pass. Use `bounds=None`, record
the exporter normalization transform, and preflight dense bounding-box allocation.

## V4 — production and persistence

After V1–V3 acceptance, add one multi-image image-derived entry connecting
calibration, proposals, accepted solving, support/placement, M5 state fusion, and
review output. CLI/desktop wiring belongs only here. Introduce a reviewed schema-2
`.mcscan` envelope with evidence blobs, digests, alternatives, gauge/export
transforms, review state, and assisted-mode provenance; preserve schema-1 loading,
with legacy records abstaining from new placement.

Acceptance is sequential: provenance/support masks and alignment; three-axis/four-
quadrant/both-side placement including negative/full translations and ambiguity
abstention; sparse deterministic volume/conflict/export behavior; then an
all-three-axis determined replay-valid fixture before any production promotion.
Normal mode accepts images and pipeline-owned evidence only. Manifests, F3 values,
coordinates, directions, bounds, manual anchors, references, and packs remain
explicit assisted/developer inputs.

Suggested isolated implementation scopes are `plane_region_support.py` (V1),
`plane_placement.py` (V2), and `plane_volume.py` (V3). No source edits or production
switch are authorized by this design task.
