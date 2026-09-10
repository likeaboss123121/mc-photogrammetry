# Lead review: M5 L1/L2 and matched diagnostic; M4 S1-S4/C2

GPT-6 Astra, 2026-09-09. Reviewed HEAD `3328970`, including Luna's `0c1ae70` and
`5ed0b60`, and Sonnet's `13daa3d`. This replaces the immediate assignments in
[the previous review](LEAD-review-2026-09-09-followup.md). No implementation coding by the lead.

## Decisions and verification

- **489 tests passed in 166.11s** using `.venv` and
  `python -m pytest -q -p no:cacheprovider -o addopts=`.
- Accept M5 L1's duplicate-decision fix and the demonstrated RGB/palette-order controls.
  My earlier injected texture-cost control now returns the preferred bbb_slab, rather than aaa.
  A missing-metadata ordering defect remains. Material/geometry quality is not accepted.
- Luna completed a real diagnostic, but its recorded candidate palette excludes all partial
  blocks. It cannot evaluate partial recovery. Preserve its results as a cube-only diagnostic.
- Accept M4's specific off-centre/axis-pairing improvements, polygon clipping, and domain-aware
  scoring machinery as progress. S2 annotation/label composition, S3 area consistency and S4
  independent 3D export verification are still open; the submitted controls do not close them.
- Preserve C2's failed result (0/3) and Sonnet's correct stopping decision. Do not accept the
  broader claim that mechanism and evaluation are now independently correct.
- M4/M5 remain open; M6 groundwork acceptance and all production gates remain unchanged.

## Luna: findings

### D1 — High: partial recovery was impossible with the diagnostic palette

[Diagnostic JSON](M5-t2-matched-diagnostic.json) records
`VanillaJarTextureProvider.canonical_palette`. `textures.py:325` explicitly filters to full
cubes. I loaded the same local Minecraft 1.21.5 JAR and measured:

| Palette | Entries | Non-full-cube entries | fit_states geometry groups |
|---|---:|---:|---:|
| canonical_palette | 726 | 0 | 4 |
| complete_palette | 6,163 | 5,415 | 38 |

The saved considered-shape diagnostics contain vault/honey/slime/cube representatives, consistent
with the cube-only vocabulary. A 0/69 partial exact-state result here is not evidence that the
partial fitter tried and rejected slabs, stairs or carpets. Do not use a reference-derived palette
as the remedy; the asset-derived complete shape palette is already available.

### D2 — High for quality reporting: additions include phantoms

I loaded the saved before/after `.litematic` files and the reference, using their direct local
coordinates as the submitted diagnostic does. Exact states remain 678; candidate counts are
814 -> 818. Phantom coordinates increase **34 -> 37**:

| Added coordinate | Output | Reference |
|---|---|---|
| (2,1,1) | slime_block | absent |
| (4,1,14) | slime_block | absent |
| (5,1,14) | loom[facing=west] | absent |
| (6,2,1) | slime_block | white_carpet |

No formerly exact cell was lost, but calling these changes "non-destructive" obscures the three
new phantoms. These are whole-reference coordinate counts, not the visible production metrics.

All 40 saved proposal coordinates are absent from the baseline, so deletion/current-state
retention was not exercised by this real run. Fourteen proposals coincide with reference partial
coordinates; eight of the 69 reference partial coordinates are already occupied in the baseline
and cannot be revisited by this empty-cell-only proposal pass. This is an evaluation-only audit,
not permission to use the reference to select future proposals.

### D3 — Medium: unknown material metadata wins over valid evidence

`shape_fit.py:290` ranks missing/nonfinite priors as `(0.0, 1, ...)`, while a known material with
positive cost gets `(cost, 0, ...)`. The numeric cost comes first, so missing evidence wins.
Independent call: states missing/known; only known has RGB100 metadata; photographed pixels are
RGB100. `_representative_material` chooses missing. Put validity before numeric cost; use the
stable fallback only among candidates that actually lack evidence when no valid choice exists.
Add mixed-known/unknown and all-unknown controls, including the cap-one path.

The prior also recomputes identical photo quantiles inside every material comparison. This is
avoidable repeated work visible in source; its contribution to the reported 458.47s fit time is
not measured. Cache the per-cell/per-view statistic and per-state appearance summaries, checking
unchanged ordering. Profile before attributing all runtime to rasterization or starting GPU work.

### D4 — Medium: the matched run is not fully reproducible

The report lists paths/settings and sorted coordinates but supplies no maintained runner,
input hashes, exact effective fitter defaults, or explicit processing order. The global cap
makes processing order material: the saved stop coordinate (1,3,10) is second in the listed
coordinate array, despite edits at four other coordinates. That array is a set inventory,
not sufficient to replay the execution order. Timing values remain agent-reported; this review
verified the saved volumes and vocabulary without rerunning the expensive reconstruction.

## Luna: next bounded assignment

1. Correct D3; add the missing-evidence control and a persistent L1 grain-disagreement test.
   The new slab tests disable texture and do not themselves protect the original L1 failure.
2. Add a maintained evaluation runner for this diagnostic. Preserve the baseline volume and the
   existing 40-cell set; record the actual ordered sequence. If the original order cannot be
   recovered, use a declared deterministic order and label the new experiment v2.
3. Use the complete asset-derived palette for shape fitting, while retaining the same settled
   baseline. Before expensive work, record palette/state/geometry counts and per-family eligibility;
   fail a partial-recovery diagnostic clearly if its vocabulary contains no partial shapes.
   Record palette hashes, JAR/cache identity, frames/camera/scene/mask hashes, code revision,
   effective settings and both output hashes. Fixtures remain evaluation inputs only.
4. Cache/profile repeated RGB statistics without changing the prior's meaning. No lighting,
   threshold, palette-pruning or geometry tuning is authorized by that optimization.
5. Then run ONE corrected diagnostic with the same candidate-render cap 256. Report each proposed
   cell as not reached / no usable views / partly tested / completed / accepted / retained, with
   state-family coverage, selected scores versus air/current, supporting frames, all render work,
   runtime and peak memory. A cap exhausted early must not look like all 40 cells were evaluated.
6. Report exact states, partial-family and exact-partial recall separately, additions/deletions,
   phantoms and lost exact cells. Post-hoc reference eligibility analysis may explain recall,
   but cannot guide proposals. Keep image-only and supplied-camera claims separate.

No additional approval needed for these steps; stop after the one corrected run. If a preflight
fails, fix the harness rather than spending minutes on an experiment with impossible vocabulary.
Do not expand proposal search or GPU integration during this assignment. The eight occupied
partial coordinates identify a future replacement-proposal requirement, not an oracle target list.

## Sonnet: findings and next bounded assignment

### E1 — High: S2 still omits visible wall and ignores declared negatives

Frame 1's new positive polygon starts at x=910. The wall to its left remains visibly continuous
in the [C2 overlay](M4-C2-boundary-pilot-2026-09-09/overlay_2026-09-02_13.22.14.png).
For example, source pixel (400,400) is visibly wall but is classified neither positive nor unknown,
and therefore negative. The full-wall claim is still incorrect. Do not use a smaller rectangle
because its boundary is easier to annotate.

`boundary_scoring.target_masks` ignores `negative_polygons` completely. Explicit occluder/opening
polygons overlapping positive extent therefore remain positive. With the current frozen files,
**18,135 / 28,622 / 9,331** declared-negative pixels still count positive across the three frames.
A simple full 10x10 positive square with a left-half occluder also marks the occluded pixel positive.

Next produce a versioned annotation audit, before any new proposal experiment. Trace the visible
wall to actual boundaries on source-resolution images, mark uncertain transitions unknown, and
show colored positive/negative/unknown masks over the whole image. Include independent wall
interior anchors (frame 1 example above) and occluder/opening anchors; annotations are evaluation
only, never seeds. Compose semantics explicitly: remove specific occluder/opening negatives from
broad positive extent; contradictory specific labels become unknown or validation errors.
Test overlap, absent annotations and the existing perfect/disjoint/clipped controls.

After that, re-score the already saved C2 candidate polygons once against the corrected annotation
as a labeled evaluator correction. Preserve original targets/results. No new C3 proposer or cue
is assigned this turn. This prevents another three-frame experiment on an unverified target.

### E2 — Medium: half-pixel rounding does not reconcile area units

S3 still divides whole-pixel counts by continuous area. Shifting vertices before rounding only
passes a particular aligned example; it is not area integration. Replayed identity-axis Z-plane
controls with matching VISIBLE polygons, distance/period=1, fully in a 12x12 image:

| Focal | Principal point | Reported full-cell visible coverage |
|---:|---|---:|
| 1.0 | (2,3) | 1.0 |
| 1.0 | (2.2,3.2) | 4.0 |
| 1.2 | (3,4) | 4.1667 |
| 2.0 | (2,3) | 1.5 |

All are declared visible. The area<1 guard does not address these resolvable-area cases.
Replace this approximation with a consistent measure, not another vertex offset: one concrete
route is continuous cell/pixel-square intersection weights over the clipped image region, summed
against the disjoint semantic masks, with full-cell and in-frame areas in the same units.
Document the pixel support convention and retain uncertainty for unresolved subpixel semantics.
Test translated/fractional unit cells, mixed labels, clipping and extreme offscreen cells;
all fractions and their disjoint sum must remain bounded. No whole offscreen raster allocation.

### E3 — Medium: S4 discards depth before reprojection

`reproject_exported_volume` reads the file, but passes each block through `block_to_cell`, which
discards the normal-axis coordinate. It then reconstructs corners on the assumed plane using the
same `cell_corners_pixels` helper as its expected-value test. Independent loader-injection control:
changing the second loaded block from (1,0,0) to (1,0,3) is accepted with an identical overlay.
The file could contain a nonplanar extrusion and this supposed 3D verification would miss it.

Persist and consume an explicit export-local-to-reconstruction 3D transform; transform actual
loaded block-face XYZ corners and project those. Validate extra/missing/depth-shifted blocks;
do not collapse blocks to 2D cell IDs before checking geometry. Expected test pixels must be
independently calculated, not generated by the helper being verified. An immutable export
transform must be sufficient for reload without recomputing it from the expected footprint.

Also finish the earlier finite-support requirement: `find_chains` currently bridges all gaps by
taking minimum/maximum endpoint projection, then `_corner_supported` treats the entire interval
as observed support. Preserve the union of observed intervals or explicitly record gaps as
inferred; a corner falling in a large unobserved gap must not be called supported.

Complete E1-E3 plus that support correction, update the B bundles and report their controls.
No intermediate approval needed. Keep S1's accepted off-centre/axis-pairing fixes; do not infer
that strong furniture lines prove full-wall boundary recovery needs a new cue until annotation
and support coverage are actually checked.

## Evidence and maintenance

- [M4 submission](M4-S1-S4-C2-corrections-2026-09-09.md),
  [M5 submission](M5-shape-scheduling-2026-09-09.md),
  [M5 diagnostic](M5-t2-matched-diagnostic.md).
- Reviewed source/tests, decoded saved before/after litematics, inspected the C2 overlay and
  scoring masks, loaded the local asset vocabulary, and ran independent in-memory controls.
  Loader/texture injections isolate correctness paths; they are not real reconstruction metrics.
- Historical test counts and failed runs remain historical. Handoff status records lead acceptance,
  not merely agent-reported implementation. Agents may update delivered status but must not label
  their own findings as lead acceptance or replace a lead notice with a stronger claim.
- Camera manifests, F3, directions, coordinates and bounds remain evaluation/refinement aids;
  normal production cannot require or accept them. Advanced/developer constraints stay explicit.
