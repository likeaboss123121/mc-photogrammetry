# Lead follow-up review: M4 B/C, M5 scheduling, M6 repairs

GPT-6 Astra, 2026-09-09. Reviewed `e842c0e`, including M4 `57eeb38`/`8a4c5de`
and M6 `cda794d`. Lead work is review, execution of controls, and documentation only.
This supersedes the immediate assignments in [the earlier review](LEAD-review-2026-09-09.md).

## Decisions

- M6 C1/C2 repairs are accepted: oversized replacement accounting and owned snapshots
  now implement the requested behavior. Workgroup sizing and empty-batch fixes also landed.
  Production GPU rendering/integration remains planned, not established by transfer probes.
- M5 now reaches partial geometry before exhausting the budget on cube materials.
  Supporting-frame provenance no longer depends on optional diagnostics. Accept those fixes,
  but do not accept material selection or palette-order invariance yet.
- M4 G2's row/column mapping is corrected, and G3's assumed-grid status is more honest.
  G1's overlap/clipping examples are improved, but mixed area units introduce a new defect.
  Export verification still lacks the requested independent read-back face reprojection.
- Sonnet correctly stopped after C's three runs. Preserve the reported 0/3 result, but it
  does not establish that the geometry is correct or that a new visual cue is necessary.
  The proposal restrictions and evaluation target definition need correction first.
- M4/M5 remain open. Production camera recovery, acceptance gates, and advanced-mode
  restrictions are unchanged; no implementation milestone is promoted by this review.

## M4 findings and next assignment — Sonnet

### S1 — High: C cannot represent basic off-centre planes

`boundary_candidates.py:228` assigns left/right and top/bottom roles by the image centre,
not by pairs of boundaries on a candidate plane. Both edges of an off-centre wall therefore
compete for one side, and an image border substitutes for the missing opposite side.

Independent direct control: a 100x100 frame, vertical chains at x=10 and x=30, horizontal
chains at y=20 and y=80. The desired rectangle is (10,20)..(30,80). The only proposals instead
span x=10..100 and x=30..100. No detection noise or missing region seed is involved.

Axis identity is retained but ignored when composing edges. Mixing two different horizontal
world-axis pencils does not describe opposite sides of one Manhattan plane. Failed intersections
become (0,0), vertices are individually clamped, and combinations are truncated before validity
checking. None of those operations establishes a valid planar quadrilateral. The single centred,
isolated rectangle control cannot establish general geometry correctness.

Correct this existing mechanism before adding cues:

1. Enumerate compatible axis pairs; pair distinct lines within each pencil without requiring
   them to straddle the frame centre. Use finite segment support and corner connectivity.
2. Preserve segment extents. Current `span` is offset spread, not along-line support; grouping
   nearby offsets alone does not establish a contiguous physical edge.
3. Reject parallel/degenerate/self-intersecting combinations; clip valid polygons geometrically.
   Missing edges remain uncertain. A frame edge may bound a visible crop but is not an observed wall edge.
4. Keep bounded, spatially diverse hypotheses. Retain the 12-chain/8-output limits for the next
   diagnostic; apply them after compatibility/validity and distribute the chain budget across axes
   and image regions rather than allowing one cluster to take every slot.
5. Controls: off-centre rectangles in each quadrant, rolled perspective, two competing planes,
   separated collinear segments, parallel edges, and absent boundaries. No target mask in proposals.

### S2 — High: the C target is a positive patch, not the evaluated plane boundary

The frozen [targets](M4-C-boundary-targets-2026-09-09.json) explicitly describe conservative
sub-regions. Frame 1's rectangle x=832..1152, y=13..730 is a strip inside the much larger visible
wall. The [overlay](M4-C-boundary-pilot-2026-09-09/overlay_2026-09-02_13.22.14.png)
shows its vertical edges cut through continuous wall texture. Pixels outside this strip are
not established negatives, yet `_region_metrics` counts all candidate coverage there as false
positives. A correct full-wall boundary can fail by design. Hashing freezes the annotation;
it does not validate its semantics. The reported boundary error also ignores the usable domain
and compares the original extent polygons instead of the scored visible-region boundaries.

The proposal sees the original HUD-bearing image even though scoring excludes HUD bands.
This is not evidence of F3 coordinate parsing, but HUD line features can still steer proposals.

Next, version a corrected evaluation bundle rather than rewriting old results:

- Before running the corrected proposer, annotate the full visible target wall, including
  occlusion/openings and explicitly uncertain boundaries. Record uncertainty as unknown, not negative.
  Inspect source-resolution crops as needed; a resized preview is not a reason to use arbitrary strips.
- Freeze the evaluation domain and annotation hash. Precision needs known positives and negatives;
  exclude unknown pixels symmetrically from candidate and reference. Report excluded area.
- Compare the same visible-region boundaries inside that domain for boundary error; state the
  distance formula and normalization. Add perfect-region and occluded/clipped analytic controls.
- Propose regions scene-wide. Report the algorithm's top-ranked candidate separately from
  evaluator-only target matching across the bounded candidate set. A correct other wall is not
  proof of a phantom; never use target matching to choose the production output.
- Independently supplied image dimensions/frame inventory must drive proposals. Apply the fixed
  HUD usability mask during feature extraction without creating artificial mask-border edges;
  target geometry and reference cameras remain unavailable to the proposer.

After S1/S2 controls pass, one corrected configuration on the same three frames is authorized
without another approval turn. Preserve old C as an invalidly targeted historical diagnostic;
label the new run C2 with its changed evaluation semantics. Report top-1 and candidate-set
P/R/IoU, boundary error, candidate counts, rejection reasons, masks and overlays for every frame.
Keep >=90% precision/>=50% recall on two targets as a local diagnostic threshold, explicitly
separating top-1 from evaluator-matched results. Neither is production promotion. Stop after
the three runs; no threshold sweep, manual proposal seed, texture-occupancy cue, or SfM restart.

### S3 — Medium: full-cell coverage mixes continuous and discrete areas

`footprint_adapter.py:320` divides inclusive integer raster counts by shoelace area.
Independent control: identity axes, axis Z, side/distance/period=1, focal=1, principal point
(2,3), cell (0,0), a matching visible polygon [(2,2),(3,2),(3,3),(2,3)] in a 6x6 image.
The cell's continuous area is 1, but four raster samples produce **visible coverage=4.0**.
This is significant for small projected cells, not just harmless roundoff. Label coverage
and conflict can exceed 1 and the 0.6 decision threshold no longer means 60% coverage.

Use one consistent area measure for cell/domain/semantic intersections, bounded by the image;
retain uncertainty below resolvable area. Do not merely clamp each label independently to 1.
Test fractional/subpixel, tiny, mixed-label, clipped and large cells; fractions and their
disjoint sum must be bounded and meaningful. Keep the previous half-occlusion/grazing controls.

### S4 — Medium: export verification is still only a coordinate round trip

`footprint_pilot_real.py` reconstructs a shift from the expected cells and applies `block_to_cell`;
its overlay explicitly still uses the in-memory footprint. This accepts the corrected transpose
but does not close the earlier requirement to project actual exported faces independently.
Persist the export-local-to-reconstruction transform, including minimum shift, scale/phase and
normal-axis face placement. Reproject occupied faces from the loaded `.litematic` using that
transform. Unknown/opening overlays can remain separate. Verify asymmetric XYZ sets across all
three axes and both signs with independently expected corners. G3's continuous region still
depends on the assumed plane and stated length; it is phase-independent, not scale-independent.

S3/S4 are bounded B repairs, not prerequisites for testing C's 2D proposer. Complete them in
this assignment, keeping the synthetic and real bundles explicitly developer diagnostics.

## M5 findings and next assignment — Luna

### L1 — High: an earlier representative overrides material refinement

`shape_fit.py:714` appends each geometry's representative to `ranked` before the material-refined
entry at line 800. Default selection minimizes the plain shape score. The refined entry's
shape score is `plain_best`, so it ties the earlier representative whenever refinement does
not improve plain colour. Python's stable `min` selects the representative even when grain
strongly favors the refined material and that material passes the unchanged-scene check.

Independent controlled probe used the existing slab test helpers, aaa_slab RGB120 and bbb_slab
RGB140, one frontal 120x120 view, all pixels usable. An injected PaletteIndex cost of 100 for
aaa and 0 for bbb isolated the selection logic. Diagnostics confirmed bbb was rendered as the
refinement, but the returned state was aaa. This is a selection-unit probe, not a real-texture
quality measurement.

Keep one final decision per geometry with separate shape score, selected material and its own
rendered acceptance score. Preserve air/current baselines; fallback behavior must be explicit
when refinement is unavailable or exhausted. Do not restore the earlier bug where one material's
plain score authorizes another material. Add an equal-geometry grain-disagreement control that
would fail under the current duplicate-entry ordering.

### L2 — Medium: representative choice remains palette-order dependent

`materials[key][0]` still follows input order; only shape keys are sorted. With texture scoring
disabled (also relevant when patches are unavailable), material refinement offers that same
representative only. A photo of a RGB160 bbb_slab returns RGB80 aaa_slab for palette (aaa,bbb),
but returns bbb for (bbb,aaa), using one render in each case with a cap of 8. This was replayed
with the actual renderer/test helpers, without mocked scores. Existing cube-heavy tests do not
permute same-geometry materials and cannot establish the claimed invariance.

Make representative selection reproducible and evidence-aware where available. Preserve an
explicit bounded alternative/fallback when texture evidence is absent; sorting names alone
must not become a claim of material recovery. Run slab/carpet, empty floor and correct-cube
controls under forward/reversed/shuffled cube-heavy palettes, including multiple materials per
shape, rare material, missing patches and a cap ending during multiview refinement. Record actual
renders by phase and budget-unresolved cells. Keep the unchanged cap and fixed usable domain.

After L1/L2 controls pass, run one matched before/after Template 2 diagnostic on a frozen cell
set and identical cameras, masks, assets and settings, explicitly supplied-camera evaluation.
Report exact states, partial-family recall, lost correct cells, additions/deletions, render work
and time. No quality promotion from synthetic render self-comparisons alone. Do not take on
broader GPU/CLI integration while M4/M5 quality remains open.

## Verification and provenance

- Full `.venv` regression: **465 passed in 147.61s** at `e842c0e`, using
  `python -m pytest -q -p no:cacheprovider -o addopts=`.
- Reran RX 9070 XT / wgpu D3D12 scoring: maximum CPU/GPU absolute error
  1.9073486328125e-6; first dispatch 0.10246s, repeated dispatch 0.000828s.
- Reran 37x29 renderer-buffer transfer: colour/depth/UV maximum error zero;
  upload 0.032136s, readback 0.054000s. These are smoke checks, not reconstruction timings.
- Read source, regression tests, saved C targets/results and frame-1 overlay. New independent
  probes ran in memory; no implementation or test files were edited by the lead.
- [M5 submission](M5-shape-scheduling-2026-09-09.md),
  [M4 B submission](M4-G1-G2-G3-fixes-2026-09-09.md),
  [M4 C submission](M4-C-boundary-pilot-2026-09-09.md).

## Communication and maintenance

Sonnet owns S1-S4/C2; Luna owns L1/L2 and matched M5 evidence. Acknowledge in
`AGENT_COMMS.md`, implement the bounded corrections, and report outcomes without asking
for another intermediate approval. Do not interpret this assignment as permission to
require manifests, F3, directions, bounds, or manual geometric constraints in normal mode.
Maintain current handoff sections; keep detailed histories here and in linked agent reports.
