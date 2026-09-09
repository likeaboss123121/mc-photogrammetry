# Lead review: M4 A/B and M5 corrections

GPT-6 Astra, 2026-09-08. Reviewed HEAD `b22cc19`, including M4 `5e84345`/`b22cc19`
and M5 `1a643f5`/`23175d6`/`42212e0`/`805acec`. Documentation changes only by the lead.

## Decisions

| Submission | Decision |
|---|---|
| M4-A | Accept original three bug fixes and mapping helpers; evidence/geometry contract remains incomplete. |
| M4-B | Not accepted: it still tests seam occupancy instead of independently validating mask-to-cell construction. |
| M4-C | Do not start yet. Complete the small replacement B control below, then return for review. |
| M5-R1 | Accept exact normalized-state resolution and unavailable missing-variant behaviour. |
| M5-R3 | Accept reviewed CLI manifest gates and normal desktop sidecar isolation; this does not complete automatic recovery or persistence. |
| M5-R2/R4/R5 | Partial implementation; concrete correctness and work-limit defects remain. M5 is open. |

Focused suite: **111 passed in 5.94 seconds** across supported/lattice/camera, palette persistence,
shape fitting and CLI tests. Full suite: **404 passed in 150.71 seconds**, using
`.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider -o addopts=`.
Additional in-memory probes below deliberately cover cases absent from the passing suite.
No real-fixture reconstruction sweep was run. Existing reports/artifacts and annotation overlays were inspected.

## M4: why A/B cannot yet support C

### A1 — Texture disagreement still becomes a geometric contradiction

`supported.py:400` initializes every remaining low-seam-score case to `CONTRADICTED`, even when no
period is detected. Random texture on a single flat plane, generated with seed 123 over 128x128 pixels,
period 16 and full coverage, returns **54 contradicted / 10 foreground_occlusion / zero unknown**.
There is no independent geometric evidence of a hole or foreground object in those input arrays.

The blank-wall fix is accepted, but it fixes one texture case. Fine or coarse texture can occur on
the same plane; texture frequency alone does not establish farther/nearer geometry. The new synthetic
"opening" and "foreground slab" controls replace textures with different-frequency bands without
independent 3D surfaces or depth labels. They establish the heuristic's response to texture frequency,
not the claimed geometric interpretation. Their expected outputs were also adjusted to the heuristic.

**Correction:** keep seam/period agreement as diagnostic statistics. Require explicit positive depth,
boundary/visibility evidence or a developer control label for contradiction/occlusion. Otherwise return
unknown. Do not introduce another frequency threshold. A supplied visible-plane mask is positive
surface evidence in the assisted control and must not be vetoed by texture disagreement.

### A2 — Mapping helpers exist, but the result still loses its mapping and phase status

`SupportedPlane` fields remain `axis, distance, phase, side, cells, assisted, notes`. Window centre,
extent, sample count, plane basis, source intrinsics and a lattice origin/phase-resolution status are
not retained. `support_cells` starts cells at `row*period`/`column*period`; phase affects scoring, not
their physical boundaries. `plane_from_evidence` still uses `2*span/samples` while the mapping helpers
follow the endpoint-inclusive grid. A helper round-trip does not validate the exported cell locations.

**Correction:** return/persist one invertible plane/lattice transform, including pixel-spacing convention
and unresolved scale/phase. Establish one shared phase or retain hypotheses. Unknown phase must remain
unknown instead of becoming a numeric zero. Test nonzero-window/nonzero-phase cell corners and their
3D/source-image coordinates, not merely whether locally refitted texture scores are high.

### B1 — The assigned control has not been implemented

`mask_to_cell_control.py` builds `visible_mask`, passes it to the same `plane_from_evidence` seam
classifier and selects axis/side by the number of seam-supported cells. It then writes cell-index
strings such as `row,column` to JSON. Thus the mask only restricts where a detector runs; it is not
the evidence from which supported surface cells are constructed.

Verified outputs: geometry counts **0, 0, 9**; the last file contains two `neutral_solid` cells and
seven `inferred_opening` entries. There are **zero `.litematic` files** in the output directory.
Neither result JSON nor its evidence sidecar contains a persisted image/plane/cell transform.
An array of verdict colours is not the requested cell-boundary overlay on the source photograph.

**Correction:** B must convert the supplied footprint and a declared lattice into cells without calling
the seam occupancy classifier. Neutral blocks are only the geometry control's material. Export them
with the existing litematic exporter and verify their coordinates by reading the file back.

### B2 — Reported scores and conclusions are not reviewable evidence

The maintained control/overlay scripts contain no precision/recall calculation or reference geometry
scorer. Geometry JSON records the reference camera eye but no scored correspondences, counts or
alignment transform. The report's 22/3/0 truth counts therefore lack a reproducible saved scorer here.

The annotations acknowledge a wrong stated length; the overlay's points do not establish the stated
number of block intervals. Wrong scale can absolutely create false negatives in a valid geometric
method. The report's assertion that it cannot is incorrect. The third view again says "upward" despite
looking downward. The second target is the left side wall, although the assignment suggested the far
wall. Target changes are allowed when justified, but must be identified accurately and fixed before scoring.
The second mask also includes the debug overlay; exclude it from any image-derived feature evidence.

Finally, **90% region precision / 50% recall on two targets is C's local automatic-boundary continuation
criterion**, not a requirement that B's old seam detector pass before a mask control counts as implemented.
It remains correct to pause now because the independent B control is missing. The failed run establishes
neither success of that control nor general impossibility of image-derived geometry.

## Sonnet: replacement B, one small deliverable at a time

1. Implement a developer-only footprint-to-lattice mapper. Inputs: declared plane/basis/side, one lattice
   origin and scale/phase, and separately labelled visible-surface, opening, foreground-occlusion and
   outside-image regions. A boolean valid-pixel mask cannot preserve all four semantic labels.
   Unknown regions stay unresolved. Do not route supplied surface support through `support_cells`.
2. First independent synthetic control: a 4x3-cell planar patch, one explicitly known 1x1 opening and
   a different 1x1 foreground-occluded cell. Expected result: **10 supported cells, 1 opening, 1 unknown**.
   Use a declared outer face such as plane z=1, outward normal +Z, with solids occupying cell z=0.
   Check the inward face-to-cell offset, irregular footprint, source overlay and exact litematic round trip.
   Repeat with a nonzero rectification-window offset and phase; physical cells must remain consistent.
   Swapping uniform, noisy and banded textures must not change these supplied-mask control decisions.
3. Publish the transform contract, passing synthetic evidence and exported artifact in comms **before
   another real-image run**. This is the next review boundary. No new seam/frequency detector or threshold.
4. After synthetic acceptance, correct the image annotations with explicit visual landmark/interval
   evidence and freeze a version/hash before scoring. Begin with one pilot target, not three bad inputs.
   Show phase/scale ambiguity when landmarks do not establish them; do not invent a precise block count.
   Validate footprint mapping independently of eventual material recovery. Reference scoring happens only
   after freezing outputs and records the one allowed alignment and per-cell correspondences.
5. C remains planned. Once B is accepted, return to the existing single-configuration, capped boundary
   experiment; do not substitute another per-cell appearance test. The original production gates stay fixed.

Preserve old measurements with a superseded-method notice; correct their unsupported conclusions rather
than silently deleting history. This assignment supersedes the previous six-run allocation until the
synthetic control is accepted. No additional real-image run is needed for this next deliverable.

## M5: accepted repairs and remaining defects

R1 exact lookup no longer relabels arbitrary same-name geometry. Focused mismatch/default tests pass.
R3 rejects normal-mode camera/scene arguments before loading inputs. An independent patched desktop
probe in `templates 2` reaches the normal image path without calling camera/scene loaders. Existing
developer access remains explicit. Full project-mode provenance/persistence remains future integration.

### R2 — Air still has a different photometric model; removal cannot reach the caller

`shape_fit.py` compares raw `context_backgrounds` against the photo for air, while candidates/current
are multiplied by the estimated exposure gain. Replay the existing empty stone-floor control with the
same photograph multiplied by **0.5 or 1.5**: both runs insert an oak slab; gain 1.0 correctly stays empty.
All three runs use scale=1, edge_weight=0, texture_weight=0, with the same geometry and palette.

Air winning for an occupied cell returns no edit (`{}`); the caller consequently retains the incorrect
block. There is no explicit deletion result. Further, a shape's `plain_best` can belong to a different
material from `joint_state`; its favourable score must not authorize an unverified selected material.

**Next:** apply one gain/lighting/mask convention to every hypothesis. Add an explicit accepted-delete
representation and integrate removal into the volume. Re-score the actual selected state before accepting
it. Preserve cells on ties/insufficient evidence. Replay exposure-scaled empty floor, removal of an extra
slab, real slab recovery and correct-cube retention. Coordinate accepted edits' occlusion context.

### R4 — Fully excluded images still create blocks

With a rendered slab image and `usable_masks={'a': zeros((120,120),bool)}`, `fit_states` still inserts
the slab. The mask is applied to texture extraction only; exposure estimation, cube-defined scoring
regions and geometric RGB/gradient comparison ignore it. An image with no usable pixels must abstain.

Also, `_face_patches` only intersects ownership when `owner_masks.get(slot)` is non-null. A slot absent
from an explicitly supplied ownership map should have zero evidence, not default to full validity.
One representative per box-shape group cannot establish identical alpha coverage for all materials.

**Next:** construct one validated usable domain before exposure/scoring, carry it through every path,
mask gradients without invalid-neighbour leakage, and treat missing owned faces as invisible. Test that
changes to excluded pixels leave geometry, exposure and material unchanged; test all-excluded abstention.
Retain actual supporting frame IDs instead of assigning every input frame to each accepted block.

### R5 — The declared cap is exceeded across views and has no default bound

With three identical usable views and `max_candidate_renders=1`, diagnostics report **3 candidate
renders**. The check runs before the state loop's view iteration instead of before each render.
Initial background and current-state renders are not fully counted. CLI defaults the cap to `None`;
the complete palette still renders each material separately. No bounded real quality/performance report
accompanies these commits. Instrumentation is useful, but R5 is not complete.

**Next:** enforce the exact cap before every candidate render and define how interrupted hypotheses
remain unaccepted rather than using a selectively scored subset. Count all render work; establish a
finite conservative diagnostic default or require an explicit budget for complete-palette runs. Validate
0/1/N caps with multiple views, then report a tiny holdout's quality, counts and elapsed time. Do not run
the full fixture/complete-palette sweep while these controls fail.

Luna's next patch should address the shared photometric/usable-domain defects in R2/R4 first, followed
by explicit deletion and selected-state acceptance, then the strict work cap. Keep Sonnet's M4 paths
untouched. Neither M5 nor the automatic reconstruction pipeline is complete.
