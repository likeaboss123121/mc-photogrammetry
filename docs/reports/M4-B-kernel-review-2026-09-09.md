# M4 B replacement review: placement accepted, image adapter next

GPT-6 Astra, 2026-09-09. Reviewed `6a26e63` at HEAD `f089a4a`.
Lead work: read code/reports, run tests and independent probes; documentation changes only.

## Decision and verification

**Accept the declared-cell placement component.** It separates visible/opening/unknown labels,
places the solid on the correct side of a face, and supports a real litematic round trip.
This resolves the previous mistake of routing the control through the seam occupancy classifier.

**B remains incomplete at the image-to-grid boundary; C remains planned.** Authorize Sonnet to finish
the synthetic image-polygon adapter and, once the checks below pass, run **one corrected real-image
pilot without another intervening approval request**. Return for review with both results before C.
Luna continues the user's R2/R4 assignment; do not pull Luna into M4 or reopen accepted palette gates.

- Lead command: `.\.venv\Scripts\python.exe -m pytest tests/test_footprint.py tests/test_lattice.py tests/test_camera.py -q -p no:cacheprovider -o addopts=`.
  **55 passed in 6.03 seconds**, including the six new footprint tests. Sonnet's full-suite count of
  410 is agent-reported; the lead did not repeat a full suite for this small isolated addition.
- Independent arithmetic probe checked all three axes and both outward-normal signs, including a
  negative lateral origin. The face-to-solid offset was correct in all six cases.
- Independent temporary export/readback recovered **10 non-air blocks**. Opening/unknown both read as
  absent solid geometry, so the distinction must survive in the evidence sidecar, as Sonnet explained.
- Metadata defect: `Metadata.TotalBlocks` is **11**, although only **10** non-air blocks exist. The
  footprint includes an explicit air observation and the exporter counts volume entries. This is an
  existing exporter assumption exposed by the new caller. It does not invalidate solid placement.

## What the current tests establish, and what remains

`build_footprint` takes already-indexed `dict[(row,column),label]` input. That is a useful final-stage
representation. It does not yet map an image-drawn polygon into those cells or calculate partial
coverage. The lack of an image parameter proves texture independence for this component; it does not
prove correctness of the missing image adapter that will supply its labels.

The `lateral_origin=(100,-7)` test checks integer translation. It is **not** a nonzero rectification-window
or fractional lattice-phase test: no camera intrinsics, pixel coordinates, rectification window or
fractional phase enter the API. Keep that distinction explicit in the report and handoff.

No persistent control `.litematic`, evidence JSON or source-image cell overlay was delivered by this
commit; its round trip is in a temporary test directory. There is also no saved transform from which a
separate process could reconstruct the source-pixel/cell correspondence. These remain B deliverables.

## Sonnet's next bounded assignment

1. **Reuse the accepted component.** Add a thin developer-only image-polygon-to-footprint adapter;
   do not replace it or add a new occupancy detector. Inputs are image-space semantic polygons,
   camera intrinsics/pose in a declared local frame, a plane, and one lattice origin/scale/phase.
   Record which inputs are image-derived versus declared diagnostic constraints.
2. **Make the geometric mapping explicit and serializable.** Intersect source-camera rays with the
   declared plane, map into its two in-plane axes, then into the one block lattice. Reuse existing
   camera/lattice helpers. Keep row/column versus u/v conventions explicit. Reject parallel/behind-camera
   intersections; preserve unresolved scale/phase. Do not silently choose phase zero or refit phase per cell.
   A nonzero image/rectification-window origin must change sample coordinates without shifting the physical grid.
3. **Rasterize semantic regions against cells.** Preserve visible coverage and separate confirmed opening,
   foreground occlusion and unobserved pixels. Partially covered/boundary cells retain their coverage and
   uncertainty; a mask's bounding rectangle cannot create support. Unknown is not an opening. Texture
   statistics cannot veto a declared visible surface or create depth evidence in this developer control.
4. **Close the synthetic image loop.** Project the existing 4x3 control through a known synthetic camera
   to generate image-space regions, then recover the expected 10 supported/1 opening/1 unknown from those
   regions. Check exact block-coordinate sets and corner reprojection, including a fractional rectified
   grid offset, nonzero window centre, image clipping and both face signs. Change synthetic textures while
   keeping the polygons fixed; recovered cells must be unchanged. Do not tune expected counts to the output.
5. **Save a reviewable bundle.** Write a diagnostic `.litematic`, semantic evidence/transform JSON and
   source-image overlay. Reload the JSON and export in a separate invocation; recover the same non-air
   coordinate set and unknown/opening labels. Include settings and hashes. Correct the 11-versus-10 block
   count: preferably emit supported solids only from this control and keep confirmed openings in the sidecar.
   If changing shared exporter behaviour instead, coordinate that named file in comms first.
6. **Then one real-image pilot, conditional on the synthetic checks passing.** Select one clearly visible
   plane patch in Template 2, with verifiable source-image landmark/interval evidence. Correct the old
   annotation mistakes; do not reuse an acknowledged wrong length. Exclude HUD/F3 from feature/mask
   evidence, freeze revised annotations with hashes before scoring, and use image-derived camera optics.
   Optional manual plane/scale/phase constraints remain developer-only. Arbitrary local origin is fine;
   manifest coordinates or F3 values must not supply the solution. If a defensible phase/length cannot be
   established, report that ambiguity with the mapped continuous region rather than inventing block indices.
7. **Stop after this bundle and one pilot.** Provide mask/cell overlay, coverage/unresolved counts,
   reprojection checks, export readback and input provenance. Any reference-based scoring happens after
   outputs are frozen and saves its alignment/correspondences. No other fixture sweep, detector experiment
   or material tuning is part of this assignment. C follows lead review of this complete B path.

A1's seam-to-geometry verdict defect remains documented and quarantined from this adapter. Fixing that
old classifier is not a prerequisite for this assignment; never consume its hole/occlusion labels here.

## Note on the later theoretical-max benchmark message

The posted T2/T6 percentages are explicitly supplied-camera diagnostics, and are not M4/B validation.
Scoring against a visibility-only litematic changes the reference set: correct hidden blocks omitted
from that subset may count as phantoms, and visibility is then evaluated on an incomplete scene.
Keep those numbers labelled as subset-reference diagnostics. They do not establish that geometry is
sound or isolate material/lighting as the only cause of error. The full-reference visible metrics and
production gates in the handoff remain authoritative. Link exact output artifacts for future comparisons;
the comms message alone is not a reproducible matched benchmark report.
