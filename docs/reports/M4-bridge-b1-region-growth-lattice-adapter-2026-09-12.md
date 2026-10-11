# M4 Bridge B1: image-only region-proposal adapter (2026-09-12)

## Change

Added public `frame_plane_readings_from_proposals` in `reconstruction.lattice_planes` and exported it from the reconstruction package. The new call accepts only an RGB frame, caller-derived usable mask, image-derived `CalibrationResult`, typed `PlaneProposalResult`, frame id, and the same reader limits as the legacy API. It copies each proposal mask through `mask > 0.5 AND usable`, rejects malformed masks/polygons/calibration with the existing `ReadingAbstention` reasons, and canonicalizes proposal order spatially for input-order-independent results. Proposal metadata `side` is ignored; `lattice.plane_window` derives the geometric side from the usable mask and image calibration.

Both public entry points now call one shared measurement/ranking/signature-context loop. The legacy `frame_plane_readings` parameter list is unchanged, and the adapter parity control feeds it the same masks to verify identical reading outputs. Neither path changes thresholds. The adapter does not pre-deduplicate same-axis/same-side proposals: distinct overlapping regions are each measured. The existing post-measure global `max_readings` and `MAX_READINGS_PER_AXIS_SIDE == 2` selection caps still apply.

There is no GT, capture, manifest, reference, or supplied camera pose in the adapter. Proposal axes are the C4 image-derived labels; proposal-side metadata is ignored and `plane_window` recomputes the geometric side. A source poison test confirms the new API is not called from the default CLI or pipeline. The legacy default route remains unchanged.

## Synthetic evidence

A procedural two-view room was passed through C4 `propose_image_planes_v2`, the new adapter, `plane_signature`, and the plane graph. Both views returned 4 proposals and 4 readings; the graph received 8 signatures, produced 6 matches, then honestly abstained with `ambiguous_quadrant`. The synthetic control verifies bridge data flow without claiming identity resolution. Additional tests cover legacy arity/output parity, two overlapping same-axis/same-derived-side regions retained independently, reversed proposal order, mask/image poison outside `usable`, and typed abstention for empty, degenerate, dimension-mismatched, and malformed numeric/source inputs.

Focused command: `.venv\\Scripts\\python.exe -m pytest tests/test_lattice_planes.py tests/test_plane_graph.py tests/test_region_growth.py -q -o addopts=''` -> **41 passed** in 33.43 s. Ruff was unavailable at `.venv\\Scripts\\ruff.exe`; pytest imported and exercised the changed modules. No real capture, label, C4 score, default CLI promotion, full suite, stage, or commit was performed.

## Files and hashes

- `src/mc_photogrammetry/reconstruction/lattice_planes.py` SHA-256 `9e79617060e184f470c0af659d23886486b87bc149a329729091d576e2519ad7`
- `src/mc_photogrammetry/reconstruction/__init__.py` SHA-256 `109dc5f06a4494ce9fee166f5b871ba759191fb5e9fc47d76104b3144e541`
- `tests/test_lattice_planes.py` SHA-256 `df8f310c4f36516c94e083a249b823f1f493e0697a29fbd82b1dd23a3ab2950f`
- `tests/test_plane_graph.py` SHA-256 `98efac6f44c85171b1b3155cf8707a487513da89fda08f3cd0225111cdd22e97`
