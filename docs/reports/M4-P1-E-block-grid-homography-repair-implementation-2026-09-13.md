# M4 P1-E block-grid homography repair implementation

Date: 2026-09-13  
Status: implementation complete; stopped for lead review  
Scope: quadrant-aware signature-grid mapping and C1/P1-D provenance migration only

## Result

Replaced the private camera-composed map with the public pure helper
`cell_grid_homography(left_signature, right_signature, relative_quadrant)`. It checks
strict quadrant and normal compatibility, finite non-empty matched 2D signature grids,
finite origin pairs, and a bijective tangent transform. It builds the exact signed
permutation from `axis_transform` and translation `P @ origin_R - origin_L`; camera
distance, focal length, principal point, rectification centre, sampling, and side do
not scale or rotate an already one-cell-per-block grid. `match_plane_pair`, C1, and P1-D
now pass the exact selected relative quadrant. The helper is exported from the
reconstruction package; no production caller uses the former two-argument helper.

C1's geometry digest still binds selected b/distance, camera axes, focal, principal
point, rectification centre, side, origins, and shape; it now also binds period,
extent, and sampling count as measurement provenance. A separate versioned grid-map
digest binds the ordered frame/plane identities, canonical b values, both shapes and
origins, axes, relative quadrant, and exact normalized signed-permutation/origin
matrix. P1-D recomputes that same digest and the corrected warp/correlation replay.
PlaneMatch fields, graph ranking and admission, selector, P1-B residue arithmetic,
P1-D domain enumeration, schemas, and artifact contents are unchanged.

## Decisive repair and changed expectation

The retained physically valid fronto-parallel case has equal focal/principal/axis
matrix/side, distances 2 and 4, left origin `(1,-2)`, and right origin `(4,3)`. The
corrected map is `[[1,0,3],[0,1,5],[0,0,1]]`. The named test-only legacy camera
composition remains `[[0.5,0,1],[0,0.5,3.5],[0,0,1]]`; a deliberately wrong right
distance still gives its distinct legacy `2/3` scale. Thus the old graph peak
`(-2,0)` (historical PSR about 4.606) changes to `(0,1)`, and the corrected result
matches the independently constructed one-cell/block warp: peak `0.49149419`, PSR
`17.15485192`, second peak `0.17503089`, fixed overlap `1935`, score `0.92877601`.
P1-D still retains the lower-ranked true carry `(1,1)`; the top peak does not remove
the integer-lift ambiguity. No other graph or selector expectation changed: the full
existing graph/selector regression set passed without expectation updates.

Literal expected P matrices cover all 12 compatible pairs of three left normals and
four relative quadrants. Additional checks cover asymmetric fractional origins,
pointwise map and inverse, camera-parameter byte invariance with provenance changes,
invalid quadrants/axes/origins/grids, non-square/holey support, P1-D replay, and the
helper's AST/import boundary. The C2-R-A rejection report remains unchanged as
historical evidence; its retained test now passes as the repair regression.

## Validation

Combined graph/C1/P1-D/C2-A/C2R-A/P1-B/lattice/selector/supported/frozen-runner
regressions: **346 passed in 111.93s**. The retained C2R-A repair/AST checks passed
again after the final comment edit: 2 passed. `py_compile`, owned-file whitespace and
final-newline checks, and `git diff --check` passed.

The R4 freeze was loaded and structurally verified without re-running its pipeline:
artifact ID `7d0a08f4a4eb09ae8b5e4568356b515a1f186194e2d51eda2b037a244b9f1ec0`,
200 saved matches, freeze SHA-256
`96493dd9f42f1d20c932a8b09f267d7f86da38b7e5ac588dec1330ad9df81c9d`. It cannot be an
exact source replay under this migration: the saved `plane_graph.py` code hash is
`3e330b5358542d618a42b7c35b105536b858f38791ab48350f45837afae2688a`; the corrected
source hash is `c0326d9a300e77aee68ed835e6fc035e397560b7b0d95709edb5f8b65b687ce6`.
The freeze remains immutable historical evidence.

P1-E source/test SHA-256 values:

- `plane_graph.py`: `c0326d9a300e77aee68ed835e6fc035e397560b7b0d95709edb5f8b65b687ce6`
- `plane_axis_support.py`: `52813651d2b75b0e84d975758250763d7e3b33d2a372e1655bee03530bb828df`
- `plane_shift_domain.py`: `2e3ef428fe4600b3c7dd676f2369549e9ef5cdb1616a3f8cee66b623a341facc`
- reconstruction `__init__.py`: `7f80f67bfe5ce8f93a4cf140f093e5d23458b6b194814a2c0a8d2973f99c5d09`
- `test_plane_graph.py`: `b7cbb89bf8d7a2cbd2e4c6741084818a89b7916aa33de71451cec5a185d97258`
- `test_plane_shift_domain.py`: `77c5f0e510a03213195abd0b5401ef284af9ca8cff4a3189ecc18c74d42d624f`
- `test_plane_c2_proof.py`: `0553882a6fc38328d20b9820de635a2fde426d534a38c0febad5302f1e906a2e`
- `test_c2r_vector_domain_proof.py`: `c2e47f092894c7e2ee14940d882e62392a219456a28526fdc363db577b17897c`
- unequal-distance fixture grid-map/C1/P1-D digest: `fe25ba31778b7289aa1fe1ec9062d59675223040e0a4c481f71f2e1f711bc2df`

Unchanged P1-B/C1 regression hashes include `plane_residue.py`
`55ec0cad8fe1088b7b33b1f57c4d58d3d3b912f79765a1f391553ac7b845be0d`,
`_joint_plane_selector.py`
`f292f08d1b14a5cadbcb395bcc25ce6904a16b353ee6244e5b955de1aed99b5c`,
`test_plane_axis_support.py`
`4b2614eb1b583fa0e8039acf07425b64cd96050c63c512f5179fc8173b5ef988`,
`test_plane_residue.py`
`4d7d00cd1b4d31ec70d964ee153bf13e648f8325e8b284f0e46a130eec43a36f`, and
`test_joint_plane_selector.py`
`c6a55ef7986233c0fa2047e712e3b650ac217e339603b5ee9d7c97ec85a0f0a0`.

For protected-surface verification, current workspace hashes at this closeout are:

- dataset schema `schemas.py`: `aa89ba78ff5cba4c276d3c0404ffe3739eadf5b7a4e81fe0dc56359b270098a3`
- frozen evaluator `evaluation_frozen_planes.py`: `3918388075dcfab7b9cde0788bbbe930df7ef295041a6871bfd045a0594aee91c`
- renderer buffers `render_buffers.py`: `ee7e4b47d4937689fc2761f03031cb8e9912087f94bcc62ef5bed8adff48cd8b`
- CLI `cli.py`: `d2742e9771ed56852dd2d0a87406227f9bc643326b88c61d61963cbd93495ac8`
- litematic exporter `litematic.py`: `1917451e46ccb451dcbf2e5194161edfbea0862dc55e62489fcdeeb14f3db794`
- export package `__init__.py`: `259fac03dcbb676b1f518aa515d7f38b26efd499071c0f29b0db232fb07e1b3e`

P1-E did not modify those schema, evaluator, renderer, CLI, export, or artifact paths.
Some are pre-existing shared-workspace changes; the hashes above identify the exact
workspace snapshot for lead review. No R5, evaluation, C2-R/V2, CLI, renderer, or
export work was run or wired. Stopped for lead review.
