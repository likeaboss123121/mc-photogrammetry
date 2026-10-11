# M4 P1-F right-to-left grid-map repair implementation

Date: 2026-09-13  
Status: map repair implemented; stopped for lead review on separate matcher/solver identity blocker  
Contract: [M4-P1-F-right-to-left-grid-map-repair-contract-2026-09-13.md](M4-P1-F-right-to-left-grid-map-repair-contract-2026-09-13.md), SHA-256 `608a2a8afdefeda65a5cc30774f1485ecad1ebf06c81401639f452a339c5c2be`

## Map repair

`cell_grid_homography(left, right, relative_quadrant)` now constructs its linear
part in the consumed right-grid-to-left-grid direction. It iterates right local
tangent labels, transforms each through the relative quadrant into its left
label, and writes the signed value at `[left_index, right_index]`. Translation
remains `P @ origin_R - origin_L`. Reflections are retained. C1's grid-map
digest is now `block-cell-grid-v2` and explicitly records
`mapping_direction=right-grid-to-left-grid`; P1-D uses the same digest helper,
so its stored homography provenance and replay hash change with the corrected
map. Measurement-geometry digest behavior is unchanged.

Before any existing matrix expectation changed, an independent literal absolute
basis test checked all 48 `(left normal, qL, qRel)` cases against
`B_L.T @ B_R`. All 48 world-normal axes were compatible, all 48 matrices were
signed permutations, both determinant signs occurred, and five transpose,
inverse, endpoint-order, sign, and origin-order mutants were rejected. Seven
distinct signed rotation/reflection matrices occurred. The first old/new
counterexample is the retained `n=0, qL=0, qRel=1` fixture:

```text
P1-E:   [[0, 1,  2], [-1, 0,  6], [0, 0, 1]]
P1-F:   [[0, 1,  2], [ 1, 0, -2], [0, 0, 1]]
```

The existing 12-case graph table was updated only after that proof passed. A
separate real-warp/correlation integration test covers representatives of all
seven matrix classes. It builds the expected homography from independent
absolute bases, then checks match shift and correlation statistics, C1 records,
and complete P1-D domains. Every class uses the new shared digest. A test-only
P1-E v1 digest is refused by P1-D. The retained A2 counterexample test now
asserts that repaired H equals the independent physical map; the historical
P1-E report and R5 artifacts were not rewritten.

## Separate identity blocker

The P1-F signatureless-reading regression exposes an existing matcher/consumer
index mismatch when a reading without a signature sorts before matched
readings. With signatureless axis 0 and matched axes 1 and 2, the matcher emits
`(axis,index)=(1,0),(2,1)`. The consumer's all-reading index resolves those
indices to `(0,0),(1,1)`. The regression
`test_leading_signatureless_reading_preserves_match_identity_at_solver_boundary`
fails on that mismatch. No graph, solver, match identity, or public interface
change was made; the P1-F contract requires a separate identity contract before
that behavior can be repaired.

## Validation and protected artifacts

The combined focused reconstruction suite completed successfully before the
new leading-signatureless regression was added: **303 passed**, in about 98
seconds. It covered graph, C1, P1-D, P1-B, C2-A/A2, lattice, selector, supported,
and frozen-runner tests, plus both P1-F theorem/integration files. The new
identity regression was then run separately and failed as described above.
`py_compile` passed for the changed sources and tests. No source change followed
the 303-test run; only the failing regression and this closeout record were
added. P1-F is therefore stopped for the separate identity review rather than
claimed fully validated.

Changed source/test SHA-256 values:

- `plane_graph.py`: `5f7a3be7755c42834ba5119974c099e1e7638b12bd2b9f40b2e3281867871af9`
- `plane_axis_support.py`: `18598afb2096b504464411e3a2cbf6a2e960cb2601c4dd2fb8f77d4b46c91f3f`
- `plane_shift_domain.py` (unchanged, recomputes the shared C1 map digest): `2e3ef428fe4600b3c7dd676f2369549e9ef5cdb1616a3f8cee66b623a341facc`
- `test_plane_graph.py`: `b4715b20128970976d696edb001334c1bc91a1dab26d339ba62139028fc1f732`
- `test_c2r_vector_domain_matrix.py`: `76023bf15dbeee67b20138fad3428797e7ee0416df5167e2f78c3ea0767a15d8`
- `test_plane_grid_map_p1f_proof.py`: `ef88e92dc09115988d812ed11c12c1d784d8e86bf8d7ea9ea193ee6759e08271`
- `test_plane_grid_map_p1f_integration.py`: `f2b7ab54d0c1afde37ee42ca464347bee4e64cca3aec81a1f2affbb4c59a7bc2`

Protected historical hashes still match their published records: R5 freeze
`8dfc8b41bad1dff018fa08db42411c5f91572233cb1e7718cc17a082ab0c218b`, R5 mask
sidecar `087d379673be1ff291f807753cdb00206ad02a64aa26f91104d24abc36a8db5b`, R5
diff `9731923b092d28a0d52453a44fadcb8875dd94974c2b4b80339c7cbbe5653089`, R4
freeze `96493dd9f42f1d20c932a8b09f267d7f86da38b7e5ac588dec1330ad9df81c9d`, and
R4 sidecar `f28cc68b2fc5086473ba3b73c4b277e0d1338c10639c0c6a4b64e94585be0bac`.
No real fixture run, evaluation, solver/threshold/schema/CLI/renderer/export
change, R6, C2-R builder, or V2 work was performed.
