# M4 P1 C2-R-A2 post-P1-E result: `FORMULA_REJECT`

Date: 2026-09-13  
Owner: Luna  
Disposition: stopped before domain-builder implementation, as required by the
13:38:45 lead directive.

## Decisive independent orientation counterexample

The corrected proof fixes world translations before generating either phase or
signature observations. It uses a common nonperiodic world texture, nonzero
unequal signature origins `left=(3,-2)` and `right=(-4,5)`, and passes signatures
through `plane_signature` so the actual endpoint-inclusive origins and coverage
are exercised.

For the valid compatible-plane case `left normal=0`, `left quadrant=0`,
`relative quadrant=1`, the literal absolute transform table gives
`right normal=2`, `right quadrant=1`. The endpoints share world normal 0 with
opposite normal signs. On their sorted world tangents `(1,2)`, the independently
composed local bases are
`B_left=((1,0),(0,1))` and `B_right=((0,1),(1,0))`. Therefore the physical
right-to-left block-grid map is `B_left.T @ B_right` with the origin translation:

```text
independent physical H = [[ 0, 1,  2],
                          [ 1, 0, -2],
                          [ 0, 0,  1]]
actual P1-E H          = [[ 0, 1,  2],
                          [-1, 0,  6],
                          [ 0, 0,  1]]
```

The helper matrix is finite and affine, but it disagrees with the independently
composed map. The independently declared endpoint translations are
`T_left=(0.2,-0.35)` and `T_right=(1.6,0.45)`, so the physical truth is
`T_right-T_left=(1.4,0.8)`. This valid map mismatch occurs before any candidate
builder is involved and blocks the required C2-R-A2 truth-membership proof. The
fixture is retained in `tests/test_c2r_vector_domain_matrix.py`.

The lead diagnosis is that `cell_grid_homography` currently iterates left
tangent labels and applies the relative transform, while the returned map is
consumed right-grid to left-grid. The likely repair is to iterate right tangent
labels and transform them into left/world labels. That repair is outside this
assignment; no reconstruction source file was changed.

## Independent smoke that passed

The q0 control fixes the same world translations before phase generation. Its
forward phase observation model derives each endpoint's local residue from its
independent local offset, then emits a phase modulo the supported rational
period. The real P1-B factory retains the known residues. The test then uses the
actual signature, P1-E homography, warp/correlation, C1 support, and complete
P1-D builder.

Observed result: correlation peak `(0,1)`; P1-D retained all 81 shifts in the
declared radius-four domain; complete entry `(1,1)` is the only entry whose
enumerated residue pairs produce the independent truth vector `(1.4,0.8)`.
The proof enumerated 162 axis-residue pairs. This validates the forward
single-view phase construction and q0 end-to-end path only; it does not offset
the orientation counterexample.

## Development history and limits

An early scratch harness computed endpoint translations from chosen residues
and a chosen carry. That made its candidate membership tautological, so its
output was discarded and is not evidence. The persisted test now declares
translations first and derives observations forward. The broader matrix was
stopped at the first valid P1-E map inconsistency, before mutant census or
remaining orientations. No vector-domain module or builder was created, and no
R5/evaluator/ground-truth data was used.

Focused validation: `pytest -q tests/test_c2r_vector_domain_matrix.py
tests/test_c2r_vector_domain_proof.py` passed, 3 tests. The broader focused
P1-B/C1/P1-D/graph/selector suite passed all 210 collected tests with exit code
0. `pytest --collect-only` independently counted 43 residue, 43 C1, 19 P1-D,
43 graph, 59 selector, 2 matrix, and 1 retained-proof tests.

## Inputs and protected hashes

```text
amended contract SHA-256: baec9047e18e9ba43e0c7b316269f2099c0186491232f1d9a9f1812005a48319
historical FORMULA_REJECT report SHA-256: 513b37a69b738e8a9f297b45ff53d4b5651a9fc93a61bc789797b92fb97410eb
P1-E source plane_graph.py SHA-256: c0326d9a300e77aee68ed835e6fc035e397560b7b0d95709edb5f8b65b687ce6
P1-E test_plane_graph.py SHA-256: b7cbb89bf8d7a2cbd2e4c6741084818a89b7916aa33de71451cec5a185d97258
P1-B test_plane_residue.py SHA-256: 4d7d00cd1b4d31ec70d964ee153bf13e648f8325e8b284f0e46a130eec43a36f
C1 test_plane_axis_support.py SHA-256: 4b2614eb1b583fa0e8039acf07425b64cd96050c63c512f5179fc8173b5ef988
P1-D test_plane_shift_domain.py SHA-256: 77c5f0e510a03213195abd0b5401ef284af9ca8cff4a3189ecc18c74d42d624f
new proof test SHA-256: f0c0260137ab6b7c118ec2a818001c1d85e1056357d7db1bc00b3a7b3132c752
retained unequal-distance proof test SHA-256: c2e47f092894c7e2ee14940d882e62392a219456a28526fdc363db577b17897c
```

No change was made to `plane_graph.py`, P1-B/C1/P1-D sources, or the historical
rejection report. The proof test and this report are the only task artifacts.
