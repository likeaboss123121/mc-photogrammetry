# M4 P1 C2-R-A analytic proof: FORMULA_REJECT

Date: 2026-09-13
Status: `FORMULA_REJECT`; no C2-R-A candidate builder published
Scope: analytic proof only; no production, schema, evaluation, or export wiring

## Result

The first end-to-end physically valid unequal-distance case shows that the existing
`plane_graph._cell_homography` does not preserve the one-cell-per-block signature
coordinates required by C2-R-A. The same fronto-parallel plane is observed at camera
distances 2 and 4 blocks with identical focal length, principal point, axis matrix, and
side. Both signatures explicitly sample one shared world texture at one cell per block.
Their nonzero origins are left `(1,-2)` and right `(4,3)`, so the independently known
zero-translation block-grid map has unit linear part and translation `(3,5)`.

With extent 4, 128 samples, and `b=1`, the retained periods are derived from the same
P1-B relation `distance = b / (period_pixels * 8/127)`. Replaying the actual helper
instead gives normalized homography

```text
[[0.5, 0, 1.0],
 [0, 0.5, 3.5],
 [0, 0, 1.0]]
```

The helper therefore scales already block-normalized cell coordinates by one half.
The existing correlation path admits a match at `(-2,0)` with PSR about `4.606`; the
same actual correlation routine under the independent block-grid map peaks at `(0,1)`.
Thus the right-to-left correlation convention is not the physical one-cell/block map
for this valid camera pair. The added focused test also includes a deliberately wrong
right distance of 3, which creates a distinct `2/3` axis scale.

Independent world translations are left `(0.2,-0.35)` and right `(1.6,0.45)`, giving
true lateral delta `(1.4,0.8)`. P1-B yields residues left `(0.8,0.35)` and right
`(0.4,0.55)`; the known true right-grid integer carry is `(row,column)=(1,1)`. The
accepted P1-D builder returns a complete domain containing that entry, and the proposed
arithmetic returns the correct vector from it. This does not rescue the proof: the
actual helper homography used to define and correlate that entry fails to preserve the
independent block-grid correspondence. A candidate builder could not truthfully assert
the required conditional homography predicate on this valid case. Stop here rather than
publishing a predicate that treats this case as incompatible or silently accepts the
wrong correlation coordinate map.

## Matrix progress and validation

A geometry-only sweep exercised 4×4 absolute quadrant pairs and all 3 common plane
normals (48 cases) with signed-permutation bases and nonzero unequal origins. The
unit-cell signed-permutation predicate passed those equal-distance geometry controls.
The full correlation matrix stopped at the unequal-distance case above, as required
after the first valid architectural counterexample. Both camera sides, all rational
denominators, phase-boundary variants, finite projective controls, and builder records
were not attempted. No C2-R-A source module or candidate builder was created.

The retained test `tests/test_c2r_vector_domain_proof.py` passes: 1 passed. It executes
the real cell-homography, match/correlation, C1 support, P1-D complete-domain, and P1-B
residue paths. It asserts the independent physical map, actual helper mismatch, the
known true carry, and the resulting vector. No accepted source, test, schema, selector,
graph, or artifact file was edited.

P1-D source/test/report and existing protected-file hashes remain as recorded in the
accepted P1-D report and 11:48 communication. This closeout is limited to the proof
fixture and report; it is not implementation approval. Stopped before C2-R-B, V2,
global cycles, CLI, renderer, export, or production wiring.
