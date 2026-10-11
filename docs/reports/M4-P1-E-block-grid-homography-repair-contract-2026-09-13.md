# M4 P1-E block-grid homography repair contract

Date: 2026-09-13  
Status: approved for isolated implementation  
Dependency: C2-R-A retained `FORMULA_REJECT`

## Defect and required result

`PlaneSignature.cells` is already sampled at one cell per Minecraft block. The current
`plane_graph._cell_homography` composes two camera-image homographies and therefore
rescales these normalized grids when the same physical plane is viewed from unequal
camera distances. The retained C2-R-A fixture gives scale `0.5` for correct distances
2 and 4, changes the correlation peak, and prevents an honest physical-coordinate
predicate.

P1-E must replace that map with the exact right-grid-index to left-grid-index signed
permutation implied by the admitted relative quadrant and the two signature origins.
Camera distance, focal length, principal point, rectification centre, and camera side
must not affect this grid map. They remain valid reading/normal-distance provenance and
must not be removed from their existing records.

This is a correctness migration. Changed match evidence and changed C1/P1-D homography
digests are expected and must be measured. Do not preserve an incorrect numeric map for
compatibility.

## Exact transform

Expose one shared pure function:

```text
cell_grid_homography(left_signature, right_signature, relative_quadrant) -> 3x3 H
```

It maps target/right cell indices `(column_R,row_R,1)` to source/left cell indices
`(column_L,row_L,1)` for `warp_signature`.

Let:

```text
L = sorted({0,1,2} - {left.reading.axis})
R = sorted({0,1,2} - {right.reading.axis})
o_L = (left.origin_u, left.origin_v)
o_R = (right.origin_u, right.origin_v)
```

First require:

```text
axis_transform(left.reading.axis, relative_quadrant).world_axis
    == right.reading.axis
```

Build a zero `2x2` matrix `P`. For each left tangent position `i` and label `L[i]`:

```text
(right_label, sign) = axis_transform(L[i], relative_quadrant)
j = the unique index with R[j] == right_label
P[i,j] = sign
```

Every row and column must contain exactly one value in `{-1,+1}` and all other values
must be zero. Then:

```text
x_L = P * (x_R + o_R) - o_L
H = [[P00, P01, (P*o_R-o_L)[0]],
     [P10, P11, (P*o_R-o_L)[1]],
     [0,   0,   1]]
```

For relative quadrant zero and equal plane-axis labels this reduces to identity plus
`o_R-o_L`. The matrix is finite, affine, unit-cell preserving, exactly invertible, and
contains no camera projection scale. Fractional origins remain fractional; do not round
or absorb them into the later integer shift.

`relative_quadrant` is a strict integer in `0..3`; booleans and modulo coercion are
invalid. Signature arrays must be valid 2D grids, origins finite pairs, axes strict
integers, and the normal/tangent mapping bijective. Invalid input raises only the
documented value error internally and becomes the caller's existing typed refusal.

## Migration surface

- Replace the private distance-derived implementation in `plane_graph` with the shared
  quadrant-aware function. Remove production use of the old two-argument helper.
- `match_plane_pair` passes its exact candidate `quadrant` to the new function before
  warping and correlation.
- `plane_axis_support.build_axis_support` passes each selected match's exact
  `quadrant_of_right`; its stored homography digest must bind the new grid map.
- `plane_shift_domain` passes the same selected match quadrant when replaying C1 and
  correlation evidence. Replay must remain exact under the new map.
- Update exports and callers so there is one implementation. A compatibility alias is
  allowed only if it also requires the quadrant and contains no independent logic.
- Replace the old graph test that reconstructs `inv(H_left)*H_right`; retain that
  camera-derived expression only inside a named legacy-regression control proving why
  it is rejected.

Do not change `warp_signature`, `_correlation_surface`, `_peak_statistics`, match scores,
PSR/structural thresholds, graph/selector algorithms, P1-B residue arithmetic, P1-D
support enumeration, schemas, artifacts, renderer, evaluator, CLI, placement, or export.

## Provenance and compatibility

Keep existing full reading/signature geometry provenance, including selected `b`,
distance, camera axes, focal, principal point, centre, side, origins, and shape. Those
fields explain how the block-normalized signature was measured even though they do not
define its grid-to-grid orientation.

Add or update the grid-homography digest so it binds:

- exact left/right ordered reading and plane identities;
- exact selected canonical `b` values;
- both grid shapes and origins;
- left/right plane-axis labels;
- strict relative quadrant;
- the exact signed-permutation matrix and affine origin term; and
- a mapping-version tag distinct from the legacy camera-composition map.

Direct construction/replacement of C1 and P1-D records must reject old or forged grid-H
digests. Current accepted P1-D functionality remains required, but its source/test/report
hashes become historical after this deliberate dependency migration. Record all new
hashes; do not edit the accepted P1-D report to disguise the change.

## Decisive tests

1. Retain the C2-R-A unequal-distance fixture. The new map must equal the independent
   unit-linear origin map, while the explicitly reconstructed legacy expression remains
   `0.5` scale. The corrected correlation path must match the independent path.
2. With fixed axes, origins, and quadrant, vary valid camera distance, focal, principal
   point, rectification centre, sampling, and side independently. The new map remains
   byte-identical; full measurement provenance still changes where appropriate.
3. Use a literal expected table for every compatible combination of three left plane
   normals and four relative quadrants. Assert exact signed-permutation matrices,
   including swaps and sign reversals, without reproducing the implementation formula
   in expected values.
4. Nonzero asymmetric integer and fractional origins verify `P*o_R-o_L`; a pointwise
   oracle checks multiple corners/interior points and inverse round trips.
5. Wrong right normal, invalid/bool quadrant, nonfinite/malformed origins, invalid axes,
   non-bijective mapping, and malformed grid geometry fail closed with typed caller
   refusals. No input is normalized silently.
6. Actual shared-world textures under positive/negative asymmetric translations verify
   row=`v`, column=`u`, endpoint direction, correlation peak, overlap, and score replay.
   Include both camera sides and unequal correct camera distances.
7. Existing equal-distance and non-square/holey-coverage controls continue to match an
   independent brute-force warp/correlation oracle.
8. C1 rebuilds exact mapped-axis records and new H digests; P1-D rebuilds complete joint
   domains and replays the newly admitted matches. Mutation and old-digest attacks fail.
9. Existing graph and selector tests are reviewed case by case. Unchanged physical cases
   remain byte-identical; any changed match/solution expectation must be justified by the
   corrected grid map, never bulk-regenerated blindly.
10. The C2-A top-vs-carry counterexample remains a valid ambiguity result under the new
    map or is replaced by an equally decisive corrected-map fixture. The C2-R-A
    unequal-distance defect test changes from rejection evidence to a passing repair
    regression without deleting the recorded historical failure.
11. AST/import checks forbid image loading, manifests, F3/debug coordinates, supplied
    cameras/bounds, references, schematics, GT/evaluator, renderer, CLI, export, and
    placement dependencies in the mapping helper.

## Validation and evidence

Run the focused graph/C1/P1-D/C2/P1-B/selector/lattice/frozen-runner suite plus
`py_compile`, whitespace/diff checks, and exact hashes. Report:

- old versus new matrices and correlation diagnostics on the retained fixture;
- every changed valid match/solution expectation and its physical explanation;
- new source/test/record hashes;
- protected schema/evaluator/renderer/CLI/export hashes; and
- whether the existing R4 frozen artifact can replay under the new source.

R4 is immutable historical evidence. A source or match mismatch is the expected honest
outcome; do not overwrite, relabel, or silently migrate it. After P1-E acceptance, define
and run one new immutable image-only R5 from the same frozen PNG inputs to measure how
the corrected map changes proposals, readings, signatures, matches, graph status, and
resource use. No R5 run is authorized inside P1-E.

## Exit

Stop for lead review after the migration and tests. Do not resume C2-R-A, implement
C2-R-B/V2, run evaluation, or wire placement/output. C2-R-A can be re-specified only
after P1-E proves the corrected grid coordinate convention and a new P1-D domain is
accepted on it.

Manual cameras, camera manifests, F3 coordinates, directions, scene bounds, references,
schematics, and ground truth remain advanced/developer/evaluation-only and are forbidden
as default reconstruction inputs.
