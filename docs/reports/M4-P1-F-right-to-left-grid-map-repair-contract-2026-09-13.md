# M4 P1-F right-to-left grid-map repair contract

Date: 2026-09-13  
Status: assigned after lead acceptance of C2-R-A2 `FORMULA_REJECT`  
Owner: Luna  
Depends on: P1-E implementation and retained C2-R-A2 orientation counterexample

## Defect and required result

`cell_grid_homography` is documented and consumed as a right-signature-grid to
left-signature-grid map. P1-E instead constructs its linear part by iterating
left tangent labels and transforming them toward right labels. Its earlier
literal 12-case table repeated that implementation convention and was not an
independent physical oracle.

C2-R-A2 found the first decisive valid case:

```text
left normal = 0
left absolute quadrant = 0
relative/right quadrant = 1
left origin = (3,-2)
right origin = (-4,5)

actual P1-E H       = [[0, 1,  2], [-1, 0,  6], [0,0,1]]
independent B_L^T B_R = [[0, 1,  2], [ 1, 0, -2], [0,0,1]]
```

P1-F must repair the public helper so its matrix direction agrees with physical
absolute tangent bases for every valid orientation, then migrate graph, C1, and
P1-D provenance/replay together. It must not change match ranking, correlation,
thresholds, scale alternatives, solver acceptance, schemas, or output wiring.

## Independent theorem

Use a literal transform table independent of production `axis_transform`. For
every left absolute quadrant `q_L`, relative quadrant `q_rel`, and local left
normal, set `q_R=(q_L+q_rel) mod 4`; resolve the compatible right local normal
whose transformed world normal axis equals the left world normal axis, allowing
the normal sign to differ because plane normals are unoriented.

Let `B_L` and `B_R` be 2x2 matrices whose columns map each endpoint's sorted
local tangent labels `(u,v)` into the same sorted pair of world tangent axes,
including signs from the literal table. The required right-to-left linear map is:

```text
P = B_L.T @ B_R
x_L = P @ (x_R + origin_R) - origin_L
H = [[P, P @ origin_R - origin_L], [0,0,1]]
```

`P` may be a signed rotation or reflection. It must have exactly one `+/-1` in
each row and column. Do not require determinant `+1`; dropping the normal-sign
relationship can make a physically valid tangent map reflective.

The proposed relative-only implementation may iterate right tangent labels,
apply the declared `q_rel` transform into the left/gauge label space, and place
the sign at `[left_index,right_index]`. This is accepted only if the independent
48-case absolute-basis matrix proves byte equality. If it does not, retain the
first counterexample and stop without another guessed table.

## Scope and provenance

Keep the public signature and validation of
`cell_grid_homography(left_signature, right_signature, relative_quadrant)`.
Require exact axis compatibility, finite nonempty grids, finite origins, and a
bijection. Camera distance, selected `b`, period, focal length, principal point,
rectification centre, samples, and side remain measurement provenance only and
must not alter `H` when axes/origins/quadrant are fixed.

Version the grid-map digest again so P1-E and P1-F records cannot collide.
Recompute it from ordered endpoint identities, canonical b values, shapes,
origins, axes, relative quadrant, map direction/version, and normalized exact H.
C1 and P1-D must both call the same helper and bind the same new digest. Keep the
separate measurement-geometry digest behavior unchanged unless a failing
invariant proves it wrong.

The retained P1-E and R5 artifacts are immutable historical diagnostics. Mark
P1-E's literal sign table as superseded evidence; do not rewrite its report,
hashes, R5 freeze, or exhaustive diff. Do not run R6 in this task.

## Decisive tests

1. Retain the exact C2-R-A2 failing case above and require repaired helper H to
   equal the independent matrix including translation.
2. Cover all 48 `(left normal, q_L, q_rel)` cases. Assert world-normal-axis
   compatibility, `P=B_L.T@B_R`, pointwise mapping, inverse round trip, and all
   signed rotation/reflection classes actually observed.
3. Use asymmetric fractional unequal origins and nonsquare grids so transpose,
   inverse, sign, endpoint-order, and origin-order mutants each fail.
4. Sample one shared asymmetric world texture under independently declared
   endpoint translations for every orientation class. The real warp and
   correlation must agree with the independent map; do not derive expected
   peaks using the helper under test.
5. Vary distance, b, period, focal, principal, centre, samples, and side one at a
   time. H remains byte-identical while the already specified measurement
   provenance changes.
6. Rebuild C1 and P1-D on representative rotation and reflection cases. Their H
   and grid-map digests must equal the helper and each other; old P1-E digests,
   forged directions, and mutated quadrants refuse.
7. Invalid quadrant, incompatible axis, nonfinite origin, empty/nonfinite grid,
   non-bijective construction, and denominator/pathological input fail closed.
8. `match_plane_graph` and `solve_component` reading identities remain stable;
   add a signatureless-reading regression so numeric indices cannot silently
   change between matcher and consumer. If fixing that needs a public interface
   change, stop and request a separate contract rather than widening P1-F.
9. Existing P1-B/C1/P1-D/P1-E/C2-R-A/A2, graph, selector, lattice, and frozen
   runner tests remain meaningful and pass. Update expected matrices only where
   the independent physical theorem proves the old expectation wrong.
10. AST/import checks preserve the reconstruction-only boundary: no metadata,
    manifests, coordinates, F3, references, GT, evaluator, renderer, CLI,
    filesystem/network/process, placement, volume, or export dependency.

## Validation and stop

Publish the full matrix census, first old/new counterexample, correlation checks,
digest-version evidence, focused and combined tests, compile/import/diff checks,
and protected hashes. Stop for lead review. Do not resume C2-R-A2, implement its
builder, run a real fixture, change the legacy solver, start C2-R-B/V2, or wire
production/export inside P1-F.
