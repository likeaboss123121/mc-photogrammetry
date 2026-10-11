# M4 P1 C2-R-A3 independent coupled-vector proof and builder contract

Date: 2026-09-13  
Status: assigned after lead acceptance of P1-F and P1-G  
Owner: Luna  
Depends on: accepted P1-B, P1-C1, P1-D, P1-F, and P1-G

## Required result

C2-R-A3 must prove, then implement, the complete coupled two-axis lateral-vector
domain defined by
[`M4-P1-C2R-vector-domain-contract-2026-09-13.md`](M4-P1-C2R-vector-domain-contract-2026-09-13.md).
The post-P1-E A2 contract remains binding except where this document replaces its
stale map-direction and reading-index premises.

The analytic matrix is a hard gate. If any physically valid case omits the
independently declared truth vector, or if the finite compatibility predicate
cannot separate valid from invalid maps, retain the first counterexample, publish
`FORMULA_REJECT`, and stop before adding a builder. Only a fully passing matrix
permits an isolated immutable domain module.

This milestone enumerates candidates. It does not rank candidates, select a
translation, solve cycles, infer absolute placement, construct voxels, or change
user-visible output.

## Circular-oracle defects that must be removed

The current A2 matrix is retained as historical regression evidence, but is not
sufficient proof because it:

- copies the production axis-transform literal into `_TRANSFORM` and then uses
  that copy for the expected basis, grid map, sampled texture, and formula signs;
- calls `match_plane_pair` and fabricates plane indices with `replace(..., 0)`;
- keys phase evidence by `(frame, 0)` instead of canonical match endpoints;
- declares `_SUPPORTED_B` but exercises only `b=1`;
- checks existence of a truth hit without retaining the exact four residue
  selections that produced it;
- writes the known truth into `candidate_deltas` instead of retaining the
  computed candidate value;
- replaces factory coverage with all-ones masks; and
- does not execute the cross-entry, complete budget, provenance, or mutant
  controls required by the parent contract.

Do not repair these by copying the production formula into another helper. The
expected physical result and every mutant verdict must be computed independently
of the candidate builder.

## Independent physical oracle

1. Declare endpoint world translations, grid origins, camera sides, distances,
   and rational block periods before generating phase evidence, signatures,
   correlation shifts, P1-B residues, or P1-D entries. Expected lateral truth is
   only `T_right - T_left` on the two common world tangents.
2. Derive endpoint bases from the declared right-handed world frame and the
   closed-form rotation `Ry(-pi*q/2)` applied to unit basis vectors. Round only
   the analytically exact quarter-turn zeros and signs. The oracle must not read
   `axis_transform`, `_AXIS_TRANSFORMS`, `_ry90`, or a copied transform table.
   Separately assert production `axis_transform` agrees with the physical model.
3. Sample one asymmetric nonperiodic world texture through those independently
   derived bases and translations. Use the real `plane_signature` factory for
   shape, endpoint-inclusive origin, interior, and coverage. Photometric cell
   values may be replaced only inside the factory-valid coverage; do not replace
   coverage or interior with constant masks.
4. Generate phase observations forward from the declared physical translation
   and pixel/block spacing. The real P1-B path must recover a set containing the
   independently known local residue. Expected truth may never be computed from
   a selected residue candidate or P1-D entry.
5. Compute an independent direct bounded normalized-correlation surface over
   the physical grids. Record its optimum, the independently known physical
   carry, and the support/value at that carry separately from the production
   whitened phase-correlation top. Do not require those two ranking methods to
   choose the same entry: the retained counterexample legitimately has direct
   raw-NCC/physical carry `(1,1)` while the graph top is `(0,1)`. Acceptance
   requires P1-D to retain the supported physical carry, not the graph top to
   equal it. Do not use `cell_grid_homography`, `_correlation_surface`,
   `_peak_statistics`, or the candidate formula to construct the independent
   surface or physical carry.
6. Enumerate every retained P1-D entry and every residue pair. For each generated
   vector, retain the actual computed delta and all four residue indices/values.
   Compare those computed vectors to the fixed world truth afterward.

## Exact coordinate theorem and mutants

For common lateral world axis `A`, resolve the unique endpoint local labels and
signs from the independent physical basis and cross-check production transforms:

```text
(A, sigma_L) = axis_transform(left_local_label, left_quadrant)
(A, sigma_R) = axis_transform(right_local_label, right_quadrant)
k_R(A,s) = s.column_shift when right local label is u, else s.row_shift
signed_world_carry = sigma_R * k_R(A,s)
delta_A = signed_world_carry + sigma_L*r_L - sigma_R*r_R
```

Both axes of one vector must use the same complete `ShiftDomainEntry`. Store raw
right-grid carry and `signed_world_carry` separately and recompute the latter
exactly once. Retain numerically equal vectors when entry or residue provenance
differs.

The independent matrix must reject each mutant on at least one asymmetric
nonzero-carry fixture: row/column transpose, endpoint reversal, local-name rather
than world-axis pairing, left-residue sign flip, right-residue sign flip, missing
carry sign, double-signed carry, origin reversal, basis transpose/inverse, and
cross-entry component mixing.

## Finite compatibility predicate

Publish and test a deterministic predicate before the builder. Given bound
right-to-left grid map `H`, endpoint origins, and the independently derived
signed permutation `P`, it accepts only when:

- every value is finite and `H[2,2]` is finite and nonzero;
- normalized `H[2,:]` is `(0,0,1)` within the declared tolerance;
- normalized `H[:2,:2]` equals `P` within tolerance;
- normalized translation equals `P @ origin_right - origin_left` within
  tolerance; and
- the P1-F `block-cell-grid-v2` digest and both C1/P1-D geometry/content digests
  replay exactly.

Translation is otherwise unrestricted. Distance, focal length, principal point,
rectification centre, samples, side, and period may change measurement
provenance but may not rescale this one-cell-per-block map. Finite projective
matrices, denominator instability, old v1 digests, or incompatible signed
permutations are typed refusals, never valid helper outputs.

## Matrix census

Publish the exact executed census and per-stratum truth-hit counts. Minimum
coverage is:

- all 48 `(left normal, left absolute quadrant, relative quadrant)` orientation
  cases end to end, with compatible right normals derived from the independent
  physical rotation;
- all seven P1-F signed rotation/reflection matrix classes and both determinant
  signs;
- equal-sign and opposite-sign endpoint cases, both sides at each endpoint, and
  cases where common world axes preserve and swap local `u/v` names;
- unequal nonzero origins; asymmetric positive and negative row/column carries;
  at least one carry component with magnitude greater than one;
- residues at ordinary values and within the declared tolerance of zero and one;
- every supported `b` value `(1/16,1/8,1/4,1/2,1,2,3,4,8,16)` in equal pairs,
  plus unequal physically coherent pairs covering every denominator/numerator
  class; adjust distance/spacing coherently instead of forcing equal images;
- independent variation of distance, focal, principal point, rectification
  centre, samples, side, period, and origin with expected correspondence fixed;
- one cyclic match set and one disconnected set, both producing domains without
  any solved/unique/placement claim; and
- the retained q0 `(0,1)` peak / `(1,1)` carry / `(1.4,0.8)` truth fixture and
  the repaired reflection case that rejected A2.

The proof must assert at least one exact supporting provenance tuple for every
valid in-radius case. Multiple truth-supporting tuples are reported as domain
ambiguity; the proof must not select the first. A cross-shaped synthetic domain
must prove that components supported by different entries cannot be mixed.

## Canonical identity and required APIs

Use `match_plane_graph(complete_readings, signatures, ...)`; do not call
`match_plane_pair` and patch numeric indices. Include a leading or interleaved
signatureless reading in the matrix. Build every lookup from the same complete
reading universe and `reading_order_key`, then require exact reading-object,
frame, axis, canonical-b, and signature-object identity.

Pass the same complete universe to `build_shift_domains(..., readings=...)` and
`validate_shift_domain_inputs(..., readings=...)`. Phase-evidence maps use the
published `left_plane/right_plane`, never a hard-coded zero. Missing, duplicate,
foreign, tied, reordered, legacy-renumbered, or content-conflicting identities
must refuse before formula enumeration.

The new C2-R builder must require the accepted C1 support, accepted P1-D result,
exact P1-B evidence, complete readings, exact signatures, explicit comparison
tolerance, and an explicit budget. No argument may default to a
signatures-derived universe. Bind a deterministic digest of the canonical
`(frame,index,reading_order_key)` universe and all exact P1-B evidence records.
Do not change `PlaneMatch`, existing schemas, C1, P1-D, graph, or selector APIs.

## Builder and work accounting

After and only after the matrix passes, implement the frozen/slotted/self-
validating records from the parent contract in a new isolated reconstruction
module. Status is exactly `complete`, `refused`, or `budget_exhausted`.
Every complete result states:

```text
completeness_scope = declared_inputs_only
physical_truth_membership = unverified
```

Preflight and runtime counters must independently reproduce:

```text
requested_match_identities = len(distinct sorted requested identities)
joint_entries = sum(len(match_domain.entries))
axis_residue_pairs = sum over entries and both lateral axes of
                     len(left_residues) * len(right_residues)
vector_candidates = sum over entries of pairs_axis_0 * pairs_axis_1
provenance_references = 7 * vector_candidates + requested_match_identities
```

The seven per-vector references are one joint P1-D entry, two C1 mapped-axis
sources, and four endpoint/axis residue selections; the per-match reference is
the normal constraint. Compare all five counters to independent limits,
including the requested-identity limit itself. Any planned excess or runtime
mismatch discards every candidate. No truncation, top-N, score/confidence gate,
or silent deduplication is permitted.

Exact-limit and one-below tests are required for all five counters. Input
permutation must produce byte-identical records. Constructor and `replace()`
attacks must reject forged signed carries, formulas, digests, residue indices,
mixed entries, reordered axes, normal-axis terms, nonfinite values, and
identity-equal evidence with changed content.

## Scope and stop

The new module may import only numeric/record primitives and the accepted P1-B,
C1, P1-D, P1-F/P1-G identity contracts. AST/import tests must prohibit images,
camera manifests, F3/coordinates, scene bounds, annotations, references/GT,
frozen artifacts/R4/R5, evaluator, renderer, CLI, export, dataset, filesystem,
network, process, solver/selector, placement, and world-writing access.

Re-run P1-B/C1/P1-D/P1-F/P1-G/graph/selector/frozen-runner regressions; publish
matrix counts, mutant census, representative record hashes, planned/runtime
work equality, focused and combined tests, compile/import/diff checks, and
protected R4/R5 hashes. Do not run another real fixture or open labels.

Stop for lead review after A3. C2-R-B remains a separate graph-wide cycle solver
requiring redundancy, bounded translations, and typed ambiguity/contradiction/
disconnection. V2 remains blocked.

## A3 matrix closeout — proof passes; builder remains gated

The independent proof is persisted in
[`tests/test_c2r_vector_domain_a3_proof.py`](../../tests/test_c2r_vector_domain_a3_proof.py).
It exercises the real canonical graph → P1-D path with the complete reading
universe, including a signatureless reading ordered before the left endpoint.
The graph's whitened peak, a bounded direct raw-NCC optimum, the independently
derived physical carry, and the coupled truth are retained as separate values.
The proof requires P1-D to retain the exact physical entry and at least one
four-residue provenance tuple for every valid case; multiple such tuples remain
reported as ambiguity rather than being selected away.

The carry oracle is forward and independent of the candidate delta formula. It
maps a fixed integer cell through the declared right basis, origin, and world
translation, then inverts that world point through the left endpoint. It also
pulls a fixed integer world lattice point back through both endpoint bases and
translations, independently checks the generated residues, and derives the
integer row/column carry from endpoint integer cell indices and signed mapped
axes. Only after that carry is fixed does the candidate formula test truth
membership. The 48 orientation cases produce 48/48 truth hits, 3,888 exhaustive
vector records, and 48 truth-supporting provenance tuples. The direct NCC
optimum is at the physical entry in 20/48 cases; minimum direct-NCC support at
the physical entry is 196. There are no cases with multiple tuples at the
physical entry. The full 48-case candidate table digest is
`91b5101be4ae51c6ec6f70876338715c8345e08b6522b746098545f4c11e2033`.

The orientation census covers all seven absolute signed-map classes, with
determinants +1 in 32 cases and −1 in 16. Side-pair counts are (+,+): 8,
(+,−): 16, (−,+): 16, (−,−): 8. The common world axes preserve local u/v labels
in 24 cases and swap them in 24. Equal rational b pairs cover all ten supported
classes: 10/10 cases, 1,747,750 vector candidates, and 346 truth-supporting
provenance tuples. Unequal coherent rational pairs cover all ten classes:
10/10 cases, 443,425 candidates, and 91 truth tuples. The combined count for
these orientation and rational-pair strata is 2,195,063 candidate vectors.
Some residue tuples map to the same truth vector; the proof retains this
ambiguity (four equal-b and three unequal-b cases have multiple truth entries;
physical-entry ambiguity occurs in three equal-b and two unequal-b cases).

Additional controls pass for residues within `2e-8` of zero and one; positive
column/negative row carries `(row=-2,column=3)`; negative column/positive row
carries `(row=2,column=-4)`; and the reflected q2 carry-sign case. The cyclic
three-edge graph yields three declared domains and the disconnected two-edge
graph yields two, without a solved, unique, placement, or cycle-consistency
claim. Geometry variants independently change focal length, principal point,
rectification centre, samples, side, period, distance, extent, and origin:
the discrete grid map remains byte-identical while measurement-geometry
provenance changes.

Two named retained examples make the distinctions concrete. In the q0 case
`(n2,qL0,qRel0)`, the graph whitened top is `(0,1)`, while both the direct raw
NCC optimum and the independently derived physical entry are `(1,1)`; truth is
`(1.4,0.8)`. Raw NCC at that physical entry is `0.8633912428887828` with support
128. The candidate record hash is
`91921a8c67f42a3de01c781497880427ac0d4df4a66c2c132e3e0d1123a04269`, the full
candidate-table hash is
`ee53703b776b9710a4211c999287d3080fd504be6c0331a8386199f21d355246`, and its
P1-D domain record hash is
`06328a887d6735b9a8fcb6c1b185eb510f3bf36489f63e2376ab2bdb0f95ac18`. In the
repaired reflection case `(n0,qL0,qRel2)`, the map is `diag(1,-1)`, truth is
`(1.4,2.8)`, the physical entry is `(row=-2,column=1)`, and raw NCC there has
support 210. Its candidate record, candidate-table, and P1-D domain hashes are
`cbdadf85e4857942086ac6b62b3bfd7934e3eeb90e400b85e1b20abaf933185c`,
`30614dcb11b7d87f0e8564f2b6cb996cc28efcf1189990ecec031bb32d5e8bdd`, and
`8c1b57d275123a69d8b32cd0df223218db2d7c4d7eadfadcd677b03d324986b2`.

The finite compatibility predicate binds normalized affine H, signed
permutation P, origins, mapping version `block-cell-grid-v2`, and replayed P1-F,
C1, and P1-D digests. Its typed refusal controls cover projective maps,
unstable/zero/nonfinite denominators, scale or signed-map mismatch, origin
reversal, legacy version, missing/malformed bindings, and mutated P1-F/C1/P1-D
digests. Formula/axis/origin/endpoint/cross-entry mutants, canonical reading
and signature permutations, omitted/foreign/duplicate/content-mutated identity
refusals, and the import boundary are also covered.

Validation: the proof file passes **125 tests**, including the complete
48-orientation matrix, 10 equal-b cases, 10 unequal-b cases, nine independent
measurement-geometry mutations, and refusal/mutant/topology/provenance controls.
The lead independently reran the 125-test proof in 18.2 seconds with all tests
passing. The prescribed combined regression command is:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_plane_residue.py tests/test_plane_axis_support.py tests/test_plane_shift_domain.py tests/test_plane_graph.py tests/test_lattice_planes.py tests/test_joint_plane_selector.py tests/test_frozen_plane_evaluation.py tests/test_frozen_image_runner.py tests/test_c2r_vector_domain_a3_proof.py
```

The combined regression completed with **436 passed, exit 0, in 126.5960604
seconds**. `py_compile` passed for the P1-B/P1-C1/P1-F/P1-G source modules, P1-D
source, and A3 proof test.
The test-file SHA-256 at proof review was
`d9bb60c92f5212211ecfe6d0473b6e4ed7ffaa5e80ea280ba6f5dbdf598b3116`.

Source SHA-256 values at proof review:

| File | SHA-256 |
|---|---|
| `src/mc_photogrammetry/reconstruction/lattice_planes.py` | `eb25a2f5d6f97f38f7c572b9dca8a0820db19f7d712db6530df88cb3225c008e` |
| `src/mc_photogrammetry/reconstruction/plane_residue.py` | `55ec0cad8fe1088b7b33b1f57c4d58d3d3b912f79765a1f391553ac7b845be0d` |
| `src/mc_photogrammetry/reconstruction/plane_graph.py` | `c5245680b469f93ded5525bb80b0ddf065b46a64e1d88d66cc1fa93115522e92` |
| `src/mc_photogrammetry/reconstruction/plane_axis_support.py` | `18598afb2096b504464411e3a2cbf6a2e960cb2601c4dd2fb8f77d4b46c91f3f` |
| `src/mc_photogrammetry/reconstruction/plane_shift_domain.py` | `512c6fb1c9d3491e3deb4ea4db11353e7f8167dcf8aaaabe8fd17ea17303d91b` |

Protected R4/R5 freeze and sidecar hashes remain: R4 freeze
`96493dd9f42f1d20c932a8b09f267d7f86da38b7e5ac588dec1330ad9df81c9d`, R4
sidecar `f28cc68b2fc5086473ba3b73c4b277e0d1338c10639c0c6a4b64e94585be0bac`,
R5 freeze `8dfc8b41bad1dff018fa08db42411c5f91572233cb1e7718cc17a082ab0c218b`,
and R5 sidecar `087d379673be1ff291f807753cdb00206ad02a64aa26f91104d24abc36a8db5b`.
The eight preregistered PNG hashes were rechecked and remain unchanged. No
builder, runner, schema, solver, selector, CLI, evaluator, renderer, export,
fixture run, or label access was added. Stop here for lead and independent audit.
