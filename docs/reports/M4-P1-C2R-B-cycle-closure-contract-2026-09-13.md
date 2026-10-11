# M4 P1 C2-R-B graph-wide relative-translation contract

Date: 2026-09-13  
Status: accepted after lead and DeepSeek audit; isolated implementation authorized  
Depends on: accepted C2-R-A3 theorem and accepted C2-R-A4 indexed domains  
Blocks: V2 cell placement, runner integration, and production promotion

## Purpose and non-claims

C2-R-B consumes the complete, factorized candidate domains produced by C2-R-A4
and solves directed relative translations over the frame graph. It preserves all
translation ambiguity needed for review and proves cycle closure where the graph
contains redundant edges.

This slice does not:

- read images, cameras, manifests, F3 data, coordinates, directions, bounds,
  reference schematics, ground truth, evaluation labels, or frozen artifacts;
- accept a caller-supplied gauge, pose, phase, integer translation, or origin;
- rank candidates by residual, PSR, confidence, source order, or provenance;
- materialize an A4 domain or a cross-edge Cartesian product;
- call the existing quadrant/least-squares selector;
- place cells, build a volume, run a renderer/evaluator, or wire CLI/UI/export.

Manual metadata remains unavailable to the default path. Future assisted or
developer wiring is outside this contract.

## Directed translation semantics

For every A4 domain, its match identity defines a directed edge `left -> right`.
The complete edge delta is

```text
d_m = t_right - t_left
```

with one value on each world axis.

The two lateral values are exactly the IEEE-754 floats returned by A4
`vector_at(g)`. For lateral factor `a`, A3/A4 compute, in their frozen evaluation
order,

```text
k_R = entry.column_shift  when right_shift_source_component == "column"
      entry.row_shift     otherwise

d_m[a] = sigma_R * k_R + sigma_L * r_L - sigma_R * r_R
```

Both lateral factors use the same P1-D entry. `sigma_L` and `sigma_R` are the
stored signed world-axis transforms, and `r_L`, `r_R` are stored P1-B candidate
floats. The rounded A4 result is authoritative; C2-R-B must not silently replace
it with an unrounded rational re-evaluation. No term may be split into an
inferred integer and fractional phase after reconstruction.

The normal value is exactly

```text
d_m[n] = domain.normal_constraint.delta_translation
```

`NormalFaceConstraint` independently replays
`left_sign*left_side*left_distance - right_sign*right_side*right_distance`.
This is the same `t_right - t_left` sign used by `plane_graph.solve_component`.
Reverse traversal negates all three components.

No rounding or assumed zero phase is permitted. All values must be finite.

## Input contract

The public solver accepts one complete `C2RVectorDomainResult`, not a loose
sequence of domains. It also receives the current A4 validation sources so the
indexed result can be replayed rather than trusted:

```python
solve_relative_translation_graph(
    vector_result,
    support,
    p1d_result,
    matches,
    phase_evidence,
    *,
    readings,
    signatures,
    comparison_tolerance,
    closure_tolerance,
    budget,
) -> C2RBResult
```

All arguments are explicit. `comparison_tolerance` must exactly equal A4's
validated tolerance. `closure_tolerance` is a finite non-negative float and has
no default. Booleans are invalid numeric inputs.

Before calling the A4 validator, the solver requires
`vector_result.status == "complete"`, at least one domain, no refusals, and
`represented_work == planned_work`. It then calls
`validate_lateral_vector_domain_inputs` with the current sources. A non-complete,
mutated, or non-replaying A4 result is refused before graph construction.

One domain is one match edge. Duplicate match identities are refused. Parallel
edges between the same frames are allowed when their full match identities
differ; each supplies an independent constraint.

## Canonical frame graph and gauge

Vertices are every frame named by an edge. Edges are sorted by the existing A4
match-identity key, independently of caller order. Connected components and
their frame tuples are sorted lexicographically.

Every component has its own mathematical gauge:

```text
anchor = lexicographically first frame in the component
t_anchor = (0.0, 0.0, 0.0)
```

The anchor is never a recovered world coordinate and is never caller-selected.
Renaming frames may change the anchor; gauge-invariant pair differences must
shift consistently. Input permutation must produce byte-identical output.

Each component uses a canonical spanning tree: breadth-first traversal from the
anchor, with incident edges ordered by match identity and neighbouring frame.
Every non-tree edge defines one fundamental cycle in canonical edge order.

Disconnected components are valid independent local reconstructions. No
relative translation between components is claimed. Disconnection therefore
affects placement readiness, not input validity.

## Semantic solution space

Choose one A4 global candidate index for every edge. The choice is feasible when
the signed sum of its complete three-axis deltas around every fundamental cycle
has absolute componentwise residual at most `closure_tolerance`.

For a chosen edge assignment, translations are propagated along the canonical
tree in its fixed order. This produces one gauge-fixed translation tuple ordered
by frame and world axis.

A **translation class** is exact equality of the canonical tuple of reduced
dyadic translations derived from the stored IEEE-754 values. Closure tolerance
does not merge distinct translation classes. This conservative rule avoids a
non-transitive approximate-equivalence relation and accumulation collisions.

Numerically equal edge deltas with different A4 provenance belong to the same
translation class but retain exact provenance multiplicity. Distinct canonical
translation tuples are ambiguous even when their difference is smaller than
the closure tolerance. No class is preferred.

For each component:

- `unique_validated`: exactly one translation class and cycle rank at least one;
- `unique_unvalidated_tree`: exactly one class and cycle rank zero;
- `ambiguous`: at least two translation classes;
- `contradiction`: cycle rank at least one and exhaustive search finds no class;

An acyclic component has no closure evidence. Its candidates can still determine
relative translations, but the result cannot call them validated.

## Operational status and placement readiness

`C2RBResult.status` describes execution, not the mathematical outcome:

- `complete`: every component reached a sound decisive outcome. Unique and
  contradictory outcomes require exhaustive search; ambiguity is decisive as
  soon as two distinct translation classes are certified;
- `refused`: invalid or non-replaying input; no component records;
- `budget_exhausted`: a preflight or runtime limit was reached atomically.

A complete result may contain unique, ambiguous, or contradictory components.
It must never rewrite ambiguity as a refusal.

`placement_readiness` is separate:

- `ready`: exactly one connected component and it is `unique_validated`;
- `review_required`: any ambiguity, contradiction, tree-only evidence, or more
  than one independently gauged component;
- `unavailable`: status is `refused` or `budget_exhausted`, so no component
  decision exists.

V2 remains blocked unless the result is `complete/ready`. Future product policy
may expose review alternatives, but it may not silently choose one here.

## Exact bounded search

General exact existence and uniqueness are worst-case exponential. A single
cycle whose edge sets encode `{0,w_i}` and whose closing edge requires target
`T` contains subset-sum. C2-R-B therefore uses bounded exact search and reports
incompletion honestly; it does not claim a polynomial solver.

The implementation must not build a tuple/list/dict containing every A4 leaf,
and must not call `materialize_lateral_vector_domains`.

Every complete delta component is first obtained through A4's frozen float
evaluation: lateral components from `vector_at(g)` or an exactly replayed copy of
its operation order, and the normal component from `delta_translation`. Each
resulting float is then converted with `float.as_integer_ratio()` to a reduced
canonical dyadic `(numerator, denominator)`. Propagated translations, required
chord values, cycle sums, and tolerance comparisons use these exact rationals.
`closure_tolerance` is converted to the same form.
Serialized certificates bind exact dyadics; finite float views are derived only
for display. This removes summation-order and float-collision ambiguity without
changing an A4 value.

Search uses these operations:

1. Build components, canonical trees, chords, and fundamental-cycle incidence.
2. Use A4 entry groups and axis factors to derive finite value bounds without
   constructing candidates.
3. The only search variables are the canonical spanning-tree edges. Traverse
   each tree edge's candidate views lazily and group them into exact complete
   dyadic delta classes with exact provenance multiplicities under the scan and
   memo budgets. No chord is assigned independently.
4. Assign tree-edge variables globally in canonical order by decreasing cycle
   participation, then smaller represented domain, then match identity. Store
   one shared assignment map for every cycle. This affects work only, never the
   semantic result.
5. Prune a partial assignment only when interval bounds or an exact joint
   reachable-sum summary proves no completion of that global assignment can
   close. A summary key contains every assigned edge value on which that summary
   depends, whether or not the edge is shared by another cycle.
6. Once all tree-path edges of a chord are assigned, their exact propagated
   translations determine one required directed chord delta. Query that chord's
   A4 factors without enumerating its leaves: require its fixed normal delta to
   lie within tolerance, iterate its P1-D entries, and for each lateral axis
   replay the A4 float operation order for each inspected left/right residue pair,
   convert that rounded result to a dyadic, and count pairs lying within tolerance
   of the required component. This factor-pair work is runtime-budgeted. Both
   axes use the same entry. Sum exact satisfying provenance counts and retain
   bounded canonical witnesses.
7. Chords are conditionally independent after the single global tree assignment:
   each only needs one candidate matching its required delta. Shared tree edges
   cannot take different values in different cycle proofs.
8. Stop semantic search after finding two distinct translation classes because
   ambiguity is then proven. Proving uniqueness or contradiction requires
   exhaustive completion within budget.

Cycle constraints sharing an edge must read the same global tree-edge assignment.
Per-cycle sumset membership alone is not sufficient and may not be used as proof
when it permits incompatible choices for a shared edge.

Candidate-view scanning may be O(sum of visited A4 leaves), but it is lazy,
bounded, and never retained wholesale. Cross-edge products are explored as a
depth-first constraint search; they are never precomputed or fully materialized.

The singleton fast path is mandatory. Eligibility is factorized and may not use
`represented_candidate_count == 1` as a proxy. For each lateral factor, scan the
stored P1-D entries only to obtain the minimum/maximum selected right-shift
component; combine those with the first/last stored left and right residue values
according to their coefficient signs. Evaluate every endpoint corner in A4's
frozen float order. Because the rounded expression is monotone in each operand,
equal rounded minimum and maximum are necessary and sufficient for every leaf to
have one numeric value. The exact provenance multiplicity is then the represented
candidate count and global index zero is the canonical witness. If any endpoint
differs, eligibility returns false without scanning leaves.

If each edge has one exact numeric delta, translation propagation and every cycle
check run in O(stored P1-D entries + vertices + edges), with zero candidate
objects, zero candidate-view scans, and no generic search nodes.

## Provenance and certificates

Every edge choice is addressed by `(match_identity, global_index)`. A compact
reference contains:

- match identity and A4 domain digest;
- global mixed-radix index and P1-D entry index/row/column;
- four residue indices and exact `provenance_key_at(index)`;
- complete axis-labelled exact-dyadic `delta_xyz` plus finite float views;
- for tree edges, exact provenance multiplicity of the exact delta class;
- for chords, exact count of provenance leaves satisfying the required delta
  within `closure_tolerance`, plus each retained witness's exact delta.

Certificates use compact public index/provenance/vector accessors and never need
the full candidate record. Solver, traversal, membership, validation, digest,
and certificate paths must therefore make zero `candidate_at` calls and construct
zero `LateralVectorCandidate` objects. `candidate_at_calls` remains a measured
runtime invariant fixed at zero, not a configurable budget counter.

For a unique translation class, retain one canonical provenance witness per edge.
Its exact assignment multiplicity is the product of each tree-edge exact-delta
multiplicity and each chord's tolerance-satisfying provenance count. Exact dyadic
propagation makes translation equality independent of float accumulation. The
witness is an audit address, not a selection claim. If exact multiplicity cannot
be computed within budget, the whole operation is `budget_exhausted`.

For ambiguity, retain certificates for the first two distinct translation
classes in canonical search order only as proof that uniqueness is false.
`ambiguity_class_count_lower_bound` is `2`; no exact total is claimed unless the
search was already exhaustive. No downstream code may treat certificate order
as rank.

Contradiction retains the canonical failing cycle set and residual bounds, with
no nearest candidate. Tree-only output records `cycle_rank=0` explicitly.

## Types and persistence

All public records are frozen, slotted, self-validating, JSON-safe, and hashed as
lowercase SHA-256 over canonical JSON (`sort_keys=True`, compact separators,
`allow_nan=False`). Proposed records:

- `C2RBudget`;
- `C2RBDeterministicWork`;
- `C2RBRuntimeWork`;
- `C2RBEdgeCandidateRef`;
- `C2RBCycleResidual`;
- `C2RBTranslationClassCertificate`;
- `C2RBComponentResult`;
- `C2RBRefusal`;
- `C2RBResult`.

Exact-dyadic fields serialize as reduced integer numerator/positive denominator
pairs. Denominators must be powers of two because every source value is an exact
finite IEEE-754 float; constructor attacks with unreduced or non-dyadic pairs are
rejected.

`C2RBResult` binds the complete A4 result/input digests, tolerances, budget,
component partition, gauges, canonical trees/cycles, outcomes, certificates,
work, connectivity, readiness, status, `input_sha256`, and `result_sha256`.

```python
validate_relative_translation_graph(
    result,
    vector_result,
    support,
    p1d_result,
    matches,
    phase_evidence,
    *,
    readings,
    signatures,
    comparison_tolerance,
    closure_tolerance,
) -> bool
```

Validation replays A4 and recomputes the complete C2-R-B result using the
recorded budget. It compares canonical bytes and never trusts caller copies.

## Work accounting and budgets

Deterministic preflight work is exact:

- components;
- frames;
- edges;
- tree edges;
- chords;
- cycle rows (`3 * chords`);
- stored P1-D entries;
- stored residue values;
- represented A4 candidate count, reported as an echo only.

Runtime work is measured, not predicted:

- candidate views scanned;
- partial assignments entered;
- closure component checks;
- factorized membership probes;
- memoized reachable-sum terms retained;
- provenance witnesses inspected;
- full `candidate_at` calls, required to remain zero;
- certificates retained.

`membership_probes` counts every frozen-order factor expression evaluated for
bounds, singleton eligibility, chord membership, or search. Exhausting this cap
before the first search node is a valid atomic runtime exhaustion, not a refusal.

`C2RBudget` has explicit positive exact-integer caps for every deterministic
counter except represented-candidate echo, plus:

- `max_candidate_views_scanned`;
- `max_partial_assignments`;
- `max_closure_component_checks`;
- `max_membership_probes`;
- `max_memoized_terms`;
- `max_provenance_witnesses`;
- `max_certificates`.

No defaults are permitted. `max_certificates` and
`max_provenance_witnesses` must each be at least two so ambiguity can be
certified; other caps are positive exact integers. Exact-limit succeeds.

Precedence is fixed: validate scalar/types and complete A4 status, replay A4
against current sources, validate graph identities, compute deterministic plan,
then check preflight budgets. Invalid input is therefore refused even when its
declared size also exceeds a budget. Preflight excess publishes the exact
deterministic plan, zero runtime work, no components, and
`budget_exhausted/planned_budget_exceeded`.

Any runtime excess discards every component decision and certificate and returns
`budget_exhausted/runtime_budget_exceeded`, preserving the measured lower-bound
work through the last completed operation. Runtime mismatch or invalid iterator
state returns `budget_exhausted/runtime_count_mismatch`. The input A4 result is
never mutated. No partial solution is externally observable after exhaustion.

## Refusal taxonomy

Input refusals are deterministic and sorted:

- `vector_result_invalid`;
- `vector_result_replay_mismatch`;
- `comparison_tolerance_mismatch`;
- `closure_tolerance_invalid`;
- `duplicate_match_identity`;
- `identity_endpoint_invalid`;
- `nonfinite_delta`;
- `factor_contract_invalid`;
- `graph_empty`.

Per-component mathematical outcomes are not refusals. Disconnection, ambiguity,
contradiction, and absent cycle redundancy are reported in component/readiness
records.

Refusal details are at most 240 characters and contain no control characters.
Refusals never include a selected candidate.

## Required proof and regression matrix

1. Directed-sign oracle independently proves all three `t_right-t_left`
   components across rotations/reflections and reverse traversal.
2. Singleton triangle closes exactly and is `unique_validated/ready`.
3. Normal-sign mutant and lateral-sign mutant are contradictions, never nearest
   candidates.
4. Two distinct closing translation classes are `ambiguous`; input order cannot
   choose one.
5. Numeric-equal deltas with distinct provenance produce one translation class
   with exact multiplicity and distinct witnesses.
6. A three-frame tree distinguishes unique-unvalidated from ambiguous and never
   claims closure validation.
7. Two disconnected closing triangles retain two independent gauges, are
   operationally complete, and require review for placement.
8. Shared-edge multi-cycle fixture rejects incompatible per-cycle witnesses.
9. Factorized chord membership matches an independent eager oracle for every
   small entry/residue combination, including cross-entry rejection.
10. Exact-limit and one-below tests cover every deterministic and configurable
    runtime budget; runtime exhaustion publishes no partial components or
    certificates. The non-configurable `candidate_at_calls` counter is asserted
    exactly zero on every outcome.
11. Constructor, `replace`, digest, source-mutation, nonfinite, duplicate-edge,
    invalid-index, and iterator-count attacks fail closed.
12. Default solve/serialize/validate paths construct zero A4 candidate objects;
    no call reaches full materialization.
13. Input permutations are byte-identical; renaming frames preserves all
    pairwise differences after canonical regauging.
14. A3 q0 and repaired-reflection truth candidates remain feasible in constructed
    cycle contexts; formula/sign mutants are rejected.
15. P1-B/C1/P1-D/P1-F/P1-G/A3/A4/graph/selector/frozen-runner regressions remain
    unchanged and pass.

## Performance gates

The first implementation must measure, not guess, production caps. Before
acceptance it must publish:

- singleton-path time and additional RSS for a 1,000-edge synthetic graph;
- bounded-search time/RSS/counter trace for small unique, ambiguous, and
  contradictory cyclic fixtures;
- the 289-candidate A4 fixture with zero materialization and zero full candidate
  objects outside any explicitly retained certificate;
- one >2-million-represented-vector factorized input proving preflight and
  singleton-eligibility detection do not scale with represented leaf count;
  when it is non-singleton, bounded general search may then exhaust atomically.

Initial hard gates:

- singleton 1,000-edge solve: <=2 seconds and <=32 MiB additional RSS;
- small proof fixtures: <=5 seconds each and <=32 MiB additional RSS;
- serialized result: <=8 MiB for every required fixture;
- general search memory: O(graph + current depth + bounded memo + certificates),
  never O(represented candidates) or O(cross-edge assignments).

These limits may be tightened after measurement. They may not be relaxed to
permit eager domain or assignment materialization.

## Isolation boundary

The production module may import standard numeric/record utilities and these
reconstruction contracts only: `plane_axis_support`, `plane_graph` for immutable
types/`axis_transform`, `plane_residue`, `plane_shift_domain`, and
`plane_vector_domain`.

AST tests prohibit image/camera/manifest/F3/coordinate/bounds/reference/GT,
evaluation, renderer, dataset, filesystem, network, process, selector,
least-squares solver, placement, CLI, export, and world-writing access. They also
prohibit `open`, `exec`, `eval`, `compile`, and dynamic import calls.

## Exit

Implementation ends after the isolated solver, proof matrix, performance report,
combined regressions, compile/import checks, and protected-artifact hashes pass.
It stops for lead and DeepSeek review. Runner wiring, V2, persistence, evaluation,
and default-product integration remain blocked until explicit later contracts.
