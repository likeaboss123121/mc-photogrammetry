# M4 P1 C2-R-A4 indexed coupled-vector domain contract

Date: 2026-09-13  
Status: approved for isolated implementation  
Owner: GPT-5.6 Luna  
Review: GPT-6 Astra; DeepSeek read-only architecture audit  
Dependencies: C2-R-A3 proof accepted; eager builder prototype diagnostic only

## Decision

The eager C2-R builder design is superseded before acceptance. It preserves the
correct mathematical candidate set, but its representation is not usable at the
declared scale.

Measured on the existing one-match fixture:

| Quantity | Result |
|---|---:|
| P1-D entries | 289 |
| Represented vector candidates | 289 |
| Serialized result | 1,211,191 bytes |
| Serialized bytes per candidate | 4,191.0 |
| Builder time | 0.888 s |
| Python traced peak | 10.55 MiB |
| RSS change | 39.45 to 52.48 MiB |
| Process peak RSS | 61.84 MiB |

The accepted A3 matrix represents 2,195,063 vectors. Linear serialization at
the measured density is about 9.20 GB before Python object overhead. The
10-million-candidate contract limit would permit roughly 41.9 GB of JSON. A
budget that cannot be represented in ordinary memory is not a useful production
boundary.

A4 stores the exact Cartesian-product factors and reconstructs any full
candidate by index. It changes representation only. The accepted A3 formula,
truth-membership proof, provenance, refusals, and physical non-claim remain
binding.

## Semantic completeness and materialization

For match domain `m`, define:

```text
E_m  = ordered P1-D ShiftDomainEntry table
L_0, R_0 = ordered left/right P1-B residues for lateral world axis 0
L_1, R_1 = ordered left/right P1-B residues for lateral world axis 1

D_m = E_m x (L_0 x R_0) x (L_1 x R_1)
N_m = |E_m| * |L_0| * |R_0| * |L_1| * |R_1|
```

`complete` means that the indexed domain denotes every element of `D_m`
exactly once under the declared inputs. It does not require allocating or
serializing `N_m` candidate objects. Equal numeric vectors with different
indices remain distinct because their provenance tuples differ.

Materialization means constructing a `LateralVectorCandidate` for one index or
a caller-bounded range. Materialization is not evidence of completeness, and
completeness does not imply materialization. No default code path may expand a
whole domain.

## Canonical factor records

All new records are frozen, slotted, self-validating, JSON-safe, and use
lowercase SHA-256 over canonical JSON with sorted keys, compact separators, and
`allow_nan=False`.

### `ResidueSideTable`

One endpoint of one lateral axis retains:

- endpoint role, frame, plane index, reading normal axis, local axis name and
  numeric tangent label;
- selected blocks-per-period value and source-region identity;
- exact `PeriodicResidueCandidates` record digest;
- the complete ordered candidate-value tuple;
- candidate count; and
- table digest.

Construction requires a non-refused P1-B record. Values are finite, in `[0,1)`,
strictly increasing, and byte-equal to the source record's candidate tuple.

### `LateralAxisFactor`

One of the two world-axis-ordered factors retains:

- world axis;
- left/right world signs;
- left/right local names and numeric tangent labels;
- right shift source, exactly `column` for right `u` and `row` for right `v`;
- one left and one right `ResidueSideTable`;
- exact C1 mapped-axis source digest;
- residue-pair count `|L| * |R|`; and
- factor digest.

The factor recomputes both `axis_transform` results and rejects the normal axis,
repeated axes, sign changes, endpoint changes, reordered labels, malformed
values, or a forged digest.

### `MatchLateralVectorDomain`

The canonical domain becomes an index record and retains once:

- exact match identity and unchanged `NormalFaceConstraint`;
- exact P1-D `MatchShiftDomain`, including the ordered joint-entry table;
- exact P1-B evidence for this match;
- the two selected C1 mapped-axis records, not the request-wide
  `C1AxisSupport` object;
- request-wide C1, P1-D-input, canonical-universe, and per-match P1-B digests;
- signature geometry/content and P1-F homography digests;
- finite comparison tolerance;
- homography, physical signed permutation, origins, and compatibility digest;
- two world-axis-ordered `LateralAxisFactor` records;
- joint-entry, per-axis pair, represented-candidate, and logical provenance
  counts;
- reconstruction schema/version and canonical ordering identifier;
- shared-provenance, candidate-product, and domain digests;
- `materialization_scope=index_only`;
- `completeness_scope=declared_inputs_only`; and
- `physical_truth_membership=unverified` plus the A3 conditional predicates.

It contains no `candidates` tuple and no full request-wide C1 support. The result
validator receives the current C1 support and recomputes its digest and selected
normal/axis projections.

### Candidate view

`LateralVectorCandidate`, `LateralAxisTerm`, and
`ResidueCandidateIdentity` remain the full immutable view of one product
element. `candidate_at(index)` reconstructs them from the factors. It must
produce the same formula replay, vector, provenance key, and candidate digest as
the eager prototype for every small-case candidate.

## Canonical index

Let:

```text
A_k = |L_k|
B_k = |R_k|
P_k = A_k * B_k
V   = P_0 * P_1
N   = |E| * V
```

For entry index `e` and residue indices `(iL0,iR0,iL1,iR1)`:

```text
q0 = iL0 * B_0 + iR0
q1 = iL1 * B_1 + iR1
g  = e * V + q0 * P_1 + q1
```

Inverse:

```text
e   = g // V
t   = g % V
q0  = t // P_1
q1  = t % P_1
iL0 = q0 // B_0
iR0 = q0 % B_0
iL1 = q1 // B_1
iR1 = q1 % B_1
```

All arithmetic uses exact integers. Valid indices satisfy `0 <= g < N`. This is
the eager order: entry `(row,column)`, then axis-0 left/right residue indices,
then axis-1 left/right residue indices. Entry `e` owns the contiguous range
`[e*V,(e+1)*V)`, so the two vector components can never use different entries.

The bijection is a theorem of the mixed-radix construction. Exhaustive small
tests and large-boundary arithmetic tests verify the implementation; sampling
is not accepted as correctness evidence.

## Candidate reconstruction

For each axis factor and index:

```text
k_R = entry.column_shift if right_local_axis == u
      else entry.row_shift

delta_A = sigma_R * k_R + sigma_L * r_L - sigma_R * r_R
```

The reconstructed term retains raw `k_R`, signed carry `sigma_R*k_R`, exact
left/right record digests and candidate indices/values, local/world mappings,
formula version, and replay digest. Both reconstructed terms receive the same
entry object. Reconstruction performs no score, ranking, confidence, dedup, or
selection operation.

Required APIs:

```text
build_lateral_vector_domains(
    support, p1d_result, matches, phase_evidence, *, readings, signatures,
    comparison_tolerance, budget
) -> C2RVectorDomainResult
validate_lateral_vector_domain_inputs(
    result, support, p1d_result, matches, phase_evidence, *, readings,
    signatures, comparison_tolerance
) -> bool
candidate_at(g) -> LateralVectorCandidate
provenance_key_at(g) -> exact provenance tuple
vector_at(g) -> two world-axis-labelled deltas
iter_candidates(start, stop) -> lazy full records
iter_views(start, stop) -> lazy compact provenance/vector views
iter_entry_groups() -> entry plus its contiguous candidate-index range
range_sha256(start, stop) -> explicit bounded audit digest
materialize_lateral_vector_domains(result, budget)
    -> tuple[tuple[LateralVectorCandidate, ...], ...]
```

Iterators are stateless, re-entrant, canonical, and use O(1) additional memory.
They yield duplicate numeric vectors separately. Invalid ranges fail closed.

## Product commitment

SHA-256 cannot commit to millions of arbitrary expanded leaf records without
O(N) hashing. The canonical commitment therefore binds the generator that
uniquely defines them.

Each side and axis factor has a digest. `shared_provenance_sha256` binds the
match identity, normal, selected C1 axes, P1-D replay, P1-B/universe/support
digests, geometry/content, P1-F mapping, tolerance, homography, permutation, and
origins. `candidate_product_sha256` binds:

```text
schema = c2r-indexed-vector-domain-v1
reconstruction_version = c2r-candidate-reconstruct-v1
formula_version = c2r-lateral-vector-v1
ordering = entry_row_major_axis0_left_right_axis1_left_right
P1-D replay digest and entry count
two ordered axis-factor digests and dimensions
represented candidate count
logical provenance-reference count
shared provenance digest
```

`domain_sha256` binds the complete index record. A full sequence digest is an
optional explicit O(N) audit and is never required for construction,
validation, or `complete`. `range_sha256` permits bounded spot artifacts but is
also not correctness evidence for unvisited indices.

## Work and budgets

The existing five counters remain logical represented-work limits:

```text
requested_match_identities = number of distinct requested identities
joint_entries = sum |E_m|
axis_residue_pairs = sum over entries and axes |L_k|*|R_k|
vector_candidates = sum N_m
provenance_references = 7*vector_candidates + requested_match_identities
```

Rename `consumed_work` to `represented_work` before acceptance. A complete
result requires `represented_work == planned_work`, independently recomputed
from the frozen factors. Budget exhaustion publishes planned work, zero
represented work, no domains, and no per-match refusals. Runtime factor/count
mismatch is atomic and returns `budget_exhausted/runtime_count_mismatch`.

Add `C2RMaterializationBudget` with explicit positive
`max_materialized_candidates` and `max_materialized_domains`. Only an explicit
`materialize_lateral_vector_domains` operation accepts it. The operation first
validates the complete indexed result and preflights the total requested domain
and candidate counts. Exact-limit succeeds. Any exceeded limit raises the typed
`C2RMaterializationBudgetExceeded` before constructing the first candidate; it
returns no result and leaves the indexed result unchanged. One-below for the
domain counter uses a request containing at least two domains. The operation
returns one candidate tuple per canonical input domain and does not create a
second domain-record type. It does not alter the logical five-counter result or
any persisted digest.

Report physical storage separately:

- stored P1-D entries;
- stored residue values;
- stored factor records;
- candidates materialized during construction, required to be zero; and
- peak traced/RSS deltas in performance tests.

## Validation

Production validation rebuilds the factor records from current C1, P1-D, P1-B,
readings, and signatures, then compares canonical index bytes and digests. It
recomputes rather than trusts caller-supplied copies of:

- canonical reading-universe digest;
- C1 support and selected source digests;
- P1-D input, replay, entries, radius, and content/geometry digests;
- P1-B record digests and complete candidate-value tuples;
- P1-F homography, signed permutation, origins, and compatibility predicate;
- all dimensions and five represented-work counters; and
- all factor/product/domain/result digests.

Constructor and `replace()` attacks must reject changed values, indices, signs,
entry order, axis order, formula version, mapping version, counts, and digests.
Identity-equal evidence with changed contents fails. Validation is exact over
the factors; it does not infer completeness from a sample of expanded records.

## C2-R-B interface

The graph-wide solver consumes factorized domains directly. It groups by the
P1-D entry range and streams compact candidate views. It calls `candidate_at`
only for a solution/refusal certificate that must be serialized. It must never
call full materialization.

Each match supplies a complete three-axis relative-translation candidate:

- the two indexed lateral deltas from one joint entry; and
- the unchanged normal-axis `NormalFaceConstraint.delta_translation`.

C2-R-B fixes the lexicographically first frame of each connected component to
zero as a deterministic mathematical gauge. No user, manifest, camera, F3,
coordinate, direction, bound, or reference input declares a gauge. The solver
propagates real-valued directed differences and checks cycle closure within an
explicit tolerance. It must not invent integer frame translations, split them
into assumed zero fractional phases, rank by residual, or select a candidate
merely because it is first.

## Acceptance tests

1. A test-local independent eager oracle covers every candidate byte, digest,
   vector, provenance key, and canonical position on small cases. The public
   pre-A4 eager domain class/path is removed rather than retained as a second
   production representation.
2. Exhaustive mixed-radix round trips cover all small dimensions; boundary and
   carry tests cover large exact integers.
3. Every P1-D entry and every residue index appears at the exact expected
   multiplicity. Cross-entry fabricated vectors are impossible.
4. Numeric-equal candidates retain distinct global indices and provenance.
5. Input permutation produces byte-identical factor/result records.
6. Exact-limit and one-below tests cover all five logical counters and both
   materialization counters.
7. No builder, validator, serializer, or digest path iterates `range(N)` or
   constructs a candidate during normal index construction.
8. The 289-candidate fixture produces zero stored/materialized candidates,
   serialized output <=256 KiB, and index-only traced peak <=4 MiB above the
   already-built P1-D/P1-B/C1 inputs.
9. An index-only reconstruction of the 2,195,063-candidate A3 census uses <=32
   MiB additional RSS and serializes <=8 MiB; it does not enumerate candidates.
10. Direct truth-index lookup on all A3 orientation/rational cases reconstructs
    the same truth-supporting candidates as the accepted independent proof.
11. Module-level compatibility refusals cover projective/unstable maps,
    mapping-version and provenance-digest mutations.
12. Full refusal taxonomy, malformed/nonfinite inputs, source mutations,
    constructor/replace attacks, and runtime-count atomicity pass.
13. The AST boundary prohibits image/camera/manifest/F3/bounds/reference/GT,
    evaluation, renderer, dataset, filesystem, network, process, selector,
    solver, placement, CLI, export, and world-writing access.
14. P1-B/C1/P1-D/P1-F/P1-G/graph/selector/frozen-runner/A3 proof regressions
    remain unchanged and pass. The combined command explicitly includes the
    A4 builder/index tests.

Performance limits may be tightened after the first factorized measurement;
they may not be relaxed merely to preserve an eager representation.

## Scope and exit

A4 changes only the isolated vector-domain representation and its tests. It
does not wire the runner, schema, graph solver, V2, renderer, evaluator, CLI, or
export; it does not open real fixtures or labels. Stop for lead and DeepSeek
review after the indexed builder passes the acceptance suite and publishes
hashes, counts, performance measurements, compile/diff checks, and protected
artifact hashes.

C2-R-B remains blocked until A4 is accepted. V2 remains blocked until C2-R-B is
accepted.
