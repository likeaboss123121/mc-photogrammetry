# M4 P1 C2-R-A coupled lateral-vector domain contract

Date: 2026-09-13  
Status: approved for proof and isolated implementation  
Dependencies: P1-B, P1-C1, and P1-D accepted; C2-A retained as `FORMULA_REJECT`

## Result required

C2-R-A must turn each accepted P1-D joint `(row, column)` shift and every compatible
pair of P1-B residue candidates into a coupled two-axis relative-translation vector.
It is a candidate-domain generator and analytic truth-membership proof. It does not
choose a shift, solve graph cycles, fix a world seam, place blocks, or change output.

The earlier scalar C2-B design is superseded. Its absolute equation failed because
the admitted top shift omitted an integer carry. C2-R-A may use the same expression
only once for every entry in the complete bounded P1-D domain and must preserve both
components under one entry identity.

## Scope boundary

Inputs are limited to:

- one accepted `C1AxisSupport`;
- the matching complete `ShiftDomainResult` from P1-D;
- the exact `HypothesisPhaseShiftEvidence` values from P1-B for those identities; and
- explicit caller-provided budgets and numeric comparison tolerance.

It must not read images, manifests, F3/debug coordinates, supplied cameras, scene
bounds, references, schematics, ground truth, evaluator labels, renderer output, CLI
state, export state, or frozen artifacts. It must not change P1-B, C1, P1-D,
`PlaneMatch`, graph/selector behavior, schema v1, scoring, ranking, or thresholds.

## Coordinate theorem to test

For one selected match and one lateral world axis `A`, derive exactly one left and
one right local tangent label that map to `A`:

```text
(A, sigma_L) = axis_transform(left_local_label, left_quadrant)
(A, sigma_R) = axis_transform(right_local_label, right_quadrant)
```

The local labels are the sorted complement of each reading normal. Local `u` is the
first label and local `v` the second. P1-D row is right `v`; column is right `u`.
Left and right local names may differ for the same world axis and must never be paired
by the strings `u` and `v` alone.

For P1-D entry `s`, let `k_R(A,s)` be its right-local component: column when the
right local axis for `A` is `u`, row when it is `v`. For residues `r_L` and `r_R`
belonging to the mapped left/right local axes, generate:

```text
delta_A(s, r_L, r_R)
  = sigma_R * k_R(A,s) + sigma_L * r_L - sigma_R * r_R
```

This is candidate generation, not a claim that any individual entry is true. The
two lateral components for one match form one vector and must use the same P1-D entry.
Taking one component from entry `s1` and the other from `s2` is forbidden unless
`s1 == s2`. Equal numeric vectors retain every supporting entry/residue provenance.

### Conditional truth-membership claim

C2-R-A may claim only:

```text
true lateral delta vector is in the generated domain
```

when all of these predicates hold:

1. C1 identity, absolute quadrants, axis transforms, selected `b` values, signature
   geometry, and homography provenance replay exactly.
2. The same physical plane lattice supplies both readings; their mapped tangent axes
   are the same two distinct world axes and exclude the normal axis.
3. Each P1-B candidate set contains the true local residue under its recorded
   endpoint-inclusive leading-coordinate convention.
4. One retained P1-D joint entry represents both true right-grid integer carries.
   It lies inside the declared radius and meets the copied P1-D
   `MIN_OVERLAP_CELLS` floor (currently nine).
5. The actual right-to-left cell homography and signature origins preserve the stated
   row/right-v, column/right-u correspondence for that truth entry; no transpose,
   endpoint reversal, or unmodelled cross-axis mapping is admitted.

P1-D proves completeness only inside its declared radius. C2-R-A must serialize that
radius and the conditional predicate. `complete` means every candidate inside those
declared inputs was retained; it does not prove that physical truth is inside them.
No status may be named `unique`, `solved`, `physical_truth`, or `placed`.
Every result must serialize `completeness_scope=declared_inputs_only` and
`physical_truth_membership=unverified`. The builder cannot inspect predicates 2-4;
the analytic proof establishes the conditional theorem and production consumers must
retain those labels.

## Analytic proof matrix

Before accepting the generator, construct truth independently of the candidate
formula, pass it through the actual signature-origin, cell-homography, and correlation
conventions, then require membership of the complete two-axis truth vector.

The matrix must cover:

- all three plane normals;
- all four absolute quadrants on each endpoint where the plane normals map compatibly;
- both camera sides;
- both possible left/right `u`/`v` world-axis pairings, including swapped local names;
- asymmetric nonzero row and column carries, positive and negative;
- nonzero and unequal signature origins;
- every supported rational `b` denominator represented by P1-B, including unequal
  left/right alternatives where the geometry is physically compatible;
- identity, affine, and finite projective homographies produced through the actual
  helper, with explicit rejection of incompatible or denominator-unstable cases; and
- boundary residues near zero and one without modulo reduction of the final delta.

For every valid case whose truth entry is inside the declared domain, assert that one
generated vector equals independently known truth within the explicit tolerance and
that its provenance identifies that single joint entry. If any valid case misses
truth, retain the fixture, report `FORMULA_REJECT`, and stop before publishing a
production-consumable domain builder.

The proof stage must publish a finite, deterministic, input-computable homography
compatibility predicate before the builder is implemented. The predicate must use only
the bound homography/signature geometry and must be independently tested against valid
and invalid matrix cases. If no such predicate separates the cases, report
`FORMULA_REJECT`; do not make compatibility an undocumented heuristic.

Negative controls must show that row/column transposition, either residue-sign change,
wrong endpoint order, pairing by local name instead of world axis, or mixing two joint
entries fails at least one asymmetric truth case. An incompatible homography must
produce a typed refusal, never a rounded candidate.

## Immutable records

Use a new isolated module with frozen, slotted, self-validating records and deterministic
JSON-safe `to_record()` output.

### `C2RVectorBudget`

Required positive integer limits, without defaults, for:

- requested match identities;
- P1-D joint entries examined;
- residue-pair combinations examined;
- lateral vector candidates buffered; and
- serialized provenance references buffered.

Booleans are invalid integers. Arithmetic is checked before enumeration.

### `LateralAxisTerm`

For one world axis, retain:

- world axis and left/right world signs;
- left/right local names and numeric labels;
- exact left/right P1-B residue-record identities and selected residue values;
- the right-local shift source (`row` or `column`) and signed integer component;
- finite `delta`; and
- exact formula/replay digest.

Construction recomputes both transforms, the selected shift component, and `delta`.

### `LateralVectorCandidate`

Retain:

- exact C1 match identity;
- one exact P1-D domain identity, replay digest, radius, content hashes, and complete
  joint `ShiftDomainEntry` including both row and column;
- exactly two `LateralAxisTerm` values on sorted distinct world axes;
- all four source residue identities and selected candidate indices/values;
- C1 signature-geometry and homography provenance; and
- a deterministic candidate digest.

The record rejects mixed P1-D entry identities, repeated/missing axes, a normal-axis
term, mismatched endpoint/b/quadrant provenance, recomputed delta mismatch, nonfinite
values, reordered fields, or a forged digest.

### `MatchLateralVectorDomain`

Retain one selected match identity, its normal `NormalFaceConstraint` unchanged, every
canonical lateral vector candidate, source counts, conditional truth-membership
predicate fields, and a domain digest. Candidates sort by joint `(row,column)`, then
world-axis ordered residue indices and exact source identities. Numerically equal
candidates are not collapsed when provenance differs.

### `C2RVectorRefusal` and `C2RVectorDomainResult`

Typed refusal reasons include:

```text
c1_status_invalid
p1d_status_invalid
requested_identity_missing
requested_identity_ambiguous
provenance_mismatch
axis_mapping_invalid
residue_evidence_missing
residue_evidence_ambiguous
residue_candidate_invalid
joint_shift_invalid
homography_incompatible
formula_replay_mismatch
budget_exhausted
```

Result status is exactly `complete`, `refused`, or `budget_exhausted`. `complete`
requires one domain for every distinct requested C1 mapped match identity and no
refusals. Either failure status contains no domains. `budget_exhausted` contains no
partial per-match refusals. `refused` may contain sorted refusals only after every
requested identity was examined within budget.

## Work accounting

Preflight the entire request before constructing candidates:

```text
joint_entries = sum(len(match_domain.entries))
axis_residue_pairs = sum over entries and both axes A of
                     len(left_residues[A]) * len(right_residues[A])
vector_candidates = sum over entries s of
                    (len(left_residues[A0]) * len(right_residues[A0]))
                  * (len(left_residues[A1]) * len(right_residues[A1]))
provenance_references = 7 * vector_candidates + requested_match_identities
```

The final vector count uses the Cartesian product of residue pairs between the two
axes for one fixed joint shift. It never takes a Cartesian product between P1-D shifts.
The seven per-vector references are one joint P1-D entry, two C1 mapped-axis sources,
and four endpoint/axis residue selections; the per-match reference is the unchanged
normal constraint. Domain-level hashes are scalar fields, not additional references.
If any plan exceeds its caller limit, return `budget_exhausted` with planned counters
and no candidates. A runtime count mismatch also discards all candidates. No truncation,
top-N, score, correlation value, confidence, or arbitrary candidate cap is allowed.

## Decisive tests

1. Retained C2-A counterexample: top `(0,+1)` gives `v=-0.2`; non-top `(+1,+1)`
   produces the truth vector `(u,v)=(1.4,0.8)` under one entry identity.
2. A deliberately cross-shaped domain has the true `u` component on one entry and true
   `v` component on another; the fabricated mixed vector is absent.
3. Literal all-normal/all-quadrant cases map left/right residues by common world axis,
   including a case where left `u` pairs with right `v`.
4. Asymmetric `np.roll(base,(3,-4)) -> (-3,+4)` pins row/right-v and column/right-u;
   sign, endpoint, and transpose mutants miss independently known truth.
5. Every supported P1-B rational candidate participates; duplicate numeric deltas keep
   distinct provenance; missing phase never becomes zero.
6. Nonzero/unequal origins and actual finite homographies pass only when the declared
   predicate holds. Projective denominator failure, cross-axis incompatibility, or
   digest mutation refuses.
7. Removing the known truth joint entry or placing it outside the radius prevents a
   truth-membership claim; the result never implies physical completeness.
8. Exact-limit and one-below budget tests cover every counter. Exhaustion yields no
   domains or partial refusals.
9. Input permutation produces byte-identical records. Duplicate/foreign identities,
   forged candidates, changed residue indices, mixed joint entries, and `replace()`
   attacks fail closed.
10. P1-B/C1/P1-D/C2-A/graph/selector/frozen-runner tests remain unchanged and pass.
    AST/import checks enforce the metadata, evaluation, renderer, export, and placement
    boundary.

## Exit and next dependency

Publish the analytic matrix result, representative record hashes, focused and combined
tests, `py_compile`, whitespace/diff checks, and protected-file hashes. Stop for lead
review before cycle solving, V2, schema/artifact/evaluation/CLI/renderer/export wiring,
or any real-fixture score run.

If C2-R-A passes, C2-R-B may solve graph-wide coupled vector domains with normal-face
constraints and integer gauges. It must preserve joint-entry choices across axes and
report ambiguity, contradiction, disconnection, or budget exhaustion. V2 remains
blocked until C2-R-B is accepted.
