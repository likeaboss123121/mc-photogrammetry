# M4 P1 C2-R-A2 post-P1-E proof and domain-builder contract

Date: 2026-09-13  
Status: assigned after lead acceptance of the immutable R5 diagnostic  
Owner: Luna  
Depends on: accepted P1-B, P1-C1, P1-D, and P1-E

## Scope

This reopens C2-R-A after P1-E removed the camera-composed grid-map defect that
caused the retained `FORMULA_REJECT`. The complete requirements, records, work
formulas, refusal vocabulary, boundaries, and ten decisive-test groups in
[`M4-P1-C2R-vector-domain-contract-2026-09-13.md`](M4-P1-C2R-vector-domain-contract-2026-09-13.md)
remain binding. This document states only the post-P1-E delta.

C2-R-A2 must complete the independent analytic matrix before publishing a
builder. If any valid case still loses the independently known two-axis truth
vector, retain that fixture, report `FORMULA_REJECT`, and stop. Passing the
previous unequal-distance fixture alone is necessary and insufficient.

No frozen-runner, solver, selector, schema, CLI, evaluator, renderer, export,
dataset, or real-fixture change belongs in this task. R5 is diagnostic context,
not proof input. Do not use its solution, labels, capture metadata, or inferred
translations to tune this work.

## P1-E premise to prove

For every compatible pair, `cell_grid_homography` maps right signature indices
to left signature indices with the exact quadrant-derived tangent signed
permutation and `P @ origin_R - origin_L`. Its linear part must remain invariant
to camera distance, focal length, principal point, rectification centre,
sampling count, and camera side. Those values remain bound measurement
provenance; none may rescale a one-cell-per-block grid.

The proof must construct world texture and camera observations independently,
then pass them through the actual signature, homography, warp, correlation,
P1-B, C1, and P1-D paths. Expected shifts and translations may not be calculated
with the formula under test.

The retained unequal-distance case must now establish all of the following:

- actual and independently constructed grid maps are identical;
- the actual correlation peak is `(0, 1)` and the complete P1-D domain contains
  the independently true joint carry `(1, 1)`;
- one candidate derived from that single entry contains the independently true
  lateral vector `(1.4, 0.8)`; and
- top-only `(0, 1)` remains insufficient, proving why the complete domain is
  required.

The proof oracle must derive or hard-code the expected vector from independently
declared world translations. It may not compute the expected value from a P1-D
entry and P1-B residues. Include swapped-local-name cases with nonzero asymmetric
carries where endpoint world signs are equal and where they differ; exercise the
sign on the carry itself, not only residue lookup.

P1-E intentionally permits only affine signed-permutation/origin grid maps. The
original request for a finite non-affine projective map "produced through the
actual helper" is superseded as impossible under the repaired coordinate model.
Direct projective matrices remain negative compatibility/digest controls and
must refuse; they are not valid outputs of `cell_grid_homography`.

## Identity and input requirements

Use exact `C1MatchIdentity` and accepted P1-D request fingerprints. Resolve the
two endpoint readings and signatures by the same ordered identity and canonical
`b` definitions used by C1/P1-D. Reject ambiguous, foreign, reordered, or
missing identities. Numeric plane indices alone are not evidence.

Each lateral component must bind a P1-B residue record for the endpoint/local
axis that maps to the common world axis. Both components of one vector must bind
the same complete `ShiftDomainEntry`; cross-entry component mixing is invalid.
Retain equal numeric vectors when their entry or residue provenance differs.

The implementation must not infer absent `phase_evidence`, substitute legacy
`phase_pixels`, convert a missing residue to zero, or rebuild evidence from a
schema-v1 freeze. Production and frozen schema-v1 currently omit evidence needed
for that replay; wiring is a later task.

Bind a canonical digest of every exact P1-B evidence record used, including its
source reading/axis/b identity and candidate ordering. Identity-equal evidence
with different values must fail. Candidate construction must cross-check that
both C1 mapped-axis geometry/grid-map digests equal the corresponding P1-D
domain digests and that the selected entry meets the domain's copied overlap
floor.

Store the raw right-grid row/column carry and the derived signed world integer as
distinct fields. Recompute the latter exactly once as `sigma_R * raw_carry`;
validation must reject double-signing or a forged signed value.

## Required result

After the complete matrix passes, implement the immutable records and bounded
builder specified by the original contract in a new isolated reconstruction
module. Result status remains exactly `complete`, `refused`, or
`budget_exhausted`. Every complete result must state:

```text
completeness_scope = declared_inputs_only
physical_truth_membership = unverified
```

No record or status may claim a selected, unique, solved, absolute, world, or
block placement. The builder enumerates the complete declared candidate domain;
it does not rank it.

Preflight and runtime accounting must reproduce the original exact formulas:

```text
requested_match_identities = len(distinct sorted requested identities)
joint_entries = sum(len(match_domain.entries))
axis_residue_pairs = sum over entries and both lateral axes of
                     len(left_residues) * len(right_residues)
vector_candidates = sum over entries of
                    pairs_for_axis_0 * pairs_for_axis_1
provenance_references = 7 * vector_candidates + requested_match_identities
```

Any exceeded or mismatched counter discards all candidate output. No truncation,
top-N selection, score threshold, or silent deduplication is allowed.
The requested-identity count has its own budget comparison and exact-limit/
one-below tests; it is not satisfied merely by its appearance in the provenance
formula. All five planned counters must equal independently accumulated runtime
counters.

## Additional decisive controls

In addition to the original ten groups:

1. Exercise the literal P1-E signed-permutation table for all three normals and
   all four relative quadrants inside the end-to-end proof, not only by directly
   testing the helper.
2. Vary distance, focal, principal, centre, samples, and side independently;
   expected grid correspondence and truth membership remain unchanged while
   provenance digests change where specified.
3. Include at least one valid cyclic texture construction and one disconnected
   pair set. C2-R-A2 emits domains for both; it must not pretend to solve either.
4. Mutate each numeric plane index while retaining conflicting reading content;
   identity validation must refuse before formula evaluation.
5. Preserve the historical rejection report and its pre-repair measurements
   unchanged. The retained proof test may evolve only to assert both the old
   counterexample and the corrected current behavior.
6. Add AST/import controls that prohibit frozen artifacts, R5 paths, metadata,
   evaluator, renderer, CLI, export, placement, and filesystem/network/process
   access from the new module. Existing tests alone do not establish this.

## Validation and stop point

Publish matrix counts, representative record hashes, exact planned/actual work
counters, focused and combined test results, compile/import checks, diff checks,
and protected hashes. Re-run the accepted P1-B/C1/P1-D/P1-E and graph/selector
regressions. Do not run R5 again or open evaluator labels.

Stop for lead review after C2-R-A2. C2-R-B will be a separate graph-wide solver
over these complete coupled domains. It must require independent cycle/redundancy
evidence and bounded translations; the legacy `solve_component` empty reason is
not an acceptance oracle.
