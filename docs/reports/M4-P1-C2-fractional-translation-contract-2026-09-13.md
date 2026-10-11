# M4 P1-C2 fractional translation and cycle-consistency contract

Date: 2026-09-13  
Status: C2-A formula rejected by retained counterexample; P1-D shift-domain redesign required  
Dependency: accepted P1-B residue candidates and P1-C1 axis/support records

## Boundary

C2 converts C1 integer lateral evidence plus P1-B phase candidates into complete
relative-translation candidate domains, then solves per-axis cycle consistency without
choosing an absolute seam. It does not place blocks, select materials, mutate the graph
solution, or enter the production runner. V2 remains blocked until C2 is accepted.

Inputs are only the accepted `C1AxisSupport`, its exact accepted `ComponentSolution`,
ordered readings, exact signatures, admitted matches, and a caller-supplied tolerance and
work budget. Images, manifests, F3/debug coordinates, directions, supplied cameras,
supplied bounds, references, schematics, ground truth, evaluator labels, CLI, renderer,
export, and placement are forbidden.

## Coordinate model and unproved lemma

For frame `f`, world axis `A`, corresponding local coordinate `p_f`, and public
`axis_transform` sign `sigma_f`:

```text
world_coordinate[A] = t_f[A] + sigma_f * p_f
```

P1-B supplies every possible residue `r_f = p_f mod 1`. C1 supplies the right-grid
integer shift `k_R`, its mapped world axis, and `sigma_R`, under:

```text
warped_left[x + k_R] ~= right[x]
cell_shift_rc = (row, column) = (v, u)
```

The candidate equation to prove is:

```text
t_R[A] - t_L[A]
  = sigma_R * k_R + sigma_L * r_L - sigma_R * r_R
```

The missing lemma was that the actual right-to-left cell homography, signature origins,
axis correspondence, and correlation convention make `sigma_R*k_R` the required integer
part of `sigma_L*N_L - sigma_R*N_R`. C2-A disproved that lemma.

## C2-A result: absolute equation rejected

The first valid independent truth fixture uses `world=t+p`, a fixed asymmetric world
texture, analytically derived local residue `-t mod 1`, exact signatures, the real
right-to-left homography, and the real correlation peak. For the `v` axis:

```text
known t_R - t_L = +0.8
admitted top integer shift = 0
left residue - right residue = -0.2
proposed full candidate = -0.2
```

The candidate and truth differ by exactly one block. The same fixture's `u` axis happens
to use lift zero; that does not repair the theorem. The surviving statement is only:

```text
t_R[A] - t_L[A]
  congruent to sigma_R*k_R + sigma_L*r_L - sigma_R*r_R  (mod 1)
```

The top correlation peak cannot select the integer carry. C2-B as specified below is
superseded and must not be implemented from one stored `cell_shift_rc`.

## P1-D prerequisite: complete bounded shift domain

Before C2 resumes, preserve the complete finite integer-shift ambiguity for each exact
selected match/signature/homography identity. Recompute the same bounded 2D correlation
surface without changing `PlaneMatch`, graph admission, ranking, selector behavior, or
frozen artifacts. Retain every `(row,column)` shift within an explicit caller-supplied
`max_shift_cells` whose shifted valid coverage satisfies the existing
`MIN_OVERLAP_CELLS`; retain its correlation value and overlap diagnostics. Apply no new
score, peak, top-N, margin, PSR, confidence, or ground-truth threshold.

The search radius and finite signature overlap bound only the declared search universe;
they neither prove that universe contains physical truth nor make one lift correct. The
active P1-D contract is
[`M4-P1-D-complete-shift-domain-contract-2026-09-13.md`](M4-P1-D-complete-shift-domain-contract-2026-09-13.md).
C2-R must later combine every retained joint 2D shift with every compatible residue pair.
Normal metric constraints and graph cycles may eliminate lifts. A tree or weak graph may
remain ambiguous or exhaust its explicit work budget.

## Superseded C2-A matrix (do not implement)

Implement an isolated analytic proof harness before adding general C2 records or search.
Construct world-truth planar lattice fields independently of the proposed equation, then
generate exact `PlaneReading`/`PlaneSignature` pairs and use the current
`_cell_homography`, `warp_signature`, correlation surface, and peak convention.

The frozen matrix covers:

- all three plane normals and every absolute quadrant pair compatible with one world axis;
- equal and opposite local signs;
- both tangent directions, proving column=`u` and row=`v`;
- non-square grids, unequal positive/negative integer `origin_uv`, nonzero fractional
  leading coordinates, and asymmetric shifts including `(-3,+4)`;
- zero, positive, negative, and boundary shifts within the declared search limit;
- integer and fractional `b`, including at least `1`, `1/2`, and `1/16`;
- off-centre principal points, unequal finite distances, and asymmetric finite
  homographies that still represent the same physical plane;
- a unique non-periodic fiducial superposed on periodic energy so the correlation peak is
  not selected by repetition or a tie.

For every case, assert independently known `t_R[A]-t_L[A]` is present in the candidate
set within an explicit test tolerance, while reversing the shift sign, either residue
sign, row/column assignment, or left/right endpoint fails. The test must also verify the
stored C1 homography and signature-geometry digests before using evidence.

Add negative controls with projective variation, cross-axis leakage, non-unit scale, an
unstable homogeneous denominator, digest mismatch, or incompatible origins. Define the
minimal finite homography-consistency predicate from the passing truth cases. Such inputs
must return typed `homography_incompatible` or `provenance_mismatch`; they must never be
rounded into a candidate.

If any valid truth case disproves the equation, retain the failing fixture/report, stop,
and request lead review. Do not tune signs case by case or proceed to C2-B.

## Superseded C2-B design (do not implement)

Use frozen immutable records with strict `to_record()` validation:

- `C2Budget`: required positive limits for edge candidates, search nodes, assignments,
  cycle checks, and buffered records; no production defaults.
- `LateralTranslationCandidate`: full C1 match/axis identity, left/right local axes and
  signs, exact residue-record identities and selected residues, copied integer evidence,
  finite full delta, signature/homography digests, and support provenance.
- `ConstraintDomain`: one frame pair/world axis with either the singleton normal-face
  delta or every lateral candidate; evidence classes remain distinct.
- `AxisTranslationAssignment`: component/world-axis identity, gauge anchor fixed to zero,
  sorted finite per-frame translations, and exact chosen candidate identities.
- `AxisTranslationResult`: C1 component identity, complete retained assignments and count,
  diagnostics, work counters, and typed status.
- `C2TranslationSupport`: ordered per-axis results plus top-level input/provenance digest.

No record contains a chosen absolute world coordinate, block seam, period index, placed
cell, material/state, score-derived winner, or silent zero for unavailable evidence.

## Candidate generation

For each non-refused C1 mapped lateral record:

1. Resolve exactly one left local axis and one right local axis mapping to its world axis.
2. Bind the exact P1-B records for those endpoints, original b pair, plane indices, and
   selected canonical alternatives.
3. Require available residue sets on both endpoints. Missing phase yields
   `phase_unavailable`; it never becomes residue zero.
4. Recompute exact signature-geometry and normalized right-to-left homography digests and
   require byte equality with C1.
5. Apply the accepted C2-A homography predicate.
6. Enumerate the full Cartesian product of left/right residue candidates and compute the
   proved equation without modulo reduction.
7. Retain candidate-pair provenance even when two pairs produce equal deltas. Equality or
   tolerance may prune search states only if the full provenance set remains serialized.

Candidate order is the C1 match key, world axis, left residue, right residue, then exact
provenance key. Confidence, PSR, support, and concentration are diagnostics only.

## Axis constraint solve

Normal-face C1 constraints are singleton domains using their recorded equation. Lateral
constraints use all admitted C2 candidates. For each C1 axis component:

- set its deterministic gauge anchor translation to exactly `0.0`;
- solve `t_right - t_left = delta` by deterministic constraint propagation and DFS;
- test every already-bound edge within caller-supplied `translation_tolerance_blocks`;
- retain every consistent assignment; do not rank or select by score;
- deduplicate numerically equivalent assignments only within the supplied tolerance while
  preserving all supporting candidate provenance;
- solve disconnected components under their own anchors, but keep
  `disconnected_from_global_gauge` and forbid V2 use;
- return `contradiction` when exhaustive bounded search finds no assignment;
- return `ambiguous` for more than one assignment and `unique` for exactly one;
- on any budget limit, return `budget_exhausted` with counters and no partial assignment;
- preserve `no_constraints`, `phase_unavailable`, `homography_incompatible`, and
  `provenance_mismatch` as typed abstentions.

All tolerances and budgets are caller-supplied finite validated inputs. C2 introduces no
hidden default, threshold fitting, beam search, approximate winner, or partial success.

## Decisive tests

1. C2-A truth matrix and all sign/origin/homography counterfactuals.
2. Every P1-B rational candidate pair is retained; no modulo collapse or selected seam.
3. Single edge, chain, consistent cycle, contradictory cycle, and multi-solution cycle.
4. Normal-only, lateral-only, and mixed constraints reproduce their exact equations.
5. Tree ambiguity remains complete; candidate/provenance multiplicity is not discarded.
6. Disconnected axes receive independent zero gauges and remain unusable by V2.
7. Missing phase, refused shift, nonfinite values, foreign signatures, digest mismatch,
   duplicate identities, and malformed C1 records fail closed.
8. Input permutation produces byte-identical records, counters, and assignment order.
9. Each budget boundary is tested at limit and limit-minus-one; exhaustion has no partial
   assignment.
10. Direct construction and `dataclasses.replace` cannot forge deltas, gauges,
    connectivity, candidate choices, provenance, counts, status, or sorting.
11. Existing C1/P1-B/graph/selector/frozen-runner outputs and protected hashes are
    unchanged.
12. AST/import/output checks enforce the production/evaluation/manual-metadata boundary.

## Exit criteria

C2-A must first publish the analytic fixtures, exact equation verdict, homography predicate,
focused tests, and hashes. After lead acceptance, C2-B may implement the bounded records and
solver above. Final C2 acceptance requires all decisive tests, focused regressions,
`py_compile`, diff checks, stable representative hashes, and no artifact/evaluation/schema/
CLI/renderer/export/placement change. V2 starts only after that review.

## C2-A first valid fixture result — FORMULA_REJECT

The required first world-truth baseline disproves the proposed equation as an absolute
translation equality. Per lead review, the fixture is valid and this is the stopping
counterexample; no matrix expansion or production change was made.

The independently constructed continuous field uses
`world = t + p`, with `t_left=(0.2,-0.35)` and `t_right=(1.6,0.45)`. Its texture combines
two incommensurate smooth frequencies and two asymmetric Gaussian fiducials. The exact
synthetic readings use plane axis 2, quadrant pair `(0,0)`, `b=1`, distance 1, equal
off-center principal points `(31,27)`, equal extent/sample grid, and zero signature
origins. The generated signature cells are direct samples of that independent field.
After binding the admitted `match_plane_pair` result to plane indices `(0,0)`, C1 verifies
the signature-geometry and normalized homography digests. The actual homography is the
identity in this baseline; it is a sign/convention counterexample, not evidence about
asymmetric homographies.

The real correlation path returns `cell_shift_rc=(0,1)` with PSR `16.6173834749` and
distinct peak `0.3840701247` versus second peak `0.1674523034`. In `(u,v)` order, C1 maps
integer evidence `(1,0)`. P1-B supplies left residues `(.8,.35)` and right residues
`(.4,.55)`. Therefore the proposed expression gives `(1.4,-0.2)`, while independent
world truth is `(1.4,0.8)`. The `v` candidate is off by exactly one block. The test
`tests/test_plane_c2_proof.py::test_world_truth_equation_has_integer_lift_counterexample`
asserts this counterexample and retains the actual correlation result.

For this `b=1` case the phase terms establish only a modulo-one relation on `v`: `-0.2`
and `0.8` are congruent. A bounded correlation search can produce a finite candidate
domain, but it cannot make one integer lift true by itself. This fixture demonstrates
why retaining only the top peak is unsafe: at the same exact `max_shift_cells=8`, the
existing surface's top `(row,column)=(0,1)` has value `0.3840701247`; the alternative
`(1,1)` has lower value `0.1965485062` but 1,521 valid overlapping cells (well above the
existing minimum of 9). Using that full shift gives `(1.4,0.8)`, including the independent
truth. Both shifts must remain alternatives. This does not establish the formula for
general `b`, homographies, or quadrants, nor prove that a chosen search radius contains
the true shift in every capture.

### P1-D feasibility note (analysis only; no implementation)

An additive evidence domain appears feasible without changing `PlaneMatch` or graph
selection. For each exact C1-admitted match, bind the same ordered readings, selected
signatures, signature content/coverage digests, and normalized homography. Require an
explicit `max_shift_cells`; recompute the existing `_cell_homography`, `warp_signature`,
and `_correlation_surface` exactly, then require the recomputed top shift/peak/PSR/second
peak/base overlap to agree with the admitted `PlaneMatch`. A mismatch is typed refusal.
Enumerate every integer `(row,column)` in the bounded square, compute its geometric
valid-overlap count with the existing correlation convention
`warped_left[y+row,x+column] & right[y,x]`, and retain all shifts with at least the
existing `MIN_OVERLAP_CELLS=9`; attach the unchanged surface value, support count, radius,
and match/signature/homography identities. Add no per-shift score/PSR threshold and no
local winner. `PlaneMatch`, `solve_component`, and frozen runner output stay unchanged.

The current frozen runner pins `max_shift_cells=32`, giving at most 4,225 shifts per
admitted edge; the proof fixture uses radius 8 and all 289 shifts meet support because
its 40x40 masks are full. Bounded mask cross-correlation can compute per-shift support
in FFT-sized work, with small exact slice-count tests validating sign and counts; C2's
existing caller budgets must still produce whole-result exhaustion rather than a
partial domain. Preserve each 2D `(row,column)` pair as one candidate identity: solving
the u and v projections independently must not combine coordinates from different
shift-pair hypotheses. The finite algorithmic radius gives a finite list, not a unique
truth. Any completeness claim also needs an image-derived reason the true shift lies
inside that radius; global cycle/normal constraints may prune candidates, but no local
peak may silently discard alternatives. This is design analysis only, pending lead
revision; no P1-D code or matrix expansion was started.

Focused command: `.\.venv\Scripts\python.exe -m pytest -q tests/test_plane_c2_proof.py` -
1 passed after encoding the counterexample as a passing refutation test. The initial
unmodified assertion failed exactly at `v`: candidate `-0.20000000000000007`, truth
`0.8`. No source/evaluator/runner/artifact changes, no broader tests, and no C2-B work.
