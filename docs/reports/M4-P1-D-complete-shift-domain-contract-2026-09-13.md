# M4 P1-D complete bounded shift-domain contract

Date: 2026-09-13  
Status: approved for implementation  
Dependency: P1-C1 accepted; C2-A retained as `FORMULA_REJECT`

## Purpose and boundary

C2-A proves that one admitted correlation peak cannot recover the integer carry of a
relative lateral translation. P1-D therefore retains every joint integer shift supported
by the exact correlation surface inside one explicitly declared search radius. It is an
additive evidence-retention layer, not a selector or reconstruction stage.

P1-D may consume only the accepted C1 support, its exact admitted `PlaneMatch` objects,
and the exact `PlaneSignature` objects used to form them. It must not read images,
manifests, F3/debug coordinates, supplied cameras, bounds, references, schematics,
ground truth, evaluator labels, renderer output, CLI state, or export state. It must not
change `PlaneMatch`, graph admission/ranking, joint-selector behavior, frozen schema-v1
records, or protected artifacts. C2-R and V2 remain blocked until P1-D is accepted.

## Exact shift and support convention

For the exact selected signature pair, recompute the existing right-grid-to-left-grid
homography and existing warped signature. Let:

```text
I = warped_left_coverage AND right_coverage
warped_left[row + row_shift, column + column_shift]
    ~= right[row, column]
```

The public shift order is exactly `(row, column) = (v, u)`. For shift `s=(sr,sc)`,
the support used by the current fixed-intersection correlation maps is:

```text
S(s) = {(r,c): I[r,c] AND I[r+sr,c+sc], both indices in bounds}
```

Retain a shift iff `abs(sr) <= max_shift_cells`, `abs(sc) <= max_shift_cells`, and
`count(S(s)) >= plane_graph.MIN_OVERLAP_CELLS`. Equality at the existing nine-cell
floor is retained. Count `I` shifted against itself: using the raw left/right coverage
masks separately would claim support from samples that the current pre-intersection
whitening zeroed and therefore did not correlate.

Recompute the correlation values with the existing `_correlation_surface` primitive,
unchanged. Its maps remain mean-subtracted on fixed `I` and zero elsewhere. Each surface
value is retained as a finite diagnostic; it must never rank, prune, select, or change the
domain. No top-N, local-maximum, margin, PSR, confidence, score, or label threshold is
permitted. The domain is the full row-major enumeration of all support-admissible shifts.

The stored `PlaneMatch.cell_shift_rc`, peak, second peak, PSR, overlap and score remain
unchanged. P1-D records whether that admitted peak belongs to the support-admissible
domain, but gives it no privileged downstream meaning.

## Required immutable records

Implement an isolated additive module with frozen, slotted, self-validating records:

- `ShiftDomainBudget`: required positive integer limits for domains, padded FFT cells,
  enumerated shift positions, overlap-cell visits, and buffered entries. It has no
  defaults; booleans are invalid integers.
- `ShiftDomainEntry`: signed row/column shift, finite surface value, positive shifted
  overlap count, and finite shifted-overlap fraction using the existing
  `min(count(warped_coverage), count(right_coverage))` denominator.
- `MatchShiftDomain`: exact C1 match identity; declared `max_shift_cells`; right-grid
  shape; the exact `MIN_OVERLAP_CELLS` value; fixed whitening-overlap count; admitted
  top shift/surface/support diagnostics; canonical entry
  tuple; C1 geometry/homography digests; new hashes of both complete signature contents;
  and an exact replay digest.
- `ShiftDomainRefusal`: exact match identity where available, typed reason, and bounded
  non-evidentiary detail.
- `ShiftDomainResult`: canonical domains/refusals, copied budget, planned and consumed
  work counters, input digest, and status in `complete`, `refused`, or
  `budget_exhausted`. `complete` requires exactly one domain for every distinct C1 mapped
  match identity and no refusals. Either failure status requires no domains;
  `budget_exhausted` carries no partially gathered per-match refusals.

Signature-content hashes must bind dtype, shape, bytes of cells/interior/coverage,
`origin_uv`, canonical `b`, and ordered reading identity. C1 geometry and normalized
homography digests must be recomputed and byte-equal to the selected mapped-axis records.
Both mapped axes for one match must share identical match/signature/homography provenance.

The declared radius is replay provenance because legacy `PlaneMatch` does not store its
original radius. Before emitting a domain, recompute `_peak_statistics` and all stored
match diagnostics from that radius and require exact equality. Finite values compare
exactly; PSR replays only when both values are the same finite value or both are positive
infinity. Do not serialize a non-finite PSR: bind the full replay through a canonical
tagged replay digest, while public numeric record fields remain finite. A mismatch returns
`correlation_replay_mismatch`; it must not be rounded or tolerated. The emitted record
proves complete retention for that declared radius only; it never claims the radius
contains the physical truth.

`to_record()` must be deterministic JSON data with no NumPy objects or non-finite values.
Entries sort by `(row_shift, column_shift)`; domains sort by C1 match identity. Direct
construction and `dataclasses.replace` must reject forged identities, digests, counts,
ranges, top diagnostics, ordering, duplicates, work counters, or status.

## Work accounting and failure semantics

Before allocating a correlation surface, preflight the entire requested selected-match
set. Account at least:

```text
padded_fft_cells = sum((rows + 2*M) * (columns + 2*M))
shift_positions = sum((2*M + 1)^2)
overlap_cell_visits = sum over shifts of each in-bounds comparison rectangle area
maximum_buffered_entries = shift_positions
```

Compute overlap counts through those clipped rectangles so the planned visit total is the
exact logical count, rather than a lower estimate for full-array scans. All arithmetic is
checked non-negative integer arithmetic. If any planned total exceeds its explicit
budget, return no completed domains. Budget failure status is `budget_exhausted`, with
planned/consumed counters and no partial entries. Do not clamp a radius, truncate a
domain, keep already completed matches, or substitute the admitted peak.

After preflight, any runtime budget discrepancy or incomplete enumeration returns
`budget_exhausted` with no domains. Any non-finite surface, replay/provenance/input
mismatch, or other evidence failure returns `refused` with no domains; sorted per-match
refusals may be retained only after every requested identity was examined within budget.
Never mix a successful domain with a refusal. Existing C1 lateral refusals are outside the
requested P1-D set; a complete result covers every distinct identity represented by C1's
mapped `u/v` pair. Downstream C2 must abstain unless P1-D status is `complete`.

Typed reasons are:

```text
budget_exhausted
selected_identity_missing
selected_identity_ambiguous
signature_missing
signature_ambiguous
signature_identity_mismatch
signature_content_invalid
homography_invalid
provenance_mismatch
correlation_replay_mismatch
correlation_nonfinite
shift_domain_incomplete
```

Do not add a residue/phase refusal: P1-D is independent of P1-B residue availability.

## Decisive tests

1. Extend the retained C2-A world-truth fixture. The admitted top is `(0,+1)`, while the
   integer carry that restores both axes is joint shift `(+1,+1)`. Assert `(+1,+1)` is
   retained with at least nine support cells, is not the top, and changes the `v` full
   candidate from `-0.2` to `+0.8`; preserve the top and original refutation unchanged.
2. Compare every emitted shift/count/value against an independent brute-force reference
   for square and non-square grids, holes, unequal coverages, nonzero origins, and all
   zero/positive/negative/boundary shifts. No candidate unions or axis-wise products.
3. Pin the support convention: exact nine cells retain, eight refuse; raw-coverage cells
   absent from fixed `I` never count.
4. Pin `np.roll(base,(3,-4)) -> cell_shift_rc=(-3,+4)` and verify row=`v`, column=`u`.
5. Verify declared-radius replay at the correct radius and refusal at a radius whose
   peak statistics differ. Radius zero, negative/bool/huge inputs, empty/non-finite maps,
   and malformed homographies fail closed.
6. Exercise every budget exactly at its limit and one below. Exhaustion returns counters
   and no domains or partial entry tuples.
7. Input/signature permutation yields byte-identical output. Duplicate/foreign matches,
   signatures, C1 axes, digest changes, array mutation, reordered entries, forged top
   fields, and invalid counters are rejected.
8. Existing C2-A, C1, P1-B, graph, selector and frozen-runner tests remain unchanged and
   pass. Exact `PlaneMatch`, selector, schema-v1 and protected artifact records remain
   byte-identical. AST/import checks enforce the manual-metadata/evaluation boundary.

## Exit and next dependency

Publish the implementation report, representative record hashes, focused regressions,
`py_compile`, diff checks, and protected-artifact/hash verification, then stop for lead
review. Do not run evaluation or implement C2-R/V2.

P1-D produces only a complete finite superset relative to a declared radius. It cannot
certify the physical integer carry, justify that the radius contains truth, or make a
weak/tree graph unique. After acceptance, C2-R may combine each retained joint shift with
all compatible P1-B residue pairs and let independent normal constraints and graph cycles
prune exact assignments. Ambiguity and budget exhaustion remain valid outcomes.
