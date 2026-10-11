# M4 P1-B contract — hypothesis-scoped phase candidates and relative shift

Date: 2026-09-13
Status: approved design after lead mathematical correction; implementation pending
Source: DeepSeek read-only/no-shell audit, corrected and reviewed by GPT-6 Astra

## Boundary

P1-B preserves two evidence classes under one exact match/b-pair hypothesis:

1. Per-reading, per-local-axis periodic-energy block-residue **candidate sets**.
2. The admitted match's unchanged relative integer cell shift and support.

P1-B does not fuse these into a cross-view offset. The two rectified windows do
not yet share a solved lateral translation frame. Adding an integer shift
inside a modulo-one phase expression also cancels the shift. Cross-view
consistency and translation support therefore belong to P1-C.

No P1-B value is a physical block seam, integer period selection, absolute
translation, world coordinate, lateral origin, face coordinate, or placement.
Inputs are only one `PlaneMatch` and its two `PlaneReading` objects. Images,
manifests, F3/debug values, coordinates, bounds, references, ground truth, and
manual pose metadata are forbidden.

## Proposed types

Add a pure `reconstruction/plane_residue.py` module with frozen types:

- `PhaseAxisQuality`: response magnitude, profile mass, concentration, and
  P1-A unavailability reason.
- `PeriodicResidueCandidates`: frame/plane/b identity, local axis `u` or `v`,
  period fraction, canonical rational numerator/denominator, sorted unique
  block-residue candidates, phase-error amplification, source-region ID,
  quality, and typed refusal.
- `RelativeShiftSupport`: exact match identity, `(row,column)=(v,u)` integer
  shift, PSR, peak, second peak, margin, overlap, structural shares, match score,
  and typed refusal.
- `HypothesisPhaseShiftEvidence`: exact left/right reading, plane, b-pair, and
  relative-quadrant identity; four canonical phase records; one shift record;
  typed top-level refusal.

Expose one total pure function:

```python
hypothesis_phase_shift_evidence(
    match: PlaneMatch,
    readings: Mapping[tuple[str, int], PlaneReading],
) -> HypothesisPhaseShiftEvidence
```

Provide deterministic JSON-safe `to_record()` output with `allow_nan=False`.

## Phase-candidate equation

For reading `r`, local direction `j=0` is `u`/column and `j=1` is `v`/row.
Let `b_r` be the hypothesis's canonical retained `blocks_per_period`, `d_r` its
unique retained distance, `P_r` the measured pixel period, and `phi_r(j)` the
explicitly available P1-A phase.

```text
f_r(j) = phi_r(j) / P_r
L_r(j) = d_r * (reading.rectification.centre[j] - reading.extent)
```

`L_r` is the continuous leading edge of the reading's untrimmed measurement
window in block units. Before the independent `MAX_SIGNATURE_CELLS` crop,
`plane_signature.origin_uv(j) = floor(L_r(j))`; trimming may shift the stored
signature origin. Candidate construction must use `L_r` directly and must not
derive it from a possibly trimmed signature. It is window-relative, not a seam
or world anchor.

Represent the canonical supported `b` as a reduced positive rational `p/q`.
The measured phase locates an energy peak at an unknown period index `n`:

```text
block_coordinate(n) = L_r(j) + (f_r(j) + n) * p/q
residue(n)           = block_coordinate(n) mod 1
```

Because `p` and `q` are coprime, all possible period indices produce exactly:

```text
C_r(j,b) = sorted_unique({(L_r(j) + f_r(j)*b + k/q) mod 1
                           for k in 0..q-1})
```

Thus integer `b` (`q=1`) yields one periodic-energy block residue. Fractional
`b` yields `q` reviewable alternatives rather than a fabricated single seam:
`1/2 -> 2`, `1/4 -> 4`, `1/8 -> 8`, `1/16 -> 16` candidates. Use a fixed
lookup derived from `DISTANCE_ALTERNATIVES`; do not infer arbitrary rationals
from untrusted floats. Canonicalize modulo endpoints and candidate ordering
deterministically. Report phase-error amplification `abs(b)`.

Only `phase_evidence` establishes availability. Legacy `phase_pixels == 0`
must never create a candidate. A genuine measured zero with nonzero response
is valid.

## Relative-shift contract

Copy the admitted match evidence without reinterpretation:

```text
cell_shift_rc = (row_shift, column_shift) = (delta_v, delta_u)
warped_left[x + cell_shift_rc] ~= right[x]
```

The shift lives in the right signature's `(row,column)` cell grid after the
existing homography/origin transforms. It is match-, quadrant-, and b-pair-
specific. Preserve zero, positive, negative, and maximum shifts exactly.
Do not map it to world axes, add a fractional phase, change its sign, treat it
as placement, or re-evaluate match admission.

Copy existing PSR/peak/second-peak/overlap/structural-share/score diagnostics.
No new support threshold may be fitted or applied.

## Refusals and invariants

Required refusal vocabulary:

- `reading_missing`
- `reading_identity_mismatch`
- `alternative_not_retained`
- `phase_evidence_absent`
- `phase_axis_unavailable`, carrying the exact P1-A subreason
- `invalid_period_fraction`
- `nonfinite_derived_value`
- `candidate_set_invalid`
- `cell_shift_absent`

Invariants:

- Evidence is a pure function of one match and its two addressed readings.
- Match/readings/b pair/quadrant/source IDs are exact and immutable.
- Phase records order left-u, left-v, right-u, right-v.
- Each candidate is finite and lies in `[0,1)`; sets are sorted and unique.
- Candidate count equals the canonical rational denominator `q`.
- Missing evidence is `None` plus a reason, never numeric zero.
- Shift fields remain exact integers in `(row,column)` order.
- No selector score, match gate, ordering key, certificate, or graph result
  reads the additive P1-B types.
- No P1-B output contains a world axis, gauge, world coordinate, seam, block
  placement, lateral origin, or chosen period index.

## Required tests

1. All ten `DISTANCE_ALTERNATIVES`: canonical `(p,q)`, candidate count `q`,
   sorted uniqueness, `[0,1)` range, and deterministic endpoint handling.
2. Unknown-period-index enumeration over at least `4q` positive/negative `n`
   values equals the closed candidate set for every b.
3. Integer b values yield one candidate; fractional b values retain all q
   candidates and never select one.
4. Analytic `L_r` agrees with the pre-trim
   `plane_signature.origin_uv=floor(L_r)` under P0 endpoint-inclusive spacing;
   a forced-trim control proves candidates still use measurement-window `L_r`
   and do not inherit the shifted stored origin.
5. Moving the window by one measured period permutes the same candidate set;
   moving the phase fraction changes it as predicted.
6. Measured zero phase produces candidates; absent evidence and every P1-A
   unavailable reason produce typed refusal and no candidate.
7. Source, reading, b-pair, and quadrant binding; malformed/nonfinite values
   fail closed.
8. Asymmetric non-square zero/interior/maximum shift controls preserve exact
   P1-A sign and `(row,column)` order, including positive/negative components.
9. Fractional and integer phase evidence never changes the stored shift or
   existing match support.
10. Input-order permutation yields byte-identical records.
11. Import/signature/output-field tests prove absence of images, manual/GT
    inputs, world/gauge/placement outputs, and cross-view phase fusion.
12. Existing lattice-plane, plane-graph, joint-selector, frozen-runner, and
    supported suites remain semantically unchanged.

## Exit criteria

- Types, pure function, rational candidate construction, refusal vocabulary,
  invariants, and deterministic `to_record()` are implemented exactly above.
- Tests 1-12 and focused regressions pass; `py_compile` and diff checks pass.
- A stable record hash and worked analytic examples for `b=1`, `1/2`, and
  `1/16` pin candidate semantics.
- No existing admission, scoring, ordering, selector, certificate, frozen
  schema/artifact, evaluation, CLI/UI, export, or production behavior changes.
- Lead review accepts P1-B only as preserved ambiguity plus relative shift;
  V2 remains blocked.

## Deferred

P1-C owns a public quadrant transform; mapping local phase candidates and
right-grid shifts into per-frame/per-world-axis relative constraints; testing
candidate-set consistency across matches; retaining multiple consistent
assignments; per-frame/per-axis translation support; and an explicit integer
translation gauge. It must account for the actual cell homography instead of
assuming the two raw rectified windows share a lateral frame.

V2 then consumes only an accepted component solution plus P1-C support to
derive face coordinate/outward sign, lateral origin, `cell_to_block`, supported
placement records, and typed placement abstention. Absolute seam identity and
integer period choice remain unresolved unless multi-view evidence proves them.
