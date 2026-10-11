# M4 V2-A global endpoint-provenance CSP preregistration

Date: 2026-09-13

Status: **PREREGISTRATION only, for lead review; V2 cell placement remains BLOCKED**

Owner of this document: DeepSeek (documentation-only). No source, test, schema,
runner, selector, CLI, evaluator, renderer, export, label, manual-metadata, or
fixture work was done for it, and no test or fixture was run or created.

Assignment: `AGENT_COMMS.md` 2026-09-13 20:48:00 -05:00 ("V2 draft REJECT; V2-A
experiment contract assigned"). Amendment assignment: `AGENT_COMMS.md`
2026-09-13 22:31:00 -05:00 ("V2-A contract amendment required; source
remediation paused").

Superseded in place: the broad "M4 V2 cell-placement contract (frozen draft)",
SHA-256 `4D41AF35F3BCEE8827807C5B5460C579F4BCAB24720BDCB8373ABF1E56E004A4`.
That draft cannot authorize placement.

Revised in place by the V2-A preregistration remediation; the V2-A revision
SHA-256
`5FB357108F199BA7FC456A7A6A78031057D59BCF045DB40A0A33CD88EFB0C6C5` was
directionally correct but not frozen, and that remediation resolved the then
open items in 3.1, 3.4, 4.1-4.3, 5, and 8 rather than leaving them as
assumptions.

Revised again in place by V2-A preregistration remediation 2; the immediately
prior revision, SHA-256
`7E9735D65B1BCAC08B5AC84296369E62BA5DDFB22C84740FD580858D472119C1`, remained
unfrozen. Remediation 2 fixes three contract errors: role-neutral endpoint
source identity with `table_sha256` restricted to edge-local A4 replay (3.2,
4.1, 4.2, 5.8), withdrawal of the impossible replay-valid stored-reference
substitution in favor of an isolated post-gate CSP-kernel independence oracle
plus an AST/dataflow mutant (5.11), and exact tree-label injectivity with the
impossible tree multiplicity-two control replaced, admitting witness
multiplicity only inside a nonzero chord tolerance band (3.4, 5.5). The
remediation-2 injectivity mechanism is itself superseded by the amendment below,
which replaces its rational recovery with enumeration-based grouping.

Amended in place by the V2-A contract amendment of 2026-09-13 22:31:00 -05:00.
The immediately prior accepted revision, SHA-256
`34263C99A832B741394F81515E55CE831980ED9779A1F337FEA1261875DCA314`, is
superseded: its 5.5 required rational recovery of the P1-D shift from the
rounded A4 float delta, which is unsound. This amendment establishes tree
per-label injectivity from the complete factorized `E_m` enumeration instead of
any algebraic inversion, adds the non-dyadic `(3.0 + 0.1) - 0.2` regression and
the duplicate-group `tree_injectivity_violation` mutant (3.4, 4.3, 5.5),
preserves chord-band multiplicity, and adds the 4.5 result-validation
cross-link requirement.

## 0. Supersession and rejection record

The superseded draft is replaced because of these lead/Luna findings:

1. C2-R-B proves a **unique translation class while choosing provenance per
   edge**. It does not prove one compatible `(reading identity, selected b, u
   residue, v residue)` assignment across every incident edge.
2. Its canonical edge reference may hide incompatible alternatives inside
   equal-delta multiplicity.
3. It required the legacy `ComponentSolution.quadrants` and used
   nondeterministic Python object identity.
4. It left V2-A without a statistic and proposed a placement contract before the
   provenance question was decided.

This document defines only the V2-A experiment and creates no placement API.

## 1. Purpose and non-claims

V2-A answers exactly one question: do the accepted C2-R-B edge deltas admit
**zero, one, or multiple globally compatible endpoint-provenance assignments**
over the accepted A4 provenance candidates? It reports that count separately
from translation uniqueness.

V2-A is not, and this document does not define: a V2 placement API; an integer or
dyadic placement of any cell; a phase/seam anchor; a quantization, rounding, or
flooring rule; a signature statistic; a volume/merge/rank/select step; or any
runner, schema, selector, CLI, evaluator, renderer, or export integration.

The later, separately contracted phase-anchor experiment is out of scope here.
It may be proposed only if V2-A returns exactly one globally compatible endpoint
assignment, and only after its own frozen contract and lead review.

## 2. Accepted authority and fail-closed gate

V2-A consumes only these accepted artifacts, all read-only (current SHA-256):

- C1 axis/support contract `M4-P1-C1-axis-and-support-contract-2026-09-13.md`
  `7b6c2fe99322808edbbe9f4cfc3007eda346025b6e559830644b1905c5dc3e54`;
- P1-D complete-shift-domain contract
  `M4-P1-D-complete-shift-domain-contract-2026-09-13.md`
  `c09e8f10344171bb97cdce011167950a6c06a589abb0889bf3166cd054b699cc`;
- P1-F grid-map repair contract
  `M4-P1-F-right-to-left-grid-map-repair-contract-2026-09-13.md`
  `608a2a8afdefeda65a5cc30774f1485ecad1ebf06c81401639f452a339c5c2be`,
  implementation `.../M4-P1-F-right-to-left-grid-map-repair-implementation-2026-09-13.md`
  `a8456e880e3ce8e2ac868d876121bfb8dfb87a624aeade8ab410cfdbaa5db0b2`;
- P1-G canonical reading-universe contract
  `M4-P1-G-canonical-reading-universe-contract-2026-09-13.md`
  `aa9ad4f6921156220e31704f8d0f10427be91a0ca05975ac6d7f14bf7f2fd50d`,
  implementation `.../M4-P1-G-canonical-reading-universe-implementation-2026-09-13.md`
  `8995b3dbc50a4b48151b650abc32c9749e67826025424e329e700a373b635393`;
- C2-R-A4 indexed-domain contract
  `M4-P1-C2R-A4-indexed-domain-contract-2026-09-13.md`
  `6f1714eb33aa524798220594a9be51dfb23c358e699d8a33522e27d220b4ea1f`,
  implementation `.../M4-P1-C2R-A4-indexed-domain-implementation-2026-09-13.md`
  `d48aa6f3541cd79b58e1b23471c53724fd5a38acb2de1d8172a7ea1cf12ae15f`;
- C2-R-B cycle-closure contract
  `M4-P1-C2R-B-cycle-closure-contract-2026-09-13.md`
  `8929b19c3be1f729c20bc2be5849df3f7ca54dfc25fd5fa528ef96bc19492ce5`,
  implementation `.../M4-P1-C2R-B-cycle-closure-implementation-2026-09-13.md`
  `bf9a1e264ec7f2717c7669ebf9ddf198d175c7f223ffc61e319f8aaff273b218`;
- V1 provenance/footprints `M4-V1-region-provenance-footprints-2026-09-13.md`
  `09497c2600ddad1355ca9aefbd3c6276fc6f3a2c20c3ff46205c8e185df9a0a7`;
- V2 prerequisite audit
  `DEEPSEEK-M4-V2-placement-prerequisite-audit-2026-09-13.md`
  `e2b2205863d283f946c930f32485b821b4ff210f2d10ac9fcf253e79852ed602`;
- volume-bridge audit `DEEPSEEK-M4-volume-bridge-audit-2026-09-13.md`
  `9cec97effed580fd6324873f5ffbe0a88ff9185ea42cfb0327b8b52fd9a4c9be`.

The V2-A input gate requires **all** of:

```text
c2rb_result.status == "complete"
c2rb_result.placement_readiness == "ready"
c2rb_result.connected is True
len(c2rb_result.components) == 1
components[0].outcome == "unique_validated"
components[0].cycle_rank >= 1
components[0].exhaustive is True
components[0].failing_chords == ()
all cycle_residuals.matching_provenance_count > 0
len(components[0].translation_classes) == 1
vector_result.status == "complete" and no refusals
vector_result.represented_work == vector_result.planned_work
```

V2-A then replays `validate_relative_translation_graph` and
`validate_lateral_vector_domain_inputs` against the current C1/P1-D/P1-B/
readings/signatures sources. Caller-supplied digests, counts, indices, and
records are never trusted. Any failure is an atomic refusal; `refused`,
`budget_exhausted`, `review_required`, ambiguity, contradiction, tree-only, and
disconnection are outside the gate. A refused/exhausted result is never evidence
even if it carries stored translations.

## 3. Canonical typed identity and accepted edge deltas

### 3.1 Identity rule

Every identity used by V2-A is a canonical typed tuple plus a lowercase SHA-256
digest over canonical JSON (`sort_keys=True`, compact separators,
`allow_nan=False`). Python object identity, `id()`, memory address, set/dict
iteration order, and cross-run object reuse are forbidden as identity or
compatibility evidence. `C1MatchIdentity` and the P1-G canonical reading index
are the two authoritative typed identities.

The reading identity is frozen as the typed tuple `(frame, canonical per-frame
reading index, reading_order_key digest, reading axis, source_region_id,
canonical-universe digest)` plus a lowercase SHA-256 digest over its canonical
JSON. `canonical per-frame reading index` and `canonical-universe digest` are
exactly the P1-G canonical reading index and universe digest; `reading_order_key
digest` is the digest of `lattice_planes.reading_order_key(reading)` in its
canonical JSON form. The selected canonical `b` is **not** part of the reading
identity: it remains a separate frozen label field in 3.2. Quadrants are the
**absolute** C1 `left_frame_quadrant` / `right_frame_quadrant` values only; the
legacy `ComponentSolution.quadrants` mapping and any relative-only quadrant are
not inputs.

### 3.2 Endpoint label

For each reading `r` in the accepted component, an endpoint label `L_r` binds:

- `reading_identity` (3.1) plus digest;
- `selected_b`: the canonical retained blocks-per-period;
- `absolute_quadrant`: the C1 absolute quadrant for `r`;
- for local axis `u` and for local axis `v`:
  - `residue_source_sha256`: the role-neutral exact P1-B
    `PeriodicResidueCandidates` source record digest
    (`ResidueSideTable.record_sha256`), which is identical for a given source
    record at both endpoint roles;
  - `residue_index`: index into the side table's `candidate_values`;
  - `residue_value`: the exact finite candidate value at that index;
  - `local_axis_label`: the numeric tangent label;
- `footprint_identity`: the lowercase SHA-256 of the canonical JSON
  (`sort_keys=True`, compact separators, `allow_nan=False`) of the complete
  `PlaneRegionFootprint.to_record()`, including both count arrays
  (`inside_region_sample_counts_base64`, `valid_sample_counts_base64`) and
  `total_samples_per_cell`; missing footprint is the abstention
  `footprint_unavailable`, never a default.

`selected_b`, `absolute_quadrant`, the source record digest, `local_axis_label`,
and `footprint_identity` are pinned by accepted authority. The only free part of
a label is the pair `(residue_index_u, residue_index_v)`. Distinct residue
indices at either local axis are distinct endpoint labels.

`ResidueSideTable.table_sha256` binds `endpoint_role` (and `local_axis_label`),
so it differs by construction between a reading's left and right incidences of
one source record. It is an edge-local A4 replay field only: it is never an
endpoint-label field, never required to be equal across left/right incidences,
and never a source-conflict or compatibility criterion.

### 3.3 A4 provenance candidate

For accepted edge `m` with A4 domain `MatchLateralVectorDomain`, a global index
`g` in `[0, N_m)` factors exactly as A4 defines:

```text
e   = g // V ; t = g % V ; q0 = t // P_1 ; q1 = t % P_1
iL0 = q0 // B_0 ; iR0 = q0 % B_0 ; iL1 = q1 // B_1 ; iR1 = q1 % B_1
```

with `A_k = |L_k|`, `B_k = |R_k|`, `P_k = A_k*B_k`, `V = P_0*P_1`. The candidate's
complete three-axis delta is the A4 frozen-float result: the two lateral
components from `vector_at(g)` and the unchanged normal
`NormalFaceConstraint.delta_translation`. Each inspected pair is replayed in
A4's exact float operation order and converted to a reduced dyadic with
`float.as_integer_ratio()`.

Enumerating `g` never iterates `range(N_m)`, never builds a cross-edge product,
and never calls `candidate_at`, `iter_candidates`, or
`materialize_lateral_vector_domains`. Only the index-only factor accessors
(`iter_entry_groups`, `provenance_key_at`, `vector_at`) and the stored side
tables are used. `candidate_at_calls` remains a hard zero invariant.

### 3.4 Accepted edge-feasible candidate set

For each accepted edge the C2-R-B certificate fixes an accepted delta:

- tree edge: the accepted delta is `T[right] - T[left]`, computed exactly from
  the unique accepted translation certificate's reduced-dyadic translations on
  the same canonical tree. The feasible set is
  `E_m = {g : delta_dyadic(g) == T[right] - T[left]}` in exact rational
  arithmetic componentwise. The certificate's stored `delta_dyadic_xyz` for the
  edge is a consistency echo of that difference, never its definition.
- chord edge: the accepted delta is
  `C2RBCycleResidual.required_chord_delta_dyadic_xyz`, itself determined by the
  same tree-path potentials. The feasible set is
  `E_m = {g : componentwise |delta_dyadic(g) - required| <= closure_tolerance}`
  in exact rational arithmetic, with the entry-independent normal component
  matching. Every tolerance-feasible chord witness must additionally satisfy the
  section 4.2 endpoint-label constraints; tolerance relaxes the numeric delta
  only and never the endpoint provenance.

`E_m` is always non-empty and contains the stored `C2RBEdgeCandidateRef`. That
membership is checked for consistency only. **V2-A must never infer endpoint
compatibility from that one stored edge reference**; it enumerates every
provenance candidate in `E_m`.

Tree per-label injectivity is established only by the complete factorized `E_m`
enumeration of 4.3, never by algebraically inverting a rounded A4 float. Group
every enumerated feasible global index `g` in `E_m` by the full endpoint label
pair it induces; for an exact tree edge (`closure_tolerance` exactly zero) every
group must contain exactly one distinct global index, and any group with two or
more distinct indices is the frozen fail-closed `tree_injectivity_violation`.
The four endpoint residue indices plus the accepted exact delta are not decoded
back into a P1-D row/column shift: A4 stores each lateral delta component as an
ordered sequence of rounded IEEE-754 operations, so exact-dyadic rearrangement
of that stored rounded delta with the operand floats can be non-integral even
when the underlying enumerated index is valid. No algebraic inversion of a
rounded float is permitted. Witness multiplicity for a single label pair is
possible only on a chord edge with a nonzero `closure_tolerance`, where several
distinct chord global indices with identical endpoint labels lie inside the
rational tolerance band. Section 5.5 tests the enumeration-based injectivity,
the `tree_injectivity_violation` duplicate-group mutant, the non-dyadic
regression, and the preserved chord-band multiplicity.

## 4. Exact bounded global endpoint-provenance CSP

### 4.1 Variables and finite domains

- For every edge `m`, one witness variable `g_m` with domain `E_m`.
- For every reading `r`, one label variable `L_r` with domain
  `Dom(L_r) = { pinned fields } x { (i_u, i_v) }`, where `i_u` and `i_v` range
  over the residue indices of the unique role-neutral source record atom that all
  incident factor references for that `(r, local axis)` agree on: same
  `PeriodicResidueCandidates` source record SHA
  (`ResidueSideTable.record_sha256`), same canonical `selected_b`, same local
  axis and `local_axis_label`, and candidate values consistent with that source.
  `ResidueSideTable.table_sha256` is excluded from this agreement test because it
  binds `endpoint_role` and therefore differs between a reading's left and right
  incidences of one source record. Incident references that disagree on the
  source record SHA, `selected_b`, absolute quadrant, local axis/label, or
  footprint identity leave that `(r, local axis)` with an empty domain, and the
  run is the frozen fail-closed `endpoint_source_conflict`; multiple legitimate
  sources at the selected `b` are this conflict, never a heuristic repair or a
  union of candidate sets.

### 4.2 Constraints

A global assignment is a pair (witness per edge, label per reading) satisfying:

1. **Edge-label agreement.** For edge `m` with axis factors `k = 0,1`, the
   chosen `g_m`'s left residue reference on the factor's left local axis must
   equal `L_{m.left}` on that axis (role-neutral source record digest, local
   axis and label, index, value), and its right reference must equal
   `L_{m.right}` on the factor's right local axis. The factor's
   `blocks_per_period` must equal each endpoint's `selected_b`; the factor
   quadrants must equal the endpoints' absolute quadrants; the endpoint footprint
   identity must match. `ResidueSideTable.table_sha256` is not compared here; it
   is bound only when A4 replays the edge's own domain.
2. **Per-reading functionality.** Every incidence of `r` across every edge
   references the same `L_r`. No reading may carry two labels.
3. **No inference from the stored reference.** Constraint 1 is evaluated
   against the enumerated candidate's own provenance, never against
   `C2RBEdgeCandidateRef`.

A global assignment is **compatible** when constraints 1-3 hold. It is counted by
its distinct label assignment `(L_r)`; the number of witnesses realizing that
label assignment per edge is retained separately as multiplicity. Two label
assignments that differ on any free index are distinct even when all numeric
edge deltas agree.

### 4.3 Exact bounded search

Search is exact, lazy, and budgeted, in canonical order: readings sorted by
P1-G canonical reading index within frame; edges sorted by `C1MatchIdentity`
sort key. Enumerate `E_m` factorized by entry and per-axis residue pairs, never
by `range(N_m)`.

For each edge, bucket the enumerated feasible candidates by the endpoint label
pair they induce, keeping exact multiplicity; every enumerated chord witness must
already satisfy the 4.2 endpoint-label constraints. For an exact tree edge the
bucket must contain exactly one distinct global index, so a tree bucket with two
or more distinct indices is the frozen fail-closed `tree_injectivity_violation`;
higher multiplicity is admissible only for a chord bucket inside a nonzero
`closure_tolerance` band. Then solve the induced
constraint problem over readings: assign labels so every edge has at least one
compatible candidate; equivalently, an edge is satisfied when both its endpoint
labels have a jointly feasible candidate. Stop after certifying the second
distinct compatible label assignment, because the count is then `>= 2`. Proving
uniqueness requires exhaustive completion within budget. A partial assignment
may be pruned only when some edge can no longer be satisfied under it.

### 4.4 Output classification (separate from translation uniqueness)

V2-A reports, independently:

- `translation_class_count`: from the accepted C2-R-B certificate (must be 1);
- `endpoint_assignment_count`: `0`, `1`, or `>= 2` (with lower bound only when
  not exhaustive), plus retained canonical witnesses;
- per-edge `|E_m|`, the stored reference's membership, and the per-edge distinct
  induced label-pair count;
- per-reading label domain sizes and any `endpoint_source_conflict`.

Typed runtime outcomes:

- `endpoint_unique`: exactly one compatible label assignment;
- `endpoint_none`: zero compatible label assignments (incompatible same-delta
  evidence);
- `endpoint_multiple`: at least two compatible label assignments;
- `budget_exhausted`: atomic exhaustion, no partial count claimed;
- `refused`: gate, input, replay, identity, or result-validation failure,
  including the frozen fail-closed `endpoint_source_conflict` of 4.1, the
  `tree_injectivity_violation` of 3.4/4.3, and the `result_validation_mismatch`
  of 4.5.

A unique translation class with `endpoint_none` or `endpoint_multiple` is a
valid and expected outcome. No outcome selects, ranks, or prefers a label.

### 4.5 Result validation and cross-linking

Every emitted witness, label, edge record, and count is self-validating and
cross-linked, and resealing a mutated record's own digest never repairs it.
Before any classification is reported, the implementation re-derives and checks,
from the same immutable inputs:

- each retained edge witness's induced endpoint label pair against the
  reading-domain label digest of each incident reading; a witness whose induced
  pair disagrees with the referenced reading label domain digest is rejected;
- each retained reading label's digest against every incident edge witness's
  left/right endpoint cross-link, so no witness can be paired with a freshly
  resealed but semantically different reading label;
- each edge record's per-label-pair multiplicity against the exact number of
  distinct enumerated global indices in that bucket, and that bucket's inspected
  candidate and enumerator work counts against the recorded edge-record work;
- every reported count (`endpoint_assignment_count`, per-edge `|E_m|`, distinct
  induced label-pair counts, domain sizes, and retained witnesses) against a
  recomputation from the retained witnesses, edge records, and label domains.

Any mismatch is the frozen fail-closed `refused` outcome with the typed reason
`result_validation_mismatch`, never a repaired, clamped, or partially trusted
result. Consistency is a relation among the cross-linked records, not a checksum
over one record in isolation.

## 5. Required independent oracles and counterexamples

The later implementation must publish all of the following, and no expected
value may be derived from the code under test:

1. **Independent synthetic oracle.** A test-local builder constructs synthetic
   readings, side tables, match identities, and edge deltas by hand with
   independently declared labels, then checks `endpoint_unique`,
   `endpoint_none`, and `endpoint_multiple` on chain, tree, and cyclic fixtures
   without calling production A4/C2-R-B to derive the expectation.
2. **Incompatible same-delta counterexample.** Two incidences sharing one
   reading whose accepted deltas are numerically equal, but whose only
   feasible provenance choices require different residue indices at the shared
   endpoint. The stored `C2RBEdgeCandidateRef` alone reports compatibility as a
   consistency echo, never as evidence; the full CSP must report `endpoint_none`.
   Retain this as the decisive proof that translation uniqueness does not imply
   endpoint compatibility.
3. **Tree-delta derivation control.** The accepted tree-edge delta is recomputed
   as `T[right] - T[left]` from the certificate's reduced-dyadic translations and
   equals the edge's stored `delta_dyadic_xyz`; the classification is unchanged
   when the stored field is sourced only from that difference, and a mutant that
   instead reads the stored edge delta as the definition fails.
4. **Chord tolerance endpoint-obedience control.** A tolerance-feasible chord
   candidate whose residue provenance violates an endpoint-label constraint is
   rejected, so a chord that closes numerically but conflicts at an endpoint is
   `endpoint_none`; a mutant that accepts tolerance feasibility without the 4.2
   endpoint constraints fails.
5. **Tree per-label injectivity, non-dyadic regression, and chord-band
   multiplicity control.** Tree per-label injectivity is proved and tested from
   the complete factorized `E_m` enumeration of 4.3, never by algebraic
   inversion of a rounded A4 float. Group every enumerated feasible global index
   by the full endpoint label pair it induces; for an exact tree edge
   (`closure_tolerance` exactly zero) every group must contain exactly one
   distinct global index. Publish a duplicate-group mutant that synthesizes two
   distinct tree global indices under one endpoint label pair from an
   independently constructed bucket; it must fail closed as
   `tree_injectivity_violation`. Do not test any algebraic inversion of the
   stored delta. A4 computes each lateral component as
   `delta_component = s_R*component + s_L*v_left[i_L] - s_R*v_right[i_R]` over an
   ordered sequence of rounded IEEE-754 operations, so the entry component
   `component = s_R*(delta_component - s_L*v_left[i_L] + s_R*v_right[i_R])`
   recovered by exact-dyadic rearrangement of the stored rounded delta with the
   operand floats can be non-integral even when the enumerated index is valid
   (the world signs `s_L`, `s_R` are `+/-1`). Publish an independent non-dyadic
   regression using the frozen A4 operation `(3.0 + 0.1) - 0.2`, whose stored
   IEEE-754 float is the nearest double to 2.9 and not the rational 29/10, so an
   exact-dyadic algebraic inversion with the operand floats is non-integral for
   that valid enumerated index; the enumeration-based injectivity check must
   accept its valid tree witness and must not falsely refuse it. Separately,
   witness multiplicity greater than one for one label is valid only on a chord
   edge with a nonzero `closure_tolerance`, where several distinct chord global
   indices with identical endpoint labels lie inside the rational tolerance band;
   test that chord case separately and require one label assignment carrying the
   exact witness multiplicity, with every witness still obeying the 4.2
   endpoint-label constraints. A distinct residue index at either endpoint is a
   distinct label assignment, never multiplicity. The injectivity grouping,
   duplicate-group mutant, non-dyadic regression, and chord-band multiplicity
   are each cross-checked by the 4.5 result-validation requirement.
6. **Multiple-assignment control.** Two genuinely different label assignments
   each satisfying every edge, reported as `endpoint_multiple`.
7. **Transform/reflection coverage.** All three normals, four absolute
   quadrants, both camera sides, and signed rotation/reflection classes; a
   mutant that substitutes a relative quadrant or a `ComponentSolution`
   quadrant fails.
8. **Identity/digest attacks.** Forged reading identity, canonical-universe
   digest, `reading_order_key` digest, canonical per-frame index, role-neutral
   source record digest, index, value, `selected_b`, quadrant, or footprint
   identity fail closed; `id()`-based or object-based identity is never accepted
   as evidence. A mutated `ResidueSideTable.table_sha256` must fail the
   edge-local A4 replay, but it is never endpoint-label equality or
   source-conflict evidence and is never required to match across one reading's
   left and right incidences.
9. **Permutation.** Permuted readings, matches, and edges produce byte-identical
   records and counts.
10. **Replay.** Rebuilding from current C1/P1-D/P1-B/A4/C2-R-B sources reproduces
    the same classification; a mutated source fails replay.
11. **Independence from the stored reference.** The stored `C2RBEdgeCandidateRef`
    remains mandatory C2-R-B input-replay state, and canonical C2-R-B replay
    fixes the representative, so no replay-valid substitution of a different
    global index as the stored reference is possible; the previously assigned
    same-delta representative-substitution control is withdrawn as impossible.
    Independence is instead proved at an isolated post-gate CSP-kernel oracle: an
    independently constructed kernel entry point receives the same accepted edge
    deltas and the same complete feasible candidate buckets while an ignored
    representative sentinel varies across runs, including a sentinel whose
    residue and provenance fields contradict every bucket. Classification,
    counts, retained witnesses, and serialized records must be byte-identical
    across those sentinel values because the kernel never reads the sentinel.
    Public input still requires the unchanged canonical C2-R-B replay and its
    stored reference for membership and certificate replay. Separately, an
    AST/dataflow mutant whose kernel consults the representative's
    residue/provenance fields instead of enumerating and bucketing `E_m` must
    produce the wrong classification and fail the incompatible same-delta (5.2),
    chord-tolerance (5.4), and multiple-assignment (5.6) controls.
12. **Result-validation cross-link control.** A mutant that mutates a retained
    reading label or edge witness and then recomputes or reseals that record's
    own digest must be rejected by the 4.5 cross-link, multiplicity, and
    work-count checks as `result_validation_mismatch`. A freshly resealed record
    that no longer matches its cross-linked reading-domain label digest, its
    edge-record multiplicity, or its recorded enumerator work is never accepted,
    even though every digest on the mutated record is individually valid.

## 6. Budgets, atomic exhaustion, isolation

Budgets are explicit positive exact-integer caps with no defaults (booleans
invalid): readings, edges, endpoint labels, residues inspected, edge candidates
inspected, partial label assignments, consistency checks, result-validation
cross-links, distinct compatible assignments, and retained witnesses.
`max_distinct_assignments` must be at least two so `endpoint_multiple` is
certifiable.

Precedence is fixed: validate scalar/types, apply the section 2 gate, replay the
accepted sources, bind identities, compute the deterministic plan, then check
preflight budgets. Invalid input is refused even when it also exceeds a budget.
Preflight excess returns `budget_exhausted/planned_budget_exceeded` with the
exact plan, no counts, and no witnesses. Any runtime excess discards every
partial assignment and witness and returns
`budget_exhausted/runtime_budget_exceeded`; a counter mismatch returns
`budget_exhausted/runtime_count_mismatch`. Exact-limit succeeds. No clamp,
truncation, partial result, or substitute witness is permitted. Inputs are never
mutated; no A4 domain is materialized.

The future module may import only standard numeric/record/JSON utilities and
`plane_axis_support`, `plane_graph` (immutable types and `axis_transform`),
`lattice_planes` (immutable `PlaneReading`, `PlaneSignature`, and
`PlaneRegionFootprint` types and `reading_order_key` only), `plane_residue`,
`plane_shift_domain`, `plane_vector_domain`, and
`plane_relative_translation` types. AST checks prohibit image/camera/manifest/
F3/coordinate/bounds/reference/GT access; evaluator, renderer, dataset,
attribution, export, world writing; CLI/UI/runner; selector and legacy solver;
volume/`BlockVolume`/`ReconstructionResult`; filesystem, network, process;
`open`, `exec`, `eval`, `compile`, and dynamic import. `candidate_at`,
`iter_candidates`, `materialize_lateral_vector_domains`, and
`solve_component` are denied by name at call sites.

## 7. Decision rule, deliverables, stop boundary

- `endpoint_unique` is the only outcome that permits requesting a **later,
  separately contracted and reviewed** experiment for an image-derived physical
  seam anchor / global dyadic phase. V2 integer placement still stays closed,
  and this document defines no such anchor or statistic.
- `endpoint_none`, `endpoint_multiple`, or `budget_exhausted` keeps V2
  placement closed; a negative result retains the first incompatible same-delta
  counterexample.
- This task produces only this preregistration. It creates no V2 placement API,
  no source or test implementation, no runner/schema/selector/CLI/evaluator/
  renderer/export change, no labels or manual metadata, and no fixtures. It
  performs no commit, stage, reset, or clean.
- After lead review freezes the contract, DeepSeek remains the preferred
  implementer. Implementation stops for lead and DeepSeek review before any
  placement, anchoring, or wiring work.

## 8. Unresolved assumptions for lead decision

Resolved by the V2-A remediations and now frozen above, not assumptions:
tree-edge deltas derive from the unique certificate translations; chord
tolerance is exact rational componentwise and never relaxes endpoint provenance;
stored-reference residue fields are never the compatibility oracle, and
independence is proved by the isolated CSP kernel of 5.11 rather than by an
impossible replay-valid substitution; per-reading disagreement on the
role-neutral source record SHA at the selected `b` is fail-closed
`endpoint_source_conflict`, while `ResidueSideTable.table_sha256` binds
`endpoint_role` and is retained only for edge-local A4 replay; exact tree
witnesses are injective per endpoint label, established by grouping the complete
factorized `E_m` enumeration with fail-closed `tree_injectivity_violation`
rather than by inverting a rounded float (3.4, 4.3, 5.5), and witness
multiplicity is admitted only inside a nonzero chord tolerance band; retained
witnesses cross-link to reading-domain label digests and edge-record
multiplicities and work counts under 4.5; `footprint_identity` and the
reading-identity tuple are frozen in 3.1/3.2. The remaining assumptions are:

1. **Feasible-set cost.** The worst-case size of `E_m` and the cost of the
   factorized, lazily budgeted enumeration at production scale are unmeasured;
   the search may legitimately exhaust and report `budget_exhausted` with no
   partial count. Measurement is an implementation-phase obligation, not a
   license to clamp or sample.
2. **No anchor statistic.** No image-derived seam/phase statistic is defined or
   evaluated here; that is a separate later contract, permitted only after an
   `endpoint_unique` result and its own frozen contract and review.
