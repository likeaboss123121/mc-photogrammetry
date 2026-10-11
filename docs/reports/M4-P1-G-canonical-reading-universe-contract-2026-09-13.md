# M4 P1-G canonical reading-universe identity contract

Date: 2026-09-13  
Status: assigned after lead acceptance of P1-F map repair  
Owner: Luna  
Depends on: P1-F map repair and its retained leading-signatureless counterexample

## Defect and required result

`match_plane_graph` assigns `PlaneMatch.left_plane/right_plane` by sorting only
readings reachable from successful signatures. `solve_component`, C1, and other
consumers resolve those integers by sorting the caller's complete reading list.
When a signatureless reading sorts before a matched reading, the same integer
names different planes across the boundary.

The retained P1-F counterexample has one leading signatureless axis-0 reading
and matched axis-1/2 readings. The matcher emits `(axis,index)=(1,0),(2,1)`;
the solver's all-reading universe resolves indices to `(0,0),(1,1)`. P1-G must
make one explicit canonical reading universe authoritative for both production
and every downstream consumer. Missing, foreign, duplicated, ambiguous, or
reordered reading identity must fail closed before solving.

## Interface decision

Make the complete `Sequence[PlaneReading]` a required argument of
`match_plane_graph`; do not provide a fallback that silently derives the index
universe from signatures. Preserve `match_plane_pair`'s temporary `-1` indices;
only the graph wrapper publishes canonical nonnegative plane indices.

The graph wrapper must:

1. validate every supplied value as a `PlaneReading` with a valid frame;
2. reject duplicate object entries and distinct readings with an identical
   `reading_order_key` inside one frame;
3. build each frame's index map from the complete readings sorted by the public
   `reading_order_key`;
4. require every signature's reading object to occur exactly once in that map;
5. assign match indices from that complete universe; and
6. validate every published match endpoint by resolving the index and comparing
   frame, reading axis, selected canonical b availability, and exact reading
   object identity before returning.

Keep `PlaneMatch` fields and schema-v1 record shape unchanged. Do not add a
second optional index map, infer identity from `(axis,b)`, search for a nearest
reading after mismatch, or renumber inside `solve_component`. Ambiguous evidence
must not be repaired heuristically.

## Call-site migration

Update every source, script, and test call to pass the exact complete reading
sequence corresponding to the signatures. Production frozen-image execution
must pass its `all_readings`; direct synthetic callers must pass their declared
universe. A signature-only caller must fail at the Python interface rather than
retain the unsafe legacy behavior.

`solve_component`, C1, P1-D, selector, and later C2-R code must continue using
the same public ordering rule. Add a shared validation helper only if it reduces
duplicated logic without widening scope. Do not change quadrant, b, score,
correlation, map, solver, or candidate-domain semantics.

## Decisive tests

1. Retain the exact leading-signatureless counterexample and require matcher
   and solver endpoints to resolve to axis 1/index 1 and axis 2/index 2 in the
   complete three-reading universe.
2. Cover leading, interleaved, trailing, and multiple signatureless readings in
   both endpoints. Matched identities remain object-identical and deterministic
   under input permutation.
3. A reading with some failed alternatives but at least one signature retains
   one canonical index; alternatives never create extra reading identities.
4. Missing signature reading, foreign equal-looking reading, duplicate object,
   tied ordering key, wrong frame, and reordered/forged match endpoint fail
   before `solve_component` can consume them.
5. Existing no-signatureless fixtures retain byte-identical match ordering,
   scores, refusals, and solution outputs except for the required call syntax.
6. C1 and P1-D accept matches produced with the complete universe and refuse an
   intentionally legacy-renumbered copy.
7. Frozen schema-v1 serialization/strict loading remains unchanged. Do not run
   R5 or create R6; synthetic runner coverage proves it supplies all readings.
8. AST/import checks prohibit manual metadata, manifests, F3/coordinates,
   references/GT/evaluator, renderer, CLI, export, filesystem/network/process,
   and placement dependencies.
9. All graph/selector/P1-B/C1/P1-D/P1-E/P1-F/C2-R proof and frozen-runner tests
   pass after explicit call-site migration. Search source/scripts/tests to prove
   no old one-argument `match_plane_graph` call remains.

## Scope boundary and stop

This is an identity-integrity repair. Do not alter `solve_component` acceptance,
add cycle gates, choose translations, resume C2-R-A2, build C2-R domains, migrate
schema, run real images/evaluation, or wire CLI/renderer/export/V2. Publish the
retained before/after counterexample, complete migrated call-site inventory,
focused/combined tests, compile/import/diff checks, and protected hashes; stop
for lead review.
