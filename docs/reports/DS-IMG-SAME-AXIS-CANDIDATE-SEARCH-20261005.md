# DS-IMG-SAME-AXIS-CANDIDATE-SEARCH — isolated synthetic experiment

Status: **IMPLEMENTED, PENDING ASTRA AND LUNA REVIEW**. Production integration,
real-input provenance acceptance, reconstruction release and accuracy claims
remain **HOLD**. This utility never returns a chosen result and always reports
`reconstruction_authorized=false`.

The new I/O-free utility exhausts alternative complete indexed candidate edge
subsets under digest-bound image-derived camera constraints. It retains every
feasible full-axis solver outcome, including failed integer-consistency outcomes,
without ranking or discarding candidates by score, cost, input order or a Pareto
rule. This is deliberately more conservative than retaining only nondominated
outcomes. Solver consistency cannot establish correspondence truth.

## Contract and enumeration

Inputs are immutable tuples of existing `PlaneReading` and `PlaneMatch` types,
an existing `ImageDerivedConnectionGraph`, externally trusted session and graph
SHA-256 pins, and optionally decreased `SearchCaps`. No arbitrary quadrant tuple
or self-asserted image-derived flag is accepted. Existing graph, budget,
per-pair evidence and yaw diagnostic constructors replay their digest/shape
contracts before search. Every evidence record must bind to the trusted session,
and the graph must contain exactly the reading-present frame universe.

These checks detect binding mismatches and tampering. They do **not** authenticate
the origin of caller-supplied objects or hashes. Authentic image-source provenance
must already have been established by the accepted upstream image-session
validator. Replaying pixels/session bindings is deliberately outside this task.
All exercised evidence in this report is synthetic. No genuine real-image camera
constraint has been validated or claimed here. A future adapter needs independent
review of this trust boundary before supplying authenticated production inputs.

Canonical candidate identity contains both frame IDs, both canonical reading
indices, both local axes, both selected b values, relative quadrant, and the
complete optional cell shift with an explicit absent-shift sentinel. All scored
metrics remain in the retained candidate payload. A repeated identity refuses;
the same identity with different metrics is a conflicting identity and refuses.
Indexed endpoints and uniquely selected b values are checked using unchanged
solver helpers. No score filter is introduced.

Solved camera relations form a Z4 graph. Contradictory cycles refuse atomically.
Each connected component's relative potentials are computed; all remaining
component offsets are enumerated with the first frame at quadrant zero as a
gauge. Each nonempty binary candidate subset is visited. Leaves with incompatible
relative quadrants, conflicting b values for one reading, or missing world axes
0, 1 or 2 are infeasible. Otherwise the unchanged `solve_component` is called
with the authentic supplied relations plus explicit fixed labels for that
enumerated assignment. Fixing labels per call avoids losing unresolved
quadrant rivals to the solver's ranked-alternative truncation. It never supplies
missing camera relations as observations: free labels remain separately retained
enumerated hypotheses.

The output retains full solver fields: translations, quadrants, residuals,
resultants, scale flag, blocks-per-period bindings, reason, and alternatives.
The post-Astra numerical correction checks finite signed-face and delta
arithmetic before invoking the solver. Each unchanged solver call runs within
a scoped NumPy error state that raises on overflow, invalid operations and
division by zero, and restores the caller's policy on either exit. Every
returned translation, residual, resultant, b value, alternative objective cost,
alternative resultant and observation is checked for finiteness; the scale flag
must be an actual boolean. Nonfinite intermediates, LAPACK faults and nonfinite
solver output refuse as `invalid_solver_output`, atomically discarding any
private prefix. No invalid numerical value can be retained as provisional.
The scale flag is the unchanged provisional solver diagnostic. A single retained
outcome is `provisional_unique`; multiple retained outcomes, including failed
full-axis rivals, are `provisional_ambiguous`. Neither authorizes reconstruction.
When no solver-consistent assignment remains, a constraint contradicts, a bound
is crossed, or a validated boundary faults, the result is `refused` with empty
outcomes and candidate payload. Work counters may survive; diagnostic results
and any private prefix do not.

## Fixed limits

| Limit | Hard maximum | Evidence |
| --- | ---: | --- |
| Candidate identities | 64 | 65 input identities refuse before enumeration |
| Search nodes | 4,096 | Full-tree preflight; 12 identities refuse |
| Solver calls | 512 | Full-leaf preflight; 10 identities refuse |
| Retained outcomes | 512 | Reserved with all potential leaves; decreased cap tested independently |
| Diagnostic representation | 2,097,152 bytes | Allocated resident object-size traversal; oversize input and actual late crossing refuse |
| Reading count | 128 | 129 readings refuse before canonical indexing |
| Frame count | 8 | Nine frames refuse before constraint enumeration |

Caps cannot increase, be zero, or be booleans. With N candidate identities and K
remaining free camera components, preflight conservatively reserves
`4**K * 2**(N+1)` nodes and `4**K * (2**N - 1)` calls/outcomes. Each quadrant
assignment charges one node and each binary tree node charges one. This makes
resource refusal independent of favorable pruning. In particular, accepting 64
input identities does not promise that their powerset fits the work budget;
with fixed camera labels the call bound permits at most nine identities.

`representation_bytes` uses CPython `sys.getsizeof` and visits each referenced
object once, including dataclass slots and nested tuples/dictionaries. The input
representation is checked before indexing/digest replay, each private retained
prefix is measured, and the complete output is measured before publication.
This is an allocation-measured diagnostic representation bound. It is **not**
a hard process peak, RSS, solver temporary-allocation, or simultaneous input-plus-
output memory bound. The traversal's bookkeeping and container temporaries are
outside the reported representation size. No input-buffer mutation or borrowed
pixel buffers were used.

## Verification

All commands used `PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH=src`, Python `-B`, and
`-p no:cacheprovider`; no scratch/probe/log files or bytecode caches were created.

1. `.venv\Scripts\python.exe -B -m pytest tests/test_same_axis_candidate_search.py -o addopts= -p no:cacheprovider -q -s`
   — **78 passed in 16.35s**, exit 0, after the Astra numerical correction.
2. `.venv\Scripts\python.exe -B -m pytest tests/test_plane_graph.py -o addopts= -p no:cacheprovider -q -k 's1_three_view or s4_single_plane or s5_two_views or solved_camera_quadrant_constraint or incomplete_camera_constraints or malformed_foreign_and_contradictory_quadrant_constraints or wrong_b_is_rejected or bu_cross_edge_scale_alias or s7_frame_relabelling or expanded_cap_is_field_exactly_equal or nine_view_component_abstains'`
   — **11 passed, 99 deselected in 0.32s**, exit 0. These selected cases are pure
   solver controls; no image-level, correlation, matcher-inference or pipeline
   control was run.

The independent tiny oracle directly enumerates the powerset and all four yaw
labels per non-anchor frame, using a literal axis table and complete endpoint
payloads. It does not invoke any search enumeration or pruning helper. It calls
the same unchanged numerical solver in canonical endpoint order and compares
complete retained payload sets exactly. Cases include extra identity variants,
partial camera constraints and all three nonzero relative quadrants. This proves
search enumeration equality for these tiny controls, not independent numerical
solver correctness; the existing pure solver suite supplies regression evidence.

The synthetic three-frame lower-score control proves that the unchanged local
same-axis filter drops the 0.30-score identity in favor of a 0.99 rival. The
complete low-score cycle remains solver-consistent; the analogous high-score
cycle returns `plane_integer_residual`. Both full-axis outcomes and every other
feasible full-axis rival are retained. Equal and differing scores preserve
ambiguity. The one-identity-per-axis control preserves unchanged solver fields.
Reading order, candidate order and connection-pair orientation are invariant.
Duplicate/conflicting identities, missing axes, empty camera constraints,
contradictory camera cycles, pin mismatch, nested digest tampering, malformed
candidates, every cap, attempts to raise limits, actual late memory exhaustion
and a late synthetic solver boundary fault all refuse without a result prefix.

The 31 additional numerical regression cases include the exact finite extreme
inputs from Astra's finding: selected distances `1e308` with opposite sides
`-1/+1` (and reversed sides) refuse before the solver because the signed delta
overflows; three finite candidate scores `1e308` refuse when the unchanged
solver's mean/objective accumulation overflows. NaN, positive infinity and
negative infinity injected into each numeric solver field and alternative
diagnostic refuse. A nonfinite second solver response discards an already
retained private prefix. Normal and extreme-but-finite boundary controls remain
provisional: opposite-side distance `8e307` with score `max_float/4`, same-side
distance `1e308` with score `1e307`, and ordinary distance/score values. NumPy
error-policy restoration is checked on success and refusal. No bound or solver
acceptance threshold was raised or weakened.

The largest tested fixed-label control has nine identities, 343 full-axis
outcomes, 1,024 nodes and 343 solver calls. Its resident diagnostic representation
measured **1,344,657 bytes**. The final test-run tracemalloc measurement recorded
current **1,131,629 bytes**, peak **4,385,834 bytes**. Peak exceeding 2 MiB is
explicit evidence that the diagnostic representation limit is not a process
memory ceiling. CPython allocation figures are environment-dependent.

## Pins and scope

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| New `_same_axis_candidate_search.py` | 17973 | `5c48c57b9795bf3c2c5a76ffe0b6c7c5523f34417f39b8134ea39f374950e03d` |
| New `test_same_axis_candidate_search.py` | 24069 | `2f490029b83fab2b7e1fa7c121a393734c876e24b0a09f890b285d73ece3cfcf` |
| Unchanged `plane_graph.py` | existing | `f697c2f4ffb512ebb3e7af8f9e497328b4fe828197f62d761b9b0f850ef8972f` |
| Unchanged `image_derived_connections.py` | existing | `a2d166bf711d6cd77c169f044da5ca28d00073e66e60b9bff992572fb56ba211` |

Only the assigned new source, new test module, this report and its JSON companion
were created, with a prefix-preserving append to `AGENT_COMMS.md`. No production
matcher/filter/solver/runner/threshold/texture/camera policy was edited. No input
images, truth, frozen records, clients, launchers, evaluator or real inference
were accessed or run. No source-footprint, correspondence-accuracy, capture-
readiness, end-to-end reconstruction or production acceptance claim follows.
Stop for Astra high-risk review and Luna independent review.
