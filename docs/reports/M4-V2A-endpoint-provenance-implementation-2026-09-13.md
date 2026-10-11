# M4 V2-A endpoint-provenance implementation

Date: 2026-09-13

Status: **isolated R5 exact-kernel-replay remediation complete; stopped for
lead/Luna review. V2 cell placement, phase anchoring, and all wiring remain
BLOCKED and untouched.**

## 0. Assignment and accepted authority

- Accepted amended contract: `docs/reports/M4-V2-cell-placement-contract-2026-09-13.md`,
  SHA-256 `9e1d636d73063a0642ec6950dcee249906ed9f79373ddf5bb4e16b2ec1cbd822`,
  562 lines. Verified by `Get-FileHash -Algorithm SHA256` before any work; it
  re-verified unchanged after all work.
- Decisive rejection: `AGENT_COMMS.md` 2026-09-13 23:34:00 -05:00, "V2-A
  remediation REJECT; narrow accounting repair assigned". The prior frozen
  remediation `41870BA93E...616F` / `853C8FA4...CE54` passed 47/47 but was
  rejected because a freshly resealed `EdgeFeasibleRecord` and a freshly
  resealed runtime work record were still accepted.
- Decisive R3 rejection: `AGENT_COMMS.md` 2026-09-13 23:58:00 -05:00, "V2-A R3
  still REJECT; kernel-work replay remediation". The R3 source
  `479B64F0...B7A2` / tests `3E60D305...88CD` passed 53/53 and closed the 23:34
  edge/cap false-passes, but `V2AEndpointResult.build` still accepted a freshly
  resealed under-cap `runtime_work.partial_label_assignments` `5 -> 6` and
  `consistency_checks` `10 -> 11`. Range-checking the caps is not sufficient:
  every kernel-produced work field must be cross-linked to the retained
  `kernel_result_sha256`.
- Decisive R4 rejection: `AGENT_COMMS.md` 2026-09-14 00:12:00 -05:00, "V2-A R4
  digest-only binding REJECT; exact kernel replay required". The R4 source
  `C08B544A...BABA6` / tests `71AE1ED7...F540` passed 55/55, but the R4 binding
  merely *rebuilt* `KernelResult` from the public projection. Luna rebuilt
  `kernel_result_sha256` alongside the under-cap
  `partial_label_assignments`/`consistency_checks` mutations, so the digest
  replay was vacuous and both complete results still passed. The repair is to
  retain the canonical `KernelEdge` tuple on the result, cross-link every edge
  record to exactly one retained edge, and then **rerun the isolated kernel** from
  the retained domains/edges over reconstructed enumeration work, requiring the
  rerun to reproduce the parent kernel phase byte for byte.
- User correction applied verbatim during restart: never retain or materialize a
  tuple/list/set of every feasible global index or candidate; keep
  `O(readings + label domains + edge buckets + retained witnesses)` records and
  stream enumeration. The robust allowed pattern is a separate top-level/per-edge
  immutable `EnumerationCommitment` sibling, streamed from the accepted A4
  `domain_sha256` plus the canonical ordered feasible tuple transcript, bucket
  aggregates and counts, cross-linked against `EdgeFeasibleRecord`. Public
  `analyze` re-derives the commitments from the accepted source enumeration
  before classification. `V2AEndpointResult.build` rejects single-record
  reseals; it does not claim to authenticate an adversary that consistently
  rebuilds every root. The R5 kernel rerun narrows that boundary: a mutated
  kernel-produced work field is now rejected even when the attacker also reseals
  `kernel_result_sha256`, because the digest is re-derived from a fresh
  execution over immutable retained inputs.
- Owned files (only these were created or modified):
  - `src/mc_photogrammetry/reconstruction/plane_endpoint_provenance.py`
  - `tests/test_plane_endpoint_provenance.py`
  - `docs/reports/M4-V2A-endpoint-provenance-implementation-2026-09-13.md`
- No `__init__`, runner, schema, selector, CLI, evaluator, renderer, export,
  fixture image, label, manual-metadata, or other module was touched. No
  junction/reparse operation, recursive deletion, encoded command, executable,
  installer, background process, persistence change, Defender/security change,
  system-directory write, launcher automation, commit, stage, reset, or clean
  was performed.

## 1. Superseded artifacts and rejection record

| Superseded artifact | Lines | SHA-256 |
| --- | --- | --- |
| `plane_endpoint_provenance.py` (22:18 rejection) | 2295 | `23d2e5b32d15e28f87c23f9ee99a11bb223d04aa7e17fec6054afcc753df0d9e` |
| `test_plane_endpoint_provenance.py` (22:18 rejection) | 864 | `d20aa29c88bca2da3e285e04aaf3c3bd443fa94eb24a7159a22731f3b392e3e4` |
| implementation report (22:18 rejection) | 217 | `4c1682051ca2ce6ad7700dc7154dad77ed4639bc19650b49007bf8c1795d5217` |
| `plane_endpoint_provenance.py` (23:34 rejection) | 2485 | `41870ba93e242fa4c0f5ba706aaf183050913d3d573c8339a34001eb884e616f` |
| `test_plane_endpoint_provenance.py` (23:34 rejection) | 1110 | `853c8fa4e68bda1df6f3b34fa9057b6f4b05feaec58f007520021e935ac9ce54` |
| `plane_endpoint_provenance.py` (23:58 R3 rejection) | 2880 | `479b64f01a2e552a12b8b514853f13cf637582974fd8bc48477ba49fad81b7a2` |
| `test_plane_endpoint_provenance.py` (23:58 R3 rejection) | 1399 | `3e60d305266ea8cb92adf2dc99d1e03f5b5a2dc024d7b29d81cf1702257188cd` |
| implementation report (23:58 R3 rejection) | 276 | `81903326edff22e3500ca2b27b747d417837982177481affbfa45df88b728d8b` |
| `plane_endpoint_provenance.py` (00:12 R4 rejection) | 2914 | `c08b544a184303a5f50606c29577f5f6ded630a7936f305f4cf3d01a9a4baba6` |
| `test_plane_endpoint_provenance.py` (00:12 R4 rejection) | 1451 | `71ae1ed7d4db164c6e6be388e7bdac76483fce428594b691c8e93da14e5fa540` |
| implementation report (00:12 R4 rejection) | 333 | `b10119f49973ba3ce7a039c96b0da7dfea55998186e04125210724fff147d3f7` |

The 23:34 rejection's exact false-passes were: (a) a resealed
`EdgeFeasibleRecord` could invent impossible/out-of-domain label pairs and an
inconsistent `feasible_count` while `V2AEndpointResult.build` accepted it; and
(b) a resealed runtime work record could exceed a budget cap and disagree with
the reported assignment/witness counts while also being accepted.

The 23:58 R3 rejection's exact remaining false-pass was narrower and
under-budget: on the public ring, resealing `runtime_work` with
`partial_label_assignments` `5 -> 6` or `consistency_checks` `10 -> 11` kept
every value inside its request cap and left every parent edge-record,
commitment, witness, and count relation satisfied, so `build` accepted both.
No budget range check can detect a counter that changed while remaining legal;
the count had to be tied to the kernel outcome itself.

The 00:12 R4 rejection's exact remaining false-pass was that the R4 binding
rebuilt `KernelResult` from the same public projection the attacker controlled.
Rebuilding `kernel_result_sha256` from the mutated under-cap
`partial_label_assignments`/`consistency_checks` work therefore made the digest
replay agree with itself, and both complete results still passed. A digest-only
cross-link cannot bind attacker-mutable work; an independent re-execution from
immutable retained evidence is required.

## 2. Frozen remediation artifact hashes

| Artifact | Lines | SHA-256 |
| --- | --- | --- |
| `src/mc_photogrammetry/reconstruction/plane_endpoint_provenance.py` | 3040 | `3cb6aacea4d9e6873d0b10edd4ea7539b1646a1e75475cef2006f81caa3316e7` |
| `tests/test_plane_endpoint_provenance.py` | 1612 | `6037e62e9ffebdf1efaf00c809442c6750f5e7401f60eff0abc2269995766833` |

The report's own SHA-256 is reported in the stop message because a file cannot
contain its own digest.

## 3. Remediation applied

### 3.1 Streaming enumeration; no retained feasible-index collection

- `_bucket_pairs` no longer builds `grouped: dict[pair, list[index]]`. It streams
  `iter_entry_groups` and the factorized residue loops, and keeps only
  `grouped[pair] = (canonical_first_index, distinct_count)` plus one rolling
  `hashlib.sha256` transcript over the canonical ordered feasible
  `(global_index, left_code, right_code)` tuples. Memory is `O(buckets)`, never
  `O(|E_m|)`; the transcript is `O(1)`.
- `_resolve_index` is the single per-index replay (feasibility and induced label
  pair) shared by the streaming loop and the stored-reference membership check;
  it uses only `vector_at` and `provenance_key_at`.
- `_bucket_records` now consumes the compact `(first_index, distinct_count)`
  aggregates and still fails closed with `tree_injectivity_violation` when an
  exact tree bucket carries two distinct indices. No part of the module
  materializes a candidate record or an index set.
- `_enumerate_pipeline` returns `(kernel_edges, edge_records, commitments)`.

### 3.2 Independent `EnumerationCommitment` sibling (user correction)

- New immutable `EnumerationCommitment` bind: `match_identity`, `edge_kind`,
  `domain_sha256` (accepted A4 domain), accepted delta, both reading digests,
  `inspected_candidates`, `residues_inspected`, `feasible_count`, `bucket_count`,
  `bucket_aggregates_sha256`, `stream_transcript_sha256`, the stored reference
  index/codes/`reference_seen`, and a self digest.
- It is computed during the stream and stored as a top-level sibling list on
  `V2AEndpointResult` (`enumeration_commitments` plus `enumeration_root_sha256`,
  the digest over the ordered per-edge commitment digests). `EdgeFeasibleRecord`
  carries `enumeration_commitment_sha256` as a link, never as the evidence.
- `analyze_endpoint_provenance` re-derives every commitment from the accepted
  A4 source enumeration before the kernel classification runs. There is no
  caller-supplied commitment input to trust.

### 3.3 Full edge-record / domain / enumeration / count semantics

- `EdgeFeasibleRecord` now also binds `domain_sha256`, the stored reference's
  induced `(left_code, right_code)`, and `enumeration_commitment_sha256`. Its own
  constructor rejects a non-positive/oversized `feasible_count`, a
  `feasible_count > inspected_candidates`, a reference outside its domain, a
  reference pair absent from the retained buckets, and tree multiplicity above
  one.
- `_cross_link_mismatch` now takes the commitment siblings and the budget and
  rejects, before any classification:
  - a commitment whose own digest does not replay, is not injective, or binds an
    absent reading domain;
  - a commitment count that does not cover every edge;
  - an edge record that disagrees with its commitment sibling on identity, kind,
    domain, accepted delta, readings, `inspected_candidates`,
    `residues_inspected == 4 x inspected`, `feasible_count`, `bucket_count`,
    reference index/codes/membership;
  - an edge record whose canonical bucket aggregate digest is not the
    commitment's `bucket_aggregates_sha256`;
  - a multiplicity sum that misses `feasible_count`, a non-positive
    multiplicity, a `feasible_count > inspected_candidates`, or a stored
    reference pair absent from the buckets;
  - any label-pair code outside `[0, domain.label_count)` of the record's own
    left/right reading domain (the exact 23:34 false-pass (a));
  - a tree bucket with multiplicity above one.
- `V2AEndpointResult.build` re-runs the whole cross-link relation and additionally
  rejects a resealed `runtime_work` whose counters exceed a budget cap or whose
  `distinct_assignments`/`retained_witnesses` disagree with the reported
  classification/witnesses (the exact 23:34 false-pass (b)). It also cross-links
  every edge record to exactly one retained canonical `KernelEdge` (3.7),
  re-derives the cross-link counter plus the deterministic replay units and
  requires the sum to equal `runtime_work.result_validation_cross_links`, and
  finally deterministically reruns the isolated kernel from the retained
  domains/edges and reconstructed enumeration work, requiring byte-identical
  classification, proofs, witnesses, runtime work and digest (3.8). A
  legal-but-resealed kernel counter can therefore no longer survive.
- Because the record now triangulates record -> commitment -> witness, the
  retained kernel-edge tuple independently links record -> kernel input, and the
  kernel independently checks witness -> kernel edge, consistency no longer
  depends on mutual trust between only `EdgeFeasibleRecord` and `KernelEdge`.

### 3.4 `result_validation_cross_links` budget, plan and counter

- `V2ABudget` gained `max_result_validation_cross_links`; `V2AWork` gained
  `result_validation_cross_links`; `_RuntimeCounter` gained the slot and
  `exceeds()` checks it.
- The plan records a deterministic upper bound
  `readings + 2*edges + sum(inspected_candidates)
   + min(max_distinct_assignments, max_retained_witnesses) * (1 + readings + edges) + 1
   + (2*edges + 1)` where the added term is
  `_kernel_replay_validation_units`, and `_preflight_exceeds` enforces it, so a
  cap below the plan is an atomic
  `budget_exhausted/planned_budget_exceeded` with the exact plan, no counts and
  no witnesses.
- The runtime counter increments per cross-link unit and raises
  `V2ABudgetStop("runtime_budget_exceeded")` atomically; `analyze` catches it and
  returns no partial artifacts.
- Every planned/runtime counter and budget cap is cross-checked inside
  `_cross_link_mismatch`.

### 3.5 Preserved accepted behaviour

Public prerequisite replay; accepted tree deltas as `T[right] - T[left]` from
the unique certificate translations with the stored delta only as a checked
echo; exact rational componentwise chord tolerance obeying the same endpoint
labels; factorized complete `E_m` enumeration by `iter_entry_groups`,
`provenance_key_at`, and `vector_at`; the global reading-label CSP with lazy
pruning and a second-assignment certificate; zero/one/multiple separated from
translation uniqueness; atomic budgets and exact-limit/one-below semantics;
immutable self-validating records; the representative-independent post-gate
kernel; and the hard zero candidate-materialization invariant are unchanged.

### 3.6 Multi-candidate A4 isolated binding/enumeration evidence, not public E2E

- A genuine A4 ring with `blocks = 0.5` and a fractional translation
  offset is built with the existing accepted A4/C2-R-B fixture builders (no
  monkeypatch, no hand-built kernel, no A4 edit). Public
  `analyze_endpoint_provenance` refuses it at the frozen section 2 gate
  (`gate_not_ready`) before any identity binding or kernel classification, so it
  emits no public acceptance for a multi-candidate component.
- The multi-candidate evidence is obtained **only** by calling the private
  `_bind_pipeline` and `_enumerate_pipeline` helpers directly. It is an isolated
  integration control, not a public end-to-end acceptance: every edge has
  `represented_candidate_count = 16` and `feasible_count > 1`, and each streamed
  `EnumerationCommitment` is cross-checked against its `EdgeFeasibleRecord` and
  `KernelEdge`. Multi-candidate public acceptance therefore remains blocked; the
  test name and docstring say so explicitly.
- Exact documented limitation: a `b < 1` endpoint table necessarily carries a
  second half-period residue candidate (`plane_residue._candidate_values` yields
  `rational.denominator` values). Every multi-candidate domain therefore also
  carries a second exact delta class, and C2-R-B reports two translation classes
  (`placement_readiness == "review_required"`), not one. The frozen section 2
  gate consequently refuses through public `analyze` (`gate_not_ready`, asserted
  atomically). No multi-candidate A4 component can reach `endpoint_unique`
  through the public gate; a chord-band witness-multiplicity control on a real
  accepted multi-candidate chord is likewise not constructible. This limitation
  is recorded rather than worked around.

### 3.7 Retained canonical kernel edges (R5)

- `V2AEndpointResult` now carries the canonical `kernel_edges` tuple top-level on
  every complete result. It is the exact immutable tuple the isolated kernel
  consumed, bounded by the number of edge buckets (`O(sum |buckets|)`), and
  `KernelEdge.edge_sha256` already excludes the opaque
  `ignored_representative` sentinel from its digest. The tuple participates in
  `result_sha256` and in `to_json_bytes`, so the serialized record is
  sentinel-independent.
- `_kernel_edge_cross_link_mismatch` requires the retained tuple to be canonically
  ordered by `match_identity.sort_key()`, injective by `edge_sha256`, and to cover
  exactly the edge records. Every `EdgeFeasibleRecord.kernel_edge_sha256` must
  reference exactly one retained edge with identical `match_identity`,
  `edge_kind`, accepted delta and endpoint readings; the record's exact
  `label_pairs` projection must equal that edge's complete ordered bucket list;
  and both endpoint readings must resolve to retained reading domains. The
  commitment/domain cross-links continue to run through `_cross_link_mismatch`.
- Complete results with refused/budget atomicity require `kernel_edges == ()`.

### 3.8 Exact deterministic kernel replay (R5)

- `_reconstructed_kernel_initial_work` rebuilds the kernel's pre-search work from
  the independently retained `EnumerationCommitment` candidate/residue totals
  with every kernel-search, result and cross-link counter set to zero. It never
  reads the mutable kernel runtime record.
- `_kernel_replay_mismatch` then deterministically reruns
  `solve_endpoint_provenance_kernel` with the retained `reading_domains`, the
  retained `kernel_edges`, the same request budget, and that reconstructed
  initial work. The rerun's `KernelResult` must match the parent kernel phase
  byte for byte: status, classification, exhaustiveness, `lower_bound_only`,
  `distinct_assignment_count` (`endpoint_assignment_count`), retained witnesses,
  full `runtime_work` (with `result_validation_cross_links` reset to its kernel
  value `0`), and `result_sha256` vs. `kernel_result_sha256`.
- Because the digest is re-derived from a fresh execution over immutable retained
  inputs, mutating **any** kernel-produced work field and resealing
  `kernel_result_sha256` to the mutated projection is rejected. The R4
  digest-only rebuild is no longer the binding; it is only the attacker's
  resealing step, which the rerun ignores.
- `_kernel_replay_validation_units(edge_count) = 2*edge_count + 1` charges the
  kernel-edge cross-link (one unit per edge), the kernel rerun (one unit per
  edge) and the single replay transaction to
  `result_validation_cross_links`. The plan includes those units, so a cap below
  the plan is the atomic `budget_exhausted/planned_budget_exceeded`, and
  `__post_init__` re-derives `link_counter + replay units` and requires equality
  with `runtime_work.result_validation_cross_links`.
- The rerun is kernel-level only: it never calls `V2AEndpointResult` validation
  (no recursion) and never enumerates or materializes an A4 candidate. It reads
  only the retained records.
- `test_kernel_work_counter_attack_matrix_with_rebuilt_digest` mutates every
  kernel-produced counter (`readings`, `edges`, `endpoint_labels`,
  `residues_inspected`, `edge_candidates_inspected`,
  `partial_label_assignments`, `consistency_checks`, `distinct_assignments`,
  `retained_witnesses`, `result_validation_cross_links`), reseals
  `kernel_result_sha256`, and requires rejection. The two R3/R4 exact
  false-passes (`partial_label_assignments` `5 -> 6`,
  `consistency_checks` `10 -> 11`) are preserved as dedicated regressions and now
  also assert that the mutated projection passes `_cross_link_mismatch`, so the
  rejection is attributable to the rerun. A direct probe confirmed that
  disabling only `_kernel_replay_mismatch` reproduces the R4 acceptance for both
  counters and re-enabling it rejects both.

## 4. Exact tests run and timing

Command (from the repository root, `.venv` Python 3.10.11):

```text
.venv\Scripts\python.exe -m pytest tests/test_plane_endpoint_provenance.py -q -o addopts=
```

Result: **59 passed in 1.86 s** (pytest's own reported time in the combined
`py_compile` + pytest foreground run). All 55 R4 controls are preserved; four
R5 controls were added:

```text
test_kernel_work_counter_attack_matrix_with_rebuilt_digest
test_reconstructed_kernel_initial_work_uses_commitments_not_runtime
test_public_result_retains_cross_linked_kernel_edges
test_resealed_retained_kernel_edge_bucket_is_rejected
```

The two R3/R4 false-pass regressions were strengthened to reseal
`kernel_result_sha256` from the mutated projection and to assert the mutated
projection passes `_cross_link_mismatch`:

```text
test_resealed_under_cap_partial_label_assignments_is_rejected
test_resealed_under_cap_consistency_checks_is_rejected
```

The R4 multi-candidate control is preserved unchanged (still the R4 name
`test_multi_candidate_a4_private_binding_evidence_is_isolated`).

`py_compile` (both owned code files, exit code 0):

```text
.venv\Scripts\python.exe -m py_compile src/mc_photogrammetry/reconstruction/plane_endpoint_provenance.py tests/test_plane_endpoint_provenance.py
```

## 5. Contract controls mapped to tests

| Contract item | Test |
| --- | --- |
| 5.1 independent zero/one/multiple oracles | `test_kernel_unique_on_hand_declared_chain`, `..._tree_star`, `..._consistent_triangle`, `..._cyclic_conflict`, `..._multiple_assignment_control` |
| 5.2 incompatible same-delta cycle | `test_kernel_incompatible_same_delta_cycle_is_endpoint_none`, `test_representative_only_mutant_is_wrong_on_incompatible_same_delta` |
| 5.3 tree-delta derivation | `test_public_tree_delta_is_certificate_difference`, `test_public_stored_delta_echo_is_never_the_definition`, `test_accepted_tree_delta_dataflow_is_the_certificate_difference` |
| 5.4 chord tolerance endpoint obedience | `test_chord_tolerance_only_mutant_is_wrong_on_endpoint_conflict`, `test_exact_tree_delta_versus_rational_chord_tolerance` |
| 5.5 enumeration injectivity / non-dyadic / chord multiplicity | `test_tree_enumeration_injectivity_holds_on_public_ring`, `test_tree_duplicate_group_mutant_fails_closed`, `test_non_dyadic_rounded_tree_delta_is_accepted_without_inversion`, `test_kernel_tree_edge_rejects_two_witnesses_for_one_label_pair`, `test_kernel_chord_band_carries_exact_witness_multiplicity`, `test_kernel_chord_distinct_residue_index_is_a_distinct_assignment` |
| 5.6 multiple-assignment control | `test_kernel_multiple_assignment_control`, `test_representative_only_mutant_is_wrong_on_multiple_assignment` |
| 5.7 orientation/side coverage | `test_oriented_labels_cover_axes_quadrants_and_sides`, `test_relative_quadrant_substitution_changes_the_label` |
| 5.8 identity/digest attacks | `test_kernel_and_public_records_fail_closed_on_tampering`, `test_every_named_identity_field_fails_closed`, equal-content binding replays, `test_module_forbids_id_calls_and_reading_object_identity` |
| 5.9 permutation byte identity | `test_kernel_permuted_inputs_produce_byte_identical_records` |
| 5.10 replay | `test_public_validate_rejects_a_replaced_source`, `test_public_ring_is_endpoint_unique_and_replays` |
| 5.11 independence from stored reference | `test_kernel_output_is_byte_identical_across_ignored_sentinels`, `test_kernel_never_reads_the_representative_sentinel`, `test_representative_only_mutant_is_wrong_*` |
| 4.5 / 5.12 result cross-links, retained kernel-edge cross-link, exact kernel replay, freshly resealed attacks | `test_result_cross_links_reject_freshly_resealed_mutations`, `test_resealed_edge_record_with_out_of_domain_pair_is_rejected`, `test_resealed_commitment_sibling_with_relinked_record_is_rejected`, `test_resealed_runtime_work_cannot_exceed_or_disagree`, `test_resealed_under_cap_partial_label_assignments_is_rejected`, `test_resealed_under_cap_consistency_checks_is_rejected`, `test_kernel_work_counter_attack_matrix_with_rebuilt_digest`, `test_reconstructed_kernel_initial_work_uses_commitments_not_runtime`, `test_public_result_retains_cross_linked_kernel_edges`, `test_resealed_retained_kernel_edge_bucket_is_rejected`, `test_public_result_validation_mismatch_is_a_typed_refusal`, `test_kernel_cross_link_rejects_a_resealed_inconsistent_witness` |
| 6 budgets / cross-link counter / AST boundary / zero materialization | `test_public_preflight_exact_limit_...`, `test_public_runtime_exact_limit_...`, `test_public_result_validation_cross_links_exact_limit_and_one_below`, `test_cross_link_runtime_counter_stops_atomically_under_a_tight_cap`, `test_kernel_runtime_exact_limit_...`, `test_kernel_retained_witness_limit_is_exact`, `test_module_imports_and_calls_stay_inside_the_isolation_boundary`, `test_module_uses_only_index_only_a4_accessors`, `test_public_path_never_materializes_a4_candidates` |
| multi-candidate A4 isolated binding/enumeration evidence (not public E2E) | `test_multi_candidate_a4_private_binding_evidence_is_isolated` |
| sparse performance | `test_kernel_sparse_chain_is_linear_and_fast` |

## 6. Limitations and open items for lead review

1. **Multi-candidate A4 cannot pass the frozen gate; public acceptance stays
   blocked.** As recorded in 3.6, a `b < 1` endpoint table always yields a second
   exact delta class and therefore two C2-R-B translation classes, so public
   `analyze` refuses at `gate_not_ready` before binding and no public
   end-to-end multi-candidate acceptance exists. The isolated control calls only
   the private `_bind_pipeline`/`_enumerate_pipeline` helpers and proves
   `represented_candidate_count > 1` and `feasible_count > 1` with commitment
   cross-links, while also asserting the atomic public refusal. A production
   chord-band multiplicity through `analyze` remains unconstructible inside the
   owned files.
2. **Scale unmeasured on production `E_m`.** The largest public edge exercised
   has per-edge feasible work in the tens; the >2.3M-candidate A4 scale case is
   not driven through V2-A. Streaming keeps memory `O(buckets)` and the plan
   bounds the transaction, but worst-case `E_m` size and transcript hashing cost
   at production scale remain contract assumption 8.1.
3. **Sibling commitment is a relation, not authentication.** A caller that
   consistently reseals every interlinked *enumeration* record and root can still
   produce an internally consistent object; the security boundary is
   `analyze`/`validate_endpoint_provenance_result` re-deriving the commitments
   from current accepted sources, exactly as the user correction requires. The
   3.8 kernel rerun does not presuppose the retained `kernel_result_sha256`: it
   re-derives the kernel digest from a fresh execution over the retained domains
   and retained `KernelEdge` tuple, so mutating any kernel-produced work field and
   resealing `kernel_result_sha256` is rejected. The remaining consistent-rebuild
   boundary is the one that also rebuilds the retained `KernelEdge` buckets,
   reading domains, commitments, edge records and witnesses together; that
   boundary is closed only by the full public source replay
   (`validate_endpoint_provenance_result`), and is not claimed otherwise.
4. **`result_validation_cross_links` is a runtime counter whose plan is an upper
   bound and is preflight-enforced.** The plan now includes the deterministic
   `_kernel_replay_validation_units(edge_count) = 2*edge_count + 1` for the
   retained-kernel-edge cross-link and the kernel rerun. A cap below the plan is
   an atomic `planned_budget_exceeded`; a cap at or above the plan cannot exhaust
   at runtime for the same records. The runtime counter is still enforced and
   cross-validated, and its atomic stop is exercised directly. The rerun's own
   search work is separately bounded by the same request budget, so the
   validation phase cannot run unbounded.
5. **Orientation coverage is record-level, not full physical public-path.** All
   three reading axes, four absolute quadrants, and both camera sides are covered
   by frozen typed records plus the relative-quadrant discriminator, and the AST
   test forbids `ComponentSolution`/relative-quadrant sources. No multi-frame
   physical cycle fixture with nonzero absolute quadrants was built; that would
   require new image-derived fixtures outside the owned files.
6. **`footprint_unavailable` is a typed abstention value.** When a selected
   endpoint signature carries no `PlaneRegionFootprint`, the label keeps the
   literal `footprint_unavailable` and incidence agreement is on that value; no
   default digest is invented.
7. **Negative result not yet observed on real accepted input.** The public ring
   is `endpoint_unique`. The decisive incompatible same-delta `endpoint_none` and
   the `endpoint_multiple` control remain kernel-oracle fixtures.
8. **No placement, anchor, statistic, or wiring was added**, per the contract.
   `endpoint_unique` here does not authorize any phase-anchor experiment; that
   requires its own frozen contract and lead review.
