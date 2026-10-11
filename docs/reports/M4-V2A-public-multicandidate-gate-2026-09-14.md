# M4 V2-A public multi-candidate gate experiment

Date: 2026-09-14
Owner: DeepSeek (DS-V2A-MC1)
Scope: answer the 00:44 lead question — can a *genuine public*
`analyze_endpoint_provenance` request pass the frozen section-2 gate while an A4
domain has `represented_candidate_count > 1` and that domain's `E_m` has
`feasible_count > 1`?

## Verdict

**Constructible.** A small deterministic two-frame fixture driven through the
full public stack returns `status == "complete"` with

* an A4 chord domain whose `represented_candidate_count == 16 > 1`, and
* that chord's `EdgeFeasibleRecord.feasible_count == 4 > 1`,
* while C2-R-B is `unique_validated` with exactly one translation class.

The construction is public-only (production builders + `analyze_endpoint_provenance`
/ `validate_endpoint_provenance_result`). It uses no monkeypatch and calls no
private V2-A helper (`_bind_pipeline`, `_enumerate_pipeline`,
`_bucket_pairs`, `_bucket_records`, `solve_endpoint_provenance_kernel` are never
called by the acceptance test).

## Frozen artifacts

| artifact | SHA-256 | state |
| --- | --- | --- |
| `src/mc_photogrammetry/reconstruction/plane_endpoint_provenance.py` | `3CB6AACEA4D9E6873D0B10EDD4EA7539B1646A1E75475CEF2006F81CAA3316E7` | unchanged, never edited |
| `tests/test_plane_endpoint_provenance.py` before | `6037E62E9FFEBDF1EFAF00C809442C6750F5E7401F60EFF0ABC2269995766833` | accepted R5 baseline |
| `tests/test_plane_endpoint_provenance.py` after | `967559F7B20184442F558C34AF803128E85AB7596FD50C2489DD8A16C5270F95` | +1 fixture +1 test |

Source hash was verified before any work and again after the test edit; it is
byte-identical to the frozen accepted value.

## Construction

Minimal public graph: frames `A`, `B`, translation difference `(2, 0)`.

Each frame carries **two distinct same-axis readings** (`axis = 2`, `side = +1`,
`distance = 2.0`, residue `0.0`):

* an integer-period reading `b = 1`, and
* a half-period reading `b = 1/2`.

Two directed matches:

* tree candidate `A -> B` using the two `b = 1` readings;
* chord candidate `B -> A` using the two `b = 1/2` readings.

Quadrant `0` gives both world signs `+1`. A4 product dimensions:

* tree domain: `1 x (1*1) x (1*1) = 1` represented candidate;
* chord domain: `1 x (2*2) x (2*2) = 16` represented candidates.

Cycle closure: the chord's required delta is `T[A] - T[B]`, which the spanning
tree fixes exactly. The half-period tables are exactly `{0.0, 0.5}` (dyadic), so
on **each** lateral axis the diagonal residue pairs `v_left = v_right` reproduce
the required delta exactly. That is `2` witnesses per lateral axis, i.e. `4`
exact `E_m` witnesses on four distinct induced endpoint label pairs, each with
multiplicity `1`.

## Why the naive shared-reading multi-candidate case is refused

The previously documented limitation is real but narrower than originally
stated.

1. `_factorized_single_delta_class` is the mandatory eligibility proof. With
   strictly increasing residue tables and never-zero signed residue
   coefficients, a domain whose factor extrema differ cannot pass it, and
   `represented_candidate_count > 1` always makes the extrema differ. So every
   multi-candidate component is routed to `_solve_general_component`; it never
   uses the singleton path.
2. In the general search, `variable_order = _tree_order(component, domains)` is
   exactly the spanning-tree edge set. Only tree edges are enumerated by exact
   dyadic `_delta_classes`; chords are consumed by `_factorized_membership`.
3. If a **tree** edge is multi-candidate, `_delta_classes` exposes at least two
   exact delta classes (`low != high` from step 1). The search must then have
   cycle closure eliminate every class but one. In the all-`b = 1/2` ring the
   half-period shift is a global symmetry, two translation classes survive, and
   C2-R-B reports `ambiguous` / `review_required`, which the frozen gate refuses
   (`gate_not_ready`). Mixed `b` on shared readings reproduces this: any frame
   with a non-integer reading contaminates an incident tree edge.

## Why the disjoint chord bypasses it

The non-integer reading is placed **only on the chord**, never on the spanning
tree:

* no tree variable is multi-candidate, so `_solve_general_component` takes the
  `all(len(groups[identity]) == 1 for identity in variable_order)` branch;
* translation uniqueness is certified by the single exact integer tree class,
  giving `outcome == unique_validated`, `len(translation_classes) == 1`,
  `cycle_rank == 1`, `failing_chords == ()`;
* the chord's whole multiplicity becomes cycle provenance:
  `matching_provenance_count == 4 > 0`.

Orientation matters. `reading_order_key` sorts the half reading first (smaller
`period_pixels`), so it receives canonical index `0`. If both edges were
oriented `A -> B`, the half edge would win the BFS adjacency tie
(`left_plane 0 < 1`) and become the **tree** edge, restoring ambiguity.
Orienting the half edge `B -> A` makes `C1MatchIdentity.sort_key` select the
integer edge (`left_frame A < B`) as the tree edge and the half edge as the
chord.

## Exact public record

`analyze_endpoint_provenance(... budget=_ring_budget())` on the fixture:

* `status == "complete"`, `refusals == ()`, `budget_failure_reason is None`
* `classification == "endpoint_multiple"`, `exhaustive is False`,
  `lower_bound_only is True`, `translation_class_count == 1`
* `endpoint_assignment_count == 2`, `len(witnesses) == 2`
* `len(reading_domains) == 4`, label counts `[1, 1, 4, 4]`
* `len(edge_records) == len(kernel_edges) == len(enumeration_commitments) == 2`
* tree record: `inspected_candidates == 1`, `feasible_count == 1`,
  `label_pairs == ((0, 0, 1),)`
* chord record: `inspected_candidates == 16`, `feasible_count == 4`,
  `induced_label_pair_count == 4`,
  `label_pairs == ((0, 0, 1), (1, 1, 1), (2, 2, 1), (3, 3, 1))`
* `planned_work.edge_candidates_inspected == runtime_work.edge_candidates_inspected == 17`
* `runtime_work.endpoint_labels == 10`,
  `runtime_work.candidate_materializations == 0`
* `validate_endpoint_provenance_result(...) is True`

Deterministic public digests (three identical trials):

* `result_sha256 = f2446d270068edd676e31412083da3f7a6225209a99d8083da40d739e718caa9`
* `input_sha256 = 990211df7dc04c64d44d228a74117f7388bd4cb117c47e1a045a09ed214443db`
* `kernel_result_sha256 = daa5ce7a01c8dfb33a97b285245447a1f0f0bccfc204ac31a62cca9d31d4041f`
* `enumeration_root_sha256 = 083372560c69abd3b2ad57c83883ac3279b2fae863ac877c1fcf3693e3477377`
* `runtime_work = {readings 4, edges 2, endpoint_labels 10, residues_inspected 68,
  edge_candidates_inspected 17, partial_label_assignments 12,
  consistency_checks 12, result_validation_cross_links 33,
  distinct_assignments 2, retained_witnesses 2, candidate_materializations 0}`

## Scope correction to the prior isolated-evidence claim

The prior test
(`test_multi_candidate_a4_private_binding_evidence_is_isolated`) asserted a
global limitation: a `b < 1` endpoint table "necessarily" yields a second exact
delta class and therefore two translation classes. That is true **only when the
non-integer reading is shared by the spanning tree**. The disjoint-chord
construction is a counterexample to the global wording, and public
multi-candidate acceptance is now demonstrated. Nothing is weakened:

* the shared-reading all-half-period ring is still refused at the gate, and its
  test is unchanged;
* a multi-candidate domain still cannot reach `endpoint_unique`, because the
  general search is never exhaustive and `_factorized_single_delta_class`
  remains the mandatory singleton eligibility proof;
* the frozen source, gate conditions, budgets and validation cross-links are
  untouched.

## Test changes and timing

Only `tests/test_plane_endpoint_provenance.py` was edited. Added:

* `_public_multicandidate_chord_case()` builder,
* module-scoped `public_multicandidate_chord` fixture,
* `test_public_multicandidate_chord_passes_gate_with_multiple_em`.

No existing test was modified. Focused run:

* `python -m pytest tests/test_plane_endpoint_provenance.py -q` → `60 passed`
  (59 accepted tests preserved + 1 new), wall `~2.22 s`;
* new test: `0.05 s` call, `0.09 s` setup;
* public `analyze_endpoint_provenance` call itself: `~0.020 s`, byte-identical
  across three trials.

## Constraints honored

* `plane_endpoint_provenance.py` SHA `3CB6AACE...316E7` frozen and unedited.
* Only the two owned paths were touched: the test file and this report.
* No monkeypatch and no private V2-A helper as acceptance evidence.
* No gate weakened, no contract correction needed.
* No junction/reparse, recursive delete, encoded/obfuscated command,
  executable/installer, background/persistence, Defender/security/system write,
  launcher/client, or commit/stage/reset/clean.
