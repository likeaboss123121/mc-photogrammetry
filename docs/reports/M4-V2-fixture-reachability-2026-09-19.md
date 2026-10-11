# M4 V2 exact-P0 fixture reachability + C2-R-B partial-failure record repair (2026-09-19)

Task: `DS-M4-C2RB-record-repair` (bounded source repair on the
`m4-v2-exact-p0-fixture` line of work)
Status: **bounded repair implemented and verified; the recorded literal blocker
is resolved, but the exact-P0 graph is a genuine `contradiction`, not a
`unique_validated` component. The assignment's stated target outcome ("valid
signed, unique, exhaustive component") is unattainable from this fixture by any
record-construction repair, so the work stops here for lead/Astra review.**

This report supersedes the earlier text at this path. The prior version recorded
the exact-P0 fixture as an unrepaired stop at the C2-R-B component record. That
recorded defect is now repaired, and the repaired solver returns a valid
component. The graph's outcome is then established by exhaustive replay to be a
contradiction, which is a separate, upstream condition.

Write scope used (exactly three paths):

- `src/mc_photogrammetry/reconstruction/plane_relative_translation.py`
- `tests/test_plane_relative_translation.py`
- `docs/reports/M4-V2-fixture-reachability-2026-09-19.md` (this file)

`tests/test_plane_placement.py` was **not** edited. No placement, V1 support,
volume, wiring, manifest, evaluator, CLI, persistence or other module was
created or edited.

## 1. Reproduction (before the repair)

Frozen interpreter `.venv\Scripts\python.exe`, project root, foreground:

```text
python -m pytest tests\test_plane_placement.py -k recipe -o addopts= -q --no-header
-> 2 passed
```

The two passing tests were `test_recipe_p0_generation_is_copied_into_reading_evidence`
and `test_recipe_first_exact_mismatch_is_the_c2rb_component_record`. The second
asserted the recorded stop:

```text
c2rb_result.status              budget_exhausted
c2rb_result.budget_failure_reason runtime_count_mismatch
c2rb_result.components          ()
```

The genuine nine-reading / nine-match graph reaches the solver with
`support.refusal is None`, `p1d_result.status == "complete"` and a `complete` A4
result with nine domains.

### 1.1 Corrected diagnosis

The prior report attributed the swallowed `ValueError` to
`_singleton_component_result`. That is not the executed path. The canonical
tree edges are `R0->R1` (normal world axis 0, `represented_candidate_count == 9`)
and `R0->R2` (normal world axis 0, `represented_candidate_count == 3`), so
`_factorized_single_delta_class` correctly returns `None` for them and
`solve_relative_translation_graph` calls `_solve_general_component`. Instrumenting
`_make_component_result` shows the swallowed cause is:

```text
ValueError("C2-R-B component result is malformed")
  at C2RBComponentResult.__post_init__
  built by _make_component_output
  for outcome "contradiction" from _solve_general_component
```

## 2. The genuine defect

`C2RBComponentResult.__post_init__` carried a universal guard

```python
or len(self.cycle_residuals) != self.cycle_rank
```

while the outcome-specific validation immediately below requires a
`contradiction` to retain **exactly its failing chords**:

```python
(self.outcome == "contradiction" and self.cycle_rank > 0
 and count == 0 and self.exhaustive and bool(self.failing_chords)
 and len(self.cycle_residuals) == len(self.failing_chords)
 and tuple(value.chord_identity for value in self.cycle_residuals)
     == self.failing_chords
 and all(value.matching_provenance_count == 0
         for value in self.cycle_residuals)
 and self.ambiguity_class_count_lower_bound == 0)
```

`_make_component_output` therefore always builds a contradiction with a residual
tuple that is a **proper subset** of the chords whenever some cycles do close;
the universal guard then rejected every such genuine record. The existing
single-chord contradiction tests never exposed this because their
`cycle_rank == 1` makes the failing subset equal to the whole cycle set. The
exact-P0 graph has `cycle_rank == 7` with 2 failing chords, so it always tripped
the guard and the catch-all in `solve_relative_translation_graph` downgraded the
whole result to `budget_exhausted` / `runtime_count_mismatch`.

A minimal private reproduction (shared helper now in
`tests/test_plane_relative_translation.py`) uses a six-edge A/B/C/D graph whose
canonical tree has three chords; two chords close and one does not
(`cycle_rank == 3`, one failing chord). Before the repair
`_solve_general_component` raised `ValueError`; after it returns a valid
component.

## 3. The repair

`src/mc_photogrammetry/reconstruction/plane_relative_translation.py`:

- Removed only the universal `len(self.cycle_residuals) != self.cycle_rank`
  guard from `C2RBComponentResult.__post_init__`.
- Added a comment binding the residual length to the outcome branches below it.

No other condition, counter, budget, digest, certificate, replay or outcome check
was changed. The outcome branches remain the sole authority:

| Outcome | Residual contract (unchanged) |
| --- | --- |
| `unique_validated` | `len == cycle_rank` and identities `== chords` |
| `unique_unvalidated_tree` | no residuals |
| `ambiguous` | `len == cycle_rank` and identities `== chords` |
| `contradiction` | `len == len(failing_chords)` and identities `== failing_chords`, all provenance counts `0`, at least one failing chord |

Because those branches already bound the residual length in both directions,
removing the redundant universal guard does not weaken uniqueness, exhaustiveness,
replay, digest or budget invariants. A resealed contradiction with dropped or
duplicated residuals is still rejected (regression below).

## 4. Focused regressions added (owned test file)

`tests/test_plane_relative_translation.py`:

1. `_GeneralSearchDomain` / `_general_search_case` — extracted the six-edge
   synthetic private domain surface shared by the general-search controls. The
   existing
   `test_general_search_uses_one_shared_assignment_for_multiple_cycles` now uses
   the helper unchanged in outcome.
2. `test_partial_failure_contradiction_seals_a_valid_component_record` —
   `cycle_rank == 3`, exactly one failing chord; asserts a real
   `C2RBComponentResult` with `outcome == "contradiction"`, `exhaustive`,
   `translation_classes == ()`, residual identities equal to `failing_chords`,
   zero provenance counts, and a digest resealed over the proper subset.
3. `test_partial_failure_contradiction_still_binds_its_failing_chords` — a
   freshly resealed contradiction with removed residuals, with empty failing
   chords, or with duplicated residuals is still rejected by `__post_init__`.

## 5. Exact-P0 graph result after the repair

The unchanged recipe fixture now clears C2-R-B and reaches the downstream gates:

```text
c2rb_result.status              complete
c2rb_result.budget_failure_reason None
c2rb_result.connected           True
c2rb_result.placement_readiness review_required
components                      1
  outcome                       contradiction
  exhaustive                    True
  cycle_rank                    7
  translation_classes           0
  failing_chords                2
  cycle_residuals               2
runtime_work                    candidate_views_scanned=12,
                                partial_assignments=3,
                                closure_component_checks=6,
                                membership_probes=134,
                                memoized_terms=12,
                                provenance_witnesses=0,
                                candidate_at_calls=0,
                                certificates_retained=0
deterministic_work              components=1, frames=3, edges=9,
                                tree_edges=2, chords=7, cycle_rows=21,
                                p1d_entries=37, residue_values=36,
                                represented_candidate_count=37
v2a_result.status               refused
v2a_result.classification       None
v2a_result.refusals             [gate_not_ready]
v2b_result.status               abstained
v2b_result.prediction.reason    input_gate_not_unique
```

Nothing was fabricated and no gate was weakened: the contradiction is derived by
the unchanged A4 search, and V2-A/V2-B refuse/abstain on the unchanged gates.

## 6. Why the graph cannot yield `unique_validated`

The repaired solver does not produce a unique component because the exact-P0
graph is genuinely inconsistent. Exhaustive replay over the canonical spanning
tree, using the solver's own `_evaluate_tree_assignment` at
`closure_tolerance == 0`, found **0 feasible global assignments out of 27**
(9 tree classes for `R0->R1` x 3 for `R0->R2`). A direct pairwise witness needs
no enumeration:

- `R0->R1` normal world axis 0 (`n0->2`): the C1 normal constraint fixes the
  world-axis-0 relative translation at `1.5` for every candidate.
- `R0->R1` normal world axis 1 (`n1->1`): every lateral candidate has
  world-axis-0 relative translation `0.5`.

Both edges connect the same two frames, so no single per-frame translation can
satisfy both. The disagreement is between the C1 normal component and the A4
lateral component (P1-D shift plus P1-B residue formula) for the recipe's mixed
absolute quadrants and mixed sides. The accepted phase-0 mixed-normal fixture
and the single-normal-axis nonzero fixture never combine both sources on the same
world axis with nonzero quadrants, so neither exposes it.

Consequences for scope:

- No change contained to `plane_relative_translation.py` can make this fixture
  `unique_validated`; the edge deltas are produced upstream by
  `plane_vector_domain.py` / `plane_axis_support.py`.
- Making the fixture unique requires either reconstructing the recipe's
  signatures/residues so the C1 normal and A4 lateral deltas agree, or an
  upstream sign-consistency repair to the C1/A4 agreement. Both are outside the
  assigned write scope.

## 7. Exact commands and outcomes

Frozen interpreter `.venv\Scripts\python.exe`, project root, foreground:

1. `python -m py_compile src\mc_photogrammetry\reconstruction\plane_relative_translation.py tests\test_plane_relative_translation.py`
   -> exit 0, no output.
2. `python -m pytest tests\test_plane_relative_translation.py -o addopts= -q --no-header`
   -> **38 passed in 2.48s**.
3. `python -m pytest tests\test_plane_relative_translation.py -o addopts= -q --no-header -k "partial_failure or general_search"`
   -> **3 passed, 35 deselected**.
4. `python -m pytest tests\test_plane_placement.py -k recipe -o addopts= -q --no-header`
   -> **1 failed, 1 passed**. The failure is
   `test_recipe_first_exact_mismatch_is_the_c2rb_component_record`:
   `assert solved.status == "budget_exhausted"` now sees `'complete'`.
5. `python -m pytest tests\test_plane_placement.py -o addopts= -q --no-header`
   -> **1 failed, 305 passed in 15.54s**; the only failure is the same stale
   recorded-stop test.
6. Adjacent gates:
   `python -m pytest tests\test_plane_endpoint_provenance.py tests\test_plane_phase_anchor.py tests\test_plane_region_support.py tests\test_plane_vector_domain.py tests\test_plane_shift_domain.py tests\test_plane_axis_support.py -o addopts= -q --no-header`
   -> **306 passed in 17.52s**.

The step-4/5 failure is not a hidden regression: that test is an out-of-scope
"recorded stop" that asserted the pre-repair buggy behavior. It is left
unmodified (`tests/test_plane_placement.py` is not in the owned write scope) and
is escalated below.

## 8. Scope, hashes and dispositions

Changed by this task:

| Path | Lines | SHA-256 |
| --- | ---: | --- |
| `src/mc_photogrammetry/reconstruction/plane_relative_translation.py` | 2018 | `6B0D971E1BADC4AE25001850670D248D3BD19F342CC4605C0513B3CC758CE113` |
| `tests/test_plane_relative_translation.py` | 2004 | `882D65B6850013E694691EC9D21542F4D8380B8222CBA306CE17AAA4E6DA5AF0` |
| `docs/reports/M4-V2-fixture-reachability-2026-09-19.md` | — | recomputed after the final edit and reported in the run result |

Unchanged (recomputed, so no placement path was touched):

| Path | SHA-256 |
| --- | --- |
| `tests/test_plane_placement.py` | `8B80C8B426780DEB266A8026ADFF88A183B9CED405CD453C55A55496AF7F3D2C` |
| `src/mc_photogrammetry/reconstruction/plane_placement.py` | `76153FDA373CD5F7EAF148B8B67DAAB8A4171E5978D82CA2560230A0578C5847` |

Disposition against the assignment:

| Requirement | Disposition |
| --- | --- |
| Reproduce `-k recipe` | **Done.** §1. |
| Fix the genuine malformed / `runtime_count_mismatch` construction defect | **Done.** §2-3. Only the erroneous universal guard changed. |
| Exact nine-reading/nine-match graph yields a valid signed unique exhaustive component | **Partially met, blocked on an upstream inconsistency.** The graph now yields a valid `contradiction` component (`contradiction`, `exhaustive`, `cycle_rank 7`, 2 failing chords); it is provably not unique. §5-6. |
| Preserve replay / uniqueness / exhaustive / budget invariants | **Preserved.** §3-4; focused and adjacent suites pass. |
| Do not weaken tests or broaden scope | **Honored.** No existing assertion was relaxed; only one redundant source guard was removed and focused regressions were added. |
| Report exact commands/results; stop on ambiguity | **Done.** §7 and §9. |

## 9. Ambiguity escalation (stop for lead/Astra review)

Two contract points are unresolved and cannot be resolved inside this scope:

1. **Target outcome.** The assignment asks for a `unique_validated` component,
   but the exact-P0 graph is a genuine contradiction (§6). The repaired solver
   cannot and should not return unique without an upstream C1/A4 or fixture
   change. Which target is authoritative: repair the fixture so the two delta
   sources agree, or accept a documented `contradiction` and drop the "unique"
   requirement?
2. **Stale owned-elsewhere test.** `tests/test_plane_placement.py:1663`
   (`test_recipe_first_exact_mismatch_is_the_c2rb_component_record`) records the
   pre-repair buggy stop and now fails by design. Updating it is the natural
   consequence of this repair, but the assignment excluded placement tests from
   the write scope. A one-line rerouting decision is needed: authorize the
   placement-test update, or move the assertion into the focused
   relative-translation file.

No promotion, V3 routing, placement, volume or production wiring is claimed.
