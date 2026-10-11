"""Synthetic-only controls for the isolated mandatory-factor endpoint CSP."""
from dataclasses import fields, replace
from itertools import product

import pytest

from mc_photogrammetry.reconstruction import plane_graph
from mc_photogrammetry.reconstruction.endpoint_hypothesis import (
    EndpointBudget, solve_endpoint_hypotheses as _solve,
)
from test_plane_graph import make_reading, make_match, lattice_component, _component_bytes


def readings_for(*frames, alternatives=2):
    result = []
    for f in frames:
        result.append(make_reading(f, 1, 1, [(float(i + 1), float(8 * (i + 1)), 1.)
                                           for i in range(alternatives)]))
        result.extend(make_reading(f, axis, 1, [(1., 8., 1.)]) for axis in (0, 2))
    return result


def solve_endpoint_hypotheses(readings, rows, **kwargs):
    """Old CSP controls plus explicit independent fixed-scale X/Z factors.

    The tested binary relations remain on vertical endpoints. These genuine
    synthetic auxiliary readings/matches make every frame observable on X/Z
    at each quadrant; they cannot repair any vertical relation inconsistency.
    Phase A exclusion controls call the real helper directly instead.
    """
    frames = tuple(sorted({r.frame for r in readings}))
    by_index, _, _ = plane_graph._canonical_reading_index(readings)
    supplied = list(rows)
    if frames:
        for right in frames[1:]:
            for q in range(4):
                for axis in (0, 2):
                    right_axis = next(a for a in (0, 2)
                                      if plane_graph.axis_transform(a, q)[0] == axis)
                    left_plane = next(i for (f, i), r in by_index.items()
                                      if f == frames[0] and r.axis == axis)
                    right_plane = next(i for (f, i), r in by_index.items()
                                       if f == right and r.axis == right_axis)
                    supplied.append(make_match(frames[0], right, axis, right_axis,
                                               q, left_plane, right_plane))
    return _solve(readings, supplied, session_frames=frames, **kwargs)


def vertical_alternatives(choice):
    return tuple(row for row in choice.alternatives if row[1] == 1)


def vertical_factors(choice):
    return tuple(f for f in choice.factors if f.endpoints[0][1] == 1)


def edge(left, right, a=0, b=0, **kw):
    return make_match(left, right, 1, 1, kw.pop("quadrant", 0),
                      kw.pop("left_plane", 1), kw.pop("right_plane", 1),
                      left_b=float(a + 1), right_b=float(b + 1), **kw)


def fixed(*frames):
    return tuple((frames[0], frame, 0) for frame in frames[1:])


def reverse(row):
    shift = row.cell_shift_rc
    return replace(row, left=row.right, right=row.left,
                   left_axis=row.right_axis, right_axis=row.left_axis,
                   left_plane=row.right_plane, right_plane=row.left_plane,
                   left_b=row.right_b, right_b=row.left_b,
                   quadrant_of_right=(-row.quadrant_of_right) % 4,
                   cell_shift_rc=None if shift is None else tuple(-x for x in shift))


def canonical_choices(result):
    return {(choice.quadrants, choice.alternatives) for choice in result.choices}


def test_unique_supported_alternatives_and_original_row_indices():
    readings = readings_for("a", "b", "c")
    rows = [edge("a", "b", 1, 1), edge("b", "c"), edge("a", "b")]
    result = solve_endpoint_hypotheses(readings, rows, quadrant_constraints=fixed("a", "b", "c"))
    assert result.status == "unique" and result.exhaustive
    assert vertical_alternatives(result.choices[0]) == (("a", 1, 0), ("b", 1, 0), ("c", 1, 0))
    assert [f.source_rows for f in vertical_factors(result.choices[0])] == [(2,), (1,)]
    assert result.counters["planned_assignments"] == result.counters["visited_assignments"] == 1
    assert result.pruning["values_removed"] == 3
    assert len(result.source_digest) == 64
    assert solve_endpoint_hypotheses(readings, rows, quadrant_constraints=fixed("a", "b", "c")) == result
    with pytest.raises(TypeError):
        result.counters["branch_nodes"] = 0


def test_r1_cross_incident_singletons_cannot_delete_an_edge():
    result = solve_endpoint_hypotheses(readings_for("a", "b", "c"),
        [edge("a", "b"), edge("a", "c", 1)], quadrant_constraints=fixed("a", "b", "c"))
    assert result.status == "no" and result.exhaustive and result.choices == ()
    assert result.counters["active_factor_visits"] == 6
    assert result.pruning["inconsistent_nodes"] == 1


def test_arc_consistent_but_globally_inconsistent_cycle():
    rows = [edge("a", "b", i, i) for i in range(2)]
    rows += [edge("b", "c", i, i) for i in range(2)]
    rows += [edge("a", "c", i, 1-i) for i in range(2)]
    result = solve_endpoint_hypotheses(readings_for("a", "b", "c"), rows,
                                     quadrant_constraints=fixed("a", "b", "c"))
    assert result.status == "no" and result.exhaustive
    assert result.counters["branch_nodes"] == 3
    assert result.pruning["inconsistent_nodes"] == 2
    assert result.reason == "canonical_constraints_inconsistent"
    assert result.counters["eligible_assignments"] == 1


@pytest.mark.parametrize("permuted", [False, True])
def test_two_interpretations_abstain_independent_score_order_duplicates(permuted):
    readings = readings_for("a", "b")
    rows = [edge("a", "b", 0, 0, score=-1e6), edge("a", "b", 1, 1, score=1e6)]
    if permuted:
        readings.reverse()
        rows = [reverse(rows[1]), reverse(rows[0]), reverse(rows[0])]
    result = solve_endpoint_hypotheses(readings, rows, quadrant_constraints=fixed("a", "b"))
    assert result.status == "multiple" and not result.exhaustive
    assert result.counters["solutions_found"] == len(result.choices) == 2
    assert {tuple(a[2] for a in vertical_alternatives(c)) for c in result.choices} == {(0, 0), (1, 1)}
    assert not result.evidence_ambiguous  # exact duplicates do not add evidence weight


def test_equivalent_noisy_b_and_distinguishable_evidence_accounted():
    readings = readings_for("a", "b", alternatives=1)
    row = edge("a", "b")
    noisy = replace(row, left_b=1.+5e-10, right_b=1.-5e-10)
    result = solve_endpoint_hypotheses(readings, [row, noisy, row], quadrant_constraints=fixed("a", "b"))
    assert result.status == "unique" and not result.evidence_ambiguous
    assert vertical_factors(result.choices[0])[0].source_rows == (0, 1, 2)
    assert vertical_factors(result.choices[0])[0].distinguishable_support_groups == ((0, 1, 2),)
    distinct = solve_endpoint_hypotheses(readings, [row, replace(row, cell_shift_rc=(1, 2))],
                                       quadrant_constraints=fixed("a", "b"))
    assert distinct.status == "unresolved_evidence" and distinct.evidence_ambiguous
    assert distinct.choices == () and distinct.counters["solutions_found"] == 1
    assert distinct.reason == "distinguishable_factor_support" and distinct.exhaustive


def test_inactive_quadrant_excluded_and_reversal_uses_negative_label():
    readings = readings_for("a", "b")
    rows = [edge("a", "b", quadrant=1), edge("a", "b", 1, 1, quadrant=2)]
    constraints = (("a", "b", 1),)
    result = solve_endpoint_hypotheses(readings, rows, quadrant_constraints=constraints)
    reversed_result = solve_endpoint_hypotheses(readings[::-1], [reverse(r) for r in rows[::-1]],
                                               quadrant_constraints=(("b", "a", 3),))
    assert result.status == reversed_result.status == "unique"
    assert canonical_choices(result) == canonical_choices(reversed_result)
    assert vertical_factors(result.choices[0])[0].source_rows == (0,)
    assert vertical_factors(reversed_result.choices[0])[0].source_rows == (1,)
    assert result.source_digest != reversed_result.source_digest


def test_same_frame_pair_different_plane_endpoints_stay_separate():
    readings = readings_for("a", "b", alternatives=1)
    readings += [replace(r, period_pixels=61.) for r in readings if r.axis == 1]
    rows = [edge("a", "b"), edge("a", "b", left_plane=2, right_plane=2)]
    result = solve_endpoint_hypotheses(readings, rows, quadrant_constraints=fixed("a", "b"))
    assert result.status == "unique" and len(result.choices[0].factors) == 4
    assert {f.endpoints for f in result.choices[0].factors if f.endpoints[0][1] in (1, 2)} == {
        (("a", 1), ("b", 1)), (("a", 2), ("b", 2))}


def test_consistent_three_axes_no_production_evaluator_or_mutation(monkeypatch):
    readings, rows, _ = lattice_component()
    before = _component_bytes(plane_graph.solve_component(readings, rows))
    snapshot = repr((readings, rows))
    monkeypatch.setattr(plane_graph, "_evaluate_assignment", lambda *a: pytest.fail("CSP called geometry"))
    result = _solve(readings, rows, session_frames=("v2", "v0", "v1"),
                                     quadrant_constraints=(("v0", "v1", 1), ("v0", "v2", 0)))
    assert result.status == "unique" and len(result.choices[0].alternatives) == len(readings)
    assert repr((readings, rows)) == snapshot
    monkeypatch.undo()
    assert _component_bytes(plane_graph.solve_component(readings, rows)) == before


@pytest.mark.parametrize("name", [f.name for f in fields(EndpointBudget)])
def test_every_budget_refuses_without_partial_winner(name):
    result = solve_endpoint_hypotheses(readings_for("a", "b", alternatives=1), [edge("a", "b")],
        quadrant_constraints=fixed("a", "b"), budget=replace(EndpointBudget(), **{name: 0}))
    assert result.status == "budget_refused" and result.choices == () and not result.exhaustive
    assert result.denial[0] == name
    _, used, requested, cap = result.denial
    assert used <= cap and used + requested > cap


def test_budget_exhaustion_after_winner_and_global_across_quadrants():
    result = solve_endpoint_hypotheses(readings_for("a", "b"),
        [edge("a", "b"), edge("a", "b", 1, 1)], quadrant_constraints=fixed("a", "b"),
        budget=replace(EndpointBudget(), solution_rows=1))
    assert result.status == "budget_refused" and result.choices == ()
    assert result.counters["solutions_found"] == 2 and result.counters["solution_rows"] == 1
    across = solve_endpoint_hypotheses(readings_for("a", "b", alternatives=1),
                                     [edge("a", "b", quadrant=q) for q in range(4)],
                                     budget=replace(EndpointBudget(), branch_nodes=1))
    assert across.status == "budget_refused" and across.choices == ()
    assert across.counters["planned_assignments"] == 4
    assert across.counters["visited_assignments"] == 2
    assert across.counters["solutions_found"] == across.counters["branch_nodes"] == 1


def test_unconstrained_nine_preserves_existing_quadrant_budget():
    result = solve_endpoint_hypotheses(readings_for(*map(str, range(9)), alternatives=1), [])
    assert result.status == "budget_refused" and result.choices == ()
    assert result.denial[0] == "quadrant_assignments"
    assert result.counters["planned_assignments"] is None
    assert result.counters["visited_assignments"] == 0
    assert plane_graph.MAX_ASSIGNMENTS == 16384


def test_structural_precharge_before_canonical_allocation(monkeypatch):
    monkeypatch.setattr(plane_graph, "_canonical_reading_index", lambda *a: pytest.fail("allocated"))
    result = _solve(readings_for("a"), [], session_frames=("a",),
                                     budget=replace(EndpointBudget(), endpoint_variables=0))
    assert result.status == "budget_refused" and result.source_digest is None


def test_contradictory_constraints_bind_all_rows_before_no_result():
    readings = readings_for("a", "b", alternatives=1)
    constraints = (("a", "b", 0), ("a", "b", 1))
    first = solve_endpoint_hypotheses(readings, [edge("a", "b")], quadrant_constraints=constraints)
    second = solve_endpoint_hypotheses(readings, [edge("a", "b", score=2.)], quadrant_constraints=constraints)
    assert first.status == second.status == "no" and first.exhaustive
    assert first.source_digest != second.source_digest
    assert first.counters["visited_assignments"] == 0


@pytest.mark.parametrize("name,value", [("branch_nodes", True), ("frames", 33), ("solution_rows", -1)])
def test_budget_cannot_raise_hard_ceilings(name, value):
    with pytest.raises(ValueError):
        replace(EndpointBudget(), **{name: value})


def test_small_exhaustive_binary_relation_oracle():
    readings = readings_for("a", "b", "c")
    universe = list(product(range(2), repeat=2))
    # Enumerate all nonempty relations on AB and BC, plus a closing inequality.
    for mask_ab in range(1, 16):
        for mask_bc in range(1, 16):
            ab = {p for i, p in enumerate(universe) if mask_ab & (1 << i)}
            bc = {p for i, p in enumerate(universe) if mask_bc & (1 << i)}
            ac = {(0, 1), (1, 0)}
            expected = {v for v in product(range(2), repeat=3)
                        if v[:2] in ab and v[1:] in bc and (v[0], v[2]) in ac}
            rows = [edge("a", "b", *p) for p in sorted(ab)]
            rows += [edge("b", "c", *p) for p in sorted(bc)]
            rows += [edge("a", "c", *p) for p in sorted(ac)]
            result = solve_endpoint_hypotheses(readings, rows,
                                             quadrant_constraints=fixed("a", "b", "c"))
            assert result.status == ("no" if not expected else "unique" if len(expected) == 1 else "multiple")
            certificates = {tuple(a[2] for a in vertical_alternatives(c)) for c in result.choices}
            assert certificates <= expected
            assert len(certificates) == min(2, len(expected))


def connected_fixture(*frames):
    """One reading/axis/frame and literal axis-spanning star factors."""
    readings = readings_for(*frames, alternatives=1)
    rows = [make_match(frames[0], frame, axis, axis, 0, axis, axis)
            for frame in frames[1:] for axis in range(3)]
    return readings, rows


def test_no_factors_excludes_every_quadrant_without_vacuous_solution():
    readings, _ = connected_fixture("a", "b")
    result = _solve(readings, [], session_frames=("b", "a"))
    assert result.status == "no" and result.reason == "no_eligible_assignment"
    assert result.exhaustive and result.choices == () and not result.evidence_ambiguous
    assert result.session_frames == ("a", "b")
    assert result.counters["planned_assignments"] == result.counters["visited_assignments"] == 4
    assert result.counters["eligible_assignments"] == result.counters["solutions_found"] == 0
    assert result.counters["branch_nodes"] == 0
    assert result.eligibility_reasons == {"no_active_factors": 4}
    assert [r[0] for r in result.excluded_assignments] == list(range(4))
    assert result.counters["eligibility_hashed_assignments"] == 4


def test_singleton_endpoint_without_active_support_is_excluded_before_dfs():
    readings, rows = connected_fixture("a", "b")
    readings.append(replace(readings[0], period_pixels=61.))  # new A vertical endpoint
    rows = [replace(row, left_plane=3) if row.left_axis == 2 else row for row in rows]
    result = _solve(readings, rows, session_frames=("a", "b"), quadrant_constraints=fixed("a", "b"))
    assert result.status == "no" and result.eligibility_reasons == {"unsupported_reading_endpoint": 1}
    assert result.counters["solutions_found"] == result.counters["branch_nodes"] == 0
    assert len(result.session_frames) == 2


@pytest.mark.parametrize("session", [("a",), ("a", "b", "c"), ("a", "a", "b"), ()])
def test_session_mismatch_never_reduces_or_fabricates_session_frames(session):
    readings, rows = connected_fixture("a", "b")
    result = _solve(readings, rows, session_frames=session)
    assert result.status == "no" and result.reason == "session_frame_mismatch"
    assert result.session_frames == tuple(sorted(session))
    assert result.source_digest and result.choices == ()
    assert result.counters["planned_assignments"] == result.counters["visited_assignments"] == 0
    assert result.counters["solutions_found"] == 0


def test_session_is_explicit_immutable_and_source_bound():
    readings, rows = connected_fixture("a", "b")
    with pytest.raises(TypeError):
        _solve(readings, rows)
    with pytest.raises(ValueError):
        _solve(readings, rows, session_frames=["a", "b"])
    normal = _solve(readings, rows, session_frames=("a", "b"), quadrant_constraints=fixed("a", "b"))
    reordered = _solve(readings, rows, session_frames=("b", "a"), quadrant_constraints=fixed("a", "b"))
    assert normal.status == reordered.status == "unique"
    assert canonical_choices(normal) == canonical_choices(reordered)
    assert normal.source_digest != reordered.source_digest
    assert normal.choices[0].quadrants[0] == ("a", 0)  # production sorted anchor


@pytest.mark.parametrize("disconnected_axis", range(3))
def test_every_axis_requires_all_frames_connected_to_production_anchor(disconnected_axis):
    frames = ("a", "b", "c", "d")
    readings, rows = connected_fixture(*frames)
    rows = [row for row in rows if row.left_axis != disconnected_axis]
    rows += [make_match("a", "b", disconnected_axis, disconnected_axis, 0,
                        disconnected_axis, disconnected_axis),
             make_match("c", "d", disconnected_axis, disconnected_axis, 0,
                        disconnected_axis, disconnected_axis)]
    result = _solve(readings, rows, session_frames=frames, quadrant_constraints=fixed(*frames))
    assert result.status == "no" and result.choices == ()
    assert result.eligibility_reasons == {f"axis_{disconnected_axis}_disconnected": 1}
    assert result.counters["branch_nodes"] == 0


def test_active_world_axis_mismatch_is_not_dropped_to_restore_connectivity():
    readings, rows = connected_fixture("a", "b")
    rows.append(make_match("a", "b", 0, 2, 0, 0, 2))
    with pytest.raises(ValueError, match="fails closed globally"):
        _solve(readings, rows, session_frames=("a", "b"), quadrant_constraints=fixed("a", "b"))


def test_only_globally_eligible_quadrant_counts_toward_unique():
    readings, rows = connected_fixture("a", "b")
    result = _solve(readings, rows, session_frames=("a", "b"))
    assert result.status == "unique" and result.exhaustive
    assert result.counters["planned_assignments"] == result.counters["visited_assignments"] == 4
    assert result.counters["eligible_assignments"] == result.counters["solutions_found"] == 1
    assert result.eligibility_reasons == {"no_active_factors": 3}
    assert result.choices[0].quadrants == (("a", 0), ("b", 0))
    assert result.counters["excluded_assignments"] + result.counters["eligible_assignments"] == 4
    assert result.counters["eligibility_hashed_assignments"] == 4


def test_evidence_ambiguity_terminal_retains_no_unique_candidate_and_is_deterministic():
    readings, rows = connected_fixture("a", "b")
    rows.append(replace(rows[0], cell_shift_rc=(2, -3)))
    result = _solve(readings, rows, session_frames=("a", "b"))
    reordered = _solve(readings[::-1], [reverse(row) for row in rows[::-1]], session_frames=("b", "a"))
    assert result.status == reordered.status == "unresolved_evidence"
    assert result.reason == reordered.reason == "distinguishable_factor_support"
    assert result.evidence_ambiguous and reordered.evidence_ambiguous
    assert result.choices == reordered.choices == ()
    assert result.counters["solutions_found"] == reordered.counters["solutions_found"] == 1
    assert result.exhaustive and reordered.exhaustive
    assert result.eligibility_digest == reordered.eligibility_digest
    assert result.source_digest != reordered.source_digest


def test_exclusion_examples_truncate_but_complete_digest_accounting_does_not():
    frames = tuple(f"f{i}" for i in range(5))
    readings = readings_for(*frames, alternatives=1)
    result = _solve(readings, [], session_frames=frames)
    assert result.status == "no" and result.exhaustive
    assert result.counters["planned_assignments"] == result.counters["visited_assignments"] == 256
    assert result.counters["excluded_assignments"] == result.counters["eligibility_hashed_assignments"] == 256
    assert result.counters["retained_exclusions"] == result.counters["omitted_exclusions"] == 128
    assert len(result.excluded_assignments) == 128
    assert result.eligibility_reasons == {"no_active_factors": 256}
    assert result.eligibility_digest == _solve(readings, [], session_frames=frames).eligibility_digest


def test_global_work_exhaustion_after_eligible_candidate_returns_no_winner():
    readings, rows = connected_fixture("a", "b")
    forced = _solve(readings, rows, session_frames=("a", "b"), quadrant_constraints=fixed("a", "b"))
    result = _solve(readings, rows, session_frames=("a", "b"),
                    budget=replace(EndpointBudget(), relation_visits=forced.counters["relation_visits"]))
    assert result.status == "budget_refused" and result.choices == ()
    assert result.counters["solutions_found"] == 1
    assert result.counters["visited_assignments"] == 2
    assert result.counters["eligibility_hashed_assignments"] == 1
    assert result.counters["eligibility_unclassified_assignments"] == 1
    assert result.counters["unvisited_assignments"] == 2
    assert result.denial[0] == "relation_visits" and not result.exhaustive


def test_multiple_eligible_quadrants_are_distinct_hypotheses_with_exact_coverage():
    readings, rows = connected_fixture("a", "b")
    rows += [replace(row, quadrant_of_right=2) for row in rows]
    result = _solve(readings, rows, session_frames=("a", "b"))
    assert result.status == "multiple" and not result.exhaustive
    assert {dict(choice.quadrants)["b"] for choice in result.choices} == {0, 2}
    assert result.counters["eligible_assignments"] == result.counters["solutions_found"] == 2
    assert result.counters["excluded_assignments"] == 1
    assert result.eligibility_reasons == {"no_active_factors": 1}
    assert result.counters["visited_assignments"] == result.counters["eligibility_hashed_assignments"] == 3
    assert result.counters["eligibility_unclassified_assignments"] == 0
    assert result.counters["unvisited_assignments"] == 1


from mc_photogrammetry.reconstruction import endpoint_hypothesis as csp


def _exhaustive_active_factor_oracle(work, index, quadrants):
    # Independent old exhaustive predicate: no frame-pair bucket lookup.
    ranked = sorted(((*entry, relative) for _, buckets in index
                     for relative, bucket in enumerate(buckets) for entry in bucket),
                    key=lambda entry: entry[0])
    result = []
    for _, (u, v, relation), relative in ranked:
        work.charge("relation_visits", 1)
        if (quadrants[v[0]] - quadrants[u[0]]) % 4 == relative:
            result.append((u, v, relation))
    return result


@pytest.mark.parametrize("reordered", [False, True])
def test_factor_activation_index_all_quadrants_canonical_order_relation_identity_and_charges(reordered):
    entries = [(((left, li), (right, ri), relative), {(0, 0): [position, position + 100]})
               for position, (left, right, li, ri, relative) in enumerate(
                   product(("a",), ("b", "c"), range(2), range(2), range(4)))]
    entries += [((("b", 0), ("c", 1), q), {(0, 0): [200 + q]}) for q in range(4)]
    registry = dict(reversed(entries) if reordered else entries)
    snapshot = [(key, id(value), tuple(value[(0, 0)])) for key, value in registry.items()]
    work = csp._Work(EndpointBudget())
    index = csp._factor_activation_index(work, registry)
    pairs = len({(u[0], v[0]) for u, v, _ in registry})
    assert work.counts["relation_visits"] == len(registry) + pairs
    for q in product(range(4), repeat=3):
        quadrants = dict(zip(("a", "b", "c"), q))
        expected = [(u, v, relation) for (u, v, relative), relation in sorted(registry.items())
                    if (quadrants[v[0]] - quadrants[u[0]]) % 4 == relative]
        before = work.counts["relation_visits"]
        actual = csp._active_factors(work, index, quadrants)
        assert [(u, v, id(relation)) for u, v, relation in actual] == [
            (u, v, id(relation)) for u, v, relation in expected]
        assert work.counts["relation_visits"] - before == pairs + len(expected)
    assert [(key, id(value), tuple(value[(0, 0)])) for key, value in registry.items()] == snapshot


@pytest.mark.parametrize("reverse_rows", [False, True])
def test_factor_index_real_source_normalization_reversed_pairs_quadrants_and_duplicate_rows(reverse_rows, monkeypatch):
    readings, rows = connected_fixture("a", "b")
    rows = rows + [replace(row, quadrant_of_right=2) for row in rows] + [rows[0]]
    if reverse_rows:
        readings, rows = readings[::-1], [reverse(row) for row in rows[::-1]]
    captured = []
    original_index = csp._factor_activation_index
    def capture(work, registry):
        captured.append(registry)
        return original_index(work, registry)
    monkeypatch.setattr(csp, "_factor_activation_index", capture)
    result = _solve(readings, rows, session_frames=("b", "a"))
    assert result.status == "multiple" and len(captured) == 1
    registry = captured[0]
    assert all(u < v for u, v, _ in registry)
    for (u, v, relative), relation in registry.items():
        for row_indices in relation.values():
            for row_index in row_indices:
                row = rows[row_index]
                assert {u, v} == {(row.left, row.left_plane), (row.right, row.right_plane)}
                assert relative == (row.quadrant_of_right if row.left == u[0] else (-row.quadrant_of_right) % 4)
    for choice in result.choices:
        q = dict(choice.quadrants)
        expected_keys = [(u, v) for u, v, relative in sorted(registry)
                         if (q[v[0]] - q[u[0]]) % 4 == relative]
        assert [factor.endpoints for factor in choice.factors] == expected_keys
        for factor in choice.factors:
            relative = (q[factor.endpoints[1][0]] - q[factor.endpoints[0][0]]) % 4
            assert factor.source_rows == tuple(registry[(*factor.endpoints, relative)][factor.alternatives])


@pytest.mark.parametrize("state", ["no", "unique", "multiple", "unresolved_evidence"])
def test_indexed_and_exhaustive_activation_semantic_proof_parity(state, monkeypatch):
    readings, rows = connected_fixture("a", "b")
    if state == "no":
        rows = []
    elif state == "multiple":
        rows += [replace(row, quadrant_of_right=2) for row in rows]
    elif state == "unresolved_evidence":
        rows.append(replace(rows[0], cell_shift_rc=(2, -3)))
    indexed = _solve(readings, rows, session_frames=("a", "b"))
    with monkeypatch.context() as patch:
        patch.setattr(csp, "_active_factors", _exhaustive_active_factor_oracle)
        exhaustive = _solve(readings, rows, session_frames=("a", "b"))
    assert indexed.status == exhaustive.status == state
    for field in fields(indexed):
        if field.name != "counters":
            assert getattr(indexed, field.name) == getattr(exhaustive, field.name), field.name
    for name in indexed.counters:
        if name != "relation_visits":
            assert indexed.counters[name] == exhaustive.counters[name], name


def _large_vertical_anchor_fixture():
    # Authored topology/control dimensions, not recovered historical evidence.
    # Every endpoint is supported; local-axis1 never transforms to world X/Z.
    frames = tuple(f"f{i}" for i in range(6))
    readings = [make_reading(frame, 1, 1, [(1., 8., 1.)], period=60. + plane)
                for frame in frames for plane in range(3)]
    required = {((frames[0], 0), (frame, plane), q)
                for frame in frames[1:] for plane in range(3) for q in range(4)}
    required.update(((frames[0], plane), (frames[1], 0), q)
                    for plane in (1, 2) for q in range(4))
    universe = [((left, lp), (right, rp), q) for left in frames for right in frames
                if left < right for lp in range(3) for rp in range(3) for q in range(4)]
    for key in universe:
        if len(required) == 214:
            break
        required.add(key)
    assert len(required) == 214
    seeds = [make_match(u[0], v[0], 1, 1, q, u[1], v[1]) for u, v, q in sorted(required)]
    rows = [seeds[i % len(seeds)] for i in range(4086)]
    return readings, rows, frames


def test_large_disconnected_topology_exhausts_all1024_under_unchanged_relation_cap(monkeypatch):
    readings, rows, frames = _large_vertical_anchor_fixture()
    snapshot = repr((readings, rows))
    indexed = _solve(readings, rows, session_frames=frames)
    assert EndpointBudget().relation_visits == 262144
    assert indexed.status == "no" and indexed.exhaustive and indexed.choices == ()
    assert indexed.reason == "no_eligible_assignment"
    assert indexed.counters["endpoint_variables"] == 18
    assert indexed.counters["input_match_rows"] == 4086 and indexed.counters["registry_factors"] == 214
    assert indexed.counters["planned_assignments"] == indexed.counters["visited_assignments"] == 1024
    assert indexed.counters["eligibility_hashed_assignments"] == indexed.counters["excluded_assignments"] == 1024
    assert indexed.counters["unvisited_assignments"] == indexed.counters["eligibility_unclassified_assignments"] == 0
    assert indexed.eligibility_reasons == {"axis_0_disconnected": 1024}
    assert indexed.counters["branch_nodes"] == indexed.counters["physical_assignments"] == 0
    assert indexed.counters["eligible_assignments"] == 0 and indexed.counters["relation_visits"] < 200000
    with monkeypatch.context() as patch:
        patch.setattr(csp, "_active_factors", _exhaustive_active_factor_oracle)
        exhaustive = _solve(readings, rows, session_frames=frames)
    assert exhaustive.status == "budget_refused" and exhaustive.choices == () and not exhaustive.exhaustive
    assert exhaustive.denial[0] == "relation_visits" and exhaustive.denial[3] == 262144
    assert exhaustive.counters["visited_assignments"] < 1024
    assert exhaustive.counters["branch_nodes"] == exhaustive.counters["physical_assignments"] == 0
    assert repr((readings, rows)) == snapshot


@pytest.mark.parametrize("fail_last_orientation", [False, True])
def test_inactive_source_row_all_four_physical_orientations_still_preflighted(fail_last_orientation, monkeypatch):
    readings, rows = connected_fixture("a", "b")
    inactive = replace(rows[1], quadrant_of_right=1)
    rows.append(inactive)
    seen = []
    original = plane_graph.signed_physical_relation
    def checked(indexed, row, q, selected, row_index, **kwargs):
        if row is inactive:
            seen.append((row_index, q[row.left], q[row.right]))
            if fail_last_orientation and q[row.left] == 3:
                raise ValueError("authored invalid orientation")
        return original(indexed, row, q, selected, row_index, **kwargs)
    monkeypatch.setattr(plane_graph, "signed_physical_relation", checked)
    if fail_last_orientation:
        monkeypatch.setattr(csp, "_factor_activation_index", lambda *a: pytest.fail("invalid source entered activation index"))
        with pytest.raises(ValueError, match="unverified physical displacement fails closed globally"):
            _solve(readings, rows, session_frames=("a", "b"), quadrant_constraints=fixed("a", "b"))
    else:
        result = _solve(readings, rows, session_frames=("a", "b"), quadrant_constraints=fixed("a", "b"))
        assert result.status == "unique"
    assert seen == [(3, absolute, (absolute + 1) % 4) for absolute in range(4)]


def test_index_exact_relation_boundary_refuses_without_winner_and_closes_partial_accounting():
    readings, rows = connected_fixture("a", "b")
    normal = _solve(readings, rows, session_frames=("a", "b"))
    cap = normal.counters["relation_visits"]
    exact = _solve(readings, rows, session_frames=("a", "b"), budget=replace(EndpointBudget(), relation_visits=cap))
    assert exact == normal and exact.status == "unique"
    refused = _solve(readings, rows, session_frames=("a", "b"), budget=replace(EndpointBudget(), relation_visits=cap - 1))
    assert refused.status == "budget_refused" and refused.choices == () and not refused.exhaustive
    assert refused.denial == ("relation_visits", cap - 1, 1, cap - 1)
    c = refused.counters
    assert c["solutions_found"] == 1
    assert c["eligible_assignments"] + c["excluded_assignments"] + c["eligibility_unclassified_assignments"] == c["visited_assignments"]
    assert c["visited_assignments"] + c["unvisited_assignments"] == c["planned_assignments"] == 4
    assert c["eligibility_unclassified_assignments"] == 1


def test_factor_index_registry_ceiling_precedes_sorted_allocation(monkeypatch):
    work = csp._Work(replace(EndpointBudget(), registry_factors=0))
    registry = {(("a", 0), ("b", 0), 0): {(0, 0): [0]}}
    monkeypatch.setattr(csp, "sorted", lambda *a, **k: pytest.fail("index allocation before bound"), raising=False)
    with pytest.raises(csp._Exhausted):
        csp._factor_activation_index(work, registry)
    assert work.denial == ("registry_factors", 0, 1, 0)
    assert work.counts["relation_visits"] == 0
