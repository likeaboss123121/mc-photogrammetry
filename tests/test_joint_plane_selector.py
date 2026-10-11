from __future__ import annotations

from dataclasses import replace
import ast
from itertools import product
import math
from pathlib import Path
import random

import pytest

from mc_photogrammetry.reconstruction import plane_graph, lattice_planes
from mc_photogrammetry.reconstruction import _joint_plane_selector as selector


RECTIFICATION = lattice_planes.PlaneRectification(
    axis_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    focal=500.0,
    principal_point=(120.0, 90.0),
    centre=(0.0, 0.0),
)


def _reading(frame: str, axis: int, *, b_values=(1.0,), frame_offset=0) -> lattice_planes.PlaneReading:
    distance = float(3 + frame_offset + axis)
    alternatives = tuple(
        lattice_planes.DistanceAlternative(float(b), distance, 1.0)
        for b in b_values
    )
    return lattice_planes.PlaneReading(
        frame=frame, axis=axis, side=-1.0, alternatives=alternatives,
        period_pixels=60.0, phase_pixels=(0.0, 0.0), extent=1.0,
        samples=512, polygon=None, coverage_cells=8, structural_share=0.6,
        confidence=0.9, rectification=RECTIFICATION,
    )


def _axis_for_relative(left_axis: int, relative_q: int) -> int:
    return {0: 2, 1: 1, 2: 0}[left_axis] if relative_q % 2 else left_axis


def _match(
    left: str,
    right: str,
    left_axis: int,
    right_axis: int,
    relative_q: int,
    left_plane: int,
    right_plane: int,
    *,
    left_b=1.0,
    right_b=1.0,
    score=1.0,
) -> plane_graph.PlaneMatch:
    return plane_graph.PlaneMatch(
        left=left, right=right, left_axis=left_axis, right_axis=right_axis,
        left_b=float(left_b), right_b=float(right_b),
        quadrant_of_right=relative_q, psr=5.0, peak=1.0,
        second_peak=0.1, overlap_cells=64, integer_residual=None,
        score=float(score), left_plane=left_plane, right_plane=right_plane,
    )


def _two_frame_case(*, qrels=(1,), vary_b_axes=(), only_axes=(0, 1, 2)):
    readings = []
    options_by_axis = {}
    for frame_index, frame in enumerate(("a", "b")):
        for axis in range(3):
            options = (1.0, 0.5) if axis in vary_b_axes else (1.0,)
            readings.append(_reading(frame, axis, b_values=options, frame_offset=frame_index))
            options_by_axis[(frame, axis)] = options
    matches = []
    for relative_q in qrels:
        for left_axis in only_axes:
            right_axis = _axis_for_relative(left_axis, relative_q)
            for left_b in options_by_axis[("a", left_axis)]:
                for right_b in options_by_axis[("b", right_axis)]:
                    matches.append(_match(
                        "a", "b", left_axis, right_axis, relative_q,
                        left_axis, right_axis, left_b=left_b, right_b=right_b))
    return readings, matches


def _chain_case(frame_names=("a", "b", "c"), links=(("a", "b"), ("b", "c"))):
    readings = [
        _reading(frame, axis, frame_offset=2 * index)
        for index, frame in enumerate(frame_names)
        for axis in range(3)
    ]
    matches = [
        _match(left, right, axis, axis, 0, axis, axis)
        for left, right in links
        for axis in range(3)
    ]
    return readings, matches


def _cache_repeat_case():
    names = ("a", "b", "c", "d")
    readings = [
        _reading(frame, axis,
                 b_values=(1.0, 0.5) if (frame, axis) in (("c", 0), ("d", 2)) else (1.0,),
                 frame_offset=2 * index)
        for index, frame in enumerate(names) for axis in range(3)
    ]
    matches = [
        _match(left, right, axis, axis, 0, axis, axis)
        for left, right in (("a", "b"), ("b", "c"))
        for axis in range(3)
    ]
    # These two q variants populate the c0/d2 b domains. For q_d=0 they are
    # inactive, so changing d2's b label repeats the exact same active rows.
    matches.extend((
        _match("c", "d", 0, 2, 1, 0, 2, left_b=0.5, right_b=0.5),
        _match("c", "d", 0, 2, 3, 0, 2, left_b=1.0, right_b=1.0),
    ))
    return readings, matches


def _shared_multi_pair_b_case(domain_size: int):
    names = ("a", "b", "c")
    values = (1.0, 0.5, 0.25, 0.125, 2.0, 4.0, 8.0)[:domain_size]
    readings = []
    for frame in names:
        for axis in range(3):
            readings.append(_reading(
                frame, axis, b_values=values if frame == "b" and axis == 0 else (1.0,)))
    matches = []
    for left, right in (("a", "b"), ("b", "c")):
        for axis in range(3):
            if axis == 0:
                for value in values:
                    left_b, right_b = ((1.0, value) if left == "a"
                                       else (value, 1.0))
                    matches.append(_match(
                        left, right, axis, axis, 0, axis, axis,
                        left_b=left_b, right_b=right_b))
            else:
                matches.append(_match(left, right, axis, axis, 0, axis, axis))
    return readings, matches, ("b", 0)


def _budget(**overrides):
    values = dict(
        max_search_nodes=1_000_000,
        max_compatibility_checks=10_000_000,
        max_objective_rows=10_000_000,
        max_axis_solves=1_000_000,
        max_pareto_comparisons=10_000_000,
        max_candidate_certificates=100_000,
        max_buffered_match_rows=10_000_000,
    )
    values.update(overrides)
    return selector.SearchBudget(**values)


def _candidate_key(candidate: selector.JointCandidate):
    return (
        candidate.quadrants, candidate.blocks_per_period,
        tuple(sorted(candidate.pair_groups)),
        candidate.axis_pair_groups, candidate.resultants, candidate.residuals,
        candidate.translations, candidate.axis_observations,
        candidate.mean_match_score, candidate.cost,
    )


_LOGICAL_COUNTERS = tuple(
    name for name in selector.SearchCounters.__dataclass_fields__
    if (name != "axis_solves"
        and not name.startswith(("objective_cache_", "physical_axis_cache_", "diagnostic_")))
)


def _assert_bounded_parity(first, second):
    assert first.status == second.status
    assert first.reason == second.reason
    assert first.certificate_sha256 == second.certificate_sha256
    assert [_candidate_key(candidate) for candidate in first.pareto] == [
        _candidate_key(candidate) for candidate in second.pareto]
    assert {name: getattr(first.counters, name) for name in _LOGICAL_COUNTERS} == {
        name: getattr(second.counters, name) for name in _LOGICAL_COUNTERS}


def _legacy_bounded(readings, matches, budget):
    return selector._search_bounded(
        readings, matches, budget,
        min_resultant=0.90, integer_tolerance=0.15, optimized=False)


def _fake_candidate(tag, groups):
    group_set = frozenset(groups)
    return selector.JointCandidate(
        quadrants=((tag, 0),), blocks_per_period=(),
        pair_groups=group_set, covered_frames=frozenset({"f"}),
        covered_readings=frozenset(),
        axis_pair_groups=((0, group_set), (1, frozenset()), (2, frozenset())),
        resultants=((0, 0.9),), residuals=((0, 0.1),),
        translations=(("f", (0.0, 0.0, 0.0)),),
        axis_observations=(0, 0, 0), scale_known=True,
        mean_match_score=0.9, cost=0.1, _active_matches=(),
    )


def test_unique_joint_label_matches_exhaustive_oracle_and_legacy_fields():
    readings, matches = _two_frame_case(qrels=(1,))
    bounded = selector.select_joint_labels(readings, matches, budget=_budget())
    oracle = selector.exhaustive_oracle(readings, matches)
    legacy = plane_graph.solve_component(readings, matches)

    assert bounded.status == oracle.status == "selected"
    assert bounded.certificate_sha256 == oracle.certificate_sha256
    assert len(bounded.pareto) == 1
    candidate = bounded.candidate
    assert candidate is not None
    assert legacy.reason == ""
    assert dict(candidate.quadrants) == dict(legacy.quadrants)
    assert dict(candidate.translations) == dict(legacy.translations)
    assert dict(candidate.resultants) == dict(legacy.axis_resultants)
    assert dict(candidate.residuals) == dict(legacy.axis_residuals)
    assert candidate.b_map == dict(legacy.blocks_per_period)
    assert candidate.cost == legacy.alternatives[0].cost
    assert candidate.axis_observations == legacy.alternatives[0].axis_observations
    assert candidate.scale_known == legacy.scale_known


def test_additive_match_cell_shift_does_not_change_graph_or_selector_semantics():
    readings, matches = _two_frame_case(qrels=(1,))
    shifted_matches = [replace(match, cell_shift_rc=(-3, 4)) for match in matches]
    before_legacy = plane_graph.solve_component(readings, matches)
    after_legacy = plane_graph.solve_component(readings, shifted_matches)
    assert before_legacy == after_legacy

    before = selector.select_joint_labels(readings, matches, budget=_budget())
    after = selector.select_joint_labels(readings, shifted_matches, budget=_budget())
    _assert_bounded_parity(before, after)
    assert [_candidate_key(candidate) for candidate in before.pareto] == [
        _candidate_key(candidate) for candidate in after.pareto]


def test_no_three_axis_label_is_typed_no_candidate_and_bound_prunes():
    readings, matches = _two_frame_case(qrels=(1,), only_axes=(1,))
    result = selector.select_joint_labels(readings, matches, budget=_budget())
    oracle = selector.exhaustive_oracle(readings, matches)
    assert result.status == oracle.status == "no_candidate"
    assert result.counters.pruned_insufficient_axis == 1
    assert result.pareto == ()


def test_exact_q_plus_two_evidence_tie_is_quadrant_ambiguity():
    readings, matches = _two_frame_case(qrels=(1, 3))
    result = selector.select_joint_labels(readings, matches, budget=_budget())
    assert result.status == "ambiguous_quadrant"
    assert {dict(candidate.quadrants)["b"] for candidate in result.pareto} == {1, 3}
    assert all(candidate.pair_groups == result.pareto[0].pair_groups
               for candidate in result.pareto)
    assert result.certificate_sha256 is not None


def test_tree_per_axis_and_disconnected_components_are_not_cycle_gated():
    readings, matches = _chain_case()
    tree = selector.select_joint_labels(readings, matches, budget=_budget())
    assert tree.status == "b_unresolved"
    full_tree = next(candidate for candidate in tree.pareto if len(candidate.pair_groups) == 6)
    assert all(len(groups) == 2 for _axis, groups in full_tree.axis_pair_groups)
    assert len(full_tree.covered_readings) == 9

    readings, matches = _chain_case(
        frame_names=("a", "b", "c", "d"),
        links=(("a", "b"), ("c", "d")),
    )
    disconnected = selector.select_joint_labels(readings, matches, budget=_budget())
    assert disconnected.status == "b_unresolved"
    assert disconnected.pareto
    assert any(candidate.covered_frames == frozenset(("a", "b", "c", "d"))
               for candidate in disconnected.pareto)
    assert any(candidate.covered_frames == frozenset(("a", "b"))
               for candidate in disconnected.pareto)


def test_duplicate_rows_are_idempotent_but_conflicting_variant_is_rejected():
    readings, matches = _two_frame_case(qrels=(1,))
    baseline = selector.select_joint_labels(readings, matches, budget=_budget())
    duplicate = selector.select_joint_labels(readings, [*matches, matches[0]], budget=_budget())
    assert duplicate.status == baseline.status
    assert duplicate.certificate_sha256 == baseline.certificate_sha256
    assert duplicate.counters.duplicate_rows_deduplicated == 1

    conflict = selector.select_joint_labels(
        readings, [*matches, replace(matches[0], score=0.75)], budget=_budget())
    assert conflict.status == "invalid_input"
    assert conflict.reason == "conflicting duplicate pair/quadrant/b variant"

    malformed = selector.select_joint_labels(
        readings, [replace(matches[0], overlap_cells=None)], budget=_budget())
    assert malformed.status == "invalid_input"
    assert malformed.reason == "match has malformed identity/count fields"


def test_match_axes_must_match_the_referenced_readings_for_bounded_and_oracle():
    readings, matches = _two_frame_case(qrels=(1,))
    invalid = [replace(matches[0], left_axis=(matches[0].left_axis + 1) % 3), *matches[1:]]
    bounded = selector.select_joint_labels(readings, invalid, budget=_budget())
    oracle = selector.exhaustive_oracle(readings, invalid)
    assert bounded.status == oracle.status == "invalid_input"
    assert bounded.reason == oracle.reason == "match axes must match referenced plane readings"
    assert bounded.pareto == oracle.pareto == ()


def test_global_b_variants_remain_alternatives_and_do_not_double_count_pair_groups():
    readings, matches = _two_frame_case(qrels=(1,), vary_b_axes=(0,))
    result = selector.select_joint_labels(readings, matches, budget=_budget())
    assert result.status == "b_unresolved"
    assert len(result.pareto) == 4
    assert all(len(candidate.pair_groups) == 3 for candidate in result.pareto)
    assert all(len(candidate._active_matches) == 3 for candidate in result.pareto)
    assert {candidate.b_map[("a", 0)] for candidate in result.pareto} == {0.5, 1.0}


def test_dense_worse_false_node_tie_superset_and_tradeoff_controls(monkeypatch):
    g1 = ("a", 0, "b", 0)
    g2 = ("a", 1, "b", 1)
    g3 = ("a", 2, "b", 2)
    a = _fake_candidate("A", (g1, g2, g3))
    b = _fake_candidate("B", (g1, g2))
    c = _fake_candidate("C", (g1,))

    values = {
        ("A", frozenset((g1, g2))): 0.9,
        ("B", frozenset((g1, g2))): 0.8,
        ("B", frozenset((g1,))): 0.9,
        ("C", frozenset((g1,))): 0.8,
        ("A", frozenset((g1,))): 0.7,
    }

    def metric(candidate, shared, _prepared, _work):
        score = values[(candidate.quadrants[0][0], shared)]
        return ({0: score}, {0: 1.0 - score}, score)

    monkeypatch.setattr(selector, "_metrics_for", metric)
    prepared = object()
    work = selector._Work(None)
    relation = lambda left, right: selector._dominates(left, right, prepared, work)
    assert relation(a, b)
    assert relation(b, c)
    assert not relation(a, c)
    assert [("C", "A", "B")[index] for index in
            selector._nondominated_indices((c, a, b), relation)] == ["A"]
    assert [("A", "B", "C")[index] for index in
            selector._nondominated_indices((a, b, c), relation)] == ["A"]

    tie_a = _fake_candidate("TA", (g1, g2, g3))
    tie_b = _fake_candidate("TB", (g1, g2))
    monkeypatch.setattr(selector, "_metrics_for",
                        lambda candidate, shared, _p, _w: ({0: 0.9}, {0: 0.1}, 0.9))
    assert not selector._dominates(tie_a, tie_b, prepared, work)
    assert not selector._dominates(tie_b, tie_a, prepared, work)

    dense_wrong = _fake_candidate("D", (g1, g2, g3))
    objective = {
        "D": ({0: 0.7}, {0: 0.3}, 0.7),
        "B": ({0: 0.9}, {0: 0.1}, 0.9),
    }
    monkeypatch.setattr(selector, "_metrics_for",
                        lambda candidate, shared, _p, _w: objective[candidate.quadrants[0][0]])
    assert not selector._dominates(dense_wrong, b, prepared, work)
    assert not selector._dominates(b, dense_wrong, prepared, work)


def test_disjoint_and_tradeoff_signatures_remain_pareto_incomparable(monkeypatch):
    g1 = ("a", 0, "b", 0)
    g2 = ("a", 1, "b", 1)
    left = _fake_candidate("L", (g1,))
    right = _fake_candidate("R", (g2,))
    prepared = object()
    work = selector._Work(None)
    assert not selector._dominates(left, right, prepared, work)
    assert not selector._dominates(right, left, prepared, work)

    axis_groups = ((0, frozenset((g1,))), (1, frozenset((g2,))), (2, frozenset()))
    trade_a = replace(
        _fake_candidate("TA", (g1, g2)), axis_pair_groups=axis_groups)
    trade_b = replace(
        _fake_candidate("TB", (g1, g2)), axis_pair_groups=axis_groups)

    def metric(candidate, _shared, _prepared, _work):
        if candidate.quadrants[0][0] == "TA":
            return ({0: 0.95, 1: 0.70}, {0: 0.05, 1: 0.30}, 0.9)
        return ({0: 0.70, 1: 0.95}, {0: 0.30, 1: 0.05}, 0.9)

    monkeypatch.setattr(selector, "_metrics_for", metric)
    assert not selector._dominates(trade_a, trade_b, prepared, work)
    assert not selector._dominates(trade_b, trade_a, prepared, work)
    assert selector._nondominated_indices(
        (trade_a, trade_b),
        lambda a, b: selector._dominates(a, b, prepared, work),
    ) == (0, 1)


def test_budget_exhaustion_discards_all_partial_certificates():
    readings, matches = _two_frame_case(qrels=(1,))
    node_limited = selector.select_joint_labels(
        readings, matches, budget=_budget(max_search_nodes=1))
    node_legacy = _legacy_bounded(readings, matches, _budget(max_search_nodes=1))
    assert node_limited.status == "budget_exhausted"
    assert node_limited.reason == "search_nodes"
    assert node_limited.candidate is None and node_limited.pareto == ()
    assert node_limited.certificate_sha256 is None
    _assert_bounded_parity(node_limited, node_legacy)

    candidate_limited = selector.select_joint_labels(
        readings, matches, budget=_budget(max_candidate_certificates=0))
    candidate_legacy = _legacy_bounded(
        readings, matches, _budget(max_candidate_certificates=0))
    assert candidate_limited.status == "budget_exhausted"
    assert candidate_limited.reason == "candidate_certificates_buffered"
    assert candidate_limited.pareto == ()
    assert candidate_limited.counters.diagnostic_post_accept_buffering_aborts == 1
    _assert_bounded_parity(candidate_limited, candidate_legacy)

    row_limited = selector.select_joint_labels(
        readings, matches, budget=_budget(max_buffered_match_rows=0))
    row_legacy = _legacy_bounded(readings, matches, _budget(max_buffered_match_rows=0))
    assert row_limited.status == "budget_exhausted"
    assert row_limited.reason == "buffered_match_rows"
    assert row_limited.candidate is None and row_limited.pareto == ()
    assert row_limited.counters.diagnostic_post_accept_buffering_aborts == 1
    _assert_bounded_parity(row_limited, row_legacy)


@pytest.mark.parametrize(
    ("counter", "limit_key", "qrels"),
    (
        ("compatibility_checks", "max_compatibility_checks", (1,)),
        ("objective_rows", "max_objective_rows", (1,)),
        ("axis_solves", "max_axis_solves", (1,)),
        ("pareto_comparisons", "max_pareto_comparisons", (1, 3)),
    ),
)
def test_each_computation_budget_exhaustion_returns_no_partial_result(counter, limit_key, qrels):
    readings, matches = _two_frame_case(qrels=qrels)
    result = selector.select_joint_labels(
        readings, matches, budget=_budget(**{limit_key: 0}))
    legacy = _legacy_bounded(readings, matches, _budget(**{limit_key: 0}))
    assert result.status == "budget_exhausted"
    assert result.reason == counter
    assert result.candidate is None and result.pareto == ()
    assert result.certificate_sha256 is None
    _assert_bounded_parity(result, legacy)


def test_row_mask_scan_preserves_partial_compatibility_budget_charge():
    readings, matches = _two_frame_case(qrels=(1, 3), vary_b_axes=(0,))
    budget = _budget(max_compatibility_checks=7)
    masked = selector.select_joint_labels(readings, matches, budget=budget)
    scanned = _legacy_bounded(readings, matches, budget)
    assert masked.status == scanned.status == "budget_exhausted"
    assert masked.reason == scanned.reason == "compatibility_checks"
    assert masked.counters.compatibility_checks == scanned.counters.compatibility_checks == 7
    _assert_bounded_parity(masked, scanned)


def test_canonical_row_masks_match_reference_scans_for_partial_and_complete_labels():
    readings, matches = _two_frame_case(qrels=(1, 3), vary_b_axes=(0,))
    prepared = selector._prepare(readings, matches)
    assert not isinstance(prepared, str)
    domain_keys = tuple(key for key, _values in prepared.b_domains)
    domains = dict(prepared.b_domains)

    for q_right in range(4):
        quadrants = {"a": 0, "b": q_right}
        for values in product(*(domains[key] for key in domain_keys)):
            b_values = dict(zip(domain_keys, values))
            masked_work = selector._Work(_budget())
            scan_work = selector._Work(_budget(), optimized=False)
            masked = selector._candidate_records(prepared, quadrants, b_values, masked_work)
            scanned = selector._candidate_records_legacy(prepared, quadrants, b_values, scan_work)
            assert tuple(map(selector._match_sort_key, masked)) == tuple(
                map(selector._match_sort_key, scanned))
            assert masked_work.values["compatibility_checks"] == scan_work.values[
                "compatibility_checks"]

    partial_b_domains = [tuple([None, *domains[key]]) for key in domain_keys]
    for q_right in (None, 0, 1, 2, 3):
        quadrants = {"a": 0, "b": q_right}
        for values in product(*partial_b_domains):
            b_values = dict(zip(domain_keys, values))
            masked_work = selector._Work(_budget())
            scan_work = selector._Work(_budget(), optimized=False)
            masked_axes = selector._potential_axes(
                prepared, quadrants, b_values, masked_work)
            scanned_axes = selector._potential_axes(
                prepared, quadrants, b_values, scan_work)
            assert masked_axes == scanned_axes
            assert masked_work.values["compatibility_checks"] == scan_work.values[
                "compatibility_checks"]


def test_objective_cache_uses_full_absolute_q_and_exact_ordered_row_subset(monkeypatch):
    readings, matches = _two_frame_case(qrels=(1, 3), vary_b_axes=(0,))
    prepared = selector._prepare(readings, matches)
    assert not isinstance(prepared, str)
    row_q1 = next(match for match in prepared.matches
                  if match.left == "a" and match.right == "b"
                  and match.left_plane == 0 and match.right_plane == 2
                  and match.quadrant_of_right == 1
                  and match.left_b == match.right_b == 1.0)
    row_q3 = next(match for match in prepared.matches
                  if match.left == "a" and match.right == "b"
                  and match.left_plane == 0 and match.right_plane == 2
                  and match.quadrant_of_right == 3
                  and match.left_b == 0.5 and match.right_b == 1.0)
    calls = []

    def objective(*_args):
        value = object()
        calls.append(value)
        return value

    monkeypatch.setattr(plane_graph, "_evaluate_assignment", objective)
    work = selector._Work(_budget())
    q0 = {"a": 0, "b": 0}
    q2 = {"a": 0, "b": 2}
    first = selector._evaluate_objective(prepared, q0, (row_q1,), work)
    assert selector._evaluate_objective(prepared, q0, (row_q1,), work) is first
    half_turn = selector._evaluate_objective(prepared, q2, (row_q1,), work)
    other_shared_subset = selector._evaluate_objective(
        prepared, q0, (row_q3,), work)
    assert selector._evaluate_objective(
        prepared, q0, (row_q3,), work) is other_shared_subset
    assert len(calls) == 3
    assert first is not half_turn and first is not other_shared_subset
    assert half_turn is not other_shared_subset
    assert work.values["objective_cache_hits"] == 2
    assert work.values["objective_cache_misses"] == 3
    assert work.values["objective_cache_unique_signatures"] == 3
    assert work.values["physical_axis_cache_hits"] == 0
    assert work.values["physical_axis_cache_misses"] == 0

    second_call_work = selector._Work(_budget())
    assert selector._evaluate_objective(
        prepared, q0, (row_q1,), second_call_work) is not first
    assert second_call_work.values["objective_cache_misses"] == 1
    assert second_call_work.values["objective_cache_hits"] == 0


def test_repeated_full_signatures_hit_per_call_cache_and_match_oracle():
    readings, matches = _cache_repeat_case()
    bounded = selector.select_joint_labels(readings, matches, budget=_budget())
    legacy_bounded = _legacy_bounded(readings, matches, _budget())
    oracle = selector.exhaustive_oracle(readings, matches)
    _assert_bounded_parity(bounded, legacy_bounded)
    assert bounded.status == oracle.status
    assert bounded.certificate_sha256 == oracle.certificate_sha256
    assert [_candidate_key(candidate) for candidate in bounded.pareto] == [
        _candidate_key(candidate) for candidate in oracle.pareto]
    assert bounded.counters.objective_cache_hits > 0
    assert bounded.counters.objective_cache_misses == (
        bounded.counters.objective_cache_unique_signatures)
    assert bounded.counters.physical_axis_cache_hits > 0
    assert bounded.counters.physical_axis_cache_misses > 0
    assert bounded.counters.axis_solve_requests > bounded.counters.axis_solves
    assert bounded.counters.axis_solves == bounded.counters.physical_axis_cache_misses
    assert bounded.counters.axis_solve_requests == legacy_bounded.counters.axis_solves
    assert legacy_bounded.counters.axis_solve_requests == legacy_bounded.counters.axis_solves
    assert legacy_bounded.counters.physical_axis_cache_hits == 0
    assert legacy_bounded.counters.physical_axis_cache_misses == legacy_bounded.counters.axis_solves


def test_physical_axis_budget_allows_cache_hits_while_request_budget_abstains():
    readings, matches = _cache_repeat_case()
    budget = _budget(max_axis_solves=189)
    physical = selector.select_joint_labels(readings, matches, budget=budget)
    request_limited = selector._search_bounded(
        readings, matches, budget, min_resultant=0.90,
        integer_tolerance=0.15, axis_budget_mode="requests")
    oracle = selector.exhaustive_oracle(readings, matches)

    assert physical.status == oracle.status == "b_unresolved"
    assert physical.certificate_sha256 == oracle.certificate_sha256
    assert [_candidate_key(item) for item in physical.pareto] == [
        _candidate_key(item) for item in oracle.pareto]
    assert physical.counters.axis_solves == 189
    assert physical.counters.axis_solve_requests == 19_149
    assert physical.counters.physical_axis_cache_misses == 189
    assert physical.counters.physical_axis_cache_hits == 18_960
    assert request_limited.status == "budget_exhausted"
    assert request_limited.reason == "axis_solve_requests"
    assert request_limited.candidate is None and request_limited.pareto == ()
    assert request_limited.certificate_sha256 is None
    assert request_limited.counters.axis_solve_requests == 189
    assert request_limited.counters.axis_solves == 84


def test_axis_miss_budget_preflights_exact_axes_without_partial_work():
    readings, matches = _two_frame_case(qrels=(1,))
    result = selector.select_joint_labels(
        readings, matches, budget=_budget(max_axis_solves=2))
    assert result.status == "budget_exhausted"
    assert result.reason == "axis_solves"
    assert result.candidate is None and result.pareto == ()
    assert result.certificate_sha256 is None
    # One three-axis miss was requested, but preflight prevented all LSQ work.
    assert result.counters.axis_solve_requests == 3
    assert result.counters.axis_solves == 0
    assert result.counters.physical_axis_cache_misses == 0


@pytest.mark.parametrize("domain_size", (4, 5, 6, 7))
def test_shared_multi_pair_b_domains_four_through_seven_preserve_outputs(domain_size):
    readings, matches, shared_key = _shared_multi_pair_b_case(domain_size)
    prepared = selector._prepare(readings, matches)
    assert not isinstance(prepared, str)
    domains = dict(prepared.b_domains)
    assert len(domains[shared_key]) == domain_size
    assert prepared.reading_row_masks[shared_key].bit_count() == 2 * domain_size

    bounded = selector.select_joint_labels(readings, matches, budget=_budget())
    constrained = selector._select_joint_labels_constraint_first(
        readings, matches, budget=_budget())
    oracle = selector.exhaustive_oracle(readings, matches)
    for result in (bounded, constrained):
        assert result.status == oracle.status
        assert result.reason == oracle.reason
        assert result.certificate_sha256 == oracle.certificate_sha256
        assert [_candidate_key(item) for item in result.pareto] == [
            _candidate_key(item) for item in oracle.pareto]

    if domain_size == 7:
        shuffled_readings, shuffled_matches = list(readings), list(matches)
        random.Random(711).shuffle(shuffled_readings)
        random.Random(712).shuffle(shuffled_matches)
        repeated = selector._select_joint_labels_constraint_first(
            shuffled_readings, shuffled_matches, budget=_budget())
        assert repeated.certificate_sha256 == constrained.certificate_sha256
        assert [_candidate_key(item) for item in repeated.pareto] == [
            _candidate_key(item) for item in constrained.pareto]


def test_objective_axis_accounting_keeps_old_request_counts_without_cache():
    readings, matches = _cache_repeat_case()
    budget = _budget()
    uncached = _legacy_bounded(readings, matches, budget)
    request_mode = selector._search_bounded(
        readings, matches, budget, min_resultant=0.90,
        integer_tolerance=0.15, optimized=False, axis_budget_mode="requests")
    assert uncached.status == request_mode.status
    assert uncached.certificate_sha256 == request_mode.certificate_sha256
    assert uncached.counters.axis_solve_requests == uncached.counters.axis_solves
    assert request_mode.counters.axis_solve_requests == request_mode.counters.axis_solves
    assert uncached.counters.physical_axis_cache_hits == 0
    assert request_mode.counters.physical_axis_cache_hits == 0
    assert uncached.counters.physical_axis_cache_misses == uncached.counters.axis_solves


@pytest.mark.parametrize("variable_order", ("natural", "constraint_first"))
def test_axis_budget_accounting_is_deterministic_under_permutation(variable_order):
    readings, matches, _shared_key = _shared_multi_pair_b_case(7)
    baseline = selector._search_bounded(
        readings, matches, _budget(max_axis_solves=10_000),
        min_resultant=0.90, integer_tolerance=0.15,
        variable_order=variable_order)
    shuffled_readings, shuffled_matches = list(readings), list(matches)
    random.Random(1701).shuffle(shuffled_readings)
    random.Random(1702).shuffle(shuffled_matches)
    permuted = selector._search_bounded(
        shuffled_readings, shuffled_matches, _budget(max_axis_solves=10_000),
        min_resultant=0.90, integer_tolerance=0.15,
        variable_order=variable_order)
    assert baseline.status == permuted.status
    assert baseline.reason == permuted.reason
    assert baseline.certificate_sha256 == permuted.certificate_sha256
    assert baseline.pareto == permuted.pareto
    assert baseline.counters == permuted.counters


@pytest.mark.parametrize(
    ("resultant", "residual", "expected"),
    (
        (0.90, 0.15, True),
        (math.nextafter(0.90, math.inf), math.nextafter(0.15, -math.inf), True),
        (math.nextafter(0.90, -math.inf), math.nextafter(0.15, -math.inf), False),
        (math.nextafter(0.90, math.inf), math.nextafter(0.15, math.inf), False),
    ),
)
def test_objective_cache_preserves_threshold_near_float_gates(
    monkeypatch, resultant, residual, expected
):
    readings, matches = _two_frame_case(qrels=(1,))
    prepared = selector._prepare(readings, matches)
    assert not isinstance(prepared, str)
    quadrants = {"a": 0, "b": 1}
    b_values = {key: values[0] for key, values in prepared.b_domains}
    work = selector._Work(_budget())
    active = selector._candidate_records(prepared, quadrants, b_values, work)
    detail = {
        "resultants": {axis: resultant for axis in range(3)},
        "residuals": {axis: residual for axis in range(3)},
        "translations": {frame: (0.0, 0.0, 0.0) for frame in prepared.frames},
    }
    alternative = plane_graph.ComponentAlternative(
        quadrants=(0, 1), cost=0.0,
        resultants=(resultant, resultant, resultant),
        axis_observations=(2, 2, 2),
    )
    monkeypatch.setattr(
        plane_graph, "_evaluate_assignment",
        lambda *_args: (0.0, alternative, detail),
    )
    optimized, optimized_gate = selector._make_candidate(
        prepared, quadrants, b_values, active, 0.15, 0.90, work)
    legacy_work = selector._Work(_budget(), optimized=False)
    legacy, legacy_gate = selector._make_candidate(
        prepared, quadrants, b_values, active, 0.15, 0.90, legacy_work)
    assert (optimized is not None) is expected
    assert (legacy is not None) is expected
    assert optimized_gate == legacy_gate
    if expected:
        assert optimized_gate == selector._CompleteLabelGate.ACCEPTED
        assert _candidate_key(optimized) == _candidate_key(legacy)
    elif resultant < 0.90:
        assert optimized_gate == selector._CompleteLabelGate.MINIMUM_RESULTANT
    else:
        assert optimized_gate == selector._CompleteLabelGate.INTEGER_TOLERANCE

def test_permutation_and_repeat_are_deterministic():
    readings, matches = _two_frame_case(qrels=(1, 3), vary_b_axes=(0,))
    baseline = selector.select_joint_labels(readings, matches, budget=_budget())
    shuffled_readings = list(readings)
    shuffled_matches = list(matches)
    random.Random(73).shuffle(shuffled_readings)
    random.Random(91).shuffle(shuffled_matches)
    permuted = selector.select_joint_labels(shuffled_readings, shuffled_matches, budget=_budget())
    repeated = selector.select_joint_labels(readings, matches, budget=_budget())
    assert baseline.status == permuted.status == repeated.status
    assert baseline.certificate_sha256 == permuted.certificate_sha256 == repeated.certificate_sha256
    assert [_candidate_key(c) for c in baseline.pareto] == [
        _candidate_key(c) for c in permuted.pareto]


def test_optional_null_psr_is_retained_and_deterministic_for_bounded_and_oracle():
    readings, matches = _two_frame_case(qrels=(0,))
    matches = list(matches)
    matches[0] = replace(matches[0], psr=None)
    matches.append(matches[0])  # byte/field-identical null diagnostic duplicate

    bounded = selector.select_joint_labels(readings, matches, budget=_budget())
    oracle = selector.exhaustive_oracle(readings, matches)
    shuffled_readings = list(readings)
    shuffled_matches = list(matches)
    random.Random(2041).shuffle(shuffled_readings)
    random.Random(2042).shuffle(shuffled_matches)
    permuted = selector.select_joint_labels(
        shuffled_readings, shuffled_matches, budget=_budget())

    assert bounded.status == oracle.status == permuted.status == "selected"
    assert bounded.certificate_sha256 == oracle.certificate_sha256 == permuted.certificate_sha256
    assert bounded.counters.duplicate_rows_deduplicated == 1
    assert oracle.counters.duplicate_rows_deduplicated == 1
    assert any(match.psr is None for match in bounded.candidate._active_matches)
    assert any(match.psr is None for match in oracle.candidate._active_matches)


def test_search_matches_cartesian_oracle_on_deterministic_generated_graphs():
    rng = random.Random(20260913)
    for _case in range(12):
        qrels = tuple(sorted(rng.sample(range(4), rng.randint(1, 3))))
        vary_b = tuple(axis for axis in range(3) if rng.random() < 0.45)
        readings, matches = _two_frame_case(qrels=qrels, vary_b_axes=vary_b)
        bounded = selector.select_joint_labels(readings, matches, budget=_budget())
        legacy_bounded = _legacy_bounded(readings, matches, _budget())
        oracle = selector.exhaustive_oracle(readings, matches)
        _assert_bounded_parity(bounded, legacy_bounded)
        assert bounded.status == oracle.status
        assert bounded.certificate_sha256 == oracle.certificate_sha256
        assert [_candidate_key(c) for c in bounded.pareto] == [
            _candidate_key(c) for c in oracle.pareto]


def test_search_matches_oracle_on_systematic_small_domain_cover():
    # Cover all four singleton q relations with both a singleton b domain and
    # one binary b domain (rotated across all three axes), then exercise q+2,
    # non-q+2, multi-q, and all-axis-binary combinations.  A fixed-seed
    # generated test above adds stochastic combinations without the quadratic
    # Pareto cost of a full Cartesian sweep.
    cases = []
    for q in range(4):
        cases.append(((q,), ()))
        cases.append(((q,), (q % 3,)))
    cases.extend((
        ((0, 2), ()),
        ((1, 3), (0, 1, 2)),
        ((0, 1), (0, 1, 2)),
        ((0, 1, 3), (1,)),
    ))
    assert len(cases) == 12
    for qrels, vary_b_axes in cases:
        readings, matches = _two_frame_case(qrels=qrels, vary_b_axes=vary_b_axes)
        bounded = selector.select_joint_labels(readings, matches, budget=_budget())
        legacy_bounded = _legacy_bounded(readings, matches, _budget())
        oracle = selector.exhaustive_oracle(readings, matches)
        _assert_bounded_parity(bounded, legacy_bounded)
        assert bounded.status == oracle.status
        assert bounded.certificate_sha256 == oracle.certificate_sha256
        assert [_candidate_key(c) for c in bounded.pareto] == [
            _candidate_key(c) for c in oracle.pareto]


def test_three_frame_multi_b_search_matches_cartesian_oracle():
    names = ("a", "b", "c")
    readings = [
        _reading(frame, axis, b_values=(1.0, 0.5) if axis == 0 else (1.0,))
        for frame in names for axis in range(3)
    ]
    matches = []
    for left, right in (("a", "b"), ("b", "c")):
        for axis in range(3):
            left_values = (1.0, 0.5) if axis == 0 else (1.0,)
            right_values = (1.0, 0.5) if axis == 0 else (1.0,)
            for left_b in left_values:
                for right_b in right_values:
                    matches.append(_match(
                        left, right, axis, axis, 0, axis, axis,
                        left_b=left_b, right_b=right_b))

    bounded = selector.select_joint_labels(readings, matches, budget=_budget())
    legacy_bounded = _legacy_bounded(readings, matches, _budget())
    oracle = selector.exhaustive_oracle(readings, matches)
    _assert_bounded_parity(bounded, legacy_bounded)
    assert bounded.status == oracle.status
    assert bounded.certificate_sha256 == oracle.certificate_sha256
    assert [_candidate_key(c) for c in bounded.pareto] == [
        _candidate_key(c) for c in oracle.pareto]
    assert len(bounded.pareto) > 1
    assert {candidate.b_map[("b", 0)] for candidate in bounded.pareto} == {0.5, 1.0}


def test_selector_remains_unwired_from_production_reconstruction_modules():
    source_root = Path(__file__).resolve().parents[1] / "src" / "mc_photogrammetry"
    selector_path = source_root / "reconstruction" / "_joint_plane_selector.py"
    for module_path in source_root.rglob("*.py"):
        if module_path == selector_path:
            continue
        tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "_joint_plane_selector" not in (node.module or "")
            elif isinstance(node, ast.Import):
                assert all("_joint_plane_selector" not in alias.name for alias in node.names)


def _assert_constraint_output_matches_oracle(readings, matches):
    constrained = selector._select_joint_labels_constraint_first(
        readings, matches, budget=_budget())
    oracle = selector.exhaustive_oracle(readings, matches)
    assert constrained.status == oracle.status
    assert constrained.reason == oracle.reason
    assert constrained.certificate_sha256 == oracle.certificate_sha256
    assert [_candidate_key(item) for item in constrained.pareto] == [
        _candidate_key(item) for item in oracle.pareto]
    return constrained


def test_constraint_first_preserves_systematic_small_domain_oracle_outputs():
    cases = []
    for q in range(4):
        cases.append(_two_frame_case(qrels=(q,)))
        cases.append(_two_frame_case(qrels=(q,), vary_b_axes=(q % 3,)))
    cases.extend((
        _two_frame_case(qrels=(0, 2)),
        _two_frame_case(qrels=(1, 3), vary_b_axes=(0, 1, 2)),
        _two_frame_case(qrels=(0, 1), vary_b_axes=(0, 1, 2)),
        _two_frame_case(qrels=(0, 1, 3), vary_b_axes=(1,)),
    ))
    for readings, matches in cases:
        _assert_constraint_output_matches_oracle(readings, matches)


def test_constraint_first_preserves_three_frame_multi_b_tree_and_disconnected_outputs():
    names = ("a", "b", "c")
    readings = [
        _reading(frame, axis, b_values=(1.0, 0.5) if axis == 0 else (1.0,))
        for frame in names for axis in range(3)
    ]
    matches = []
    for left, right in (("a", "b"), ("b", "c")):
        for axis in range(3):
            values = (1.0, 0.5) if axis == 0 else (1.0,)
            for left_b in values:
                for right_b in values:
                    matches.append(_match(
                        left, right, axis, axis, 0, axis, axis,
                        left_b=left_b, right_b=right_b))
    _assert_constraint_output_matches_oracle(readings, matches)

    tree_readings, tree_matches = _chain_case()
    _assert_constraint_output_matches_oracle(tree_readings, tree_matches)
    disconnected_readings, disconnected_matches = _chain_case(
        frame_names=("a", "b", "c", "d"),
        links=(("a", "b"), ("c", "d")),
    )
    _assert_constraint_output_matches_oracle(disconnected_readings, disconnected_matches)


def test_constraint_first_variable_order_is_static_and_domain_constrained():
    readings, matches = _two_frame_case(qrels=(1,), vary_b_axes=(0,))
    prepared = selector._prepare(readings, matches)
    assert not isinstance(prepared, str)
    order = selector._constraint_first_variable_order(prepared)
    assert order
    # The binary reading scales precede every four-way q variable.
    first_q = next(index for index, (kind, _key) in enumerate(order) if kind == "q")
    assert all(kind == "b" for kind, _key in order[:first_q])
    assert len(order) == len(set(order))
    assert order == selector._constraint_first_variable_order(prepared)


def test_incremental_masks_preserve_reference_partial_axes_on_three_and_four_frames():
    cases = [
        _chain_case(frame_names=("a", "b", "c")),
        _chain_case(frame_names=("a", "b", "c", "d")),
    ]

    # Three frames with q and binary b evidence on both tree edges.
    names = ("a", "b", "c")
    readings = [
        _reading(frame, axis, b_values=(1.0, 0.5) if axis == 0 else (1.0,))
        for frame in names for axis in range(3)
    ]
    matches = []
    for left, right, qrel in (("a", "b", 0), ("b", "c", 2)):
        for axis in range(3):
            values = (1.0, 0.5) if axis == 0 else (1.0,)
            for left_b in values:
                for right_b in values:
                    matches.append(_match(
                        left, right, axis, axis, qrel, axis, axis,
                        left_b=left_b, right_b=right_b))
    cases.append((readings, matches))

    for case_readings, case_matches in cases:
        prepared = selector._prepare(case_readings, case_matches)
        assert not isinstance(prepared, str)
        domains = dict(prepared.b_domains)
        for q_tail in product((None, 0, 1, 2, 3), repeat=len(prepared.frames) - 1):
            quadrants = {prepared.frames[0]: 0}
            quadrants.update(zip(prepared.frames[1:], q_tail))
            value_products = product(*(
                (None, *domains[key]) for key, _values in prepared.b_domains))
            for b_tuple in value_products:
                b_values = dict(zip(domains, b_tuple))
                row_mask = selector._constraint_first_row_mask(
                    prepared, quadrants, b_values)
                masked_work = selector._Work(_budget())
                reference_work = selector._Work(_budget(), optimized=False)
                masked_axes = selector._constraint_first_potential_axes(
                    prepared, quadrants, row_mask, masked_work)
                reference_axes = selector._potential_axes(
                    prepared, quadrants, b_values, reference_work)
                assert masked_axes == reference_axes
                assert masked_work.values["compatibility_checks"] == len(prepared.matches)
                assert reference_work.values["compatibility_checks"] == len(prepared.matches)

                if (all(value is not None for value in quadrants.values())
                        and all(value is not None for value in b_values.values())):
                    active_mask = row_mask & prepared.candidate_b_endpoint_mask
                    masked_rows = tuple(
                        prepared.matches[index]
                        for index in selector._row_indices(active_mask))
                    ref_work = selector._Work(_budget(), optimized=False)
                    reference_rows = selector._candidate_records_legacy(
                        prepared, quadrants, b_values, ref_work)
                    assert tuple(map(selector._match_sort_key, masked_rows)) == tuple(
                        map(selector._match_sort_key, reference_rows))


def test_constraint_first_is_deterministic_under_input_permutation_and_relabel():
    readings, matches = _two_frame_case(qrels=(1, 3), vary_b_axes=(0,))
    baseline = _assert_constraint_output_matches_oracle(readings, matches)
    shuffled_readings, shuffled_matches = list(readings), list(matches)
    random.Random(141).shuffle(shuffled_readings)
    random.Random(142).shuffle(shuffled_matches)
    permuted = selector._select_joint_labels_constraint_first(
        shuffled_readings, shuffled_matches, budget=_budget())
    assert permuted.certificate_sha256 == baseline.certificate_sha256
    assert [_candidate_key(item) for item in permuted.pareto] == [
        _candidate_key(item) for item in baseline.pareto]

    rename = {"a": "m", "b": "z"}  # preserves canonical frame ordering
    renamed_readings = [replace(item, frame=rename[item.frame]) for item in readings]
    renamed_matches = [replace(
        item, left=rename[item.left], right=rename[item.right]) for item in matches]
    renamed = selector._select_joint_labels_constraint_first(
        renamed_readings, renamed_matches, budget=_budget())
    assert renamed.status == baseline.status
    assert len(renamed.pareto) == len(baseline.pareto)
    assert [tuple((rename[frame], value) for frame, value in item.quadrants)
            for item in baseline.pareto] == [item.quadrants for item in renamed.pareto]
    assert [tuple((rename[frame], plane, value)
                  for frame, plane, value in item.blocks_per_period)
            for item in baseline.pareto] == [item.blocks_per_period for item in renamed.pareto]


@pytest.mark.parametrize(
    ("limit_key", "expected_reason", "qrels"),
    (
        ("max_search_nodes", "search_nodes", (1,)),
        ("max_compatibility_checks", "compatibility_checks", (1,)),
        ("max_objective_rows", "objective_rows", (1,)),
        ("max_axis_solves", "axis_solves", (1,)),
        ("max_pareto_comparisons", "pareto_comparisons", (1, 3)),
        ("max_candidate_certificates", "candidate_certificates_buffered", (1,)),
        ("max_buffered_match_rows", "buffered_match_rows", (1,)),
    ),
)
def test_constraint_first_budget_exhaustion_is_deterministic_and_whole(
    limit_key, expected_reason, qrels
):
    readings, matches = _two_frame_case(qrels=qrels)
    budget = _budget(**{limit_key: 0})
    first = selector._select_joint_labels_constraint_first(
        readings, matches, budget=budget)
    second = selector._select_joint_labels_constraint_first(
        readings, matches, budget=budget)
    assert first.status == second.status == "budget_exhausted"
    assert first.reason == second.reason == expected_reason
    assert first.candidate is second.candidate is None
    assert first.pareto == second.pareto == ()
    assert first.certificate_sha256 is second.certificate_sha256 is None
    assert first.counters == second.counters


def _diagnostic_complete_label_gate_counts(counters):
    names = (
        "diagnostic_gate_no_compatible_records",
        "diagnostic_gate_objective_rejected",
        "diagnostic_gate_insufficient_axes",
        "diagnostic_gate_minimum_resultant",
        "diagnostic_gate_integer_tolerance",
        "diagnostic_gate_accepted",
        "diagnostic_gate_budget_aborted",
    )
    return {name: getattr(counters, name) for name in names}


def test_complete_label_gate_counts_are_typed_mutually_exclusive_and_ordered(monkeypatch):
    readings, matches = _two_frame_case(qrels=(1,))
    prepared = selector._prepare(readings, matches)
    assert not isinstance(prepared, str)
    q_values = {"a": 0, "b": 1}
    b_values = {key: values[0] for key, values in prepared.b_domains}
    active = tuple(prepared.matches)

    def fake_objective(resultants, residuals):
        return (
            0.1,
            plane_graph.ComponentAlternative(
                quadrants=(0, 1), cost=0.1,
                resultants=tuple(resultants.get(axis, 0.0) for axis in range(3)),
                axis_observations=(2, 2, 2)),
            {"resultants": dict(resultants), "residuals": dict(residuals),
             "translations": {"a": (0.0, 0.0, 0.0), "b": (1.0, 1.0, 1.0)}},
        )

    cases = (
        (selector._CompleteLabelGate.NO_COMPATIBLE_RECORDS, (), None),
        (selector._CompleteLabelGate.OBJECTIVE_REJECTED, active, None),
        (selector._CompleteLabelGate.INSUFFICIENT_AXES, active,
         fake_objective({0: 0.99, 1: 0.99}, {0: 0.01, 1: 0.01})),
        # Both numeric gates fail here; the existing minimum-resultant test
        # precedes the residual test and remains the single recorded outcome.
        (selector._CompleteLabelGate.MINIMUM_RESULTANT, active,
         fake_objective({0: 0.80, 1: 0.99, 2: 0.99},
                        {0: 0.20, 1: 0.01, 2: 0.01})),
        (selector._CompleteLabelGate.INTEGER_TOLERANCE, active,
         fake_objective({0: 0.99, 1: 0.99, 2: 0.99},
                        {0: 0.20, 1: 0.01, 2: 0.01})),
        (selector._CompleteLabelGate.ACCEPTED, active,
         fake_objective({0: 0.99, 1: 0.99, 2: 0.99},
                        {0: 0.01, 1: 0.01, 2: 0.01})),
    )
    for expected, active_rows, objective in cases:
        monkeypatch.setattr(
            selector, "_candidate_records", lambda *_args, rows=active_rows: rows)
        monkeypatch.setattr(plane_graph, "_evaluate_assignment", lambda *_args: objective)
        work = selector._Work(None)
        selector._accept_complete_label(
            prepared, q_values, b_values, 0.15, 0.90, work, [])
        gate_counts = _diagnostic_complete_label_gate_counts(work.snapshot())
        assert sum(gate_counts.values()) == 1
        assert gate_counts[f"diagnostic_gate_{expected.value}"] == 1
        assert all(type(value) is int for value in gate_counts.values())


def test_complete_label_gate_budget_abort_is_counted_without_partial_result():
    readings, matches = _two_frame_case(qrels=(1,))
    prepared = selector._prepare(readings, matches)
    assert not isinstance(prepared, str)
    q_values = {"a": 0, "b": 1}
    b_values = {key: values[0] for key, values in prepared.b_domains}
    work = selector._Work(_budget(max_compatibility_checks=0))
    with pytest.raises(selector._BudgetExceeded, match="compatibility_checks"):
        selector._accept_complete_label(
            prepared, q_values, b_values, 0.15, 0.90, work, [])
    gate_counts = _diagnostic_complete_label_gate_counts(work.snapshot())
    assert gate_counts["diagnostic_gate_budget_aborted"] == 1
    assert sum(gate_counts.values()) == work.values["complete_labels"] == 1


@pytest.mark.parametrize(
    ("budget_limit", "expected_reason"),
    (
        ("max_candidate_certificates", "candidate_certificates_buffered"),
        ("max_buffered_match_rows", "buffered_match_rows"),
    ),
)
def test_post_accept_buffering_budget_abort_has_separate_diagnostic(
    monkeypatch, budget_limit, expected_reason
):
    readings, matches = _two_frame_case(qrels=(1,))
    prepared = selector._prepare(readings, matches)
    assert not isinstance(prepared, str)
    quadrants = {"a": 0, "b": 1}
    b_values = {key: values[0] for key, values in prepared.b_domains}
    objective_detail = {
        "resultants": {0: 0.99, 1: 0.99, 2: 0.99},
        "residuals": {0: 0.01, 1: 0.01, 2: 0.01},
        "translations": {"a": (0.0, 0.0, 0.0), "b": (1.0, 1.0, 1.0)},
    }
    alternative = plane_graph.ComponentAlternative(
        quadrants=(0, 1), cost=0.1, resultants=(0.99, 0.99, 0.99),
        axis_observations=(2, 2, 2))
    monkeypatch.setattr(
        selector, "_evaluate_objective",
        lambda *_args: (0.1, alternative, objective_detail))
    candidates = []
    limits = {budget_limit: 0}
    if budget_limit == "max_buffered_match_rows":
        limits["max_candidate_certificates"] = 10
    work = selector._Work(_budget(**limits))
    with pytest.raises(selector._BudgetExceeded, match=expected_reason):
        selector._accept_complete_label(
            prepared, quadrants, b_values, 0.15, 0.90, work, candidates,
            active_matches=tuple(prepared.matches))

    gates = _diagnostic_complete_label_gate_counts(work.snapshot())
    assert gates["diagnostic_gate_accepted"] == 1
    assert gates["diagnostic_gate_budget_aborted"] == 0
    assert sum(gates.values()) == work.values["complete_labels"] == 1
    assert work.values["diagnostic_post_accept_buffering_aborts"] == 1
    assert candidates == []


def test_semantic_signatures_bind_exact_labels_masks_rows_and_objective_state():
    readings, matches = _cache_repeat_case()
    prepared = selector._prepare(readings, matches)
    assert not isinstance(prepared, str)
    domains = dict(prepared.b_domains)
    q_values = {frame: None for frame in prepared.frames}
    q_values[prepared.frames[0]] = 0
    b_values = {key: (values[0] if len(values) == 1 else None)
                for key, values in prepared.b_domains}
    baseline = selector._partial_semantic_signature(
        prepared, q_values, b_values, 0)
    assert baseline.feasible_row_mask == selector._semantic_feasible_row_mask(
        prepared, q_values, b_values)
    assert baseline.active_row_indices == tuple(selector._row_indices(
        baseline.feasible_row_mask & prepared.candidate_b_endpoint_mask))
    assert baseline.possible_axes == selector._semantic_possible_axes(
        prepared, q_values, baseline.feasible_row_mask)

    changed_q = dict(q_values)
    changed_q["c"] = 0
    assert selector._partial_semantic_signature(
        prepared, changed_q, b_values, 1) != baseline
    changed_b = dict(b_values)
    changed_b[("c", 0)] = 0.5
    assert selector._partial_semantic_signature(
        prepared, q_values, changed_b, 0) != baseline

    all_zero_q = {frame: 0 for frame in prepared.frames}
    complete_b = {key: values[0] for key, values in prepared.b_domains}
    repeated_rows_b = dict(complete_b)
    d2_values = domains[("d", 2)]
    assert len(d2_values) == 2
    repeated_rows_b[("d", 2)] = d2_values[-1]
    first_leaf = selector._complete_leaf_signature(
        prepared, all_zero_q, complete_b)
    second_leaf = selector._complete_leaf_signature(
        prepared, all_zero_q, repeated_rows_b)
    # q/b labels distinguish these leaves even though inactive c-d rows leave
    # the feasible mask and active ordered rows unchanged.
    assert first_leaf.feasible_row_mask == second_leaf.feasible_row_mask
    assert first_leaf.active_row_indices == second_leaf.active_row_indices
    assert first_leaf.objective_rows == second_leaf.objective_rows
    assert first_leaf.blocks_per_period_labels != second_leaf.blocks_per_period_labels
    assert first_leaf != second_leaf

    permuted_readings, permuted_matches = list(readings), list(matches)
    random.Random(1201).shuffle(permuted_readings)
    random.Random(1202).shuffle(permuted_matches)
    permuted = selector._prepare(permuted_readings, permuted_matches)
    assert not isinstance(permuted, str)
    assert selector._partial_semantic_signature(
        permuted, q_values, b_values, 0) == baseline
    assert selector._complete_leaf_signature(
        permuted, all_zero_q, complete_b) == first_leaf


def test_semantic_signature_masks_match_reference_axes_and_active_row_order():
    fixtures = (
        _two_frame_case(qrels=(1, 3), vary_b_axes=(0,)),
        _cache_repeat_case(),
        _chain_case(frame_names=("a", "b", "c")),
    )
    for readings, matches in fixtures:
        prepared = selector._prepare(readings, matches)
        assert not isinstance(prepared, str)
        singleton_b = {
            key: values[0] if len(values) == 1 else None
            for key, values in prepared.b_domains
        }
        complete_b = {key: values[0] for key, values in prepared.b_domains}
        q_states = [{frame: None for frame in prepared.frames}]
        q_states[0][prepared.frames[0]] = 0
        fully_labeled_q = {frame: index % 4 for index, frame in enumerate(prepared.frames)}
        q_states.append(fully_labeled_q)
        if len(prepared.frames) > 1:
            partial_q = dict(fully_labeled_q)
            partial_q[prepared.frames[-1]] = None
            q_states.append(partial_q)

        b_states = [singleton_b, complete_b]
        for q_values in q_states:
            for b_values in b_states:
                signature = selector._partial_semantic_signature(
                    prepared, q_values, b_values, 0)
                reference_work = selector._Work(None, optimized=False)
                reference_axes = selector._potential_axes(
                    prepared, q_values, b_values, reference_work)
                assert signature.possible_axes == tuple(sorted(reference_axes))

                # Independent row-by-row feasibility reference: q must admit
                # one prepared world-axis option and assigned b endpoints must
                # exactly agree with the canonical row.
                expected_mask = 0
                for index, (match, (left_key, right_key)) in enumerate(
                    zip(prepared.matches, prepared.match_readings)
                ):
                    left_b, right_b = b_values.get(left_key), b_values.get(right_key)
                    if ((left_b is not None and left_b != match.left_b)
                            or (right_b is not None and right_b != match.right_b)):
                        continue
                    left_q, right_q = q_values[match.left], q_values[match.right]
                    if any(
                        (left_q is None or left_q == option_left)
                        and (right_q is None or right_q == option_right)
                        for option_left, option_right, _axis
                        in prepared.match_possible_axes[index]
                    ):
                        expected_mask |= 1 << index
                assert signature.feasible_row_mask == expected_mask
                expected_active = tuple(selector._row_indices(
                    expected_mask & prepared.candidate_b_endpoint_mask))
                assert signature.active_row_indices == expected_active

        leaf = selector._complete_leaf_signature(
            prepared, fully_labeled_q, complete_b)
        candidate_work = selector._Work(None)
        actual_active = selector._candidate_records(
            prepared, fully_labeled_q, complete_b, candidate_work)
        assert leaf.active_row_indices == tuple(
            prepared.match_index_by_identity[id(match)] for match in actual_active)
        expected_objective_rows = []
        for index in leaf.active_row_indices:
            match = prepared.matches[index]
            left = prepared.read_by_index[(match.left, match.left_plane)]
            right = prepared.read_by_index[(match.right, match.right_plane)]
            axis, left_sign = plane_graph.axis_transform(
                left.axis, fully_labeled_q[match.left])
            right_axis, right_sign = plane_graph.axis_transform(
                right.axis, fully_labeled_q[match.right])
            assert right_axis == axis
            left_face = left_sign * left.side * left.distance_for(match.left_b)
            right_face = right_sign * right.side * right.distance_for(match.right_b)
            expected_objective_rows.append((
                index, axis, float(left_face), float(right_face), float(match.score)))
        assert leaf.objective_rows == tuple(expected_objective_rows)


def test_semantic_signature_census_is_bounded_and_reports_truncation():
    readings, matches = _two_frame_case(qrels=(1,))
    prepared = selector._prepare(readings, matches)
    assert not isinstance(prepared, str)
    q_values = {"a": 0, "b": None}
    b_values = {key: values[0] for key, values in prepared.b_domains}
    signature = selector._partial_semantic_signature(
        prepared, q_values, b_values, 0)
    work = selector._Work(None, signature_limit=1)
    work.observe_partial_state(signature)
    work.observe_partial_state(signature)
    work.observe_partial_state(replace(signature, depth=1))
    counters = work.snapshot()
    assert counters.diagnostic_partial_state_observations == 3
    assert counters.diagnostic_partial_state_unique == 1
    assert counters.diagnostic_partial_state_repeats == 1
    assert counters.diagnostic_partial_state_truncated == 1

    complete_signature = selector._complete_leaf_signature(
        prepared, {"a": 0, "b": 1},
        {key: values[0] for key, values in prepared.b_domains})
    alternate_signature = replace(
        complete_signature,
        quadrant_labels=(("a", 0), ("b", 2)),
    )
    work.observe_leaf_signature(complete_signature)
    work.observe_leaf_signature(complete_signature)
    work.observe_leaf_signature(alternate_signature)
    counters = work.snapshot()
    assert counters.diagnostic_leaf_signature_observations == 3
    assert counters.diagnostic_leaf_signature_unique == 1
    assert counters.diagnostic_leaf_signature_repeats == 1
    assert counters.diagnostic_leaf_signature_truncated == 1


def test_terminal_gate_and_signature_diagnostics_do_not_change_search_output():
    readings, matches = _two_frame_case(qrels=(1,))
    bounded = selector.select_joint_labels(readings, matches, budget=_budget())
    oracle = selector.exhaustive_oracle(readings, matches)
    assert bounded.status == oracle.status
    assert bounded.reason == oracle.reason
    assert bounded.certificate_sha256 == oracle.certificate_sha256
    assert [_candidate_key(item) for item in bounded.pareto] == [
        _candidate_key(item) for item in oracle.pareto]
    assert bounded.counters.diagnostic_partial_state_observations == (
        bounded.counters.search_nodes)
    assert bounded.counters.diagnostic_partial_state_unique == (
        bounded.counters.diagnostic_partial_state_observations)
    assert bounded.counters.diagnostic_partial_state_repeats == 0
    assert bounded.counters.diagnostic_partial_state_truncated == 0
    assert bounded.counters.diagnostic_leaf_signature_observations == (
        bounded.counters.complete_labels)
    assert bounded.counters.diagnostic_leaf_signature_unique == (
        bounded.counters.diagnostic_leaf_signature_observations)
    assert bounded.counters.diagnostic_leaf_signature_repeats == 0
    assert bounded.counters.diagnostic_leaf_signature_truncated == 0
    assert sum(_diagnostic_complete_label_gate_counts(bounded.counters).values()) == (
        bounded.counters.complete_labels)


def test_objective_axis_count_prediction_matches_actual_result_axes_across_shapes():
    fixtures = [
        _two_frame_case(qrels=(1,)),
        _two_frame_case(qrels=(1,), only_axes=(0,)),
        _two_frame_case(qrels=(1,), only_axes=(0, 1)),
        _chain_case(frame_names=("a", "b", "c")),
    ]
    duplicate_readings, duplicate_matches = _two_frame_case(qrels=(1,))
    fixtures.append((duplicate_readings, duplicate_matches + [duplicate_matches[0]]))
    observed_axis_counts = set()
    rng = random.Random(3317)
    for readings, matches in fixtures:
        prepared = selector._prepare(readings, matches)
        assert not isinstance(prepared, str)
        if len(prepared.frames) == 1:
            continue
        for q_tail in product(range(4), repeat=len(prepared.frames) - 1):
            q_values = {prepared.frames[0]: 0}
            q_values.update(zip(prepared.frames[1:], q_tail))
            subsets = [
                (),
                prepared.matches,
                prepared.matches[::2],
                tuple(rng.choice(prepared.matches) for _ in range(min(3, len(prepared.matches)))),
            ]
            for active_rows in subsets:
                predicted = selector._objective_axis_count_for(
                    prepared, q_values, active_rows)
                actual_result = plane_graph._evaluate_assignment(
                    prepared.frames, q_values, active_rows, prepared.read_by_index)
                actual = selector._objective_axis_count(actual_result)
                observed_axis_counts.add(actual)
                assert predicted == actual, (
                    len(prepared.frames), q_values,
                    tuple(selector._match_sort_key(item) for item in active_rows),
                    predicted, actual)
    assert observed_axis_counts.issuperset({0, 1, 2, 3})
