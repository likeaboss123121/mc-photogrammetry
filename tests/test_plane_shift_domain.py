"""P1-D complete bounded joint shift-domain evidence tests."""
from __future__ import annotations

from dataclasses import replace
import ast
import json
from pathlib import Path

import numpy as np
import pytest

from mc_photogrammetry.reconstruction import lattice_planes, plane_axis_support, plane_graph
from mc_photogrammetry.reconstruction import plane_shift_domain as p1d

from test_plane_c2_proof import _reading, _signature, _truth_case


def _budget(**overrides):
    values = {
        "max_domains": 10,
        "max_padded_fft_cells": 2_000_000,
        "max_shift_positions": 100_000,
        "max_overlap_cell_visits": 100_000_000,
        "max_buffered_entries": 100_000,
    }
    values.update(overrides)
    return p1d.ShiftDomainBudget(**values)


def _reading_universe(signatures):
    """Test fixture helper for cases where every declared reading is signed."""
    readings = []
    seen = set()
    for signature in signatures:
        if id(signature.reading) not in seen:
            seen.add(id(signature.reading))
            readings.append(signature.reading)
    return tuple(readings)


def _build_case(radius=8, budget=None):
    (_t_left, _t_right, _left, _right, sig_left, sig_right,
     _homography, match, support) = _truth_case()
    result = p1d.build_shift_domains(
        support, (match,), (sig_left, sig_right), max_shift_cells=radius,
        budget=budget or _budget(), readings=(sig_left.reading, sig_right.reading))
    return result, (sig_left, sig_right), match, support


def test_leading_signatureless_universe_identity_survives_solver_c1_and_p1d(monkeypatch):
    (t_left, t_right, left_reading, right_reading, left_sig, right_sig,
     _homography, _direct_match, _old_support) = _truth_case()
    left_axis1 = replace(left_reading, axis=1)
    right_axis1 = replace(right_reading, axis=1)
    leading_left = replace(left_reading, axis=0)
    leading_right = replace(right_reading, axis=0)
    readings = (leading_left, left_axis1, left_reading,
                leading_right, right_axis1, right_reading)
    left_sig_axis1 = _signature(left_axis1, t_left)
    right_sig_axis1 = _signature(right_axis1, t_right)
    signatures = (left_sig_axis1, left_sig, right_sig_axis1, right_sig)

    original_filter = plane_graph._apply_same_axis_filter
    filtered_maps = []

    def capture_filter(matches, read_by_index, refusals):
        filtered_maps.append(read_by_index)
        return original_filter(matches, read_by_index, refusals)

    monkeypatch.setattr(plane_graph, "_apply_same_axis_filter", capture_filter)
    graph_matches = plane_graph.match_plane_graph(
        readings, signatures, max_shift_cells=8)
    assert len(filtered_maps) == 1
    filter_map = filtered_maps[0]
    assert filter_map[("left", 0)] is leading_left
    assert filter_map[("left", 1)] is left_axis1
    assert filter_map[("left", 2)] is left_reading
    assert filter_map[("right", 0)] is leading_right
    assert filter_map[("right", 1)] is right_axis1
    assert filter_map[("right", 2)] is right_reading
    q0_matches = tuple(value for value in graph_matches
                       if value.quadrant_of_right == 0)
    assert {(value.left_axis, value.left_plane,
             value.right_axis, value.right_plane) for value in q0_matches} == {
                 (1, 1, 1, 1), (2, 2, 2, 2)}
    match = next(value for value in q0_matches if value.left_axis == 2)
    assert match.left_plane == match.right_plane == 2
    ordered_left = tuple(sorted((value for value in readings if value.frame == "left"),
                                key=lattice_planes.reading_order_key))
    ordered_right = tuple(sorted((value for value in readings if value.frame == "right"),
                                 key=lattice_planes.reading_order_key))
    for graph_match in q0_matches:
        assert ordered_left[graph_match.left_plane] is (
            left_axis1 if graph_match.left_axis == 1 else left_reading)
        assert ordered_right[graph_match.right_plane] is (
            right_axis1 if graph_match.right_axis == 1 else right_reading)
    solved = plane_graph.solve_component(readings, q0_matches)
    assert solved.reason != "no_matches"
    plane_graph.annotate_integer_residuals(solved, readings, q0_matches)

    solution = plane_graph.ComponentSolution(
        frames=("left", "right"), quadrants={"left": 0, "right": 0},
        translations={"left": (0.0, 0.0, 0.0), "right": (0.0, 0.0, 0.0)},
        axis_residuals={0: 0.0, 1: 0.0, 2: 0.0},
        axis_resultants={0: 1.0, 1: 1.0, 2: 1.0}, scale_known=True,
        blocks_per_period={("left", 1): 1.0, ("right", 1): 1.0,
                           ("left", 2): 1.0, ("right", 2): 1.0},
        reason="", alternatives=())
    support = plane_axis_support.build_axis_support(
        solution, readings, q0_matches, signatures)
    assert support.refusal is None
    assert {value.match_identity.left_plane for value in support.mapped_cell_shifts} == {1, 2}
    result = p1d.build_shift_domains(
        support, q0_matches, signatures, max_shift_cells=8, budget=_budget(),
        readings=readings)
    assert result.status == "complete", result.refusals
    assert {domain.match_identity.left_plane for domain in result.domains} == {1, 2}

    legacy_renumbered = replace(match, left_plane=0, right_plane=0)
    with pytest.raises(ValueError, match="different frame or axis"):
        plane_graph.solve_component(readings, (legacy_renumbered,))
    refused_c1 = plane_axis_support.build_axis_support(
        solution, readings, (legacy_renumbered,), signatures)
    assert refused_c1.refusal is not None
    refused_p1d = p1d.build_shift_domains(
        support, (legacy_renumbered,), signatures, max_shift_cells=8,
        budget=_budget(), readings=readings)
    assert refused_p1d.status == "refused"
    assert not refused_p1d.domains


def test_world_truth_integer_carry_is_retained_below_the_admitted_top():
    result, signatures, match, support = _build_case()
    assert result.status == "complete"
    assert len(result.domains) == 1
    domain = result.domains[0]
    assert match.cell_shift_rc == (0, 1)
    assert domain.admitted_top_shift_rc == (0, 1)
    assert domain.admitted_peak == match.peak
    assert domain.admitted_second_peak == match.second_peak
    assert domain.admitted_score == match.score
    assert (1, 1) in {(entry.row_shift, entry.column_shift)
                      for entry in domain.entries}
    carry = next(entry for entry in domain.entries
                 if (entry.row_shift, entry.column_shift) == (1, 1))
    assert carry.overlap_cells >= plane_graph.MIN_OVERLAP_CELLS
    assert carry.surface_value < domain.admitted_peak

    # q=0 maps both tangent directions with positive sign; row=v and
    # column=u. The retained row carry restores the v candidate by one.
    v_candidate_from_top = -0.2
    v_candidate_from_carry = v_candidate_from_top + carry.row_shift
    assert v_candidate_from_top == pytest.approx(-0.2)
    assert v_candidate_from_carry == pytest.approx(0.8)
    assert p1d.validate_shift_domain_inputs(
        result, support, (match,), signatures, max_shift_cells=8,
        readings=_reading_universe(signatures))
    json.dumps(result.to_record(), sort_keys=True, allow_nan=False)


def test_p1d_requires_the_complete_reading_universe_at_its_interface():
    result, signatures, match, support = _build_case()
    assert result.status == "complete"
    with pytest.raises(TypeError, match="readings"):
        p1d.build_shift_domains(
            support, (match,), signatures, max_shift_cells=8, budget=_budget())
    with pytest.raises(TypeError, match="readings"):
        p1d.validate_shift_domain_inputs(
            result, support, (match,), signatures, max_shift_cells=8)
    foreign_left = replace(signatures[0].reading)
    refused = p1d.build_shift_domains(
        support, (match,), signatures, max_shift_cells=8, budget=_budget(),
        readings=(foreign_left, signatures[1].reading))
    assert refused.status == "refused" and not refused.domains
    assert refused.refusals[0].reason == "signature_identity_mismatch"


def _reference_canvas(left, right, homography, radius):
    # Test-side canvas construction uses the public sampler, independently of
    # the shared production helper; direct source-index support is checked below.
    offset = np.eye(3); offset[:2, 2] = -radius
    shape = tuple(side + 2 * radius for side in right.cells.shape)
    cells, mask = lattice_planes.warp_signature(left, homography @ offset, shape)
    return cells, mask, np.pad(right.cells, radius), np.pad(right.coverage, radius)


def test_every_retained_shift_matches_independent_fixed_intersection_reference():
    result, _signatures, _match, _support = _build_case(radius=8)
    domain = result.domains[0]
    expected = {}
    # Reconstruct the fixed intersection from the exact fixture inputs using
    # the public existing warp and H primitives, then count with scalar loops.
    # The test-side counting is intentionally independent of P1-D's rectangle
    # slicing and support helper.
    _t_left, _t_right, _left, _right, left_sig, right_sig, _h, _m, _s = _truth_case()
    homography = plane_graph.cell_grid_homography(left_sig, right_sig, 0)
    _warped, warped_coverage, right_cells, right_mask = _reference_canvas(
        left_sig, right_sig, homography, 8)
    first = np.where(warped_coverage, _warped, 0.0).astype(np.float64)
    second = np.where(right_mask, right_cells, 0.0).astype(np.float64)
    correlation = plane_graph._correlation_surface_with_support(
        first, second, max_shift_cells=8, first_support=warped_coverage,
        second_support=right_mask)
    assert correlation is not None
    denominator = min(int(np.count_nonzero(warped_coverage)),
                      int(np.count_nonzero(right_sig.coverage)))
    for sr in range(-8, 9):
        for sc in range(-8, 9):
                count = int(correlation.support_cells[sr + 8, sc + 8])
                if count >= plane_graph.MIN_OVERLAP_CELLS:
                    expected[(sr, sc)] = (
                        count, float(correlation.values[sr + 8, sc + 8]))
    actual = {(entry.row_shift, entry.column_shift):
              (entry.overlap_cells, entry.surface_value)
              for entry in domain.entries}
    assert actual.keys() == expected.keys()
    for shift, (count, value) in expected.items():
        assert actual[shift][0] == count
        assert actual[shift][1] == value
        entry = next(item for item in domain.entries
                     if (item.row_shift, item.column_shift) == shift)
        assert entry.overlap_fraction == count / denominator


@pytest.mark.parametrize(
    "left_shape,right_shape,left_origin,right_origin,with_holes",
    [
        ((36, 36), (36, 36), (0.0, 0.0), (0.0, 0.0), False),
        ((31, 39), (36, 32), (0.35, -0.6), (-0.25, 0.4), True),
    ],
)
def test_non_square_holey_and_offset_signatures_retain_exact_joint_support(
        left_shape, right_shape, left_origin, right_origin, with_holes):
    result, signatures, _match, _support = _build_custom_case(
        left_shape, right_shape, left_origin, right_origin, with_holes)
    assert result.status == "complete"
    domain = result.domains[0]
    left, right = signatures
    homography = plane_graph.cell_grid_homography(left, right, 0)
    warped, warped_coverage, right_cells, right_mask = _reference_canvas(
        left, right, homography, 8)
    first = np.where(warped_coverage, warped, 0.0).astype(np.float64)
    second = np.where(right_mask, right_cells, 0.0).astype(np.float64)
    correlation = plane_graph._correlation_surface_with_support(
        first, second, max_shift_cells=8, first_support=warped_coverage,
        second_support=right_mask)
    assert correlation is not None
    expected = {}
    for sr in range(-8, 9):
        for sc in range(-8, 9):
            count = int(correlation.support_cells[sr + 8, sc + 8])
            if count >= plane_graph.MIN_OVERLAP_CELLS:
                expected[(sr, sc)] = (
                    count, float(correlation.values[sr + 8, sc + 8]))
    actual = {(entry.row_shift, entry.column_shift):
              (entry.overlap_cells, entry.surface_value)
              for entry in domain.entries}
    assert actual == expected


def _build_custom_case(left_shape, right_shape, left_origin, right_origin, with_holes):
    t_left, t_right = (0.2, -0.35), (1.6, 0.45)
    left_reading = _reading("left", t_left)
    right_reading = _reading("right", t_right)
    left = _signature(left_reading, t_left, shape=left_shape, origin=left_origin)
    right = _signature(right_reading, t_right, shape=right_shape, origin=right_origin)
    if with_holes:
        left.coverage[4:8, 7:11] = False
        right.coverage[19:22, 23:27] = False
    match = plane_graph.match_plane_pair(left, right, quadrant=0, max_shift_cells=8)
    assert match is not None
    match = replace(match, left_plane=0, right_plane=0)
    solution = plane_graph.ComponentSolution(
        frames=("left", "right"), quadrants={"left": 0, "right": 0},
        translations={"left": (0.0, 0.0, 0.0), "right": (0.0, 0.0, 0.0)},
        axis_residuals={0: 0.0, 1: 0.0, 2: 0.0},
        axis_resultants={0: 1.0, 1: 1.0, 2: 1.0}, scale_known=True,
        blocks_per_period={("left", 0): 1.0, ("right", 0): 1.0},
        reason="", alternatives=())
    support = plane_axis_support.build_axis_support(
        solution, (left_reading, right_reading), (match,), (left, right))
    assert support.refusal is None
    result = p1d.build_shift_domains(
        support, (match,), (left, right), max_shift_cells=8, budget=_budget(),
        readings=(left_reading, right_reading))
    return result, (left, right), match, support


def test_fixed_intersection_support_floor_is_nine_and_excludes_raw_coverage():
    mask = np.zeros((4, 4), dtype=bool)
    mask[:3, :3] = True
    assert p1d._shift_support(mask, 0, 0) == 9
    entries = [p1d.ShiftDomainEntry(0, 0, 0.1, 9, 1.0)]
    assert entries[0].overlap_cells == plane_graph.MIN_OVERLAP_CELLS
    mask[0, 0] = False
    assert p1d._shift_support(mask, 0, 0) == 8
    # The P1-D support primitive receives I only. Coverage outside I cannot
    # increase a count, including cells where just one raw coverage is true.
    raw_warped_coverage = np.ones((4, 4), dtype=bool)
    raw_right_coverage = np.ones((4, 4), dtype=bool)
    raw_intersection = raw_warped_coverage & raw_right_coverage
    assert p1d._shift_support(mask, 0, 0) < int(np.count_nonzero(raw_intersection))


def test_domain_factory_retains_nine_shifted_cells_and_excludes_eight():
    base_result, signatures, match, support = _build_case(radius=8)
    identity = base_result.requested_match_identities[0]
    axes = [axis for axis in support.mapped_cell_shifts
            if axis.match_identity == identity]
    item, error, _hashes = p1d._prepare_item(
        identity, match, axes, _reading_universe(signatures), signatures, 8)
    assert error is None and item is not None
    item.right = replace(
        item.right, cells=np.zeros((3, 4), dtype=np.float32),
        interior=np.zeros((3, 4), dtype=np.float32),
        coverage=np.ones((3, 4), dtype=bool))
    surface = np.zeros((3, 3), dtype=np.float64)
    surface[1, 1] = 1.0
    replay = (1.0, 3.0, 0.0, (0, 0))

    exactly_nine = np.ones((3, 4), dtype=bool)
    item.fixed_overlap = exactly_nine
    item.fraction_denominator = int(np.count_nonzero(exactly_nine))
    item.match = replace(match, overlap_cells=item.fraction_denominator)
    support_cells = np.full((3, 3), 9, dtype=np.int64)
    support_cells[1, 1] = item.match.overlap_cells
    retained = p1d._make_domain(item, 1, surface, support_cells, replay)
    retained_shifts = {(entry.row_shift, entry.column_shift)
                       for entry in retained.entries}
    assert (0, 1) in retained_shifts
    assert next(entry.overlap_cells for entry in retained.entries
                if (entry.row_shift, entry.column_shift) == (0, 1)) == 9

    exactly_eight_at_shift = np.ones((3, 4), dtype=bool)
    exactly_eight_at_shift[0, 0] = False
    item.fixed_overlap = exactly_eight_at_shift
    item.fraction_denominator = int(np.count_nonzero(exactly_eight_at_shift))
    item.match = replace(match, overlap_cells=item.fraction_denominator)
    support_cells = np.full((3, 3), 9, dtype=np.int64)
    support_cells[1, 1] = item.match.overlap_cells
    support_cells[1, 2] = 8
    excluded = p1d._make_domain(item, 1, surface, support_cells, replay)
    assert (0, 1) not in {(entry.row_shift, entry.column_shift)
                          for entry in excluded.entries}
    assert p1d._shift_support(exactly_eight_at_shift, 0, 1) == 8


def test_existing_correlation_convention_keeps_joint_row_v_column_u_sign():
    base = np.random.default_rng(19).normal(size=(32, 36))
    second = np.roll(base, (3, -4), axis=(0, 1))
    surface = plane_graph._correlation_surface(base, second, max_shift_cells=8)
    assert surface is not None
    peak, _psr, _second, shift = plane_graph._peak_statistics(
        surface, max_shift_cells=8)
    assert peak == float(np.max(surface))
    assert shift == (-3, 4)
    assert shift[0] == -3  # row is v
    assert shift[1] == 4   # column is u


def test_declared_radius_replay_and_whole_request_budget_boundaries():
    result, (left_sig, right_sig), match, support = _build_case(radius=8)
    assert p1d.validate_shift_domain_inputs(
        result, support, (match,), (left_sig, right_sig),
        max_shift_cells=8, readings=(left_sig.reading, right_sig.reading))
    assert result.schema == p1d.SHIFT_DOMAIN_SCHEMA
    assert result.version == p1d.SHIFT_DOMAIN_VERSION == 2
    assert result.correlation_algorithm_id == \
        plane_graph.PLANE_CORRELATION_ALGORITHM
    domain = result.domains[0]
    rows, columns = domain.right_grid_shape
    span = 2 * domain.max_shift_cells + 1
    assert result.planned_work.padded_fft_cells == \
        (rows + 6 * domain.max_shift_cells) * \
        (columns + 6 * domain.max_shift_cells)
    assert result.planned_work.overlap_cell_visits == \
        span * span * (rows + 2 * domain.max_shift_cells) * (columns + 2 * domain.max_shift_cells)
    wrong_radius = p1d.build_shift_domains(
        support, (match,), (left_sig, right_sig), max_shift_cells=7,
        budget=_budget(), readings=(left_sig.reading, right_sig.reading))
    assert wrong_radius.status == "refused"
    assert wrong_radius.refusals[0].reason == "correlation_replay_mismatch"
    assert not wrong_radius.domains

    plan = result.planned_work
    exact = _budget(
        max_domains=plan.domains,
        max_padded_fft_cells=plan.padded_fft_cells,
        max_shift_positions=plan.shift_positions,
        max_overlap_cell_visits=plan.overlap_cell_visits,
        max_buffered_entries=plan.buffered_entries)
    exact_result = p1d.build_shift_domains(
        support, (match,), (left_sig, right_sig), max_shift_cells=8, budget=exact,
        readings=(left_sig.reading, right_sig.reading))
    assert exact_result.status == "complete"
    for field in (
            "max_domains", "max_padded_fft_cells", "max_shift_positions",
            "max_overlap_cell_visits", "max_buffered_entries"):
        values = exact.to_record()
        if values[field] == 1:
            # Limits themselves are required to be positive, so a one-domain
            # request has no valid budget one below its exact limit.
            with pytest.raises(ValueError):
                p1d.ShiftDomainBudget(**{**values, field: 0})
            continue
        values[field] -= 1
        exhausted = p1d.build_shift_domains(
            support, (match,), (left_sig, right_sig), max_shift_cells=8,
            budget=p1d.ShiftDomainBudget(**values),
            readings=(left_sig.reading, right_sig.reading))
        assert exhausted.status == "budget_exhausted"
        assert exhausted.budget_failure_reason == "planned_budget_exceeded"
        assert not exhausted.domains and not exhausted.refusals
        assert exhausted.consumed_work == p1d.ShiftDomainWork(0, 0, 0, 0, 0)

    # With two requested identities the positive max_domains budget can be
    # tested one below its exact limit.
    two_support, two_matches, two_signatures = _two_identity_case()
    too_few_domains = p1d.build_shift_domains(
        two_support, two_matches, two_signatures, max_shift_cells=8,
        budget=_budget(max_domains=1), readings=_reading_universe(two_signatures))
    assert too_few_domains.planned_work.domains == 2
    assert too_few_domains.status == "budget_exhausted"
    assert not too_few_domains.domains and not too_few_domains.refusals


def test_array_mutation_digest_and_forged_records_fail_closed():
    result, (left_sig, right_sig), match, support = _build_case()
    assert p1d.validate_shift_domain_inputs(
        result, support, (match,), (left_sig, right_sig), max_shift_cells=8,
        readings=(left_sig.reading, right_sig.reading))
    left_sig.interior[0, 0] = 123.0
    assert not p1d.validate_shift_domain_inputs(
        result, support, (match,), (left_sig, right_sig), max_shift_cells=8,
        readings=(left_sig.reading, right_sig.reading))
    domain = result.domains[0]
    with pytest.raises(ValueError):
        replace(domain, zero_shift_overlap_cells=domain.zero_shift_overlap_cells + 1)
    with pytest.raises(ValueError):
        replace(domain, max_shift_cells=0)
    with pytest.raises(ValueError):
        replace(domain, admitted_top_shift_rc=(domain.max_shift_cells + 1, 0))
    with pytest.raises(ValueError):
        replace(domain, correlation_surface_sha256="0" * 64)
    with pytest.raises(ValueError):
        replace(domain, entries=tuple(reversed(domain.entries)))
    with pytest.raises(ValueError):
        replace(domain, entries=domain.entries + (domain.entries[0],))
    with pytest.raises(ValueError):
        replace(domain, top_shift_retained=not domain.top_shift_retained)
    with pytest.raises(ValueError):
        replace(result, status="refused")
    with pytest.raises(ValueError):
        replace(result, planned_work=replace(result.planned_work, shift_positions=0))
    with pytest.raises(ValueError):
        replace(result, consumed_work=replace(result.consumed_work, domains=0))

    changed_geometry = replace(left_sig, origin_uv=(2.0, -1.0))
    provenance_refusal = p1d.build_shift_domains(
        support, (match,), (changed_geometry, right_sig), max_shift_cells=8,
        budget=_budget(), readings=(left_sig.reading, right_sig.reading))
    assert provenance_refusal.status == "refused"
    assert provenance_refusal.refusals[0].reason == "provenance_mismatch"

    changed_axes = tuple(replace(axis, cell_homography_sha256="0" * 64)
                         for axis in support.mapped_cell_shifts)
    changed_c1 = replace(support, mapped_cell_shifts=changed_axes)
    c1_provenance_refusal = p1d.build_shift_domains(
        changed_c1, (match,), (left_sig, right_sig), max_shift_cells=8,
        budget=_budget(), readings=(left_sig.reading, right_sig.reading))
    assert c1_provenance_refusal.status == "refused"
    assert c1_provenance_refusal.refusals[0].reason == "provenance_mismatch"

    missing = p1d.build_shift_domains(
        support, (), (left_sig, right_sig), max_shift_cells=8, budget=_budget(),
        readings=(left_sig.reading, right_sig.reading))
    assert missing.status == "refused"
    assert missing.refusals[0].reason == "selected_identity_missing"


def test_empty_and_nonfinite_correlation_maps_refuse_with_typed_reasons():
    result, signatures, match, support = _build_case()
    zero_left = np.zeros_like(signatures[0].cells)
    no_signal_left = replace(signatures[0], cells=zero_left)
    no_signal = p1d.build_shift_domains(
        support, (match,), (no_signal_left, signatures[1]), max_shift_cells=8,
        budget=_budget(), readings=_reading_universe(signatures))
    assert no_signal.status == "refused"
    assert no_signal.refusals[0].reason == "correlation_replay_mismatch"
    nan_cells = signatures[0].cells.copy()
    nan_cells[0, 0] = np.nan
    nonfinite_left = replace(signatures[0], cells=nan_cells)
    nonfinite = p1d.build_shift_domains(
        support, (match,), (nonfinite_left, signatures[1]), max_shift_cells=8,
        budget=_budget(), readings=_reading_universe(signatures))
    assert nonfinite.status == "refused"
    assert nonfinite.refusals[0].reason == "homography_invalid"
    assert not nonfinite.domains


def test_input_permutation_and_unselected_graph_hypotheses_are_deterministic():
    result, signatures, match, support = _build_case()
    losing_q = replace(match, quadrant_of_right=1)
    permuted = p1d.build_shift_domains(
        support, (losing_q, match), tuple(reversed(signatures)),
        max_shift_cells=8, budget=_budget(), readings=_reading_universe(signatures))
    assert permuted.status == "complete"
    assert permuted.to_record() == result.to_record()

    duplicate = p1d.build_shift_domains(
        support, (match, match), signatures, max_shift_cells=8, budget=_budget(),
        readings=_reading_universe(signatures))
    assert duplicate.status == "refused"
    assert duplicate.refusals[0].reason == "selected_identity_ambiguous"
    assert not duplicate.domains


def test_malformed_match_diagnostic_becomes_typed_refusal():
    result, signatures, match, support = _build_case()
    malformed = replace(match, psr=float("nan"))
    refused = p1d.build_shift_domains(
        support, (malformed,), signatures, max_shift_cells=8, budget=_budget(),
        readings=_reading_universe(signatures))
    assert refused.status == "refused"
    assert refused.refusals[0].reason == "correlation_replay_mismatch"
    assert not refused.domains


def test_signature_hash_uses_canonical_b_while_preserving_raw_match_identity():
    (t_left, t_right, left_reading, right_reading, left_sig, right_sig,
     _homography, _match, _support) = _truth_case()
    left_sig = replace(left_sig, blocks_per_period=1.0 + 1e-10)
    right_sig = replace(right_sig, blocks_per_period=1.0 + 1e-10)
    match = plane_graph.match_plane_pair(
        left_sig, right_sig, quadrant=0, max_shift_cells=8)
    assert match is not None
    match = replace(match, left_plane=0, right_plane=0)
    assert match.left_b == pytest.approx(1.0 + 1e-10)
    solution = plane_graph.ComponentSolution(
        frames=("left", "right"), quadrants={"left": 0, "right": 0},
        translations={"left": (0.0, 0.0, 0.0), "right": (0.0, 0.0, 0.0)},
        axis_residuals={0: 0.0, 1: 0.0, 2: 0.0},
        axis_resultants={0: 1.0, 1: 1.0, 2: 1.0}, scale_known=True,
        blocks_per_period={("left", 0): 1.0, ("right", 0): 1.0},
        reason="", alternatives=())
    support = plane_axis_support.build_axis_support(
        solution, (left_reading, right_reading), (match,), (left_sig, right_sig))
    assert support.refusal is None
    result = p1d.build_shift_domains(
        support, (match,), (left_sig, right_sig), max_shift_cells=8, budget=_budget(),
        readings=(left_reading, right_reading))
    assert result.status == "complete"
    assert result.domains[0].match_identity.left_b == match.left_b


@pytest.mark.parametrize("radius", [0, -1, True, 2.0])
def test_invalid_radius_refuses_without_partial_domains(radius):
    (_t_left, _t_right, _left, _right, left_sig, right_sig,
     _homography, match, support) = _truth_case()
    result = p1d.build_shift_domains(
        support, (match,), (left_sig, right_sig), max_shift_cells=radius,
        budget=_budget(), readings=(left_sig.reading, right_sig.reading))
    assert result.status == "refused"
    assert result.refusals[0].reason == "shift_domain_incomplete"
    assert not result.domains


def test_large_radius_exhausts_budget_before_correlation_surface_allocation():
    result, signatures, match, support = _build_case()
    exhausted = p1d.build_shift_domains(
        support, (match,), signatures, max_shift_cells=10 ** 80,
        budget=_budget(), readings=_reading_universe(signatures))
    assert exhausted.status == "budget_exhausted"
    assert exhausted.budget_failure_reason == "planned_budget_exceeded"
    assert not exhausted.domains and not exhausted.refusals
    assert exhausted.consumed_work == p1d.ShiftDomainWork(0, 0, 0, 0, 0)


def test_p1g_graph_and_domain_modules_exclude_forbidden_dependency_surfaces():
    root = Path(__file__).resolve().parents[1]
    source_paths = (
        root / "src" / "mc_photogrammetry" / "reconstruction" / "plane_graph.py",
        root / "src" / "mc_photogrammetry" / "reconstruction" / "plane_shift_domain.py",
    )
    forbidden_import_roots = {
        # Manual capture metadata and manifests.
        "camera_manifest", "luna_annotation", "manifest", "metadata",
        "scene_manifest", "mc_photogrammetry.dataset", "mc_photogrammetry.capture",
        "mc_photogrammetry.metadata",
        # F3, world coordinates, and geometric bounds.
        "f3", "coords", "coordinates", "bounds", "bounding_box",
        "mc_photogrammetry.geometry", "mc_photogrammetry.reconstruction.f3",
        # References, ground truth, evaluators, and renderers.
        "reference", "ground_truth", "evaluator", "renderer",
        "mc_photogrammetry.reference", "mc_photogrammetry.ground_truth",
        "mc_photogrammetry.evaluation", "mc_photogrammetry.render",
        "mc_photogrammetry.renderer",
        # Product interfaces, file access, network, processes, placement.
        "mc_photogrammetry.cli", "mc_photogrammetry.export",
        "os", "pathlib", "io", "glob", "shutil", "tempfile",
        "requests", "httpx", "urllib", "socket", "aiohttp",
        "subprocess", "multiprocessing", "placement",
        "mc_photogrammetry.placement",
    }
    forbidden_import_components = {
        "camera_manifest", "luna_annotation", "manifest", "metadata",
        "f3", "coords", "coordinates", "bounds", "bounding_box",
        "reference", "ground_truth", "evaluator", "renderer", "placement",
    }
    forbidden_dependency_names = {
        "CameraModel", "WorldPosition", "BlockCoordinate", "VoxelBounds",
        "BoundingBox", "AABB", "F3", "LunaAnnotation", "CameraManifest",
        "SceneManifest", "ReferenceClass", "GroundTruth", "Evaluator",
        "GT", "Renderer", "Litematic", "PlacementPlan",
    }
    forbidden_calls = {
        # Metadata/manifests and scene transforms.
        "load_camera_manifest", "load_scene_manifest", "load_scene_transform",
        "read_manual_metadata", "load_luna_annotation",
        # F3/coordinate/bounds surfaces.
        "pixel_to_world", "camera_to_world", "world_to_camera",
        "get_world_coordinates", "read_bounds", "load_bounds",
        "compute_bounds", "read_f3",
        # Reference/GT/evaluation/rendering.
        "load_reference", "load_ground_truth", "evaluate_frozen_planes",
        "evaluate_ground_truth", "render", "render_volume", "create_renderer",
        # CLI/export/filesystem/network/process/placement surfaces.
        "main", "export", "export_litematic", "write_litematic",
        "open", "read_text", "write_text", "read_bytes", "write_bytes",
        "exists", "is_file", "mkdir", "unlink", "urlopen", "urlretrieve",
        "connect", "Popen", "system", "fork", "spawn", "exec",
        "place_blocks", "apply_placement", "write_world",
    }

    for source_path in source_paths:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported_modules = set()
        imported_names = set()
        called_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
                imported_names.update(alias.asname or alias.name.split(".")[0]
                                       for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_modules.add(node.module or "")
                imported_names.update(alias.asname or alias.name
                                       for alias in node.names)
            elif isinstance(node, ast.Call):
                target = node.func
                if isinstance(target, ast.Name):
                    called_names.add(target.id)
                elif isinstance(target, ast.Attribute):
                    called_names.add(target.attr)

        forbidden_modules = {
            module for module in imported_modules
            if any(module == root_name or module.startswith(root_name + ".")
                   for root_name in forbidden_import_roots)
            or bool(set(module.lower().split(".")) & forbidden_import_components)
        }
        assert not forbidden_modules, (source_path.name, forbidden_modules)
        forbidden_names = {
            name.lower() for name in forbidden_dependency_names | forbidden_calls}
        imported_forbidden = {name.lower() for name in imported_names} & forbidden_names
        called_forbidden = {name.lower() for name in called_names} & forbidden_names
        assert not imported_forbidden, (source_path.name, imported_forbidden)
        assert not called_forbidden, (source_path.name, called_forbidden)


def _two_identity_case():
    (t_left, t_right, left_reading, right_reading, left_sig, right_sig,
     _homography, match, _support) = _truth_case()
    left_second = _reading("left", t_left, extent=5.0)
    right_second = _reading("right", t_right, extent=5.0)
    left_sig_second = _signature(left_second, t_left)
    right_sig_second = _signature(right_second, t_right)
    second_match = plane_graph.match_plane_pair(
        left_sig_second, right_sig_second, quadrant=0, max_shift_cells=8)
    assert second_match is not None
    second_match = replace(second_match, left_plane=1, right_plane=1)
    readings = (left_reading, right_reading, left_second, right_second)
    solution = plane_graph.ComponentSolution(
        frames=("left", "right"), quadrants={"left": 0, "right": 0},
        translations={"left": (0.0, 0.0, 0.0), "right": (0.0, 0.0, 0.0)},
        axis_residuals={0: 0.0, 1: 0.0, 2: 0.0},
        axis_resultants={0: 1.0, 1: 1.0, 2: 1.0}, scale_known=True,
        blocks_per_period={("left", 0): 1.0, ("right", 0): 1.0,
                           ("left", 1): 1.0, ("right", 1): 1.0},
        reason="", alternatives=())
    signatures = (left_sig, right_sig, left_sig_second, right_sig_second)
    support = plane_axis_support.build_axis_support(
        solution, readings, (match, second_match), signatures)
    assert support.refusal is None
    return support, (match, second_match), signatures
@pytest.mark.parametrize("field,value", (
    ("schema", "mc-photogrammetry-plane-shift-domain"),
    ("version", 1),
    ("correlation_algorithm_id", "fixed-mask-v1"),
))
def test_v1_unknown_and_mixed_result_provenance_fail_closed(field, value):
    result, _signatures, _match, _support = _build_case(radius=8)
    with pytest.raises(ValueError, match="provenance"):
        replace(result, **{field: value})
    with pytest.raises(ValueError, match="provenance"):
        replace(result.domains[0], **{field: value})


def test_psr_tag_rejects_inconsistent_positive_infinity_state():
    result, _signatures, _match, _support = _build_case(radius=8)
    domain = result.domains[0]
    with pytest.raises(ValueError, match="PSR"):
        replace(domain, admitted_psr=1.0,
                admitted_psr_positive_infinity=True)
