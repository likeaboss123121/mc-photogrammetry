"""Array-only control of declared grid origins, canvas placement and shift reach."""
import json
from dataclasses import asdict, replace

import pytest

import numpy as np

from mc_photogrammetry.reconstruction import lattice_planes, plane_graph


def _signature(frame, cells, coverage, origin, structural_share=0.6):
    rectification = lattice_planes.PlaneRectification(
        axis_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        focal=500.0, principal_point=(120.0, 90.0), centre=(0.0, 0.0))
    reading = lattice_planes.PlaneReading(
        frame=frame, axis=0, side=1.0,
        alternatives=(lattice_planes.DistanceAlternative(1.0, 4.0, 1.0),),
        period_pixels=60.0, phase_pixels=(0.0, 0.0), extent=1.0, samples=512,
        polygon=None, coverage_cells=int(np.count_nonzero(coverage)),
        structural_share=structural_share, confidence=0.9, rectification=rectification)
    assert reading.distance_for(1.0) == 4.0
    return lattice_planes.PlaneSignature(
        reading=reading, blocks_per_period=1.0, cells=cells,
        interior=np.zeros_like(cells), coverage=coverage, origin_uv=origin)


def _direct_support(left, right, radius):
    """Enumerate covered cell pairs in the two declared q0 coordinate grids.

    This uses no homography, warp, canvas, FFT, roll or production support helper.
    A shifted right cell has grid coordinate origin_right + index + shift;
    subtracting origin_left gives the left index (column before row).
    """
    result = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.int64)
    delta_column = int(right.origin_uv[0] - left.origin_uv[0])
    delta_row = int(right.origin_uv[1] - left.origin_uv[1])
    left_rows, left_columns = left.coverage.shape
    right_rows, right_columns = right.coverage.shape
    for row_shift in range(-radius, radius + 1):
        for column_shift in range(-radius, radius + 1):
            count = 0
            for right_row in range(right_rows):
                left_row = right_row + delta_row + row_shift
                if not 0 <= left_row < left_rows:
                    continue
                for right_column in range(right_columns):
                    left_column = right_column + delta_column + column_shift
                    if (0 <= left_column < left_columns and right.coverage[right_row, right_column]
                            and left.coverage[left_row, left_column]):
                        count += 1
            result[row_shift + radius, column_shift + radius] = count
    return result


def test_match_plane_pair_recovers_authored_nonperiodic_shift_beyond_radius32():
    # Authored cell energy/coverage only: no rendered PNG, camera/truth file or pipeline.
    seed = 20261009
    patch = np.random.default_rng(seed).permutation(np.arange(1, 401)).reshape(20, 20).astype(np.float32)
    # Pin nonperiodicity of the authored texture without consulting the matcher.
    for lag in range(1, 20):
        assert not np.array_equal(patch[:, :-lag], patch[:, lag:])
        assert not np.array_equal(patch[:-lag, :], patch[lag:, :])
    left_cells = np.zeros((91, 91), dtype=np.float32)
    left_cells[0:20, 70:90] = patch
    left_coverage = np.zeros((91, 91), dtype=bool)
    left_coverage[0:20, 70:90] = True
    left = _signature("synthetic-nonperiodic-left", left_cells, left_coverage, (-153.0, 29.0))
    right = _signature("synthetic-nonperiodic-right", patch.copy(),
                       np.ones((20, 20), dtype=bool), (-19.0, -1.0))
    # Pin the authored q0 inputs and the production thresholds this test leans on.
    assert (left.origin_uv, right.origin_uv) == ((-153.0, 29.0), (-19.0, -1.0))
    assert (left.coverage.shape, right.coverage.shape) == ((91, 91), (20, 20))
    assert (int(np.count_nonzero(left.coverage)), int(np.count_nonzero(right.coverage))) == (400, 400)
    assert plane_graph.MIN_OVERLAP_CELLS == 9
    assert lattice_planes.MIN_STRUCTURAL_SHARE == 0.15
    assert left.reading.structural_share >= lattice_planes.MIN_STRUCTURAL_SHARE
    # Independent truth: the unique maximal-overlap shift in the declared grids.
    support = _direct_support(left, right, 64)
    expected_support = int(support.max())
    peaks = np.argwhere(support == expected_support)
    assert expected_support == 400
    assert len(peaks) == 1
    expected_shift = (int(peaks[0][0]) - 64, int(peaks[0][1]) - 64)
    assert expected_shift == (30, -64)
    assert max(abs(component) for component in expected_shift) > 32
    assert max(abs(component) for component in expected_shift) <= 64
    # radius32 (the unchanged production default) reaches no shared cell at all.
    refusals32 = []
    assert plane_graph.match_plane_pair(left, right, quadrant=0, max_shift_cells=32,
                                        refusals=refusals32) is None
    assert [refusal.reason for refusal in refusals32] == ["no_signal"]
    # radius64 recovers the authored shift exactly, with no sub-cell residue.
    refusals64 = []
    match = plane_graph.match_plane_pair(left, right, quadrant=0, max_shift_cells=64,
                                         refusals=refusals64)
    assert refusals64 == []
    assert match is not None
    assert (match.left, match.right) == ("synthetic-nonperiodic-left", "synthetic-nonperiodic-right")
    assert (match.left_axis, match.right_axis, match.quadrant_of_right) == (0, 0, 0)
    assert match.cell_shift_rc == expected_shift == (30, -64)
    assert match.overlap_cells == expected_support == 400
    assert match.integer_residual is None
    assert match.psr >= 3.0
    print(json.dumps({"seed": seed, "expected_shift_rc": expected_shift,
                      "recovered_shift_rc": match.cell_shift_rc, "overlap_cells": match.overlap_cells,
                      "radius32_refusal": refusals32[0].reason, "psr": match.psr}))


def test_match_plane_pair_refuses_repeated_periodic_alias_without_arbitrary_admission():
    period, rows, right_columns, left_columns = 8, 24, 24, 72
    base = (np.arange(period, dtype=np.float32) + 1.0) * 3.0

    def wall(columns):
        return np.tile(base[np.arange(columns) % period], (rows, 1)).astype(np.float32)

    left_cells, right_cells = wall(left_columns), wall(right_columns)
    # structural_share is pinned above the gate so a refusal could only come from
    # the correlation evidence, never from the structural-share shortcut.
    left = _signature("synthetic-alias-left", left_cells,
                      np.ones((rows, left_columns), dtype=bool), (0.0, 0.0), structural_share=0.6)
    right = _signature("synthetic-alias-right", right_cells,
                       np.ones((rows, right_columns), dtype=bool), (24.0, 0.0), structural_share=0.6)
    assert (left.origin_uv, right.origin_uv) == ((0.0, 0.0), (24.0, 0.0))
    assert left.reading.structural_share >= lattice_planes.MIN_STRUCTURAL_SHARE

    # Independent alias oracle: no production helper establishes the ambiguity.
    # The right block maps to left columns 24..47 at column shift 0; the left wall
    # repeats every 8 columns over 72 covered columns, so every shift that is a
    # multiple of 8 aligns identical cell values with full support.
    def aligned_pairs(column_shift):
        return [(float(left_cells[row, column + 24 + column_shift]), float(right_cells[row, column]))
                for row in range(rows) for column in range(right_columns)
                if 0 <= column + 24 + column_shift < left_columns]

    reference = aligned_pairs(0)
    assert len(reference) == rows * right_columns == 576
    for column_shift in (-24, -16, -8, 8, 16, 24):
        assert aligned_pairs(column_shift) == reference
    support = _direct_support(left, right, 32)
    assert int(support.max()) == 576
    for column_shift in (0, 8, 16, 24, -8, -16, -24):
        assert int(support[32, column_shift + 32]) == 576

    # Ground truth: seven value-identical, equally supported shifts leave no
    # unique correspondence, so the conservative answer is refusal, not a pick.
    refusals = []
    match = plane_graph.match_plane_pair(left, right, quadrant=0, max_shift_cells=32,
                                         refusals=refusals)
    print(json.dumps({"admitted": match is not None,
                      "admitted_shift_rc": None if match is None else match.cell_shift_rc,
                      "psr": None if match is None else match.psr,
                      "refusals": [refusal.reason for refusal in refusals]}))
    assert match is None, (
        "periodic alias admitted at "
        f"{None if match is None else match.cell_shift_rc} although shifts 0, +/-8, +/-16, "
        "+/-24 are value-identical and equally supported")
    assert [refusal.reason for refusal in refusals] == ["repetition_unresolved"]


def test_observed_q0_origins_reach_shared_patch_at_radius64_but_not32():
    # Authored cell energy/coverage only: no rendered PNG, camera/truth input or pipeline.
    seed = 20261009
    patch = np.random.default_rng(seed).permutation(np.arange(1, 401)).reshape(20, 20).astype(np.float32)
    left_cells = np.zeros((91, 91), dtype=np.float32)
    left_cells[0:20, 70:90] = patch
    left_coverage = np.zeros((91, 91), dtype=bool)
    left_coverage[0:20, 70:90] = True
    left = _signature("synthetic-left", left_cells, left_coverage, (-153.0, 29.0))
    right = _signature("synthetic-right", patch.copy(), np.ones((20, 20), dtype=bool), (-19.0, -1.0))
    known_shift = (30, -64)
    actual = plane_graph.cell_grid_homography(left, right, 0)
    np.testing.assert_array_equal(actual, ((1.0, 0.0, 134.0), (0.0, 1.0, -30.0), (0.0, 0.0, 1.0)))
    summaries = []
    for radius in (32, 64):
        expected = _direct_support(left, right, radius)
        first, first_support, second, second_support = plane_graph._common_canvas_inputs(left, right, actual, radius)
        surface = plane_graph._correlation_surface_with_support(
            first, second, max_shift_cells=radius,
            first_support=first_support, second_support=second_support)
        assert surface is not None
        np.testing.assert_array_equal(surface.support_cells, expected)
        summary = {"radius": radius, "direct_max_support": int(expected.max()),
                   "production_max_support": int(surface.support_cells.max()),
                   "left_warped_covered_cells": int(np.count_nonzero(first_support)),
                   "right_covered_cells": int(np.count_nonzero(second_support))}
        if radius == 32:
            assert not np.any(expected)
            assert not np.any(first_support)
            assert summary["right_covered_cells"] == 400
            summary["peak_shift_rc"] = None
        else:
            assert expected[known_shift[0] + radius, known_shift[1] + radius] == 400
            assert np.count_nonzero(expected == 400) == 1
            assert summary["left_warped_covered_cells"] == summary["right_covered_cells"] == 400
            position = np.unravel_index(int(np.argmax(surface.values)), surface.values.shape)
            peak_shift = (int(position[0]) - radius, int(position[1]) - radius)
            assert peak_shift == known_shift
            assert surface.support_cells[position] == 400
            admissible = surface.support_cells >= plane_graph.MIN_OVERLAP_CELLS
            assert plane_graph._peak_statistics(surface.values, max_shift_cells=radius, admissible=admissible)[3] == known_shift
            summary["peak_shift_rc"] = peak_shift
        summaries.append(summary)
    print(json.dumps({"seed": seed, "left_shape": left.cells.shape, "right_shape": right.cells.shape,
                      "left_origin_uv": left.origin_uv, "right_origin_uv": right.origin_uv,
                      "actual_translation_uv": [float(actual[0, 2]), float(actual[1, 2])],
                      "known_shift_rc": known_shift, "known_support_pairs": 400,
                      "controls": summaries, "production_defaults_changed": False}, sort_keys=True))


# These small canvas controls isolate exact set/value equivalence independently
# of FFT ranking. The production matcher controls above and below keep real FFTs.
def _alias_canvas(*, winner_mask=None, candidate_mask=None, perturb=False):
    radius, rows, columns = 5, 3, 4
    values = np.arange(1, rows * columns + 1, dtype=np.float32).reshape(rows, columns)
    cells = np.zeros((rows + 2 * radius, columns + 2 * radius), np.float32)
    coverage = np.zeros(cells.shape, bool)
    right_coverage = np.zeros(cells.shape, bool)
    right_coverage[radius:radius + rows, radius:radius + columns] = True
    winner_mask = np.ones(values.shape, bool) if winner_mask is None else winner_mask
    candidate_mask = np.ones(values.shape, bool) if candidate_mask is None else candidate_mask
    for offset, mask in ((0, winner_mask), (5, candidate_mask)):
        cells[radius:radius + rows, radius + offset:radius + offset + columns] = values
        coverage[radius:radius + rows, radius + offset:radius + offset + columns] = mask
    if perturb:
        cells[radius, radius + 5] = np.nextafter(values[0, 0], np.float32(np.inf))
    admissible = np.zeros((2 * radius + 1, 2 * radius + 1), bool)
    admissible[radius, radius] = admissible[radius, radius + 5] = True
    return cells, coverage, right_coverage, admissible, (0, 0), radius


def _check_alias(case, budget=None):
    cells, coverage, right_coverage, admissible, winner, radius = case
    # Independent direct support-count oracle for the private canvas controls.
    # Production obtains this table from its existing correlation return.
    supports = np.zeros(admissible.shape, np.int64)
    right_cells = list(zip(*np.nonzero(right_coverage)))
    for row in range(admissible.shape[0]):
        for column in range(admissible.shape[1]):
            dr, dc = row - radius, column - radius
            supports[row, column] = sum(
                bool(coverage[rr + dr, cc + dc])
                for rr, cc in right_cells
                if 0 <= rr + dr < coverage.shape[0]
                and 0 <= cc + dc < coverage.shape[1])
    supports.flags.writeable = False
    return plane_graph._exact_overlap_alias_check(
        cells, coverage, right_coverage, admissible, winner,
        support_cells=supports, max_shift_cells=radius, budget=budget)


def test_exact_alias_requires_full_supported_right_identity_set_not_counts_or_intersection():
    winner = np.zeros((3, 4), bool); winner[:, :3] = True
    competitor = np.zeros((3, 4), bool); competitor[:, 1:] = True
    assert winner.sum() == competitor.sum() == 9
    assert np.count_nonzero(winner & competitor) == 6
    assert not np.array_equal(winner, competitor)
    result = _check_alias(_alias_canvas(winner_mask=winner, candidate_mask=competitor))
    assert result.status == "complete" and result.alias_shift_rc is None
    assert result.shift_positions_charged == 121


@pytest.mark.parametrize("candidate_size", (8, 10, 12))
def test_unequal_support_with_identical_intersection_is_not_an_exact_alias(candidate_size):
    winner = np.zeros((3, 4), bool); winner.flat[:9] = True
    competitor = np.zeros((3, 4), bool); competitor.flat[:candidate_size] = True
    assert np.all((winner & competitor).ravel()[:min(9, candidate_size)])
    result = _check_alias(_alias_canvas(winner_mask=winner, candidate_mask=competitor))
    assert result.status == "complete" and result.alias_shift_rc is None


def test_partial_support_exact_alias_ignores_uncovered_values_and_reaches_boundary():
    mask = np.zeros((3, 4), bool); mask.flat[:9] = True
    case = _alias_canvas(winner_mask=mask, candidate_mask=mask)
    cells = case[0]
    cells[7, 6:9] = (-120., -130., -140.)  # uncovered winner row cells
    cells[7, 11:14] = (220., 230., 240.)  # different uncovered candidate values
    # The three deliberately different positions correspond to mask.flat[9:12].
    result = _check_alias(case)
    assert result.status == "alias" and result.alias_shift_rc == (0, 5)
    assert result.alias_shift_rc[1] == case[-1]


def test_one_float32_ulp_perturbation_breaks_exact_alias_without_rounding():
    exact = _check_alias(_alias_canvas())
    perturbed = _check_alias(_alias_canvas(perturb=True))
    assert exact.status == "alias"
    assert perturbed.status == "complete" and perturbed.alias_shift_rc is None


@pytest.mark.parametrize("budget_field,reason", (
    ("max_shift_positions", "shift_positions_charged"),
    ("max_mask_visits", "mask_visits_charged"),
    ("max_value_visits", "value_visits_charged"),
    ("max_temp_bytes", "temp_peak_bytes"),
))
def test_exact_alias_budget_bounds_and_inconclusive_not_absence(budget_field, reason):
    limits = dict(max_shift_positions=100_000, max_mask_visits=64_000_000,
                  max_value_visits=64_000_000, max_temp_bytes=64 * 1024 * 1024)
    limits[budget_field] = 0
    result = _check_alias(_alias_canvas(), plane_graph._OverlapAliasBudget(**limits))
    assert result.status == "inconclusive" and result.reason == reason
    assert result.alias_shift_rc is None
    assert result.shift_positions_charged <= limits["max_shift_positions"]
    assert result.mask_visits_charged <= limits["max_mask_visits"]
    assert result.value_visits_charged <= limits["max_value_visits"]
    assert result.temp_peak_bytes <= limits["max_temp_bytes"]


def test_alias_found_before_late_exhaustion_does_not_claim_complete_alias_check():
    case = _alias_canvas()
    cells, coverage, _right, admissible, _winner, _radius = case
    cells[:] = 1; coverage[:] = True
    admissible[5, 6] = True
    # Winner costs12+3*12=48 value visits, first alias12+8*12=108.
    budget = plane_graph._OverlapAliasBudget(100_000, 64_000_000, 156, 64 * 1024 * 1024)
    result = _check_alias(case, budget)
    assert result.status == "inconclusive" and result.reason == "value_visits_charged"
    assert result.value_visits_charged == 156 and result.alias_shift_rc is None


def test_alias_guard_rejects_outside_halo_instead_of_wrapping():
    case = _alias_canvas(); case[2][0, 0] = True
    result = _check_alias(case)
    assert result.status == "inconclusive" and result.reason == "right_support_outside_halo"


@pytest.mark.parametrize("quadrant", (0, 1, 2, 3))
@pytest.mark.parametrize("right_b", (1.0, 0.5))
def test_unique_match_preserved_across_reflection_swapped_axes_and_unequal_alternatives(
        quadrant, right_b, monkeypatch):
    patch = np.random.default_rng(118).permutation(np.arange(1, 401)).reshape(20, 20).astype(np.float32)
    # Explicit renderer-free signed-axis oracle: q1 swaps tangent axes; q2
    # reflects rows; q3 swaps and reflects. No production H creates the texture.
    patterns = (patch, patch.T, patch[::-1, :], patch[::-1, :].T)
    origins = ((0., 0.), (0., 0.), (0., -19.), (-19., 0.))
    right_axes = (0, 2, 0, 2)
    left = _signature("unique-transform-left", patch, np.ones(patch.shape, bool), (0., 0.))
    right = _signature("unique-transform-right", patterns[quadrant].copy(),
                       np.ones(patch.shape, bool), origins[quadrant])
    right = replace(right, reading=replace(right.reading, axis=right_axes[quadrant],
                    alternatives=(lattice_planes.DistanceAlternative(right_b, 4., 1.),)),
                    blocks_per_period=right_b)
    calls = []; original = plane_graph._exact_overlap_alias_check
    def checked(*args, **kwargs):
        value = original(*args, **kwargs); calls.append(value); return value
    monkeypatch.setattr(plane_graph, "_exact_overlap_alias_check", checked)
    refusals = []
    match = plane_graph.match_plane_pair(left, right, quadrant=quadrant,
                                         max_shift_cells=8, refusals=refusals)
    assert match is not None and refusals == []
    assert match.cell_shift_rc == (0, 0) and match.overlap_cells == 400
    assert (match.left_b, match.right_b) == (1., right_b)
    assert len(calls) == 1 and calls[0].status == "complete"


@pytest.mark.parametrize("quadrant", (0, 1, 2, 3))
def test_periodic_alias_refuses_also_under_actual_reflection_and_swapped_axes(quadrant):
    stripe = (np.arange(8, dtype=np.float32) + 1.) * 3.
    left_cells = np.tile(stripe[np.arange(72) % 8], (24, 1)).astype(np.float32)
    patch = left_cells[:, 24:48].copy()
    patterns = (patch, patch.T, patch[::-1, :], patch[::-1, :].T)
    origins = ((24., 0.), (0., 24.), (24., -23.), (-23., 24.))
    right_axes = (0, 2, 0, 2)
    left = _signature("repeat-transform-left", left_cells, np.ones(left_cells.shape, bool), (0., 0.))
    right = _signature("repeat-transform-right", patterns[quadrant].copy(),
                       np.ones(patch.shape, bool), origins[quadrant])
    right = replace(right, reading=replace(right.reading, axis=right_axes[quadrant]))
    refusals = []
    assert plane_graph.match_plane_pair(left, right, quadrant=quadrant,
                                        max_shift_cells=32, refusals=refusals) is None
    assert [r.reason for r in refusals] == ["repetition_unresolved"]


@pytest.mark.parametrize("constant", (
    "MAX_ALIAS_SHIFT_POSITIONS", "MAX_ALIAS_MASK_VISITS",
    "MAX_ALIAS_VALUE_VISITS", "MAX_ALIAS_TEMP_BYTES",
))
def test_matcher_budget_exhaustion_is_typed_serializable_inconclusive_before_witness(
        constant, monkeypatch):
    cells = np.random.default_rng(9).permutation(np.arange(1, 401)).reshape(20, 20).astype(np.float32)
    left = _signature("budget-left", cells, np.ones(cells.shape, bool), (0., 0.))
    right = _signature("budget-right", cells.copy(), np.ones(cells.shape, bool), (0., 0.))
    monkeypatch.setattr(plane_graph, constant, 0)
    refusals, witnesses = [], []
    trace = plane_graph.PlaneGraphTrace([])
    observer = plane_graph.region_growth._ScalarAttritionObserver()
    with plane_graph.region_growth._scalar_attrition_context(observer):
        match = plane_graph.match_plane_pair(left, right, quadrant=0, max_shift_cells=8,
                    refusals=refusals, diagnostic_sink=trace, witness_sink=witnesses)
    assert match is None and witnesses == []
    assert [r.reason for r in refusals] == ["ambiguity_check_inconclusive"]
    assert refusals[0].reason in plane_graph.MATCH_REFUSAL_REASONS
    assert "exact overlap alias check inconclusive" in refusals[0].detail
    assert len(trace.pairs) == 1 and trace.pairs[0].reason == refusals[0].reason
    assert not trace.pairs[0].admitted
    stage = observer.stages["pair"]
    assert list(stage["terminal"].values()) == ["ambiguity_check_inconclusive"]
    assert sum(group["count"] for group in stage["counts"].values()) == 1
    assert stage["examples"][0]["subcause"].startswith("overlap_alias_")
    assert not observer.incomplete
    json.dumps(asdict(refusals[0]), allow_nan=False)


def test_complete_no_alias_check_preserves_original_match_and_trace_scalars(monkeypatch):
    cells = np.random.default_rng(17).permutation(np.arange(1, 401)).reshape(20, 20).astype(np.float32)
    left = _signature("parity-left", cells, np.ones(cells.shape, bool), (0., 0.))
    right = _signature("parity-right", cells.copy(), np.ones(cells.shape, bool), (0., 0.))
    before = plane_graph.PlaneGraphTrace([])
    with monkeypatch.context() as local:
        local.setattr(plane_graph, "_exact_overlap_alias_check", lambda *args, **kwargs:
                      plane_graph._OverlapAliasCheck("complete", "", None, 0, 0, 0, 0))
        old_gate_result = plane_graph.match_plane_pair(left, right, quadrant=0,
                            max_shift_cells=8, diagnostic_sink=before)
    after = plane_graph.PlaneGraphTrace([])
    guarded = plane_graph.match_plane_pair(left, right, quadrant=0,
                            max_shift_cells=8, diagnostic_sink=after)
    assert guarded is not None and guarded == old_gate_result
    assert after == before


@pytest.mark.parametrize("candidate_size", (8, 10, 12))
def test_different_source_support_count_skips_mask_construction(candidate_size):
    winner = np.zeros((3, 4), bool); winner.flat[:9] = True
    competitor = np.zeros((3, 4), bool); competitor.flat[:candidate_size] = True
    case = _alias_canvas(winner_mask=winner, candidate_mask=competitor)
    # Covers initial scans/coordinates/winner exactly, with no remaining budget
    # for even one competitor mask; source count inequality safely avoids it.
    mask_limit = 2 * case[2].size + 7 * int(case[2].sum())
    budget = plane_graph._OverlapAliasBudget(100_000, mask_limit, 64_000_000, 64 * 1024 * 1024)
    result = _check_alias(case, budget)
    assert result.status == "complete" and result.alias_shift_rc is None
    assert result.mask_visits_charged == mask_limit


def test_large_normal_unique_match_finishes_using_original_support_count_prefilter(monkeypatch):
    side, radius = 90, 32
    cells = np.random.default_rng(104).permutation(np.arange(1, side * side + 1)).reshape(side, side).astype(np.float32)
    coverage = np.ones(cells.shape, bool)
    left = _signature("large-unique-left", cells, coverage, (0., 0.))
    right = _signature("large-unique-right", cells.copy(), coverage.copy(), (0., 0.))
    original = plane_graph._exact_overlap_alias_check
    observed = []
    def checked(*args, **kwargs):
        source_counts = kwargs["support_cells"]
        assert source_counts.dtype == np.dtype(np.int64) and not source_counts.flags.writeable
        assert source_counts.shape == (65, 65)
        assert source_counts[32, 32] == side * side
        assert np.count_nonzero(source_counts == side * side) == 1
        # Old all-admissible mask gathering would already exceed its64M budget.
        assert 4 * side * side * (int(args[3].sum()) - 1) > plane_graph.MAX_ALIAS_MASK_VISITS
        result = original(*args, **kwargs); observed.append(result); return result
    monkeypatch.setattr(plane_graph, "_exact_overlap_alias_check", checked)
    refusals = []
    match = plane_graph.match_plane_pair(left, right, quadrant=0, refusals=refusals)
    assert match is not None and match.cell_shift_rc == (0, 0)
    assert match.overlap_cells == 8100 and refusals == []
    assert len(observed) == 1 and observed[0].status == "complete"
    assert observed[0].shift_positions_charged == 4225
    assert observed[0].mask_visits_charged == 2 * (side + 2 * radius) ** 2 + 7 * side * side
    assert observed[0].value_visits_charged == 4 * side * side
    assert observed[0].temp_peak_bytes == 64 * side * side + 4096
