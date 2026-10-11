from dataclasses import replace

import numpy as np
import pytest

from scripts.probe_image_derived_shift_overlap import build_fixture
from mc_photogrammetry.reconstruction import lattice, lattice_planes, plane_graph


@pytest.mark.parametrize("shift", (-32, -20, 0, 20, 32))
def test_selected_pairs_conserve_support_and_do_not_change_match(shift):
    left, template, _known = build_fixture("fixed_whitening")
    coverage = np.zeros(left.cells.shape, dtype=bool)
    coverage[8:72, 6:45] = True
    right = replace(template, cells=np.roll(left.cells, -shift, axis=1),
                    coverage=np.roll(coverage, -shift, axis=1))
    left = replace(left, coverage=coverage)
    plain = plane_graph.match_plane_pair(left, right, quadrant=0,
                                         max_shift_cells=32)
    sink = []
    observed = plane_graph.match_plane_pair(
        left, right, quadrant=0, max_shift_cells=32, witness_sink=sink)
    assert observed == plain and observed is not None
    assert len(sink) == 1
    witness = sink[0]
    assert witness.selected_shift_rc == observed.cell_shift_rc
    assert len(witness.logical_pairs) == witness.selected_support == observed.overlap_cells
    rows = tuple(map(tuple, witness.logical_pairs.tolist()))
    assert rows == tuple(sorted(rows, key=lambda row: (row[2], row[3])))


def test_reverse_pair_witness_is_conservative_and_deterministic():
    left, right, _known = build_fixture("zero_overlap")
    forward, reverse = [], []
    first = plane_graph.match_plane_pair(left, right, quadrant=0,
                                         max_shift_cells=32, witness_sink=forward)
    second = plane_graph.match_plane_pair(right, left, quadrant=0,
                                          max_shift_cells=32, witness_sink=reverse)
    assert first is not None and second is not None
    assert first.overlap_cells == len(forward[0].logical_pairs)
    assert second.overlap_cells == len(reverse[0].logical_pairs)
    again = []
    plane_graph.match_plane_pair(left, right, quadrant=0,
                                 max_shift_cells=32, witness_sink=again)
    assert again[0].witness_sha256 == forward[0].witness_sha256


def test_warp_witness_records_rint_clipping_ties_and_all_quadrant_homographies():
    left, right, _known = build_fixture("zero_overlap")
    tie = np.eye(3); tie[0, 2] = 0.5
    sink = []
    plain = lattice_planes.warp_signature(left, tie, right.cells.shape)
    observed = lattice_planes.warp_signature(
        left, tie, right.cells.shape, witness_sink=sink)
    np.testing.assert_array_equal(observed[0], plain[0])
    np.testing.assert_array_equal(observed[1], plain[1])
    assert sink[0].source_columns[0, 0] == 0  # np.rint(0.5) is ties-to-even
    for quadrant in range(4):
        mapped_axis, _sign = plane_graph.axis_transform(left.reading.axis, quadrant)
        compatible = replace(right, reading=replace(right.reading, axis=mapped_axis))
        homography = plane_graph.cell_grid_homography(left, compatible, quadrant)
        rows = []
        lattice_planes.warp_signature(
            left, homography, compatible.cells.shape, witness_sink=rows)
        assert len(rows) == 1 and rows[0].homography.shape == (3, 3)
    repeated = np.eye(3); repeated[0, 0] = repeated[1, 1] = 0.4
    repeated_rows = []
    lattice_planes.warp_signature(
        left, repeated, (8, 8), witness_sink=repeated_rows)
    witness = repeated_rows[0]
    mapped = np.column_stack((witness.source_rows.ravel(),
                              witness.source_columns.ravel()))
    assert len(np.unique(mapped, axis=0)) < len(mapped)


def test_all_quadrants_and_unequal_grid_origins_conserve_exact_pairs():
    left, template, _known = build_fixture("zero_overlap")
    for quadrant in range(4):
        mapped_axis, _sign = plane_graph.axis_transform(left.reading.axis, quadrant)
        reading = replace(template.reading, axis=mapped_axis)
        seed = replace(template, reading=reading)
        rotation = plane_graph.cell_grid_homography(left, seed, quadrant)[:2, :2]
        desired = np.asarray([
            seed.cells.shape[1] - 1 if rotation[0].min() < 0 else 0,
            seed.cells.shape[0] - 1 if rotation[1].min() < 0 else 0], dtype=float)
        left_origin = np.asarray(left.origin_uv)
        right_origin = rotation.T @ (desired + left_origin)
        seed = replace(seed, origin_uv=tuple(map(float, right_origin)))
        homography = plane_graph.cell_grid_homography(left, seed, quadrant)
        cells, coverage = lattice_planes.warp_signature(
            left, homography, seed.cells.shape)
        right = replace(seed, cells=cells, coverage=coverage)
        sink = []
        match = plane_graph.match_plane_pair(
            left, right, quadrant=quadrant, max_shift_cells=32,
            witness_sink=sink)
        assert match is not None and len(sink) == 1
        assert len(sink[0].logical_pairs) == match.overlap_cells

    # A larger left grid and shifted origin still maps into the exact right-grid
    # canvas without indexing the source by right-grid dimensions.
    padded_cells = np.pad(left.cells, ((2, 3), (4, 1)))
    padded_coverage = np.pad(left.coverage, ((2, 3), (4, 1)))
    larger = replace(left, cells=padded_cells, interior=np.zeros_like(padded_cells),
                     coverage=padded_coverage,
                     origin_uv=(left.origin_uv[0] - 4, left.origin_uv[1] - 2))
    sink = []
    match = plane_graph.match_plane_pair(
        larger, template, quadrant=0, max_shift_cells=32, witness_sink=sink)
    assert match is not None
    assert len(sink[0].logical_pairs) == match.overlap_cells
    assert np.all(sink[0].logical_pairs[:, 0] < larger.cells.shape[0])
    assert np.all(sink[0].logical_pairs[:, 1] < larger.cells.shape[1])


def test_pair_witness_tamper_and_budget_refusal_are_atomic():
    left, right, _known = build_fixture("zero_overlap")
    sink = []
    match = plane_graph.match_plane_pair(left, right, quadrant=0,
                                         max_shift_cells=32, witness_sink=sink)
    assert match is not None
    witness = sink[0]
    pairs = witness.logical_pairs.copy(); pairs[0, 0] += 1; pairs.flags.writeable = False
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(witness, logical_pairs=pairs)
    refused = []
    observed = plane_graph.match_plane_pair(
        left, right, quadrant=0, max_shift_cells=32, witness_sink=refused,
        witness_budget=lattice.WitnessBudgetV1(max_pair_entries=0))
    assert observed == match
    assert refused == [lattice.WitnessRefusalV1(
        "refused", "budget_exhausted", "warp_entries", 0,
        (right.cells.shape[0] + 64) * (right.cells.shape[1] + 64), 0)]
