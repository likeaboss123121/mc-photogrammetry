from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scripts.probe_image_derived_shift_overlap import build_fixture
from mc_photogrammetry.reconstruction import lattice_planes, plane_graph, plane_residue


@pytest.mark.parametrize("column_shift", (-32, -24, -20, 20, 24, 32))
def test_support_aware_halo_finds_border_and_interior_shifts(column_shift):
    left, _right, _known = build_fixture("fixed_whitening")
    cells = np.array(left.cells, copy=True)
    # Both positive wrapped cases must actually observe nonperiodic evidence.
    # At +24 the true left overlap is columns24..39, at +32 it is32..39.
    # The generator's old lone cue at column13 is absent from both.
    cells[17, 33] += np.float32(197.)
    cue = cells[17, 33]
    assert np.count_nonzero(cells == cue) == 1
    coverage = np.zeros(cells.shape, dtype=bool)
    coverage[10:70, 8:40] = True
    right_cells = np.roll(cells, -column_shift, axis=1)
    right_coverage = np.roll(coverage, -column_shift, axis=1)
    left = replace(left, cells=cells, coverage=coverage)
    right = replace(_right, cells=right_cells, coverage=right_coverage)
    def direct_pairs(row_shift, col_shift):
        # Independent linear-grid sampler: no H, canvas, warp, roll or FFT.
        pairs = {}
        for rr in range(right_cells.shape[0]):
            for rc in range(right_cells.shape[1]):
                lr, lc = rr + row_shift, rc + col_shift
                if (right_coverage[rr, rc] and 0 <= lr < cells.shape[0]
                        and 0 <= lc < cells.shape[1] and coverage[lr, lc]):
                    pairs[rr, rc] = (float(cells[lr, lc]), float(right_cells[rr, rc]))
        return pairs
    expected = direct_pairs(0, column_shift)
    assert len(expected) == 60 * (32 if column_shift < 0 else 40 - column_shift)
    assert all(first == second for first, second in expected.values())
    cue_right = (17, (33 - column_shift) % 80)
    assert expected[cue_right] == (float(cue), float(cue))
    assert np.count_nonzero(right_cells == cue) == 1
    # A distinct displacement must move this uniquely valued, supported right
    # cell away from the unique left cue. Thus no same-support exact alias exists.
    if column_shift in (24, 32):
        aliases_without_cue = (15,) if column_shift == 24 else (14, 23)
        for candidate in aliases_without_cue:
            alternative = direct_pairs(0, candidate)
            assert set(alternative) == set(expected)
            assert alternative[cue_right][0] != expected[cue_right][0]
    match = plane_graph.match_plane_pair(
        left, right, quadrant=0, max_shift_cells=32)
    assert match is not None
    assert match.cell_shift_rc == (0, column_shift)
    correlation = plane_graph._correlation_surface_with_support(
        left.cells, right.cells, max_shift_cells=32,
        first_support=left.coverage, second_support=right.coverage)
    assert correlation is not None
    assert match.overlap_cells == len(expected) == correlation.support_cells[32, column_shift + 32]


def test_support_surface_is_complete_read_only_and_versioned():
    left, right, known = build_fixture("zero_overlap")
    result = plane_graph._correlation_surface_with_support(
        left.cells, right.cells, max_shift_cells=32,
        first_support=left.coverage, second_support=right.coverage)
    assert result is not None
    assert result.algorithm == plane_graph.PLANE_CORRELATION_ALGORITHM
    assert result.support_cells.shape == (65, 65)
    assert result.support_cells[known[0] + 32, known[1] + 32] == 1200
    assert not result.values.flags.writeable
    assert not result.support_cells.flags.writeable


def test_no_support_admissible_shift_is_typed_refusal():
    left, right, _known = build_fixture("zero_overlap")
    left = replace(left, coverage=np.eye(80, dtype=bool))
    right = replace(right, coverage=np.fliplr(np.eye(80, dtype=bool)))
    refusals = []
    assert plane_graph.match_plane_pair(
        left, right, quadrant=0, max_shift_cells=32, refusals=refusals) is None
    assert refusals[0].reason == "insufficient_overlap"


def test_infinite_psr_requires_a_tagged_finite_schema():
    left, right, _known = build_fixture("zero_overlap")
    match = plane_graph.match_plane_pair(left, right, quadrant=0, max_shift_cells=32)
    assert match is not None and match.psr == float("inf")
    indexed = {("left", 0): left.reading, ("right", 0): right.reading}
    evidence = plane_residue.hypothesis_phase_shift_evidence(
        replace(match, left_plane=0, right_plane=0), indexed)
    assert evidence.relative_shift.refusal is None
    assert evidence.relative_shift.to_record()["psr"] == {
        "value": None, "positive_infinity": True}


def _small_pair(side=4, origin=(8., 0.)):
    left, right, _known = build_fixture("fixed_whitening")
    cells = np.random.default_rng(431).uniform(1, 30, (side, side)).astype(np.float32)
    fields = dict(cells=cells, coverage=np.ones(cells.shape, bool),
                  interior=np.zeros_like(cells))
    return (replace(left, **fields, origin_uv=(0., 0.)),
            replace(right, **fields, origin_uv=origin))


def _source_pairs(left, right, homography, shift):
    """Independent direct sampler: no canvas, warp helper, roll, or FFT."""
    pairs = []
    for rr in range(right.cells.shape[0]):
        for rc in range(right.cells.shape[1]):
            if not right.coverage[rr, rc]:
                continue
            u, v, w = homography @ np.array([rc + shift[1], rr + shift[0], 1.])
            u, v = u / w, v / w
            rows, columns = left.cells.shape
            if not (-.5 <= u <= columns - .5 and -.5 <= v <= rows - .5):
                continue
            lr = min(rows - 1, max(0, int(np.rint(v))))
            lc = min(columns - 1, max(0, int(np.rint(u))))
            if left.coverage[lr, lc]:
                pairs.append((lr, lc, rr, rc))
    return pairs


def test_four_by_four_recovers_support_but_preserves_undefined_psr_refusal():
    left, right = _small_pair()
    h = plane_graph.cell_grid_homography(left, right, 0)
    assert not lattice_planes.warp_signature(left, h, right.cells.shape)[1].any()
    first, fm, second, sm = plane_graph._common_canvas_inputs(left, right, h, 16)
    surface = plane_graph._correlation_surface_with_support(
        first, second, max_shift_cells=16, first_support=fm, second_support=sm)
    position = np.unravel_index(np.argmax(surface.values), surface.values.shape)
    assert tuple(x - 16 for x in position) == (0, -8)
    assert surface.support_cells[position] == 16
    assert len(_source_pairs(left, right, h, (0, -8))) == 16
    refusals = []
    assert plane_graph.match_plane_pair(left, right, quadrant=0, max_shift_cells=16,
                                       refusals=refusals) is None
    assert refusals[0].reason == "repetition_unresolved"
    assert "sidelobe" in refusals[0].detail


def test_larger_asymmetric_origin_displacement_admits_exact_real_pairs():
    left, right = _small_pair(side=5)
    witness, trace = [], plane_graph.PlaneGraphTrace([])
    match = plane_graph.match_plane_pair(
        left, right, quadrant=0, max_shift_cells=16, witness_sink=witness,
        diagnostic_sink=trace)
    assert match is not None and match.cell_shift_rc == (0, -8)
    assert match.overlap_cells == 25 and trace.pairs[0].valid_cells == 25
    assert trace.pairs[0].overlap_fraction == 1.
    h = plane_graph.cell_grid_homography(left, right, 0)
    assert witness[0].logical_pairs.tolist() == [list(p) for p in _source_pairs(left, right, h, (0, -8))]
    assert witness[0].right_grid_shape == (5, 5)
    assert witness[0].warp_witness.target_shape == (37, 37)
    assert witness[0].canvas_shape == (101, 101)
    assert np.all(witness[0].logical_pairs < 5)
    forged = witness[0].logical_pairs.copy()
    forged[0, 2] = 5  # valid padded index, invalid original right index
    forged.flags.writeable = False
    with pytest.raises(ValueError, match="outside its grid"):
        replace(witness[0], logical_pairs=forged, witness_sha256="")
    with pytest.raises(ValueError, match="outside its grid"):
        replace(witness[0], logical_pairs=forged, warp_witness=None, witness_sha256="")
    changed_h = witness[0].homography.copy(); changed_h[0, 2] += 1
    changed_h.flags.writeable = False
    with pytest.raises(ValueError, match="geometry differs"):
        replace(witness[0], homography=changed_h, witness_sha256="")


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("quadrant", range(4))
@pytest.mark.parametrize("shift", ((-4, 4), (4, -4), (0, 0), (-10, 10), (10, -10)))
def test_common_canvas_support_matches_direct_enumerator_all_signed_axes(axis, quadrant, shift):
    left, right = _small_pair(side=5, origin=(0., 0.))
    left = replace(left, reading=replace(left.reading, axis=axis),
                   cells=np.ones((9, 11), np.float32),
                   coverage=np.ones((9, 11), bool), interior=np.zeros((9, 11), np.float32))
    right_axis, _ = plane_graph.axis_transform(axis, quadrant)
    right = replace(right, reading=replace(right.reading, axis=right_axis),
                    cells=np.ones((4, 5), np.float32), coverage=np.ones((4, 5), bool),
                    interior=np.zeros((4, 5), np.float32))
    rotation = plane_graph.cell_grid_homography(left, right, quadrant)[:2, :2]
    anchor = np.array([2 if rotation[0].min() >= 0 else 8,
                       2 if rotation[1].min() >= 0 else 6], float)
    translation = anchor - rotation @ np.array([shift[1], shift[0]])
    right = replace(right, origin_uv=tuple(map(float, rotation.T @ translation)))
    h = plane_graph.cell_grid_homography(left, right, quadrant)
    known = _source_pairs(left, right, h, shift)
    assert len(known) == 20
    coverage = np.zeros(left.cells.shape, bool)
    cells = np.zeros(left.cells.shape, np.float32)
    values = np.random.default_rng(19).uniform(1, 30, right.cells.shape).astype(np.float32)
    for lr, lc, rr, rc in known:
        coverage[lr, lc] = True
        cells[lr, lc] = values[rr, rc]
    left = replace(left, cells=cells, coverage=coverage)
    right = replace(right, cells=values)
    first, fm, second, sm = plane_graph._common_canvas_inputs(left, right, h, 4)
    surface = plane_graph._correlation_surface_with_support(
        first, second, max_shift_cells=4, first_support=fm, second_support=sm)
    if surface is None:  # out-of-window source patch has no reachable cells
        assert not fm.any()
    for sr in range(-4, 5):
        for sc in range(-4, 5):
            expected = len(_source_pairs(left, right, h, (sr, sc)))
            actual = 0 if surface is None else int(surface.support_cells[sr + 4, sc + 4])
            assert actual == expected
    refusals, witnesses = [], []
    match = plane_graph.match_plane_pair(left, right, quadrant=quadrant, max_shift_cells=4,
                                       refusals=refusals, witness_sink=witnesses)
    if max(abs(s) for s in shift) <= 4:
        assert match is not None
        assert match.cell_shift_rc == shift and match.overlap_cells == 20
        assert witnesses[0].logical_pairs.tolist() == [list(p) for p in known]
    else:
        assert match is None
        assert refusals[0].reason == "no_signal"


@pytest.mark.parametrize("kind", ("sparse", "constant", "repetitive", "outside"))
def test_canvas_retains_signal_support_and_repetition_refusals(kind):
    left, right = _small_pair(side=5)
    if kind == "sparse":
        mask = np.zeros((5, 5), bool); mask[:2, :4] = True
        left, right = replace(left, coverage=mask), replace(right, coverage=mask)
    elif kind == "constant":
        left, right = replace(left, cells=np.ones((5, 5), np.float32)), replace(right, cells=np.ones((5, 5), np.float32))
    elif kind == "repetitive":
        left = replace(left, reading=replace(left.reading, structural_share=0.))
        right = replace(right, reading=replace(right.reading, structural_share=0.))
    else:
        right = replace(right, origin_uv=(30., 0.))
    refusals = []
    assert plane_graph.match_plane_pair(left, right, quadrant=0, max_shift_cells=16,
                                       refusals=refusals) is None
    assert refusals[0].reason == dict(sparse="insufficient_overlap", constant="no_signal",
                                    repetitive="repetition_unresolved", outside="no_signal")[kind]


@pytest.mark.parametrize("radius", (-1, True, 1.5, 10**9))
def test_canvas_bounds_refuse_before_warp_or_allocation(monkeypatch, radius):
    left, right = _small_pair()
    def forbidden(*args, **kwargs):
        raise AssertionError("allocation/warp reached after failed bounds")
    h = plane_graph.cell_grid_homography(left, right, 0)
    monkeypatch.setattr(plane_graph, "cell_grid_homography", lambda *args: h)
    monkeypatch.setattr(plane_graph, "warp_signature", forbidden)
    monkeypatch.setattr(plane_graph.np, "zeros", forbidden)
    refusals = []
    assert plane_graph.match_plane_pair(left, right, quadrant=0, max_shift_cells=radius,
                                       refusals=refusals) is None
    assert refusals[0].reason == "no_signal" and "canvas" in refusals[0].detail


def test_canvas_dimensions_obey_declared_maxima():
    assert plane_graph._common_canvas_geometry((256, 256), 32) == ((320, 320), (448, 448))
    for shape, radius in (((1001, 1000), 0), ((800, 800), 150), ((1, 1), 158)):
        with pytest.raises(ValueError, match="budget"):
            plane_graph._common_canvas_geometry(shape, radius)
