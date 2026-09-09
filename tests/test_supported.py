"""Deciding which cells of a plane an image actually supports.

The distinction that matters is between a cell the picture contradicts, a cell
that simply has too little to go on, and a cell it never saw. Collapsing any
pair of these is how a doorway gets filled in or a flat wool wall gets read as
active evidence against itself, so the tests below are mostly about keeping
the five verdicts apart and about bounds coming from support rather than from
a rectangle. The five synthetic controls at the end (`test_control_*`) are the
ones a lead review asked for by name: uniform-wall, opening, foreground-slab,
invalid-pixel, nonzero-phase - each checked against real ground truth built
into the synthetic image, not against another run of the code under test.
"""
import numpy as np
import pytest

from mc_photogrammetry.reconstruction.supported import (
    CONTRADICTED,
    FOREGROUND_OCCLUSION,
    OUTSIDE_IMAGE,
    SUPPORT_THRESHOLD,
    SUPPORTED,
    UNOBSERVED,
    WEAK_EVIDENCE,
    CellSupport,
    SupportedPlane,
    _edge_profiles,
    phase_at_period,
    support_cells,
)


def _gridded(size, period, phase=0.0, noise=0.0, seed=1):
    """A patch whose edges sit exactly on a lattice of the given period."""
    field = np.zeros((size, size), np.float32)
    lines = np.arange(phase, size, period)
    for line in lines:
        index = int(round(line)) % size
        field[index, :] = 1.0
        field[:, index] = 1.0
    if noise:
        field = field + np.random.default_rng(seed).normal(0, noise, field.shape)
    return field.astype(np.float32)


def _scrambled(size, seed=2):
    """Edges everywhere, agreeing with no particular grid."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, (size, size)).astype(np.float32)


def test_a_patch_on_its_own_lattice_is_supported():
    patch = _gridded(256, 32.0)
    coverage = np.ones_like(patch)
    found = support_cells(patch, coverage, 32.0, (0.0, 0.0))
    assert found
    verdicts = [cell.verdict for cell in found]
    assert verdicts.count(SUPPORTED) > len(found) * 0.7


def test_edges_that_ignore_the_grid_are_contradicted_not_supported():
    patch = _scrambled(256)
    found = support_cells(patch, np.ones_like(patch), 32.0, (0.0, 0.0))
    assert found
    supported = sum(1 for cell in found if cell.verdict == SUPPORTED)
    assert supported <= len(found) * 0.3, "noise should not read as a lattice"


def test_a_cell_outside_the_frame_is_unobserved_rather_than_contradicted():
    """A hole in the photograph is not a hole in the wall.

    Reporting an unseen cell as contradicted would make missing evidence look
    like evidence of absence, and reporting it as supported would invent a wall.
    """
    patch = _gridded(256, 32.0)
    coverage = np.ones_like(patch)
    coverage[:128, :] = 0.0          # the top half was never in frame
    found = support_cells(patch, coverage, 32.0, (0.0, 0.0))
    top = [c for c in found if c.cell[0] < 4]
    assert top and all(c.verdict == UNOBSERVED for c in top)
    bottom = [c for c in found if c.cell[0] >= 4]
    assert any(c.verdict == SUPPORTED for c in bottom)


def test_a_lattice_at_the_wrong_phase_does_not_read_as_support():
    # Same spacing, offset by half a block: a surface at another depth.
    patch = _gridded(256, 32.0, phase=16.0)
    found = support_cells(patch, np.ones_like(patch), 32.0, (0.0, 0.0))
    supported = sum(1 for cell in found if cell.verdict == SUPPORTED)
    assert supported <= len(found) * 0.3


def test_bounds_come_from_supported_cells_and_not_from_a_rectangle():
    cells = (CellSupport((2, 3), SUPPORTED, 0.9, 1.0),
             CellSupport((5, 7), SUPPORTED, 0.9, 1.0),
             CellSupport((9, 9), CONTRADICTED, 0.1, 1.0),
             CellSupport((0, 0), UNOBSERVED, 0.0, 0.0))
    plane = SupportedPlane(axis=2, distance=6.0, phase=(0.0, 0.0), cells=cells)
    assert plane.bounds() == ((2, 3), (5, 7))
    assert len(plane.supported) == 2


def test_a_plane_with_nothing_supported_states_no_extent():
    plane = SupportedPlane(axis=0, distance=4.0, phase=(0.0, 0.0),
                           cells=(CellSupport((1, 1), CONTRADICTED, 0.0, 1.0),))
    assert plane.bounds() is None
    assert plane.holes() == ()


def test_a_doorway_inside_the_wall_is_reported_as_a_hole():
    """Contradicted cells inside the supported extent are the interesting ones.

    A wall with a doorway is one plane and not one rectangle; filling the
    doorway because the extent looked rectangular is inventing geometry.
    """
    cells = (CellSupport((0, 0), SUPPORTED, 0.9, 1.0),
             CellSupport((0, 4), SUPPORTED, 0.9, 1.0),
             CellSupport((4, 0), SUPPORTED, 0.9, 1.0),
             CellSupport((4, 4), SUPPORTED, 0.9, 1.0),
             CellSupport((2, 2), CONTRADICTED, 0.1, 1.0),
             CellSupport((9, 9), CONTRADICTED, 0.1, 1.0))
    plane = SupportedPlane(axis=1, distance=8.0, phase=(0.0, 0.0), cells=cells)
    assert [c.cell for c in plane.holes()] == [(2, 2)]


def test_a_period_too_small_to_divide_is_refused():
    patch = _gridded(64, 32.0)
    assert support_cells(patch, np.ones_like(patch), 1.0, (0.0, 0.0)) == []


def test_coverage_is_reported_per_cell():
    patch = _gridded(128, 32.0)
    coverage = np.ones_like(patch)
    coverage[:32, :32] = 0.25
    found = {cell.cell: cell for cell in support_cells(patch, coverage, 32.0, (0.0, 0.0))}
    assert found[(0, 0)].coverage == pytest.approx(0.25, abs=0.05)
    assert found[(0, 0)].verdict == UNOBSERVED


def test_phase_at_period_finds_where_the_seams_are():
    """The distance gives the period; the phase still has to be read."""
    period = 32.0
    profile = np.zeros(512, np.float32)
    profile[np.arange(11, 512, int(period))] = 1.0
    found = phase_at_period(profile, np.arange(profile.size), period)
    # Phase is circular, so 11 and 11 +/- period are the same answer.
    error = abs((found - 11.0 + period / 2) % period - period / 2)
    assert error <= 2.0, f"read the phase as {found}, not 11"


def test_fitting_the_phase_does_not_let_noise_clear_the_threshold():
    """Fitting raises what chance scores, so the threshold has to clear the
    *fitted* chance level and not the fixed one.

    Measured at 0.378 mean and 0.409 worst over four seeds; SUPPORT_THRESHOLD is
    0.55. This test is the guard on that margin: lowering the threshold towards
    chance, or widening the tolerance without re-measuring, breaks it.
    """
    rng = np.random.default_rng(7)
    patch = rng.normal(0.0, 1.0, (512, 512)).astype(np.float32)
    across, down = _edge_profiles(patch)
    phase = (phase_at_period(across.mean(axis=0), np.arange(across.shape[1]), 32.0),
             phase_at_period(down.mean(axis=1), np.arange(down.shape[0]), 32.0))
    found = support_cells(patch, np.ones(patch.shape), 32.0, phase)
    scores = np.array([cell.score for cell in found])
    assert scores.max() < SUPPORT_THRESHOLD, (
        f"noise reached {scores.max():.3f} against a threshold of "
        f"{SUPPORT_THRESHOLD}")


def test_a_photographed_blank_wall_is_weak_evidence_not_an_unseen_one():
    """Three ways to end up with no seam agreement, and they are not the same
    claim. A region the camera saw and found featureless (a flat material)
    carries no evidence either way - `weak_evidence`, corrected from an
    earlier, wrong version of this test that asserted `contradicted`, which
    was exactly the bug a lead review caught: a uniform test plane scored
    every cell as active evidence against itself. A region the camera never
    saw is a different claim again - `outside_image` - and reporting either
    one as the other is how a hole in the photograph becomes a hole in the
    wall, or a flat material becomes a doorway.
    """
    blank = np.zeros((256, 256), np.float32)
    photographed = support_cells(blank, np.ones_like(blank), 32.0, (0.0, 0.0))
    assert photographed and all(c.verdict == WEAK_EVIDENCE for c in photographed)
    unseen = support_cells(blank, np.zeros_like(blank), 32.0, (0.0, 0.0))
    assert unseen and all(c.verdict == OUTSIDE_IMAGE for c in unseen)


def test_local_phase_finds_a_lattice_whose_phase_drifts_across_the_patch():
    """A single global phase cannot fit a plane whose predicted seams have
    drifted out of step with the real ones by the far edge - which is exactly
    what calibration's own small residual error does over a wide patch. Local
    windowing must still find each region's cells supported.
    """
    size = 512
    period = 32.0
    field = np.zeros((size, size), np.float32)
    # The real lattice drifts half a period across the patch width - like a
    # small, compounding calibration error would.
    for row in range(size):
        drift = (row / size) * (period / 2.0)
        lines = np.arange(drift, size, period)
        for line in lines:
            index = int(round(line)) % size
            field[row, index] = 1.0
    coverage = np.ones_like(field)

    # A single phase fit at the top of the patch cannot also fit the bottom.
    global_found = support_cells(field, coverage, period, (0.0, 0.0))
    global_supported = sum(1 for c in global_found if c.verdict == SUPPORTED)

    local_found = support_cells(field, coverage, period, (0.0, 0.0),
                                local_window_periods=1.5)
    local_supported = sum(1 for c in local_found if c.verdict == SUPPORTED)
    assert local_supported > global_supported * 1.5


def test_local_window_does_not_raise_noise_above_the_threshold():
    """The chance level under local fitting must stay measured, not assumed:
    an unbiased phase estimator should not let a smaller sample invent
    agreement, unlike a max-over-offsets search would.
    """
    rng = np.random.default_rng(11)
    patch = rng.normal(0.0, 1.0, (512, 512)).astype(np.float32)
    found = support_cells(patch, np.ones_like(patch), 32.0, (0.0, 0.0),
                          local_window_periods=1.5)
    scores = np.array([c.score for c in found])
    assert scores.max() < SUPPORT_THRESHOLD, (
        f"noise reached {scores.max():.3f} under local windowing")


def test_local_window_of_none_reproduces_the_global_behaviour():
    patch = _gridded(256, 32.0)
    coverage = np.ones_like(patch)
    with_none = support_cells(patch, coverage, 32.0, (0.0, 0.0),
                              local_window_periods=None)
    without_arg = support_cells(patch, coverage, 32.0, (0.0, 0.0))
    assert [c.score for c in with_none] == [c.score for c in without_arg]


def _bands(size, period):
    """Alternating bright/dark square bands: dense edge energy everywhere, not
    just at thin grid lines - so a cell inside a coarse or fine occluder region
    has real content to test, not a lucky blank interior."""
    idx = np.arange(size)
    row_band = (idx // int(period)) % 2
    col_band = (idx // int(period)) % 2
    return np.logical_xor(row_band[:, None], col_band[None, :]).astype(np.float32)


def test_control_uniform_wall_is_weak_evidence_not_contradicted():
    """A flat wool or concrete face - a real, common Minecraft material, not a
    hypothetical - carries no seam evidence either way and must not read as
    active evidence against itself."""
    patch = np.full((256, 256), 128.0, dtype=np.float32)
    found = support_cells(patch, np.ones_like(patch), 32.0, (0.0, 0.0))
    assert found
    assert all(c.verdict == WEAK_EVIDENCE for c in found)


def test_control_opening_onto_a_farther_surface_is_contradicted():
    """A farther surface, seen past an opening, reads *finer* once rectified
    as if it were at the nearer plane's distance - and should read
    `contradicted`, not `foreground_occlusion`."""
    period = 32.0
    size = 900
    patch = _gridded(size, period)
    region, off = 500, (900 - 500) // 2
    patch[off:off + region, off:off + region] = _bands(region, period / 3.0)
    found = support_cells(patch, np.ones_like(patch), period, (0.0, 0.0),
                          local_window_periods=1.5)
    by_cell = {c.cell: c.verdict for c in found}
    lo = int((off + 100) // period)
    hi = int((off + region - 100) // period)
    interior = [by_cell.get((r, c)) for r in range(lo, hi) for c in range(lo, hi)]
    assert interior and all(v == CONTRADICTED for v in interior)


def test_control_foreground_slab_is_occlusion_not_contradiction():
    """A nearer surface reads *coarser* once rectified as if it were at the
    farther plane's distance. Some interior cells legitimately read as
    `weak_evidence` (a large uniform band's own interior has little local
    detail at the cell's own scale) or coincidentally `supported` (an
    unrelated period can align with the predicted seams by chance) - the
    property this control actually checks is that wherever the cell *does*
    read as a disagreement, it reads as the right kind of disagreement.
    """
    period = 32.0
    size = 900
    patch = _gridded(size, period)
    region, off = 500, (900 - 500) // 2
    patch[off:off + region, off:off + region] = _bands(region, period * 2.2)
    found = support_cells(patch, np.ones_like(patch), period, (0.0, 0.0),
                          local_window_periods=1.5)
    by_cell = {c.cell: c.verdict for c in found}
    lo = int((off + 100) // period)
    hi = int((off + region - 100) // period)
    interior = [by_cell.get((r, c)) for r in range(lo, hi) for c in range(lo, hi)]
    occlusion = sum(1 for v in interior if v == FOREGROUND_OCCLUSION)
    contradicted = sum(1 for v in interior if v == CONTRADICTED)
    assert occlusion > 0, "the control should find at least some occlusion"
    assert occlusion >= 4 * max(contradicted, 1), (
        f"disagreeing cells should read mostly as occlusion, not contradiction: "
        f"{occlusion} occlusion vs {contradicted} contradicted")


def test_control_invalid_pixels_are_outside_image():
    patch = _gridded(256, 32.0)
    coverage = np.ones_like(patch)
    coverage[:64, :] = 0.0
    found = support_cells(patch, coverage, 32.0, (0.0, 0.0))
    top = [c for c in found if c.cell[0] < 2]
    assert top and all(c.verdict == OUTSIDE_IMAGE for c in top)


def test_control_nonzero_phase_lattice_is_still_supported():
    """A real lattice offset from the phase the caller happens to pass in must
    still be recognised - local windowing re-reads the true phase per cell
    rather than trusting a caller's possibly-wrong global guess."""
    patch = _gridded(512, 32.0, phase=11.0)
    found = support_cells(patch, np.ones_like(patch), 32.0, (0.0, 0.0),
                          local_window_periods=1.5)
    supported = sum(1 for c in found if c.verdict == SUPPORTED)
    assert supported > len(found) * 0.7
