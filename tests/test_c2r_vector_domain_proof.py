"""Retained C2-R-A analytic counterexample; no vector-domain builder exists."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from mc_photogrammetry.reconstruction import lattice_planes, plane_graph
from mc_photogrammetry.reconstruction import plane_residue
from mc_photogrammetry.reconstruction import plane_axis_support as c1
from mc_photogrammetry.reconstruction import plane_shift_domain as p1d


_SAMPLE_STEP = 8.0 / 127.0


def _world_texture(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (2.0 + 0.31 * np.cos(2.0 * np.pi * x / 3.7)
            + 0.23 * np.sin(2.0 * np.pi * y / 4.3)
            + 1.8 * np.exp(-((x - 21.3) ** 2 / 3.2
                             + (y - 17.8) ** 2 / 5.1))
            + 0.7 * np.exp(-((x - 13.2) ** 2 / 7.0
                             + (y - 31.4) ** 2 / 2.1)))


def _physical_signature(
    frame: str,
    translation: tuple[float, float],
    origin: tuple[float, float],
    distance: float,
) -> tuple[lattice_planes.PlaneReading, lattice_planes.PlaneSignature]:
    """Sample one common fronto-parallel plane in one cell per block."""
    b_value = 1.0
    period = b_value / (distance * _SAMPLE_STEP)
    phase = tuple(((-value) % 1.0) * period for value in translation)
    rectification = lattice_planes.PlaneRectification(
        axis_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        focal=80.0,
        principal_point=(31.0, 27.0),
        centre=(4.0, 4.0),
    )
    phase_evidence = lattice_planes.PeriodicEnergyPhaseEvidence(
        phase_uv_pixels=phase,
        period_pixels=period,
        patch_centre_uv=(4.0, 4.0),
        extent=4.0,
        samples=128,
        response_magnitude_uv=(8.0, 7.0),
        profile_mass_uv=(16.0, 14.0),
        concentration_uv=(0.5, 0.5),
        unavailable_reason_uv=(None, None),
    )
    reading = lattice_planes.PlaneReading(
        frame=frame,
        axis=2,
        side=1.0,
        alternatives=(lattice_planes.DistanceAlternative(b_value, distance, 1.0),),
        period_pixels=period,
        phase_pixels=phase,
        extent=4.0,
        samples=128,
        polygon=None,
        coverage_cells=1600,
        structural_share=1.0,
        confidence=1.0,
        rectification=rectification,
        phase_evidence=phase_evidence,
    )
    rows, columns = np.indices((48, 48), dtype=np.float64)
    cells = _world_texture(
        translation[0] + origin[0] + columns,
        translation[1] + origin[1] + rows,
    ).astype(np.float32)
    signature = lattice_planes.PlaneSignature(
        reading=reading,
        blocks_per_period=b_value,
        cells=cells,
        interior=np.zeros_like(cells),
        coverage=np.ones(cells.shape, dtype=bool),
        origin_uv=origin,
    )
    return reading, signature


def _surface_peak(
    left: lattice_planes.PlaneSignature,
    right: lattice_planes.PlaneSignature,
    homography: np.ndarray,
) -> tuple[int, int]:
    warped, warped_coverage = lattice_planes.warp_signature(
        left, homography, right.cells.shape)
    overlap = warped_coverage & right.coverage
    first = np.where(overlap, warped, 0.0).astype(np.float64)
    second = np.where(overlap, right.cells, 0.0).astype(np.float64)
    first[overlap] -= float(first[overlap].mean())
    second[overlap] -= float(second[overlap].mean())
    surface = plane_graph._correlation_surface(
        first, second, max_shift_cells=8)
    assert surface is not None
    return plane_graph._peak_statistics(surface, max_shift_cells=8)[-1]


def test_unequal_distance_frontoparallel_case_uses_repaired_block_grid_map():
    """One-cell/block grids stay unit-scale across unequal camera distances."""
    left_translation = (0.2, -0.35)
    right_translation = (1.6, 0.45)
    left_origin = (1.0, -2.0)
    right_origin = (4.0, 3.0)
    left_reading, left = _physical_signature(
        "left", left_translation, left_origin, distance=2.0)
    right_reading, right = _physical_signature(
        "right", right_translation, right_origin, distance=4.0)

    # Independently, each stored cell is one block. With no surface rotation,
    # equal world points at zero endpoint translation map by identity plus the
    # difference between the declared block-grid origins.
    expected_block_grid_map = np.asarray((
        (1.0, 0.0, right_origin[0] - left_origin[0]),
        (0.0, 1.0, right_origin[1] - left_origin[1]),
        (0.0, 0.0, 1.0),
    ))
    actual = plane_graph.cell_grid_homography(left, right, 0)
    np.testing.assert_allclose(expected_block_grid_map[:2, :2], np.eye(2))
    np.testing.assert_array_equal(actual, expected_block_grid_map)

    # Named legacy comparison: composing image-plane maps applies the distance
    # ratio a second time to signatures that are already measured per block.
    left_distance = left_reading.distance_for(left.blocks_per_period)
    right_distance = right_reading.distance_for(right.blocks_per_period)
    assert left_distance == 2.0 and right_distance == 4.0
    left_context, right_context = left_reading.rectification, right_reading.rectification
    left_image_h = lattice_planes.rectification_homography(
        left_reading.axis, np.asarray(left_context.axis_matrix), left_context.focal,
        left_context.principal_point, left_reading.side, left_distance)
    right_image_h = lattice_planes.rectification_homography(
        right_reading.axis, np.asarray(right_context.axis_matrix), right_context.focal,
        right_context.principal_point, right_reading.side, right_distance)
    left_origin = np.asarray(((1.0, 0.0, -left.origin_uv[0]),
                              (0.0, 1.0, -left.origin_uv[1]),
                              (0.0, 0.0, 1.0)))
    right_origin = np.asarray(((1.0, 0.0, right.origin_uv[0]),
                               (0.0, 1.0, right.origin_uv[1]),
                               (0.0, 0.0, 1.0)))
    legacy_camera_composed_homography = (
        left_origin @ np.linalg.inv(left_image_h) @ right_image_h @ right_origin)
    legacy_camera_composed_homography /= legacy_camera_composed_homography[2, 2]
    np.testing.assert_allclose(legacy_camera_composed_homography[:2, :2],
                               0.5 * np.eye(2), atol=1e-12)

    # A deliberately inconsistent selected distance produces another
    # axis-preserving scale. Merely relaxing the predicate to accept arbitrary
    # positive diagonal scale would also accept such wrong b/distance geometry.
    wrong_right_reading = replace(
        right_reading,
        alternatives=(lattice_planes.DistanceAlternative(1.0, 3.0, 1.0),),
    )
    wrong_right = replace(right, reading=wrong_right_reading)
    wrong_distance = wrong_right.reading.distance_for(wrong_right.blocks_per_period)
    assert wrong_distance == 3.0
    wrong_image_h = lattice_planes.rectification_homography(
        wrong_right_reading.axis, np.asarray(right_context.axis_matrix), right_context.focal,
        right_context.principal_point, wrong_right_reading.side, wrong_distance)
    legacy_wrong_distance_homography = (
        left_origin @ np.linalg.inv(left_image_h) @ wrong_image_h @ right_origin)
    legacy_wrong_distance_homography /= legacy_wrong_distance_homography[2, 2]
    np.testing.assert_allclose(legacy_wrong_distance_homography[:2, :2],
                               (2.0 / 3.0) * np.eye(2), atol=1e-12)
    np.testing.assert_array_equal(
        plane_graph.cell_grid_homography(left, wrong_right, 0), actual)

    # Exercise both paths: the repaired helper and independent block-grid map
    # now use the same correlation convention.
    actual_match = plane_graph.match_plane_pair(
        left, right, quadrant=0, max_shift_cells=8)
    assert actual_match is not None
    assert actual_match.cell_shift_rc == (0, 1)
    assert _surface_peak(left, right, expected_block_grid_map) == (0, 1)

    # The repaired H and direct independent path agree on correspondence. The
    # exact lower-ranked carry remains in P1-D's domain because the top alone
    # still cannot determine the integer lift.
    truth_carry_rc = (1, 1)
    actual_match = replace(actual_match, left_plane=0, right_plane=0)
    solution = plane_graph.ComponentSolution(
        frames=("left", "right"),
        quadrants={"left": 0, "right": 0},
        translations={"left": (0.0, 0.0, 0.0),
                      "right": (0.0, 0.0, 0.0)},
        axis_residuals={0: 0.0, 1: 0.0, 2: 0.0},
        axis_resultants={0: 1.0, 1: 1.0, 2: 1.0},
        scale_known=True,
        blocks_per_period={("left", 0): 1.0, ("right", 0): 1.0},
        reason="",
        alternatives=(),
    )
    support = c1.build_axis_support(
        solution, (left_reading, right_reading), (actual_match,), (left, right))
    assert support.refusal is None
    shift_domains = p1d.build_shift_domains(
        support,
        (actual_match,),
        (left, right),
        max_shift_cells=8,
        readings=(left_reading, right_reading),
        budget=p1d.ShiftDomainBudget(
            max_domains=4,
            max_padded_fft_cells=100_000,
            max_shift_positions=100_000,
            max_overlap_cell_visits=10_000_000,
            max_buffered_entries=100_000,
        ),
    )
    assert shift_domains.status == "complete"
    entry = next(
        entry for entry in shift_domains.domains[0].entries
        if (entry.row_shift, entry.column_shift) == truth_carry_rc)
    phase = plane_residue.hypothesis_phase_shift_evidence(
        actual_match,
        {("left", 0): left_reading, ("right", 0): right_reading},
    )
    assert phase.refusal is None
    residues = {
        (record.frame, record.local_axis): record.residue_candidates[0]
        for record in phase.phase_candidates
    }
    truth_vector = (
        entry.column_shift + residues[("left", "u")] - residues[("right", "u")],
        entry.row_shift + residues[("left", "v")] - residues[("right", "v")],
    )
    np.testing.assert_allclose(truth_vector, (1.4, 0.8), atol=1e-12)
