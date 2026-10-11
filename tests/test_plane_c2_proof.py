"""C2-A proof harness: first world-truth counterexample through real correlation code."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mc_photogrammetry.reconstruction import lattice_planes, plane_graph, plane_residue
from mc_photogrammetry.reconstruction import plane_axis_support as c1


PERIOD_PIXELS = 31.0 / 8.0
PATCH_EXTENT = 4.0
PATCH_SAMPLES = 32


def _world_texture(x, y):
    """Nonperiodic material frequencies plus a unique asymmetric fiducial."""
    return (2.0 + 0.31 * np.cos(2.0 * np.pi * x / 3.7)
            + 0.23 * np.sin(2.0 * np.pi * y / 4.3)
            + 1.8 * np.exp(-((x - 21.3) ** 2 / 3.2 + (y - 17.8) ** 2 / 5.1))
            + 0.7 * np.exp(-((x - 13.2) ** 2 / 7.0 + (y - 31.4) ** 2 / 2.1)))


def _reading(frame, t_xy, *, axis_matrix=None, focal=80.0,
             principal=(31.0, 27.0), distance=1.0, side=1.0,
             centre=(4.0, 4.0), extent=PATCH_EXTENT, samples=PATCH_SAMPLES,
             b=1.0):
    axis_matrix = np.eye(3) if axis_matrix is None else np.asarray(axis_matrix, dtype=float)
    phases = tuple(((-float(t) % 1.0) * PERIOD_PIXELS) % PERIOD_PIXELS
                   for t in t_xy)
    rect = lattice_planes.PlaneRectification(
        axis_matrix=tuple(tuple(float(x) for x in row) for row in axis_matrix),
        focal=float(focal), principal_point=tuple(map(float, principal)),
        centre=tuple(map(float, centre)))
    phase_evidence = lattice_planes.PeriodicEnergyPhaseEvidence(
        phase_uv_pixels=phases, period_pixels=PERIOD_PIXELS,
        patch_centre_uv=tuple(map(float, centre)), extent=float(extent),
        samples=int(samples), response_magnitude_uv=(8.0, 7.0),
        profile_mass_uv=(16.0, 14.0), concentration_uv=(0.5, 0.5),
        unavailable_reason_uv=(None, None))
    return lattice_planes.PlaneReading(
        frame=frame, axis=2, side=float(side),
        alternatives=(lattice_planes.DistanceAlternative(
            float(b), float(distance), 1.0),),
        period_pixels=PERIOD_PIXELS, phase_pixels=phases,
        extent=float(extent), samples=int(samples), polygon=None,
        coverage_cells=36, structural_share=1.0, confidence=1.0,
        rectification=rect, phase_evidence=phase_evidence)


def _signature(reading, t_xy, *, shape=(40, 40), origin=(0.0, 0.0)):
    rows, columns = np.indices(shape, dtype=np.float64)
    x = float(t_xy[0]) + float(origin[0]) + columns
    y = float(t_xy[1]) + float(origin[1]) + rows
    cells = _world_texture(x, y).astype(np.float32)
    return lattice_planes.PlaneSignature(
        reading=reading, blocks_per_period=reading.alternatives[0].blocks_per_period,
        cells=cells, interior=np.zeros_like(cells),
        coverage=np.ones(shape, dtype=bool), origin_uv=tuple(map(float, origin)))


def _truth_case(*, t_left=(0.2, -0.35), t_right=(1.6, 0.45)):
    left_reading = _reading("left", t_left)
    right_reading = _reading("right", t_right)
    left_signature = _signature(left_reading, t_left)
    right_signature = _signature(right_reading, t_right)
    homography = plane_graph.cell_grid_homography(left_signature, right_signature, 0)
    match = plane_graph.match_plane_pair(
        left_signature, right_signature, quadrant=0, max_shift_cells=8)
    assert match is not None, "synthetic fiducial should produce a unique match"
    # match_plane_pair intentionally leaves candidate plane indices unbound;
    # bind this admitted pair to the exact signature/readings for C1 evidence.
    match = replace(match, left_plane=0, right_plane=0)
    solution = plane_graph.ComponentSolution(
        frames=("left", "right"), quadrants={"left": 0, "right": 0},
        translations={"left": (0.0, 0.0, 0.0), "right": (0.0, 0.0, 0.0)},
        axis_residuals={0: 0.0, 1: 0.0, 2: 0.0},
        axis_resultants={0: 1.0, 1: 1.0, 2: 1.0}, scale_known=True,
        blocks_per_period={("left", 0): 1.0, ("right", 0): 1.0},
        reason="", alternatives=())
    support = c1.build_axis_support(
        solution, (left_reading, right_reading), (match,),
        (left_signature, right_signature))
    assert support.refusal is None
    return (t_left, t_right, left_reading, right_reading,
            left_signature, right_signature, homography, match, support)


def test_world_truth_equation_has_integer_lift_counterexample():
    (t_left, t_right, left, right, sig_left, sig_right,
     homography, match, support) = _truth_case()
    # Verify the exact homography inputs and normalized result C1 bound.
    left_geom = c1._signature_geometry_digest(sig_left, 0)
    right_geom = c1._signature_geometry_digest(sig_right, 0)
    homography_digest = c1._grid_homography_digest(
        sig_left, sig_right, 0, 0, 0, homography)
    assert left_geom is not None and right_geom is not None
    assert homography_digest is not None
    mapped = {record.right_local_axis: record for record in support.mapped_cell_shifts}
    assert {record.left_signature_geometry_sha256 for record in mapped.values()} == {left_geom}
    assert {record.right_signature_geometry_sha256 for record in mapped.values()} == {right_geom}
    assert {record.cell_homography_sha256 for record in mapped.values()} == {homography_digest}

    phase = plane_residue.hypothesis_phase_shift_evidence(
        match, {("left", 0): left, ("right", 0): right})
    assert phase.refusal is None
    truth = (t_right[0] - t_left[0], t_right[1] - t_left[1])
    for local_axis, delta in (("u", truth[0]), ("v", truth[1])):
        label = 0 if local_axis == "u" else 1
        left_residues = next(value for value in phase.phase_candidates
                             if value.frame == "left" and value.local_axis == local_axis)
        right_residues = next(value for value in phase.phase_candidates
                              if value.frame == "right" and value.local_axis == local_axis)
        integer = mapped[local_axis].signed_world_integer
        candidate = integer + left_residues.residue_candidates[0] \
            - right_residues.residue_candidates[0]
        if local_axis == "u":
            assert candidate == pytest.approx(delta, abs=2e-5)
        else:
            # The independently known relative translation is +0.8, while the
            # admitted evidence yields -0.2. They agree modulo one block only.
            assert delta == pytest.approx(0.8, abs=2e-5)
            assert candidate == pytest.approx(-0.2, abs=2e-5)
            assert candidate - delta == pytest.approx(-1.0, abs=2e-5)
            assert (candidate - delta) == pytest.approx(round(candidate - delta))
        assert abs(candidate - (integer - left_residues.residue_candidates[0]
                                + right_residues.residue_candidates[0])) > 0.1
        assert mapped[local_axis].right_local_axis_label == label
        assert match.cell_shift_rc[1 if local_axis == "u" else 0] \
            == mapped[local_axis].right_grid_shift
