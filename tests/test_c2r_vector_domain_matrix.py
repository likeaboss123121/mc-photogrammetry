"""Independent post-P1-E analytic truth matrix for C2-R-A2.

These tests deliberately stop at the upstream complete P1-D domain.  The
candidate equation is exercised here as a proof oracle before a domain builder
is allowed to exist.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mc_photogrammetry.reconstruction import lattice_planes, plane_graph
from mc_photogrammetry.reconstruction import plane_residue
from mc_photogrammetry.reconstruction import plane_axis_support as c1
from mc_photogrammetry.reconstruction import plane_shift_domain as p1d


# Literal independent transform table.  Do not replace with axis_transform.
_TRANSFORM = (
    ((0, 1), (1, 1), (2, 1)),
    ((2, 1), (1, 1), (0, -1)),
    ((0, -1), (1, 1), (2, -1)),
    ((2, -1), (1, 1), (0, 1)),
)
_SUPPORTED_B = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 3, 4, 8, 16)
_ORIGIN_L = (3.0, -2.0)
_ORIGIN_R = (-4.0, 5.0)
_GRID_SIDE = 48
_RADIUS = 4


def _texture(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Positive asymmetric nonperiodic world texture, evaluated independently."""
    rng = np.random.default_rng(411)
    centres = rng.uniform(-30.0, 85.0, size=(48, 2))
    widths = rng.uniform(0.45, 2.7, size=(48, 2))
    amplitudes = rng.uniform(0.25, 1.2, size=48)
    result = np.full(np.broadcast(x, y).shape, 3.0, dtype=np.float64)
    for (cx, cy), (wx, wy), amplitude in zip(centres, widths, amplitudes):
        result += amplitude * np.exp(-((x - cx) ** 2 / wx + (y - cy) ** 2 / wy))
    return result


def _local_labels(normal: int) -> tuple[int, int]:
    return tuple(axis for axis in range(3) if axis != normal)  # type: ignore[return-value]


def _world_normal(local_normal: int, quadrant: int) -> tuple[int, int]:
    return _TRANSFORM[quadrant][local_normal]


def _basis(local_normal: int, absolute_quadrant: int,
           world_tangents: tuple[int, int]) -> np.ndarray:
    """Independent 2x2 local-(u,v) to common-world tangent map."""
    result = np.zeros((2, 2), dtype=np.float64)
    for local_column, label in enumerate(_local_labels(local_normal)):
        world_axis, sign = _TRANSFORM[absolute_quadrant][label]
        result[world_tangents.index(world_axis), local_column] = sign
    return result


def _reading(frame: str, axis: int, b_value: float, distance: float,
             side: float, origin: tuple[float, float],
             phase_fraction: tuple[float, float]) -> lattice_planes.PlaneReading:
    extent = 8.0
    # plane_signature's own endpoint-inclusive origin rule returns `origin`.
    centre = tuple(extent + value / distance
                   for value in origin)
    period = 20.0
    phase = tuple(value * period for value in phase_fraction)
    camera_axes = np.zeros((3, 3), dtype=np.float64)
    camera_axes[axis, 2] = side
    for coordinate, label in enumerate(_local_labels(axis)):
        camera_axes[label, coordinate] = 1.0
    rect = lattice_planes.PlaneRectification(
        axis_matrix=tuple(tuple(float(value) for value in row) for row in camera_axes),
        focal=20.0, principal_point=(512.0, 512.0), centre=centre)
    phase_evidence = lattice_planes.PeriodicEnergyPhaseEvidence(
        phase_uv_pixels=phase, period_pixels=period, patch_centre_uv=centre,
        extent=extent, samples=128, response_magnitude_uv=(8.0, 7.0),
        profile_mass_uv=(16.0, 14.0), concentration_uv=(0.5, 0.5),
        unavailable_reason_uv=(None, None))
    return lattice_planes.PlaneReading(
        frame=frame, axis=axis, side=side,
        alternatives=(lattice_planes.DistanceAlternative(b_value, distance, 1.0),),
        period_pixels=period, phase_pixels=phase, extent=extent, samples=128,
        polygon=None, coverage_cells=1600, structural_share=1.0, confidence=1.0,
        rectification=rect, phase_evidence=phase_evidence)


def _phase_candidates(match: plane_graph.PlaneMatch,
                      readings: tuple[lattice_planes.PlaneReading, ...]):
    evidence = plane_residue.hypothesis_phase_shift_evidence(
        match, {(value.frame, 0): value for value in readings})
    assert evidence.refusal is None
    assert len(evidence.phase_candidates) == 4
    return evidence


def _signature(reading: lattice_planes.PlaneReading, b_value: float,
               origin: tuple[float, float], basis: np.ndarray,
               offset: np.ndarray) -> lattice_planes.PlaneSignature:
    # Run the real signature factory to establish the endpoint-inclusive grid
    # origin and coverage from camera geometry. Replace only its measured cell
    # values with the independently sampled analytic world texture below.
    measured = lattice_planes.plane_signature(
        np.ones((1024, 1024, 3), dtype=np.float32), reading, b_value,
        max_cells=_GRID_SIDE)
    assert measured.origin_uv == origin
    rows, columns = np.indices(measured.cells.shape, dtype=np.float64)
    local_u = origin[0] + columns
    local_v = origin[1] + rows
    world = basis @ np.stack((local_u, local_v), axis=0).reshape(2, -1)
    world[0] += offset[0]
    world[1] += offset[1]
    cells = _texture(world[0], world[1]).reshape(measured.cells.shape).astype(np.float32)
    return replace(measured, cells=cells, interior=np.zeros_like(cells),
                   coverage=np.ones(cells.shape, dtype=bool))


def _upstream_case(left_normal: int, left_q: int, relative_q: int,
                   left_b: float, right_b: float,
                   left_side: float, right_side: float,
                   *, stop_after_map: bool = False):
    right_normal, _ = _TRANSFORM[relative_q][left_normal]
    right_q = (left_q + relative_q) % 4
    left_world_normal = _world_normal(left_normal, left_q)[0]
    right_world_normal = _world_normal(right_normal, right_q)[0]
    assert right_world_normal == left_world_normal
    tangents = tuple(axis for axis in range(3) if axis != left_world_normal)
    basis_l = _basis(left_normal, left_q, tangents)
    basis_r = _basis(right_normal, right_q, tangents)

    # Independent physical endpoint translations are fixed before any P1-B
    # residues or P1-D entries are generated. They are the truth oracle.
    position_l = np.asarray((0.2, -0.35), dtype=np.float64)
    position_r = np.asarray((1.6, 0.45), dtype=np.float64)
    truth_world_delta = position_r - position_l

    # Forward observation model: the phase signal measures local translation
    # modulo one period. P1-B later expands that measurement into residue aliases.
    left_local_position = basis_l.T @ position_l
    right_local_position = basis_r.T @ position_r
    left_true_residues = tuple(float((-value) % 1.0)
                               for value in left_local_position)
    right_true_residues = tuple(float((-value) % 1.0)
                                for value in right_local_position)

    def measured_fraction(residue: float, b_value: float, origin_value: float,
                          distance: float, extent: float) -> float:
        from fractions import Fraction

        rational = Fraction(b_value).limit_denominator(16)
        assert float(rational) == b_value
        centre = extent + origin_value / distance
        leading = distance * (centre - extent)
        phase_offset = (residue - leading) % (1.0 / rational.denominator)
        return float(phase_offset / b_value)

    # Make P1-B records from the physical translations, not from a candidate
    # carry. The integer signature origins are nonzero and make leading integral.
    fractions_l = tuple(measured_fraction(value, left_b, _ORIGIN_L[index], 2.0, 8.0)
                        for index, value in enumerate(left_true_residues))
    fractions_r = tuple(measured_fraction(value, right_b, _ORIGIN_R[index], 3.0, 8.0)
                        for index, value in enumerate(right_true_residues))
    left = _reading("left", left_normal, left_b, 2.0, left_side,
                    _ORIGIN_L, fractions_l)
    right = _reading("right", right_normal, right_b, 3.0, right_side,
                     _ORIGIN_R, fractions_r)

    sig_l = _signature(left, left_b, _ORIGIN_L, basis_l, position_l)
    sig_r = _signature(right, right_b, _ORIGIN_R, basis_r, position_r)
    actual_h = plane_graph.cell_grid_homography(sig_l, sig_r, relative_q)
    physical_p = basis_l.T @ basis_r
    physical_translation = physical_p @ np.asarray(_ORIGIN_R) - np.asarray(_ORIGIN_L)
    independent_physical_h = np.asarray((
        (physical_p[0, 0], physical_p[0, 1], physical_translation[0]),
        (physical_p[1, 0], physical_p[1, 1], physical_translation[1]),
        (0.0, 0.0, 1.0),
    ))
    np.testing.assert_array_equal(actual_h, independent_physical_h)
    if stop_after_map:
        return {
            "actual_h": actual_h,
            "independent_physical_h": independent_physical_h,
            "basis_left": basis_l,
            "basis_right": basis_r,
            "position_left": position_l,
            "position_right": position_r,
            "truth_world_delta": truth_world_delta,
            "left_origin": sig_l.origin_uv,
            "right_origin": sig_r.origin_uv,
        }
    match_refusals = []
    match = plane_graph.match_plane_pair(
        sig_l, sig_r, quadrant=relative_q, max_shift_cells=_RADIUS,
        refusals=match_refusals)
    assert match is not None, match_refusals
    match = replace(match, left_plane=0, right_plane=0)
    assert match.cell_shift_rc is not None
    assert all(abs(value) <= _RADIUS for value in match.cell_shift_rc)
    evidence = _phase_candidates(match, (left, right))

    solution = plane_graph.ComponentSolution(
        frames=("left", "right"), quadrants={"left": left_q, "right": right_q},
        translations={"left": (0.0, 0.0, 0.0), "right": (0.0, 0.0, 0.0)},
        axis_residuals={0: 0.0, 1: 0.0, 2: 0.0},
        axis_resultants={0: 1.0, 1: 1.0, 2: 1.0}, scale_known=True,
        blocks_per_period={("left", 0): left_b, ("right", 0): right_b},
        reason="", alternatives=())
    support = c1.build_axis_support(solution, (left, right), (match,), (sig_l, sig_r))
    assert support.refusal is None, support.refusal
    domains = p1d.build_shift_domains(
        support, (match,), (sig_l, sig_r), max_shift_cells=_RADIUS,
        readings=(left, right),
        budget=p1d.ShiftDomainBudget(
            max_domains=1, max_padded_fft_cells=100_000,
            max_shift_positions=100_000, max_overlap_cell_visits=10_000_000,
            max_buffered_entries=100_000))
    assert domains.status == "complete", domains.refusals
    domain = domains.domains[0]
    assert all(entry.overlap_cells >= domain.min_overlap_cells
               for entry in domain.entries)

    phase_by_key = {(item.frame, item.local_axis): item
                    for item in evidence.phase_candidates}
    for frame, normal, quadrant, residues in (
            ("left", left_normal, left_q, left_true_residues),
            ("right", right_normal, right_q, right_true_residues)):
        for local_index, residue in enumerate(residues):
            local_name = "u" if local_index == 0 else "v"
            values = phase_by_key[(frame, local_name)].residue_candidates
            assert any(abs(value - residue) <= 1e-12 for value in values), (
                frame, local_name, residue, values)

    candidate_deltas = {}
    true_entry_hits = []
    pair_counts = 0
    for entry in domain.entries:
        entry_axis_hits = {}
        for world_axis_index, world_axis in enumerate(tangents):
            left_local_col = next(index for index, label in enumerate(_local_labels(left_normal))
                                  if _TRANSFORM[left_q][label][0] == world_axis)
            right_local_col = next(index for index, label in enumerate(_local_labels(right_normal))
                                   if _TRANSFORM[right_q][label][0] == world_axis)
            left_local = "u" if left_local_col == 0 else "v"
            right_local = "u" if right_local_col == 0 else "v"
            left_sign = _TRANSFORM[left_q][_local_labels(left_normal)[left_local_col]][1]
            right_sign = _TRANSFORM[right_q][_local_labels(right_normal)[right_local_col]][1]
            left_values = phase_by_key[("left", left_local)].residue_candidates
            right_values = phase_by_key[("right", right_local)].residue_candidates
            right_shift = (entry.column_shift if right_local == "u"
                           else entry.row_shift)
            axis_values = []
            axis_truth_hit = False
            for left_index, left_r in enumerate(left_values):
                for right_index, right_r in enumerate(right_values):
                    pair_counts += 1
                    delta = (right_sign * right_shift + left_sign * left_r
                             - right_sign * right_r)
                    axis_values.append((delta, left_index, right_index))
                    if abs(delta - truth_world_delta[world_axis_index]) <= 1e-12:
                        axis_truth_hit = True
            entry_axis_hits[world_axis] = (axis_truth_hit, axis_values)
        if all(entry_axis_hits[axis][0] for axis in tangents):
            true_entry_hits.append(entry)
        # Materialize every compatible vector logically as the Cartesian
        # product of the two fully enumerated axis-pair lists for this entry.
        vector_count_for_entry = (len(entry_axis_hits[tangents[0]][1])
                                  * len(entry_axis_hits[tangents[1]][1]))
        assert vector_count_for_entry > 0
    assert true_entry_hits, {
        "normal": left_normal, "left_q": left_q, "relative_q": relative_q,
        "truth": truth_world_delta.tolist(), "candidate_counts": [
            len(item.residue_candidates) for item in evidence.phase_candidates],
    }
    # The truth vector uses one and the same complete P1-D entry. Every residue
    # pair for both axes was traversed above; no true carry was preselected.
    truth_entry = true_entry_hits[0]
    for world_axis_index, world_axis in enumerate(tangents):
        left_local_col = next(index for index, label in enumerate(_local_labels(left_normal))
                              if _TRANSFORM[left_q][label][0] == world_axis)
        right_local_col = next(index for index, label in enumerate(_local_labels(right_normal))
                               if _TRANSFORM[right_q][label][0] == world_axis)
        left_local = "u" if left_local_col == 0 else "v"
        right_local = "u" if right_local_col == 0 else "v"
        left_sign = _TRANSFORM[left_q][_local_labels(left_normal)[left_local_col]][1]
        right_sign = _TRANSFORM[right_q][_local_labels(right_normal)[right_local_col]][1]
        left_values = phase_by_key[("left", left_local)].residue_candidates
        right_values = phase_by_key[("right", right_local)].residue_candidates
        right_shift = (truth_entry.column_shift if right_local == "u"
                       else truth_entry.row_shift)
        matching = [(left_index, right_index)
                    for left_index, left_r in enumerate(left_values)
                    for right_index, right_r in enumerate(right_values)
                    if abs((right_sign * right_shift + left_sign * left_r
                            - right_sign * right_r)
                           - truth_world_delta[world_axis_index]) <= 1e-12]
        assert matching
        candidate_deltas[world_axis] = truth_world_delta[world_axis_index]
    return {
        "normal": left_normal, "left_q": left_q, "right_q": right_q,
        "relative_q": relative_q, "left_world_basis": basis_l,
        "right_world_basis": basis_r,
        "peak_rc": match.cell_shift_rc, "truth_world_delta": truth_world_delta,
        "candidate_delta": candidate_deltas, "domain": domain,
        "left_residues": left_true_residues, "right_residues": right_true_residues,
        "true_entry_hits": tuple((entry.row_shift, entry.column_shift)
                                  for entry in true_entry_hits),
        "enumerated_axis_residue_pairs": pair_counts,
        "left_b": left_b, "right_b": right_b,
        "left_side": left_side, "right_side": right_side,
    }


def test_valid_q0_smoke_passes_independent_world_truth_through_c1_and_p1d():
    result = _upstream_case(2, 0, 0, 1.0, 1.0, 1.0, 1.0)
    assert result["true_entry_hits"]


def test_valid_normal_zero_relative_quadrant_one_matches_independent_grid_map():
    result = _upstream_case(
        0, 0, 1, 1.0, 1.0, 1.0, -1.0, stop_after_map=True)
    np.testing.assert_array_equal(result["left_origin"], (3.0, -2.0))
    np.testing.assert_array_equal(result["right_origin"], (-4.0, 5.0))
    np.testing.assert_array_equal(result["position_left"], (0.2, -0.35))
    np.testing.assert_array_equal(result["position_right"], (1.6, 0.45))
    np.testing.assert_allclose(result["truth_world_delta"], (1.4, 0.8), atol=1e-15)
    np.testing.assert_array_equal(
        result["actual_h"],
        ((0.0, 1.0, 2.0), (1.0, 0.0, -2.0), (0.0, 0.0, 1.0)))
    np.testing.assert_array_equal(
        result["independent_physical_h"],
        ((0.0, 1.0, 2.0), (1.0, 0.0, -2.0), (0.0, 0.0, 1.0)))
    legacy_p1e = np.asarray(((0.0, 1.0), (-1.0, 0.0)))
    legacy_translation = legacy_p1e @ np.asarray(result["right_origin"]) - np.asarray(result["left_origin"])
    legacy_h = np.eye(3)
    legacy_h[:2, :2] = legacy_p1e
    legacy_h[:2, 2] = legacy_translation
    assert not np.array_equal(legacy_h, result["independent_physical_h"])
