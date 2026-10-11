"""Independent physical basis checks for the post-P1-G C2-R-A3 gate."""
from __future__ import annotations

import ast
import hashlib
import inspect
import sys
import numpy as np
import pytest
from dataclasses import replace

from mc_photogrammetry.reconstruction import lattice_planes, plane_graph
from mc_photogrammetry.reconstruction import plane_residue
from mc_photogrammetry.reconstruction import plane_axis_support as c1
from mc_photogrammetry.reconstruction import plane_shift_domain as p1d


def _physical_yaw(quadrant: int) -> np.ndarray:
    """Closed-form right-handed Ry(-pi*q/2), rounded at exact quarter turns."""
    theta = -np.pi * quadrant / 2.0
    cosine, sine = np.cos(theta), np.sin(theta)
    matrix = np.asarray(((cosine, 0.0, sine),
                         (0.0, 1.0, 0.0),
                         (-sine, 0.0, cosine)), dtype=np.float64)
    matrix[np.abs(matrix) < 1e-14] = 0.0
    matrix[np.abs(matrix - 1.0) < 1e-14] = 1.0
    matrix[np.abs(matrix + 1.0) < 1e-14] = -1.0
    return matrix


def _world_texture(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(411)
    centres = rng.uniform(-25.0, 35.0, size=(36, 2))
    widths = rng.uniform(0.4, 2.8, size=(36, 2))
    amplitudes = rng.uniform(0.2, 1.0, size=36)
    value = np.full(np.broadcast(x, y).shape, 1.0, dtype=np.float64)
    for (cx, cy), (wx, wy), amplitude in zip(centres, widths, amplitudes):
        value += amplitude * np.exp(-((x - cx) ** 2 / wx + (y - cy) ** 2 / wy))
    return value


def _reading(frame: str, axis: int, b: float, distance: float, side: float,
             origin: tuple[int, int], local_residues: tuple[float, float],
             *, samples: int = 128, extent: float = 4.0,
             focal: float = 20.0,
             principal_point: tuple[float, float] = (512.0, 512.0),
             centre_offset: tuple[float, float] = (0.0, 0.0),
             period_scale: float = 1.0):
    pixel_step = (2.0 * extent) / (samples - 1)
    period = b / (distance * pixel_step) * period_scale
    centre = tuple(extent + value / distance + centre_offset[index]
                   for index, value in enumerate(origin))
    denominator = __import__("fractions").Fraction(b).limit_denominator(16).denominator
    phases = []
    for local_axis, residue in enumerate(local_residues):
        leading = distance * (centre[local_axis] - extent)
        phase_fraction = ((residue - leading) % (1.0 / denominator)) / b
        phases.append(phase_fraction * period)
    axes = np.zeros((3, 3), dtype=np.float64)
    axes[axis, 2] = side
    for camera_axis, local_label in enumerate(i for i in range(3) if i != axis):
        axes[local_label, camera_axis] = 1.0
    rect = lattice_planes.PlaneRectification(
        axis_matrix=tuple(tuple(float(x) for x in row) for row in axes),
        focal=focal, principal_point=principal_point, centre=centre)
    phase = tuple(phases)
    quality = lattice_planes.PeriodicEnergyPhaseEvidence(
        phase_uv_pixels=phase, period_pixels=period, patch_centre_uv=centre,
        extent=extent, samples=samples, response_magnitude_uv=(8.0, 7.0),
        profile_mass_uv=(16.0, 14.0), concentration_uv=(0.5, 0.5),
        unavailable_reason_uv=(None, None))
    reading = lattice_planes.PlaneReading(
        frame=frame, axis=axis, side=side,
        alternatives=(lattice_planes.DistanceAlternative(b, distance, 1.0),),
        period_pixels=period, phase_pixels=phase, extent=extent, samples=samples,
        polygon=None, coverage_cells=1600, structural_share=1.0, confidence=1.0,
        rectification=rect, phase_evidence=quality)
    return reading


def _signature(reading, b: float, basis: np.ndarray,
               translation: np.ndarray) -> lattice_planes.PlaneSignature:
    rng = np.random.default_rng(991 if reading.frame.endswith("_left") else 992)
    image = rng.random((1024, 1024, 3), dtype=np.float32) + 0.1
    measured = lattice_planes.plane_signature(image, reading, b, max_cells=32)
    assert np.asarray(measured.interior).std() > 0.0
    assert np.asarray(measured.coverage).any()
    rows, columns = np.indices(measured.cells.shape, dtype=np.float64)
    local = np.stack((measured.origin_uv[0] + columns,
                      measured.origin_uv[1] + rows), axis=0).reshape(2, -1)
    world = basis @ local + translation.reshape(2, 1)
    sampled = _world_texture(world[0], world[1]).reshape(measured.cells.shape)
    cells = np.asarray(measured.cells).copy()
    valid = np.asarray(measured.coverage, dtype=bool)
    cells[valid] = sampled[valid]
    return replace(measured, cells=cells)


SUPPORT_ISLAND_SIDE = 12


def _centred_coverage(shape, side: int = SUPPORT_ISLAND_SIDE) -> np.ndarray:
    """Centred square support island with measurable sidelobe evidence.

    ``match_plane_pair`` measures the peak-to-sidelobe ratio outside a 3x3
    peak-exclusion window, and every admissible shift must overlap at least
    ``plane_graph.MIN_OVERLAP_CELLS`` (9) support cells.  A centred 3x3 or 4x4
    island leaves the peak as the only admissible shift, so the matcher
    correctly refuses it as ``repetition_unresolved``; the synthetic fixtures
    therefore sample a wider footprint so the intended shift keeps a genuine
    sidelobe sample.  The island is centred on the signature grid and may not
    exceed it.
    """
    if side < 5:
        raise ValueError("support island leaves no admissible sidelobe")
    if side > min(shape):
        raise ValueError("support island exceeds the signature grid")
    coverage = np.zeros(shape, dtype=bool)
    row = (shape[0] - side) // 2
    column = (shape[1] - side) // 2
    coverage[row:row + side, column:column + side] = True
    return coverage


def _canonical_identity(readings, endpoint):
    ordered = tuple(sorted((reading for reading in readings
                            if reading.frame == endpoint.frame),
                           key=lattice_planes.reading_order_key))
    index = next(index for index, reading in enumerate(ordered)
                 if reading is endpoint)
    return endpoint.frame, index


def _finite_compatible_map(homography: np.ndarray, physical_permutation: np.ndarray,
                           left_origin, right_origin, tolerance: float,
                           binding: dict) -> str | None:
    """Typed finite predicate including P1-F/C1/P1-D digest replay."""
    if (type(tolerance) not in (int, float) or not np.isfinite(tolerance)
            or tolerance < 0.0):
        return "invalid_tolerance"
    value = np.asarray(homography, dtype=np.float64)
    permutation = np.asarray(physical_permutation, dtype=np.float64)
    if value.shape != (3, 3) or permutation.shape != (2, 2):
        return "homography_shape_invalid"
    if not np.isfinite(value).all() or not np.isfinite(permutation).all():
        return "homography_nonfinite"
    if not np.isfinite(value[2, 2]) or value[2, 2] == 0.0:
        return "homography_denominator_invalid"
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        normalized = value / value[2, 2]
    if not np.isfinite(normalized).all():
        return "homography_denominator_unstable"
    if not np.allclose(normalized[2], (0.0, 0.0, 1.0),
                       rtol=0.0, atol=tolerance):
        return "homography_projective"
    if (not np.isin(permutation, (-1.0, 0.0, 1.0)).all()
            or not np.array_equal(np.count_nonzero(permutation, axis=0), (1, 1))
            or not np.array_equal(np.count_nonzero(permutation, axis=1), (1, 1))
            or not np.allclose(normalized[:2, :2], permutation,
                               rtol=0.0, atol=tolerance)):
        return "grid_basis_incompatible"
    expected_translation = (permutation @ np.asarray(right_origin, dtype=np.float64)
                            - np.asarray(left_origin, dtype=np.float64))
    if not np.allclose(normalized[:2, 2], expected_translation,
                       rtol=0.0, atol=tolerance):
        return "grid_origin_incompatible"
    required_binding = {
        "mapping_version", "replayed_homography_sha256",
        "c1_geometry_digests", "recomputed_c1_geometry_digests",
        "c1_homography_digests", "p1d_geometry_digests",
        "recomputed_p1d_geometry_digests", "p1d_content_digests",
        "recomputed_p1d_content_digests", "p1d_homography_digest",
    }
    if not isinstance(binding, dict) or not required_binding <= binding.keys():
        return "compatibility_binding_missing"
    tuple_bindings = (
        "c1_geometry_digests", "recomputed_c1_geometry_digests",
        "c1_homography_digests", "p1d_geometry_digests",
        "recomputed_p1d_geometry_digests", "p1d_content_digests",
        "recomputed_p1d_content_digests",
    )
    if any(type(binding[key]) is not tuple for key in tuple_bindings):
        return "compatibility_binding_invalid"
    if (len(binding["c1_geometry_digests"]) != 2
            or len(binding["recomputed_c1_geometry_digests"]) != 2
            or len(binding["p1d_geometry_digests"]) != 2
            or len(binding["recomputed_p1d_geometry_digests"]) != 2
            or len(binding["p1d_content_digests"]) != 2
            or len(binding["recomputed_p1d_content_digests"]) != 2
            or not binding["c1_homography_digests"]):
        return "compatibility_binding_invalid"
    if binding["mapping_version"] != "block-cell-grid-v2":
        return "homography_version_unsupported"
    replayed = binding["replayed_homography_sha256"]
    if (not isinstance(replayed, str) or len(replayed) != 64
            or any(character not in "0123456789abcdef" for character in replayed)):
        return "p1f_homography_digest_invalid"
    if binding["c1_geometry_digests"] != binding["recomputed_c1_geometry_digests"]:
        return "c1_geometry_digest_mismatch"
    if (not binding["c1_homography_digests"]
            or any(digest != replayed
                   for digest in binding["c1_homography_digests"])):
        return "c1_homography_digest_mismatch"
    if binding["p1d_geometry_digests"] != binding["recomputed_p1d_geometry_digests"]:
        return "p1d_geometry_digest_mismatch"
    if binding["p1d_content_digests"] != binding["recomputed_p1d_content_digests"]:
        return "p1d_content_digest_mismatch"
    if binding["p1d_homography_digest"] != replayed:
        return "p1d_homography_digest_mismatch"
    return None


def _forward_physical_carry(left_basis: np.ndarray, right_basis: np.ndarray,
                            left_translation: np.ndarray,
                            right_translation: np.ndarray,
                            left_origin: tuple[int, int],
                            right_origin: tuple[int, int],
                            left_residues: tuple[float, float],
                            right_residues: tuple[float, float]) -> dict:
    """Derive integer carries from endpoint cell maps and lattice anchors only."""
    left_origin_array = np.asarray(left_origin, dtype=np.float64)
    right_origin_array = np.asarray(right_origin, dtype=np.float64)
    fixed_right_cell = np.asarray((11.0, -7.0), dtype=np.float64)
    right_world_point = (right_basis @ (right_origin_array + fixed_right_cell)
                         + right_translation)
    mapped_left_cell = (left_basis.T @ (right_world_point - left_translation)
                        - left_origin_array)
    physical_map = left_basis.T @ right_basis
    grid_translation = physical_map @ right_origin_array - left_origin_array
    expected_left_cell = (
        physical_map @ fixed_right_cell + grid_translation
        + left_basis.T @ (right_translation - left_translation))
    np.testing.assert_allclose(mapped_left_cell, expected_left_cell,
                               rtol=0.0, atol=1e-12)

    # Pull one fixed common integer-world lattice point into each endpoint.
    # Its fractional cell coordinates are the independently forward-generated
    # phase residues; its integer cell coordinates determine the carry.
    world_lattice_point = np.asarray((0.0, 0.0), dtype=np.float64)
    left_lattice_cell = left_basis.T @ (world_lattice_point - left_translation)
    right_lattice_cell = right_basis.T @ (world_lattice_point - right_translation)
    left_integer = np.floor(left_lattice_cell).astype(np.int64)
    right_integer = np.floor(right_lattice_cell).astype(np.int64)
    left_fraction = left_lattice_cell - left_integer
    right_fraction = right_lattice_cell - right_integer
    np.testing.assert_allclose(left_fraction, left_residues,
                               rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(right_fraction, right_residues,
                               rtol=0.0, atol=1e-12)

    right_local_carries: list[int | None] = [None, None]
    raw_world_carries = []
    signed_world_carries = []
    local_correspondence = []
    for world_axis in range(2):
        left_local = int(np.flatnonzero(
            np.abs(left_basis[world_axis]) > 0.5)[0])
        right_local = int(np.flatnonzero(
            np.abs(right_basis[world_axis]) > 0.5)[0])
        left_sign = int(round(float(left_basis[world_axis, left_local])))
        right_sign = int(round(float(right_basis[world_axis, right_local])))
        sign_between_labels = left_sign * right_sign
        carry = int(sign_between_labels * left_integer[left_local]
                    - right_integer[right_local])
        right_local_carries[right_local] = carry
        raw_world_carries.append(carry)
        signed_world_carries.append(right_sign * carry)
        local_correspondence.append(
            (world_axis, left_local, right_local,
             left_sign, right_sign, int(left_integer[left_local]),
             int(right_integer[right_local])))
    assert all(type(value) is int for value in right_local_carries)
    return {
        "entry_rc": (right_local_carries[1], right_local_carries[0]),
        "raw_carries_uv": tuple(right_local_carries),
        "raw_carries_by_world": tuple(raw_world_carries),
        "signed_world_carries": tuple(signed_world_carries),
        "left_lattice_cell": tuple(float(value) for value in left_lattice_cell),
        "right_lattice_cell": tuple(float(value) for value in right_lattice_cell),
        "left_integer_lattice_cell": tuple(int(value) for value in left_integer),
        "right_integer_lattice_cell": tuple(int(value) for value in right_integer),
        "left_fractional_lattice_cell": tuple(float(value) for value in left_fraction),
        "right_fractional_lattice_cell": tuple(float(value) for value in right_fraction),
        "fixed_right_cell": tuple(float(value) for value in fixed_right_cell),
        "right_world_point": tuple(float(value) for value in right_world_point),
        "mapped_left_cell": tuple(float(value) for value in mapped_left_cell),
        "local_correspondence": tuple(local_correspondence),
    }


def _direct_normalized_census(left: lattice_planes.PlaneSignature,
                              right: lattice_planes.PlaneSignature,
                              left_basis: np.ndarray, right_basis: np.ndarray,
                              radius: int) -> dict[tuple[int, int], tuple[float, int]]:
    """Independent physical-map warp and bounded overlap-NCC enumeration."""
    p = left_basis.T @ right_basis
    left_origin = np.asarray(left.origin_uv, dtype=np.float64)
    right_origin = np.asarray(right.origin_uv, dtype=np.float64)
    rows, columns = right.cells.shape
    first = np.zeros((rows, columns), dtype=np.float64)
    valid_first = np.zeros((rows, columns), dtype=bool)
    for row in range(rows):
        for column in range(columns):
            right_point = np.asarray((column, row), dtype=np.float64) + right_origin
            left_point = p @ right_point - left_origin
            left_column, left_row = np.rint(left_point).astype(int)
            if (0 <= left_row < left.cells.shape[0]
                    and 0 <= left_column < left.cells.shape[1]
                    and left.coverage[left_row, left_column]
                    and right.coverage[row, column]):
                first[row, column] = left.cells[left_row, left_column]
                valid_first[row, column] = True
    scores = {}
    second = np.asarray(right.cells, dtype=np.float64)
    valid_second = np.asarray(right.coverage, dtype=bool)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            rr0, rr1 = max(0, -dr), min(rows, rows - dr)
            cc0, cc1 = max(0, -dc), min(columns, columns - dc)
            a = first[rr0 + dr:rr1 + dr, cc0 + dc:cc1 + dc]
            b = second[rr0:rr1, cc0:cc1]
            mask = (valid_first[rr0 + dr:rr1 + dr, cc0 + dc:cc1 + dc]
                    & valid_second[rr0:rr1, cc0:cc1])
            av, bv = a[mask], b[mask]
            if av.size < 9:
                continue
            av = av - av.mean()
            bv = bv - bv.mean()
            denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
            scores[(dr, dc)] = (float(np.dot(av, bv) / denom) if denom else -np.inf,
                                 int(av.size))
    return scores


def _direct_normalized_peak(left: lattice_planes.PlaneSignature,
                            right: lattice_planes.PlaneSignature,
                            left_basis: np.ndarray, right_basis: np.ndarray,
                            radius: int) -> tuple[int, int]:
    scores = _direct_normalized_census(left, right, left_basis, right_basis, radius)
    return max(scores, key=lambda shift: scores[shift][0])


def test_first_physical_smoke_uses_graph_identity_through_p1d():
    """The declared world translation is recovered without patched indices."""
    left_translation = np.asarray((0.2, -0.35), dtype=np.float64)
    right_translation = np.asarray((1.6, 0.45), dtype=np.float64)
    truth = right_translation - left_translation
    left_origin, right_origin = (3, -2), (-4, 5)
    left_basis = np.eye(2, dtype=np.float64)
    right_basis = np.eye(2, dtype=np.float64)
    left_residues = tuple(float((-value) % 1.0)
                          for value in left_basis.T @ left_translation)
    right_residues = tuple(float((-value) % 1.0)
                           for value in right_basis.T @ right_translation)

    left = _reading("a_left", 2, 1.0, 2.0, 1.0, left_origin, left_residues)
    right = _reading("b_right", 2, 1.0, 3.0, 1.0, right_origin, right_residues)
    signatureless = _reading("a_left", 0, 1.0, 4.0, 1.0,
                             (0, 0), (0.21, 0.37))
    left_signature = _signature(left, 1.0, left_basis, left_translation)
    right_signature = _signature(right, 1.0, right_basis, right_translation)
    readings = (right, left, signatureless)
    signatures = (right_signature, left_signature)
    left_identity = _canonical_identity(readings, left)
    right_identity = _canonical_identity(readings, right)

    matches = plane_graph.match_plane_graph(
        readings, signatures, max_shift_cells=4)
    match = next((value for value in matches
                  if (value.left, value.right, value.quadrant_of_right)
                  == ("a_left", "b_right", 0)), None)
    assert match is not None
    assert (match.left_plane, match.right_plane) == (
        left_identity[1], right_identity[1]) == (1, 0)
    assert match.cell_shift_rc is not None
    read_map = {left_identity: left, right_identity: right}
    evidence = plane_residue.hypothesis_phase_shift_evidence(match, read_map)
    assert evidence.refusal is None
    assert len(evidence.phase_candidates) == 4
    by_axis = {(record.frame, record.local_axis): record.residue_candidates
               for record in evidence.phase_candidates}
    assert all(any(abs(value - truth_value) <= 1e-12 for value in by_axis[(frame, axis)])
               for frame, values in (("a_left", left_residues),
                                     ("b_right", right_residues))
               for axis, truth_value in zip(("u", "v"), values))

    direct_ncc_scores = _direct_normalized_census(
        left_signature, right_signature, left_basis, right_basis, 4)
    independently_expected_shift = max(
        direct_ncc_scores, key=lambda shift: direct_ncc_scores[shift][0])
    forward_carry = _forward_physical_carry(
        left_basis, right_basis, left_translation, right_translation,
        left_origin, right_origin, left_residues, right_residues)
    assert forward_carry["entry_rc"] == (1, 1)
    physical_ncc_value, physical_ncc_support = direct_ncc_scores[
        forward_carry["entry_rc"]]
    assert physical_ncc_support >= 9 and np.isfinite(physical_ncc_value)
    # The independent overlap-NCC maximum and the matcher’s whitened phase
    # peak are separate observables here. This fixture intentionally retains
    # the lower-ranked (1,1) carry while P1-B supplies a fractional truth lift.
    assert match.cell_shift_rc == (0, 1)
    assert independently_expected_shift == (1, 1)

    solution = plane_graph.ComponentSolution(
        frames=("a_left", "b_right"),
        quadrants={"a_left": 0, "b_right": 0},
        translations={"a_left": (0.0, 0.0, 0.0),
                     "b_right": (0.0, 0.0, 0.0)},
        axis_residuals={0: 0.0, 1: 0.0, 2: 0.0},
        axis_resultants={0: 1.0, 1: 1.0, 2: 1.0}, scale_known=True,
        blocks_per_period={left_identity: 1.0, right_identity: 1.0},
        reason="", alternatives=())
    support = c1.build_axis_support(
        solution, readings, (match,), signatures)
    assert support.refusal is None
    result = p1d.build_shift_domains(
        support, (match,), signatures, readings=readings,
        max_shift_cells=4,
        budget=p1d.ShiftDomainBudget(
            max_domains=1, max_padded_fft_cells=100_000,
            max_shift_positions=100_000, max_overlap_cell_visits=10_000_000,
            max_buffered_entries=100_000))
    assert result.status == "complete", result.refusals
    assert len(result.domains) == 1
    domain = result.domains[0]
    domain_entry_rcs = {(entry.row_shift, entry.column_shift)
                        for entry in domain.entries}
    assert forward_carry["entry_rc"] in domain_entry_rcs
    assert (1, 1) in domain_entry_rcs
    hit_tuples = []
    for entry in domain.entries:
        axis_candidates = []
        for local_axis, shift in (("u", entry.column_shift),
                                  ("v", entry.row_shift)):
            axis_candidates.append(tuple(
                (shift + left_value - right_value, left_index, right_index,
                 left_value, right_value)
                for left_index, left_value in enumerate(by_axis[("a_left", local_axis)])
                for right_index, right_value in enumerate(by_axis[("b_right", local_axis)])))
        for u_value in axis_candidates[0]:
            for v_value in axis_candidates[1]:
                vector = (u_value[0], v_value[0])
                if np.allclose(vector, truth, rtol=0.0, atol=1e-12):
                    hit_tuples.append((entry.row_shift, entry.column_shift,
                                       u_value[1:], v_value[1:]))
    assert hit_tuples, {"truth": truth.tolist(), "peak": match.cell_shift_rc}
    assert any((row, column) == forward_carry["entry_rc"]
               for row, column, _u, _v in hit_tuples)


def _evaluate_physical_case(left_normal: int, left_quadrant: int,
                            relative_quadrant: int, *,
                            left_b: float = 1.0, right_b: float = 1.0,
                            left_side: float = 1.0, right_side: float = -1.0,
                            left_distance: float = 2.0,
                            right_distance: float = 3.0,
                            left_origin: tuple[int, int] = (3, -2),
                            right_origin: tuple[int, int] = (-4, 5),
                            left_translation: tuple[float, float] = (0.2, -0.35),
                            right_translation: tuple[float, float] = (1.6, 0.45),
                            left_samples: int = 128,
                            right_samples: int = 128,
                            left_geometry: dict | None = None,
                            right_geometry: dict | None = None,
                            radius: int = 4):
    """Run one independently declared endpoint pair through canonical P1-D."""
    right_quadrant = (left_quadrant + relative_quadrant) % 4
    left_rotation, right_rotation = (_physical_yaw(left_quadrant),
                                     _physical_yaw(right_quadrant))
    relative_normal = _physical_yaw(relative_quadrant)[:, left_normal]
    right_normal = int(np.flatnonzero(np.abs(relative_normal) > 0.5)[0])
    left_world_normal = left_rotation[:, left_normal]
    right_world_normal = right_rotation[:, right_normal]
    assert abs(float(left_world_normal @ right_world_normal)) == 1.0
    world_normal = int(np.flatnonzero(np.abs(left_world_normal) > 0.5)[0])
    world_tangents = tuple(axis for axis in range(3) if axis != world_normal)
    left_labels = tuple(axis for axis in range(3) if axis != left_normal)
    right_labels = tuple(axis for axis in range(3) if axis != right_normal)
    left_basis = left_rotation[np.ix_(world_tangents, left_labels)]
    right_basis = right_rotation[np.ix_(world_tangents, right_labels)]
    physical_map = left_basis.T @ right_basis
    assert np.isin(physical_map, (-1.0, 0.0, 1.0)).all()
    assert np.array_equal(np.count_nonzero(physical_map, axis=0), (1, 1))
    assert np.array_equal(np.count_nonzero(physical_map, axis=1), (1, 1))
    truth = np.asarray(right_translation, dtype=np.float64) - np.asarray(
        left_translation, dtype=np.float64)

    for quadrant, rotation in ((left_quadrant, left_rotation),
                               (right_quadrant, right_rotation)):
        for local_label in range(3):
            column = rotation[:, local_label]
            world_axis = int(np.flatnonzero(np.abs(column) > 0.5)[0])
            world_sign = int(round(float(column[world_axis])))
            assert plane_graph.axis_transform(local_label, quadrant) == (
                world_axis, world_sign)

    left_local_position = left_basis.T @ np.asarray(left_translation, dtype=np.float64)
    right_local_position = right_basis.T @ np.asarray(right_translation, dtype=np.float64)
    left_residues = tuple(float((-value) % 1.0) for value in left_local_position)
    right_residues = tuple(float((-value) % 1.0) for value in right_local_position)
    frame_left, frame_right = "a_left", "b_right"
    left_config = {"samples": left_samples, **(left_geometry or {})}
    right_config = {"samples": right_samples, **(right_geometry or {})}
    left = _reading(frame_left, left_normal, left_b, left_distance,
                    left_side, left_origin, left_residues, **left_config)
    right = _reading(frame_right, right_normal, right_b, right_distance,
                     right_side, right_origin, right_residues, **right_config)
    # A leading signatureless reading proves that matching and P1-D share the
    # canonical complete universe rather than a signatures-only index space.
    dummy = _reading(
        frame_left, left_normal, left_b, left_distance * 2.0, left_side,
        (0, 0), (0.21, 0.37), **left_config)
    sig_left = _signature(left, left_b, left_basis,
                          np.asarray(left_translation, dtype=np.float64))
    sig_right = _signature(right, right_b, right_basis,
                           np.asarray(right_translation, dtype=np.float64))
    readings = (right, dummy, left)
    signatures = (sig_right, sig_left)
    left_identity = _canonical_identity(readings, left)
    right_identity = _canonical_identity(readings, right)
    graph_refusals = []
    graph_matches = plane_graph.match_plane_graph(
        readings, signatures, max_shift_cells=radius, refusals=graph_refusals)
    selected = next((match for match in graph_matches
                     if (match.left == frame_left and match.right == frame_right
                         and match.quadrant_of_right == relative_quadrant)), None)
    assert selected is not None, {
        "normal": left_normal, "left_q": left_quadrant,
        "relative_q": relative_quadrant,
        "right_normal": right_normal,
        "refusals": tuple((value.quadrant_of_right, value.reason, value.detail)
                          for value in graph_refusals),
        "matches": tuple((value.quadrant_of_right, value.cell_shift_rc,
                           value.peak, value.psr) for value in graph_matches),
    }
    assert (selected.left_plane, selected.right_plane) == (
        left_identity[1], right_identity[1]) == (1, 0)
    assert selected.left_axis == left_normal and selected.right_axis == right_normal
    assert selected.left_b == left_b and selected.right_b == right_b
    assert sig_left.reading is left and sig_right.reading is right
    assert sig_left.blocks_per_period == left_b
    assert sig_right.blocks_per_period == right_b
    read_map = {left_identity: left, right_identity: right}
    phase = plane_residue.hypothesis_phase_shift_evidence(selected, read_map)
    assert phase.refusal is None
    phase_by_key = {(record.frame, record.local_axis): record.residue_candidates
                    for record in phase.phase_candidates}
    for frame, expected_residues in ((frame_left, left_residues),
                                     (frame_right, right_residues)):
        for local_name, expected in zip(("u", "v"), expected_residues):
            assert any(abs(candidate - expected) <= 1e-12
                       for candidate in phase_by_key[(frame, local_name)])

    # Derive the physical entry only from forward endpoint cell coordinates:
    # map one fixed right cell through world space, invert it at the left, then
    # decompose a fixed common integer lattice point into endpoint integer and
    # fractional cell indices. No candidate delta equation participates here.
    forward_carry = _forward_physical_carry(
        left_basis, right_basis,
        np.asarray(left_translation, dtype=np.float64),
        np.asarray(right_translation, dtype=np.float64),
        left_origin, right_origin, left_residues, right_residues)
    physical_raw_carries = forward_carry["raw_carries_by_world"]
    physical_signed_carries = forward_carry["signed_world_carries"]
    physical_entry_rc = forward_carry["entry_rc"]
    assert max(map(abs, physical_entry_rc)) <= radius

    # Directly evaluate a bounded overlap-normalized correlation over the
    # physical absolute bases; keep this diagnostic distinct from the graph's
    # whitened phase-correlation peak.
    direct_ncc_scores = _direct_normalized_census(
        sig_left, sig_right, left_basis, right_basis, radius)
    direct_ncc_peak = max(
        direct_ncc_scores, key=lambda shift: direct_ncc_scores[shift][0])
    direct_ncc_peak_value, direct_ncc_peak_support = direct_ncc_scores[direct_ncc_peak]
    assert max(map(abs, direct_ncc_peak)) <= radius
    assert selected.cell_shift_rc is not None

    left_index = left_identity[1]
    right_index = right_identity[1]
    solution = plane_graph.ComponentSolution(
        frames=(frame_left, frame_right),
        quadrants={frame_left: left_quadrant, frame_right: right_quadrant},
        translations={frame_left: (0.0, 0.0, 0.0),
                      frame_right: (0.0, 0.0, 0.0)},
        axis_residuals={0: 0.0, 1: 0.0, 2: 0.0},
        axis_resultants={0: 1.0, 1: 1.0, 2: 1.0}, scale_known=True,
        blocks_per_period={left_identity: left_b, right_identity: right_b},
        reason="", alternatives=())
    support = c1.build_axis_support(
        solution, readings, (selected,), signatures)
    assert support.refusal is None, support.refusal
    domains = p1d.build_shift_domains(
        support, (selected,), signatures, readings=readings,
        max_shift_cells=radius,
        budget=p1d.ShiftDomainBudget(
            max_domains=1, max_padded_fft_cells=1_000_000,
            max_shift_positions=1_000_000, max_overlap_cell_visits=100_000_000,
            max_buffered_entries=1_000_000))
    assert domains.status == "complete", domains.refusals
    assert len(domains.domains) == 1
    domain = domains.domains[0]
    assert p1d.validate_shift_domain_inputs(
        domains, support, (selected,), signatures,
        max_shift_cells=radius, readings=readings)
    assert all(entry.overlap_cells >= domain.min_overlap_cells
               for entry in domain.entries)

    hits = []
    candidate_records_hash = hashlib.sha256()
    candidate_records = []
    candidate_vector_count = 0
    for entry in domain.entries:
        axis_options = []
        for world_index in range(2):
            left_local = int(np.flatnonzero(np.abs(left_basis[world_index]) > 0.5)[0])
            right_local = int(np.flatnonzero(np.abs(right_basis[world_index]) > 0.5)[0])
            left_sign = int(round(float(left_basis[world_index, left_local])))
            right_sign = int(round(float(right_basis[world_index, right_local])))
            right_shift = (entry.column_shift if right_local == 0
                           else entry.row_shift)
            axis_name_left = "u" if left_local == 0 else "v"
            axis_name_right = "u" if right_local == 0 else "v"
            options = tuple(
                (right_sign * right_shift + left_sign * left_r
                 - right_sign * right_r,
                 left_residue_index, right_residue_index, left_r, right_r,
                 right_shift, left_sign, right_sign)
                for left_residue_index, left_r in enumerate(
                    phase_by_key[(frame_left, axis_name_left)])
                for right_residue_index, right_r in enumerate(
                    phase_by_key[(frame_right, axis_name_right)]))
            axis_options.append(options)
        first = np.asarray(axis_options[0], dtype=np.float64)
        second = np.asarray(axis_options[1], dtype=np.float64)
        first_count, second_count = len(first), len(second)
        pair_count = first_count * second_count
        candidate_vector_count += pair_count
        first_repeat = lambda column: np.repeat(first[:, column], second_count)
        second_tile = lambda column: np.tile(second[:, column], first_count)
        exhaustive_records = np.column_stack((
            np.full(pair_count, entry.row_shift),
            np.full(pair_count, entry.column_shift),
            first_repeat(0), second_tile(0),
            first_repeat(1), first_repeat(2),
            second_tile(1), second_tile(2),
            first_repeat(3), first_repeat(4),
            second_tile(3), second_tile(4),
            first_repeat(5), second_tile(5),
            first_repeat(6), second_tile(6),
            first_repeat(7) * first_repeat(5),
            second_tile(7) * second_tile(5),
            first_repeat(7), second_tile(7),
        )).astype("<f8", copy=False)
        candidate_records_hash.update(exhaustive_records.tobytes(order="C"))
        for row_record in exhaustive_records:
            candidate_records.append({
                "entry_rc": (int(row_record[0]), int(row_record[1])),
                "entry_overlap_cells": entry.overlap_cells,
                "world_deltas": (float(row_record[2]), float(row_record[3])),
                "residue_indices": (
                    (int(row_record[4]), int(row_record[5])),
                    (int(row_record[6]), int(row_record[7]))),
                "residue_values": (
                    (float(row_record[8]), float(row_record[9])),
                    (float(row_record[10]), float(row_record[11]))),
                "raw_carries": (int(row_record[12]), int(row_record[13])),
                "left_world_signs": (int(row_record[14]), int(row_record[15])),
                "signed_carries": (int(row_record[16]), int(row_record[17])),
                "right_world_signs": (int(row_record[18]), int(row_record[19])),
            })
        truth_mask = ((np.abs(first[:, 0, None] - truth[0]) <= 1e-12)
                      & (np.abs(second[None, :, 0] - truth[1]) <= 1e-12))
        for first_index, second_index in np.argwhere(truth_mask):
            a, b = first[first_index], second[second_index]
            hit = {
                "entry_rc": (entry.row_shift, entry.column_shift),
                "deltas": (float(a[0]), float(b[0])),
                "residue_indices": ((int(a[1]), int(a[2])),
                                     (int(b[1]), int(b[2]))),
                "residue_values": ((float(a[3]), float(a[4])),
                                   (float(b[3]), float(b[4]))),
                "raw_carries": (int(a[5]), int(b[5])),
                "signed_carries": (int(a[7] * a[5]), int(b[7] * b[5])),
                "signed_carry_signs": (int(a[7]), int(b[7])),
            }
            ncc_value, ncc_support = direct_ncc_scores.get(
                hit["entry_rc"], (float("nan"), 0))
            hit["raw_ncc_at_entry"] = {
                "value": ncc_value, "support": ncc_support,
                "is_ncc_optimum": hit["entry_rc"] == direct_ncc_peak,
            }
            hits.append(hit)
    assert hits, {
        "normal": left_normal, "left_q": left_quadrant,
        "relative_q": relative_quadrant, "truth": truth.tolist(),
        "peak": selected.cell_shift_rc, "direct_ncc": direct_ncc_peak,
        "domain_entries": len(domain.entries),
        "candidate_vector_count": candidate_vector_count,
        "candidate_records_sha256": candidate_records_hash.hexdigest(),
    }
    assert candidate_vector_count == len(candidate_records)
    all_record_keys = {
        (record["entry_rc"], record["residue_indices"],
         record["world_deltas"], record["residue_values"])
        for record in candidate_records}
    assert len(all_record_keys) == candidate_vector_count
    truth_hit_keys = {
        (hit["entry_rc"], hit["residue_indices"],
         hit["deltas"], hit["residue_values"])
        for hit in hits}
    assert truth_hit_keys <= all_record_keys
    exact_entry_hits = tuple(hit for hit in hits
                             if hit["entry_rc"] == physical_entry_rc)
    assert exact_entry_hits, {
        "physical_entry_rc": physical_entry_rc,
        "truth_hits": tuple(hit["entry_rc"] for hit in hits),
    }
    assert all(hit["raw_ncc_at_entry"]["support"] >= 9
               and np.isfinite(hit["raw_ncc_at_entry"]["value"])
               for hit in exact_entry_hits)
    assert all(hit["raw_carries"] == tuple(physical_raw_carries)
               and hit["signed_carries"] == tuple(physical_signed_carries)
               for hit in exact_entry_hits)
    exact_truth_provenance = tuple({
        "entry_rc": hit["entry_rc"],
        "residue_indices": hit["residue_indices"],
        "residue_values": hit["residue_values"],
        "raw_carries": hit["raw_carries"],
        "signed_carries": hit["signed_carries"],
    } for hit in hits)
    return {
        "truth": tuple(float(value) for value in truth),
        "peak": selected.cell_shift_rc,
        "direct_ncc_peak": direct_ncc_peak,
        "direct_ncc_optimum": {
            "entry_rc": direct_ncc_peak,
            "value": direct_ncc_peak_value,
            "support": direct_ncc_peak_support,
        },
        "domain_entries": len(domain.entries),
        "hit_tuples": tuple(hits),
        "physical_map": physical_map.copy(),
        "left_basis": left_basis.copy(),
        "right_basis": right_basis.copy(),
        "left_residues": left_residues,
        "right_residues": right_residues,
        "endpoint_indices": (selected.left_plane, selected.right_plane),
        "endpoint_identities": (left_identity, right_identity),
        "selected_match": selected,
        "support": support,
        "domains": domains,
        "readings": readings,
        "signatures": signatures,
        "left_signature": sig_left,
        "right_signature": sig_right,
        "grid_homography": plane_graph.cell_grid_homography(
            sig_left, sig_right, relative_quadrant),
        "domain_record": domain.to_record(),
        "domain": domain,
        "phase_residues": {key: tuple(value) for key, value in phase_by_key.items()},
        "side_pair": (left_side, right_side),
        "b_pair": (left_b, right_b),
        "origin_pair": (left_origin, right_origin),
        "raw_carry_magnitude": max(
            abs(item) for hit in hits for item in hit["raw_carries"]),
        "physical_entry_rc": physical_entry_rc,
        "physical_raw_carries": tuple(physical_raw_carries),
        "physical_signed_carries": tuple(physical_signed_carries),
        "forward_carry": forward_carry,
        "raw_ncc_at_physical_carry": {
            "value": direct_ncc_scores[physical_entry_rc][0],
            "support": direct_ncc_scores[physical_entry_rc][1],
        },
        "truth_supporting_tuple_count": len(exact_truth_provenance),
        "truth_supporting_entry_count": len({
            item["entry_rc"] for item in exact_truth_provenance}),
        "truth_supporting_provenance": exact_truth_provenance,
        "physical_entry_truth_tuple_count": len(exact_entry_hits),
        "physical_entry_is_ambiguous": len(exact_entry_hits) > 1,
        "candidate_vector_count": candidate_vector_count,
        "candidate_records_sha256": candidate_records_hash.hexdigest(),
        "candidate_records": tuple(candidate_records),
    }


@pytest.mark.parametrize("left_normal", range(3))
@pytest.mark.parametrize("left_quadrant", range(4))
@pytest.mark.parametrize("relative_quadrant", range(4))
def test_all_48_physical_orientation_cases_retain_joint_truth(
        left_normal: int, left_quadrant: int, relative_quadrant: int):
    case_number = left_normal * 16 + left_quadrant * 4 + relative_quadrant
    left_side = 1.0 if case_number % 2 == 0 else -1.0
    right_side = left_side if case_number % 3 == 0 else -left_side
    right_quadrant = (left_quadrant + relative_quadrant) % 4
    left_rotation, right_rotation = (_physical_yaw(left_quadrant),
                                     _physical_yaw(right_quadrant))
    right_normal_vector = _physical_yaw(relative_quadrant)[:, left_normal]
    right_normal = int(np.flatnonzero(np.abs(right_normal_vector) > 0.5)[0])
    common_normal = int(np.flatnonzero(
        np.abs(left_rotation[:, left_normal]) > 0.5)[0])
    world_tangents = tuple(axis for axis in range(3) if axis != common_normal)
    left_labels = tuple(axis for axis in range(3) if axis != left_normal)
    right_labels = tuple(axis for axis in range(3) if axis != right_normal)
    basis_l = left_rotation[np.ix_(world_tangents, left_labels)]
    basis_r = right_rotation[np.ix_(world_tangents, right_labels)]
    left_origin = np.asarray((7, -5), dtype=np.float64)
    # Equal-size block windows share their independently declared physical
    # centre; reflections therefore remain overlapping without hand-tuning H.
    half_index = 7.5  # extent=4, distance=2 -> a 16 by 16 signature.
    right_origin_float = (basis_r.T @ (basis_l @ (left_origin + half_index))
                          - half_index)
    assert np.allclose(right_origin_float, np.rint(right_origin_float),
                       rtol=0.0, atol=0.0)
    right_origin = tuple(int(value) for value in np.rint(right_origin_float))
    result = _evaluate_physical_case(
        left_normal, left_quadrant, relative_quadrant,
        left_side=left_side, right_side=right_side,
        left_distance=2.0, right_distance=2.0,
        left_origin=tuple(int(value) for value in left_origin),
        right_origin=right_origin)
    assert result["truth"] == pytest.approx((1.4, 0.8), abs=1e-15)
    assert result["endpoint_indices"] == (1, 0)
    assert result["domain_entries"] > 1
    assert result["hit_tuples"]
    assert result["truth_supporting_tuple_count"] >= 1
    assert result["truth_supporting_entry_count"] >= 1
    assert result["physical_entry_truth_tuple_count"] >= 1
    assert result["physical_entry_is_ambiguous"] == (
        result["physical_entry_truth_tuple_count"] > 1)
    assert result["raw_ncc_at_physical_carry"]["support"] >= 9
    assert np.isfinite(result["raw_ncc_at_physical_carry"]["value"])
    assert result["direct_ncc_optimum"]["support"] >= 9


_SUPPORTED_BLOCK_PERIODS = (1 / 16, 1 / 8, 1 / 4, 1 / 2,
                           1.0, 2.0, 3.0, 4.0, 8.0, 16.0)


@pytest.mark.parametrize("blocks_per_period", _SUPPORTED_BLOCK_PERIODS)
def test_equal_rational_block_period_pairs_keep_truth(
        blocks_per_period: float):
    result = _evaluate_physical_case(
        2, 0, 0, left_b=blocks_per_period, right_b=blocks_per_period,
        left_side=1.0, right_side=-1.0, left_distance=2.0,
        right_distance=2.0, left_origin=(7, -5), right_origin=(7, -5),
        left_samples=2048, right_samples=2048,
        radius=2)
    assert result["b_pair"] == (blocks_per_period, blocks_per_period)
    assert result["hit_tuples"]
    assert result["candidate_vector_count"] > 0
    assert len(result["candidate_records_sha256"]) == 64


@pytest.mark.parametrize("pair_index", range(len(_SUPPORTED_BLOCK_PERIODS)))
def test_unequal_rational_pairs_cover_every_supported_class(pair_index: int):
    left_b = _SUPPORTED_BLOCK_PERIODS[pair_index]
    right_b = _SUPPORTED_BLOCK_PERIODS[(pair_index + 1) % len(_SUPPORTED_BLOCK_PERIODS)]
    result = _evaluate_physical_case(
        2, 0, 0,
        left_b=left_b, right_b=right_b,
        left_side=1.0 if pair_index % 2 == 0 else -1.0,
        right_side=-1.0 if pair_index % 2 == 0 else 1.0,
        left_distance=2.0, right_distance=2.0,
        left_origin=(7, -5), right_origin=(7, -5),
        left_samples=2048, right_samples=2048, radius=2)
    assert result["b_pair"] == (left_b, right_b)
    assert left_b != right_b
    assert result["hit_tuples"]


def _support_geometry_pairs(result) -> tuple[tuple[str, str], ...]:
    return tuple((row.left_signature_geometry_sha256,
                  row.right_signature_geometry_sha256)
                 for row in result["support"].mapped_cell_shifts)


def test_measurement_geometry_changes_do_not_rescale_grid_map():
    base = _evaluate_physical_case(
        2, 0, 0, left_side=1.0, right_side=-1.0,
        left_distance=2.0, right_distance=2.0,
        left_origin=(7, -5), right_origin=(7, -5))
    base_h = base["grid_homography"]
    base_geometry = _support_geometry_pairs(base)
    variants = (
        {"left_geometry": {"focal": 24.0}},
        {"left_geometry": {"principal_point": (490.0, 520.0)}},
        {"left_geometry": {"centre_offset": (0.02, 0.03)}},
        {"left_samples": 256},
        {"left_side": -1.0},
        {"left_geometry": {"period_scale": 1.5}},
        {"left_distance": 2.25},
        {"left_geometry": {"extent": 5.0}},
        {"left_origin": (8, -4), "right_origin": (8, -4)},
    )
    for variant in variants:
        arguments = dict(
            left_side=1.0, right_side=-1.0,
            left_distance=2.0, right_distance=2.0,
            left_origin=(7, -5), right_origin=(7, -5))
        arguments.update(variant)
        changed = _evaluate_physical_case(2, 0, 0, **arguments)
        np.testing.assert_array_equal(changed["grid_homography"], base_h)
        assert changed["truth"] == base["truth"]
        assert changed["hit_tuples"]
        assert _support_geometry_pairs(changed) != base_geometry


def test_finite_compatibility_predicate_and_p1f_digest_replay():
    left_origin = np.asarray((7, -5), dtype=np.float64)
    left_basis, right_basis = np.eye(2), np.asarray(((0.0, 1.0), (1.0, 0.0)))
    right_origin = tuple(int(value) for value in np.rint(
        right_basis.T @ (left_basis @ (left_origin + 7.5)) - 7.5))
    result = _evaluate_physical_case(
        0, 0, 1, left_side=1.0, right_side=-1.0,
        left_origin=tuple(int(value) for value in left_origin),
        right_origin=right_origin,
        left_distance=2.0, right_distance=2.0)
    left_signature = result["left_signature"]
    right_signature = result["right_signature"]
    left_origin, right_origin = left_signature.origin_uv, right_signature.origin_uv
    permutation = result["physical_map"]
    expected = np.eye(3, dtype=np.float64)
    expected[:2, :2] = permutation
    expected[:2, 2] = (permutation @ np.asarray(right_origin)
                       - np.asarray(left_origin))
    from mc_photogrammetry.reconstruction import plane_axis_support as c1
    c1_rows = result["support"].mapped_cell_shifts
    assert c1_rows
    left_index, right_index = result["endpoint_indices"]
    left_digest = c1._signature_geometry_digest(left_signature, left_index)
    right_digest = c1._signature_geometry_digest(right_signature, right_index)
    assert left_digest is not None and right_digest is not None
    assert (left_digest, right_digest) == (
        c1_rows[0].left_signature_geometry_sha256,
        c1_rows[0].right_signature_geometry_sha256)
    replay_digest = c1._grid_homography_digest(
        left_signature, right_signature, 1, left_index, right_index, expected)
    p1d_match = result["domains"].domains[0]
    from mc_photogrammetry.reconstruction import plane_shift_domain as p1d
    left_content = p1d._signature_content_sha256(
        left_signature, left_index, result["b_pair"][0])
    right_content = p1d._signature_content_sha256(
        right_signature, right_index, result["b_pair"][1])
    binding = {
        "mapping_version": "block-cell-grid-v2",
        "replayed_homography_sha256": replay_digest,
        "c1_geometry_digests": (
            c1_rows[0].left_signature_geometry_sha256,
            c1_rows[0].right_signature_geometry_sha256),
        "recomputed_c1_geometry_digests": (left_digest, right_digest),
        "c1_homography_digests": tuple(
            row.cell_homography_sha256 for row in c1_rows),
        "p1d_geometry_digests": (
            p1d_match.left_signature_geometry_sha256,
            p1d_match.right_signature_geometry_sha256),
        "recomputed_p1d_geometry_digests": (left_digest, right_digest),
        "p1d_content_digests": (
            p1d_match.left_signature_content_sha256,
            p1d_match.right_signature_content_sha256),
        "recomputed_p1d_content_digests": (left_content, right_content),
        "p1d_homography_digest": p1d_match.cell_homography_sha256,
    }
    assert _finite_compatible_map(
        expected, permutation, left_origin, right_origin, 1e-12, binding) is None

    distinct_left_origin = np.asarray((7.0, -5.0))
    distinct_right_origin = np.asarray((9.0, 3.0))
    reversed_origin = expected.copy()
    reversed_origin[:2, 2] = (permutation @ distinct_left_origin
                              - distinct_right_origin)
    assert _finite_compatible_map(
        reversed_origin, permutation, distinct_left_origin,
        distinct_right_origin, 1e-12, binding) == "grid_origin_incompatible"

    invalid_maps = []
    projective = expected.copy()
    projective[2, 0] = 1e-4
    invalid_maps.append((projective, "homography_projective"))
    scale_error = expected.copy()
    scale_error[0, :2] *= 1.001
    invalid_maps.append((scale_error, "grid_basis_incompatible"))
    wrong_translation = expected.copy()
    wrong_translation[0, 2] += 0.25
    invalid_maps.append((wrong_translation, "grid_origin_incompatible"))
    zero_h22 = expected.copy()
    zero_h22[2, 2] = 0.0
    invalid_maps.append((zero_h22, "homography_denominator_invalid"))
    nonfinite = expected.copy()
    nonfinite[0, 0] = np.inf
    invalid_maps.append((nonfinite, "homography_nonfinite"))
    unstable = expected.copy()
    unstable[2, 2] = 1e-320
    invalid_maps.append((unstable, "homography_denominator_unstable"))
    for invalid, reason in invalid_maps:
        assert _finite_compatible_map(
            invalid, permutation, left_origin, right_origin, 1e-12,
            binding) == reason

    mutations = (
        ("mapping_version", "block-cell-grid-v1",
         "homography_version_unsupported"),
        ("c1_geometry_digests", ("0" * 64, right_digest),
         "c1_geometry_digest_mismatch"),
        ("c1_homography_digests", ("0" * 64,),
         "c1_homography_digest_mismatch"),
        ("p1d_geometry_digests", (left_digest, "0" * 64),
         "p1d_geometry_digest_mismatch"),
        ("p1d_content_digests", ("0" * 64, right_content),
         "p1d_content_digest_mismatch"),
        ("p1d_homography_digest", "0" * 64,
         "p1d_homography_digest_mismatch"),
        ("replayed_homography_sha256", "bad",
         "p1f_homography_digest_invalid"),
    )
    for field, bad_value, reason in mutations:
        changed = dict(binding)
        changed[field] = bad_value
        assert _finite_compatible_map(
            expected, permutation, left_origin, right_origin, 1e-12,
            changed) == reason
    missing_binding = dict(binding)
    del missing_binding["p1d_content_digests"]
    assert _finite_compatible_map(
        expected, permutation, left_origin, right_origin, 1e-12,
        missing_binding) == "compatibility_binding_missing"
    malformed_binding = dict(binding)
    malformed_binding["c1_geometry_digests"] = None
    assert _finite_compatible_map(
        expected, permutation, left_origin, right_origin, 1e-12,
        malformed_binding) == "compatibility_binding_invalid"

    incompatible_basis = permutation.copy()
    incompatible_basis[0] *= -1.0
    assert _finite_compatible_map(
        expected, incompatible_basis, left_origin, right_origin, 1e-12,
        binding) == "grid_basis_incompatible"
    p1f_mutant = expected.copy()
    p1f_mutant[0, 0] += 0.5
    assert _finite_compatible_map(
        p1f_mutant, permutation, left_origin, right_origin, 1e-12,
        binding) == "grid_basis_incompatible"
    assert p1d_match.cell_homography_sha256 == replay_digest
    assert p1d_match.left_signature_geometry_sha256 == left_digest
    assert p1d_match.right_signature_geometry_sha256 == right_digest
    assert (p1d_match.left_signature_content_sha256,
            p1d_match.right_signature_content_sha256) == (
                left_content, right_content)


def test_absolute_orientation_catalog_has_seven_classes_and_both_determinants():
    matrices = set()
    determinant_counts = {1: 0, -1: 0}
    local_pairings = set()
    side_pairs = set()
    orientation_cases = 0
    for left_normal in range(3):
        for left_quadrant in range(4):
            for relative_quadrant in range(4):
                right_quadrant = (left_quadrant + relative_quadrant) % 4
                right_normal_vector = (_physical_yaw(relative_quadrant)
                                       @ np.eye(3)[:, left_normal])
                right_normal = int(np.flatnonzero(
                    np.abs(right_normal_vector) > 0.5)[0])
                left_rotation, right_rotation = (_physical_yaw(left_quadrant),
                                                 _physical_yaw(right_quadrant))
                world_normal = int(np.flatnonzero(
                    np.abs(left_rotation[:, left_normal]) > 0.5)[0])
                world_tangents = tuple(axis for axis in range(3)
                                       if axis != world_normal)
                left_labels = tuple(axis for axis in range(3)
                                    if axis != left_normal)
                right_labels = tuple(axis for axis in range(3)
                                     if axis != right_normal)
                physical = (left_rotation[np.ix_(world_tangents, left_labels)].T
                            @ right_rotation[np.ix_(world_tangents, right_labels)])
                matrices.add(tuple(int(value) for value in physical.ravel()))
                determinant_counts[int(round(np.linalg.det(physical)))] += 1
                local_pairings.add(tuple(int(value) for value in np.argmax(
                    np.abs(physical), axis=1)))
                case_number = left_normal * 16 + left_quadrant * 4 + relative_quadrant
                left_side = 1.0 if case_number % 2 == 0 else -1.0
                right_side = left_side if case_number % 3 == 0 else -left_side
                side_pairs.add((left_side, right_side))
                orientation_cases += 1
    assert orientation_cases == 48
    assert len(matrices) == 7
    assert determinant_counts == {1: 32, -1: 16}
    assert local_pairings == {(0, 1), (1, 0)}
    assert side_pairs == {(1.0, 1.0), (1.0, -1.0),
                          (-1.0, 1.0), (-1.0, -1.0)}


def test_boundary_residues_and_asymmetric_signed_carries_survive():
    boundary = _evaluate_physical_case(
        2, 0, 0, left_side=1.0, right_side=-1.0,
        left_distance=2.0, right_distance=2.0,
        left_origin=(7, -5), right_origin=(7, -5),
        left_translation=(1e-8, -1e-8),
        right_translation=(1.40000001, 0.79999999))
    assert boundary["left_residues"][0] > 1.0 - 2e-8
    assert 0.0 < boundary["left_residues"][1] < 2e-8
    assert boundary["hit_tuples"]

    positive_column_negative_row = _evaluate_physical_case(
        2, 0, 0, left_side=1.0, right_side=-1.0,
        left_distance=2.0, right_distance=2.0,
        left_origin=(7, -5), right_origin=(7, -5),
        left_translation=(0.2, -0.35),
        right_translation=(3.6, -2.55))
    assert positive_column_negative_row["physical_entry_rc"] == (-2, 3)
    assert positive_column_negative_row["raw_carry_magnitude"] > 1
    assert positive_column_negative_row["hit_tuples"]

    negative_column_positive_row = _evaluate_physical_case(
        2, 0, 0, left_side=-1.0, right_side=1.0,
        left_distance=2.0, right_distance=2.0,
        left_origin=(7, -5), right_origin=(7, -5),
        left_translation=(0.2, -0.35),
        right_translation=(-3.2, 1.85))
    assert negative_column_positive_row["physical_entry_rc"] == (2, -4)
    assert negative_column_positive_row["raw_carry_magnitude"] > 1
    assert negative_column_positive_row["hit_tuples"]


def test_world_axis_pairing_and_joint_entry_mutants_are_rejected():
    # Large asymmetric raw carries expose row/column and residue/carry signs.
    big = _evaluate_physical_case(
        2, 0, 0, left_side=1.0, right_side=-1.0,
        left_distance=2.0, right_distance=2.0,
        left_origin=(7, -5), right_origin=(7, -5),
        left_translation=(0.2, -0.35), right_translation=(3.6, -2.55))
    supporting = next(hit for hit in big["hit_tuples"]
                      if hit["entry_rc"] == big["physical_entry_rc"])
    assert supporting["raw_carries"] == (3, -2)
    assert supporting["deltas"] == pytest.approx((3.4, -2.2), abs=1e-12)
    assert supporting["raw_ncc_at_entry"]["support"] >= 9
    assert np.isfinite(supporting["raw_ncc_at_entry"]["value"])
    left_u, right_u = supporting["residue_values"][0]
    left_v, right_v = supporting["residue_values"][1]
    row, column = supporting["entry_rc"]
    mutants = {
        "row_column_transpose": (row + left_u - right_u,
                                 column + left_v - right_v),
        "endpoint_reversal": (-supporting["deltas"][0],
                             -supporting["deltas"][1]),
        "left_residue_sign": (3 - left_u - right_u,
                              -2 - left_v - right_v),
        "right_residue_sign": (3 + left_u + right_u,
                               -2 + left_v + right_v),
    }
    for name, delta in mutants.items():
        assert not np.allclose(delta, (3.4, -2.2), rtol=0.0, atol=1e-12), name

    # A non-symmetric signed rotation catches local-name pairing and the
    # transpose/inverse map mutants that q0 identity cannot expose.
    left_origin = np.asarray((7.0, -5.0))
    oriented_right_basis = np.asarray(((0.0, 1.0), (-1.0, 0.0)))
    oriented_right_origin = tuple(int(value) for value in np.rint(
        oriented_right_basis.T @ (left_origin + 7.5) - 7.5))
    oriented = _evaluate_physical_case(
        0, 0, 3, left_side=1.0, right_side=-1.0,
        left_distance=2.0, right_distance=2.0,
        left_origin=tuple(int(value) for value in left_origin),
        right_origin=oriented_right_origin)
    assert np.array_equal(oriented["physical_map"],
                          np.asarray(((0.0, 1.0), (-1.0, 0.0))))
    assert not np.array_equal(oriented["physical_map"],
                              oriented["physical_map"].T)
    assert oriented["hit_tuples"]
    oriented_hit = next(hit for hit in oriented["hit_tuples"]
                        if hit["entry_rc"] == oriented["physical_entry_rc"])
    assert oriented_hit["deltas"] == pytest.approx(oriented["truth"], abs=1e-12)
    # Pairing u-with-u and v-with-v ignores that this physical quarter turn
    # maps the common world axes to opposite local names.
    left_u, right_u = oriented_hit["residue_values"][0]
    left_v, right_v = oriented_hit["residue_values"][1]
    row, column = oriented_hit["entry_rc"]
    local_name_mutant = (column + left_u - right_u,
                         row + left_v - right_v)
    assert not np.allclose(local_name_mutant, oriented["truth"],
                           rtol=0.0, atol=1e-12)
    # Transposing the right absolute basis (equivalently using the inverse
    # relative permutation here) selects wrong world signs and misses truth.
    inverse_map = oriented["physical_map"].T
    transposed_basis_map = (
        oriented["left_basis"].T @ oriented["right_basis"].T)
    np.testing.assert_array_equal(transposed_basis_map, inverse_map)
    phase = oriented["phase_residues"]
    inverse_deltas = []
    for left_local in range(2):
        right_local = int(np.flatnonzero(np.abs(inverse_map[left_local]) > 0.5)[0])
        right_sign = int(inverse_map[left_local, right_local])
        raw_shift = column if right_local == 0 else row
        left_name, right_name = ("u", "v")[left_local], ("u", "v")[right_local]
        left_residue = phase[("a_left", left_name)][0]
        right_residue = phase[("b_right", right_name)][0]
        inverse_deltas.append(
            right_sign * raw_shift + left_residue - right_sign * right_residue)
    assert not np.allclose(inverse_deltas, oriented["truth"],
                           rtol=0.0, atol=1e-12)

    # With a reflected right tangent, dropping or applying the right carry
    # sign twice yields a different world displacement. Both malformed
    # implementations are exercised against the same independently fixed
    # truth (their numeric expression coincides for a sign of -1).
    reflected = _evaluate_physical_case(
        0, 0, 2, left_side=1.0, right_side=-1.0,
        left_distance=2.0, right_distance=2.0,
        left_origin=(7, -5), right_origin=(7, -10),
        left_translation=(0.2, -0.35), right_translation=(1.6, 2.45))
    reflected_hit = next(hit for hit in reflected["hit_tuples"]
                         if hit["entry_rc"] == reflected["physical_entry_rc"])
    assert reflected["physical_map"].tolist() == [[1.0, 0.0], [0.0, -1.0]]
    assert reflected_hit["signed_carry_signs"] == (1, -1)
    assert reflected_hit["raw_carries"] == (1, -2)
    assert reflected_hit["signed_carries"] == (1, 2)
    assert reflected_hit["deltas"] == pytest.approx((1.4, 2.8), abs=1e-12)
    row_shift, column_shift = reflected_hit["entry_rc"]
    (left_u, right_u), (left_v, right_v) = reflected_hit["residue_values"]
    assert not np.allclose(
        (column_shift + left_u - right_u,
         row_shift + left_v + right_v), (1.4, 2.8), rtol=0.0, atol=1e-12)
    assert (column_shift + left_u - right_u,
            -row_shift + left_v + right_v) == pytest.approx((1.4, 2.8), abs=1e-12)
    # For this sign, missing the right-carry sign and applying it twice are
    # numerically identical; each means substituting raw row_shift for the
    # already-signed carry.
    missing_sign = row_shift + left_v + right_v
    double_sign = -(-row_shift) + left_v + right_v
    assert missing_sign == pytest.approx(double_sign, abs=1e-12)
    assert not np.allclose((column_shift + left_u - right_u, missing_sign),
                           (1.4, 2.8), rtol=0.0, atol=1e-12)

    # The cross-shaped synthetic support has each true component only on a
    # different P1-D entry; a joint vector cannot be assembled across entries.
    simple = _evaluate_physical_case(
        2, 0, 0, left_side=1.0, right_side=-1.0,
        left_distance=2.0, right_distance=2.0,
        left_origin=(7, -5), right_origin=(7, -5))
    residue = simple["phase_residues"]
    u_left, u_right = residue[("a_left", "u")][0], residue[("b_right", "u")][0]
    v_left, v_right = residue[("a_left", "v")][0], residue[("b_right", "v")][0]
    cross_entries = {(0, 1), (1, 0)}
    u_true_entries, v_true_entries = set(), set()
    for entry in simple["domain"].entries:
        rc = (entry.row_shift, entry.column_shift)
        if rc not in cross_entries:
            continue
        if abs(entry.column_shift + u_left - u_right - 1.4) <= 1e-12:
            u_true_entries.add(rc)
        if abs(entry.row_shift + v_left - v_right - 0.8) <= 1e-12:
            v_true_entries.add(rc)
    assert u_true_entries == {(0, 1)}
    assert v_true_entries == {(1, 0)}
    assert not (u_true_entries & v_true_entries)


def test_cyclic_and_disconnected_match_sets_produce_only_declared_domains():
    def run_case(specifications):
        readings, signatures = [], []
        for frame, normal, translation in specifications:
            translation = np.asarray(translation, dtype=np.float64)
            residues = tuple(float((-value) % 1.0) for value in translation)
            reading = _reading(frame, normal, 1.0, 2.0, 1.0, (7, -5), residues)
            basis = np.eye(2, dtype=np.float64)
            signature = _signature(reading, 1.0, basis, translation)
            readings.append(reading)
            signatures.append(signature)
        readings, signatures = tuple(readings), tuple(signatures)
        matches = tuple(
            match for match in plane_graph.match_plane_graph(
                readings, signatures, max_shift_cells=4)
            if match.quadrant_of_right == 0)
        solution = plane_graph.ComponentSolution(
            frames=tuple(reading.frame for reading in readings),
            quadrants={reading.frame: 0 for reading in readings},
            translations={reading.frame: (0.0, 0.0, 0.0) for reading in readings},
            axis_residuals={0: 0.0, 1: 0.0, 2: 0.0},
            axis_resultants={0: 1.0, 1: 1.0, 2: 1.0},
            scale_known=True,
            blocks_per_period={(reading.frame, 0): 1.0 for reading in readings},
            reason="", alternatives=())
        support = c1.build_axis_support(solution, readings, matches, signatures)
        assert support.refusal is None
        result = p1d.build_shift_domains(
            support, matches, signatures, readings=readings,
            max_shift_cells=4,
            budget=p1d.ShiftDomainBudget(
                max_domains=8, max_padded_fft_cells=1_000_000,
                max_shift_positions=1_000_000,
                max_overlap_cell_visits=100_000_000,
                max_buffered_entries=1_000_000))
        assert result.status == "complete", result.refusals
        assert not result.refusals
        assert all(domain.entries for domain in result.domains)
        return matches, result

    cyclic_specs = (
        ("a", 2, (0.2, -0.35)),
        ("b", 2, (1.6, 0.45)),
        ("c", 2, (2.2, 1.2)),
    )
    cyclic_matches, cyclic = run_case(cyclic_specs)
    cyclic_edges = {(match.left, match.right) for match in cyclic_matches}
    assert cyclic_edges == {("a", "b"), ("a", "c"), ("b", "c")}
    assert len(cyclic.domains) == 3

    disconnected_specs = (
        ("a", 2, (0.2, -0.35)),
        ("b", 2, (1.6, 0.45)),
        ("c", 1, (0.2, -0.35)),
        ("d", 1, (1.6, 0.45)),
    )
    disconnected_matches, disconnected = run_case(disconnected_specs)
    disconnected_edges = {
        (match.left, match.right) for match in disconnected_matches}
    assert disconnected_edges == {("a", "b"), ("c", "d")}
    assert len(disconnected.domains) == 2
    assert {
        (domain.match_identity.left_frame, domain.match_identity.right_frame)
        for domain in disconnected.domains
    } == disconnected_edges


def test_canonical_universe_replay_permutation_and_identity_refusals():
    result = _evaluate_physical_case(
        2, 0, 0, left_side=1.0, right_side=-1.0,
        left_distance=2.0, right_distance=2.0,
        left_origin=(7, -5), right_origin=(7, -5))
    domain_result = result["domains"]
    support = result["support"]
    match = result["selected_match"]
    readings = result["readings"]
    signatures = result["signatures"]
    radius = result["domain"].max_shift_cells
    assert p1d.validate_shift_domain_inputs(
        domain_result, support, (match,), signatures,
        max_shift_cells=radius, readings=readings)

    permuted = p1d.build_shift_domains(
        support, (match,), tuple(reversed(signatures)),
        max_shift_cells=radius, budget=p1d.ShiftDomainBudget(
            max_domains=1, max_padded_fft_cells=1_000_000,
            max_shift_positions=1_000_000,
            max_overlap_cell_visits=100_000_000,
            max_buffered_entries=1_000_000),
        readings=tuple(reversed(readings)))
    assert permuted.status == "complete"
    assert permuted.to_record() == domain_result.to_record()

    missing_leading = (readings[0], readings[2])
    assert not p1d.validate_shift_domain_inputs(
        domain_result, support, (match,), signatures,
        max_shift_cells=radius, readings=missing_leading)
    missing = p1d.build_shift_domains(
        support, (match,), signatures, max_shift_cells=radius,
        budget=p1d.ShiftDomainBudget(
            max_domains=1, max_padded_fft_cells=1_000_000,
            max_shift_positions=1_000_000,
            max_overlap_cell_visits=100_000_000,
            max_buffered_entries=1_000_000),
        readings=missing_leading)
    assert missing.status == "refused" and not missing.domains

    left = result["left_signature"]
    foreign_left = replace(left.reading)
    foreign_universe = tuple(
        foreign_left if reading is left.reading else reading for reading in readings)
    foreign = p1d.build_shift_domains(
        support, (match,), signatures, max_shift_cells=radius,
        budget=p1d.ShiftDomainBudget(
            max_domains=1, max_padded_fft_cells=1_000_000,
            max_shift_positions=1_000_000,
            max_overlap_cell_visits=100_000_000,
            max_buffered_entries=1_000_000),
        readings=foreign_universe)
    assert foreign.status == "refused" and not foreign.domains

    duplicate_identity = p1d.build_shift_domains(
        support, (match, match), signatures, max_shift_cells=radius,
        budget=p1d.ShiftDomainBudget(
            max_domains=1, max_padded_fft_cells=1_000_000,
            max_shift_positions=1_000_000,
            max_overlap_cell_visits=100_000_000,
            max_buffered_entries=1_000_000),
        readings=readings)
    assert duplicate_identity.status == "refused"
    assert duplicate_identity.refusals[0].reason == "selected_identity_ambiguous"
    assert not duplicate_identity.domains

    changed_cells = left.cells.copy()
    changed_cells[0, 0] += 0.125
    changed_left = replace(left, cells=changed_cells)
    changed_signatures = tuple(
        changed_left if signature is left else signature for signature in signatures)
    changed_content = p1d.build_shift_domains(
        support, (match,), changed_signatures, max_shift_cells=radius,
        budget=p1d.ShiftDomainBudget(
            max_domains=1, max_padded_fft_cells=1_000_000,
            max_shift_positions=1_000_000,
            max_overlap_cell_visits=100_000_000,
            max_buffered_entries=1_000_000),
        readings=readings)
    assert changed_content.status == "refused"
    assert not changed_content.domains


def test_a3_proof_import_boundary_excludes_external_evidence_surfaces():
    module_source = inspect.getsource(sys.modules[__name__])
    tree = ast.parse(module_source)
    allowed_roots = {
        "__future__", "ast", "dataclasses", "hashlib", "inspect", "sys", "numpy", "pytest",
        "mc_photogrammetry",
    }
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0]
                                  for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots <= allowed_roots
    assert not imported_roots & {
        "os", "pathlib", "socket", "requests", "urllib", "subprocess",
        "PIL", "cv2", "torch",
    }


@pytest.mark.parametrize("left_normal", range(3))
@pytest.mark.parametrize("left_quadrant", range(4))
@pytest.mark.parametrize("relative_quadrant", range(4))
def test_closed_form_absolute_bases_crosscheck_every_physical_map(
        left_normal: int, left_quadrant: int, relative_quadrant: int):
    """Derive compatible normals/bases independently, then cross-check P1-F."""
    right_quadrant = (left_quadrant + relative_quadrant) % 4
    left_basis_3d = _physical_yaw(left_quadrant)
    right_basis_3d = _physical_yaw(right_quadrant)

    # The common plane normal is unoriented. Derive its right local label by
    # applying the physical relative quarter-turn, then verify in world space.
    left_normal_vector = np.eye(3, dtype=np.float64)[:, left_normal]
    right_normal_vector_local = (
        _physical_yaw(relative_quadrant) @ left_normal_vector)
    right_normal = int(np.flatnonzero(np.abs(right_normal_vector_local) > 0.5)[0])
    assert abs(float(left_basis_3d[:, left_normal]
                     @ right_basis_3d[:, right_normal])) == 1.0

    left_labels = tuple(axis for axis in range(3) if axis != left_normal)
    right_labels = tuple(axis for axis in range(3) if axis != right_normal)
    physical_map = left_basis_3d[:, left_labels].T @ right_basis_3d[:, right_labels]
    assert np.isin(physical_map, (-1.0, 0.0, 1.0)).all()
    assert np.array_equal(np.count_nonzero(physical_map, axis=0), (1, 1))
    assert np.array_equal(np.count_nonzero(physical_map, axis=1), (1, 1))

    # Compare every local label with the independent closed-form absolute
    # rotation before cross-checking the production relative map.
    for quadrant, physical in ((left_quadrant, left_basis_3d),
                               (right_quadrant, right_basis_3d)):
        for local_label in range(3):
            world_vector = physical[:, local_label]
            world_axis = int(np.flatnonzero(np.abs(world_vector) > 0.5)[0])
            world_sign = int(round(float(world_vector[world_axis])))
            assert plane_graph.axis_transform(local_label, quadrant) == (
                world_axis, world_sign)

    production_map = np.zeros((2, 2), dtype=np.float64)
    for right_index, right_label in enumerate(right_labels):
        left_label, sign = plane_graph.axis_transform(
            right_label, relative_quadrant)
        assert left_label in left_labels
        production_map[left_labels.index(left_label), right_index] = sign
    np.testing.assert_array_equal(production_map, physical_map)
