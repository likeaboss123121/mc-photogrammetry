"""P1-F real correlation, C1 and P1-D checks on independent basis maps."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import numpy as np

from mc_photogrammetry.reconstruction import lattice_planes, plane_axis_support, plane_graph
from mc_photogrammetry.reconstruction import plane_shift_domain as p1d


# Independent absolute orientation table: local label -> (world label, sign).
_ABSOLUTE = (
    ((0, 1), (1, 1), (2, 1)),
    ((2, 1), (1, 1), (0, -1)),
    ((0, -1), (1, 1), (2, -1)),
    ((2, -1), (1, 1), (0, 1)),
)
_REPRESENTATIVES = (
    (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3),
    (1, 0, 1), (1, 0, 2), (2, 0, 2),
)


def _local_tangents(normal: int) -> tuple[int, int]:
    return tuple(label for label in range(3) if label != normal)  # type: ignore[return-value]


def _basis(normal: int, quadrant: int, world_tangents: tuple[int, int]) -> np.ndarray:
    value = np.zeros((2, 2), dtype=np.float64)
    for column, local in enumerate(_local_tangents(normal)):
        world, sign = _ABSOLUTE[quadrant][local]
        value[world_tangents.index(world), column] = sign
    return value


def _texture(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (2.0 + 0.37 * np.cos(2.0 * np.pi * x / 7.3)
            + 0.29 * np.sin(2.0 * np.pi * y / 9.1)
            + 1.7 * np.exp(-((x - 19.3) ** 2 / 5.7 + (y - 25.7) ** 2 / 3.1))
            + 0.63 * np.exp(-((x - 32.1) ** 2 / 3.4 + (y - 11.2) ** 2 / 8.3)))


def _reading(frame: str, axis: int, side: float) -> lattice_planes.PlaneReading:
    return lattice_planes.PlaneReading(
        frame=frame, axis=axis, side=side,
        alternatives=(lattice_planes.DistanceAlternative(1.0, 4.0, 1.0),),
        period_pixels=60.0, phase_pixels=(0.0, 0.0), extent=1.0,
        samples=64, polygon=None, coverage_cells=64 * 64,
        structural_share=1.0, confidence=1.0,
        rectification=lattice_planes.PlaneRectification(
            axis_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            focal=500.0, principal_point=(31.0, 31.0), centre=(0.0, 0.0)))


def _signature(reading, basis: np.ndarray, world_tangents: tuple[int, int],
               origin=(0.0, 0.0)):
    shape = (64, 64)
    rows, columns = np.indices(shape, dtype=np.float64)
    local = np.stack((columns, rows), axis=0).reshape(2, -1)
    origin_array = np.asarray(origin, dtype=np.float64).reshape(2, 1)
    world = basis @ (local + origin_array)
    cells = _texture(world[0].reshape(shape), world[1].reshape(shape)).astype(np.float32)
    return lattice_planes.PlaneSignature(
        reading=reading, blocks_per_period=1.0, cells=cells,
        interior=np.zeros_like(cells), coverage=np.ones(shape, dtype=bool),
        origin_uv=tuple(map(float, origin)))


def _budget() -> p1d.ShiftDomainBudget:
    return p1d.ShiftDomainBudget(
        max_domains=4, max_padded_fft_cells=2_000_000,
        max_shift_positions=10_000, max_overlap_cell_visits=10_000_000,
        max_buffered_entries=10_000)


def test_all_signed_rotation_and_reflection_classes_reach_match_c1_and_p1d():
    matrices = set()
    for left_normal, left_q, relative_q in _REPRESENTATIVES:
        right_q = (left_q + relative_q) % 4
        right_normal, _normal_sign = _ABSOLUTE[relative_q][left_normal]
        left_world_normal, _ = _ABSOLUTE[left_q][left_normal]
        right_world_normal, _ = _ABSOLUTE[right_q][right_normal]
        assert left_world_normal == right_world_normal
        world_tangents = tuple(axis for axis in range(3) if axis != left_world_normal)
        left_basis = _basis(left_normal, left_q, world_tangents)
        right_basis = _basis(right_normal, right_q, world_tangents)
        physical_p = left_basis.T @ right_basis
        matrices.add(tuple(physical_p.reshape(-1).astype(int)))
        # Give each reflected source axis the full-grid origin offset needed
        # for the independently predicted map to land inside the left image.
        offset_left = np.where(np.min(physical_p, axis=1) < 0.0, 63.0, 0.0)
        origin_right = physical_p.T @ offset_left
        independent_h = np.eye(3, dtype=np.float64)
        independent_h[:2, :2] = physical_p
        independent_h[:2, 2] = offset_left

        left_reading = _reading("left", left_normal, 1.0)
        right_reading = _reading("right", right_normal, -1.0)
        left = _signature(left_reading, left_basis, world_tangents)
        right = _signature(right_reading, right_basis, world_tangents,
                           origin=origin_right)
        actual_h = plane_graph.cell_grid_homography(left, right, relative_q)
        np.testing.assert_array_equal(actual_h, independent_h)

        # The independent physical map, rather than the helper, drives this
        # expected warp and correlation replay.
        expected_warp, expected_coverage = lattice_planes.warp_signature(
            left, independent_h, right.cells.shape)
        fixed = expected_coverage & right.coverage
        first = np.where(fixed, expected_warp, 0.0).astype(np.float64)
        second = np.where(fixed, right.cells, 0.0).astype(np.float64)
        first[fixed] -= float(first[fixed].mean())
        second[fixed] -= float(second[fixed].mean())
        surface = plane_graph._correlation_surface(first, second, max_shift_cells=8)
        assert surface is not None
        expected_peak, expected_psr, expected_second, expected_shift = \
            plane_graph._peak_statistics(surface, max_shift_cells=8)

        match = plane_graph.match_plane_pair(
            left, right, quadrant=relative_q, max_shift_cells=8)
        assert match is not None
        match = replace(match, left_plane=0, right_plane=0)
        assert match.cell_shift_rc == expected_shift
        assert match.peak == expected_peak
        assert match.psr == expected_psr
        assert match.second_peak == expected_second
        assert match.overlap_cells == int(np.count_nonzero(fixed))

        solution = plane_graph.ComponentSolution(
            frames=("left", "right"),
            quadrants={"left": left_q, "right": right_q},
            translations={"left": (0.0, 0.0, 0.0),
                          "right": (0.0, 0.0, 0.0)},
            axis_residuals={0: 0.0, 1: 0.0, 2: 0.0},
            axis_resultants={0: 1.0, 1: 1.0, 2: 1.0}, scale_known=True,
            blocks_per_period={("left", 0): 1.0, ("right", 0): 1.0},
            reason="", alternatives=())
        support = plane_axis_support.build_axis_support(
            solution, (left_reading, right_reading), (match,), (left, right))
        assert support.refusal is None
        digest = plane_axis_support._grid_homography_digest(
            left, right, relative_q, 0, 0, independent_h)
        assert digest is not None
        assert digest == next(iter({
            value.cell_homography_sha256 for value in support.mapped_cell_shifts}))

        result = p1d.build_shift_domains(
            support, (match,), (left, right), max_shift_cells=8, budget=_budget(),
            readings=(left_reading, right_reading))
        assert result.status == "complete"
        assert len(result.domains) == 1
        assert result.domains[0].cell_homography_sha256 == digest

    assert len(matrices) == 7
    assert {int(round(np.linalg.det(np.asarray(value).reshape(2, 2))))
            for value in matrices} == {-1, 1}


def test_grid_digest_v2_binds_map_direction_and_p1d_rejects_v1_provenance():
    # A direct v1 record represents the old P1-E bytes. The current C1/P1-D
    # path must not accept it after the direction repair.
    left_reading = _reading("left", 0, 1.0)
    right_reading = _reading("right", 2, -1.0)
    left = _signature(left_reading, np.eye(2), (1, 2))
    right_basis = np.asarray(((0.0, 1.0), (1.0, 0.0)))
    right = _signature(right_reading, right_basis, (1, 2))
    match = plane_graph.match_plane_pair(left, right, quadrant=1, max_shift_cells=8)
    assert match is not None
    match = replace(match, left_plane=0, right_plane=0)
    h = plane_graph.cell_grid_homography(left, right, 1)
    normalized_digest = plane_axis_support._normalized_homography_digest(h)
    assert normalized_digest is not None
    legacy_record = {
        "mapping_version": "block-cell-grid-v1",
        "left": {"frame": "left", "plane": 0, "axis": 0, "b": 1.0,
                 "shape": [64, 64], "origin_uv": [0.0, 0.0]},
        "right": {"frame": "right", "plane": 0, "axis": 2, "b": 1.0,
                  "shape": [64, 64], "origin_uv": [0.0, 0.0]},
        "relative_quadrant": 1, "homography": h.tolist(),
        "normalized_homography_sha256": normalized_digest,
    }
    legacy_digest = hashlib.sha256(json.dumps(
        legacy_record, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("ascii")).hexdigest()
    solution = plane_graph.ComponentSolution(
        frames=("left", "right"), quadrants={"left": 0, "right": 1},
        translations={"left": (0.0, 0.0, 0.0), "right": (0.0, 0.0, 0.0)},
        axis_residuals={0: 0.0, 1: 0.0, 2: 0.0},
        axis_resultants={0: 1.0, 1: 1.0, 2: 1.0}, scale_known=True,
        blocks_per_period={("left", 0): 1.0, ("right", 0): 1.0},
        reason="", alternatives=())
    support = plane_axis_support.build_axis_support(
        solution, (left_reading, right_reading), (match,), (left, right))
    assert support.refusal is None
    current = plane_axis_support._grid_homography_digest(left, right, 1, 0, 0, h)
    assert current is not None and current != legacy_digest
    assert all(value.cell_homography_sha256 == current
               for value in support.mapped_cell_shifts)
    forged_support = replace(support, mapped_cell_shifts=tuple(
        replace(value, cell_homography_sha256=legacy_digest)
        for value in support.mapped_cell_shifts))
    refused = p1d.build_shift_domains(
        forged_support, (match,), (left, right), max_shift_cells=8, budget=_budget(),
        readings=(left_reading, right_reading))
    assert refused.status == "refused"
    assert not refused.domains
    assert refused.refusals[0].reason == "provenance_mismatch"
