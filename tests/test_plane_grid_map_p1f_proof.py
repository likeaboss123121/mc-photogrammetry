"""P1-F independent right-to-left map theorem; no production expectations."""
from __future__ import annotations

import numpy as np


# Independent literal table: local label -> (world label, sign).
_T = (
    ((0, 1), (1, 1), (2, 1)),
    ((2, 1), (1, 1), (0, -1)),
    ((0, -1), (1, 1), (2, -1)),
    ((2, -1), (1, 1), (0, 1)),
)
_ORIGIN_L = np.asarray((3.25, -2.5), dtype=np.float64)
_ORIGIN_R = np.asarray((-1.75, 4.5), dtype=np.float64)
_LEFT_SHAPE = (33, 29)
_RIGHT_SHAPE = (27, 41)


def _tangents(normal: int) -> tuple[int, int]:
    return tuple(axis for axis in range(3) if axis != normal)  # type: ignore[return-value]


def _absolute_basis(normal: int, quadrant: int,
                    world_tangents: tuple[int, int]) -> np.ndarray:
    basis = np.zeros((2, 2), dtype=np.float64)
    for local_column, local_label in enumerate(_tangents(normal)):
        world_label, sign = _T[quadrant][local_label]
        basis[world_tangents.index(world_label), local_column] = sign
    return basis


def _right_to_left_relative_map(left_normal: int, right_normal: int,
                                relative_quadrant: int) -> np.ndarray:
    """Construct H's linear part from right labels toward left labels."""
    left_labels = _tangents(left_normal)
    right_labels = _tangents(right_normal)
    result = np.zeros((2, 2), dtype=np.float64)
    for right_column, right_label in enumerate(right_labels):
        left_label, sign = _T[relative_quadrant][right_label]
        result[left_labels.index(left_label), right_column] = sign
    return result


def test_all_48_right_to_left_maps_equal_independent_absolute_bases():
    observed = set()
    determinants = set()
    rejected_mutants = set()
    old_direction_mismatches = []
    cases = 0
    for left_normal in range(3):
        for left_q in range(4):
            for relative_q in range(4):
                right_q = (left_q + relative_q) % 4
                right_normal = _T[relative_q][left_normal][0]
                left_world_normal, _ = _T[left_q][left_normal]
                right_world_normal, _ = _T[right_q][right_normal]
                assert left_world_normal == right_world_normal
                world_tangents = tuple(axis for axis in range(3)
                                       if axis != left_world_normal)
                b_left = _absolute_basis(left_normal, left_q, world_tangents)
                b_right = _absolute_basis(right_normal, right_q, world_tangents)
                physical = b_left.T @ b_right
                relative = _right_to_left_relative_map(
                    left_normal, right_normal, relative_q)
                np.testing.assert_array_equal(relative, physical)
                assert np.isin(relative, (-1.0, 0.0, 1.0)).all()
                assert np.all(np.count_nonzero(relative, axis=0) == 1)
                assert np.all(np.count_nonzero(relative, axis=1) == 1)
                determinant = int(round(float(np.linalg.det(relative))))
                determinants.add(determinant)
                observed.add(tuple(relative.reshape(-1).astype(int)))

                h = np.eye(3, dtype=np.float64)
                h[:2, :2] = relative
                h[:2, 2] = relative @ _ORIGIN_R - _ORIGIN_L
                assert _LEFT_SHAPE[0] != _LEFT_SHAPE[1]
                assert _RIGHT_SHAPE[0] != _RIGHT_SHAPE[1]
                assert _LEFT_SHAPE != _RIGHT_SHAPE
                right_points = ((0.0, 0.0), (20.0, 13.0),
                                (_RIGHT_SHAPE[1] - 1.0, _RIGHT_SHAPE[0] - 1.0),
                                (7.25, 11.5))
                for right_column, right_row in right_points:
                    right_point = np.asarray((right_column, right_row), dtype=np.float64)
                    world_point = b_right @ (right_point + _ORIGIN_R)
                    expected_left = b_left.T @ world_point - _ORIGIN_L
                    homogeneous_left = h @ np.asarray((right_point[0],
                                                        right_point[1], 1.0))
                    np.testing.assert_array_equal(homogeneous_left[:2], expected_left)
                    np.testing.assert_allclose(np.linalg.inv(h) @ homogeneous_left,
                                               (right_point[0], right_point[1], 1.0),
                                               rtol=0.0, atol=1e-12)

                if not np.array_equal(relative.T, physical):
                    rejected_mutants.add("transpose")
                if not np.array_equal(np.linalg.inv(relative), physical):
                    rejected_mutants.add("inverse")
                reversed_endpoints = relative.T
                if not np.array_equal(reversed_endpoints, physical):
                    rejected_mutants.add("endpoint_order")
                sign_mutant = relative.copy()
                first_nonzero = tuple(np.argwhere(sign_mutant != 0.0)[0])
                sign_mutant[first_nonzero] *= -1.0
                if not np.array_equal(sign_mutant, physical):
                    rejected_mutants.add("sign")
                swapped_origin_translation = relative @ _ORIGIN_L - _ORIGIN_R
                if not np.array_equal(swapped_origin_translation,
                                      h[:2, 2]):
                    rejected_mutants.add("origin_order")

                # Record the old left-label iteration's first false map, without
                # using that direction to construct the accepted expectation.
                old = np.zeros((2, 2), dtype=np.float64)
                for left_row, left_label in enumerate(_tangents(left_normal)):
                    right_label, sign = _T[relative_q][left_label]
                    old[left_row, _tangents(right_normal).index(right_label)] = sign
                if not np.array_equal(old, physical):
                    old_direction_mismatches.append(
                        (left_normal, left_q, relative_q, old.copy(), physical.copy()))
                cases += 1
    assert cases == 48
    assert determinants == {-1, 1}
    assert observed == {
        (-1, 0, 0, -1), (-1, 0, 0, 1), (1, 0, 0, -1), (1, 0, 0, 1),
        (0, -1, 1, 0), (0, 1, -1, 0), (0, 1, 1, 0),
    }
    assert rejected_mutants == {
        "transpose", "inverse", "endpoint_order", "sign", "origin_order"}
    first = old_direction_mismatches[0]
    assert first[:3] == (0, 0, 1)
    np.testing.assert_array_equal(first[3], ((0, 1), (-1, 0)))
    np.testing.assert_array_equal(first[4], ((0, 1), (1, 0)))
