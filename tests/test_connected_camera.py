"""Public-output controls for the bounded two-view pair solver (DS-11).

Every scene is built by projecting independent 3D points through the shipped
``camera.project_points``, so the tests exercise the same world-to-camera
convention the module consumes. The left view is anchored at yaw class 0; the
right view's true yaw is 95 degrees, so its calibration input is 5 degrees
modulo 90 and the solver must select right class 1 to recover the pair.

Controls: exact recovery of class/rotation/baseline, deterministic 20% outliers,
cheirality sign, unequal explicit intrinsics, and refusals for insufficient,
pure-rotation, identity-self-match, coplanar-with-baseline and ambiguous-quadrant
pairs. A single plane that does not contain the baseline must still solve (DS-10
correction C4), and the left anchor yaw stays a gauge (correction C2).
"""
import math

import numpy as np
import pytest

from mc_photogrammetry.pipeline.connected_camera import PairGates, solve_pair
from mc_photogrammetry.pipeline.contracts import CameraEstimate, CameraIntrinsics, CameraPose
from mc_photogrammetry.reconstruction.camera import (
    CameraModel,
    camera_rotation_matrix,
    intrinsics_for,
    project_points,
)

WIDTH, HEIGHT = 1280, 720
LEFT_POSITION = (0.0, 3.0, 0.0)
RIGHT_POSITION = (2.5, 3.0, 1.0)
LEFT_YAW, LEFT_PITCH = 85.0, 3.0
RIGHT_YAW, RIGHT_PITCH = 95.0, -2.0
RIGHT_YAW_CLASS = 1

_POINTS = (
    (-9.0, 3.0, 0.5), (-8.0, 2.0, 1.5), (-10.0, 4.0, -0.5), (-9.5, 1.5, 2.0),
    (-8.5, 5.0, 0.0), (-11.0, 3.5, 1.0), (-7.5, 2.5, -1.0), (-10.5, 2.0, 2.5),
    (-9.0, 5.5, -1.5), (-12.0, 4.5, 0.5), (-8.0, 3.8, 2.8), (-10.0, 1.0, -0.8),
    (-13.0, 3.0, 1.5), (-7.0, 3.2, 0.8), (-11.5, 2.8, -1.8), (-9.8, 4.8, 2.2),
    (-8.8, 1.2, 0.2), (-12.5, 2.2, 2.0), (-10.8, 5.2, -1.2), (-7.8, 4.2, -1.8),
)


def _camera(position, yaw, pitch, *, intrinsics=None):
    return CameraModel(position, yaw, pitch, eye_height=0.0, intrinsics=intrinsics)


def _estimate(camera, *, width=WIDTH, height=HEIGHT):
    """The calibration-style estimate: yaw reported modulo 90 degrees."""
    return CameraEstimate(
        intrinsics=intrinsics_for(camera, width, height),
        pose=CameraPose(
            position=None,
            yaw_degrees=float(camera.yaw_degrees % 90.0),
            pitch_degrees=float(camera.pitch_degrees),
            roll_degrees=float(camera.roll_degrees),
        ),
        residual_pixels=0.5,
        confidence=0.0,
    )


def _estimate_with_yaw(camera, yaw_degrees, *, width=WIDTH, height=HEIGHT):
    """An estimate whose yaw representative is stated explicitly (gauge control)."""
    return CameraEstimate(
        intrinsics=intrinsics_for(camera, width, height),
        pose=CameraPose(
            position=None,
            yaw_degrees=float(yaw_degrees),
            pitch_degrees=float(camera.pitch_degrees),
            roll_degrees=float(camera.roll_degrees),
        ),
        residual_pixels=0.5,
        confidence=0.0,
    )


def _matches(left_camera, right_camera, points, *,
             left_size=(WIDTH, HEIGHT), right_size=(WIDTH, HEIGHT)):
    left_pixels = project_points(points, left_camera, width=left_size[0], height=left_size[1])
    right_pixels = project_points(points, right_camera, width=right_size[0], height=right_size[1])
    rows = []
    for left_pixel, right_pixel in zip(left_pixels, right_pixels):
        assert left_pixel is not None, "left view point fell behind the camera"
        assert right_pixel is not None, "right view point fell behind the camera"
        rows.append((left_pixel[0], left_pixel[1], right_pixel[0], right_pixel[1]))
    return np.asarray(rows, dtype=np.float64)


def _scene():
    left = _camera(LEFT_POSITION, LEFT_YAW, LEFT_PITCH)
    right = _camera(RIGHT_POSITION, RIGHT_YAW, RIGHT_PITCH)
    return left, right, _matches(left, right, _POINTS)


def _rotation_angle_degrees(first, second):
    product = np.asarray(first, dtype=np.float64).T @ np.asarray(second, dtype=np.float64)
    cosine = np.clip((np.trace(product) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def _direction_angle_degrees(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    return math.degrees(math.acos(float(np.clip(np.dot(first, second), -1.0, 1.0))))


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def _expected_baseline():
    return _unit(np.asarray(RIGHT_POSITION) - np.asarray(LEFT_POSITION))


def test_yaw_quadrant_copy_is_a_right_multiplication():
    """R(yaw + 90) = R(yaw) @ Ry(90) in this repository's convention."""
    base = _camera((0.0, 0.0, 0.0), 85.0, 3.0)
    shifted = _camera((0.0, 0.0, 0.0), 175.0, 3.0)
    ry90 = np.asarray(((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)))
    assert camera_rotation_matrix(shifted) == pytest.approx(camera_rotation_matrix(base) @ ry90)


def test_exact_pair_recovers_class_rotation_and_world_baseline():
    left, right, matches = _scene()
    result = solve_pair(_estimate(left), _estimate(right), matches, left_id="a", right_id="b")

    assert result.solved is True
    assert result.reason == ""
    assert result.left_id == "a" and result.right_id == "b"
    assert result.correspondences == len(_POINTS)
    assert result.right_yaw_class == RIGHT_YAW_CLASS
    assert result.inliers == len(_POINTS)
    assert result.inlier_fraction == pytest.approx(1.0)
    assert result.median_rho_degrees is not None and result.median_rho_degrees < 0.01
    assert result.parallax_degrees is not None and result.parallax_degrees >= 3.0
    assert result.cheirality_fraction == pytest.approx(1.0)
    assert len(result.classes) == 4
    assert result.relative_rotation is not None
    assert result.baseline_direction is not None

    expected_rotation = camera_rotation_matrix(right) @ camera_rotation_matrix(left).T
    assert _rotation_angle_degrees(result.relative_rotation, expected_rotation) <= 1.0
    assert _direction_angle_degrees(result.baseline_direction, _expected_baseline()) <= 1.0


def test_baseline_sign_is_fixed_by_cheirality():
    left, right, matches = _scene()
    result = solve_pair(_estimate(left), _estimate(right), matches)

    assert result.solved is True
    assert result.baseline_direction is not None
    expected = _expected_baseline()
    assert float(np.dot(result.baseline_direction, expected)) > 0.999
    assert float(np.linalg.norm(result.baseline_direction)) == pytest.approx(1.0)
    assert result.cheirality_fraction == pytest.approx(1.0)


def test_twenty_percent_outliers_are_rejected_deterministically():
    left, right, matches = _scene()
    corrupted = matches.copy()
    generator = np.random.default_rng(20260912)
    outliers = generator.choice(len(corrupted), size=4, replace=False)
    offsets = generator.uniform(80.0, 200.0, size=(4, 2)) * generator.choice(
        np.asarray((-1.0, 1.0)), size=(4, 2)
    )
    for row, index in enumerate(outliers):
        corrupted[index, 2:4] += offsets[row]

    first = solve_pair(_estimate(left), _estimate(right), corrupted)
    second = solve_pair(_estimate(left), _estimate(right), corrupted)

    assert first.solved is True and second.solved is True
    assert first.inliers == len(_POINTS) - 4
    assert first.right_yaw_class == RIGHT_YAW_CLASS
    assert first.baseline_direction == second.baseline_direction
    assert first.right_yaw_class == second.right_yaw_class
    assert _direction_angle_degrees(first.baseline_direction, _expected_baseline()) <= 1.0


def test_unequal_explicit_intrinsics_and_principal_points():
    left_intrinsics = CameraIntrinsics((1280, 720), (900.0, 880.0), (600.0, 380.0))
    right_intrinsics = CameraIntrinsics((1024, 768), (700.0, 760.0), (540.0, 350.0))
    left = _camera(LEFT_POSITION, LEFT_YAW, LEFT_PITCH, intrinsics=left_intrinsics)
    right = _camera(RIGHT_POSITION, RIGHT_YAW, RIGHT_PITCH, intrinsics=right_intrinsics)
    matches = _matches(left, right, _POINTS, left_size=(1280, 720), right_size=(1024, 768))

    result = solve_pair(
        _estimate(left, width=1280, height=720),
        _estimate(right, width=1024, height=768),
        matches,
    )

    assert result.solved is True
    assert result.right_yaw_class == RIGHT_YAW_CLASS
    assert result.relative_rotation is not None and result.baseline_direction is not None
    expected_rotation = camera_rotation_matrix(right) @ camera_rotation_matrix(left).T
    assert _rotation_angle_degrees(result.relative_rotation, expected_rotation) <= 1.0
    assert _direction_angle_degrees(result.baseline_direction, _expected_baseline()) <= 1.0


def test_insufficient_correspondences_refuse():
    left, right, matches = _scene()
    result = solve_pair(_estimate(left), _estimate(right), matches[:5])

    assert result.solved is False
    assert result.reason == "insufficient_correspondences"
    assert result.correspondences == 5
    assert result.baseline_direction is None
    assert result.relative_rotation is None
    assert result.classes == ()


def test_pure_rotation_refuses():
    left = _camera(LEFT_POSITION, LEFT_YAW, LEFT_PITCH)
    right = _camera(LEFT_POSITION, RIGHT_YAW, RIGHT_PITCH)
    matches = _matches(left, right, _POINTS)

    result = solve_pair(_estimate(left), _estimate(right), matches)

    assert result.solved is False
    assert result.reason == "pure_rotation"
    assert result.baseline_direction is None and result.relative_rotation is None
    assert result.right_yaw_class is None


def test_identity_self_match_refuses():
    camera = _camera(LEFT_POSITION, LEFT_YAW, LEFT_PITCH)
    matches = _matches(camera, camera, _POINTS)
    assert np.allclose(matches[:, 0:2], matches[:, 2:4])

    result = solve_pair(_estimate(camera), _estimate(camera), matches)

    assert result.solved is False
    assert result.reason == "pure_rotation"
    assert result.baseline_direction is None


def _ground_points():
    return tuple((x, 0.0, z) for (x, _y, z) in _POINTS)


def _tilted_plane_points():
    """One plane with a non-vertical normal (``y + z = 6``) and a skew baseline.

    A plane whose normal is the vertical axis makes the plane-homography
    two-fold ambiguity land exactly on the 180-degree yaw class, so that scene
    is genuinely ambiguous. Tilting the plane keeps it a single plane while
    leaving the true quadrant identifiable, which is what DS-10 correction C4
    requires the solver to do instead of refusing planarity outright.
    """
    return tuple((x, y, 6.0 - y) for (x, y, _z) in _POINTS)


def test_coplanar_with_baseline_refuses_by_conditioning():
    left = _camera((0.0, 0.0, 0.0), LEFT_YAW, LEFT_PITCH)
    right = _camera((2.5, 0.0, 1.0), RIGHT_YAW, RIGHT_PITCH)
    matches = _matches(left, right, _ground_points())

    result = solve_pair(_estimate(left), _estimate(right), matches)

    assert result.solved is False
    assert result.reason == "rank_deficient"
    assert result.baseline_direction is None and result.relative_rotation is None
    assert len(result.classes) == 4


def test_single_plane_without_baseline_still_solves():
    left = _camera((0.0, 4.0, 0.0), LEFT_YAW, LEFT_PITCH)
    right = _camera((2.5, 4.0, 1.0), RIGHT_YAW, RIGHT_PITCH)
    matches = _matches(left, right, _tilted_plane_points())

    result = solve_pair(_estimate(left), _estimate(right), matches)

    assert result.solved is True
    assert result.right_yaw_class == RIGHT_YAW_CLASS
    assert result.relative_rotation is not None and result.baseline_direction is not None
    assert _direction_angle_degrees(result.baseline_direction, _expected_baseline()) <= 1.0


def test_ambiguous_yaw_quadrant_refuses():
    left, right, matches = _scene()
    assert solve_pair(_estimate(left), _estimate(right), matches).solved is True

    permissive = PairGates(
        inlier_residual_degrees=90.0,
        max_median_residual_degrees=90.0,
        residual_margin_degrees=360.0,
    )
    result = solve_pair(_estimate(left), _estimate(right), matches, gates=permissive)

    assert result.solved is False
    assert result.reason == "ambiguous_yaw_class"
    assert result.baseline_direction is None and result.relative_rotation is None
    assert result.right_yaw_class is None
    assert len(result.classes) == 4
    assert any(score.passed for score in result.classes)


def test_left_anchor_yaw_is_gauge():
    left, right, matches = _scene()
    first = solve_pair(_estimate(left), _estimate(right), matches)
    second = solve_pair(_estimate_with_yaw(left, LEFT_YAW + 90.0), _estimate(right), matches)

    assert first.solved is True and second.solved is True
    assert first.right_yaw_class is not None and second.right_yaw_class is not None
    assert second.right_yaw_class == (first.right_yaw_class + 1) % 4
    assert first.relative_rotation is not None and second.relative_rotation is not None
    assert _rotation_angle_degrees(first.relative_rotation, second.relative_rotation) <= 1e-6
    assert first.baseline_direction is not None and second.baseline_direction is not None
    ry_minus_90 = np.asarray(((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)))
    expected = ry_minus_90 @ np.asarray(first.baseline_direction)
    assert _direction_angle_degrees(second.baseline_direction, expected) <= 1e-6
