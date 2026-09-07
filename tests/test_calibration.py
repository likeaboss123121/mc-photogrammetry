"""Recovering a camera from the lattice in the picture.

The estimator has to be checked against a camera whose answer is known, and the
only trustworthy source of that is the projection the rest of the project uses.
So the synthetic cases here build segments with `project_points` and ask the
estimator to recover the camera that made them: if the vanishing-point derivation
disagrees with the shipped projection, these fail.
"""
import numpy as np
import pytest

from mc_photogrammetry.reconstruction.calibration import (
    CalibrationResult,
    axis_directions,
    estimate_camera,
    segment_residuals,
    vanishing_points,
)
from mc_photogrammetry.reconstruction.camera import CameraModel, project_points

WIDTH, HEIGHT = 1280, 720


def _camera(yaw, pitch, fov):
    return CameraModel(position=(0.5, 0.5, 0.5), yaw_degrees=yaw,
                       pitch_degrees=pitch, fov_degrees=fov)


def _focal(fov, height=HEIGHT):
    return (height / 2.0) / np.tan(np.radians(fov) / 2.0)


def _lattice_segments(camera, width=WIDTH, height=HEIGHT, reach=9):
    """Project the edges of a block lattice, as a detector would see them."""
    eye = np.asarray(camera.position) + [0.0, camera.eye_height, 0.0]
    rows = []
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = 1.0
        for a in range(-reach, reach + 1):
            for b in range(-reach, reach + 1):
                start = np.array(eye, dtype=float)
                other = [i for i in range(3) if i != axis]
                start[other[0]] += a * 2.0
                start[other[1]] += b * 2.0
                start[axis] += 6.0
                pair = project_points([tuple(start), tuple(start + step * 2.0)],
                                      camera, width=width, height=height)
                if any(point is None for point in pair):
                    continue
                (x0, y0, _d0), (x1, y1, _d1) = pair
                if not (0 <= x0 < width and 0 <= x1 < width
                        and 0 <= y0 < height and 0 <= y1 < height):
                    continue
                if np.hypot(x1 - x0, y1 - y0) < 40.0:
                    continue
                rows.append((x0, y0, x1, y1))
    return np.asarray(rows, dtype=float)


def test_the_vertical_axis_points_straight_up_in_camera_space():
    # World +Y must have no sideways component whatever the yaw, or the zero-roll
    # assumption the whole estimator rests on is wrong.
    for yaw in (0.0, 37.0, -122.0):
        right, _up, _forward = axis_directions(yaw, 12.0)[1]
        assert abs(right) < 1e-12


def test_a_vanishing_point_is_where_a_receding_edge_ends_up():
    """Cross-check against `project_points` rather than against the derivation."""
    camera = _camera(31.0, -9.0, 70.0)
    points = vanishing_points(_focal(70.0), 31.0, -9.0, WIDTH, HEIGHT)
    eye = np.asarray(camera.position) + [0.0, camera.eye_height, 0.0]
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = 1.0
        far = project_points([tuple(eye + step * 40000.0)], camera,
                             width=WIDTH, height=HEIGHT)[0]
        if far is None:  # that axis recedes behind the camera; try the other way
            far = project_points([tuple(eye - step * 40000.0)], camera,
                                 width=WIDTH, height=HEIGHT)[0]
        x, y, w = points[axis]
        assert abs(w) > 1e-9
        assert np.allclose([x / w, y / w], [far[0], far[1]], atol=0.5)


def test_a_level_camera_puts_the_vertical_vanishing_point_at_infinity():
    x, y, w = vanishing_points(_focal(70.0), 20.0, 0.0, WIDTH, HEIGHT)[1]
    assert abs(w) < 1e-12, "a level camera has parallel vertical seams"
    assert abs(x) < 1e-9 and abs(y) > 0.0


def test_a_segment_pointing_at_its_vanishing_point_has_no_residual():
    points = vanishing_points(_focal(70.0), 0.0, -20.0, WIDTH, HEIGHT)
    x, y, w = points[1]
    target = np.asarray([x / w, y / w])
    midpoint = np.asarray([600.0, 400.0])
    direction = target - midpoint
    direction /= np.linalg.norm(direction)
    segment = np.concatenate([midpoint - direction * 30.0,
                              midpoint + direction * 30.0])[None, :]
    assert segment_residuals(segment, points)[0, 1] < 1e-6


def test_a_perpendicular_segment_is_penalised_by_its_own_length():
    points = vanishing_points(_focal(70.0), 0.0, -20.0, WIDTH, HEIGHT)
    x, y, w = points[1]
    target = np.asarray([x / w, y / w])
    midpoint = np.asarray([600.0, 400.0])
    direction = target - midpoint
    direction /= np.linalg.norm(direction)
    across = np.asarray([-direction[1], direction[0]])
    segment = np.concatenate([midpoint - across * 30.0,
                              midpoint + across * 30.0])[None, :]
    assert segment_residuals(segment, points)[0, 1] == pytest.approx(30.0, rel=1e-6)


@pytest.mark.parametrize("yaw,pitch,fov", [
    (23.0, -14.0, 70.0),
    (61.0, 8.0, 90.0),
    (7.0, -31.0, 50.0),
])
def test_the_camera_that_made_the_segments_is_recovered(yaw, pitch, fov):
    camera = _camera(yaw, pitch, fov)
    segments = _lattice_segments(camera)
    assert len(segments) > 60, "the synthetic fixture produced too little to fit"
    found = estimate_camera(np.zeros((HEIGHT, WIDTH), np.uint8), segments=segments)
    assert found is not None
    assert found.pitch_degrees == pytest.approx(pitch, abs=0.5)
    assert found.yaw_degrees % 90.0 == pytest.approx(yaw % 90.0, abs=0.5)
    assert found.fov_degrees == pytest.approx(fov, rel=0.02)


def test_too_few_segments_returns_nothing_rather_than_a_guess():
    assert estimate_camera(np.zeros((HEIGHT, WIDTH), np.uint8),
                           segments=np.zeros((3, 4))) is None


def test_the_result_reports_its_own_field_of_view():
    result = CalibrationResult(focal_pixels=_focal(70.0), yaw_degrees=0.0,
                               pitch_degrees=0.0, width=WIDTH, height=HEIGHT,
                               inliers=10, segments=20, median_residual=1.0)
    assert result.fov_degrees == pytest.approx(70.0, abs=1e-6)
    assert result.inlier_fraction == pytest.approx(0.5)
