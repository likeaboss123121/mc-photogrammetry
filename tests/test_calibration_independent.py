"""Calibration checked against a camera this project did not write.

The existing synthetic tests build their segments with `project_points` and then
score them with `vanishing_points`, and both now route through the same shared
rotation. That is circular: a systematic error in the shared convention would
appear identically on each side and the tests would pass anyway.

So the camera here is rebuilt from first principles - elementary rotation
matrices composed by hand and a textbook pinhole projection - using only the
conventions stated in prose in `camera.py`, never its code:

* yaw 0 looks toward world +Z, and screen right is world -X, so facing south
  puts west on the right of frame;
* pitch is positive looking down;
* the field-of-view setting is the *vertical* one, so focal length follows the
  image height;
* roll spins the frame about the view direction.

`test_the_projects_projection_matches_an_independent_one` is the load-bearing
test. If it fails, the disagreement is real and everything else here is
measuring the wrong thing.
"""
import numpy as np
import pytest

from mc_photogrammetry.reconstruction.calibration import estimate_camera
from mc_photogrammetry.reconstruction.camera import CameraModel, project_points


def _rotation(yaw_degrees, pitch_degrees, roll_degrees):
    """Camera basis as rows ``(right, up, forward)``, built from scratch."""
    yaw, pitch, roll = np.radians([yaw_degrees, pitch_degrees, roll_degrees])

    # Level frame: forward follows yaw around the vertical, right is world -X
    # when facing +Z, and up is world up.
    forward = np.asarray([-np.sin(yaw), 0.0, np.cos(yaw)])
    right = np.asarray([-np.cos(yaw), 0.0, -np.sin(yaw)])
    up = np.asarray([0.0, 1.0, 0.0])

    # Pitch tips forward and up about the right axis; positive looks down.
    pitched_forward = np.cos(pitch) * forward - np.sin(pitch) * up
    pitched_up = np.sin(pitch) * forward + np.cos(pitch) * up

    # Roll spins right and up about the view direction.
    rolled_right = np.cos(roll) * right - np.sin(roll) * pitched_up
    rolled_up = np.sin(roll) * right + np.cos(roll) * pitched_up
    return np.stack([rolled_right, rolled_up, pitched_forward])


def _project(points, eye, yaw, pitch, roll, fov, width, height):
    """Textbook pinhole projection: rotate, divide by depth, offset by centre."""
    basis = _rotation(yaw, pitch, roll)
    focal = (height / 2.0) / np.tan(np.radians(fov) / 2.0)
    offsets = np.asarray(points, dtype=float) - np.asarray(eye, dtype=float)
    camera = offsets @ basis.T                      # columns: right, up, forward
    out = []
    for x, y, depth in camera:
        if depth <= 1e-9:
            out.append(None)
            continue
        out.append((width / 2.0 + x * focal / depth,
                    height / 2.0 - y * focal / depth,
                    depth))
    return out


def _segments(eye, yaw, pitch, roll, fov, width, height, reach=8, step=2.0):
    """Lattice edges as this independent camera sees them."""
    rows = []
    for axis in range(3):
        along = np.zeros(3)
        along[axis] = step
        others = [i for i in range(3) if i != axis]
        for a in range(-reach, reach + 1):
            for b in range(-reach, reach + 1):
                start = np.asarray(eye, dtype=float).copy()
                start[others[0]] += a * step
                start[others[1]] += b * step
                start[axis] += 6.0
                pair = _project([start, start + along], eye, yaw, pitch, roll,
                                fov, width, height)
                if any(p is None for p in pair):
                    continue
                (x0, y0, _d0), (x1, y1, _d1) = pair
                if not (0 <= x0 < width and 0 <= x1 < width
                        and 0 <= y0 < height and 0 <= y1 < height):
                    continue
                if np.hypot(x1 - x0, y1 - y0) < 40.0:
                    continue
                rows.append((x0, y0, x1, y1))
    return np.asarray(rows, dtype=float) if rows else np.zeros((0, 4))


def _fold(degrees):
    return (degrees + 45.0) % 90.0 - 45.0


@pytest.mark.parametrize("yaw,pitch,roll", [
    (0.0, 0.0, 0.0), (31.0, -9.0, 0.0), (-76.0, 17.0, 0.0),
    (123.0, 40.0, 0.0), (45.0, -30.0, 12.0), (-15.0, 5.0, -7.5),
])
def test_the_projects_projection_matches_an_independent_one(yaw, pitch, roll):
    """The check the rest of this file depends on.

    Both sides claim the same pinhole camera; if they disagree, one of the two
    conventions is wrong and every synthetic result built on it is suspect.
    """
    width, height, fov = 1280, 720, 70.0
    camera = CameraModel(position=(3.0, 4.0, -2.0), yaw_degrees=yaw,
                         pitch_degrees=pitch, fov_degrees=fov,
                         roll_degrees=roll)
    eye = np.asarray(camera.position, float) + [0.0, camera.eye_height, 0.0]
    rng = np.random.default_rng(19)
    points = eye + rng.uniform(-30.0, 30.0, size=(60, 3))

    theirs = project_points([tuple(p) for p in points], camera,
                            width=width, height=height)
    mine = _project(points, eye, yaw, pitch, roll, fov, width, height)
    compared = 0
    for a, b in zip(theirs, mine):
        if a is None or b is None:
            assert (a is None) == (b is None), "disagreed about visibility"
            continue
        assert np.allclose(a, b, atol=1e-6), f"{a} vs {b}"
        compared += 1
    assert compared > 20, "too few points in front of the camera to be a check"


@pytest.mark.parametrize("yaw,pitch,fov", [
    (23.0, -14.0, 70.0), (61.0, 8.0, 90.0), (7.0, -31.0, 50.0),
])
def test_a_camera_built_elsewhere_is_recovered(yaw, pitch, fov):
    width, height = 1280, 720
    eye = (0.5, 0.5, 0.5)
    segments = _segments(eye, yaw, pitch, 0.0, fov, width, height)
    assert len(segments) > 60
    found = estimate_camera(np.zeros((height, width), np.uint8), segments=segments)
    assert found is not None
    assert found.pitch_degrees == pytest.approx(pitch, abs=0.5)
    assert _fold(found.yaw_degrees - yaw) == pytest.approx(0.0, abs=0.5)
    assert found.fov_degrees == pytest.approx(fov, rel=0.02)


def test_a_rolled_camera_is_recovered_from_independent_segments():
    """Roll is the parameter whose sign was wrong until the lead caught it."""
    width, height, roll = 1280, 720, 3.0
    segments = _segments((0.5, 0.5, 0.5), 23.0, -14.0, roll, 70.0, width, height)
    assert len(segments) > 60
    found = estimate_camera(np.zeros((height, width), np.uint8), segments=segments)
    assert found is not None
    assert found.roll_degrees == pytest.approx(roll, abs=1.0)
    assert found.pitch_degrees == pytest.approx(-14.0, abs=1.0)


def test_a_cropped_independent_frame_moves_its_optical_centre():
    width, height = 1280, 720
    full = _segments((0.5, 0.5, 0.5), 23.0, -14.0, 0.0, 70.0, width, height)
    cut_x, cut_y = 200, 100
    kept = full - np.asarray([cut_x, cut_y, cut_x, cut_y], dtype=float)
    kept = kept[(kept[:, [0, 2]] >= 0).all(axis=1) & (kept[:, [1, 3]] >= 0).all(axis=1)]
    assert len(kept) > 40

    cropped_width, cropped_height = width - cut_x, height - cut_y
    found = estimate_camera(np.zeros((cropped_height, cropped_width), np.uint8),
                            segments=kept)
    assert found is not None
    assert found.principal_point is not None, "the crop went undetected"
    assert found.principal_point[0] == pytest.approx(width / 2.0 - cut_x,
                                                     abs=0.25 * cropped_width)


def test_a_crop_beyond_the_searchable_range_is_not_guessed_at():
    """The limit is deliberate, so the behaviour at its edge should be stated.

    `PRINCIPAL_LIMIT` bounds how far the optical centre may move, on the grounds
    that a harder crop is better handled by being told about it. A crop past that
    bound must not come back with a confidently wrong centre.
    """
    from mc_photogrammetry.reconstruction.calibration import PRINCIPAL_LIMIT

    width, height = 1280, 720
    full = _segments((0.5, 0.5, 0.5), 23.0, -14.0, 0.0, 70.0, width, height)
    cut_x, cut_y = 420, 240
    kept = full - np.asarray([cut_x, cut_y, cut_x, cut_y], dtype=float)
    kept = kept[(kept[:, [0, 2]] >= 0).all(axis=1) & (kept[:, [1, 3]] >= 0).all(axis=1)]
    if len(kept) < 40:
        pytest.skip("the hard crop left too little to fit")

    cropped_width = width - cut_x
    wanted = width / 2.0 - cut_x
    reach = PRINCIPAL_LIMIT * cropped_width
    assert abs(wanted - cropped_width / 2.0) > reach, "this crop is inside the limit"

    found = estimate_camera(np.zeros((height - cut_y, cropped_width), np.uint8),
                            segments=kept)
    assert found is not None
    if found.principal_point is not None:
        # If it does claim a centre it must at least be toward the true side,
        # never confidently the wrong way.
        claimed = found.principal_point[0] - cropped_width / 2.0
        assert claimed * (wanted - cropped_width / 2.0) >= 0.0


@pytest.mark.parametrize("width,height", [(1280, 720), (640, 360), (900, 900)])
def test_the_field_of_view_is_recovered_whatever_the_frame_shape(width, height):
    """Focal length in pixels follows the height; the field of view must not.

    A vertical field of view is a property of the camera and a focal length in
    pixels is a property of the sampling, so the same scene at three shapes must
    return three focal lengths and one field of view.
    """
    segments = _segments((0.5, 0.5, 0.5), 23.0, -14.0, 0.0, 70.0, width, height)
    if len(segments) < 40:
        pytest.skip("this frame shape did not yield enough segments")
    found = estimate_camera(np.zeros((height, width), np.uint8), segments=segments)
    assert found is not None
    assert found.fov_degrees == pytest.approx(70.0, rel=0.03)


def test_a_single_view_cannot_name_the_horizontal_axes():
    """The ambiguity is a property of the evidence, not a defect to be fixed.

    Two cameras a quarter turn apart see the same set of vanishing points, so
    the estimator must report the same yaw for both and leave the quadrant to
    something with more information.
    """
    width, height = 1280, 720
    results = []
    for yaw in (23.0, 113.0):
        segments = _segments((0.5, 0.5, 0.5), yaw, -14.0, 0.0, 70.0, width, height)
        if len(segments) < 40:
            pytest.skip("insufficient segments for the ambiguity check")
        found = estimate_camera(np.zeros((height, width), np.uint8),
                                segments=segments)
        assert found is not None
        results.append(found.yaw_degrees)
    assert _fold(results[0] - results[1]) == pytest.approx(0.0, abs=0.5)
