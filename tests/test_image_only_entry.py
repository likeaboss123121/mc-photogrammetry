"""Tests for the image-only single-view camera seam (DS-9).

The seam is a thin adapter over the existing M3 calibrator, so every test here
monkeypatches ``calibration.estimate_camera`` rather than running the real
solver: independent delegation, RGB-to-grey conversion, and default GUI
zero-fill are what it owns.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from mc_photogrammetry.pipeline.contracts import (
    CameraEstimate,
    CameraIntrinsics,
    CameraPose,
    CameraSolution,
)
from mc_photogrammetry.pipeline.image_only import (
    LOCAL_FRAME_ORIGIN,
    ImageOnlyCameraError,
    camera_model_from_estimate,
    solve_image_cameras,
)
from mc_photogrammetry.reconstruction.calibration import CalibrationResult
from mc_photogrammetry.vision.gui_mask import create_gui_mask


@pytest.fixture(autouse=True)
def _forbid_shared_calibration(monkeypatch):
    from mc_photogrammetry.reconstruction import calibration

    def forbidden(*args, **kwargs):
        pytest.fail("image-only seam used shared-focal calibration")

    monkeypatch.setattr(calibration, "estimate_session", forbidden)


def _estimate(
    *,
    image_size: tuple[int, int] = (2000, 1000),
    focal: tuple[float, float] = (900.0, 880.0),
    principal: tuple[float, float] = (1050.0, 480.0),
    yaw: float = 14.0,
    pitch: float = 17.0,
    roll: float = 3.5,
) -> CameraEstimate:
    return CameraEstimate(
        intrinsics=CameraIntrinsics(image_size, focal, principal),
        pose=CameraPose(
            position=None, yaw_degrees=yaw, pitch_degrees=pitch, roll_degrees=roll
        ),
        residual_pixels=0.4,
    )


def test_the_adapter_carries_every_measured_optical_value():
    estimate = _estimate(
        image_size=(2000, 1000), focal=(900.0, 880.0), principal=(1050.0, 480.0),
        yaw=14.0, pitch=17.0, roll=3.5,
    )

    camera = camera_model_from_estimate(estimate, position=LOCAL_FRAME_ORIGIN)

    assert camera.position == LOCAL_FRAME_ORIGIN
    assert camera.intrinsics is estimate.intrinsics, "intrinsics were rebuilt, not carried"
    assert camera.intrinsics.image_size == (2000, 1000)
    assert camera.intrinsics.focal_length_pixels == (900.0, 880.0)
    assert camera.intrinsics.principal_point_pixels == (1050.0, 480.0)
    assert camera.yaw_degrees == pytest.approx(14.0)
    assert camera.pitch_degrees == pytest.approx(17.0)
    assert camera.roll_degrees == pytest.approx(3.5)


def test_the_adapter_derives_vertical_fov_from_fy_and_image_height():
    # Minecraft's FOV setting is vertical, so a focal length on the width would
    # stretch the render by the aspect ratio.
    estimate = _estimate(image_size=(2560, 1000), focal=(1600.0, 800.0))

    camera = camera_model_from_estimate(estimate, position=LOCAL_FRAME_ORIGIN)

    assert camera.fov_degrees == pytest.approx(
        math.degrees(2.0 * math.atan(1000 / 2.0 / 800.0))
    )
    assert camera.fov_degrees != pytest.approx(
        math.degrees(2.0 * math.atan(2560 / 2.0 / 1600.0))
    )


def test_the_adapter_requires_an_explicit_frame_origin():
    # Not defaulted to LOCAL_FRAME_ORIGIN: the caller must state that this is a
    # local frame rather than a recovered world position.
    with pytest.raises(TypeError):
        camera_model_from_estimate(_estimate())  # type: ignore[call-arg]


def test_a_pose_less_estimate_is_rejected():
    estimate = CameraEstimate(
        intrinsics=CameraIntrinsics((100, 100), (50.0, 50.0), (50.0, 50.0)),
        pose=None,
    )

    with pytest.raises(ImageOnlyCameraError, match="pose"):
        camera_model_from_estimate(estimate, position=LOCAL_FRAME_ORIGIN)


def test_an_estimate_without_intrinsics_is_rejected():
    estimate = CameraEstimate(
        intrinsics=None,  # type: ignore[arg-type]
        pose=CameraPose(position=None, yaw_degrees=0.0, pitch_degrees=0.0),
    )

    with pytest.raises(ImageOnlyCameraError, match="intrinsics"):
        camera_model_from_estimate(estimate, position=LOCAL_FRAME_ORIGIN)


def test_solving_converts_rgb_to_grey_and_delegates_to_the_m3_calibrator(monkeypatch):
    from mc_photogrammetry.reconstruction import calibration

    rgb = np.zeros((40, 60, 3), dtype=np.uint8)
    rgb[:, :20] = (255, 0, 0)
    rgb[:, 20:40] = (0, 255, 0)
    rgb[:, 40:] = (0, 0, 255)
    seen: dict[str, np.ndarray] = {}

    def fake_estimate_camera(grey):
        assert not seen, "the frame was estimated more than once"
        seen["frame.png"] = grey
        return CalibrationResult(
            focal_pixels=100.0, yaw_degrees=14.0, pitch_degrees=17.0,
            width=60, height=40, inliers=30, segments=40,
            median_residual=0.4, roll_degrees=3.5, principal_point=(41.0, 12.0),
        )

    monkeypatch.setattr(calibration, "estimate_camera", fake_estimate_camera)

    solution = solve_image_cameras({"frame.png": rgb})

    grey = seen["frame.png"]
    assert grey.shape == (40, 60)
    assert grey.dtype == np.uint8
    # Luminance weighted, not a flat channel mean (which would be 85 everywhere).
    assert 74 <= int(grey[5, 5]) <= 78
    assert 147 <= int(grey[5, 30]) <= 152
    assert 27 <= int(grey[5, 50]) <= 31

    assert isinstance(solution, CameraSolution)
    assert set(solution.cameras) == {"frame.png"}
    estimate = solution.cameras["frame.png"]
    assert estimate.pose is not None and estimate.pose.position is None
    assert estimate.intrinsics.principal_point_pixels == (41.0, 12.0)
    assert solution.scale_known is False
    for wanted in ("position", "scale", "yaw quadrant"):
        assert any(wanted in note for note in solution.unresolved), wanted


def test_an_already_grey_frame_is_zero_filled_without_mutating_input(monkeypatch):
    from mc_photogrammetry.reconstruction import calibration

    grey = np.full((20, 20), 100, dtype=np.uint8)
    seen = []
    monkeypatch.setattr(
        calibration, "estimate_camera", lambda frame: seen.append(frame),
    )

    solution = solve_image_cameras({"frame.png": grey})

    assert solution == CameraSolution(cameras={}, components=(), unresolved=(), scale_known=False)
    assert len(seen) == 1
    excluded = create_gui_mask(20, 20)
    assert seen[0] is not grey
    np.testing.assert_array_equal(seen[0][~excluded], grey[~excluded])
    assert np.all(seen[0][excluded] == 0)
    assert np.all(grey == 100)


def test_excluded_gui_rgb_pixels_do_not_affect_estimator_input_or_camera_solution(monkeypatch):
    from mc_photogrammetry.reconstruction import calibration

    base = np.full((100, 160, 3), 100, dtype=np.uint8)
    changed = base.copy()
    excluded = create_gui_mask(100, 160)
    changed[excluded] = (20, 220, 55)
    original_changed = changed.copy()
    seen = []

    def estimate(grey):
        seen.append(grey.copy())
        return CalibrationResult(
            focal_pixels=float(grey.mean()), yaw_degrees=14.0, pitch_degrees=17.0,
            width=160, height=100, inliers=30, segments=40, median_residual=0.4,
        )

    monkeypatch.setattr(calibration, "estimate_camera", estimate)
    base_solution = solve_image_cameras({"frame.png": base})
    changed_solution = solve_image_cameras({"frame.png": changed})

    assert len(seen) == 2
    np.testing.assert_array_equal(seen[0], seen[1])
    assert np.all(seen[0][excluded] == 0)
    assert np.all(seen[0][~excluded] == 100)
    assert set(base_solution.cameras) == {"frame.png"}
    assert base_solution == changed_solution
    assert np.all(base == 100)
    np.testing.assert_array_equal(changed, original_changed)


def test_an_empty_frame_mapping_is_rejected():
    with pytest.raises(ImageOnlyCameraError, match="at least one"):
        solve_image_cameras({})


@pytest.mark.parametrize("frame_count", [1, 2], ids=["single", "multiple"])
def test_all_failed_fits_return_an_empty_solution(monkeypatch, frame_count):
    from mc_photogrammetry.reconstruction import calibration

    seen = []
    monkeypatch.setattr(calibration, "estimate_camera", lambda frame: seen.append(frame))
    frames = {f"frame-{index}": np.zeros((20, 20), dtype=np.uint8)
              for index in range(frame_count)}

    solution = solve_image_cameras(frames)

    assert isinstance(solution, CameraSolution)
    assert solution.cameras == {}
    assert solution.components == ()
    assert solution.unresolved == ()
    assert solution.scale_known is False
    assert len(seen) == frame_count
    for actual, expected in zip(seen, frames.values()):
        np.testing.assert_array_equal(actual, expected)
        assert actual is not expected


def test_non_uint8_frames_are_rejected_before_any_solve(monkeypatch):
    from mc_photogrammetry.reconstruction import calibration

    def unexpected_estimate(frame):
        pytest.fail("invalid input reached the estimator")

    monkeypatch.setattr(calibration, "estimate_camera", unexpected_estimate)
    with pytest.raises(ImageOnlyCameraError, match="uint8"):
        solve_image_cameras({"frame.png": np.zeros((20, 20, 3), dtype=np.float32)})


def test_a_frame_that_is_not_rgb_or_grey_is_rejected(monkeypatch):
    from mc_photogrammetry.reconstruction import calibration

    def unexpected_estimate(frame):
        pytest.fail("invalid input reached the estimator")

    monkeypatch.setattr(calibration, "estimate_camera", unexpected_estimate)
    with pytest.raises(ImageOnlyCameraError, match="RGB"):
        solve_image_cameras({"frame.png": np.zeros((20, 20, 4), dtype=np.uint8)})


@pytest.mark.parametrize("order", [("first", "second"), ("second", "first")],
                         ids=["forward", "reverse"])
def test_equal_size_frames_keep_independent_focals_and_do_not_change_each_other(monkeypatch, order):
    from mc_photogrammetry.reconstruction import calibration

    frames = {"first": np.full((40, 60), 10, dtype=np.uint8),
              "second": np.full((40, 60), 20, dtype=np.uint8)}
    results = {
        frame_id: CalibrationResult(
            focal_pixels=focal, yaw_degrees=14.0, pitch_degrees=17.0,
            width=60, height=40, inliers=30, segments=40, median_residual=0.4,
        )
        for frame_id, focal in (("first", 90.0), ("second", 160.0))
    }
    seen = []

    def independent_estimate(grey):
        excluded = create_gui_mask(*grey.shape)
        frame_id = next(name for name, frame in frames.items()
                        if np.array_equal(grey[~excluded], frame[~excluded]))
        assert np.all(grey[excluded] == 0)
        seen.append(frame_id)
        return results[frame_id]

    monkeypatch.setattr(calibration, "estimate_camera", independent_estimate)
    together = solve_image_cameras({name: frames[name] for name in order})
    assert seen == list(order)
    assert set(together.cameras) == set(frames)
    assert together.scale_known is False
    for frame_id, focal in (("first", 90.0), ("second", 160.0)):
        single = solve_image_cameras({frame_id: frames[frame_id]})
        assert together.cameras[frame_id] == single.cameras[frame_id]
        assert together.cameras[frame_id].intrinsics.focal_length_pixels == (focal, focal)
        assert together.cameras[frame_id].pose.position is None
        assert tuple(note for note in together.unresolved if note.startswith(frame_id + ":")) == single.unresolved
    assert seen == list(order) + ["first", "second"]


def test_partial_abstention_keeps_only_the_successful_frame_and_exact_estimate(monkeypatch):
    from mc_photogrammetry.reconstruction import calibration

    unavailable = np.zeros((40, 60), dtype=np.uint8)
    available = np.ones((40, 60), dtype=np.uint8)
    result = CalibrationResult(
        focal_pixels=120.0, yaw_degrees=14.0, pitch_degrees=17.0,
        width=60, height=40, inliers=30, segments=40, median_residual=0.4,
    )
    seen = []
    deliveries = []
    original_adapter = calibration.to_camera_solution

    def independent_estimate(grey):
        seen.append(grey)
        excluded = create_gui_mask(*grey.shape)
        assert np.all(grey[excluded] == 0)
        if np.array_equal(grey[~excluded], unavailable[~excluded]):
            return None
        np.testing.assert_array_equal(grey[~excluded], available[~excluded])
        return result

    def capture_adapter(results):
        assert set(results) == {"available"}
        assert results["available"] is result
        deliveries.append(results)
        return original_adapter(results)

    monkeypatch.setattr(calibration, "estimate_camera", independent_estimate)
    monkeypatch.setattr(calibration, "to_camera_solution", capture_adapter)
    solution = solve_image_cameras({"unavailable": unavailable, "available": available})

    assert len(seen) == 2
    excluded = create_gui_mask(40, 60)
    np.testing.assert_array_equal(seen[0][~excluded], unavailable[~excluded])
    np.testing.assert_array_equal(seen[1][~excluded], available[~excluded])
    assert np.all(unavailable == 0) and np.all(available == 1)
    assert len(deliveries) == 1
    assert set(solution.cameras) == {"available"}
    assert solution.cameras["available"].intrinsics.focal_length_pixels == (120.0, 120.0)
    assert solution.cameras["available"].pose.position is None
    assert solution.scale_known is False
    assert all(note.startswith("available:") for note in solution.unresolved)
    for wanted in ("position", "scale", "yaw quadrant"):
        assert any(wanted in note for note in solution.unresolved)
