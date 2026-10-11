"""Image-only single-view camera recovery: pixels in, a real camera out.

This is the production seam DS-8 identified as missing. Before it, the normal
CLI single-view path had no camera it could trust: it either needed a supplied
manifest or fell through to `reconstruct_facade`'s pixel-grid body, which turns
image columns and rows into lattice coordinates with no focal length, pose, or
scale.

This module does one thing. It hands the frame to the existing M3 calibrator
(:func:`calibration.estimate_camera`, independently per frame and manifest-free),
carries the answer through the shared :class:`CameraSolution` contract, and
adapts each solved estimate to the :class:`CameraModel` the reconstruction
expects. No calibration algorithm is reimplemented and no manifest, F3/debug
value, coordinate, direction, or bounds are read.

What one view cannot establish stays explicitly unresolved and is never
invented:

* **Translation.** Vanishing points are unchanged by moving along a ray, so an
  adapted camera sits at :data:`LOCAL_FRAME_ORIGIN` - a local frame, not a
  recovered world position.
* **Scale.** Pixel focal length and orientation do not name a lattice period,
  so ``CameraSolution.scale_known`` stays false.
* **The yaw quadrant.** One view cannot tell the two horizontal lattice axes
  apart, so yaw is meaningful only modulo 90 degrees.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from mc_photogrammetry.pipeline.contracts import CameraEstimate, CameraSolution
from mc_photogrammetry.reconstruction import calibration
from mc_photogrammetry.reconstruction.camera import CameraModel
from mc_photogrammetry.vision.gui_mask import create_gui_mask


LOCAL_FRAME_ORIGIN: tuple[float, float, float] = (0.0, 0.0, 0.0)
"""The only position image-only recovery may put on a camera.

One view fixes orientation and focal length, never translation. Callers must
pass this origin explicitly so no code path can pass it off as a recovered
world position.
"""


class ImageOnlyCameraError(RuntimeError):
    """Image-only camera recovery could not produce a usable camera."""


def solve_image_cameras(frames: Mapping[str, np.ndarray]) -> CameraSolution:
    """Recover usable cameras from image evidence, retaining estimator abstention.

    Delegates independently to :func:`calibration.estimate_camera` for each
    frame and carries successful estimates through
    :func:`calibration.to_camera_solution`. No shared focal or fixed-focal
    second pass is assumed. Frames are converted to the single-channel
    ``uint8`` image the calibrator expects, with default GUI exclusions
    zero-filled before fitting. ``None`` remains a per-frame
    abstention, including an empty :class:`CameraSolution` when every frame
    abstains so session construction can retain typed exclusions. Raises
    :class:`ImageOnlyCameraError` for an empty mapping or malformed inputs.
    """
    if not frames:
        raise ImageOnlyCameraError("image-only camera recovery needs at least one frame")
    grayscale = {frame_id: _to_grayscale(image) for frame_id, image in frames.items()}
    results = {}
    for frame_id, grey in grayscale.items():
        excluded = create_gui_mask(grey.shape[0], grey.shape[1])
        grey = grey.copy()
        grey[excluded] = 0
        result = calibration.estimate_camera(grey)
        if result is not None:
            results[frame_id] = result
    if not results:
        return CameraSolution(cameras={}, components=(), unresolved=(), scale_known=False)
    return calibration.to_camera_solution(results)


def camera_model_from_estimate(
    estimate: CameraEstimate,
    *,
    position: tuple[float, float, float],
) -> CameraModel:
    """Adapt a solved :class:`CameraEstimate` to the renderer's camera.

    Measured optics travel intact: explicit intrinsics (image size, focal
    length, and principal point including a crop-shifted centre), yaw, pitch and
    roll. The vertical field of view is derived from the focal length on the
    image height, because Minecraft's FOV setting is vertical. ``position`` is
    required so the caller states the frame origin rather than a translation
    being inferred; a missing pose, missing intrinsics, or unusable values raise
    :class:`ImageOnlyCameraError`.
    """
    intrinsics = estimate.intrinsics
    if intrinsics is None:
        raise ImageOnlyCameraError("camera estimate has no intrinsics")
    pose = estimate.pose
    if pose is None:
        raise ImageOnlyCameraError("camera estimate has no pose")
    width, height = intrinsics.image_size
    _focal_x, focal_y = intrinsics.focal_length_pixels
    if width <= 0 or height <= 0 or focal_y <= 0.0:
        raise ImageOnlyCameraError("camera estimate intrinsics are not usable")
    fov_degrees = math.degrees(2.0 * math.atan(height / 2.0 / focal_y))
    try:
        return CameraModel(
            position=position,
            yaw_degrees=pose.yaw_degrees,
            pitch_degrees=pose.pitch_degrees,
            fov_degrees=fov_degrees,
            intrinsics=intrinsics,
            roll_degrees=pose.roll_degrees,
        )
    except ValueError as error:
        raise ImageOnlyCameraError(f"camera estimate is not a usable camera: {error}") from error


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """RGB (or already-grey) frame to the single-channel ``uint8`` calibrator input."""
    frame = np.asarray(image)
    if frame.dtype != np.uint8:
        raise ImageOnlyCameraError("image-only camera recovery expects uint8 frames")
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 3:
        import cv2

        # The shared SfM path converts RGB this way; reusing it keeps the
        # calibrator's input convention in one place.
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    raise ImageOnlyCameraError(
        f"expected an RGB frame with shape (height, width, 3), got {frame.shape}"
    )
