from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from mc_photogrammetry.domain.models import BlockObservation, BlockState, BlockVolume, ObservationKind
from .camera import CameraModel
from .scene import SceneTransform


@dataclass(frozen=True, slots=True)
class SfmSettings:
    max_features: int = 4000
    ratio_test: float = 0.72
    min_matches: int = 12
    max_reprojection_error: float = 8.0
    max_points: int = 100_000


def reconstruct_from_calibrated_views(
    frames: Mapping[str, np.ndarray],
    cameras: Mapping[str, CameraModel],
    *,
    scene: SceneTransform | None = None,
    palette: tuple[tuple[BlockState, tuple[int, int, int]], ...] = (),
    settings: SfmSettings = SfmSettings(),
) -> BlockVolume:
    """Triangulate stable features from calibrated Minecraft screenshots."""
    if len(frames) < 2:
        raise ValueError("at least two calibrated frames are required")
    missing = set(frames) - set(cameras)
    if missing:
        raise ValueError(f"missing camera poses for: {sorted(missing)}")
    if settings.min_matches < 1 or not 0.0 < settings.ratio_test < 1.0:
        raise ValueError("invalid SFM matching settings")

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("calibrated multi-view reconstruction requires opencv-python") from error

    names = [name for name in frames if name in cameras]
    first_image = _validate_frame(frames[names[0]])
    height, width = first_image.shape[:2]
    detector = cv2.SIFT_create(nfeatures=settings.max_features)
    keypoints: dict[str, list[object]] = {}
    descriptors: dict[str, np.ndarray] = {}
    for name in names:
        image = _validate_frame(frames[name])
        if image.shape[:2] != (height, width):
            raise ValueError("calibrated frames must have matching dimensions")
        feature_mask = np.full((height, width), 255, dtype=np.uint8)
        feature_mask[: int(height * 0.11), : int(width * 0.32)] = 0
        feature_mask[int(height * 0.88) :, :] = 0
        feature_mask[int(height * 0.45) : int(height * 0.88), : int(width * 0.28)] = 0
        center_x, center_y = width // 2, height // 2
        radius = max(4, int(min(width, height) * 0.02))
        feature_mask[center_y - radius : center_y + radius, center_x - radius : center_x + radius] = 0
        points, descriptor = detector.detectAndCompute(
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), feature_mask
        )
        if descriptor is not None:
            keypoints[name] = points
            descriptors[name] = descriptor

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    votes: dict[tuple[int, int, int], list[BlockState]] = defaultdict(list)
    view_counts: dict[tuple[int, int, int], set[str]] = defaultdict(set)
    pair_count = 0
    for left_index, left_name in enumerate(names):
        if left_name not in descriptors:
            continue
        for right_name in names[left_index + 1 :]:
            if right_name not in descriptors:
                continue
            pairs = matcher.knnMatch(descriptors[left_name], descriptors[right_name], k=2)
            good = [
                pair[0]
                for pair in pairs
                if len(pair) == 2 and pair[0].distance < settings.ratio_test * pair[1].distance
            ]
            if len(good) < settings.min_matches:
                continue
            pair_count += 1
            left_pixels = np.float64([keypoints[left_name][match.queryIdx].pt for match in good]).T
            right_pixels = np.float64([keypoints[right_name][match.trainIdx].pt for match in good]).T
            points = _triangulate(left_pixels, right_pixels, cameras[left_name], cameras[right_name], width, height)
            for match_index, (point, match) in enumerate(zip(points, good)):
                if not np.all(np.isfinite(point)):
                    continue
                if scene is not None:
                    local = tuple(int(round(value)) for value in scene.to_local(tuple(point)))
                    if not all(0 <= local[index] < scene.size[index] for index in range(3)):
                        continue
                else:
                    local = tuple(int(round(value)) for value in point)
                if (
                    _reprojection_error(
                        point, left_pixels[:, match_index], cameras[left_name], width, height
                    )
                    > settings.max_reprojection_error
                    or _reprojection_error(
                        point, right_pixels[:, match_index], cameras[right_name], width, height
                    )
                    > settings.max_reprojection_error
                ):
                    continue
                state = _classify_pixel(
                    frames[left_name],
                    tuple(int(round(value)) for value in left_pixels[:, match_index]),
                    palette,
                )
                if state is not None:
                    votes[local].append(state)
                view_counts[local].update((left_name, right_name))
                if len(votes) >= settings.max_points:
                    break
            if len(votes) >= settings.max_points:
                break
        if len(votes) >= settings.max_points:
            break

    volume = BlockVolume()
    for coordinate, states in votes.items():
        state = max(set(states), key=states.count)
        support = len(view_counts[coordinate])
        confidence = min(1.0, 0.35 + 0.15 * len(states) + 0.05 * support)
        volume.set(
            coordinate,
            BlockObservation(
                state=state,
                confidence=confidence,
                kind=ObservationKind.OBSERVED,
                frame_ids=sorted(view_counts[coordinate]),
            ),
        )
    if pair_count == 0:
        raise RuntimeError("no calibrated image pairs produced enough stable matches")
    return volume


def _triangulate(
    left: np.ndarray,
    right: np.ndarray,
    left_camera: CameraModel,
    right_camera: CameraModel,
    width: int,
    height: int,
) -> np.ndarray:
    import cv2

    left_projection = _projection_matrix(left_camera, width, height)
    right_projection = _projection_matrix(right_camera, width, height)
    homogeneous = cv2.triangulatePoints(left_projection, right_projection, left, right)
    return (homogeneous[:3] / homogeneous[3]).T


def _projection_matrix(camera: CameraModel, width: int, height: int) -> np.ndarray:
    import math

    yaw = math.radians(camera.yaw_degrees)
    pitch = math.radians(camera.pitch_degrees)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rotation = np.array(
        [[cy, 0.0, sy], [sp * sy, -cp, -sp * cy], [-cp * sy, -sp, cp * cy]],
        dtype=np.float64,
    )
    eye = np.asarray(camera.position, dtype=np.float64) + np.array([0.0, camera.eye_height, 0.0])
    translation = -(rotation @ eye).reshape(3, 1)
    focal = (width / 2.0) / math.tan(math.radians(camera.fov_degrees) / 2.0)
    intrinsics = np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])
    return intrinsics @ np.hstack((rotation, translation))


def _reprojection_error(
    point: np.ndarray, pixel: np.ndarray, camera: CameraModel, width: int, height: int
) -> float:
    projection = _projection_matrix(camera, width, height) @ np.append(point, 1.0)
    if projection[2] <= 0:
        return float("inf")
    projected = projection[:2] / projection[2]
    return float(np.linalg.norm(projected - pixel))


def _classify_pixel(
    image: np.ndarray,
    pixel: tuple[int, int],
    palette: tuple[tuple[BlockState, tuple[int, int, int]], ...],
) -> BlockState | None:
    x, y = pixel
    if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
        return None
    if not palette:
        return BlockState("minecraft:stone")
    rgb = image[y, x].astype(np.float32)
    return min(palette, key=lambda item: float(np.linalg.norm(rgb - np.asarray(item[1], dtype=np.float32))))[0]


def _validate_frame(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("calibrated frames must be RGB uint8 arrays")
    return image
