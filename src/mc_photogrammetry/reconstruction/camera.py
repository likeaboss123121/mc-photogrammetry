from __future__ import annotations

from dataclasses import dataclass
import math
import json
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class CameraModel:
    """Minecraft-style pinhole camera in block/world coordinates."""

    position: tuple[float, float, float]
    yaw_degrees: float
    pitch_degrees: float
    fov_degrees: float = 70.0
    eye_height: float = 1.62

    def __post_init__(self) -> None:
        if not 1.0 <= self.fov_degrees < 180.0:
            raise ValueError("fov_degrees must be between 1 and 180")
        if not -90.0 <= self.pitch_degrees <= 90.0:
            raise ValueError("pitch_degrees must be between -90 and 90")

    def to_dict(self) -> dict[str, object]:
        return {
            "position": list(self.position),
            "yaw_degrees": self.yaw_degrees,
            "pitch_degrees": self.pitch_degrees,
            "fov_degrees": self.fov_degrees,
            "eye_height": self.eye_height,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CameraModel:
        position = tuple(float(value) for value in data["position"])  # type: ignore[union-attr]
        if len(position) != 3:
            raise ValueError("camera position must contain three values")
        return cls(
            position=position,
            yaw_degrees=float(data["yaw_degrees"]),
            pitch_degrees=float(data["pitch_degrees"]),
            fov_degrees=float(data.get("fov_degrees", 110.0)),
            eye_height=float(data.get("eye_height", 1.62)),
        )


def load_camera_manifest(path: str | Path) -> dict[str, CameraModel]:
    """Load screenshot-name to Minecraft camera pose mappings."""
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("camera manifest root must be an object")
    cameras: dict[str, CameraModel] = {}
    for filename, value in data.items():
        if not isinstance(value, dict):
            raise ValueError(f"camera entry for {filename} must be an object")
        cameras[str(filename)] = CameraModel.from_dict(value)
    return cameras


def _focal_length(fov_degrees: float, height: int) -> float:
    """Focal length in pixels for a Minecraft field of view.

    Minecraft's FOV setting is the *vertical* field of view - it goes to
    ``Matrix4f.perspective`` as the fovy - so the focal length follows the image
    height, not its width. Deriving it from the width instead stretches the
    render by the aspect ratio: on a 2560x1369 frame that is 1.87x too long,
    which is exact at the centre of frame and drifts further out towards the
    edges, so it survives a casual look at a wireframe overlay.
    """
    return (height / 2.0) / math.tan(math.radians(fov_degrees) / 2.0)


def camera_space_points(
    points: Iterable[tuple[float, float, float]], camera: CameraModel
) -> np.ndarray:
    """Transform world points into the shared Minecraft camera convention.

    Columns are screen-right, screen-up, and positive depth.  Keeping this
    transform separate from raster projection makes SfM, ray casting, and the
    renderer use the same handedness.
    """
    values = np.asarray(list(points), dtype=np.float64)
    if values.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    values = values.reshape(-1, 3)
    yaw = math.radians(camera.yaw_degrees)
    pitch = math.radians(camera.pitch_degrees)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    cos_pitch, sin_pitch = math.cos(pitch), math.sin(pitch)
    eye = np.asarray(camera.position, dtype=np.float64) + np.asarray([0.0, camera.eye_height, 0.0])
    dx, dy, dz = (values - eye).T
    horizontal = -dx * sin_yaw + dz * cos_yaw
    return np.stack(
        (
            -(dx * cos_yaw + dz * sin_yaw),
            dy * cos_pitch + horizontal * sin_pitch,
            horizontal * cos_pitch - dy * sin_pitch,
        ),
        axis=1,
    )


def camera_projection_matrix(camera: CameraModel, *, width: int, height: int) -> np.ndarray:
    """Return a homogeneous matrix that exactly matches :func:`project_points`."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    yaw = math.radians(camera.yaw_degrees)
    pitch = math.radians(camera.pitch_degrees)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    cos_pitch, sin_pitch = math.cos(pitch), math.sin(pitch)
    rotation = np.asarray(
        (
            (-cos_yaw, 0.0, -sin_yaw),
            (-sin_pitch * sin_yaw, cos_pitch, sin_pitch * cos_yaw),
            (-cos_pitch * sin_yaw, -sin_pitch, cos_pitch * cos_yaw),
        ),
        dtype=np.float64,
    )
    eye = np.asarray(camera.position, dtype=np.float64) + np.asarray([0.0, camera.eye_height, 0.0])
    extrinsics = np.hstack((rotation, (-(rotation @ eye)).reshape(3, 1)))
    focal = _focal_length(camera.fov_degrees, height)
    intrinsics = np.asarray(
        ((focal, 0.0, width / 2.0), (0.0, -focal, height / 2.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    return intrinsics @ extrinsics


def project_clipped_polygon(
    points: Iterable[tuple[float, float, float]],
    camera: CameraModel,
    *,
    width: int,
    height: int,
    near_clip: float = 1e-3,
) -> np.ndarray:
    """Project the part of a convex world polygon in front of the near plane.

    Returning a polygon instead of dropping a face with one vertex behind the
    lens prevents foreground walls from leaking the far geometry into a renderer.
    Each returned row is ``(pixel_x, pixel_y, camera_depth)``.
    """
    if near_clip <= 0:
        raise ValueError("near_clip must be positive")
    polygon = camera_space_points(points, camera)
    if len(polygon) < 3:
        return np.empty((0, 3), dtype=np.float64)
    kept: list[np.ndarray] = []
    for index, current in enumerate(polygon):
        following = polygon[(index + 1) % len(polygon)]
        here = current[2] >= near_clip
        there = following[2] >= near_clip
        if here:
            kept.append(current)
        if here != there:
            fraction = (near_clip - current[2]) / (following[2] - current[2])
            kept.append(current + fraction * (following - current))
    if len(kept) < 3:
        return np.empty((0, 3), dtype=np.float64)
    values = np.asarray(kept, dtype=np.float64)
    focal = _focal_length(camera.fov_degrees, height)
    return np.column_stack((
        width / 2.0 + values[:, 0] * focal / values[:, 2],
        height / 2.0 - values[:, 1] * focal / values[:, 2],
        values[:, 2],
    ))


def project_points(
    points: Iterable[tuple[float, float, float]],
    camera: CameraModel,
    *,
    width: int,
    height: int,
) -> list[tuple[float, float, float] | None]:
    """Project world points to pixels as ``(x, y, camera_depth)``."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    focal = _focal_length(camera.fov_degrees, height)
    projected: list[tuple[float, float, float] | None] = []
    for view_x, rotated_y, depth in camera_space_points(points, camera):
        if depth <= 1e-6:
            projected.append(None)
            continue
        projected.append(
            (
                width / 2.0 + view_x * focal / depth,
                height / 2.0 - rotated_y * focal / depth,
                depth,
            )
        )
    return projected


def unproject_pixel(
    pixel_x: float,
    pixel_y: float,
    camera: CameraModel,
    *,
    width: int,
    height: int,
    depth: float,
) -> tuple[float, float, float]:
    """Return a world point at a given camera-space depth."""
    if width <= 0 or height <= 0 or depth <= 0:
        raise ValueError("width, height, and depth must be positive")
    focal = _focal_length(camera.fov_degrees, height)
    view_x = -(pixel_x - width / 2.0) * depth / focal
    rotated_y = -(pixel_y - height / 2.0) * depth / focal
    yaw = math.radians(camera.yaw_degrees)
    pitch = math.radians(camera.pitch_degrees)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    cos_pitch, sin_pitch = math.cos(pitch), math.sin(pitch)
    horizontal_depth = rotated_y * sin_pitch + depth * cos_pitch
    view_y = rotated_y * cos_pitch - depth * sin_pitch
    world_x = view_x * cos_yaw - horizontal_depth * sin_yaw + camera.position[0]
    world_y = view_y + camera.position[1] + camera.eye_height
    world_z = view_x * sin_yaw + horizontal_depth * cos_yaw + camera.position[2]
    return world_x, world_y, world_z
