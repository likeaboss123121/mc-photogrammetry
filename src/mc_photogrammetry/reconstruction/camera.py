from __future__ import annotations

from dataclasses import dataclass
import math
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from mc_photogrammetry.pipeline.contracts import CameraIntrinsics, FrameTransform


@dataclass(frozen=True, slots=True)
class CameraModel:
    """Minecraft-style pinhole camera in block/world coordinates."""

    position: tuple[float, float, float]
    yaw_degrees: float
    pitch_degrees: float
    fov_degrees: float = 70.0
    eye_height: float = 1.62
    intrinsics: CameraIntrinsics | None = None
    roll_degrees: float = 0.0

    def __post_init__(self) -> None:
        if not 1.0 <= self.fov_degrees < 180.0:
            raise ValueError("fov_degrees must be between 1 and 180")
        if not -90.0 <= self.pitch_degrees <= 90.0:
            raise ValueError("pitch_degrees must be between -90 and 90")
        if self.intrinsics is not None:
            width, height = self.intrinsics.image_size
            focal_x, focal_y = self.intrinsics.focal_length_pixels
            if width <= 0 or height <= 0 or focal_x <= 0 or focal_y <= 0:
                raise ValueError("camera intrinsics must have positive dimensions and focal lengths")

    def to_dict(self) -> dict[str, object]:
        return {
            "position": list(self.position),
            "yaw_degrees": self.yaw_degrees,
            "pitch_degrees": self.pitch_degrees,
            "fov_degrees": self.fov_degrees,
            "eye_height": self.eye_height,
            "roll_degrees": self.roll_degrees,
            "intrinsics": None if self.intrinsics is None else {
                "image_size": list(self.intrinsics.image_size),
                "focal_length_pixels": list(self.intrinsics.focal_length_pixels),
                "principal_point_pixels": list(self.intrinsics.principal_point_pixels),
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CameraModel:
        position = tuple(float(value) for value in data["position"])  # type: ignore[union-attr]
        if len(position) != 3:
            raise ValueError("camera position must contain three values")
        raw_intrinsics = data.get("intrinsics")
        intrinsics = None
        if isinstance(raw_intrinsics, dict):
            intrinsics = CameraIntrinsics(
                tuple(int(value) for value in raw_intrinsics["image_size"]),  # type: ignore[index]
                tuple(float(value) for value in raw_intrinsics["focal_length_pixels"]),  # type: ignore[index]
                tuple(float(value) for value in raw_intrinsics["principal_point_pixels"]),  # type: ignore[index]
            )
        return cls(
            position=position,
            yaw_degrees=float(data["yaw_degrees"]),
            pitch_degrees=float(data["pitch_degrees"]),
            fov_degrees=float(data.get("fov_degrees", 110.0)),
            eye_height=float(data.get("eye_height", 1.62)),
            intrinsics=intrinsics,
            roll_degrees=float(data.get("roll_degrees", 0.0)),
        )

    def transformed(self, transform: FrameTransform) -> "CameraModel":
        """Return this camera calibrated for a cropped/resized derived frame."""
        return CameraModel(
            self.position, self.yaw_degrees, self.pitch_degrees, self.fov_degrees, self.eye_height,
            intrinsics_for(self, *transform.source_size).transformed(transform), self.roll_degrees,
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


def intrinsics_for(camera: CameraModel, width: int, height: int) -> CameraIntrinsics:
    """Explicit intrinsics at the requested frame size, scaling a stored calibration."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if camera.intrinsics is None:
        focal = _focal_length(camera.fov_degrees, height)
        return CameraIntrinsics((width, height), (focal, focal), (width / 2.0, height / 2.0))
    base_width, base_height = camera.intrinsics.image_size
    scale_x, scale_y = width / base_width, height / base_height
    return CameraIntrinsics(
        (width, height),
        (camera.intrinsics.focal_length_pixels[0] * scale_x, camera.intrinsics.focal_length_pixels[1] * scale_y),
        (camera.intrinsics.principal_point_pixels[0] * scale_x, camera.intrinsics.principal_point_pixels[1] * scale_y),
    )


def camera_rotation_matrix(camera: CameraModel) -> np.ndarray:
    """World-to-camera rotation with rows right, up, and forward."""
    yaw, pitch = math.radians(camera.yaw_degrees), math.radians(camera.pitch_degrees)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    cos_pitch, sin_pitch = math.cos(pitch), math.sin(pitch)
    rotation = np.asarray((
        (-cos_yaw, 0.0, -sin_yaw),
        (-sin_pitch * sin_yaw, cos_pitch, sin_pitch * cos_yaw),
        (-cos_pitch * sin_yaw, -sin_pitch, cos_pitch * cos_yaw),
    ), dtype=np.float64)
    roll = math.radians(camera.roll_degrees)
    roll_rotation = np.asarray(((math.cos(roll), -math.sin(roll), 0.0),
                                (math.sin(roll), math.cos(roll), 0.0),
                                (0.0, 0.0, 1.0)), dtype=np.float64)
    return roll_rotation @ rotation


def camera_eye(camera: CameraModel) -> np.ndarray:
    return np.asarray(camera.position, dtype=np.float64) + np.asarray([0.0, camera.eye_height, 0.0])


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
    return (camera_rotation_matrix(camera) @ (values - camera_eye(camera)).T).T


def camera_projection_matrix(camera: CameraModel, *, width: int, height: int) -> np.ndarray:
    """Return a homogeneous matrix that exactly matches :func:`project_points`."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    rotation = camera_rotation_matrix(camera)
    eye = camera_eye(camera)
    extrinsics = np.hstack((rotation, (-(rotation @ eye)).reshape(3, 1)))
    calibration = intrinsics_for(camera, width, height)
    focal_x, focal_y = calibration.focal_length_pixels
    principal_x, principal_y = calibration.principal_point_pixels
    intrinsics = np.asarray(
        ((focal_x, 0.0, principal_x), (0.0, -focal_y, principal_y), (0.0, 0.0, 1.0)),
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
    projected, _uv = project_clipped_polygon_uv(
        points, camera, width=width, height=height, near_clip=near_clip
    )
    return projected


def project_clipped_polygon_uv(
    points: Iterable[tuple[float, float, float]],
    camera: CameraModel,
    *,
    width: int,
    height: int,
    uv: np.ndarray | None = None,
    near_clip: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a clipped convex polygon while carrying perspective texture UVs.

    Near-plane intersections are new polygon vertices, not new texture corners.
    Interpolating their UVs before projection keeps cut-out masks aligned on
    triangles, quads, and pentagons created by clipping.
    """
    if near_clip <= 0:
        raise ValueError("near_clip must be positive")
    polygon = camera_space_points(points, camera)
    if len(polygon) < 3:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    if uv is None:
        if len(polygon) != 4:
            raise ValueError("UVs are required for a non-quad polygon")
        uv = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64)
    else:
        uv = np.asarray(uv, dtype=np.float64)
    if uv.shape != (len(polygon), 2):
        raise ValueError("UVs must have one pair per polygon vertex")
    kept: list[np.ndarray] = []
    kept_uv: list[np.ndarray] = []
    for index, current in enumerate(polygon):
        following = polygon[(index + 1) % len(polygon)]
        current_uv, following_uv = uv[index], uv[(index + 1) % len(uv)]
        here = current[2] >= near_clip
        there = following[2] >= near_clip
        if here:
            kept.append(current)
            kept_uv.append(current_uv)
        if here != there:
            fraction = (near_clip - current[2]) / (following[2] - current[2])
            kept.append(current + fraction * (following - current))
            kept_uv.append(current_uv + fraction * (following_uv - current_uv))
    if len(kept) < 3:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    values = np.asarray(kept, dtype=np.float64)
    calibration = intrinsics_for(camera, width, height)
    focal_x, focal_y = calibration.focal_length_pixels
    principal_x, principal_y = calibration.principal_point_pixels
    return np.column_stack((
        principal_x + values[:, 0] * focal_x / values[:, 2],
        principal_y - values[:, 1] * focal_y / values[:, 2],
        values[:, 2],
    )), np.asarray(kept_uv, dtype=np.float64)


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
    calibration = intrinsics_for(camera, width, height)
    focal_x, focal_y = calibration.focal_length_pixels
    principal_x, principal_y = calibration.principal_point_pixels
    projected: list[tuple[float, float, float] | None] = []
    for view_x, rotated_y, depth in camera_space_points(points, camera):
        if depth <= 1e-6:
            projected.append(None)
            continue
        projected.append(
            (
                principal_x + view_x * focal_x / depth,
                principal_y - rotated_y * focal_y / depth,
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
    calibration = intrinsics_for(camera, width, height)
    view = np.asarray(((pixel_x - calibration.principal_point_pixels[0]) * depth / calibration.focal_length_pixels[0],
                       -(pixel_y - calibration.principal_point_pixels[1]) * depth / calibration.focal_length_pixels[1], depth))
    return tuple(camera_point_to_world(view, camera))


def camera_point_to_world(point, camera: CameraModel) -> np.ndarray:
    """A camera-space point (as `camera_space_points` would produce it) to world space.

    The inverse of `camera_space_points` for a single point: that function maps
    a world point to ``R @ (world - eye)``, so this applies ``R^T`` (rotation is
    orthogonal, so transpose is inverse) and adds the eye back. Verified by
    round-tripping a real point through both directions to float precision - see
    `test_camera.py`. Factored out of `unproject_pixel` so other callers that
    already have a camera-space point (rather than a pixel and a depth, e.g. a
    point built from `lattice.plane_coordinates_to_camera_point`) do not need to
    re-derive or copy this.
    """
    return camera_rotation_matrix(camera).T @ np.asarray(point, dtype=float) + camera_eye(camera)
