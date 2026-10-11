from __future__ import annotations

import numpy as np

from mc_photogrammetry.blocks.appearance import DEFAULT_PALETTE, PaletteColor
from mc_photogrammetry.domain.models import (
    BlockObservation,
    BlockState,
    BlockVolume,
    ObservationKind,
)
from .camera import CameraModel, unproject_pixel
from .scene import SceneTransform

__all__ = ["DEFAULT_PALETTE", "PaletteColor", "estimate_horizon", "reconstruct_facade"]


def reconstruct_facade(
    image: np.ndarray,
    gui_mask: np.ndarray | None = None,
    *,
    columns: int = 32,
    rows: int = 32,
    depth_layers: int = 1,
    horizon: float | None = None,
    camera: CameraModel | None = None,
    scene_transform: SceneTransform | None = None,
    palette: tuple[PaletteColor, ...] = DEFAULT_PALETTE,
) -> BlockVolume:
    """Create a grid-aligned single-view reconstruction from an RGB image.

    Depth is estimated from screen-space perspective when ``depth_layers`` is greater
    than one. The front face is observed; blocks behind it are explicitly inferred.
    """
    _validate_image(image)
    if columns <= 0 or rows <= 0:
        raise ValueError("columns and rows must be positive")
    if depth_layers <= 0:
        raise ValueError("depth_layers must be positive")
    if horizon is not None and not 0.0 < horizon < 1.0:
        raise ValueError("horizon must be between 0 and 1")
    if not palette:
        raise ValueError("palette cannot be empty")
    if gui_mask is not None:
        if gui_mask.shape != image.shape[:2] or gui_mask.dtype != np.bool_:
            raise ValueError("gui_mask must match the image height and width as boolean data")

    height, width = image.shape[:2]
    resolved_horizon = horizon if horizon is not None else estimate_horizon(image, gui_mask)
    row_edges = _perspective_row_edges(rows, height, resolved_horizon) if depth_layers > 1 else np.array(
        [(index * height) // rows for index in range(rows + 1)]
    )
    volume = BlockVolume()
    for row in range(rows):
        y_start, y_end = row_edges[row], row_edges[row + 1]
        for column in range(columns):
            x_start, x_end = _cell_bounds(column, columns, width)
            cell = image[y_start:y_end, x_start:x_end]
            if cell.size == 0:
                continue
            visible = np.ones(cell.shape[:2], dtype=bool)
            if gui_mask is not None:
                visible &= ~gui_mask[y_start:y_end, x_start:x_end]
            visible_count = int(visible.sum())
            if visible_count == 0:
                continue
            pixels = cell[visible]
            mean_rgb = pixels.astype(np.float32).mean(axis=0)
            if float(mean_rgb.mean()) < 8.0:
                continue
            state, distance = _nearest_palette(mean_rgb, palette)
            confidence = max(0.0, min(1.0, (1.0 - distance / 441.67) * visible_count / visible.size))
            texture_factor = float(np.clip(np.std(pixels.astype(np.float32)) / 64.0, 0.0, 1.0))
            depth = (
                _estimate_depth(row, rows, resolved_horizon, depth_layers, texture_factor)
                if depth_layers > 1
                else 0
            )
            # Keep image columns aligned. Depth is a camera-ray estimate, not a reason to shear X.
            if camera is not None:
                pixel_x = (x_start + x_end) / 2.0
                pixel_y = (y_start + y_end) / 2.0
                normalized_depth = depth / max(1, depth_layers - 1)
                camera_depth = (
                    _scene_ray_depth(
                        pixel_x,
                        pixel_y,
                        camera,
                        scene_transform,
                        width,
                        height,
                        normalized_depth,
                    )
                    if scene_transform is not None
                    else 4.0 + (1.0 - normalized_depth) * 28.0
                )
                if camera_depth is None or camera_depth <= 1e-6:
                    continue
                point = unproject_pixel(
                    pixel_x,
                    pixel_y,
                    camera,
                    width=width,
                    height=height,
                    depth=camera_depth,
                )
                world_coordinate = tuple(int(round(value)) for value in point)
                coordinate = (
                    scene_transform.to_local(world_coordinate)
                    if scene_transform is not None
                    else world_coordinate
                )
                if scene_transform is not None and not all(
                    0 <= coordinate[index] < scene_transform.size[index] for index in range(3)
                ):
                    continue
            else:
                coordinate = (column, rows - row - 1, depth)
            # Keep only the visible surface hit. Depth is estimated, but blocks behind
            # that hit are unknown and must not be fabricated.
            existing = volume.get(coordinate)
            observation = BlockObservation(
                state=state,
                confidence=confidence,
                kind=ObservationKind.OBSERVED,
                frame_ids=["image-0000"],
            )
            if existing is None or observation.confidence > existing.confidence:
                volume.set(coordinate, observation)
    return volume


def estimate_horizon(image: np.ndarray, gui_mask: np.ndarray | None = None) -> float:
    """Estimate the perspective horizon from horizontal edge energy."""
    _validate_image(image)
    grayscale = image.astype(np.float32).mean(axis=2)
    differences = np.abs(np.diff(grayscale, axis=0))
    if gui_mask is not None:
        valid = ~(gui_mask[:-1] | gui_mask[1:])
        differences = np.where(valid, differences, 0.0)
    row_energy = differences.mean(axis=1)
    start = max(1, int(image.shape[0] * 0.2))
    end = min(image.shape[0] - 2, int(image.shape[0] * 0.75))
    if end <= start:
        return 0.45
    smoothed = np.convolve(row_energy, np.ones(5) / 5.0, mode="same")
    if float(smoothed[start:end].max()) <= 1e-6:
        return 0.45
    peak = start + int(np.argmax(smoothed[start:end]))
    return float(np.clip((peak + 1) / image.shape[0], 0.2, 0.7))


def _cell_bounds(index: int, count: int, size: int) -> tuple[int, int]:
    return (index * size) // count, ((index + 1) * size) // count


def _scene_ray_depth(
    pixel_x: float,
    pixel_y: float,
    camera: CameraModel,
    scene: SceneTransform,
    width: int,
    height: int,
    normalized_depth: float,
) -> float | None:
    """Return a depth along a camera ray inside the calibrated scene AABB."""
    eye = np.asarray(camera.position, dtype=np.float64) + np.array([0.0, camera.eye_height, 0.0])
    near_point = np.asarray(
        unproject_pixel(pixel_x, pixel_y, camera, width=width, height=height, depth=1.0)
    )
    direction = near_point - eye
    minimum = np.asarray(scene.world_origin, dtype=np.float64)
    maximum = minimum + np.asarray(scene.size, dtype=np.float64)
    lower, upper = -np.inf, np.inf
    for axis in range(3):
        if abs(direction[axis]) < 1e-9:
            if eye[axis] < minimum[axis] or eye[axis] > maximum[axis]:
                return None
            continue
        first = (minimum[axis] - eye[axis]) / direction[axis]
        last = (maximum[axis] - eye[axis]) / direction[axis]
        lower = max(lower, min(first, last))
        upper = min(upper, max(first, last))
    if upper <= max(lower, 0.0):
        return None
    ray_parameter = max(lower, 0.0) + (upper - max(lower, 0.0)) * (1.0 - normalized_depth)
    return float(ray_parameter * np.linalg.norm(direction))


def _perspective_row_edges(rows: int, height: int, horizon: float) -> np.ndarray:
    """Space projected block bands more densely near the camera."""
    horizon_pixel = horizon * height
    span = height - horizon_pixel
    positions = np.linspace(0.0, 1.0, rows + 1)
    below_horizon = np.clip((positions - horizon) / (1.0 - horizon), 0.0, 1.0)
    projected = np.where(
        positions <= horizon,
        positions / horizon * horizon_pixel,
        horizon_pixel + below_horizon**1.65 * span,
    )
    edges = np.rint(projected).astype(int)
    return np.maximum.accumulate(edges)


def _estimate_depth(
    row: int, rows: int, horizon: float, depth_layers: int, texture_factor: float
) -> int:
    """Map distance below the vanishing horizon to a discrete camera-ray depth."""
    screen_y = (row + 0.5) / rows
    distance = screen_y - horizon
    if distance <= 0:
        return 0
    normalized = min(1.0, distance / max(1e-6, 1.0 - horizon))
    perspective = normalized ** 0.5
    return min(depth_layers - 1, max(0, int(round(perspective * (depth_layers - 1) * (0.7 + 0.3 * texture_factor)))))


def _nearest_palette(
    rgb: np.ndarray, palette: tuple[PaletteColor, ...]
) -> tuple[BlockState, float]:
    distances = [float(np.linalg.norm(rgb - np.asarray(color.rgb, dtype=np.float32))) for color in palette]
    nearest = min(range(len(distances)), key=distances.__getitem__)
    return palette[nearest].state, distances[nearest]


def _validate_image(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be an RGB uint8 array")
