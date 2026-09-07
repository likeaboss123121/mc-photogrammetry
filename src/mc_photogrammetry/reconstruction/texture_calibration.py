from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.export.compare import LitematicSnapshot
from mc_photogrammetry.reconstruction.facade import PaletteColor
from .camera import CameraModel, project_points
from .scene import SceneTransform


@dataclass(frozen=True, slots=True)
class AppearanceCalibration:
    """Per-channel affine correction from texture colors to screenshot colors."""

    scale: tuple[float, float, float]
    offset: tuple[float, float, float]

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        return rgb * np.asarray(self.scale, dtype=np.float32) + np.asarray(self.offset, dtype=np.float32)


def calibrate_appearance(
    image: np.ndarray,
    reference: LitematicSnapshot,
    camera: CameraModel,
    scene: SceneTransform,
    texture_colors: dict[BlockState, tuple[int, int, int]],
) -> AppearanceCalibration:
    points = [tuple(value + 0.5 for value in scene.to_world(coordinate)) for coordinate in reference.blocks]
    projected = project_points(points, camera, width=image.shape[1], height=image.shape[0])
    source: list[tuple[int, int, int]] = []
    target: list[tuple[int, int, int]] = []
    for (coordinate, state), projection in zip(reference.blocks.items(), projected):
        if projection is None or state not in texture_colors:
            continue
        x, y, _ = projection
        x, y = round(x), round(y)
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            source.append(texture_colors[state])
            target.append(tuple(int(value) for value in image[y, x]))
    if len(source) < 6:
        return AppearanceCalibration((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    source_array = np.asarray(source, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    design = np.column_stack((source_array, np.ones(len(source_array))))
    coefficients = np.linalg.lstsq(design, target_array, rcond=None)[0]
    return AppearanceCalibration(
        scale=tuple(float(coefficients[index, index]) for index in range(3)),
        offset=tuple(float(coefficients[3, index]) for index in range(3)),
    )


def calibrate_appearance_from_palette(
    image: np.ndarray,
    camera: CameraModel,
    scene: SceneTransform,
    palette: tuple[PaletteColor, ...],
    *,
    confidence_fraction: float = 0.3,
) -> AppearanceCalibration:
    """Estimate screenshot brightness from confident palette matches only."""
    if not palette or not 0.0 < confidence_fraction <= 1.0:
        return AppearanceCalibration((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    coordinates = [
        (x, y, z)
        for x in range(scene.size[0])
        for y in range(scene.size[1])
        for z in range(scene.size[2])
    ]
    points = [tuple(value + 0.5 for value in scene.to_world(coordinate)) for coordinate in coordinates]
    projected = project_points(points, camera, width=image.shape[1], height=image.shape[0])
    palette_rgb = np.asarray([item.rgb for item in palette], dtype=np.float32)
    calibration = AppearanceCalibration((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    for _ in range(3):
        observations: list[tuple[float, np.ndarray, np.ndarray]] = []
        expected_rgb = calibration.apply(palette_rgb)
        for projection in projected:
            if projection is None:
                continue
            x, y, _ = projection
            pixel_x, pixel_y = round(x), round(y)
            if not (0 <= pixel_x < image.shape[1] and 0 <= pixel_y < image.shape[0]):
                continue
            observed = image[pixel_y, pixel_x].astype(np.float32)
            distances = np.linalg.norm(expected_rgb - observed, axis=1)
            order = np.argsort(distances)
            margin = float(distances[order[1]] - distances[order[0]]) if len(order) > 1 else 0.0
            observations.append((margin, observed, palette_rgb[order[0]]))
        if not observations:
            break
        observations.sort(key=lambda item: item[0], reverse=True)
        selected = observations[: max(6, int(len(observations) * confidence_fraction))]
        ratios = [observed / np.maximum(expected, 24.0) for _, observed, expected in selected]
        scale = np.clip(np.median(np.asarray(ratios), axis=0), 0.5, 2.0)
        calibration = AppearanceCalibration(tuple(float(value) for value in scale), (0.0, 0.0, 0.0))
    return calibration
