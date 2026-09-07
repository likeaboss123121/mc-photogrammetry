from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mc_photogrammetry.domain.models import BlockCoordinate, BlockState, BlockVolume
from .facade import DEFAULT_PALETTE, PaletteColor
from .camera import CameraModel, project_points
from .scene import SceneTransform


@dataclass(frozen=True, slots=True)
class VisibleBlock:
    coordinate: BlockCoordinate
    state: BlockState
    pixel: tuple[int, int]
    depth: float
    frame_id: str


def visible_blocks_for_camera(
    volume: BlockVolume,
    camera: CameraModel,
    *,
    width: int,
    height: int,
    frame_id: str,
    pixel_radius: int = 2,
) -> list[VisibleBlock]:
    """Return front-most volume blocks projected into one calibrated image."""
    if pixel_radius < 0:
        raise ValueError("pixel_radius cannot be negative")
    coordinates = list(volume.blocks)
    projections = project_points(coordinates, camera, width=width, height=height)
    front: dict[tuple[int, int], tuple[float, BlockCoordinate]] = {}
    for coordinate, projection in zip(coordinates, projections):
        if projection is None:
            continue
        pixel_x, pixel_y, depth = projection
        if not (-pixel_radius <= pixel_x < width + pixel_radius and -pixel_radius <= pixel_y < height + pixel_radius):
            continue
        key = (round(pixel_x), round(pixel_y))
        previous = front.get(key)
        if previous is None or depth < previous[0]:
            front[key] = (depth, coordinate)

    visible: list[VisibleBlock] = []
    for (pixel_x, pixel_y), (depth, coordinate) in sorted(front.items()):
        observation = volume.blocks[coordinate]
        visible.append(VisibleBlock(coordinate, observation.state, (pixel_x, pixel_y), depth, frame_id))
    return visible


def fuse_calibrated_views(
    volume: BlockVolume,
    cameras: dict[str, CameraModel],
    *,
    width: int,
    height: int,
) -> dict[BlockCoordinate, list[VisibleBlock]]:
    """Index which calibrated views can see each candidate block."""
    fused: dict[BlockCoordinate, list[VisibleBlock]] = {}
    for frame_id, camera in cameras.items():
        for visible in visible_blocks_for_camera(
            volume, camera, width=width, height=height, frame_id=frame_id
        ):
            fused.setdefault(visible.coordinate, []).append(visible)
    return fused


def reconstruct_visible_surfaces(
    frames: dict[str, np.ndarray],
    cameras: dict[str, CameraModel],
    *,
    scene: SceneTransform,
    columns: int = 32,
    rows: int = 32,
    depth_layers: int = 8,
    palette: tuple[PaletteColor, ...] = DEFAULT_PALETTE,
) -> BlockVolume:
    """Merge one camera-aware visible surface hit per cell across calibrated views."""
    from .facade import reconstruct_facade

    merged = BlockVolume()
    for frame_id, image in frames.items():
        partial = reconstruct_facade(
            image,
            columns=columns,
            rows=rows,
            depth_layers=depth_layers,
            camera=cameras[frame_id],
            scene_transform=scene,
            palette=palette,
        )
        for coordinate, observation in partial.items():
            observation.frame_ids = [frame_id]
            existing = merged.get(coordinate)
            if existing is None:
                merged.set(coordinate, observation)
            else:
                existing.frame_ids = sorted(set(existing.frame_ids + [frame_id]))
                existing.confidence = max(existing.confidence, observation.confidence)
    return merged
