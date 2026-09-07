"""Multi-view stereo along the ray: recover interior geometry the shell cannot.

The geometry stage returns the region's bounding shell on every fixture, because
every cell is the nearest along some ray in some frame and the surface of
everything is its boundary. It recovers no interior at all - a fixture containing
an igloo, a witch hut and an end city came back as an empty box.

Six ways of turning cross-view appearance agreement into occupancy were measured
and all produced interior proposals 8-23% correct, because Minecraft is repetitive:
a phantom cell in front of a wall reprojects onto a *different part of the same
wall*, and where that wall is a flat expanse of one block the two patches correlate
as well as the true surface's do. Appearance similarity alone cannot separate them.

Two standard stereo devices fix it, and the combination is what this module is.

**Uniqueness.** Do not accept the best depth along a ray; accept it only when it
beats the runner-up by a margin. On a repeated texture several depths tie, none
clears the margin, and the ray is left unknown rather than guessed. A missing block
costs recall; a wrong one is unrecoverable, so this is the right way to be wrong.

**Cross-camera consistency.** A surface seen from two places is found in the same
place from both. A phantom's coincidence depends on one camera's line of sight and
rarely survives a second.

With both, and with agreement measured on colour as well as pattern, interior
proposals reach 75-98% precision instead of 8-23%.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Mapping

import numpy as np

from mc_photogrammetry.blocks.model_atlas import FACE_BASIS, FACE_SLOTS
from mc_photogrammetry.domain.models import BlockCoordinate
from mc_photogrammetry.reconstruction.camera import CameraModel, project_points
from mc_photogrammetry.reconstruction.face_sampling import (
    default_face_corners,
    quad_area,
    rectify_face,
)
from mc_photogrammetry.reconstruction.occupancy import (
    MINIMUM_FACE_AREA,
    normalised_pattern,
)
from mc_photogrammetry.reconstruction.scene import SceneTransform

RASTER_SCALE = 4
"""Pixels per side collapsed into one competing cell. Finer trades precision for yield."""

UNIQUENESS_MARGIN = 0.25
"""How far the best depth along a ray must beat the runner-up to be believed.

Measured on the three fixtures with poses: at 0.25 interior proposals are 75-80%
correct on Templates 4 and 6, against 8-23% for every method without this test.
Raising it to 0.30 reaches 98% on Template 4 but halves the yield.
"""

MINIMUM_WINS = 2
"""Coarse pixels a cell must hold in one camera before that camera has an opinion."""

MINIMUM_CAMERAS = 2
"""Cameras that must independently place a cell before it is believed."""

COLOUR_SCALE = 24.0
"""RGB spread, in levels, at which two views stop agreeing about a face's colour."""

COLOUR_WEIGHT = 1.0
"""Colour's weight against pattern. Colour carries the stronger signal here."""


def face_scores(
    scene: SceneTransform,
    cameras: Mapping[str, CameraModel],
    frames: Mapping[str, np.ndarray],
    *,
    colour_weight: float = COLOUR_WEIGHT,
    colour_scale: float = COLOUR_SCALE,
    minimum_area: float = MINIMUM_FACE_AREA,
) -> dict[tuple[BlockCoordinate, str], float]:
    """How well the views of each candidate face agree, on pattern and on colour.

    Occlusion is ignored deliberately, as in ``occupancy.face_patterns``: whether a
    face is in front is the question being asked, so depth-buffering here would
    assume the answer. Depth-buffering also returns almost nothing, since with
    every cell solid a camera can see only its own six neighbours.
    """
    cells = [
        (x, y, z)
        for x in range(scene.size[0])
        for y in range(scene.size[1])
        for z in range(scene.size[2])
    ]
    gathered: dict[tuple[BlockCoordinate, str], list[tuple[np.ndarray, np.ndarray]]]
    gathered = defaultdict(list)
    for frame_id, image in frames.items():
        camera = cameras.get(frame_id)
        if camera is None:
            continue
        height, width = image.shape[:2]
        eye = np.asarray(camera.position, float) + np.asarray([0.0, camera.eye_height, 0.0])
        keys: list[tuple[BlockCoordinate, str]] = []
        points: list[tuple[float, float, float]] = []
        for coordinate in cells:
            centre = np.asarray(scene.to_world(coordinate), float) + 0.5
            for slot in FACE_SLOTS:
                normal = np.asarray(FACE_BASIS[slot][0], float)
                if float(normal @ (eye - centre)) <= 0.05:
                    continue
                keys.append((coordinate, slot))
                points.extend(tuple(p) for p in centre + default_face_corners(slot))
        if not keys:
            continue
        projected = project_points(points, camera, width=width, height=height)
        for index, key in enumerate(keys):
            corners = projected[index * 4: index * 4 + 4]
            if any(point is None for point in corners):
                continue
            quad = np.asarray([[p[0], p[1]] for p in corners], np.float32)
            if (quad[:, 0].min() < 1 or quad[:, 0].max() >= width - 1
                    or quad[:, 1].min() < 1 or quad[:, 1].max() >= height - 1):
                continue
            if quad_area(quad) < minimum_area:
                continue
            patch = rectify_face(image, quad)
            if patch is None:
                continue
            pattern = normalised_pattern(patch)
            if pattern is not None:
                gathered[key].append((pattern, patch.reshape(-1, 3).mean(axis=0)))

    scores: dict[tuple[BlockCoordinate, str], float] = {}
    for key, views in gathered.items():
        count = len(views)
        if count < 2:
            continue
        patterns = np.stack([view[0] for view in views])
        total = patterns.sum(axis=0)
        shape = (float(total @ total) - count) / (count * (count - 1))
        colours = np.stack([view[1] for view in views])
        spread = float(np.mean(np.linalg.norm(colours - colours.mean(axis=0), axis=1)))
        tint = float(np.exp(-spread / max(colour_scale, 1e-6)))
        scores[key] = (shape + colour_weight * tint) / (1.0 + colour_weight)
    return scores


def stereo_surface(
    scene: SceneTransform,
    cameras: Mapping[str, CameraModel],
    frames: Mapping[str, np.ndarray],
    *,
    scores: Mapping[tuple[BlockCoordinate, str], float] | None = None,
    scale: int = RASTER_SCALE,
    margin: float = UNIQUENESS_MARGIN,
    minimum_wins: int = MINIMUM_WINS,
    minimum_cameras: int = MINIMUM_CAMERAS,
) -> set[BlockCoordinate]:
    """Cells that win a ray unambiguously in at least ``minimum_cameras`` views."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    if minimum_cameras < 1:
        raise ValueError("minimum_cameras must be at least one")
    if scores is None:
        scores = face_scores(scene, cameras, frames)
    if not scores:
        return set()
    try:
        import cv2
    except ImportError:
        return set()

    wins: dict[BlockCoordinate, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for frame_id, image in frames.items():
        camera = cameras.get(frame_id)
        if camera is None:
            continue
        height, width = image.shape[:2]
        rows, columns = height // scale, width // scale
        if rows <= 0 or columns <= 0:
            continue
        eye = np.asarray(camera.position, float) + np.asarray([0.0, camera.eye_height, 0.0])
        entries: list[tuple[float, np.ndarray, BlockCoordinate]] = []
        for (coordinate, slot), score in scores.items():
            centre = np.asarray(scene.to_world(coordinate), float) + 0.5
            normal = np.asarray(FACE_BASIS[slot][0], float)
            if float(normal @ (eye - centre)) <= 0.05:
                continue
            points = project_points(
                [tuple(p) for p in centre + default_face_corners(slot)],
                camera, width=width, height=height,
            )
            if any(point is None for point in points):
                continue
            quad = np.asarray([[p[0], p[1]] for p in points], np.float32)
            entries.append((score, quad, coordinate))
        if not entries:
            continue

        # Painting in ascending score leaves the best candidate holding each pixel
        # and the one painted just before it as the runner-up. A plain depth buffer
        # keeps only the winner and throws away exactly what uniqueness needs.
        best = np.full((rows, columns), -np.inf, dtype=np.float32)
        second = np.full((rows, columns), -np.inf, dtype=np.float32)
        owner = np.full((rows, columns), -1, dtype=np.int64)
        entries.sort(key=lambda entry: entry[0])
        for index, (score, quad, _coordinate) in enumerate(entries):
            stamp = np.zeros((rows, columns), np.uint8)
            cv2.fillConvexPoly(stamp, (quad / scale).astype(np.int32), 1)
            mask = stamp.astype(bool)
            if not mask.any():
                continue
            second[mask] = best[mask]
            best[mask] = score
            owner[mask] = index

        runner_up = np.where(np.isfinite(second), second, -1.0)
        decided = (owner >= 0) & ((best - runner_up) >= margin)
        if not decided.any():
            continue
        for index in np.unique(owner[decided]):
            count = int((owner[decided] == index).sum())
            wins[entries[int(index)][2]][frame_id] += count

    return {
        coordinate
        for coordinate, per_frame in wins.items()
        if sum(1 for count in per_frame.values() if count >= minimum_wins) >= minimum_cameras
    }
