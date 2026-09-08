"""Propose cells from where the reconstruction and the photograph disagree.

The error attribution in `export.attribution` reads the reference litematic, so it
can only ever be a diagnostic. The same loop *without* the reference is a runtime
signal: render the reconstruction from a known pose, compare it against the actual
photograph, and wherever they disagree something along that ray is wrong.

That is what the missing thin blocks need. Template 2's stairs and carpets are 8%
of its cells and three quarters of its visible error, and 61 of its 69 partial
cells are never proposed by any stage - the palette and the shape fit cannot name a
block in a cell nothing puts a candidate in. Stereo reaches eight of them. This
reaches about a third of the top forty, which is four and a half times chance, and
unlike cross-view agreement it does not care that Minecraft is repetitive.

The empty cell *in front of* a disagreeing face is the proposal: along the face
normal, toward the camera. For a floor's top face that is where a carpet would lie;
for a wall, where a stair or a slab would sit.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from mc_photogrammetry.blocks.model_atlas import FACE_BASIS
from mc_photogrammetry.domain.models import BlockCoordinate, BlockState
from mc_photogrammetry.export.attribution import (
    face_colours,
    face_model_faces,
    face_shapes,
    render_result,
)
from mc_photogrammetry.reconstruction.camera import CameraModel
from mc_photogrammetry.reconstruction.scene import SceneTransform

RENDER_SCALE = 0.35
"""Fraction of source resolution the residual renders at."""

PROPOSAL_LIMIT = 40
"""How many cells to propose.

Measured on Template 2: forty proposals lift exact state from 79.71% to 80.79%,
sixty only to 80.07%. The precision falls faster than the recall rises, and a
wrong proposal stands in front of a surface that was already right.
"""

RESIDUAL_THRESHOLD = 1.0
"""How far above a frame's median disagreement a face must sit to be worth doubting."""

MINIMUM_PIXELS = 8
"""Pixels a face must occupy before its disagreement means anything."""


def face_residuals(
    blocks: Mapping[BlockCoordinate, BlockState],
    scene: SceneTransform,
    cameras: Mapping[str, CameraModel],
    frames: Mapping[str, np.ndarray],
    palette: Sequence,
    *,
    scale: float = RENDER_SCALE,
    minimum_pixels: int = MINIMUM_PIXELS,
) -> dict[tuple[BlockCoordinate, str], float]:
    """Each rendered face's disagreement with the photograph, relative to its frame.

    Relative, because the renderer is flat and the photographs are lit: an absolute
    difference measures exposure rather than error.
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover - diagnostic path
        return {}
    colours, shapes, model_faces = face_colours(palette), face_shapes(palette), face_model_faces(palette)
    gathered: dict[tuple[BlockCoordinate, str], list[float]] = defaultdict(list)

    for frame_id, image in frames.items():
        camera = cameras.get(frame_id)
        if camera is None:
            continue
        height, width = image.shape[:2]
        rendered, index = render_result(
            blocks, scene, camera, width, height, colours, shapes, scale, ambient=True, model_faces=model_faces)
        canvas, owner = rendered.colour, rendered.ownership
        seen = owner >= 0
        if not seen.any():
            continue
        photo = cv2.resize(image, (canvas.shape[1], canvas.shape[0]),
                           interpolation=cv2.INTER_AREA).astype(np.float32)
        gain = float(np.median(photo[seen].mean(axis=1))) / max(
            float(np.median(canvas[seen].mean(axis=1))), 1e-6)
        difference = np.linalg.norm(photo - canvas * gain, axis=2)
        typical = float(np.median(difference[seen])) or 1.0
        relative = difference / typical
        for row, column in zip(*np.nonzero(seen)):
            gathered[index[owner[row, column]]].append(float(relative[row, column]))

    return {
        key: float(np.mean(values))
        for key, values in gathered.items()
        if len(values) >= minimum_pixels
    }


def residual_proposals(
    blocks: Mapping[BlockCoordinate, BlockState],
    scene: SceneTransform,
    cameras: Mapping[str, CameraModel],
    frames: Mapping[str, np.ndarray],
    palette: Sequence,
    *,
    limit: int = PROPOSAL_LIMIT,
    threshold: float = RESIDUAL_THRESHOLD,
    scale: float = RENDER_SCALE,
) -> dict[BlockCoordinate, float]:
    """Empty cells standing in front of the worst-disagreeing faces.

    Keyed by the residual that proposed them, worst first, so a caller can take
    fewer than ``limit`` without recomputing.
    """
    if limit <= 0:
        return {}
    residuals = face_residuals(blocks, scene, cameras, frames, palette, scale=scale)
    if not residuals:
        return {}
    occupied = set(blocks)
    found: dict[BlockCoordinate, float] = {}
    for (coordinate, slot), score in sorted(residuals.items(), key=lambda item: -item[1]):
        if score < threshold:
            break
        normal = FACE_BASIS[slot][0]
        ahead = (coordinate[0] + int(normal[0]),
                 coordinate[1] + int(normal[1]),
                 coordinate[2] + int(normal[2]))
        if ahead in occupied or ahead in found:
            continue
        if not all(0 <= ahead[axis] < scene.size[axis] for axis in range(3)):
            continue
        found[ahead] = score
        if len(found) >= limit:
            break
    return found
