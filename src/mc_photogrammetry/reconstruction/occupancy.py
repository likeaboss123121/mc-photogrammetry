"""Find the scene's visible surface by cross-view agreement.

The reconstruction used to decide occupancy by marking a cell seen wherever its
projected centre was the nearest candidate at some pixel, then dropping the cells
whose six neighbours were all present. A frame carries two million pixels against
a few thousand cells, so the first step marks essentially the whole region and the
second is then left pruning a solid block - which returns the region's bounding-box
shell and nothing else. That is the right answer only when the scene happens to
fill its own bounding box, as a room does; for a cave, whose visible surface is the
rock/air boundary deep inside the region, it is the wrong set entirely.

What decides the question without assuming an answer is agreement between views. A
face that really is an exposed solid surface shows the same material to every
camera that can see it. A face hypothesised on an air cell, or on a cell buried
behind rock, is a window onto whatever lies beyond it, and each camera looks
through that window from a different angle and sees something different. So the
faces whose rectified patches agree across views are the real surface, and no prior
occupancy is needed to find them - which is what makes this non-circular.

Textureless surfaces are the classic weakness of matching by agreement, and
Minecraft softens it: block boundaries put a hard edge every sixteen pixels, so a
surface hypothesised at the wrong depth misaligns that grid even when the material
either side of it is flat.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..blocks.model_atlas import FACE_BASIS
from ..domain.models import BlockCoordinate
from .camera import CameraModel, project_points
from .face_sampling import (
    FACE_SLOTS,
    MINIMUM_FACE_AREA,
    default_face_corners,
    quad_area,
    rectify_face,
)
from .scene import SceneTransform

AGREEMENT_THRESHOLD = 0.35
"""How alike two views of a face must look before the face is called solid.

Correlation between zero-mean, unit-norm patches, so 1.0 is identical and 0.0 is
unrelated. Lower values keep more of the scene and risk holding cells that are
really air; higher values carve harder and risk eating thin structures.
"""

PROPOSAL_BUDGET = 200_000
"""Cell-frame projections the proposal pass may spend.

Template 5 is 277,160 cells against 39 frames - ten million face
projections, hours of work. Beyond the budget the frames are thinned.
"""

MINIMUM_AGREEMENT_VIEWS = 2
"""Cameras that must see a face before its agreement means anything."""


def normalised_pattern(patch: np.ndarray) -> np.ndarray | None:
    """Zero-mean, unit-norm luminance of a face patch, or ``None`` if it is flat.

    Normalising away the mean and the magnitude leaves only the pattern, so two
    views of one face still match when one of them caught it brighter.
    """
    luminance = patch.mean(axis=2).reshape(-1)
    centred = luminance - luminance.mean()
    norm = float(np.linalg.norm(centred))
    if norm < 1e-3:
        return None
    return (centred / norm).astype(np.float32)


def face_patterns(
    coordinates: list[BlockCoordinate],
    camera: CameraModel,
    scene: SceneTransform,
    image: np.ndarray,
    *,
    minimum_area: float = MINIMUM_FACE_AREA,
    gui_mask: np.ndarray | None = None,
) -> dict[tuple[BlockCoordinate, str], np.ndarray]:
    """Rectify every camera-facing face of every candidate cell.

    Deliberately ignores occlusion. Whether a face is really in front is the
    question being asked, so answering it here would assume the conclusion; a
    hidden face simply rectifies whatever stands in front of it and fails to agree
    with the other views later.
    """
    height, width = image.shape[:2]
    eye = np.asarray(camera.position, dtype=float) + np.asarray([0.0, camera.eye_height, 0.0])
    faces: list[tuple[BlockCoordinate, str]] = []
    points: list[tuple[float, float, float]] = []
    for coordinate in coordinates:
        center = np.asarray(scene.to_world(coordinate), dtype=float) + 0.5
        for slot in FACE_SLOTS:
            normal = np.asarray(FACE_BASIS[slot][0], dtype=float)
            if float(normal @ (eye - center)) <= 0.05:
                continue
            faces.append((coordinate, slot))
            points.extend(tuple(point) for point in center + default_face_corners(slot))
    if not faces:
        return {}

    projected = project_points(points, camera, width=width, height=height)
    patterns: dict[tuple[BlockCoordinate, str], np.ndarray] = {}
    for index, key in enumerate(faces):
        corners = projected[index * 4 : index * 4 + 4]
        if any(point is None for point in corners):
            continue
        quad = np.asarray([[point[0], point[1]] for point in corners], dtype=np.float32)
        if (
            quad[:, 0].min() < 1
            or quad[:, 0].max() >= width - 1
            or quad[:, 1].min() < 1
            or quad[:, 1].max() >= height - 1
        ):
            continue
        if quad_area(quad) < minimum_area:
            continue
        if gui_mask is not None and gui_mask.any():
            column = np.clip(quad[:, 0].astype(int), 0, width - 1)
            row = np.clip(quad[:, 1].astype(int), 0, height - 1)
            if gui_mask[row, column].any():
                continue
        patch = rectify_face(image, quad)
        if patch is None:
            continue
        pattern = normalised_pattern(patch)
        if pattern is not None:
            patterns[key] = pattern
    return patterns


def face_agreement(
    scene: SceneTransform,
    cameras: dict[str, CameraModel],
    frames: dict[str, np.ndarray],
    *,
    minimum_area: float = MINIMUM_FACE_AREA,
    gui_masks: dict[str, np.ndarray] | None = None,
    minimum_views: int = MINIMUM_AGREEMENT_VIEWS,
) -> dict[tuple[BlockCoordinate, str], float]:
    """Mean correlation between the views of each candidate face."""
    coordinates = [
        (x, y, z)
        for x in range(scene.size[0])
        for y in range(scene.size[1])
        for z in range(scene.size[2])
    ]
    collected: dict[tuple[BlockCoordinate, str], list[np.ndarray]] = defaultdict(list)
    for frame_id, image in frames.items():
        camera = cameras.get(frame_id)
        if camera is None:
            continue
        found = face_patterns(
            coordinates, camera, scene, image,
            minimum_area=minimum_area,
            gui_mask=None if gui_masks is None else gui_masks.get(frame_id),
        )
        for key, pattern in found.items():
            collected[key].append(pattern)

    agreement: dict[tuple[BlockCoordinate, str], float] = {}
    for key, patterns in collected.items():
        count = len(patterns)
        if count < max(2, minimum_views):
            continue
        # Every pattern is unit-norm, so the sum of all pairwise correlations
        # falls out of the norm of their sum and costs one dot product instead
        # of one per pair.
        total = np.sum(patterns, axis=0)
        pairs = count * (count - 1)
        agreement[key] = float((float(total @ total) - count) / pairs)
    return agreement


def visible_surface(
    scene: SceneTransform,
    cameras: dict[str, CameraModel],
    frames: dict[str, np.ndarray],
    *,
    threshold: float = AGREEMENT_THRESHOLD,
    minimum_area: float = MINIMUM_FACE_AREA,
    gui_masks: dict[str, np.ndarray] | None = None,
    minimum_views: int = MINIMUM_AGREEMENT_VIEWS,
) -> set[BlockCoordinate]:
    """The cells carrying at least one face the cameras agree on."""
    agreement = face_agreement(
        scene, cameras, frames,
        minimum_area=minimum_area, gui_masks=gui_masks, minimum_views=minimum_views,
    )
    best: dict[BlockCoordinate, float] = {}
    for (coordinate, _slot), score in agreement.items():
        if score > best.get(coordinate, -np.inf):
            best[coordinate] = score
    return {coordinate for coordinate, score in best.items() if score >= threshold}


def bounding_shell(size: tuple[int, int, int]) -> set[BlockCoordinate]:
    """Every cell on the outside of the region.

    The trivial answer the old occupancy pass collapsed to, kept so that a
    recovered surface can be compared against it rather than merely looking
    plausible.
    """
    width, height, depth = size
    return {
        (x, y, z)
        for x in range(width)
        for y in range(height)
        for z in range(depth)
        if x in (0, width - 1) or y in (0, height - 1) or z in (0, depth - 1)
    }


COLOUR_GAP = 0.0
"""How closely two views must agree in colour before a cell is called solid.

Zero disables the proposal entirely. Small values admit only cells whose views
record almost the same colour, which is what two looks at one real surface do.
"""

PROPOSAL_AGREEMENT = 0.7
"""How closely the two views must also agree in pattern, alongside colour."""


def proposed_cells(
    scene: SceneTransform,
    cameras: dict[str, CameraModel],
    frames: dict[str, np.ndarray],
    *,
    colour_gap: float = COLOUR_GAP,
    agreement: float = PROPOSAL_AGREEMENT,
    gui_masks: dict[str, np.ndarray] | None = None,
    minimum_area: float = MINIMUM_FACE_AREA,
    budget: int = PROPOSAL_BUDGET,
) -> set[BlockCoordinate]:
    """Cells two or more cameras agree on, whatever the occupancy pass concluded.

    The occupancy pass collapses to the region's bounding shell, so anything
    standing inside a room - a pillar, a ruined portal, the light sources that let
    the lighting simulation run at all - is never proposed. This looks for those
    cells directly: a face two cameras record in the same colour *and* the same
    pattern is a real surface, whether or not the shell contains it.

    Colour carries most of the signal and pattern alone carries little. Earlier
    attempts compared only normalised patterns, which subtract the mean and so
    discard exactly the colour that distinguishes one surface from another.
    """
    if colour_gap <= 0.0:
        return set()
    cells = scene.size[0] * scene.size[1] * scene.size[2]
    chosen = sorted(frames)
    if budget > 0 and cells * len(chosen) > budget:
        # Every cell is projected into every frame, so the work grows with the
        # product. Past the budget, thin the frames rather than the region: a
        # cell needs only two views to be proposed at all, and evenly spaced
        # frames keep the parallax that makes the agreement mean anything.
        keep = max(2, budget // max(cells, 1))
        step = max(1, len(chosen) // keep)
        chosen = chosen[::step][:keep]
    coordinates = [
        (x, y, z)
        for x in range(scene.size[0])
        for y in range(scene.size[1])
        for z in range(scene.size[2])
    ]
    seen: dict[tuple[BlockCoordinate, str], list[tuple[np.ndarray, np.ndarray | None]]] = {}
    for frame_id in chosen:
        image = frames[frame_id]
        camera = cameras.get(frame_id)
        if camera is None:
            continue
        for key, patch in _face_patches(
            coordinates, camera, scene, image,
            minimum_area=minimum_area,
            gui_mask=None if gui_masks is None else gui_masks.get(frame_id),
        ).items():
            seen.setdefault(key, []).append(
                (np.median(patch.reshape(-1, 3), axis=0), normalised_pattern(patch))
            )

    proposed: set[BlockCoordinate] = set()
    for (coordinate, _slot), views in seen.items():
        if len(views) < 2 or coordinate in proposed:
            continue
        for first in range(len(views)):
            for second in range(first + 1, len(views)):
                gap = float(np.linalg.norm(views[first][0] - views[second][0]))
                if gap > colour_gap:
                    continue
                one, two = views[first][1], views[second][1]
                if agreement > 0.0 and (one is None or two is None or one @ two < agreement):
                    continue
                proposed.add(coordinate)
                break
            if coordinate in proposed:
                break
    return proposed


def _face_patches(
    coordinates: list[BlockCoordinate],
    camera: CameraModel,
    scene: SceneTransform,
    image: np.ndarray,
    *,
    minimum_area: float,
    gui_mask: np.ndarray | None,
) -> dict[tuple[BlockCoordinate, str], np.ndarray]:
    """Rectified patch for every camera-facing face, occlusion deliberately ignored.

    Every corner is projected in one call. Projecting face by face costs the same
    arithmetic but pays Python's overhead per face, and this runs over the whole
    region for every frame - ten million faces on the larger fixtures.
    """
    height, width = image.shape[:2]
    eye = np.asarray(camera.position, dtype=float) + np.asarray([0.0, camera.eye_height, 0.0])
    faces: list[tuple[BlockCoordinate, str]] = []
    points: list[tuple[float, float, float]] = []
    for coordinate in coordinates:
        center = np.asarray(scene.to_world(coordinate), dtype=float) + 0.5
        toward = eye - center
        for slot in FACE_SLOTS:
            normal = np.asarray(FACE_BASIS[slot][0], dtype=float)
            if float(normal @ toward) <= 0.05:
                continue
            faces.append((coordinate, slot))
            points.extend(tuple(point) for point in center + default_face_corners(slot))
    if not faces:
        return {}

    projected = project_points(points, camera, width=width, height=height)
    patches: dict[tuple[BlockCoordinate, str], np.ndarray] = {}
    for index, key in enumerate(faces):
        corners = projected[index * 4 : index * 4 + 4]
        if any(point is None for point in corners):
            continue
        quad = np.asarray([[point[0], point[1]] for point in corners], dtype=np.float32)
        if (
            quad[:, 0].min() < 1
            or quad[:, 0].max() >= width - 1
            or quad[:, 1].min() < 1
            or quad[:, 1].max() >= height - 1
        ):
            continue
        if quad_area(quad) < minimum_area:
            continue
        if gui_mask is not None and gui_mask.any():
            column = np.clip(quad[:, 0].astype(int), 0, width - 1)
            row = np.clip(quad[:, 1].astype(int), 0, height - 1)
            if gui_mask[row, column].any():
                continue
        patch = rectify_face(image, quad)
        if patch is not None:
            patches[key] = patch
    return patches
