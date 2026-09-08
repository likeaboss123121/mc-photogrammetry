"""Choose a partial block's shape and material by rendering it into the photograph.

The classifier compares a rectified face against a texture, which cannot separate a
stair from the planks it is cut from: identical texture, identical face colour. The
older answer scored shapes analytically - each visible face plus a "negative space"
term for the part of the cell the shape fails to cover - and reached 44.4% top-1
over 34 shapes.

This asks the photograph instead. Draw the candidate into the scene, compare the
pixels it produces against the pixels that are there, and keep whichever candidate
explains them best. Negative space stops being a term of its own: the part of a
cell a carpet does not fill simply shows whatever stands behind it.

Four things had to be right before the idea worked at all, each measured:

* **Score inside the cell's silhouette, not its bounding box.** A carpet fills a
  sixteenth of its cell, so most of the box is background whichever candidate is
  drawn and every score lands within noise of every other.
* **Match the render's level to the photograph's.** The renderer is flat, so
  without a gain every candidate is penalised in proportion to the pixels it
  paints and the winners are the shapes that paint almost none.
* **Shade by ambient occlusion.** Minecraft darkens a recess whatever the gamma;
  a flat render puts it at full brightness, and the correct shape then paints a
  large region in the wrong tone and loses to one that abstains.
* **Search only the free parameters.** A fence's connections, a wall's shape and a
  stair's corners are *derived* from neighbours. Searching them offers twenty-four
  states for one staircase where two are in question.

Material is not searched by rendering. A shape decides which pixels belong to which
face slot, so the best material for that shape follows from those pixels directly -
one render per shape instead of one per shape and material.

Two questions are being asked at once and they want different evidence. *Which
shape* is a question about silhouette, answered by rendering. *Which material* is a
question about texture, and comparing a flat average colour against textured pixels
cannot tell birch from oak - the fit was right about the family 69.6% of the time
and about the block only 26.1%. So each candidate face is also rectified out of the
photograph and scored by the same `PaletteIndex` the classifier uses, grain and all.

The two verdicts must stay separate. Summing them into one score let the grain vote
on the shape too and cost 2.9pp of family accuracy; deciding the shape by silhouette
and only then the material by grain kept both. Together: 69.6% family unchanged,
26.1% -> 43.5% right block, 20.3% -> 33.3% exact state.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from mc_photogrammetry.blocks.model_atlas import FACE_BASIS, FACE_SHADE, FACE_SLOTS
from mc_photogrammetry.domain.models import BlockCoordinate, BlockState
from mc_photogrammetry.export.attribution import (
    box_face_corners,
    face_colours,
    face_model_faces,
    face_occluding_slots,
    face_shapes,
    render_result,
    render_volume,
)
from mc_photogrammetry.reconstruction.camera import CameraModel, project_points
from mc_photogrammetry.reconstruction.lighting import ambient_occlusion
from mc_photogrammetry.reconstruction.scene import SceneTransform

RENDER_SCALE = 0.5
"""Fraction of source resolution the fit renders at."""

EDGE_WEIGHT = 0.25
"""Weight on gradient agreement beside colour. 0.6 measured worse than 0.25."""

MINIMUM_PIXELS = 6
"""Silhouette pixels a camera must offer before its opinion counts."""

TEXTURE_WEIGHT = 64.0
"""Weight on the rectified-texture cost when choosing a material.

Large on purpose. Swept against Template 2's 69 real partial cells, the right
block name rose the whole way - 26.1% with no texture, 31.9% at 0.3, 37.7% at 1,
42.0% at 16, 43.5% at 64, and no further - so the rendered difference is not
merely weaker evidence about material than the grain is, it is *harmful*: a flat
average colour under vanilla shading matches the wrong wood as readily as the
right one. At this weight the grain decides outright and the render term is left
as the fallback for cells where no face projects large enough to read.
"""

MINIMUM_FACE_AREA = 12.0
"""Projected pixels a candidate face needs before its grain is worth reading."""

PARTIAL_FAMILIES = ("_stairs", "_slab", "_carpet", "_fence", "_wall",
                    "_pane", "_bars", "torch")
"""Block families a built scene is made of, and the only shapes worth offering."""

DERIVED = ("north", "south", "east", "west", "up", "in_wall", "open")
"""Properties Minecraft computes from neighbours rather than the builder choosing."""

_OFF = ("false", "none")


def is_canonical(state: BlockState) -> bool:
    """Whether every derived property of this state sits at its off value."""
    properties = dict(state.properties)
    if any(properties.get(key, "false") not in _OFF for key in DERIVED):
        return False
    if properties.get("shape", "straight") != "straight":
        return False
    return properties.get("waterlogged", "false") == "false"


def candidate_states(palette: Sequence, families: Sequence[str] = PARTIAL_FAMILIES):
    """Canonical partial states, grouped by geometry.

    Returns ``(shapes, materials)``: the distinct shapes to try, and for each the
    states that share it.
    """
    shapes: dict[tuple, tuple] = {}
    materials: dict[tuple, list[BlockState]] = defaultdict(list)
    for item in palette:
        if getattr(item, "full_cube", True):
            continue
        if not any(key in item.state.name for key in families):
            continue
        if not is_canonical(item.state):
            continue
        boxes = tuple(getattr(item, "shape_boxes", ()) or ())
        if not boxes:
            continue
        key = tuple(
            (tuple(round(v, 3) for v in low), tuple(round(v, 3) for v in high))
            for low, high in sorted(boxes)
        )
        shapes.setdefault(key, boxes)
        materials[key].append(item.state)
    return shapes, materials


def _gradient(image: np.ndarray) -> np.ndarray:
    import cv2

    grey = image.mean(axis=2).astype(np.float32)
    dx = cv2.Scharr(grey, cv2.CV_32F, 1, 0)
    dy = cv2.Scharr(grey, cv2.CV_32F, 0, 1)
    return np.sqrt(dx * dx + dy * dy)


def _window(scene, camera, coordinate, shape, scale):
    """Bounding box and silhouette mask of one cell, as this camera sees it."""
    import cv2

    height, width = shape
    centre = np.asarray(scene.to_world(coordinate), dtype=float) + 0.5
    corners = np.asarray([[x, y, z] for x in (-0.5, 0.5)
                          for y in (-0.5, 0.5) for z in (-0.5, 0.5)])
    points = project_points([tuple(p) for p in centre + corners], camera,
                            width=int(width / scale), height=int(height / scale))
    if any(point is None for point in points):
        return None
    flat = np.asarray([[p[0] * scale, p[1] * scale] for p in points], np.float32)
    x0 = int(max(0, flat[:, 0].min() - 2))
    x1 = int(min(width, flat[:, 0].max() + 2))
    y0 = int(max(0, flat[:, 1].min() - 2))
    y1 = int(min(height, flat[:, 1].max() + 2))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(flat).astype(np.int32), 1)
    inside = mask[y0:y1, x0:x1].astype(bool)
    if int(inside.sum()) < MINIMUM_PIXELS:
        return None
    return x0, y0, x1, y1, inside


def _slot_masks(scene, camera, coordinate, boxes, occupied, scale, shape):
    """Which pixels each of a candidate's face slots claims, and their shading."""
    import cv2

    height, width = shape
    eye = np.asarray(camera.position, float) + np.asarray([0.0, camera.eye_height, 0.0])
    centre = np.asarray(scene.to_world(coordinate), float) + 0.5
    faces = []
    for low, high in boxes:
        for slot in FACE_SLOTS:
            normal = np.asarray(FACE_BASIS[slot][0], float)
            if float(normal @ (eye - centre)) <= 0.05:
                continue
            points = project_points(
                [tuple(p) for p in centre + box_face_corners(low, high, slot)],
                camera, width=int(width / scale), height=int(height / scale))
            if any(point is None for point in points):
                continue
            quad = np.asarray([[p[0] * scale, p[1] * scale] for p in points], np.float32)
            depth = float(np.mean([p[2] for p in points]))
            shade = FACE_SHADE.get(slot, 0.8) * ambient_occlusion(coordinate, slot, occupied)
            faces.append((depth, quad, slot, shade))

    stamped: dict[str, np.ndarray] = {}
    shading: dict[str, float] = {}
    covered = np.zeros((height, width), np.uint8)
    for _depth, quad, slot, shade in sorted(faces, key=lambda entry: -entry[0]):
        layer = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(layer, quad.astype(np.int32), 1)
        drawn = layer.astype(bool)
        if not drawn.any():
            continue
        for other in stamped:
            stamped[other] = stamped[other] & ~drawn
        stamped[slot] = (stamped.get(slot, np.zeros((height, width), bool)) | drawn)
        shading[slot] = shade
        covered |= layer
    return stamped, shading, covered.astype(bool)


def _face_patches(scene, camera, coordinate, boxes, image, scale):
    """Rectify the photograph through each of a candidate's own face quads.

    The shape decides the quad, so the patch this returns is what the camera would
    have seen *if the candidate were there*. That is exactly the input the
    classifier is built for, and it carries the grain that separates birch from
    oak where an average colour cannot.
    """
    from mc_photogrammetry.reconstruction.face_sampling import quad_area, rectify_face

    height, width = image.shape[:2]
    eye = np.asarray(camera.position, float) + np.asarray([0.0, camera.eye_height, 0.0])
    centre = np.asarray(scene.to_world(coordinate), float) + 0.5
    found = []
    for low, high in boxes:
        for slot in FACE_SLOTS:
            normal = np.asarray(FACE_BASIS[slot][0], float)
            if float(normal @ (eye - centre)) <= 0.05:
                continue
            points = project_points(
                [tuple(p) for p in centre + box_face_corners(low, high, slot)],
                camera, width=width, height=height)
            if any(point is None for point in points):
                continue
            quad = np.asarray([[p[0], p[1]] for p in points], np.float32)
            if (quad[:, 0].min() < 1 or quad[:, 0].max() >= width - 1
                    or quad[:, 1].min() < 1 or quad[:, 1].max() >= height - 1):
                continue
            area = quad_area(quad)
            if area < MINIMUM_FACE_AREA:
                continue
            patch = rectify_face(image, quad)
            if patch is None:
                continue
            found.append((slot, patch, float(np.sqrt(area))))
    return found


def fit_states(
    cells: Sequence[BlockCoordinate],
    blocks: Mapping[BlockCoordinate, BlockState],
    scene: SceneTransform,
    cameras: Mapping[str, CameraModel],
    frames: Mapping[str, np.ndarray],
    palette: Sequence,
    *,
    scale: float = RENDER_SCALE,
    edge_weight: float = EDGE_WEIGHT,
    texture_weight: float = TEXTURE_WEIGHT,
    shape_from_silhouette: bool = True,
    families: Sequence[str] = PARTIAL_FAMILIES,
) -> dict[BlockCoordinate, BlockState]:
    """Best partial state for each cell, judged against the photographs.

    ``blocks`` is the scene the cells stand in; the cells themselves are excluded
    from the background so each is judged against the same picture.
    """
    import cv2

    shapes, materials = candidate_states(palette, families)
    if not shapes:
        return {}
    colours, block_shapes, model_faces = face_colours(palette), face_shapes(palette), face_model_faces(palette)
    occluding_slots = face_occluding_slots(palette)
    index, positions = None, {}
    if texture_weight > 0.0:
        from mc_photogrammetry.reconstruction.raycast import (
            DEFAULT_FAMILY_PRIORS,
            PaletteIndex,
            _normalised_pattern,
            _palette_face_colors,
        )

        wanted = {state for group in materials.values() for state in group}
        subset = tuple(item for item in palette if item.state in wanted)
        if subset:
            index = PaletteIndex(subset, _palette_face_colors(subset),
                                 DEFAULT_FAMILY_PRIORS, 1.0)
            positions = {state: number for number, state in enumerate(index.states)}
    targets = set(cells)
    background_blocks = {c: s for c, s in blocks.items() if c not in targets}
    backgrounds, photos, edges, gains = {}, {}, {}, {}
    for frame_id, image in frames.items():
        camera = cameras.get(frame_id)
        if camera is None:
            continue
        height, width = image.shape[:2]
        canvas, _o, _i, _d = render_volume(background_blocks, scene, camera, width,
                                           height, colours, block_shapes, scale,
                                           ambient=True, occluding_slots=occluding_slots,
                                           model_faces=model_faces)
        photo = cv2.resize(image, (canvas.shape[1], canvas.shape[0]),
                           interpolation=cv2.INTER_AREA).astype(np.float32)
        lit = canvas.mean(axis=2) > 1.0
        gain = (float(np.median(photo[lit].mean(axis=1)))
                / max(float(np.median(canvas[lit].mean(axis=1))), 1e-6)) if lit.any() else 1.0
        backgrounds[frame_id] = canvas * gain
        photos[frame_id] = photo
        edges[frame_id] = _gradient(photo)
        gains[frame_id] = gain

    chosen: dict[BlockCoordinate, BlockState] = {}
    for coordinate in cells:
        views = []
        for frame_id, camera in cameras.items():
            if frame_id not in backgrounds:
                continue
            window = _window(scene, camera, coordinate,
                             backgrounds[frame_id].shape[:2], scale)
            if window is not None:
                views.append((frame_id, camera, window))
        if not views:
            continue

        # Per shape: the silhouette's own verdict, and the material the grain
        # picks within it. Which of the two settles the shape is a flag.
        ranked: list[tuple[float, float, BlockState]] = []
        for key, boxes in shapes.items():
            # The grain the candidate's own faces would show, gathered once per
            # shape and scored for every material that shares it.
            texture_cost = None
            if index is not None:
                total_weight = 0.0
                pooled = np.zeros(index.count, dtype=np.float64)
                for frame_id, camera, _seen in views:
                    for slot, patch, weight in _face_patches(
                            scene, camera, coordinate, boxes, frames[frame_id], scale):
                        costs, _best = index.costs(
                            patch.reshape(-1, 3).mean(axis=0), (), slot,
                            pattern=_normalised_pattern(patch), texture_weight=1.0)
                        if costs.size:
                            pooled += weight * costs
                            total_weight += weight
                if total_weight > 0.0:
                    texture_cost = pooled / total_weight

            plain_best, joint_best, joint_state = None, None, None
            for state in materials[key]:
                table = colours.get(state.name, {})
                if not table:
                    continue
                total, counted = 0.0, 0
                candidate_blocks = {**background_blocks, coordinate: state}
                for frame_id, camera, _window_data in views:
                    height, width = frames[frame_id].shape[:2]
                    rendered, face_index = render_result(
                        candidate_blocks, scene, camera, width, height, colours, block_shapes, scale,
                        ambient=True, occluding_slots=occluding_slots, model_faces=model_faces,
                    )
                    owned = np.zeros(rendered.ownership.shape, dtype=bool)
                    for identifier, (cell, _slot) in enumerate(face_index):
                        if cell == coordinate:
                            owned |= rendered.ownership == identifier
                    if not owned.any():
                        continue
                    painted = rendered.colour * gains[frame_id]
                    difference = np.linalg.norm(painted - photos[frame_id], axis=2)
                    if edge_weight > 0.0:
                        difference = difference + edge_weight * np.abs(
                            _gradient(painted) - edges[frame_id])
                    total += float(difference[owned].mean())
                    counted += 1
                if not counted:
                    continue
                plain = total / counted
                score = plain
                if texture_cost is not None:
                    position = positions.get(state)
                    if position is not None:
                        score += texture_weight * float(texture_cost[position])
                if plain_best is None or plain < plain_best:
                    plain_best = plain
                if joint_best is None or score < joint_best:
                    joint_best, joint_state = score, state
            if joint_state is not None:
                ranked.append((plain_best, joint_best, joint_state))
        if ranked:
            # Silhouette answers "which shape", grain answers "which material".
            # Letting the grain vote on the shape as well cost 2.9pp of family.
            order = 0 if shape_from_silhouette else 1
            chosen[coordinate] = min(ranked, key=lambda entry: entry[order])[2]
    return chosen
