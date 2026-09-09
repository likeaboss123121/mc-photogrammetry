"""Choose a partial block's shape and material by rendering it into the photograph.

The classifier compares a rectified face against a texture, which cannot separate a
stair from the planks it is cut from: identical texture, identical face colour. The
older answer scored shapes analytically - each visible face plus a "negative space"
term for the part of the cell the shape fails to cover - and reached 44.4% top-1
over 34 shapes.

This asks the photograph instead. Draw the candidate into the scene, compare it
with the pixels that are there over one common, depth-visible cell region, and
keep whichever candidate explains them best. The part of a cell a carpet does
not fill is still scored: it simply renders as the background behind it.

Four things had to be right before the idea worked at all, each measured:

* **Score inside one depth-visible cell silhouette, not candidate ownership.** A
  carpet fills a sixteenth of its cell, so candidate-owned scoring lets it skip
  the photographed slab surface it fails to explain. The region includes what
  is revealed when a candidate has no geometry, but excludes cells occluded by
  the surrounding scene.
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
import time
from typing import Mapping, MutableMapping, Sequence

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

DEFAULT_MAX_CANDIDATE_RENDERS = 256
"""Conservative default work budget for renderer-backed partial fitting."""

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
FULL_CUBE = (((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),)


class FitStateResult(dict):
    """Accepted shape edits with their actual supporting frame provenance.

    This remains a ``dict`` for callers that only need the edits, while making
    support available without relying on optional diagnostics logging.
    """

    def __init__(
        self,
        values: Mapping[BlockCoordinate, BlockState | None] | None = None,
        *,
        supporting_frames: Mapping[BlockCoordinate, Sequence[str]] | None = None,
    ) -> None:
        super().__init__(values or {})
        self.supporting_frames = {
            coordinate: tuple(frames)
            for coordinate, frames in (supporting_frames or {}).items()
        }


def is_canonical(state: BlockState) -> bool:
    """Whether every derived property of this state sits at its off value."""
    properties = dict(state.properties)
    if any(properties.get(key, "false") not in _OFF for key in DERIVED):
        return False
    if properties.get("shape", "straight") != "straight":
        return False
    return properties.get("waterlogged", "false") == "false"


def candidate_states(
    palette: Sequence,
    families: Sequence[str] = PARTIAL_FAMILIES,
    *,
    include_full_cubes: bool = False,
):
    """Canonical partial states, grouped by geometry.

    Returns ``(shapes, materials)``: the distinct shapes to try, and for each the
    states that share it.
    """
    shapes: dict[tuple, tuple] = {}
    materials: dict[tuple, list[BlockState]] = defaultdict(list)
    for item in palette:
        full_cube = bool(getattr(item, "full_cube", True))
        if full_cube and not include_full_cubes:
            continue
        if not full_cube and not any(key in item.state.name for key in families):
            continue
        if not is_canonical(item.state):
            continue
        boxes = tuple(getattr(item, "shape_boxes", ()) or (FULL_CUBE if full_cube else ()))
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


def _frame_masks(
    frame_id: str,
    image_shape: tuple[int, int],
    scaled_shape: tuple[int, int],
    usable_masks: Mapping[str, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed usable domain at source and render resolutions.

    An explicitly supplied mask is authoritative.  A missing mask for a frame is
    treated as unusable rather than silently turning that frame into evidence.
    The default GUI mask is kept for the ordinary diagnostic path.
    """
    import cv2

    height, width = image_shape
    if usable_masks is None:
        from mc_photogrammetry.vision.gui_mask import create_gui_mask

        source = ~create_gui_mask(height, width)
    else:
        supplied = usable_masks.get(frame_id)
        if supplied is None:
            source = np.zeros((height, width), dtype=bool)
        else:
            source = np.asarray(supplied, dtype=bool)
            if source.shape != (height, width):
                raise ValueError("usable mask must match the image dimensions")
    if source.shape == scaled_shape:
        return source, source.copy()
    # Keep only output pixels whose source footprint is wholly usable.  This
    # prevents an excluded HUD/pixel from bleeding into a downsampled score.
    coverage = cv2.resize(
        source.astype(np.float32), (scaled_shape[1], scaled_shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    scaled = coverage >= (1.0 - 1e-6)
    return source, scaled


def _score_image(
    rendered: np.ndarray,
    photo: np.ndarray,
    valid: np.ndarray,
    edge_weight: float,
    photo_edges: np.ndarray,
) -> float | None:
    """Score one hypothesis over a fixed domain without invalid-edge leakage."""
    import cv2

    if valid.shape != rendered.shape[:2] or not valid.any():
        return None
    difference = np.linalg.norm(rendered - photo, axis=2)
    if edge_weight > 0.0:
        # Scharr reads a 3x3 neighbourhood.  Only use gradients whose complete
        # neighbourhood is usable, so HUD/excluded pixels cannot affect them.
        safe_gradient = valid & cv2.erode(
            valid.astype(np.uint8), np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
        if safe_gradient.any():
            difference[safe_gradient] += edge_weight * np.abs(
                _gradient(rendered)[safe_gradient] - photo_edges[safe_gradient]
            )
    return float(difference[valid].mean())


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


def _coordinate_ownership(
    ownership: np.ndarray, face_index: Sequence[tuple[BlockCoordinate, str]], coordinate: BlockCoordinate,
) -> np.ndarray:
    """Pixels assigned to any renderer face of one cell."""
    region = np.zeros(ownership.shape, dtype=bool)
    for identifier, (cell, _slot) in enumerate(face_index):
        if cell == coordinate:
            region |= ownership == identifier
    return region


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


def _face_patches(
    scene, camera, coordinate, boxes, image, scale,
    *, usable_mask: np.ndarray | None = None,
    owner_masks: Mapping[str, np.ndarray] | None = None,
):
    """Rectify the photograph through each of a candidate's own face quads.

    The shape decides the quad, so the patch this returns is what the camera would
    have seen *if the candidate were there*. That is exactly the input the
    classifier is built for, and it carries the grain that separates birch from
    oak where an average colour cannot.
    """
    from mc_photogrammetry.reconstruction.face_sampling import (
        quad_area, rectify_face_valid,
    )

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
            area = quad_area(quad)
            if area < MINIMUM_FACE_AREA:
                continue
            valid = np.ones((height, width), dtype=bool)
            if usable_mask is not None:
                if usable_mask.shape != (height, width):
                    raise ValueError("usable_mask must match the image dimensions")
                valid &= usable_mask
            if owner_masks is not None:
                owner = owner_masks.get(slot)
                # An explicitly supplied ownership map is authoritative.  A
                # missing slot means that no pixels are owned by this face.
                if owner is None:
                    continue
                if owner.shape != (height, width):
                    raise ValueError("owner mask must match the image dimensions")
                valid &= owner
            rectified = rectify_face_valid(image, quad, valid)
            if rectified is None:
                continue
            patch, valid_patch, coverage = rectified
            if int(valid_patch.sum()) < MINIMUM_PIXELS:
                continue
            found.append((slot, patch, float(np.sqrt(area)), valid_patch, coverage))
    return found


def _face_ownership(
    scene, camera, coordinate, state, context_blocks, colours, block_shapes,
    occluding_slots, model_faces, image_shape, scale,
) -> dict[str, np.ndarray]:
    """Upscale exact renderer ownership masks for one candidate's visible faces."""
    import cv2

    height, width = image_shape
    rendered, face_index = render_result(
        {**context_blocks, coordinate: state}, scene, camera, width, height,
        colours, block_shapes, scale, ambient=True,
        occluding_slots=occluding_slots, model_faces=model_faces,
    )
    masks: dict[str, np.ndarray] = {}
    for identifier, (cell, slot) in enumerate(face_index):
        if cell != coordinate:
            continue
        owned = rendered.ownership == identifier
        if owned.shape != (height, width):
            owned = cv2.resize(
                owned.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        masks[slot] = masks.get(slot, np.zeros((height, width), dtype=bool)) | owned
    return masks


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
    usable_masks: Mapping[str, np.ndarray] | None = None,
    diagnostics: MutableMapping[str, object] | None = None,
    max_candidate_renders: int | None = DEFAULT_MAX_CANDIDATE_RENDERS,
) -> dict[BlockCoordinate, BlockState | None]:
    """Best partial state for each cell, judged against the photographs.

    ``blocks`` is the scene the cells stand in; the cells themselves are excluded
    from the background so each is judged against the same picture.
    """
    import cv2

    if max_candidate_renders is not None and max_candidate_renders < 0:
        raise ValueError("max_candidate_renders cannot be negative")

    started = time.perf_counter()
    render_counts = {
        "background_renders": 0,
        "context_renders": 0,
        "probe_renders": 0,
        "ownership_renders": 0,
        "current_renders": 0,
        "candidate_renders": 0,
    }
    capped = False
    considered_shapes: list[str] = []
    material_refinements: list[str] = []
    unresolved_budget_exits: list[dict[str, object]] = []

    def count(name: str) -> None:
        render_counts[name] += 1

    shapes, materials = candidate_states(
        palette, families, include_full_cubes=True
    )
    if not shapes:
        return FitStateResult()
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
    source_masks: dict[str, np.ndarray] = {}
    scaled_masks: dict[str, np.ndarray] = {}
    for frame_id, image in frames.items():
        camera = cameras.get(frame_id)
        if camera is None:
            continue
        height, width = image.shape[:2]
        canvas, _o, _i, _d = render_volume(background_blocks, scene, camera, width,
                                           height, colours, block_shapes, scale,
                                           ambient=True, occluding_slots=occluding_slots,
                                           model_faces=model_faces)
        count("background_renders")
        source_mask, usable_mask = _frame_masks(
            frame_id, image.shape[:2], canvas.shape[:2], usable_masks,
        )
        # Downsample only usable source pixels.  A normal area resize would let
        # excluded pixels influence neighbouring valid pixels at the boundary.
        source_weight = source_mask.astype(np.float32)
        weighted = cv2.resize(
            image.astype(np.float32) * source_weight[..., None],
            (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_AREA,
        )
        weights = cv2.resize(
            source_weight, (canvas.shape[1], canvas.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        photo = np.divide(
            weighted, weights[..., None],
            out=np.zeros_like(weighted), where=weights[..., None] > 1e-6,
        )
        source_masks[frame_id] = source_mask
        scaled_masks[frame_id] = usable_mask
        lit = (canvas.mean(axis=2) > 1.0) & usable_mask
        gain = (float(np.median(photo[lit].mean(axis=1)))
                / max(float(np.median(canvas[lit].mean(axis=1))), 1e-6)) if lit.any() else 1.0
        backgrounds[frame_id] = canvas * gain
        photos[frame_id] = photo
        edges[frame_id] = _gradient(photo)
        gains[frame_id] = gain

    chosen: dict[BlockCoordinate, BlockState | None] = {}
    supporting_frames: dict[BlockCoordinate, list[str]] = {}
    working_blocks = dict(blocks)
    for coordinate in cells:
        current_state = working_blocks.get(coordinate)
        context_blocks = {
            cell: state for cell, state in working_blocks.items() if cell != coordinate
        }
        context_backgrounds: dict[str, np.ndarray] = {}
        for frame_id, camera in cameras.items():
            if frame_id not in backgrounds:
                continue
            height, width = frames[frame_id].shape[:2]
            context_backgrounds[frame_id], _owner, _index, _depth = render_volume(
                context_blocks, scene, camera, width, height, colours, block_shapes,
                scale, ambient=True, occluding_slots=occluding_slots,
                model_faces=model_faces,
            )
            # Air/current/candidates must use the same exposure convention.
            context_backgrounds[frame_id] *= gains[frame_id]
            count("context_renders")
        # Render this cell as a full cube once to determine the scoring region.
        # It is deliberately independent of every hypothesis. Pixels where the
        # cube is hidden by the surrounding scene cannot be evidence for any
        # candidate; pixels where a carpet leaves the cube empty remain in the
        # region and compare the background render with the photograph.
        probe = BlockState("minecraft:shape_fit_scoring_probe")
        views = []
        for frame_id, camera in cameras.items():
            if frame_id not in backgrounds:
                continue
            window = _window(scene, camera, coordinate,
                             backgrounds[frame_id].shape[:2], scale)
            if window is None:
                continue
            height, width = frames[frame_id].shape[:2]
            probe_render, probe_index = render_result(
                {**context_blocks, coordinate: probe}, scene, camera, width, height,
                colours, block_shapes, scale, ambient=True,
                occluding_slots=occluding_slots, model_faces=model_faces,
            )
            count("probe_renders")
            usable = _coordinate_ownership(probe_render.ownership, probe_index, coordinate)
            usable &= scaled_masks[frame_id]
            if int(usable.sum()) >= MINIMUM_PIXELS:
                views.append((frame_id, camera, usable))
        if not views:
            continue

        # Per shape: the silhouette's own verdict, and the material the grain
        # picks within it. Which of the two settles the shape is a flag.
        # Air is the unchanged background for an empty proposal. Existing
        # blocks get their own baseline below; either way, a proposal must beat
        # what is already there before it can alter the scene.
        air_scores = []
        for frame_id, _camera, usable in views:
            score = _score_image(
                context_backgrounds[frame_id], photos[frame_id], usable,
                edge_weight, edges[frame_id],
            )
            if score is not None:
                air_scores.append(score)
        if not air_scores:
            continue
        air_score = float(np.mean(air_scores))
        baseline_score = air_score
        if current_state is not None:
            current_scores = []
            current_blocks = {**context_blocks, coordinate: current_state}
            for frame_id, camera, usable in views:
                height, width = frames[frame_id].shape[:2]
                rendered, _face_index = render_result(
                    current_blocks, scene, camera, width, height, colours,
                    block_shapes, scale, ambient=True,
                    occluding_slots=occluding_slots, model_faces=model_faces,
                )
                count("current_renders")
                painted = rendered.colour * gains[frame_id]
                score = _score_image(
                    painted, photos[frame_id], usable, edge_weight, edges[frame_id],
                )
                if score is not None:
                    current_scores.append(score)
            if current_scores:
                baseline_score = float(np.mean(current_scores))

        # Spend the bounded render budget on one representative per geometry
        # hypothesis before considering material variants.  A complete vanilla
        # palette can contain hundreds of full-cube states; walking those first
        # used to exhaust the cap before a slab or carpet was ever rendered.
        full_cube_key = tuple(
            (tuple(round(v, 3) for v in low), tuple(round(v, 3) for v in high))
            for low, high in FULL_CUBE
        )
        shape_keys = sorted(
            shapes,
            key=lambda key: (key == full_cube_key, repr(key)),
        )
        representative_scores: dict[tuple, float] = {}
        representative_states: dict[tuple, BlockState] = {}
        for key in shape_keys:
            representative = materials[key][0]
            considered_shapes.append(representative.name)
            total, counted = 0.0, 0
            candidate_blocks = {**context_blocks, coordinate: representative}
            complete = True
            for frame_id, camera, usable in views:
                if (
                    max_candidate_renders is not None
                    and render_counts["candidate_renders"] >= max_candidate_renders
                ):
                    capped = True
                    complete = False
                    unresolved_budget_exits.append({
                        "coordinate": coordinate,
                        "phase": "shape_representative",
                    })
                    break
                height, width = frames[frame_id].shape[:2]
                rendered, _face_index = render_result(
                    candidate_blocks, scene, camera, width, height, colours,
                    block_shapes, scale, ambient=True,
                    occluding_slots=occluding_slots, model_faces=model_faces,
                )
                count("candidate_renders")
                if (
                    max_candidate_renders is not None
                    and render_counts["candidate_renders"] >= max_candidate_renders
                ):
                    capped = True
                painted = rendered.colour * gains[frame_id]
                score = _score_image(
                    painted, photos[frame_id], usable, edge_weight, edges[frame_id],
                )
                if score is not None:
                    total += score
                    counted += 1
            if not complete or counted != len(views):
                continue
            representative_scores[key] = total / counted
            representative_states[key] = representative
        if capped and not representative_scores:
            break

        ranked: list[tuple[float, float, float, BlockState | None]] = [
            (air_score, air_score, air_score, None)
        ]
        if current_state is not None:
            ranked.append((baseline_score, baseline_score, baseline_score, current_state))
        for key in shape_keys:
            if key not in representative_scores:
                continue
            boxes = shapes[key]
            # The grain the candidate's own faces would show, gathered once per
            # shape and scored for every material that shares it.
            texture_cost = None
            if index is not None:
                total_weight = 0.0
                pooled = np.zeros(index.count, dtype=np.float64)
                representative = materials[key][0]
                for frame_id, camera, _usable in views:
                    owner_masks = _face_ownership(
                        scene, camera, coordinate, representative, context_blocks,
                        colours, block_shapes, occluding_slots, model_faces,
                        frames[frame_id].shape[:2], scale,
                    )
                    count("ownership_renders")
                    mask = source_masks[frame_id]
                    for slot, patch, weight, _valid, coverage in _face_patches(
                            scene, camera, coordinate, boxes, frames[frame_id], scale,
                            usable_mask=mask, owner_masks=owner_masks):
                        pixels = patch.reshape(-1, 3)
                        weights = coverage.reshape(-1)
                        if float(weights.sum()) <= 1e-6:
                            continue
                        mean = np.average(pixels, axis=0, weights=weights)
                        costs, _best = index.costs(
                            mean, (), slot,
                            pattern=_normalised_pattern(patch, coverage), texture_weight=1.0)
                        if costs.size:
                            pooled += weight * costs
                            total_weight += weight
                if total_weight > 0.0:
                    texture_cost = pooled / total_weight

            plain_best, plain_state = None, None
            joint_best, joint_state = None, None
            plain_scores: dict[BlockState, float] = {}
            representative = representative_states[key]
            representative_score = representative_scores[key]
            representative_joint = representative_score
            if texture_cost is not None:
                position = positions.get(representative)
                if position is not None:
                    representative_joint += texture_weight * float(texture_cost[position])
            ranked.append((
                representative_score, representative_score, representative_joint,
                representative,
            ))
            plain_scores[representative] = representative_score
            plain_best, plain_state = representative_score, representative
            joint_best, joint_state = representative_joint, representative

            # Texture ranking is computed from rectified faces.  Render only the
            # best material for each geometry in this bounded refinement pass;
            # rendering every material variant is what starved rare shapes.
            material_candidates = list(materials[key])
            if texture_cost is not None:
                material_candidates.sort(key=lambda state: (
                    float(texture_cost[positions[state]]), state.name, state.properties,
                ))
            else:
                material_candidates = material_candidates[:1]
            for state in material_candidates[:1]:
                if state == representative:
                    continue
                if (
                    max_candidate_renders is not None
                    and render_counts["candidate_renders"] >= max_candidate_renders
                ):
                    capped = True
                    unresolved_budget_exits.append({
                        "coordinate": coordinate,
                        "phase": "material_refinement",
                    })
                    break
                table = colours.get(state.name, {})
                if not table:
                    continue
                total, counted = 0.0, 0
                candidate_blocks = {**context_blocks, coordinate: state}
                complete = True
                for frame_id, camera, usable in views:
                    if (
                        max_candidate_renders is not None
                        and render_counts["candidate_renders"] >= max_candidate_renders
                    ):
                        capped = True
                        complete = False
                        break
                    height, width = frames[frame_id].shape[:2]
                    rendered, _face_index = render_result(
                        candidate_blocks, scene, camera, width, height, colours, block_shapes, scale,
                        ambient=True, occluding_slots=occluding_slots, model_faces=model_faces,
                    )
                    count("candidate_renders")
                    if (
                        max_candidate_renders is not None
                        and render_counts["candidate_renders"] >= max_candidate_renders
                    ):
                        capped = True
                    material_refinements.append(state.name)
                    painted = rendered.colour * gains[frame_id]
                    score = _score_image(
                        painted, photos[frame_id], usable, edge_weight, edges[frame_id],
                    )
                    if score is not None:
                        total += score
                        counted += 1
                # A hypothesis scored on only a prefix of the views is not a
                # candidate: selective-view acceptance can manufacture a win.
                if not complete or counted != len(views):
                    continue
                if not counted:
                    continue
                plain = total / counted
                plain_scores[state] = plain
                score = plain
                if texture_cost is not None:
                    position = positions.get(state)
                    if position is not None:
                        score += texture_weight * float(texture_cost[position])
                if plain_best is None or plain < plain_best:
                    plain_best, plain_state = plain, state
                if joint_best is None or score < joint_best:
                    joint_best, joint_state = score, state
            if joint_state is not None:
                # The selected material must itself pass the rendered evidence
                # test; do not authorize it with another material's plain score.
                selected_plain = plain_scores[joint_state]
                shape_score = plain_best if shape_from_silhouette else joint_best
                ranked.append((shape_score, selected_plain, joint_best, joint_state))
            if capped:
                break
        if ranked:
            # Silhouette answers "which shape", grain answers "which material".
            # Letting the grain vote on the shape as well cost 2.9pp of family.
            order = 0 if shape_from_silhouette else 2
            _shape_score, selected_plain, _joint_score, best_state = min(
                ranked, key=lambda entry: entry[order]
            )
            # Keep air and the current state on ties or when a candidate does
            # not improve the rendered evidence. Returning only accepted edits
            # lets the caller preserve empty cells and existing geometry.
            if best_state is None and current_state is not None and selected_plain + 1e-6 < baseline_score:
                # Explicit deletion is distinct from "no proposal": the caller
                # must remove an occupied cell when air is the validated winner.
                chosen[coordinate] = None
                working_blocks.pop(coordinate, None)
                supporting_frames[coordinate] = [frame_id for frame_id, _camera, _usable in views]
            elif best_state is not None and best_state == current_state:
                # Report the retained state for callers that use this function
                # as a per-cell decision, while leaving the volume unchanged.
                chosen[coordinate] = current_state
                working_blocks[coordinate] = current_state
                supporting_frames[coordinate] = [frame_id for frame_id, _camera, _usable in views]
            elif (
                best_state is not None
                and selected_plain + 1e-6 < baseline_score
            ):
                chosen[coordinate] = best_state
                working_blocks[coordinate] = best_state
                supporting_frames[coordinate] = [frame_id for frame_id, _camera, _usable in views]
        if capped:
            break
    if diagnostics is not None:
        diagnostics.update({
            f"shape_fit_{name}": value for name, value in render_counts.items()
        })
        diagnostics["shape_fit_candidate_render_cap"] = max_candidate_renders
        diagnostics["shape_fit_candidate_render_cap_reached"] = capped
        diagnostics["shape_fit_render_work"] = sum(render_counts.values())
        diagnostics["shape_fit_seconds"] = round(time.perf_counter() - started, 6)
        diagnostics["shape_fit_considered_shapes"] = list(considered_shapes)
        diagnostics["shape_fit_material_refinements"] = list(material_refinements)
        diagnostics["shape_fit_unresolved_budget_exits"] = list(unresolved_budget_exits)
        diagnostics["shape_fit_supporting_frames"] = {
            str(coordinate): frames
            for coordinate, frames in supporting_frames.items()
        }
    return FitStateResult(chosen, supporting_frames=supporting_frames)
