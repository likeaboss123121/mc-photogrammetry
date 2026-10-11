"""Sample block faces from a frame by rectifying their projected quads.

A block face projects to a quadrilateral. Warping that quad to a square gives a
face-on view whose rows and columns follow the face's own texture axes, so a
sample is an average over exactly one face rather than over whatever happened to
sit under the block's centre pixel.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass

import numpy as np

from mc_photogrammetry.blocks.model_atlas import FACE_BASIS, FACE_SLOTS
from mc_photogrammetry.domain.models import BlockCoordinate, BlockState
from .camera import CameraModel, camera_space_points, intrinsics_for, project_clipped_polygon_uv
from .scene import SceneTransform

PATCH_SIZE = 16
DEPTH_SCALE = 1.0
"""Resolution of the visibility buffer, as a fraction of the frame.

Sampling and residual rendering must agree on which source pixels survive.
The former quarter-resolution default quantised thin faces into a few pixels,
which could both reject a visible face and disagree with the renderer. Callers
may still request a lower scale for an explicitly labelled preview.
"""

OPPOSITE_SLOT = {
    "up": "down", "down": "up", "north": "south",
    "south": "north", "east": "west", "west": "east",
}
"""The face of a neighbour that meets a given face of this block."""

MINIMUM_FACE_AREA = 24.0
DEFAULT_SHRINK = 0.35

CUBE_GEOMETRY = "cube_geometry"
EXACT_MODEL = "exact_model"
"""The two geometries one sampling call may use, never mixed within a call."""

STATES_UNAVAILABLE = "states_unavailable"
STATES_INCOMPLETE = "states_incomplete"
MODEL_METADATA_UNAVAILABLE = "model_metadata_unavailable"
"""Why a call stayed on the cube geometry instead of selecting exact models."""

PASS_SURFACE = "surface"
PASS_SURFACE_PROPOSED = "surface_proposed"
"""Pass names recorded with each geometry decision."""


@dataclass(frozen=True, slots=True)
class SamplingGeometryDecision:
    """The geometry one ``sample_visible_faces`` call resolved to, before sampling.

    Recorded per call in the reconstruction diagnostics so a reviewer can see
    whether exact baked-state geometry was selected, over how many targets, and -
    when it was not - exactly which prerequisite was missing. ``mode`` is either
    :data:`EXACT_MODEL` or :data:`CUBE_GEOMETRY`; ``fallback_reason`` is ``None``
    for an exact call and one of :data:`STATES_UNAVAILABLE`,
    :data:`STATES_INCOMPLETE` or :data:`MODEL_METADATA_UNAVAILABLE` otherwise.
    """

    pass_name: str
    mode: str
    target_count: int
    missing_state_count: int
    missing_model_count: int
    fallback_reason: str | None = None

    def to_record(self) -> dict[str, object]:
        """This decision as primitive, JSON-safe diagnostics fields.

        Diagnostics are written to disk and compared across runs, so the frozen
        decision itself must not leak into them: a dataclass needs a custom
        encoder, and a count that arrived as a numpy integer would round-trip
        through ``json`` as a different type. Every field here is a ``str``,
        ``int`` or ``None``, and ``fallback_reason`` stays ``None`` exactly when
        the call was exact. Live decisions remain typed records; only this
        projection is stored.
        """
        return {
            "pass_name": str(self.pass_name),
            "mode": str(self.mode),
            "target_count": int(self.target_count),
            "missing_state_count": int(self.missing_state_count),
            "missing_model_count": int(self.missing_model_count),
            "fallback_reason": (
                None if self.fallback_reason is None else str(self.fallback_reason)
            ),
        }


@dataclass(frozen=True, slots=True)
class FaceSample:
    """One rectified, face-on observation of a block face."""

    coordinate: BlockCoordinate
    slot: str
    patch: np.ndarray
    """``(PATCH_SIZE, PATCH_SIZE, 3)`` float32 RGB."""
    area: float
    depth: float
    visible: float = 1.0
    """Share of the face the camera actually sees, from the depth buffer."""
    facing: float = 1.0
    """Cosine between the face normal and the direction back to the camera.

    One is dead-on; near zero is edge-on, where the face lands on a sliver of
    pixels and its rectification stretches whatever noise it caught.
    """
    valid: np.ndarray | None = None
    """Face-patch pixels surviving ownership and GUI masking."""
    coverage: np.ndarray | None = None
    """Fractional source ownership surviving each rectified patch pixel."""

    @property
    def pixels(self) -> np.ndarray:
        return self.patch.reshape(-1, 3) if self.valid is None else self.patch[self.valid]

    @property
    def weights(self) -> np.ndarray:
        if self.coverage is None:
            return np.ones(len(self.pixels), dtype=np.float32)
        valid = np.ones(self.coverage.shape, dtype=bool) if self.valid is None else self.valid
        return self.coverage[valid]

    @property
    def observed_patch(self) -> np.ndarray:
        """Patch with invalid pixels neutralised for fixed-shape descriptors."""
        if self.valid is None or self.valid.all():
            return self.patch
        result = self.patch.copy()
        result[~self.valid] = self.rgb if self.pixels.size else 0.0
        return result

    @property
    def rgb(self) -> np.ndarray:
        pixels, weights = self.pixels, self.weights
        if not len(pixels) or float(weights.sum()) <= 1e-6:
            return np.zeros(3, dtype=np.float32)
        return np.average(pixels, axis=0, weights=weights).astype(np.float32)


def default_face_corners(slot: str) -> np.ndarray:
    """Block-local corners of an unrotated face, in texture-UV order."""
    _normal, origin, edge_u, edge_v = FACE_BASIS[slot]
    origin = np.asarray(origin, dtype=float)
    edge_u = np.asarray(edge_u, dtype=float)
    edge_v = np.asarray(edge_v, dtype=float)
    return np.asarray([origin, origin + edge_u, origin + edge_u + edge_v, origin + edge_v])


def quad_area(quad: np.ndarray) -> float:
    """Shoelace area of a projected quadrilateral."""
    return 0.5 * abs(
        float(np.dot(quad[:, 0], np.roll(quad[:, 1], -1)) - np.dot(quad[:, 1], np.roll(quad[:, 0], -1)))
    )


def rectify_face(image: np.ndarray, quad: np.ndarray, size: int = PATCH_SIZE) -> np.ndarray | None:
    """Warp a projected face quad into a square, face-on patch.

    ``cv2.warpPerspective`` has no area-averaging interpolation: asking it to
    write a large face straight into a 16x16 output point-samples the face and
    returns noise. Warp at roughly the face's on-screen resolution first, then
    area-average down.
    """
    if quad.shape != (4, 2) or size < 2:
        raise ValueError("quad must have shape (4, 2) and size must be at least 2")
    try:
        import cv2
    except ImportError:
        return None
    area = quad_area(quad)
    native = int(np.clip(round(np.sqrt(max(area, 1.0))), size, 8 * size))
    native = int(np.ceil(native / size) * size)
    destination = np.asarray(
        [[0.0, 0.0], [native, 0.0], [native, native], [0.0, native]], dtype=np.float32
    )
    transform = cv2.getPerspectiveTransform(quad.astype(np.float32), destination)
    # A face can extend outside the screenshot while still leaving a useful
    # interior patch. Replicating the nearest surviving edge is preferable to
    # OpenCV's black fill, which would otherwise manufacture a dark material.
    warped = cv2.warpPerspective(
        image, transform, (native, native), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if native != size:
        warped = cv2.resize(warped, (size, size), interpolation=cv2.INTER_AREA)
    return warped.astype(np.float32)


def rectify_face_valid(
    image: np.ndarray, quad: np.ndarray, source_valid: np.ndarray, size: int = PATCH_SIZE
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Rectify RGB, validity and fractional ownership through one warp.

    RGB is multiplied by source ownership before interpolation, then divided by
    the interpolated ownership. This prevents invalid black (or HUD) pixels from
    darkening a partially observed output pixel.
    """
    if source_valid.shape != image.shape[:2]:
        raise ValueError("source_valid must match the image dimensions")
    if quad.shape != (4, 2) or size < 2:
        raise ValueError("quad must have shape (4, 2) and size must be at least 2")
    try:
        import cv2
    except ImportError:
        return None
    area = quad_area(quad)
    native = int(np.clip(round(np.sqrt(max(area, 1.0))), size, 8 * size))
    native = int(np.ceil(native / size) * size)
    destination = np.asarray(
        [[0.0, 0.0], [native, 0.0], [native, native], [0.0, native]], dtype=np.float32
    )
    transform = cv2.getPerspectiveTransform(quad.astype(np.float32), destination)
    source_coverage = source_valid.astype(np.float32)
    weighted = image.astype(np.float32) * source_coverage[..., None]
    patch = cv2.warpPerspective(
        weighted, transform, (native, native), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    coverage = cv2.warpPerspective(
        source_coverage, transform, (native, native), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    if native != size:
        patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
        coverage = cv2.resize(coverage, (size, size), interpolation=cv2.INTER_AREA)
    valid = coverage > 0.01
    np.divide(patch, coverage[..., None], out=patch, where=valid[..., None])
    patch[~valid] = 0.0
    return patch.astype(np.float32), valid, coverage.astype(np.float32)


def _hidden_by(
    neighbour: BlockCoordinate,
    slot: str,
    occupied,
    occluding_slots=None,
) -> bool:
    """Whether the neighbour in a direction hides this block's face there.

    The neighbour hides it only if the neighbour's own opposing face is solid: a
    block's upward face is covered when whatever sits above it has a closed
    underside. A full cube always does, which is why treating occupancy alone as
    occlusion was right until partial models existed.
    """
    if occluding_slots is not None:
        covered = occluding_slots.get(neighbour)
        return covered is not None and OPPOSITE_SLOT[slot] in covered
    return neighbour in occupied


def _nearest_surface(
    quads: list[tuple[float, np.ndarray, np.ndarray, BlockCoordinate, str, float, np.ndarray]],
    width: int,
    height: int,
    scale: float,
) -> np.ndarray:
    """Index of the nearest face at each pixel, or -1 where nothing was drawn.

    Depth is compared per pixel rather than per face. Inverse depth is affine in
    screen space for a planar surface, so interpolating it across the quad is
    exact under perspective rather than an approximation.

    The buffer shape truncates ``width * scale`` and ``height * scale`` the way
    the shared renderer sizes its ownership buffer, with a one-pixel floor so a
    degenerate scale still leaves indexable storage. Rounding instead would give
    a different shape at fractional scales (99 rows becomes 50 rather than 49),
    so ``source_owner`` and the renderer's ``ownership`` would disagree about
    which pixels exist.
    """
    import cv2

    columns = max(1, int(width * scale))
    rows = max(1, int(height * scale))
    owner = np.full((rows, columns), -1, dtype=np.int32)
    nearest = np.full((rows, columns), -np.inf, dtype=np.float64)
    for index, entry in enumerate(quads):
        mean_depth, quad, depths, _coordinate, _slot, _facing = entry[:6]
        points = np.asarray(quad, dtype=np.float64) * scale
        left = max(0, int(np.floor(points[:, 0].min())))
        right = min(columns, int(np.ceil(points[:, 0].max())) + 1)
        top = max(0, int(np.floor(points[:, 1].min())))
        bottom = min(rows, int(np.ceil(points[:, 1].max())) + 1)
        if right <= left or bottom <= top:
            continue
        patch = np.zeros((bottom - top, right - left), dtype=np.uint8)
        cv2.fillConvexPoly(patch, np.round(points - [left, top]).astype(np.int32), 1)
        rows_hit, columns_hit = np.nonzero(patch)
        if rows_hit.size == 0:
            continue
        pixel_x = columns_hit + left
        pixel_y = rows_hit + top
        safe = np.maximum(np.asarray(depths, dtype=np.float64), 1e-6)
        matrix = np.column_stack((points[:3, 0], points[:3, 1], np.ones(3)))
        try:
            plane = np.linalg.solve(matrix, 1.0 / safe[:3])
            inverse_depth = plane[0] * pixel_x + plane[1] * pixel_y + plane[2]
        except np.linalg.LinAlgError:
            # Degenerate on screen, so it has no gradient worth interpolating.
            inverse_depth = np.full(pixel_x.shape, 1.0 / max(mean_depth, 1e-6))
        closer = inverse_depth > nearest[pixel_y, pixel_x]
        if not closer.any():
            continue
        chosen_y, chosen_x = pixel_y[closer], pixel_x[closer]
        nearest[chosen_y, chosen_x] = inverse_depth[closer]
        owner[chosen_y, chosen_x] = index
    return owner


def _owner_pixels_by_face(
    ownership: np.ndarray,
    face_index,
) -> dict[tuple[BlockCoordinate, str], np.ndarray]:
    """Flat pixel positions owned by each ``(coordinate, slot)`` in a render buffer.

    The renderer assigns every drawn pixel exactly one sequential face
    identifier, and several identifiers can describe the same cell/slot (a model
    with repeated same-slot faces). Grouping an identifier's pixels under its
    owner key and deduplicating within each key reproduces exactly the pixels an
    ``H x W`` boolean ``logical_or`` mask held per face, at one integer per owned
    pixel instead of one complete boolean frame retained per rendered face.
    Background (``-1``) pixels belong to no face and are dropped.
    """
    flat = ownership.reshape(-1)
    owned_pixels = np.flatnonzero(flat >= 0)
    grouped: dict[tuple[BlockCoordinate, str], list[np.ndarray]] = {}
    if owned_pixels.size:
        identifiers = flat[owned_pixels]
        # Sort so each identifier's pixels form one contiguous slice; stable
        # order keeps an owner's pixel list ascending, matching ``np.nonzero``.
        order = np.argsort(identifiers, kind="stable")
        sorted_identifiers = identifiers[order]
        sorted_pixels = owned_pixels[order]
        boundaries = np.flatnonzero(np.diff(sorted_identifiers)) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [sorted_identifiers.size]))
        for start, end in zip(starts, ends):
            owned = face_index[int(sorted_identifiers[start])]
            grouped.setdefault(owned, []).append(sorted_pixels[start:end])
    return {
        owned: pixels[0] if len(pixels) == 1 else np.unique(np.concatenate(pixels))
        for owned, pixels in grouped.items()
    }


def _is_block_coordinate(value: object) -> bool:
    """Whether a mapping key is a well-formed three-integer block coordinate."""
    if not isinstance(value, tuple) or len(value) != 3:
        return False
    return all(
        isinstance(part, (int, np.integer)) and not isinstance(part, bool)
        for part in value
    )


def validate_sampling_states(
    sampling_states: Mapping[BlockCoordinate, BlockState] | None,
) -> dict[BlockCoordinate, BlockState] | None:
    """Validate an existing-hypothesis snapshot, inventing and coercing nothing.

    ``None`` stays ``None``. A non-mapping raises ``TypeError``; a malformed
    coordinate raises ``ValueError`` and a non-state value raises ``TypeError``.
    The returned mapping is a plain ``dict`` with Python-integer coordinates, so
    later membership tests cannot be perturbed by a mapping that mutates or by
    numpy integer keys that compare differently.
    """
    if sampling_states is None:
        return None
    if not isinstance(sampling_states, Mapping):
        raise TypeError("sampling_states must be a mapping from coordinates to block states")
    validated: dict[BlockCoordinate, BlockState] = {}
    for coordinate, state in sampling_states.items():
        if not _is_block_coordinate(coordinate):
            raise ValueError("sampling_states coordinates must be tuples of three integers")
        if not isinstance(state, BlockState):
            raise TypeError("sampling_states values must be BlockState instances")
        validated[(int(coordinate[0]), int(coordinate[1]), int(coordinate[2]))] = state
    return validated


def exact_model_states(palette) -> frozenset:
    """Palette states carrying exact baked model polygons, matched by identity.

    Only the exact state is admitted. ``face_model_faces`` also exposes a
    name-keyed alias for callers that have nothing better, but selecting geometry
    through that alias would let one state's polygons stand in for a different
    state that merely shares its name, so it is deliberately not consulted here.
    """
    return frozenset(
        item.state
        for item in palette
        if tuple(getattr(item, "model_faces", ()) or ())
    )


def resolve_sampling_geometry(
    targets,
    sampling_states: Mapping[BlockCoordinate, BlockState] | None,
    palette,
    pass_name: str,
) -> SamplingGeometryDecision:
    """Decide one sampling call's geometry before it samples anything.

    A call may select :data:`EXACT_MODEL` only when every target coordinate has a
    supplied state and that state carries exact baked model polygons in this
    palette. Anything less keeps the whole call on the existing cube geometry -
    never a per-target or per-frame mixture - and names the missing prerequisite:

    * :data:`STATES_UNAVAILABLE` - no snapshot was supplied at all;
    * :data:`STATES_INCOMPLETE` - a snapshot exists but some target has no state;
    * :data:`MODEL_METADATA_UNAVAILABLE` - every target has a state, but at least
      one of those states has no exact model polygons in the palette.

    Entries in the snapshot outside ``targets`` are ignored, not counted and not
    turned into evidence.
    """
    target_set = set(targets)
    if sampling_states is None:
        return SamplingGeometryDecision(
            pass_name=pass_name,
            mode=CUBE_GEOMETRY,
            target_count=len(target_set),
            missing_state_count=len(target_set),
            missing_model_count=0,
            fallback_reason=STATES_UNAVAILABLE,
        )
    missing_states = sum(1 for coordinate in target_set if coordinate not in sampling_states)
    if missing_states:
        return SamplingGeometryDecision(
            pass_name=pass_name,
            mode=CUBE_GEOMETRY,
            target_count=len(target_set),
            missing_state_count=missing_states,
            missing_model_count=0,
            fallback_reason=STATES_INCOMPLETE,
        )
    modelled = exact_model_states(palette)
    missing_models = sum(1 for coordinate in target_set if sampling_states[coordinate] not in modelled)
    if missing_models:
        return SamplingGeometryDecision(
            pass_name=pass_name,
            mode=CUBE_GEOMETRY,
            target_count=len(target_set),
            missing_state_count=0,
            missing_model_count=missing_models,
            fallback_reason=MODEL_METADATA_UNAVAILABLE,
        )
    return SamplingGeometryDecision(
        pass_name=pass_name,
        mode=EXACT_MODEL,
        target_count=len(target_set),
        missing_state_count=0,
        missing_model_count=0,
        fallback_reason=None,
    )


def exact_sampling_blocks(
    decision: SamplingGeometryDecision,
    targets,
    sampling_states: Mapping[BlockCoordinate, BlockState] | None,
) -> dict[BlockCoordinate, BlockState] | None:
    """The complete coordinate→state mapping an exact call passes to the sampler.

    ``None`` for a cube-geometry decision, so the caller omits the sampler's
    ``blocks``/``palette`` arguments entirely and keeps the legacy ownership
    buffer. An exact decision guarantees every target has a state, so the mapping
    contains the whole target set: no target is dropped and no coordinate the
    snapshot mentions outside the targets is added.
    """
    if decision.mode != EXACT_MODEL:
        return None
    assert sampling_states is not None
    return {coordinate: sampling_states[coordinate] for coordinate in targets}


def sample_visible_faces(
    coordinates: set[BlockCoordinate],
    camera: CameraModel,
    scene: SceneTransform,
    image: np.ndarray,
    *,
    gui_mask: np.ndarray | None = None,
    minimum_area: float = MINIMUM_FACE_AREA,
    minimum_visible_fraction: float = 0.6,
    shrink: float = DEFAULT_SHRINK,
    rejections: MutableMapping[str, int] | None = None,
    occluding_slots: Mapping[BlockCoordinate, frozenset[str]] | None = None,
    depth_scale: float = DEPTH_SCALE,
    blocks: Mapping[BlockCoordinate, BlockState] | None = None,
    palette=None,
) -> list[FaceSample]:
    """Rectify every block face this camera can actually see.

    Faces are depth-buffered against each other, dropped when the HUD overlaps
    them, and shrunk about their centre so a camera-pose error does not pull a
    neighbouring block into the sample. ``shrink`` trades resolution for that
    tolerance and should be set from how well the poses are actually known.

    Supplying both ``blocks`` and ``palette`` switches ownership to the shared
    exact-model renderer. The caller still receives rectified face patches, but
    only pixels that the same alpha-tested model renderer attributes to that
    cell/slot are evidence. Coordinate-only callers retain the cube fallback
    while geometry is still unknown.
    """
    try:
        import cv2
    except ImportError:
        return []
    if not 0.0 < shrink <= 1.0:
        raise ValueError("shrink must be within (0, 1]")
    # Both ownership paths need this: the fallback buffer indexes itself by
    # scale, and the exact renderer refuses a scale outside (0, 1].
    if not 0.0 < depth_scale <= 1.0:
        raise ValueError("depth_scale must be within (0, 1]")
    if (blocks is None) != (palette is None):
        raise ValueError("blocks and palette must be supplied together for exact-model sampling")
    if blocks is not None:
        coordinates = set(blocks)

    def reject(reason: str) -> None:
        if rejections is not None:
            rejections[reason] = rejections.get(reason, 0) + 1

    height, width = image.shape[:2]
    eye = np.asarray(camera.position, dtype=float) + np.asarray([0.0, camera.eye_height, 0.0])
    quads: list[tuple[float, np.ndarray, np.ndarray, BlockCoordinate, str, float, np.ndarray]] = []
    exact_faces = None
    if blocks is not None:
        from mc_photogrammetry.export.attribution import face_model_faces

        exact_faces = face_model_faces(palette)
    for coordinate in coordinates:
        center = np.asarray(scene.to_world(coordinate), dtype=float) + 0.5
        state = None if blocks is None else blocks[coordinate]
        records = () if exact_faces is None else exact_faces.get(state, exact_faces.get(state.name, ()))
        specs = (
            ((record[0], np.asarray(record[2], dtype=float), center + np.asarray(record[1], dtype=float))
             for record in records)
            if records else
            ((slot, np.asarray(FACE_BASIS[slot][0], dtype=float), center + default_face_corners(slot))
             for slot in FACE_SLOTS)
        )
        for slot, normal, world_corners in specs:
            towards = eye - center
            if float(normal @ towards) <= 0.05:
                reject("back_facing")
                continue
            facing = float(normal @ towards / max(np.linalg.norm(towards), 1e-9))
            neighbour = (
                coordinate[0] + int(normal[0]),
                coordinate[1] + int(normal[1]),
                coordinate[2] + int(normal[2]),
            )
            if blocks is None and _hidden_by(neighbour, slot, coordinates, occluding_slots):
                reject("neighbour_occupied")
                continue
            projected, _uv = project_clipped_polygon_uv(world_corners, camera, width=width, height=height)
            if len(projected) < 3 or len(world_corners) != 4:
                reject("behind_camera")
                continue
            quad = projected[:, :2].astype(np.float32)
            depths = projected[:, 2]
            # The original UV square is still projective even where part of it
            # lies behind the lens. Its visible clipped polygon supplies the
            # source-valid mask, so border replication never becomes evidence.
            view = camera_space_points(world_corners, camera)
            if np.any(np.abs(view[:, 2]) <= 1e-6):
                reject("near_plane_vertex")
                continue
            calibration = intrinsics_for(camera, width, height)
            texture_quad = np.column_stack((
                calibration.principal_point_pixels[0] + view[:, 0] * calibration.focal_length_pixels[0] / view[:, 2],
                calibration.principal_point_pixels[1] - view[:, 1] * calibration.focal_length_pixels[1] / view[:, 2],
            )).astype(np.float32)
            quads.append((float(depths.mean()), quad, depths, coordinate, slot, facing, texture_quad))
    if not quads:
        return []

    # Which face owns each pixel, by depth rather than by paint order. Sorting
    # whole faces by their mean depth and painting back to front gets the order
    # wrong wherever two faces interpenetrate on screen, which is exactly what
    # neighbouring faces do at a grazing angle, and the reduced resolution then
    # quantised a face's visible share into a handful of samples. A face is
    # planar, so its inverse depth is affine across the screen and can be
    # interpolated exactly.
    buffer = None
    exact_owner_pixels: dict[tuple[BlockCoordinate, str], np.ndarray] = {}
    exact_owner_scratch: np.ndarray | None = None
    exact_owner_scratch_flat: np.ndarray | None = None
    exact_owner_shape: tuple[int, ...] | None = None
    if blocks is not None:
        from mc_photogrammetry.export.attribution import (
            face_colours, face_model_faces, face_occluding_slots, face_shapes, render_result,
        )

        rendered, face_index = render_result(
            blocks, scene, camera, width, height, face_colours(palette), face_shapes(palette),
            scale=depth_scale, occluding_slots=face_occluding_slots(palette),
            model_faces=face_model_faces(palette),
        )
        # Every owner mask must be the size the renderer actually produced, not
        # a shape recomputed from the scale: rounding and the renderer's
        # truncation disagree whenever the product is fractional. One scratch
        # mask is reused for every face instead of one full frame per face.
        exact_owner_shape = rendered.ownership.shape
        exact_owner_pixels = _owner_pixels_by_face(rendered.ownership, face_index)
        exact_owner_scratch = np.zeros(exact_owner_shape, dtype=bool)
        exact_owner_scratch_flat = exact_owner_scratch.reshape(-1)
    else:
        buffer = _nearest_surface(quads, width, height, depth_scale)

    footprint = np.zeros((height, width), dtype=np.uint8)
    samples: list[FaceSample] = []
    owner_pixels_seen: np.ndarray | None = None
    for index, (depth, quad, _depths, coordinate, slot, facing, texture_quad) in enumerate(quads):
        area = quad_area(quad)
        if area < minimum_area:
            reject("minimum_area")
            continue
        if blocks is not None:
            assert exact_owner_scratch is not None and exact_owner_scratch_flat is not None
            # Reconstruct this owner into the shared scratch, clearing only the
            # pixels the previous face set. Clearing the whole frame per face
            # would reintroduce the full-frame scan this index exists to avoid,
            # and a face with no owner must still read as all-false rather than
            # inherit its predecessor's pixels.
            if owner_pixels_seen is not None:
                exact_owner_scratch_flat[owner_pixels_seen] = False
            owner_pixels = exact_owner_pixels.get((coordinate, slot))
            if owner_pixels is None:
                owner_pixels_seen = None
                visible_pixels = 0
            else:
                exact_owner_scratch_flat[owner_pixels] = True
                owner_pixels_seen = owner_pixels
                visible_pixels = int(owner_pixels.size)
            source_owner = exact_owner_scratch
            visible = visible_pixels / (depth_scale * depth_scale)
        else:
            assert buffer is not None
            source_owner = buffer == index
            visible = float(source_owner.sum()) / (depth_scale * depth_scale)
        # The raster buffer already clips to the frame. Compare surviving pixels
        # with the in-frame footprint, not the original face area, so an edge
        # face can still contribute its unobscured portion. The footprint buffer
        # is reused too, so a call does not allocate a full frame per quad.
        footprint[:] = 0
        cv2.fillConvexPoly(footprint, quad.astype(np.int32), 1)
        in_frame_area = float(footprint.sum())
        if in_frame_area == 0.0:
            reject("off_frame")
            continue
        if visible < minimum_visible_fraction * in_frame_area:
            reject("visible_fraction")
            continue
        source_valid = source_owner
        if source_valid.shape != image.shape[:2]:
            source_valid = cv2.resize(
                source_valid.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        if gui_mask is not None:
            source_valid &= ~gui_mask
        centroid = texture_quad.mean(axis=0)
        inner = (centroid + (texture_quad - centroid) * shrink).astype(np.float32)
        rectified = rectify_face_valid(image, inner, source_valid)
        if rectified is None:
            reject("rectify_failed")
            continue
        patch, valid, coverage = rectified
        if not valid.any():
            reject("gui_mask")
            continue
        reject("accepted")
        samples.append(
            FaceSample(
                coordinate, slot, patch, area, depth,
                min(1.0, visible / max(area, 1e-6)), facing, valid, coverage,
            )
        )
    return samples
