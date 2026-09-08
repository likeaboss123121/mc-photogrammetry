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
from mc_photogrammetry.domain.models import BlockCoordinate
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

    @property
    def pixels(self) -> np.ndarray:
        return self.patch.reshape(-1, 3) if self.valid is None else self.patch[self.valid]

    @property
    def observed_patch(self) -> np.ndarray:
        """Patch with invalid pixels neutralised for fixed-shape descriptors."""
        if self.valid is None or self.valid.all():
            return self.patch
        result = self.patch.copy()
        result[~self.valid] = self.pixels.mean(axis=0) if self.pixels.size else 0.0
        return result

    @property
    def rgb(self) -> np.ndarray:
        return self.pixels.mean(axis=0)


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
) -> tuple[np.ndarray, np.ndarray] | None:
    """Rectify a patch and the source-pixel ownership mask through one warp."""
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
    patch = cv2.warpPerspective(
        image, transform, (native, native), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    valid = cv2.warpPerspective(
        source_valid.astype(np.uint8), transform, (native, native), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(bool)
    if native != size:
        patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
        valid = cv2.resize(valid.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA) > 0.01
    return patch.astype(np.float32), valid


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
    """
    import cv2

    columns = max(1, int(round(width * scale)))
    rows = max(1, int(round(height * scale)))
    owner = np.full((rows, columns), -1, dtype=np.int32)
    nearest = np.full((rows, columns), -np.inf, dtype=np.float64)
    for index, (mean_depth, quad, depths, _coordinate, _slot, _facing, _texture_quad) in enumerate(quads):
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
) -> list[FaceSample]:
    """Rectify every block face this camera can actually see.

    Faces are depth-buffered against each other, dropped when the HUD overlaps
    them, and shrunk about their centre so a camera-pose error does not pull a
    neighbouring block into the sample. ``shrink`` trades resolution for that
    tolerance and should be set from how well the poses are actually known.
    """
    try:
        import cv2
    except ImportError:
        return []
    if not 0.0 < shrink <= 1.0:
        raise ValueError("shrink must be within (0, 1]")

    def reject(reason: str) -> None:
        if rejections is not None:
            rejections[reason] = rejections.get(reason, 0) + 1

    height, width = image.shape[:2]
    eye = np.asarray(camera.position, dtype=float) + np.asarray([0.0, camera.eye_height, 0.0])
    quads: list[tuple[float, np.ndarray, np.ndarray, BlockCoordinate, str, float, np.ndarray]] = []
    for coordinate in coordinates:
        center = np.asarray(scene.to_world(coordinate), dtype=float) + 0.5
        for slot in FACE_SLOTS:
            normal = np.asarray(FACE_BASIS[slot][0], dtype=float)
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
            if _hidden_by(neighbour, slot, coordinates, occluding_slots):
                reject("neighbour_occupied")
                continue
            world_corners = center + default_face_corners(slot)
            projected, _uv = project_clipped_polygon_uv(world_corners, camera, width=width, height=height)
            if len(projected) < 3:
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
    buffer = _nearest_surface(quads, width, height, depth_scale)

    samples: list[FaceSample] = []
    for index, (depth, quad, _depths, coordinate, slot, facing, texture_quad) in enumerate(quads):
        area = quad_area(quad)
        if area < minimum_area:
            reject("minimum_area")
            continue
        visible = float((buffer == index).sum()) / (depth_scale * depth_scale)
        # The raster buffer already clips to the frame. Compare surviving pixels
        # with the in-frame footprint, not the original face area, so an edge
        # face can still contribute its unobscured portion.
        footprint = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(footprint, quad.astype(np.int32), 1)
        in_frame_area = float(footprint.sum())
        if in_frame_area == 0.0:
            reject("off_frame")
            continue
        if visible < minimum_visible_fraction * in_frame_area:
            reject("visible_fraction")
            continue
        source_valid = buffer == index
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
        patch, valid = rectified
        if not valid.any():
            reject("gui_mask")
            continue
        reject("accepted")
        samples.append(
            FaceSample(
                coordinate, slot, patch, area, depth,
                min(1.0, visible / max(area, 1e-6)), facing, valid,
            )
        )
    return samples
