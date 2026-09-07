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
from .camera import CameraModel, project_points
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

    @property
    def rgb(self) -> np.ndarray:
        return self.patch.reshape(-1, 3).mean(axis=0)


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
    warped = cv2.warpPerspective(image, transform, (native, native), flags=cv2.INTER_LINEAR)
    if native != size:
        warped = cv2.resize(warped, (size, size), interpolation=cv2.INTER_AREA)
    return warped.astype(np.float32)


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
    quads: list[tuple[float, np.ndarray, np.ndarray, BlockCoordinate, str, float]],
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
    for index, (mean_depth, quad, depths, _coordinate, _slot, _facing) in enumerate(quads):
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
    quads: list[tuple[float, np.ndarray, np.ndarray, BlockCoordinate, str, float]] = []
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
            projected = project_points(
                [tuple(point) for point in center + default_face_corners(slot)],
                camera,
                width=width,
                height=height,
            )
            if any(point is None for point in projected):
                reject("behind_camera")
                continue
            quad = np.asarray([[point[0], point[1]] for point in projected], dtype=np.float32)
            depths = np.asarray([point[2] for point in projected], dtype=np.float64)
            quads.append((float(depths.mean()), quad, depths, coordinate, slot, facing))
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
    for index, (depth, quad, _depths, coordinate, slot, facing) in enumerate(quads):
        area = quad_area(quad)
        if area < minimum_area:
            reject("minimum_area")
            continue
        if (
            quad[:, 0].min() < 1
            or quad[:, 0].max() >= width - 1
            or quad[:, 1].min() < 1
            or quad[:, 1].max() >= height - 1
        ):
            reject("off_frame")
            continue
        visible = float((buffer == index).sum()) / (depth_scale * depth_scale)
        if visible < minimum_visible_fraction * area:
            reject("visible_fraction")
            continue
        if gui_mask is not None and gui_mask.any():
            polygon = np.zeros(gui_mask.shape, dtype=np.uint8)
            cv2.fillConvexPoly(polygon, quad.astype(np.int32), 1)
            covered = polygon.astype(bool)
            if covered.any() and (covered & gui_mask).sum() > 0.02 * covered.sum():
                reject("gui_mask")
                continue
        centroid = quad.mean(axis=0)
        inner = (centroid + (quad - centroid) * shrink).astype(np.float32)
        patch = rectify_face(image, inner)
        if patch is None:
            reject("rectify_failed")
            continue
        reject("accepted")
        samples.append(
            FaceSample(
                coordinate, slot, patch, area, depth,
                min(1.0, visible / max(area, 1e-6)), facing,
            )
        )
    return samples
