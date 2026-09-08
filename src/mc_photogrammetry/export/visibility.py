"""Which of a reference's blocks any camera could possibly have seen.

Every accuracy figure in this project divides by the whole reference, which
silently charges the reconstruction for blocks buried under furniture, walled
into a floor, or simply never pointed at. Those are not failures of the
algorithm; no method reading these photographs can recover them. The fixtures
deliberately contain such blocks.

This casts the reference itself through the manifest's cameras and keeps the
cells that survive: a **theoretical maximum** litematic that a run can be scored
against instead of the raw reference. The difference between the two numbers is
the part of the gap that better inference could actually close.

Correctness matters more than speed here, because the result becomes a
denominator. Three things are therefore done exactly rather than approximately:

* **A real depth buffer, not painter's algorithm.** Faces are not sorted and
  overpainted; every covered pixel gets its true camera depth from a ray-plane
  intersection and wins or loses on that. Sorting by a face's mean depth lets a
  large floor quad overwrite a small nearer one, which would delete a block that
  is plainly visible.
* **Shapes, not cubes.** A carpet occludes a sixteenth of what a cube would, so
  treating it as solid would hide the wall behind it and understate the maximum.
* **Occlusion culling only where it is provably safe.** A face lying on the cell
  wall is dropped only when the neighbour is a known full cube. `attribution`
  drops it for *any* occupied neighbour, which is right for a diagnostic and
  wrong here - it would hide the floor a carpet lies on.

A cell counts as reachable when at least one pixel in at least one frame lands on
it. That is the literal reading of "theoretically capable of being detected": one
pixel is not enough to identify a block, but it is evidence the block is there,
and anything stricter is a judgement about the algorithm rather than about the
data. `visible_cells` returns the pixel counts, so a stricter threshold is free.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from mc_photogrammetry.blocks.model_atlas import FACE_BASIS, FACE_SLOTS
from mc_photogrammetry.domain.models import BlockCoordinate, BlockState
from mc_photogrammetry.export.attribution import (
    CUBE,
    box_face_corners,
    face_colours,
    face_model_faces,
    face_occluding_slots,
    face_shapes,
    on_cell_wall,
    render_result,
)
from mc_photogrammetry.reconstruction.camera import (
    CameraModel,
    camera_eye,
    camera_rotation_matrix,
    camera_space_points,
    intrinsics_for,
    project_clipped_polygon,
)
from mc_photogrammetry.reconstruction.scene import SceneTransform

PARALLEL_EPSILON = 1e-9
"""Below this the ray runs along the face's plane and cannot meet it."""

NEAR_PLANE = 1e-4
"""Camera depth under which an intersection sits behind the lens."""

NEAR_CLIP = 1e-3
"""Camera depth the face polygon is clipped to before projecting.

A face with one corner behind the eye cannot be projected, and dropping the whole
face instead is not a conservative simplification - it deletes exactly the near
wall that should have occluded everything behind it. With cameras standing inside
a room, as every fixture here does, that leaked buried blocks into the maximum by
the hundred. Clipping the polygon keeps the part that is in front.
"""


def _camera_space(points, camera) -> np.ndarray:
    """World points as ``(view_x, rotated_y, depth)``, matching `project_points`."""
    return camera_space_points(points, camera)


def _clip_halfplane(polygon: np.ndarray, keep, where) -> np.ndarray:
    """Sutherland-Hodgman against one half-plane, interpolating the crossings.

    Both clips below are half-planes and a convex polygon stays convex through
    them, which is what lets the result go straight to `fillConvexPoly`.
    """
    if polygon.shape[0] == 0:
        return polygon
    inside = keep(polygon)
    if inside.all():
        return polygon
    if not inside.any():
        return polygon[:0]
    output = []
    for index in range(polygon.shape[0]):
        current, following = polygon[index], polygon[(index + 1) % polygon.shape[0]]
        here, there = inside[index], inside[(index + 1) % polygon.shape[0]]
        if here:
            output.append(current)
        if here != there:
            output.append(current + (following - current) * where(current, following))
    return np.asarray(output, dtype=np.float64) if output else polygon[:0]


def _project_clipped(corners, camera, width: int, height: int) -> np.ndarray:
    """The face's screen polygon, clipped to the near plane and then to the frame.

    The near clip is the one that was measured to matter; see `NEAR_CLIP`.
    Clipping to the frame afterwards keeps the rasterised coordinates bounded by
    the image rather than by how close the camera stands - a face a twentieth of
    a block from the eye otherwise projects past +/-80,000 - so the polygon handed
    to OpenCV is always the size of the region it actually covers.
    """
    projected = project_clipped_polygon(
        corners, camera, width=width, height=height, near_clip=NEAR_CLIP
    )
    if projected.shape[0] < 3:
        return projected[:, :2]
    screen = projected[:, :2]
    for axis, limit, lower in ((0, width, True), (0, width, False),
                               (1, height, True), (1, height, False)):
        bound = 0.0 if lower else float(limit)
        if lower:
            screen = _clip_halfplane(
                screen, lambda p, a=axis: p[:, a] >= 0.0,
                lambda u, v, a=axis: (0.0 - u[a]) / (v[a] - u[a]))
        else:
            screen = _clip_halfplane(
                screen, lambda p, a=axis, b=bound: p[:, a] <= b,
                lambda u, v, a=axis, b=bound: (b - u[a]) / (v[a] - u[a]))
        if screen.shape[0] < 3:
            return screen[:0]
    return screen


def ray_directions(
    xs: np.ndarray,
    ys: np.ndarray,
    camera: CameraModel,
    width: int,
    height: int,
) -> np.ndarray:
    """World directions through the given pixels, scaled so camera depth is 1.

    A vectorised `unproject_pixel` at unit depth. Because the scale is pinned to
    a camera depth of one, the ray parameter *is* the camera depth - the same
    quantity `project_points` returns - so it can be compared against a depth
    buffer directly and it interpolates linearly.
    """
    calibration = intrinsics_for(camera, width, height)
    view = np.column_stack((
        (np.asarray(xs, dtype=np.float64) - calibration.principal_point_pixels[0])
        / calibration.focal_length_pixels[0],
        -(np.asarray(ys, dtype=np.float64) - calibration.principal_point_pixels[1])
        / calibration.focal_length_pixels[1],
        np.ones(np.broadcast(np.asarray(xs), np.asarray(ys)).shape, dtype=np.float64),
    ))
    return view @ camera_rotation_matrix(camera)


def full_cube_keys(palette: Sequence) -> set:
    """Everything whose model fills its cell, and so hides the face behind it.

    Keyed by exact `BlockState` where the palette has one, and by bare name as a
    fallback - but a name is only admitted when *every* state carrying it is a
    full cube. `oak_slab` is a full cube as `type=double` and a half block
    otherwise, and a name-level guess in the optimistic direction would cull the
    face behind every slab in the fixture and understate the maximum.
    """
    found: set = set()
    by_name: dict[str, list[bool]] = {}
    for item in palette:
        cube = bool(getattr(item, "full_cube", True))
        by_name.setdefault(item.state.name, []).append(cube)
        if cube:
            found.add(item.state)
    found.update(name for name, flags in by_name.items() if all(flags))
    return found


def shape_table(palette: Sequence) -> dict:
    """Model boxes keyed by exact state, with a bare-name fallback.

    `attribution.face_shapes` keys by name alone and takes whichever state it
    meets first, which for a slab is a coin toss between a half block and a whole
    one. Here the reference supplies exact states, so they are worth using.
    """
    shapes: dict = {}
    for item in palette:
        boxes = tuple(getattr(item, "shape_boxes", ()) or CUBE)
        shapes[item.state] = boxes
        shapes.setdefault(item.state.name, boxes)
    return shapes


def _faces(blocks, scene, shapes, full_cubes, eye):
    """Every face that could turn toward the camera: (cell, axis, plane, corners)."""
    for coordinate, state in blocks.items():
        centre = np.asarray(scene.to_world(coordinate), dtype=float) + 0.5
        boxes = shapes.get(state) or shapes.get(state.name) or CUBE
        for low, high in boxes:
            for slot in FACE_SLOTS:
                normal = np.asarray(FACE_BASIS[slot][0], dtype=float)
                if float(normal @ (eye - centre)) <= 0.0:
                    continue
                if on_cell_wall(low, high, slot):
                    neighbour = (
                        coordinate[0] + int(normal[0]),
                        coordinate[1] + int(normal[1]),
                        coordinate[2] + int(normal[2]),
                    )
                    other = blocks.get(neighbour)
                    if other is not None and (
                            other in full_cubes or other.name in full_cubes):
                        continue
                corners = centre + box_face_corners(low, high, slot)
                axis = int(np.argmax(np.abs(normal)))
                yield coordinate, axis, float(corners[0][axis]), corners


def visible_cells(
    blocks: Mapping[BlockCoordinate, BlockState],
    scene: SceneTransform,
    cameras: Mapping[str, CameraModel],
    resolutions: Mapping[str, tuple[int, int]],
    *,
    full_cubes: set | None = None,
    shapes: Mapping | None = None,
    palette: Sequence | None = None,
    progress=None,
) -> dict[BlockCoordinate, int]:
    """Pixels each reference cell wins, summed across every camera.

    ``resolutions`` maps a frame id to its ``(width, height)``. A Minecraft
    screenshot's geometry depends on both, so the real frame size is used rather
    than a nominal one.
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover - the raycast needs a rasteriser
        raise RuntimeError("visibility raycasting needs OpenCV")

    shapes = {} if shapes is None else shapes
    full_cubes = set() if full_cubes is None else full_cubes
    colours = face_colours(palette) if palette is not None else None
    model_faces = face_model_faces(palette) if palette is not None else None
    occluding_slots = face_occluding_slots(palette) if palette is not None else None
    model_shapes = face_shapes(palette) if palette is not None else None
    totals: dict[BlockCoordinate, int] = {}

    for frame_id, camera in cameras.items():
        size = resolutions.get(frame_id)
        if size is None:
            continue
        width, height = int(size[0]), int(size[1])
        if colours is not None:
            rendered, face_index = render_result(
                blocks, scene, camera, width, height, colours, model_shapes, scale=1.0,
                occluding_slots=occluding_slots, model_faces=model_faces,
            )
            for identifier, count in zip(*np.unique(rendered.ownership[rendered.ownership >= 0], return_counts=True)):
                coordinate, _slot = face_index[int(identifier)]
                totals[coordinate] = totals.get(coordinate, 0) + int(count)
            if progress is not None:
                progress(frame_id, len(totals))
            continue
        eye = camera_eye(camera)
        depth = np.full((height, width), np.inf, dtype=np.float64)
        owner = np.full((height, width), -1, dtype=np.int64)
        cells: list[BlockCoordinate] = []

        for coordinate, axis, plane, corners in _faces(
                blocks, scene, shapes, full_cubes, eye):
            quad = _project_clipped(corners, camera, width, height)
            if quad.shape[0] < 3:
                continue
            x0 = max(int(np.floor(quad[:, 0].min())), 0)
            y0 = max(int(np.floor(quad[:, 1].min())), 0)
            x1 = min(int(np.ceil(quad[:, 0].max())) + 1, width)
            y1 = min(int(np.ceil(quad[:, 1].max())) + 1, height)
            if x1 <= x0 or y1 <= y0:
                continue

            stamp = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            local = np.round(quad - [x0, y0]).astype(np.int32)
            cv2.fillConvexPoly(stamp, local, 1)
            rows, columns = np.nonzero(stamp)
            if rows.size == 0:
                # Edge-on, thinner than a pixel. Fabricating a pixel for it
                # would invent evidence for a face nobody could resolve.
                continue
            gy, gx = rows + y0, columns + x0

            directions = ray_directions(gx + 0.5, gy + 0.5, camera, width, height)
            denominator = directions[:, axis]
            usable = np.abs(denominator) > PARALLEL_EPSILON
            if not usable.any():
                continue
            distance = np.full(denominator.shape, np.inf, dtype=np.float64)
            np.divide(plane - eye[axis], denominator, out=distance, where=usable)
            usable &= distance > NEAR_PLANE
            if not usable.any():
                continue

            gy, gx, distance = gy[usable], gx[usable], distance[usable]
            nearer = distance < depth[gy, gx]
            if not nearer.any():
                continue
            gy, gx = gy[nearer], gx[nearer]
            depth[gy, gx] = distance[nearer]
            owner[gy, gx] = len(cells)
            cells.append(coordinate)

        if cells:
            for index, count in zip(*np.unique(owner[owner >= 0], return_counts=True)):
                coordinate = cells[int(index)]
                totals[coordinate] = totals.get(coordinate, 0) + int(count)
        if progress is not None:
            progress(frame_id, len(totals))

    return totals


def theoretical_maximum(
    blocks: Mapping[BlockCoordinate, BlockState],
    counts: Mapping[BlockCoordinate, int],
    *,
    minimum_pixels: int = 1,
) -> dict[BlockCoordinate, BlockState]:
    """The reference restricted to the cells some camera actually reached."""
    return {
        coordinate: state
        for coordinate, state in blocks.items()
        if counts.get(coordinate, 0) >= minimum_pixels
    }
