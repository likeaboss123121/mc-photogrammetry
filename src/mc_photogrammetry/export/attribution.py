"""Attribute reconstruction error to blocks, by rendering both volumes and diffing.

Every other measure in this project counts *cells*, so a wrong block filling half
the frame and one glimpsed edge-on at forty blocks count the same. That is not how
the result looks and not how the error is distributed: on Template 2, partial
blocks are 8.1% of the cells and **76.3% of the visible error**.

This renders the reference and the candidate from the same pose, differences the
images, and traces each disagreeing pixel back into both volumes. The renderer does
not need to be faithful - both volumes go through the same one, so its bias cancels
in the difference - but it does need to draw each model's *boxes* rather than unit
cubes, because a carpet drawn as a solid cube covers a whole cell it never fills.

Causes come from where each ray stopped, not from whether it hit anything: both
volumes are usually closed, so both rays almost always hit something. Asking the
latter pinned ``missing`` and ``phantom`` at zero and hid their content inside a
catch-all.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from mc_photogrammetry.blocks.model_atlas import FACE_BASIS, FACE_SHADE, FACE_SLOTS
from mc_photogrammetry.domain.models import BlockCoordinate, BlockState
from mc_photogrammetry.reconstruction.camera import CameraModel, project_clipped_polygon
from mc_photogrammetry.reconstruction.lighting import ambient_occlusion
from mc_photogrammetry.reconstruction.scene import SceneTransform
from mc_photogrammetry.pipeline.contracts import RenderResult

RENDER_SCALE = 0.35
"""Fraction of the source resolution the diagnostic renders at."""

DIFFERENCE_THRESHOLD = 12.0
"""RGB distance below which two renders are treated as agreeing."""

DEPTH_SLACK = 0.15
"""Depth difference, in blocks, below which two hits count as the same surface."""

CUBE = (((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),)

MISSING = "missing"
PHANTOM = "phantom"
MATERIAL = "material"
DISPLACED = "displaced"
SILHOUETTE = "silhouette"

_AXIS_WALL = {
    "up": (1, 1, 1.0), "down": (1, 0, 0.0), "east": (0, 1, 1.0),
    "west": (0, 0, 0.0), "south": (2, 1, 1.0), "north": (2, 0, 0.0),
}

_OPPOSITE_SLOT = {
    "up": "down", "down": "up", "east": "west", "west": "east", "north": "south", "south": "north",
}


@dataclass(frozen=True)
class ErrorReport:
    """Visible error, split by cause and by the block responsible."""

    energy: dict[str, float] = field(default_factory=dict)
    pixels: dict[str, int] = field(default_factory=dict)
    blamed: dict[str, float] = field(default_factory=dict)
    confusions: dict[tuple[str, str], float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return sum(self.energy.values())

    def share(self) -> dict[str, float]:
        """Each cause as a fraction of the total, largest first."""
        total = max(self.total, 1e-9)
        return {
            cause: value / total
            for cause, value in sorted(self.energy.items(), key=lambda item: -item[1])
        }

    def worst_blocks(self, limit: int = 5) -> list[tuple[str, float]]:
        total = max(self.total, 1e-9)
        ranked = sorted(self.blamed.items(), key=lambda item: -item[1])
        return [(name, value / total) for name, value in ranked[:limit]]

    def worst_confusions(self, limit: int = 5) -> list[tuple[str, str, float]]:
        total = max(self.total, 1e-9)
        ranked = sorted(self.confusions.items(), key=lambda item: -item[1])
        return [(want, got, value / total) for (want, got), value in ranked[:limit]]


def face_colours(palette: Sequence) -> dict[object, dict[str, np.ndarray]]:
    """Per-state, per-slot RGB, taken from the palette the run already has."""
    table: dict[object, dict[str, np.ndarray]] = {}
    for item in palette:
        state = item.state
        mapping = dict(getattr(item, "face_map", ()) or ())
        default = np.asarray(item.rgb, dtype=float)
        colours = {
            slot: np.asarray(mapping.get(slot, item.rgb), dtype=float)
            for slot in FACE_SLOTS
        } if mapping else {slot: default for slot in FACE_SLOTS}
        table[state] = colours
        table.setdefault(state.name, colours)
    return table


def face_shapes(palette: Sequence) -> dict[object, tuple]:
    """Per-state model boxes, so a carpet is drawn as a carpet."""
    shapes: dict[object, tuple] = {}
    for item in palette:
        boxes = tuple(getattr(item, "shape_boxes", ()) or CUBE)
        shapes[item.state] = boxes
        shapes.setdefault(item.state.name, boxes)
    return shapes


def face_occluding_slots(palette: Sequence) -> dict[object, frozenset[str]]:
    """Faces that actually close a cell boundary, keyed by exact block state."""
    slots: dict[object, frozenset[str]] = {}
    for item in palette:
        covered = frozenset(getattr(item, "covered_slots", ()) or ())
        slots[item.state] = covered
        slots.setdefault(item.state.name, covered)
    return slots


def box_face_corners(low: Sequence[float], high: Sequence[float], slot: str) -> np.ndarray:
    """The four corners of one box face, relative to the cell centre."""
    x0, y0, z0 = low
    x1, y1, z1 = high
    if slot == "up":
        points = ((x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1))
    elif slot == "down":
        points = ((x0, y0, z0), (x0, y0, z1), (x1, y0, z1), (x1, y0, z0))
    elif slot == "east":
        points = ((x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0))
    elif slot == "west":
        points = ((x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1))
    elif slot == "south":
        points = ((x0, y0, z1), (x0, y1, z1), (x1, y1, z1), (x1, y0, z1))
    elif slot == "north":
        points = ((x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0))
    else:
        raise ValueError(f"unknown face slot: {slot}")
    return np.asarray([[value - 0.5 for value in point] for point in points], dtype=float)


def on_cell_wall(low, high, slot: str, tolerance: float = 1e-3) -> bool:
    axis, side, wall = _AXIS_WALL[slot]
    value = (high if side else low)[axis]
    return abs(value - wall) <= tolerance


def render_volume(
    blocks: Mapping[BlockCoordinate, BlockState],
    scene: SceneTransform,
    camera: CameraModel,
    width: int,
    height: int,
    colours: Mapping[object, Mapping[str, np.ndarray]],
    shapes: Mapping[object, tuple] | None = None,
    scale: float = RENDER_SCALE,
    ambient: bool = False,
    occluding_slots: Mapping[object, frozenset[str]] | None = None,
):
    """Painter render, with the cell and depth behind every pixel.

    Only faces lying on the cell wall can be hidden by a neighbour: a slab's top
    face at y=0.5 is interior to its own cell and always visible.

    ``ambient`` darkens each face the way vanilla smooth lighting does. Minecraft
    applies it whatever the gamma, so a recess reads dark even in a fullbright
    room, and a flat render puts it at full brightness. That mismatch is what caps
    the shape scorer: a candidate is judged on pixels whose brightness the renderer
    cannot otherwise predict, and the more of them it paints the worse it does.
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover - diagnostic only
        raise RuntimeError("attribution rendering needs OpenCV")
    if not 0.0 < scale <= 1.0:
        raise ValueError("scale must be within (0, 1]")

    rows, columns = int(height * scale), int(width * scale)
    canvas = np.zeros((rows, columns, 3), dtype=np.float32)
    owner = np.full((rows, columns), -1, dtype=np.int64)
    depth_buffer = np.full((rows, columns), np.inf, dtype=np.float32)
    occupied = set(blocks)
    eye = np.asarray(camera.position, dtype=float) + np.asarray([0.0, camera.eye_height, 0.0])

    faces = []
    for coordinate, state in blocks.items():
        centre = np.asarray(scene.to_world(coordinate), dtype=float) + 0.5
        boxes = CUBE if shapes is None else shapes.get(state, shapes.get(state.name, CUBE))
        for low, high in boxes:
            for slot in FACE_SLOTS:
                normal = np.asarray(FACE_BASIS[slot][0], dtype=float)
                if float(normal @ (eye - centre)) <= 0.05:
                    continue
                if on_cell_wall(low, high, slot):
                    neighbour = (
                        coordinate[0] + int(normal[0]),
                        coordinate[1] + int(normal[1]),
                        coordinate[2] + int(normal[2]),
                    )
                    neighbour_state = blocks.get(neighbour)
                    if neighbour_state is not None and (
                        occluding_slots is None
                        or _OPPOSITE_SLOT[slot] in occluding_slots.get(
                            neighbour_state, occluding_slots.get(neighbour_state.name, frozenset())
                        )
                    ):
                        continue
                projected = project_clipped_polygon(
                    centre + box_face_corners(low, high, slot), camera, width=width, height=height,
                )
                if len(projected) < 3:
                    continue
                quad = projected[:, :2].astype(np.float32) * scale
                colour = colours.get(state, colours.get(state.name, {})).get(slot)
                if colour is None:
                    colour = np.asarray((128.0, 128.0, 128.0))
                shade = FACE_SHADE.get(slot, 0.8)
                if ambient:
                    shade *= ambient_occlusion(coordinate, slot, occupied)
                faces.append((
                    quad,
                    projected[:, 2],
                    np.asarray(colour, dtype=float) * shade,
                    (coordinate, slot),
                ))

    # Each entry is (cell, slot): the residual pass needs the face's own normal to
    # know which way to propose, and a cell alone cannot say.
    index: list[tuple[BlockCoordinate, str]] = []
    for quad, depths, colour, owned in faces:
        stamp = np.zeros((rows, columns), dtype=np.uint8)
        cv2.fillConvexPoly(stamp, quad.astype(np.int32), 1)
        mask = stamp.astype(bool)
        if not mask.any():
            continue
        pixel_rows, pixel_columns = np.nonzero(mask)
        # A face is planar, so inverse depth is affine in screen space. This is
        # exact perspective depth at the raster sample and avoids the painter's
        # mean-depth failure where two slanted quads overlap.
        matrix = np.column_stack((quad[:3, 0], quad[:3, 1], np.ones(3)))
        try:
            plane = np.linalg.solve(matrix, 1.0 / np.maximum(depths[:3], 1e-6))
            inverse_depth = plane[0] * pixel_columns + plane[1] * pixel_rows + plane[2]
            pixel_depth = np.divide(
                1.0, inverse_depth, out=np.full_like(inverse_depth, np.inf), where=inverse_depth > 1e-9
            )
        except np.linalg.LinAlgError:
            pixel_depth = np.full(pixel_rows.shape, float(depths.mean()), dtype=np.float64)
        nearer = pixel_depth < depth_buffer[pixel_rows, pixel_columns]
        if nearer.any():
            chosen_rows, chosen_columns = pixel_rows[nearer], pixel_columns[nearer]
            canvas[chosen_rows, chosen_columns] = colour
            owner[chosen_rows, chosen_columns] = len(index)
            depth_buffer[chosen_rows, chosen_columns] = pixel_depth[nearer]
        index.append(owned)
    return canvas, owner, index, depth_buffer


def render_result(
    blocks: Mapping[BlockCoordinate, BlockState], scene: SceneTransform, camera: CameraModel,
    width: int, height: int, colours: Mapping[object, Mapping[str, np.ndarray]],
    shapes: Mapping[object, tuple] | None = None, scale: float = RENDER_SCALE,
    ambient: bool = False, occluding_slots: Mapping[object, frozenset[str]] | None = None,
) -> tuple[RenderResult, list[tuple[BlockCoordinate, str]]]:
    """Contract-shaped rendering result plus the face table used by ownership IDs."""
    colour, ownership, index, depth = render_volume(
        blocks, scene, camera, width, height, colours, shapes, scale, ambient, occluding_slots
    )
    return RenderResult(colour=colour, depth=depth, ownership=ownership, valid_pixels=ownership >= 0), index


def attribute_error(
    reference: Mapping[BlockCoordinate, BlockState],
    candidate: Mapping[BlockCoordinate, BlockState],
    scene: SceneTransform,
    cameras: Mapping[str, CameraModel],
    frames: Mapping[str, np.ndarray],
    palette: Sequence,
    *,
    scale: float = RENDER_SCALE,
    threshold: float = DIFFERENCE_THRESHOLD,
    slack: float = DEPTH_SLACK,
) -> ErrorReport:
    """Where the reconstruction visibly differs from the reference, and why."""
    colours = face_colours(palette)
    shapes = face_shapes(palette)
    occluding_slots = face_occluding_slots(palette)
    energy: dict[str, float] = defaultdict(float)
    pixels: dict[str, int] = defaultdict(int)
    blamed: dict[str, float] = defaultdict(float)
    confusions: dict[tuple[str, str], float] = defaultdict(float)

    for frame_id, image in frames.items():
        camera = cameras.get(frame_id)
        if camera is None:
            continue
        height, width = image.shape[:2]
        want, want_owner, want_index, want_depth = render_volume(
            reference, scene, camera, width, height, colours, shapes, scale, occluding_slots=occluding_slots)
        got, got_owner, got_index, got_depth = render_volume(
            candidate, scene, camera, width, height, colours, shapes, scale, occluding_slots=occluding_slots)
        difference = np.linalg.norm(want - got, axis=2)
        for row, column in zip(*np.nonzero(difference > threshold)):
            here = int(want_owner[row, column])
            there = int(got_owner[row, column])
            if here < 0 and there < 0:
                continue
            weight = float(difference[row, column])
            if here < 0 or there < 0:
                cause = SILHOUETTE
                cell = (got_index[there] if here < 0 else want_index[here])[0]
                source = candidate if here < 0 else reference
                blamed[source[cell].name] += weight
            else:
                wanted, obtained = want_index[here][0], got_index[there][0]
                wanted_name = reference[wanted].name
                obtained_name = candidate[obtained].name
                nearer = float(want_depth[row, column]) - float(got_depth[row, column])
                if wanted == obtained:
                    cause = MATERIAL
                elif nearer > slack:
                    cause = PHANTOM
                elif nearer < -slack:
                    cause = MISSING
                else:
                    cause = DISPLACED
                blamed[wanted_name] += weight
                confusions[(wanted_name, obtained_name)] += weight
            energy[cause] += weight
            pixels[cause] += 1

    return ErrorReport(dict(energy), dict(pixels), dict(blamed), dict(confusions))
