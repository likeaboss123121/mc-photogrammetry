"""Image-space semantic polygons to `footprint.py`'s declared cells.

`footprint.py` takes a cell already labelled `VISIBLE`/`OPENING`/
`FOREGROUND_OCCLUSION` and turns it into placed geometry - deliberately with
no image parameter anywhere in it, so no texture cue can ever reach it. This
module is the piece upstream of that: a person draws polygons on a
photograph, and this turns those into the labelled cells `footprint.py`
expects. It is still not a texture cue - a polygon is a person's positive
assertion, not a pixel measurement, and nothing in this module reads pixel
*values* at all, only where a person's drawn regions fall once projected
through a declared camera and plane.

One `LatticeTransform`, saved once, drives both directions: an image pixel to
a lattice cell (`pixel_to_lateral`, used to rasterize a person's polygons),
and a lattice coordinate back to an image pixel (`lateral_to_pixel`, used to
check that a projected cell corner lands where it should). Checking a
transform against itself both ways is what makes a reprojection test mean
anything - two independently-written formulas could agree by coincidence,
or disagree from a real bug either one introduced alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mc_photogrammetry.reconstruction.footprint import (
    FOREGROUND_OCCLUSION,
    OPENING,
    VISIBLE,
)
from mc_photogrammetry.reconstruction.lattice import plane_basis

VISIBLE_COVERAGE_THRESHOLD = 0.6
"""Share of a cell's projected area a single label must cover before that
label is declared for the cell. Below this, more than one thing is going on
inside the cell's own footprint - a boundary cell straddling an opening's
edge, say - and the cell stays undeclared (resolves to `unknown` downstream)
rather than one label winning a coin flip. Matches `supported.py`'s
`MINIMUM_COVERAGE` in spirit: a mask's bounding rectangle must not create
support a person did not actually draw over that specific cell.
"""


@dataclass(frozen=True)
class LatticeTransform:
    """Where a plane's cell grid sits, and how to move between an image pixel
    and a cell on it. Everything here is a declared fact, not a measurement -
    ``distance`` in particular is exactly the kind of thing `scale_from_known_length`
    already resolves from a stated fact, not something this module derives.
    """

    axis: int
    side: float
    distance: float
    axis_matrix: np.ndarray
    """Camera-space direction of each world axis - `calibration.axis_directions`
    for a real photograph, or a synthetic matrix for a synthetic test. This is
    the "image-derived camera optics" input; nothing here reads it from a
    manifest."""
    focal: float
    principal_point: tuple[float, float]
    lateral_origin: tuple[float, float] = (0.0, 0.0)
    """World-unit (u, v) coordinate of local cell (0, 0)'s low corner. Need not
    be an integer or aligned to anything - an arbitrary local origin is fine,
    and a fractional one is exactly what exercises real phase handling rather
    than only translation."""
    period: float = 1.0
    """Cell size in world units along both lateral axes. 1.0 is one block."""

    def __post_init__(self) -> None:
        if self.axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1 or 2")
        if self.side not in (1.0, -1.0):
            raise ValueError("side must be +1.0 or -1.0")
        if self.distance <= 0.0:
            raise ValueError("distance must be positive")


def pixel_to_lateral(transform: LatticeTransform, pixel: tuple[float, float]) -> tuple[float, float] | None:
    """An image pixel to (u, v) *real-world* lateral coordinates (in blocks)
    on the declared plane.

    Unprojects the pixel to a camera-space ray, then solves for where that ray
    crosses the plane ``point . normal = side * distance`` - the same plane
    equation `rectify_plane` is built from (see `plane_basis`). Returns `None`
    if the ray is parallel to the plane, or the intersection falls behind the
    camera - both genuinely unresolvable, not defaulted to a guessed cell.

    `plane_coordinates_to_camera_point`'s own ``u``/``v`` are coordinates on
    the *unit-distance* plane - dimensionless multiples that `rectify_plane`
    then scales by the real distance itself. A lattice period of "1.0" has to
    mean one real block, so this multiplies through by ``distance`` before
    returning - the scaling `lateral_to_pixel` undoes on the way back.
    """
    normal, first, second = plane_basis(transform.axis, transform.axis_matrix)
    px, py = pixel
    cx, cy = transform.principal_point
    ray = np.array(((px - cx) / transform.focal, -(py - cy) / transform.focal, 1.0))
    denominator = float(np.dot(ray, normal))
    if abs(denominator) < 1e-9:
        return None
    t = (transform.side * transform.distance) / denominator
    if t <= 1e-9:
        return None
    point = t * ray
    u = float(np.dot(point, first))
    v = float(np.dot(point, second))
    return (u, v)


def lateral_to_pixel(transform: LatticeTransform, lateral: tuple[float, float]) -> tuple[float, float] | None:
    """The inverse of `pixel_to_lateral`: a real-world lateral coordinate (in
    blocks) to the image pixel it projects to. `None` for a point behind the
    camera.
    """
    from mc_photogrammetry.reconstruction.lattice import plane_coordinates_to_camera_point

    u, v = lateral
    camera_point = plane_coordinates_to_camera_point(
        transform.axis, transform.axis_matrix, transform.side, transform.distance,
        u / transform.distance, v / transform.distance)
    if camera_point[2] <= 1e-9:
        return None
    px = transform.principal_point[0] + camera_point[0] * transform.focal / camera_point[2]
    py = transform.principal_point[1] - camera_point[1] * transform.focal / camera_point[2]
    return (px, py)


def lateral_to_cell(transform: LatticeTransform, lateral: tuple[float, float]) -> tuple[int, int]:
    """Which cell a lateral coordinate falls in, given the declared origin/period."""
    u, v = lateral
    row = int(np.floor((v - transform.lateral_origin[1]) / transform.period))
    column = int(np.floor((u - transform.lateral_origin[0]) / transform.period))
    return (row, column)


def cell_corners_lateral(transform: LatticeTransform, cell: tuple[int, int]) -> list[tuple[float, float]]:
    """A cell's four corners in (u, v) world-lateral coordinates, in order."""
    row, column = cell
    u0 = transform.lateral_origin[0] + column * transform.period
    u1 = u0 + transform.period
    v0 = transform.lateral_origin[1] + row * transform.period
    v1 = v0 + transform.period
    return [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]


def cell_corners_pixels(transform: LatticeTransform, cell: tuple[int, int]) -> list[tuple[float, float]] | None:
    """A cell's four corners projected to image pixels, or `None` if any
    corner falls behind the camera - a cell straddling the camera plane is
    not a projectable quadrilateral at all."""
    pixels = []
    for corner in cell_corners_lateral(transform, cell):
        pixel = lateral_to_pixel(transform, corner)
        if pixel is None:
            return None
        pixels.append(pixel)
    return pixels


@dataclass(frozen=True)
class RegionEvidence:
    """One image-space polygon a person drew, and what they said it means."""

    label: str
    """`VISIBLE`, `OPENING`, or `FOREGROUND_OCCLUSION` - `footprint.py`'s own labels."""
    polygon: tuple[tuple[float, float], ...]
    """Image pixel coordinates, in order."""


@dataclass(frozen=True)
class CellEvidence:
    """One candidate cell's rasterized coverage against every declared region."""

    cell: tuple[int, int]
    coverage_by_label: dict[str, float]
    """Each label's overlap as a fraction of the cell's *true, full* projected
    area (via the shoelace formula, not a raster count) - the denominator a
    decision is made against. A cell mostly outside the photograph cannot
    reach a high fraction here even if the sliver that *is* in frame is
    completely covered by one label, which is the property a lead review's
    clipping reproduction (`visible=1.0` on a cell 90% outside the image)
    found missing. Labels are mutually exclusive by construction (see
    `rasterize_cells`): a pixel `OPENING` and `FOREGROUND_OCCLUSION` both
    claim contributes to neither, not to whichever region a caller happened
    to list first.
    """
    coverage_of_in_frame: dict[str, float]
    """The same overlaps, but as a fraction of only the cell's in-frame
    pixels - how confident the visible portion alone looks, kept separately
    because it answers a different question than `coverage_by_label` and a
    lead review asked for both rather than one silently standing in for the
    other."""
    declared: str | None
    """The label whose `coverage_by_label` cleared `VISIBLE_COVERAGE_THRESHOLD`,
    or `None` if none did - a genuinely mixed, conflicting, or under-covered
    cell, left undeclared rather than guessed. Decided against
    `coverage_by_label` (the full-cell denominator), never
    `coverage_of_in_frame`."""
    outside_image_fraction: float
    conflict_fraction: float
    """Share of the cell's full area where a declared `OPENING` polygon and a
    declared `FOREGROUND_OCCLUSION` polygon overlap each other - two specific
    claims disagreeing about the same location. Excluded from every label's
    coverage rather than resolved by priority, since neither assertion is
    more authoritative than the other."""


def _polygon_area(polygon: list[tuple[float, float]]) -> float:
    """Shoelace formula: a polygon's signed area from its vertices alone, with
    no rasterization and so no allocation bounded by the polygon's own
    extent. A near-grazing cell can project to a quadrilateral spanning
    hundreds of thousands of pixels on a side; measuring its area this way
    costs the same four multiplications regardless."""
    area = 0.0
    for index in range(len(polygon)):
        x0, y0 = polygon[index]
        x1, y1 = polygon[(index + 1) % len(polygon)]
        area += x0 * y1 - x1 * y0
    return abs(area) / 2.0


def _rasterize_polygon(polygon: list[tuple[float, float]], shape: tuple[int, int]) -> "np.ndarray":
    import cv2

    mask = np.zeros(shape, dtype=np.uint8)
    points = np.array([[round(x), round(y)] for x, y in polygon], dtype=np.int32)
    cv2.fillPoly(mask, [points], 1)
    return mask


def _label_masks(regions: list[RegionEvidence], image_size: tuple[int, int]) -> dict[str, np.ndarray]:
    """One mutually-exclusive label per image pixel, bounded by the image
    itself - built once for the whole call, not per cell, and never sized by
    a cell's own (potentially enormous) projected extent.

    ``OPENING`` and ``FOREGROUND_OCCLUSION`` are drawn over ``VISIBLE``
    (a person typically draws a broad visible-extent polygon for a whole
    wall, then a smaller region inside it marking a specific doorway or
    occluder - the smaller, more specific claim is the one that applies
    where they overlap). Where ``OPENING`` and ``FOREGROUND_OCCLUSION``
    overlap *each other*, that is a genuine conflict between two specific
    claims - resolved to neither, not by which region a caller happened to
    list first.
    """
    width, height = image_size
    shape = (height, width)

    def union(label: str) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        for region in regions:
            if region.label == label:
                mask |= _rasterize_polygon([(p[0], p[1]) for p in region.polygon], shape)
        return mask

    visible = union(VISIBLE).astype(bool)
    opening = union(OPENING).astype(bool)
    occlusion = union(FOREGROUND_OCCLUSION).astype(bool)
    conflict = opening & occlusion

    return {
        VISIBLE: visible & ~opening & ~occlusion,
        OPENING: opening & ~occlusion,
        FOREGROUND_OCCLUSION: occlusion & ~opening,
        "_conflict": conflict,
    }


def rasterize_cells(
    transform: LatticeTransform,
    regions: list[RegionEvidence],
    image_size: tuple[int, int],
    rows: range,
    columns: range,
) -> list[CellEvidence]:
    """Project every candidate cell into image space and measure how much of
    it each declared region covers, against the cell's true full area.

    Every raster allocation here is bounded by ``image_size`` - the label
    masks once, up front, and each cell's own crop clipped to the image
    before it is ever rasterized. A cell's projected quadrilateral can be
    far larger than the photograph itself (a near-grazing candidate,
    especially); nothing about its own extent is used to size an array.
    """
    width, height = image_size
    masks = _label_masks(regions, image_size)

    found: list[CellEvidence] = []
    for row in rows:
        for column in columns:
            cell = (row, column)
            corners = cell_corners_pixels(transform, cell)
            if corners is None:
                found.append(CellEvidence(cell, {}, {}, None, 1.0, 0.0))
                continue
            full_area = _polygon_area(corners)
            if full_area <= 0:
                found.append(CellEvidence(cell, {}, {}, None, 1.0, 0.0))
                continue

            xs = [p[0] for p in corners]
            ys = [p[1] for p in corners]
            crop_x0 = max(0, int(np.floor(min(xs))))
            crop_y0 = max(0, int(np.floor(min(ys))))
            crop_x1 = min(width, int(np.ceil(max(xs))) + 1)
            crop_y1 = min(height, int(np.ceil(max(ys))) + 1)
            if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
                found.append(CellEvidence(cell, {}, {}, None, 1.0, 0.0))
                continue
            crop_shape = (crop_y1 - crop_y0, crop_x1 - crop_x0)
            local_corners = [(px - crop_x0, py - crop_y0) for px, py in corners]
            cell_mask = _rasterize_polygon(local_corners, crop_shape).astype(bool)
            in_frame_pixels = int(cell_mask.sum())

            coverage: dict[str, float] = {}
            coverage_in_frame: dict[str, float] = {}
            for label in (VISIBLE, OPENING, FOREGROUND_OCCLUSION):
                label_crop = masks[label][crop_y0:crop_y1, crop_x0:crop_x1]
                overlap = int((cell_mask & label_crop).sum())
                coverage[label] = overlap / full_area
                coverage_in_frame[label] = overlap / in_frame_pixels if in_frame_pixels else 0.0
            conflict_crop = masks["_conflict"][crop_y0:crop_y1, crop_x0:crop_x1]
            conflict_fraction = int((cell_mask & conflict_crop).sum()) / full_area
            outside_fraction = max(0.0, 1.0 - in_frame_pixels / full_area)

            declared = None
            for label in (OPENING, FOREGROUND_OCCLUSION, VISIBLE):
                if coverage[label] >= VISIBLE_COVERAGE_THRESHOLD:
                    declared = label
                    break
            found.append(CellEvidence(cell, coverage, coverage_in_frame, declared,
                                      outside_fraction, conflict_fraction))
    return found


def declared_labels_from_evidence(evidence: list[CellEvidence]) -> dict[tuple[int, int], str]:
    """The plain `dict[(row, column), label]` `footprint.build_footprint` wants,
    keeping only cells a region actually won - everything else is left absent,
    which `build_footprint` already resolves to `unknown` on its own."""
    return {e.cell: e.declared for e in evidence if e.declared is not None}
