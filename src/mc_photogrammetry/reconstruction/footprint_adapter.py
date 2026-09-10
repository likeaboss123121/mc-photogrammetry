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

RESOLVABLE_AREA_PIXELS = 1.0
"""A cell whose true (shoelace) projected area is smaller than one pixel
cannot be measured by counting whole raster pixels at all - even a single
rounded pixel is already several times the cell's own continuous area, so
no coverage ratio computed against it is trustworthy at any threshold.
Such a cell is never declared, regardless of what its raw ratio says -
"retain uncertainty below resolvable area" rather than report a number
integer rasterization cannot actually back up.
"""

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


def reproject_exported_volume(
    transform: LatticeTransform,
    footprint,
    litematic_path,
    *,
    row_offset: int = 0,
    column_offset: int = 0,
    minimum_shift: tuple[int, int, int] | None = None,
    _snapshot=None,
) -> tuple[dict[tuple[int, int], list[tuple[float, float]]], tuple[int, int, int]]:
    """Read a litematic back and independently reproject every solid block it
    actually contains through ``transform`` - not the pre-export footprint's
    own `cell_to_block`/`block_to_cell` round trip on the same in-memory
    cell set. A lead review named that gap directly: an earlier verification
    read the file only to confirm which cells came back, then recomputed
    their pixel corners from the *pre-export* cell set, never from the block
    values the file itself holds; every corner returned here traces back
    through a block tuple read straight out of ``litematic_path``.

    A follow-up review found a second, deeper gap in that same fix: every
    read-back block was passed through `block_to_cell`, which - correctly,
    for its own purpose - keeps only the two *lateral* coordinates and
    discards the normal-axis one entirely. A block sitting at the right
    lateral position but the wrong depth (corrupted, or from a nonplanar
    extrusion) would still produce a valid-looking cell and pass, since
    nothing ever checked the coordinate that would have caught it. Every
    block is now checked against the plane's own declared normal-axis
    placement (`cell_to_block`'s own ``axis_value`` formula - see its
    docstring) *before* it is reduced to a 2D cell at all; a block that
    fails this is not a supported cell with a slightly-off corner, it is
    not on the declared plane, and this raises rather than silently
    accepting it.

    Returns ``(reprojected, minimum_shift)``. ``reprojected`` maps each
    footprint-local ``(row, column)`` cell (`footprint.py`'s own indexing,
    from `block_to_cell`) the file actually contains to its pixel corners.
    ``minimum_shift`` is the exported volume's own minimum corner, added to
    a read-back local block coordinate before `block_to_cell` - a
    single-region litematic always stores its shape local to its own
    minimum corner rather than an absolute position (see `export.litematic`
    and `export.validation`; this is a tested format invariant, not a bug
    routed around here). Pass a previously-returned ``minimum_shift`` back
    in explicitly to verify a reload needs nothing recomputed from the
    footprint's own expected cells - an immutable value a caller can
    persist (e.g. in ``export_transform.json``) once and reuse; when
    omitted it is computed from ``footprint``'s pre-export `SUPPORTED`
    cells, as before.

    ``row_offset``/``column_offset`` translate `footprint.py`'s own local
    (0-indexed) cell space into ``transform``'s lattice coordinate space,
    for a caller (a real-image pilot, say) whose footprint was built
    against a translated sub-grid rather than starting at the transform's
    own cell (0, 0).

    Raises `ValueError` if any read-back block sits off the declared plane,
    or if the file's own solid-block cell set does not match the
    footprint's pre-export `SUPPORTED` cells.
    """
    from mc_photogrammetry.reconstruction.footprint import SUPPORTED, block_to_cell, cell_to_block
    from mc_photogrammetry.export.compare import load_litematic

    snapshot = _snapshot if _snapshot is not None else load_litematic(litematic_path)
    expected_cells = {c.cell for c in footprint.of(SUPPORTED)}
    if minimum_shift is None:
        exported_blocks = {cell_to_block(footprint, cell) for cell in expected_cells}
        minimum_shift = tuple(min(block[axis] for block in exported_blocks) for axis in range(3))

    expected_axis_value = (footprint.face_coordinate - 1 if footprint.side > 0
                           else footprint.face_coordinate)

    reprojected: dict[tuple[int, int], list[tuple[float, float]]] = {}
    off_plane: list[tuple[int, int, int]] = []
    for local_block, state in snapshot.blocks.items():
        if state.name == "minecraft:air":
            continue
        absolute_block = tuple(local_block[index] + minimum_shift[index] for index in range(3))
        if absolute_block[footprint.axis] != expected_axis_value:
            off_plane.append(absolute_block)
            continue
        local_cell = block_to_cell(footprint, absolute_block)
        world_cell = (local_cell[0] + row_offset, local_cell[1] + column_offset)
        corners = cell_corners_pixels(transform, world_cell)
        if corners is not None:
            reprojected[local_cell] = corners

    if off_plane:
        raise ValueError(
            f"{len(off_plane)} read-back block(s) sit off the declared plane "
            f"(axis {footprint.axis} must equal {expected_axis_value}): {sorted(off_plane)}")

    readback_cells = set(reprojected)
    if readback_cells != expected_cells:
        raise ValueError(
            f"read-back cells {sorted(readback_cells)} != pre-export supported cells "
            f"{sorted(expected_cells)}")
    return reprojected, minimum_shift


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
    `coverage_by_label` - a *certified lower bound*, never the raw estimate
    and never `coverage_of_in_frame` - so a declared label's true coverage is
    guaranteed to clear the threshold, not just estimated to."""
    outside_image_fraction: float
    conflict_fraction: float
    """Share of the cell's full area where a declared `OPENING` polygon and a
    declared `FOREGROUND_OCCLUSION` polygon overlap each other - two specific
    claims disagreeing about the same location. Excluded from every label's
    coverage rather than resolved by priority, since neither assertion is
    more authoritative than the other."""
    coverage_upper_by_label: dict[str, float] = field(default_factory=dict)
    """The same overlaps as `coverage_by_label`, but the certified *upper*
    bound rather than the lower one - a cell whose lower bound misses the
    threshold but whose upper bound clears it is a genuinely undecidable
    near-threshold case (both are reported so a caller can see that, rather
    than only a single number that looks like a confident miss)."""


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
    """Rasterize a polygon given in the same continuous convention
    `_polygon_area` uses - pixel ``(x, y)`` occupies ``[x, x+1) x [y, y+1)`` -
    into the integer pixel index grid `cv2.fillPoly` actually fills.

    `cv2.fillPoly` treats an integer vertex as sitting at a pixel's own
    index/centre and includes boundary pixels, not as a corner on the
    continuous grid. Feeding it continuous corner coordinates directly (as
    this module used to) silently reinterprets them as centres: a unit
    square with corners at the continuous coordinates ``(2, 2)`` and
    ``(3, 3)`` - true area 1 - was filled from pixel-centre ``2`` through
    pixel-centre ``3`` inclusive and rasterized as a 2x2, 4-pixel block,
    exactly the mismatch a lead review's independent control reproduced
    (`rasterize_cells` returning `coverage_by_label=4.0` for a true area-1
    cell). Shifting every coordinate by half a pixel before rounding
    aligns the two conventions: continuous corner ``2`` becomes centre-grid
    ``1.5``, which correctly falls between pixel centres 1 and 2 rather
    than landing on one.
    """
    import cv2

    mask = np.zeros(shape, dtype=np.uint8)
    points = np.array([[round(x - 0.5), round(y - 0.5)] for x, y in polygon], dtype=np.int32)
    cv2.fillPoly(mask, [points], 1)
    return mask


MAX_SUPERSAMPLED_CELLS = 4_000_000
"""Budget for one cell's local supersampled raster (crop area x factor^2).
Bounds the per-cell cost regardless of how finely `_supersample_factor`
would otherwise want to subdivide a small crop, and regardless of how
large a crop itself already is."""

MAX_SUPERSAMPLE_FACTOR = 2000
"""A safety ceiling on subdivision, not a target - `_supersample_factor`'s
own `sqrt(MAX_SUPERSAMPLED_CELLS / area)` formula already keeps a crop's
total cost within budget, so this only guards the degenerate zero/near-zero
area case from an unbounded result. Unlike the previous point-sampled
estimate, finer subdivision directly *tightens* the certified interval
(`BOUNDARY_BAND_SUBCELLS` worth of sub-pixels around every edge, in
continuous units, shrinks as the sub-pixel itself shrinks) - a small crop
should use all the resolution its own area allows within the shared budget,
not stop early the way the old "diminishing returns" reasoning assumed."""

BOUNDARY_BAND_SUBCELLS = 3
"""Width, in supersampled sub-pixels, of the margin drawn around every
polygon edge (`_fill_and_boundary`) before a sub-pixel is trusted to be
unambiguously inside or outside that polygon. A straight edge that only
grazes a sub-pixel's corner can fail to mark it through `cv2.polylines`'s
own line rasterization at a bare 1-subcell width, and `_rasterize_polygon`'s
half-pixel coordinate convention (see its own docstring) shifts every vertex
by up to half a sub-pixel again - 3 gives headroom for both without needing
either accounted for exactly. Widening this only makes the reported interval
more conservative (wider, never wrong in the unsafe direction), never less
correct - see `_fill_and_boundary`."""


def _supersample_factor(crop_shape: tuple[int, int]) -> int:
    """How finely to subdivide each pixel of a crop this size when measuring
    continuous overlap area - finer for a small, typical cell crop (where a
    lead review's counter-examples showed single-sample rasterization is
    badly wrong for anything not pixel-grid-aligned), coarser for a crop
    that is already most of the image (where the same relative error
    matters far less and a fine subdivision would not stay bounded).

    Returns 1 (no subdivision at all) once the crop's own *unsupersampled*
    pixel count already reaches the budget - `rasterize_cells` tiles that
    case in row-bands instead (`_tile_row_bounds`) rather than ever
    allocating an array sized to the crop itself, however large it is."""
    area = max(1, crop_shape[0] * crop_shape[1])
    if area >= MAX_SUPERSAMPLED_CELLS:
        return 1
    factor = int((MAX_SUPERSAMPLED_CELLS // area) ** 0.5)
    return max(1, min(MAX_SUPERSAMPLE_FACTOR, factor))


def _tile_row_bounds(crop_y0: int, crop_y1: int, crop_width: int, supersample: int) -> list[tuple[int, int]]:
    """Row-band boundaries covering ``[crop_y0, crop_y1)`` such that no
    single band's own supersampled array (``band_rows * crop_width *
    supersample**2``) exceeds `MAX_SUPERSAMPLED_CELLS`, however tall the
    crop itself is - the bound `_supersample_factor` enforces per-pixel is
    worthless if the crop is simply rasterized in one piece regardless of
    its own height, which is exactly the gap a lead review named directly:
    "the current factor>=1 fallback does not cap a crop already larger than
    four million samples". `rasterize_cells` processes one band at a time,
    accumulating each band's own area sums, so the full crop's mask never
    exists as one array."""
    sample_area = supersample * supersample
    per_row_cost = max(1, crop_width * sample_area)
    band_rows = max(1, MAX_SUPERSAMPLED_CELLS // per_row_cost)
    bounds = []
    row = crop_y0
    while row < crop_y1:
        end = min(crop_y1, row + band_rows)
        bounds.append((row, end))
        row = end
    return bounds


def _clip_polygon_to_rect(polygon, x0: float, y0: float, x1: float, y1: float):
    """Sutherland-Hodgman clip of a region polygon - which may extend far
    outside one cell's own crop, even across the whole image or beyond -
    down to that crop's rectangle, so rasterizing it at a finer local
    resolution stays bounded by the crop rather than by the polygon's own
    (potentially enormous) extent."""
    def clip(points, inside, intersect):
        if not points:
            return points
        result = []
        prev = points[-1]
        prev_inside = inside(prev)
        for curr in points:
            curr_inside = inside(curr)
            if curr_inside:
                if not prev_inside:
                    result.append(intersect(prev, curr))
                result.append(curr)
            elif prev_inside:
                result.append(intersect(prev, curr))
            prev, prev_inside = curr, curr_inside
        return result

    def edge_intersect(axis_index: int, value: float):
        def intersect(a, b):
            denominator = b[axis_index] - a[axis_index]
            t = (value - a[axis_index]) / denominator
            return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        return intersect

    points = list(polygon)
    points = clip(points, lambda p: p[0] >= x0, edge_intersect(0, x0))
    points = clip(points, lambda p: p[0] <= x1, edge_intersect(0, x1))
    points = clip(points, lambda p: p[1] >= y0, edge_intersect(1, y0))
    points = clip(points, lambda p: p[1] <= y1, edge_intersect(1, y1))
    return points


def _fill_and_boundary(
    polygons: list[list[tuple[float, float]]], crop_x0: int, crop_y0: int, crop_x1: int,
    crop_y1: int, supersample: int, high_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """A polygon group's *certain* and *possible* sub-pixel masks - a
    rigorous conservative interval, not a point estimate. A lead follow-up
    review found the previous point-sampled supersampling (one sample per
    sub-pixel, majority-vote in effect) was "not a confidence bound": its
    own exact counter-example (a cell 55% covered, true area below the 60%
    export threshold) still reported 61.23% and exported an unsupported
    cell, because a raw estimate has no guaranteed direction of error.

    For each polygon, `cv2.fillPoly` still gives the ordinary point-sampled
    fill, but `cv2.polylines` (drawn `BOUNDARY_BAND_SUBCELLS` sub-pixels
    thick around every edge) marks a *boundary band* around it. Away from
    that band, a straight polygon edge cannot pass through a sub-pixel
    without crossing the band that surrounds it - by the Jordan curve
    theorem, a sub-pixel disjoint from every edge is either entirely inside
    the polygon or entirely outside it, regardless of concavity, so its
    fill-mask sample is not just probably right there, it is *certified*
    right. `fill & ~boundary` is exactly this "certified inside" set (a
    guaranteed *lower* bound on the true area); `fill | boundary` is
    "possibly inside" (a guaranteed *upper* bound, since every genuinely
    covered sub-pixel is either certified or sits in the band). Both stay
    correct for a union of several polygons for one label: a sub-pixel away
    from *every* one of their bands has an unambiguous status for each, and
    OR-ing the per-polygon fills computes the union correctly at exactly the
    sub-pixels where that status is trustworthy.
    """
    import cv2

    fill = np.zeros(high_shape, dtype=np.uint8)
    boundary = np.zeros(high_shape, dtype=np.uint8)
    for polygon in polygons:
        clipped = _clip_polygon_to_rect(list(polygon), crop_x0, crop_y0, crop_x1, crop_y1)
        if len(clipped) < 3:
            continue
        local = [((px - crop_x0) * supersample, (py - crop_y0) * supersample) for px, py in clipped]
        fill |= _rasterize_polygon(local, high_shape)
        points = np.array([[round(px - 0.5), round(py - 0.5)] for px, py in local], dtype=np.int32)
        cv2.polylines(boundary, [points], True, 1, BOUNDARY_BAND_SUBCELLS, cv2.LINE_8)
    fill_bool = fill.astype(bool)
    boundary_bool = boundary.astype(bool)
    return fill_bool & ~boundary_bool, fill_bool | boundary_bool


def _tile_area_sums(
    cell_corners: list[tuple[float, float]], regions: list[RegionEvidence],
    crop_x0: int, band_y0: int, crop_x1: int, band_y1: int, supersample: int,
) -> dict[str, float]:
    """Certain/possible area sums (in original-image pixel units) for one
    cell against every declared label, restricted to one row-band of its
    crop - `rasterize_cells` accumulates these across bands so a crop far
    larger than `MAX_SUPERSAMPLED_CELLS` is never rasterized as one array
    (`_tile_row_bounds`); a crop that fits in one band is just the trivial
    single-tile case.

    ``OPENING`` and ``FOREGROUND_OCCLUSION`` are drawn over ``VISIBLE`` (a
    person typically draws a broad visible-extent polygon for a whole wall,
    then a smaller region inside it marking a specific doorway or occluder
    - the smaller, more specific claim is the one that applies where they
    overlap). A sub-pixel only counts toward a label's *certain* area when
    it is certainly that label *and* certainly not any label that overrides
    it (not even possibly, since "possibly overridden" is not "certainly
    still this label"); it counts toward *possible* area when it might be
    that label and is not *certainly* overridden. Where ``OPENING`` and
    ``FOREGROUND_OCCLUSION`` overlap each other, that is a genuine conflict
    between two specific claims - excluded from both, not resolved by
    priority.
    """
    high_shape = ((band_y1 - band_y0) * supersample, (crop_x1 - crop_x0) * supersample)
    sample_area = float(supersample * supersample)

    def group(label: str) -> list[list[tuple[float, float]]]:
        return [list(region.polygon) for region in regions if region.label == label]

    cell_certain, cell_possible = _fill_and_boundary(
        [cell_corners], crop_x0, band_y0, crop_x1, band_y1, supersample, high_shape)
    visible_certain, visible_possible = _fill_and_boundary(
        group(VISIBLE), crop_x0, band_y0, crop_x1, band_y1, supersample, high_shape)
    opening_certain, opening_possible = _fill_and_boundary(
        group(OPENING), crop_x0, band_y0, crop_x1, band_y1, supersample, high_shape)
    occlusion_certain, occlusion_possible = _fill_and_boundary(
        group(FOREGROUND_OCCLUSION), crop_x0, band_y0, crop_x1, band_y1, supersample, high_shape)

    label_certain = {
        VISIBLE: visible_certain & ~opening_possible & ~occlusion_possible,
        OPENING: opening_certain & ~occlusion_possible,
        FOREGROUND_OCCLUSION: occlusion_certain & ~opening_possible,
    }
    label_possible = {
        VISIBLE: visible_possible & ~opening_certain & ~occlusion_certain,
        OPENING: opening_possible & ~occlusion_certain,
        FOREGROUND_OCCLUSION: occlusion_possible & ~opening_certain,
    }
    conflict_certain = opening_certain & occlusion_certain
    conflict_possible = opening_possible & occlusion_possible

    sums = {
        "cell_certain": float(cell_certain.sum()) / sample_area,
        "cell_possible": float(cell_possible.sum()) / sample_area,
        "conflict_lower": float((conflict_certain & cell_certain).sum()) / sample_area,
        "conflict_upper": float((conflict_possible & cell_possible).sum()) / sample_area,
    }
    for label in (VISIBLE, OPENING, FOREGROUND_OCCLUSION):
        sums[f"{label}_lower"] = float((label_certain[label] & cell_certain).sum()) / sample_area
        sums[f"{label}_upper"] = float((label_possible[label] & cell_possible).sum()) / sample_area
    return sums


def rasterize_cells(
    transform: LatticeTransform,
    regions: list[RegionEvidence],
    image_size: tuple[int, int],
    rows: range,
    columns: range,
) -> list[CellEvidence]:
    """Project every candidate cell into image space and measure how much of
    it each declared region covers, against the cell's true full area
    (`_polygon_area`, exact shoelace - never a raster count divided by it,
    which only agrees when a shape happens to land on the pixel grid).

    `coverage_by_label` is a certified *lower* bound on the true coverage
    fraction, and `coverage_upper_by_label` an upper bound (`_tile_area_sums`
    / `_fill_and_boundary`) - not a single point estimate. A lead follow-up
    review's own exact counter-example (true coverage 55%, threshold 60%)
    showed a raw supersampled estimate is not a confidence bound: it read
    61.23% and exported an unsupported cell. `declared` is only ever set
    from the lower bound clearing `VISIBLE_COVERAGE_THRESHOLD`, so a
    declared label's true coverage is *guaranteed* to clear the threshold -
    a cell whose true coverage sits close enough to the threshold that this
    approximation cannot certify either side of it (including exactly at
    the threshold) stays undeclared rather than guessing, per the review's
    own "abstain if an approximation cannot certify it".

    Every raster allocation here is bounded by `MAX_SUPERSAMPLED_CELLS`
    regardless of a cell's own size: a crop that fits the budget at some
    supersample factor (`_supersample_factor`) is measured in one pass:
    one larger than the budget even with no subdivision at all - a lead
    review named this gap directly, "the current factor>=1 fallback does
    not cap a crop already larger than four million samples" - is instead
    processed in bounded row-bands (`_tile_row_bounds`), accumulating area
    sums across them, so the full crop's mask is never one array. A cell's
    projected quadrilateral can itself be far larger than the photograph (a
    near-grazing candidate, especially); nothing about its own extent is
    used to size an array either way.
    """
    width, height = image_size

    found: list[CellEvidence] = []
    for row in rows:
        for column in columns:
            cell = (row, column)
            corners = cell_corners_pixels(transform, cell)
            if corners is None:
                found.append(CellEvidence(cell, {}, {}, None, 1.0, 0.0, {}))
                continue
            full_area = _polygon_area(corners)
            if full_area <= 0:
                found.append(CellEvidence(cell, {}, {}, None, 1.0, 0.0, {}))
                continue

            xs = [p[0] for p in corners]
            ys = [p[1] for p in corners]
            crop_x0 = max(0, int(np.floor(min(xs))))
            crop_y0 = max(0, int(np.floor(min(ys))))
            crop_x1 = min(width, int(np.ceil(max(xs))) + 1)
            crop_y1 = min(height, int(np.ceil(max(ys))) + 1)
            if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
                found.append(CellEvidence(cell, {}, {}, None, 1.0, 0.0, {}))
                continue
            crop_shape = (crop_y1 - crop_y0, crop_x1 - crop_x0)
            supersample = _supersample_factor(crop_shape)

            if crop_shape[1] * supersample * supersample > MAX_SUPERSAMPLED_CELLS:
                # Even a single row-band would exceed the budget (the crop's
                # own width alone, at this supersample factor, already does)
                # - stop with an explicit unresolved cell rather than tiling
                # down to a fraction of a row or allocating past the budget.
                found.append(CellEvidence(cell, {}, {}, None, 1.0, 0.0, {}))
                continue

            totals: dict[str, float] = {}
            for band_y0, band_y1 in _tile_row_bounds(crop_y0, crop_y1, crop_shape[1], supersample):
                band_sums = _tile_area_sums(corners, regions, crop_x0, band_y0, crop_x1, band_y1,
                                            supersample)
                for key, value in band_sums.items():
                    totals[key] = totals.get(key, 0.0) + value

            # How much of the cell's own continuous area lies within the
            # photograph - a fact about the image frame, not about label
            # coverage, so measured by exactly clipping the cell's own
            # (convex) corners to the image rectangle (both convex,
            # `_clip_polygon_to_rect` is exact for that case) rather than
            # from `totals["cell_certain"]`, which carries the same
            # boundary-band margin every certified label area does and so
            # would never read exactly "fully in frame" even for a cell
            # nowhere near the image edge.
            in_frame_polygon = _clip_polygon_to_rect(corners, 0, 0, width, height)
            in_frame_area = _polygon_area(in_frame_polygon) if len(in_frame_polygon) >= 3 else 0.0

            coverage: dict[str, float] = {}
            coverage_upper: dict[str, float] = {}
            coverage_in_frame: dict[str, float] = {}
            for label in (VISIBLE, OPENING, FOREGROUND_OCCLUSION):
                coverage[label] = totals[f"{label}_lower"] / full_area
                coverage_upper[label] = min(1.0, totals[f"{label}_upper"] / full_area)
                coverage_in_frame[label] = (totals[f"{label}_lower"] / in_frame_area
                                            if in_frame_area else 0.0)
            conflict_fraction = totals["conflict_lower"] / full_area
            outside_fraction = max(0.0, 1.0 - in_frame_area / full_area)

            declared = None
            if full_area >= RESOLVABLE_AREA_PIXELS:
                for label in (OPENING, FOREGROUND_OCCLUSION, VISIBLE):
                    if coverage[label] >= VISIBLE_COVERAGE_THRESHOLD:
                        declared = label
                        break
            found.append(CellEvidence(cell, coverage, coverage_in_frame, declared,
                                      outside_fraction, conflict_fraction, coverage_upper))
    return found


def declared_labels_from_evidence(evidence: list[CellEvidence]) -> dict[tuple[int, int], str]:
    """The plain `dict[(row, column), label]` `footprint.build_footprint` wants,
    keeping only cells a region actually won - everything else is left absent,
    which `build_footprint` already resolves to `unknown` on its own."""
    return {e.cell: e.declared for e in evidence if e.declared is not None}
