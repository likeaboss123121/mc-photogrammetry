"""Supported cells built directly from a declared visibility label, not a cue.

`supported.py` asks the image whether it agrees with a plane hypothesis - it
is a *classifier*, and Part A spent a long session establishing, honestly,
that its per-cell classification does not separate real occupancy from not on
this project's data. Routing an "independent control" for the mask-to-cell
*export mechanics* through that same classifier does not test anything new:
it just re-runs the thing already found not to work, and reports its already-
known failure as if it were a finding about the export pipeline instead.

This module is the correction. A footprint here is built purely from a
caller's declared labels - `VISIBLE` (a person's positive assertion that a
region of the plane genuinely shows solid material), `OPENING` (a person's
positive assertion of a confirmed gap), `FOREGROUND_OCCLUSION` (something
nearer blocks the view; the true state behind it is not resolved by this
declaration), or simply absent (never addressed at all). No pixel is read.
No seam is scored. The only question this module answers is: given what a
person said they saw, does the geometry/export machinery construct the right
irregular footprint, place blocks on the correct side of the declared face,
and survive a litematic round trip? That is a different, narrower, and
answerable question from "does an automatic cue work."
"""
from __future__ import annotations

from dataclasses import dataclass

VISIBLE = "visible"
OPENING = "opening"
FOREGROUND_OCCLUSION = "foreground_occlusion"

SUPPORTED = "supported"
"""Resolved geometry label for a declared-VISIBLE cell."""
GEOMETRY_OPENING = "opening"
"""Resolved geometry label for a declared-OPENING cell: a confirmed gap."""
UNKNOWN = "unknown"
"""Resolved geometry label for a declared-FOREGROUND_OCCLUSION cell, or any
cell never declared at all. Declaring an occluder is not declaring what is
behind it - the two are different claims, and collapsing them into either
`supported` or `opening` would assert something the declaration did not."""

_GEOMETRY_FOR_DECLARED = {
    VISIBLE: SUPPORTED,
    OPENING: GEOMETRY_OPENING,
    FOREGROUND_OCCLUSION: UNKNOWN,
}


@dataclass(frozen=True)
class FootprintCell:
    """One cell of a declared footprint and its resolved geometry label."""

    cell: tuple[int, int]
    declared: str | None
    """Exactly what the caller supplied - `None` if the cell was never
    addressed at all, which resolves the same way `foreground_occlusion`
    does (unknown) but is worth keeping distinct in the record: one is a
    person actively noting an occluder, the other is a person's declaration
    simply not covering that cell."""
    geometry: str
    """`supported`, `opening`, or `unknown` - what the export step acts on."""


@dataclass(frozen=True)
class DeclaredFootprint:
    """A footprint plane: which world axis it is perpendicular to, which side
    of it the solid material sits on, and its cells' declared labels.

    ``lateral_origin`` is ``(u, v)``, the world coordinate of local cell
    (0, 0)'s low corner along the two axes other than ``axis``: ``u``
    addresses ``others[0]``, ``v`` addresses ``others[1]``, with ``others``
    taken in ascending world-axis-index order - the same convention
    `lattice.plane_basis` and `footprint_adapter.LatticeTransform` use, so a
    transform built by the adapter and a footprint built from its output
    describe the same grid rather than two independently-labelled ones. See
    `cell_to_block` for the row/column-to-u/v mapping this depends on. A
    footprint's own local grid is never assumed to start at world (0, 0); a
    nonzero origin is exercised explicitly in the tests, because that is
    exactly the kind of offset a texture-classifier-free path could still get
    wrong through pure arithmetic, independent of whether any cue works.
    """

    axis: int
    face_coordinate: int
    """The declared face's world coordinate along ``axis`` - not a distance
    from a camera, and not derived from any measurement in this module."""
    side: float
    """+1 or -1: which way the face's outward normal points along ``axis``.
    The solid material this footprint describes sits on the *opposite* side
    from the outward normal - see `cell_to_block`."""
    lateral_origin: tuple[int, int]
    cells: tuple[FootprintCell, ...]

    def of(self, geometry: str) -> tuple[FootprintCell, ...]:
        return tuple(c for c in self.cells if c.geometry == geometry)

    def bounds(self) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """Extent of the `supported` cells only - never a declared rectangle,
        for the same reason `supported.SupportedPlane.bounds` never uses one:
        a declared visible region can itself be irregular (an L-shaped wall
        segment, say), and its bounding box is not the same claim as its
        actual footprint."""
        found = self.of(SUPPORTED)
        if not found:
            return None
        rows = [c.cell[0] for c in found]
        columns = [c.cell[1] for c in found]
        return (min(rows), min(columns)), (max(rows), max(columns))


def build_footprint(
    axis: int,
    face_coordinate: int,
    side: float,
    lateral_origin: tuple[int, int],
    rows: int,
    columns: int,
    declared: dict[tuple[int, int], str],
) -> DeclaredFootprint:
    """Resolve every cell in a ``rows`` x ``columns`` grid from ``declared``.

    A cell not present in ``declared`` resolves to `UNKNOWN`, not to
    `SUPPORTED` and not to `GEOMETRY_OPENING` - an undeclared cell is not a
    positive claim of either kind, and treating a gap in the declaration as a
    hole (or as a fill) is exactly the mistake this whole line of work exists
    to avoid making at the pixel-cue level; it would be just as wrong to make
    it again here at the declaration level.
    """
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    if side not in (1.0, -1.0):
        raise ValueError("side must be +1.0 or -1.0")
    cells = []
    for row in range(rows):
        for column in range(columns):
            label = declared.get((row, column))
            geometry = _GEOMETRY_FOR_DECLARED.get(label, UNKNOWN)
            cells.append(FootprintCell((row, column), label, geometry))
    return DeclaredFootprint(axis=axis, face_coordinate=face_coordinate, side=side,
                             lateral_origin=lateral_origin, cells=tuple(cells))


def cell_to_block(footprint: DeclaredFootprint, cell: tuple[int, int]) -> tuple[int, int, int]:
    """The solid block coordinate a `supported` (or `opening`) cell describes.

    The declared face sits at ``face_coordinate`` along ``axis``, with its
    outward normal pointing in the ``side`` direction. The material this
    footprint describes is behind the face - on the *opposite* side from the
    normal: if the normal points +axis (``side=1``), the block is one step in
    the -axis direction from the face (``face_coordinate - 1``); if the
    normal points -axis (``side=-1``), the block's own near face already sits
    exactly at ``face_coordinate`` (a block occupies [c, c+1), so its face at
    the low end, the one facing -axis, is at c itself). Astra's example -
    "plane z=1, outward normal +Z, with solids occupying cell z=0" - is the
    ``side=1`` case: ``1 - 1 = 0``, checked directly in the tests.

    ``cell`` is ``(row, column)``; ``lateral_origin`` is ``(u, v)`` where
    ``u`` addresses ``others[0]`` and ``v`` addresses ``others[1]`` -
    ``others`` taken in ascending world-axis-index order, matching
    `lattice.plane_basis`'s own ``(normal, first, second)`` convention, where
    ``first = axis_matrix[others[0]]``. A lead review found this module used
    to place **row** on ``others[0]`` and **column** on ``others[1]`` here,
    while `footprint_adapter.cell_corners_lateral` - the only real source of
    `cell` values - places **column** on ``u``/``others[0]`` and **row** on
    ``v``/``others[1]``. Both sides used their own convention consistently,
    which is exactly why nothing caught the mismatch: a symmetric test cell
    where row and column carry the same value can't distinguish transposed
    axes from correct ones, and the modules were never checked against a
    single non-symmetric case until this row/column labelling was unified
    to match the adapter's.
    """
    axis_value = footprint.face_coordinate - 1 if footprint.side > 0 else footprint.face_coordinate
    others = [i for i in range(3) if i != footprint.axis]
    row, column = cell
    lateral = (footprint.lateral_origin[0] + column, footprint.lateral_origin[1] + row)
    block = [0, 0, 0]
    block[footprint.axis] = axis_value
    block[others[0]] = lateral[0]
    block[others[1]] = lateral[1]
    return tuple(block)


def block_to_cell(footprint: DeclaredFootprint, block: tuple[int, int, int]) -> tuple[int, int]:
    """The inverse of `cell_to_block`: a solid block coordinate back to its
    ``(row, column)`` cell.

    This exists so verification can start from what was actually *exported* -
    a block coordinate read back out of a real litematic file - rather than
    from the in-memory footprint that produced it. Reprojecting the pre-export
    grid only checks that the code agrees with itself; reprojecting a block
    coordinate that came from reading the file back also exercises the
    exporter and the loader, which is the gap a lead review named directly:
    "the overlay and scored geometry must originate from the read-back
    volume, not the pre-export grid."
    """
    others = [i for i in range(3) if i != footprint.axis]
    column = block[others[0]] - footprint.lateral_origin[0]
    row = block[others[1]] - footprint.lateral_origin[1]
    return (row, column)


NEUTRAL_PLACEHOLDER = "minecraft:light_gray_wool"
"""What `supported` cells export as: a visually distinct placeholder, not a
material guess. Nothing about this module infers what block is actually
there - only that the geometry is real."""


def footprint_to_volume(footprint: DeclaredFootprint, *, include_confirmed_openings: bool = False):
    """A `BlockVolume` for the footprint's resolved cells.

    ``supported`` cells get the neutral placeholder. ``unknown`` cells are
    always left out - not a hole, not a fill, unresolved. ``opening`` cells
    are left out too by default: an independent export/readback probe found
    that an explicit `minecraft:air` observation is still counted as a block
    by the existing exporter's metadata (`Metadata.TotalBlocks` came out 11
    for a control with 10 actual solids), so including it made the exported
    file's own header lie about how many blocks it holds. Pass
    ``include_confirmed_openings=True`` to get the old behaviour back if a
    caller specifically needs the air written into the volume itself - the
    default keeps the file's solid-block count honest and pushes the
    opening/unknown distinction fully onto the evidence sidecar.

    That sidecar dependency is a real, worth-stating limitation rather than a
    design choice this module can fully honour: the litematic format itself
    has no third state. A cell that is absent from the exported region reads
    back, on any loader, as air - identical, once read back, to a declared
    opening that was written as explicit air. Within the litematic alone, an
    unknown cell is indistinguishable from a confirmed opening either way.
    The evidence sidecar (the per-cell `declared`/`geometry` record this
    function does not produce, but that a caller should save alongside the
    volume) is where that distinction actually survives - not the exported
    geometry file, which this module does not pretend otherwise about.
    """
    from mc_photogrammetry.domain.models import BlockObservation, BlockState, BlockVolume

    volume = BlockVolume()
    for cell in footprint.cells:
        if cell.geometry == SUPPORTED:
            volume.set(cell_to_block(footprint, cell.cell),
                      BlockObservation(state=BlockState(NEUTRAL_PLACEHOLDER)))
        elif cell.geometry == GEOMETRY_OPENING and include_confirmed_openings:
            volume.set(cell_to_block(footprint, cell.cell),
                      BlockObservation(state=BlockState("minecraft:air")))
        # UNKNOWN: always omitted. OPENING: omitted by default - see above.
    return volume
