"""Supported footprints built from a declared label, never from a texture cue.

The point of this module is what it does *not* do: no image, no patch, no
period, no phase enters `build_footprint` or `cell_to_block` at all - the
function signatures carry no such parameter, which is the structural
guarantee that swapping a photograph's texture (uniform, noisy, banded, or
anything else) cannot change a single result here. These tests check the pure
geometry instead: does a declared label resolve to the right export state,
does a block land on the correct side of a declared face, and does a real
litematic round trip preserve what this module actually claims to know.
"""
from pathlib import Path

import pytest

from mc_photogrammetry.reconstruction.footprint import (
    FOREGROUND_OCCLUSION,
    GEOMETRY_OPENING,
    OPENING,
    SUPPORTED,
    UNKNOWN,
    VISIBLE,
    build_footprint,
    cell_to_block,
    footprint_to_volume,
)


def test_astras_4x3_synthetic_control_resolves_10_supported_1_opening_1_unknown():
    """The exact scenario from the lead review: a 4x3 grid, one declared
    opening, one declared foreground-occluded cell, everything else visible.
    """
    declared = {(r, c): VISIBLE for r in range(3) for c in range(4)}
    declared[(1, 2)] = OPENING
    declared[(2, 0)] = FOREGROUND_OCCLUSION
    footprint = build_footprint(axis=2, face_coordinate=1, side=1.0,
                               lateral_origin=(0, 0), rows=3, columns=4,
                               declared=declared)
    assert len(footprint.of(SUPPORTED)) == 10
    assert len(footprint.of(GEOMETRY_OPENING)) == 1
    assert len(footprint.of(UNKNOWN)) == 1
    assert footprint.of(GEOMETRY_OPENING)[0].cell == (1, 2)
    assert footprint.of(UNKNOWN)[0].cell == (2, 0)
    assert footprint.of(UNKNOWN)[0].declared == FOREGROUND_OCCLUSION


def test_an_undeclared_cell_is_unknown_not_supported_or_opening():
    """A cell the declaration never addresses is a gap in the *declaration*,
    not a positive claim of either kind - exactly the mistake this module
    exists to avoid making again at a level above the pixel cue."""
    footprint = build_footprint(axis=1, face_coordinate=0, side=-1.0,
                               lateral_origin=(0, 0), rows=1, columns=1,
                               declared={})
    assert footprint.cells[0].geometry == UNKNOWN
    assert footprint.cells[0].declared is None


def test_cell_to_block_matches_astras_worked_example():
    """'plane z=1, outward normal +Z, with solids occupying cell z=0.'"""
    footprint = build_footprint(axis=2, face_coordinate=1, side=1.0,
                               lateral_origin=(5, 10), rows=1, columns=1,
                               declared={(0, 0): VISIBLE})
    block = cell_to_block(footprint, (0, 0))
    assert block == (5, 10, 0)


def test_cell_to_block_on_the_negative_side_places_the_block_at_the_face():
    """A face whose outward normal points -axis has the block's own near
    face - not one step further - exactly at the declared coordinate."""
    footprint = build_footprint(axis=1, face_coordinate=64, side=-1.0,
                               lateral_origin=(0, 0), rows=1, columns=1,
                               declared={(0, 0): VISIBLE})
    block = cell_to_block(footprint, (0, 0))
    assert block[1] == 64


def test_a_nonzero_lateral_origin_shifts_every_block_by_the_same_offset():
    """Repeats the 4x3 control with a nonzero rectification-window-style
    offset. The *shape* of the footprint (which local cells are supported)
    must be identical; only the absolute block coordinates move."""
    declared = {(r, c): VISIBLE for r in range(3) for c in range(4)}
    declared[(1, 2)] = OPENING
    declared[(2, 0)] = FOREGROUND_OCCLUSION

    zeroed = build_footprint(axis=2, face_coordinate=1, side=1.0,
                             lateral_origin=(0, 0), rows=3, columns=4,
                             declared=declared)
    offset = build_footprint(axis=2, face_coordinate=41, side=1.0,
                             lateral_origin=(100, -7), rows=3, columns=4,
                             declared=declared)

    zeroed_geometry = [c.geometry for c in zeroed.cells]
    offset_geometry = [c.geometry for c in offset.cells]
    assert zeroed_geometry == offset_geometry, "shape must not depend on origin"

    for cell in offset.of(SUPPORTED):
        base = cell_to_block(zeroed, cell.cell)
        shifted = cell_to_block(offset, cell.cell)
        assert shifted == (base[0] + 100, base[1] - 7, base[2] + 40)


def test_footprint_to_volume_round_trips_through_a_real_litematic(tmp_path: Path):
    """Export the 4x3 control to an actual .litematic and read it back -
    not a mock, the real exporter and the real reader already used
    throughout this project.
    """
    from mc_photogrammetry.domain.models import BlockState
    from mc_photogrammetry.export.compare import load_litematic
    from mc_photogrammetry.export.litematic import LitematicExporter

    declared = {(r, c): VISIBLE for r in range(3) for c in range(4)}
    declared[(1, 2)] = OPENING
    declared[(2, 0)] = FOREGROUND_OCCLUSION
    footprint = build_footprint(axis=2, face_coordinate=1, side=1.0,
                               lateral_origin=(0, 0), rows=3, columns=4,
                               declared=declared)
    volume = footprint_to_volume(footprint)

    destination = tmp_path / "control.litematic"
    LitematicExporter().export(volume, destination, name="footprint control",
                               description="M4 Part B synthetic control")
    snapshot = load_litematic(destination)

    for cell in footprint.of(SUPPORTED):
        block = cell_to_block(footprint, cell.cell)
        assert snapshot.blocks[block].name == "minecraft:light_gray_wool"
    opening_block = cell_to_block(footprint, footprint.of(GEOMETRY_OPENING)[0].cell)
    # An explicit air observation and a block the exporter never saw are the
    # same thing on read-back - the litematic format has no third state, so
    # this only confirms the opening reads as *not solid*, not that it is
    # distinguishable here from an unresolved cell. That distinction lives in
    # the evidence sidecar a caller saves alongside the volume, not in the
    # exported geometry itself - stated plainly in footprint.py's docstring.
    default_air = BlockState("minecraft:air")
    assert snapshot.blocks.get(opening_block, default_air).name == "minecraft:air"

    unknown_block = cell_to_block(footprint, footprint.of(UNKNOWN)[0].cell)
    assert unknown_block not in snapshot.blocks or snapshot.blocks[unknown_block].name == "minecraft:air"
