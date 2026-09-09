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


def test_cell_to_block_matches_a_nonsymmetric_hand_computed_case_per_axis():
    """A lead review found footprint.py placed row on others[0] and column on
    others[1], while footprint_adapter.py - the only real source of cell
    values - places column on others[0] (u) and row on others[1] (v). Every
    prior test used either a symmetric cell (row == column) or a uniform
    shift applied to both, neither of which can tell a transposed axis
    mapping apart from a correct one. These do: row and column differ, and
    the expected block is computed by hand, independently, for two different
    axes and both face signs.
    """
    # axis=0 (X), side=+1, face_coordinate=5, origin=(u=2,v=3), cell=(row=1,column=4).
    # others=[1,2]=(Y,Z). u=origin_u+column=2+4=6 -> Y. v=origin_v+row=3+1=4 -> Z.
    # axis_value(X) = face_coordinate-1 = 4 (side>0).
    fp = build_footprint(axis=0, face_coordinate=5, side=1.0, lateral_origin=(2, 3),
                         rows=5, columns=5, declared={(1, 4): VISIBLE})
    assert cell_to_block(fp, (1, 4)) == (4, 6, 4)

    # axis=1 (Y), side=-1, face_coordinate=10, origin=(0,0), cell=(row=2,column=7).
    # others=[0,2]=(X,Z). u=0+7=7 -> X. v=0+2=2 -> Z. axis_value(Y)=10 (side<0).
    fp2 = build_footprint(axis=1, face_coordinate=10, side=-1.0, lateral_origin=(0, 0),
                          rows=5, columns=10, declared={(2, 7): VISIBLE})
    assert cell_to_block(fp2, (2, 7)) == (7, 10, 2)


def test_cell_to_block_agrees_with_the_adapters_own_lateral_convention():
    """Cross-checks footprint.py against footprint_adapter.py directly,
    rather than each module's own independently-typed expectation: given the
    same axis/side/origin/period, cell_to_block's (u, v) decomposition must
    equal cell_corners_lateral's low corner for the same cell.
    """
    from mc_photogrammetry.reconstruction.calibration import axis_directions
    from mc_photogrammetry.reconstruction.footprint_adapter import (
        LatticeTransform, cell_corners_lateral)

    axes = axis_directions(5.0, -3.0, 2.0)
    transform = LatticeTransform(axis=2, side=1.0, distance=8.0, axis_matrix=axes,
                                 focal=800.0, principal_point=(0.0, 0.0),
                                 lateral_origin=(2.0, 3.0), period=1.0)
    footprint = build_footprint(axis=2, face_coordinate=1, side=1.0,
                                lateral_origin=(2, 3), rows=5, columns=7,
                                declared={(1, 4): VISIBLE})
    others = [1, 0]  # axis=2 -> others = [0, 1]; block[0]=X, block[1]=Y for u,v resp.
    block = cell_to_block(footprint, (1, 4))
    u0, v0 = cell_corners_lateral(transform, (1, 4))[0]
    assert (block[0], block[1]) == (u0, v0)


def test_block_to_cell_inverts_cell_to_block_exactly():
    """block_to_cell exists specifically to verify read-back geometry - it
    has to be the exact inverse of cell_to_block, checked here across the
    same nonsymmetric, multi-axis cases as the transposition regression
    test above, not just a round trip on one convenient cell.
    """
    from mc_photogrammetry.reconstruction.footprint import block_to_cell

    # lateral_origin here is footprint.py's *export* origin - always integer,
    # since a real block coordinate has to be. It is a separate, arbitrary
    # choice from the adapter's continuous, possibly-fractional camera-space
    # origin; see G3 in the lead review for why those two are not the same
    # value and the relationship between them has to stay explicit.
    cases = [
        (0, 5, 1.0, (2, 3), (1, 4)),
        (1, 10, -1.0, (0, 0), (2, 7)),
        (2, 1, 1.0, (-5, 12), (3, 8)),
    ]
    for axis, face_coordinate, side, origin, cell in cases:
        fp = build_footprint(axis=axis, face_coordinate=face_coordinate, side=side,
                             lateral_origin=origin, rows=20, columns=20,
                             declared={cell: VISIBLE})
        block = cell_to_block(fp, cell)
        assert block_to_cell(fp, block) == cell


def test_a_single_region_litematic_stores_shape_local_to_its_own_minimum():
    """A single-region litematic always writes Position (0, 0, 0) - a tested
    invariant (`export.validation`), not a bug: it stores a shape local to
    its own minimum corner rather than an absolute placement, because
    litematics are meant to be relocatable. That means naively comparing
    `cell_to_block`'s output against `load_litematic`'s read-back keys only
    works by coincidence when the exported volume's own minimum happens to
    already be (0, 0, 0) - exactly what every other round-trip test in this
    file exercises, since their VISIBLE region always includes local (0, 0).
    This test's supported cells deliberately exclude it, so the exported
    minimum corner is nonzero, and checks that the shift the format applies
    is exactly the supported cells' own minimum block coordinate.
    """
    from mc_photogrammetry.export.compare import load_litematic
    from mc_photogrammetry.export.litematic import LitematicExporter

    # Only cells (1,1) and (2,2) are visible - local (0,0) is not, so the
    # exported volume's own minimum corner is not the footprint's origin.
    declared = {(1, 1): VISIBLE, (2, 2): VISIBLE}
    footprint = build_footprint(axis=2, face_coordinate=1, side=1.0,
                               lateral_origin=(0, 0), rows=4, columns=4,
                               declared=declared)
    volume = footprint_to_volume(footprint)

    destination_dir = Path(__file__).resolve().parent
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        destination = Path(tmp) / "shifted.litematic"
        LitematicExporter().export(volume, destination, name="shift control",
                                   description="nonzero-minimum round trip")
        snapshot = load_litematic(destination)

    exported_blocks = {cell_to_block(footprint, cell) for cell in declared}
    minimum = tuple(min(b[axis] for b in exported_blocks) for axis in range(3))
    assert snapshot.position == (0, 0, 0), "the tested single-region invariant"
    # Read-back keys are local coordinates; shifting by the exported
    # volume's own minimum must recover exactly the original block set.
    shifted_back = {tuple(local[i] + minimum[i] for i in range(3))
                    for local in snapshot.blocks}
    assert shifted_back == exported_blocks
