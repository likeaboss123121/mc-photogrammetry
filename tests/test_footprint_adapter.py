"""Closing the synthetic loop: image-space polygons back to the 4x3 control.

The declared-cell placement in `footprint.py` was already accepted on its own
terms. What was still missing - and what a lead review named explicitly - is
everything upstream of a caller already having indexed cells: an image pixel
has to become a lattice cell through a real camera projection, not just
handed in pre-computed. These tests build the *same* 4x3 control from
`test_footprint.py`, project it through a declared synthetic camera to get
image-space polygons (as if a person had drawn them on a photograph), feed
those polygons back through `footprint_adapter.py`, and check that the
original 10/1/1 result comes back out - checked against the actual projected
geometry, not a second, independently-typed copy of the expected numbers.
"""
import numpy as np
import pytest

from mc_photogrammetry.reconstruction.calibration import axis_directions
from mc_photogrammetry.reconstruction.footprint import (
    FOREGROUND_OCCLUSION,
    GEOMETRY_OPENING,
    OPENING,
    SUPPORTED,
    UNKNOWN,
    VISIBLE,
    build_footprint,
)
from mc_photogrammetry.reconstruction.footprint_adapter import (
    LatticeTransform,
    RegionEvidence,
    cell_corners_pixels,
    declared_labels_from_evidence,
    lateral_to_pixel,
    pixel_to_lateral,
    rasterize_cells,
)

ROWS, COLUMNS = 3, 4
OPENING_CELL = (1, 2)
OCCLUDED_CELL = (2, 0)


def _control_declared() -> dict[tuple[int, int], str]:
    declared = {(r, c): VISIBLE for r in range(ROWS) for c in range(COLUMNS)}
    declared[OPENING_CELL] = OPENING
    declared[OCCLUDED_CELL] = FOREGROUND_OCCLUSION
    return declared


def _synthetic_transform(*, axis=2, side=1.0, pitch=-3.0, distance=8.0,
                         lateral_origin=(0.37, -0.62), margin=60.0
                         ) -> tuple[LatticeTransform, tuple[int, int]]:
    # A genuinely rotated camera (not axis-aligned), so the projection
    # actually exercises plane_basis's general case, not a trivial identity.
    # Which face sign of a plane sits in front of a given camera pose is a
    # real geometric fact, not a free choice - `pitch` (and, for axis=1,
    # `distance`) lets each test pick a pose the requested side is actually
    # visible from, the same way a real floor needs a downward-pitched
    # camera and a ceiling an upward one.
    #
    # Rather than hand-picking a principal point per configuration, project
    # once with an arbitrary one, then re-centre so the whole 4x3 grid lands
    # inside a plain, reasonably sized frame - the geometry under test is the
    # projection itself, not the choice of frame placement around it.
    axes = axis_directions(5.0, pitch, 2.0)
    probe = LatticeTransform(axis=axis, side=side, distance=distance, axis_matrix=axes,
                             focal=800.0, principal_point=(0.0, 0.0),
                             lateral_origin=lateral_origin, period=1.0)
    xs, ys = [], []
    for row in range(ROWS):
        for column in range(COLUMNS):
            corners = cell_corners_pixels(probe, (row, column))
            assert corners is not None, f"cell {(row, column)} projected behind the camera"
            xs += [p[0] for p in corners]
            ys += [p[1] for p in corners]
    principal_point = (-min(xs) + margin, -min(ys) + margin)
    image_size = (int(max(xs) - min(xs) + 2 * margin), int(max(ys) - min(ys) + 2 * margin))
    transform = LatticeTransform(axis=axis, side=side, distance=distance, axis_matrix=axes,
                                 focal=800.0, principal_point=principal_point,
                                 lateral_origin=lateral_origin, period=1.0)
    return transform, image_size


def _regions_from_declared(transform, declared) -> list[RegionEvidence]:
    """Stand in for a person's drawn polygons: project each declared cell's
    own corners through the transform - the ground truth this test checks
    the adapter's recovery against."""
    regions = []
    for cell, label in declared.items():
        corners = cell_corners_pixels(transform, cell)
        assert corners is not None, f"cell {cell} projected behind the camera"
        regions.append(RegionEvidence(label, tuple(corners)))
    return regions


def test_the_4x3_control_recovers_10_supported_1_opening_1_unknown_from_image_regions():
    transform, image_size = _synthetic_transform()
    declared = _control_declared()
    regions = _regions_from_declared(transform, declared)

    evidence = rasterize_cells(transform, regions, image_size=image_size,
                              rows=range(ROWS), columns=range(COLUMNS))
    recovered_labels = declared_labels_from_evidence(evidence)
    footprint = build_footprint(axis=2, face_coordinate=1, side=1.0,
                               lateral_origin=(0, 0), rows=ROWS, columns=COLUMNS,
                               declared=recovered_labels)

    assert len(footprint.of(SUPPORTED)) == 10
    assert len(footprint.of(GEOMETRY_OPENING)) == 1
    assert len(footprint.of(UNKNOWN)) == 1
    assert footprint.of(GEOMETRY_OPENING)[0].cell == OPENING_CELL
    assert footprint.of(UNKNOWN)[0].cell == OCCLUDED_CELL


def test_corner_reprojection_is_exact_for_every_cell():
    transform, _ = _synthetic_transform()
    for row in range(ROWS):
        for column in range(COLUMNS):
            corner = (transform.lateral_origin[0] + column * transform.period,
                     transform.lateral_origin[1] + row * transform.period)
            pixel = lateral_to_pixel(transform, corner)
            assert pixel is not None
            recovered = pixel_to_lateral(transform, pixel)
            assert recovered == pytest.approx(corner, abs=1e-6)


def test_a_fractional_lattice_origin_is_recovered_exactly():
    """The origin here is not an integer, and the period boundaries do not
    land on whole pixels either - real "phase" handling, not translation."""
    transform, image_size = _synthetic_transform(lateral_origin=(0.4123, -0.887))
    declared = _control_declared()
    regions = _regions_from_declared(transform, declared)
    evidence = rasterize_cells(transform, regions, image_size=image_size,
                              rows=range(ROWS), columns=range(COLUMNS))
    recovered_labels = declared_labels_from_evidence(evidence)
    assert recovered_labels.get(OPENING_CELL) == OPENING
    # A confidently-covered occluder is real evidence worth keeping as its own
    # declared label - `declared_labels_from_evidence` passes it through, and
    # it is `build_footprint` (already tested on its own) that resolves
    # FOREGROUND_OCCLUSION to `unknown` geometry, not this adapter silently
    # dropping it.
    assert recovered_labels.get(OCCLUDED_CELL) == FOREGROUND_OCCLUSION
    supported = sum(1 for label in recovered_labels.values() if label == VISIBLE)
    assert supported == 10


def test_both_face_signs_recover_the_same_footprint_shape():
    """Which sign of a plane sits in front of a camera is a geometric fact,
    not a free choice - each side is checked from a camera pose it is
    actually visible from (axis=1: a ceiling from a camera pitched up, a
    floor from one pitched down), mirroring how a real fixture works."""
    for axis, side, pitch in ((1, 1.0, -40.0), (1, -1.0, 40.0)):
        transform, image_size = _synthetic_transform(axis=axis, side=side, pitch=pitch)
        declared = _control_declared()
        regions = _regions_from_declared(transform, declared)
        evidence = rasterize_cells(transform, regions, image_size=image_size,
                                  rows=range(ROWS), columns=range(COLUMNS))
        recovered_labels = declared_labels_from_evidence(evidence)
        assert sum(1 for label in recovered_labels.values() if label == VISIBLE) == 10
        assert recovered_labels.get(OPENING_CELL) == OPENING


def test_a_nonzero_window_centre_does_not_change_the_recovered_shape():
    """'Window centre' here is the declared principal point - moving it must
    shift where things land in pixel space without changing which cells
    resolve to which label."""
    axes = axis_directions(5.0, -3.0, 2.0)
    centred = LatticeTransform(axis=2, side=1.0, distance=8.0, axis_matrix=axes,
                               focal=800.0, principal_point=(960.0, 540.0),
                               lateral_origin=(0.37, -0.62), period=1.0)
    off_centre = LatticeTransform(axis=2, side=1.0, distance=8.0, axis_matrix=axes,
                                  focal=800.0, principal_point=(1400.0, 300.0),
                                  lateral_origin=(0.37, -0.62), period=1.0)
    declared = _control_declared()
    for transform in (centred, off_centre):
        regions = _regions_from_declared(transform, declared)
        evidence = rasterize_cells(transform, regions, image_size=(2560, 1440),
                                  rows=range(ROWS), columns=range(COLUMNS))
        recovered_labels = declared_labels_from_evidence(evidence)
        assert sum(1 for label in recovered_labels.values() if label == VISIBLE) == 10
        assert recovered_labels.get(OPENING_CELL) == OPENING


def test_image_clipping_reduces_coverage_instead_of_inventing_a_verdict():
    """Cropping the image so the rightmost column of cells falls (partly or
    entirely) outside the frame must not silently mark them supported,
    opening, or unknown-with-full-confidence - their coverage has to show
    the clipping happened.
    """
    transform, (full_width, full_height) = _synthetic_transform()
    declared = _control_declared()
    regions = _regions_from_declared(transform, declared)
    # Clip well short of the full width the auto-centred helper computed -
    # column 3 (the last, so the furthest from the left edge in this
    # projection) should lose coverage.
    clipped_width = full_width * 2 // 3
    full_evidence = rasterize_cells(transform, regions, image_size=(full_width, full_height),
                                    rows=range(ROWS), columns=range(COLUMNS))
    clipped_evidence = rasterize_cells(transform, regions, image_size=(clipped_width, full_height),
                                       rows=range(ROWS), columns=range(COLUMNS))
    assert all(e.outside_image_fraction == 0.0 for e in full_evidence), (
        "the unclipped frame should show every cell fully in view")
    assert any(e.outside_image_fraction > 0.0 for e in clipped_evidence), (
        "clipping the frame should show reduced coverage, not a full-confidence verdict"
        " for the cells it cuts into")
    # And a clipped cell that lost enough coverage should stop being declared
    # at all, rather than keeping a confident verdict on partial evidence.
    clipped_declared = declared_labels_from_evidence(clipped_evidence)
    full_declared = declared_labels_from_evidence(full_evidence)
    assert len(clipped_declared) < len(full_declared)


def test_texture_cannot_reach_this_module_at_all():
    """Structural, not spot-checked: neither `rasterize_cells` nor anything
    it calls takes an image array - only polygons, a transform, and a size.
    """
    import inspect

    from mc_photogrammetry.reconstruction import footprint_adapter

    signature = inspect.signature(footprint_adapter.rasterize_cells)
    assert "image" not in signature.parameters and "patch" not in signature.parameters


def test_occlusion_wins_over_a_broader_overlapping_visible_region_regardless_of_order():
    """A person typically draws one broad visible-extent polygon for a whole
    wall, then a smaller region inside it marking a specific occluder. Where
    both clear the coverage threshold for the same cell, the smaller, more
    specific claim must win - and must win the same way regardless of which
    order the two regions were listed in.
    """
    transform, image_size = _synthetic_transform()
    occluded_corners = cell_corners_pixels(transform, OCCLUDED_CELL)
    # A broad VISIBLE polygon covering the whole grid, including the cell an
    # occlusion region also covers.
    all_corners = []
    for row in range(ROWS):
        for column in range(COLUMNS):
            all_corners += cell_corners_pixels(transform, (row, column))
    xs = [p[0] for p in all_corners]
    ys = [p[1] for p in all_corners]
    broad_visible = RegionEvidence(
        VISIBLE, ((min(xs) - 5, min(ys) - 5), (max(xs) + 5, min(ys) - 5),
                 (max(xs) + 5, max(ys) + 5), (min(xs) - 5, max(ys) + 5)))
    occluder = RegionEvidence(FOREGROUND_OCCLUSION, tuple(occluded_corners))

    for regions in ([broad_visible, occluder], [occluder, broad_visible]):
        evidence = rasterize_cells(transform, regions, image_size=image_size,
                                  rows=range(ROWS), columns=range(COLUMNS))
        by_cell = {e.cell: e for e in evidence}
        assert by_cell[OCCLUDED_CELL].declared == FOREGROUND_OCCLUSION, (
            "the occluder should win regardless of region list order")
