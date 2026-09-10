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
    VISIBLE_COVERAGE_THRESHOLD,
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


def test_g1_half_occluded_cell_under_a_broad_visible_polygon_stays_undeclared():
    """A lead review's exact reproduction: a broad VISIBLE polygon covering a
    whole cell, with a FOREGROUND_OCCLUSION polygon over its left half. The
    old code reported visible=1.0 (declared visible) because each label's
    coverage was measured independently, never subtracting the occluded
    area. Real, unobscured coverage is about 50%, below the 60% threshold -
    the cell must stay undeclared, not confidently visible.
    """
    axes = axis_directions(0.0, 0.0, 0.0)
    transform = LatticeTransform(axis=2, side=1.0, distance=10.0, axis_matrix=axes,
                                 focal=1000.0, principal_point=(200.0, 200.0),
                                 lateral_origin=(0.0, 0.0), period=1.0)
    corners = cell_corners_pixels(transform, (0, 0))
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    visible = RegionEvidence(VISIBLE, ((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
    occluder = RegionEvidence(FOREGROUND_OCCLUSION,
                              ((x0, y0), ((x0 + x1) / 2, y0), ((x0 + x1) / 2, y1), (x0, y1)))
    evidence = rasterize_cells(transform, [visible, occluder], image_size=(400, 400),
                               rows=range(1), columns=range(1))[0]
    assert evidence.coverage_by_label[VISIBLE] == pytest.approx(0.5, abs=0.02)
    assert evidence.declared is None


def test_g1_a_cell_mostly_outside_the_image_cannot_be_declared_from_its_sliver():
    """A lead review's other exact reproduction: a cell 90% outside the
    photograph, with the small in-frame sliver entirely covered by VISIBLE.
    The old code measured coverage against the in-frame crop alone and
    reported visible=1.0, declared visible. Coverage-of-full-cell must be
    the denominator a decision is made on - about 10% here, not enough - while
    coverage-of-in-frame is still reported separately and can legitimately
    read 1.0.
    """
    axes = axis_directions(0.0, 0.0, 0.0)
    transform = LatticeTransform(axis=2, side=1.0, distance=10.0, axis_matrix=axes,
                                 focal=1000.0, principal_point=(100.0, 200.0),
                                 lateral_origin=(0.0, 0.0), period=1.0)
    corners = cell_corners_pixels(transform, (0, 0))
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    visible = RegionEvidence(VISIBLE, ((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
    evidence = rasterize_cells(transform, [visible], image_size=(10, 400),
                               rows=range(1), columns=range(1))[0]
    assert evidence.coverage_by_label[VISIBLE] == pytest.approx(0.10, abs=0.02)
    # `coverage_by_label` is now a certified lower bound (F4), so even a
    # fully-covered in-frame sliver reads a hair under 1.0 here - the
    # boundary margin around the sliver's own edges - rather than exactly.
    assert evidence.coverage_of_in_frame[VISIBLE] == pytest.approx(1.0, abs=0.02)
    assert evidence.declared is None
    assert evidence.outside_image_fraction == pytest.approx(0.90, abs=0.02)


def test_g1_an_opening_and_an_occlusion_polygon_conflicting_declares_neither():
    """Two specific claims about the same location disagreeing is not
    resolved by priority order - it is a genuine conflict, excluded from
    both labels' coverage rather than one silently winning.
    """
    axes = axis_directions(0.0, 0.0, 0.0)
    transform = LatticeTransform(axis=2, side=1.0, distance=10.0, axis_matrix=axes,
                                 focal=1000.0, principal_point=(200.0, 200.0),
                                 lateral_origin=(0.0, 0.0), period=1.0)
    corners = cell_corners_pixels(transform, (0, 0))
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    whole_cell = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    opening = RegionEvidence(OPENING, whole_cell)
    occlusion = RegionEvidence(FOREGROUND_OCCLUSION, whole_cell)
    evidence = rasterize_cells(transform, [opening, occlusion], image_size=(400, 400),
                               rows=range(1), columns=range(1))[0]
    assert evidence.coverage_by_label[OPENING] == pytest.approx(0.0, abs=1e-6)
    assert evidence.coverage_by_label[FOREGROUND_OCCLUSION] == pytest.approx(0.0, abs=1e-6)
    # A small cell's continuous (shoelace) area and its integer rasterization
    # do not match to the pixel - a real, expected discretization effect, not
    # the property under test here (that the shared area counts as conflict,
    # not as either label).
    assert evidence.conflict_fraction == pytest.approx(1.0, abs=0.05)
    assert evidence.declared is None


def test_g1_extreme_grazing_does_not_allocate_a_raster_sized_to_the_quad():
    """A near-grazing cell's own projected quadrilateral - not a hand-drawn
    region, the *cell itself* - can span well over a million pixels even
    though the photograph is a couple of thousand wide, if the cell sits
    close enough to the plane's vanishing direction. Column 90 of this
    rotation does exactly that (found by walking columns until the span blew
    up): about 1.5 million pixels wide, over a thousand times the width of
    any real image this project renders. `_polygon_area` (shoelace) must
    size the cell without ever allocating an array as large as its own
    extent - checked by requiring the call to complete quickly, not by
    inspecting memory directly.
    """
    import time

    axes = axis_directions(5.0, -3.0, 2.0)
    transform = LatticeTransform(axis=2, side=1.0, distance=8.0, axis_matrix=axes,
                                 focal=800.0, principal_point=(960.0, 540.0),
                                 lateral_origin=(0.0, 0.0), period=1.0)
    corners = cell_corners_pixels(transform, (0, 90))
    assert corners is not None
    xs = [p[0] for p in corners]
    assert max(xs) - min(xs) > 1_000_000, "test setup no longer produces an extreme span"

    started = time.perf_counter()
    evidence = rasterize_cells(transform, [], image_size=(1920, 1080),
                               rows=range(1), columns=range(90, 91))
    elapsed = time.perf_counter() - started
    assert elapsed < 2.0, f"took {elapsed:.2f}s - looks like it rasterized the full quad"
    # A cell this size relative to the frame has almost none of its own area
    # actually inside the photograph.
    assert evidence[0].outside_image_fraction > 0.99


def test_s3_pixel_aligned_unit_cell_matches_its_continuous_area_closely():
    """The original reproduction of this bug: identity axes, a cell
    projecting to the pixel-aligned unit square (2,2)-(3,3) - true
    continuous area 1 - fully covered by a matching VISIBLE polygon. The
    original rasterizer fed continuous corner coordinates straight to
    `cv2.fillPoly`, which treats an integer vertex as a pixel's own centre
    and includes boundary pixels, filling pixel-centres 2 through 3
    inclusive in both axes - a 2x2, four-pixel block - so `coverage_by_label`
    came back 4.0 for a cell whose true area is 1. A follow-up review found
    that fix (a half-pixel coordinate shift) only passed this one
    pixel-grid-aligned case, not the general one (see the supersampling
    controls below) - so this only checks the result is now close, not exact.
    """
    axes = np.eye(3)
    transform = LatticeTransform(axis=2, side=1.0, distance=1.0, axis_matrix=axes,
                                 focal=1.0, principal_point=(2.0, 3.0),
                                 lateral_origin=(0.0, 0.0), period=1.0)
    regions = [RegionEvidence(VISIBLE, ((2, 2), (3, 2), (3, 3), (2, 3)))]
    evidence = rasterize_cells(transform, regions, image_size=(6, 6),
                               rows=range(1), columns=range(1))[0]
    assert evidence.coverage_by_label[VISIBLE] == pytest.approx(1.0, abs=0.1)
    assert evidence.declared == VISIBLE


def test_s3_a_subpixel_cell_never_wins_a_declared_verdict():
    """A cell whose true projected area is a small fraction of a single
    pixel cannot be measured by counting whole raster pixels at all - even
    one rounded pixel is already several times the cell's own area, so no
    ratio computed against it is trustworthy at any threshold. Such a cell
    must retain uncertainty (`declared is None`) regardless of how
    confidently a raw raster ratio would otherwise read - it is genuinely
    below the resolution rasterization can measure, not evidence either way.
    """
    axes = axis_directions(0.0, 0.0, 0.0)
    transform = LatticeTransform(axis=2, side=1.0, distance=10.0, axis_matrix=axes,
                                 focal=1000.0, principal_point=(200.0, 200.0),
                                 lateral_origin=(0.0, 0.0), period=0.005)
    corners = cell_corners_pixels(transform, (0, 0))
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    assert (x1 - x0) * (y1 - y0) < 1.0, "test setup no longer produces a sub-pixel cell"
    visible = RegionEvidence(VISIBLE, ((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
    evidence = rasterize_cells(transform, [visible], image_size=(400, 400),
                               rows=range(1), columns=range(1))[0]
    assert evidence.declared is None


def test_s3_a_large_multi_pixel_cell_area_measure_stays_exact():
    """The fix must hold at ordinary cell sizes too, not just the toy
    single-pixel reproduction - a 50-ish pixel cell fully covered by a
    matching region should read almost exactly 1.0, the same as the
    original half-occlusion/grazing controls already checked at their own
    scale (kept above, unmodified by this fix)."""
    axes = axis_directions(0.0, 0.0, 0.0)
    transform = LatticeTransform(axis=2, side=1.0, distance=10.0, axis_matrix=axes,
                                 focal=1000.0, principal_point=(200.0, 200.0),
                                 lateral_origin=(0.0, 0.0), period=1.0)
    corners = cell_corners_pixels(transform, (0, 0))
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    assert (x1 - x0) > 20 and (y1 - y0) > 20, "test setup no longer produces a large cell"
    visible = RegionEvidence(VISIBLE, ((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
    evidence = rasterize_cells(transform, [visible], image_size=(400, 400),
                               rows=range(1), columns=range(1))[0]
    assert evidence.coverage_by_label[VISIBLE] == pytest.approx(1.0, abs=0.03)


def test_s3_mixed_labels_within_one_cell_never_sum_past_the_cells_own_area():
    """VISIBLE and OPENING each cover a disjoint half of one cell. Since
    `_label_masks` already makes every label mutually exclusive, their
    `coverage_by_label` fractions of the *same* cell must sum to no more
    than 1 (plus whatever fraction sits outside the image) - the property
    that motivated this fix in the first place, checked directly rather
    than only on a single full-coverage case.
    """
    axes = axis_directions(0.0, 0.0, 0.0)
    transform = LatticeTransform(axis=2, side=1.0, distance=10.0, axis_matrix=axes,
                                 focal=1000.0, principal_point=(200.0, 200.0),
                                 lateral_origin=(0.0, 0.0), period=1.0)
    corners = cell_corners_pixels(transform, (0, 0))
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    midpoint = (x0 + x1) / 2
    visible = RegionEvidence(VISIBLE, ((x0, y0), (midpoint, y0), (midpoint, y1), (x0, y1)))
    opening = RegionEvidence(OPENING, ((midpoint, y0), (x1, y0), (x1, y1), (midpoint, y1)))
    evidence = rasterize_cells(transform, [visible, opening], image_size=(400, 400),
                               rows=range(1), columns=range(1))[0]
    total = (evidence.coverage_by_label[VISIBLE] + evidence.coverage_by_label[OPENING]
            + evidence.coverage_by_label[FOREGROUND_OCCLUSION] + evidence.conflict_fraction)
    assert total <= 1.0 + 0.05
    assert evidence.coverage_by_label[VISIBLE] == pytest.approx(0.5, abs=0.05)
    assert evidence.coverage_by_label[OPENING] == pytest.approx(0.5, abs=0.05)


def test_s3_a_clipped_cell_keeps_a_bounded_in_frame_coverage():
    """A cell half outside the image (clipped, not grazing) must still
    report a bounded, sensible `coverage_of_in_frame` for its visible
    sliver even under the corrected rasterization convention - the earlier
    G1 clipping control (`test_g1_a_cell_mostly_outside_the_image...`)
    covers the 90%-outside case already; this checks a milder 50% clip."""
    axes = axis_directions(0.0, 0.0, 0.0)
    transform = LatticeTransform(axis=2, side=1.0, distance=10.0, axis_matrix=axes,
                                 focal=1000.0, principal_point=(50.0, 200.0),
                                 lateral_origin=(0.0, 0.0), period=1.0)
    corners = cell_corners_pixels(transform, (0, 0))
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    assert x0 < 0 < x1, "test setup no longer straddles the image's left edge"
    visible = RegionEvidence(VISIBLE, ((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
    # Only x in [0, x1) is in frame - half the cell's own 100px width.
    evidence = rasterize_cells(transform, [visible], image_size=(int(x1), 400),
                               rows=range(1), columns=range(1))[0]
    assert evidence.coverage_by_label[VISIBLE] == pytest.approx(0.5, abs=0.05)
    assert evidence.coverage_of_in_frame[VISIBLE] == pytest.approx(1.0, abs=0.05)


def test_s4_reproject_exported_volume_matches_expected_corners_across_axes_and_signs(tmp_path):
    """S4: `reproject_exported_volume` must independently reproject blocks
    actually read out of a real litematic file, across all three world
    axes and both face signs, for an asymmetric (non-square-symmetric)
    supported footprint - the shape a row/column mix-up (G2's actual bug)
    would misplace. Reprojected corners are checked against
    `cell_corners_pixels` called directly on the *cell identity* the file
    should recover, which is a different, independent formula path from
    the export/import round trip itself.
    """
    from mc_photogrammetry.export.litematic import LitematicExporter
    from mc_photogrammetry.reconstruction.footprint import (
        SUPPORTED, build_footprint, footprint_to_volume,
    )
    from mc_photogrammetry.reconstruction.footprint_adapter import reproject_exported_volume

    rows, columns = 3, 4
    # Asymmetric: not a square, and not symmetric under row<->column swap -
    # exactly the shape that could not distinguish a transposed axis
    # convention from a correct one.
    supported_cells = {(0, 0), (0, 3), (2, 1)}

    # Which face of a plane sits in front of a given camera pose is a
    # geometric fact, not a free choice (as `_synthetic_transform`'s own
    # docstring notes for axis=1's pitch sign) - axis 0/2 need the camera
    # turned to the opposite horizontal side instead, found empirically the
    # same way: the yaw that keeps every cell of this grid in front of the
    # camera for that axis/side.
    poses = {
        (0, 1.0): (270.0, -3.0), (0, -1.0): (5.0, -3.0),
        (1, 1.0): (5.0, -40.0), (1, -1.0): (5.0, 40.0),
        (2, 1.0): (5.0, -3.0), (2, -1.0): (185.0, -3.0),
    }

    for axis, side in ((0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0), (2, 1.0), (2, -1.0)):
        yaw, pitch = poses[(axis, side)]
        axes = axis_directions(yaw, pitch, 2.0)
        probe = LatticeTransform(axis=axis, side=side, distance=8.0, axis_matrix=axes,
                                 focal=800.0, principal_point=(0.0, 0.0),
                                 lateral_origin=(0.0, 0.0), period=1.0)
        xs, ys = [], []
        for row in range(rows):
            for column in range(columns):
                corners = cell_corners_pixels(probe, (row, column))
                assert corners is not None, (axis, side, row, column)
                xs += [p[0] for p in corners]
                ys += [p[1] for p in corners]
        margin = 60.0
        principal_point = (-min(xs) + margin, -min(ys) + margin)
        image_size = (int(max(xs) - min(xs) + 2 * margin), int(max(ys) - min(ys) + 2 * margin))
        transform = LatticeTransform(axis=axis, side=side, distance=8.0, axis_matrix=axes,
                                     focal=800.0, principal_point=principal_point,
                                     lateral_origin=(0.0, 0.0), period=1.0)

        declared = {cell: VISIBLE for cell in supported_cells}
        footprint = build_footprint(axis=axis, face_coordinate=1, side=side,
                                    lateral_origin=(0, 0), rows=rows, columns=columns,
                                    declared=declared)
        assert {c.cell for c in footprint.of(SUPPORTED)} == supported_cells

        volume = footprint_to_volume(footprint)
        litematic_path = tmp_path / f"s4_axis{axis}_side{int(side)}.litematic"
        LitematicExporter().export(volume, litematic_path, name="s4 control",
                                   description="S4 asymmetric axis/sign control")

        reprojected, minimum = reproject_exported_volume(transform, footprint, litematic_path)

        assert set(reprojected) == supported_cells, (axis, side, set(reprojected))
        for cell in supported_cells:
            expected_corners = cell_corners_pixels(transform, cell)
            assert reprojected[cell] == expected_corners, (axis, side, cell)
        assert isinstance(minimum, tuple) and len(minimum) == 3


def test_e2_area_units_stay_consistent_off_the_pixel_grid():
    """A lead follow-up review's exact counter-examples to the half-pixel
    coordinate shift: that fix only reconciled raster and continuous area
    for a cell whose corners happen to land on integer pixel-grid
    coordinates. A non-integer principal point or non-unit focal moves the
    cell off-grid without changing its true area, and the shift alone still
    reported 4.0, 4.1667 and 1.5 for cells whose true area is 1 - not
    "declared" mislabeling, actual area measurement.
    """
    axes = np.eye(3)
    cases = [(1.0, (2.0, 3.0)), (1.0, (2.2, 3.2)), (1.2, (3.0, 4.0)), (2.0, (2.0, 3.0))]
    for focal, principal_point in cases:
        transform = LatticeTransform(axis=2, side=1.0, distance=1.0, axis_matrix=axes,
                                     focal=focal, principal_point=principal_point,
                                     lateral_origin=(0.0, 0.0), period=1.0)
        corners = cell_corners_pixels(transform, (0, 0))
        regions = [RegionEvidence(VISIBLE, tuple(corners))]
        evidence = rasterize_cells(transform, regions, image_size=(12, 12),
                                   rows=range(1), columns=range(1))[0]
        assert evidence.coverage_by_label[VISIBLE] == pytest.approx(1.0, abs=0.1), (focal, principal_point)
        assert evidence.declared == VISIBLE, (focal, principal_point)


def test_e2_a_translated_fractional_cell_still_measures_its_true_area():
    """A cell whose corners sit at fractional, non-grid-aligned coordinates
    throughout (not just offset from one integer origin) - translated by a
    non-integer amount in both axes - covered by a matching region."""
    axes = np.eye(3)
    transform = LatticeTransform(axis=2, side=1.0, distance=1.0, axis_matrix=axes,
                                 focal=1.0, principal_point=(7.63, 9.41),
                                 lateral_origin=(0.0, 0.0), period=1.0)
    corners = cell_corners_pixels(transform, (0, 0))
    regions = [RegionEvidence(VISIBLE, tuple(corners))]
    evidence = rasterize_cells(transform, regions, image_size=(20, 20),
                               rows=range(1), columns=range(1))[0]
    assert evidence.coverage_by_label[VISIBLE] == pytest.approx(1.0, abs=0.1)


def test_e3_a_depth_shifted_block_is_rejected_not_silently_reprojected(tmp_path):
    """A lead follow-up review's independent loader-injection control:
    `reproject_exported_volume` used to pass every read-back block through
    `block_to_cell`, which keeps only the two lateral coordinates and
    silently discards the normal-axis one - so a block sitting at the
    right lateral position but the wrong depth still produced a
    valid-looking cell. This injects exactly that: a real exported/loaded
    snapshot with one block's normal coordinate changed after loading, and
    checks it is rejected rather than reprojected as if nothing were wrong.
    """
    import dataclasses

    from mc_photogrammetry.export.litematic import LitematicExporter
    from mc_photogrammetry.export.compare import load_litematic
    from mc_photogrammetry.reconstruction.footprint import build_footprint, footprint_to_volume
    from mc_photogrammetry.reconstruction.footprint_adapter import reproject_exported_volume

    axes = np.eye(3)
    transform = LatticeTransform(axis=2, side=1.0, distance=10.0, axis_matrix=axes,
                                 focal=1000.0, principal_point=(200.0, 200.0),
                                 lateral_origin=(0.0, 0.0), period=1.0)
    footprint = build_footprint(axis=2, face_coordinate=1, side=1.0, lateral_origin=(0, 0),
                               rows=1, columns=2, declared={(0, 0): VISIBLE, (0, 1): VISIBLE})
    volume = footprint_to_volume(footprint)
    litematic_path = tmp_path / "e3_control.litematic"
    LitematicExporter().export(volume, litematic_path, name="e3 control",
                               description="E3 depth-shift injection control")

    genuine = load_litematic(litematic_path)
    assert len(genuine.blocks) == 2
    corrupted_local_blocks = list(genuine.blocks.items())
    (second_local, second_state) = corrupted_local_blocks[1]
    # Shift the second block's normal-axis coordinate (index 2 = Z, since
    # footprint.axis=2) away from the plane - exactly the review's
    # (1,0,0) -> (1,0,3) example, adapted to this footprint's own axis.
    depth_shifted_local = (second_local[0], second_local[1], second_local[2] + 3)
    corrupted_blocks = dict(corrupted_local_blocks)
    del corrupted_blocks[second_local]
    corrupted_blocks[depth_shifted_local] = second_state
    injected = dataclasses.replace(genuine, blocks=corrupted_blocks)

    with pytest.raises(ValueError, match="off the declared plane"):
        reproject_exported_volume(transform, footprint, litematic_path, _snapshot=injected)

    # The genuine, uninjected file must still verify cleanly - this is not
    # a false-positive rejection of ordinary read-back data.
    reprojected, _ = reproject_exported_volume(transform, footprint, litematic_path)
    assert len(reprojected) == 2


def test_e3_a_persisted_minimum_shift_is_sufficient_without_recomputing(tmp_path):
    """An immutable export transform (here, just its `minimum_shift`) must
    be enough to verify a reload on its own - a caller should be able to
    persist it once (as `export_transform.json` does) and pass it back in
    on a later call without that call needing to recompute it from the
    footprint's own pre-export `SUPPORTED` cells."""
    from mc_photogrammetry.export.litematic import LitematicExporter
    from mc_photogrammetry.reconstruction.footprint import build_footprint, footprint_to_volume
    from mc_photogrammetry.reconstruction.footprint_adapter import reproject_exported_volume

    axes = np.eye(3)
    transform = LatticeTransform(axis=2, side=1.0, distance=10.0, axis_matrix=axes,
                                 focal=1000.0, principal_point=(200.0, 200.0),
                                 lateral_origin=(0.0, 0.0), period=1.0)
    footprint = build_footprint(axis=2, face_coordinate=1, side=1.0, lateral_origin=(0, 0),
                               rows=2, columns=2,
                               declared={(0, 0): VISIBLE, (1, 1): VISIBLE})
    volume = footprint_to_volume(footprint)
    litematic_path = tmp_path / "e3_persisted.litematic"
    LitematicExporter().export(volume, litematic_path, name="e3 persisted control",
                               description="E3 persisted-minimum-shift control")

    _, minimum_shift = reproject_exported_volume(transform, footprint, litematic_path)

    reprojected_again, returned_shift = reproject_exported_volume(
        transform, footprint, litematic_path, minimum_shift=minimum_shift)
    assert returned_shift == minimum_shift
    assert len(reprojected_again) == 2


def _rectangle_cell(*, focal=1000.0, principal_point=(200.0, 200.0), distance=10.0, period=1.0):
    """A single (0, 0) cell under identity axes - the same well-resolved,
    comfortably in-frame setup `test_g1_half_occluded_cell...` above already
    uses (about 100px on a side at the defaults) - returning the transform
    and its projected extent as axis-sorted (x0, x1, y0, y1) bounds (never
    assuming a particular corner order or winding, matching how the
    existing G1/S3 controls above build their own rectangles). Callers must
    still use an image_size generous enough to hold the whole cell (the
    y-axis is flipped relative to lateral v, so a `principal_point` picked
    without accounting for that can leave half the cell projecting to
    negative pixel coordinates and clipped by the image edge - a real, but
    different, effect from the one these F4 controls test)."""
    axes = np.eye(3)
    transform = LatticeTransform(axis=2, side=1.0, distance=distance, axis_matrix=axes,
                                 focal=focal, principal_point=principal_point,
                                 lateral_origin=(0.0, 0.0), period=period)
    corners = cell_corners_pixels(transform, (0, 0))
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return transform, (min(xs), max(xs), min(ys), max(ys))


def test_f4_the_reviews_exact_55_percent_case_never_certifies_visible():
    """The lead follow-up review's own reproduction: identity-axis Z plane,
    side/distance/period=1, focal=1, principal point (2.2, 3.2), cell (0,0) -
    the projected cell is [2.2,3.2]x[2.2,3.2]. A VISIBLE rectangle starting
    at its lower-left corner and covering only 55% of its width. The old
    supersampled point estimate reported 0.6123046875 here and declared the
    cell visible at the 0.6 threshold - a real decision error, not a
    cosmetic one, since true coverage (55%) is genuinely below threshold.
    `coverage_by_label` is now a certified lower bound, so it must never
    read above the true 0.55 value (a bound that overshoots the truth is not
    a bound at all) and must never clear 0.6 here.
    """
    transform, (x0, x1, y0, y1) = _rectangle_cell(focal=1.0, principal_point=(2.2, 3.2))
    width = x1 - x0
    region = RegionEvidence(VISIBLE, ((x0, y0), (x0 + 0.55 * width, y0),
                                      (x0 + 0.55 * width, y1), (x0, y1)))
    evidence = rasterize_cells(transform, [region], image_size=(12, 12),
                               rows=range(1), columns=range(1))[0]
    assert evidence.coverage_by_label[VISIBLE] <= 0.55 + 1e-6
    assert evidence.coverage_by_label[VISIBLE] < VISIBLE_COVERAGE_THRESHOLD
    assert evidence.declared is None


@pytest.mark.parametrize("pct,expect_declared", [
    (55, False), (59, False), (60, None), (61, True), (100, True),
])
def test_f4_threshold_decisions_are_certified_not_estimated(pct, expect_declared):
    """At a well-resolved cell size (50px, not the toy 1px reproduction
    above), the certified lower bound should sit close enough to the true
    coverage fraction to resolve every case except the one truly ambiguous
    to an approximation: exactly at the threshold. 55%/59% must stay
    undeclared (a real miss); 61%/100% must declare VISIBLE (a real,
    guaranteed clear); exactly 60% - per the review's own "abstain if an
    approximation cannot certify it" - must also stay undeclared, since a
    conservative lower bound can never certify a coverage sitting exactly on
    its own threshold.
    """
    transform, (x0, x1, y0, y1) = _rectangle_cell()
    width = x1 - x0
    fraction = pct / 100.0
    region = RegionEvidence(VISIBLE, ((x0, y0), (x0 + fraction * width, y0),
                                      (x0 + fraction * width, y1), (x0, y1)))
    evidence = rasterize_cells(transform, [region], image_size=(400, 400),
                               rows=range(1), columns=range(1))[0]
    assert evidence.coverage_by_label[VISIBLE] == pytest.approx(fraction, abs=0.02)
    if expect_declared is None:
        assert evidence.declared is None, (
            "an approximation cannot certify a coverage sitting exactly on its own threshold")
    elif expect_declared:
        assert evidence.declared == VISIBLE
    else:
        assert evidence.declared is None


def test_f4_upper_bound_never_undershoots_the_lower_bound():
    """Basic soundness: whatever the true coverage, the certified upper
    bound must never read below the certified lower bound for the same
    label - if it did, the two would not bracket a real interval at all."""
    transform, (x0, x1, y0, y1) = _rectangle_cell()
    width = x1 - x0
    for fraction in (0.1, 0.3, 0.55, 0.6, 0.61, 0.9, 1.0):
        region = RegionEvidence(VISIBLE, ((x0, y0), (x0 + fraction * width, y0),
                                          (x0 + fraction * width, y1), (x0, y1)))
        evidence = rasterize_cells(transform, [region], image_size=(400, 400),
                                   rows=range(1), columns=range(1))[0]
        assert evidence.coverage_by_label[VISIBLE] <= evidence.coverage_upper_by_label[VISIBLE] + 1e-9, fraction
        assert evidence.coverage_upper_by_label[VISIBLE] >= fraction - 0.02, fraction


def test_f4_a_translated_fractional_cell_still_certifies_the_threshold_correctly():
    """The same 59%/61% reject/accept check, but at a non-integer principal
    point and a non-unit focal (both off the pixel grid throughout, not just
    translated from one integer origin) - the review's own concern that a
    fix must generalize, not just pass the one reported case."""
    transform, (x0, x1, y0, y1) = _rectangle_cell(focal=983.7, principal_point=(207.7, 213.1))
    width = x1 - x0
    for fraction, expect_visible in ((0.59, False), (0.61, True)):
        region = RegionEvidence(VISIBLE, ((x0, y0), (x0 + fraction * width, y0),
                                          (x0 + fraction * width, y1), (x0, y1)))
        evidence = rasterize_cells(transform, [region], image_size=(400, 400),
                                   rows=range(1), columns=range(1))[0]
        assert (evidence.declared == VISIBLE) == expect_visible, fraction


def test_f4_mixed_labels_lower_and_upper_bounds_both_stay_within_the_cells_area():
    """VISIBLE and OPENING each cover a disjoint half of one cell (the same
    split `test_s3_mixed_labels_within_one_cell_never_sum_past_the_cells_own_area`
    checks for the lower bound alone). The certified *upper* bounds may
    legitimately overlap a little at the shared boundary - both labels'
    boundary bands can cover the same ambiguous sub-pixels - but must still
    stay within [0, 1] and not blow up past a small margin past the cell's
    own area."""
    axes = axis_directions(0.0, 0.0, 0.0)
    transform = LatticeTransform(axis=2, side=1.0, distance=10.0, axis_matrix=axes,
                                 focal=1000.0, principal_point=(200.0, 200.0),
                                 lateral_origin=(0.0, 0.0), period=1.0)
    corners = cell_corners_pixels(transform, (0, 0))
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    midpoint = (x0 + x1) / 2
    visible = RegionEvidence(VISIBLE, ((x0, y0), (midpoint, y0), (midpoint, y1), (x0, y1)))
    opening = RegionEvidence(OPENING, ((midpoint, y0), (x1, y0), (x1, y1), (midpoint, y1)))
    evidence = rasterize_cells(transform, [visible, opening], image_size=(400, 400),
                               rows=range(1), columns=range(1))[0]
    for label in (VISIBLE, OPENING, FOREGROUND_OCCLUSION):
        assert 0.0 <= evidence.coverage_by_label[label] <= 1.0
        assert 0.0 <= evidence.coverage_upper_by_label[label] <= 1.0
        assert evidence.coverage_by_label[label] <= evidence.coverage_upper_by_label[label] + 1e-9
    assert evidence.coverage_by_label[VISIBLE] == pytest.approx(0.5, abs=0.05)
    assert evidence.coverage_by_label[OPENING] == pytest.approx(0.5, abs=0.05)


def test_f4_a_crop_larger_than_the_supersample_budget_is_tiled_not_allocated_whole():
    """A lead follow-up review named this gap directly: '"the current
    factor>=1 fallback does not cap a crop already larger than four million
    samples'". A cell whose own projected crop already exceeds
    `MAX_SUPERSAMPLED_CELLS` before any subdivision at all (here, close to
    the whole of a 3000x3000 frame - 9 million pixels) must still be
    measured - in bounded row-bands (`_tile_row_bounds`), never as one
    array sized to the crop - and complete quickly rather than attempting a
    single oversized allocation.
    """
    import time

    from mc_photogrammetry.reconstruction.footprint_adapter import MAX_SUPERSAMPLED_CELLS

    axes = np.eye(3)
    # A centred (not corner-anchored) lateral origin so the cell's own
    # 3000x3000 projected extent lands squarely inside the 3000x3000 frame
    # rather than being clipped by it - the property under test here is the
    # tiling budget, not image clipping (already covered elsewhere).
    transform = LatticeTransform(axis=2, side=1.0, distance=1.0, axis_matrix=axes,
                                 focal=3000.0, principal_point=(1500.0, 1500.0),
                                 lateral_origin=(-0.5, -0.5), period=1.0)
    corners = cell_corners_pixels(transform, (0, 0))
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    crop_pixels = (max(xs) - min(xs)) * (max(ys) - min(ys))
    assert crop_pixels > MAX_SUPERSAMPLED_CELLS, "test setup no longer exceeds the tiling threshold"

    region = RegionEvidence(VISIBLE, tuple(corners))
    started = time.perf_counter()
    evidence = rasterize_cells(transform, [region], image_size=(3000, 3000),
                               rows=range(1), columns=range(1))[0]
    elapsed = time.perf_counter() - started
    assert elapsed < 5.0, f"took {elapsed:.2f}s - looks like it allocated the whole crop"
    assert evidence.coverage_by_label[VISIBLE] == pytest.approx(1.0, abs=0.01)
    assert evidence.declared == VISIBLE


def test_f4_a_crop_whose_width_alone_exceeds_the_budget_is_left_unresolved():
    """The pathological edge `_tile_row_bounds` cannot itself absorb: a crop
    so wide that even a single-row band would exceed the budget. Rather than
    allocate past it, `rasterize_cells` must stop with an explicit
    unresolved cell (`declared is None`, no coverage claimed either way).
    Exercised by lowering the module's own budget rather than constructing
    an implausibly many-million-pixel-wide real image."""
    import mc_photogrammetry.reconstruction.footprint_adapter as footprint_adapter

    axes = np.eye(3)
    transform = LatticeTransform(axis=2, side=1.0, distance=1.0, axis_matrix=axes,
                                 focal=200.0, principal_point=(100.0, 100.0),
                                 lateral_origin=(0.0, 0.0), period=1.0)
    region = RegionEvidence(VISIBLE, tuple(cell_corners_pixels(transform, (0, 0))))

    original_budget = footprint_adapter.MAX_SUPERSAMPLED_CELLS
    footprint_adapter.MAX_SUPERSAMPLED_CELLS = 50
    try:
        evidence = rasterize_cells(transform, [region], image_size=(200, 200),
                                   rows=range(1), columns=range(1))[0]
    finally:
        footprint_adapter.MAX_SUPERSAMPLED_CELLS = original_budget
    assert evidence.declared is None
    assert evidence.coverage_by_label == {}
