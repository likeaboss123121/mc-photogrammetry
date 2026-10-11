"""Controls for the M4-C automatic boundary proposal.

Two kinds of control, per the lead review's S1 assignment:

- Direct controls on `propose_boundary_candidates` itself, using hand-built
  `Chain` objects - no image, no detector noise, isolating the pairing/
  quadrilateral geometry the review found broken (role assignment by image
  centre, mixed-axis "opposite" pairs, silently substituted (0,0) corners).
- End-to-end controls that draw real line segments on lines passing exactly
  through a real calibration's vanishing points, exercising
  `find_chains`+`propose_boundary_candidates` together.
"""
from __future__ import annotations

import numpy as np
import pytest
import time
from unittest.mock import patch
from pathlib import Path
import ast

cv2 = pytest.importorskip("cv2")

from mc_photogrammetry.reconstruction.boundary_candidates import (
    Chain, PlaneProposalResult, find_chains, propose_boundary_candidates,
    propose_image_planes,
)
from mc_photogrammetry.reconstruction.calibration import (
    CalibrationResult, vanishing_points,
)

WIDTH, HEIGHT = 640, 480
FOCAL, YAW, PITCH, ROLL = 500.0, 25.0, 0.0, 0.0


# ---------------------------------------------------------------------------
# Direct controls: hand-built Chains, no image, no calibration search.
# ---------------------------------------------------------------------------

def _chain(axis, point, direction, offset, span, strength=10.0, segment_count=3):
    """`span` here is the simple (low, high) envelope test callers already
    use; wrapped as the single-interval form `Chain.span` now expects."""
    return Chain(axis=axis, strength=strength, point=point, direction=direction,
                offset=offset, span=(span,), segment_count=segment_count)


def test_off_centre_rectangle_is_recovered_not_stretched_to_the_frame_edge():
    """The lead review's exact reproduction: a 100x100 frame, verticals at
    x=10 and x=30 (both left of centre), horizontals at y=20 and y=80. The
    old centre-split role assignment put both verticals in the "left"
    bucket and filled "right" with the frame edge, producing x=10..100 and
    x=30..100 instead of the true (10,20)-(30,80) rectangle."""
    left = _chain(axis=1, point=(10.0, 50.0), direction=(0.0, 1.0), offset=10.0,
                 span=(-200.0, 200.0))
    right = _chain(axis=1, point=(30.0, 50.0), direction=(0.0, 1.0), offset=30.0,
                   span=(-200.0, 200.0))
    top = _chain(axis=0, point=(20.0, 20.0), direction=(1.0, 0.0), offset=20.0,
                span=(-200.0, 200.0))
    bottom = _chain(axis=0, point=(20.0, 80.0), direction=(1.0, 0.0), offset=80.0,
                   span=(-200.0, 200.0))

    candidates = propose_boundary_candidates([left, right, top, bottom], 100, 100)

    assert candidates
    best = candidates[0]
    assert best.complete
    xs = sorted(p[0] for p in best.polygon)
    ys = sorted(p[1] for p in best.polygon)
    assert xs[0] == pytest.approx(10.0, abs=1e-6)
    assert xs[-1] == pytest.approx(30.0, abs=1e-6)
    assert ys[0] == pytest.approx(20.0, abs=1e-6)
    assert ys[-1] == pytest.approx(80.0, abs=1e-6)


@pytest.mark.parametrize("cx,cy", [(15, 15), (85, 15), (15, 85), (85, 85)])
def test_off_centre_rectangles_in_every_quadrant_are_recovered(cx, cy):
    """A 20x20 rectangle placed near each corner of a 100x100 frame, none
    of them straddling the centre on either axis - the exact condition the
    old role split by image centre could not represent correctly."""
    x0, x1 = cx - 10, cx + 10
    y0, y1 = cy - 10, cy + 10
    left = _chain(axis=1, point=(float(x0), 50.0), direction=(0.0, 1.0), offset=float(x0),
                 span=(-200.0, 200.0))
    right = _chain(axis=1, point=(float(x1), 50.0), direction=(0.0, 1.0), offset=float(x1),
                   span=(-200.0, 200.0))
    top = _chain(axis=0, point=(50.0, float(y0)), direction=(1.0, 0.0), offset=float(y0),
                span=(-200.0, 200.0))
    bottom = _chain(axis=0, point=(50.0, float(y1)), direction=(1.0, 0.0), offset=float(y1),
                   span=(-200.0, 200.0))

    candidates = propose_boundary_candidates([left, right, top, bottom], 100, 100)

    assert candidates
    best = candidates[0]
    xs = sorted(p[0] for p in best.polygon)
    ys = sorted(p[1] for p in best.polygon)
    assert xs[0] == pytest.approx(float(x0), abs=1e-6)
    assert xs[-1] == pytest.approx(float(x1), abs=1e-6)
    assert ys[0] == pytest.approx(float(y0), abs=1e-6)
    assert ys[-1] == pytest.approx(float(y1), abs=1e-6)


def test_two_lines_from_different_axes_never_pair_as_opposite_edges():
    """Two chains that look similarly oriented in the image but come from
    different calibrated axes do not describe opposite sides of one plane.
    With only one chain per axis, propose_boundary_candidates must combine
    them as adjacent edges (axis 0 x axis 1), never invent a same-axis
    opposite partner that does not exist."""
    only_axis0 = _chain(axis=0, point=(40.0, 40.0), direction=(1.0, 0.05), offset=0.0,
                        span=(-50.0, 50.0))
    only_axis2 = _chain(axis=2, point=(60.0, 60.0), direction=(1.0, -0.05), offset=0.0,
                        span=(-50.0, 50.0))

    candidates = propose_boundary_candidates([only_axis0, only_axis2], 100, 100)

    for candidate in candidates:
        assert set(candidate.axis_pair) == {0, 2}
        for side in candidate.edges.values():
            # each pencil here has exactly one real chain; the other slot
            # must be the explicit None/frame-edge fallback, never a
            # fabricated second chain borrowed from the other axis.
            assert side["low"] is None or side["high"] is None
        assert not candidate.complete


def test_absent_second_axis_yields_no_candidates_rather_than_a_guess():
    """Only one axis has any inlier chains at all (e.g. every detected
    segment happened to fit one lattice direction). No plane boundary can
    be proposed from a single pencil of parallel lines - correctly nothing,
    not a fabricated square."""
    a = _chain(axis=1, point=(10.0, 50.0), direction=(0.0, 1.0), offset=10.0, span=(-50.0, 50.0))
    b = _chain(axis=1, point=(30.0, 50.0), direction=(0.0, 1.0), offset=30.0, span=(-50.0, 50.0))

    assert propose_boundary_candidates([a, b], 100, 100) == []


def test_near_parallel_lines_across_axes_are_rejected_not_intersected_far_away():
    """Two chains from different axes that are nearly parallel to each
    other in image space intersect (if at all) far outside the frame - the
    old code clamped that corner to a fixed range and kept the candidate;
    this must reject it instead."""
    almost_horizontal_a = _chain(axis=0, point=(50.0, 50.0), direction=(1.0, 0.001), offset=0.0,
                                 span=(-1000.0, 1000.0))
    almost_horizontal_b = _chain(axis=2, point=(50.0, 51.0), direction=(1.0, 0.0011), offset=0.0,
                                 span=(-1000.0, 1000.0))
    other_a = _chain(axis=0, point=(50.0, 200.0), direction=(1.0, 0.001), offset=999.0,
                     span=(-1000.0, 1000.0))
    other_b = _chain(axis=2, point=(50.0, -100.0), direction=(1.0, 0.0011), offset=-999.0,
                     span=(-1000.0, 1000.0))

    candidates = propose_boundary_candidates(
        [almost_horizontal_a, almost_horizontal_b, other_a, other_b], 100, 100)

    diagonal = float(np.hypot(100, 100))
    for candidate in candidates:
        for x, y in candidate.polygon:
            assert abs(x) <= 3.0 * diagonal + 1.0
            assert abs(y) <= 3.0 * diagonal + 1.0


def test_a_corner_far_past_a_chains_observed_extent_is_rejected():
    """A chain seen only over a short stretch cannot honestly support a
    corner far beyond that stretch - the review's "corner connectivity"
    requirement. Shrinking one chain's real span from wide to narrow (with
    everything else unchanged) must turn an accepted candidate into a
    rejected one."""
    top = _chain(axis=0, point=(20.0, 20.0), direction=(1.0, 0.0), offset=20.0,
                span=(-200.0, 200.0))
    bottom = _chain(axis=0, point=(20.0, 80.0), direction=(1.0, 0.0), offset=80.0,
                   span=(-200.0, 200.0))
    wide_left = _chain(axis=1, point=(10.0, 50.0), direction=(0.0, 1.0), offset=10.0,
                       span=(-200.0, 200.0))
    right = _chain(axis=1, point=(30.0, 50.0), direction=(0.0, 1.0), offset=30.0,
                   span=(-200.0, 200.0))
    wide_candidates = propose_boundary_candidates([wide_left, right, top, bottom], 100, 100)
    assert any(c.complete for c in wide_candidates)

    narrow_left = _chain(axis=1, point=(10.0, 50.0), direction=(0.0, 1.0), offset=10.0,
                         span=(-2.0, 2.0), strength=10.0)
    narrow_candidates = propose_boundary_candidates([narrow_left, right, top, bottom], 100, 100)
    assert not any(c.complete for c in narrow_candidates)


# ---------------------------------------------------------------------------
# End-to-end controls: real detected segments through a real calibration.
# ---------------------------------------------------------------------------

def _line_point(vp, other_point):
    x0, y0, w0 = vp
    if abs(w0) > 1e-9:
        point = np.array([x0 / w0, y0 / w0])
        direction = np.array(other_point) - point
    else:
        direction = np.array([x0, y0])
    direction = direction / np.linalg.norm(direction)
    return direction


def _draw_rectangle(points, left_x=180, right_x=460, top_y=120, bottom_y=360,
                    width=WIDTH, height=HEIGHT):
    image = np.full((height, width, 3), 40, dtype=np.uint8)

    def draw_through(vp, anchor, length=2000.0):
        direction = _line_point(vp, anchor)
        anchor = np.array(anchor, dtype=float)
        a = anchor - direction * length
        b = anchor + direction * length
        cv2.line(image, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)),
                 (220, 220, 220), 2, cv2.LINE_AA)

    draw_through(points[1], (left_x, (top_y + bottom_y) / 2.0))
    draw_through(points[1], (right_x, (top_y + bottom_y) / 2.0))
    draw_through(points[0], (width / 2.0, top_y))
    draw_through(points[0], (width / 2.0, bottom_y))
    return image, (left_x, right_x, top_y, bottom_y)


def _analytic_intersection(vp_a, anchor_a, vp_b, anchor_b):
    a_direction = _line_point(vp_a, anchor_a)
    b_direction = _line_point(vp_b, anchor_b)
    matrix = np.array([[a_direction[0], -b_direction[0]], [a_direction[1], -b_direction[1]]])
    rhs = np.array([anchor_b[0] - anchor_a[0], anchor_b[1] - anchor_a[1]])
    t = np.linalg.solve(matrix, rhs)[0]
    return np.array(anchor_a) + t * a_direction


def test_find_chains_recovers_four_edges_from_pure_line_evidence():
    points = vanishing_points(FOCAL, YAW, PITCH, WIDTH, HEIGHT, ROLL)
    image, _ = _draw_rectangle(points)

    chains = find_chains(image, focal=FOCAL, yaw_degrees=YAW, pitch_degrees=PITCH,
                         roll_degrees=ROLL)

    assert len(chains) >= 4
    axis0 = [c for c in chains if c.axis == 0]
    axis1 = [c for c in chains if c.axis == 1]
    assert len(axis0) >= 2
    assert len(axis1) >= 2


@pytest.mark.parametrize("yaw,pitch,roll", [(25.0, 0.0, 0.0), (25.0, 0.0, 7.0)])
def test_propose_boundary_candidates_recovers_the_drawn_rectangle(yaw, pitch, roll):
    """Includes a rolled-camera variant: the geometry must still recover the
    true perspective-converging corners with a nonzero roll, not just the
    unrolled case."""
    points = vanishing_points(FOCAL, yaw, pitch, WIDTH, HEIGHT, roll)
    image, (left_x, right_x, top_y, bottom_y) = _draw_rectangle(points)

    chains = find_chains(image, focal=FOCAL, yaw_degrees=yaw, pitch_degrees=pitch,
                         roll_degrees=roll)
    candidates = propose_boundary_candidates(chains, WIDTH, HEIGHT)

    assert candidates
    best = candidates[0]
    assert best.strength == max(c.strength for c in candidates)

    left_anchor = (left_x, (top_y + bottom_y) / 2.0)
    right_anchor = (right_x, (top_y + bottom_y) / 2.0)
    top_anchor = (WIDTH / 2.0, top_y)
    bottom_anchor = (WIDTH / 2.0, bottom_y)
    expected = [
        _analytic_intersection(points[0], top_anchor, points[1], left_anchor),
        _analytic_intersection(points[0], top_anchor, points[1], right_anchor),
        _analytic_intersection(points[0], bottom_anchor, points[1], right_anchor),
        _analytic_intersection(points[0], bottom_anchor, points[1], left_anchor),
    ]
    expected_xs = sorted(p[0] for p in expected)
    expected_ys = sorted(p[1] for p in expected)
    actual_xs = sorted(p[0] for p in best.polygon)
    actual_ys = sorted(p[1] for p in best.polygon)
    assert actual_xs[0] == pytest.approx(expected_xs[0], abs=3.0)
    assert actual_xs[-1] == pytest.approx(expected_xs[-1], abs=3.0)
    assert actual_ys[0] == pytest.approx(expected_ys[0], abs=3.0)
    assert actual_ys[-1] == pytest.approx(expected_ys[-1], abs=3.0)


def test_find_chains_on_a_blank_image_returns_nothing():
    image = np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)
    chains = find_chains(image, focal=FOCAL, yaw_degrees=YAW, pitch_degrees=PITCH,
                         roll_degrees=ROLL)
    assert chains == []


def test_a_broken_edge_with_a_along_line_gap_still_forms_one_chain_with_two_intervals():
    """Two segment pieces on the exact same physical line, separated by a
    gap along the line itself (a furniture item interrupting a wall seam,
    say) - offset-based clustering must still merge them into one *chain*,
    since they share the same offset from the vanishing point regardless of
    how far apart they sit along the line. But the gap between the two
    pieces was never actually observed, so `span` must record it as two
    separate intervals, not one envelope spanning straight across it."""
    points = vanishing_points(FOCAL, YAW, PITCH, WIDTH, HEIGHT, ROLL)
    image = np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)
    direction = _line_point(points[1], (300.0, 200.0))
    anchor = np.array([300.0, 200.0])

    def segment(t0, t1):
        a = anchor + direction * t0
        b = anchor + direction * t1
        cv2.line(image, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)),
                 (220, 220, 220), 2, cv2.LINE_AA)

    segment(-150.0, -50.0)
    segment(50.0, 150.0)  # a 100px along-line gap between the two pieces

    chains = find_chains(image, focal=FOCAL, yaw_degrees=YAW, pitch_degrees=PITCH,
                         roll_degrees=ROLL)
    vertical_like = [c for c in chains if abs(c.direction[1]) >= abs(c.direction[0])]
    assert len(vertical_like) == 1
    chain = vertical_like[0]
    assert chain.segment_count >= 2
    assert len(chain.span) == 2
    lows = sorted(low for low, _ in chain.span)
    highs = sorted(high for _, high in chain.span)
    assert lows[0] == pytest.approx(-150.0, abs=5.0)
    assert highs[0] == pytest.approx(-50.0, abs=5.0)
    assert lows[1] == pytest.approx(50.0, abs=5.0)
    assert highs[1] == pytest.approx(150.0, abs=5.0)


def test_a_corner_in_the_gap_between_two_observed_intervals_is_not_supported():
    """The review's own example: bridging min/max across a genuine gap
    would let a corner that sits in the unobserved middle pass as
    supported just because two real detections happen to sit on either
    side of it. It must not."""
    gapped = Chain(axis=1, strength=10.0, point=(0.0, 0.0), direction=(0.0, 1.0),
                  offset=0.0, span=((-150.0, -50.0), (50.0, 150.0)), segment_count=2)

    from mc_photogrammetry.reconstruction.boundary_candidates import _corner_supported

    assert _corner_supported((0.0, -100.0), gapped, 5.0)  # inside the first interval
    assert _corner_supported((0.0, 100.0), gapped, 5.0)   # inside the second interval
    assert not _corner_supported((0.0, 0.0), gapped, 5.0)  # squarely in the unobserved gap


def test_two_separate_planes_both_produce_candidates():
    """Two distinct rectangles in one frame (two competing planes) should
    both be recoverable among the candidate set, not merged into one or
    have one erase the other."""
    points = vanishing_points(FOCAL, YAW, PITCH, WIDTH, HEIGHT, ROLL)
    image = np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)

    def draw_through(anchor, vp, length=2000.0):
        direction = _line_point(vp, anchor)
        anchor = np.array(anchor, dtype=float)
        a = anchor - direction * length
        b = anchor + direction * length
        cv2.line(image, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)),
                 (220, 220, 220), 2, cv2.LINE_AA)

    rectangles = [(60, 200, 60, 200), (420, 580, 300, 440)]
    for left_x, right_x, top_y, bottom_y in rectangles:
        draw_through((left_x, (top_y + bottom_y) / 2.0), points[1])
        draw_through((right_x, (top_y + bottom_y) / 2.0), points[1])
        draw_through((WIDTH / 2.0, top_y), points[0])
        draw_through((WIDTH / 2.0, bottom_y), points[0])

    chains = find_chains(image, focal=FOCAL, yaw_degrees=YAW, pitch_degrees=PITCH,
                         roll_degrees=ROLL, max_chains=16)
    candidates = propose_boundary_candidates(chains, WIDTH, HEIGHT, max_candidates=16)

    def expected_corners(rect):
        left_x, right_x, top_y, bottom_y = rect
        left_anchor = (left_x, (top_y + bottom_y) / 2.0)
        right_anchor = (right_x, (top_y + bottom_y) / 2.0)
        top_anchor, bottom_anchor = (WIDTH / 2.0, top_y), (WIDTH / 2.0, bottom_y)
        # These edges converge toward a finite vanishing point - as in the
        # single-rectangle control, the true corners are the analytic
        # line intersections, not a naive axis-aligned box.
        return [
            _analytic_intersection(points[0], top_anchor, points[1], left_anchor),
            _analytic_intersection(points[0], top_anchor, points[1], right_anchor),
            _analytic_intersection(points[0], bottom_anchor, points[1], right_anchor),
            _analytic_intersection(points[0], bottom_anchor, points[1], left_anchor),
        ]

    def matches(candidate, rect):
        expected = expected_corners(rect)
        expected_xs, expected_ys = sorted(p[0] for p in expected), sorted(p[1] for p in expected)
        xs, ys = sorted(p[0] for p in candidate.polygon), sorted(p[1] for p in candidate.polygon)
        return (abs(xs[0] - expected_xs[0]) < 5 and abs(xs[-1] - expected_xs[-1]) < 5
               and abs(ys[0] - expected_ys[0]) < 5 and abs(ys[-1] - expected_ys[-1]) < 5)

    for rectangle in rectangles:
        assert any(matches(candidate, rectangle) for candidate in candidates), rectangle


# ---------------------------------------------------------------------------
# M4-C3 image-only plane proposals. These controls use synthetic pixels and
# image-derived calibration records; the orthogonal-wall case supplies exact
# synthetic segments to isolate region/closure association from LSD variance.
# ---------------------------------------------------------------------------

def _synthetic_calibration(width=WIDTH, height=HEIGHT, yaw=0.0, pitch=0.0):
    return CalibrationResult(
        focal_pixels=500.0, yaw_degrees=yaw, pitch_degrees=pitch,
        width=width, height=height, inliers=40, segments=60,
        median_residual=0.5,
    )


def _render_front_wall(*, truncated=False, furniture=False, line_delta=40):
    image = np.full((HEIGHT, WIDTH, 3), 70, dtype=np.uint8)
    x0 = -60 if truncated else 50
    x1 = 540 if truncated else 590
    y0, y1 = 55, 425
    wall = (112, 112, 112)
    seam = (min(255, 112 + line_delta),) * 3
    cv2.rectangle(image, (x0, y0), (x1, y1), wall, -1)
    first_x = max(5, x0 + 5)
    for x in range(first_x, x1 - 5, 55):
        cv2.line(image, (x, y0 + 5), (x, y1 - 5), seam, 2)
    for y in range(y0 + 5, y1 - 5, 45):
        cv2.line(image, (max(5, x0 + 5), y), (x1 - 5, y), seam, 2)
    if furniture:
        # A distinct, foreground rectangle occludes the repeating wall grid.
        cv2.rectangle(image, (265, 210), (385, 420), (40, 40, 40), -1)
        cv2.rectangle(image, (265, 210), (385, 420), (220, 220, 220), 3)
    return image


def _image_proposals(image, *, calibration=None, usable=None, cap=32):
    height, width = image.shape[:2]
    if usable is None:
        usable = np.ones((height, width), dtype=bool)
    if calibration is None:
        calibration = _synthetic_calibration(width, height)
    return propose_image_planes(image, usable, calibration, max_candidates=cap)


def test_image_proposer_recovers_truncated_wall_from_usable_image_evidence():
    image = _render_front_wall(truncated=True)
    result = _image_proposals(image)

    assert isinstance(result, PlaneProposalResult)
    assert result.status == "ok"
    wall = [proposal for proposal in result.proposals if proposal.axis == 2]
    assert wall
    assert any(np.any(proposal.mask[:, :4]) for proposal in wall)
    region_scales = {source for proposal in wall for source in proposal.sources
                     if source.startswith("region_tile_")}
    assert "region_tile_48" in region_scales
    assert "region_tile_96" in region_scales
    assert all(not proposal.mask.flags.writeable for proposal in result.proposals)
    assert result.stats["chains"] <= 36


def test_image_proposer_keeps_wall_regions_separate_around_occluding_furniture():
    image = _render_front_wall(furniture=True)
    usable = np.ones((HEIGHT, WIDTH), dtype=bool)
    usable[:30, :80] = False
    result = _image_proposals(image, usable=usable)

    furniture = np.zeros((HEIGHT, WIDTH), dtype=bool)
    furniture[210:421, 265:386] = True
    region_proposals = [proposal for proposal in result.proposals
                        if proposal.axis == 2
                        and any(source.startswith("region_tile_")
                                for source in proposal.sources)]
    assert region_proposals
    assert any(np.count_nonzero(proposal.mask & furniture) / max(
        int(proposal.mask.sum()), 1) < 0.10 for proposal in region_proposals)
    assert all(not np.any(proposal.mask & ~usable) for proposal in result.proposals)


def test_image_proposer_detects_low_contrast_plane_from_line_closures():
    image = _render_front_wall(line_delta=18)
    result = _image_proposals(image)

    assert result.status == "ok"
    assert result.stats["detected_segments"] > 0
    assert any(proposal.axis == 2 and "axis_chain_closure" in proposal.sources
               for proposal in result.proposals)


def test_image_proposer_passes_image_derived_roll_to_multiscale_votes():
    from mc_photogrammetry.reconstruction import surfaces

    image = _render_front_wall(line_delta=40)
    calibration = _synthetic_calibration(yaw=25.0)
    calibration = CalibrationResult(
        focal_pixels=calibration.focal_pixels,
        yaw_degrees=calibration.yaw_degrees,
        pitch_degrees=calibration.pitch_degrees,
        width=calibration.width, height=calibration.height,
        inliers=calibration.inliers, segments=calibration.segments,
        median_residual=calibration.median_residual, roll_degrees=7.0,
    )

    with patch("mc_photogrammetry.reconstruction.surfaces.planar_regions",
               wraps=surfaces.planar_regions) as region_votes:
        _image_proposals(image, calibration=calibration)

    assert region_votes.call_count == 3
    assert {call.kwargs["roll_degrees"] for call in region_votes.call_args_list} == {7.0}


def test_image_proposer_separates_same_texture_orthogonal_walls_without_union():
    image = np.full((HEIGHT, WIDTH, 3), 80, dtype=np.uint8)
    cv2.rectangle(image, (45, 45), (600, 435), (130, 130, 130), -1)
    segments = []

    def draw(a, b):
        cv2.line(image, tuple(map(int, a)), tuple(map(int, b)), (220, 220, 220), 2)
        segments.append((*a, *b))

    # Same wall appearance on both sides of the corner. The left/front plane
    # contains axes 0+1 (normal 2); the right plane contains axes 1+2 (normal 0).
    for y in range(60, 431, 30):
        draw((50, y), (320, y))
    for x in range(60, 321, 40):
        draw((x, 50), (x, 430))
    for x in range(340, 601, 40):
        draw((x, 50), (x, 430))
    for y in range(60, 431, 30):
        draw((320, 240), (600, y))

    with patch(
        "mc_photogrammetry.reconstruction.boundary_candidates.detect_segments",
        return_value=np.asarray(segments, dtype=float),
    ):
        result = _image_proposals(image, calibration=_synthetic_calibration(yaw=0.0))

    left = [proposal for proposal in result.proposals if proposal.axis == 2]
    right = [proposal for proposal in result.proposals if proposal.axis == 0]
    assert left and right
    assert any(proposal.mask[:, :320].sum() > proposal.mask[:, 320:].sum() * 4
               for proposal in left)
    assert any(proposal.mask[:, 320:].sum() > proposal.mask[:, :320].sum() * 4
               for proposal in right)
    assert all(not (proposal.mask[:, :320].sum() > proposal.mask.sum() * 0.35
                    and proposal.mask[:, 320:].sum() > proposal.mask.sum() * 0.35)
               for proposal in result.proposals)


def test_image_proposer_order_is_deterministic_and_hard_cap_is_bounded():
    first = _render_front_wall(line_delta=80)
    second = _render_front_wall(truncated=True, line_delta=18)
    images = {"first": first, "second": second}

    def summaries(order):
        found = {}
        for name in order:
            result = _image_proposals(images[name], cap=80)
            found[name] = tuple(
                (proposal.axis, proposal.side, proposal.sources, proposal.polygon,
                 proposal.mask.tobytes()) for proposal in result.proposals
            )
            assert len(result.proposals) <= 32
            assert result.stats["chains"] <= 36
        return found

    started = time.perf_counter()
    forward = summaries(("first", "second"))
    reverse = summaries(("second", "first"))
    elapsed = time.perf_counter() - started

    assert forward == reverse
    assert elapsed < 10.0


def test_image_proposer_returns_typed_abstentions_for_missing_evidence():
    image = _render_front_wall()
    valid = _synthetic_calibration()
    unusable = np.zeros((HEIGHT, WIDTH), dtype=bool)

    missing = propose_image_planes(image, np.ones((HEIGHT, WIDTH), bool), None)
    insufficient = propose_image_planes(image, unusable, valid)
    invalid = propose_image_planes(image, np.ones((HEIGHT, WIDTH), bool), {})
    mismatched = propose_image_planes(image, np.ones((HEIGHT, WIDTH), bool),
                                      _synthetic_calibration(width=WIDTH - 1))
    weak_calibration = CalibrationResult(500, 0, 0, WIDTH, HEIGHT, 2, 60, 0.5)
    weak = propose_image_planes(image, np.ones((HEIGHT, WIDTH), bool), weak_calibration)

    assert (missing.status == insufficient.status == invalid.status
            == mismatched.status == weak.status == "abstain")
    assert missing.reason == "missing_calibration"
    assert insufficient.reason == "insufficient_usable_coverage"
    assert invalid.reason == "invalid_or_mismatched_calibration"
    assert mismatched.reason == "invalid_or_mismatched_calibration"
    assert weak.reason == "insufficient_calibration_inliers"
    assert not missing.proposals and not insufficient.proposals


def test_production_plane_proposer_contains_no_evaluation_artifact_access():
    source = Path(__file__).parents[1] / "src" / "mc_photogrammetry" / "reconstruction" / "boundary_candidates.py"
    text = source.read_text(encoding="utf-8")
    tree = ast.parse(text)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")

    assert not any("json" in item or "manifest" in item.lower()
                   or "litematic" in item.lower() for item in imports)
    assert "M4-C2-boundary-targets" not in text
    assert "reference.litematic" not in text
    assert "cameras.json" not in text
