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

cv2 = pytest.importorskip("cv2")

from mc_photogrammetry.reconstruction.boundary_candidates import (
    Chain, find_chains, propose_boundary_candidates,
)
from mc_photogrammetry.reconstruction.calibration import vanishing_points

WIDTH, HEIGHT = 640, 480
FOCAL, YAW, PITCH, ROLL = 500.0, 25.0, 0.0, 0.0


# ---------------------------------------------------------------------------
# Direct controls: hand-built Chains, no image, no calibration search.
# ---------------------------------------------------------------------------

def _chain(axis, point, direction, offset, span, strength=10.0, segment_count=3):
    return Chain(axis=axis, strength=strength, point=point, direction=direction,
                offset=offset, span=span, segment_count=segment_count)


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


def test_a_broken_edge_with_a_along_line_gap_still_forms_one_chain():
    """Two segment pieces on the exact same physical line, separated by a
    gap along the line itself (a furniture item interrupting a wall seam,
    say) - offset-based clustering must still merge them into one chain,
    since they share the same offset from the vanishing point regardless of
    how far apart they sit along the line."""
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
    # The merged chain's real support must span both pieces, not just one -
    # a detector fragmenting a thick line into several short pieces near a
    # single location would also report segment_count > 1 without this.
    assert chain.span[1] - chain.span[0] > 250.0


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
