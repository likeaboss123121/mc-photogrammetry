"""Synthetic control for the M4-C automatic boundary proposal.

Draws a rectangle's four edges as real line segments on lines that pass
exactly through a real, image-derived-style calibration's vanishing
points (computed by `calibration.vanishing_points`, not hand-picked), then
checks that `find_chains`/`propose_boundary_candidates` recover the
rectangle from nothing but those segments - no colour, no target, no
manifest.
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from mc_photogrammetry.reconstruction.boundary_candidates import (
    find_chains, propose_boundary_candidates,
)
from mc_photogrammetry.reconstruction.calibration import vanishing_points

WIDTH, HEIGHT = 640, 480
FOCAL, YAW, PITCH, ROLL = 500.0, 25.0, 0.0, 0.0


def _line_point(vp, other_point):
    """A point on the line through `vp` (homogeneous) and `other_point`, far enough
    along to draw a long, well-conditioned segment."""
    x0, y0, w0 = vp
    if abs(w0) > 1e-9:
        point = np.array([x0 / w0, y0 / w0])
        direction = np.array(other_point) - point
    else:
        direction = np.array([x0, y0])
    direction = direction / np.linalg.norm(direction)
    return direction


def _draw_rectangle(points):
    """A blank frame with a rectangle's four edges, each drawn exactly on the
    line through the calibrated vanishing point and a chosen anchor - a
    synthetic Manhattan-world facade, not an arbitrary axis-aligned box."""
    image = np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)
    left_x, right_x, top_y, bottom_y = 180, 460, 120, 360

    def draw_through(vp, anchor, length=1000.0):
        # Each anchor defines its own line through the vanishing point - a
        # pencil of parallel-in-world lines through one point has a
        # different image-space direction for every point other than the
        # vanishing point itself, so the direction cannot be shared
        # between two differently-positioned edges.
        direction = _line_point(vp, anchor)
        anchor = np.array(anchor, dtype=float)
        a = anchor - direction * length
        b = anchor + direction * length
        cv2.line(image, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)),
                 (220, 220, 220), 2, cv2.LINE_AA)

    left_anchor = (left_x, (top_y + bottom_y) / 2.0)
    right_anchor = (right_x, (top_y + bottom_y) / 2.0)
    draw_through(points[1], left_anchor)
    draw_through(points[1], right_anchor)
    top_anchor = (WIDTH / 2.0, top_y)
    bottom_anchor = (WIDTH / 2.0, bottom_y)
    draw_through(points[0], top_anchor)
    draw_through(points[0], bottom_anchor)
    return image, (left_x, right_x, top_y, bottom_y)


def test_find_chains_recovers_four_edges_from_pure_line_evidence():
    points = vanishing_points(FOCAL, YAW, PITCH, WIDTH, HEIGHT, ROLL)
    image, _ = _draw_rectangle(points)

    chains = find_chains(image, focal=FOCAL, yaw_degrees=YAW, pitch_degrees=PITCH,
                         roll_degrees=ROLL)

    assert len(chains) >= 4
    vertical = [c for c in chains if abs(c.direction[1]) >= abs(c.direction[0])]
    horizontal = [c for c in chains if abs(c.direction[1]) < abs(c.direction[0])]
    assert len(vertical) >= 2
    assert len(horizontal) >= 2


def _analytic_intersection(vp_a, anchor_a, vp_b, anchor_b):
    """Independent line-line intersection, using the same anchors/vanishing
    points `_draw_rectangle` drew the edges through - a second, separately
    written implementation so this test does not just restate the module's
    own arithmetic back at itself."""
    a_direction = _line_point(vp_a, anchor_a)
    b_direction = _line_point(vp_b, anchor_b)
    matrix = np.array([[a_direction[0], -b_direction[0]], [a_direction[1], -b_direction[1]]])
    rhs = np.array([anchor_b[0] - anchor_a[0], anchor_b[1] - anchor_a[1]])
    t = np.linalg.solve(matrix, rhs)[0]
    return np.array(anchor_a) + t * a_direction


def test_propose_boundary_candidates_recovers_the_drawn_rectangle():
    points = vanishing_points(FOCAL, YAW, PITCH, WIDTH, HEIGHT, ROLL)
    image, (left_x, right_x, top_y, bottom_y) = _draw_rectangle(points)

    chains = find_chains(image, focal=FOCAL, yaw_degrees=YAW, pitch_degrees=PITCH,
                         roll_degrees=ROLL)
    candidates = propose_boundary_candidates(chains, WIDTH, HEIGHT)

    assert candidates
    best = candidates[0]
    assert best.strength == max(c.strength for c in candidates)

    left_anchor = (left_x, (top_y + bottom_y) / 2.0)
    right_anchor = (right_x, (top_y + bottom_y) / 2.0)
    top_anchor = (WIDTH / 2.0, top_y)
    bottom_anchor = (WIDTH / 2.0, bottom_y)
    # The top/bottom edges converge toward a finite vanishing point, so they
    # are not horizontal in image space - the true corners are wherever
    # those converging lines actually cross the (image-plane-parallel,
    # genuinely vertical) left/right edges, not a naive axis-aligned box.
    expected = [
        _analytic_intersection(points[0], top_anchor, points[1], left_anchor),
        _analytic_intersection(points[0], top_anchor, points[1], right_anchor),
        _analytic_intersection(points[0], bottom_anchor, points[1], right_anchor),
        _analytic_intersection(points[0], bottom_anchor, points[1], left_anchor),
    ]
    for corner, target in zip(best.polygon, expected):
        assert corner[0] == pytest.approx(target[0], abs=3.0)
        assert corner[1] == pytest.approx(target[1], abs=3.0)


def test_find_chains_on_a_blank_image_returns_nothing():
    image = np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)
    chains = find_chains(image, focal=FOCAL, yaw_degrees=YAW, pitch_degrees=PITCH,
                         roll_degrees=ROLL)
    assert chains == []


def test_propose_boundary_candidates_falls_back_to_frame_edges_when_a_side_is_unsupported():
    points = vanishing_points(FOCAL, YAW, PITCH, WIDTH, HEIGHT, ROLL)
    image, (left_x, right_x, top_y, bottom_y) = _draw_rectangle(points)
    # Blank out everything left of centre so no vertical chain survives there.
    image[:, : WIDTH // 2] = 40

    chains = find_chains(image, focal=FOCAL, yaw_degrees=YAW, pitch_degrees=PITCH,
                         roll_degrees=ROLL)
    candidates = propose_boundary_candidates(chains, WIDTH, HEIGHT)

    assert candidates
    best = candidates[0]
    assert best.edges["left"] is None
    xs = [p[0] for p in best.polygon]
    assert min(xs) == pytest.approx(0.0, abs=1e-6)
