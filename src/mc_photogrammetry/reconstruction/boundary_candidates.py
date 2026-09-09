"""Automatic 2D structural-boundary proposals from image-derived line evidence.

This is the M4-C experiment: given nothing but a photograph and its
image-derived Manhattan-world calibration (`calibration.py` - never a
manifest, never a manually-supplied length or plane), propose candidate
quadrilateral boundaries for the dominant visible planar surface in frame.
It answers a narrower question than block-scale reconstruction: does the
2D footprint of a visible plane fall out of the same line evidence that
already calibrates the camera, without reading pixel colour or texture at
all? Region masks a caller freezes for scoring are read only by the caller
that scores this module's output against them - nothing in this module
ever receives a target polygon, per the project's standing rule that a
manually-supplied region is an evaluation control, never a proposal input.

The approach: every segment `calibration.detect_segments` finds is already
scored against the three calibrated vanishing directions
(`calibration.segment_residuals`) to find which lattice axis it belongs to
and how well it fits. This module goes one step further and groups the
inlying segments for one axis into `Chain`s - contiguous stretches of the
same physical edge, distinguished from other segments sharing the same
*direction* by their distinct offset from that direction's pencil of
lines. A chain's own image-space geometry (nearly vertical or nearly
horizontal, regardless of which of the three calibrated axes produced it)
decides whether it is offered as a left/right or a top/bottom candidate
edge; the four strongest, most compatible edges combine into a candidate
quadrilateral. No colour, no material, no seam/period occupancy gate, and
no plane distance: this is 2D-only, by design.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .calibration import (
    INLIER_PIXELS,
    detect_segments,
    enhance,
    segment_residuals,
    structural_weights,
    vanishing_points,
)

CHAIN_GAP_PIXELS = 10.0
"""Offset gap, in pixels, beyond which two same-direction segments are
considered different physical edges rather than pieces of the same one.
Comparable in scale to `calibration.INLIER_PIXELS`'s own tolerance, just
applied along the direction's pencil rather than across a single line."""

MAX_CHAINS = 12
MAX_EDGE_CANDIDATES = 3
"""How many of the strongest chains on each side (left/right/top/bottom)
are offered as edge candidates before combining into quadrilaterals."""
MAX_BOUNDARY_CANDIDATES = 8


@dataclass(frozen=True)
class Chain:
    """A group of same-direction, same-offset segments: one candidate edge."""

    axis: int
    """Which of the three calibrated lattice directions this chain's
    segments were closest to - kept for provenance, not used to decide the
    chain's left/right-vs-top/bottom role (see module docstring)."""
    strength: float
    """Sum of contrast-weighted segment length. The chain's own confidence,
    computed without reference to any target - this is what ranks
    candidates, never a match score against ground truth."""
    point: tuple[float, float]
    direction: tuple[float, float]
    """A point on the chain's line and its unit direction, in image pixels."""
    span: tuple[float, float]
    """Extent of the chain's member segments' offsets, for diagnostics."""
    segment_count: int


@dataclass(frozen=True)
class BoundaryCandidate:
    """One candidate quadrilateral, plus the chains (or frame edges) that formed it."""

    polygon: tuple[tuple[float, float], ...]
    strength: float
    edges: dict
    """``{"left": Chain | None, "right": ..., "top": ..., "bottom": ...}``.
    ``None`` means no chain supported that side and the image frame's own
    edge was used instead - an honest fallback, not a detection."""


def _to_grey(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    import cv2

    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _chain_offsets(midpoints: np.ndarray, vp: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Position of each midpoint along the direction's pencil of lines.

    For a finite vanishing point this is an arc-length (angle from the
    reference point times the reference's own radius), which keeps the
    unit comparable in pixels to the infinite case below regardless of how
    far away the vanishing point sits. For a direction parallel to the
    image plane (``w == 0``, the classical case for the vertical axis on a
    level, unrolled camera) it is the plain perpendicular offset.
    """
    x0, y0, w0 = vp
    if abs(w0) > 1e-9:
        point = np.array([x0 / w0, y0 / w0])
        to_reference = reference - point
        radius = max(float(np.linalg.norm(to_reference)), 1.0)
        reference_angle = float(np.arctan2(to_reference[1], to_reference[0]))
        to_mid = midpoints - point
        angles = np.arctan2(to_mid[:, 1], to_mid[:, 0])
        delta = np.mod(angles - reference_angle + np.pi, 2.0 * np.pi) - np.pi
        return delta * radius
    direction = np.array([x0, y0])
    direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
    perpendicular = np.array([-direction[1], direction[0]])
    return (midpoints - reference) @ perpendicular


def find_chains(
    image: np.ndarray,
    *,
    focal: float,
    yaw_degrees: float,
    pitch_degrees: float,
    roll_degrees: float = 0.0,
    principal_point: tuple[float, float] | None = None,
    threshold: float = INLIER_PIXELS,
    chain_gap: float = CHAIN_GAP_PIXELS,
    max_chains: int = MAX_CHAINS,
) -> list[Chain]:
    """Group calibration-inlier segments into ranked candidate edges.

    Reuses `calibration.detect_segments`/`structural_weights` for line
    evidence and `calibration.vanishing_points`/`segment_residuals` for the
    already-calibrated camera's three directions - no new detector, no
    material read, no plane/period assumption.
    """
    grey = _to_grey(image)
    segments = detect_segments(enhance(grey))
    if not len(segments):
        return []
    weights = structural_weights(grey, segments)
    height, width = grey.shape[:2]
    points = vanishing_points(focal, yaw_degrees, pitch_degrees, width, height,
                              roll_degrees, principal_point)
    residuals = segment_residuals(segments, points)
    owner = residuals.argmin(axis=1)
    closest = residuals.min(axis=1)
    inlier = closest <= threshold
    lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
    strengths = lengths * weights
    midpoints = np.stack([(segments[:, 0] + segments[:, 2]) / 2.0,
                          (segments[:, 1] + segments[:, 3]) / 2.0], axis=1)
    reference = np.array([width / 2.0, height / 2.0])

    chains: list[Chain] = []
    for axis in range(3):
        mask = inlier & (owner == axis)
        if not mask.any():
            continue
        axis_midpoints = midpoints[mask]
        axis_strengths = strengths[mask]
        offsets = _chain_offsets(axis_midpoints, points[axis], reference)
        order = np.argsort(offsets)
        groups: list[list[int]] = [[int(order[0])]]
        for index in order[1:]:
            if offsets[index] - offsets[groups[-1][-1]] > chain_gap:
                groups.append([])
            groups[-1].append(int(index))
        for group in groups:
            group_strengths = axis_strengths[group]
            total = float(group_strengths.sum())
            mean_point = np.average(axis_midpoints[group], axis=0, weights=group_strengths)
            group_offsets = offsets[group]
            x0, y0, w0 = points[axis]
            if abs(w0) > 1e-9:
                vp_point = np.array([x0 / w0, y0 / w0])
                delta = mean_point - vp_point
                norm = float(np.linalg.norm(delta))
                direction = delta / norm if norm > 1e-9 else np.array([1.0, 0.0])
            else:
                direction = np.array([x0, y0])
                direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
            chains.append(Chain(
                axis=axis, strength=total,
                point=(float(mean_point[0]), float(mean_point[1])),
                direction=(float(direction[0]), float(direction[1])),
                span=(float(group_offsets.min()), float(group_offsets.max())),
                segment_count=len(group),
            ))
    chains.sort(key=lambda chain: -chain.strength)
    return chains[:max_chains]


def _intersect(point_a, direction_a, point_b, direction_b) -> tuple[float, float] | None:
    matrix = np.array([[direction_a[0], -direction_b[0]], [direction_a[1], -direction_b[1]]])
    determinant = np.linalg.det(matrix)
    if abs(determinant) < 1e-9:
        return None
    rhs = np.array([point_b[0] - point_a[0], point_b[1] - point_a[1]])
    t = np.linalg.solve(matrix, rhs)[0]
    return (point_a[0] + t * direction_a[0], point_a[1] + t * direction_a[1])


def propose_boundary_candidates(
    chains: list[Chain],
    width: int,
    height: int,
    *,
    max_edge_candidates: int = MAX_EDGE_CANDIDATES,
    max_candidates: int = MAX_BOUNDARY_CANDIDATES,
) -> list[BoundaryCandidate]:
    """Combine chains into candidate quadrilaterals, ranked by their own
    supporting strength - never by comparison to any target.

    A chain's role (left/right edge vs. top/bottom edge) comes from its own
    projected orientation in this image, not from which calibrated axis
    produced it: a nearly-vertical chain is a left/right candidate, a
    nearly-horizontal one a top/bottom candidate, regardless of axis. Where
    no chain supports a side at all, the image's own frame edge fills that
    side - an explicit, zero-strength fallback, not a detection.
    """
    center_x, center_y = width / 2.0, height / 2.0
    vertical = [c for c in chains if abs(c.direction[1]) >= abs(c.direction[0])]
    horizontal = [c for c in chains if abs(c.direction[1]) < abs(c.direction[0])]

    def top_n(items, condition):
        matching = sorted((c for c in items if condition(c)), key=lambda c: -c.strength)
        return matching[:max_edge_candidates]

    left = top_n(vertical, lambda c: c.point[0] <= center_x) or [None]
    right = top_n(vertical, lambda c: c.point[0] > center_x) or [None]
    top = top_n(horizontal, lambda c: c.point[1] <= center_y) or [None]
    bottom = top_n(horizontal, lambda c: c.point[1] > center_y) or [None]

    frame_edges = {
        "left": ((0.0, 0.0), (0.0, 1.0)),
        "right": ((float(width), 0.0), (0.0, 1.0)),
        "top": ((0.0, 0.0), (1.0, 0.0)),
        "bottom": ((0.0, float(height)), (1.0, 0.0)),
    }

    def line_of(role: str, chain: Chain | None):
        if chain is None:
            return frame_edges[role]
        return chain.point, chain.direction

    combos = []
    for left_chain, right_chain, top_chain, bottom_chain in product(left, right, top, bottom):
        edges = {"left": left_chain, "right": right_chain, "top": top_chain, "bottom": bottom_chain}
        strength = sum(c.strength for c in edges.values() if c is not None)
        combos.append((strength, edges))
    combos.sort(key=lambda item: -item[0])

    candidates = []
    for strength, edges in combos[:max_candidates]:
        top_line = line_of("top", edges["top"])
        bottom_line = line_of("bottom", edges["bottom"])
        left_line = line_of("left", edges["left"])
        right_line = line_of("right", edges["right"])
        corners = []
        for horizontal_line, vertical_line in (
            (top_line, left_line), (top_line, right_line),
            (bottom_line, right_line), (bottom_line, left_line),
        ):
            point = _intersect(horizontal_line[0], horizontal_line[1],
                               vertical_line[0], vertical_line[1])
            if point is None:
                point = (0.0, 0.0)
            corners.append((
                float(np.clip(point[0], -width, 2 * width)),
                float(np.clip(point[1], -height, 2 * height)),
            ))
        candidates.append(BoundaryCandidate(polygon=tuple(corners), strength=strength, edges=edges))
    return candidates
