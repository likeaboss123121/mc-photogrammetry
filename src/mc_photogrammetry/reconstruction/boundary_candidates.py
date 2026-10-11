"""Automatic 2D structural-boundary proposals from image-derived line evidence.

This is the M4-C experiment: given nothing but a photograph and its
image-derived Manhattan-world calibration (`calibration.py` - never a
manifest, never a manually-supplied length or plane), propose candidate
quadrilateral boundaries for visible planar surfaces in frame. It answers a
narrower question than block-scale reconstruction: does the 2D footprint of
a visible plane fall out of the same line evidence that already calibrates
the camera, without reading pixel colour or texture at all? Region masks a
caller freezes for scoring are read only by the caller that scores this
module's output against them - nothing in this module ever receives a
target polygon, per the project's standing rule that a manually-supplied
region is an evaluation control, never a proposal input.

The approach: every segment `calibration.detect_segments` finds is already
scored against the three calibrated vanishing directions
(`calibration.segment_residuals`) to find which lattice axis it belongs to
and how well it fits. This module groups the inlying segments for one axis
into `Chain`s - contiguous stretches of the same physical edge, distinguished
from other segments sharing the same *direction* by their distinct offset
from that direction's pencil of lines - then pairs chains **within one
axis's own pencil** as candidate opposite edges of a plane. A Manhattan
plane's two in-plane directions are two of the three *world* axes; its
opposite sides are therefore always two lines from the *same* calibrated
axis, never one line from one axis paired with a line from a different one
(a lead review named this precisely: two different horizontal world-axis
pencils do not describe opposite sides of one plane, even when both happen
to look similarly oriented in the image). A candidate quadrilateral
combines one opposite-edge pair from each of two distinct axes; degenerate,
self-intersecting or unsupported-by-any-real-segment corners are rejected
rather than silently substituted. No colour, no material, no seam/period
occupancy gate, and no plane distance: this is 2D-only, by design.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .calibration import (
    CalibrationResult,
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
MAX_BOUNDARY_CANDIDATES = 8
MAX_IMAGE_PLANE_PROPOSALS = 32
IMAGE_PLANE_CHAIN_POOL = 36
IMAGE_PLANE_CHAINS_PER_AXIS = 12
IMAGE_PLANE_SCALES = (24, 48, 96)
IMAGE_PLANE_MINIMUM_AREA = 6 * 48 * 48
IMAGE_PLANE_REGIONS_PER_SCALE = 12
IMAGE_PLANE_MINIMUM_USABLE_FRACTION = 0.20
IMAGE_PLANE_MINIMUM_CALIBRATION_INLIERS = 12
IMAGE_PLANE_MINIMUM_MASK_IOU = 0.55
IMAGE_PLANE_MINIMUM_SEGMENT_RUN = 24.0
CORNER_CONNECTIVITY_TOLERANCE = 20.0
"""How far, in pixels, a candidate's corner may sit beyond a chain's own
observed along-line extent before the corner is judged unsupported by any
real segment rather than a reasonable extrapolation of one."""
OUT_OF_FRAME_MARGIN = 3.0
"""A candidate whose corners sit further than this many image diagonals
from the frame is rejected outright rather than clipped - clipping is for
a plane that genuinely crosses the frame boundary, not for salvaging a
combination that never described anything near the photograph."""


@dataclass(frozen=True)
class Chain:
    """A group of same-direction, same-offset segments: one candidate edge."""

    axis: int
    """Which of the three calibrated lattice directions this chain's
    segments were closest to. Two chains only pair as opposite sides of one
    candidate plane when they share this value - see module docstring."""
    strength: float
    """Sum of contrast-weighted segment length. The chain's own confidence,
    computed without reference to any target - this is what ranks
    candidates, never a match score against ground truth."""
    point: tuple[float, float]
    direction: tuple[float, float]
    """A point on the chain's line and its unit direction, in image pixels."""
    offset: float
    """This chain's position along its axis's pencil of lines - the same
    coordinate `find_chains` clusters segments on. Used only to order
    distinct chains within one axis before pairing them; never compared
    across axes and never compared to the image centre."""
    span: tuple[tuple[float, float], ...]
    """The chain's real along-line support, as the union of its member
    segments' own endpoint intervals (merged where they touch or overlap) -
    not the single min-to-max envelope, and not the offset spread used for
    clustering. Two segments detected 100px apart along the same line with
    nothing detected between them are *not* evidence for that 100px gap;
    treating the envelope as if it were fully observed would let a corner
    sitting in that gap pass as supported. A candidate corner that requires
    extrapolating past every one of these intervals is evidence-free, not
    a detection (see `CORNER_CONNECTIVITY_TOLERANCE`)."""
    segment_count: int


@dataclass(frozen=True)
class BoundaryCandidate:
    """One candidate quadrilateral, plus the chains that formed it."""

    polygon: tuple[tuple[float, float], ...]
    """Corners in order, already clipped to the image rectangle. May carry
    more than 4 vertices where clipping cut across a corner."""
    strength: float
    axis_pair: tuple[int, int]
    edges: dict
    """``{"axis_a": {"axis": int, "low": Chain | None, "high": Chain | None},
    "axis_b": {...}}``. ``low``/``high`` order the pencil's two chains along
    their own axis, not by any left/right/top/bottom label. ``None`` means
    no second chain existed in that pencil; the missing side was filled
    from the image's own frame edge and `complete` is False - a boundary a
    plane's own visible crop happens to touch, not an observed wall edge."""
    complete: bool
    """True only when all four sides came from real, detected chains."""


@dataclass(frozen=True)
class PlaneProposal:
    """One deterministic, image-derived candidate plane.

    `axis` is the missing calibrated Manhattan direction, and `side` is the
    sign of that direction's component along the calibrated camera ray through
    the candidate centroid. `mask` is the pixel support used for evaluation;
    it is read-only to prevent an evaluator from changing the proposal in place.
    """

    axis: int
    side: int
    polygon: tuple[tuple[float, float], ...]
    mask: np.ndarray = field(compare=False, repr=False)
    strength: float = 0.0
    sources: tuple[str, ...] = ()
    coverage: float = 0.0


@dataclass(frozen=True)
class PlaneProposalResult:
    """Typed outcome, including abstentions when image evidence is unusable."""

    status: str
    reason: str
    proposals: tuple[PlaneProposal, ...] = ()
    stats: dict = field(default_factory=dict, compare=False)


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


def _distribute(items: list, key, cap: int) -> list:
    """Take up to `cap` items (assumed pre-sorted by descending strength),
    spreading across the groups `key` assigns them to so that one cluster
    cannot consume every slot, then filling any remaining slots by strength
    alone. Used both for capping chains across axes and for capping
    candidates across the axis-pairs/regions that produced them."""
    if len(items) <= cap:
        return list(items)
    groups: dict = {}
    for item in items:
        groups.setdefault(key(item), []).append(item)
    share = max(1, cap // max(1, len(groups)))
    selected, remaining = [], []
    for group_items in groups.values():
        selected.extend(group_items[:share])
        remaining.extend(group_items[share:])
    remaining.sort(key=lambda item: -item.strength)
    for item in remaining:
        if len(selected) >= cap:
            break
        selected.append(item)
    selected.sort(key=lambda item: -item.strength)
    return selected[:cap]


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
    material read, no plane/period assumption. The final `max_chains`
    selection is distributed across axes rather than taken as a global top-N,
    so one strong, chain-rich axis (say, many parallel floor-plank seams)
    cannot use up every slot before a less textured axis gets any.
    """
    grey = _to_grey(image)
    segments = detect_segments(enhance(grey))
    if not len(segments):
        return []
    weights = structural_weights(grey, segments)
    return _chains_from_evidence(
        grey, segments, weights, focal=focal, yaw_degrees=yaw_degrees,
        pitch_degrees=pitch_degrees, roll_degrees=roll_degrees,
        principal_point=principal_point, threshold=threshold,
        chain_gap=chain_gap, max_chains=max_chains,
    )


def _chains_from_evidence(
    grey: np.ndarray,
    segments: np.ndarray,
    weights: np.ndarray,
    *,
    focal: float,
    yaw_degrees: float,
    pitch_degrees: float,
    roll_degrees: float,
    principal_point: tuple[float, float] | None,
    threshold: float = INLIER_PIXELS,
    chain_gap: float = CHAIN_GAP_PIXELS,
    max_chains: int = MAX_CHAINS,
) -> list[Chain]:
    """Build chains from already-detected, optionally mask-clipped evidence."""
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
        axis_segments = segments[mask]
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
            mean_offset = float(np.average(offsets[group], weights=group_strengths))
            x0, y0, w0 = points[axis]
            if abs(w0) > 1e-9:
                vp_point = np.array([x0 / w0, y0 / w0])
                delta = mean_point - vp_point
                norm = float(np.linalg.norm(delta))
                direction = delta / norm if norm > 1e-9 else np.array([1.0, 0.0])
            else:
                direction = np.array([x0, y0])
                direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
            # Real along-line support: each member segment's own two
            # endpoints, projected onto the chain's direction, form one
            # observed interval - not just the pooled min/max of every
            # endpoint, which would silently call an unobserved gap between
            # two segments "supported" too.
            group_segments = axis_segments[group]
            starts = (group_segments[:, 0:2] - mean_point) @ direction
            ends = (group_segments[:, 2:4] - mean_point) @ direction
            intervals = sorted(
                (float(min(s, e)), float(max(s, e))) for s, e in zip(starts, ends))
            merged: list[list[float]] = []
            for low, high in intervals:
                if merged and low <= merged[-1][1] + chain_gap:
                    merged[-1][1] = max(merged[-1][1], high)
                else:
                    merged.append([low, high])
            chains.append(Chain(
                axis=axis, strength=total,
                point=(float(mean_point[0]), float(mean_point[1])),
                direction=(float(direction[0]), float(direction[1])),
                offset=mean_offset,
                span=tuple((low, high) for low, high in merged),
                segment_count=len(group),
            ))
    chains.sort(key=lambda chain: -chain.strength)
    return _distribute(chains, key=lambda chain: _chain_region_key(chain, width, height),
                       cap=max_chains)


def _chain_region_key(chain: Chain, width: int, height: int) -> tuple:
    """Groups a chain by its axis and a coarse 3x3 bucket of its own
    position, so the chain budget spreads across the image rather than one
    cluttered region (many parallel floor-plank seams, say) using up every
    slot before a real edge elsewhere in frame gets one."""
    column = min(2, max(0, int(3.0 * chain.point[0] / max(width, 1))))
    row = min(2, max(0, int(3.0 * chain.point[1] / max(height, 1))))
    return (chain.axis, row, column)


def _intersect(point_a, direction_a, point_b, direction_b) -> tuple[float, float] | None:
    matrix = np.array([[direction_a[0], -direction_b[0]], [direction_a[1], -direction_b[1]]])
    determinant = np.linalg.det(matrix)
    if abs(determinant) < 1e-9:
        return None
    rhs = np.array([point_b[0] - point_a[0], point_b[1] - point_a[1]])
    t = np.linalg.solve(matrix, rhs)[0]
    point = (point_a[0] + t * direction_a[0], point_a[1] + t * direction_a[1])
    if not (np.isfinite(point[0]) and np.isfinite(point[1])):
        return None
    return point


def _edge_pairs(chains_in_axis: list[Chain]) -> list[tuple[Chain, Chain | None]]:
    """Every distinct pair of chains within one axis's pencil, ordered along
    the pencil - candidate opposite edges of a plane along that axis. A
    pencil with only one detected chain offers that chain paired with
    `None`: one real, observed edge, with the plane's other bound along
    this axis left explicitly unsupported rather than assumed."""
    ordered = sorted(chains_in_axis, key=lambda chain: chain.offset)
    if len(ordered) >= 2:
        return list(combinations(ordered, 2))
    return [(ordered[0], None)]


def _fallback_line(known: Chain, width: int, height: int):
    """A frame-edge line to stand in for a pencil's missing second chain,
    on whichever side of the image the known chain does not already
    occupy - a visible crop boundary, explicitly not a detection."""
    vertical_like = abs(known.direction[1]) >= abs(known.direction[0])
    if vertical_like:
        edge_x = float(width) if known.point[0] <= width / 2.0 else 0.0
        return (edge_x, 0.0), (0.0, 1.0)
    edge_y = float(height) if known.point[1] <= height / 2.0 else 0.0
    return (0.0, edge_y), (1.0, 0.0)


def _line_of(chain: Chain | None, other: Chain, width: int, height: int):
    if chain is not None:
        return chain.point, chain.direction
    return _fallback_line(other, width, height)


def _is_simple_quad(corners: list[tuple[float, float]]) -> bool:
    """Reject self-intersecting/bowtie orderings: a properly paired quad
    from two opposite-edge pencils turns the same way at every corner."""
    signs = []
    count = len(corners)
    for index in range(count):
        a, b, c = corners[index], corners[(index + 1) % count], corners[(index + 2) % count]
        v1 = (b[0] - a[0], b[1] - a[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) > 1e-9:
            signs.append(cross > 0)
    return len(set(signs)) <= 1


def _corner_supported(corner, chain: Chain | None, tolerance: float) -> bool:
    """Whether `corner` falls near enough to at least one of `chain`'s own
    observed intervals - never near a gap *between* two intervals, which
    carries no segment evidence at all regardless of how close the two
    real detections on either side of it are."""
    if chain is None:
        return True
    projection = ((corner[0] - chain.point[0]) * chain.direction[0]
                 + (corner[1] - chain.point[1]) * chain.direction[1])
    # A flat tolerance per interval, not scaled by that interval's own
    # length: scaling by length was meant to forgive extrapolating a
    # *short* observation a little further, but applied to two intervals
    # split by a real gap it would just re-widen each side enough to
    # bridge the very gap `span` now exists to record.
    return any(low - tolerance <= projection <= high + tolerance for low, high in chain.span)


def _clip_to_rect(polygon: list[tuple[float, float]], width: int, height: int):
    """Sutherland-Hodgman clip against the image rectangle - a geometric
    clip of an already-valid polygon, not a per-vertex clamp that would
    distort a plane's true corner angles."""
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

    def edge_intersect(axis_index, value):
        def intersect(a, b):
            denominator = b[axis_index] - a[axis_index]
            t = (value - a[axis_index]) / denominator
            return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        return intersect

    points = list(polygon)
    points = clip(points, lambda p: p[0] >= 0.0, edge_intersect(0, 0.0))
    points = clip(points, lambda p: p[0] <= float(width), edge_intersect(0, float(width)))
    points = clip(points, lambda p: p[1] >= 0.0, edge_intersect(1, 0.0))
    points = clip(points, lambda p: p[1] <= float(height), edge_intersect(1, float(height)))
    return points


def _build_quadrilateral(a_low, a_high, b_low, b_high, width: int, height: int):
    """Returns ``(polygon_or_None, reason)``. ``reason`` is ``"ok"`` on
    success, or the specific check that rejected the combination - used by
    `propose_boundary_candidates` to report how many combinations were
    tried and why each rejected one was thrown out, not just the survivors."""
    a_low_line = _line_of(a_low, a_high, width, height) if a_low else _fallback_line(a_high, width, height)
    a_high_line = _line_of(a_high, a_low, width, height) if a_high else _fallback_line(a_low, width, height)
    b_low_line = _line_of(b_low, b_high, width, height) if b_low else _fallback_line(b_high, width, height)
    b_high_line = _line_of(b_high, b_low, width, height) if b_high else _fallback_line(b_low, width, height)

    corner_order = (
        (a_low_line, b_low_line, a_low, b_low),
        (a_low_line, b_high_line, a_low, b_high),
        (a_high_line, b_high_line, a_high, b_high),
        (a_high_line, b_low_line, a_high, b_low),
    )
    corners = []
    for line1, line2, chain1, chain2 in corner_order:
        point = _intersect(line1[0], line1[1], line2[0], line2[1])
        if point is None:
            return None, "degenerate_intersection"
        corners.append(point)

    diagonal = float(np.hypot(width, height))
    if any(abs(x) > OUT_OF_FRAME_MARGIN * diagonal or abs(y) > OUT_OF_FRAME_MARGIN * diagonal
          for x, y in corners):
        return None, "out_of_frame"
    if not _is_simple_quad(corners):
        return None, "self_intersecting"
    for (_, _, chain1, chain2), corner in zip(corner_order, corners):
        if not (_corner_supported(corner, chain1, CORNER_CONNECTIVITY_TOLERANCE)
               and _corner_supported(corner, chain2, CORNER_CONNECTIVITY_TOLERANCE)):
            return None, "corner_unsupported"

    clipped = _clip_to_rect(corners, width, height)
    if len(clipped) < 3:
        return None, "clipped_away"
    return tuple(clipped), "ok"


def propose_boundary_candidates(
    chains: list[Chain],
    width: int,
    height: int,
    *,
    max_candidates: int = MAX_BOUNDARY_CANDIDATES,
    stats: dict | None = None,
) -> list[BoundaryCandidate]:
    """Combine chains into candidate quadrilaterals, ranked by their own
    supporting strength - never by comparison to any target.

    A candidate's two pairs of opposite edges each come from one calibrated
    axis's own pencil of parallel-in-world lines - never one line from one
    axis paired against a line from a different axis, which would not
    describe opposite sides of a single Manhattan plane regardless of how
    similar the two lines look in the image. Degenerate, self-intersecting
    combinations and corners unsupported by any real chain's own observed
    extent are rejected outright, not silently substituted with a frame
    edge or a clamped coordinate.

    If ``stats`` is given (an empty dict the caller owns), it is filled with
    ``combinations_tried``, ``accepted`` and a ``rejected`` breakdown by
    reason - diagnostics for a report, not something the ranking itself
    consults.
    """
    by_axis: dict[int, list[Chain]] = {}
    for chain in chains:
        by_axis.setdefault(chain.axis, []).append(chain)
    axes_present = sorted(by_axis)
    if stats is not None:
        stats.update({"combinations_tried": 0, "accepted": 0, "rejected": {}})
    if len(axes_present) < 2:
        return []

    candidates = []
    for axis_a, axis_b in combinations(axes_present, 2):
        pairs_a = _edge_pairs(by_axis[axis_a])
        pairs_b = _edge_pairs(by_axis[axis_b])
        for a_low, a_high in pairs_a:
            for b_low, b_high in pairs_b:
                if stats is not None:
                    stats["combinations_tried"] += 1
                polygon, reason = _build_quadrilateral(a_low, a_high, b_low, b_high, width, height)
                if polygon is None:
                    if stats is not None:
                        stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1
                    continue
                if stats is not None:
                    stats["accepted"] += 1
                complete = all(chain is not None for chain in (a_low, a_high, b_low, b_high))
                strength = sum(chain.strength for chain in (a_low, a_high, b_low, b_high)
                              if chain is not None)
                candidates.append(BoundaryCandidate(
                    polygon=polygon, strength=strength, axis_pair=(axis_a, axis_b),
                    edges={
                        "axis_a": {"axis": axis_a, "low": a_low, "high": a_high},
                        "axis_b": {"axis": axis_b, "low": b_low, "high": b_high},
                    },
                    complete=complete,
                ))

    candidates.sort(key=lambda candidate: -candidate.strength)
    selected = _distribute(candidates, key=lambda candidate: _region_key(candidate, width, height),
                           cap=max_candidates)
    if stats is not None:
        stats["candidates_before_distribution"] = len(candidates)
        stats["candidates_returned"] = len(selected)
    return selected


def _region_key(candidate: "BoundaryCandidate", width: int, height: int) -> tuple:
    """Groups a candidate by its axis pair and a coarse 3x3 bucket of its
    own centroid - so two genuinely different planes in different parts of
    the frame (the review's "distribute... across axes and image regions")
    are not left to compete purely on strength against every candidate
    that happens to share their axis pair, which would let one strong,
    cluttered region crowd out a real plane elsewhere in the photograph."""
    centroid_x = sum(p[0] for p in candidate.polygon) / len(candidate.polygon)
    centroid_y = sum(p[1] for p in candidate.polygon) / len(candidate.polygon)
    column = min(2, max(0, int(3.0 * centroid_x / max(width, 1))))
    row = min(2, max(0, int(3.0 * centroid_y / max(height, 1))))
    return (candidate.axis_pair, row, column)


def _clip_segments_to_usable_mask(
    segments: np.ndarray,
    usable_mask: np.ndarray,
    *,
    sample_step: float = 4.0,
    minimum_run: float = IMAGE_PLANE_MINIMUM_SEGMENT_RUN,
) -> np.ndarray:
    """Split structural lines at excluded pixels and keep supported runs only."""
    height, width = usable_mask.shape
    kept: list[tuple[float, float, float, float]] = []
    for segment in np.asarray(segments, dtype=float).reshape((-1, 4)):
        x0, y0, x1, y1 = segment
        length = float(np.hypot(x1 - x0, y1 - y0))
        steps = max(2, int(np.ceil(length / sample_step)) + 1)
        fractions = np.linspace(0.0, 1.0, steps)
        xs = np.clip(np.rint(x0 + fractions * (x1 - x0)).astype(int), 0, width - 1)
        ys = np.clip(np.rint(y0 + fractions * (y1 - y0)).astype(int), 0, height - 1)
        valid = usable_mask[ys, xs]
        starts = np.flatnonzero(valid & np.r_[True, ~valid[:-1]])
        ends = np.flatnonzero(valid & np.r_[~valid[1:], True])
        for start, end in zip(starts, ends):
            if end <= start:
                continue
            ax, ay, bx, by = xs[start], ys[start], xs[end], ys[end]
            if float(np.hypot(bx - ax, by - ay)) >= minimum_run:
                kept.append((float(ax), float(ay), float(bx), float(by)))
    if not kept:
        return np.empty((0, 4), dtype=float)
    return np.asarray(kept, dtype=float)


def _polygon_mask(polygon, height: int, width: int, usable: np.ndarray) -> np.ndarray:
    import cv2

    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.rint(np.asarray(polygon, dtype=float)).astype(np.int32)
    if len(points) >= 3:
        cv2.fillPoly(mask, [points.reshape((-1, 1, 2))], 1)
    return (mask != 0) & usable


def _mask_polygon(mask: np.ndarray) -> tuple[tuple[float, float], ...]:
    import cv2

    contours, _hierarchy = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ()
    contour = max(contours, key=lambda item: (cv2.contourArea(item), -len(item)))
    perimeter = float(cv2.arcLength(contour, True))
    simplified = cv2.approxPolyDP(contour, max(1.0, perimeter * 0.002), True)
    return tuple((float(point[0][0]), float(point[0][1])) for point in simplified)


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = int(np.count_nonzero(mask_a & mask_b))
    if intersection == 0:
        return 0.0
    union = int(np.count_nonzero(mask_a | mask_b))
    return intersection / union if union else 0.0


def _plane_side(
    axis: int,
    polygon: tuple[tuple[float, float], ...],
    calibration: CalibrationResult,
    width: int,
    height: int,
) -> int:
    from .calibration import axis_directions

    centre_x = float(np.mean([point[0] for point in polygon]))
    centre_y = float(np.mean([point[1] for point in polygon]))
    principal_x, principal_y = calibration.principal_point or (width / 2.0, height / 2.0)
    ray = np.asarray([(centre_x - principal_x) / calibration.focal_pixels,
                      (principal_y - centre_y) / calibration.focal_pixels,
                      1.0], dtype=float)
    direction = axis_directions(calibration.yaw_degrees, calibration.pitch_degrees,
                                calibration.roll_degrees)[axis]
    signed = float(np.dot(direction, ray))
    if abs(signed) <= 1e-9:
        return 0
    return 1 if signed > 0.0 else -1


def _proposal_key(proposal: PlaneProposal, width: int, height: int) -> tuple:
    ys, xs = np.nonzero(proposal.mask)
    if len(xs):
        x, y = float(xs.mean()), float(ys.mean())
    else:
        x, y = width / 2.0, height / 2.0
    column = min(2, max(0, int(3.0 * x / max(width, 1))))
    row = min(2, max(0, int(3.0 * y / max(height, 1))))
    return proposal.axis, proposal.side, row, column


def _limit_planar_regions(regions, width: int, height: int, cap: int):
    """Bound per-scale memory while retaining axis and image-area coverage."""
    ranked = sorted(regions, key=lambda region: (
        -region.confidence, -region.area_tiles, region.axis,
        region.centroid()[1], region.centroid()[0], region.tiles,
    ))
    groups: dict[tuple, list] = {}
    for region in ranked:
        x, y = region.centroid()
        column = min(2, max(0, int(3.0 * x / max(width, 1))))
        row = min(2, max(0, int(3.0 * y / max(height, 1))))
        groups.setdefault((region.axis, row, column), []).append(region)
    selected = []
    cursors = {key: 0 for key in sorted(groups)}
    while len(selected) < cap:
        changed = False
        for key in sorted(groups):
            cursor = cursors[key]
            if cursor < len(groups[key]):
                selected.append(groups[key][cursor])
                cursors[key] = cursor + 1
                changed = True
                if len(selected) >= cap:
                    break
        if not changed:
            break
    return selected


def _empty_plane_result(reason: str, stats: dict | None = None) -> PlaneProposalResult:
    return PlaneProposalResult(status="abstain", reason=reason, proposals=(),
                               stats={} if stats is None else dict(stats))


def propose_image_planes(
    image: np.ndarray,
    usable_mask: np.ndarray,
    calibration: CalibrationResult | None,
    *,
    max_candidates: int = MAX_IMAGE_PLANE_PROPOSALS,
) -> PlaneProposalResult:
    """Propose diverse planes from pixels, a derived usable mask, and calibration.

    The detector runs once. Three bounded tile scales vote for regions whose
    missing Manhattan direction is their normal; the same mask-clipped segment
    set forms supported line chains and axis-compatible quadrilateral closures.
    Overlapping evidence may strengthen a proposal, but proposal masks are never
    unioned: spatially distinct and orthogonal surfaces remain distinct records.
    """
    from .surfaces import planar_regions

    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError("image must be a 2D or 3D numpy array")
    grey = _to_grey(image)
    height, width = grey.shape[:2]
    usable = np.asarray(usable_mask, dtype=bool)
    if usable.shape != (height, width):
        raise ValueError("usable_mask must match the image height and width")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    cap = min(int(max_candidates), MAX_IMAGE_PLANE_PROPOSALS)
    usable_count = int(usable.sum())
    usable_fraction = usable_count / max(height * width, 1)
    stats = {
        "usable_fraction": float(usable_fraction),
        "detected_segments": 0,
        "usable_segments": 0,
        "region_candidates": 0,
        "chains": 0,
        "closures": 0,
        "proposals_before_cap": 0,
        "proposals_returned": 0,
    }
    if calibration is None:
        return _empty_plane_result("missing_calibration", stats)
    if not isinstance(calibration, CalibrationResult):
        return _empty_plane_result("invalid_or_mismatched_calibration", stats)
    numeric = (calibration.focal_pixels, calibration.yaw_degrees,
               calibration.pitch_degrees, calibration.roll_degrees,
               calibration.median_residual)
    principal_valid = (
        calibration.principal_point is None
        or (len(calibration.principal_point) == 2
            and all(np.isfinite(value) for value in calibration.principal_point))
    )
    if (not all(np.isfinite(value) for value in numeric)
            or not principal_valid
            or calibration.focal_pixels <= 0.0
            or calibration.width != width or calibration.height != height):
        return _empty_plane_result("invalid_or_mismatched_calibration", stats)
    if calibration.inliers < IMAGE_PLANE_MINIMUM_CALIBRATION_INLIERS:
        return _empty_plane_result("insufficient_calibration_inliers", stats)
    if usable_fraction < IMAGE_PLANE_MINIMUM_USABLE_FRACTION:
        return _empty_plane_result("insufficient_usable_coverage", stats)
    if usable_count == 0:
        return _empty_plane_result("insufficient_usable_coverage", stats)

    raw_segments = detect_segments(enhance(grey))
    stats["detected_segments"] = int(len(raw_segments))
    if not len(raw_segments):
        return _empty_plane_result("no_detected_segments", stats)
    segments = _clip_segments_to_usable_mask(raw_segments, usable)
    stats["usable_segments"] = int(len(segments))
    if not len(segments):
        return _empty_plane_result("no_usable_segments", stats)
    reverse = ((segments[:, 0] > segments[:, 2])
               | ((segments[:, 0] == segments[:, 2])
                  & (segments[:, 1] > segments[:, 3])))
    segments[reverse] = segments[reverse][:, [2, 3, 0, 1]]
    segment_order = np.lexsort((segments[:, 3], segments[:, 2],
                                segments[:, 1], segments[:, 0]))
    segments = segments[segment_order]
    segment_weights = structural_weights(grey, segments)

    provisional: list[PlaneProposal] = []
    for tile in IMAGE_PLANE_SCALES:
        minimum_tiles = max(1, int(np.ceil(IMAGE_PLANE_MINIMUM_AREA / float(tile * tile))))
        regions = planar_regions(
            segments, calibration.focal_pixels, calibration.yaw_degrees,
            calibration.pitch_degrees, width, height, tile=tile,
            minimum_tiles=minimum_tiles,
            roll_degrees=calibration.roll_degrees,
        )
        regions = _limit_planar_regions(
            regions, width, height, IMAGE_PLANE_REGIONS_PER_SCALE)
        for region in regions:
            mask = (region.mask(height, width) > 0.0) & usable
            area = int(mask.sum())
            if area < IMAGE_PLANE_MINIMUM_AREA:
                continue
            polygon = _mask_polygon(mask)
            if len(polygon) < 3:
                continue
            side = _plane_side(region.axis, polygon, calibration, width, height)
            area_factor = min(1.0, area / float(IMAGE_PLANE_MINIMUM_AREA * 2))
            strength = float(region.confidence * np.sqrt(area_factor))
            mask.setflags(write=False)
            provisional.append(PlaneProposal(
                axis=region.axis, side=side, polygon=polygon, mask=mask,
                strength=strength, sources=(f"region_tile_{tile}",),
                coverage=area / max(usable_count, 1),
            ))
    stats["region_candidates"] = len(provisional)

    all_chains = _chains_from_evidence(
        grey, segments, segment_weights,
        focal=calibration.focal_pixels, yaw_degrees=calibration.yaw_degrees,
        pitch_degrees=calibration.pitch_degrees,
        roll_degrees=calibration.roll_degrees,
        principal_point=calibration.principal_point,
        max_chains=10_000,
    )
    chains: list[Chain] = []
    for axis in range(3):
        axis_chains = [chain for chain in all_chains if chain.axis == axis]
        axis_chains.sort(key=lambda chain: (
            -chain.strength, chain.offset, chain.point[1], chain.point[0]))
        chains.extend(_distribute(
            axis_chains,
            key=lambda chain: _chain_region_key(chain, width, height),
            cap=IMAGE_PLANE_CHAINS_PER_AXIS,
        ))
    chains.sort(key=lambda chain: (chain.axis, -chain.strength, chain.offset,
                                   chain.point[1], chain.point[0]))
    chains = chains[:IMAGE_PLANE_CHAIN_POOL]
    stats["chains"] = len(chains)

    closure_stats: dict = {}
    closures = propose_boundary_candidates(
        chains, width, height, max_candidates=MAX_IMAGE_PLANE_PROPOSALS,
        stats=closure_stats,
    )
    stats["closure_combinations_tried"] = int(closure_stats.get("combinations_tried", 0))
    stats["closure_candidates_before_cap"] = int(
        closure_stats.get("candidates_before_distribution", 0))
    closure_strength_max = max((candidate.strength for candidate in closures), default=1.0)
    for candidate in closures:
        normal_axis = 3 - sum(candidate.axis_pair)
        mask = _polygon_mask(candidate.polygon, height, width, usable)
        area = int(mask.sum())
        if area < max(64, IMAGE_PLANE_MINIMUM_AREA // 8):
            continue
        side = _plane_side(normal_axis, candidate.polygon, calibration, width, height)
        area_factor = min(1.0, area / float(IMAGE_PLANE_MINIMUM_AREA))
        chain_score = candidate.strength / max(closure_strength_max, 1e-9)
        strength = float(chain_score * np.sqrt(area_factor))
        mask.setflags(write=False)
        provisional.append(PlaneProposal(
            axis=normal_axis, side=side, polygon=tuple(candidate.polygon),
            mask=mask, strength=strength, sources=("axis_chain_closure",),
            coverage=area / max(usable_count, 1),
        ))
    stats["closures"] = len(closures)

    provisional.sort(key=lambda proposal: (
        -proposal.strength, proposal.axis, proposal.side,
        _proposal_key(proposal, width, height), proposal.sources,
    ))
    deduplicated: list[PlaneProposal] = []
    for proposal in provisional:
        duplicate_index = next((index for index, prior in enumerate(deduplicated)
                                if prior.axis == proposal.axis
                                and prior.side == proposal.side
                                and _mask_iou(prior.mask, proposal.mask)
                                >= IMAGE_PLANE_MINIMUM_MASK_IOU), None)
        if duplicate_index is None:
            deduplicated.append(proposal)
            continue
        prior = deduplicated[duplicate_index]
        sources = tuple(sorted(set(prior.sources + proposal.sources)))
        deduplicated[duplicate_index] = PlaneProposal(
            axis=prior.axis, side=prior.side, polygon=prior.polygon,
            mask=prior.mask, strength=min(1.0, prior.strength + 0.10 * proposal.strength),
            sources=sources, coverage=prior.coverage,
        )

    deduplicated.sort(key=lambda proposal: (
        -proposal.strength, proposal.axis, proposal.side,
        _proposal_key(proposal, width, height), proposal.sources,
    ))
    stats["proposals_before_cap"] = len(deduplicated)
    selected = _distribute(
        deduplicated,
        key=lambda proposal: _proposal_key(proposal, width, height),
        cap=cap,
    )
    stats["proposals_returned"] = len(selected)
    if not selected:
        return _empty_plane_result("no_plane_evidence", stats)
    return PlaneProposalResult(status="ok", reason="image_evidence", proposals=tuple(selected),
                               stats=stats)
