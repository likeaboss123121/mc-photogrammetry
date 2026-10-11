"""Directed-candidate research matcher for one camera pair (DS-18).

DS-14 built a *reciprocal* top-k descriptor graph: an edge survives only when each
endpoint is in the other's k nearest neighbours. DS-15 then measured that matcher
over all 37 frozen pairs and DS-16/DS-16B attributed most of the remaining loss to
the directed and later stages, not to the oracle. DS-18 asks the complementary
research question: **if reciprocity is not required, does a matcher that scores
each left endpoint first recover geometry the symmetric graph discards?**

This module is the smallest such matcher. It is research-only and no production
path imports it:

* build the **directed left -> right top-k descriptor graph** (default ``k = 5``,
  hard edge cap, truncation reported, fully deterministic) - a left keypoint keeps
  its k nearest right descriptors whatever the right side thinks of it;
* for each relative yaw class reuse DS-14's corrected **2-edge endpoint-distinct**
  geometric hypotheses (deterministic seeded RANSAC, rank checks, unit baseline);
* score **per left endpoint first**: keep each left endpoint's best
  residual-qualified edge, *then* resolve right-endpoint collisions
  deterministically, so no directed truth is discarded before geometry sees it;
* rank hypotheses by unique-left support, unique-right support, spatial bins,
  median selected residual and descriptor cost;
* enforce DS-14's support / spread / parallax / cheirality / conditioning gates
  and a distinct-hypothesis margin; abstain with a typed reason rather than
  relaxing a gate;
* hand the final one-to-one set to DS-11's public :func:`solve_pair` and require
  the same yaw class.

Every primitive is reused from the accepted DS-12/DS-14 modules so a threshold
cannot drift: feature detection and the authoritative descriptor-footprint mask
(DS-12), the graph distance kernel (DS-12), the epipolar residual / gate helpers
(DS-14) and the public pair solver (DS-11). ``DirectedMatchSettings`` is the
DS-14 ``JointMatchSettings`` type itself - the declared bounds are not re-defined
and not tuned. The extraction half stays blind: it reads the two frames and the
usable mask only, and camera estimates are consumed by the geometric stage alone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mc_photogrammetry.pipeline.connected_camera import PairGates, PairPose, solve_pair
from mc_photogrammetry.pipeline.contracts import CameraEstimate
from mc_photogrammetry.pipeline.correspondence import (
    CorrespondenceSettings,
    FrameFeatures,
    _descriptors,
    _k_nearest,
    detect_features,
    select_mutual_matches,
    usable_mask_for,
)
from mc_photogrammetry.pipeline.joint_matching import (
    JOINT_REFUSAL_REASONS,
    JointMatchSettings,
    _bins_covered,
    _camera_rays,
    _cameras_ok,
    _median_angle_degrees,
    _positive_depth_fraction,
    _residuals,
    _ry90,
    _sample_rank_ok,
    _singular_values,
    _vector_angle_degrees,
    _winner_refusal,
    _world_to_camera,
)

__all__ = [
    "DIRECTED_REFUSAL_REASONS",
    "DirectedCandidateGraph",
    "DirectedHypothesisScore",
    "DirectedMatchSettings",
    "DirectedPairResult",
    "directed_topk_graph",
    "match_and_solve_pair",
    "match_feature_frames",
]

_TINY = 1e-12
_YAW_CLASSES: tuple[int, ...] = (0, 1, 2, 3)
_RATIO_TEST = 0.75

DIRECTED_REFUSAL_REASONS: tuple[str, ...] = JOINT_REFUSAL_REASONS
"""The typed refusal tokens this matcher may report (DS-14's, unchanged).

The gates are DS-14's gates, so an invented token would be misleading; the
directed-specific statistics (unique-left / unique-right support) travel on the
result and the per-hypothesis scores instead.
"""

DirectedMatchSettings = JointMatchSettings
"""DS-18's declared bounds are DS-14's ``JointMatchSettings`` object, unchanged.

A distinct class here would be a second place to keep the same numbers in sync;
the research question is about the graph and the scoring order, not the bounds.
"""


@dataclass(frozen=True, slots=True)
class DirectedCandidateGraph:
    """Directed left -> right top-k descriptor graph for one frame pair.

    ``edges`` rows are ``(left_index, right_index)`` of the k nearest right
    descriptors of every left keypoint. An edge needs no agreement from the right
    side, so the graph is directed and many-to-one by construction; one-to-one
    adjudication is the geometry stage's job. Edges are sorted by
    ``(distance, left_index, right_index)`` and truncated to the declared cap;
    ``truncated`` counts edges dropped by that cap and is reported, never silent.
    """

    edges: np.ndarray
    distances: np.ndarray
    left_keypoints: int
    right_keypoints: int
    total_edges: int
    truncated: int


@dataclass(frozen=True, slots=True)
class DirectedHypothesisScore:
    """Diagnostics for one directed (yaw class, baseline mode) hypothesis.

    ``support`` is the final one-to-one selection the hypothesis survives with.
    ``left_support`` and ``right_support`` are the pre-collision statistics the
    hypotheses are ranked by: the number of distinct left endpoints holding a
    residual-qualified edge, and the number of distinct right endpoints among
    those edges. ``median_all_edge_residual_degrees`` is transparency only and is
    deliberately not a gate - the candidate graph is mostly false edges.
    """

    yaw_class: int
    mode_index: int
    support: int
    left_support: int
    right_support: int
    spatial_bins_left: int
    spatial_bins_right: int
    spatial_bins: int
    median_selected_residual_degrees: float
    descriptor_cost: float
    median_all_edge_residual_degrees: float
    parallax_degrees: float | None
    cheirality_fraction: float | None
    singular_ratio: float | None
    baseline_direction: tuple[float, float, float]
    passed: bool
    refusal: str


@dataclass(frozen=True, slots=True)
class DirectedPairResult:
    """One pair's directed result: a DS-11 pose on the selected set, or a refusal.

    On a solve, ``pose`` is the *public DS-11* :class:`PairPose` produced by
    :func:`~mc_photogrammetry.pipeline.connected_camera.solve_pair` on ``matches``
    and the winner's yaw class equals ``pose.right_yaw_class``. On a refusal every
    pose field is ``None``; ``matches`` still carries the best final one-to-one
    set (possibly empty) so the refusal is auditable, and ``hypotheses`` carries
    every candidate the matcher ranked.
    """

    left_id: str
    right_id: str
    status: str
    reason: str
    pose: PairPose | None
    right_yaw_class: int | None
    matches: np.ndarray
    raw_keypoints: tuple[int, int]
    boundary_dropped: tuple[int, int]
    candidate_edges: int
    total_graph_edges: int
    truncated: int
    ratio_survivors: tuple[int, int]
    mutual_survivors: int
    support: int
    left_support: int
    right_support: int
    support_floor: int
    spatial_bins_left: int
    spatial_bins_right: int
    spatial_bins_total: int
    median_selected_residual_degrees: float | None
    descriptor_cost: float | None
    parallax_degrees: float | None
    cheirality_fraction: float | None
    singular_ratio: float | None
    ambiguity_margin: int | None
    hypotheses: tuple[DirectedHypothesisScore, ...]

    @property
    def solved(self) -> bool:
        return self.pose is not None and bool(self.pose.solved)


@dataclass(frozen=True, slots=True)
class _Mode:
    """One accepted (yaw class, baseline) hypothesis before public formatting."""

    yaw_class: int
    mode_index: int
    support: int
    left_support: int
    right_support: int
    matches: np.ndarray
    selection: np.ndarray
    spatial_bins_left: int
    spatial_bins_right: int
    median_selected_residual: float
    descriptor_cost: float
    median_all_edge_residual: float
    parallax: float | None
    cheirality: float | None
    singular_ratio: float | None
    baseline_direction: tuple[float, float, float]


def directed_topk_graph(
    left_descriptors: np.ndarray,
    right_descriptors: np.ndarray,
    *,
    neighbors: int = 5,
    max_edges: int = 8192,
    distance_ceiling: float | None = None,
) -> DirectedCandidateGraph:
    """Build the directed left -> right top-k descriptor graph, deterministically.

    Each left keypoint keeps its ``neighbors`` nearest right descriptors. No
    reciprocity is required, so an edge can exist while the reverse nearest
    neighbour disagrees. The distance kernel is DS-12's blocked Gram expansion
    (``correspondence._k_nearest``), whose tie-break is the train index, so the
    graph is bit-identical across runs and memory orders. A cap truncates the
    least trustworthy (largest-distance) edges first.
    """
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    if max_edges < 1:
        raise ValueError("max_edges must be positive")
    if distance_ceiling is not None and distance_ceiling < 0.0:
        raise ValueError("distance_ceiling must be non-negative")
    left = _descriptors(left_descriptors, "left_descriptors")
    right = _descriptors(right_descriptors, "right_descriptors")
    left_count, right_count = len(left), len(right)
    if left_count == 0 or right_count == 0 or left.size == 0 or right.size == 0:
        return _empty_graph(left_count, right_count)
    if left.shape[1] != right.shape[1]:
        raise ValueError("descriptor dimensions must match")

    keep = min(int(neighbors), right_count)
    index, distance = _k_nearest(left, right, keep)
    rows: list[tuple[int, int, float]] = []
    for origin in range(left_count):
        for slot in range(index.shape[1]):
            target = int(index[origin, slot])
            if target < 0:
                continue
            value = float(distance[origin, slot])
            if distance_ceiling is not None and value > distance_ceiling:
                continue
            rows.append((origin, target, value))

    if not rows:
        return _empty_graph(left_count, right_count)
    table = np.asarray(rows, dtype=np.float64)
    order = np.lexsort((table[:, 1], table[:, 0], table[:, 2]))
    table = table[order]
    total = int(len(table))
    truncated = max(0, total - int(max_edges))
    table = table[: int(max_edges)]
    return DirectedCandidateGraph(
        edges=table[:, :2].astype(np.int64),
        distances=table[:, 2].astype(np.float64),
        left_keypoints=left_count,
        right_keypoints=right_count,
        total_edges=total,
        truncated=truncated,
    )


def match_and_solve_pair(
    left_id: str,
    left_rgb: np.ndarray,
    right_id: str,
    right_rgb: np.ndarray,
    *,
    left_camera: CameraEstimate,
    right_camera: CameraEstimate,
    usable_masks=None,
    settings: CorrespondenceSettings | None = None,
    matcher: JointMatchSettings | None = None,
    gates: PairGates | None = None,
) -> DirectedPairResult:
    """Detect, directed-match and DS-11-solve one pair.

    Extraction is blind: only the two frames, the authoritative usable mask and
    the declared settings are read. The camera estimates are used by the geometric
    stage alone.
    """
    bounds = JointMatchSettings() if matcher is None else matcher
    detection = (
        CorrespondenceSettings(
            max_features=bounds.max_features,
            mask_boundary_pad_px=bounds.mask_boundary_pad_px,
        )
        if settings is None
        else settings
    )
    left_mask = usable_mask_for(left_id, left_rgb.shape, usable_masks)
    right_mask = usable_mask_for(right_id, right_rgb.shape, usable_masks)
    left = detect_features(left_id, left_rgb, usable_mask=left_mask, settings=detection)
    right = detect_features(right_id, right_rgb, usable_mask=right_mask, settings=detection)
    return match_feature_frames(
        left,
        right,
        left_camera,
        right_camera,
        matcher=bounds,
        gates=gates,
        ratio_test=detection.ratio_test,
    )


def match_feature_frames(
    left: FrameFeatures,
    right: FrameFeatures,
    left_camera: CameraEstimate,
    right_camera: CameraEstimate,
    *,
    matcher: JointMatchSettings | None = None,
    gates: PairGates | None = None,
    ratio_test: float = _RATIO_TEST,
) -> DirectedPairResult:
    """Directed-match two already-detected feature frames and solve the pair.

    This is the injectable seam: tests supply :class:`FrameFeatures` directly, so
    the geometric stage is exercised without SIFT. It reads no pixels. The DS-12
    mutual selection is still computed, but only as a reported counter - it never
    gates the directed graph.
    """
    bounds = JointMatchSettings() if matcher is None else matcher
    pair_gates = PairGates() if gates is None else gates
    graph = directed_topk_graph(
        left.descriptors,
        right.descriptors,
        neighbors=bounds.neighbors,
        max_edges=bounds.max_candidate_edges,
        distance_ceiling=bounds.graph_distance_ceiling,
    )
    raw_left, raw_right = len(left.points), len(right.points)
    support_floor = max(
        int(bounds.min_support_absolute),
        int(math.ceil(bounds.min_support_fraction * min(raw_left, raw_right))),
    )
    mutual = select_mutual_matches(
        left.points,
        right.points,
        left.descriptors,
        right.descriptors,
        ratio_test=ratio_test,
    )
    counters = {
        "raw_keypoints": (raw_left, raw_right),
        "boundary_dropped": (int(left.boundary_dropped), int(right.boundary_dropped)),
        "candidate_edges": int(len(graph.edges)),
        "total_graph_edges": int(graph.total_edges),
        "truncated": int(graph.truncated),
        "ratio_survivors": (int(mutual.ratio_left), int(mutual.ratio_right)),
        "mutual_survivors": int(mutual.mutual),
        "support_floor": support_floor,
    }
    empty_matches = np.empty((0, 4), dtype=np.float64)

    if not _cameras_ok(left_camera, right_camera):
        return _refused("invalid_input", left_id=left.frame_id, right_id=right.frame_id,
                        matches=empty_matches, counters=counters, bounds=bounds)
    if len(graph.edges) == 0:
        return _refused("candidate_graph_empty", left_id=left.frame_id, right_id=right.frame_id,
                        matches=empty_matches, counters=counters, bounds=bounds)

    left_rotation = _world_to_camera(left_camera)
    right_rotation = _world_to_camera(right_camera)
    if left_rotation is None or right_rotation is None:
        return _refused("invalid_input", left_id=left.frame_id, right_id=right.frame_id,
                        matches=empty_matches, counters=counters, bounds=bounds)

    left_rays = _camera_rays(left.points, left_camera.intrinsics)
    right_rays = _camera_rays(right.points, right_camera.intrinsics)
    edge_left = graph.edges[:, 0]
    edge_right = graph.edges[:, 1]
    edge_left_rays = left_rays[edge_left]
    edge_right_rays = right_rays[edge_right]
    world_left = edge_left_rays @ left_rotation

    modes: list[_Mode] = []
    for yaw_class in _YAW_CLASSES:
        rotation_right_class = right_rotation @ _ry90(yaw_class)
        relative = rotation_right_class @ left_rotation.T
        rows = np.cross(edge_left_rays @ relative.T, edge_right_rays)
        norms = np.linalg.norm(rows, axis=1)
        world_right = edge_right_rays @ rotation_right_class
        for mode_index, mode in enumerate(
            _class_modes(rows, norms, graph, bounds, support_floor)
        ):
            t, selection, left_support, right_support, residual = mode
            selected = np.flatnonzero(selection)
            world_left_selected = world_left[selection]
            world_right_selected = world_right[selection]
            candidate = -(rotation_right_class.T @ t)
            positive = _positive_depth_fraction(
                world_left_selected, world_right_selected, candidate
            )
            negative = _positive_depth_fraction(
                world_left_selected, world_right_selected, -candidate
            )
            if negative > positive:
                baseline, cheirality = -candidate, negative
            else:
                baseline, cheirality = candidate, positive
            left_points = left.points[edge_left[selection]]
            right_points = right.points[edge_right[selection]]
            bins_left = _bins_covered(left_points, left.width, left.height, bounds.spatial_grid)
            bins_right = _bins_covered(right_points, right.width, right.height, bounds.spatial_grid)
            support_rows = rows[selection]
            singular = _singular_values(support_rows)
            ratio = singular[1] / singular[0] if singular[0] > _TINY else 0.0
            matches = np.column_stack((left_points, right_points))
            modes.append(
                _Mode(
                    yaw_class=yaw_class,
                    mode_index=mode_index,
                    support=int(len(selected)),
                    left_support=left_support,
                    right_support=right_support,
                    matches=matches,
                    selection=selected,
                    spatial_bins_left=bins_left,
                    spatial_bins_right=bins_right,
                    median_selected_residual=float(np.median(residual[selection])),
                    descriptor_cost=float(np.median(graph.distances[selection])),
                    median_all_edge_residual=float(np.median(residual)),
                    parallax=_median_angle_degrees(world_left_selected, world_right_selected),
                    cheirality=cheirality,
                    singular_ratio=float(ratio),
                    baseline_direction=(float(baseline[0]), float(baseline[1]), float(baseline[2])),
                )
            )

    if not modes:
        return _refused("insufficient_one_to_one_support", left_id=left.frame_id,
                        right_id=right.frame_id, matches=empty_matches, counters=counters,
                        bounds=bounds, support_floor=support_floor)

    # The per-left-first selection already resolved right collisions, so the
    # winner's unique-left and unique-right supports are equal by construction;
    # ranking on that resolved support is what keeps a mostly-colliding noise mode
    # (high unique-left, low unique-right) below a genuinely one-to-one mode. The
    # pre-collision supports are the next keys, then spatial spread, selected
    # residual and descriptor cost - never an all-edge median.
    modes.sort(
        key=lambda mode: (
            -mode.support,
            -mode.left_support,
            -mode.right_support,
            -min(mode.spatial_bins_left, mode.spatial_bins_right),
            mode.median_selected_residual,
            mode.descriptor_cost,
            mode.yaw_class,
            mode.mode_index,
        )
    )
    best = modes[0]
    competitor_support = max(
        (mode.support for mode in modes[1:] if mode.support >= support_floor),
        default=None,
    )
    ambiguity_margin = None if competitor_support is None else best.support - competitor_support

    refusal = _winner_refusal(best, modes, support_floor, ambiguity_margin, bounds)
    scores = _score_modes(modes, best, refusal, support_floor, bounds)
    if refusal:
        return _refused(
            refusal,
            left_id=left.frame_id,
            right_id=right.frame_id,
            matches=best.matches,
            counters=counters,
            bounds=bounds,
            modes=modes,
            scores=scores,
            support_floor=support_floor,
            ambiguity_margin=ambiguity_margin,
        )

    pose = solve_pair(
        left_camera,
        right_camera,
        best.matches,
        left_id=left.frame_id,
        right_id=right.frame_id,
        gates=pair_gates,
    )
    if not pose.solved:
        return _refused(
            "solve_pair_refused",
            left_id=left.frame_id,
            right_id=right.frame_id,
            matches=best.matches,
            counters=counters,
            bounds=bounds,
            modes=modes,
            scores=scores,
            support_floor=support_floor,
            ambiguity_margin=ambiguity_margin,
            pose=pose,
        )
    if pose.right_yaw_class != best.yaw_class:
        return _refused(
            "selection_unstable",
            left_id=left.frame_id,
            right_id=right.frame_id,
            matches=best.matches,
            counters=counters,
            bounds=bounds,
            modes=modes,
            scores=scores,
            support_floor=support_floor,
            ambiguity_margin=ambiguity_margin,
            pose=pose,
        )
    return _solved(
        left_id=left.frame_id,
        right_id=right.frame_id,
        pose=pose,
        best=best,
        counters=counters,
        bounds=bounds,
        scores=scores,
        support_floor=support_floor,
        ambiguity_margin=ambiguity_margin,
    )


def _class_modes(
    rows: np.ndarray,
    norms: np.ndarray,
    graph: DirectedCandidateGraph,
    bounds: JointMatchSettings,
    support_floor: int,
) -> list[tuple[np.ndarray, np.ndarray, int, int, np.ndarray]]:
    """Directed candidate modes for one class.

    Deterministic seeded RANSAC draws 2 endpoint-distinct edges, rejects
    rank-deficient samples (DS-14's sampling, reused), solves the sample's unit
    null vector, and scores every candidate edge. The best few angularly distinct
    modes that reach the support floor are then refined by
    :func:`_select_and_refit`, which applies the directed selection.
    """
    edge_count = len(rows)
    if edge_count < 2:
        return []
    left_index = graph.edges[:, 0]
    right_index = graph.edges[:, 1]
    candidates: list[np.ndarray] = []
    all_rows_t = _null_vector(rows)
    if all_rows_t is not None:
        candidates.append(all_rows_t)

    generator = np.random.default_rng(int(bounds.ransac_seed))
    accepted = 0
    attempts = 0
    max_attempts = max(1, int(bounds.ransac_iterations)) * 4
    while accepted < int(bounds.ransac_iterations) and attempts < max_attempts:
        attempts += 1
        sample = generator.choice(edge_count, size=2, replace=False)
        first, second = int(sample[0]), int(sample[1])
        if first == second:
            continue
        if left_index[first] == left_index[second] or right_index[first] == right_index[second]:
            continue
        sample_rows = rows[sample]
        if not _sample_rank_ok(sample_rows, bounds.min_singular_ratio):
            continue
        candidate = _null_vector(sample_rows)
        if candidate is None:
            continue
        candidates.append(candidate)
        accepted += 1

    # Score each proposal the way DS-18 selects, not by its many-to-many edge
    # count: a degenerate baseline near the viewing direction can fit hundreds of
    # false edges, so counting edges inflates exactly the wrong hypotheses. The
    # bound ``min(unique-left, unique-right)`` is the largest one-to-one support
    # the qualified set can yield after the per-left pass and right-collision
    # resolution below.
    scored: list[tuple[int, int, float, int, np.ndarray]] = []
    for order, candidate in enumerate(candidates):
        residual = _residuals(rows, norms, candidate)
        mask = residual <= bounds.inlier_residual_degrees
        if not np.any(mask):
            continue
        seen_left = np.zeros(graph.left_keypoints, dtype=bool)
        seen_right = np.zeros(graph.right_keypoints, dtype=bool)
        seen_left[left_index[mask]] = True
        seen_right[right_index[mask]] = True
        left_support = int(np.count_nonzero(seen_left))
        right_support = int(np.count_nonzero(seen_right))
        bound = min(left_support, right_support)
        if bound < support_floor:
            continue
        scored.append(
            (bound, left_support, float(np.median(residual[mask])), order, candidate)
        )
    scored.sort(key=lambda entry: (-entry[0], -entry[1], entry[2], entry[3]))

    modes: list[np.ndarray] = []
    for bound, _left_support, _median, _order, candidate in scored:
        if bound < support_floor:
            break
        if any(
            _vector_angle_degrees(candidate, kept) < bounds.mode_separation_degrees
            for kept in modes
        ):
            continue
        modes.append(candidate)
        if len(modes) >= int(bounds.max_modes_per_class):
            break

    results: list[tuple[np.ndarray, np.ndarray, int, int, np.ndarray]] = []
    for candidate in modes:
        refined = _select_and_refit(rows, norms, graph, candidate, bounds)
        if refined is None:
            continue
        t, selection, left_support, right_support = refined
        results.append((t, selection, left_support, right_support, _residuals(rows, norms, t)))
    return results


def _select_and_refit(
    rows: np.ndarray,
    norms: np.ndarray,
    graph: DirectedCandidateGraph,
    candidate: np.ndarray,
    bounds: JointMatchSettings,
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    """Directed per-left selection, then SVD refit to a fixed point.

    The refit re-solves ``t`` on the selected support and re-selects, so the
    returned selection, supports and residual all belong to the same ``t``.
    """
    t = candidate
    previous: np.ndarray | None = None
    for _ in range(int(bounds.refit_iterations)):
        selection = _directed_selection(rows, norms, graph, t, bounds.inlier_residual_degrees)
        if selection is None:
            return None
        left_support, right_support, mask = selection
        if previous is not None and np.array_equal(mask, previous):
            break
        previous = mask
        refined = _null_vector(rows[mask])
        if refined is None:
            break
        t = refined
    selection = _directed_selection(rows, norms, graph, t, bounds.inlier_residual_degrees)
    if selection is None:
        return None
    left_support, right_support, mask = selection
    return t, mask, left_support, right_support


def _directed_selection(
    rows: np.ndarray,
    norms: np.ndarray,
    graph: DirectedCandidateGraph,
    t: np.ndarray,
    threshold_degrees: float,
) -> tuple[int, int, np.ndarray] | None:
    """Directed support for ``t``: per-left best edges, then right collisions.

    Step 1 keeps, for every left endpoint, its single best residual-qualified
    edge (ties broken by descriptor distance then right index). Step 2 walks those
    survivors in a deterministic ``(residual, distance, left, right)`` order and
    keeps an edge only when its right endpoint is still unused, which makes the
    final set one-to-one. Right collisions are therefore resolved *after* the
    hypothesis has been scored, never by dropping a left endpoint's evidence
    before the geometry sees it.
    """
    residual = _residuals(rows, norms, t)
    order = np.lexsort((graph.edges[:, 1], graph.edges[:, 0], graph.distances, residual))
    best_left: dict[int, int] = {}
    for position in order:
        if residual[position] > threshold_degrees:
            break
        left = int(graph.edges[position, 0])
        if left not in best_left:
            best_left[left] = int(position)
    if not best_left:
        return None

    survivors = sorted(
        best_left.values(),
        key=lambda position: (
            float(residual[position]),
            float(graph.distances[position]),
            int(graph.edges[position, 0]),
            int(graph.edges[position, 1]),
        ),
    )
    used_right: set[int] = set()
    selected: list[int] = []
    for position in survivors:
        right = int(graph.edges[position, 1])
        if right in used_right:
            continue
        used_right.add(right)
        selected.append(position)
    if not selected:
        return None
    mask = np.zeros(len(rows), dtype=bool)
    mask[np.asarray(selected, dtype=np.int64)] = True
    left_support = len(best_left)
    right_support = len({int(graph.edges[position, 1]) for position in best_left.values()})
    return left_support, right_support, mask


def _null_vector(rows: np.ndarray) -> np.ndarray | None:
    """Unit null vector of the epipolar rows, without a quadratic allocation.

    DS-14 keeps ``np.linalg.svd(rows, full_matrices=True)``'s ``vt[-1]``, which
    materialises the ``M x M`` left factor - the exact ``8 * M**2`` allocation
    DS-16B had to cap at 4096 rows. The smallest right-singular vector of ``A`` is
    the eigenvector of the ``3 x 3`` Gram matrix ``A^T A`` for its smallest
    eigenvalue, so the same unit direction is available in constant memory. The
    direction is defined up to sign, and every consumer here uses it only through
    ``|a^T t|`` residuals and an absolute-dot mode comparison.
    """
    if rows.size == 0:
        return None
    gram = rows.T @ rows
    if not np.all(np.isfinite(gram)):
        return None
    values, vectors = np.linalg.eigh(gram)
    vector = vectors[:, int(np.argmin(values))]
    norm = float(np.linalg.norm(vector))
    if norm <= _TINY:
        return None
    return vector / norm


def _score_modes(
    modes: list[_Mode],
    best: _Mode,
    winner_refusal: str,
    support_floor: int,
    bounds: JointMatchSettings,
) -> tuple[DirectedHypothesisScore, ...]:
    """Public per-hypothesis diagnostics; only the winner can carry ``passed``."""
    scores: list[DirectedHypothesisScore] = []
    for mode in modes:
        local = _winner_refusal(mode, modes, support_floor, None, bounds)
        is_best = mode is best
        refusal = winner_refusal if is_best else local
        scores.append(
            DirectedHypothesisScore(
                yaw_class=mode.yaw_class,
                mode_index=mode.mode_index,
                support=mode.support,
                left_support=mode.left_support,
                right_support=mode.right_support,
                spatial_bins_left=mode.spatial_bins_left,
                spatial_bins_right=mode.spatial_bins_right,
                spatial_bins=min(mode.spatial_bins_left, mode.spatial_bins_right),
                median_selected_residual_degrees=mode.median_selected_residual,
                descriptor_cost=mode.descriptor_cost,
                median_all_edge_residual_degrees=mode.median_all_edge_residual,
                parallax_degrees=mode.parallax,
                cheirality_fraction=mode.cheirality,
                singular_ratio=mode.singular_ratio,
                baseline_direction=mode.baseline_direction,
                passed=(refusal == "") and is_best,
                refusal=refusal,
            )
        )
    return tuple(scores)


def _solved(
    *,
    left_id: str,
    right_id: str,
    pose: PairPose,
    best: _Mode,
    counters: dict,
    bounds: JointMatchSettings,
    scores: tuple[DirectedHypothesisScore, ...],
    support_floor: int,
    ambiguity_margin: int | None,
) -> DirectedPairResult:
    return DirectedPairResult(
        left_id=left_id,
        right_id=right_id,
        status="solved",
        reason="",
        pose=pose,
        right_yaw_class=pose.right_yaw_class,
        matches=best.matches,
        raw_keypoints=counters["raw_keypoints"],
        boundary_dropped=counters["boundary_dropped"],
        candidate_edges=counters["candidate_edges"],
        total_graph_edges=counters["total_graph_edges"],
        truncated=counters["truncated"],
        ratio_survivors=counters["ratio_survivors"],
        mutual_survivors=counters["mutual_survivors"],
        support=best.support,
        left_support=best.left_support,
        right_support=best.right_support,
        support_floor=support_floor,
        spatial_bins_left=best.spatial_bins_left,
        spatial_bins_right=best.spatial_bins_right,
        spatial_bins_total=int(bounds.spatial_grid[0] * bounds.spatial_grid[1]),
        median_selected_residual_degrees=best.median_selected_residual,
        descriptor_cost=best.descriptor_cost,
        parallax_degrees=best.parallax,
        cheirality_fraction=best.cheirality,
        singular_ratio=best.singular_ratio,
        ambiguity_margin=ambiguity_margin,
        hypotheses=scores,
    )


def _refused(
    reason: str,
    *,
    left_id: str,
    right_id: str,
    matches: np.ndarray,
    counters: dict,
    bounds: JointMatchSettings,
    modes: list[_Mode] | None = None,
    scores: tuple[DirectedHypothesisScore, ...] = (),
    support_floor: int = 0,
    ambiguity_margin: int | None = None,
    pose: PairPose | None = None,
) -> DirectedPairResult:
    best = None if not modes else modes[0]
    return DirectedPairResult(
        left_id=left_id,
        right_id=right_id,
        status="refused",
        reason=reason,
        pose=pose,
        right_yaw_class=None,
        matches=matches,
        raw_keypoints=counters["raw_keypoints"],
        boundary_dropped=counters["boundary_dropped"],
        candidate_edges=counters["candidate_edges"],
        total_graph_edges=counters["total_graph_edges"],
        truncated=counters["truncated"],
        ratio_survivors=counters["ratio_survivors"],
        mutual_survivors=counters["mutual_survivors"],
        support=0 if best is None else best.support,
        left_support=0 if best is None else best.left_support,
        right_support=0 if best is None else best.right_support,
        support_floor=support_floor,
        spatial_bins_left=0 if best is None else best.spatial_bins_left,
        spatial_bins_right=0 if best is None else best.spatial_bins_right,
        spatial_bins_total=int(bounds.spatial_grid[0] * bounds.spatial_grid[1]),
        median_selected_residual_degrees=(
            None if best is None else best.median_selected_residual
        ),
        descriptor_cost=None if best is None else best.descriptor_cost,
        parallax_degrees=None if best is None else best.parallax,
        cheirality_fraction=None if best is None else best.cheirality,
        singular_ratio=None if best is None else best.singular_ratio,
        ambiguity_margin=ambiguity_margin,
        hypotheses=scores,
    )


def _empty_graph(left_count: int, right_count: int) -> DirectedCandidateGraph:
    return DirectedCandidateGraph(
        edges=np.empty((0, 2), dtype=np.int64),
        distances=np.empty(0, dtype=np.float64),
        left_keypoints=left_count,
        right_keypoints=right_count,
        total_edges=0,
        truncated=0,
    )
