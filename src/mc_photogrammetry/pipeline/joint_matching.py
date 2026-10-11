"""Repetition-aware joint correspondence/pose matcher for one camera pair (DS-14).

DS-12 could extract a clean mutual correspondence set but refused five of six real
pairs because the ratio test and 1-to-1 reciprocity discard the very edges that
carry the geometry. DS-13 designed an alternative and Astra corrected its
objective before implementation (DS-13 section 9). This module is that corrected
package:

* build the **reciprocal top-k descriptor graph** (default ``k = 5``, hard cap on
  edges, truncation reported);
* adjudicate it by **geometry only**: for each relative yaw class ``k in 0..3``
  draw deterministic seeded RANSAC samples of **2 endpoint-distinct edges**,
  reject rank-deficient samples, solve the unit baseline ``t``, and score every
  candidate edge by its epipolar residual;
* deterministically **greedily select one-to-one pixel-endpoint support** sorted
  by residual, descriptor distance and indices, then refit ``t`` on that support
  to a fixed point; SIFT orientation rows at the same pixel are alternatives,
  not independent geometric support;
* **rank hypotheses by support count first** (then spatial bins, median selected
  residual, descriptor cost) - never by an all-edge median, which the top-k graph
  is built to make meaningless (its rows are mostly false);
* require the support floor ``max(8, ceil(0.02 * min(unique pixel endpoints)))``, spatial
  spread, parallax, cheirality and conditioning, and a declared margin over the
  best distinct competing class **or baseline**; otherwise abstain with a typed
  reason;
* hand the selected set to DS-11's public :func:`solve_pair` as the compatibility
  check and report **its** pose, requiring the same yaw class.

The extraction half stays blind: :func:`match_and_solve_pair` detects features
through :mod:`mc_photogrammetry.pipeline.correspondence` using pixels and the
authoritative usable mask only. Camera estimates are consumed solely for the
geometric stage (this module's own ``t`` and DS-11's public solver); no scene,
lattice, manifest, coordinate or asserted geometry is reachable from here, and
the matcher's typed reasons are module-local - they are never appended to
``connected_camera.REFUSAL_REASONS``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mc_photogrammetry.pipeline.connected_camera import PairGates, PairPose, solve_pair
from mc_photogrammetry.pipeline.contracts import CameraEstimate, CameraIntrinsics
from mc_photogrammetry.pipeline.correspondence import (
    CorrespondenceSettings,
    FrameFeatures,
    ReciprocalGraph,
    detect_features,
    reciprocal_topk_graph,
    select_mutual_matches,
    usable_mask_for,
)
from mc_photogrammetry.reconstruction.camera import CameraModel, camera_rotation_matrix

__all__ = [
    "JOINT_REFUSAL_REASONS",
    "HypothesisScore",
    "JointMatchSettings",
    "JointPairResult",
    "match_and_solve_pair",
    "match_feature_frames",
]

_TINY = 1e-12
_ZERO_ROW = 1e-9
_YAW_CLASSES: tuple[int, ...] = (0, 1, 2, 3)
_RATIO_TEST = 0.75

JOINT_REFUSAL_REASONS: tuple[str, ...] = (
    "invalid_input",
    "candidate_graph_empty",
    "rank_deficient",
    "insufficient_one_to_one_support",
    "low_spatial_spread",
    "low_parallax",
    "cheirality_failed",
    "ambiguous_hypothesis",
    "solve_pair_refused",
    "selection_unstable",
)
"""Every typed reason :class:`JointPairResult` may report for a refusal.

These are deliberately *not* DS-11's ``REFUSAL_REASONS``: this stage can fail for
reasons (graph support, ambiguity between competing hypotheses) that the public
pair solver cannot express, and it never invents a DS-11 token.
"""


@dataclass(frozen=True, slots=True)
class JointMatchSettings:
    """Declared bounds for one joint match; defaults are fixed, not tuned per pair."""

    neighbors: int = 5
    max_candidate_edges: int = 8192
    max_features: int = 4000
    mask_boundary_pad_px: float = 8.0
    graph_distance_ceiling: float | None = None
    ransac_seed: int = 0
    ransac_iterations: int = 512
    inlier_residual_degrees: float = 1.0
    refit_iterations: int = 10
    min_support_absolute: int = 8
    min_support_fraction: float = 0.02
    spatial_grid: tuple[int, int] = (4, 4)
    min_spatial_bins: int = 4
    min_parallax_degrees: float = 3.0
    min_cheirality_fraction: float = 0.75
    min_singular_ratio: float = 1e-3
    ambiguity_support_margin: int = 2
    mode_separation_degrees: float = 15.0
    max_modes_per_class: int = 3

    def __post_init__(self) -> None:
        if self.neighbors < 1:
            raise ValueError("neighbors must be positive")
        if self.max_candidate_edges < 1:
            raise ValueError("max_candidate_edges must be positive")
        if self.max_features < 1:
            raise ValueError("max_features must be positive")
        if self.mask_boundary_pad_px < 0.0:
            raise ValueError("mask_boundary_pad_px must be non-negative")
        if self.graph_distance_ceiling is not None and self.graph_distance_ceiling < 0.0:
            raise ValueError("graph_distance_ceiling must be non-negative")
        if self.ransac_iterations < 0:
            raise ValueError("ransac_iterations must be non-negative")
        if not 0.0 < self.inlier_residual_degrees < 90.0:
            raise ValueError("inlier_residual_degrees must lie in (0, 90)")
        if self.refit_iterations < 1:
            raise ValueError("refit_iterations must be positive")
        if self.min_support_absolute < 1:
            raise ValueError("min_support_absolute must be positive")
        if not 0.0 <= self.min_support_fraction <= 1.0:
            raise ValueError("min_support_fraction must lie in [0, 1]")
        grid_x, grid_y = self.spatial_grid
        if grid_x < 1 or grid_y < 1:
            raise ValueError("spatial_grid dimensions must be positive")
        if not 0 <= self.min_spatial_bins <= grid_x * grid_y:
            raise ValueError("min_spatial_bins must lie in [0, grid_x * grid_y]")
        if self.min_parallax_degrees < 0.0:
            raise ValueError("min_parallax_degrees must be non-negative")
        if not 0.0 <= self.min_cheirality_fraction <= 1.0:
            raise ValueError("min_cheirality_fraction must lie in [0, 1]")
        if self.min_singular_ratio < 0.0:
            raise ValueError("min_singular_ratio must be non-negative")
        if self.ambiguity_support_margin < 1:
            raise ValueError("ambiguity_support_margin must be positive")
        if not 0.0 <= self.mode_separation_degrees <= 180.0:
            raise ValueError("mode_separation_degrees must lie in [0, 180]")
        if self.max_modes_per_class < 1:
            raise ValueError("max_modes_per_class must be positive")


@dataclass(frozen=True, slots=True)
class HypothesisScore:
    """Diagnostics for one (yaw class, baseline mode) hypothesis.

    ``median_all_edge_residual_degrees`` is reported for transparency only. It is
    deliberately **not** a gate: the reciprocal top-k graph is mostly false edges,
    so an all-edge median measures the ambiguity mass, not the hypothesis.
    """

    yaw_class: int
    mode_index: int
    support: int
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
class JointPairResult:
    """One pair's joint result: a DS-11 pose on the selected set, or a refusal.

    On a solve, ``pose`` is the *public DS-11* :class:`PairPose` produced by
    :func:`~mc_photogrammetry.pipeline.connected_camera.solve_pair` on
    ``matches`` and the winner's yaw class equals ``pose.right_yaw_class``. On a
    refusal every vector field is ``None``; ``matches`` still carries the best
    selected one-to-one set (possibly empty) so the refusal is auditable, and
    ``hypotheses`` carries every candidate the matcher ranked.
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
    hypotheses: tuple[HypothesisScore, ...]

    @property
    def solved(self) -> bool:
        return self.pose is not None and bool(self.pose.solved)


@dataclass(frozen=True, slots=True)
class _Mode:
    """One accepted (yaw class, baseline) hypothesis before public formatting."""

    yaw_class: int
    mode_index: int
    support: int
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
) -> JointPairResult:
    """Detect, jointly match and DS-11-solve one pair.

    Extraction is blind: only the two frames, the authoritative usable mask and
    the declared settings are read. The camera estimates are used by the
    geometric stage alone, exactly as the diagnostic's phase 1 holds them.
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
) -> JointPairResult:
    """Jointly match two already-detected feature frames and solve the pair.

    This is the injectable seam: tests (and any future caller that already has
    feature tables) supply :class:`FrameFeatures` directly, so the geometric
    stage is exercised without SIFT. It reads no pixels.
    """
    bounds = JointMatchSettings() if matcher is None else matcher
    pair_gates = PairGates() if gates is None else gates
    graph = reciprocal_topk_graph(
        left.descriptors,
        right.descriptors,
        neighbors=bounds.neighbors,
        max_edges=bounds.max_candidate_edges,
        distance_ceiling=bounds.graph_distance_ceiling,
    )
    raw_left, raw_right = len(left.points), len(right.points)
    endpoint_ids = (_pixel_endpoint_ids(left.points), _pixel_endpoint_ids(right.points))
    physical_counts = tuple(len(np.unique(ids)) for ids in endpoint_ids)
    support_floor = max(
        int(bounds.min_support_absolute),
        int(math.ceil(bounds.min_support_fraction * min(physical_counts))),
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
        for mode_index, (t, selection, residual) in enumerate(
            _class_modes(rows, norms, graph, bounds, support_floor, endpoint_ids=endpoint_ids)
        ):
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

    # Support first, then spatial bins, selected residual, descriptor cost, class.
    modes.sort(
        key=lambda mode: (
            -mode.support,
            -min(mode.spatial_bins_left, mode.spatial_bins_right),
            mode.median_selected_residual,
            mode.descriptor_cost,
            mode.yaw_class,
            mode.mode_index,
        )
    )
    best = modes[0]
    competitor_support = max(
        (mode.support for mode in modes[1:] if mode.support >= support_floor), default=None
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
    graph: ReciprocalGraph,
    bounds: JointMatchSettings,
    support_floor: int,
    *,
    endpoint_ids: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Candidate ``(t, boolean selection, residual degrees)`` modes for one class.

    Deterministic seeded RANSAC draws 2 endpoint-distinct edges, rejects
    rank-deficient samples, solves the sample's unit null vector, and scores every
    candidate edge. Shortlist support and residual medians use the same greedy
    one-to-one physical endpoints as final selection, so descriptor alternatives
    cannot inflate a mode's rank. The best few angularly distinct modes that
    reach the support floor are refit to a fixed point. Hypothesis generation
    remains bounded and may still depend on descriptor-alternative multiplicity.
    """
    edge_count = len(rows)
    if edge_count < 2:
        return []
    left_index = graph.edges[:, 0]
    right_index = graph.edges[:, 1]
    if endpoint_ids is not None:
        left_index = endpoint_ids[0][left_index]
        right_index = endpoint_ids[1][right_index]
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

    scored: list[tuple[int, float, int, np.ndarray]] = []
    for order, candidate in enumerate(candidates):
        residual = _residuals(rows, norms, candidate)
        mask = _greedy_selection(rows, norms, graph, candidate,
                                 bounds.inlier_residual_degrees, endpoint_ids=endpoint_ids)
        if mask is None:
            continue
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        scored.append((count, float(np.median(residual[mask])), order, candidate))
    scored.sort(key=lambda entry: (-entry[0], entry[1], entry[2]))

    modes: list[np.ndarray] = []
    for count, _median, _order, candidate in scored:
        if count < support_floor:
            break
        if any(
            _vector_angle_degrees(candidate, kept) < bounds.mode_separation_degrees
            for kept in modes
        ):
            continue
        modes.append(candidate)
        if len(modes) >= int(bounds.max_modes_per_class):
            break

    results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for candidate in modes:
        refined = _select_and_refit(rows, norms, graph, candidate, bounds, endpoint_ids=endpoint_ids)
        if refined is None:
            continue
        t, selection = refined
        results.append((t, selection, _residuals(rows, norms, t)))
    return results


def _select_and_refit(
    rows: np.ndarray,
    norms: np.ndarray,
    graph: ReciprocalGraph,
    candidate: np.ndarray,
    bounds: JointMatchSettings,
    *,
    endpoint_ids: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Greedy one-to-one support, then SVD refit to a fixed point.

    The selection is the only place the one-to-one assignment is enforced: edges
    are ordered by ``(residual, descriptor distance, left index, right index)`` and
    an edge is kept only when neither physical pixel endpoint is already used.
    Multiple descriptor orientation rows remain competing edge alternatives.
    The refit
    re-solves ``t`` on the selected support and re-selects, so the returned
    selection and residual belong to the same ``t``.
    """
    t = candidate
    previous: np.ndarray | None = None
    for _ in range(int(bounds.refit_iterations)):
        selection = _greedy_selection(rows, norms, graph, t, bounds.inlier_residual_degrees,
                                      endpoint_ids=endpoint_ids)
        if selection is None:
            return None
        if previous is not None and np.array_equal(selection, previous):
            break
        previous = selection
        refined = _null_vector(rows[selection])
        if refined is None:
            break
        t = refined
    selection = _greedy_selection(rows, norms, graph, t, bounds.inlier_residual_degrees,
                                  endpoint_ids=endpoint_ids)
    if selection is None:
        return None
    return t, selection


def _greedy_selection(
    rows: np.ndarray,
    norms: np.ndarray,
    graph: ReciprocalGraph,
    t: np.ndarray,
    threshold_degrees: float,
    *,
    endpoint_ids: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray | None:
    """Boolean one-to-one support for ``t``; ``None`` when nothing survives."""
    residual = _residuals(rows, norms, t)
    order = np.lexsort(
        (graph.edges[:, 1], graph.edges[:, 0], graph.distances, residual)
    )
    used_left: set[int] = set()
    used_right: set[int] = set()
    selected: list[int] = []
    for position in order:
        if residual[position] > threshold_degrees:
            break
        left = int(graph.edges[position, 0])
        right = int(graph.edges[position, 1])
        if endpoint_ids is not None:
            left = int(endpoint_ids[0][left])
            right = int(endpoint_ids[1][right])
        if left in used_left or right in used_right:
            continue
        used_left.add(left)
        used_right.add(right)
        selected.append(int(position))
    if not selected:
        return None
    mask = np.zeros(len(rows), dtype=bool)
    mask[np.asarray(selected, dtype=np.int64)] = True
    return mask


def _pixel_endpoint_ids(points: np.ndarray) -> np.ndarray:
    """First row identity for each exact observed pixel; no proximity rounding.

    Unique-coordinate tables retain their original identities and tie ordering.
    Orientation/scale descriptors stay in the reciprocal graph, but coincident
    endpoints cannot multiply a minimal sample, selected support or SVD refit.
    """
    _unique, first, inverse = np.unique(points, axis=0, return_index=True, return_inverse=True)
    return first[inverse]


def _winner_refusal(
    best: _Mode,
    modes: list[_Mode],
    support_floor: int,
    ambiguity_margin: int | None,
    bounds: JointMatchSettings,
) -> str:
    """First gate the ranked winner fails, as a typed module-local token."""
    if best.support < support_floor:
        return "insufficient_one_to_one_support"
    if min(best.spatial_bins_left, best.spatial_bins_right) < bounds.min_spatial_bins:
        return "low_spatial_spread"
    if best.singular_ratio is None or best.singular_ratio < bounds.min_singular_ratio:
        return "rank_deficient"
    if best.parallax is None or best.parallax < bounds.min_parallax_degrees:
        return "low_parallax"
    if best.cheirality is None or best.cheirality < bounds.min_cheirality_fraction:
        return "cheirality_failed"
    if ambiguity_margin is not None and ambiguity_margin < bounds.ambiguity_support_margin:
        return "ambiguous_hypothesis"
    return ""


def _score_modes(
    modes: list[_Mode],
    best: _Mode,
    winner_refusal: str,
    support_floor: int,
    bounds: JointMatchSettings,
) -> tuple[HypothesisScore, ...]:
    """Public per-hypothesis diagnostics; only the winner can carry ``passed``."""
    scores: list[HypothesisScore] = []
    for mode in modes:
        local = _winner_refusal(mode, modes, support_floor, None, bounds)
        is_best = mode is best
        refusal = winner_refusal if is_best else local
        scores.append(
            HypothesisScore(
                yaw_class=mode.yaw_class,
                mode_index=mode.mode_index,
                support=mode.support,
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
    scores: tuple[HypothesisScore, ...],
    support_floor: int,
    ambiguity_margin: int | None,
) -> JointPairResult:
    return JointPairResult(
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
    scores: tuple[HypothesisScore, ...] = (),
    support_floor: int = 0,
    ambiguity_margin: int | None = None,
    pose: PairPose | None = None,
) -> JointPairResult:
    best = None if not modes else modes[0]
    return JointPairResult(
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


def _ry90(yaw_class: int) -> np.ndarray:
    """``Ry(90 * yaw_class)``: the world relabelling for one quadrant."""
    angle = math.radians(90.0 * (yaw_class % 4))
    cos, sin = math.cos(angle), math.sin(angle)
    return np.asarray(
        ((cos, 0.0, sin), (0.0, 1.0, 0.0), (-sin, 0.0, cos)), dtype=np.float64
    )


def _world_to_camera(camera: CameraEstimate) -> np.ndarray | None:
    pose = camera.pose
    if pose is None:
        return None
    values = (pose.yaw_degrees, pose.pitch_degrees, pose.roll_degrees)
    if not all(math.isfinite(float(value)) for value in values):
        return None
    try:
        return camera_rotation_matrix(
            CameraModel(
                position=(0.0, 0.0, 0.0),
                yaw_degrees=float(pose.yaw_degrees),
                pitch_degrees=float(pose.pitch_degrees),
                roll_degrees=float(pose.roll_degrees),
                eye_height=0.0,
            )
        )
    except ValueError:
        return None


def _cameras_ok(left: CameraEstimate, right: CameraEstimate) -> bool:
    if left is None or right is None:
        return False
    if left.pose is None or right.pose is None:
        return False
    if left.intrinsics is None or right.intrinsics is None:
        return False
    return _intrinsics_ok(left.intrinsics) and _intrinsics_ok(right.intrinsics)


def _intrinsics_ok(intrinsics: CameraIntrinsics) -> bool:
    focal_x, focal_y = intrinsics.focal_length_pixels
    centre_x, centre_y = intrinsics.principal_point_pixels
    width, height = intrinsics.image_size
    if not all(math.isfinite(float(value)) for value in (focal_x, focal_y, centre_x, centre_y)):
        return False
    return focal_x > 0.0 and focal_y > 0.0 and width > 0 and height > 0


def _camera_rays(pixels: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Unit image-up camera rays; mirrors DS-11's convention exactly."""
    focal_x, focal_y = intrinsics.focal_length_pixels
    centre_x, centre_y = intrinsics.principal_point_pixels
    x = (pixels[:, 0] - centre_x) / focal_x
    y = -(pixels[:, 1] - centre_y) / focal_y
    rays = np.column_stack((x, y, np.ones(len(pixels), dtype=np.float64)))
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def _residuals(rows: np.ndarray, norms: np.ndarray, t: np.ndarray) -> np.ndarray:
    """``|a^T t| / ||a||`` in degrees; a degenerate zero row scores worst (90)."""
    projections = np.abs(rows @ t)
    values = np.ones(len(rows), dtype=np.float64)
    safe = norms > _TINY
    values[safe] = np.clip(projections[safe] / norms[safe], 0.0, 1.0)
    return np.degrees(np.arcsin(values))


def _null_vector(rows: np.ndarray) -> np.ndarray | None:
    """Unit right-singular vector of the smallest singular value (the null space)."""
    if rows.size == 0:
        return None
    _u, _s, vt = np.linalg.svd(rows, full_matrices=True)
    vector = vt[-1]
    norm = float(np.linalg.norm(vector))
    if norm <= _TINY:
        return None
    return vector / norm


def _sample_rank_ok(rows: np.ndarray, min_singular_ratio: float) -> bool:
    """A 2-row minimal sample must have two independent epipolar rows."""
    if rows.shape[0] < 2:
        return False
    singular = np.linalg.svd(rows, full_matrices=False)[1]
    if singular[0] <= _TINY:
        return False
    return float(singular[1] / singular[0]) >= min_singular_ratio


def _singular_values(rows: np.ndarray) -> tuple[float, float, float]:
    if rows.size == 0:
        return (0.0, 0.0, 0.0)
    _u, singular, _vt = np.linalg.svd(rows, full_matrices=True)
    padded = np.zeros(3, dtype=np.float64)
    padded[: min(3, len(singular))] = singular[:3]
    return (float(padded[0]), float(padded[1]), float(padded[2]))


def _median_angle_degrees(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) == 0:
        return None
    dots = np.clip(np.sum(first * second, axis=1), -1.0, 1.0)
    return float(np.degrees(np.median(np.arccos(dots))))


def _vector_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    dot = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return float(np.degrees(np.arccos(abs(dot))))


def _positive_depth_fraction(
    world_left: np.ndarray, world_right: np.ndarray, baseline: np.ndarray
) -> float:
    """Fraction of selected rays triangulating in front of both views.

    Mirrors DS-11's unit-baseline normal-equations solve; a wrong baseline sign
    flips both depths, so the fraction separates the two signs.
    """
    if len(world_left) == 0:
        return 0.0
    cosine = np.clip(np.sum(world_left * world_right, axis=1), -1.0, 1.0)
    determinant = 1.0 - cosine * cosine
    safe = determinant > 1e-12
    if not np.any(safe):
        return 0.0
    first = world_left @ baseline
    second = -(world_right @ baseline)
    lambda_left = (first[safe] + cosine[safe] * second[safe]) / determinant[safe]
    lambda_right = (second[safe] + cosine[safe] * first[safe]) / determinant[safe]
    positive = int(np.count_nonzero((lambda_left > 0.0) & (lambda_right > 0.0)))
    return positive / float(len(world_left))


def _bins_covered(
    points: np.ndarray, width: int, height: int, grid: tuple[int, int]
) -> int:
    """Distinct spatial bins holding at least one selected endpoint."""
    if points.size == 0:
        return 0
    grid_x, grid_y = grid
    column = np.clip((points[:, 0] / max(width, 1) * grid_x).astype(int), 0, grid_x - 1)
    row = np.clip((points[:, 1] / max(height, 1) * grid_y).astype(int), 0, grid_y - 1)
    return int(np.unique(column * grid_y + row).size)
