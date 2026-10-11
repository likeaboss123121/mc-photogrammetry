"""Evaluation-only truth-filtered oracle candidate-graph upper bound (DS-R5K).

DS-R5J attributed the measured DS-R5J loss to the stage between the reciprocal
top-k candidate graph and the joint selection: in Canal Town the candidate graph
reached 345/1,215 truth pairs while the joint selector kept only 52. DS-R5K asks a
single, deliberately oracle-assisted counterfactual question: **if the candidate
graph contained only strict same-world-surface keypoint pairs, would the unchanged
production geometric selector solve?**

For every unordered frame pair of one canonical Canal Town export the diagnostic
runs the unchanged production
:func:`mc_photogrammetry.pipeline.joint_matching.match_feature_frames` twice:

1. **baseline** - the production reciprocal descriptor graph
   (:func:`reciprocal_topk_graph`), exactly as the matcher computes it;
2. **oracle** - a graph whose edges are exactly the strict same-world-surface
   keypoint pixel pairs of the frozen DS-R5I truth rule, with the **actual L2
   descriptor distances** of the mapped keypoint pair preserved and injected only
   for the duration of that one matcher call.

Camera poses and ground truth are evaluation-only oracle inputs, and the oracle
graph is an explicit truth-assisted upper bound. It is **not** a production result,
not a model-performance estimate, and makes no generalization claim. Normal
reconstruction remains image-only.

Per pair the sealed record reports, using the corrected unique-pixel-pair scorer
from :mod:`mc_photogrammetry.evaluation_pair_correspondence`:

* the raw mutual-SIFT baseline (context);
* the baseline reciprocal candidate graph: edge count, correspondence reach and
  unique-pixel-pair precision;
* the oracle candidate graph: edge count, correspondence reach and precision;
* the selected matches of both runs: precision/recall/reach;
* status and typed refusal reason, selection/pose state, support/spatial/
  cheirality/parallax/singular gates and every ranked hypothesis for both runs;
* whether the oracle candidate set made the existing geometric selector solve.

Graph injection restores the production ``reciprocal_topk_graph`` in a ``finally``
block, so nothing leaks into the interpreter or into production behaviour.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from mc_photogrammetry.evaluation_pair_correspondence import (
    SURFACE_TOLERANCE_BLOCKS,
    DatasetSceneInput,
    FinalizedScene,
    FrameInput,
    PairGeometry,
    ScoreTotals,
    SurfaceHit,
    build_group_manifest,
    build_scene,
    check_fold_leakage,
    check_split_leakage,
    construct_truth_pairs,
    discover_finalized_entries,
    geometry_policy_record,
    match_indices,
    metric_value,
    raycast_surface,
    score_pair_matches,
    seal_record,
)
from mc_photogrammetry.pipeline.connected_camera import PairGates
from mc_photogrammetry.pipeline.contracts import (
    CameraEstimate,
    CameraPose,
)
from mc_photogrammetry.pipeline.correspondence import (
    CorrespondenceSettings,
    FrameFeatures,
    ReciprocalGraph,
    detect_features,
    reciprocal_topk_graph,
    select_mutual_matches,
    usable_mask_for,
)
from mc_photogrammetry.pipeline.joint_matching import (
    HypothesisScore,
    JointMatchSettings,
    JointPairResult,
    match_feature_frames,
)
from mc_photogrammetry.reconstruction.camera import intrinsics_for

__all__ = [
    "CAMERA_POLICY_MODE",
    "CANONICAL_SCENE_PREFIX",
    "DEFAULT_EXPORTS_ROOT",
    "DEFAULT_OUTPUT",
    "LIMITATIONS",
    "ORACLE_GRAPH_MODE",
    "SCHEMA",
    "OracleCandidateGraphError",
    "OracleFrameObservation",
    "OraclePairAttribution",
    "attribute_pair",
    "build_oracle_graph",
    "build_payload",
    "camera_record",
    "canonical_entry_record",
    "describe_outcome",
    "detection_settings",
    "frame_record",
    "geometry_record",
    "hypothesis_record",
    "injected_candidate_graph",
    "main",
    "observe_oracle_frame",
    "oracle_camera_estimate",
    "pair_record",
    "parse_args",
    "result_record",
    "select_canonical_entry",
    "summarise_attributions",
]

SCHEMA = "mc-photogrammetry-oracle-candidate-graph-upper-bound-v1"
"""Sealed schema of this evaluation-only truth-assisted upper bound."""

DEFAULT_EXPORTS_ROOT = "exports/datasets"
DEFAULT_OUTPUT = ".tmp-oracle-candidate-graph-20261003.json"

CAMERA_POLICY_MODE = "oracle_factory_capture_metadata"
"""The only camera source this diagnostic uses, and it is assisted."""

ORACLE_GRAPH_MODE = "oracle_truth_same_surface_pairs"
"""The injected graph is the strict same-world-surface truth pair set."""

CANONICAL_SCENE_PREFIX = "canal-town"
"""Scene-id prefix identifying the Canal Town topology group."""

_RATIO_TEST = CorrespondenceSettings().ratio_test
"""The production detection ratio test; ``JointMatchSettings`` has no copy."""

SELECTION_SOLVED = "selected_solved"
SELECTION_ON_REFUSAL = "selected_on_refusal"
SELECTION_ABSENT = "absent_selection"
POSE_SOLVED = "pose_solved"
POSE_ABSTAINED = "pose_abstained"
POSE_CLASS_MISMATCH = "pose_class_mismatch"
POSE_NOT_ATTEMPTED = "pose_not_attempted"

GEOMETRY_STAGES = (
    "mutual_stage",
    "baseline_candidate_stage",
    "oracle_candidate_stage",
    "baseline_selected_stage",
    "oracle_selected_stage",
)

LIMITATIONS = (
    "The oracle candidate graph is a deliberate truth-assisted UPPER BOUND, not a "
    "production result and not a model-performance estimate. It uses the "
    "registered ground-truth surfaces and the recorded factory capture cameras, "
    "which normal reconstruction never sees.",
    "The factory capture metadata (recorded pose, field of view and frame "
    "resolution) is an explicit oracle-camera evaluation aid. This assisted "
    "diagnostic is not the default image-only reconstruction path and makes no "
    "claim about image-derived calibration.",
    "Only the single canonical Canal Town export is evaluated; one scene export, "
    "one run. Only three independent topology groups exist and every index split "
    "is 'train', so no generalization claim is made.",
    "Pair hypotheses are not truth correspondences; precision/recall are scored "
    "against an evaluator-built truth model, not against independent labels.",
    "Partial block models (bottom/top slabs, stairs, trapdoors, fences, walls, "
    "panes, plants, fluids) are unavailable without the vanilla model assets; "
    "they occlude as full cubes but are excluded from truth ownership.",
    "No texture identity is available anywhere in the pipeline: the block texture "
    "is never read, hashed or compared. same_state_same_slot_other_block is only "
    "an appearance proxy and is not texture ground truth.",
    "Truth pairs and every returned identity are deduplicated by unique pixel "
    "pair; repeated descriptor keypoints at exactly the same pixel are one "
    "identity, and distinct pixels are never collapsed.",
    "The oracle edges preserve the actual detected descriptor distances, but the "
    "descriptor graph itself is replaced, so candidate reach under the oracle is "
    "an upper bound on what the descriptor stage could ever supply.",
    "No matcher, threshold, gate, solver, production, project or promotion setting "
    "was changed; the production reciprocal graph builder is restored after every "
    "oracle call.",
)


class OracleCandidateGraphError(ValueError):
    """Invalid, unverifiable or inconsistent evaluation input."""


Pixel = tuple[float, float]
"""Exact keypoint pixel, the evaluation-only identity of one observed ray."""


@dataclass(frozen=True, slots=True)
class OracleFrameObservation:
    """One frame's joint features plus the R5I truth surface per keypoint."""

    frame_id: str
    width: int
    height: int
    usable_pixels: int
    boundary_dropped: int
    features: FrameFeatures
    hits: tuple[SurfaceHit | None, ...]
    no_truth_reasons: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class OraclePairAttribution:
    """One unordered frame pair's baseline-versus-oracle stage ledger."""

    left_id: str
    right_id: str
    baseline_graph: ReciprocalGraph
    oracle_graph: ReciprocalGraph
    baseline_result: JointPairResult
    oracle_result: JointPairResult
    mutual: PairGeometry
    baseline_candidate: PairGeometry
    oracle_candidate: PairGeometry
    baseline_selected: PairGeometry
    oracle_selected: PairGeometry

    @property
    def truth_pairs(self) -> int:
        return self.baseline_selected.truth_pairs

    @property
    def oracle_graph_empty(self) -> bool:
        return int(self.oracle_graph.edges.shape[0]) == 0

    @property
    def oracle_solved(self) -> bool:
        return self.oracle_result.status == "solved"

    @property
    def baseline_solved(self) -> bool:
        return self.baseline_result.status == "solved"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports-root", default=DEFAULT_EXPORTS_ROOT,
                        help="root containing finalized exports/datasets/*/index.json")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="sealed JSON output path (refuses to overwrite)")
    parser.add_argument("--dataset-id", default=None,
                        help="override the deterministic canonical Canal Town entry "
                             "(must name exactly one Canal Town scene-prefix candidate)")
    return parser.parse_args(list(argv))


def detection_settings(matcher: JointMatchSettings) -> CorrespondenceSettings:
    """The production joint detection settings, exactly as ``match_and_solve_pair``.

    The joint matcher's own settings do not carry ``ratio_test``; detection
    borrows ``CorrespondenceSettings`` and only overrides the feature cap and the
    descriptor-footprint boundary pad, so this mirrors the shipped path.
    """
    return CorrespondenceSettings(
        max_features=int(matcher.max_features),
        mask_boundary_pad_px=float(matcher.mask_boundary_pad_px),
    )


def oracle_camera_estimate(frame: FrameInput) -> CameraEstimate:
    """Build the assisted oracle ``CameraEstimate`` from factory capture metadata."""
    intrinsics = intrinsics_for(frame.camera, frame.width, frame.height)
    pose = CameraPose(
        position=tuple(float(value) for value in frame.camera.position),
        yaw_degrees=float(frame.camera.yaw_degrees),
        pitch_degrees=float(frame.camera.pitch_degrees),
        roll_degrees=float(frame.camera.roll_degrees),
    )
    return CameraEstimate(intrinsics=intrinsics, pose=pose)


def observe_oracle_frame(
    scene: FinalizedScene,
    frame: FrameInput,
    *,
    detection: CorrespondenceSettings | None = None,
) -> OracleFrameObservation:
    """Detect joint-matcher features and resolve each keypoint's truth surface."""
    from PIL import Image

    settings = CorrespondenceSettings() if detection is None else detection
    with Image.open(frame.screenshot_path) as opened:
        opened.load()
        image = opened.convert("RGB")
    if image.size != (frame.width, frame.height):
        raise OracleCandidateGraphError(
            f"{frame.frame_id}: screenshot {image.size} does not match declared "
            f"{(frame.width, frame.height)}")
    pixels = np.asarray(image, dtype=np.uint8)
    usable = usable_mask_for(frame.frame_id, pixels.shape, None)
    features = detect_features(frame.frame_id, pixels, usable_mask=usable,
                               settings=settings)
    hits: list[SurfaceHit | None] = []
    reasons: dict[str, int] = {}
    for point in features.points:
        outcome = raycast_surface(
            frame.frame_id, frame.camera, frame.width, frame.height, scene.occupancy,
            float(point[0]), float(point[1]),
        )
        hits.append(outcome.hit)
        reasons[outcome.reason] = reasons.get(outcome.reason, 0) + 1
    return OracleFrameObservation(
        frame_id=frame.frame_id,
        width=frame.width,
        height=frame.height,
        usable_pixels=int(features.usable_pixels),
        boundary_dropped=int(features.boundary_dropped),
        features=features,
        hits=tuple(hits),
        no_truth_reasons=tuple(sorted(reasons.items())),
    )


@contextmanager
def injected_candidate_graph(graph: ReciprocalGraph) -> Iterator[ReciprocalGraph]:
    """Temporarily make production ``match_feature_frames`` use ``graph``.

    The production reciprocal graph builder is resolved from the
    ``mc_photogrammetry.pipeline.joint_matching`` module globals at call time, so
    replacing that one attribute redirects the matcher's candidate stage for the
    duration of the context. The original function object is restored in a
    ``finally`` block even if the matcher raises, so no production behaviour leaks
    past the call.
    """
    module = importlib.import_module("mc_photogrammetry.pipeline.joint_matching")
    original = module.reciprocal_topk_graph

    def _injected(*_args: Any, **_kwargs: Any) -> ReciprocalGraph:
        return graph

    module.reciprocal_topk_graph = _injected
    try:
        yield graph
    finally:
        module.reciprocal_topk_graph = original


def _canonical_pixel_index(points: Sequence[Sequence[float]]) -> dict[Pixel, int]:
    """Exact pixel to canonical (first) keypoint index, the unique pixel identity."""
    table = np.asarray(points, dtype=np.float64)
    lookup: dict[Pixel, int] = {}
    for index, row in enumerate(table):
        lookup.setdefault((float(row[0]), float(row[1])), index)
    return lookup


def build_oracle_graph(
    left: FrameFeatures,
    right: FrameFeatures,
    left_hits: Sequence[SurfaceHit | None],
    right_hits: Sequence[SurfaceHit | None],
    *,
    tolerance: float = SURFACE_TOLERANCE_BLOCKS,
) -> ReciprocalGraph:
    """The strict same-world-surface truth pair set as a candidate graph.

    Edges are exactly the unique ``(left pixel, right pixel)`` truth pairs of the
    frozen DS-R5I rule (same block face, world points within ``tolerance``),
    mapped to their canonical first keypoint index. Each edge keeps the *actual*
    L2 descriptor distance of that mapped keypoint pair, so the matcher's
    descriptor-cost diagnostics remain real; nothing is fabricated. Repeated
    keypoints at exactly one pixel collapse to one edge, distinct pixels stay
    distinct. Empty truth yields an empty graph, which the unchanged matcher
    reports as a ``candidate_graph_empty`` refusal.
    """
    truth = construct_truth_pairs(left_hits, right_hits, tolerance=tolerance)
    left_points = np.asarray(left.points, dtype=np.float64)
    right_points = np.asarray(right.points, dtype=np.float64)
    left_lookup = _canonical_pixel_index(left_points)
    right_lookup = _canonical_pixel_index(right_points)
    left_descriptors = np.asarray(left.descriptors, dtype=np.float64)
    right_descriptors = np.asarray(right.descriptors, dtype=np.float64)
    rows: list[tuple[int, int, float]] = []
    for left_pixel, right_pixel in sorted(truth):
        left_index = left_lookup.get(left_pixel)
        right_index = right_lookup.get(right_pixel)
        if left_index is None or right_index is None:
            raise OracleCandidateGraphError(
                "a truth pixel pair does not map back to a detected keypoint")
        distance = float(np.linalg.norm(
            left_descriptors[left_index] - right_descriptors[right_index]))
        rows.append((left_index, right_index, distance))
    if not rows:
        return ReciprocalGraph(
            edges=np.empty((0, 2), dtype=np.int64),
            distances=np.empty(0, dtype=np.float64),
            left_keypoints=len(left_points),
            right_keypoints=len(right_points),
            total_edges=0,
            truncated=0,
        )
    rows.sort(key=lambda row: (row[2], row[0], row[1]))
    table = np.asarray(rows, dtype=np.float64)
    return ReciprocalGraph(
        edges=table[:, :2].astype(np.int64),
        distances=table[:, 2].astype(np.float64),
        left_keypoints=len(left_points),
        right_keypoints=len(right_points),
        total_edges=len(rows),
        truncated=0,
    )


def describe_outcome(result: JointPairResult) -> tuple[str, str, bool, bool]:
    """Classify a joint result into a selection state and a pose state.

    The joint ``status`` is authoritative, not :attr:`JointPairResult.solved`:
    ``selection_unstable`` is a refusal whose carried DS-11 pose *did* solve.
    """
    matches = np.asarray(result.matches)
    matches_available = bool(matches.size)
    pose_available = result.pose is not None
    if result.status == "solved":
        return SELECTION_SOLVED, POSE_SOLVED, matches_available, pose_available
    if not matches_available:
        return SELECTION_ABSENT, POSE_NOT_ATTEMPTED, False, pose_available
    if result.pose is None:
        return SELECTION_ON_REFUSAL, POSE_NOT_ATTEMPTED, True, False
    if result.pose.solved:
        return SELECTION_ON_REFUSAL, POSE_CLASS_MISMATCH, True, True
    return SELECTION_ON_REFUSAL, POSE_ABSTAINED, True, True


def attribute_pair(
    left: OracleFrameObservation,
    right: OracleFrameObservation,
    left_camera: CameraEstimate,
    right_camera: CameraEstimate,
    *,
    matcher: JointMatchSettings | None = None,
    gates: PairGates | None = None,
    ratio_test: float = _RATIO_TEST,
    tolerance: float = SURFACE_TOLERANCE_BLOCKS,
) -> OraclePairAttribution:
    """Run the unchanged matcher twice (baseline graph, then oracle graph).

    The baseline reciprocal graph is recomputed here only so its returned
    identities can be scored against the same truth model; the matcher itself
    computes the identical graph internally. The oracle graph is injected for
    exactly one ``match_feature_frames`` call and the production builder is
    restored immediately afterwards. All other matcher settings, gates and the
    ratio test are untouched.
    """
    bounds = JointMatchSettings() if matcher is None else matcher
    baseline_graph = reciprocal_topk_graph(
        left.features.descriptors,
        right.features.descriptors,
        neighbors=int(bounds.neighbors),
        max_edges=int(bounds.max_candidate_edges),
        distance_ceiling=bounds.graph_distance_ceiling,
    )
    oracle_graph = build_oracle_graph(
        left.features, right.features, left.hits, right.hits, tolerance=tolerance)
    mutual = select_mutual_matches(
        left.features.points,
        right.features.points,
        left.features.descriptors,
        right.features.descriptors,
        ratio_test=ratio_test,
    )
    baseline_result = match_feature_frames(
        left.features,
        right.features,
        left_camera,
        right_camera,
        matcher=bounds,
        gates=gates,
        ratio_test=ratio_test,
    )
    with injected_candidate_graph(oracle_graph):
        oracle_result = match_feature_frames(
            left.features,
            right.features,
            left_camera,
            right_camera,
            matcher=bounds,
            gates=gates,
            ratio_test=ratio_test,
        )
    return OraclePairAttribution(
        left_id=baseline_result.left_id,
        right_id=baseline_result.right_id,
        baseline_graph=baseline_graph,
        oracle_graph=oracle_graph,
        baseline_result=baseline_result,
        oracle_result=oracle_result,
        mutual=score_pair_matches(
            match_indices(mutual.matches, left.features.points, right.features.points),
            left.hits, right.hits, left.features.points, right.features.points,
            tolerance=tolerance,
        ),
        baseline_candidate=score_pair_matches(
            np.asarray(baseline_graph.edges, dtype=np.int64),
            left.hits, right.hits, left.features.points, right.features.points,
            tolerance=tolerance,
        ),
        oracle_candidate=score_pair_matches(
            np.asarray(oracle_graph.edges, dtype=np.int64),
            left.hits, right.hits, left.features.points, right.features.points,
            tolerance=tolerance,
        ),
        baseline_selected=score_pair_matches(
            match_indices(baseline_result.matches, left.features.points,
                          right.features.points),
            left.hits, right.hits, left.features.points, right.features.points,
            tolerance=tolerance,
        ),
        oracle_selected=score_pair_matches(
            match_indices(oracle_result.matches, left.features.points,
                          right.features.points),
            left.hits, right.hits, left.features.points, right.features.points,
            tolerance=tolerance,
        ),
    )


def _geometry_record(geometry: PairGeometry) -> dict[str, Any]:
    return {
        "returned": geometry.returned,
        "positive": geometry.positive,
        "same_face_out_of_tolerance": geometry.same_face_out_of_tolerance,
        "same_state_same_slot_other_block": geometry.same_state_same_slot_other_block,
        "wrong_surface": geometry.wrong_surface,
        "unattributed": geometry.unattributed,
        "truth_pairs": geometry.truth_pairs,
        "truth_pairs_returned": geometry.truth_pairs_returned,
        "precision": geometry.precision,
        "recall": geometry.recall,
    }


geometry_record = _geometry_record
"""Public alias so tests and readers share one geometry JSON shape."""


def _geometry_totals(geometries: Sequence[PairGeometry]) -> ScoreTotals:
    total = ScoreTotals()
    for geometry in geometries:
        total = total + ScoreTotals(
            pairs=1,
            returned=geometry.returned,
            positive=geometry.positive,
            same_face_out_of_tolerance=geometry.same_face_out_of_tolerance,
            same_state_same_slot_other_block=(
                geometry.same_state_same_slot_other_block),
            wrong_surface=geometry.wrong_surface,
            unattributed=geometry.unattributed,
            truth_pairs=geometry.truth_pairs,
            truth_pairs_returned=geometry.truth_pairs_returned,
        )
    return total


def _totals_record(totals: ScoreTotals) -> dict[str, Any]:
    return {
        "returned": totals.returned,
        "positive": totals.positive,
        "same_face_out_of_tolerance": totals.same_face_out_of_tolerance,
        "same_state_same_slot_other_block": totals.same_state_same_slot_other_block,
        "wrong_surface": totals.wrong_surface,
        "unattributed": totals.unattributed,
        "truth_pairs": totals.truth_pairs,
        "truth_pairs_returned": totals.truth_pairs_returned,
        "positive_identity_mismatch": totals.positive - totals.truth_pairs_returned,
        "precision": metric_value(totals.positive, totals.returned,
                                  undefined_reason="no_returned_matches"),
        "recall": metric_value(totals.truth_pairs_returned, totals.truth_pairs,
                               undefined_reason="no_truth_keypoint_pairs"),
    }


def _counter(values: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def hypothesis_record(hypothesis: HypothesisScore) -> dict[str, Any]:
    """One ranked hypothesis's gate diagnostics as JSON data."""
    return {
        "yaw_class": hypothesis.yaw_class,
        "mode_index": hypothesis.mode_index,
        "support": hypothesis.support,
        "spatial_bins": {
            "left": hypothesis.spatial_bins_left,
            "right": hypothesis.spatial_bins_right,
            "min": hypothesis.spatial_bins,
        },
        "median_selected_residual_degrees": (
            hypothesis.median_selected_residual_degrees),
        "descriptor_cost": hypothesis.descriptor_cost,
        "median_all_edge_residual_degrees": (
            hypothesis.median_all_edge_residual_degrees),
        "parallax_degrees": hypothesis.parallax_degrees,
        "cheirality_fraction": hypothesis.cheirality_fraction,
        "singular_ratio": hypothesis.singular_ratio,
        "baseline_direction": list(hypothesis.baseline_direction),
        "passed": hypothesis.passed,
        "refusal": hypothesis.refusal,
    }


def _pose_record(result: JointPairResult) -> dict[str, Any]:
    pose = result.pose
    return {
        "right_yaw_class": None if pose is None else pose.right_yaw_class,
        "reason": "" if pose is None else str(pose.reason),
        "inliers": 0 if pose is None else int(pose.inliers),
        "inlier_fraction": None if pose is None else pose.inlier_fraction,
        "median_rho_degrees": None if pose is None else pose.median_rho_degrees,
    }


def result_record(result: JointPairResult, graph: ReciprocalGraph) -> dict[str, Any]:
    """One matcher run's graph counts, outcome, gates and hypotheses."""
    selection_state, pose_state, matches_available, pose_available = describe_outcome(
        result)
    return {
        "status": result.status,
        "reason": result.reason,
        "selection_state": selection_state,
        "pose_state": pose_state,
        "matches_available": matches_available,
        "pose_available": pose_available,
        "graph_edges": int(graph.edges.shape[0]),
        "candidate_edges": int(result.candidate_edges),
        "total_graph_edges": int(result.total_graph_edges),
        "truncated": int(result.truncated),
        "raw_keypoints": {"left": int(result.raw_keypoints[0]),
                          "right": int(result.raw_keypoints[1])},
        "boundary_dropped": {"left": int(result.boundary_dropped[0]),
                             "right": int(result.boundary_dropped[1])},
        "ratio_survivors": {"left": int(result.ratio_survivors[0]),
                            "right": int(result.ratio_survivors[1])},
        "mutual_survivors": int(result.mutual_survivors),
        "support": int(result.support),
        "support_floor": int(result.support_floor),
        "spatial_bins": {
            "left": int(result.spatial_bins_left),
            "right": int(result.spatial_bins_right),
            "total": int(result.spatial_bins_total),
        },
        "gates": {
            "median_selected_residual_degrees": (
                result.median_selected_residual_degrees),
            "descriptor_cost": result.descriptor_cost,
            "parallax_degrees": result.parallax_degrees,
            "cheirality_fraction": result.cheirality_fraction,
            "singular_ratio": result.singular_ratio,
            "ambiguity_margin": result.ambiguity_margin,
        },
        "pose": _pose_record(result),
        "hypotheses": [hypothesis_record(item) for item in result.hypotheses],
    }


def pair_record(pair: OraclePairAttribution) -> dict[str, Any]:
    """One pair's baseline-versus-oracle ledger, reach and denominator split."""
    return {
        "left_frame_id": pair.left_id,
        "right_frame_id": pair.right_id,
        "truth_pairs": pair.truth_pairs,
        "oracle_graph_empty": pair.oracle_graph_empty,
        "oracle_solved": pair.oracle_solved,
        "baseline_solved": pair.baseline_solved,
        "oracle_available_when_baseline_refused": (
            pair.oracle_solved and not pair.baseline_solved),
        "baseline_result": result_record(pair.baseline_result, pair.baseline_graph),
        "oracle_result": result_record(pair.oracle_result, pair.oracle_graph),
        "stages": {
            "mutual_stage": _geometry_record(pair.mutual),
            "baseline_candidate_stage": _geometry_record(pair.baseline_candidate),
            "oracle_candidate_stage": _geometry_record(pair.oracle_candidate),
            "baseline_selected_stage": _geometry_record(pair.baseline_selected),
            "oracle_selected_stage": _geometry_record(pair.oracle_selected),
        },
        "denominators": {
            "correspondence_truth_pairs": pair.truth_pairs,
            "geometry_mutual_returned": pair.mutual.returned,
            "geometry_baseline_candidate_edges": pair.baseline_candidate.returned,
            "geometry_oracle_candidate_edges": pair.oracle_candidate.returned,
            "geometry_baseline_selected_returned": pair.baseline_selected.returned,
            "geometry_oracle_selected_returned": pair.oracle_selected.returned,
        },
        "reach": {
            "mutual_correspondence_reach": pair.mutual.truth_pairs_returned,
            "baseline_candidate_correspondence_reach": (
                pair.baseline_candidate.truth_pairs_returned),
            "oracle_candidate_correspondence_reach": (
                pair.oracle_candidate.truth_pairs_returned),
            "baseline_selected_correspondence_reach": (
                pair.baseline_selected.truth_pairs_returned),
            "oracle_selected_correspondence_reach": (
                pair.oracle_selected.truth_pairs_returned),
        },
    }


def frame_record(observation: OracleFrameObservation) -> dict[str, Any]:
    """One frame's joint features, truth coverage and no-truth reason tally."""
    with_truth = sum(1 for hit in observation.hits if hit is not None)
    return {
        "frame_id": observation.frame_id,
        "width": observation.width,
        "height": observation.height,
        "usable_pixels": observation.usable_pixels,
        "keypoints": len(observation.features.points),
        "boundary_dropped": observation.boundary_dropped,
        "keypoints_with_truth": with_truth,
        "keypoints_without_truth": len(observation.hits) - with_truth,
        "no_truth_reasons": dict(observation.no_truth_reasons),
    }


def camera_record(frame: FrameInput, estimate: CameraEstimate) -> dict[str, Any]:
    """The assisted oracle camera actually handed to the matcher's geometric stage."""
    intrinsics = estimate.intrinsics
    pose = estimate.pose
    return {
        "frame_id": frame.frame_id,
        "camera_source": CAMERA_POLICY_MODE,
        "assisted": True,
        "image_size": [int(value) for value in intrinsics.image_size],
        "focal_length_pixels": [float(value) for value in intrinsics.focal_length_pixels],
        "principal_point_pixels": [
            float(value) for value in intrinsics.principal_point_pixels],
        "pose": {
            "position": None if pose is None or pose.position is None
            else [float(value) for value in pose.position],
            "yaw_degrees": None if pose is None else float(pose.yaw_degrees),
            "pitch_degrees": None if pose is None else float(pose.pitch_degrees),
            "roll_degrees": None if pose is None else float(pose.roll_degrees),
        },
    }


def select_canonical_entry(
    entries: Sequence[DatasetSceneInput],
    *,
    dataset_id: str | None = None,
    scene_prefix: str = CANONICAL_SCENE_PREFIX,
) -> DatasetSceneInput:
    """Choose exactly one canonical Canal Town export, deterministically.

    The candidate set is always scoped to entries whose ``scene_id`` starts with
    the Canal Town prefix, so no non-Canal entry (for example a Train Station
    export) can ever be selected. ``dataset_id`` (the runner's optional override)
    must then name exactly one entry *within* that scoped set. The default takes
    the **most complete** export (largest screenshot/frame count), tie-broken by
    ``(scene_id, directory, dataset_id)``. In the frozen corpus this selects the
    8-frame ``fresh-c4-canal-town-framed-2026-09-12`` export over the 4-frame
    duplicate; the considered candidates are recorded in the sealed payload.
    """
    candidates = [entry for entry in entries if entry.scene_id.startswith(scene_prefix)]
    if dataset_id is not None:
        matches = [entry for entry in candidates if entry.dataset_id == dataset_id]
        if len(matches) != 1:
            raise OracleCandidateGraphError(
                f"--dataset-id {dataset_id!r} matched {len(matches)} Canal Town "
                f"entries with scene_id prefix {scene_prefix!r}")
        return matches[0]
    if not candidates:
        raise OracleCandidateGraphError(
            f"no Canal Town entry with scene_id prefix {scene_prefix!r}")
    ordered = sorted(candidates, key=lambda entry: (
        -len(entry.screenshots), entry.scene_id, entry.directory, entry.dataset_id))
    return ordered[0]


def canonical_entry_record(
    chosen: DatasetSceneInput,
    entries: Sequence[DatasetSceneInput],
    *,
    dataset_id: str | None = None,
    scene_prefix: str = CANONICAL_SCENE_PREFIX,
) -> dict[str, Any]:
    """The canonical-entry decision, its rule and every considered candidate.

    ``considered_entries`` is always the same Canal Town scene-prefix candidate
    set that :func:`select_canonical_entry` scoped the choice to, whether or not
    an explicit ``--dataset-id`` override selected one of them. Non-Canal entries
    are never listed.
    """
    candidates = [entry for entry in entries if entry.scene_id.startswith(scene_prefix)]
    return {
        "dataset_id": chosen.dataset_id,
        "scene_id": chosen.scene_id,
        "scene_group": chosen.scene_group,
        "split": chosen.split,
        "directory": chosen.directory,
        "frame_count": len(chosen.screenshots),
        "index_sha256": chosen.index_sha256,
        "selection_rule": (
            "explicit --dataset-id override"
            if dataset_id is not None
            else "most complete Canal Town export (largest frame count), "
                 "tie-broken by (scene_id, directory, dataset_id)"
        ),
        "considered_entries": [
            {
                "dataset_id": entry.dataset_id,
                "scene_id": entry.scene_id,
                "scene_group": entry.scene_group,
                "directory": entry.directory,
                "frame_count": len(entry.screenshots),
                "index_sha256": entry.index_sha256,
            }
            for entry in sorted(candidates, key=lambda item: (
                -len(item.screenshots), item.scene_id, item.directory,
                item.dataset_id))
        ],
    }


def summarise_attributions(pairs: Sequence[OraclePairAttribution]) -> dict[str, Any]:
    """Pooled stage totals, outcome histograms and the reach/denominator split."""
    totals = {
        "mutual_stage": _geometry_totals([pair.mutual for pair in pairs]),
        "baseline_candidate_stage": _geometry_totals(
            [pair.baseline_candidate for pair in pairs]),
        "oracle_candidate_stage": _geometry_totals(
            [pair.oracle_candidate for pair in pairs]),
        "baseline_selected_stage": _geometry_totals(
            [pair.baseline_selected for pair in pairs]),
        "oracle_selected_stage": _geometry_totals(
            [pair.oracle_selected for pair in pairs]),
    }
    summary: dict[str, Any] = {
        "pair_count": len(pairs),
        "truth_pairs": sum(pair.truth_pairs for pair in pairs),
        "pairs_with_truth": sum(1 for pair in pairs if pair.truth_pairs > 0),
        "pairs_with_empty_truth": sum(1 for pair in pairs if pair.truth_pairs == 0),
        "oracle_graph_empty_pairs": sum(1 for pair in pairs if pair.oracle_graph_empty),
        "baseline_solved": sum(1 for pair in pairs if pair.baseline_solved),
        "oracle_solved": sum(1 for pair in pairs if pair.oracle_solved),
        "oracle_solved_where_baseline_refused": sum(
            1 for pair in pairs if pair.oracle_solved and not pair.baseline_solved),
        "oracle_lost_where_baseline_solved": sum(
            1 for pair in pairs if pair.baseline_solved and not pair.oracle_solved),
        "baseline_refusal_reasons": dict(sorted(_counter([
            pair.baseline_result.reason for pair in pairs
            if pair.baseline_result.status != "solved"]).items())),
        "oracle_refusal_reasons": dict(sorted(_counter([
            pair.oracle_result.reason for pair in pairs
            if pair.oracle_result.status != "solved"]).items())),
        "baseline_selection_states": dict(sorted(_counter([
            describe_outcome(pair.baseline_result)[0] for pair in pairs]).items())),
        "oracle_selection_states": dict(sorted(_counter([
            describe_outcome(pair.oracle_result)[0] for pair in pairs]).items())),
        "baseline_pose_states": dict(sorted(_counter([
            describe_outcome(pair.baseline_result)[1] for pair in pairs]).items())),
        "oracle_pose_states": dict(sorted(_counter([
            describe_outcome(pair.oracle_result)[1] for pair in pairs]).items())),
        "baseline_candidate_edges": sum(
            int(pair.baseline_graph.edges.shape[0]) for pair in pairs),
        "oracle_candidate_edges": sum(
            int(pair.oracle_graph.edges.shape[0]) for pair in pairs),
        "stages": {stage: _totals_record(total) for stage, total in totals.items()},
        "denominators": {
            "correspondence_truth_pairs": totals["oracle_selected_stage"].truth_pairs,
            "geometry_mutual_returned": totals["mutual_stage"].returned,
            "geometry_baseline_candidate_edges": (
                totals["baseline_candidate_stage"].returned),
            "geometry_oracle_candidate_edges": (
                totals["oracle_candidate_stage"].returned),
            "geometry_baseline_selected_returned": (
                totals["baseline_selected_stage"].returned),
            "geometry_oracle_selected_returned": (
                totals["oracle_selected_stage"].returned),
        },
        "reach": {
            "mutual_correspondence_reach": (
                totals["mutual_stage"].truth_pairs_returned),
            "baseline_candidate_correspondence_reach": (
                totals["baseline_candidate_stage"].truth_pairs_returned),
            "oracle_candidate_correspondence_reach": (
                totals["oracle_candidate_stage"].truth_pairs_returned),
            "baseline_selected_correspondence_reach": (
                totals["baseline_selected_stage"].truth_pairs_returned),
            "oracle_selected_correspondence_reach": (
                totals["oracle_selected_stage"].truth_pairs_returned),
        },
    }
    summary["positive_identity_mismatch"] = {
        stage: total.positive - total.truth_pairs_returned
        for stage, total in totals.items()
    }
    return summary


def build_payload(
    exports_root: str,
    entries: Sequence[DatasetSceneInput],
    manifest: Mapping[str, Any],
    canonical_entry: DatasetSceneInput,
    *,
    dataset_id: str | None = None,
    matcher: JointMatchSettings | None = None,
    tolerance: float = SURFACE_TOLERANCE_BLOCKS,
) -> dict[str, Any]:
    """Run the baseline-versus-oracle diagnostic over the canonical entry only."""
    bounds = JointMatchSettings() if matcher is None else matcher
    detection = detection_settings(bounds)
    scene = build_scene(canonical_entry)
    observations = [observe_oracle_frame(scene, frame, detection=detection)
                    for frame in scene.frames]
    cameras = [oracle_camera_estimate(frame) for frame in scene.frames]
    pairs: list[OraclePairAttribution] = []
    for left_index in range(len(observations)):
        for right_index in range(left_index + 1, len(observations)):
            pairs.append(attribute_pair(
                observations[left_index],
                observations[right_index],
                cameras[left_index],
                cameras[right_index],
                matcher=bounds,
                ratio_test=detection.ratio_test,
                tolerance=tolerance,
            ))
    frame_records: list[dict[str, Any]] = []
    keypoints_total = 0
    keypoints_with_truth_total = 0
    boundary_dropped_total = 0
    no_truth_totals: dict[str, int] = {}
    for frame, observation, camera in zip(scene.frames, observations, cameras):
        record = frame_record(observation)
        record["camera"] = camera_record(frame, camera)
        frame_records.append(record)
        keypoints_total += int(record["keypoints"])
        keypoints_with_truth_total += int(record["keypoints_with_truth"])
        boundary_dropped_total += int(record["boundary_dropped"])
        for reason, count in observation.no_truth_reasons:
            no_truth_totals[reason] = no_truth_totals.get(reason, 0) + int(count)

    summary = summarise_attributions(pairs)
    mismatch = summary["positive_identity_mismatch"]
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete",
        "created_for": "DS-R5K",
        "tool": "scripts/evaluation/evaluate_oracle_candidate_graph.py",
        "camera_policy": {
            "mode": CAMERA_POLICY_MODE,
            "assisted": True,
            "default_image_only_reconstruction": False,
            "oracle_inputs": [
                "capture.json recorded actual camera pose (planned pose fallback)",
                "capture.json field of view and frame resolution for intrinsics",
            ],
            "separation": (
                "feature extraction reads pixels and the authoritative usable mask "
                "only; the factory CameraEstimate records are consumed solely by the "
                "matcher's geometric stage"),
            "label": "assisted_oracle_camera",
        },
        "oracle_policy": {
            "mode": ORACLE_GRAPH_MODE,
            "assisted": True,
            "truth_assisted_upper_bound": True,
            "production_result": False,
            "definition": (
                "candidate edges are exactly the strict same-world-surface keypoint "
                "pixel pairs of the frozen DS-R5I truth rule (same block face and "
                "world surface points within the preregistered tolerance), mapped to "
                "their canonical first keypoint index"),
            "distances": (
                "each oracle edge keeps the actual L2 descriptor distance of its "
                "mapped keypoint pair; no descriptor or distance is fabricated"),
            "selector": (
                "the unchanged production match_feature_frames geometric selector, "
                "settings, gates and ratio test run on the injected graph"),
            "injection": (
                "mc_photogrammetry.pipeline.joint_matching.reciprocal_topk_graph is "
                "replaced for exactly one matcher call and restored in a finally "
                "block; the baseline run uses the production builder"),
        },
        "parameters": {
            "joint_match_settings": {
                "neighbors": bounds.neighbors,
                "max_candidate_edges": bounds.max_candidate_edges,
                "max_features": bounds.max_features,
                "mask_boundary_pad_px": bounds.mask_boundary_pad_px,
                "graph_distance_ceiling": bounds.graph_distance_ceiling,
                "ransac_seed": bounds.ransac_seed,
                "ransac_iterations": bounds.ransac_iterations,
                "inlier_residual_degrees": bounds.inlier_residual_degrees,
                "refit_iterations": bounds.refit_iterations,
                "min_support_absolute": bounds.min_support_absolute,
                "min_support_fraction": bounds.min_support_fraction,
                "spatial_grid": list(bounds.spatial_grid),
                "min_spatial_bins": bounds.min_spatial_bins,
                "min_parallax_degrees": bounds.min_parallax_degrees,
                "min_cheirality_fraction": bounds.min_cheirality_fraction,
                "min_singular_ratio": bounds.min_singular_ratio,
                "ambiguity_support_margin": bounds.ambiguity_support_margin,
                "mode_separation_degrees": bounds.mode_separation_degrees,
                "max_modes_per_class": bounds.max_modes_per_class,
            },
            "detection_settings": {
                "max_features": detection.max_features,
                "ratio_test": detection.ratio_test,
                "mask_boundary_pad_px": detection.mask_boundary_pad_px,
            },
            "surface_tolerance_blocks": tolerance,
            "identity": "unique pixel pair; repeated keypoints at one pixel collapse",
        },
        "inputs": {
            "exports_root": exports_root,
            "indexes": [
                {"dataset_id": entry.dataset_id, "index_path": entry.index_path,
                 "index_sha256": entry.index_sha256}
                for entry in entries
            ],
            "entries": [
                {
                    "dataset_id": entry.dataset_id,
                    "scene_id": entry.scene_id,
                    "scene_group": entry.scene_group,
                    "split": entry.split,
                    "directory": entry.directory,
                    "hashes": [list(item) for item in entry.hashes],
                    "screenshots": [list(item) for item in entry.screenshots],
                }
                for entry in entries
            ],
            "group_manifest": manifest,
            "split_leakage_issues": list(check_split_leakage(entries)),
            "fold_leakage_issues": list(check_fold_leakage(manifest)),
            "canonical_entry": canonical_entry_record(
                canonical_entry, entries, dataset_id=dataset_id),
        },
        "scene": {
            "dataset_id": canonical_entry.dataset_id,
            "scene_id": canonical_entry.scene_id,
            "scene_group": canonical_entry.scene_group,
            "split": canonical_entry.split,
            "directory": canonical_entry.directory,
            "scene_root": canonical_entry.scene_root,
            "frame_count": len(scene.frames),
            "pair_count": len(pairs),
            "frames": frame_records,
            "pairs": [pair_record(pair) for pair in pairs],
            "summary": summary,
        },
        "summary": summary,
        "renderer_geometry": {
            **geometry_policy_record(),
            "camera_policy_mode": CAMERA_POLICY_MODE,
            "camera_policy_assisted": True,
            "oracle_graph_mode": ORACLE_GRAPH_MODE,
            "frames": len(scene.frames),
            "keypoints": keypoints_total,
            "boundary_dropped": boundary_dropped_total,
            "keypoints_with_truth": keypoints_with_truth_total,
            "keypoints_without_truth": keypoints_total - keypoints_with_truth_total,
            "no_truth_reason_totals": dict(sorted(no_truth_totals.items())),
        },
        "consistency": {
            "positive_identity_mismatch": mismatch,
            "ok": all(value == 0 for value in mismatch.values()),
        },
        "limitations": list(LIMITATIONS),
    }
    payload["seal_sha256"] = seal_record(payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    output = Path(args.output)
    if output.exists():
        print(f"refusing to overwrite {output}", file=sys.stderr)
        return 2
    try:
        entries = discover_finalized_entries(args.exports_root)
        manifest = build_group_manifest(entries)
        canonical = select_canonical_entry(entries, dataset_id=args.dataset_id)
        payload = build_payload(
            args.exports_root, entries, manifest, canonical,
            dataset_id=args.dataset_id)
    except Exception as error:  # noqa: BLE001 - one-shot runner reports and exits
        print(f"evaluation failed before writing {output}: {error}", file=sys.stderr)
        return 1
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = payload["summary"]
    print(f"canonical={payload['scene']['dataset_id']} "
          f"frames={payload['scene']['frame_count']} pairs={summary['pair_count']}")
    print(f"truth_pairs={summary['truth_pairs']} "
          f"baseline_solved={summary['baseline_solved']} "
          f"oracle_solved={summary['oracle_solved']} "
          f"oracle_graph_empty={summary['oracle_graph_empty_pairs']}")
    print(f"baseline candidate reach={summary['reach']['baseline_candidate_correspondence_reach']} "
          f"oracle candidate reach={summary['reach']['oracle_candidate_correspondence_reach']} "
          f"baseline selected reach={summary['reach']['baseline_selected_correspondence_reach']} "
          f"oracle selected reach={summary['reach']['oracle_selected_correspondence_reach']}")
    print(f"wrote {output} seal={payload['seal_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
