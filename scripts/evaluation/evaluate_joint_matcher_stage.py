"""Evaluation-only joint-matcher stage attribution for finalized captures.

DS-R5J. This extends the DS-R5I-R1 truth ledger from the raw mutual-SIFT
correspondence stage to the production joint matcher
:func:`mc_photogrammetry.pipeline.joint_matching.match_feature_frames` and its
refusal/pose stage. It edits no production module, threshold, matcher, solver,
project or promotion decision, and it refuses to overwrite an existing output.

The stage ledger reuses the DS-R5I scene/frame/hash verification and truth
construction unchanged:

* hash-pinned ``exports/datasets/**/index.json`` entries and screenshots;
* the recorded *actual* camera pose and the registered ground-truth region;
* the sparse-z-buffer surface ownership, the one-vanilla-texel same-surface
  tolerance and the unique-pixel-pair identity.

The joint matcher needs :class:`~mc_photogrammetry.pipeline.contracts.CameraEstimate`
records. This diagnostic builds them from the **exact factory capture metadata**
(the recorded pose plus the frame resolution and field of view) and labels the run
**assisted**: it is an explicit oracle-camera diagnostic, not the default
image-only reconstruction path, and it makes no claim about image-derived
calibration. Feature extraction still reads only pixels and the authoritative
usable mask; the oracle cameras are consumed solely by the matcher's geometric
stage.

Per unordered frame pair the sealed record reports, in stage order:

1. the raw mutual-SIFT baseline (``select_mutual_matches``);
2. the reciprocal top-k candidate graph counts (``reciprocal_topk_graph``);
3. the selected ``JointPairResult.matches`` surface precision/recall;
4. the support/gate diagnostics and every ranked hypothesis;
5. the solved/refused reason and the pose outcome;
6. whether selected matches were available on a refusal, so an absent selection is
   distinguished from a pose abstention;
7. the correspondence denominator (truth-visible keypoint pairs) separated from
   the geometry denominators (mutual rows, candidate edges, selected matches).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
    discover_finalized_entries,
    geometry_policy_record,
    macro_average,
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
    "DEFAULT_EXPORTS_ROOT",
    "DEFAULT_OUTPUT",
    "LIMITATIONS",
    "SCHEMA",
    "JointFrameObservation",
    "JointMatcherStageEvaluationError",
    "JointPairAttribution",
    "attribute_pair",
    "build_payload",
    "camera_record",
    "describe_outcome",
    "detection_settings",
    "frame_record",
    "geometry_record",
    "hypothesis_record",
    "main",
    "observe_joint_frame",
    "oracle_camera_estimate",
    "pair_record",
    "parse_args",
    "summarise_attributions",
]

SCHEMA = "mc-photogrammetry-joint-matcher-stage-attribution-v1"
"""Sealed schema of this evaluation-only stage-attribution scorecard."""

DEFAULT_EXPORTS_ROOT = "exports/datasets"
DEFAULT_OUTPUT = ".tmp-joint-matcher-stage-20261003.json"

CAMERA_POLICY_MODE = "oracle_factory_capture_metadata"
"""The only camera source this diagnostic uses, and it is assisted."""

_RATIO_TEST = CorrespondenceSettings().ratio_test
"""The production detection ratio test; ``JointMatchSettings`` has no copy."""

SELECTION_SOLVED = "selected_solved"
SELECTION_ON_REFUSAL = "selected_on_refusal"
SELECTION_ABSENT = "absent_selection"
POSE_SOLVED = "pose_solved"
POSE_ABSTAINED = "pose_abstained"
POSE_CLASS_MISMATCH = "pose_class_mismatch"
POSE_NOT_ATTEMPTED = "pose_not_attempted"

LIMITATIONS = (
    "The factory capture metadata (recorded pose, field of view and frame "
    "resolution) is an explicit oracle-camera evaluation aid. This assisted "
    "diagnostic is not the default image-only reconstruction path and makes no "
    "claim about image-derived calibration.",
    "Pair hypotheses are not truth correspondences; precision/recall are scored "
    "against an evaluator-built truth model, not against independent labels.",
    "Only three independent topology groups exist and every index split is "
    "'train', so leave-one-group-out is exploratory and no generalization claim "
    "is made.",
    "Partial block models (bottom/top slabs, stairs, trapdoors, fences, walls, "
    "panes, plants, fluids) are unavailable without the vanilla model assets; "
    "they occlude as full cubes but are excluded from truth ownership.",
    "No texture identity is available anywhere in the pipeline: the block "
    "texture is never read, hashed or compared. "
    "same_state_same_slot_other_block is only an appearance proxy (same exact "
    "BlockState, same face slot, different block coordinate) and is not texture "
    "ground truth.",
    "Truth pairs and every returned identity are deduplicated by unique pixel "
    "pair; repeated descriptor keypoints at exactly the same pixel are one "
    "identity, and distinct pixels are never collapsed.",
    "One corpus and one run; no matcher, threshold, solver, production, project "
    "or promotion setting was changed.",
)


class JointMatcherStageEvaluationError(ValueError):
    """Invalid, unverifiable or inconsistent evaluation input."""


@dataclass(frozen=True, slots=True)
class JointFrameObservation:
    """One frame's joint-matcher features plus the R5I truth surface per keypoint.

    ``features`` is the exact :class:`FrameFeatures` the joint matcher consumes
    (detected with the production joint detection settings, so the
    descriptor-footprint boundary drop is the production one). ``hits`` and
    ``no_truth_reasons`` are the DS-R5I-R1 truth construction, reused unchanged.
    """

    frame_id: str
    width: int
    height: int
    usable_pixels: int
    boundary_dropped: int
    features: FrameFeatures
    hits: tuple[SurfaceHit | None, ...]
    no_truth_reasons: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class JointPairAttribution:
    """One unordered frame pair's joint-matcher stage ledger.

    ``mutual``, ``candidate`` and ``selected`` are :class:`PairGeometry` records
    over the same truth model; only their returned-identity sets differ (raw
    mutual rows, reciprocal top-k edges, selected one-to-one matches).
    """

    left_id: str
    right_id: str
    status: str
    reason: str
    selection_state: str
    pose_state: str
    matches_available: bool
    pose_available: bool
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
    right_yaw_class: int | None
    pose_reason: str
    pose_inliers: int
    pose_inlier_fraction: float | None
    pose_median_rho_degrees: float | None
    hypotheses: tuple[HypothesisScore, ...]
    mutual: PairGeometry
    candidate: PairGeometry
    selected: PairGeometry


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports-root", default=DEFAULT_EXPORTS_ROOT,
                        help="root containing finalized exports/datasets/*/index.json")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="sealed JSON output path (refuses to overwrite)")
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
    """Build the assisted oracle ``CameraEstimate`` from factory capture metadata.

    The pose is the frame's recorded (actual when present) factory pose and the
    intrinsics come from the factory field of view at the frame resolution. This
    is an explicit evaluation aid, not an image-derived calibration; the record is
    labelled assisted wherever it appears.
    """
    intrinsics = intrinsics_for(frame.camera, frame.width, frame.height)
    pose = CameraPose(
        position=tuple(float(value) for value in frame.camera.position),
        yaw_degrees=float(frame.camera.yaw_degrees),
        pitch_degrees=float(frame.camera.pitch_degrees),
        roll_degrees=float(frame.camera.roll_degrees),
    )
    return CameraEstimate(intrinsics=intrinsics, pose=pose)


def observe_joint_frame(
    scene: FinalizedScene,
    frame: FrameInput,
    *,
    detection: CorrespondenceSettings | None = None,
) -> JointFrameObservation:
    """Detect joint-matcher features and resolve each keypoint's truth surface.

    This is the DS-R5I ``observe_frame`` truth path with the production joint
    detection settings and the joint ``FrameFeatures`` (including the descriptor
    counts) retained.
    """
    from PIL import Image

    settings = CorrespondenceSettings() if detection is None else detection
    with Image.open(frame.screenshot_path) as opened:
        opened.load()
        image = opened.convert("RGB")
    if image.size != (frame.width, frame.height):
        raise JointMatcherStageEvaluationError(
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
    return JointFrameObservation(
        frame_id=frame.frame_id,
        width=frame.width,
        height=frame.height,
        usable_pixels=int(features.usable_pixels),
        boundary_dropped=int(features.boundary_dropped),
        features=features,
        hits=tuple(hits),
        no_truth_reasons=tuple(sorted(reasons.items())),
    )


def describe_outcome(result: JointPairResult) -> tuple[str, str, bool, bool]:
    """Classify a joint result into a selection state and a pose state.

    Selection state is one of ``selected_solved``, ``selected_on_refusal`` or
    ``absent_selection``. Pose state is one of ``pose_solved``,
    ``pose_class_mismatch`` (DS-11 solved a different yaw class, the
    ``selection_unstable`` refusal), ``pose_abstained`` (DS-11 ran and refused) or
    ``pose_not_attempted``. This makes an absent selection distinguishable from a
    pose abstention on a refusal.

    The joint ``status`` is authoritative, not :attr:`JointPairResult.solved`:
    ``selection_unstable`` is a refusal whose carried DS-11 pose *did* solve, so
    the property is ``True`` while the pair was refused.
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
    left: JointFrameObservation,
    right: JointFrameObservation,
    left_camera: CameraEstimate,
    right_camera: CameraEstimate,
    *,
    matcher: JointMatchSettings | None = None,
    gates: PairGates | None = None,
    ratio_test: float = _RATIO_TEST,
    tolerance: float = SURFACE_TOLERANCE_BLOCKS,
) -> JointPairAttribution:
    """Attribute one unordered frame pair across the three matcher stages.

    The raw mutual baseline and the reciprocal top-k graph are recomputed here so
    their returned identities can be scored against the same truth model, while
    the joint selection comes from the unchanged production
    :func:`match_feature_frames`.
    """
    bounds = JointMatchSettings() if matcher is None else matcher
    graph = reciprocal_topk_graph(
        left.features.descriptors,
        right.features.descriptors,
        neighbors=int(bounds.neighbors),
        max_edges=int(bounds.max_candidate_edges),
        distance_ceiling=bounds.graph_distance_ceiling,
    )
    mutual = select_mutual_matches(
        left.features.points,
        right.features.points,
        left.features.descriptors,
        right.features.descriptors,
        ratio_test=ratio_test,
    )
    result = match_feature_frames(
        left.features,
        right.features,
        left_camera,
        right_camera,
        matcher=bounds,
        gates=gates,
        ratio_test=ratio_test,
    )
    mutual_geometry = score_pair_matches(
        match_indices(mutual.matches, left.features.points, right.features.points),
        left.hits, right.hits, left.features.points, right.features.points,
        tolerance=tolerance,
    )
    candidate_geometry = score_pair_matches(
        np.asarray(graph.edges, dtype=np.int64),
        left.hits, right.hits, left.features.points, right.features.points,
        tolerance=tolerance,
    )
    selected_geometry = score_pair_matches(
        match_indices(result.matches, left.features.points, right.features.points),
        left.hits, right.hits, left.features.points, right.features.points,
        tolerance=tolerance,
    )
    selection_state, pose_state, matches_available, pose_available = describe_outcome(
        result)
    pose = result.pose
    return JointPairAttribution(
        left_id=result.left_id,
        right_id=result.right_id,
        status=result.status,
        reason=result.reason,
        selection_state=selection_state,
        pose_state=pose_state,
        matches_available=matches_available,
        pose_available=pose_available,
        raw_keypoints=(int(result.raw_keypoints[0]), int(result.raw_keypoints[1])),
        boundary_dropped=(int(result.boundary_dropped[0]), int(result.boundary_dropped[1])),
        candidate_edges=int(result.candidate_edges),
        total_graph_edges=int(result.total_graph_edges),
        truncated=int(result.truncated),
        ratio_survivors=(int(result.ratio_survivors[0]), int(result.ratio_survivors[1])),
        mutual_survivors=int(result.mutual_survivors),
        support=int(result.support),
        support_floor=int(result.support_floor),
        spatial_bins_left=int(result.spatial_bins_left),
        spatial_bins_right=int(result.spatial_bins_right),
        spatial_bins_total=int(result.spatial_bins_total),
        median_selected_residual_degrees=result.median_selected_residual_degrees,
        descriptor_cost=result.descriptor_cost,
        parallax_degrees=result.parallax_degrees,
        cheirality_fraction=result.cheirality_fraction,
        singular_ratio=result.singular_ratio,
        ambiguity_margin=result.ambiguity_margin,
        right_yaw_class=result.right_yaw_class,
        pose_reason="" if pose is None else str(pose.reason),
        pose_inliers=0 if pose is None else int(pose.inliers),
        pose_inlier_fraction=None if pose is None else pose.inlier_fraction,
        pose_median_rho_degrees=None if pose is None else pose.median_rho_degrees,
        hypotheses=tuple(result.hypotheses),
        mutual=mutual_geometry,
        candidate=candidate_geometry,
        selected=selected_geometry,
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


def _geometry_totals(geometries: Iterable[PairGeometry]) -> ScoreTotals:
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
        "precision": metric_value(totals.positive, totals.returned,
                                  undefined_reason="no_returned_matches"),
        "recall": metric_value(totals.truth_pairs_returned, totals.truth_pairs,
                               undefined_reason="no_truth_keypoint_pairs"),
    }


def _counter(values: Iterable[str]) -> dict[str, int]:
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


def pair_record(pair: JointPairAttribution) -> dict[str, Any]:
    """One pair's stage ledger as JSON data, including its denominator split."""
    return {
        "left_frame_id": pair.left_id,
        "right_frame_id": pair.right_id,
        "status": pair.status,
        "reason": pair.reason,
        "selection_state": pair.selection_state,
        "pose_state": pair.pose_state,
        "matches_available": pair.matches_available,
        "pose_available": pair.pose_available,
        "raw_keypoints": {"left": pair.raw_keypoints[0], "right": pair.raw_keypoints[1]},
        "boundary_dropped": {
            "left": pair.boundary_dropped[0], "right": pair.boundary_dropped[1]},
        "candidate_edges": pair.candidate_edges,
        "total_graph_edges": pair.total_graph_edges,
        "truncated": pair.truncated,
        "ratio_survivors": {"left": pair.ratio_survivors[0],
                            "right": pair.ratio_survivors[1]},
        "mutual_survivors": pair.mutual_survivors,
        "support": pair.support,
        "support_floor": pair.support_floor,
        "spatial_bins": {
            "left": pair.spatial_bins_left,
            "right": pair.spatial_bins_right,
            "total": pair.spatial_bins_total,
        },
        "gates": {
            "median_selected_residual_degrees": pair.median_selected_residual_degrees,
            "descriptor_cost": pair.descriptor_cost,
            "parallax_degrees": pair.parallax_degrees,
            "cheirality_fraction": pair.cheirality_fraction,
            "singular_ratio": pair.singular_ratio,
            "ambiguity_margin": pair.ambiguity_margin,
        },
        "pose": {
            "right_yaw_class": pair.right_yaw_class,
            "reason": pair.pose_reason,
            "inliers": pair.pose_inliers,
            "inlier_fraction": pair.pose_inlier_fraction,
            "median_rho_degrees": pair.pose_median_rho_degrees,
        },
        "hypotheses": [hypothesis_record(item) for item in pair.hypotheses],
        "mutual_stage": _geometry_record(pair.mutual),
        "candidate_stage": _geometry_record(pair.candidate),
        "selected_stage": _geometry_record(pair.selected),
        "denominators": {
            "correspondence_truth_pairs": pair.selected.truth_pairs,
            "geometry_mutual_returned": pair.mutual.returned,
            "geometry_candidate_edges": pair.candidate.returned,
            "geometry_selected_returned": pair.selected.returned,
            "mutual_correspondence_reach": pair.mutual.truth_pairs_returned,
            "candidate_correspondence_reach": pair.candidate.truth_pairs_returned,
            "selected_correspondence_reach": pair.selected.truth_pairs_returned,
        },
    }


def frame_record(observation: JointFrameObservation) -> dict[str, Any]:
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


def summarise_attributions(pairs: Sequence[JointPairAttribution]) -> dict[str, Any]:
    """Pooled stage totals, outcome histogram and the denominator split."""
    mutual = _geometry_totals(pair.mutual for pair in pairs)
    candidate = _geometry_totals(pair.candidate for pair in pairs)
    selected = _geometry_totals(pair.selected for pair in pairs)
    refusals = [pair for pair in pairs if pair.status != "solved"]
    return {
        "pair_count": len(pairs),
        "solved": sum(1 for pair in pairs if pair.status == "solved"),
        "refused": len(refusals),
        "refusal_reasons": dict(sorted(_counter(
            pair.reason for pair in pairs if pair.reason).items())),
        "selection_states": dict(sorted(_counter(
            pair.selection_state for pair in pairs).items())),
        "pose_states": dict(sorted(_counter(pair.pose_state for pair in pairs).items())),
        "refusals_with_selected_matches": sum(
            1 for pair in refusals if pair.matches_available),
        "refusals_without_selected_matches": sum(
            1 for pair in refusals if not pair.matches_available),
        "refusals_with_pose_attempt": sum(
            1 for pair in refusals if pair.pose_available),
        "candidate_edges": sum(pair.candidate_edges for pair in pairs),
        "total_graph_edges": sum(pair.total_graph_edges for pair in pairs),
        "truncated": sum(pair.truncated for pair in pairs),
        "pairs_with_candidate_edges": sum(
            1 for pair in pairs if pair.candidate_edges > 0),
        "selected_matches": selected.returned,
        "mutual_stage": _totals_record(mutual),
        "candidate_stage": _totals_record(candidate),
        "selected_stage": _totals_record(selected),
        "denominators": {
            "correspondence_truth_pairs": selected.truth_pairs,
            "geometry_mutual_returned": mutual.returned,
            "geometry_candidate_edges": candidate.returned,
            "geometry_selected_returned": selected.returned,
            "mutual_correspondence_reach": mutual.truth_pairs_returned,
            "candidate_correspondence_reach": candidate.truth_pairs_returned,
            "selected_correspondence_reach": selected.truth_pairs_returned,
        },
        "positive_identity_mismatch": {
            "mutual": mutual.positive - mutual.truth_pairs_returned,
            "candidate": candidate.positive - candidate.truth_pairs_returned,
            "selected": selected.positive - selected.truth_pairs_returned,
        },
    }


def _group_record(
    scene_group: str,
    entries: Sequence[DatasetSceneInput],
    pairs: Sequence[JointPairAttribution],
) -> dict[str, Any]:
    summary = summarise_attributions(pairs)
    return {
        "scene_group": scene_group,
        "scene_ids": sorted({entry.scene_id for entry in entries}),
        "datasets": sorted({entry.dataset_id for entry in entries}),
        "scene_entry_count": len(entries),
        "pair_count": len(pairs),
        **summary,
    }


def _scene_record(
    entry: DatasetSceneInput,
    frames: Sequence[FrameInput],
    observations: Sequence[JointFrameObservation],
    cameras: Sequence[CameraEstimate],
    pairs: Sequence[JointPairAttribution],
) -> dict[str, Any]:
    frame_records = []
    for frame, observation, camera in zip(frames, observations, cameras):
        record = frame_record(observation)
        record["camera"] = camera_record(frame, camera)
        frame_records.append(record)
    return {
        "dataset_id": entry.dataset_id,
        "scene_id": entry.scene_id,
        "scene_group": entry.scene_group,
        "split": entry.split,
        "directory": entry.directory,
        "scene_root": entry.scene_root,
        "frame_count": len(frames),
        "pair_count": len(pairs),
        "frames": frame_records,
        "pairs": [pair_record(pair) for pair in pairs],
        "summary": summarise_attributions(pairs),
    }


def build_payload(
    exports_root: str,
    entries: Sequence[DatasetSceneInput],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the assisted joint-matcher stage ledger over every finalized scene."""
    matcher = JointMatchSettings()
    detection = detection_settings(matcher)
    scene_records: list[dict[str, Any]] = []
    pairs_by_group: dict[str, list[JointPairAttribution]] = {}
    all_pairs: list[JointPairAttribution] = []
    frames_total = 0
    keypoints_total = 0
    keypoints_with_truth_total = 0
    boundary_dropped_total = 0
    no_truth_totals: dict[str, int] = {}

    for entry in entries:
        scene = build_scene(entry)
        observations = [observe_joint_frame(scene, frame, detection=detection)
                        for frame in scene.frames]
        cameras = [oracle_camera_estimate(frame) for frame in scene.frames]
        pairs: list[JointPairAttribution] = []
        for left_index in range(len(observations)):
            for right_index in range(left_index + 1, len(observations)):
                pairs.append(attribute_pair(
                    observations[left_index],
                    observations[right_index],
                    cameras[left_index],
                    cameras[right_index],
                    matcher=matcher,
                    ratio_test=detection.ratio_test,
                ))
        for observation in observations:
            frames_total += 1
            keypoints_total += len(observation.features.points)
            keypoints_with_truth_total += sum(
                1 for hit in observation.hits if hit is not None)
            boundary_dropped_total += int(observation.boundary_dropped)
            for reason, count in observation.no_truth_reasons:
                no_truth_totals[reason] = no_truth_totals.get(reason, 0) + int(count)
        scene_records.append(_scene_record(
            entry, scene.frames, observations, cameras, pairs))
        pairs_by_group.setdefault(entry.scene_group, []).extend(pairs)
        all_pairs.extend(pairs)

    group_records = [
        _group_record(group, [entry for entry in entries if entry.scene_group == group],
                      pairs_by_group.get(group, []))
        for group in sorted({entry.scene_group for entry in entries})
    ]
    pooled = summarise_attributions(all_pairs)
    macro = {
        "selected_precision": macro_average(
            [record["selected_stage"]["precision"]["value"] for record in group_records]),
        "selected_recall": macro_average(
            [record["selected_stage"]["recall"]["value"] for record in group_records]),
        "mutual_precision": macro_average(
            [record["mutual_stage"]["precision"]["value"] for record in group_records]),
        "mutual_recall": macro_average(
            [record["mutual_stage"]["recall"]["value"] for record in group_records]),
        "candidate_recall": macro_average(
            [record["candidate_stage"]["recall"]["value"] for record in group_records]),
    }
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete",
        "created_for": "DS-R5J",
        "tool": "scripts/evaluation/evaluate_joint_matcher_stage.py",
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
                "joint matcher's geometric stage"),
            "label": "assisted_oracle_camera",
        },
        "parameters": {
            "joint_match_settings": {
                "neighbors": matcher.neighbors,
                "max_candidate_edges": matcher.max_candidate_edges,
                "max_features": matcher.max_features,
                "mask_boundary_pad_px": matcher.mask_boundary_pad_px,
                "graph_distance_ceiling": matcher.graph_distance_ceiling,
                "ransac_seed": matcher.ransac_seed,
                "ransac_iterations": matcher.ransac_iterations,
                "inlier_residual_degrees": matcher.inlier_residual_degrees,
                "refit_iterations": matcher.refit_iterations,
                "min_support_absolute": matcher.min_support_absolute,
                "min_support_fraction": matcher.min_support_fraction,
                "spatial_grid": list(matcher.spatial_grid),
                "min_spatial_bins": matcher.min_spatial_bins,
                "min_parallax_degrees": matcher.min_parallax_degrees,
                "min_cheirality_fraction": matcher.min_cheirality_fraction,
                "min_singular_ratio": matcher.min_singular_ratio,
                "ambiguity_support_margin": matcher.ambiguity_support_margin,
                "mode_separation_degrees": matcher.mode_separation_degrees,
                "max_modes_per_class": matcher.max_modes_per_class,
            },
            "detection_settings": {
                "max_features": detection.max_features,
                "ratio_test": detection.ratio_test,
                "mask_boundary_pad_px": detection.mask_boundary_pad_px,
            },
            "surface_tolerance_blocks": SURFACE_TOLERANCE_BLOCKS,
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
        },
        "scenes": scene_records,
        "groups": group_records,
        "pooled": pooled,
        "macro": macro,
        "stage_attribution": {
            "raw_mutual_baseline": pooled["mutual_stage"],
            "reciprocal_topk_candidate": pooled["candidate_stage"],
            "joint_selected": pooled["selected_stage"],
            "denominators": pooled["denominators"],
            "outcomes": {
                "solved": pooled["solved"],
                "refused": pooled["refused"],
                "refusal_reasons": pooled["refusal_reasons"],
                "selection_states": pooled["selection_states"],
                "pose_states": pooled["pose_states"],
                "refusals_with_selected_matches": (
                    pooled["refusals_with_selected_matches"]),
                "refusals_without_selected_matches": (
                    pooled["refusals_without_selected_matches"]),
                "refusals_with_pose_attempt": pooled["refusals_with_pose_attempt"],
            },
            "candidate_graph": {
                "candidate_edges": pooled["candidate_edges"],
                "total_graph_edges": pooled["total_graph_edges"],
                "truncated": pooled["truncated"],
                "pairs_with_candidate_edges": pooled["pairs_with_candidate_edges"],
            },
        },
        "renderer_geometry": {
            **geometry_policy_record(),
            "camera_policy_mode": CAMERA_POLICY_MODE,
            "camera_policy_assisted": True,
            "frames": frames_total,
            "keypoints": keypoints_total,
            "boundary_dropped": boundary_dropped_total,
            "keypoints_with_truth": keypoints_with_truth_total,
            "keypoints_without_truth": keypoints_total - keypoints_with_truth_total,
            "no_truth_reason_totals": dict(sorted(no_truth_totals.items())),
        },
        "consistency": {
            "positive_identity_mismatch": pooled["positive_identity_mismatch"],
            "ok": all(value == 0
                      for value in pooled["positive_identity_mismatch"].values()),
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
        payload = build_payload(args.exports_root, entries, manifest)
    except Exception as error:  # noqa: BLE001 - one-shot runner reports and exits
        print(f"evaluation failed before writing {output}: {error}", file=sys.stderr)
        return 1
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pooled = payload["pooled"]
    print(f"scenes={len(payload['scenes'])} groups={len(payload['groups'])} "
          f"frames={payload['renderer_geometry']['frames']} "
          f"pairs={pooled['pair_count']}")
    print(f"solved={pooled['solved']} refused={pooled['refused']} "
          f"candidate_edges={pooled['candidate_edges']} "
          f"selected={pooled['selected_matches']}")
    print(f"selected precision={pooled['selected_stage']['precision']['value']} "
          f"recall={pooled['selected_stage']['recall']['value']} "
          f"candidate recall={pooled['candidate_stage']['recall']['value']}")
    print(f"wrote {output} seal={payload['seal_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
