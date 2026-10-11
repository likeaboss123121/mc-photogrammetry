"""Evaluation-only same-surface world-hit proximity oracle vs mode selection (DS-R5O).

DS-R5K ran the unchanged production matcher on a strict same-world-surface oracle
candidate graph and found that it raised candidate reach from 345 to all 1,213
Canal Town truth pairs but still solved only 4/28 pairs; DS-R5N then showed a
support-floor-only control gained no solve. The frozen DS-R5I truth rule behind
that oracle requires both keypoints to own the *same block face* and their raycast
world surface points to lie within ``SURFACE_TOLERANCE_BLOCKS`` = 1/16 block.

A shared block face is not a shared feature: two distinct detector keypoints can
land on one block face a small fraction of a block apart. DS-R5O varies that
continuous world-hit separation on the same canonical 8-view Canal Town export:
**at a tighter or looser same-surface world-hit proximity, does the unchanged
solver still fail, and how do reach, selection and solve success move with the
tolerance?** This is a single-fixture *sensitivity* sweep only. Because candidate
count, achieved surface precision and each arm's own proximity denominator change
together across arms, it cannot isolate within-face correspondence ambiguity from
the unchanged support, spatial and ambiguity gates, and it neither confirms nor
rules out any single cause of the R5K/R5N failures.

Every pair is run through the unchanged production
:func:`mc_photogrammetry.pipeline.joint_matching.match_feature_frames` under six
arms that share identical detection, assisted cameras, gates, ratio test,
thresholds and solver; only the injected candidate graph differs:

1. ``production_baseline`` - the production reciprocal descriptor graph;
2. ``surface_oracle`` - the reviewed DS-R5K same-surface graph at the frozen
   1/16 separation tolerance;
3. ``point_oracle_0.05`` / ``_0.125`` / ``_0.25`` / ``_0.5`` - the same-surface
   graph rebuilt under a **fixed, preregistered continuous world-hit separation
   sweep**. A pair is admitted only when both detector keypoints raycast into the
   *same visible surface* (same block coordinate and face slot) **and** their
   continuous hit points are within the swept separation tolerance. The
   ``point_oracle_*`` arm keys and the ``exact_point`` schema/tool naming are
   legacy labels retained for continuity with the sealed artifact: the rule is
   continuous world-hit proximity at each tolerance, not literal feature-point
   identity. No tolerance is selected from downstream solve success and none is a
   production threshold.

Reused reviewed machinery (loaded by path, never by reading a sealed artifact):
the DS-R5K canonical-entry selection, oracle camera estimate, detection settings,
frame observation, graph injection/restoration, outcome classification and JSON
records, and the DS-R5N pinned preflight for the immutable DS-R5K index, scene and
screenshot hashes. The sealed ``.tmp-oracle-candidate-graph-20261003.json`` and
``.tmp-oracle-support-sensitivity-20261003.json`` artifacts are neither read nor
written.

Camera poses and ground truth are evaluation-only oracle inputs. This is a
single-fixture, truth-assisted diagnostic: it is not a production result, not a
model-performance estimate and makes no generalization claim. Normal
reconstruction remains image-only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mc_photogrammetry.evaluation_pair_correspondence import (
    SURFACE_TOLERANCE_BLOCKS,
    DatasetSceneInput,
    FinalizedScene,
    FrameInput,
    PairGeometry,
    build_group_manifest,
    build_scene,
    check_fold_leakage,
    check_split_leakage,
    construct_truth_pairs,
    discover_finalized_entries,
    geometry_policy_record,
    match_indices,
    score_pair_matches,
    seal_record,
)
from mc_photogrammetry.pipeline.connected_camera import PairGates
from mc_photogrammetry.pipeline.contracts import CameraEstimate
from mc_photogrammetry.pipeline.correspondence import (
    CorrespondenceSettings,
    ReciprocalGraph,
    reciprocal_topk_graph,
)
from mc_photogrammetry.pipeline.joint_matching import (
    JointMatchSettings,
    JointPairResult,
    match_feature_frames,
)

__all__ = [
    "ARM_ORDER",
    "ARM_PRODUCTION",
    "ARM_SURFACE",
    "DEFAULT_EXPORTS_ROOT",
    "DEFAULT_OUTPUT",
    "LIMITATIONS",
    "POINT_ARM_TOLERANCES",
    "POINT_TOLERANCES",
    "SCHEMA",
    "SEALED_ARTIFACTS",
    "SURFACE_TOLERANCE",
    "ArmAttribution",
    "ExactPointOracleError",
    "ExactPointPairAttribution",
    "arm_record",
    "attribute_pair",
    "build_payload",
    "build_point_oracle_graph",
    "load_r5n_module",
    "main",
    "output_targets_a_sealed_artifact",
    "pair_record",
    "parse_args",
    "point_arm",
    "summarise_attributions",
    "tolerance_label",
]

SCHEMA = "mc-photogrammetry-exact-point-oracle-v1"
"""Sealed schema of this evaluation-only same-surface world-hit proximity diagnostic.

The ``exact-point`` schema string is a legacy label: the measured rule is
continuous raycast world-hit proximity at each swept tolerance, not feature-point
identity.
"""

DEFAULT_EXPORTS_ROOT = "exports/datasets"
DEFAULT_OUTPUT = ".tmp-exact-point-oracle-20261004.json"
"""Reserved sealed output path of this diagnostic; never a prior sealed artifact."""

SEALED_ARTIFACTS = (
    ".tmp-oracle-candidate-graph-20261003.json",
    ".tmp-oracle-support-sensitivity-20261003.json",
)
"""Reviewed sealed artifacts this run must neither read nor write."""

SURFACE_TOLERANCE = SURFACE_TOLERANCE_BLOCKS
"""The frozen DS-R5I same-surface separation tolerance (1/16 block)."""

POINT_TOLERANCES = (0.05, 0.125, 0.25, 0.5)
"""The fixed, preregistered same-surface world-hit separation sweep, in blocks.

The R5K surface oracle sits at :data:`SURFACE_TOLERANCE` (0.0625), so the sweep
brackets it: 0.05 is stricter than the surface rule and 0.125/0.25/0.5 are
progressively looser. The tolerance bounds the separation of continuous raycast
world-hit points on one shared surface; it is not a literal feature-identity test.
No value is chosen from downstream solve success and this is a single-fixture
sensitivity instrument, not a causal experiment.
"""

ARM_PRODUCTION = "production_baseline"
ARM_SURFACE = "surface_oracle"
"""The retained production arm and the reviewed DS-R5K same-surface arm."""


def tolerance_label(tolerance: float) -> str:
    """Deterministic short label for one separation tolerance."""
    return f"{float(tolerance):g}"


def point_arm(tolerance: float) -> str:
    """Legacy arm key of the same-surface world-hit proximity oracle.

    The ``point_oracle_<tolerance>`` key is retained for continuity with the
    sealed artifact; the measured rule is continuous world-hit proximity.
    """
    return f"point_oracle_{tolerance_label(tolerance)}"


POINT_ARM_TOLERANCES: Mapping[str, float] = {
    point_arm(tolerance): float(tolerance) for tolerance in POINT_TOLERANCES
}
"""Legacy point-oracle arm key -> the separation tolerance it was built at."""

ARM_ORDER = (
    (ARM_PRODUCTION, ARM_SURFACE)
    + tuple(point_arm(tolerance) for tolerance in POINT_TOLERANCES)
)
"""Every arm, in deterministic order: baseline, surface, then the point sweep."""

SAME_SURFACE_NOT_SAME_FEATURE_POINT = (
    "A same-surface oracle edge means its two endpoints observe the same block "
    "face with continuous raycast world-hit points separated by no more than the "
    "arm's tolerance; it does NOT establish that they are the same texture feature "
    "point, and the tolerance is not a feature-identity test. The point-oracle "
    "arms tighten/loosen that continuous proximity bound only to measure candidate "
    "reach and selection, never to claim feature identity."
)

EXACT_POINT_CAVEAT = (
    "The point-oracle separation sweep is a fixed evaluation instrument on one "
    "fixture: the arms differ jointly in candidate count, achieved surface "
    "precision and their own proximity denominator, so the sweep is a sensitivity "
    "measurement only and cannot isolate correspondence ambiguity or localize one "
    "remaining bottleneck. No tolerance is selected from solve success and none "
    "may be promoted to a production matcher, solver, gate or support threshold."
)

LIMITATIONS = (
    "This is an evaluation-only, truth-assisted diagnostic. Every oracle arm is an "
    "UPPER BOUND, not a production result and not a model-performance estimate.",
    SAME_SURFACE_NOT_SAME_FEATURE_POINT,
    EXACT_POINT_CAVEAT,
    "The tolerance sweep is a single-fixture sensitivity measurement: reach, "
    "achieved surface precision and each arm's own proximity denominator change "
    "together across arms, so the evidence neither confirms nor rules out "
    "within-face correspondence ambiguity as a cause and does not causally "
    "localize a remaining bottleneck.",
    "The assisted oracle camera uses the recorded factory capture metadata (actual "
    "pose with planned fallback, field of view, frame resolution). Normal "
    "reconstruction never sees it; this makes no image-derived calibration claim.",
    "Only the single canonical 8-view Canal Town export is evaluated: one export, "
    "one run. Only three topology groups exist and every index split is 'train', so "
    "no generalization claim is made.",
    "Candidate and selected precision/recall are scored against the evaluator's "
    "unique-pixel-pair surface truth model, not against independent feature-point "
    "or texture labels; joint hypotheses are not truth correspondences.",
    "Partial block models (bottom/top slabs, stairs, trapdoors, fences, walls, "
    "panes, plants, fluids) occlude as full cubes but are excluded from truth "
    "ownership; keypoints without a registered surface are censored, never scored.",
    "No texture identity is available anywhere in the pipeline; "
    "same_state_same_slot_other_block is an appearance proxy only.",
    "Truth pairs and every returned identity are deduplicated by unique pixel pair; "
    "repeated descriptor keypoints at exactly one pixel collapse and distinct "
    "pixels are never collapsed.",
    "The production reciprocal graph builder is restored in a finally block after "
    "every oracle call; the production baseline arm never uses an injected graph.",
    "No matcher, threshold, gate, solver, production, project or promotion setting "
    "is changed, and no prior sealed artifact is read or written.",
)

_RATIO_TEST = CorrespondenceSettings().ratio_test
"""The production detection ratio test; ``JointMatchSettings`` has no copy."""

_R5N_SCRIPT = Path(__file__).with_name("evaluate_support_sensitivity.py")


class ExactPointOracleError(ValueError):
    """Invalid, unverifiable or inconsistent evaluation input."""


def load_r5n_module(path: str | Path | None = None):
    """Load the sealed DS-R5N evaluation module by path (never its artifact).

    DS-R5N owns the reviewed pinned preflight for the immutable DS-R5K index,
    scene-file and screenshot hashes plus the DS-R5K module itself. Reusing them
    here guarantees the two diagnostics cannot drift apart. Loading the script
    does not read ``.tmp-oracle-support-sensitivity-20261003.json`` and does not
    run its ``main``.
    """
    script = _R5N_SCRIPT if path is None else Path(path)
    spec = importlib.util.spec_from_file_location("ds_r5o_support_sensitivity", script)
    if spec is None or spec.loader is None:
        raise ExactPointOracleError(f"cannot load the DS-R5N module from {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


R5N = load_r5n_module()
"""The reviewed DS-R5N module, which itself loads and owns the DS-R5K module."""

R5K = R5N.R5K
"""The reviewed DS-R5K oracle machinery this diagnostic reuses verbatim."""

# Reviewed DS-R5K machinery (all evaluated by the DS-R5K/DS-R5N reviews).
select_canonical_entry = R5K.select_canonical_entry
canonical_entry_record = R5K.canonical_entry_record
oracle_camera_estimate = R5K.oracle_camera_estimate
detection_settings = R5K.detection_settings
describe_outcome = R5K.describe_outcome
hypothesis_record = R5K.hypothesis_record
geometry_record = R5K.geometry_record
frame_record = R5K.frame_record
camera_record = R5K.camera_record
observe_oracle_frame = R5K.observe_oracle_frame
injected_candidate_graph = R5K.injected_candidate_graph
_pose_record = R5K._pose_record
_geometry_totals = R5K._geometry_totals
_totals_record = R5K._totals_record

# Reviewed DS-R5N pinned preflight and pose-error records.
verify_r5k_pins = R5N.verify_r5k_pins
verify_canonical_scene_pins = R5N.verify_canonical_scene_pins
pose_error_record = R5N.pose_error_record
yaw_class_record = R5N.yaw_class_record

CAMERA_POLICY_MODE = R5K.CAMERA_POLICY_MODE
ORACLE_GRAPH_MODE = R5K.ORACLE_GRAPH_MODE


def build_point_oracle_graph(
    left,
    right,
    left_hits,
    right_hits,
    *,
    tolerance: float,
) -> ReciprocalGraph:
    """The same-surface world-hit proximity oracle graph at one tolerance.

    Both detector keypoints must raycast into the *same visible surface* (same
    block coordinate and face slot) and their retained continuous hit points must
    be separated by no more than ``tolerance`` blocks. That is exactly the frozen
    DS-R5I truth rule :func:`construct_truth_pairs` evaluated at this tolerance,
    mapped by the reviewed DS-R5K graph builder to canonical first keypoint
    indices with the actual L2 descriptor distances preserved and edges sorted by
    ``(distance, left index, right index)``. The bound is continuous world-hit
    proximity on one surface, not literal feature-point identity.

    Fail-closed: a non-positive/non-finite tolerance, a truth table whose length
    does not match the detected keypoints, or a truth pixel with no detected
    keypoint raises :class:`ExactPointOracleError` instead of silently dropping
    evidence. Empty truth yields an empty graph (the unchanged matcher then
    reports its typed ``candidate_graph_empty`` refusal).
    """
    value = float(tolerance)
    if not math.isfinite(value) or value <= 0.0:
        raise ExactPointOracleError(
            "the point-oracle separation tolerance must be a positive finite number")
    if len(left_hits) != len(left.points) or len(right_hits) != len(right.points):
        raise ExactPointOracleError(
            "the truth hit table must carry one entry per detected keypoint")
    try:
        return R5K.build_oracle_graph(
            left, right, left_hits, right_hits, tolerance=value)
    except R5K.OracleCandidateGraphError as error:
        raise ExactPointOracleError(str(error)) from error


@dataclass(frozen=True, slots=True)
class ArmAttribution:
    """One arm's graph, unchanged-matcher result and scored stage geometry."""

    arm: str
    graph: ReciprocalGraph
    result: JointPairResult
    separation_tolerance_blocks: float | None
    candidate_surface_truth: PairGeometry
    selected_surface_truth: PairGeometry
    candidate_arm_truth: PairGeometry | None
    selected_arm_truth: PairGeometry | None

    @property
    def solved(self) -> bool:
        return self.result.status == "solved"

    @property
    def graph_empty(self) -> bool:
        return int(self.graph.edges.shape[0]) == 0


@dataclass(frozen=True, slots=True)
class ExactPointPairAttribution:
    """One unordered frame pair's six-arm same-surface proximity ledger."""

    left_id: str
    right_id: str
    left_points: np.ndarray
    right_points: np.ndarray
    left_camera: CameraEstimate
    right_camera: CameraEstimate
    arms: tuple[ArmAttribution, ...]
    surface_truth_pairs: int
    point_truth_pairs: tuple[tuple[float, int], ...]
    left_keypoints_with_truth: int
    right_keypoints_with_truth: int
    left_keypoints_without_truth: int
    right_keypoints_without_truth: int

    def arm(self, name: str) -> ArmAttribution:
        for item in self.arms:
            if item.arm == name:
                return item
        raise ExactPointOracleError(f"unknown arm {name!r}")

    @property
    def production_arm(self) -> ArmAttribution:
        return self.arm(ARM_PRODUCTION)

    @property
    def surface_arm(self) -> ArmAttribution:
        return self.arm(ARM_SURFACE)

    @property
    def point_arms(self) -> tuple[ArmAttribution, ...]:
        return tuple(self.arm(point_arm(tolerance)) for tolerance in POINT_TOLERANCES)


def _scored(
    indices: np.ndarray,
    left,
    right,
    *,
    tolerance: float,
) -> PairGeometry:
    return score_pair_matches(
        indices,
        left.hits,
        right.hits,
        left.features.points,
        right.features.points,
        tolerance=tolerance,
    )


def _selected_indices(result: JointPairResult, left_points, right_points) -> np.ndarray:
    return match_indices(result.matches, left_points, right_points)


def _arm_attribution(
    arm: str,
    graph: ReciprocalGraph,
    result: JointPairResult,
    left,
    right,
    *,
    separation_tolerance: float | None,
    surface_tolerance: float,
) -> ArmAttribution:
    left_points = left.features.points
    right_points = right.features.points
    selected = _selected_indices(result, left_points, right_points)
    candidate_arm = (
        None if separation_tolerance is None
        else _scored(graph.edges, left, right, tolerance=separation_tolerance)
    )
    selected_arm = (
        None if separation_tolerance is None
        else _scored(selected, left, right, tolerance=separation_tolerance)
    )
    return ArmAttribution(
        arm=arm,
        graph=graph,
        result=result,
        separation_tolerance_blocks=separation_tolerance,
        candidate_surface_truth=_scored(
            graph.edges, left, right, tolerance=surface_tolerance),
        selected_surface_truth=_scored(
            selected, left, right, tolerance=surface_tolerance),
        candidate_arm_truth=candidate_arm,
        selected_arm_truth=selected_arm,
    )


def attribute_pair(
    left,
    right,
    left_camera: CameraEstimate,
    right_camera: CameraEstimate,
    *,
    matcher: JointMatchSettings | None = None,
    gates: PairGates | None = None,
    ratio_test: float = _RATIO_TEST,
    surface_tolerance: float = SURFACE_TOLERANCE,
    point_tolerances: Sequence[float] = POINT_TOLERANCES,
) -> ExactPointPairAttribution:
    """Run every arm on one pair through the unchanged production matcher.

    The production baseline runs first on its own graph, before any injection.
    Then the reviewed DS-R5K same-surface graph and each point-oracle graph are
    injected for exactly one ``match_feature_frames`` call apiece; the production
    graph builder is restored by the reusable context manager even if a matcher
    call raises. Joint settings, gates and the ratio test are byte-identical in
    every arm.
    """
    bounds = JointMatchSettings() if matcher is None else matcher
    production_graph = reciprocal_topk_graph(
        left.features.descriptors,
        right.features.descriptors,
        neighbors=int(bounds.neighbors),
        max_edges=int(bounds.max_candidate_edges),
        distance_ceiling=bounds.graph_distance_ceiling,
    )
    surface_graph = R5K.build_oracle_graph(
        left.features, right.features, left.hits, right.hits,
        tolerance=surface_tolerance,
    )
    point_graphs: dict[str, ReciprocalGraph] = {}
    for tolerance in point_tolerances:
        point_graphs[point_arm(tolerance)] = build_point_oracle_graph(
            left.features, right.features, left.hits, right.hits,
            tolerance=tolerance,
        )

    production_result = match_feature_frames(
        left.features, right.features, left_camera, right_camera,
        matcher=bounds, gates=gates, ratio_test=ratio_test,
    )
    results: dict[str, JointPairResult] = {ARM_PRODUCTION: production_result}
    injected = [ARM_SURFACE] + [point_arm(t) for t in point_tolerances]
    graphs: dict[str, ReciprocalGraph] = {ARM_SURFACE: surface_graph, **point_graphs}
    for name in injected:
        with injected_candidate_graph(graphs[name]):
            results[name] = match_feature_frames(
                left.features, right.features, left_camera, right_camera,
                matcher=bounds, gates=gates, ratio_test=ratio_test,
            )

    arms: list[ArmAttribution] = [
        _arm_attribution(
            ARM_PRODUCTION, production_graph, results[ARM_PRODUCTION], left, right,
            separation_tolerance=None, surface_tolerance=surface_tolerance,
        ),
        _arm_attribution(
            ARM_SURFACE, surface_graph, results[ARM_SURFACE], left, right,
            separation_tolerance=float(surface_tolerance),
            surface_tolerance=surface_tolerance,
        ),
    ]
    for tolerance in point_tolerances:
        name = point_arm(tolerance)
        arms.append(_arm_attribution(
            name, point_graphs[name], results[name], left, right,
            separation_tolerance=float(tolerance),
            surface_tolerance=surface_tolerance,
        ))

    with_truth = sum(1 for hit in left.hits if hit is not None)
    right_with_truth = sum(1 for hit in right.hits if hit is not None)
    point_truth_pairs = tuple(
        (float(tolerance),
         len(construct_truth_pairs(left.hits, right.hits, tolerance=float(tolerance))))
        for tolerance in point_tolerances
    )
    return ExactPointPairAttribution(
        left_id=production_result.left_id,
        right_id=production_result.right_id,
        left_points=np.asarray(left.features.points, dtype=np.float64),
        right_points=np.asarray(right.features.points, dtype=np.float64),
        left_camera=left_camera,
        right_camera=right_camera,
        arms=tuple(arms),
        surface_truth_pairs=len(construct_truth_pairs(
            left.hits, right.hits, tolerance=surface_tolerance)),
        point_truth_pairs=point_truth_pairs,
        left_keypoints_with_truth=with_truth,
        right_keypoints_with_truth=right_with_truth,
        left_keypoints_without_truth=len(left.hits) - with_truth,
        right_keypoints_without_truth=len(right.hits) - right_with_truth,
    )


def _gates_record(result: JointPairResult) -> dict[str, Any]:
    return {
        "median_selected_residual_degrees": result.median_selected_residual_degrees,
        "descriptor_cost": result.descriptor_cost,
        "parallax_degrees": result.parallax_degrees,
        "cheirality_fraction": result.cheirality_fraction,
        "singular_ratio": result.singular_ratio,
        "ambiguity_margin": result.ambiguity_margin,
    }


def _absent_axis_stage() -> dict[str, Any]:
    """Typed unavailable plane-graph/axis record for this pair-level seam.

    This diagnostic stops at the unchanged pair matcher/solver; it never runs the
    production plane-graph or placement stages, so their axis coverage, resultants
    and residuals do not exist here. They are reported as explicitly unavailable,
    never as zero.
    """
    return {
        "available": False,
        "reason": "pair_level_diagnostic_has_no_plane_graph_record",
        "axis_coverage": None,
        "resultants": None,
        "residuals": None,
    }


def _absent_placement_stage() -> dict[str, Any]:
    """Typed unavailable placed-cell geometry record for this pair-level seam."""
    return {
        "available": False,
        "reason": "pair_level_diagnostic_has_no_placement_record",
        "placed_cells": None,
        "placed_cell_count": None,
    }


def arm_record(
    arm_attribution: ArmAttribution,
    left_points: np.ndarray,
    right_points: np.ndarray,
    left_camera: CameraEstimate,
    right_camera: CameraEstimate,
) -> dict[str, Any]:
    """One arm's graph, gate, pose, hypothesis and stage-geometry record."""
    result = arm_attribution.result
    selection_state, pose_state, matches_available, pose_available = describe_outcome(
        result)
    pose = result.pose
    return {
        "arm": arm_attribution.arm,
        "graph_mode": (
            "production_reciprocal_topk_graph"
            if arm_attribution.arm == ARM_PRODUCTION
            else ORACLE_GRAPH_MODE
        ),
        "separation_tolerance_blocks": arm_attribution.separation_tolerance_blocks,
        "graph_edges": int(arm_attribution.graph.edges.shape[0]),
        "candidate_edges": int(result.candidate_edges),
        "total_graph_edges": int(result.total_graph_edges),
        "truncated": int(result.truncated),
        "raw_keypoints": {
            "left": int(result.raw_keypoints[0]),
            "right": int(result.raw_keypoints[1]),
        },
        "boundary_dropped": {
            "left": int(result.boundary_dropped[0]),
            "right": int(result.boundary_dropped[1]),
        },
        "ratio_survivors": {
            "left": int(result.ratio_survivors[0]),
            "right": int(result.ratio_survivors[1]),
        },
        "mutual_survivors": int(result.mutual_survivors),
        "support": int(result.support),
        "support_floor": int(result.support_floor),
        "selected_support": int(result.support),
        "selected_support_edges": [
            [int(left), int(right)]
            for left, right in _selected_indices(result, left_points, right_points)
        ],
        "spatial_bins": {
            "left": int(result.spatial_bins_left),
            "right": int(result.spatial_bins_right),
            "total": int(result.spatial_bins_total),
        },
        "status": str(result.status),
        "reason": str(result.reason),
        "selection_state": selection_state,
        "pose_state": pose_state,
        "matches_available": matches_available,
        "pose_available": pose_available,
        "hypothesis_count": len(result.hypotheses),
        "gates": _gates_record(result),
        "pose": _pose_record(result),
        "pose_class": None if pose is None else pose.right_yaw_class,
        "pose_classes": (
            [] if pose is None else [yaw_class_record(item) for item in pose.classes]
        ),
        "pose_error": pose_error_record(result, left_camera, right_camera),
        "hypotheses": [hypothesis_record(item) for item in result.hypotheses],
        "candidate_stage_surface_truth": geometry_record(
            arm_attribution.candidate_surface_truth),
        "selected_stage_surface_truth": geometry_record(
            arm_attribution.selected_surface_truth),
        "candidate_stage_arm_tolerance": (
            None if arm_attribution.candidate_arm_truth is None
            else geometry_record(arm_attribution.candidate_arm_truth)
        ),
        "selected_stage_arm_tolerance": (
            None if arm_attribution.selected_arm_truth is None
            else geometry_record(arm_attribution.selected_arm_truth)
        ),
    }


def pair_record(pair: ExactPointPairAttribution) -> dict[str, Any]:
    """One pair's six-arm ledger, denominators, censoring and typed absences."""
    record: dict[str, Any] = {
        "left_frame_id": pair.left_id,
        "right_frame_id": pair.right_id,
        "surface_truth_pairs": pair.surface_truth_pairs,
        "point_truth_pairs": {
            tolerance_label(tolerance): count
            for tolerance, count in pair.point_truth_pairs
        },
        "left_keypoints_with_truth": pair.left_keypoints_with_truth,
        "right_keypoints_with_truth": pair.right_keypoints_with_truth,
        "left_keypoints_without_truth": pair.left_keypoints_without_truth,
        "right_keypoints_without_truth": pair.right_keypoints_without_truth,
        "arms": list(ARM_ORDER),
        "axis_stage": _absent_axis_stage(),
        "placed_cell_geometry": _absent_placement_stage(),
    }
    for name in ARM_ORDER:
        record[name] = arm_record(
            pair.arm(name), pair.left_points, pair.right_points,
            pair.left_camera, pair.right_camera,
        )
    record["tolerance_sensitivity"] = {
        tolerance_label(tolerance): {
            "arm": point_arm(tolerance),
            "separation_tolerance_blocks": float(tolerance),
            "point_truth_pairs": count,
            "oracle_solved": pair.arm(point_arm(tolerance)).solved,
            "baseline_solved": pair.production_arm.solved,
            "surface_solved": pair.surface_arm.solved,
        }
        for tolerance, count in pair.point_truth_pairs
    }
    return record


def _counter(values: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _stage_totals(arms: Sequence[ArmAttribution], attribute: str) -> dict[str, Any]:
    geometries = [getattr(item, attribute) for item in arms]
    if any(geometry is None for geometry in geometries):
        return {"available": False, "reason": "arm_has_no_arm_tolerance_stage"}
    return _totals_record(_geometry_totals(geometries))


def summarise_attributions(
    pairs: Sequence[ExactPointPairAttribution],
) -> dict[str, Any]:
    """Pooled per-arm stage totals, solve outcome and tolerance sensitivity."""
    arms: dict[str, Any] = {}
    for name in ARM_ORDER:
        arm_attributions = [pair.arm(name) for pair in pairs]
        results = [item.result for item in arm_attributions]
        arms[name] = {
            "graph_mode": (
                "production_reciprocal_topk_graph" if name == ARM_PRODUCTION
                else ORACLE_GRAPH_MODE
            ),
            "separation_tolerance_blocks": (
                arm_attributions[0].separation_tolerance_blocks if arm_attributions
                else None
            ),
            "oracle_arm": name != ARM_PRODUCTION,
            "solved": sum(1 for result in results if result.status == "solved"),
            "refusal_reasons": dict(sorted(_counter([
                result.reason for result in results if result.status != "solved"
            ]).items())),
            "selection_states": dict(sorted(_counter([
                describe_outcome(result)[0] for result in results
            ]).items())),
            "pose_states": dict(sorted(_counter([
                describe_outcome(result)[1] for result in results
            ]).items())),
            "graph_edges_total": sum(
                int(item.graph.edges.shape[0]) for item in arm_attributions),
            "graph_empty_pairs": sum(
                1 for item in arm_attributions if item.graph_empty),
            "pairs_with_a_candidate_mode": sum(
                1 for result in results if len(result.hypotheses) > 0),
            "pairs_with_selected_support": sum(
                1 for result in results if int(result.support) > 0),
            "support_total": sum(int(result.support) for result in results),
            "support_floor_min": (
                min(int(result.support_floor) for result in results) if results else None
            ),
            "support_floor_max": (
                max(int(result.support_floor) for result in results) if results else None
            ),
            "candidate_stage_surface_truth": _totals_record(_geometry_totals(
                [item.candidate_surface_truth for item in arm_attributions])),
            "selected_stage_surface_truth": _totals_record(_geometry_totals(
                [item.selected_surface_truth for item in arm_attributions])),
            "candidate_stage_arm_tolerance": _stage_totals(
                arm_attributions, "candidate_arm_truth"),
            "selected_stage_arm_tolerance": _stage_totals(
                arm_attributions, "selected_arm_truth"),
        }

    point_reach = {
        tolerance_label(tolerance): {
            "arm": point_arm(tolerance),
            "separation_tolerance_blocks": float(tolerance),
            "point_truth_pairs": sum(
                count for item in pairs
                for tolerance_value, count in item.point_truth_pairs
                if tolerance_value == float(tolerance)),
            "candidate_reach": arms[point_arm(tolerance)][
                "candidate_stage_surface_truth"]["truth_pairs_returned"],
            "selected_reach": arms[point_arm(tolerance)][
                "selected_stage_surface_truth"]["truth_pairs_returned"],
            "solved": arms[point_arm(tolerance)]["solved"],
        }
        for tolerance in POINT_TOLERANCES
    }
    surface_candidate = arms[ARM_SURFACE]["candidate_stage_surface_truth"]
    baseline_candidate = arms[ARM_PRODUCTION]["candidate_stage_surface_truth"]
    summary: dict[str, Any] = {
        "pair_count": len(pairs),
        "surface_truth_pairs": sum(pair.surface_truth_pairs for pair in pairs),
        "pairs_with_surface_truth": sum(
            1 for pair in pairs if pair.surface_truth_pairs > 0),
        "pairs_with_empty_surface_truth": sum(
            1 for pair in pairs if pair.surface_truth_pairs == 0),
        "left_keypoints_with_truth": sum(
            pair.left_keypoints_with_truth for pair in pairs),
        "right_keypoints_with_truth": sum(
            pair.right_keypoints_with_truth for pair in pairs),
        "left_keypoints_without_truth": sum(
            pair.left_keypoints_without_truth for pair in pairs),
        "right_keypoints_without_truth": sum(
            pair.right_keypoints_without_truth for pair in pairs),
        "point_truth_pairs": {
            tolerance_label(tolerance): sum(
                count for item in pairs
                for tolerance_value, count in item.point_truth_pairs
                if tolerance_value == float(tolerance))
            for tolerance in POINT_TOLERANCES
        },
        "arms": arms,
        "tolerance_sensitivity": point_reach,
        "ceilings": {
            "production_candidate_reach": baseline_candidate["truth_pairs_returned"],
            "surface_oracle_candidate_reach": surface_candidate["truth_pairs_returned"],
            "same_point_oracle_candidate_reach": {
                tolerance_label(tolerance): point_reach[
                    tolerance_label(tolerance)]["candidate_reach"]
                for tolerance in POINT_TOLERANCES
            },
        },
        "baseline_vs_oracle": {
            "surface_solved_not_baseline": sum(
                1 for pair in pairs
                if pair.surface_arm.solved and not pair.production_arm.solved),
            "baseline_solved_not_surface": sum(
                1 for pair in pairs
                if pair.production_arm.solved and not pair.surface_arm.solved),
            "point_solved_where_surface_refused": {
                tolerance_label(tolerance): sum(
                    1 for pair in pairs
                    if pair.arm(point_arm(tolerance)).solved
                    and not pair.surface_arm.solved)
                for tolerance in POINT_TOLERANCES
            },
            "surface_solved_where_point_refused": {
                tolerance_label(tolerance): sum(
                    1 for pair in pairs
                    if pair.surface_arm.solved
                    and not pair.arm(point_arm(tolerance)).solved)
                for tolerance in POINT_TOLERANCES
            },
        },
        "same_surface_not_same_feature_point": SAME_SURFACE_NOT_SAME_FEATURE_POINT,
        "exact_point_tolerance_caveat": EXACT_POINT_CAVEAT,
    }
    summary["positive_identity_mismatch"] = {
        name: {
            "candidate_stage_surface_truth": (
                arms[name]["candidate_stage_surface_truth"]["positive"]
                - arms[name]["candidate_stage_surface_truth"]["truth_pairs_returned"]
            ),
            "selected_stage_surface_truth": (
                arms[name]["selected_stage_surface_truth"]["positive"]
                - arms[name]["selected_stage_surface_truth"]["truth_pairs_returned"]
            ),
        }
        for name in ARM_ORDER
    }
    return summary


def _normalised_path_key(path: str | Path) -> str:
    """Absolute, lexically-collapsed, case-folded key; never opens or resolves."""
    absolute = os.path.normpath(os.path.abspath(os.fspath(path)))
    return absolute.replace("\\", "/").casefold()


def output_targets_a_sealed_artifact(path: str | Path) -> bool:
    """Whether ``path`` names one of the reviewed sealed artifacts.

    Windows paths are case-insensitive, so a mixed-case spelling resolves to the
    same file; both operands are normalised without touching the filesystem.
    """
    key = _normalised_path_key(path)
    return any(key.endswith(name.casefold()) for name in SEALED_ARTIFACTS)


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


def build_payload(
    exports_root: str,
    entries: Sequence[DatasetSceneInput],
    manifest: Mapping[str, Any],
    canonical_entry: DatasetSceneInput,
    *,
    dataset_id: str | None = None,
    matcher: JointMatchSettings | None = None,
    surface_tolerance: float = SURFACE_TOLERANCE,
    point_tolerances: Sequence[float] = POINT_TOLERANCES,
    pin_record: Mapping[str, Any] | None = None,
    scene_pin_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the six-arm diagnostic over the canonical export and seal the payload.

    The caller is responsible for the strict pin/leakage preflight; the pin
    records are stored verbatim for audit.
    """
    bounds = JointMatchSettings() if matcher is None else matcher
    detection = detection_settings(bounds)
    scene = build_scene(canonical_entry)
    observations = [observe_oracle_frame(scene, frame, detection=detection)
                    for frame in scene.frames]
    cameras = [oracle_camera_estimate(frame) for frame in scene.frames]
    pairs: list[ExactPointPairAttribution] = []
    for left_index in range(len(observations)):
        for right_index in range(left_index + 1, len(observations)):
            pairs.append(attribute_pair(
                observations[left_index],
                observations[right_index],
                cameras[left_index],
                cameras[right_index],
                matcher=bounds,
                ratio_test=detection.ratio_test,
                surface_tolerance=surface_tolerance,
                point_tolerances=point_tolerances,
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
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete",
        "created_for": "DS-R5O",
        "tool": "scripts/evaluation/evaluate_exact_point_oracle.py",
        "evaluation_only": True,
        "promotion_or_production_change": False,
        "sealed_artifacts_untouched": list(SEALED_ARTIFACTS),
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
                "only; the assisted CameraEstimate records are consumed solely by the "
                "matcher's geometric stage"),
            "label": "assisted_oracle_camera",
        },
        "oracle_policy": {
            "mode": ORACLE_GRAPH_MODE,
            "assisted": True,
            "truth_assisted_upper_bound": True,
            "production_result": False,
            "surface_oracle_definition": (
                "the reviewed DS-R5K candidate graph: unique same-block-face keypoint "
                "pixel pairs of the frozen DS-R5I rule whose raycast world surface "
                "points are within SURFACE_TOLERANCE_BLOCKS = 1/16 block"),
            "point_oracle_definition": (
                "the same-surface continuous world-hit proximity rule: both detector "
                "keypoints raycast through the frozen actual camera into the same "
                "visible surface (same block coordinate and face slot) and their "
                "retained continuous hit points are within the arm's preregistered "
                "separation tolerance; this is proximity, not literal feature "
                "identity, and the point_oracle_* key is a legacy label"),
            "definitions_are_equivalent_at_the_surface_tolerance": True,
            "tolerance_sweep_blocks": [float(t) for t in point_tolerances],
            "tolerance_selection": (
                "fixed and preregistered; never selected from solve success and never "
                "a production threshold; a single-fixture sensitivity sweep only, "
                "which cannot isolate correspondence ambiguity or localize one "
                "remaining bottleneck because candidate count, achieved surface "
                "precision and the arm's own proximity denominator move together"),
            "distances": (
                "each oracle edge keeps the actual L2 descriptor distance of its "
                "mapped keypoint pair; no descriptor or distance is fabricated"),
            "selector": (
                "the unchanged production match_feature_frames geometric selector, "
                "settings, gates and ratio test run on every injected graph"),
            "injection": (
                "mc_photogrammetry.pipeline.joint_matching.reciprocal_topk_graph is "
                "replaced for exactly one matcher call per oracle arm and restored in "
                "a finally block; the production baseline arm runs before injection"),
            "same_surface_not_same_feature_point": (
                SAME_SURFACE_NOT_SAME_FEATURE_POINT),
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
            "surface_tolerance_blocks": float(surface_tolerance),
            "point_tolerances_blocks": [float(t) for t in point_tolerances],
            "arms": list(ARM_ORDER),
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
            "r5k_pin_verification": dict(pin_record) if pin_record else {},
            "r5k_scene_pin_verification": (
                dict(scene_pin_record) if scene_pin_record else {}),
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
        "stage_availability": {
            "pair_solve": {
                "available": True,
                "source": "unchanged production match_feature_frames",
            },
            "plane_graph_axis_stage": {
                "available": False,
                "reason": "pair_level_diagnostic_has_no_plane_graph_record",
            },
            "placed_cell_geometry": {
                "available": False,
                "reason": "pair_level_diagnostic_has_no_placement_record",
            },
        },
        "renderer_geometry": {
            **geometry_policy_record(),
            "camera_policy_mode": CAMERA_POLICY_MODE,
            "camera_policy_assisted": True,
            "oracle_graph_mode": ORACLE_GRAPH_MODE,
            "arms": list(ARM_ORDER),
            "point_tolerances_blocks": [float(t) for t in point_tolerances],
            "frames": len(scene.frames),
            "keypoints": keypoints_total,
            "boundary_dropped": boundary_dropped_total,
            "keypoints_with_truth": keypoints_with_truth_total,
            "keypoints_without_truth": keypoints_total - keypoints_with_truth_total,
            "no_truth_reason_totals": dict(sorted(no_truth_totals.items())),
        },
        "consistency": {
            "positive_identity_mismatch": summary["positive_identity_mismatch"],
            "ok": all(
                value == 0
                for stages in summary["positive_identity_mismatch"].values()
                for value in stages.values()),
        },
        "limitations": list(LIMITATIONS),
    }
    payload["seal_sha256"] = seal_record(payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    output = Path(args.output)
    if output_targets_a_sealed_artifact(output):
        print(f"refusing to touch a sealed artifact {output}", file=sys.stderr)
        return 2
    if output.exists():
        print(f"refusing to overwrite {output}", file=sys.stderr)
        return 2
    try:
        entries = discover_finalized_entries(args.exports_root)
        pin_record = verify_r5k_pins(entries)
        manifest = build_group_manifest(entries)
        canonical = select_canonical_entry(entries, dataset_id=args.dataset_id)
        scene_pin_record = verify_canonical_scene_pins(canonical)
        payload = build_payload(
            args.exports_root, entries, manifest, canonical,
            dataset_id=args.dataset_id,
            pin_record=pin_record,
            scene_pin_record=scene_pin_record,
        )
    except Exception as error:  # noqa: BLE001 - one-shot runner reports and exits
        print(f"evaluation failed before writing {output}: {error}", file=sys.stderr)
        return 1
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = payload["summary"]
    arms = summary["arms"]
    print(f"canonical={payload['scene']['dataset_id']} "
          f"frames={payload['scene']['frame_count']} pairs={summary['pair_count']}")
    print(f"surface_truth_pairs={summary['surface_truth_pairs']} "
          f"baseline_solved={arms[ARM_PRODUCTION]['solved']} "
          f"surface_solved={arms[ARM_SURFACE]['solved']}")
    for tolerance in POINT_TOLERANCES:
        name = point_arm(tolerance)
        print(f"{name} point_truth={summary['point_truth_pairs'][tolerance_label(tolerance)]} "
              f"solved={arms[name]['solved']} "
              f"candidate_reach={arms[name]['candidate_stage_surface_truth']['truth_pairs_returned']} "
              f"selected_reach={arms[name]['selected_stage_surface_truth']['truth_pairs_returned']}")
    print(f"wrote {output} seal={payload['seal_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
