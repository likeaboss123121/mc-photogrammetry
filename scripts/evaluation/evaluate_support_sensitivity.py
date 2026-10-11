"""Evaluation-only oracle support-floor sensitivity diagnostic (DS-R5N).

DS-R5K ran the unchanged production
:func:`mc_photogrammetry.pipeline.joint_matching.match_feature_frames` on a
truth-filtered oracle candidate graph and found that the graph reached all 1,213
Canal Town truth pixel pairs while the selector solved only four pairs. Its
report section 5.2 identified a hard support-reachability ceiling: 19/28 pairs
were refused as ``insufficient_one_to_one_support`` because the whole oracle
surface-pair edge count was below the production support floor
``max(8, ceil(0.02 * min(raw_keypoints)))``, and the four ``solve_pair_refused``
pairs had no serialized DS-11 per-yaw ``YawClassScore`` evidence at all.

DS-R5N is a separate **evaluation-only** diagnostic that carries the same
canonical 8-view Canal Town export, the same immutable DS-R5K input pins, the
same strict discovery/leakage preflight and the same oracle surface-pair graph,
then runs three arms per unordered frame pair:

1. ``production_baseline`` - the production reciprocal descriptor graph with the
   unchanged production :class:`JointMatchSettings` (retained DS-R5K context);
2. ``oracle_default`` - the oracle truth-surface graph with the unchanged
   production settings, i.e. the DS-R5K oracle arm; and
3. ``oracle_control`` - the **oracle-only** control
   ``min_support_absolute = 8, min_support_fraction = 0.0``.

The control is a deliberate counterfactual, never a candidate production change:
it only asks how many currently unreachable pairs gain a candidate mode, a
one-to-one selected support set and a DS-11 solve when the *fractional* half of
the support floor is removed. Because the oracle edges are strict
same-world-surface pixel pairs and **same surface is not the same feature
point**, a control-arm solve is not evidence of a correct feature
correspondence. No threshold, gate, matcher, solver, production, project or
promotion setting is changed, and only a human may authorize any corpus run.

The sealed control run for this diagnostic is reserved at
``.tmp-oracle-support-sensitivity-20261003.json``. This module never reads,
writes or modifies the sealed DS-R5K artifact
``.tmp-oracle-candidate-graph-20261003.json``.

Per pair the sealed record serializes, for every arm: candidate graph edges,
the computed support floor, raw keypoints, one-to-one selected support, spatial
bins, every ranked joint hypothesis, every DS-11 ``solve_pair`` ``YawClassScore``
field (when the pose stage ran), the resolved pose class, and the
assisted-camera ground-truth yaw/baseline pose error where technically
well-defined. It carries no pixel payload.
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
    discover_finalized_entries,
    geometry_policy_record,
    match_indices,
    score_pair_matches,
    seal_record,
)
from mc_photogrammetry.pipeline.connected_camera import PairGates, YawClassScore
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
from mc_photogrammetry.reconstruction.camera import CameraModel, camera_rotation_matrix

__all__ = [
    "ARM_CONTROL",
    "ARM_DEFAULT",
    "ARM_ORDER",
    "ARM_PRODUCTION",
    "CONTROL_SETTINGS",
    "CONTROL_SUPPORT_ABSOLUTE",
    "CONTROL_SUPPORT_FRACTION",
    "DEFAULT_ARM",
    "DEFAULT_EXPORTS_ROOT",
    "DEFAULT_OUTPUT",
    "DEFAULT_SETTINGS",
    "LIMITATIONS",
    "R5K_CANONICAL_DATASET_ID",
    "R5K_CANONICAL_SCENE_FILE_PINS",
    "R5K_CANONICAL_SCREENSHOT_PINS",
    "R5K_INDEX_PINS",
    "R5K_SEALED_ARTIFACT",
    "SAME_SURFACE_NOT_SAME_FEATURE_POINT",
    "SCHEMA",
    "UNREACHABLE_REASON",
    "SupportPairAttribution",
    "SupportSensitivityError",
    "angle_between",
    "arm_record",
    "attribute_pair",
    "build_payload",
    "control_settings",
    "default_settings",
    "load_r5k_module",
    "main",
    "output_targets_sealed_r5k_artifact",
    "pair_record",
    "parse_args",
    "pose_error_record",
    "reference_yaw_class",
    "summarise_attributions",
    "verify_canonical_scene_pins",
    "verify_r5k_pins",
    "yaw_class_record",
]

SCHEMA = "mc-photogrammetry-oracle-support-sensitivity-v1"
"""Sealed schema of this evaluation-only support-floor sensitivity diagnostic."""

DEFAULT_EXPORTS_ROOT = "exports/datasets"
DEFAULT_OUTPUT = ".tmp-oracle-support-sensitivity-20261003.json"
"""Reserved sealed output path of this diagnostic; never the DS-R5K artifact."""

R5K_SEALED_ARTIFACT = ".tmp-oracle-candidate-graph-20261003.json"
"""The immutable DS-R5K artifact this evaluation must never read or write."""

ARM_PRODUCTION = "production_baseline"
ARM_DEFAULT = "oracle_default"
ARM_CONTROL = "oracle_control"
ARM_ORDER = (ARM_PRODUCTION, ARM_DEFAULT, ARM_CONTROL)
"""The three per-pair arms: retained context, DS-R5K oracle, and the control."""

DEFAULT_ARM = ARM_DEFAULT
"""The default joint-settings arm the control is compared against."""

CONTROL_SUPPORT_ABSOLUTE = 8
CONTROL_SUPPORT_FRACTION = 0.0
"""The oracle-only control support floor: the absolute production floor, no fraction."""

UNREACHABLE_REASON = "insufficient_one_to_one_support"
"""The DS-R5K refusal class the control is designed to probe."""

SAME_SURFACE_NOT_SAME_FEATURE_POINT = (
    "A strict same-world-surface oracle edge means its two endpoints observe the "
    "same block face within the preregistered tolerance; it does NOT mean they are "
    "the same feature point. Distinct keypoints on one surface are different "
    "physical points. Relaxing the support floor therefore admits "
    "same-surface-but-different-point correspondences, so a control-arm candidate "
    "mode, selection or solve is not evidence of a correct feature correspondence "
    "and must never be read as a reason to lower the production support floor."
)
"""The caveat that governs every control-arm reading of this diagnostic."""

R5K_CANONICAL_DATASET_ID = "fresh-c4-canal-town-framed-2026-09-12"
"""The DS-R5K selected canonical 8-view Canal Town export."""

R5K_INDEX_PINS: Mapping[str, str] = {
    "fresh-c4-canal-town-framed-2026-09-12":
        "0637caba1a523bbdf7351dedd7c4f93bfdb611f0b549fc54b3a781f3d25ee46b",
    "fresh-c4-canal-town-2026-09-12":
        "d3dc69b048bf8b827d75df402aa041445678ae63ba5124e6eefc6586278413dd",
    "recon-canal-houses-2026-09-12":
        "5d0db614b1802bd7fcd047c390dd4637eb20dcd88f550d76a624d3bf6ba6fae9",
    "pilot-batch-train-station-2026-09-12":
        "4ef40f8be5663108036bd3827e298286c93747a71ed9bbb4dcd2a3266ecf4837",
    "smoke-train-station-ds20c-2026-09-12":
        "a2a21eb49d4b7664f603eaaca77bf8a8095defee520a2c64dc2592fb1625a3e0",
}
"""Immutable DS-R5K index pins for every discovered export (dataset_id -> sha256)."""

R5K_CANONICAL_SCENE_FILE_PINS: Mapping[str, str] = {
    "capture.json":
        "c6c44d43408779006854c81742a68d00cfe5b546f506ffa08ea59ba7b812abe6",
    "ground_truth.json":
        "36114feebe61d7b358927095a159d56dc5762e02c5a52a52e8a12ae17ce4a35a",
    "plan.json":
        "70fcb26bd49e37ca241ba89dcdb22128e859d821ad742419d0bcb5e158a593bc",
    "recipe.json":
        "10d415c85f5f8f7b7ab5f8e9a3c8d78bad3e7a0233b2971bc563f70eefa8ca4a",
    "reference.litematic":
        "f2601f55aee8156da8208aa7225c7406cdd67b6f5f078b312ed4d8c89c3eaa74",
}
"""Immutable DS-R5K scene-file pins for the canonical Canal Town export."""

R5K_CANONICAL_SCREENSHOT_PINS = frozenset({
    "48ba0ff58c77b588acd5aa06a2c630fc2f5103c6d0d67b45b4bc21b81ecf533c",
    "8750bb197329eb50144896264db40eef70a6c44cc3dce60eac3a4f1e9aa79696",
    "2d017c8f804999446f54dbc742fb87d7748af76ad50249c0e2853754c55b6a95",
    "e71bfda3ece7a0b74d2da2fd62834645fcbcf662fb4b5889fae95fa26c9a09e9",
    "8797d70e0e09475d296c3cf18e5892f6b22d41750ed18d6c3364da80ba60a81f",
    "6f844d5e3600e853bf3045fc25b55a271c0b665226053602b6d30c945f0f016b",
    "53c38526d378a0597ab6e00fb5e01a6c7aa0e2dc11619c1b7eaa3cfc0ccf8923",
    "fdd8a3709479795b204f67153b10f44b49e85f2bc0050a46d9fdd0461f025556",
})
"""Immutable DS-R5K screenshot pins for the canonical 8-view export."""

GAUGE_ROTATIONS = (0, 90, 180, 270)
"""The four-fold world yaw gauge a single-view pose leaves free."""

_RATIO_TEST = CorrespondenceSettings().ratio_test
"""The production detection ratio test; ``JointMatchSettings`` has no copy."""

_R5K_SCRIPT = Path(__file__).with_name("evaluate_oracle_candidate_graph.py")


class SupportSensitivityError(ValueError):
    """Invalid, unverifiable or inconsistent evaluation input."""


def load_r5k_module(path: str | Path | None = None):
    """Load the sealed DS-R5K evaluation module by path (never its artifact).

    The DS-R5K script owns the reviewed oracle graph, canonical selection and
    image observation helpers; this diagnostic reuses those objects directly so
    the two evaluations can never drift apart. Loading the script does not touch
    ``.tmp-oracle-candidate-graph-20261003.json`` or run its ``main``.
    """
    script = _R5K_SCRIPT if path is None else Path(path)
    spec = importlib.util.spec_from_file_location("ds_r5n_r5k_oracle_graph", script)
    if spec is None or spec.loader is None:
        raise SupportSensitivityError(f"cannot load the DS-R5K module from {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


R5K = load_r5k_module()

# Reviewed DS-R5K machinery reused verbatim by this diagnostic.
select_canonical_entry = R5K.select_canonical_entry
canonical_entry_record = R5K.canonical_entry_record
build_oracle_graph = R5K.build_oracle_graph
injected_candidate_graph = R5K.injected_candidate_graph
oracle_camera_estimate = R5K.oracle_camera_estimate
detection_settings = R5K.detection_settings
describe_outcome = R5K.describe_outcome
hypothesis_record = R5K.hypothesis_record
geometry_record = R5K.geometry_record
frame_record = R5K.frame_record
camera_record = R5K.camera_record
_pose_record = R5K._pose_record
_geometry_totals = R5K._geometry_totals
_totals_record = R5K._totals_record

CAMERA_POLICY_MODE = R5K.CAMERA_POLICY_MODE
ORACLE_GRAPH_MODE = R5K.ORACLE_GRAPH_MODE


def default_settings() -> JointMatchSettings:
    """The unchanged production joint settings (retained DS-R5K baseline)."""
    return JointMatchSettings()


def control_settings() -> JointMatchSettings:
    """The oracle-only control: production absolute floor, no fractional floor."""
    return JointMatchSettings(
        min_support_absolute=CONTROL_SUPPORT_ABSOLUTE,
        min_support_fraction=CONTROL_SUPPORT_FRACTION,
    )


DEFAULT_SETTINGS = default_settings()
CONTROL_SETTINGS = control_settings()

LIMITATIONS = (
    "This is an evaluation-only, truth-assisted counterfactual. The oracle "
    "candidate graph is an UPPER BOUND, not a production result and not a "
    "model-performance estimate.",
    SAME_SURFACE_NOT_SAME_FEATURE_POINT,
    "The control setting min_support_absolute=8, min_support_fraction=0.0 is "
    "applied to the oracle graph ONLY. It is a measurement instrument, not a "
    "recommendation; no promotion, production threshold, gate, matcher, solver or "
    "project setting may be changed from this diagnostic.",
    "The assisted oracle camera uses the recorded factory capture metadata (pose, "
    "field of view, frame resolution). Normal reconstruction never sees it and "
    "this diagnostic makes no image-derived calibration claim.",
    "Only the single canonical Canal Town 8-view export is evaluated; one export, "
    "one run. Every index split is 'train' and only three topology groups exist, "
    "so no generalization claim is made.",
    "Joint hypotheses and oracle truth pairs are not the same object: precision "
    "and recall are scored against an evaluator-built unique-pixel-pair surface "
    "model, not against independent feature-point labels.",
    "Partial block models (bottom/top slabs, stairs, trapdoors, fences, walls, "
    "panes, plants, fluids) are unavailable without the vanilla model assets; "
    "they occlude as full cubes but are excluded from truth ownership.",
    "No texture identity is available anywhere in the pipeline: the block texture "
    "is never read, hashed or compared. same_state_same_slot_other_block is only "
    "an appearance proxy and is never texture ground truth.",
    "Truth pairs and every returned identity are deduplicated by unique pixel "
    "pair; repeated descriptor keypoints at exactly the same pixel are one "
    "identity, and distinct pixels are never collapsed.",
    "The oracle edges keep the actual detected descriptor distances, but the "
    "descriptor graph itself is replaced, so oracle reach is an upper bound on "
    "what the descriptor stage could ever supply.",
    "The production reciprocal graph builder is restored in a finally block after "
    "every oracle call; the production baseline arm never uses the injected graph.",
)


def observe_oracle_frame(
    scene: FinalizedScene,
    frame: FrameInput,
    *,
    detection: CorrespondenceSettings | None = None,
):
    """Reuse the DS-R5K joint feature detection plus truth-surface resolution."""
    return R5K.observe_oracle_frame(scene, frame, detection=detection)


def _ry_degrees(degrees: float) -> np.ndarray:
    """``Ry(degrees)`` about the world Y axis (the matcher's yaw gauge)."""
    angle = math.radians(float(degrees))
    cos, sin = math.cos(angle), math.sin(angle)
    return np.asarray(
        ((cos, 0.0, sin), (0.0, 1.0, 0.0), (-sin, 0.0, cos)), dtype=np.float64
    )


def _camera_rotation(camera: CameraEstimate) -> np.ndarray | None:
    """The world-to-camera rotation of an assisted estimate, or ``None``."""
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


def angle_between(first: np.ndarray, second: np.ndarray) -> float:
    """Geodesic angle in degrees between two rotation matrices."""
    trace = np.clip((np.trace(first @ second.T) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def reference_yaw_class(
    left_camera: CameraEstimate, right_camera: CameraEstimate
) -> int | None:
    """Ground-truth right yaw quadrant from the assisted camera rotations.

    The matcher's class-``k`` hypothesis is
    ``R_right @ Ry(90 k) @ R_left.T``; the assisted cameras are the recorded
    factory poses, so their relative rotation fixes the quadrant. ``None`` when
    either assisted rotation is unavailable.
    """
    left_rotation = _camera_rotation(left_camera)
    right_rotation = _camera_rotation(right_camera)
    if left_rotation is None or right_rotation is None:
        return None
    true_rotation = right_rotation @ left_rotation.T
    errors: dict[int, float] = {}
    for quadrant in range(4):
        candidate = right_rotation @ _ry_degrees(90.0 * quadrant) @ left_rotation.T
        errors[quadrant] = angle_between(candidate, true_rotation)
    return int(min(errors, key=lambda quadrant: (errors[quadrant], quadrant)))


def pose_error_record(
    result: JointPairResult,
    left_camera: CameraEstimate,
    right_camera: CameraEstimate,
) -> dict[str, Any]:
    """Assisted-camera ground-truth yaw/baseline error, where well-defined.

    ``defined`` means the DS-11 pose stage solved and its relative rotation can be
    compared with the assisted factory relative rotation. The baseline direction
    is compared after removing the four-fold world yaw gauge the pose leaves free,
    mirroring the approved ``diagnose_joint_pairs`` reference metric, and is
    reported separately under ``baseline_error_defined``.
    """
    pose = result.pose
    solved_class = None if pose is None else pose.right_yaw_class
    record: dict[str, Any] = {
        "assisted": True,
        "defined": False,
        "undefined_reason": None,
        "true_yaw_class": reference_yaw_class(left_camera, right_camera),
        "solved_yaw_class": solved_class,
        "yaw_class_correct": None,
        "relative_rotation_error_degrees": None,
        "baseline_error_defined": False,
        "baseline_undefined_reason": None,
        "gauge_index": None,
        "baseline_error_degrees_gauge_aligned": None,
        "baseline_error_degrees_without_gauge": None,
        "baseline_error_degrees_signless": None,
        "true_baseline_length_blocks": None,
    }
    if pose is None:
        record["undefined_reason"] = "pose_not_attempted"
        return record
    if not pose.solved or pose.relative_rotation is None or pose.baseline_direction is None:
        record["undefined_reason"] = "pose_unsolved"
        return record
    left_rotation = _camera_rotation(left_camera)
    right_rotation = _camera_rotation(right_camera)
    if left_rotation is None or right_rotation is None:
        record["undefined_reason"] = "assisted_camera_rotation_unavailable"
        return record
    true_rotation = right_rotation @ left_rotation.T
    got_rotation = np.asarray(pose.relative_rotation, dtype=np.float64)
    record["defined"] = True
    record["relative_rotation_error_degrees"] = angle_between(got_rotation, true_rotation)
    true_class = record["true_yaw_class"]
    record["yaw_class_correct"] = (
        None if solved_class is None or true_class is None
        else bool(int(solved_class) == int(true_class))
    )
    left_pose = left_camera.pose
    right_pose = right_camera.pose
    if (left_pose is None or right_pose is None
            or left_pose.position is None or right_pose.position is None):
        record["baseline_undefined_reason"] = "assisted_camera_position_unavailable"
        return record
    baseline_true = (
        np.asarray(right_pose.position, dtype=np.float64)
        - np.asarray(left_pose.position, dtype=np.float64)
    )
    length = float(np.linalg.norm(baseline_true))
    record["true_baseline_length_blocks"] = length
    if length <= 1e-9:
        record["baseline_undefined_reason"] = "degenerate_true_baseline"
        return record
    baseline_true = baseline_true / length
    baseline_got = np.asarray(pose.baseline_direction, dtype=np.float64)
    got_norm = float(np.linalg.norm(baseline_got))
    if got_norm <= 1e-12:
        record["baseline_undefined_reason"] = "degenerate_solved_baseline"
        return record
    baseline_got = baseline_got / got_norm
    errors: dict[int, float] = {}
    for step in GAUGE_ROTATIONS:
        candidate = _ry_degrees(step) @ baseline_true
        cosine = float(np.clip(np.dot(candidate, baseline_got), -1.0, 1.0))
        errors[step] = float(np.degrees(np.arccos(cosine)))
    best = min(errors, key=lambda step: (errors[step], step))
    record["baseline_error_defined"] = True
    record["gauge_index"] = int(best // 90)
    record["baseline_error_degrees_gauge_aligned"] = errors[best]
    record["baseline_error_degrees_without_gauge"] = errors[0]
    cosine = float(np.clip(np.dot(baseline_true, baseline_got), -1.0, 1.0))
    record["baseline_error_degrees_signless"] = float(
        np.degrees(np.arccos(abs(cosine))))
    return record


def yaw_class_record(score: YawClassScore) -> dict[str, Any]:
    """Every field of one DS-11 ``solve_pair`` ``YawClassScore`` as JSON data."""
    return {
        "yaw_class": int(score.yaw_class),
        "passed": bool(score.passed),
        "refusal": str(score.refusal),
        "inliers": int(score.inliers),
        "informative": int(score.informative),
        "inlier_fraction": float(score.inlier_fraction),
        "median_rho_degrees": score.median_rho_degrees,
        "parallax_degrees": score.parallax_degrees,
        "cheirality_fraction": score.cheirality_fraction,
        "singular_ratio": score.singular_ratio,
        "singular_values": (
            None if score.singular_values is None
            else [float(value) for value in score.singular_values]
        ),
    }


def _selected_support_edges(
    result: JointPairResult,
    left_points: np.ndarray,
    right_points: np.ndarray,
) -> list[list[int]]:
    """The one-to-one selected support as canonical ``(left, right)`` indices."""
    indices = match_indices(result.matches, left_points, right_points)
    return [[int(left), int(right)] for left, right in indices]


def _gates_record(result: JointPairResult) -> dict[str, Any]:
    return {
        "median_selected_residual_degrees": result.median_selected_residual_degrees,
        "descriptor_cost": result.descriptor_cost,
        "parallax_degrees": result.parallax_degrees,
        "cheirality_fraction": result.cheirality_fraction,
        "singular_ratio": result.singular_ratio,
        "ambiguity_margin": result.ambiguity_margin,
    }


def arm_record(
    arm: str,
    result: JointPairResult,
    graph: ReciprocalGraph,
    settings: JointMatchSettings,
    geometry: PairGeometry,
    left_points: np.ndarray,
    right_points: np.ndarray,
    left_camera: CameraEstimate,
    right_camera: CameraEstimate,
) -> dict[str, Any]:
    """One arm's graph, floor, support, spatial, pose-class and pose-error record."""
    selection_state, pose_state, matches_available, pose_available = describe_outcome(
        result)
    pose = result.pose
    return {
        "arm": arm,
        "graph_mode": (
            "production_reciprocal_topk_graph" if arm == ARM_PRODUCTION
            else ORACLE_GRAPH_MODE
        ),
        "min_support_absolute": int(settings.min_support_absolute),
        "min_support_fraction": float(settings.min_support_fraction),
        "graph_edges": int(graph.edges.shape[0]),
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
        "support_floor": int(result.support_floor),
        "support": int(result.support),
        "selected_support": int(result.support),
        "selected_support_edges": _selected_support_edges(
            result, left_points, right_points),
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
        "pose_class": None if pose is None else pose.right_yaw_class,
        "pose": _pose_record(result),
        "pose_classes_available": pose_available,
        "pose_classes": (
            [] if pose is None else [yaw_class_record(item) for item in pose.classes]
        ),
        "pose_error": pose_error_record(result, left_camera, right_camera),
        "hypotheses": [hypothesis_record(item) for item in result.hypotheses],
        "selected_stage": geometry_record(geometry),
    }


@dataclass(frozen=True, slots=True)
class SupportPairAttribution:
    """One unordered frame pair's three-arm support-floor sensitivity ledger."""

    left_id: str
    right_id: str
    production_graph: ReciprocalGraph
    oracle_graph: ReciprocalGraph
    production_result: JointPairResult
    default_result: JointPairResult
    control_result: JointPairResult
    production_selected: PairGeometry
    default_selected: PairGeometry
    control_selected: PairGeometry
    oracle_candidate: PairGeometry
    left_points: np.ndarray
    right_points: np.ndarray
    left_camera: CameraEstimate
    right_camera: CameraEstimate

    @property
    def truth_pairs(self) -> int:
        return self.oracle_candidate.truth_pairs

    @property
    def oracle_graph_empty(self) -> bool:
        return int(self.oracle_graph.edges.shape[0]) == 0

    @property
    def production_solved(self) -> bool:
        return self.production_result.status == "solved"

    @property
    def default_solved(self) -> bool:
        return self.default_result.status == "solved"

    @property
    def control_solved(self) -> bool:
        return self.control_result.status == "solved"


def _scored_selection(
    result: JointPairResult,
    observation_left,
    observation_right,
    *,
    tolerance: float,
) -> PairGeometry:
    return score_pair_matches(
        match_indices(
            result.matches,
            observation_left.features.points,
            observation_right.features.points,
        ),
        observation_left.hits,
        observation_right.hits,
        observation_left.features.points,
        observation_right.features.points,
        tolerance=tolerance,
    )


def attribute_pair(
    left,
    right,
    left_camera: CameraEstimate,
    right_camera: CameraEstimate,
    *,
    default_matcher: JointMatchSettings | None = None,
    control_matcher: JointMatchSettings | None = None,
    gates: PairGates | None = None,
    ratio_test: float = _RATIO_TEST,
    tolerance: float = SURFACE_TOLERANCE_BLOCKS,
) -> SupportPairAttribution:
    """Run the three arms on one pair: production context, oracle default, control.

    The production arm uses the production reciprocal graph and is complete before
    any injection. The two oracle arms share one truth-surface graph; only the
    ``min_support_absolute``/``min_support_fraction`` settings differ. The
    production builder is restored by the reusable R5K context manager even if a
    matcher call raises.
    """
    bounds = DEFAULT_SETTINGS if default_matcher is None else default_matcher
    control = CONTROL_SETTINGS if control_matcher is None else control_matcher
    production_graph = reciprocal_topk_graph(
        left.features.descriptors,
        right.features.descriptors,
        neighbors=int(bounds.neighbors),
        max_edges=int(bounds.max_candidate_edges),
        distance_ceiling=bounds.graph_distance_ceiling,
    )
    oracle_graph = build_oracle_graph(
        left.features, right.features, left.hits, right.hits, tolerance=tolerance)
    production_result = match_feature_frames(
        left.features,
        right.features,
        left_camera,
        right_camera,
        matcher=bounds,
        gates=gates,
        ratio_test=ratio_test,
    )
    with injected_candidate_graph(oracle_graph):
        default_result = match_feature_frames(
            left.features,
            right.features,
            left_camera,
            right_camera,
            matcher=bounds,
            gates=gates,
            ratio_test=ratio_test,
        )
        control_result = match_feature_frames(
            left.features,
            right.features,
            left_camera,
            right_camera,
            matcher=control,
            gates=gates,
            ratio_test=ratio_test,
        )
    return SupportPairAttribution(
        left_id=production_result.left_id,
        right_id=production_result.right_id,
        production_graph=production_graph,
        oracle_graph=oracle_graph,
        production_result=production_result,
        default_result=default_result,
        control_result=control_result,
        production_selected=_scored_selection(
            production_result, left, right, tolerance=tolerance),
        default_selected=_scored_selection(
            default_result, left, right, tolerance=tolerance),
        control_selected=_scored_selection(
            control_result, left, right, tolerance=tolerance),
        oracle_candidate=score_pair_matches(
            np.asarray(oracle_graph.edges, dtype=np.int64),
            left.hits, right.hits, left.features.points, right.features.points,
            tolerance=tolerance,
        ),
        left_points=np.asarray(left.features.points, dtype=np.float64),
        right_points=np.asarray(right.features.points, dtype=np.float64),
        left_camera=left_camera,
        right_camera=right_camera,
    )


def _graph_for(pair: SupportPairAttribution, arm: str) -> ReciprocalGraph:
    return pair.production_graph if arm == ARM_PRODUCTION else pair.oracle_graph


def _settings_for(arm: str) -> JointMatchSettings:
    return DEFAULT_SETTINGS if arm != ARM_CONTROL else CONTROL_SETTINGS


def _result_for(pair: SupportPairAttribution, arm: str) -> JointPairResult:
    if arm == ARM_PRODUCTION:
        return pair.production_result
    if arm == ARM_CONTROL:
        return pair.control_result
    return pair.default_result


def _selected_for(pair: SupportPairAttribution, arm: str) -> PairGeometry:
    if arm == ARM_PRODUCTION:
        return pair.production_selected
    if arm == ARM_CONTROL:
        return pair.control_selected
    return pair.default_selected


def comparison_record(pair: SupportPairAttribution) -> dict[str, Any]:
    """The paired default-versus-control outcome of one frame pair."""
    default = pair.default_result
    control = pair.control_result
    return {
        "production_solved": pair.production_solved,
        "default_solved": pair.default_solved,
        "control_solved": pair.control_solved,
        "default_refusal_reason": default.reason,
        "control_refusal_reason": control.reason,
        "default_hypothesis_count": len(default.hypotheses),
        "control_hypothesis_count": len(control.hypotheses),
        "default_support": int(default.support),
        "control_support": int(control.support),
        "support_delta": int(control.support) - int(default.support),
        "default_support_floor": int(default.support_floor),
        "control_support_floor": int(control.support_floor),
        "support_floor_delta": int(control.support_floor) - int(default.support_floor),
        "default_spatial_bins": {
            "left": int(default.spatial_bins_left),
            "right": int(default.spatial_bins_right),
        },
        "control_spatial_bins": {
            "left": int(control.spatial_bins_left),
            "right": int(control.spatial_bins_right),
        },
        "default_reason_was_insufficient_one_to_one_support": (
            default.reason == UNREACHABLE_REASON),
        "control_gained_candidate_mode": (
            len(default.hypotheses) == 0 and len(control.hypotheses) > 0),
        "control_gained_selection": (
            int(default.support) == 0 and int(control.support) > 0),
        "control_gained_solve": (not pair.default_solved and pair.control_solved),
        "control_lost_solve": (pair.default_solved and not pair.control_solved),
    }


def pair_record(pair: SupportPairAttribution) -> dict[str, Any]:
    """One pair's three-arm ledger, shared oracle graph and paired comparison."""
    oracle_edges = [
        [int(left), int(right)] for left, right in np.asarray(pair.oracle_graph.edges)
    ]
    record: dict[str, Any] = {
        "left_frame_id": pair.left_id,
        "right_frame_id": pair.right_id,
        "truth_pairs": pair.truth_pairs,
        "oracle_graph_empty": pair.oracle_graph_empty,
        "oracle_graph_edges": oracle_edges,
        "oracle_graph_edge_count": len(oracle_edges),
        "arms": list(ARM_ORDER),
    }
    # The two oracle arms share the graph; the production arm carries its own.
    for arm in ARM_ORDER:
        record[arm] = arm_record(
            arm,
            _result_for(pair, arm),
            _graph_for(pair, arm),
            _settings_for(arm),
            _selected_for(pair, arm),
            pair.left_points,
            pair.right_points,
            pair.left_camera,
            pair.right_camera,
        )
    record["comparison"] = comparison_record(pair)
    return record


def _median_or_none(values: Sequence[float | None]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return None if not numbers else float(np.median(numbers))


def _counter(values: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def summarise_attributions(
    pairs: Sequence[SupportPairAttribution],
) -> dict[str, Any]:
    """Pooled per-arm totals and the paired default-versus-control outcome."""
    arms: dict[str, Any] = {}
    for arm in ARM_ORDER:
        results = [_result_for(pair, arm) for pair in pairs]
        selected = [_selected_for(pair, arm) for pair in pairs]
        errors = [
            pose_error_record(_result_for(pair, arm), pair.left_camera, pair.right_camera)
            for pair in pairs
        ]
        arms[arm] = {
            "graph_mode": (
                "production_reciprocal_topk_graph" if arm == ARM_PRODUCTION
                else ORACLE_GRAPH_MODE
            ),
            "min_support_absolute": int(_settings_for(arm).min_support_absolute),
            "min_support_fraction": float(_settings_for(arm).min_support_fraction),
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
                int(_graph_for(pair, arm).edges.shape[0]) for pair in pairs),
            "raw_keypoints_total": sum(
                int(result.raw_keypoints[0] + result.raw_keypoints[1])
                for result in results),
            "support_total": sum(int(result.support) for result in results),
            "support_floor_min": (
                min(int(result.support_floor) for result in results) if results else None
            ),
            "support_floor_max": (
                max(int(result.support_floor) for result in results) if results else None
            ),
            "pairs_with_a_candidate_mode": sum(
                1 for result in results if len(result.hypotheses) > 0),
            "pairs_with_selected_support": sum(
                1 for result in results if int(result.support) > 0),
            "pose_class_histogram": dict(sorted(_counter([
                "none" if result.pose is None or result.pose.right_yaw_class is None
                else str(result.pose.right_yaw_class)
                for result in results
            ]).items())),
            "pose_error_defined": sum(1 for error in errors if error["defined"]),
            "yaw_class_correct": sum(
                1 for error in errors if error["yaw_class_correct"] is True),
            "median_relative_rotation_error_degrees": _median_or_none(
                [error["relative_rotation_error_degrees"] for error in errors]),
            "median_baseline_error_degrees_gauge_aligned": _median_or_none(
                [error["baseline_error_degrees_gauge_aligned"] for error in errors]),
            "selected_stage": _totals_record(_geometry_totals(selected)),
        }

    unreachable = [
        pair for pair in pairs if pair.default_result.reason == UNREACHABLE_REASON
    ]
    summary: dict[str, Any] = {
        "pair_count": len(pairs),
        "truth_pairs": sum(pair.truth_pairs for pair in pairs),
        "pairs_with_truth": sum(1 for pair in pairs if pair.truth_pairs > 0),
        "pairs_with_empty_truth": sum(1 for pair in pairs if pair.truth_pairs == 0),
        "oracle_graph_empty_pairs": sum(
            1 for pair in pairs if pair.oracle_graph_empty),
        "arms": arms,
        "default_vs_control": {
            "both_solved": sum(
                1 for pair in pairs if pair.default_solved and pair.control_solved),
            "only_default_solved": sum(
                1 for pair in pairs if pair.default_solved and not pair.control_solved),
            "only_control_solved": sum(
                1 for pair in pairs if not pair.default_solved and pair.control_solved),
            "neither_solved": sum(
                1 for pair in pairs
                if not pair.default_solved and not pair.control_solved),
            "default_insufficient_one_to_one_support_pairs": len(unreachable),
            "control_gained_candidate_mode_pairs": sum(
                1 for pair in pairs
                if len(pair.default_result.hypotheses) == 0
                and len(pair.control_result.hypotheses) > 0),
            "control_gained_selection_pairs": sum(
                1 for pair in pairs
                if int(pair.default_result.support) == 0
                and int(pair.control_result.support) > 0),
            "control_gained_solve_pairs": sum(
                1 for pair in pairs
                if not pair.default_solved and pair.control_solved),
            "control_lost_solve_pairs": sum(
                1 for pair in pairs
                if pair.default_solved and not pair.control_solved),
            "unreachable_gained_candidate_mode": sum(
                1 for pair in unreachable
                if len(pair.control_result.hypotheses) > 0),
            "unreachable_gained_selection": sum(
                1 for pair in unreachable if int(pair.control_result.support) > 0),
            "unreachable_gained_solve": sum(
                1 for pair in unreachable if pair.control_solved),
            "control_gained_solve_pair_ids": [
                f"{pair.left_id}|{pair.right_id}"
                for pair in pairs if not pair.default_solved and pair.control_solved
            ],
            "unreachable_gained_solve_pair_ids": [
                f"{pair.left_id}|{pair.right_id}"
                for pair in unreachable if pair.control_solved
            ],
        },
    }
    mismatches = {
        arm: (
            arms[arm]["selected_stage"]["positive"]
            - arms[arm]["selected_stage"]["truth_pairs_returned"]
        )
        for arm in ARM_ORDER
    }
    summary["positive_identity_mismatch"] = mismatches
    summary["same_surface_not_same_feature_point"] = SAME_SURFACE_NOT_SAME_FEATURE_POINT
    return summary


def verify_r5k_pins(entries: Sequence[DatasetSceneInput]) -> dict[str, Any]:
    """Verify every discovered index against the immutable DS-R5K index pins.

    The discovered dataset-id set must equal the five-entry DS-R5K corpus and
    every index sha256 must match. This is stricter than the index's own declared
    scene/screenshot hashes, which :func:`discover_finalized_entries` already
    re-verifies before this runs.
    """
    found = {entry.dataset_id: entry.index_sha256 for entry in entries}
    if len(found) != len(entries):
        raise SupportSensitivityError("duplicate dataset ids among discovered entries")
    expected = dict(R5K_INDEX_PINS)
    if set(found) != set(expected):
        missing = sorted(set(expected) - set(found))
        extra = sorted(set(found) - set(expected))
        raise SupportSensitivityError(
            f"discovered dataset ids do not match the DS-R5K pins; "
            f"missing={missing} extra={extra}")
    mismatched = {
        dataset_id: {"found": found[dataset_id], "pinned": expected[dataset_id]}
        for dataset_id in sorted(expected)
        if found[dataset_id] != expected[dataset_id]
    }
    if mismatched:
        raise SupportSensitivityError(
            f"DS-R5K index pin mismatch: {mismatched}")
    return {
        "all_match": True,
        "entry_count": len(found),
        "index_pins": {dataset_id: expected[dataset_id] for dataset_id in sorted(expected)},
    }


def verify_canonical_scene_pins(canonical: DatasetSceneInput) -> dict[str, Any]:
    """Verify the canonical export against the immutable DS-R5K scene/screenshot pins."""
    if canonical.dataset_id != R5K_CANONICAL_DATASET_ID:
        raise SupportSensitivityError(
            f"canonical dataset {canonical.dataset_id!r} is not the DS-R5K "
            f"canonical {R5K_CANONICAL_DATASET_ID!r}")
    files = dict(canonical.hashes)
    expected_files = dict(R5K_CANONICAL_SCENE_FILE_PINS)
    if files != expected_files:
        raise SupportSensitivityError(
            "DS-R5K canonical scene-file pin mismatch: "
            f"found={sorted(files)} pinned={sorted(expected_files)}")
    digests = sorted(digest for _filename, digest in canonical.screenshots)
    if len(digests) != len(R5K_CANONICAL_SCREENSHOT_PINS) or sorted(
            R5K_CANONICAL_SCREENSHOT_PINS) != digests:
        raise SupportSensitivityError(
            "DS-R5K canonical screenshot pin mismatch: "
            f"found {len(digests)} screenshots, pinned "
            f"{len(R5K_CANONICAL_SCREENSHOT_PINS)}")
    return {
        "all_match": True,
        "dataset_id": canonical.dataset_id,
        "scene_file_pins": {name: expected_files[name] for name in sorted(expected_files)},
        "screenshot_count": len(digests),
        "screenshot_sha256": digests,
    }


def build_payload(
    exports_root: str,
    entries: Sequence[DatasetSceneInput],
    manifest: Mapping[str, Any],
    canonical_entry: DatasetSceneInput,
    *,
    dataset_id: str | None = None,
    default_matcher: JointMatchSettings | None = None,
    control_matcher: JointMatchSettings | None = None,
    tolerance: float = SURFACE_TOLERANCE_BLOCKS,
    pin_record: Mapping[str, Any] | None = None,
    scene_pin_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the three-arm diagnostic over the canonical export and seal the payload.

    The caller is responsible for having run the strict pin and leakage preflight;
    ``pin_record``/``scene_pin_record`` are recorded verbatim for audit.
    """
    bounds = DEFAULT_SETTINGS if default_matcher is None else default_matcher
    control = CONTROL_SETTINGS if control_matcher is None else control_matcher
    detection = detection_settings(bounds)
    scene = build_scene(canonical_entry)
    observations = [observe_oracle_frame(scene, frame, detection=detection)
                    for frame in scene.frames]
    cameras = [oracle_camera_estimate(frame) for frame in scene.frames]
    pairs: list[SupportPairAttribution] = []
    for left_index in range(len(observations)):
        for right_index in range(left_index + 1, len(observations)):
            pairs.append(attribute_pair(
                observations[left_index],
                observations[right_index],
                cameras[left_index],
                cameras[right_index],
                default_matcher=bounds,
                control_matcher=control,
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
    pair_records = [pair_record(pair) for pair in pairs]
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete",
        "created_for": "DS-R5N",
        "tool": "scripts/evaluation/evaluate_support_sensitivity.py",
        "evaluation_only": True,
        "promotion_or_production_change": False,
        "r5k_sealed_artifact_untouched": R5K_SEALED_ARTIFACT,
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
                "replaced for exactly the two oracle matcher calls and restored in a "
                "finally block; the production baseline arm uses the production "
                "builder before any injection"),
            "same_surface_not_same_feature_point": (
                SAME_SURFACE_NOT_SAME_FEATURE_POINT),
        },
        "arm_policy": {
            "arms": list(ARM_ORDER),
            "default_arm": DEFAULT_ARM,
            "control_arm": ARM_CONTROL,
            "control_is_oracle_only": True,
            "control_settings": {
                "min_support_absolute": int(control.min_support_absolute),
                "min_support_fraction": float(control.min_support_fraction),
            },
            "default_settings": {
                "min_support_absolute": int(bounds.min_support_absolute),
                "min_support_fraction": float(bounds.min_support_fraction),
            },
            "control_measurement_only": (
                "the control measures how many unreachable pairs gain a candidate "
                "mode, selected support and solve; it is not a production "
                "recommendation and no threshold may be promoted from it"),
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
            "control_joint_match_settings": {
                "min_support_absolute": control.min_support_absolute,
                "min_support_fraction": control.min_support_fraction,
                "other_fields": "identical to joint_match_settings",
            },
            "detection_settings": {
                "max_features": detection.max_features,
                "ratio_test": detection.ratio_test,
                "mask_boundary_pad_px": detection.mask_boundary_pad_px,
            },
            "surface_tolerance_blocks": tolerance,
            "identity": "unique pixel pair; repeated keypoints at one pixel collapse",
            "support_floor_rule": (
                "max(min_support_absolute, ceil(min_support_fraction * "
                "min(raw_keypoints)))"),
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
            "pairs": pair_records,
            "summary": summary,
        },
        "summary": summary,
        "renderer_geometry": {
            **geometry_policy_record(),
            "camera_policy_mode": CAMERA_POLICY_MODE,
            "camera_policy_assisted": True,
            "oracle_graph_mode": ORACLE_GRAPH_MODE,
            "control_arm": ARM_CONTROL,
            "control_min_support_absolute": int(control.min_support_absolute),
            "control_min_support_fraction": float(control.min_support_fraction),
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
                for value in summary["positive_identity_mismatch"].values()),
        },
        "limitations": list(LIMITATIONS),
    }
    payload["seal_sha256"] = seal_record(payload)
    return payload


def _sealed_output_key(path: str | Path) -> str:
    """Absolute, lexically-collapsed, case-folded key for a candidate path.

    Built only from :func:`os.path.abspath`/:func:`os.path.normpath`, so the
    guard never opens, stats or resolves the sealed artifact through the
    filesystem. Case folding is unconditional because the guard must hold on
    Windows, where path names are case-insensitive.
    """
    absolute = os.path.normpath(os.path.abspath(os.fspath(path)))
    return absolute.replace("\\", "/").casefold()


def output_targets_sealed_r5k_artifact(path: str | Path) -> bool:
    """Whether ``path`` names the sealed DS-R5K artifact, case-insensitively.

    Windows filenames are case-insensitive, so a mixed-case spelling such as
    ``.Tmp-Oracle-Candidate-Graph-20261003.json`` resolves to the same file as
    :data:`R5K_SEALED_ARTIFACT`. A raw ``str``/``name`` comparison therefore
    admitted a case-folded path; both operands are normalised here instead.
    """
    return _sealed_output_key(path).endswith(
        R5K_SEALED_ARTIFACT.replace("\\", "/").casefold())


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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    output = Path(args.output)
    if output_targets_sealed_r5k_artifact(output):
        print(f"refusing to touch the sealed DS-R5K artifact {output}", file=sys.stderr)
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
    paired = summary["default_vs_control"]
    print(f"canonical={payload['scene']['dataset_id']} "
          f"frames={payload['scene']['frame_count']} pairs={summary['pair_count']}")
    print(f"truth_pairs={summary['truth_pairs']} "
          f"default_solved={summary['arms'][ARM_DEFAULT]['solved']} "
          f"control_solved={summary['arms'][ARM_CONTROL]['solved']}")
    print(f"unreachable={paired['default_insufficient_one_to_one_support_pairs']} "
          f"gained_mode={paired['unreachable_gained_candidate_mode']} "
          f"gained_selection={paired['unreachable_gained_selection']} "
          f"gained_solve={paired['unreachable_gained_solve']}")
    print(f"wrote {output} seal={payload['seal_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
