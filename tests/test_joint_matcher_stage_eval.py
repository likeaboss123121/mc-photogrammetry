"""Focused tests for the evaluation-only joint-matcher stage attribution.

DS-R5J. These tests exercise the pure stage ledger on synthetic same/wrong/
occluded surfaces, the refusal-versus-pose classification (including a refusal
that carries selected matches and a solved DS-11 pose), the unique pixel-pair
dedupe, the grouped split manifest and the runner guards. They do not run SIFT on
the corpus; the single corpus evaluation is a separate authorized run.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.evaluation_pair_correspondence import (
    DatasetSceneInput,
    FinalizedScene,
    FrameInput,
    PairCorrespondenceEvaluationError,
    SurfaceHit,
    build_group_manifest,
    check_fold_leakage,
    check_split_leakage,
    seal_record,
)
from mc_photogrammetry.pipeline.connected_camera import PairPose
from mc_photogrammetry.pipeline.correspondence import FrameFeatures
from mc_photogrammetry.pipeline.joint_matching import (
    JointMatchSettings,
    JointPairResult,
)
from mc_photogrammetry.reconstruction.camera import (
    CameraModel,
    intrinsics_for,
    project_points,
)

WIDTH, HEIGHT = 1280, 720
LEFT_POSITION = (0.0, 3.0, 0.0)
RIGHT_POSITION = (2.5, 3.0, 1.0)
LEFT_YAW, LEFT_PITCH = 85.0, 3.0
RIGHT_YAW, RIGHT_PITCH = 95.0, -2.0

_POINTS = (
    (-9.0, 3.0, 0.5), (-8.0, 2.0, 1.5), (-10.0, 4.0, -0.5), (-9.5, 1.5, 2.0),
    (-8.5, 5.0, 0.0), (-11.0, 3.5, 1.0), (-7.5, 2.5, -1.0), (-10.5, 2.0, 2.5),
    (-9.0, 5.5, -1.5), (-12.0, 4.5, 0.5), (-8.0, 3.8, 2.8), (-10.0, 1.0, -0.8),
    (-13.0, 3.0, 1.5), (-7.0, 3.2, 0.8), (-11.5, 2.8, -1.8), (-9.8, 4.8, 2.2),
    (-8.8, 1.2, 0.2), (-12.5, 2.2, 2.0), (-10.8, 5.2, -1.2), (-7.8, 4.2, -1.8),
)


@pytest.fixture(scope="module")
def stage_module():
    """Load the evaluation script as a module (it defines module-level dataclasses)."""
    script = (Path(__file__).resolve().parents[1]
              / "scripts" / "evaluation" / "evaluate_joint_matcher_stage.py")
    spec = importlib.util.spec_from_file_location("ds_r5j_stage", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --- synthetic geometry and truth ------------------------------------------------


def _camera(position, yaw, pitch):
    return CameraModel(position, yaw, pitch, eye_height=0.0)


def _frame_input(frame_id, camera, *, width=WIDTH, height=HEIGHT):
    return FrameInput(
        frame_id=frame_id, filename=f"{frame_id}.png", camera=camera,
        width=width, height=height, screenshot_path="", screenshot_sha256="",
    )


def _features(frame_id, points, descriptors):
    return FrameFeatures(
        frame_id=frame_id, points=points, descriptors=descriptors,
        width=WIDTH, height=HEIGHT, usable_pixels=WIDTH * HEIGHT,
    )


def _observed_hits(points, slot, world_points, state_name="minecraft:stone"):
    return tuple(
        SurfaceHit(
            frame_id="frame", coordinate=(index, 0, 0), slot=slot,
            world_point=tuple(float(value) for value in world_points[index]),
            camera_depth=1.0, state=BlockState(state_name),
            pixel=(float(points[index][0]), float(points[index][1])),
        )
        for index in range(len(points))
    )


def _observation(stage_module, frame_id, points, descriptors, hits):
    return stage_module.JointFrameObservation(
        frame_id=frame_id, width=WIDTH, height=HEIGHT,
        usable_pixels=WIDTH * HEIGHT, boundary_dropped=0,
        features=_features(frame_id, points, descriptors), hits=hits,
        no_truth_reasons=(),
    )


def _scene(stage_module, *, right_mode="same"):
    """Project 20 independent 3D points and attach a synthetic truth surface.

    ``right_mode`` selects same-surface truth, a structurally different surface
    (``wrong``) or no surface at all (``occluded``) in the right frame.
    """
    left_camera = _camera(LEFT_POSITION, LEFT_YAW, LEFT_PITCH)
    right_camera = _camera(RIGHT_POSITION, RIGHT_YAW, RIGHT_PITCH)
    left_projected = project_points(_POINTS, left_camera, width=WIDTH, height=HEIGHT)
    right_projected = project_points(_POINTS, right_camera, width=WIDTH, height=HEIGHT)
    left_points = np.asarray([[p[0], p[1]] for p in left_projected], dtype=np.float64)
    right_points = np.asarray([[p[0], p[1]] for p in right_projected], dtype=np.float64)
    descriptors = np.random.default_rng(3).normal(size=(len(left_points), 16))
    left = _observation(
        stage_module, "a", left_points, descriptors,
        _observed_hits(left_points, "north", _POINTS))
    if right_mode == "same":
        right_hits = _observed_hits(right_points, "north", _POINTS)
    elif right_mode == "wrong":
        shifted = tuple((point[0] + 50.0, point[1], point[2]) for point in _POINTS)
        right_hits = _observed_hits(right_points, "south", shifted, "minecraft:dirt")
    elif right_mode == "occluded":
        right_hits = (None,) * len(right_points)
    else:
        raise ValueError(right_mode)
    right = _observation(stage_module, "b", right_points, descriptors, right_hits)
    left_camera_estimate = stage_module.oracle_camera_estimate(
        _frame_input("a", left_camera))
    right_camera_estimate = stage_module.oracle_camera_estimate(
        _frame_input("b", right_camera))
    return left, right, left_camera_estimate, right_camera_estimate


def _matches(left, right, count):
    return np.column_stack(
        (left.features.points[:count], right.features.points[:count]))


def _pair_pose(*, solved, right_yaw_class=1):
    return PairPose(
        left_id="a", right_id="b", correspondences=8, solved=solved,
        reason="" if solved else "no_yaw_class_passes_gates",
        right_yaw_class=right_yaw_class if solved else None,
        relative_rotation=np.eye(3) if solved else None,
        baseline_direction=(1.0, 0.0, 0.0) if solved else None,
        inliers=8 if solved else 0, inlier_fraction=1.0 if solved else 0.0,
        median_rho_degrees=0.1 if solved else None,
        parallax_degrees=5.0 if solved else None,
        cheirality_fraction=1.0 if solved else None, classes=(),
    )


def _result(*, status, reason, matches, pose=None, right_yaw_class=None, support=8):
    return JointPairResult(
        left_id="a", right_id="b", status=status, reason=reason, pose=pose,
        right_yaw_class=right_yaw_class, matches=matches,
        raw_keypoints=(len(_POINTS), len(_POINTS)), boundary_dropped=(0, 0),
        candidate_edges=76, total_graph_edges=76, truncated=0,
        ratio_survivors=(20, 20), mutual_survivors=20, support=support,
        support_floor=8, spatial_bins_left=4, spatial_bins_right=4,
        spatial_bins_total=16, median_selected_residual_degrees=0.1,
        descriptor_cost=0.1, parallax_degrees=5.0, cheirality_fraction=1.0,
        singular_ratio=0.5, ambiguity_margin=3, hypotheses=(),
    )


# --- oracle camera and production settings ---------------------------------------


def test_oracle_camera_estimate_copies_exact_factory_metadata(stage_module) -> None:
    camera = _camera((1.5, 2.5, -3.5), LEFT_YAW, LEFT_PITCH)
    estimate = stage_module.oracle_camera_estimate(_frame_input("a", camera))

    expected_focal = (HEIGHT / 2.0) / math.tan(math.radians(70.0) / 2.0)
    assert estimate.intrinsics == intrinsics_for(camera, WIDTH, HEIGHT)
    assert estimate.intrinsics.focal_length_pixels == pytest.approx(
        (expected_focal, expected_focal))
    assert estimate.intrinsics.principal_point_pixels == (WIDTH / 2.0, HEIGHT / 2.0)
    assert estimate.pose is not None
    assert estimate.pose.position == (1.5, 2.5, -3.5)
    assert estimate.pose.yaw_degrees == LEFT_YAW
    assert estimate.pose.pitch_degrees == LEFT_PITCH
    assert estimate.pose.roll_degrees == 0.0


def test_camera_record_labels_the_assisted_oracle_camera(stage_module) -> None:
    camera = _camera(LEFT_POSITION, LEFT_YAW, LEFT_PITCH)
    frame = _frame_input("a", camera)
    record = stage_module.camera_record(frame, stage_module.oracle_camera_estimate(frame))

    assert record["camera_source"] == stage_module.CAMERA_POLICY_MODE
    assert record["assisted"] is True
    assert record["image_size"] == [WIDTH, HEIGHT]
    assert record["pose"]["yaw_degrees"] == LEFT_YAW


def test_detection_settings_mirror_the_production_joint_path(stage_module) -> None:
    matcher = JointMatchSettings()
    detection = stage_module.detection_settings(matcher)

    assert detection.max_features == matcher.max_features == 4000
    assert detection.mask_boundary_pad_px == matcher.mask_boundary_pad_px == 8.0
    assert detection.ratio_test == 0.75


# --- synthetic same/wrong/occluded surfaces --------------------------------------


def test_same_surface_selection_is_an_exact_positive(stage_module) -> None:
    left, right, left_camera, right_camera = _scene(stage_module, right_mode="same")
    pair = stage_module.attribute_pair(left, right, left_camera, right_camera)

    assert pair.status == "solved", pair.reason
    assert pair.selection_state == stage_module.SELECTION_SOLVED
    assert pair.pose_state == stage_module.POSE_SOLVED
    assert pair.matches_available is True and pair.pose_available is True
    assert pair.selected.truth_pairs == len(_POINTS)
    assert pair.selected.returned == len(_POINTS)
    assert pair.selected.positive == len(_POINTS)
    assert pair.selected.truth_pairs_returned == len(_POINTS)
    assert pair.selected.precision == 1.0
    assert pair.selected.recall == 1.0
    assert pair.support >= pair.support_floor >= 8
    # Every stage denominator is reported even when the selection is exact.
    assert pair.candidate_edges == pair.total_graph_edges >= len(_POINTS)
    assert pair.candidate.truth_pairs == len(_POINTS)
    assert pair.candidate.truth_pairs_returned == len(_POINTS)
    assert pair.mutual.truth_pairs_returned == len(_POINTS)
    assert pair.mutual.returned == pair.mutual_survivors


def test_wrong_surface_selection_has_no_truth_denominator(stage_module) -> None:
    left, right, left_camera, right_camera = _scene(stage_module, right_mode="wrong")
    pair = stage_module.attribute_pair(left, right, left_camera, right_camera)

    assert pair.selected.truth_pairs == 0
    assert pair.selected.positive == 0
    assert pair.selected.wrong_surface == pair.selected.returned
    assert pair.selected.precision == 0.0
    assert pair.selected.recall is None


def test_occluded_surface_is_unattributed_not_a_positive(stage_module) -> None:
    left, right, left_camera, right_camera = _scene(stage_module, right_mode="occluded")
    pair = stage_module.attribute_pair(left, right, left_camera, right_camera)

    assert pair.selected.truth_pairs == 0
    assert pair.selected.positive == 0
    assert pair.selected.unattributed == pair.selected.returned
    assert pair.selected.precision == 0.0
    assert pair.selected.recall is None


def test_pair_record_splits_correspondence_from_geometry_denominators(stage_module) -> None:
    left, right, left_camera, right_camera = _scene(stage_module, right_mode="same")
    record = stage_module.pair_record(
        stage_module.attribute_pair(left, right, left_camera, right_camera))

    denominators = record["denominators"]
    assert denominators["correspondence_truth_pairs"] == len(_POINTS)
    assert denominators["geometry_mutual_returned"] == record["mutual_stage"]["returned"]
    assert denominators["geometry_candidate_edges"] == record["candidate_stage"]["returned"]
    assert denominators["geometry_selected_returned"] == record["selected_stage"]["returned"]
    assert record["selected_stage"]["precision"] == 1.0
    assert record["selected_stage"]["recall"] == 1.0


# --- refusal versus pose abstention ----------------------------------------------


def test_describe_outcome_distinguishes_absent_selection_from_pose_abstention(
    stage_module,
) -> None:
    empty = np.empty((0, 4), dtype=np.float64)
    selection = np.asarray([[1.0, 2.0, 3.0, 4.0]])

    absent = stage_module.describe_outcome(_result(
        status="refused", reason="candidate_graph_empty", matches=empty))
    assert absent == (stage_module.SELECTION_ABSENT,
                      stage_module.POSE_NOT_ATTEMPTED, False, False)

    no_pose = stage_module.describe_outcome(_result(
        status="refused", reason="insufficient_one_to_one_support", matches=selection))
    assert no_pose == (stage_module.SELECTION_ON_REFUSAL,
                       stage_module.POSE_NOT_ATTEMPTED, True, False)

    abstained = stage_module.describe_outcome(_result(
        status="refused", reason="solve_pair_refused", matches=selection,
        pose=_pair_pose(solved=False)))
    assert abstained == (stage_module.SELECTION_ON_REFUSAL,
                         stage_module.POSE_ABSTAINED, True, True)

    # ``selection_unstable`` carries a *solved* DS-11 pose yet is still a refusal.
    mismatch = stage_module.describe_outcome(_result(
        status="refused", reason="selection_unstable", matches=selection,
        pose=_pair_pose(solved=True, right_yaw_class=2)))
    assert mismatch == (stage_module.SELECTION_ON_REFUSAL,
                        stage_module.POSE_CLASS_MISMATCH, True, True)

    solved = stage_module.describe_outcome(_result(
        status="solved", reason="", matches=selection,
        pose=_pair_pose(solved=True), right_yaw_class=1))
    assert solved == (stage_module.SELECTION_SOLVED,
                      stage_module.POSE_SOLVED, True, True)


def test_refusal_with_selected_matches_flows_through_attribute_pair(
    stage_module, monkeypatch,
) -> None:
    left, right, left_camera, right_camera = _scene(stage_module, right_mode="same")
    selection = _matches(left, right, 8)

    monkeypatch.setattr(stage_module, "match_feature_frames", lambda *a, **k: _result(
        status="refused", reason="solve_pair_refused", matches=selection,
        pose=_pair_pose(solved=False)))
    abstained = stage_module.attribute_pair(left, right, left_camera, right_camera)
    assert abstained.selection_state == stage_module.SELECTION_ON_REFUSAL
    assert abstained.matches_available is True
    assert abstained.pose_available is True
    assert abstained.pose_state == stage_module.POSE_ABSTAINED
    assert abstained.selected.returned == 8
    assert abstained.selected.positive == 8

    monkeypatch.setattr(stage_module, "match_feature_frames", lambda *a, **k: _result(
        status="refused", reason="candidate_graph_empty",
        matches=np.empty((0, 4), dtype=np.float64)))
    absent = stage_module.attribute_pair(left, right, left_camera, right_camera)
    assert absent.selection_state == stage_module.SELECTION_ABSENT
    assert absent.matches_available is False
    assert absent.pose_available is False
    assert absent.selected.returned == 0


# --- unique pixel-pair identity ---------------------------------------------------


def test_duplicate_pixel_pairs_are_counted_once(stage_module, monkeypatch) -> None:
    left, right, left_camera, right_camera = _scene(stage_module, right_mode="same")
    duplicated = np.vstack((_matches(left, right, 1), _matches(left, right, 1)))

    monkeypatch.setattr(stage_module, "match_feature_frames", lambda *a, **k: _result(
        status="solved", reason="", matches=duplicated,
        pose=_pair_pose(solved=True), right_yaw_class=1))
    pair = stage_module.attribute_pair(left, right, left_camera, right_camera)

    assert pair.selected.returned == 1
    assert pair.selected.positive == 1
    assert pair.selected.truth_pairs == len(_POINTS)
    assert pair.selected.truth_pairs_returned == 1
    assert pair.selected.precision == 1.0


def test_distinct_pixel_pairs_are_never_collapsed(stage_module, monkeypatch) -> None:
    left, right, left_camera, right_camera = _scene(stage_module, right_mode="same")
    first = _matches(left, right, 1)
    # A second row shares the right pixel but starts from a different left pixel.
    second = np.column_stack((left.features.points[1:2], right.features.points[0:1]))
    distinct = np.vstack((first, second))

    monkeypatch.setattr(stage_module, "match_feature_frames", lambda *a, **k: _result(
        status="solved", reason="", matches=distinct,
        pose=_pair_pose(solved=True), right_yaw_class=1))
    pair = stage_module.attribute_pair(left, right, left_camera, right_camera)

    assert pair.selected.returned == 2
    assert pair.selected.positive == 1
    # The second identity is a different surface (the same-state/same-slot proxy).
    assert pair.selected.same_state_same_slot_other_block == 1


def _duplicate_pixel_observations(stage_module, *, right_attributed: bool):
    """Two left indices at one pixel and two right indices at one pixel."""
    left_points = np.asarray(
        [[10.0, 20.0], [10.0, 20.0], [40.0, 50.0]], dtype=np.float64)
    right_points = np.asarray([[30.0, 40.0], [70.0, 80.0]], dtype=np.float64)
    left_descriptors = np.random.default_rng(21).normal(size=(3, 16))
    right_descriptors = np.random.default_rng(22).normal(size=(2, 16))

    def hit_at(frame_id, coordinate, world_point, pixel):
        return SurfaceHit(
            frame_id=frame_id, coordinate=coordinate, slot="south",
            world_point=world_point, camera_depth=1.0,
            state=BlockState("minecraft:stone"), pixel=pixel,
        )

    left_hits = (
        hit_at("a", (0, 0, 0), (0.5, 0.5, 1.0), (10.0, 20.0)),
        hit_at("a", (0, 0, 0), (0.5, 0.5, 1.0), (10.0, 20.0)),
        hit_at("a", (1, 0, 0), (1.5, 0.5, 1.0), (40.0, 50.0)),
    )
    right_hits = (
        (
            hit_at("b", (0, 0, 0), (0.5, 0.5, 1.0), (30.0, 40.0)),
            hit_at("b", (0, 0, 0), (0.5, 0.5, 1.0), (70.0, 80.0)),
        )
        if right_attributed
        else (None, None)
    )
    left = stage_module.JointFrameObservation(
        frame_id="a", width=WIDTH, height=HEIGHT, usable_pixels=WIDTH * HEIGHT,
        boundary_dropped=0, features=_features("a", left_points, left_descriptors),
        hits=left_hits, no_truth_reasons=(),
    )
    right = stage_module.JointFrameObservation(
        frame_id="b", width=WIDTH, height=HEIGHT, usable_pixels=WIDTH * HEIGHT,
        boundary_dropped=0, features=_features("b", right_points, right_descriptors),
        hits=right_hits, no_truth_reasons=(),
    )
    return left, right


def _patch_stage_inputs(stage_module, monkeypatch, edges):
    """Deterministically inject candidate edges and an empty selection."""
    monkeypatch.setattr(
        stage_module, "reciprocal_topk_graph",
        lambda *a, **k: types.SimpleNamespace(edges=edges))
    monkeypatch.setattr(
        stage_module, "select_mutual_matches",
        lambda *a, **k: types.SimpleNamespace(
            matches=np.empty((0, 4), dtype=np.float64)))
    monkeypatch.setattr(stage_module, "match_feature_frames", lambda *a, **k: _result(
        status="solved", reason="", matches=np.empty((0, 4), dtype=np.float64),
        pose=_pair_pose(solved=True), right_yaw_class=1))


def _duplicate_pixel_cameras(stage_module):
    return (
        stage_module.oracle_camera_estimate(
            _frame_input("a", _camera(LEFT_POSITION, LEFT_YAW, LEFT_PITCH))),
        stage_module.oracle_camera_estimate(
            _frame_input("b", _camera(RIGHT_POSITION, RIGHT_YAW, RIGHT_PITCH))),
    )


def test_attribute_pair_threads_keypoint_points_into_every_stage_scorer(
    stage_module, monkeypatch,
) -> None:
    left, right, left_camera, right_camera = _scene(stage_module, right_mode="same")
    captured: list[tuple[np.ndarray, np.ndarray]] = []
    real = stage_module.score_pair_matches

    def spy(indices, left_hits, right_hits, left_points, right_points, **kwargs):
        captured.append((left_points, right_points))
        return real(indices, left_hits, right_hits, left_points, right_points, **kwargs)

    monkeypatch.setattr(stage_module, "score_pair_matches", spy)
    stage_module.attribute_pair(left, right, left_camera, right_camera)

    # Mutual, candidate and selected all receive the same detected point arrays.
    assert len(captured) == 3
    for left_points, right_points in captured:
        assert left_points is left.features.points
        assert right_points is right.features.points


def test_candidate_stage_dedupes_duplicate_indices_by_pixel_pair(
    stage_module, monkeypatch,
) -> None:
    left, right = _duplicate_pixel_observations(stage_module, right_attributed=True)
    _patch_stage_inputs(
        stage_module, monkeypatch,
        np.asarray([[0, 0], [1, 0], [2, 0]], dtype=np.int64))
    left_camera, right_camera = _duplicate_pixel_cameras(stage_module)

    pair = stage_module.attribute_pair(left, right, left_camera, right_camera)

    # Indices 0 and 1 share one pixel pair, so the duplicate edge is one identity.
    assert pair.candidate.returned == 2
    assert pair.candidate.positive == 1
    assert pair.candidate.same_state_same_slot_other_block == 1
    assert pair.candidate.truth_pairs == 2
    assert pair.candidate.truth_pairs_returned == 1
    assert pair.candidate.precision == 0.5


def test_candidate_stage_dedupes_unattributed_duplicate_indices(
    stage_module, monkeypatch,
) -> None:
    left, right = _duplicate_pixel_observations(stage_module, right_attributed=False)
    _patch_stage_inputs(
        stage_module, monkeypatch,
        np.asarray([[0, 0], [1, 0], [2, 0]], dtype=np.int64))
    left_camera, right_camera = _duplicate_pixel_cameras(stage_module)

    pair = stage_module.attribute_pair(left, right, left_camera, right_camera)

    # An unattributed row has no SurfaceHit pixel; the point arrays are the only
    # source of its identity, so the two indices collapse to one candidate row.
    assert pair.candidate.returned == 2
    assert pair.candidate.unattributed == 2
    assert pair.candidate.positive == 0
    assert pair.candidate.truth_pairs == 0


# --- pooled outcome accounting ----------------------------------------------------


def test_summarise_attributions_counts_refusal_outcomes(stage_module, monkeypatch) -> None:
    left, right, left_camera, right_camera = _scene(stage_module, right_mode="same")
    selection = _matches(left, right, 8)

    monkeypatch.setattr(stage_module, "match_feature_frames", lambda *a, **k: _result(
        status="solved", reason="", matches=selection,
        pose=_pair_pose(solved=True), right_yaw_class=1))
    solved = stage_module.attribute_pair(left, right, left_camera, right_camera)

    monkeypatch.setattr(stage_module, "match_feature_frames", lambda *a, **k: _result(
        status="refused", reason="solve_pair_refused", matches=selection,
        pose=_pair_pose(solved=False)))
    abstained = stage_module.attribute_pair(left, right, left_camera, right_camera)

    monkeypatch.setattr(stage_module, "match_feature_frames", lambda *a, **k: _result(
        status="refused", reason="candidate_graph_empty",
        matches=np.empty((0, 4), dtype=np.float64)))
    absent = stage_module.attribute_pair(left, right, left_camera, right_camera)

    summary = stage_module.summarise_attributions([solved, abstained, absent])
    assert summary["pair_count"] == 3
    assert summary["solved"] == 1
    assert summary["refused"] == 2
    assert summary["refusals_with_selected_matches"] == 1
    assert summary["refusals_without_selected_matches"] == 1
    assert summary["refusals_with_pose_attempt"] == 1
    assert summary["selection_states"][stage_module.SELECTION_ABSENT] == 1
    assert summary["pose_states"][stage_module.POSE_ABSTAINED] == 1
    # Each attribution contributes its own copy of the same pair's truth surface.
    assert summary["denominators"]["correspondence_truth_pairs"] == 3 * len(_POINTS)


# --- sealed payload assembly ------------------------------------------------------


def test_build_payload_seals_and_labels_the_assisted_camera(
    stage_module, monkeypatch,
) -> None:
    left, right, _left_camera, _right_camera = _scene(stage_module, right_mode="same")
    left_camera_model = _camera(LEFT_POSITION, LEFT_YAW, LEFT_PITCH)
    right_camera_model = _camera(RIGHT_POSITION, RIGHT_YAW, RIGHT_PITCH)
    entry = _entry("d1", "scene", "g1")
    scene = FinalizedScene(
        input=entry,
        frames=(_frame_input("a", left_camera_model), _frame_input("b", right_camera_model)),
        occupancy={}, region_min=(0, 0, 0), region_max=(1, 1, 1),
    )

    monkeypatch.setattr(stage_module, "build_scene", lambda _entry: scene)
    monkeypatch.setattr(
        stage_module, "observe_joint_frame",
        lambda _scene, frame, detection=None: left if frame.frame_id == "a" else right)
    manifest = build_group_manifest([entry])
    payload = stage_module.build_payload("exports/datasets", [entry], manifest)

    assert payload["schema"] == stage_module.SCHEMA
    assert payload["camera_policy"]["mode"] == stage_module.CAMERA_POLICY_MODE
    assert payload["camera_policy"]["assisted"] is True
    assert payload["camera_policy"]["default_image_only_reconstruction"] is False
    assert payload["pooled"]["pair_count"] == 1
    assert payload["stage_attribution"]["outcomes"]["solved"] == 1
    assert payload["scenes"][0]["frames"][0]["camera"]["assisted"] is True
    assert payload["inputs"]["fold_leakage_issues"] == []
    assert payload["consistency"]["ok"] is True
    # The payload is JSON-safe and its self-seal reproduces over the body.
    json.dumps(payload, sort_keys=True, allow_nan=False)
    assert payload["seal_sha256"] == seal_record(payload)


# --- grouped split manifest and fold leakage -------------------------------------


def _entry(dataset_id: str, scene_id: str, group: str, split: str = "train",
           directory: str | None = None) -> DatasetSceneInput:
    directory = directory or f"scenes/{dataset_id}"
    return DatasetSceneInput(
        dataset_id=dataset_id,
        index_path=f"exports/datasets/{dataset_id}/index.json",
        index_sha256="0" * 64,
        scene_id=scene_id, scene_group=group, split=split, directory=directory,
        scene_root=f"exports/datasets/{dataset_id}/{directory}", hashes=(),
        screenshots=(),
    )


def test_duplicate_exports_of_one_scene_cannot_cross_folds() -> None:
    inputs = [
        _entry("canal-town", "canal-town-bbox", "group-canal"),
        _entry("canal-town-framed", "canal-town-bbox", "group-canal"),
        _entry("train-station", "train-station-spawn", "group-train"),
    ]
    manifest = build_group_manifest(inputs)

    assert manifest["group_count"] == 2
    assert check_fold_leakage(manifest) == ()
    assert check_split_leakage(inputs) == ()
    for fold in manifest["folds"]:
        assert set(fold["held_out_entries"]).isdisjoint(fold["retained_entries"])


def test_split_leakage_is_rejected_before_any_run() -> None:
    same_scene_two_groups = [
        _entry("d1", "scene", "g1"),
        _entry("d2", "scene", "g2"),
    ]
    assert check_split_leakage(same_scene_two_groups)
    with pytest.raises(PairCorrespondenceEvaluationError):
        build_group_manifest(same_scene_two_groups)


def test_fold_leakage_is_detected_when_a_group_is_on_both_sides() -> None:
    manifest = {
        "groups": [{"scene_group": "g1", "entries": ["d1:scenes/a", "d2:scenes/a"]}],
        "folds": [{
            "fold": "heldout-g1",
            "held_out_entries": ["d1:scenes/a"],
            "retained_entries": ["d2:scenes/a"],
        }],
    }
    issues = check_fold_leakage(manifest)
    assert issues and "g1" in issues[0]


# --- runner guards -----------------------------------------------------------------


def test_runner_refuses_to_overwrite_an_existing_output(
    stage_module, tmp_path: Path,
) -> None:
    output = tmp_path / "out.json"
    output.write_text("keep", encoding="utf-8")
    assert stage_module.main(
        ["--output", str(output), "--exports-root", str(tmp_path)]) == 2
    assert output.read_text(encoding="utf-8") == "keep"


def test_runner_reports_failure_without_writing_when_no_exports(
    stage_module, tmp_path: Path,
) -> None:
    output = tmp_path / "absent.json"
    missing = tmp_path / "missing-root"
    assert stage_module.main(
        ["--output", str(output), "--exports-root", str(missing)]) == 1
    assert not output.exists()
