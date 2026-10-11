"""Focused tests for the DS-R5O exact feature-point oracle diagnostic.

These tests exercise the pure evaluation-only pieces on synthetic geometry: the
same-surface separation-tolerance admission rule and its boundaries, same-surface
versus different-surface rejection, canonical pixel/pair identity, actual
descriptor-distance preservation, fail-closed missing truth/keypoint context, the
production builder/settings invariants, the six-arm pair ledger and typed absences,
pooled accounting and the runner guards. They do **not** run SIFT on the corpus;
the single pinned evaluation is a separate run recorded in the report.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import mc_photogrammetry.pipeline.joint_matching as joint_matching
from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.evaluation_pair_correspondence import (
    SURFACE_TOLERANCE_BLOCKS,
    DatasetSceneInput,
    FinalizedScene,
    FrameInput,
    SurfaceHit,
    build_group_manifest,
)
from mc_photogrammetry.pipeline.correspondence import FrameFeatures
from mc_photogrammetry.pipeline.joint_matching import JointMatchSettings
from mc_photogrammetry.reconstruction.camera import CameraModel, project_points

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

_OBSERVATION_TYPE = None
"""The reviewed DS-R5K ``OracleFrameObservation`` class, bound by the fixture."""


@pytest.fixture(scope="module")
def point_module():
    """Load the evaluation script as a module (it defines module-level dataclasses)."""
    script = (Path(__file__).resolve().parents[1]
              / "scripts" / "evaluation" / "evaluate_exact_point_oracle.py")
    spec = importlib.util.spec_from_file_location("ds_r5o_exact_point_oracle", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    global _OBSERVATION_TYPE
    _OBSERVATION_TYPE = module.R5K.OracleFrameObservation
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
        frame_id=frame_id, points=np.asarray(points, dtype=np.float64),
        descriptors=np.asarray(descriptors, dtype=np.float64),
        width=WIDTH, height=HEIGHT, usable_pixels=WIDTH * HEIGHT,
    )


def _hit(frame_id, coordinate, slot, world_point, pixel,
         state_name="minecraft:stone"):
    return SurfaceHit(
        frame_id=frame_id, coordinate=coordinate, slot=slot,
        world_point=tuple(float(value) for value in world_point),
        camera_depth=1.0, state=BlockState(state_name),
        pixel=(float(pixel[0]), float(pixel[1])),
    )


def _observation(frame_id, points, descriptors, hits):
    assert _OBSERVATION_TYPE is not None, "the point_module fixture must load first"
    return _OBSERVATION_TYPE(
        frame_id=frame_id, width=WIDTH, height=HEIGHT,
        usable_pixels=WIDTH * HEIGHT, boundary_dropped=0,
        features=_features(frame_id, points, descriptors), hits=hits,
        no_truth_reasons=(),
    )


def _pair_tables(separation, *, slot="north", right_slot=None,
                 right_coordinate=(0, 0, 0)):
    """Two keypoints per frame on one surface, offset by ``separation`` in x.

    ``right_slot``/``right_coordinate`` move the right frame's observed surface
    while keeping the same continuous world points, so a different-coordinate or
    different-slot surface can be tested at identical separation.
    """
    left_points = np.asarray([[10.0, 20.0], [11.0, 21.0]], dtype=np.float64)
    right_points = np.asarray([[30.0, 40.0], [31.0, 41.0]], dtype=np.float64)
    left_descriptors = np.asarray([[1.0, 0.0], [2.0, 1.0]], dtype=np.float64)
    right_descriptors = np.asarray([[0.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    left_hits = (
        _hit("a", (0, 0, 0), slot, (0.5, 0.5, 1.0), (10.0, 20.0)),
        _hit("a", (0, 0, 0), slot, (0.5 + separation, 0.5, 1.0), (11.0, 21.0)),
    )
    observed_slot = slot if right_slot is None else right_slot
    right_world = ((0.5, 0.5, 1.0), (0.5 + separation, 0.5, 1.0))
    right_hits = tuple(
        _hit("b", right_coordinate, observed_slot, right_world[index],
             right_points[index])
        for index in range(2)
    )
    left = _observation("a", left_points, left_descriptors, left_hits)
    right = _observation("b", right_points, right_descriptors, right_hits)
    return left, right


def _edges(graph):
    return sorted(tuple(int(value) for value in row) for row in graph.edges)


def _distances(graph):
    return [float(value) for value in graph.distances]


# --- point-oracle graph construction and tolerance boundaries --------------------


def test_point_oracle_admits_same_surface_pairs_within_the_tolerance(
    point_module,
) -> None:
    left, right = _pair_tables(0.1)
    tight = point_module.build_point_oracle_graph(
        left.features, right.features, left.hits, right.hits, tolerance=0.05)
    loose = point_module.build_point_oracle_graph(
        left.features, right.features, left.hits, right.hits, tolerance=0.125)

    # Offset 0.1 exceeds 0.05 but is inside 0.125: the sweep changes admission.
    assert _edges(tight) == [(0, 0), (1, 1)]
    assert _edges(loose) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert loose.total_edges == 4
    assert loose.truncated == 0


def test_point_oracle_rejects_pairs_above_each_tolerance(point_module) -> None:
    left, right = _pair_tables(0.4)
    for tolerance in (0.05, 0.125, 0.25):
        graph = point_module.build_point_oracle_graph(
            left.features, right.features, left.hits, right.hits,
            tolerance=tolerance)
        assert _edges(graph) == [(0, 0), (1, 1)]
    admitted = point_module.build_point_oracle_graph(
        left.features, right.features, left.hits, right.hits, tolerance=0.5)
    assert _edges(admitted) == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_point_oracle_includes_pairs_exactly_at_the_tolerance(point_module) -> None:
    """The admission rule is inclusive: separation == tolerance is admitted."""
    left, right = _pair_tables(0.25)  # 0.25 is exactly representable.
    exact = point_module.build_point_oracle_graph(
        left.features, right.features, left.hits, right.hits, tolerance=0.25)
    below = point_module.build_point_oracle_graph(
        left.features, right.features, left.hits, right.hits, tolerance=0.125)

    assert _edges(exact) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert _edges(below) == [(0, 0), (1, 1)]

    slightly_more = _pair_tables(0.25 + 1e-6)
    over = point_module.build_point_oracle_graph(
        slightly_more[0].features, slightly_more[1].features,
        slightly_more[0].hits, slightly_more[1].hits, tolerance=0.25)
    assert _edges(over) == [(0, 0), (1, 1)]


def test_point_oracle_rejects_a_different_surface_at_any_proximity(
    point_module,
) -> None:
    left, right = _pair_tables(0.0, right_slot="south", right_coordinate=(9, 0, 0))
    for tolerance in point_module.POINT_TOLERANCES:
        graph = point_module.build_point_oracle_graph(
            left.features, right.features, left.hits, right.hits,
            tolerance=tolerance)
        # Identical world points on a different coordinate/slot are not the same
        # visible surface, so no pair is admitted.
        assert graph.edges.shape == (0, 2)
        assert graph.total_edges == 0


def test_point_oracle_rejects_the_same_coordinate_on_a_different_slot(
    point_module,
) -> None:
    left, right = _pair_tables(0.0, right_slot="south", right_coordinate=(0, 0, 0))
    graph = point_module.build_point_oracle_graph(
        left.features, right.features, left.hits, right.hits, tolerance=0.5)
    assert graph.edges.shape == (0, 2)


def test_point_oracle_matches_the_reviewed_surface_graph_at_its_tolerance(
    point_module,
) -> None:
    left, right = _pair_tables(0.05)
    point = point_module.build_point_oracle_graph(
        left.features, right.features, left.hits, right.hits,
        tolerance=SURFACE_TOLERANCE_BLOCKS)
    surface = point_module.R5K.build_oracle_graph(
        left.features, right.features, left.hits, right.hits,
        tolerance=SURFACE_TOLERANCE_BLOCKS)

    assert _edges(point) == _edges(surface)
    assert _distances(point) == pytest.approx(_distances(surface))


def test_point_oracle_preserves_actual_descriptor_distances_and_sort_order(
    point_module,
) -> None:
    left, right = _pair_tables(0.5)
    graph = point_module.build_point_oracle_graph(
        left.features, right.features, left.hits, right.hits, tolerance=0.5)

    for position, (left_index, right_index) in enumerate(graph.edges):
        expected = float(np.linalg.norm(
            left.features.descriptors[int(left_index)]
            - right.features.descriptors[int(right_index)]))
        assert graph.distances[position] == pytest.approx(expected)
    order = [
        (float(graph.distances[index]), int(graph.edges[index, 0]),
         int(graph.edges[index, 1]))
        for index in range(graph.edges.shape[0])
    ]
    assert order == sorted(order)
    assert float(np.mean(graph.distances)) > 0.0


def test_point_oracle_collapses_duplicate_pixels_to_one_canonical_edge(
    point_module,
) -> None:
    left_points = np.asarray(
        [[10.0, 20.0], [10.0, 20.0], [40.0, 50.0]], dtype=np.float64)
    right_points = np.asarray([[30.0, 40.0], [70.0, 80.0]], dtype=np.float64)
    left_descriptors = np.asarray(
        [[1.0, 0.0], [9.0, 9.0], [0.0, 1.0]], dtype=np.float64)
    right_descriptors = np.asarray([[0.0, 1.0], [5.0, 5.0]], dtype=np.float64)
    left_hits = (
        _hit("a", (0, 0, 0), "south", (0.5, 0.5, 1.0), (10.0, 20.0)),
        _hit("a", (0, 0, 0), "south", (0.5, 0.5, 1.0), (10.0, 20.0)),
        _hit("a", (1, 0, 0), "south", (1.5, 0.5, 1.0), (40.0, 50.0)),
    )
    right_hits = (
        _hit("b", (0, 0, 0), "south", (0.5, 0.5, 1.0), (30.0, 40.0)),
        _hit("b", (2, 0, 0), "south", (2.5, 0.5, 1.0), (70.0, 80.0)),
    )
    left = _observation("a", left_points, left_descriptors, left_hits)
    right = _observation("b", right_points, right_descriptors, right_hits)

    graph = point_module.build_point_oracle_graph(
        left.features, right.features, left.hits, right.hits, tolerance=0.05)

    assert graph.edges.shape == (1, 2)
    assert tuple(int(value) for value in graph.edges[0]) == (0, 0)
    expected = float(np.linalg.norm(left_descriptors[0] - right_descriptors[0]))
    assert graph.distances[0] == pytest.approx(expected)


def test_point_oracle_empty_truth_builds_an_empty_graph(point_module) -> None:
    left, right = _pair_tables(0.0)
    occluded = (None, None)
    left = _observation("a", left.features.points, left.features.descriptors, occluded)
    right = _observation("b", right.features.points, right.features.descriptors, occluded)

    graph = point_module.build_point_oracle_graph(
        left.features, right.features, left.hits, right.hits, tolerance=0.5)
    assert graph.edges.shape == (0, 2)
    assert graph.distances.size == 0
    assert graph.left_keypoints == len(left.features.points)


# --- fail-closed context ----------------------------------------------------------


def test_point_oracle_rejects_a_truth_pixel_without_a_keypoint(point_module) -> None:
    left_points = np.asarray([[10.0, 20.0]], dtype=np.float64)
    right_points = np.asarray([[30.0, 40.0]], dtype=np.float64)
    descriptors = np.asarray([[1.0, 1.0]], dtype=np.float64)
    surface = _hit("a", (0, 0, 0), "south", (0.5, 0.5, 1.0), (10.0, 20.0))
    orphan = _hit("b", (0, 0, 0), "south", (0.5, 0.5, 1.0), (999.0, 999.0))
    left = _observation("a", left_points, descriptors, (surface,))
    right = _observation("b", right_points, descriptors, (orphan,))

    with pytest.raises(point_module.ExactPointOracleError):
        point_module.build_point_oracle_graph(
            left.features, right.features, left.hits, right.hits, tolerance=0.5)


def test_point_oracle_rejects_a_short_truth_table(point_module) -> None:
    left, right = _pair_tables(0.0)
    with pytest.raises(point_module.ExactPointOracleError):
        point_module.build_point_oracle_graph(
            left.features, right.features, left.hits[:1], right.hits, tolerance=0.5)


@pytest.mark.parametrize("tolerance", [0.0, -0.125, float("nan"), float("inf")])
def test_point_oracle_rejects_a_bad_tolerance(point_module, tolerance) -> None:
    left, right = _pair_tables(0.0)
    with pytest.raises(point_module.ExactPointOracleError):
        point_module.build_point_oracle_graph(
            left.features, right.features, left.hits, right.hits,
            tolerance=tolerance)


# --- full pair runs ---------------------------------------------------------------


def _scene(module, *, right_mode="same", descriptors=None,
           right_descriptors=None):
    """Project 20 independent 3D points and attach a synthetic truth surface."""
    left_camera = _camera(LEFT_POSITION, LEFT_YAW, LEFT_PITCH)
    right_camera = _camera(RIGHT_POSITION, RIGHT_YAW, RIGHT_PITCH)
    left_projected = project_points(_POINTS, left_camera, width=WIDTH, height=HEIGHT)
    right_projected = project_points(_POINTS, right_camera, width=WIDTH, height=HEIGHT)
    left_points = np.asarray([[p[0], p[1]] for p in left_projected], dtype=np.float64)
    right_points = np.asarray([[p[0], p[1]] for p in right_projected], dtype=np.float64)
    if descriptors is None:
        descriptors = np.random.default_rng(3).normal(size=(len(left_points), 16))
    left_hits = tuple(
        _hit("a", (index, 0, 0), "north", _POINTS[index], left_points[index])
        for index in range(len(_POINTS)))
    left = _observation("a", left_points, descriptors, left_hits)
    if right_mode == "same":
        right_hits = tuple(
            _hit("b", (index, 0, 0), "north", _POINTS[index], right_points[index])
            for index in range(len(_POINTS)))
    elif right_mode == "wrong":
        right_hits = tuple(
            _hit("b", (index, 0, 0), "south",
                 (_POINTS[index][0] + 50.0, _POINTS[index][1], _POINTS[index][2]),
                 right_points[index], "minecraft:dirt")
            for index in range(len(_POINTS)))
    elif right_mode == "occluded":
        right_hits = (None,) * len(right_points)
    else:
        raise ValueError(right_mode)
    right = _observation(
        "b", right_points,
        descriptors if right_descriptors is None else right_descriptors, right_hits)
    return (
        left, right,
        module.oracle_camera_estimate(_frame_input("a", left_camera)),
        module.oracle_camera_estimate(_frame_input("b", right_camera)),
    )


def test_attribute_pair_runs_ordered_arms_and_restores_the_production_builder(
    point_module,
) -> None:
    original = joint_matching.reciprocal_topk_graph
    left, right, left_camera, right_camera = _scene(point_module)

    pair = point_module.attribute_pair(left, right, left_camera, right_camera)

    assert [item.arm for item in pair.arms] == list(point_module.ARM_ORDER)
    assert joint_matching.reciprocal_topk_graph is original
    assert pair.left_id == "a" and pair.right_id == "b"


def test_attribute_pair_leaves_production_settings_unchanged(point_module) -> None:
    left, right, left_camera, right_camera = _scene(point_module)
    settings = JointMatchSettings()
    before = dataclasses.asdict(settings)

    point_module.attribute_pair(
        left, right, left_camera, right_camera, matcher=settings)

    assert dataclasses.asdict(settings) == before
    assert dataclasses.asdict(JointMatchSettings()) == before


def test_attribute_pair_binds_each_arm_tolerance_and_denominators(point_module) -> None:
    left, right, left_camera, right_camera = _scene(point_module)
    pair = point_module.attribute_pair(left, right, left_camera, right_camera)

    assert pair.production_arm.separation_tolerance_blocks is None
    assert pair.surface_arm.separation_tolerance_blocks == pytest.approx(
        SURFACE_TOLERANCE_BLOCKS)
    for tolerance, arm in zip(point_module.POINT_TOLERANCES, pair.point_arms):
        assert arm.separation_tolerance_blocks == pytest.approx(float(tolerance))
    assert pair.surface_truth_pairs == len(_POINTS)
    assert dict(pair.point_truth_pairs) == {
        float(tolerance): len(_POINTS) for tolerance in point_module.POINT_TOLERANCES
    }
    assert pair.left_keypoints_with_truth == len(_POINTS)
    assert pair.left_keypoints_without_truth == 0


def test_attribute_pair_is_deterministic_across_calls(point_module) -> None:
    scene = _scene(point_module)
    first = point_module.attribute_pair(*scene)
    second = point_module.attribute_pair(*scene)

    for arm in point_module.ARM_ORDER:
        assert np.array_equal(first.arm(arm).graph.edges, second.arm(arm).graph.edges)
        assert first.arm(arm).result.status == second.arm(arm).result.status
        assert first.arm(arm).result.reason == second.arm(arm).result.reason


def test_wrong_surface_and_occluded_scenes_refuse_with_typed_emptiness(
    point_module,
) -> None:
    wrong = point_module.attribute_pair(*_scene(point_module, right_mode="wrong"))
    assert wrong.surface_truth_pairs == 0
    assert wrong.surface_arm.graph_empty is True
    assert wrong.surface_arm.result.reason == "candidate_graph_empty"
    assert wrong.surface_arm.solved is False

    occluded = point_module.attribute_pair(*_scene(point_module, right_mode="occluded"))
    assert occluded.surface_truth_pairs == 0
    assert occluded.surface_arm.graph_empty is True
    assert occluded.surface_arm.solved is False


def test_surface_oracle_and_exact_point_oracle_differ_on_within_face_points(
    point_module,
) -> None:
    """Mutation control: surface-only evidence is not sufficient for the sweep."""
    left, right = _pair_tables(0.1)
    surface = point_module.build_point_oracle_graph(
        left.features, right.features, left.hits, right.hits,
        tolerance=SURFACE_TOLERANCE_BLOCKS)
    tight = point_module.build_point_oracle_graph(
        left.features, right.features, left.hits, right.hits, tolerance=0.05)
    loose = point_module.build_point_oracle_graph(
        left.features, right.features, left.hits, right.hits, tolerance=0.125)

    assert surface.edges.shape[0] == 2
    assert tight.edges.shape[0] == 2
    assert loose.edges.shape[0] == 4
    # The exact-point sweep therefore changes both the denominator and the graph.
    assert len(point_module.construct_truth_pairs(
        left.hits, right.hits, tolerance=0.05)) == 2
    assert len(point_module.construct_truth_pairs(
        left.hits, right.hits, tolerance=0.125)) == 4


# --- records and pooled accounting ------------------------------------------------


def test_pair_record_reports_denominators_censoring_and_typed_absences(
    point_module,
) -> None:
    left, right, left_camera, right_camera = _scene(point_module)
    pair = point_module.attribute_pair(left, right, left_camera, right_camera)
    record = point_module.pair_record(pair)

    assert record["left_frame_id"] == "a" and record["right_frame_id"] == "b"
    assert record["arms"] == list(point_module.ARM_ORDER)
    assert record["surface_truth_pairs"] == len(_POINTS)
    assert set(record["point_truth_pairs"]) == {"0.05", "0.125", "0.25", "0.5"}
    assert record["left_keypoints_with_truth"] == len(_POINTS)
    assert record["left_keypoints_without_truth"] == 0
    assert record["axis_stage"]["available"] is False
    assert record["axis_stage"]["axis_coverage"] is None
    assert record["placed_cell_geometry"]["available"] is False
    assert record["placed_cell_geometry"]["placed_cells"] is None
    assert set(record["tolerance_sensitivity"]) == {"0.05", "0.125", "0.25", "0.5"}
    assert record[point_module.ARM_SURFACE]["status"] == "solved"
    json.dumps(record, sort_keys=True, allow_nan=False)


def test_summarise_attributions_keeps_arms_separate(point_module) -> None:
    solved = point_module.attribute_pair(*_scene(point_module))
    refused = point_module.attribute_pair(*_scene(point_module, right_mode="occluded"))

    summary = point_module.summarise_attributions([solved, refused])

    assert summary["pair_count"] == 2
    assert summary["surface_truth_pairs"] == len(_POINTS)
    assert summary["pairs_with_surface_truth"] == 1
    assert summary["pairs_with_empty_surface_truth"] == 1
    assert set(summary["arms"]) == set(point_module.ARM_ORDER)
    assert summary["arms"][point_module.ARM_PRODUCTION]["solved"] == 2
    assert summary["arms"][point_module.ARM_SURFACE]["solved"] == 1
    assert summary["arms"][point_module.ARM_SURFACE][
        "candidate_stage_surface_truth"]["truth_pairs_returned"] == len(_POINTS)
    assert summary["ceilings"]["surface_oracle_candidate_reach"] == len(_POINTS)
    assert summary["ceilings"]["same_point_oracle_candidate_reach"]["0.05"] == (
        len(_POINTS))
    assert summary["same_surface_not_same_feature_point"]
    assert summary["exact_point_tolerance_caveat"]
    for stages in summary["positive_identity_mismatch"].values():
        assert stages["candidate_stage_surface_truth"] == 0
        assert stages["selected_stage_surface_truth"] == 0
    json.dumps(summary, sort_keys=True, allow_nan=False)


# --- sealed payload assembly ------------------------------------------------------


def _entry(dataset_id: str, scene_id: str, group: str, frames: int,
           split: str = "train") -> DatasetSceneInput:
    return DatasetSceneInput(
        dataset_id=dataset_id,
        index_path=f"exports/datasets/{dataset_id}/index.json",
        index_sha256="0" * 64,
        scene_id=scene_id, scene_group=group, split=split,
        directory=f"scenes/{scene_id}",
        scene_root=f"exports/datasets/{dataset_id}/scenes/{scene_id}",
        hashes=(),
        screenshots=tuple((f"frame-{index:02d}.png", "1" * 64)
                          for index in range(frames)),
    )


def test_build_payload_seals_and_labels_the_exact_point_oracle(
    point_module, monkeypatch,
) -> None:
    left, right, _lc, _rc = _scene(point_module)
    entry = _entry("fresh-c4-canal-town-framed-2026-09-12",
                   "canal-town-architectural-component-bbox", "group-canal", 8)
    left_camera = _camera(LEFT_POSITION, LEFT_YAW, LEFT_PITCH)
    right_camera = _camera(RIGHT_POSITION, RIGHT_YAW, RIGHT_PITCH)
    scene = FinalizedScene(
        input=entry,
        frames=(_frame_input("a", left_camera), _frame_input("b", right_camera)),
        occupancy={}, region_min=(0, 0, 0), region_max=(1, 1, 1),
    )

    monkeypatch.setattr(point_module, "build_scene", lambda _entry: scene)
    monkeypatch.setattr(
        point_module, "observe_oracle_frame",
        lambda _scene, frame, detection=None: left if frame.frame_id == "a" else right)
    manifest = build_group_manifest([entry])
    payload = point_module.build_payload(
        "exports/datasets", [entry], manifest, entry)

    assert payload["schema"] == point_module.SCHEMA
    assert payload["created_for"] == "DS-R5O"
    assert payload["camera_policy"]["assisted"] is True
    assert payload["camera_policy"]["default_image_only_reconstruction"] is False
    assert payload["oracle_policy"]["truth_assisted_upper_bound"] is True
    assert payload["oracle_policy"]["production_result"] is False
    assert payload["oracle_policy"]["tolerance_sweep_blocks"] == [0.05, 0.125, 0.25, 0.5]
    assert payload["scene"]["pair_count"] == 1
    assert payload["summary"]["arms"][point_module.ARM_SURFACE]["solved"] == 1
    assert payload["stage_availability"]["plane_graph_axis_stage"]["available"] is False
    assert payload["consistency"]["ok"] is True
    assert payload["seal_sha256"] == point_module.seal_record(payload)
    json.dumps(payload, sort_keys=True, allow_nan=False)


# --- runner guards -----------------------------------------------------------------


def test_detection_settings_mirror_the_production_joint_path(point_module) -> None:
    matcher = JointMatchSettings()
    detection = point_module.detection_settings(matcher)

    assert detection.max_features == matcher.max_features == 4000
    assert detection.mask_boundary_pad_px == matcher.mask_boundary_pad_px == 8.0
    assert detection.ratio_test == 0.75


def test_sealed_artifact_guard_is_case_insensitive(point_module) -> None:
    assert point_module.output_targets_a_sealed_artifact(
        ".tmp-oracle-candidate-graph-20261003.json") is True
    assert point_module.output_targets_a_sealed_artifact(
        ".TMP-ORACLE-CANDIDATE-GRAPH-20261003.JSON") is True
    assert point_module.output_targets_a_sealed_artifact(
        "sub/dir/.tmp-oracle-support-sensitivity-20261003.json") is True
    assert point_module.output_targets_a_sealed_artifact(
        point_module.DEFAULT_OUTPUT) is False


def test_runner_refuses_to_overwrite_an_existing_output(
    point_module, tmp_path: Path,
) -> None:
    output = tmp_path / "out.json"
    output.write_text("keep", encoding="utf-8")
    assert point_module.main(
        ["--output", str(output), "--exports-root", str(tmp_path)]) == 2
    assert output.read_text(encoding="utf-8") == "keep"


def test_runner_refuses_to_write_a_sealed_artifact(point_module, tmp_path: Path) -> None:
    output = tmp_path / ".TMP-ORACLE-CANDIDATE-GRAPH-20261003.JSON"
    assert point_module.main(
        ["--output", str(output), "--exports-root", str(tmp_path)]) == 2
    assert not output.exists()


def test_runner_reports_failure_without_writing_when_no_exports(
    point_module, tmp_path: Path,
) -> None:
    output = tmp_path / "absent.json"
    missing = tmp_path / "missing-root"
    assert point_module.main(
        ["--output", str(output), "--exports-root", str(missing)]) == 1
    assert not output.exists()


def test_point_oracle_surface_distances_are_finite(point_module) -> None:
    left, right = _pair_tables(0.5)
    graph = point_module.build_point_oracle_graph(
        left.features, right.features, left.hits, right.hits, tolerance=0.5)
    assert all(math.isfinite(float(value)) for value in graph.distances)
