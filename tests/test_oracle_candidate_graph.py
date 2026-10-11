"""Focused tests for the DS-R5K oracle candidate-graph upper bound.

These tests exercise the pure evaluation-only pieces on synthetic geometry: the
strict same-surface oracle graph, duplicate feature-coordinate collapsing, actual
descriptor-distance preservation, the injection/restoration of the production
reciprocal graph builder, empty-truth refusal, exact same/wrong surface
denominators, canonical-entry selection and the one-shot runner guards. They do
**not** run SIFT on the corpus; the single corpus evaluation is a separate
authorized run.
"""
from __future__ import annotations

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
    DatasetSceneInput,
    FinalizedScene,
    FrameInput,
    SurfaceHit,
    build_group_manifest,
)
from mc_photogrammetry.pipeline.correspondence import FrameFeatures, ReciprocalGraph
from mc_photogrammetry.pipeline.joint_matching import JointMatchSettings, JointPairResult
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


@pytest.fixture(scope="module")
def oracle_module():
    """Load the evaluation script as a module (it defines module-level dataclasses)."""
    script = (Path(__file__).resolve().parents[1]
              / "scripts" / "evaluation" / "evaluate_oracle_candidate_graph.py")
    spec = importlib.util.spec_from_file_location("ds_r5k_oracle_graph", script)
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


def _observation(oracle_module, frame_id, points, descriptors, hits):
    return oracle_module.OracleFrameObservation(
        frame_id=frame_id, width=WIDTH, height=HEIGHT,
        usable_pixels=WIDTH * HEIGHT, boundary_dropped=0,
        features=_features(frame_id, points, descriptors), hits=hits,
        no_truth_reasons=(),
    )


def _scene(oracle_module, *, right_mode="same", descriptors=None,
           right_descriptors=None):
    """Project 20 independent 3D points and attach a synthetic truth surface.

    ``right_mode`` selects same-surface truth, a structurally different surface
    (``wrong``) or no surface at all (``occluded``) in the right frame. The two
    frames share ``descriptors`` by default (so the true pairs are exact nearest
    neighbours and the baseline matcher solves); ``right_descriptors`` overrides
    the right table for descriptor-distance assertions.
    """
    left_camera = _camera(LEFT_POSITION, LEFT_YAW, LEFT_PITCH)
    right_camera = _camera(RIGHT_POSITION, RIGHT_YAW, RIGHT_PITCH)
    left_projected = project_points(_POINTS, left_camera, width=WIDTH, height=HEIGHT)
    right_projected = project_points(_POINTS, right_camera, width=WIDTH, height=HEIGHT)
    left_points = np.asarray([[p[0], p[1]] for p in left_projected], dtype=np.float64)
    right_points = np.asarray([[p[0], p[1]] for p in right_projected], dtype=np.float64)
    if descriptors is None:
        descriptors = np.random.default_rng(3).normal(size=(len(left_points), 16))
    left = _observation(
        oracle_module, "a", left_points, descriptors,
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
    right = _observation(
        oracle_module, "b", right_points,
        descriptors if right_descriptors is None else right_descriptors, right_hits)
    left_camera_estimate = oracle_module.oracle_camera_estimate(
        _frame_input("a", left_camera))
    right_camera_estimate = oracle_module.oracle_camera_estimate(
        _frame_input("b", right_camera))
    return left, right, left_camera_estimate, right_camera_estimate


# --- oracle graph construction ---------------------------------------------------


def test_oracle_graph_uses_unique_truth_pixel_pairs(oracle_module) -> None:
    left, right, _left_camera, _right_camera = _scene(oracle_module, right_mode="same")
    graph = oracle_module.build_oracle_graph(
        left.features, right.features, left.hits, right.hits)

    assert isinstance(graph, ReciprocalGraph)
    assert graph.edges.shape == (len(_POINTS), 2)
    assert graph.total_edges == len(_POINTS)
    assert graph.truncated == 0
    assert graph.left_keypoints == len(left.features.points)
    assert graph.right_keypoints == len(right.features.points)
    # Every projected surface is distinct, so the canonical identity is (i, i).
    assert sorted(tuple(int(v) for v in row) for row in graph.edges) == [
        (index, index) for index in range(len(_POINTS))
    ]


def test_oracle_graph_preserves_actual_descriptor_distances(oracle_module) -> None:
    shared = np.random.default_rng(3).normal(size=(len(_POINTS), 16))
    left, right, _left_camera, _right_camera = _scene(
        oracle_module, right_mode="same", descriptors=shared,
        right_descriptors=shared + 0.5)
    graph = oracle_module.build_oracle_graph(
        left.features, right.features, left.hits, right.hits)

    left_descriptors = left.features.descriptors
    right_descriptors = right.features.descriptors
    for position, (left_index, right_index) in enumerate(graph.edges):
        expected = float(np.linalg.norm(
            left_descriptors[int(left_index)] - right_descriptors[int(right_index)]))
        assert graph.distances[position] == pytest.approx(expected)
    # The distances are the real non-trivial descriptor costs.
    assert float(np.mean(graph.distances)) > 0.0


def test_oracle_graph_sorts_edges_by_distance_then_indices(oracle_module) -> None:
    shared = np.random.default_rng(3).normal(size=(len(_POINTS), 16))
    left, right, _left_camera, _right_camera = _scene(
        oracle_module, right_mode="same", descriptors=shared,
        right_descriptors=shared + 0.5)
    graph = oracle_module.build_oracle_graph(
        left.features, right.features, left.hits, right.hits)

    order = [
        (float(graph.distances[index]), int(graph.edges[index, 0]),
         int(graph.edges[index, 1]))
        for index in range(graph.edges.shape[0])
    ]
    assert order == sorted(order)


def test_duplicate_feature_coordinates_collapse_to_one_canonical_edge(
    oracle_module,
) -> None:
    left_points = np.asarray(
        [[10.0, 20.0], [10.0, 20.0], [40.0, 50.0]], dtype=np.float64)
    right_points = np.asarray([[30.0, 40.0], [70.0, 80.0]], dtype=np.float64)
    left_descriptors = np.asarray(
        [[1.0, 0.0], [9.0, 9.0], [0.0, 1.0]], dtype=np.float64)
    right_descriptors = np.asarray([[0.0, 1.0], [5.0, 5.0]], dtype=np.float64)

    def hit(frame_id, coordinate, world_point, pixel):
        return SurfaceHit(
            frame_id=frame_id, coordinate=coordinate, slot="south",
            world_point=world_point, camera_depth=1.0,
            state=BlockState("minecraft:stone"), pixel=pixel,
        )

    # Indices 0 and 1 are two keypoints at the very same pixel/surface.
    left_hits = (
        hit("a", (0, 0, 0), (0.5, 0.5, 1.0), (10.0, 20.0)),
        hit("a", (0, 0, 0), (0.5, 0.5, 1.0), (10.0, 20.0)),
        hit("a", (1, 0, 0), (1.5, 0.5, 1.0), (40.0, 50.0)),
    )
    right_hits = (
        hit("b", (0, 0, 0), (0.5, 0.5, 1.0), (30.0, 40.0)),
        hit("b", (2, 0, 0), (2.5, 0.5, 1.0), (70.0, 80.0)),
    )
    left = _observation(oracle_module, "a", left_points, left_descriptors, left_hits)
    right = _observation(oracle_module, "b", right_points, right_descriptors, right_hits)

    graph = oracle_module.build_oracle_graph(
        left.features, right.features, left.hits, right.hits)

    # One unique truth pixel pair -> one edge, using the first (canonical) index.
    assert graph.edges.shape == (1, 2)
    assert tuple(int(value) for value in graph.edges[0]) == (0, 0)
    expected = float(np.linalg.norm(left_descriptors[0] - right_descriptors[0]))
    assert graph.distances[0] == pytest.approx(expected)
    assert graph.distances[0] != pytest.approx(
        float(np.linalg.norm(left_descriptors[1] - right_descriptors[0])))


def test_oracle_graph_rejects_a_truth_pixel_without_a_keypoint(oracle_module) -> None:
    left_points = np.asarray([[10.0, 20.0]], dtype=np.float64)
    right_points = np.asarray([[30.0, 40.0]], dtype=np.float64)
    descriptors = np.asarray([[1.0, 1.0]], dtype=np.float64)
    surface = SurfaceHit(
        frame_id="a", coordinate=(0, 0, 0), slot="south",
        world_point=(0.5, 0.5, 1.0), camera_depth=1.0,
        state=BlockState("minecraft:stone"), pixel=(10.0, 20.0),
    )
    orphan = SurfaceHit(
        frame_id="b", coordinate=(0, 0, 0), slot="south",
        world_point=(0.5, 0.5, 1.0), camera_depth=1.0,
        state=BlockState("minecraft:stone"), pixel=(999.0, 999.0),
    )
    left = _observation(oracle_module, "a", left_points, descriptors, (surface,))
    right = _observation(oracle_module, "b", right_points, descriptors, (orphan,))

    with pytest.raises(oracle_module.OracleCandidateGraphError):
        oracle_module.build_oracle_graph(
            left.features, right.features, left.hits, right.hits)


def test_empty_truth_builds_an_empty_graph(oracle_module) -> None:
    left, right, _left_camera, _right_camera = _scene(
        oracle_module, right_mode="occluded")
    graph = oracle_module.build_oracle_graph(
        left.features, right.features, left.hits, right.hits)

    assert graph.edges.shape == (0, 2)
    assert graph.distances.size == 0
    assert graph.total_edges == 0
    assert graph.left_keypoints == len(left.features.points)


# --- graph injection and restoration ---------------------------------------------


def _one_edge_graph() -> ReciprocalGraph:
    return ReciprocalGraph(
        edges=np.asarray([[0, 0]], dtype=np.int64),
        distances=np.asarray([0.25], dtype=np.float64),
        left_keypoints=1, right_keypoints=1, total_edges=1, truncated=0,
    )


def test_injected_candidate_graph_replaces_and_restores(oracle_module) -> None:
    original = joint_matching.reciprocal_topk_graph
    graph = _one_edge_graph()

    with oracle_module.injected_candidate_graph(graph) as injected:
        assert injected is graph
        assert joint_matching.reciprocal_topk_graph(object(), object()) is graph
        assert joint_matching.reciprocal_topk_graph is not original

    assert joint_matching.reciprocal_topk_graph is original


def test_injected_candidate_graph_restores_after_an_exception(oracle_module) -> None:
    original = joint_matching.reciprocal_topk_graph

    with pytest.raises(RuntimeError):
        with oracle_module.injected_candidate_graph(_one_edge_graph()):
            raise RuntimeError("selector blew up")

    assert joint_matching.reciprocal_topk_graph is original


def test_attribute_pair_restores_the_production_builder(oracle_module) -> None:
    original = joint_matching.reciprocal_topk_graph
    left, right, left_camera, right_camera = _scene(oracle_module, right_mode="same")

    oracle_module.attribute_pair(left, right, left_camera, right_camera)

    assert joint_matching.reciprocal_topk_graph is original


def test_attribute_pair_builds_the_oracle_graph_from_truth(oracle_module) -> None:
    left, right, left_camera, right_camera = _scene(oracle_module, right_mode="same")
    pair = oracle_module.attribute_pair(left, right, left_camera, right_camera)

    assert pair.oracle_graph.edges.shape == (len(_POINTS), 2)
    assert pair.oracle_candidate.returned == len(_POINTS)
    assert pair.oracle_candidate.truth_pairs == len(_POINTS)


# --- synthetic same/wrong/occluded surfaces and denominators ---------------------


def test_same_surface_oracle_graph_solves_with_exact_reach(oracle_module) -> None:
    left, right, left_camera, right_camera = _scene(oracle_module, right_mode="same")
    pair = oracle_module.attribute_pair(left, right, left_camera, right_camera)

    assert pair.baseline_solved is True
    assert pair.oracle_solved is True
    assert pair.truth_pairs == len(_POINTS)
    # The oracle candidate graph reaches the whole truth denominator exactly.
    assert pair.oracle_candidate.truth_pairs_returned == len(_POINTS)
    assert pair.oracle_candidate.precision == 1.0
    # And the unchanged selector keeps all of it.
    assert pair.oracle_selected.returned == len(_POINTS)
    assert pair.oracle_selected.positive == len(_POINTS)
    assert pair.oracle_selected.truth_pairs_returned == len(_POINTS)
    assert pair.oracle_selected.precision == 1.0
    assert pair.oracle_selected.recall == 1.0
    assert pair.oracle_result.support >= pair.oracle_result.support_floor >= 1


def test_wrong_surface_oracle_graph_has_no_truth_denominator(oracle_module) -> None:
    left, right, left_camera, right_camera = _scene(oracle_module, right_mode="wrong")
    pair = oracle_module.attribute_pair(left, right, left_camera, right_camera)

    assert pair.truth_pairs == 0
    assert pair.oracle_graph_empty is True
    assert pair.oracle_candidate.returned == 0
    assert pair.oracle_candidate.positive == 0
    assert pair.oracle_candidate.recall is None
    assert pair.oracle_selected.positive == 0
    assert pair.oracle_selected.recall is None


def test_empty_truth_oracle_refuses_without_raising(oracle_module) -> None:
    left, right, left_camera, right_camera = _scene(
        oracle_module, right_mode="occluded")
    pair = oracle_module.attribute_pair(left, right, left_camera, right_camera)

    assert pair.truth_pairs == 0
    assert pair.oracle_graph_empty is True
    assert pair.oracle_solved is False
    assert pair.oracle_result.status == "refused"
    assert pair.oracle_result.reason == "candidate_graph_empty"
    assert pair.oracle_result.matches.size == 0
    record = oracle_module.pair_record(pair)
    assert record["oracle_graph_empty"] is True
    assert record["oracle_solved"] is False
    assert record["oracle_result"]["reason"] == "candidate_graph_empty"
    assert record["stages"]["oracle_candidate_stage"]["returned"] == 0


def test_pair_record_reports_reach_and_denominator_split(oracle_module) -> None:
    left, right, left_camera, right_camera = _scene(oracle_module, right_mode="same")
    record = oracle_module.pair_record(
        oracle_module.attribute_pair(left, right, left_camera, right_camera))

    denominators = record["denominators"]
    reach = record["reach"]
    assert denominators["correspondence_truth_pairs"] == len(_POINTS)
    assert denominators["geometry_oracle_candidate_edges"] == len(_POINTS)
    assert denominators["geometry_baseline_candidate_edges"] == (
        record["baseline_result"]["graph_edges"])
    assert reach["oracle_candidate_correspondence_reach"] == len(_POINTS)
    assert reach["oracle_selected_correspondence_reach"] == len(_POINTS)
    assert record["stages"]["oracle_candidate_stage"]["precision"] == 1.0
    assert record["oracle_result"]["status"] == "solved"
    assert record["baseline_result"]["status"] == "solved"


def test_result_record_exposes_the_gate_diagnostics(oracle_module) -> None:
    left, right, left_camera, right_camera = _scene(oracle_module, right_mode="same")
    pair = oracle_module.attribute_pair(left, right, left_camera, right_camera)
    record = oracle_module.result_record(pair.oracle_result, pair.oracle_graph)

    assert set(record["gates"]) == {
        "median_selected_residual_degrees", "descriptor_cost", "parallax_degrees",
        "cheirality_fraction", "singular_ratio", "ambiguity_margin",
    }
    assert record["selection_state"] == oracle_module.SELECTION_SOLVED
    assert record["pose_state"] == oracle_module.POSE_SOLVED
    assert record["hypotheses"]
    assert all("passed" in item and "refusal" in item
               for item in record["hypotheses"])


# --- pooled accounting -----------------------------------------------------------


def test_summarise_attributions_counts_oracle_recovery(oracle_module) -> None:
    left, right, left_camera, right_camera = _scene(oracle_module, right_mode="same")
    solved = oracle_module.attribute_pair(left, right, left_camera, right_camera)
    occluded = _scene(oracle_module, right_mode="occluded")
    refused = oracle_module.attribute_pair(*occluded)

    summary = oracle_module.summarise_attributions([solved, refused])

    assert summary["pair_count"] == 2
    assert summary["truth_pairs"] == len(_POINTS)
    assert summary["pairs_with_truth"] == 1
    assert summary["pairs_with_empty_truth"] == 1
    assert summary["oracle_graph_empty_pairs"] == 1
    assert summary["baseline_solved"] == 2
    assert summary["oracle_solved"] == 1
    assert summary["oracle_refusal_reasons"] == {"candidate_graph_empty": 1}
    assert summary["positive_identity_mismatch"]["oracle_candidate_stage"] == 0
    assert summary["positive_identity_mismatch"]["oracle_selected_stage"] == 0
    assert summary["denominators"]["correspondence_truth_pairs"] == len(_POINTS)
    assert summary["reach"]["oracle_selected_correspondence_reach"] == len(_POINTS)


def _joint_result(*, status, reason, matches, support=0):
    return JointPairResult(
        left_id="a", right_id="b", status=status, reason=reason, pose=None,
        right_yaw_class=None, matches=matches,
        raw_keypoints=(len(_POINTS), len(_POINTS)), boundary_dropped=(0, 0),
        candidate_edges=0, total_graph_edges=0, truncated=0,
        ratio_survivors=(0, 0), mutual_survivors=0, support=support,
        support_floor=8, spatial_bins_left=0, spatial_bins_right=0,
        spatial_bins_total=16, median_selected_residual_degrees=None,
        descriptor_cost=None, parallax_degrees=None, cheirality_fraction=None,
        singular_ratio=None, ambiguity_margin=None, hypotheses=(),
    )


def test_summarise_attributions_marks_recovery_where_baseline_refused(
    oracle_module, monkeypatch,
) -> None:
    left, right, left_camera, right_camera = _scene(oracle_module, right_mode="same")
    original = joint_matching.reciprocal_topk_graph

    def _fake(*_args, **_kwargs):
        # The oracle call is the only one where the injected builder is installed.
        if joint_matching.reciprocal_topk_graph is not original:
            return _joint_result(
                status="solved", reason="",
                matches=np.column_stack(
                    (left.features.points, right.features.points)), support=len(_POINTS))
        return _joint_result(
            status="refused", reason="cheirality_failed",
            matches=np.empty((0, 4), dtype=np.float64))

    monkeypatch.setattr(oracle_module, "match_feature_frames", _fake)
    pair = oracle_module.attribute_pair(left, right, left_camera, right_camera)
    summary = oracle_module.summarise_attributions([pair])

    assert pair.baseline_solved is False
    assert pair.oracle_solved is True
    assert pair.baseline_selected.returned == 0
    assert pair.oracle_selected.positive == len(_POINTS)
    assert summary["oracle_solved_where_baseline_refused"] == 1
    assert summary["oracle_lost_where_baseline_solved"] == 0


# --- canonical entry selection ---------------------------------------------------


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


def _canal_entries() -> list[DatasetSceneInput]:
    return [
        _entry("fresh-c4-canal-town-2026-09-12",
               "canal-town-architectural-component-bbox", "group-canal", 4),
        _entry("fresh-c4-canal-town-framed-2026-09-12",
               "canal-town-architectural-component-bbox", "group-canal", 8),
        _entry("smoke-train-station-ds20c-2026-09-12",
               "train-station-spawn-208-96", "group-train", 4),
    ]


def test_canonical_entry_is_the_most_complete_canal_town_export(oracle_module) -> None:
    entries = _canal_entries()
    chosen = oracle_module.select_canonical_entry(entries)

    assert chosen.dataset_id == "fresh-c4-canal-town-framed-2026-09-12"
    record = oracle_module.canonical_entry_record(chosen, entries)
    assert record["frame_count"] == 8
    assert record["scene_group"] == "group-canal"
    assert len(record["considered_entries"]) == 2
    assert record["considered_entries"][0]["dataset_id"] == chosen.dataset_id


def test_canonical_entry_dataset_override_selects_exactly_one(oracle_module) -> None:
    entries = _canal_entries()
    chosen = oracle_module.select_canonical_entry(
        entries, dataset_id="fresh-c4-canal-town-2026-09-12")

    assert chosen.dataset_id == "fresh-c4-canal-town-2026-09-12"
    with pytest.raises(oracle_module.OracleCandidateGraphError):
        oracle_module.select_canonical_entry(entries, dataset_id="no-such-entry")


def test_canonical_entry_requires_a_canal_town_entry(oracle_module) -> None:
    entries = [_entry("train", "train-station-spawn-208-96", "group-train", 4)]
    with pytest.raises(oracle_module.OracleCandidateGraphError):
        oracle_module.select_canonical_entry(entries)


def test_canonical_entry_override_stays_within_the_canal_town_scope(
    oracle_module,
) -> None:
    """The override accepts both Canal Town IDs and rejects a known Train Station ID."""
    entries = _canal_entries()
    scoped = [
        "fresh-c4-canal-town-framed-2026-09-12",
        "fresh-c4-canal-town-2026-09-12",
    ]
    for dataset_id in scoped:
        chosen = oracle_module.select_canonical_entry(entries, dataset_id=dataset_id)
        assert chosen.dataset_id == dataset_id
        assert chosen.scene_group == "group-canal"
        record = oracle_module.canonical_entry_record(
            chosen, entries, dataset_id=dataset_id)
        assert record["selection_rule"] == "explicit --dataset-id override"
        considered = [item["dataset_id"] for item in record["considered_entries"]]
        # canonical_entry_record lists the same scoped candidates, never the train entry.
        assert considered == scoped
        assert "smoke-train-station-ds20c-2026-09-12" not in considered

    with pytest.raises(oracle_module.OracleCandidateGraphError):
        oracle_module.select_canonical_entry(
            entries, dataset_id="smoke-train-station-ds20c-2026-09-12")


def test_canonical_entry_override_rejects_a_lone_non_canal_entry(oracle_module) -> None:
    """A single discovered non-Canal entry still cannot satisfy the scoped override."""
    entries = _canal_entries()
    assert oracle_module.select_canonical_entry(
        entries, dataset_id="fresh-c4-canal-town-2026-09-12").dataset_id == (
            "fresh-c4-canal-town-2026-09-12")
    train_only = [_entry("smoke-train-station-ds20c-2026-09-12",
                         "train-station-spawn-208-96", "group-train", 4)]
    with pytest.raises(oracle_module.OracleCandidateGraphError):
        oracle_module.select_canonical_entry(
            train_only, dataset_id="smoke-train-station-ds20c-2026-09-12")


# --- sealed payload assembly ------------------------------------------------------


def test_build_payload_seals_and_labels_the_oracle(oracle_module, monkeypatch) -> None:
    left, right, _lc, _rc = _scene(oracle_module, right_mode="same")
    entry = _entry("fresh-c4-canal-town-framed-2026-09-12",
                   "canal-town-architectural-component-bbox", "group-canal", 8)
    left_camera = _camera(LEFT_POSITION, LEFT_YAW, LEFT_PITCH)
    right_camera = _camera(RIGHT_POSITION, RIGHT_YAW, RIGHT_PITCH)
    scene = FinalizedScene(
        input=entry,
        frames=(_frame_input("a", left_camera), _frame_input("b", right_camera)),
        occupancy={}, region_min=(0, 0, 0), region_max=(1, 1, 1),
    )

    monkeypatch.setattr(oracle_module, "build_scene", lambda _entry: scene)
    monkeypatch.setattr(
        oracle_module, "observe_oracle_frame",
        lambda _scene, frame, detection=None: left if frame.frame_id == "a" else right)
    manifest = build_group_manifest([entry])
    payload = oracle_module.build_payload(
        "exports/datasets", [entry], manifest, entry)

    assert payload["schema"] == oracle_module.SCHEMA
    assert payload["camera_policy"]["assisted"] is True
    assert payload["camera_policy"]["default_image_only_reconstruction"] is False
    assert payload["oracle_policy"]["truth_assisted_upper_bound"] is True
    assert payload["oracle_policy"]["production_result"] is False
    assert payload["scene"]["pair_count"] == 1
    assert payload["summary"]["baseline_solved"] == 1
    assert payload["summary"]["oracle_solved"] == 1
    assert payload["inputs"]["canonical_entry"]["dataset_id"] == entry.dataset_id
    assert payload["consistency"]["ok"] is True
    json.dumps(payload, sort_keys=True, allow_nan=False)
    assert payload["seal_sha256"] == oracle_module.seal_record(payload)


def test_build_payload_reports_a_json_safe_oracle_graph_edge_count(
    oracle_module, monkeypatch,
) -> None:
    left, right, _lc, _rc = _scene(oracle_module, right_mode="same")
    entry = _entry("fresh-c4-canal-town-framed-2026-09-12",
                   "canal-town-architectural-component-bbox", "group-canal", 8)
    left_camera = _camera(LEFT_POSITION, LEFT_YAW, LEFT_PITCH)
    right_camera = _camera(RIGHT_POSITION, RIGHT_YAW, RIGHT_PITCH)
    scene = FinalizedScene(
        input=entry,
        frames=(_frame_input("a", left_camera), _frame_input("b", right_camera)),
        occupancy={}, region_min=(0, 0, 0), region_max=(1, 1, 1),
    )
    monkeypatch.setattr(oracle_module, "build_scene", lambda _entry: scene)
    monkeypatch.setattr(
        oracle_module, "observe_oracle_frame",
        lambda _scene, frame, detection=None: left if frame.frame_id == "a" else right)
    manifest = build_group_manifest([entry])
    payload = oracle_module.build_payload(
        "exports/datasets", [entry], manifest, entry)

    pair = payload["scene"]["pairs"][0]
    assert pair["oracle_result"]["graph_edges"] == len(_POINTS)
    assert pair["baseline_result"]["graph_edges"] >= 0
    assert isinstance(pair["baseline_solved"], bool)
    assert isinstance(pair["oracle_graph_empty"], bool)


# --- runner guards -----------------------------------------------------------------


def test_runner_refuses_to_overwrite_an_existing_output(
    oracle_module, tmp_path: Path,
) -> None:
    output = tmp_path / "out.json"
    output.write_text("keep", encoding="utf-8")
    assert oracle_module.main(
        ["--output", str(output), "--exports-root", str(tmp_path)]) == 2
    assert output.read_text(encoding="utf-8") == "keep"


def test_runner_reports_failure_without_writing_when_no_exports(
    oracle_module, tmp_path: Path,
) -> None:
    output = tmp_path / "absent.json"
    missing = tmp_path / "missing-root"
    assert oracle_module.main(
        ["--output", str(output), "--exports-root", str(missing)]) == 1
    assert not output.exists()


def test_detection_settings_mirror_the_production_joint_path(oracle_module) -> None:
    matcher = JointMatchSettings()
    detection = oracle_module.detection_settings(matcher)

    assert detection.max_features == matcher.max_features == 4000
    assert detection.mask_boundary_pad_px == matcher.mask_boundary_pad_px == 8.0
    assert detection.ratio_test == 0.75


def test_module_surface_distances_are_finite(oracle_module) -> None:
    left, right, _lc, _rc = _scene(oracle_module, right_mode="same")
    graph = oracle_module.build_oracle_graph(
        left.features, right.features, left.hits, right.hits)
    assert all(math.isfinite(float(value)) for value in graph.distances)
