"""Focused tests for the evaluation-only pair-correspondence scorecard.

DS-R5I. These tests exercise the pure truth/scoring arithmetic, the synthetic
visibility raycast, and the grouped split manifest. They do not run SIFT on the
corpus; the corpus evaluator is a separate authorized one-shot run.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from mc_photogrammetry.domain.models import BlockState
from mc_photogrammetry.evaluation_pair_correspondence import (
    DatasetSceneInput,
    PairCorrespondenceEvaluationError,
    ScoreTotals,
    SurfaceHit,
    build_group_manifest,
    check_fold_leakage,
    check_split_leakage,
    classify_returned_pair,
    construct_truth_pairs,
    discover_finalized_entries,
    is_full_cube_truth_state,
    macro_average,
    match_indices,
    metric_value,
    raycast_surface,
    score_pair_matches,
    seal_record,
    summarise_scene,
    totals_from_pairs,
)
from mc_photogrammetry.reconstruction.camera import CameraModel

TOLERANCE = 0.0625


def hit(
    coordinate: tuple[int, int, int],
    slot: str,
    point: tuple[float, float, float],
    name: str = "minecraft:stone",
    *,
    properties: tuple[tuple[str, str], ...] = (),
    pixel: tuple[float, float] | None = None,
) -> SurfaceHit:
    """One synthetic hit.

    ``pixel`` defaults to a stand-in derived from the world point so ordinary
    tests get distinct pixel identities; duplicate-pixel tests pass it explicitly.
    """
    return SurfaceHit(
        frame_id="frame",
        coordinate=coordinate,
        slot=slot,
        world_point=point,
        camera_depth=1.0,
        state=BlockState(name, properties),
        pixel=(float(point[0]), float(point[1])) if pixel is None else pixel,
    )


# --- truth model -----------------------------------------------------------------


def test_full_cube_classifier_handles_double_slabs_and_full_cube_terrain() -> None:
    assert is_full_cube_truth_state(BlockState("minecraft:stone")) is True
    assert is_full_cube_truth_state(BlockState("minecraft:grass_block")) is True
    assert is_full_cube_truth_state(
        BlockState("minecraft:smooth_quartz_slab", (("type", "double"),))) is True
    assert is_full_cube_truth_state(
        BlockState("minecraft:smooth_quartz_slab", (("type", "bottom"),))) is False
    assert is_full_cube_truth_state(
        BlockState("minecraft:iron_trapdoor", (("facing", "north"),))) is False
    assert is_full_cube_truth_state(BlockState("minecraft:short_grass")) is False


# --- exact same-surface match ----------------------------------------------------


def test_exact_same_surface_match_is_positive_precision_and_recall_one() -> None:
    left = [hit((0, 0, 0), "south", (0.50, 0.50, 1.0))]
    right = [hit((0, 0, 0), "south", (0.50, 0.50, 1.0))]
    indices = match_indices(
        [[0.5, 0.5, 0.5, 0.5]], [[0.5, 0.5]], [[0.5, 0.5]])
    geometry = score_pair_matches(
        indices, left, right, [[0.5, 0.5]], [[0.5, 0.5]], tolerance=TOLERANCE)
    assert geometry.returned == 1
    assert geometry.positive == 1
    assert geometry.truth_pairs == 1
    assert geometry.truth_pairs_returned == 1
    assert geometry.precision == 1.0
    assert geometry.recall == 1.0


# --- appearance proxy and wrong-surface buckets ----------------------------------


def test_same_state_same_slot_other_block_is_an_appearance_proxy_not_a_positive() -> None:
    left = [hit((0, 0, 0), "north", (0.5, 0.5, 0.0), "minecraft:white_concrete")]
    right = [hit((5, 0, 0), "north", (5.5, 0.5, 0.0), "minecraft:white_concrete")]
    geometry = score_pair_matches(
        match_indices([[0.5, 0.5, 5.5, 0.5]], [[0.5, 0.5]], [[5.5, 0.5]]),
        left, right, [[0.5, 0.5]], [[5.5, 0.5]], tolerance=TOLERANCE)
    assert classify_returned_pair(left[0], right[0], tolerance=TOLERANCE) == (
        "same_state_same_slot_other_block")
    assert geometry.positive == 0
    assert geometry.same_state_same_slot_other_block == 1
    assert geometry.truth_pairs == 0
    assert geometry.precision == 0.0
    assert geometry.recall is None


def test_same_block_different_slot_is_wrong_surface_not_an_appearance_proxy() -> None:
    left = [hit((0, 0, 0), "north", (0.5, 0.5, 0.0), "minecraft:white_concrete")]
    right = [hit((0, 0, 0), "south", (0.5, 0.5, 1.0), "minecraft:white_concrete")]
    verdict = classify_returned_pair(left[0], right[0], tolerance=TOLERANCE)
    assert verdict == "wrong_surface"


def test_same_state_name_with_different_properties_is_wrong_surface() -> None:
    left = [hit((0, 0, 0), "north", (0.5, 0.5, 0.0), "minecraft:white_concrete")]
    right = [hit((5, 0, 0), "north", (5.5, 0.5, 0.0), "minecraft:white_concrete",
                 properties=(("waterlogged", "true"),))]
    assert classify_returned_pair(left[0], right[0], tolerance=TOLERANCE) == "wrong_surface"


def test_same_state_different_coordinate_and_slot_is_wrong_surface() -> None:
    left = [hit((0, 0, 0), "north", (0.5, 0.5, 0.0), "minecraft:white_concrete")]
    right = [hit((5, 0, 0), "south", (5.5, 0.5, 1.0), "minecraft:white_concrete")]
    assert classify_returned_pair(left[0], right[0], tolerance=TOLERANCE) == "wrong_surface"


def test_a_different_block_name_on_another_face_is_wrong_surface() -> None:
    left = [hit((0, 0, 0), "north", (0.5, 0.5, 0.0), "minecraft:white_concrete")]
    right = [hit((5, 0, 0), "north", (5.5, 0.5, 0.0), "minecraft:stone")]
    verdict = classify_returned_pair(left[0], right[0], tolerance=TOLERANCE)
    assert verdict == "wrong_surface"


# --- occlusion in one frame ------------------------------------------------------


def test_occluded_surface_in_the_other_frame_is_not_a_positive() -> None:
    left = [hit((0, 0, 0), "south", (0.5, 0.5, 1.0), "minecraft:stone")]
    # The same world point is visible from the left frame but the right frame's
    # keypoint ray lands on the nearer occluder, so it never reaches the surface.
    right = [hit((0, 0, 1), "north", (0.5, 0.5, 1.0), "minecraft:dirt")]
    geometry = score_pair_matches(
        match_indices([[0.5, 0.5, 0.5, 0.5]], [[0.5, 0.5]], [[0.5, 0.5]]),
        left, right, [[0.5, 0.5]], [[0.5, 0.5]], tolerance=TOLERANCE)
    assert geometry.positive == 0
    assert geometry.wrong_surface == 1
    assert geometry.truth_pairs == 0
    assert construct_truth_pairs(left, right, tolerance=TOLERANCE) == frozenset()


def test_no_truth_surface_in_one_frame_is_unattributed() -> None:
    left = [hit((0, 0, 0), "south", (0.5, 0.5, 1.0))]
    right = [None]
    geometry = score_pair_matches(
        match_indices([[0.5, 0.5, 5.5, 5.5]], [[0.5, 0.5]], [[5.5, 5.5]]),
        left, right, [[0.5, 0.5]], [[5.5, 5.5]], tolerance=TOLERANCE)
    assert geometry.unattributed == 1
    assert geometry.positive == 0
    assert geometry.truth_pairs_returned == 0


# --- reprojection / tolerance boundary ------------------------------------------


def test_same_face_tolerance_boundary_is_inclusive_below_and_exclusive_above() -> None:
    below = hit((0, 0, 0), "south", (0.5, 0.5, 1.0))
    at = hit((0, 0, 0), "south", (0.5, 0.5, 1.0 + TOLERANCE))
    above = hit((0, 0, 0), "south", (0.5, 0.5, 1.0 + TOLERANCE * 2.0))
    assert classify_returned_pair(below, below, tolerance=TOLERANCE) == "positive"
    assert classify_returned_pair(below, at, tolerance=TOLERANCE) == "positive"
    assert classify_returned_pair(below, above, tolerance=TOLERANCE) == "same_face_out_of_tolerance"


def test_truth_pairs_find_only_within_tolerance_pairs() -> None:
    left = [hit((0, 0, 0), "south", (0.5, 0.5, 1.0), pixel=(1.0, 1.0))]
    right = [
        hit((0, 0, 0), "south", (0.5, 0.5, 1.0 + TOLERANCE * 0.5), pixel=(2.0, 1.0)),
        hit((0, 0, 0), "south", (0.5, 0.5, 1.0 + TOLERANCE * 4.0), pixel=(2.0, 2.0)),
        hit((9, 9, 9), "south", (9.5, 9.5, 10.0), pixel=(2.0, 3.0)),
        None,
    ]
    assert construct_truth_pairs(left, right, tolerance=TOLERANCE) == frozenset(
        {((1.0, 1.0), (2.0, 1.0))})


# --- unique pixel-pair identity ---------------------------------------------------


def test_duplicate_keypoint_coordinates_do_not_inflate_the_truth_denominator() -> None:
    left = [
        hit((0, 0, 0), "south", (0.5, 0.5, 1.0), pixel=(10.0, 20.0)),
        hit((0, 0, 0), "south", (0.5, 0.5, 1.0), pixel=(10.0, 20.0)),
    ]
    right = [
        hit((0, 0, 0), "south", (0.5, 0.5, 1.0), pixel=(30.0, 40.0)),
        hit((0, 0, 0), "south", (0.5, 0.5, 1.0), pixel=(30.0, 40.0)),
    ]
    assert construct_truth_pairs(left, right, tolerance=TOLERANCE) == frozenset(
        {((10.0, 20.0), (30.0, 40.0))})


def test_distinct_keypoint_pixels_are_never_collapsed() -> None:
    left = [
        hit((0, 0, 0), "south", (0.5, 0.5, 1.0), pixel=(10.0, 20.0)),
        hit((0, 0, 0), "south", (0.5, 0.5, 1.0), pixel=(11.0, 20.0)),
    ]
    right = [
        hit((0, 0, 0), "south", (0.5, 0.5, 1.0), pixel=(30.0, 40.0)),
        hit((0, 0, 0), "south", (0.5, 0.5, 1.0), pixel=(31.0, 40.0)),
    ]
    truth = construct_truth_pairs(left, right, tolerance=TOLERANCE)
    assert len(truth) == 4


def test_duplicate_returned_identities_are_counted_once_in_precision() -> None:
    left = [hit((0, 0, 0), "south", (0.5, 0.5, 1.0), pixel=(10.0, 20.0))]
    right = [hit((0, 0, 0), "south", (0.5, 0.5, 1.0), pixel=(30.0, 40.0))]
    indices = match_indices(
        [[10.0, 20.0, 30.0, 40.0], [10.0, 20.0, 30.0, 40.0]], [[10.0, 20.0]], [[30.0, 40.0]])
    geometry = score_pair_matches(
        indices, left, right, [[10.0, 20.0]], [[30.0, 40.0]], tolerance=TOLERANCE)
    assert geometry.returned == 1
    assert geometry.positive == 1
    assert geometry.truth_pairs == 1
    assert geometry.truth_pairs_returned == 1
    assert geometry.precision == 1.0


def test_duplicate_indices_at_one_pixel_pair_are_one_attributed_identity() -> None:
    """Two distinct keypoint indices at exactly one pixel pair are one identity."""
    left = [
        hit((0, 0, 0), "south", (0.5, 0.5, 1.0), pixel=(10.0, 20.0)),
        hit((0, 0, 0), "south", (0.5, 0.5, 1.0), pixel=(10.0, 20.0)),
    ]
    right = [
        hit((0, 0, 0), "south", (0.5, 0.5, 1.0), pixel=(30.0, 40.0)),
        hit((0, 0, 0), "south", (0.5, 0.5, 1.0), pixel=(30.0, 40.0)),
    ]
    geometry = score_pair_matches(
        np.asarray([[0, 0], [1, 1]], dtype=np.int64), left, right,
        [[10.0, 20.0], [10.0, 20.0]], [[30.0, 40.0], [30.0, 40.0]],
        tolerance=TOLERANCE)
    assert geometry.returned == 1
    assert geometry.positive == 1
    assert geometry.truth_pairs == 1
    assert geometry.truth_pairs_returned == 1
    assert geometry.precision == 1.0


def test_duplicate_indices_with_unattributed_hits_are_one_identity() -> None:
    """Unattributed rows need the point arrays: their hit carries no pixel."""
    geometry = score_pair_matches(
        np.asarray([[0, 0], [1, 1]], dtype=np.int64), [None, None], [None, None],
        [[10.0, 20.0], [10.0, 20.0]], [[30.0, 40.0], [30.0, 40.0]],
        tolerance=TOLERANCE)
    assert geometry.returned == 1
    assert geometry.unattributed == 1
    assert geometry.positive == 0
    assert geometry.precision == 0.0
    assert geometry.truth_pairs == 0
    assert geometry.recall is None


def test_duplicate_indices_never_collapse_distinct_pixels() -> None:
    left = [
        hit((0, 0, 0), "south", (0.5, 0.5, 1.0), pixel=(10.0, 20.0)),
        hit((0, 0, 0), "south", (0.5, 0.5, 1.0), pixel=(10.0, 20.0)),
        hit((0, 0, 0), "south", (0.5, 0.5, 1.0), pixel=(11.0, 20.0)),
    ]
    right = [hit((0, 0, 0), "south", (0.5, 0.5, 1.0), pixel=(30.0, 40.0))]
    geometry = score_pair_matches(
        np.asarray([[0, 0], [1, 0], [2, 0]], dtype=np.int64), left, right,
        [[10.0, 20.0], [10.0, 20.0], [11.0, 20.0]], [[30.0, 40.0]],
        tolerance=TOLERANCE)
    assert geometry.returned == 2
    assert geometry.positive == 2
    assert geometry.truth_pairs == 2
    assert geometry.truth_pairs_returned == 2


def test_category_counts_sum_to_unique_returned_pairs() -> None:
    left = [
        hit((0, 0, 0), "south", (0.5, 0.5, 1.0), pixel=(10.0, 20.0)),
        hit((0, 0, 0), "south", (0.5, 0.5, 1.0), pixel=(10.0, 20.0)),
        hit((5, 0, 0), "north", (5.5, 0.5, 0.0), "minecraft:white_concrete",
            pixel=(11.0, 20.0)),
        hit((0, 0, 0), "south", (0.5, 0.5, 1.0), pixel=(12.0, 20.0)),
    ]
    right = [
        hit((0, 0, 0), "south", (0.5, 0.5, 1.0), pixel=(30.0, 40.0)),
        hit((7, 0, 0), "north", (7.5, 0.5, 0.0), "minecraft:white_concrete",
            pixel=(31.0, 40.0)),
        None,
    ]
    # Rows: (0,0) positive, (1,0) its duplicate pixel pair, (2,1)
    # same-state/same-slot proxy, (3,2) unattributed with no truth surface.
    geometry = score_pair_matches(
        np.asarray([[0, 0], [1, 0], [2, 1], [3, 2]], dtype=np.int64), left, right,
        [[10.0, 20.0], [10.0, 20.0], [11.0, 20.0], [12.0, 20.0]],
        [[30.0, 40.0], [31.0, 40.0], [32.0, 40.0]], tolerance=TOLERANCE)
    categories = (
        geometry.positive + geometry.same_face_out_of_tolerance
        + geometry.same_state_same_slot_other_block + geometry.wrong_surface
        + geometry.unattributed)
    assert categories == geometry.returned == 3
    assert geometry.positive == 1
    assert geometry.same_state_same_slot_other_block == 1
    assert geometry.unattributed == 1
    assert geometry.truth_pairs == 2
    assert geometry.truth_pairs_returned == 1
    assert geometry.precision == geometry.positive / geometry.returned
    assert geometry.recall == geometry.truth_pairs_returned / geometry.truth_pairs


def test_conflicting_verdicts_for_one_pixel_pair_raise() -> None:
    left = [
        hit((0, 0, 0), "south", (0.5, 0.5, 1.0), pixel=(10.0, 20.0)),
        hit((9, 9, 9), "north", (9.5, 9.5, 0.0), pixel=(10.0, 20.0)),
    ]
    right = [hit((0, 0, 0), "south", (0.5, 0.5, 1.0), pixel=(30.0, 40.0))]
    with pytest.raises(PairCorrespondenceEvaluationError):
        score_pair_matches(
            np.asarray([[0, 0], [1, 0]], dtype=np.int64), left, right,
            [[10.0, 20.0], [10.0, 20.0]], [[30.0, 40.0]], tolerance=TOLERANCE)


def test_match_index_outside_the_pixel_table_raises() -> None:
    with pytest.raises(PairCorrespondenceEvaluationError):
        score_pair_matches(
            np.asarray([[1, 0]], dtype=np.int64), [None, None], [None],
            [[10.0, 20.0]], [[30.0, 40.0]], tolerance=TOLERANCE)


# --- empty denominators ----------------------------------------------------------


def test_empty_denominators_are_undefined_not_zero() -> None:
    geometry = score_pair_matches(
        match_indices([], [[0.0, 0.0]], [[0.0, 0.0]]), [None], [None],
        [[0.0, 0.0]], [[0.0, 0.0]], tolerance=TOLERANCE)
    assert geometry.returned == 0
    assert geometry.truth_pairs == 0
    assert geometry.precision is None
    assert geometry.recall is None
    assert metric_value(0, 0, undefined_reason="no_returned_matches") == {
        "value": None,
        "numerator": 0,
        "denominator": 0,
        "undefined_reason": "no_returned_matches",
    }


def test_summarise_scene_with_no_pairs_is_undefined() -> None:
    summary = summarise_scene([])
    assert summary["pair_count"] == 0
    assert summary["precision"]["value"] is None
    assert summary["recall"]["value"] is None
    assert summary["precision"]["undefined_reason"] == "no_returned_matches"


# --- metric arithmetic -----------------------------------------------------------


def _pair(returned: int, positive: int, truth: int, truth_returned: int):
    from mc_photogrammetry.evaluation_pair_correspondence import PairGeometry, PairScore

    return PairScore(
        left_id="a", right_id="b", status="ok", reason="",
        left_keypoints=10, right_keypoints=10, ratio_left=5, ratio_right=5, mutual=4,
        match_count=returned, spatial_coverage=0.5, mutual_fraction=0.4,
        geometry=PairGeometry(
            returned=returned, positive=positive, same_face_out_of_tolerance=0,
            same_state_same_slot_other_block=0, wrong_surface=0, unattributed=0,
            truth_pairs=truth, truth_pairs_returned=truth_returned),
    )


def test_pooled_metrics_are_ratio_of_sums_not_mean_of_ratios() -> None:
    pairs = [_pair(2, 1, 2, 1), _pair(8, 4, 8, 4)]
    totals = totals_from_pairs(pairs)
    assert isinstance(totals, ScoreTotals)
    assert totals.returned == 10
    assert totals.positive == 5
    assert totals.precision == 0.5
    assert totals.truth_pairs == 10
    assert totals.truth_pairs_returned == 5
    assert totals.recall == 0.5
    summary = summarise_scene(pairs)
    assert summary["totals"]["returned"] == 10
    assert summary["precision"]["value"] == 0.5


def test_macro_average_skips_undefined_groups_and_reports_the_count() -> None:
    macro = macro_average([0.5, None, 1.0])
    assert macro["value"] == pytest.approx(0.75)
    assert macro["defined_groups"] == 2
    assert macro["total_groups"] == 3
    assert macro_average([None, None])["value"] is None
    assert macro_average([None, None])["undefined_reason"] == (
        "no_group_with_a_defined_metric")


# --- duplicate exports and folds -------------------------------------------------


def _entry(dataset_id: str, scene_id: str, group: str, split: str = "train",
           directory: str | None = None) -> DatasetSceneInput:
    directory = directory or f"scenes/{dataset_id}"
    return DatasetSceneInput(
        dataset_id=dataset_id,
        index_path=f"exports/datasets/{dataset_id}/index.json",
        index_sha256=hashlib.sha256(dataset_id.encode()).hexdigest(),
        scene_id=scene_id,
        scene_group=group,
        split=split,
        directory=directory,
        scene_root=f"exports/datasets/{dataset_id}/{directory}",
        hashes=(),
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
    assert manifest["entry_count"] == 3
    canonical = next(group for group in manifest["groups"]
                     if group["scene_group"] == "group-canal")
    assert canonical["entry_count"] == 2
    assert check_fold_leakage(manifest) == ()
    for fold in manifest["folds"]:
        held = set(fold["held_out_entries"])
        retained = set(fold["retained_entries"])
        assert held.isdisjoint(retained)
    canal_fold = next(fold for fold in manifest["folds"]
                      if fold["held_out_scene_group"] == "group-canal")
    assert canal_fold["held_out_entries"] == [
        "canal-town-framed:scenes/canal-town-framed",
        "canal-town:scenes/canal-town",
    ]


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


def test_split_leakage_flags_scene_or_group_inconsistency() -> None:
    same_scene_two_groups = [
        _entry("d1", "scene", "g1"),
        _entry("d2", "scene", "g2"),
    ]
    assert check_split_leakage(same_scene_two_groups)
    group_two_splits = [
        _entry("d1", "scene-a", "g1", split="train"),
        _entry("d2", "scene-b", "g1", split="test"),
    ]
    assert check_split_leakage(group_two_splits)
    with pytest.raises(PairCorrespondenceEvaluationError):
        build_group_manifest(group_two_splits)


# --- synthetic visibility raycast ------------------------------------------------


def _camera() -> CameraModel:
    return CameraModel(
        position=(0.5, 0.5, 3.5), yaw_degrees=180.0, pitch_degrees=0.0,
        fov_degrees=90.0, eye_height=0.0)


def test_raycast_returns_the_exact_entry_face_and_point() -> None:
    occupancy = {(0, 0, 0): BlockState("minecraft:stone")}
    outcome = raycast_surface("frame", _camera(), 100, 100, occupancy, 50.0, 50.0)
    assert outcome.reason == "hit"
    assert outcome.hit is not None
    assert outcome.hit.coordinate == (0, 0, 0)
    assert outcome.hit.slot == "south"
    assert outcome.hit.camera_depth == pytest.approx(2.5)
    assert outcome.hit.world_point == pytest.approx((0.5, 0.5, 1.0))


def test_raycast_excludes_partial_model_owners_and_keeps_double_slabs() -> None:
    partial = {(0, 0, 0): BlockState(
        "minecraft:stone_slab", (("type", "bottom"),))}
    outcome = raycast_surface("frame", _camera(), 100, 100, partial, 50.0, 50.0)
    assert outcome.hit is None
    assert outcome.reason == "excluded_partial_owner"

    double = {(0, 0, 0): BlockState(
        "minecraft:stone_slab", (("type", "double"),))}
    outcome = raycast_surface("frame", _camera(), 100, 100, double, 50.0, 50.0)
    assert outcome.reason == "hit"


def test_match_indices_round_trips_and_rejects_unknown_pixels() -> None:
    indices = match_indices([[1.0, 2.0, 3.0, 4.0]], [[1.0, 2.0]], [[3.0, 4.0]])
    assert indices.tolist() == [[0, 0]]
    with pytest.raises(PairCorrespondenceEvaluationError):
        match_indices([[9.0, 9.0, 3.0, 4.0]], [[1.0, 2.0]], [[3.0, 4.0]])


# --- discovery and hash pinning --------------------------------------------------


def _write_scene(root: Path) -> tuple[Path, str, str]:
    scene = root / "scenes" / "scene-a"
    (scene / "captures").mkdir(parents=True)
    ground_truth = {
        "schema_version": 1, "mod_contract_version": 1, "scene_id": "scene-a",
        "world_identity": "world", "region": {"min": [0, 0, 0], "max": [1, 1, 1]},
        "blocks": [{"pos": [0, 0, 0], "name": "minecraft:stone", "properties": {}}],
    }
    capture = {
        "schema_version": 1, "scene_id": "scene-a", "region": {"min": [0, 0, 0],
                                                              "max": [1, 1, 1]},
        "frames": [{"frame_id": "image-0000", "filename": "image-0000.png",
                    "captured": True, "actual": {"framebuffer_width": 4,
                                                 "framebuffer_height": 4}}],
        "cameras": [CameraModel(position=(0.0, 0.0, 0.0), yaw_degrees=0.0,
                                pitch_degrees=0.0).to_dict()],
        "resolutions": [[4, 4]],
        "screenshots": [{"frame_id": "image-0000", "filename": "image-0000.png"}],
    }
    (scene / "ground_truth.json").write_text(json.dumps(ground_truth), encoding="utf-8")
    (scene / "capture.json").write_text(json.dumps(capture), encoding="utf-8")
    png = scene / "captures" / "image-0000.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"synthetic")
    capture_sha = hashlib.sha256((scene / "capture.json").read_bytes()).hexdigest()
    gt_sha = hashlib.sha256((scene / "ground_truth.json").read_bytes()).hexdigest()
    png_sha = hashlib.sha256(png.read_bytes()).hexdigest()
    index = {
        "schema_version": 1, "advanced_only": True,
        "scenes": [{
            "scene_id": "scene-a", "split": "train", "scene_group": "group-a",
            "directory": "scenes/scene-a",
            "hashes": {"capture.json": capture_sha, "ground_truth.json": gt_sha},
            "screenshots": [{"filename": "image-0000.png", "sha256": png_sha}],
        }],
    }
    index_path = root / "index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    return index_path, capture_sha, png_sha


def test_discovery_hash_pins_every_entry(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    index_path, _capture_sha, png_sha = _write_scene(root)
    entries = discover_finalized_entries(tmp_path)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.scene_id == "scene-a"
    assert entry.scene_group == "group-a"
    assert entry.index_sha256 == hashlib.sha256(index_path.read_bytes()).hexdigest()
    assert entry.screenshots == (("image-0000.png", png_sha),)


def test_discovery_rejects_a_changed_screenshot(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    _write_scene(root)
    (root / "scenes" / "scene-a" / "captures" / "image-0000.png").write_bytes(b"changed")
    with pytest.raises(PairCorrespondenceEvaluationError):
        discover_finalized_entries(tmp_path)


def test_seal_is_stable_and_excludes_its_own_field() -> None:
    payload = {"a": 1, "b": [1, 2, 3], "seal_sha256": "ignored"}
    first = seal_record(payload)
    assert first == seal_record({"a": 1, "b": [1, 2, 3]})
    assert first != seal_record({"a": 2, "b": [1, 2, 3]})
    assert len(first) == 64
    assert math.isfinite(float(int(first, 16)))


# --- runner guard ------------------------------------------------------------------


def test_runner_refuses_to_overwrite_an_existing_output(tmp_path: Path) -> None:
    import importlib.util

    script = (Path(__file__).resolve().parents[1]
              / "scripts" / "evaluation" / "evaluate_pair_correspondence.py")
    spec = importlib.util.spec_from_file_location("ds_r5i_runner", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "out.json"
    output.write_text("keep", encoding="utf-8")
    assert module.main(["--output", str(output), "--exports-root", str(tmp_path)]) == 2
    assert output.read_text(encoding="utf-8") == "keep"


# --- end-to-end pipeline on a synthetic capture ------------------------------------


def _write_capture_scene(root: Path, width: int, height: int) -> None:
    from mc_photogrammetry.reconstruction.camera import CameraModel
    from PIL import Image
    import numpy as np

    scene = root / "scenes" / "synthetic"
    (scene / "captures").mkdir(parents=True)
    blocks = []
    for x in range(10):
        for y in range(10):
            blocks.append({"pos": [x, y, 0], "name": "minecraft:stone", "properties": {}})
    ground_truth = {
        "schema_version": 1, "mod_contract_version": 1, "scene_id": "synthetic",
        "world_identity": "world", "region": {"min": [0, 0, 0], "max": [9, 9, 0]},
        "blocks": blocks,
    }
    frames = []
    cameras = []
    resolutions = []
    for index, eye_x in enumerate((5.0, 3.0)):
        frame_id = f"image-{index:04d}"
        filename = f"{frame_id}.png"
        frames.append({
            "frame_id": frame_id, "filename": filename, "captured": True,
            "actual": {"eye_position": [eye_x, 5.0, 10.0], "yaw_degrees": 180.0,
                       "pitch_degrees": 0.0, "fov_degrees": 70.0,
                       "framebuffer_width": width, "framebuffer_height": height},
        })
        cameras.append(CameraModel(position=(eye_x, 5.0, 10.0), yaw_degrees=180.0,
                                   pitch_degrees=0.0, fov_degrees=70.0).to_dict())
        resolutions.append([width, height])
        rng = np.random.default_rng(index)
        texture = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        Image.fromarray(texture, "RGB").save(scene / "captures" / filename)
    capture = {
        "schema_version": 1, "scene_id": "synthetic", "region": {"min": [0, 0, 0],
                                                                "max": [9, 9, 0]},
        "frames": frames, "cameras": cameras, "resolutions": resolutions,
        "screenshots": [{"frame_id": frame["frame_id"], "filename": frame["filename"]}
                        for frame in frames],
    }
    (scene / "ground_truth.json").write_text(json.dumps(ground_truth), encoding="utf-8")
    (scene / "capture.json").write_text(json.dumps(capture), encoding="utf-8")


def test_evaluate_scene_runs_end_to_end_on_synthetic_capture(tmp_path: Path) -> None:
    from mc_photogrammetry.evaluation_pair_correspondence import build_scene, evaluate_scene

    _write_capture_scene(tmp_path, width=320, height=240)
    entry = DatasetSceneInput(
        dataset_id="synthetic", index_path=str(tmp_path / "index.json"),
        index_sha256="0" * 64, scene_id="synthetic", scene_group="group-synthetic",
        split="train", directory="scenes/synthetic",
        scene_root=str(tmp_path / "scenes" / "synthetic"), hashes=(), screenshots=(),
    )
    scene = build_scene(entry)
    evaluation = evaluate_scene(scene)
    assert len(evaluation.observations) == 2
    assert len(evaluation.pairs) == 1
    pair = evaluation.pairs[0]
    assert pair.left_id == "image-0000"
    assert pair.right_id == "image-0001"
    # The production matcher returned identities that all map back to keypoints,
    # and the truth denominator is never negative.
    assert pair.geometry.returned == pair.match_count
    assert pair.geometry.truth_pairs >= pair.geometry.truth_pairs_returned
    assert pair.geometry.positive >= pair.geometry.truth_pairs_returned
    summary = summarise_scene(list(evaluation.pairs))
    assert summary["pair_count"] == 1
