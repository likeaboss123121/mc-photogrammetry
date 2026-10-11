"""Focused tests for the DS-R5N oracle support-floor sensitivity diagnostic.

These tests exercise the evaluation-only pieces on synthetic geometry and
injected payload scenes. They do **not** read the corpus, hash a screenshot,
touch the sealed DS-R5K artifact, or run the corpus evaluator; the single
authorized corpus run is a separate lead-authorized action. Covered here:
default-path/floor-setting isolation, the oracle-only control settings, per-pair
and per-yaw serialization, no output on pin/leakage failure, overwrite refusal,
the sealed-artifact guard and deterministic sealing.
"""
from __future__ import annotations

import copy
import importlib.util
import json
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
    PairCorrespondenceEvaluationError,
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

N_TRUTH = 20
N_NOISE = 1000
"""Noise keypoints pad ``min(raw_keypoints)`` past 1,000 so the default floor is 21.

The oracle truth graph for this synthetic pair has exactly 20 edges, so the
default fractional floor refuses it while the control absolute floor of 8 does
not; that gap is the behaviour DS-R5N measures.
"""

_POINTS = (
    (-9.0, 3.0, 0.5), (-8.0, 2.0, 1.5), (-10.0, 4.0, -0.5), (-9.5, 1.5, 2.0),
    (-8.5, 5.0, 0.0), (-11.0, 3.5, 1.0), (-7.5, 2.5, -1.0), (-10.5, 2.0, 2.5),
    (-9.0, 5.5, -1.5), (-12.0, 4.5, 0.5), (-8.0, 3.8, 2.8), (-10.0, 1.0, -0.8),
    (-13.0, 3.0, 1.5), (-7.0, 3.2, 0.8), (-11.5, 2.8, -1.8), (-9.8, 4.8, 2.2),
    (-8.8, 1.2, 0.2), (-12.5, 2.2, 2.0), (-10.8, 5.2, -1.2), (-7.8, 4.2, -1.8),
)

YAW_CLASS_FIELDS = {
    "yaw_class", "passed", "refusal", "inliers", "informative", "inlier_fraction",
    "median_rho_degrees", "parallax_degrees", "cheirality_fraction",
    "singular_ratio", "singular_values",
}
ARM_FIELDS = {
    "graph_edges", "support_floor", "raw_keypoints", "support", "selected_support",
    "selected_support_edges", "spatial_bins", "pose_class", "pose_classes",
    "pose_error", "status", "reason", "hypotheses", "selection_state", "pose_state",
}


@pytest.fixture(scope="module")
def module():
    """Load the evaluation script as a module without running its ``main``."""
    script = (Path(__file__).resolve().parents[1]
              / "scripts" / "evaluation" / "evaluate_support_sensitivity.py")
    spec = importlib.util.spec_from_file_location("ds_r5n_support_sensitivity", script)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


# --- synthetic geometry -----------------------------------------------------------


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


def _truth_hits(frame_id, truth_points):
    hits = []
    for index in range(N_TRUTH):
        pixel = (float(truth_points[index][0]), float(truth_points[index][1]))
        hits.append(SurfaceHit(
            frame_id=frame_id, coordinate=(index, 0, 0), slot="north",
            world_point=tuple(float(value) for value in _POINTS[index]),
            camera_depth=1.0, state=BlockState("minecraft:stone"), pixel=pixel,
        ))
    return tuple(hits)


def _scene(module):
    """Two synthetic frames: 20 exact truth pairs plus 1,000 no-truth keypoints."""
    rng = np.random.default_rng(17)
    left_camera = _camera(LEFT_POSITION, LEFT_YAW, LEFT_PITCH)
    right_camera = _camera(RIGHT_POSITION, RIGHT_YAW, RIGHT_PITCH)
    left_projected = project_points(_POINTS, left_camera, width=WIDTH, height=HEIGHT)
    right_projected = project_points(_POINTS, right_camera, width=WIDTH, height=HEIGHT)
    truth_left = np.asarray([[p[0], p[1]] for p in left_projected], dtype=np.float64)
    truth_right = np.asarray([[p[0], p[1]] for p in right_projected], dtype=np.float64)
    noise_left = np.column_stack(
        (rng.uniform(0.0, WIDTH, N_NOISE), rng.uniform(0.0, HEIGHT, N_NOISE)))
    noise_right = np.column_stack(
        (rng.uniform(0.0, WIDTH, N_NOISE), rng.uniform(0.0, HEIGHT, N_NOISE)))
    left_points = np.vstack([truth_left, noise_left])
    right_points = np.vstack([truth_right, noise_right])
    descriptors = rng.normal(size=(N_TRUTH + N_NOISE, 16))

    left = module.R5K.OracleFrameObservation(
        frame_id="a", width=WIDTH, height=HEIGHT,
        usable_pixels=WIDTH * HEIGHT, boundary_dropped=0,
        features=_features("a", left_points, descriptors),
        hits=_truth_hits("a", truth_left) + (None,) * N_NOISE,
        no_truth_reasons=(("exceeded_max_distance", N_NOISE),),
    )
    right = module.R5K.OracleFrameObservation(
        frame_id="b", width=WIDTH, height=HEIGHT,
        usable_pixels=WIDTH * HEIGHT, boundary_dropped=0,
        features=_features("b", right_points, descriptors),
        hits=_truth_hits("b", truth_right) + (None,) * N_NOISE,
        no_truth_reasons=(("exceeded_max_distance", N_NOISE),),
    )
    cameras = [
        module.oracle_camera_estimate(_frame_input("a", left_camera)),
        module.oracle_camera_estimate(_frame_input("b", right_camera)),
    ]
    frames = (_frame_input("a", left_camera), _frame_input("b", right_camera))
    return left, right, cameras, frames


def _pair(module):
    left, right, cameras, _frames = _scene(module)
    return module.attribute_pair(left, right, cameras[0], cameras[1])


# --- settings and default-path isolation -----------------------------------------


def test_control_settings_match_the_assigned_oracle_only_override(module) -> None:
    control = module.control_settings()

    assert control.min_support_absolute == 8
    assert control.min_support_fraction == 0.0
    assert module.CONTROL_SUPPORT_ABSOLUTE == 8
    assert module.CONTROL_SUPPORT_FRACTION == 0.0
    # The default arm is the untouched production settings.
    assert module.default_settings().min_support_fraction == 0.02
    assert module.DEFAULT_SETTINGS.min_support_fraction == 0.02
    assert module.DEFAULT_SETTINGS is not module.CONTROL_SETTINGS


def test_default_path_isolation_leaves_matcher_defaults_and_builder_intact(module) -> None:
    """Running all three arms must not change production settings or the builder."""
    original_builder = joint_matching.reciprocal_topk_graph
    defaults_before = JointMatchSettings()

    pair = _pair(module)

    assert joint_matching.reciprocal_topk_graph is original_builder
    assert JointMatchSettings() == defaults_before
    assert JointMatchSettings().min_support_fraction == 0.02
    assert JointMatchSettings().min_support_absolute == 8
    # The default arm is the DS-R5K oracle at production settings.
    assert pair.default_result.support_floor == 21
    assert pair.default_result.candidate_edges == pair.oracle_graph.edges.shape[0]
    # The control arm is the same graph under the assigned control floor.
    assert pair.control_result.support_floor == 8
    assert pair.control_result.candidate_edges == pair.oracle_graph.edges.shape[0]


def test_default_arm_refuses_where_the_control_reaches_the_same_oracle_graph(
    module,
) -> None:
    pair = _pair(module)

    assert pair.truth_pairs == N_TRUTH
    assert pair.oracle_graph.edges.shape[0] == N_TRUTH
    # Default production floor 21 is above the whole oracle graph: unreachable.
    assert pair.default_result.status == "refused"
    assert pair.default_result.reason == module.UNREACHABLE_REASON
    assert pair.default_result.support == 0
    assert pair.default_result.hypotheses == ()
    assert pair.default_solved is False
    # The control floor 8 admits the mode, the selection and the DS-11 solve.
    assert pair.control_result.support_floor == 8
    assert len(pair.control_result.hypotheses) > 0
    assert pair.control_result.support > 0
    assert pair.control_solved is True

    record = module.pair_record(pair)
    comparison = record["comparison"]
    assert comparison["control_gained_candidate_mode"] is True
    assert comparison["control_gained_selection"] is True
    assert comparison["control_gained_solve"] is True
    assert comparison["default_refusal_reason"] == module.UNREACHABLE_REASON
    assert comparison["support_floor_delta"] == 8 - 21

    summary = module.summarise_attributions([pair])
    paired = summary["default_vs_control"]
    assert paired["default_insufficient_one_to_one_support_pairs"] == 1
    assert paired["unreachable_gained_candidate_mode"] == 1
    assert paired["unreachable_gained_selection"] == 1
    assert paired["unreachable_gained_solve"] == 1
    assert paired["only_control_solved"] == 1
    assert summary["same_surface_not_same_feature_point"] == (
        module.SAME_SURFACE_NOT_SAME_FEATURE_POINT)


def test_arm_graph_isolation_uses_each_arms_own_graph(module) -> None:
    pair = _pair(module)

    assert pair.production_result.candidate_edges == pair.production_graph.edges.shape[0]
    assert pair.default_result.candidate_edges == pair.oracle_graph.edges.shape[0]
    assert pair.control_result.candidate_edges == pair.oracle_graph.edges.shape[0]
    # The production graph is the reciprocal descriptor graph, not the truth graph.
    assert pair.production_graph.edges.shape[0] != pair.oracle_graph.edges.shape[0]


# --- per-pair and per-yaw serialization ------------------------------------------


def test_pair_record_serializes_every_required_field(module) -> None:
    pair = _pair(module)
    record = module.pair_record(pair)

    assert record["oracle_graph_edge_count"] == N_TRUTH
    assert len(record["oracle_graph_edges"]) == N_TRUTH
    assert record["oracle_graph_edges"][0] and len(record["oracle_graph_edges"][0]) == 2
    assert record["truth_pairs"] == N_TRUTH
    for arm in module.ARM_ORDER:
        arm_record = record[arm]
        assert ARM_FIELDS <= set(arm_record)
        assert arm_record["raw_keypoints"] == {
            "left": N_TRUTH + N_NOISE, "right": N_TRUTH + N_NOISE}
        assert arm_record["support"] == arm_record["selected_support"]
        assert len(arm_record["selected_support_edges"]) == arm_record["support"]
        assert all(len(edge) == 2 for edge in arm_record["selected_support_edges"])
        assert set(arm_record["spatial_bins"]) == {"left", "right", "total"}
        assert isinstance(arm_record["hypotheses"], list)
        assert "pose_error" in arm_record
    assert record[module.ARM_CONTROL]["hypotheses"]


def test_per_yaw_pose_classes_serialize_all_solve_pair_fields(module) -> None:
    pair = _pair(module)
    assert pair.control_solved is True
    record = module.pair_record(pair)
    control = record[module.ARM_CONTROL]

    assert control["pose_classes_available"] is True
    classes = control["pose_classes"]
    assert [item["yaw_class"] for item in classes] == [0, 1, 2, 3]
    for item in classes:
        assert set(item) == YAW_CLASS_FIELDS
    assert control["pose_class"] == pair.control_result.pose.right_yaw_class
    assert control["pose"]["right_yaw_class"] == control["pose_class"]
    # The assisted factory cameras fix the ground-truth yaw quadrant.
    assert control["pose_error"]["true_yaw_class"] == 0
    assert control["pose_error"]["yaw_class_correct"] is True


def test_assisted_camera_pose_error_is_defined_for_a_solved_pose(module) -> None:
    pair = _pair(module)
    record = module.pair_record(pair)
    error = record[module.ARM_CONTROL]["pose_error"]

    assert error["assisted"] is True
    assert error["defined"] is True
    assert error["undefined_reason"] is None
    assert error["relative_rotation_error_degrees"] == pytest.approx(0.0, abs=1e-6)
    assert error["baseline_error_defined"] is True
    assert error["true_baseline_length_blocks"] == pytest.approx(
        float(np.linalg.norm(
            np.asarray(RIGHT_POSITION) - np.asarray(LEFT_POSITION))), rel=1e-6)
    assert error["gauge_index"] in (0, 1, 2, 3)
    assert error["baseline_error_degrees_gauge_aligned"] is not None
    assert error["baseline_error_degrees_signless"] is not None


def test_pose_error_is_undefined_when_the_pose_stage_never_ran(module) -> None:
    pair = _pair(module)
    record = module.pair_record(pair)
    error = record[module.ARM_DEFAULT]["pose_error"]

    assert error["defined"] is False
    assert error["undefined_reason"] == "pose_not_attempted"
    assert error["solved_yaw_class"] is None
    assert record[module.ARM_DEFAULT]["pose_classes"] == []


def test_payload_carries_no_pixel_payload(module, monkeypatch) -> None:
    entry, scene, left, right = _payload_scene(module)
    monkeypatch.setattr(module, "build_scene", lambda _entry: scene)
    monkeypatch.setattr(
        module, "observe_oracle_frame",
        lambda _scene, frame, detection=None: left if frame.frame_id == "a" else right)
    payload = module.build_payload(
        "exports/datasets", [entry], build_group_manifest([entry]), entry)

    forbidden = {"pixels", "matches", "screenshot_path", "points", "descriptors",
                 "rgb", "image"}
    found: list[str] = []

    def _walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in forbidden:
                    found.append(key)
                _walk(item)
        elif isinstance(value, list):
            for item in value:
                _walk(item)

    _walk(payload)
    assert found == []
    json.dumps(payload, sort_keys=True, allow_nan=False)


# --- payload assembly, sealing and runner guards ---------------------------------


def _entry(module, *, dataset_id=None, scene_id="canal-town-architectural-component-bbox"):
    dataset_id = module.R5K_CANONICAL_DATASET_ID if dataset_id is None else dataset_id
    return DatasetSceneInput(
        dataset_id=dataset_id, index_path=f"exports/datasets/{dataset_id}/index.json",
        index_sha256=dict(module.R5K_INDEX_PINS).get(dataset_id, "0" * 64),
        scene_id=scene_id, scene_group="group-canal", split="train",
        directory="scenes/canal-town", scene_root="exports/datasets/x/scenes/canal-town",
        hashes=tuple(sorted(module.R5K_CANONICAL_SCENE_FILE_PINS.items())),
        screenshots=tuple(
            (f"frame-{index:02d}.png", digest)
            for index, digest in enumerate(sorted(module.R5K_CANONICAL_SCREENSHOT_PINS))
        ),
    )


def _payload_scene(module):
    left, right, _cameras, frames = _scene(module)
    entry = _entry(module)
    scene = FinalizedScene(
        input=entry, frames=frames, occupancy={}, region_min=(0, 0, 0),
        region_max=(1, 1, 1),
    )
    return entry, scene, left, right


def test_build_payload_seals_deterministically(module, monkeypatch) -> None:
    entry, scene, left, right = _payload_scene(module)
    monkeypatch.setattr(module, "build_scene", lambda _entry: scene)
    monkeypatch.setattr(
        module, "observe_oracle_frame",
        lambda _scene, frame, detection=None: left if frame.frame_id == "a" else right)
    manifest = build_group_manifest([entry])

    payload = module.build_payload("exports/datasets", [entry], manifest, entry)
    again = module.build_payload("exports/datasets", [entry], manifest, entry)

    assert payload["schema"] == module.SCHEMA
    assert payload["seal_sha256"] == again["seal_sha256"]
    assert payload["seal_sha256"] == module.seal_record(payload)
    assert payload["scene"]["pair_count"] == 1
    assert payload["arm_policy"]["control_is_oracle_only"] is True
    assert payload["arm_policy"]["control_settings"] == {
        "min_support_absolute": 8, "min_support_fraction": 0.0}
    assert payload["promotion_or_production_change"] is False
    assert payload["r5k_sealed_artifact_untouched"] == module.R5K_SEALED_ARTIFACT
    assert payload["oracle_policy"]["same_surface_not_same_feature_point"] == (
        module.SAME_SURFACE_NOT_SAME_FEATURE_POINT)
    assert payload["limitations"].count(module.SAME_SURFACE_NOT_SAME_FEATURE_POINT) == 1
    assert payload["consistency"]["ok"] is True
    assert payload["summary"]["default_vs_control"]["unreachable_gained_solve"] == 1

    tampered = copy.deepcopy(payload)
    tampered["status"] = "tampered"
    assert module.seal_record(tampered) != payload["seal_sha256"]


def test_reserved_output_paths_are_distinct_and_correct(module) -> None:
    assert module.DEFAULT_OUTPUT == ".tmp-oracle-support-sensitivity-20261003.json"
    assert module.R5K_SEALED_ARTIFACT == ".tmp-oracle-candidate-graph-20261003.json"
    assert module.DEFAULT_OUTPUT != module.R5K_SEALED_ARTIFACT
    assert module.R5K_SEALED_ARTIFACT not in module.DEFAULT_OUTPUT


def test_verify_r5k_pins_accepts_the_frozen_set_and_rejects_a_mismatch(module) -> None:
    entries = [
        _entry(module, dataset_id=dataset_id)
        for dataset_id in sorted(module.R5K_INDEX_PINS)
    ]
    record = module.verify_r5k_pins(entries)
    assert record["all_match"] is True
    assert record["entry_count"] == len(module.R5K_INDEX_PINS)

    tampered = list(entries)
    original = tampered[0]
    tampered[0] = DatasetSceneInput(
        dataset_id=original.dataset_id, index_path=original.index_path,
        index_sha256="f" * 64, scene_id=original.scene_id,
        scene_group=original.scene_group, split=original.split,
        directory=original.directory, scene_root=original.scene_root,
        hashes=original.hashes, screenshots=original.screenshots)
    with pytest.raises(module.SupportSensitivityError):
        module.verify_r5k_pins(tampered)

    with pytest.raises(module.SupportSensitivityError):
        module.verify_r5k_pins(entries[:-1])


def test_verify_canonical_scene_pins_requires_the_exact_r5k_pins(module) -> None:
    canonical = _entry(module)
    record = module.verify_canonical_scene_pins(canonical)
    assert record["all_match"] is True
    assert record["screenshot_count"] == len(module.R5K_CANONICAL_SCREENSHOT_PINS)

    wrong_dataset = _entry(module, dataset_id="fresh-c4-canal-town-2026-09-12")
    with pytest.raises(module.SupportSensitivityError):
        module.verify_canonical_scene_pins(wrong_dataset)

    mutated = DatasetSceneInput(
        dataset_id=canonical.dataset_id, index_path=canonical.index_path,
        index_sha256=canonical.index_sha256, scene_id=canonical.scene_id,
        scene_group=canonical.scene_group, split=canonical.split,
        directory=canonical.directory, scene_root=canonical.scene_root,
        hashes=tuple(sorted((name, "0" * 64) for name, _digest in [
            list(canonical.hashes)[0]])), screenshots=canonical.screenshots)
    with pytest.raises(module.SupportSensitivityError):
        module.verify_canonical_scene_pins(mutated)


def test_runner_refuses_the_sealed_r5k_artifact_path(module, tmp_path: Path) -> None:
    output = tmp_path / module.R5K_SEALED_ARTIFACT
    assert module.main(
        ["--output", str(output), "--exports-root", str(tmp_path)]) == 2
    assert not output.exists()


def test_runner_refuses_a_mixed_case_sealed_r5k_artifact_path(
    module, tmp_path: Path,
) -> None:
    """Windows is case-insensitive: a case-folded spelling names the sealed file.

    The path lives under ``tmp_path`` and is never created, so the real sealed
    artifact is neither read nor written; only the runner guard is exercised.
    """
    mixed_case = ".Tmp-OrAcLe-CaNdIdAtE-GrApH-20261003.JsOn"
    assert mixed_case != module.R5K_SEALED_ARTIFACT
    assert mixed_case.casefold() == module.R5K_SEALED_ARTIFACT.casefold()
    output = tmp_path / mixed_case

    assert module.main(
        ["--output", str(output), "--exports-root", str(tmp_path)]) == 2
    assert not output.exists()


def test_runner_refuses_to_overwrite_an_existing_output(module, tmp_path: Path) -> None:
    output = tmp_path / "out.json"
    output.write_text("keep", encoding="utf-8")

    assert module.main(
        ["--output", str(output), "--exports-root", str(tmp_path)]) == 2
    assert output.read_text(encoding="utf-8") == "keep"


def test_runner_reports_pin_failure_without_writing(
    module, tmp_path: Path, monkeypatch,
) -> None:
    output = tmp_path / "absent.json"
    monkeypatch.setattr(module, "discover_finalized_entries", lambda _root: (_entry(module),))
    monkeypatch.setattr(
        module, "verify_r5k_pins",
        lambda _entries: (_ for _ in ()).throw(
            module.SupportSensitivityError("pin mismatch")))

    assert module.main(
        ["--output", str(output), "--exports-root", str(tmp_path)]) == 1
    assert not output.exists()


def test_runner_reports_leakage_failure_without_writing(
    module, tmp_path: Path, monkeypatch,
) -> None:
    output = tmp_path / "absent.json"
    monkeypatch.setattr(module, "discover_finalized_entries", lambda _root: (_entry(module),))
    monkeypatch.setattr(module, "verify_r5k_pins", lambda _entries: {"all_match": True})
    monkeypatch.setattr(
        module, "build_group_manifest",
        lambda _entries: (_ for _ in ()).throw(
            PairCorrespondenceEvaluationError("fold leakage")))

    assert module.main(
        ["--output", str(output), "--exports-root", str(tmp_path)]) == 1
    assert not output.exists()


def test_runner_writes_once_then_refuses_a_second_run(
    module, tmp_path: Path, monkeypatch,
) -> None:
    output = tmp_path / "sealed.json"
    monkeypatch.setattr(module, "discover_finalized_entries", lambda _root: (_entry(module),))
    monkeypatch.setattr(module, "verify_r5k_pins", lambda _entries: {"all_match": True})
    monkeypatch.setattr(module, "build_group_manifest", lambda _entries: {"groups": []})
    monkeypatch.setattr(module, "select_canonical_entry", lambda _entries, dataset_id=None: _entry(module))
    monkeypatch.setattr(module, "verify_canonical_scene_pins", lambda _entry: {"all_match": True})
    monkeypatch.setattr(
        module, "build_payload",
        lambda *_args, **_kwargs: {
            "schema": module.SCHEMA, "seal_sha256": "0" * 64,
            "scene": {"dataset_id": "x", "frame_count": 0},
            "summary": {
                "pair_count": 0, "truth_pairs": 0,
                "arms": {module.ARM_DEFAULT: {"solved": 0},
                         module.ARM_CONTROL: {"solved": 0}},
                "default_vs_control": {
                    "default_insufficient_one_to_one_support_pairs": 0,
                    "unreachable_gained_candidate_mode": 0,
                    "unreachable_gained_selection": 0,
                    "unreachable_gained_solve": 0,
                },
            },
        })

    assert module.main(
        ["--output", str(output), "--exports-root", str(tmp_path)]) == 0
    assert output.exists()
    assert module.main(
        ["--output", str(output), "--exports-root", str(tmp_path)]) == 2


def test_runner_reports_failure_without_writing_when_exports_are_missing(
    module, tmp_path: Path,
) -> None:
    output = tmp_path / "absent.json"
    missing = tmp_path / "missing-root"

    assert module.main(
        ["--output", str(output), "--exports-root", str(missing)]) == 1
    assert not output.exists()
