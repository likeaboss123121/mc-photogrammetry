"""Focused synthetic tests for the DS-STAGEWISE-K1 paired camera arm.

These tests are self-contained: they read no fixture image, reference, camera
manifest, mask, JAR or truth payload, and they never run a reconstruction or a
production/evaluator path.  They pin the contracts the lead entry requires:

* evaluator order - A0 reconstruction, A0 serialization/seal, evaluator camera,
  A1 reconstruction, both arms frozen, then truth;
* evaluator access is refused before the A0 seal and before both arms are frozen;
* the A1 record is refused unless it is labelled ``reference-assisted`` and out of
  promotion;
* a malformed/absent evaluator camera is a typed refusal (default loader and the
  arm builder);
* both arms differ only in the ``CameraModel`` object: the image, GUI mask,
  palette, rows/columns/depth, ``horizon`` and ``scene_transform`` are identical
  objects/values;
* exact code/input hashes are present and stable, and the production path cannot
  import the arm; ``reconstruct_facade``'s signature is unchanged;
* a visible-denominator mismatch against the fixed ``V`` refuses instead of
  silently rescoring.
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mc_photogrammetry import evaluation_image_only_attribution as attribution
from mc_photogrammetry.blocks.appearance import DEFAULT_PALETTE
from mc_photogrammetry.domain.models import (
    BlockObservation,
    BlockState,
    BlockVolume,
    ObservationKind,
)
from mc_photogrammetry.reconstruction.camera import CameraModel
from mc_photogrammetry.reconstruction.facade import reconstruct_facade

REPO_ROOT = Path(__file__).resolve().parents[1]
ARM_PATH = REPO_ROOT / "scripts" / "evaluation" / "run_image_only_route_a_camera_arm.py"

STONE = BlockState("minecraft:stone")
DIRT = BlockState("minecraft:dirt")
CELL = (0, 0, 0)


def _load_arm():
    name = "_k1_camera_arm_under_test"
    spec = importlib.util.spec_from_file_location(name, ARM_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def arm():
    return _load_arm()


def _volume(cells_and_states):
    volume = BlockVolume()
    for coordinate, state in cells_and_states.items():
        volume.set(coordinate, BlockObservation(
            state=state, kind=ObservationKind.OBSERVED))
    return volume


def _synthetic_outcome():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=bool)
    frontier = SimpleNamespace(
        to_record=lambda: {"route": "route_a", "stage": "facade_grid"},
        terminal_policy={"stage": "facade_grid"}, terminal_status="solved")
    return SimpleNamespace(
        image=image, mask=mask, palette=DEFAULT_PALETTE[:2],
        camera=CameraModel(position=(0.0, 0.0, 0.0), yaw_degrees=0.0,
                           pitch_degrees=0.0, fov_degrees=70.0),
        camera_error="",
        facade=_volume({CELL: STONE}),
        decoded_export=_volume({CELL: STONE}),
        route_a=SimpleNamespace(frontier=frontier))


def _synthetic_truth(cells=None):
    reference = {CELL: STONE} if cells is None else cells
    visible = attribution.visible_reference_from_counts(
        reference, {coordinate: 5 for coordinate in reference},
        source="synthetic")
    return {
        "reference": reference,
        "visible_reference": visible,
        "reference_sha256": "0" * 64,
        "mask_origin": "synthetic",
        "mask_pins_match": True,
        "reference_pin_matches": True,
        "frames": [],
        "caveats": [],
    }


def _evaluator_camera():
    return CameraModel(position=(-1006.1, 108.0, 8017.8), yaw_degrees=-42.7,
                       pitch_degrees=-4.6, fov_degrees=110.0)


def _ordered_hooks(arm, workdir, outcome, truth, captured):
    events = []

    def run_a0(**kwargs):
        events.append("a0_hook")
        return outcome

    def load_camera(**_kwargs):
        assert (workdir / "a0-seal.json").is_file(), "camera read before the A0 seal"
        events.append("camera_hook")
        return _evaluator_camera()

    def reconstruct(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _volume({CELL: STONE})

    def load_truth(**_kwargs):
        assert (workdir / "a0-seal.json").is_file(), "truth read before the A0 seal"
        assert (workdir / "arms-frozen.json").is_file(), "truth read before freeze"
        events.append("truth_hook")
        return truth

    def record(event, detail):
        events.append(event)

    hooks = arm.ArmHooks(run_a0=run_a0, load_evaluator_camera=load_camera,
                         reconstruct=reconstruct, load_truth=load_truth,
                         record=record)
    return hooks, events


def _run(arm, tmp_path, hooks, expected_visible=1):
    workdir = tmp_path / "work"
    return arm.run_camera_arm(
        image=tmp_path / "frame.png", fixture=tmp_path / "fixture",
        reference=tmp_path / "ref.litematic", scene=tmp_path / "scene.json",
        cameras=tmp_path / "cameras.json", masks=tmp_path / "masks",
        pins=tmp_path / "pins.json", fixture_role="synthetic",
        workdir=workdir, hooks=hooks,
        expected_visible_cells=expected_visible)


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

def test_order_is_a0_seal_then_camera_then_a1_then_freeze_then_truth(arm, tmp_path):
    outcome = _synthetic_outcome()
    captured = {}
    hooks, events = _ordered_hooks(
        arm, tmp_path / "work", outcome, _synthetic_truth(), captured)
    summary = _run(arm, tmp_path, hooks)

    assert events == [
        "start", "a0_hook", "a0_reconstructed", "a0_sealed", "camera_hook",
        "evaluator_camera_loaded", "a1_reconstructed", "arms_frozen",
        "truth_hook", "truth_loaded", "scored",
    ]
    assert summary["order_sentinel"] == arm.ORDER_SENTINEL
    assert summary["a1_provenance"] == arm.REFERENCE_ASSISTED
    assert summary["a1_out_of_promotion"] is True
    assert summary["camera_is_the_only_changed_object"] is True
    assert summary["fixed_visible_cells"] == {"expected": 1, "observed": 1}


def test_evaluator_access_is_refused_before_the_a0_seal_and_freeze(arm):
    guard = arm.ArmOrderingGuard()
    with pytest.raises(arm.ArmOrderingRefusal, match="before A0"):
        guard.require_sealed("the evaluator camera load")
    with pytest.raises(arm.ArmOrderingRefusal, match="before the A0 seal"):
        guard.mark_frozen()
    guard.mark_sealed()
    assert guard.sealed is True
    with pytest.raises(arm.ArmOrderingRefusal, match="before both reconstructions"):
        guard.require_frozen("the visible-reference truth load")
    guard.mark_frozen()
    assert guard.frozen is True
    guard.require_frozen("the visible-reference truth load")


def test_a_failed_a0_seal_stops_before_any_evaluator_access(arm, tmp_path):
    loaded = []

    def run_a0(**_kwargs):
        raise arm.ArmRefusal("a0-stop")

    def load_camera(**_kwargs):
        loaded.append("camera")
        return _evaluator_camera()

    def load_truth(**_kwargs):
        loaded.append("truth")
        return _synthetic_truth()

    def reconstruct(*_args, **_kwargs):
        return _volume({CELL: STONE})

    hooks = arm.ArmHooks(run_a0=run_a0, load_evaluator_camera=load_camera,
                         reconstruct=reconstruct, load_truth=load_truth,
                         record=lambda event, detail: None)
    with pytest.raises(arm.ArmRefusal, match="a0-stop"):
        _run(arm, tmp_path, hooks)
    assert loaded == []
    assert not (tmp_path / "work" / "a0-seal.json").exists()


# ---------------------------------------------------------------------------
# Reference-assisted label
# ---------------------------------------------------------------------------

def _assisted_kwargs(module, **overrides):
    values = {
        "arm": "A1", "provenance": module.REFERENCE_ASSISTED,
        "out_of_promotion": True,
        "evaluator_camera": {"position": [0.0, 0.0, 0.0]},
        "evaluator_camera_sha256": "0" * 64, "facade_block_count": 1,
        "facade_volume_sha256": "1" * 64,
    }
    values.update(overrides)
    return values


def test_a1_record_without_the_reference_assisted_label_is_refused(arm):
    with pytest.raises(arm.ArmRefusal, match="reference-assisted"):
        arm.ReferenceAssistedArm(**_assisted_kwargs(arm, provenance=arm.IMAGE_ONLY))
    with pytest.raises(arm.ArmRefusal, match="out of promotion"):
        arm.ReferenceAssistedArm(**_assisted_kwargs(arm, out_of_promotion=False))
    with pytest.raises(arm.ArmRefusal, match="arm A1"):
        arm.ReferenceAssistedArm(**_assisted_kwargs(arm, arm="A0"))
    record = arm.ReferenceAssistedArm(**_assisted_kwargs(arm)).to_record()
    assert record["provenance"] == arm.REFERENCE_ASSISTED
    assert record["out_of_promotion"] is True
    assert record["role"] == "evaluator-only oracle-camera sensitivity arm"


# ---------------------------------------------------------------------------
# Malformed / absent evaluator camera
# ---------------------------------------------------------------------------

def test_default_camera_loader_refuses_absent_and_malformed_entries(arm, tmp_path):
    manifest = tmp_path / "cameras.json"
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(arm.ArmCameraRefusal, match="no entry"):
        arm._default_load_evaluator_camera(
            cameras_path=manifest, frame_id="frame.png")
    manifest.write_text('{"frame.png": {"position": [1, 2]}}', encoding="utf-8")
    with pytest.raises(arm.ArmCameraRefusal, match="unusable"):
        arm._default_load_evaluator_camera(
            cameras_path=manifest, frame_id="frame.png")
    manifest.write_text('{"frame.png": {"position": [1, 2, 3], '
                        '"yaw_degrees": 0, "pitch_degrees": 200}}', encoding="utf-8")
    with pytest.raises(arm.ArmCameraRefusal, match="unusable"):
        arm._default_load_evaluator_camera(
            cameras_path=manifest, frame_id="frame.png")


def test_arm_refuses_a_non_camera_evaluator_object(arm, tmp_path):
    outcome = _synthetic_outcome()
    hooks = arm.ArmHooks(
        run_a0=lambda **_kwargs: outcome,
        load_evaluator_camera=lambda **_kwargs: object(),
        reconstruct=lambda *_args, **_kwargs: _volume({CELL: STONE}),
        load_truth=lambda **_kwargs: _synthetic_truth(),
        record=lambda event, detail: None)
    with pytest.raises(arm.ArmCameraRefusal, match="CameraModel"):
        _run(arm, tmp_path, hooks)
    with pytest.raises(arm.ArmCameraRefusal, match="CameraModel"):
        arm.build_a1_arm(outcome=outcome, camera=object(), settings=arm.ArmSettings(),
                         reconstruct=lambda *_args, **_kwargs: None, factors={})


# ---------------------------------------------------------------------------
# Single factor: only the camera object changes
# ---------------------------------------------------------------------------

def test_both_arms_share_every_non_camera_object_and_setting(arm, tmp_path):
    outcome = _synthetic_outcome()
    captured = {}
    hooks, _events = _ordered_hooks(
        arm, tmp_path / "work", outcome, _synthetic_truth(), captured)
    summary = _run(arm, tmp_path, hooks)

    args = captured["args"]
    kwargs = captured["kwargs"]
    assert args[0] is outcome.image
    assert args[1] is outcome.mask
    assert kwargs["palette"] is outcome.palette
    assert kwargs["columns"] == 32 and kwargs["rows"] == 32
    assert kwargs["depth_layers"] == 8
    assert kwargs["horizon"] is None
    assert kwargs["scene_transform"] is None
    assert isinstance(kwargs["camera"], CameraModel)
    assert kwargs["camera"] is not outcome.camera
    # The two factor digests are identical because every non-camera factor is the
    # same object/value; the frozen record asserts it too.
    assert summary["camera_is_the_only_changed_object"] is True


def test_seal_records_exact_input_code_and_camera_digests(arm, tmp_path):
    outcome = _synthetic_outcome()
    settings = arm.ArmSettings()
    hashes = {"scripts/evaluation/run_image_only_route_a.py": "a" * 64}
    first = arm.build_a0_seal(outcome, settings, hashes)
    second = arm.build_a0_seal(outcome, settings, hashes)
    assert first == second
    assert first["a0_seal_sha256"] == second["a0_seal_sha256"]
    assert first["order_sentinel"] == arm.ORDER_SENTINEL
    assert first["provenance"] == arm.IMAGE_ONLY
    assert first["sealed"] is True
    assert first["evaluator_reads_before_seal"] == 0
    assert first["code_hashes"] == hashes
    assert first["camera_sha256"] == arm._digest_json(outcome.camera.to_dict())
    assert first["factor_record_sha256"] == arm._digest_json(first["factor_record"])
    assert first["export_round_trip_exact"] is True
    assert first["facade_block_count"] == 1


def test_score_arm_reports_exact_cell_sets_and_deltas(arm):
    reference = {(0, 0, 0): STONE, (1, 0, 0): DIRT, (2, 0, 0): STONE}
    truth = _synthetic_truth(reference)
    perfect = arm.score_arm(label="A0", provenance=arm.IMAGE_ONLY,
                            states=dict(reference), truth=truth,
                            out_of_promotion=False)
    assert perfect["visible_cells"] == 3
    assert perfect["correct_cells"] == 3
    assert perfect["missing_cells"] == 0
    assert perfect["false_cells"] == 0
    assert perfect["recall"] == 1.0
    assert perfect["precision"] == 1.0
    assert perfect["f1"] == 1.0
    assert len(perfect["correct_cell_sha256"]) == 64
    assert perfect["correct_cell_sha256"] != perfect["missing_cell_sha256"]

    partial = {(0, 0, 0): STONE, (1, 0, 0): DIRT, (9, 9, 9): STONE}
    scored = arm.score_arm(label="A1", provenance=arm.REFERENCE_ASSISTED,
                           states=partial, truth=truth, out_of_promotion=True)
    assert scored["visible_cells"] == 3
    assert scored["correct_cells"] == 2
    assert scored["missing_cells"] == 1
    assert scored["false_cells"] == 1
    assert scored["recall"] == pytest.approx(2 / 3)
    assert scored["precision"] == pytest.approx(2 / 3)
    assert scored["f1"] == pytest.approx(2 / 3)
    deltas = arm._delta(perfect, scored,
                        ("correct_cells", "missing_cells", "false_cells"))
    assert deltas == {"correct_cells": -1, "missing_cells": 1, "false_cells": 1}


def test_visible_denominator_mismatch_refuses_instead_of_rescoring(arm, tmp_path):
    outcome = _synthetic_outcome()
    captured = {}
    hooks, _events = _ordered_hooks(
        arm, tmp_path / "work", outcome, _synthetic_truth(), captured)
    with pytest.raises(arm.ArmRefusal, match="fixed V"):
        _run(arm, tmp_path, hooks, expected_visible=398)


# ---------------------------------------------------------------------------
# Production isolation and unchanged production signature
# ---------------------------------------------------------------------------

def test_no_production_module_imports_the_arm():
    needle = "run_image_only_route_a_camera_arm"
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "src").rglob("*.py")
        if needle in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_reconstruct_facade_signature_is_unchanged(arm):
    signature = inspect.signature(reconstruct_facade)
    assert list(signature.parameters) == [
        "image", "gui_mask", "columns", "rows", "depth_layers", "horizon",
        "camera", "scene_transform", "palette",
    ]
    assert signature.parameters["camera"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["camera"].default is None
    assert signature.parameters["scene_transform"].default is None
    assert signature.parameters["columns"].default == 32
    assert signature.parameters["rows"].default == 32
    assert signature.parameters["depth_layers"].default == 1
    assert list(inspect.signature(arm.reconstruct_facade).parameters) == \
        list(signature.parameters)


def test_harness_is_a_sibling_evaluation_runner(arm):
    assert ARM_PATH.parent == REPO_ROOT / "scripts" / "evaluation"
    assert arm.HARNESS_PATH == ARM_PATH
    assert arm.ROUTE_A_RUNNER_PATH.name == "run_image_only_route_a.py"
    assert callable(arm.DEFAULT_HOOKS.run_a0)
    assert callable(arm.DEFAULT_HOOKS.load_truth)
