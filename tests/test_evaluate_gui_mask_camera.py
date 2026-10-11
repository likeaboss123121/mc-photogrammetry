"""Synthetic offline checks; never open the pinned images or evaluation truth."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from mc_photogrammetry.pipeline.contracts import CameraEstimate, CameraIntrinsics, CameraPose, CameraSolution
from mc_photogrammetry.reconstruction import calibration
from mc_photogrammetry.vision.gui_mask import create_gui_mask
from scripts.evaluation import evaluate_gui_mask_camera as d

SOURCE_CLOSURE_SHA = "A" * 64
RUNTIME_CLOSURE_SHA = "B" * 64
CLOSURE_PINS = dict(source_closure_sha256=SOURCE_CLOSURE_SHA,
                    runtime_closure_sha256=RUNTIME_CLOSURE_SHA)


@pytest.fixture(autouse=True)
def no_real_payloads(monkeypatch):
    """Fail locally if any test accidentally accesses a restricted payload."""
    forbidden = {(d.ROOT / d.CAPTURE_PATH).resolve(), (d.ROOT / d.BASELINE_PATH).resolve()}
    scene = (d.ROOT / d.SCENE).resolve()
    original = Path.open

    def watched(path, *args, **kwargs):
        actual = path.resolve()
        if actual in forbidden or actual.is_relative_to(scene) or path.name == "ground_truth.json":
            pytest.fail("offline test attempted restricted payload access")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", watched)

    def no_shared(*args, **kwargs):
        pytest.fail("harness attempted shared-focal calibration")

    monkeypatch.setattr(calibration, "estimate_session", no_shared)


def seals():
    return [dict(frame=identity, ordinal=i, path=f"synthetic-{i:02d}.png", bytes=100 + i,
                 sha256=str(i) * 64) for i, identity in enumerate(d.IDS)]


def arrays():
    return {identity: np.full((80, 100, 3), 20 + i, dtype=np.uint8)
            for i, identity in enumerate(d.IDS)}


def estimate(scalar=float):
    return CameraEstimate(CameraIntrinsics((100, 80), (scalar(50.), scalar(50.)),
                          (scalar(50.), scalar(40.))),
                          CameraPose(None, scalar(0.), scalar(0.), scalar(0.)))


def predictions(monkeypatch, unavailable=()):
    solution = CameraSolution({identity: estimate() for i, identity in enumerate(d.IDS)
                               if i not in unavailable})
    monkeypatch.setattr(d.image_only, "solve_image_cameras", lambda frames: solution)
    return d.predict_cameras(arrays(), seals(), **CLOSURE_PINS)


def scalar_truth(tmp_path, monkeypatch, table):
    """Write scalar synthetic records in the exact source-defined schemas."""
    capture, baseline = {"frames": [], "screenshots": []}, {"frames": [], "input_seals": []}
    for current, seal in zip(table["frames"], table["input_seals"], strict=True):
        filename = Path(seal["path"]).name
        actual = dict(eye_position=[0., 0., 0.], yaw_degrees=0., pitch_degrees=0.,
                      roll_degrees=0., fov_degrees=float(np.degrees(2 * np.arctan(40 / 50))),
                      framebuffer_width=100, framebuffer_height=80,
                      screenshot_sha256=seal["sha256"], screenshot_bytes=seal["bytes"])
        capture["frames"].append(dict(frame_id=Path(filename).stem, filename=filename,
                                      captured=True, actual=actual))
        capture["screenshots"].append(dict(filename=filename, frame_id=Path(filename).stem,
                                           sha256=seal["sha256"], bytes=seal["bytes"]))
        old = dict(frame_id=current["frame_id"], status="available", image_size=[100, 80],
                   yaw_degrees=0., pitch_degrees=0., roll_degrees=0.,
                   focal_length_pixels=[50., 50.], principal_point_pixels=[50., 40.])
        baseline["frames"].append(dict(frame_id=current["frame_id"], shape=[80, 100, 3], image_camera=old))
        baseline["input_seals"].append({k: seal[k] for k in ("path", "bytes", "sha256")})
    capture_path, baseline_path = tmp_path / "synthetic-capture.json", tmp_path / "synthetic-baseline.json"
    write_truth(monkeypatch, capture_path, baseline_path, capture, baseline)
    return capture_path, baseline_path, capture, baseline


def write_truth(monkeypatch, capture_path, baseline_path, capture, baseline):
    for value, path, prefix in ((capture, capture_path, "CAPTURE"), (baseline, baseline_path, "BASELINE")):
        raw = d.encoded(value)
        path.write_bytes(raw)
        monkeypatch.setattr(d, prefix + "_BYTES", len(raw))
        monkeypatch.setattr(d, prefix + "_SHA", d.digest(raw))


def run_frozen(tmp_path, monkeypatch, table):
    capture_path, baseline_path, _, _ = scalar_truth(tmp_path, monkeypatch, table)
    path = tmp_path / "predictions.json"
    seal = d.freeze_predictions(path, table)
    return d.evaluate_frozen(path, seal, capture_path, baseline_path)


def test_single_exact_production_adapter_call_and_one_estimation_per_frame(monkeypatch):
    frames = arrays()
    before = {key: value.copy() for key, value in frames.items()}
    original = d.image_only.solve_image_cameras
    delegated, calls = [], []

    def observed(mapping):
        delegated.append(mapping)
        return original(mapping)

    def fitted(grey):
        calls.append(grey.copy())
        index = len(calls) - 1
        mask = create_gui_mask(*grey.shape)
        assert np.all(grey[mask] == 0)
        assert np.all(grey[~mask] == 20 + index)
        return calibration.CalibrationResult(50. + index, 0., 0., 100, 80, 12, 12, 0.)

    monkeypatch.setattr(d.image_only, "solve_image_cameras", observed)
    monkeypatch.setattr(calibration, "estimate_camera", fitted)
    table = d.predict_cameras(frames, seals(), **CLOSURE_PINS)
    assert len(delegated) == 1 and delegated[0] is frames and len(calls) == 6
    assert [row["camera"]["focal_length_pixels"] for row in table["frames"]] == [[50. + i] * 2 for i in range(6)]
    assert all(row["camera"]["translation"] is None and row["camera"]["scale"] is None for row in table["frames"])
    for key in frames:
        np.testing.assert_array_equal(frames[key], before[key])


@pytest.mark.parametrize("grey", [False, True])
def test_excluded_gui_pixels_invariant_and_inputs_not_mutated(monkeypatch, grey):
    frames = arrays()
    if grey:
        frames = {key: value[:, :, 0].copy() for key, value in frames.items()}
    changed = {key: value.copy() for key, value in frames.items()}
    mask = create_gui_mask(80, 100)
    for value in changed.values():
        value[mask] = 250 if grey else (250, 0, 120)
    before = {key: value.copy() for key, value in changed.items()}
    calls = []

    def fitted(image):
        calls.append(image.copy())
        return calibration.CalibrationResult(float(image.mean()), 0., 0., 100, 80, 12, 12, 0.)

    monkeypatch.setattr(calibration, "estimate_camera", fitted)
    baseline = d.predict_cameras(frames, seals(), **CLOSURE_PINS)
    masked = d.predict_cameras(changed, seals(), **CLOSURE_PINS)
    assert masked == baseline and len(calls) == 12
    for index, key in enumerate(d.IDS):
        np.testing.assert_array_equal(calls[index], calls[index + 6])
        np.testing.assert_array_equal(changed[key], before[key])
        assert np.all(calls[index][mask] == 0)
        assert np.all(frames[key] == 20 + index)


@pytest.mark.parametrize("absent", [(3,), (0, 1, 2, 3, 4, 5)])
def test_partial_and_all_production_abstention_freeze_and_fail_criterion(tmp_path, monkeypatch, absent):
    calls = []

    def fitted(grey):
        index = len(calls)
        calls.append(grey)
        return None if index in absent else calibration.CalibrationResult(50., 0., 0., 100, 80, 12, 12, 0.)

    monkeypatch.setattr(calibration, "estimate_camera", fitted)
    table = d.predict_cameras(arrays(), seals(), **CLOSURE_PINS)
    assert len(calls) == 6
    assert [i for i, row in enumerate(table["frames"]) if row["status"] == "abstained"] == list(absent)
    report = run_frozen(tmp_path, monkeypatch, table)
    assert not report["joint_camera_success"] and not report["primary_success"]
    assert report["decision"] == "stop_mask_camera_recovery_hypothesis"
    for i in absent:
        row = report["frames"][i]
        assert row["current"]["rotation_error_degrees"] is None
        assert row["current"]["qualifies"] is False
        assert all(value is None for value in row["deltas"].values())


@pytest.mark.parametrize("scalar", [np.float32, np.float64, np.int64])
def test_numpy_camera_scalars_detach_and_freeze(tmp_path, monkeypatch, scalar):
    camera = estimate(scalar)
    monkeypatch.setattr(d.image_only, "solve_image_cameras", lambda frames: CameraSolution({key: camera for key in d.IDS}))
    table = d.predict_cameras(arrays(), seals(), **CLOSURE_PINS)
    row = table["frames"][0]["camera"]
    assert type(row["yaw_degrees"]) is float
    assert all(type(v) is float for key in ("focal_length_pixels", "principal_point_pixels") for v in row[key])
    assert type(camera.pose.yaw_degrees) is scalar
    path = tmp_path / "numpy.json"
    seal = d.freeze_predictions(path, table)
    assert seal == {"bytes": path.stat().st_size, "sha256": d.digest(path.read_bytes())}
    d.validate_predictions(json.loads(path.read_bytes()))


@pytest.mark.parametrize("bad", [True, np.bool_(False), float("inf"), np.float32("nan"), "12"])
@pytest.mark.parametrize("field", ["yaw", "focal", "principal"])
def test_boolean_nonfinite_and_nonnumeric_predictions_refused(monkeypatch, bad, field):
    camera = CameraEstimate(CameraIntrinsics((100, 80), (50., bad if field == "focal" else 50.),
                          (bad if field == "principal" else 50., 40.)),
                          CameraPose(None, bad if field == "yaw" else 0., 0., 0.))
    monkeypatch.setattr(d.image_only, "solve_image_cameras", lambda frames: CameraSolution({key: camera for key in d.IDS}))
    with pytest.raises(ValueError, match="numeric scalar"):
        d.predict_cameras(arrays(), seals(), **CLOSURE_PINS)


def test_exclusive_freeze_close_size_sha_and_truth_order(tmp_path, monkeypatch):
    table = predictions(monkeypatch)
    capture_path, baseline_path, _, _ = scalar_truth(tmp_path, monkeypatch, table)
    path = tmp_path / "predictions.json"
    seal = d.freeze_predictions(path, table)
    with pytest.raises(FileExistsError):
        d.freeze_predictions(path, table)
    original = d.pinned_bytes
    reads = []

    def watched(source, size, sha, cap=d.MAX_JSON_BYTES):
        reads.append(Path(source))
        if Path(source) in (capture_path, baseline_path):
            assert reads[0] == path
            assert path.stat().st_size == seal["bytes"] and d.digest(path.read_bytes()) == seal["sha256"]
        return original(source, size, sha, cap)

    monkeypatch.setattr(d, "pinned_bytes", watched)
    report = d.evaluate_frozen(path, seal, capture_path, baseline_path)
    assert reads == [path, capture_path, baseline_path]
    assert report["joint_camera_success"] and report["truth_read_after_prediction_freeze"]
    assert report["decision"] == "one_scene_camera_evidence_only"
    assert report["camera_selected_by_truth"] is False


@pytest.mark.parametrize("fault", ["missing", "changed", "wrong_size", "no_seal", "invalid_table"])
def test_failed_or_missing_prediction_freeze_prevents_truth_parsing(tmp_path, monkeypatch, fault):
    table = predictions(monkeypatch)
    path = tmp_path / "predictions.json"
    seal = d.freeze_predictions(path, table)
    if fault == "missing":
        path.unlink()
    elif fault == "changed":
        path.write_bytes(path.read_bytes().replace(b'"available"', b'"abstained"', 1))
    elif fault == "wrong_size":
        seal["bytes"] += 1
    elif fault == "no_seal":
        seal = None
    else:
        table["frames"].pop()
        path.write_bytes(d.encoded(table))
        seal = {"bytes": path.stat().st_size, "sha256": d.digest(path.read_bytes())}
    original = d.pinned_bytes

    def watched(source, *args, **kwargs):
        assert Path(source) == path, "truth accessed without valid prediction freeze"
        return original(source, *args, **kwargs)

    monkeypatch.setattr(d, "pinned_bytes", watched)
    with pytest.raises((ValueError, FileNotFoundError)):
        d.evaluate_frozen(path, seal, tmp_path / "absent-capture", tmp_path / "absent-baseline")


@pytest.mark.parametrize("fault", ["filename", "order", "capture_hash", "capture_size", "screenshot_hash",
                                    "screenshot_size", "duplicate_shot", "baseline_order", "baseline_hash",
                                    "baseline_size", "baseline_path", "dimensions", "baseline_camera_id"])
def test_wrong_metadata_join_refused_after_freeze(tmp_path, monkeypatch, fault):
    table = predictions(monkeypatch)
    capture_path, baseline_path, capture, baseline = scalar_truth(tmp_path, monkeypatch, table)
    if fault == "filename": capture["frames"][0]["filename"] = "foreign.png"
    elif fault == "order": capture["frames"][0], capture["frames"][1] = capture["frames"][1], capture["frames"][0]
    elif fault == "capture_hash": capture["frames"][0]["actual"]["screenshot_sha256"] = "F" * 64
    elif fault == "capture_size": capture["frames"][0]["actual"]["screenshot_bytes"] = True
    elif fault == "screenshot_hash": capture["screenshots"][0]["sha256"] = "F" * 64
    elif fault == "screenshot_size": capture["screenshots"][0]["bytes"] += 1
    elif fault == "duplicate_shot": capture["screenshots"][1] = copy.deepcopy(capture["screenshots"][0])
    elif fault == "baseline_order": baseline["frames"][0], baseline["frames"][1] = baseline["frames"][1], baseline["frames"][0]
    elif fault == "baseline_hash": baseline["input_seals"][0]["sha256"] = "F" * 64
    elif fault == "baseline_size": baseline["input_seals"][0]["bytes"] += 1
    elif fault == "baseline_path": baseline["input_seals"][0]["path"] = "foreign.png"
    elif fault == "dimensions": capture["frames"][0]["actual"]["framebuffer_height"] += 1
    else: baseline["frames"][0]["image_camera"]["frame_id"] = "image-0001"
    write_truth(monkeypatch, capture_path, baseline_path, capture, baseline)
    path = tmp_path / "prediction.json"
    seal = d.freeze_predictions(path, table)
    with pytest.raises(ValueError, match="join|mismatch"):
        d.evaluate_frozen(path, seal, capture_path, baseline_path)


@pytest.mark.parametrize("fault", ["capture", "baseline"])
def test_metadata_pins_enforced_after_freeze(tmp_path, monkeypatch, fault):
    table = predictions(monkeypatch)
    capture_path, baseline_path, _, _ = scalar_truth(tmp_path, monkeypatch, table)
    path = tmp_path / "prediction.json"
    seal = d.freeze_predictions(path, table)
    (capture_path if fault == "capture" else baseline_path).write_bytes(b"{}")
    with pytest.raises(ValueError, match="pinned byte"):
        d.evaluate_frozen(path, seal, capture_path, baseline_path)


def test_quarter_turn_so3_pitch_focal_principal_and_deltas(tmp_path, monkeypatch):
    table = predictions(monkeypatch)
    camera = table["frames"][3]["camera"]
    camera.update(yaw_degrees=90., pitch_degrees=2., focal_length_pixels=[50., 55.], principal_point_pixels=[52., 37.])
    report = run_frozen(tmp_path, monkeypatch, table)
    row = report["frames"][3]
    assert row["current"]["rotation_error_degrees"] == pytest.approx(2., abs=1e-6)
    assert row["current"]["pitch_error_degrees"] == 2.
    assert row["current"]["vertical_focal_error_relative"] == pytest.approx(.1)
    assert row["principal_point_change_pixels"] == [2., -3.]
    assert row["principal_point_change_norm_pixels"] == pytest.approx(np.sqrt(13.))
    assert row["deltas"]["rotation_error_degrees"] == pytest.approx(2., abs=1e-6)
    assert report["joint_camera_success"]


@pytest.mark.parametrize("failed", [0, 3])
def test_fixed_primary_and_regression_interpretation(tmp_path, monkeypatch, failed):
    table = predictions(monkeypatch)
    table["frames"][failed]["camera"]["focal_length_pixels"][1] = 61.
    report = run_frozen(tmp_path, monkeypatch, table)
    assert not report["joint_camera_success"]
    assert report["decision"] == ("stop_mask_camera_recovery_hypothesis" if failed == 3 else "mixed_no_general_recovery_claim")


@pytest.mark.parametrize("fault", ["bool_angle", "extra_pixels", "ordinal", "scale", "missing", "bool_rule"])
def test_invalid_scalar_census_cannot_freeze(tmp_path, monkeypatch, fault):
    table = predictions(monkeypatch)
    if fault == "bool_angle": table["frames"][0]["camera"]["yaw_degrees"] = True
    elif fault == "extra_pixels": table["frames"][0]["pixels"] = [[1]]
    elif fault == "ordinal": table["frames"][0]["ordinal"] = False
    elif fault == "scale": table["frames"][0]["scale_resolved"] = True
    elif fault == "bool_rule": table["rule"]["adapter_calls"] = True
    else: table["frames"].pop()
    path = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        d.freeze_predictions(path, table)
    assert not path.exists()


def test_input_manifest_inventory_and_all_seals_checked_without_real_images(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "ROOT", tmp_path)
    rows = seals()
    paths = [tmp_path / row["path"] for row in rows]
    manifest_rows = [dict(row, absolute_path=str(path)) for row, path in zip(rows, paths)]
    manifest_path = tmp_path / "manifest.json"
    raw = d.encoded({"inputs": manifest_rows})
    manifest_path.write_bytes(raw)
    monkeypatch.setattr(d, "MANIFEST_SHA", d.digest(raw))
    original = d.pinned_bytes
    opened = []

    def virtual(path, size, sha, cap=d.MAX_JSON_BYTES):
        if Path(path) == manifest_path:
            return original(path, size, sha, cap)
        opened.append(Path(path))
        index = paths.index(Path(path))
        assert size == rows[index]["bytes"] and sha == rows[index]["sha256"]
        return b""  # No temporary image files; seals themselves are virtual here.

    original_stat = Path.stat

    def stat(path, *args, **kwargs):
        if path in paths:
            from types import SimpleNamespace
            return SimpleNamespace(st_size=rows[paths.index(path)]["bytes"])
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    monkeypatch.setattr(d, "pinned_bytes", virtual)
    assert d.sealed_input_preflight(manifest_path, paths) == rows
    assert opened == paths
    opened.clear()
    with pytest.raises(ValueError, match="path/order"):
        d.sealed_input_preflight(manifest_path, paths[::-1])
    assert not opened


def test_main_failed_freeze_stops_before_truth_parse(tmp_path, monkeypatch):
    table = predictions(monkeypatch)
    capture_path, baseline_path, _, _ = scalar_truth(tmp_path, monkeypatch, table)
    monkeypatch.setattr(d, "ROOT", tmp_path)
    monkeypatch.setattr(d, "MANIFEST_PATH", Path("manifest.json"))
    monkeypatch.setattr(d, "CAPTURE_PATH", Path(capture_path.name))
    monkeypatch.setattr(d, "BASELINE_PATH", Path(baseline_path.name))
    monkeypatch.setattr(d, "sealed_input_preflight", lambda *args: seals())
    monkeypatch.setattr(d, "decode_frames", lambda *args: arrays())

    def failed_freeze(*args):
        raise ValueError("simulated prediction freeze failure")

    def forbidden(*args):
        pytest.fail("evaluation entered after failed freeze")

    monkeypatch.setattr(d, "freeze_predictions", failed_freeze)
    monkeypatch.setattr(d, "evaluate_frozen", forbidden)
    monkeypatch.setattr(d.json, "loads", forbidden)
    (tmp_path / "new").mkdir()
    argv = [arg for i in range(6) for arg in ("--image", str(tmp_path / f"virtual-{i}.png"))]
    argv += ["--input-manifest", str(tmp_path / "manifest.json"), "--evaluation-capture", str(capture_path),
             "--baseline-report", str(baseline_path), "--prediction-output", str(tmp_path / "new" / "prediction.json"),
             "--report-output", str(tmp_path / "new" / "report.json"),
             "--source-closure-sha256", SOURCE_CLOSURE_SHA, "--runtime-closure-sha256", RUNTIME_CLOSURE_SHA]
    with pytest.raises(ValueError, match="simulated prediction freeze"):
        d.main(argv)
    assert not (tmp_path / "new" / "report.json").exists()


@pytest.mark.parametrize("case", ["existing_parent", "missing_parent", "same_output",
                                   "existing_output", "source_collision", "parent_is_file"])
def test_main_output_validation_matches_runner_without_payload_work(tmp_path, monkeypatch, case):
    monkeypatch.setattr(d, "ROOT", tmp_path)
    monkeypatch.setattr(d, "MANIFEST_PATH", Path("manifest.json"))
    monkeypatch.setattr(d, "CAPTURE_PATH", Path("synthetic-capture.json"))
    monkeypatch.setattr(d, "BASELINE_PATH", Path("synthetic-baseline.json"))
    parent = tmp_path / "outputs"
    if case == "parent_is_file":
        parent.write_text("{}", encoding="ascii")
    elif case != "missing_parent":
        parent.mkdir()
    prediction, report = parent / "prediction.json", parent / "report.json"
    if case == "same_output":
        report = prediction
    elif case == "existing_output":
        report.write_text("{}", encoding="ascii")
    elif case == "source_collision":
        prediction = tmp_path / "manifest.json"
        report = tmp_path / "report.json"
    reached = []

    def stop_at_preflight(*args):
        reached.append(args)
        raise RuntimeError("offline validation reached preflight")

    def forbidden(*args):
        pytest.fail("CLI validation test reached payload or camera work")

    monkeypatch.setattr(d, "sealed_input_preflight", stop_at_preflight)
    monkeypatch.setattr(d, "pinned_bytes", forbidden)
    monkeypatch.setattr(d, "decode_frames", forbidden)
    monkeypatch.setattr(d.image_only, "solve_image_cameras", forbidden)
    argv = [arg for i in range(6) for arg in ("--image", str(tmp_path / f"virtual-{i}.png"))]
    argv += ["--input-manifest", str(tmp_path / "manifest.json"),
             "--evaluation-capture", str(tmp_path / "synthetic-capture.json"),
             "--baseline-report", str(tmp_path / "synthetic-baseline.json"),
             "--prediction-output", str(prediction), "--report-output", str(report),
             "--source-closure-sha256", SOURCE_CLOSURE_SHA, "--runtime-closure-sha256", RUNTIME_CLOSURE_SHA]
    if case == "existing_parent":
        with pytest.raises(RuntimeError, match="offline validation reached preflight"):
            d.main(argv)
        assert len(reached) == 1
    else:
        with pytest.raises(ValueError, match="existing output parent.*distinct absent"):
            d.main(argv)
        assert not reached
    if case == "missing_parent":
        assert not parent.exists()
    assert not prediction.exists()


@pytest.mark.parametrize("fault", ["size", "sha"])
def test_seal_refusal_uses_same_scalar_bytes_as_snapshot(tmp_path, fault):
    path = tmp_path / "scalar.json"
    raw = d.encoded({"synthetic": 1})
    path.write_bytes(raw)
    size = len(raw) + (fault == "size")
    sha = d.digest(raw) if fault != "sha" else "F" * 64
    with pytest.raises(ValueError, match="pinned byte size/SHA"):
        d.pinned_bytes(path, size, sha)


@pytest.mark.parametrize("fault", ["order", "dtype", "dimension"])
def test_invalid_frame_mapping_rejected_before_production(monkeypatch, fault):
    frames = arrays()
    if fault == "order": frames = dict(reversed(list(frames.items())))
    elif fault == "dtype": frames[d.IDS[5]] = np.zeros((80, 100, 3), np.float32)
    else: frames[d.IDS[5]] = np.zeros((0, 100, 3), np.uint8)

    def forbidden(*args):
        pytest.fail("invalid six-view mapping reached production")

    monkeypatch.setattr(d.image_only, "solve_image_cameras", forbidden)
    with pytest.raises(ValueError):
        d.predict_cameras(frames, seals(), **CLOSURE_PINS)


def test_freeze_readback_failure_and_scalar_cap(tmp_path, monkeypatch):
    table = predictions(monkeypatch)
    original = d.pinned_bytes

    def corrupt(path, *args, **kwargs):
        Path(path).write_bytes(b"{}")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(d, "pinned_bytes", corrupt)
    with pytest.raises(ValueError, match="pinned byte"):
        d.freeze_predictions(tmp_path / "failed.json", table)
    with pytest.raises(ValueError, match="byte cap"):
        d.freeze_json(tmp_path / "large.json", {"scalar": "a" * d.MAX_JSON_BYTES})
    assert not (tmp_path / "large.json").exists()


def test_verified_closure_pins_normalize_freeze_and_reach_report(tmp_path, monkeypatch):
    camera = estimate()
    monkeypatch.setattr(d.image_only, "solve_image_cameras", lambda frames: CameraSolution({key: camera for key in d.IDS}))
    table = d.predict_cameras(arrays(), seals(), source_closure_sha256=SOURCE_CLOSURE_SHA.lower(),
                              runtime_closure_sha256=RUNTIME_CLOSURE_SHA.lower())
    capture_path, baseline_path, _, _ = scalar_truth(tmp_path, monkeypatch, table)
    path = tmp_path / "closure-predictions.json"
    freeze = d.freeze_predictions(path, table)
    frozen = json.loads(path.read_bytes())
    report = d.evaluate_frozen(path, freeze, capture_path, baseline_path)
    report_path = tmp_path / "closure-report.json"
    d.freeze_json(report_path, report)
    persisted_report = json.loads(report_path.read_bytes())
    for key, expected in CLOSURE_PINS.items():
        assert table[key] == frozen[key] == report[key] == persisted_report[key] == expected


@pytest.mark.parametrize("key", ["source_closure_sha256", "runtime_closure_sha256"])
@pytest.mark.parametrize("bad", [None, "", "A" * 63, "A" * 65, "G" * 64, True, 123])
def test_missing_or_malformed_closure_input_refused_before_production(monkeypatch, key, bad):
    pins = dict(CLOSURE_PINS)
    if bad is None:
        pins.pop(key)  # An omitted argument must fail just as an explicit null does.
    else:
        pins[key] = bad

    def forbidden(*args):
        pytest.fail("unbound closure provenance reached production")

    monkeypatch.setattr(d.image_only, "solve_image_cameras", forbidden)
    with pytest.raises(ValueError, match="closure SHA256"):
        d.predict_cameras(arrays(), seals(), **pins)


@pytest.mark.parametrize("key", ["source_closure_sha256", "runtime_closure_sha256"])
@pytest.mark.parametrize("bad", [None, "", "F" * 63, "Z" * 64, False])
def test_missing_or_malformed_frozen_closure_refuses_before_truth(tmp_path, monkeypatch, key, bad):
    table = predictions(monkeypatch)
    if bad is None:
        table.pop(key)
    else:
        table[key] = bad
    path = tmp_path / "invalid-closure.json"
    with pytest.raises(ValueError):
        d.freeze_predictions(path, table)
    assert not path.exists()
    # A file/hash supplied by another producer still must satisfy the strict
    # frozen schema before this evaluator accesses any evaluation metadata.
    freeze = d.freeze_json(path, table)
    original = d.pinned_bytes

    def watched(source, *args, **kwargs):
        assert Path(source) == path, "truth accessed with missing/malformed closure provenance"
        return original(source, *args, **kwargs)

    monkeypatch.setattr(d, "pinned_bytes", watched)
    with pytest.raises(ValueError):
        d.evaluate_frozen(path, freeze, tmp_path / "absent-capture", tmp_path / "absent-baseline")
