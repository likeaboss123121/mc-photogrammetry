"""Offline producer parity, bounded basin selection and deferred-truth controls."""
from dataclasses import asdict
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.evaluation import camera_basin_availability as d


def test_coarse_census_first_tie_best_focal_and_finite_cap(monkeypatch):
    census = d.CoarseCensus()
    census.remember(3., 50., 0., 0.)
    census.remember(3., 60., 0., 0.)
    assert census.rows[(0., 0.)] == (3., 50., 0., 0., 0)
    census.remember(4., 70., 0., 0.)
    assert census.rows[(0., 0.)] == (4., 70., 0., 0., 2)
    with pytest.raises(ValueError, match="nonfinite"):
        census.remember(float("nan"), 1., 0., 0.)
    census.calls = d.COARSE_CALLS
    with pytest.raises(ValueError, match="call cap"):
        census.remember(5., 80., 0., 0.)
    monkeypatch.setattr(d, "MAX_ORIENTATIONS", 1)
    census.calls = 0
    with pytest.raises(ValueError, match="orientation cap"):
        census.remember(5., 80., 2., 0.)


def test_basins_greedy_objective_distinctness_quarter_gauge_and_K():
    rows = [(100., 70., 0., 0., 0), (99., 90., 88., 0., 1),
            (98., 100., 0., 20., 2), (97., 100., 0., -20., 3),
            (96., 100., 30., 0., 4), (95., 100., 60., 0., 5)]
    selected = d.select_basins(rows[::-1])
    assert [row[4] for row in selected] == [0, 2, 3, 4]
    assert d.angular_distance(d.rotation(1., 2.), d.rotation(91., 2.)) < 1e-5
    assert all(d.angular_distance(d.rotation(*a[2:4]), d.rotation(*b[2:4])) > 10.
               for i, a in enumerate(selected) for b in selected[i+1:])


def test_basin_exact_ties_preserve_source_enumeration():
    rows = [(5., 80., 0., 0., 6), (5., 90., 0., 25., 2)]
    assert d.select_basins(rows)[0][4] == 2


@pytest.mark.parametrize("bad", [(), (1., 1., 0., 0., True),
                                  (1., 1., 0., float("inf"), 8),
                                  (1., 0., 0., 0., 8), (1., 1., 90., 0., 8)])
def test_all_coarse_rows_validate_even_beyond_K(bad):
    valid = [(10.-i, 70., 0., float(20*i-40), i) for i in range(4)]
    with pytest.raises(ValueError, match="invalid coarse"):
        d.select_basins(valid + [bad])


def test_source_bound_baseline_and_local_refinement_are_exact(monkeypatch):
    # Reuse independent world-to-pixel projection, never fixture screenshots.
    spec = importlib.util.spec_from_file_location("basin_projector", Path(__file__).with_name("test_calibration_independent.py"))
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    segments = fixture._segments((.5, .5, .5), 23., -14., 0., 70., 1280, 720)
    grey = np.zeros((720, 1280), np.uint8)
    original_detector = d.c.detect_segments
    census = d.CoarseCensus()
    calls = []
    def detector(*args, **kwargs):
        calls.append((args[0].shape, kwargs))
        return segments
    baseline, refine = d.source_functions(census, detector)
    expected = d.c.estimate_camera(grey, segments=segments)
    actual = baseline(grey)
    assert asdict(actual) == asdict(expected)
    assert len(calls) == 1 and calls[0][1] == {"scales": d.c.DETECTION_SCALES, "minimum_length": d.c.MINIMUM_LENGTH}
    assert census.calls == d.COARSE_CALLS and len(census.rows) == d.MAX_ORIENTATIONS
    seed = d.select_basins(list(census.rows.values()))[0]
    assert asdict(refine(grey, segments, seed)) == asdict(expected)
    assert len(calls) == 1 and d.c.detect_segments is original_detector
    assert hashlib.sha256(Path(d.c.__file__).read_bytes()).hexdigest().upper() == d.SOURCE_SHA


def test_source_drift_refuses_clone(monkeypatch):
    monkeypatch.setattr(d, "SOURCE_SHA", "0"*64)
    with pytest.raises(ValueError, match="source pin"):
        d.source_functions(d.CoarseCensus(), lambda *a, **k: None)


def table():
    candidate = dict(focal_pixels=50., yaw_degrees=0., pitch_degrees=0.,
                     width=100, height=80, inliers=12, segments=12, median_residual=0.,
                     per_axis_inliers=[4,4,4], roll_degrees=0., principal_point=None,
                     notes=[], rank=0, incumbent=True, objective_score=3.,
                     coarse_seed=dict(score=3., focal=50., yaw=0., pitch=0., enumeration=0))
    frames = [dict(frame_id=f"image-{i:04d}", shape=[80,100,3], status="complete", reason="",
                   detector_passes=1, coarse_passes=1, coarse_calls=d.COARSE_CALLS,
                   orientation_rows=d.MAX_ORIENTATIONS, refinements=1,
                   candidates=[copy.deepcopy(candidate)], seconds=1.) for i in range(6)]
    return dict(schema="mc-camera-basin-candidates-v1", calibration_source_sha256=d.SOURCE_SHA,
                rule=copy.deepcopy(d.CANDIDATE_RULE), frames=frames,
                input_seals=[dict(path=f"fake-{i}.png", bytes=10, sha256=str(i)*64) for i in range(6)])


def test_six_frame_candidate_freeze_readback_and_truth_order(tmp_path, monkeypatch):
    data = table()
    d.validate_table(data)
    path = tmp_path/"candidate.json"
    seal = d.freeze_candidates(path, data)
    assert seal == hashlib.sha256(d.encoded(data)).hexdigest().upper()
    with pytest.raises(FileExistsError):
        d.freeze_candidates(path, data)
    capture, prior = {"frames": [], "screenshots": []}, {"frames": []}
    for frame, source in zip(data["frames"], data["input_seals"]):
        actual = dict(eye_position=[0,0,0], yaw_degrees=0., pitch_degrees=0.,
                      fov_degrees=float(np.degrees(2*np.arctan(40/50))),
                      framebuffer_width=100, framebuffer_height=80, roll_degrees=0.,
                      screenshot_sha256=source["sha256"], screenshot_bytes=10)
        capture["frames"].append(dict(filename=source["path"], actual=actual))
        capture["screenshots"].append(dict(filename=source["path"], sha256=source["sha256"]))
        prior["frames"].append(dict(frame_id=frame["frame_id"], image_camera=dict(status="available",
            yaw_degrees=0., pitch_degrees=0., roll_degrees=0., focal_length_pixels=[50.,50.], principal_point_pixels=[50.,40.])))
    truth = tmp_path/"capture.json"
    old = tmp_path/"baseline.json"
    truth.write_bytes(d.encoded(capture))
    old.write_bytes(d.encoded(prior))
    real_snapshot = d.snapshot
    reads = []
    def watched(p, cap=d.MAX_JSON_BYTES):
        reads.append(Path(p))
        if Path(p) in (truth, old):
            assert reads[0] == path and hashlib.sha256(path.read_bytes()).hexdigest().upper() == seal
        return real_snapshot(p, cap)
    monkeypatch.setattr(d, "snapshot", watched)
    result = d.evaluate_frozen(path, seal, truth, old)
    assert reads == [path, truth, old]
    assert result["image0003_candidate_available"] and result["other_five_incumbents_preserved"]
    assert result["decision"] == "availability_only_requires_image_only_selector"
    assert result["camera_selected_by_truth"] is False and result["state_match"] is None
    assert result["evaluation_capture_sha256"] == hashlib.sha256(truth.read_bytes()).hexdigest().upper()
    path.write_bytes(path.read_bytes()+b" ")
    reads.clear()
    with pytest.raises(ValueError, match="digest mismatch"):
        d.evaluate_frozen(path, seal, truth, old)
    assert reads == [path]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "seventh", "bad_census", "bad_count", "bool", "bad_pin", "nonfinite"])
def test_incomplete_table_refuses_before_truth(tmp_path, mutation):
    data = table()
    if mutation == "missing": data["frames"].pop()
    elif mutation == "duplicate": data["frames"][1]["frame_id"] = "image-0000"
    elif mutation == "seventh": data["frames"].append(data["frames"][0])
    elif mutation == "bad_census": data["frames"][0]["coarse_calls"] -= 1
    elif mutation == "bad_count": data["frames"][0]["candidates"][0]["per_axis_inliers"] = [1,1,1]
    elif mutation == "bool": data["frames"][0]["candidates"][0]["rank"] = False
    elif mutation == "bad_pin": data["calibration_source_sha256"] = "0"*64
    else: data["frames"][0]["candidates"][0]["focal_pixels"] = float("inf")
    with pytest.raises(ValueError): d.validate_table(data)
    if mutation != "nonfinite":
        path = tmp_path/"table.json"
        seal = d.freeze_candidates(path, data)
        with pytest.raises(ValueError):
            d.evaluate_frozen(path, seal, tmp_path/"absent-truth", tmp_path/"absent-baseline")


def test_abstention_and_table_caps(tmp_path, monkeypatch):
    monkeypatch.setattr(d.c, "detect_segments", lambda *a, **k: np.empty((0,4)))
    row = d.frame_candidates("image-0000", np.zeros((80,100,3), np.uint8))
    assert row["status"] == "unavailable" and row["detector_passes"] == 1 and row["coarse_calls"] == 0
    data = table()
    data["frames"][0] = row
    d.validate_table(data)
    with pytest.raises(ValueError, match="byte cap"):
        d.freeze_candidates(tmp_path/"oversize", {"x": "a"*d.MAX_JSON_BYTES})
    assert not (tmp_path/"oversize").exists()


def test_direct_cli_help_bootstrap_without_truth_or_pixels(tmp_path):
    result = subprocess.run([sys.executable, "-B", str(Path(d.__file__)), "--help"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--candidate-output" in result.stdout


def fake_inputs(tmp_path, monkeypatch):
    rows, paths = [], []
    monkeypatch.setattr(d, "ROOT", tmp_path)
    for index in range(6):
        path = tmp_path/f"fake-{index}.png"
        raw = f"not an image {index}".encode()
        path.write_bytes(raw)
        paths.append(str(path))
        rows.append(dict(frame=f"image-{index:04d}", ordinal=index,
                         path=path.name, absolute_path=str(path), bytes=len(raw),
                         sha256=hashlib.sha256(raw).hexdigest().upper()))
    manifest = tmp_path/"seals.json"
    manifest.write_bytes(d.encoded(dict(inputs=rows)))
    monkeypatch.setattr(d, "INPUT_MANIFEST_SHA", hashlib.sha256(manifest.read_bytes()).hexdigest().upper())
    return manifest, paths


def test_preflight_all_six_seals_before_any_decode_and_after_run(tmp_path, monkeypatch):
    manifest, paths = fake_inputs(tmp_path, monkeypatch)
    seals, digest = d.sealed_input_preflight(manifest, paths)
    assert digest == d.INPUT_MANIFEST_SHA and len(seals) == 6
    Path(paths[5]).write_bytes(b"not an image X")  # Equal size, wrong hash.
    with pytest.raises(ValueError, match="seal mismatch"):
        d.sealed_input_preflight(manifest, paths)
    with pytest.raises(ValueError, match="seal mismatch"):
        d.verified_input_bytes(paths[5], seals[5])


@pytest.mark.parametrize("fault", ["order", "size", "pin", "foreign"])
def test_preflight_inventory_refuses_before_opening_any_image(tmp_path, monkeypatch, fault):
    manifest, paths = fake_inputs(tmp_path, monkeypatch)
    if fault == "order": paths[0], paths[1] = paths[1], paths[0]
    elif fault == "size": Path(paths[5]).write_bytes(b"short")
    elif fault == "pin": monkeypatch.setattr(d, "INPUT_MANIFEST_SHA", "0"*64)
    else: paths[5] = str(tmp_path/"absent.png")
    reads = []
    original = Path.read_bytes
    def watched(path):
        reads.append(path)
        if path.suffix == ".png": raise AssertionError("image opened before full inventory check")
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", watched)
    with pytest.raises(ValueError): d.sealed_input_preflight(manifest, paths)
    assert reads == [manifest]


def test_main_no_candidate_work_or_truth_read_on_bad_source_hash(tmp_path, monkeypatch):
    manifest, paths = fake_inputs(tmp_path, monkeypatch)
    Path(paths[3]).write_bytes(b"not an image X")
    def forbidden(*a, **k): raise AssertionError("candidate or truth called before preflight")
    monkeypatch.setattr(d, "frame_candidates", forbidden)
    monkeypatch.setattr(d, "evaluate_frozen", forbidden)
    argv = [arg for path in paths for arg in ("--image", path)] + [
        "--input-manifest", str(manifest), "--evaluation-capture", str(tmp_path/"absent"),
        "--baseline-report", str(tmp_path/"absent2"), "--candidate-output", str(tmp_path/"candidate"),
        "--report-output", str(tmp_path/"report")]
    with pytest.raises(ValueError, match="seal mismatch"): d.main(argv)
    assert not (tmp_path/"candidate").exists() and not (tmp_path/"report").exists()


def test_alternative_seed_rescores_all_segments_before_fine_search(monkeypatch):
    segments = np.tile([[10.,20.,100.,30.]], (301,1))
    calls = []
    def score(rows, weights, focal, yaw, pitch, *args):
        if len(calls) < 2: calls.append((len(rows), focal, yaw, pitch))
        return float(len(rows))
    monkeypatch.setattr(d.c, "_score", score)
    _, refine = d.source_functions(d.CoarseCensus(), lambda *a, **k: segments)
    seed = (300.,70.,12.,14.,0)
    result = refine(np.zeros((80,100),np.uint8), segments, seed)
    assert result is not None
    assert calls[0] == (301,70.,12.,14.)
    assert calls[1][0] == 301 and calls[1][2:4] == (10.,12.)


@pytest.mark.parametrize("scalar", [np.float32, np.float64])
def test_numpy_principal_point_detaches_validates_and_freezes(tmp_path, scalar):
    point = (scalar(51.25), scalar(39.75))
    found = d.c.CalibrationResult(50., 0., 0., 100, 80, 12, 12, 0.,
                                  per_axis_inliers=(4,4,4), principal_point=point)
    segments = np.tile([[10.,20.,90.,30.]], (12,1))
    row = d.candidate_row(found, (3.,50.,0.,0.,0), segments, 0)
    assert row["principal_point"] == (51.25,39.75)
    assert all(type(value) is float for value in row["principal_point"])
    assert all(type(value) is scalar for value in found.principal_point)
    data = table()
    data["frames"][0]["candidates"] = [row]
    d.validate_table(data)
    path = tmp_path/"numpy-principal.json"
    digest = d.freeze_candidates(path, data)
    frozen, readback = d.snapshot(path)
    assert digest == readback
    assert frozen["frames"][0]["candidates"][0]["principal_point"] == [51.25,39.75]
    d.validate_table(frozen)


@pytest.mark.parametrize("point", [(np.float32("nan"),np.float64(1.)),
                                   (np.float64("inf"),np.float32(1.)),
                                   (np.bool_(True),np.float64(1.))])
def test_principal_point_normalization_refuses_nonfinite_and_bool(point):
    found = d.c.CalibrationResult(50.,0.,0.,100,80,12,12,0.,principal_point=point)
    with pytest.raises(ValueError, match="principal point"):
        d.candidate_row(found,(3.,50.,0.,0.,0),np.tile([[10.,20.,90.,30.]],(12,1)),0)
