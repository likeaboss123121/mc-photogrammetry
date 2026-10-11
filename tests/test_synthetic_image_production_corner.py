"""Generated-only fixture correctness and one unmocked production invocation.

A passing harness records a production abstention honestly; it does not imply
that reconstruction succeeded. No production return or camera is patched.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mc_photogrammetry.project.store import load_project
from mc_photogrammetry.domain.models import BlockVolume
from mc_photogrammetry.reconstruction.calibration import CalibrationResult


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluation/probe_synthetic_image_production_corner.py"
spec = importlib.util.spec_from_file_location("synthetic_corner_probe", SCRIPT)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


def test_renderer_nearest_face_keeps_hidden_depth_out_of_pixels():
    # Expected world intersections stay exclusively in test assertions.
    eye = np.array([-3.0, 2.0, -3.0])
    directions = np.array([[1.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 2.0]])
    points, faces = subject.trace_corner(eye, directions)
    np.testing.assert_allclose(points[0], (0.0, 2.0, 0.0))
    np.testing.assert_allclose(points[2], (0.0, 2.0, 3.0))
    assert faces.tolist() == [0, -1, 0]
    # Neither the far back face nor the box interior is the visible intersection.
    assert tuple(points[0]) != (6.0, 2.0, 6.0)


def test_frozen_renderer_is_deterministic_and_views_are_distinct():
    first = subject.render_view(0)
    np.testing.assert_array_equal(first, subject.render_view(0))
    assert first.shape == (480, 720, 3) and first.dtype == np.uint8
    assert len(np.unique(first.reshape(-1, 3), axis=0)) > 50
    assert not np.array_equal(first, subject.render_view(1))


@pytest.mark.parametrize("index", [-1, 3, True])
def test_no_adaptive_or_extra_views(index):
    with pytest.raises(ValueError):
        subject.render_view(index)


def test_only_generated_paths_cross_the_production_call_boundary():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "run_probe")
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and node.func.id == "derive_image_output_from_paths"]
    assert len(calls) == 1
    assert [ast.unparse(value) for value in calls[0].args] == ["paths"]
    assert [(value.arg, ast.unparse(value.value)) for value in calls[0].keywords] == [
        ("project_path", "project_path")]
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr in ("setattr", "patch") for node in ast.walk(tree))


def test_generated_camera_fixture_has_adequate_default_detector_segments(tmp_path):
    paths = subject.write_fixture(tmp_path / "revised-inputs")
    report = subject.detector_counts(paths)
    assert report["production_invocations"] == 0
    assert report["row_arrays_persisted"] is False
    assert report["settings"] == {"scales": [1.0, 0.5], "minimum_length": 40.0,
                                   "camera_gate": 12}
    assert [row["label"] for row in report["rows"]] == ["view-00.png", "view-01.png", "view-02.png"]
    assert all(row["retained_segments"] >= 12 and row["meets_camera_gate"]
               for row in report["rows"])
    assert all(row["minimum_retained_length"] >= 40 for row in report["rows"])
    payload = dict(report)
    digest = payload.pop("report_sha256")
    assert subject.canonical_digest(payload) == digest
    for path, row in zip(paths, report["rows"]):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["encoded_sha256"]
    # Persist this diagnostic under the caller's fresh synthetic-only pytest root.
    (tmp_path / "detector-counts.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="ascii")


def _projection_fixture(calibrations, *, frame_ids=("image-0000", "image-0001")):
    ids = ("image-0000", "image-0001")
    session = SimpleNamespace(
        version=1, frames=tuple(SimpleNamespace(frame_id=name) for name in ids),
        input_frames=tuple(SimpleNamespace(frame_id=name) for name in ids),
        cameras={}, scale_known=False)
    lattice = SimpleNamespace(
        frame_ids=frame_ids, camera_calibrations=calibrations,
        status="unavailable", reason="no_lattice_reading", detail="no accepted reading",
        terminal="geometry_unavailable", readings=(), abstentions=())
    outcome = SimpleNamespace(
        entry=SimpleNamespace(session=session, graph=None, pose_graph=None),
        lattice=lattice, geometry=None, output=None, result_sha256="9" * 64)
    record = {"stage": "lattice", "status": "geometry_unavailable",
              "reason": "lattice_no_lattice_reading", "detail": "no accepted reading",
              **{name: str(index) * 64 for index, name in enumerate((
                  "entry_sha256", "session_sha256", "lattice_sha256",
                  "geometry_sha256", "output_sha256"), start=1)}}
    project = SimpleNamespace(image_derived_production=record,
                              volume=BlockVolume(), source_paths=["a.png", "b.png"])
    return outcome, project


def _calibration(segments=20, inliers=15, counts=(7, 5, 3)):
    return CalibrationResult(focal_pixels=320., yaw_degrees=30., pitch_degrees=20.,
                             width=720, height=480, segments=segments, inliers=inliers,
                             median_residual=.5, per_axis_inliers=counts)


def test_populated_calibrations_join_values_to_frame_ids_without_unpacking_records():
    outcome, project = _projection_fixture((_calibration(), _calibration(30, 18, (8, 6, 4))))
    summary = subject.summarize_safely(outcome, project)
    assert summary["diagnostic_status"] == "complete"
    assert summary["lattice"]["calibration_accounting_available"] is True
    assert summary["lattice"]["calibrations"] == [
        {"frame_id": "image-0000", "segments": 20, "inliers": 15, "axis_inliers": [7, 5, 3]},
        {"frame_id": "image-0001", "segments": 30, "inliers": 18, "axis_inliers": [8, 6, 4]}]


def test_empty_calibrations_are_unavailable_measurements_not_zero_counts():
    outcome, project = _projection_fixture(())
    summary = subject.summarize_safely(outcome, project)
    assert summary["diagnostic_status"] == "complete"
    assert summary["lattice"]["calibration_accounting_available"] is False
    assert summary["lattice"]["calibrations"] == []
    assert summary["reason"] == "lattice_no_lattice_reading"


@pytest.mark.parametrize("kind", ["short", "long", "foreign", "missing", "reordered", "wrong-frame", "counters"])
def test_invalid_projection_preserves_persisted_terminal_and_digest_fields(kind):
    calibrations = (_calibration(), _calibration())
    frame_ids = ("image-0000", "image-0001")
    if kind == "short":
        calibrations = calibrations[:1]
    elif kind == "long":
        calibrations = (*calibrations, _calibration())
    elif kind == "foreign":
        calibrations = (object(), calibrations[1])
    elif kind == "missing":
        calibrations = None
    elif kind == "reordered":
        frame_ids = tuple(reversed(frame_ids))
    elif kind == "wrong-frame":
        frame_ids = ("image-0000", "foreign")
    else:
        calibrations = (_calibration(counts=(7, 5, 2)), calibrations[1])
    outcome, project = _projection_fixture(calibrations, frame_ids=frame_ids)
    summary = subject.summarize_safely(outcome, project)
    for name in ("stage", "status", "reason", "detail", "entry_sha256", "session_sha256",
                 "lattice_sha256", "geometry_sha256", "output_sha256"):
        assert summary[name] == project.image_derived_production[name]
    assert summary["diagnostic_status"] == "incomplete"
    assert summary["diagnostic_failure"] == {"reason": "diagnostic_projection_failed",
                                              "error_type": "ValueError"}
    assert summary["block_count"] == 0
    assert summary["live_accounting_available"] is False
    assert "lattice" not in summary
    assert "invalid calibration" not in json.dumps(summary)
    payload = dict(summary)
    digest = subject.canonical_digest(payload)
    payload["reason"] = "tampered"
    assert subject.canonical_digest(payload) != digest


def test_projection_failure_redacts_exception_text_and_bounds_persisted_detail():
    class BadCamera:
        @property
        def intrinsics(self):
            raise RuntimeError("sensitive/path/" + "payload" * 1000)

    outcome, project = _projection_fixture(())
    outcome.entry.session.cameras = {"image-0000": BadCamera()}
    project.image_derived_production["detail"] = "x" * 1000
    summary = subject.summarize_safely(outcome, project)
    assert summary["diagnostic_failure"]["error_type"] == "RuntimeError"
    assert summary["detail"] == "x" * 512 and summary["detail_truncated"] is True
    assert summary["stage"] == project.image_derived_production["stage"]
    assert "sensitive" not in json.dumps(summary) and "payload" not in json.dumps(summary)


@pytest.fixture(scope="module")
def real_probe(tmp_path_factory):
    destination = tmp_path_factory.mktemp("corner-parent") / "probe"
    return destination, subject.run_probe(destination)


def test_one_unmocked_production_result_retains_original_inventory_and_honest_terminal(real_probe):
    destination, report = real_probe
    assert report["production_invocations"] == 1
    assert report["production_defaults_unchanged"] is True
    terminal = report["terminal"]
    assert terminal["status"] not in ("exception", "unpersisted_refusal")
    assert terminal["stage"] in ("entry", "lattice", "geometry", "output")
    project = load_project(destination / "corner.mcscan")
    assert project.schema_version == 3
    record = project.image_derived_production
    for key in ("stage", "status", "reason", "detail", "entry_sha256", "session_sha256",
                "lattice_sha256", "geometry_sha256", "output_sha256"):
        assert terminal[key] == record[key]
    assert terminal["source_labels"] == project.source_paths
    assert len(project.source_paths) == 3
    if terminal["live_accounting_available"]:
        assert terminal["input_frame_ids"] == ["image-0000", "image-0001", "image-0002"]
        assert [row["frame_id"] for row in terminal["camera_admissions"]] == terminal["input_frame_ids"]
        admitted = [row["frame_id"] for row in terminal["camera_admissions"] if row["status"] == "admitted"]
        assert terminal["active_frame_ids"] == admitted
        assert terminal["scale_known"] is False
        assert all(row["position_resolved"] is False for row in terminal["cameras"])
    if terminal["status"] != "promoted":
        assert terminal["block_count"] == 0 and not project.volume.blocks
        assert terminal["observation_kinds"] == []
    else:
        # No expected coordinate injection: all emitted states must be promoted
        # source-bound evidence, not a renderer-generated block set or hidden fill.
        assert terminal["stage"] == "output" and terminal["block_count"] > 0
        assert terminal["observation_kinds"] == ["observed"]
        assert all(row.frame_ids for _, row in project.volume.items())
    assert terminal["litematic_created"] is False
    assert not list(destination.rglob("*.litematic"))


def test_report_digest_and_encoded_input_pins_are_independent_of_renderer_truth(real_probe):
    destination, report = real_probe
    persisted = json.loads((destination / "report.json").read_text(encoding="ascii"))
    assert persisted == report
    payload = dict(report)
    digest = payload.pop("report_sha256")
    assert subject.canonical_digest(payload) == digest
    assert [row["ordinal"] for row in report["inputs"]] == [0, 1, 2]
    for row in report["inputs"]:
        data = (destination / "inputs" / row["label"]).read_bytes()
        assert row["byte_length"] == len(data)
        assert row["encoded_sha256"] == hashlib.sha256(data).hexdigest()
    assert all(value is False for value in report["claims"].values())
    assert not any(name in json.dumps(report) for name in (
        "expected_positions", "ground_truth", "manifest", "renderer_camera", "BOX"))
