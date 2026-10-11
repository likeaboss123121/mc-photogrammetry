"""Focused tests for the DS-R5F solver-ablation utility (no image workload)."""
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluation/replay_graph_solver_ablation.py"
SPEC = importlib.util.spec_from_file_location("ds_r5f_ablation", SCRIPT)
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)

from mc_photogrammetry.reconstruction import plane_graph  # noqa: E402


def _match(*, left="image-0000", right="image-0002", left_plane=0, right_plane=0,
           shift=(0, 0), score=0.5):
    return plane_graph.PlaneMatch(
        left=left, right=right, left_axis=0, right_axis=0, left_b=1.0, right_b=1.0,
        quadrant_of_right=0, psr=2.0, peak=1.0, second_peak=0.5, overlap_cells=12,
        integer_residual=None, score=score, left_plane=left_plane,
        right_plane=right_plane, cell_shift_rc=shift)


def _matches():
    return (
        _match(left_plane=0, right_plane=1, shift=(32, 0)),
        _match(left_plane=1, right_plane=2, shift=(0, -32)),
        _match(left_plane=3, right_plane=4, shift=(5, 5)),
        _match(left_plane=0, right_plane=5, shift=(31, 0)),
    )


@dataclass
class FakeAlternative:
    axis_observations: tuple


class FakeSolution:
    def __init__(self, reason="", resultants=None, residuals=None,
                 observations=(4, 3, 2)):
        self.reason = reason
        self.axis_resultants = ({0: 0.1, 1: 0.2, 2: 0.3}
                                if resultants is None else resultants)
        self.axis_residuals = ({0: 0.01, 1: 0.02, 2: 0.03}
                               if residuals is None else residuals)
        self.alternatives = (FakeAlternative(observations),)


def _solver(calls):
    def solve(readings, matches, **kwargs):
        calls.append((readings, tuple(matches), dict(kwargs)))
        return FakeSolution()
    return solve


def test_boundary_and_conflict_selection_is_exact():
    matches = _matches()
    assert subject.boundary_match_indices(matches, 32) == (0, 1)
    assert subject.alias_conflict_match_indices(
        matches, [["image-0000", 0], ["image-0002", 4]]) == (0, 2, 3)
    index_sets = subject.ablation_index_sets(
        matches, [["image-0000", 0], ["image-0002", 4]], 32)
    assert index_sets["baseline"] == ()
    assert index_sets["boundary"] == (0, 1)
    assert index_sets["alias_conflict"] == (0, 2, 3)
    assert index_sets["union"] == (0, 1, 2, 3)


def test_malformed_shift_is_refused():
    class Malformed:
        cell_shift_rc = (32.0, 0)

    with pytest.raises(ValueError, match="malformed cell_shift_rc"):
        subject.boundary_match_indices((Malformed(),), 32)


def test_run_ablations_reuses_readings_and_constraints():
    matches = _matches()
    calls = []
    kwargs = {"integer_tolerance": 0.15, "min_resultant": 0.9,
              "ambiguous_margin": 0.02,
              "quadrant_constraints": (("image-0000", "image-0002", 1),)}
    records = subject.run_ablations(
        readings=("r0", "r1"), matches=matches, solver_kwargs=kwargs,
        conflicting_endpoints=[["image-0000", 0], ["image-0002", 4]],
        solver=_solver(calls), max_shift_cells=32,
        identity=lambda item: [item.left, item.left_plane, item.right,
                               item.right_plane])
    assert [row["label"] for row in records] == list(subject.ABLATION_LABELS)
    assert len(calls) == 4
    assert [len(row[1]) for row in calls] == [4, 2, 1, 0]
    assert [row["removed_match_count"] for row in records] == [0, 2, 3, 4]
    assert [row["retained_match_count"] for row in records] == [4, 2, 1, 0]
    for readings, matches_arg, seen_kwargs in calls:
        assert readings == ("r0", "r1")
        assert seen_kwargs == kwargs
        assert all(isinstance(item, plane_graph.PlaneMatch) for item in matches_arg)
    assert records[0]["removed_match_identities"] == []
    assert records[1]["removed_match_identities"] == [
        ["image-0000", 0, "image-0002", 1],
        ["image-0000", 1, "image-0002", 2]]


def test_run_ablations_can_reuse_the_captured_baseline_solution():
    matches = _matches()
    calls = []
    baseline = FakeSolution(reason="plane_integer_residual")
    records = subject.run_ablations(
        readings=("r0",), matches=matches, solver_kwargs={},
        conflicting_endpoints=[["image-0000", 0]], solver=_solver(calls),
        baseline_solution=baseline, identity=lambda item: [item.left])
    assert len(calls) == 3
    assert records[0]["reason"] == "plane_integer_residual"
    assert records[0]["axes"] == [[0, 0.1, 0.01, 4], [1, 0.2, 0.02, 3],
                                  [2, 0.3, 0.03, 2]]


def test_ablation_record_reports_absent_axis_diagnostics():
    record = subject.ablation_record(
        "boundary", (_match(),), (0,), FakeSolution(reason="no_matches"),
        identity=lambda item: [item.left])
    assert record["reason"] == "no_matches"
    assert record["axes"] == [[0, 0.1, 0.01, 4], [1, 0.2, 0.02, 3],
                              [2, 0.3, 0.03, 2]]


def _artifact_and_evidence():
    trace = {
        "graph_result_sha256": "a" * 64,
        "reason": "plane_integer_residual",
        "candidate_counts": {"pre_admission": 7476, "post_admission": 1495,
                             "post_same_axis": 54},
        "solver_failure": {
            "reason": "plane_integer_residual",
            "axes": [[0, 0.39, 0.17, 12], [1, 0.24, 0.20, 6], [2, 0.24, 0.18, 10]],
            "active_edge_aliases": {
                "input_match_count": 54,
                "input_match_identity_sha256": "b" * 64},
            "boundary_saturation": {
                "input_match_identity_sha256": "c" * 64,
                "kept_boundary_counts": [1, 1, 0],
                "max_shift_cells": 32}},
        "attribution_sha256": "d" * 64,
    }
    artifact = {
        "artifact_sha256": "e" * 64,
        "graph_sha256": "a" * 64,
        "project_sha256": "f" * 64,
        "inputs": [{"frame_id": "image-0000", "path": "p0", "sha256": "0" * 64}],
        "graph": {"result_sha256": "a" * 64, "payload": {
            "session_sha256": "1" * 64, "lattice_sha256": "2" * 64,
            "signatures_sha256": "3" * 64, "signature_digests": ["4" * 64],
            "reason": "plane_integer_residual"}},
        "trace_sha256": "d" * 64,
        "trace": trace,
    }
    evidence = {
        "graph_sha256": "a" * 64,
        "trace": trace,
        "trace_sha256": "d" * 64,
        "session_sha256": "1" * 64,
        "lattice_sha256": "2" * 64,
        "signatures_sha256": "3" * 64,
        "signature_digests": ["4" * 64],
        "candidate_counts": trace["candidate_counts"],
        "reason": "plane_integer_residual",
        "axes": [[0, 0.39, 0.17, 12], [1, 0.24, 0.20, 6], [2, 0.24, 0.18, 10]],
        "selected_match_count": 54,
        "alias_identity_sha256": "b" * 64,
        "boundary_identity_sha256": "c" * 64,
        "boundary_kept_counts": [1, 1, 0],
        "max_shift_cells": 32,
        "inputs": artifact["inputs"],
        "artifact_sha256": "e" * 64,
        "project_sha256": "f" * 64,
        "project_session_sha256": "1" * 64,
    }
    return artifact, evidence


def test_validate_replay_passes_and_fails_closed():
    artifact, evidence = _artifact_and_evidence()
    subject.validate_replay(artifact, evidence)

    tampered = dict(evidence)
    tampered["trace"] = dict(evidence["trace"])
    tampered["trace"]["reason"] = "no_matches"
    with pytest.raises(subject.ReplayMismatch) as error:
        subject.validate_replay(artifact, tampered)
    assert error.value.field == "trace.record"

    tampered = dict(evidence)
    tampered["axes"] = [[0, 0.0, 0.0, 1], [1, 0.0, 0.0, 1], [2, 0.0, 0.0, 1]]
    with pytest.raises(subject.ReplayMismatch) as error:
        subject.validate_replay(artifact, tampered)
    assert error.value.field == "solver_failure.axes"

    tampered = dict(evidence)
    tampered["graph_sha256"] = "9" * 64
    with pytest.raises(subject.ReplayMismatch):
        subject.validate_replay(artifact, tampered)

    tampered = dict(evidence)
    tampered["selected_match_count"] = 53
    with pytest.raises(subject.ReplayMismatch) as error:
        subject.validate_replay(artifact, tampered)
    assert error.value.field == "selected_match_count"


def test_seal_replays_and_write_new_refuses_overwrite(tmp_path):
    payload = subject.seal({"schema": subject.SCHEMA, "value": [1, 2, 3]})
    subject.verify_seal(payload)
    assert payload["artifact_sha256"] == hashlib.sha256(
        subject.canonical_json({"schema": subject.SCHEMA, "value": [1, 2, 3]})
    ).hexdigest()

    target = tmp_path / "out.json"
    subject.write_new(target, payload)
    first = target.read_bytes()
    with pytest.raises(FileExistsError):
        subject.write_new(target, subject.seal({"schema": "other"}))
    assert target.read_bytes() == first
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    assert not list(tmp_path.glob("*.partial"))


def test_refusal_payload_is_sealed_and_status_refused():
    value = subject.refusal_payload("replay_mismatch", field="trace.record",
                                    detail="short")
    subject.verify_seal(value)
    assert value["status"] == "refused"
    assert value["reason"] == "replay_mismatch"
    assert value["field"] == "trace.record"


def test_jsonable_normalizes_tuples_through_the_strict_codec():
    assert subject.jsonable((1, 2, None)) == [1, 2, None]
    assert subject.jsonable({"a": (0.5, 0.25)}) == {"a": [0.5, 0.25]}


def test_default_identity_matches_the_graph_module_projection():
    match = _match(left_plane=2, right_plane=3, shift=(4, 5))
    expected = subject.default_identity(match)
    from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
    assert expected == graph_mod._input_match_identity(match)
    assert len(expected) == 9


def test_input_identities_bind_frame_ids_and_hashes(tmp_path):
    PIL = pytest.importorskip("PIL.Image")
    paths = []
    for index in range(4):
        path = tmp_path / f"image-{index:04d}.png"
        PIL.new("RGB", (8, 6), (index, 2, 3)).save(path)
        paths.append(path)
    identities = subject.input_identities(paths)
    assert [row["frame_id"] for row in identities] == [
        "image-0000", "image-0001", "image-0002", "image-0003"]
    assert [row["width"] for row in identities] == [8, 8, 8, 8]
    assert [row["height"] for row in identities] == [6, 6, 6, 6]
    assert identities[0]["sha256"] == subject.sha256_file(paths[0])


def test_registered_output_paths_are_the_named_ones():
    assert subject.OUTPUT_PATH == ROOT / ".tmp-ds-r5f-20261003.json"
    assert subject.ARTIFACT_PATH.name == (
        ".tmp-axis0-graph-stage-attribution-v6-20261003.json")
    assert subject.PROJECT_PATH.name == (
        ".tmp-axis0-graph-stage-attribution-v6-20261003.mcscan")
    assert subject.DEADLINE_SECONDS == 600
    assert subject.LIMIT_BYTES == 8 * 1024 ** 3
