"""Isolated authored-scalar controls: no generated/real images or production call."""
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib
import importlib.util
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location(
    "synthetic_endpoint_observer", ROOT / "scripts/evaluation/probe_synthetic_image_production_corner.py")
helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helper)


class ScalarOnly(NS):
    def __getattr__(self, name):
        if name in {"readings", "candidate_matches", "pixels", "points", "positions", "validate", "replay"}:
            raise AssertionError("prohibited payload or replay accessed")
        raise AttributeError(name)


def authored(status="no"):
    counters = dict.fromkeys(helper._SELECTION_COUNTERS, 0)
    counters.update(planned_assignments=3, visited_assignments=3,
                    eligibility_hashed_assignments=3, excluded_assignments=3,
                    retained_exclusions=1, omitted_exclusions=2)
    proof = {"status": status, "session_frames": ["image-0000"], "exhaustive": True,
             "counters": counters,
             "eligibility_reasons": {"no_active_factors": 1, "axis_1_disconnected": 2},
             "physical_reasons": {},
             "choices": [{"private_payload": "SECRET-POSITION-AND-PATH"}],
             "excluded_assignments": [[1, [["image-0000", 0]], "axis_1_disconnected"]]}
    sources = {key: "a" * 64 for key in (
        "session_sha256", "lattice_sha256", "signatures_sha256", "connection_sha256")}
    sources["connection_sha256"] = ""
    cert = ScalarOnly(status=status, reason={"no": "no_eligible_assignment",
        "unique": "joint_geometry_not_evaluated", "multiple": "multiple_endpoint_hypotheses",
        "unresolved_evidence": "unresolved_evidence", "budget_refused": "endpoint_budget_refused"}[status],
        frame_ids=("image-0000",), candidate_count=4, result_sha256="b" * 64, **sources)
    graph = ScalarOnly(selection=cert, frame_ids=cert.frame_ids, result_sha256="c" * 64,
                       status="unavailable", reason=cert.reason, detail=status, **sources)
    seal(graph, proof)
    return graph, proof


def seal(graph, proof):
    graph.selection.proof_json = json.dumps(proof, sort_keys=True, separators=(",", ":"))


@pytest.mark.parametrize("status", ["no", "unique", "multiple", "unresolved_evidence", "budget_refused"])
def test_carried_status_counters_and_first_failure_histograms_preserved(status):
    graph, proof = authored(status)
    if status == "budget_refused":
        proof["exhaustive"] = False
        proof["counters"].update(planned_assignments=None, unvisited_assignments=None)
        seal(graph, proof)
    record = helper._endpoint_selection_record(graph)
    assert record["available"] and not record["incomplete"]
    assert record["status"] == status and record["reason"] == graph.selection.reason
    assert record["counters"] == proof["counters"]
    assert record["eligibility_reasons"] == proof["eligibility_reasons"]
    assert record["physical_reasons"] == {}
    assert record["sources"] == {key: getattr(graph, key) for key in record["sources"]}
    assert record["proof_sha256"] == hashlib.sha256(graph.selection.proof_json.encode()).hexdigest()
    digest = record.pop("projection_sha256")
    assert digest == helper.canonical_digest(record)
    record["counters"]["excluded_assignments"] += 1
    assert helper.canonical_digest(record) != digest
    assert proof["counters"]["excluded_assignments"] == 3
    assert "SECRET" not in json.dumps(record) and "choices" not in record
    assert "excluded_assignments" not in record  # only the scalar counter is exported
    assert "positions" not in record and "pixels" not in record


def test_physical_histogram_and_unclassified_partial_work_survive():
    graph, proof = authored("budget_refused")
    proof["exhaustive"] = False
    proof["counters"].update(planned_assignments=8, unvisited_assignments=4,
        visited_assignments=4, eligibility_unclassified_assignments=1, branch_nodes=2,
        physical_assignments=2, physical_hashed_assignments=2, physical_feasible=1,
        physical_rejected=1, retained_physical_exclusions=1)
    proof["physical_reasons"] = {"physical_frame_cycle": 1}
    seal(graph, proof)
    result = helper._endpoint_selection_record(graph)
    assert result["available"] and result["counters"]["unvisited_assignments"] == 4
    assert result["counters"]["eligibility_unclassified_assignments"] == 1
    assert result["physical_reasons"] == {"physical_frame_cycle": 1}
    assert not result["exhaustive"]


@pytest.mark.parametrize("kind", ["bool", "negative", "huge", "float", "nonfinite", "missing", "unknown",
                                 "histogram", "foreign_reason", "physical", "work", "null", "exhaustive",
                                 "frames", "source", "status", "reason", "candidate_cap", "digest"])
def test_invalid_projection_fails_closed_without_exception_text(kind):
    graph, proof = authored()
    c = proof["counters"]
    if kind in {"bool", "negative", "huge", "float", "nonfinite"}:
        c["input_match_rows"] = {"bool": True, "negative": -1, "huge": 2**63,
                                 "float": 1.0, "nonfinite": float("nan")}[kind]
    elif kind == "missing": c.pop("visited_assignments")
    elif kind == "unknown": c["foreign"] = 0
    elif kind == "histogram": proof["eligibility_reasons"]["axis_1_disconnected"] = 1
    elif kind == "foreign_reason": proof["eligibility_reasons"] = {"SECRET-PATH": 3}
    elif kind == "physical": c["physical_rejected"] = 1
    elif kind == "work": c["planned_assignments"] = 4
    elif kind == "null": c["planned_assignments"] = None
    elif kind == "exhaustive": proof["exhaustive"] = 1
    elif kind == "frames": proof["session_frames"] = ["foreign/path"]
    elif kind == "source": graph.session_sha256 = "d" * 64
    elif kind == "status": proof["status"] = "multiple"
    elif kind == "reason": graph.selection.reason = "SECRET/PATH"
    elif kind == "candidate_cap": graph.selection.candidate_count = 16385
    elif kind == "digest": graph.result_sha256 = "SECRET/PATH"
    seal(graph, proof)
    record = helper._endpoint_selection_record(graph)
    assert record == {"available": False, "incomplete": True,
                      "reason": "invalid_selection_projection", "error_type": "ValueError"}
    assert "SECRET" not in json.dumps(record)


@pytest.mark.parametrize("raw", ["x" * (4 * 1024 * 1024 + 1), "?", "[]", '{"status":"no","status":"no"}'],
                         ids=["byte-cap", "non-ascii", "foreign-root", "duplicate-key"])
def test_bounded_proof_refusal(raw):
    graph, _ = authored()
    graph.selection.proof_json = raw
    assert helper._endpoint_selection_record(graph)["incomplete"]


def test_oversized_proof_is_refused_before_json_parser(monkeypatch):
    graph, _ = authored()
    graph.selection.proof_json = "x" * (4 * 1024 * 1024 + 1)
    monkeypatch.setattr(helper.json, "loads", lambda *a, **k: pytest.fail("parser reached"))
    assert helper._endpoint_selection_record(graph)["incomplete"]


def test_unreached_selection_is_unknown_not_empty_histogram():
    assert helper._endpoint_selection_record(None) == {
        "available": False, "incomplete": False, "reason": "stage_not_reached"}
    assert helper._endpoint_selection_record(NS())["incomplete"]


def test_pair_missing_zero_and_existing_hypothesis_counts_have_distinct_scope():
    missing = helper._pair_spatial_record(ScalarOnly())
    assert not missing["pair_aggregate_available"] and missing["hypotheses"] is None
    assert all(missing[key] is None for key in ("spatial_bins_left", "spatial_bins_right", "spatial_bins_total"))
    pair = ScalarOnly(spatial_bins_left=0, spatial_bins_right=2, spatial_bins_total=2,
        hypotheses=(NS(yaw_class=0, spatial_bins=0), NS(yaw_class=0, spatial_bins=2)))
    record = helper._pair_spatial_record(pair)
    assert record["pair_aggregate_available"] and record["spatial_bins_left"] == 0
    assert record["hypothesis_scope"] == "per_yaw_hypothesis"
    assert record["hypotheses"] == [{"ordinal": 0, "yaw_class": 0, "spatial_bins": 0},
                                      {"ordinal": 1, "yaw_class": 0, "spatial_bins": 2}]
    assert not record["incomplete"]
    # Production adapter exposes per-hypothesis bins but not aggregate bins.
    del pair.spatial_bins_left, pair.spatial_bins_right, pair.spatial_bins_total
    assert not helper._pair_spatial_record(pair)["pair_aggregate_available"]
    assert helper._pair_spatial_record(pair)["hypotheses"] == record["hypotheses"]


@pytest.mark.parametrize("pair", [NS(spatial_bins_left=True), NS(spatial_bins_right=-1),
    NS(hypotheses=(NS(yaw_class=4, spatial_bins=2),)),
    NS(hypotheses=(NS(yaw_class=0, spatial_bins=float("inf")),)),
    NS(hypotheses=()), NS(hypotheses=(NS(yaw_class=0, spatial_bins=1),) * 65)],
    ids=["bool", "negative", "foreign-yaw", "nonfinite", "empty", "cap"])
def test_pair_projection_bounds(pair):
    result = helper._pair_spatial_record(pair)
    assert result["incomplete"] == (getattr(pair, "hypotheses", None) != ())
    assert "points" not in result


def fake_terminal(graph):
    frame = NS(frame_id="image-0000")
    session = NS(input_frames=(frame,), frames=(frame,), scale_known=False,
                 version=1, cameras={})
    entry = NS(session=session, graph=None, pose_graph=None)
    outcome = NS(entry=entry, lattice=None, geometry=NS(records=(("f1", NS(
        status="assembled", reason="signatures_assembled", detail="")), ("g", graph))),
        output=None, result_sha256="e" * 64)
    record = {"stage": "geometry", "status": "geometry_unavailable", "reason": "g_unavailable",
              "detail": "no_eligible_assignment", **dict.fromkeys(("entry_sha256", "session_sha256",
              "lattice_sha256", "geometry_sha256", "output_sha256"), "a" * 64)}
    volume = NS(blocks={}, items=lambda: ())
    project = NS(image_derived_production=record, volume=volume, source_paths=("view-00.png",))
    return outcome, project


def test_projection_integration_preserves_terminal_and_digest_without_production(monkeypatch):
    graph, proof = authored()
    outcome, project = fake_terminal(graph)
    monkeypatch.setattr(helper, "run_probe", lambda *a: pytest.fail("production probe reached"))
    result = helper.summarize_safely(outcome, project)
    assert result["diagnostic_status"] == "complete"
    assert result["stage"] == "geometry" and result["reason"] == "g_unavailable"
    assert result["detail"] == "no_eligible_assignment" and result["block_count"] == 0
    assert [row["name"] for row in result["geometry_prefix"]] == ["f1", "g"]
    assert result["endpoint_selection"]["eligibility_reasons"] == proof["eligibility_reasons"]
    digest = helper.canonical_digest(result)
    result["endpoint_selection"]["eligibility_reasons"]["no_active_factors"] = 2
    assert helper.canonical_digest(result) != digest
    graph.selection.proof_json = "invalid"
    incomplete = helper.summarize_safely(outcome, project)
    assert incomplete["diagnostic_status"] == "incomplete"
    assert incomplete["reason"] == "g_unavailable" and incomplete["block_count"] == 0
    assert incomplete["endpoint_selection"]["incomplete"]
    assert not incomplete["litematic_created"]
