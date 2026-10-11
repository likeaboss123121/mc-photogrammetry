"""Synthetic-only contract checks for the bounded partial-core worker and packet."""
import hashlib
import importlib.util
import json
from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT / "scripts/evaluation/partial_core_image_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("partial_core_image_diagnostic", WORKER_PATH)
worker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = worker
SPEC.loader.exec_module(worker)


def test_worker_calls_each_stage_once_and_reuses_live_graph(tmp_path, monkeypatch):
    paths = tuple(str((ROOT / seal["path"]).resolve()) for seal in worker.INPUT_SEALS)
    calls = []
    graph = SimpleNamespace(
        status="assembled", reason="", frame_ids=("anchor", "a"),
        selection=SimpleNamespace(status="unique", frame_ids=("a", "b"), observability=(SimpleNamespace(
            anchor="anchor", resolved_axes=lambda frame: (0, 2) if frame == "a" else (0, 1, 2)),)),
        axis_offsets=(("a", (0.0, None, 1.0)),),
        core_solution=SimpleNamespace(frames=("a", "b", "c")),
        candidate_matches=(1, 2), readings=(1, 2, 3), matches=(1,))
    entry = SimpleNamespace(reason="", terminal="geometry_unavailable", graph=SimpleNamespace(status="ok", reason=""),
                            session=SimpleNamespace(frames=(1, 2, 3)), pose_graph=SimpleNamespace(status="ok", reason=""))
    lattice = SimpleNamespace(status="assembled", reason="")
    signatures = SimpleNamespace(status="assembled", reason="")
    support = SimpleNamespace(status="assembled", reason="", core_reading_indices=(1, 2), source_rows=(3,))

    def build(inputs, *, project_path):
        calls.append(("entry", tuple(inputs), project_path))
        return entry
    def measure(*, session, pose_graph):
        calls.append(("lattice", session, pose_graph))
        return lattice
    def derive_signatures(*, session, lattice_result):
        calls.append(("signatures", session, lattice_result))
        return signatures
    def derive_graph(**kwargs):
        calls.append(("graph", kwargs["session"], kwargs["lattice_result"], kwargs["signatures_result"]))
        return graph
    def derive_support(*, graph_result):
        calls.append(("support", graph_result))
        return support

    monkeypatch.setattr(worker, "build_image_derived_entry", build)
    monkeypatch.setattr(worker, "measure_image_derived_lattice", measure)
    monkeypatch.setattr(worker, "derive_image_derived_signatures", derive_signatures)
    monkeypatch.setattr(worker, "derive_partial_image_derived_graph", derive_graph)
    monkeypatch.setattr(worker, "derive_partial_core_axis_support", derive_support)
    report_path = tmp_path / "synthetic.json"
    result = worker.run_diagnostic(paths, "fresh-project.mcscan", report_path)

    assert [event[0] for event in calls] == ["entry", "lattice", "signatures", "graph", "support"]
    assert calls[-1] == ("support", graph)
    assert result["has_unique_assignment"] is True
    assert result["unique_frame_count"] == 2
    assert result["core_frame_count"] == 3
    assert (result["resolved_axis_count"], result["unknown_axis_count"]) == (2, 1)
    assert result["candidate_count"] == 2 and result["reading_count"] == 3 and result["factor_count"] == 1
    assert result["per_axis_observability"] == {
        "0": {"resolved_frame_count": 1, "unknown_frame_count": 0},
        "1": {"resolved_frame_count": 0, "unknown_frame_count": 1},
        "2": {"resolved_frame_count": 1, "unknown_frame_count": 0},
    }
    assert result["partial_c1_coverage"] == {
        "core_reading_count": 2, "available_reading_count": 3,
        "core_factor_count": 1, "selected_factor_count": 1}
    assert json.loads(report_path.read_text(encoding="utf-8")) == result
    assert set(result) <= {"schema", "scope", "input_seals", "stages", "selection_status",
                           "has_unique_assignment", "unique_frame_count", "core_frame_count", "resolved_axis_count",
                           "unknown_axis_count", "per_axis_observability", "candidate_count", "reading_count", "factor_count",
                           "partial_c1_coverage", "endpoint_budget", "endpoint_search",
                           "endpoint_search_budget", "timings_seconds"}
    assert result["endpoint_search_budget"] == {"profile": "legacy", "relation_visits": 262144,
                                                "numerical_relation_visits": 262144}


def test_worker_refuses_other_input_order_before_any_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "build_image_derived_entry", lambda *a, **k: pytest.fail("stage called"))
    paths = tuple(str((ROOT / seal["path"]).resolve()) for seal in reversed(worker.INPUT_SEALS))
    with pytest.raises(ValueError, match="input_inventory_mismatch"):
        worker.run_diagnostic(paths, "unused.mcscan", tmp_path / "unused.json")


def test_stage_exception_still_writes_scalar_report(tmp_path, monkeypatch):
    paths = tuple(str((ROOT / seal["path"]).resolve()) for seal in worker.INPUT_SEALS)
    monkeypatch.setattr(worker, "build_image_derived_entry",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("private detail")))
    report_path = tmp_path / "raised.json"
    result = worker.run_diagnostic(paths, "fresh-project.mcscan", report_path)
    assert result["stages"]["entry"] == {
        "status": "raised", "reason": "stage_raised", "error_type": "RuntimeError",
        "terminal": "not_reached", "frame_count": 0}
    assert result["stages"]["lattice"]["status"] == "not_reached"
    assert "private detail" not in report_path.read_text(encoding="utf-8")


def test_c1_is_not_called_for_refused_partial_graph(tmp_path, monkeypatch):
    from mc_photogrammetry.reconstruction import plane_placement
    paths = tuple(str((ROOT / seal["path"]).resolve()) for seal in worker.INPUT_SEALS)
    entry = SimpleNamespace(reason="", terminal="geometry_unavailable",
        graph=SimpleNamespace(status="ok", reason=""),
        pose_graph=SimpleNamespace(status="ok", reason=""),
        session=SimpleNamespace(frames=(1, 2, 3)))
    lattice = SimpleNamespace(status="assembled", reason="")
    signatures = SimpleNamespace(status="assembled", reason="")
    graph = plane_placement._make_refusal("invalid_input", {"stage": "synthetic"})
    calls = []
    monkeypatch.setattr(worker, "build_image_derived_entry", lambda *a, **k: entry)
    monkeypatch.setattr(worker, "measure_image_derived_lattice", lambda **k: lattice)
    monkeypatch.setattr(worker, "derive_image_derived_signatures", lambda **k: signatures)
    monkeypatch.setattr(worker, "derive_partial_image_derived_graph", lambda **k: graph)
    monkeypatch.setattr(worker, "derive_partial_core_axis_support",
                        lambda **k: calls.append("C1") or pytest.fail("C1 must not run"))
    result = worker.run_diagnostic(paths, "fresh-project.mcscan", tmp_path / "refused.json")
    assert calls == []
    assert result["stages"]["partial_graph"] == {
        "status": "refused", "reason": "invalid_input", "result_type": "PlacementRefusal"}
    assert result["stages"]["partial_c1"] == {"status": "not_reached", "reason": ""}
    assert result["endpoint_budget"] == {"status": "not_reached", "reason": "no_selection",
        "category": None, "used": None, "amount": None, "cap": None}


@pytest.mark.parametrize("category", sorted(worker.ENDPOINT_BUDGET_CATEGORIES))
def test_endpoint_budget_refusal_emits_only_whitelisted_scalars(category):
    selection = SimpleNamespace(status="budget_refused",
        proof_record=lambda: {"denial": [category, 100, 1, 100],
                              "counters": {"secret_counter": 999}, "private": "omit"})
    assert worker._endpoint_budget(selection) == {
        "status": "budget_refused", "reason": "", "category": category,
        "used": 100, "amount": 1, "cap": 100}


@pytest.mark.parametrize("proof", [
    {"denial": ["unapproved_counter", 1, 2, 2], "private": "omit"},
    {"denial": ["branch_nodes", True, 1, 1], "private": "omit"},
    {"denial": ["branch_nodes", 1, 0, 1], "private": "omit"},
    {"denial": ["branch_nodes", 0, 1, 1], "private": "omit"},
])
def test_malformed_endpoint_proof_is_refused_without_payload(proof):
    selection = SimpleNamespace(status="budget_refused", proof_record=lambda: proof)
    result = worker._endpoint_budget(selection)
    assert result == {"status": "malformed", "reason": "invalid_denial",
                      "category": None, "used": None, "amount": None, "cap": None}
    assert "private" not in result and "counters" not in result


def test_endpoint_budget_non_refusal_and_unavailable_are_explicit():
    assert worker._endpoint_budget(None) == {
        "status": "not_reached", "reason": "no_selection",
        "category": None, "used": None, "amount": None, "cap": None}
    ordinary = SimpleNamespace(status="multiple", proof_record=lambda: pytest.fail("proof not read"))
    assert worker._endpoint_budget(ordinary) == {
        "status": "not_reached", "reason": "selection_not_budget_refused",
        "category": None, "used": None, "amount": None, "cap": None}


def test_worker_bootstraps_current_repo_source_before_project_imports(monkeypatch):
    # Later pytest collection can prepend tests/ after this worker's initial
    # import. Exercise a fresh bootstrap on an isolated path list instead.
    monkeypatch.setattr(sys, "path", sys.path.copy())
    SPEC.loader.exec_module(worker)
    import mc_photogrammetry
    assert worker.REPO_ROOT == ROOT
    assert sys.path[0] == str(ROOT / "src")
    assert Path(mc_photogrammetry.__file__).resolve().is_relative_to(ROOT / "src")


@pytest.mark.parametrize("profile", ["legacy", "relations-524288"])
def test_cli_budget_profile_is_explicit_and_never_retries(monkeypatch, profile):
    calls = []
    monkeypatch.setattr(worker, "run_diagnostic", lambda *a, **k: calls.append((a, k)) or {})
    assert worker.main(["--image", "synthetic", "--project-output", "unused.mcscan",
                        "--report-output", "unused.json", "--partial-search-budget-profile", profile]) == 0
    assert len(calls) == 1
    if profile == "legacy":
        assert calls[0][1] == {}
    else:
        assert type(calls[0][1]["budget"]) is worker.csp.PartialEndpointBudget
        assert calls[0][1]["budget"].relation_visits == 524288


def test_cli_budget_profile_omitted_is_exact_legacy_call_shape(monkeypatch):
    calls = []
    monkeypatch.setattr(worker, "run_diagnostic", lambda *a, **k: calls.append(k) or {})
    assert worker.main(["--image", "synthetic", "--project-output", "unused.mcscan",
                        "--report-output", "unused.json"]) == 0
    assert calls == [{}]
    with pytest.raises(SystemExit):
        worker.main(["--image", "synthetic", "--project-output", "unused.mcscan",
                     "--report-output", "unused.json", "--partial-search-budget-profile", "arbitrary"])
    assert calls == [{}]


def test_explicit_profile_reaches_real_partial_graph_seam_once(tmp_path, monkeypatch):
    paths = tuple(str((ROOT / seal["path"]).resolve()) for seal in worker.INPUT_SEALS)
    entry = SimpleNamespace(reason="", terminal="geometry_unavailable",
        graph=SimpleNamespace(status="ok", reason=""), pose_graph=SimpleNamespace(status="ok", reason=""),
        session=SimpleNamespace(frames=(1, 2)))
    from test_partial_axis_observability import fixture, certificate
    cert, _ = certificate(fixture(), worker.csp.PartialEndpointBudget())
    graph = SimpleNamespace(status="unavailable", reason="partial_geometry_unavailable", selection=cert)
    calls = []
    monkeypatch.setattr(worker, "build_image_derived_entry", lambda *a, **k: entry)
    monkeypatch.setattr(worker, "measure_image_derived_lattice", lambda **k: SimpleNamespace(status="assembled", reason=""))
    monkeypatch.setattr(worker, "derive_image_derived_signatures", lambda **k: SimpleNamespace(status="assembled", reason=""))
    monkeypatch.setattr(worker, "derive_partial_image_derived_graph", lambda **k: calls.append(k) or graph)
    monkeypatch.setattr(worker, "derive_partial_core_axis_support", lambda **k: pytest.fail("unavailable graph reached C1"))
    budget = worker.csp.PartialEndpointBudget()
    result = worker.run_diagnostic(paths, "unused.mcscan", tmp_path / "profile.json", budget=budget)
    assert len(calls) == 1 and calls[0]["budget"] is budget
    assert result["endpoint_search_budget"] == {"profile": "partial-relations-524288-v1",
        "relation_visits": 524288, "numerical_relation_visits": 262144}
    assert result["endpoint_search"]["status"] == "complete"
    assert "proof" not in json.dumps(result) and "counters" not in result["endpoint_search"]


def test_foreign_budget_refused_before_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "build_image_derived_entry", lambda *a, **k: pytest.fail("entry called"))
    with pytest.raises(ValueError):
        worker.run_diagnostic((), "unused.mcscan", tmp_path / "unused.json", budget=object())
    assert not (tmp_path / "unused.json").exists()


def test_search_counter_whitelist_reconciles_and_excludes_payload():
    from test_partial_axis_observability import fixture, certificate
    cert, _ = certificate(fixture(), worker.csp.PartialEndpointBudget())
    proof = cert.proof_record()
    proof["counters"]["private_counter"] = "sensitive payload"
    result = worker._endpoint_search(SimpleNamespace(proof_record=lambda: proof), cert.budget)
    assert result["status"] == "complete"
    assert result["relation_visits"] == proof["counters"]["relation_visits"]
    assert (sum(result["relation_visit_attribution"].values())
            + result["unattributed_relation_visits"] == result["relation_visits"])
    assert len(json.dumps(result).encode()) < 2048
    assert "sensitive" not in json.dumps(result) and "choices" not in result and "proof" not in result


@pytest.mark.parametrize("change", ["bool", "oversize", "visited", "physical", "attribution", "missing"])
def test_search_counter_bounds_fail_closed_without_dumping_proof(change):
    from test_partial_axis_observability import fixture, certificate
    cert, _ = certificate(fixture())
    proof = cert.proof_record()
    counts = proof["counters"]
    if change == "bool":
        counts["branch_nodes"] = True
    elif change == "oversize":
        counts["relation_visits"] = 524289
    elif change == "visited":
        counts["visited_assignments"] = counts["planned_assignments"] + 1
    elif change == "physical":
        counts["physical_assignments"] = counts["branch_nodes"] + 1
    elif change == "attribution":
        counts["preflight_visits"] = counts["relation_visits"] + 1
    else:
        del counts["relation_visits"]
    proof["private"] = "hidden"
    result = worker._endpoint_search(SimpleNamespace(proof_record=lambda: proof), cert.budget)
    assert result["status"] == "malformed" and result["relation_visits"] is None
    assert all(value is None for value in result["relation_visit_attribution"].values())
    assert "hidden" not in json.dumps(result)


def test_direct_script_bootstrap_works_without_pythonpath():
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-B", str(WORKER_PATH), "--help"],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--project-output" in result.stdout and "--report-output" in result.stdout


def test_packet_is_static_single_invocation_and_binds_all_source_and_input_seals():
    run_dir = ROOT / "agent swarm instantiation template/runs"
    packet_path = run_dir / "SOL-PARTIAL-RELATION-BUDGET-01.packet.json"
    closure_path = run_dir / "SOL-PARTIAL-RELATION-BUDGET-01.source-closure.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    expected_members = {p.relative_to(ROOT).as_posix() for p in (ROOT / "src").rglob("*.py")}
    expected_members.update({"scripts/evaluation/partial_core_image_diagnostic.py",
                             "tests/test_partial_core_image_diagnostic.py",
                             "tests/test_partial_endpoint_budget.py",
                             "tests/test_partial_axis_observability.py",
                             "tests/test_image_derived_partial_joint_graph.py",
                             "tools/run_image_packet.ps1", "pyproject.toml"})
    assert {pin["path"] for pin in closure["repo_pins"]} == expected_members
    assert closure["closure_policy"] == "full-src-python-superset-v1"
    assert packet["schema"] == "mc-photogrammetry-image-run-packet-v1"
    assert packet["lead_authorized"] is False and packet["launch_allowed"] is False
    assert packet["maximum_invocations"] == 1
    assert packet["limits"] == {"sampling_seconds": 1, "wall_timeout_seconds": 900,
                                 "working_set_threshold_bytes": 2 * 1024**3}
    assert [item["sha256"] for item in packet["inputs"]] == [item["sha256"] for item in worker.INPUT_SEALS]
    assert [item["bytes"] for item in packet["inputs"]] == [item["bytes"] for item in worker.INPUT_SEALS]
    assert [item["path"] for item in packet["inputs"]] == [item["path"] for item in worker.INPUT_SEALS]
    assert [item["absolute_path"] for item in packet["inputs"]] == [str((ROOT / item["path"]).resolve()) for item in worker.INPUT_SEALS]
    assert packet["arguments"].count("--image") == 6
    assert packet["arguments"][0:2] == ["-B", "scripts/evaluation/partial_core_image_diagnostic.py"]
    profile_index = packet["arguments"].index("--partial-search-budget-profile")
    assert packet["arguments"][profile_index + 1] == "relations-524288"
    argv_images = [packet["arguments"][i + 1] for i, value in enumerate(packet["arguments"][:-1])
                   if value == "--image"]
    assert argv_images == [item["absolute_path"] for item in packet["inputs"]]
    assert not any("litematic" in arg.lower() or "manifest" in arg.lower() or "f3" == arg.lower()
                   for arg in packet["arguments"])
    assert packet["source_closure_sha256"] == hashlib.sha256(closure_path.read_bytes()).hexdigest().upper()
    for pin in closure["repo_pins"]:
        content = (ROOT / pin["path"]).read_bytes()
        assert len(content) == pin["bytes"] and hashlib.sha256(content).hexdigest().upper() == pin["sha256"]
    for field in ("project_output", "json_output", "stdout_log", "stderr_log", "monitor_receipt"):
        assert not (ROOT / packet[field]).exists()
