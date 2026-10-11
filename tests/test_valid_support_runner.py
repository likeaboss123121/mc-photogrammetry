"""Actual-driver controls with synthetic bytes and source trace DATA classes only."""
import ast
import base64
from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mc_photogrammetry.reconstruction import evaluation_gui_masked_segments as support
from mc_photogrammetry.reconstruction import evaluation_valid_stencil_segments as stencil
from mc_photogrammetry.reconstruction.evaluation_valid_stencil_segments import (
    AcquisitionIntegrityError, AcquisitionCapError, FROZEN_CONFIG,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("valid_runner_under_test", ROOT / "scripts/evaluation/run_valid_support.py")
runner = importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _source_data_classes():
    tree = ast.parse((ROOT / "scripts/evaluation/run_camera_mixed_depth.py").read_text(encoding="utf-8"))
    selected = ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef)
        and n.name in ("TracePolicy", "TraceRecorder")], type_ignores=[])
    namespace = {"dataclass": dataclass, "asdict": asdict, "np": np, "_json": runner.canonical,
                 "_sha": runner.sha, "_require": runner.require}
    exec(compile(selected, "source-trace-data-only", "exec"), namespace)
    return namespace["TracePolicy"], namespace["TraceRecorder"]


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    """Mock registry is a unit fixture, not a generated evaluation cohort."""
    plan_path = tmp_path / "valid-plan.json"
    sidecar = tmp_path / "valid-payload.gz"
    claim = tmp_path / "valid-claim.json"
    mask = np.zeros((80, 160), dtype=bool)
    mask[:, 150:] = True
    _, band = support.mask_transition_band(mask)
    registry = []
    for i in range(12):
        rgb = np.full((80, 160, 3), i, dtype=np.uint8)
        registry.append({"identity": f"unit-{i}", "family": f"family-{i%6}",
            "base_identity": None if i < 6 else f"unit-{i-6}",
            "stratum": "ordinary" if i < 6 else "stress", "rgb_shape": list(rgb.shape),
            "rgb_sha256": runner.sha(rgb.tobytes()), "rgb_base64": base64.b64encode(rgb.tobytes()).decode(),
            "mask_sha256": runner.sha(mask.tobytes()), "expanded_mask_sha256": runner.sha((mask|band).tobytes())})
    plan = {"schema": runner.SCHEMA, "study": runner.STUDY, "config": asdict(FROZEN_CONFIG),
        "config_sha256": FROZEN_CONFIG.sha256, "solver": runner.SOLVER, "policy": runner.POLICY,
        "source_sha256": runner.source_pins(), "runtime": runner.runtime_pins(), "score_sites": runner.score_sites(),
        "execution_contract": runner.execution_contract(),
        "paths": {"plan": str(plan_path), "sidecar": str(sidecar), "claim": str(claim)},
        "retired_paths": [], "retired_rgb_sha256": [], "registry": registry,
        "truth_registry": [{"identity": row["identity"], "reference": "mock-ref-"+row["identity"],
                            "sha256": runner.sha(runner.canonical({"unit_truth": row["identity"]}))} for row in registry],
        "execution_state": "NOT_STARTED", "sealed_predictions": [], "seal_validation": None,
        "evaluations": None, "summary": None, "failure": None}
    plan["preregistration_sha256"] = runner.preregistration_digest(plan)
    def write():
        plan_path.write_bytes((runner.canonical(plan)+"\n").encode())
    write()
    policy, recorder = _source_data_classes()
    module = SimpleNamespace(TracePolicy=policy, TraceRecorder=recorder,
        bind_trace_contract=lambda *args, **kwargs: None, clear_trace_contract=lambda: None)
    raw = np.asarray([[2., 5.+i*4., 140., 5.+i*4.] for i in range(12)], dtype=np.float32)
    candidate = np.asarray([[3., 4.+i*4., 140., 4.+i*4.] for i in range(13)], dtype=np.float64)
    state = {"refusal": False, "calls": [], "truth": [], "score": [], "mask_calls": 0,
             "readbacks": 0, "budgets": [], "fail_at": None, "large_stream": False,
             "zero_inliers": False, "prediction_update": None, "scored_predictions": [],
             "expected_digest": plan["preregistration_sha256"]}
    def automatic(h, w):
        state["mask_calls"] += 1
        assert (h, w) == mask.shape
        return mask.copy()
    calibration = SimpleNamespace(detect_segments=lambda gray, **kwargs: raw)
    gui = SimpleNamespace(create_gui_mask=automatic)
    # Exact synthetic support telemetry for the candidate stub; no extractor,
    # native detector or camera solver executes in these driver tests.
    edgels = np.zeros(13*138, dtype=stencil.EDGEL)
    edges = np.zeros(13*137, dtype=stencil.EDGE)
    paths = np.zeros(13, dtype=stencil.PATH)
    groups = np.zeros(13, dtype=stencil.GROUP)
    lines = np.zeros(13, dtype=stencil.LINE)
    samples = 13*275
    projection = np.full((samples+3)//4, 255, dtype=np.uint8)
    if samples%4: projection[-1] = (1 << (2*(samples%4)))-1
    for line in range(13):
        begin = line*138
        for j in range(138): edgels[begin+j] = (3+j, 4+line*4, 64, 0, 4096)
        for j in range(137): edges[line*137+j] = (begin+j, begin+j+1)
        paths[line] = (begin, 138)
        groups[line] = (line, begin, begin+138, -1, 0, 1)
        lines[line] = (line, begin, 138, 138*4096, 0., 0, line*275, 275)
    extraction_arrays = {"edgels": edgels, "graph_edges": edges, "paths": paths,
        "path_membership": np.arange(len(edgels), dtype="<u4"), "cycle_closures": np.empty(0, dtype=stencil.EDGE),
        "fit_groups": groups, "support_membership": np.arange(len(edgels), dtype="<u4"), "line_ledger": lines,
        "projection_occupancy": projection, "admitted_candidate_ids": np.arange(13, dtype="<u4")}
    counts = dict.fromkeys(("footprint_frame", "footprint_original", "footprint_band", "footprint_valid",
        "below_threshold", "nms_suppressed", "edgels", "neighbor_probes", "graph_edges", "paths", "singletons",
        "cycle_closures", "fit_nodes", "fit_points", "degenerate_groups", "pre_length", "duplicate", "frame",
        "original", "band", "short", "admitted", "projection_samples"), 0)
    counts.update(footprint_frame=80*160-len(edgels), footprint_valid=len(edgels), edgels=len(edgels),
        graph_edges=len(edges), paths=13, fit_nodes=13, fit_points=2*len(edgels), pre_length=13,
        admitted=13, projection_samples=samples)
    def extract(gray, excluded, **kwargs):
        assert not np.shares_memory(gray, excluded)
        return SimpleNamespace(segments=candidate, pre_length=candidate.copy(), arrays=extraction_arrays, counts=counts)
    monkeypatch.setattr(support, "extract_segments", extract)
    def trace(image, frame_id):
        state["calls"].append((frame_id, id(image), image.tobytes()))
        if len(state["calls"]) == state["fail_at"]:
            raise AcquisitionCapError("unit-cap")
        r = module.TraceRecorder(policy())
        original = gui.create_gui_mask(*image.shape[:2])
        gray = np.zeros(image.shape[:2], dtype=np.uint8)
        def stage(name, output, input_sha=None):
            r.add(r.stages, {"stage": name, "input_sha256": input_sha,
                "output": output, "delegate_seconds": .01}, 32)
        stage("create_gui_mask", r.array(original))
        stage("enhance", r.array(gray), r.array(gray))
        segments = calibration.detect_segments(gray)
        segment_sha = r.array(segments, segments=True)
        stage("detect_segments", segment_sha, r.array(gray))
        prediction = None
        if not state["refusal"] and (not state["large_stream"] or len(state["calls"]) == 1):
            weights = np.linalg.norm(segments[:, 2:]-segments[:, :2], axis=1)
            event = {"site": "fine", "span": next([v["start"], v["end"]] for v in plan["score_sites"] if v["tag"] == "fine"),
                "segments_sha256": segment_sha, "weights_sha256": r.array(weights),
                "camera": {"focal": 150., "yaw": 20., "pitch": -10., "width": 160, "height": 80,
                           "threshold": 3., "roll": 0., "principal_point": None},
                "winner": {"best_before": [0., 150., 20., -10.]}, "score": .9}
            for record_index in range(76095 if state["large_stream"] else 1):
                # Fresh full events, not a list and not a truncated trace.
                r.add(r.scores, dict(event, score=0. if state["zero_inliers"] else .9 + record_index/1000000.), 120000)
            prediction = {"focal_pixels": 150., "yaw_degrees": 20., "pitch_degrees": -10., "width": 160,
                "height": 80, "inliers": 12, "segments": len(segments), "median_residual": .3,
                "per_axis_inliers": [4, 4, 4], "roll_degrees": 0., "principal_point": None, "notes": []}
            if state["zero_inliers"]:
                prediction.update(inliers=0, per_axis_inliers=[0, 0, 0], median_residual=None)
            if state["prediction_update"]:
                prediction.update(state["prediction_update"])
        stage("estimate_camera", prediction, segment_sha)
        return {"status": "ordinary_refusal" if prediction is None else "returned",
            "reason": "image_calibration_failed" if prediction is None else "", "prediction": prediction,
            "trace": r.payload(), "seconds": .5, "rgb_sha256": runner.sha(image.tobytes()), "rgb_shape": list(image.shape)}
    module.trace_rgb = trace
    dependencies = runner.Dependencies(trace, module, calibration, gui)
    # Only dependency attestation is replaced: no native modules are imported.
    monkeypatch.setattr(runner, "validate_dependencies", lambda d: runner.require(d is dependencies, "unit-dependencies"))
    monkeypatch.setattr(runner, "validate_scoring", lambda loader, evaluator: None)
    monkeypatch.setattr(runner.scoring, "validate_truth", lambda truth, row: truth)
    real_read = support.ValidAcquisitionMemberStore.read_member
    def read(store, descriptor):
        if store._active_budget is None:
            state["readbacks"] += 1
        return real_read(store, descriptor)
    monkeypatch.setattr(support.ValidAcquisitionMemberStore, "read_member", read)
    real_append = support.ValidAcquisitionMemberStore.append
    def append(store, payload, plan, identity, arm, *, budget=None):
        assert budget is not None and budget.live > len(payload)
        state["budgets"].append(budget)
        return real_append(store, payload, plan, identity, arm, budget=budget)
    monkeypatch.setattr(support.ValidAcquisitionMemberStore, "append", append)
    def truth(ref):
        durable = json.loads(plan_path.read_text(encoding="utf-8"))
        assert durable["execution_state"] == "PREDICTIONS_VALIDATED"
        assert len(durable["sealed_predictions"]) == 24 and state["readbacks"] == 24
        state["truth"].append(ref["identity"])
        return runner.canonical({"unit_truth": ref["identity"]}).encode()
    def evaluate(prediction, truth, gates):
        assert "unit_truth" in truth and gates == runner.POLICY["camera_gates"]
        state["score"].append(prediction["segments"])
        state["scored_predictions"].append(prediction)
        return {"diagnostic_pass": True, "focal_relative_error": .01,
            "orientation_degrees_mod90": .1, "principal_error_over_diagonal": .001}
    def run():
        return runner.run_plan(plan_path, expected_preregistration_sha256=state["expected_digest"],
            dependencies=dependencies, truth_loader=truth, evaluator=evaluate)
    return SimpleNamespace(plan=plan, path=plan_path, sidecar=sidecar, claim=claim, state=state, write=write,
        run=run, dependencies=dependencies, raw=raw, candidate=candidate)


def test_actual_driver_success_pairs_and_independent_barrier(fixture):
    f = fixture
    result = f.run()
    assert result["execution_state"] == "COMPLETE"
    assert len(result["sealed_predictions"]) == 24
    assert f.state["readbacks"] == 24 and len(f.state["truth"]) == 12 and len(f.state["score"]) == 24
    assert f.state["mask_calls"] == 12
    for a, b in zip(f.state["calls"][::2], f.state["calls"][1::2]):
        assert a == b
    assert f.state["score"] == [12, 13]*12
    assert all(b.peak <= FROZEN_CONFIG.max_live_bytes for b in f.state["budgets"])
    assert result["summary"]["accepted"] is False  # Same passes give no added cameras.
    assert len(result["summary"]["family_outcomes"]) == 12
    claim = json.loads(f.claim.read_text())
    assert claim["expected_preregistration_sha256"] == f.state["expected_digest"] == result["preregistration_sha256"]


def test_actual_driver_all_ordinary_refusals_continue(fixture):
    fixture.state["refusal"] = True
    result = fixture.run()
    assert result["execution_state"] == "COMPLETE" and len(fixture.state["calls"]) == 24
    assert len(fixture.state["truth"]) == 12 and fixture.state["score"] == []
    assert all(v["status"] == "ordinary_refusal" and not v["passed"] for v in result["evaluations"])
    assert not result["summary"]["accepted"]


def test_typed_cap_preserves_partial_seals_and_blocks_truth(fixture):
    fixture.state["fail_at"] = 4
    with pytest.raises(AcquisitionCapError, match="unit-cap"):
        fixture.run()
    result = json.loads(fixture.path.read_text())
    assert result["execution_state"] == "FAILED" and len(result["sealed_predictions"]) == 3
    assert not result["failure"]["post_hoc_truth_accessed"] and not result["failure"]["retry_performed"]
    assert fixture.state["truth"] == fixture.state["score"] == []
    with pytest.raises(AcquisitionIntegrityError, match="attempt_consumed"):
        fixture.run()


@pytest.mark.parametrize("tamper", ["schema", "binding", "config", "prediction", "refusal", "score_reference",
    "score_count", "trace_bytes", "mask", "mask_derivation", "ledger", "run_bytes", "projection", "missing_stage", "inner_hash"])
def test_actual_driver_receipt_tampering_fails_before_publication(fixture, monkeypatch, tamper):
    f = fixture
    original = support.valid_acquisition_payload
    def payload(bindings, traced, captured, **kwargs):
        encoded = original(bindings, traced, captured, **kwargs)
        if bindings["arm"] == "baseline":
            return encoded
        value = json.loads(encoded)
        if tamper == "schema": value["schema"] = "generic-bypass"
        if tamper == "binding": value["identity"] = "wrong-case"
        if tamper == "config": value["config_sha256"] = "0"*64
        if tamper == "prediction": value["trace_result"]["prediction"]["width"] += 1
        if tamper == "refusal": value["trace_result"]["status"] = "ordinary_refusal"
        if tamper == "score_reference":
            # Fully valid gzip/count/hash but nonexistent referenced array.
            budget = support.AllocationBudget(FROZEN_CONFIG.max_live_bytes)
            stream = support.LosslessScoreStream(budget)
            def copy(index, event):
                event["segments_sha256"] = "0"*64
                stream.append(event)
            support.validate_score_stream(value["trace_result"]["trace"]["scores"], budget, on_record=copy)
            value["trace_result"]["trace"]["scores"] = stream.finish()
        if tamper == "score_count": value["trace_result"]["trace"]["score_events"] += 1
        if tamper == "trace_bytes": value["trace_result"]["trace"]["trace_event_bytes"] += 1
        if tamper == "mask": value["arrays"]["expanded_mask"]["sha256"] = "0"*64
        if tamper in ("mask_derivation", "run_bytes", "projection"):
            name = {"mask_derivation": "expanded_mask", "run_bytes": "run_ledger", "projection": "projection_occupancy"}[tamper]
            meta = value["arrays"][name]
            raw = bytearray(base64.b64decode(meta["base64"]))
            raw[-1] ^= 1
            meta["base64"] = base64.b64encode(raw).decode()
            meta["sha256"] = runner.sha(raw)
        if tamper == "ledger": value["split_counts"]["samples"] += 1
        if tamper == "missing_stage": value["trace_result"]["trace"]["stages"].pop(0)
        if tamper == "inner_hash": value["trace_result"]["trace"]["scores"]["raw_sha256"] = "0"*64
        return runner.canonical(value).encode()
    monkeypatch.setattr(support, "valid_acquisition_payload", payload)
    with pytest.raises(AcquisitionIntegrityError): f.run()
    result = json.loads(f.path.read_text())
    assert len(result["sealed_predictions"]) == 1
    assert f.state["truth"] == f.state["score"] == []


@pytest.mark.parametrize("tamper", ["missing", "reordered", "descriptor_hash", "gzip", "tail", "source_drift"])
def test_actual_driver_barrier_tampering_never_calls_truth(fixture, monkeypatch, tamper):
    persist = runner._persist
    def intercepted(path, plan, sidecar, **kwargs):
        if plan["execution_state"] == "PREDICTIONS_SEALED":
            if tamper == "missing": plan["sealed_predictions"].pop()
            if tamper == "reordered": plan["sealed_predictions"][0:2] = reversed(plan["sealed_predictions"][0:2])
            if tamper == "descriptor_hash": plan["sealed_predictions"][0]["payload_sha256"] = "0"*64
            if tamper == "gzip":
                with sidecar.open("r+b") as stream:
                    stream.seek(15); value = stream.read(1); stream.seek(15); stream.write(bytes([value[0]^1]))
            if tamper == "tail":
                with sidecar.open("ab") as stream: stream.write(b"unregistered")
            if tamper == "source_drift": monkeypatch.setattr(runner, "source_pins", lambda: {})
        persist(path, plan, sidecar, **kwargs)
    monkeypatch.setattr(runner, "_persist", intercepted)
    with pytest.raises(AcquisitionIntegrityError): fixture.run()
    assert fixture.state["truth"] == fixture.state["score"] == []
    assert json.loads(fixture.path.read_text())["execution_state"] == "FAILED"


@pytest.mark.parametrize("change,reason", [
    ("source", "source_drift"), ("runtime", "runtime_drift"), ("digest", "preregistration_pin"),
    ("duplicate", "duplicate_or_retired"), ("family", "stress_family"), ("count", "registry_count"),
    ("retired_rgb", "duplicate_or_retired"), ("retired_path", "retired_path"),
    ("sidecar", "nonempty_sidecar"), ("claim", "claim_consumed"), ("path", "path_pin")])
def test_preflight_rejection_consumes_nothing(fixture, change, reason):
    f = fixture
    if change == "source": f.plan["source_sha256"] = {}
    if change == "runtime": f.plan["runtime"] = {}
    if change == "digest": f.plan["preregistration_sha256"] = "0"*64
    if change == "duplicate": f.plan["registry"][1]["identity"] = f.plan["registry"][0]["identity"]
    if change == "family": f.plan["registry"][6]["family"] = "bad-family"
    if change == "count": f.plan["registry"].pop()
    if change == "retired_rgb": f.plan["retired_rgb_sha256"] = [f.plan["registry"][0]["rgb_sha256"]]
    if change == "retired_path": f.plan["retired_paths"] = [str(f.path)]
    if change == "sidecar": f.sidecar.write_bytes(b"prior")
    if change == "claim": f.claim.write_text("consumed")
    if change == "path": f.plan["paths"]["plan"] = str(f.path.with_name("different.json"))
    if change != "digest": f.plan["preregistration_sha256"] = runner.preregistration_digest(f.plan)
    f.write()
    with pytest.raises(AcquisitionIntegrityError, match=reason): f.run()
    assert f.state["calls"] == f.state["truth"] == []
    assert json.loads(f.path.read_text())["execution_state"] == "NOT_STARTED"


def test_short_append_preserves_unregistered_tail_and_no_retry(fixture, monkeypatch):
    real = support.GzipMemberStore._write_append
    count = 0
    def short(store, blob):
        nonlocal count
        count += 1
        return real(store, blob if count == 1 else blob[:len(blob)//2])
    monkeypatch.setattr(support.GzipMemberStore, "_write_append", short)
    with pytest.raises(AcquisitionIntegrityError): fixture.run()
    result = json.loads(fixture.path.read_text())
    assert len(result["sealed_predictions"]) == 1
    assert fixture.sidecar.stat().st_size > result["sealed_predictions"][0]["compressed_bytes"]
    assert fixture.state["truth"] == fixture.state["score"] == []
    result["execution_state"] = "NOT_STARTED"
    result["sealed_predictions"] = []
    result["failure"] = None
    fixture.path.write_bytes((runner.canonical(result)+"\n").encode())
    with pytest.raises(AcquisitionIntegrityError, match="claim_consumed"): fixture.run()


def test_actual_driver_76095_events_stay_lossless_and_streamed(fixture):
    fixture.state["large_stream"] = True
    result = fixture.run()
    assert result["execution_state"] == "COMPLETE"
    assert fixture.state["score"] == [12]
    descriptor = result["sealed_predictions"][0]
    encoded = support.ValidAcquisitionMemberStore(fixture.sidecar).read_member(descriptor)
    packet = json.loads(encoded)["trace_result"]["trace"]["scores"]
    assert packet["record_count"] == 76095
    ends = []
    def visitor(index, event):
        if index in (0, 76094): ends.append(event["score"])
    support.validate_score_stream(packet, support.AllocationBudget(FROZEN_CONFIG.max_live_bytes), on_record=visitor)
    assert ends == [.9, .9+76094/1000000.]
    print({"driver_records": packet["record_count"], "member_json_bytes": len(encoded),
           "score_raw_bytes": packet["raw_bytes"], "score_gzip_bytes": packet["compressed_bytes"],
           "active_managed_peak": max(b.peak for b in fixture.state["budgets"])})


def test_truth_hash_failure_loads_no_partial_scores(fixture):
    fixture.plan["truth_registry"][-1]["sha256"] = "0"*64
    fixture.plan["preregistration_sha256"] = runner.preregistration_digest(fixture.plan)
    # This test explicitly approves a plan with a mismatched truth-byte receipt.
    fixture.state["expected_digest"] = fixture.plan["preregistration_sha256"]
    fixture.write()
    with pytest.raises(AcquisitionIntegrityError, match="truth_reference_hash"): fixture.run()
    assert len(fixture.state["truth"]) == 12 and fixture.state["score"] == []
    result = json.loads(fixture.path.read_text())
    assert result["execution_state"] == "FAILED" and result["evaluations"] is None


def test_retained_driver_budget_overflow_stops_before_any_seal(fixture, monkeypatch):
    real = runner._driver_recorder
    monkeypatch.setattr(runner, "_driver_recorder", lambda base, retained: real(base, FROZEN_CONFIG.max_live_bytes))
    with pytest.raises(AcquisitionCapError, match="driver_retained_input_cap"): fixture.run()
    assert fixture.state["truth"] == fixture.state["score"] == []
    assert json.loads(fixture.path.read_text())["sealed_predictions"] == []


def test_zero_inlier_returned_cameras_complete_barrier_and_are_scored(fixture):
    fixture.state["zero_inliers"] = True
    result = fixture.run()
    assert result["execution_state"] == "COMPLETE" and result["seal_validation"]["prediction_count"] == 24
    assert fixture.state["readbacks"] == 24 and len(fixture.state["truth"]) == 12
    assert len(fixture.state["scored_predictions"]) == 24
    assert all(p["median_residual"] is None and p["inliers"] == 0
        and p["per_axis_inliers"] == [0, 0, 0] and p["focal_pixels"] == 150.
        for p in fixture.state["scored_predictions"])
    # Zero inliers neither relabels the outcome nor overrides camera metrics.
    assert all(v["status"] == "returned" and v["passed"] is True for v in result["evaluations"])


@pytest.mark.parametrize("update", [
    {"median_residual": None},
    {"inliers": 0, "per_axis_inliers": [1, 0, 0], "median_residual": None},
    {"inliers": 0, "per_axis_inliers": [0, 0, 0], "median_residual": .3},
    {"focal_pixels": None}, {"yaw_degrees": None}, {"pitch_degrees": None},
    {"roll_degrees": None}, {"focal_pixels": float("inf")},
    {"yaw_degrees": float("nan")}, {"median_residual": float("inf")},
    {"median_residual": -.1}, {"principal_point": [None, 40.]},
])
def test_malformed_camera_or_residual_still_aborts(fixture, update):
    fixture.state["prediction_update"] = update
    with pytest.raises(AcquisitionIntegrityError): fixture.run()
    result = json.loads(fixture.path.read_text())
    assert result["execution_state"] == "FAILED" and result["sealed_predictions"] == []
    assert fixture.state["truth"] == fixture.state["score"] == []


def test_rehashed_valid_plan_mutation_fails_external_anchor_before_claim(fixture):
    approved_digest = fixture.state["expected_digest"]
    # Valid immutable metadata mutation; the attacker recomputes the embedded hash.
    fixture.plan["truth_registry"][0]["reference"] = "changed-but-well-formed-reference"
    fixture.plan["preregistration_sha256"] = runner.preregistration_digest(fixture.plan)
    assert fixture.plan["preregistration_sha256"] != approved_digest
    fixture.write()
    before = fixture.path.read_bytes()
    fixture.sidecar.write_bytes(b"")
    with pytest.raises(AcquisitionIntegrityError, match="external_preregistration_anchor"): fixture.run()
    assert fixture.path.read_bytes() == before and fixture.sidecar.read_bytes() == b""
    assert not fixture.claim.exists()
    assert fixture.state["calls"] == fixture.state["truth"] == fixture.state["score"] == []


@pytest.mark.parametrize("anchor", [None, "", "0"*63, "G"*64, 123])
def test_external_anchor_syntax_rejected_before_claim(fixture, anchor):
    fixture.state["expected_digest"] = anchor
    with pytest.raises(AcquisitionIntegrityError, match="expected_preregistration_digest_syntax"): fixture.run()
    assert not fixture.claim.exists() and not fixture.sidecar.exists()
    assert fixture.state["calls"] == fixture.state["truth"] == []


def test_external_anchor_is_mandatory(fixture):
    with pytest.raises(TypeError, match="expected_preregistration_sha256"):
        runner.run_plan(fixture.path, dependencies=fixture.dependencies,
            truth_loader=lambda reference: b"{}", evaluator=lambda *args: {})
    assert not fixture.claim.exists() and fixture.state["calls"] == []


# New static modules import only safe Python/NumPy production definitions. No
# detector, estimator, legacy harness, renderer or OpenCV is executed below.
from scripts.evaluation import valid_camera_trace as safe_trace
from scripts.evaluation import valid_camera_scoring as safe_scoring
from types import ModuleType
import inspect
import hashlib
import math


def _nodes(relative):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(
        (ROOT/relative).read_text(encoding="utf-8")).body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def test_exact_static_trace_and_metric_ast_parity():
    old = _nodes("scripts/evaluation/run_camera_mixed_depth.py")
    trace = _nodes("scripts/evaluation/valid_camera_trace.py")
    score = _nodes("scripts/evaluation/valid_camera_scoring.py")
    assert all(trace[name] == old[name] for name in runner.TRACE_DEFINITIONS)
    assert all(score[name] == old[name] for name in ("literal_rotation", "camera_metrics"))
    assert len(safe_trace.PINNED_SITES) == 7
    assert list(safe_trace.PINNED_SITES) == runner.score_sites()


@pytest.mark.parametrize("name", ["valid_camera_trace.py", "valid_camera_scoring.py"])
def test_static_modules_have_no_forbidden_import_or_top_level_execution(name):
    tree = ast.parse((ROOT/"scripts/evaluation"/name).read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            imports = [a.name for a in n.names] + ([n.module] if isinstance(n, ast.ImportFrom) else [])
            assert not any(any(part in value.lower() for part in
                ("cv2", "run_unit_face", "run_camera_mixed", "render", "fixture", "scenes"))
                for value in imports if value)
    safe_top_calls = {"Path", "inspect.signature", "threading.Lock", "score_call_sites",
        "CALIBRATION_PATH.read_text"}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue  # Module docstring only.
        assert not isinstance(node, (ast.Expr, ast.If, ast.For, ast.With, ast.Try))
        for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
            label = ast.unparse(call.func)
            # Path(__file__).resolve().parents metadata is the only chained call.
            assert label in safe_top_calls or label == "Path(__file__).resolve"
    assert "scripts.evaluation.run_camera_mixed_depth" not in sys.modules
    assert "scripts.evaluation.run_unit_face_input" not in sys.modules
    assert "cv2" not in sys.modules


def _real_dependencies():
    return runner.Dependencies(safe_trace.trace_rgb, safe_trace, safe_trace.calibration, safe_trace.gui_mask)


def test_safe_actual_module_attestation_no_native_execution():
    runner.validate_dependencies(_real_dependencies())
    runner.validate_scoring(safe_scoring.load_truth, safe_scoring.evaluate_prediction)
    assert safe_trace._runtime_pins() == runner.runtime_pins()
    assert "cv2" not in sys.modules


@pytest.mark.parametrize("target", ["trace", "verify", "helper", "module", "solver", "score", "loader", "defaults"])
def test_independent_source_identity_rejects_tampered_callable_or_origin(monkeypatch, target):
    dependencies = _real_dependencies()
    if target == "trace": monkeypatch.setattr(safe_trace, "trace_rgb", lambda *a, **k: None)
    if target == "verify": monkeypatch.setattr(safe_trace, "_verify_preregistration", lambda: None)
    if target == "helper": monkeypatch.setattr(safe_trace, "_code_signature", lambda code: ())
    if target == "module": monkeypatch.setattr(safe_trace.__spec__, "origin", str(ROOT/"scripts/evaluation/run_camera_mixed_depth.py"))
    if target == "solver": monkeypatch.setattr(safe_trace.calibration, "estimate_camera", lambda *a: None)
    if target == "score": monkeypatch.setattr(safe_scoring, "camera_metrics", lambda *a: {})
    if target == "loader": monkeypatch.setattr(safe_scoring, "load_truth", lambda ref: b"{}")
    if target == "defaults": monkeypatch.setattr(safe_trace.trace_rgb, "__kwdefaults__", {"policy": safe_trace.TracePolicy(max_score_events=1)})
    with pytest.raises(AcquisitionIntegrityError):
        if target in ("score", "loader"):
            runner.validate_scoring(safe_scoring.load_truth, safe_scoring.evaluate_prediction)
        else:
            runner.validate_dependencies(dependencies)


def test_strict_callback_identity_rejects_arbitrary_callable():
    with pytest.raises(AcquisitionIntegrityError, match="scoring_callback_identity"):
        runner.validate_scoring(lambda ref: b"{}", safe_scoring.evaluate_prediction)


def test_sanitized_trace_contract_binding_unbound_tamper_and_drift(monkeypatch):
    safe_trace.clear_trace_contract()
    with pytest.raises(safe_trace.CameraStudyError, match="unbound"):
        safe_trace.trace_rgb(np.zeros((2, 2, 3), np.uint8), "unit")
    contract = {"schema": "CAM-VALID-trace-contract-v1", "external_preregistration_sha256": "a"*64,
        "source_sha256": runner.source_pins(), "runtime": runner.runtime_pins(),
        "score_sites": runner.score_sites(), "policy": runner.SOLVER["trace_policy"]}
    digest = runner.sha(runner.canonical(contract))
    with pytest.raises(safe_trace.CameraStudyError, match="anchor"):
        safe_trace.bind_trace_contract(contract, expected_contract_sha256="b"*64)
    safe_trace.bind_trace_contract(contract, expected_contract_sha256=digest)
    # Caller changes cannot mutate the retained canonical contract.
    contract["policy"]["max_score_events"] = 1
    # SOLVER was aliased by the unit setup; restore the source fixed limit.
    runner.SOLVER["trace_policy"]["max_score_events"] = 120000
    safe_trace._verify_preregistration()
    with monkeypatch.context() as patch:
        patch.setattr(safe_trace, "_runtime_pins", lambda: {})
        with pytest.raises(safe_trace.CameraStudyError, match="runtime_drift"):
            safe_trace._verify_preregistration()
    safe_trace.clear_trace_contract()
    assert safe_trace._CONTRACT_JSON is None
    bad = {**contract, "truth_registry": []}
    with pytest.raises(safe_trace.CameraStudyError, match="schema"):
        safe_trace.bind_trace_contract(bad, expected_contract_sha256=runner.sha(runner.canonical(bad)))


@pytest.fixture
def copied_trace_stub(monkeypatch):
    """Execute exact copied trace with synthetic delegates, explicitly unattested."""
    _REAL_VALIDATE_DEPENDENCIES(_real_dependencies())
    real = safe_trace.calibration
    c = ModuleType("pure_calibration_stub")
    g = ModuleType("pure_gui_stub")
    counts = {"score": 0, "enhance": 0, "detect": 0, "estimate": 0, "mask": 0}
    state = {"refusal": False, "zero": False, "mutate": False, "repeat": False, "thread": False}
    segments = np.asarray([[2., 5.+4*i, 140., 5.+4*i] for i in range(12)], np.float32)
    def score(segments, lengths, focal, yaw, pitch, width, height, threshold, roll=0., principal_point=None):
        counts["score"] += 1
        if state["mutate"]: segments[0, 0] += 1
        return 0. if state["zero"] else .9
    def enhance(gray): counts["enhance"] += 1; return gray.copy()
    def detect(gray, **kwargs): counts["detect"] += 1; return segments.copy()
    def mask(h, w): counts["mask"] += 1; out=np.zeros((h,w),bool); out[:,150:]=True; return out
    c._score=score; c.enhance=enhance; c.detect_segments=detect
    c.refine_optics=lambda *args: None
    c.__dict__.update(np=np, counts=counts, state=state, Result=real.CalibrationResult)
    site = next(s for s in safe_trace.PINNED_SITES if s['tag']=='fine')
    # The stub's actual caller code uses the real fine-call source span. This
    # proves caller/serialization control flow only, not native solver parity.
    pre = "def estimate_camera(segments, width, height):\n    counts['estimate'] += 1\n    best = (0., 150., 20., -10.)\n    lengths = np.hypot(segments[:,2]-segments[:,0], segments[:,3]-segments[:,1])\n"
    source = pre + "\n"*(site['start']-len(pre.splitlines())-1)
    source += "    value = _score(segments, lengths, 150., 20., -10., width, height, 3.)\n"
    source += "    if state['refusal']: return None\n    return Result(150.,20.,-10.,width,height,0 if state['zero'] else 12,len(segments),None if state['zero'] else .3,(0,0,0) if state['zero'] else (4,4,4))\n"
    exec(compile(source, "pure-calibration-caller", "exec"), c.__dict__)
    def pipeline(images):
        result = {}
        for identity, rgb in images.items():
            g.create_gui_mask(*rgb.shape[:2])
            gray = c.enhance(np.zeros(rgb.shape[:2], np.uint8))
            if state["repeat"]: c.enhance(gray)
            if state["thread"]:
                errors=[]
                import threading
                def other():
                    try: c.detect_segments(gray)
                    except Exception as error: errors.append(error)
                thread=threading.Thread(target=other);thread.start();thread.join()
                if errors: raise errors[0]
            found=c.detect_segments(gray)
            result[identity]=c.estimate_camera(found, rgb.shape[1], rgb.shape[0])
        return result
    c.estimate_image_calibrations=pipeline;g.create_gui_mask=mask
    monkeypatch.setattr(safe_trace, "calibration", c)
    monkeypatch.setattr(safe_trace, "gui_mask", g)
    monkeypatch.setattr(safe_trace, "_PINNED_CAMERA_CODE", c.estimate_camera.__code__)
    monkeypatch.setattr(safe_trace, "_PINNED_REFINE_CODE", c.refine_optics.__code__)
    monkeypatch.setattr(safe_trace, "_SCORE_SIGNATURE", inspect.signature(score))
    monkeypatch.setattr(safe_trace, "_verify_preregistration", lambda: None)
    tick=iter(i/1000 for i in range(100000))
    monkeypatch.setattr(safe_trace, "time", SimpleNamespace(perf_counter=lambda: next(tick)))
    originals = [(m,n,getattr(m,n)) for m,n in ((c,'_score'),(c,'enhance'),(c,'detect_segments'),
        (c,'estimate_camera'),(c,'refine_optics'),(g,'create_gui_mask'))]
    return SimpleNamespace(c=c,g=g,counts=counts,state=state,originals=originals,
        image=np.zeros((80,160,3),np.uint8))


@pytest.mark.parametrize("mode", ["returned", "refusal", "zero", "mutate", "repeat", "thread", "cap"])
def test_exact_trace_control_serialization_and_restoration(copied_trace_stub, mode):
    f=copied_trace_stub
    if mode in f.state: f.state[mode]=True
    policy=safe_trace.TracePolicy(max_score_events=0) if mode=='cap' else safe_trace.TracePolicy()
    result=safe_trace.trace_rgb(f.image,'unit',policy=policy)
    expected='ordinary_refusal' if mode=='refusal' else 'returned' if mode in ('returned','zero') else 'trace_refusal'
    assert result['status']==expected
    assert all(getattr(m,n) is old for m,n,old in f.originals)
    assert not safe_trace._TRACE_LOCK.locked()
    if expected!='trace_refusal':
        trace=result['trace']
        assert trace['score_events']==1 and trace['scores'][0]['score']==(0. if mode=='zero' else .9)
        assert trace['trace_event_bytes']==sum(len(safe_trace._json(event).encode()) for event in trace['scores']+trace['stages'])
        assert f.counts==dict(score=1,enhance=1,detect=1,estimate=1,mask=1)
        assert [s['stage'] for s in trace['stages']]==['create_gui_mask','enhance','detect_segments','estimate_camera']
        assert trace['scores'][0]['winner']=={'best_before':[0.,150.,20.,-10.]}
        for digest,meta in trace['arrays'].items():
            assert digest==safe_trace._sha(safe_trace._json(meta))
            assert len(bytes.fromhex(meta['bytes_hex']))==math.prod(meta['shape'])*np.dtype(meta['dtype']).itemsize
        if mode=='zero': assert result['prediction']['median_residual'] is None
    else:
        assert not result['trace']['trace_complete'] and result['prediction'] is None
        if mode=='cap': assert f.counts['score']==0


def test_exact_trace_lock_refuses_without_releasing_other_owner(copied_trace_stub):
    assert safe_trace._TRACE_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(safe_trace.CameraStudyError, match="reentrancy"):
            safe_trace.trace_rgb(copied_trace_stub.image,'unit')
        assert safe_trace._TRACE_LOCK.locked()
    finally: safe_trace._TRACE_LOCK.release()


def _analytic_inputs():
    truth={'schema':safe_scoring.TRUTH_SCHEMA,'identity':'unit','rgb_sha256':'a'*64,
        'shape':[80,160],'focal':150.,'yaw':20.,'pitch':-10.,'roll':0.,'principal':[80.,40.]}
    prediction={'focal_pixels':150.,'yaw_degrees':20.,'pitch_degrees':-10.,'roll_degrees':0.,
        'width':160,'height':80,'principal_point':None,'inliers':0,'median_residual':None}
    return prediction,truth


@pytest.mark.parametrize("quarter", range(4))
def test_scoring_literal_rotation_quarter_gauge_and_absent_principal(quarter):
    p,t=_analytic_inputs();p['yaw_degrees']-=90*quarter
    metrics=safe_scoring.evaluate_prediction(p,t,dict(safe_scoring.GATES))
    assert metrics['diagnostic_pass'] and metrics['horizontal_gauge_quarter_turn']==quarter
    assert metrics['orientation_degrees_mod90']<2e-6
    assert metrics['focal_relative_error']==metrics['principal_error_over_diagonal']==0
    assert np.allclose(safe_scoring.literal_rotation(0,0),np.diag([-1.,1.,1.]),atol=1e-14)
    assert metrics['projection_behind_count']==0


@pytest.mark.parametrize("field,delta", [('yaw_degrees',3.),('pitch_degrees',3.),('roll_degrees',3.)])
def test_scoring_non_equivalent_rotation_fails(field,delta):
    p,t=_analytic_inputs();p[field]+=delta
    value=safe_scoring.evaluate_prediction(p,t,dict(safe_scoring.GATES))
    assert value['orientation_degrees_mod90']==pytest.approx(3.,abs=1e-10)
    assert not value['diagnostic_pass']


@pytest.mark.parametrize("which,factor,passed", [('focal',1-1e-9,True),('focal',1+1e-9,False),
    ('principal',1.,True),('principal',1-1e-9,True),('principal',1+1e-9,False),
    ('orientation',1-1e-7,True),('orientation',1+1e-7,False)])
def test_exact_camera_gate_boundary_sides(which,factor,passed):
    p,t=_analytic_inputs()
    if which=='focal': p['focal_pixels']=t['focal']*(1+.05*factor)
    if which=='principal':
        # Origin chosen so exact displacement avoids subtractive cancellation.
        t['principal']=[0.,0.];p['principal_point']=[.01*math.hypot(160,80)*factor,0.]
    if which=='orientation': p['pitch_degrees']=t['pitch']+factor
    result=safe_scoring.evaluate_prediction(p,t,dict(safe_scoring.GATES))
    assert result['diagnostic_pass'] is passed
    assert result['diagnostic_pass']==all(result[k]<=limit for k,limit in safe_scoring.GATES.items())


def test_exact_focal_boundary_preserves_legacy_binary_arithmetic():
    p,t=_analytic_inputs();p['focal_pixels']=150.*1.05
    result=safe_scoring.evaluate_prediction(p,t,dict(safe_scoring.GATES))
    assert result['focal_relative_error']==abs(1.05-1.)
    assert result['diagnostic_pass'] is (result['focal_relative_error']<=.05)
    # No epsilon is added to reinterpret a rounded boundary as passing.


@pytest.mark.parametrize("field,value", [('shape',[True,160]),('shape',[0,160]),('shape',[80]),
    ('focal',0),('focal',float('nan')),('yaw',float('inf')),('principal',[0,float('nan')]),
    ('identity','wrong'),('rgb_sha256','b'*64),('schema','unknown')])
def test_truth_strict_schema_binding_and_finiteness(field,value):
    p,t=_analytic_inputs();row={'identity':'unit','rgb_sha256':'a'*64,'rgb_shape':[80,160,3]}
    t[field]=value
    with pytest.raises(AcquisitionIntegrityError):safe_scoring.validate_truth(t,row)


def test_scoring_inputs_remain_unmodified_and_copies_do_not_alias(monkeypatch):
    p,t=_analytic_inputs();oldp=runner.canonical(p);oldt=runner.canonical(t)
    real=safe_scoring.camera_metrics
    def mutate(prediction,truth):
        result=real(prediction,truth);prediction['inliers']=999;truth['principal'][0]=999;return result
    monkeypatch.setattr(safe_scoring,'camera_metrics',mutate)
    safe_scoring.evaluate_prediction(p,t,dict(safe_scoring.GATES))
    assert runner.canonical(p)==oldp and runner.canonical(t)==oldt


def test_fixed_loader_reads_only_supplied_bounded_synthetic_bytes(tmp_path):
    _,truth=_analytic_inputs();raw=runner.canonical(truth).encode();path=tmp_path/'unit-truth.json';path.write_bytes(raw)
    reference={'identity':'unit','reference':str(path.resolve()),'sha256':runner.sha(raw)}
    assert safe_scoring.load_truth(reference)==raw
    with pytest.raises(AcquisitionIntegrityError,match='hash'):
        safe_scoring.load_truth({**reference,'sha256':'0'*64})
    path.write_bytes(b'x'*(1024*1024+1))
    with pytest.raises(AcquisitionCapError):safe_scoring.load_truth(reference)
    with pytest.raises(AcquisitionIntegrityError,match='authority'):
        safe_scoring.load_truth({**reference,'reference':'relative-unit.json'})


def test_scoring_identity_gate_before_claim_in_actual_driver(fixture,monkeypatch):
    # Restore the real identity validator; arbitrary legacy unit callbacks fail.
    monkeypatch.setattr(runner,'validate_scoring',_REAL_VALIDATE_SCORING)
    monkeypatch.setattr(safe_scoring,'validate_truth',_REAL_VALIDATE_TRUTH)
    with pytest.raises(AcquisitionIntegrityError,match='scoring_callback_identity'):fixture.run()
    assert not fixture.claim.exists() and fixture.state['calls']==fixture.state['truth']==[]


_REAL_VALIDATE_DEPENDENCIES=runner.validate_dependencies
_REAL_VALIDATE_SCORING=runner.validate_scoring
_REAL_VALIDATE_TRUTH=safe_scoring.validate_truth


def test_pinned_scoring_actual_driver_barrier_all_truth_then_metrics(fixture,monkeypatch):
    f=fixture;truth_paths=[]
    for row,ref in zip(f.plan['registry'],f.plan['truth_registry']):
        _,truth=_analytic_inputs();truth.update(identity=row['identity'],rgb_sha256=row['rgb_sha256'])
        raw=runner.canonical(truth).encode();path=f.path.parent/(row['identity']+'-synthetic-truth.json');path.write_bytes(raw)
        ref.update(reference=str(path.resolve()),sha256=runner.sha(raw));truth_paths.append(path)
    f.plan['preregistration_sha256']=runner.preregistration_digest(f.plan)
    f.state['expected_digest']=f.plan['preregistration_sha256'];f.write()
    monkeypatch.setattr(runner,'validate_scoring',_REAL_VALIDATE_SCORING)
    monkeypatch.setattr(safe_scoring,'validate_truth',_REAL_VALIDATE_TRUTH)
    original_open=Path.open;loaded=[]
    def observe(path,*args,**kwargs):
        if path in truth_paths:
            durable=json.loads(f.path.read_text());assert durable['execution_state']=='PREDICTIONS_VALIDATED'
            assert f.state['readbacks']==24;loaded.append(path)
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',observe)
    evaluated=[]
    def profile(frame,event,arg):
        if event=='call' and frame.f_code is safe_scoring.evaluate_prediction.__code__:
            assert loaded==truth_paths; evaluated.append(True)
    previous=sys.getprofile();sys.setprofile(profile)
    try:
        result=runner.run_plan(f.path,expected_preregistration_sha256=f.state['expected_digest'],
            dependencies=f.dependencies,truth_loader=safe_scoring.load_truth,evaluator=safe_scoring.evaluate_prediction)
    finally:sys.setprofile(previous)
    assert len(evaluated)==24
    assert result['execution_state']=='COMPLETE' and loaded==truth_paths
    assert len(result['evaluations'])==24 and all(v['passed'] for v in result['evaluations'])
    assert f.state['readbacks']==24


def test_runner_rejects_callback_mutation_after_full_barrier(fixture,monkeypatch):
    f=fixture
    # Existing synthetic callback is authorized only by the unit attestation stub.
    def mutation(prediction,truth,gates):
        truth['unit_truth']='mutated';prediction['notes'].append('mutated')
        return dict(diagnostic_pass=True,focal_relative_error=0.,orientation_degrees_mod90=0.,principal_error_over_diagonal=0.)
    with pytest.raises(AcquisitionIntegrityError,match='scoring_input_mutation'):
        runner.run_plan(f.path,expected_preregistration_sha256=f.state['expected_digest'],
            dependencies=f.dependencies,truth_loader=lambda ref: runner.canonical({'unit_truth':ref['identity']}).encode(),
            evaluator=mutation)
    result=json.loads(f.path.read_text())
    assert result['evaluations'] is None and result['seal_validation']['prediction_count']==24


@pytest.mark.parametrize('target', ['score_defaults', 'detector_defaults', 'metric_defaults'])
def test_default_tampering_rejected_from_independent_source(monkeypatch,target):
    if target=='score_defaults':
        monkeypatch.setattr(safe_trace.calibration._score,'__defaults__',(9.,None))
    if target=='detector_defaults':
        monkeypatch.setattr(safe_trace.calibration.detect_segments,'__kwdefaults__',dict(scales=(1.,),minimum_length=1.))
    if target=='metric_defaults':
        monkeypatch.setattr(safe_scoring.literal_rotation,'__defaults__',(90.,))
    with pytest.raises(AcquisitionIntegrityError,match='default_identity'):
        if target=='metric_defaults':runner.validate_scoring(safe_scoring.load_truth,safe_scoring.evaluate_prediction)
        else:runner.validate_dependencies(_real_dependencies())


def test_exact_copied_trace_bootstrap_through_runner_paired_streams(fixture,copied_trace_stub,monkeypatch):
    # Production source origins were independently attested before installing
    # synthetic delegates in copied_trace_stub. No native solver is run. Only
    # this test replaces the verifier/dependency attestation for synthetic code.
    f=fixture;stub=copied_trace_stub
    dependencies=runner.Dependencies(safe_trace.trace_rgb,safe_trace,stub.c,stub.g)
    monkeypatch.setattr(runner,'validate_dependencies',lambda d: runner.require(d is dependencies,'unit-safe-trace'))
    result=runner.run_plan(f.path,expected_preregistration_sha256=f.state['expected_digest'],dependencies=dependencies,
        truth_loader=lambda ref:runner.canonical({'unit_truth':ref['identity']}).encode(),
        evaluator=lambda p,t,g:dict(diagnostic_pass=True,focal_relative_error=0.,orientation_degrees_mod90=0.,principal_error_over_diagonal=0.))
    assert result['execution_state']=='COMPLETE' and f.state['readbacks']==24
    assert len(result['sealed_predictions'])==24 and len(result['evaluations'])==24
    assert stub.counts==dict(score=24,enhance=24,detect=12,estimate=24,mask=12)
    assert all(getattr(m,n) is original for m,n,original in stub.originals)
    assert safe_trace._CONTRACT_JSON is None and not safe_trace._TRACE_LOCK.locked()
    first=support.ValidAcquisitionMemberStore(f.sidecar).read_member(result['sealed_predictions'][0])
    value=json.loads(first)
    assert value['trace_result']['trace']['scores']['record_count']==1
    assert value['trace_result']['trace']['trace_complete']



def test_runtime_binary_hashing_is_streamed_and_source_compile_input_bounded(tmp_path,monkeypatch):
    target=tmp_path/'synthetic-binary.bin';target.write_bytes(b'x'*(2*1024*1024+17))
    original_open=Path.open;reads=[]
    class Observed:
        def __init__(self,stream):self.stream=stream
        def __enter__(self):self.stream.__enter__();return self
        def __exit__(self,*args):return self.stream.__exit__(*args)
        def read(self,size):
            reads.append(size);assert 0<size<=65536;return self.stream.read(size)
    def open(path,*args,**kwargs):
        stream=original_open(path,*args,**kwargs)
        return Observed(stream) if path==target else stream
    monkeypatch.setattr(Path,'open',open)
    assert safe_trace._file_sha256(target)==hashlib.sha256(b'x'*(2*1024*1024+17)).hexdigest()
    assert len(reads)>32 and set(reads)=={65536}
    monkeypatch.setattr(Path,'open',original_open)
    with pytest.raises(safe_trace.CameraStudyError,match='metadata_cap'):safe_trace._source_text(target)


def test_orientation_exact_nominal_boundary_keeps_source_rounding():
    p,t=_analytic_inputs();p['pitch_degrees']+=1.
    value=safe_scoring.evaluate_prediction(p,t,dict(safe_scoring.GATES))
    assert value['orientation_degrees_mod90']==pytest.approx(1.,abs=1e-10)
    assert value['diagnostic_pass'] is (value['orientation_degrees_mod90']<=1.)


def test_last_truth_schema_failure_all_receipts_no_partial_evaluation(fixture,monkeypatch):
    f=fixture;raws={}
    for index,(row,ref) in enumerate(zip(f.plan['registry'],f.plan['truth_registry'])):
        _,truth=_analytic_inputs();truth.update(identity=row['identity'],rgb_sha256=row['rgb_sha256'])
        if index==11:truth['shape']=[81,160]
        raw=runner.canonical(truth).encode();raws[row['identity']]=raw;ref['sha256']=runner.sha(raw)
    f.plan['preregistration_sha256']=runner.preregistration_digest(f.plan)
    f.state['expected_digest']=f.plan['preregistration_sha256'];f.write()
    monkeypatch.setattr(safe_scoring,'validate_truth',_REAL_VALIDATE_TRUTH)
    loaded=[];scored=[]
    def truth(ref):loaded.append(ref['identity']);return raws[ref['identity']]
    def evaluate(*args):scored.append(True);return {}
    with pytest.raises(AcquisitionIntegrityError,match='truth_camera_binding'):
        runner.run_plan(f.path,expected_preregistration_sha256=f.state['expected_digest'],dependencies=f.dependencies,
            truth_loader=truth,evaluator=evaluate)
    assert len(loaded)==12 and not scored
    result=json.loads(f.path.read_text());assert result['evaluations'] is None and result['seal_validation']['prediction_count']==24
