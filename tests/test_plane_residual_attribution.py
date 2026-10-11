from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
import attribute_plane_residuals as subject  # noqa: E402


ARTIFACT = REPO / subject.ARTIFACT_RELATIVE


def test_r8_all_nested_seals_and_cross_bindings_validate():
    artifact = subject.validate_pinned_artifact(ARTIFACT)
    observation = artifact["solver_failure_observation"]
    assert observation["phase_capture"]["record_sha256"] == subject.PHASE_CAPTURE_SHA256
    assert (observation["solver_input_bindings"]["record_sha256"]
            == subject.SOLVER_INPUT_BINDINGS_SHA256)
    assert observation["projected"]["active_edge_aliases"]["active_edge_count"] == 10


def test_full_set_recomposition_matches_all_captured_axes():
    phase = subject.validate_pinned_artifact(ARTIFACT)["solver_failure_observation"][
        "phase_capture"]
    result = subject.recompute_full_set(phase)
    assert [row["axis"] for row in result] == [0, 1, 2]
    assert [row["face_count"] for row in result] == [8, 6, 6]
    assert [(round(row["resultant"], 6), round(row["residual"], 6))
            for row in result] == [(0.789357, 0.093065), (0.013942, 0.247102),
                                   (0.876999, 0.067258)]


def test_leave_one_edge_out_is_ordered_and_deterministic():
    phase = subject.validate_pinned_artifact(ARTIFACT)["solver_failure_observation"][
        "phase_capture"]
    first = subject.leave_one_edge_out(phase)
    second = subject.leave_one_edge_out(phase)
    assert first == second
    assert len(first) == 10
    assert [row["match_index"] for row in first] == sorted(
        row["match_index"] for row in first)
    assert all(row["remaining_face_count"] >= 4 for row in first)
    assert [(row["match_index"], round(row["resultant"], 6),
             round(row["residual"], 6)) for row in first] == [
        (4, 0.916964, 0.044599), (5, 0.016052, 0.247102),
        (9, 0.844710, 0.075524), (16, 0.767440, 0.100918),
        (17, 0.786695, 0.096919), (19, 0.016969, 0.247102),
        (23, 0.938134, 0.050726), (26, 0.901812, 0.050726),
        (33, 0.011848, 0.247102), (37, 0.751978, 0.098947)]


def test_changed_phase_endpoint_fails_cross_binding():
    artifact = subject.validate_pinned_artifact(ARTIFACT)
    observation = artifact["solver_failure_observation"]
    phase = copy.deepcopy(observation["phase_capture"])
    inputs = observation["solver_input_bindings"]
    aliases = observation["projected"]["active_edge_aliases"]
    phase["faces"][0][5] += 0.01
    _, plane_graph = subject._load_production_validators(REPO / "src")
    with pytest.raises(subject.ArtifactError, match="phase_endpoint_value_binding"):
        subject._validate_edge_phase_bindings(phase, inputs, aliases, plane_graph)


def test_outer_file_tamper_is_rejected(tmp_path):
    payload = bytearray(ARTIFACT.read_bytes())
    payload[0] ^= 1
    changed = tmp_path / "observation.json"
    changed.write_bytes(payload)
    with pytest.raises(subject.ArtifactError, match="artifact_file_sha256_mismatch"):
        subject.validate_pinned_artifact(changed)


@pytest.mark.parametrize(
    ("cap_name", "cap_value", "reason"),
    [("MAX_ARTIFACT_BYTES", 1, "artifact_byte_cap"),
     ("MAX_ACTIVE_EDGES", 9, "active_edge_cap"),
     ("MAX_PHASE_FACES", 19, "phase_face_cap"),
     ("MAX_FRAMES", 2, "frame_cap")],
)
def test_all_diagnostic_caps_fail_closed(monkeypatch, cap_name, cap_value, reason):
    monkeypatch.setattr(subject, cap_name, cap_value)
    with pytest.raises(subject.ArtifactError, match=reason):
        subject.validate_pinned_artifact(ARTIFACT)


def test_analysis_preserves_original_artifact_and_never_calls_solver(monkeypatch):
    before = ARTIFACT.read_bytes()
    before_sha = hashlib.sha256(before).hexdigest()
    artifact = subject.validate_pinned_artifact(ARTIFACT)
    graph_sha = artifact["graph_sha256"]
    trace_sha = artifact["trace_sha256"]
    _, plane_graph = subject._load_production_validators(REPO / "src")

    def forbidden_solver(*_args, **_kwargs):
        raise AssertionError("offline attribution must not rerun solve_component")

    monkeypatch.setattr(plane_graph, "solve_component", forbidden_solver)
    result = subject.analyze(ARTIFACT)
    after = ARTIFACT.read_bytes()
    assert result["status"] == "complete"
    assert result["solver_rerun"] is False and result["refit_performed"] is False
    assert hashlib.sha256(after).hexdigest() == before_sha
    assert artifact["graph_sha256"] == graph_sha
    assert artifact["trace_sha256"] == trace_sha


def digest(value):
    return hashlib.sha256(subject._canonical(value)).hexdigest()


def reseal_container(monkeypatch, artifact):
    """Lift only enclosing pins in memory to exercise each inner replay gate."""
    observation = artifact["solver_failure_observation"]
    body = dict(observation)
    body.pop("observation_sha256", None)
    observation["observation_sha256"] = digest(body)
    monkeypatch.setattr(subject, "OBSERVATION_SHA256", observation["observation_sha256"])
    body = dict(artifact)
    body.pop("artifact_sha256", None)
    artifact["artifact_sha256"] = digest(body)
    raw = subject._canonical(artifact)
    monkeypatch.setattr(subject, "ARTIFACT_PAYLOAD_SHA256", artifact["artifact_sha256"])
    monkeypatch.setattr(subject, "ARTIFACT_SHA256", hashlib.sha256(raw).hexdigest())
    original_read = Path.read_bytes
    def read(path):
        return raw if path == ARTIFACT else original_read(path)
    monkeypatch.setattr(Path, "read_bytes", read)


@pytest.mark.parametrize("nested", ["graph", "trace"])
def test_resealed_outer_container_cannot_hide_changed_inner_payload(monkeypatch, nested):
    artifact = json.loads(ARTIFACT.read_bytes())
    if nested == "graph":
        artifact["graph"]["payload"]["detail"] += "resealed nested tamper"
    else:
        artifact["trace"]["detail"] += "resealed nested tamper"
    reseal_container(monkeypatch, artifact)
    monkeypatch.setattr(subject, "recompute_full_set", lambda *_: pytest.fail("metrics before digest"))
    monkeypatch.setattr(subject, "leave_one_edge_out", lambda *_: pytest.fail("LOO before digest"))
    with pytest.raises(subject.ArtifactError, match=nested+"_payload_digest_mismatch"):
        subject.analyze(ARTIFACT)


@pytest.mark.parametrize("tamper", ["inactive_b", "inactive_order"])
def test_resealed_input_record_replays_full_ordered_identity_including_inactive(monkeypatch, tamper):
    artifact = json.loads(ARTIFACT.read_bytes())
    inputs = artifact["solver_failure_observation"]["solver_input_bindings"]
    inactive = [row[0] for row in inputs["edges"] if row[4] is False]
    assert len(inactive) >= 2
    if tamper == "inactive_b":
        edge = inputs["edges"][inactive[0]]
        readings = {row[0]: row for row in inputs["readings"]}
        left = edge[1]
        alternatives = readings[left[0]][5]
        replacement = (left[1]+1) % len(alternatives)
        left[1:5] = [replacement, *alternatives[replacement]]
    else:
        i, j = inactive[:2]
        inputs["edges"][i], inputs["edges"][j] = inputs["edges"][j], inputs["edges"][i]
        inputs["edges"][i][0], inputs["edges"][j][0] = i, j
    body = dict(inputs)
    body.pop("record_sha256")
    inputs["record_sha256"] = digest({"domain": "solver_input_bindings",
        "schema": inputs["schema"], "payload": body})
    # This is a structurally valid newly sealed record retaining a stale full
    # input-identity claim. The old generic validator alone accepts that claim.
    graph_mod, _ = subject._load_production_validators(REPO / "src")
    graph_mod.validate_solver_input_bindings(inputs)
    assert subject._ordered_match_identity_digest(inputs) != subject.INPUT_MATCH_IDENTITY_SHA256
    monkeypatch.setattr(subject, "SOLVER_INPUT_BINDINGS_SHA256", inputs["record_sha256"])
    reseal_container(monkeypatch, artifact)
    monkeypatch.setattr(subject, "recompute_full_set", lambda *_: pytest.fail("metrics before identity"))
    with pytest.raises(subject.ArtifactError, match="ordered_match_identity_digest_mismatch"):
        subject.analyze(ARTIFACT)


def test_ordered_identity_recomputed_from_all_38_rows_and_order_refuses():
    inputs = subject.validate_pinned_artifact(ARTIFACT)["solver_failure_observation"]["solver_input_bindings"]
    assert len(inputs["edges"]) == 38
    assert subject._ordered_match_identity_digest(inputs) == subject.INPUT_MATCH_IDENTITY_SHA256
    changed = copy.deepcopy(inputs)
    changed["edges"][0], changed["edges"][1] = changed["edges"][1], changed["edges"][0]
    with pytest.raises(subject.ArtifactError, match="ordered_match_identity_order"):
        subject._ordered_match_identity_digest(changed)


@pytest.mark.parametrize("relative", sorted(subject.SOURCE_PINS))
def test_each_documented_source_pin_enforced_before_artifact_or_metrics(monkeypatch, relative):
    changed = REPO / relative
    original_read = Path.read_bytes
    def read(path):
        if path == changed:
            return original_read(path)+b"upstream drift"
        if path == ARTIFACT:
            pytest.fail("artifact opened before source gate")
        return original_read(path)
    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(subject, "_load_production_validators", lambda *_: pytest.fail("helper loaded before source gate"))
    with pytest.raises(subject.ArtifactError, match="source_sha256_mismatch"):
        subject.analyze(ARTIFACT)


def test_missing_source_and_wrong_loaded_module_path_refuse(monkeypatch):
    changed = REPO / next(iter(subject.SOURCE_PINS))
    original_read = Path.read_bytes
    def read(path):
        if path == changed: raise FileNotFoundError("source disappeared")
        return original_read(path)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", read)
        with pytest.raises(subject.ArtifactError, match="source_unreadable"):
            subject.validate_pinned_artifact(ARTIFACT)
    graph_mod, _ = subject._load_production_validators(REPO / "src")
    monkeypatch.setattr(graph_mod, "__file__", str(REPO / "foreign-copy.py"))
    with pytest.raises(subject.ArtifactError, match="loaded_source_path_mismatch"):
        subject.validate_pinned_artifact(ARTIFACT)


def test_direct_recomposition_and_loo_also_enforce_source_pins(monkeypatch):
    phase = subject.validate_pinned_artifact(ARTIFACT)["solver_failure_observation"]["phase_capture"]
    pins = dict(subject.SOURCE_PINS)
    pins[next(iter(pins))] = "0"*64
    monkeypatch.setattr(subject, "SOURCE_PINS", pins)
    for replay in (subject.recompute_full_set, subject.leave_one_edge_out):
        with pytest.raises(subject.ArtifactError, match="source_sha256_mismatch"):
            replay(phase)
