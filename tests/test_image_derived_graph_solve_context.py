"""Focused tests for the DS-IMG-SOLVER-CONTEXT-CAPTURE failure diagnostic.

These cover, in order, the output-neutral default path, the observer's exact
capture of the effective ``solve_component`` call arguments (including empty and
non-empty ``quadrant_constraints``) with serialization/digest replay and tamper
refusal, and the runner's fail-closed observation validation with historical
compatibility.
"""
from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import pytest

from test_image_derived_graph_attribution import _solver_solution
from test_image_derived_lattice_measure import (
    _build_two_frame_session,
    _pose_graph_for_two,
    _result,
)
from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
from mc_photogrammetry.pipeline import image_derived_graph_attribution as attribution_mod
from mc_photogrammetry.pipeline import image_derived_signatures as signatures_mod
from mc_photogrammetry.pipeline import image_derived_lattice_measure as lattice_mod
from mc_photogrammetry.reconstruction import plane_graph


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/run_image_derived_graph_stage.py"
SPEC = importlib.util.spec_from_file_location("graph_stage_solve_context", PATH)
stage = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stage
SPEC.loader.exec_module(stage)


def _matched_pair():
    return plane_graph.PlaneMatch(
        "f0", "f1", 0, 0, 1., 1., 0, 5., 1., .1, 16, None, .9, 0, 0)


def _abstaining():
    return _solver_solution(
        "plane_integer_residual",
        resultants={0: 0.5, 1: 0.95, 2: 0.95},
        residuals={0: 0.1, 1: 0.1, 2: 0.1},
        alternatives=(plane_graph.ComponentAlternative(
            (0, 0), 1.0, (0.0, 0.0, 0.0), (2, 3, 4)),))


def _recording_solver(calls):
    """Record the exact kwargs of the single solve call, then abstain."""
    def solver(readings, matches, **kwargs):
        calls.append(kwargs)
        return _abstaining()
    return solver


def _problem_inputs():
    patch = pytest.MonkeyPatch()
    try:
        session, _pose, lattice = _result(patch)
        signatures = signatures_mod.derive_image_derived_signatures(
            session=session, lattice_result=lattice)
    finally:
        patch.undo()
    return {"session": session, "lattice_result": lattice,
            "signatures_result": signatures}


def _two_frame_inputs():
    """The pinned two-frame room fixture, so non-empty constraints can exist."""
    patch = pytest.MonkeyPatch()
    try:
        session = _build_two_frame_session(patch)
        lattice = lattice_mod.measure_image_derived_lattice(
            session=session, pose_graph=_pose_graph_for_two(session))
        assert lattice.status == "assembled"
        signatures = signatures_mod.derive_image_derived_signatures(
            session=session, lattice_result=lattice)
    finally:
        patch.undo()
    return {"session": session, "lattice_result": lattice,
            "signatures_result": signatures}


# ---------------------------------------------------------------------------
# Default-path output neutrality: nothing new is allocated without the observer.
# ---------------------------------------------------------------------------

def test_default_path_never_builds_a_solve_context(monkeypatch):
    inputs = _problem_inputs()
    monkeypatch.setattr(plane_graph, "match_plane_graph",
                        lambda *_a, **_k: (_matched_pair(),))
    monkeypatch.setattr(plane_graph, "solve_component",
                        lambda *_a, **_k: _abstaining())

    def forbidden(*_args, **_kwargs):
        raise AssertionError("default path allocated a solve context")

    monkeypatch.setattr(graph_mod, "failed_solve_context", forbidden)
    result = graph_mod.derive_image_derived_graph(**inputs)
    assert result.status == "unavailable"
    assert result.reason == "plane_integer_residual"


def test_observer_is_output_neutral_when_capturing_the_solve_context(monkeypatch):
    inputs = _problem_inputs()
    monkeypatch.setattr(plane_graph, "match_plane_graph",
                        lambda *_a, **_k: (_matched_pair(),))
    calls = []
    monkeypatch.setattr(plane_graph, "solve_component",
                        _recording_solver(calls))
    control = graph_mod.derive_image_derived_graph(**inputs)
    assert len(calls) == 1 and calls[0]["quadrant_constraints"] == ()

    seen = []
    result, attribution = attribution_mod.derive_image_derived_graph_attribution(
        **inputs, solver_failure_observer=seen.append)
    assert len(calls) == 2
    assert result == control
    assert result.result_sha256 == control.result_sha256
    assert len(seen) == 1
    snapshot = seen[0]
    context = snapshot.solve_context
    assert isinstance(context, graph_mod.SolverSolveContext)
    assert context.recorded is True
    # The capture is the exact effective call, not a copied default.
    assert context.integer_tolerance == calls[1]["integer_tolerance"] == 0.15
    assert context.min_resultant == calls[1]["min_resultant"] == 0.90
    assert context.ambiguous_margin == calls[1]["ambiguous_margin"] == 0.02
    assert context.constraints == calls[1]["quadrant_constraints"] == ()
    assert context.frames == tuple(snapshot.frames)
    # The projected attribution record itself is unchanged by the extra capture.
    assert "solve_context" not in json.dumps(
        attribution.solver_failure.to_record(), sort_keys=True)
    record = attribution_mod.solver_failure_snapshot_record(snapshot)
    assert record["solve_context"] == context.to_record()
    assert record["solve_context"]["constraints"] == []
    graph_mod.validate_solver_solve_context(record["solve_context"])


def test_observer_captures_the_exact_solve_call_arguments(monkeypatch):
    inputs = _two_frame_inputs()
    frames = tuple(frame.frame_id for frame in inputs["session"].frames)
    assert frames == ("f0", "f1")
    constraints = (("f0", "f1", 1),)
    monkeypatch.setattr(graph_mod, "_connection_constraints",
                        lambda *_a, **_k: constraints)
    monkeypatch.setattr(plane_graph, "match_plane_graph",
                        lambda *_a, **_k: (_matched_pair(),))
    calls = []
    monkeypatch.setattr(plane_graph, "solve_component",
                        _recording_solver(calls))
    seen = []
    result, _attribution = attribution_mod.derive_image_derived_graph_attribution(
        **inputs, solver_failure_observer=seen.append,
        integer_tolerance=0.1, min_resultant=1.1, ambiguous_margin=0.03)
    assert result.status == "unavailable"
    assert calls[0]["quadrant_constraints"] == constraints
    context = seen[0].solve_context
    assert context.recorded is True
    assert context.constraints == constraints
    assert context.integer_tolerance == 0.1
    assert context.min_resultant == 1.1
    assert context.ambiguous_margin == 0.03
    record = context.to_record()
    assert record["constraints"] == [
        {"left": left, "right": right, "yaw": yaw}
        for left, right, yaw in constraints]
    # Assignment selection is reproducible only if the exact ordered tuple survived.
    assert tuple((row["left"], row["right"], row["yaw"])
                 for row in record["constraints"]) == constraints
    assert context.frames == tuple(seen[0].frames)
    graph_mod.validate_solver_solve_context(record)
    graph_mod.validate_solver_solve_context(record, seen[0].frames)


def test_observer_snapshot_record_omits_solve_context_when_absent():
    snapshot = graph_mod.SolverFailureSnapshot(
        "plane_integer_residual", ("f0", "f1"), (0, 0),
        ((0, 0.5), (1, 0.95)), ((0, 0.1), (1, 0.1)), (0, 0), (2, 3, 4))
    record = attribution_mod.solver_failure_snapshot_record(snapshot)
    assert "solve_context" not in record


@pytest.mark.parametrize("error", [ValueError, RuntimeError])
def test_injected_builder_failure_never_serializes_as_absent_context(
        monkeypatch, error):
    """Lead correction 2026-10-04: a failed build must never look historical.

    An absent ``solve_context`` key is exactly the pre-change no-context artifact
    shape, so a failed capture that degraded to ``None`` would be silently
    indistinguishable from a run that never captured context at all.
    """
    inputs = _problem_inputs()
    monkeypatch.setattr(plane_graph, "match_plane_graph",
                        lambda *_a, **_k: (_matched_pair(),))
    monkeypatch.setattr(plane_graph, "solve_component",
                        lambda *_a, **_k: _abstaining())
    # The default/public path must stay output-neutral even with a broken builder.
    control = graph_mod.derive_image_derived_graph(**inputs)

    def broken_builder(*_args, **_kwargs):
        raise error("injected solve-context builder failure")

    monkeypatch.setattr(graph_mod, "failed_solve_context", broken_builder)
    seen = []
    if error is RuntimeError:
        # An unbounded programming failure fails the observer run outright; it is
        # never swallowed into an absent-context record.
        with pytest.raises(RuntimeError):
            attribution_mod.derive_image_derived_graph_attribution(
                **inputs, solver_failure_observer=seen.append)
        assert seen == []
        assert control.status == "unavailable"
        return
    result, attribution = attribution_mod.derive_image_derived_graph_attribution(
        **inputs, solver_failure_observer=seen.append)
    assert result == control
    assert result.result_sha256 == control.result_sha256
    assert len(seen) == 1
    snapshot = seen[0]
    context = snapshot.solve_context
    assert isinstance(context, graph_mod.SolverSolveContext)
    assert context.recorded is False
    record = attribution_mod.solver_failure_snapshot_record(snapshot)
    assert "solve_context" in record
    assert record["solve_context"]["status"] == "refused"
    assert record["solve_context"]["invalid_fields"] == [
        graph_mod.SOLVER_SOLVE_CONTEXT_BUILD_FAILURE]
    assert record["solve_context"]["constraints"] == []
    assert record["solve_context"]["frames"] == list(snapshot.frames)
    graph_mod.validate_solver_solve_context(
        record["solve_context"], snapshot.frames)
    projected = attribution.solver_failure.to_record()
    assert "solve_context" not in projected
    value, payload, trace, inner = _completed_value(
        raw=record, projected=projected, frame_count=len(snapshot.frames))
    stage._validate_solver_failure_observation(value, payload, trace, inner)


# ---------------------------------------------------------------------------
# Serialization/digest replay and tamper refusal.
# ---------------------------------------------------------------------------

def _context_record(*, constraints=(("f0", "f1", 1),), frames=("f0", "f1")):
    return graph_mod.failed_solve_context(
        0.15, 0.90, 0.02, constraints, frames).to_record()


def _reseal(record):
    body = {key: value for key, value in record.items()
            if key != "record_sha256"}
    if body.get("invalid_fields") == []:
        body.pop("invalid_fields")
    body["record_sha256"] = graph_mod._sha({
        "domain": graph_mod.SOLVER_SOLVE_CONTEXT_DIGEST_DOMAIN,
        "schema": body["schema"], "payload": body})
    return body


def test_solve_context_record_replays_and_digest_is_deterministic():
    first = _context_record()
    second = _context_record()
    assert first == second
    assert first["schema"] == graph_mod.SOLVER_SOLVE_CONTEXT_SCHEMA
    assert first["basis"] == graph_mod.SOLVER_SOLVE_CONTEXT_BASIS
    assert first["status"] == "complete"
    assert first["integer_tolerance"] == 0.15
    assert first["min_resultant"] == 0.90
    assert first["ambiguous_margin"] == 0.02
    assert first["constraints"] == [
        {"left": "f0", "right": "f1", "yaw": 1}]
    assert first["frames"] == ["f0", "f1"]
    assert "invalid_fields" not in first
    graph_mod.validate_solver_solve_context(first)
    # The digest is bound to the exact ordered constraint tuple.
    changed = _context_record(constraints=(("f0", "f1", 2),))
    assert changed["record_sha256"] != first["record_sha256"]


def test_solve_context_digest_binds_the_ordered_constraint_tuple():
    ordered = _context_record(
        constraints=(("f0", "f1", 1), ("f1", "f2", 3)), frames=("f0", "f1", "f2"))
    reversed_rows = _reseal({
        **ordered,
        "constraints": list(reversed(ordered["constraints"])),
        "frames": ["f0", "f1", "f2"]})
    # The same rows in a different order reseal to a different digest, so the
    # exact ordered tuple -- what the solver actually receives -- is what binds.
    assert reversed_rows["record_sha256"] != ordered["record_sha256"]
    graph_mod.validate_solver_solve_context(reversed_rows)


@pytest.mark.parametrize("case", [
    "digest", "nonhex", "schema", "status", "extra", "numeric", "substitute",
    "foreign_frame", "missing_numeric", "unknown_invalid",
])
def test_solve_context_validation_is_fail_closed(case):
    record = _context_record()
    if case == "digest":
        record["min_resultant"] = 0.5
    elif case == "nonhex":
        record["record_sha256"] = "z" * 64
    elif case == "schema":
        record["schema"] = "mc-photogrammetry-solver-solve-context-v9"
        record = _reseal(record)
    elif case == "status":
        record["status"] = "unknown"
        record = _reseal(record)
    elif case == "extra":
        record["extra"] = 1
        record = _reseal(record)
    elif case == "numeric":
        record["min_resultant"] = None
        record = _reseal(record)
    elif case == "substitute":
        record["constraints"] = [{"left": "f0", "right": "f1", "yaw": 2}]
        # A substitution changes the ordered tuple, so the unaltered digest no
        # longer replays; a re-sealed context is a different, honest capture.
        assert _reseal(record)["record_sha256"] != _context_record()["record_sha256"]
        return
    elif case == "foreign_frame":
        record["constraints"] = [{"left": "f0", "right": "f9", "yaw": 1}]
        record = _reseal(record)
    elif case == "missing_numeric":
        del record["ambiguous_margin"]
        record = _reseal(record)
    elif case == "unknown_invalid":
        record["invalid_fields"] = ["not_a_field"]
        record["status"] = "refused"
        record = _reseal(record)
    with pytest.raises(ValueError):
        graph_mod.validate_solver_solve_context(record)


def test_declared_refusal_replays_and_cannot_claim_completeness():
    # A malformed constraint tuple yields one bounded refused capture.
    refused = graph_mod.failed_solve_context(
        0.15, 0.90, 0.02, "not-a-tuple", ("f0", "f1"))
    assert refused.recorded is False
    record = refused.to_record()
    assert record["status"] == "refused"
    assert record["constraints"] == []
    assert record["invalid_fields"] == ["constraints"]
    graph_mod.validate_solver_solve_context(record)

    claimed = _reseal({**record, "status": "complete"})
    with pytest.raises(ValueError):
        graph_mod.validate_solver_solve_context(claimed)


def test_empty_constraints_round_trip_is_a_complete_capture():
    record = _context_record(constraints=())
    assert record["status"] == "complete"
    assert record["constraints"] == []
    assert "invalid_fields" not in record
    assert json.loads(json.dumps(record)) == record
    graph_mod.validate_solver_solve_context(record)


def test_refused_solve_context_is_total_and_replays():
    # The explicit build-failure fallback must itself never raise and must
    # produce a replayable refused record, even for hostile thresholds.
    refused = graph_mod.refused_solve_context(
        0.15, float("nan"), True, ("f0", "f1"))
    record = refused.to_record()
    assert record["status"] == "refused"
    assert record["invalid_fields"] == [
        graph_mod.SOLVER_SOLVE_CONTEXT_BUILD_FAILURE]
    assert record["min_resultant"] == {
        "unavailable": graph_mod.SOLVER_SOLVE_CONTEXT_INVALID,
        "field": "min_resultant", "type": "float"}
    # ``bool`` is never a number, so it is an implied invalid field too.
    assert record["ambiguous_margin"]["field"] == "ambiguous_margin"
    assert record["frames"] == ["f0", "f1"]
    graph_mod.validate_solver_solve_context(record, ("f0", "f1"))
    # A refused build can never be resealed as a complete capture.
    with pytest.raises(ValueError):
        graph_mod.validate_solver_solve_context(
            _reseal({**record, "status": "complete"}))


# ---------------------------------------------------------------------------
# Runner observation validation: fail closed, but historically compatible.
# ---------------------------------------------------------------------------

def _observation(record, *, frames=("image-0000", "image-0001", "image-0002")):
    raw = {"schema": attribution_mod.SOLVER_FAILURE_OBSERVATION_SCHEMA,
           "solver_id": attribution_mod.SOLVER_ID, "reason": "plane_integer_residual",
           "frames": list(frames), "quadrants": [0, 0, 0],
           "axis_resultants": [{"axis": 0, "value": 0.5}],
           "axis_residuals": [{"axis": 0, "value": 0.1}],
           "best_alternative_quadrants": [0, 0, 0],
           "best_alternative_axis_observations": [2, 0, 0],
           "invalid_field": None}
    if record is not None:
        raw["solve_context"] = record
    return raw


def _completed_value(*, raw, projected, frame_count=3):
    observation = {"schema": stage.OBSERVATION_SCHEMA,
                   "frame_count": frame_count, "raw": raw, "projected": projected,
                   "binding": {"input_sha256": ["a" * 64],
                               "entry_sha256": "e" * 64,
                               "lattice_sha256": "2" * 64,
                               "signatures_sha256": "3" * 64,
                               "graph_sha256": "c" * 64}}
    body = dict(observation)
    observation["observation_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8")).hexdigest()
    value = {"inputs": [{"sha256": "a" * 64}], "entry_sha256": "e" * 64,
             "lattice_sha256": "2" * 64, "signatures_sha256": "3" * 64,
             "solver_failure_observation": observation}
    payload = {"lattice_sha256": "2" * 64, "signatures_sha256": "3" * 64}
    trace = {"solver_failure": projected}
    return value, payload, trace, "c" * 64


def test_runner_validates_context_bound_to_the_same_failure_snapshot():
    snapshot = graph_mod.SolverFailureSnapshot(
        "plane_integer_residual", ("image-0000", "image-0001", "image-0002"),
        (0, 0, 0), ((0, 0.5), (1, 0.95)), ((0, 0.1), (1, 0.1)), (0, 0), (2, 0, 0),
        solve_context=graph_mod.failed_solve_context(
            0.15, 0.90, 0.02, (("image-0000", "image-0001", 1),),
            ("image-0000", "image-0001", "image-0002")))
    record = attribution_mod.solver_failure_snapshot_record(snapshot)
    assert "solve_context" in record
    projected = attribution_mod.solver_failure_diagnostic(
        snapshot, 0.90, 0.15, 0.02).to_record()
    value, payload, trace, inner = _completed_value(
        raw=_observation(record["solve_context"]), projected=projected)
    stage._validate_solver_failure_observation(value, payload, trace, inner)

    tampered = _observation(record["solve_context"])
    tampered["solve_context"]["constraints"] = [
        {"left": "image-0000", "right": "image-0001", "yaw": 2}]
    value, payload, trace, inner = _completed_value(
        raw=tampered, projected=projected)
    with pytest.raises(ValueError):
        stage._validate_solver_failure_observation(value, payload, trace, inner)


def test_runner_validates_observer_context_through_the_envelope(monkeypatch):
    """The real observer path stays bound from capture to envelope replay.

    The raw snapshot record and the projected failure must come from the same
    failing solve, and the context must stay bound to that raw snapshot's own
    frame universe inside the observation envelope.
    """
    inputs = _two_frame_inputs()
    frames = tuple(frame.frame_id for frame in inputs["session"].frames)
    assert frames == ("f0", "f1")
    constraints = (("f0", "f1", 1),)
    monkeypatch.setattr(graph_mod, "_connection_constraints",
                        lambda *_a, **_k: constraints)
    monkeypatch.setattr(plane_graph, "match_plane_graph",
                        lambda *_a, **_k: (_matched_pair(),))
    monkeypatch.setattr(plane_graph, "solve_component",
                        lambda *_a, **_k: _abstaining())
    seen = []
    result, attribution = attribution_mod.derive_image_derived_graph_attribution(
        **inputs, solver_failure_observer=seen.append)
    assert result.status == "unavailable"
    assert len(seen) == 1
    snapshot = seen[0]
    raw = attribution_mod.solver_failure_snapshot_record(snapshot)
    assert raw["frames"] == list(frames)
    assert raw["solve_context"]["frames"] == list(frames)
    assert raw["solve_context"]["constraints"] == [
        {"left": "f0", "right": "f1", "yaw": 1}]
    projected = attribution.solver_failure.to_record()
    value, payload, trace, inner = _completed_value(
        raw=raw, projected=projected, frame_count=len(frames))
    stage._validate_solver_failure_observation(value, payload, trace, inner)

    # A substituted constraint changes the capture's own digest, so the
    # unaltered record can never replay as the original call.
    substituted = dict(raw)
    substituted["solve_context"] = {
        **raw["solve_context"],
        "constraints": [{"left": "f0", "right": "f1", "yaw": 2}]}
    value, payload, trace, inner = _completed_value(
        raw=substituted, projected=projected, frame_count=len(frames))
    with pytest.raises(ValueError):
        stage._validate_solver_failure_observation(value, payload, trace, inner)

    # A re-sealed context whose frame universe is reordered is refused against
    # the raw snapshot's own ordered universe, not merely by a digest mismatch.
    reordered = _reseal({**raw["solve_context"], "frames": list(reversed(frames))})
    rebound = dict(raw)
    rebound["solve_context"] = reordered
    value, payload, trace, inner = _completed_value(
        raw=rebound, projected=projected, frame_count=len(frames))
    with pytest.raises(ValueError):
        stage._validate_solver_failure_observation(value, payload, trace, inner)

    # The envelope binds the projected failure too: a substituted projection is
    # refused even though the raw snapshot and context are intact.
    value, payload, trace, inner = _completed_value(
        raw=raw, projected=projected, frame_count=len(frames))
    trace["solver_failure"] = {**projected, "reason": "substituted"}
    with pytest.raises(ValueError):
        stage._validate_solver_failure_observation(value, payload, trace, inner)


def test_runner_refuses_foreign_context_frames():
    projected = {"reason": "plane_integer_residual"}
    foreign = graph_mod.failed_solve_context(
        0.15, 0.90, 0.02, (("image-0000", "image-0009", 1),),
        ("image-0000", "image-0009")).to_record()
    # Bind it to a snapshot whose own universe is different: the runner must
    # reject the mismatch rather than trust the capture's self-declared frames.
    value, payload, trace, inner = _completed_value(
        raw=_observation(foreign), projected=projected)
    with pytest.raises(ValueError):
        stage._validate_solver_failure_observation(value, payload, trace, inner)

    # A context that merely reorders the same frame universe is also foreign to
    # the snapshot's own order, which the solver's assignment enumeration uses.
    reordered = _context_record(
        constraints=(("image-0000", "image-0001", 1),),
        frames=("image-0000", "image-0001", "image-0002"))
    reordered["frames"] = ["image-0001", "image-0000", "image-0002"]
    reordered = _reseal(reordered)
    value, payload, trace, inner = _completed_value(
        raw=_observation(reordered), projected=projected)
    with pytest.raises(ValueError):
        stage._validate_solver_failure_observation(value, payload, trace, inner)


def test_runner_accepts_historical_artifact_without_solve_context():
    snapshot = graph_mod.SolverFailureSnapshot(
        "plane_integer_residual", ("image-0000", "image-0001", "image-0002"),
        (0, 0, 0), ((0, 0.5), (1, 0.95)), ((0, 0.1), (1, 0.1)), (0, 0), (2, 0, 0))
    record = attribution_mod.solver_failure_snapshot_record(snapshot)
    assert "solve_context" not in record  # pre-change shape is preserved
    projected = attribution_mod.solver_failure_diagnostic(
        snapshot, 0.90, 0.15, 0.02).to_record()
    value, payload, trace, inner = _completed_value(raw=record, projected=projected)
    stage._validate_solver_failure_observation(value, payload, trace, inner)


def test_runner_fails_closed_on_malformed_context():
    projected = {"reason": "plane_integer_residual"}
    record = _context_record(
        constraints=(("image-0000", "image-0001", 1),),
        frames=("image-0000", "image-0001", "image-0002"))
    record = _reseal({**record, "frames": ["image-0000", "image-0001",
                                           "image-0002"]})

    broken = []
    no_digest = dict(record)
    no_digest["record_sha256"] = "not-a-digest"
    broken.append(no_digest)
    missing = dict(record)
    missing.pop("record_sha256")
    broken.append(missing)
    broken.append(_reseal({**record,
                           "schema": "mc-photogrammetry-solver-solve-context-v9"}))
    broken.append(_reseal({**record, "extra": 1}))
    for candidate in broken:
        value, payload, trace, inner = _completed_value(
            raw=_observation(candidate), projected=projected)
        with pytest.raises(ValueError):
            stage._validate_solver_failure_observation(
                value, payload, trace, inner)
