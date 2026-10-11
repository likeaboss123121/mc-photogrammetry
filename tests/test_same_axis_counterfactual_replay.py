"""DS-IMG-SAMEAXIS-R10-REPLAY-BASELINE-FAILCLOSED-RETRY-20261004 focused tests.

Evaluation-only, read-only tests for
``scripts/evaluation/replay_same_axis_counterfactual.py``.  They load the sealed
R10 observation (never an image), rebuild the serialized provenance, and prove:

* file/record/identity/trace digest and schema refusals fail closed;
* the 38-row trace/input join and the reading-table remap are bijective;
* the pinned ``[0, 3, 1]`` baseline reproduces R10 exactly;
* the known unpinned control mismatch builds and solves **zero** treatment arms
  (no C1-C5 solver call) and yields ``treatments.arms == {}``; and
* the candidate mapping/substitution helpers and payload seal behave exactly.

DS-IMG-SOLVER-CONTEXT-R11-REPLAY-CONTROL-20261004 extends the same focused,
read-only suite to the R11 exact-context control: the R11 ``solve_context``
capture must replay through the production validator, bind its digest to the
pinned value and bind its frame universe to the R10 reconstruction, R10/R11
shared graph/trace/source inputs must reconcile, and exactly one unpruned
context-pinned solver call must reproduce R10 with no treatment arm.

DS-IMG-SOLVER-CONTEXT-R11-C1-C5-COUNTERFACTUAL-20261004 extends it again: the
accepted sealed R11 baseline replay is pinned three ways (file, seal, control),
reconciled against freshly loaded R10/R11 artifacts, and then exactly the frozen
C1-C5 one-for-one substitutions run once each under the identical captured
thresholds/ordered constraints.  The accepted baseline is never rerun, and every
refusal (bad pin, bad seal, tampered control, unfrozen substitution) fires
before any arm is solved.

DS-IMG-SOLVER-CONTEXT-R11-AXES02-C3-COUNTERFACTUAL-20261004 extends it once
more: the accepted sealed C1-C5 replay is pinned by file and seal, consumed only
for its sealed C3 outcome, and every frozen axes-0/2 substitution is validated
(row/group/survivor/duplicate-edge/world-axis) before exactly eight combined
arms run once each under the identical captured context.  The baseline control
and the C3 arm are never rerun, and every pin or semantic refusal fires before a
single arm is solved.

No test writes, deletes or creates a file: pytest is used for assertions only.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = ROOT / "scripts" / "evaluation" / "replay_same_axis_counterfactual.py"
ARTIFACT_PATH = (
    ROOT / ".tmp-geom-prefilter-capture-r10-20261004" / "observation.json")
R11_ARTIFACT_PATH = (
    ROOT / ".tmp-geom-solver-context-r11r2-20261004" / "observation.json")
R11_BASELINE_PATH = (
    ROOT / ".tmp-geom-counterfactual-r11-20261004" / "counterfactual_replay.json")
R11_C1_C5_PATH = (
    ROOT / ".tmp-geom-counterfactual-r11-arms-20261004"
    / "counterfactual_replay.json")
R11_AXES02_PATH = (
    ROOT / ".tmp-geom-counterfactual-r11-axes02-20261004"
    / "counterfactual_replay.json")
R11_AXES02_EXHAUSTIVE_PATH = (
    ROOT / ".tmp-geom-counterfactual-r11-axes02-exhaustive-20261004"
    / "counterfactual_replay.json")

_SPEC = importlib.util.spec_from_file_location(
    "same_axis_counterfactual_replay", HARNESS_PATH)
mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)


@pytest.fixture(scope="module")
def artifact():
    with open(ARTIFACT_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def record(artifact):
    return artifact["same_axis_candidates"]


@pytest.fixture(scope="module")
def trace(artifact):
    return artifact["trace"]


@pytest.fixture(scope="module")
def r11_artifact():
    with open(R11_ARTIFACT_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def r11_baseline():
    with open(R11_BASELINE_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def r11_c1_c5():
    with open(R11_C1_C5_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def r11_axes02():
    with open(R11_AXES02_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def r11_axes02_exhaustive():
    with open(R11_AXES02_EXHAUSTIVE_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _selected_positions(pairs):
    return [index for index, pair in enumerate(pairs)
            if isinstance(pair, dict) and pair.get("selected") is True]


# ---------------------------------------------------------------------------
# artifact / file / record digest and schema refusal
# ---------------------------------------------------------------------------

def test_replay_refuses_wrong_file_digest():
    with pytest.raises(mod.ReplayValidationError, match="SHA-256"):
        mod.replay_artifact(ARTIFACT_PATH, expected_sha256="0" * 64)


def test_replay_refuses_missing_artifact():
    missing = ROOT / "no-such-r10-replay-artifact" / "observation.json"
    with pytest.raises(mod.ReplayValidationError, match="absent"):
        mod.replay_artifact(missing)


def test_record_digest_tamper_is_refused(record, trace):
    tampered = copy.deepcopy(record)
    tampered["record_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        mod.validate_record_bindings(tampered, trace)


def test_record_identity_digest_tamper_is_refused(record, trace):
    tampered = copy.deepcopy(record)
    tampered["input_match_identity_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        mod.validate_record_bindings(tampered, trace)


def test_record_schema_tamper_is_refused(record, trace):
    tampered = copy.deepcopy(record)
    tampered["schema"] = (
        "mc-photogrammetry-same-axis-candidate-capture-v1")
    with pytest.raises(ValueError):
        mod.validate_record_bindings(tampered, trace)


def test_trace_binding_digest_refusal(record, trace):
    bad_assignment = copy.deepcopy(trace)
    bad_assignment["solver_failure"]["active_edge_aliases"][
        "failed_assignment_sha256"] = "f" * 64
    with pytest.raises(mod.ReplayValidationError, match="failed assignment"):
        mod.validate_record_bindings(record, bad_assignment)

    bad_identity = copy.deepcopy(trace)
    bad_identity["solver_failure"]["active_edge_aliases"][
        "input_match_identity_sha256"] = "f" * 64
    with pytest.raises(mod.ReplayValidationError, match="input identity"):
        mod.validate_record_bindings(record, bad_identity)

    absent = copy.deepcopy(trace)
    absent["solver_failure"] = None
    with pytest.raises(mod.ReplayValidationError, match="solver_failure"):
        mod.validate_record_bindings(record, absent)

    incomplete = copy.deepcopy(trace)
    incomplete["solver_failure"]["active_edge_aliases"]["status"] = "refused"
    with pytest.raises(mod.ReplayValidationError, match="aliases"):
        mod.validate_record_bindings(record, incomplete)


# ---------------------------------------------------------------------------
# bijective trace/input joins and reading-table remap
# ---------------------------------------------------------------------------

def test_trace_selected_rows_join_is_bijective(record, trace):
    joined = mod.select_trace_rows(record, trace)
    assert len(joined) == 38
    assert len(record["input_identities"]) == 38


def test_trace_join_refuses_duplicate_selected_rows(record, trace):
    bad = copy.deepcopy(trace)
    selected = _selected_positions(bad["pairs"])
    assert len(selected) == 38
    bad["pairs"][selected[1]] = copy.deepcopy(bad["pairs"][selected[0]])
    with pytest.raises(mod.ReplayValidationError, match="unique"):
        mod.select_trace_rows(record, bad)


def test_trace_join_refuses_key_set_mismatch(record, trace):
    bad = copy.deepcopy(trace)
    selected = _selected_positions(bad["pairs"])[0]
    bad["pairs"][selected]["right_plane"] = 999
    with pytest.raises(mod.ReplayValidationError, match="join the identity"):
        mod.select_trace_rows(record, bad)


def test_trace_join_refuses_missing_selected_row(record, trace):
    bad = copy.deepcopy(trace)
    selected = _selected_positions(bad["pairs"])[0]
    bad["pairs"].pop(selected)
    with pytest.raises(mod.ReplayValidationError, match="count"):
        mod.select_trace_rows(record, bad)


def test_identity_rows_must_be_pairwise_unique(record, trace):
    bad = copy.deepcopy(record)
    bad["input_identities"][1] = copy.deepcopy(bad["input_identities"][0])
    with pytest.raises(mod.ReplayValidationError, match="unique"):
        mod.select_trace_rows(bad, trace)


def test_reading_universe_remap_is_bijective(record):
    universe = mod.build_reading_universe(record)
    assert len(universe.readings) == 15
    assert len(universe.canonical_index) == 15
    assert len(universe.reading_by_recorded) == 15
    for frame in universe.frames:
        positions = sorted(
            index for (name, _plane), index in universe.canonical_index.items()
            if name == frame)
        assert positions == list(range(len(positions)))
        recorded = sorted(
            plane for (name, plane) in universe.recorded_plane if name == frame)
        assert positions == list(range(len(recorded)))


def test_reading_universe_refuses_duplicate_frame_plane(record):
    bad = copy.deepcopy(record)
    bad["readings"].append(copy.deepcopy(bad["readings"][0]))
    with pytest.raises(mod.ReplayValidationError, match="repeat"):
        mod.build_reading_universe(bad)


def test_reading_universe_refuses_empty_table(record):
    bad = copy.deepcopy(record)
    bad["readings"] = []
    with pytest.raises(mod.ReplayValidationError, match="empty"):
        mod.build_reading_universe(bad)


# ---------------------------------------------------------------------------
# pinned baseline reproduces R10; unpinned control fails closed
# ---------------------------------------------------------------------------

def test_pinned_baseline_reproduces_r10_and_fails_closed():
    payload = mod.replay_artifact(ARTIFACT_PATH)

    assert payload["schema"] == mod.SCHEMA
    assert payload["status"] == "baseline_unreconstructed"
    assert payload["input"]["file_sha256"] == mod.INPUT_FILE_SHA256

    gate = payload["control_gate"]
    assert gate["passed"] is False
    assert gate["mismatches"]
    assert gate["pinned_baseline_passed"] is True
    assert gate["pinned_baseline_mismatches"] == []

    assert payload["treatments"] == {"interpretable": False, "arms": {}}
    assert payload["control_unpinned"]["assignment"] != [0, 3, 1]

    pinned = payload["pinned_baseline"]
    expected = payload["r10_expected"]
    assert pinned["assignment"] == [0, 3, 1]
    assert pinned["reason"] == "plane_integer_residual"
    assert pinned["accepted"] is False
    assert mod.control_mismatches(pinned, expected, mod.CONTROL_ABS_TOLERANCE) == []

    for axis in ("0", "1", "2"):
        assert math.isclose(pinned["resultants"][axis], expected["resultants"][axis],
                            rel_tol=0.0, abs_tol=1e-12)
        assert math.isclose(pinned["residuals"][axis], expected["residuals"][axis],
                            rel_tol=0.0, abs_tol=1e-12)
    assert pinned["observations"] == expected["observations"]
    for frame, value in expected["translations"].items():
        assert pinned["translations"][frame] == pytest.approx(value, abs=1e-12)

    assert payload["seal_sha256"] == mod.seal_payload(payload)
    assert payload["validation"]["identity_join_bijective"] is True
    assert payload["validation"]["solver_input_binding_cross_check"] == "pass"


def test_unpinned_mismatch_runs_zero_treatment_arms(monkeypatch):
    calls = []
    real_run_arm = mod.run_arm

    def counting_run_arm(*args, **kwargs):
        calls.append(kwargs.get("target"))
        return real_run_arm(*args, **kwargs)

    def forbidden_build_treatments(*args, **kwargs):
        raise AssertionError(
            "build_treatments must not run after a control mismatch")

    monkeypatch.setattr(mod, "run_arm", counting_run_arm)
    monkeypatch.setattr(mod, "build_treatments", forbidden_build_treatments)

    payload = mod.replay_artifact(ARTIFACT_PATH)

    assert payload["status"] == "baseline_unreconstructed"
    assert payload["control_gate"]["passed"] is False
    assert payload["treatments"]["interpretable"] is False
    assert payload["treatments"]["arms"] == {}
    # Exactly the two mandated control runs: pinned baseline then unpinned
    # control.  Any C1-C5 arm would add a third (or more) solver invocation.
    assert calls == [(0, 3, 1), None]
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# candidate mapping / substitution helpers (no solver treatment calls)
# ---------------------------------------------------------------------------

def test_candidate_helpers_are_solver_free(record, trace, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("candidate helpers must not invoke a solver")

    monkeypatch.setattr(mod, "run_arm", forbidden)

    universe = mod.build_reading_universe(record)
    baseline = mod.reconstruct_matches(record, trace, universe)
    assert len(baseline) == 38

    overrides = mod._kept_candidate_overrides(record)
    assert overrides[19][11] == baseline[19].peak
    assert overrides[19][12] == baseline[19].second_peak

    match_203, row_203 = mod.candidate_match(record, universe, 203)
    match_383, row_383 = mod.candidate_match(record, universe, 383)
    match_204, _ = mod.candidate_match(record, universe, 204)
    match_206, _ = mod.candidate_match(record, universe, 206)
    match_384, _ = mod.candidate_match(record, universe, 384)

    assert row_203[0] == 4 and row_203[17] == "dropped"
    assert row_383[0] == 7 and row_383[17] == "dropped"
    assert (match_203.left, match_203.right) == ("image-0000", "image-0002")
    assert (match_383.left, match_383.right) == ("image-0001", "image-0002")
    assert math.isclose(match_203.left_b, 0.25, rel_tol=0.0, abs_tol=0.0)
    assert math.isclose(match_203.right_b, 0.5, rel_tol=0.0, abs_tol=0.0)
    assert math.isclose(match_383.left_b, 0.125, rel_tol=0.0, abs_tol=0.0)
    assert math.isclose(match_383.right_b, 1.0, rel_tol=0.0, abs_tol=0.0)

    single = mod.substitute(baseline, {19: match_203})
    assert len(single) == 38
    assert single[19] is match_203
    assert single[33] is baseline[33]

    joint = mod.substitute(baseline, {19: match_203, 33: match_383})
    assert len(joint) == 38
    assert joint[19] is match_203 and joint[33] is match_383
    assert joint[5] is baseline[5]

    with pytest.raises(mod.ReplayValidationError):
        mod.substitute(baseline, {99: match_203})
    with pytest.raises(mod.ReplayValidationError, match="out of range"):
        mod.candidate_match(record, universe, 448)

    kept_flat = next(
        index for index, row in enumerate(record["candidates"])
        if row[17] == "kept")
    with pytest.raises(mod.ReplayValidationError, match="dropped"):
        mod.candidate_match(record, universe, kept_flat)


# ---------------------------------------------------------------------------
# payload seal
# ---------------------------------------------------------------------------

def test_seal_payload_is_deterministic_and_field_sensitive():
    payload = {"a": 1, "b": [1, 2, 3], "c": {"d": "e"}}
    first = mod.seal_payload(payload)
    assert first == mod.seal_payload(dict(payload))
    assert mod.seal_payload({"a": 1, "b": [1, 2, 4], "c": {"d": "e"}}) != first

    payload["seal_sha256"] = first
    assert mod.seal_payload(payload) == first
    with pytest.raises(ValueError):
        mod.seal_payload({"x": float("nan")})


def test_write_sealed_refuses_existing_root():
    existing = ROOT / "docs" / "unused-r10-replay-output.json"
    with pytest.raises(mod.ReplayValidationError, match="fresh"):
        mod.write_sealed(existing, {"a": 1})


# ---------------------------------------------------------------------------
# R11 exact solve context: digest/frame binding and refusals
# ---------------------------------------------------------------------------

def test_r11_exact_context_binds_the_pinned_digest_and_frames(r11_artifact):
    context = mod.exact_solve_context(
        r11_artifact, expected_frames=("image-0000", "image-0001", "image-0002"))

    assert context.record_sha256 == mod.R11_CONTEXT_SHA256
    assert context.frames == ("image-0000", "image-0001", "image-0002")
    assert context.constraints == (
        ("image-0000", "image-0001", 3),
        ("image-0000", "image-0002", 1))
    assert context.integer_tolerance == 0.15
    assert context.min_resultant == 0.90
    assert context.ambiguous_margin == 0.02
    assert context.call_arguments() == {
        "integer_tolerance": 0.15,
        "min_resultant": 0.90,
        "ambiguous_margin": 0.02,
        "quadrant_constraints": context.constraints,
    }
    assert context.to_report()["constraints"] == [
        {"left": "image-0000", "right": "image-0001", "yaw": 3},
        {"left": "image-0000", "right": "image-0002", "yaw": 1}]


def test_r11_exact_context_binds_to_the_r10_reconstruction_frames(r11_artifact):
    reconstruction = mod.load_r10_reconstruction(ARTIFACT_PATH)
    context = mod.exact_solve_context(
        r11_artifact, expected_frames=reconstruction.universe.frames)
    assert context.frames == reconstruction.universe.frames


def test_r11_exact_context_refuses_a_foreign_frame_universe(r11_artifact):
    with pytest.raises(mod.ReplayValidationError, match="validation"):
        mod.exact_solve_context(
            r11_artifact,
            expected_frames=("image-0000", "image-0001", "image-0003"))


def test_r11_exact_context_refuses_an_unpinned_digest(r11_artifact):
    with pytest.raises(mod.ReplayValidationError, match="pinned"):
        mod.exact_solve_context(r11_artifact, expected_context_sha256="0" * 64)


def test_r11_exact_context_refuses_absent_capture(artifact):
    with pytest.raises(mod.ReplayValidationError, match="absent"):
        mod.exact_solve_context(artifact)


def test_r11_exact_context_refuses_a_refused_capture(r11_artifact):
    frames = tuple(
        r11_artifact["solver_failure_observation"]["raw"]["frames"])
    refused = mod.graph_mod.refused_solve_context(
        0.15, 0.90, 0.02, frames).to_record()
    tampered = copy.deepcopy(r11_artifact)
    tampered["solver_failure_observation"]["raw"]["solve_context"] = refused
    with pytest.raises(mod.ReplayValidationError, match="complete capture"):
        mod.exact_solve_context(tampered)


def test_r11_exact_context_refuses_a_resealed_substitution(r11_artifact):
    frames = tuple(
        r11_artifact["solver_failure_observation"]["raw"]["frames"])
    resealed = mod.graph_mod.failed_solve_context(
        0.15, 0.90, 0.02,
        (("image-0000", "image-0001", 1), ("image-0000", "image-0002", 1)),
        frames).to_record()
    # A self-consistent reseal passes the production replay but cannot borrow
    # the pinned digest that the producing call actually recorded.
    mod.graph_mod.validate_solver_solve_context(resealed)
    assert resealed["record_sha256"] != mod.R11_CONTEXT_SHA256
    tampered = copy.deepcopy(r11_artifact)
    tampered["solver_failure_observation"]["raw"]["solve_context"] = resealed
    with pytest.raises(mod.ReplayValidationError, match="pinned"):
        mod.exact_solve_context(tampered)


def test_r11_exact_context_refuses_a_resealed_reordered_universe(r11_artifact):
    reordered = mod.graph_mod.failed_solve_context(
        0.15, 0.90, 0.02, (),
        ("image-0001", "image-0000", "image-0002")).to_record()
    tampered = copy.deepcopy(r11_artifact)
    tampered["solver_failure_observation"]["raw"]["solve_context"] = reordered
    with pytest.raises(mod.ReplayValidationError, match="validation"):
        mod.exact_solve_context(
            tampered,
            expected_frames=("image-0000", "image-0001", "image-0002"))


# ---------------------------------------------------------------------------
# R10/R11 shared-input reconciliation
# ---------------------------------------------------------------------------

def test_reconcile_shared_inputs_accepts_the_pinned_pair(artifact, r11_artifact):
    reconciled = mod.reconcile_shared_inputs(artifact, r11_artifact)
    assert reconciled["graph_sha256"] == artifact["graph_sha256"]
    assert reconciled["trace_sha256"] == artifact["trace_sha256"]
    assert reconciled["source_digests"] == artifact["trace"]["source_digests"]
    assert reconciled["record_sha256"] == (
        artifact["same_axis_candidates"]["record_sha256"])
    assert reconciled["observation_sha256"]["r10"] != (
        reconciled["observation_sha256"]["r11"])
    assert "same_axis_candidates" in reconciled["reconciled_keys"]


@pytest.mark.parametrize("key", ["graph_sha256", "trace_sha256",
                                 "same_axis_candidates", "inputs"])
def test_reconcile_shared_inputs_refuses_each_divergence(
        artifact, r11_artifact, key):
    tampered = copy.deepcopy(r11_artifact)
    if key in ("graph_sha256", "trace_sha256"):
        tampered[key] = "f" * 64
    elif key == "same_axis_candidates":
        tampered[key]["record_sha256"] = "f" * 64
    else:
        tampered[key] = [{"frame": "foreign"}]
    with pytest.raises(mod.ReplayValidationError, match="differ"):
        mod.reconcile_shared_inputs(artifact, tampered)


# ---------------------------------------------------------------------------
# one unpruned R11 exact-context control, zero treatment arms
# ---------------------------------------------------------------------------

def test_r11_control_reproduces_r10_with_exactly_one_solver_call(
        monkeypatch, r11_artifact):
    calls = []
    real_run_arm = mod.run_arm

    def counting_run_arm(*args, **kwargs):
        calls.append((kwargs.get("target"), kwargs.get("context")))
        return real_run_arm(*args, **kwargs)

    def forbidden_build_treatments(*args, **kwargs):
        raise AssertionError(
            "no treatment arm may be built for the R11 control")

    monkeypatch.setattr(mod, "run_arm", counting_run_arm)
    monkeypatch.setattr(mod, "build_treatments", forbidden_build_treatments)

    payload = mod.replay_r11_control(ARTIFACT_PATH, R11_ARTIFACT_PATH)

    # Exactly one solver call: the R11 exact-context, unpruned control.
    assert len(calls) == 1
    target, context = calls[0]
    assert target is None
    assert context is not None
    assert context.record_sha256 == mod.R11_CONTEXT_SHA256

    assert payload["schema"] == mod.R11_CONTROL_SCHEMA
    assert payload["status"] == "complete"
    assert payload["control_gate"]["passed"] is True
    assert payload["control_gate"]["mismatches"] == []
    assert payload["control_gate"]["r11_recorded_mismatches"] == []
    assert payload["control_gate"]["tolerance_absolute"] == (
        mod.CONTROL_ABS_TOLERANCE)
    assert payload["solver_calls_executed"] == 1
    assert payload["treatments"] == {
        "authorized": False,
        "interpretable": False,
        "arms": {},
        "arms_executed": 0,
    }

    control = payload["control"]
    assert control["mode"] == "exact_context"
    assert control["pinned"] is True
    assert control["assignment"] == [0, 3, 1]
    assert control["reason"] == "plane_integer_residual"
    assert control["accepted"] is False
    assert control["observations"] == [8, 6, 6]
    assert control["duplicate_b"] is False

    expected = payload["r10_expected"]
    assert payload["r11_recorded"] == expected
    assert mod.control_mismatches(control, expected, mod.CONTROL_ABS_TOLERANCE) == []
    for axis in ("0", "1", "2"):
        assert math.isclose(
            control["resultants"][axis], expected["resultants"][axis],
            rel_tol=0.0, abs_tol=1e-12)
        assert math.isclose(
            control["residuals"][axis], expected["residuals"][axis],
            rel_tol=0.0, abs_tol=1e-12)
    for frame, value in expected["translations"].items():
        assert control["translations"][frame] == pytest.approx(value, abs=1e-12)

    assert payload["validation"]["r11_context_replay"] == "pass"
    assert payload["validation"]["r11_context_frame_binding"] == "pass"
    assert payload["validation"]["r10_r11_reconciliation"] == "pass"
    assert payload["reconciliation"]["graph_sha256"] == (
        mod.load_r10_reconstruction(ARTIFACT_PATH).artifact["graph_sha256"])
    assert payload["seal_sha256"] == mod.seal_payload(payload)


def test_r11_control_refuses_wrong_or_missing_r11_input():
    with pytest.raises(mod.ReplayValidationError, match="absent"):
        mod.replay_r11_control(
            ARTIFACT_PATH, ROOT / "no-such-r11-artifact" / "observation.json")
    with pytest.raises(mod.ReplayValidationError, match="SHA-256"):
        mod.replay_r11_control(
            ARTIFACT_PATH, R11_ARTIFACT_PATH, expected_r11_sha256="0" * 64)


def test_r11_control_refuses_wrong_r10_input():
    with pytest.raises(mod.ReplayValidationError, match="SHA-256"):
        mod.replay_r11_control(
            ARTIFACT_PATH, R11_ARTIFACT_PATH, expected_r10_sha256="0" * 64)


def test_run_arm_refuses_a_double_pin(r11_artifact):
    reconstruction = mod.load_r10_reconstruction(ARTIFACT_PATH)
    context = mod.exact_solve_context(
        r11_artifact, expected_frames=reconstruction.universe.frames)
    with pytest.raises(mod.ReplayValidationError, match="both"):
        mod.run_arm(
            reconstruction.universe.readings, reconstruction.matches,
            reconstruction.universe.frames, target=(0, 3, 1), context=context)


# ---------------------------------------------------------------------------
# R11 preregistered C1-C5 arms: three-way reconciliation, one call per arm
# ---------------------------------------------------------------------------

def test_r11_arms_reconciles_three_sealed_artifacts_and_runs_c1_c5_once(
        monkeypatch):
    calls = []
    real_run_arm = mod.run_arm

    def counting_run_arm(*args, **kwargs):
        calls.append((kwargs.get("target"), kwargs.get("context")))
        return real_run_arm(*args, **kwargs)

    def forbidden_build_treatments(*args, **kwargs):
        raise AssertionError("the legacy build_treatments path must not run")

    monkeypatch.setattr(mod, "run_arm", counting_run_arm)
    monkeypatch.setattr(mod, "build_treatments", forbidden_build_treatments)

    payload = mod.replay_r11_arms(
        ARTIFACT_PATH, R11_ARTIFACT_PATH, R11_BASELINE_PATH)

    # Exactly one context-pinned solver call per arm; the accepted baseline was
    # consumed, never rerun.
    assert len(calls) == 5
    assert [context.record_sha256 for _target, context in calls] == (
        [mod.R11_CONTEXT_SHA256] * 5)
    assert all(target is None for target, _context in calls)
    assert all(context is not None for _target, context in calls)

    assert payload["schema"] == mod.ARMS_SCHEMA
    assert payload["task_id"] == mod.ARMS_TASK_ID
    assert payload["status"] == "complete"
    assert payload["solver_calls_executed"] == 5
    assert payload["solver_calls_expected"] == 5
    assert payload["baseline_rerun"] is False
    assert payload["arm_order"] == ["C1", "C2", "C3", "C4", "C5"]
    assert list(payload["arms"]) == ["C1", "C2", "C3", "C4", "C5"]

    control = payload["sealed_control"]
    assert control["assignment"] == [0, 3, 1]
    assert control["reason"] == "plane_integer_residual"
    assert control["observations"] == [8, 6, 6]
    assert control["file_sha256"] == mod.R11_BASELINE_FILE_SHA256
    assert control["seal_sha256"] == mod.R11_BASELINE_SEAL_SHA256

    for _name, arm in payload["arms"].items():
        outcome = arm["outcome"]
        assert outcome["mode"] == "exact_context"
        assert outcome["pinned"] is True
        assert outcome["assignment"] == [0, 3, 1]
        assert outcome["exact_context"]["record_sha256"] == (
            mod.R11_CONTEXT_SHA256)
        assert arm["overall_geometry_label"] == outcome["reason"]

        # The substituted groups are world-axis 1, so axes 0 and 2 cannot move.
        deltas = arm["deltas_vs_sealed_control"]
        for axis in ("0", "2"):
            assert deltas["resultants"][axis] == 0.0
            assert deltas["residuals"][axis] == 0.0
        assert deltas["assignment_changed"] is False

        diagnostics = arm["diagnostics"]
        assert diagnostics["diagnostic_only"] is True
        assert diagnostics["min_resultant"] == 0.9
        assert diagnostics["integer_tolerance"] == 0.15
        assert diagnostics["active_match_count"] == (
            outcome["active_match_count"])
        assert diagnostics["axis_coverage_complete"] is True
        for axis, per_axis in diagnostics["per_axis"].items():
            assert per_axis["resultant"] == outcome["resultants"][axis]
            assert per_axis["residual"] == outcome["residuals"][axis]
            assert per_axis["axis_gate_pass"] == (
                per_axis["resultant"] >= 0.9
                and per_axis["residual"] <= 0.15)

        assert arm["preregistered_check"] is not None
        assert arm["provenance"]
        for provenance in arm["provenance"]:
            assert provenance["candidate_disposition"] == "dropped"
            assert provenance["candidate_reason"] == (
                mod.plane_graph.SAME_AXIS_DISPOSITION_DROPPED)
            assert provenance["candidate_world_axis"] == 1
            assert provenance["candidate_right_world_axis"] == 1
            assert provenance["candidate_same_axis_under_control"] is True
            assert provenance["candidate_active_under_control"] is True

    assert payload["validation"]["baseline_rerun"] is False
    assert payload["validation"][
        "sealed_baseline_reconciliation_replays"] == "pass"
    assert payload["validation"][
        "sealed_control_reproduced_recorded_outcome"] == "pass"
    assert payload["seal_sha256"] == mod.seal_payload(payload)

    preregistered = payload["preregistered_check"]
    assert preregistered["all_matched"] == (not preregistered["deviations"])
    for name in ("C1", "C2", "C3", "C4", "C5"):
        assert preregistered["arms"][name] is not None


def test_r11_arm_substitutions_are_frozen_dropped_same_axis_rows(record):
    universe = mod.build_reading_universe(record)
    claims = mod._group_claims(record)
    assert mod.SUBSTITUTIONS == {
        "C1": ((203, 19, 4),),
        "C2": ((383, 33, 7),),
        "C3": ((203, 19, 4), (383, 33, 7)),
        "C4": ((204, 19, 4), (384, 33, 7)),
        "C5": ((206, 19, 4), (384, 33, 7)),
    }
    for specification in mod.SUBSTITUTIONS.values():
        for flat_index, match_index, group_index in specification:
            _match, row = mod.candidate_match(record, universe, flat_index)
            assert int(row[0]) == group_index
            assert row[17] == "dropped"
            assert row[18] == mod.plane_graph.SAME_AXIS_DISPOSITION_DROPPED
            assert match_index in claims[group_index]


def test_r11_arms_refuses_wrong_or_missing_baseline_before_any_arm(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("no arm may run after a pin mismatch")

    monkeypatch.setattr(mod, "run_arm", forbidden)
    missing = ROOT / "no-such-r11-baseline" / "counterfactual_replay.json"
    with pytest.raises(mod.ReplayValidationError, match="absent"):
        mod.replay_r11_arms(ARTIFACT_PATH, R11_ARTIFACT_PATH, missing)
    with pytest.raises(mod.ReplayValidationError, match="SHA-256"):
        mod.replay_r11_arms(
            ARTIFACT_PATH, R11_ARTIFACT_PATH, R11_BASELINE_PATH,
            expected_baseline_sha256="0" * 64)
    with pytest.raises(mod.ReplayValidationError, match="seal"):
        mod.replay_r11_arms(
            ARTIFACT_PATH, R11_ARTIFACT_PATH, R11_BASELINE_PATH,
            expected_baseline_seal_sha256="0" * 64)
    with pytest.raises(mod.ReplayValidationError, match="SHA-256"):
        mod.replay_r11_arms(
            ARTIFACT_PATH, R11_ARTIFACT_PATH, R11_BASELINE_PATH,
            expected_r10_sha256="0" * 64)


def test_sealed_baseline_payload_refusals(r11_baseline):
    accepted = copy.deepcopy(r11_baseline)
    assert mod._validate_r11_baseline_payload(
        accepted,
        expected_seal_sha256=mod.R11_BASELINE_SEAL_SHA256) is accepted

    failed_gate = copy.deepcopy(r11_baseline)
    failed_gate["control_gate"]["passed"] = False
    failed_gate["seal_sha256"] = mod.seal_payload(failed_gate)
    with pytest.raises(mod.ReplayValidationError, match="gate"):
        mod._validate_r11_baseline_payload(
            failed_gate,
            expected_seal_sha256=mod.seal_payload(failed_gate))

    arms_carried = copy.deepcopy(r11_baseline)
    arms_carried["treatments"]["arms"] = {"C1": {}}
    arms_carried["seal_sha256"] = mod.seal_payload(arms_carried)
    with pytest.raises(mod.ReplayValidationError, match="treatment arms"):
        mod._validate_r11_baseline_payload(
            arms_carried,
            expected_seal_sha256=mod.seal_payload(arms_carried))

    foreign_context = copy.deepcopy(r11_baseline)
    foreign_context["inputs"]["r11"]["solve_context"]["record_sha256"] = (
        "0" * 64)
    foreign_context["seal_sha256"] = mod.seal_payload(foreign_context)
    with pytest.raises(mod.ReplayValidationError, match="solve-context"):
        mod._validate_r11_baseline_payload(
            foreign_context,
            expected_seal_sha256=mod.seal_payload(foreign_context))

    broken_seal = copy.deepcopy(r11_baseline)
    broken_seal["control_gate"]["passed"] = False
    with pytest.raises(mod.ReplayValidationError, match="seal"):
        mod._validate_r11_baseline_payload(broken_seal)


def test_reconcile_arms_inputs_refuses_a_tampered_sealed_control(
        r11_artifact, r11_baseline):
    reconstruction = mod.load_r10_reconstruction(ARTIFACT_PATH)
    context = mod.exact_solve_context(
        r11_artifact, expected_frames=reconstruction.universe.frames)

    bad_control = copy.deepcopy(r11_baseline)
    bad_control["control"]["resultants"]["1"] = 0.5
    with pytest.raises(mod.ReplayValidationError, match="recorded R10"):
        mod.reconcile_arms_inputs(
            reconstruction, r11_artifact, context, bad_control,
            r11_file_sha256=mod.R11_INPUT_FILE_SHA256)

    bad_reconciliation = copy.deepcopy(r11_baseline)
    bad_reconciliation["reconciliation"]["graph_sha256"] = "f" * 64
    with pytest.raises(mod.ReplayValidationError, match="reconciliation"):
        mod.reconcile_arms_inputs(
            reconstruction, r11_artifact, context, bad_reconciliation,
            r11_file_sha256=mod.R11_INPUT_FILE_SHA256)

    bad_context = copy.deepcopy(r11_baseline)
    bad_context["control"]["exact_context"]["integer_tolerance"] = 0.2
    with pytest.raises(mod.ReplayValidationError, match="context"):
        mod.reconcile_arms_inputs(
            reconstruction, r11_artifact, context, bad_context,
            r11_file_sha256=mod.R11_INPUT_FILE_SHA256)

    with pytest.raises(mod.ReplayValidationError, match="R11 file"):
        mod.reconcile_arms_inputs(
            reconstruction, r11_artifact, context, r11_baseline,
            r11_file_sha256="0" * 64)

    accepted = mod.reconcile_arms_inputs(
        reconstruction, r11_artifact, context, r11_baseline,
        r11_file_sha256=mod.R11_INPUT_FILE_SHA256)
    assert accepted["sealed_baseline"] == r11_baseline["reconciliation"]
    assert accepted["sealed_control_reproduced_recorded_outcome"] is True
    assert accepted["sealed_context_matches_capture"] is True


def test_build_context_arms_refuses_unfrozen_substitutions(
        monkeypatch, r11_artifact, r11_baseline):
    reconstruction = mod.load_r10_reconstruction(ARTIFACT_PATH)
    context = mod.exact_solve_context(
        r11_artifact, expected_frames=reconstruction.universe.frames)

    def forbidden(*args, **kwargs):
        raise AssertionError("a refused substitution must not reach the solver")

    monkeypatch.setattr(mod, "run_arm", forbidden)

    monkeypatch.setitem(mod.SUBSTITUTIONS, "C1", ((203, 5, 4),))
    with pytest.raises(mod.ReplayValidationError, match="survivor"):
        mod.build_context_arms(
            reconstruction.record, reconstruction.universe,
            reconstruction.matches, context, r11_baseline["control"])

    monkeypatch.setitem(mod.SUBSTITUTIONS, "C1", ((383, 19, 4),))
    with pytest.raises(mod.ReplayValidationError, match="declared group"):
        mod.build_context_arms(
            reconstruction.record, reconstruction.universe,
            reconstruction.matches, context, r11_baseline["control"])


# ---------------------------------------------------------------------------
# R11 axes-0/2 + C3 combined arms: frozen definitions, pre-arm validation,
# exactly eight context-pinned calls, sealed C3 reference
# ---------------------------------------------------------------------------

def _sealed_quadrants(r11_c1_c5):
    c3 = mod.sealed_c3_reference(r11_c1_c5)
    frames = ("image-0000", "image-0001", "image-0002")
    assert c3["outcome"]["assignment"] == [0, 3, 1]
    return c3, {frame: int(value)
                for frame, value in zip(frames, c3["outcome"]["assignment"])}


def test_axes02_c3_frozen_definitions_match_the_cross_product():
    assert mod.C3_SUBSTITUTIONS == ((203, 19, 4, 1), (383, 33, 7, 1))
    assert mod.AXES02_ARMS == {
        "A50-295": ((50, 4, 0, 0), (295, 23, 5, 2)),
        "A50-304": ((50, 4, 0, 0), (304, 26, 6, 2)),
        "A151-295": ((151, 16, 3, 0), (295, 23, 5, 2)),
        "A151-304": ((151, 16, 3, 0), (304, 26, 6, 2)),
        "A188-295": ((188, 17, 3, 0), (295, 23, 5, 2)),
        "A188-304": ((188, 17, 3, 0), (304, 26, 6, 2)),
        "A424-295": ((424, 37, 8, 0), (295, 23, 5, 2)),
        "A424-304": ((424, 37, 8, 0), (304, 26, 6, 2)),
    }
    assert list(mod.AXES02_ARMS) == [
        "A50-295", "A50-304", "A151-295", "A151-304",
        "A188-295", "A188-304", "A424-295", "A424-304"]


def test_axes02_c3_arm_constraints_replay_before_any_solver(r11_c1_c5):
    reconstruction = mod.load_r10_reconstruction(ARTIFACT_PATH)
    record = reconstruction.record
    universe = reconstruction.universe
    _c3, quadrants = _sealed_quadrants(r11_c1_c5)

    c3_replacements, c3_provenance = mod.plan_combined_arm(
        record, universe, reconstruction.matches, quadrants,
        mod.C3_SUBSTITUTIONS, label="C3")
    assert set(c3_replacements) == {19, 33}
    assert [(entry["candidate_flat_row"], entry["replaces_match_index"])
            for entry in c3_provenance] == [(203, 19), (383, 33)]

    for name, specification in mod.AXES02_ARMS.items():
        replacements, provenance = mod.plan_combined_arm(
            record, universe, reconstruction.matches, quadrants,
            specification, label=name)
        assert set(replacements) == {entry[1] for entry in specification}
        for entry, row_provenance in zip(specification, provenance):
            flat_row, match_index, group_index, world_axis = entry
            assert row_provenance["candidate_flat_row"] == flat_row
            assert row_provenance["candidate_group"] == group_index
            assert row_provenance["replaces_match_index"] == match_index
            assert row_provenance["candidate_disposition"] == "dropped"
            assert row_provenance["candidate_reason"] == (
                mod.plane_graph.SAME_AXIS_DISPOSITION_DROPPED)
            assert row_provenance["candidate_world_axis"] == world_axis
            assert row_provenance["candidate_right_world_axis"] == world_axis
            assert world_axis in (0, 2)
            assert row_provenance["candidate_same_axis_under_control"] is True
            assert row_provenance["candidate_active_under_control"] is True
            assert row_provenance["candidate_key_replays_survivor"] is True


def test_axes02_c3_group3_target_is_forced_by_edge_identity(r11_c1_c5):
    reconstruction = mod.load_r10_reconstruction(ARTIFACT_PATH)
    record = reconstruction.record
    universe = reconstruction.universe
    _c3, quadrants = _sealed_quadrants(r11_c1_c5)

    for flat_row, declared_target in ((151, 16), (188, 17)):
        replacements, provenance = mod.plan_combined_arm(
            record, universe, reconstruction.matches, quadrants,
            ((flat_row, declared_target, 3, 0),), label="group3")
        assert set(replacements) == {declared_target}
        assert provenance[0]["candidate_key_replays_survivor"] is True

    # The opposite target would leave the other survivor's edge duplicated.
    with pytest.raises(mod.ReplayValidationError, match="duplicate"):
        mod.plan_combined_arm(
            record, universe, reconstruction.matches, quadrants,
            ((151, 17, 3, 0),), label="group3")
    with pytest.raises(mod.ReplayValidationError, match="duplicate"):
        mod.plan_combined_arm(
            record, universe, reconstruction.matches, quadrants,
            ((188, 16, 3, 0),), label="group3")
    with pytest.raises(mod.ReplayValidationError, match="survivor"):
        mod.plan_combined_arm(
            record, universe, reconstruction.matches, quadrants,
            ((151, 5, 3, 0),), label="group3")
    with pytest.raises(mod.ReplayValidationError, match="declared group"):
        mod.plan_combined_arm(
            record, universe, reconstruction.matches, quadrants,
            ((295, 4, 0, 0),), label="group3")
    with pytest.raises(mod.ReplayValidationError, match="world axis"):
        mod.plan_combined_arm(
            record, universe, reconstruction.matches, quadrants,
            ((151, 16, 3, 2),), label="group3")


def test_axes02_c3_runs_exactly_eight_context_pinned_calls(monkeypatch):
    calls = []
    real_run_arm = mod.run_arm

    def counting_run_arm(*args, **kwargs):
        calls.append((kwargs.get("target"), kwargs.get("context")))
        return real_run_arm(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("no legacy/baseline arm path may run")

    monkeypatch.setattr(mod, "run_arm", counting_run_arm)
    monkeypatch.setattr(mod, "build_context_arms", forbidden)
    monkeypatch.setattr(mod, "build_treatments", forbidden)
    monkeypatch.setattr(mod, "replay_r11_arms", forbidden)
    monkeypatch.setattr(mod, "replay_r11_control", forbidden)

    payload = mod.replay_r11_axes02_c3(
        ARTIFACT_PATH, R11_ARTIFACT_PATH, R11_BASELINE_PATH, R11_C1_C5_PATH)

    assert len(calls) == 8
    assert all(target is None for target, _context in calls)
    assert all(context is not None for _target, context in calls)
    assert {context.record_sha256 for _target, context in calls} == {
        mod.R11_CONTEXT_SHA256}

    assert payload["schema"] == mod.AXES02_C3_SCHEMA
    assert payload["task_id"] == mod.AXES02_C3_TASK_ID
    assert payload["status"] == "complete"
    assert payload["solver_calls_executed"] == 8
    assert payload["solver_calls_expected"] == 8
    assert payload["baseline_rerun"] is False
    assert payload["c3_rerun"] is False
    assert payload["arm_order"] == list(mod.AXES02_ARMS)
    assert list(payload["arms"]) == list(mod.AXES02_ARMS)
    assert not ({"C1", "C2", "C3", "C4", "C5"} & set(payload["arms"]))

    sealed = payload["sealed_c3"]
    assert sealed["file_sha256"] == mod.R11_C1_C5_FILE_SHA256
    assert sealed["seal_sha256"] == mod.R11_C1_C5_SEAL_SHA256
    assert sealed["reason"] == "plane_integer_residual"
    assert sealed["assignment"] == [0, 3, 1]
    assert sealed["observations"] == [8, 6, 6]
    assert sealed["diagnostics"]["diagnostic_only"] is True

    for name, arm in payload["arms"].items():
        specification = mod.AXES02_ARMS[name]
        outcome = arm["outcome"]
        assert outcome["mode"] == "exact_context"
        assert outcome["pinned"] is True
        assert outcome["assignment"] == [0, 3, 1]
        assert outcome["exact_context"]["record_sha256"] == (
            mod.R11_CONTEXT_SHA256)
        assert arm["includes_c3"] is True
        assert arm["c3_rows"] == [203, 383]
        assert arm["frozen_axes02_pair"] == [
            specification[0][0], specification[1][0]]
        assert arm["overall_geometry_label"] == outcome["reason"]

        deltas = arm["deltas_vs_sealed_c3"]
        assert deltas["assignment_changed"] is False
        assert set(deltas["resultants"]) == {"0", "1", "2"}
        for axis in ("0", "1", "2"):
            assert deltas["resultants"][axis] == pytest.approx(
                outcome["resultants"][axis] - sealed["resultants"][axis],
                abs=1e-15)
            assert deltas["residuals"][axis] == pytest.approx(
                outcome["residuals"][axis] - sealed["residuals"][axis],
                abs=1e-15)
        # The substituted groups are axes 0/2 only, so axis 1 is unchanged.
        assert deltas["resultants"]["1"] == 0.0
        assert deltas["residuals"]["1"] == 0.0

        diagnostics = arm["diagnostics"]
        assert diagnostics["diagnostic_only"] is True
        assert diagnostics["min_resultant"] == 0.9
        assert diagnostics["integer_tolerance"] == 0.15
        assert diagnostics["active_match_count"] == (
            outcome["active_match_count"])
        assert diagnostics["axis_coverage_complete"] is True

    assert payload["validation"]["baseline_rerun"] is False
    assert payload["validation"]["c3_rerun"] is False
    assert payload["validation"][
        "frozen_substitutions_validated_before_any_arm"] == "pass"
    assert payload["validation"]["sealed_c3_provenance_replays"] == "pass"
    assert payload["seal_sha256"] == mod.seal_payload(payload)


def test_axes02_c3_validates_every_arm_before_any_solver_call(monkeypatch):
    calls = []
    real_run_arm = mod.run_arm

    def counting_run_arm(*args, **kwargs):
        calls.append(1)
        return real_run_arm(*args, **kwargs)

    monkeypatch.setattr(mod, "run_arm", counting_run_arm)
    monkeypatch.setitem(
        mod.AXES02_ARMS, "A424-304", ((424, 26, 8, 0), (304, 26, 6, 2)))
    with pytest.raises(mod.ReplayValidationError, match="survivor"):
        mod.replay_r11_axes02_c3(
            ARTIFACT_PATH, R11_ARTIFACT_PATH, R11_BASELINE_PATH, R11_C1_C5_PATH)
    assert calls == []


def test_axes02_c3_refuses_wrong_or_missing_pins_before_any_arm(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("no arm may run after a pin mismatch")

    monkeypatch.setattr(mod, "run_arm", forbidden)

    missing = ROOT / "no-such-r11-c1-c5" / "counterfactual_replay.json"
    with pytest.raises(mod.ReplayValidationError, match="absent"):
        mod.replay_r11_axes02_c3(
            ARTIFACT_PATH, R11_ARTIFACT_PATH, R11_BASELINE_PATH, missing)
    with pytest.raises(mod.ReplayValidationError, match="SHA-256"):
        mod.replay_r11_axes02_c3(
            ARTIFACT_PATH, R11_ARTIFACT_PATH, R11_BASELINE_PATH,
            R11_C1_C5_PATH, expected_c1_c5_sha256="0" * 64)
    with pytest.raises(mod.ReplayValidationError, match="seal"):
        mod.replay_r11_axes02_c3(
            ARTIFACT_PATH, R11_ARTIFACT_PATH, R11_BASELINE_PATH,
            R11_C1_C5_PATH, expected_c1_c5_seal_sha256="0" * 64)
    with pytest.raises(mod.ReplayValidationError, match="baseline"):
        mod.replay_r11_axes02_c3(
            ARTIFACT_PATH, R11_ARTIFACT_PATH, R11_BASELINE_PATH,
            R11_C1_C5_PATH, expected_baseline_seal_sha256="0" * 64)
    with pytest.raises(mod.ReplayValidationError, match="SHA-256"):
        mod.replay_r11_axes02_c3(
            ARTIFACT_PATH, R11_ARTIFACT_PATH, R11_BASELINE_PATH,
            R11_C1_C5_PATH, expected_r10_sha256="0" * 64)


def test_sealed_c1_c5_payload_refusals(r11_c1_c5):
    accepted = copy.deepcopy(r11_c1_c5)
    assert mod._validate_r11_c1_c5_payload(accepted) is accepted

    incomplete = copy.deepcopy(r11_c1_c5)
    incomplete["status"] = "partial"
    incomplete["seal_sha256"] = mod.seal_payload(incomplete)
    with pytest.raises(mod.ReplayValidationError, match="completed"):
        mod._validate_r11_c1_c5_payload(
            incomplete, expected_seal_sha256=mod.seal_payload(incomplete))

    fewer_arms = copy.deepcopy(r11_c1_c5)
    fewer_arms["arms"].pop("C5")
    fewer_arms["seal_sha256"] = mod.seal_payload(fewer_arms)
    with pytest.raises(mod.ReplayValidationError, match="frozen"):
        mod._validate_r11_c1_c5_payload(
            fewer_arms, expected_seal_sha256=mod.seal_payload(fewer_arms))

    foreign_context = copy.deepcopy(r11_c1_c5)
    foreign_context["inputs"]["r11"]["solve_context"]["record_sha256"] = "0" * 64
    foreign_context["seal_sha256"] = mod.seal_payload(foreign_context)
    with pytest.raises(mod.ReplayValidationError, match="context"):
        mod._validate_r11_c1_c5_payload(
            foreign_context, expected_seal_sha256=mod.seal_payload(
                foreign_context))

    foreign_baseline = copy.deepcopy(r11_c1_c5)
    foreign_baseline["sealed_control"]["seal_sha256"] = "0" * 64
    foreign_baseline["seal_sha256"] = mod.seal_payload(foreign_baseline)
    with pytest.raises(mod.ReplayValidationError, match="sealed control seal"):
        mod._validate_r11_c1_c5_payload(
            foreign_baseline, expected_seal_sha256=mod.seal_payload(
                foreign_baseline))

    broken_seal = copy.deepcopy(r11_c1_c5)
    broken_seal["status"] = "partial"
    with pytest.raises(mod.ReplayValidationError, match="seal"):
        mod._validate_r11_c1_c5_payload(broken_seal)


def test_axes02_c3_refuses_a_foreign_sealed_c3(monkeypatch, r11_c1_c5):
    tampered = copy.deepcopy(r11_c1_c5)
    tampered["arms"]["C3"]["provenance"][0]["candidate_flat_row"] = 204
    tampered["seal_sha256"] = mod.seal_payload(tampered)
    monkeypatch.setattr(mod, "load_r11_c1_c5", lambda *args, **kwargs: tampered)

    def forbidden(*args, **kwargs):
        raise AssertionError("a foreign sealed C3 must not reach the solver")

    monkeypatch.setattr(mod, "run_arm", forbidden)
    with pytest.raises(mod.ReplayValidationError, match="frozen C3"):
        mod.replay_r11_axes02_c3(
            ARTIFACT_PATH, R11_ARTIFACT_PATH, R11_BASELINE_PATH, R11_C1_C5_PATH)


# ---------------------------------------------------------------------------
# R11 exhaustive axes-0/2 + C3 arms: exactly 201 prevalidated substitutions,
# 201 context-pinned one-for-one calls plus one measured-best combined arm
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def axes02_exhaustive_payload():
    return mod.replay_r11_axes02_exhaustive(
        ARTIFACT_PATH, R11_ARTIFACT_PATH, R11_BASELINE_PATH, R11_C1_C5_PATH,
        R11_AXES02_PATH)


def _fake_axis_substitution(flat_row, target_match, score, world_axis=0):
    return mod.AxisSubstitution(
        world_axis=world_axis, group=0, flat_row=flat_row,
        target_match=target_match, score=score, delta=0.0, match=None)


def _fake_axis_arm(arm_id, world_axis, resultant, residual, score, flat_row,
                   target_match, gate):
    return {
        "arm_id": arm_id,
        "world_axis": world_axis,
        "candidate_flat_row": flat_row,
        "candidate_group": 0,
        "candidate_target_match": target_match,
        "candidate_score": score,
        "candidate_delta": 0.0,
        "candidate_key_replays_survivor": False,
        "diagnostics": {
            "per_axis": {
                str(world_axis): {
                    "axis_gate_pass": gate, "resultant": resultant,
                    "residual": residual},
                "1": {
                    "axis_gate_pass": False, "resultant": 0.5,
                    "residual": 0.5},
            },
        },
    }


def test_axes02_exhaustive_prevalidation_reproduces_exactly_201(r11_c1_c5):
    reconstruction = mod.load_r10_reconstruction(ARTIFACT_PATH)
    _c3, quadrants = _sealed_quadrants(r11_c1_c5)
    frames = tuple(reconstruction.universe.frames)
    c3_replacements, _c3_provenance = mod.plan_combined_arm(
        reconstruction.record, reconstruction.universe, reconstruction.matches,
        quadrants, mod.C3_SUBSTITUTIONS, label="C3")

    plans = mod.plan_exhaustive_arms(
        reconstruction.record, reconstruction.universe, reconstruction.matches,
        frames, quadrants, c3_replacements)
    substitutions = plans["substitutions"]

    assert len(substitutions) == mod.EXHAUSTIVE_EXPECTED_TOTAL == 201
    assert sum(1 for sub in substitutions if sub.world_axis == 0) == 164
    assert sum(1 for sub in substitutions if sub.world_axis == 2) == 37
    assert plans["report"]["observed_axis_counts"] == {"0": 164, "2": 37}
    assert plans["report"]["observed_per_group_axis"] == {
        "0:0": 81, "3:0": 53, "5:2": 26, "6:2": 11, "8:0": 30}
    assert plans["report"]["observed_per_group_target"] == {
        "0:4": 81, "3:16": 25, "3:17": 28, "5:23": 26, "6:26": 11, "8:37": 30}
    assert plans["report"]["validated_before_any_solver_call"] is True
    assert plans["report"]["combined_arms_duplicate_free_all"] is True
    pair = plans["report"]["pair_prevalidation"]
    assert pair["pairs_checked"] == 164 * 37 == 6068
    assert pair["pairs_valid"] == 6068
    assert pair["all_pairs_valid"] is True
    # Candidate edge identity may legitimately differ from the survivor's, so
    # the exhaustive set is strictly larger than the earlier rank-1 choices.
    assert any(
        (sub.flat_row, sub.target_match) not in {
            (50, 4), (151, 16), (188, 17), (424, 37),
            (295, 23), (304, 26)}
        for sub in substitutions)


def test_axes02_exhaustive_reports_a_non_201_count_before_any_solver(
        monkeypatch, r11_c1_c5):
    reconstruction = mod.load_r10_reconstruction(ARTIFACT_PATH)
    _c3, quadrants = _sealed_quadrants(r11_c1_c5)
    frames = tuple(reconstruction.universe.frames)
    monkeypatch.setitem(mod.EXHAUSTIVE_EXPECTED_PER_GROUP_TARGET, (3, 16), 24)
    with pytest.raises(mod.ReplayValidationError, match="201"):
        mod.enumerate_valid_axis_substitutions(
            reconstruction.record, reconstruction.universe,
            reconstruction.matches, frames, quadrants)

    def forbidden(*args, **kwargs):
        raise AssertionError("no solver call after a prevalidation mismatch")

    monkeypatch.setattr(mod, "run_arm", forbidden)
    with pytest.raises(mod.ReplayValidationError, match="201"):
        mod.replay_r11_axes02_exhaustive(
            ARTIFACT_PATH, R11_ARTIFACT_PATH, R11_BASELINE_PATH,
            R11_C1_C5_PATH, R11_AXES02_PATH)


def test_axes02_exhaustive_runs_202_context_pinned_calls(monkeypatch):
    calls = []
    real_run_arm = mod.run_arm

    def counting_run_arm(*args, **kwargs):
        calls.append((kwargs.get("target"), kwargs.get("context")))
        return real_run_arm(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("no legacy/baseline arm path may run")

    monkeypatch.setattr(mod, "run_arm", counting_run_arm)
    monkeypatch.setattr(mod, "build_context_arms", forbidden)
    monkeypatch.setattr(mod, "build_treatments", forbidden)
    monkeypatch.setattr(mod, "replay_r11_arms", forbidden)
    monkeypatch.setattr(mod, "replay_r11_control", forbidden)
    monkeypatch.setattr(mod, "replay_r11_axes02_c3", forbidden)

    payload = mod.replay_r11_axes02_exhaustive(
        ARTIFACT_PATH, R11_ARTIFACT_PATH, R11_BASELINE_PATH, R11_C1_C5_PATH,
        R11_AXES02_PATH)

    assert len(calls) == 202
    assert all(target is None for target, _context in calls)
    assert all(context is not None for _target, context in calls)
    assert {context.record_sha256 for _target, context in calls} == {
        mod.R11_CONTEXT_SHA256}

    assert payload["schema"] == mod.AXES02_EXHAUSTIVE_SCHEMA
    assert payload["task_id"] == mod.AXES02_EXHAUSTIVE_TASK_ID
    assert payload["status"] == "complete"
    assert payload["solver_calls_authorized"] == 202
    assert payload["solver_calls_expected"] == 202
    assert payload["solver_calls_executed"] == 202
    assert payload["baseline_rerun"] is False
    assert payload["c3_rerun"] is False
    assert payload["axes02_rerun"] is False
    assert payload["c1_c5_rerun"] is False
    assert len(payload["arms"]) == 201
    assert payload["prevalidation"]["observed_total"] == 201
    assert payload["validation"][
        "exhaustive_prevalidation_reproduces_201"] == "pass"
    assert payload["validation"][
        "all_201_substitutions_validated_before_any_arm"] == "pass"
    assert payload["validation"][
        "all_cross_axis_pairs_validated_solver_free"] == "pass"
    assert payload["seal_sha256"] == mod.seal_payload(payload)


def test_axes02_exhaustive_selection_uses_measured_metrics_not_score():
    def pick(pairs, world_axis=0):
        substitutions = [sub for sub, _arm in pairs]
        arms = {sub.arm_id: arm for sub, arm in pairs}
        return mod.select_best_axis_substitution(substitutions, arms, world_axis)

    # Gate-pass status outranks a higher resultant and a higher score.
    lower = _fake_axis_substitution(11, 4, score=0.1)
    higher = _fake_axis_substitution(10, 4, score=0.9)
    assert pick((
        (lower, _fake_axis_arm(lower.arm_id, 0, 0.91, 0.10, 0.1, 11, 4, True)),
        (higher, _fake_axis_arm(higher.arm_id, 0, 0.99, 0.01, 0.9, 10, 4, False)),
    )) is lower

    # Resultant descending decides among gate-passing arms.
    a = _fake_axis_substitution(20, 4, 0.5)
    b = _fake_axis_substitution(21, 4, 0.5)
    assert pick((
        (a, _fake_axis_arm(a.arm_id, 0, 0.95, 0.05, 0.5, 20, 4, True)),
        (b, _fake_axis_arm(b.arm_id, 0, 0.97, 0.06, 0.5, 21, 4, True)),
    )) is b

    # Residual ascending.
    c = _fake_axis_substitution(30, 4, 0.5)
    d = _fake_axis_substitution(31, 4, 0.5)
    assert pick((
        (c, _fake_axis_arm(c.arm_id, 0, 0.97, 0.02, 0.5, 30, 4, True)),
        (d, _fake_axis_arm(d.arm_id, 0, 0.97, 0.01, 0.5, 31, 4, True)),
    )) is d

    # Captured score descending.
    e = _fake_axis_substitution(40, 4, 0.5)
    f = _fake_axis_substitution(41, 4, 0.9)
    assert pick((
        (e, _fake_axis_arm(e.arm_id, 0, 0.97, 0.01, 0.5, 40, 4, True)),
        (f, _fake_axis_arm(f.arm_id, 0, 0.97, 0.01, 0.9, 41, 4, True)),
    )) is f

    # Flat row ascending.
    g = _fake_axis_substitution(50, 4, 0.5)
    h = _fake_axis_substitution(51, 4, 0.5)
    assert pick((
        (g, _fake_axis_arm(g.arm_id, 0, 0.97, 0.01, 0.5, 50, 4, True)),
        (h, _fake_axis_arm(h.arm_id, 0, 0.97, 0.01, 0.5, 51, 4, True)),
    )) is g

    # Survivor index ascending is the final tie-break.
    i = _fake_axis_substitution(60, 4, 0.5)
    j = _fake_axis_substitution(60, 5, 0.5)
    assert pick((
        (i, _fake_axis_arm(i.arm_id, 0, 0.97, 0.01, 0.5, 60, 4, True)),
        (j, _fake_axis_arm(j.arm_id, 0, 0.97, 0.01, 0.5, 60, 5, True)),
    )) is i

    # Axis 2 uses its own moved-axis diagnostic, not axis 0's.
    k = _fake_axis_substitution(70, 23, 0.5, world_axis=2)
    l = _fake_axis_substitution(71, 26, 0.5, world_axis=2)
    assert pick((
        (k, _fake_axis_arm(k.arm_id, 2, 0.80, 0.01, 0.9, 70, 23, False)),
        (l, _fake_axis_arm(l.arm_id, 2, 0.70, 0.01, 0.1, 71, 26, True)),
    ), world_axis=2) is l


def test_axes02_exhaustive_final_arm_is_the_measured_best_pair(
        axes02_exhaustive_payload):
    payload = axes02_exhaustive_payload
    selection = payload["selection"]
    best0 = selection["best_axis0_arm"]
    best2 = selection["best_axis2_arm"]
    assert selection["best_axis0"]["arm_id"] == best0
    assert selection["best_axis2"]["arm_id"] == best2

    axis0_arms = [arm for arm in payload["arms"].values()
                  if arm["world_axis"] == 0]
    axis2_arms = [arm for arm in payload["arms"].values()
                  if arm["world_axis"] == 2]
    assert min(axis0_arms, key=mod._exhaustive_sort_key)["arm_id"] == best0
    assert min(axis2_arms, key=mod._exhaustive_sort_key)["arm_id"] == best2

    final = selection["final_arm"]
    assert final is not None
    assert final["axis0_arm_id"] == best0
    assert final["axis2_arm_id"] == best2
    assert final["pair_key_prevalidated"] is True
    assert final["outcome"]["mode"] == "exact_context"
    assert final["outcome"]["pinned"] is True
    assert final["outcome"]["exact_context"]["record_sha256"] == (
        mod.R11_CONTEXT_SHA256)
    assert final["outcome"]["assignment"] == [0, 3, 1]
    assert final["outcome"]["observations"] == [8, 6, 6]
    assert final["outcome"]["duplicate_b"] is False
    assert final["diagnostics"]["axis_coverage_complete"] is True
    assert final["overall_geometry_label"] == final["outcome"]["reason"]
    assert payload["summary"]["final_combined_pair_key_prevalidated"] is True
    assert payload["summary"]["stopped_at_201"] is False
    assert payload["summary"]["final_combined"]["arm_id"] == final["arm_id"]


def test_axes02_exhaustive_stops_at_201_without_a_valid_pair(monkeypatch):
    real_plan = mod.plan_exhaustive_arms

    def deny_pairs(*args, **kwargs):
        plans = real_plan(*args, **kwargs)
        plans["valid_pair_keys"] = frozenset()
        return plans

    calls = []
    real_run_arm = mod.run_arm

    def counting_run_arm(*args, **kwargs):
        calls.append(1)
        return real_run_arm(*args, **kwargs)

    monkeypatch.setattr(mod, "plan_exhaustive_arms", deny_pairs)
    monkeypatch.setattr(mod, "run_arm", counting_run_arm)

    payload = mod.replay_r11_axes02_exhaustive(
        ARTIFACT_PATH, R11_ARTIFACT_PATH, R11_BASELINE_PATH, R11_C1_C5_PATH,
        R11_AXES02_PATH)

    assert len(calls) == 201
    assert payload["status"] == "stopped_at_201_no_valid_pair"
    assert payload["solver_calls_executed"] == 201
    assert payload["solver_calls_authorized"] == 202
    assert payload["selection"]["final_arm"] is None
    assert payload["selection"]["pair_key_prevalidated"] is False
    assert payload["summary"]["stopped_at_201"] is True
    assert payload["summary"]["final_combined"] is None


def test_axes02_exhaustive_deltas_are_against_sealed_c3(
        axes02_exhaustive_payload):
    payload = axes02_exhaustive_payload
    c3 = payload["sealed_c3"]
    assert c3["reason"] == "plane_integer_residual"
    assert c3["assignment"] == [0, 3, 1]
    assert c3["file_sha256"] == mod.R11_C1_C5_FILE_SHA256

    for arm in payload["arms"].values():
        assert arm["outcome"]["mode"] == "exact_context"
        assert arm["outcome"]["pinned"] is True
        assert arm["outcome"]["assignment"] == [0, 3, 1]
        assert arm["outcome"]["observations"] == [8, 6, 6]
        assert arm["outcome"]["duplicate_b"] is False
        assert arm["includes_c3"] is True
        assert arm["c3_rows"] == [203, 383]
        assert arm["overall_geometry_label"] == arm["outcome"]["reason"]
        diagnostic = arm["diagnostics"]["per_axis"][str(arm["world_axis"])]
        assert diagnostic["resultant_at_or_above_min"] == (
            diagnostic["resultant"] >= 0.9)
        assert diagnostic["residual_at_or_below_tolerance"] == (
            diagnostic["residual"] <= 0.15)
        assert diagnostic["axis_gate_pass"] == (
            diagnostic["resultant"] >= 0.9
            and diagnostic["residual"] <= 0.15)
        deltas = arm["deltas_vs_sealed_c3"]
        # Only axes 0/2 move; axis 1 is exactly the sealed C3 substitution.
        assert deltas["resultants"]["1"] == 0.0
        assert deltas["residuals"]["1"] == 0.0
        assert deltas["assignment_changed"] is False
        for axis in ("0", "1", "2"):
            assert deltas["resultants"][axis] == pytest.approx(
                arm["outcome"]["resultants"][axis] - c3["resultants"][axis],
                abs=1e-15)
            assert deltas["residuals"][axis] == pytest.approx(
                arm["outcome"]["residuals"][axis] - c3["residuals"][axis],
                abs=1e-15)

    final = payload["selection"]["final_arm"]
    assert final["deltas_vs_sealed_c3"]["resultants"]["1"] == 0.0
    assert final["deltas_vs_sealed_c3"]["residuals"]["1"] == 0.0
    assert payload["summary"]["threshold_passes"]["definition"]


def test_axes02_exhaustive_refuses_wrong_or_missing_axes02_pins(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("no arm may run after a pin mismatch")

    monkeypatch.setattr(mod, "run_arm", forbidden)

    missing = ROOT / "no-such-axes02-replay" / "counterfactual_replay.json"
    with pytest.raises(mod.ReplayValidationError, match="absent"):
        mod.replay_r11_axes02_exhaustive(
            ARTIFACT_PATH, R11_ARTIFACT_PATH, R11_BASELINE_PATH,
            R11_C1_C5_PATH, missing)
    with pytest.raises(mod.ReplayValidationError, match="SHA-256"):
        mod.replay_r11_axes02_exhaustive(
            ARTIFACT_PATH, R11_ARTIFACT_PATH, R11_BASELINE_PATH,
            R11_C1_C5_PATH, R11_AXES02_PATH, expected_axes02_sha256="0" * 64)
    with pytest.raises(mod.ReplayValidationError, match="seal"):
        mod.replay_r11_axes02_exhaustive(
            ARTIFACT_PATH, R11_ARTIFACT_PATH, R11_BASELINE_PATH,
            R11_C1_C5_PATH, R11_AXES02_PATH,
            expected_axes02_seal_sha256="0" * 64)


def test_sealed_axes02_payload_refusals(r11_axes02):
    accepted = copy.deepcopy(r11_axes02)
    assert mod._validate_r11_axes02_payload(accepted) is accepted

    wrong_task = copy.deepcopy(r11_axes02)
    wrong_task["task_id"] = "DS-OTHER"
    wrong_task["seal_sha256"] = mod.seal_payload(wrong_task)
    with pytest.raises(mod.ReplayValidationError, match="task id"):
        mod._validate_r11_axes02_payload(
            wrong_task, expected_seal_sha256=mod.seal_payload(wrong_task))

    fewer_arms = copy.deepcopy(r11_axes02)
    fewer_arms["arms"].pop("A50-295")
    fewer_arms["seal_sha256"] = mod.seal_payload(fewer_arms)
    with pytest.raises(mod.ReplayValidationError, match="frozen eight-arm"):
        mod._validate_r11_axes02_payload(
            fewer_arms, expected_seal_sha256=mod.seal_payload(fewer_arms))

    foreign_context = copy.deepcopy(r11_axes02)
    foreign_context["inputs"]["r11"]["solve_context"]["record_sha256"] = "0" * 64
    foreign_context["seal_sha256"] = mod.seal_payload(foreign_context)
    with pytest.raises(mod.ReplayValidationError, match="context"):
        mod._validate_r11_axes02_payload(
            foreign_context,
            expected_seal_sha256=mod.seal_payload(foreign_context))

    foreign_c3 = copy.deepcopy(r11_axes02)
    foreign_c3["sealed_c3"]["seal_sha256"] = "0" * 64
    foreign_c3["seal_sha256"] = mod.seal_payload(foreign_c3)
    with pytest.raises(mod.ReplayValidationError, match="sealed C3 seal"):
        mod._validate_r11_axes02_payload(
            foreign_c3, expected_seal_sha256=mod.seal_payload(foreign_c3))

    foreign_provenance = copy.deepcopy(r11_axes02)
    foreign_provenance["sealed_c3"]["provenance"][0]["candidate_flat_row"] = 204
    foreign_provenance["seal_sha256"] = mod.seal_payload(foreign_provenance)
    with pytest.raises(mod.ReplayValidationError, match="frozen substitution"):
        mod._validate_r11_axes02_payload(
            foreign_provenance,
            expected_seal_sha256=mod.seal_payload(foreign_provenance))

    broken_seal = copy.deepcopy(r11_axes02)
    broken_seal["status"] = "partial"
    with pytest.raises(mod.ReplayValidationError, match="seal"):
        mod._validate_r11_axes02_payload(broken_seal)


# ---------------------------------------------------------------------------
# R11 axes-0 x axis-2 pair scan: 6,068-pair solver-free prevalidation, exactly
# 6,068 context-pinned calls, exact 6-D Pareto frontiers, no scalar winner
# ---------------------------------------------------------------------------

def _pair_scan_inputs(r11_c1_c5):
    reconstruction = mod.load_r10_reconstruction(ARTIFACT_PATH)
    _c3, quadrants = _sealed_quadrants(r11_c1_c5)
    frames = tuple(reconstruction.universe.frames)
    c3_replacements, _provenance = mod.plan_combined_arm(
        reconstruction.record, reconstruction.universe, reconstruction.matches,
        quadrants, mod.C3_SUBSTITUTIONS, label="C3")
    plans = mod.plan_exhaustive_arms(
        reconstruction.record, reconstruction.universe, reconstruction.matches,
        frames, quadrants, c3_replacements)
    pair_plan = mod.plan_pair_scan(
        plans, reconstruction.matches, frames, quadrants, c3_replacements)
    return {
        "reconstruction": reconstruction,
        "frames": frames,
        "quadrants": quadrants,
        "c3_replacements": c3_replacements,
        "plans": plans,
        "pair_plan": pair_plan,
    }


def test_pair_scan_constants_are_the_six_dimensional_contract():
    assert mod.PAIR_SCAN_SCHEMA == (
        "mc-photogrammetry-same-axis-counterfactual-replay-r11-axes02-pair-scan-v1")
    assert mod.PAIR_SCAN_TASK_ID == "DS-IMG-R11-AXES02-PAIR-SCAN-20261004"
    assert mod.PAIR_SCAN_EXPECTED_AXIS_COUNTS == {0: 164, 2: 37}
    assert mod.PAIR_SCAN_EXPECTED_PAIRS == 6068
    assert mod.PAIR_SCAN_AUTHORIZED_SOLVER_CALLS == 6068
    assert mod.PAIR_SCAN_ACCEPTED_REFERENCE_PAIR == (20, 4, 308, 26)
    assert mod.PAIR_SCAN_ACCEPTED_REFERENCE_LABEL == "A020-308"
    assert mod.PAIR_SCAN_OBJECTIVES == (
        "score_delta_axis0",
        "score_delta_axis2",
        "resultant0",
        "resultant2",
        "negated_residual0",
        "negated_residual2",
    )
    # The superseded 2-D pooled constant must not survive into the 6-D contract.
    assert not hasattr(mod, "PAIR_SCAN_POOLED_OBJECTIVES")
    assert "no pooled score" in mod.PAIR_SCAN_PARETO_RULE.lower()
    assert "six-dimensional" in mod.PAIR_SCAN_PARETO_RULE
    # DS-IMG-R11-PAIR-SCAN-ACCEPTANCE-ERRATUM: primary acceptance is the
    # unchanged solver's own reason == "" outcome; the moved-axis gates are a
    # separate diagnostic, and frontier_accepted_solver is the named
    # accepted-arm frontier while frontier_accepted_gate stays a diagnostic.
    accepted_definition = mod.PAIR_SCAN_ACCEPTED_DEFINITION
    assert "primary accepted pair = the unchanged solver" in accepted_definition
    assert "reason == ''" in accepted_definition
    assert "accepted true" in accepted_definition
    assert "never labelled" in accepted_definition
    assert "separate" in accepted_definition
    assert "frontier_accepted_solver" in mod.PAIR_SCAN_PARETO_RULE
    assert "frontier_accepted_gate" in mod.PAIR_SCAN_PARETO_RULE
    assert "diagnostic" in mod.PAIR_SCAN_PARETO_RULE.lower()
    assert mod.PAIR_SCAN_PAIR_KEY_SHA256.lower() == (
        "8cdc32850ad53bc3388ba68e2e4170fe204bf7256ff6570a4b02a36b7f63fd55")


def test_pair_scan_prevalidation_reproduces_6068_and_pair_key_digest(
        r11_c1_c5, r11_axes02_exhaustive):
    inputs = _pair_scan_inputs(r11_c1_c5)
    pair_plan = inputs["pair_plan"]
    report = pair_plan["report"]

    assert len(pair_plan["pairs"]) == mod.PAIR_SCAN_EXPECTED_PAIRS == 6068
    assert len(set(pair_plan["keys"])) == 6068
    assert report["observed_axis_counts"] == {"0": 164, "2": 37}
    assert report["observed_pairs"] == 6068
    assert report["duplicate_free_all"] is True
    assert report["pair_keys_unique"] is True
    assert report["validated_before_any_solver_call"] is True
    assert report["pair_key_sha256_matches_pin"] is True
    assert report["pair_key_sha256_matches_exhaustive"] is True
    assert pair_plan["digest"] == mod.PAIR_SCAN_PAIR_KEY_SHA256.lower()
    assert pair_plan["digest"] == (
        r11_axes02_exhaustive["prevalidation"]["pair_prevalidation"][
            "valid_pairs_sha256"].lower())

    # A tampered pin must stop the solver-free plan before anything solves.
    monkeypatch_note = "0" * 64
    original = mod.PAIR_SCAN_PAIR_KEY_SHA256
    mod.PAIR_SCAN_PAIR_KEY_SHA256 = monkeypatch_note
    try:
        with pytest.raises(mod.ReplayValidationError, match="digest"):
            mod.plan_pair_scan(
                inputs["plans"], inputs["reconstruction"].matches,
                inputs["frames"], inputs["quadrants"],
                inputs["c3_replacements"])
    finally:
        mod.PAIR_SCAN_PAIR_KEY_SHA256 = original


def test_pair_scan_runs_exactly_6068_pinned_calls_and_reports_frontiers(
        monkeypatch, r11_c1_c5, r11_axes02_exhaustive):
    sealed_outcome = r11_axes02_exhaustive["selection"]["final_arm"]["outcome"]
    calls = []

    def stub_run_arm(readings, matches, frames, *, target=None, context=None):
        calls.append({"target": target, "context": context})
        return copy.deepcopy(sealed_outcome)

    monkeypatch.setattr(mod, "run_arm", stub_run_arm)

    payload = mod.replay_r11_axes02_pair_scan(
        ARTIFACT_PATH, R11_ARTIFACT_PATH, R11_BASELINE_PATH, R11_C1_C5_PATH,
        R11_AXES02_PATH, R11_AXES02_EXHAUSTIVE_PATH)

    assert len(calls) == 6068
    assert all(call["target"] is None for call in calls)
    assert all(call["context"] is not None for call in calls)
    assert {call["context"].record_sha256 for call in calls} == {
        mod.R11_CONTEXT_SHA256}

    assert payload["schema"] == mod.PAIR_SCAN_SCHEMA
    assert payload["task_id"] == mod.PAIR_SCAN_TASK_ID
    assert payload["status"] == "complete"
    assert payload["solver_calls_authorized"] == 6068
    assert payload["solver_calls_expected"] == 6068
    assert payload["solver_calls_executed"] == 6068
    assert payload["retries"] == 0
    assert payload["baseline_rerun"] is False
    assert payload["c3_rerun"] is False
    assert payload["axes02_rerun"] is False
    assert payload["exhaustive_rerun"] is False
    assert payload["c1_c5_rerun"] is False
    assert len(payload["pairs"]) == 6068
    assert len(payload["pair_order"]) == 6068
    assert payload["prevalidation"]["pair_scan"][
        "pair_key_sha256_matches_pin"] is True
    assert payload["validation"][
        "pair_scan_prevalidation_reproduces_6068"] == "pass"
    assert payload["validation"][
        "all_6068_pairs_duplicate_free_before_any_solver"] == "pass"
    assert payload["validation"][
        "all_6068_calls_context_pinned_once"] == "pass"
    assert payload["validation"][
        "accepted_reference_pair_cross_checked_against_sealed_final_arm"
    ] == "pass"
    assert payload["seal_sha256"] == mod.seal_payload(payload)

    # With the sealed final-arm outcome returned for every pair the whole scan
    # is accepted, so the all-arm frontier equals the accepted-arm frontier.
    assert payload["summary"]["accepted_gate_pair_count"] == 6068
    assert payload["summary"]["accepted_solver_pair_count"] == 6068
    assert payload["summary"]["duplicate_pair_count"] == 0
    assert payload["summary"]["axis1_constant_across_all_pairs"] is True
    frontier_all = payload["pareto"]["frontier_all"]
    frontier_gate = payload["pareto"]["frontier_accepted_gate"]
    assert frontier_all["size"] >= 1
    assert frontier_all["pair_ids"] == frontier_gate["pair_ids"]
    assert frontier_all["size"] == len(frontier_all["members"])
    assert set(frontier_all["pair_ids"]) <= set(payload["pair_order"])
    for member in frontier_all["members"]:
        assert list(member["objectives"]) == list(mod.PAIR_SCAN_OBJECTIVES)
        assert len(member["objectives"]) == 6
    assert payload["summary"]["highest_score_pair_id"] in payload["pair_order"]
    assert payload["summary"]["accepted_reference_pair_id"] == "A020-308"
    assert payload["highest_score_pair"]["row"]["pair_id"] == (
        payload["summary"]["highest_score_pair_id"])


def test_pair_scan_refuses_wrong_or_missing_exhaustive_pins_before_any_call(
        monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("no solver call after a pin mismatch")

    monkeypatch.setattr(mod, "run_arm", forbidden)

    missing = ROOT / "no-such-exhaustive-replay" / "counterfactual_replay.json"
    with pytest.raises(mod.ReplayValidationError, match="absent"):
        mod.replay_r11_axes02_pair_scan(
            ARTIFACT_PATH, R11_ARTIFACT_PATH, R11_BASELINE_PATH,
            R11_C1_C5_PATH, R11_AXES02_PATH, missing)
    with pytest.raises(mod.ReplayValidationError, match="SHA-256"):
        mod.replay_r11_axes02_pair_scan(
            ARTIFACT_PATH, R11_ARTIFACT_PATH, R11_BASELINE_PATH,
            R11_C1_C5_PATH, R11_AXES02_PATH, R11_AXES02_EXHAUSTIVE_PATH,
            expected_exhaustive_sha256="0" * 64)
    with pytest.raises(mod.ReplayValidationError, match="seal"):
        mod.replay_r11_axes02_pair_scan(
            ARTIFACT_PATH, R11_ARTIFACT_PATH, R11_BASELINE_PATH,
            R11_C1_C5_PATH, R11_AXES02_PATH, R11_AXES02_EXHAUSTIVE_PATH,
            expected_exhaustive_seal_sha256="0" * 64)


def test_pair_scan_pareto_frontier_is_exact_6d_and_sign_aware():
    base = (0.0, 0.0, 0.95, 0.95, -0.05, -0.05)
    better = (0.1, 0.1, 0.96, 0.96, -0.04, -0.04)
    assert mod._exact_dominates(better, base) is True
    assert mod._exact_dominates(base, better) is False
    assert mod._exact_dominates(base, base) is False
    assert mod._exact_dominates(
        (0.2, 0.0, 0.95, 0.95, -0.05, -0.05),
        (0.0, 0.2, 0.95, 0.95, -0.05, -0.05)) is False
    # A larger raw residual is worse: flipping its sign makes it dominated.
    assert mod._exact_dominates(
        (0.0, 0.0, 0.95, 0.95, -0.01, -0.01),
        (0.0, 0.0, 0.95, 0.95, -0.20, -0.20)) is True

    vectors = {
        "dominated": (0.0, 0.0, 0.90, 0.90, -0.10, -0.10),
        "dominator": (0.5, 0.5, 0.99, 0.99, -0.01, -0.01),
        "tie_a": (0.6, 0.6, 0.95, 0.95, -0.05, -0.05),
        "tie_b": (0.6, 0.6, 0.95, 0.95, -0.05, -0.05),
        "trade_a": (0.7, 0.1, 0.95, 0.95, -0.05, -0.05),
        "trade_b": (0.1, 0.7, 0.95, 0.95, -0.05, -0.05),
    }
    frontier = mod.pareto_frontier(vectors)
    assert "dominated" not in frontier
    assert "dominator" in frontier
    assert "tie_a" in frontier and "tie_b" in frontier
    assert "trade_a" in frontier and "trade_b" in frontier
    assert frontier == sorted(frontier)
    assert frontier == [
        "dominator", "tie_a", "tie_b", "trade_a", "trade_b"]

    with pytest.raises(mod.ReplayValidationError, match="non-finite"):
        mod.pareto_frontier(
            {"bad": (float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0)})


def test_pair_scan_pareto_frontier_matches_a_pure_python_oracle():
    import random

    def oracle(vectors):
        ids = list(vectors)
        frontier = []
        for pair_id in ids:
            vector = vectors[pair_id]
            dominated = False
            for other in ids:
                if other == pair_id:
                    continue
                reference = vectors[other]
                if (all(a >= b for a, b in zip(reference, vector))
                        and any(a > b for a, b in zip(reference, vector))):
                    dominated = True
                    break
            if not dominated:
                frontier.append(pair_id)
        return sorted(frontier)

    random.seed(20261004)
    for _trial in range(40):
        vectors = {
            f"p{index}": tuple(
                random.choice([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
                for _ in range(6))
            for index in range(25)}
        assert mod.pareto_frontier(vectors) == oracle(vectors)


def test_pair_scan_reference_pair_matches_sealed_exhaustive_final_arm(
        r11_c1_c5, r11_axes02_exhaustive, r11_artifact):
    inputs = _pair_scan_inputs(r11_c1_c5)
    reconstruction = inputs["reconstruction"]
    frames = inputs["frames"]
    quadrants = inputs["quadrants"]
    c3_replacements = inputs["c3_replacements"]
    target = mod.PAIR_SCAN_ACCEPTED_REFERENCE_PAIR
    matched = [
        pair for pair in inputs["pair_plan"]["pairs"]
        if (pair[0].flat_row, pair[0].target_match,
            pair[1].flat_row, pair[1].target_match) == target]
    assert len(matched) == 1
    first, second = matched[0]

    context = mod.exact_solve_context(
        r11_artifact, expected_frames=reconstruction.universe.frames,
        expected_context_sha256=mod.R11_CONTEXT_SHA256)
    treatment = mod.substitute(reconstruction.matches, {
        **c3_replacements,
        first.target_match: first.match,
        second.target_match: second.match})
    mod._assert_no_active_duplicates(treatment, frames, quadrants)
    outcome = mod.run_arm(
        reconstruction.universe.readings, treatment, frames, target=None,
        context=context)
    row = mod._pair_scan_row(
        0, first, second, reconstruction.matches, outcome, context)
    sealed = r11_axes02_exhaustive["selection"]["final_arm"]

    assert row["pair_id"] == mod.PAIR_SCAN_ACCEPTED_REFERENCE_LABEL
    assert row["reason"] == sealed["outcome"]["reason"] == ""
    assert row["accepted"] is True
    assert row["assignment"] == sealed["outcome"]["assignment"] == [0, 3, 1]
    assert row["resultants"] == sealed["outcome"]["resultants"]
    assert row["residuals"] == sealed["outcome"]["residuals"]
    assert row["observations"] == sealed["outcome"]["observations"]
    assert row["active_match_count"] == sealed["outcome"]["active_match_count"]
    assert row["duplicate_b"] == sealed["outcome"]["duplicate_b"] is False
    assert row["gate"]["moved_axes_pass"] is True
    assert row["gate"]["all_axes_pass"] is True


def test_sealed_axes02_exhaustive_payload_refusals(r11_axes02_exhaustive):
    accepted = copy.deepcopy(r11_axes02_exhaustive)
    assert mod._validate_r11_axes02_exhaustive_payload(accepted) is accepted

    wrong_task = copy.deepcopy(r11_axes02_exhaustive)
    wrong_task["task_id"] = "DS-OTHER"
    wrong_task["seal_sha256"] = mod.seal_payload(wrong_task)
    with pytest.raises(mod.ReplayValidationError, match="task id"):
        mod._validate_r11_axes02_exhaustive_payload(
            wrong_task, expected_seal_sha256=mod.seal_payload(wrong_task))

    wrong_schema = copy.deepcopy(r11_axes02_exhaustive)
    wrong_schema["schema"] = "mc-photogrammetry-other-v1"
    wrong_schema["seal_sha256"] = mod.seal_payload(wrong_schema)
    with pytest.raises(mod.ReplayValidationError, match="schema"):
        mod._validate_r11_axes02_exhaustive_payload(
            wrong_schema, expected_seal_sha256=mod.seal_payload(wrong_schema))

    fewer_pairs = copy.deepcopy(r11_axes02_exhaustive)
    fewer_pairs["prevalidation"]["pair_prevalidation"]["pairs_valid"] = 6067
    fewer_pairs["seal_sha256"] = mod.seal_payload(fewer_pairs)
    with pytest.raises(mod.ReplayValidationError, match="6,068"):
        mod._validate_r11_axes02_exhaustive_payload(
            fewer_pairs, expected_seal_sha256=mod.seal_payload(fewer_pairs))

    foreign_digest = copy.deepcopy(r11_axes02_exhaustive)
    foreign_digest["prevalidation"]["pair_prevalidation"][
        "valid_pairs_sha256"] = "0" * 64
    foreign_digest["seal_sha256"] = mod.seal_payload(foreign_digest)
    with pytest.raises(mod.ReplayValidationError, match="pair-key digest"):
        mod._validate_r11_axes02_exhaustive_payload(
            foreign_digest, expected_seal_sha256=mod.seal_payload(
                foreign_digest))

    foreign_pair = copy.deepcopy(r11_axes02_exhaustive)
    foreign_pair["selection"]["pair_key"] = [20, 4, 307, 26]
    foreign_pair["seal_sha256"] = mod.seal_payload(foreign_pair)
    with pytest.raises(mod.ReplayValidationError, match="A020-308"):
        mod._validate_r11_axes02_exhaustive_payload(
            foreign_pair, expected_seal_sha256=mod.seal_payload(foreign_pair))

    broken_seal = copy.deepcopy(r11_axes02_exhaustive)
    broken_seal["status"] = "partial"
    with pytest.raises(mod.ReplayValidationError, match="seal"):
        mod._validate_r11_axes02_exhaustive_payload(broken_seal)
