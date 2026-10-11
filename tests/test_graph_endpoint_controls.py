"""Focused tests for the DS-R5G endpoint-control utility (no image workload)."""
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluation/replay_graph_endpoint_controls.py"
SPEC = importlib.util.spec_from_file_location("ds_r5g_controls", SCRIPT)
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)

from mc_photogrammetry.reconstruction import plane_graph  # noqa: E402


def _match(*, left="image-0000", right="image-0002", left_plane=0, right_plane=0,
           left_axis=0, right_axis=0, shift=(0, 0), quadrant=0, score=0.5,
           left_b=1.0, right_b=1.0):
    return plane_graph.PlaneMatch(
        left=left, right=right, left_axis=left_axis, right_axis=right_axis,
        left_b=left_b, right_b=right_b, quadrant_of_right=quadrant, psr=2.0,
        peak=1.0, second_peak=0.5, overlap_cells=12, integer_residual=None,
        score=score, left_plane=left_plane, right_plane=right_plane,
        cell_shift_rc=shift)


def _matches():
    return (
        _match(left_plane=0, right_plane=1, left_axis=0, right_axis=0),
        _match(left_plane=1, right_plane=2, left_axis=1, right_axis=1),
        _match(left_plane=3, right_plane=4, left_axis=2, right_axis=2,
               left="image-0001", right="image-0003"),
        _match(left_plane=0, right_plane=5, left_axis=0, right_axis=0,
               left="image-0001", right="image-0003"),
        _match(left_plane=2, right_plane=0, left_axis=1, right_axis=1,
               left="image-0000", right="image-0003"),
        _match(left_plane=4, right_plane=0, left_axis=2, right_axis=2,
               left="image-0002", right="image-0003"),
    )


def _quadrants():
    return {"image-0000": 0, "image-0001": 1, "image-0002": 2,
            "image-0003": 3}


def _identity(match):
    return [match.left, match.left_plane, match.right, match.right_plane]


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


def _solver(calls, reason="plane_integer_residual"):
    def solve(readings, matches, **kwargs):
        calls.append((readings, tuple(matches), dict(kwargs)))
        return FakeSolution(reason=reason)
    return solve


# --------------------------------------------------------------------------
# endpoint identity and seeding
# --------------------------------------------------------------------------

def test_endpoint_identity_is_strict():
    assert subject.canonical_endpoint_identity("image-0001", 3) == ("image-0001", 3)
    for bad in (("", 0), (None, 0), ("image-0001", -1), ("image-0001", True),
                ("image-0001", 1.0)):
        with pytest.raises(ValueError):
            subject.canonical_endpoint_identity(*bad)


def test_replicate_seed_is_the_documented_sha256_text():
    expected = hashlib.sha256(
        b"image-0001 # 3 # 7").hexdigest()
    assert subject.replicate_seed("image-0001", 3, 7) == expected
    assert subject.canonical_endpoint_text("image-0001", 3) == "image-0001 # 3"


def test_replicate_seed_changes_with_identity_and_index():
    seeds = {
        subject.replicate_seed("image-0000", 0, 0),
        subject.replicate_seed("image-0000", 1, 0),
        subject.replicate_seed("image-0001", 0, 0),
        subject.replicate_seed("image-0000", 0, 1),
    }
    assert len(seeds) == 4
    with pytest.raises(ValueError):
        subject.replicate_seed("image-0000", 0, -1)


# --------------------------------------------------------------------------
# incident selection and baseline axis classification
# --------------------------------------------------------------------------

def test_endpoint_incident_indices_match_the_alias_conflict_projection():
    matches = _matches()
    assert subject.endpoint_incident_indices(matches, "image-0000", 0) == (0,)
    assert subject.endpoint_incident_indices(matches, "image-0000", 3) == ()
    assert subject.endpoint_incident_indices(matches, "image-0003", 0) == (4, 5)
    assert subject.endpoint_incident_indices(matches, "image-0003", 1) == ()
    # The alias-wide projection is exactly the union of the endpoint selections.
    endpoints = [["image-0000", 0], ["image-0000", 1], ["image-0001", 3],
                 ["image-0001", 0], ["image-0002", 2], ["image-0002", 4],
                 ["image-0003", 0], ["image-0003", 5]]
    union = set()
    for frame, plane in endpoints:
        union |= set(subject.endpoint_incident_indices(matches, frame, plane))
    assert union == set(subject.alias_conflict_match_indices(matches, endpoints))


def test_baseline_match_axis_uses_the_baseline_quadrant():
    match = _match(left="image-0001", left_axis=2)
    # baseline quadrants {image-0001: 1} -> axis_transform((2, 1)) = (0, -1)
    assert subject.baseline_match_axis(
        match, {"image-0000": 0, "image-0001": 1}) == 0
    assert subject.baseline_match_axis(
        match, {"image-0000": 0, "image-0001": 2}) == 2
    with pytest.raises(subject.ReplayUnavailable):
        subject.baseline_match_axis(match, {"image-0000": 0})


def test_baseline_axis_counts_are_per_axis_totals():
    matches = _matches()
    quadrants = _quadrants()
    # axis_transform under the baseline quadrants is identity on the even
    # quadrants and the (2,1,0) swap on the odd ones:
    #   image-0000 q0 -> (0, 1, 2); image-0001 q1 -> (2, 1, 0)
    #   image-0002 q2 -> (0, 1, 2); image-0003 q3 -> (2, 1, 0)
    assert subject.baseline_axis_counts(matches, quadrants, ()) == (0, 0, 0)
    assert subject.baseline_axis_counts(matches, quadrants, range(len(matches))) == (
        2, 2, 2)
    assert subject.baseline_axis_counts(matches, quadrants, (3,)) == (0, 0, 1)
    assert subject.baseline_axis_counts(matches, quadrants, (0, 1, 2)) == (2, 1, 0)


# --------------------------------------------------------------------------
# eligible control population
# --------------------------------------------------------------------------

def test_control_candidates_exclude_only_the_target_endpoint():
    matches = _matches()
    quadrants = _quadrants()
    eligible = subject.control_candidate_indices(matches, quadrants, "image-0000", 0)
    flat = sorted(index for axis in subject.AXES for index in eligible[axis])
    assert 0 not in flat
    assert flat == [1, 2, 3, 4, 5]
    # Every index lands in exactly one baseline world-axis bin.
    assert sum(len(eligible[axis]) for axis in subject.AXES) == len(flat)


def test_control_candidates_are_grouped_by_baseline_axis():
    matches = _matches()
    eligible = subject.control_candidate_indices(
        matches, _quadrants(), "image-0000", 0)
    assert eligible[0] == (2,)
    assert eligible[1] == (1, 4)
    assert eligible[2] == (3, 5)

# --------------------------------------------------------------------------
# deterministic without-replacement stratified draws
# --------------------------------------------------------------------------

def test_control_selection_is_deterministic_and_exactly_stratified():
    matches = _matches()
    quadrants = _quadrants()
    required = {0: 1, 1: 1, 2: 0}
    first = subject.control_removal_selection(
        matches, quadrants, "image-0000", 0, required, 0, _identity)
    second = subject.control_removal_selection(
        matches, quadrants, "image-0000", 0, required, 0, _identity)
    assert first == second
    assert len(first) == len(set(first)) == 2
    counts = subject.baseline_axis_counts(matches, quadrants, first)
    assert counts == (1, 1, 0)
    # Controls never touch the target endpoint.
    assert not (set(first) & set(subject.endpoint_incident_indices(
        matches, "image-0000", 0)))


def test_control_selection_changes_with_the_replicate_index():
    matches = _matches()
    quadrants = _quadrants()
    required = {0: 1, 1: 1, 2: 1}
    selections = {
        subject.control_removal_selection(
            matches, quadrants, "image-0000", 0, required, index, _identity)
        for index in range(12)
    }
    assert len(selections) > 1
    assert all(len(value) == 3 for value in selections)


def test_control_selection_without_replacement_covers_the_whole_bin():
    matches = _matches()
    quadrants = _quadrants()
    eligible = subject.control_candidate_indices(
        matches, quadrants, "image-0000", 0)
    selection = subject.control_removal_selection(
        matches, quadrants, "image-0000", 0, {0: 1, 1: 2, 2: 2}, 3, _identity)
    assert sorted(selection) == sorted(
        index for axis in subject.AXES for index in eligible[axis])


def test_control_selection_refuses_an_unfillable_bin():
    matches = _matches()
    quadrants = _quadrants()
    with pytest.raises(subject.ControlUnavailable) as error:
        subject.control_removal_selection(
            matches, quadrants, "image-0000", 0, {0: 9, 1: 0, 2: 0}, 0, _identity)
    assert error.value.axis == 0
    assert error.value.required == 9
    assert error.value.eligible == 1
    with pytest.raises(subject.ControlUnavailable) as error:
        subject.control_removal_selection(
            matches, quadrants, "image-0000", 0, {0: 0, 1: 3, 2: 0}, 0, _identity)
    assert error.value.axis == 1
    assert error.value.eligible == 2


def test_unfillable_bin_is_recorded_typed_and_never_relaxed():
    matches = _matches()
    calls = []
    experiments = subject.run_endpoint_experiments(
        readings=("r0",), matches=matches, solver_kwargs={},
        conflicting_endpoints=[["image-0000", 0]],
        baseline_quadrants=_quadrants(), solver=_solver(calls),
        identity=_identity, controls_per_endpoint=3,
        baseline_solution=None)
    endpoint = experiments[0]
    assert len(endpoint["controls"]) == 3
    # The endpoint incident count is (1, 1, 0); the bins hold (2, 1, 1), so every
    # replicate is solvable here and the treatment must be solved too.
    assert all(row["status"] == "solved" for row in endpoint["controls"])
    assert endpoint["treatment"]["status"] == "solved"


def test_insufficient_bin_status_is_typed_control_unavailable():
    matches = _matches()[:1]  # the only match, incident to image-0000 plane 0
    calls = []
    experiments = subject.run_endpoint_experiments(
        readings=("r0",), matches=matches, solver_kwargs={},
        conflicting_endpoints=[["image-0000", 0]],
        baseline_quadrants=_quadrants(), solver=_solver(calls),
        identity=_identity, controls_per_endpoint=2, baseline_solution=None)
    endpoint = experiments[0]
    assert endpoint["treatment_required_axis_counts"] == [1, 0, 0]
    assert endpoint["eligible_control_counts"] == [0, 0, 0]
    for row in endpoint["controls"]:
        assert row["status"] == "control_unavailable"
        assert row["reason"] == "control_unavailable"
        assert row["removed_match_count"] == 0
        assert row["axis_coverage"] == []
        assert row["axes"] == []
        assert row["required_count"] == 1
        assert row["eligible_count"] == 0
        assert len(row["seed_sha256"]) == 64
    # Only the treatment was handed to the solver: no relaxed substitution.
    assert len(calls) == 1


# --------------------------------------------------------------------------
# treatment/control solving reuses the captured solver inputs
# --------------------------------------------------------------------------

def test_experiments_reuse_readings_constraints_and_solver():
    matches = _matches()
    calls = []
    kwargs = {"integer_tolerance": 0.15, "min_resultant": 0.9,
              "ambiguous_margin": 0.02,
              "quadrant_constraints": (("image-0000", "image-0002", 2),)}
    experiments = subject.run_endpoint_experiments(
        readings=("r0", "r1"), matches=matches, solver_kwargs=kwargs,
        conflicting_endpoints=[["image-0000", 0], ["image-0003", 0]],
        baseline_quadrants=_quadrants(), solver=_solver(calls),
        identity=_identity, controls_per_endpoint=4, baseline_solution=None)
    assert len(experiments) == 2
    assert len(calls) == 2 * (1 + 4)
    for readings, matches_arg, seen_kwargs in calls:
        assert readings == ("r0", "r1")
        assert seen_kwargs == kwargs
        assert all(isinstance(item, plane_graph.PlaneMatch) for item in matches_arg)
    for endpoint in experiments:
        required = tuple(endpoint["treatment_required_axis_counts"])
        treatment = endpoint["treatment"]
        assert treatment["removed_match_count"] == sum(required)
        assert subject.baseline_axis_counts(
            matches, _quadrants(), treatment["removed_match_indices"]) == required
        assert treatment["retained_match_count"] == (
            len(matches) - treatment["removed_match_count"])
        for row in endpoint["controls"]:
            assert sorted(row["removed_match_indices"]) == row["removed_match_indices"]
            assert subject.baseline_axis_counts(
                matches, _quadrants(), row["removed_match_indices"]) == (
                    tuple(endpoint["treatment_required_axis_counts"]))
            assert len(row["removed_match_identities"]) == row["removed_match_count"]


def test_treatment_removes_exactly_the_incident_matches():
    matches = _matches()
    calls = []
    experiments = subject.run_endpoint_experiments(
        readings=("r0",), matches=matches, solver_kwargs={},
        conflicting_endpoints=[["image-0000", 0]],
        baseline_quadrants=_quadrants(), solver=_solver(calls),
        identity=_identity, controls_per_endpoint=2, baseline_solution=None)
    endpoint = experiments[0]
    assert endpoint["incident_match_indices"] == [0]
    assert endpoint["treatment"]["removed_match_indices"] == [0]
    assert endpoint["treatment"]["removed_match_identities"] == [
        ["image-0000", 0, "image-0002", 1]]
    removed = set(endpoint["treatment"]["removed_match_indices"])
    for row in endpoint["controls"]:
        assert not (set(row["removed_match_indices"]) & removed)


def test_solution_record_reports_reason_coverage_and_axes():
    record = subject.solution_record(
        "treatment", _matches(), [0], FakeSolution(reason="insufficient_axis_coverage",
                                                   resultants={2: 1.0},
                                                   residuals={2: 0.0},
                                                   observations=(0, 0, 4)),
        _identity)
    assert record["reason"] == "insufficient_axis_coverage"
    assert record["axis_coverage"] == [2]
    assert record["axes"] == [[0, None, None, 0], [1, None, None, 0],
                              [2, 1.0, 0.0, 4]]
    assert record["retained_match_count"] == len(_matches()) - 1


def test_axis_coverage_and_absent_diagnostics():
    record = subject.solution_record(
        "control-0000", _matches(), (), FakeSolution(reason="no_matches"),
        _identity)
    assert record["axis_coverage"] == [0, 1, 2]
    assert record["axes"] == [[0, 0.1, 0.01, 4], [1, 0.2, 0.02, 3],
                              [2, 0.3, 0.03, 2]]


# --------------------------------------------------------------------------
# baseline requirement cross-check
# --------------------------------------------------------------------------

_BASELINE_QUADRANTS = {"image-0000": 0, "image-0001": 1}
# (2, 1, 0): under image-0001's quadrant-1 transform, world axis w is right local
# axis (2, 1, 0)[w].
_RIGHT_AXIS_FOR_WORLD = (2, 1, 0)


def _active_match(world_axis):
    """One baseline-active match whose baseline world axis is ``world_axis``."""
    return _match(left="image-0000", right="image-0001", left_axis=world_axis,
                  right_axis=_RIGHT_AXIS_FOR_WORLD[world_axis], quadrant=1)


def _conserved_matches():
    """14 baseline-active matches: 6 on axis 0, 3 on axis 1 and 5 on axis 2."""
    return tuple(_active_match(world_axis)
                 for world_axis, count in enumerate((6, 3, 5))
                 for _ in range(count))


def _baseline_world(match):
    world, _sign = plane_graph.axis_transform(
        match.left_axis, _BASELINE_QUADRANTS[match.left])
    return int(world)


def _alias_edges(matches):
    """Well-formed 13-field alias rows binding each match to its baseline axis."""
    return [[index, match.left, match.left_plane, match.left_axis,
             None, None, match.right, match.right_plane, match.right_axis,
             None, None, match.quadrant_of_right, _baseline_world(match)]
            for index, match in enumerate(matches)]


def test_baseline_axis_requirement_accepts_conserved_counts():
    matches = _conserved_matches()
    quadrants, active, declared = subject.baseline_axis_requirement(
        _BASELINE_QUADRANTS, (12, 6, 10), matches, _alias_edges(matches))
    assert quadrants == {"image-0000": 0, "image-0001": 1}
    assert active == (6, 3, 5)
    assert declared == (6, 3, 5)


def test_baseline_axis_requirement_fails_closed():
    matches = _conserved_matches()
    edges = _alias_edges(matches)

    # An odd face count cannot halve into a baseline-active match count.
    with pytest.raises(subject.ReplayMismatch) as error:
        subject.baseline_axis_requirement(
            _BASELINE_QUADRANTS, (12, 6, 11), matches, edges)
    assert error.value.field == "baseline_axis_observations"

    # Dropping one axis-2 alias row leaves every remaining edge well formed but
    # makes the declared alias counts disagree with the baseline-active matches.
    with pytest.raises(subject.ReplayMismatch) as error:
        subject.baseline_axis_requirement(
            _BASELINE_QUADRANTS, (12, 6, 10), matches, edges[:-1])
    assert error.value.field == "baseline_axis_edge_counts"


# --------------------------------------------------------------------------
# fail-closed replay validation (DS-R5F check set)
# --------------------------------------------------------------------------

def _artifact_and_evidence():
    trace = {
        "graph_result_sha256": "a" * 64,
        "reason": "plane_integer_residual",
        "candidate_counts": {"pre_admission": 7476, "post_admission": 1495,
                             "post_same_axis": 54},
        "selected_quadrants": [["image-0000", 0], ["image-0001", 1]],
        "solver_failure": {
            "reason": "plane_integer_residual",
            "axes": [[0, 0.39, 0.17, 12], [1, 0.24, 0.20, 6], [2, 0.24, 0.18, 10]],
            "active_edge_aliases": {
                "input_match_count": 54,
                "input_match_identity_sha256": "b" * 64,
                "axis_edge_counts": [6, 3, 5]},
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

    for field, value in (("graph_sha256", "9" * 64),
                         ("selected_match_count", 53),
                         ("alias_identity_sha256", "9" * 64),
                         ("boundary_kept_counts", [0, 0, 0]),
                         ("project_sha256", "9" * 64),
                         ("project_session_sha256", "9" * 64)):
        tampered = dict(evidence)
        tampered[field] = value
        with pytest.raises(subject.ReplayMismatch):
            subject.validate_replay(artifact, tampered)


# --------------------------------------------------------------------------
# output discipline
# --------------------------------------------------------------------------

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


def test_verify_seal_rejects_a_tampered_payload():
    payload = subject.seal({"schema": subject.SCHEMA, "value": 1})
    payload["value"] = 2
    with pytest.raises(subject.ReplayMismatch):
        subject.verify_seal(payload)


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


def test_registered_output_paths_are_the_named_ones():
    assert subject.OUTPUT_PATH == ROOT / ".tmp-ds-r5g-20261003.json"
    assert subject.ARTIFACT_PATH.name == (
        ".tmp-axis0-graph-stage-attribution-v6-20261003.json")
    assert subject.PROJECT_PATH.name == (
        ".tmp-axis0-graph-stage-attribution-v6-20261003.mcscan")
    assert subject.DEADLINE_SECONDS == 600
    assert subject.LIMIT_BYTES == 8 * 1024 ** 3
    assert subject.CONTROLS_PER_ENDPOINT == 20
    assert subject.PRODUCER_PATHS[0].name == "replay_graph_endpoint_controls.py"
    assert subject.DS_R5F_UTILITY.name == "replay_graph_solver_ablation.py"


def test_producer_records_cover_the_named_sources():
    records = subject.producer_records()
    paths = {row["path"] for row in records}
    assert str(subject.Path(subject.__file__).resolve()) in paths
    assert str(subject.DS_R5F_UTILITY.resolve()) in paths
    assert str((ROOT / "src/mc_photogrammetry/reconstruction/plane_graph.py")
               .resolve()) in paths
    assert all(len(row["sha256"]) == 64 for row in records)
