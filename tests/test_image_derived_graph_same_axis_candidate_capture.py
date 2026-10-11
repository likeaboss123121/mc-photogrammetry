"""DS-IMG-SAMEAXIS-PREFILTER-CAPTURE-R9 focused controls.

These controls exercise the evaluation-only ``SameAxisCandidateCapture`` record
against a synthetic component whose real ``plane_graph.match_plane_graph`` admits
competing same-axis candidates and whose equality filter drops the lower-scoring
one. They prove the true pre/post linkage (a dropped pre-filter ``b`` alternative
is captured beside the survivor the solver received), the strict complete-record
replay and digest, the shared assignment/input-identity binding with the R7
observer, output/default-digest parity for the matcher, the trace, the solver and
the public graph result, and atomic over-cap refusal with no partial rows.

No production gate, matcher, filter, solver, threshold, truth, camera, mask or
reference input is read or changed here; no image or reconstruction workload
runs.
"""
from __future__ import annotations

import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from test_image_derived_lattice_measure import _result
from test_plane_graph import make_reading, make_signature
from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
from mc_photogrammetry.pipeline import image_derived_signatures as signatures_mod
from mc_photogrammetry.reconstruction import plane_graph


# Candidate specs for the controlled matcher: (left frame, right frame, left b,
# right b, relative quadrant) -> admitted score. Every entry is a same-axis,
# world-axis-1 group hypothesis; the two ``a``-``b`` entries deliberately
# disagree on the measured displacement so the equality filter must drop one.
_SPECS = {
    ("a", "b", 1.0, 1.0, 0): 0.9,
    ("a", "b", 0.5, 1.0, 0): 0.5,
    ("a", "c", 1.0, 1.0, 0): 0.9,
    ("b", "c", 1.0, 1.0, 0): 0.9,
}


def _cells(seed, shape=(12, 12)):
    return np.random.default_rng(seed).random(shape)


def _candidate_fixture():
    """Three axis-1 frames with one admitted same-axis disagreement.

    Reading ``a0`` keeps two measured alternatives (``b`` 1.0 at distance 3.0
    and ``b`` 0.5 at distance 1.5); the controlled matcher emits both against
    ``b0``, so the equality filter sees two candidates in one group and drops the
    lower-scoring one while the survivor reaches the solver.
    """
    a0 = make_reading("a", 1, 1.0, [(1.0, 3.0, 1.0), (0.5, 1.5, 0.8)])
    b0 = make_reading("b", 1, 1.0, [(1.0, 4.0, 1.0), (0.5, 2.0, 0.7)])
    c0 = make_reading("c", 1, 1.0, [(1.0, 5.0, 1.0)])
    readings = (a0, b0, c0)
    signatures = (
        make_signature(a0, _cells(1), b=1.0),
        make_signature(a0, _cells(2), b=0.5),
        make_signature(b0, _cells(3), b=1.0),
        make_signature(c0, _cells(4), b=1.0),
    )
    return readings, signatures


def _controlled_pair(left_signature, right_signature, *, quadrant, max_shift_cells,
                     refusals=None, diagnostic_sink=None, **_kwargs):
    """A deterministic stand-in for ``plane_graph.match_plane_pair``.

    It appends exactly one diagnostic trace row per call (admitted or not), just
    like the real matcher, so the canonical ``PlaneGraphTrace`` contract holds,
    and it returns a well-formed ``PlaneMatch`` only for the declared specs.
    """
    left = left_signature.reading
    right = right_signature.reading
    key = (left.frame, right.frame, float(left_signature.blocks_per_period),
           float(right_signature.blocks_per_period), int(quadrant))
    score = _SPECS.get(key)
    admitted = score is not None
    if diagnostic_sink is not None:
        diagnostic_sink.pairs.append(plane_graph.PlanePairTrace(
            left.frame, right.frame, left.axis, right.axis, int(quadrant),
            admitted, "" if admitted else "no_signal",
            None, None, None, None, None, None,
            psr=5.0 if admitted else None,
            score=score,
            overlap_cells=64 if admitted else None,
            cell_shift_rc=(0, 0) if admitted else None))
    if not admitted:
        return None
    return plane_graph.PlaneMatch(
        left=left.frame, right=right.frame, left_axis=left.axis,
        right_axis=right.axis,
        left_b=float(left_signature.blocks_per_period),
        right_b=float(right_signature.blocks_per_period),
        quadrant_of_right=int(quadrant), psr=5.0, peak=1.0, second_peak=0.1,
        overlap_cells=64, integer_residual=None, score=score,
        cell_shift_rc=(0, 0))


def _solution(readings, matches):
    return plane_graph.solve_component(
        readings, matches,
        quadrant_constraints=(("a", "b", 0), ("a", "c", 0)))


def _captured(monkeypatch):
    """Run the controlled matcher once with a capture sink attached."""
    readings, signatures = _candidate_fixture()
    monkeypatch.setattr(plane_graph, "match_plane_pair", _controlled_pair)
    trace = plane_graph.PlaneGraphTrace([])
    sink: list = []
    matches = plane_graph.match_plane_graph(
        readings, signatures, diagnostic_sink=trace,
        candidate_observation_sink=sink)
    return readings, signatures, matches, trace, sink[0]


@pytest.fixture(scope="module")
def records():
    patch = pytest.MonkeyPatch()
    try:
        session, _pose, lattice = _result(patch)
        signatures = signatures_mod.derive_image_derived_signatures(
            session=session, lattice_result=lattice)
        assert signatures.status == "assembled"
        yield session, lattice, signatures
    finally:
        patch.undo()


def _resign(record):
    body = {key: value for key, value in record.items()
            if key != "record_sha256"}
    domain = (graph_mod.SAME_AXIS_CANDIDATES_DIGEST_DOMAIN_V2
              if record["schema"] == graph_mod.SAME_AXIS_CANDIDATES_SCHEMA_V2
              else graph_mod.SAME_AXIS_CANDIDATES_DIGEST_DOMAIN)
    record["record_sha256"] = hashlib.sha256(json.dumps(
        {"domain": domain,
         "schema": record["schema"], "payload": body},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8")).hexdigest()
    return record


def _row_alternative_b(record, row, side):
    table = {entry[0]: entry for entry in record["readings"]}
    reading = table[row[3] if side == "left" else row[5]]
    alternative = row[4] if side == "left" else row[6]
    return alternative, reading[5][alternative]


# ---------------------------------------------------------------------------
# True pre/post candidate linkage
# ---------------------------------------------------------------------------

def test_pre_and_post_candidates_link_exactly(monkeypatch):
    readings, _signatures, matches, _trace, observation = _captured(monkeypatch)
    assert isinstance(observation, plane_graph.SameAxisCandidateObservation)
    assert len(observation.candidates) == len(observation.dispositions) == 4
    kept = [candidate for candidate, disposition in zip(
        observation.candidates, observation.dispositions)
        if disposition == plane_graph.SAME_AXIS_DISPOSITION_KEPT]
    dropped = [candidate for candidate, disposition in zip(
        observation.candidates, observation.dispositions)
        if disposition == plane_graph.SAME_AXIS_DISPOSITION_DROPPED]

    # The filter's survivors are exactly the post-filter solver inputs, by
    # object identity -- not a re-derived lookalike.
    assert len(kept) == len(matches) == 3
    assert all(any(candidate is match for candidate in kept)
               for match in matches)
    assert len(dropped) == 1
    assert dropped[0].left_b == 0.5
    assert dropped[0].left == "a" and dropped[0].right == "b"
    # The dropped candidate shares the group key of a retained survivor.
    survivor = next(candidate for candidate in kept
                    if candidate.left == "a" and candidate.right == "b")
    assert (dropped[0].left, dropped[0].right, dropped[0].left_axis,
            dropped[0].quadrant_of_right) == (
        survivor.left, survivor.right, survivor.left_axis,
        survivor.quadrant_of_right)

    solution = _solution(readings, matches)
    assert solution.alternatives
    record = graph_mod.failed_same_axis_candidates(
        readings, matches, solution, observation)
    assert record.recorded is True, record.refusal_reason
    payload = record.to_record()
    graph_mod.validate_same_axis_candidates(payload)

    assert payload["group_count"] == 3
    assert payload["row_count"] == 4
    assert payload["input_match_count"] == 3
    assert payload["axis"] == graph_mod.SAME_AXIS_CANDIDATES_AXIS

    dropped_rows = [row for row in payload["candidates"]
                    if row[17] == "dropped"]
    assert len(dropped_rows) == 1
    alternative, measured = _row_alternative_b(payload, dropped_rows[0], "left")
    assert alternative == 1
    assert measured[0] == 0.5 and measured[1] == 1.5
    assert dropped_rows[0][18] == plane_graph.SAME_AXIS_DISPOSITION_DROPPED
    assert dropped_rows[0][1] == -1

    # Every group binds its retained survivor's post-filter match index and the
    # shared scalar input-match identity.
    survivor_indices = [row[1] for row in payload["candidates"]
                        if row[17] == "kept"]
    assert sorted(survivor_indices) == [0, 1, 2]
    assert [group[5] for group in payload["groups"]] == [0, 1, 2]
    for group in payload["groups"]:
        identity = group[6]
        assert (group[1], group[2], group[3], group[4]) == (
            identity[0], identity[3], identity[2], identity[6])
        kept_row = next(row for row in payload["candidates"]
                        if row[0] == group[0] and row[1] == group[5])
        assert kept_row[17] == "kept"


def test_capture_binds_the_shared_assignment_and_input_identity(monkeypatch):
    readings, _signatures, matches, _trace, observation = _captured(monkeypatch)
    solution = _solution(readings, matches)
    record = graph_mod.failed_same_axis_candidates(
        readings, matches, solution, observation)
    bindings = graph_mod.failed_solver_input_bindings(readings, matches, solution)
    assert bindings.recorded is True, bindings.refusal_reason

    # Both observers name the same returned best assignment and the same ordered
    # post-filter solver-input identity sequence.
    assert record.failed_assignment_sha256 == bindings.failed_assignment_sha256
    assert record.input_match_identity_sha256 == (
        bindings.input_match_identity_sha256)
    assert record.input_match_count == bindings.input_match_count == 3

    expected_assignment = hashlib.sha256(json.dumps(
        {"domain": graph_mod.FAILED_ASSIGNMENT_DIGEST_DOMAIN,
         "schema": graph_mod.FAILED_ASSIGNMENT_DIGEST_SCHEMA,
         "payload": [[frame, solution.quadrants[frame]]
                     for frame in solution.frames]},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8")).hexdigest()
    assert record.failed_assignment_sha256 == expected_assignment


def test_capture_complete_record_replays_and_is_finite(monkeypatch):
    readings, _signatures, matches, _trace, observation = _captured(monkeypatch)
    solution = _solution(readings, matches)
    record = graph_mod.failed_same_axis_candidates(
        readings, matches, solution, observation)
    payload = record.to_record()
    graph_mod.validate_same_axis_candidates(payload)
    assert record.record_sha256 == (
        graph_mod.same_axis_candidate_record_digest(record))
    assert payload["record_sha256"] == record.record_sha256
    assert payload["schema"] == graph_mod.SAME_AXIS_CANDIDATES_SCHEMA
    assert payload["status"] == "complete"
    assert payload["basis"] == graph_mod.SAME_AXIS_CANDIDATES_BASIS

    table = {row[0]: row for row in payload["readings"]}
    referenced = {ref for row in payload["candidates"]
                  for ref in (row[3], row[5])}
    assert set(table) == referenced
    assert [row[0] for row in payload["readings"]] == sorted(referenced)
    for row in payload["candidates"]:
        # Every captured row is a world-axis-1 active group candidate whose
        # endpoints resolve through the retained alternative table.
        assert row[7] in (0, 1, 2, 3)
        for reading_ref, alternative in ((row[3], row[4]), (row[5], row[6])):
            reading = table[reading_ref]
            assert alternative < len(reading[5])
            b, distance, support = reading[5][alternative]
            assert b > 0.0 and distance > 0.0 and support >= 0.0
            assert math.isfinite(row[8]) and math.isfinite(row[16])
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    assert len(encoded) <= graph_mod.SAME_AXIS_CANDIDATES_MAX_RECORD_BYTES


def test_capture_digest_replay_refuses_resealed_tampering(monkeypatch):
    readings, _signatures, matches, _trace, observation = _captured(monkeypatch)
    solution = _solution(readings, matches)
    record = graph_mod.failed_same_axis_candidates(
        readings, matches, solution, observation).to_record()
    graph_mod.validate_same_axis_candidates(record)

    plain = json.loads(json.dumps(record))
    plain["row_count"] = 99
    with pytest.raises(ValueError):
        graph_mod.validate_same_axis_candidates(plain)

    cases = []

    delta = _resign(json.loads(json.dumps(record)))
    delta["candidates"][0][16] = float(delta["candidates"][0][16]) + 0.5
    cases.append(("delta", _resign(delta)))

    disposition = _resign(json.loads(json.dumps(record)))
    kept_row = next(row for row in disposition["candidates"]
                    if row[17] == "kept")
    kept_row[17] = "dropped"
    kept_row[1] = -1
    kept_row[18] = plane_graph.SAME_AXIS_DISPOSITION_DROPPED
    cases.append(("disposition", _resign(disposition)))

    alternative = _resign(json.loads(json.dumps(record)))
    alternative["candidates"][0][4] = int(alternative["candidates"][0][4]) + 1
    cases.append(("alternative", _resign(alternative)))

    identity = _resign(json.loads(json.dumps(record)))
    identity["groups"][0][6][7] = float(identity["groups"][0][6][7]) + 0.25
    cases.append(("survivor identity", _resign(identity)))

    key = _resign(json.loads(json.dumps(record)))
    key["groups"][0][3] = (int(key["groups"][0][3]) + 1) % 3
    cases.append(("group key", _resign(key)))

    duplicate = _resign(json.loads(json.dumps(record)))
    duplicate["groups"][1][1] = duplicate["groups"][0][1]
    duplicate["groups"][1][2] = duplicate["groups"][0][2]
    duplicate["groups"][1][3] = duplicate["groups"][0][3]
    duplicate["groups"][1][4] = duplicate["groups"][0][4]
    cases.append(("repeats a group key", _resign(duplicate)))

    axis = _resign(json.loads(json.dumps(record)))
    axis["axis"] = 2
    cases.append(("axis 1", _resign(axis)))

    assignment = _resign(json.loads(json.dumps(record)))
    assignment["assignments"][0] = (assignment["assignments"][0] + 1) % 4
    cases.append(("assignment digest", _resign(assignment)))

    for pattern, tampered in cases:
        with pytest.raises(ValueError, match=pattern):
            graph_mod.validate_same_axis_candidates(tampered)

    missing = json.loads(json.dumps(record))
    missing.pop("record_sha256")
    with pytest.raises(ValueError):
        graph_mod.validate_same_axis_candidates(missing)

    dropped_reading = _resign(json.loads(json.dumps(record)))
    referenced = dropped_reading["candidates"][0][3]
    dropped_reading["readings"] = [row for row in dropped_reading["readings"]
                                   if row[0] != referenced]
    _resign(dropped_reading)
    with pytest.raises(ValueError, match="missing reading"):
        graph_mod.validate_same_axis_candidates(dropped_reading)

    extra_reading = _resign(json.loads(json.dumps(record)))
    highest = max(row[0] for row in extra_reading["readings"])
    extra_reading["readings"].append(
        [highest + 1, "extra", 0, 1, 1, [[1.0, 1.0, 1.0]]])
    _resign(extra_reading)
    with pytest.raises(ValueError, match="exact set"):
        graph_mod.validate_same_axis_candidates(extra_reading)


def test_capture_refuses_malformed_and_over_cap_inputs(monkeypatch):
    readings, _signatures, matches, _trace, observation = _captured(monkeypatch)
    solution = _solution(readings, matches)

    for (broken_observation, broken_readings, broken_matches,
         broken_solution, field) in (
            (None, readings, matches, solution, "observation"),
            (observation, "not-readings", matches, solution, "readings"),
            (observation, readings, "not-matches", solution, "matches"),
            (observation, readings, matches, None, "solution")):
        refused = graph_mod.failed_same_axis_candidates(
            broken_readings, broken_matches, broken_solution,
            broken_observation)
        assert refused.recorded is False
        assert refused.refusal_reason == "invalid_same_axis_candidate_capture"
        assert refused.field == field
        payload = refused.to_record()
        assert set(payload) == {"schema", "status", "reason", "field",
                                "record_sha256"}
        assert "candidates" not in payload and "groups" not in payload
        assert refused.failed_assignment_sha256 == ""

    for attr, limit, field in (
            ("SAME_AXIS_CANDIDATES_MAX_GROUPS", 1, "groups"),
            ("SAME_AXIS_CANDIDATES_MAX_ROWS", 1, "rows"),
            ("SAME_AXIS_CANDIDATES_MAX_RECORD_BYTES", 64, "record_bytes")):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(graph_mod, attr, limit)
            refused = graph_mod.failed_same_axis_candidates(
                readings, matches, solution, observation)
        assert refused.recorded is False
        assert refused.refusal_reason == "budget_exhausted"
        assert refused.field == field
        payload = refused.to_record()
        assert set(payload) == {"schema", "status", "reason", "field",
                                "record_sha256"}
        assert "candidates" not in payload and "groups" not in payload
        graph_mod.validate_same_axis_candidates(payload)


# ---------------------------------------------------------------------------
# Output-neutral parity
# ---------------------------------------------------------------------------

def test_capture_preserves_matches_trace_and_solver(monkeypatch):
    readings, signatures = _candidate_fixture()
    monkeypatch.setattr(plane_graph, "match_plane_pair", _controlled_pair)

    def run(capture):
        trace = plane_graph.PlaneGraphTrace([])
        sink: list = []
        matches = plane_graph.match_plane_graph(
            readings, signatures, diagnostic_sink=trace,
            candidate_observation_sink=sink if capture else None)
        solution = _solution(readings, matches)
        record = (graph_mod.failed_same_axis_candidates(
            readings, matches, solution, sink[0]) if capture else None)
        return matches, trace, solution, record

    control_matches, control_trace, control_solution, _control_record = run(False)
    captured_matches, captured_trace, captured_solution, first = run(True)
    _matches, _trace, _sol, second = run(True)

    assert captured_matches == control_matches
    assert captured_trace == control_trace
    assert captured_solution == control_solution
    assert first.to_record() == second.to_record()
    assert first.recorded is True
    assert first.group_count == 3 and first.row_count == 4


def test_public_result_digest_is_unchanged_by_the_capture(
        records, monkeypatch):
    session, lattice, signatures = records
    readings, synthetic_signatures = _candidate_fixture()
    evidence = tuple(SimpleNamespace(signature=signature)
                     for signature in synthetic_signatures)
    monkeypatch.setattr(graph_mod, "_ordered_universe",
                        lambda *_a, **_k: (readings, evidence))
    monkeypatch.setattr(plane_graph, "match_plane_pair", _controlled_pair)
    original_solve = plane_graph.solve_component
    runs: dict = {}

    def run(capture):
        trace = plane_graph.PlaneGraphTrace([])
        captured: list = []
        seen: dict = {}

        def solver_spy(readings_arg, matches_arg, **kwargs):
            seen["matches"] = tuple(matches_arg)
            solution = original_solve(readings_arg, matches_arg, **kwargs)
            seen["solution"] = solution
            return solution

        monkeypatch.setattr(plane_graph, "solve_component", solver_spy)
        result = graph_mod._derive_image_derived_graph(
            session=session, lattice_result=lattice,
            signatures_result=signatures, diagnostic_sink=trace,
            same_axis_candidate_sink=captured.append if capture else None)
        runs[capture] = (result, trace, seen, captured)

    run(False)
    run(True)
    control_result, control_trace, control_seen, control_captured = runs[False]
    result, trace, seen, captured = runs[True]

    assert control_captured == []
    assert result == control_result
    assert result.result_sha256 == control_result.result_sha256
    assert result.status == "unavailable"
    assert trace == control_trace
    assert seen["matches"] == control_seen["matches"]
    assert seen["solution"] == control_seen["solution"]
    assert len(captured) == 1
    record = captured[0]
    assert record.recorded is True, record.refusal_reason
    assert record.group_count == 3 and record.row_count == 4
    graph_mod.validate_same_axis_candidates(record.to_record())


def test_capture_is_never_built_on_the_default_path(records, monkeypatch):
    session, lattice, signatures = records
    readings, synthetic_signatures = _candidate_fixture()
    evidence = tuple(SimpleNamespace(signature=signature)
                     for signature in synthetic_signatures)
    monkeypatch.setattr(graph_mod, "_ordered_universe",
                        lambda *_a, **_k: (readings, evidence))
    monkeypatch.setattr(plane_graph, "match_plane_pair", _controlled_pair)
    seen_kwargs: dict = {}
    original_match = plane_graph.match_plane_graph

    def match_spy(*args, **kwargs):
        seen_kwargs.update(kwargs)
        return original_match(*args, **kwargs)

    monkeypatch.setattr(plane_graph, "match_plane_graph", match_spy)
    calls = {"capture": 0}
    original_capture = graph_mod.failed_same_axis_candidates

    def capture_spy(*args, **kwargs):
        calls["capture"] += 1
        return original_capture(*args, **kwargs)

    monkeypatch.setattr(graph_mod, "failed_same_axis_candidates", capture_spy)
    result = graph_mod._derive_image_derived_graph(
        session=session, lattice_result=lattice, signatures_result=signatures)
    assert result.status == "unavailable"
    assert calls["capture"] == 0
    assert "candidate_observation_sink" not in seen_kwargs


def test_capture_does_not_change_the_real_matcher_trace(monkeypatch):
    left = make_reading("a", 1, 1.0, [(1.0, 6.0, 1.0)], share=0.6)
    right = make_reading("b", 1, 1.0, [(1.0, 6.0, 1.0)], share=0.6)
    rng = np.random.default_rng(721)
    cells = rng.random((20, 20))
    signatures = (make_signature(left, cells),
                  make_signature(right, np.roll(cells, (2, -3), axis=(0, 1))))
    readings = (left, right)

    control_trace = plane_graph.PlaneGraphTrace([])
    control = plane_graph.match_plane_graph(
        readings, signatures, diagnostic_sink=control_trace, max_shift_cells=8)
    capture_trace = plane_graph.PlaneGraphTrace([])
    sink: list = []
    captured = plane_graph.match_plane_graph(
        readings, signatures, diagnostic_sink=capture_trace,
        candidate_observation_sink=sink, max_shift_cells=8)

    assert control, "the synthetic real-matcher control must admit at least one pair"
    assert captured == control
    assert capture_trace == control_trace
    assert len(sink) == 1
    observation = sink[0]
    assert len(observation.candidates) == len(control)
    assert all(disposition == plane_graph.SAME_AXIS_DISPOSITION_KEPT
               for disposition in observation.dispositions)


# ---------------------------------------------------------------------------
# DS-IMG-SAMEAXIS-ALLAXIS-CAPTURE-V2: all-world-axis aggregate
#
# The same raw ``SameAxisCandidateObservation`` is captured for every world axis
# the active post-filter input references. The synthetic universe below gives one
# local reading per axis in each frame, so a quadrant-0 assignment maps local
# axis A to world axis A exactly (the literal ``_AXIS_TRANSFORMS`` row for
# quadrant 0). Frame ``aa`` carries a second axis-1 reading so one captured group
# legitimately retains two compatible matches, which is the v1 scalar-survivor
# assumption v2 has to generalize.
# ---------------------------------------------------------------------------

_V2_SPECS = {
    # (left frame, right frame, left reading, right reading, quadrant): score
    ("a", "b", "a0", "b0", 0): 0.9,
    ("a", "b", "a1", "b1", 0): 0.5,
    ("a", "b", "aa1", "b1", 0): 0.5,
    ("a", "b", "a2", "b2", 0): 0.9,
    ("a", "c", "a0", "c0", 0): 0.9,
    ("a", "c", "a1", "c1", 0): 0.9,
    ("b", "c", "b2", "c2", 0): 0.9,
}

# Identity map from the fixture's reading names to their objects; the controlled
# pair resolves a candidate's declared spec by reading identity.
_V2_NAMES: dict = {}


def _v2_fixture():
    """Three frames, one reading per world axis, plus a second axis-1 reading.

    Every reading is local-axis ``A`` at side ``+1`` with a distance of ``4.0``, so
    with every frame at quadrant 0 each match's world axis is its local axis and
    the two axis-1 ``aa``/``a1`` candidates share one group key and one delta.
    """
    def reading(frame, axis, distances, *, period=60.0):
        return make_reading(frame, axis, 1.0, distances, period=period)

    readings = (
        reading("a", 0, [(1.0, 4.0, 1.0)]),
        reading("a", 1, [(1.0, 4.0, 1.0)]),
        # A distinct period keeps this second axis-1 reading's canonical order
        # key unambiguous; every measurement that feeds a delta stays identical.
        reading("a", 1, [(1.0, 4.0, 1.0)], period=61.0),
        reading("a", 2, [(1.0, 4.0, 1.0)]),
        reading("b", 0, [(1.0, 4.0, 1.0)]),
        reading("b", 1, [(1.0, 4.0, 1.0)]),
        reading("b", 2, [(1.0, 4.0, 1.0)]),
        reading("c", 0, [(1.0, 4.0, 1.0)]),
        reading("c", 1, [(1.0, 4.0, 1.0)]),
        reading("c", 2, [(1.0, 4.0, 1.0)]),
    )
    names = ("a0", "a1", "aa1", "a2", "b0", "b1", "b2", "c0", "c1", "c2")
    signatures = tuple(make_signature(reading_value, _cells(100 + index))
                       for index, reading_value in enumerate(readings))
    return readings, signatures, dict(zip(names, readings))


def _v2_solution(readings):
    """A real returned-solution shape with every frame at quadrant 0."""
    _, index_by_object, frames = plane_graph._canonical_reading_index(readings)
    quadrants = tuple(0 for _ in frames)
    return plane_graph.ComponentSolution(
        frames=tuple(frames), quadrants={frame: 0 for frame in frames},
        translations={frame: (0.0, 0.0, 0.0) for frame in frames},
        axis_residuals={}, axis_resultants={}, scale_known=False,
        blocks_per_period={},
        reason="plane_integer_residual",
        alternatives=(plane_graph.ComponentAlternative(
            quadrants=quadrants, cost=0.0, resultants=(0.0, 0.0, 0.0),
            axis_observations=(0, 0, 0)),)), index_by_object


def _v2_controlled_pair(*, index_by_object):
    def pair(left_signature, right_signature, *, quadrant, max_shift_cells,
             refusals=None, diagnostic_sink=None, **_kwargs):
        left = left_signature.reading
        right = right_signature.reading
        left_name = next(name for name, reading in _V2_NAMES.items()
                         if reading is left)
        right_name = next(name for name, reading in _V2_NAMES.items()
                          if reading is right)
        score = _V2_SPECS.get(
            (left.frame, right.frame, left_name, right_name, int(quadrant)))
        admitted = score is not None
        if diagnostic_sink is not None:
            diagnostic_sink.pairs.append(plane_graph.PlanePairTrace(
                left.frame, right.frame, left.axis, right.axis, int(quadrant),
                admitted, "" if admitted else "no_signal",
                None, None, None, None, None, None,
                psr=5.0 if admitted else None, score=score,
                overlap_cells=64 if admitted else None,
                cell_shift_rc=(0, 0) if admitted else None))
        if not admitted:
            return None
        return plane_graph.PlaneMatch(
            left=left.frame, right=right.frame, left_axis=left.axis,
            right_axis=right.axis, left_b=1.0, right_b=1.0,
            quadrant_of_right=int(quadrant), psr=5.0, peak=1.0, second_peak=0.1,
            overlap_cells=64, integer_residual=None, score=score,
            left_plane=index_by_object[id(left)][1],
            right_plane=index_by_object[id(right)][1], cell_shift_rc=(0, 0))
    return pair


def _v2_captured(monkeypatch, axes=None):
    readings, signatures, names = _v2_fixture()
    _V2_NAMES.clear()
    _V2_NAMES.update(names)
    solution, index_by_object = _v2_solution(readings)
    monkeypatch.setattr(plane_graph, "match_plane_pair",
                        _v2_controlled_pair(index_by_object=index_by_object))
    trace = plane_graph.PlaneGraphTrace([])
    sink: list = []
    matches = plane_graph.match_plane_graph(
        readings, signatures, diagnostic_sink=trace,
        candidate_observation_sink=sink)
    record = graph_mod.failed_same_axis_candidates(
        readings, matches, solution, sink[0], axes)
    return readings, matches, solution, sink[0], record


def test_v2_captures_every_world_axis_group(monkeypatch):
    readings, matches, solution, observation, record = _v2_captured(
        monkeypatch, graph_mod.SAME_AXIS_CANDIDATES_AXES_ALL)
    assert record.recorded is True, record.refusal_reason
    payload = record.to_record()
    graph_mod.validate_same_axis_candidates(payload)

    assert payload["schema"] == graph_mod.SAME_AXIS_CANDIDATES_SCHEMA_V2
    assert payload["basis"] == graph_mod.SAME_AXIS_CANDIDATES_BASIS_V2
    assert payload["axes"] == [0, 1, 2]
    assert "axis" not in payload
    assert payload["input_match_count"] == len(matches) == 7

    # Every post-filter match is retained exactly once, and each captured group's
    # declared world axis is the production ``axis_transform`` of its left axis
    # under the captured assignment -- never the local axis relabelled.
    retained = [index for group in payload["groups"] for index in group[5]]
    assert sorted(retained) == list(range(7))
    assignment = dict(zip(payload["frames"], payload["assignments"]))
    axes_seen = set()
    for group in payload["groups"]:
        left_axis_local = group[3]
        left_world, _sign = plane_graph.axis_transform(
            left_axis_local, assignment[group[1]])
        assert left_world in (0, 1, 2)
        axes_seen.add(left_world)
        kept = [row for row in payload["candidates"]
                if row[0] == group[0] and row[17] == "kept"]
        assert [row[1] for row in kept] == list(group[5])
        assert len(group[6]) == len(group[5])
        for identity, row in zip(group[6], kept):
            assert identity[2] == left_axis_local == left_world
            assert identity[6] == row[7]
    assert axes_seen == {0, 1, 2}

    # The axis-0 and axis-2 families (local axis 0 and 2) are only visible in v2.
    local_axes = sorted({group[3] for group in payload["groups"]})
    assert local_axes == [0, 1, 2]
    v1 = graph_mod.failed_same_axis_candidates(
        readings, matches, solution, observation,
        graph_mod.SAME_AXIS_CANDIDATES_AXES_V1)
    assert v1.recorded is True, v1.refusal_reason
    assert sorted({group[3] for group in v1.groups}) == [1]


def test_v2_binds_a_multi_survivor_group(monkeypatch):
    readings, matches, solution, observation, record = _v2_captured(
        monkeypatch, graph_mod.SAME_AXIS_CANDIDATES_AXES_ALL)
    assert record.recorded is True, record.refusal_reason
    payload = record.to_record()

    multi = [group for group in payload["groups"] if len(group[5]) > 1]
    assert len(multi) == 1, "the synthetic fixture must retain one two-match group"
    group = multi[0]
    assert group[3] == 1, "the two retained matches share the local axis-1 group"
    assert len(group[5]) == 2 and len(group[6]) == 2
    assert group[5] == sorted(group[5]) and group[5][0] != group[5][1]
    kept = [row for row in payload["candidates"]
            if row[0] == group[0] and row[17] == "kept"]
    assert len(kept) == 2
    assert sorted(row[1] for row in kept) == group[5]
    by_index = {row[1]: row for row in kept}
    table = {row[0]: row for row in payload["readings"]}
    for match_index, identity in zip(group[5], group[6]):
        row = by_index[match_index]
        # identity = [left, left_plane, left_axis, right, right_plane, right_axis,
        #             quadrant, left_b, right_b]; row[4]/row[6] are alternative
        # indices, so the plane/axis identity is read back through the table.
        assert identity[0] == table[row[3]][1] and identity[1] == table[row[3]][2]
        assert identity[2] == table[row[3]][3]
        assert identity[3] == table[row[5]][1] and identity[4] == table[row[5]][2]
        assert identity[5] == table[row[5]][3]
        assert identity[6] == row[7]
        assert identity[7] == table[row[3]][5][row[4]][0]
        assert identity[8] == table[row[5]][5][row[6]][0]
    # The complete ordered input identity replays from the retained groups.
    graph_mod.validate_same_axis_candidates(payload)
    assert payload["group_count"] == len(payload["groups"])
    assert payload["row_count"] == len(payload["candidates"])


def test_v2_kept_coverage_is_exact_and_replays(monkeypatch):
    readings, matches, solution, observation, record = _v2_captured(
        monkeypatch, graph_mod.SAME_AXIS_CANDIDATES_AXES_ALL)
    payload = record.to_record()
    graph_mod.validate_same_axis_candidates(payload)

    # Every kept candidate index is present in exactly one group claim and no
    # dropped candidate carries an index.
    claimed = [index for group in payload["groups"] for index in group[5]]
    assert len(claimed) == len(set(claimed))
    for row in payload["candidates"]:
        if row[17] == "kept":
            assert row[1] in claimed and row[18] == ""
        else:
            assert row[1] == -1
            assert row[18] == plane_graph.SAME_AXIS_DISPOSITION_DROPPED

    # Dropping a retained index from its group breaks exact coverage even after
    # the record is resealed with a self-consistent new digest.
    trimmed = json.loads(json.dumps(payload))
    longest = max(range(len(trimmed["groups"])),
                  key=lambda index: len(trimmed["groups"][index][5]))
    trimmed["groups"][longest][5] = trimmed["groups"][longest][5][:-1]
    trimmed["groups"][longest][6] = trimmed["groups"][longest][6][:-1]
    with pytest.raises(ValueError):
        graph_mod.validate_same_axis_candidates(_resign(trimmed))

    # Removing a retained candidate row cannot be hidden: the group retention and
    # the exact reading/coverage re-derivation both reject it.
    removed = json.loads(json.dumps(payload))
    removed["candidates"] = [row for row in removed["candidates"]
                             if not (row[0] == 0 and row[1] == claimed[0])]
    removed["row_count"] = len(removed["candidates"])
    for position, row in enumerate(
            [row for row in removed["candidates"] if row[0] == 0]):
        row[2] = position
    with pytest.raises(ValueError):
        graph_mod.validate_same_axis_candidates(_resign(removed))

    # A group row that claims a retained index another group already owns is
    # rejected as an ambiguous retention binding.
    duplicated = json.loads(json.dumps(payload))
    other = next(group for group in duplicated["groups"]
                 if group[0] != duplicated["groups"][0][0]
                 and group[5][0] not in duplicated["groups"][0][5])
    duplicated["groups"][0][5] = [other[5][0]]
    duplicated["groups"][0][6] = [duplicated["groups"][0][6][0]]
    with pytest.raises(ValueError):
        graph_mod.validate_same_axis_candidates(_resign(duplicated))

    # An identity digest that no longer names the full ordered input is rejected.
    identity = json.loads(json.dumps(payload))
    identity["input_match_identity_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="identity digest does not replay"):
        graph_mod.validate_same_axis_candidates(_resign(identity))


def test_v2_refuses_malformed_axes_and_over_cap_inputs(monkeypatch):
    readings, matches, solution, observation, record = _v2_captured(
        monkeypatch, graph_mod.SAME_AXIS_CANDIDATES_AXES_ALL)
    assert record.recorded is True, record.refusal_reason

    for axes in ((0, 1), (0, 1, 2, 3), (0, 0, 1, 2), (2, 1, 0),
                 "not-axes", [], None):
        refused = graph_mod.failed_same_axis_candidates(
            readings, matches, solution, observation, axes)
        assert refused.recorded is False
        assert refused.refusal_reason == "invalid_same_axis_candidate_capture"
        assert refused.field == "axes"
        payload = refused.to_record()
        assert set(payload) == {"schema", "status", "reason", "field",
                                "record_sha256"}
        assert "candidates" not in payload and "groups" not in payload

    # The v1 path is unchanged for the default and for the explicit v1 axes tuple.
    for axes in (graph_mod.SAME_AXIS_CANDIDATES_AXES_V1,):
        v1 = graph_mod.failed_same_axis_candidates(
            readings, matches, solution, observation, axes)
        assert v1.recorded is True
        assert v1.schema == graph_mod.SAME_AXIS_CANDIDATES_SCHEMA

    for attr, limit, field in (
            ("SAME_AXIS_CANDIDATES_MAX_GROUPS", 1, "groups"),
            ("SAME_AXIS_CANDIDATES_MAX_ROWS", 1, "rows"),
            ("SAME_AXIS_CANDIDATES_MAX_RECORD_BYTES", 64, "record_bytes")):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(graph_mod, attr, limit)
            refused = graph_mod.failed_same_axis_candidates(
                readings, matches, solution, observation,
                graph_mod.SAME_AXIS_CANDIDATES_AXES_ALL)
        assert refused.recorded is False
        assert refused.refusal_reason == "budget_exhausted"
        assert refused.field == field
        payload = refused.to_record()
        assert set(payload) == {"schema", "status", "reason", "field",
                                "record_sha256"}
        assert "candidates" not in payload and "groups" not in payload
        graph_mod.validate_same_axis_candidates(payload)


def test_v2_digest_and_output_are_deterministic_and_default_parity(monkeypatch):
    readings, matches, solution, observation, first = _v2_captured(
        monkeypatch, graph_mod.SAME_AXIS_CANDIDATES_AXES_ALL)
    second = graph_mod.failed_same_axis_candidates(
        readings, matches, solution, observation,
        graph_mod.SAME_AXIS_CANDIDATES_AXES_ALL)
    assert first.to_record() == second.to_record()
    assert first.record_sha256 == second.record_sha256
    assert first.record_sha256 == graph_mod.same_axis_candidate_record_digest(first)

    # The default call is still the frozen v1 record; the identity/assignment
    # digests are shared, so the two records still describe one solver input.
    v1 = graph_mod.failed_same_axis_candidates(
        readings, matches, solution, observation)
    assert v1.schema == graph_mod.SAME_AXIS_CANDIDATES_SCHEMA
    assert v1.basis == graph_mod.SAME_AXIS_CANDIDATES_BASIS
    assert v1.axes == graph_mod.SAME_AXIS_CANDIDATES_AXES_V1
    assert v1.failed_assignment_sha256 == first.failed_assignment_sha256
    assert v1.input_match_identity_sha256 == first.input_match_identity_sha256
    assert v1.input_match_count == first.input_match_count
    graph_mod.validate_same_axis_candidates(v1.to_record())
    graph_mod.validate_same_axis_candidates(v1.to_record())
    # Re-sealing the v1 payload under the v2 domain must not validate.
    forged = json.loads(json.dumps(v1.to_record()))
    forged["axes"] = [0, 1, 2]
    forged.pop("axis", None)
    forged["schema"] = graph_mod.SAME_AXIS_CANDIDATES_SCHEMA_V2
    with pytest.raises(ValueError):
        graph_mod.validate_same_axis_candidates(_resign(forged))


def test_v2_public_result_digest_is_unchanged_by_all_axis_capture(
        records, monkeypatch):
    session, lattice, signatures = records
    readings, synthetic_signatures, names = _v2_fixture()
    _V2_NAMES.clear()
    _V2_NAMES.update(names)
    solution, index_by_object = _v2_solution(readings)
    # The capture runs only on the solver-abstention path, so the spy returns the
    # same returned-solution shape with a typed abstention reason while the real
    # solver is still called and its matches/solution recorded for comparison.
    abstaining = plane_graph.ComponentSolution(
        frames=solution.frames, quadrants=dict(solution.quadrants),
        translations=dict(solution.translations), axis_residuals={},
        axis_resultants={}, scale_known=False, blocks_per_period={},
        reason="plane_integer_residual",
        alternatives=(plane_graph.ComponentAlternative(
            quadrants=tuple(solution.quadrants[frame] for frame in solution.frames),
            cost=0.0, resultants=(0.0, 0.0, 0.0),
            axis_observations=(0, 0, 0)),))
    evidence = tuple(SimpleNamespace(signature=signature)
                     for signature in synthetic_signatures)
    monkeypatch.setattr(graph_mod, "_ordered_universe",
                        lambda *_a, **_k: (readings, evidence))
    monkeypatch.setattr(plane_graph, "match_plane_pair",
                        _v2_controlled_pair(index_by_object=index_by_object))
    original_solve = plane_graph.solve_component
    runs: dict = {}

    def run(axes):
        trace = plane_graph.PlaneGraphTrace([])
        captured: list = []
        seen: dict = {}

        def solver_spy(readings_arg, matches_arg, **kwargs):
            seen["matches"] = tuple(matches_arg)
            original_solve(readings_arg, matches_arg, **kwargs)
            seen["solution"] = abstaining
            return abstaining

        monkeypatch.setattr(plane_graph, "solve_component", solver_spy)
        result = graph_mod._derive_image_derived_graph(
            session=session, lattice_result=lattice,
            signatures_result=signatures, diagnostic_sink=trace,
            same_axis_candidate_sink=captured.append, same_axis_capture_axes=axes)
        runs[axes] = (result, trace, seen, captured)

    run(None)
    run(graph_mod.SAME_AXIS_CANDIDATES_AXES_ALL)
    control_result, control_trace, control_seen, control_captured = runs[None]
    result, trace, seen, captured = runs[graph_mod.SAME_AXIS_CANDIDATES_AXES_ALL]

    # The abstention path always produces one record; the default/``None`` call
    # stays the frozen v1 single-axis record while the explicit all-axis call is
    # v2, and the returned public result is byte-identical either way.
    assert len(control_captured) == 1
    assert control_captured[0].schema == graph_mod.SAME_AXIS_CANDIDATES_SCHEMA
    assert control_captured[0].axes == graph_mod.SAME_AXIS_CANDIDATES_AXES_V1
    assert result == control_result
    assert result.result_sha256 == control_result.result_sha256
    assert result.status == "unavailable"
    assert trace == control_trace
    assert seen["matches"] == control_seen["matches"]
    assert seen["solution"] == control_seen["solution"]
    assert len(captured) == 1
    record = captured[0]
    assert record.recorded is True, record.refusal_reason
    assert record.schema == graph_mod.SAME_AXIS_CANDIDATES_SCHEMA_V2
    assert sorted({group[3] for group in record.groups}) == [0, 1, 2]
    graph_mod.validate_same_axis_candidates(record.to_record())


# ---------------------------------------------------------------------------
# DS-IMG-SAMEAXIS-ALLAXIS-CAPTURE-V2-CORRECTION: subset survivors, full input
#
# The realistic cardinality the rejected v2 rule could not represent: 38 ordered
# post-filter solver inputs, only 9 solver-active same-axis groups (3 per world
# axis), 10 retained survivors including one two-survivor group, and 28 inputs
# no captured group contains. The record carries the full 38-row
# ``input_identities`` domain, so the shared identity digest replays even though
# the candidate rows are only the bounded subset. Built at the record-payload
# level only; the strict validator re-derives every row, key, retention and
# digest, and no matcher/solver/filter is run or weakened.
# ---------------------------------------------------------------------------

def realistic_v2_subset_payload():
    """One 38-input / 9-group / 10-survivor v2 record with 28 uncaptured inputs."""
    frames = ("image-0000", "image-0001", "image-0002")
    assignments = (0, 0, 0)
    frame_quadrant = dict(zip(frames, assignments))

    def reading(index, frame, plane, axis):
        return [index, frame, plane, axis, 1, [[1.0, 4.0, 1.0]]]

    readings = (
        reading(0, frames[0], 0, 0), reading(1, frames[1], 1, 0),
        reading(2, frames[2], 2, 0), reading(3, frames[0], 3, 1),
        reading(4, frames[1], 4, 1), reading(5, frames[2], 5, 1),
        reading(6, frames[0], 6, 1), reading(7, frames[0], 7, 2),
        reading(8, frames[1], 8, 2), reading(9, frames[2], 9, 2))

    def candidate(group, match, position, left, right):
        # delta = +1*4.0 - (+1)*+1*4.0 = 0.0 for every endpoint pair.
        return [group, match, position, left, 0, right, 0, 0, 0.9, 5.0, False,
                1.0, 0.1, 64, 0, 0, 0.0, "kept", ""]

    # (group, left frame, right frame, left axis, claim, endpoint reading pairs)
    layout = (
        (0, frames[0], frames[1], 0, (1,), ((0, 1),)),
        (1, frames[0], frames[2], 0, (2,), ((0, 2),)),
        (2, frames[1], frames[2], 0, (3,), ((1, 2),)),
        (3, frames[0], frames[1], 1, (4, 5), ((3, 4), (6, 4))),
        (4, frames[0], frames[2], 1, (6,), ((3, 5),)),
        (5, frames[1], frames[2], 1, (7,), ((4, 5),)),
        (6, frames[0], frames[1], 2, (8,), ((7, 8),)),
        (7, frames[0], frames[2], 2, (9,), ((7, 9),)),
        (8, frames[1], frames[2], 2, (10,), ((8, 9),)))

    groups = []
    candidates = []
    for group, left_frame, right_frame, left_axis, claim, pairs in layout:
        identities = []
        for position, (match_index, (left_ref, right_ref)) in enumerate(
                zip(claim, pairs)):
            candidates.append(candidate(
                group, match_index, position, left_ref, right_ref))
            left_row = readings[left_ref]
            right_row = readings[right_ref]
            identities.append([
                left_row[1], left_row[2], left_row[3],
                right_row[1], right_row[2], right_row[3], 0, 1.0, 1.0])
        groups.append([group, left_frame, right_frame, left_axis, 0,
                       list(claim), identities])

    def uncaptured_identity(index):
        return [frames[index % 3], index, index % 3, frames[(index + 1) % 3],
                index + 1, (index + 1) % 3, 0, 1.0, 1.0]

    input_identities = [uncaptured_identity(index) for index in range(38)]
    for group in groups:
        for match_index, identity in zip(group[5], group[6]):
            input_identities[match_index] = list(identity)

    record = graph_mod.SameAxisCandidateCapture(
        schema=graph_mod.SAME_AXIS_CANDIDATES_SCHEMA_V2, recorded=True,
        basis=graph_mod.SAME_AXIS_CANDIDATES_BASIS_V2, refusal_reason=None,
        failed_assignment_sha256=graph_mod._failed_assignment_digest(
            frames, frame_quadrant),
        input_match_identity_sha256=graph_mod._alias_domain_digest(
            graph_mod.SOLVER_INPUT_BINDINGS_INPUT_MATCH_IDENTITY_DOMAIN,
            graph_mod.SOLVER_INPUT_BINDINGS_INPUT_MATCH_IDENTITY_SCHEMA,
            input_identities),
        input_match_count=38, group_count=len(groups),
        row_count=len(candidates), frames=frames, assignments=assignments,
        readings=tuple(readings), groups=tuple(groups),
        candidates=tuple(candidates), axes=graph_mod.SAME_AXIS_CANDIDATES_AXES_ALL,
        input_identities=tuple(
            tuple(identity) for identity in input_identities))
    return record.to_record()


def test_v2_subset_survivors_replay_the_full_input_identity_digest():
    payload = realistic_v2_subset_payload()
    graph_mod.validate_same_axis_candidates(payload)

    # Two separate domains: the full 38-input identity list and the bounded
    # 9-group / 10-survivor capture that references only a subset of it.
    assert payload["input_match_count"] == 38
    assert len(payload["input_identities"]) == 38
    assert payload["group_count"] == 9
    claims = [index for group in payload["groups"] for index in group[5]]
    assert len(claims) == 10 and len(set(claims)) == 10
    assert set(claims) != set(range(38))
    assert len(set(range(38)) - set(claims)) == 28
    assert max(len(group[5]) for group in payload["groups"]) == 2
    assert sorted({group[3] for group in payload["groups"]}) == [0, 1, 2]

    # Every captured group's ordered survivor index/identity pair is the exact
    # full-input identity row at that index.
    for group in payload["groups"]:
        for match_index, identity in zip(group[5], group[6]):
            assert identity == payload["input_identities"][match_index]

    # No candidate row pretends an unselected input has a captured candidate.
    kept = sorted(row[1] for row in payload["candidates"]
                  if row[17] == "kept")
    assert kept == sorted(claims)
    assert all(row[1] == -1 for row in payload["candidates"]
               if row[17] != "kept")


def test_v2_subset_binding_refuses_tampered_survivors_and_identities():
    payload = realistic_v2_subset_payload()
    graph_mod.validate_same_axis_candidates(payload)

    # Missing survivor: drop one claim and its parallel identity, keep the rows.
    # The independent active-group derivation still expects both indices, so the
    # active-survivor coverage check rejects it even though the remaining group
    # rows are internally self-consistent.
    missing = json.loads(json.dumps(payload))
    multi = max(missing["groups"], key=lambda group: len(group[5]))
    multi[5] = multi[5][:-1]
    multi[6] = multi[6][:-1]
    with pytest.raises(ValueError, match="active survivors do not replay"):
        graph_mod.validate_same_axis_candidates(_resign(missing))

    # Duplicate survivor: one group claims an index another group already owns.
    duplicate = json.loads(json.dumps(payload))
    duplicate["groups"][1][5] = list(duplicate["groups"][0][5])
    duplicate["groups"][1][6] = list(duplicate["groups"][0][6])
    with pytest.raises(ValueError):
        graph_mod.validate_same_axis_candidates(_resign(duplicate))

    # Out-of-range survivor: claim an index at the declared input count.
    out_of_range = json.loads(json.dumps(payload))
    out_of_range["groups"][-1][5] = [38]
    out_of_range["groups"][-1][6] = [list(payload["input_identities"][0])]
    with pytest.raises(ValueError, match="out of range"):
        graph_mod.validate_same_axis_candidates(_resign(out_of_range))

    # A survivor identity that no longer matches its full-input identity row.
    # Substituting another input's identity also changes that index's derived key,
    # so the independent active-group coverage check rejects the record first.
    crossed = json.loads(json.dumps(payload))
    crossed["input_identities"][1] = list(payload["input_identities"][0])
    with pytest.raises(ValueError,
                       match="does not cover the active group set"):
        graph_mod.validate_same_axis_candidates(_resign(crossed))

    # Input-identity domain: a missing row no longer matches the input count.
    short = json.loads(json.dumps(payload))
    short["input_identities"] = short["input_identities"][:-1]
    with pytest.raises(ValueError, match="do not match the input count"):
        graph_mod.validate_same_axis_candidates(_resign(short))

    # Input-identity domain: a malformed row shape is rejected.
    malformed = json.loads(json.dumps(payload))
    malformed["input_identities"][0] = [payload["frames"][0], 0, 0]
    with pytest.raises(ValueError, match="input identity must be a list"):
        graph_mod.validate_same_axis_candidates(_resign(malformed))

    # Input-identity domain: a reordered list keeps its length but not its digest.
    reordered = json.loads(json.dumps(payload))
    reordered["input_identities"][0], reordered["input_identities"][11] = (
        reordered["input_identities"][11], reordered["input_identities"][0])
    with pytest.raises(ValueError, match="identity digest does not replay"):
        graph_mod.validate_same_axis_candidates(_resign(reordered))


# ---------------------------------------------------------------------------
# DS-IMG-SAMEAXIS-ALLAXIS-CAPTURE-V2-COVERAGE: exact active-group coverage
#
# The full 38-row input identity list independently implies the 9 solver-active
# groups (3 per world axis). A v2 record must therefore capture exactly that set
# and, per group, exactly the derived active input indices -- it may never drop an
# entire active group or claim a non-active survivor. The same identities make the
# in-range-but-inactive complement (here 28 inputs) explicit, so it must not be
# required to carry candidate rows.
# ---------------------------------------------------------------------------

def test_v2_requires_every_world_axis_and_exact_active_group_coverage():
    payload = realistic_v2_subset_payload()
    graph_mod.validate_same_axis_candidates(payload)

    # The independently derived active set is exactly the captured 9 groups with
    # exactly the claimed survivor indices; the 28 inactive inputs stay uncaptured.
    claims = [index for group in payload["groups"] for index in group[5]]
    assert payload["axes"] == [0, 1, 2]
    assert payload["group_count"] == 9 and len(claims) == 10
    assert len(set(range(38)) - set(claims)) == 28

    # V2 is defined as all axes: a self-consistent subset must not reseal. ``[0, 1]``
    # keeps a valid nonempty canonical list, so only the exact-axes rule rejects it.
    subset = json.loads(json.dumps(payload))
    subset["axes"] = [0, 1]
    with pytest.raises(ValueError, match="must bind every world axis"):
        graph_mod.validate_same_axis_candidates(_resign(subset))

    # Dropping a whole active group fails closed. Re-sealing alone cannot hide it,
    # because the still-present input identities at its survivor indices imply the
    # group; the check is what keeps a truncated capture from validating.
    dropped = json.loads(json.dumps(payload))
    removed_key = tuple(dropped["groups"][-1][:4])
    dropped["groups"] = dropped["groups"][:-1]
    dropped["group_count"] = len(dropped["groups"])
    remaining = sorted(row[0] for row in dropped["groups"])
    dropped["candidates"] = [row for row in dropped["candidates"]
                             if row[0] in remaining]
    dropped["candidates"] = [[remaining.index(row[0])] + row[1:]
                             for row in dropped["candidates"]]
    dropped["row_count"] = len(dropped["candidates"])
    active_indices = {tuple(group[:4]): list(group[5])
                      for group in payload["groups"]}
    assert removed_key in active_indices
    with pytest.raises(ValueError,
                       match="does not cover the active group set"):
        graph_mod.validate_same_axis_candidates(_resign(dropped))

    # Rewriting the omitted group's input identities so they no longer look active
    # is caught by the same exactness rule from the other side: the rewritten keys
    # are derived but never captured.
    foreign = json.loads(json.dumps(payload))
    removed_indices = active_indices[removed_key]
    removed_group = foreign["groups"][-1]
    assert list(removed_group[5]) == removed_indices
    for match_index in removed_indices:
        foreign["input_identities"][match_index] = [
            foreign["frames"][2], match_index, 0,
            foreign["frames"][1], match_index, 0, 0, 1.0, 1.0]
    foreign["groups"] = foreign["groups"][:-1]
    foreign["group_count"] = len(foreign["groups"])
    remaining = sorted(row[0] for row in foreign["groups"])
    foreign["candidates"] = [row for row in foreign["candidates"]
                             if row[0] in remaining]
    foreign["candidates"] = [[remaining.index(row[0])] + row[1:]
                             for row in foreign["candidates"]]
    foreign["row_count"] = len(foreign["candidates"])
    foreign["input_match_identity_sha256"] = graph_mod._alias_domain_digest(
        graph_mod.SOLVER_INPUT_BINDINGS_INPUT_MATCH_IDENTITY_DOMAIN,
        graph_mod.SOLVER_INPUT_BINDINGS_INPUT_MATCH_IDENTITY_SCHEMA,
        foreign["input_identities"])
    with pytest.raises(ValueError,
                       match="does not cover the active group set"):
        graph_mod.validate_same_axis_candidates(_resign(foreign))


def test_v2_refuses_foreign_frame_and_non_active_survivors():
    payload = realistic_v2_subset_payload()
    graph_mod.validate_same_axis_candidates(payload)

    # A claimed survivor that is in range but not active for that group's key is
    # rejected even when the group's own identity rows still replay.
    crossed = json.loads(json.dumps(payload))
    inactive = next(index for index in range(38)
                    if index not in {value
                                     for group in payload["groups"]
                                     for value in group[5]})
    target = crossed["groups"][3]
    target[5] = [inactive]
    target[6] = [list(payload["input_identities"][inactive])]
    crossed["row_count"] = len(crossed["candidates"])
    with pytest.raises(ValueError):
        graph_mod.validate_same_axis_candidates(_resign(crossed))

    # An input identity that names a frame absent from the record's assignments
    # cannot be silently classified; it fails closed. The identity chosen is in the
    # inactive complement, so the exact active-group set is unchanged and only the
    # foreign-frame rule rejects the record.
    foreign = json.loads(json.dumps(payload))
    foreign["input_identities"][inactive][0] = "image-9999"
    foreign["input_match_identity_sha256"] = graph_mod._alias_domain_digest(
        graph_mod.SOLVER_INPUT_BINDINGS_INPUT_MATCH_IDENTITY_DOMAIN,
        graph_mod.SOLVER_INPUT_BINDINGS_INPUT_MATCH_IDENTITY_SCHEMA,
        foreign["input_identities"])
    with pytest.raises(ValueError, match="foreign frame"):
        graph_mod.validate_same_axis_candidates(_resign(foreign))
