"""DS-R7 focused controls for the complete solver-input binding observer.

These controls exercise the evaluation-only ``SolverInputBindings`` record in
``image_derived_graph`` against a synthetic component that reproduces the shape
of the retained failure universe: 38 post-same-axis solver inputs, competing
``b`` alternatives that share one match identity, and a best assignment under
which only part of the input set is active. They cover the complete payload and
its fixed byte budget, a genuine abstaining/best-assignment solve, every
endpoint's alternative index binding to the canonical reading's measured
alternative table, the derived active flag, digest replay/tampering refusal,
typed refusal on malformed or partial input, the shared failed-assignment digest
with the active-edge alias record, default result/digest parity, and the runner's
observation binding.

No production gate, matcher, solver, threshold, truth, camera, mask or reference
input is read or changed here; no image or reconstruction workload runs.
"""
from __future__ import annotations

import hashlib
import json
import math

import pytest

from test_plane_graph import LABELS, SIGNS, make_match, make_reading
from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
from mc_photogrammetry.pipeline import image_derived_graph_attribution as attribution_mod
from mc_photogrammetry.reconstruction import lattice_planes
from mc_photogrammetry.reconstruction import plane_graph
from scripts import run_image_derived_graph_stage as stage_subject


TARGET_INPUTS = 38


def _alias_caps():
    return {
        "max_readings": graph_mod.ACTIVE_EDGE_ALIASES_MAX_READINGS,
        "max_matches": graph_mod.ACTIVE_EDGE_ALIASES_MAX_MATCHES,
        "max_active_edges": graph_mod.ACTIVE_EDGE_ALIASES_MAX_ACTIVE_EDGES,
        "max_endpoint_alternatives": (
            graph_mod.ACTIVE_EDGE_ALIASES_MAX_ENDPOINT_ALTERNATIVES),
        "max_frame_id_bytes": graph_mod.ACTIVE_EDGE_ALIASES_MAX_FRAME_ID_BYTES,
        "max_record_bytes": graph_mod.ACTIVE_EDGE_ALIASES_MAX_RECORD_BYTES,
    }


def _binding_universe():
    """38 mixed relative-quadrant inputs over a known lattice with duplicates.

    Every reading keeps two measured alternatives (``b`` 1.0 and 0.5); matches
    are emitted in alternating true/opposite relative quadrants and all four
    ``b`` combinations, so the 38 solver inputs contain competing identities and
    a genuine active/inactive mix under the pinned best assignment.
    """
    frames = ("v0", "v1", "v2")
    quadrants = {"v0": 0, "v1": 1, "v2": 2}
    eyes = {"v0": (5, 2, 5), "v1": (6, 3, 7), "v2": (4, 1, 9)}
    planes = ((0, 0), (1, 0), (2, 0), (0, 10))
    built = []
    for frame in frames:
        quadrant = quadrants[frame]
        for world_axis, coord in planes:
            label = LABELS[(world_axis, quadrant)]
            world_side = 1.0 if coord > eyes[frame][world_axis] else -1.0
            distance = abs(eyes[frame][world_axis] - coord)
            if distance <= 0.0:
                continue
            side = world_side * SIGNS[(label, quadrant)]
            reading = make_reading(frame, label, side, [
                (1.0, distance, 1.0), (0.5, distance / 2.0, 0.8)])
            built.append((frame, world_axis, coord, reading))
    readings = [entry[3] for entry in built]
    index: dict[tuple[str, int, int], int] = {}
    for frame in frames:
        ordered = sorted((entry for entry in built if entry[0] == frame),
                         key=lambda entry: lattice_planes.reading_order_key(entry[3]))
        for position, (_frame, world_axis, coord, _reading) in enumerate(ordered):
            index[(frame, world_axis, coord)] = position
    reading_pairs = []
    for first_position, first in enumerate(frames):
        for second in frames[first_position + 1:]:
            true_relative = (quadrants[second] - quadrants[first]) % 4
            for world_axis, coord in planes:
                if ((first, world_axis, coord) not in index
                        or (second, world_axis, coord) not in index):
                    continue
                reading_pairs.append(
                    (first, second, world_axis, coord, true_relative))
    combos = ((1.0, 1.0), (0.5, 1.0), (1.0, 0.5), (0.5, 0.5))
    matches = []
    variant = 0
    for (first, second, world_axis, coord, true_relative) in reading_pairs:
        for left_b, right_b in combos:
            quadrant = (true_relative if variant % 2 == 0
                        else (true_relative + 2) % 4)
            variant += 1
            matches.append(make_match(
                first, second, LABELS[(world_axis, quadrants[first])],
                LABELS[(world_axis, quadrants[second])], quadrant,
                index[(first, world_axis, coord)],
                index[(second, world_axis, coord)],
                left_b=left_b, right_b=right_b))
            if len(matches) == TARGET_INPUTS:
                break
        if len(matches) == TARGET_INPUTS:
            break
    assert len(matches) == TARGET_INPUTS
    constraints = (("v0", "v1", 1), ("v0", "v2", 2), ("v1", "v2", 1))
    solution = plane_graph.solve_component(
        readings, matches, quadrant_constraints=constraints,
        integer_tolerance=0.15, min_resultant=0.90, ambiguous_margin=0.02)
    assert solution.alternatives
    assert dict(solution.quadrants) == quadrants
    return readings, matches, solution, quadrants


@pytest.fixture(scope="module")
def binding_inputs():
    return _binding_universe()


def _record():
    readings, matches, solution, _quadrants = _binding_universe()
    value = graph_mod.failed_solver_input_bindings(readings, matches, solution)
    assert value.recorded is True, value.refusal_reason
    return readings, matches, solution, value


def _resign(record):
    body = {key: value for key, value in record.items()
            if key != "record_sha256"}
    record["record_sha256"] = hashlib.sha256(json.dumps(
        {"domain": graph_mod.SOLVER_INPUT_BINDINGS_DIGEST_DOMAIN,
         "schema": record["schema"], "payload": body},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8")).hexdigest()
    return record


def _expected_assignment_digest(frames, quadrants):
    return hashlib.sha256(json.dumps(
        {"domain": graph_mod.FAILED_ASSIGNMENT_DIGEST_DOMAIN,
         "schema": graph_mod.FAILED_ASSIGNMENT_DIGEST_SCHEMA,
         "payload": [[frame, quadrants[frame]] for frame in frames]},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8")).hexdigest()


def _watch_replay(record):
    graph_mod.validate_solver_input_bindings(record)


def test_bindings_complete_record_covers_all_inputs_and_replays():
    _readings, matches, solution, value = _record()
    record = value.to_record()
    _watch_replay(record)
    assert record["schema"] == graph_mod.SOLVER_INPUT_BINDINGS_SCHEMA
    assert record["status"] == "complete"
    assert record["basis"] == graph_mod.SOLVER_INPUT_BINDINGS_BASIS
    assert record["input_match_count"] == len(matches) == TARGET_INPUTS
    assert len(record["edges"]) == TARGET_INPUTS
    assert record["active_edge_count"] == sum(
        1 for row in record["edges"] if row[4])
    assert record["frames"] == list(solution.frames)
    assert record["assignments"] == [solution.quadrants[frame]
                                     for frame in solution.frames]
    # The full 38-input contract: edge rows and endpoint rows are the declared
    # fixed-width schemas, and every input match index appears exactly once.
    assert all(len(row) == 5 for row in record["edges"])
    assert all(len(endpoint) == 8 for row in record["edges"]
               for endpoint in row[1:3])
    assert [row[0] for row in record["edges"]] == list(range(TARGET_INPUTS))
    assert all(len(row) == 6 for row in record["readings"])
    assert all(len(entry) == 3 for row in record["readings"]
               for entry in row[5])
    json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert value.record_sha256 == graph_mod.solver_input_bindings_record_digest(
        value)
    assert record["record_sha256"] == value.record_sha256


def test_bindings_every_endpoint_resolves_to_its_reading_alternative_table():
    _readings, _matches, _solution, value = _record()
    record = value.to_record()
    table = {row[0]: row for row in record["readings"]}
    frames = record["frames"]
    assignments = dict(zip(frames, record["assignments"]))
    for row in record["edges"]:
        match_index, left, right, relative, active = row
        assert match_index in range(TARGET_INPUTS)
        assert relative in (0, 1, 2, 3) and isinstance(active, bool)
        for endpoint in (left, right):
            reading_ref, alternative_index, b, distance, support, side, sign, axis = endpoint
            reading_row = table[reading_ref]
            _index, frame, _plane, expected_axis, expected_side, alternatives = reading_row
            assert frame in assignments
            assert alternative_index < len(alternatives)
            assert [b, distance, support] == alternatives[alternative_index]
            assert axis == expected_axis
            assert side == expected_side and side in (-1, 1)
            world, expected_sign = plane_graph.axis_transform(axis, assignments[frame])
            assert sign == expected_sign and sign in (-1, 1)
            # The table's canonical reading and full measured alternative table
            # are what the alternative index binds to; a flatter record could
            # not re-derive the value from the table alone.
            assert b == alternatives[alternative_index][0]
            assert distance == alternatives[alternative_index][1]
            assert support == alternatives[alternative_index][2]
            assert math.isfinite(b) and b > 0.0
            assert math.isfinite(distance) and distance > 0.0
            assert math.isfinite(support) and support >= 0.0
            assert expected_axis in (0, 1, 2)


def test_bindings_active_flag_matches_best_assignment_predicate():
    readings, matches, solution, value = _record()
    record = value.to_record()
    quadrants = dict(solution.quadrants)
    # Independent re-derivation of the solver's own activation predicate, from
    # the source matches -- never from the record's own active flag.
    expected = []
    for match in matches:
        left_world, _ = plane_graph.axis_transform(
            match.left_axis, quadrants[match.left])
        right_world, _ = plane_graph.axis_transform(
            match.right_axis, quadrants[match.right])
        active = bool(
            ((quadrants[match.right] - quadrants[match.left]) % 4
             == match.quadrant_of_right) and left_world == right_world)
        expected.append(active)
    assert [row[4] for row in record["edges"]] == expected
    assert record["active_edge_count"] == sum(expected)
    assert 0 < record["active_edge_count"] < TARGET_INPUTS
    assert any(not flag for flag in expected) and any(expected)
    assert readings  # the source universe is real, not the record's own table


def test_bindings_reading_table_is_exact_union_and_ordered():
    _readings, _matches, _solution, value = _record()
    record = value.to_record()
    referenced = set()
    for row in record["edges"]:
        referenced.add(row[1][0])
        referenced.add(row[2][0])
    table = {row[0]: row for row in record["readings"]}
    assert set(table) == referenced
    assert [row[0] for row in record["readings"]] == sorted(referenced)
    seen_canonical = set()
    for row in record["readings"]:
        canonical = (row[1], row[2])
        assert canonical not in seen_canonical
        seen_canonical.add(canonical)
        previous = None
        for b, distance, support in row[5]:
            if previous is not None:
                assert support <= previous[0]
                if support == previous[0]:
                    assert b <= previous[1]
            previous = (support, b)
        assert all(value > 0.0 for value in (row[5][0][0], row[5][0][1]))


def test_bindings_competing_duplicate_identities_are_recorded_not_refused():
    _readings, matches, _solution, value = _record()
    record = value.to_record()
    _watch_replay(record)
    keys = [(match.left, match.left_plane, match.right, match.right_plane,
             match.quadrant_of_right) for match in matches]
    duplicates = {key for key in keys if keys.count(key) > 1}
    assert duplicates, "the fixture must carry competing alternative identities"
    # The record keeps every input, so a duplicate identity is described, not
    # collapsed or refused.
    assert record["input_match_count"] == len(keys)
    assert record["status"] == "complete"


def test_bindings_failed_assignment_digest_binds_alias_record():
    readings, matches, solution, value = _record()
    aliases = graph_mod._failed_active_edges(
        readings, matches, solution, **_alias_caps())
    assert aliases.recorded is True, aliases.refusal_reason
    expected = _expected_assignment_digest(solution.frames, solution.quadrants)
    assert value.failed_assignment_sha256 == expected
    assert aliases.failed_assignment_sha256 == expected
    assert value.to_record()["failed_assignment_sha256"] == expected
    # The shared input-match-identity domain lets both records be independently
    # shown to bind the same ordered 38-match input identity sequence.
    assert (value.input_match_identity_sha256
            == aliases.input_match_identity_sha256)
    assert len(value.input_match_identity_sha256) == 64
    # A different assignment can never share the binding.
    other = dict(solution.quadrants)
    other[solution.frames[1]] = (other[solution.frames[1]] + 1) % 4
    assert _expected_assignment_digest(
        solution.frames, other) != expected


def test_bindings_digest_replay_refuses_resealed_tampering():
    _readings, _matches, _solution, value = _record()
    record = value.to_record()
    _watch_replay(record)

    # Unsealed tampering is caught by the digest.
    plain = json.loads(json.dumps(record))
    plain["edges"][0][1][1] = int(plain["edges"][0][1][1]) + 1
    with pytest.raises(ValueError):
        graph_mod.validate_solver_input_bindings(plain)

    cases = []

    alternative = _resign(json.loads(json.dumps(record)))
    alternative["edges"][0][1][1] = int(alternative["edges"][0][1][1]) + 1
    cases.append(("alternative table", _resign(alternative)))

    measured = _resign(json.loads(json.dumps(record)))
    measured["edges"][0][1][2] = float(measured["edges"][0][1][2]) + 0.125
    cases.append(("alternative table", _resign(measured)))

    side = _resign(json.loads(json.dumps(record)))
    side["edges"][0][1][5] = -int(side["edges"][0][1][5])
    cases.append(("side", _resign(side)))

    sign = _resign(json.loads(json.dumps(record)))
    sign["edges"][0][1][6] = -int(sign["edges"][0][1][6])
    cases.append(("sign", _resign(sign)))

    active = _resign(json.loads(json.dumps(record)))
    active["edges"][0][4] = not active["edges"][0][4]
    cases.append(("active flag", _resign(active)))

    quadrant = _resign(json.loads(json.dumps(record)))
    quadrant["edges"][0][3] = (int(quadrant["edges"][0][3]) + 1) % 4
    cases.append(("active flag", _resign(quadrant)))

    assignment = _resign(json.loads(json.dumps(record)))
    assignment["assignments"][0] = (assignment["assignments"][0] + 1) % 4
    cases.append(("assignment digest", _resign(assignment)))

    for pattern, tampered in cases:
        with pytest.raises(ValueError, match=pattern):
            graph_mod.validate_solver_input_bindings(tampered)

    missing = json.loads(json.dumps(record))
    missing.pop("record_sha256")
    with pytest.raises(ValueError):
        graph_mod.validate_solver_input_bindings(missing)

    # A referenced reading cannot be dropped, and an unreferenced one cannot be
    # appended: the table is the exact referenced set.
    dropped = _resign(json.loads(json.dumps(record)))
    referenced = dropped["edges"][0][1][0]
    dropped["readings"] = [row for row in dropped["readings"]
                           if row[0] != referenced]
    _resign(dropped)
    with pytest.raises(ValueError):
        graph_mod.validate_solver_input_bindings(dropped)

    extra = _resign(json.loads(json.dumps(record)))
    highest = max(row[0] for row in extra["readings"])
    extra["readings"].append([highest + 1, "v0", 99, 0, -1, [[1.0, 1.0, 1.0]]])
    _resign(extra)
    with pytest.raises(ValueError, match="exact set"):
        graph_mod.validate_solver_input_bindings(extra)


def test_bindings_refuses_malformed_and_partial_inputs():
    readings, matches, solution, _value = _record()
    for broken_readings, broken_matches, broken_solution, field in (
            ("not-readings", matches, solution, "readings"),
            (readings, "not-matches", solution, "matches"),
            (readings, matches, None, "solution")):
        refused = graph_mod.failed_solver_input_bindings(
            broken_readings, broken_matches, broken_solution)
        assert refused.recorded is False
        assert refused.refusal_reason == "invalid_solver_input_bindings_diagnostic"
        assert refused.field == field
        payload = refused.to_record()
        assert set(payload) == {"schema", "status", "reason", "field",
                                "record_sha256"}
        assert "edges" not in payload and "readings" not in payload
        assert refused.failed_assignment_sha256 == ""

    import dataclasses
    broken = [dataclasses.replace(reading, side=0.5)
              if reading.frame == "v1" and reading.axis == 1 else reading
              for reading in readings]
    refused = graph_mod.failed_solver_input_bindings(broken, matches, solution)
    assert refused.recorded is False
    assert refused.refusal_reason == "invalid_solver_input_bindings_diagnostic"
    assert refused.field in ("endpoint_side", "match_endpoint", "readings")

    empty = graph_mod.failed_solver_input_bindings((), matches, solution)
    assert empty.recorded is False
    assert empty.field in ("match_endpoint", "readings")


def test_bindings_refuses_over_budget_record(monkeypatch):
    readings, matches, solution, _value = _record()
    monkeypatch.setattr(graph_mod, "SOLVER_INPUT_BINDINGS_MAX_RECORD_BYTES", 512)
    refused = graph_mod.failed_solver_input_bindings(readings, matches, solution)
    assert refused.recorded is False
    assert refused.refusal_reason == "budget_exhausted"
    assert refused.field == "record_bytes"
    assert "edges" not in refused.to_record()


def test_bindings_artifact_size_estimate_stays_within_budget():
    _readings, _matches, _solution, value = _record()
    encoded = json.dumps(value.to_record(), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    # Measured for this synthetic 38-input / 12-reading / 19-active case: 3,441
    # bytes (the retained R4 16-reading universe is only estimated, well under
    # 16 KiB). These are observed sizes of those cases, not a bound on the
    # largest structurally valid record: the structural caps limit collection
    # work, and a record above the declared 262,144-byte ceiling refuses
    # atomically even while within them.
    assert len(encoded) <= graph_mod.SOLVER_INPUT_BINDINGS_MAX_RECORD_BYTES
    assert len(encoded) < 65_536
    edges = value.to_record()["edges"]
    per_edge = len(encoded) / max(len(edges), 1)
    assert per_edge < 1_024


def test_bindings_default_result_and_digest_parity(monkeypatch):
    from test_image_derived_lattice_measure import _result
    from mc_photogrammetry.pipeline import image_derived_signatures as signatures_mod

    patch = pytest.MonkeyPatch()
    try:
        session, _pose, lattice = _result(patch)
        signatures = signatures_mod.derive_image_derived_signatures(
            session=session, lattice_result=lattice)
    finally:
        patch.undo()
    inputs = {"session": session, "lattice_result": lattice,
              "signatures_result": signatures}

    _readings, matches, solution, _quadrants = _binding_universe()
    monkeypatch.setattr(plane_graph, "match_plane_graph",
                        lambda *_a, **_k: tuple(matches))
    monkeypatch.setattr(plane_graph, "solve_component",
                        lambda *_a, **_k: solution)
    calls = {"bindings": 0}
    original = graph_mod.failed_solver_input_bindings

    def spy(*args, **kwargs):
        calls["bindings"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(graph_mod, "failed_solver_input_bindings", spy)
    control = graph_mod.derive_image_derived_graph(**inputs)
    assert calls["bindings"] == 0  # the default path never allocates the observer

    seen = []
    result, attribution = attribution_mod.derive_image_derived_graph_attribution(
        **inputs, solver_failure_observer=seen.append)
    assert result == control
    assert result.result_sha256 == control.result_sha256
    assert calls["bindings"] == 1
    assert len(seen) == 1
    snapshot = seen[0]
    assert isinstance(snapshot.solver_input_bindings,
                      graph_mod.SolverInputBindings)
    assert attribution.solver_failure is not None
    assert "solver_input_bindings" not in json.dumps(
        attribution.solver_failure.to_record(), sort_keys=True)

    again = graph_mod.derive_image_derived_graph(**inputs)
    assert again == control and again.result_sha256 == control.result_sha256


def test_runner_binds_solver_input_bindings_to_projected_alias_record():
    readings, matches, solution, value = _record()
    aliases = graph_mod._failed_active_edges(
        readings, matches, solution, **_alias_caps())
    assert aliases.recorded is True
    snapshot = graph_mod._solver_failure_snapshot(
        solution, active_edge_aliases=aliases, solver_input_bindings=value)
    projected = attribution_mod.solver_failure_diagnostic(
        snapshot, 0.90, 0.15, 0.02).to_record()
    assert projected["active_edge_aliases"]["failed_assignment_sha256"] == \
        value.failed_assignment_sha256
    raw = attribution_mod.solver_failure_snapshot_record(snapshot)
    assert "solver_input_bindings" not in raw
    assert "active_edge_aliases" not in raw

    inner = "c" * 64
    payload = {"lattice_sha256": "2" * 64, "signatures_sha256": "3" * 64}
    trace = {"solver_failure": projected}
    observation = {
        "schema": stage_subject.OBSERVATION_SCHEMA,
        "frame_count": len(solution.frames),
        "raw": raw,
        "projected": projected,
        "solver_input_bindings": value.to_record(),
        "binding": {"entry_sha256": "e" * 64, "graph_sha256": inner,
                    "lattice_sha256": "2" * 64,
                    "signatures_sha256": "3" * 64,
                    "input_sha256": ["a" * 64]},
    }
    body = dict(observation)
    observation["observation_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8")).hexdigest()
    record = {
        "inputs": [{"sha256": "a" * 64}],
        "entry_sha256": "e" * 64,
        "solver_failure_observation": observation,
    }
    stage_subject._validate_solver_failure_observation(
        record, payload, trace, inner)

    tampered = json.loads(json.dumps(record))
    tampered_bindings = _resign(
        tampered["solver_failure_observation"]["solver_input_bindings"])
    tampered_bindings["failed_assignment_sha256"] = "0" * 64
    _resign(tampered_bindings)
    tampered_body = {key: item for key, item
                     in tampered["solver_failure_observation"].items()
                     if key != "observation_sha256"}
    tampered["solver_failure_observation"]["observation_sha256"] = \
        hashlib.sha256(json.dumps(
            tampered_body, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode("utf-8")).hexdigest()
    with pytest.raises(ValueError, match="assignment"):
        stage_subject._validate_solver_failure_observation(
            tampered, payload, trace, inner)

    identity = json.loads(json.dumps(record))
    tampered_identity = _resign(
        identity["solver_failure_observation"]["solver_input_bindings"])
    tampered_identity["input_match_identity_sha256"] = "1" * 64
    _resign(tampered_identity)
    identity_body = {key: item for key, item
                     in identity["solver_failure_observation"].items()
                     if key != "observation_sha256"}
    identity["solver_failure_observation"]["observation_sha256"] = \
        hashlib.sha256(json.dumps(
            identity_body, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode("utf-8")).hexdigest()
    with pytest.raises(ValueError, match="input identity"):
        stage_subject._validate_solver_failure_observation(
            identity, payload, trace, inner)
