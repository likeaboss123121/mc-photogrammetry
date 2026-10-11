"""DS-IMG-GEOM-PHASE-CAPTURE-R2 focused controls for the phase observer.

These controls exercise the evaluation-only per-face phase capture in
``image_derived_graph`` against the exact integer-lattice component fixture the
accepted alias controls use, so the solver's own returned per-axis metrics are
genuine and can be independently recomposed from the captured faces. They cover
a complete failure payload, every field/index equation (including the serialized
endpoint sides, modulo-1 phase and unit phasor), all three axis counts, the
shared failed-assignment digest binding the phase capture to the active-edge
alias record, digest replay/tampering refusal, gauge invariance, typed refusal
on a metric mismatch and on partial data, and default result/digest parity.

R2 adds an independent fractional-distance component fixture (non-integral
``distance_blocks`` and both endpoint ``side`` signs) so the captured scalar is
checked against the solver's exact face equation
``translation + axis_sign * side * distance``. The integer-only parity fixture
cannot see a missing ``side`` because ``2 * distance`` is an even integer there,
so it is never the sole evidence: the fractional rows differ from the omitted-
side expression and the replay validator refuses an omitted or flipped side.
Each match emits two reversed face rows, so the replay validator additionally
binds the partner endpoint tuple (columns 8-14) of one row to the own endpoint
tuple (columns 1-7) of the other; a re-sealed single-row partner-side tamper is
refused.

No production gate, matcher, solver, threshold, truth, camera, mask or reference
input is read or changed here; no image or reconstruction workload runs.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math

import pytest

from test_image_derived_lattice_measure import _result
from test_plane_graph import (
    LABELS,
    SIGNS,
    TRANSPARENT,
    lattice_component,
    make_match,
    make_reading,
)
from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
from mc_photogrammetry.pipeline import image_derived_graph_attribution as attribution_mod
from mc_photogrammetry.pipeline import image_derived_signatures as signatures_mod
from mc_photogrammetry.reconstruction import lattice_planes
from mc_photogrammetry.reconstruction import plane_graph
from scripts import run_image_derived_graph_stage as stage_subject


@pytest.fixture(scope="module")
def inputs():
    patch = pytest.MonkeyPatch()
    try:
        session, _pose, lattice = _result(patch)
        signatures = signatures_mod.derive_image_derived_signatures(
            session=session, lattice_result=lattice)
        yield {"session": session, "lattice_result": lattice,
               "signatures_result": signatures}
    finally:
        patch.undo()


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


def _phase_fixture():
    """A genuinely abstaining solve of the exact integer-lattice component.

    ``min_resultant`` is raised only in this test call so the already-unambiguous
    best assignment returns ``plane_integer_residual``; the returned per-axis
    resultants/residuals remain the solver's own, so the independent
    recomposition must reconcile with them.
    """
    readings, matches, _eyes = lattice_component(decoy=True)
    solution = plane_graph.solve_component(
        readings, matches,
        quadrant_constraints=(("v0", "v1", 1), ("v0", "v2", 0), ("v1", "v2", 3)),
        integer_tolerance=0.15, min_resultant=1.1, ambiguous_margin=0.02)
    assert solution.reason == "plane_integer_residual"
    return readings, matches, solution


def _complete_capture():
    readings, matches, solution = _phase_fixture()
    value = graph_mod.failed_phase_capture(readings, matches, solution)
    assert value.recorded is True, value.refusal_reason
    return readings, matches, solution, value


def _expected_assignment_digest(frames, quadrants):
    return hashlib.sha256(json.dumps(
        {"domain": graph_mod.FAILED_ASSIGNMENT_DIGEST_DOMAIN,
         "schema": graph_mod.FAILED_ASSIGNMENT_DIGEST_SCHEMA,
         "payload": [[frame, quadrants[frame]] for frame in frames]},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8")).hexdigest()


def _resign_phase(record):
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    record["record_sha256"] = hashlib.sha256(json.dumps(
        {"domain": graph_mod.PHASE_CAPTURE_DIGEST_DOMAIN,
         "schema": record["schema"], "payload": body},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8")).hexdigest()
    return record


def _world_axis_of(row):
    return plane_graph.axis_transform(int(row[3]), int(row[17]))[0]


def _fractional_component():
    """One fractional-distance, mixed-side component with an exactly known room.

    The eyes are quarter-block offsets from the shared planes ``((0, 0), (1, 0),
    (2, 0))``, so every ``distance_blocks`` is non-integral and the integer-only
    parity fixture's even-integer side degeneracy cannot hide a missing ``side``.
    """
    quadrants = {"v0": 0, "v1": 1, "v2": 0}
    eyes = {"v0": (5.75, 2.25, 5.25), "v1": (6.25, 3.75, 7.5),
            "v2": (4.25, 1.5, 9.75)}
    planes = ((0, 0), (1, 0), (2, 0))
    frames = sorted(eyes)
    built = []
    for frame in frames:
        quadrant = quadrants[frame]
        for world_axis, coord in planes:
            label = LABELS[(world_axis, quadrant)]
            world_side = 1.0 if coord > eyes[frame][world_axis] else -1.0
            distance = abs(eyes[frame][world_axis] - coord)
            side = world_side * SIGNS[(label, quadrant)]
            built.append((frame, world_axis, coord,
                          make_reading(frame, label, side, [(1.0, distance, 1.0)])))
    readings = [entry[3] for entry in built]
    index = {}
    for frame in frames:
        ordered = sorted((entry for entry in built if entry[0] == frame),
                         key=lambda entry: lattice_planes.reading_order_key(entry[3]))
        for position, (_frame, world_axis, coord, _reading) in enumerate(ordered):
            index[(frame, world_axis, coord)] = position
    matches = []
    for first_position, first in enumerate(frames):
        for second in frames[first_position + 1:]:
            relative = (quadrants[second] - quadrants[first]) % 4
            for world_axis, coord in planes:
                left_key = (first, world_axis, coord)
                right_key = (second, world_axis, coord)
                if left_key not in index or right_key not in index:
                    continue
                matches.append(make_match(
                    first, second, LABELS[(world_axis, quadrants[first])],
                    LABELS[(world_axis, quadrants[second])], relative,
                    index[left_key], index[right_key]))
    return quadrants, eyes, readings, matches


def _fractional_capture():
    quadrants, _eyes, readings, matches = _fractional_component()
    solution = plane_graph.solve_component(
        readings, matches,
        quadrant_constraints=(("v0", "v1", 1), ("v0", "v2", 0), ("v1", "v2", 3)),
        integer_tolerance=0.15, min_resultant=1.1, ambiguous_margin=0.02)
    assert solution.reason == "plane_integer_residual"
    assert dict(solution.quadrants) == quadrants
    value = graph_mod.failed_phase_capture(readings, matches, solution)
    assert value.recorded is True, value.refusal_reason
    return readings, matches, solution, value


def test_phase_capture_complete_payload_is_bounded_and_replays():
    _readings, _matches, solution, value = _complete_capture()
    record = value.to_record()
    graph_mod.validate_solver_phase_capture(record)
    assert record["schema"] == graph_mod.PHASE_CAPTURE_SCHEMA
    assert record["status"] == "complete"
    assert record["basis"] == graph_mod.PHASE_CAPTURE_BASIS
    assert record["compatibility_decimals"] == \
        graph_mod.PHASE_CAPTURE_COMPATIBILITY_DECIMALS
    assert record["mismatch_reason"] is None
    assert record["frames"] == list(solution.frames)
    assert record["assignments"] == [solution.quadrants[frame]
                                     for frame in solution.frames]
    assert [row[0] for row in record["translations"]] == list(solution.frames)
    # The declared gauge is anchored at the first frame.
    assert record["translations"][0][1] == [0.0, 0.0, 0.0]
    assert len(record["faces"]) == sum(
        solution.alternatives[0].axis_observations)
    assert len(record["faces"]) <= graph_mod.PHASE_CAPTURE_MAX_FACES
    assert len(record["axes"]) == 3
    # Every face row is the serialized 22-column schema, not a private value.
    assert all(len(row) == 22 for row in record["faces"])
    json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)


def test_phase_capture_face_row_equations_bind_every_index():
    _readings, matches, solution, value = _complete_capture()
    record = value.to_record()
    frames = record["frames"]
    translations = {frame: vector for frame, vector in record["translations"]}
    match_by_index = {index: match for index, match in enumerate(matches)}
    for row in record["faces"]:
        (match_index, left_frame, left_plane, left_axis, left_b, left_distance,
         left_side, left_sign, right_frame, right_plane, right_axis, right_b,
         right_distance, right_side, right_sign, quadrant, scalar, own_quadrant,
         other_quadrant, phase, phasor_real, phasor_imag) = row
        assert left_frame in frames and right_frame in frames
        assert match_index in match_by_index
        left_world, expect_left_sign = plane_graph.axis_transform(
            left_axis, own_quadrant)
        right_world, expect_right_sign = plane_graph.axis_transform(
            right_axis, other_quadrant)
        assert left_world == right_world
        assert left_sign == expect_left_sign
        assert right_sign == expect_right_sign
        assert left_side in (-1, 1) and right_side in (-1, 1)
        assert left_distance > 0.0 and right_distance > 0.0
        assert left_b == float(left_b) and right_b == float(right_b)
        match = match_by_index[match_index]
        assert quadrant == match.quadrant_of_right
        if left_frame == match.left:
            assert right_frame == match.right
            assert (other_quadrant - own_quadrant) % 4 == quadrant
        else:
            assert left_frame == match.right and right_frame == match.left
            assert (own_quadrant - other_quadrant) % 4 == quadrant
        assert scalar == pytest.approx(
            translations[left_frame][left_world]
            + left_sign * left_side * left_distance)
        expected_phase = float(scalar % 1.0)
        assert phase == expected_phase
        assert phasor_real == math.cos(2.0 * math.pi * expected_phase)
        assert phasor_imag == math.sin(2.0 * math.pi * expected_phase)
        assert math.hypot(phasor_real, phasor_imag) == pytest.approx(1.0)


def test_phase_capture_axis_rows_bind_solver_metrics_and_counts():
    _readings, _matches, solution, value = _complete_capture()
    record = value.to_record()
    face_counts = {0: 0, 1: 0, 2: 0}
    face_scalars = {0: [], 1: [], 2: []}
    for row in record["faces"]:
        world = _world_axis_of(row)
        face_counts[world] += 1
        face_scalars[world].append(float(row[16]))
    assert set(face_counts) == {0, 1, 2}
    assert all(count > 0 for count in face_counts.values())
    observations = tuple(solution.alternatives[0].axis_observations)
    assert observations == (face_counts[0], face_counts[1], face_counts[2])
    assert len(record["axes"]) == 3
    for row in record["axes"]:
        axis = row[0]
        assert row[1] == pytest.approx(solution.axis_resultants[axis])
        assert row[2] == pytest.approx(solution.axis_residuals[axis])
        resultant, residual = graph_mod._recompose_phase_rows(
            face_scalars[axis])
        assert row[3] == pytest.approx(resultant)
        assert row[4] == pytest.approx(residual)
        # observation_count is the solver's own per-axis face count, not 2x.
        assert row[5] == face_counts[axis] == len(face_scalars[axis])
        assert row[6] == face_counts[axis]


def test_phase_capture_assignment_and_alias_digest_bind_same_identity():
    readings, matches, solution, value = _complete_capture()
    aliases = graph_mod._failed_active_edges(
        readings, matches, solution, **_alias_caps())
    assert aliases.recorded is True
    assert value.source_sha256 == aliases.failed_assignment_sha256
    expected = _expected_assignment_digest(
        solution.frames, solution.quadrants)
    assert value.source_sha256 == expected
    assert graph_mod._failed_assignment_digest(
        solution.frames, solution.quadrants) == expected
    # A different assignment can never share the binding.
    other_quadrants = dict(solution.quadrants)
    other_quadrants[solution.frames[1]] = \
        (other_quadrants[solution.frames[1]] + 1) % 4
    assert _expected_assignment_digest(
        solution.frames, other_quadrants) != expected


def test_phase_capture_digest_replay_refuses_tampering():
    _readings, _matches, _solution, value = _complete_capture()
    record = value.to_record()
    graph_mod.validate_solver_phase_capture(record)

    tampered = json.loads(json.dumps(record))
    tampered["axes"][0][1] = float(tampered["axes"][0][1]) + 0.5
    with pytest.raises(ValueError):
        graph_mod.validate_solver_phase_capture(tampered)

    resigned = _resign_phase(json.loads(json.dumps(record)))
    resigned["axes"][0][1] = float(resigned["axes"][0][1]) + 0.5
    _resign_phase(resigned)
    with pytest.raises(ValueError, match="solver metrics"):
        graph_mod.validate_solver_phase_capture(resigned)

    phase_tamper = _resign_phase(json.loads(json.dumps(record)))
    phase_tamper["faces"][0][19] = (float(phase_tamper["faces"][0][19])
                                    + 0.25) % 1.0
    _resign_phase(phase_tamper)
    with pytest.raises(ValueError, match="phase/phasor"):
        graph_mod.validate_solver_phase_capture(phase_tamper)

    sign_tamper = _resign_phase(json.loads(json.dumps(record)))
    sign_tamper["faces"][0][7] = -int(sign_tamper["faces"][0][7])
    _resign_phase(sign_tamper)
    with pytest.raises(ValueError, match="sign"):
        graph_mod.validate_solver_phase_capture(sign_tamper)

    row_tamper = json.loads(json.dumps(record))
    row_tamper["faces"][0] = list(row_tamper["faces"][0])[:19]
    with pytest.raises(ValueError):
        graph_mod.validate_solver_phase_capture(row_tamper)


def test_phase_capture_fractional_fixture_covers_both_sides_and_non_integral():
    """The independent fixture must genuinely exercise the R2 equation."""
    readings, matches, solution, value = _fractional_capture()
    record = value.to_record()
    graph_mod.validate_solver_phase_capture(record)
    sides = {float(reading.side) for reading in readings}
    distances = {float(alternative.distance_blocks)
                 for reading in readings for alternative in reading.alternatives}
    assert sides == {-1.0, 1.0}
    assert any(distance != int(distance) for distance in distances)
    assert any(row[6] == -1 or row[13] == -1 for row in record["faces"])
    # The integer-only parity fixture uses only integer distances; this fixture
    # proves the capture covers the non-integral case the parity fixture cannot.
    _parity_readings, _parity_matches, _parity_solution, parity = _complete_capture()
    assert all(float(row[5]) == int(float(row[5]))
               and float(row[12]) == int(float(row[12]))
               for row in parity.to_record()["faces"])
    assert matches and solution.reason == "plane_integer_residual"


def test_phase_capture_fractional_scalars_match_solver_face_equation():
    """Each row scalar/phase/phasor equals the solver's exact face equation.

    The equation is reproduced independently from the solver's own
    ``plane_graph.axis_transform`` table and each row's serialized
    side/sign/distance -- never from the captured scalar -- and the fractional
    distances make the omitted-side expression ``translation + sign * distance``
    differ from at least one row, so the integer-only parity fixture is not the
    sole evidence.
    """
    readings, matches, solution, value = _fractional_capture()
    record = value.to_record()
    assert record["mismatch_reason"] is None
    graph_mod.validate_solver_phase_capture(record)
    translations = {frame: vector for frame, vector in record["translations"]}
    match_by_index = {index: match for index, match in enumerate(matches)}
    reading_by = {(reading.frame, reading.axis): reading for reading in readings}
    scalar_by_match: dict[int, list[float]] = {}
    differences = 0
    for row in record["faces"]:
        (match_index, left_frame, left_plane, left_axis, left_b, left_distance,
         left_side, left_sign, right_frame, right_plane, right_axis, right_b,
         right_distance, right_side, right_sign, quadrant, scalar, own_quadrant,
         other_quadrant, phase, phasor_real, phasor_imag) = row
        # Re-derive the endpoint geometry from the source readings, not the row.
        left = reading_by[(left_frame, left_axis)]
        right = reading_by[(right_frame, right_axis)]
        left_world, expect_left_sign = plane_graph.axis_transform(
            left_axis, own_quadrant)
        right_world, expect_right_sign = plane_graph.axis_transform(
            right_axis, other_quadrant)
        assert left_world == right_world
        assert left_axis == left.axis and right_axis == right.axis
        assert left_sign == expect_left_sign and right_sign == expect_right_sign
        assert left_side == int(left.side) and right_side == int(right.side)
        assert left_distance == pytest.approx(left.distance_for(left_b))
        assert right_distance == pytest.approx(right.distance_for(right_b))
        # Exact solver expression: translation + axis_sign * side * distance.
        expect_scalar = (translations[left_frame][left_world]
                         + expect_left_sign * left_side * left_distance)
        assert scalar == expect_scalar
        omitted_side = (translations[left_frame][left_world]
                        + expect_left_sign * left_distance)
        if scalar != omitted_side:
            differences += 1
        match = match_by_index[match_index]
        assert {left_frame, right_frame} == {match.left, match.right}
        assert (other_quadrant - own_quadrant) % 4 == (
            match.quadrant_of_right
            if left_frame == match.left
            else (-match.quadrant_of_right) % 4)
        expected_phase = float(scalar % 1.0)
        assert phase == expected_phase
        assert phasor_real == math.cos(2.0 * math.pi * expected_phase)
        assert phasor_imag == math.sin(2.0 * math.pi * expected_phase)
        assert math.hypot(phasor_real, phasor_imag) == pytest.approx(1.0)
        scalar_by_match.setdefault(match_index, []).append(scalar)
    assert scalar_by_match and all(len(rows) == 2 for rows in scalar_by_match.values())
    # The fixture is side-sensitive: the omitted-side expression would be wrong.
    assert differences > 0
    # The independent recomposition still reconciles with the solver's metrics.
    for axis in range(3):
        scalars = [float(row[16]) for row in record["faces"]
                   if _world_axis_of(row) == axis]
        resultant, residual = graph_mod._recompose_phase_rows(scalars)
        assert record["axes"][axis][1] == pytest.approx(
            solution.axis_resultants[axis])
        assert record["axes"][axis][2] == pytest.approx(
            solution.axis_residuals[axis])
        assert record["axes"][axis][3] == pytest.approx(resultant)
        assert record["axes"][axis][4] == pytest.approx(residual)
        assert record["axes"][axis][5] == len(scalars)
    # And the cross-check is only possible because the capture recorded it.
    assert value.recorded is True


def test_phase_capture_replay_refuses_omitted_or_flipped_own_side():
    """Omission or flip of a row's own side is a typed refusal, not a replay."""
    _readings, _matches, _solution, value = _fractional_capture()
    record = value.to_record()
    graph_mod.validate_solver_phase_capture(record)
    own_row = next(row for row in record["faces"] if row[6] == -1)
    position = record["faces"].index(own_row)

    # Replaying the omitted-side scalar (the R1 production expression) under a
    # re-sealed record contradicts the row's own serialized side equation.
    omitted = json.loads(json.dumps(record))
    omitted["faces"][position][16] = (
        float(own_row[16]) + 2.0 * float(own_row[5]))
    _resign_phase(omitted)
    with pytest.raises(ValueError, match="side equation"):
        graph_mod.validate_solver_phase_capture(omitted)

    # A missing own side can never serialize into a valid record.
    missing = json.loads(json.dumps(record))
    missing["faces"][position][6] = None
    _resign_phase(missing)
    with pytest.raises(ValueError, match="side"):
        graph_mod.validate_solver_phase_capture(missing)

    # A flipped own side without a matching scalar is refused by the side equation.
    flipped = json.loads(json.dumps(record))
    flipped["faces"][position][6] = 1
    _resign_phase(flipped)
    with pytest.raises(ValueError, match="side equation"):
        graph_mod.validate_solver_phase_capture(flipped)


def test_phase_capture_replay_refuses_partner_endpoint_side_tamper():
    """The serialized partner endpoint side is bound, not only the own endpoint."""
    _readings, _matches, _solution, value = _fractional_capture()
    record = value.to_record()
    graph_mod.validate_solver_phase_capture(record)
    own_row = next(row for row in record["faces"] if row[6] == -1)
    position = record["faces"].index(own_row)
    partner_position = next(index for index, row in enumerate(record["faces"])
                            if row[0] == own_row[0] and index != position)
    partner_row = record["faces"][partner_position]
    # The two rows of one match are the reversed endpoint pair.
    assert tuple(own_row[1:8]) == tuple(partner_row[8:15])
    assert tuple(own_row[8:15]) == tuple(partner_row[1:8])

    # Omitting the partner copy of the endpoint side in one row only is refused
    # by the row schema.
    omitted = json.loads(json.dumps(record))
    omitted["faces"][position][13] = None
    _resign_phase(omitted)
    with pytest.raises(ValueError, match="side"):
        graph_mod.validate_solver_phase_capture(omitted)

    # Flipping the partner copy in one row only is refused by the endpoint-pair
    # binding even though the row's own side equation still holds.
    flipped = json.loads(json.dumps(record))
    flipped["faces"][position][13] = -int(flipped["faces"][position][13])
    _resign_phase(flipped)
    with pytest.raises(ValueError, match="partner"):
        graph_mod.validate_solver_phase_capture(flipped)

    # A tamper that also mirrors the partner copy, follows the face equation and
    # rewrites the recorded recomposition still cannot reconcile with the
    # solver's own returned resultants/residuals.
    consistent = json.loads(json.dumps(record))
    consistent_row = consistent["faces"][position]
    consistent_partner = consistent["faces"][partner_position]
    axis = _world_axis_of(consistent_row)
    consistent_row[6] = 1
    consistent_row[16] = (float(own_row[16])
                          + 2.0 * float(own_row[7]) * float(own_row[5]))
    consistent_row[19] = float(consistent_row[16] % 1.0)
    consistent_row[20] = math.cos(2.0 * math.pi * consistent_row[19])
    consistent_row[21] = math.sin(2.0 * math.pi * consistent_row[19])
    consistent_partner[13] = 1
    scalars = [float(row[16]) for row in consistent["faces"]
               if _world_axis_of(row) == axis]
    resultant, residual = graph_mod._recompose_phase_rows(scalars)
    consistent["axes"][axis][3] = resultant
    consistent["axes"][axis][4] = residual
    _resign_phase(consistent)
    with pytest.raises(ValueError, match="solver metrics"):
        graph_mod.validate_solver_phase_capture(consistent)


def test_phase_capture_refuses_malformed_endpoint_side():
    """A non-+/-1 or boolean side is a typed malformed refusal at capture."""
    readings, matches, solution, _value = _fractional_capture()
    for bad_side in (0.0, 2.0, 0.5, True, None):
        broken = [dataclasses.replace(reading, side=bad_side)
                  if reading.frame == "v1" and reading.axis == 0 else reading
                  for reading in readings]
        refused = graph_mod.failed_phase_capture(broken, matches, solution)
        assert refused.recorded is False
        assert refused.refusal_reason in (
            "invalid_phase_capture_diagnostic", "phase_recomposition_mismatch")
        assert "faces" not in refused.to_record()


def test_phase_capture_recomposition_is_gauge_invariant():
    scalars = [0.25, 1.75, -3.5, 8.125]
    baseline = graph_mod._recompose_phase_rows(scalars)
    for offset in (0.37, -12.5, 1000.0):
        shifted = graph_mod._recompose_phase_rows(
            [scalar + offset for scalar in scalars])
        assert shifted[0] == pytest.approx(baseline[0], abs=1e-12)
        assert shifted[1] == pytest.approx(baseline[1], abs=1e-12)

    readings, matches, solution, value = _complete_capture()
    shifted_solution = dataclasses.replace(
        solution, translations={frame: (13.0, -7.0, 3.0)
                                for frame in solution.frames})
    shifted = graph_mod.failed_phase_capture(readings, matches, shifted_solution)
    assert shifted.recorded is True
    assert shifted.to_record() == value.to_record()
    assert shifted.record_sha256 == value.record_sha256


def test_phase_capture_typed_refusal_on_metric_mismatch():
    readings, matches, solution = _phase_fixture()
    mismatched = dataclasses.replace(
        solution, axis_resultants={axis: value + 0.1
                                   for axis, value in solution.axis_resultants.items()})
    refused = graph_mod.failed_phase_capture(readings, matches, mismatched)
    assert refused.recorded is False
    assert refused.refusal_reason == "phase_recomposition_mismatch"
    assert refused.field in ("axis_recomposition", "axis_observation_count",
                             "axis_metric_coverage")
    payload = refused.to_record()
    assert set(payload) == {"schema", "status", "reason", "field",
                            "record_sha256"}
    assert "faces" not in payload and "axes" not in payload
    assert refused.source_sha256 == ""


def test_phase_capture_typed_refusal_on_partial_data():
    readings, matches, solution = _phase_fixture()
    empty = graph_mod.failed_phase_capture((), matches, solution)
    assert empty.recorded is False
    assert empty.refusal_reason == "invalid_phase_capture_diagnostic"
    assert empty.field in ("readings_b", "matches")
    assert "faces" not in empty.to_record()

    malformed = graph_mod.failed_phase_capture("not-readings", matches, solution)
    assert malformed.recorded is False
    assert malformed.refusal_reason == "invalid_phase_capture_diagnostic"
    assert malformed.field == "readings"
    assert malformed.source_sha256 == ""
    assert readings  # the supplied input is never mutated by a refusal


def test_phase_capture_default_result_and_digest_parity(monkeypatch, inputs):
    _readings, matches, solution = _phase_fixture()
    monkeypatch.setattr(plane_graph, "match_plane_graph",
                        lambda *_a, **_k: tuple(matches))
    monkeypatch.setattr(plane_graph, "solve_component",
                        lambda *_a, **_k: solution)
    calls = {"phase": 0}
    original = graph_mod.failed_phase_capture

    def spy(*args, **kwargs):
        calls["phase"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(graph_mod, "failed_phase_capture", spy)
    control = graph_mod.derive_image_derived_graph(**inputs)
    assert calls["phase"] == 0  # the default path never allocates the observer

    seen = []
    result, attribution = attribution_mod.derive_image_derived_graph_attribution(
        **inputs, solver_failure_observer=seen.append)
    assert result == control
    assert result.result_sha256 == control.result_sha256
    assert len(seen) == 1
    snapshot = seen[0]
    assert isinstance(snapshot.phase_capture, graph_mod.SolverPhaseCapture)
    assert attribution.solver_failure is not None
    assert "phase_capture" not in json.dumps(
        attribution.solver_failure.to_record(), sort_keys=True)

    again = graph_mod.derive_image_derived_graph(**inputs)
    assert again == control and again.result_sha256 == control.result_sha256


def test_runner_binds_phase_capture_to_projected_alias_record():
    readings, matches, solution, value = _complete_capture()
    aliases = graph_mod._failed_active_edges(
        readings, matches, solution, **_alias_caps())
    snapshot = graph_mod._solver_failure_snapshot(
        solution, active_edge_aliases=aliases, phase_capture=value)
    projected = attribution_mod.solver_failure_diagnostic(
        snapshot, 1.1, 0.15, 0.02).to_record()
    assert projected["active_edge_aliases"]["failed_assignment_sha256"] == \
        value.source_sha256
    raw = attribution_mod.solver_failure_snapshot_record(snapshot)
    assert "active_edge_aliases" not in raw

    inner = "c" * 64
    payload = {"lattice_sha256": "2" * 64, "signatures_sha256": "3" * 64}
    trace = {"solver_failure": projected}
    record = {
        "inputs": [{"sha256": "a" * 64}],
        "entry_sha256": "e" * 64,
        "solver_failure_observation": {
            "schema": stage_subject.OBSERVATION_SCHEMA,
            "frame_count": len(solution.frames),
            "raw": raw,
            "projected": projected,
            "phase_capture": value.to_record(),
            "binding": {"entry_sha256": "e" * 64, "graph_sha256": inner,
                        "lattice_sha256": "2" * 64,
                        "signatures_sha256": "3" * 64,
                        "input_sha256": ["a" * 64]},
        },
    }
    observation = record["solver_failure_observation"]
    body = dict(observation)
    observation["observation_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8")).hexdigest()
    stage_subject._validate_solver_failure_observation(
        record, payload, trace, inner)

    tampered = json.loads(json.dumps(record))
    tampered_phase = _resign_phase(tampered["solver_failure_observation"]
                                   ["phase_capture"])
    tampered_phase["source_sha256"] = "0" * 64
    _resign_phase(tampered_phase)
    tampered_body = {key: item for key, item
                     in tampered["solver_failure_observation"].items()
                     if key != "observation_sha256"}
    tampered["solver_failure_observation"]["observation_sha256"] = \
        hashlib.sha256(json.dumps(
            tampered_body, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode("utf-8")).hexdigest()
    with pytest.raises(ValueError, match="phase capture source mismatch"):
        stage_subject._validate_solver_failure_observation(
            tampered, payload, trace, inner)
