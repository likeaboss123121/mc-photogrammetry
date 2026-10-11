"""Synthetic physical endpoint contract, using real selection and output guards."""
from __future__ import annotations

from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
from mc_photogrammetry.pipeline import image_derived_joint_graph as joint_graph
from mc_photogrammetry.pipeline import image_derived_signatures as signatures_mod
from mc_photogrammetry.reconstruction import endpoint_hypothesis as csp
from mc_photogrammetry.reconstruction import lattice_planes, plane_graph

import dataclasses
import json

import numpy as np
import pytest

FRAME_LEFT = "a"
FRAME_RIGHT = "b"
SESSION_FRAMES = (FRAME_LEFT, FRAME_RIGHT)
QUADRANT_CONSTRAINTS = ((FRAME_LEFT, FRAME_RIGHT, 0),)
# Production axis transform at q0 is the identity, so local axis == world axis.
WORLD_X, WORLD_Y, WORLD_Z = 0, 1, 2

_ENDPOINT_COUNT = 7
# Distinct content-only period labels keep each frame's canonical index stable.
_PERIODS = (60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0)
_SUPPORTED_LEFT_X = 10.0
_CONTRADICTORY_LEFT_X = 9.0
_ALIAS_LEFT_X = 8.0
_RIGHT_X = 11.0


def _reading(frame, axis, distance, period, alternatives=None):
    """One synthetic plane reading; default is a single retained alternative."""
    options = ((1.0, float(distance), 1.0),) if alternatives is None else tuple(
        (float(b), float(value), 1.0) for b, value in alternatives)
    return lattice_planes.PlaneReading(
        frame=frame, axis=int(axis), side=1.0,
        alternatives=tuple(lattice_planes.DistanceAlternative(b, value, support)
                           for b, value, support in options),
        period_pixels=float(period), phase_pixels=(0.0, 0.0), extent=1.0, samples=512,
        polygon=None, coverage_cells=8, structural_share=0.6, confidence=0.9,
        rectification=lattice_planes.PlaneRectification(
            axis_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            focal=500.0, principal_point=(120.0, 90.0), centre=(0.0, 0.0)))


def _match(left_plane, right_plane, left_axis, right_axis, score=1.0, left_b=1.0, right_b=1.0):
    return plane_graph.PlaneMatch(
        left=FRAME_LEFT, right=FRAME_RIGHT, left_axis=left_axis, right_axis=right_axis,
        left_b=float(left_b), right_b=float(right_b), quadrant_of_right=0, psr=5.0, peak=1.0,
        second_peak=0.1, overlap_cells=64, integer_residual=None, score=float(score),
        left_plane=left_plane, right_plane=right_plane)


def _evidence(readings):
    """Deterministic synthetic signature evidence, one per retained alternative.

    ``readings`` is the complete global universe in canonical input order, which
    is exactly what :func:`selection.reading_bindings` validates and hashes.
    """
    evidence = []
    for index, reading in enumerate(readings):
        for alternative_index, alternative in enumerate(reading.alternatives):
            seed = 100 + reading.axis + 10_000 * alternative_index
            cells = np.random.default_rng(seed).normal(size=(11, 9)).astype(np.float32)
            signature = lattice_planes.PlaneSignature(
                reading, alternative.blocks_per_period,
                signatures_mod._readonly_array(cells),
                signatures_mod._readonly_array(np.zeros_like(cells)),
                signatures_mod._readonly_array(np.ones(cells.shape, dtype=bool)), (0.0, 0.0))
            evidence.append(signatures_mod.SignatureEvidence(
                reading.frame, index, reading.axis, alternative_index,
                alternative.blocks_per_period, signature))
    return tuple(evidence)


def _select(readings, rows):
    return selection.select_endpoint_hypotheses(
        readings=tuple(readings), signatures=_evidence(readings), candidate_matches=tuple(rows),
        frame_ids=SESSION_FRAMES, quadrant_constraints=QUADRANT_CONSTRAINTS,
        budget=csp.EndpointBudget(), session_sha256="a" * 64, lattice_sha256="b" * 64,
        signatures_sha256="c" * 64)


def _canonical_endpoints(readings):
    """Map each ``(frame, axis, distinct period)`` label to its canonical position.

    The content-only canonical key orders by axis, then side, period, phase,
    extent, samples, polygon, structural share, confidence and alternatives, so
    distinct period labels give a deterministic per-frame plane index.
    """
    index, _, frames = plane_graph._canonical_reading_index(readings)
    assert frames == SESSION_FRAMES
    endpoints = {}
    for (frame, position), reading in index.items():
        endpoints[(frame, reading.axis, float(reading.period_pixels))] = position
    assert len(endpoints) == len(readings)
    return endpoints


def _readings_for(second_left_x_distance):
    """Two frames, seven endpoints, one alternative per reading."""
    return (
        _reading(FRAME_LEFT, WORLD_X, _SUPPORTED_LEFT_X, _PERIODS[0]),
        _reading(FRAME_LEFT, WORLD_X, second_left_x_distance, _PERIODS[1]),
        _reading(FRAME_LEFT, WORLD_Y, 20.0, _PERIODS[2]),
        _reading(FRAME_LEFT, WORLD_Z, 20.0, _PERIODS[3]),
        _reading(FRAME_RIGHT, WORLD_X, _RIGHT_X, _PERIODS[4]),
        _reading(FRAME_RIGHT, WORLD_Y, 30.0, _PERIODS[5]),
        _reading(FRAME_RIGHT, WORLD_Z, 30.0, _PERIODS[6]),
    )


def _seven_endpoint_fixture(*, second_left_x_distance):
    """Four same-axis rows over the seven mandatory endpoints.

    The supported control makes both left-X candidates imply the same
    displacement; the contradictory control changes the second left X distance so
    the two left-X candidates share the right-X alternative yet imply 1 versus 2.
    """
    readings = _readings_for(second_left_x_distance)
    endpoints = _canonical_endpoints(readings)
    rows = (
        _match(endpoints[(FRAME_LEFT, WORLD_X, _PERIODS[0])],
               endpoints[(FRAME_RIGHT, WORLD_X, _PERIODS[4])], WORLD_X, WORLD_X, score=0.4),
        _match(endpoints[(FRAME_LEFT, WORLD_X, _PERIODS[1])],
               endpoints[(FRAME_RIGHT, WORLD_X, _PERIODS[4])], WORLD_X, WORLD_X, score=0.3),
        _match(endpoints[(FRAME_LEFT, WORLD_Y, _PERIODS[2])],
               endpoints[(FRAME_RIGHT, WORLD_Y, _PERIODS[5])], WORLD_Y, WORLD_Y),
        _match(endpoints[(FRAME_LEFT, WORLD_Z, _PERIODS[3])],
               endpoints[(FRAME_RIGHT, WORLD_Z, _PERIODS[6])], WORLD_Z, WORLD_Z),
    )
    return readings, rows


def _forced_adversarial_fixture(conflict=True, free_count=4, alternatives=8):
    readings = _readings_for(9.0 if conflict else 10.0)
    readings += tuple(_reading("a", 0, 10.0, 100.0 + i,
        alternatives=tuple((1.0, 10.0 + a) for a in range(alternatives)))
        for i in range(free_count))
    endpoints = _canonical_endpoints(readings)
    labels = ((0, _PERIODS[0], _PERIODS[4]), (0, _PERIODS[1], _PERIODS[4]),
              (1, _PERIODS[2], _PERIODS[5]), (2, _PERIODS[3], _PERIODS[6]))
    rows = tuple(_match(endpoints[("a", axis, lp)], endpoints[("b", axis, rp)], axis, axis)
                 for axis, lp, rp in labels)
    rows += tuple(_match(endpoints[("a", 0, 100.0 + i)], endpoints[("b", 0, _PERIODS[4])], 0, 0)
                  for i in range(free_count))
    return readings, rows


class _UnprunedControl:
    """Pinned v2 search control: no preflight or forced-factor search work.

    Original endpoint implementation SHA256:
    E6CB51482DDABC1A70848105E6254C0F8E975364BAEC52CEE12390D6E1034C79.
    Registry, propagation, DFS, full physical checks and their charges stay live;
    expected charged visits below pin this control against silent changes.
    """
    def __init__(self, work, *args):
        self.journal = []
        self.positions = {key: i for i, key in enumerate(work.endpoint_keys)}

    def singleton(self, *args): pass
    def check(self, *args): return True
    def undo(self, *args): pass


def _solve_fixture(readings, rows, **kwargs):
    return csp.solve_endpoint_hypotheses(readings, rows, session_frames=SESSION_FRAMES,
        quadrant_constraints=QUADRANT_CONSTRAINTS, **kwargs)


def test_forced_adversarial_net_charged_savings_and_supported_unique(monkeypatch):
    readings, rows = _forced_adversarial_fixture()
    pruned = _solve_fixture(readings, rows)
    with monkeypatch.context() as control:
        control.setattr(csp, "_ForcedDiameter", _UnprunedControl)
        control.setattr(csp, "_preflight_relations", lambda *args: None)
        unpruned = _solve_fixture(readings, rows)
    assert pruned.status == unpruned.status == "no"
    assert pruned.exhaustive and unpruned.exhaustive
    assert pruned.counters["partial_prunes"] == 1
    assert pruned.counters["physical_assignments"] == 0
    assert unpruned.counters["physical_assignments"] == 4096
    assert 0 < pruned.counters["preflight_visits"] < pruned.counters["relation_visits"]
    assert pruned.counters["partial_undo_visits"] > 0
    assert pruned.counters["relation_visits"] < unpruned.counters["relation_visits"]
    assert pruned.counters["relation_visits"] == 451
    assert unpruned.counters["relation_visits"] == 205492
    readings, rows = _forced_adversarial_fixture(False)
    supported = _solve_fixture(readings, rows)
    with monkeypatch.context() as control:
        control.setattr(csp, "_ForcedDiameter", _UnprunedControl)
        control.setattr(csp, "_preflight_relations", lambda *args: None)
        expected = _solve_fixture(readings, rows)
    assert supported.status == expected.status == "unique"
    assert supported.choices == expected.choices
    assert supported.exhaustive and supported.counters["partial_prunes"] > 0
    assert supported == _solve_fixture(readings, rows)
    _closed_scalar_accounting(pruned)
    _closed_scalar_accounting(supported)


def test_supported_seven_endpoint_assignment_is_unique_with_complete_mandatory_support():
    """Control 1: compatible X displacement and consistent Y/Z stay unique."""
    readings, rows = _seven_endpoint_fixture(second_left_x_distance=_SUPPORTED_LEFT_X)
    assert len(readings) == _ENDPOINT_COUNT and len(rows) == 4
    cert = _select(readings, rows)
    assert cert.status == "unique", (cert.status, cert.reason)
    assert cert.reason == "joint_geometry_not_evaluated"
    proof = cert.proof_record()
    assert proof["exhaustive"] and not proof["evidence_ambiguous"]
    assert proof["counters"]["solutions_found"] == 1

    # Every original endpoint remains mandatory and covered by real source rows.
    factors = proof["choices"][0]["factors"]
    covered = {tuple(pair) for factor in factors for pair in factor["endpoints"]}
    expected = {(frame, position) for (frame, _axis, _period), position
                in _canonical_endpoints(readings).items()}
    assert covered == expected and len(covered) == _ENDPOINT_COUNT
    assert all(factor["source_rows"] for factor in factors)
    # One source row per mandatory factor, so no dropped row was reinstated.
    assert sorted(row["source_rows"][0] for row in factors) == [0, 1, 2, 3]
    # Both left-X candidates imply one displacement within the preserved tolerance.
    index = plane_graph._canonical_reading_index(readings)[0]
    deltas = [plane_graph._delta(row, index) for row in rows if row.left_axis == WORLD_X]
    assert len(deltas) == 2
    assert abs(deltas[0] - deltas[1]) <= plane_graph.SAME_AXIS_TOLERANCE

    selection.validate_candidate_certificate(cert, readings, _evidence(readings), rows)


def test_contradictory_left_x_displacement_is_refused_before_unique_and_geometry():
    """Control 2: displacement 1 versus 2 must not be counted unique.

    Exact failing regression assertion for the current implementation gap: the
    endpoint CSP has no physical-displacement relation, so it returns ``unique``
    for a complete assignment whose two left-X rows disagree by 1 block, far
    beyond the preserved 0.15 tolerance. The selector must refuse before any
    geometry evaluation; a coverage-only rescue must never make this unique.
    """
    readings, rows = _seven_endpoint_fixture(second_left_x_distance=_CONTRADICTORY_LEFT_X)
    index = plane_graph._canonical_reading_index(readings)[0]
    deltas = [plane_graph._delta(row, index) for row in rows if row.left_axis == WORLD_X]
    assert len(deltas) == 2
    # Both X rows share the same right-X plane alternative, so endpoint
    # consistency alone cannot reject the contradiction.
    assert rows[0].right_plane == rows[1].right_plane
    assert rows[0].left_plane != rows[1].left_plane
    assert abs(deltas[0] - deltas[1]) > plane_graph.SAME_AXIS_TOLERANCE
    # Signed same-axis displacements in the left frame's gauge: 10 - 11 = -1 and
    # 9 - 11 = -2, so the two left-X candidates disagree by exactly one block.
    assert sorted(deltas) == [-2.0, -1.0], deltas
    assert abs(abs(deltas[0] - deltas[1]) - 1.0) <= 1e-9

    cert = _select(readings, rows)
    # Required contract: fail closed BEFORE geometry is accepted. This is the
    # exact failing assertion for the current gap (the selector returns unique).
    assert cert.status == "no", (
        "physically contradictory complete assignment counted as {!r}; the endpoint "
        "CSP enforces only structural endpoint/axis connectivity and carries no "
        "same-axis displacement relation".format(cert.status))
    assert cert.representatives == () and cert.proof_record()["choices"] == []


def test_integer_alias_displacement_is_refused_by_the_production_geometry_gate():
    """Refuse the selector, then independently guard a genuine kernel result."""
    readings, rows = _seven_endpoint_fixture(second_left_x_distance=_ALIAS_LEFT_X)
    indexed, _, frames = plane_graph._canonical_reading_index(readings)
    assert sorted(plane_graph._delta(row, indexed) for row in rows[:2]) == [-3.0, -1.0]
    cert = _select(readings, rows)
    assert cert.status == "no" and cert.representatives == ()
    assert cert.proof_record()["counters"]["partial_prunes"] == 1
    assert cert.proof_record()["counters"]["physical_rejected"] == 0
    q, selected = {"a": 0, "b": 0}, dict.fromkeys(indexed, 0)
    actual = plane_graph._evaluate_assignment(frames, q, rows, indexed)
    assert actual is not None
    _, _, detail = actual
    assert all(value == pytest.approx(1.0) for value in detail["resultants"].values())
    assert all(value == pytest.approx(0.0) for value in detail["residuals"].values())
    assert plane_graph.physical_displacement_refusal(
        indexed, rows, q, selected, detail["translations"]) == "plane_integer_residual"


def test_equal_score_incompatible_alternatives_remain_multiple_without_score_winner():
    """Control 3: no score-based winner; two complete mutually exclusive answers.

    A left-X and a right-X endpoint each retain two alternatives; two equal-score
    rows over the *same* endpoint pair select the incompatible pairings
    (signed displacement -1 versus -2). Production
    eligibility still requires all three axes, so a synthetic Y and Z edge is
    present. The real selector must keep both complete, mutually exclusive
    assignments (multiple/ambiguity) and must not pick either row by score.
    """
    left_alternatives = ((1.0, 10.0), (2.0, 9.0))
    right_alternatives = ((1.0, 11.0), (2.0, 11.0))
    readings = (
        _reading(FRAME_LEFT, WORLD_X, 10.0, _PERIODS[0], left_alternatives),
        _reading(FRAME_RIGHT, WORLD_X, 11.0, _PERIODS[1], right_alternatives),
        _reading(FRAME_LEFT, WORLD_Y, 20.0, _PERIODS[2]),
        _reading(FRAME_RIGHT, WORLD_Y, 30.0, _PERIODS[3]),
        _reading(FRAME_LEFT, WORLD_Z, 20.0, _PERIODS[4]),
        _reading(FRAME_RIGHT, WORLD_Z, 30.0, _PERIODS[5]),
    )
    endpoints = _canonical_endpoints(readings)
    x_pair = (endpoints[(FRAME_LEFT, WORLD_X, _PERIODS[0])],
              endpoints[(FRAME_RIGHT, WORLD_X, _PERIODS[1])])
    rows = (
        _match(*x_pair, WORLD_X, WORLD_X, score=1.0, left_b=1.0, right_b=1.0),
        _match(*x_pair, WORLD_X, WORLD_X, score=1.0, left_b=2.0, right_b=2.0),
        _match(endpoints[(FRAME_LEFT, WORLD_Y, _PERIODS[2])],
               endpoints[(FRAME_RIGHT, WORLD_Y, _PERIODS[3])], WORLD_Y, WORLD_Y, score=1.0),
        _match(endpoints[(FRAME_LEFT, WORLD_Z, _PERIODS[4])],
               endpoints[(FRAME_RIGHT, WORLD_Z, _PERIODS[5])], WORLD_Z, WORLD_Z, score=1.0),
    )
    assert rows[0].score == rows[1].score == 1.0
    # The two rows answer the same endpoint pair with incompatible alternatives.
    assert rows[0].left_plane == rows[1].left_plane
    assert rows[0].right_plane == rows[1].right_plane
    assert (rows[0].left_b, rows[0].right_b) != (rows[1].left_b, rows[1].right_b)
    # The 1/1 and 2/2 pairings imply signed displacements 10 - 11 = -1 and
    # 9 - 11 = -2, so the two complete answers differ by exactly one block.
    index = plane_graph._canonical_reading_index(readings)[0]
    row_deltas = [plane_graph._delta(row, index) for row in rows[:2]]
    assert row_deltas == [-1.0, -2.0], row_deltas

    cert = _select(readings, rows)
    assert cert.status == "multiple", (cert.status, cert.reason)
    assert cert.reason == "multiple_endpoint_hypotheses"
    assert cert.representatives == ()
    proof = cert.proof_record()
    # Two distinct eligible choices stop the search, so "multiple" is not
    # exhaustive by the existing contract; both choices are retained instead.
    assert not proof["exhaustive"] and proof["counters"]["solutions_found"] == 2
    assert len(proof["choices"]) == 2
    universe = {(frame, position) for (frame, _axis, _period), position
                in endpoints.items()}
    answers = []
    for choice in proof["choices"]:
        selected = {(frame, position): alternative
                    for frame, position, alternative in choice["alternatives"]}
        assert set(selected) == universe
        assert len(choice["factors"]) == 3  # one X, one Y, one Z mandatory factor
        answers.append(selected)
    # The two complete answers are mutually exclusive: they disagree on both X
    # alternatives, so neither can be jointly true, and both stay counted.
    assert answers[0] != answers[1]
    x_endpoint = (FRAME_LEFT, endpoints[(FRAME_LEFT, WORLD_X, _PERIODS[0])])
    right_endpoint = (FRAME_RIGHT, endpoints[(FRAME_RIGHT, WORLD_X, _PERIODS[1])])
    assert {answer[x_endpoint] for answer in answers} == {0, 1}
    # Selected alternative indices pick the paired distances, proving the two
    # retained answers carry signed displacements -1 and -2 respectively.
    left_value = {slot: value for slot, (_b, value) in enumerate(left_alternatives)}
    right_value = {slot: value for slot, (_b, value) in enumerate(right_alternatives)}
    selected_deltas = sorted(
        left_value[answer[x_endpoint]] - right_value[answer[right_endpoint]]
        for answer in answers)
    assert selected_deltas == [-2.0, -1.0], selected_deltas
    # Equal scores are not a tie-break, so no score-based winner may be chosen.


def test_budget_tolerances_and_provenance_are_preserved_and_fail_closed():
    """The contract uses the exact production budget and deterministic digests."""
    readings, rows = _seven_endpoint_fixture(second_left_x_distance=_SUPPORTED_LEFT_X)
    first = _select(readings, rows)
    second = _select(readings, rows)
    assert first.status == second.status == "unique"
    assert first.result_sha256 == second.result_sha256
    assert first.session_sha256 == "a" * 64 and first.lattice_sha256 == "b" * 64
    assert first.candidate_count == len(rows) == 4
    # Preserved production ceilings and tolerances, not relaxed here.
    budget = csp.EndpointBudget()
    assert budget.solution_rows == 2
    assert budget.alternatives_per_endpoint == 16
    assert budget.input_match_rows == 16_384
    assert plane_graph.SAME_AXIS_TOLERANCE == 0.15
    assert plane_graph.MAX_ASSIGNMENTS == 16_384


def test_supported_assignment_passes_unchanged_geometry_once(monkeypatch):
    readings, rows = _seven_endpoint_fixture(second_left_x_distance=10.0)
    cert = _select(readings, rows)
    actual, count = plane_graph._evaluate_assignment, []
    def observed(*args):
        count.append(1)
        return actual(*args)
    monkeypatch.setattr(plane_graph, "_evaluate_assignment", observed)
    result = joint_graph._evaluate_selected(readings, tuple(r.match for r in cert.representatives), cert)
    assert result.reason == "" and result.scale_known and len(count) == 1
    assert set(result.axis_resultants) == {0, 1, 2}
    assert all(v >= 0.90 for v in result.axis_resultants.values())
    assert all(v <= 0.15 for v in result.axis_residuals.values())


def test_distinguishable_support_equal_displacement_remains_unresolved():
    readings, rows = _seven_endpoint_fixture(second_left_x_distance=10.0)
    rows = rows + (dataclasses.replace(rows[0], cell_shift_rc=(1, 0)),)
    cert = _select(readings, rows)
    assert cert.status == "unresolved_evidence" and cert.representatives == ()
    assert cert.proof_record()["counters"]["physical_feasible"] == 1


def _reverse(row):
    return dataclasses.replace(row, left=row.right, right=row.left,
        left_plane=row.right_plane, right_plane=row.left_plane,
        left_axis=row.right_axis, right_axis=row.left_axis,
        left_b=row.right_b, right_b=row.left_b, quadrant_of_right=(-row.quadrant_of_right) % 4)


def test_reversed_rows_preserve_canonical_signed_displacement():
    readings, rows = _seven_endpoint_fixture(second_left_x_distance=10.0)
    first, reverse = _select(readings, rows), _select(readings, tuple(_reverse(row) for row in rows))
    assert first.status == reverse.status == "unique"
    groups = first.proof_record()["choices"][0]["physical_groups"]
    reversed_groups = reverse.proof_record()["choices"][0]["physical_groups"]
    assert [g[:3] for g in groups] == [g[:3] for g in reversed_groups]
    assert [g[3][1:4] for g in groups] == [g[3][1:4] for g in reversed_groups]


def test_different_local_axes_are_grouped_only_after_world_mapping():
    readings, rows = _seven_endpoint_fixture(second_left_x_distance=10.0)
    transformed = []
    for reading in readings:
        if reading.frame == "b":
            axis = {0: 2, 1: 1, 2: 0}[reading.axis]
            world, sign = plane_graph.axis_transform(axis, 1)
            assert world == reading.axis
            reading = dataclasses.replace(reading, axis=axis, side=float(sign))
        transformed.append(reading)
    transformed = tuple(transformed)
    indexed, identities, _ = plane_graph._canonical_reading_index(transformed)
    old = plane_graph._canonical_reading_index(readings)[0]
    by_period = {(r.frame, r.period_pixels): identities[id(r)][1] for r in transformed}
    mapped = tuple(dataclasses.replace(row,
        left_plane=by_period[(row.left, old[(row.left,row.left_plane)].period_pixels)],
        right_plane=by_period[(row.right, old[(row.right,row.right_plane)].period_pixels)],
        right_axis={0: 2, 1: 1, 2: 0}[row.right_axis], quadrant_of_right=1) for row in rows)
    cert = selection.select_endpoint_hypotheses(readings=transformed, signatures=_evidence(transformed),
        candidate_matches=mapped, frame_ids=SESSION_FRAMES, quadrant_constraints=(("a","b",1),),
        budget=csp.EndpointBudget(), session_sha256="a"*64,lattice_sha256="b"*64,signatures_sha256="c"*64)
    assert cert.status == "unique"
    assert [g[0][2] for g in cert.proof_record()["choices"][0]["physical_groups"]] == [0, 1, 2]
    component = joint_graph._evaluate_selected(transformed, tuple(r.match for r in cert.representatives), cert)
    assert component.reason == ""
    reversed_rows = tuple(_reverse(row) for row in mapped)
    reversed_cert = selection.select_endpoint_hypotheses(readings=transformed, signatures=_evidence(transformed),
        candidate_matches=reversed_rows, frame_ids=SESSION_FRAMES, quadrant_constraints=(("a","b",1),),
        budget=csp.EndpointBudget(), session_sha256="a"*64,lattice_sha256="b"*64,signatures_sha256="c"*64)
    assert reversed_cert.status == "unique"
    assert [g[:3] for g in reversed_cert.proof_record()["choices"][0]["physical_groups"]] == [
        g[:3] for g in cert.proof_record()["choices"][0]["physical_groups"]]
    selection.validate_candidate_certificate(reversed_cert, transformed, _evidence(transformed), reversed_rows)


@pytest.mark.parametrize("damage", ["missing_index", "bad_index", "unknown_side", "nonfinite", "overflow"])
def test_unknown_or_nonfinite_displacement_refuses_real_output_guard(damage):
    readings, rows = _seven_endpoint_fixture(second_left_x_distance=10.0)
    indexed, _, frames = plane_graph._canonical_reading_index(readings)
    q, selected = {"a":0,"b":0}, dict.fromkeys(indexed, 0)
    _, _, detail = plane_graph._evaluate_assignment(frames,q,rows,indexed)
    if damage == "missing_index": selected.pop(("a", 0))
    elif damage == "bad_index": selected[("a", 0)] = 99
    else:
        reading = indexed[("a", 0)]
        if damage == "unknown_side": reading = dataclasses.replace(reading,side=0.0)
        elif damage == "nonfinite":
            reading = dataclasses.replace(reading,alternatives=(dataclasses.replace(reading.alternatives[0],distance_blocks=float("nan")),))
        else:
            reading = dataclasses.replace(reading,alternatives=(dataclasses.replace(reading.alternatives[0],distance_blocks=1e308),))
            right = indexed[("b",0)]
            indexed[("b",0)] = dataclasses.replace(right,side=-1.0,
                alternatives=(dataclasses.replace(right.alternatives[0],distance_blocks=1e308),))
        indexed[("a",0)] = reading
    assert plane_graph.physical_displacement_refusal(indexed,rows,q,selected,detail["translations"]) == "unverified_physical_displacement"


@pytest.mark.parametrize("damage", ["side", "nonfinite", "foreign_row", "unresolvable_b"])
def test_unverified_full_universe_fails_closed_globally(damage):
    readings, rows = _seven_endpoint_fixture(second_left_x_distance=10.0)
    if damage == "side": readings = (dataclasses.replace(readings[0],side=0.0),) + readings[1:]
    elif damage == "nonfinite":
        alt=dataclasses.replace(readings[0].alternatives[0],distance_blocks=float("inf"))
        readings=(dataclasses.replace(readings[0],alternatives=(alt,)),)+readings[1:]
    elif damage == "foreign_row": rows=(dataclasses.replace(rows[0],left_plane=999),)+rows[1:]
    else:
        # Even a fully supported valid row cannot conceal invalid admitted evidence.
        rows=rows+(dataclasses.replace(rows[0],left_b=9.0),)
    with pytest.raises(ValueError): _select(readings,rows)


def test_tolerance_chain_is_refused_by_group_diameter():
    readings, rows = _seven_endpoint_fixture(second_left_x_distance=10.1)
    extra = _reading("a",0,10.2,67.0)
    readings = readings + (extra,)
    indexed, ids, _ = plane_graph._canonical_reading_index(readings)
    rows = tuple(dataclasses.replace(row,left_plane=row.left_plane+1) if row.left_axis != 0 else row for row in rows)
    rows += (_match(ids[id(extra)][1],0,0,0),)
    cert = _select(readings,rows)
    assert cert.status == "no" and cert.representatives == ()
    assert cert.proof_record()["counters"]["partial_prunes"] == 1
    witness = cert.proof_record()["partial_exclusions"][0][-1][-2:]
    assert witness[1][4] - witness[0][4] == pytest.approx(0.2)


def _cycle_fixture(closing_delta):
    # Distinct physical planes per X edge prevent telescoping by construction.
    values={"a":(10.0,31.0+closing_delta),"b":(11.0,20.0),"c":(21.0,31.0)}
    readings=tuple(_reading(frame,axis,distance,60.0+slot)
        for frame in ("a","b","c")
        for slot,(axis,distance) in enumerate(((0,values[frame][0]),(0,values[frame][1]),(1,10.0),(2,10.0))))
    rows=[]
    for f,g,left,right,axis in (("a","b",0,0,0),("b","c",1,0,0),("a","c",1,1,0),
            ("a","b",2,2,1),("b","c",2,2,1),("a","b",3,3,2),("b","c",3,3,2)):
        rows.append(dataclasses.replace(_match(left,right,axis,axis),left=f,right=g))
    return readings,tuple(rows)


@pytest.mark.parametrize("closing,expected", [(-2.0,"unique"),(-3.0,"no")])
def test_three_frame_integer_cycle_is_globally_checked(closing,expected):
    readings,rows=_cycle_fixture(closing)
    cert=selection.select_endpoint_hypotheses(readings=readings,signatures=_evidence(readings),
        candidate_matches=rows,frame_ids=("a","b","c"),quadrant_constraints=(("a","b",0),("a","c",0)),
        budget=csp.EndpointBudget(),session_sha256="a"*64,lattice_sha256="b"*64,signatures_sha256="c"*64)
    assert cert.status == expected
    assert cert.proof_record()["counters"]["physical_cycle_relaxations"] > 0
    if expected == "no":
        assert cert.representatives == ()
        assert cert.proof_record()["physical_reasons"] == {"physical_frame_cycle":1}
        assert 0 < len(cert.proof_record()["physical_exclusions"][0][-1]) <= 3


def test_relation_exact_limit_and_one_over_are_atomic():
    readings,rows=_seven_endpoint_fixture(second_left_x_distance=10.0)
    base=csp.solve_endpoint_hypotheses(readings,rows,session_frames=SESSION_FRAMES,quadrant_constraints=QUADRANT_CONSTRAINTS)
    limit=base.counters["relation_visits"]
    for cap,status in ((limit,"unique"),(limit-1,"budget_refused")):
        budget=dataclasses.replace(csp.EndpointBudget(),relation_visits=cap)
        result=csp.solve_endpoint_hypotheses(readings,rows,session_frames=SESSION_FRAMES,
            quadrant_constraints=QUADRANT_CONSTRAINTS,budget=budget)
        assert result.status == status and result.counters["relation_visits"] <= cap
        if status == "budget_refused": assert result.choices == ()


def test_finite_relations_with_overflowing_cycle_paths_fail_closed_globally():
    readings,rows=_cycle_fixture(-2.0)
    distances={"a":(0.1,10.0),"b":(1e308,0.1),"c":(1e308,10.0)}
    readings=tuple(dataclasses.replace(reading, alternatives=(dataclasses.replace(
        reading.alternatives[0],distance_blocks=distances[reading.frame][int(reading.period_pixels-60.0)]),))
        if reading.axis == 0 else reading for reading in readings)
    with pytest.raises(ValueError,match="unverified physical cycle arithmetic"):
        csp.solve_endpoint_hypotheses(readings,rows,session_frames=("a","b","c"),
            quadrant_constraints=(("a","b",0),("a","c",0)))


@pytest.mark.parametrize("cap_name", ["MAX_PROOF_BYTES", "MAX_RECORD_BYTES"])
def test_proof_and_envelope_exact_byte_limits_refuse_atomically(monkeypatch,cap_name):
    readings,rows=_seven_endpoint_fixture(second_left_x_distance=10.0)
    base=_select(readings,rows)
    size=(len(base.proof_json.encode()) if cap_name == "MAX_PROOF_BYTES"
          else len(selection._encode(selection._payload(base))))
    monkeypatch.setattr(selection,cap_name,size)
    exact=_select(readings,rows)
    assert exact.status == "unique" and exact.result_sha256 == base.result_sha256
    selection.validate_candidate_certificate(exact,readings,_evidence(readings),rows)
    monkeypatch.setattr(selection,cap_name,size-1)
    refused=_select(readings,rows)
    assert refused.status == "budget_refused" and refused.representatives == ()
    assert refused.proof_record()["choices"] == []
    assert refused.proof_record()["denial"][0] == ("proof_bytes" if cap_name == "MAX_PROOF_BYTES" else "record_bytes")
    selection.validate_candidate_certificate(refused,readings,_evidence(readings),rows)


@pytest.mark.parametrize("change", ["proof_distance","proof_sign","proof_side","proof_index","proof_digest","omit","reorder","selected_distance","selected_side"])
def test_bound_physical_proof_and_full_candidates_cannot_be_resealed(change):
    readings,rows=_seven_endpoint_fixture(second_left_x_distance=10.0)
    cert=_select(readings,rows)
    if change.startswith("proof"):
        proof=cert.proof_record()
        if change == "proof_digest": proof["physical_digest"]="0"*64
        else:
            reference=proof["choices"][0]["physical_groups"][0][3]
            field={"proof_distance":4,"proof_sign":4,"proof_side":3,"proof_index":0}[change]
            reference[field]+=1
        cert=dataclasses.replace(cert,proof_json=selection._encode(proof).decode(),result_sha256="")
    elif change == "omit": rows=rows[:-1]
    elif change == "reorder": rows=tuple(reversed(rows))
    elif change == "selected_side": readings=(dataclasses.replace(readings[0],side=-1.0),)+readings[1:]
    else:
        alt=dataclasses.replace(readings[0].alternatives[0],distance_blocks=9.0)
        readings=(dataclasses.replace(readings[0],alternatives=(alt,)),)+readings[1:]
    with pytest.raises(ValueError): selection.validate_candidate_certificate(cert,readings,_evidence(readings),rows)


def test_new_identities_and_old_v3_refuse_before_upstream_or_matcher(monkeypatch):
    from mc_photogrammetry.pipeline import image_derived_graph_v3 as old
    assert (selection.ENDPOINT_SELECTION_SCHEMA,selection.ENDPOINT_SELECTION_PRODUCER) == (
        "mc-photogrammetry-endpoint-selection-v4","m9-g-endpoint-selection-4")
    assert (joint_graph.IMAGE_DERIVED_GRAPH_SCHEMA,joint_graph.IMAGE_DERIVED_GRAPH_PRODUCER) == (
        "mc-photogrammetry-image-derived-graph-v6","m9-g-image-derived-graph-10")
    assert (old.IMAGE_DERIVED_GRAPH_SCHEMA,old.IMAGE_DERIVED_GRAPH_PRODUCER) == (
        "mc-photogrammetry-image-derived-graph-v3","m9-g-image-derived-graph-6")
    monkeypatch.setattr(old,"_sources",lambda **kwargs:pytest.fail("obsolete upstream reached"))
    monkeypatch.setattr(plane_graph,"match_plane_graph",lambda *a,**kw:pytest.fail("obsolete matcher reached"))
    result=old.derive_image_derived_graph_v3(session=None,lattice_result=None,signatures_result=None)
    assert isinstance(result,old.placement.PlacementRefusal)
    assert result.reason == "provenance_mismatch"
    readings,rows=_seven_endpoint_fixture(second_left_x_distance=10.0)
    cert=_select(readings,rows)
    for field,value in (("schema","mc-photogrammetry-endpoint-selection-v1"),
                        ("producer","m9-g-endpoint-selection-1"),
                        ("schema","mc-photogrammetry-endpoint-selection-v2"),
                        ("producer","m9-g-endpoint-selection-2"),
                        ("schema","mc-photogrammetry-endpoint-selection-v3"),
                        ("producer","m9-g-endpoint-selection-3")):
        with pytest.raises(ValueError): dataclasses.replace(cert,**{field:value},result_sha256="")


def test_production_consumes_full_admission_and_source_replay_matches(monkeypatch):
    from test_image_derived_graph_v3 import _source_records
    source=_source_records(monkeypatch)
    original=plane_graph._apply_same_axis_filter
    monkeypatch.setattr(plane_graph,"_apply_same_axis_filter",lambda matches,indexed,refusals:[])
    graph=joint_graph.derive_image_derived_graph(**source)
    assert isinstance(graph,joint_graph.ImageDerivedGraphResult),graph
    assert graph.status == "assembled" and graph.geometry_evaluations == 1
    assert graph.candidate_matches and set(graph.selection.candidate_dispositions) == {"same_axis_mismatch"}
    assert joint_graph.replay_image_derived_graph(graph_result=graph,**source) is graph
    second=joint_graph.derive_image_derived_graph(**source)
    assert graph.result_sha256 == second.result_sha256
    assert graph.selection.proof_json == second.selection.proof_json
    monkeypatch.setattr(plane_graph,"_apply_same_axis_filter",original)
    result=joint_graph.replay_image_derived_graph(graph_result=graph,**source)
    assert isinstance(result,joint_graph.placement.PlacementRefusal)


# Compact transcript controls operate solely on synthetic admitted universes.
def _closed_scalar_accounting(result):
    counts = result.counters
    assert sum(counts[k + "_bytes"] for k in csp.BYTE_CATEGORIES) == counts["scalar_payload_bytes"]
    assert counts["scalar_records_attempted"] == (counts["scalar_records_committed"]
                                                 + counts["scalar_records_uncommitted"])
    assert counts["search_events_attempted"] == counts["search_events_committed"] + counts["search_events_uncommitted"]
    assert counts["search_events_committed"] == counts["physical_hashed_assignments"] + counts["partial_prunes"]
    assert counts["partial_prunes"] == counts["retained_partial_exclusions"] + counts["omitted_partial_exclusions"]
    assert counts["retained_partial_exclusions"] + counts["retained_physical_exclusions"] <= 128


def test_compact_relations_recover_exact_source_fields_without_full_row_hash(monkeypatch):
    readings, rows = _seven_endpoint_fixture(second_left_x_distance=10.0)
    indexed = plane_graph._canonical_reading_index(readings)[0]
    q, selected = {"a": 0, "b": 0}, dict.fromkeys(indexed, 0)
    full = [plane_graph.signed_physical_relation(indexed, row, q, selected, i)
            for i, row in enumerate(rows)]
    def forbidden(*args, **kwargs):
        pytest.fail("compact relation reserialized a complete source row")
    monkeypatch.setattr(plane_graph.json, "dumps", forbidden)
    for i, row in enumerate(rows):
        compact = plane_graph.signed_physical_relation(indexed, row, q, selected, i, compact=True)
        assert compact[:3] == full[i][:3]
        assert compact[3:] == tuple(endpoint[1] for endpoint in full[i][5])
    result = csp.solve_endpoint_hypotheses(readings, rows, session_frames=SESSION_FRAMES,
                                         quadrant_constraints=QUADRANT_CONSTRAINTS)
    assert result.status == "unique"
    _closed_scalar_accounting(result)


def test_more_than_128_physical_exclusions_are_all_hashed_and_charged(monkeypatch):
    # Isolate the complete-leaf stream contract; default mixed events have their
    # own shared-slot regression below.
    monkeypatch.setattr(csp, "_ForcedDiameter", _UnprunedControl)
    readings, rows = _seven_endpoint_fixture(second_left_x_distance=9.0)
    # Equal b labels intentionally address distinct canonical alternative slots.
    readings = tuple(dataclasses.replace(r, alternatives=r.alternatives * (3 if i == 0 else 2))
                     for i, r in enumerate(readings))
    emitted, records = [], []
    actual = csp._Work.physical
    def observed(work, *args):
        before = work.counts["disposition_bytes"]
        actual(work, *args)
        index, q, selected, reason, digest, witness = args
        records.append((index, tuple(selected[key] for key in work.endpoint_keys),
                        csp.PHYSICAL_REASON_CODES[reason], digest, witness))
        emitted.append(work.counts["disposition_bytes"] - before)
    monkeypatch.setattr(csp._Work, "physical", observed)
    result = csp.solve_endpoint_hypotheses(readings, rows, session_frames=SESSION_FRAMES,
                                         quadrant_constraints=QUADRANT_CONSTRAINTS)
    assert result.status == "no" and result.exhaustive
    counts = result.counters
    assert counts["physical_assignments"] == counts["physical_hashed_assignments"] == 192
    assert counts["physical_rejected"] == 192 and counts["retained_physical_exclusions"] == 128
    assert counts["omitted_physical_exclusions"] == 64
    assert len(emitted) == 192 and min(emitted[128:]) > 0
    assert len({row[1] for row in records}) == 192
    import hashlib
    meter, stream = csp._Work(csp.EndpointBudget()), hashlib.sha256()
    stream.update(meter.encode(["physical-stream", csp.TRANSCRIPT_POLICY, result.source_digest],
                              category="disposition"))
    for row in records:
        stream.update(meter.encode(["leaf", row], category="disposition"))
    assert stream.hexdigest() == result.physical_digest
    assert result == csp.solve_endpoint_hypotheses(readings, rows, session_frames=SESSION_FRAMES,
                                                quadrant_constraints=QUADRANT_CONSTRAINTS)
    _closed_scalar_accounting(result)


@pytest.mark.parametrize("resource", ["scalar_payload_bytes", "scalar_row_bytes"])
def test_compact_scalar_exact_cap_and_one_over_close_partial_accounting(resource):
    readings, rows = _seven_endpoint_fixture(second_left_x_distance=10.0)
    base = csp.solve_endpoint_hypotheses(readings, rows, session_frames=SESSION_FRAMES,
                                       quadrant_constraints=QUADRANT_CONSTRAINTS)
    size = base.counters[resource]
    exact = csp.solve_endpoint_hypotheses(readings, rows, session_frames=SESSION_FRAMES,
        quadrant_constraints=QUADRANT_CONSTRAINTS,
        budget=dataclasses.replace(csp.EndpointBudget(), **{resource: size}))
    assert exact.status == "unique" and exact.choices == base.choices
    refused = csp.solve_endpoint_hypotheses(readings, rows, session_frames=SESSION_FRAMES,
        quadrant_constraints=QUADRANT_CONSTRAINTS,
        budget=dataclasses.replace(csp.EndpointBudget(), **{resource: size - 1}))
    assert refused.status == "budget_refused" and not refused.exhaustive and refused.choices == ()
    assert refused.denial[0] == resource
    assert refused.counters["scalar_records_uncommitted"] == 1
    _closed_scalar_accounting(exact)
    _closed_scalar_accounting(refused)


def test_disposition_exhaustion_mid_record_never_commits_a_unique_leaf(monkeypatch):
    readings, rows = _seven_endpoint_fixture(second_left_x_distance=10.0)
    actual, start = csp._Work.physical, []
    def observed(work, *args):
        start.append(work.counts["scalar_payload_bytes"])
        return actual(work, *args)
    monkeypatch.setattr(csp._Work, "physical", observed)
    base = csp.solve_endpoint_hypotheses(readings, rows, session_frames=SESSION_FRAMES,
                                       quadrant_constraints=QUADRANT_CONSTRAINTS)
    cap = start[0] + 8
    result = csp.solve_endpoint_hypotheses(readings, rows, session_frames=SESSION_FRAMES,
        quadrant_constraints=QUADRANT_CONSTRAINTS,
        budget=dataclasses.replace(csp.EndpointBudget(), scalar_payload_bytes=cap))
    assert base.status == "unique" and result.status == "budget_refused"
    assert result.counters["physical_assignments"] == 1
    assert result.counters["physical_hashed_assignments"] == result.counters["solutions_found"] == 0
    assert result.counters["disposition_bytes"] > 0 and result.choices == ()
    _closed_scalar_accounting(result)


@pytest.mark.parametrize("damage", ["policy", "row", "alternative", "sign", "digest", "witness"])
def test_compact_retained_reference_mutations_fail_exact_replay(damage):
    readings, rows = _seven_endpoint_fixture(second_left_x_distance=9.0)
    cert = _select(readings, rows)
    proof = cert.proof_record()
    row = proof["partial_exclusions"][0]
    if damage == "policy": proof["transcript_policy"] = "source-bound-physical-transcript-v2"
    elif damage == "digest": proof["physical_digest"] = "0" * 64
    elif damage == "witness": row[-1].pop()
    elif damage == "row": row[-1][3][0] += 1
    elif damage == "alternative": row[4][0][2] += 1
    else: row[-1][3][4] *= -1
    with pytest.raises(ValueError):
        forged = dataclasses.replace(cert, proof_json=selection._encode(proof).decode(), result_sha256="")
        selection.validate_candidate_certificate(forged, readings, _evidence(readings), rows)


@pytest.mark.parametrize("state", ["no", "unique", "multiple", "unresolved_evidence"])
def test_compact_matches_bounded_exhaustive_physical_oracle(state):
    from itertools import product
    readings, rows = _seven_endpoint_fixture(second_left_x_distance=9.0 if state == "no" else 10.0)
    if state == "multiple":
        first = readings[0]
        readings = (dataclasses.replace(first, alternatives=first.alternatives + (
            dataclasses.replace(first.alternatives[0], distance_blocks=10.05),)),) + readings[1:]
    if state == "unresolved_evidence":
        rows += (dataclasses.replace(rows[0], cell_shift_rc=(1, 0)),)
    indexed = plane_graph._canonical_reading_index(readings)[0]
    keys, solutions, ambiguity = tuple(sorted(indexed)), set(), False
    # Enumerate all alternatives; independent signed arithmetic and diameter oracle.
    for vector in product(*(range(len(indexed[key].alternatives)) for key in keys)):
        selected, extrema, groups = dict(zip(keys, vector)), {}, {}
        for i, row in enumerate(rows):
            u, v = (row.left, row.left_plane), (row.right, row.right_plane)
            left, right = indexed[u], indexed[v]
            if (left.alternatives[selected[u]] not in left.matching_alternatives(row.left_b)
                    or right.alternatives[selected[v]] not in right.matching_alternatives(row.right_b)):
                continue
            delta = (left.side * left.alternatives[selected[u]].distance_blocks
                     - right.side * right.alternatives[selected[v]].distance_blocks)
            extrema.setdefault(left.axis, []).append(delta)
            cue = {k: v for k, v in csp._row_payload(row).items()
                   if k not in ("left", "right", "left_axis", "right_axis", "left_plane",
                                "right_plane", "left_b", "right_b", "quadrant_of_right")}
            groups.setdefault((u, v), set()).add(selection._sha(cue))
        if any(max(values) - min(values) > 0.15 for values in extrema.values()):
            continue
        solutions.add(tuple((f, i, selected[(f, i)]) for f, i in keys))
        ambiguity |= any(len(group) > 1 for group in groups.values())
    expected = ("no" if not solutions else "multiple" if len(solutions) > 1
                else "unresolved_evidence" if ambiguity else "unique")
    cert = _select(readings, rows)
    assert expected == cert.status == state
    actual = {tuple(tuple(a) for a in choice["alternatives"]) for choice in cert.proof_record()["choices"]}
    assert actual == (set() if expected == "unresolved_evidence" else solutions)
    selection.validate_candidate_certificate(cert, readings, _evidence(readings), rows)


def test_propagation_singletons_and_sibling_undo_preserve_source_identity(monkeypatch):
    readings, rows = _forced_adversarial_fixture(False, free_count=1, alternatives=2)
    # First branch conflicts, second survives: stale extrema must not leak.
    readings = tuple(dataclasses.replace(r, alternatives=tuple(reversed(r.alternatives)))
                     if r.period_pixels == 100.0 else r for r in readings)
    result = _solve_fixture(readings, rows)
    assert result.status == "unique" and result.counters["partial_prunes"] == 1
    assert result.counters["partial_undo_visits"] > 0
    with monkeypatch.context() as control:
        control.setattr(csp, "_ForcedDiameter", _UnprunedControl)
        control.setattr(csp, "_preflight_relations", lambda *args: None)
        assert result.choices == _solve_fixture(readings, rows).choices
    # Add an initially free alternative unsupported by the admitted row. Arc
    # propagation creates the two mandatory X singletons before free branching.
    readings, rows = _forced_adversarial_fixture()
    readings = tuple(dataclasses.replace(r, alternatives=r.alternatives + (
        lattice_planes.DistanceAlternative(2.0, 17.0, 1.0),))
        if r.frame == "a" and r.period_pixels in _PERIODS[:2] else r for r in readings)
    propagated = _solve_fixture(readings, rows)
    assert propagated.status == "no" and propagated.counters["partial_prunes"] == 1
    assert propagated.counters["branch_nodes"] == 1 and propagated.counters["physical_assignments"] == 0
    assert propagated.pruning["values_removed"] == 2


def test_global_preflight_visits_original_pairs_all_four_orientations_before_search(monkeypatch):
    readings, rows = _forced_adversarial_fixture()
    # Exact duplicate row IDs and identical-b alternative slots stay distinct.
    readings = tuple(dataclasses.replace(r, alternatives=r.alternatives * 2)
                     if r.frame == "b" and r.axis == 1 else r for r in readings)
    rows += (rows[2],)
    actual, checked = plane_graph.signed_physical_relation, []
    def observed(indexed, row, q, selected, i, **kwargs):
        checked.append((i, selected[(row.left, row.left_plane)], selected[(row.right, row.right_plane)],
                        q[row.left], tuple(sorted(q))))
        return actual(indexed, row, q, selected, i, **kwargs)
    monkeypatch.setattr(plane_graph, "signed_physical_relation", observed)
    result = _solve_fixture(readings, rows)
    indexed = plane_graph._canonical_reading_index(readings)[0]
    expected = []
    for i, row in enumerate(rows):
        left, right = indexed[(row.left, row.left_plane)], indexed[(row.right, row.right_plane)]
        il = [a for a, alt in enumerate(left.alternatives)
              if any(alt is option for option in left.matching_alternatives(row.left_b))]
        ir = [b for b, alt in enumerate(right.alternatives)
              if any(alt is option for option in right.matching_alternatives(row.right_b))]
        expected.extend((i, a, b, q, SESSION_FRAMES) for a in il for b in ir for q in range(4))
    assert result.status == "no" and checked[:len(expected)] == expected
    assert any(row[0] == len(rows) - 1 for row in expected)


@pytest.mark.parametrize("invalid", ["unknown", "nan", "inf", "overflow"])
def test_global_preflight_cannot_hide_late_invalid_evidence(invalid):
    readings, rows = _forced_adversarial_fixture()
    assert _solve_fixture(readings, rows).counters["partial_prunes"] == 1
    if invalid == "unknown":
        rows += (dataclasses.replace(rows[-1], left_b=123.0),)
    elif invalid in ("nan", "inf"):
        readings = tuple(dataclasses.replace(r, alternatives=(dataclasses.replace(
            r.alternatives[0], distance_blocks=float(invalid)),) + r.alternatives[1:])
            if r.period_pixels == 100.0 else r for r in readings)
    else:
        # A separate late factor overflows while the original forced X factors
        # would have already closed the root frontier.
        extra = _reading("b", 0, -1e308, 1000.0)
        readings += (extra,)
        readings = tuple(dataclasses.replace(r, alternatives=(dataclasses.replace(
            r.alternatives[0], distance_blocks=1e308),) + r.alternatives[1:])
            if r.period_pixels == 100.0 else r for r in readings)
        endpoints = _canonical_endpoints(readings)
        rows += (_match(endpoints[("a", 0, 100.0)], endpoints[("b", 0, 1000.0)], 0, 0),)
        # Inserting a second right-X reading moves the Y/Z canonical indices.
        rows = tuple(dataclasses.replace(row, right_plane=row.right_plane + 1)
                     if row.right_axis != 0 else row for row in rows)
    with pytest.raises(ValueError): _solve_fixture(readings, rows)


@pytest.mark.parametrize("attribution", ["preflight", "partial_support", "partial_update", "partial_undo"])
def test_each_partial_relation_charge_exact_boundary_and_one_over(monkeypatch, attribution):
    readings, rows = _forced_adversarial_fixture()
    actual, observed = csp._Work.relation_charge, []
    def meter(work, category, amount=1):
        if category == attribution:
            observed.append((work.counts["relation_visits"], amount, work.counts[category + "_visits"]))
        return actual(work, category, amount)
    monkeypatch.setattr(csp._Work, "relation_charge", meter)
    baseline = _solve_fixture(readings, rows)
    used, amount, attributed = observed[0]
    for cap, target_count in ((used + amount, attributed + amount), (used + amount - 1, attributed)):
        refused = _solve_fixture(readings, rows,
            budget=dataclasses.replace(csp.EndpointBudget(), relation_visits=cap))
        assert refused.status == "budget_refused" and refused.choices == () and not refused.exhaustive
        assert refused.counters[attribution + "_visits"] == target_count
        _closed_scalar_accounting(refused)
    assert baseline.status == "no"


def test_undo_exhaustion_after_feasible_choice_aborts_without_a_winner():
    readings, rows = _forced_adversarial_fixture(False, free_count=1, alternatives=2)
    baseline = _solve_fixture(readings, rows)
    cap = baseline.counters["relation_visits"]
    exact = _solve_fixture(readings, rows, budget=dataclasses.replace(csp.EndpointBudget(), relation_visits=cap))
    refused = _solve_fixture(readings, rows, budget=dataclasses.replace(csp.EndpointBudget(), relation_visits=cap - 1))
    assert exact.status == "unique" and exact.choices == baseline.choices
    assert refused.status == "budget_refused" and refused.choices == ()
    assert refused.counters["solutions_found"] == 1
    assert refused.counters["partial_undo_visits"] + 1 == baseline.counters["partial_undo_visits"]
    _closed_scalar_accounting(refused)


def test_mid_partial_scalar_event_never_commits_a_closed_subtree(monkeypatch):
    readings, rows = _forced_adversarial_fixture()
    actual, starts = csp._Work.partial, []
    def observed(work, *args):
        starts.append(work.counts["scalar_payload_bytes"])
        return actual(work, *args)
    monkeypatch.setattr(csp._Work, "partial", observed)
    assert _solve_fixture(readings, rows).status == "no"
    refused = _solve_fixture(readings, rows, budget=dataclasses.replace(csp.EndpointBudget(),
        scalar_payload_bytes=starts[0] + 8))
    assert refused.status == "budget_refused" and not refused.exhaustive and refused.choices == ()
    assert refused.counters["search_events_attempted"] == refused.counters["search_events_uncommitted"] == 1
    assert refused.counters["partial_prunes"] == refused.counters["search_events_committed"] == 0
    assert refused.partial_exclusions == ()
    _closed_scalar_accounting(refused)


@pytest.mark.parametrize("damage", ["path", "mask", "forced_index", "row", "sign", "kind", "count", "digest", "policy", "candidate_order"])
def test_partial_frontier_tampering_cannot_reseal_source_replay(damage):
    readings, rows = _forced_adversarial_fixture(False, free_count=1, alternatives=3)
    cert = _select(readings, rows)
    selection.validate_candidate_certificate(cert, readings, _evidence(readings), rows)
    proof = cert.proof_record()
    event = proof["partial_exclusions"][0]
    assert event[2]
    if damage == "path": event[2][0][1] = 2
    elif damage == "mask": event[3][event[2][0][0]] = 4
    elif damage == "forced_index": event[4][-1][2] = 2
    elif damage == "row": event[-1][3][0] = event[-1][4][0]
    elif damage == "sign": event[-1][3][4] *= -1
    elif damage == "kind": event[0] = "leaf"
    elif damage == "count": proof["counters"]["partial_prunes"] += 1
    elif damage == "digest": proof["physical_digest"] = "0" * 64
    elif damage == "policy": proof["transcript_policy"] = "source-bound-physical-transcript-v2"
    else: rows = tuple(reversed(rows))
    with pytest.raises(ValueError):
        forged = dataclasses.replace(cert, proof_json=selection._encode(proof).decode(), result_sha256="")
        selection.validate_candidate_certificate(forged, readings, _evidence(readings), rows)


def _mixed_cycle_fixture():
    readings, rows = _cycle_fixture(-3.0)
    old = plane_graph._canonical_reading_index(readings)[0]
    readings = tuple(dataclasses.replace(r, alternatives=(
        dataclasses.replace(r.alternatives[0], distance_blocks=11.0), r.alternatives[0]))
        if r.frame == "a" and r.period_pixels == 60.0 else
        dataclasses.replace(r, alternatives=r.alternatives * 2) if r.axis in (1, 2) else
        dataclasses.replace(r, alternatives=r.alternatives * 3) if r.frame == "c" and r.period_pixels == 61.0 else r
        for r in readings)
    extra = _reading("a", 0, 10.0, 1000.0)
    readings += (extra,)
    indexed, identities, _ = plane_graph._canonical_reading_index(readings)
    positions = {(r.frame, r.axis, r.period_pixels): key[1] for key, r in indexed.items()}
    def position(frame, plane):
        r = old[(frame, plane)]
        return positions[(frame, r.axis, r.period_pixels)]
    rows = tuple(dataclasses.replace(row, left_plane=position(row.left, row.left_plane),
        right_plane=position(row.right, row.right_plane)) for row in rows)
    rows += (dataclasses.replace(_match(identities[id(extra)][1], 0, 0, 0), left="a", right="b"),)
    return readings, rows


def test_more_than_128_mixed_leaf_partial_events_share_slots_and_hash_omitted(monkeypatch):
    import hashlib
    readings, rows = _mixed_cycle_fixture()
    actual, records, byte_charges = csp._Work.search_event, [], []
    def observed(work, kind, row, retained):
        before = work.counts["disposition_bytes"]
        encoded = actual(work, kind, row, retained)
        records.append((kind, row))
        byte_charges.append(work.counts["disposition_bytes"] - before)
        return encoded
    monkeypatch.setattr(csp._Work, "search_event", observed)
    result = csp.solve_endpoint_hypotheses(readings, rows, session_frames=("a", "b", "c"),
        quadrant_constraints=(("a", "b", 0), ("a", "c", 0)))
    assert result.status == "no" and result.exhaustive
    counts = result.counters
    assert counts["physical_assignments"] == counts["physical_hashed_assignments"] == counts["physical_rejected"] == 192
    assert counts["partial_prunes"] == 1
    assert counts["retained_partial_exclusions"] == 1 and counts["retained_physical_exclusions"] == 127
    assert counts["omitted_physical_exclusions"] == 65 and counts["omitted_partial_exclusions"] == 0
    assert len(records) == 193 and min(byte_charges[128:]) > 0
    assert records[0][0] == "partial" and records[-1][0] == "leaf"
    meter, stream = csp._Work(csp.EndpointBudget()), hashlib.sha256()
    stream.update(meter.encode(["physical-stream", csp.TRANSCRIPT_POLICY, result.source_digest], category="disposition"))
    for kind, row in records: stream.update(meter.encode([kind, row], category="disposition"))
    assert stream.hexdigest() == result.physical_digest
    _closed_scalar_accounting(result)
    with monkeypatch.context() as fresh:
        fresh.setattr(csp._Work, "search_event", actual)
        assert result == csp.solve_endpoint_hypotheses(readings, rows, session_frames=("a", "b", "c"),
            quadrant_constraints=(("a", "b", 0), ("a", "c", 0)))
    cert = selection.select_endpoint_hypotheses(readings=readings, signatures=_evidence(readings),
        candidate_matches=rows, frame_ids=("a", "b", "c"),
        quadrant_constraints=(("a", "b", 0), ("a", "c", 0)), budget=csp.EndpointBudget(),
        session_sha256="a" * 64, lattice_sha256="b" * 64, signatures_sha256="c" * 64)
    selection.validate_candidate_certificate(cert, readings, _evidence(readings), rows)


@pytest.mark.parametrize("state", ["no", "unique", "multiple", "unresolved_evidence"])
def test_four_q_frontiers_agree_with_unpruned_control_and_exact_choices(monkeypatch, state):
    readings, rows = _seven_endpoint_fixture(second_left_x_distance=9.0 if state == "no" else 10.0)
    if state == "multiple":
        first = readings[0]
        readings = (dataclasses.replace(first, alternatives=first.alternatives + (
            dataclasses.replace(first.alternatives[0], distance_blocks=10.05),)),) + readings[1:]
    if state == "unresolved_evidence": rows += (dataclasses.replace(rows[0], cell_shift_rc=(1, 0)),)
    from itertools import product
    indexed = plane_graph._canonical_reading_index(readings)[0]
    keys, oracle, ambiguous = tuple(sorted(indexed)), set(), False
    for absolute in range(4):
        q = {"a": 0, "b": absolute}
        mandatory = {}
        for i, row in enumerate(rows):
            if (q[row.right] - q[row.left]) % 4 != row.quadrant_of_right: continue
            pair = tuple(sorted(((row.left, row.left_plane), (row.right, row.right_plane))))
            mandatory.setdefault(pair, []).append((i, row))
        if {key for pair in mandatory for key in pair} != set(keys): continue
        axes = {plane_graph.axis_transform(indexed[pair[0]].axis, q[pair[0][0]])[0] for pair in mandatory}
        if axes != {0, 1, 2}: continue
        for vector in product(*(range(len(indexed[key].alternatives)) for key in keys)):
            selected, extrema, distinguishable = dict(zip(keys, vector)), {}, False
            supported = True
            for pair, source in mandatory.items():
                cues = set()
                for i, row in source:
                    u, v = (row.left, row.left_plane), (row.right, row.right_plane)
                    left, right = indexed[u], indexed[v]
                    la, ra = left.alternatives[selected[u]], right.alternatives[selected[v]]
                    if (not any(la is a for a in left.matching_alternatives(row.left_b))
                            or not any(ra is a for a in right.matching_alternatives(row.right_b))): continue
                    axis, sign = plane_graph.axis_transform(left.axis, q[row.left])
                    other_axis, other_sign = plane_graph.axis_transform(right.axis, q[row.right])
                    assert axis == other_axis
                    delta = sign * left.side * la.distance_blocks - other_sign * right.side * ra.distance_blocks
                    if row.right < row.left: delta = -delta
                    extrema.setdefault((min(row.left, row.right), max(row.left, row.right), axis), []).append(delta)
                    cue = {name: value for name, value in csp._row_payload(row).items()
                        if name not in ("left", "right", "left_axis", "right_axis", "left_plane",
                                        "right_plane", "left_b", "right_b", "quadrant_of_right")}
                    cues.add(selection._sha(cue))
                if not cues: supported = False; break
                distinguishable |= len(cues) > 1
            if not supported or any(max(values) - min(values) > 0.15 for values in extrema.values()): continue
            oracle.add((tuple(q.items()), tuple((f, i, selected[(f, i)]) for f, i in keys)))
            ambiguous |= distinguishable
    oracle_status = ("no" if not oracle else "multiple" if len(oracle) > 1
                     else "unresolved_evidence" if ambiguous else "unique")
    result = csp.solve_endpoint_hypotheses(readings, rows, session_frames=SESSION_FRAMES)
    with monkeypatch.context() as control:
        control.setattr(csp, "_ForcedDiameter", _UnprunedControl)
        control.setattr(csp, "_preflight_relations", lambda *args: None)
        expected = csp.solve_endpoint_hypotheses(readings, rows, session_frames=SESSION_FRAMES)
    assert result.status == expected.status == oracle_status == state
    assert result.choices == expected.choices
    assert {(choice.quadrants, choice.alternatives) for choice in result.choices} == (
        set() if oracle_status == "unresolved_evidence" else oracle)
    assert result.exhaustive == expected.exhaustive
    assert result.counters["planned_assignments"] == expected.counters["planned_assignments"] == 4
    if state != "multiple":
        assert result.counters["visited_assignments"] == 4 and result.counters["unvisited_assignments"] == 0
    else:
        assert result.counters["visited_assignments"] == 1 and result.counters["unvisited_assignments"] == 3
    _closed_scalar_accounting(result)


def test_more_than_128_partial_exclusions_commit_all_omitted_events(monkeypatch):
    readings, rows = _seven_endpoint_fixture(second_left_x_distance=9.0)
    readings = tuple(dataclasses.replace(r, alternatives=r.alternatives * 16)
                     if r.frame == "a" and r.axis == 0 else r for r in readings)
    actual, events = csp._Work.partial, []
    def observed(work, row):
        before = work.counts["disposition_bytes"]
        actual(work, row)
        events.append((row, work.counts["disposition_bytes"] - before))
    monkeypatch.setattr(csp._Work, "partial", observed)
    result = _solve_fixture(readings, rows)
    assert result.status == "no" and result.exhaustive
    assert result.counters["partial_prunes"] == result.counters["search_events_committed"] == 256
    assert result.counters["retained_partial_exclusions"] == result.counters["omitted_partial_exclusions"] == 128
    assert result.counters["physical_assignments"] == result.counters["physical_rejected"] == 0
    assert len(events) == 256 and all(charge > 0 for _, charge in events[128:])
    import hashlib
    meter, stream = csp._Work(csp.EndpointBudget()), hashlib.sha256()
    stream.update(meter.encode(["physical-stream", csp.TRANSCRIPT_POLICY, result.source_digest], category="disposition"))
    for row, _ in events: stream.update(meter.encode(["partial", row], category="disposition"))
    assert stream.hexdigest() == result.physical_digest
    _closed_scalar_accounting(result)


def test_structurally_consistent_partial_counter_forgery_still_requires_source_replay():
    readings, rows = _forced_adversarial_fixture(False, free_count=1, alternatives=3)
    cert = _select(readings, rows)
    proof = cert.proof_record()
    # All structural closure equations still hold after a fabricated omitted
    # event. Exact independent replay must expose the additional closed frontier.
    for name in ("partial_checks", "partial_prunes", "omitted_partial_exclusions",
                 "search_events_attempted", "search_events_committed"):
        proof["counters"][name] += 1
    forged = dataclasses.replace(cert, proof_json=selection._encode(proof).decode(), result_sha256="")
    with pytest.raises(ValueError, match="does not recompute"):
        selection.validate_candidate_certificate(forged, readings, _evidence(readings), rows)


def test_partial_scalar_caps_and_final_proof_caps_remain_atomic(monkeypatch):
    readings, rows = _forced_adversarial_fixture()
    base = _solve_fixture(readings, rows)
    for resource in ("scalar_payload_bytes", "scalar_row_bytes"):
        size = base.counters[resource]
        exact = _solve_fixture(readings, rows, budget=dataclasses.replace(csp.EndpointBudget(), **{resource: size}))
        assert exact == base
        refused = _solve_fixture(readings, rows, budget=dataclasses.replace(csp.EndpointBudget(), **{resource: size - 1}))
        assert refused.status == "budget_refused" and refused.choices == () and not refused.exhaustive
        _closed_scalar_accounting(refused)
    cert = _select(readings, rows)
    proof_bytes = len(cert.proof_json.encode())
    record_bytes = len(selection._encode(selection._payload(cert)))
    for name, size in (("MAX_PROOF_BYTES", proof_bytes), ("MAX_RECORD_BYTES", record_bytes)):
        with monkeypatch.context() as cap:
            cap.setattr(selection, name, size)
            assert _select(readings, rows).status == "no"
            cap.setattr(selection, name, size - 1)
            refused = _select(readings, rows)
            assert refused.status == "budget_refused" and refused.proof_record()["choices"] == []
            counts = refused.proof_record()["counters"]
            assert counts["retained_partial_exclusions"] == 0 and counts["omitted_partial_exclusions"] == 1
