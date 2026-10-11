"""Declared-complete synthetic controls only: no images/captures/inference."""
from dataclasses import fields, replace
from itertools import combinations, product
import hashlib
import json

import numpy as np
import pytest

from mc_photogrammetry.pipeline import image_derived_connections as connections
from mc_photogrammetry.reconstruction import plane_graph
from mc_photogrammetry.reconstruction import _same_axis_candidate_search as oracle
from mc_photogrammetry.reconstruction import _same_axis_constraint_search as search
from mc_photogrammetry.reconstruction.lattice_planes import (
    DistanceAlternative, PlaneReading, PlaneRectification, reading_order_key,
)


def reading(frame, axis, distance, *, b=1.0, side=-1.0):
    return PlaneReading(
        frame, axis, side, (DistanceAlternative(b, distance, 1.0),),
        60.0, (0.0, 0.0), 1.0, 512, None, 8, 0.6, 0.9,
        PlaneRectification(((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
                           500.0, (120.0, 90.0), (0.0, 0.0)))


def match(readings, left, right, axis, *, right_axis=None, right_distance=None,
          left_b=1., right_b=1., yaw=0, shift=(0, 0), score=.8):
    right_axis = axis if right_axis is None else right_axis
    by_frame = {frame: sorted((r for r in readings if r.frame == frame), key=reading_order_key)
                for frame in (left, right)}
    lp = next(i for i, r in enumerate(by_frame[left]) if r.axis == axis)
    rp = next(i for i, r in enumerate(by_frame[right]) if r.axis == right_axis
              and (right_distance is None or r.alternatives[0].distance_blocks == right_distance))
    return plane_graph.PlaneMatch(left, right, axis, right_axis, left_b, right_b, yaw,
                                 5., 1., .1, 64, None, score, lp, rp, shift)


def graph(frames, *, solved_pairs=None, relative=None):
    relative = relative or {}
    records = []
    for left, right in combinations(frames, 2):
        solved = solved_pairs is None or (left, right) in solved_pairs
        records.append(connections.RelativeCameraEvidence(
            left_id=left, right_id=right, status="solved" if solved else "refused",
            reason="" if solved else "insufficient_one_to_one_support",
            baseline_direction=(1., 0., 0.) if solved else None,
            right_yaw_class=relative.get((left, right), 0) if solved else None,
            relative_rotation=((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)) if solved else None,
            match_count=8, support=8, support_floor=8, raw_keypoints=(8, 8),
            candidate_edges=8, mutual_survivors=8, median_selected_residual_degrees=0.,
            parallax_degrees=5., cheirality_fraction=1., singular_ratio=.1,
            ambiguity_margin=4, hypotheses=(), matches_sha256="1" * 64,
            session_sha256="2" * 64))
    records = tuple(records)
    return connections.ImageDerivedConnectionGraph(
        connections.IMAGE_DERIVED_CONNECTIONS_SCHEMA, 1, "ok", "", "",
        "2" * 64, connections.ConnectionBudgets(28, 512), tuple(frames),
        len(records), records, connections._solved_components(tuple(frames), records),
        ("scale", "translation"), False)


def observation(matches, dispositions=None):
    return plane_graph.SameAxisCandidateObservation(tuple(matches), tuple(dispositions)
                                                  if dispositions else ("kept",) * len(matches))


def seal(readings, obs, connection):
    result = search.declare_synthetic_envelope(
        tuple(readings), obs, connection_graph=connection,
        trusted_session_sha256=connection.session_sha256,
        trusted_graph_sha256=connection.graph_sha256)
    assert result.status == "sealed", result.reason
    assert result.envelope is not None and not result.reconstruction_authorized
    return result.envelope


def bind(readings, obs, connection, envelope=None, **kwargs):
    if "trusted_session_sha256" not in kwargs:
        kwargs["trusted_session_sha256"] = connection.session_sha256
    if "trusted_graph_sha256" not in kwargs:
        kwargs["trusted_graph_sha256"] = connection.graph_sha256
    kwargs.setdefault("source_kind", search.SOURCE_KIND)
    return search.bind_constraint_universe(
        tuple(readings), obs, connection_graph=connection,
        producer_envelope=envelope if envelope is not None else seal(readings, obs, connection), **kwargs)


def run(readings, matches, connection=None, *, caps=search.HARD_CAPS, dispositions=None):
    connection = connection or graph(tuple(sorted({r.frame for r in readings})))
    bound = bind(readings, observation(matches, dispositions), connection)
    assert bound.status == "bound", bound.reason
    assert not bound.reconstruction_authorized and not bound.universe.reconstruction_authorized
    return search.search_constraint_universe(bound.universe, caps=caps)


def control(frames=("a", "b")):
    readings = tuple(reading(frame, axis, 2. + i) for i, frame in enumerate(frames) for axis in range(3))
    matches = tuple(match(readings, "a", "b", axis) for axis in range(3))
    return readings, matches


def assert_atomic(result, reason=None):
    assert result.status == "refused"
    if reason:
        assert result.reason == reason
    assert result.outcomes == result.candidates == ()
    assert result.universe_sha256 == "" and not result.reconstruction_authorized


def old_run(readings, matches, connection):
    return oracle.search_same_axis_candidates(
        readings, matches, connection_graph=connection,
        trusted_session_sha256=connection.session_sha256,
        trusted_graph_sha256=connection.graph_sha256)


def parity(readings, matches, connection):
    old, new = old_run(readings, matches, connection), run(readings, matches, connection)
    assert (new.status, new.reason) == (old.status, old.reason)
    assert tuple(o.oracle_outcome for o in new.outcomes) == old.outcomes
    assert new.candidates == old.candidates
    complete = tuple(oracle.candidate_identity(m) for m in new.candidates)
    for outcome in new.outcomes:
        expected = tuple(identity for identity in complete if identity not in outcome.oracle_outcome.identities)
        assert outcome.omitted_identities == expected
        assert not outcome.reconstruction_authorized
    return new


@pytest.mark.parametrize("n", range(3, 10))
def test_complete_payload_parity_three_through_nine(n):
    readings, matches = control()
    matches += tuple(replace(matches[i % 3], cell_shift_rc=(i + 1, 0), score=.2 + i / 20)
                     for i in range(n - 3))
    result = parity(readings, matches, graph(("a", "b")))
    assert result.status.startswith("provisional_")
    if n > 3:
        assert any(o.oracle_outcome.solver_reason == "b_unresolved" for o in result.outcomes)


def lower_fixture():
    readings, base = control(("a", "b", "c"))
    readings += (reading("c", 0, 9.),)
    low = match(readings, "b", "c", 0, right_distance=4., score=.3)
    high = match(readings, "b", "c", 0, right_distance=9., score=.99)
    fixed = (match(readings, "a", "b", 0), match(readings, "a", "c", 0, right_distance=4.),
             match(readings, "a", "b", 1), match(readings, "a", "b", 2))
    return readings, fixed + (high, low)


def test_low_score_and_solver_refused_rivals_exact_parity():
    readings, matches = lower_fixture()
    result = parity(readings, matches, graph(("a", "b", "c")))
    assert result.status == "provisional_ambiguous"
    assert any(o.oracle_outcome.solver_reason == "plane_integer_residual" for o in result.outcomes)
    assert any(oracle.candidate_identity(matches[-1]) in o.oracle_outcome.identities
               and o.oracle_outcome.solver_reason == "" for o in result.outcomes)


def test_exact_endpoint_b_conflicts_and_partial_old_domain_parity():
    readings, matches = control(("a", "b", "c"))
    readings = tuple(replace(r, alternatives=r.alternatives + (DistanceAlternative(2., 2. * r.alternatives[0].distance_blocks, .5),))
                     if r.axis == 0 else r for r in readings)
    matches += (replace(matches[0], left_b=2., right_b=2.),)
    result = parity(readings, matches, graph(("a", "b", "c"), solved_pairs={("a", "b")}))
    assert len(result.outcomes) == 8
    assert {dict(o.oracle_outcome.quadrants)["c"] for o in result.outcomes} == {0, 1, 2, 3}


def test_atomic_semantic_refusal_order_matches_old_oracle():
    readings, matches = control()
    parity(readings, matches[:2], graph(("a", "b")))
    readings, matches = lower_fixture()
    # Force every full-axis leaf to contain an inconsistent fractional phase.
    readings = tuple(replace(r, alternatives=(DistanceAlternative(1., 2.4, 1.),))
                     if r.frame == "a" and r.axis == 0 else r for r in readings)
    subset = (matches[0], matches[1], matches[2], matches[3], matches[-1])
    parity(readings, subset, graph(("a", "b", "c")))
    readings, matches = control()
    contradictory = graph(("a", "b", "c"), relative={("a", "c"): 1})
    extra = readings + tuple(reading("c", axis, 4.) for axis in range(3))
    # Both validated graph contracts contain a contradictory solved cycle.
    envelope = search.declare_synthetic_envelope(
        extra, observation(matches), connection_graph=contradictory,
        trusted_session_sha256=contradictory.session_sha256,
        trusted_graph_sha256=contradictory.graph_sha256)
    assert envelope.status == "refused" and envelope.reason == "constraint_contradiction"
    assert old_run(extra, matches, contradictory).reason == "constraint_contradiction"


def finite_camera_fixture(frames):
    readings = tuple(reading(frame, axis, 2.) for frame in frames for axis in range(3))
    # All four yaw choices can support all three world axes: horizontal axes
    # swap for odd quadrants. These are independent literal relation rows.
    rows = []
    for q in range(4):
        for axis in range(3):
            right_axis = axis if q % 2 == 0 or axis == 1 else 2 - axis
            rows.append(match(readings, "a", "b", axis, right_axis=right_axis, yaw=q,
                              shift=(q, axis)))
    return readings, tuple(rows)


def literal_assignments(frames, edges):
    return {tuple(zip(frames, (0, *tail))) for tail in product(range(4), repeat=len(frames) - 1)
            if all((dict(zip(frames, (0, *tail)))[right]
                    - dict(zip(frames, (0, *tail)))[left]) % 4 == q for left, right, q in edges)}


def literal_outcomes(readings, matches, frames, allowed_yaws):
    retained = []
    ordered = tuple(sorted(matches, key=oracle.candidate_identity))
    for yaw_rows in sorted(allowed_yaws):
        qmap = dict(yaw_rows)
        for mask in range(1, 1 << len(ordered)):
            selected = tuple(m for i, m in enumerate(ordered) if mask & (1 << i))
            if any((qmap[m.right] - qmap[m.left]) % 4 != m.quadrant_of_right for m in selected):
                continue
            if {plane_graph.axis_transform(m.left_axis, qmap[m.left])[0] for m in selected} != {0, 1, 2}:
                continue
            bmap = {}
            conflict = False
            for m in selected:
                for endpoint, b in (((m.left, m.left_plane), m.left_b), ((m.right, m.right_plane), m.right_b)):
                    conflict |= endpoint in bmap and bmap[endpoint] != b
                    bmap[endpoint] = b
            if conflict:
                continue
            solved = plane_graph.solve_component(readings, selected,
                quadrant_constraints=tuple((frames[0], f, qmap[f]) for f in frames[1:]))
            retained.append(oracle.CandidateOutcome(
                tuple(oracle.candidate_identity(m) for m in selected), solved.frames,
                tuple(sorted(solved.quadrants.items())), tuple(sorted(solved.translations.items())),
                tuple(sorted(solved.axis_residuals.items())), tuple(sorted(solved.axis_resultants.items())),
                tuple(sorted(solved.blocks_per_period.items())), solved.reason, solved.scale_known, solved.alternatives))
    return tuple(sorted(retained, key=lambda o: (o.quadrants, o.identities)))


@pytest.mark.parametrize("mode", ("empty", "partial", "intermediate", "refused"))
def test_independent_literal_camera_domain_controls(mode):
    frames = ("a", "b", "c") if mode == "partial" else ("a", "b")
    readings, matches = finite_camera_fixture(frames)
    if mode in ("empty", "refused"):
        connection = graph(frames, solved_pairs=set())
        expected_edges = ()
    elif mode == "partial":
        connection = graph(frames, solved_pairs={("a", "b")}, relative={("a", "b"): 2})
        expected_edges = (("a", "b", 2),)
    else:
        connection = graph(("a", "b", "middle"), solved_pairs={("a", "middle"), ("b", "middle")},
                           relative={("a", "middle"): 1, ("b", "middle"): 3})
        expected_edges = (("a", "b", 2),)
    bound = bind(readings, observation(matches), connection)
    assert bound.universe.projected_relations == expected_edges
    assert bound.universe.omitted_frame_ids == (("middle",) if mode == "intermediate" else ())
    result = search.search_constraint_universe(bound.universe)
    allowed = literal_assignments(frames, expected_edges)
    expected = literal_outcomes(readings, matches, frames, allowed)
    assert result.status.startswith("provisional_")
    assert tuple(o.oracle_outcome for o in result.outcomes) == expected
    assert {o.oracle_outcome.quadrants for o in result.outcomes} == allowed
    assert all(dict(o.oracle_outcome.quadrants)["a"] == 0 for o in result.outcomes)


def test_omitted_camera_cycle_contradiction_refuses_before_projection():
    readings, matches = control()
    connection = graph(("a", "b", "x", "y", "z"),
                       solved_pairs={("a", "b"), ("x", "y"), ("x", "z"), ("y", "z")},
                       relative={("x", "z"): 1})
    # External envelope supplied before bind; graph contradiction has priority
    # over envelope mismatch and cannot disappear with omitted reading frames.
    envelope = seal(readings, observation(matches), graph(("a", "b")))
    bound = bind(readings, observation(matches), connection, envelope)
    assert bound.status == "refused" and bound.reason == "constraint_contradiction"
    assert bound.universe is None


def test_first_reading_frame_gauge_when_graph_root_is_omitted():
    readings, matches = finite_camera_fixture(("a", "b"))
    connection = graph(("0", "a", "b"), relative={("0", "a"): 1, ("0", "b"): 3, ("a", "b"): 2})
    result = run(readings, matches, connection)
    assert {o.oracle_outcome.quadrants for o in result.outcomes} == {(("a", 0), ("b", 2))}


@pytest.mark.parametrize("side", (-1., 1.))
def test_independent_signed_distance_translation(side):
    readings, matches = control()
    readings = tuple(replace(r, side=side) for r in readings)
    result = run(readings, matches)
    assert result.status == "provisional_unique"
    outcome = result.outcomes[0]
    assert dict(outcome.oracle_outcome.translations)["a"] == (0., 0., 0.)
    assert dict(outcome.oracle_outcome.translations)["b"] == (-side, -side, -side)
    assert dict(outcome.oracle_outcome.axis_resultants) == {0: 1., 1: 1., 2: 1.}
    assert dict(outcome.axis_ranks) == {0: 1, 1: 1, 2: 1}


def test_nonzero_cycle_passes_frozen_phase_gates_independent_phase_recomposition():
    # Axis 0 targets: a->b=1, a->c=2.06, b->c=1. Nonzero cycle
    # 1+1-2.06=-.06. Anchored LS gives b=1.02, c=2.04. Faces:
    # (-2,-1.98), (-2,-2.02), (-1.98,-1.96). Every phase remains
    # near zero; equation closure is not a frozen solver gate.
    readings, _ = control(("a", "b", "c"))
    readings += (reading("c", 0, 4.06),)
    matches = (match(readings, "a", "b", 0),
               match(readings, "a", "c", 0, right_distance=4.06),
               match(readings, "b", "c", 0, right_distance=4.),
               match(readings, "a", "b", 1), match(readings, "a", "b", 2))
    result = parity(readings, matches, graph(("a", "b", "c")))
    all_ids = tuple(sorted(oracle.candidate_identity(m) for m in matches))
    outcome = next(o for o in result.outcomes if o.oracle_outcome.identities == all_ids)
    assert outcome.oracle_outcome.solver_reason == ""
    assert abs(dict(outcome.cycle_sums)[0][0]) == pytest.approx(.06)
    translations = dict(outcome.oracle_outcome.translations)
    assert translations["b"][0] == pytest.approx(1.02)
    assert translations["c"][0] == pytest.approx(2.04)
    faces = np.array([-2., -1.98, -2., -2.02, -1.98, -1.96])
    phase = np.exp(2j * np.pi * np.mod(faces, 1.))
    expected_r = abs(phase.mean())
    expected_rho = np.abs(np.angle(phase / phase.mean())).mean() / (2 * np.pi)
    assert dict(outcome.oracle_outcome.axis_resultants)[0] == pytest.approx(expected_r)
    assert dict(outcome.oracle_outcome.axis_residuals)[0] == pytest.approx(expected_rho)
    assert expected_r > .90 and expected_rho < .15


def test_rank_deficiency_is_diagnostic_and_axis_coverage_matches_frozen_solver():
    readings, matches = control(("a", "b", "c"))
    result = parity(readings, matches, graph(("a", "b", "c")))
    assert result.status == "provisional_unique"
    assert dict(result.outcomes[0].axis_ranks) == {0: 1, 1: 1, 2: 1}
    assert all(rank < len(readings) // 3 - 1 for _, rank in result.outcomes[0].axis_ranks)
    frozen = plane_graph.solve_component(readings, matches[:2],
                                        quadrant_constraints=(("a", "b", 0), ("a", "c", 0)))
    assert frozen.reason == "insufficient_axis_coverage"
    assert_atomic(run(readings, matches[:2]), "missing_required_axes")


@pytest.mark.parametrize("distance,faces", (
    (3.5, (-2., -1.75, -2., -2.25)),
    (3.8, (-2., -1.6, -2., -2.4)),
))
def test_independent_literal_phase_refusals_are_retained(distance, faces):
    readings, _ = control()
    readings += (reading("b", 0, distance),)
    matches = tuple(match(readings, "a", "b", axis, right_distance=3.) for axis in range(3))
    matches += (match(readings, "a", "b", 0, right_distance=distance),)
    result = parity(readings, matches, graph(("a", "b")))
    full = tuple(sorted(oracle.candidate_identity(m) for m in matches))
    outcome = next(o.oracle_outcome for o in result.outcomes if o.oracle_outcome.identities == full)
    phasors = np.exp(2j * np.pi * np.mod(np.array(faces), 1.))
    expected_r = abs(phasors.mean())
    expected_rho = np.abs(np.angle(phasors / phasors.mean())).mean() / (2 * np.pi)
    assert dict(outcome.axis_resultants)[0] == pytest.approx(expected_r)
    assert dict(outcome.axis_residuals)[0] == pytest.approx(expected_rho)
    assert expected_r < .90
    assert outcome.solver_reason == "plane_integer_residual"
    if distance == 3.8:
        assert expected_rho > .15


def test_iterative_digest_matches_independent_canonical_json_and_digest_fault(monkeypatch):
    payload = [{"b": [1, 2, None], "a": {"unicode": "\u2603", "bool": True}}, "x"]
    expected = hashlib.sha256(b"fixture" + json.dumps(payload, sort_keys=True,
        separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()
    assert search._digest("fixture", payload, 10000, search._Budget(search.HARD_CAPS)) == expected
    readings, matches = control()
    bound = bind(readings, observation(matches), graph(("a", "b")))
    def fault(*args, **kwargs):
        raise RuntimeError("synthetic digest fault")
    # Exercise the public atomic path without modifying any global dependency.
    class FaultHash:
        sha256 = staticmethod(fault)
    monkeypatch.setattr(search, "hashlib", FaultHash)
    assert_atomic(search.search_constraint_universe(bound.universe), "digest_fault")


def test_binding_permutation_stability_graph_order_is_bound():
    readings, matches = lower_fixture()
    connection = graph(("a", "b", "c"))
    dispositions = ("kept", "same_axis_mismatch", "kept", "kept", "kept", "same_axis_mismatch")
    obs = observation(matches, dispositions)
    envelope = seal(readings, obs, connection)
    base = bind(readings, obs, connection, envelope)
    for reading_order in (readings[::-1], readings[2:] + readings[:2]):
        pairs = tuple(zip(matches, dispositions))[::-1]
        perm = observation(tuple(m for m, _ in pairs), tuple(d for _, d in pairs))
        bound = bind(reading_order, perm, connection, envelope)
        assert bound == base
        assert search.search_constraint_universe(bound.universe) == search.search_constraint_universe(base.universe)
    # Full graph ordering and pins change even when reading/yaw relations agree.
    changed = graph(("b", "a", "c"))
    refused = bind(readings, obs, changed, envelope)
    assert refused.status == "refused" and refused.reason == "producer_envelope_mismatch"


@pytest.mark.parametrize("mutation", (
    {"side": 1.}, {"axis": 2}, {"period_pixels": 61.}, {"phase_pixels": (.25, .5)},
    {"extent": 1.5}, {"samples": 256}, {"structural_share": .25}, {"confidence": .25},
    {"alternatives": (DistanceAlternative(1., 2., .25),)},
    {"alternatives": (DistanceAlternative(1., 2., 1.), DistanceAlternative(2., 4., .5))},
))
def test_every_reading_mutation_and_unused_reading_tamper_refuses(mutation):
    readings, matches = control()
    readings += (reading("a", 0, 12.),)
    matches = tuple(match(readings, "a", "b", axis) for axis in range(3))
    obs = observation(matches)
    connection = graph(("a", "b"))
    envelope = seal(readings, obs, connection)
    for position in (0, len(readings) - 1):
        mutated = tuple(replace(r, **mutation) if i == position else r for i, r in enumerate(readings))
        bound = bind(mutated, obs, connection, envelope)
        assert bound.status == "refused" and bound.universe is None


@pytest.mark.parametrize("mutation", (
    {"left_plane": 1}, {"right_plane": 1}, {"left_b": 2.}, {"right_b": 2.},
    {"quadrant_of_right": 2}, {"score": .123}, {"psr": 4.}, {"peak": .7},
    {"second_peak": .2}, {"overlap_cells": 65}, {"cell_shift_rc": (1, 3)},
))
def test_every_candidate_mutation_refuses_original_envelope(mutation):
    readings, matches = control()
    connection = graph(("a", "b"))
    envelope = seal(readings, observation(matches), connection)
    changed = observation((replace(matches[0], **mutation), *matches[1:]))
    refused = bind(readings, changed, connection, envelope)
    assert refused.status == "refused" and refused.universe is None


def test_dispositions_yaw_graph_counts_envelope_and_omissions_refuse():
    readings, matches = control()
    connection = graph(("a", "b"))
    obs = observation(matches)
    envelope = seal(readings, obs, connection)
    cases = ((readings, observation(matches, ("same_axis_mismatch", "kept", "kept")), connection, envelope),
             (readings, observation(matches[:2]), connection, envelope),
             (readings, obs, graph(("a", "b"), relative={("a", "b"): 2}), envelope),
             (readings, obs, connection, replace(envelope, sha256="3" * 64)))
    for args in cases:
        refused = bind(*args)
        assert refused.status == "refused" and refused.universe is None
    for field in fields(search.SyntheticEnvelope):
        if field.name != "sha256":
            refused = bind(readings, obs, connection, replace(envelope, **{field.name: getattr(envelope, field.name) + 1}))
            assert refused.reason == "producer_envelope_mismatch"
    assert bind(readings, obs, connection, envelope, trusted_universe_sha256="4" * 64).reason == "binding_mismatch"
    assert bind(readings, obs, connection, envelope, trusted_graph_sha256="4" * 64).reason == "unbound_constraints"
    assert bind(readings, obs, connection, envelope, trusted_session_sha256="4" * 64).reason == "unbound_constraints"


@pytest.mark.parametrize("kind", ("real", "image_derived", "", None))
def test_unsupported_provenance_is_typed_atomic_refusal(kind):
    readings, matches = control()
    connection = graph(("a", "b"))
    envelope = seal(readings, observation(matches), connection)
    result = bind(readings, observation(matches), connection, envelope, source_kind=kind)
    assert result.reason == "unsupported_provenance" and result.universe is None


@pytest.mark.parametrize("conflict", (False, True))
def test_duplicate_or_conflicting_identity_refuses(conflict):
    readings, matches = control()
    obs = observation((*matches, replace(matches[0], score=.1) if conflict else matches[0]))
    result = bind(readings, obs, graph(("a", "b")), search.SyntheticEnvelope("0" * 64, 4, 6, 2, 4, 0))
    assert result.reason == ("conflicting_identity" if conflict else "duplicate_identity")
    assert result.universe is None


def structured_437():
    readings, matches = control()
    readings = tuple(replace(r, alternatives=tuple(DistanceAlternative(float(b), float(b) * r.alternatives[0].distance_blocks, 1. / b)
                                                  for b in range(1, 5))) if r.axis == 0 else r for r in readings)
    compatible = tuple(replace(matches[0], left_b=float(b), right_b=float(b)) for b in range(1, 5)) + matches[1:]
    incompatible = tuple(match(readings, "a", "b", 1, yaw=i % 3 + 1, shift=(i + 1, 0))
                         for i in range(431))
    all_rows = compatible + incompatible
    dispositions = tuple("same_axis_mismatch" if i % 2 else "kept" for i in range(len(all_rows)))
    return readings, all_rows, dispositions


def test_structured_437_complete_certificate_and_deterministic_counters():
    readings, matches, dispositions = structured_437()
    result = run(readings, matches, dispositions=dispositions)
    assert result.status == "provisional_ambiguous", result.reason
    assert len(result.candidates) == 437 and len(result.outcomes) == 4
    assert {m.quadrant_of_right for m in result.candidates} == {0, 1, 2, 3}
    assert all(o.oracle_outcome.solver_reason == "" for o in result.outcomes)
    assert {dict(o.oracle_outcome.blocks_per_period)[("a", 0)] for o in result.outcomes} == {1., 2., 3., 4.}
    assert all(len(o.oracle_outcome.identities) == 3 and len(o.omitted_identities) == 434 for o in result.outcomes)
    assert all(not o.reconstruction_authorized for o in result.outcomes)
    assert result.counters.numerical_evaluations == result.counters.retained_outcomes == 4
    assert result.counters.visited_nodes < 100
    repeat = run(readings[::-1], matches[::-1], dispositions=dispositions[::-1])
    assert result == repeat
    print("DSCALE-02 structured437", result.universe_sha256, result.counters)


def test_adversarial_branching_caps_after_private_progress_are_atomic():
    readings, matches = control()
    matches += tuple(replace(matches[i % 3], cell_shift_rc=(i + 1, 0)) for i in range(9))
    bound = bind(readings, observation(matches), graph(("a", "b")))
    limits = (("visited_nodes", 100), ("numerical_evaluations", 2), ("retained_outcomes", 2),
              ("diagnostic_bytes", 70000))
    for field, cap in limits:
        result = search.search_constraint_universe(bound.universe, caps=replace(search.HARD_CAPS, **{field: cap}))
        assert_atomic(result, field + "_cap")
        assert result.counters.visited_nodes > 0
        assert result.counters.numerical_evaluations > 0
        print("DSCALE-02 adversarial", field, result.counters)


def test_oversize_preflight_before_candidate_workspace(monkeypatch):
    readings, matches = control()
    connection = graph(("a", "b"))
    envelope = seal(readings, observation(matches), connection)
    def forbidden(*args, **kwargs):
        pytest.fail("no canonical copy/index/solver after shape preflight refusal")
    monkeypatch.setattr(plane_graph, "_canonical_reading_index", forbidden)
    monkeypatch.setattr(plane_graph, "solve_component", forbidden)
    huge = observation(tuple(replace(matches[0], cell_shift_rc=(i, 0)) for i in range(4097)))
    result = bind(readings, huge, connection, envelope)
    assert result.reason == "candidate_identities_cap" and result.counters.memo_bytes == 0
    result = bind(readings * 43, observation(matches), connection, envelope)
    assert result.reason == "readings_cap" and result.counters.memo_bytes == 0
    alt = tuple(DistanceAlternative(float(i + 1), float(i + 1), 1.) for i in range(65))
    changed = (replace(readings[0], alternatives=alt), *readings[1:])
    result = bind(changed, observation(matches), connection, envelope)
    assert result.reason == "alternatives_per_reading_cap" and result.counters.memo_bytes == 0
    result = bind(readings, observation(matches), graph(tuple(f"frame-{i}" for i in range(33))), envelope)
    assert result.reason == "frames_cap" and result.counters.memo_bytes == 0
    huge_readings = (replace(readings[0], frame="x" * (16 * 1024 * 1024)), *readings[1:])
    result = bind(huge_readings, observation(matches), connection, envelope)
    assert result.reason == "canonical_input_bytes_cap"


def test_expanded_input_preflight_before_candidate_workspace(monkeypatch):
    readings, matches = control()
    obs = observation(matches)
    connection = graph(("a", "b"))
    envelope = seal(readings, obs, connection)
    scanned = search._scan((readings, obs, connection), search._Budget(search.HARD_CAPS))
    reserve = 3 * scanned + 16384
    at_cap = bind(readings, obs, connection, envelope,
                  caps=replace(search.HARD_CAPS, canonical_input_bytes=reserve))
    assert at_cap.status == "bound", at_cap.reason
    assert at_cap.counters.canonical_input_bytes == reserve

    def forbidden(*args, **kwargs):
        pytest.fail("no canonical workspace after expanded input preflight refusal")

    monkeypatch.setattr(plane_graph, "_canonical_reading_index", forbidden)
    monkeypatch.setattr(plane_graph, "solve_component", forbidden)
    monkeypatch.setattr(search, "_camera_projection", forbidden)
    monkeypatch.setattr(search, "_primitive", forbidden)
    caps = replace(search.HARD_CAPS, canonical_input_bytes=reserve - 1)
    assert scanned <= caps.canonical_input_bytes < reserve
    results = (
        bind(readings, obs, connection, envelope, caps=caps),
        search.declare_synthetic_envelope(
            readings, obs, connection_graph=connection,
            trusted_session_sha256=connection.session_sha256,
            trusted_graph_sha256=connection.graph_sha256, caps=caps),
    )
    for result in results:
        assert result.status == "refused" and result.reason == "canonical_input_bytes_cap"
        assert result.universe is None and result.envelope is None
        assert result.counters.canonical_input_bytes == scanned
        assert not result.reconstruction_authorized


@pytest.mark.parametrize("field", [f.name for f in fields(search.ConstraintCaps)])
def test_caps_cannot_be_raised_or_malformed(field):
    readings, matches = control()
    connection = graph(("a", "b"))
    bound = bind(readings, observation(matches), connection)
    for value in (True, 0, getattr(search.HARD_CAPS, field) + 1):
        assert_atomic(search.search_constraint_universe(bound.universe,
                      caps=replace(search.HARD_CAPS, **{field: value})), "invalid_caps")


def test_malformed_unbound_graph_and_binding_replay_refuse_atomically():
    readings, matches = control()
    connection = graph(("a", "b"))
    obs = observation(matches)
    envelope = seal(readings, obs, connection)
    bound = bind(readings, obs, connection, envelope)
    tampered = replace(bound.universe, omitted_frame_ids=("invented",))
    assert_atomic(search.search_constraint_universe(tampered), "binding_mismatch")
    tampered = replace(bound.universe, universe_sha256="0" * 64)
    assert_atomic(search.search_constraint_universe(tampered), "binding_mismatch")
    assert_atomic(search.search_constraint_universe(None), "invalid_binding")
    assert bind(readings, obs, None, envelope, trusted_session_sha256="2" * 64,
                trusted_graph_sha256="0" * 64).reason == "unbound_constraints"
    # Deliberately bypass frozen dataclass construction: nested digest replay
    # must fail, rather than treating malformed evidence as an empty domain.
    object.__setattr__(connection.evidence[0], "right_yaw_class", 2)
    refused = bind(readings, obs, connection, envelope)
    assert refused.status == "refused" and refused.universe is None


def test_solver_fault_and_invalid_numerics_discard_private_outcomes(monkeypatch):
    readings, matches = control()
    matches += (replace(matches[0], cell_shift_rc=(1, 0)),)
    bound = bind(readings, observation(matches), graph(("a", "b")))
    original = plane_graph.solve_component
    calls = []
    def fault(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("synthetic solver fault")
        return original(*args, **kwargs)
    monkeypatch.setattr(plane_graph, "solve_component", fault)
    result = search.search_constraint_universe(bound.universe)
    assert_atomic(result, "invalid_solver_output")
    assert result.counters.retained_outcomes == 1 and result.counters.numerical_evaluations == 2


@pytest.mark.parametrize("numeric", (float("nan"), float("inf")))
def test_nonfinite_solver_diagnostics_refuse_atomically(monkeypatch, numeric):
    readings, matches = control()
    bound = bind(readings, observation(matches), graph(("a", "b")))
    original = plane_graph.solve_component
    def invalid(*args, **kwargs):
        solved = original(*args, **kwargs)
        return replace(solved, axis_resultants={0: numeric, 1: 1., 2: 1.})
    monkeypatch.setattr(plane_graph, "solve_component", invalid)
    assert_atomic(search.search_constraint_universe(bound.universe), "invalid_solver_output")


def test_signed_distance_overflow_is_a_typed_atomic_fault():
    readings, matches = control()
    readings = tuple(replace(r, side=-1. if r.frame == "a" else 1.,
                            alternatives=(DistanceAlternative(1., 1e308, 1.),))
                     if r.axis == 0 else r for r in readings)
    result = run(readings, matches)
    assert_atomic(result, "invalid_solver_output")


def test_operation_workspace_evidence_caps_are_typed_and_atomic():
    readings, matches = control()
    bound = bind(readings, observation(matches), graph(("a", "b")))
    for field, cap in (("candidate_incidence_checks", 10), ("memo_bytes", 1024),
                       ("canonical_input_bytes", 100), ("diagnostic_bytes", 100)):
        result = search.search_constraint_universe(bound.universe,
                                                  caps=replace(search.HARD_CAPS, **{field: cap}))
        assert_atomic(result, field + "_cap")


def test_integrity_only_resealing_cannot_prove_real_completeness():
    readings, matches = control()
    connection = graph(("a", "b"))
    full = seal(readings, observation(matches), connection)
    subset = observation(matches[:2])
    assert bind(readings, subset, connection, full).reason == "producer_envelope_mismatch"
    resealed = seal(readings, subset, connection)
    assert resealed != full
    assert bind(readings, subset, connection, resealed).status == "bound"
    assert_atomic(run(readings, matches[:2]), "missing_required_axes")
