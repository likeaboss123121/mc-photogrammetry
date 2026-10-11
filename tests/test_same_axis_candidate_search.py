"""Synthetic pure controls only; no images, saved captures or inference."""
from dataclasses import fields, replace
from itertools import combinations, product
import sys
import tracemalloc

import numpy as np
import pytest

from mc_photogrammetry.pipeline import image_derived_connections as connections
from mc_photogrammetry.reconstruction import plane_graph
from mc_photogrammetry.reconstruction import _same_axis_candidate_search as search
from mc_photogrammetry.reconstruction.lattice_planes import (
    DistanceAlternative, PlaneReading, PlaneRectification, reading_order_key,
)


def reading(frame, axis, distance, *, b=1.0):
    return PlaneReading(
        frame, axis, -1.0, (DistanceAlternative(b, distance, 1.0),),
        60.0, (0.0, 0.0), 1.0, 512, None, 8, 0.6, 0.9,
        PlaneRectification(((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
                           500.0, (120.0, 90.0), (0.0, 0.0)))


def match(readings, left, right, axis, *, right_distance=None, score=0.8):
    ordered = {frame: sorted((r for r in readings if r.frame == frame),
                            key=reading_order_key) for frame in (left, right)}
    lp = next(i for i, r in enumerate(ordered[left]) if r.axis == axis)
    rp = next(i for i, r in enumerate(ordered[right]) if r.axis == axis
              and (right_distance is None or r.alternatives[0].distance_blocks == right_distance))
    return plane_graph.PlaneMatch(
        left, right, axis, axis, 1., 1., 0, 5., 1., 0.1, 64, None, score,
        lp, rp, (0, 0))


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


def run(readings, candidates, connection=None, **kwargs):
    connection = connection or graph(tuple(sorted({r.frame for r in readings})))
    return search.search_same_axis_candidates(
        tuple(readings), tuple(candidates), connection_graph=connection,
        trusted_session_sha256=connection.session_sha256,
        trusted_graph_sha256=connection.graph_sha256, **kwargs)


def observation(matches, dispositions):
    return plane_graph.SameAxisCandidateObservation(
        tuple(matches), tuple(dispositions))


def bind(readings, obs, connection=None, **kwargs):
    connection = connection or graph(tuple(sorted({r.frame for r in readings})))
    kwargs.setdefault("trusted_session_sha256", connection.session_sha256)
    kwargs.setdefault("trusted_graph_sha256", connection.graph_sha256)
    return search.bind_same_axis_candidate_universe(
        tuple(readings), obs, connection_graph=connection, **kwargs)


def bound_run(readings, obs, connection=None, **kwargs):
    connection = connection or graph(tuple(sorted({r.frame for r in readings})))
    kwargs.setdefault("trusted_session_sha256", connection.session_sha256)
    kwargs.setdefault("trusted_graph_sha256", connection.graph_sha256)
    return search.search_bound_same_axis_candidates(
        tuple(readings), obs, connection_graph=connection, **kwargs)


def control():
    readings = tuple(reading(frame, axis, 2. + position)
                     for position, frame in enumerate(("a", "b")) for axis in range(3))
    matches = tuple(match(readings, "a", "b", axis) for axis in range(3))
    return readings, matches


def independently_exhaust(readings, matches, connection):
    """Literal tiny powerset/quadrant oracle, with no search helper calls."""
    frames = tuple(sorted({r.frame for r in readings}))
    by_frame = {f: sorted((r for r in readings if r.frame == f), key=reading_order_key)
                for f in frames}
    records = []
    for tail in product(range(4), repeat=len(frames) - 1):
        q = dict(zip(frames, (0, *tail)))
        if any((q[r.right_id] - q[r.left_id]) % 4 != r.right_yaw_class
               for r in connection.evidence if r.status == "solved"):
            continue
        for size in range(1, len(matches) + 1):
            for chosen in combinations(matches, size):
                chosen = tuple(sorted(chosen, key=lambda m: (
                    m.left, m.right, m.left_plane, m.right_plane, m.left_axis,
                    m.right_axis, m.left_b, m.right_b, m.quadrant_of_right,
                    (0,) if m.cell_shift_rc is None else (1, *m.cell_shift_rc))))
                if any((q[m.right] - q[m.left]) % 4 != m.quadrant_of_right for m in chosen):
                    continue
                labels = {0: (0, 1, 2), 1: (2, 1, 0), 2: (0, 1, 2), 3: (2, 1, 0)}
                if {labels[q[m.left]][m.left_axis] for m in chosen} != {0, 1, 2}:
                    continue
                bs = {}
                conflict = False
                for m in chosen:
                    for endpoint, value in (((m.left, m.left_plane), m.left_b),
                                            ((m.right, m.right_plane), m.right_b)):
                        assert by_frame[endpoint[0]][endpoint[1]].distance_for(value) is not None
                        if endpoint in bs and bs[endpoint] != value:
                            conflict = True
                        bs[endpoint] = value
                if conflict:
                    continue
                solved = plane_graph.solve_component(
                    readings, chosen, quadrant_constraints=tuple((frames[0], f, q[f])
                                                               for f in frames[1:]))
                # Compare complete payloads using literal endpoint identities.
                ids = tuple(sorted((m.left, m.right, m.left_plane, m.right_plane,
                                    m.left_axis, m.right_axis, m.left_b, m.right_b,
                                    m.quadrant_of_right, (0,) if m.cell_shift_rc is None
                                    else (1, *m.cell_shift_rc)) for m in chosen))
                records.append((ids, tuple(sorted(solved.quadrants.items())),
                                tuple(sorted(solved.translations.items())), solved.reason,
                                tuple(sorted(solved.axis_residuals.items())),
                                tuple(sorted(solved.axis_resultants.items())),
                                tuple(sorted(solved.blocks_per_period.items())),
                                solved.scale_known, solved.alternatives))
    return set(records)


def outcome_records(result):
    return {(r.identities, r.quadrants, r.translations, r.solver_reason, r.axis_residuals,
             r.axis_resultants, r.blocks_per_period, r.scale_known, r.alternatives)
            for r in result.outcomes}


def assert_atomic(result, reason):
    assert result.status == "refused"
    assert result.reason == reason
    assert result.outcomes == result.candidates == ()
    assert result.diagnostic_bytes == 0
    assert result.reconstruction_authorized is False


def test_control_preserves_unchanged_solver_fields_and_authorizes_nothing():
    readings, matches = control()
    solved = plane_graph.solve_component(readings, matches, quadrant_constraints=(("a", "b", 0),))
    result = run(readings, matches)
    assert result.status == "provisional_unique"
    only = result.outcomes[0]
    assert only.solver_reason == solved.reason == ""
    assert only.translations == tuple(sorted(solved.translations.items()))
    assert only.alternatives == solved.alternatives
    assert result.search_nodes == 16 and result.solver_calls == 1
    assert not result.reconstruction_authorized


@pytest.mark.parametrize("extra", (False, True))
def test_independent_tiny_exhaustive_oracle_payload_equality(extra):
    readings, matches = control()
    if extra:
        readings += (reading("b", 0, 8.),)
        matches = tuple(match(readings, "a", "b", axis, right_distance=3.)
                        for axis in range(3)) + (match(readings, "a", "b", 0, right_distance=8.),)
    connection = graph(("a", "b"))
    result = run(readings, matches, connection)
    assert result.status.startswith("provisional_")
    assert outcome_records(result) == independently_exhaust(readings, matches, connection)


def lower_fixture():
    readings = tuple(reading(frame, axis, 2. + position)
                     for position, frame in enumerate(("a", "b", "c")) for axis in range(3))
    readings += (reading("c", 0, 9.),)
    low = match(readings, "b", "c", 0, right_distance=4., score=.3)
    high = match(readings, "b", "c", 0, right_distance=9., score=.99)
    fixed = (match(readings, "a", "b", 0),
             match(readings, "a", "c", 0, right_distance=4.),
             match(readings, "a", "b", 1), match(readings, "a", "b", 2))
    return readings, fixed + (high, low), fixed, high, low


def test_lower_score_jointly_consistent_identity_survives_local_drop():
    readings, matches, fixed, high, low = lower_fixture()
    index, _, _ = plane_graph._canonical_reading_index(readings)
    assert plane_graph._apply_same_axis_filter([high, low], index, []) == [high]
    result = run(readings, matches)
    low_ids = tuple(sorted(search.candidate_identity(m) for m in (*fixed, low)))
    high_ids = tuple(sorted(search.candidate_identity(m) for m in (*fixed, high)))
    assert next(o for o in result.outcomes if o.identities == low_ids).solver_reason == ""
    assert next(o for o in result.outcomes if o.identities == high_ids).solver_reason == "plane_integer_residual"
    assert result.status == "provisional_ambiguous"
    assert outcome_records(result) == independently_exhaust(readings, matches, graph(("a", "b", "c")))


@pytest.mark.parametrize("score", (.8, .2, .99))
def test_equal_and_score_tradeoffs_preserve_every_rival(score):
    readings, matches = control()
    readings += (reading("b", 0, 8.),)
    matches = tuple(match(readings, "a", "b", axis, right_distance=3.) for axis in range(3))
    matches += (match(readings, "a", "b", 0, right_distance=8., score=score),)
    result = run(readings, matches)
    assert result.status == "provisional_ambiguous"
    assert len(result.outcomes) == 3
    assert len([r for r in result.outcomes if r.solver_reason == ""]) == 2
    assert not result.reconstruction_authorized


def test_permutation_invariance_including_complete_identities_and_counts():
    readings, matches, *_ = lower_fixture()
    baseline = run(readings, matches)
    for r in (readings[::-1], readings[3:] + readings[:3]):
        for m in (matches[::-1], matches[2:] + matches[:2]):
            assert run(r, m) == baseline


def test_partial_constraints_retain_all_quadrant_rivals():
    readings = tuple(reading(frame, axis, 2.) for frame in ("a", "b", "c") for axis in range(3))
    matches = tuple(match(readings, "a", "b", axis) for axis in range(3))
    connection = graph(("a", "b", "c"), solved_pairs={("a", "b")})
    result = run(readings, matches, connection)
    assert result.status == "provisional_ambiguous" and len(result.outcomes) == 4
    assert {dict(r.quadrants)["c"] for r in result.outcomes} == {0, 1, 2, 3}
    assert outcome_records(result) == independently_exhaust(readings, matches, connection)


def test_missing_axes_and_empty_constraints_abstain():
    readings, matches = control()
    assert_atomic(run(readings, matches[:2]), "missing_required_axes")
    assert_atomic(run(readings, matches, graph(("a", "b"), solved_pairs=set())), "empty_constraints")


def test_conflicting_constraint_cycle_abstains_before_search():
    readings = tuple(reading(frame, axis, 2.) for frame in ("a", "b", "c") for axis in range(3))
    matches = tuple(match(readings, "a", "b", axis) for axis in range(3))
    connection = graph(("a", "b", "c"), relative={("a", "c"): 1})
    result = run(readings, matches, connection)
    assert_atomic(result, "constraint_contradiction")
    assert result.search_nodes == result.solver_calls == 0


@pytest.mark.parametrize("conflict", (False, True))
def test_duplicate_and_conflicting_complete_identities_abstain(conflict):
    readings, matches = control()
    extra = replace(matches[0], score=.1) if conflict else matches[0]
    assert_atomic(run(readings, (*matches, extra)),
                  "conflicting_identity" if conflict else "duplicate_identity")


@pytest.mark.parametrize("field,limit,reason", (
    ("candidate_identities", 2, "candidate_identities_cap"),
    ("search_nodes", 15, "search_nodes_cap"),
    ("solver_calls", 6, "solver_calls_cap"),
    ("retained_outcomes", 6, "retained_outcomes_cap"),
    ("diagnostic_bytes", 128, "diagnostic_bytes_cap"),
    ("readings", 5, "readings_cap"),
    ("frames", 1, "frames_cap"),
))
def test_every_cap_refuses_atomically_before_enumeration(field, limit, reason, monkeypatch):
    readings, matches = control()
    def forbidden(*args, **kwargs):
        pytest.fail("solver must not run after preflight refusal")
    monkeypatch.setattr(plane_graph, "solve_component", forbidden)
    result = run(readings, matches, caps=replace(search.HARD_CAPS, **{field: limit}))
    assert_atomic(result, reason)
    assert result.search_nodes == result.solver_calls == 0


@pytest.mark.parametrize("field", [f.name for f in fields(search.SearchCaps)])
def test_caps_cannot_be_raised_or_malformed(field):
    readings, matches = control()
    for value in (getattr(search.HARD_CAPS, field) + 1, True, 0):
        assert_atomic(run(readings, matches, caps=replace(search.HARD_CAPS, **{field: value})),
                      "invalid_caps")


def test_hard_identity_and_tree_caps_preflight():
    readings, matches = control()
    # Distinct shifts are complete identities, even when geometry is equal.
    assert_atomic(run(readings, tuple(replace(matches[0], cell_shift_rc=(i, 0))
                                     for i in range(65))), "candidate_identities_cap")
    assert_atomic(run(readings, tuple(replace(matches[i % 3], cell_shift_rc=(i, 0))
                                     for i in range(12))), "search_nodes_cap")
    assert_atomic(run(readings, tuple(replace(matches[i % 3], cell_shift_rc=(i, 0))
                                     for i in range(10))), "solver_calls_cap")


def test_binding_tamper_raw_tuple_and_unsolved_yaw_refuse():
    readings, matches = control()
    connection = graph(("a", "b"))
    result = search.search_same_axis_candidates(
        readings, matches, connection_graph=(("a", "b", 0),),
        trusted_session_sha256="2" * 64, trusted_graph_sha256="3" * 64)
    assert_atomic(result, "unbound_constraints")
    result = search.search_same_axis_candidates(
        readings, matches, connection_graph=connection,
        trusted_session_sha256="3" * 64, trusted_graph_sha256=connection.graph_sha256)
    assert_atomic(result, "unbound_constraints")
    object.__setattr__(connection.evidence[0], "right_yaw_class", 2)
    assert_atomic(run(readings, matches, connection), "invalid_input")


@pytest.mark.parametrize("mutation", (
    {"left_plane": 99}, {"left_b": 2.}, {"score": float("nan")},
    {"quadrant_of_right": True}, {"right_axis": 2}, {"integer_residual": .1},
))
def test_malformed_candidate_refuses_atomically(mutation):
    readings, matches = control()
    result = run(readings, (replace(matches[0], **mutation), *matches[1:]))
    assert result.status == "refused" and not result.outcomes


def test_no_solver_consistent_assignment_abstains(monkeypatch):
    readings, matches = control()
    original = plane_graph.solve_component
    def refused(*args, **kwargs):
        return replace(original(*args, **kwargs), reason="plane_integer_residual")
    monkeypatch.setattr(plane_graph, "solve_component", refused)
    assert_atomic(run(readings, matches), "no_consistent_assignment")


def test_diagnostic_cap_uses_allocation_not_encoded_length():
    readings, matches = control()
    result = run(readings, matches)
    assert result.diagnostic_bytes == search.representation_bytes(result)
    assert result.diagnostic_bytes < 2 * 1024 * 1024
    huge = "x" * (2 * 1024 * 1024)
    readings = tuple(replace(r, frame=huge) if r.frame == "a" else r for r in readings)
    assert_atomic(run(readings, matches), "diagnostic_bytes_cap")


def test_late_diagnostic_exhaustion_discards_every_private_outcome():
    readings, matches, *_ = lower_fixture()
    result = run(readings, matches, caps=replace(search.HARD_CAPS, diagnostic_bytes=20000))
    assert_atomic(result, "diagnostic_bytes_cap")
    assert result.solver_calls > 1


@pytest.mark.parametrize("quadrant", (1, 2, 3))
def test_typed_nonzero_quadrants_keep_signed_geometry(quadrant):
    # Synthetic typed camera evidence; authenticity is an upstream prerequisite.
    readings = tuple(reading(frame, axis, 2.) for frame in ("a", "b") for axis in range(3))
    labels = {1: (2, 1, 0), 2: (0, 1, 2), 3: (2, 1, 0)}[quadrant]
    matches = tuple(replace(match(readings, "a", "b", axis),
                            right_axis=labels[axis], right_plane=labels[axis],
                            quadrant_of_right=quadrant) for axis in range(3))
    connection = graph(("a", "b"), relative={("a", "b"): quadrant})
    result = run(readings, matches, connection)
    assert result.status == "provisional_unique"
    assert outcome_records(result) == independently_exhaust(readings, matches, connection)


def test_candidate_solver_fault_after_private_prefix_is_atomic(monkeypatch):
    readings, matches, *_ = lower_fixture()
    original = plane_graph.solve_component
    calls = 0
    def failing(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("synthetic solver boundary fault")
        return original(*args, **kwargs)
    monkeypatch.setattr(plane_graph, "solve_component", failing)
    result = run(readings, matches)
    assert_atomic(result, "invalid_input")
    assert result.solver_calls == 2


def test_connection_pair_orientation_does_not_break_permutation_invariance():
    readings, matches = control()
    assert run(readings, matches) == run(readings, matches, graph(("b", "a")))


@pytest.mark.parametrize("which", ("frames", "readings"))
def test_hard_reading_and_frame_input_caps(which):
    readings, matches = control()
    if which == "frames":
        readings += tuple(reading(f"extra{i}", 0, 2.) for i in range(31))
        reason = "frames_cap"
    else:
        readings += tuple(reading("a", 0, 20. + i) for i in range(123))
        reason = "readings_cap"
    result = run(readings, matches)
    assert_atomic(result, reason)
    assert result.search_nodes == result.solver_calls == 0


def test_bounded_largest_leaf_control_allocation_measurement():
    readings = tuple(reading(frame, axis, 2.) for frame in ("a", "b", "c") for axis in range(3))
    matches = tuple(match(readings, left, right, axis)
                    for left, right in combinations(("a", "b", "c"), 2) for axis in range(3))
    tracemalloc.start()
    try:
        result = run(readings, matches)
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result.status == "provisional_ambiguous"
    assert result.search_nodes == 1024 and result.solver_calls == 343
    assert len(result.outcomes) == 343
    assert 0 < result.diagnostic_bytes <= 2 * 1024 * 1024
    print(f"allocation_control: resident={result.diagnostic_bytes}, traced_current={current}, "
          f"traced_peak={peak}, nodes={result.search_nodes}, calls={result.solver_calls}")


def test_representation_independent_size_for_simple_graph():
    first = ("a", 12)
    value = (first, first)
    expected = sys.getsizeof(value) + sys.getsizeof(first) + sys.getsizeof("a") + sys.getsizeof(12)
    assert search.representation_bytes(value) == expected


@pytest.mark.parametrize("left_side,right_side", ((-1., 1.), (1., -1.)))
def test_exact_finite_1e308_opposite_faces_refuse_before_solver(left_side, right_side, monkeypatch):
    readings, matches = control()
    readings = tuple(replace(r, side=left_side if r.frame == "a" else right_side,
                             alternatives=(DistanceAlternative(1., 1e308, 1.),))
                     for r in readings)
    def forbidden(*args, **kwargs):
        pytest.fail("overflowed Python-float targets must not reach the solver")
    monkeypatch.setattr(plane_graph, "solve_component", forbidden)
    result = run(readings, matches)
    assert_atomic(result, "invalid_solver_output")
    assert result.solver_calls == 0


def test_exact_finite_1e308_scores_refuse_intermediate_objective_overflow():
    readings, matches = control()
    matches = tuple(replace(m, score=1e308) for m in matches)
    previous = np.geterr()
    result = run(readings, matches)
    assert_atomic(result, "invalid_solver_output")
    assert result.solver_calls == 1
    assert np.geterr() == previous


@pytest.mark.parametrize("distance,score,opposite", (
    (8e307, sys.float_info.max / 4., True),
    (1e308, 1e307, False),
    (2., .8, True),
))
def test_boundary_finite_arithmetic_remains_provisional(distance, score, opposite):
    readings, matches = control()
    readings = tuple(replace(r, side=1. if opposite and r.frame == "b" else -1.,
                             alternatives=(DistanceAlternative(1., distance, 1.),))
                     for r in readings)
    matches = tuple(replace(m, score=score) for m in matches)
    previous = np.geterr()
    result = run(readings, matches)
    assert result.status == "provisional_unique" and result.outcomes[0].solver_reason == ""
    assert not result.reconstruction_authorized
    assert np.geterr() == previous
    assert all(np.isfinite(r.alternatives[0].cost) for r in result.outcomes)


@pytest.mark.parametrize("field", (
    "translation", "residual", "resultant", "blocks_per_period", "scale",
    "objective_cost", "alternative_resultant", "alternative_observation",
))
@pytest.mark.parametrize("nonfinite", (float("nan"), float("inf"), -float("inf")))
def test_every_nonfinite_solver_numeric_field_refuses(field, nonfinite, monkeypatch):
    readings, matches = control()
    original = plane_graph.solve_component
    def invalid(*args, **kwargs):
        value = original(*args, **kwargs)
        if field == "translation":
            return replace(value, translations={"a": (nonfinite, 0., 0.),
                                                 "b": value.translations["b"]})
        if field == "residual":
            return replace(value, axis_residuals={0: nonfinite, 1: 0., 2: 0.})
        if field == "resultant":
            return replace(value, axis_resultants={0: nonfinite, 1: 1., 2: 1.})
        if field == "blocks_per_period":
            return replace(value, blocks_per_period={**value.blocks_per_period, ("a", 0): nonfinite})
        if field == "scale":
            return replace(value, scale_known=nonfinite)
        alternative = value.alternatives[0]
        if field == "objective_cost":
            alternative = replace(alternative, cost=nonfinite)
        elif field == "alternative_resultant":
            alternative = replace(alternative, resultants=(1., nonfinite, 1.))
        else:
            alternative = replace(alternative, axis_observations=(2, nonfinite, 2))
        return replace(value, alternatives=(alternative,))
    monkeypatch.setattr(plane_graph, "solve_component", invalid)
    assert_atomic(run(readings, matches), "invalid_solver_output")


def test_nonfinite_output_after_private_prefix_discards_all_outcomes(monkeypatch):
    readings, matches, *_ = lower_fixture()
    original = plane_graph.solve_component
    calls = 0
    def late_invalid(*args, **kwargs):
        nonlocal calls
        calls += 1
        value = original(*args, **kwargs)
        if calls == 2:
            return replace(value, axis_residuals={0: float("nan"), 1: 0., 2: 0.})
        return value
    monkeypatch.setattr(plane_graph, "solve_component", late_invalid)
    result = run(readings, matches)
    assert_atomic(result, "invalid_solver_output")
    assert result.solver_calls == 2


# --- DS-CANDIDATE-UNIVERSE-ADAPTER-01: bound raw-observation adapter ---------

def test_bound_universe_preserves_kept_and_dropped_candidates():
    readings, matches, fixed, high, low = lower_fixture()
    connection = graph(("a", "b", "c"))
    dispositions = ("kept", "kept", "kept", "kept", "kept",
                    plane_graph.SAME_AXIS_DISPOSITION_DROPPED)
    obs = observation(matches, dispositions)
    binding = bind(readings, obs, connection)
    assert binding.status == "bound"
    universe = binding.universe
    assert universe.kept_count == 5 and universe.dropped_count == 1
    assert len(universe.candidates) == len(universe.dispositions) == 6
    assert universe.dispositions.count(plane_graph.SAME_AXIS_DISPOSITION_DROPPED) == 1
    identities = {search.candidate_identity(m) for m in universe.candidates}
    assert search.candidate_identity(high) in identities
    assert search.candidate_identity(low) in identities
    assert universe.session_sha256 == connection.session_sha256
    assert universe.graph_sha256 == connection.graph_sha256
    assert len(universe.universe_sha256) == 64
    # The dropped lower-score rival still reaches a solver-consistent outcome.
    bound = bound_run(readings, obs, connection)
    plain = run(readings, matches, connection)
    assert bound.status == plain.status
    assert bound.outcomes == plain.outcomes
    low_ids = tuple(sorted(search.candidate_identity(m) for m in (*fixed, low)))
    assert any(o.identities == low_ids and o.solver_reason == "" for o in bound.outcomes)
    assert bound.reconstruction_authorized is False
    assert bound.universe_sha256 == universe.universe_sha256


def test_bound_universe_preserves_cross_quadrant_rivals():
    readings = tuple(reading(frame, axis, 2.) for frame in ("a", "b", "c")
                     for axis in range(3))
    quad0 = tuple(match(readings, "a", "b", axis) for axis in range(3))
    quad2 = tuple(replace(match(readings, "a", "b", axis), right_plane=axis,
                          quadrant_of_right=2) for axis in range(3))
    connection = graph(("a", "b", "c"), solved_pairs={("a", "b")},
                       relative={("a", "b"): 2})
    obs = observation(quad0 + quad2, ("kept",) * 6)
    binding = bind(readings, obs, connection)
    assert binding.status == "bound"
    universe = binding.universe
    assert {m.quadrant_of_right for m in universe.candidates} == {0, 2}
    preserved = {search.candidate_identity(m) for m in universe.candidates}
    assert {search.candidate_identity(m) for m in quad0} <= preserved
    assert {search.candidate_identity(m) for m in quad2} <= preserved
    result = bound_run(readings, obs, connection)
    assert result.status == "provisional_ambiguous" and len(result.outcomes) == 4
    assert {dict(r.quadrants)["c"] for r in result.outcomes} == {0, 1, 2, 3}
    assert len(result.candidates) == 6
    assert result.reconstruction_authorized is False


def test_bound_universe_preserves_mixed_b_alternatives():
    readings = control()[0]
    b_axis0 = next(r for r in readings if r.frame == "b" and r.axis == 0)
    readings = tuple(
        replace(r, alternatives=(DistanceAlternative(1.0, 3.0, 1.0),
                                 DistanceAlternative(2.0, 8.0, 1.0)))
        if r is b_axis0 else r for r in readings)
    low_b = match(readings, "a", "b", 0)
    high_b = replace(low_b, right_b=2.0)
    assert search.candidate_identity(low_b) != search.candidate_identity(high_b)
    obs = observation((low_b, high_b, match(readings, "a", "b", 1),
                       match(readings, "a", "b", 2)), ("kept",) * 4)
    connection = graph(("a", "b"))
    binding = bind(readings, obs, connection)
    assert binding.status == "bound"
    universe = binding.universe
    axis0 = [m for m in universe.candidates if m.left_axis == 0]
    assert len(axis0) == 2
    assert {m.right_b for m in axis0} == {1.0, 2.0}
    assert len({m.right_plane for m in axis0}) == 1
    result = bound_run(readings, obs, connection)
    assert result.status.startswith("provisional_")
    groups = [o.identities for o in result.outcomes]
    assert any(search.candidate_identity(low_b) in group for group in groups)
    assert any(search.candidate_identity(high_b) in group for group in groups)
    assert result.reconstruction_authorized is False


def test_bound_universe_replay_rejects_tampering_and_bad_pins():
    readings, matches = control()
    connection = graph(("a", "b"))
    obs = observation(matches, ("kept",) * len(matches))
    first = bind(readings, obs, connection)
    assert first.status == "bound"
    digest = first.universe.universe_sha256
    assert len(digest) == 64
    replay = bind(readings, obs, connection, trusted_universe_sha256=digest)
    assert replay.status == "bound" and replay.universe == first.universe
    # A scored-payload mutation changes the canonical digest...
    tampered = replace(obs, candidates=(replace(matches[0], score=.123), *matches[1:]))
    fresh = bind(readings, tampered, connection)
    assert fresh.status == "bound" and fresh.universe.universe_sha256 != digest
    # ...so replaying the original digest refuses atomically with no universe.
    refused = bind(readings, tampered, connection, trusted_universe_sha256=digest)
    assert refused.status == "refused" and refused.reason == "binding_mismatch"
    assert refused.universe is None
    # Canonical reading identity is bound too: changing the selected distance
    # behind an unchanged candidate identity changes the digest.
    mutated_readings = tuple(
        replace(r, alternatives=(DistanceAlternative(1.0, 9.0, 1.0),))
        if r.frame == "a" and r.axis == 0 else r for r in readings)
    mutated = bind(mutated_readings, obs, graph(("a", "b")))
    assert mutated.status == "bound" and mutated.universe.universe_sha256 != digest
    # Pins must be well formed and match the trusted session/graph.
    assert bind(readings, obs, connection,
                trusted_session_sha256="not-a-digest").reason == "invalid_pins"
    assert bind(readings, obs, connection,
                trusted_universe_sha256="short").reason == "invalid_pins"
    assert bind(readings, obs, connection,
                trusted_graph_sha256="3" * 64).reason == "unbound_constraints"
    assert bind(readings, obs, connection,
                trusted_session_sha256="3" * 64).reason == "unbound_constraints"


def test_bound_universe_permutation_invariance():
    readings, matches, *_ = lower_fixture()
    connection = graph(("a", "b", "c"))
    pairs = tuple(zip(matches, ("kept", "kept", "kept", "kept", "kept",
                                plane_graph.SAME_AXIS_DISPOSITION_DROPPED)))
    base = bind(readings, observation([p[0] for p in pairs],
                                      [p[1] for p in pairs]), connection)
    assert base.status == "bound"
    for perm in (pairs[::-1], pairs[2:] + pairs[:2]):
        permuted = bind(readings, observation([p[0] for p in perm],
                                              [p[1] for p in perm]), connection)
        assert permuted.status == "bound" and permuted.universe == base.universe
    baseline = bound_run(readings, observation([p[0] for p in pairs],
                                               [p[1] for p in pairs]), connection)
    assert baseline.status == "provisional_ambiguous"
    for r in (readings[::-1], readings[3:] + readings[:3]):
        assert bound_run(r, observation([p[0] for p in pairs],
                                        [p[1] for p in pairs]), connection) == baseline


@pytest.mark.parametrize("conflict", (False, True))
def test_bound_universe_duplicate_and_conflicting_identities_refuse(conflict):
    readings, matches = control()
    connection = graph(("a", "b"))
    extra = replace(matches[0], score=.1) if conflict else matches[0]
    obs = observation((*matches, extra), ("kept",) * (len(matches) + 1))
    refused = bind(readings, obs, connection)
    assert refused.status == "refused"
    assert refused.reason == ("conflicting_identity" if conflict
                              else "duplicate_identity")
    assert refused.universe is None


def test_bound_universe_rejects_malformed_observation_and_rows():
    readings, matches = control()
    connection = graph(("a", "b"))
    assert bind(readings, matches, connection).reason == "invalid_observation"
    unpaired = observation(matches, ("kept",) * len(matches))
    object.__setattr__(unpaired, "dispositions", ("kept",) * (len(matches) - 1))
    refused = bind(readings, unpaired, connection)
    assert refused.status == "refused" and refused.reason == "unpaired_observation"
    assert refused.universe is None
    unknown = observation(matches, ("kept",) * len(matches))
    object.__setattr__(unknown, "dispositions", ("bogus",) + ("kept",)
                       * (len(matches) - 1))
    assert bind(readings, unknown, connection).reason == "invalid_disposition"
    assert bind(readings, unknown, connection).universe is None


@pytest.mark.parametrize("mutation", (
    {"left_plane": 99}, {"left_b": 2.}, {"score": float("nan")},
    {"quadrant_of_right": True}, {"right_axis": 2}, {"integer_residual": .1},
))
def test_bound_universe_malformed_candidate_refuses_atomically(mutation):
    readings, matches = control()
    obs = observation((replace(matches[0], **mutation), *matches[1:]),
                      ("kept",) * len(matches))
    result = bound_run(readings, obs)
    assert result.status == "refused" and not result.outcomes
    assert result.reconstruction_authorized is False


def test_bound_universe_oversized_preflight_refuses_before_solver(monkeypatch):
    readings, matches = control()
    connection = graph(("a", "b"))

    def forbidden(*args, **kwargs):
        pytest.fail("solver must not run after bound preflight refusal")
    monkeypatch.setattr(plane_graph, "solve_component", forbidden)
    oversized = observation(
        tuple(replace(matches[0], cell_shift_rc=(i, 0)) for i in range(65)),
        ("kept",) * 65)
    result = bound_run(readings, oversized, connection)
    assert_atomic(result, "candidate_identities_cap")
    assert result.search_nodes == result.solver_calls == 0
    # The diagnostic allocation cap also refuses before any solver call.
    huge = "x" * (2 * 1024 * 1024)
    big_readings = tuple(replace(r, frame=huge) if r.frame == "a" else r
                         for r in readings)
    assert_atomic(bound_run(big_readings, observation(matches, ("kept",) * 3)),
                  "diagnostic_bytes_cap")


def test_bound_universe_requires_present_trusted_constraints():
    readings, matches = control()
    obs = observation(matches, ("kept",) * len(matches))
    assert bind(readings, obs,
                graph(("a", "b"), solved_pairs=set())).reason == "empty_constraints"


def test_bound_universe_grants_no_geometry_authority():
    readings, matches = control()
    connection = graph(("a", "b"))
    obs = observation(matches, ("kept",) * len(matches))
    binding = bind(readings, obs, connection)
    assert binding.status == "bound" and binding.universe is not None
    result = bound_run(readings, obs, connection)
    assert result.status == "provisional_unique"
    assert result.reconstruction_authorized is False
    assert result.universe_sha256 == binding.universe.universe_sha256
    assert len(result.universe_sha256) == 64
    # An atomic refusal never exposes a partial universe or any outcome.
    refused = bound_run(readings, observation(matches[:2], ("kept", "kept")),
                        connection)
    assert_atomic(refused, "missing_required_axes")
    assert refused.universe_sha256 == ""


# --- DS-CANDIDATE-UNIVERSE-REPAIR-01: full canonical endpoint reading identity

def test_endpoint_side_flip_changes_digest_and_prior_replay_refuses():
    """A valid side flip changes solver inputs but not the candidate identity."""
    readings, matches = control()
    connection = graph(("a", "b"))
    obs = observation(matches, ("kept",) * len(matches))
    first = bind(readings, obs, connection)
    assert first.status == "bound"
    digest = first.universe.universe_sha256

    index, _, _ = plane_graph._canonical_reading_index(readings)
    left = index[("a", 0)]
    # The flip keeps frame/index, axis, selected b and the resolved distance.
    flipped_readings = tuple(replace(r, side=1.0) if r.frame == "a" else r
                             for r in readings)
    flipped_index, _, _ = plane_graph._canonical_reading_index(flipped_readings)
    flipped_left = flipped_index[("a", 0)]
    assert left.side == -1.0 and flipped_left.side == 1.0
    assert flipped_left.frame == left.frame and flipped_left.axis == left.axis
    assert flipped_left.distance_for(1.0) == left.distance_for(1.0) == 2.0
    # ...yet the signed face distance the solver multiplies by side flips.
    assert (flipped_left.side * flipped_left.distance_for(1.0)
            != left.side * left.distance_for(1.0))

    fresh = bind(flipped_readings, obs, connection)
    assert fresh.status == "bound"
    assert fresh.universe.candidates == first.universe.candidates
    assert fresh.universe.dispositions == first.universe.dispositions
    assert fresh.universe.universe_sha256 != digest

    refused = bind(flipped_readings, obs, connection,
                   trusted_universe_sha256=digest)
    assert refused.status == "refused"
    assert refused.reason == "binding_mismatch"
    assert refused.universe is None


@pytest.mark.parametrize("mutation", (
    {"side": 1.0},
    {"period_pixels": 61.0},
    {"phase_pixels": (0.25, 0.5)},
    {"extent": 1.5},
    {"samples": 256},
    {"polygon": ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))},
    {"structural_share": 0.25},
    {"confidence": 0.25},
    {"alternatives": (DistanceAlternative(1.0, 2.0, 0.25),)},
))
def test_every_canonical_reading_field_is_bound_on_both_endpoints(mutation):
    """Solver-relevant reading fields outside the old flat payload are bound."""
    readings, matches = control()
    connection = graph(("a", "b"))
    obs = observation(matches, ("kept",) * len(matches))
    base = bind(readings, obs, connection)
    assert base.status == "bound"
    digest = base.universe.universe_sha256
    for frame in ("a", "b"):
        mutated = tuple(replace(r, **mutation) if r.frame == frame else r
                        for r in readings)
        fresh = bind(mutated, obs, connection)
        assert fresh.status == "bound"
        # Only the reading identity changed; candidate identity is untouched.
        assert fresh.universe.candidates == base.universe.candidates
        assert fresh.universe.dispositions == base.universe.dispositions
        assert fresh.universe.universe_sha256 != digest
        refused = bind(mutated, obs, connection, trusted_universe_sha256=digest)
        assert refused.status == "refused"
        assert refused.reason == "binding_mismatch"
        assert refused.universe is None
