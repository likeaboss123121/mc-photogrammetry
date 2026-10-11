from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import replace

import pytest

from scripts.probe_image_derived_shift_overlap import build_fixture
from test_image_derived_lattice_measure import _result
from test_plane_graph import lattice_component, make_match, make_reading
from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
from mc_photogrammetry.pipeline import image_derived_graph_attribution as subject
from mc_photogrammetry.pipeline import image_derived_signatures as signatures_mod
from mc_photogrammetry.reconstruction import lattice_planes, plane_graph


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


def test_attribution_is_output_neutral_deterministic_and_digest_bound(inputs):
    normal = graph_mod.derive_image_derived_graph(**inputs)
    first_result, first = subject.derive_image_derived_graph_attribution(**inputs)
    second_result, second = subject.derive_image_derived_graph_attribution(**inputs)
    assert first_result == normal == second_result
    assert first_result.result_sha256 == normal.result_sha256
    assert first == second
    assert first.attribution_sha256 == second.attribution_sha256
    assert first.algorithm_id == plane_graph.PLANE_CORRELATION_ALGORITHM
    assert dict(first.effective_settings)["max_shift_cells"] == 32
    assert first.schema == "mc-photogrammetry-image-derived-graph-attribution-v5"
    assert subject.IMAGE_DERIVED_GRAPH_ATTRIBUTION_SCHEMA == first.schema
    assert len(first.matcher_admitted_sha256) == 64


def test_first_failure_and_candidate_counts_are_typed(inputs):
    result, attribution = subject.derive_image_derived_graph_attribution(**inputs)
    assert result.status == "unavailable"
    assert attribution.first_failing_substage in {
        "admission", "same_axis_filter", "solver", "joint_selection"}
    assert attribution.reason == result.reason
    assert attribution.pre_admission_candidates >= \
        attribution.post_admission_candidates >= \
        attribution.post_same_axis_candidates


def test_optional_matcher_witness_collection_is_output_neutral(inputs):
    normal_result, normal_attribution = \
        subject.derive_image_derived_graph_attribution(**inputs)
    witnesses = []
    result, attribution = subject.derive_image_derived_graph_attribution(
        **inputs, matcher_witness_sink=witnesses)
    assert result == normal_result and attribution == normal_attribution
    assert len(witnesses) == attribution.post_same_axis_candidates
    assert all(isinstance(row, plane_graph.GraphMatcherWitnessEnvelopeV1)
               for row in witnesses)
    assert all(row.left_signature_sha256 and row.right_signature_sha256
               for row in witnesses)


@pytest.mark.parametrize("case", ("zero_overlap", "fixed_whitening"))
def test_pair_trace_uses_raw_and_supported_maxima_from_same_surface(case):
    left, right, known = build_fixture(case)
    trace = plane_graph.PlaneGraphTrace([])
    before_left, before_right = left.cells.copy(), right.cells.copy()
    match = plane_graph.match_plane_pair(
        left, right, quadrant=0, max_shift_cells=32, diagnostic_sink=trace)
    assert match is not None and len(trace.pairs) == 1
    row = trace.pairs[0]
    assert row.admitted and row.supported_max_shift_rc == known
    assert row.supported_max_support == match.overlap_cells
    assert row.psr == match.psr and row.score == match.score
    assert row.overlap_cells == match.overlap_cells
    assert row.valid_cells >= row.overlap_cells
    assert row.overlap_fraction == row.overlap_cells / row.valid_cells
    assert row.cell_shift_rc == match.cell_shift_rc
    assert row.left_structural_share == left.reading.structural_share
    assert row.right_structural_share == right.reading.structural_share
    assert row.raw_max_shift_rc is not None and row.raw_max_support is not None
    assert (left.cells == before_left).all() and (right.cells == before_right).all()


def test_trace_is_opt_in_and_does_not_change_matcher_output():
    left, right, _known = build_fixture("fixed_whitening")
    normal = plane_graph.match_plane_pair(left, right, quadrant=0)
    trace = plane_graph.PlaneGraphTrace([])
    diagnosed = plane_graph.match_plane_pair(
        left, right, quadrant=0, diagnostic_sink=trace)
    assert diagnosed == normal
    assert trace.post_admission_candidates == 0  # graph-level count is separate
    assert len(trace.pairs) == 1


def test_graph_trace_retains_admitted_candidates_before_solver_abstention():
    left, right, _known = build_fixture("fixed_whitening")
    trace = plane_graph.PlaneGraphTrace([])
    matches = plane_graph.match_plane_graph(
        (left.reading, right.reading), (left, right), diagnostic_sink=trace)
    solution = plane_graph.solve_component((left.reading, right.reading), matches)
    admitted = [row for row in trace.pairs if row.admitted]
    assert solution.reason
    assert admitted
    assert all(row.left_plane is not None and row.right_plane is not None
               and row.psr is not None and row.score is not None
               and row.valid_cells is not None and row.overlap_fraction is not None
               for row in admitted)


def _admitted_pair(*, psr):
    return plane_graph.PlanePairTrace(
        "a", "b", 0, 0, 0, True, "", (0, 0), 1.0, 4, (0, 0), 1.0, 4,
        left_plane=0, right_plane=0, selected=True, psr=psr, score=0.5,
        overlap_cells=4, valid_cells=8, overlap_fraction=0.5,
        cell_shift_rc=(0, 0), left_structural_share=0.6,
        right_structural_share=0.6)


def _attribution_with_pair(pair):
    digest = "0123456789abcdef" * 4
    return subject.ImageDerivedGraphAttribution(
        "mc-photogrammetry-image-derived-graph-attribution-v4",
        plane_graph.PLANE_CORRELATION_ALGORITHM,
        digest, digest, digest, digest, digest,
        (("max_shift_cells", 0),),
        "complete", "", "", 0, 0, 0, (pair,), (), (), ())


def test_positive_infinity_pair_psr_is_tagged_and_digest_is_deterministic():
    pair = _admitted_pair(psr=float("inf"))
    first = _attribution_with_pair(pair)
    second = _attribution_with_pair(pair)
    payload = first.to_record(include_digest=False)
    assert payload["pairs"][0]["psr"] == "positive_infinity"
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         allow_nan=False)
    assert "Infinity" not in encoded and "NaN" not in encoded
    expected = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    assert first.attribution_sha256 == expected
    assert first.attribution_sha256 == second.attribution_sha256
    assert first.schema == "mc-photogrammetry-image-derived-graph-attribution-v4"


@pytest.mark.parametrize("psr", (2.5, None))
def test_finite_and_none_pair_psr_are_preserved(psr):
    attribution = _attribution_with_pair(_admitted_pair(psr=psr))
    assert attribution.to_record()["pairs"][0]["psr"] == psr


@pytest.mark.parametrize("psr", (float("nan"), float("-inf")))
def test_nan_and_negative_infinity_pair_psr_are_rejected(psr):
    with pytest.raises(ValueError):
        _attribution_with_pair(_admitted_pair(psr=psr))


def _solver_alternative(quadrants, observations, resultants=(0.0, 0.0, 0.0)):
    return plane_graph.ComponentAlternative(
        tuple(quadrants), 1.0, tuple(resultants), tuple(observations))


def _solver_solution(reason, *, resultants=None, residuals=None,
                     alternatives=(), frames=("f0", "f1"), quadrants=None):
    if quadrants is None:
        quadrants = tuple(0 for _frame in frames)
    return plane_graph.ComponentSolution(
        frames=tuple(frames),
        quadrants={frame: quadrants[index]
                   for index, frame in enumerate(frames)},
        translations={frame: (0., 0., 0.) for frame in frames},
        axis_residuals=dict(residuals or {}),
        axis_resultants=dict(resultants or {}),
        scale_known=False, blocks_per_period={}, reason=reason,
        alternatives=tuple(alternatives))


def _diagnosed(monkeypatch, inputs, solution, **overrides):
    match = plane_graph.PlaneMatch(
        "f0", "f1", 0, 0, 1., 1., 0, 5., 1., .1, 16, None, .9, 0, 0)
    monkeypatch.setattr(plane_graph, "match_plane_graph",
                        lambda *_a, **_k: (match,))
    monkeypatch.setattr(plane_graph, "solve_component", lambda *_a, **_k: solution)
    return subject.derive_image_derived_graph_attribution(**inputs, **overrides)


def _alias_caps(**changes):
    caps = {
        "max_readings": graph_mod.ACTIVE_EDGE_ALIASES_MAX_READINGS,
        "max_matches": graph_mod.ACTIVE_EDGE_ALIASES_MAX_MATCHES,
        "max_active_edges": graph_mod.ACTIVE_EDGE_ALIASES_MAX_ACTIVE_EDGES,
        "max_endpoint_alternatives": (
            graph_mod.ACTIVE_EDGE_ALIASES_MAX_ENDPOINT_ALTERNATIVES),
        "max_frame_id_bytes": graph_mod.ACTIVE_EDGE_ALIASES_MAX_FRAME_ID_BYTES,
        "max_record_bytes": graph_mod.ACTIVE_EDGE_ALIASES_MAX_RECORD_BYTES,
    }
    caps.update(changes)
    return {f"alias_{name}": value for name, value in caps.items()}


def _ordering_digest(domain, payload):
    raw = json.dumps({"domain": domain,
                      "schema": graph_mod.FAILED_ACTIVE_EDGE_ALIASES_SCHEMA,
                      "payload": payload},
                     sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _record(result, attribution):
    assert result.status == "unavailable"
    assert attribution.solver_failure is not None
    record = attribution.solver_failure.to_record()
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"),
                         allow_nan=False)
    assert "NaN" not in encoded and "Infinity" not in encoded
    return record


def _full_alternative(observations=(2, 3, 4), quadrants=(0, 0)):
    return _solver_alternative(quadrants, observations)


def test_low_resultant_only_fails_the_resultant_predicate(monkeypatch, inputs):
    solution = _solver_solution(
        "plane_integer_residual",
        resultants={0: 0.5, 1: 0.95, 2: 0.95},
        residuals={0: 0.1, 1: 0.1, 2: 0.1},
        alternatives=(_full_alternative(),))
    result, attribution = _diagnosed_fixed(monkeypatch, inputs, solution)
    record = _record(result, attribution)
    assert record["solver_id"] == "plane_graph._evaluate_assignment"
    assert record["reason"] == "plane_integer_residual"
    assert record["matched_axes"] == [0, 1, 2]
    assert record["axes"] == [[0, 0.5, 0.1, 2], [1, 0.95, 0.1, 3],
                              [2, 0.95, 0.1, 4]]
    assert record["predicate_basis"] == subject.PREDICATE_BASIS
    assert record["first_failed_predicate"] == "minimum_resultant"
    assert record["resultant_below_min_axes"] == [0]
    assert record["residual_above_tolerance_axes"] == []
    assert record["residual_predicate_evaluated"] is False


def test_residual_only_fails_the_integer_tolerance_predicate(monkeypatch, inputs):
    solution = _solver_solution(
        "plane_integer_residual",
        resultants={0: 0.95, 1: 0.95, 2: 0.95},
        residuals={0: 0.1, 1: 0.2, 2: 0.1},
        alternatives=(_full_alternative(),))
    result, attribution = _diagnosed_fixed(monkeypatch, inputs, solution)
    record = _record(result, attribution)
    assert record["first_failed_predicate"] == "integer_tolerance"
    assert record["resultant_below_min_axes"] == []
    assert record["residual_above_tolerance_axes"] == [1]
    assert record["residual_predicate_evaluated"] is True


def test_both_predicates_keep_low_resultant_precedence(monkeypatch, inputs):
    solution = _solver_solution(
        "plane_integer_residual",
        resultants={0: 0.5, 1: 0.95, 2: 0.95},
        residuals={0: 0.1, 1: 0.1, 2: 0.25},
        alternatives=(_full_alternative(),))
    result, attribution = _diagnosed_fixed(monkeypatch, inputs, solution)
    record = _record(result, attribution)
    assert record["resultant_below_min_axes"] == [0]
    assert record["residual_above_tolerance_axes"] == [2]
    assert record["first_failed_predicate"] == "minimum_resultant"
    assert record["residual_predicate_evaluated"] is False


def test_exact_threshold_equality_passes_without_adjustment(monkeypatch, inputs):
    equal = _solver_solution(
        "plane_integer_residual",
        resultants={0: 0.90, 1: 0.90, 2: 0.90},
        residuals={0: 0.15, 1: 0.15, 2: 0.15})
    result, attribution = _project_detached_diagnostic(monkeypatch, inputs, equal)
    record = _record(result, attribution)
    assert record["refused"] is True
    assert record["field"] == "plane_integer_residual"
    below = _solver_solution(
        "plane_integer_residual",
        resultants={0: 0.8999, 1: 0.90, 2: 0.90},
        residuals={0: 0.15, 1: 0.15, 2: 0.15})
    result, attribution = _project_detached_diagnostic(monkeypatch, inputs, below)
    record = _record(result, attribution)
    assert record["resultant_below_min_axes"] == [0]
    assert record["residual_above_tolerance_axes"] == []


def test_partial_axis_coverage_is_null_not_zero(monkeypatch, inputs):
    solution = _solver_solution(
        "insufficient_axis_coverage",
        resultants={0: 0.5}, residuals={0: 0.1},
        alternatives=(_solver_alternative((0, 0), (2, 0, 0)),))
    result, attribution = _diagnosed_fixed(monkeypatch, inputs, solution)
    record = _record(result, attribution)
    assert record["matched_axes"] == [0]
    assert record["axes"] == [[0, 0.5, 0.1, 2], [1, None, None, None],
                              [2, None, None, None]]
    assert record["predicate_basis"] is None
    assert record["first_failed_predicate"] is None
    assert record["resultant_below_min_axes"] == []
    assert record["residual_above_tolerance_axes"] == []
    assert record["residual_predicate_evaluated"] is False


def test_earlier_reason_retains_metrics_but_marks_predicates_unreached(
        monkeypatch, inputs):
    solution = _solver_solution(
        "ambiguous_quadrant",
        resultants={0: 0.2, 1: 0.95, 2: 0.95},
        residuals={0: 0.05, 1: 0.05, 2: 0.4},
        alternatives=(_full_alternative(),))
    result, attribution = _project_detached_diagnostic(monkeypatch, inputs, solution)
    record = _record(result, attribution)
    assert record["reason"] == "ambiguous_quadrant"
    assert record["axes"][0] == [0, 0.2, 0.05, 2]
    assert record["predicate_basis"] is None
    assert record["first_failed_predicate"] is None
    assert record["resultant_below_min_axes"] == []
    assert record["residual_above_tolerance_axes"] == []
    assert record["residual_predicate_evaluated"] is False


def test_earlier_reason_without_metrics_is_all_null(monkeypatch, inputs):
    solution = _solver_solution("no_plane_reading")
    result, attribution = _project_detached_diagnostic(monkeypatch, inputs, solution)
    record = _record(result, attribution)
    assert record["reason"] == "no_plane_reading"
    assert record["matched_axes"] == []
    assert record["axes"] == [[0, None, None, None], [1, None, None, None],
                              [2, None, None, None]]


def test_observations_use_only_the_coherent_best_alternative(monkeypatch, inputs):
    solution = _solver_solution(
        "insufficient_axis_coverage",
        resultants={0: 0.5, 1: 0.5}, residuals={0: 0.1, 1: 0.1},
        alternatives=(_full_alternative((5, 6, 7)),
                      _solver_alternative((1, 0), (99, 99, 99))))
    result, attribution = _project_detached_diagnostic(monkeypatch, inputs, solution)
    record = _record(result, attribution)
    assert [row[3] for row in record["axes"]] == [5, 6, None]
    mismatched = _solver_solution(
        "insufficient_axis_coverage",
        resultants={0: 0.5, 1: 0.5}, residuals={0: 0.1, 1: 0.1},
        alternatives=(_solver_alternative((1, 0), (5, 6, 7)),))
    result, attribution = _project_detached_diagnostic(monkeypatch, inputs, mismatched)
    record = _record(result, attribution)
    assert [row[3] for row in record["axes"]] == [None, None, None]
    missing = _solver_solution(
        "insufficient_axis_coverage",
        resultants={0: 0.5, 1: 0.5}, residuals={0: 0.1, 1: 0.1})
    result, attribution = _project_detached_diagnostic(monkeypatch, inputs, missing)
    record = _record(result, attribution)
    assert [row[3] for row in record["axes"]] == [None, None, None]


@pytest.mark.parametrize("changes,field", (
    ({"resultants": {0: float("nan")}, "residuals": {0: 0.1}},
     "axis_resultants[0]"),
    ({"resultants": {3: 0.5}, "residuals": {3: 0.1}}, "axis_resultants.key"),
    ({"resultants": {0: 0.5, 1: 0.5}, "residuals": {0: 0.1}},
     "axis_coverage"),
    ({"resultants": {0: 0.5}, "residuals": {0: float("inf")}},
     "axis_residuals[0]"),
))
def test_invalid_metrics_produce_a_bounded_refusal(monkeypatch, inputs,
                                                   changes, field):
    solution = _solver_solution("insufficient_axis_coverage", **changes)
    result, attribution = _project_detached_diagnostic(monkeypatch, inputs, solution)
    record = _record(result, attribution)
    assert record["refused"] is True
    assert record["reason"] == subject.INVALID_DIAGNOSTIC
    assert record["field"] == field
    assert set(record) == {"solver_id", "refused", "reason", "field"}


def test_invalid_observations_and_settings_refuse_without_changing_result(
        monkeypatch, inputs):
    solution = _solver_solution(
        "insufficient_axis_coverage",
        resultants={0: 0.5}, residuals={0: 0.1},
        alternatives=(_solver_alternative((0, 0), (1, -2, 3)),))
    result, attribution = _project_detached_diagnostic(monkeypatch, inputs, solution)
    record = _record(result, attribution)
    assert record["refused"] is True
    assert record["field"] == "alternatives[0].axis_observations"
    normal_result, normal = _project_detached_diagnostic(monkeypatch, inputs, solution)
    assert result.result_sha256 == normal_result.result_sha256
    assert result.reason == normal_result.reason == "insufficient_axis_coverage"
    assert normal.solver_failure.recorded is False


def test_invalid_effective_thresholds_refuse_at_the_diagnostic_seam():
    snapshot = graph_mod.SolverFailureSnapshot(
        "plane_integer_residual", ("f0", "f1"), (0, 0),
        ((0, 0.5), (1, 0.95), (2, 0.95)),
        ((0, 0.1), (1, 0.1), (2, 0.1)), (0, 0), (2, 3, 4))
    for threshold, field in (
            ({"min_resultant": float("nan")},
             "effective_settings.min_resultant"),
            ({"integer_tolerance": float("inf")},
             "effective_settings.integer_tolerance")):
        settings = {"min_resultant": 0.90, "integer_tolerance": 0.15}
        settings.update(threshold)
        refusal = subject.solver_failure_diagnostic(
            snapshot, settings["min_resultant"], settings["integer_tolerance"])
        assert refusal.recorded is False
        assert refusal.to_record() == {
            "solver_id": subject.SOLVER_ID, "refused": True,
            "reason": subject.INVALID_DIAGNOSTIC, "field": field}


def test_solver_failure_is_absent_when_the_solver_never_abstains(
        monkeypatch, inputs):
    solved = _solver_solution(
        "", resultants={0: 0.95, 1: 0.95, 2: 0.95},
        residuals={0: 0.1, 1: 0.1, 2: 0.1},
        alternatives=(_full_alternative(),))
    result, attribution = _diagnosed(monkeypatch, inputs, solved)
    assert result.status == "assembled"
    assert attribution.solver_failure is None
    assert "solver_failure" not in attribution.to_record()
    assert result.result_sha256

    calls = []
    monkeypatch.setattr(plane_graph, "match_plane_graph", lambda *_a, **_k: ())
    monkeypatch.setattr(plane_graph, "solve_component",
                        lambda *_a, **_k: calls.append(1))
    result, attribution = subject.derive_image_derived_graph_attribution(**inputs)
    assert result.reason == "no_matches" and calls == []
    assert attribution.solver_failure is None
    assert "solver_failure" not in attribution.to_record()


def test_solver_failure_is_bound_into_the_v3_digest(monkeypatch, inputs):
    low = _solver_solution(
        "plane_integer_residual",
        resultants={0: 0.5, 1: 0.95, 2: 0.95},
        residuals={0: 0.1, 1: 0.1, 2: 0.1},
        alternatives=(_full_alternative(),))
    residual = _solver_solution(
        "plane_integer_residual",
        resultants={0: 0.95, 1: 0.95, 2: 0.95},
        residuals={0: 0.1, 1: 0.25, 2: 0.1},
        alternatives=(_full_alternative(),))
    first_result, first = _project_detached_diagnostic(monkeypatch, inputs, low)
    second_result, second = _project_detached_diagnostic(monkeypatch, inputs, residual)
    assert first_result.result_sha256 == second_result.result_sha256
    assert first.attribution_sha256 != second.attribution_sha256
    assert first.schema == second.schema == subject.IMAGE_DERIVED_GRAPH_ATTRIBUTION_SCHEMA
    record = first.to_record()
    assert record["solver_failure"]["reason"] == "plane_integer_residual"
    assert record["graph_result_sha256"] == first_result.result_sha256
    assert record["source_digests"]["session"] == first_result.session_sha256
    assert record["source_digests"]["lattice"] == first_result.lattice_sha256
    assert record["source_digests"]["signatures"] == first_result.signatures_sha256
    assert "solver_failure" in first.to_record(include_digest=False)
    again_result, again = _project_detached_diagnostic(monkeypatch, inputs, low)
    assert again.attribution_sha256 == first.attribution_sha256
    assert again_result == first_result


def _strict_attribution_digest(attribution):
    """Recompute the whole strict digest so no nested field escapes."""
    payload = attribution.to_record(include_digest=False)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         allow_nan=False)
    assert "NaN" not in encoded and "Infinity" not in encoded
    assert hashlib.sha256(encoded.encode("utf-8")).hexdigest() == \
        attribution.attribution_sha256
    json.dumps(attribution.to_record(), sort_keys=True,
               separators=(",", ":"), allow_nan=False)
    return payload


def _raw_alternative(quadrants, observations, resultants=(0.0, 0.0, 0.0)):
    """One alternative with the raw payload kept verbatim for malformed cases."""
    return plane_graph.ComponentAlternative(
        quadrants, 1.0, resultants, observations)


@pytest.mark.parametrize("setting", ("min_resultant", "integer_tolerance",
                                     "ambiguous_margin"))
@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_invalid_effective_threshold_is_nulled_and_strict_serializable(
        monkeypatch, inputs, setting, value):
    solution = _solver_solution(
        "plane_integer_residual",
        resultants={0: 0.5, 1: 0.95, 2: 0.95},
        residuals={0: 0.1, 1: 0.1, 2: 0.1},
        alternatives=(_full_alternative(),))
    control_result, control = _project_detached_diagnostic(monkeypatch, inputs, solution)
    assert control.solver_failure.recorded is True
    result, attribution = _project_detached_diagnostic(
        monkeypatch, inputs, solution, **{setting: value})
    assert result.status == "unavailable" and result == control_result
    assert result.result_sha256 == control_result.result_sha256
    record = _record(result, attribution)
    assert record["refused"] is True
    assert record["reason"] == subject.INVALID_DIAGNOSTIC
    assert record["field"] == f"effective_settings.{setting}"
    payload = _strict_attribution_digest(attribution)
    assert payload["effective_settings"][setting] is None
    assert {"field": setting, "reason": subject.INVALID_EFFECTIVE_SETTING,
            "type": "float"} in payload["invalid_settings"]
    assert all(value is not None
               for name, value in payload["effective_settings"].items()
               if name != setting)


def test_huge_int_threshold_is_refused_before_serialization(monkeypatch, inputs):
    huge = 10 ** 1000
    solution = _solver_solution(
        "plane_integer_residual",
        resultants={0: 0.5, 1: 0.95, 2: 0.95},
        residuals={0: 0.1, 1: 0.1, 2: 0.1},
        alternatives=(_full_alternative(),))
    control_result, _control = _project_detached_diagnostic(monkeypatch, inputs, solution)
    result, attribution = _project_detached_diagnostic(
        monkeypatch, inputs, solution, min_resultant=huge)
    assert result.result_sha256 == control_result.result_sha256
    record = _record(result, attribution)
    assert record["refused"] is True
    assert record["field"] == "effective_settings.min_resultant"
    payload = _strict_attribution_digest(attribution)
    assert payload["effective_settings"]["min_resultant"] is None
    assert {"field": "min_resultant",
            "reason": subject.INVALID_EFFECTIVE_SETTING,
            "type": "int"} in payload["invalid_settings"]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         allow_nan=False)
    assert str(huge) not in encoded


@pytest.mark.parametrize("field,resultants,residuals", (
    ("axis_resultants[0]", {0: 10 ** 1000, 1: 0.95, 2: 0.95},
     {0: 0.1, 1: 0.1, 2: 0.1}),
    ("axis_residuals[0]", {0: 0.95, 1: 0.95, 2: 0.95},
     {0: 10 ** 1000, 1: 0.1, 2: 0.1}),
))
def test_huge_int_metric_is_refused_without_changing_the_result(
        monkeypatch, inputs, field, resultants, residuals):
    control = _solver_solution(
        "plane_integer_residual",
        resultants={0: 0.5, 1: 0.95, 2: 0.95},
        residuals={0: 0.1, 1: 0.1, 2: 0.1},
        alternatives=(_full_alternative(),))
    control_result, _control = _project_detached_diagnostic(monkeypatch, inputs, control)
    solution = _solver_solution(
        "plane_integer_residual", resultants=resultants, residuals=residuals,
        alternatives=(_full_alternative(),))
    result, attribution = _project_detached_diagnostic(monkeypatch, inputs, solution)
    assert result.status == "unavailable"
    assert result.result_sha256 == control_result.result_sha256
    record = _record(result, attribution)
    assert record["refused"] is True
    assert record["field"] == field
    payload = _strict_attribution_digest(attribution)
    assert "invalid_settings" not in payload


def test_missing_best_observations_are_unavailable_and_non_throwing(
        monkeypatch, inputs):
    solution = _solver_solution(
        "insufficient_axis_coverage",
        resultants={0: 0.5, 1: 0.5}, residuals={0: 0.1, 1: 0.1},
        alternatives=(_raw_alternative((0, 0), None),))
    control = _solver_solution(
        "insufficient_axis_coverage",
        resultants={0: 0.5, 1: 0.5}, residuals={0: 0.1, 1: 0.1})
    control_result, _control = _project_detached_diagnostic(monkeypatch, inputs, control)
    result, attribution = _project_detached_diagnostic(monkeypatch, inputs, solution)
    assert result.result_sha256 == control_result.result_sha256
    record = _record(result, attribution)
    assert attribution.solver_failure.recorded is True
    assert "refused" not in record
    assert [row[3] for row in record["axes"]] == [None, None, None]
    _strict_attribution_digest(attribution)


@pytest.mark.parametrize("alternative,field", (
    (_raw_alternative((0, 0), 7), "alternatives[0].axis_observations"),
    (_raw_alternative(7, (2, 3, 4)), "alternatives[0].quadrants"),
))
def test_malformed_best_alternative_is_a_bounded_refusal(
        monkeypatch, inputs, alternative, field):
    control = _solver_solution(
        "insufficient_axis_coverage",
        resultants={0: 0.5, 1: 0.5}, residuals={0: 0.1, 1: 0.1},
        alternatives=(_full_alternative(),))
    control_result, _control = _project_detached_diagnostic(monkeypatch, inputs, control)
    solution = _solver_solution(
        "insufficient_axis_coverage",
        resultants={0: 0.5, 1: 0.5}, residuals={0: 0.1, 1: 0.1},
        alternatives=(alternative,))
    result, attribution = _project_detached_diagnostic(monkeypatch, inputs, solution)
    assert result.result_sha256 == control_result.result_sha256
    record = _record(result, attribution)
    assert record["refused"] is True
    assert record["field"] == field
    _strict_attribution_digest(attribution)


def _component(**changes):
    values = dict(
        frames=("f0", "f1"), quadrants={"f0": 0, "f1": 0},
        translations={}, axis_residuals={0: 0.1}, axis_resultants={0: 0.5},
        scale_known=False, blocks_per_period={},
        reason="insufficient_axis_coverage", alternatives=())
    values.update(changes)
    return plane_graph.ComponentSolution(**values)


@pytest.mark.parametrize("changes,field", (
    ({"axis_resultants": [(0, 0.5)]}, "axis_resultants"),
    ({"axis_residuals": 3}, "axis_residuals"),
    ({"frames": None}, "frames"),
    ({"quadrants": None}, "quadrants"),
))
def test_malformed_metric_map_extraction_is_a_bounded_refusal(
        monkeypatch, inputs, changes, field):
    control_result, _control = _project_detached_diagnostic(monkeypatch, inputs, _component())
    result, attribution = _project_detached_diagnostic(
        monkeypatch, inputs, _component(**changes))
    assert result.status == "unavailable"
    assert result.result_sha256 == control_result.result_sha256
    record = _record(result, attribution)
    assert record["refused"] is True
    assert record["field"] == field
    _strict_attribution_digest(attribution)


def test_valid_settings_are_retained_exactly_without_invalid_marker(
        monkeypatch, inputs):
    solution = _solver_solution(
        "insufficient_axis_coverage",
        resultants={0: 0.5}, residuals={0: 0.1})
    _result_value, attribution = _project_detached_diagnostic(
        monkeypatch, inputs, solution, integer_tolerance=0.2,
        min_resultant=0.85, ambiguous_margin=0.03, max_shift_cells=16)
    settings = dict(attribution.effective_settings)
    assert settings["integer_tolerance"] == 0.2
    assert settings["min_resultant"] == 0.85
    assert settings["ambiguous_margin"] == 0.03
    assert settings["max_shift_cells"] == 16
    assert attribution.invalid_settings == ()
    assert "invalid_settings" not in attribution.to_record(include_digest=False)
    _strict_attribution_digest(attribution)


def _local_alias_case(inputs):
    """A coherent match/solution pair drawn from the real local lattice readings.

    The returned reading tuple is the ordinary path's own local universe and the
    match is built from one of its canonical readings, so the production seam can
    derive a complete alias record without any caller-supplied reading or match
    list.
    """
    readings = tuple(inputs["lattice_result"].readings)
    read_by_index, _index_by_object, _frames = \
        plane_graph._canonical_reading_index(readings)
    frame = sorted({reading.frame for reading in readings})[0]
    reading = read_by_index[(frame, 0)]
    canonical_b = float(reading.alternatives[0].blocks_per_period)
    match = plane_graph.PlaneMatch(
        frame, frame, reading.axis, reading.axis, canonical_b, canonical_b, 0,
        5.0, 1.0, 0.1, 16, None, 0.9, 0, 0)
    observations = tuple(2 if axis == reading.axis else 0 for axis in range(3))
    solution = plane_graph.ComponentSolution(
        frames=(frame,), quadrants={frame: 0},
        translations={frame: (0.0, 0.0, 0.0)},
        axis_residuals={0: 0.1, 1: 0.1, 2: 0.1},
        axis_resultants={axis: 0.05 for axis in range(3)}, scale_known=False,
        blocks_per_period={}, reason="plane_integer_residual",
        alternatives=(plane_graph.ComponentAlternative(
            (0,), 1.0, (0.05, 0.05, 0.05), observations),))
    return readings, (match,), solution


def test_bz_ordinary_attribution_derives_alias_record_from_local_inputs(monkeypatch,inputs):
    source,calls=_current_failure_source(monkeypatch);captured={};original=graph_mod._failed_active_edges
    def spy(readings,matches,solution,**caps):
        captured.update(readings=readings,matches=matches,solution=solution)
        return original(readings,matches,solution,**caps)
    monkeypatch.setattr(graph_mod,"_failed_active_edges",spy)
    plain=graph_mod.derive_image_derived_graph(**source);calls.update(match=0,evaluate=0)
    result,attribution=subject.derive_image_derived_graph_attribution(**source)
    assert calls=={"match":1,"evaluate":1} and result==plain and result.result_sha256==plain.result_sha256
    aliases=attribution.solver_failure.active_edge_aliases
    assert aliases.recorded and aliases.active_edge_count==3 and aliases.axis_edge_counts==(1,1,1)
    assert aliases.conflicting_endpoints==()
    assert captured["readings"] is source["lattice_result"].readings
    assert captured["matches"]==tuple(r.match for r in result.selection.representatives)
    assert captured["solution"].reason=="plane_integer_residual"
    record=aliases.to_record();assert record["input_match_count"]==3
    graph_mod.validate_failed_active_edge_aliases(record);_strict_attribution_digest(attribution)


def test_bz_foreign_alias_overrides_are_rejected(inputs):
    for name in ("readings","matches","alias_readings","alias_matches"):
        with pytest.raises(TypeError):subject.derive_image_derived_graph_attribution(**inputs,**{name:()})
    with pytest.raises(TypeError):graph_mod.derive_historical_image_derived_graph(**inputs,solver_failure_sink=[].append,alias_readings=(),alias_matches=())
    from mc_photogrammetry.pipeline import image_derived_joint_graph as current
    from test_image_derived_graph import _current_sources
    with pytest.raises(TypeError):current.derive_image_derived_graph(**_current_sources(),alias_readings=(),alias_matches=())


def test_bz_malformed_best_identity_is_a_typed_refusal_and_output_neutral(monkeypatch,inputs):
    source,calls=_current_failure_source(monkeypatch)
    result,attribution=subject.derive_image_derived_graph_attribution(**source);before=result.result_sha256
    snapshot=[];subject.derive_image_derived_graph_attribution(**source,solver_failure_observer=snapshot.append)
    graph=graph_mod.derive_image_derived_graph(**source)
    # Malformed detached identity validation cannot alter/admit the live graph.
    solution=plane_graph.ComponentSolution(("a","b"),{"a":0,"b":0},{}, {0:.1,1:.1,2:.1}, {0:.5,1:.95,2:.95},False,{},
        "plane_integer_residual",(_solver_alternative((1,0),(2,2,2)),))
    aliases=graph_mod._failed_active_edges(result.readings,tuple(r.match for r in result.selection.representatives),solution,**_edge_caps()).to_record()
    assert aliases["status"]=="refused" and aliases["reason"]=="best_assignment_unavailable" and aliases["field"]=="best_assignment_identity"
    assert "edges" not in aliases and "conflicting_endpoints" not in aliases
    assert graph.result_sha256==before and graph.matches==() and graph.component_solution is None
    _strict_attribution_digest(attribution)


def test_bz_malformed_inactive_match_identity_is_a_typed_refusal(monkeypatch,inputs):
    source,calls=_current_failure_source(monkeypatch)
    result,attribution=subject.derive_image_derived_graph_attribution(**source)
    rows=tuple(r.match for r in result.selection.representatives)
    inactive=replace(rows[0],quadrant_of_right=2,left_b=float("nan"))
    solution=plane_graph.ComponentSolution(("a","b"),{"a":0,"b":0},{}, {0:.1,1:.1,2:.1},{0:.5,1:.95,2:.95},False,{},
        "plane_integer_residual",(_solver_alternative((0,0),(2,2,2)),))
    aliases=graph_mod._failed_active_edges(result.readings,rows+(inactive,),solution,**_edge_caps()).to_record()
    assert aliases["status"]=="refused" and aliases["reason"]=="invalid_active_edge_diagnostic" and aliases["field"]=="input_match_identity"
    assert "edges" not in aliases and "conflicting_endpoints" not in aliases
    assert graph_mod.derive_image_derived_graph(**source).result_sha256==result.result_sha256
    _strict_attribution_digest(attribution)


def test_v4_alias_digests_replay_from_the_supplied_source_matches():
    """Detached digests alone are not proof: the rows must replay from source."""
    readings, matches, _solved = _alias_fixture()
    target = next(match for match in matches
                  if match.left == "v0" and match.right == "v1"
                  and match.left_axis == 0 and match.left_plane == 0)
    mixed_matches = [replace(match, left_b=1.0 / 16.0) if match is target else match
                     for match in matches]
    solution = _solver_solution(
        "plane_integer_residual",
        resultants={0: 0.5, 1: 0.95, 2: 0.95},
        residuals={0: 0.1, 1: 0.1, 2: 0.1},
        frames=("v0", "v1", "v2"),
        quadrants=(0, 1, 0),
        alternatives=(_solver_alternative((0, 1, 0), (6, 6, 12)),))
    value = graph_mod._failed_active_edges(
        readings, mixed_matches, solution, **_edge_caps())
    graph_mod.validate_failed_active_edge_aliases(value.to_record())
    aliases = value.to_record()
    assert aliases["status"] == "complete"
    assert aliases["active_edge_count"] == 12
    assert aliases["axis_edge_counts"] == [3, 3, 6]
    assert aliases["conflicting_endpoints"] == [["v0", 0]]

    # Recompute the two source digests directly from the supplied inputs.
    quadrants = dict(solution.quadrants)
    expected_assignment = _ordering_digest(
        "failed_best_assignment",
        [[frame, quadrants[frame]] for frame in solution.frames])
    expected_identity = _ordering_digest(
        "input_match_identity",
        [[match.left, match.left_plane, match.left_axis, match.right,
          match.right_plane, match.right_axis, match.quadrant_of_right,
          float(match.left_b), float(match.right_b)]
         for match in mixed_matches])
    assert aliases["failed_assignment_sha256"] == expected_assignment
    assert aliases["input_match_identity_sha256"] == expected_identity
    assert aliases["input_match_count"] == len(mixed_matches)

    # Every reported row replays against the original ordered match sequence.
    rows = aliases["edges"]
    assert len(rows) == 12
    index, _index_by_object, _frames = plane_graph._canonical_reading_index(readings)
    for row in rows:
        (match_index, left_frame, left_plane, left_axis, left_b,
         left_canonical_b, right_frame, right_plane, right_axis, right_b,
         right_canonical_b, quadrant, world_axis) = row
        source = mixed_matches[match_index]
        assert (left_frame, right_frame) == (source.left, source.right)
        assert (left_plane, right_plane) == (source.left_plane,
                                             source.right_plane)
        assert (left_axis, right_axis) == (source.left_axis, source.right_axis)
        assert quadrant == source.quadrant_of_right
        assert left_b == source.left_b and right_b == source.right_b
        left_reading = index[(left_frame, left_plane)]
        right_reading = index[(right_frame, right_plane)]
        assert left_reading.axis == left_axis
        assert right_reading.axis == right_axis
        assert left_reading.distance_for(left_b) is not None
        assert right_reading.distance_for(right_b) is not None
        assert left_canonical_b == float(
            left_reading.matching_alternatives(left_b)[0].blocks_per_period)
        assert right_canonical_b == float(
            right_reading.matching_alternatives(right_b)[0].blocks_per_period)
        assert world_axis in (0, 1, 2)
    # The conflicting endpoint is exactly the shared one used at two aliases.
    endpoint_aliases = {}
    for row in rows:
        endpoint_aliases.setdefault((row[1], row[2]), set()).add(row[5])
        endpoint_aliases.setdefault((row[6], row[7]), set()).add(row[10])
    conflicts = sorted(endpoint for endpoint, values in endpoint_aliases.items()
                       if len(values) > 1)
    assert conflicts == [("v0", 0)]
    assert [row[4] for row in rows
            if row[1] == "v0" and row[2] == 0] == [1.0 / 16.0, 1.0]
    assert [row[4] for row in rows
            if row[1] == "v1" and row[2] == 0] == [1.0]
    # Tampering with one row invalidates the nested digest.
    tampered = dict(aliases)
    tampered_rows = [list(row) for row in rows]
    tampered_rows[0][4] = 0.5
    tampered["edges"] = tampered_rows
    tampered.pop("record_sha256")
    assert _ordering_digest(
        graph_mod.FAILED_ACTIVE_EDGE_ALIASES_DIGEST_DOMAIN, tampered) != \
        aliases["record_sha256"]


def test_bx_alias_rows_bind_the_source_raw_b_and_the_canonical_b():
    """DS-R3-BX: a raw label that is not already canonical still replays.

    The rejected v4 record wrote the canonical ``b`` into the raw column, so the
    serialized rows only replayed while the source labels happened to already be
    canonical. Here two distinct raw labels collapse onto one canonical
    alternative: the nested record must keep both raw columns distinct, one
    canonical column, no conflict, and bind those distinct raw labels.
    """
    readings, matches, _solved = _alias_fixture()
    target = next(match for match in matches
                  if match.left == "v0" and match.right == "v1"
                  and match.left_axis == 0 and match.left_plane == 0)
    noisy = 1.0 + 1e-12
    target_index = next(position for position, match in enumerate(matches)
                        if match is target)
    noisy_matches = [replace(match, left_b=noisy) if match is target else match
                     for match in matches]
    solution = _solver_solution(
        "plane_integer_residual",
        resultants={0: 0.5, 1: 0.95, 2: 0.95},
        residuals={0: 0.1, 1: 0.1, 2: 0.1},
        frames=("v0", "v1", "v2"),
        quadrants=(0, 1, 0),
        alternatives=(_solver_alternative((0, 1, 0), (6, 6, 12)),))
    result = graph_mod._failed_active_edges(readings, noisy_matches, solution,
                                            **_edge_caps())
    aliases = result.to_record()
    graph_mod.validate_failed_active_edge_aliases(aliases)
    assert aliases["status"] == "complete"
    assert aliases["active_edge_count"] == 12
    assert aliases["conflicting_endpoints"] == []
    row = next(edge for edge in aliases["edges"] if edge[0] == target_index)
    assert row[4] == noisy and row[5] == 1.0 and row[4] != row[5]
    shared = [edge for edge in aliases["edges"] if (edge[1], edge[2]) == ("v0", 0)]
    assert len(shared) >= 2
    assert sorted(edge[4] for edge in shared) == (
        [1.0] * (len(shared) - 1) + [noisy])
    assert {edge[5] for edge in shared} == {1.0}
    # The distinct raw label is bound: rewriting it must change the digest.
    tampered = dict(aliases)
    rows = [list(edge) for edge in aliases["edges"]]
    next(edge for edge in rows if edge[0] == target_index)[4] = 1.0
    tampered["edges"] = rows
    tampered.pop("record_sha256")
    assert _ordering_digest(
        graph_mod.FAILED_ACTIVE_EDGE_ALIASES_DIGEST_DOMAIN, tampered) != \
        aliases["record_sha256"]


def test_v4_alias_refusal_is_atomic_and_output_neutral(monkeypatch,inputs):
    source,calls=_current_failure_source(monkeypatch)
    control_result,control=subject.derive_image_derived_graph_attribution(**source)
    result,attribution=subject.derive_image_derived_graph_attribution(**source,**_alias_caps(max_matches=0))
    assert result==control_result and result.result_sha256==control_result.result_sha256
    aliases=_record(result,attribution)["active_edge_aliases"]
    assert set(aliases)=={"schema","status","reason","field","record_sha256"}
    assert aliases["status"]=="refused" and aliases["reason"]=="budget_exhausted" and aliases["field"]=="input_match_count"
    _strict_attribution_digest(attribution);assert attribution.attribution_sha256!=control.attribution_sha256


def test_v4_alias_diagnostic_is_absent_when_no_failure_is_solved(monkeypatch,inputs):
    from test_image_derived_graph import _current_sources,_current_unavailable
    def forbidden(*_a,**_k):pytest.fail("alias work ran outside numerical failure")
    monkeypatch.setattr(graph_mod,"_failed_active_edges",forbidden)
    result,attribution=subject.derive_image_derived_graph_attribution(**_current_sources(monkeypatch))
    assert result.status=="assembled" and result.geometry_evaluations==1
    assert attribution.solver_failure is None and "solver_failure" not in attribution.to_record()
    unavailable=_current_unavailable()
    assert unavailable.geometry_evaluations==0
    # Default/no-factor attribution uses a validated one-frame source universe.
    result,attribution=subject.derive_image_derived_graph_attribution(**inputs)
    assert result.geometry_evaluations==0 and attribution.solver_failure is None

def _edge_caps():
    """Every fixed cap at its default bound, for the v4 alias diagnostic."""
    return dict(
        max_readings=graph_mod.ACTIVE_EDGE_ALIASES_MAX_READINGS,
        max_matches=graph_mod.ACTIVE_EDGE_ALIASES_MAX_MATCHES,
        max_active_edges=graph_mod.ACTIVE_EDGE_ALIASES_MAX_ACTIVE_EDGES,
        max_endpoint_alternatives=(
            graph_mod.ACTIVE_EDGE_ALIASES_MAX_ENDPOINT_ALTERNATIVES),
        max_frame_id_bytes=graph_mod.ACTIVE_EDGE_ALIASES_MAX_FRAME_ID_BYTES,
        max_record_bytes=graph_mod.ACTIVE_EDGE_ALIASES_MAX_RECORD_BYTES)
def _alias_record(value):
    """Strict canonical encoding and domain-separated digest of one record."""
    record = value.to_record()
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"),
                         allow_nan=False)
    assert "NaN" not in encoded and "Infinity" not in encoded
    payload = dict(record)
    digest = payload.pop("record_sha256")
    recomputed = hashlib.sha256(json.dumps(
        {"domain": graph_mod.FAILED_ACTIVE_EDGE_ALIASES_DIGEST_DOMAIN,
         "schema": value.schema, "payload": payload},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    assert digest == recomputed
    assert digest == graph_mod.failed_edge_alias_record_digest(value)
    # Tampering with any bound field must invalidate the digest.
    tampered = dict(payload)
    tampered["active_edge_count"] = int(tampered.get("active_edge_count", 0)) + 1
    other = hashlib.sha256(json.dumps(
        {"domain": graph_mod.FAILED_ACTIVE_EDGE_ALIASES_DIGEST_DOMAIN,
         "schema": value.schema, "payload": tampered},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    assert other != digest
    return record
def test_bv_mixed_cross_edge_alias_reports_exactly_its_conflicting_endpoint():
    """The BU control's shared b endpoint is the one conflict the solver used.
    DS-R3-BV section 6. The failed best assignment of the BU mixed fixture must
    be reported as active edges whose resolved canonical ``b`` disagrees on the
    shared ``(v0, plane 0)`` endpoint and nowhere else. Restoring the coherent
    fixture must leave a complete but empty conflict list, and the inactive
    conflicting edge must be excluded from the active set.
    """
    gates = {"integer_tolerance": 0.15, "min_resultant": 0.90,
             "ambiguous_margin": 0.02}
    readings, matches, _eyes = lattice_component(decoy=True)
    constraints = (("v0", "v1", 1), ("v0", "v2", 0), ("v1", "v2", 3))
    baseline = plane_graph.solve_component(
        readings, matches, quadrant_constraints=constraints, **gates)
    assert baseline.reason == ""
    target = next(match for match in matches
                  if match.left == "v0" and match.right == "v1" and match.left_axis == 0)
    other = next(match for match in matches
                 if match.left == "v0" and match.right == "v2" and match.left_axis == 0)
    assert (target.left, target.left_plane) == ("v0", 0)
    assert (other.left, other.left_plane) == ("v0", 0)
    assert target.left_b == other.left_b == 1.0
    read_by_index, _index_by_object, _frames = plane_graph._canonical_reading_index(readings)
    endpoint = read_by_index[("v0", target.left_plane)]
    assert endpoint.distance_for(1.0) == pytest.approx(5.0)
    assert endpoint.distance_for(1.0 / 16.0) == pytest.approx(5.0 / 16.0)
    mixed_matches = [replace(match, left_b=1.0 / 16.0) if match is target else match
                     for match in matches]
    mixed = plane_graph.solve_component(
        readings, mixed_matches, quadrant_constraints=constraints, **gates)
    assert mixed.reason == "plane_integer_residual"
    value = graph_mod._failed_active_edges(readings, mixed_matches, mixed,
                                           **_edge_caps())
    assert value.recorded is True
    assert value.schema == "mc-photogrammetry-failed-active-edge-aliases-v1"
    assert value.basis == "derived_from_returned_best_assignment"
    assert value.active_edge_count == len(value.edges) == sum(value.axis_edge_counts)
    assert value.axis_edge_counts == (3, 3, 6)
    assert value.conflicting_endpoints == (("v0", target.left_plane),)
    assert ("v0", other.left_plane) in value.conflicting_endpoints
    active = [edge for edge in value.edges if edge[12] == 0]
    # Both shared-plane edges survive, and exactly one carries the mixed alias.
    assert [edge[2] for edge in active].count(target.left_plane) == 2
    assert [edge[2] for edge in active].count(other.left_plane) == 2
    assert sorted(edge[4] for edge in active) == [1.0 / 16.0, 1.0, 1.0]
    # Every active edge's two endpoints agree on the single world axis 0, even
    # though the frames carry different local axis labels.
    assert all(edge[12] == 0 for edge in active)
    assert {edge[3] for edge in active} == {0, 2}
    assert all(edge[4] == edge[5] and edge[9] == edge[10] for edge in value.edges)
    assert len(value.failed_assignment_sha256) == 64
    assert len(value.input_match_identity_sha256) == 64
    assert len(value.record_sha256) == 64
    record = _alias_record(value)
    assert record["status"] == "complete"
    assert record["input_match_count"] == len(mixed_matches)
    assert [list(pair) for pair in value.conflicting_endpoints] == [[
        "v0", target.left_plane]]
    assert int(record["active_edge_count"]) == len(record["edges"])
    # The same inputs replay to the exact same bounded record bytes.
    again = graph_mod._failed_active_edges(readings, mixed_matches, mixed,
                                           **_edge_caps())
    assert again.to_record() == value.to_record()
    assert again.record_sha256 == value.record_sha256
    # Restoring the coherent aliases must leave a complete, empty conflict list.
    coherent = graph_mod._failed_active_edges(readings, matches, baseline,
                                              **_edge_caps())
    assert coherent.recorded is True
    assert coherent.conflicting_endpoints == ()
    assert coherent.active_edge_count == len(coherent.edges) == 12
    assert coherent.axis_edge_counts == (3, 3, 6)
    # An edge that would conflict but is inactive in the solved assignment
    # cannot establish damage for it: this duplicate keeps the shared endpoints
    # but names the opposite relative quadrant, so the solver's own predicate
    # never activates it and it contributes no conflict.
    foreign = replace(target, quadrant_of_right=-target.quadrant_of_right,
                      left_b=1.0 / 16.0)
    active_quadrant = (baseline.quadrants["v1"] - baseline.quadrants["v0"]) % 4
    assert foreign.quadrant_of_right != active_quadrant
    assert target.quadrant_of_right == active_quadrant
    extra = graph_mod._failed_active_edges(readings, list(matches) + [foreign],
                                           baseline, **_edge_caps())
    assert extra.recorded is True
    assert extra.active_edge_count == 12
    assert extra.input_match_count == len(matches) + 1
    assert extra.conflicting_endpoints == ()
    assert all(edge[0] != len(matches) for edge in extra.edges)
def _alias_fixture():
    """A coherent solved lattice component for the v4 alias controls.
    Reuses the exact BU integer-room fixture with its two distance alternatives
    per reading, so every match the solver activates is genuinely axis-coherent
    and the returned observation counts (6, 6, 12) are the solver's own. No
    matcher, render or image workload is involved.
    """
    readings, matches, _eyes = lattice_component(decoy=True)
    constraints = (("v0", "v1", 1), ("v0", "v2", 0), ("v1", "v2", 3))
    gates = {"integer_tolerance": 0.15, "min_resultant": 0.90,
             "ambiguous_margin": 0.02}
    solution = plane_graph.solve_component(
        readings, matches, quadrant_constraints=constraints, **gates)
    assert solution.reason == ""
    assert solution.alternatives[0].axis_observations == (6, 6, 12)
    return readings, matches, solution
def _alias_alternative(observations, quadrants=(0, 1, 0)):
    return plane_graph.ComponentAlternative(
        tuple(quadrants), 1.0, (0.5, 0.5, 0.5), tuple(observations))
def _alias_solution_like(solution, observations, quadrants=(0, 1, 0)):
    """A same-identity alternative carrying chosen observation counts."""
    return plane_graph.ComponentSolution(
        frames=solution.frames, quadrants=dict(solution.quadrants),
        translations=dict(solution.translations),
        axis_residuals=dict(solution.axis_residuals),
        axis_resultants=dict(solution.axis_resultants), scale_known=False,
        blocks_per_period={}, reason="plane_integer_residual",
        alternatives=(_alias_alternative(observations, quadrants),))
def test_bv_alias_diagnostic_covers_roles_axes_multiplicity_and_refusals():
    """The v4 alias diagnostic covers every Section 6 control it must.
    Role reuse, distinct planes sharing one axis, duplicate full-edge
    multiplicity, missing and ambiguous aliases, frame-identity mismatch and the
    observation conservation invariant are all exercised against the exact
    expected record shape, without rerunning the matcher or the solver.
    """
    readings, matches, solution = _alias_fixture()
    target = next(match for match in matches
                  if match.left == "v0" and match.right == "v1"
                  and match.left_axis == 0 and match.left_plane == 0)
    assert (target.left, target.left_plane) == ("v0", 0)
    read_by_index, _index_by_object, _frames = \
        plane_graph._canonical_reading_index(readings)
    endpoint = read_by_index[("v0", target.left_plane)]
    assert endpoint.distance_for(1.0) == pytest.approx(5.0)
    assert endpoint.distance_for(1.0 / 16.0) == pytest.approx(5.0 / 16.0)
    mixed_matches = [replace(match, left_b=1.0 / 16.0) if match is target else match
                     for match in matches]
    mixed = plane_graph.solve_component(
        readings, mixed_matches,
        quadrant_constraints=(("v0", "v1", 1), ("v0", "v2", 0), ("v1", "v2", 3)),
        integer_tolerance=0.15, min_resultant=0.90, ambiguous_margin=0.02)
    assert mixed.reason == "plane_integer_residual"
    value = graph_mod._failed_active_edges(readings, mixed_matches, mixed,
                                           **_edge_caps())
    assert value.recorded is True
    assert value.schema == "mc-photogrammetry-failed-active-edge-aliases-v1"
    assert value.basis == "derived_from_returned_best_assignment"
    assert value.active_edge_count == len(value.edges) == 12
    assert value.axis_edge_counts == (3, 3, 6)
    assert value.conflicting_endpoints == (("v0", 0),)
    assert value.input_match_count == len(mixed_matches)
    assert len(value.failed_assignment_sha256) == 64
    assert len(value.input_match_identity_sha256) == 64
    assert [edge[0] for edge in value.edges] == list(range(len(mixed_matches)))
    # The mixed row keeps the shared endpoint's left/right roles and canonical b.
    target_index = next(position for position, match in enumerate(mixed_matches)
                        if match is not target)
    mixed_row = value.edges[target_index]
    assert mixed_row[1:6] == ("v0", 0, 0, 1.0 / 16.0, 1.0 / 16.0)
    assert all(edge[4] == edge[5] and edge[9] == edge[10] for edge in value.edges)
    assert all(edge[12] in (0, 1, 2) for edge in value.edges)
    record = _alias_record(value)
    assert record["status"] == "complete"
    assert record["axis_edge_counts"] == [3, 3, 6]
    assert [list(pair) for pair in value.conflicting_endpoints] == [["v0", 0]]
    assert int(record["active_edge_count"]) == len(record["edges"])
    # Restoring the coherent aliases must leave a complete, empty conflict list.
    coherent = graph_mod._failed_active_edges(readings, matches, solution,
                                              **_edge_caps())
    assert coherent.recorded is True
    assert coherent.conflicting_endpoints == ()
    assert coherent.active_edge_count == len(coherent.edges) == 12
    assert coherent.axis_edge_counts == (3, 3, 6)
    # Distinct planes sharing one world axis never merge into a conflict: the
    # same world axis is carried by two different per-frame plane indices here.
    assert {edge[2] for edge in coherent.edges} == {0, 1, 2, 3}
    assert ("v0", 3) not in coherent.conflicting_endpoints
    # The same resolved canonical b spelled through either raw label is one
    # alias, so float noise cannot manufacture a conflict. DS-R3-BX: the row must
    # also keep BOTH labels -- the exact source raw label and the resolver's
    # canonical alternative -- because one shared column made the row
    # unreplayable whenever the fixture's raw label was already canonical.
    near_matches = [replace(match, left_b=1.0 + 1e-12) if match is target else match
                    for match in matches]
    near = graph_mod._failed_active_edges(readings, near_matches, solution,
                                          **_edge_caps())
    assert near.recorded is True
    assert near.conflicting_endpoints == ()
    near_row = next(edge for edge in near.edges if edge[0] == target_index)
    assert near_row[4] == 1.0 + 1e-12 and near_row[5] == 1.0
    assert near_row[4] != near_row[5]
    other_left = next(edge for edge in near.edges
                      if (edge[1], edge[2]) == ("v0", 0) and edge[0] != target_index)
    assert other_left[4] == other_left[5] == 1.0
    shared_left = [edge for edge in near.edges if (edge[1], edge[2]) == ("v0", 0)]
    assert len(shared_left) >= 2
    assert sorted(edge[4] for edge in shared_left) == (
        [1.0] * (len(shared_left) - 1) + [1.0 + 1e-12])
    assert {edge[5] for edge in shared_left} == {1.0}
    # Both endpoint roles pool into one derived endpoint: here the same v1 plane
    # appears as an edge's right endpoint and as another edge's left endpoint, so
    # the conflict is only visible when both roles are collected.
    role_plane = target.right_plane
    role_matches = [replace(match, right_b=1.0 / 16.0) if match is target
                    else match for match in matches]
    assert any(match.left == "v1" and match.left_plane == role_plane
               for match in role_matches)
    roles = graph_mod._failed_active_edges(readings, role_matches, solution,
                                           **_edge_caps())
    assert roles.recorded is True
    assert roles.active_edge_count == 12
    assert roles.conflicting_endpoints == (("v1", role_plane),)
    # Only the returned best assignment is read: a non-best alternative with a
    # contradictory identity can never influence the record.
    nonbest = plane_graph.ComponentSolution(
        frames=solution.frames, quadrants=dict(solution.quadrants),
        translations=dict(solution.translations),
        axis_residuals=dict(solution.axis_residuals),
        axis_resultants=dict(solution.axis_resultants), scale_known=False,
        blocks_per_period={}, reason="plane_integer_residual",
        alternatives=(_alias_alternative((6, 6, 12)),
                      _alias_alternative((6, 6, 12), (0, 0, 0))))
    best_only = graph_mod._failed_active_edges(readings, matches, nonbest,
                                               **_edge_caps())
    assert best_only.recorded is True
    assert best_only.conflicting_endpoints == ()
    # Duplicate multiplicity: an exact repeated full edge keeps its own row and
    # match index, and the returned observation counts still have to conserve.
    doubled = plane_graph.ComponentSolution(
        frames=solution.frames, quadrants=dict(solution.quadrants),
        translations=dict(solution.translations),
        axis_residuals=dict(solution.axis_residuals),
        axis_resultants=dict(solution.axis_resultants), scale_known=False,
        blocks_per_period={}, reason="plane_integer_residual",
        alternatives=(_alias_alternative((8, 6, 12)),))
    duplicate = replace(target, left_b=1.0)
    repeated = graph_mod._failed_active_edges(readings, list(matches) + [duplicate],
                                              doubled, **_edge_caps())
    assert repeated.recorded is True
    assert repeated.active_edge_count == 13
    assert repeated.axis_edge_counts == (4, 3, 6)
    assert repeated.conflicting_endpoints == ()
    assert repeated.edges[-1][0] == len(matches)
    assert repeated.edges[-1][1:13] == repeated.edges[0][1:13]
    # A missing alias refuses atomically and never claims a conflict.
    missing_matches = [replace(match, left_b=3.0) if match is target else match
                       for match in matches]
    refused = graph_mod._failed_active_edges(readings, missing_matches, solution,
                                             **_edge_caps())
    assert refused.recorded is False
    assert refused.refusal_reason == "invalid_active_edge_diagnostic"
    assert refused.field == "endpoint_alias"
    assert refused.edges == () and refused.conflicting_endpoints == ()
    assert refused.failed_assignment_sha256 == ""
    assert _alias_record(refused)["status"] == "refused"
    # An ambiguous alias (two retained alternatives at one b) refuses too.
    ambiguous_readings = list(readings)
    ambiguous_readings[0] = make_reading(
        "v0", ambiguous_readings[0].axis, ambiguous_readings[0].side,
        [(1.0, 5.0, 1.0), (1.0, 5.0, 0.5)],
        share=ambiguous_readings[0].structural_share,
        period=ambiguous_readings[0].period_pixels)
    ambiguous = graph_mod._failed_active_edges(ambiguous_readings, matches,
                                               solution, **_edge_caps())
    assert ambiguous.recorded is False
    assert ambiguous.field == "endpoint_alias"
    # A best alternative whose quadrants disagree with the returned solution
    # cannot identify the evaluated assignment.
    identity = graph_mod._failed_active_edges(readings, matches, solution,
                                              **_edge_caps())
    assert identity.conflicting_endpoints == ()
    mismatched = plane_graph.ComponentSolution(
        frames=solution.frames, quadrants=dict(solution.quadrants),
        translations=dict(solution.translations),
        axis_residuals=dict(solution.axis_residuals),
        axis_resultants=dict(solution.axis_resultants), scale_known=False,
        blocks_per_period={}, reason="plane_integer_residual",
        alternatives=(_alias_alternative((6, 6, 12), (0, 0, 0)),))
    mismatch = graph_mod._failed_active_edges(readings, matches, mismatched,
                                              **_edge_caps())
    assert mismatch.recorded is False
    assert mismatch.refusal_reason == "best_assignment_unavailable"
    assert mismatch.field == "best_assignment_identity"
    # Observation conservation: 2 * axis edges must equal the returned counts.
    broken = graph_mod._failed_active_edges(
        readings, matches, _alias_solution_like(solution, (6, 6, 13)),
        **_edge_caps())
    assert broken.recorded is False
    assert broken.refusal_reason == "invalid_active_edge_diagnostic"
    assert broken.field == "observation_conservation"
    # No best alternative at all is its own typed refusal.
    absent = _alias_solution_like(solution, (6, 6, 12))
    object.__setattr__(absent, "alternatives", ())
    unavailable = graph_mod._failed_active_edges(readings, matches, absent,
                                                 **_edge_caps())
    assert unavailable.recorded is False
    assert unavailable.refusal_reason == "best_assignment_unavailable"
    assert unavailable.field == "best_assignment"
def test_bx_edge_rows_keep_the_source_raw_b_beside_the_canonical_b():
    """DS-R3-BX: every edge row replays from the source match's own raw ``b``.
    The rejected v4 snapshot wrote the resolver's canonical ``b`` into the raw
    column as well, so a row was only replayable while the fixture happened to
    use labels that were already canonical and the separate identity digest
    could not compensate. These controls force the two columns apart and require
    both: distinct raw labels that uniquely resolve to one canonical alternative
    keep distinct raw columns, one canonical column and no conflict; genuinely
    different canonical aliases keep both columns distinct and still conflict.
    """
    readings, matches, solution = _alias_fixture()
    target = next(match for match in matches
                  if match.left == "v0" and match.right == "v1"
                  and match.left_axis == 0 and match.left_plane == 0)
    target_index = next(position for position, match in enumerate(matches)
                        if match is target)
    index, _index_by_object, _frames = plane_graph._canonical_reading_index(readings)
    # Two distinct raw labels, one unique canonical alternative, no conflict.
    noisy = 1.0 + 1e-12
    near_matches = [replace(match, left_b=noisy) if match is target else match
                    for match in matches]
    near = graph_mod._failed_active_edges(readings, near_matches, solution,
                                          **_edge_caps())
    assert near.recorded is True
    assert near.conflicting_endpoints == ()
    near_row = next(edge for edge in near.edges if edge[0] == target_index)
    assert near_row[4] == noisy and near_row[5] == 1.0
    assert near_row[4] != near_row[5]
    shared_left = [edge for edge in near.edges if (edge[1], edge[2]) == ("v0", 0)]
    assert len(shared_left) >= 2
    assert sorted(edge[4] for edge in shared_left) == (
        [1.0] * (len(shared_left) - 1) + [noisy])
    assert {edge[5] for edge in shared_left} == {1.0}
    _alias_record(near)
    # A genuinely different canonical alias keeps both columns distinct and
    # still establishes the shared endpoint conflict.
    mixed_matches = [replace(match, left_b=1.0 / 16.0) if match is target else match
                     for match in matches]
    mixed_solution = plane_graph.solve_component(
        readings, mixed_matches,
        quadrant_constraints=(("v0", "v1", 1), ("v0", "v2", 0), ("v1", "v2", 3)),
        integer_tolerance=0.15, min_resultant=0.90, ambiguous_margin=0.02)
    assert mixed_solution.reason == "plane_integer_residual"
    mixed = graph_mod._failed_active_edges(readings, mixed_matches, mixed_solution,
                                           **_edge_caps())
    assert mixed.recorded is True
    assert mixed.conflicting_endpoints == (("v0", 0),)
    mixed_row = next(edge for edge in mixed.edges if edge[0] == target_index)
    assert mixed_row[4] == 1.0 / 16.0 and mixed_row[5] == 1.0 / 16.0
    mixed_shared = [edge for edge in mixed.edges if (edge[1], edge[2]) == ("v0", 0)]
    assert sorted(edge[4] for edge in mixed_shared) == (
        [1.0 / 16.0] + [1.0] * (len(mixed_shared) - 1))
    assert {edge[5] for edge in mixed_shared} == {1.0 / 16.0, 1.0}
    record = _alias_record(mixed)
    assert record["conflicting_endpoints"] == [["v0", 0]]
    # Every row in either record replays from the supplied source: the raw
    # column is the match's own label and the canonical column is exactly what
    # the existing unique-alternative resolver returns for it.
    for value, sources in ((near, near_matches), (mixed, mixed_matches)):
        assert value.edges
        for edge in value.edges:
            source = sources[edge[0]]
            assert edge[4] == float(source.left_b)
            assert edge[9] == float(source.right_b)
            left_reading = index[(edge[1], edge[2])]
            right_reading = index[(edge[6], edge[7])]
            assert edge[5] == float(
                left_reading.matching_alternatives(edge[4])[0].blocks_per_period)
            assert edge[10] == float(
                right_reading.matching_alternatives(edge[9])[0].blocks_per_period)
            assert left_reading.distance_for(edge[4]) is not None
            assert right_reading.distance_for(edge[9]) is not None
            assert len(edge) == 13
def test_bv_alias_diagnostic_caps_refuse_atomically_at_every_boundary():
    """Zero, below, equal and above each cap: no partial record ever escapes."""
    readings, matches, solution = _alias_fixture()
    sizes = _edge_caps()
    baseline = graph_mod._failed_active_edges(readings, matches, solution, **sizes)
    assert baseline.recorded is True and baseline.active_edge_count == 12
    def refused(**changes):
        caps = dict(sizes)
        caps.update(changes)
        value = graph_mod._failed_active_edges(readings, matches, solution, **caps)
        assert value.recorded is False
        assert value.refusal_reason == "budget_exhausted"
        assert value.edges == () and value.conflicting_endpoints == ()
        assert value.failed_assignment_sha256 == ""
        assert value.input_match_identity_sha256 == ""
        assert set(_alias_record(value)) == {
            "schema", "status", "reason", "field", "record_sha256"}
        return value
    assert refused(max_matches=0).field == "input_match_count"
    assert refused(max_matches=len(matches) - 1).field == "input_match_count"
    assert graph_mod._failed_active_edges(
        readings, matches, solution, **{**sizes, "max_matches": len(matches)}
    ).recorded is True
    assert refused(max_readings=0).field == "reading_count"
    assert refused(max_readings=len(readings) - 1).field == "reading_count"
    assert graph_mod._failed_active_edges(
        readings, matches, solution, **{**sizes, "max_readings": len(readings)}
    ).recorded is True
    assert refused(max_active_edges=0).field == "active_edge_count"
    assert refused(max_active_edges=11).field == "active_edge_count"
    assert graph_mod._failed_active_edges(
        readings, matches, solution, **{**sizes, "max_active_edges": 12}
    ).recorded is True
    # The per-endpoint alias budget refuses below the retained alternative count
    # and passes at it. A reading whose alternatives are individually distinct
    # still resolves one canonical alias at that bound.
    assert refused(max_endpoint_alternatives=0).field == "endpoint_alternatives"
    wide = make_reading(
        "v0", 0, -1.0,
        [(1.0, 5.0, 1.0)] + [(1.0 + (index + 1) * 1e-3, 5.0, 1.0)
                             for index in range(64)])
    wide_readings = [wide if reading.frame == "v0" and reading.axis == 0
                     else reading for reading in readings]
    capped = graph_mod._failed_active_edges(
        wide_readings, matches, solution,
        **{**sizes, "max_endpoint_alternatives": 64})
    assert capped.recorded is False
    assert capped.refusal_reason == "budget_exhausted"
    assert capped.field == "endpoint_alternatives"
    assert capped.edges == () and capped.conflicting_endpoints == ()
    assert graph_mod._failed_active_edges(
        wide_readings, matches, solution,
        **{**sizes, "max_endpoint_alternatives": 65}).recorded is True
    assert refused(max_frame_id_bytes=1).field == "frame_id"
    assert refused(max_record_bytes=1).field == "record_bytes"
    assert refused(max_record_bytes=900).field == "record_bytes"
    assert graph_mod._failed_active_edges(
        readings, matches, solution,
        **{**sizes, "max_record_bytes": graph_mod.ACTIVE_EDGE_ALIASES_MAX_RECORD_BYTES}
    ).recorded is True
    # A frame id at the UTF-8 cap is accepted; one byte over is a cap refusal,
    # never a malformed join.
    for excess, expected in ((0, True), (1, False)):
        long_frame = "f" * (
            graph_mod.ACTIVE_EDGE_ALIASES_MAX_FRAME_ID_BYTES + excess)
        long_readings = [
            make_reading(long_frame, 0, -1.0, [(1.0, 5.0, 1.0)], period=61.0),
            make_reading("v1", 0, -1.0, [(1.0, 6.0, 1.0)], period=62.0)]
        quick = plane_graph.ComponentSolution(
            frames=(long_frame, "v1"), quadrants={long_frame: 0, "v1": 0},
            translations={long_frame: (0.0, 0.0, 0.0), "v1": (0.0, 0.0, 0.0)},
            axis_residuals={0: 0.2}, axis_resultants={0: 0.1}, scale_known=False,
            blocks_per_period={}, reason="plane_integer_residual",
            alternatives=(plane_graph.ComponentAlternative(
                (0, 0), 1.0, (0.1,), (2, 0, 0)),))
        long_matches = [make_match(long_frame, "v1", 0, 0, 0, 0, 0,
                                   left_b=1.0, right_b=1.0)]
        overlong = graph_mod._failed_active_edges(long_readings, long_matches,
                                                  quick, **sizes)
        assert overlong.recorded is expected
        if expected:
            assert overlong.active_edge_count == 1
            assert overlong.conflicting_endpoints == ()
        else:
            assert overlong.refusal_reason == "budget_exhausted"
            assert overlong.field == "frame_id"
    # The byte budget covers the COMPLETE serialized record, including its own
    # record_sha256 field: exactly-at-cap is accepted and one byte under refuses.
    encoded = json.dumps(baseline.to_record(), sort_keys=True,
                         separators=(",", ":"), ensure_ascii=True,
                         allow_nan=False).encode("utf-8")
    exact = len(encoded)
    assert graph_mod._failed_active_edges(
        readings, matches, solution,
        **{**sizes, "max_record_bytes": exact}).recorded is True
    at_edge = graph_mod._failed_active_edges(
        readings, matches, solution,
        **{**sizes, "max_record_bytes": exact - 1})
    assert at_edge.recorded is False and at_edge.field == "record_bytes"
    # An inactive match is still bound by the input identity, so an oversized
    # frame identifier there must refuse as well.
    oversized = replace(
        matches[0],
        left="x" * (graph_mod.ACTIVE_EDGE_ALIASES_MAX_FRAME_ID_BYTES + 1),
        right="y" * (graph_mod.ACTIVE_EDGE_ALIASES_MAX_FRAME_ID_BYTES + 1))
    inactive = graph_mod._failed_active_edges(
        readings, list(matches) + [oversized], solution, **sizes)
    assert inactive.recorded is False
    assert inactive.refusal_reason == "budget_exhausted"
    assert inactive.field == "frame_id"
    # Malformed collections and joins are bounded refusals, never exceptions.
    no_matches = graph_mod._failed_active_edges(readings, None, solution, **sizes)
    assert no_matches.recorded is False
    assert no_matches.refusal_reason == "invalid_active_edge_diagnostic"
    assert no_matches.field == "matches"
    assert graph_mod._failed_active_edges(
        "not-a-sequence", matches, solution, **sizes).field == "readings"
    assert graph_mod._failed_active_edges(
        None, matches, solution, **sizes).field == "readings"
    assert graph_mod._failed_active_edges(
        readings, (object(),), solution, **sizes).field == "input_match_identity"
    broken_endpoint = replace(matches[0], left_plane=99)
    assert graph_mod._failed_active_edges(
        readings, (broken_endpoint,), solution, **sizes).field == "edge_endpoint"


def test_bz_over_cap_structural_inputs_never_reach_index_or_resolution(
        monkeypatch):
    """Cap checks short-circuit before indexing and before alias resolution."""
    readings, matches, solution = _alias_fixture()
    sizes = _edge_caps()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("over-cap input reached canonical indexing")

    monkeypatch.setattr(plane_graph, "_canonical_reading_index", forbidden)
    over = graph_mod._failed_active_edges(
        readings, matches, solution,
        **{**sizes, "max_matches": len(matches) - 1})
    assert (over.refusal_reason, over.field) == (
        "budget_exhausted", "input_match_count")
    over = graph_mod._failed_active_edges(
        readings, matches, solution,
        **{**sizes, "max_readings": len(readings) - 1})
    assert (over.refusal_reason, over.field) == (
        "budget_exhausted", "reading_count")
    over = graph_mod._failed_active_edges(
        readings, matches, solution, **{**sizes, "max_frame_id_bytes": 1})
    assert (over.refusal_reason, over.field) == ("budget_exhausted", "frame_id")
    over = graph_mod._failed_active_edges(readings, (object(),), solution, **sizes)
    assert (over.refusal_reason, over.field) == (
        "invalid_active_edge_diagnostic", "input_match_identity")
    monkeypatch.undo()

    resolutions = []
    original = lattice_planes.PlaneReading.matching_alternatives

    def spy(self, requested):
        resolutions.append(requested)
        return original(self, requested)

    monkeypatch.setattr(lattice_planes.PlaneReading, "matching_alternatives", spy)
    capped = graph_mod._failed_active_edges(
        readings, matches, solution,
        **{**sizes, "max_endpoint_alternatives": 0})
    assert (capped.refusal_reason, capped.field) == (
        "budget_exhausted", "endpoint_alternatives")
    assert resolutions == []
    assert capped.edges == () and capped.conflicting_endpoints == ()


def test_bz_semantic_record_validation_rejects_contradictory_records():
    """A recomputed digest must not bless a contradictory record."""
    readings, matches, _solved = _alias_fixture()
    target = next(match for match in matches
                  if match.left == "v0" and match.right == "v1"
                  and match.left_axis == 0 and match.left_plane == 0)
    mixed_matches = [replace(match, left_b=1.0 / 16.0) if match is target else match
                     for match in matches]
    mixed_solution = plane_graph.solve_component(
        readings, mixed_matches,
        quadrant_constraints=(("v0", "v1", 1), ("v0", "v2", 0), ("v1", "v2", 3)),
        integer_tolerance=0.15, min_resultant=0.90, ambiguous_margin=0.02)
    value = graph_mod._failed_active_edges(readings, mixed_matches, mixed_solution,
                                           **_edge_caps())
    record = value.to_record()
    graph_mod.validate_failed_active_edge_aliases(record)
    assert record["conflicting_endpoints"] == [["v0", 0]]

    def resealed(payload):
        payload = dict(payload)
        payload.pop("record_sha256", None)
        payload["record_sha256"] = _ordering_digest(
            graph_mod.FAILED_ACTIVE_EDGE_ALIASES_DIGEST_DOMAIN, payload)
        return payload

    # Contradictory axis counts, a dropped conflict and an unsorted or
    # duplicated conflict list are each refused even with a fresh digest.
    for change in ({"axis_edge_counts": [4, 3, 5]},
                   {"conflicting_endpoints": []},
                   {"conflicting_endpoints": [["v0", 0], ["v0", 0]]}):
        tampered = dict(record)
        tampered.update(change)
        with pytest.raises(ValueError):
            graph_mod.validate_failed_active_edge_aliases(resealed(tampered))
    # Edge arity, index bounds/order and canonical b are re-derived from rows.
    for mutate in (
            lambda rows: rows[0].pop(),
            lambda rows: rows[0].__setitem__(0, rows[1][0]),
            lambda rows: rows[0].__setitem__(5, "not-a-number")):
        rows = [list(row) for row in record["edges"]]
        mutate(rows)
        tampered = dict(record)
        tampered["edges"] = rows
        with pytest.raises(ValueError):
            graph_mod.validate_failed_active_edge_aliases(resealed(tampered))
    # Digest presence/replay is checked before any semantics.
    with pytest.raises(ValueError):
        graph_mod.validate_failed_active_edge_aliases(
            {**record, "record_sha256": "0" * 64})
    missing = dict(record)
    missing.pop("record_sha256")
    with pytest.raises(ValueError):
        graph_mod.validate_failed_active_edge_aliases(missing)
    # A refused record is accepted only with a bounded reason and field.
    refused = graph_mod._failed_active_edges(
        readings, matches, mixed_solution,
        **{**_edge_caps(), "max_active_edges": 0})
    refused_record = refused.to_record()
    assert refused_record["status"] == "refused"
    graph_mod.validate_failed_active_edge_aliases(refused_record)
    with pytest.raises(ValueError):
        graph_mod.validate_failed_active_edge_aliases(
            resealed({**refused_record, "reason": "made_up"}))
    with pytest.raises(ValueError):
        graph_mod.validate_failed_active_edge_aliases(
            resealed({**refused_record, "edges": []}))


def _reseal(payload):
    """Recompute one record's own domain-separated digest over its payload."""
    resealed = dict(payload)
    resealed.pop("record_sha256", None)
    resealed["record_sha256"] = _ordering_digest(
        graph_mod.FAILED_ACTIVE_EDGE_ALIASES_DIGEST_DOMAIN, resealed)
    return resealed


def test_ca_over_cap_alternatives_refuse_before_index_or_resolution(monkeypatch):
    """DS-R3-CA: an over-cap alternative set refuses before any traversal.

    ``_canonical_reading_index`` reaches a reading's alternatives through
    ``reading_order_key``, so the retained-alternative cap has to be preflighted
    over the input readings. The over-cap case may not touch the canonical
    index, the order key, or the unique-alias resolver, while ordinary bounded
    inputs still derive the complete record through all three.
    """
    readings, matches, solution = _alias_fixture()
    sizes = _edge_caps()
    calls = {"index": 0, "order_key": 0, "resolve": 0}

    original_index = plane_graph._canonical_reading_index
    original_key = plane_graph.reading_order_key
    original_resolve = lattice_planes.PlaneReading.matching_alternatives

    def spy_index(value):
        calls["index"] += 1
        return original_index(value)

    def spy_key(reading):
        calls["order_key"] += 1
        return original_key(reading)

    def spy_resolve(self, requested):
        calls["resolve"] += 1
        return original_resolve(self, requested)

    monkeypatch.setattr(plane_graph, "_canonical_reading_index", spy_index)
    monkeypatch.setattr(plane_graph, "reading_order_key", spy_key)
    monkeypatch.setattr(lattice_planes.PlaneReading, "matching_alternatives",
                        spy_resolve)

    # Every fixture reading retains two alternatives, so a cap of zero or one is
    # over-cap for all of them, including readings no active edge uses.
    for cap in (0, 1):
        over = graph_mod._failed_active_edges(
            readings, matches, solution,
            **{**sizes, "max_endpoint_alternatives": cap})
        assert over.recorded is False
        assert (over.refusal_reason, over.field) == (
            "budget_exhausted", "endpoint_alternatives")
        assert over.edges == () and over.conflicting_endpoints == ()
        assert over.failed_assignment_sha256 == ""
        assert over.input_match_identity_sha256 == ""
    assert calls == {"index": 0, "order_key": 0, "resolve": 0}

    # Conservative scope: an over-cap reading that no active edge uses still
    # refuses, because the cap is preflighted over the whole reading universe
    # rather than only the endpoints the failed assignment happens to activate.
    wide = make_reading(
        "v9", 0, -1.0,
        [(1.0, 5.0, 1.0)]
        + [(1.0 + (index + 1) * 1e-3, 5.0, 1.0) for index in range(64)])
    assert len(wide.alternatives) == 65
    assert all(match.left != "v9" and match.right != "v9" for match in matches)
    universe = list(readings) + [wide]
    conservative = graph_mod._failed_active_edges(
        universe, matches, solution,
        **{**sizes, "max_endpoint_alternatives": 64})
    assert conservative.recorded is False
    assert (conservative.refusal_reason, conservative.field) == (
        "budget_exhausted", "endpoint_alternatives")
    assert calls == {"index": 0, "order_key": 0, "resolve": 0}
    still_bounded = graph_mod._failed_active_edges(
        universe, matches, solution,
        **{**sizes, "max_endpoint_alternatives": 65})
    assert still_bounded.recorded is True
    assert still_bounded.active_edge_count == 12

    calls.update({"index": 0, "order_key": 0, "resolve": 0})
    ok = graph_mod._failed_active_edges(readings, matches, solution, **sizes)
    assert ok.recorded is True and ok.conflicting_endpoints == ()
    assert ok.active_edge_count == 12
    assert calls["index"] >= 1 and calls["order_key"] >= 1
    assert calls["resolve"] >= 1


class _UnsizedAlternatives:
    """An alternative collection with no length and an unbounded iterator.

    ``_retained_alternative_count`` refuses to guess a size, and the canonical
    order key would walk it; iterating it here is a hard failure so the exact
    point of traversal is observable.
    """

    def __iter__(self):
        raise AssertionError("unsized alternatives were iterated")

    def __len__(self):
        raise AssertionError("unsized alternatives were sized")


def test_cb_unknown_alternative_count_refuses_before_index_or_resolution(
        monkeypatch):
    """DS-R3-CB: an unknown retained count fails closed before any traversal.

    DS-R3-CA left ``_retained_alternative_count``'s ``None`` unhandled, so a
    malformed or unsized alternative collection skipped the preflight and let
    ``_canonical_reading_index`` reach ``reading_order_key`` with an unbounded
    collection to walk. The unknown count must instead be refused as a typed
    ``invalid_active_edge_diagnostic / endpoint_alternatives`` refusal before
    the canonical index, the order key or the alias resolver is ever entered,
    and the refusal may carry no partial edge, conflict or digest.
    """
    readings, matches, solution = _alias_fixture()
    sizes = _edge_caps()
    calls = {"index": 0, "order_key": 0, "resolve": 0}

    original_index = plane_graph._canonical_reading_index
    original_key = plane_graph.reading_order_key
    original_resolve = lattice_planes.PlaneReading.matching_alternatives

    def spy_index(value):
        calls["index"] += 1
        return original_index(value)

    def spy_key(reading):
        calls["order_key"] += 1
        return original_key(reading)

    def spy_resolve(self, requested):
        calls["resolve"] += 1
        return original_resolve(self, requested)

    monkeypatch.setattr(plane_graph, "_canonical_reading_index", spy_index)
    monkeypatch.setattr(plane_graph, "reading_order_key", spy_key)
    monkeypatch.setattr(lattice_planes.PlaneReading, "matching_alternatives",
                        spy_resolve)

    # A separate reading on a frame that is neither in the solution nor any
    # match endpoint: it can never be activated by the failed assignment, so the
    # conservative preflight has to reject it on its own before the active-edge
    # walk could decide which endpoints it activates.
    unsized = replace(readings[0], frame="unreferenced-v9",
                      alternatives=_UnsizedAlternatives())
    # Both unreferenced conditions, asserted explicitly rather than inferred.
    assert unsized.frame not in solution.frames
    assert all(match.left != unsized.frame and match.right != unsized.frame
               for match in matches)
    universe = list(readings) + [unsized]
    refused = graph_mod._failed_active_edges(
        universe, matches, solution, **sizes)
    assert refused.recorded is False
    assert (refused.refusal_reason, refused.field) == (
        "invalid_active_edge_diagnostic", "endpoint_alternatives")
    # Atomic refusal: no edge, conflict, count or digest may survive.
    assert refused.edges == () and refused.conflicting_endpoints == ()
    assert refused.basis is None
    assert refused.active_edge_count == 0
    assert refused.axis_edge_counts == (0, 0, 0)
    assert refused.failed_assignment_sha256 == ""
    assert refused.input_match_identity_sha256 == ""
    assert refused.input_match_count == 0
    assert refused.to_record()["status"] == "refused"
    graph_mod.validate_failed_active_edge_aliases(refused.to_record())
    # Nothing reached the index, the order key or the alias resolver.
    assert calls == {"index": 0, "order_key": 0, "resolve": 0}

    # Ordinary bounded inputs are unaffected and still traverse all three.
    calls.update({"index": 0, "order_key": 0, "resolve": 0})
    ok = graph_mod._failed_active_edges(readings, matches, solution, **sizes)
    assert ok.recorded is True and ok.active_edge_count == 12
    assert calls["index"] >= 1 and calls["order_key"] >= 1
    assert calls["resolve"] >= 1


def test_ca_alias_b_columns_require_json_numeric_scalars():
    """DS-R3-CA: every serialized ``b`` column must already be a JSON number.

    A valid record is resealed after one raw ``b`` is replaced by the numeric
    string ``"1.0"``; the fresh digest must not bless it, because replay binds
    the JSON scalar types a real record carries rather than a Python coercion.
    Bools, nulls and non-numbers are refused in each of the four ``b`` columns.
    """
    readings, matches, solution = _alias_fixture()
    value = graph_mod._failed_active_edges(readings, matches, solution,
                                           **_edge_caps())
    assert value.recorded is True
    record = value.to_record()
    graph_mod.validate_failed_active_edge_aliases(record)
    graph_mod.validate_failed_active_edge_aliases(_reseal(record))
    assert len(record["edges"]) == 12

    # The exact DS-R3-CA control: raw ``b`` rewritten to the string "1.0" with
    # a freshly recomputed, otherwise-valid digest.
    rows = [list(row) for row in record["edges"]]
    rows[0][4] = "1.0"
    coercible = _reseal({**record, "edges": rows})
    assert coercible["record_sha256"] != record["record_sha256"]
    with pytest.raises(ValueError):
        graph_mod.validate_failed_active_edge_aliases(coercible)
    # The same string in the canonical column refuses as well.
    rows = [list(row) for row in record["edges"]]
    rows[0][5] = "1.0"
    with pytest.raises(ValueError):
        graph_mod.validate_failed_active_edge_aliases(
            _reseal({**record, "edges": rows}))
    # Every ``b`` column rejects a numeric string, a bool, null and a container.
    for position in (4, 5, 9, 10):
        for bad in ("1.0", "2", True, False, None, [1.0]):
            rows = [list(row) for row in record["edges"]]
            rows[0][position] = bad
            with pytest.raises(ValueError):
                graph_mod.validate_failed_active_edge_aliases(
                    _reseal({**record, "edges": rows}))


def _oversized_complete_record():
    """A semantically valid complete record whose strict bytes exceed the cap."""
    cap = graph_mod.ACTIVE_EDGE_ALIASES_MAX_FRAME_ID_BYTES
    # Two distinct identifiers, each two UTF-8 bytes per character and so
    # exactly at the per-frame byte cap. JSON ``ensure_ascii`` expands each
    # character to a six-byte escape, so the full 64-edge record outgrows the
    # complete-record budget while every row stays semantically valid.
    left_frame = "\u00e9" * (cap // 2)
    right_frame = "\u00e8" * (cap // 2)
    assert len(left_frame.encode("utf-8")) == cap
    assert len(right_frame.encode("utf-8")) == cap
    left = make_reading(left_frame, 0, -1.0, [(1.0, 5.0, 1.0)], period=61.0)
    right = make_reading(right_frame, 0, -1.0, [(1.0, 6.0, 1.0)], period=62.0)
    count = graph_mod.ACTIVE_EDGE_ALIASES_MAX_ACTIVE_EDGES
    quick = plane_graph.ComponentSolution(
        frames=(left_frame, right_frame),
        quadrants={left_frame: 0, right_frame: 0},
        translations={left_frame: (0.0, 0.0, 0.0),
                      right_frame: (0.0, 0.0, 0.0)},
        axis_residuals={0: 0.2}, axis_resultants={0: 0.1}, scale_known=False,
        blocks_per_period={}, reason="plane_integer_residual",
        alternatives=(plane_graph.ComponentAlternative(
            (0, 0), 1.0, (0.1,), (2 * count, 0, 0)),))
    long_matches = [make_match(left_frame, right_frame, 0, 0, 0, 0, 0,
                               left_b=1.0, right_b=1.0) for _ in range(count)]
    return graph_mod._failed_active_edges(
        (left, right), long_matches, quick,
        **{**_edge_caps(), "max_record_bytes": 10 ** 7})


def test_ca_refusal_field_and_record_bytes_are_bounded_on_replay():
    """DS-R3-CA: refusal fields and complete-record bytes are capped on replay."""
    readings, matches, solution = _alias_fixture()
    refused = graph_mod._failed_active_edges(
        readings, matches, solution,
        **{**_edge_caps(), "max_active_edges": 0})
    refused_record = refused.to_record()
    assert refused_record["status"] == "refused"
    graph_mod.validate_failed_active_edge_aliases(refused_record)

    # An oversized producer field refuses on its byte cap; a short but unknown
    # code refuses on the bounded producer code set. Both are resealed.
    oversized_field = "x" * (graph_mod.ACTIVE_EDGE_ALIASES_MAX_FIELD_BYTES + 1)
    with pytest.raises(ValueError):
        graph_mod.validate_failed_active_edge_aliases(
            _reseal({**refused_record, "field": oversized_field}))
    with pytest.raises(ValueError):
        graph_mod.validate_failed_active_edge_aliases(
            _reseal({**refused_record, "field": "not_a_producer_field"}))
    # Real producer codes still replay, including the alternative-budget code.
    assert "endpoint_alternatives" in graph_mod._ACTIVE_EDGE_ALIASES_FIELD_CODES
    capped = graph_mod._failed_active_edges(
        readings, matches, solution,
        **{**_edge_caps(), "max_endpoint_alternatives": 0})
    assert capped.field in graph_mod._ACTIVE_EDGE_ALIASES_FIELD_CODES
    graph_mod.validate_failed_active_edge_aliases(capped.to_record())

    # An otherwise-valid complete record above the fixed byte budget refuses
    # even though its own digest was just recomputed over the exact payload.
    big = _oversized_complete_record()
    assert big.recorded is True
    assert big.active_edge_count == graph_mod.ACTIVE_EDGE_ALIASES_MAX_ACTIVE_EDGES
    big_record = big.to_record()
    encoded = json.dumps(big_record, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    assert len(encoded) > graph_mod.ACTIVE_EDGE_ALIASES_MAX_RECORD_BYTES
    resealed = _reseal(big_record)
    assert resealed["record_sha256"] == big_record["record_sha256"]
    with pytest.raises(ValueError):
        graph_mod.validate_failed_active_edge_aliases(resealed)


# ---------------------------------------------------------------------------
# DS-R5: optional failure-only boundary-saturation attestation
# ---------------------------------------------------------------------------


def _saturation_trace_row(left, right, quadrant, shift, *, admitted=True):
    """One synthetic matcher trace row, mirroring the real trace's shape."""
    return plane_graph.PlanePairTrace(
        left, right, 0, 0, quadrant, admitted, "", shift, 1.0, 4,
        shift, 1.0, 4, psr=5.0, score=0.5, overlap_cells=4, valid_cells=8,
        overlap_fraction=0.5, cell_shift_rc=shift, left_structural_share=0.6,
        right_structural_share=0.6)


def _saturation_match(left, right, quadrant, shift, *, left_b=1.0, right_b=1.0):
    return plane_graph.PlaneMatch(
        left, right, 0, 0, left_b, right_b, quadrant, 5.0, 1.0, 0.1, 16,
        None, 0.9, 0, 0, shift)


def _saturation(matches, rows, max_shift_cells=32):
    return graph_mod.boundary_saturation_attestation(
        matches, plane_graph.PlaneGraphTrace(list(rows)), max_shift_cells)


def _kept_saturation_record(value):
    """Strict canonical encoding plus the domain-separated digest replay."""
    record = value.to_record()
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False)
    assert "NaN" not in encoded and "Infinity" not in encoded
    payload = dict(record)
    digest = payload.pop("record_sha256")
    recomputed = hashlib.sha256(json.dumps(
        {"domain": graph_mod.BOUNDARY_SATURATION_DIGEST_DOMAIN,
         "schema": value.schema, "payload": payload},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    assert digest == recomputed
    assert digest == graph_mod.boundary_saturation_record_digest(value)
    graph_mod.validate_boundary_saturation_attestation(record)
    return record


def _reseal_saturation(record):
    """Recompute the attestation's own digest over a tampered payload."""
    resealed = dict(record)
    resealed.pop("record_sha256", None)
    resealed["record_sha256"] = hashlib.sha256(json.dumps(
        {"domain": graph_mod.BOUNDARY_SATURATION_DIGEST_DOMAIN,
         "schema": graph_mod.BOUNDARY_SATURATION_SCHEMA, "payload": resealed},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    return resealed


def test_ds_r5_interior_shifts_never_saturate_the_boundary():
    """Shifts strictly inside the search domain produce exact zero counts."""
    shifts = ((0, 0), (1, -1), (31, -31), (-31, 31), (0, 17))
    matches = tuple(_saturation_match("a", "b", quadrant, shift)
                    for quadrant, shift in enumerate(shifts))
    rows = tuple(_saturation_trace_row("a", "b", quadrant, shift)
                 for quadrant, shift in enumerate(shifts))
    value = _saturation(matches, rows)
    assert value.recorded is True
    assert value.basis == "derived_from_matcher_diagnostic_trace"
    assert value.algorithm_id == plane_graph.PLANE_CORRELATION_ALGORITHM
    assert value.max_shift_cells == 32
    assert value.input_match_count == 5
    assert value.admitted_match_count == 5
    assert value.same_axis_kept_match_count == 5
    assert value.admitted_boundary_counts == (0, 0, 0)
    assert value.kept_boundary_counts == (0, 0, 0)
    record = _kept_saturation_record(value)
    assert record["status"] == "complete"
    assert record["admitted_boundary_counts"] == [0, 0, 0]
    assert record["kept_boundary_counts"] == [0, 0, 0]


def test_ds_r5_boundary_counts_are_exact_and_per_axis():
    """Both signs of the bound count once per axis; a corner counts twice."""
    matches = (
        _saturation_match("a", "b", 0, (0, 0)),
        _saturation_match("d", "e", 0, (32, 0)),
        _saturation_match("f", "g", 1, (-32, 0), left_b=2.0, right_b=2.0),
        _saturation_match("h", "i", 2, (0, 32), left_b=3.0, right_b=3.0),
        _saturation_match("j", "k", 3, (0, -32), left_b=4.0, right_b=4.0),
        _saturation_match("l", "m", 0, (32, -32), left_b=5.0, right_b=5.0),
        _saturation_match("n", "o", 0, (7, 7), left_b=6.0, right_b=6.0),
    )
    rows = (
        # An interior row on the same trace must stay uncounted.
        _saturation_trace_row("a", "b", 0, (5, 5)),
        # A boundary row the matcher never admitted is not admitted damage.
        _saturation_trace_row("r", "s", 0, (0, 32), admitted=False),
        _saturation_trace_row("d", "e", 0, (32, 0)),
        _saturation_trace_row("f", "g", 1, (-32, 0)),
        _saturation_trace_row("h", "i", 2, (0, 32)),
        _saturation_trace_row("j", "k", 3, (0, -32)),
        _saturation_trace_row("l", "m", 0, (32, -32)),
        _saturation_trace_row("n", "o", 0, (7, 7)),
    )
    value = _saturation(matches, rows)
    assert value.recorded is True
    # Row bound: +32, -32 and the +32 share of the corner -> 3.
    # Column bound: +32, -32 and the -32 share of the corner -> 3.
    assert value.admitted_boundary_counts == (3, 3, 0)
    assert value.kept_boundary_counts == (3, 3, 0)
    assert value.admitted_match_count == 7
    assert value.same_axis_kept_match_count == 7
    assert value.input_match_count == 7
    record = _kept_saturation_record(value)
    assert record["admitted_boundary_counts"] == [3, 3, 0]
    assert record["kept_boundary_counts"] == [3, 3, 0]
    # A zero-height row of all-interior matches stays exactly zero.
    assert _saturation((), ()).kept_boundary_counts == (0, 0, 0)


def test_ds_r5_decoupled_admitted_and_kept_populations_record_exactly():
    """A real-shape 1,495 admitted / 54 same-axis-kept trace stays complete.

    The admitted population is the matcher's full diagnostic trace; the
    solver-kept population is the post-filter match tuple handed to
    ``solve_component``. They are different sizes, so the record must enforce
    ``kept <= admitted`` -- not the old ``admitted <= input_match_count`` -- and
    still keep every exact count.
    """
    kept = 54
    admitted = 1_495
    max_shift_cells = 32
    matches = tuple(
        _saturation_match("k%03d" % index, "k%03d" % (index + 1),
                          index % 4,
                          (32 if index < 7 else index % max_shift_cells, 0),
                          left_b=1.0 + index, right_b=2.0 + index)
        for index in range(kept))
    rows = [
        _saturation_trace_row("k%03d" % index, "k%03d" % (index + 1),
                              index % 4,
                              (32 if index < 7 else index % max_shift_cells, 0))
        for index in range(kept)
    ]
    # The remaining admitted candidates are interior and must not add a count.
    rows.extend(
        _saturation_trace_row("a%04d" % index, "a%04d" % (index + 1),
                              index % 4,
                              (index % (max_shift_cells - 1), index % 3))
        for index in range(admitted - kept))
    # A non-admitted boundary row can never inflate the admitted count.
    rows.append(_saturation_trace_row("x", "y", 0, (32, 0), admitted=False))
    assert len(rows) <= graph_mod.BOUNDARY_SATURATION_MAX_TRACE_ROWS
    value = _saturation(matches, rows)
    assert value.recorded is True
    assert value.input_match_count == kept
    assert value.same_axis_kept_match_count == kept
    assert value.admitted_match_count == admitted
    # Seven kept matches pin the row bound; none of the other admitted rows and
    # none of the non-admitted rows sit on it.
    assert value.kept_boundary_counts == (7, 0, 0)
    assert value.admitted_boundary_counts == (7, 0, 0)
    record = _kept_saturation_record(value)
    assert record["status"] == "complete"
    assert record["input_match_count"] == kept
    assert record["same_axis_kept_match_count"] == kept
    assert record["admitted_match_count"] == admitted
    assert record["admitted_boundary_counts"] == [7, 0, 0]
    assert record["kept_boundary_counts"] == [7, 0, 0]
    graph_mod.validate_boundary_saturation_attestation(record)


def test_ds_r5_each_match_is_counted_once_even_when_aliases_collapse():
    """Two raw labels resolving to one canonical alias stay two matches.

    The shift is a match property, not an alias property: the attestation must
    count every selected match exactly once and never merge distinct matches
    that happen to share a canonical ``b``.
    """
    target = _saturation_match("v0", "v1", 0, (32, 0), left_b=1.0)
    noisy = replace(target, left_b=1.0 + 1e-12)
    rows = (_saturation_trace_row("v0", "v1", 0, (32, 0)),
            _saturation_trace_row("v0", "v1", 0, (32, 0)))
    value = _saturation((target, noisy), rows)
    assert value.recorded is True
    assert value.input_match_count == 2
    assert value.admitted_match_count == 2
    assert value.same_axis_kept_match_count == 2
    assert value.kept_boundary_counts == (2, 0, 0)
    assert value.admitted_boundary_counts == (2, 0, 0)
    # The identity digest binds both distinct raw labels in order.
    expected = hashlib.sha256(json.dumps({
        "domain": graph_mod.BOUNDARY_SATURATION_INPUT_MATCH_IDENTITY_DOMAIN,
        "schema": graph_mod.BOUNDARY_SATURATION_SCHEMA,
        "payload": [
            ["v0", 0, 0, "v1", 0, 0, 0, 1.0, 1.0],
            ["v0", 0, 0, "v1", 0, 0, 0, 1.0 + 1e-12, 1.0]]},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    assert value.input_match_identity_sha256 == expected
    assert value.input_match_identity_sha256 != hashlib.sha256(
        b"boundary").hexdigest()


def test_ds_r5_absent_sink_is_silent_and_payload_digest_parity_holds(monkeypatch,inputs):
    source,calls=_current_failure_source(monkeypatch);attest=[];original=graph_mod._boundary_saturation_attestation
    def spy(*args,**kwargs):attest.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(graph_mod,"_boundary_saturation_attestation",spy)
    plain=graph_mod.derive_image_derived_graph(**source)
    assert calls=={"match":1,"evaluate":1} and attest==[]
    result,attribution=subject.derive_image_derived_graph_attribution(**source)
    assert calls=={"match":2,"evaluate":2} and attest==[1]
    assert result==plain and result.result_sha256==plain.result_sha256
    saturation=attribution.solver_failure.boundary_saturation
    assert saturation.recorded and saturation.input_match_count==3
    assert saturation.kept_boundary_counts==(0,0,0) and saturation.admitted_boundary_counts==(0,0,0)
    graph_mod.validate_boundary_saturation_attestation(saturation.to_record());_strict_attribution_digest(attribution)

def test_ds_r5_failure_only_sink_is_silent_on_success_and_pre_solver(
        monkeypatch, inputs):
    """Success, pre-solver abstention and the non-finite guard never count."""
    solution = _solver_solution(
        "plane_integer_residual",
        resultants={0: 0.5, 1: 0.95, 2: 0.95},
        residuals={0: 0.1, 1: 0.1, 2: 0.1},
        alternatives=(_full_alternative(),))
    _result_value, attribution = _diagnosed(monkeypatch, inputs, solution)
    # The monkeypatched matcher published no trace and the injected match
    # carries no cell shift, so the failure-only attestation refuses atomically
    # rather than inventing a count. A shift-less match is exactly the case the
    # record may not under-report.
    saturation = attribution.solver_failure.boundary_saturation
    assert saturation is not None and saturation.recorded is False
    assert saturation.basis is None
    assert (saturation.refusal_reason, saturation.field) == (
        graph_mod.BOUNDARY_SATURATION_INVALID, "shift_components")
    assert saturation.admitted_boundary_counts == (0, 0, 0)
    assert saturation.kept_boundary_counts == (0, 0, 0)
    assert saturation.to_record()["status"] == "refused"
    graph_mod.validate_boundary_saturation_attestation(saturation.to_record())

    # With a real matcher trace and a shifted match, the same failure seam does
    # produce a complete record; the refusal above was the missing provenance,
    # not the failure path itself.
    traced = _saturation((_saturation_match("f0", "f1", 0, (32, 0)),),
                         (_saturation_trace_row("f0", "f1", 0, (32, 0)),))
    assert traced.recorded is True
    assert traced.kept_boundary_counts == (1, 0, 0)

    solved = _solver_solution(
        "", resultants={0: 0.95, 1: 0.95, 2: 0.95},
        residuals={0: 0.1, 1: 0.1, 2: 0.1},
        alternatives=(_full_alternative(),))
    result, attribution = _diagnosed(monkeypatch, inputs, solved)
    assert result.status == "assembled"
    assert attribution.solver_failure is None
    assert "solver_failure" not in attribution.to_record()

    # Pre-solver abstention: no matcher output at all is still silent.
    monkeypatch.setattr(plane_graph, "match_plane_graph", lambda *_a, **_k: ())
    monkeypatch.setattr(plane_graph, "solve_component",
                        lambda *_a, **_k: pytest.fail("solver ran without matches"))
    result, attribution = subject.derive_image_derived_graph_attribution(**inputs)
    assert result.reason == "no_matches"
    assert attribution.solver_failure is None

    # The non-finite solver guard abstains after the solver, but it is not a
    # solver *failure*, so the failure-only sink still stays silent.
    monkeypatch.setattr(
        plane_graph, "match_plane_graph",
        lambda *_a, **_k: (_saturation_match("f0", "f1", 0, (32, 0)),))
    monkeypatch.setattr(
        plane_graph, "solve_component",
        lambda *_a, **_k: _solver_solution(
            "", resultants={0: float("inf")}, residuals={0: 0.1}))
    result, attribution = subject.derive_image_derived_graph_attribution(**inputs)
    assert result.reason == graph_mod.NON_FINITE_SOLVER_OUTPUT
    assert attribution.solver_failure is None


def test_ds_r5_attestation_is_deterministic_and_calls_the_matcher_once(monkeypatch,inputs):
    source,calls=_current_failure_source(monkeypatch)
    first_result,first=subject.derive_image_derived_graph_attribution(**source)
    second_result,second=subject.derive_image_derived_graph_attribution(**source)
    assert calls=={"match":2,"evaluate":2} and first_result==second_result
    first_record=first.solver_failure.boundary_saturation.to_record();second_record=second.solver_failure.boundary_saturation.to_record()
    assert first_record==second_record and first_record["status"]=="complete"
    assert first_record["input_match_count"]==3
    assert first.attribution_sha256==second.attribution_sha256
    graph_mod.validate_boundary_saturation_attestation(first_record)


def test_ds_r5_malformed_and_over_cap_inputs_refuse_atomically():
    """Every malformed or over-cap input yields one bounded refusal, never a count."""
    good_rows = (_saturation_trace_row("f0", "f1", 0, (32, 0)),)
    good_matches = (_saturation_match("f0", "f1", 0, (32, 0)),)
    baseline = _saturation(good_matches, good_rows)
    assert baseline.recorded is True
    assert baseline.kept_boundary_counts == (1, 0, 0)

    def refused(*, matches, rows, max_shift_cells=32):
        value = _saturation(matches, rows, max_shift_cells)
        assert value.recorded is False
        assert value.basis is None
        assert value.algorithm_id == "" and value.max_shift_cells == 0
        assert value.input_match_count == 0
        assert value.input_match_identity_sha256 == ""
        assert value.admitted_match_count == 0
        assert value.same_axis_kept_match_count == 0
        assert value.admitted_boundary_counts == (0, 0, 0)
        assert value.kept_boundary_counts == (0, 0, 0)
        record = value.to_record()
        assert set(record) == {
            "schema", "status", "reason", "field", "record_sha256"}
        graph_mod.validate_boundary_saturation_attestation(record)
        return value

    assert refused(matches="not-a-sequence", rows=good_rows).field == "matches"
    assert refused(matches=None, rows=good_rows).field == "matches"
    assert refused(matches=good_matches, rows=good_rows,
                   max_shift_cells=0).field == "max_shift_cells"
    assert refused(matches=good_matches, rows=good_rows,
                   max_shift_cells=True).field == "max_shift_cells"
    assert refused(matches=good_matches, rows=good_rows,
                   max_shift_cells=(graph_mod
                                    .BOUNDARY_SATURATION_MAX_SHIFT_CELLS_CAP + 1)
                   ).field == "max_shift_cells"
    no_trace = graph_mod.boundary_saturation_attestation(
        good_matches, None, 32)
    assert (no_trace.refusal_reason, no_trace.field) == (
        graph_mod.BOUNDARY_SATURATION_UNAVAILABLE, "trace")
    assert graph_mod.boundary_saturation_attestation(
        good_matches, object(), 32).field == "pairs"
    assert refused(matches=good_matches, rows=(object(),)).field == "admitted"
    assert refused(matches=(object(),), rows=good_rows).field == (
        "input_match_identity")
    # A malformed shift (bool, float, wrong arity or non-tuple) is refused
    # rather than coerced, on either the admitted or the kept side. The
    # constructor itself already rejects a malformed shift, so the raw payload
    # is forced into the slot to prove the attestation re-validates it.
    assert refused(
        matches=good_matches,
        rows=(replace(good_rows[0], cell_shift_rc=(True, 0)),)).field == "admitted"
    malformed = object.__new__(plane_graph.PlaneMatch)
    for field in dataclasses.fields(good_matches[0]):
        object.__setattr__(malformed, field.name,
                           getattr(good_matches[0], field.name))
    object.__setattr__(malformed, "cell_shift_rc", (32.0, 0))
    assert refused(matches=(malformed,), rows=good_rows).field == "shift_components"
    object.__setattr__(malformed, "cell_shift_rc", (32,))
    assert refused(matches=(malformed,), rows=good_rows).field == "shift_components"
    object.__setattr__(malformed, "cell_shift_rc", None)
    assert refused(matches=(malformed,), rows=good_rows).field == "shift_components"
    # The match-count cap refuses before any counting happens.
    over = tuple(_saturation_match("f0", "f1", 0, (32, 0))
                 for _ in range(
                     graph_mod.ACTIVE_EDGE_ALIASES_MAX_MATCHES + 1))
    capped = _saturation(over, good_rows)
    assert capped.recorded is False
    assert (capped.refusal_reason, capped.field) == (
        "budget_exhausted", "matches")

    # DS-R5B: the explicit trace-row/admitted cap refuses on the raw row count
    # before the trace is iterated, so an unbounded admitted population can
    # never be partially counted. The row collection raises if it is ever
    # walked, so a red test here proves the pre-iteration refusal is real.
    class _NeverIteratedRows(list):
        def __iter__(self):
            raise AssertionError("over-cap trace rows were iterated")

    over_rows = _NeverIteratedRows()
    list.extend(over_rows, (_saturation_trace_row("f0", "f1", 0, (0, 0))
                            for _ in range(graph_mod
                                           .BOUNDARY_SATURATION_MAX_TRACE_ROWS + 1)))
    over_trace = plane_graph.PlaneGraphTrace([])
    over_trace.pairs = over_rows
    capped_trace = graph_mod.boundary_saturation_attestation(
        good_matches, over_trace, 32)
    assert capped_trace.recorded is False
    assert (capped_trace.refusal_reason, capped_trace.field) == (
        "budget_exhausted", "admitted")
    graph_mod.validate_boundary_saturation_attestation(
        capped_trace.to_record())
    # A trace exactly at the cap is still admitted, so the refusal is strictly
    # ``> cap`` and the 7,476-candidate frozen fixture is safely below it.
    at_cap = _saturation(
        good_matches,
        tuple(_saturation_trace_row("f0", "f1", 0, (0, 0))
              for _ in range(graph_mod.BOUNDARY_SATURATION_MAX_TRACE_ROWS)))
    assert at_cap.recorded is True
    assert at_cap.admitted_match_count == (
        graph_mod.BOUNDARY_SATURATION_MAX_TRACE_ROWS)


def test_ds_r5_semantic_replay_rejects_contradictory_records():
    """A freshly recomputed digest must not bless a contradictory attestation."""
    matches = (_saturation_match("f0", "f1", 0, (32, 0)),)
    rows = (_saturation_trace_row("f0", "f1", 0, (32, 0)),)
    value = _saturation(matches, rows)
    record = _kept_saturation_record(value)
    assert record["kept_boundary_counts"] == [1, 0, 0]
    assert record["max_shift_cells"] == 32

    # Digest presence and replay are checked before any semantics.
    with pytest.raises(ValueError):
        graph_mod.validate_boundary_saturation_attestation(
            {**record, "record_sha256": "0" * 64})
    missing = dict(record)
    missing.pop("record_sha256")
    with pytest.raises(ValueError):
        graph_mod.validate_boundary_saturation_attestation(missing)

    # Count nesting, per-axis component bounds, radius, algorithm and identity
    # format are each independently re-derived from the payload. The record
    # deliberately retains only counts plus the source identity digest, so a
    # resealed *redistribution* among axes is not distinguishable from a real
    # reading -- only an internally contradictory record is refused.
    for change in ({"admitted_match_count": 0},
                   {"input_match_count": 0},
                   {"same_axis_kept_match_count": 0},
                   {"kept_boundary_counts": [2, 0, 0]},
                   {"admitted_boundary_counts": [3, 0, 0]},
                   {"admitted_boundary_counts": [-1, 1, 0]},
                   {"kept_boundary_counts": [1, 1]},
                   {"max_shift_cells": 0},
                   {"max_shift_cells": graph_mod
                    .BOUNDARY_SATURATION_MAX_SHIFT_CELLS_CAP + 1},
                   {"algorithm_id": "other"},
                   {"basis": "made_up"},
                   {"input_match_identity_sha256": "not-hex"},
                   {"input_match_identity_sha256": "A" * 64}):
        with pytest.raises(ValueError):
            graph_mod.validate_boundary_saturation_attestation(
                _reseal_saturation({**record, **change}))
    # DS-R5B: an admitted population larger than the solver-input identity count
    # is no longer a contradiction -- it is the normal unequal-population shape
    # as long as the kept set nests inside it.
    graph_mod.validate_boundary_saturation_attestation(
        _reseal_saturation({**record, "admitted_match_count": 2}))
    # A genuine, internally consistent record still replays after resealing.
    graph_mod.validate_boundary_saturation_attestation(_reseal_saturation(record))

    # A refusal is accepted only with a bounded reason and a known field code.
    refused = _saturation("not-a-sequence", ())
    refused_record = refused.to_record()
    graph_mod.validate_boundary_saturation_attestation(refused_record)
    for change in ({"reason": "made_up"},
                   {"field": "not_a_producer_field"},
                   {"extra": 1}):
        with pytest.raises(ValueError):
            graph_mod.validate_boundary_saturation_attestation(
                _reseal_saturation({**refused_record, **change}))
    # A refused payload may not smuggle a complete payload or vice versa.
    with pytest.raises(ValueError):
        graph_mod.validate_boundary_saturation_attestation(
            _reseal_saturation({**refused_record,
                                "admitted_boundary_counts": [0, 0, 0]}))


def test_observer_receives_the_exact_snapshot_and_keeps_output_parity(monkeypatch,inputs):
    source,calls=_current_failure_source(monkeypatch);snapshots=[];original=graph_mod._solver_failure_snapshot
    def spy(*args,**kwargs):
        snapshot=original(*args,**kwargs);snapshots.append(snapshot);return snapshot
    monkeypatch.setattr(graph_mod,"_solver_failure_snapshot",spy)
    plain,control=subject.derive_image_derived_graph_attribution(**source)
    seen=[];result,attribution=subject.derive_image_derived_graph_attribution(**source,solver_failure_observer=seen.append)
    assert len(seen)==1 and seen[0] is snapshots[-1]
    assert seen[0].reason=="plane_integer_residual" and dict(seen[0].axis_resultants)=={0:.5,1:1.,2:1.}
    assert result==plain and attribution==control and result.result_sha256==plain.result_sha256
    assert subject.solver_failure_snapshot_record(seen[0])["reason"]==attribution.to_record()["solver_failure"]["reason"]


def test_observer_is_silent_when_the_solver_never_abstains(monkeypatch, inputs):
    solved = _solver_solution(
        "", resultants={0: 0.95, 1: 0.95, 2: 0.95},
        residuals={0: 0.1, 1: 0.1, 2: 0.1},
        alternatives=(_full_alternative(),))
    seen = []
    result, attribution = _diagnosed(
        monkeypatch, inputs, solved, solver_failure_observer=seen.append)
    assert result.status == "assembled" and seen == []
    assert attribution.solver_failure is None


def test_raw_snapshot_record_keeps_partial_axes_when_projection_refuses():
    snapshot = graph_mod.SolverFailureSnapshot(
        "plane_integer_residual", ("f0", "f1"), (0, 0),
        ((0, 0.5), (1, 0.95)), ((0, 0.1), (1, 0.12)), (0, 0), (2, 3, 4))
    projected = subject.solver_failure_diagnostic(snapshot, 0.90, 0.15, 0.02)
    # The bounded projection refuses the partial axis map, so the raw record is
    # the only place the actual per-axis metrics survive.
    assert projected.recorded is False
    assert projected.to_record() == {
        "solver_id": subject.SOLVER_ID, "refused": True,
        "reason": subject.INVALID_DIAGNOSTIC, "field": "plane_integer_residual"}
    raw = subject.solver_failure_snapshot_record(snapshot)
    assert raw["schema"] == subject.SOLVER_FAILURE_OBSERVATION_SCHEMA
    assert raw["solver_id"] == subject.SOLVER_ID
    assert raw["reason"] == "plane_integer_residual"
    assert raw["frames"] == ["f0", "f1"] and raw["quadrants"] == [0, 0]
    assert raw["axis_resultants"] == [
        {"axis": 0, "value": 0.5}, {"axis": 1, "value": 0.95}]
    assert raw["axis_residuals"] == [
        {"axis": 0, "value": 0.1}, {"axis": 1, "value": 0.12}]
    assert raw["best_alternative_quadrants"] == [0, 0]
    assert raw["best_alternative_axis_observations"] == [2, 3, 4]
    assert raw["invalid_field"] is None
    json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False)


def test_raw_snapshot_record_tags_nonfinite_values_without_dropping_them():
    snapshot = graph_mod.SolverFailureSnapshot(
        "plane_integer_residual", (), (),
        ((0, float("nan")), (1, float("inf"))), ((0, 0.1), (1, 0.2)), None, None)
    raw = subject.solver_failure_snapshot_record(snapshot)
    assert raw["axis_resultants"] == [
        {"axis": 0, "value": {"unavailable": subject.INVALID_DIAGNOSTIC,
                              "type": "float"}},
        {"axis": 1, "value": {"unavailable": subject.INVALID_DIAGNOSTIC,
                              "type": "float"}}]
    assert raw["best_alternative_quadrants"] is None
    assert raw["best_alternative_axis_observations"] is None
    encoded = json.dumps(raw, sort_keys=True, separators=(",", ":"),
                         allow_nan=False)
    assert "NaN" not in encoded and "Infinity" not in encoded




def _diagnosed_fixed(monkeypatch, inputs, solution, **overrides):
    """Real matcher and joint selection; inject only selected-Q numeric values."""
    from test_image_derived_graph import _current_sources
    source=_current_sources(monkeypatch);original=plane_graph._evaluate_assignment
    def evaluate(*args,**kwargs):
        cost,alternative,detail=original(*args,**kwargs);detail=dict(detail)
        detail["resultants"]=solution.axis_resultants;detail["residuals"]=solution.axis_residuals
        if solution.alternatives:alternative=solution.alternatives[0]
        return cost,alternative,detail
    monkeypatch.setattr(plane_graph,"_evaluate_assignment",evaluate)
    plain=graph_mod.derive_image_derived_graph(**source)
    result,attribution=subject.derive_image_derived_graph_attribution(**source,**overrides)
    assert result==plain and result.result_sha256==plain.result_sha256
    assert result.geometry_evaluations==1 and result.reason==solution.reason
    return result,attribution


def _project_detached_diagnostic(monkeypatch, inputs, solution, **overrides):
    """Malformed/historical snapshot projection, never a production evaluator.

    First verify genuine producer7 ordinary/observed output parity. Then test
    the existing pure bounded snapshot projection with deliberately detached
    synthetic fields. No malformed component is installed in a live graph.
    """
    from test_image_derived_graph import _current_sources,_low_resultant
    source=_current_sources(monkeypatch)
    original=plane_graph._evaluate_assignment
    def evaluate(*args,**kwargs):
        cost,alternative,detail=_low_resultant(original(*args,**kwargs))
        if solution.reason=="insufficient_axis_coverage":
            detail=dict(detail);detail["resultants"]={0:.5};detail["residuals"]={0:.1}
        return cost,alternative,detail
    monkeypatch.setattr(plane_graph,"_evaluate_assignment",evaluate)
    plain=graph_mod.derive_image_derived_graph(**source)
    result,attribution=subject.derive_image_derived_graph_attribution(**source)
    assert result==plain and result.result_sha256==plain.result_sha256 and result.geometry_evaluations==1
    raw={"max_shift_cells":32,"integer_tolerance":.15,"min_resultant":.90,"ambiguous_margin":.02,
         "min_overlap_cells":plane_graph.MIN_OVERLAP_CELLS,"min_structural_share":plane_graph.MIN_STRUCTURAL_SHARE}
    raw.update(overrides)
    if overrides:
        refused=graph_mod.derive_image_derived_graph(**source,**overrides)
        from mc_photogrammetry.reconstruction import plane_placement
        assert isinstance(refused,plane_placement.PlacementRefusal) and refused.reason=="invalid_input"
    settings,invalid=subject._effective_settings(raw)
    projected=subject.solver_failure_diagnostic(graph_mod._solver_failure_snapshot(solution),
                    raw["min_resultant"],raw["integer_tolerance"],raw["ambiguous_margin"])
    detached=dataclasses.replace(attribution,effective_settings=settings,invalid_settings=invalid,
                                solver_failure=projected,attribution_sha256="")
    return result,detached


def _current_failure_source(monkeypatch):
    from test_image_derived_graph import _current_sources,_low_resultant
    source=_current_sources(monkeypatch);actual_match=plane_graph.match_plane_graph;actual_evaluate=plane_graph._evaluate_assignment
    calls={"match":0,"evaluate":0}
    def matcher(*args,**kwargs):calls["match"]+=1;return actual_match(*args,**kwargs)
    def evaluate(*args,**kwargs):calls["evaluate"]+=1;return _low_resultant(actual_evaluate(*args,**kwargs))
    monkeypatch.setattr(plane_graph,"match_plane_graph",matcher)
    monkeypatch.setattr(plane_graph,"_evaluate_assignment",evaluate)
    return source,calls
