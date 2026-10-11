"""Focused controls for the output-neutral axis-coverage diagnostic."""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_image_derived_entry import _valid_result  # noqa: E402
from mc_photogrammetry.pipeline import image_derived_production as production  # noqa: E402
from mc_photogrammetry.reconstruction import plane_graph  # noqa: E402
from test_plane_graph import lattice_component, make_match, make_reading  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / \
    "diagnose_image_derived_axis_coverage.py"
SPEC = importlib.util.spec_from_file_location("axis_diagnostic", SCRIPT)
subject = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(subject)


@pytest.fixture()
def normal_result(monkeypatch, tmp_path):
    entry = _valid_result(monkeypatch, tmp_path)
    monkeypatch.setattr(production.entry_mod, "build_image_derived_entry",
                        lambda *_args, **_kwargs: entry)
    return production.derive_image_output_from_paths(
        [tmp_path], project_path=tmp_path / "normal.mcscan")


def test_repeat_is_canonical_json_deterministic(normal_result):
    first = subject.diagnose_production_result(normal_result)
    second = subject.diagnose_production_result(normal_result)
    assert first == second
    assert subject.canonical_json(first) == subject.canonical_json(second)


def test_probe_is_output_neutral_and_records_exact_sources(normal_result):
    before = (normal_result.reason, normal_result.detail,
              normal_result.result_sha256,
              normal_result.entry.graph.graph_sha256)
    result = subject.diagnose_production_result(normal_result)
    after = (normal_result.reason, normal_result.detail,
             normal_result.result_sha256,
             normal_result.entry.graph.graph_sha256)
    assert after == before
    assert result["normal_mode"]["unchanged"] is True
    assert result["normal_mode"]["reason"] == normal_result.reason
    assert result["normal_mode"]["detail"] == normal_result.detail
    assert result["normal_mode"]["graph_sha256"] == before[3]


def test_unavailable_normal_result_remains_atomic(normal_result):
    if normal_result.status != "geometry_unavailable":
        pytest.skip("fixture advanced beyond the unavailable boundary")
    geometry = normal_result.geometry
    before = None if geometry is None else (
        geometry.status, geometry.reason, geometry.placement_records,
        geometry.universe, geometry.supports)
    diagnostic = subject.diagnose_production_result(normal_result)
    after = None if geometry is None else (
        geometry.status, geometry.reason, geometry.placement_records,
        geometry.universe, geometry.supports)
    assert after == before
    if geometry is not None and geometry.status == "unavailable":
        assert geometry.placement_records == () and geometry.universe is None
        assert geometry.supports == ()
    assert diagnostic["fired_gate"]


def test_gate_mapping_uses_exact_solver_fields():
    readings, matches, _eyes = lattice_component()
    solution = plane_graph.solve_component(readings, matches)
    assert solution.reason == ""
    payload = {
        "reading_counts": subject._counts(reading.axis for reading in readings),
        "matched_pair_counts": subject._matched_counts(
            tuple(matches), dict(solution.quadrants)),
        "matched_axes": sorted(solution.axis_resultants),
        "resultants": {str(key): value for key, value in
                       sorted(solution.axis_resultants.items())},
        "residuals": {str(key): value for key, value in
                      sorted(solution.axis_residuals.items())},
        "quadrants": dict(solution.quadrants),
        "fired_gate": solution.reason or "assembled",
    }
    assert payload["reading_counts"] == {"0": 4, "1": 3, "2": 5}
    assert payload["matched_pair_counts"] == {"0": 3, "1": 3, "2": 6}
    assert payload["matched_axes"] == [0, 1, 2]
    assert payload["fired_gate"] == "assembled"
    assert set(payload["resultants"]) == {"0", "1", "2"}
    assert set(payload["residuals"]) == {"0", "1", "2"}


def test_selected_match_diagnostics_are_permutation_invariant_and_complete():
    readings, matches, _eyes = lattice_component()
    solution = plane_graph.solve_component(readings, matches)
    first = subject._selected_match_diagnostics(
        tuple(readings), tuple(matches), (), solution)
    second = subject._selected_match_diagnostics(
        tuple(readings), tuple(reversed(matches)), (), solution)
    assert first == second
    assert first
    required = {"left", "right", "left_axis", "right_axis", "left_b",
                "right_b", "score", "psr", "peak", "second_peak",
                "overlap_cells", "cell_shift_rc", "competing_candidates",
                "circular_phase_residual", "leave_one_out"}
    assert all(required <= set(item) for item in first)
    assert all({"frame", "plane", "source_region_id"} <= set(item["left"])
               for item in first)
    assert subject.canonical_json({"selected_matches": first}) == \
        subject.canonical_json({"selected_matches": second})


def test_sparse_conflicting_and_equal_candidates_remain_abstentions():
    sparse_readings, sparse_matches, _eyes = lattice_component(
        planes=((0, 0),), eyes={"v0": (5, 2, 5), "v1": (6, 3, 7)},
        quadrants={"v0": 0, "v1": 0})
    sparse = plane_graph.solve_component(sparse_readings, sparse_matches)
    assert sparse.reason == "insufficient_axis_coverage"
    left = make_reading("v0", 1, -1.0, [(1.0, 2.0, 1.0)])
    right = make_reading("v1", 1, -1.0, [(1.0, 3.0, 1.0)])
    equal = (make_match("v0", "v1", 1, 1, 0, 0, 0),
             make_match("v0", "v1", 1, 1, 2, 0, 0))
    ambiguous = plane_graph.solve_component((left, right), equal)
    assert ambiguous.reason == "ambiguous_quadrant"
    conflicting = plane_graph.solve_component(
        (left, right), equal[:1],
        quadrant_constraints=(("v0", "v1", 0), ("v1", "v0", 2)))
    assert conflicting.reason == "quadrant_constraint_inconsistent"
    for solution in (sparse, ambiguous, conflicting):
        assert solution.reason in plane_graph.COMPONENT_ABSTENTION_REASONS


def test_injected_outlier_is_diagnostic_only():
    readings, matches, _eyes = lattice_component()
    production_solution = plane_graph.solve_component(readings, matches)
    before = (production_solution.reason, dict(production_solution.quadrants),
              dict(production_solution.axis_resultants))
    outlier = replace(matches[0], score=-999.0, psr=0.01,
                      peak=0.01, second_peak=0.01)
    diagnostic = subject._selected_match_diagnostics(
        tuple(readings), tuple(matches) + (outlier,), (), production_solution)
    after = (production_solution.reason, dict(production_solution.quadrants),
             dict(production_solution.axis_resultants))
    assert after == before
    assert any(item["score"] == -999.0 for item in diagnostic)
    assert any(any(candidate["score"] == -999.0
                   for candidate in item["competing_candidates"])
               for item in diagnostic)


def test_script_surface_has_only_ordinary_inputs_and_fixed_thresholds():
    import inspect

    assert tuple(inspect.signature(
        subject.diagnose_image_derived_axis_coverage).parameters) == (
            "image_inputs", "project_path", "recursive")
    assert (subject.MAX_SHIFT_CELLS, subject.INTEGER_TOLERANCE,
            subject.MIN_RESULTANT, subject.AMBIGUOUS_MARGIN) == (
                32, 0.15, 0.90, 0.02)
    source = SCRIPT.read_text(encoding="utf-8").lower()
    diagnostic_source = inspect.getsource(subject.diagnose_production_result)
    assert "graph_mod._connection_constraints" in diagnostic_source
    assert "quadrant_constraints=constraints" in diagnostic_source
    for forbidden in ("manifest", "f3", "coordinate", "bounds", "synthetic"):
        assert forbidden not in source


def test_diagnostic_schema_and_interpretation_are_the_ablation_contract(normal_result):
    diagnostic = subject.diagnose_production_result(normal_result)
    assert diagnostic["schema"] == "mc-photogrammetry-image-derived-axis-diagnostic-2"
    assert diagnostic["interpretation"] == subject.ABLATION_NOTE
    assert "ablation" in subject.ABLATION_NOTE.lower()
    assert "not camera accuracy" in subject.ABLATION_NOTE.lower()
    assert subject.canonical_json(diagnostic)


def test_unavailable_result_reports_empty_connection_fields(normal_result):
    diagnostic = subject.diagnose_production_result(normal_result)
    assert diagnostic["normal_mode"]["unchanged"] is True
    assert diagnostic["pair_evidence"] == []
    assert diagnostic["solved_edge_count"] == 0
    assert diagnostic["yaw_constraints"] == []
    assert diagnostic["yaw_constraint_count"] == 0
    assert diagnostic["constraint_ablation"] is None


def test_pair_evidence_payload_reports_status_reason_and_support(normal_result):
    graph = normal_result.entry.graph
    records, solved = subject._pair_evidence_payload(graph)
    assert len(records) == len(graph.evidence)
    assert solved == sum(1 for item in graph.evidence if item.status == "solved")
    required = {"left", "right", "status", "reason", "support", "support_floor",
                "match_count", "candidate_edges", "mutual_survivors",
                "right_yaw_class", "hypotheses"}
    assert all(required <= set(item) for item in records)
    assert all(item["status"] in ("solved", "refused") for item in records)
    assert all(isinstance(item["support"], int) for item in records)
    assert all(isinstance(item["reason"], str) for item in records)
    solved_rows = [item for item in records if item["status"] == "solved"]
    assert all(item["right_yaw_class"] is not None for item in solved_rows)
    assert subject.canonical_json({"pair_evidence": records})


def test_pair_evidence_payload_is_empty_without_a_graph():
    assert subject._pair_evidence_payload(None) == ([], 0)


def test_zero_constraints_ablation_equals_the_production_baseline():
    readings, matches, _eyes = lattice_component()
    ablation = subject.constraint_ablation(readings, matches, ())
    assert ablation["baseline"] == ablation["ablated"]
    assert ablation["changed"] is False
    assert ablation["baseline"]["fired_gate"] == "assembled"


def test_solved_constraint_ablation_keeps_the_agreed_assignment():
    readings, matches, _eyes = lattice_component()
    constraints = (("v0", "v1", 1), ("v0", "v2", 0))
    production = plane_graph.solve_component(
        readings, matches, quadrant_constraints=constraints)
    assert production.reason == ""
    ablation = subject.constraint_ablation(readings, matches, constraints)
    assert ablation["baseline"]["fired_gate"] == "assembled"
    assert ablation["baseline"]["quadrants"] == {"v0": 0, "v1": 1, "v2": 0}
    assert ablation["changed"] is False


def test_conflicting_constraints_ablation_changes_only_the_ablation():
    readings, matches, _eyes = lattice_component()
    constraints = (("v0", "v1", 0), ("v1", "v0", 2))
    ablation = subject.constraint_ablation(readings, matches, constraints)
    assert ablation["baseline"]["fired_gate"] == "quadrant_constraint_inconsistent"
    assert ablation["baseline"]["matched_axes"] == []
    assert ablation["baseline"]["resultants"] == {}
    assert ablation["ablated"]["fired_gate"] == "assembled"
    assert ablation["changed"] is True
    assert subject.canonical_json(ablation)


def test_constraint_ablation_is_deterministic_and_json_safe():
    readings, matches, _eyes = lattice_component()
    constraints = (("v0", "v1", 1), ("v0", "v2", 0))
    first = subject.constraint_ablation(readings, matches, constraints)
    second = subject.constraint_ablation(readings, matches, constraints)
    assert first == second
    assert subject.canonical_json(first) == subject.canonical_json(second)
    assert subject._constraint_rows(constraints) == [
        ["v0", "v1", 1], ["v0", "v2", 0]]


def test_ablation_does_not_mutate_the_captured_inputs():
    readings, matches, _eyes = lattice_component()
    before = tuple(matches)
    subject.constraint_ablation(readings, matches, (("v0", "v1", 1),))
    assert tuple(matches) == before
