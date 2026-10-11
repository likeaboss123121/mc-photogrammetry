"""Source-bound partial numerical geometry; synthetic inputs only."""
from dataclasses import replace

import numpy as np
import pytest

from mc_photogrammetry.pipeline import image_derived_partial_joint_graph as subject
from mc_photogrammetry.pipeline import image_derived_partial_axis_support as support_mod
from mc_photogrammetry.pipeline import image_derived_joint_graph as full
from mc_photogrammetry.pipeline import image_derived_graph as legacy
from mc_photogrammetry.reconstruction import plane_graph, plane_placement as placement
from test_partial_axis_observability import fixture, certificate
from test_plane_graph import make_reading, make_match
from test_image_derived_graph_v3 import _source_records


def assemble(data):
    cert, evidence = certificate(data)
    return subject._assemble(data[0], evidence, data[1], cert)


def test_three_frame_core_nonlexical_anchor_and_unknown_peripherals():
    graph = assemble(fixture())
    assert type(graph) is subject.PartialJointGraphResult
    assert graph.status == "assembled" and graph.geometry_evaluations == 1
    assert graph.core_solution.frames == ("b", "c", "d")
    assert graph.core_solution.translations == {f: (0., 0., 0.) for f in ("b", "c", "d")}
    assert dict(graph.axis_offsets)["a"] == (0., 0., None)
    assert dict(graph.axis_offsets)["e"] == (None, 0., 0.)
    assert len(graph.matches) == len(fixture()[1])
    assert all(rank == variables for _, rank, variables, _ in graph.axis_ranks)
    assert graph.geometry_work <= graph.selection.budget.relation_visits
    assert replace(graph) == graph


def test_known_nonzero_translation_and_reversed_frame_order_relation():
    readings, rows, frames, constraints = fixture()
    readings = tuple(make_reading(r.frame, r.axis, 1,
        [(1., 12. if r.frame == "a" else 10. if r.frame == "c" else 8., 1.)]) for r in readings)
    graph = assemble((readings, rows, frames, constraints))
    assert graph.status == "assembled"
    assert np.allclose(graph.core_solution.translations["c"], (-2., -2., -2.))
    assert np.allclose(graph.core_solution.translations["b"], (0., 0., 0.))
    assert np.allclose(graph.core_solution.translations["d"], (0., 0., 0.))
    assert np.allclose(dict(graph.axis_offsets)["a"][:2], (-4., -4.))
    for axis in range(3):
        assert graph.core_solution.axis_resultants[axis] >= .90
        assert graph.core_solution.axis_residuals[axis] <= .15


def test_floating_offsets_never_receive_zero_and_all_factors_survive():
    data = fixture(floating=True)
    graph = assemble(data)
    assert graph.status == "assembled"
    assert dict(graph.axis_offsets)["f"] == dict(graph.axis_offsets)["g"] == (None, None, None)
    assert len(graph.matches) == len(data[1])
    fabricated = tuple((f, (0., 0., 0.) if f == "f" else v) for f, v in graph.axis_offsets)
    with pytest.raises(ValueError, match="unknown offset"):
        replace(graph, axis_offsets=fabricated, result_sha256="")


def test_contradictory_peripheral_prevents_numerical_evaluation(monkeypatch):
    readings, rows, frames, constraints = fixture()
    readings += (make_reading("e", 2, 1, [(1., 20., 1.)]),)
    indexed, _, _ = plane_graph._canonical_reading_index(readings)
    old, _, _ = plane_graph._canonical_reading_index(fixture()[0])
    rebound = []
    for row in rows:
        index = next(i for (f, i), r in indexed.items() if f == row.right and r == old[(row.right, row.right_plane)])
        rebound.append(replace(row, right_plane=index))
    b = next(i for (f, i), r in indexed.items() if f == "b" and r.axis == 2)
    e = next(i for (f, i), r in indexed.items() if f == "e" and r.axis == 2 and r.alternatives[0].distance_blocks == 20.)
    data = readings, (*rebound, make_match("b", "e", 2, 2, 0, b, e)), frames, constraints
    monkeypatch.setattr(subject, "_evaluate_core", lambda *a: pytest.fail("contradiction evaluated"))
    graph = assemble(data)
    assert graph.selection.status == "no" and graph.status == "unavailable"
    assert graph.geometry_evaluations == 0 and graph.axis_offsets == ()


def test_ambiguous_scale_has_no_geometry(monkeypatch):
    monkeypatch.setattr(subject, "_evaluate_core", lambda *a: pytest.fail("ambiguity evaluated"))
    graph = assemble(fixture(alternatives=2))
    assert graph.selection.status == "multiple" and graph.status == "unavailable"
    assert graph.geometry_evaluations == 0 and graph.core_solution is None


@pytest.mark.parametrize("fault", ["rank", "nonfinite", "inconsistent"])
def test_numerical_rank_nonfinite_and_equation_refusals_are_atomic(monkeypatch, fault):
    original = np.linalg.lstsq
    def damaged(*args, **kwargs):
        x, residual, rank, singular = original(*args, **kwargs)
        if fault == "rank":
            rank -= 1
        elif fault == "nonfinite":
            x[0] = np.nan
        else:
            x[0] += 1.
        return x, residual, rank, singular
    monkeypatch.setattr(np.linalg, "lstsq", damaged)
    graph = assemble(fixture())
    assert graph.status == "unavailable" and graph.reason == "partial_geometry_unavailable"
    assert graph.geometry_evaluations == 1 and not graph.axis_offsets and graph.core_solution is None


@pytest.mark.parametrize("field,value", [("geometry_evaluations", 0), ("axis_ranks", ()),
    ("geometry_work", -1), ("matches", ()), ("result_sha256", "0" * 64)])
def test_geometry_or_factor_tamper_rejected(field, value):
    graph = assemble(fixture())
    kwargs = {field: value}
    if field != "result_sha256":
        kwargs["result_sha256"] = ""
    with pytest.raises(ValueError):
        replace(graph, **kwargs)


def test_constructor_copy_needs_actual_source_replay_for_downstream():
    graph = assemble(fixture())
    copy = replace(graph)
    assert graph.result_sha256 == copy.result_sha256
    result = support_mod.derive_partial_core_axis_support(graph_result=copy)
    assert isinstance(result, placement.PlacementRefusal)
    with pytest.raises(ValueError, match="source replay"):
        subject._require_live(copy)


def test_rank_bool_and_resealed_work_counter_tamper_refused():
    graph = assemble(fixture())
    ranks = ((False, *graph.axis_ranks[0][1:]), *graph.axis_ranks[1:])
    with pytest.raises(ValueError):
        replace(graph, axis_ranks=ranks, result_sha256="")
    with pytest.raises(ValueError, match="work census"):
        replace(graph, geometry_work=graph.geometry_work + 1, result_sha256="")


def test_core_c1_preserves_complete_partial_authority_and_original_projection():
    graph = assemble(fixture(floating=True))
    # Authored certificate/numerical fixture capability; no source truncation.
    full._register_live(graph)
    support = support_mod.derive_partial_core_axis_support(graph_result=graph)
    assert type(support) is support_mod.PartialCoreAxisSupportResult, support
    assert support.status == "assembled", support.reason
    assert support.graph is graph and support.axis_support.frames == ("b", "c", "d")
    assert len(graph.readings) > len(support.core_reading_indices)
    assert len(graph.matches) > len(support.source_rows)
    assert all(graph.readings[i].frame == f and graph.readings[i].axis == a
               for i, f, a in support.core_reading_indices)
    assert all(graph.candidate_matches[i].left in ("b", "c", "d")
               and graph.candidate_matches[i].right in ("b", "c", "d") for i in support.source_rows)
    assert dict(graph.axis_offsets)["f"] == (None, None, None)
    with pytest.raises(ValueError, match="projection types"):
        replace(support, source_rows=(False, *support.source_rows[1:]), result_sha256="")


@pytest.fixture(scope="module")
def sources():
    patch = pytest.MonkeyPatch()
    try:
        yield _source_records(patch)
    finally:
        patch.undo()


def test_actual_source_derivation_replay_and_full_mode_parity(sources, monkeypatch):
    before = full.derive_image_derived_graph(**sources)
    assert before.status == "assembled"
    calls = {"matcher": 0, "geometry": 0}
    original_match = plane_graph.match_plane_graph
    original_evaluate = subject._evaluate_core
    def matched(*args, **kwargs):
        calls["matcher"] += 1
        return original_match(*args, **kwargs)
    def evaluated(*args, **kwargs):
        calls["geometry"] += 1
        return original_evaluate(*args, **kwargs)
    monkeypatch.setattr(plane_graph, "match_plane_graph", matched)
    monkeypatch.setattr(subject, "_evaluate_core", evaluated)
    graph = subject.derive_partial_image_derived_graph(**sources)
    assert type(graph) is subject.PartialJointGraphResult, graph
    assert graph.status == "assembled", graph.reason
    assert graph.core_solution.translations == before.component_solution.translations
    assert calls == {"matcher": 1, "geometry": 1}
    copied = replace(graph)
    assert isinstance(support_mod.derive_partial_core_axis_support(graph_result=copied), placement.PlacementRefusal)
    assert subject.replay_partial_image_derived_graph(graph_result=copied, **sources) is copied
    assert calls == {"matcher": 2, "geometry": 2}
    assert subject._require_live(copied) is copied
    after = full.derive_image_derived_graph(**sources)
    assert after.result_sha256 == before.result_sha256
    assert legacy._solution_payload(after.component_solution) == legacy._solution_payload(before.component_solution)
    core_support = support_mod.derive_partial_core_axis_support(graph_result=copied)
    assert type(core_support) is support_mod.PartialCoreAxisSupportResult, core_support
    assert core_support.status == "assembled", core_support.reason
    assert core_support.axis_support.frames == graph.core_solution.frames
    assert core_support.source_rows == tuple(r.source_row for r in graph.selection.representatives)
    with pytest.raises(ValueError):
        replace(core_support, source_rows=(), result_sha256="")
    # The existing full graph consumer cannot accept the separate partial type.
    with pytest.raises((ValueError, TypeError)):
        full._require_live(copied)


def test_source_mismatch_and_budget_mismatch_refuse_replay(sources):
    graph = subject.derive_partial_image_derived_graph(**sources)
    foreign = dict(sources)
    foreign["session"] = object()
    assert isinstance(subject.replay_partial_image_derived_graph(graph_result=graph, **foreign), placement.PlacementRefusal)
    bad_budget = replace(graph.selection.budget, relation_visits=0)
    assert isinstance(subject.replay_partial_image_derived_graph(graph_result=graph, budget=bad_budget, **sources), placement.PlacementRefusal)


def test_core_paths_through_peripheral_do_not_fabricate_c1_bridge():
    readings, rows, frames, constraints = fixture()
    indexed, _, _ = plane_graph._canonical_reading_index(readings)
    keys = {(f, r.axis): i for (f, i), r in indexed.items()}
    rows = tuple(make_match("a", "c", 0, 0, 0, keys[("a", 0)], keys[("c", 0)])
                 if row.right == "c" and row.left_axis == 0 else row for row in rows)
    graph = assemble((readings, rows, frames, constraints))
    assert graph.status == "assembled" and graph.core_solution.frames == ("b", "c", "d")
    # Explicitly register the authored synthetic graph capability for the C1
    # guard test. Genuine source registration/replay is covered separately.
    full._register_live(graph)
    result = support_mod.derive_partial_core_axis_support(graph_result=graph)
    assert type(result) is support_mod.PartialCoreAxisSupportResult
    assert result.status == "unavailable" and result.reason == "partial_core_direct_support_unavailable"
    assert result.axis_support is None
