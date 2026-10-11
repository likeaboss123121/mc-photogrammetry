"""Synthetic source-record replay controls for the unavailable Phase B graph."""
import dataclasses
import json

import numpy as np
import pytest

from mc_photogrammetry.pipeline import image_derived_graph as legacy
from mc_photogrammetry.pipeline import image_derived_graph_v3 as subject
from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
from mc_photogrammetry.pipeline import image_derived_signatures as f1
from mc_photogrammetry.pipeline import image_derived_lattice_measure as measure
from mc_photogrammetry.pipeline import image_derived_connections as connections
from mc_photogrammetry.reconstruction import endpoint_hypothesis as csp
from mc_photogrammetry.reconstruction import lattice, lattice_planes as lp, plane_graph, plane_placement
from mc_photogrammetry.reconstruction import plane_geometry_provenance as provenance
from test_image_derived_connections import _simple_session, _budgets, _solved_result
from test_plane_geometry_provenance import _reading, _weights, _mask, POLY, _split_patch
from test_image_derived_endpoint_selection import _evidence, _forge


def _source_records(monkeypatch, *, alternatives=False):
    """Accepted synthetic records; no real images or estimator is run.

    Structural/held-out patches are produced and rebound by existing evidence
    helpers. Read-only signature arrays are explicit synthetic matcher inputs,
    not claimed measurements of a real scene or an accuracy fixture.
    """
    session = _simple_session(monkeypatch, ("a", "b"))
    graph = connections.build_image_derived_connections(session, budgets=_budgets(),
                matcher=lambda left, _li, right, _ri, **kw: _solved_result(left, right, yaw_class=0))
    assert graph.status == "ok"
    readings, structural_patches = [], []
    for frame in ("a", "b"):
        for axis in range(3):
            region = lp._make_region_evidence(_mask(), axis, POLY, ("phase-b-synthetic",))
            r = _reading(region=region, axis=axis, frame=frame)
            patch = np.abs(5. + np.random.default_rng(710 + axis).normal(size=(16, 16))).astype("<f4")
            phase = lp._periodic_energy_phase_evidence(patch, r.period_pixels,
                        centre=r.rectification.centre, extent=r.extent, samples=r.samples,
                        source_region_id=region.source_region_id)
            r = dataclasses.replace(r, structural_share=float(lp._structural_share(patch, r.period_pixels)),
                        phase_evidence=phase,
                        phase_pixels=tuple(0. if p is None else float(p) for p in phase.phase_uv_pixels))
            if alternatives:
                r = dataclasses.replace(r, alternatives=(r.alternatives[0], lp.DistanceAlternative(2., 1., 1.)))
            readings.append(r)
            structural_patches.append(patch)
    readings = tuple(readings)
    weights, sources = [], []
    for index, r in enumerate(readings):
        weights.append(_weights(r, patch=structural_patches[index], frame=r.frame, canonical_index=r.axis))
        patch = _split_patch()
        budget = lp._make_budget({name: 1_000_000 for name in lp._SPLIT_BUDGET_FIELDS})
        geometry = lp.SplitPhaseGeometry(centre=(0., 0.), extent=1., samples=8,
                    rectified_pixel_step=lattice.rectified_pixel_step(1., 8), period_pixels=4.)
        split = lp.produce_split_phase_evidence(patch, geometry=geometry, frame=r.frame,
                    plane_index=r.axis, reading_axis=r.axis, source_region_id=r.region_evidence.source_region_id,
                    partition_config=lp.SPLIT_PHASE_PARTITION_CONFIG, budget=budget)
        assert split.status == "complete"
        sources.append(provenance.SplitSourceEvidence(r.frame, r.axis, r.axis, split, budget,
                        patch.tobytes(), *patch.shape))
    bridge = provenance.bind_geometry_provenance(readings=readings, structural_weights=tuple(weights),
                                               split_sources=tuple(sources))
    assert isinstance(bridge, provenance.GeometryProvenanceBridge), bridge
    measured = measure.ImageDerivedLatticeResult(measure.IMAGE_DERIVED_LATTICE_MEASURE_SCHEMA,
                measure.IMAGE_DERIVED_LATTICE_MEASURE_PRODUCER, measure.GEOMETRY_UNAVAILABLE,
                "assembled", "", "", session.session_sha256, "d"*64, graph.graph_sha256,
                ("a", "b"), (), readings, bridge, ())
    evidence = _evidence(readings, alternative_specific=alternatives)
    masks = tuple(f1.UsableMask(frame.frame_id, *frame.image_shape[:2],
                              np.ones(frame.image_shape[:2], dtype=np.bool_).tobytes()) for frame in session.frames)
    signatures = f1.ImageDerivedSignaturesResult(f1.IMAGE_DERIVED_SIGNATURES_SCHEMA,
                f1.IMAGE_DERIVED_SIGNATURES_PRODUCER, f1.SIGNATURES_TERMINAL, "assembled", "", "",
                session.session_sha256, measured.result_sha256, ("a", "b"), masks, evidence, ())
    return dict(session=session, lattice_result=measured, signatures_result=signatures, connection_graph=graph)


@pytest.fixture(scope="module")
def sources():
    patch = pytest.MonkeyPatch()
    try:
        yield _source_records(patch)
    finally:
        patch.undo()


def _graph(sources):
    result = subject.derive_image_derived_graph_v3(**sources)
    assert isinstance(result, subject.ImageDerivedGraphResultV3), result
    return result


def _reseal_graph(graph, **changes):
    return dataclasses.replace(graph, **changes, result_sha256="")


def _reseal_selection(graph, **changes):
    cert = dataclasses.replace(graph.selection, **changes, result_sha256="")
    return _reseal_graph(graph, selection=cert, reason=cert.reason, detail=cert.status)


def _refused(result):
    assert isinstance(result, plane_placement.PlacementRefusal), result
    assert result.reason == "provenance_mismatch"


def test_actual_matcher_source_replay_unique_remains_geometry_unavailable(sources, monkeypatch):
    monkeypatch.setattr(plane_graph, "solve_component", lambda *a, **k: pytest.fail("solver forbidden"))
    monkeypatch.setattr(plane_graph, "_evaluate_assignment", lambda *a, **k: pytest.fail("evaluator forbidden"))
    graph = _graph(sources)
    assert graph.schema == "mc-photogrammetry-image-derived-graph-v3"
    assert graph.producer == "m9-g-image-derived-graph-6"
    assert graph.selection.status == "unique"
    assert graph.status == "unavailable" and graph.reason == "joint_geometry_not_evaluated"
    assert graph.matches == () and graph.component_solution is None and graph.candidate_matches
    assert subject.replay_image_derived_graph_v3(graph_result=graph, **sources) is graph
    assert dataclasses.replace(graph) == graph


def test_one_matcher_pass_receives_exact_complete_source_universe(sources, monkeypatch):
    actual = plane_graph.match_plane_graph
    calls = []
    def matcher(readings, signatures, **kwargs):
        calls.append((readings, signatures, kwargs))
        return actual(readings, signatures, **kwargs)
    monkeypatch.setattr(plane_graph, "match_plane_graph", matcher)
    graph = _graph(sources)
    assert len(calls) == 1 and calls[0][0] is sources["lattice_result"].readings
    assert all(a is b.signature for a,b in zip(calls[0][1], sources["signatures_result"].signatures))
    assert calls[0][2] == {"max_shift_cells": 32}
    assert graph.candidate_matches == actual(*calls[0][:2], **calls[0][2])


@pytest.mark.parametrize("field,value", [("schema", "mc-photogrammetry-image-derived-graph-v2"),
                                         ("producer", "m9-g-image-derived-graph-5")])
def test_resealed_old_graph_identity_cannot_replay_new_schema(sources, field, value):
    graph = _graph(sources)
    stale = _forge(graph, **{field:value})
    stale = _forge(stale, result_sha256=selection._sha(subject._payload(stale)))
    with pytest.raises(ValueError): dataclasses.replace(stale)
    _refused(subject.replay_image_derived_graph_v3(graph_result=stale, **sources))
    old = legacy.ImageDerivedGraphResult(legacy.IMAGE_DERIVED_GRAPH_SCHEMA, legacy.IMAGE_DERIVED_GRAPH_PRODUCER,
        legacy.GRAPH_TERMINAL, "unavailable", "no_matches", "", graph.session_sha256,
        graph.lattice_sha256, graph.signatures_sha256, graph.frame_ids, graph.readings, graph.signatures, (), None)
    _refused(subject.replay_image_derived_graph_v3(graph_result=old, **sources))


def test_nested_stale_selection_identity_refuses_before_matcher(sources, monkeypatch):
    graph = _graph(sources)
    stale = _forge(graph.selection, producer="m9-g-endpoint-selection-0")
    stale = _forge(stale, result_sha256=selection._sha(selection._payload(stale)))
    graph = _forge(graph, selection=stale)
    graph = _forge(graph, result_sha256=selection._sha(subject._payload(graph)))
    monkeypatch.setattr(plane_graph, "match_plane_graph", lambda *a, **k: pytest.fail("stale nested source"))
    _refused(subject.replay_image_derived_graph_v3(graph_result=graph, **sources))


@pytest.mark.parametrize("change", ["omit", "order", "score"])
def test_self_resealed_complete_candidate_forgery_fails_source_replay(sources, change):
    graph = _graph(sources)
    rows = graph.candidate_matches
    if change == "omit": rows = rows[:-1]
    elif change == "order": rows = tuple(reversed(rows))
    else: rows = (dataclasses.replace(rows[0], score=.123), *rows[1:])
    cert = selection.select_endpoint_hypotheses(readings=graph.readings, signatures=graph.signatures,
            candidate_matches=tuple(rows), frame_ids=graph.frame_ids, quadrant_constraints=graph.selection.quadrant_constraints,
            budget=graph.selection.budget, session_sha256=graph.session_sha256, lattice_sha256=graph.lattice_sha256,
            signatures_sha256=graph.signatures_sha256, connection_sha256=graph.connection_sha256)
    altered = _reseal_graph(graph, candidate_matches=tuple(rows), selection=cert, reason=cert.reason, detail=cert.status)
    assert altered.result_sha256 != graph.result_sha256
    _refused(subject.replay_image_derived_graph_v3(graph_result=altered, **sources))


def test_altered_constraints_budget_and_frames_have_independent_source_binding(sources):
    graph = _graph(sources)
    altered = _reseal_selection(graph, quadrant_constraints=())
    _refused(subject.replay_image_derived_graph_v3(graph_result=altered, **sources))
    altered = _reseal_selection(graph, budget=dataclasses.replace(csp.EndpointBudget(), branch_nodes=100))
    _refused(subject.replay_image_derived_graph_v3(graph_result=altered, **sources))
    cert = dataclasses.replace(graph.selection, frame_ids=tuple(reversed(graph.frame_ids)), result_sha256="")
    altered = _reseal_graph(graph, frame_ids=cert.frame_ids, selection=cert)
    _refused(subject.replay_image_derived_graph_v3(graph_result=altered, **sources))


def test_missing_signature_evidence_and_changed_alternative_or_binding_fail_closed(sources):
    graph = _graph(sources)
    with pytest.raises(ValueError): _reseal_graph(graph, signatures=graph.signatures[:-1])
    bindings = list(graph.selection.reading_bindings)
    row = list(bindings[0]); row[4] = ((2., 1., 1.),); bindings[0] = tuple(row)
    with pytest.raises(ValueError): _reseal_selection(graph, reading_bindings=tuple(bindings))
    row = list(graph.selection.reading_bindings[0]); row[1] = 99; bindings[0] = tuple(row)
    with pytest.raises(ValueError): _reseal_selection(graph, reading_bindings=tuple(bindings))
    _refused(subject.replay_image_derived_graph_v3(graph_result=graph, **{**sources,"connection_graph":None}))


def test_old_measure_producer_and_nested_source_digest_refuse_before_matcher(sources, monkeypatch):
    historical = _forge(sources["lattice_result"], producer="m9c-image-derived-lattice-measure-1")
    historical = _forge(historical, result_sha256=plane_placement._sha256(measure._result_payload(historical)))
    monkeypatch.setattr(plane_graph, "match_plane_graph", lambda *a, **k: pytest.fail("stale source reaches matcher"))
    _refused(subject.derive_image_derived_graph_v3(**{**sources,"lattice_result":historical}))
    evidence = sources["signatures_result"].signatures
    bad = _forge(evidence[0], signature_sha256="f"*64)
    stale = _forge(sources["signatures_result"], signatures=(bad,*evidence[1:]))
    _refused(subject.derive_image_derived_graph_v3(**{**sources,"signatures_result":stale}))


def test_changed_source_digest_and_exhaustive_cannot_bypass_exact_source_replay(sources, monkeypatch):
    base = _graph(sources)
    rows = base.candidate_matches
    # Shape-valid source digest forgery is distinguished from source-backed
    # uniqueness. Matcher replacement preserves the exact original tuple.
    def matcher(*a, **k): return rows
    monkeypatch.setattr(plane_graph,"match_plane_graph",matcher)
    proof = base.selection.proof_record()
    proof["source_digest"] = "f"*64  # self-sealed internal shape is valid, derivation is not
    cert = dataclasses.replace(base.selection, proof_json=selection._encode(proof).decode(), result_sha256="")
    forged = _reseal_graph(base, selection=cert)
    _refused(subject.replay_image_derived_graph_v3(graph_result=forged, **sources))
    proof["exhaustive"] = False
    with pytest.raises(ValueError):
        dataclasses.replace(base.selection, proof_json=selection._encode(proof).decode(),result_sha256="")


def test_explicit_small_budget_replays_only_with_same_independently_requested_budget(sources):
    budget = dataclasses.replace(csp.EndpointBudget(), branch_nodes=0)
    graph = subject.derive_image_derived_graph_v3(**sources,budget=budget)
    assert graph.selection.status == "budget_refused" and graph.matches == ()
    _refused(subject.replay_image_derived_graph_v3(graph_result=graph, **sources))
    assert subject.replay_image_derived_graph_v3(graph_result=graph, **sources,budget=budget) is graph


def test_genuine_multiple_canonical_interpretations_cannot_be_resealed_unique(monkeypatch):
    sources=_source_records(monkeypatch,alternatives=True)
    graph=_graph(sources)
    assert graph.selection.status=="multiple", (graph.selection.reason,
        graph.selection.proof_record()["eligibility_reasons"],
        [(r.left_plane,r.right_plane,r.left_b,r.right_b,r.quadrant_of_right) for r in graph.candidate_matches])
    assert {(row.left_axis, row.left_b) for row in graph.candidate_matches if row.left_b == row.right_b} == {
        (axis, b) for axis in range(3) for b in (1., 2.)}
    proof=graph.selection.proof_record()
    proof["status"]="unique";proof["exhaustive"]=True;proof["choices"]=proof["choices"][:1]
    proof["counters"]["solutions_found"]=proof["counters"]["solution_rows"]=1
    assert proof["counters"]["unvisited_assignments"]==0  # forced source Q; ambiguity is alternatives
    selected={(f,i):a for f,i,a in proof["choices"][0]["alternatives"]}
    canonical,_,_=plane_graph._canonical_reading_index(graph.readings)
    reps=[]
    for i,factor in enumerate(proof["choices"][0]["factors"]):
        source_row=factor["source_rows"][0];row=graph.candidate_matches[source_row]
        row=dataclasses.replace(row,
             left_b=canonical[(row.left,row.left_plane)].alternatives[selected[(row.left,row.left_plane)]].blocks_per_period,
             right_b=canonical[(row.right,row.right_plane)].alternatives[selected[(row.right,row.right_plane)]].blocks_per_period)
        reps.append(selection.RepresentativeConstraint(i,source_row,row))
    cert=dataclasses.replace(graph.selection,status="unique",reason="joint_geometry_not_evaluated",
             proof_json=selection._encode(proof).decode(),representatives=tuple(reps),result_sha256="")
    forged=_reseal_graph(graph,selection=cert,reason=cert.reason,detail=cert.status)
    assert dataclasses.replace(forged)==forged  # complete internal joins do not prove exhaustive uniqueness
    _refused(subject.replay_image_derived_graph_v3(graph_result=forged,**sources))


def test_cumulative_validation_refusal_cannot_reach_source_matcher(sources,monkeypatch):
    graph=_graph(sources)
    proof=graph.selection.proof_record();proof["counters"]["relation_visits"]=0
    cert=dataclasses.replace(graph.selection,budget=dataclasses.replace(graph.selection.budget,relation_visits=16),
               proof_json=selection._encode(proof).decode(),result_sha256="")
    forged=_forge(graph,selection=cert)
    forged=_forge(forged,result_sha256=selection._sha(subject._payload(forged)))
    monkeypatch.setattr(plane_graph,"match_plane_graph",lambda *a,**k:pytest.fail("cumulative refusal reached matcher"))
    _refused(subject.replay_image_derived_graph_v3(graph_result=forged,**sources,budget=cert.budget))
