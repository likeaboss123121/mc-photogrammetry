"""Synthetic production selector/evaluator controls; no image files are read."""
import dataclasses
import copy
import hashlib
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

from mc_photogrammetry.pipeline import image_derived_joint_graph as subject
from mc_photogrammetry.pipeline import image_derived_graph_v3 as phase_b
from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
from mc_photogrammetry.pipeline import image_derived_signatures as f1
from mc_photogrammetry.reconstruction import endpoint_hypothesis as csp
from mc_photogrammetry.reconstruction import plane_graph, plane_placement as placement
from test_image_derived_graph_v3 import _source_records


@pytest.fixture(scope="module")
def sources():
    patch = pytest.MonkeyPatch()
    try:
        yield _source_records(patch)
    finally:
        patch.undo()


def _calls(monkeypatch):
    counts = {"matcher": 0, "selector": 0, "evaluator": 0}
    for name, module, attr in (("matcher", plane_graph, "match_plane_graph"),
                               ("selector", selection, "select_endpoint_hypotheses"),
                               ("evaluator", plane_graph, "_evaluate_assignment")):
        original = getattr(module, attr)
        def wrapped(*args, _key=name, _fn=original, **kwargs):
            counts[_key] += 1
            return _fn(*args, **kwargs)
        monkeypatch.setattr(module, attr, wrapped)
    monkeypatch.setattr(plane_graph, "solve_component", lambda *a, **k: pytest.fail("old quadrant search called"))
    return counts


def test_genuine_three_axis_unique_has_finite_component_and_one_evaluation(sources, monkeypatch):
    counts = _calls(monkeypatch)
    graph = subject.derive_image_derived_graph(**sources)
    assert isinstance(graph, subject.ImageDerivedGraphResult), graph
    assert graph.status == "assembled", (graph.reason, graph.detail)
    assert graph.selection.status == "unique" and graph.geometry_evaluations == 1
    assert graph.component_solution.scale_known and not graph.component_solution.reason
    assert set(graph.component_solution.axis_resultants) == {0, 1, 2}
    assert graph.matches == tuple(r.match for r in graph.selection.representatives)
    assert counts == {"matcher": 1, "selector": 1, "evaluator": 1}
    assert subject.replay_image_derived_graph(graph_result=graph, **sources) is graph
    assert counts == {"matcher": 2, "selector": 2, "evaluator": 2}


def test_genuine_multiple_has_no_geometry_calls(monkeypatch):
    sources = _source_records(monkeypatch, alternatives=True)
    counts = _calls(monkeypatch)
    graph = subject.derive_image_derived_graph(**sources)
    assert graph.selection.status == "multiple"
    assert graph.status == "unavailable" and graph.matches == () and graph.component_solution is None
    assert counts == {"matcher": 1, "selector": 1, "evaluator": 0}


def _vertical_factor_sources(monkeypatch, *, anchor_right_distance=0.5):
    """Authored source records; independent vertical factors share a translation.

    The target's period-2 distance implies a different displacement in frame b.
    An independent period-3 anchor permits only period 1. Exact self-correlations
    tie in the actual local scored filter. The anchor occupies a disjoint block
    window, preventing target-anchor cross-pair overlap within the fixed radius.
    No matcher rows or CSP outcomes are supplied by the fixture.
    """
    from mc_photogrammetry.reconstruction import lattice_planes as lp
    from mc_photogrammetry.reconstruction import plane_geometry_provenance as provenance
    from test_image_derived_endpoint_selection import _evidence
    from test_plane_geometry_provenance import _mask, POLY

    base = _source_records(monkeypatch, alternatives=True)
    original = base["lattice_result"]
    readings, weights, split_sources = [], [], []
    for frame_index, frame in enumerate(("a", "b")):
        originals = original.readings[3 * frame_index:3 * frame_index + 3]
        target = dataclasses.replace(originals[1], alternatives=(originals[1].alternatives[0],
            lp.DistanceAlternative(2., 1. if frame == "a" else 1.5, 1.)))
        anchor_region = lp._make_region_evidence(np.flip(_mask(), axis=1).copy(), 1,
            POLY, ("graph10-independent-vertical",))
        old_weight = original.bridge.structural_weights[3 * frame_index + 1]
        structural_patch = np.frombuffer(old_weight.patch_float32_le, dtype="<f4").reshape(
            old_weight.patch_rows, old_weight.patch_columns)
        anchor_phase = lp._periodic_energy_phase_evidence(structural_patch, originals[1].period_pixels,
            centre=originals[1].rectification.centre, extent=originals[1].extent,
            samples=originals[1].samples, source_region_id=anchor_region.source_region_id)
        anchor = dataclasses.replace(originals[1], region_evidence=anchor_region, phase_evidence=anchor_phase,
            alternatives=(lp.DistanceAlternative(3., 0.5 if frame == "a" else anchor_right_distance, 1.),))
        assert target.region_evidence.source_region_id != anchor.region_evidence.source_region_id
        frame_readings = (dataclasses.replace(originals[0], alternatives=originals[0].alternatives[:1]),
                          target, anchor,
                          dataclasses.replace(originals[2], alternatives=originals[2].alternatives[:1]))
        for index, reading in enumerate(frame_readings):
            old_index = 3 * frame_index + reading.axis
            readings.append(reading)
            weights.append(dataclasses.replace(original.bridge.structural_weights[old_index],
                canonical_index=index, source_region_id=reading.region_evidence.source_region_id,
                phase_evidence=reading.phase_evidence, evidence_sha256=""))
            old_source = original.bridge.split_sources[old_index]
            _, patch, budget = old_source.to_source()
            split = lp.produce_split_phase_evidence(patch, geometry=old_source.result.geometry,
                frame=frame, plane_index=index, reading_axis=reading.axis,
                source_region_id=reading.region_evidence.source_region_id,
                partition_config=lp.SPLIT_PHASE_PARTITION_CONFIG, budget=budget)
            assert split.status == "complete"
            split_sources.append(dataclasses.replace(old_source, canonical_index=index,
                result=split, source_sha256=""))
    readings = tuple(readings)
    canonical, _, _ = plane_graph._canonical_reading_index(readings)
    assert all(canonical[(reading.frame, index % 4)] is reading
               for index, reading in enumerate(readings))
    bridge = provenance.bind_geometry_provenance(readings=readings,
        structural_weights=tuple(weights), split_sources=tuple(split_sources))
    assert isinstance(bridge, provenance.GeometryProvenanceBridge), bridge
    measured = dataclasses.replace(original, readings=readings, bridge=bridge, result_sha256=None)
    evidence = []
    for item in _evidence(readings, alternative_specific=True):
        signature = item.signature
        if item.reading_axis == 1 and item.blocks_per_period == 3.:
            signature = dataclasses.replace(signature, origin_uv=(100., 0.))
        evidence.append(dataclasses.replace(item, signature=signature, signature_sha256=""))
    signatures = dataclasses.replace(base["signatures_result"], lattice_sha256=measured.result_sha256,
        signatures=tuple(evidence), signatures_sha256="", result_sha256="")
    return dict(base, lattice_result=measured, signatures_result=signatures)


@pytest.fixture(scope="module")
def full_universe_sources():
    patch = pytest.MonkeyPatch()
    try:
        yield _vertical_factor_sources(patch)
    finally:
        patch.undo()


def _observe_full_universe(monkeypatch):
    """Observe actual matcher return/publication and exact selector ingress."""
    seen = []
    matcher = plane_graph.match_plane_graph
    selector = selection.select_endpoint_hypotheses

    def match(*args, **kwargs):
        filtered = matcher(*args, **kwargs)
        assert len(kwargs["admitted_candidate_sink"]) == 1
        universe = kwargs["admitted_candidate_sink"][0]
        seen.append((universe, filtered))
        return filtered

    def select(**kwargs):
        universe, _ = seen[-1]
        assert kwargs["candidate_matches"] is universe.candidates
        assert kwargs["candidate_dispositions"] is universe.dispositions
        return selector(**kwargs)

    monkeypatch.setattr(plane_graph, "match_plane_graph", match)
    monkeypatch.setattr(selection, "select_endpoint_hypotheses", select)
    return seen


def test_real_full_universe_selects_locally_dropped_globally_consistent_source_row(
        full_universe_sources, monkeypatch):
    counts = _calls(monkeypatch)
    seen = _observe_full_universe(monkeypatch)
    graph = subject.derive_image_derived_graph(**full_universe_sources)
    assert isinstance(graph, subject.ImageDerivedGraphResult), graph
    universe, filtered = seen[0]
    assert graph.producer == "m9-g-image-derived-graph-10"
    assert graph.candidate_matches == universe.candidates
    assert graph.selection.candidate_dispositions == universe.dispositions
    assert universe.source_rows == tuple(range(len(universe.candidates)))
    active = tuple(row for row in universe.candidates if row.quadrant_of_right == 0)
    assert tuple((row.left_plane, row.right_plane, row.left_axis, row.right_axis,
                  row.left_b, row.right_b) for row in active) == (
        (0, 0, 0, 0, 1., 1.),  # X.
        (1, 1, 1, 1, 1., 1.), (1, 1, 1, 1, 1., 2.),
        (1, 1, 1, 1, 2., 1.), (1, 1, 1, 1, 2., 2.),  # All four target-period rows.
        (2, 2, 1, 1, 3., 3.),  # Independent vertical anchor.
        (3, 3, 2, 2, 1., 1.),  # Z.
    ), active
    assert graph.selection.status == "unique" and graph.status == "assembled"
    assert any(universe.dispositions[rep.source_row] == plane_graph.SAME_AXIS_DISPOSITION_DROPPED
               and universe.candidates[rep.source_row] not in filtered
               for rep in graph.selection.representatives)
    assert any(row.left_b == row.right_b == 2. and row.left_axis == 1 for row in active)
    assert all(rep.match.left_b != 2. and rep.match.right_b != 2.
               for rep in graph.selection.representatives)
    assert all(rep.match == universe.candidates[rep.source_row]
               for rep in graph.selection.representatives)
    proof = graph.selection.proof_record()
    assert proof["counters"]["partial_prunes"] > 0 and proof["partial_exclusions"]
    assert len(proof["choices"]) == 1 and proof["exhaustive"]
    assert graph.geometry_settings == (0.15, 0.90, 0.02)
    assert graph.geometry_evaluations == 1 and graph.component_solution.scale_known
    assert all(graph.component_solution.axis_resultants[a] >= 0.90
               and graph.component_solution.axis_residuals[a] <= 0.15 for a in range(3))
    assert counts == {"matcher": 1, "selector": 1, "evaluator": 1}


def test_real_false_mandatory_vertical_factor_refuses_without_pruning_factor(monkeypatch):
    source = _vertical_factor_sources(monkeypatch, anchor_right_distance=1.25)
    counts = _calls(monkeypatch)
    seen = _observe_full_universe(monkeypatch)
    graph = subject.derive_image_derived_graph(**source)
    assert isinstance(graph, subject.ImageDerivedGraphResult), graph
    universe, _ = seen[0]
    active = tuple(row for row in universe.candidates if row.quadrant_of_right == 0)
    assert tuple((row.left_plane, row.right_plane, row.left_axis, row.right_axis,
                  row.left_b, row.right_b) for row in active) == (
        (0, 0, 0, 0, 1., 1.),  # X.
        (1, 1, 1, 1, 1., 1.), (1, 1, 1, 1, 1., 2.),
        (1, 1, 1, 1, 2., 1.), (1, 1, 1, 1, 2., 2.),  # All four target-period rows.
        (2, 2, 1, 1, 3., 3.),  # Source-bound contradictory anchor remains mandatory.
        (3, 3, 2, 2, 1., 1.),  # Z.
    ), active
    assert graph.candidate_matches == universe.candidates
    assert graph.selection.status == "no" and graph.status == "unavailable"
    proof = graph.selection.proof_record()
    assert proof["reason"] == "canonical_constraints_inconsistent"
    assert proof["counters"]["active_factor_visits"] == 4
    assert proof["counters"]["partial_prunes"] > 0 and proof["partial_exclusions"]
    assert proof["choices"] == [] and proof["exhaustive"]
    assert graph.matches == () and graph.component_solution is None and graph.geometry_evaluations == 0
    assert counts == {"matcher": 1, "selector": 1, "evaluator": 0}


@pytest.mark.parametrize("association", ["duplicate", "reordered"])
def test_admitted_dispositions_reject_duplicate_or_reordered_source_row_associations(sources, association):
    graph = subject.derive_image_derived_graph(**sources)
    rows = tuple(range(len(graph.candidate_matches)))
    assert len(rows) >= 3
    bad_rows = (rows[0], rows[0], *rows[2:]) if association == "duplicate" else rows[::-1]
    with pytest.raises(ValueError, match="invalid bounded admitted candidate universe"):
        plane_graph.AdmittedCandidateUniverse(graph.candidate_matches,
            graph.selection.candidate_dispositions, bad_rows)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "unknown"])
def test_candidate_dispositions_reject_cardinality_or_label_tampering(sources, mutation):
    graph = subject.derive_image_derived_graph(**sources)
    dispositions = graph.selection.candidate_dispositions
    bad = (dispositions + dispositions[:1] if mutation == "duplicate" else
           dispositions[:-1] if mutation == "missing" else ("forged", *dispositions[1:]))
    with pytest.raises(ValueError, match="endpoint admitted universe dispositions differ"):
        dataclasses.replace(graph.selection, candidate_dispositions=bad, result_sha256="")


@pytest.mark.parametrize("mutation", ["reordered", "relabeled"])
def test_coherently_resealed_dispositions_fail_actual_source_replay(
        full_universe_sources, mutation):
    graph = subject.derive_image_derived_graph(**full_universe_sources)
    assert isinstance(graph, subject.ImageDerivedGraphResult), graph
    dispositions = list(graph.selection.candidate_dispositions)
    kept = dispositions.index(plane_graph.SAME_AXIS_DISPOSITION_KEPT)
    dropped = dispositions.index(plane_graph.SAME_AXIS_DISPOSITION_DROPPED)
    if mutation == "reordered":
        dispositions[kept], dispositions[dropped] = dispositions[dropped], dispositions[kept]
    else:
        dispositions[dropped] = plane_graph.SAME_AXIS_DISPOSITION_KEPT
    certificate = dataclasses.replace(graph.selection, candidate_dispositions=tuple(dispositions), result_sha256="")
    forged = dataclasses.replace(graph, selection=certificate, result_sha256="")
    assert forged.result_sha256 != graph.result_sha256
    assert forged.selection.proof_json == graph.selection.proof_json
    refusal = subject.replay_image_derived_graph(graph_result=forged, **full_universe_sources)
    assert isinstance(refusal, placement.PlacementRefusal), refusal
    assert refusal.reason == "provenance_mismatch" and refusal.detail["field"] == "joint_graph.source_replay"


def test_production_orchestration_delegates_to_graph10_with_fixed_geometry(sources, monkeypatch):
    from mc_photogrammetry.pipeline import image_derived_geometry_orchestration as orchestration
    from mc_photogrammetry.pipeline import image_derived_graph as legacy

    monkeypatch.setattr(f1, "derive_image_derived_signatures", lambda **kwargs: sources["signatures_result"])
    monkeypatch.setattr(legacy, "_derive_image_derived_graph",
        lambda *args, **kwargs: pytest.fail("legacy production graph fallback"))
    counts = _calls(monkeypatch)
    derive = subject.derive_image_derived_graph
    delegated = []

    def observed(**kwargs):
        assert kwargs["max_shift_cells"] == 32
        assert (kwargs["integer_tolerance"], kwargs["min_resultant"], kwargs["ambiguous_margin"]) == (0.15, 0.90, 0.02)
        graph = derive(**kwargs)
        delegated.append(graph)
        return graph

    monkeypatch.setattr(subject, "derive_image_derived_graph", observed)
    result = orchestration.derive_image_derived_geometry(session=sources["session"],
        lattice_result=sources["lattice_result"], connection_graph=sources["connection_graph"])
    assert isinstance(result, orchestration.ImageDerivedGeometryResult), result
    records = dict(result.records)
    assert len(delegated) == 1 and records["g"] is delegated[0]
    graph = records["g"]
    assert graph.schema == "mc-photogrammetry-image-derived-graph-v6"
    assert graph.producer == "m9-g-image-derived-graph-10" and graph.status == "assembled"
    assert graph.selection.status == "unique" and graph.geometry_evaluations == 1
    assert graph.component_solution.scale_known and not graph.component_solution.reason
    assert all(graph.component_solution.axis_resultants[a] >= 0.90
               and graph.component_solution.axis_residuals[a] <= 0.15 for a in range(3))
    assert result.joint_lineage == (graph.schema, graph.producer, graph.result_sha256,
        graph.selection.schema, graph.selection.producer, graph.selection.result_sha256)
    assert records["j"].graph_result is graph and result.status == "unavailable" and result.reason == "k_refused"
    assert counts == {"matcher": 1, "selector": 1, "evaluator": 1}


def test_global_budget_refusal_never_evaluates_partial_selection(sources, monkeypatch):
    counts = _calls(monkeypatch)
    graph = subject.derive_image_derived_graph(**sources, budget=dataclasses.replace(csp.EndpointBudget(), branch_nodes=0))
    assert graph.selection.status == "budget_refused" and graph.geometry_evaluations == 0
    assert graph.matches == () and graph.component_solution is None
    assert graph.selection.reason == "endpoint_budget_refused"
    proof = graph.selection.proof_record()
    assert proof["denial"][0] == "branch_nodes" and proof["denial"][3] == 0
    assert not proof["exhaustive"] and proof["choices"] == []
    assert counts == {"matcher": 1, "selector": 1, "evaluator": 0}


@pytest.mark.parametrize("identity", [("schema", "mc-photogrammetry-image-derived-graph-v2"),
    ("producer", "m9-g-image-derived-graph-5"), ("producer", "m9-g-image-derived-graph-6"),
    ("schema", "mc-photogrammetry-image-derived-graph-v5"), ("producer", "m9-g-image-derived-graph-9")])
def test_stale_and_phase_b_resealed_identity_refused(sources, identity):
    graph = subject.derive_image_derived_graph(**sources)
    with pytest.raises(ValueError):
        dataclasses.replace(graph, **{identity[0]: identity[1]}, result_sha256="")
    old = phase_b.derive_image_derived_graph_v3(**sources)
    assert isinstance(subject.replay_image_derived_graph(graph_result=old, **sources), placement.PlacementRefusal)


def test_fixed_production_gates_cannot_be_relaxed(sources, monkeypatch):
    monkeypatch.setattr(plane_graph, "match_plane_graph", lambda *a, **k: pytest.fail("changed settings reached matcher"))
    result = subject.derive_image_derived_graph(**sources, min_resultant=0.1)
    assert isinstance(result, placement.PlacementRefusal)


def test_geometry_failure_does_not_reconsider_selection(sources, monkeypatch):
    counts = _calls(monkeypatch)
    # The actual numerical kernel still runs; only its numerical output is
    # perturbed to exercise unchanged refusal, not uniqueness or matcher gates.
    original = plane_graph._evaluate_assignment
    def low_resultant(*args, **kwargs):
        cost, alt, detail = original(*args, **kwargs)
        detail = dict(detail, resultants={0: 0.89, 1: 1., 2: 1.})
        return cost, alt, detail
    monkeypatch.setattr(plane_graph, "_evaluate_assignment", low_resultant)
    graph = subject.derive_image_derived_graph(**sources)
    assert graph.reason == "plane_integer_residual" and graph.geometry_evaluations == 1
    assert graph.matches == () and graph.component_solution is None
    assert counts == {"matcher": 1, "selector": 1, "evaluator": 1}


def test_genuine_matcher_joint_failure_binds_all_axis_candidate_rows(sources, monkeypatch):
    original = plane_graph._evaluate_assignment
    def low_resultant(*args, **kwargs):
        cost, alt, detail = original(*args, **kwargs)
        return cost, alt, dict(detail, resultants={0: .89, 1: 1., 2: 1.})
    monkeypatch.setattr(plane_graph, "_evaluate_assignment", low_resultant)
    captures = []
    trace = plane_graph.PlaneGraphTrace([])
    graph = subject.derive_image_derived_graph(
        **sources, same_axis_candidate_sink=captures.append, diagnostic_sink=trace,
        same_axis_capture_axes=subject.legacy.SAME_AXIS_CANDIDATES_AXES_ALL)
    assert graph.reason == "plane_integer_residual"
    assert graph.selection.status == "unique"
    assert len(captures) == 1 and captures[0].recorded, captures
    record = captures[0].to_record()
    subject.legacy.validate_same_axis_candidates(record)
    assert record["schema"] == subject.legacy.SAME_AXIS_CANDIDATES_SCHEMA_V3
    assert record["axes"] == [0, 1, 2]
    assert record["selected_source_rows"] == [r.source_row for r in graph.selection.representatives]
    assert record["input_match_count"] == len(graph.selection.representatives)
    assert record["matcher_capture"]["input_match_count"] == len(graph.candidate_matches)
    assert len(record["admitted_rows"]) >= len(graph.candidate_matches)
    assert trace.post_admission_candidates == len(record["admitted_rows"])
    assert trace.admitted_ledger_sha256 == record["admitted_ledger_sha256"]
    def reseal(body):
        body.pop("record_sha256")
        body["record_sha256"] = subject.legacy._sha({
            "domain": subject.legacy.SAME_AXIS_CANDIDATES_DIGEST_DOMAIN_V3,
            "schema": body["schema"], "payload": body})
        return body
    mutations = []
    foreign = copy.deepcopy(record)
    foreign["selected_source_rows"][0] = len(graph.candidate_matches)
    mutations.append(foreign)
    substituted = copy.deepcopy(record)
    substituted["selected_source_rows"][0] = (
        substituted["selected_source_rows"][0] + 1) % len(graph.candidate_matches)
    mutations.append(substituted)
    reordered = copy.deepcopy(record)
    reordered["selected_source_rows"].reverse()
    mutations.append(reordered)
    changed_identity = copy.deepcopy(record)
    changed_identity["input_identities"][0][7] += 1.
    changed_identity["input_match_identity_sha256"] = subject.legacy._alias_domain_digest(
        subject.legacy.SOLVER_INPUT_BINDINGS_INPUT_MATCH_IDENTITY_DOMAIN,
        subject.legacy.SOLVER_INPUT_BINDINGS_INPUT_MATCH_IDENTITY_SCHEMA,
        changed_identity["input_identities"])
    mutations.append(changed_identity)
    omitted = copy.deepcopy(record)
    omitted["admitted_rows"].pop()
    mutations.append(omitted)
    missing_proof = copy.deepcopy(record)
    missing_proof["selection_payload"]["proof_json"] = "{}"
    missing_proof["selection_sha256"] = subject.legacy._sha(
        missing_proof["selection_payload"])
    mutations.append(missing_proof)
    changed_score = copy.deepcopy(record)
    changed_score["selection_payload"]["representatives"][0][2][12] += .01
    changed_score["selection_sha256"] = subject.legacy._sha(
        changed_score["selection_payload"])
    mutations.append(changed_score)
    changed_shift = copy.deepcopy(record)
    changed_shift["selection_payload"]["representatives"][0][2][15][0] += 1
    changed_shift["selection_sha256"] = subject.legacy._sha(
        changed_shift["selection_payload"])
    mutations.append(changed_shift)
    recursive_matcher = copy.deepcopy(record)
    recursive_matcher["matcher_capture"]["schema"] = subject.legacy.SAME_AXIS_CANDIDATES_SCHEMA_V3
    mutations.append(recursive_matcher)
    for mutation in mutations:
        with pytest.raises(ValueError):
            subject.legacy.validate_same_axis_candidates(reseal(mutation))


def test_joint_capture_refuses_omitted_matcher_admitted_row(sources, monkeypatch):
    original_matcher = plane_graph.match_plane_graph
    original_evaluator = plane_graph._evaluate_assignment
    def omit_observation_row(*args, **kwargs):
        matches = original_matcher(*args, **kwargs)
        sink = kwargs["candidate_observation_sink"]
        observation = sink[0]
        # The matcher-owned ledger says a dropped admitted row existed, while
        # the observation supplied to the collector omits that row entirely.
        omitted = dataclasses.replace(observation.candidates[-1], score=.01)
        kwargs["pre_filter_witness_sink"].append(plane_graph._pre_filter_witness_row(
            omitted, admitted_row_index=len(observation.candidates),
            filtered_index={}, candidate_witnesses={},
            signature_sha256_by_identity=None))
        return matches
    def low_resultant(*args, **kwargs):
        cost, alt, detail = original_evaluator(*args, **kwargs)
        return cost, alt, dict(detail, resultants={0: .89, 1: 1., 2: 1.})
    monkeypatch.setattr(plane_graph, "match_plane_graph", omit_observation_row)
    monkeypatch.setattr(plane_graph, "_evaluate_assignment", low_resultant)
    captures = []
    graph = subject.derive_image_derived_graph(
        **sources, same_axis_candidate_sink=captures.append,
        same_axis_capture_axes=subject.legacy.SAME_AXIS_CANDIDATES_AXES_ALL)
    assert graph.reason == "plane_integer_residual"
    assert len(captures) == 1 and not captures[0].recorded
    assert captures[0].field == "candidate_rows"


def _certificate(graph, candidates, **changes):
    values = dict(readings=graph.readings, signatures=graph.signatures,candidate_matches=candidates,
        frame_ids=graph.frame_ids,quadrant_constraints=graph.selection.quadrant_constraints,
        budget=graph.selection.budget,session_sha256=graph.session_sha256,lattice_sha256=graph.lattice_sha256,
        signatures_sha256=graph.signatures_sha256,connection_sha256=graph.connection_sha256)
    values.update(changes)
    return selection.select_endpoint_hypotheses(**values)


def _record_with_certificate(graph, cert, candidates):
    matches=tuple(r.match for r in cert.representatives)
    solution=subject._evaluate_selected(graph.readings,matches,cert) if cert.status == "unique" else None
    assert solution is None or not solution.reason
    return dataclasses.replace(graph,selection=cert,candidate_matches=candidates,
        status="assembled" if solution is not None else "unavailable",
        reason="" if solution is not None else cert.reason,detail="" if solution is not None else cert.status,
        matches=matches,component_solution=solution,geometry_evaluations=int(solution is not None),result_sha256="")


@pytest.mark.parametrize("change",["omit","reorder","score","duplicate","constraints","budget"])
def test_resealed_source_forgery_cannot_pass_explicit_replay(sources,change):
    graph=subject.derive_image_derived_graph(**sources)
    candidates=graph.candidate_matches
    alterations={}
    if change == "omit": candidates=candidates[:-1]
    elif change == "reorder": candidates=tuple(reversed(candidates))
    elif change == "score": candidates=(dataclasses.replace(candidates[0],score=candidates[0].score+0.001),)+candidates[1:]
    elif change == "duplicate": candidates=candidates+(candidates[0],)
    elif change == "constraints": alterations["quadrant_constraints"]=()
    elif change == "budget": alterations["budget"]=dataclasses.replace(graph.selection.budget,branch_nodes=graph.selection.budget.branch_nodes-1)
    cert=_certificate(graph,candidates,**alterations)
    if change == "duplicate": assert cert.status == graph.selection.status
    forged=_record_with_certificate(graph,cert,candidates)
    assert forged.result_sha256 != graph.result_sha256
    assert isinstance(subject.replay_image_derived_graph(graph_result=forged,**sources),placement.PlacementRefusal)


def test_genuine_multiple_cannot_reseal_a_geometry_winner(monkeypatch):
    sources=_source_records(monkeypatch,alternatives=True)
    graph=subject.derive_image_derived_graph(**sources)
    assert graph.selection.status == "multiple"
    proof=graph.selection.proof_record()
    proof.update(status="unique",exhaustive=True,choices=proof["choices"][:1])
    proof["counters"]["solutions_found"]=proof["counters"]["solution_rows"]=1
    selected={(f,i):a for f,i,a in proof["choices"][0]["alternatives"]}
    indexed,_,_=plane_graph._canonical_reading_index(graph.readings)
    representatives=[]
    for n,factor in enumerate(proof["choices"][0]["factors"]):
        row_index=factor["source_rows"][0];row=graph.candidate_matches[row_index]
        canonical=dataclasses.replace(row,
          left_b=indexed[(row.left,row.left_plane)].alternatives[selected[(row.left,row.left_plane)]].blocks_per_period,
          right_b=indexed[(row.right,row.right_plane)].alternatives[selected[(row.right,row.right_plane)]].blocks_per_period)
        representatives.append(selection.RepresentativeConstraint(n,row_index,canonical))
    cert=dataclasses.replace(graph.selection,status="unique",reason="joint_geometry_not_evaluated",
        proof_json=selection._encode(proof).decode(),representatives=tuple(representatives),result_sha256="")
    counts=_calls(monkeypatch)
    with pytest.raises(ValueError, match="physical proof does not recompute"):
        selection.validate_candidate_certificate(cert,graph.readings,graph.signatures,graph.candidate_matches)
    with pytest.raises(ValueError, match="physical proof does not recompute"):
        dataclasses.replace(graph,selection=cert,result_sha256="")
    assert counts == {"matcher":0,"selector":0,"evaluator":0}
    # A bypassed constructor also fails replay before any source or geometry work.
    forged=object.__new__(type(graph))
    for field in dataclasses.fields(graph):
        object.__setattr__(forged,field.name,cert if field.name == "selection" else getattr(graph,field.name))
    assert isinstance(subject.replay_image_derived_graph(graph_result=forged,**sources),placement.PlacementRefusal)
    assert counts == {"matcher":0,"selector":0,"evaluator":0}


def test_changed_alternative_signature_index_frames_and_nested_identity_refuse(sources):
    graph=subject.derive_image_derived_graph(**sources)
    for changes in ({"frame_ids":("a",)}, {"readings":tuple(reversed(graph.readings))},
                    {"signatures":graph.signatures[:-1]}):
        with pytest.raises(ValueError):dataclasses.replace(graph,**changes,result_sha256="")
    with pytest.raises(ValueError):
        dataclasses.replace(graph.selection,producer="m9-g-endpoint-selection-0",result_sha256="")
    proof=graph.selection.proof_record();proof["choices"][0]["alternatives"][0][2]=99
    with pytest.raises(ValueError):
        dataclasses.replace(graph.selection,proof_json=selection._encode(proof).decode(),result_sha256="")


def test_unresolved_evidence_is_not_a_geometry_choice(monkeypatch):
    from test_image_derived_endpoint_selection import _fixture
    cert,readings,evidence,rows=_fixture()
    ambiguous=(rows[0],dataclasses.replace(rows[0],cell_shift_rc=(1,0)))+rows[1:]
    unresolved=selection.select_endpoint_hypotheses(readings=readings,signatures=evidence,
        candidate_matches=ambiguous,frame_ids=cert.frame_ids,quadrant_constraints=cert.quadrant_constraints,
        budget=cert.budget,session_sha256=cert.session_sha256,lattice_sha256=cert.lattice_sha256,
        signatures_sha256=cert.signatures_sha256)
    assert unresolved.status == "unresolved_evidence" and unresolved.representatives == ()
    monkeypatch.setattr(plane_graph,"_evaluate_assignment",lambda *a,**k:pytest.fail("unresolved evidence reached geometry"))
    with pytest.raises(ValueError):subject._evaluate_selected(readings,(),unresolved)


def _numerical_refusal(monkeypatch):
    original=plane_graph._evaluate_assignment
    def low_resultant(*args,**kwargs):
        cost,alternative,detail=original(*args,**kwargs)
        return cost,alternative,dict(detail,resultants={0:0.89,1:1.,2:1.})
    monkeypatch.setattr(plane_graph,"_evaluate_assignment",low_resultant)


@pytest.mark.parametrize("option",[
    "alias_max_readings","alias_max_matches","alias_max_active_edges",
    "alias_max_endpoint_alternatives","alias_max_frame_id_bytes","alias_max_record_bytes"])
@pytest.mark.parametrize("limit",[None,0,True,"invalid",-1])
def test_attribution_alias_limits_are_bounded_and_output_neutral(sources,monkeypatch,option,limit):
    from mc_photogrammetry.pipeline import image_derived_graph_attribution as attribution
    _numerical_refusal(monkeypatch)
    counts=_calls(monkeypatch)
    off=subject.derive_image_derived_graph(**sources)
    observed=[]
    options={} if limit is None else {option:limit}
    on,record=attribution.derive_image_derived_graph_attribution(**sources,
        solver_failure_observer=observed.append,**options)
    assert isinstance(on,subject.ImageDerivedGraphResult),on
    assert off.reason == on.reason == "plane_integer_residual"
    assert on.result_sha256 == off.result_sha256
    assert counts == {"matcher":2,"selector":2,"evaluator":2}
    assert len(observed)==1 and record.solver_failure is not None
    diagnostic=record.solver_failure.to_record()
    assert diagnostic["solver_id"] == "plane_graph._evaluate_assignment"
    aliases=observed[0].active_edge_aliases.to_record()
    assert aliases["status"] == ("complete" if limit is None else "refused")
    if limit is not None:
        assert aliases["reason"] and not aliases.get("edges")


def test_failure_observer_attachment_preserves_attribution_digest(sources,monkeypatch):
    from mc_photogrammetry.pipeline import image_derived_graph_attribution as attribution
    _numerical_refusal(monkeypatch)
    counts=_calls(monkeypatch)
    off,off_record=attribution.derive_image_derived_graph_attribution(**sources)
    observed=[]
    on,on_record=attribution.derive_image_derived_graph_attribution(**sources,
        solver_failure_observer=observed.append)
    assert on.result_sha256 == off.result_sha256
    assert on_record.attribution_sha256 == off_record.attribution_sha256
    assert len(observed)==1 and counts == {"matcher":2,"selector":2,"evaluator":2}


@pytest.mark.parametrize("exhausted",[False,True])
def test_matcher_witness_budget_preserves_graph_output(sources,monkeypatch,exhausted):
    from mc_photogrammetry.pipeline import image_derived_graph_attribution as attribution
    from mc_photogrammetry.reconstruction import lattice
    counts=_calls(monkeypatch)
    off=subject.derive_image_derived_graph(**sources)
    witnesses=[]
    budget=lattice.WitnessBudgetV1(max_pair_entries=0) if exhausted else lattice.WitnessBudgetV1()
    on,record=attribution.derive_image_derived_graph_attribution(**sources,
        matcher_witness_sink=witnesses,witness_budget=budget)
    assert isinstance(on,subject.ImageDerivedGraphResult),on
    assert on.result_sha256 == off.result_sha256 and record is not None
    assert witnesses and counts == {"matcher":2,"selector":2,"evaluator":2}
    assert any(isinstance(w,lattice.WitnessRefusalV1) for w in witnesses) is exhausted


@pytest.mark.parametrize("disposition",["no","multiple","budget_refused"])
def test_zero_evaluator_selection_refusal_is_attributed_to_joint_selection(sources,monkeypatch,disposition):
    from mc_photogrammetry.pipeline import image_derived_graph_attribution as attribution
    if disposition == "multiple":sources=_source_records(monkeypatch,alternatives=True)
    if disposition == "no":
        from test_image_derived_joint_geometry import _source_with_empty_axis
        sources=_source_with_empty_axis(sources,None)
    options={"budget":dataclasses.replace(csp.EndpointBudget(),branch_nodes=0)} if disposition == "budget_refused" else {}
    counts=_calls(monkeypatch)
    graph,record=attribution.derive_image_derived_graph_attribution(**sources,**options)
    assert graph.selection.status == disposition and graph.geometry_evaluations == 0
    assert record.first_failing_substage == "joint_selection" and record.solver_failure is None
    assert counts == {"matcher":1,"selector":1,"evaluator":0}


def _pinned_pre_extraction_graph(monkeypatch):
    """Load only the reviewed before bytes, never a reopened/pyc baseline module."""
    path = Path(__file__).resolve().parents[1] / (
        "agent swarm instantiation template/runs/"
        "SOL-GRAPH10-SHARED-EVALUATOR-01.before.image_derived_joint_graph.py")
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "60f4f55bc887bef50c60427a00fca54f5ad854a1785ebd5b1319c4b823df25ab"
    module = ModuleType("_graph10_shared_evaluator_before")
    module.__file__ = str(path)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("state", ["unique", "numerical", "nonfinite", "guard", "no", "multiple", "budget", "tampered"])
def test_shared_evaluator_preserves_pinned_production_payload_counts_and_sinks(sources, monkeypatch, state):
    from test_image_derived_endpoint_selection import _forge

    baseline = _pinned_pre_extraction_graph(monkeypatch)
    source = _source_records(monkeypatch, alternatives=True) if state == "multiple" else sources
    if state == "no":
        from test_image_derived_joint_geometry import _source_with_empty_axis
        source = _source_with_empty_axis(sources, None)
    options = {"budget": dataclasses.replace(csp.EndpointBudget(), branch_nodes=0)} if state == "budget" else {}
    if state == "numerical":
        _numerical_refusal(monkeypatch)
    elif state in ("nonfinite", "guard"):
        # Perturb only a genuine evaluated component; retain the real guards.
        for module in (baseline, subject):
            original = module._evaluate_selected
            def perturb(*args, _fn=original, **kwargs):
                component = _fn(*args, **kwargs)
                assert not component.reason
                if state == "guard":
                    return dataclasses.replace(component, blocks_per_period={})
                translations = dict(component.translations)
                frame = component.frames[-1]
                translations[frame] = (float("nan"), *translations[frame][1:])
                return dataclasses.replace(component, translations=translations)
            monkeypatch.setattr(module, "_evaluate_selected", perturb)
    elif state == "tampered":
        selector = selection.select_endpoint_hypotheses
        def tampered(**kwargs):
            certificate = selector(**kwargs)
            assert certificate.status == "unique"
            return _forge(certificate, candidate_matches_sha256="0" * 64)
        monkeypatch.setattr(selection, "select_endpoint_hypotheses", tampered)

    counts = _calls(monkeypatch)
    helper = subject._evaluate_selection
    outcomes = []
    def observe_helper(**kwargs):
        outcome = helper(**kwargs)
        outcomes.append(outcome)
        return outcome
    monkeypatch.setattr(subject, "_evaluate_selection", observe_helper)

    def invoke(module):
        from mc_photogrammetry.pipeline.image_derived_graph_attribution import solver_failure_snapshot_record

        captured = {name: [] for name in ("solver", "same_axis", "support", "complete")}
        trace = plane_graph.PlaneGraphTrace([])
        previous = dict(counts)
        result = module.derive_image_derived_graph(**source, **options, diagnostic_sink=trace,
            solver_failure_sink=captured["solver"].append,
            same_axis_candidate_sink=captured["same_axis"].append,
            candidate_support_sink=captured["support"].append,
            complete_support_sink=captured["complete"].append,
            same_axis_capture_axes=subject.legacy.SAME_AXIS_CANDIDATES_AXES_ALL)
        evaluations = int(state in ("unique", "numerical", "nonfinite", "guard"))
        assert {name: counts[name] - previous[name] for name in counts} == {
            "matcher": 1, "selector": 1, "evaluator": evaluations}
        assert {name: len(values) for name, values in captured.items()} == {
            "solver": int(state == "numerical"),
            "same_axis": int(state in ("numerical", "nonfinite", "guard")),
            "support": int(state in ("numerical", "nonfinite", "guard")),
            "complete": int(state in ("numerical", "nonfinite", "guard"))}
        records = {name: [value.to_record() for value in values]
                   for name, values in captured.items() if name != "solver"}
        records["solver"] = [{
            "snapshot": solver_failure_snapshot_record(value),
            "captures": {field: (getattr(value, field).to_record() if getattr(value, field) is not None else None)
                         for field in ("active_edge_aliases", "boundary_saturation", "phase_capture", "solver_input_bindings")},
        } for value in captured["solver"]]
        return result, dataclasses.asdict(trace), records

    before, before_trace, before_sinks = invoke(baseline)
    assert outcomes == []
    after, after_trace, after_sinks = invoke(subject)
    assert before_trace == after_trace and before_sinks == after_sinks
    if state == "tampered":
        assert isinstance(before, placement.PlacementRefusal) and before == after
        assert before.reason == "provenance_mismatch" and outcomes == []
        return
    assert isinstance(before, baseline.ImageDerivedGraphResult)
    assert isinstance(after, subject.ImageDerivedGraphResult)
    assert baseline._payload(before) == subject._payload(after)
    assert before.result_sha256 == after.result_sha256
    assert len(outcomes) == 1 and outcomes[0].calls == after.geometry_evaluations
    assert outcomes[0].matches == after.matches and outcomes[0].solution is after.component_solution
    expected = {"unique": ("assembled", ""), "numerical": ("unavailable", "plane_integer_residual"),
        "nonfinite": ("unavailable", subject.legacy.NON_FINITE_SOLVER_OUTPUT),
        "guard": ("unavailable", "graph_endpoint_scale_conflict")}
    if state in expected:
        assert (after.status, after.reason) == expected[state]
    else:
        assert after.selection.status == {"no": "no", "multiple": "multiple", "budget": "budget_refused"}[state]
        assert outcomes[0].selected == () and outcomes[0].evaluated is None
    if state == "unique":
        assert outcomes[0].matches is outcomes[0].selected
        assert outcomes[0].solution is outcomes[0].evaluated
    else:
        assert outcomes[0].matches == () and outcomes[0].solution is None


def test_shared_evaluator_has_frozen_result_and_exact_guard_order_without_rematching(sources, monkeypatch):
    graph = subject.derive_image_derived_graph(**sources)
    assert graph.status == "assembled"
    counts = _calls(monkeypatch)
    monkeypatch.setattr(plane_graph, "match_plane_graph", lambda *a, **k: pytest.fail("helper rematched"))
    monkeypatch.setattr(selection, "select_endpoint_hypotheses", lambda *a, **k: pytest.fail("helper reselected"))
    events = []
    for module, name, event in ((selection, "validate_candidate_certificate", "certificate"),
                               (subject, "_evaluate_selected", "geometry"),
                               (subject.legacy, "_non_finite_solution_field", "nonfinite"),
                               (subject.legacy, "_production_solution_refusal", "production_guard")):
        original = getattr(module, name)
        def observed(*args, _fn=original, _event=event, **kwargs):
            events.append(_event)
            return _fn(*args, **kwargs)
        monkeypatch.setattr(module, name, observed)
    result = subject._evaluate_selection(readings=graph.readings, signatures=graph.signatures,
        candidate_matches=graph.candidate_matches, frame_ids=graph.frame_ids, certificate=graph.selection)
    assert events == ["certificate", "geometry", "nonfinite", "production_guard"]
    assert counts == {"matcher": 0, "selector": 0, "evaluator": 1}
    assert result.reason == result.detail == "" and result.calls == 1
    assert result.matches is result.selected and result.solution is result.evaluated
    assert result.selected == tuple(rep.match for rep in graph.selection.representatives)
    assert subject.legacy._solution_payload(result.solution) == subject.legacy._solution_payload(graph.component_solution)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.reason = "forged"


@pytest.mark.parametrize("mutation", ["certificate_digest", "candidate_digest", "representative_source_row"])
def test_shared_evaluator_tampering_refuses_before_numerics_or_guards(sources, monkeypatch, mutation):
    from test_image_derived_endpoint_selection import _forge

    graph = subject.derive_image_derived_graph(**sources)
    assert graph.selection.status == "unique"
    certificate = graph.selection
    if mutation == "certificate_digest":
        certificate = _forge(certificate, result_sha256="0" * 64)
    elif mutation == "candidate_digest":
        certificate = _forge(certificate, candidate_matches_sha256="0" * 64)
    else:
        representative = dataclasses.replace(certificate.representatives[0], source_row=len(graph.candidate_matches))
        certificate = _forge(certificate, representatives=(representative, *certificate.representatives[1:]))
    for module, name in ((plane_graph, "match_plane_graph"), (selection, "select_endpoint_hypotheses"),
                         (subject, "_evaluate_selected"), (subject.legacy, "_non_finite_solution_field"),
                         (subject.legacy, "_production_solution_refusal")):
        monkeypatch.setattr(module, name, lambda *a, **k: pytest.fail("tampering passed certificate replay"))
    with pytest.raises(ValueError):
        subject._evaluate_selection(readings=graph.readings, signatures=graph.signatures,
            candidate_matches=graph.candidate_matches, frame_ids=graph.frame_ids, certificate=certificate)
