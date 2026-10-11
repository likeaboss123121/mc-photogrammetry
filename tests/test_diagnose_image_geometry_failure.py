"""Synthetic, same-call evidence controls; no captured images or evaluator data."""
from dataclasses import asdict, replace
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evaluation import diagnose_image_geometry_failure as subject
from mc_photogrammetry.pipeline import image_derived_entry as entry_mod
from mc_photogrammetry.pipeline import image_derived_graph as graph_mod
from mc_photogrammetry.pipeline import image_derived_graph_attribution as trace_mod
from mc_photogrammetry.pipeline import image_derived_lattice_measure as lattice_mod
from mc_photogrammetry.pipeline import image_derived_signatures as signatures_mod
from mc_photogrammetry.pipeline.frame_ingress import build_frame_ingress_provenance
from mc_photogrammetry.pipeline.contracts import FrameTransform
from mc_photogrammetry.pipeline.image_derived_session import image_sha256
from mc_photogrammetry.reconstruction import plane_graph
from test_candidate_support_provenance import _png_bytes
from test_image_derived_endpoint_selection import _evidence, _fixture as _endpoint_fixture
from test_same_axis_candidate_search import control

_SOURCE_SOLVE_COMPONENT = plane_graph.solve_component
_SOURCE_EVALUATE_ASSIGNMENT = plane_graph._evaluate_assignment


def test_endpoint_conflict_is_joint_selection_abstention_without_geometry(harness, tmp_path, monkeypatch):
    """Competing same-endpoint hypotheses now abstain inside the joint selector.

    The current joint graph never reaches the fixed-Q evaluator when two
    canonical endpoint hypotheses survive, so there is no published
    ``endpoint_period_conflict`` graph reason and the endpoint observer stays
    honestly unvisited instead of closing a fabricated witness.
    """
    from mc_photogrammetry.reconstruction.lattice_planes import DistanceAlternative
    readings = tuple(replace(r, alternatives=(*r.alternatives,
                    DistanceAlternative(1./16, r.alternatives[0].distance_blocks/16, 1.)))
                     for r in harness.readings)
    supplied = (*harness.matches, replace(harness.matches[0], left_b=1./16))
    harness.lattice.readings = readings
    harness.signatures.signatures = _evidence(readings)
    def match(readings_arg, signatures_arg, **kwargs):
        harness.counts["matcher"] += 1
        assert readings_arg == readings and len(signatures_arg) == len(harness.signatures.signatures)
        trace = kwargs["diagnostic_sink"]
        trace.pre_admission_candidates = trace.post_admission_candidates = len(supplied)
        trace.post_same_axis_candidates = len(supplied)
        kwargs["admitted_candidate_sink"].append(plane_graph.AdmittedCandidateUniverse(
            tuple(supplied), ("kept",) * len(supplied), tuple(range(len(supplied)))))
        return supplied
    monkeypatch.setattr(plane_graph, "match_plane_graph", match)
    output = subject.run(images=harness.paths, output=tmp_path / "endpoint-conflict")
    record = json.loads(Path(output["path"]).read_text())
    assert record["status"] == "incomplete"
    assert record["graph"]["payload"]["reason"] == "multiple_endpoint_hypotheses"
    assert record["graph"]["payload"]["geometry_evaluations"] == 0
    assert record["solver_failure_snapshot"] is None
    assert all(harness.counts[name] == 1 for name in harness.counts if name != "solver")
    assert harness.counts["solver"] == 0
    witness = record["joint_endpoint_selection"]
    assert witness["status"] == "incomplete"
    assert witness["selection_status"] == "multiple"
    assert witness["proof_exhaustive"] is False
    assert witness["proof_counters"]["solutions_found"] == 2
    assert witness["source_bindings"]["graph_sha256"] == record["graph"]["result_sha256"]
    assert "joint_endpoint_selection:incomplete" in record["issues"]


def test_endpoint_observer_off_on_parity_and_unique_closure(harness, tmp_path):
    """Joint observation closes the exhaustive certificate without changing G."""
    off = subject._run_image_chain(harness.paths, tmp_path / "off.mcscan", lambda _: None)
    harness.entry.project_lease.discard()
    on = subject._run_image_chain(harness.paths, tmp_path / "on.mcscan", lambda _: None,
                                  joint_endpoint_observer=True)
    harness.entry.project_lease.discard()
    _, inputs = subject._input_records(harness.paths)
    off_record, on_record = subject._evidence(off, inputs), subject._evidence(on, inputs)
    assert off_record["stage_digests"] == on_record["stage_digests"]
    assert off_record["graph"] == on_record["graph"]
    assert off_record["attribution"] == on_record["attribution"]
    assert off_record["same_axis_candidates"] == on_record["same_axis_candidates"]
    assert "endpoint_period_conflicts" not in off_record
    assert harness.counts == {name: 2 for name in harness.counts}
    captured = on_record["joint_endpoint_selection"]
    assert captured["status"] == "complete"
    assert captured["selection_status"] == "unique"
    assert captured["proof_exhaustive"] is True
    assert captured["proof_counters"]["unvisited_assignments"] == 0
    assert "joint_endpoint_selection:incomplete" not in on_record["issues"]
    assert captured["source_bindings"]["lattice_sha256"] == on_record["stage_digests"]["lattice"]
    assert len(subject._bounded_bytes(captured)) < subject.MAX_JOINT_ENDPOINT_BYTES
    body = dict(captured); body.pop("record_sha256")
    assert subject._digest(body) == captured["record_sha256"]


def test_endpoint_observer_resource_refusal_is_atomic(harness, tmp_path, monkeypatch):
    """The observer's byte cap is a real, source-bound refusal that rolls back.

    The cap is consumed by the numerical kernel's assignment accounting, so the
    refusal is produced at that kernel seam and then injected back through the
    runner's observer hook to prove the diagnostic still refuses atomically.
    """
    readings, matches = control()
    monkeypatch.setattr(plane_graph, "_evaluate_assignment", _SOURCE_EVALUATE_ASSIGNMENT)
    monkeypatch.setattr(plane_graph._EndpointConflictObserver, "MAX_BYTES", 1)
    observer = plane_graph._EndpointConflictObserver()
    with plane_graph._endpoint_conflict_context(observer):
        solution = _SOURCE_SOLVE_COMPONENT(
            readings, matches, quadrant_constraints=(("a", "b", 0),))
    assert solution.reason == "" and set(solution.quadrants) == {"a", "b"}
    refused = observer.record()
    assert refused["status"] == "refused" and refused["reason"]
    with pytest.raises(subject.DiagnosticError, match="endpoint_period_conflicts") as error:
        subject._validate_endpoint_conflicts(refused)
    assert error.value.reason == "evidence_cap"
    monkeypatch.setattr(subject, "MAX_JOINT_ENDPOINT_BYTES", 1)
    output = tmp_path / "resource-refused"
    with pytest.raises(subject.DiagnosticError, match="joint_endpoint_observation") as error:
        subject.run(images=harness.paths, output=output)
    assert error.value.reason == "evidence_cap"
    assert not output.exists()


def test_endpoint_missing_hook_and_false_complete_closure(harness, tmp_path):
    capture_result, inputs = capture(harness, tmp_path)
    missing = plane_graph._EndpointConflictObserver().record()
    capture_result.endpoint_conflicts = missing
    record = subject._evidence(capture_result, inputs)
    assert record["endpoint_period_conflicts"]["status"] == "incomplete"
    assert "endpoint_period_conflicts:incomplete" in record["issues"]
    false_complete = dict(missing, status="complete")
    with pytest.raises(subject.DiagnosticError, match="endpoint_conflict_closure"):
        subject._validate_endpoint_conflicts(false_complete)


def _joint_observation_fixture(rows=None, budget=None):
    """A real selector certificate in an inert graph shell; no image chain."""
    from mc_photogrammetry.pipeline import image_derived_joint_graph as joint
    from mc_photogrammetry.reconstruction.endpoint_hypothesis import EndpointBudget

    cert, readings, evidence, candidates = _endpoint_fixture(
        rows, EndpointBudget() if budget is None else budget)
    graph = object.__new__(joint.ImageDerivedGraphResult)
    for name, value in (("selection", cert), ("readings", readings),
                        ("signatures", evidence), ("candidate_matches", candidates),
                        ("result_sha256", "f" * 64)):
        object.__setattr__(graph, name, value)
    return graph


def _partial_witness_fixture(*, conflict=False, reverse=False, quadrant=0,
                             retained_cap=False, mixed=False, unconstrained=False):
    from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
    from mc_photogrammetry.pipeline import image_derived_joint_graph as joint
    from mc_photogrammetry.reconstruction.endpoint_hypothesis import EndpointBudget
    from test_joint_endpoint_displacement_contract import (
        _forced_adversarial_fixture, _mixed_cycle_fixture, _seven_endpoint_fixture,
        _evidence as full_evidence, _reverse)

    if mixed:
        readings, candidates = _mixed_cycle_fixture()
    elif retained_cap:
        readings, candidates = _seven_endpoint_fixture(second_left_x_distance=9.0)
        readings = tuple(replace(r, alternatives=r.alternatives * 16)
                         if r.frame == "a" and r.axis == 0 else r for r in readings)
    else:
        readings, candidates = _forced_adversarial_fixture(conflict, free_count=1, alternatives=3)
    if quadrant:
        assert not mixed
        old = plane_graph._canonical_reading_index(readings)[0]
        readings = tuple(replace(r, axis={0: 2, 1: 1, 2: 0}[r.axis],
            side=float(plane_graph.axis_transform({0: 2, 1: 1, 2: 0}[r.axis], quadrant)[1]))
            if r.frame == "b" else r for r in readings)
        _, identities, _ = plane_graph._canonical_reading_index(readings)
        positions = {(r.frame, r.period_pixels): identities[id(r)][1] for r in readings}
        candidates = tuple(replace(row,
            left_plane=positions[(row.left, old[(row.left, row.left_plane)].period_pixels)],
            right_plane=positions[(row.right, old[(row.right, row.right_plane)].period_pixels)],
            right_axis={0: 2, 1: 1, 2: 0}[row.right_axis], quadrant_of_right=quadrant)
            for row in candidates)
    if reverse:
        candidates = tuple(_reverse(row) for row in candidates)
    if unconstrained:
        candidates += tuple(replace(row, quadrant_of_right=2) for row in candidates)
    frames = tuple(sorted({r.frame for r in readings}))
    constraints = (() if unconstrained else
                   tuple((frames[0], frame, quadrant) for frame in frames[1:]))
    evidence = full_evidence(readings)
    cert = selection.select_endpoint_hypotheses(readings=readings, signatures=evidence,
        candidate_matches=candidates, frame_ids=frames, quadrant_constraints=constraints,
        budget=EndpointBudget(), session_sha256="a" * 64, lattice_sha256="b" * 64,
        signatures_sha256="c" * 64)
    graph = object.__new__(joint.ImageDerivedGraphResult)
    for name, value in (("selection", cert), ("readings", readings),
                        ("signatures", evidence), ("candidate_matches", candidates),
                        ("result_sha256", "f" * 64)):
        object.__setattr__(graph, name, value)
    return graph


@pytest.mark.parametrize("conflict", (False, True))
def test_partial_witness_v2_exact_validated_projection_and_deterministic_bytes(conflict):
    from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
    graph = _partial_witness_fixture(conflict=conflict)
    selection.validate_candidate_certificate(graph.selection, graph.readings, graph.signatures, graph.candidate_matches)
    record = subject._joint_endpoint_observation(graph)
    proof = graph.selection.proof_record()
    assert record["schema"] == "mc-photogrammetry-joint-endpoint-observation-v3"
    assert record["selection_status"] == ("no" if conflict else "unique")
    for name in ("transcript_policy", "physical_policy", "physical_tolerance", "physical_digest",
                 "partial_exclusions", "physical_exclusions"):
        assert record[name] == proof[name]
    assert record["representative_counterexample"] == {
        "status": "representative_counterexample_only", "partial_exclusion_index": 0}
    assert record["retained_example_coverage"] == sorted({event[1] for event in proof["partial_exclusions"]})
    assert subject._bounded_bytes(record) == subject._bounded_bytes(subject._joint_endpoint_observation(graph))
    assert subject._validate_joint_endpoint_observation(record, graph) == record
    assert subject._validate_joint_endpoint_observation(json.loads(subject._bounded_bytes(record)), graph) == record
    assert [row["source_row"] for row in record["witness_rows"]] == sorted(set(
        ref[0] for event in proof["partial_exclusions"] for ref in event[-1][-2:]))
    for source in record["witness_rows"]:
        row = graph.candidate_matches[source["source_row"]]
        assert source["row_sha256"] == hashlib.sha256(selection._encode(
            selection.legacy._match_payload(row), limit=graph.selection.budget.scalar_row_bytes)).hexdigest()
        assert "score" not in source and "psr" not in source and "cell_shift_rc" not in source


@pytest.mark.parametrize("reverse", (False, True))
def test_partial_witness_nonzero_q_and_reversed_original_alternatives_resolve_exactly(reverse):
    graph = _partial_witness_fixture(reverse=reverse, quadrant=1)
    record = subject._joint_endpoint_observation(graph)
    assert record["selection_status"] == "unique" and record["partial_exclusions"]
    indexed = plane_graph._canonical_reading_index(graph.readings)[0]
    bindings = {tuple(row[0]): row for row in graph.selection.reading_bindings}
    for example, event in zip(record["resolved_extrema"], record["partial_exclusions"]):
        assert example["quadrants"] == [["a", 0], ["b", 1]]
        assert example["assignment_index"] == event[1]
        assert example["frame_pair"] == event[-1][0][:2]
        for label, ref in zip(("low", "high"), event[-1][-2:]):
            witness = example[label]
            row = graph.candidate_matches[ref[0]]
            a, b = (ref[1], ref[2]) if not reverse else (ref[2], ref[1])
            assert witness["original_left_alternative_index"] == a
            assert witness["original_right_alternative_index"] == b
            relation = plane_graph.signed_physical_relation(indexed, row, {"a": 0, "b": 1},
                {(row.left, row.left_plane): a, (row.right, row.right_plane): b}, ref[0], compact=True)
            assert relation[1] == witness["delta"] == ref[4]
    for alternative in record["witness_alternatives"]:
        endpoint, a = tuple(alternative["endpoint"]), alternative["alternative_index"]
        binding, reading = bindings[endpoint], indexed[endpoint]
        assert alternative["reading_index"] == binding[1]
        assert graph.readings[binding[1]] is reading
        assert alternative["local_axis"] == reading.axis and alternative["side"] == reading.side
        assert alternative["blocks_per_period"] == reading.alternatives[a].blocks_per_period
        assert alternative["distance_blocks"] == reading.alternatives[a].distance_blocks
        assert alternative["reading_binding_sha256"] == binding[3]
        assert alternative["signature_sha256"] == binding[5][a]
    assert any(row["side"] == -1.0 for row in record["witness_alternatives"])
    subject._validate_joint_endpoint_observation(record, graph)


@pytest.mark.parametrize("mixed", (False, True))
def test_partial_witness_max_128_shared_retention_and_omission_closure(mixed):
    graph = _partial_witness_fixture(mixed=mixed, retained_cap=not mixed)
    record = subject._joint_endpoint_observation(graph)
    counts = record["proof_counters"]
    assert len(record["partial_exclusions"]) + len(record["physical_exclusions"]) == 128
    for kind, total in (("partial", "partial_prunes"), ("physical", "physical_rejected")):
        observed = record[kind + "_exclusion_counts"]
        assert observed == {"total": counts[total], "retained": counts["retained_" + kind + "_exclusions"],
                            "omitted": counts["omitted_" + kind + "_exclusions"]}
        assert observed["retained"] + observed["omitted"] == observed["total"]
    if mixed:
        assert record["partial_exclusion_counts"] == {"total": 1, "retained": 1, "omitted": 0}
        assert record["physical_exclusion_counts"] == {"total": 192, "retained": 127, "omitted": 65}
    else:
        assert record["partial_exclusion_counts"] == {"total": 256, "retained": 128, "omitted": 128}
    assert len(record["witness_rows"]) <= 256 and len(record["witness_alternatives"]) <= 512
    subject._validate_joint_endpoint_observation(record, graph)


def test_partial_witness_absence_is_unavailable_and_q_coverage_is_only_retained():
    graph = _partial_witness_fixture(conflict=True, retained_cap=True, unconstrained=True)
    record = subject._joint_endpoint_observation(graph)
    assert record["proof_counters"]["planned_assignments"] == 4
    assert record["proof_counters"]["eligible_assignments"] == 2
    assert record["retained_example_coverage"] == [0]
    assert len(record["resolved_extrema"]) == 128
    empty = subject._joint_endpoint_observation(_joint_observation_fixture(rows=()))
    assert empty["selection_status"] == "no" and empty["proof_counters"]["physical_assignments"] == 0
    assert empty["witness_availability"] == "unavailable" and empty["representative_counterexample"] is None
    assert empty["resolved_extrema"] == empty["witness_rows"] == empty["witness_alternatives"] == []


@pytest.mark.parametrize("damage", ("row", "q", "index", "alternative", "distance", "sign", "tolerance",
    "kind", "path", "mask", "source", "proof", "digest", "omitted", "representative", "schema"))
def test_partial_witness_resealed_observation_tampering_refuses(damage):
    graph = _partial_witness_fixture()
    record = subject._joint_endpoint_observation(graph)
    changed = copy.deepcopy(record)
    if damage == "row": changed["witness_rows"][0]["row_sha256"] = "0" * 64
    elif damage == "q": changed["resolved_extrema"][0]["quadrants"][1] = ("b", 2)
    elif damage == "index": changed["resolved_extrema"][0]["assignment_index"] += 1
    elif damage == "alternative": changed["witness_alternatives"][0]["alternative_index"] += 1
    elif damage == "distance": changed["witness_alternatives"][0]["distance_blocks"] += 1.0
    elif damage == "sign": changed["resolved_extrema"][0]["low"]["delta"] *= -1
    elif damage == "tolerance": changed["physical_tolerance"] = 0.16
    elif damage == "kind": changed["partial_exclusions"][0][0] = "leaf"
    elif damage == "path": changed["partial_exclusions"][0][2][0][1] += 1
    elif damage == "mask": changed["partial_exclusions"][0][3][0] = 3
    elif damage == "source": changed["source_bindings"]["graph_sha256"] = "0" * 64
    elif damage == "proof": changed["proof_sha256"] = "0" * 64
    elif damage == "digest": changed["physical_digest"] = "0" * 64
    elif damage == "omitted": changed["partial_exclusion_counts"]["omitted"] += 1
    elif damage == "representative": changed["representative_counterexample"]["partial_exclusion_index"] = 1
    else: changed["schema"] = "mc-photogrammetry-joint-endpoint-observation-v1"
    changed.pop("record_sha256")
    changed["record_sha256"] = subject._digest(changed)
    with pytest.raises(subject.DiagnosticError): subject._validate_joint_endpoint_observation(changed, graph)


def test_partial_witness_consistently_resealed_false_certificate_requires_existing_replay():
    from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
    graph = _partial_witness_fixture()
    proof = graph.selection.proof_record()
    for name in ("partial_checks", "partial_prunes", "omitted_partial_exclusions",
                 "search_events_attempted", "search_events_committed"):
        proof["counters"][name] += 1
    forged = replace(graph.selection, proof_json=selection._encode(proof).decode(), result_sha256="")
    object.__setattr__(graph, "selection", forged)
    with pytest.raises(subject.DiagnosticError, match="joint_endpoint_certificate"):
        subject._joint_endpoint_observation(graph)


def test_witness_projection_uses_only_existing_validation_csp_replay(monkeypatch):
    from mc_photogrammetry.reconstruction import endpoint_hypothesis as csp
    graph = _partial_witness_fixture()
    actual, calls = csp.solve_endpoint_hypotheses, []
    def counted(*args, **kwargs):
        calls.append(1)
        return actual(*args, **kwargs)
    monkeypatch.setattr(csp, "solve_endpoint_hypotheses", counted)
    record = subject._joint_endpoint_observation(graph)
    assert len(calls) == 1
    subject._validate_joint_endpoint_observation(record, graph)
    assert len(calls) == 2
    monkeypatch.setattr(csp, "solve_endpoint_hypotheses", lambda *_a, **_k: pytest.fail("projection searched CSP"))
    assert subject._joint_resolved_extrema(graph, graph.selection.proof_record())[0] == record["resolved_extrema"]


def test_witness_planner_is_one_traversal_stopping_at_last_requested_index(monkeypatch):
    from mc_photogrammetry.reconstruction import endpoint_hypothesis as csp
    graph = _partial_witness_fixture(conflict=True, unconstrained=True)
    proof = graph.selection.proof_record()
    actual, calls, yielded = plane_graph._quadrant_assignment_plan, [], []
    def counted(*args):
        calls.append(1)
        reason, count, plans = actual(*args)
        def observed():
            for index, q in enumerate(plans):
                yielded.append(index)
                yield q
        return reason, count, observed()
    monkeypatch.setattr(plane_graph, "_quadrant_assignment_plan", counted)
    monkeypatch.setattr(csp, "solve_endpoint_hypotheses", lambda *_a, **_k: pytest.fail("planner searched CSP"))
    maps = subject._joint_witness_quadrants(graph.selection, proof, {0})
    assert calls == [1] and yielded == [0] and maps == {0: {"a": 0, "b": 0}}
    calls.clear(); yielded.clear()
    maps = subject._joint_witness_quadrants(graph.selection, proof, {1, 3})
    assert calls == [1] and yielded == [0, 1, 2, 3]
    assert maps == {1: {"a": 0, "b": 1}, 3: {"a": 0, "b": 3}}


@pytest.mark.parametrize("damage", ("count", "foreign", "missing", "duplicate", "requested_cap", "planner_cap"))
def test_witness_q_mapping_refuses_malformed_or_over_cap_plans(monkeypatch, damage):
    graph = _partial_witness_fixture(conflict=True, unconstrained=True)
    proof = graph.selection.proof_record()
    requested = {0, 1}
    if damage == "requested_cap":
        proof["counters"]["planned_assignments"] = 16384
        requested = set(range(129))
    elif damage == "planner_cap":
        proof["counters"]["planned_assignments"] = 16385
    else:
        plans = ({"a": 0, "b": 0}, {"a": 0, "b": 1})
        count = 4
        if damage == "count": count = 3
        elif damage == "foreign": plans = ({"a": 0, "foreign": 0}, plans[1])
        elif damage == "missing": plans = plans[:1]
        else: plans = (plans[0], plans[0])
        monkeypatch.setattr(plane_graph, "_quadrant_assignment_plan", lambda *_a: (None, count, iter(plans)))
    with pytest.raises(subject.DiagnosticError):
        subject._joint_witness_quadrants(graph.selection, proof, requested)


def test_witness_duplicate_rows_and_same_b_alternative_slots_preserve_indices():
    from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
    graph = _partial_witness_fixture(retained_cap=True)
    candidates = graph.candidate_matches + (graph.candidate_matches[0],)
    cert = selection.select_endpoint_hypotheses(readings=graph.readings, signatures=graph.signatures,
        candidate_matches=candidates, frame_ids=graph.selection.frame_ids,
        quadrant_constraints=graph.selection.quadrant_constraints, budget=graph.selection.budget,
        session_sha256="a" * 64, lattice_sha256="b" * 64, signatures_sha256="c" * 64)
    object.__setattr__(graph, "selection", cert)
    object.__setattr__(graph, "candidate_matches", candidates)
    record = subject._joint_endpoint_observation(graph)
    duplicate_index = len(candidates) - 1
    assert any(duplicate_index in factor[-1] and 0 in factor[-1]
               for event in record["partial_exclusions"] for factor in event[4])
    # Sparse rows contain only actual extrema, with the deterministic earliest
    # original row as the tie witness. Exact duplicated support remains above.
    assert duplicate_index not in [row["source_row"] for row in record["witness_rows"]]
    alternatives = record["witness_alternatives"]
    identities = [(tuple(a["endpoint"]), a["alternative_index"]) for a in alternatives]
    assert identities == sorted(set(identities))
    assert len(alternatives) > len({(a["blocks_per_period"], a["distance_blocks"]) for a in alternatives})


@pytest.mark.parametrize("damage", ("candidate", "reading", "side", "proof"))
def test_witness_graph_source_tampering_refuses(damage):
    graph = _partial_witness_fixture()
    record = subject._joint_endpoint_observation(graph)
    if damage == "candidate": object.__setattr__(graph, "candidate_matches", tuple(reversed(graph.candidate_matches)))
    elif damage in ("reading", "side"):
        first = graph.readings[0]
        changed = (replace(first, side=-first.side) if damage == "side" else
                   replace(first, alternatives=(replace(first.alternatives[0], distance_blocks=123.0),)))
        object.__setattr__(graph, "readings", (changed,) + graph.readings[1:])
    else: object.__setattr__(graph.selection, "proof_json", graph.selection.proof_json.replace('"partial"', '"leaf"', 1))
    with pytest.raises(subject.DiagnosticError): subject._validate_joint_endpoint_observation(record, graph)


def test_nested_four_mib_limit_includes_digest_and_refuses_before_source_replay(monkeypatch):
    from mc_photogrammetry.reconstruction import endpoint_hypothesis as csp
    graph = _partial_witness_fixture()
    record = subject._joint_endpoint_observation(graph)
    limit = subject.MAX_JOINT_ENDPOINT_BYTES
    changed = dict(record, presentation_padding="")
    changed.pop("record_sha256")
    changed["record_sha256"] = subject._digest(changed)
    overhead = len(subject._bounded_bytes(changed, limit=limit))
    changed["presentation_padding"] = "x" * (limit - overhead)
    body = dict(changed); body.pop("record_sha256")
    changed["record_sha256"] = subject._digest(body)
    assert len(subject._bounded_bytes(changed, limit=limit)) == limit
    # An exact-limit but invented presentation field reaches source equality,
    # where it must refuse. One extra byte refuses before touching the source.
    with pytest.raises(subject.DiagnosticError, match="joint_endpoint_certificate"):
        subject._validate_joint_endpoint_observation(changed, graph)
    changed["presentation_padding"] += "x"
    monkeypatch.setattr(csp, "solve_endpoint_hypotheses", lambda *_a, **_k: pytest.fail("oversize record replayed"))
    with pytest.raises(subject.DiagnosticError) as error:
        subject._validate_joint_endpoint_observation(changed, graph)
    assert error.value.reason == "evidence_cap"


def test_outer_sixteen_mib_and_item_depth_limits_remain_exact():
    limit = subject.MAX_RECORD_BYTES
    overhead = len(subject._bounded_bytes({"padding": ""}))
    record = {"padding": "x" * (limit - 1 - overhead)}
    assert len(subject._bounded_bytes(record)) + 1 == limit
    record["padding"] += "x"
    with pytest.raises(subject.DiagnosticError): subject._bounded_bytes(record)
    assert subject._bounded_bytes([None] * (subject.MAX_JSON_ITEMS - 1))
    with pytest.raises(subject.DiagnosticError): subject._bounded_bytes([None] * subject.MAX_JSON_ITEMS)
    deep = None
    for _ in range(subject.MAX_JSON_DEPTH): deep = [deep]
    assert subject._bounded_bytes(deep)
    with pytest.raises(subject.DiagnosticError): subject._bounded_bytes([deep])


def test_partial_witness_exact_local_record_limit_and_atomic_publication_refusal(harness, tmp_path, monkeypatch):
    graph = _partial_witness_fixture()
    record = subject._joint_endpoint_observation(graph)
    size = len(subject._bounded_bytes(record))
    monkeypatch.setattr(subject, "MAX_JOINT_ENDPOINT_BYTES", size)
    assert subject._joint_endpoint_observation(graph) == record
    monkeypatch.setattr(subject, "MAX_JOINT_ENDPOINT_BYTES", size - 1)
    with pytest.raises(subject.DiagnosticError): subject._joint_endpoint_observation(graph)
    # Refusal from witness projection travels through the existing staging
    # rollback. The production graph and synthetic input bytes are untouched.
    actual = subject._joint_endpoint_observation
    monkeypatch.setattr(subject, "_joint_endpoint_observation", lambda _graph: actual(graph))
    output = tmp_path / "witness_cap_refusal"
    with pytest.raises(subject.DiagnosticError): subject.run(images=harness.paths, output=output)
    assert not output.exists() and all(path.is_file() for path in harness.paths)


def test_extended_observer_production_parity_counts_validation_replays_separately(harness, tmp_path, monkeypatch):
    from mc_photogrammetry.reconstruction import endpoint_hypothesis as csp
    from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
    actual_solve, actual_validate = csp.solve_endpoint_hypotheses, selection.validate_candidate_certificate
    calls = {"production_search": 0, "validation_replay": 0}
    validating = []
    def counted_solve(*args, **kwargs):
        calls["validation_replay" if validating else "production_search"] += 1
        return actual_solve(*args, **kwargs)
    def counted_validate(*args, **kwargs):
        validating.append(True)
        try: return actual_validate(*args, **kwargs)
        finally: validating.pop()
    monkeypatch.setattr(csp, "solve_endpoint_hypotheses", counted_solve)
    monkeypatch.setattr(selection, "validate_candidate_certificate", counted_validate)
    off = subject._run_image_chain(harness.paths, tmp_path / "off_v2.mcscan", lambda _: None)
    harness.entry.project_lease.discard()
    before = dict(calls)
    on = subject._run_image_chain(harness.paths, tmp_path / "on_v2.mcscan", lambda _: None, joint_endpoint_observer=True)
    harness.entry.project_lease.discard()
    assert calls["production_search"] == 2 * before["production_search"] == 2
    assert calls["validation_replay"] == 2 * before["validation_replay"] + 1
    assert off.graph == on.graph and off.attribution == on.attribution
    assert harness.counts == {name: 2 for name in harness.counts}
    _, inputs = subject._input_records(harness.paths)
    off_record, on_record = subject._evidence(off, inputs), subject._evidence(on, inputs)
    assert off_record["stage_digests"] == on_record["stage_digests"]
    assert off_record["graph"] == on_record["graph"]
    assert calls["production_search"] == 2
    assert calls["validation_replay"] == 2 * before["validation_replay"] + 2


@pytest.mark.parametrize("state", ("no", "unique", "multiple",
                                   "unresolved_evidence", "budget_refused"))
def test_joint_endpoint_observation_terminal_accounting(state):
    from mc_photogrammetry.reconstruction.endpoint_hypothesis import EndpointBudget
    from test_endpoint_hypothesis import connected_fixture

    _, original = connected_fixture("a", "b")
    rows, budget = tuple(original), EndpointBudget()
    if state == "no":
        rows = ()
    elif state == "multiple":
        rows += tuple(replace(row, quadrant_of_right=2) for row in rows)
    elif state == "unresolved_evidence":
        rows += (replace(rows[0], cell_shift_rc=(1, 2)),)
    elif state == "budget_refused":
        budget = replace(budget, branch_nodes=0)
    graph = _joint_observation_fixture(rows, budget)
    record = subject._joint_endpoint_observation(graph)
    assert record["selection_status"] == state
    assert record["status"] == {"no": "complete", "unique": "complete",
                                "multiple": "incomplete", "unresolved_evidence": "incomplete",
                                "budget_refused": "refused"}[state]
    assert record["candidate_count"] == len(rows)
    assert record["proof_counters"]["input_match_rows"] <= len(rows)
    proof = graph.selection.proof_record()
    assert record["eligibility_reasons"] == proof["eligibility_reasons"]
    assert record["excluded_assignments"] == proof["excluded_assignments"]
    assert len(record["excluded_assignments"]) == record["proof_counters"]["retained_exclusions"]
    assert sum(record["eligibility_reasons"].values()) == record["proof_counters"]["excluded_assignments"]
    assert subject._validate_joint_endpoint_observation(record, graph) == record
    if state in ("no", "unique", "unresolved_evidence"):
        assert record["proof_exhaustive"] is True
        assert record["proof_counters"]["unvisited_assignments"] == 0
    else:
        assert record["proof_exhaustive"] is False


def test_joint_endpoint_refused_work_preserves_unclassified_counts_and_exclusions():
    """Nonzero unclassified work and retained rows survive verbatim.

    The reference charge is the established relation_visits exhaustion control
    from ``tests/test_endpoint_hypothesis.py``: a budget equal to the constrained
    solution's exact relation charge. At that exact budget the selector finds one
    eligible assignment and then refuses on the next, so a visited assignment is
    never classified. Adding the following assignment's registry sweep lets that
    assignment reach its exclusion classification first, leaving a nonempty
    histogram and a retained row beside the same unclassified work.
    """
    from mc_photogrammetry.reconstruction.endpoint_hypothesis import (
        EndpointBudget, solve_endpoint_hypotheses)
    from test_endpoint_hypothesis import connected_fixture, fixed

    readings, rows = connected_fixture("a", "b")
    forced = solve_endpoint_hypotheses(readings, rows, session_frames=("a", "b"),
                                       quadrant_constraints=fixed("a", "b"))
    assert forced.status == "unique" and forced.counters["solutions_found"] == 1
    reference = forced.counters["relation_visits"]
    cases = (
        # extra charge, exact histogram, visited, hashed, unvisited
        (0, {}, 2, 1, 2),
        (forced.counters["registry_factors"], {"no_active_factors": 1}, 3, 2, 1),
    )
    for extra, reasons, visited, hashed, unvisited in cases:
        budget = replace(EndpointBudget(), relation_visits=reference + extra)
        control = solve_endpoint_hypotheses(readings, rows, session_frames=("a", "b"),
                                            budget=budget)
        assert control.status == "budget_refused" and control.choices == ()
        assert control.denial[0] == "relation_visits" and not control.exhaustive
        assert control.counters["eligibility_unclassified_assignments"] == 1
        assert control.counters["visited_assignments"] == visited
        assert control.counters["eligibility_hashed_assignments"] == hashed
        assert control.counters["unvisited_assignments"] == unvisited
        assert dict(control.eligibility_reasons) == reasons

        graph = _joint_observation_fixture(rows, budget)
        record = subject._joint_endpoint_observation(graph)
        proof = graph.selection.proof_record()
        assert graph.selection.status == "budget_refused"
        assert record["status"] == "refused" and record["proof_exhaustive"] is False
        assert record["proof_counters"] == proof["counters"] == dict(control.counters)
        assert record["proof_counters"]["eligibility_unclassified_assignments"] == 1
        assert record["eligibility_reasons"] == proof["eligibility_reasons"] == reasons
        assert record["excluded_assignments"] == proof["excluded_assignments"] == json.loads(
            json.dumps(control.excluded_assignments))
        assert len(record["excluded_assignments"]) == record["proof_counters"]["retained_exclusions"]
        assert record["eligibility_digest"] == proof["eligibility_digest"] == control.eligibility_digest
        assert subject._validate_joint_endpoint_observation(record, graph) == record


def test_joint_endpoint_no_observation_exposes_multiple_reason_classes():
    from test_endpoint_hypothesis import connected_fixture

    _, original = connected_fixture("a", "b")
    graph = _joint_observation_fixture(tuple(original[:1]))
    record = subject._joint_endpoint_observation(graph)
    assert record["status"] == "complete" and record["selection_status"] == "no"
    assert record["eligibility_reasons"] == {
        "unsupported_reading_endpoint": 1, "no_active_factors": 3}
    assert [row[2] for row in record["excluded_assignments"]] == [
        row[2] for row in graph.selection.proof_record()["excluded_assignments"]]
    assert subject._validate_joint_endpoint_observation(record, graph) == record


def test_joint_endpoint_exclusion_retention_and_omission_below_byte_cap():
    from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
    from mc_photogrammetry.reconstruction.endpoint_hypothesis import EndpointBudget
    from test_endpoint_hypothesis import readings_for

    frames = tuple(f"f{i}" for i in range(5))
    readings = tuple(readings_for(*frames, alternatives=1))
    evidence = _evidence(readings)
    cert = selection.select_endpoint_hypotheses(
        readings=readings, signatures=evidence, candidate_matches=(),
        frame_ids=frames, quadrant_constraints=(), budget=EndpointBudget(),
        session_sha256="a" * 64, lattice_sha256="b" * 64, signatures_sha256="c" * 64)
    from mc_photogrammetry.pipeline import image_derived_joint_graph as joint
    graph = object.__new__(joint.ImageDerivedGraphResult)
    for name, value in (("selection", cert), ("readings", readings),
                        ("signatures", evidence), ("candidate_matches", ()),
                        ("result_sha256", "f" * 64)):
        object.__setattr__(graph, name, value)
    record = subject._joint_endpoint_observation(graph)
    assert record["eligibility_reasons"] == {"no_active_factors": 256}
    assert record["proof_counters"]["retained_exclusions"] == 128
    assert record["proof_counters"]["omitted_exclusions"] == 128
    assert len(record["excluded_assignments"]) == 128
    assert record["excluded_assignments"] == cert.proof_record()["excluded_assignments"]
    assert len(subject._bounded_bytes(record)) <= subject.MAX_JOINT_ENDPOINT_BYTES
    assert subject._validate_joint_endpoint_observation(record, graph) == record


def test_joint_endpoint_complete_record_byte_cap_exact_boundary(monkeypatch):
    graph = _joint_observation_fixture()
    record = subject._joint_endpoint_observation(graph)
    encoded_bytes = len(subject._bounded_bytes(record))
    assert encoded_bytes > len(subject._bounded_bytes(
        {key: value for key, value in record.items() if key != "record_sha256"}))
    monkeypatch.setattr(subject, "MAX_JOINT_ENDPOINT_BYTES", encoded_bytes)
    assert subject._joint_endpoint_observation(graph) == record
    assert subject._validate_joint_endpoint_observation(record, graph) == record
    monkeypatch.setattr(subject, "MAX_JOINT_ENDPOINT_BYTES", encoded_bytes - 1)
    with pytest.raises(subject.DiagnosticError, match="joint_endpoint_observation") as generated:
        subject._joint_endpoint_observation(graph)
    with pytest.raises(subject.DiagnosticError, match="joint_endpoint_observation") as replayed:
        subject._validate_joint_endpoint_observation(record, graph)
    assert generated.value.reason == replayed.value.reason == "evidence_cap"


def test_joint_endpoint_raw_conflicting_rows_can_still_close_unique():
    from test_endpoint_hypothesis import connected_fixture

    _, original = connected_fixture("a", "b")
    rows = tuple(original) + (replace(original[0], left_b=1. + 5e-10),)
    graph = _joint_observation_fixture(rows)
    assert graph.candidate_matches[-1].left_b != graph.candidate_matches[0].left_b
    record = subject._joint_endpoint_observation(graph)
    assert record["status"] == "complete" and record["selection_status"] == "unique"
    assert "conflicted_assignments" not in record
    assert record["choices"][0]["factor_source_rows"][0] == [0, 3]


@pytest.mark.parametrize("field", ("selection_sha256", "candidate_matches_sha256",
                                   "proof_counters", "choices", "representative_source_rows",
                                   "eligibility_reasons", "excluded_assignments"))
def test_joint_endpoint_resealed_certificate_or_row_observation_refuses(field):
    graph = _joint_observation_fixture()
    record = subject._joint_endpoint_observation(graph)
    changed = copy.deepcopy(record)
    if field in ("selection_sha256", "candidate_matches_sha256"):
        changed[field] = "0" * 64
    elif field == "proof_counters":
        changed[field]["visited_assignments"] += 1
    elif field == "choices":
        changed[field][0]["factor_source_rows"][0] = [1]
    elif field == "eligibility_reasons":
        changed[field]["no_active_factors"] += 1
    elif field == "excluded_assignments":
        changed[field][0][2] = "forged_reason"
    else:
        changed[field][0] = 1
    changed.pop("record_sha256")
    changed["record_sha256"] = subject._digest(changed)
    with pytest.raises(subject.DiagnosticError, match="joint_endpoint_certificate"):
        subject._validate_joint_endpoint_observation(changed, graph)
    old = graph.candidate_matches
    object.__setattr__(graph, "candidate_matches", old[:-1])
    with pytest.raises(subject.DiagnosticError, match="joint_endpoint_certificate"):
        subject._validate_joint_endpoint_observation(record, graph)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    readings, matches = control()
    evidence = _evidence(readings)
    paths = []
    frames = []
    for frame_id, colour in (("a", (100, 60, 20)), ("b", (30, 80, 120))):
        data = _png_bytes(colour)
        path = tmp_path / (frame_id + ".png")
        path.write_bytes(data)
        paths.append(path)
        pixels = np.full((12, 12, 3), colour, dtype=np.uint8)
        ingress = build_frame_ingress_provenance(
            frame_id=frame_id, source_label=str(path), encoded_bytes=data,
            decoded_pixels=pixels, decoded_pixel_sha256=image_sha256(pixels),
            transform=FrameTransform(source_size=(12, 12)))
        frames.append(SimpleNamespace(frame_id=frame_id, ingress=ingress))
    session = SimpleNamespace(session_sha256="a" * 64, frames=tuple(frames))
    connections = SimpleNamespace(status="ok", graph_sha256="b" * 64,
                                  session_sha256=session.session_sha256)
    entry = SimpleNamespace(entry_sha256="c" * 64, session=session, graph=connections,
                            pose_graph=SimpleNamespace(status="ok"), project_lease=None)
    lattice = SimpleNamespace(status="assembled", result_sha256="d" * 64,
                              session_sha256=session.session_sha256,
                              frame_ids=("a", "b"), readings=readings)
    signatures = SimpleNamespace(status="assembled", result_sha256="e" * 64,
                                 session_sha256=session.session_sha256,
                                 lattice_sha256=lattice.result_sha256,
                                 frame_ids=("a", "b"), signatures=evidence)
    counts = {name: 0 for name in ("entry", "lattice", "signatures", "graph", "matcher", "solver")}

    class Lease:
        def __init__(self, path):
            self.path, self.active = path, True
            path.write_bytes(b"synthetic staging project")

        def discard(self):
            if self.active:
                self.path.unlink()
                self.active = False

    def build(paths_arg, **kwargs):
        counts["entry"] += 1
        assert tuple(paths_arg) == tuple(paths)
        assert kwargs["capture_ingress"] is True
        assert kwargs["project_max_bytes"] == subject.MAX_PROJECT_BYTES
        assert kwargs["retain_project_lease"] is True
        entry.project_lease = Lease(kwargs["project_path"])
        return entry

    def measure(**kwargs):
        counts["lattice"] += 1
        assert kwargs["session"] is session
        return lattice

    def signature(**kwargs):
        counts["signatures"] += 1
        assert kwargs["session"] is session
        return signatures

    def match(readings_arg, signatures_arg, **kwargs):
        counts["matcher"] += 1
        assert readings_arg == readings
        assert len(signatures_arg) == len(evidence)
        trace = kwargs["diagnostic_sink"]
        trace.pre_admission_candidates = trace.post_admission_candidates = len(matches)
        trace.post_same_axis_candidates = len(matches)
        kwargs["admitted_candidate_sink"].append(plane_graph.AdmittedCandidateUniverse(
            tuple(matches), ("kept",) * len(matches), tuple(range(len(matches)))))
        sink = kwargs.get("candidate_observation_sink")
        if sink is not None:
            # The genuine matcher-produced observation binds its own candidate
            # objects, exactly as production does; the joint graph then reports
            # whatever identity binding the selected representatives allow.
            sink.append(plane_graph.SameAxisCandidateObservation(
                matches, ("kept",) * len(matches)))
        pre_filter = kwargs.get("pre_filter_witness_sink")
        if pre_filter is not None:
            for index, match in enumerate(matches):
                pre_filter.append(plane_graph._pre_filter_witness_row(
                    match, admitted_row_index=index,
                    filtered_index={id(row): position for position, row in enumerate(matches)},
                    candidate_witnesses={}, signature_sha256_by_identity=None))
        trace.admitted_ledger_sha256 = plane_graph.admitted_candidate_ledger_sha256(
            (index, "kept", index, hashlib.sha256(json.dumps(
                plane_graph._indexed_match_payload(match), sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest())
            for index, match in enumerate(matches))
        return matches

    def evaluate(frames_arg, quadrants, selected, indexed):
        # Only the numerical evaluator is replaced. The joint selector, the
        # candidate/reading/signature bindings and every payload digest stay real.
        counts["solver"] += 1
        assert frames_arg == ("a", "b")
        assert quadrants == {"a": 0, "b": 0}
        assert tuple(selected) == tuple(
            replace(match, left_b=1., right_b=1.) for match in matches)
        value = _SOURCE_EVALUATE_ASSIGNMENT(frames_arg, quadrants, selected, indexed)
        cost, alternative, detail = value
        detail = dict(detail)
        detail["resultants"] = {0: .5, 1: .95, 2: .95}
        detail["residuals"] = {0: .1, 1: 0., 2: 0.}
        return cost, alternative, detail

    original = trace_mod.derive_image_derived_graph_attribution

    def graph(**kwargs):
        counts["graph"] += 1
        return original(**kwargs)

    monkeypatch.setattr(entry_mod, "build_image_derived_entry", build)
    monkeypatch.setattr(lattice_mod, "measure_image_derived_lattice", measure)
    monkeypatch.setattr(signatures_mod, "derive_image_derived_signatures", signature)
    monkeypatch.setattr(graph_mod, "_replay_session", lambda value: value)
    monkeypatch.setattr(graph_mod, "_replay", lambda value, *_: value)
    monkeypatch.setattr(graph_mod, "_connection_constraints", lambda *_: (("a", "b", 0),))
    monkeypatch.setattr(plane_graph, "match_plane_graph", match)
    monkeypatch.setattr(plane_graph, "_evaluate_assignment", evaluate)
    monkeypatch.setattr(trace_mod, "derive_image_derived_graph_attribution", graph)
    return SimpleNamespace(paths=tuple(paths), entry=entry, lattice=lattice,
                           signatures=signatures, counts=counts, original=original,
                           readings=readings, matches=matches)


def capture(harness, tmp_path):
    result = subject._run_image_chain(harness.paths, tmp_path / "staging.mcscan", lambda _: None)
    harness.entry.project_lease.discard()
    _, inputs = subject._input_records(harness.paths)
    return result, inputs


def test_one_chain_pass_serializes_exact_failure_all_axes_and_constraints(harness, tmp_path):
    result = subject.run(images=harness.paths, output=tmp_path / "output")
    assert harness.counts == {name: 1 for name in harness.counts}
    record = json.loads(Path(result["path"]).read_text())
    # Synthetic source records and a synthetic candidate producer never emit the
    # scalar attrition hooks or the endpoint-conflict kernel events, so those
    # accounting receipts stay explicitly incomplete while the geometry evidence
    # itself closes on the genuine joint selector and injected evaluator failure.
    assert result["status"] == record["status"] == "incomplete"
    assert record["vertical_attrition"]["status"] == "incomplete"
    assert "vertical_attrition:incomplete" in record["issues"]
    # The matcher-owned admitted ledger binds the original candidate objects;
    # the joint selector's copied representatives retain explicit source rows.
    same_axis = record["same_axis_candidates"]
    assert same_axis["status"] == "complete"
    assert same_axis["schema"] == graph_mod.SAME_AXIS_CANDIDATES_SCHEMA_V3
    assert same_axis["axes"] == [0, 1, 2]
    assert len(same_axis["admitted_rows"]) == len(matches := harness.matches)
    assert len(same_axis["selected_source_rows"]) == same_axis["input_match_count"]
    assert same_axis["matcher_capture"]["status"] == "complete"
    assert "same_axis_candidates:refused" not in record["issues"]
    assert record["solve_context"]["constraints"] == [{"left": "a", "right": "b", "yaw": 0}]
    assert record["solver_failure_snapshot"]["axis_resultants"] == [
        {"axis": 0, "value": .5}, {"axis": 1, "value": .95}, {"axis": 2, "value": .95}]
    assert record["axis_metrics"][0] == {"axis": 0, "resultant": .5, "residual": .1, "observations": 2}
    assert record["callback_counts"] == {"same_axis_candidates": 1, "solver_failure": 1}
    assert record["graph"]["result_sha256"] == record["attribution"]["graph_result_sha256"]
    assert record["reconstruction_authorized"] is False
    assert record["alternatives"]["supported_complete_alternative_established"] is False
    assert sorted(p.name for p in (tmp_path / "output").iterdir()) == ["diagnostic.json"]
    body = dict(record)
    body.pop("record_sha256")
    assert subject._digest(body) == record["record_sha256"]


def test_same_axis_collector_binds_genuine_source_candidates_at_direct_seam():
    """Positive collector control at its own valid, source-bound seam.

    The collector receives the exact matcher-produced candidate objects together
    with the solution the kernel solved from those same objects, so retained-row
    identity is genuine. This is the direct collector seam only; no synthetic
    observation is promoted into the joint graph's production evidence.
    """
    readings, matches = control()
    solution = _SOURCE_SOLVE_COMPONENT(
        readings, matches, quadrant_constraints=(("a", "b", 0),))
    assert solution.reason == "" and solution.quadrants == {"a": 0, "b": 0}
    observation = plane_graph.SameAxisCandidateObservation(
        matches, ("kept",) * len(matches))
    record = graph_mod.failed_same_axis_candidates(
        readings, matches, solution, observation,
        graph_mod.SAME_AXIS_CANDIDATES_AXES_ALL)
    assert record.recorded is True, record.refusal_reason
    payload = record.to_record()
    graph_mod.validate_same_axis_candidates(payload)
    assert payload["status"] == "complete"
    assert payload["axes"] == [0, 1, 2]
    assert payload["row_count"] == payload["group_count"] == 3
    assert payload["input_match_count"] == 3
    assert all(row[17] == "kept" for row in payload["candidates"])
    assert record.to_record() == payload
    assert _SOURCE_SOLVE_COMPONENT(readings, matches,
                                   quadrant_constraints=(("a", "b", 0),)) == solution


def test_existing_observation_sink_invariant_on_synthetic_same_call(harness, tmp_path):
    diagnosed, _ = capture(harness, tmp_path)
    normal_graph, normal_trace = harness.original(
        session=harness.entry.session, lattice_result=harness.lattice,
        signatures_result=harness.signatures, connection_graph=harness.entry.graph,
        **subject.DEFAULT_SETTINGS)
    assert diagnosed.graph == normal_graph
    assert diagnosed.attribution == normal_trace
    assert diagnosed.graph.result_sha256 == normal_graph.result_sha256
    assert diagnosed.attribution.attribution_sha256 == normal_trace.attribution_sha256
    assert harness.counts["solver"] == harness.counts["matcher"] == 2


@pytest.mark.parametrize("field", ("failure", "same_axis"))
def test_callback_multiplicity_refuses(field):
    sink = subject._SingleCapture(field)
    sink.append(object())
    with pytest.raises(subject.DiagnosticError, match="callback_multiplicity"):
        sink.append(object())


@pytest.mark.parametrize("mutation", ("candidate_digest", "context_digest", "graph_digest", "foreign_assignment"))
def test_tampering_refuses_before_output(harness, tmp_path, mutation):
    diagnosed, inputs = capture(harness, tmp_path)
    if mutation == "candidate_digest":
        object.__setattr__(diagnosed.same_axis.value, "record_sha256", "0" * 64)
    elif mutation == "context_digest":
        object.__setattr__(diagnosed.failure.value.solve_context, "record_sha256", "0" * 64)
    elif mutation == "graph_digest":
        object.__setattr__(diagnosed.graph, "result_sha256", "0" * 64)
    else:
        diagnosed.failure.value = replace(diagnosed.failure.value, quadrants=(0, 2))
    with pytest.raises((subject.DiagnosticError, ValueError)):
        subject._evidence(diagnosed, inputs)


@pytest.mark.parametrize("mutation", ("omission", "same_count_substitution"))
def test_resealed_inactive_dropped_row_refuses_against_matcher_trace(
        harness, tmp_path, monkeypatch, mutation):
    original = plane_graph.match_plane_graph
    def matcher_with_inactive_dropped(*args, **kwargs):
        matches = original(*args, **kwargs)
        observation = kwargs["candidate_observation_sink"][0]
        dropped = replace(observation.candidates[-1], quadrant_of_right=1, score=.01)
        kwargs["candidate_observation_sink"][0] = plane_graph.SameAxisCandidateObservation(
            observation.candidates + (dropped,),
            observation.dispositions + (plane_graph.SAME_AXIS_DISPOSITION_DROPPED,))
        witness = plane_graph._pre_filter_witness_row(
            dropped, admitted_row_index=len(observation.candidates),
            filtered_index={}, candidate_witnesses={},
            signature_sha256_by_identity=None)
        kwargs["pre_filter_witness_sink"].append(witness)
        trace = kwargs["diagnostic_sink"]
        trace.post_admission_candidates += 1
        trace.admitted_ledger_sha256 = plane_graph.admitted_candidate_ledger_sha256(
            (row.admitted_row_index, row.disposition, row.post_match_index,
             row.match_sha256) for row in kwargs["pre_filter_witness_sink"])
        return matches
    monkeypatch.setattr(plane_graph, "match_plane_graph", matcher_with_inactive_dropped)
    diagnosed, inputs = capture(harness, tmp_path)
    honest = subject._evidence(diagnosed, inputs)
    assert honest["same_axis_candidates"]["status"] == "complete"
    original_capture = diagnosed.same_axis.value
    assert original_capture.admitted_rows[-1][1] == plane_graph.SAME_AXIS_DISPOSITION_DROPPED
    rows = copy.deepcopy(list(original_capture.admitted_rows))
    if mutation == "omission":
        rows.pop()
    else:
        row = list(rows[-1])
        row[3]["score"] = .02
        row[4] = graph_mod._sha(row[3])
        rows[-1] = tuple(row)
    forged = replace(
        original_capture, admitted_rows=tuple(rows),
        admitted_ledger_sha256=plane_graph.admitted_candidate_ledger_sha256(
            (row[0], row[1], row[2], row[4]) for row in rows),
        record_sha256="")
    graph_mod.validate_same_axis_candidates(forged.to_record())
    diagnosed.same_axis.value = forged
    with pytest.raises(subject.DiagnosticError, match="candidate_admitted_ledger"):
        subject._evidence(diagnosed, inputs)


def test_resealed_foreign_selection_digest_refuses_against_joint_graph(harness, tmp_path):
    diagnosed, inputs = capture(harness, tmp_path)
    original_capture = diagnosed.same_axis.value
    foreign = copy.deepcopy(original_capture.selection_payload)
    foreign["session_sha256"] = "0" * 64
    forged = replace(original_capture, selection_payload=foreign,
                     selection_sha256=graph_mod._sha(foreign), record_sha256="")
    graph_mod.validate_same_axis_candidates(forged.to_record())
    diagnosed.same_axis.value = forged
    with pytest.raises(subject.DiagnosticError, match="candidate_selection"):
        subject._evidence(diagnosed, inputs)


@pytest.mark.parametrize("mode", ("missing", "refused", "non_finite", "partial_axes"))
def test_unavailable_evidence_is_explicitly_incomplete(harness, tmp_path, mode):
    diagnosed, inputs = capture(harness, tmp_path)
    if mode == "missing":
        diagnosed.failure.value = replace(diagnosed.failure.value, solve_context=None, solver_input_bindings=None)
    elif mode == "refused":
        diagnosed.same_axis.value = graph_mod._same_axis_candidates_refusal("budget_exhausted", "rows")
    else:
        values = ((0, float("nan")), (1, .95), (2, .95)) if mode == "non_finite" else ((0, .5),)
        diagnosed.failure.value = replace(diagnosed.failure.value, axis_resultants=values)
        diagnostic = trace_mod.solver_failure_diagnostic(diagnosed.failure.value, .90, .15)
        diagnosed.attribution = replace(diagnosed.attribution, solver_failure=diagnostic, attribution_sha256="")
    record = subject._evidence(diagnosed, inputs)
    assert record["status"] == "incomplete" and record["issues"]
    subject._bounded_bytes(record)


def test_structure_and_size_caps_refuse_without_truncating(monkeypatch):
    for value in ({"x": float("nan")}, [0] * (subject.MAX_JSON_ITEMS + 1)):
        with pytest.raises(subject.DiagnosticError):
            subject._bounded_bytes(value)
    monkeypatch.setattr(subject, "MAX_RECORD_BYTES", 32)
    with pytest.raises(subject.DiagnosticError, match="evidence_cap"):
        subject._bounded_bytes({"x": "z" * 40})


def test_collision_preserves_existing_output_and_runs_no_chain(harness, tmp_path):
    root = tmp_path / "existing"
    root.mkdir()
    sentinel = root / "diagnostic.json"
    sentinel.write_bytes(b"keep")
    with pytest.raises(subject.DiagnosticError, match="output_collision"):
        subject.run(images=harness.paths, output=root)
    assert sentinel.read_bytes() == b"keep"
    assert sum(harness.counts.values()) == 0


@pytest.mark.parametrize("failure", ("write", "fsync"))
def test_write_failure_removes_only_created_temp_and_staging(harness, tmp_path, monkeypatch, failure):
    original_open = Path.open

    class FaultyWriter:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def write(self, data):
            self.stream.write(data[:20])
            raise OSError("synthetic write failure")

    def open_path(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return FaultyWriter(stream) if path.name == ".diagnostic.json.partial" else stream

    if failure == "write":
        monkeypatch.setattr(Path, "open", open_path)
    else:
        def fail(_):
            raise OSError("synthetic fsync failure")
        monkeypatch.setattr(subject.os, "fsync", fail)
    root = tmp_path / "failed"
    with pytest.raises(OSError, match=failure):
        subject.run(images=harness.paths, output=root)
    assert not root.exists()
    assert all(path.is_file() for path in harness.paths)


def test_final_path_absent_during_temp_write_flush_and_fsync(tmp_path, monkeypatch):
    destination = tmp_path / "diagnostic.json"
    temporary = tmp_path / ".diagnostic.json.partial"
    data = b'{"status":"complete"}\n'
    stages = []
    original_open, original_fsync, original_link = Path.open, subject.os.fsync, subject.os.link

    class ObservedWriter:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def write(self, value):
            assert not destination.exists()
            stages.append("write")
            return self.stream.write(value)

        def flush(self):
            assert not destination.exists()
            stages.append("flush")
            return self.stream.flush()

        def fileno(self):
            return self.stream.fileno()

    def open_path(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return ObservedWriter(stream) if path == temporary and args == ("xb",) else stream

    def fsync(handle):
        assert not destination.exists()
        stages.append("fsync")
        return original_fsync(handle)

    def link(source, target):
        assert stages == ["write", "flush", "fsync"]
        assert not destination.exists() and source.read_bytes() == data
        return original_link(source, target)

    monkeypatch.setattr(Path, "open", open_path)
    monkeypatch.setattr(subject.os, "fsync", fsync)
    monkeypatch.setattr(subject.os, "link", link)
    subject._publish_new(destination, data)
    assert destination.read_bytes() == data and not temporary.exists()


def test_publication_collision_preserves_exact_final_bytes(harness, tmp_path, monkeypatch):
    original_link = subject.os.link
    root = tmp_path / "publication-collision"

    def collide(source, destination):
        assert source.read_bytes()
        destination.write_bytes(b"existing-final-artifact")
        return original_link(source, destination)

    monkeypatch.setattr(subject.os, "link", collide)
    with pytest.raises(FileExistsError):
        subject.run(images=harness.paths, output=root)
    assert (root / "diagnostic.json").read_bytes() == b"existing-final-artifact"
    assert sorted(path.name for path in root.iterdir()) == ["diagnostic.json"]


def test_successful_atomic_publish_exposes_complete_digest_and_no_temp(harness, tmp_path):
    import hashlib

    root = tmp_path / "published"
    result = subject.run(images=harness.paths, output=root)
    data = (root / "diagnostic.json").read_bytes()
    record = json.loads(data)
    assert record["status"] == "incomplete"
    assert record["vertical_attrition"]["status"] == "incomplete"
    assert hashlib.sha256(data).hexdigest() == result["file_sha256"]
    assert record.pop("record_sha256") == subject._digest(record)
    assert sorted(path.name for path in root.iterdir()) == ["diagnostic.json"]


def test_unsupported_atomic_publication_fails_closed(harness, tmp_path, monkeypatch):
    def unsupported(*_):
        raise OSError("hard links unsupported")

    monkeypatch.setattr(subject.os, "link", unsupported)
    root = tmp_path / "unsupported"
    with pytest.raises(OSError, match="unsupported"):
        subject.run(images=harness.paths, output=root)
    assert not root.exists()


def test_evidence_failure_rolls_back_before_publication(harness, tmp_path, monkeypatch):
    def fail(*_):
        raise subject.DiagnosticError("invalid_evidence", "synthetic")
    monkeypatch.setattr(subject, "_evidence", fail)
    root = tmp_path / "failed"
    with pytest.raises(subject.DiagnosticError):
        subject.run(images=harness.paths, output=root)
    assert not root.exists()


def test_duplicate_callback_rolls_back_before_publication(harness, tmp_path, monkeypatch):
    original = trace_mod.derive_image_derived_graph_attribution

    def duplicate(**kwargs):
        result = original(**kwargs)
        sink = kwargs["same_axis_candidate_sink"]
        sink(sink.__self__.value)
        return result

    monkeypatch.setattr(trace_mod, "derive_image_derived_graph_attribution", duplicate)
    root = tmp_path / "duplicate"
    with pytest.raises(subject.DiagnosticError, match="callback_multiplicity"):
        subject.run(images=harness.paths, output=root)
    assert not root.exists()


def test_interface_has_no_metadata_truth_or_evaluator_loader():
    assert set(inspect.signature(subject.run).parameters) == {"images", "output", "local_period_support"}
    assert inspect.signature(subject.run).parameters["local_period_support"].default is False
    arguments = subject._parse_args(["--image", "a.png", "--output", "out"])
    assert set(vars(arguments)) == {"image", "output", "local_period_support"}
    assert arguments.local_period_support is False
    source = Path(subject.__file__).read_text()
    assert "run_candidate_support_ownership" not in source
    assert "load_projected_ownership" not in source
    assert "_r10_reference" not in source
    for option in ("--truth", "--metadata", "--bounds", "--camera-manifest", "--palette"):
        with pytest.raises(SystemExit):
            subject._parse_args(["--image", "a.png", "--output", "out", option, "x"])


def test_invalid_image_refuses_before_creating_output(harness, tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    root = tmp_path / "invalid"
    with pytest.raises(subject.DiagnosticError, match="image_header"):
        subject.run(images=[harness.paths[0], bad], output=root)
    assert not root.exists() and sum(harness.counts.values()) == 0


@pytest.mark.parametrize("failure", ("missing", "wrong_suffix", "stat_error"))
def test_preopen_path_refusal_has_bounded_os_context_without_image_read(
        tmp_path, monkeypatch, failure):
    valid = tmp_path / "first.png"
    second = tmp_path / ("second.txt" if failure == "wrong_suffix" else "second.png")
    valid.write_bytes(_png_bytes((20, 30, 40)))
    if failure == "wrong_suffix":
        second.write_text("metadata-only fixture")
    real_stat = os.stat
    if failure == "stat_error":
        def injected_stat(path, *args, **kwargs):
            if os.fspath(path) == str(second):
                error = OSError(13, "injected metadata refusal")
                error.winerror = 5
                raise error
            return real_stat(path, *args, **kwargs)
        monkeypatch.setattr(subject.os, "stat", injected_stat)
    def forbidden_open(*_args, **_kwargs):
        pytest.fail("image bytes opened before path refusal")
    monkeypatch.setattr(Path, "open", forbidden_open)
    # A later valid path must not be opened after the first refusal.
    with pytest.raises(subject.DiagnosticError) as raised:
        subject._input_records((second, valid))
    error = raised.value
    assert (error.reason, error.field) == ("invalid_inputs", "ordinary_image")
    context = error.context
    assert context["schema"] == "mc-photogrammetry-image-input-refusal-context-v1"
    assert context["arg_index"] == 0
    assert context["raw_argument_repr"] == repr(second)
    assert context["cwd_repr"] == repr(os.getcwd())
    assert context["resolved_path_repr"] == repr(str(second.absolute()))
    assert context["suffix_repr"] == repr(second.suffix.lower())
    assert all(value is False for value in context["truncated"].values())
    if failure == "wrong_suffix":
        assert context["stat"]["status"] == "ok"
        assert context["stat"]["type"] == "regular_file"
        assert type(context["stat"]["mode"]) is int
    else:
        assert context["stat"]["status"] == "error"
        assert context["stat"]["errno"] == (13 if failure == "stat_error" else 2)
        if failure == "stat_error":
            assert context["stat"]["winerror"] == 5
        else:
            assert type(context["stat"]["winerror"]) is int


def test_preopen_path_refusal_identifies_later_argument(tmp_path, monkeypatch):
    valid = tmp_path / "valid.png"
    valid.write_bytes(_png_bytes((20, 30, 40)))
    missing = tmp_path / "missing.png"
    monkeypatch.setattr(Path, "open", lambda *_a, **_k: pytest.fail("earlier image opened"))
    with pytest.raises(subject.DiagnosticError) as raised:
        subject._input_records((valid, missing))
    assert raised.value.context["arg_index"] == 1
    assert raised.value.context["resolved_path_repr"] == repr(str(missing.absolute()))


def test_preopen_path_refusal_cli_keeps_status_reason_field_and_no_output(tmp_path, capsys, monkeypatch):
    missing = tmp_path / "missing.png"
    output = tmp_path / "not_created"
    monkeypatch.setattr(Path, "open", lambda *_a, **_k: pytest.fail("image opened"))
    assert subject.main(["--image", str(missing), "--image", str(missing),
                         "--output", str(output)]) == 2
    refused = json.loads(capsys.readouterr().err)
    assert (refused["status"], refused["reason"], refused["field"]) == (
        "refused", "invalid_inputs", "ordinary_image")
    assert refused["context"]["arg_index"] == 0
    assert refused["context"]["stat"]["status"] == "error"
    assert not output.exists()


def test_preopen_path_context_truncates_long_argument(tmp_path):
    long_name = "x" * (subject.MAX_INPUT_CONTEXT_TEXT + 10) + ".png"
    with pytest.raises(subject.DiagnosticError) as raised:
        subject._input_records((tmp_path / long_name, tmp_path / "unused.png"))
    context = raised.value.context
    assert len(context["raw_argument_repr"]) == subject.MAX_INPUT_CONTEXT_TEXT
    assert context["truncated"]["raw_argument"] is True


@pytest.mark.parametrize("from_repo_root", (True, False))
def test_direct_script_bootstraps_source_without_pythonpath(tmp_path, from_repo_root):
    script = Path(subject.__file__).resolve()
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    # Isolated Python also excludes the CWD and user site packages, proving the
    # direct script's own bootstrap resolves its startup package import.
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(script), "--help"],
        cwd=subject.REPO_ROOT if from_repo_root else tmp_path,
        env=environment, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "--image" in result.stdout and "--output" in result.stdout
    assert result.stderr == ""
    assert list(tmp_path.iterdir()) == []


def _pair(axis, *, admitted=True, selected=False, psr=5., score=.8):
    return plane_graph.PlanePairTrace(
        "a", "b", axis, axis, 0, admitted,
        "" if admitted else "insufficient_overlap",
        (0, 0), .9, 64, (0, 0), .8, 32,
        left_plane=axis, right_plane=axis, selected=selected,
        psr=psr, score=score, overlap_cells=32, valid_cells=64,
        overlap_fraction=.5, cell_shift_rc=(0, 0),
        left_structural_share=.6, right_structural_share=.7)


def test_more_than_8192_pairs_preserve_all_aggregates_and_solver_evidence(harness, tmp_path):
    diagnosed, inputs = capture(harness, tmp_path)
    rows = (_pair(0, selected=True), _pair(1, psr=float("inf")),
            _pair(2, admitted=False, psr=None, score=None)) * 2731
    assert len(rows) == 8193
    diagnosed.attribution = replace(
        diagnosed.attribution, pair_attribution=rows,
        pre_admission_candidates=8193, post_admission_candidates=5462,
        post_same_axis_candidates=2731, attribution_sha256="")
    record, summary = subject._summarize_attribution(
        diagnosed.attribution, diagnosed.graph.frame_ids)
    assert (record, summary) == subject._summarize_attribution(
        diagnosed.attribution, diagnosed.graph.frame_ids)
    assert summary["status"] == "complete_aggregation"
    assert summary["source_row_count"] == 8193
    assert summary["admitted_row_count"] == 5462
    assert summary["selected_row_count"] == summary["refused_row_count"] == 2731
    assert [row["left_local_axis"] for row in summary["groups"]] == [0, 1, 2]
    assert all(row["rows"] == 2731 for row in summary["groups"])
    assert summary["groups"][1]["metrics"]["psr"]["positive_infinity_count"] == 2731
    assert summary["groups"][2]["metrics"]["psr"]["missing_count"] == 2731
    assert summary["groups"][2]["refused_by_reason"] == {"insufficient_overlap": 2731}
    assert summary["examples_status"] == "truncated"
    assert summary["retained_example_count"] == subject.MAX_PAIR_EXAMPLES
    assert summary["omitted_example_count"] == 8193 - subject.MAX_PAIR_EXAMPLES
    assert len(record["pairs"]) == subject.MAX_PAIR_EXAMPLES
    assert record["pairs_capture"]["status"] == "truncated"
    assert summary["verified_native_attribution_sha256"] == diagnosed.attribution.attribution_sha256
    assert record["candidate_counts"] == {
        "pre_admission": 8193, "post_admission": 5462, "post_same_axis": 2731}
    with pytest.raises(subject.DiagnosticError, match="candidate_admitted_ledger"):
        subject._evidence(diagnosed, inputs)
    assert len(subject._bounded_bytes(record)) < subject.MAX_RECORD_BYTES
    assert harness.counts["graph"] == harness.counts["solver"] == 1


def test_streamed_replay_equals_native_digest_with_unicode_and_infinite_psr(harness, tmp_path):
    import hashlib

    diagnosed, _ = capture(harness, tmp_path)
    trace = replace(diagnosed.attribution, detail="synthetic \u03c0", pair_attribution=(
        _pair(0), _pair(1, psr=float("inf")), _pair(2, admitted=False)), attribution_sha256="")
    record, summary = subject._summarize_attribution(trace, diagnosed.graph.frame_ids)
    assert record == trace.to_record()
    assert summary["examples_status"] == "complete" and summary["omitted_example_count"] == 0
    encoded_pairs = json.dumps(trace.to_record()["pairs"], sort_keys=True,
                               separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    assert summary["pair_sequence_sha256"] == hashlib.sha256(encoded_pairs).hexdigest()


@pytest.mark.parametrize("mutation", ("tampered", "non_finite", "foreign_frame"))
def test_omitted_pair_rows_are_still_verified(harness, tmp_path, mutation):
    diagnosed, _ = capture(harness, tmp_path)
    rows = (_pair(0),) * (subject.MAX_PAIR_EXAMPLES + 2)
    trace = replace(diagnosed.attribution, pair_attribution=rows, attribution_sha256="")
    last = replace(rows[-1], score=.2 if mutation == "tampered" else float("nan"))
    if mutation == "foreign_frame":
        last = replace(rows[-1], left="foreign")
    object.__setattr__(trace, "pair_attribution", (*rows[:-1], last))
    with pytest.raises(subject.DiagnosticError):
        subject._summarize_attribution(trace, diagnosed.graph.frame_ids)


def test_diagnostic_image_ceiling_uses_production_resource_bound_and_accepts_nine(tmp_path):
    assert subject.MAX_IMAGES == entry_mod.MAX_FRAMES_CAP == 32
    paths = []
    for index in range(9):
        path = tmp_path / f"view-{index}.png"
        path.write_bytes(_png_bytes((20 + index, 40 + index, 60 + index)))
        paths.append(path)
    resolved, records = subject._input_records(paths)
    assert len(resolved) == len(records) == 9
    with pytest.raises(subject.DiagnosticError, match="image_count"):
        subject._input_records([tmp_path / "unused.png"] * (entry_mod.MAX_FRAMES_CAP + 1))


def _reselect_provenance_graph(graph, *, readings=None, candidates=None, signatures=None):
    """Bind changed synthetic sources through the real selector, not a mock join."""
    from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
    from test_joint_endpoint_displacement_contract import _evidence as full_evidence
    readings = graph.readings if readings is None else tuple(readings)
    candidates = graph.candidate_matches if candidates is None else tuple(candidates)
    signatures = full_evidence(readings) if signatures is None else tuple(signatures)
    cert = selection.select_endpoint_hypotheses(readings=readings, signatures=signatures,
        candidate_matches=candidates, frame_ids=graph.selection.frame_ids,
        quadrant_constraints=graph.selection.quadrant_constraints, budget=graph.selection.budget,
        session_sha256="a" * 64, lattice_sha256="b" * 64, signatures_sha256="c" * 64)
    for name, value in (("selection", cert), ("readings", readings),
                        ("signatures", signatures), ("candidate_matches", candidates)):
        object.__setattr__(graph, name, value)
    return graph


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("permutation", ((0, 1, 2, 3, 4), (4, 3, 2, 1, 0), (2, 4, 1, 0, 3)))
def test_first_provenance_original_rows_permutation_direction_and_q(reverse, permutation):
    from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
    graph = _partial_witness_fixture(conflict=True, reverse=reverse, quadrant=1)
    graph = _reselect_provenance_graph(graph, candidates=[graph.candidate_matches[i] for i in permutation])
    record = subject._joint_endpoint_observation(graph)
    p = record["first_witness_provenance"]
    assert p["type"] == "available" and p["partial_exclusion_index"] == 0
    assert len(p["rows"]) == 2 and len(p["readings"]) == 3
    assert p["quadrants"] == [["a", 0], ["b", 1]]
    assert p["proof_sha256"] == record["proof_sha256"]
    indexed = plane_graph._canonical_reading_index(graph.readings)[0]
    for projected in p["rows"]:
        row = graph.candidate_matches[projected["source_row"]]
        assert projected["source_universe_sha256"] == record["candidate_matches_sha256"]
        assert projected["row_sha256"] == hashlib.sha256(selection._encode(selection.legacy._match_payload(row))).hexdigest()
        assert projected["original_direction"] == [[row.left, row.left_plane], [row.right, row.right_plane]]
        assert projected["canonical_direction"] == sorted(projected["original_direction"])
        assert projected["local_axes"] == [row.left_axis, row.right_axis]
        for join in projected["joins"]:
            a, b = join["original_alternative_indices"]
            assert join["canonical_alternative_indices"] == ([b, a] if reverse else [a, b])
            relation = plane_graph.signed_physical_relation(indexed, row, dict(p["quadrants"]),
                {(row.left, row.left_plane): a, (row.right, row.right_plane): b}, projected["source_row"], compact=True)
            assert relation[1] == join["canonical_signed_delta"]
            assert join["original_signed_delta"] == (-relation[1] if reverse else relation[1])
    assert subject._validate_joint_endpoint_observation(json.loads(subject._bounded_bytes(record)), graph) == record


def test_first_provenance_full_rosters_measurement_vs_signature_sampling_and_regions():
    from mc_photogrammetry.reconstruction import lattice, lattice_planes
    from test_joint_endpoint_displacement_contract import _evidence as full_evidence
    graph = _partial_witness_fixture(conflict=True)
    readings = []
    polygon = ((0.0, 0.0), (8.0, 0.0), (8.0, 6.0), (0.0, 6.0))
    for r in graph.readings:
        region = lattice_planes._make_region_evidence(np.ones((6, 8), dtype=bool), r.axis, polygon, ("synthetic",))
        original = r.alternatives[0]
        readings.append(replace(r, polygon=polygon, region_evidence=region, alternatives=tuple(
            lattice_planes.DistanceAlternative(b, original.distance_blocks * b, support)
            for b, support in ((0.5, 0.6), (1.0, 1.0), (2.0, 0.7)))))
    evidence = []
    for e in full_evidence(tuple(readings)):
        sig = e.signature
        footprint = lattice_planes.PlaneRegionFootprint(sig.reading.region_evidence.source_region_id,
            e.blocks_per_period, sig.origin_uv, sig.cells.shape, 3,
            bytes([8]) * sig.cells.size, bytes([9]) * sig.cells.size, 9)
        evidence.append(replace(e, signature=replace(sig, region_footprint=footprint), signature_sha256=""))
    graph = _reselect_provenance_graph(graph, readings=readings, signatures=evidence)
    record = subject._joint_endpoint_observation(graph)
    p = record["first_witness_provenance"]
    assert p["type"] == "available" and p["formula_residual_policy"] == "observation_only"
    assert graph.selection.status == "no"
    for joined in p["readings"]:
        r = graph.readings[joined["reading_index"]]
        assert joined["measurement_samples"] == 512
        assert joined["period_pixels"] == r.period_pixels
        assert joined["extent"] == r.extent
        assert joined["rectified_pixel_step"] == lattice.rectified_pixel_step(r.extent, r.samples)
        period = r.period_pixels * joined["rectified_pixel_step"]
        assert joined["world_units_per_period"] == period
        assert joined["source_region"]["source_region_id"] == r.region_evidence.source_region_id
        assert joined["source_region"]["mask_sha256"] == r.region_evidence.mask_sha256
        assert "packed_mask_base64" not in joined["source_region"]
        assert joined["rectification"] == json.loads(subject._bounded_bytes(asdict(r.rectification)))
        assert [a["blocks_per_period"] for a in joined["alternatives"]] == [0.5, 1.0, 2.0]
        for i, a in enumerate(joined["alternatives"]):
            original = r.alternatives[i]
            assert a["alternative_index"] == i and a["support"] == original.support
            assert a["distance_blocks"] == original.distance_blocks
            assert a["formula_distance_blocks"] == original.blocks_per_period / period
            assert a["distance_minus_formula_blocks"] == original.distance_blocks - original.blocks_per_period / period
            assert a["signature"]["sampling"]["subsample_factor"] == 3
            assert a["signature"]["sampling"]["total_samples_per_cell"] == 9
            assert a["signature"]["sampling"]["source_region_id"] == r.region_evidence.source_region_id
            assert a["signature"]["sha256"] == next(e.signature_sha256 for e in graph.signatures
                if e.canonical_index == joined["reading_index"] and e.alternative_index == i)
    # Large residuals from manual fixtures are recorded without changing admission.
    assert any(abs(a["distance_minus_formula_blocks"]) > 1 for r in p["readings"] for a in r["alternatives"])


@pytest.mark.parametrize("group", ("winning_round", "ranked_support_floor_candidates", "raw_supported_correlation_maxima"))
def test_first_provenance_each_optional_group_unavailable_independently(group):
    record = subject._joint_endpoint_observation(_partial_witness_fixture(conflict=True))
    optional = record["first_witness_provenance"]["optional_groups"][group]
    assert optional == {"type": "unavailable", "reason": "not_retained_in_current_joinable_stream"}
    assert "value" not in optional


def test_first_provenance_all_optional_groups_unavailable_keeps_unknown_causes_explicit():
    graph = _partial_witness_fixture(conflict=True)
    record = subject._joint_endpoint_observation(graph)
    p = record["first_witness_provenance"]
    assert len(p["optional_groups"]) == 3
    assert all(v == {"type": "unavailable", "reason": "not_retained_in_current_joinable_stream"}
               for v in p["optional_groups"].values())
    assert p["physical_identity"] == "not_established"
    assert p["unresolved_causal_questions"] == ["why_estimator_selected_period",
        "which_rejected_alternative_lost_at_support_floor", "whether_competing_peak_had_sufficient_support",
        "which_physical_correspondence_is_wrong"]
    assert subject._validate_joint_endpoint_observation(record, graph) == record


@pytest.mark.parametrize("field,value,tag", (("psr", float("inf"), "positive_infinity"),
    ("peak", float("-inf"), "negative_infinity"), ("second_peak", float("nan"), "nan")))
def test_first_provenance_nonfinite_metrics_are_typed_not_numeric(field, value, tag):
    graph = _partial_witness_fixture(conflict=True)
    candidates = tuple(replace(row, **{field: value}) for row in graph.candidate_matches)
    if field != "psr":
        # The source serializer refuses these values before certificate creation.
        # Keep that refusal; exercise the typed scalar codec without inventing a
        # source-admitted certificate for an inadmissible metric.
        with pytest.raises(ValueError, match="non-finite"):
            _reselect_provenance_graph(graph, candidates=candidates)
        assert subject._provenance_metric(value) == {"type": "nonfinite", "value": tag}
        object.__setattr__(graph, "candidate_matches", candidates)
        with pytest.raises(subject.DiagnosticError): subject._joint_endpoint_observation(graph)
        return
    graph = _reselect_provenance_graph(graph, candidates=candidates)
    record = subject._joint_endpoint_observation(graph)
    for row in record["first_witness_provenance"]["rows"]:
        assert row["matcher_metrics"][field] == {"type": "nonfinite", "value": tag}
        assert row["matcher_metrics"]["integer_residual"] == {
            "type": "unavailable", "reason": "not_present_in_source_record"}
    assert json.loads(subject._bounded_bytes(record)) == record


@pytest.mark.parametrize("damage", ("universe", "row", "alternative", "signature", "endpoint", "quadrant", "direction"))
def test_first_provenance_resealed_identity_tampering_refuses(damage):
    graph = _partial_witness_fixture(conflict=True)
    record = subject._joint_endpoint_observation(graph)
    p = record["first_witness_provenance"]
    if damage == "universe": p["source_universe_sha256"] = "0" * 64
    elif damage == "row": p["rows"][0]["row_sha256"] = "0" * 64
    elif damage == "alternative": p["rows"][0]["joins"][0]["original_alternative_indices"][0] += 1
    elif damage == "signature": p["readings"][0]["alternatives"][0]["signature"]["sha256"] = "0" * 64
    elif damage == "endpoint": p["readings"][0]["endpoint"][1] += 1
    elif damage == "quadrant": p["quadrants"][1][1] = 1
    else: p["rows"][0]["original_direction"].reverse()
    body = dict(record); body.pop("record_sha256")
    record["record_sha256"] = subject._digest(body)
    with pytest.raises(subject.DiagnosticError, match="joint_endpoint_certificate"):
        subject._validate_joint_endpoint_observation(record, graph)


@pytest.mark.parametrize("damage", ("duplicate_signature", "missing_signature", "wrong_signature_hash",
    "missing_reading", "ambiguous_reading", "duplicate_binding", "wrong_alternative"))
def test_first_provenance_required_source_joins_refuse_despite_optional_unavailable(damage):
    graph = _partial_witness_fixture(conflict=True)
    record = subject._joint_endpoint_observation(graph)
    if damage == "duplicate_signature": object.__setattr__(graph, "signatures", graph.signatures + graph.signatures[:1])
    elif damage == "missing_signature": object.__setattr__(graph, "signatures", graph.signatures[1:])
    elif damage == "wrong_signature_hash": object.__setattr__(graph.signatures[0], "signature_sha256", "0" * 64)
    elif damage == "missing_reading": object.__setattr__(graph, "readings", graph.readings[1:])
    elif damage == "ambiguous_reading": object.__setattr__(graph, "readings", graph.readings + graph.readings[:1])
    elif damage == "duplicate_binding": object.__setattr__(graph.selection, "reading_bindings",
        graph.selection.reading_bindings + graph.selection.reading_bindings[:1])
    else: object.__setattr__(graph.readings[0].alternatives[0], "distance_blocks", 12345.0)
    with pytest.raises(subject.DiagnosticError): subject._joint_endpoint_observation(graph)
    with pytest.raises(subject.DiagnosticError): subject._validate_joint_endpoint_observation(record, graph)


def test_first_provenance_precharges_lazy_roster_bytes_before_materialization(monkeypatch):
    consumed = []
    def roster():
        for index in range(100):
            consumed.append(index)
            yield {"value": "x" * 200}
    monkeypatch.setattr(subject, "MAX_JOINT_ENDPOINT_BYTES", 300)
    monkeypatch.setattr(subject.json, "loads", lambda *_: pytest.fail("unpaid projection allocated"))
    with pytest.raises(subject.DiagnosticError, match="evidence_cap"):
        subject._precharged_projection({"roster": subject._ProjectionSequence(roster())})
    assert consumed == [0, 1]


@pytest.mark.parametrize("resource", ("items", "depth", "bytes"))
def test_first_provenance_aggregate_structure_precharge_and_exact_byte_limit(monkeypatch, resource):
    if resource == "bytes":
        projected = subject._precharged_projection({"a": subject._ProjectionSequence(iter((1, 2, 3)))})
        size = len(json.dumps(projected, separators=(",", ":")).encode())
        monkeypatch.setattr(subject, "MAX_JOINT_ENDPOINT_BYTES", size)
        assert subject._precharged_projection({"a": [1, 2, 3]}) == projected
        monkeypatch.setattr(subject, "MAX_JOINT_ENDPOINT_BYTES", size - 1)
        value = {"a": [1, 2, 3]}
    elif resource == "items":
        monkeypatch.setattr(subject, "MAX_JSON_ITEMS", 8)
        value = {"a": subject._ProjectionSequence(iter([None] * 8))}
    else:
        monkeypatch.setattr(subject, "MAX_JSON_DEPTH", 2)
        value = {"a": [[None]]}
    monkeypatch.setattr(subject.json, "loads", lambda *_: pytest.fail("unpaid projection materialized"))
    with pytest.raises(subject.DiagnosticError, match="evidence_cap"):
        subject._precharged_projection(value)


def test_first_provenance_join_count_preflight_and_signature_ambiguity(monkeypatch):
    graph = _partial_witness_fixture(conflict=True)
    proof = graph.selection.proof_record()
    resolved = subject._joint_resolved_extrema(graph, proof)[0]
    forged = copy.deepcopy(resolved[:1])
    # Two disjoint rows would need four readings and cannot be partially published.
    forged[0]["low"]["source_row"] = 2
    forged[0]["high"]["source_row"] = 3
    monkeypatch.setattr(subject, "_precharged_projection", lambda *_: pytest.fail("over-cap joins allocated"))
    with pytest.raises(subject.DiagnosticError, match="joint_endpoint_provenance_joins"):
        subject._joint_first_witness_provenance(graph, forged)
    object.__setattr__(graph, "signatures", graph.signatures + graph.signatures[:1])
    with pytest.raises(subject.DiagnosticError, match="joint_endpoint_provenance_signature"):
        subject._joint_first_witness_provenance(graph, resolved)


def test_first_provenance_observer_work_adds_no_producer_or_csp_calls(monkeypatch):
    from mc_photogrammetry.reconstruction import endpoint_hypothesis as csp, lattice_planes
    graph = _partial_witness_fixture(conflict=True)
    proof = graph.selection.proof_record()
    resolved = subject._joint_resolved_extrema(graph, proof)[0]
    def forbidden(*_args, **_kwargs): pytest.fail("provenance replayed producer work")
    monkeypatch.setattr(csp, "solve_endpoint_hypotheses", forbidden)
    monkeypatch.setattr(lattice_planes, "_bounded_measure", forbidden)
    monkeypatch.setattr(lattice_planes, "plane_signature", forbidden)
    monkeypatch.setattr(plane_graph, "match_plane_pair", forbidden)
    monkeypatch.setattr(plane_graph, "match_plane_graph", forbidden)
    p = subject._joint_first_witness_provenance(graph, resolved)
    assert p["type"] == "available"
    assert len(p["rows"]) == 2 and len(p["readings"]) == 3


def test_first_provenance_atomic_no_clobber_readback_and_replay_accounting(harness, tmp_path, monkeypatch):
    graph = _partial_witness_fixture(conflict=True)
    record = subject._joint_endpoint_observation(graph)
    actual = subject._joint_endpoint_observation
    monkeypatch.setattr(subject, "_joint_endpoint_observation", lambda _graph: actual(graph))
    output = tmp_path / "first_provenance_publish"
    published = subject.run(images=harness.paths, output=output)
    data = (output / "diagnostic.json").read_bytes()
    assert published["file_sha256"] == hashlib.sha256(data).hexdigest()
    assert json.loads(data)["joint_endpoint_selection"] == record
    assert record["observer_work"] == {"csp_solves": 0, "measurement_calls": 0,
        "signature_regenerations": 0, "matcher_calls": 0, "authenticity_certificate_replays": 1}
    before = {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}
    with pytest.raises(subject.DiagnosticError): subject.run(images=harness.paths, output=output)
    assert {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()} == before
