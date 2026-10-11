"""Paired diagnostic controls with authored sources; no image decode or inputs."""
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from scripts.evaluation import evaluate_graph10_paired_universe as subject
from mc_photogrammetry.pipeline import image_derived_joint_graph as graph
from mc_photogrammetry.pipeline import image_derived_endpoint_selection as selection
from mc_photogrammetry.reconstruction import endpoint_hypothesis as csp
from mc_photogrammetry.reconstruction import plane_graph
from test_endpoint_hypothesis import connected_fixture
from test_image_derived_endpoint_selection import _fixture, _forge
from test_image_derived_graph_v3 import _source_records
from test_image_derived_joint_graph import _vertical_factor_sources
from test_image_derived_joint_geometry import _source_with_empty_axis


@pytest.fixture(autouse=True)
def synchronous_protocol(monkeypatch):
    # A fake thread inventory isolates the standalone protocol from pytest plugins.
    monkeypatch.setattr(subject.threading, "active_count", lambda: 1)


def _restored(modules):
    return (modules.matcher.match_plane_graph, modules.csp.solve_endpoint_hypotheses,
            modules.selection.validate_candidate_certificate)


def _assert_explicit_validations(record, status):
    for arm in ("full", "kept"):
        events = [event for event in record["certificate_validations"]
                  if event["phase"] == arm + ".explicit_validation"]
        assert len(events) == 1
        assert events[0] == {"phase": arm + ".explicit_validation", "status": status,
                             "accepted": True, "csp_recomputations": 1}
    calls = record["csp_calls"]
    assert sum(call["purpose"] == "solve" for call in calls) == 2
    assert all(call["purpose"] == "validation" for call in calls
               if call["phase"].endswith("explicit_validation"))


def test_real_single_pass_same_inputs_original_kept_and_production_parity(monkeypatch):
    sources = _source_records(monkeypatch)
    baseline = graph.derive_image_derived_graph(**sources)
    budget = csp.EndpointBudget()
    modules = subject._modules()
    captures, selectors, evaluations, registered, derives = [], [], [], [], []
    original_match = plane_graph.match_plane_graph
    original_select = selection.select_endpoint_hypotheses
    original_eval = graph._evaluate_selection
    original_derive = graph.derive_image_derived_graph
    original_register = graph._register_live

    def matcher(readings, signatures, **kwargs):
        kept = original_match(readings, signatures, **kwargs)
        captures.append((readings, signatures, kwargs, kwargs["admitted_candidate_sink"][0], kept))
        return kept

    def selector(**kwargs):
        selectors.append(kwargs)
        return original_select(**kwargs)

    def evaluator(**kwargs):
        evaluations.append(kwargs)
        return original_eval(**kwargs)

    def derive(**kwargs):
        derives.append(kwargs)
        return original_derive(**kwargs)

    def register(value):
        registered.append(value)
        return original_register(value)

    monkeypatch.setattr(plane_graph, "match_plane_graph", matcher)
    monkeypatch.setattr(selection, "select_endpoint_hypotheses", selector)
    monkeypatch.setattr(graph, "_evaluate_selection", evaluator)
    monkeypatch.setattr(graph, "derive_image_derived_graph", derive)
    monkeypatch.setattr(graph, "_register_live", register)
    before = _restored(modules)
    result = subject.compare_sources(**sources, budget=budget)
    assert result.record["status"] == "complete", result.record
    assert result.record["comparison"] == "same"
    assert result.record["matcher_calls"] == len(captures) == len(derives) == 1
    assert len(selectors) == len(evaluations) == 2
    assert registered == [result.full_graph]
    assert type(result.kept_evaluation) is graph._SelectionEvaluation
    assert not isinstance(result.kept_evaluation, graph.ImageDerivedGraphResult)
    assert result.record["kept_production_authorized"] is False
    assert result.record["accuracy_or_generalization_claim"] is False
    assert _restored(modules) == before
    assert graph._payload(result.full_graph) == graph._payload(baseline)
    assert result.full_graph.result_sha256 == baseline.result_sha256
    assert result.record["full_graph_sha256"] == baseline.result_sha256
    assert result.record["full_graph_payload"] == graph._payload(baseline)
    assert derives[0] == dict(sources, max_shift_cells=32, integer_tolerance=0.15,
                             min_resultant=0.90, ambiguous_margin=0.02, budget=budget)
    readings, signatures, options, universe, kept = captures[0]
    assert readings is result.full_graph.readings
    assert all(raw is evidence.signature for raw, evidence in zip(signatures, result.full_graph.signatures))
    assert options["max_shift_cells"] == 32
    assert selectors[0]["candidate_matches"] is universe.candidates
    assert selectors[1]["candidate_matches"] is kept
    for key in ("readings", "signatures", "frame_ids", "quadrant_constraints", "budget"):
        assert selectors[0][key] is selectors[1][key]
    for key in ("session_sha256", "lattice_sha256", "signatures_sha256", "connection_sha256"):
        assert selectors[0][key] == selectors[1][key]
    for key in ("readings", "signatures", "frame_ids"):
        assert evaluations[0][key] is evaluations[1][key]
    assert evaluations[1]["candidate_matches"] is kept
    row_map = result.record["kept_to_full_source_rows"]
    assert row_map == tuple(index for index, disposition in enumerate(universe.dispositions)
                            if disposition == plane_graph.SAME_AXIS_DISPOSITION_KEPT)
    assert all(row is universe.candidates[index] for row, index in zip(kept, row_map))
    assert result.record["universe"] == {
        "rows": [graph.legacy._match_payload(row) for row in universe.candidates],
        "source_rows": universe.source_rows, "dispositions": universe.dispositions}
    for rep in result.record["kept"]["representatives"]:
        assert rep["full_source_row"] == row_map[rep["local_source_row"]]
        assert rep["match"] == graph.legacy._match_payload(universe.candidates[rep["full_source_row"]])
    _assert_explicit_validations(result.record, "unique")


def test_real_dropped_rows_stay_full_and_kept_refuses_missing_factors(monkeypatch):
    sources = _vertical_factor_sources(monkeypatch)
    result = subject.compare_sources(**sources)
    assert result.record["status"] == "complete", result.record
    assert result.full_graph.selection.status == "unique"
    assert result.full_graph.status == "assembled"
    rows = result.full_graph.candidate_matches
    active = [row for row in rows if row.quadrant_of_right == 0]
    roles = tuple((row.left_plane, row.right_plane, row.left_axis, row.right_axis,
                   row.left_b, row.right_b) for row in active)
    assert roles == ((0, 0, 0, 0, 1., 1.),
                     (1, 1, 1, 1, 1., 1.), (1, 1, 1, 1, 1., 2.),
                     (1, 1, 1, 1, 2., 1.), (1, 1, 1, 1, 2., 2.),
                     (2, 2, 1, 1, 3., 3.), (3, 3, 2, 2, 1., 1.))
    dispositions = result.record["universe"]["dispositions"]
    assert any(value == plane_graph.SAME_AXIS_DISPOSITION_DROPPED for value in dispositions)
    target = next(rep for rep in result.full_graph.selection.representatives
                  if rep.match.left_plane == rep.match.right_plane == 1)
    assert target.match.left_b == target.match.right_b == 1.
    assert dispositions[target.source_row] == plane_graph.SAME_AXIS_DISPOSITION_DROPPED
    assert result.record["kept"]["selection_status"] == "no"
    assert result.record["kept"]["status"] == "unavailable"
    assert result.record["kept"]["geometry_evaluations"] == 0
    assert result.record["comparison"] == "different_consistency_or_availability"
    assert result.record["matcher_calls"] == 1


@pytest.mark.parametrize("state", ["no", "multiple", "budget_refused"])
def test_real_nonunique_budget_and_refusal_are_typed_not_improvements(monkeypatch, state):
    sources = _source_records(monkeypatch, alternatives=state == "multiple")
    if state == "no":
        sources = _source_with_empty_axis(sources, None)
    budget = dataclasses.replace(csp.EndpointBudget(), branch_nodes=0) if state == "budget_refused" else csp.EndpointBudget()
    result = subject.compare_sources(**sources, budget=budget)
    record = result.record
    assert record["matcher_calls"] == 1
    for arm in ("full", "kept"):
        assert record[arm]["selection_status"] == state
        assert record[arm]["geometry_evaluations"] == 0
        assert record[arm]["accepted_matches"] == [] and record[arm]["component"] is None
        assert record[arm]["counters"] == json.loads(record[arm]["certificate"]["proof_json"])["counters"]
    _assert_explicit_validations(record, state)
    if state == "budget_refused":
        assert record["status"] == record["comparison"] == "inconclusive"
        assert record["reason"] == "budget_exhausted"
        assert all(record[arm]["denial"] is not None and not record[arm]["exhaustive"] for arm in ("full", "kept"))
    else:
        assert record["status"] == "complete" and record["comparison"] == "same"


def test_numerical_gate_refusals_use_shared_evaluator_without_reconsideration(monkeypatch):
    sources = _source_records(monkeypatch)
    actual, evaluations = plane_graph._evaluate_assignment, []

    def evaluate(*args, **kwargs):
        cost, alternative, detail = actual(*args, **kwargs)
        evaluations.append(detail)
        return cost, alternative, dict(detail, resultants={0: 0.89, 1: 1., 2: 1.})

    monkeypatch.setattr(plane_graph, "_evaluate_assignment", evaluate)
    result = subject.compare_sources(**sources)
    assert result.record["status"] == "complete", result.record
    assert result.record["matcher_calls"] == 1 and len(evaluations) == 2
    assert result.record["comparison"] == "same"
    for arm in ("full", "kept"):
        assert result.record[arm]["selection_status"] == "unique"
        assert result.record[arm]["status"] == "unavailable" and result.record[arm]["reason"]
        assert result.record[arm]["geometry_evaluations"] == 1
        assert result.record[arm]["component"] is None and result.record[arm]["accepted_matches"] == []


def test_mismatched_upstream_source_pin_refuses_without_matcher_or_fallback(monkeypatch):
    sources = _source_records(monkeypatch)
    sources["lattice_result"] = dataclasses.replace(sources["lattice_result"], session_sha256="0" * 64,
                                                    result_sha256=None)
    result = subject.compare_sources(**sources)
    assert result.record["status"] == "inconclusive"
    assert result.record["reason"] == "provenance_mismatch" and result.record["field"] == "full.production"
    assert result.record["matcher_calls"] == 0 and result.record["csp_calls"] == []
    assert result.kept_evaluation is None


@pytest.mark.parametrize("state", ["unique", "no", "multiple", "unresolved_evidence", "budget_refused"])
def test_every_certificate_status_recomputes_real_csp_without_matcher(monkeypatch, state):
    # Pure certificate API controls; these rows are authored, not claimed as a matcher capture.
    _, rows = connected_fixture("a", "b")
    budget = csp.EndpointBudget()
    if state == "no":
        rows = []
    elif state == "multiple":
        rows += [dataclasses.replace(row, quadrant_of_right=2) for row in rows]
    elif state == "unresolved_evidence":
        rows.append(dataclasses.replace(rows[0], cell_shift_rc=(1, 2)))
    elif state == "budget_refused":
        budget = dataclasses.replace(budget, branch_nodes=0)
    cert, readings, evidence, candidates = _fixture(rows, budget)
    assert cert.status == state
    context = SimpleNamespace(session_sha256=cert.session_sha256, lattice_sha256=cert.lattice_sha256,
        signatures_sha256=cert.signatures_sha256, connection_sha256=cert.connection_sha256,
        frame_ids=cert.frame_ids, readings=readings, signatures=evidence, selection=cert)
    modules, observed = subject._modules(), subject._ObservedCalls()
    before = _restored(modules)
    with subject._observe(modules, observed):
        observed.phase = "authored.explicit_validation"
        subject._validate_arm(modules, cert, context, candidates, cert.candidate_dispositions, budget)
    assert observed.matcher_calls == 0
    assert observed.validations == [{"phase": "authored.explicit_validation", "status": state,
                                     "accepted": True, "csp_recomputations": 1}]
    assert len(observed.csp_calls) == 1 and observed.csp_calls[0]["purpose"] == "validation"
    assert _restored(modules) == before


@pytest.mark.parametrize("change", ["duplicate", "reorder", "bool", "omit", "equal_copy", "payload", "disposition"])
def test_mapping_exact_identity_cardinality_and_dispositions_refuse_tampering(monkeypatch, change):
    sources = _source_records(monkeypatch)
    modules, observed = subject._modules(), subject._ObservedCalls()
    with subject._observe(modules, observed):
        produced = graph.derive_image_derived_graph(**sources)
    assert type(produced) is graph.ImageDerivedGraphResult
    universe, kept = observed.universe, observed.kept
    row_map = tuple(index for index, value in enumerate(universe.dispositions) if value == "kept")
    subject._verify_mapping(modules, universe, kept, row_map)
    if change == "duplicate": row_map += row_map[:1]
    elif change == "reorder": row_map = tuple(reversed(row_map))
    elif change == "bool": row_map = (False,) + row_map[1:]
    elif change == "omit": row_map = row_map[:-1]
    elif change == "equal_copy": kept = (dataclasses.replace(kept[0]),) + kept[1:]
    elif change == "payload": kept = (dataclasses.replace(kept[0], peak=kept[0].peak + 1.),) + kept[1:]
    else:
        values = list(universe.dispositions)
        values[row_map[0]] = "same_axis_mismatch"
        universe = dataclasses.replace(universe, dispositions=tuple(values))
    with pytest.raises(subject.PairedRefusal, match="source_mismatch"):
        subject._verify_mapping(modules, universe, kept, row_map)


@pytest.mark.parametrize("change", ["digest", "source", "dispositions", "proof"])
def test_kept_certificate_tamper_refuses_before_kept_geometry_and_restores(monkeypatch, change):
    sources = _source_records(monkeypatch)
    actual, calls = selection.select_endpoint_hypotheses, []

    def select(**kwargs):
        cert = actual(**kwargs)
        calls.append(cert)
        if len(calls) == 1:
            return cert
        if change == "digest": return _forge(cert, result_sha256="0" * 64)
        if change == "source": return _forge(cert, session_sha256="0" * 64)
        if change == "dispositions": return _forge(cert, candidate_dispositions=())
        proof = cert.proof_record()
        proof["counters"]["solutions_found"] += 1
        forged = _forge(cert, proof_json=selection._encode(proof).decode())
        return _forge(forged, result_sha256=selection._sha(selection._payload(forged)))

    monkeypatch.setattr(selection, "select_endpoint_hypotheses", select)
    modules = subject._modules()
    before = _restored(modules)
    result = subject.compare_sources(**sources)
    assert result.record["status"] == "inconclusive" and result.record["matcher_calls"] == 1
    assert len(calls) == 2 and result.kept_evaluation is None
    assert not any(call["phase"] == "kept.shared_evaluation" for call in result.record["csp_calls"])
    assert result.full_graph.geometry_evaluations == 1
    assert _restored(modules) == before


def test_equal_resealed_full_copy_has_no_live_source_receipt(monkeypatch):
    sources = _source_records(monkeypatch)
    actual = graph.derive_image_derived_graph
    monkeypatch.setattr(graph, "derive_image_derived_graph",
                        lambda **kwargs: dataclasses.replace(actual(**kwargs)))
    result = subject.compare_sources(**sources)
    assert result.record["status"] == "inconclusive"
    assert result.record["reason"] == "validation_or_evaluation_refused"
    assert result.record["matcher_calls"] == 1 and result.kept_evaluation is None
    assert all(not call["phase"].startswith("kept.") for call in result.record["csp_calls"])


def test_no_rematch_after_refusal_and_second_matcher_call_is_never_forwarded(monkeypatch):
    sources = _source_records(monkeypatch)
    actual = graph.derive_image_derived_graph

    def derive(**kwargs):
        result = actual(**kwargs)
        plane_graph.match_plane_graph(result.readings, tuple(value.signature for value in result.signatures),
                                     admitted_candidate_sink=[])
        pytest.fail("second real matcher was forwarded")

    monkeypatch.setattr(graph, "derive_image_derived_graph", derive)
    modules = subject._modules()
    before = _restored(modules)
    result = subject.compare_sources(**sources)
    assert result.record["status"] == "inconclusive" and result.record["reason"] == "matcher_multiplicity"
    assert result.record["matcher_calls"] == 1 and result.kept_evaluation is None
    assert _restored(modules) == before


def test_observation_refuses_threads_reentrancy_and_validation_budget(monkeypatch):
    modules, observed = subject._modules(), subject._ObservedCalls()
    before = _restored(modules)
    monkeypatch.setattr(subject.threading, "active_count", lambda: 2)
    with pytest.raises(subject.PairedRefusal, match="inconclusive"):
        with subject._observe(modules, observed): pytest.fail("threaded observer admitted")
    monkeypatch.setattr(subject.threading, "active_count", lambda: 1)
    with subject._observe(modules, observed):
        with pytest.raises(subject.PairedRefusal, match="inconclusive"):
            with subject._observe(modules, subject._ObservedCalls()): pytest.fail("reentrant observer admitted")
        monkeypatch.setattr(subject, "MAX_CSP_CALLS", 0)
        with pytest.raises(subject.PairedRefusal, match="inconclusive"):
            modules.csp.solve_endpoint_hypotheses((), (), frame_ids=())
    assert _restored(modules) == before


def test_png_only_fake_upstream_once_no_manual_or_sibling_ingress(monkeypatch):
    paths = (Path("authored-a.png"), Path("authored-b.png"))
    labels = tuple(str(path) for path in paths)
    ids = ("image-0000", "image-0001")
    ingress = tuple(SimpleNamespace(source_label=label, frame_id=frame_id, byte_sha256="a" * 64, byte_length=1,
        to_record=lambda label=label, frame_id=frame_id: {
        "frame_id": frame_id, "source_label": label, "byte_sha256": "a" * 64, "byte_length": 1}) for label, frame_id in zip(labels, ids))
    frames = tuple(SimpleNamespace(frame_id=pin.frame_id, ingress=pin) for pin in ingress)
    session = SimpleNamespace(input_frames=frames, frames=frames, version=1, admissions=(), session_sha256="c" * 64)
    pose, connection = object(), object()
    entry = SimpleNamespace(source_paths=labels, session=session, pose_graph=pose, graph=connection, entry_sha256="e" * 64)
    lattice = SimpleNamespace(status="assembled")
    signatures = SimpleNamespace(status="assembled")
    modules, calls = subject._modules(), []

    def build(value, **kwargs):
        calls.append(("entry", value, kwargs))
        modules.image_only.solve_image_cameras(camera_inputs)
        return entry

    def measure(**kwargs):
        calls.append(("lattice", kwargs))
        _emit_authored_attrition(modules.region_growth, ids, returned=1)
        return lattice

    def sign(**kwargs):
        calls.append(("signatures", kwargs))
        return signatures

    def compare(**kwargs):
        calls.append(("paired", kwargs))
        return subject.PairedResult({"status": "complete", "record_sha256": "discard-before-reseal"},
                                    full_graph=connection, kept_evaluation=pose)

    monkeypatch.setattr(subject, "_png_paths", lambda value: paths if value == labels else pytest.fail("input changed"))
    monkeypatch.setattr(modules.entry, "build_image_derived_entry", build)
    monkeypatch.setattr(modules.lattice, "measure_image_derived_lattice", measure)
    monkeypatch.setattr(modules.signatures, "derive_image_derived_signatures", sign)
    monkeypatch.setattr(subject, "_modules", lambda: modules)
    monkeypatch.setattr(subject, "compare_sources", compare)
    camera_inputs = _install_authored_camera(modules, ids, (16, 18))
    lattice.frame_ids, lattice.camera_calibrations, lattice.abstentions = ids, _authored_calibrations(ids), ()
    result = subject.evaluate_pngs(labels, project_path="authored-project.mcscan")
    assert result.record["status"] == "complete", result.record
    assert calls == [("entry", paths, {"project_path": "authored-project.mcscan", "recursive": False,
                                      "capture_ingress": True, "max_frames": 32}),
                     ("lattice", {"session": session, "pose_graph": pose}),
                     ("signatures", {"session": session, "lattice_result": lattice}),
                     ("paired", {"session": session, "lattice_result": lattice,
                                 "signatures_result": signatures, "connection_graph": connection})]
    assert result.record["ordered_png_paths"] == list(labels)
    assert result.record["input_ingress"] == [pin.to_record() for pin in ingress]
    assert result.record["entry_sha256"] == entry.entry_sha256
    assert result.record["session_sha256"] == session.session_sha256
    assert result.full_graph is connection and result.kept_evaluation is pose
    payload = dict(result.record)
    digest = payload.pop("record_sha256")
    assert digest == subject._digest(payload)


@pytest.mark.parametrize("case", ["entry_order", "missing_ingress", "lattice", "signatures", "entry_error"])
def test_fake_upstream_refusals_stop_without_downstream_or_retry(monkeypatch, case):
    paths = (Path("authored-a.png"), Path("authored-b.png"))
    labels = tuple(str(path) for path in paths)
    frames = tuple(SimpleNamespace(ingress=SimpleNamespace(source_label=label, byte_sha256="a" * 64, byte_length=1,
        to_record=lambda label=label: {
        "source_label": label, "byte_sha256": "a" * 64, "byte_length": 1})) for label in labels)
    if case == "missing_ingress": frames[0].ingress = None
    entry = SimpleNamespace(source_paths=tuple(reversed(labels)) if case == "entry_order" else labels,
                            session=SimpleNamespace(input_frames=frames, session_sha256="c" * 64),
                            pose_graph=object(), entry_sha256="e" * 64)
    calls = []

    def build(*args, **kwargs):
        calls.append("entry")
        if case == "entry_error": raise ValueError("authored typed failure")
        return entry

    def measure(**kwargs):
        calls.append("lattice")
        return SimpleNamespace(status="unavailable" if case == "lattice" else "assembled")

    def sign(**kwargs):
        calls.append("signatures")
        return SimpleNamespace(status="unavailable")

    modules = SimpleNamespace(entry=SimpleNamespace(build_image_derived_entry=build),
        lattice=SimpleNamespace(measure_image_derived_lattice=measure),
        signatures=SimpleNamespace(derive_image_derived_signatures=sign),
        placement=subject._modules().placement)
    monkeypatch.setattr(subject, "_modules", lambda: modules)
    monkeypatch.setattr(subject, "_png_paths", lambda _: paths)
    monkeypatch.setattr(subject, "compare_sources", lambda **k: pytest.fail("refusal reached graph"))
    result = subject.evaluate_pngs(labels, project_path="unwritten.mcscan")
    assert result.record["status"] == "inconclusive"
    assert result.record["matcher_calls"] == 0 and result.record["csp_calls"] == []
    expected = ["entry"]
    if case in ("lattice", "signatures"): expected += ["lattice"]
    if case == "signatures": expected += ["signatures"]
    assert calls == expected


def _authored_metadata_entry():
    from mc_photogrammetry.pipeline.contracts import FrameTransform
    from mc_photogrammetry.pipeline.frame_ingress import FrameIngressProvenanceV1

    paths = (Path("authored-a.png"), Path("authored-b.png"))
    labels = tuple(str(path) for path in paths)
    pins = tuple(FrameIngressProvenanceV1(frame_id=f"image-{index:04d}", source_label=label,
        byte_sha256=("a" if index == 0 else "b") * 64, byte_length=11 + index,
        decoded_shape=(2, 3, 3), decoded_dtype="|u1", decoded_pixel_sha256="f" * 64,
        transform=FrameTransform(source_size=(3, 2))) for index, label in enumerate(labels))

    class MetadataOnlyFrame:
        def __init__(self, pin):
            self.frame_id, self.ingress = pin.frame_id, pin

        @property
        def image(self):
            pytest.fail("diagnostic touched pixel data")

    session = SimpleNamespace(input_frames=tuple(MetadataOnlyFrame(pin) for pin in pins),
                              session_sha256="c" * 64, version=1, admissions=())
    session.frames = session.input_frames
    entry = SimpleNamespace(source_paths=labels, entry_sha256="e" * 64,
                            session=session, pose_graph=object(), graph=object())
    return paths, entry, pins


def _fake_metadata_chain(monkeypatch, failed_stage, returned=None, raised=None):
    paths, entry, pins = _authored_metadata_entry()
    calls = []
    assembled = SimpleNamespace(status="assembled", reason="", detail="accepted source evidence", result_sha256="d" * 64)

    def build(*args, **kwargs):
        calls.append("entry")
        modules.image_only.solve_image_cameras(camera_inputs)
        return entry

    def stage(name):
        def invoke(**kwargs):
            calls.append(name)
            if name == "lattice":
                _emit_authored_attrition(modules.region_growth, ids, returned=1)
            if name == failed_stage:
                if raised is not None: raise raised
                return returned
            return assembled
        return invoke

    def forbidden(*args, **kwargs):
        pytest.fail("upstream refusal reached matcher/CSP/paired evaluation")

    modules = SimpleNamespace(entry=SimpleNamespace(build_image_derived_entry=build),
        lattice=SimpleNamespace(measure_image_derived_lattice=stage("lattice")),
        signatures=SimpleNamespace(derive_image_derived_signatures=stage("signatures")),
        placement=subject._modules().placement,
        matcher=SimpleNamespace(match_plane_graph=forbidden), csp=SimpleNamespace(solve_endpoint_hypotheses=forbidden))
    ids = tuple(pin.frame_id for pin in pins)
    camera_inputs = _install_authored_camera(modules, ids, (16, 18))
    assembled.frame_ids, assembled.camera_calibrations, assembled.abstentions = ids, _authored_calibrations(ids), ()
    monkeypatch.setattr(subject, "_modules", lambda: modules)
    monkeypatch.setattr(subject, "_png_paths", lambda _: paths)
    monkeypatch.setattr(subject, "compare_sources", forbidden)
    return paths, entry, pins, calls


def _assert_failed_metadata(result, entry, pins, stage):
    record = result.record
    assert record["status"] == "inconclusive" and record["field"] == stage
    assert record["ordered_png_paths"] == list(entry.source_paths)
    assert record["entry_sha256"] == entry.entry_sha256
    assert record["session_sha256"] == entry.session.session_sha256
    assert record["input_ingress"] == [pin.to_record() for pin in pins]
    assert [(row["source_label"], row["byte_sha256"], row["byte_length"]) for row in record["input_ingress"]] == [
        (pin.source_label, pin.byte_sha256, pin.byte_length) for pin in pins]
    assert record["matcher_calls"] == 0 and record["csp_calls"] == [] and record["certificate_validations"] == []
    assert result.full_graph is None and result.kept_evaluation is None
    assert "full" not in record and "kept" not in record and "comparison" not in record
    assert record["accuracy_or_generalization_claim"] is record["kept_production_authorized"] is False
    assert "image" not in record and "pixels" not in record and "session" not in record
    assert "exception_text" not in record and "traceback" not in record
    payload = dict(record)
    digest = payload.pop("record_sha256")
    assert digest == subject._digest(payload)
    # Earlier metadata and the returned stage summary are inside the final digest.
    for field in ("entry_sha256", "session_sha256", "ordered_png_paths", "input_ingress", "upstream_stages"):
        changed = dict(payload)
        changed.pop(field)
        assert subject._digest(changed) != digest


@pytest.mark.parametrize("stage", ["lattice", "signatures"])
@pytest.mark.parametrize("kind", ["unavailable", "placement_refusal", "exception", "foreign_paired_refusal"])
def test_upstream_failures_preserve_digest_bound_entry_ingress_without_pixels(monkeypatch, stage, kind):
    placement = subject._modules().placement
    returned, raised = None, None
    if kind == "unavailable":
        returned = SimpleNamespace(status="unavailable", reason="image_unavailable",
            detail="retained pixels are unavailable", terminal="geometry_unavailable", result_sha256="d" * 64)
    elif kind == "placement_refusal":
        returned = placement._make_refusal("provenance_mismatch", {"field": "session", "detail": "ValueError"})
    elif kind == "exception":
        class NoExceptionText(RuntimeError):
            def __str__(self): pytest.fail("exception was stringified")
        raised = NoExceptionText("C:\\private\\source.png", "secret-exception-data" * 100_000)
    else:
        raised = subject.PairedRefusal("C:\\private\\source.png", "secret-exception-data" * 100_000)
    paths, entry, pins, calls = _fake_metadata_chain(monkeypatch, stage, returned, raised)
    result = subject.evaluate_pngs(paths, project_path="unwritten.mcscan")
    _assert_failed_metadata(result, entry, pins, stage)
    assert calls == ["entry", "lattice"] + (["signatures"] if stage == "signatures" else [])
    record, summary = result.record, result.record["upstream_stages"][stage]
    if kind in ("exception", "foreign_paired_refusal"):
        assert record["reason"] == "upstream_error"
        assert record["exception_class"] == ("NoExceptionText" if kind == "exception" else "PairedRefusal")
        assert summary == {"kind": "exception", "status": None, "reason": None,
                           "detail": None, "detail_capture": "not_returned", "source_result_sha256": None}
        assert "secret-exception-data" not in subject.canonical_bytes(record).decode()
        assert "private" not in subject.canonical_bytes(record).decode()
    else:
        assert record["reason"] == "upstream_unavailable" and "exception_class" not in record
        assert summary["reason"] == returned.reason and summary["detail"] == returned.detail
        assert summary["detail_capture"] == "complete"
        if kind == "unavailable":
            assert summary["kind"] == "result" and summary["status"] == "unavailable"
            assert summary["source_result_sha256"] == returned.result_sha256
        else:
            assert summary["kind"] == "placement_refusal" and summary["status"] is None
            assert summary["source_result_sha256"] == returned.refusal_sha256
    if stage == "signatures":
        assert record["upstream_stages"]["lattice"]["status"] == "assembled"


@pytest.mark.parametrize("detail,expected,capture", [
    ({"field": "session", "pixels": [[1, 2, 3]], "detail": "C:\\private\\source.png"}, {"field": "session"}, "partial"),
    ({"field": "session", "detail": object()}, {"field": "session"}, "partial"),
    ({"field": "session", "observed": float("inf")}, {"field": "session"}, "partial"),
    ({"field": "session", "detail": "x" * 1025}, {"field": "session"}, "partial"),
    ({str(index): 1 for index in range(9)}, None, "omitted_cap"),
    ([1, 2, 3], None, "omitted_unsupported"),
], ids=["pixel-and-path-fields", "object-field", "nonfinite-field", "text-cap", "key-cap", "nested-detail"])
def test_returned_stage_detail_is_bounded_allowlisted_plain_scalars(monkeypatch, detail, expected, capture):
    class SummaryOnly(SimpleNamespace):
        def to_record(self): pytest.fail("whole stage payload serialized")
    returned = SummaryOnly(status="unavailable", reason="provenance_mismatch", detail=detail)
    paths, entry, pins, calls = _fake_metadata_chain(monkeypatch, "lattice", returned)
    result = subject.evaluate_pngs(paths, project_path="unwritten.mcscan")
    _assert_failed_metadata(result, entry, pins, "lattice")
    summary = result.record["upstream_stages"]["lattice"]
    assert summary["detail"] == expected and summary["detail_capture"] == capture
    assert calls == ["entry", "lattice"]
    assert "private" not in subject.canonical_bytes(result.record).decode()


@pytest.mark.parametrize("case", ["entry_digest", "session_digest", "byte_hash", "byte_length", "byte_hash_substitution",
                                  "byte_length_substitution", "label", "pixel_field"])
def test_failure_metadata_requires_real_digest_fields_and_exact_ingress_byte_join(monkeypatch, case):
    paths, entry, pins, calls = _fake_metadata_chain(monkeypatch, "lattice", SimpleNamespace(status="unavailable"))
    if case == "entry_digest": entry.entry_sha256 = "invalid"
    elif case == "session_digest": entry.session.session_sha256 = "invalid"
    else:
        record = pins[0].to_record()
        if case == "byte_hash": record["byte_sha256"] = "invalid"
        elif case == "byte_length": record["byte_length"] = True
        elif case == "byte_hash_substitution": record["byte_sha256"] = pins[1].byte_sha256
        elif case == "byte_length_substitution": record["byte_length"] = pins[1].byte_length
        elif case == "label": record["source_label"] = pins[1].source_label
        else: record["pixels"] = [[1, 2, 3]]
        replacement = SimpleNamespace(source_label=pins[0].source_label, frame_id=pins[0].frame_id,
            byte_sha256=pins[0].byte_sha256, byte_length=pins[0].byte_length, to_record=lambda: record)
        entry.session.input_frames[0].ingress = replacement
    result = subject.evaluate_pngs(paths, project_path="unwritten.mcscan")
    assert result.record["status"] == "inconclusive" and result.record["reason"] == "input_inventory_changed"
    assert calls == ["entry"] and result.record["matcher_calls"] == 0 and result.record["csp_calls"] == []
    assert "input_ingress" not in result.record
    assert result.record["ordered_png_paths"] == list(entry.source_paths)
    assert result.full_graph is result.kept_evaluation is None
    assert "pixels" not in subject.canonical_bytes(result.record).decode()


@pytest.mark.parametrize("case", ["directory", "duplicate", "reordered", "non_png", "missing", "too_few", "bool"])
def test_authored_path_refusals_precede_entry_and_no_file_is_decoded(tmp_path, monkeypatch, case):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(b"authored opaque buffer a")
    b.write_bytes(b"authored opaque buffer b")
    values = [a, b]
    if case == "directory": values[0] = tmp_path
    elif case == "duplicate": values[1] = a
    elif case == "reordered": values.reverse()
    elif case == "non_png":
        values[0] = tmp_path / "a.json"
        values[0].write_bytes(b"authored forbidden sidecar")
    elif case == "missing": values[1] = tmp_path / "missing.png"
    elif case == "too_few": values = [a]
    else: values[0] = True
    modules = subject._modules()
    monkeypatch.setattr(modules.entry, "build_image_derived_entry", lambda *a, **k: pytest.fail("entry reached"))
    monkeypatch.setattr(subject, "_modules", lambda: modules)
    result = subject.evaluate_pngs(values, project_path=tmp_path / "unwritten.mcscan")
    assert result.record["status"] == "inconclusive" and result.record["reason"] == "invalid_inputs"
    assert result.record["matcher_calls"] == 0
    assert not (tmp_path / "unwritten.mcscan").exists()


@pytest.mark.parametrize("case", ["depth", "items", "bytes", "nonfinite", "integer", "object", "key"])
def test_bounded_canonical_reports_refuse_before_json_encoder(monkeypatch, case):
    if case == "depth":
        value = None
        for _ in range(33): value = [value]
    elif case == "items":
        monkeypatch.setattr(subject, "MAX_REPORT_ITEMS", 3)
        value = [1, 2, 3]
    elif case == "bytes":
        monkeypatch.setattr(subject, "MAX_REPORT_BYTES", 3)
        value = "four"
    elif case == "nonfinite": value = float("inf")
    elif case == "integer": value = 1 << 65
    elif case == "object": value = object()
    else: value = {1: None}
    monkeypatch.setattr(subject.json.JSONEncoder, "iterencode", lambda *a, **k: pytest.fail("encoder reached"))
    with pytest.raises(subject.PairedRefusal):
        subject.canonical_bytes(value)


class _OpaqueArray:
    """Identity-only authored stand-in; no pixel operations are permitted."""
    def __eq__(self, other): pytest.fail("array equality used instead of identity")
    def __iter__(self): pytest.fail("pixel array traversed")
    def __repr__(self): pytest.fail("pixel array represented")


class _AuthoredSegments:
    def __init__(self, count): self.count = count
    def __len__(self): return self.count
    def __iter__(self): pytest.fail("segment coordinates traversed")


def _authored_line_rows(count, *, eligible=None, scale=1.0):
    """Numeric detector rows; image stand-ins remain identity-only."""
    eligible = count if eligible is None else eligible
    rows = np.zeros((count, 1, 4), dtype=np.float64)
    rows[:eligible, 0, 2] = 80.0 * scale
    rows[eligible:, 0, 2] = 20.0 * scale
    return rows


def _install_authored_camera(modules, ids, counts):
    from mc_photogrammetry.reconstruction import region_growth

    raw = {frame: _OpaqueArray() for frame in ids}
    gray = {frame: _OpaqueArray() for frame in ids}
    enhanced = {frame: _OpaqueArray() for frame in ids}
    resized = {frame: _OpaqueArray() for frame in ids}
    retained = {frame: _AuthoredSegments(count) for frame, count in zip(ids, counts)}
    found = {frame: (_authored_line_rows(count), None) for frame, count in zip(ids, counts)}
    detector_outputs = {frame: tuple((item, None, None, None) for item in found[frame]) for frame in ids}
    estimates = {frame: SimpleNamespace(segments=count, inliers=min(count, 15),
                  per_axis_inliers=(min(count, 15), 0, 0)) if count >= 12 else None
                  for frame, count in zip(ids, counts)}
    trace = []
    image_only, calibration, cv2 = SimpleNamespace(), SimpleNamespace(), SimpleNamespace(INTER_AREA=3)

    def convert(image):
        return next(gray[frame] for frame in ids if raw[frame] is image)

    def enhance(image):
        frame = next(frame for frame in ids if gray[frame] is image)
        result = enhanced[frame]
        trace.append(("enhance", (image,), {}, result))
        return result

    def resize(image, size, **kwargs):
        frame = next(frame for frame in ids if enhanced[frame] is image)
        result = resized[frame]
        trace.append(("resize", (image, size), kwargs, result))
        return result

    class Detector:
        def detect(self, image):
            frame, ordinal = next((frame, index) for frame in ids for index, value in enumerate((enhanced[frame], resized[frame])) if value is image)
            result = detector_outputs[frame][ordinal]
            trace.append(("detector", (image,), {}, result))
            return result

    def factory(*args, **kwargs):
        result = Detector()
        trace.append(("factory", args, kwargs, result))
        return result

    def detect(image, *, scales=(1.0, 0.5), minimum_length=40.0):
        frame = next(frame for frame in ids if enhanced[frame] is image)
        detector = cv2.createLineSegmentDetector()
        for scale in scales:
            source = image if scale == 1.0 else cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            detector.detect(source)
        result = retained[frame]
        trace.append(("segments", (image,), {"scales": scales, "minimum_length": minimum_length}, result))
        return result

    def fit(image, *args, **kwargs):
        frame = next((frame for frame in ids if gray[frame] is image), None)
        if frame is None: return object()
        if kwargs.get("segments") is None:
            calibration.detect_segments(calibration.enhance(image), scales=calibration.DETECTION_SCALES, minimum_length=40.0)
        return estimates[frame]

    def session(frames, *args, **kwargs):
        return {frame: calibration.estimate_camera(image) for frame, image in frames.items()}

    def solve(frames):
        assert tuple(frames) == ids
        assert all(frames[frame] is raw[frame] for frame in ids)
        grayscale = {frame: image_only._to_grayscale(image) for frame, image in frames.items()}
        results = {}
        for frame, image in grayscale.items():
            result = calibration.estimate_camera(image)
            if result is not None: results[frame] = result
        return results

    image_only.solve_image_cameras, image_only._to_grayscale = solve, convert
    calibration.estimate_session, calibration.estimate_camera = session, fit
    calibration.enhance, calibration.detect_segments = enhance, detect
    calibration.DETECTION_SCALES, calibration.MINIMUM_LENGTH = (1.0, 0.5), 40.0
    cv2.resize, cv2.createLineSegmentDetector = resize, factory
    modules.image_only, modules.calibration, modules.region_growth, modules.cv2 = image_only, calibration, region_growth, cv2
    modules.camera_fixture = SimpleNamespace(gray=gray, enhanced=enhanced, resized=resized, retained=retained,
        detector_outputs=detector_outputs, estimates=estimates, trace=trace)
    return raw


def _authored_calibrations(ids):
    return tuple(SimpleNamespace(segments=20, inliers=15, per_axis_inliers=(5, 6, 4)) for _ in ids)


def _emit_authored_attrition(region, ids, *, returned=1, disposition="accepted", outcome=None):
    observer = region._attrition_state.get()
    assert observer is not None
    for frame in ids:
        observer.expect("frame", (frame, "proposal"))
        observer.finish("frame", (frame, "proposal"), outcome or (
            "image_edge_graph_growth" if returned else "no_grown_plane_regions"), returned_count=returned)
        for ordinal in range(returned):
            observer.expect("proposal", (frame, ordinal))
            observer.finish("proposal", (frame, ordinal), "returned", axis=0)
            observer.bind(("proposal_return", frame, ordinal), (frame, ordinal))
            observer.expect("reading", (frame, ordinal))
            observer.finish("reading", (frame, ordinal), disposition, axis=0)
        if returned:
            for name in ("proposal_generation", "proposal_returned", "proposal_to_reading"):
                observer.check(name + ":" + frame, True)


def _camera_metadata(ids, counts):
    active = tuple(frame for frame, count in zip(ids, counts) if count >= 12)
    rows = tuple(SimpleNamespace(frame_id=frame, status="admitted" if frame in active else "excluded",
        stage="camera_recovery", reason="" if frame in active else "camera_estimate_unavailable") for frame in ids)
    frames = tuple(SimpleNamespace(frame_id=frame) for frame in active)
    return SimpleNamespace(version=2 if len(active) != len(ids) else 1,
        frames=frames, input_frames=tuple(SimpleNamespace(frame_id=frame) for frame in ids),
        admissions=rows if len(active) != len(ids) else ())


def _camera_callbacks(modules):
    return (modules.image_only.solve_image_cameras, modules.image_only._to_grayscale,
            modules.calibration.estimate_session, modules.calibration.estimate_camera,
            modules.calibration.enhance, modules.calibration.detect_segments,
            modules.cv2.resize, modules.cv2.createLineSegmentDetector)


@pytest.mark.parametrize("counts", [(16, 11, 0, 3), (16, 12, 13, 14), (11, 0, 3, 8)],
                         ids=["partial", "all-admitted-v1", "all-abstain"])
def test_camera_same_pass_identity_joins_four_inputs_three_exclusions_and_v1(counts):
    ids = tuple(f"image-{index:04d}" for index in range(4))
    modules = SimpleNamespace()
    raw = _install_authored_camera(modules, ids, counts)
    before = _camera_callbacks(modules)
    accounting = subject._CameraAccounting()
    with subject._observe_camera(modules, accounting): original_result = modules.image_only.solve_image_cameras(raw)
    assert _camera_callbacks(modules) == before
    assert set(original_result) == {frame for frame, count in zip(ids, counts) if count >= 12}
    record = subject._camera_record(_camera_metadata(ids, counts), accounting, [{"frame_id": frame} for frame in ids])
    assert record["status"] == "complete" and record["solve_calls"] == 1
    assert record["session_calls"] == 0 and record["fit_calls"] == len(ids)
    assert record["fitting_mode"] == "independent_per_frame"
    assert [row["frame_id"] for row in record["frames"]] == list(ids)
    assert [row["segments"] for row in record["frames"]] == list(counts)
    assert all(row["segments_unavailable_reason"] is None and row["fixed_focal_disposition"] is None
               and row["capture"] == "complete" and row["calibration_evidence"]["identity_references_cleared"] is True
               for row in record["frames"])
    assert record["active_frame_ids"] == list(original_result)
    assert record["admission_evidence"] == ("session_v1_all_admitted" if all(count >= 12 for count in counts)
                                            else "explicit_session_admissions")
    assert type(record["admission_evidence"]) is str
    for frame, count, row in zip(ids, counts, record["frames"]):
        assert row["frame_id"] == frame
        fit_evidence = row["calibration_evidence"]
        assert type(fit_evidence) is dict and fit_evidence["status"] == "complete"
        assert fit_evidence["retained_segment_count"] == count
        passes = fit_evidence["raw_detections_per_pass"]
        assert [(item["scale"], item["raw_detection_count"]) for item in passes] == [(1.0, count), (0.5, 0)]
        for item, expected_count in zip(passes, (count, 0)):
            assert item["finite_row_count"] == item["exact_ge40_count"] == expected_count
            assert item["nonfinite_row_count"] == item["nonfinite_eligible_count"] == 0
            assert item["lifted_length_bins"] == {
                "lt39": 0, "39_to_lt40": 0, "40_to_lt41": 0, "ge41": expected_count}
        assert sum(item["exact_ge40_count"] for item in passes) == fit_evidence["retained_segment_count"]
        if count >= 12:
            assert fit_evidence["result_segments"] == count
            assert fit_evidence["inliers"] == min(count, 15)
            assert fit_evidence["per_axis_inliers"] == [min(count, 15), 0, 0]
            assert fit_evidence["abstention_reason"] is None
        else:
            assert fit_evidence["result_segments"] is None
            assert fit_evidence["inliers"] is None and fit_evidence["per_axis_inliers"] is None
            assert fit_evidence["abstention_reason"] == "insufficient_segment_count"
    assert [row["initial_disposition"] for row in record["frames"]] == ["returned" if count >= 12 else "unavailable" for count in counts]
    assert [row["lattice_stage"] for row in record["frames"]] == ["pending" if count >= 12 else "not_reached" for count in counts]
    assert not accounting.inputs and not accounting.grays


def test_camera_forwards_same_objects_arguments_results_and_outside_call():
    ids, counts = ("image-0000", "image-0001"), (16, 18)
    modules = SimpleNamespace()
    raw = _install_authored_camera(modules, ids, counts)
    original = modules.calibration.estimate_camera
    received, returned = [], []
    def fit(*args, **kwargs):
        received.append((args, kwargs))
        result = original(*args, **kwargs)
        returned.append(result)
        return result
    modules.calibration.estimate_camera = fit
    before = _camera_callbacks(modules)
    accounting = subject._CameraAccounting()
    with subject._observe_camera(modules, accounting):
        result = modules.image_only.solve_image_cameras(raw)
        outside_gray, outside_segments = received[0][0][0], _AuthoredSegments(16)
        count = len(accounting.events)
        assert modules.calibration.estimate_camera(outside_gray, segments=outside_segments, focal=900) is returned[-1]
        assert received[-1][0][0] is outside_gray and received[-1][1]["segments"] is outside_segments
        assert received[-1][1]["focal"] == 900
        assert len(accounting.events) == count and accounting.fit_calls == len(ids)
    assert _camera_callbacks(modules) == before
    assert result[ids[0]] is returned[0] and result[ids[1]] is returned[1]
    assert all(not kwargs for _, kwargs in received[:2])
    assert [event["phase"] for event in accounting.events] == ["free_focal"] * 2


@pytest.mark.parametrize("case", ["rgb_alias", "gray_alias", "missing_fit", "duplicate_fit", "unknown_gray",
    "missing_conversion", "reordered_fits", "unexpected_session", "fixed_focal", "cached_segments", "second_solve", "extra_fit"])
def test_camera_ambiguous_missing_duplicate_unexpected_or_capped_capture_is_incomplete(case):
    ids, counts = ("image-0000", "image-0001"), (16, 18)
    modules = SimpleNamespace()
    raw = _install_authored_camera(modules, ids, counts)
    image_only, calibration = modules.image_only, modules.calibration
    if case == "rgb_alias": raw[ids[1]] = raw[ids[0]]
    elif case == "gray_alias":
        common = _OpaqueArray()
        image_only._to_grayscale = lambda image: common
    elif case != "second_solve":
        def solve(frames):
            grayscale = {frame: (_OpaqueArray() if case == "missing_conversion" else image_only._to_grayscale(image))
                         for frame, image in frames.items()}
            results = {}
            if case == "missing_fit": return results
            if case == "duplicate_fit": calibration.estimate_camera(grayscale[ids[0]])
            if case == "unknown_gray": calibration.estimate_camera(_OpaqueArray())
            if case == "unexpected_session": return calibration.estimate_session(grayscale)
            ordered = list(grayscale.items())
            if case == "reordered_fits": ordered.reverse()
            for frame, grey in ordered:
                kwargs = {"focal": 900} if case == "fixed_focal" else {"segments": _AuthoredSegments(16)} if case == "cached_segments" else {}
                results[frame] = calibration.estimate_camera(grey, **kwargs)
            if case == "extra_fit": calibration.estimate_camera(grayscale[ids[-1]])
            return results
        image_only.solve_image_cameras = solve
    before = _camera_callbacks(modules)
    accounting = subject._CameraAccounting()
    with subject._observe_camera(modules, accounting):
        modules.image_only.solve_image_cameras(raw)
        if case == "second_solve": modules.image_only.solve_image_cameras(raw)
    assert _camera_callbacks(modules) == before
    record = subject._camera_record(_camera_metadata(ids, counts), accounting, [{"frame_id": frame} for frame in ids])
    assert record["status"] == "incomplete" and all(row["capture"] == "incomplete" for row in record["frames"])
    assert all(row["segments"] is None and row["fixed_focal_disposition"] is None for row in record["frames"])
    assert record["session_calls"] == int(case == "unexpected_session")
    assert not accounting.inputs and not accounting.grays


def test_camera_identity_mapping_rejects_fit_call_reorder():
    ids, counts = ("image-0000", "image-0001"), (16, 18)
    modules = SimpleNamespace()
    raw = _install_authored_camera(modules, ids, counts)
    def reversed_fits(frames):
        gray = {frame: modules.image_only._to_grayscale(image) for frame, image in frames.items()}
        return {frame: modules.calibration.estimate_camera(image) for frame, image in reversed(list(gray.items()))}
    modules.image_only.solve_image_cameras = reversed_fits
    accounting = subject._CameraAccounting()
    with subject._observe_camera(modules, accounting): result = modules.image_only.solve_image_cameras(raw)
    assert set(result) == set(ids)
    record = subject._camera_record(_camera_metadata(ids, counts), accounting, [{"frame_id": frame} for frame in ids])
    assert record["status"] == "incomplete" and record["fit_calls"] == 2
    assert accounting.events[0]["frame_id"] == ids[1] and accounting.events[1]["frame_id"] == ids[0]
    assert modules.image_only.solve_image_cameras is reversed_fits


@pytest.mark.parametrize("stage", ["solve", "gray", "session", "fit"])
def test_camera_exception_identity_propagates_and_all_callbacks_restore(stage):
    ids = ("image-0000", "image-0001")
    modules = SimpleNamespace()
    raw = _install_authored_camera(modules, ids, (16, 18))
    error = RuntimeError("C:\\private\\image.png")
    def raise_original(*args, **kwargs): raise error
    owner, name = {"solve": (modules.image_only, "solve_image_cameras"), "gray": (modules.image_only, "_to_grayscale"),
        "session": (modules.calibration, "estimate_session"), "fit": (modules.calibration, "estimate_camera")}[stage]
    setattr(owner, name, raise_original)
    if stage == "session":
        modules.image_only.solve_image_cameras = lambda frames: modules.calibration.estimate_session({
            frame: modules.image_only._to_grayscale(image) for frame, image in frames.items()})
    before = _camera_callbacks(modules)
    accounting = subject._CameraAccounting()
    with pytest.raises(RuntimeError) as raised:
        with subject._observe_camera(modules, accounting): modules.image_only.solve_image_cameras(raw)
    assert raised.value is error and _camera_callbacks(modules) == before
    assert not accounting.inputs and not accounting.grays and accounting.complete is False
    record = subject._camera_record(_camera_metadata(ids, (16, 18)), accounting, [{"frame_id": frame} for frame in ids])
    assert record["status"] == "incomplete" and b"private" not in subject.canonical_bytes(record)
    if stage == "fit": assert accounting.fit_calls == 1 and accounting.events[0]["disposition"] == "raised"


@pytest.mark.parametrize("case", ["admission_order", "missing_admission", "foreign_active", "status_contradiction", "ingress_order",
                                  "input_order", "missing_input"])
def test_camera_admission_active_and_ingress_joins_fail_closed(case):
    ids, counts = ("image-0000", "image-0001"), (16, 11)
    modules = SimpleNamespace()
    raw = _install_authored_camera(modules, ids, counts)
    accounting = subject._CameraAccounting()
    with subject._observe_camera(modules, accounting): modules.image_only.solve_image_cameras(raw)
    session, ingress = _camera_metadata(ids, counts), [{"frame_id": frame} for frame in ids]
    if case == "admission_order": session.admissions = tuple(reversed(session.admissions))
    elif case == "missing_admission": session.admissions = session.admissions[:1]
    elif case == "foreign_active": session.frames = (SimpleNamespace(frame_id="foreign"),)
    elif case == "status_contradiction": session.admissions[1].status = "admitted"
    elif case == "input_order": session.input_frames = tuple(reversed(session.input_frames))
    elif case == "missing_input": session.input_frames = session.input_frames[:1]
    else: ingress.reverse()
    record = subject._camera_record(session, accounting, ingress)
    assert record["status"] == "incomplete" and all(row["capture"] == "incomplete" for row in record["frames"])


def test_camera_observer_on_off_preserves_underlying_calls_and_exact_solve_return():
    ids, counts = ("image-0000", "image-0001"), (16, 11)
    modules = SimpleNamespace()
    raw = _install_authored_camera(modules, ids, counts)
    fit, solve = modules.calibration.estimate_camera, modules.image_only.solve_image_cameras
    calls, returns = [], []
    def recorded_fit(*args, **kwargs):
        calls.append((args, kwargs))
        return fit(*args, **kwargs)
    def recorded_solve(*args, **kwargs):
        result = solve(*args, **kwargs)
        returns.append(result)
        return result
    modules.calibration.estimate_camera = recorded_fit
    modules.image_only.solve_image_cameras = recorded_solve
    before = _camera_callbacks(modules)
    plain = modules.image_only.solve_image_cameras(raw)
    baseline = list(calls)
    calls.clear()
    accounting = subject._CameraAccounting()
    with subject._observe_camera(modules, accounting): observed = modules.image_only.solve_image_cameras(raw)
    assert plain is returns[0] and observed is returns[1] and observed[ids[0]] is plain[ids[0]]
    assert len(calls) == len(baseline) == len(ids)
    assert all(left[0][0] is right[0][0] and left[1] == right[1] == {} for left, right in zip(calls, baseline))
    assert _camera_callbacks(modules) == before and not accounting.inputs and not accounting.grays
    record = subject._camera_record(_camera_metadata(ids, counts), accounting, [{"frame_id": frame} for frame in ids])
    assert record["status"] == "complete" and record["session_calls"] == 0


@pytest.mark.parametrize("returned,disposition,outcome", [
    (0, "unused", "no_grown_plane_regions"), (0, "unused", "insufficient_calibration_inliers"),
    (2, "periodicity", "image_edge_graph_growth"), (2, "accepted", "image_edge_graph_growth")])
def test_real_scalar_observer_splits_proposal_abstention_from_reader_rejection(returned, disposition, outcome):
    from mc_photogrammetry.reconstruction import region_growth
    ids = ("image-0000",)
    observer = region_growth._ScalarAttritionObserver(max_items=subject.ATTRITION_ITEMS, max_bytes=subject.ATTRITION_BYTES)
    before = region_growth._attrition_state.get()
    with region_growth._scalar_attrition_context(observer):
        _emit_authored_attrition(region_growth, ids, returned=returned, disposition=disposition, outcome=outcome)
    assert region_growth._attrition_state.get() is before
    record = subject._attrition_record(observer.record(), ids)
    assert record["status"] == "complete"
    assert record["frames"] == [{"frame_id": ids[0], "proposal_returned_count": returned,
        "proposal_outcome": outcome, "reader_stage": "reached" if returned else "not_reached", "reader_classified": returned,
        "reader_accepted": returned if disposition == "accepted" else 0,
        "reader_dispositions": {disposition: returned} if returned else {}}]
    assert "examples" not in subject.canonical_bytes(record).decode()
    assert "handoffs" not in record and "identity" not in subject.canonical_bytes(record).decode()


@pytest.mark.parametrize("case", ["incomplete", "omissions", "handoff_omissions", "missing_reader", "false_join", "count_mismatch",
    "unknown_frame", "path_code", "boolean_count", "count_cap", "row_cap", "missing_transition", "boolean_metric",
    "missing_proposal", "proposal_under_count", "proposal_over_count", "missing_generation_check"])
def test_attrition_incomplete_or_tampered_aggregate_cannot_claim_complete(case):
    from mc_photogrammetry.reconstruction import region_growth
    ids = ("image-0000",)
    observer = region_growth._ScalarAttritionObserver()
    with region_growth._scalar_attrition_context(observer): _emit_authored_attrition(region_growth, ids, returned=1)
    raw = observer.record()
    if case == "incomplete": raw["status"] = "incomplete"
    elif case == "omissions": raw["omitted_accounting_events"] = 1
    elif case == "handoff_omissions": raw["omitted_proposal_handoffs"] = 1
    elif case == "missing_reader": raw["stages"].pop("reading")
    elif case == "false_join": raw["checks"]["proposal_to_reading:" + ids[0]] = False
    elif case == "count_mismatch": raw["stages"]["reading"]["counts"][0]["count"] = 2
    elif case == "unknown_frame": raw["stages"]["reading"]["counts"][0]["frame"] = "foreign"
    elif case == "path_code": raw["stages"]["reading"]["counts"][0]["disposition"] = "C:\\private\\image.png"
    elif case == "boolean_count": raw["stages"]["reading"]["counts"][0]["count"] = True
    elif case == "count_cap": raw["stages"]["reading"]["classified"] = subject.ATTRITION_ITEMS + 1
    elif case == "row_cap": raw["stages"]["reading"]["counts"] *= subject.MAX_DIAGNOSTIC_ROWS + 1
    elif case == "missing_transition": raw["stages"]["frame"]["counts"] = []
    elif case == "boolean_metric": raw["stages"]["frame"]["counts"][0]["metrics"]["returned_count"]["total"] = True
    elif case == "missing_proposal": raw["stages"].pop("proposal")
    elif case == "proposal_under_count": raw["stages"]["proposal"]["counts"][0]["count"] = 0
    elif case == "proposal_over_count": raw["stages"]["proposal"]["counts"][0]["count"] = 2
    else: raw["checks"].pop("proposal_generation:" + ids[0])
    record = subject._attrition_record(raw, ids)
    assert record["status"] == "incomplete"
    assert "private" not in subject.canonical_bytes(record).decode()


def test_attrition_projection_does_not_traverse_examples_paths_or_detailed_payloads():
    from mc_photogrammetry.reconstruction import region_growth
    observer = region_growth._ScalarAttritionObserver()
    with region_growth._scalar_attrition_context(observer): _emit_authored_attrition(region_growth, ("image-0000",), returned=1)
    raw = observer.record()
    for stage in raw["stages"].values():
        stage["examples"] = _OpaqueArray()
        for group in stage["counts"]:
            group["details"] = {"pixels": _OpaqueArray(), "path": "C:\\private\\image.png"}
    raw["proposal_handoffs"] = _OpaqueArray()
    raw["stages"]["period_axis"] = _OpaqueArray()
    result = subject._attrition_record(raw, ("image-0000",))
    assert result["status"] == "complete"
    encoded = subject.canonical_bytes(result).decode()
    assert all(word not in encoded for word in ("examples", "details", "pixels", "private", "period_axis", "identity"))


@pytest.mark.parametrize("case", ["frame_order", "calibration_count", "abstention_frame", "path_reason", "abstention_cap",
    "inliers_cap", "inliers_over_segments", "axis_sum", "boolean_segment"])
def test_lattice_scalar_calibration_and_abstention_joins_are_bounded(case):
    ids = ("image-0000", "image-0001")
    result = SimpleNamespace(frame_ids=ids, camera_calibrations=_authored_calibrations(ids),
        abstentions=(SimpleNamespace(frame=ids[0], reason="unmeasurable_plane", detail=_OpaqueArray()),))
    baseline = subject._lattice_record(result, ids)
    assert baseline["status"] == "complete"
    assert baseline["frames"][0]["abstention_reasons"] == {"unmeasurable_plane": 1}
    if case == "frame_order": result.frame_ids = tuple(reversed(ids))
    elif case == "calibration_count": result.camera_calibrations = result.camera_calibrations[:1]
    elif case == "abstention_frame": result.abstentions[0].frame = "foreign"
    elif case == "path_reason": result.abstentions[0].reason = "C:\\private\\image.png"
    elif case == "abstention_cap": result.abstentions *= subject.MAX_DIAGNOSTIC_ROWS + 1
    elif case == "inliers_cap": result.camera_calibrations[0].inliers = subject.MAX_DIAGNOSTIC_COUNT + 1
    elif case == "inliers_over_segments": result.camera_calibrations[0].inliers = 21
    elif case == "axis_sum": result.camera_calibrations[0].per_axis_inliers = (1, 1, 1)
    else: result.camera_calibrations[0].segments = True
    assert subject._lattice_record(result, ids)["status"] == "incomplete"


@pytest.mark.parametrize("case", ["proposal_empty", "proposal_abstained", "reader_rejection", "observer_cap"])
def test_refused_lattice_keeps_same_pass_camera_attrition_digest_and_zero_matcher_csp(monkeypatch, case):
    paths, entry, pins, calls = _fake_metadata_chain(monkeypatch, "lattice")
    modules = subject._modules()
    ids = tuple(pin.frame_id for pin in pins)
    entry.session.version = 2
    entry.session.frames = (entry.session.input_frames[0],)
    entry.session.admissions = _camera_metadata(ids, (16, 11)).admissions
    camera_inputs = _install_authored_camera(modules, ids, (16, 11))

    def build(*args, **kwargs):
        calls.append("entry")
        modules.image_only.solve_image_cameras(camera_inputs)
        return entry

    def lattice(**kwargs):
        calls.append("lattice")
        returned = 2 if case == "reader_rejection" else 0
        outcome = "insufficient_calibration_inliers" if case == "proposal_abstained" else None
        _emit_authored_attrition(modules.region_growth, ids[:1], returned=returned, disposition="periodicity", outcome=outcome)
        reason = "unmeasurable_plane" if returned else outcome or "no_grown_plane_regions"
        return SimpleNamespace(status="unavailable", reason="no_lattice_reading", detail="no accepted reading",
            result_sha256="d" * 64, frame_ids=ids[:1], camera_calibrations=_authored_calibrations(ids[:1]),
            abstentions=(SimpleNamespace(frame=ids[0], reason=reason, detail=_OpaqueArray()),))

    modules.entry.build_image_derived_entry, modules.lattice.measure_image_derived_lattice = build, lattice
    if case == "observer_cap": monkeypatch.setattr(subject, "ATTRITION_ITEMS", 1)
    callbacks = _camera_callbacks(modules)
    original_context = modules.region_growth._attrition_state.get()
    result = subject.evaluate_pngs(paths, project_path="unwritten.mcscan")
    _assert_failed_metadata(result, entry, pins, "lattice")
    assert calls == ["entry", "lattice"] and _camera_callbacks(modules) == callbacks
    assert modules.region_growth._attrition_state.get() is original_context
    record = result.record
    assert record["camera_accounting"]["status"] == record["lattice_accounting"]["status"] == "complete"
    assert record["camera_accounting"]["frames"][1]["segments"] == 11
    assert record["camera_accounting"]["frames"][1]["segments_unavailable_reason"] is None
    assert record["camera_accounting"]["frames"][1]["calibration_evidence"]["abstention_reason"] == "insufficient_segment_count"
    assert record["camera_accounting"]["fitting_mode"] == "independent_per_frame"
    assert record["camera_accounting"]["session_calls"] == 0 and record["camera_accounting"]["fit_calls"] == len(ids)
    assert record["camera_accounting"]["frames"][1]["lattice_stage"] == "not_reached"
    assert record["camera_accounting"]["frames"][0]["lattice_stage"] == "reached"
    assert record["lattice_attrition"]["status"] == ("incomplete" if case == "observer_cap" else "complete")
    for field in ("camera_accounting", "lattice_accounting", "lattice_attrition", "diagnostic_accounting_status"):
        payload = dict(record)
        digest = payload.pop("record_sha256")
        payload.pop(field)
        assert subject._digest(payload) != digest


def test_incomplete_scalar_observer_refuses_paired_dispatch_without_fabricated_success(monkeypatch):
    paths, entry, pins, calls = _fake_metadata_chain(monkeypatch, "none")
    monkeypatch.setattr(subject, "ATTRITION_ITEMS", 1)
    result = subject.evaluate_pngs(paths, project_path="unwritten.mcscan")
    assert calls == ["entry", "lattice", "signatures"]
    assert result.record["reason"] == "diagnostic_accounting_incomplete"
    assert result.record["status"] == "inconclusive" and result.record["diagnostic_accounting_status"] == "incomplete"
    assert result.record["matcher_calls"] == 0 and result.record["csp_calls"] == []
    assert result.full_graph is result.kept_evaluation is None


def test_camera_and_attrition_frame_caps_refuse_without_payload_access():
    accounting = subject._CameraAccounting()
    ingress = [_OpaqueArray()] * (subject.MAX_IMAGES + 1)
    assert subject._camera_record(_OpaqueArray(), accounting, ingress)["status"] == "incomplete"
    assert subject._attrition_record(_OpaqueArray(), ["image-0000"] * (subject.MAX_IMAGES + 1))["status"] == "incomplete"
    assert subject._attrition_record(_OpaqueArray(), ["image-0000", "image-0000"])["status"] == "incomplete"


def test_camera_event_cap_preserves_original_calls_with_bounded_incomplete_record():
    ids = ("image-0000", "image-0001")
    modules = SimpleNamespace()
    raw = _install_authored_camera(modules, ids, (16, 18))
    called = []
    original_fit = modules.calibration.estimate_camera
    def fit(*args, **kwargs):
        called.append(1)
        return original_fit(*args, **kwargs)
    modules.calibration.estimate_camera = fit
    def solve(frames):
        gray = {frame: modules.image_only._to_grayscale(image) for frame, image in frames.items()}
        for _ in range(2 * subject.MAX_IMAGES + 2):
            modules.calibration.estimate_camera(gray[ids[0]])
        return {}
    modules.image_only.solve_image_cameras = solve
    before = _camera_callbacks(modules)
    accounting = subject._CameraAccounting()
    with subject._observe_camera(modules, accounting): modules.image_only.solve_image_cameras(raw)
    assert _camera_callbacks(modules) == before
    assert len(called) == accounting.fit_calls == 2 * subject.MAX_IMAGES + 2
    assert len(accounting.events) == 2 * subject.MAX_IMAGES and accounting.complete is False
    record = subject._camera_record(_camera_metadata(ids, (16, 18)), accounting, [{"frame_id": frame} for frame in ids])
    assert record["status"] == "incomplete" and all(row["segments"] is None for row in record["frames"])
    assert not accounting.inputs and not accounting.grays


@pytest.mark.parametrize("value,expected", [
    ("image-0000", "image-0000"), ("image-0031", "image-0031"), ("image-0032", None),
    ("image-9999", None), ("foreign", None), ("C:\\private\\image-0000.png", None),
    ("../image-0000", None), ("image-000", None), ("image-00000", None), ("Image-0000", None),
    ("image-000a", None), ("image-００００", None), (None, None), (True, None), (0, None), ([], None),
], ids=["first-production", "last-production", "index-cap", "index-over-cap", "foreign", "windows-path",
    "relative-path", "short-width", "long-width", "wrong-case", "not-digit", "unicode-digits", "none", "bool", "int", "list"])
def test_production_frame_id_validator_is_separate_from_reason_codes(value, expected):
    assert subject._frame_id(value) == expected
    assert subject._code("image-0000") is None
    assert subject._code("C:\\private\\image.png") is None
    assert subject._code("no_lattice_reading") == "no_lattice_reading"


@pytest.mark.parametrize("frame", ["image-0000", "image-0031", "foreign", "C:\\private\\image-0000.png", "image-0032"],
    ids=["first-production", "last-production", "foreign", "path", "index-cap"])
def test_frame_id_projection_joins_accept_production_ids_and_reject_foreign_paths(frame):
    modules = SimpleNamespace()
    raw = _install_authored_camera(modules, (frame,), (16,))
    accounting = subject._CameraAccounting()
    with subject._observe_camera(modules, accounting): modules.image_only.solve_image_cameras(raw)
    camera = subject._camera_record(_camera_metadata((frame,), (16,)), accounting, [{"frame_id": frame}])
    lattice = subject._lattice_record(SimpleNamespace(frame_ids=(frame,), camera_calibrations=_authored_calibrations((frame,)),
        abstentions=(SimpleNamespace(frame=frame, reason="unmeasurable_plane"),)), (frame,))
    observer = modules.region_growth._ScalarAttritionObserver()
    with modules.region_growth._scalar_attrition_context(observer):
        _emit_authored_attrition(modules.region_growth, (frame,), returned=1)
    attrition = subject._attrition_record(observer.record(), (frame,))
    valid = frame in ("image-0000", "image-0031")
    assert [item["status"] for item in (camera, lattice, attrition)] == (["complete"] * 3 if valid else ["incomplete"] * 3)
    if valid:
        assert camera["active_frame_ids"] == [frame]
        assert lattice["frames"][0]["frame_id"] == attrition["frames"][0]["frame_id"] == frame
    else:
        assert camera["active_frame_ids"] == camera["frames"] == lattice["frames"] == attrition["frames"] == []
        assert "private" not in subject.canonical_bytes({"camera": camera, "lattice": lattice, "attrition": attrition}).decode()


def test_three_frame_same_pass_telemetry_preserves_transitive_objects_outputs_and_counts():
    ids, counts = ("image-0000", "image-0001", "image-0002"), (16, 11, 18)
    modules = SimpleNamespace()
    raw = _install_authored_camera(modules, ids, counts)
    fixture = modules.camera_fixture
    raw_counts = ((14, 10), (8, 5), (20, 0))
    for frame, retained, pass_counts in zip(ids, counts, raw_counts):
        first = min(retained, pass_counts[0])
        fixture.detector_outputs[frame] = tuple((None if count == 0 else _authored_line_rows(
            count, eligible=eligible, scale=scale), None, None, None)
            for count, eligible, scale in zip(pass_counts, (first, retained - first), (1.0, 0.5)))
    callbacks = _camera_callbacks(modules)
    plain = modules.image_only.solve_image_cameras(raw)
    trace = list(fixture.trace)
    fixture.trace.clear()
    accounting = subject._CameraAccounting()
    with subject._observe_camera(modules, accounting): observed = modules.image_only.solve_image_cameras(raw)
    assert _camera_callbacks(modules) == callbacks
    assert set(observed) == set(plain) == {ids[0], ids[2]}
    assert all(observed[frame] is plain[frame] is fixture.estimates[frame] for frame in observed)
    assert len(trace) == len(fixture.trace) == 6 * len(ids)
    for left, right in zip(trace, fixture.trace):
        assert left[0] == right[0] and len(left[1]) == len(right[1])
        assert all(a is b for a, b in zip(left[1], right[1]))
        assert left[2] == right[2]
        if left[0] != "factory": assert left[3] is right[3]
    assert sum(row[0] == "detector" for row in fixture.trace) == 2 * len(ids)
    assert sum(row[0] == "resize" for row in fixture.trace) == len(ids)
    record = subject._camera_record(_camera_metadata(ids, counts), accounting, [{"frame_id": frame} for frame in ids])
    assert record["status"] == "complete" and record["session_calls"] == 0 and record["fit_calls"] == 3
    for row, frame, retained, pass_counts in zip(record["frames"], ids, counts, raw_counts):
        evidence = row["calibration_evidence"]
        assert evidence["status"] == "complete" and evidence["identity_references_cleared"] is True
        passes = evidence["raw_detections_per_pass"]
        assert [(row["scale"], row["raw_detection_count"]) for row in passes] == list(zip((1.0, 0.5), pass_counts))
        assert sum(row["exact_ge40_count"] for row in passes) == retained
        assert all(row["finite_row_count"] == row["raw_detection_count"] and row["nonfinite_row_count"] == 0
                   and row["nonfinite_eligible_count"] == 0 for row in passes)
        assert evidence["retained_segment_count"] == row["segments"] == retained
        assert evidence["enhance_calls"] == evidence["segment_calls"] == 1
        assert evidence["detector_calls"] == 2 and evidence["resize_calls"] == 1
        if retained < 12:
            assert evidence["abstention_reason"] == "insufficient_segment_count"
            assert evidence["result_segments"] is evidence["inliers"] is evidence["per_axis_inliers"] is None
        else:
            result = fixture.estimates[frame]
            assert evidence["result_segments"] == result.segments
            assert evidence["inliers"] == result.inliers and evidence["per_axis_inliers"] == list(result.per_axis_inliers)
    assert not accounting.inputs and not accounting.grays
    encoded = subject.canonical_bytes(record)
    assert b"pixels" not in encoded and b"coordinates" not in encoded and b"private" not in encoded


def test_same_pass_zero_detections_are_observed_not_missing_or_unknown():
    ids = ("image-0000",)
    modules = SimpleNamespace()
    raw = _install_authored_camera(modules, ids, (0,))
    modules.camera_fixture.detector_outputs[ids[0]] = ((None, None, None, None),) * 2
    accounting = subject._CameraAccounting()
    with subject._observe_camera(modules, accounting): assert modules.image_only.solve_image_cameras(raw) == {}
    record = subject._camera_record(_camera_metadata(ids, (0,)), accounting, [{"frame_id": ids[0]}])
    evidence = record["frames"][0]["calibration_evidence"]
    assert record["status"] == "complete" and record["frames"][0]["segments"] == 0
    assert [row["raw_detection_count"] for row in evidence["raw_detections_per_pass"]] == [0, 0]
    assert evidence["detector_calls"] == 2 and evidence["abstention_reason"] == "insufficient_segment_count"


@pytest.mark.parametrize("case", [
    "foreign_enhance", "missing_enhance", "duplicate_enhance", "foreign_segments", "missing_segments", "duplicate_segments",
    "foreign_resize", "foreign_detector", "missing_resize", "duplicate_resize", "aliased_resize", "missing_detector",
    "extra_detector", "reordered_passes", "bypass_factory", "raw_cap", "retained_cap", "retained_exceeds_raw",
    "result_segments", "inlier_overflow", "axis_sum", "unknown_abstention"])
def test_same_pass_missing_foreign_aliased_or_contradictory_evidence_fails_closed(case):
    ids, counts = ("image-0000", "image-0001"), (16, 18)
    modules = SimpleNamespace()
    raw = _install_authored_camera(modules, ids, counts)
    fixture, calibration, cv2 = modules.camera_fixture, modules.calibration, modules.cv2
    original_fit, original_detect, original_resize = calibration.estimate_camera, calibration.detect_segments, cv2.resize
    if case in ("foreign_enhance", "missing_enhance", "duplicate_enhance", "foreign_segments", "missing_segments", "duplicate_segments"):
        def fit(image, *args, **kwargs):
            if image is not fixture.gray[ids[0]]: return original_fit(image, *args, **kwargs)
            enhanced = fixture.enhanced[ids[0]] if case == "missing_enhance" else calibration.enhance(
                fixture.gray[ids[1]] if case == "foreign_enhance" else image)
            if case == "duplicate_enhance": enhanced = calibration.enhance(image)
            if case != "missing_segments": calibration.detect_segments(
                fixture.enhanced[ids[1]] if case == "foreign_segments" else enhanced)
            if case == "duplicate_segments": calibration.detect_segments(enhanced)
            return fixture.estimates[ids[0]]
        calibration.estimate_camera = fit
    elif case in ("foreign_resize", "foreign_detector", "missing_resize", "duplicate_resize", "missing_detector",
                  "extra_detector", "reordered_passes", "bypass_factory"):
        delegate = cv2.createLineSegmentDetector() if case == "bypass_factory" else None
        fixture.trace.clear()
        def detect(image, **kwargs):
            if image is not fixture.enhanced[ids[0]]: return original_detect(image, **kwargs)
            detector = delegate if delegate is not None else cv2.createLineSegmentDetector()
            if case == "reordered_passes":
                resized = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                detector.detect(resized)
                detector.detect(image)
            else:
                detector.detect(image)
                if case != "missing_detector":
                    resized = fixture.resized[ids[0]] if case == "missing_resize" else cv2.resize(
                        fixture.enhanced[ids[1]] if case == "foreign_resize" else image,
                        None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                    if case == "duplicate_resize": resized = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                    detector.detect(fixture.resized[ids[1]] if case == "foreign_detector" else resized)
                    if case == "extra_detector": detector.detect(resized)
            return fixture.retained[ids[0]]
        calibration.detect_segments = detect
    elif case == "aliased_resize": cv2.resize = lambda image, *args, **kwargs: image
    elif case == "raw_cap": fixture.detector_outputs[ids[0]] = ((_AuthoredSegments(subject.MAX_DIAGNOSTIC_COUNT + 1), None), (None, None))
    elif case == "retained_cap": fixture.retained[ids[0]].count = subject.MAX_DIAGNOSTIC_COUNT + 1
    elif case == "retained_exceeds_raw": fixture.detector_outputs[ids[0]] = ((_AuthoredSegments(1), None), (None, None))
    elif case == "result_segments": fixture.estimates[ids[0]].segments += 1
    elif case == "inlier_overflow":
        fixture.estimates[ids[0]].inliers = 17
        fixture.estimates[ids[0]].per_axis_inliers = (17, 0, 0)
    elif case == "axis_sum": fixture.estimates[ids[0]].per_axis_inliers = (14, 0, 0)
    else: fixture.estimates[ids[0]] = None
    callbacks = _camera_callbacks(modules)
    accounting = subject._CameraAccounting()
    with subject._observe_camera(modules, accounting): result = modules.image_only.solve_image_cameras(raw)
    assert _camera_callbacks(modules) == callbacks and not accounting.inputs and not accounting.grays
    record = subject._camera_record(_camera_metadata(ids, counts), accounting, [{"frame_id": frame} for frame in ids])
    assert record["status"] == "incomplete" and record["frames"][0]["segments"] is None
    evidence = accounting.events[0]["calibration_evidence"]
    assert evidence["status"] == "incomplete" and evidence["identity_references_cleared"] is True
    assert len(evidence["raw_detections_per_pass"]) <= 4
    if case == "unknown_abstention": assert evidence["abstention_reason"] is None and ids[0] not in result
    subject.canonical_bytes(record)


@pytest.mark.parametrize("stage", ["enhance", "segments", "resize", "factory", "detector"])
def test_same_pass_exception_identity_redaction_and_all_callback_restoration(stage):
    ids = ("image-0000",)
    modules = SimpleNamespace()
    raw = _install_authored_camera(modules, ids, (16,))
    error = RuntimeError("C:\\private\\screenshot.png")
    def fail(*args, **kwargs): raise error
    if stage == "enhance": modules.calibration.enhance = fail
    elif stage == "segments": modules.calibration.detect_segments = fail
    elif stage == "resize": modules.cv2.resize = fail
    elif stage == "factory": modules.cv2.createLineSegmentDetector = fail
    else: modules.cv2.createLineSegmentDetector = lambda: SimpleNamespace(detect=fail)
    callbacks = _camera_callbacks(modules)
    accounting = subject._CameraAccounting()
    with pytest.raises(RuntimeError) as raised:
        with subject._observe_camera(modules, accounting): modules.image_only.solve_image_cameras(raw)
    assert raised.value is error and _camera_callbacks(modules) == callbacks
    assert not accounting.inputs and not accounting.grays
    evidence = accounting.events[0]["calibration_evidence"]
    assert evidence["status"] == "incomplete" and evidence["identity_references_cleared"] is True
    assert accounting.events[0]["disposition"] == "raised"
    record = subject._camera_record(_camera_metadata(ids, (16,)), accounting, [{"frame_id": ids[0]}])
    assert record["status"] == "incomplete" and b"private" not in subject.canonical_bytes(record)


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("scale", [1.0, 0.75, 0.5], ids=["full", "three-quarter", "half"])
@pytest.mark.parametrize("layout", [(4,), (1, 4)], ids=["opencv5-flat", "legacy-channel"])
def test_line_pass_metrics_boundary_scaling_and_nonfinite_production_comparison(dtype, scale, layout):
    # The production comparison keeps an infinite-length row and drops NaN.
    # Finite counts/bins describe evidence, without changing that comparison.
    full = np.asarray([[0, 0, length, 0] for length in (38, 39, 39.999, 40, 40.001, 41)] +
                      [[0, 0, 24, 32], [0, 0, float("nan"), 0], [0, 0, float("inf"), 0]], dtype=dtype)
    rows = (full * scale).reshape(-1, *layout)
    before = rows.tobytes()
    metrics = subject._line_pass_metrics((rows, None, None, None), scale)
    assert metrics == {"raw_detection_count": 9, "finite_row_count": 7, "nonfinite_row_count": 2,
        "lifted_length_bins": {"lt39": 1, "39_to_lt40": 2, "40_to_lt41": 3, "ge41": 1},
        "exact_ge40_count": 5, "nonfinite_eligible_count": 1}
    assert rows.tobytes() == before
    with np.errstate(all="ignore"):
        lifted = rows.reshape(-1, 4) / scale
        eligible = np.hypot(lifted[:, 2] - lifted[:, 0], lifted[:, 3] - lifted[:, 1]) >= 40.0
    assert int(eligible.sum()) == metrics["exact_ge40_count"]
    assert sum(metrics["lifted_length_bins"].values()) == metrics["finite_row_count"]
    assert subject.canonical_bytes(metrics)


@pytest.mark.parametrize("found", [None, np.empty((0, 1, 4), dtype=np.float32), np.empty((0, 4), dtype=np.float32)],
                         ids=["none", "empty-channel", "empty-flat"])
def test_line_pass_metrics_none_and_empty_have_explicit_zero_counts(found):
    metrics = subject._line_pass_metrics((found,), 0.5)
    assert all(value == 0 for key, value in metrics.items() if key != "lifted_length_bins")
    assert all(value == 0 for value in metrics["lifted_length_bins"].values())


@pytest.mark.parametrize("case", ["shape", "dtype", "subclass", "cap", "tuple", "scale", "arithmetic"])
def test_line_pass_metrics_malformed_capped_or_failed_measurements_are_redacted(monkeypatch, case):
    rows = _authored_line_rows(2)
    result, scale = (rows,), 1.0
    if case == "shape": result = (rows.reshape(2, 2, 2),)
    elif case == "dtype": result = (rows.astype(object),)
    elif case == "subclass":
        class Foreign(np.ndarray): pass
        result = (rows.view(Foreign),)
    elif case == "cap": monkeypatch.setattr(subject, "MAX_LINE_PASS_ROWS", 1)
    elif case == "tuple": result = [rows]
    elif case == "scale": scale = True
    else:
        def fail(*args, **kwargs): raise ValueError("C:\\private\\rows.png")
        monkeypatch.setattr(np, "hypot", fail)
    metrics = subject._line_pass_metrics(result, scale)
    assert all(value is None for value in metrics.values())
    assert b"private" not in subject.canonical_bytes(metrics)


@pytest.mark.parametrize("mismatch", [-1, 1], ids=["retained_under", "retained_over"])
def test_line_pass_metrics_retained_eligibility_mismatch_marks_observer_incomplete(mismatch):
    ids, counts = ("image-0000",), (16,)
    modules = SimpleNamespace()
    raw = _install_authored_camera(modules, ids, counts)
    fixture = modules.camera_fixture
    fixture.detector_outputs[ids[0]] = ((_authored_line_rows(16, eligible=16 + min(mismatch, 0)),),
                                       (_authored_line_rows(max(mismatch, 0), scale=0.5),))
    callbacks = _camera_callbacks(modules)
    accounting = subject._CameraAccounting()
    with subject._observe_camera(modules, accounting): observed = modules.image_only.solve_image_cameras(raw)
    assert observed[ids[0]] is fixture.estimates[ids[0]]
    assert _camera_callbacks(modules) == callbacks
    evidence = accounting.events[0]["calibration_evidence"]
    assert evidence["status"] == "incomplete" and evidence["retained_segment_count"] == 16
    assert sum(row["exact_ge40_count"] for row in evidence["raw_detections_per_pass"]) == 16 + mismatch
    assert evidence["identity_references_cleared"] is True
    record = subject._camera_record(_camera_metadata(ids, counts), accounting, [{"frame_id": ids[0]}])
    assert record["status"] == "incomplete" and record["frames"][0]["segments"] is None
    assert b"coordinates" not in subject.canonical_bytes(record)


@pytest.mark.parametrize("layout", [(4,), (1, 4)], ids=["opencv5-flat", "legacy-channel"])
def test_line_pass_metrics_real_filter_preserves_nan_inf_returns_and_two_calls(monkeypatch, layout):
    import sys
    from mc_photogrammetry.reconstruction import calibration

    ids, counts = ("image-0000",), (10,)
    modules = SimpleNamespace()
    raw = _install_authored_camera(modules, ids, counts)
    fixture = modules.camera_fixture
    full = np.asarray([[0, 0, length, 0] for length in (38, 39, 39.999, 40, 40.001, 41)] +
                      [[0, 0, 24, 32], [0, 0, float("nan"), 0], [0, 0, float("inf"), 0]], dtype=np.float32)
    fixture.detector_outputs[ids[0]] = ((full.reshape(-1, *layout),), ((full * 0.5).reshape(-1, *layout),))
    modules.calibration.detect_segments = calibration.detect_segments
    monkeypatch.setitem(sys.modules, "cv2", modules.cv2)
    callbacks = _camera_callbacks(modules)
    accounting = subject._CameraAccounting()
    with subject._observe_camera(modules, accounting): assert modules.image_only.solve_image_cameras(raw) == {}
    assert _camera_callbacks(modules) == callbacks
    evidence = accounting.events[0]["calibration_evidence"]
    assert evidence["status"] == "complete" and evidence["retained_segment_count"] == 10
    assert [row["nonfinite_row_count"] for row in evidence["raw_detections_per_pass"]] == [2, 2]
    assert [row["nonfinite_eligible_count"] for row in evidence["raw_detections_per_pass"]] == [1, 1]
    detector_trace = [row for row in fixture.trace if row[0] == "detector"]
    assert len(detector_trace) == 2
    assert all(row[3] is expected for row, expected in zip(detector_trace, fixture.detector_outputs[ids[0]]))
    assert evidence["identity_references_cleared"] is True
    assert b"Infinity" not in subject.canonical_bytes(evidence)


@pytest.mark.parametrize("case", ["shape", "cap", "arithmetic"])
def test_line_pass_metrics_observation_failure_preserves_outputs_and_callbacks(monkeypatch, case):
    ids = ("image-0000",)
    modules = SimpleNamespace()
    raw = _install_authored_camera(modules, ids, (16,))
    fixture = modules.camera_fixture
    if case == "shape": fixture.detector_outputs[ids[0]] = ((np.zeros((16, 2, 2)),), (None,))
    elif case == "cap": monkeypatch.setattr(subject, "MAX_LINE_PASS_ROWS", 1)
    else:
        def fail(*args, **kwargs): raise RuntimeError("C:\\private\\payload.png")
        monkeypatch.setattr(np, "hypot", fail)
    callbacks = _camera_callbacks(modules)
    accounting = subject._CameraAccounting()
    with subject._observe_camera(modules, accounting): observed = modules.image_only.solve_image_cameras(raw)
    assert observed[ids[0]] is fixture.estimates[ids[0]] and _camera_callbacks(modules) == callbacks
    evidence = accounting.events[0]["calibration_evidence"]
    assert evidence["status"] == "incomplete" and evidence["identity_references_cleared"] is True
    assert sum(row[0] == "detector" for row in fixture.trace) == 2
    assert b"private" not in subject.canonical_bytes(evidence)


@pytest.mark.parametrize("scales", [subject.BASE_DETECTOR_SCALES, subject.CANDIDATE_DETECTOR_SCALES],
                         ids=["baseline", "candidate"])
def test_line_pass_metrics_installed_opencv_generated_seams_match_production_return(monkeypatch, scales):
    import cv2
    from mc_photogrammetry.reconstruction import calibration

    # Authored pixels only; installed detector representation is part of this
    # regression, so a native-return layout cannot be hidden by a fake fixture.
    image = np.full((384, 512), 50, dtype=np.uint8)
    for x in (80, 160, 240, 320, 400, 480): image[24:360, x:x + 2] = 210
    for y in (80, 160, 240, 320): image[y:y + 2, 24:488] = 210
    enhanced = calibration.enhance(image)
    original_factory = cv2.createLineSegmentDetector
    outputs, metrics, factories = [], [], []

    class Proxy:
        def __init__(self, delegate): self.delegate = delegate
        def detect(self, actual):
            result = self.delegate.detect(actual)
            ordinal = len(outputs)
            assert ordinal < len(scales)
            outputs.append(result)
            before = result[0].tobytes() if result[0] is not None else None
            metrics.append(subject._line_pass_metrics(result, scales[ordinal]))
            assert result[0] is None or result[0].tobytes() == before
            return result

    def factory(*args, **kwargs):
        factories.append((args, kwargs))
        return Proxy(original_factory(*args, **kwargs))

    with monkeypatch.context() as patch:
        patch.setattr(cv2, "createLineSegmentDetector", factory)
        retained = calibration.detect_segments(enhanced, scales=scales)
    assert cv2.createLineSegmentDetector is original_factory
    assert factories == [((), {})] and len(outputs) == len(metrics) == len(scales)
    assert all(row["raw_detection_count"] is not None and row["raw_detection_count"] > 0 for row in metrics)
    assert sum(row["exact_ge40_count"] for row in metrics) == len(retained) > 0
    assert all(row["finite_row_count"] == row["raw_detection_count"] and row["nonfinite_row_count"] == 0
               for row in metrics)
    lifted = np.vstack([result[0].reshape(-1, 4) / scale
                        for result, scale in zip(outputs, scales)])
    lengths = np.hypot(lifted[:, 2] - lifted[:, 0], lifted[:, 3] - lifted[:, 1])
    assert np.array_equal(retained, lifted[lengths >= calibration.MINIMUM_LENGTH])
    assert b"coordinates" not in subject.canonical_bytes(metrics)


def _enable_authored_three_scale_camera(modules, ids, counts):
    fixture, calibration, cv2 = modules.camera_fixture, modules.calibration, modules.cv2
    middle = {frame: _OpaqueArray() for frame in ids}
    received = []
    for frame, count in zip(ids, counts):
        first, second = count // 3, count // 3
        pass_counts = (first, second, count - first - second)
        fixture.detector_outputs[frame] = tuple((_authored_line_rows(n, scale=scale),)
            for n, scale in zip(pass_counts, subject.CANDIDATE_DETECTOR_SCALES))

    def resize(image, size, **kwargs):
        frame = next(frame for frame in ids if fixture.enhanced[frame] is image)
        result = middle[frame] if kwargs["fx"] == 0.75 else fixture.resized[frame]
        fixture.trace.append(("resize", (image, size), kwargs, result))
        return result

    class Detector:
        def detect(self, image):
            frame, ordinal = next((frame, ordinal) for frame in ids for ordinal, value in enumerate(
                (fixture.enhanced[frame], middle[frame], fixture.resized[frame])) if value is image)
            result = fixture.detector_outputs[frame][ordinal]
            fixture.trace.append(("detector", (image,), {}, result))
            return result

    def factory(*args, **kwargs):
        result = Detector()
        fixture.trace.append(("factory", args, kwargs, result))
        return result

    def fit(image, *args, **kwargs):
        received.append((image, args, dict(kwargs)))
        frame = next(frame for frame in ids if fixture.gray[frame] is image)
        calibration.detect_segments(calibration.enhance(image),
            scales=kwargs.get("scales", calibration.DETECTION_SCALES), minimum_length=40.0)
        return fixture.estimates[frame]

    cv2.resize, cv2.createLineSegmentDetector, calibration.estimate_camera = resize, factory, fit
    return received


@pytest.mark.parametrize("count", [7, 18], ids=["abstain", "admitted"])
def test_candidate_scales_three_pass_identity_counts_forwarding_and_defaults_unchanged(count):
    from mc_photogrammetry.reconstruction import calibration as production

    defaults = dict(production.estimate_camera.__kwdefaults__)
    production_scales = production.DETECTION_SCALES
    ids = ("image-0000",)
    modules = SimpleNamespace()
    raw = _install_authored_camera(modules, ids, (count,))
    received = _enable_authored_three_scale_camera(modules, ids, (count,))
    callbacks = _camera_callbacks(modules)
    accounting = subject._CameraAccounting()
    with subject._observe_camera(modules, accounting, detector_scales=subject.CANDIDATE_DETECTOR_SCALES):
        observed = modules.image_only.solve_image_cameras(raw)
        # Outside an active entry solve the original callback gets no override.
        modules.calibration.estimate_camera(modules.camera_fixture.gray[ids[0]])
    assert received[0][0] is received[1][0] is modules.camera_fixture.gray[ids[0]]
    assert received[0][1:] == ((), {"scales": subject.CANDIDATE_DETECTOR_SCALES})
    assert received[1][1:] == ((), {})
    assert modules.calibration.DETECTION_SCALES == subject.BASE_DETECTOR_SCALES
    assert production.DETECTION_SCALES is production_scales
    assert production.estimate_camera.__kwdefaults__ == defaults
    assert _camera_callbacks(modules) == callbacks
    assert accounting.fit_calls == 1 and accounting.session_calls == 0
    evidence = accounting.events[0]["calibration_evidence"]
    assert evidence["status"] == "complete" and evidence["identity_references_cleared"] is True
    assert evidence["enhance_calls"] == evidence["segment_calls"] == 1
    assert evidence["detector_calls"] == 3 and evidence["resize_calls"] == 2
    assert [row["scale"] for row in evidence["raw_detections_per_pass"]] == [1.0, 0.75, 0.5]
    assert sum(row["exact_ge40_count"] for row in evidence["raw_detections_per_pass"]) == count
    if count < 12:
        assert observed == {}
    else:
        assert observed[ids[0]] is modules.camera_fixture.estimates[ids[0]]
    assert evidence["abstention_reason"] == ("insufficient_segment_count" if count < 12 else None)


@pytest.mark.parametrize("scales", [(1.0, 0.5), (1.0, 0.8, 0.5), (1.0, 0.75, 0.75, 0.5),
                                     [1.0, 0.75, 0.5], (True, 0.75, 0.5)])
def test_candidate_scales_unknown_configuration_refuses_before_callback_mutation(scales):
    modules = SimpleNamespace()
    _install_authored_camera(modules, ("image-0000",), (18,))
    callbacks = _camera_callbacks(modules)
    with pytest.raises(subject.PairedRefusal):
        with subject._observe_camera(modules, subject._CameraAccounting(), detector_scales=scales):
            pytest.fail("invalid scales entered observation")
    assert _camera_callbacks(modules) == callbacks and modules.camera_fixture.trace == []


def test_candidate_scales_exception_is_forwarded_and_all_callbacks_restored():
    modules = SimpleNamespace()
    ids = ("image-0000",)
    raw = _install_authored_camera(modules, ids, (18,))
    _enable_authored_three_scale_camera(modules, ids, (18,))
    error = RuntimeError("C:\\private\\candidate.png")
    def fail(image, **kwargs):
        assert kwargs == {"scales": subject.CANDIDATE_DETECTOR_SCALES}
        raise error
    modules.calibration.estimate_camera = fail
    callbacks = _camera_callbacks(modules)
    accounting = subject._CameraAccounting()
    with pytest.raises(RuntimeError) as raised:
        with subject._observe_camera(modules, accounting, detector_scales=subject.CANDIDATE_DETECTOR_SCALES):
            modules.image_only.solve_image_cameras(raw)
    assert raised.value is error and _camera_callbacks(modules) == callbacks
    assert not accounting.inputs and not accounting.grays
    assert accounting.events[0]["calibration_evidence"]["status"] == "incomplete"
    assert b"private" not in subject.canonical_bytes(accounting.events)


@pytest.mark.parametrize("lattice_status", ["assembled", "unavailable"])
def test_candidate_scales_stops_at_lattice_with_diagnostic_status_and_digest(monkeypatch, lattice_status):
    returned = SimpleNamespace(status=lattice_status, reason="" if lattice_status == "assembled" else "no_lattice_reading",
        detail="authored lattice outcome", result_sha256="d" * 64,
        frame_ids=("image-0000", "image-0001"), camera_calibrations=_authored_calibrations(("image-0000", "image-0001")),
        abstentions=())
    paths, entry, pins, calls = _fake_metadata_chain(monkeypatch, "lattice", returned=returned)
    modules = subject._modules()
    _enable_authored_three_scale_camera(modules, returned.frame_ids, (16, 18))
    returned.camera_calibrations = tuple(modules.camera_fixture.estimates[frame] for frame in returned.frame_ids)
    def forbidden(*args, **kwargs): pytest.fail("candidate reached downstream evaluation")
    modules.signatures.derive_image_derived_signatures = forbidden
    result = subject.evaluate_pngs(paths, project_path="unwritten.mcscan", candidate_detector_scales=True)
    record = result.record
    assert calls == ["entry", "lattice"]
    assert record["status"] == "inconclusive" and record["field"] == "lattice"
    assert record["reason"] == ("candidate_camera_lattice_diagnostic" if lattice_status == "assembled" else "upstream_unavailable")
    assert record["diagnostic_accounting_status"] == "complete"
    assert record["experiment_settings"]["detector_scales"] == [1.0, 0.75, 0.5]
    assert record["experiment_settings"]["baseline_detector_scales"] == [1.0, 0.5]
    assert record["experiment_settings"]["minimum_length"] == 40 and record["experiment_settings"]["camera_segment_gate"] == 12
    assert record["experiment_settings"]["lattice_inlier_gate"] == 12
    assert record["geometry_promoted"] is record["state_scoring_performed"] is False
    assert record["matcher_calls"] == 0 and record["csp_calls"] == [] and record["certificate_validations"] == []
    assert result.full_graph is result.kept_evaluation is None
    assert record["upstream_stages"]["lattice"]["status"] == lattice_status
    assert [row["segments"] for row in record["lattice_accounting"]["frames"]] == [16, 18]
    assert [row["inliers"] for row in record["lattice_accounting"]["frames"]] == [15, 15]
    payload = dict(record)
    digest = payload.pop("record_sha256")
    assert subject._digest(payload) == digest
    changed = dict(payload); changed.pop("experiment_settings")
    assert subject._digest(changed) != digest


def test_candidate_scales_cli_flag_is_explicit_and_defaults_are_not_overwritten(monkeypatch, tmp_path):
    received = []
    def evaluate(images, **kwargs):
        received.append((images, kwargs))
        return subject.PairedResult({"status": "inconclusive", "reason": "candidate_camera_lattice_diagnostic"})
    monkeypatch.setattr(subject, "evaluate_pngs", evaluate)
    output = tmp_path / "candidate.json"
    assert subject.main(["--project", str(tmp_path / "unused.mcscan"), "--output", str(output),
                         "--candidate-detector-scales", "authored.png"]) == 2
    assert received == [(["authored.png"], {"project_path": str(tmp_path / "unused.mcscan"),
                                          "candidate_detector_scales": True})]
    assert json.loads(output.read_bytes())["status"] == "inconclusive"
