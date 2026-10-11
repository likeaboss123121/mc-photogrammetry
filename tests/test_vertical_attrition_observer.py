"""Scalar-only synthetic accounting and output-neutral observer controls."""
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from mc_photogrammetry.reconstruction import region_growth as growth
from mc_photogrammetry.reconstruction import lattice_planes as planes, plane_graph
from mc_photogrammetry.pipeline import image_derived_lattice_measure as measure
from mc_photogrammetry.pipeline import image_derived_signatures as signatures
from scripts.evaluation import diagnose_image_geometry_failure as runner
from test_lattice_planes import _v1_region_control, _controlled_v1_readings
from test_plane_graph import make_reading, make_signature
from test_image_derived_signatures import records
from test_image_derived_lattice_measure import _result


def _closed(observer, stage):
    record = observer.record()
    assert record["status"] == "complete", record
    runner._validate_attrition_record(record)
    row = record["stages"][stage]
    assert row["expected"] == row["classified"]
    assert not row["unclassified"] and not row["multiply_classified"] and not row["unexpected"]
    return row


def _track(monkeypatch, targets):
    calls = []
    for owner, name in targets:
        original = getattr(owner, name)
        def observed(*args, _original=original, _name=name, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)
        monkeypatch.setattr(owner, name, observed)
    return calls


def test_proposal_absence_and_exact_seed_gates_preserve_refusal(monkeypatch):
    proposal, usable, calibration, _polygon = _v1_region_control()
    frame = SimpleNamespace(frame_id="absent", image=np.zeros((*usable.shape, 3), np.uint8))
    monkeypatch.setattr(growth, "propose_image_planes_v2",
                        lambda *_a, **_k: growth._empty("insufficient_plane_seeds"))
    calls = _track(monkeypatch, ((growth, "propose_image_planes_v2"),))
    normal = measure._measure_frame(frame, calibration)
    normal_calls = list(calls)
    calls.clear()
    observer = growth._ScalarAttritionObserver()
    with growth._scalar_attrition_context(observer):
        observed = measure._measure_frame(frame, calibration)
    assert normal == observed and calls == normal_calls
    assert observed[1][0].reason == "insufficient_plane_seeds"
    assert _closed(observer, "seed")["classified"] == 6
    graph = SimpleNamespace(node_area=np.ones(3), plane_scores=np.zeros((3, 3)),
                            supports=np.zeros((3, 3)), sides=np.ones((3, 3)))
    seed_observer = growth._ScalarAttritionObserver()
    with growth._scalar_attrition_context(seed_observer, frame="seed"):
        assert growth._seed_and_grow(graph, 1, 1, growth.DEFAULT_REGION_GROWTH_CONFIG) == []
    assert _closed(seed_observer, "seed")["examples"][0]["eligible_seed_count"] == 0


@pytest.fixture
def proposal_control(monkeypatch):
    image = np.zeros((24, 24, 3), np.uint8)
    usable = np.ones((24, 24), bool)
    labels = np.zeros((24, 24), np.int32)
    labels[:12, 12:] = 1
    labels[12:, :12] = 2
    labels[12:, 12:] = 3
    labels[0, 0] = 4
    calibration = replace(_v1_region_control()[2], width=24, height=24)
    graph = SimpleNamespace(plane_scores=np.tile((0., .8, 0.), (5, 1)),
                            adjacency=((),) * 5)
    monkeypatch.setattr(growth.time, "perf_counter", lambda: 1.)
    monkeypatch.setattr(growth, "_working_image", lambda *_a: (
        image, np.zeros((24, 24)), image.astype(float), usable, 1., 1.))
    monkeypatch.setattr(growth, "_local_lab_xy_atoms", lambda *_a, **_k: (
        labels, {"connectivity_mask_preserved": True}))
    monkeypatch.setattr(growth, "_build_graph", lambda *_a: (
        graph, {"segments_clipped": 1}))
    monkeypatch.setattr(growth, "_seed_and_grow", lambda _g, axis, side, _c:
                        [[], [4], [3], [0], [0], [1], [2]] if (axis, side) == (1, -1) else [])
    polygon = growth._component_polygon
    monkeypatch.setattr(growth, "_component_polygon",
                        lambda mask: () if mask[23, 23] else polygon(mask))
    calls = _track(monkeypatch, ((growth, "_seed_and_grow"), (growth, "_component_polygon"),
                                (growth, "_select_diverse")))
    return image, usable, calibration, calls


def test_generated_proposals_have_one_disposition_and_same_order(proposal_control):
    image, usable, calibration, calls = proposal_control
    normal = growth.propose_image_planes_v2(image, usable, calibration, max_candidates=1)
    normal_calls = list(calls)
    calls.clear()
    observer = growth._ScalarAttritionObserver()
    with growth._scalar_attrition_context(observer, frame="regions"):
        observed = growth.propose_image_planes_v2(image, usable, calibration, max_candidates=1)
    assert normal_calls == calls and normal.stats == observed.stats
    assert normal.status == observed.status and normal.reason == observed.reason
    assert len(normal.proposals) == len(observed.proposals) == 1
    np.testing.assert_array_equal(normal.proposals[0].mask, observed.proposals[0].mask)
    assert normal.proposals[0].polygon == observed.proposals[0].polygon
    row = _closed(observer, "proposal")
    assert row["expected"] == 7
    assert {v["disposition"]: v["count"] for v in row["counts"]} == {
        "mask_empty": 1, "area_rejected": 1, "polygon_rejected": 1,
        "deduplicated": 1, "diversity_cap": 2, "returned": 1}
    assert observer.record()["checks"]["proposal_generation:regions"]
    assert observer.record()["checks"]["proposal_returned:regions"]
    handoffs = observer.record()["proposal_handoffs"]
    assert len(handoffs) == 1 and handoffs[0]["returned_identity"] == ("regions", 0)


def test_missing_both_proposal_hooks_fails_independent_generation_census(proposal_control, monkeypatch):
    image, usable, calibration, calls = proposal_control
    normal = growth.propose_image_planes_v2(image, usable, calibration, max_candidates=1)
    normal_calls = list(calls)
    calls.clear()
    observer = growth._ScalarAttritionObserver()
    expect, finish = observer.expect, observer.finish
    monkeypatch.setattr(observer, "expect", lambda stage, identity:
                        None if stage == "proposal" and identity == ("regions", 0)
                        else expect(stage, identity))
    monkeypatch.setattr(observer, "finish", lambda stage, identity, disposition, **fields:
                        None if stage == "proposal" and identity == ("regions", 0)
                        else finish(stage, identity, disposition, **fields))
    with growth._scalar_attrition_context(observer, frame="regions"):
        observed = growth.propose_image_planes_v2(image, usable, calibration, max_candidates=1)
    assert calls == normal_calls and observed.stats == normal.stats
    assert observed.proposals[0].polygon == normal.proposals[0].polygon
    np.testing.assert_array_equal(observed.proposals[0].mask, normal.proposals[0].mask)
    record = observer.record()
    stage = record["stages"]["proposal"]
    assert stage["expected"] == stage["classified"] == 6
    assert stage["unclassified"] == stage["multiply_classified"] == stage["unexpected"] == 0
    assert record["checks"]["proposal_generation:regions"] is False
    assert record["status"] == "incomplete"


def test_returned_ordinal_substitution_fails_exact_reading_handoff(proposal_control, monkeypatch):
    image, usable, calibration, calls = proposal_control
    source = growth.propose_image_planes_v2(image, usable, calibration, max_candidates=3)
    _controlled_v1_readings(monkeypatch, source.proposals, usable, calibration, frame="handoff")
    monkeypatch.setattr(measure, "create_gui_mask", lambda height, width: np.zeros((height, width), bool))
    frame = SimpleNamespace(frame_id="handoff", image=image)
    calls.clear()
    normal = measure._measure_frame(frame, calibration)
    normal_calls = list(calls)
    calls.clear()
    observer = growth._ScalarAttritionObserver()
    expect, finish = observer.expect, observer.finish
    def substituted(stage, identity):
        return ("handoff", 99) if stage == "reading" and identity == ("handoff", 0) else identity
    monkeypatch.setattr(observer, "expect", lambda stage, identity: expect(stage, substituted(stage, identity)))
    monkeypatch.setattr(observer, "finish", lambda stage, identity, disposition, **fields:
                        finish(stage, substituted(stage, identity), disposition, **fields))
    with growth._scalar_attrition_context(observer):
        observed = measure._measure_frame(frame, calibration)
    assert observed == normal and calls == normal_calls
    record = observer.record()
    stage = record["stages"]["reading"]
    assert stage["expected"] == stage["classified"] == len(source.proposals)
    assert stage["unclassified"] == stage["multiply_classified"] == stage["unexpected"] == 0
    assert record["checks"]["proposal_generation:handoff"] is True
    assert record["checks"]["proposal_returned:handoff"] is True
    assert record["checks"]["proposal_to_reading:handoff"] is False
    assert record["status"] == "incomplete"


def _label_wholly_stubbed_measurement(monkeypatch):
    # These reading-policy controls replace the complete measurement function;
    # label that test-only parent explicitly, without fabricating round data.
    original = planes._bounded_measure
    def stub(*args, **kwargs):
        result = original(*args, **kwargs)
        growth._scalar_period_finish("period_measurement", "synthetic_stub",
                                    measurement_state="not_evaluated")
        return result
    monkeypatch.setattr(planes, "_bounded_measure", stub)


@pytest.mark.parametrize("cause", ("window_incidence", "window_invalid", "periodicity",
                                   "structural_energy", "distance_alternative", "coverage"))
def test_reading_rejections_close_and_preserve_output_and_calls(monkeypatch, cause):
    proposal, usable, calibration, _polygon = _v1_region_control()
    image, _ = _controlled_v1_readings(monkeypatch, (proposal,), usable, calibration, frame="reader")
    if cause == "window_incidence":
        monkeypatch.setattr(planes.lattice, "plane_window", lambda *_a, **_k: None)
    elif cause == "window_invalid":
        monkeypatch.setattr(planes.lattice, "plane_window", lambda *_a, **_k: (0., 0., -1., 1.))
    elif cause == "periodicity":
        monkeypatch.setattr(planes, "_bounded_measure", lambda *_a, **_k: (None,) * 5)
    elif cause == "structural_energy":
        monkeypatch.setattr(planes.lattice, "rectify_plane", lambda *_a, **_k: None)
    elif cause == "distance_alternative":
        monkeypatch.setattr(planes, "_distance_alternatives", lambda *_a, **_k: ())
    else:
        estimate = SimpleNamespace(period_pixels=256., strength=.8)
        patch = np.ones((128, 128), np.float32)
        monkeypatch.setattr(planes, "_bounded_measure", lambda *_a, **_k: (estimate, 128, patch, patch, 0))
    _label_wholly_stubbed_measurement(monkeypatch)
    calls = _track(monkeypatch, ((planes.lattice, "plane_window"), (planes, "_bounded_measure"),
                                (planes.lattice, "rectify_plane"), (planes, "_distance_alternatives")))
    proposals = growth.PlaneProposalResult("ok", "synthetic", (proposal,))
    def run():
        diagnostics = {}
        output = planes.frame_plane_readings_from_proposals(
            image, usable, calibration, proposals, frame="reader", diagnostics=diagnostics)
        return output, diagnostics
    normal = run()
    normal_calls = list(calls)
    calls.clear()
    observer = growth._ScalarAttritionObserver()
    with growth._scalar_attrition_context(observer):
        observed = run()
    assert observed == normal and calls == normal_calls and observed[0] == ()
    assert _closed(observer, "reading")["examples"][0]["disposition"] == cause


@pytest.mark.parametrize("cap", ("reading_cap", "per_axis_side_cap", "proposal_validation"))
def test_reading_caps_and_invalid_proposals_have_exact_closure(monkeypatch, cap):
    base, usable, calibration, _polygon = _v1_region_control()
    count = planes.MAX_READINGS_PER_AXIS_SIDE + 2
    proposals = tuple(replace(base, sources=(str(index),)) for index in range(count))
    image, _ = _controlled_v1_readings(monkeypatch, proposals, usable, calibration, frame="caps")
    if cap == "proposal_validation":
        proposals = (replace(base, mask=np.zeros_like(base.mask)),)
    _label_wholly_stubbed_measurement(monkeypatch)
    source = growth.PlaneProposalResult("ok", "synthetic", proposals)
    maximum = 1 if cap == "reading_cap" else planes.MAX_READINGS
    def run():
        return planes.frame_plane_readings_from_proposals(
            image, usable, calibration, source, frame="caps", max_readings=maximum)
    normal = run()
    observer = growth._ScalarAttritionObserver()
    with growth._scalar_attrition_context(observer):
        observed = run()
    assert observed == normal
    row = _closed(observer, "reading")
    assert row["expected"] == len(proposals)
    assert sum(v["count"] for v in row["counts"] if v["disposition"] == "accepted") == len(observed)
    assert any(v["disposition"] == cap for v in row["counts"])


def _pair(cells=None, share=.6):
    if cells is None:
        cells = np.random.default_rng(4).normal(size=(20, 20)).astype(np.float32)
    return tuple(make_signature(make_reading(frame, 1, -1., [(1., 3., 1.)], share=share), cells)
                 for frame in ("a", "b"))


@pytest.mark.parametrize("cause", ("invalid_grid_homography", "left_sparse", "right_sparse",
                                   "left_constant", "right_constant", "correlation_unavailable",
                                   "insufficient_overlap"))
def test_pair_terminal_subcauses_preserve_outputs_refusals_and_calls(monkeypatch, cause):
    left, right = _pair()
    if cause == "invalid_grid_homography":
        def invalid(*_a):
            raise ValueError("synthetic invalid grid")
        monkeypatch.setattr(plane_graph, "cell_grid_homography", invalid)
    elif cause == "left_sparse":
        left = replace(left, coverage=np.zeros_like(left.coverage))
    elif cause == "right_sparse":
        right = replace(right, coverage=np.zeros_like(right.coverage))
    elif cause == "left_constant":
        left = replace(left, cells=np.ones_like(left.cells))
    elif cause == "right_constant":
        right = replace(right, cells=np.ones_like(right.cells))
    elif cause == "correlation_unavailable":
        monkeypatch.setattr(plane_graph, "_correlation_surface_with_support", lambda *_a, **_k: None)
    else:
        monkeypatch.setattr(plane_graph, "_correlation_surface_with_support", lambda *_a, **_k:
                            plane_graph.CorrelationSurface(plane_graph.PLANE_CORRELATION_ALGORITHM,
                                                           np.ones((3, 3)), np.full((3, 3), 4)))
    expected_left_std = float(np.std(left.cells.astype(np.float64)))
    targets = ((plane_graph, "cell_grid_homography"), (plane_graph, "warp_signature"),
               (plane_graph, "_correlation_surface_with_support"), (plane_graph.np, "std"))
    if cause in ("left_sparse", "right_sparse", "left_constant", "right_constant"):
        targets += ((plane_graph.np, "count_nonzero"),)
    calls = _track(monkeypatch, targets)
    def run():
        refusals, trace = [], plane_graph.PlaneGraphTrace(pairs=[])
        output = plane_graph.match_plane_pair(left, right, quadrant=0, max_shift_cells=1,
                                              refusals=refusals, diagnostic_sink=trace)
        return output, refusals, trace
    normal = run()
    normal_calls = list(calls)
    calls.clear()
    observer = growth._ScalarAttritionObserver()
    with growth._scalar_attrition_context(observer):
        observed = run()
    assert observed == normal and calls == normal_calls and observed[0] is None
    row = _closed(observer, "pair")["examples"][0]
    if cause == "insufficient_overlap":
        assert row["disposition"] == cause
        assert row["maximum_searched_overlap"] == 4
        assert row["maximum_overlap_shift_count"] == 9
    else:
        assert row["disposition"] == "no_signal"
        expected_cause = ("fewer_than_two_covered_cells" if cause.endswith("sparse") else
                          "constant_covered_energy" if cause.endswith("constant") else cause)
        assert row["subcause"] == expected_cause
    if cause in ("left_sparse", "right_sparse", "left_constant", "right_constant"):
        assert row["signal_failure_operand"] == ("left_warped" if cause.startswith("left") else "right")
        measurements = {
            "left_sparse": (0, None, None, None),
            "right_sparse": (400, 0, None, None),
            "left_constant": (400, 400, 0., None),
            "right_constant": (400, 400, expected_left_std, 0.),
        }[cause]
        keys = ("left_warped_covered_cells", "right_covered_cells",
                "left_warped_standard_deviation", "right_standard_deviation")
        for key, value in zip(keys, measurements):
            assert row[key] == value
            assert row[key + "_state"] == ("not_evaluated" if value is None else "measured")
        expected_count_calls, expected_std_calls = {
            "left_sparse": (1, 0), "right_sparse": (2, 0),
            "left_constant": (2, 1), "right_constant": (2, 2),
        }[cause]
        # Grid validation makes two earlier count calls. The exact gate suffix
        # starts after the sole warp and retains the original short circuit.
        gate_calls = calls[calls.index("warp_signature") + 1:]
        assert calls.count("count_nonzero") == expected_count_calls + 2
        assert gate_calls.count("count_nonzero") == expected_count_calls
        assert calls.count("std") == expected_std_calls
        assert calls.count("warp_signature") == 1
        assert calls.count("_correlation_surface_with_support") == 0


def _bind_universe(observer, values):
    _readings, indices, _frames = plane_graph._canonical_reading_index(
        tuple(signature.reading for signature in values))
    for index, signature in enumerate(values):
        identity = plane_graph._attrition_signature_identity(signature, indices[id(signature.reading)][1])
        observer.bind(("reading", id(signature.reading)), identity[:2])
        observer.expect("signature", identity)
        observer.finish("signature", identity, "built", axis=int(signature.reading.axis))
        observer.bind(("signature", id(signature)), identity)


@pytest.mark.parametrize("low_share", (False, True))
def test_graph_pair_product_closes_low_share_and_selected_paths(monkeypatch, low_share):
    values = _pair(share=.01 if low_share else .6)
    readings = tuple(value.reading for value in values)
    calls = _track(monkeypatch, ((plane_graph, "match_plane_pair"),
                                (plane_graph, "_apply_same_axis_filter")))
    def run():
        trace, refusals = plane_graph.PlaneGraphTrace(pairs=[]), []
        output = plane_graph.match_plane_graph(readings, values, refusals=refusals, diagnostic_sink=trace)
        return output, refusals, trace
    normal = run()
    normal_calls = list(calls)
    calls.clear()
    observer = growth._ScalarAttritionObserver()
    _bind_universe(observer, values)
    with growth._scalar_attrition_context(observer):
        observed = run()
    assert observed == normal and calls == normal_calls
    row = _closed(observer, "pair")
    assert row["expected"] == 4
    if low_share:
        assert row["counts"][0]["disposition"] == "both_low_share_pre_trace_skip"
        assert row["counts"][0]["count"] == 4
        assert "match_plane_pair" not in calls
    else:
        assert len(observed[0]) > 0
        assert any(v["disposition"] == "selected_admitted" for v in row["counts"])


@pytest.mark.parametrize("failure", (False, True))
def test_signature_census_preserves_stage_digests_and_refusals(records, monkeypatch, failure):
    session, lattice = records
    if failure:
        def fail(*_a, **_k):
            raise ValueError("synthetic coverage failure")
        monkeypatch.setattr(planes, "plane_signature", fail)
    calls = _track(monkeypatch, ((planes, "plane_signature"),))
    def run():
        return signatures.derive_image_derived_signatures(session=session, lattice_result=lattice)
    normal = run()
    normal_calls = list(calls)
    calls.clear()
    observer = growth._ScalarAttritionObserver()
    with growth._scalar_attrition_context(observer):
        observed = run()
    assert calls == normal_calls
    assert observed.result_sha256 == normal.result_sha256
    assert observed.signatures_sha256 == normal.signatures_sha256
    assert observed.masks_sha256 == normal.masks_sha256
    assert observed.status == normal.status and observed.abstentions == normal.abstentions
    row = _closed(observer, "signature")
    assert row["expected"] == sum(len(r.alternatives) for r in lattice.readings)
    if failure:
        assert row["examples"][0]["disposition"] == "signature_failure"
        assert len(calls) == 1
    else:
        assert all(v["disposition"] == "built" for v in row["counts"])


def test_same_axis_dropped_candidates_are_terminal_without_changing_matches():
    cells = np.random.default_rng(8).normal(size=(20, 20)).astype(np.float32)
    values = tuple(make_signature(make_reading(frame, 1, -1., [(1., distance, 1.)]), cells)
                   for frame, distance in (("a", 2.), ("a", 6.), ("b", 3.), ("b", 8.)))
    readings = tuple(value.reading for value in values)
    normal = plane_graph.match_plane_graph(readings, values)
    observer = growth._ScalarAttritionObserver()
    _bind_universe(observer, values)
    with growth._scalar_attrition_context(observer):
        observed = plane_graph.match_plane_graph(readings, values)
    assert normal == observed
    row = _closed(observer, "pair")
    assert row["expected"] == 16
    assert any(v["disposition"] == "same_axis_dropped" for v in row["counts"])


def test_signature_census_closes_exactly_against_matcher_universe(records):
    session, lattice = records
    observer = growth._ScalarAttritionObserver()
    _index, object_indices, _frames = plane_graph._canonical_reading_index(lattice.readings)
    for reading in lattice.readings:
        observer.bind(("reading", id(reading)), object_indices[id(reading)])
    with growth._scalar_attrition_context(observer):
        result = signatures.derive_image_derived_signatures(session=session, lattice_result=lattice)
        plane_graph.match_plane_graph(lattice.readings, tuple(item.signature for item in result.signatures))
    record = observer.record()
    assert record["status"] == "complete", record
    assert record["checks"]["signature_universe"] is True
    assert record["checks"]["reading_universe"] is True
    runner._validate_attrition_record(record)


def test_complete_synthetic_chain_preserves_stage_digests_and_call_order(monkeypatch):
    calls = _track(monkeypatch, ((measure, "_measure_frame"), (growth, "propose_image_planes_v2"),
                                (planes, "frame_plane_readings_from_proposals"),
                                (planes, "_bounded_measure"), (planes, "plane_signature"),
                                (plane_graph, "match_plane_graph"), (plane_graph, "match_plane_pair")))
    def chain():
        session, _pose, lattice = _result(monkeypatch)
        result = signatures.derive_image_derived_signatures(session=session, lattice_result=lattice)
        matches = plane_graph.match_plane_graph(lattice.readings, tuple(item.signature for item in result.signatures))
        return lattice, result, matches
    normal = chain()
    normal_calls = list(calls)
    calls.clear()
    observer = growth._ScalarAttritionObserver()
    with growth._scalar_attrition_context(observer):
        observed = chain()
    assert calls == normal_calls
    assert observed[0].result_sha256 == normal[0].result_sha256
    assert observed[1].result_sha256 == normal[1].result_sha256
    assert observed[2] == normal[2]
    record = observer.record()
    assert record["status"] == "complete", record
    assert set(record["stages"]) == {"frame", "seed", "proposal", "reading", "signature", "pair",
                                     *growth._PERIOD_SCALAR_STAGES}
    assert all(record["checks"].values())
    runner._validate_attrition_record(record)


@pytest.mark.parametrize("fault", ("missing", "duplicate", "unexpected"))
def test_missing_or_duplicate_terminal_rows_fail_closed_as_evidence(fault):
    observer = growth._ScalarAttritionObserver()
    observer.expect("reading", ("f", 0))
    if fault != "missing":
        observer.finish("reading", ("f", 0), "accepted")
        observer.finish("reading", ("f", 0 if fault == "duplicate" else 1), "accepted")
    record = observer.record()
    assert record["status"] == "incomplete"
    runner._validate_attrition_record(record)
    record["status"] = "complete"
    with pytest.raises(runner.DiagnosticError, match="vertical_attrition_closure"):
        runner._validate_attrition_record(record)


def test_scalar_budget_exhaustion_is_incomplete_and_never_fails_inference():
    observer = growth._ScalarAttritionObserver(max_items=3, max_bytes=1024)
    with growth._scalar_attrition_context(observer):
        output = plane_graph.match_plane_pair(*_pair(), quadrant=0)
    assert output is not None
    record = observer.record()
    assert record["status"] == "incomplete" and record["omitted_accounting_events"] > 0
    assert len(json.dumps(record)) < 4096
    runner._validate_attrition_record(record)


def test_aggregate_examples_are_bounded_with_exact_omission_and_digest():
    observer = growth._ScalarAttritionObserver()
    digest = hashlib.sha256()
    for index in range(200):
        identity = ("f", index)
        observer.expect("reading", identity)
        observer.finish("reading", identity, "accepted", axis=1)
        digest.update(observer._encode(dict(identity=identity, disposition="accepted", axis=1)) + b"\n")
    row = _closed(observer, "reading")
    assert len(row["examples"]) == 128 and row["omitted_examples"] == 72
    assert row["counts"][0]["count"] == 200
    assert row["terminal_sha256"] == digest.hexdigest()


def test_observer_fault_cannot_change_pair_result_or_exception(monkeypatch):
    normal = plane_graph.match_plane_pair(*_pair(), quadrant=0)
    observer = growth._ScalarAttritionObserver()
    monkeypatch.setattr(observer, "finish", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("observer fault")))
    with growth._scalar_attrition_context(observer):
        observed = plane_graph.match_plane_pair(*_pair(), quadrant=0)
    assert observed == normal and observer.record()["status"] == "incomplete"
    def fail(*_a):
        raise RuntimeError("original inference failure")
    monkeypatch.setattr(plane_graph, "warp_signature", fail)
    for enabled in (False, True):
        with growth._scalar_attrition_context(observer if enabled else None):
            with pytest.raises(RuntimeError, match="original inference failure"):
                plane_graph.match_plane_pair(*_pair(), quadrant=0)
