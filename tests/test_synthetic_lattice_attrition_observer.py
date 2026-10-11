"""Authored scalar lifecycle controls; no images, project reads or production."""
from pathlib import Path
from types import SimpleNamespace as NS
import importlib.util
import json

import pytest
from mc_photogrammetry.reconstruction import region_growth as rg

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("synthetic_lattice_observer", ROOT /
    "scripts/evaluation/probe_synthetic_image_production_corner.py")
helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helper)

ACTIVE = ("image-0000", "image-0001", "image-0002")


def terminal(observer, stage, identity, disposition, **fields):
    observer.expect(stage, identity)
    observer.finish(stage, identity, disposition, **fields)


def seed(observer, frame, axis, side, eligible=0, reached=True):
    fields = dict(axis=axis, side=side)
    if reached:
        fields.update(node_count=5, side_match_count=3, normal_score_pass_count=4,
            margin_pass_count=3, tangent_support_pass_count=2, after_side_count=3,
            after_score_count=2, after_margin_count=1, after_support_count=eligible,
            eligible_seed_count=eligible, max_side_normal_score=.8,
            max_side_margin=.2, max_side_tangent_support=20.)
    else:
        fields["eligible_seed_count"] = None
    terminal(observer, "seed", (frame, axis, side),
             "not_reached" if not reached else "available" if eligible else "absent", **fields)


def authored(reader_code="accepted", reader_side=-1, *, proposal_code="returned", seed_eligible=1):
    observer = helper._new_lattice_observer()
    observer.lattice_calls = 1
    readings = []
    for frame in ACTIVE:
        for axis in range(3):
            for side in (-1, 1):
                seed(observer, frame, axis, side, int(axis == 0 and side == -1) * seed_eligible)
        generated = (frame, 7)
        terminal(observer, "proposal", generated, proposal_code, axis=0, side=-1)
        if proposal_code != "returned":
            terminal(observer, "frame", (frame, "proposal"), "no_grown_plane_regions", returned_count=0)
            for prefix in ("proposal_generation", "proposal_returned"):
                observer.check(prefix + ":" + frame, True)
            continue
        observer.bind(("proposal_return", *generated), (frame, 0))
        terminal(observer, "frame", (frame, "proposal"), "image_edge_graph_growth", returned_count=1)
        reader_fields = dict(axis=0, side=reader_side)
        if reader_code == "accepted":
            reading = NS(frame=frame, axis=0, side=float(reader_side))
            readings.append(reading)
            reader_fields["plane_index"] = 0
            observer.bind(("reading", id(reading)), (frame, 0))
        terminal(observer, "reading", (frame, 0), reader_code, **reader_fields)
        for prefix in ("proposal_generation", "proposal_returned", "proposal_to_reading"):
            observer.check(prefix + ":" + frame, True)
    lattice = NS(frame_ids=ACTIVE, readings=tuple(readings), result_sha256="a" * 64)
    return observer, lattice


def project(observer, lattice):
    return helper._lattice_pass_record(lattice, observer, ACTIVE)


def test_complete_frame_normal_side_joins_first_failure_denominator_and_digest():
    observer, lattice = authored()
    result = project(observer, lattice)
    assert result["status"] == "complete" and not result["incomplete"]
    assert result["totals"] == dict(seed_expected=18, seed_observed=18,
        proposal_expected=3, proposal_observed=3, proposal_returned=3, handoffs=3,
        reader_expected=3, reader_observed=3, accepted_readings=3)
    assert [r["normal_counts"] for r in result["frames"]] == [[1, 0, 0]] * 3
    assert len(result["groups"]) == 18
    for cell in result["groups"]:
        assert cell["seed"]["node_count"] == 5
        expected = {"side": 2, "normal_score": 1, "margin": 1,
                    "tangent_support": 0 if cell["normal_axis"] == 0 and cell["side"] == -1 else 1}
        assert cell["seed"]["first_failure_counts"] == expected
        assert sum(expected.values()) + cell["seed"]["eligible_seed_count"] == 5
        assert cell["proposal_returned"] == cell["handoffs"] == cell["reader_expected"] == cell["reader_observed"]
    digest = result.pop("projection_sha256")
    assert digest == helper.canonical_digest(result)
    result["groups"][0]["seed"]["eligible_seed_count"] += 1
    assert helper.canonical_digest(result) != digest


@pytest.mark.parametrize("code", ["window_incidence", "window_invalid", "periodicity", "structural_energy",
    "distance_alternative", "coverage", "per_axis_side_cap", "reading_cap", "proposal_validation", "frame_rejection"])
def test_original_reader_rejections_are_counted_without_fabricated_readings(code):
    observer, lattice = authored(code, reader_side=None)
    result = project(observer, lattice)
    assert result["status"] == "complete"
    assert result["totals"]["accepted_readings"] == 0
    cells = [r for r in result["groups"] if r["proposal_returned"]]
    assert len(cells) == 3 and all(r["reader_dispositions"] == {code: 1} for r in cells)
    assert all(r["reader_side_transitions"] == {"unknown": 1} for r in cells)


def test_recomputed_reader_side_is_joined_separately_not_assumed_equal():
    observer, lattice = authored(reader_side=1)
    result = project(observer, lattice)
    assert result["status"] == "complete"
    for frame in ACTIVE:
        cells = {r["side"]: r for r in result["groups"] if r["frame_id"] == frame and r["normal_axis"] == 0}
        assert cells[-1]["reader_dispositions"] == {"accepted": 1}
        assert cells[-1]["reader_side_transitions"] == {"1": 1}
        assert cells[-1]["accepted_readings_at_side"] == 0
        assert cells[1]["accepted_readings_at_side"] == 1


def test_no_proposals_and_not_reached_seed_are_unknown_not_invented_zero_predicates():
    observer = helper._new_lattice_observer(); observer.lattice_calls = 1
    for frame in ACTIVE:
        for axis in range(3):
            for side in (-1, 1): seed(observer, frame, axis, side, reached=False)
        terminal(observer, "frame", (frame, "proposal"), "insufficient_calibration_inliers", returned_count=0)
    lattice = NS(frame_ids=ACTIVE, readings=(), result_sha256="a" * 64)
    result = project(observer, lattice)
    assert result["status"] == "complete" and result["totals"]["handoffs"] == 0
    assert all(r["seed"]["first_failure_counts"] is None and r["seed"]["eligible_seed_count"] is None
               for r in result["groups"])
    assert helper._lattice_pass_record(None, None, ()) == {"status": "not_reached", "incomplete": False, "groups": None}


@pytest.mark.parametrize("mutation", ["missing_seed", "duplicate_seed", "foreign_frame", "bad_counter", "bad_predicate",
    "missing_handoff", "foreign_handoff", "duplicate_handoff", "missing_reader", "foreign_axis", "foreign_side",
    "missing_live_binding", "changed_live_binding", "missing_check", "failed_check", "foreign_check", "frame_count",
    "extra_call", "observer_omission", "observer_truncation", "scalar_cap", "foreign_disposition", "reading_cap"])
def test_missing_contradictory_or_capped_accounting_is_incomplete(mutation):
    observer, lattice = authored()
    rows = observer.scalar_rows
    if mutation == "missing_seed": rows["seed"].pop((ACTIVE[0], 0, -1))
    elif mutation == "duplicate_seed": observer.finish("seed", (ACTIVE[0], 0, -1), "available", axis=0, side=-1)
    elif mutation == "foreign_frame": lattice.frame_ids = ("SECRET/path",) + ACTIVE[1:]
    elif mutation == "bad_counter": rows["seed"][ACTIVE[0], 0, -1]["node_count"] = True
    elif mutation == "bad_predicate": rows["seed"][ACTIVE[0], 0, -1]["after_score_count"] = 4
    elif mutation == "missing_handoff": observer.bindings.pop(("proposal_return", ACTIVE[0], 7))
    elif mutation == "foreign_handoff": observer.bindings[("proposal_return", ACTIVE[0], 7)] = ("SECRET/path", 0)
    elif mutation == "duplicate_handoff": observer.bindings[("proposal_return", ACTIVE[0], 8)] = (ACTIVE[0], 0)
    elif mutation == "missing_reader": rows["reading"].pop((ACTIVE[0], 0))
    elif mutation == "foreign_axis": rows["reading"][ACTIVE[0], 0]["axis"] = 1
    elif mutation == "foreign_side": rows["reading"][ACTIVE[0], 0]["side"] = 0
    elif mutation == "missing_live_binding": observer.bindings.pop(("reading", id(lattice.readings[0])))
    elif mutation == "changed_live_binding": observer.bindings[("reading", id(lattice.readings[0]))] = (ACTIVE[0], 8)
    elif mutation == "missing_check": observer.checks.pop("proposal_to_reading:" + ACTIVE[0])
    elif mutation == "failed_check": observer.checks["proposal_to_reading:" + ACTIVE[0]] = False
    elif mutation == "foreign_check": observer.checks["SECRET/path"] = True
    elif mutation == "frame_count": rows["frame"][ACTIVE[0], "proposal"]["returned_count"] = 2
    elif mutation == "extra_call": observer.lattice_calls = 2
    elif mutation == "observer_omission": observer.omitted = 1
    elif mutation == "observer_truncation": observer.incomplete = True
    elif mutation == "scalar_cap": observer.scalar_incomplete = True
    elif mutation == "foreign_disposition": observer.finish("reading", (ACTIVE[0], 9), "SECRET/path")
    elif mutation == "reading_cap": lattice.readings = lattice.readings * 7
    result = project(observer, lattice)
    assert result["status"] == "incomplete" and result["incomplete"] and result["groups"] is None
    assert "SECRET" not in json.dumps(result)


def test_raw_observer_example_truncation_does_not_lose_exact_scalar_join_counts():
    observer, lattice = authored()
    for stage in observer.stages.values():
        stage["examples"] = []
        stage["omitted_examples"] = 999
    result = project(observer, lattice)
    assert result["status"] == "complete" and result["totals"]["reader_observed"] == 3
    assert "examples" not in json.dumps(result) and "proposal_handoffs" not in result


def test_scalar_capture_cap_and_nonfinite_fail_closed_without_propagation():
    observer = helper._new_lattice_observer()
    observer.scalar_rows["proposal"] = {(ACTIVE[0], i): {} for i in range(4096)}
    observer.expect("proposal", (ACTIVE[0], 5000))
    assert observer.finish("proposal", (ACTIVE[0], 5000), "returned", axis=0, side=-1) is None
    assert observer.scalar_incomplete
    observer = helper._new_lattice_observer()
    seed(observer, ACTIVE[0], 0, -1)
    observer.finish("reading", (ACTIVE[0], 1), "coverage", axis=0, side=float("nan"))
    assert observer.scalar_incomplete


def test_original_lattice_call_forwarding_identity_context_and_restoration():
    sentinel, arg, kwarg = object(), object(), object()
    calls = []
    def original(*args, **kwargs):
        calls.append((args, kwargs, rg._attrition_state.get()))
        return sentinel
    module = NS(measure_image_derived_lattice=original)
    before = rg._attrition_state.get()
    with helper.observe_lattice_pass(module) as observer:
        assert module.measure_image_derived_lattice(arg, opaque=kwarg) is sentinel
        assert calls == [((arg,), {"opaque": kwarg}, observer)]
        assert observer.lattice_calls == 1
    assert module.measure_image_derived_lattice is original and rg._attrition_state.get() is before


def test_original_exception_identity_restoration_and_unexpected_replacement():
    error = RuntimeError("SECRET/path")
    def original(*args, **kwargs): raise error
    module = NS(measure_image_derived_lattice=original)
    before = rg._attrition_state.get()
    with pytest.raises(RuntimeError) as caught:
        with helper.observe_lattice_pass(module) as observer:
            module.measure_image_derived_lattice(object())
    assert caught.value is error and observer.lattice_calls == 1
    assert module.measure_image_derived_lattice is original and rg._attrition_state.get() is before
    with helper.observe_lattice_pass(module) as observer:
        module.measure_image_derived_lattice = lambda: None
    assert observer.scalar_incomplete and module.measure_image_derived_lattice is original


@pytest.mark.parametrize("code", ["mask_empty", "area_rejected", "polygon_rejected", "deduplicated", "diversity_cap"])
def test_eligible_seed_without_returned_proposals_identifies_growth_attrition(code):
    observer, lattice = authored(proposal_code=code)
    result = project(observer, lattice)
    assert result["status"] == "complete" and result["totals"]["proposal_returned"] == 0
    cells = [r for r in result["groups"] if r["proposals_expected"]]
    assert all(r["seed"]["eligible_seed_count"] == 1 and r["proposal_dispositions"] == {code: 1}
               and r["reader_expected"] == 0 for r in cells)


def test_missing_seed_support_is_distinct_from_growth_and_reader_losses():
    observer = helper._new_lattice_observer(); observer.lattice_calls = 1
    for frame in ACTIVE:
        for axis in range(3):
            for side in (-1, 1): seed(observer, frame, axis, side)
        terminal(observer, "frame", (frame, "proposal"), "insufficient_plane_seeds", returned_count=0)
    result = project(observer, NS(frame_ids=ACTIVE, readings=(), result_sha256="a" * 64))
    assert result["status"] == "complete"
    assert all(r["seed"]["eligible_seed_count"] == 0 and r["proposals_expected"] == 0 for r in result["groups"])


def test_summary_keeps_typed_refusal_and_digest_binds_new_scalar_projection(monkeypatch):
    observer, lattice = authored("periodicity", reader_side=None)
    base = {"stage": "geometry", "status": "geometry_unavailable", "reason": "g_unavailable",
        "detail": "no_eligible_assignment", "block_count": 0, "active_frame_ids": list(ACTIVE),
        "endpoint_selection": {"incomplete": False}, "pairs": []}
    monkeypatch.setattr(helper, "summarize", lambda *args: dict(base))
    result = helper.summarize_safely(NS(lattice=lattice), NS(), lattice_observer=observer)
    assert result["reason"] == "g_unavailable" and result["block_count"] == 0
    assert result["diagnostic_status"] == "complete"
    digest = helper.canonical_digest(result)
    result["lattice_attrition"]["totals"]["reader_observed"] += 1
    assert helper.canonical_digest(result) != digest
    observer.scalar_incomplete = True
    result = helper.summarize_safely(NS(lattice=lattice), NS(), lattice_observer=observer)
    assert result["diagnostic_status"] == "incomplete" and result["reason"] == "g_unavailable"
    assert helper._lattice_pass_record(lattice, None, ()) == {
        "status": "observer_not_enabled", "incomplete": False, "groups": None}


def test_every_frame_normal_and_side_has_exact_positive_original_handoffs():
    observer = helper._new_lattice_observer(); observer.lattice_calls = 1
    readings = []
    for frame in ACTIVE:
        for index, (axis, side) in enumerate((a, s) for a in range(3) for s in (-1, 1)):
            seed(observer, frame, axis, side, eligible=1)
            terminal(observer, "proposal", (frame, index), "returned", axis=axis, side=side)
            observer.bind(("proposal_return", frame, index), (frame, index))
            reading = NS(frame=frame, axis=axis, side=float(side)); readings.append(reading)
            observer.bind(("reading", id(reading)), (frame, index))
            terminal(observer, "reading", (frame, index), "accepted", axis=axis, side=float(side), plane_index=index)
        terminal(observer, "frame", (frame, "proposal"), "image_edge_graph_growth", returned_count=6)
        for prefix in ("proposal_generation", "proposal_returned", "proposal_to_reading"):
            observer.check(prefix + ":" + frame, True)
    result = project(observer, NS(frame_ids=ACTIVE, readings=tuple(readings), result_sha256="a" * 64))
    assert result["status"] == "complete" and result["totals"]["accepted_readings"] == 18
    assert all(r["normal_counts"] == [2, 2, 2] for r in result["frames"])
    assert all(r["proposal_returned"] == r["handoffs"] == r["reader_expected"] == r["reader_observed"]
               == r["accepted_readings_at_side"] == 1 for r in result["groups"])
