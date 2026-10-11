"""Synthetic source-seam seed predicates; no source images."""
import copy
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
from mc_photogrammetry.reconstruction import region_growth as growth

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluation/diagnose_image_seed_gate.py"
spec = importlib.util.spec_from_file_location("seed_gate_test_subject", SCRIPT)
subject = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = subject
spec.loader.exec_module(subject)


def graph_for_masks(masks, dtype=np.float32):
    masks = np.asarray(masks, dtype=np.uint8)
    n = len(masks)
    best = np.where(masks & 1, .5, .1).astype(dtype)
    other = np.where(masks & 2, best - .1, best + .1).astype(dtype)
    scores = np.zeros((n, 3), dtype=dtype)
    scores[:, 0], scores[:, 1] = best, other
    supports = np.zeros((n, 3), dtype=dtype)
    supports[:, 1] = np.where(masks & 4, 20., 0.)
    sides = np.ones((n, 3), dtype=np.int8)
    sides[:, 0] = np.where(masks & 8, 1, -1)
    return SimpleNamespace(node_area=np.ones(n), plane_scores=scores, supports=supports,
        sides=sides, node_lab=np.zeros((n, 3), dtype=dtype), adjacency=[[] for _ in range(n)])


def observe_graph(graph, observer, axis=0, side=1, config=growth.DEFAULT_REGION_GROWTH_CONFIG):
    if "a" not in observer.active_frames:
        observer.begin_frame("a")
    with growth._scalar_attrition_context(observer, frame="a"):
        result = growth._seed_and_grow(graph, axis, side, config)
    return result, observer.seed_rows[("a", axis, side)]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_real_conjunction_all16_masks_exact_overlap_and_precedence(dtype):
    graph = graph_for_masks(range(16), dtype)
    off = growth._seed_and_grow(graph, 0, 1, growth.DEFAULT_REGION_GROWTH_CONFIG)
    observer = subject.StreamingObserver()
    on, row = observe_graph(graph, observer)
    assert off == on == [[15]]
    assert row["histogram"] == [1] * 16 and row["denominator"] == 16
    assert row["counts"] == dict(passes=[8]*4, cumulative=[8,4,2,1], first_failures=[8,4,2,1], all_pass=1)
    assert row["eligible_seed_count"] == 1 and not observer.incomplete
    assert sum(row["counts"]["first_failures"]) + row["counts"]["all_pass"] == 16


@pytest.mark.parametrize("mask", range(16))
def test_each_individual_and_overlapping_failure(mask):
    _, row = observe_graph(graph_for_masks([mask]), subject.StreamingObserver())
    assert row["histogram"] == [int(i == mask) for i in range(16)]
    assert row["counts"]["all_pass"] == int(mask == 15)
    expected = next((i for i in range(4) if not mask & (1 << i)), None)
    assert row["counts"]["first_failures"] == [int(i == expected) for i in range(4)]


@pytest.mark.parametrize("dtype", [np.float32,np.float64])
@pytest.mark.parametrize("field,threshold", [("score",.26),("margin",.035),("support",12.)])
def test_dtype_sensitive_exact_threshold_boundary(dtype, field, threshold):
    graph = graph_for_masks([15,15,15], dtype)
    boundary = dtype(threshold)
    values = np.array([np.nextafter(boundary, dtype(-np.inf)), boundary,
                       np.nextafter(boundary, dtype(np.inf))], dtype=dtype)
    config = growth.DEFAULT_REGION_GROWTH_CONFIG
    if field == "score":
        graph.plane_scores[:,0] = values
        graph.plane_scores[:,1] = 0.
    elif field == "margin":
        graph.plane_scores[:,0] = values
        graph.plane_scores[:,1] = 0.
        config = replace(config,min_seed_score=0.)
    else:
        graph.supports[:,1] = values
    observer = subject.StreamingObserver()
    _,row = observe_graph(graph,observer,config=config)
    bit = {"score":0,"margin":1,"support":2}[field]
    expected = (values >= getattr(config,"min_seed_"+field)).tolist()
    assert row["counts"]["passes"][bit] == sum(expected)
    assert row["histogram"][15] == sum(expected)
    assert not observer.incomplete


def test_observed_empty_is_not_unreached():
    observer = subject.StreamingObserver()
    result,row = observe_graph(graph_for_masks([]),observer)
    assert result == [] and row["state"] == "reached" and row["denominator"] == 0
    assert row["histogram"] == [0]*16 and row["eligible_seed_count"] == 0
    assert row["reason"] == ""


def test_explicit_upstream_not_reached_preserves_reason():
    observer = subject.StreamingObserver();observer.begin_frame("a")
    observer.expect("frame",("a","proposal"));observer.finish("frame",("a","proposal"),"missing_calibration")
    for axis in range(3):
        for side in (-1,1):
            identity=("a",axis,side)
            observer.expect("seed",identity)
            observer.finish("seed",identity,"not_reached",axis=axis,side=side,
                eligible_seed_count=None,reason="missing_calibration")
    observer.end_frame("a")
    record=observer.seed_record()
    assert all(row["state"]=="not_reached" and row["denominator"] is None and row["histogram"] is None for row in record["rows"])
    assert observer.record()["status"]=="complete"
    subject.validate_seed_record(record,subject._observer_record(observer),{"a"})


def test_source_six_identities_join_and_no_array_retained():
    graph=graph_for_masks(range(16));references=[weakref.ref(v) for v in vars(graph).values() if type(v) is np.ndarray]
    observer=subject.StreamingObserver();observer.begin_frame("a")
    with growth._scalar_attrition_context(observer,frame="a"):
        for axis in range(3):
            for side in (-1,1):growth._seed_and_grow(graph,axis,side,growth.DEFAULT_REGION_GROWTH_CONFIG)
    observer.end_frame("a")
    assert observer.record()["status"]=="complete"
    subject.validate_seed_record(observer.seed_record(),subject._observer_record(observer),{"a"})
    del graph
    assert all(ref() is None for ref in references)
    assert observer.atom_visits==96 and observer.predicate_visits==384


@pytest.mark.parametrize("failure",["throw","missing","duplicate","late"])
def test_sink_failure_missing_duplicate_late_is_sticky_and_neutral(failure):
    graph=graph_for_masks(range(16));off=growth._seed_and_grow(graph,0,1,growth.DEFAULT_REGION_GROWTH_CONFIG)
    observer=subject.StreamingObserver();observer.begin_frame("a")
    original=observer.seed_predicates
    if failure=="throw":
        def throwing(*a,**k):raise RuntimeError("diagnostic only")
        observer.seed_predicates=throwing
    elif failure=="missing":observer.seed_predicates=lambda *a,**k:None
    on,_=observe_graph(graph,observer)
    if failure=="duplicate":observe_graph(graph,observer)
    elif failure=="late":
        # A closed or unregistered identity must never reopen a seed record.
        observer.active_frames.remove("a")
        original(("a",0,1),count=16,best=graph.plane_scores[:,0],other=graph.plane_scores[:,1],
            seed_support=graph.supports[:,1],sides=graph.sides[:,0],seed=np.zeros(16,dtype=bool),config=growth.DEFAULT_REGION_GROWTH_CONFIG)
    elif failure=="missing":observer.end_frame("a")
    assert off==on==[[15]] and observer.incomplete
    before=observer.atom_visits
    observer.begin_frame("next")
    assert observer.atom_visits==before and observer.incomplete


@pytest.mark.parametrize("limit",["atoms_per_record","atom_visits","predicate_visits","counter_max"])
def test_seed_work_and_counter_exact_then_one_over(limit):
    values={"atoms_per_record":2,"atom_visits":2,"predicate_visits":8,"counter_max":2}
    observer=subject.StreamingObserver(seed_limits={limit:values[limit]})
    _,row=observe_graph(graph_for_masks([0,0]),observer)
    assert row["state"]=="reached" and not observer.incomplete
    other=subject.StreamingObserver(seed_limits={limit:values[limit]})
    observe_graph(graph_for_masks([0,0,0]),other)
    assert other.incomplete
    assert other.atom_visits <= (3 if limit=="counter_max" else values[limit])


@pytest.mark.parametrize("limit,value",[("frames",1),("records",6),("aggregate_bytes",6*(1024+3))])
def test_seed_scope_reservation_exact_and_one_over(limit,value):
    observer=subject.StreamingObserver(seed_limits={limit:value});observer.begin_frame("a")
    assert len(observer.seed_rows)==6 and not observer.incomplete
    observer.begin_frame("b")
    assert observer.incomplete and len(observer.seed_rows)==6


def test_invalid_and_nonfinite_config_refuses_diagnostic_without_gate_change():
    graph=graph_for_masks([0])
    for value in (float("nan"),float("inf")):
        config=replace(growth.DEFAULT_REGION_GROWTH_CONFIG,min_seed_score=value)
        off=growth._seed_and_grow(graph,0,1,config)
        observer=subject.StreamingObserver()
        on,_=observe_graph(graph,observer,config=config)
        assert on==off==[] and observer.incomplete


def test_native_source_exception_identity_survives_observer():
    graph=graph_for_masks([15]);graph.node_lab=None
    with pytest.raises(TypeError):growth._seed_and_grow(graph,0,1,growth.DEFAULT_REGION_GROWTH_CONFIG)
    observer=subject.StreamingObserver();observer.begin_frame("a")
    with growth._scalar_attrition_context(observer,frame="a"):
        with pytest.raises(TypeError):growth._seed_and_grow(graph,0,1,growth.DEFAULT_REGION_GROWTH_CONFIG)
    assert observer.seed_rows[("a",0,1)]["state"]=="reached"


def test_aggregate_work_precharge_does_not_enter_bin_kernel(monkeypatch):
    observer=subject.StreamingObserver(seed_limits={"atoms_per_record":2})
    graph=graph_for_masks([0,0,0])
    monkeypatch.setattr(np,"bincount",lambda *a,**k:pytest.fail("diagnostic allocation before precharge"))
    result,_=observe_graph(graph,observer)
    assert result==[] and observer.incomplete and observer.atom_visits==0


def test_checked_cumulative_work_exhaustion_across_identities(monkeypatch):
    observer=subject.StreamingObserver(seed_limits={"atom_visits":2})
    graph=graph_for_masks([0,0])
    observe_graph(graph,observer)
    assert observer.atom_visits==2 and not observer.incomplete
    monkeypatch.setattr(np,"bincount",lambda *a,**k:pytest.fail("extra diagnostic work"))
    observe_graph(graph,observer,side=-1)
    assert observer.atom_visits==2 and observer.incomplete


def test_scope_reopen_and_missing_transition_refuse():
    observer=subject.StreamingObserver();observer.begin_frame("a")
    observer.end_frame("a")
    assert observer.incomplete
    count=len(observer.seed_rows)
    observer.begin_frame("a")
    assert observer.incomplete and len(observer.seed_rows)==count


def test_frame_scalar_failure_isolated_before_original_measurement(monkeypatch):
    graph=graph_for_masks([15])
    off=growth._seed_and_grow(graph,0,1,growth.DEFAULT_REGION_GROWTH_CONFIG)
    observer=subject.StreamingObserver()
    def fail(*a,**k):raise RuntimeError("diagnostic encoding")
    monkeypatch.setattr(observer,"_encode",fail)
    observer.begin_frame("a")
    with growth._scalar_attrition_context(observer,frame="a"):
        on=growth._seed_and_grow(graph,0,1,growth.DEFAULT_REGION_GROWTH_CONFIG)
    assert on==off==[[0]] and observer.incomplete


def test_fixed16_bin_registry_and_immutable_limits():
    with pytest.raises(subject.AttritionError):subject.StreamingObserver(seed_limits={"histogram_bins":15})
    observer=subject.StreamingObserver()
    with pytest.raises(TypeError):observer.seed_limits["histogram_bins"]=17
    assert observer.seed_limits["histogram_bins"]==16


def test_genuine_zero_side_boundary_mass_fails_both_without_complement_assumption():
    graph=graph_for_masks(range(16));graph.sides[:,0]=0
    observer=subject.StreamingObserver();observer.begin_frame("a")
    with growth._scalar_attrition_context(observer,frame="a"):
        for axis in range(3):
            for side in (-1,1):growth._seed_and_grow(graph,axis,side,growth.DEFAULT_REGION_GROWTH_CONFIG)
    observer.end_frame("a")
    record=observer.seed_record()
    subject.validate_seed_record(record,subject._observer_record(observer),{"a"})
    plus=observer.seed_rows[("a",0,1)]["histogram"]
    minus=observer.seed_rows[("a",0,-1)]["histogram"]
    assert plus==minus==[2]*8+[0]*8
    assert sum(plus[b]-minus[b+8] for b in range(8))==16
    assert observer.seed_rows[("a",0,1)]["eligible_seed_count"]==0
    assert observer.seed_rows[("a",0,-1)]["eligible_seed_count"]==0


def test_all_six_empty_but_reached_source_identities_remain_valid():
    graph=graph_for_masks([])
    observer=subject.StreamingObserver();observer.begin_frame("a")
    with growth._scalar_attrition_context(observer,frame="a"):
        for axis in range(3):
            for side in (-1,1):
                assert growth._seed_and_grow(graph,axis,side,growth.DEFAULT_REGION_GROWTH_CONFIG)==[]
    observer.end_frame("a")
    record=observer.seed_record()
    assert all(row["state"]=="reached" and row["denominator"]==0 and row["histogram"]==[0]*16 for row in record["rows"])
    subject.validate_seed_record(record,subject._observer_record(observer),{"a"})


def test_all_unreached_seed_identities_require_one_frame_terminal_reason():
    observer=subject.StreamingObserver();observer.begin_frame("a")
    for reason in ("missing_calibration","no_atoms"):
        identity=("a",reason)
        observer.expect("frame",identity);observer.finish("frame",identity,reason)
    for axis in range(3):
        for side in (-1,1):
            identity=("a",axis,side)
            observer.expect("seed",identity)
            observer.finish("seed",identity,"not_reached",axis=axis,side=side,
                eligible_seed_count=None,reason="no_atoms" if axis==0 else "missing_calibration")
    observer.end_frame("a")
    with pytest.raises(subject.AttritionError,match="shared_frame_terminal_reason"):
        subject.validate_seed_record(observer.seed_record(),subject._observer_record(observer),{"a"})
