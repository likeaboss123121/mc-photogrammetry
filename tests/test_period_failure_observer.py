"""Scalar same-pass period causes; generated arrays only, no station inference."""
import json

import numpy as np
import pytest

from mc_photogrammetry.reconstruction import lattice, lattice_planes as planes, region_growth as growth
from scripts.evaluation import diagnose_image_geometry_failure as runner
from test_lattice import _striped


def _observe(call, stage, identity=("frame-3", 2, 0)):
    observer=growth._ScalarAttritionObserver()
    with growth._scalar_attrition_context(observer, frame=identity[0]):
        with growth._scalar_period_context(identity=identity, axis=1):
            growth._scalar_period_expect(stage)
            result=call()
    record=observer.record()
    runner._validate_attrition_record(record)
    return result, record


def _details(record, stage):
    return [row["details"] for row in record["stages"][stage]["counts"]]


def _measure(patch, max_side=64):
    return planes._bounded_measure(patch, 1, np.eye(3), 64., mask=np.ones(patch.shape),
        centre=(0.,0.),principal_point=(32.,32.),extent=1.,side=1.,max_rectified_side=max_side)


@pytest.mark.parametrize("branch", ("image", "mask", "zero"))
def test_measure_rectification_short_circuits_and_scalar_states(monkeypatch, branch):
    calls=[]
    patch=_striped(64,8.)
    def rectify(image,*args,**kwargs):
        calls.append(kwargs["samples"])
        if len(calls)==1: return None if branch=="image" else patch
        return None if branch=="mask" else np.zeros(patch.shape)
    monkeypatch.setattr(lattice,"rectify_plane",rectify)
    normal=_measure(patch); baseline=list(calls); calls.clear()
    observed,record=_observe(lambda:_measure(patch),"period_measurement",identity=("frame-3",2))
    assert observed==normal and calls==baseline
    assert record["status"]=="complete"
    row=_details(record,"period_round")[0]
    assert row["identity"]==("frame-3",2,0)
    assert row["disposition"]==dict(image="image_rectification_failed",mask="mask_rectification_failed",zero="zero_mask_support")[branch]
    assert row["mask_mass_state"]==("measured" if branch=="zero" else "not_evaluated")
    assert row["selected_period_state"]=="not_evaluated"
    assert "period_estimate" not in record["stages"]


@pytest.mark.parametrize("cause", ("short_profile","flat_profile","empty_lag_interval","no_eligible_peak","nonpositive_peaks","nonpositive_zero_lag"))
def test_period_terminal_causes_reuse_variance_and_search_values(monkeypatch,cause):
    patch=np.random.default_rng(15).normal(size=(16,16)).astype(np.float32)
    kwargs={}
    if cause=="short_profile": patch=np.ones((3,4),np.float32)
    elif cause=="flat_profile": patch=np.ones((16,16),np.float32)
    elif cause=="empty_lag_interval": kwargs["minimum_period"]=20.
    if cause in ("no_eligible_peak","nonpositive_peaks","nonpositive_zero_lag"):
        corr=np.linspace(-1.,-.2,15)
        corr[0]=0. if cause=="nonpositive_zero_lag" else 1.
        if cause=="nonpositive_peaks": corr[3:6]=(-.4,-.1,-.4)
        monkeypatch.setattr(lattice.np,"correlate",lambda *a,**k:np.concatenate((np.zeros(14),corr)))
    result,record=_observe(lambda:lattice.estimate_period(patch,**kwargs),"period_estimate",identity=("frame-4",1,0,0))
    assert result is None and record["status"]=="complete"
    row=_details(record,"period_estimate")[0]
    assert row["disposition"]==cause
    assert row["zero_lag_state"]==("not_evaluated" if cause in ("short_profile","flat_profile") else "measured")
    assert row["variance_state"]==("not_evaluated" if cause=="short_profile" else "measured")
    assert row["tallest_state"]==("measured" if cause=="nonpositive_peaks" else "not_evaluated")
    if cause=="nonpositive_peaks": assert row["tallest"]==-.1
    if cause=="no_eligible_peak": assert row["peaks_inside"]==0
    if cause=="empty_lag_interval": assert (row["low"],row["high"])==(20,5)
    profile=_details(record,"period_profile")[0]
    assert profile["variance"]==row["variance"]
    assert profile["variance_state"]==row["variance_state"]


def test_single_direction_and_harmonic_reconciliation_are_exact(monkeypatch):
    patch=_striped(128,8.)
    normal=lattice._plane_period_with_direction(patch)
    result,record=_observe(lambda:lattice._plane_period_with_direction(patch),"period_reconciliation")
    assert result==normal and record["status"]=="complete"
    row=_details(record,"period_reconciliation")[0]
    assert row["disposition"]=="single_direction" and row["attempted_directions"]==(0,1)
    assert row["selected_direction"]==0
    yy,xx=np.indices((128,128))
    patch=((xx%8<1).astype(float)+.7*(yy%16<1)).astype(np.float32)
    result,record=_observe(lambda:lattice._plane_period_with_direction(patch),"period_reconciliation")
    assert result is not None and record["status"]=="complete"
    assert _details(record,"period_reconciliation")[0]["disposition"]=="harmonic_reconciliation"
    assert result[0].period_pixels==pytest.approx(8.,abs=.1)


def test_real_measure_resampling_cap_counts_and_input_identity(monkeypatch):
    patch=_striped(64,8.)
    calls=[]
    def rectify(image,*a,**k):
        calls.append((id(image),k["samples"]))
        return patch if image is patch else np.ones(patch.shape)
    monkeypatch.setattr(lattice,"rectify_plane",rectify)
    source=patch.copy()
    off=_measure(patch); off_calls=list(calls); calls.clear()
    on,record=_observe(lambda:_measure(patch),"period_measurement",identity=("frame-4",3))
    assert on[0]==off[0] and on[1]==off[1] and on[4]==off[4]
    # Mask objects are independently created by this test helper; image identity
    # and ordered numerical requests must remain exact.
    assert [c[1] for c in calls]==[c[1] for c in off_calls]==[64,64]
    assert calls[0][0]==off_calls[0][0]==id(patch)
    np.testing.assert_array_equal(patch,source)
    assert record["status"]=="complete"
    assert _details(record,"period_round")[0]["disposition"]=="resampling_cap"
    assert _details(record,"period_measurement")[0]["attempted_rounds"]==1
    assert _details(record,"period_round")[0]["mask_mass"]==4096.


def test_observer_kernel_counts_outputs_and_profile_masks_are_identical(monkeypatch):
    patch=_striped(128,8.)
    weights=np.ones(patch.shape); weights[:,:5]=0.
    originals={name:getattr(np,name) for name in ("diff","divide","correlate","flatnonzero","exp","angle")}
    calls=[]
    for name,original in originals.items():
        def wrapper(*a,_name=name,_original=original,**k):
            calls.append(_name); return _original(*a,**k)
        monkeypatch.setattr(np,name,wrapper)
    before=(patch.tobytes(),weights.tobytes())
    off=lattice._plane_period_with_direction(patch,weights=weights); off_calls=list(calls); calls.clear()
    on,record=_observe(lambda:lattice._plane_period_with_direction(patch,weights=weights),"period_reconciliation")
    assert on==off and calls==off_calls and record["status"]=="complete"
    assert (patch.tobytes(),weights.tobytes())==before
    profiles=_details(record,"period_profile")
    assert sorted(row["positive_weight_bins"] for row in profiles)==[122,127]


def test_true_missing_profile_hook_and_duplicate_terminal_fail_closure(monkeypatch):
    patch=_striped(128,8.)
    original=lattice._profile
    monkeypatch.setattr(lattice,"_profile",lambda *a,**k:np.arange(127,dtype=float))
    result,record=_observe(lambda:lattice.estimate_period(patch),"period_estimate",identity=("frame-3",0,0,0))
    assert record["status"]=="incomplete"
    assert record["stages"]["period_profile"]["unclassified"]==1
    monkeypatch.setattr(lattice,"_profile",original)
    observer=growth._ScalarAttritionObserver()
    with growth._scalar_attrition_context(observer):
        with growth._scalar_period_context(identity=("frame-3",0,0,0),axis=1):
            growth._scalar_period_expect("period_estimate")
            lattice.estimate_period(patch)
            growth._scalar_period_finish("period_estimate","duplicated")
    assert observer.record()["status"]=="incomplete"
    assert observer.record()["stages"]["period_estimate"]["multiply_classified"]==1


def test_exceptions_and_exhaustion_preserve_behavior_and_atomic_incomplete(tmp_path,monkeypatch):
    patch=_striped(128,8.)
    def fail(*a,**k): raise RuntimeError("synthetic kernel failure")
    monkeypatch.setattr(lattice.np,"correlate",fail)
    with pytest.raises(RuntimeError,match="synthetic kernel failure"): lattice.estimate_period(patch)
    observer=growth._ScalarAttritionObserver(max_items=1,max_bytes=256)
    with growth._scalar_attrition_context(observer):
        with growth._scalar_period_context(identity=("frame-4",2,0,0),axis=1):
            growth._scalar_period_expect("period_estimate")
            with pytest.raises(RuntimeError,match="synthetic kernel failure"): lattice.estimate_period(patch)
    record=observer.record()
    assert record["status"]=="incomplete" and record["omitted_accounting_events"]>0
    runner._validate_attrition_record(record)
    path=tmp_path/"scalar.json"
    runner._publish_new(path,runner._bounded_bytes(record))
    assert json.loads(path.read_bytes())["status"]=="incomplete"
    assert list(tmp_path.iterdir())==[path]
    forged=dict(record,status="complete")
    with pytest.raises(runner.DiagnosticError): runner._validate_attrition_record(forged)


class _KernelArray(np.ndarray):
    def __array_finalize__(self, source):
        self.calls=getattr(source,"calls",None)
    def mean(self,*a,**k):
        if self.calls is not None: self.calls.append("mean")
        return super().mean(*a,**k)
    def var(self,*a,**k):
        if self.calls is not None: self.calls.append("variance")
        return super().var(*a,**k)
    def sum(self,*a,**k):
        if self.calls is not None: self.calls.append("sum")
        return super().sum(*a,**k)


@pytest.mark.parametrize("short", (False,True))
def test_exact_existing_variance_kernel_count(short):
    patch=(_striped(128,8.) if not short else np.ones((3,4),np.float32)).view(_KernelArray)
    calls=[]; patch.calls=calls
    off=lattice.estimate_period(patch); original=list(calls); calls.clear()
    on,record=_observe(lambda:lattice.estimate_period(patch),"period_estimate",identity=("frame-3",0,0,0))
    assert on==off and calls==original
    assert calls.count("variance")==int(not short)
    assert record["status"]=="complete"


def test_measurement_three_round_join_and_limit_are_exact(monkeypatch):
    requests=[]
    def rectify(image,*a,**k):
        requests.append(k["samples"])
        return _striped(k["samples"],8.) if image is patch else np.ones((k["samples"],k["samples"]))
    patch=np.zeros((8,8),np.float32)
    monkeypatch.setattr(lattice,"rectify_plane",rectify)
    off=_measure(patch,max_side=2048); original=list(requests); requests.clear()
    on,record=_observe(lambda:_measure(patch,max_side=2048),"period_measurement",identity=("frame-4",5))
    assert on[0]==off[0] and on[1:2]==off[1:2] and on[4]==off[4] and requests==original
    assert record["status"]=="complete"
    rounds=sorted(_details(record,"period_round"),key=lambda row:row["identity"])
    assert [row["identity"] for row in rounds]==[("frame-4",5,i) for i in range(3)]
    assert [row["disposition"] for row in rounds]==["resample","resample","round_limit"]
    assert _details(record,"period_measurement")[0]["attempted_rounds"]==3
    estimates=_details(record,"period_estimate")
    assert {row["identity"] for row in estimates}=={("frame-4",5,r,d) for r in range(3) for d in (0,1)}


def test_measurement_missing_reconciliation_hook_is_incomplete(monkeypatch):
    patch=_striped(64,8.)
    monkeypatch.setattr(lattice,"rectify_plane",lambda *a,**k:patch)
    monkeypatch.setattr(lattice,"_plane_period_with_direction",lambda *a,**k:None)
    result,record=_observe(lambda:_measure(patch),"period_measurement",identity=("frame-3",1))
    assert result[0] is None and record["status"]=="incomplete"
    assert record["stages"]["period_reconciliation"]["unclassified"]==1


def test_full_scalar_details_survive_example_truncation_and_charge_budget():
    observer=growth._ScalarAttritionObserver()
    for ordinal in range(130):
        with growth._scalar_attrition_context(observer):
            with growth._scalar_period_context(identity=("frame-4",ordinal,0,0),axis=1):
                growth._scalar_period_expect("period_estimate")
                growth._scalar_period_finish("period_estimate","short_profile",profile_length=3,
                                            variance=None,variance_state="not_evaluated")
    record=observer.record()
    runner._validate_attrition_record(record)
    stage=record["stages"]["period_estimate"]
    assert record["status"]=="complete" and len(stage["counts"])==130
    assert stage["omitted_examples"]==2 and len(stage["examples"])==128
    assert {tuple(row["identity"])[1] for row in stage["counts"]}==set(range(130))
    assert observer.bytes>130*1024


def test_profile_scalar_iteration_cap_marks_incomplete_without_changing_estimate():
    patch=_striped(2050,8.)[:4,:]
    weights=np.ones(patch.shape)
    off=lattice.estimate_period(patch,weights=weights)
    on,record=_observe(lambda:lattice.estimate_period(patch,weights=weights),"period_estimate",identity=("frame-4",0,0,0))
    assert on==off and record["status"]=="incomplete"
    assert _details(record,"period_profile")[0]["positive_weight_bins_state"]=="not_evaluated"


def test_observer_fault_and_disabled_nested_scope_preserve_original_result(monkeypatch):
    patch=_striped(128,8.)
    off=lattice.estimate_period(patch)
    observer=growth._ScalarAttritionObserver()
    monkeypatch.setattr(observer,"finish",lambda *a,**k:(_ for _ in ()).throw(RuntimeError("observer")))
    with growth._scalar_attrition_context(observer):
        with growth._scalar_period_context(identity=("frame-3",0,0,0),axis=1):
            growth._scalar_period_expect("period_estimate")
            on=lattice.estimate_period(patch)
            with growth._scalar_attrition_context(None):
                disabled=lattice.estimate_period(patch)
    assert on==off==disabled and observer.incomplete


@pytest.mark.parametrize("accepted", (False,True))
def test_real_measurement_target_or_both_direction_failure(monkeypatch,accepted):
    patch=_striped(64,16.) if accepted else np.zeros((64,64),np.float32)
    mask=np.ones(patch.shape).view(_KernelArray)
    calls=[]; mask.calls=calls
    monkeypatch.setattr(lattice,"rectify_plane",lambda image,*a,**k:patch if image is patch else mask)
    off=_measure(patch); original=list(calls); calls.clear()
    on,record=_observe(lambda:_measure(patch),"period_measurement",identity=("frame-3",9))
    assert on[0]==off[0] and on[1]==off[1] and on[4]==off[4] and calls==original
    assert record["status"]=="complete"
    assert _details(record,"period_round")[0]["disposition"]==("target_period_reached" if accepted else "no_directional_period")
    assert len(_details(record,"period_estimate"))==2
    reconciliation=_details(record,"period_reconciliation")[0]
    for name in ("selected_direction","selected_period","selected_strength"):
        assert reconciliation[name+"_state"]==("measured" if accepted else "not_evaluated")


def test_genuine_missing_whole_measurement_hook_has_no_stub_exemption(monkeypatch):
    from test_lattice_planes import _v1_region_control,_controlled_v1_readings
    proposal,usable,calibration,_polygon=_v1_region_control()
    image,_=_controlled_v1_readings(monkeypatch,(proposal,),usable,calibration,frame="frame-3")
    proposals=growth.PlaneProposalResult("ok","synthetic",(proposal,))
    observer=growth._ScalarAttritionObserver()
    with growth._scalar_attrition_context(observer):
        planes.frame_plane_readings_from_proposals(image,usable,calibration,proposals,frame="frame-3")
    record=observer.record()
    assert record["status"]=="incomplete"
    assert record["stages"]["period_measurement"]["unclassified"]==1
    assert record["stages"]["reading"]["unclassified"]==0
