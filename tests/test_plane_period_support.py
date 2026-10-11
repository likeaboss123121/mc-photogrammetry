"""Synthetic/in-memory observer QA; no real R8/RGB/registered capture."""
import copy
from dataclasses import asdict
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import threading

import numpy as np
import pytest

from mc_photogrammetry.reconstruction import lattice, lattice_planes as lp, plane_graph as pg

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("period_support_capture",ROOT/"scripts/evaluation/capture_plane_period_support.py")
subject=importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name]=subject; SPEC.loader.exec_module(subject)


def scope(observer,signatures=(),matches=()):
    return lattice._period_support_scope(observer,run_id="synthetic-only",signature_targets=signatures,match_targets=matches)


def collect():
    events=[]
    def receive(kind,context,payload): events.append((kind,context,payload))
    return events,receive


def fingerprint(value):
    if isinstance(value,np.ndarray): return (value.dtype.str,value.shape,value.tobytes(),value.flags.writeable)
    if isinstance(value,(tuple,list)): return tuple(fingerprint(v) for v in value)
    if isinstance(value,dict): return tuple((key,fingerprint(item)) for key,item in value.items())
    return value


def spy(monkeypatch,module,name,calls):
    original=getattr(module,name)
    def delegated(*args,**kwargs):
        calls.append((name,fingerprint(args),fingerprint(kwargs)))
        return original(*args,**kwargs)
    monkeypatch.setattr(module,name,delegated)


def patch():
    yy,xx=np.indices((128,128)); return (((xx%12<2)|(yy%12<2))*200).astype(np.float32)


def seed_source(semantic,frame,image,usable,axes,focal,grey=None):
    rgb=image if image.ndim==3 else np.repeat(image.astype(np.uint8)[...,None],3,axis=2)
    sha=hashlib.sha256(rgb.tobytes()).hexdigest(); h,w=usable.shape
    ingress=dict(frame_id=frame,byte_sha256="a"*64,decoded_shape=rgb.shape,
                 decoded_dtype=rgb.dtype.str,decoded_pixel_sha256=sha)
    semantic.trusted_ingress[frame]=ingress
    payload=dict(frame=frame,rgb=rgb,input_identity=dict(frame_id=frame,height=h,width=w,sha256="a"*64),
                 ingress=ingress,decoded_sha256=sha)
    semantic("frame_source",{},payload)
    import cv2
    source_grey=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY) if grey is None else grey
    source_grey=source_grey.copy(); source_grey[~usable]=0
    semantic("frame_measure_source",{},dict(frame=frame,usable=usable,grey=source_grey,
             structural_weights=np.zeros((h,w),dtype=np.float32),axis_matrix=axes,
             focal=focal,principal_point=(w/2.,h/2.),selection_policy=subject.registered_reading_policy()))


@pytest.mark.parametrize("dtype",["u1","?","<i8",">i8","<f4",">f8"])
def test_native_arrays_lossless_endian_and_immutable(dtype):
    array=np.arange(12).astype(dtype).reshape(3,4)[:,::-1]
    record=subject.array_record(array); restored=subject.decode(record)
    assert restored.dtype.str==array.dtype.str and restored.tobytes()==array.tobytes()
    assert not restored.flags.writeable
    with pytest.raises(ValueError): restored.flags.writeable=True


@pytest.mark.parametrize("field,value",[("dtype","O"),("byte_order",">"),("shape",[-1]),("byte_length",999),("sha256","f"*64),("payload_base64","AAAA")])
def test_resealed_native_descriptor_tamper_refuses(field,value):
    descriptor=subject.array_record(np.arange(8,dtype=np.int64))["native_array"]
    descriptor[field]=value; body={k:v for k,v in descriptor.items() if k!="descriptor_sha256"}
    descriptor["descriptor_sha256"]=subject.digest("native-array",body)
    with pytest.raises(subject.CaptureRefusal): subject.array_decode(descriptor)


def test_unsupported_dtype_and_native_boundary(monkeypatch):
    monkeypatch.setitem(subject.CAPS,"native_array",64)
    assert subject.decode(subject.array_record(np.zeros(64,dtype=np.uint8))).nbytes==64
    with pytest.raises(subject.CaptureRefusal,match="native_array"): subject.array_record(np.zeros(65,dtype=np.uint8))
    with pytest.raises(subject.CaptureRefusal): subject.array_record(np.zeros(1,dtype=np.complex128))


def test_journal_multimember_order_digest_and_bounded_replay():
    file=io.BytesIO(); journal=subject.Journal(file,run_id="pure")
    journal.append("first",{"run_id":"pure"},{"array":np.arange(16,dtype=np.float32)})
    journal.append("second",{"run_id":"pure"},{"kept":True})
    seen=[]; result=subject.replay(file,journal.members,run_id="pure",expected_sha256=journal.sha.hexdigest(),consumer=lambda *args:seen.append(args))
    assert result["events"]==2 and [row[0] for row in seen]==["first","second"]
    assert np.array_equal(seen[0][2]["array"],np.arange(16,dtype=np.float32))


@pytest.mark.parametrize("corruption",["trailer","crc","junk","extra_member","swap","digest"])
def test_stream_corruption_refuses(corruption):
    file=io.BytesIO(); journal=subject.Journal(file,run_id="pure")
    journal.append("first",{},{}); journal.append("second",{}, {})
    raw=file.getvalue(); members=copy.deepcopy(journal.members); expected=journal.sha.hexdigest()
    if corruption=="trailer": raw=raw[:-1]
    elif corruption=="crc": raw=raw[:-8]+bytes([raw[-8]^1])+raw[-7:]
    elif corruption=="junk": raw+=b"junk"
    elif corruption=="extra_member": raw+=raw[:members[0]["length"]]
    elif corruption=="swap": members.reverse()
    elif corruption=="digest": expected="0"*64
    with pytest.raises(subject.CaptureRefusal): subject.replay(io.BytesIO(raw),members,run_id="pure",expected_sha256=expected)


@pytest.mark.parametrize("cap",["compressed_member","compressed_total","decoded_member","decoded_total"])
def test_exact_codec_cap_allowed_next_byte_refuses(cap):
    seed=subject.Journal(io.BytesIO(),run_id="pure"); seed.append("one",{}, {"text":"a"*40})
    bound=seed.total if cap.startswith("compressed") else seed.decoded_total
    caps=dict(subject.CAPS); caps[cap]=bound
    exact=subject.Journal(io.BytesIO(),run_id="pure",caps=caps); exact.append("one",{}, {"text":"a"*40})
    assert exact.total==seed.total
    caps[cap]=bound-1
    too_small=subject.Journal(io.BytesIO(),run_id="pure",caps=caps)
    with pytest.raises(subject.CaptureRefusal): too_small.append("one",{}, {"text":"a"*40})
    assert too_small.failed


def test_partial_later_failure_preserves_previous_complete_member():
    file=io.BytesIO(); journal=subject.Journal(file,run_id="pure")
    journal.append("first",{}, {"token":1}); prefix=file.getvalue()
    journal.caps["compressed_total"]=journal.total+1
    with pytest.raises(subject.CaptureRefusal): journal.append("later",{}, {"token":2})
    assert file.getvalue().startswith(prefix) and journal.members[0]["length"]==len(prefix)
    assert subject._decode_member(prefix,subject.CAPS)["kind"]=="first"
    with pytest.raises(subject.CaptureRefusal): journal.append("retry",{}, {})


def test_snapshot_callbacks_cannot_mutate_original_buffers_or_flags():
    image=patch(); original=image.copy(); seen=[]
    def attempt(kind,context,payload):
        if kind=="profile":
            observed=payload["profile"]; seen.append(observed)
            with pytest.raises(ValueError): observed.flags.writeable=True
            with pytest.raises(ValueError): observed[0]=0
            payload.clear(); context.clear()
    off=lattice._plane_period_with_direction(image)
    with scope(attempt) as state: on=lattice._plane_period_with_direction(image)
    assert on==off and state.incomplete is None and seen
    assert np.array_equal(image,original) and image.flags.writeable


@pytest.mark.parametrize("fault",[RuntimeError("write"),lattice._PeriodSupportIncomplete("cap_exhausted"),ValueError("serialize")])
def test_capture_fault_latches_and_preserves_numeric_period(monkeypatch,fault):
    image=patch(); original=image.copy(); calls=[]
    def broken(*_args): calls.append(1); raise fault
    off=lattice._plane_period_with_direction(image)
    with scope(broken) as state: on=lattice._plane_period_with_direction(image)
    assert on==off and state.incomplete["status"]=="incomplete" and len(calls)==1
    assert np.array_equal(image,original)
    assert lattice._PERIOD_SUPPORT_CONTEXT.get() is None


@pytest.mark.parametrize("interrupt",[KeyboardInterrupt,SystemExit])
def test_interrupts_propagate_and_scope_resets(interrupt):
    def interrupted(*_args): raise interrupt()
    with pytest.raises(interrupt):
        with scope(interrupted): lattice.estimate_period(patch())
    assert lattice._PERIOD_SUPPORT_CONTEXT.get() is None
    with scope(lambda *_args:None) as state: assert state.incomplete is None


def test_nesting_foreign_thread_and_run_are_rejected():
    events,receive=collect()
    with scope(receive) as state:
        with pytest.raises(lattice._PeriodSupportIncomplete):
            with scope(receive): pass
        context=__import__("contextvars").copy_context()
        thread=threading.Thread(target=lambda:context.run(lattice._period_support_event,"foreign"))
        thread.start(); thread.join()
        assert state.incomplete["reason"]=="identity_drift" and not events
    journal=subject.Journal(io.BytesIO(),run_id="owned"); observer=subject.Observer(journal,run_id="owned")
    with pytest.raises(subject.CaptureRefusal): observer("test",{"run_id":"foreign"},{})
    with scope(receive): lattice._period_support_event("later",ok=True)
    assert [row[0] for row in events]==["later"]


def test_all_directional_period_profile_calls_and_args_match(monkeypatch):
    image=patch(); weights=np.ones_like(image); calls=[]
    for name in ("_profile","estimate_period","_plane_period_with_direction"): spy(monkeypatch,lattice,name,calls)
    off=lattice._plane_period_with_direction(image,weights=weights); off_calls=list(calls); calls.clear()
    events,receive=collect()
    with scope(receive): on=lattice._plane_period_with_direction(image,weights=weights)
    assert on==off and calls==off_calls
    assert [row[2]["along"] for row in events if row[0]=="period_estimate"]==[0,1]
    assert any(row[0]=="period_reconciliation" for row in events)


def test_actual_bounded_measurement_and_semantic_profiles_match(monkeypatch):
    image=patch(); usable=np.ones_like(image); axes=np.asarray(((1,0,0),(0,0,1),(0,-1,0)),dtype=float)
    calls=[]
    for name in ("rectify_plane","_profile","estimate_period","_plane_period_with_direction"): spy(monkeypatch,lattice,name,calls)
    window=lattice.plane_window(usable,1,axes,64.,principal_point=(64.,64.))
    kwargs=dict(mask=usable,centre=window[:2],principal_point=(64.,64.),extent=window[2],side=window[3],max_rectified_side=128)
    off=lp._bounded_measure(image,1,axes,64.,**kwargs); before=list(calls); calls.clear()
    events,receive=collect()
    with scope(receive):
        with lattice._period_support_context(frame="synthetic",candidate_id=("synthetic",0),candidate_axis=1):
            on=lp._bounded_measure(image,1,axes,64.,**kwargs)
    assert off[0] is not None and on[0]==off[0] and calls==before
    assert all(np.array_equal(on[index],off[index]) for index in (2,3))
    mask=np.ones((128,128),dtype=bool); polygon=((0.,0.),(127.,0.),(127.,127.),(0.,127.))
    region=lp._make_region_evidence(mask,1,polygon,("synthetic",))
    semantic=subject.SemanticReplay(target_readings=(),target_matches=())
    seed_source(semantic,"synthetic",image,mask,axes,64.,grey=image)
    semantic("candidate_begin",{"candidate_id":("synthetic",0)},lattice._period_support_snapshot(dict(axis=1,polygon=polygon,region_evidence=region)))
    semantic("candidate_window",{"candidate_id":("synthetic",0)},lattice._period_support_snapshot(dict(window=window,region_evidence=region)))
    for kind,context,payload in events: semantic(kind,context,payload)
    assert semantic.candidates[("synthetic",0)]["rounds"]>=1


def test_three_adaptive_rounds_same_rectifications_and_final_sampling(monkeypatch):
    def rectify(_image,*_args,samples,**_kwargs): return np.ones((samples,samples),dtype=np.float32)
    monkeypatch.setattr(lattice,"rectify_plane",rectify)
    def run(observed):
        periods=iter((5.,7.,9.)); calls=[]
        def period(_patch,**_kwargs): calls.append(_patch.shape); return lattice.PeriodEstimate(next(periods),0.,.9),1
        monkeypatch.setattr(lattice,"_plane_period_with_direction",period)
        args=(np.ones((8,8)),1,np.eye(3),30.)
        kwargs=dict(mask=np.ones((8,8)),centre=(0.,0.),principal_point=(4.,4.),extent=1.,side=1.,max_rectified_side=400)
        if observed:
            events,receive=collect()
            with scope(receive): result=lp._bounded_measure(*args,**kwargs)
            return result,calls,events
        return lp._bounded_measure(*args,**kwargs),calls,[]
    off,before,_=run(False); on,after,events=run(True)
    assert on[0]==off[0] and on[1]==off[1] and before==after and len(after)==3
    assert [row[2]["round_index"] for row in events if row[0]=="measurement_round"]==[0,1,2]


def test_all_evaluated_b_rejects_prune_and_numeric_args_preserved(monkeypatch):
    image=patch(); estimate=lattice.PeriodEstimate(12.,0.,.9,((2,.05),(4,.8)))
    calls=[]; spy(monkeypatch,lp,"_autocorrelation_support",calls)
    kwargs=dict(extent=1.,samples=128,patch=image,weights=np.ones_like(image),profile_direction=1)
    off=lp._distance_alternatives(estimate,**kwargs); before=list(calls); calls.clear()
    events,receive=collect()
    with scope(receive): on=lp._distance_alternatives(estimate,**kwargs)
    assert on==off and calls==before and len(calls)==5
    event=next(row[2] for row in events if row[0]=="distance_alternatives")
    assert len(event["evaluated"])>len(on) and any(s<event["prune_floor"] for b,s in event["evaluated"])
    candidate=("s",0); semantic=subject.SemanticReplay(target_readings=(),target_matches=())
    profile=lattice._profile(image,1,np.ones_like(image))
    semantic.candidates[candidate]={"supra":{},"last_round":dict(samples=128,extent=1.,
        estimate=lattice._period_support_snapshot(estimate),direction=1,profiles={1:profile})}
    for kind,context,payload in events:
        if kind in ("profile","supra_support","distance_alternatives"):
            semantic(kind,dict(context,candidate_id=candidate),payload)
    tampered=copy.deepcopy(event); first=dict(tampered["retained"][0]); first["fields"]["distance_blocks"]+=.01; tampered["retained"]=(first,*tampered["retained"][1:])
    with pytest.raises(subject.CaptureRefusal): semantic("distance_alternatives",{"candidate_id":candidate},tampered)


def signatures():
    from test_plane_graph import make_reading,make_signature
    rng=np.random.default_rng(991); cells=rng.normal(size=(17,19)).astype(np.float32)
    left=make_reading("s0",2,1.,((1.,2.,1.),)); right=make_reading("s1",2,1.,((1.,2.,1.),))
    return (left,right),(make_signature(left,cells),make_signature(right,cells))


def test_real_matcher_exact_target_only_parity_and_witness(monkeypatch):
    readings,sigs=signatures(); target=("s0",0,2,1.,"s1",0,2,1.,0)
    calls=[]
    for name in ("match_plane_pair","cell_grid_homography","warp_signature","_correlation_surface_with_support","_peak_statistics"): spy(monkeypatch,pg,name,calls)
    off=pg.match_plane_graph(readings,sigs); before=list(calls); calls.clear(); events,receive=collect()
    with scope(receive,matches=(target,)) as state: on=pg.match_plane_graph(readings,sigs)
    assert off and on==off and calls==before and state.incomplete is None
    pairs=[row for row in events if row[0]=="matcher_pair"]
    assert len(pairs)==1 and tuple(pairs[0][1]["match_identity"])==target
    assert pairs[0][2]["logical_pairs"].shape[0]==pairs[0][2]["metadata"]["selected_support"]
    assert sum(row[0]=="matcher_warp" for row in events)==1


@pytest.mark.parametrize("fault",["construction","write","cap"])
def test_real_matcher_diagnostic_failure_preserves_matches_and_refusals(monkeypatch,fault):
    readings,sigs=signatures(); target=("s0",0,2,1.,"s1",0,2,1.,0)
    before=[]; off=pg.match_plane_graph(readings,sigs,refusals=before)
    if fault=="construction":
        monkeypatch.setattr(pg,"MatcherPairWitnessV1",lambda *_args,**_kwargs:(_ for _ in ()).throw(RuntimeError("witness")))
    def receive(*_args):
        if fault=="write": raise OSError("write")
        if fault=="cap": raise lattice._PeriodSupportIncomplete("cap_exhausted")
    after=[]
    with scope(receive,matches=(target,)) as state: on=pg.match_plane_graph(readings,sigs,refusals=after)
    assert on==off and after==before and state.incomplete is not None


def test_non_target_pair_assembles_no_private_witness(monkeypatch):
    readings,sigs=signatures()
    monkeypatch.setattr(pg,"MatcherPairWitnessV1",lambda *_args,**_kwargs:pytest.fail("non-target witness"))
    with scope(lambda *_args:None,matches=(("other",0,2,1.,"s1",0,2,1.,0),)) as state:
        assert pg.match_plane_graph(readings,sigs)
    assert state.incomplete is None


def test_target_signature_same_numeric_calls_and_source_witness(monkeypatch):
    yy,xx=np.indices((128,128)); image=np.repeat(((xx*7+yy*3)%251).astype(np.uint8)[...,None],3,axis=2)
    usable=np.ones((128,128),dtype=bool); polygon=((0.,0.),(127.,0.),(127.,127.),(0.,127.))
    region=lp._make_region_evidence(usable,2,polygon,("synthetic",))
    reading=lp.PlaneReading("s",2,1.,(lp.DistanceAlternative(1.,127./32.,1.),),16.,(0.,0.),1.,128,
        polygon,8,.6,.9,lp.PlaneRectification(tuple(tuple(float(v) for v in row) for row in np.eye(3)),32.,(64.,64.),(0.,0.)),region)
    b=1.; calls=[]; spy(monkeypatch,lattice,"rectify_plane",calls)
    off=lp.plane_signature(image,reading,b,usable=usable,max_cells=8); before=list(calls); calls.clear()
    events,receive=collect()
    with scope(receive,signatures=(("s",0),)) as state:
        assert lattice._period_support_signature(("s",0,0,b))
        with lattice._period_support_context(signature_target=True,signature_identity=("s",0,0,b)):
            on=lp.plane_signature(image,reading,b,usable=usable,max_cells=8)
    assert calls==before and state.incomplete is None
    assert subject.same(lattice._period_support_snapshot(on),lattice._period_support_snapshot(off))
    assert sum(row[0]=="signature_rectification" for row in events)==3
    assert sum(row[0]=="signature_sampling" for row in events)==1
    semantic=subject.SemanticReplay(target_readings=(("s",0),),target_matches=())
    seed_source(semantic,"s",image,usable,np.eye(3),32.)
    semantic.readings[("s",0)]=lattice._period_support_snapshot(reading)
    for kind,context,payload in events: semantic(kind,context,payload)
    assert semantic.signature_parts[("s",0,0,b)]["witness_sha256"]


def test_prereg_upstream_and_runtime_drift_refuse_without_loading(monkeypatch):
    registration=subject.freeze_preregistration(); assert subject.freeze_preregistration()==registration
    original=subject.dependency_hashes
    monkeypatch.setattr(subject,"dependency_hashes",lambda:dict(original(),**{"src/upstream.py":"0"*64}))
    with pytest.raises(subject.CaptureRefusal): subject.verify_preregistration(registration)
    monkeypatch.setattr(subject,"dependency_hashes",original)
    runtime=subject.runtime_pins()
    monkeypatch.setattr(subject,"runtime_pins",lambda:dict(runtime,numpy="changed"))
    with pytest.raises(subject.CaptureRefusal): subject.verify_preregistration(registration)
    assert not subject.EVIDENCE.exists()


def test_run_claim_durable_consumed_before_loader_and_no_retry(tmp_path,monkeypatch):
    report=tmp_path/"report.md"; evidence=tmp_path/"unused.gz"; report.write_text(subject.MARKER)
    events=[]; monkeypatch.setattr(subject.os,"fsync",lambda _fd:events.append("fsync"))
    subject.claim_run(report,evidence)
    assert events==["fsync"] and "RUNNING" in report.read_text()
    with pytest.raises(subject.CaptureRefusal): subject.claim_run(report,evidence)


def test_preexisting_evidence_refuses_without_consuming_marker(tmp_path):
    report=tmp_path/"report.md"; evidence=tmp_path/"present.gz"; report.write_text(subject.MARKER); evidence.write_bytes(b"prefix")
    with pytest.raises(subject.CaptureRefusal): subject.claim_run(report,evidence)
    assert report.read_text()==subject.MARKER and evidence.read_bytes()==b"prefix"


def test_missing_chain_and_duplicate_identity_fail_closed():
    semantic=subject.SemanticReplay(target_readings=(),target_matches=())
    with pytest.raises(subject.CaptureRefusal): semantic.finish()
    semantic("core_final",{}, {})
    with pytest.raises(subject.CaptureRefusal): semantic("core_final",{}, {})
    events,receive=collect()
    with scope(receive,signatures=(("s",0),)) as state:
        assert lattice._period_support_signature(("s",0,0,1.))
        assert not lattice._period_support_signature(("s",0,0,1.))
    assert state.incomplete["reason"]=="ambiguous_join"


def test_owned_python_ast_and_byte_whitespace():
    import ast
    paths=[ROOT/"scripts/evaluation/capture_plane_period_support.py",Path(__file__)]
    for path in paths:
        raw=path.read_bytes(); ast.parse(raw.decode())
        assert raw.endswith(b"\n") and all(line.rstrip(b" \t")==line for line in raw.splitlines())
        assert not any(isinstance(node,ast.Assert) for node in ast.walk(ast.parse(raw.decode()))) if path.name.startswith("capture_") else True


@pytest.fixture(scope="module")
def graph_inputs():
    from test_image_derived_lattice_measure import _result
    from mc_photogrammetry.pipeline import image_derived_signatures as signatures_mod
    patcher=pytest.MonkeyPatch()
    try:
        session,_pose,readings=_result(patcher)
        result=signatures_mod.derive_image_derived_signatures(session=session,lattice_result=readings)
        yield dict(session=session,lattice_result=readings,signatures_result=result)
    finally: patcher.undo()


@pytest.mark.parametrize("outcome",["success","solver_refusal","no_matches"])
@pytest.mark.parametrize("capture_failure",[False,True])
def test_graph_adapter_default_observer_and_failed_observer_exact_parity(
        graph_inputs,monkeypatch,outcome,capture_failure):
    import dataclasses
    from test_image_derived_graph_phase_capture import _phase_fixture
    from mc_photogrammetry.pipeline import image_derived_graph as graph
    _readings,matches,solution=_phase_fixture()
    if outcome=="success": solution=dataclasses.replace(solution,reason=None)
    calls=[]
    def match(*args,**kwargs):
        calls.append(("match",fingerprint(args),fingerprint(kwargs)))
        return () if outcome=="no_matches" else matches
    def solve(*args,**kwargs):
        calls.append(("solve",fingerprint(args),fingerprint(kwargs)))
        return solution
    monkeypatch.setattr(pg,"match_plane_graph",match)
    monkeypatch.setattr(pg,"solve_component",solve)
    off=graph.derive_image_derived_graph(**graph_inputs)
    assert isinstance(off,graph.ImageDerivedGraphResult)
    expected=list(calls); calls.clear(); events=[]
    def receive(*event):
        if capture_failure: raise OSError("diagnostic-only")
        events.append(event)
    with scope(receive) as state: on=graph.derive_image_derived_graph(**graph_inputs)
    assert on==off and on.result_sha256==off.result_sha256 and calls==expected
    assert sum(row[0]=="solve" for row in calls)==(outcome!="no_matches")
    assert (state.incomplete is not None)==(capture_failure and outcome!="no_matches")
    if not capture_failure and outcome!="no_matches":
        assert [row[0] for row in events]==["solver_call","solver_result"]
        assert subject.same(events[1][2]["solution"],lattice._period_support_snapshot(solution))
    assert lattice._PERIOD_SUPPORT_CONTEXT.get() is None


def test_monotone_memory_boundary_and_aggregate_payloads_precede_copy(monkeypatch):
    budget=lattice._PeriodSupportMemory(100)
    budget.reserve(40,"retained"); budget.reserve(60,"overlap")
    assert budget.used==budget.peak==100
    with pytest.raises(lattice._PeriodSupportIncomplete): budget.reserve(1,"next")
    assert budget.used==100
    events,receive=collect(); payload=dict(text="x"*100,blob=b"y"*100,array=np.zeros(100,dtype=np.uint8))
    planned=lattice._period_support_size_plan(payload)
    with scope(receive) as state:
        state.memory=lattice._PeriodSupportMemory(planned)
        lattice._period_support_snapshot(payload)
        assert state.memory.used==planned
        with pytest.raises(lattice._PeriodSupportIncomplete): lattice._period_support_snapshot(payload)
    with scope(receive) as state:
        state.memory=lattice._PeriodSupportMemory(10)
        lattice._period_support_event("too_big",**payload)
        assert state.incomplete["reason"]=="cap_exhausted" and not events


def test_journal_memory_failure_precedes_allocation_and_write(monkeypatch):
    caps=dict(subject.CAPS,extra_memory=1)
    journal=subject.Journal(io.BytesIO(),run_id="pure",caps=caps)
    monkeypatch.setattr(subject,"sealed_event",lambda *_args:pytest.fail("allocated before cap"))
    with pytest.raises(subject.CaptureRefusal,match="aggregate_memory"):
        journal.append("too_big",{}, {"blob":b"x"*50})
    assert journal.failed and journal.file.getvalue()==b""


def test_decoder_memory_failure_precedes_decompress_allocation(monkeypatch):
    journal=subject.Journal(io.BytesIO(),run_id="pure"); journal.append("one",{}, {})
    monkeypatch.setattr(subject.zlib,"decompressobj",lambda *_args:pytest.fail("decoder allocated before cap"))
    with pytest.raises(subject.CaptureRefusal,match="aggregate_memory"):
        subject._decode_member(journal.file.getvalue(),dict(subject.CAPS,extra_memory=1))


def test_context_memory_failure_is_diagnostic_and_preserves_production_period():
    off=lattice.estimate_period(patch())
    with scope(lambda *_args:None) as state:
        state.memory=lattice._PeriodSupportMemory(1)
        on=lattice.estimate_period(patch())
    assert on==off and state.incomplete["reason"]=="cap_exhausted"
    assert lattice._PERIOD_SUPPORT_CONTEXT.get() is None


@pytest.mark.parametrize("permutation",[(0,1),(1,0)])
def test_signature_observer_uses_matcher_order_without_changing_iteration(
        graph_inputs,monkeypatch,permutation):
    import dataclasses
    from types import SimpleNamespace
    from test_plane_graph import make_reading,make_signature
    from mc_photogrammetry.pipeline import image_derived_signatures as sigmod
    frame=graph_inputs["session"].frames[0].frame_id
    high=dataclasses.replace(make_reading(frame,2,1.,((1.,2.,1.),)),period_pixels=30.,confidence=.9)
    low=dataclasses.replace(high,period_pixels=10.,confidence=.1)
    readings=tuple((high,low)[i] for i in permutation)
    base=graph_inputs["lattice_result"]
    carrier=SimpleNamespace(status="assembled",readings=readings,frame_ids=base.frame_ids,
                            session_sha256=base.session_sha256,result_sha256=base.result_sha256)
    monkeypatch.setattr(sigmod,"_replay_lattice",lambda _value:carrier)
    calls=[]
    def sampled(_image,reading,b,**kwargs):
        calls.append((reading.period_pixels,b,fingerprint(kwargs)))
        return make_signature(reading,np.arange(16,dtype=np.float32).reshape(4,4))
    monkeypatch.setattr(lp,"plane_signature",sampled)
    off=sigmod.derive_image_derived_signatures(session=graph_inputs["session"],lattice_result=base)
    before=list(calls); calls.clear(); identities=[]
    original=lattice._period_support_signature
    def selected(identity): identities.append(tuple(identity)); return original(identity)
    monkeypatch.setattr(lattice,"_period_support_signature",selected)
    with scope(lambda *_args:None):
        on=sigmod.derive_image_derived_signatures(session=graph_inputs["session"],lattice_result=base)
    assert subject.same(lattice._period_support_snapshot(off),lattice._period_support_snapshot(on)) and calls==before
    assert [item[1] for item in identities]==[1 if r is high else 0 for r in readings]
    assert [subject.reading_record_order_key(lattice._period_support_snapshot(r)) for r in sorted(readings,key=lp.reading_order_key)]==sorted(subject.reading_record_order_key(lattice._period_support_snapshot(r)) for r in readings)


@pytest.mark.parametrize("finish_failure",[False,True])
@pytest.mark.parametrize("outcome",["return","refusal","exception","interrupt"])
def test_stage_scope_finalization_and_restoration(monkeypatch,finish_failure,outcome):
    stage=subject._import_stage(); calls=[]
    result=dict(status="refused" if outcome=="refusal" else "complete")
    def body(**kwargs):
        calls.append(kwargs)
        if outcome=="exception": raise ValueError("production")
        if outcome=="interrupt": raise KeyboardInterrupt()
        return result
    monkeypatch.setattr(stage,"_execute_graph_stage",body)
    class Capture:
        run_id="synthetic"; signature_targets=(); match_targets=(); incomplete=None
        def __call__(self,*_args): pass
        def finish_capture(self,value):
            self.value=value
            if finish_failure: raise OSError("finish")
    observer=Capture(); kwargs=dict(png_paths=(),expected_sha256=(),project_path="unused",output_path="unused",telemetry_path="unused")
    if outcome in ("exception","interrupt"):
        with pytest.raises(ValueError if outcome=="exception" else KeyboardInterrupt):
            stage.execute_graph_stage(**kwargs,_period_support_observer=observer)
    else:
        assert stage.execute_graph_stage(**kwargs,_period_support_observer=observer) is result
        assert stage.execute_graph_stage(**kwargs) is result
        assert calls[0]==calls[1]
    assert lattice._PERIOD_SUPPORT_CONTEXT.get() is None
    assert (observer.incomplete is not None)==finish_failure


def reseal(kind,context,payload):
    value=subject.sealed_event(0,"0"*64,"synthetic",kind,context,payload)
    return subject.validate_event(value,0,"0"*64,"synthetic")


@pytest.mark.parametrize("change",["correlation","peaks","harmonics","period","phase","strength"])
def test_resealed_signal_peak_harmonic_and_phase_tamper(change):
    events,receive=collect()
    with scope(receive): lattice.estimate_period(patch(),0)
    payload=copy.deepcopy(next(p for k,c,p in events if k=="period_estimate"))
    subject.verify_period_arithmetic(payload,payload["profile"])
    if change in ("correlation","peaks","harmonics"):
        key={"correlation":"correlation","peaks":"peaks","harmonics":"harmonic_candidates"}[change]
        altered=payload[key].copy(); altered[0]+=1; payload[key]=altered
    else:
        key={"period":"period_pixels","phase":"phase_pixels","strength":"strength"}[change]
        payload["estimate"]["fields"][key]+=.01
    _context,checked=reseal("period_estimate",{},payload)
    with pytest.raises(subject.CaptureRefusal): subject.verify_period_arithmetic(checked,checked["profile"])


def test_resealed_foreign_rgb_and_ingress_substitution_rejected():
    image=np.zeros((8,8,3),dtype=np.uint8); usable=np.ones((8,8),dtype=bool)
    semantic=subject.SemanticReplay(target_readings=(),target_matches=())
    seed_source(semantic,"s",image,usable,np.eye(3),8.)
    trusted=copy.deepcopy(semantic.trusted_ingress)
    other=image.copy(); other[3,3]=255; changed=hashlib.sha256(other.tobytes()).hexdigest()
    supplied=dict(trusted["s"],decoded_pixel_sha256=changed)
    payload=dict(frame="s",rgb=other,decoded_sha256=changed,ingress=supplied,
                 input_identity=dict(frame_id="s",height=8,width=8,sha256="a"*64))
    context,checked=reseal("frame_source",{},payload)
    clean=subject.SemanticReplay(target_readings=(),target_matches=(),trusted_ingress=trusted)
    with pytest.raises(subject.CaptureRefusal,match="trusted_ingress"): clean("frame_source",context,checked)


def test_resealed_candidate_window_substitution_rejected():
    mask=np.ones((8,8),dtype=bool); axes=np.eye(3)
    source=dict(axis_matrix=axes,focal=8.,principal_point=(4.,4.))
    window=lattice.plane_window(mask,2,axes,8.,principal_point=(4.,4.))
    subject.verify_window(mask,source,2,window)
    _context,checked=reseal("candidate_window",{},dict(window=(window[0]+.1,*window[1:])))
    with pytest.raises(subject.CaptureRefusal): subject.verify_window(mask,source,2,checked["window"])


def test_resealed_phase_energy_and_legacy_sentinel_tamper():
    weights=np.zeros((64,64),dtype=np.float32); weights[::12,:]=3.; weights[:,::12]+=2.
    evidence=lp._periodic_energy_phase_evidence(weights,12.,centre=(0.,0.),extent=1.,samples=64,source_region_id="a"*64)
    reading=dict(period_pixels=12.,phase_pixels=tuple(p if p is not None else 0. for p in evidence.phase_uv_pixels),
                 phase_evidence=lattice._period_support_snapshot(evidence))
    subject.verify_phase(reading,weights)
    reading["phase_evidence"]["fields"]["concentration_uv"]=(.7,.8)
    _context,checked=reseal("reading",{},reading)
    with pytest.raises(subject.CaptureRefusal): subject.verify_phase(checked,weights)


def test_all_retained_alternatives_required_not_one_signature_per_reading():
    from test_plane_graph import make_reading
    reading=lattice._period_support_snapshot(make_reading("s",2,1.,((1.,2.,1.),(2.,4.,.8))))
    semantic=subject.SemanticReplay(target_readings=(("s",0),),target_matches=())
    semantic.readings[("s",0)]=reading; semantic.solution={}; semantic.solver_matches=()
    semantic.core=dict(graph_sha256="a"*64,trace_sha256="b"*64)
    semantic.signature_bindings[("s",0,0,1.)]="c"*64
    with pytest.raises(subject.CaptureRefusal,match="all_retained_signature_alternatives"): semantic.finish()


@pytest.mark.parametrize("change",["emitted_signature","warp_geometry","warp_values","right_coverage"])
def test_resealed_matcher_foreign_signature_and_warp_join_tamper(change):
    readings,sigs=signatures(); mid=("s0",0,2,1.,"s1",0,2,1.,0)
    events,receive=collect()
    with scope(receive,matches=(mid,)): pg.match_plane_graph(readings,sigs)
    pair=copy.deepcopy(next(p for k,c,p in events if k=="matcher_pair"))
    warp=copy.deepcopy(next(p for k,c,p in events if k=="matcher_warp"))
    semantic=subject.SemanticReplay(target_readings=(),target_matches=(mid,))
    semantic.signatures[("s0",0,0,1.)]=lattice._period_support_snapshot(sigs[0])
    semantic.signatures[("s1",0,0,1.)]=lattice._period_support_snapshot(sigs[1])
    if change=="emitted_signature":
        v=pair["left_signature"]["fields"]["cells"].copy(); v[0,0]+=1.
        pair["left_signature"]["fields"]["cells"]=v
    elif change=="right_coverage": pair["right_coverage"]=~pair["right_coverage"]
    elif change=="warp_values": warp["warped_cells"]=warp["warped_cells"]+1.
    else:
        original=lp.WarpSignatureWitnessV1(**subject.fields(lattice._period_support_snapshot(warp["witness"]),"WarpSignatureWitnessV1"))
        changed=original.mapped_u.copy(); changed[0,0]+=.1; changed.flags.writeable=False
        import dataclasses
        replaced=dataclasses.replace(original,mapped_u=changed,witness_sha256="")
        warp["witness"]=lattice._period_support_snapshot(replaced)
        pair["metadata"]["warp_witness_sha256"]=replaced.witness_sha256
    c,p=reseal("matcher_warp",dict(match_identity=mid),warp)
    semantic("matcher_warp",c,p)
    c,p=reseal("matcher_pair",dict(match_identity=mid),pair)
    with pytest.raises((subject.CaptureRefusal,ValueError)): semantic("matcher_pair",c,p)


def test_unverified_kernel_refuses_before_claim_or_any_baseline_stage_io(monkeypatch):
    monkeypatch.setattr(subject,"load_preregistration",lambda:{"preregistration_sha256":"pure"})
    monkeypatch.setattr(subject,"claim_run",lambda:pytest.fail("one-shot consumed"))
    monkeypatch.setattr(subject,"_import_stage",lambda:pytest.fail("stage imported"))
    with pytest.raises(subject.CaptureRefusal,match="IPP_CALLS_ENFORCED") as refused:
        subject.execute_registered()
    assert refused.value.reason=="unsupported_kernel"


def test_native_descriptor_and_event_hash_budget_precedes_canonical_allocation(monkeypatch):
    descriptor=subject.array_record(np.arange(16,dtype=np.float32))["native_array"]
    event=subject.sealed_event(0,"0"*64,"synthetic","pure",{},dict(a=1))
    monkeypatch.setattr(subject,"canonical",lambda *_args:pytest.fail("hash allocated before budget"))
    with subject.memory_scope(lattice._PeriodSupportMemory(1)):
        with pytest.raises(subject.CaptureRefusal,match="aggregate_memory"): subject.array_decode(descriptor)
        with pytest.raises(subject.CaptureRefusal,match="aggregate_memory"):
            subject.validate_event(event,0,"0"*64,"synthetic")


def test_recursive_preallocation_plans_have_depth_and_node_caps():
    value=[]
    for _ in range(26): value=[value]
    with pytest.raises(lattice._PeriodSupportIncomplete): lattice._period_support_size_plan(value)
    with pytest.raises(subject.CaptureRefusal): subject.json_bound(value)
    with pytest.raises(lattice._PeriodSupportIncomplete): lattice._period_support_size_plan([0]*100001)
    with pytest.raises(subject.CaptureRefusal): subject.json_bound([0]*100001)


@pytest.mark.parametrize("failure",[False,True])
def test_genuine_lattice_signature_graph_chain_default_observer_parity(monkeypatch,failure):
    from test_image_derived_lattice_measure import _build_session,_pose_graph_for_two
    from mc_photogrammetry.pipeline import image_derived_lattice_measure as measure
    from mc_photogrammetry.pipeline import image_derived_signatures as signatures_mod
    from mc_photogrammetry.pipeline import image_derived_graph as graph
    session=_build_session(monkeypatch,frame_ids=("f0","f1")); pose=_pose_graph_for_two(session); calls=[]; actual_matches=[]
    def compact(value):
        import dataclasses
        if isinstance(value,np.ndarray): return (value.dtype.str,value.shape,hashlib.sha256(value.tobytes()).hexdigest())
        if dataclasses.is_dataclass(value):
            return (type(value).__name__,tuple((f.name,compact(getattr(value,f.name))) for f in dataclasses.fields(value)))
        if isinstance(value,dict): return tuple((k,compact(v)) for k,v in value.items())
        if isinstance(value,(tuple,list)): return tuple(compact(v) for v in value)
        return value
    for module,name in ((lattice,"rectify_plane"),(lattice,"estimate_period"),
                        (lattice,"_profile"),(lp,"plane_signature"),(pg,"match_plane_graph"),(pg,"solve_component")):
        original=getattr(module,name)
        def delegated(*args,_original=original,_name=name,**kwargs):
            calls.append((_name,compact(args),compact(kwargs)))
            returned=_original(*args,**kwargs)
            if _name=="match_plane_graph": actual_matches.append(returned)
            return returned
        monkeypatch.setattr(module,name,delegated)
    def chain():
        result=measure.measure_image_derived_lattice(session=session,pose_graph=pose)
        assert isinstance(result,measure.ImageDerivedLatticeResult)
        sig=signatures_mod.derive_image_derived_signatures(session=session,lattice_result=result,max_cells=8)
        output=graph.derive_image_derived_graph(session=session,lattice_result=result,signatures_result=sig)
        assert isinstance(output,graph.ImageDerivedGraphResult)
        return result,sig,output
    off=chain(); baseline_calls=list(calls); calls.clear(); kinds=[]
    assert actual_matches[-1]
    match=actual_matches[-1][0]
    mid=(match.left,match.left_plane,match.left_axis,match.left_b,
         match.right,match.right_plane,match.right_axis,match.right_b,match.quadrant_of_right)
    targets=((match.left,match.left_plane),(match.right,match.right_plane))
    def receive(*_args):
        kinds.append(_args[0])
        if failure: raise OSError("isolated whole-chain observer failure")
    with scope(receive,signatures=targets,matches=(mid,)) as state: on=chain()
    assert calls==baseline_calls and kinds
    assert tuple(x.result_sha256 for x in on)==tuple(x.result_sha256 for x in off)
    assert compact(on)==compact(off) and lattice._PERIOD_SUPPORT_CONTEXT.get() is None
    if failure: assert state.incomplete is not None
    else:
        assert state.incomplete is None
        assert "signature_sampling" in kinds and "matcher_pair" in kinds


@pytest.fixture(scope="module")
def structural_capture():
    from test_image_derived_lattice_measure import _build_session,_pose_graph
    from mc_photogrammetry.pipeline import image_derived_lattice_measure as measure
    patcher=pytest.MonkeyPatch()
    try:
        session=_build_session(patcher); pose=_pose_graph(session); events,receive=collect()
        with scope(receive): result=measure.measure_image_derived_lattice(session=session,pose_graph=pose)
        assert isinstance(result,measure.ImageDerivedLatticeResult)
        structural=next(p for k,c,p in events if k=="frame_structural_source")
        measured=next(p for k,c,p in events if k=="frame_measure_source")
        yield session.frames[0].image,structural,measured,events
    finally: patcher.undo()


@pytest.mark.parametrize("change",[None,"weights","raster","grayscale","camera"])
def test_sealed_structural_source_pixel_raster_and_camera_joins(structural_capture,change):
    image,structural,measured,_events=structural_capture
    structural=copy.deepcopy(structural); measured=copy.deepcopy(measured); frame=structural["frame"]
    semantic=subject.SemanticReplay(target_readings=(),target_matches=())
    ingress=dict(frame_id=frame,byte_sha256="a"*64,decoded_shape=image.shape,
        decoded_dtype=image.dtype.str,decoded_pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest())
    semantic.trusted_ingress[frame]=ingress
    semantic("frame_source",{},dict(frame=frame,rgb=image,ingress=ingress,
        input_identity=dict(frame_id=frame,height=image.shape[0],width=image.shape[1],sha256="a"*64),
        decoded_sha256=ingress["decoded_pixel_sha256"]))
    if change=="weights": structural["segment_weights"][0]+=.01
    elif change=="raster": structural["structural_weights"][0,0]+=.01
    elif change=="grayscale": structural["grey"][0,0]^=1
    elif change=="camera": measured["focal"]+=.01
    context,payload=reseal("frame_structural_source",{},structural)
    def run():
        semantic("frame_structural_source",context,payload)
        c,p=reseal("frame_measure_source",{},measured); semantic("frame_measure_source",c,p)
    if change is None: run(); assert frame in semantic.structural_sources and frame in semantic.measure_sources
    else:
        with pytest.raises(subject.CaptureRefusal): run()


@pytest.mark.parametrize("samples",[10**9,128])
def test_geometry_scalar_shape_guard_precedes_all_grid_allocations(monkeypatch,samples):
    maps=np.zeros((2,2),dtype=np.float32); valid=np.ones((2,2),dtype=bool)
    geometry=dict(samples=samples,centre=(0.,0.),extent=1.)
    for name in ("linspace","meshgrid","asarray"):
        monkeypatch.setattr(np,name,lambda *_args,**_kwargs:pytest.fail("untrusted grid allocated"))
    with pytest.raises(subject.CaptureRefusal): subject.verify_geometry(geometry,maps,maps,valid)


def small_geometry():
    n=8; u,v=np.meshgrid(np.linspace(-1.,1.,n),np.linspace(-1.,1.,n))
    return (dict(samples=n,centre=(0.,0.),extent=1.,axis=2,side=1.,axis_matrix=np.eye(3),
                 principal_point=(4.,4.),focal=4.),(4.+4.*u).astype(np.float32),
            (4.-4.*v).astype(np.float32),np.ones((n,n),dtype=bool))


def test_geometry_exact_planned_shape_budget_boundary_and_allocation_sentinel(monkeypatch):
    geometry,x,y,valid=small_geometry(); planned=640*8*8+128*8+65536
    budget=lattice._PeriodSupportMemory(planned)
    with subject.memory_scope(budget): subject.verify_geometry(geometry,x,y,valid)
    assert budget.used==planned
    for name in ("linspace","meshgrid","asarray"):
        monkeypatch.setattr(np,name,lambda *_args,**_kwargs:pytest.fail("grid allocated before reservation"))
    with subject.memory_scope(lattice._PeriodSupportMemory(planned-1)):
        with pytest.raises(subject.CaptureRefusal,match="aggregate_memory"):
            subject.verify_geometry(geometry,x,y,valid)


@pytest.mark.parametrize("path",["candidate","materialize","missing_trust"])
def test_resealed_region_shape_guard_precedes_any_decode_constructor_allocation(monkeypatch,path):
    from test_plane_graph import make_reading
    mask=np.ones((8,8),dtype=bool); polygon=((0.,0.),(7.,0.),(7.,7.),(0.,7.))
    region=lp._make_region_evidence(mask,2,polygon,("synthetic",))
    record=lattice._period_support_snapshot(region)
    if path!="missing_trust": record["fields"]["image_shape"]=(10**9,10**9)
    reading=lattice._period_support_snapshot(make_reading("s",2,1.,((1.,2.,1.),)))
    reading["fields"]["region_evidence"]=record; reading["fields"]["polygon"]=polygon
    semantic=subject.SemanticReplay(target_readings=(),target_matches=())
    semantic.source_frames["s"]=dict(shape=(8,8))
    context,payload=reseal("candidate_begin",dict(candidate_id=("s",0)),dict(axis=2,polygon=polygon,region_evidence=record))
    monkeypatch.setattr(lp,"_decode_region_evidence",lambda *_args:pytest.fail("untrusted region decoded"))
    monkeypatch.setattr(np,"zeros",lambda *_args,**_kwargs:pytest.fail("untrusted region allocated"))
    with pytest.raises(subject.CaptureRefusal):
        if path=="candidate": semantic("candidate_begin",context,payload)
        else: subject.materialize(reading,None if path=="missing_trust" else {"s":(8,8)})


def test_region_exact_planned_allocation_boundary_and_predecode_refusal(monkeypatch):
    from test_plane_graph import make_reading
    mask=np.ones((8,8),dtype=bool); polygon=((0.,0.),(7.,0.),(7.,7.),(0.,7.))
    region=lattice._period_support_snapshot(lp._make_region_evidence(mask,2,polygon,("synthetic",)))
    planned=64+12*64+12*8+65536; budget=lattice._PeriodSupportMemory(planned)
    with subject.memory_scope(budget): subject.validate_region_plan(region,(8,8))
    assert budget.used==planned
    reading=lattice._period_support_snapshot(make_reading("s",2,1.,((1.,2.,1.),)))
    reading["fields"]["region_evidence"]=region; reading["fields"]["polygon"]=polygon
    monkeypatch.setattr(lp,"_decode_region_evidence",lambda *_args:pytest.fail("region decoded before reserve"))
    with subject.memory_scope(lattice._PeriodSupportMemory(planned-1)):
        with pytest.raises(subject.CaptureRefusal,match="aggregate_memory"): subject.materialize(reading,{"s":(8,8)})


@pytest.mark.parametrize("change",[None,"integer_tolerance","min_resultant","ambiguous_margin","constraint","order","context"])
def test_effective_solver_parameters_join_resealed_solve_context(change):
    from mc_photogrammetry.pipeline import image_derived_graph as graph
    frames=("s0","s1","s2"); constraints=(("s0","s1",0),("s1","s2",1))
    parameters=dict(integer_tolerance=.1,min_resultant=.8,ambiguous_margin=.01,quadrant_constraints=constraints)
    context=graph.failed_solve_context(.1,.8,.01,constraints,frames).to_record()
    if change in ("integer_tolerance","min_resultant","ambiguous_margin"): parameters[change]+=.01
    elif change=="constraint": parameters["quadrant_constraints"]=(constraints[0],("s1","s2",2))
    elif change=="order": parameters["quadrant_constraints"]=constraints[::-1]
    elif change=="context":
        context["integer_tolerance"]+=.01
        body={k:v for k,v in context.items() if k!="record_sha256"}
        context["record_sha256"]=graph._sha(dict(domain=graph.SOLVER_SOLVE_CONTEXT_DIGEST_DOMAIN,schema=context["schema"],payload=body))
    graph.validate_solver_solve_context(context,expected_frames=frames)
    _c,p=reseal("solver_call",{},parameters)
    if change is None: subject.verify_solve_parameters(p,context,frames)
    else:
        with pytest.raises(subject.CaptureRefusal): subject.verify_solve_parameters(p,context,frames)


@pytest.mark.parametrize("change",[None,"candidate_bit","ranked_bit","threshold","max_readings","max_per_axis_side","max_rectified_side","source_threshold"])
def test_resealed_eligibility_effective_threshold_and_selection_caps(structural_capture,change):
    image,_structural,measured,events=structural_capture
    selections=[p for k,c,p in events if k=="frame_reading_selection"]
    assert selections
    selected=copy.deepcopy(selections[0]); semantic=subject.SemanticReplay(target_readings=(),target_matches=())
    frame=selected["frame"]
    semantic.source_frames[frame]=dict(shape=image.shape[:2],rgb=image)
    semantic.measure_sources[frame]=copy.deepcopy(measured)
    target=None
    for kind,context,payload in events:
        if kind!="candidate_reading": continue
        cid=tuple(context["candidate_id"]); reading=subject.fields(payload["reading"],"PlaneReading")
        semantic.candidates[cid]=dict(axis=reading["axis"],reading=payload["reading"],eligible=payload["eligible"],
            region_id=subject.fields(reading["region_evidence"],"PlaneRegionEvidence")["source_region_id"])
        if reading["axis"]!=1 and target is None: target=(context,payload)
    assert target is not None
    if change=="source_threshold":
        altered=copy.deepcopy(measured); altered["selection_policy"]["min_structural_share"]+=.01
        c,p=reseal("frame_measure_source",{},altered)
        with pytest.raises(subject.CaptureRefusal,match="effective_reading_policy"): semantic("frame_measure_source",c,p)
        return
    if change=="candidate_bit":
        altered=copy.deepcopy(target[1]); altered["eligible"]=not altered["eligible"]
        c,p=reseal("candidate_reading",target[0],altered)
        with pytest.raises(subject.CaptureRefusal,match="candidate_eligibility"): semantic("candidate_reading",c,p)
        return
    if change=="ranked_bit":
        rows=list(selected["ranked"]); cid,eligible,reading=rows[0]; rows[0]=(cid,not eligible,reading)
        selected["ranked"]=tuple(rows)
    elif change=="threshold": selected["min_structural_share"]+=.01
    elif change in ("max_readings","max_per_axis_side","max_rectified_side"): selected[change]+=1
    c,p=reseal("frame_reading_selection",{},selected)
    if change is None:
        semantic("candidate_reading",target[0],target[1]); semantic("frame_reading_selection",c,p)
        assert semantic.selected_candidates
    else:
        with pytest.raises(subject.CaptureRefusal): semantic("frame_reading_selection",c,p)


@pytest.mark.parametrize("tamper",[False,True])
def test_core_chain_binds_resealed_solve_context_before_phase_replay(monkeypatch,tamper):
    from mc_photogrammetry.pipeline import image_derived_graph as graph
    frames=("s0","s1"); parameters=dict(integer_tolerance=.1,min_resultant=.8,
        ambiguous_margin=.01,quadrant_constraints=())
    context=graph.failed_solve_context(.11 if tamper else .1,.8,.01,(),frames).to_record()
    graph.validate_solver_solve_context(context,expected_frames=frames)
    graph_sha=graph._sha({}); trace_body=dict(synthetic=True)
    trace_sha=hashlib.sha256(subject.canonical(trace_body)).hexdigest()
    observation=dict(solve_context=context,phase_capture={},solver_input_bindings={})
    observation["observation_sha256"]=hashlib.sha256(subject.canonical(observation)).hexdigest()
    core=dict(graph=dict(payload={},result_sha256=graph_sha),graph_sha256=graph_sha,
        trace=dict(trace_body,attribution_sha256=trace_sha),trace_sha256=trace_sha,
        solver_failure_observation=observation)
    solution=dict(record_type="ComponentSolution",fields=dict(frames=frames))
    class AfterContext(BaseException): pass
    monkeypatch.setattr(graph,"validate_solver_phase_capture",lambda *_args:(_ for _ in ()).throw(AfterContext()))
    if tamper:
        with pytest.raises(subject.CaptureRefusal,match="integer_tolerance"):
            subject.verify_core_chain(core,solution,(),parameters)
    else:
        with pytest.raises(AfterContext): subject.verify_core_chain(core,solution,(),parameters)


# Common-canvas consumer controls use only tiny generated signature arrays.
def _canvas_event_fixture(permuted=False):
    from dataclasses import replace
    from test_plane_graph_shifted_support import _small_pair
    left, right = _small_pair(side=5)
    quadrant, radius = 0, 16
    expected = [(r, c, r, c) for r in range(5) for c in range(5)]
    known = (0, -8)
    if permuted:
        # Floor q=1 swaps and signs the tangent axes; the left and right
        # grids differ in both dimensions. Values come from direct H sampling.
        left = replace(left, reading=replace(left.reading, axis=1),
                       cells=np.zeros((9, 11), np.float32),
                       coverage=np.zeros((9, 11), bool), interior=np.zeros((9, 11), np.float32))
        right = replace(right, reading=replace(right.reading, axis=1),
                        cells=np.ones((5, 4), np.float32), coverage=np.ones((5, 4), bool),
                        interior=np.zeros((5, 4), np.float32), origin_uv=(0., 0.))
        quadrant, radius, known = 1, 8, (3, -4)
        rotation = pg.cell_grid_homography(left, right, quadrant)[:2, :2]
        translation = np.array([8., 2.]) - rotation @ np.array([known[1], known[0]])
        right = replace(right, origin_uv=tuple(map(float, rotation.T @ translation)))
        h = pg.cell_grid_homography(left, right, quadrant)
        values = np.random.default_rng(731).uniform(1, 30, right.cells.shape).astype(np.float32)
        expected = []
        for rr in range(5):
            for rc in range(4):
                u, v, _ = h @ np.array([rc + known[1], rr + known[0], 1.])
                lr, lc = int(v), int(u)
                assert 0 <= lr < 9 and 0 <= lc < 11
                left.cells[lr, lc] = values[rr, rc]
                left.coverage[lr, lc] = True
                expected.append((lr, lc, rr, rc))
        right = replace(right, cells=values)
    mid=(left.reading.frame,0,left.reading.axis,float(left.blocks_per_period),
         right.reading.frame,0,right.reading.axis,float(right.blocks_per_period),quadrant)
    return (left, right), mid, radius, known, expected


def _canvas_capture(permuted=False):
    signatures, mid, radius, known, expected = _canvas_event_fixture(permuted)
    readings = tuple(s.reading for s in signatures)
    before = pg.match_plane_graph(readings, signatures, max_shift_cells=radius)
    stream = io.BytesIO()
    journal = subject.Journal(stream, run_id="synthetic-only")
    observer = subject.Observer(journal, run_id="synthetic-only", signature_targets=(), match_targets=(mid,))
    with scope(observer, matches=(mid,)) as state:
        after = pg.match_plane_graph(readings, signatures, max_shift_cells=radius)
    assert after == before and state.incomplete is None
    matches = [m for m in after if (m.left,m.left_plane,m.left_axis,m.left_b,
               m.right,m.right_plane,m.right_axis,m.right_b,m.quadrant_of_right)==mid]
    assert len(matches)==1 and matches[0].cell_shift_rc==known
    return signatures, mid, stream, journal, expected


def _seed_canvas_semantic(signatures, mid):
    semantic=subject.SemanticReplay(target_readings=(), target_matches=(mid,))
    for offset,signature in ((0,signatures[0]),(4,signatures[1])):
        semantic.signatures[(mid[offset],mid[offset+1],0,mid[offset+3])]=lattice._period_support_snapshot(signature)
    return semantic


@pytest.mark.parametrize("permuted", (False, True))
def test_real_common_canvas_events_observer_journal_semantic_replay(permuted):
    signatures, mid, stream, journal, expected = _canvas_capture(permuted)
    semantic = _seed_canvas_semantic(signatures, mid)
    events=[]
    def consume(kind, context, payload):
        events.append((kind, context, payload))
        if kind in ("matcher_warp", "matcher_pair"):
            semantic(kind, context, payload)
    receipt=subject.replay(stream,journal.members,run_id="synthetic-only",
                           expected_sha256=journal.sha.hexdigest(),consumer=consume)
    assert receipt["status"]=="verified"
    assert [k for k,c,p in events if k in ("matcher_warp","matcher_pair")]==["matcher_warp","matcher_pair"]
    pair=next(p for k,c,p in events if k=="matcher_pair")
    warp=next(p for k,c,p in events if k=="matcher_warp")
    assert pair["logical_pairs"].tolist()==[list(row) for row in expected]
    assert len(expected)==(20 if permuted else 25)
    assert semantic.matcher_pairs[mid]["overlap_cells"]==len(expected)
    metadata=pair["metadata"]
    witness=pg.MatcherPairWitnessV1(
        metadata["schema"],metadata["algorithm"],metadata["max_shift_cells"],metadata["quadrant"],
        pair["homography"],metadata["warp_witness_sha256"],tuple(metadata["canvas_shape"]),
        tuple(metadata["selected_shift_rc"]),metadata["selected_support"],pair["logical_pairs"],
        metadata["witness_sha256"],lp.WarpSignatureWitnessV1(**subject.fields(warp["witness"],"WarpSignatureWitnessV1")),
        right_grid_shape=tuple(metadata["right_grid_shape"]))
    assert subject.same(witness.to_record(),metadata)
    assert witness.witness_sha256==metadata["witness_sha256"]
    assert witness.right_grid_shape==signatures[1].cells.shape
    np.testing.assert_array_equal(witness.homography,pg.cell_grid_homography(*signatures,metadata["quadrant"]))


@pytest.mark.parametrize("change", ("original_grid", "padded_right_index"))
def test_resealed_common_canvas_events_refuse_original_grid_or_padded_index(change):
    signatures, mid, stream, journal, expected = _canvas_capture()
    events=[]
    subject.replay(stream,journal.members,run_id="synthetic-only",
                   expected_sha256=journal.sha.hexdigest(),consumer=lambda *event:events.append(event))
    pair=copy.deepcopy(next(p for k,c,p in events if k=="matcher_pair"))
    if change=="original_grid":
        pair["metadata"]["right_grid_shape"]=[6,5]
    else:
        # This is within the expanded warp, but outside the original 5x5 grid.
        pairs=pair["logical_pairs"].copy(); pairs[0,2]=5; pairs.flags.writeable=False
        pair["logical_pairs"]=pairs
    forged_stream=io.BytesIO(); forged=subject.Journal(forged_stream,run_id="synthetic-only")
    observer=subject.Observer(forged,run_id="synthetic-only",signature_targets=(),match_targets=(mid,))
    for kind,context,payload in events:
        observer(kind,context,pair if kind=="matcher_pair" else payload)
    semantic=_seed_canvas_semantic(signatures,mid)
    def consume(kind,context,payload):
        if kind in ("matcher_warp","matcher_pair"):
            semantic(kind,context,payload)
    with pytest.raises((subject.CaptureRefusal,ValueError)):
        subject.replay(forged_stream,forged.members,run_id="synthetic-only",
                       expected_sha256=forged.sha.hexdigest(),consumer=consume)
    assert not semantic.matcher_pairs
