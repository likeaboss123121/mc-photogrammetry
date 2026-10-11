"""Synthetic factor semantics, bounded census, source parity and atomic output."""
import copy
from dataclasses import replace
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
from mc_photogrammetry.reconstruction import region_growth as growth

SCRIPT=Path(__file__).resolve().parents[1]/"scripts/evaluation/diagnose_image_seed_factors.py"
spec=importlib.util.spec_from_file_location("seed_factor_test_subject",SCRIPT)
subject=importlib.util.module_from_spec(spec);sys.modules[spec.name]=subject;spec.loader.exec_module(subject)


def graph(supports, sides=None, dtype=np.float32):
    supports=np.asarray(supports,dtype=dtype).reshape(-1,3)
    n=len(supports)
    return SimpleNamespace(supports=supports,plane_scores=growth._normal_scores(supports),
        sides=np.ones((n,3),dtype=np.int8) if sides is None else np.asarray(sides,dtype=np.int8).reshape(n,3),
        node_area=np.ones(n),node_lab=np.zeros((n,3),dtype=np.float32),adjacency=[[] for _ in range(n)])


def baseline(graphs):
    observer=subject.c9.StreamingObserver()
    for frame,g in sorted(graphs.items()):
        observer.begin_frame(frame)
        if g is None:
            observer.expect("frame",(frame,"proposal"));observer.finish("frame",(frame,"proposal"),"no_atoms")
            for axis in range(3):
                for side in (-1,1):
                    identity=(frame,axis,side);observer.expect("seed",identity)
                    observer.finish("seed",identity,"not_reached",axis=axis,side=side,eligible_seed_count=None,reason="no_atoms")
        else:
            with growth._scalar_attrition_context(observer,frame=frame):
                for axis in range(3):
                    for side in (-1,1):growth._seed_and_grow(g,axis,side,growth.DEFAULT_REGION_GROWTH_CONFIG)
        observer.end_frame(frame)
    record=observer.seed_record()
    subject.c9.validate_seed_record(record,subject.c9._observer_record(observer),set(graphs))
    return record["rows"]


def captured(graphs,limits=None):
    rows=baseline(graphs);collector=subject.FactorCollector(sorted(graphs),limits=limits)
    for frame,g in sorted(graphs.items()):collector.capture_graph(frame,g)
    return collector,rows


def results_for(g):
    collector,rows=captured({"a":g})
    record=collector.record(rows)
    result=subject.replay(record,rows)
    return result,record,rows


def row(result,variant,axis=0,side=1):
    return next(r for r in result[variant] if r["axis"]==axis and r["side"]==side)


@pytest.mark.parametrize("dtype",[np.float32,np.float64])
def test_exact_source_dtype_baseline_all36_rows(dtype):
    graphs={f"image-{i:04d}":graph([[0,24,24],[0,48,0],[20,20,20],[1,100,2],[0,0,0]],dtype=dtype) for i in range(6)}
    collector,rows=captured(graphs)
    replay=subject.replay(collector.record(rows),rows)
    assert len(rows)==len(replay["baseline"])==36
    for source,recovered in zip(rows,replay["baseline"]):
        assert recovered["histogram"]==source["histogram"]
        assert recovered["eligible"]==source["eligible_seed_count"]
        assert recovered["newly_eligible"]==0
    for g in graphs.values():
        assert subject.scores_from_support(g.supports).tobytes()==g.plane_scores.tobytes()


def test_two_direction_balanced_plane_is_already_eligible():
    result,_,_=results_for(graph([[0,30,30]]))
    assert row(result,"baseline")["eligible"]==1
    assert all(row(result,v)["eligible"]==1 for v in subject.VARIANTS)


def test_split_tangent_directions_and_one_direction_stripes_only_balance_upper_bound():
    result,record,_=results_for(graph([[0,30,0],[0,0,30],[0,30,0]]))
    assert row(result,"baseline")["eligible"]==0
    assert row(result,"balance")["eligible"]==3
    assert row(result,"normal_suppression")["eligible"]==row(result,"reliability")["eligible"]==0
    assert len(record["frames"][0]["bins"])==2
    assert sorted(count for _,count in record["frames"][0]["bins"])==[1,2]


def test_corner_competing_scores_stay_original_in_each_intervention():
    g=graph([[20,20,20]])
    result,_,_=results_for(g)
    assert row(result,"baseline")["eligible"]==0
    assert row(result,"balance")["eligible"]==0
    assert row(result,"normal_suppression")["eligible"]==1
    assert row(result,"reliability")["eligible"]==1
    # Recomputing ALL opposing scores would preserve a zero margin and wrongly
    # erase these bounds. The experiment fixes original competing evidence.
    changed=subject.scores_from_support(g.supports,"normal_suppression")
    assert changed[0,0]-changed[0,1]==0
    assert changed[0,0]-g.plane_scores[0,1]>.035


def test_constant_total_different_balance_and_uniform_scaling():
    g=graph([[0,30,30],[0,59,1],[0,3,3]])
    assert g.supports[0].sum()==g.supports[1].sum()
    assert g.plane_scores[0,0]>g.plane_scores[1,0]
    assert g.plane_scores[0,0]>g.plane_scores[2,0]
    result,_,_=results_for(g)
    assert row(result,"baseline")["eligible"]==1
    assert row(result,"balance")["eligible"]==2
    assert row(result,"reliability")["eligible"]==1  # low support remains below12


@pytest.mark.parametrize("side",[-1,0,1])
def test_side_signs_including_zero_are_never_intervened(side):
    result,_,_=results_for(graph([[0,30,0]],sides=[[side,side,side]]))
    for selected in (-1,1):
        assert row(result,"balance",side=selected)["eligible"]==int(side==selected)
    assert all(row(result,v,side=-side if side else 1)["eligible"]==0 for v in subject.VARIANTS)


def test_score_improvement_does_not_override_support_or_margin():
    result,_,_=results_for(graph([[0,5,5],[100,100,100]]))
    # Reliability upper bound scores the first atom highly, but support10<12.
    assert row(result,"reliability")["eligible"]==1  # only the corner bound
    assert row(result,"balance")["eligible"]==0  # balanced corner stays tied
    assert row(result,"baseline")["eligible"]==0


def test_empty_reached_and_unreached_are_distinct():
    collector,rows=captured({"a":graph([]),"b":None})
    result=subject.replay(collector.record(rows),rows)
    assert result["baseline"][0]["histogram"]==[0]*16
    assert result["baseline"][-1]["histogram"] is None
    assert all(r["state"]=="not_reached" and r["newly_eligible"] is None for r in result["balance"][6:])


def test_no_arrays_or_graph_references_retained_and_permutation_invariance():
    g=graph([[0,30,30],[0,30,0],[0,30,30]])
    before=tuple(value.tobytes() for value in (g.supports,g.plane_scores,g.sides))
    refs=[weakref.ref(value) for value in (g.supports,g.plane_scores,g.sides)]
    collector,rows=captured({"a":g});record=collector.record(rows)
    assert before==tuple(value.tobytes() for value in (g.supports,g.plane_scores,g.sides))
    other,_=captured({"a":graph([[0,30,30],[0,30,30],[0,30,0]])})
    assert other.record(rows)==record
    del g
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("bound",["atoms_per_frame","atoms","unique_bins","predicate_work","aggregate_bytes","counter_max"])
def test_exact_resource_and_one_over_sticky(bound):
    g=graph([[0,30,30],[0,30,0],[0,30,30]])
    normal,rows=captured({"a":g})
    needed={"atoms_per_frame":3,"atoms":3,"unique_bins":2,"predicate_work":288,
            "aggregate_bytes":normal.resources["aggregate_bytes"],"counter_max":2}[bound]
    exact=subject.FactorCollector(["a"],limits={bound:needed});exact.capture_graph("a",g)
    assert exact.error is None
    short=subject.FactorCollector(["a"],limits={bound:needed-1});short.capture_graph("a",g)
    assert short.error is not None
    first=short.error;state=copy.deepcopy(short.resources)
    short.capture_graph("a",graph([]))
    assert short.error is first and short.resources==state
    with pytest.raises(subject.Error):short.record(rows)


def test_frame_bound_uses_bounded_iterator_and_registry_duplicate_refuses():
    visited=[]
    def frames():
        for i in range(100):visited.append(i);yield str(i)
    with pytest.raises(subject.Error):subject.FactorCollector(frames(),limits={"frames":2})
    assert visited==[0,1,2]
    with pytest.raises(subject.Error):subject.FactorCollector(["a","a"])
    with pytest.raises(TypeError):subject.CAPS["frames"]=33


@pytest.mark.parametrize("failure",["duplicate","unexpected","missing","malformed","nonfinite","negative","dtype","score_mismatch"])
def test_malformed_incomplete_and_score_replay_refuse(failure):
    g=graph([[0,30,30]]);rows=baseline({"a":g});collector=subject.FactorCollector(["a"])
    if failure=="missing":
        with pytest.raises(subject.Error,match="missing_graph"):collector.record(rows)
        return
    if failure=="malformed":g.sides=g.sides.reshape(-1)
    elif failure=="nonfinite":g.supports[0,0]=np.nan
    elif failure=="negative":g.supports[0,0]=-1
    elif failure=="dtype":g.supports=g.supports.astype(np.float16)
    elif failure=="score_mismatch":g.plane_scores[0,0]=np.nextafter(g.plane_scores[0,0],np.float32(np.inf))
    collector.capture_graph("other" if failure=="unexpected" else "a",g)
    if failure=="duplicate":collector.capture_graph("a",g)
    if failure=="score_mismatch":
        with pytest.raises(subject.Error,match="normal_score_bytes"):subject.replay(collector.record(rows),rows)
    else:
        assert collector.error is not None
        with pytest.raises(subject.Error):collector.record(rows)


def test_baseline_must_close_all_frames_before_any_counterfactual(monkeypatch):
    collector,rows=captured({"a":graph([[0,30,0]]),"b":graph([[0,30,30]])})
    rows=copy.deepcopy(rows);rows[-1]["histogram"][0]+=1
    calls=[];original=subject.scores_from_support
    def counted(support,variant="baseline"):
        calls.append(variant);return original(support,variant)
    monkeypatch.setattr(subject,"scores_from_support",counted)
    with pytest.raises(subject.Error,match="baseline_mismatch"):subject.replay(collector.record(rows),rows)
    assert calls and set(calls)=={"baseline"}


def test_capture_callback_failure_and_native_exception_restore_source(monkeypatch):
    g=graph([[0,30,30]]);result=(g,{"source":1});collector=subject.FactorCollector(["a"])
    monkeypatch.setattr(growth,"_build_graph",lambda *a,**k:result)
    original=growth._build_graph
    def raising(*a,**k):raise RuntimeError("diagnostic only")
    monkeypatch.setattr(collector,"capture_graph",raising)
    with collector.scoped_source():
        assert growth._build_graph() is result
    assert growth._build_graph is original and collector.error.reason=="factor_capture_failed"
    native=ValueError("native")
    def fail(*a,**k):raise native
    monkeypatch.setattr(growth,"_build_graph",fail)
    with pytest.raises(ValueError) as caught:
        with collector.scoped_source():growth._build_graph()
    assert caught.value is native and growth._build_graph is fail


def full_report(tmp_path,monkeypatch):
    """Explicit source hooks for record tests with an existing typed stub-G fixture."""
    import test_diagnose_image_proposal_attrition_v2 as inherited
    collector=subject.FactorCollector(["a","b"])
    def events(observer,frames=("a","b")):
        for frame in frames:
            g=graph([[0,100,100],[100,0,100],[100,100,0]])
            collector.capture_graph(frame,g);observer.begin_frame(frame)
            observer.expect("frame",(frame,"proposal"));observer.finish("frame",(frame,"proposal"),"synthetic")
            with growth._scalar_attrition_context(observer,frame=frame):
                for axis in range(3):
                    for side in (-1,1):growth._seed_and_grow(g,axis,side,growth.DEFAULT_REGION_GROWTH_CONFIG)
                    identity=(frame,axis);observer.expect("proposal",identity)
                    observer.finish("proposal",identity,"returned",axis=axis,returned_ordinal=axis)
                    observer.bind(("proposal_return",frame,axis),identity)
                    observer.expect("reading",identity);observer.finish("reading",identity,"accepted",axis=axis)
            for prefix in ("proposal_generation","proposal_returned","proposal_to_reading"):observer.check(prefix+":"+frame,True)
            observer.end_frame(frame)
    monkeypatch.setattr(inherited,"subject",subject.c9);monkeypatch.setattr(inherited,"scalar_events",events)
    outcome,_,_=inherited.run(tmp_path,monkeypatch)
    assert outcome.diagnostic_error is None
    report=subject.build_report(collector,outcome.report,outcome.g7_outcome.report,subject.source_binding())
    return report,outcome


def test_real_generated_pixel_default_driver_off_on_exact_call_digest_object_parity(tmp_path,monkeypatch):
    import test_diagnose_image_proposal_attrition_v2 as inherited
    monkeypatch.setattr(inherited,"subject",subject.c9)
    original_run=subject.c9.run_once;original_build=growth._build_graph
    collectors=[];normal_calls=[]
    original_normal=growth._normal_scores
    def normal(*args,**kwargs):normal_calls.append(1);return original_normal(*args,**kwargs)
    monkeypatch.setattr(growth,"_normal_scores",normal)
    def observed_run(*args,**kwargs):
        collector=subject.FactorCollector(["a","b"]);collectors.append(collector)
        original_capture=collector.capture_graph
        def capture(frame,g):
            if g is not None:
                arrays=(g.supports,g.plane_scores,g.sides);before=tuple(a.tobytes() for a in arrays)
                original_capture(frame,g)
                assert before==tuple(a.tobytes() for a in arrays)
                assert all(a is b for a,b in zip(arrays,(g.supports,g.plane_scores,g.sides)))
            else:original_capture(frame,g)
        monkeypatch.setattr(collector,"capture_graph",capture)
        with collector.scoped_source():return original_run(*args,**kwargs)
    monkeypatch.setattr(subject.c9,"run_once",observed_run)
    inherited.test_real_default_driver_off_on_pixel_universe_and_kernel_parity(tmp_path,monkeypatch)
    assert normal_calls==[1]*4  # two frames off/on; replay never invokes production scoring
    assert len(collectors)==1 and collectors[0].error is None
    assert all(value["state"]=="reached" for value in collectors[0].frames.values())
    assert growth._build_graph is original_build


@pytest.mark.parametrize("mutation",["count","key","order","duplicate","dtype","resource","variant","input","source","baseline","graph","selection","omission","bool"])
def test_resealed_factor_and_source_replay_mutations_refuse(mutation,tmp_path,monkeypatch):
    report,outcome=full_report(tmp_path,monkeypatch);report=copy.deepcopy(report)
    frame=report["factors"]["frames"][0]
    if mutation=="count":frame["bins"][0][1]+=1
    elif mutation=="key":
        original=frame["bins"][0][0]
        frame["bins"][0][0]=original[:-2]+"00"
        assert frame["bins"][0][0]!=original
    elif mutation=="order":frame["bins"].reverse()
    elif mutation=="duplicate":frame["bins"].append(frame["bins"][0])
    elif mutation=="dtype":frame["dtype"]="<f8"
    elif mutation=="resource":report["factors"]["resources"]["atoms"]+=1
    elif mutation=="variant":report["variants"]["balance"][0]["eligible"]+=1
    elif mutation=="input":report["inputs"][0]["sha256"]="0"*64
    elif mutation=="source":report["source_sha256"]["src/mc_photogrammetry/reconstruction/region_growth.py"]="0"*64
    elif mutation=="baseline":report["baseline_sha256"]="0"*64
    elif mutation=="graph":report["graph_sha256"]="0"*64
    elif mutation=="selection":report["selection_sha256"]="0"*64
    elif mutation=="omission":del report["variants"]["baseline"]
    else:frame["bins"][0][1]=True
    with pytest.raises((subject.Error,ValueError)):
        subject.validate_report(subject._seal(report),outcome.report,outcome.g7_outcome.report)


def test_source_drift_refuses_before_any_delivery(tmp_path,monkeypatch):
    original=subject._hash
    monkeypatch.setattr(subject,"_hash",lambda p:"0"*64 if str(p).endswith('region_growth.py') else original(p))
    called=[]
    with pytest.raises(subject.Error,match="source_pin_mismatch"):
        subject.run_once([tmp_path/'not_read.png'],project=tmp_path/'out',destination=tmp_path/'dest',diagnostic=tmp_path/'g7',sidecar=tmp_path/'seed',factors=tmp_path/'factors',name='test',description='test',delivery=lambda *a,**k:called.append(1))
    assert called==[]


def test_exact_historical_c9_guard_and_valid_number_seed_mutation():
    root=SCRIPT.parents[2]/'exports/diagnostics/render-loop-pilot-2026-10-08'
    baseline_record=json_load(root/'train-station-phase-c9.seed-gate.json')
    graph_record=json_load(root/'train-station-phase-c9.g7-diagnostic.json')
    subject.require_c9_reference(baseline_record,graph_record)
    changed=copy.deepcopy(baseline_record);changed['seed_attribution']['rows'][0]['histogram'][0]+=1
    with pytest.raises(subject.Error,match='reference_mismatch'):subject.require_c9_reference(changed,graph_record)
    changed=copy.deepcopy(graph_record);changed['graph']['counts']['readings']+=1
    with pytest.raises(subject.Error,match='reference_mismatch'):subject.require_c9_reference(baseline_record,changed)


def json_load(path):
    import json
    assert path.stat().st_size<=subject.CAPS['sidecar_bytes']
    return json.loads(path.read_text())


@pytest.mark.parametrize('failure',['short_write','fsync','link','race','temporary_readback','final_readback'])
def test_atomic_publication_failure_preserves_original_outputs(failure,tmp_path,monkeypatch):
    report,outcome=full_report(tmp_path,monkeypatch)
    target=tmp_path/'factors.json';before={p:p.read_bytes() for p in (tmp_path/'out.mcscan',tmp_path/'g7.json',tmp_path/'attrition.json')}
    if failure=='short_write':
        original=subject.os.fdopen
        class ShortWriter:
            def __init__(self,*args):self.stream=original(*args)
            def __enter__(self):return self
            def __exit__(self,*args):self.stream.close()
            def write(self,data):return self.stream.write(data[:-1])
            def flush(self):self.stream.flush()
            def fileno(self):return self.stream.fileno()
        monkeypatch.setattr(subject.os,'fdopen',ShortWriter)
    elif failure=='fsync':monkeypatch.setattr(subject.os,'fsync',lambda _:(_ for _ in ()).throw(OSError('fsync')))
    elif failure=='link':monkeypatch.setattr(subject.os,'link',lambda *a:(_ for _ in ()).throw(OSError('unsupported')))
    elif failure=='race':
        def race(*args):target.write_bytes(b'foreign racer');raise FileExistsError('race')
        monkeypatch.setattr(subject.os,'link',race)
    else:
        original=subject._readback
        def fail(path,*args):
            if (Path(path)==target)==(failure=='final_readback'):raise subject.Error('readback_mismatch',failure)
            return original(path,*args)
        monkeypatch.setattr(subject,'_readback',fail)
    with pytest.raises((subject.Error,OSError)):
        subject.publish_report(target,report,outcome.report,outcome.g7_outcome.report)
    assert all(p.read_bytes()==raw for p,raw in before.items())
    assert target.read_bytes()==b'foreign racer' if failure=='race' else not target.exists()
    assert not list(tmp_path.glob('.seed-factors-*.tmp'))


def test_success_fsync_before_visible_exact_readback_and_collision(tmp_path,monkeypatch):
    report,outcome=full_report(tmp_path,monkeypatch);target=tmp_path/'factors.json'
    original=subject.os.fsync;called=[]
    def fsync(fd):assert not target.exists();called.append(1);return original(fd)
    monkeypatch.setattr(subject.os,'fsync',fsync)
    subject.publish_report(target,report,outcome.report,outcome.g7_outcome.report)
    assert called==[1]
    assert subject._readback(target,outcome.report,outcome.g7_outcome.report)==subject.c9.canonical_bytes(report)
    with pytest.raises(subject.Error,match='output_collision'):subject.publish_report(target,report,outcome.report,outcome.g7_outcome.report)


def test_final_byte_boundary_precharged_before_encoder(monkeypatch):
    value='x'*(subject.CAPS['sidecar_bytes']-2)
    assert len(subject.c9.canonical_bytes(value))==subject.CAPS['sidecar_bytes']
    called=[];monkeypatch.setattr(subject.c9.json,'dumps',lambda *a,**k:called.append(1))
    with pytest.raises(subject.Error):subject.c9.canonical_bytes(value+'x')
    assert called==[]


def test_every_single_factor_row_preserves_support_and_side_joint_marginal():
    result,_,_=results_for(graph([[0,30,0],[0,5,5],[20,20,20],[0,30,30]],sides=[[1,0,-1],[-1,1,0],[0,-1,1],[1,1,1]]))
    for index,base in enumerate(result["baseline"]):
        for variant in subject.VARIANTS[1:]:
            changed=result[variant][index]
            assert (changed["frame"],changed["axis"],changed["side"],changed["denominator"])==(base["frame"],base["axis"],base["side"],base["denominator"])
            for mask in (0,4,8,12):
                assert sum(n for i,n in enumerate(base["histogram"]) if i&12==mask)==sum(n for i,n in enumerate(changed["histogram"]) if i&12==mask)


def test_future_reference_rejects_unknown_ordered_inputs_before_delivery(tmp_path,monkeypatch):
    source=tmp_path/"synthetic-source.png";source.write_bytes(b"no decoding")
    called=[]
    with pytest.raises(subject.Error,match="ordered_inputs"):
        subject.run_once([source],project=tmp_path/"project",destination=tmp_path/"dest",diagnostic=tmp_path/"g7",sidecar=tmp_path/"seed",factors=tmp_path/"factors",name="synthetic",description="synthetic",delivery=lambda *a,**k:called.append(1))
    assert called==[] and source.read_bytes()==b"no decoding"


def test_renderer_overrun_refuses_before_temp_creation_and_preserves_outputs(tmp_path,monkeypatch):
    report,outcome=full_report(tmp_path,monkeypatch)
    target=tmp_path/"factors.json"
    before={p:p.read_bytes() for p in (tmp_path/"out.mcscan",tmp_path/"g7.json",tmp_path/"attrition.json")}
    original=subject.c9.canonical_bytes;created=[]
    def renderer(value,*args,**kwargs):
        # Structural validation/replay still uses the actual accepted encoder.
        # Simulate only the final renderer unexpectedly returning oversize data.
        if value is report:return b"x"*(subject.CAPS["sidecar_bytes"]+1)
        return original(value,*args,**kwargs)
    monkeypatch.setattr(subject.c9,"canonical_bytes",renderer)
    monkeypatch.setattr(subject.tempfile,"mkstemp",lambda *a,**k:created.append(1))
    with pytest.raises(subject.Error,match="factor_budget: sidecar_bytes"):
        subject.publish_report(target,report,outcome.report,outcome.g7_outcome.report)
    assert created==[] and not target.exists()
    assert all(p.read_bytes()==raw for p,raw in before.items())
    assert not list(tmp_path.glob(".seed-factors-*.tmp"))
