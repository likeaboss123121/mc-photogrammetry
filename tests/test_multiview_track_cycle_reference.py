"""Guard components only. No projected32-row packet or main A/B invocation."""
import ast
from dataclasses import replace
import importlib.util
import inspect
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'scripts/evaluation/multiview_track_cycle_reference.py'
spec=importlib.util.spec_from_file_location('mv_track_guards',PATH)
f=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=f
spec.loader.exec_module(f)


@pytest.fixture(scope='module')
def runtime():
    result=f.load_runtime()  # source imports/identity checks; no numerical solver
    print('GUARD_ENVIRONMENT '+f.canonical(result.identity).decode())
    return result


def tiny_graph(runtime,n=4):
    np=runtime.np
    d=f.readonly(np,np.full((n,128),32,dtype=np.float32),np.float32)
    graph=runtime.correspondence.ReciprocalGraph(
        f.readonly(np,[(i,i) for i in range(n)],np.int64),
        f.readonly(np,np.ones(n),np.float64),n,n,n,0)
    return graph,d


def test_source_bootstrap_no_forbidden_modules(runtime):
    runtime.attest()
    assert f.source_pins()==f.PINS
    assert 'cv2' not in sys.modules
    assert not any(name.endswith('run_camera_mixed_depth') for name in sys.modules)
    assert not f._CONSUMED


def test_inverse_ast_exact_one_expr_and_original_A(runtime):
    source=(ROOT/next(iter(f.PINS))).read_text(encoding='utf-8')
    original,changed,index=f.inserted_class_modes(source)
    assert len(changed.body)==len(original.body)+1
    assert ast.unparse(changed.body[index])== \
        '_evaluation_append_seed(candidates, rows, norms, graph, bounds, support_floor)'
    del changed.body[index]
    assert ast.dump(changed,include_attributes=True)==ast.dump(original,include_attributes=True)
    joint,ds11,corr=f.source_namespaces(runtime)
    assert joint['_class_modes'].__code__ is runtime.joint._class_modes.__code__
    assert joint['match_feature_frames'].__code__ is runtime.joint.match_feature_frames.__code__
    assert joint['solve_pair'] is ds11['solve_pair']
    assert ds11['solve_pair'].__code__ is runtime.ds11.solve_pair.__code__
    assert joint['select_mutual_matches'] is corr['select_mutual_matches']
    assert joint['np'] is ds11['np'] is corr['np'] is runtime.np
    assert joint['_null_vector'] is not ds11['_null_vector']
    assert joint['_null_vector'].__globals__ is joint
    assert ds11['_null_vector'].__globals__ is ds11
    for name in ('_select_and_refit','_greedy_selection','_winner_refusal','_score_modes'):
        assert joint[name].__code__ is getattr(runtime.joint,name).__code__
        assert joint[name].__defaults__==getattr(runtime.joint,name).__defaults__
        assert joint[name].__annotations__==getattr(runtime.joint,name).__annotations__


def test_closed_triangles_open_chain_and_conflict():
    closed=f.closed_triangles(((0,0),(1,1)),((0,0),(1,1)),((0,0),(1,1)),(2,2,2))
    assert closed.triangles==((0,0,0,0),(1,1,1,1))
    assert closed.eligible_indices==(0,1) and not closed.quarantined
    open_chain=f.closed_triangles(((0,0),),(),((0,0),),(1,1,1))
    assert not open_chain.triangles and not open_chain.eligible_indices
    conflict=f.closed_triangles(((0,0),(0,1)),((0,0),),((0,0),(1,0)),(1,2,1))
    assert conflict.eligible_indices==()
    assert conflict.quarantined==(((0,0),(1,0),(1,1),(2,0)),)
    two_c=f.closed_triangles(((0,0),),((0,0),(0,1)),((0,0),(0,1)),(1,1,2))
    assert not two_c.eligible_indices


def test_permutation_control_preserves_wrong_closures_not_oracle_labels():
    # Index-only control; no FrameFeatures/camera/ray packet construction.
    ab=tuple(sorted(tuple((i,(i+d)%32) for i in range(32) for d in (1,5,9,13))+
                    tuple((i,i) for i in range(16))))
    ac=tuple((i,(i+1)%32) for i in range(32))
    bc=tuple((i,i) for i in range(32))
    result=f.closed_triangles(ab,ac,bc,(32,32,32))
    assert len(result.triangles)==len(result.eligible_indices)==32
    assert all(a!=b and b==c for a,b,c,_ in result.triangles)


@pytest.mark.parametrize('ab,counts', [(((0,0),(0,0)),(1,1,1)),
                                      (((1,0),),(1,1,1)),
                                      (((0,0),),(33,1,1)),
                                      (((0,1),(0,0)),(1,2,1)),
                                      (tuple((0,i) for i in range(6)),(1,6,1))])
def test_invalid_or_over_cap_graph(ab,counts):
    with pytest.raises(f.IntegrityError):
        f.closed_triangles(ab,(),(),counts)


def test_source_graph_binding_and_readonly_owner(runtime):
    graph,d=tiny_graph(runtime)
    driver=f.Driver(runtime,'B',graph,d,d,(0,1),f.Meter())
    driver.attest()
    assert driver.graph_provider(d,d,neighbors=5,max_edges=8192,distance_ceiling=None) is graph
    with pytest.raises(f.IntegrityError,match='graph_binding'):
        driver.graph_provider(d,d,neighbors=5,max_edges=8192,distance_ceiling=None)
    driver2=f.Driver(runtime,'A',graph,d,d,(),f.Meter())
    with pytest.raises(f.IntegrityError,match='graph_binding'):
        driver2.graph_provider(d.copy(),d,neighbors=5,max_edges=8192,distance_ceiling=None)
    for invalid in (replace(graph,truncated=1),replace(graph,total_edges=99),
                    replace(graph,left_keypoints=5),replace(graph,edges=graph.edges.copy())):
        with pytest.raises(f.IntegrityError):
            f.Driver(runtime,'A',invalid,d,d,(),f.Meter())
    with pytest.raises(ValueError):
        d.setflags(write=True)


def test_namespace_tamper_detected_before_main(runtime):
    graph,d=tiny_graph(runtime)
    driver=f.Driver(runtime,'B',graph,d,d,(0,1),f.Meter())
    driver.joint['_winner_refusal']=lambda *a:''
    with pytest.raises(f.IntegrityError,match='namespace_binding_tamper'):
        driver.attest()
    driver=f.Driver(runtime,'B',graph,d,d,(0,1),f.Meter())
    driver.joint['_evaluation_append_seed'].indices=(0,)
    with pytest.raises(f.IntegrityError,match='seed_callback_binding'):
        driver.attest()


def test_default_and_constant_tamper_in_fake_namespace(runtime):
    # Never mutate a real source module; copied namespace exercises verifier.
    fake=ModuleType(runtime.joint.__name__)
    fake.__dict__.update(runtime.joint.__dict__)
    for name,fn in runtime.joint.__dict__.items():
        if inspect.isfunction(fn) and fn.__module__==runtime.joint.__name__:
            fake.__dict__[name]=f.clone(fn,fake.__dict__)
    fake._class_modes.__defaults__=(0,)
    with pytest.raises(f.IntegrityError,match='function_contract'):
        f.verify_module(fake,next(iter(f.PINS)))
    fake._class_modes.__defaults__=None
    fake._YAW_CLASSES=(3,2,1,0)
    with pytest.raises(f.IntegrityError,match='source_constant'):
        f.verify_module(fake,next(iter(f.PINS)))


def guarded_class(runtime,arm,seed):
    """4-edge stub component ONLY, no main matcher or actual SVD."""
    np=runtime.np
    graph,d=tiny_graph(runtime)
    meter=f.Meter()
    driver=f.Driver(runtime,arm,graph,d,d,(0,1) if seed else (),meter)
    vectors=tuple(np.eye(3)[i].copy() for i in range(3))
    new=np.array([1.,1.,1.])/np.sqrt(3.)
    state={'samples':0}
    def rank_stub(rows,threshold): return True
    def null_stub(rows):
        if len(rows)==4: return vectors[0].copy()
        state['samples']+=1
        return vectors[1+(state['samples']-1)%2].copy()
    def residual_stub(rows,norms,candidate):
        count=4 if np.allclose(candidate,new) else (3 if candidate[0]==1 else 2 if candidate[1]==1 else 1)
        return np.array([.2]*count+[90.]*(4-count))
    def refit_stub(rows,norms,graph,candidate,bounds):
        return candidate,(residual_stub(rows,norms,candidate)<=1.)
    driver.joint['_sample_rank_ok']=driver._delegate(rank_stub,'joint','_sample_rank_ok')
    driver.joint['_null_vector']=driver._delegate(null_stub,'joint','_null_vector')
    driver.joint['_residuals']=driver._delegate(residual_stub,'joint','_residuals')
    driver.joint['_select_and_refit']=driver._refit_delegate(refit_stub)
    if arm=='B':
        callback=driver.joint['_evaluation_append_seed']
        callback.rank_fit=driver._delegate(rank_stub,'joint','_sample_rank_ok')
        callback.null_fit=driver._delegate(lambda rows:new.copy(),'joint','_null_vector')
    # Stubbed namespace is deliberately not eligible for Driver.run; only this
    # generic source-class component with floor1 is being checked.
    rows=np.ones((4,3)); norms=np.ones(4)
    assert sys.gettrace() is None
    sys.settrace(driver.trace)
    try:
        modes=driver.joint['_class_modes'](rows,norms,graph,runtime.joint.JointMatchSettings(),1)
    finally:
        sys.settrace(None)
    assert len(driver.classes)==1
    return driver,driver.classes[0],modes


def test_empty_seed_source_candidate_and_RNG_equivalence(runtime):
    a,ta,_=guarded_class(runtime,'A',False)
    b,tb,_=guarded_class(runtime,'B',False)
    assert ta['prefix']==tb['prefix']
    assert ta['generation_calls']==tb['generation_calls']
    assert ta['RNG_state']==tb['RNG_state']
    assert ta['accepted']==tb['accepted']==512 and ta['attempts']==tb['attempts']==512
    assert ta['scored']==tb['scored'] and ta['kept_orders']==tb['kept_orders']
    assert tb['seed']['reason']=='no_unambiguous_rows' and tb['seed']['order'] is None
    assert ta['mode_origins']==tb['mode_origins']
    assert a.meter.transient_bytes==b.meter.transient_bytes==0


def test_seed_before_scoring_displaces_original_cap_mode_with_lineage(runtime):
    _,a,_=guarded_class(runtime,'A',False)
    _,b,modes=guarded_class(runtime,'B',True)
    assert a['prefix']==b['prefix'] and a['RNG_state']==b['RNG_state']
    assert len(b['candidates'])==len(a['candidates'])+1
    assert b['seed']['order']==513 and b['scored'][0][2]==513
    assert len(a['kept_orders'])==len(b['kept_orders'])==len(modes)==3
    assert a['kept_orders']==(0,1,2)
    assert b['kept_orders']==(513,0,1)
    assert b['mode_origins']==[513,0,1]
    assert b['refits'][0]['candidate_order']==513


def test_per_candidate_residuals_retain_digests_only(runtime):
    driver,context,_=guarded_class(runtime,'B',True)
    observations=context['residual_observations']
    assert len(observations)==517  #514 scoring passes +3 returned-mode passes
    assert all(set(x)=={'dtype','shape','sha256'} and x['dtype']=='<f8' and
               x['shape']==(4,) and len(x['sha256'])==64 for x in observations)
    assert all('data' not in x and not any(isinstance(v,runtime.np.ndarray) for v in x.values())
               for x in observations)
    # The small proposal vectors still identify exact candidate/seed lineage.
    assert len(context['candidates'])==514
    assert all(x['shape']==(3,) and len(x['data'])==24 for x in context['candidates'])
    assert '_generated_objects' not in context and '_candidates_live' not in context
    assert all('_result' not in x for x in context['refits'])
    assert driver.current is None and driver.meter.transient_bytes==0


def test_digest_precharge_before_copy_and_returned_bytes_not_retained(runtime):
    class CopyForbidden(runtime.np.ndarray):
        def tobytes(self,*args,**kwargs): pytest.fail('copy before digest precharge')
    values=runtime.np.arange(4,dtype=runtime.np.float64)
    meter=f.Meter(evidence_limit=255)
    with pytest.raises(f.IntegrityError,match='metadata_precharge'):
        meter.digest(runtime.np,values.view(CopyForbidden))
    assert meter.evidence_bytes==0
    meter=f.Meter()
    receipt=meter.digest(runtime.np,values)
    assert receipt=={'dtype':'<f8','shape':(4,),'sha256':f.sha(values.tobytes())}
    assert meter.evidence_bytes==256 and 'data' not in receipt


def test_class_output_precharge_before_source_work(runtime):
    graph,d=tiny_graph(runtime)
    meter=f.Meter(output_limit=f.CLASS_OUTPUT_RESERVATION-1)
    driver=f.Driver(runtime,'B',graph,d,d,(),meter)
    frame=SimpleNamespace(f_code=driver._class_code,
                          f_locals={'rows':runtime.np.ones((4,3)),'norms':runtime.np.ones(4)})
    with pytest.raises(f.IntegrityError,match='output_precharge'):
        driver.trace(frame,'call',None)
    assert meter.output_reserved==0 and meter.svd_calls==meter.residual_rows==0
    assert driver.current is None and not driver.classes


def test_output_static_reservation_covers_source_pinned_receipt_schema():
    per_class=4097*128+514*(384+192+192+128)+517*256+3*2048+8192
    assert per_class<f.CLASS_OUTPUT_RESERVATION
    assert f.MAIN_OUTPUT_RESERVATION==8*f.CLASS_OUTPUT_RESERVATION+\
        2*f.ARM_OUTPUT_RESERVATION+f.ENVELOPE_OUTPUT_RESERVATION<f.MAX_EVIDENCE
    meter=f.Meter()
    meter.reserve_output(f.ENVELOPE_OUTPUT_RESERVATION)
    for _ in range(8): meter.reserve_output(f.CLASS_OUTPUT_RESERVATION)
    for _ in range(2): meter.reserve_output(f.ARM_OUTPUT_RESERVATION)
    assert meter.output_reserved==f.MAIN_OUTPUT_RESERVATION
    planned=f.Meter()
    planned.preallocate_output(f.MAIN_OUTPUT_RESERVATION)
    assert planned.output_reserved==f.MAIN_OUTPUT_RESERVATION and planned.output_claimed==0
    planned.claim_output(f.ENVELOPE_OUTPUT_RESERVATION)
    for _ in range(8): planned.claim_output(f.CLASS_OUTPUT_RESERVATION)
    for _ in range(2): planned.claim_output(f.ARM_OUTPUT_RESERVATION)
    assert planned.output_claimed==planned.output_reserved==f.MAIN_OUTPUT_RESERVATION
    with pytest.raises(f.IntegrityError,match='output_claim'):
        planned.claim_output(1)
    with pytest.raises(f.IntegrityError,match='output_plan_repeated'):
        planned.preallocate_output(1)
    with pytest.raises(f.IntegrityError,match='output_precharge'):
        f.Meter(output_limit=f.MAIN_OUTPUT_RESERVATION-1).preallocate_output(f.MAIN_OUTPUT_RESERVATION)


def test_wire_component_bounds_use_full_hex_width_and_longest_scalar_repr():
    # Schema-only receipts: no projected coordinates/packet/source numeric call.
    wide=1.7976931348623157e308
    key=('<f8',(3,),(8,),True,'f'*64)
    vector={'dtype':'<f8','shape':(3,),'data':bytes(24),'sha256':'f'*64}
    digest={'dtype':'<f8','shape':(192,),'sha256':'f'*64}
    selection={'dtype':'|b1','shape':(192,),'data':bytes(192),'sha256':'f'*64}
    assert f.wire_size(('_sample_rank_ok',(191,191),False))<=128
    assert f.wire_size(key)<=192 and f.wire_size(vector)<=384
    assert f.wire_size((192,wide,513))<=128 and f.wire_size(digest)<=256
    assert f.wire_size({'candidate_order':513,'returned':True,'t':vector,'selection':selection})<=2048
    # Deliberately padded strings exceed any finite float/3-vector/scalar reason
    # in pinned source diagnostics;20 fields also exceed the actual15/11 fields.
    oversized_score={'field_'+str(i):'x'*128 for i in range(20)}
    assert f.wire_size(oversized_score)<=20*192
    arm_bound=2*12*(20*192)+4*(20*192)+12*512+16384
    assert arm_bound<f.ARM_OUTPUT_RESERVATION


def test_output_preflight_rejects_before_hex_tokens_or_first_write(monkeypatch):
    stream=io.BytesIO()
    report={'raw':bytes(8),'context':['tiny',True,None]}
    monkeypatch.setattr(f,'wire_tokens',lambda value:pytest.fail('tokens before complete preflight'))
    with pytest.raises(f.IntegrityError,match='stdout_evidence_preflight'):
        f.emit_report(report,stream,limit=f.wire_size(report))  # newline also charged
    assert stream.getvalue()==b''


def test_output_exact_size_preflight_and_stream_agree():
    report={'quote':'"\\\n\u2603','nested':[{'blob':b'\x00\xff'},(),0,-1,1.25,False,None]}
    stream=io.BytesIO()
    expected={'quote':report['quote'],'nested':[{'blob':{'hex':'00ff'}},[],0,-1,1.25,False,None]}
    size=f.wire_size(report)+1
    assert f.emit_report(report,stream,limit=size)==size
    assert len(stream.getvalue())==size
    assert json.loads(stream.getvalue())==expected


def test_seed_rank_none_and_mutation_guard(runtime):
    np=runtime.np
    graph,_=tiny_graph(runtime)
    rows=np.ones((4,3)); norms=np.ones(4); candidate=np.array([1.,0.,0.])
    calls=[]
    append=f.SeedAppender(np,(0,1),lambda *a:False,lambda *a:pytest.fail('null after rank refusal'),
                          f.Meter(),lambda *a:calls.append(a))
    candidates=[candidate]
    append(candidates,rows,norms,graph,runtime.joint.JointMatchSettings(),1)
    assert candidates==[candidate] and calls[0][2]=='rank_deficient'
    def mutate(subset,threshold):
        candidate[0]=2.; return False
    append=f.SeedAppender(np,(0,1),mutate,lambda *a:None,f.Meter(),lambda *a:None)
    with pytest.raises(f.IntegrityError,match='prefix_mutation'):
        append(candidates,rows,norms,graph,runtime.joint.JointMatchSettings(),1)


def test_transparent_delegate_arg_layout_and_result_identity(runtime):
    graph,d=tiny_graph(runtime)
    driver=f.Driver(runtime,'A',graph,d,d,(),f.Meter())
    np=runtime.np
    array=np.ones((2,3)); token=np.array([1.,0.,0.]); calls=[]
    def stub(value):
        calls.append(value); return token
    wrapped=driver._delegate(stub,'joint','_null_vector')
    key=f.array_key(array)
    assert wrapped(array) is token and calls==[array] and f.array_key(array)==key
    assert driver.meter.svd_calls==1 and driver.meter.transient_bytes==0


def test_numeric_precharge_precedes_delegate(runtime):
    graph,d=tiny_graph(runtime)
    for name,meter,args in (
        ('_null_vector',f.Meter(svd_limit=0),(runtime.np.ones((2,3)),)),
        ('_residuals',f.Meter(residual_limit=0),(runtime.np.ones((2,3)),runtime.np.ones(2),runtime.np.ones(3)))):
        driver=f.Driver(runtime,'A',graph,d,d,(),meter)
        fn=driver._delegate(lambda *a:pytest.fail('delegate before failed precharge'),'joint',name)
        with pytest.raises(f.IntegrityError,match='precharge'):
            fn(*args)
    meter=f.Meter(evidence_limit=1)
    with pytest.raises(f.IntegrityError,match='evidence_precharge'):
        meter.capture(runtime.np,runtime.np.ones(2))
    with pytest.raises(f.IntegrityError,match='metadata_precharge'):
        meter.metadata(2)
    clock=lambda:31.
    with pytest.raises(f.IntegrityError,match='watchdog'):
        f.Meter(started=0.,clock=clock).residual(1,'guard')


def test_work_accounting_and_scratch_outputs():
    b=f.work_bounds()
    assert b['residual_A']==4*513*192+4*3*11*192+12*192+4*(513+10+1)*32
    assert b['residual_B']==4*514*192+4*3*11*192+12*192+4*(513+10+1)*32
    assert b['residual_AB']==978176<1_000_000
    assert b['svd_AB']==b['svd_joint_A']+b['svd_joint_B']+2*b['svd_DS11']==37240<40000
    assert b['full_U_max']==294912 and b['DS11_U_max']==8192


@pytest.mark.parametrize('changed,margin,expected', [
    ({'support':7},None,'insufficient_one_to_one_support'),
    ({'spatial_bins_left':3},None,'low_spatial_spread'),
    ({'singular_ratio':0.0009},None,'rank_deficient'),
    ({'parallax':2.99},None,'low_parallax'),
    ({'cheirality':0.749},None,'cheirality_failed'),
    ({},1,'ambiguous_hypothesis'),({},2,''),({},None,'')])
def test_original_joint_refusal_gates(runtime,changed,margin,expected):
    candidate=SimpleNamespace(support=8,spatial_bins_left=4,spatial_bins_right=4,
                              singular_ratio=.01,parallax=4.,cheirality=1.)
    candidate.__dict__.update(changed)
    assert runtime.joint._winner_refusal(candidate,[],8,margin,runtime.joint.JointMatchSettings())==expected


@pytest.mark.parametrize('changed,parallax,cheirality,expected', [
    ({'row_max':0.},4.,1.,'pure_rotation'),
    ({'t':None},4.,1.,'rank_deficient'),
    ({'singular_ratio':.0009},4.,1.,'rank_deficient'),
    ({},2.99,1.,'pure_rotation'),
    ({'inlier_count':7},4.,1.,'insufficient_inliers'),
    ({'total':20},4.,1.,'low_inlier_fraction'),
    ({'median_rho_degrees':1.01},4.,1.,'high_residual'),
    ({},4.,.749,'cheirality_failed'),({},4.,1.,'')])
def test_original_DS11_class_gates(runtime,changed,parallax,cheirality,expected):
    fit=SimpleNamespace(row_max=1.,informative_count=8,t=(1.,0.,0.),singular_ratio=.01,
                        inlier_count=8,total=8,median_rho_degrees=.1)
    fit.__dict__.update(changed)
    assert runtime.ds11._class_refusal(fit,parallax,cheirality,runtime.ds11.PairGates())==expected


def test_source_public_class_agreement_not_transformed(runtime):
    tree=ast.parse((ROOT/next(iter(f.PINS))).read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='match_feature_frames')
    assert any(isinstance(n,ast.Compare) and ast.unparse(n)=='pose.right_yaw_class != best.yaw_class'
               for n in ast.walk(fn))
    assert runtime.joint.solve_pair is runtime.ds11.solve_pair


def test_freeze_schema_and_guard_scope():
    tree=ast.parse(Path(__file__).read_text(encoding='utf-8'))
    calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and
           isinstance(n.func,ast.Attribute)]
    assert not set(calls)&{'execute_once','build_fixed_packet','match_feature_frames','solve_pair','run'}
    assert not f._CONSUMED
    assert f.PACKET_SPEC['projection']=='u=cx+fx*X/Z;v=cy-fy*Y/Z'
    assert f.PACKET_SPEC['result_scope']=='TRANSIENT_PLUS_COMMS'
    fields=set(f.RunFreeze.__dataclass_fields__)
    assert {'harness_sha256','guard_tests_sha256','source_hashes','environment_sha256',
            'specification_sha256','command_sha256','external_go_sha256'}==fields


def freeze_bootstrap_stub():
    """Source bytes/stdlib only; no real runtime, packet, numeric import or GO."""
    events=[]
    runtime=SimpleNamespace(identity={'guard_stub':True},attest=lambda:events.append('attest'))
    command=('pure-guard-command',)
    record={'harness_sha256':f.sha(PATH.read_bytes()),
            'guard_tests_sha256':f.sha(Path(__file__).read_bytes()),'source_hashes':f.PINS.copy(),
            'environment_sha256':f.sha(f.canonical(runtime.identity)),
            'specification_sha256':f.sha(f.canonical(f.PACKET_SPEC)),
            'command_sha256':f.sha(f.canonical(command)),'external_go_sha256':'0'*64}
    return runtime,command,record,events


def test_freeze_bootstrap_strict_stdin_and_limit():
    _,_,record,_=freeze_bootstrap_stub()
    raw=f.canonical(record)
    class Bounded(io.BytesIO):
        def read(self,n):
            assert n==f.FREEZE_LIMIT+1
            return super().read(n)
    result=f.read_freeze(Bounded(raw))
    assert result==f.RunFreeze.from_mapping(record)
    assert 'packet_sha256' not in result.__dataclass_fields__
    assert f.read_freeze(Bounded(raw+b' '*(f.FREEZE_LIMIT-len(raw))))==result
    with pytest.raises(f.IntegrityError,match='freeze_record_cap'):
        f.read_freeze(Bounded(raw+b' '*(f.FREEZE_LIMIT+1-len(raw))))


@pytest.mark.parametrize('body,reason',[
    (b'', 'freeze_malformed'),(b'{}{}','freeze_malformed'),(b'{} trailing','freeze_malformed'),
    (b'\xff','freeze_malformed'),(b'[]','freeze_schema'),(b'null','freeze_schema'),
    (b'{"x":NaN}','freeze_nonfinite'),(b'{"x":Infinity}','freeze_nonfinite'),
    (b'{"x":-Infinity}','freeze_nonfinite'),(b'{"x":1e999}','freeze_nonfinite'),
    (b'{"x":1,"x":2}','freeze_duplicate_key'),
    (b'{"source_hashes":{"x":1,"x":2}}','freeze_duplicate_key')])
def test_freeze_bootstrap_invalid_json(body,reason):
    with pytest.raises(f.IntegrityError,match=reason): f.read_freeze(io.BytesIO(body))


@pytest.mark.parametrize('mutation', ['obsolete','unknown','missing','digest_type','upper_digest','sources'])
def test_freeze_bootstrap_rejects_schema_and_obsolete_expected_hash(mutation):
    _,_,record,_=freeze_bootstrap_stub()
    if mutation=='obsolete': record['packet_sha256']='0'*64
    elif mutation=='unknown': record['unknown']='0'*64
    elif mutation=='missing': del record['external_go_sha256']
    elif mutation=='digest_type': record['harness_sha256']=False
    elif mutation=='upper_digest': record['harness_sha256']='A'*64
    else: record['source_hashes']={}
    with pytest.raises(f.IntegrityError): f.read_freeze(io.BytesIO(f.canonical(record)))


@pytest.mark.parametrize('name', ['harness_sha256','guard_tests_sha256','source_hashes',
                                'environment_sha256','specification_sha256','command_sha256'])
def test_freeze_bootstrap_cheap_pin_failure_does_not_consume(monkeypatch,name):
    monkeypatch.setattr(f,'_CONSUMED',False)
    runtime,command,record,_=freeze_bootstrap_stub()
    if name=='source_hashes': record[name]={}
    else: record[name]='f'*64
    freeze=f.RunFreeze(**record)
    with pytest.raises(f.IntegrityError): f.prepare_attempt(freeze,runtime,command)
    assert f._CONSUMED is False


def test_freeze_bootstrap_output_preflight_does_not_consume(monkeypatch):
    monkeypatch.setattr(f,'_CONSUMED',False)
    runtime,command,record,_=freeze_bootstrap_stub()
    monkeypatch.setattr(f,'MAIN_OUTPUT_RESERVATION',f.MAX_EVIDENCE+1)
    with pytest.raises(f.IntegrityError,match='output_precharge'):
        f.prepare_attempt(f.RunFreeze.from_mapping(record),runtime,command)
    assert f._CONSUMED is False


@pytest.mark.parametrize('fail_stage', ['builder','validator','digest','none'])
def test_freeze_bootstrap_single_stub_build_boundary_and_failure_consumption(monkeypatch,fail_stage):
    monkeypatch.setattr(f,'_CONSUMED',False)
    runtime,command,record,events=freeze_bootstrap_stub()
    meter=f.prepare_attempt(f.RunFreeze.from_mapping(record),runtime,command)
    assert events==['attest'] and not f._CONSUMED
    assert meter.output_reserved==f.MAIN_OUTPUT_RESERVATION
    token=object()
    def builder(arg):
        assert f._CONSUMED is True and arg is runtime
        events.append('stub_builder')
        if fail_stage=='builder': raise RuntimeError('stub build failure')
        return token
    def validator(arg,packet):
        assert f._CONSUMED is True and arg is runtime and packet is token
        events.append('stub_validate')
        if fail_stage=='validator': raise RuntimeError('stub validation failure')
        return 'invalid' if fail_stage=='digest' else 'b'*64
    if fail_stage=='none':
        packet,observed=f.consume_and_build(runtime,meter,builder,validator)
        assert packet is token and observed=='b'*64
        a=f.with_observed_digest({'arm':'A'},observed)
        b=f.with_observed_digest({'arm':'B'},observed)
        report=f.with_observed_digest({'arms':[a,b]},observed)
        assert report['observed_packet_sha256']==a['observed_packet_sha256']==b['observed_packet_sha256']
    else:
        with pytest.raises((RuntimeError,f.IntegrityError)):
            f.consume_and_build(runtime,meter,builder,validator)
    assert f._CONSUMED and events.count('stub_builder')==1
    with pytest.raises(f.IntegrityError,match='attempt_consumed'):
        f.consume_and_build(runtime,meter,builder,validator)
    assert events.count('stub_builder')==1


def test_freeze_bootstrap_source_order_and_packet_contract():
    tree=ast.parse(PATH.read_text(encoding='utf-8'))
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    attempt=functions['execute_once']
    names=[n.func.id for stmt in attempt.body for n in ast.walk(stmt) if isinstance(n,ast.Call)
           and isinstance(n.func,ast.Name)]
    assert names.index('prepare_attempt')<names.index('consume_and_build')<names.index('Driver')
    calls=[n for n in ast.walk(attempt) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
           and n.func.id=='consume_and_build']
    assert len(calls)==1 and [ast.unparse(x) for x in calls[0].args]==\
        ['runtime','meter','build_fixed_packet','validate_packet']
    boundary=functions['consume_and_build'].body
    set_at=next(i for i,n in enumerate(boundary) if isinstance(n,ast.Assign) and
                any(isinstance(t,ast.Name) and t.id=='_CONSUMED' for t in n.targets))
    assert ast.unparse(boundary[set_at])=='_CONSUMED = True'
    assert ast.unparse(boundary[set_at+1])=='packet = builder(runtime)'
    assert ast.unparse(boundary[set_at+2])=='observed = validator(runtime, packet)'
    loop=next(n for n in attempt.body if isinstance(n,ast.For))
    assert ast.literal_eval(loop.iter)==('A','B') and not any(isinstance(n,ast.If) for n in loop.body)
    source=ast.unparse(functions['validate_packet'])
    for gate in ('packet_counts','packet_frame_arrays','packet_camera_primitives','packet_graph_arrays',
                 'packet_graph_recipe','packet_edge_recipe','packet_closure','packet_observed_digest_integrity'):
        assert gate in source
    assert 'immutable_array' in source and \
        'observed = packet_digest(packet.frames, packet.cameras, packet.graphs)' in source
    assert 'packet_sha256' not in f.RunFreeze.__dataclass_fields__


def test_freeze_bootstrap_no_numerical_imports_or_packet_execution():
    assert not any(name=='numpy' or name.startswith('numpy.') or name=='cv2' or
                   name.startswith('mc_photogrammetry.') for name in sys.modules)
    assert not f._CONSUMED
