"""Stub-only barriers. No image/oracle/production import, inference or disk artifact."""
import ast
import base64
from array import array
from dataclasses import dataclass, make_dataclass
from fractions import Fraction
import importlib.util
import io
import json
import math
import struct
import hashlib
import inspect
import marshal
from dataclasses import fields, is_dataclass
from collections.abc import Mapping
import builtins
from pathlib import Path
import sys
import subprocess
import textwrap
from types import SimpleNamespace
from types import MappingProxyType

import pytest

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'scripts/evaluation/sealed_image_only_baseline.py'
spec=importlib.util.spec_from_file_location('sealed_baseline_guards',PATH)
f=importlib.util.module_from_spec(spec); sys.modules[spec.name]=f; spec.loader.exec_module(f)


class Handle(io.BytesIO):
    def __init__(self,owner,path):
        super().__init__(); self.owner=owner; self.path=path
    def close(self):
        if not self.closed: self.owner.files[self.path]=self.getvalue()
        super().close()


class MemoryIO:
    def __init__(self):
        self.files={}; self.handles={}; self.directories=set(); self.events=[]; self.tamper=False; self.flush_fail=False
    def reserve(self,parent,name):
        path=Path(parent)/name
        if path in self.directories: raise FileExistsError('reserved')
        self.directories.add(path); self.events.append('reserve'); return path
    def exists(self,path): return path in self.files or path in self.handles
    def open_exclusive(self,path):
        if self.exists(path): raise FileExistsError('exclusive')
        self.events.append('exclusive:'+path.name)
        handle=Handle(self,path); self.handles[path]=handle; return handle
    def flush(self,stream):
        self.events.append('fsync:'+stream.path.name)
        if self.flush_fail and stream.path.name=='native-receipt.json' and stream.tell()!=f.MAX_RECEIPT:
            raise OSError('stub fsync failure')
    def data(self,path):
        handle=self.handles.get(path)
        return handle.getvalue() if handle is not None and not handle.closed else self.files[path]
    def read_hash(self,path,limit):
        self.events.append('readback:'+path.name)
        if path in tuple(f.ROOT/p for p in f.INPUTS):
            return {'sha256':f.INPUT_HASHES[tuple(f.ROOT/p for p in f.INPUTS).index(path)],'bytes':123}
        data=self.data(path)
        assert len(data)<=limit
        return {'sha256':'0'*64 if self.tamper and path.name=='native-receipt.json' else f.sha(data),'bytes':len(data)}
    def read_small(self,path,limit):
        self.events.append('read:'+path.name)
        data=self.data(path); assert len(data)<=limit; return data


@dataclass
class PlacementRefusal:
    reason: str='stub_refusal'
    detail: str='stub detail'


@dataclass
class Outcome:
    status: str='geometry_unavailable'
    reason: str='stub_entry_refusal'


def fixture(predict,targets=()):
    manifest=f.source_manifest()  # source-byte hashes only; no production import/data
    callables={'prediction':f.callable_identity(predict)}
    callables.update({stage:f.callable_identity(getattr(module,name)) for module,name,stage in targets})
    runtime=f.Runtime(predict,targets,{'stub_runtime':True,'callables':callables})
    command=('stub','sealed-test')
    freeze=f.Freeze(f.digest(manifest),f.sha(PATH.read_bytes()),f.sha(Path(__file__).read_bytes()),
                    f.digest(runtime.provenance),f.digest(command),'0'*64)
    return runtime,freeze,command,manifest


def run(predict,storage=None,targets=(),loader=None,name='stub'):
    storage=storage or MemoryIO()
    runtime,freeze,command,manifest=fixture(predict,targets)
    result=f.run_once(runtime,freeze,command,storage,Path('/memory'),name,manifest,loader)
    return result,storage


def test_delegate_exactly_once_unchanged_arguments_and_result():
    value=object(); result=object(); calls=[]; options={'x':'native'}
    def native(arg,**kwargs):
        calls.append((arg,kwargs)); return result
    observer=f.Observer()
    returned=observer.delegate('pair',native)(value,options=options)
    assert returned is result and len(calls)==1 and calls[0][0] is value and calls[0][1]['options'] is options
    assert observer.stages['pair']['calls']==1 and options=={'x':'native'}


def test_single_prediction_literal_inputs_and_fresh_target():
    calls=[]
    def native(paths,**kwargs):
        calls.append((paths,kwargs)); return Outcome()
    result,storage=run(native)
    assert result['state']=='SEALED' and not result['metadata_access']
    assert len(calls)==1 and calls[0][0]==[f.ROOT/p for p in f.INPUTS]
    assert set(calls[0][1])=={'project_path'}
    assert calls[0][1]['project_path']==Path('/memory/stub/prediction.mcscan')
    assert storage.events.index('reserve')<storage.events.index('fsync:native-receipt.json')
    assert storage.events.index('fsync:native-receipt.json')<storage.events.index('readback:native-receipt.json')
    assert storage.events.index('readback:native-receipt.json')<storage.events.index('exclusive:COMPLETE.json')
    native_record=f.strict_json(storage.files[Path('/memory/stub/native-receipt.json')])
    assert all(x['state']=='not_reached' for x in native_record['stages'].values())
    assert native_record['terminal']['kind']=='returned'


@pytest.mark.parametrize('kind',['bare','exception'])
def test_bare_refusal_and_native_error_are_sealed_with_missing_stages(kind):
    calls=[]
    def native(paths,**kwargs):
        calls.append(1)
        if kind=='exception': raise ValueError('stub native exception')
        return PlacementRefusal()
    result,storage=run(native)
    assert len(calls)==1 and result['state']=='SEALED', native_record(storage)['instrumentation_failure']
    record=f.strict_json(storage.files[Path('/memory/stub/native-receipt.json')])
    assert record['terminal']['kind']==('bare_placement_refusal' if kind=='bare' else 'raised')
    assert record['stages']['G_solver']['state']=='not_reached'
    assert record['published_project'] is None


def test_observer_installs_and_restores_source_stub_and_tracks_refusal():
    token=PlacementRefusal(); calls=[]
    def pair(left,**kwargs): calls.append((left,kwargs)); return token
    module=SimpleNamespace(pair=pair)
    observer=f.Observer()
    with observer.install(((module,'pair','pair'),)):
        assert module.pair('native',camera=None) is token
    assert module.pair is pair and calls==[('native',{'camera':None})]
    assert observer.stages['pair']['state']=='refused' and sys.gettrace() is None


def test_G_trace_observes_prefilter_without_extra_native_kwargs():
    # Source-code stub has the same locals; no actual graph/geometry call.
    def matcher(readings,signatures,**kwargs):
        assert kwargs=={'max_shift_cells':32}
        matches=['original','dropped']; filtered=matches[:1]; refusals=['same_axis_mismatch']
        return filtered
    module=SimpleNamespace(matcher=matcher)
    observer=f.Observer()
    with observer.install(((module,'matcher','G_matcher'),)):
        result=module.matcher(['r'],['s'],max_shift_cells=32)
    assert result==['original'] and module.matcher is matcher
    traced=resolve_projection(observer.trace_seen[0],observer.projection.definitions)
    assert traced['matches']==['original','dropped']
    assert traced['filtered']==['original']
    assert traced['dispositions']==['kept','dropped']


def test_metadata_denied_before_seal_and_loaded_only_after_verified_marker():
    storage=MemoryIO(); writer=f.Writer(storage,Path('/memory'),'guard')
    calls=[]
    with pytest.raises(f.IntegrityError,match='unsealed metadata barrier'):
        writer.posthoc(None,lambda:calls.append('oracle'))
    seal=writer.seal({'native':'complete'},True)
    assert writer.posthoc(seal,lambda:calls.append('oracle')) is None
    assert calls==['oracle'] and storage.events.index('read:COMPLETE.json')>storage.events.index('exclusive:COMPLETE.json')


@pytest.mark.parametrize('failure',['readback','fsync'])
def test_seal_failure_retains_evidence_and_blocks_metadata(failure):
    storage=MemoryIO(); calls=[]
    if failure=='readback': storage.tamper=True
    else: storage.flush_fail=True
    result,storage=run(lambda paths,**kwargs:Outcome(),storage,loader=lambda:calls.append('oracle'))
    assert result['state']=='UNSEALED' and not calls and not result['metadata_access']
    assert storage.exists(Path('/memory/stub/native-receipt.json'))
    assert not storage.exists(Path('/memory/stub/COMPLETE.json'))


def test_instrumentation_failure_sealed_but_oracle_blocked():
    calls=[]
    def feature(image): return 'x'*(f.MAX_STRING+1)
    module=SimpleNamespace(feature=feature)
    def native(paths,**kwargs): module.feature('image'); return Outcome()
    result,storage=run(native,targets=((module,'feature','features'),),loader=lambda:calls.append('oracle'))
    assert result['state']=='SEALED_INSTRUMENTATION_FAILURE' and not calls
    assert module.feature is feature
    record=f.strict_json(storage.files[Path('/memory/stub/native-receipt.json')])
    assert record['stages']['features']['state']=='instrumentation_failure'


def test_prewrite_cap_before_encoding_allocation_or_payload_write(monkeypatch):
    storage=MemoryIO(); writer=f.Writer(storage,Path('/memory'),'cap')
    before=writer.stream.tell()
    monkeypatch.setattr(f,'encoded_size',lambda value:f.MAX_RECEIPT)
    monkeypatch.setattr(f,'tokens',lambda value:pytest.fail('encoding before cap rejection'))
    with pytest.raises(f.IntegrityError,match='receipt prewrite cap'): writer.seal({'x':'bounded'},True)
    assert writer.stream.tell()==before and not storage.exists(writer.marker)


def test_overwrite_and_automatic_repeat_denied_before_second_prediction():
    calls=[]; storage=MemoryIO()
    def native(paths,**kwargs): calls.append(1); return Outcome()
    run(native,storage)
    with pytest.raises(FileExistsError): run(native,storage)
    assert len(calls)==1
    with pytest.raises(FileExistsError): storage.open_exclusive(Path('/memory/stub/native-receipt.json'))


@pytest.mark.parametrize('which',['marker','receipt'])
def test_postseal_tamper_denies_oracle(which):
    storage=MemoryIO(); writer=f.Writer(storage,Path('/memory'),'tamper')
    seal=writer.seal({'native':'complete'},True); calls=[]
    path=writer.marker if which=='marker' else writer.path
    storage.files[path]+=b' '
    with pytest.raises(f.IntegrityError): writer.posthoc(seal,lambda:calls.append('oracle'))
    assert not calls


def test_call_and_evidence_caps_precharge_before_native_delegate():
    observer=f.Observer(); calls=[]
    wrapped=observer.delegate('pair',lambda value:calls.append(value))
    wrapped('first')
    with pytest.raises(f.IntegrityError,match='call cap'): wrapped('second')
    assert calls==['first']
    budget=f.Budget(reserved=f.MAX_OBSERVER)
    with pytest.raises(f.IntegrityError,match='evidence precharge'): f.Snapshot(budget).take('x')
    budget=f.Budget(started=0.,clock=lambda:121.)
    with pytest.raises(f.IntegrityError,match='watchdog'): budget.charge(1)


def test_readonly_mapping_cameras_are_observed_and_terminal_capture_failure_is_not_native_error():
    snapshot=f.Snapshot(f.Budget())
    assert snapshot.take(MappingProxyType({'cameras':{'image-0000':{'yaw':12.}}}))==\
        {'cameras':{'image-0000':{'yaw':12.}}}
    calls=[]
    result,storage=run(lambda paths,**kwargs:'x'*(f.MAX_STRING+1),loader=lambda:calls.append('oracle'))
    assert result['state']=='SEALED_INSTRUMENTATION_FAILURE' and not calls
    record=f.strict_json(storage.files[Path('/memory/stub/native-receipt.json')])
    assert record['terminal']['kind']=='instrumentation_failure'


def test_empty_native_array_hash_and_coordinate_allocation_precharge():
    class Buffer(array):
        pass
    empty=Buffer('d')
    empty.dtype=SimpleNamespace(str='<f8'); empty.shape=(0,4); empty.strides=(32,8)
    empty.nbytes=0; empty.size=0; empty.flags=SimpleNamespace(writeable=False)
    receipt=f.Snapshot(f.Budget()).take(empty,'matches')
    assert receipt['sha256']==f.sha(b'') and numeric_bytes(receipt)==b''
    values=Buffer('d',[1.,2.,3.,4.])
    values.dtype=empty.dtype; values.shape=(2,2); values.strides=(16,8)
    values.nbytes=32; values.size=4; values.flags=SimpleNamespace(writeable=False)
    assert struct.unpack('<4d',numeric_bytes(f.Snapshot(f.Budget()).take(values,'edges')))==(1.,2.,3.,4.)
    values[0]=float('nan')
    nonfinite=f.Snapshot(f.Budget()).take(values,'edges')
    assert math.isnan(struct.unpack('<4d',numeric_bytes(nonfinite))[0])
    assert f.encoded_size(nonfinite)>0
    with pytest.raises(f.IntegrityError,match='evidence precharge'):
        f.Snapshot(f.Budget(reserved=f.MAX_OBSERVER-1100)).take(values,'points')


def test_exact_rational_native_scalar_receipt():
    assert f.Snapshot(f.Budget()).take(Fraction(3,7))==\
        {'type':'fractions.Fraction','numerator':3,'denominator':7}


def test_runtime_callable_tamper_rejected_before_reservation():
    def stage(value): return value
    module=SimpleNamespace(stage=stage)
    runtime,freeze,command,manifest=fixture(lambda *a,**k:Outcome(),((module,'stage','features'),))
    module.stage=lambda value:None
    storage=MemoryIO()
    with pytest.raises(f.IntegrityError,match='runtime callable changed'):
        f.run_once(runtime,freeze,command,storage,Path('/memory'),'tamper',manifest)
    assert not storage.directories


def test_callable_pin_stable_with_live_code_references_and_exception_traceback():
    def native(): raise ValueError('retained traceback')
    before=f.callable_identity(native)
    references=[native.__code__]*4
    try: native()
    except ValueError as error:
        assert references[0] is native.__code__ and error.__traceback__ is not None
        assert f.callable_identity(native)==before
    native.__code__=(lambda:None).__code__
    assert f.callable_identity(native)!=before


def test_callable_default_identity_stable_in_two_independent_stub_processes():
    program=textwrap.dedent('''
        import importlib.util, json, os, sys
        from pathlib import Path
        spec=importlib.util.spec_from_file_location('sealed_baseline_guards',sys.argv[1])
        f=importlib.util.module_from_spec(spec); sys.modules[spec.name]=f; spec.loader.exec_module(f)
        def match_and_solve_pair(left: 'str', *, settings=None, gates=None):
            raise AssertionError('stub callable must not execute')
        def build_image_derived_connections(session: 'ImageDerivedSession', *,
                                            matcher=match_and_solve_pair, budgets=None):
            raise AssertionError('stub callable must not execute')
        shared={'threshold':[1,2]}
        build_image_derived_connections.__kwdefaults__['options']=(shared,shared)
        identity=f.callable_identity(build_image_derived_connections)
        runtime={'callables':{'connections':identity},'stub_only':True}
        assert not any(n=='numpy' or n=='cv2' or n=='PIL' or n.startswith('mc_photogrammetry') for n in sys.modules)
        print(json.dumps({'pid':os.getpid(),'identity':identity,'runtime_sha256':f.digest(runtime)},sort_keys=True))
    ''')
    children=[]
    for _ in range(2):
        child=subprocess.run([sys.executable,'-I','-B','-c',program,str(PATH)],
                             capture_output=True,check=True,timeout=10)
        assert not child.stderr and len(child.stdout)<16_384
        children.append(json.loads(child.stdout))
    assert children[0]['pid']!=children[1]['pid']
    assert children[0]['identity']==children[1]['identity']
    assert children[0]['runtime_sha256']==children[1]['runtime_sha256']
    assert '0x' not in f.canonical(children[0]['identity']).decode()


def test_callable_defaults_shared_objects_distinct_from_equal_independent_objects():
    def native(left=None,right=None): pass
    shared={'threshold':[1,2]}
    native.__defaults__=(shared,shared)
    shared_pin=f.callable_identity(native)
    assert shared_pin['defaults']['items'][1]=={'reference':shared_pin['defaults']['items'][0]['node']}
    native.__defaults__=({'threshold':[1,2]},{'threshold':[1,2]})
    independent_pin=f.callable_identity(native)
    assert independent_pin!=shared_pin
    other={'threshold':[1,2]}
    native.__defaults__=(other,other)
    assert f.callable_identity(native)==shared_pin
    other['threshold'].append(3)
    assert f.callable_identity(native)!=shared_pin


@pytest.mark.parametrize('mutation',['outer-code','positional','keyword','nested-code','nested-default','annotation','container'])
def test_callable_identity_detects_code_and_every_default_mutation(mutation):
    def matcher(value=7): return value
    options={'threshold':[1,2]}
    def native(value=1,*,matcher=matcher,options=options): return value
    before=f.callable_identity(native)
    if mutation=='outer-code': native.__code__=(lambda value=1,**kwargs:2).__code__
    elif mutation=='positional': native.__defaults__=(2,)
    elif mutation=='keyword': native.__kwdefaults__['matcher']=None
    elif mutation=='nested-code': matcher.__code__=(lambda value=7:8).__code__
    elif mutation=='nested-default': matcher.__defaults__=(8,)
    elif mutation=='annotation': native.__annotations__['value']='changed-type'
    else: options['threshold'].append(3)
    assert f.callable_identity(native)!=before


def test_nested_default_mutation_refuses_runtime_attestation_before_reservation():
    def matcher(value=7): return value
    def connections(session,*,matcher=matcher): return session
    module=SimpleNamespace(connections=connections)
    runtime,freeze,command,manifest=fixture(lambda *a,**k:pytest.fail('must not predict'),
                                          ((module,'connections','connections'),))
    matcher.__defaults__=(8,)
    storage=MemoryIO()
    with pytest.raises(f.IntegrityError,match='runtime callable changed'):
        f.run_once(runtime,freeze,command,storage,Path('/memory'),'default-tamper',manifest)
    assert not storage.directories


@pytest.mark.parametrize('location',['positional','keyword','nested','annotation'])
def test_unknown_pointer_values_refuse_without_invoking_repr(location):
    class NeverRepr:
        def __repr__(self): pytest.fail('unstable/custom repr must not execute')
    foreign=NeverRepr()
    def native(value=None,*,option=None): pass
    if location=='positional': native.__defaults__=(foreign,)
    elif location=='keyword': native.__kwdefaults__['option']=foreign
    elif location=='nested': native.__kwdefaults__['option']=[{'nested':foreign}]
    else: native.__annotations__['value']=foreign
    with pytest.raises(f.IntegrityError,match='unsupported callable identity value'):
        f.callable_identity(native)


@pytest.mark.parametrize('cycle',['container','function'])
def test_callable_default_cycles_fail_boundedly_and_cleanup(cycle):
    def native(value=None): pass
    value=[]; value.append(value)
    native.__defaults__=(value if cycle=='container' else native,)
    encoder=f.CallableIdentityEncoder()
    with pytest.raises(f.IntegrityError,match='callable identity cycle'): encoder.take(native)
    assert not encoder.active and encoder.nodes<512 and encoder.reserved<=16_384


def test_callable_default_depth_precharge_order_and_container_types():
    def native(value=None): pass
    nested=None
    for _ in range(17): nested=[nested]
    native.__defaults__=(nested,)
    with pytest.raises(f.IntegrityError,match='depth'): f.callable_identity(native)
    native.__defaults__=([None]*512,)
    with pytest.raises(f.IntegrityError,match='precharge'): f.callable_identity(native)
    native.__defaults__=(('a',),)
    tuple_pin=f.callable_identity(native)
    native.__defaults__=(['a'],)
    assert f.callable_identity(native)!=tuple_pin
    native.__defaults__=({'a':1,'b':2},)
    mapping_pin=f.callable_identity(native)
    native.__defaults__=({'b':2,'a':1},)
    assert f.callable_identity(native)!=mapping_pin


def test_default_callable_closure_refuses_and_code_size_precedes_marshal(monkeypatch):
    state=[]
    def matcher(): return state
    def native(*,matcher=matcher): pass
    with pytest.raises(f.IntegrityError,match='default callable closure'): f.callable_identity(native)
    def oversized(): pass
    oversized.__code__=oversized.__code__.replace(co_consts=(None,'x'*(f.MAX_SOURCE_BYTES//4)))
    monkeypatch.setattr(f.marshal,'dumps',lambda *args:pytest.fail('marshal before code precharge'))
    with pytest.raises(f.IntegrityError,match='code byte precharge'): f.callable_identity(oversized)


@pytest.mark.parametrize('field',['source_manifest_sha256','evaluator_sha256','tests_sha256','runtime_sha256','command_sha256'])
def test_caller_frozen_digests_required_before_prediction(field):
    runtime,freeze,command,manifest=fixture(lambda *a,**k:pytest.fail('prediction before freeze'))
    values={name:getattr(freeze,name) for name in freeze.__dataclass_fields__}; values[field]='f'*64
    storage=MemoryIO()
    with pytest.raises(f.IntegrityError):
        f.run_once(runtime,f.Freeze(**values),command,storage,Path('/memory'),'pins',manifest)
    assert not storage.directories


@pytest.mark.parametrize('raw',[b'{}{}',b'{"x":NaN}',b'{"x":1e999}',b'{"x":1,"x":2}',b'[]',b' '*(f.BLOCK+1)],
                         ids=['trailing','nan','overflow','duplicate','array','oversize'])
def test_freeze_parser_rejects_unbounded_duplicate_nonfinite_or_unknown(raw):
    with pytest.raises(f.IntegrityError): f.Freeze.read(io.BytesIO(raw))


def test_exact_JSON_size_and_bounded_source_manifest():
    value={'unicode':'\u2603','quote':'"\\\n','nested':[None,False,1,1.25]}
    assert f.encoded_size(value)==len(b''.join(f.tokens(value)))==len(f.canonical(value))
    manifest=f.source_manifest()
    assert len(manifest)==183 and [x['path'] for x in manifest]==sorted(x['path'] for x in manifest)
    assert all(f.is_digest(x['sha256']) for x in manifest)


def resolve_projection(raw,definitions=None):
    definitions={} if definitions is None else definitions
    def tuple_key(node):
        return tuple(tuple_key(child) for child in node) if type(node) is list else node
    def resolve(node):
        if type(node) is dict:
            if set(node)=={'native_mapping_entries'}:
                result={}
                for entry in node['native_mapping_entries']:
                    key=resolve(entry['key'])
                    if entry['key_type']=='tuple': key=tuple_key(key)
                    result[key]=resolve(entry['value'])
                return result
            if set(node)=={'semantic_reference'}:
                record=definitions[node['semantic_reference']]
                if record.get('encoding')=='G-container-v1': return resolve(record['projection']['items'])
                if record.get('encoding')=='G-dispositions-v1':
                    body=record['projection']; bits=base64.b64decode(body['bits_base64'])
                    return ['kept' if bits[index//8]&(1<<(index%8)) else 'dropped' for index in range(body['count'])]
                if record.get('encoding') in ('G-field-vector-v1','geometry-field-vector-v1'):
                    names=(f.G_RECORD_SCHEMAS[record['projection']['schema']][0] if record['encoding']=='G-field-vector-v1'
                           else f.GEOMETRY_RECORD_SCHEMAS[record['projection']['schema']][0])
                    values=json.loads(record['projection'].get('values_json') or ''.join(record['projection']['values_json_chunks']))
                    return {'type':record['projection']['schema'],
                            'fields':{name:resolve(value) for name,value in zip(names,values)}}
                return resolve(record.get('values_projection',record['projection']))
            return {key:resolve(value) for key,value in node.items()}
        if type(node) is list: return [resolve(value) for value in node]
        return node
    return resolve(raw)


def native_record(storage):
    raw=f.strict_json(storage.files[Path('/memory/stub/native-receipt.json')])
    return resolve_projection(raw,raw.get('semantic_definitions',{}))


def legacy_logical_wire(node,definitions,encoding=None):
    """Test-only materialization: expand NEW inline kinds, never legacy refs."""
    if encoding=='G-container-v1': return legacy_logical_wire(node['items'],definitions)
    if encoding=='G-dispositions-v1':
        bits=base64.b64decode(node['bits_base64'])
        return ['kept' if bits[index//8]&(1<<(index%8)) else 'dropped' for index in range(node['count'])]
    if encoding in ('G-field-vector-v1','geometry-field-vector-v1'):
        names=(f.G_RECORD_SCHEMAS[node['schema']][0] if encoding=='G-field-vector-v1'
               else f.GEOMETRY_RECORD_SCHEMAS[node['schema']][0])
        values=json.loads(node.get('values_json') or ''.join(node['values_json_chunks']))
        return {'type':node['schema'],'fields':{'native_mapping_entries':[
            {'key_type':'str','key':name,'value':legacy_logical_wire(value,definitions)}
            for name,value in zip(names,values)]}}
    if type(node) is dict:
        if set(node)=={'semantic_reference'}:
            record=definitions[node['semantic_reference']]
            if record['kind'] in ('g_inline_record','g_sequence','g_mapping','geometry_inline_record'):
                return legacy_logical_wire(record['projection'],definitions,record['encoding'])
            return node
        return {key:legacy_logical_wire(value,definitions) for key,value in node.items()}
    if type(node) is list: return [legacy_logical_wire(value,definitions) for value in node]
    return node


# Independent pre-G-compaction reference. Frozen from the accepted b5da5f7141
# contract: matches/source records/arrays remain references; refusals inline;
# maps always typed ordered rows; list and tuple views both become JSON lists.
# It does not call current projection/digest/schema helpers.
def frozen_native_fingerprint(value):
    result=hashlib.sha256(); active=set()
    def emit(text):
        raw=text.encode('utf-8'); result.update(str(len(raw)).encode()+b':'); result.update(raw)
    def walk(item):
        kind=type(item); emit(kind.__module__+'.'+kind.__qualname__)
        if item is None: return
        if kind in (bool,int,float): emit(item.hex() if kind is float else str(item)); return
        if isinstance(item,(str,Path)):
            text=str(item); emit(str(len(text))); result.update(text.encode('utf-8')); return
        if kind is bytes: emit(str(len(item))); result.update(item); return
        if hasattr(item,'dtype') and hasattr(item,'shape') and hasattr(item,'nbytes'):
            emit(item.dtype.str); emit(str(tuple(item.shape))); emit(str(tuple(item.strides)))
            emit(str(bool(item.flags.writeable))); emit(str(item.nbytes)); result.update(memoryview(item).cast('B')); return
        if kind is Fraction: walk(item.numerator); walk(item.denominator); return
        if inspect.isfunction(item):
            emit(item.__module__+'.'+item.__qualname__); emit(hashlib.sha256(marshal.dumps(item.__code__,2)).hexdigest()); return
        if not (is_dataclass(item) or isinstance(item,(Mapping,list,tuple))): return
        assert id(item) not in active; active.add(id(item))
        try:
            if is_dataclass(item):
                for field in fields(item): emit(field.name); walk(getattr(item,field.name))
            elif isinstance(item,Mapping):
                emit(str(len(item)))
                for key,child in item.items(): walk(key); walk(child)
            else:
                emit(str(len(item)))
                for child in item: walk(child)
        finally: active.remove(id(item))
    walk(value); return result.hexdigest()


def frozen_prechange_projection(value):
    # Literal frozen namespace set deliberately excludes PlaneMatchRefusal.
    registered={'mc_photogrammetry.reconstruction.plane_graph.PlaneMatch',
        'mc_photogrammetry.reconstruction.plane_graph.ComponentSolution',
        'mc_photogrammetry.reconstruction.lattice_planes.PlaneReading',
        'mc_photogrammetry.reconstruction.lattice_planes.PlaneSignature'}
    def mapping(items):
        return {'native_mapping_entries':[{'key_type':type(key).__name__,'key':walk(key),'value':walk(child)}
                                          for key,child in items.items()]}
    def walk(item):
        qualified=type(item).__module__+'.'+type(item).__qualname__
        if is_dataclass(item):
            if qualified in registered: return {'semantic_reference':'record:'+frozen_native_fingerprint(item)}
            return {'type':qualified,'fields':mapping({field.name:getattr(item,field.name) for field in fields(item)})}
        if hasattr(item,'dtype') and hasattr(item,'shape') and hasattr(item,'nbytes'):
            return {'semantic_reference':'array:'+frozen_native_fingerprint(item)}
        if isinstance(item,Mapping): return mapping(item)
        if isinstance(item,(list,tuple)): return [walk(child) for child in item]
        if type(item) is float and not math.isfinite(item): return {'nonfinite':repr(item)}
        if type(item) is bytes: return {'bytes':len(item),'sha256':hashlib.sha256(item).hexdigest()}
        if isinstance(item,Path): return str(item)
        assert item is None or type(item) in (bool,int,float,str)
        return item
    return walk(value)


def frozen_canonical_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()


def frozen_record_body(value):
    return {'type':type(value).__module__+'.'+type(value).__qualname__,
            'fields':{'native_mapping_entries':[{'key_type':'str','key':field.name,
                'value':frozen_prechange_projection(getattr(value,field.name))} for field in fields(value)]}}


def test_native_fingerprint_body_remains_exact_accepted_AST():
    tree=ast.parse(PATH.read_text(encoding='utf-8'))
    declaration=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='StageProjection')
    native=next(node for node in declaration.body if isinstance(node,ast.FunctionDef) and node.name=='native_digest')
    assert hashlib.sha256(ast.dump(native,include_attributes=False).encode()).hexdigest()==\
        'cc744fa1ac1140b1581036dc07ee90f1295906c01c948c09ba39d2d264f73e1d'


@pytest.mark.parametrize('change',['none','reorder','mutate','tuple','negative-zero','nonfinite'])
def test_G_legacy_digest_matches_independent_frozen_prechange_oracle(change):
    match=source_shaped_record('PlaneMatch',left='left',right='right',left_axis=0,right_axis=1,
        left_b=-0.,right_b=1.,quadrant_of_right=2,psr=float('inf'),peak=float('nan'),second_peak=float('-inf'),
        overlap_cells=3,integer_residual=None,score=2.,left_plane=4,right_plane=5,cell_shift_rc=(7,-8))
    refusal=source_shaped_record('PlaneMatchRefusal',left='left',right='right',left_axis=0,right_axis=1,
        quadrant_of_right=2,reason='same_axis_mismatch',detail='quote"\\\u2603')
    reading=source_shaped_record('PlaneReading',frame='left',axis=0,side=2.)
    array_value=RawNumericBuffer(struct.pack('<2d',-0.,float('inf')),'<f8')
    value={'matches':[match],('frame',1):{'refusals':(refusal,),1:[array_value,reading]},
           'semantic_reference':'ordinary','nested':[None,-0.,float('nan'),float('inf'),float('-inf')]}
    observer=f.Observer(); projection=observer.projection
    with observer.budget.scope('G_solver'):
        first=projection.take(value)
        first_expected=frozen_prechange_projection(value)
        first_record=frozen_record_body(match)
        if change=='reorder': value=dict(reversed(tuple(value.items())))
        elif change=='mutate': refusal.detail='changed'; match.score=3.
        elif change=='tuple': value['matches']=tuple(value['matches'])
        elif change=='negative-zero': value['nested'][1]=0.
        elif change=='nonfinite': value['nested'][2]=float('-inf')
        current=projection.take(value)
        assert projection.native_digest(value)==frozen_native_fingerprint(value)
    logical=legacy_logical_wire(current,projection.definitions)
    assert frozen_canonical_digest(logical)==frozen_canonical_digest(frozen_prechange_projection(value))
    assert projection.projected_digest(current)==frozen_canonical_digest(frozen_prechange_projection(value))
    assert projection.projected_digest(first)==frozen_canonical_digest(first_expected)
    old_match_key=first_expected['native_mapping_entries'][0]['value'][0]['semantic_reference']
    assert projection.definitions[old_match_key]['content_sha256']==frozen_canonical_digest(first_record)
    current_match_key='record:'+frozen_native_fingerprint(match)
    assert projection.definitions[current_match_key]['content_sha256']==frozen_canonical_digest(frozen_record_body(match))
    if change!='none': assert current!=first
    projection.validate((first,current))


def test_large_production_shaped_mask_digest_through_coordinator():
    @dataclass
    class RegionEvidence:
        packed_mask_base64: str
    @dataclass
    class PlaneReading:
        region_evidence: RegionEvidence
    payload='A'*(2*f.BLOCK+4)
    reading=PlaneReading(RegionEvidence(payload)); calls=[]
    def lattice(): calls.append('lattice'); return [reading]
    def matcher(readings): calls.append('G'); return readings
    module=SimpleNamespace(lattice=lattice,matcher=matcher)
    def prediction(paths,**kwargs):
        module.matcher(module.lattice()); return Outcome()
    result,storage=run(prediction,targets=((module,'lattice','lattice'),(module,'matcher','G_matcher')))
    assert result['state']=='SEALED' and calls==['lattice','G'] and f.MAX_STRING==4096
    record=native_record(storage)
    observed=record['stages']['lattice']['records'][0]['result'][0]
    receipt=observed['fields']['region_evidence']['fields']['packed_mask_base64']
    assert receipt=={'type':'builtins.str','field':'packed_mask_base64','encoding':'base64',
                    'text_encoding':'ascii','encoded_bytes':len(payload),'characters':len(payload),
                    'sha256':f.sha(payload.encode('ascii')),'payload_policy':'digest_only; no decoding'}
    assert payload not in storage.files[Path('/memory/stub/native-receipt.json')].decode()
    assert record['stages']['G_matcher']['state']=='returned'


def test_packed_mask_precharge_before_slice_or_hash_and_text_cap_unchanged():
    class Untouchable(str):
        def __getitem__(self,index): pytest.fail('slice before allowance')
    with pytest.raises(f.IntegrityError,match='evidence precharge'):
        f.Snapshot(f.Budget(reserved=f.MAX_OBSERVER-1000)).packed_mask_receipt(Untouchable('A'*8192))
    class Oversized(Untouchable):
        def __len__(self): return f.MAX_ARRAY+1
    with pytest.raises(f.IntegrityError,match='encoded mask cap'):
        f.Snapshot(f.Budget()).packed_mask_receipt(Oversized('A'))
    with pytest.raises(f.IntegrityError,match='string cap'):
        f.Snapshot(f.Budget()).take('A'*8192,'other_encoded_field')


def test_source_shaped_patch_and_profile_payloads_through_observer_coordinator(monkeypatch):
    @dataclass(frozen=True)
    class SplitPhasePatchRef:
        mode: str
        patch_sha256: str
        rows: int
        columns: int
        dtype: str
        byte_length: int
        finite_nonnegative: bool
        inline_samples_float64_le_base64: str | None
        inline_sample_count: int | None
        patch_ref_sha256: str
    @dataclass(frozen=True)
    class SplitSeamProfileEvidence:
        local_axis: str
        period_pixels: float
        profile_samples_float64_le_base64: str
        profile_weights_float64_le_base64: str
        axis_bin_indices_int64_le_base64: str
        profile_samples_encoding: str
        profile_weights_encoding: str
        axis_bin_indices_encoding: str
        sample_count: int
        weight_count: int
        index_count: int
        coverage: dict
        payload_sha256: str
        d_profile_sha256: str
        D_a_sha256: str
        seam_evidence_sha256: str
    @dataclass(frozen=True)
    class LatticeReading:
        patch_ref: SplitPhasePatchRef
        seam_profiles: tuple
    @dataclass(frozen=True)
    class ImageDerivedLatticeResult:
        readings: tuple
    raw=b'\0'*(64*256*8)
    patch_text=base64.b64encode(raw).decode('ascii')
    profile_text=base64.b64encode(b'\0'*(1024*8)).decode('ascii')
    patch=SplitPhasePatchRef('inline',f.sha(raw),64,256,'float64',len(raw),True,
                             patch_text,64*256,'0'*64)
    profile=SplitSeamProfileEvidence('u',8.,profile_text,profile_text,profile_text,
                                    'float64-le-base64','float64-le-base64','int64-le-base64',
                                    1024,1024,1024,{},'1'*64,'2'*64,'3'*64,'4'*64)
    lattice_result=ImageDerivedLatticeResult((LatticeReading(patch,(profile,)),))
    calls=[]
    def lattice(): calls.append('lattice'); return lattice_result
    def matcher(value): calls.append('G'); return None
    module=SimpleNamespace(lattice=lattice,matcher=matcher)
    def prediction(paths,**kwargs):
        module.matcher(module.lattice()); return Outcome()
    monkeypatch.setattr(base64,'b64decode',lambda *a,**k:pytest.fail('observer must not decode payload'))
    result,storage=run(prediction,targets=((module,'lattice','lattice'),(module,'matcher','G_matcher')))
    assert result['state']=='SEALED' and calls==['lattice','G'] and f.MAX_STRING==4096
    record=native_record(storage)
    reading=record['stages']['lattice']['records'][0]['result']['fields']['readings'][0]['fields']
    patch_receipt=reading['patch_ref']['fields']['inline_samples_float64_le_base64']
    assert len(patch_text)>2*f.BLOCK and patch_receipt['sha256']==f.sha(patch_text.encode('ascii'))
    assert patch_receipt['encoded_bytes']==patch_receipt['characters']==len(patch_text)
    assert patch_receipt['field']=='inline_samples_float64_le_base64'
    assert patch_receipt['type']=='builtins.str' and patch_receipt['encoding']=='base64'
    assert patch_receipt['text_encoding']=='ascii' and patch_receipt['byte_order']=='little'
    assert patch_receipt['scalar_type']=='float64' and patch_receipt['dtype']=='<f8'
    fields=reading['seam_profiles'][0]['fields']
    for key in ('profile_samples_float64_le_base64','profile_weights_float64_le_base64','axis_bin_indices_int64_le_base64'):
        receipt=fields[key]
        assert len(profile_text)>f.MAX_STRING and receipt['sha256']==f.sha(profile_text.encode('ascii'))
        assert receipt['field']==key and receipt['characters']==receipt['encoded_bytes']==len(profile_text)
        assert receipt['type']=='builtins.str' and receipt['encoding']=='base64' and receipt['text_encoding']=='ascii'
        assert receipt['byte_order']=='little'
        assert receipt['scalar_type']==('int64' if key.startswith('axis_') else 'float64')
        assert receipt['dtype']==('<i8' if key.startswith('axis_') else '<f8')
        assert receipt['payload_policy']=='digest_only; no decoding'
    wire=storage.files[Path('/memory/stub/native-receipt.json')]
    assert patch_text.encode() not in wire and profile_text.encode() not in wire
    assert record['stages']['G_matcher']['state']=='returned'


@pytest.mark.parametrize('key',['detail','source_label','inline_samples_float64_le_base64_extra'])
def test_ordinary_or_similar_long_text_still_refuses_through_coordinator(key):
    @dataclass
    class LatticeRecord:
        details: dict
    def lattice(): return LatticeRecord({key:'A'*(f.MAX_STRING+1)})
    module=SimpleNamespace(lattice=lattice); oracle=[]
    def prediction(paths,**kwargs): module.lattice(); return Outcome()
    result,storage=run(prediction,targets=((module,'lattice','lattice'),),loader=lambda:oracle.append(1))
    assert result['state']=='SEALED_INSTRUMENTATION_FAILURE' and not oracle
    record=native_record(storage)
    assert record['instrumentation_failure']['message']=='string cap'
    assert record['stages']['G_matcher']['state']=='not_reached'


@pytest.mark.parametrize('key',['inline_samples_float64_le_base64','profile_samples_float64_le_base64',
                              'profile_weights_float64_le_base64','axis_bin_indices_int64_le_base64'])
def test_lattice_payload_caps_and_precharge_before_slice(key):
    class Untouchable(str):
        def __getitem__(self,index): pytest.fail('slice before allowance')
    with pytest.raises(f.IntegrityError,match='evidence precharge'):
        f.Snapshot(f.Budget(reserved=f.MAX_OBSERVER-1000)).encoded_payload_receipt(Untouchable('A'*8192),key)
    class Oversized(Untouchable):
        def __len__(self): return f.MAX_ARRAY+1
    with pytest.raises(f.IntegrityError,match='encoded lattice cap'):
        f.Snapshot(f.Budget()).encoded_payload_receipt(Oversized('A'),key)
    with pytest.raises(f.IntegrityError,match='encoded lattice alphabet'):
        f.Snapshot(f.Budget()).take('!',key)
    assert f.Snapshot(f.Budget()).take(None,key) is None


def test_encoded_field_whitelist_is_exactly_the_five_authorized_names():
    assert set(f.ENCODED_PAYLOAD_FIELDS)=={'packed_mask_base64','inline_samples_float64_le_base64',
        'profile_samples_float64_le_base64','profile_weights_float64_le_base64','axis_bin_indices_int64_le_base64'}
    with pytest.raises(f.IntegrityError,match='encoded payload field'):
        f.Snapshot(f.Budget()).encoded_payload_receipt('AAAA','arbitrary_base64')


@pytest.mark.parametrize('mismatch_slot',[None,0,1],ids=['correct','first-mismatch','second-mismatch'])
def test_actual_decoder_byte_receipts_before_original_delegation(monkeypatch,mismatch_slot):
    packets=(b'stub ordered PNG zero',b'stub ordered PNG one')
    monkeypatch.setattr(f,'INPUT_HASHES',tuple(f.sha(x) for x in packets))
    calls=[]; public=[]; oracle=[]
    def decoder(data,*,suffix): calls.append((data,suffix)); return 'stub pixels'
    module=SimpleNamespace(decoder=decoder)
    def prediction(paths,**kwargs):
        public.append(1)
        for slot,data in enumerate(packets):
            module.decoder(b'changed ingress' if slot==mismatch_slot else data,suffix='.png')
        return Outcome()
    result,storage=run(prediction,targets=((module,'decoder','decode'),),loader=lambda:oracle.append(1))
    record=native_record(storage)
    assert public==[1]  # decoder lives inside sole path-only call; no extra pre-read
    if mismatch_slot is None:
        assert result['state']=='SEALED' and calls==[(x,'.png') for x in packets], record['instrumentation_failure']
        assert oracle==[1] and len(record['actual_decoder_inputs'])==2
        assert all(x['delegated'] for x in record['actual_decoder_inputs'])
    else:
        assert result['state']=='SEALED_INSTRUMENTATION_FAILURE' and not oracle
        assert len(calls)==mismatch_slot
        assert record['actual_decoder_inputs'][-1]['delegated'] is False
        assert record['actual_decoder_inputs'][-1]['sha256']==f.sha(b'changed ingress')
        assert record['stages']['decode']['state']=='instrumentation_failure'
    assert [x['path'] for x in record['actual_decoder_inputs']]==list(f.INPUTS[:len(record['actual_decoder_inputs'])])
    assert module.decoder is decoder


@pytest.mark.parametrize('pin',['manifest','evaluator','tests'])
@pytest.mark.parametrize('native_raises',[False,True],ids=['returned','native-error'])
def test_postrun_pin_drift_blocks_success_and_metadata(monkeypatch,pin,native_raises):
    changed=False; calls=[]; oracle=[]
    original_manifest=f.source_manifest; original_hash=f.file_hash
    def manifest():
        value=original_manifest()
        if changed and pin=='manifest': value=({**value[0],'sha256':'f'*64},*value[1:])
        return value
    def file_hash(path,limit):
        value=original_hash(path,limit)
        if changed and ((pin=='evaluator' and Path(path)==PATH) or (pin=='tests' and Path(path)==Path(__file__))):
            return {**value,'sha256':'f'*64}
        return value
    monkeypatch.setattr(f,'source_manifest',manifest); monkeypatch.setattr(f,'file_hash',file_hash)
    def prediction(paths,**kwargs):
        nonlocal changed
        calls.append(1); changed=True
        if native_raises: raise ValueError('original native error')
        return Outcome()
    result,storage=run(prediction,loader=lambda:oracle.append(1))
    assert calls==[1] and result['state']=='SEALED_INSTRUMENTATION_FAILURE' and not oracle
    record=native_record(storage)
    assert record['terminal']['kind']=='instrumentation_failure'
    assert f.strict_json(storage.files[Path('/memory/stub/COMPLETE.json')])['native_complete'] is False


def test_source_shaped_score_trace_winner_distinct_from_final_local_through_coordinator():
    first={'name':'winner'}; last={'name':'last'}
    def score_modes(modes,best,winner_refusal,support_floor,bounds):
        scores=[]
        for mode in modes:
            local='last_mode_refusal'
            refusal=winner_refusal if mode is best else local
            scores.append({'refusal':refusal,'passed':refusal=='' and mode is best})
        return tuple(scores)
    module=SimpleNamespace(score_modes=score_modes)
    def prediction(paths,**kwargs):
        module.score_modes([first,last],first,'winner_cheirality',12,{'cap':3}); return Outcome()
    result,storage=run(prediction,targets=((module,'score_modes','modes'),))
    assert result['state']=='SEALED'
    trace=native_record(storage)['G_and_mode_locals'][0]
    assert trace['winner_index']==0 and trace['winner_refusal']=='winner_cheirality'
    assert trace['support_floor']==12 and trace['bounds']=={'cap':3}
    assert trace['scores'][0]['refusal']=='winner_cheirality' and trace['scores'][1]['refusal']=='last_mode_refusal'
    assert 'refusal' not in trace and 'floor' not in trace


@pytest.mark.parametrize('count',[f.MAX_ITEMS+1,40000],ids=['count-limit','allowance-limit'])
def test_kept_ID_precharge_precedes_iteration_through_coordinator(count):
    class NeverIterate:
        def __len__(self): return count
        def __iter__(self): pytest.fail('kept set/list allocated before count/allowance gate')
    def matcher():
        matches=NeverIterate(); filtered=NeverIterate(); refusals=[]
        return None
    module=SimpleNamespace(matcher=matcher); oracle=[]
    def prediction(paths,**kwargs): module.matcher(); return Outcome()
    result,storage=run(prediction,targets=((module,'matcher','G_matcher'),),loader=lambda:oracle.append(1))
    assert result['state']=='SEALED_INSTRUMENTATION_FAILURE' and not oracle
    assert not native_record(storage)['G_and_mode_locals']


def test_test_scope_no_native_import_or_data_or_artifact_access():
    assert not any(n=='numpy' or n.startswith('numpy.') or n=='cv2' or n=='PIL' or
                   n.startswith('mc_photogrammetry.') for n in sys.modules)
    tree=ast.parse(Path(__file__).read_text(encoding='utf-8'))
    calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert not set(calls)&{'load_runtime','load_posthoc_metadata','derive_image_output_from_paths'}


_SOURCE_FIELD_CACHE={}
def source_shaped_record(name,**values):
    """Field names from source AST only; no native constructor/import/validation.

    Unused fields explicitly contain None. These objects test receipt coverage,
    not physical validity, source digest validity, or a camera/G outcome.
    """
    extra_modules={'CameraIntrinsics':'pipeline.contracts','CameraPose':'pipeline.contracts',
        **dict.fromkeys(('ImageDerivedSignaturesResult','SignatureEvidence','UsableMask','SignatureAbstention'),'pipeline.image_derived_signatures'),
        'ImageDerivedGraphResult':'pipeline.image_derived_graph',
        'EndpointSelectionCertificate':'pipeline.image_derived_endpoint_selection',
        'PlaneRegionFootprint':'reconstruction.lattice_planes','PlaneRectification':'reconstruction.lattice_planes',
        'DistanceAlternative':'reconstruction.lattice_planes','PlaneMatchRefusal':'reconstruction.plane_graph',
        'SplitPhaseProducerResult':'reconstruction.lattice_planes'}
    module=f.PROJECTION_MODULES.get(name,extra_modules.get(name,'pipeline.joint_matching'))
    if type(module) is tuple: module=module[1] if 'readings' in values else module[0]
    path=ROOT/'src/mc_photogrammetry'/Path(module.replace('.','/')+'.py')
    cache_key=(module,name)
    if cache_key not in _SOURCE_FIELD_CACHE:
        tree=ast.parse(path.read_text(encoding='utf-8'))
        declaration=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name==name)
        _SOURCE_FIELD_CACHE[cache_key]=tuple((node.target.id,ast.unparse(node.annotation)) for node in declaration.body if isinstance(node,ast.AnnAssign))
    declared=_SOURCE_FIELD_CACHE[cache_key]
    names=[field for field,annotation in declared]
    assert set(values)<=set(names)
    annotations=dict(declared)
    record=make_dataclass(name,[(field,annotations[field]) for field in names])
    record.__module__='mc_photogrammetry.'+module
    return record(**{field:values.get(field) for field in names})


class ProjectionBuffer(array):
    def __new__(cls,values): return super().__new__(cls,'d',values)
    @property
    def dtype(self): return SimpleNamespace(str='<f8')
    @property
    def shape(self): return (len(self),)
    @property
    def strides(self): return (8,)
    @property
    def nbytes(self): return len(self)*8
    @property
    def size(self): return len(self)
    @property
    def flags(self): return SimpleNamespace(c_contiguous=True,writeable=True)


def raw_native_record(storage):
    return f.strict_json(storage.files[Path('/memory/stub/native-receipt.json')])


def numeric_bytes(receipt):
    chunks=receipt[f.NUMERIC_VALUES_FIELD]
    assert all(len(text)<=4096 for text in chunks)
    assert all(len(text)==4096 for text in chunks[:-1])
    return b''.join(base64.b64decode(text,validate=True) for text in chunks)


class ScaleBuffer(array):
    def __new__(cls,typecode,values,shape):
        obj=super().__new__(cls,typecode,values)
        tags={'d':'<f8','f':'<f4','q':'<i8','B':'|u1'}
        obj.dtype=SimpleNamespace(str=tags[typecode],hasobject=False,fields=None,subdtype=None)
        obj.shape=tuple(shape); obj.size=len(obj); obj.nbytes=len(obj)*obj.itemsize
        obj.strides=tuple(obj.itemsize*math.prod(shape[index+1:]) for index in range(len(shape)))
        obj.flags=SimpleNamespace(c_contiguous=True,writeable=False)
        return obj


def scale_upstream(module,support,camera,calls):
    """Two fixed synthetic packets; no numerical solver or source import.

    Every mode selection is ordered and one-to-one. Different modes swap a
    different endpoint pair; all data remains independently encoded, not recipes.
    """
    left=ScaleBuffer('d',(float(i) for index in range(4000) for i in (index,index%97)),(4000,2))
    right=ScaleBuffer('d',(float(i) for index in range(4000) for i in (index+.5,index%101)),(4000,2))
    descriptors=[ScaleBuffer('f',((index%128+slot)/128 for index in range(4000*128)),(4000,128)) for slot in (0,1)]
    features=[source_shaped_record('FrameFeatures',frame_id=name,points=point,descriptors=descriptor,
        width=4096,height=4096,usable_pixels=4096*4096,
        sizes=ScaleBuffer('f',(float(2+index%17+slot) for index in range(4000)),(4000,)),boundary_dropped=slot)
        for slot,(name,point,descriptor) in enumerate(zip(('left','right'),(left,right),descriptors))]
    edge_pairs=[(index,index) for index in range(4000)]+[(index,index^1) for index in range(4000)]+[(index,(index+2)%4000) for index in range(192)]
    edges=ScaleBuffer('q',(n for pair in edge_pairs for n in pair),(8192,2))
    distances=ScaleBuffer('d',(index/8192 for index in range(8192)),(8192,))
    graph=source_shaped_record('ReciprocalGraph',edges=edges,distances=distances,
        left_keypoints=4000,right_keypoints=4000,total_edges=8192,truncated=0)
    modes=[]
    for index in range(12):
        selected=list(range(support))
        if index:
            pair=2*(index-1); selected.remove(pair); selected.remove(pair+1)
            selected.extend((4000+pair,4000+pair+1)); selected.sort()
        pairs=[edge_pairs[row] for row in selected]
        assert len({a for a,b in pairs})==len({b for a,b in pairs})==support
        matches=ScaleBuffer('d',(n for a,b in pairs for n in (left[2*a],left[2*a+1],right[2*b],right[2*b+1])),(support,4))
        selection=ScaleBuffer('q',selected,(support,))
        modes.append(source_shaped_record('_Mode',yaw_class=index//3,mode_index=index%3,support=support,
            matches=matches,selection=selection,spatial_bins_left=16,spatial_bins_right=16,
            median_selected_residual=.1,descriptor_cost=.2,median_all_edge_residual=2.,
            parallax=4.,cheirality=.8,singular_ratio=.2,baseline_direction=(1.,0.,0.)))
    bounds=source_shaped_record('JointMatchSettings',neighbors=5,max_candidate_edges=8192,max_features=4000,
        max_modes_per_class=3,spatial_grid=(4,4),min_support_absolute=8,min_support_fraction=.02)
    invoked={'left':0,'right':0,'graph':0,'modes':0,'pair':0,'camera':0}
    def features_stage(frame_id,rgb,**kwargs):
        invoked[frame_id]+=1; calls.append('features'); return features[0 if frame_id=='left' else 1]
    def graph_stage(left_descriptors,right_descriptors,**kwargs):
        invoked['graph']+=1; calls.append('reciprocal_graph')
        assert left_descriptors is descriptors[0] and right_descriptors is descriptors[1]; return graph
    def score_modes(modes,best,winner_refusal,support_floor,bounds):
        invoked['modes']+=1; calls.append('modes'); scores=[]
        for mode in modes:
            scores.append(source_shaped_record('HypothesisScore',yaw_class=mode.yaw_class,mode_index=mode.mode_index,
                support=mode.support,spatial_bins_left=16,spatial_bins_right=16,spatial_bins=16,
                median_selected_residual_degrees=.1,descriptor_cost=.2,median_all_edge_residual_degrees=2.,
                parallax_degrees=4.,cheirality_fraction=.8,singular_ratio=.2,baseline_direction=(1.,0.,0.),
                passed=False,refusal=winner_refusal if mode is best else 'other_toy_refusal'))
        return tuple(scores)
    def pair_stage(left_id,left_rgb,right_id,right_rgb,**kwargs):
        invoked['pair']+=1; calls.append('pair')
        l=module.features(left_id,left_rgb); r=module.features(right_id,right_rgb)
        g=module.graph(l.descriptors,r.descriptors,max_edges=8192)
        scores=module.modes(modes,modes[0],'toy_refusal',8,bounds)
        return source_shaped_record('JointPairResult',left_id=left_id,right_id=right_id,status='refused',reason='toy_refusal',
            pose=None,right_yaw_class=None,matches=modes[0].matches,raw_keypoints=(4000,4000),boundary_dropped=(0,0),
            candidate_edges=8192,total_graph_edges=8192,truncated=0,ratio_survivors=(4000,4000),mutual_survivors=4000,
            support=support,support_floor=8,spatial_bins_left=16,spatial_bins_right=16,spatial_bins_total=16,
            median_selected_residual_degrees=.1,descriptor_cost=.2,parallax_degrees=4.,cheirality_fraction=.8,
            singular_ratio=.2,ambiguity_margin=0,hypotheses=scores)
    def camera_stage(frames):
        invoked['camera']+=1; calls.append('auto_camera')
        module.pair('left','toy pixels','right','toy pixels',left_camera=camera['left'],right_camera=camera['right'])
        return camera
    module.features=features_stage; module.graph=graph_stage; module.modes=score_modes
    module.pair=pair_stage; module.camera=camera_stage
    targets=tuple((module,name,stage) for name,stage in (('features','features'),('graph','reciprocal_graph'),
        ('modes','modes'),('pair','pair'),('camera','auto_camera')))
    return (features,graph,modes),targets,invoked


@pytest.mark.parametrize('mode_support,g_envelope',[(None,None),(400,None),(4000,None),
    (400,(760,0,'kept')),(400,(0,760,'refused')),(400,(760,0,'dropped')),(400,(162,760,'mixed')),
    (400,(760,0,'pressure')),(400,(162,598,'geometry-refusal'))],
    ids=['G-only','representative-support','default-upper-support','760-kept','760-refused','760-dropped','mixed-nonempty','760-dropped-upstream-pressure','geometry-refusal-pressure'])
def test_source_shaped_repeated_definitions_reach_G_solver_terminal_and_account_every_stage(capsys,monkeypatch,mode_support,g_envelope):
    points=ProjectionBuffer([float(i) for i in range(160)])
    cameras={name:source_shaped_record('CameraEstimate',
        intrinsics=source_shaped_record('CameraIntrinsics',image_size=(4096,4096),
            focal_length_pixels=(2000.+index,2000.+index),principal_point_pixels=(2048.,2048.)),
        pose=source_shaped_record('CameraPose',position=(float(index),1.62,0.),yaw_degrees=float(index*90),pitch_degrees=0.,roll_degrees=0.),
        residual_pixels=.1+index*.1,confidence=.8+index*.01) for index,name in enumerate(('left','right'))}
    frames=tuple(source_shaped_record('ImageDerivedFrame',frame_id=frame,source_label='toy',
        image=points,image_shape=(160,),image_sha256='0'*64) for frame in ('left','right'))
    session=source_shaped_record('ImageDerivedSession',schema='toy',version=1,frames=frames,
        cameras=cameras,components=(('left','right'),),unresolved=(),
        scale_known=False,local_cameras={},session_sha256='1'*64)
    entry=source_shaped_record('ImageDerivedEntryResult',session=session)
    reading_count=36 if g_envelope is None else 9
    readings=tuple(source_shaped_record('PlaneReading',frame=('left' if i%2==0 else 'right') if g_envelope is None else ('left' if i<6 else 'right'),
        axis=i%3,side=float(i),alternatives=(),period_pixels=8.,phase_pixels=(0.,0.),extent=8.,samples=64,
        polygon=((0.,0.),(8.,0.),(8.,8.),(0.,8.)),coverage_cells=64,structural_share=.8,confidence=.9)
        for i in range(reading_count))
    for index,reading in enumerate(readings):
        region_id=f.sha(('toy region '+str(index)).encode())
        reading.region_evidence=source_shaped_record('PlaneRegionEvidence',source_region_id=region_id,
            axis=reading.axis,sources=('toy proposal '+str(index),),image_shape=(4096,4096),bbox_xyxy=(0,0,8,8),
            crop_shape=(8,8),encoding='base64',bit_order='little',packed_mask_base64='//////////8=',
            mask_sha256=f.sha(b'\xff'*8),pixel_count=64)
        reading.phase_evidence=source_shaped_record('PeriodicEnergyPhaseEvidence',phase_uv_pixels=(0.,0.),
            period_pixels=8.,patch_centre_uv=(0.,0.),extent=8.,samples=64,response_magnitude_uv=(1.+index,1.),
            profile_mass_uv=(64.,64.),concentration_uv=(.8,.9),unavailable_reason_uv=(None,None),source_region_id=region_id)
        reading.rectification=source_shaped_record('PlaneRectification',centre=(0.,0.),
            axis_matrix=((1.,0.,0.),(0.,1.,0.),(0.,0.,1.)),focal=2000.+index,principal_point=(2048.,2048.))
        alternatives=2 if g_envelope is None else [1,2,7,7,1,1,2,5,3][index]
        reading.alternatives=tuple(source_shaped_record('DistanceAlternative',blocks_per_period=float(slot+1),
            distance_blocks=float(index+slot+2),support=.8-slot*.01) for slot in range(alternatives))
    signature_readings=readings if g_envelope is None else tuple(reading for reading in readings for _ in reading.alternatives)
    signatures=tuple(source_shaped_record('PlaneSignature',reading=reading,blocks_per_period=1.,
        cells=ScaleBuffer('f',(float(index+slot)/64 for slot in range(64)),(8,8)),
        interior=ScaleBuffer('f',(float(index+slot+1)/128 for slot in range(64)),(8,8)),
        coverage=ScaleBuffer('B',((index+slot)%2 for slot in range(64)),(8,8)),origin_uv=(float(index),0.),
        region_footprint=source_shaped_record('PlaneRegionFootprint',source_region_id=reading.region_evidence.source_region_id,
            blocks_per_period=1.,origin_uv=(float(index),0.),grid_shape=(8,8),subsample_factor=2,
            inside_region_sample_counts=bytes((index+slot)%5 for slot in range(64)),
            valid_sample_counts=b'\x04'*64,total_samples_per_cell=4)) for index,reading in enumerate(signature_readings))
    split_sources=[]
    if g_envelope is not None:
        for index,reading in enumerate(readings):
            payload=base64.b64encode(bytes(8*8*8)).decode('ascii')
            patch=source_shaped_record('SplitPhasePatchRef',mode='inline',patch_sha256=f.sha(str(index).encode()),
                rows=8,columns=8,dtype='float64',byte_length=512,finite_nonnegative=True,
                inline_samples_float64_le_base64=payload,inline_sample_count=64,patch_ref_sha256=f.sha(('ref'+str(index)).encode()))
            profiles=tuple(source_shaped_record('SplitSeamProfileEvidence',local_axis=axis,period_pixels=8.,
                profile_samples_float64_le_base64=payload,profile_weights_float64_le_base64=payload,
                axis_bin_indices_int64_le_base64=payload,profile_samples_encoding='float64-le-base64',
                profile_weights_encoding='float64-le-base64',axis_bin_indices_encoding='int64-le-base64',
                sample_count=64,weight_count=64,index_count=64,coverage={'total':64,'valid':64,'per_bin':(1,)*64},
                payload_sha256=f.sha((str(index)+axis).encode()),d_profile_sha256='2'*64,D_a_sha256='3'*64,
                seam_evidence_sha256=f.sha(('seam'+str(index)+axis).encode())) for axis in ('u','v'))
            producer=source_shaped_record('SplitPhaseProducerResult',status='complete',producer_version_hex='00',
                producer_version_sha256='4'*64,reading_identity={'frame':reading.frame,'plane_index':index,'reading_axis':reading.axis},
                patch_ref=patch,seam_profile_evidence=profiles,refusals=(),result_sha256='5'*64)
            split_sources.append(source_shaped_record('SplitSourceEvidence',frame=reading.frame,canonical_index=index,
                reading_axis=reading.axis,result=producer,patch_float64_le=bytes(512),patch_rows=8,patch_columns=8,
                source_sha256=f.sha(('source'+str(index)).encode())))
    bridge=source_shaped_record('GeometryProvenanceBridge',reading_universe_sha256='6'*64,readings=readings,
        structural_weights=(),split_sources=tuple(split_sources),bridge_sha256='7'*64) if split_sources else None
    lattice=source_shaped_record('ImageDerivedLatticeResult',readings=readings,session_sha256='1'*64,bridge=bridge)
    connection=source_shaped_record('ImageDerivedConnectionGraph')
    pose=source_shaped_record('ImageDerivedPoseGraph')
    solution=source_shaped_record('ComponentSolution',frames=('left','right'),quadrants={'left':0,'right':0},
        translations={'left':(0.,0.,0.),'right':(1.,0.,0.)},axis_residuals={0:0.},
        axis_resultants={0:1.},scale_known=False,blocks_per_period={('left',0):1.},
        reason='toy_abstention',alternatives=())
    geometry_refusal=g_envelope is not None and g_envelope[2]=='geometry-refusal'
    if geometry_refusal: solution.reason='ambiguous_quadrant'
    row_count=4 if g_envelope is None else g_envelope[0]
    rows=[source_shaped_record('PlaneMatch',left='left',right='right',left_axis=i%3,right_axis=i%3,
        left_b=1.,right_b=1.,quadrant_of_right=0,psr=2.+i,peak=5.,second_peak=1.,overlap_cells=64,
        integer_residual=None,score=float(i),left_plane=i%6,right_plane=i%3,cell_shift_rc=(i,0)) for i in range(row_count)]
    # Equal-valued independent candidates stay distinct in ordinal identity/order.
    if g_envelope is None:
        rows.append(source_shaped_record('PlaneMatch',**{field.name:getattr(rows[0],field.name)
            for field in rows[0].__dataclass_fields__.values()}))
    elif rows:
        rows[-1]=source_shaped_record('PlaneMatch',**{field.name:getattr(rows[0],field.name)
            for field in rows[0].__dataclass_fields__.values()})
    calls=[]; refusals=[]
    if g_envelope is not None:
        refusals.extend(source_shaped_record('PlaneMatchRefusal',left='left',right='right',left_axis=i%3,
            right_axis=i%3,quadrant_of_right=i%4,reason='toy_correlation_refused',detail=str(i)) for i in range(g_envelope[1]))
    incoming_refusals=len(refusals)
    observed={}; retained_pressure=[]
    if g_envelope is not None and g_envelope[2] in ('pressure','geometry-refusal'):
        original_init=f.Observer.__init__
        def observer_init(self,*args,**kwargs):
            original_init(self,*args,**kwargs); observed['observer']=self
        monkeypatch.setattr(f.Observer,'__init__',observer_init)
    expected_indices=[4,0,2] if g_envelope is None else (list(range(len(rows))) if g_envelope[2]=='kept'
        else list(range(12)) if geometry_refusal else list(range(0,len(rows),3)) if g_envelope[2]=='mixed' else [])
    def session_stage():
        calls.append('session')
        if mode_support is not None: module.camera(frames)
        cameras['right'].confidence=.95  # a changed source-record version after camera return
        return entry
    def connections_stage(arg): assert arg is entry; calls.append('connections'); return connection
    def pose_stage(arg,graph): assert arg is entry and graph is connection; calls.append('pose_graph'); return pose
    def lattice_stage(arg,graph): assert arg is entry and graph is pose; calls.append('lattice'); return lattice
    reading_index={(reading.frame,(index//2 if g_envelope is None else index if index<6 else index-6)):reading
        for index,reading in enumerate(readings)}
    def prefilter(matches,reading_index_arg,refusals):
        calls.append('G_prefilter'); assert matches is rows and reading_index_arg is reading_index
        if g_envelope is None: refusals.append({'reason':'same_axis_conflict','candidate_index':1})
        else:
            kept_indices=set(expected_indices)
            refusals.extend(source_shaped_record('PlaneMatchRefusal',left=match.left,right=match.right,left_axis=match.left_axis,
                right_axis=match.right_axis,quadrant_of_right=match.quadrant_of_right,reason='same_axis_mismatch',detail='drop'+str(i))
                for i,match in enumerate(matches) if i not in kept_indices)
        return [matches[index] for index in expected_indices]
    def matcher(readings_arg,signatures_arg,*,max_shift_cells,refusals):
        calls.append('G_matcher'); assert readings_arg is readings and signatures_arg is signatures and max_shift_cells==32
        matches=rows
        filtered=module.prefilter(matches,reading_index,refusals)
        signatures_arg[0].cells[0]+=.5  # argument-after gets a changed array/record version
        return filtered
    def solver(readings_arg,matches,*,integer_tolerance,min_resultant,ambiguous_margin,quadrant_constraints):
        calls.append('G_solver'); assert readings_arg is readings and matches==[rows[index] for index in expected_indices]
        assert integer_tolerance==.05 and min_resultant==.9 and ambiguous_margin==.02
        assert quadrant_constraints==(('left','right',0),); return solution
    def geometry_stage(arg=None,measured=None,**kwargs):
        if geometry_refusal:
            assert kwargs=={'session':session,'lattice_result':lattice,'connection_graph':connection}
            arg=entry; measured=lattice
        calls.append('geometry'); assert arg is entry and measured is lattice
        if observed and not geometry_refusal:
            observer=observed['observer']; prior=observer.budget.pool_counts['upstream']
            # Truthful capacity stress: allocate and keep the charged neutral
            # slot/buffer objects. This is not a claimed replay of real payloads.
            extra_items=58921-prior['items']; extra_bytes=6974960-prior['bytes']
            assert extra_items>8 and extra_bytes>8*extra_items+512
            with observer.budget.scope('session'):
                observer.budget.charge(extra_bytes,extra_items)
            slots=[None]*(extra_items-8)
            buffer=bytearray(extra_bytes-sys.getsizeof(slots)-512)
            assert sys.getsizeof(slots)+sys.getsizeof(buffer)+sys.getsizeof(retained_pressure)+256<=extra_bytes
            retained_pressure.extend((slots,buffer))
            assert observer.budget.pool_counts['upstream']=={'items':58921,'bytes':6974960}
        filtered=module.matcher(readings,signatures,max_shift_cells=32,refusals=refusals)
        solved=module.solver(readings,filtered,integer_tolerance=.05,min_resultant=.9,
            ambiguous_margin=.02,quadrant_constraints=(('left','right',0),))
        if geometry_refusal:
            evidence=tuple(source_shaped_record('SignatureEvidence',frame_id=sig.reading.frame,
                canonical_index=index%reading_count,reading_axis=sig.reading.axis,alternative_index=index,
                blocks_per_period=sig.blocks_per_period,signature=sig,signature_sha256=f.sha(('signature '+str(index)).encode()))
                for index,sig in enumerate(signatures))
            masks=tuple(source_shaped_record('UsableMask',frame_id=name,height=8,width=8,usable_bool_bytes=bytes(64),mask_sha256='1'*64)
                for name in ('left','right'))
            f1=source_shaped_record('ImageDerivedSignaturesResult',schema='mc-photogrammetry-image-derived-signatures',producer='toy',
                terminal='geometry_unavailable',status='assembled',reason='',detail='',session_sha256='1'*64,lattice_sha256='2'*64,
                frame_ids=('left','right'),usable_masks=masks,signatures=evidence,abstentions=(),masks_sha256='3'*64,
                signatures_sha256='4'*64,result_sha256='5'*64)
            graph_result=source_shaped_record('ImageDerivedGraphResult',schema='mc-photogrammetry-image-derived-graph',producer='toy',
                terminal='geometry_unavailable',status='unavailable',reason=solved.reason,detail='component solver abstained',
                session_sha256='1'*64,lattice_sha256='2'*64,signatures_sha256='5'*64,frame_ids=('left','right'),
                readings=readings,signatures=evidence,matches=(),component_solution=None,correlation_algorithm='toy',max_shift_cells=32,result_sha256='6'*64)
            value=source_shaped_record('ImageDerivedGeometryResult',schema='mc-photogrammetry-image-derived-geometry',producer='toy',
                terminal='geometry_unavailable',status='unavailable',reason='g_unavailable',detail=solved.reason,
                session_sha256='1'*64,lattice_sha256='2'*64,stage_digests=(('f1','5'*64),('g','6'*64)),records=(('f1',f1),('g',graph_result)),
                placement_records=(),universe=None,supports=(),result_sha256='7'*64)
            observed['expected_geometry']=frozen_record_body(value)
            return value
        return source_shaped_record('ImageDerivedGeometryResult',reason='toy',status='geometry_unavailable')
    def output_stage(arg,measured,geometry):
        calls.append('output'); assert arg is entry and measured is lattice
        return source_shaped_record('ImageDerivedOutputResult',status='geometry_unavailable',reason='toy')
    module=SimpleNamespace(session=session_stage,connections=connections_stage,pose=pose_stage,lattice=lattice_stage,
        geometry=geometry_stage,matcher=matcher,prefilter=prefilter,solver=solver,output=output_stage)
    targets=tuple((module,name,stage) for name,stage in (
        ('session','session'),('connections','connections'),('pose','pose_graph'),('lattice','lattice'),
        ('geometry','geometry'),('matcher','G_matcher'),('prefilter','G_prefilter'),('solver','G_solver'),('output','output')))
    if mode_support is not None:
        upstream,upstream_targets,upstream_calls=scale_upstream(module,mode_support,cameras,calls)
        targets+=upstream_targets
    public=[]
    def prediction(paths,**kwargs):
        public.append(1)
        arg=module.session(); graph=module.connections(arg); posed=module.pose(arg,graph)
        measured=module.lattice(arg,posed)
        if geometry_refusal:
            observer=observed['observer']; prior=observer.budget.pool_counts['upstream']
            extra_items=59183-prior['items']; extra_bytes=7005912-prior['bytes']
            assert extra_items>8 and extra_bytes>8*extra_items+512
            observer.budget.charge(extra_bytes,extra_items)
            slots=[None]*(extra_items-8); buffer=bytearray(extra_bytes-sys.getsizeof(slots)-512)
            assert sys.getsizeof(slots)+sys.getsizeof(buffer)+sys.getsizeof(retained_pressure)+256<=extra_bytes
            retained_pressure.extend((slots,buffer))
            assert observer.budget.pool_counts['upstream']=={'items':59183,'bytes':7005912}
            geometry=module.geometry(session=session,lattice_result=lattice,connection_graph=connection)
            assert geometry.status=='unavailable'; output=None
        else:
            geometry=module.geometry(arg,measured); output=module.output(arg,measured,geometry)
        return source_shaped_record('ImageDerivedProductionResult',status='geometry_unavailable',
            reason='toy',entry=arg,lattice=measured,geometry=geometry,output=output)
    result,storage=run(prediction,targets=targets)
    raw=raw_native_record(storage); record=native_record(storage)
    if result['state']!='SEALED':
        def sort_items(node):
            if type(node) is dict: return len(node)+sum(sort_items(value) for value in node.values())
            if type(node) is list: return sum(sort_items(value) for value in node)
            return 0
        cost=sum(sort_items(definition['projection'])+sort_items(definition['values_projection'].get(f.NUMERIC_VALUES_FIELD))
            if definition['kind']=='array' else sort_items(definition['projection']) for definition in raw['semantic_definitions'].values())
        print('SYNTHETIC_SCALE_FAILURE='+json.dumps({'support':mode_support,'pools':raw['allocation_pool_counts'],
            'stages':raw['stage_budget_counts'],'state':{k:v['state'] for k,v in raw['stages'].items()},
            'definitions':len(raw['semantic_definitions']),'terminal':raw['terminal'],'definition_sort_items':cost},sort_keys=True))
    assert result['state']=='SEALED',record['instrumentation_failure']
    assert public==[1] and len(calls)==(14 if geometry_refusal else 9 if mode_support is None else 15)
    assert set(calls)=={stage for _,_,stage in targets if not geometry_refusal or stage!='output'}
    assert all(raw['stages'][stage]['calls']==(0 if geometry_refusal and stage=='output' else 2 if stage=='features' else 1) for _,_,stage in targets)
    assert raw['terminal']['kind']=='returned' and raw['schema'].endswith('-v2')
    assert len(raw['semantic_definitions'])>(200 if mode_support is None else 250)
    entry_ref=raw['stages']['session']['records'][0]['result']
    assert raw['stages']['lattice']['records'][0]['arguments'][0]==entry_ref
    if geometry_refusal:
        projected=raw['stages']['geometry']['records'][0]['result']; definition=raw['semantic_definitions'][projected['semantic_reference']]
        assert legacy_logical_wire(definition['projection'],raw['semantic_definitions'],definition['encoding'])==observed['expected_geometry']
        assert definition['content_sha256']==frozen_canonical_digest(observed['expected_geometry'])
        assert raw['stages']['output']['state']=='not_reached' and raw['stages']['output']['records']==[]
        assert record['stages']['geometry']['records'][0]['result']['fields']['detail']=='ambiguous_quadrant'
        assert len(record['stages']['geometry']['records'][0]['result']['fields']['records'][0][1]['fields']['signatures'])==29
        assert raw['stages']['geometry']['records'][0]['kwargs']==raw['stages']['geometry']['records'][0]['kwargs_after']
    else: assert raw['stages']['output']['records'][0]['arguments'][0]==entry_ref
    pre=record['stages']['G_prefilter']['records'][0]
    projected_refusals=[{'type':type(value).__module__+'.'+type(value).__qualname__,
        'fields':{name:getattr(value,name) for name in value.__dataclass_fields__}} if is_dataclass(value) else value for value in refusals]
    assert pre['arguments'][2]==projected_refusals[:incoming_refusals]
    assert pre['arguments_after'][2]==projected_refusals and pre['kwargs']=={}
    trace=record['G_and_mode_locals'][-1]
    assert trace['kept_candidate_indices']==expected_indices
    assert trace['dispositions']==['kept' if index in expected_indices else 'dropped' for index in range(len(rows))]
    assert len(set(trace['candidate_identity']))==len(rows)
    if rows: assert trace['matches'][0]==trace['matches'][-1]
    assert pre['ordered_argument_identity']['0']==trace['candidate_identity']
    matcher_record=raw['stages']['G_matcher']['records'][0]
    before_arguments=raw['semantic_definitions'][matcher_record['arguments']['semantic_reference']]['projection']['items']
    after_arguments=raw['semantic_definitions'][matcher_record['arguments_after']['semantic_reference']]['projection']['items']
    assert before_arguments!=after_arguments
    solver_raw=raw['stages']['G_solver']['records'][0]
    assert matcher_record['ordered_argument_identity']['0']==solver_raw['ordered_argument_identity']['0']
    signature_ids=record['stages']['G_matcher']['records'][0]['ordered_argument_identity']['signature_reading:1']
    reading_ids=record['stages']['G_matcher']['records'][0]['ordered_argument_identity']['0']
    assert set(signature_ids)<=set(reading_ids) and len(signature_ids)==len(signatures)
    assert len(set(reading_ids))==len(readings)
    assert trace['refusals']==projected_refusals
    if split_sources:
        carried=record['stages']['lattice']['records'][0]['result']['fields']['bridge']['fields']['split_sources']
        assert len(carried)==9
        assert all(len(item['fields']['result']['fields']['seam_profile_evidence'])==2 for item in carried)
    solver_record=record['stages']['G_solver']['records'][0]
    assert solver_record['arguments'][1]==trace['filtered']
    assert solver_record['kwargs']=={'integer_tolerance':.05,'min_resultant':.9,
        'ambiguous_margin':.02,'quadrant_constraints':[['left','right',0]]}
    assert solver_record['result']['fields']['reason']==('ambiguous_quadrant' if geometry_refusal else 'toy_abstention')
    for definition in raw['semantic_definitions'].values():
        if definition['kind']=='record':
            assert definition['content_sha256']==f.digest(legacy_logical_wire(definition['projection'],
                raw['semantic_definitions'],definition.get('encoding')))
    stages=raw['stage_budget_counts']; pools=raw['allocation_pool_counts']
    assert sum(row['items'] for row in stages.values())==sum(row['items'] for row in pools.values())==raw['observer_items']
    assert sum(row['bytes'] for row in stages.values())==sum(row['bytes'] for row in pools.values())==raw['observer_reserved_bytes']
    assert sum(raw['projection_coverage']['validation_walk_counts'].values())==raw['projection_coverage']['validation_visits']
    assert {'native_digest','projected_digest','projected_keys','projected_key_tokens','references'}<=set(raw['projection_coverage']['validation_walk_counts'])
    assert sum(row['items'] for row in raw['allocation_plan'].values())==100000
    assert sum(row['bytes'] for row in raw['allocation_plan'].values())==15*1024*1024
    for pool,count in pools.items(): assert all(count[key]<=raw['allocation_plan'][pool][key] for key in ('items','bytes'))
    if observed and not geometry_refusal:
        assert pools['upstream']=={'items':58921,'bytes':6974960} and len(retained_pressure)==2
        assert pools['G']['items']>0 and pools['G']['bytes']>0
    if mode_support is not None:
        assert upstream_calls=={'left':1,'right':1,'graph':1,'modes':1,'pair':1,'camera':1}
        assert record['stages']['auto_camera']['records'][0]['result']['right']['fields']['confidence']==.81
        assert record['stages']['session']['records'][0]['result']['fields']['session']['fields']['cameras']['right']['fields']['confidence']==.95
        assert all(feature['fields']['sizes']['shape']==[4000]
            for feature in [record['stages']['features']['records'][slot]['result'] for slot in range(2)])
        modes_trace=record['G_and_mode_locals'][0]
        assert len(modes_trace['modes'])==12 and modes_trace['winner_index']==0 and modes_trace['winner_refusal']=='toy_refusal'
        assert len(modes_trace['scores'])==12 and modes_trace['support_floor']==8
        for item in modes_trace['modes']:
            fields=item['fields']; assert fields['support']==mode_support
            assert len(numeric_bytes(fields['matches']))==mode_support*4*8
            assert len(numeric_bytes(fields['selection']))==mode_support*8
        for definition in raw['semantic_definitions'].values():
            if definition['kind']=='array' and f.NUMERIC_VALUES_FIELD in definition['values_projection']:
                assert f.NUMERIC_VALUES_FIELD not in definition['projection']  # no tiny duplication
                numeric=numeric_bytes(definition['values_projection'])
                assert len(numeric)==definition['projection']['bytes'] and f.sha(numeric)==definition['projection']['sha256']
    print('SYNTHETIC_STAGE_LEDGER='+json.dumps({'mode_support':mode_support,'G_envelope':g_envelope,'stages':stages,'pools':pools,
        'items':raw['observer_items'],'bytes':raw['observer_reserved_bytes'],
        'definitions':len(raw['semantic_definitions']),'validation_visits':raw['projection_coverage']['validation_visits'],
        'validation_walk_counts':raw['projection_coverage']['validation_walk_counts'],
        'headroom':{pool:{key:raw['allocation_plan'][pool][key]-count[key] for key in ('items','bytes')}
                    for pool,count in pools.items()}},sort_keys=True))
    captured=capsys.readouterr().out
    assert 'SYNTHETIC_STAGE_LEDGER=' in captured
    sys.stdout.write(captured)


def test_definition_content_mutation_alias_order_and_namespace_controls():
    observer=f.Observer(); projection=observer.projection
    reading=source_shaped_record('PlaneReading',frame='left',axis=0,side=1.)
    equal=source_shaped_record('PlaneReading',frame='left',axis=0,side=1.)
    before=projection.take(reading); assert projection.take(equal)==before
    assert observer.identity(reading)!=observer.identity(equal)
    reading.side=2.; after=projection.take(reading)
    assert after!=before and projection.take(equal)==before
    assert projection.take([reading,equal])==[after,before]
    assert projection.take([equal,reading])==[before,after]
    @dataclass
    class PlaneReading:
        frame: str
    assert 'semantic_reference' not in projection.take(PlaneReading('impostor'))
    projection.validate([before,after])
    projected_fields=projection.definitions[before['semantic_reference']]['projection']['fields']['native_mapping_entries']
    next(row for row in projected_fields if row['key']=='side')['value']=99
    with pytest.raises(f.IntegrityError,match='semantic definition changed'): projection.validate()


def test_undefined_reference_and_source_cycle_fail_closed():
    projection=f.Observer().projection
    with pytest.raises(f.IntegrityError,match='undefined semantic reference'):
        projection.validate([{'semantic_reference':'record:'+'0'*64}])
    record=source_shaped_record('PlaneReading'); record.region_evidence=record
    with pytest.raises(f.IntegrityError,match='semantic record cycle'): projection.take(record)


def test_array_hash_first_coordinate_upgrade_and_mutation_consistency():
    projection=f.Observer().projection; values=ProjectionBuffer([float(i) for i in range(100)])
    first=projection.take(values,'descriptors'); definition=projection.definitions[first['semantic_reference']]
    assert f.NUMERIC_VALUES_FIELD not in definition['values_projection']
    assert projection.take(values,'points')==first and numeric_bytes(definition['values_projection'])==values.tobytes()
    values[0]=111.; assert projection.take(values,'points')!=first
    projection.validate()
    definition['values_projection'][f.NUMERIC_VALUES_FIELD][0]='A'*len(definition['values_projection'][f.NUMERIC_VALUES_FIELD][0])
    with pytest.raises(f.IntegrityError,match='semantic array values changed'): projection.validate()
    class ChangingBuffer(ProjectionBuffer):
        @property
        def flags(self):
            self[0]+=1; return SimpleNamespace(c_contiguous=True,writeable=True)
    with pytest.raises(f.IntegrityError,match='array changed during projection'):
        f.Observer().projection.take(ChangingBuffer([1.,2.]),'points')


@pytest.mark.parametrize('pool,stage',[('upstream','lattice'),('G','G_prefilter'),('G','geometry'),
                                      ('G','output'),('G','terminal'),('terminal_failure','failure')])
def test_pool_precharge_refuses_before_native_delegation_and_preserves_other_pools(pool,stage):
    observer=f.Observer(); calls=[]
    with observer.budget.scope(stage): observer.budget.charge(0,f.ALLOCATION_PLAN[pool]['items'])
    if stage in f.STAGES:
        with pytest.raises(f.IntegrityError,match='stage item reservation'):
            observer.delegate(stage,lambda:calls.append(1))()
    else:
        with observer.budget.scope(stage),pytest.raises(f.IntegrityError,match='stage item reservation'):
            observer.projection.take('terminal')
    assert not calls
    other='G_solver' if pool=='upstream' else 'session'
    with observer.budget.scope(other): observer.budget.charge(1,1)
    assert observer.budget.items<=f.MAX_ITEMS


def test_geometry_and_output_use_existing_G_reserve_and_restore_nested_stage():
    observer=f.Observer(); budget=observer.budget
    budget.partitioned=True
    budget.pool_counts={'upstream':{'bytes':f.ALLOCATION_PLAN['upstream']['bytes'],
                                   'items':f.ALLOCATION_PLAN['upstream']['items']},
                        'G':{'bytes':0,'items':0},
                        'terminal_failure':{'bytes':0,'items':0}}
    assert budget.pool()=='upstream'
    native_calls=[]; output_token=object()
    geometry=observer.delegate('geometry',lambda value: native_calls.append(('geometry',value)) or None)
    output=observer.delegate('output',lambda value: native_calls.append(('output',value)) or output_token)
    source=object()
    assert geometry(source) is None
    assert budget.stage=='unscoped'
    assert output(source) is output_token
    assert budget.stage=='unscoped'
    assert native_calls==[('geometry',source),('output',source)]
    # Geometry and output observations share the existing G reserve; they do
    # not borrow from or reset the exhausted upstream pool.
    with budget.scope('geometry'):
        used=budget.pool_counts['G']
        budget.charge(f.ALLOCATION_PLAN['G']['bytes']-used['bytes'],
                      f.ALLOCATION_PLAN['G']['items']-used['items'])
    before=(budget.reserved,budget.items,dict(budget.pool_counts['G']))
    with budget.scope('output'),pytest.raises(f.IntegrityError,match='stage byte reservation'):
        budget.charge(1,0)
    assert budget.stage=='unscoped'
    assert (budget.reserved,budget.items,budget.pool_counts['G'])==before
    assert budget.pool_counts['upstream']=={'bytes':8*1024*1024,'items':60000}
    assert sum(row['items'] for row in budget.pool_counts.values())==95000
    assert sum(row['bytes'] for row in budget.pool_counts.values())==14*1024*1024


def test_output_postreturn_observer_failure_never_marks_native_complete(monkeypatch):
    calls=[]; posthoc=[]
    module=SimpleNamespace()
    def native_output():
        calls.append('output')
        return Outcome(status='complete',reason='stub')
    module.output=native_output
    def prediction(paths,**kwargs):
        return module.output()
    original_take=f.StageProjection.take
    def fail_on_output_result(self,value,key=''):
        if type(value) is Outcome: raise f.IntegrityError('synthetic output postreturn observer failure')
        return original_take(self,value,key)
    monkeypatch.setattr(f.StageProjection,'take',fail_on_output_result)
    result,storage=run(prediction,targets=((module,'output','output'),),
                       loader=lambda:posthoc.append('metadata'))
    receipt=f.strict_json(storage.files[Path('/memory/stub/native-receipt.json')])
    marker=f.strict_json(storage.files[Path('/memory/stub/COMPLETE.json')])
    assert calls==['output'] and module.output is native_output
    assert result['state']=='SEALED_INSTRUMENTATION_FAILURE'
    assert result['terminal']=='instrumentation_failure' and not posthoc
    assert marker['native_complete'] is False and receipt['published_project'] is None
    assert receipt['stages']['output']['state']=='instrumentation_failure'


@pytest.mark.parametrize('stage,pool',[('lattice','upstream'),('G_solver','G')])
def test_pool_byte_gate_blocks_wrapper_before_native_call(stage,pool):
    observer=f.Observer(); calls=[]
    with observer.budget.scope(stage): observer.budget.charge(f.ALLOCATION_PLAN[pool]['bytes'],0)
    with pytest.raises(f.IntegrityError,match='stage byte reservation'):
        observer.delegate(stage,lambda:calls.append(1))()
    assert not calls and observer.stages[stage]['state']=='instrumentation_failure'


def test_reserved_G_and_failure_caps_sum_without_global_budget_raise():
    observer=f.Observer()
    with observer.budget.scope('session'): observer.budget.charge(f.ALLOCATION_PLAN['upstream']['bytes'],60000)
    with observer.budget.scope('G_solver'): observer.budget.charge(f.ALLOCATION_PLAN['G']['bytes'],35000)
    with observer.budget.scope('failure'): observer.budget.charge(f.ALLOCATION_PLAN['terminal_failure']['bytes'],5000)
    assert observer.budget.items==100000 and observer.budget.reserved==15*1024*1024
    with observer.budget.scope('session'),pytest.raises(f.IntegrityError,match='item precharge'):
        observer.budget.charge(0,1)


def test_validation_work_cap_refuses_without_allocating_new_semantic_definition():
    observer=f.Observer(); reading=source_shaped_record('PlaneReading',frame='left')
    observer.projection.validation_visits=f.MAX_VALIDATION_WORK
    calls=[]
    with pytest.raises(f.IntegrityError,match='semantic validation work cap'):
        observer.delegate('G_matcher',lambda arg:calls.append(arg))(reading)
    assert not calls and not observer.projection.definitions


def test_mutable_arguments_after_native_exception_preserve_versions_and_no_retry():
    observer=f.Observer(); options={'refusals':[]}; calls=[]
    def original(arg,*,options):
        calls.append(1); arg.append('added'); options['refusals'].append('native refusal')
        raise ValueError('original failure')
    with pytest.raises(ValueError,match='original failure'):
        observer.delegate('G_solver',original)(['before'],options=options)
    record=resolve_projection(observer.stages['G_solver']['records'][0],observer.projection.definitions)
    assert calls==[1] and observer.stages['G_solver']['state']=='raised'
    assert record['arguments']==[['before']] and record['arguments_after']==[['before','added']]
    assert record['kwargs']['options']['refusals']==[]
    assert record['kwargs_after']['options']['refusals']==['native refusal']


@pytest.mark.parametrize('matching',[True,False],ids=['existing-definition-key','unknown-definition-key'])
def test_literal_reference_mapping_collision_preserved_through_coordinator(matching):
    reading=source_shaped_record('PlaneReading',frame='left',axis=0)
    observed=[]; literal=None
    def lattice(): observed.append('lattice'); return reading
    def solver(arg,*,options):
        observed.append('solver'); assert arg is reading and options is literal
        return {'literal_mapping_entries':options,'semantic_reference':options['semantic_reference']}
    module=SimpleNamespace(lattice=lattice,solver=solver)
    def prediction(paths,**kwargs):
        nonlocal literal
        arg=module.lattice()
        # The key exactly matches the key generated by the actual observer.
        lookup='record:'+f.Observer().projection.native_digest(arg)
        literal={'semantic_reference':lookup if matching else 'record:'+'0'*64}
        result=module.solver(arg,options=literal)
        return result
    result,storage=run(prediction,targets=((module,'lattice','lattice'),(module,'solver','G_solver')))
    raw=raw_native_record(storage); record=native_record(storage)
    assert result['state']=='SEALED',record['instrumentation_failure']
    assert observed==['lattice','solver']
    key=literal['semantic_reference']
    assert (key in raw['semantic_definitions']) is matching
    logical_kwargs=legacy_logical_wire(raw['stages']['G_solver']['records'][0]['kwargs'],raw['semantic_definitions'])
    options=next(row['value'] for row in logical_kwargs['native_mapping_entries']
        if row['key']=='options')
    assert options=={'native_mapping_entries':[{'key_type':'str','key':'semantic_reference','value':key}]}
    assert record['stages']['G_solver']['records'][0]['kwargs']['options']==literal
    assert record['terminal']['record']=={'literal_mapping_entries':literal,'semantic_reference':key}


def test_reserved_escape_key_nested_mapping_and_dataclass_field_remain_literal():
    @dataclass
    class LiteralFields:
        semantic_reference: str
    projection=f.Observer().projection
    native={'literal_mapping_entries':{'semantic_reference':'ordinary'},'nested':LiteralFields('ordinary')}
    projected=projection.take(native)
    projection.validate(projected)
    assert projected['native_mapping_entries'][0]['key']=='literal_mapping_entries'
    nested=projected['native_mapping_entries'][1]['value']
    assert nested['fields']=={'native_mapping_entries':[{'key_type':'str','key':'semantic_reference','value':'ordinary'}]}


@pytest.mark.parametrize('path',['projected_digest','references'],ids=['JSON-token-walk','root-reference-walk'])
def test_each_validation_traversal_refuses_before_iterating_at_shared_cap(monkeypatch,path):
    projection=f.Observer().projection
    projection.validation_visits=f.MAX_VALIDATION_WORK
    if path=='projected_digest':
        monkeypatch.setattr(f,'canonical',lambda value:pytest.fail('encoding after work cap'))
        with pytest.raises(f.IntegrityError,match='semantic validation work cap'):
            projection.projected_digest({'must_not_encode':'payload'})
    else:
        class NeverIterate(list):
            def __iter__(self): pytest.fail('root traversal after work cap')
        with pytest.raises(f.IntegrityError,match='semantic validation work cap'):
            projection.validate(NeverIterate(['payload']))
    assert projection.validation_visits==f.MAX_VALIDATION_WORK and not projection.validation_walk_counts


def test_projected_key_walk_and_reference_child_walk_consume_shared_bound():
    projection=f.Observer().projection
    projection.validation_visits=f.MAX_VALIDATION_WORK-1
    with pytest.raises(f.IntegrityError,match='semantic validation work cap'):
        projection.projected_digest({'must_not_encode':'payload'})
    assert projection.validation_walk_counts=={'projected_digest':1}
    projection=f.Observer().projection
    projection.validation_visits=f.MAX_VALIDATION_WORK-2
    with pytest.raises(f.IntegrityError,match='semantic validation work cap'):
        projection.validate(['first','second'])
    assert projection.validation_walk_counts=={'references':2} and projection.validation_visits==f.MAX_VALIDATION_WORK


def test_projected_digest_scratch_is_precharged_and_traversal_does_not_raise_global_caps():
    projection=f.StageProjection(f.Budget(reserved=f.MAX_OBSERVER))
    with pytest.raises(f.IntegrityError,match='evidence precharge'):
        projection.projected_digest('payload')
    assert projection.validation_visits==0
    observer=f.Observer(); projection=observer.projection
    body={'unicode':'\u2603','nested':[1,True,None]}
    assert projection.projected_digest(body)==f.digest(body)
    assert projection.validation_visits==sum(projection.validation_walk_counts.values())
    assert projection.validation_walk_counts['projected_digest']==6
    assert projection.validation_walk_counts['projected_keys']==2
    assert projection.validation_walk_counts['projected_key_tokens']==2
    assert f.MAX_ITEMS==100000 and f.MAX_OBSERVER==15*1024*1024


@pytest.mark.parametrize('cap',['items','bytes'])
def test_projected_dictionary_uses_precharged_scratch_without_key_list_allocations_even_at_retained_cap(monkeypatch,cap):
    observer=f.Observer(); projection=observer.projection; projection.ensure_scratch()
    with observer.budget.scope('G_solver'):
        observer.budget.charge(f.ALLOCATION_PLAN['G']['bytes'] if cap=='bytes' else 0,
            f.ALLOCATION_PLAN['G']['items'] if cap=='items' else 0)
        monkeypatch.setattr(f,'sorted',lambda value:pytest.fail('sorting after allocation cap'),raising=False)
        before=(observer.budget.items,observer.budget.reserved)
        assert projection.projected_digest({'key':'value'})==f.digest({'key':'value'})
        assert (observer.budget.items,observer.budget.reserved)==before


def test_projected_dictionary_nested_repeat_walk_has_no_retained_lists_and_charges_every_scan():
    observer=f.Observer(); projection=observer.projection; projection.ensure_scratch()
    before=observer.budget.items
    projection.projected_digest({'outer':{'inner':1}})
    assert observer.budget.items==before
    scans=projection.validation_walk_counts['projected_key_scan']; assert scans==2
    projection.projected_digest({'outer':{'inner':1}})
    assert observer.budget.items==before and projection.validation_walk_counts['projected_key_scan']==2*scans


def test_integer_mapping_and_literal_typed_marker_are_distinct_and_round_trip():
    projection=f.Observer().projection
    numeric={1:'x'}
    literal={'typed_mapping_entries':[{'key_type':'int','key':1,'value':'x'}]}
    left=projection.take(numeric); right=projection.take(literal)
    assert left!=right and f.canonical(left)!=f.canonical(right)
    assert resolve_projection(left)==numeric and resolve_projection(right)==literal
    projection.validate((left,right))


@pytest.mark.parametrize('marker',['semantic_reference','literal_mapping_entries','typed_mapping_entries','native_mapping_entries'])
def test_full_mapping_encoding_round_trips_mixed_reserved_string_integer_tuple_keys(marker):
    projection=f.Observer().projection
    original={marker:'ordinary',1:{'native_mapping_entries':'nested ordinary'},('frame',2):'tuple key'}
    projected=projection.take(original)
    assert [row['key_type'] for row in projected['native_mapping_entries']]==['str','int','tuple']
    reconstructed=resolve_projection(projected)
    assert list(reconstructed)==list(original) and reconstructed==original
    projection.validate(projected)


def test_dataclass_typed_mapping_marker_fields_and_mapping_lookalike_remain_distinct():
    @dataclass
    class LiteralFields:
        typed_mapping_entries: list
        native_mapping_entries: dict
    value=LiteralFields([{'key_type':'int','key':1,'value':'x'}],{1:'native field'})
    projection=f.Observer().projection
    projected=projection.take(value)
    assert set(projected['fields'])=={'native_mapping_entries'}
    assert resolve_projection(projected)['fields']=={
        'typed_mapping_entries':value.typed_mapping_entries,'native_mapping_entries':value.native_mapping_entries}
    # Even a native mapping that looks exactly like a dataclass projection is
    # encoded as a source map, not mistaken for a generated record envelope.
    lookalike={'type':projected['type'],'fields':resolve_projection(projected)['fields']}
    mapped=projection.take(lookalike)
    assert mapped!=projected and resolve_projection(mapped)==lookalike
    projection.validate((projected,mapped))


@pytest.mark.parametrize('phase',['projected_digest','projected_key_scan','references'])
def test_shared_meter_exhaustion_at_final_coordinator_validation_blocks_success_and_posthoc(monkeypatch,phase):
    calls=[]; public=[]; posthoc=[]; observed={}
    record=source_shaped_record('PlaneReading',frame='left',axis=0,side=1.)
    def solver(arg): calls.append(arg); return record
    module=SimpleNamespace(solver=solver)
    def prediction(paths,**kwargs): public.append(1); return module.solver(record)
    original_work=f.StageProjection.work; original_validate=f.StageProjection.validate
    final_validation=False
    def validate(self,roots=()):
        nonlocal final_validation
        if self.budget.stage=='terminal':
            final_validation=True
            observed['prior_visits']=self.validation_visits
            observed['prior_counts']=dict(self.validation_walk_counts)
        return original_validate(self,roots)
    def work(self,walk):
        # Lower only the test's work limit at the chosen actual final coverage
        # walk. Never alter the accumulated count: earlier production-delegate
        # projection and final coverage work share this exact same meter.
        if final_validation and self.budget.stage=='terminal' and walk==phase and 'cap' not in observed:
            observed['cap']=self.validation_visits+2
            monkeypatch.setattr(f,'MAX_VALIDATION_WORK',observed['cap'])
        try: return original_work(self,walk)
        finally:
            observed['final_visits']=self.validation_visits
            observed['final_counts']=dict(self.validation_walk_counts)
    monkeypatch.setattr(f.StageProjection,'validate',validate)
    monkeypatch.setattr(f.StageProjection,'work',work)
    result,storage=run(prediction,targets=((module,'solver','G_solver'),),loader=lambda:posthoc.append(1))
    raw=raw_native_record(storage)
    marker=f.strict_json(storage.files[Path('/memory/stub/COMPLETE.json')])
    assert public==[1] and calls==[record] and not posthoc and not result['metadata_access']
    assert result['state']=='SEALED_INSTRUMENTATION_FAILURE' and result['terminal']=='instrumentation_failure'
    assert marker['native_complete'] is False and raw['projection_coverage'] is None
    assert raw['instrumentation_failure']['message']=='semantic validation work cap'
    assert raw['stages']['G_solver']['calls']==1 and raw['stages']['G_solver']['state']=='returned'
    assert observed['prior_counts']['native_digest']>0 and observed['prior_counts']['projected_digest']>0
    assert observed['final_visits']==observed['cap']==sum(observed['final_counts'].values())
    assert observed['final_counts'][phase]>observed['prior_counts'].get(phase,0)
    assert observed['cap']>observed['prior_visits']
    assert f.MAX_ITEMS==100000 and f.MAX_OBSERVER==15*1024*1024


@pytest.mark.parametrize('body',[
    {},{'z':1,'a':2,'':3}, {'\u2603':1,'a"':2,'a\\':3,'\U0001f600':4},
    {'nested':{'z':[True,None,{'second':2,'first':1}],'a':()},'a':-0.},
    {str(index):index for index in range(19,-1,-1)},
])
def test_projected_key_selection_preserves_canonical_digest_without_sort_or_retained_growth(monkeypatch,body):
    projection=f.Observer().projection; projection.ensure_scratch()
    before=(projection.budget.items,projection.budget.reserved)
    monkeypatch.setattr(f,'sorted',lambda *args:pytest.fail('key list allocation'),raising=False)
    assert projection.projected_digest(body)==f.digest(body)
    assert projection.projected_digest(dict(reversed(tuple(body.items()))))==f.digest(body)
    assert (projection.budget.items,projection.budget.reserved)==before
    assert sum(projection.validation_walk_counts.values())==projection.validation_visits


def test_projected_key_selection_scan_cap_refuses_before_key_scalar_encoding(monkeypatch):
    projection=f.Observer().projection
    original_work=f.StageProjection.work; before={}
    def work(self,kind):
        if kind=='projected_key_scan' and not before:
            before.update(self.validation_walk_counts)
            monkeypatch.setattr(f,'MAX_VALIDATION_WORK',self.validation_visits)
        return original_work(self,kind)
    monkeypatch.setattr(f.StageProjection,'work',work)
    monkeypatch.setattr(f,'canonical',lambda *args:pytest.fail('scalar encoding before scan gate'))
    with pytest.raises(f.IntegrityError,match='semantic validation work cap'):
        projection.projected_digest({'b':2,'a':1})
    assert before=={'projected_digest':1,'projected_keys':2}
    assert projection.validation_visits==3


@pytest.mark.parametrize('tag',['=f8','=u1','<i1','>u1','<b1','>b1','|f8'])
def test_numeric_dtype_tags_refuse_host_native_or_noncanonical_one_byte_endian(monkeypatch,tag):
    width=int(tag[2:]); value=RawNumericBuffer(b'\0'*width,tag)
    monkeypatch.setattr(f.base64,'b64encode',lambda *args:pytest.fail('encode ambiguous dtype'))
    with pytest.raises(f.IntegrityError,match='numeric array endian unsupported'):
        f.Observer().projection.take(value,'points')


class RawNumericBuffer(bytearray):
    """Logical-byte stub; does not pretend to be a native ndarray constructor."""
    def __init__(self,raw,tag='|u1',shape=None,strides=None,writeable=False):
        super().__init__(raw)
        width=int(tag[2:]); self.size=len(self)//width
        self.dtype=SimpleNamespace(str=tag,hasobject=False,fields=None,subdtype=None)
        self.shape=(self.size,) if shape is None else shape
        self.strides=tuple(width*math.prod(self.shape[index+1:]) for index in range(len(self.shape))) if strides is None else strides
        self.nbytes=len(self); self.flags=SimpleNamespace(c_contiguous=True,writeable=writeable)


@pytest.mark.parametrize('tag,raw',[
    ('<f8',bytes.fromhex('010000000000f87f000000000000f07f000000000000f0ff00000000000000800000000000000000')),
    ('>f8',bytes.fromhex('7ff80000000000017ff0000000000000fff000000000000080000000000000000000000000000000')),
    ('<f4',bytes.fromhex('0100c07f0000807f000080ff0000008000000000')),
    ('>f4',bytes.fromhex('7fc000017f800000ff8000008000000000000000')),
    ('<f2',bytes.fromhex('017e007c00fc00800000')),
    ('>f2',bytes.fromhex('7e017c00fc0080000000')),
    ('<c16',struct.pack('<4d',-0.,float('inf'),float('nan'),2.)),
    ('>c8',struct.pack('>4f',-0.,float('-inf'),1.,2.)),
    ('<i8',struct.pack('<3q',-(2**63),0,2**63-1)),
    ('>i4',struct.pack('>3i',-(2**31),0,2**31-1)),
    ('<i2',struct.pack('<3h',-32768,0,32767)),
    ('|i1',bytes((128,0,127))),
    ('>u8',struct.pack('>3Q',0,1,2**64-1)),
    ('<u4',struct.pack('<3I',0,1,2**32-1)),
    ('>u2',struct.pack('>3H',0,1,65535)),
    ('|u1',bytes((0,1,255))),
    ('|b1',bytes((0,1,0))),
])
def test_numeric_codec_preserves_exact_primitive_endian_nan_payload_and_signed_zero_bits(tag,raw):
    value=RawNumericBuffer(raw,tag,writeable=True)
    projection=f.Observer().projection; root=projection.take(value,'points')
    definition=projection.definitions[root['semantic_reference']]
    receipt=definition['values_projection']
    assert numeric_bytes(receipt)==raw and receipt['sha256']==f.sha(raw)
    assert receipt['dtype']==tag and receipt['shape']==list(value.shape)
    assert receipt['strides']==list(value.strides) and receipt['writeable'] is True
    assert f.NUMERIC_VALUES_FIELD not in definition['projection']
    projection.validate(root)


@pytest.mark.parametrize('length',[0,1,2,3,3071,3072,3073,6144,6145])
def test_numeric_codec_canonical_chunk_boundaries_and_exact_order(length):
    raw=bytes(index%251 for index in range(length))
    receipt=f.Snapshot(f.Budget()).take(RawNumericBuffer(raw),'edges')
    chunks=receipt[f.NUMERIC_VALUES_FIELD]
    assert len(chunks)==(length+3071)//3072
    assert sum(map(len,chunks))==4*((length+2)//3)
    assert numeric_bytes(receipt)==raw
    f.Snapshot(f.Budget()).validate_numeric(receipt)


@pytest.mark.parametrize('shape,raw',[((),struct.pack('<d',-0.)),((0,4),b'')])
def test_numeric_codec_scalar_and_empty_shape(shape,raw):
    value=RawNumericBuffer(raw,'<f8',shape)
    receipt=f.Snapshot(f.Budget()).take(value,'matches')
    assert receipt['shape']==list(shape) and numeric_bytes(receipt)==raw


@pytest.mark.parametrize('step',[2,-1])
def test_numeric_codec_noncontiguous_logical_order_preserves_source_strides_and_charges_copies(monkeypatch,step):
    source=array('d',[1.,-0.,float('inf'),2.,3.,4.]); view=memoryview(source)[::step]
    class Strided:
        dtype=SimpleNamespace(str='<f8',hasobject=False,fields=None,subdtype=None)
        shape=(len(view),); strides=view.strides; size=len(view); nbytes=view.nbytes
        flags=SimpleNamespace(c_contiguous=False,writeable=False)
        def tobytes(self,order='C'): return view.tobytes(order=order)
    value=Strided()
    monkeypatch.setattr(f,'memoryview',lambda item:view if item is value else builtins.memoryview(item),raising=False)
    projection=f.Observer().projection; before=projection.budget.reserved
    root=projection.take(value,'points'); receipt=projection.definitions[root['semantic_reference']]['values_projection']
    assert numeric_bytes(receipt)==view.tobytes(order='C') and receipt['strides']==list(view.strides)
    # Two native rehash copies plus one payload logical copy, each precharged.
    assert projection.budget.reserved-before>=3*(view.nbytes+256)
    assert receipt['writeable'] is False
    projection.validate(root)


def test_numeric_codec_upgrade_once_equal_alias_stride_version_and_single_tiny_payload(monkeypatch):
    projection=f.Observer().projection
    value=RawNumericBuffer(bytes(range(100)))
    root=projection.take(value,'descriptors'); record=projection.definitions[root['semantic_reference']]
    assert f.NUMERIC_VALUES_FIELD not in record['values_projection']
    assert projection.take(value,'points')==root
    assert numeric_bytes(record['values_projection'])==bytes(range(100))
    original_encode=f.base64.b64encode
    def forbidden_encode(*args,**kwargs): raise AssertionError('repeat payload encoding')
    monkeypatch.setattr(f.base64,'b64encode',forbidden_encode)
    assert projection.take(value,'edges')==root
    equal=RawNumericBuffer(bytes(range(100)))
    assert projection.take(equal,'points')==root
    monkeypatch.setattr(f.base64,'b64encode',original_encode)
    different_stride=RawNumericBuffer(bytes(range(100)),strides=(2,))
    assert projection.take(different_stride,'points')!=root
    value[0]=255; changed=projection.take(value,'points')
    assert changed!=root and numeric_bytes(projection.definitions[changed['semantic_reference']]['values_projection'])[0]==255
    tiny=projection.take(RawNumericBuffer(b'xyz'),'points'); definition=projection.definitions[tiny['semantic_reference']]
    assert f.NUMERIC_VALUES_FIELD not in definition['projection']
    assert definition['values_projection'][f.NUMERIC_VALUES_FIELD]==['eHl6']
    projection.validate((root,changed,tiny))


def test_numeric_codec_mutation_between_chunk_reads_refuses_without_scalar_list(monkeypatch):
    value=RawNumericBuffer(bytes(index%251 for index in range(4000)))
    value.tolist=lambda:(_ for _ in ()).throw(AssertionError('scalar expansion'))
    original=f.base64.b64encode; calls=[]
    def mutate(piece):
        encoded=original(piece); calls.append(1)
        if len(calls)==1: value[3500]^=1
        return encoded
    monkeypatch.setattr(f.base64,'b64encode',mutate)
    with pytest.raises(f.IntegrityError,match='array changed during projection'):
        f.Observer().projection.take(value,'points')
    assert len(calls)==2


@pytest.mark.parametrize('kind',['count','order','short','alphabet','padding','pad_bits','hash'])
def test_numeric_codec_malformed_payload_is_refused_even_with_updated_projected_digest(kind):
    raw=bytes(index%251 for index in range(3073))
    projection=f.Observer().projection; root=projection.take(RawNumericBuffer(raw),'points')
    record=projection.definitions[root['semantic_reference']]; chunks=record['values_projection'][f.NUMERIC_VALUES_FIELD]
    if kind=='count': chunks.pop()
    elif kind=='order': chunks.reverse()
    elif kind=='short': chunks[0]=chunks[0][:-4]
    elif kind=='alphabet': chunks[0]='!'+chunks[0][1:]
    elif kind=='padding': chunks[-1]='===='
    elif kind=='pad_bits': chunks[-1]=chunks[-1][0]+'B=='  # valid decode, noncanonical low bits
    else: chunks[0]='AAAA'+chunks[0][4:]
    record['values_sha256']=projection.projected_digest(chunks)
    with pytest.raises(f.IntegrityError,match='numeric (chunk|payload)'):
        projection.validate(root)


@pytest.mark.parametrize('kind',['object','structured','subarray','unsupported','endian','byte_count','rank','stride_type','oversize'])
def test_numeric_codec_unsupported_metadata_refuses_before_encoder(monkeypatch,kind):
    value=RawNumericBuffer(b'abcdefgh','<f8')
    if kind=='object': value.dtype.hasobject=True
    elif kind=='structured': value.dtype.fields={'field':('f8',0)}
    elif kind=='subarray': value.dtype.subdtype=('f8',(1,))
    elif kind=='unsupported': value.dtype.str='<M8'
    elif kind=='endian': value.dtype.str='|f8'
    elif kind=='byte_count': value.size=2
    elif kind=='rank': value.shape=(1,1,1,1,1); value.strides=(8,8,8,8,8)
    elif kind=='stride_type': value.strides=('pointer-like',)
    else: value.nbytes=f.MAX_ARRAY+1
    monkeypatch.setattr(f.base64,'b64encode',lambda *args:pytest.fail('encoder before refusal'))
    with pytest.raises(f.IntegrityError): f.Snapshot(f.Budget()).take(value,'points')


@pytest.mark.parametrize('limit',['bytes','items','scratch','work'])
def test_numeric_codec_precharges_before_encoding_allocation(monkeypatch,limit):
    value=RawNumericBuffer(b'x'*6144)
    budget=f.Budget(); snapshot=f.Snapshot(budget)
    if limit=='bytes': budget.charge(f.MAX_OBSERVER-20000,0)
    elif limit=='items': budget.charge(0,f.MAX_ITEMS-20)
    elif limit=='scratch': budget.charge(f.MAX_OBSERVER-1100,0)
    else: monkeypatch.setattr(f,'MAX_VALIDATION_WORK',1)
    monkeypatch.setattr(f.base64,'b64encode',lambda *args:pytest.fail('encoder before precharge refusal'))
    with pytest.raises(f.IntegrityError): snapshot.take(value,'points')


@pytest.mark.parametrize('failure',['payload','mutation','bytes','items','work','validate_work','unsupported','oversize'])
def test_numeric_coordinator_failures_seal_false_delegate_once_zero_posthoc(monkeypatch,failure):
    calls=[]; public=[]; posthoc=[]; work_before={}
    value=RawNumericBuffer(b'x'*32768)
    if failure=='unsupported': value.dtype.hasobject=True
    if failure=='oversize': value.nbytes=f.MAX_ARRAY+1
    def features(): calls.append(1); return {'points':value}
    module=SimpleNamespace(features=features)
    def prediction(paths,**kwargs): public.append(1); return module.features()
    original_init=f.Observer.__init__
    def init(self):
        original_init(self)
        if failure in ('bytes','items'):
            with self.budget.scope('features'):
                used=self.budget.pool_counts.get('upstream',{'bytes':0,'items':0})
                if failure=='bytes': self.budget.charge(f.ALLOCATION_PLAN['upstream']['bytes']-used['bytes']-40000,0)
                else: self.budget.charge(0,f.ALLOCATION_PLAN['upstream']['items']-used['items']-50)
        if failure=='work': monkeypatch.setattr(f,'MAX_VALIDATION_WORK',1)
    monkeypatch.setattr(f.Observer,'__init__',init)
    if failure=='mutation':
        original_encode=f.base64.b64encode; mutated=[]
        def mutate(piece):
            encoded=original_encode(piece)
            if not mutated: value[0]^=1; mutated.append(1)
            return encoded
        monkeypatch.setattr(f.base64,'b64encode',mutate)
    if failure=='validate_work':
        original_work=f.Snapshot.numeric_work
        def exhaust(self,kind):
            if kind=='numeric_validate' and not work_before:
                work_before.update(self.validation_walk_counts)
                monkeypatch.setattr(f,'MAX_VALIDATION_WORK',self.validation_visits+1)
            return original_work(self,kind)
        monkeypatch.setattr(f.Snapshot,'numeric_work',exhaust)
    if failure=='payload':
        original=f.Snapshot.array_snapshot
        def corrupt(self,arg,key=''):
            receipt=original(self,arg,key)
            if f.NUMERIC_VALUES_FIELD in receipt: receipt[f.NUMERIC_VALUES_FIELD][0]='!'+receipt[f.NUMERIC_VALUES_FIELD][0][1:]
            return receipt
        monkeypatch.setattr(f.Snapshot,'array_snapshot',corrupt)
    result,storage=run(prediction,targets=((module,'features','features'),),loader=lambda:posthoc.append(1))
    raw=raw_native_record(storage); marker=f.strict_json(storage.files[Path('/memory/stub/COMPLETE.json')])
    assert result['state']=='SEALED_INSTRUMENTATION_FAILURE' and result['terminal']=='instrumentation_failure'
    assert public==[1] and calls==[1] and not posthoc and result['metadata_access'] is False
    assert marker['native_complete'] is False
    if failure in ('payload','work','validate_work'): assert raw['projection_coverage'] is None
    assert raw['instrumentation_failure'] and raw['stages']['features']['calls']==1
    if failure=='validate_work':
        assert work_before['native_digest']>0 and work_before['projected_digest']>0
        assert work_before['numeric_encode']>0 and work_before['numeric_hash']>0


def compact_match(**changes):
    values=dict(left='left',right='right',left_axis=0,right_axis=1,left_b=-0.,right_b=1.,
        quadrant_of_right=2,psr=2.,peak=3.,second_peak=1.,overlap_cells=64,integer_residual=None,
        score=4.,left_plane=0,right_plane=1,cell_shift_rc=(2,-3))
    values.update(changes); return source_shaped_record('PlaneMatch',**values)


def test_G_registry_and_vector_headers_are_physically_precharged_and_borrow_immutable_schema():
    observer=f.Observer(); projection=observer.projection
    with observer.budget.scope('G_prefilter'):
        projection.ensure_g_registry()
        visited=set()
        def immutable_size(value):
            if id(value) in visited: return 0,0
            visited.add(id(value)); size=sys.getsizeof(value); nodes=1
            if type(value) is dict:
                for key,item in value.items():
                    extra,count=immutable_size(key); size+=extra; nodes+=count
                    extra,count=immutable_size(item); size+=extra; nodes+=count
            elif type(value) is tuple:
                for item in value:
                    extra,count=immutable_size(item); size+=extra; nodes+=count
            else: assert type(value) is str
            return size,nodes
        size,nodes=immutable_size(projection.g_registry)
        assert size<=8192 and nodes<=64
        assert observer.budget.pool_counts['G']=={'items':64,'bytes':8192}
        for length in (1,4096):
            row=compact_match(left='\u2603'*length)
            ref=projection.take(row); definition=projection.definitions[ref['semantic_reference']]; body=definition['projection']
            registry_key=next(key for key in projection.g_registry if key.endswith('.PlaneMatch'))
            assert body['schema'] is registry_key
            if 'values_json' in body:
                assert len(body['values_json'])<=4096
                actual=sys.getsizeof(body)+sys.getsizeof(body['values_json'])
                upper=512+len(body['values_json'])+128
            else:
                chunks=body['values_json_chunks']; assert len(chunks)>1
                actual=sys.getsizeof(body)+sys.getsizeof(chunks)+sum(sys.getsizeof(piece) for piece in chunks)
                upper=512+sum(map(len,chunks))+128*len(chunks)
            assert actual<=upper
            # Both record debits precede retention. Combined body/definition
            # metadata allowance also covers table storage:256B per definition
            # conservatively includes dictionary growth, even the first entry.
            metadata=sum(sys.getsizeof(value) for value in (definition,ref,ref['semantic_reference'],
                definition['native_sha256'],definition['content_sha256'],definition['wire_sha256']))+256
            assert actual+metadata<=upper+1024
            assert sys.getsizeof(ref)+sys.getsizeof(ref['semantic_reference'])<=384
    projection.validate()


@pytest.mark.parametrize('failure',['unknown-schema','length','type','noncanonical','long-string','chunks','unknown-encoding','kind','order','wire','undefined','cycle'])
def test_G_compact_definition_schema_vector_digest_and_reference_controls(failure):
    observer=f.Observer(); projection=observer.projection
    with observer.budget.scope('G_prefilter'):
        row=compact_match(); ref=projection.take(row); container=projection.take([row])
    definition=projection.definitions[ref['semantic_reference']]; body=definition['projection']
    if failure=='unknown-schema': body['schema']='foreign.PlaneMatch'
    elif failure=='length': body['values_json']='[]'
    elif failure=='type':
        values=json.loads(body['values_json']); values[2]='0'; body['values_json']=json.dumps(values,separators=(',',':'))
    elif failure=='noncanonical': body['values_json']=' '+body['values_json']
    elif failure=='long-string':
        values=json.loads(body['values_json']); values[0]='x'*4097
        text=json.dumps(values,separators=(',',':')); del body['values_json']
        body['values_json_chunks']=[text[offset:offset+4096] for offset in range(0,len(text),4096)]
    elif failure=='chunks':
        del body['values_json']; body['values_json_chunks']=['[]','[]']
    elif failure=='unknown-encoding': definition['encoding']='G-unknown'
    elif failure=='kind': definition['kind']='g_inline_record'
    elif failure=='order':
        values=json.loads(body['values_json']); values[0],values[1]=values[1],values[0]
        body['values_json']=json.dumps(values,separators=(',',':'))
    elif failure=='wire': definition['wire_sha256']='0'*64
    else:
        sequence=projection.definitions[container['semantic_reference']]
        sequence['projection']['items']=[{'semantic_reference':container['semantic_reference'] if failure=='cycle' else 'record:'+'0'*64}]
    with pytest.raises(f.IntegrityError): projection.validate((ref,container))


@pytest.mark.parametrize('failure',['source','annotations','field-order','shift-kind','field-value'])
def test_G_pinned_schema_namespace_and_native_field_contract_refuses(failure,monkeypatch):
    row=compact_match()
    if failure=='source': monkeypatch.setattr(f,'G_SOURCE_SHA256','0'*64)
    elif failure=='annotations': type(row).__dataclass_fields__['left'].type='int'
    elif failure=='field-order': type(row).__dataclass_fields__=dict(reversed(tuple(type(row).__dataclass_fields__.items())))
    elif failure=='shift-kind': row.cell_shift_rc=[2,-3]
    else: row.score={'nonfinite':'nan'}
    observer=f.Observer()
    with observer.budget.scope('G_prefilter'),pytest.raises(f.IntegrityError): observer.projection.take(row)


def test_G_foreign_namespace_does_not_claim_source_schema_and_mutable_versions_are_distinct():
    observer=f.Observer(); projection=observer.projection; row=compact_match(); equal=compact_match()
    foreign=compact_match(); type(foreign).__module__='foreign.plane_graph'
    with observer.budget.scope('G_prefilter'):
        before=projection.take([row,equal]); assert projection.take(row)==projection.take(equal)
        assert observer.identity(row)!=observer.identity(equal)
        fake=projection.take(foreign); assert 'semantic_reference' not in fake
        row.score=9.; after=projection.take([row,equal]); assert after!=before
        assert projection.take((row,equal))!=after
        literal={'semantic_reference':before['semantic_reference'],'schema':'native ordinary',1:'int',('k',2):'tuple'}
        source=projection.take(literal)
    assert resolve_projection(source,projection.definitions)==literal
    projection.validate((before,after,source))


@pytest.mark.parametrize('count',[0,1,7,8,9,760])
def test_G_disposition_bitmap_roundtrip_and_legacy_digest(count):
    observer=f.Observer(); matches=[object() for _ in range(count)]; kept={id(value) for value in matches[::3]}
    with observer.budget.scope('G_matcher'):
        ref=observer.projection.g_dispositions(matches,kept)
    expected=['kept' if index%3==0 else 'dropped' for index in range(count)]
    assert resolve_projection(ref,observer.projection.definitions)==expected
    assert observer.projection.projected_digest(ref)==frozen_canonical_digest(expected)
    assert ref['semantic_reference']=='g_sequence:'+frozen_native_fingerprint(expected)
    observer.projection.validate(ref)


@pytest.mark.parametrize('failure',['length','alphabet','padding','tail','count','kind'])
def test_G_disposition_bitmap_malformed_controls(failure):
    observer=f.Observer(); matches=[object() for _ in range(9)]
    with observer.budget.scope('G_matcher'):
        ref=observer.projection.g_dispositions(matches,{id(matches[0])})
    definition=observer.projection.definitions[ref['semantic_reference']]; body=definition['projection']
    if failure=='length': body['bits_base64']=''
    elif failure=='alphabet': body['bits_base64']='!!!!'
    elif failure=='padding': body['bits_base64']='AQAA'
    elif failure=='tail': body['bits_base64']=base64.b64encode(bytes((1,128))).decode()
    elif failure=='count': body['count']=True
    else: body['container_type']='builtins.tuple'
    with pytest.raises(f.IntegrityError): observer.projection.validate(ref)


@pytest.mark.parametrize('failure',['duplicate-universe','duplicate-kept','foreign-kept','items','bytes','work'])
def test_G_coordinator_integrity_failure_is_sealed_false_once_without_downstream_or_posthoc(failure,monkeypatch):
    rows=[compact_match(),compact_match(score=2.)]; calls=[]; public=[]; posthoc=[]; retained=[]
    if failure=='duplicate-universe': rows[1]=rows[0]
    original_init=f.Observer.__init__; captured={}
    def observer_init(self,*args,**kwargs): original_init(self,*args,**kwargs); captured['observer']=self
    monkeypatch.setattr(f.Observer,'__init__',observer_init)
    def prefilter(matches,index,refusals):
        calls.append('prefilter')
        if failure=='duplicate-kept': return [matches[0],matches[0]]
        if failure=='foreign-kept': return [compact_match()]
        return matches[:1]
    def matcher():
        calls.append('matcher'); matches=rows; refusals=[]
        if failure in ('items','bytes'):
            budget=captured['observer'].budget; used=budget.pool_counts['G']; cap=f.ALLOCATION_PLAN['G']
            if failure=='items':
                count=cap['items']-used['items']; amount=8*count+512
                budget.charge(amount,count); retained.append([None]*(count-8))
            else:
                amount=cap['bytes']-used['bytes']; budget.charge(amount,2); retained.append(bytearray(amount-256))
        filtered=module.prefilter(matches,{('left',0):'reading'},refusals)
        return filtered
    def solver(matches): calls.append('solver'); return None
    module=SimpleNamespace(matcher=matcher,prefilter=prefilter,solver=solver)
    if failure=='work':
        original_work=f.StageProjection.work
        def work(self,phase):
            if self.budget.stage=='G_prefilter' and phase=='g_schema':
                monkeypatch.setattr(f,'MAX_VALIDATION_WORK',self.validation_visits)
            return original_work(self,phase)
        monkeypatch.setattr(f.StageProjection,'work',work)
    def prediction(paths,**kwargs): public.append(1); module.solver(module.matcher()); return Outcome()
    result,storage=run(prediction,targets=((module,'matcher','G_matcher'),(module,'prefilter','G_prefilter'),
        (module,'solver','G_solver')),loader=lambda:posthoc.append(1))
    raw=raw_native_record(storage); marker=f.strict_json(storage.files[Path('/memory/stub/COMPLETE.json')])
    assert result['state']=='SEALED_INSTRUMENTATION_FAILURE' and public==[1] and calls.count('matcher')==1
    assert 'solver' not in calls and not posthoc and not result['metadata_access'] and marker['native_complete'] is False
    assert raw['stages']['G_solver']['calls']==0
    assert calls.count('prefilter')==(0 if failure in ('items','bytes','work') else 1)


@pytest.mark.parametrize('limit',['item','byte'])
def test_G_vector_precharge_precedes_canonical_value_encoding(limit,monkeypatch):
    observer=f.Observer(); projection=observer.projection
    with observer.budget.scope('G_prefilter'):
        projection.ensure_g_registry(); projection.ensure_g_json_scratch(); projection.ensure_scratch()
        used=observer.budget.pool_counts['G']; cap=f.ALLOCATION_PLAN['G']
        if limit=='item': observer.budget.charge(128*(cap['items']-used['items']),cap['items']-used['items'])
        else: observer.budget.charge(cap['bytes']-used['bytes'],1)
        original=f.canonical
        def canonical(value):
            if type(value) is list: pytest.fail('G vector encoding before reservation')
            return original(value)
        monkeypatch.setattr(f,'canonical',canonical)
        with pytest.raises(f.IntegrityError): projection.take(compact_match())


@pytest.mark.parametrize('shape',['list','tuple','nested'])
@pytest.mark.parametrize('key',['matches','points'])
def test_G_cached_container_coordinate_context_upgrades_one_lossless_payload(shape,key):
    values=RawNumericBuffer(struct.pack('<100d',*range(100)),'<f8')
    container=[values] if shape=='list' else (values,) if shape=='tuple' else [[(values,values)],(values,)]
    observer=f.Observer(); projection=observer.projection
    with observer.budget.scope('G_prefilter'):
        first=projection.take(container,'descriptors')
        array_key='array:'+frozen_native_fingerprint(values)
        definition=projection.definitions[array_key]
        assert f.NUMERIC_VALUES_FIELD not in definition['values_projection']
        before_hashes={name:(entry['content_sha256'],entry.get('wire_sha256')) for name,entry in projection.definitions.items()}
        before_encodes=projection.validation_walk_counts.get('numeric_encode',0)
        current=projection.take(container,key)
        assert current==first and numeric_bytes(definition['values_projection'])==bytes(values)
        assert projection.validation_walk_counts['numeric_encode']-before_encodes==1
        payload=definition['values_projection'][f.NUMERIC_VALUES_FIELD]
        assert len(payload)==1
        full=definition['values_projection']
        assert projection.take(container,key)==first and projection.take(container,'points')==first
        assert projection.take(container,'matches')==first
        assert definition['values_projection'] is full and definition['values_projection'][f.NUMERIC_VALUES_FIELD] is payload
        assert projection.validation_walk_counts['numeric_encode']-before_encodes==1
        assert before_hashes=={name:(entry['content_sha256'],entry.get('wire_sha256')) for name,entry in projection.definitions.items()}
        assert projection.native_digest(container)==frozen_native_fingerprint(container)
        assert projection.projected_digest(first)==frozen_canonical_digest(frozen_prechange_projection(container))
    projection.validate((first,current))


def test_G_nested_mapping_field_context_upgrades_prior_cached_sequence_without_outer_key_leak():
    values=RawNumericBuffer(struct.pack('<100d',-0.,float('inf'),*range(98)),'<f8'); sequence=[(values,values)]
    observer=f.Observer(); projection=observer.projection
    with observer.budget.scope('G_prefilter'):
        first=projection.take(sequence,'descriptors')
        definition=projection.definitions['array:'+frozen_native_fingerprint(values)]
        assert f.NUMERIC_VALUES_FIELD not in definition['values_projection']
        mapped=projection.take({'matches':sequence},'descriptors')
        assert numeric_bytes(definition['values_projection'])==bytes(values)
        assert projection.validation_walk_counts['numeric_encode']==1
        assert projection.take({'matches':sequence},'points')==mapped
        assert projection.projected_digest(first)==frozen_canonical_digest(frozen_prechange_projection(sequence))
        # A non-coordinate mapping field overrides an outer coordinate key.
        other=RawNumericBuffer(struct.pack('<100d',*range(100)),'<f8')
        ordinary=projection.take({'descriptors':[other]},'matches')
        assert f.NUMERIC_VALUES_FIELD not in projection.definitions['array:'+frozen_native_fingerprint(other)]['values_projection']
    projection.validate((mapped,ordinary))


@pytest.mark.parametrize('shape',['list','tuple','nested'])
@pytest.mark.parametrize('failure',[None,'work','mutation'])
def test_G_container_context_upgrade_through_coordinator_preserves_once_seal_barriers(shape,failure,monkeypatch):
    values=RawNumericBuffer(struct.pack('<100d',*range(100)),'<f8')
    container=[values] if shape=='list' else (values,) if shape=='tuple' else [[(values,values)],(values,)]
    native=[]; public=[]; posthoc=[]; observed={}
    original_init=f.Observer.__init__
    def observer_init(self,*args,**kwargs): original_init(self,*args,**kwargs); observed['observer']=self
    monkeypatch.setattr(f.Observer,'__init__',observer_init)
    if failure=='work':
        original_work=f.StageProjection.work
        def work(self,phase):
            if phase=='numeric_context_upgrade': monkeypatch.setattr(f,'MAX_VALIDATION_WORK',self.validation_visits)
            return original_work(self,phase)
        monkeypatch.setattr(f.StageProjection,'work',work)
    elif failure=='mutation':
        original_upgrade=f.StageProjection.upgrade_numeric_descendants
        mutated=False
        def upgrade(self,value,key,depth):
            nonlocal mutated
            original_upgrade(self,value,key,depth)
            if value is container and not mutated: values[0]^=1; mutated=True
        monkeypatch.setattr(f.StageProjection,'upgrade_numeric_descendants',upgrade)
    def prefilter(*,descriptors):
        assert descriptors is container; native.append('prefilter')
        definition=observed['observer'].projection.definitions['array:'+frozen_native_fingerprint(values)]
        assert f.NUMERIC_VALUES_FIELD not in definition['values_projection']
    def solver(*,matches):
        assert matches is container; native.append('solver')
        definition=observed['observer'].projection.definitions['array:'+frozen_native_fingerprint(values)]
        assert numeric_bytes(definition['values_projection'])==bytes(values)
        assert observed['observer'].projection.validation_walk_counts['numeric_encode']==1
        return Outcome()
    module=SimpleNamespace(prefilter=prefilter,solver=solver)
    def prediction(paths,**kwargs):
        public.append(1); module.prefilter(descriptors=container); return module.solver(matches=container)
    result,storage=run(prediction,targets=((module,'prefilter','G_prefilter'),(module,'solver','G_solver')),
                       loader=None if failure is None else lambda:posthoc.append(1))
    raw=raw_native_record(storage); marker=f.strict_json(storage.files[Path('/memory/stub/COMPLETE.json')])
    assert public==[1] and raw['stages']['G_prefilter']['calls']==1 and raw['stages']['G_solver']['calls']==1
    assert not posthoc and not result['metadata_access']
    if failure is None:
        assert result['state']=='SEALED' and native==['prefilter','solver'] and marker['native_complete'] is True
        assert raw['projection_coverage']['validation_walk_counts']['numeric_encode']==1
        record=native_record(storage)
        assert record['stages']['G_prefilter']['records'][0]['kwargs']['descriptors']==record['stages']['G_solver']['records'][0]['kwargs']['matches']
    else:
        assert result['state']=='SEALED_INSTRUMENTATION_FAILURE' and native==['prefilter'] and marker['native_complete'] is False
        assert raw['instrumentation_failure']['message']==('semantic validation work cap' if failure=='work' else 'container changed during numeric upgrade')


def geometry_wrapper_packet():
    reading=source_shaped_record('PlaneReading',frame='left',axis=0,alternatives=())
    sig=source_shaped_record('PlaneSignature',reading=reading,blocks_per_period=1.,
        cells=RawNumericBuffer(struct.pack('<2d',-0.,float('inf')),'<f8'),origin_uv=(-0.,1.))
    evidence=source_shaped_record('SignatureEvidence',frame_id='left',canonical_index=0,reading_axis=0,
        alternative_index=1,blocks_per_period=-0.,signature=sig,signature_sha256='1'*64)
    equal=source_shaped_record('SignatureEvidence',**{field.name:getattr(evidence,field.name) for field in fields(evidence)})
    mask=source_shaped_record('UsableMask',frame_id='left',height=1,width=2,usable_bool_bytes=b'\x00\x01',mask_sha256='2'*64)
    abstention=source_shaped_record('SignatureAbstention',frame_id='right',reason='toy',detail='quote"\\\u2603',canonical_index=None,alternative_index=None)
    f1=source_shaped_record('ImageDerivedSignaturesResult',schema='toy',producer='toy',terminal='geometry_unavailable',
        status='assembled',reason='',detail='',session_sha256='3'*64,lattice_sha256='4'*64,frame_ids=('left','right'),
        usable_masks=(mask,),signatures=(evidence,equal,evidence),abstentions=(abstention,),masks_sha256='5'*64,
        signatures_sha256='6'*64,result_sha256='7'*64)
    selection=source_shaped_record('EndpointSelectionCertificate',schema='toy',producer='toy',status='none',reason='toy',
        session_sha256='3'*64,lattice_sha256='4'*64,signatures_sha256='6'*64,connection_sha256='',
        frame_ids=('left','right'),quadrant_constraints=(),budget=None,reading_bindings=(),candidate_count=0,
        candidate_matches_sha256='8'*64,proof_json='{}',representatives=(),result_sha256='9'*64)
    graph=source_shaped_record('ImageDerivedGraphResult',schema='toy',producer='toy',terminal='geometry_unavailable',status='unavailable',
        reason='ambiguous_quadrant',detail='component solver abstained',session_sha256='3'*64,lattice_sha256='4'*64,
        signatures_sha256='7'*64,frame_ids=('left','right'),readings=(reading,),signatures=f1.signatures,selection=selection,matches=(),
        component_solution=None,correlation_algorithm='toy',max_shift_cells=32,result_sha256='8'*64)
    geometry=source_shaped_record('ImageDerivedGeometryResult',schema='toy',producer='toy',terminal='geometry_unavailable',
        status='unavailable',reason='g_unavailable',detail=graph.reason,session_sha256='3'*64,lattice_sha256='4'*64,
        stage_digests=(('f1','7'*64),('g','8'*64)),records=(('f1',f1),('g',graph)),placement_records=(),
        universe=None,supports=(),result_sha256='9'*64)
    return geometry,f1,graph,evidence,equal


@pytest.mark.parametrize('change',['none','field','reorder','kind','nonfinite','typed-markers','stale-digest'])
def test_geometry_vectors_complete_frozen_legacy_projection(change):
    geometry,f1,graph,evidence,equal=geometry_wrapper_packet()
    observer=f.Observer(); p=observer.projection
    with observer.budget.scope('geometry'):
        first=p.take(geometry); expected=frozen_record_body(geometry)
        assert evidence is not equal and first['semantic_reference'].startswith('record:')
        if change=='field': evidence.alternative_index=2
        elif change=='reorder': f1.signatures=tuple(reversed(f1.signatures)); f1.signatures=(equal,evidence,equal)
        elif change=='kind': graph.signatures=list(graph.signatures)
        elif change=='nonfinite': evidence.blocks_per_period=float('nan')
        elif change=='typed-markers': graph.detail={'semantic_reference':'literal',1:'integer',('k',2):'tuple','typed_mapping_entries':[]}
        elif change=='stale-digest': evidence.signature.origin_uv=(2.,3.)
        current=p.take(geometry)
        if change=='kind': assert first!=current
        if change in ('field','nonfinite','typed-markers','stale-digest'): assert first!=current
    with observer.budget.scope('terminal'): p.validate((first,current))
    initial=p.definitions[first['semantic_reference']]
    final=p.definitions[current['semantic_reference']]
    assert legacy_logical_wire(initial['projection'],p.definitions,initial['encoding'])==expected
    assert initial['content_sha256']==frozen_canonical_digest(expected)
    assert legacy_logical_wire(final['projection'],p.definitions,final['encoding'])==frozen_record_body(geometry)
    assert final['content_sha256']==frozen_canonical_digest(frozen_record_body(geometry))
    assert p.native_digest(geometry)==frozen_native_fingerprint(geometry)
    sig_key='record:'+frozen_native_fingerprint(evidence.signature)
    assert sig_key in p.definitions  # pre-existing PlaneSignature reference stays opaque
    assert p.geometry_live_nodes==0


@pytest.mark.parametrize('which',['namespace','annotation','order','source','registry','cycle'])
def test_geometry_schema_and_cycle_controls(which,monkeypatch):
    geometry,f1,graph,evidence,equal=geometry_wrapper_packet(); observer=f.Observer(); p=observer.projection
    if which=='namespace': type(evidence).__module__='foreign'
    elif which=='annotation': type(evidence).__dataclass_fields__['alternative_index'].type='float'
    elif which=='order':
        declared=type(evidence).__dataclass_fields__; type(evidence).__dataclass_fields__=dict(reversed(tuple(declared.items())))
    elif which=='source': monkeypatch.setattr(f,'GEOMETRY_SOURCE_PINS',dict(f.GEOMETRY_SOURCE_PINS,**{'pipeline/image_derived_signatures.py':'0'*64}))
    elif which=='cycle': geometry.records=(('cycle',geometry),)
    with observer.budget.scope('geometry'):
        if which=='registry':
            p.take(geometry); p.geometry_registry.pop(next(iter(p.geometry_registry)))
        with pytest.raises(f.IntegrityError): p.take(geometry)
    assert p.geometry_live_nodes==0


@pytest.mark.parametrize('which',['length','canonical','schema','chunk','undefined','wire','content','kind','parser-cap'])
def test_geometry_malformed_vectors_refuse(which,monkeypatch):
    geometry,*_=geometry_wrapper_packet(); observer=f.Observer(); p=observer.projection
    with observer.budget.scope('geometry'): reference=p.take(geometry)
    definition=p.definitions[reference['semantic_reference']]; body=definition['projection']
    if which=='length': body['values_json']='[]'
    elif which=='canonical': body['values_json']=' '+body['values_json']
    elif which=='schema': body['schema']='foreign.Geometry'
    elif which=='chunk': body.pop('values_json'); body['values_json_chunks']=['[]','[]']
    elif which=='undefined':
        values=json.loads(body['values_json']); values[12]={'semantic_reference':'record:'+'0'*64}
        body['values_json']=f.canonical(values).decode()
        definition['content_sha256']=p.projected_digest(body,encoding=definition['encoding'])
        definition['wire_sha256']=p.projected_digest(body,legacy=False)
    elif which=='wire': definition['wire_sha256']='0'*64
    elif which=='content': definition['content_sha256']='0'*64
    elif which=='kind': definition['kind']='geometry_inline_record'
    elif which=='parser-cap': monkeypatch.setattr(f,'GEOMETRY_SCRATCH_NODES',1)
    with observer.budget.scope('terminal'):
        with pytest.raises(f.IntegrityError): p.validate((reference,))
    assert p.geometry_live_nodes==0


@pytest.mark.parametrize('first',['descriptor','match','geometry'])
def test_geometry_retention_first_use_order_and_single_array_payload(first):
    geometry,_,graph,_,_=geometry_wrapper_packet()
    values=RawNumericBuffer(bytes(65*8),'<f8'); container=[values]; graph.matches=container
    observer=f.Observer(); p=observer.projection
    with observer.budget.scope('G_prefilter'):
        if first!='geometry': p.take(container,'descriptors' if first=='descriptor' else 'matches')
    with observer.budget.scope('geometry'):
        before=p.take(geometry); after=p.take(geometry)
    with observer.budget.scope('terminal'): p.validate((before,after))
    full=p.definitions['array:'+frozen_native_fingerprint(values)]['values_projection']
    assert numeric_bytes(full)==bytes(values)
    assert before==after and p.validation_walk_counts['numeric_encode']==2  # 2 chunks: 520B array plus tiny signature
    decoded=legacy_logical_wire(p.definitions[before['semantic_reference']]['projection'],p.definitions,'geometry-field-vector-v1')
    assert decoded==frozen_record_body(geometry)


@pytest.mark.parametrize('failure',['encode-item','encode-byte','encode-work','mutation','schema-mutation','output-item','final-work','final-parser'])
def test_geometry_coordinator_resource_and_mutation_barriers(failure,monkeypatch):
    geometry,*_=geometry_wrapper_packet(); captured={}; delegates=[]; public=[]; posthoc=[]
    original_init=f.Observer.__init__
    def init(self,*args,**kwargs): original_init(self,*args,**kwargs); captured['observer']=self
    monkeypatch.setattr(f.Observer,'__init__',init)
    original_record=f.StageProjection.geometry_record
    original_validate=f.StageProjection.validate
    armed=False
    def geometry_record(self,value,*args,**kwargs):
        nonlocal armed
        if value is geometry and not armed:
            armed=True
            if failure in ('encode-item','encode-byte'):
                self.ensure_geometry_registry(); used=self.budget.pool_counts['G']; cap=f.ALLOCATION_PLAN['G']
                self.budget.charge(cap['bytes']-used['bytes'] if failure=='encode-byte' else 0,
                    cap['items']-used['items'] if failure=='encode-item' else 0)
            elif failure=='encode-work': monkeypatch.setattr(f,'MAX_VALIDATION_WORK',self.validation_visits)
        return original_record(self,value,*args,**kwargs)
    monkeypatch.setattr(f.StageProjection,'geometry_record',geometry_record)
    if failure in ('mutation','schema-mutation'):
        original_tokens=f.StageProjection.geometry_tokens; mutated=False
        def tokens(self,value,key='',depth=0):
            nonlocal mutated
            yield from original_tokens(self,value,key,depth)
            if key=='result_sha256' and value==geometry.result_sha256 and not mutated:
                if failure=='mutation': geometry.detail='changed after encode'
                else: type(geometry).__dataclass_fields__['status'].type='int'
                mutated=True
        monkeypatch.setattr(f.StageProjection,'geometry_tokens',tokens)
    def validate(self,roots=()):
        if failure=='final-work': monkeypatch.setattr(f,'MAX_VALIDATION_WORK',self.validation_visits)
        elif failure=='final-parser': monkeypatch.setattr(f,'GEOMETRY_SCRATCH_NODES',1)
        return original_validate(self,roots)
    monkeypatch.setattr(f.StageProjection,'validate',validate)
    def geometry_stage(): delegates.append('geometry'); return geometry
    def output_stage(value): delegates.append('output'); return Outcome()
    module=SimpleNamespace(geometry=geometry_stage,output=output_stage)
    def predict(paths,**kwargs):
        public.append(1); value=module.geometry()
        if failure=='output-item':
            observer=captured['observer']; used=observer.budget.pool_counts['G']['items']
            with observer.budget.scope('geometry'):
                observer.budget.charge(0,f.ALLOCATION_PLAN['G']['items']-used)
            module.output(value)
        return source_shaped_record('ImageDerivedProductionResult',geometry=value)
    result,storage=run(predict,targets=((module,'geometry','geometry'),(module,'output','output')),loader=lambda:posthoc.append(1))
    raw=raw_native_record(storage); marker=f.strict_json(storage.files[Path('/memory/stub/COMPLETE.json')])
    assert result['state']=='SEALED_INSTRUMENTATION_FAILURE' and marker['native_complete'] is False
    assert public==[1] and delegates==['geometry'] and not posthoc and not result['metadata_access']
    assert raw['stages']['geometry']['calls']==1
    assert raw['stages']['output']['calls']==(0 if failure!='output-item' else 0)
    assert captured['observer'].projection.geometry_live_nodes==0


def test_geometry_schema_registry_and_fixed_workspace_allocation_bounds():
    seen=set()
    def size(value):
        if id(value) in seen: return (0,0)
        seen.add(id(value)); byte_count=sys.getsizeof(value); items=1
        children=tuple(value.keys())+tuple(value.values()) if isinstance(value,dict) else value if type(value) in (tuple,list,frozenset) else ()
        for child in children:
            amount,nodes=size(child); byte_count+=amount; items+=nodes
        return byte_count,items
    geometry,*_=geometry_wrapper_packet(); observer=f.Observer(); p=observer.projection
    with observer.budget.scope('geometry'): ref=p.take(geometry)
    byte_count,items=size((p.geometry_registry,f.GEOMETRY_SOURCE_PINS,f.GEOMETRY_RECORD_NAMES,p.geometry_registry_sha256))
    print('GEOMETRY_REGISTRY_BASE='+json.dumps({'bytes':byte_count,'items':items}))
    assert byte_count<=16384 and items<=256,(byte_count,items)
    # Classes are borrowed source/runtime objects, not newly allocated types.
    # Count fresh dictionary header/slots; the fixed precharge covers all64.
    assert items+1+2*f.GEOMETRY_CLASS_LIMIT<=256
    assert byte_count+sys.getsizeof(p.geometry_classes)<=16384
    assert all(kind is type(value) for kind in p.geometry_classes for value in
               [next(item for item in (geometry,geometry.records[0][1],geometry.records[1][1],geometry.records[1][1].selection,
                *geometry.records[0][1].signatures,*geometry.records[0][1].usable_masks,*geometry.records[0][1].abstentions)
                if type(item) is kind)])
    assert all(p.geometry_registry[key] is f.GEOMETRY_RECORD_SCHEMAS[key] for key in p.geometry_registry)
    for key,definition in p.definitions.items():
        if definition.get('encoding')!='geometry-field-vector-v1': continue
        body=definition['projection']; text=body.get('values_json'); chunks=body.get('values_json_chunks',())
        count=1 if text is not None else len(chunks); length=len(text) if text is not None else sum(map(len,chunks))
        # Actual combined body/chunks/dict/strings/reference/table allowance.
        retained=sys.getsizeof(body)+sys.getsizeof(definition)+sys.getsizeof(key)+sys.getsizeof({'semantic_reference':key})+256
        retained+=sum(sys.getsizeof(definition[name]) for name in ('native_sha256','content_sha256','wire_sha256'))
        retained+=sys.getsizeof(text) if text is not None else sys.getsizeof(chunks)+sum(map(sys.getsizeof,chunks))
        assert retained<=512+length+128*count+1024,(retained,length)
    # 24 fixed encoder buffers, 24 decoder raw fields, conservative live-node
    # object/slot allowance plus four scalar/canonical conversion buffers.
    assert 2*f.MAX_DEPTH*f.GEOMETRY_JSON_BYTES+512*f.GEOMETRY_SCRATCH_NODES+4*f.GEOMETRY_JSON_BYTES<=1024*1024
    assert f.MAX_ITEMS==100000 and f.MAX_VALIDATION_WORK==2000000
    assert f.ALLOCATION_PLAN=={'upstream':{'items':60000,'bytes':8*1024*1024},'G':{'items':35000,'bytes':6*1024*1024},
                              'terminal_failure':{'items':5000,'bytes':1024*1024}}


@pytest.mark.parametrize('length',[4096,4097,16384])
def test_geometry_chunk_roundtrip_and_preallocation_wire_gate(length):
    geometry,*_=geometry_wrapper_packet(); geometry.detail='x'*min(length,f.MAX_STRING)
    # Fill several distinct legal ordinary fields, rather than bypass MAX_STRING.
    geometry.reason='y'*min(length,f.MAX_STRING)
    geometry.producer='z'*min(length,f.MAX_STRING)
    observer=f.Observer(); p=observer.projection
    with observer.budget.scope('geometry'): ref=p.take(geometry)
    body=p.definitions[ref['semantic_reference']]['projection']
    assert len(body['values_json_chunks']) in (3,4)
    assert all(len(chunk)==4096 for chunk in body['values_json_chunks'][:-1])
    with observer.budget.scope('terminal'): p.validate((ref,))
    assert legacy_logical_wire(body,p.definitions,'geometry-field-vector-v1')==frozen_record_body(geometry)
    geometry.status='a'*4096
    with observer.budget.scope('geometry'):
        with pytest.raises(f.IntegrityError,match='geometry value bytes'): p.take(geometry)


def test_geometry_mutation_after_return_has_fresh_compact_terminal_version():
    geometry,_,graph,evidence,_=geometry_wrapper_packet(); observer=f.Observer(); p=observer.projection
    with observer.budget.scope('geometry'): first=p.take(geometry); expected=frozen_record_body(geometry)
    evidence.signature_sha256='f'*64  # no trust in the stale parent result_sha256
    with observer.budget.scope('terminal'): second=p.take(geometry); p.validate((first,second))
    assert first!=second
    for reference,body in ((first,expected),(second,frozen_record_body(geometry))):
        definition=p.definitions[reference['semantic_reference']]
        assert definition['encoding']=='geometry-field-vector-v1'
        assert definition['content_sha256']==frozen_canonical_digest(body)
        assert legacy_logical_wire(definition['projection'],p.definitions,definition['encoding'])==body


def test_geometry_exact_source_schema_and_independent_oracle_pins():
    for qualified,expected in f.GEOMETRY_RECORD_SCHEMAS.items():
        module,name=qualified.removeprefix('mc_photogrammetry.').rsplit('.',1)
        path=ROOT/'src/mc_photogrammetry'/Path(module.replace('.','/')+'.py')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==f.GEOMETRY_SOURCE_PINS[module.replace('.','/')+'.py']
        node=next(item for item in ast.parse(path.read_text()).body if isinstance(item,ast.ClassDef) and item.name==name)
        actual=tuple((item.target.id,ast.unparse(item.annotation)) for item in node.body if isinstance(item,ast.AnnAssign))
        assert actual==tuple(zip(*expected))
    seven=f.GEOMETRY_RECORD_SCHEMAS['mc_photogrammetry.pipeline.image_derived_signatures.SignatureEvidence']
    assert seven[0]==('frame_id','canonical_index','reading_axis','alternative_index',
                      'blocks_per_period','signature','signature_sha256')
    expected={'frozen_native_fingerprint':'25537010dc5132d4a62b9f8cbe330ed73c5e01f49fd517009eb9bc54fca61703',
              'frozen_prechange_projection':'254b06c71d98a9fc04dabae94722ccab743db2f0e65def9f296b655abb0d9a35',
              'frozen_canonical_digest':'13cfc794f82bedb1a7198b391390f907112d787dcafd517e9888bd0991828f24',
              'frozen_record_body':'3cf9dc2f4d27dafc39207173e3f9bce539dd8191294e349f9c2059b7ce8346c3'}
    tree=ast.parse(Path(__file__).read_text())
    for name,pin in expected.items():
        node=next(item for item in tree.body if isinstance(item,ast.FunctionDef) and item.name==name)
        assert hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()==pin


@pytest.mark.parametrize('where',['cached-root','final-validation'])
@pytest.mark.parametrize('which',['annotation','order','namespace','spoof-annotation'])
def test_cached_nested_geometry_metadata_drift_refuses(where,which):
    geometry,_,_,evidence,_=geometry_wrapper_packet(); observer=f.Observer(); p=observer.projection
    with observer.budget.scope('geometry'): reference=p.take(geometry)
    before=frozen_native_fingerprint(geometry)
    if which=='annotation': type(evidence).__dataclass_fields__['alternative_index'].type='float'
    elif which=='spoof-annotation':
        class Spoof:
            def __eq__(self,other): raise AssertionError('foreign annotation equality must never execute')
        type(evidence).__dataclass_fields__['alternative_index'].type=Spoof()
    elif which=='order':
        declared=type(evidence).__dataclass_fields__; type(evidence).__dataclass_fields__=dict(reversed(tuple(declared.items())))
    elif which=='namespace': type(evidence).__module__='foreign'
    if which=='annotation': assert frozen_native_fingerprint(geometry)==before
    with observer.budget.scope('geometry' if where=='cached-root' else 'terminal'):
        with pytest.raises(f.IntegrityError):
            if where=='cached-root': p.take(geometry)
            else: p.validate((reference,))
    assert p.geometry_live_nodes==0


def test_geometry_live_class_receipts_bounded_inside_registry_reservation():
    observer=f.Observer(); p=observer.projection; borrowed=[]
    with observer.budget.scope('geometry'):
        p.ensure_geometry_registry(); before=(observer.budget.items,observer.budget.reserved)
        for index in range(f.GEOMETRY_CLASS_LIMIT):
            value=source_shaped_record('SignatureEvidence'); borrowed.append(type(value)); p.geometry_schema(value)
        assert set(p.geometry_classes)==set(borrowed) and len(p.geometry_classes)==64
        # No further observer-retained table/node debit: bounded slots already
        # included in16384B/256items, never silently exceeding that reservation.
        assert (observer.budget.items,observer.budget.reserved)==before
        assert sys.getsizeof(p.geometry_classes)<=4096
        with pytest.raises(f.IntegrityError,match='geometry live class receipt cap'):
            p.geometry_schema(source_shaped_record('SignatureEvidence'))


@pytest.mark.parametrize('point',['cache','final','late-encoding','in-flight-cache','late-cache-walk','late-postwalk'])
def test_nested_schema_coordinator_failure_seals_false_once(point,monkeypatch):
    geometry,_,_,evidence,_=geometry_wrapper_packet(); observed={}; calls=[]; public=[]; metadata=[]
    original_init=f.Observer.__init__
    def init(self,*args,**kwargs): original_init(self,*args,**kwargs); observed['observer']=self
    monkeypatch.setattr(f.Observer,'__init__',init)
    mutated=False
    def mutate(): type(evidence).__dataclass_fields__['alternative_index'].type='float'
    if point=='late-encoding':
        original=f.StageProjection.geometry_tokens
        def tokens(self,value,key='',depth=0):
            nonlocal mutated
            yield from original(self,value,key,depth)
            if key=='result_sha256' and value==geometry.result_sha256 and not mutated:
                mutate(); mutated=True  # descendant own encoding already ended
        monkeypatch.setattr(f.StageProjection,'geometry_tokens',tokens)
    elif point in ('in-flight-cache','late-cache-walk','late-postwalk'):
        original=f.StageProjection.geometry_upgrade
        def upgrade(self,value,depth=0):
            nonlocal mutated
            if point=='in-flight-cache' and value is geometry and observed.get('cache_requested') and not mutated:
                mutate(); mutated=True
            result=original(self,value,depth)
            if value is geometry and not mutated and (point=='late-postwalk' or
                    point=='late-cache-walk' and observed.get('cache_requested')):
                mutate(); mutated=True
            return result
        monkeypatch.setattr(f.StageProjection,'geometry_upgrade',upgrade)
    def geometry_stage(): calls.append('geometry'); return geometry
    def output_stage(value): calls.append('output'); return Outcome()
    module=SimpleNamespace(geometry=geometry_stage,output=output_stage)
    def predict(paths,**kwargs):
        public.append(1); result=module.geometry()
        if point=='cache':
            mutate()
            with observed['observer'].budget.scope('geometry'): observed['observer'].projection.take(result)
        elif point=='final':
            observed['stale_ref']=observed['observer'].stages['geometry']['records'][0]['result']; mutate()
            return Outcome()  # no recapture: final validate alone must reject
        elif point in ('in-flight-cache','late-cache-walk'):
            observed['cache_requested']=True
            with observed['observer'].budget.scope('geometry'): observed['observer'].projection.take(result)
        # Genuine unavailable branch never invokes output.
        return source_shaped_record('ImageDerivedProductionResult',geometry=result)
    result,storage=run(predict,targets=((module,'geometry','geometry'),(module,'output','output')),loader=lambda:metadata.append(1))
    raw=raw_native_record(storage); marker=f.strict_json(storage.files[Path('/memory/stub/COMPLETE.json')])
    assert public==[1] and calls==['geometry'] and not metadata and not result['metadata_access']
    assert raw['stages']['geometry']['calls']==1 and raw['stages']['output']['calls']==0
    assert result['state']=='SEALED_INSTRUMENTATION_FAILURE' and marker['native_complete'] is False
    assert raw['instrumentation_failure']['message']=='geometry schema fields/annotations'
    if point=='final': assert observed['stale_ref']['semantic_reference'] in raw['semantic_definitions']
    assert observed['observer'].projection.geometry_live_nodes==0
