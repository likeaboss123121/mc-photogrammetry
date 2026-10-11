"""Guarded injected-graph third-view proposal experiment; main is unexecuted.

No extraction, real images, source edits, camera priors or production integration.
Guard tests never call execute_once. A future separate run grant must freeze this
file, source dependencies, environment and packet specification. Results are
transient, bounded bytes plus Comms; there is no cross-process durable claim.
"""
from __future__ import annotations

import argparse
import ast
import copy
from dataclasses import asdict, dataclass, fields, is_dataclass
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import platform
import sys
import time
from types import CodeType, FunctionType

ROOT = Path(__file__).resolve().parents[2]
PINS = {
    'src/mc_photogrammetry/pipeline/joint_matching.py': '9303f56ab194600c58a27d97321fa2f2e4323e4bb78276a2df0372e8ed8ae4e3',
    'src/mc_photogrammetry/pipeline/connected_camera.py': '3b5df735084daddccb7f25511af1b8549c0300d02e4841ade39e39461b8275a6',
    'src/mc_photogrammetry/pipeline/correspondence.py': '192a4851e428d8116e0fbffd28c7ff157d1aedd8035e499c6df7ec2a9d787fa1',
    'src/mc_photogrammetry/pipeline/contracts.py': 'db24e730bfed52ec983ae387a26f1b612fd94abc3e7608053080915c899ed5cc',
    'src/mc_photogrammetry/reconstruction/camera.py': '0515c5bd82f01d3dc174ced6cb54405adef0a755b94c577dae513fc53c4fe680',
}
MODULES = (
    'mc_photogrammetry.pipeline.joint_matching',
    'mc_photogrammetry.pipeline.connected_camera',
    'mc_photogrammetry.pipeline.correspondence',
    'mc_photogrammetry.pipeline.contracts',
    'mc_photogrammetry.reconstruction.camera',
)
MAX_EDGES, MAX_ROWS, MAX_TRIANGLES = 192, 32, 128
MAX_RESIDUAL_ROWS, MAX_SVD = 1_000_000, 40_000
MAX_EVIDENCE, MANAGED_RESERVATION = 16*1024*1024, 64*1024*1024
# Source-pinned schema ceilings, charged before constructing each section.
# Per class: <=4097 generation events*128; <=514 candidates*(384 capture
# +192 original prefix +192 seed prefix +128 scored); <=517 residual digests
# *256; <=3 refits*2048; and8192 other keys/scalars. This is <1.3MB.
CLASS_OUTPUT_RESERVATION = 1_300_000
# Two12-score diagnostic lists (<=20 fields*192B), four DS-11 class records,
# <=12 mode tuples, bounded32-row arrays and other scalar keys fit128KiB.
ARM_OUTPUT_RESERVATION = 131_072
ENVELOPE_OUTPUT_RESERVATION = 262_144
MAIN_OUTPUT_RESERVATION = 8*CLASS_OUTPUT_RESERVATION+2*ARM_OUTPUT_RESERVATION+ENVELOPE_OUTPUT_RESERVATION
PACKET_SPEC = {
    'version':'mv-proposal-seed-1', 'rows':32, 'size':[256,256],
    'focal':100, 'principal':[128,128], 'projection':'u=cx+fx*X/Z;v=cy-fy*Y/Z',
    'points':'x=2*(i%4)-3;y=(2*(i//4)-7)/2;z=4+(i%3)/2;i=0..31',
    'centers':[[0,0,0],[1,0,0],[0.25,1,0]],
    'AB':'(i,(i+d)%32),d=1,5,9,13;plus(i,i),i<16',
    'AC':'(i,i),i<16', 'BC':'(i,i),i<16', 'distance':1,
    'descriptor':'float32[32,128]=32', 'sizes':8,
    'settings':'source JointMatchSettings()/PairGates(), ratio=.75',
    'treatment':'append one seed/class after generation before scored=[]',
    'result_scope':'TRANSIENT_PLUS_COMMS',
}
_CONSUMED = False
FREEZE_LIMIT = 65_536


class IntegrityError(RuntimeError):
    pass


def require(condition, reason):
    if not condition:
        raise IntegrityError(reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def wire_size(value):
    """Exact serialized length without creating any hex expansion/output."""
    if type(value) is bytes: return 10+2*len(value)  # {"hex":"..."}
    if isinstance(value,dict):
        require(all(type(k) is str for k in value),'wire_keys')
        return 2+max(0,len(value)-1)+sum(len(canonical(k))+1+wire_size(v) for k,v in value.items())
    if isinstance(value,(list,tuple)):
        return 2+max(0,len(value)-1)+sum(map(wire_size,value))
    require(value is None or type(value) in (str,int,float,bool),'wire_scalar')
    return len(canonical(value))


def wire_tokens(value):
    if type(value) is bytes:
        yield canonical({'hex':value.hex()}); return
    if isinstance(value,dict):
        yield b'{'
        for i,key in enumerate(sorted(value)):
            if i: yield b','
            yield canonical(key)+b':'
            yield from wire_tokens(value[key])
        yield b'}'; return
    if isinstance(value,(tuple,list)):
        yield b'['
        for i,item in enumerate(value):
            if i: yield b','
            yield from wire_tokens(item)
        yield b']'; return
    yield canonical(value)


def emit_report(report,stream,limit=MAX_EVIDENCE):
    # Integrity/resource rejection happens before hex(), token creation or writes.
    require(type(limit) is int and 0<=limit<=MAX_EVIDENCE,'stdout_cap_raise')
    expected=wire_size(report)+1
    require(expected<=limit,'stdout_evidence_preflight')
    for token in wire_tokens(report): stream.write(token)
    stream.write(b'\n')
    return expected


def source_pins():
    actual = {name:sha((ROOT/name).read_bytes()) for name in PINS}
    require(actual == PINS,'source_drift')
    return actual


def code_key(code):
    """Semantic compiler fingerprint, including names/constants/bytecode/flags."""
    return (code.co_code,code.co_names,code.co_varnames,code.co_freevars,
            code.co_cellvars,code.co_flags,code.co_argcount,code.co_kwonlyargcount,
            code.co_posonlyargcount,tuple(code_key(x) if isinstance(x,CodeType)
                                        else (type(x).__name__,repr(x)) for x in code.co_consts))


def clone(fn, namespace):
    result = FunctionType(fn.__code__,namespace,fn.__name__,fn.__defaults__,fn.__closure__)
    result.__kwdefaults__ = None if fn.__kwdefaults__ is None else fn.__kwdefaults__.copy()
    result.__annotations__ = fn.__annotations__.copy()
    result.__qualname__,result.__module__ = fn.__qualname__,fn.__module__
    return result


def verify_module(module, relative):
    path = ROOT/relative
    require(Path(module.__file__).resolve() == path.resolve(),'module_origin')
    require(module.__spec__ is not None and Path(module.__spec__.origin).resolve() == path.resolve(),
            'module_spec_origin')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    future = [node for node in tree.body if isinstance(node,ast.ImportFrom) and node.module=='__future__']
    receipts,constants,classes = {},{},{}
    for statement in tree.body:
        if isinstance(statement,(ast.Assign,ast.AnnAssign)) and statement.value is not None:
            try: expected_value=ast.literal_eval(statement.value)
            except (ValueError,TypeError): continue
            targets=statement.targets if isinstance(statement,ast.Assign) else [statement.target]
            for target in targets:
                if isinstance(target,ast.Name):
                    require(getattr(module,target.id,None)==expected_value,'source_constant:'+target.id)
                    constants[target.id]=copy.deepcopy(expected_value)
    for node in tree.body:
        if not isinstance(node,(ast.FunctionDef,ast.ClassDef)):
            continue
        loaded = getattr(module,node.name,None)
        scratch = module.__dict__.copy()
        unit = ast.Module(body=copy.deepcopy(future)+[copy.deepcopy(node)],type_ignores=[])
        exec(compile(unit,str(path),'exec',dont_inherit=True),scratch)
        expected = scratch[node.name]
        if isinstance(node,ast.FunctionDef):
            require(isinstance(loaded,FunctionType) and loaded.__module__==module.__name__,
                    'function_identity:'+node.name)
            require(loaded.__globals__ is module.__dict__,'function_globals:'+node.name)
            require(code_key(loaded.__code__)==code_key(expected.__code__),'function_code:'+node.name)
            require(loaded.__defaults__==expected.__defaults__ and
                    loaded.__kwdefaults__==expected.__kwdefaults__ and
                    loaded.__annotations__==expected.__annotations__,'function_contract:'+node.name)
            receipts[node.name] = loaded
        elif is_dataclass(expected):
            require(is_dataclass(loaded) and loaded.__module__==module.__name__,'dataclass_identity')
            require([(f.name,f.type,f.init,f.default,f.default_factory) for f in fields(loaded)] ==
                    [(f.name,f.type,f.init,f.default,f.default_factory) for f in fields(expected)],
                    'dataclass_fields:'+node.name)
            require(loaded.__annotations__==expected.__annotations__,'class_annotations')
            for name in ('__init__','__post_init__'):
                if hasattr(expected,name):
                    require(code_key(getattr(loaded,name).__code__)==code_key(getattr(expected,name).__code__),
                            'dataclass_method:'+node.name+name)
            methods={}
            for name in set(['__init__','__post_init__']+[n.name for n in node.body if isinstance(n,ast.FunctionDef)]):
                if not hasattr(expected,name): continue
                left,right=getattr(loaded,name),getattr(expected,name)
                if isinstance(left,property): left,right=left.fget,right.fget
                if inspect.ismethod(left): left,right=left.__func__,right.__func__
                require(code_key(left.__code__)==code_key(right.__code__),'class_method_code')
                methods[name]=(left,left.__code__)
            classes[node.name]=(loaded,copy.deepcopy(loaded.__annotations__),methods)
    return receipts,constants,classes


@dataclass(frozen=True)
class Runtime:
    joint: object
    ds11: object
    correspondence: object
    contracts: object
    camera: object
    np: object
    functions: tuple
    identity: dict
    constants: tuple
    classes: tuple
    svd_identity: object

    def attest(self):
        source_pins()
        require('cv2' not in sys.modules,'forbidden_cv2_import')
        for module,name,fn,code,defaults,kwdefaults,annotations in self.functions:
            require(getattr(module,name,None) is fn and fn.__code__ is code and
                    fn.__defaults__==defaults and fn.__kwdefaults__==kwdefaults and
                    fn.__annotations__==annotations and fn.__globals__ is module.__dict__,
                    'loaded_function_tamper:'+name)
        for module,name,value in self.constants:
            require(getattr(module,name,None)==value,'loaded_constant_tamper:'+name)
        for module,name,cls,annotations,methods in self.classes:
            require(getattr(module,name,None) is cls and cls.__annotations__==annotations,'loaded_class_tamper')
            for method,(original,code) in methods.items():
                current=getattr(cls,method)
                if isinstance(current,property): current=current.fget
                if inspect.ismethod(current): current=current.__func__
                require(current is original and current.__code__ is code,'loaded_class_method_tamper')
        require(self.joint.solve_pair is self.ds11.solve_pair,'DS11_binding')
        require(self.joint.FrameFeatures is self.correspondence.FrameFeatures and
                self.joint.ReciprocalGraph is self.correspondence.ReciprocalGraph,
                'correspondence_type_binding')
        require(self.joint.CameraEstimate is self.contracts.CameraEstimate and
                self.joint.CameraIntrinsics is self.contracts.CameraIntrinsics and
                self.ds11.CameraEstimate is self.contracts.CameraEstimate and
                self.ds11.CameraPose is self.contracts.CameraPose and
                self.joint.CameraModel is self.camera.CameraModel and
                self.joint.camera_rotation_matrix is self.camera.camera_rotation_matrix and
                self.ds11.camera_rotation_matrix is self.camera.camera_rotation_matrix,'camera_binding')
        require(all(module.np is self.np for module in (self.joint,self.ds11,self.correspondence,self.camera)),
                'numpy_binding')
        require(self.np.linalg.svd is self.svd_identity,'SVD_runtime_identity')


def load_runtime():
    source_pins()
    require('cv2' not in sys.modules,'forbidden_cv2_import')
    modules = tuple(importlib.import_module(name) for name in MODULES)
    records,constants,classes = [],[],[]
    for module,relative in zip(modules,PINS):
        functions,literals,types=verify_module(module,relative)
        constants.extend((module,name,value) for name,value in literals.items())
        classes.extend((module,name,cls,annotations,methods) for name,(cls,annotations,methods) in types.items())
        for name,fn in functions.items():
            records.append((module,name,fn,fn.__code__,fn.__defaults__,
                            None if fn.__kwdefaults__ is None else fn.__kwdefaults__.copy(),fn.__annotations__.copy()))
    np = modules[0].np
    identity = {'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
                'flags':repr(sys.flags),
                'numpy':np.__version__,'numpy_origin':np.__file__,
                'svd_module':np.linalg.svd.__module__,
                'svd_code':sha(repr(code_key(np.linalg.svd.__wrapped__.__code__)).encode())
                    if hasattr(np.linalg.svd,'__wrapped__') else 'UNAVAILABLE',
                'settings':asdict(modules[0].JointMatchSettings()),
                'gates':asdict(modules[1].PairGates())}
    runtime = Runtime(*modules,np,tuple(records),identity,tuple(constants),tuple(classes),np.linalg.svd)
    runtime.attest()
    return runtime


def inserted_class_modes(source_text):
    tree = ast.parse(source_text)
    nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_class_modes']
    require(len(nodes)==1,'class_modes_ast_unique')
    original = nodes[0]
    transformed = copy.deepcopy(original)
    indices = [i for i,n in enumerate(transformed.body) if
               (isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id=='scored') or
               (isinstance(n,ast.Assign) and len(n.targets)==1 and
                isinstance(n.targets[0],ast.Name) and n.targets[0].id=='scored')]
    require(len(indices)==1,'scoring_seam_unique')
    at = indices[0]
    require(isinstance(transformed.body[at-1],ast.While) and
            isinstance(transformed.body[at+1],ast.For) and
            ast.unparse(transformed.body[at+1].iter)=='enumerate(candidates)', 'scoring_seam_shape')
    call = ast.parse('_evaluation_append_seed(candidates, rows, norms, graph, bounds, support_floor)').body[0]
    ast.copy_location(call,transformed.body[at])
    transformed.body.insert(at,call)
    inverse = copy.deepcopy(transformed)
    del inverse.body[at]
    require(ast.dump(inverse,include_attributes=True)==ast.dump(original,include_attributes=True),
            'inverse_ast_mismatch')
    ast.fix_missing_locations(transformed)
    return original,transformed,at


def compile_inserted(namespace, source_text=None):
    if source_text is None:
        source_text = (ROOT/next(iter(PINS))).read_text(encoding='utf-8')
    original,node,_ = inserted_class_modes(source_text)
    future = ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    unit = ast.fix_missing_locations(ast.Module(body=[future,node],type_ignores=[]))
    scratch = namespace.copy()
    exec(compile(unit,str(ROOT/next(iter(PINS))),'exec',dont_inherit=True),scratch)
    fn = scratch['_class_modes']
    # A function's code is compiled in scratch but runs in the explicit closure.
    return clone(fn,namespace)


def readonly(np, values, dtype):
    value = np.ascontiguousarray(values,dtype=dtype)
    return np.frombuffer(value.tobytes(),dtype=value.dtype).reshape(value.shape)


def array_key(value):
    return (value.dtype.str,tuple(value.shape),tuple(value.strides),bool(value.flags.writeable),
            sha(value.tobytes(order='C')))


def immutable_array(np,value):
    require(isinstance(value,np.ndarray) and not value.flags.writeable and value.flags.c_contiguous,
            'immutable_array_flags')
    owner=value
    while isinstance(owner,np.ndarray):
        require(not owner.flags.writeable,'mutable_base')
        owner=owner.base
    require(type(owner) is bytes,'reopenable_array_owner')


def validate_edges(edges, counts):
    require(type(edges) is tuple and len(edges)<=MAX_EDGES,'edge_container_cap')
    require(len(set(edges))==len(edges),'duplicate_edge')
    degree = ({},{})
    for edge in edges:
        require(type(edge) is tuple and len(edge)==2 and
                all(type(x) is int for x in edge),'edge_schema')
        require(0<=edge[0]<counts[0] and 0<=edge[1]<counts[1],'edge_index')
        for side in (0,1):
            degree[side][edge[side]] = degree[side].get(edge[side],0)+1
            require(degree[side][edge[side]]<=5,'edge_degree_cap')
    require(edges==tuple(sorted(edges)),'edge_order')


@dataclass(frozen=True)
class Closure:
    triangles: tuple
    eligible_indices: tuple
    quarantined: tuple
    probes: int


def closed_triangles(ab,ac,bc,counts):
    require(type(counts) is tuple and len(counts)==3 and
            all(type(n) is int and 0<n<=MAX_ROWS for n in counts),'row_counts')
    for edges,pair in ((ab,(0,1)),(ac,(0,2)),(bc,(1,2))):
        validate_edges(edges,(counts[pair[0]],counts[pair[1]]))
    ac_set = set(ac)
    by_b = {}
    for b,c in bc: by_b.setdefault(b,[]).append(c)
    triangles,probes = [],0
    for index,(a,b) in enumerate(ab):
        for c in by_b.get(b,()):
            probes += 1
            require(probes<=1024,'closure_work_cap')
            if (a,c) in ac_set:
                require(len(triangles)<MAX_TRIANGLES,'triangle_cap')
                triangles.append((a,b,c,index))
    parents = {}
    def find(key):
        while parents[key]!=key: key=parents[key]
        return key
    for a,b,c,_ in triangles:
        nodes = ((0,a),(1,b),(2,c))
        for node in nodes: parents.setdefault(node,node)
        roots = tuple(find(node) for node in nodes)
        low = min(roots)
        for root in roots: parents[root]=low
    components = {}
    for node in parents: components.setdefault(find(node),[]).append(node)
    bad = set()
    quarantine = []
    for root,nodes in sorted(components.items()):
        if any(sum(node[0]==frame for node in nodes)>1 for frame in range(3)):
            bad.add(root); quarantine.append(tuple(sorted(nodes)))
    per_target = {}
    for a,b,c,index in triangles: per_target.setdefault(index,[]).append(c)
    eligible = tuple(index for index in sorted(per_target) if len(per_target[index])==1
                     and find((0,ab[index][0])) not in bad)
    return Closure(tuple(triangles),eligible,tuple(quarantine),probes)


@dataclass
class Meter:
    residual_rows: int = 0
    svd_calls: int = 0
    evidence_bytes: int = 0
    transient_bytes: int = 0
    residual_limit: int = MAX_RESIDUAL_ROWS
    svd_limit: int = MAX_SVD
    evidence_limit: int = MAX_EVIDENCE
    output_reserved: int = 0
    output_limit: int = MAX_EVIDENCE
    output_claimed: int = 0
    output_preallocated: bool = False
    started: float | None = None
    clock: object = time.monotonic
    roles: dict | None = None

    def __post_init__(self):
        require(0<=self.residual_limit<=MAX_RESIDUAL_ROWS and 0<=self.svd_limit<=MAX_SVD and
                0<=self.evidence_limit<=MAX_EVIDENCE and
                0<=self.output_claimed<=self.output_reserved<=self.output_limit<=MAX_EVIDENCE,
                'meter_cap_raise')
        if self.started is None: self.started=self.clock()
        if self.roles is None: self.roles={}

    def check(self):
        require(self.clock()-self.started<=30.0,'watchdog')

    def residual(self,n,role):
        self.check(); require(0<=n<=MAX_EDGES,'residual_shape')
        require(self.residual_rows+n<=self.residual_limit,'residual_precharge')
        self.residual_rows+=n; self.roles[role]=self.roles.get(role,0)+n

    def svd(self,n,full,role):
        self.check(); require(0<n<=MAX_EDGES,'svd_shape')
        require(self.svd_calls+1<=self.svd_limit,'svd_precharge')
        scratch = (n*n if full else n*min(n,3))*8 + 3*n*8 + 96
        require(32*1024*1024+self.evidence_bytes+self.transient_bytes+scratch<=MANAGED_RESERVATION,
                'managed_precharge')
        self.svd_calls+=1; self.roles[role]=self.roles.get(role,0)+1
        self.transient_bytes+=scratch
        return scratch

    def release(self,n):
        require(0<=n<=self.transient_bytes,'transient_release')
        self.transient_bytes-=n

    def reserve_output(self,n):
        self.check()
        require(type(n) is int and n>=0 and self.output_reserved+n<=self.output_limit,
                'output_precharge')
        self.output_reserved+=n

    def preallocate_output(self,n):
        require(self.output_reserved==self.output_claimed==0 and not self.output_preallocated,
                'output_plan_repeated')
        self.reserve_output(n)
        self.output_preallocated=True

    def claim_output(self,n):
        self.check()
        require(type(n) is int and n>=0,'output_claim_schema')
        if not self.output_preallocated: self.reserve_output(n)
        require(self.output_claimed+n<=self.output_reserved,'output_claim')
        self.output_claimed+=n

    def capture(self,np,value):
        self.check()
        require(isinstance(value,np.ndarray),'capture_array')
        require(self.evidence_bytes+value.nbytes<=self.evidence_limit,'evidence_precharge')
        self.evidence_bytes+=value.nbytes
        data=value.tobytes(order='C')
        return {'dtype':value.dtype.str,'shape':tuple(value.shape),'data':data,'sha256':sha(data)}

    def digest(self,np,value):
        """Hash returned residual bytes transiently; retain no vector/data copy."""
        self.check()
        require(isinstance(value,np.ndarray) and value.dtype==np.float64 and
                value.ndim==1 and len(value)<=MAX_EDGES,'digest_array')
        self.metadata(256)
        require(32*1024*1024+self.evidence_bytes+self.transient_bytes+value.nbytes<=MANAGED_RESERVATION,
                'digest_managed_precharge')
        data=value.tobytes(order='C')
        return {'dtype':value.dtype.str,'shape':tuple(value.shape),'sha256':sha(data)}

    def metadata(self,n):
        self.check()
        require(type(n) is int and n>=0 and self.evidence_bytes+n<=self.evidence_limit,'metadata_precharge')
        self.evidence_bytes+=n


def work_bounds():
    return {'residual_A':488704,'residual_B':489472,'residual_AB':978176,
            'svd_joint_A':4*(2048+1+2048+30+3),
            'svd_joint_B':4*(2048+1+2048+30+3+2),
            'svd_DS11':4*(513+10+1),'svd_AB':37240,
            'full_U_max':192*192*8,'DS11_U_max':32*32*8}


def _forbidden(*args,**kwargs):
    raise IntegrityError('forbidden_acquisition')


def source_namespaces(runtime):
    """Clone exact source functions, preserving distinct helper namespaces."""
    runtime.attest()
    spaces=[]
    for module in (runtime.joint,runtime.ds11,runtime.correspondence):
        space=module.__dict__.copy()
        for name,fn in module.__dict__.items():
            if isinstance(fn,FunctionType) and fn.__module__==module.__name__:
                space[name]=clone(fn,space)
        spaces.append(space)
    joint,ds11,corr=spaces
    joint['solve_pair']=ds11['solve_pair']
    joint['select_mutual_matches']=corr['select_mutual_matches']
    for space in spaces:
        for name in ('match_and_solve_pair','detect_features','usable_mask_for','extract_pair_matches',
                     'boundary_distances','descriptor_footprint_keep','_validate_frame','_to_grey'):
            if name in space: space[name]=_forbidden
    return joint,ds11,corr


class SeedAppender:
    def __init__(self,np,indices,rank_fit,null_fit,meter,recorder):
        self.np,self.indices=np,tuple(indices)
        self.rank_fit,self.null_fit=rank_fit,null_fit
        self.meter,self.recorder=meter,recorder

    def __call__(self,candidates,rows,norms,graph,bounds,support_floor):
        before=(array_key(rows),array_key(norms),array_key(graph.edges),array_key(graph.distances))
        prefix=tuple(array_key(x) for x in candidates)
        prefix_ids=tuple(id(x) for x in candidates)
        require(len(candidates)<=513,'original_candidate_cap')
        require(len(set(self.indices))==len(self.indices) and
                all(0<=x<len(rows) for x in self.indices),'seed_indices')
        seed=None; reason='no_unambiguous_rows'
        if len(self.indices)>=2:
            edges=graph.edges[list(self.indices)]
            require(len(set(map(int,edges[:,0])))==len(edges) and
                    len(set(map(int,edges[:,1])))==len(edges),'seed_endpoint_conflict')
            subset=rows[list(self.indices)]
            if self.rank_fit(subset,bounds.min_singular_ratio):
                seed=self.null_fit(subset)
                if seed is not None:
                    require(seed.shape==(3,) and self.np.all(self.np.isfinite(seed)) and
                            float(self.np.linalg.norm(seed))>0,'seed_fit_contract')
                    candidates.append(seed); reason='appended'
                else: reason='null_fit_none'
            else: reason='rank_deficient'
        require(len(candidates)<=514,'seed_candidate_cap')
        require(tuple(id(x) for x in candidates[:len(prefix)])==prefix_ids and
                tuple(array_key(x) for x in candidates[:len(prefix)])==prefix,'prefix_mutation')
        require(before==(array_key(rows),array_key(norms),array_key(graph.edges),array_key(graph.distances)),
                'seed_input_mutation')
        self.recorder(prefix,seed,reason)


class Driver:
    """Source closure and transparent trace. Creating it does not score a packet."""
    def __init__(self,runtime,arm,graph,left_descriptors,right_descriptors,eligible,meter):
        require(arm in ('A','B'),'arm')
        np=runtime.np
        require(isinstance(graph,runtime.correspondence.ReciprocalGraph),'graph_type')
        for d in (left_descriptors,right_descriptors):
            immutable_array(np,d)
            require(d.dtype==np.float32 and d.ndim==2 and d.shape[1]==128 and
                    0<len(d)<=MAX_ROWS and np.all(np.isfinite(d)),'descriptor_boundary')
        for a in (graph.edges,graph.distances): immutable_array(np,a)
        require(graph.edges.dtype==np.int64 and graph.edges.shape==(len(graph.edges),2) and
                graph.distances.dtype==np.float64 and graph.distances.shape==(len(graph.edges),) and
                np.all(graph.distances==1.0),'graph_arrays')
        require(graph.truncated==0 and graph.total_edges==len(graph.edges) and
                graph.left_keypoints==len(left_descriptors) and graph.right_keypoints==len(right_descriptors),
                'graph_counters')
        validate_edges(tuple(tuple(int(x) for x in row) for row in graph.edges),
                       (len(left_descriptors),len(right_descriptors)))
        require(tuple(eligible)==tuple(sorted(set(eligible))) and
                all(type(i) is int and 0<=i<len(graph.edges) for i in eligible),'eligible_boundary')
        self.runtime,self.arm,self.graph,self.meter=runtime,arm,graph,meter
        self.eligible=tuple(eligible); self.classes=[]; self.current=None; self.final=None
        self.joint,self.ds11,self.corr=source_namespaces(runtime)
        self.left_descriptors,self.right_descriptors=left_descriptors,right_descriptors
        self._class_code=None; self._match_code=self.joint['match_feature_frames'].__code__
        self._original_joint=dict(self.joint); self._original_ds11=dict(self.ds11)
        self._graph_calls=0
        self.joint['reciprocal_topk_graph']=self.graph_provider
        for space,domain in ((self.joint,'joint'),(self.ds11,'DS11')):
            for name in ('_null_vector','_sample_rank_ok','_singular_values','_residuals'):
                if name in space:
                    space[name]=self._delegate(space[name],domain,name)
        self.joint['_select_and_refit']=self._refit_delegate(self.joint['_select_and_refit'])
        if arm=='B':
            self.joint['_evaluation_append_seed']=SeedAppender(
                runtime.np,eligible,self.joint['_sample_rank_ok'],self.joint['_null_vector'],meter,self.seed_record)
            self.joint['_class_modes']=compile_inserted(self.joint)
        self._class_code=self.joint['_class_modes'].__code__
        self.matcher=self.joint['match_feature_frames']
        # Both functions' globals are the explicit same joint namespace.
        require(self.matcher.__globals__ is self.joint and
                self.joint['_class_modes'].__globals__ is self.joint,'namespace_identity')
        self._binding_receipts=[]
        for space in (self.joint,self.ds11,self.corr):
            for name,value in space.items():
                contract=None
                if isinstance(value,FunctionType):
                    contract=(value.__code__,value.__defaults__,
                              None if value.__kwdefaults__ is None else value.__kwdefaults__.copy(),
                              value.__annotations__.copy())
                self._binding_receipts.append((space,name,value,contract))

    def attest(self):
        self.runtime.attest()
        for space,name,value,contract in self._binding_receipts:
            require(space.get(name) is value,'namespace_binding_tamper:'+name)
            if contract is not None:
                code,defaults,kwdefaults,annotations=contract
                require(value.__code__ is code and value.__defaults__==defaults and
                        value.__kwdefaults__==kwdefaults and value.__annotations__==annotations,
                        'namespace_callable_tamper:'+name)
        if self.arm=='B':
            callback=self.joint['_evaluation_append_seed']
            require(callback.indices==self.eligible and
                    callback.rank_fit is self.joint['_sample_rank_ok'] and
                    callback.null_fit is self.joint['_null_vector'],'seed_callback_binding')

    def graph_provider(self,left,right,**kwargs):
        require(self._graph_calls==0 and left is self.left_descriptors and right is self.right_descriptors,
                'graph_binding')
        require(kwargs=={'neighbors':5,'max_edges':8192,'distance_ceiling':None},'graph_settings')
        self._graph_calls+=1
        return self.graph

    def _delegate(self,fn,domain,name):
        np=self.runtime.np
        def wrapped(*args,**kwargs):
            self.meter.check()
            arrays=tuple(x for x in args if isinstance(x,np.ndarray))
            keys=tuple(array_key(x) for x in arrays)
            caller=inspect.currentframe().f_back
            role=domain+name+':'+caller.f_code.co_name+':'+str(caller.f_lineno)
            scratch=0
            if name=='_residuals':
                self.meter.residual(len(args[0]),role)
            elif args[0].size and (name!='_sample_rank_ok' or len(args[0])>=2):
                scratch=self.meter.svd(len(args[0]),name!='_sample_rank_ok',role)
            try:
                result=fn(*args,**kwargs)  # exact-code clone, once, unaltered return
            finally:
                self.meter.release(scratch)
            require(tuple(array_key(x) for x in arrays)==keys,'delegate_arg_mutation')
            if self.current is not None and domain=='joint' and caller.f_code is self._class_code:
                if name in ('_null_vector','_sample_rank_ok'):
                    self.meter.metadata(32)
                    sample=caller.f_locals.get('sample')
                    sample=None if sample is None else tuple(map(int,sample))
                    self.current['generation_calls'].append((name,sample,
                        result is not None if name=='_null_vector' else bool(result)))
                if name=='_null_vector' and result is not None:
                    self.current['_generated_objects'].append(result)
                if name=='_residuals':
                    self.current['residual_observations'].append(self.meter.digest(np,result))
            return result
        wrapped._delegate_target=fn
        return wrapped

    def _refit_delegate(self,fn):
        def wrapped(rows,norms,graph,candidate,bounds):
            require(self.current is not None,'refit_outside_class')
            prefix=self.current['_candidates_live']()
            origins=[i for i,x in enumerate(prefix) if x is candidate]
            require(len(origins)==1,'refit_origin_ambiguous')
            result=fn(rows,norms,graph,candidate,bounds)
            record={'candidate_order':origins[0],'returned':result is not None,
                    '_result':result}
            if result is not None:
                record['t']=self.meter.capture(self.runtime.np,result[0])
                record['selection']=self.meter.capture(self.runtime.np,result[1])
            self.current['refits'].append(record)
            return result
        wrapped._delegate_target=fn
        return wrapped

    def seed_record(self,prefix,seed,reason):
        require(self.current is not None and self.current['seed'] is None,'seed_context')
        self.current['seed']={'prefix':prefix,'order':len(prefix) if seed is not None else None,
                              'reason':reason,'vector':None if seed is None else self.meter.capture(self.runtime.np,seed)}

    def trace(self,frame,event,arg):
        if frame.f_code is self._class_code:
            if event=='call':
                require(self.current is None and len(self.classes)<4,'class_order')
                self.meter.claim_output(CLASS_OUTPUT_RESERVATION)
                context={'yaw':len(self.classes),'rows':array_key(frame.f_locals['rows']),
                         'norms':array_key(frame.f_locals['norms']),'seed':None,
                         'refits':[],'residual_observations':[],'generation_calls':[],'_generated_objects':[],
                         '_candidates_live':lambda:frame.f_locals.get('candidates',[])}
                self.current=context
            elif event=='return' and self.current is not None:
                context=self.current
                candidates=frame.f_locals.get('candidates',[])
                original=context['_generated_objects']
                require(len(original)<=513,'generated_candidate_cap')
                require(all(candidates[i] is x for i,x in enumerate(original)),'generation_prefix_identity')
                context['prefix']=tuple(array_key(x) for x in original)
                context['candidates']=tuple(self.meter.capture(self.runtime.np,x) for x in candidates)
                if self.arm=='B':
                    require(context['seed'] is not None and context['seed']['prefix']==context['prefix'],
                            'seed_prefix_lineage')
                scored=frame.f_locals.get('scored',[])
                context['scored']=tuple((int(c),float(m),int(o)) for c,m,o,_ in scored)
                kept=frame.f_locals.get('modes',[])
                context['kept_orders']=tuple(next(i for i,x in enumerate(candidates) if x is y) for y in kept)
                context['accepted']=frame.f_locals.get('accepted',0)
                context['attempts']=frame.f_locals.get('attempts',0)
                generator=frame.f_locals.get('generator')
                context['RNG_state']=None if generator is None else sha(canonical(generator.bit_generator.state))
                require(context['accepted']<=512 and context['attempts']<=2048,'proposal_work_cap')
                context['mode_origins']=[]
                for t,selection,_ in (arg or []):
                    matches=[r for r in context['refits'] if r['_result'] is not None and
                             r['_result'][0] is t and r['_result'][1] is selection]
                    require(len(matches)==1,'mode_refit_lineage')
                    context['mode_origins'].append(matches[0]['candidate_order'])
                for record in context['refits']: del record['_result']
                del context['_generated_objects']; del context['_candidates_live']
                require(len(context['generation_calls'])<=4097 and len(candidates)<=514 and
                        len(context['scored'])<=514 and len(context['residual_observations'])<=517 and
                        len(context['refits'])<=3 and len(context['kept_orders'])<=3,'class_receipt_counts')
                require(wire_size(context)<=CLASS_OUTPUT_RESERVATION,'class_output_reservation')
                self.classes.append(context); self.current=None
        elif frame.f_code is self._match_code and event=='return':
            self.meter.claim_output(ARM_OUTPUT_RESERVATION)
            best=frame.f_locals.get('best')
            modes=frame.f_locals.get('modes',[])
            self.final={'mode_order':tuple((m.yaw_class,m.mode_index,m.support,
                           m.spatial_bins_left,m.spatial_bins_right,m.median_selected_residual,
                           m.descriptor_cost,m.median_all_edge_residual,m.parallax,m.cheirality,
                           m.singular_ratio,m.baseline_direction) for m in modes),
                        'scores':tuple(asdict(x) for x in frame.f_locals.get('scores',())),
                        'winner':None,'result':self._result_record(arg)}
            if best is not None:
                require(best.yaw_class<len(self.classes),'winner_class_lineage')
                orders=self.classes[best.yaw_class]['mode_origins']
                require(best.mode_index<len(orders),'winner_mode_lineage')
                order=orders[best.mode_index]
                seed=self.classes[best.yaw_class]['seed']
                self.final['winner']={'yaw':best.yaw_class,'mode_index':best.mode_index,
                    'candidate_order':order,'seed_origin':seed is not None and seed['order']==order,
                    'eligible_selected':len(set(map(int,best.selection)) & set(self.eligible)),
                    'selection':self.meter.capture(self.runtime.np,best.selection),
                    'matches':self.meter.capture(self.runtime.np,best.matches)}
            require(wire_size(self.final)<=ARM_OUTPUT_RESERVATION,'arm_output_reservation')
        return self.trace if frame.f_code in (self._class_code,self._match_code) else None

    def _result_record(self,result):
        require(isinstance(result,self.runtime.joint.JointPairResult),'result_identity')
        record={f.name:getattr(result,f.name) for f in fields(result) if f.name not in ('matches','pose','hypotheses')}
        record['matches']=self.meter.capture(self.runtime.np,result.matches)
        record['hypotheses']=tuple(asdict(x) for x in result.hypotheses)
        record['pose']=None
        if result.pose is not None:
            pose=result.pose
            record['pose']={f.name:getattr(pose,f.name) for f in fields(pose) if f.name not in ('relative_rotation','classes')}
            record['pose']['classes']=tuple(asdict(x) for x in pose.classes)
            record['pose']['relative_rotation']=None if pose.relative_rotation is None else self.meter.capture(
                self.runtime.np,pose.relative_rotation)
        return record

    def run(self,left,right,lc,rc):
        self.attest()
        require(sys.gettrace() is None,'preexisting_trace')
        bounds,gates=self.runtime.joint.JointMatchSettings(),self.runtime.ds11.PairGates()
        before=tuple(array_key(x) for x in (left.points,right.points,left.descriptors,right.descriptors,
                                         self.graph.edges,self.graph.distances))
        sys.settrace(self.trace)
        try:
            result=self.matcher(left,right,lc,rc,matcher=bounds,gates=gates,ratio_test=.75)
        finally:
            sys.settrace(None)
        require(tuple(array_key(x) for x in (left.points,right.points,left.descriptors,right.descriptors,
                          self.graph.edges,self.graph.distances))==before,'arm_input_mutation')
        require(self._graph_calls==1 and len(self.classes)==4 and self.final is not None,'arm_trace_complete')
        self.attest()
        return result


@dataclass(frozen=True)
class Packet:
    frames: tuple
    cameras: tuple
    graphs: tuple
    edge_records: tuple
    closure: Closure
    input_sha256: str


def build_fixed_packet(runtime):
    """Not called by guard tests; immutable synthetic source input construction."""
    np=runtime.np
    physical=np.array([(2*(i%4)-3,(2*(i//4)-7)/2,4+(i%3)/2) for i in range(32)],dtype=np.float64)
    frames=[]
    for name,center in zip(('A','B','C'),PACKET_SPEC['centers']):
        relative=physical-np.array(center,dtype=np.float64)
        pixels=np.column_stack((128+100*relative[:,0]/relative[:,2],128-100*relative[:,1]/relative[:,2]))
        # Producer values are not retained in Packet; matching receives only pixels.
        points=readonly(np,pixels,np.float64)
        descriptors=readonly(np,np.full((32,128),32,dtype=np.float32),np.float32)
        frames.append(runtime.correspondence.FrameFeatures(name,points,descriptors,256,256,65536,
                      readonly(np,np.full(32,8,dtype=np.float64),np.float64),0))
    intrinsics=runtime.contracts.CameraIntrinsics((256,256),(100.,100.),(128.,128.))
    cameras=tuple(runtime.contracts.CameraEstimate(intrinsics,
                  runtime.contracts.CameraPose(None,0.,0.,0.)) for _ in range(3))
    ab=tuple(sorted(tuple((i,(i+d)%32) for i in range(32) for d in (1,5,9,13))+
                    tuple((i,i) for i in range(16))))
    ac=bc=tuple((i,i) for i in range(16))
    closure=closed_triangles(ab,ac,bc,(32,32,32))
    require(len(ab)==144 and len(ac)==len(bc)==16 and len(closure.triangles)==16 and
            len(closure.eligible_indices)==16,'fixed_packet_counts')
    graphs=tuple(runtime.correspondence.ReciprocalGraph(readonly(np,edges,np.int64),
                 readonly(np,np.ones(len(edges)),np.float64),32,32,len(edges),0) for edges in (ab,ac,bc))
    frames=tuple(frames)
    return Packet(frames,cameras,graphs,(ab,ac,bc),closure,packet_digest(frames,cameras,graphs))


def packet_digest(frames,cameras,graphs):
    return sha(canonical({'frames':[(array_key(f.points),array_key(f.descriptors),array_key(f.sizes))
                                   for f in frames],
                         'graphs':[(array_key(g.edges),array_key(g.distances)) for g in graphs],
                         'cameras':[asdict(c) for c in cameras],'spec':PACKET_SPEC}))


def validate_packet(runtime,packet):
    """Single built packet, checked before A; no externally expected byte hash."""
    require(type(packet) is Packet and all(type(v) is tuple and len(v)==3 for v in
            (packet.frames,packet.cameras,packet.graphs,packet.edge_records)),'packet_counts')
    np=runtime.np
    for name,frame,camera in zip(('A','B','C'),packet.frames,packet.cameras):
        require(type(frame) is runtime.correspondence.FrameFeatures and frame.frame_id==name and
                (frame.width,frame.height)==(256,256) and frame.usable_pixels==65536 and
                frame.boundary_dropped==0,'packet_frame_primitives')
        for array,dtype,shape in ((frame.points,np.float64,(32,2)),
                                  (frame.descriptors,np.float32,(32,128)),(frame.sizes,np.float64,(32,))):
            immutable_array(np,array)
            require(array.dtype==dtype and array.shape==shape and np.all(np.isfinite(array)),
                    'packet_frame_arrays')
        require(np.all(frame.descriptors==32) and np.all(frame.sizes==8),'packet_descriptor_recipe')
        require(type(camera) is runtime.contracts.CameraEstimate and
                type(camera.intrinsics) is runtime.contracts.CameraIntrinsics and
                type(camera.pose) is runtime.contracts.CameraPose,'packet_camera_types')
        require(asdict(camera)=={'intrinsics':{'image_size':(256,256),'focal_length_pixels':(100.,100.),
                'principal_point_pixels':(128.,128.)},'pose':{'position':None,'yaw_degrees':0.,
                'pitch_degrees':0.,'roll_degrees':0.},'residual_pixels':None,'confidence':0.},
                'packet_camera_primitives')
    ab=tuple(sorted(tuple((i,(i+d)%32) for i in range(32) for d in (1,5,9,13))+
                    tuple((i,i) for i in range(16))))
    identical=tuple((i,i) for i in range(16))
    expected=(ab,identical,identical)
    require(packet.edge_records==expected,'packet_edge_recipe')
    for graph,edges in zip(packet.graphs,expected):
        require(type(graph) is runtime.correspondence.ReciprocalGraph,'packet_graph_type')
        for array,dtype,shape in ((graph.edges,np.int64,(len(edges),2)),
                                  (graph.distances,np.float64,(len(edges),))):
            immutable_array(np,array)
            require(array.dtype==dtype and array.shape==shape,'packet_graph_arrays')
        require(tuple(tuple(map(int,row)) for row in graph.edges)==edges and
                np.all(graph.distances==1.) and graph.left_keypoints==graph.right_keypoints==32 and
                graph.total_edges==len(edges) and graph.truncated==0,'packet_graph_recipe')
    require(type(packet.closure) is Closure and packet.closure==closed_triangles(*expected,(32,32,32)),
            'packet_closure')
    observed=packet_digest(packet.frames,packet.cameras,packet.graphs)
    require(packet.input_sha256==observed,'packet_observed_digest_integrity')
    return observed


@dataclass(frozen=True)
class RunFreeze:
    harness_sha256: str
    guard_tests_sha256: str
    source_hashes: dict
    environment_sha256: str
    specification_sha256: str
    command_sha256: str
    external_go_sha256: str

    @classmethod
    def from_mapping(cls,record):
        require(type(record) is dict and set(record)=={f.name for f in fields(cls)},'freeze_schema')
        for name,value in record.items():
            if name=='source_hashes':
                require(type(value) is dict and value==PINS,'source_freeze_schema')
            else:
                require(valid_digest(value),'freeze_digest_schema:'+name)
        return cls(**record)

    def validate(self,runtime,command):
        self.from_mapping(asdict(self))
        require(source_pins()==PINS,'source_pin_readback')
        runtime.attest()
        require(self.harness_sha256==sha(Path(__file__).read_bytes()),'harness_freeze')
        require(self.guard_tests_sha256==sha((ROOT/'tests/test_multiview_track_cycle_reference.py').read_bytes()),
                'guard_tests_freeze')
        require(self.source_hashes==PINS,'source_freeze')
        require(self.environment_sha256==sha(canonical(runtime.identity)),'environment_freeze')
        require(self.specification_sha256==sha(canonical(PACKET_SPEC)),'spec_freeze')
        require(self.command_sha256==sha(canonical(command)),'command_freeze')
        require(type(self.external_go_sha256) is str and len(self.external_go_sha256)==64 and
                all(x in '0123456789abcdef' for x in self.external_go_sha256),'external_go_receipt')


def valid_digest(value):
    return type(value) is str and len(value)==64 and all(x in '0123456789abcdef' for x in value)


def read_freeze(stream):
    """Strict JSON from bounded stdin or an already supplied file; no writes."""
    raw=stream.read(FREEZE_LIMIT+1)
    require(type(raw) is bytes and len(raw)<=FREEZE_LIMIT,'freeze_record_cap')
    def pairs(items):
        result={}
        for name,value in items:
            require(name not in result,'freeze_duplicate_key')
            result[name]=value
        return result
    def nonfinite(value):
        raise IntegrityError('freeze_nonfinite')
    def finite_float(text):
        value=float(text)
        require(math.isfinite(value),'freeze_nonfinite')
        return value
    try:
        record=json.loads(raw.decode('utf-8'),object_pairs_hook=pairs,
                          parse_constant=nonfinite,parse_float=finite_float)
    except (ValueError,UnicodeError,RecursionError) as error:
        raise IntegrityError('freeze_malformed') from error
    return RunFreeze.from_mapping(record)


def prepare_attempt(freeze,runtime,command):
    """Cheap pins/reservations only; never consumes/builds/scores an attempt."""
    require(not _CONSUMED,'attempt_consumed')
    require(type(freeze) is RunFreeze,'freeze_type')
    freeze.validate(runtime,command)
    require(wire_size({'freeze':asdict(freeze),'environment':runtime.identity})<=
            ENVELOPE_OUTPUT_RESERVATION//2,'envelope_identity_bound')
    meter=Meter()
    meter.preallocate_output(MAIN_OUTPUT_RESERVATION)
    meter.claim_output(ENVELOPE_OUTPUT_RESERVATION)
    return meter


def consume_and_build(runtime,meter,builder,validator):
    """The irrevocable process boundary; tests supply nonnumerical stubs only."""
    global _CONSUMED
    require(not _CONSUMED,'attempt_consumed')
    require(meter.output_reserved==MAIN_OUTPUT_RESERVATION and
            meter.output_claimed==ENVELOPE_OUTPUT_RESERVATION,'attempt_reservation')
    meter.check()
    _CONSUMED=True
    packet=builder(runtime)
    observed=validator(runtime,packet)
    require(valid_digest(observed),'observed_digest_format')
    return packet,observed


def with_observed_digest(record,observed):
    require(valid_digest(observed) and 'observed_packet_sha256' not in record,'observed_receipt')
    return {'observed_packet_sha256':observed,**record}


def execute_once(freeze: RunFreeze,command: tuple):
    """Recipe-frozen single build then A/B; observed digest is a run receipt."""
    require(not _CONSUMED,'attempt_consumed')
    runtime=load_runtime()
    meter=prepare_attempt(freeze,runtime,command)
    packet,observed=consume_and_build(runtime,meter,build_fixed_packet,validate_packet)
    outputs=[]
    for arm in ('A','B'):
        driver=Driver(runtime,arm,packet.graphs[0],packet.frames[0].descriptors,packet.frames[1].descriptors,
                      packet.closure.eligible_indices,meter)
        driver.run(packet.frames[0],packet.frames[1],packet.cameras[0],packet.cameras[1])
        outputs.append(with_observed_digest({'arm':arm,'classes':driver.classes,'final':driver.final},observed))
    a,b=outputs
    require(all(x['prefix']==y['prefix'] and x['rows']==y['rows'] and x['norms']==y['norms']
                and x['accepted']==y['accepted'] and x['attempts']==y['attempts'] and
                x['RNG_state']==y['RNG_state'] and x['generation_calls']==y['generation_calls']
                for x,y in zip(a['classes'],b['classes'])),'A_B_prefix_rng_parity')
    for x,y in zip(a['classes'],b['classes']):
        y['displaced_original_orders']=tuple(i for i in x['kept_orders'] if i not in y['kept_orders'])
    winner=b['final']['winner']
    success=(a['final']['result']['status']=='refused' and b['final']['result']['status']=='solved'
             and winner is not None and winner['seed_origin'] and winner['eligible_selected']>=8)
    result=with_observed_digest({'status':'MECHANISM_PASS' if success else 'NO_GO',
            'scope':'SYNTHETIC_INJECTED_GRAPH_TRANSIENT_PLUS_COMMS',
            'freeze':asdict(freeze),'environment':runtime.identity,
            'arms':outputs,'residual_rows':meter.residual_rows,'SVD_attempts':meter.svd_calls,
            'fit_roles':meter.roles,'typed_evidence_bytes':meter.evidence_bytes,
            'output_reserved_bytes':meter.output_reserved,
            'seconds':meter.clock()-meter.started,'RSS':'UNMEASURED','durable_replay':False},observed)
    require(meter.transient_bytes==0,'scratch_lifetime')
    require(meter.output_claimed==meter.output_reserved==MAIN_OUTPUT_RESERVATION and
            wire_size(result)+1<=meter.output_reserved,'main_output_reservation')
    runtime.attest()
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-fixed',action='store_true')
    parser.add_argument('--freeze-json',help='- reads bounded strict JSON from stdin; otherwise existing file')
    args=parser.parse_args()
    require(args.execute_fixed and args.freeze_json is not None,'separate_main_run_GO_required')
    if args.freeze_json=='-':
        freeze=read_freeze(sys.stdin.buffer)
    else:
        with Path(args.freeze_json).open('rb') as stream: freeze=read_freeze(stream)
    # No extra artifact writes; final detailed report is stdout/transient only.
    report=execute_once(freeze,(sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]))
    emit_report(report,sys.stdout.buffer)
