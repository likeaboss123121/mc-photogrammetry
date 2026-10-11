"""Isolated camera/G diagnostic. An explicit separate image-run GO is required.

Production receives only the two literal ordered PNG paths and a fresh target.
Native receipts are sealed before optional evaluation metadata is read. No export,
visible geometry/state score, historical report pins, or automatic retry.
"""
from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from fractions import Fraction
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import marshal
import math
import os
from pathlib import Path
import platform
import sys
import time
from types import CodeType

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT/'tests/test_sealed_image_only_baseline.py'
BASE = 'exports/datasets/fresh-c4-canal-town-framed-2026-09-12'
SCENE = BASE+'/scenes/canal-town-architectural-component-bbox'
INPUTS = (SCENE+'/captures/view-canal-town-architectural-component-bbox-05.png',
          SCENE+'/captures/view-canal-town-architectural-component-bbox-06.png')
INPUT_HASHES = ('6f844d5e3600e853bf3045fc25b55a271c0b665226053602b6d30c945f0f016b',
                '53c38526d378a0597ab6e00fb5e01a6c7aa0e2dc11619c1b7eaa3cfc0ccf8923')
METADATA = {BASE+'/index.json':'0637caba1a523bbdf7351dedd7c4f93bfdb611f0b549fc54b3a781f3d25ee46b',
            SCENE+'/capture.json':'c6c44d43408779006854c81742a68d00cfe5b546f506ffa08ea59ba7b812abe6'}
MAX_RECEIPT = 16*1024*1024
MAX_OBSERVER = MAX_RECEIPT-1024*1024
MAX_ARRAY = 64*1024*1024
MANAGED_RESERVATION = 128*1024*1024
MAX_SOURCE_BYTES = 4*1024*1024
MAX_FILE_BYTES = 64*1024*1024
MAX_BINARY_BYTES = 256*1024*1024
MAX_METADATA = 8*1024*1024
MAX_ITEMS = 100_000
MAX_STRING = 4096
MAX_DEPTH = 24
MAX_COORDINATE_ELEMENTS = 32_768
MAX_SECONDS = 120.
MAX_VALIDATION_WORK = 2_000_000
BLOCK = 65_536
NUMERIC_RAW_CHUNK = 3072
NUMERIC_TEXT_CHUNK = 4096
NUMERIC_VALUES_FIELD = 'base64_c_order_4096_chunks_v1'
NUMERIC_VALUE_KEYS = frozenset(('points','matches','selection','edges','distances','logical_pairs'))
ENCODED_PAYLOAD_FIELDS = {
    'packed_mask_base64':{},
    'inline_samples_float64_le_base64':{'scalar_type':'float64','byte_order':'little','dtype':'<f8'},
    'profile_samples_float64_le_base64':{'scalar_type':'float64','byte_order':'little','dtype':'<f8'},
    'profile_weights_float64_le_base64':{'scalar_type':'float64','byte_order':'little','dtype':'<f8'},
    'axis_bin_indices_int64_le_base64':{'scalar_type':'int64','byte_order':'little','dtype':'<i8'},
}
CAPS = dict(receipt=MAX_RECEIPT,observer=MAX_OBSERVER,array=MAX_ARRAY,
            managed_reservation=MANAGED_RESERVATION,items=MAX_ITEMS,depth=MAX_DEPTH,
            seconds=MAX_SECONDS,file_bytes=MAX_FILE_BYTES)
STAGES = {'discovery':1,'decode':2,'auto_camera':1,'session':1,'pair':1,
          'features':2,'reciprocal_graph':1,'modes':1,'connections':1,'pose_graph':1,
          'lattice':1,'geometry':1,'G_matcher':1,'G_prefilter':1,'G_solver':1,'output':1}
PROJECTION_TYPES = frozenset(('ImageDerivedFrame','ImageDerivedSession','ImageDerivedEntryResult',
    'ImageDerivedLatticeResult','PlaneReading','PlaneSignature','PlaneRegionEvidence',
    'PeriodicEnergyPhaseEvidence','SplitPhasePatchRef','SplitSeamProfileEvidence',
    'GeometryProvenanceBridge','SplitSourceEvidence','CalibrationResult','CameraEstimate','CameraModel',
    'FrameFeatures','ReciprocalGraph','JointPairResult','ImageDerivedConnectionGraph',
    'ImageDerivedPoseGraph','ImageDerivedGraphResult','ImageDerivedGeometryResult','ImageDerivedProductionResult',
    'ImageDerivedOutputResult','PlaneMatch','ComponentSolution'))
ALLOCATION_PLAN = {'upstream':{'items':60_000,'bytes':8*1024*1024},
                   'G':{'items':35_000,'bytes':6*1024*1024},
                   'terminal_failure':{'items':5000,'bytes':1024*1024}}
PROJECTION_MODULES = {
    **dict.fromkeys(('ImageDerivedFrame','ImageDerivedSession'),'pipeline.image_derived_session'),
    'ImageDerivedEntryResult':'pipeline.image_derived_entry',
    'ImageDerivedLatticeResult':('pipeline.image_derived_lattice','pipeline.image_derived_lattice_measure'),
    **dict.fromkeys(('PlaneReading','PlaneSignature','PlaneRegionEvidence','PeriodicEnergyPhaseEvidence',
                   'SplitPhasePatchRef','SplitSeamProfileEvidence'),'reconstruction.lattice_planes'),
    **dict.fromkeys(('GeometryProvenanceBridge','SplitSourceEvidence'),'reconstruction.plane_geometry_provenance'),
    'CalibrationResult':'reconstruction.calibration','CameraEstimate':'pipeline.contracts',
    'CameraModel':'reconstruction.camera',
    **dict.fromkeys(('FrameFeatures','ReciprocalGraph'),'pipeline.correspondence'),
    'JointPairResult':'pipeline.joint_matching','ImageDerivedConnectionGraph':'pipeline.image_derived_connections',
    'ImageDerivedPoseGraph':'pipeline.image_derived_pose_graph',
    'ImageDerivedGraphResult':'pipeline.image_derived_joint_graph',
    'ImageDerivedGeometryResult':'pipeline.image_derived_geometry_orchestration',
    'ImageDerivedProductionResult':'pipeline.image_derived_production',
    'ImageDerivedOutputResult':'pipeline.image_derived_output',
    **dict.fromkeys(('PlaneMatch','ComponentSolution'),'reconstruction.plane_graph')}
G_SOURCE_SHA256 = 'c47aa2a7530341b8ab766a48324d51caf263b247bbaed4e6cd297377430ae61a'
G_RECORD_SCHEMAS = {
    'mc_photogrammetry.reconstruction.plane_graph.PlaneMatch': (
        ('left','right','left_axis','right_axis','left_b','right_b','quadrant_of_right','psr','peak',
         'second_peak','overlap_cells','integer_residual','score','left_plane','right_plane','cell_shift_rc'),
        ('str','str','int','int','float','float','int','float','float','float','int','float | None','float',
         'int','int','tuple[int, int] | None')),
    'mc_photogrammetry.reconstruction.plane_graph.PlaneMatchRefusal': (
        ('left','right','left_axis','right_axis','quadrant_of_right','reason','detail'),
        ('str','str','int','int','int','str','str'))}

# Exact source declarations, not a name-only or native result-digest contract.
GEOMETRY_SOURCE_PINS = {
    'pipeline/image_derived_geometry_orchestration.py':'cbd2d8e2e8e744799628159b4eb307207024435272441d4b90215bb6f10add1e',
    'pipeline/image_derived_signatures.py':'7e287aecb6c1e6bb111c5d5a6c33abc876b6083ce48abdf3c4fff1306819e43f',
    'pipeline/image_derived_joint_graph.py':'06045b62c69945d33928a3e922219786da28de885a5722d0666ef83467c585bc',
    'pipeline/image_derived_endpoint_selection.py':'0a5494c95bce8973d8a4c39fb2f9d9d877fde5d11e11d397190c4c32ca7dc152'}
_GEOMETRY_RECORD_SCHEMA_ROWS = {
    'mc_photogrammetry.pipeline.image_derived_geometry_orchestration.ImageDerivedGeometryResult':
        (('schema','str'),('producer','str'),('terminal','str'),('status','str'),('reason','str'),('detail','str'),('session_sha256','str'),('lattice_sha256','str'),('stage_digests','tuple[tuple[str, str], ...]'),('records','tuple[tuple[str, object], ...]'),('placement_records','tuple[tuple[str, object], ...]'),('universe','placement.PlacementUniverse | None'),('supports','tuple'),('result_sha256','str')),
    'mc_photogrammetry.pipeline.image_derived_signatures.ImageDerivedSignaturesResult':
        (('schema','str'),('producer','str'),('terminal','str'),('status','str'),('reason','str'),('detail','str'),('session_sha256','str'),('lattice_sha256','str'),('frame_ids','tuple[str, ...]'),('usable_masks','tuple[UsableMask, ...]'),('signatures','tuple[SignatureEvidence, ...]'),('abstentions','tuple[SignatureAbstention, ...]'),('masks_sha256','str'),('signatures_sha256','str'),('result_sha256','str')),
    'mc_photogrammetry.pipeline.image_derived_signatures.SignatureEvidence':
        (('frame_id','str'),('canonical_index','int'),('reading_axis','int'),('alternative_index','int'),('blocks_per_period','float'),('signature','lattice_planes.PlaneSignature'),('signature_sha256','str')),
    'mc_photogrammetry.pipeline.image_derived_signatures.UsableMask':
        (('frame_id','str'),('height','int'),('width','int'),('usable_bool_bytes','bytes'),('mask_sha256','str')),
    'mc_photogrammetry.pipeline.image_derived_signatures.SignatureAbstention':
        (('frame_id','str'),('reason','str'),('detail','str'),('canonical_index','int | None'),('alternative_index','int | None')),
    'mc_photogrammetry.pipeline.image_derived_joint_graph.ImageDerivedGraphResult':
        (('schema','str'),('producer','str'),('terminal','str'),('status','str'),('reason','str'),('detail','str'),('session_sha256','str'),('lattice_sha256','str'),('signatures_sha256','str'),('connection_sha256','str'),('frame_ids','tuple'),('readings','tuple'),('signatures','tuple'),('candidate_matches','tuple'),('selection','selection.EndpointSelectionCertificate'),('matches','tuple'),('component_solution','plane_graph.ComponentSolution | None'),('correlation_algorithm','str'),('max_shift_cells','int'),('geometry_settings','tuple'),('geometry_evaluations','int'),('result_sha256','str')),
    'mc_photogrammetry.pipeline.image_derived_endpoint_selection.EndpointSelectionCertificate':
        (('schema','str'),('producer','str'),('status','str'),('reason','str'),('session_sha256','str'),('lattice_sha256','str'),('signatures_sha256','str'),('connection_sha256','str'),('frame_ids','tuple[str, ...]'),('quadrant_constraints','tuple[tuple[str, str, int], ...]'),('budget','csp.EndpointBudget'),('reading_bindings','tuple'),('candidate_count','int'),('candidate_matches_sha256','str'),('proof_json','str'),('representatives','tuple[RepresentativeConstraint, ...]'),('result_sha256','str'))}
# Retain parallel immutable vectors instead of one tuple object per field.
# This keeps the frozen registry within its existing physical item reservation
# for the current joint-graph result with its endpoint-selection certificate.
GEOMETRY_RECORD_SCHEMAS = {
    schema: (tuple(name for name, _ in rows),
             tuple(annotation for _, annotation in rows))
    for schema, rows in _GEOMETRY_RECORD_SCHEMA_ROWS.items()}
del _GEOMETRY_RECORD_SCHEMA_ROWS
GEOMETRY_JSON_BYTES = 16384
GEOMETRY_SCRATCH_NODES = 256
GEOMETRY_RECORD_NAMES = frozenset(schema.rsplit('.',1)[1] for schema in GEOMETRY_RECORD_SCHEMAS)
GEOMETRY_CLASS_LIMIT = 64


class IntegrityError(RuntimeError): pass
class ObservationError(IntegrityError): pass


def require(value,message):
    if not value: raise IntegrityError(message)


def sha(value): return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()


def digest(value): return sha(canonical(value))


def is_digest(value):
    return type(value) is str and len(value)==64 and all(x in '0123456789abcdef' for x in value)


def encoded_size(value):
    if isinstance(value,dict):
        require(all(type(k) is str for k in value),'JSON key type')
        return 2+max(0,len(value)-1)+sum(len(canonical(k))+1+encoded_size(v) for k,v in value.items())
    if isinstance(value,(list,tuple)):
        return 2+max(0,len(value)-1)+sum(map(encoded_size,value))
    require(value is None or type(value) in (str,int,float,bool),'JSON scalar type')
    return len(canonical(value))


def tokens(value):
    if isinstance(value,dict):
        yield b'{'
        for i,key in enumerate(sorted(value)):
            if i: yield b','
            yield canonical(key)+b':'
            yield from tokens(value[key])
        yield b'}'
    elif isinstance(value,(list,tuple)):
        yield b'['
        for i,item in enumerate(value):
            if i: yield b','
            yield from tokens(item)
        yield b']'
    else: yield canonical(value)


def strict_json(raw):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,'duplicate JSON key')
            result[key]=value
        return result
    def bad(value): raise IntegrityError('nonfinite JSON')
    def number(value):
        result=float(value); require(math.isfinite(result),'nonfinite JSON'); return result
    try:
        return json.loads(raw.decode('utf-8'),object_pairs_hook=pairs,parse_constant=bad,parse_float=number)
    except (ValueError,UnicodeError,RecursionError) as error:
        raise IntegrityError('malformed JSON') from error


def file_hash(path,limit):
    require(path.is_file() and not path.is_symlink(),'file type')
    require(path.stat().st_size<=limit,'file byte cap')
    result=hashlib.sha256(); count=0
    with path.open('rb') as stream:
        while data:=stream.read(BLOCK):
            count+=len(data); require(count<=limit,'file grew beyond cap'); result.update(data)
    return {'sha256':result.hexdigest(),'bytes':count}


def source_manifest(root=ROOT):
    paths=sorted((root/'src/mc_photogrammetry').rglob('*.py'))
    require(len(paths)==183,'source manifest count')
    return tuple({'path':p.relative_to(root).as_posix(),**file_hash(p,MAX_SOURCE_BYTES)} for p in paths)


@dataclass(frozen=True)
class Freeze:
    source_manifest_sha256: str
    evaluator_sha256: str
    tests_sha256: str
    runtime_sha256: str
    command_sha256: str
    external_go_sha256: str

    @classmethod
    def read(cls,stream):
        raw=stream.read(BLOCK+1)
        require(type(raw) is bytes and len(raw)<=BLOCK,'freeze byte cap')
        value=strict_json(raw)
        require(type(value) is dict and set(value)=={f.name for f in fields(cls)},'freeze schema')
        require(all(is_digest(x) for x in value.values()),'freeze digests')
        return cls(**value)

    def validate(self,manifest,provenance,command):
        require(all(is_digest(getattr(self,f.name)) for f in fields(self)),'freeze digest type')
        require(self.source_manifest_sha256==digest(manifest),'source manifest freeze')
        require(self.evaluator_sha256==file_hash(Path(__file__),MAX_SOURCE_BYTES)['sha256'],'evaluator freeze')
        require(self.tests_sha256==file_hash(TEST,MAX_SOURCE_BYTES)['sha256'],'tests freeze')
        require(self.runtime_sha256==digest(provenance),'runtime freeze')
        require(self.command_sha256==digest(command),'command freeze')


@dataclass
class Budget:
    reserved: int = 0
    items: int = 0
    started: float | None = None
    clock: object = time.monotonic

    def __post_init__(self):
        if self.started is None: self.started=self.clock()
        self.stage='unscoped'; self.partitioned=False; self.stage_counts={}; self.pool_counts={}

    @contextmanager
    def scope(self,stage):
        require(stage in STAGES or stage in ('unscoped','terminal','failure'),'budget stage')
        prior=self.stage; self.stage=stage
        try: yield
        finally: self.stage=prior

    def pool(self):
        # Geometry/output snapshots are the terminal part of the reconstruction
        # path and share the existing G reservation.  Keep the original stage
        # name: geometry serialization has stage-specific schema handling.
        if self.stage.startswith('G_') or self.stage in ('geometry','output','terminal'): return 'G'
        return 'terminal_failure' if self.stage=='failure' else 'upstream'

    def charge(self,amount,items=1):
        require(self.clock()-self.started<=MAX_SECONDS,'observer watchdog')
        require(type(amount) is int and amount>=0 and self.reserved+amount<=MAX_OBSERVER,'evidence precharge')
        require(type(items) is int and items>=0,'item amount')
        require(self.items+items<=MAX_ITEMS,'item precharge')
        if amount==0 and items==0: return  # watchdog-only check allocates no ledger row
        if self.partitioned:
            pool=self.pool(); used=self.pool_counts.get(pool,{'bytes':0,'items':0})
            cap=ALLOCATION_PLAN[pool]
            require(used['bytes']+amount<=cap['bytes'],'stage byte reservation')
            require(used['items']+items<=cap['items'],'stage item reservation')
            self.pool_counts[pool]={'bytes':used['bytes']+amount,'items':used['items']+items}
        stage=self.stage_counts.setdefault(self.stage,{'bytes':0,'items':0})
        stage['bytes']+=amount; stage['items']+=items
        self.reserved+=amount; self.items+=items


class Snapshot:
    """Bounded typed data: array hashes; selected coordinate arrays also values."""
    def __init__(self,budget): self.budget=budget

    def packed_mask_receipt(self,value):
        return self.encoded_payload_receipt(value,'packed_mask_base64')

    def encoded_payload_receipt(self,value,key):
        require(key in ENCODED_PAYLOAD_FIELDS,'encoded payload field')
        label='encoded mask' if key=='packed_mask_base64' else 'encoded lattice'
        require(len(value)<=MAX_ARRAY,label+' cap')
        self.budget.charge(1024)  # bounded receipt before any slicing/encoding/hash
        require(2*BLOCK+self.budget.reserved+MAX_RECEIPT<=MANAGED_RESERVATION,label+' scratch cap')
        fingerprint=hashlib.sha256()
        for offset in range(0,len(value),BLOCK):
            self.budget.charge(0,0)
            piece=value[offset:offset+BLOCK]
            require(piece.isascii() and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in piece),
                    label+' alphabet')
            fingerprint.update(piece.encode('ascii'))
        return {'type':'builtins.str','field':key,'encoding':'base64',**ENCODED_PAYLOAD_FIELDS[key],
                'text_encoding':'ascii','encoded_bytes':len(value),'characters':len(value),
                'sha256':fingerprint.hexdigest(),'payload_policy':'digest_only; no decoding'}

    def take(self,value,key='',depth=0,seen=None):
        require(depth<=MAX_DEPTH,'snapshot depth')
        self.budget.charge(32)
        if value is None or type(value) in (bool,int,float):
            if type(value) is float and not math.isfinite(value):
                return {'nonfinite':repr(value)}
            require(type(value) is not int or value.bit_length()<=128,'integer cap')
            return value
        if isinstance(value,(str,Path)):
            if type(value) is str and key in ENCODED_PAYLOAD_FIELDS:
                return self.encoded_payload_receipt(value,key)
            text=str(value); require(len(text)<=MAX_STRING,'string cap')
            self.budget.charge(12*len(text)); return text
        if hasattr(value,'dtype') and hasattr(value,'shape') and hasattr(value,'nbytes'):
            return self.array_snapshot(value,key)
        if type(value) is bytes:
            require(len(value)<=MAX_ARRAY,'buffer cap'); self.budget.charge(256)
            return {'bytes':len(value),'sha256':sha(value)}
        if inspect.isfunction(value):
            self.budget.charge(1024)
            return {'callable':value.__module__+'.'+value.__qualname__,
                    'code_encoding':'marshal-v2','code_sha256':sha(marshal.dumps(value.__code__,2))}
        if type(value) is Fraction:
            require(value.numerator.bit_length()<=128 and value.denominator.bit_length()<=128,'rational cap')
            self.budget.charge(256)
            return {'type':'fractions.Fraction','numerator':value.numerator,'denominator':value.denominator}
        seen=set() if seen is None else seen
        if id(value) in seen: return {'shared_or_cycle':type(value).__module__+'.'+type(value).__qualname__}
        seen.add(id(value))
        if is_dataclass(value):
            declared=fields(value); self.budget.charge(len(declared)*256,len(declared))
            projected={f.name:self.take(getattr(value,f.name),f.name,depth+1,seen) for f in declared}
            result={'type':type(value).__module__+'.'+type(value).__qualname__,
                    'fields':self.record_fields(projected)}
        elif isinstance(value,Mapping):
            self.budget.charge(len(value)*256,len(value))
            result={}
            for name,item in value.items():
                if type(name) is tuple: name=repr(name)
                require(type(name) is str and len(name)<=MAX_STRING,'mapping key cap')
                self.budget.charge(12*len(name))
                result[name]=self.take(item,name,depth+1,seen)
        elif isinstance(value,(list,tuple)):
            self.budget.charge(len(value)*64,len(value))
            result=[self.take(item,key,depth+1,seen) for item in value]
        elif hasattr(value,'item') and type(value).__module__.startswith('numpy'):
            result=self.take(value.item(),key,depth+1,seen)
        else:
            self.budget.charge(1024)
            result={'opaque_type':type(value).__module__+'.'+type(value).__qualname__}
        seen.remove(id(value))
        return result

    def numeric_work(self,kind):
        if hasattr(self,'work'): self.work(kind)
        else:
            visits=getattr(self,'numeric_visits',0)
            require(visits<MAX_VALIDATION_WORK,'semantic validation work cap')
            self.numeric_visits=visits+1; self.budget.charge(0,0)

    def array_contract(self,value):
        require(type(value.nbytes) is int and 0<=value.nbytes<=MAX_ARRAY,'array cap')
        dtype=value.dtype; tag=dtype.str
        require(not getattr(dtype,'hasobject',False),'object array unsupported')
        require(getattr(dtype,'fields',None) is None and getattr(dtype,'subdtype',None) is None,'compound array dtype unsupported')
        require(type(tag) is str and 3<=len(tag)<=4 and tag[0] in '<>=|' and tag[2:].isdigit(),
                'numeric array dtype unsupported')
        widths={'b':(1,),'i':(1,2,4,8),'u':(1,2,4,8),'f':(2,4,8),'c':(8,16)}
        width=int(tag[2:]); require(tag[2:]==str(width) and width in widths.get(tag[1],()),'numeric array dtype unsupported')
        require(tag[0]=='|' if width==1 else tag[0] in '<>','numeric array endian unsupported')
        require(len(value.shape)<=4 and len(value.strides)==len(value.shape),'array rank')
        require(all(type(n) is int and n>=0 for n in value.shape),'array shape')
        require(type(value.size) is int and all(type(n) is int for n in value.strides),'array metadata type')
        require(math.prod(value.shape)==value.size and value.nbytes==value.size*width,'numeric array byte count')
        return tag

    def ensure_numeric_scratch(self):
        if not getattr(self,'numeric_scratch',False):
            self.budget.charge(16*1024,16); self.numeric_scratch=True

    def array_snapshot(self,value,key=''):
        tag=self.array_contract(value)
        self.budget.charge(1024)
        view=memoryview(value); require(view.nbytes==value.nbytes,'numeric array buffer count')
        if view.c_contiguous:
            raw=view.cast('B') if value.nbytes else memoryview(b'')
        else:
            require(value.nbytes+self.budget.reserved+MAX_RECEIPT<=MANAGED_RESERVATION,'array scratch cap')
            self.budget.charge(value.nbytes+256,1)  # before logical C-order copy
            raw=memoryview(view.tobytes(order='C'))
        fingerprint=hashlib.sha256()
        for offset in range(0,len(raw),BLOCK):
            self.numeric_work('numeric_hash'); fingerprint.update(raw[offset:offset+BLOCK])
        result={'dtype':tag,'shape':list(value.shape),'strides':list(value.strides),
                'writeable':bool(value.flags.writeable),'sha256':fingerprint.hexdigest(),'bytes':value.nbytes}
        if (key in NUMERIC_VALUE_KEYS or value.size<=64) and value.size<=MAX_COORDINATE_ELEMENTS:
            count=(value.nbytes+NUMERIC_RAW_CHUNK-1)//NUMERIC_RAW_CHUNK
            encoded=4*((value.nbytes+2)//3)
            # Actual ASCII payload, list/string/JSON overhead and bounded
            # temporary encode/decode buffers; no scalar Python list exists.
            self.ensure_numeric_scratch()
            self.budget.charge(encoded+128*count+1024,count+8)
            chunks=[]; bound=hashlib.sha256()
            for offset in range(0,len(raw),NUMERIC_RAW_CHUNK):
                self.numeric_work('numeric_encode')  # before slice/encoder allocation
                piece=raw[offset:offset+NUMERIC_RAW_CHUNK]
                bound.update(piece); chunks.append(base64.b64encode(piece).decode('ascii'))
            require(bound.hexdigest()==result['sha256'],'array changed during projection')
            # Field name fixes version/order/chunk bound once in the schema;
            # the payload is the ordered string list, with no repeated per-array
            # codec metadata dictionary or duplicate scalar representation.
            result[NUMERIC_VALUES_FIELD]=chunks
        return result

    def validate_numeric(self,full):
        self.ensure_numeric_scratch()
        chunks=full[NUMERIC_VALUES_FIELD]; count=(full['bytes']+NUMERIC_RAW_CHUNK-1)//NUMERIC_RAW_CHUNK
        require(type(chunks) is list and len(chunks)==count,'numeric chunk count')
        fingerprint=hashlib.sha256(); consumed=0
        for index,text in enumerate(chunks):
            self.numeric_work('numeric_validate')  # before ASCII/decode/re-encode allocation
            size=min(NUMERIC_RAW_CHUNK,full['bytes']-consumed)
            require(type(text) is str and len(text)==4*((size+2)//3) and len(text)<=NUMERIC_TEXT_CHUNK and
                    text.isascii(),'numeric chunk length/type')
            try: decoded=base64.b64decode(text,validate=True)
            except ValueError as error: raise IntegrityError('numeric chunk alphabet/padding') from error
            require(len(decoded)==size and base64.b64encode(decoded).decode('ascii')==text,'numeric chunk canonical')
            fingerprint.update(decoded); consumed+=len(decoded)
        require(consumed==full['bytes'] and fingerprint.hexdigest()==full['sha256'],'numeric payload raw binding')

    def record_fields(self,projected): return projected


class StageProjection(Snapshot):
    """Explicit source-record/array definitions, recomputed content references.

    All declared fields are projected, preserving native digests as fields.
    Merkle definitions use freshly recomputed child content, never object ids or
    a trusted native digest alone. Lists/maps always have fresh ordered versions.
    """
    def __init__(self,budget):
        super().__init__(budget); self.definitions={}; self.projecting=set()
        self.validation_visits=0; self.scratch_reserved=False
        self.validation_walk_counts={}
        self.g_registry=None; self.g_registry_sha256=None; self.g_containers=False
        self.g_json_scratch=False
        self.geometry_registry=None; self.geometry_registry_sha256=None
        self.geometry_classes=None  # allocated only after registry precharge
        self.geometry_live_nodes=0

    def ensure_geometry_registry(self):
        self.ensure_g_json_scratch()
        if self.geometry_registry is None:
            # Includes the full borrowed immutable schema graph and new tables.
            self.budget.charge(16384,256)
            for path,pin in GEOMETRY_SOURCE_PINS.items():
                require(file_hash(ROOT/'src/mc_photogrammetry'/path,MAX_SOURCE_BYTES)['sha256']==pin,
                        'geometry schema source pin')
            self.geometry_registry=dict(GEOMETRY_RECORD_SCHEMAS)
            self.geometry_registry_sha256=digest(self.geometry_registry)
            # Borrow native class objects/schema strings; retain bounded dict
            # slots only. Full registry allowance includes these live bindings.
            self.geometry_classes={}
        for schema,declared in GEOMETRY_RECORD_SCHEMAS.items():
            self.work('geometry_registry')
            for _ in declared[0]: self.work('geometry_registry')
        require(self.geometry_registry_sha256==digest(GEOMETRY_RECORD_SCHEMAS)==digest(self.geometry_registry),
                'geometry schema registry changed')

    def geometry_schema(self,value):
        return self.geometry_type_schema(type(value),bind=True)

    def geometry_type_schema(self,kind,bind=False):
        qualified=kind.__module__+'.'+kind.__qualname__
        require(qualified in self.geometry_registry,'geometry schema namespace')
        for schema in self.geometry_registry:
            if schema==qualified: qualified=schema; break
        declared=kind.__dataclass_fields__
        names,annotations=self.geometry_registry[qualified]
        require(len(declared)==len(names)==len(annotations),'geometry schema fields')
        for (name,field),(wanted,annotation) in zip(declared.items(),zip(names,annotations)):
            self.work('geometry_schema')
            require(type(name) is str and type(field.type) is str and name==wanted and field.type==annotation,
                    'geometry schema fields/annotations')
        if kind in self.geometry_classes:
            require(self.geometry_classes[kind]==qualified,'geometry live class changed')
        elif bind:
            require(len(self.geometry_classes)<GEOMETRY_CLASS_LIMIT,'geometry live class receipt cap')
            self.work('geometry_class_binding')
            self.geometry_classes[kind]=qualified
        return qualified

    def validate_geometry_classes(self):
        # Recheck the whole bounded receipt table after source traversal: a late
        # descendant can drift after its own earlier check without changing the
        # value fingerprint. Never infer metadata validity from that digest.
        for kind,qualified in self.geometry_classes.items():
            self.work('geometry_live_class_validation')
            require(self.geometry_type_schema(kind)==qualified,'geometry live class changed')

    def geometry_upgrade(self,value,depth=0):
        self.work('geometry_context_upgrade')
        require(depth<=MAX_DEPTH,'geometry context depth')
        if is_dataclass(value):
            if type(value).__qualname__ in GEOMETRY_RECORD_NAMES:
                self.geometry_schema(value)  # metadata is NOT in native_digest
            for field in fields(value):
                child=getattr(value,field.name)
                if field.name in NUMERIC_VALUE_KEYS:
                    self.upgrade_numeric_descendants(child,field.name,depth+1)
                if is_dataclass(child) or isinstance(child,(list,tuple,Mapping)):
                    self.geometry_upgrade(child,depth+1)
        elif isinstance(value,Mapping):
            for name,child in value.items():
                if name in NUMERIC_VALUE_KEYS: self.upgrade_numeric_descendants(child,name,depth+1)
                if is_dataclass(child) or isinstance(child,(list,tuple,Mapping)):
                    self.geometry_upgrade(child,depth+1)
        elif isinstance(value,(list,tuple)):
            for child in value:
                if is_dataclass(child) or isinstance(child,(list,tuple,Mapping)):
                    self.geometry_upgrade(child,depth+1)

    def geometry_tokens(self,value,key='',depth=0):
        # No intermediate projected row/list/ref graph: emit directly into one
        # precharged fixed buffer. Existing take() reference allocations retain
        # their ordinary charges. New wrapper references are wire tokens only.
        self.work('geometry_encode'); require(depth<=MAX_DEPTH,'geometry encode depth')
        qualified=type(value).__module__+'.'+type(value).__qualname__
        if is_dataclass(value) and qualified in self.geometry_registry:
            lookup=self.geometry_record(value,key,depth,reference=False)
            yield b'{"semantic_reference":'+canonical(lookup)+b'}'; return
        if isinstance(value,(list,tuple,Mapping)):
            if self.g_containers:
                native=self.native_digest(value,key)
                kind='g_mapping' if isinstance(value,Mapping) else 'g_sequence'
                if kind+':'+native in self.definitions:
                    yield from self.geometry_projected_tokens(self.take(value,key,depth),depth); return
            if isinstance(value,Mapping):
                yield b'{"native_mapping_entries":['
                for index,(name,child) in enumerate(value.items()):
                    require(type(name) in (str,int,tuple),'semantic mapping key type')
                    if index: yield b','
                    yield b'{"key":'
                    yield from self.geometry_tokens(name,'',depth+1)
                    yield b',"key_type":'+canonical(type(name).__name__)+b',"value":'
                    yield from self.geometry_tokens(child,name if type(name) is str else '',depth+1)
                    yield b'}'
                yield b']}'; return
            yield b'['
            for index,child in enumerate(value):
                if index: yield b','
                yield from self.geometry_tokens(child,key,depth+1)
            yield b']'; return
        if value is None or type(value) in (bool,int,float,str):
            require(type(value) is not int or value.bit_length()<=128,'integer cap')
            require(type(value) is not str or len(value)<=MAX_STRING,'string cap')
            if type(value) is float and not math.isfinite(value):
                yield b'{"nonfinite":'+canonical(repr(value))+b'}'
            else:
                require(encoded_size(value)<=GEOMETRY_JSON_BYTES,'geometry scalar bytes')
                yield canonical(value)
            return
        yield from self.geometry_projected_tokens(self.take(value,key,depth),depth)

    def geometry_projected_tokens(self,node,depth=0):
        # Stream already charged fallback projections too: canonical() on an
        # entire foreign record could otherwise allocate before the wire gate.
        self.work('geometry_projected_encode'); require(depth<=MAX_DEPTH,'geometry encode depth')
        if type(node) is dict:
            yield b'{'; prior=None
            for index in range(len(node)):
                selected=None
                for name in node:
                    self.work('geometry_projected_keys')
                    require(type(name) is str and len(name)<=MAX_STRING,'projected digest key cap')
                    if (prior is None or name>prior) and (selected is None or name<selected): selected=name
                require(selected is not None,'geometry projected mapping changed')
                if index: yield b','
                require(encoded_size(selected)<=GEOMETRY_JSON_BYTES,'geometry scalar bytes')
                yield canonical(selected)+b':'
                yield from self.geometry_projected_tokens(node[selected],depth+1); prior=selected
            yield b'}'
        elif type(node) in (list,tuple):
            yield b'['
            for index,child in enumerate(node):
                if index: yield b','
                yield from self.geometry_projected_tokens(child,depth+1)
            yield b']'
        else:
            require(node is None or type(node) in (str,int,float,bool),'projected digest scalar type')
            require(encoded_size(node)<=GEOMETRY_JSON_BYTES,'geometry scalar bytes')
            yield canonical(node)

    def geometry_record(self,value,key='',depth=0,reference=True):
        require(depth<MAX_DEPTH,'geometry record workspace depth')
        self.ensure_geometry_registry(); qualified=self.geometry_schema(value)
        native=self.native_digest(value,key)
        kind='record' if qualified.endswith('.ImageDerivedGeometryResult') else 'geometry_inline_record'
        lookup=kind+':'+native
        if lookup in self.definitions:
            self.geometry_upgrade(value,depth)
            require(self.native_digest(value,key)==native,'geometry record changed during upgrade')
            self.validate_geometry_classes()
            if reference:
                self.budget.charge(384,2); return {'semantic_reference':lookup}
            return lookup  # borrowed wire token; no reference dict allocated
        require(id(value) not in self.projecting,'geometry record cycle')
        # Fixed temporary buffers use the already reserved sequential workspace.
        # Retained ASCII/chunks are charged after bounded encoding determines
        # their size, BEFORE their first allocation (no whole encode then split
        # beyond the charged fixed buffer). Definition metadata is charged too.
        self.projecting.add(id(value))
        try:
            buffer=bytearray(GEOMETRY_JSON_BYTES); offset=0
            def append(token):
                nonlocal offset
                require(offset+len(token)<=GEOMETRY_JSON_BYTES,'geometry value bytes')
                buffer[offset:offset+len(token)]=token; offset+=len(token)
            append(b'[')
            names,_annotations=self.geometry_registry[qualified]
            for index,name in enumerate(names):
                if index: append(b',')
                for token in self.geometry_tokens(getattr(value,name),name,depth+1): append(token)
            append(b']')
            chunks=(offset+4095)//4096
            self.budget.charge(512+offset+128*chunks,5 if chunks==1 else 6+chunks)
            text=bytes(memoryview(buffer)[:offset]).decode('ascii')
            body={'schema':qualified,'values_json':text} if offset<=4096 else {
                'schema':qualified,'values_json_chunks':[text[start:start+4096] for start in range(0,offset,4096)]}
            require(self.geometry_schema(value)==qualified,'geometry schema changed during projection')
            self.geometry_upgrade(value,depth)  # also verify late descendants
            require(self.native_digest(value,key)==native,'geometry record changed during projection')
            self.validate_geometry_classes()
            result=self.g_define(kind,body,native,'geometry-field-vector-v1')
            return result if reference else result['semantic_reference']
        finally: self.projecting.remove(id(value))

    def geometry_fields(self,body):
        self.ensure_geometry_registry()
        require(type(body) is dict and type(body.get('schema')) is str and
                body['schema'] in self.geometry_registry and
                (set(body)=={'schema','values_json'} or set(body)=={'schema','values_json_chunks'}),
                'geometry vector envelope')
        text=body.get('values_json'); chunks=body.get('values_json_chunks')
        if text is not None:
            require(type(text) is str and 0<len(text)<=4096,'geometry value text'); chunks=(text,)
        else:
            require(type(chunks) is list and 2<=len(chunks)<=4,'geometry value chunks')
        for index,piece in enumerate(chunks):
            self.work('geometry_json_chunk')
            require(type(piece) is str and piece.isascii() and 0<len(piece)<=4096 and
                    (index==len(chunks)-1 or len(piece)==4096),'geometry value chunk')
        raw=''.join(chunks)
        require(len(raw)<=GEOMETRY_JSON_BYTES and raw.startswith('[') and raw.endswith(']'),'geometry value bytes')
        # Parse one field at a time, so a parent vector's other 29 references
        # are never live while a descendant vector is parsed. Nodes AND JSON
        # separators reserve object/slot capacity, summed across live fields.
        start=1; index=0; depth=count=0; quoted=escaped=primitive=False
        for offset in range(1,len(raw)):
            char=raw[offset]; boundary=not quoted and depth==0 and (char==',' or offset==len(raw)-1)
            if boundary:
                require(index<len(self.geometry_registry[body['schema']][0]) and offset>start,'geometry vector length')
                self.geometry_live_nodes+=count
                try:
                    text=raw[start:offset]; value=strict_json(text.encode('ascii'))
                    require(canonical(value).decode('ascii')==text,'geometry value canonical')
                    yield index,value
                finally: self.geometry_live_nodes-=count
                index+=1; start=offset+1; count=0; primitive=False
                continue
            increment=0
            if quoted:
                if escaped: escaped=False
                elif char=='\\': escaped=True
                elif char=='"': quoted=False
            elif char=='"': quoted=True; increment=1; primitive=False
            elif char in '[{': depth+=1; increment=1; primitive=False
            elif char in ']}': depth-=1; primitive=False
            elif char in ',:': increment=1; primitive=False
            elif not char.isspace() and not primitive: increment=1; primitive=True
            if increment: self.work('geometry_parser'); count+=increment
            require(0<=depth<=MAX_DEPTH and self.geometry_live_nodes+count<=GEOMETRY_SCRATCH_NODES,
                    'geometry parser workspace')
        require(not quoted and depth==0 and index==len(self.geometry_registry[body['schema']][0]) and start==len(raw),
                'geometry vector length')

    def ensure_g_json_scratch(self):
        if not self.g_json_scratch:
            self.budget.charge(1024*1024,256); self.g_json_scratch=True

    def g_values(self,body):
        self.ensure_g_json_scratch()
        if 'values_json' in body:
            require(type(body['values_json']) is str and 0<len(body['values_json'])<=4096,'G value text')
            chunks=(body['values_json'],)  # sequential parser workspace, not retained output
        else: chunks=body.get('values_json_chunks')
        require(type(chunks) in (list,tuple) and 0<len(chunks)<=32,'G value chunks')
        for index,piece in enumerate(chunks):
            self.work('g_json_chunk')
            require(type(piece) is str and 0<len(piece)<=4096 and piece.isascii() and
                    (index==len(chunks)-1 or len(piece)==4096),'G value chunk')
        # Bound parser nodes/depth before parsing; quoted punctuation is data.
        depth=nodes=0; quoted=escaped=False
        for piece in chunks:
            self.work('g_json_preflight')
            for char in piece:
                if quoted:
                    if escaped: escaped=False
                    elif char=='\\': escaped=True
                    elif char=='"': quoted=False
                elif char=='"': quoted=True; nodes+=1
                elif char in '[{': depth+=1; nodes+=1
                elif char in ']}': depth-=1
                elif char in ',:': nodes+=1
                require(0<=depth<=3 and nodes<=128,'G value parser bound')
        require(not quoted and depth==0,'G value parser bound')
        values=strict_json(''.join(chunks).encode('ascii'))
        require(type(values) is list,'G value vector')
        names,annotations=self.g_registry[body['schema']]
        require(len(values)==len(names),'G vector length')
        for value,annotation in zip(values,annotations): self.g_check_value(value,annotation)
        require(canonical(values).decode('ascii')==''.join(chunks),'G value canonical')
        return values

    def g_check_value(self,value,annotation,decoded=True):
        self.work('g_field_type')
        if value is None: return
        if annotation=='str': require(type(value) is str and len(value)<=MAX_STRING,'G field string'); return
        if annotation=='tuple[int, int] | None':
            require(type(value) in ((tuple,list) if decoded else (tuple,)) and len(value)==2 and all(type(x) is int and x.bit_length()<=128 for x in value),
                    'G shift value'); return
        if annotation=='int': require(type(value) is int and value.bit_length()<=128,'G integer value'); return
        require(type(value) in (int,float) and (type(value) is not int or value.bit_length()<=128) or
                decoded and type(value) is dict and len(value)==1 and value.get('nonfinite') in ('nan','inf','-inf'),'G float value')

    def g_scope(self): return self.budget.stage.startswith('G_')

    def ensure_g_registry(self):
        if self.g_registry is None:
            self.budget.charge(8192,64)  # registry copies, schemas, keys, source receipt
            require(file_hash(ROOT/'src/mc_photogrammetry/reconstruction/plane_graph.py',MAX_SOURCE_BYTES)['sha256']==G_SOURCE_SHA256,
                    'G schema source pin')
            self.g_registry=dict(G_RECORD_SCHEMAS)
            self.g_registry_sha256=digest(self.g_registry)
        self.work('g_registry_digest'); current=digest(G_RECORD_SCHEMAS)
        self.work('g_registry_digest'); retained=digest(self.g_registry)
        require(self.g_registry_sha256==current==retained,'G schema registry changed')

    def g_scalar(self,value):
        self.work('g_scalar')
        if value is None or type(value) in (bool,int,float):
            require(type(value) is not int or value.bit_length()<=128,'integer cap')
            if type(value) is float and not math.isfinite(value):
                self.budget.charge(256,3); return {'nonfinite':repr(value)}
            return value  # borrowed primitive; enclosing vector slot is charged
        if type(value) is str:
            require(len(value)<=MAX_STRING,'string cap'); self.budget.charge(12*len(value),0)
            return value
        require(type(value) in (list,tuple) and len(value)==2 and all(type(x) is int for x in value),'G field value type')
        self.budget.charge(256,3)
        return [self.g_scalar(x) for x in value]

    def g_record(self,value,qualified,key):
        self.ensure_g_registry()
        # Retain the registry's already charged qualified-name object, not a
        # fresh concatenated string in every row.
        for registered in self.g_registry:
            if registered==qualified: qualified=registered; break
        names,annotations=self.g_registry[qualified]
        declared=type(value).__dataclass_fields__
        require(len(declared)==len(names),'G schema fields')
        for index,(name,field) in enumerate(declared.items()):
            self.work('g_schema')
            require(name==names[index] and field.type==annotations[index],'G schema fields/annotations')
        native=self.native_digest(value,key)
        kind='record' if qualified.endswith('.PlaneMatch') else 'g_inline_record'
        lookup=kind+':'+native
        if lookup in self.definitions:
            self.budget.charge(384,2); return {'semantic_reference':lookup}
        self.ensure_g_json_scratch()
        # One temporary <=16-value vector lives in the charged parser workspace.
        # Its canonical ASCII is retained as chunks; no per-field map or scalar
        # vector slots remain in the receipt after encoding.
        values=[]
        for name,annotation in zip(names,annotations):
            item=getattr(value,name)
            self.g_check_value(item,annotation,decoded=False)
            values.append({'nonfinite':repr(item)} if type(item) is float and not math.isfinite(item) else item)
        size=encoded_size(values); require(size<=128*1024,'G value bytes')
        count=(size+4095)//4096
        # Two-entry dict + chunk-list header fit512B; each ASCII chunk has
        #128B allocator/list/JSON overhead plus its actual characters. The
        # separately charged1024B definition covers hashes/lookup/ref metadata.
        self.budget.charge(512+size+128*count,5 if count==1 else 6+count)
        text=canonical(values).decode('ascii')
        # Short vectors retain the actual string directly: no singleton chunk
        # list/header/slot. Long vectors still use bounded ordered chunks.
        body={'schema':qualified,'values_json':text} if count==1 else {
            'schema':qualified,'values_json_chunks':[text[offset:offset+4096] for offset in range(0,len(text),4096)]}
        require(self.native_digest(value,key)==native,'record changed during projection')
        return self.g_define(kind,body,native,'G-field-vector-v1')

    def g_define(self,kind,body,native,encoding):
        self.budget.charge(1024,7)
        lookup=kind+':'+native
        self.definitions[lookup]={'kind':kind,'native_sha256':native,'projection':body,'encoding':encoding,
            'content_sha256':self.projected_digest(body,encoding=encoding),
            'wire_sha256':self.projected_digest(body,legacy=False)}
        return {'semantic_reference':lookup}

    def g_dispositions(self,matches,kept,identity=None):
        # The trace's two-valued ordered evidence is a bit vector, not a second
        # retained candidate-length string list. Its legacy view remains the
        # exact list of 'kept'/'dropped' strings, including empty lists.
        self.ensure_g_json_scratch()
        count=len(matches); require(count<=MAX_ITEMS,'disposition count cap')
        size=(count+7)//8; text_size=4*((size+2)//3)
        self.budget.charge(2048+4*size+text_size,16)
        bits=bytearray(size); fingerprint=hashlib.sha256()
        def emit(text):
            self.work('g_disposition_digest')
            raw=text.encode(); fingerprint.update(str(len(raw)).encode()+b':'); fingerprint.update(raw)
        emit('builtins.list'); emit(str(count))
        for index,item in enumerate(matches):
            accepted=(id(item) if identity is None else identity(item)) in kept
            if accepted: bits[index//8]|=1<<(index%8)
            word='kept' if accepted else 'dropped'
            emit('builtins.str'); emit(str(len(word))); fingerprint.update(word.encode())
        native=fingerprint.hexdigest(); lookup='g_sequence:'+native
        if lookup in self.definitions:
            self.budget.charge(384,2); return {'semantic_reference':lookup}
        body={'container_type':'builtins.list','count':count,
              'bits_base64':base64.b64encode(bits).decode('ascii')}
        return self.g_define('g_sequence',body,native,'G-dispositions-v1')

    def g_container(self,value,key,depth,seen):
        native=self.native_digest(value,key)
        kind='g_mapping' if isinstance(value,Mapping) else 'g_sequence'
        lookup=kind+':'+native
        if lookup in self.definitions:
            # Native content/kind is independent of the capture key. Sequence
            # children inherit that key, so a descriptor-first cache entry must
            # still honor a later coordinates/matches retention contract.
            if key in NUMERIC_VALUE_KEYS and isinstance(value,(list,tuple)):
                self.upgrade_numeric_descendants(value,key,depth)
                require(self.native_digest(value,key)==native,'container changed during numeric upgrade')
            self.budget.charge(384,2); return {'semantic_reference':lookup}
        require(self.g_scope(),'new G container outside G scope')
        self.g_containers=True
        module=type(value).__module__; name=type(value).__qualname__
        require(type(module) is str and type(name) is str and len(module)+len(name)+1<=MAX_STRING,'G container kind cap')
        self.budget.charge(1024+12*(len(module)+len(name)+1),6)
        if kind=='g_mapping':
            body={'container_type':type(value).__module__+'.'+type(value).__qualname__,
                  'items':self.mapping_projection(value,depth,seen)}
        else:
            self.budget.charge(128*len(value),len(value)+2)
            body={'container_type':type(value).__module__+'.'+type(value).__qualname__,
                  'items':[self.g_scalar(child) if child is None or type(child) in (bool,int,float,str)
                           else self.take(child,key,depth+1,seen) for child in value]}
        require(self.native_digest(value,key)==native,'container changed during projection')
        return self.g_define(kind,body,native,'G-container-v1')

    def upgrade_numeric_descendants(self,value,key,depth):
        # A selective, scratch-only walk: do not reallocate cached rows/refs.
        # Only list/tuple descendants inherit the outer key. Mapping values and
        # dataclass fields use their own field names, so their retention context
        # was fixed when those definitions were originally projected.
        self.work('numeric_context_upgrade')
        require(depth<=MAX_DEPTH,'numeric context depth')
        if isinstance(value,(list,tuple)):
            for child in value: self.upgrade_numeric_descendants(child,key,depth+1)
        elif hasattr(value,'dtype') and hasattr(value,'shape') and hasattr(value,'nbytes'):
            fingerprint=self.native_digest(value,key); lookup='array:'+fingerprint
            require(lookup in self.definitions,'undefined numeric context reference')
            definition=self.definitions[lookup]
            if value.size<=MAX_COORDINATE_ELEMENTS and NUMERIC_VALUES_FIELD not in definition['values_projection']:
                # Existing direct-array upgrade precharges the one payload and
                # rehashes after capture. The returned temporary ref is charged.
                self.take(value,key,depth)

    def ensure_scratch(self):
        if not self.scratch_reserved:
            # Shared sequential walker/hash/JSON-token scratch, not retained
            # output nodes. Covers four BLOCK buffers plus bounded recursion.
            self.budget.charge(4*BLOCK+MAX_DEPTH*4096,MAX_DEPTH)
            self.scratch_reserved=True

    def work(self,walk):
        require(self.validation_visits<MAX_VALIDATION_WORK,'semantic validation work cap')
        self.validation_visits+=1
        self.validation_walk_counts[walk]=self.validation_walk_counts.get(walk,0)+1
        self.budget.charge(0,0)

    def mapping_projection(self,items,depth=0,seen=None,projected=False):
        # Every source mapping has this representation, including empty maps,
        # ordinary marker-named string keys and mixed string/int/tuple keys.
        # Values already projected for a dataclass field map are not projected
        # again: an actual internal reference must remain an internal reference.
        self.budget.charge(512+512*len(items),4+4*len(items))
        require(all(type(k) in (str,int,tuple) for k in items),'semantic mapping key type')
        seen=set() if seen is None else seen
        require(id(items) not in seen,'semantic mapping cycle')
        seen.add(id(items))
        try:
            return {'native_mapping_entries':[{'key_type':type(k).__name__,
                'key':self.take(k,'',depth+1,seen),
                'value':v if projected else self.take(v,k if type(k) is str else '',depth+1,seen)}
                for k,v in items.items()]}
        finally: seen.remove(id(items))

    def record_fields(self,projected):
        return self.mapping_projection(projected,projected=True)

    def native_digest(self,value,key=''):
        # A bounded, streaming validation walk, not a retained snapshot. Charge
        # reusable hashing/recursion scratch once; separately bound work visits.
        # Every allocated receipt node is still charged by take/define below.
        self.ensure_scratch()
        fingerprint=hashlib.sha256(); active=set()
        def emit(text):
            require(type(text) is str and len(text)<=MAX_STRING,'semantic token cap')
            raw=text.encode('utf-8'); fingerprint.update(str(len(raw)).encode()+b':'); fingerprint.update(raw)
        def walk(item,name='',depth=0):
            self.work('native_digest')
            require(depth<=MAX_DEPTH,'semantic validation depth')
            kind=type(item); emit(kind.__module__+'.'+kind.__qualname__)
            if item is None: return
            if kind in (bool,int,float):
                require(kind is not int or item.bit_length()<=128,'integer cap')
                emit(item.hex() if kind is float else str(item)); return
            if isinstance(item,(str,Path)):
                text=str(item)
                require(len(text)<=(MAX_ARRAY if name in ENCODED_PAYLOAD_FIELDS else MAX_STRING),'string cap')
                require(name not in ENCODED_PAYLOAD_FIELDS or text.isascii(),'encoded lattice ASCII')
                emit(str(len(text)))
                for offset in range(0,len(text),BLOCK):
                    self.budget.charge(0,0); fingerprint.update(text[offset:offset+BLOCK].encode('utf-8'))
                return
            if kind is bytes:
                require(len(item)<=MAX_ARRAY,'buffer cap'); emit(str(len(item)))
                view=memoryview(item)
                for offset in range(0,len(item),BLOCK):
                    self.budget.charge(0,0); fingerprint.update(view[offset:offset+BLOCK])
                return
            if hasattr(item,'dtype') and hasattr(item,'shape') and hasattr(item,'nbytes'):
                self.array_contract(item)
                emit(item.dtype.str); emit(str(tuple(item.shape))); emit(str(tuple(item.strides)))
                emit(str(bool(item.flags.writeable))); emit(str(item.nbytes))
                if item.flags.c_contiguous:
                    view=memoryview(item).cast('B') if item.nbytes else memoryview(b'')
                    for offset in range(0,item.nbytes,BLOCK):
                        self.numeric_work('numeric_hash'); fingerprint.update(view[offset:offset+BLOCK])
                else:
                    require(item.nbytes+self.budget.reserved+MAX_RECEIPT<=MANAGED_RESERVATION,'array scratch cap')
                    self.budget.charge(item.nbytes+256,1)
                    self.numeric_work('numeric_hash')
                    fingerprint.update(item.tobytes(order='C'))
                return
            if kind is Fraction:
                walk(item.numerator,depth=depth+1); walk(item.denominator,depth=depth+1); return
            if inspect.isfunction(item):
                # Same observable function contract as Snapshot; no invocation.
                emit(item.__module__+'.'+item.__qualname__); emit(sha(marshal.dumps(item.__code__,2))); return
            if hasattr(item,'item') and kind.__module__.startswith('numpy'):
                walk(item.item(),name,depth+1); return
            compound=is_dataclass(item) or isinstance(item,(Mapping,list,tuple))
            if not compound: return  # opaque source field: type only, explicitly reported
            require(id(item) not in active,'semantic record cycle'); active.add(id(item))
            try:
                if is_dataclass(item):
                    for declared in fields(item):
                        emit(declared.name); walk(getattr(item,declared.name),declared.name,depth+1)
                elif isinstance(item,Mapping):
                    emit(str(len(item)))
                    for field,child in item.items():
                        require(type(field) in (str,int,tuple),'semantic mapping key type')
                        walk(field,'',depth+1); walk(child,field if type(field) is str else '',depth+1)
                else:
                    emit(str(len(item)))
                    for child in item: walk(child,name,depth+1)
            finally: active.remove(id(item))
        walk(value,key)
        return fingerprint.hexdigest()

    def projected_digest(self,body,encoding=None,legacy=True):
        self.ensure_scratch()
        result=hashlib.sha256(); expanding=set()
        def encoded(node,mode,depth):
            self.work('g_logical_view')
            require(depth<=MAX_DEPTH,'projected digest depth')
            if mode=='geometry-field-vector-v1':
                yield b'{"fields":{"native_mapping_entries":['
                for index,value in self.geometry_fields(node):
                    name=self.geometry_registry[node['schema']][0][index]
                    self.work('geometry_logical_field')
                    if index: yield b','
                    yield b'{"key":'+canonical(name)+b',"key_type":"str","value":'
                    yield from walk(value,depth+4)
                    yield b'}'
                yield b']},"type":'+canonical(node['schema'])+b'}'
                return
            if mode=='G-container-v1':
                require(type(node) is dict and len(node)==2 and type(node.get('container_type')) is str and
                        'items' in node,'G container encoding')
                yield from walk(node['items'],depth)
                return
            if mode=='G-dispositions-v1':
                self.ensure_g_json_scratch()
                require(type(node) is dict and set(node)=={'container_type','count','bits_base64'} and
                        node['container_type']=='builtins.list' and type(node['count']) is int and
                        0<=node['count']<=MAX_ITEMS,'G disposition encoding')
                count=node['count']; size=(count+7)//8; text=node['bits_base64']
                require(type(text) is str and text.isascii() and len(text)==4*((size+2)//3),
                        'G disposition length')
                self.work('g_disposition_decode')
                try: bits=base64.b64decode(text,validate=True)
                except ValueError as error: raise IntegrityError('G disposition alphabet') from error
                require(len(bits)==size and base64.b64encode(bits).decode('ascii')==text and
                        (not count%8 or not bits[-1]>>(count%8)),'G disposition canonical')
                yield b'['
                for index in range(count):
                    self.work('g_disposition_logical')
                    if index: yield b','
                    yield b'"kept"' if bits[index//8]&(1<<(index%8)) else b'"dropped"'
                yield b']'
                return
            require(mode=='G-field-vector-v1','G encoding')
            self.ensure_g_registry()
            require(type(node) is dict and len(node)==2 and type(node.get('schema')) is str and node['schema'] in self.g_registry and
                    (set(node)=={'schema','values_json'} or set(node)=={'schema','values_json_chunks'}),'G record encoding')
            names,_=self.g_registry[node['schema']]
            values=self.g_values(node)
            # Canonical legacy record/typed-map field view. The fixed wrappers
            # are emitted, never allocated; only newly-inline definitions expand.
            yield b'{"fields":{"native_mapping_entries":['
            for index,name in enumerate(names):
                self.work('g_logical_field')
                if index: yield b','
                yield b'{"key":'+canonical(name)+b',"key_type":"str","value":'
                yield from walk(values[index],depth+4)
                yield b'}'
            yield b']},"type":'+canonical(node['schema'])+b'}'
        def walk(node,depth=0):
            # Work refusal precedes canonical scalar encoding/key sorting.
            self.work('projected_digest')
            require(depth<=MAX_DEPTH,'projected digest depth')
            if type(node) is dict:
                if legacy and len(node)==1 and 'semantic_reference' in node:
                    reference=node['semantic_reference']; record=self.definitions.get(reference) if type(reference) is str else None
                    if record is not None and record['kind'] in ('g_inline_record','g_sequence','g_mapping','geometry_inline_record'):
                        require(reference not in expanding,'G logical reference cycle')
                        expanding.add(reference)
                        try: yield from encoded(record['projection'],record['encoding'],depth)
                        finally: expanding.remove(reference)
                        return
                for key in node:
                    self.work('projected_keys')
                    require(type(key) is str and len(key)<=MAX_STRING,'projected digest key cap')
                # Select the next lexical key by rescanning the native dict.
                # No sorted-key list or retained cache is allocated. At most
                # MAX_DEPTH iterator/candidate references coexist; the existing
                # per-depth scratch precharge covers these bounded frames.
                # Every inspected key is work-metered before comparison, so
                # quadratic work on wide dictionaries refuses at the shared cap.
                yield b'{'
                prior=None
                for index in range(len(node)):
                    key=None
                    for candidate in node:
                        self.work('projected_key_scan')
                        if (prior is None or candidate>prior) and (key is None or candidate<key): key=candidate
                    require(key is not None,'projected dictionary changed')
                    self.work('projected_key_tokens')
                    if index: yield b','
                    yield canonical(key)+b':'
                    yield from walk(node[key],depth+1)
                    prior=key
                yield b'}'
            elif type(node) in (list,tuple):
                yield b'['
                for index,child in enumerate(node):
                    if index: yield b','
                    yield from walk(child,depth+1)
                yield b']'
            else:
                require(node is None or type(node) in (str,int,float,bool),'projected digest scalar type')
                require(type(node) is not str or len(node)<=MAX_STRING,'projected digest string cap')
                require(type(node) is not int or node.bit_length()<=128,'projected digest integer cap')
                yield canonical(node)
        for token in (encoded(body,encoding,0) if encoding is not None else walk(body)): result.update(token)
        return result.hexdigest()

    def define(self,kind,body,native):
        self.budget.charge(256,2)  # digest/reference/lookup metadata before allocation
        content=self.projected_digest(body); key=kind+':'+native
        if key not in self.definitions:
            self.budget.charge(512,3)
            self.definitions[key]={'kind':kind,'native_sha256':native,'content_sha256':content,'projection':body}
        else:
            require(self.definitions[key]['content_sha256']==content,'semantic definition binding')
        return {'semantic_reference':key}

    def take(self,value,key='',depth=0,seen=None):
        require(depth<=MAX_DEPTH,'snapshot depth')
        module=type(value).__module__; name=type(value).__qualname__
        require(type(module) is str and type(name) is str and len(module)+len(name)+1<=MAX_STRING,'semantic type name cap')
        qualified=module+'.'+name
        if self.budget.stage=='geometry' and is_dataclass(value) and name in GEOMETRY_RECORD_NAMES:
            require(qualified in GEOMETRY_RECORD_SCHEMAS,'geometry schema namespace')
        if is_dataclass(value) and qualified in GEOMETRY_RECORD_SCHEMAS and (
                self.budget.stage=='geometry' and (name!='ImageDerivedGeometryResult' or getattr(value,'records',None) is not None)
                or self.geometry_registry is not None):
            return self.geometry_record(value,key,depth)
        if is_dataclass(value) and qualified in G_RECORD_SCHEMAS:
            kind='record' if qualified.endswith('.PlaneMatch') else 'g_inline_record'
            if self.g_scope() or self.g_registry is not None and kind+':'+self.native_digest(value,key) in self.definitions:
                return self.g_record(value,qualified,key)
        if isinstance(value,(Mapping,list,tuple)) and (self.g_scope() or self.g_containers):
            kind='g_mapping' if isinstance(value,Mapping) else 'g_sequence'
            if self.g_scope() or kind+':'+self.native_digest(value,key) in self.definitions:
                return self.g_container(value,key,depth,seen)
        if isinstance(value,Mapping): return self.mapping_projection(value,depth,seen)
        source=PROJECTION_MODULES.get(type(value).__name__,())
        source=(source,) if type(source) is str else source
        projected=is_dataclass(value) and type(value).__module__ in tuple('mc_photogrammetry.'+x for x in source)
        if projected:
            native=self.native_digest(value,key); lookup='record:'+native
            if lookup in self.definitions:
                self.budget.charge(256,2)
                return {'semantic_reference':lookup}
            require(id(value) not in self.projecting,'semantic record cycle')
            self.projecting.add(id(value))
            try:
                body=super().take(value,key,depth,seen)
                require(self.native_digest(value,key)==native,'record changed during projection')
                return self.define('record',body,native)
            finally: self.projecting.remove(id(value))
        if hasattr(value,'dtype') and hasattr(value,'shape') and hasattr(value,'nbytes'):
            # Hash-only validation first. A changed array has a new definition;
            # ordered coordinate/edge values expand only on its first definition.
            fingerprint=self.native_digest(value,key); lookup='array:'+fingerprint
            self.budget.charge(256,2)
            if lookup not in self.definitions:
                self.budget.charge(32)
                full=self.array_snapshot(value,key)
                self.budget.charge(1024,7)
                body={name:item for name,item in full.items() if name!=NUMERIC_VALUES_FIELD}
                self.budget.charge(768,5)
                require(self.native_digest(value,key)==fingerprint,'array changed during projection')
                require(all(full[k]==v for k,v in body.items()),'array changed during projection')
                self.definitions[lookup]={'kind':'array','native_sha256':fingerprint,'content_sha256':self.projected_digest(body),
                                           'projection':body,'values_projection':full,
                                           'values_sha256':self.projected_digest(full[NUMERIC_VALUES_FIELD]) if NUMERIC_VALUES_FIELD in full else None}
            elif key in NUMERIC_VALUE_KEYS and value.size<=MAX_COORDINATE_ELEMENTS and \
                    NUMERIC_VALUES_FIELD not in self.definitions[lookup]['values_projection']:
                # A hash-only first use cannot erase a later coordinate contract.
                self.budget.charge(32); full=self.array_snapshot(value,key)
                require(self.native_digest(value,key)==fingerprint,'array changed during projection')
                require(all(full[k]==v for k,v in self.definitions[lookup]['projection'].items()),'array changed during projection')
                self.definitions[lookup]['values_projection']=full
                self.definitions[lookup]['values_sha256']=self.projected_digest(full[NUMERIC_VALUES_FIELD])
            return {'semantic_reference':lookup}
        return super().take(value,key,depth,seen)

    def validate(self,roots=()):
        # Called before publication; no reference can claim an undefined body.
        self.budget.charge(4096+256*len(PROJECTION_MODULES),4+4*len(PROJECTION_MODULES))
        self.ensure_scratch()
        if self.geometry_registry is not None:
            self.ensure_geometry_registry()
            self.validate_geometry_classes()
        for key,record in self.definitions.items():
            if 'encoding' in record:
                mode=record['encoding']; body=record['projection']; kind=record['kind']
                require(mode in ('G-field-vector-v1','G-container-v1','G-dispositions-v1','geometry-field-vector-v1'),'G encoding')
                if mode=='geometry-field-vector-v1':
                    require(type(body) is dict and body.get('schema') in GEOMETRY_RECORD_SCHEMAS and
                            kind==('record' if body['schema'].endswith('.ImageDerivedGeometryResult') else 'geometry_inline_record'),
                            'geometry definition schema/kind')
                elif mode=='G-field-vector-v1':
                    require(type(body) is dict and type(body.get('schema')) is str and
                            body['schema'] in G_RECORD_SCHEMAS and kind==('record' if body['schema'].endswith('.PlaneMatch') else 'g_inline_record'),
                            'G definition schema/kind')
                elif mode=='G-dispositions-v1': require(kind=='g_sequence','G definition kind')
                else:
                    require(kind in ('g_sequence','g_mapping') and type(body) is dict and
                            (type(body.get('items')) is list if kind=='g_sequence' else type(body.get('items')) is dict),
                            'G definition container kind')
            require(key==record['kind']+':'+record['native_sha256'] and
                    record['content_sha256']==self.projected_digest(record['projection'],encoding=record.get('encoding')),
                    'semantic definition changed')
            if 'encoding' in record:
                require(record['wire_sha256']==self.projected_digest(record['projection'],legacy=False),'G wire binding')
            if record['kind']=='array':
                full=record['values_projection']; has_values=NUMERIC_VALUES_FIELD in full
                require(len(full)==len(record['projection'])+int(has_values) and
                        all(full[k]==v for k,v in record['projection'].items()),
                        'semantic array binding')
                require(record['values_sha256']==(self.projected_digest(full[NUMERIC_VALUES_FIELD]) if has_values else None),
                        'semantic array values changed')
                if has_values: self.validate_numeric(full)
        def references(node,depth=0):
            self.work('references')
            require(depth<=MAX_DEPTH,'semantic reference depth')
            if type(node) is dict:
                if 'semantic_reference' in node:
                    require(len(node)==1 and type(node['semantic_reference']) is str and
                            node['semantic_reference'] in self.definitions,
                            'undefined semantic reference')
                else:
                    for child in node.values(): references(child,depth+1)
            elif type(node) in (list,tuple):
                for child in node: references(child,depth+1)
        for record in self.definitions.values():
            if record.get('encoding')=='geometry-field-vector-v1':
                for index,value in self.geometry_fields(record['projection']): references(value)
            else: references(record['projection'])
        references(roots)
        return {'schema':'stage-definitions-v5','types':PROJECTION_MODULES,
                'G_schemas':self.g_registry,'G_schema_sha256':self.g_registry_sha256,
                'geometry_schemas':self.geometry_registry,'geometry_schema_sha256':self.geometry_registry_sha256,
                'native_mappings':'all source and dataclass field maps: ordered native_mapping_entries with typed keys',
                'record_fields':'all declared; nested semantic references',
                'arrays':'primitive dtype/endian, shape, source strides/writeability; lossless logical C-order base64 chunks once; backing layout not reconstructed',
                'mutable':'fresh before/after argument and kwargs versions',
                'definition_count':len(self.definitions),'validation_visits':self.validation_visits,
                'validation_walk_counts':dict(self.validation_walk_counts),
                'validation_work_cap':MAX_VALIDATION_WORK,'opaque_fields':'type only; no hidden-state claim'}


def error_record(error):
    # Avoid unconstrained repr/str of a foreign exception/object.
    result={'type':type(error).__module__+'.'+type(error).__qualname__}
    for name in ('reason','detail'):
        value=getattr(error,name,None)
        if type(value) is str: result[name]=value[:MAX_STRING]
    result['message']=str(error.args[0])[:MAX_STRING] if error.args and type(error.args[0]) is str else ''
    return result


class Observer:
    def __init__(self,budget=None):
        self.budget=budget or Budget(); self.snapshot=Snapshot(self.budget)
        self.budget.partitioned=True; self.projection=StageProjection(self.budget)
        require(sum(x['items'] for x in ALLOCATION_PLAN.values())==MAX_ITEMS and
                sum(x['bytes'] for x in ALLOCATION_PLAN.values())==MAX_OBSERVER,'allocation ledger')
        self.stages={name:{'state':'not_reached','calls':0,'records':[]} for name in STAGES}
        self.failure=None; self.g_codes={}; self.g_stages={}; self.trace_seen=[]; self.decoded_inputs=[]
        self.identities={}

    def identity(self,value):
        # Ordinal alias binding only, not a semantic snapshot cache. Strong
        # references prevent object-address reuse during this invocation.
        if id(value) not in self.identities:
            self.budget.charge(128,2)
            self.identities[id(value)]=(value,len(self.identities))
        return self.identities[id(value)][1]

    def ordinal_vector(self,value,signature_readings=False):
        projection=self.projection; projection.ensure_scratch()
        count=len(value); require(count<=MAX_ITEMS,'ordinal count cap')
        def ordinal(item):
            if signature_readings:
                return self.identity(item.reading) if type(item).__name__=='PlaneSignature' and \
                    type(item).__module__=='mc_photogrammetry.reconstruction.lattice_planes' else None
            return self.identity(item)
        def fingerprint(values):
            result=hashlib.sha256()
            def emit(text):
                projection.work('g_ordinal_digest')
                raw=text.encode(); result.update(str(len(raw)).encode()+b':'); result.update(raw)
            emit('builtins.list'); emit(str(count))
            for item in values:
                number=ordinal(item); emit('builtins.NoneType' if number is None else 'builtins.int')
                if number is not None: emit(str(number))
            return result.hexdigest()
        native=fingerprint(value); lookup='g_sequence:'+native
        if lookup in projection.definitions:
            self.budget.charge(384,2); return {'semantic_reference':lookup}
        self.budget.charge(1024+128*count,6+count)
        values=[ordinal(item) for item in value]
        require(projection.native_digest(values)==native,'ordinal sequence changed')
        body={'container_type':'builtins.list','items':values}
        return projection.g_define('g_sequence',body,native,'G-container-v1')

    def ordered_identities(self,args):
        self.budget.charge(256,2)
        result={}
        for index,value in enumerate(args):
            if isinstance(value,(list,tuple)):
                result[str(index)]=self.ordinal_vector(value)
                if any(type(item).__name__=='PlaneSignature' and
                       type(item).__module__=='mc_photogrammetry.reconstruction.lattice_planes' for item in value):
                    result['signature_reading:'+str(index)]=self.ordinal_vector(value,True)
            elif isinstance(value,Mapping):
                result[str(index)]=self.ordinal_vector(value.values())
        return result

    def bind_decode(self,args,kwargs,slot):
        # The public path-only call is already entered; block original decoder
        # forwarding on TOCTOU/ordering mismatch without a second filesystem read.
        self.budget.charge(1024)
        require(len(args)==1 and type(args[0]) is bytes and set(kwargs)=={'suffix'} and
                kwargs['suffix']=='.png','decoder input contract')
        data=args[0]; require(len(data)<=MAX_FILE_BYTES,'decoder input cap')
        fingerprint=hashlib.sha256()
        view=memoryview(data)
        for offset in range(0,len(data),BLOCK):
            self.budget.charge(0,0); fingerprint.update(view[offset:offset+BLOCK])
        receipt={'slot':slot,'path':INPUTS[slot],'bytes':len(data),'sha256':fingerprint.hexdigest(),
                 'expected_sha256':INPUT_HASHES[slot],'delegated':False}
        self.decoded_inputs.append(receipt)
        require(receipt['sha256']==INPUT_HASHES[slot],'actual decoder input pin')
        return receipt

    def delegate(self,stage,original):
        require(stage in STAGES,'observer stage')
        def capture(*args,**kwargs):
            state=self.stages[stage]
            try:
                self.budget.charge(2048)
                require(state['calls']<STAGES[stage],'observer call cap')
                state['calls']+=1; state['state']='reached'
                decoded=self.bind_decode(args,kwargs,state['calls']-1) if stage=='decode' else None
                self.budget.charge(64*len(args),len(args))
                record={'arguments':self.projection.take(args),'kwargs':self.projection.take(kwargs),
                        'argument_identity':[self.identity(x) for x in args]}
                if stage.startswith('G_'): record['ordered_argument_identity']=self.ordered_identities(args)
                state['records'].append(record)
            except BaseException as error:
                self.failure=error_record(error); state['state']='instrumentation_failure'; raise
            try:
                if decoded is not None: decoded['delegated']=True
                result=original(*args,**kwargs)  # identical args; exactly once
            except BaseException as error:
                record['exception']=error_record(error); state['state']='raised'
                try:
                    record['arguments_after']=self.projection.take(args)
                    record['kwargs_after']=self.projection.take(kwargs)
                    if stage.startswith('G_'): record['ordered_argument_identity_after']=self.ordered_identities(args)
                except BaseException as after:
                    self.failure=error_record(after); state['state']='instrumentation_failure'
                raise
            try:
                record['arguments_after']=self.projection.take(args)
                record['kwargs_after']=self.projection.take(kwargs)
                if stage.startswith('G_'): record['ordered_argument_identity_after']=self.ordered_identities(args)
                record['result']=self.projection.take(result)
                record['result_identity']=self.identity(result)
                state['state']='refused' if (type(result).__name__=='PlacementRefusal' or
                    getattr(result,'status',None) in ('refused','unavailable','geometry_unavailable')) else 'returned'
            except BaseException as error:
                self.failure=error_record(error); state['state']='instrumentation_failure'; raise
            return result  # exact original returned object
        def observed(*args,**kwargs):
            with self.budget.scope(stage): return capture(*args,**kwargs)
        return observed

    def trace(self,frame,event,arg):
        if event=='return' and frame.f_code in self.g_codes:
            try:
                with self.budget.scope(self.g_stages.get(frame.f_code,self.budget.stage)):
                    self.budget.charge(2048)
                    selected={name:frame.f_locals[name] for name in self.g_codes[frame.f_code] if name in frame.f_locals}
                    controlled={}
                    if 'matches' in selected and 'filtered' in selected:
                        count=len(selected['matches']); kept_count=len(selected['filtered'])
                        require(0<=kept_count<=count<=MAX_ITEMS,'candidate count cap')
                        # Position dict slots plus newly enumerated position
                        # integers, kept-set slots and kept-index list slots.
                        # Keys reuse already retained stable ordinal integers;
                        # no fresh address integers become dictionary/set keys.
                        # Ordinals reuse their interned vector; dispositions
                        # allocate a charged bitmap instead of a string list.
                        self.budget.charge(512*(count+kept_count),2*(count+kept_count))
                        positions={self.identity(x):i for i,x in enumerate(selected['matches'])}
                        require(len(positions)==count,'duplicate candidate identity')
                        require(all(self.identity(x) in positions for x in selected['filtered']),'kept candidate outside universe')
                        kept={self.identity(x) for x in selected['filtered']}
                        require(len(kept)==kept_count,'duplicate kept candidate identity')
                        controlled['candidate_identity']=self.ordinal_vector(selected['matches'])
                        selected['kept_candidate_indices']=[positions[self.identity(x)] for x in selected['filtered']]
                        controlled['dispositions']=self.projection.g_dispositions(selected['matches'],kept,self.identity)
                    if 'modes' in selected and 'best' in selected:
                        self.budget.charge(256)
                        selected['winner_index']=next((i for i,m in enumerate(selected['modes']) if m is selected['best']),None)
                    self.budget.charge(256*(len(selected)+len(controlled)),len(selected)+len(controlled))
                    # Controlled observer metadata; source values each receive
                    # their own source mapping/record projection.
                    projected={name:self.projection.take(value,name) for name,value in selected.items()}
                    projected.update(controlled); self.trace_seen.append(projected)
            except BaseException as error:
                self.failure=error_record(error); raise
        return self.trace if frame.f_code in self.g_codes else None

    @contextmanager
    def install(self,targets):
        require(sys.gettrace() is None,'existing trace')
        originals=[]
        try:
            for module,name,stage in targets:
                original=getattr(module,name)
                require(inspect.isfunction(original),'observer callable type')
                originals.append((module,name,original))
                if stage=='G_matcher': self.g_codes[original.__code__]=('matches','filtered','refusals')
                if stage=='modes': self.g_codes[original.__code__]=('modes','best','winner_refusal','support_floor','bounds','scores')
                self.g_stages[original.__code__]=stage
                setattr(module,name,self.delegate(stage,original))
            sys.settrace(self.trace)
            yield
        finally:
            sys.settrace(None)
            for module,name,original in reversed(originals): setattr(module,name,original)
            require(all(getattr(m,n) is f for m,n,f in originals),'observer restore')


class RealIO:
    """Only used by a separately authorized run; tests use an in-memory adapter."""
    def reserve(self,parent,name):
        require(type(name) is str and 1<=len(name)<=64 and
                all(x.isalnum() or x in '-_' for x in name),'run name')
        parent=Path(parent).resolve(strict=True)
        require(parent.is_dir() and not parent.is_symlink(),'output parent')
        target=parent/name
        target.mkdir()  # exclusive; never overwrite/reuse/delete
        return target

    def exists(self,path): return Path(path).exists()
    def open_exclusive(self,path): return Path(path).open('xb')
    def flush(self,stream): stream.flush(); os.fsync(stream.fileno())
    def read_hash(self,path,limit): return file_hash(Path(path),limit)
    def read_small(self,path,limit):
        require(Path(path).is_file() and not Path(path).is_symlink(),'metadata file type')
        with Path(path).open('rb') as stream: data=stream.read(limit+1)
        require(len(data)<=limit,'read byte cap'); return data


@dataclass(frozen=True)
class Seal:
    directory: Path
    receipt_sha256: str
    receipt_bytes: int
    marker_sha256: str
    native_complete: bool


class Writer:
    def __init__(self,io,parent,name):
        self.io=io; self.directory=io.reserve(parent,name); self.sealed=None
        self.path=self.directory/'native-receipt.json'
        self.marker=self.directory/'COMPLETE.json'
        self.project=self.directory/'prediction.mcscan'
        require(not io.exists(self.project) and not io.exists(self.marker),'fresh target')
        self.stream=io.open_exclusive(self.path)
        # Bound physical receipt writes before allocation; reserve before inference.
        require(MAX_RECEIPT<=MANAGED_RESERVATION and MAX_RECEIPT%BLOCK==0,'preallocation cap')
        zeros=bytes(BLOCK)
        for _ in range(MAX_RECEIPT//BLOCK): self.stream.write(zeros)
        io.flush(self.stream); self.stream.seek(0)

    def seal(self,record,native_complete):
        require(self.sealed is None,'seal repeat')
        size=encoded_size(record)+1
        require(size<=MAX_RECEIPT,'receipt prewrite cap')
        fingerprint=hashlib.sha256(); count=0
        for token in tokens(record):
            require(count+len(token)<=size,'receipt encoding mismatch')
            self.stream.write(token); fingerprint.update(token); count+=len(token)
        self.stream.write(b'\n'); fingerprint.update(b'\n'); count+=1
        require(count==size,'receipt encoding mismatch')
        self.stream.truncate(size); self.io.flush(self.stream); self.stream.close()
        readback=self.io.read_hash(self.path,MAX_RECEIPT)
        require(readback=={'sha256':fingerprint.hexdigest(),'bytes':size},'receipt readback mismatch')
        marker={'schema':'sealed-native-receipt-v1','receipt_sha256':readback['sha256'],
                'receipt_bytes':size,'native_complete':bool(native_complete)}
        marker_bytes=canonical(marker)+b'\n'
        require(len(marker_bytes)<=4096,'marker prewrite cap')
        with self.io.open_exclusive(self.marker) as stream:
            stream.write(marker_bytes); self.io.flush(stream)
        require(self.io.read_small(self.marker,4096)==marker_bytes,'marker readback mismatch')
        seal=Seal(self.directory,readback['sha256'],size,sha(marker_bytes),bool(native_complete))
        self.verify(seal,allow_failure=True)
        self.sealed=seal
        return seal

    def verify(self,seal,allow_failure=False):
        require(type(seal) is Seal and seal.directory==self.directory,'seal identity')
        marker=self.io.read_small(self.marker,4096)
        require(sha(marker)==seal.marker_sha256,'completion marker changed')
        expected={'schema':'sealed-native-receipt-v1','receipt_sha256':seal.receipt_sha256,
                  'receipt_bytes':seal.receipt_bytes,'native_complete':seal.native_complete}
        require(strict_json(marker)==expected,'completion marker binding')
        require(self.io.read_hash(self.path,MAX_RECEIPT)==
                {'sha256':seal.receipt_sha256,'bytes':seal.receipt_bytes},'sealed receipt changed')
        require(allow_failure or seal.native_complete,'incomplete native observation barrier')

    def posthoc(self,seal,loader):
        require(self.sealed is not None and self.sealed is seal,'unsealed metadata barrier')
        self.verify(seal)
        return loader()  # only after independently verified completion marker


@dataclass(frozen=True)
class Runtime:
    predict: object
    targets: tuple
    provenance: dict

    def attest(self):
        observed={'prediction':callable_identity(self.predict)}
        observed.update({stage:callable_identity(getattr(module,name)) for module,name,stage in self.targets})
        require(observed==self.provenance['callables'],'runtime callable changed')


class CallableIdentityEncoder:
    """No foreign repr, getters or calls. Unsupported/cyclic values fail closed."""
    def __init__(self):
        self.reserved=0; self.nodes=0; self.active=set()
        self.code_bytes=0; self.code_items=0
        self.references={}

    def code_digest(self,code):
        # Bound recursive immutable code constants before marshal allocation.
        pending=[code]
        while pending:
            value=pending.pop(); kind=type(value)
            self.code_items+=1
            require(self.code_items<=4096,'callable code item cap')
            cost=128
            if kind is CodeType:
                children=(value.co_code,value.co_consts,value.co_names,value.co_varnames,
                          value.co_freevars,value.co_cellvars,value.co_filename,value.co_name,
                          getattr(value,'co_qualname',''),value.co_lnotab,
                          getattr(value,'co_linetable',b''),getattr(value,'co_exceptiontable',b''))
            elif kind in (tuple,frozenset): children=value
            elif kind is str: cost+=4*len(value); children=()
            elif kind is bytes: cost+=len(value); children=()
            elif kind is int: cost+=2*((value.bit_length()+7)//8); children=()
            elif value is None or value is Ellipsis or kind in (bool,float,complex): children=()
            else: raise IntegrityError('unsupported callable code constant')
            self.code_bytes+=cost
            require(self.code_bytes<=MAX_SOURCE_BYTES,'callable code byte precharge')
            require(self.code_items+len(pending)+len(children)<=4096,'callable code item cap')
            pending.extend(children)
        return sha(marshal.dumps(code,2))

    def charge(self,amount=64):
        require(self.reserved+amount<=16_384,'callable provenance precharge')
        require(self.nodes<512,'callable provenance item cap')
        self.reserved+=amount; self.nodes+=1

    def take(self,value,depth=0):
        require(depth<=16,'callable provenance depth')
        self.charge()
        kind=type(value)
        if value is None or kind is bool: return value
        if kind is int:
            require(value.bit_length()<=128,'callable integer cap'); return {'int':value}
        if kind is float: return {'float_hex':value.hex()}
        if kind is str:
            require(len(value)<=MAX_STRING,'callable string cap')
            self.charge(12*len(value)); return value
        if kind is bytes:
            require(len(value)<=16_384,'callable bytes cap')
            self.charge(256); return {'type':'bytes','bytes':len(value),'sha256':sha(value)}
        require(kind in (tuple,list,dict) or inspect.isfunction(value),'unsupported callable identity value')
        require(id(value) not in self.active,'callable identity cycle')
        if id(value) in self.references: return {'reference':self.references[id(value)]}
        self.charge(64)
        node=len(self.references)  # deterministic traversal label; never an address
        self.references[id(value)]=node
        self.active.add(id(value))
        try:
            if inspect.isfunction(value):
                require(depth==0 or value.__closure__ is None,'unsupported default callable closure')
                return {'identity_schema':'bounded-defaults-v2','node':node,
                        'module':self.take(value.__module__,depth+1),
                        'qualname':self.take(value.__qualname__,depth+1),
                        'code_encoding':'marshal-v2',
                        'code_sha256':self.code_digest(value.__code__),
                        'defaults':self.take(value.__defaults__,depth+1),
                        'kwdefaults':self.take(value.__kwdefaults__,depth+1),
                        'annotations':self.take(value.__annotations__,depth+1),
                        'globals_module':self.take(value.__globals__.get('__name__'),depth+1)}
            self.charge(64*len(value))  # before iteration or list allocation
            if kind is dict:
                require(all(type(key) is str for key in value),'callable mapping key type')
                # Preserve insertion order: a default dictionary's iteration can
                # affect behavior; do not normalize away an ordering mutation.
                return {'type':'dict','node':node,'items':[[self.take(key,depth+1),self.take(item,depth+1)]
                                              for key,item in value.items()]}
            return {'type':kind.__name__,'node':node,'items':[self.take(item,depth+1) for item in value]}
        finally:
            self.active.remove(id(value))


def callable_identity(fn):
    require(inspect.isfunction(fn),'native callable type')
    result=CallableIdentityEncoder().take(fn)
    require(encoded_size(result)<=16_384,'callable provenance cap')
    return result


def load_runtime():
    """Future GO only. Imports code/dependencies; no fixture access or inference."""
    production=importlib.import_module('mc_photogrammetry.pipeline.image_derived_production')
    entry=production.entry_mod
    joint=importlib.import_module('mc_photogrammetry.pipeline.joint_matching')
    sessions=importlib.import_module('mc_photogrammetry.pipeline.image_derived_session')
    planes=importlib.import_module('mc_photogrammetry.reconstruction.plane_graph')
    targets=((entry,'_discover','discovery'),(entry,'load_image_bytes','decode'),
             (sessions.image_only,'solve_image_cameras','auto_camera'),
             (entry,'build_image_derived_session','session'),(entry,'match_and_solve_pair','pair'),
             (joint,'detect_features','features'),(joint,'reciprocal_topk_graph','reciprocal_graph'),
             (joint,'_score_modes','modes'),(entry.connections_mod,'build_image_derived_connections','connections'),
             (entry.pose_graph_mod,'compose_image_derived_pose_graph','pose_graph'),
             (production.lattice_mod,'measure_image_derived_lattice','lattice'),
             (production.geometry_mod,'derive_image_derived_geometry','geometry'),
             (planes,'match_plane_graph','G_matcher'),(planes,'_apply_same_axis_filter','G_prefilter'),
             (planes,'solve_component','G_solver'),(production.output_mod,'derive_image_derived_output','output'))
    np=importlib.import_module('numpy'); cv2=importlib.import_module('cv2'); pil=importlib.import_module('PIL')
    modules={}; binaries={}; package_sources={}
    for name,module in sorted(tuple(sys.modules.items())):
        origin=getattr(module,'__file__',None)
        if not origin: continue
        if name.startswith(('mc_photogrammetry','numpy','cv2','PIL')):
            path=Path(origin).resolve(); modules[name]={'origin':str(path),**file_hash(path,MAX_FILE_BYTES)}
    for package in ('numpy','opencv-python','Pillow'):
        distribution=importlib.metadata.distribution(package)
        for name in distribution.files or ():
            if str(name).lower().endswith(('.dll','.pyd','.so','.dylib')):
                path=Path(distribution.locate_file(name)).resolve()
                binaries[str(path)]=file_hash(path,MAX_BINARY_BYTES)
            elif str(name).lower().endswith('.py'):
                path=Path(distribution.locate_file(name)).resolve()
                package_sources[str(path)]=file_hash(path,MAX_SOURCE_BYTES)
    python_binary=Path(sys.executable).resolve()
    binaries[str(python_binary)]=file_hash(python_binary,MAX_BINARY_BYTES)
    for path in Path(sys.base_prefix).glob('*.dll'):
        binaries[str(path.resolve())]=file_hash(path.resolve(),MAX_BINARY_BYTES)
    provenance={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
                'flags':repr(sys.flags),'versions':{'numpy':np.__version__,'opencv':cv2.__version__,'Pillow':pil.__version__},
                'modules':modules,'binaries':binaries,'package_python_sources':package_sources,
                'python_base_prefix':sys.base_prefix,'numpy_blas':np.show_config(mode='dicts'),
                'opencv_build':cv2.getBuildInformation(),'opencv_threads':cv2.getNumThreads(),
                'thread_environment':{k:os.environ.get(k) for k in
                    ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
                'launch_environment':{k:os.environ.get(k) for k in ('PYTHONPATH','PYTHONDONTWRITEBYTECODE','PYTHONHASHSEED')},
                'callables':dict({'prediction':callable_identity(production.derive_image_output_from_paths)},
                    **{stage:callable_identity(getattr(module,name)) for module,name,stage in targets})}
    require(encoded_size(provenance)<512*1024,'runtime provenance cap')
    return Runtime(production.derive_image_output_from_paths,targets,provenance)


def load_posthoc_metadata(io):
    """Called exclusively behind Writer.posthoc. No JAR/world/reference access."""
    records={}
    for path,expected in METADATA.items():
        raw=io.read_small(ROOT/path,MAX_METADATA)
        require(sha(raw)==expected,'posthoc metadata pin')
        records[path]=strict_json(raw)
    index,capture=records[BASE+'/index.json'],records[SCENE+'/capture.json']
    scene=index['scenes'][0]
    require(scene['directory']=='scenes/canal-town-architectural-component-bbox','posthoc scene')
    require(scene['hashes']['capture.json']==METADATA[SCENE+'/capture.json'],'posthoc capture binding')
    comparisons=[]
    for slot,ordinal in enumerate((5,6)):
        frame=capture['frames'][ordinal]; screenshot=scene['screenshots'][ordinal]
        require(frame['filename']==screenshot['filename']==Path(INPUTS[slot]).name,'posthoc filename')
        require(frame['frame_id']==screenshot['frame_id'] and
                screenshot['sha256']==screenshot['source_sha256']==frame['actual']['screenshot_sha256']==INPUT_HASHES[slot],
                'posthoc encoded binding')
        comparisons.append({'pipeline_frame':f'image-{slot:04d}','capture_frame_id':frame['frame_id'],
            'recorded_actual':frame['actual'],'recorded_camera':capture['cameras'][ordinal]})
    budget=Budget(); result=Snapshot(budget).take(comparisons)
    require(encoded_size(result)<=1024*1024,'posthoc report cap')
    return {'scope':'posthoc camera comparison inputs only','records':result,
            'accuracy_metrics':'NOT_COMPUTED; camera/G admission attribution only; no visible geometry/state metric'}


def run_once(runtime,freeze,command,io,parent,name,manifest,metadata_loader=None):
    """An invocation owns an exclusive durable reservation; no reuse/retry."""
    freeze.validate(manifest,runtime.provenance,command)
    freeze.validate(source_manifest(),runtime.provenance,command)
    runtime.attest()
    require(len(manifest)==183,'manifest coverage')
    require(encoded_size({'manifest':manifest,'runtime':runtime.provenance})<=1024*1024,'provenance precharge')
    writer=Writer(io,parent,name)  # exclusive/preallocated before any prediction
    observer=Observer(); terminal=None; observed_inputs=[]; snapshot_phase=False; coverage=None
    # Reserve failure/envelope bookkeeping before any upstream evidence. The
    # remainder of this pool remains inaccessible to upstream and G captures.
    with observer.budget.scope('failure'): observer.budget.charge(64*1024,256)
    try:
        for path,expected in zip(INPUTS,INPUT_HASHES):
            receipt=io.read_hash(ROOT/path,MAX_FILE_BYTES)
            require(receipt['sha256']==expected,'encoded input pin')
            observed_inputs.append({'path':path,**receipt})
        with observer.install(runtime.targets):
            outcome=runtime.predict([ROOT/path for path in INPUTS],project_path=writer.project)  # sole call
        snapshot_phase=True
        with observer.budget.scope('terminal'):
            terminal={'kind':'returned','record':observer.projection.take(outcome)}
            coverage=observer.projection.validate((observer.stages,observer.trace_seen,terminal))
        if type(outcome).__name__=='PlacementRefusal': terminal['kind']='bare_placement_refusal'
    except BaseException as error:
        if snapshot_phase or isinstance(error,IntegrityError): observer.failure=error_record(error)
        terminal={'kind':'instrumentation_failure' if observer.failure else 'raised',
                  'exception':error_record(error)}
    try:
        runtime.attest()
        freeze.validate(source_manifest(),runtime.provenance,command)
        observer.budget.charge(0,0)
    except BaseException as error:
        observer.failure=error_record(error)
        terminal={'kind':'instrumentation_failure','exception':error_record(error),'prior_terminal':terminal}
    native_complete=observer.failure is None and terminal['kind'] in ('returned','bare_placement_refusal','raised')
    if len(observed_inputs)!=2: native_complete=False
    published=None
    try:
        if io.exists(writer.project): published=io.read_hash(writer.project,MAX_FILE_BYTES)
    except BaseException as error:
        observer.failure=error_record(error); native_complete=False
    with observer.budget.scope('failure'):
        try:
            if coverage is None: coverage=observer.projection.validate((observer.stages,observer.trace_seen,terminal))
        except BaseException as error:
            observer.failure=error_record(error); native_complete=False
    require(sum(x['items'] for x in observer.budget.stage_counts.values())==observer.budget.items and
            sum(x['bytes'] for x in observer.budget.stage_counts.values())==observer.budget.reserved,'stage ledger binding')
    require(sum(x['items'] for x in observer.budget.pool_counts.values())==observer.budget.items and
            sum(x['bytes'] for x in observer.budget.pool_counts.values())==observer.budget.reserved,'pool ledger binding')
    report={'schema':'sealed-image-only-camera-G-v2','scope':'one fixed image-only fixture; diagnostic only',
            'freeze':{f.name:getattr(freeze,f.name) for f in fields(freeze)},'source_manifest':manifest,
            'runtime':runtime.provenance,'command':command,'ordered_inputs':observed_inputs,'caps':CAPS,
            'actual_decoder_inputs':observer.decoded_inputs,
            'terminal':terminal,'stages':observer.stages,'G_and_mode_locals':observer.trace_seen,
            'semantic_definitions':observer.projection.definitions,'projection_coverage':coverage,
            'allocation_plan':ALLOCATION_PLAN,'allocation_pool_counts':observer.budget.pool_counts,
            'stage_budget_counts':observer.budget.stage_counts,
            'instrumentation_failure':observer.failure,'published_project':published,
            'observer_reserved_bytes':observer.budget.reserved,'observer_items':observer.budget.items,
            'seconds':observer.budget.clock()-observer.budget.started,
            'RSS':'UNMEASURED','watchdog':'Python boundaries; cannot preempt native/I/O calls',
            'coverage_limits':'Camera/G terminal admission attribution. Missing stages are not-reached, never successes. '
                'No visible geometry F1/exact state accuracy; no export; native memory/time hard bound unverified.'}
    try:
        seal=writer.seal(report,native_complete)
    except BaseException as error:
        # Partial native evidence/reservation stays; no completion/no metadata.
        try: writer.stream.close()
        except BaseException: pass
        return {'state':'UNSEALED','directory':str(writer.directory),'failure':error_record(error),
                'metadata_access':False}
    result={'state':'SEALED' if native_complete else 'SEALED_INSTRUMENTATION_FAILURE',
            'directory':str(seal.directory),'receipt_sha256':seal.receipt_sha256,'receipt_bytes':seal.receipt_bytes,
            'marker_sha256':seal.marker_sha256,'terminal':terminal['kind'],'metadata_access':False}
    if metadata_loader is not None and native_complete:
        result['posthoc']=writer.posthoc(seal,metadata_loader); result['metadata_access']=True
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-image-only',action='store_true')
    parser.add_argument('--freeze-stdin',action='store_true')
    parser.add_argument('--output-parent',type=Path,required=True)
    parser.add_argument('--run-name',required=True)
    parser.add_argument('--posthoc',action='store_true')
    args=parser.parse_args()
    require(args.execute_image_only and args.freeze_stdin,'separate image-run GO required')
    freeze=Freeze.read(sys.stdin.buffer)
    manifest=source_manifest()
    runtime=load_runtime()
    command=(sys.executable,str(Path(__file__).resolve()),*sys.argv[1:])
    io=RealIO()
    result=run_once(runtime,freeze,command,io,args.output_parent,args.run_name,manifest,
                    (lambda:load_posthoc_metadata(io)) if args.posthoc else None)
    require(encoded_size(result)<=1024*1024,'summary cap')
    sys.stdout.buffer.write(canonical(result)+b'\n')
