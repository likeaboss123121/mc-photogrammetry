"""Synthetic-only bounded paired evidence; no image ingress or runtime driver."""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import struct
import weakref

import numpy as np
from scripts.evaluation import spatial_context_descriptor as scd

SCHEMA = "spatial-context-paired-evidence-v2"
TABLE_SCHEMA = "spatial-context-tile-table-v1"
TABLE_DOMAIN = b"spatial-context-tile-table-v1\x00"
EVIDENCE_CAP = 16777216
AGGREGATE_CAP = 37748736
SMALL_CAP = 65536
ROW_CAPS = {"site":128, "directed":256, "pair":192}
SECTION_CAPS = {"source":262144, "summary":131072, "failure":65536, "framing":65536}
MAX_MODEL = 10092544
MAX_PUBLICATION = 20381696
ACCEPTED_SOURCE_SHA256 = "066147b78b35ff4a9da84a433463394b6fb711431fe902135e34c6c8be056c42"
CONTRACT_SHA256 = "c14d8ce72fc7131ea6b17b7c26df735a9475d2bec5eacd9b6e04bcf0f9391bcb"
DEPTH_CAP = 16
TOKEN_CAP = 262144
STRING_CAP = 1024
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX16 = re.compile(r"[0-9a-f]{16}\Z")
RUNTIME_KEYS = frozenset(("python", "numpy", "platform", "implementation", "python_sha256", "numpy_sha256", "platform_sha256", "implementation_sha256"))
TREATMENT = {"contract":scd.CONTRACT_VERSION, "spacing":16, "offset":8, "window":48,
             "arms":["A","B"], "ratio":0.75, "containment":"whole-source-and-usable-window"}
STATUSES = frozenset(("insufficient_support", "candidate_discrimination_witness",
    "witness_budget_inconclusive", "reciprocal_pairs_without_common_witness", "no_admitted_pairs"))
NEAREST_STATUSES = frozenset(("absent_second_neighbor", "nearest_distance_tie", "ratio_rejected", "ratio_accepted"))

Refusal = scd.Refusal
canonical = scd.canonical
sha = scd.sha


def require(condition, code):
    if not condition:
        raise Refusal(code)


def exact_keys(value, keys):
    require(type(value) is dict and set(value)==set(keys), "object_keys")


def integer(value, low, high):
    require(type(value) is int and low<=value<=high, "integer_bound")
    return value


def digest(value):
    require(type(value) is str and HEX64.fullmatch(value), "digest_type")
    return value


def byte_guard(count, cap, code):
    integer(count, 0, 2**63-1)
    require(count<=cap, code)


def row_size(row, kind):
    # Includes the separator reserved even for the last list member.
    require(type(row) is list and len(row)<=6,"row_shape")
    require(all(type(v) in (str,int,bool) or v is None for v in row),"row_value")
    size=_bounded_object(row,ROW_CAPS[kind]-1)+1
    byte_guard(size, ROW_CAPS[kind], "row_byte_limit")
    return size


def size_model(left=256, right=4096):
    scd.check_site_caps(left, right)
    count=left+right
    evidence=1536*count+128*count+512*count+384*min(left,right)+524288
    return {"evidence":evidence, "aggregate":2*evidence+3*SMALL_CAP,
            "hard_evidence":EVIDENCE_CAP, "hard_aggregate":AGGREGATE_CAP}


def bits(value):
    require(math.isfinite(float(value)) and float(value)>=0, "distance_value")
    return struct.pack("<d", float(value)).hex()


def distance(value):
    require(type(value) is str and HEX16.fullmatch(value), "distance_bits")
    result=struct.unpack("<d", bytes.fromhex(value))[0]
    require(math.isfinite(result) and result>=0, "distance_value")
    return result


def table_header(count):
    return {"schema":TABLE_SCHEMA, "dtype":"<f4", "order":"C", "shape":[count,9,32], "raw_nbytes":count*1152}


def encode_table(array, role):
    integer(role,0,1)
    cap=(256,4096)[role]
    require(isinstance(array,np.ndarray) and array.dtype==scd.FLOAT32_LE, "table_dtype")
    require(array.ndim==3 and array.shape[1:]==(9,32) and array.shape[0]<=cap, "table_shape")
    require(array.flags.c_contiguous, "table_order")
    require(np.isfinite(array).all(), "table_nonfinite")
    header=table_header(array.shape[0]);raw=array.tobytes(order="C")
    return {**header, "raw_sha256":sha(raw), "table_sha256":sha(TABLE_DOMAIN+canonical(header)+b"\x00"+raw),
            "data":base64.b64encode(raw).decode("ascii")}


def validate_table_header(record, role):
    integer(role,0,1)
    exact_keys(record, (*table_header(0),"raw_sha256","table_sha256","data"))
    shape=record["shape"]
    require(type(shape) is list and len(shape)==3, "table_shape")
    count=integer(shape[0],0,(256,4096)[role])
    require(all(type(x) is int for x in shape) and shape[1:]==[9,32], "table_shape")
    require({k:record[k] for k in table_header(0)}==table_header(count), "table_header")
    integer(record["raw_nbytes"],0,(256,4096)[role]*1152)
    digest(record["raw_sha256"]);digest(record["table_sha256"])
    require(type(record["data"]) is str and len(record["data"])==1536*count, "table_encoded_length")
    return count


def decode_table(record, role):
    count=validate_table_header(record,role)
    try:
        raw=base64.b64decode(record["data"],validate=True)
    except (ValueError, UnicodeError):
        raise Refusal("table_base64") from None
    require(len(raw)==count*1152 and base64.b64encode(raw).decode("ascii")==record["data"], "table_base64_canonical")
    header=table_header(count)
    require(sha(raw)==record["raw_sha256"] and sha(TABLE_DOMAIN+canonical(header)+b"\x00"+raw)==record["table_sha256"], "table_digest")
    result=np.frombuffer(raw,dtype=scd.FLOAT32_LE).reshape(count,9,32)
    require(np.isfinite(result).all(), "table_nonfinite")
    return result


class _Lexical:
    """Capped grammar scan: no long string copies, no decoded table allocation."""
    def __init__(self, data):
        require(type(data) is bytes,"wire_bytes")
        byte_guard(len(data),EVIDENCE_CAP,"evidence_byte_limit")
        require(data.isascii(), "wire_ascii")
        self.data=data;self.index=0;self.tokens=0;self.long_fields=set()

    def token(self):
        self.tokens+=1
        require(self.tokens<=TOKEN_CAP,"token_limit")

    def space(self):
        while self.index<len(self.data) and self.data[self.index] in b" \t\r\n":self.index+=1

    def string(self,path,key=False):
        self.token();start=self.index;self.index+=1;escaped=False
        while self.index<len(self.data):
            c=self.data[self.index]
            if c==34 and not escaped:
                self.index+=1;break
            require(c>=32,"string_control")
            if escaped:escaped=False
            elif c==92:escaped=True
            self.index+=1
        else:raise Refusal("string_unterminated")
        length=self.index-start-2
        special=path in (("frames",0,"table","data"),("frames",1,"table","data")) and not key
        if special:
            require(path not in self.long_fields,"duplicate_table_field")
            self.long_fields.add(path)
            byte_guard(length,(256,4096)[path[1]]*1536,"table_encoded_length")
            require(b"\\" not in self.data[start:self.index],"base64_wire_escape")
            return None
        byte_guard(length,STRING_CAP,"string_limit")
        try:return json.loads(self.data[start:self.index])
        except (ValueError,UnicodeError):raise Refusal("string_syntax") from None

    def value(self,path=(),depth=0):
        require(depth<=DEPTH_CAP,"depth_limit");self.space()
        require(self.index<len(self.data),"wire_incomplete")
        c=self.data[self.index]
        if c==34:self.string(path);return
        if c in (123,91):
            self.token();self.index+=1;self.space();end=125 if c==123 else 93
            if self.index<len(self.data) and self.data[self.index]==end:self.token();self.index+=1;return
            item=0;seen=set()
            while True:
                self.space()
                if c==123:
                    require(self.index<len(self.data) and self.data[self.index]==34,"object_key")
                    key=self.string(path,key=True);require(key not in seen,"duplicate_json_key");seen.add(key)
                    self.space();require(self.index<len(self.data) and self.data[self.index]==58,"object_colon")
                    self.token();self.index+=1;child=path+(key,)
                else:child=path+(item,)
                self.value(child,depth+1);item+=1;self.space()
                require(self.index<len(self.data),"wire_incomplete");delimiter=self.data[self.index]
                require(delimiter in (44,end),"container_separator");self.token();self.index+=1
                if delimiter==end:return
        self.token();start=self.index
        while self.index<len(self.data) and self.data[self.index] not in b",]} \t\r\n":self.index+=1
        text=self.data[start:self.index]
        if text in (b"true",b"false",b"null"):return
        require(re.fullmatch(rb"-?(0|[1-9][0-9]*)(\.[0-9]+)?",text) is not None,"number_syntax")
        require(len(text.lstrip(b"-").split(b".")[0])<=20,"integer_token_limit")
        require(len(text)<=40,"number_token_limit")

    def scan(self):
        self.value();self.space();require(self.index==len(self.data),"wire_trailing")


def _unique(pairs):
    result={}
    for key,value in pairs:
        require(key not in result,"duplicate_json_key");result[key]=value
    return result


def parse(data):
    _Lexical(data).scan()
    try:
        value=json.loads(data,object_pairs_hook=_unique,parse_constant=lambda _: (_ for _ in ()).throw(Refusal("nonfinite_json")))
    except (ValueError, UnicodeError, RecursionError):raise Refusal("json_syntax") from None
    require(canonical(value)+b"\n"==data,"wire_noncanonical")
    return value


@dataclass(frozen=True)
class SharedFrame:
    plan: scd.DomainPlan
    tiles: np.ndarray
    central_zero: np.ndarray
    neighbor_zero_counts: np.ndarray
    frozen_sha256: str
    source_refs: tuple | None = None


def _source_digest(pixels,source,usable):
    return tuple(sha(a.tobytes(order="C")) for a in (pixels,source,usable))


PREPARED_FRAME_VERSION = "spatial-context-prepared-frame-v1"
_PREPARED_KEY = object()
_PREPARED = weakref.WeakKeyDictionary()


class _PreparedFrame:
    """Opaque local handle; provenance is held separately, never on the wire."""
    __slots__ = ("_version", "_role", "_plan", "_source_refs", "__weakref__")

    def __init__(self,key,role,plan,refs):
        require(key is _PREPARED_KEY,"prepared_receipt_authenticity")
        self._version=PREPARED_FRAME_VERSION;self._role=role
        self._plan=plan;self._source_refs=refs

    @property
    def version(self):return self._version

    @property
    def role(self):return self._role

    @property
    def plan(self):return self._plan

    def __reduce_ex__(self,protocol):
        raise Refusal("prepared_receipt_serialization")


@dataclass(frozen=True)
class _PreparedState:
    version: str
    process_id: int
    role: int
    plan: scd.DomainPlan
    plan_content: tuple
    source_refs: tuple
    source_digests: tuple


def _plan_content(plan,role):
    # Check cardinalities before traversing metadata; no alternate planner.
    require(type(plan) is scd.DomainPlan,"prepared_plan_type")
    require(type(plan.sites) is tuple,"prepared_plan_sites")
    require(len(plan.sites)<=(256,4096)[role],"shared_frame_site_limit")
    require(type(plan.eligible_count) is int and plan.eligible_count==len(plan.sites),"prepared_plan_count")
    require(plan.contract_version==scd.CONTRACT_VERSION,"prepared_plan_contract")
    require(type(plan.shape) is tuple and len(plan.shape)==2
            and all(type(v) is int and 0<=v<=cap for v,cap in zip(plan.shape,(900,1600))),"prepared_plan_shape")
    expected_grid=len(range(scd.OFFSET,plan.shape[0],scd.SPACING))*len(range(scd.OFFSET,plan.shape[1],scd.SPACING))
    require(type(plan.grid_centers) is int and plan.grid_centers==expected_grid,"prepared_plan_grid")
    digests=(plan.frame_sha256,plan.source_mask_sha256,plan.usable_mask_sha256)
    for value in digests:digest(value)
    sites=[];previous=None
    for site in plan.sites:
        require(type(site) is scd.Site,"prepared_site_type")
        digest(site.site_id)
        require(type(site.x) is int and type(site.y) is int
                and site.x%scd.SPACING==scd.OFFSET and site.y%scd.SPACING==scd.OFFSET
                and scd._window_in_bounds(site.x,site.y,plan.shape),"prepared_site_bounds")
        point=(site.y,site.x)
        require(previous is None or previous<point,"prepared_site_order")
        previous=point;sites.append((site.site_id,site.x,site.y))
    return (plan.contract_version,plan.shape,*digests,plan.grid_centers,plan.eligible_count,tuple(sites))


def prepare_shared_frame(pixels,source,usable,role=0):
    """Plan once without converting pixels or building any tile/descriptor."""
    integer(role,0,1)
    height,width=scd.validate_rgb(pixels)
    require(height<=900 and width<=1600,"frame_pixel_limit")
    plan=scd.plan_sites(pixels,source,usable)
    before=(plan.frame_sha256,plan.source_mask_sha256,plan.usable_mask_sha256)
    require(_source_digest(pixels,source,usable)==before,"input_mutation")
    content=_plan_content(plan,role)
    require(plan.shape==(height,width),"prepared_plan_shape")
    refs=(pixels,source,usable)
    receipt=_PreparedFrame(_PREPARED_KEY,role,plan,refs)
    _PREPARED[receipt]=_PreparedState(PREPARED_FRAME_VERSION,os.getpid(),role,plan,content,refs,before)
    return receipt


def extract_prepared_frame(receipt):
    """Consume an authentic unchanged local receipt; never repeat planning."""
    require(type(receipt) is _PreparedFrame,"prepared_receipt_authenticity")
    state=_PREPARED.pop(receipt,None)
    require(state is not None,"prepared_receipt_unavailable")
    # Consume before validation/extraction so even a failed attempt is terminal.
    require(os.getpid()==state.process_id,"prepared_receipt_process")
    require(receipt._version==state.version,"prepared_receipt_version")
    require(type(receipt._role) is int and receipt._role==state.role,"prepared_receipt_role")
    require(receipt._plan is state.plan,"prepared_receipt_plan")
    refs=receipt._source_refs
    require(type(refs) is tuple and len(refs)==3
            and all(a is b for a,b in zip(refs,state.source_refs)),"prepared_receipt_inputs")
    require(_plan_content(state.plan,state.role)==state.plan_content,"prepared_plan_mutation")
    pixels,source,usable=state.source_refs
    shape=scd.validate_rgb(pixels)
    require(shape==state.plan.shape,"input_mutation")
    scd.validate_mask(source,shape,"source_mask");scd.validate_mask(usable,shape,"usable_mask")
    before=state.source_digests
    require(_source_digest(pixels,source,usable)==before,"input_mutation")
    plan=state.plan
    luma=scd.luma_from_rgb(pixels)
    table=np.empty((len(plan.sites),9,32),scd.FLOAT32_LE)
    central=np.empty(len(plan.sites),bool);neighbors=np.empty(len(plan.sites),np.int64)
    for index,site in enumerate(plan.sites):
        table[index],flags=scd.tile_table(luma,site.x,site.y)
        central[index]=flags[4];neighbors[index]=sum(flags[:4])+sum(flags[5:])
    require(_source_digest(pixels,source,usable)==before,"input_mutation")
    require(_plan_content(plan,state.role)==state.plan_content,"prepared_plan_mutation")
    for array in (table,central,neighbors):array.setflags(write=False)
    return SharedFrame(plan,table,central,neighbors,sha(table.tobytes(order="C")),(pixels,source,usable))


def shared_frame(pixels,source,usable,role=0):
    return extract_prepared_frame(prepare_shared_frame(pixels,source,usable,role))


def derive(frame,arm):
    scd.dimension_for_arm(arm)
    if frame.source_refs is not None:
        require(_source_digest(*frame.source_refs)==(frame.plan.frame_sha256,frame.plan.source_mask_sha256,frame.plan.usable_mask_sha256),"input_mutation")
    require(sha(frame.tiles.tobytes(order="C"))==frame.frozen_sha256,"table_mutation")
    descriptors=frame.tiles[:,4,:] if arm=="A" else np.asarray([scd.build_context_b(t) for t in frame.tiles],dtype=scd.FLOAT32_LE).reshape(-1,288)
    scd.dimension_for_arm(arm);descriptors.setflags(write=False)
    return scd.ArmDescriptors(arm,frame.plan.sites,descriptors,frame.central_zero,frame.neighbor_zero_counts,
                              frame.tiles,frame.plan.frame_sha256,frame.plan.shape)


def effective_hash(frame,arm,array):
    return sha(canonical({"schema":"spatial-context-effective-descriptor-v1","contract":scd.CONTRACT_VERSION,
        "arm":arm,"shape":list(array.shape),"table_sha256":frame.frozen_sha256,
        "site_order_sha256":sha(canonical([s.site_id for s in frame.plan.sites])),
        "frame_sha256":frame.plan.frame_sha256,"source_mask_sha256":frame.plan.source_mask_sha256,
        "usable_mask_sha256":frame.plan.usable_mask_sha256})+b"\x00"+array.tobytes(order="C"))


def _bin(x,y,shape):
    return (y*4//shape[0])*4+x*4//shape[1]


def frame_record(frame,role):
    table=encode_table(frame.tiles,role)
    rows=[[s.site_id,s.x,s.y,bool(frame.central_zero[i]),int(frame.neighbor_zero_counts[i]),_bin(s.x,s.y,frame.plan.shape)]
          for i,s in enumerate(frame.plan.sites)]
    for row in rows:row_size(row,"site")
    return {"shape":list(frame.plan.shape),"frame_sha256":frame.plan.frame_sha256,
        "source_mask_sha256":frame.plan.source_mask_sha256,"usable_mask_sha256":frame.plan.usable_mask_sha256,
        "sites":rows,"table":table}


def _compact_support(arm):
    indices=list(map(int,arm.shared_indices));ids=[arm.sites[i].site_id for i in indices]
    xy=[arm.sites[i].xy for i in indices];block=scd._support_block(ids,xy,arm.shape)
    lookup={site.site_id:i for i,site in enumerate(arm.sites)}
    return {"count":len(ids),"bins":[len(b) for b in block["bins"]],
        "bbox":[min(p[0] for p in xy),min(p[1] for p in xy),max(p[0] for p in xy),max(p[1] for p in xy)] if xy else [],
        "first":[lookup[i] for i in block["first_four_ids"]],"first_ready":block["first_four_ready"],
        "all_general":block["all_domain_general_position"]}


def compact_match(match,frames,arms):
    lookup=[{s.site_id:i for i,s in enumerate(f.plan.sites)} for f in frames]
    def directed(evidence,origin,target):
        rows=[]
        for entry in evidence:
            rows.append([lookup[origin][entry["origin"]],lookup[target].get(entry.get("nearest"),-1),
                lookup[target].get(entry.get("second"),-1),bits(entry["nearest_squared"]) if "nearest_squared" in entry else None,
                bits(entry["second_squared"]) if "second_squared" in entry else None,entry["status"]])
        return rows
    pairs=[[p["pair_id"],lookup[0][p["left_id"]],lookup[1][p["right_id"]],bits(p["distance_squared"])] for p in match.pairs]
    pairs_lookup={p["pair_id"]:i for i,p in enumerate(match.pairs)}
    witness={"status":match.witness["status"],"pairs":[pairs_lookup[i] for i in match.witness["pair_ids"]],
        "source_binding_sha256":match.witness["source_binding_sha256"],"node_work":match.witness["node_work"],
        "determinant_work":match.witness["determinant_work"]}
    return {"arm":match.arm,"directed_state":"evaluated" if all(len(a.shared_indices) for a in arms) else "skipped_empty_domain",
        "forward":directed(match.forward_evidence,0,1),"reverse":directed(match.reverse_evidence,1,0),
        "pairs":pairs,"summary":{"status":match.status,"forward_counts":match.forward_counts,"reverse_counts":match.reverse_counts,
        "witness":witness,"comparison_work":match.comparison_work,"support":[_compact_support(a) for a in arms],
        "domain_hashes":[sha(canonical([a.sites[int(i)].site_id for i in a.shared_indices])) for a in arms],
        "effective_hashes":[effective_hash(f,match.arm,a.descriptors) for f,a in zip(frames,arms)],
        "distance_matrix_sha256":sha(canonical({"shape":list(match.distance_matrix.shape),"dtype":"<f8"})+b"\x00"+match.distance_matrix.astype("<f8",copy=False).tobytes(order="C"))}}


def paired(frames,bindings,runtime):
    require(len(frames)==2,"frame_count");scd.check_site_caps(*(len(f.plan.sites) for f in frames))
    bound=size_model();byte_guard(bound["evidence"],EVIDENCE_CAP,"evidence_byte_limit")
    byte_guard(bound["aggregate"],AGGREGATE_CAP,"aggregate_byte_limit")
    _bounded_object({"bindings":bindings,"runtime":runtime,"treatment":TREATMENT},SECTION_CAPS["source"])
    arms=[];binding=scd.source_binding(*(f.plan for f in frames))
    for name in ("A","B"):
        descriptors=[derive(f,name) for f in frames]
        match=scd.match_arms(*descriptors,source_binding_sha256=binding)
        arms.append(compact_match(match,frames,descriptors))
        # One matrix live at a time. Neither matrix is serialized.
        del match,descriptors
        for frame in frames:require(sha(frame.tiles.tobytes(order="C"))==frame.frozen_sha256,"table_mutation")
    body={"schema":SCHEMA,"treatment":TREATMENT,"bindings":bindings,"runtime":runtime,"size_bound":size_model(),
        "frames":[frame_record(f,i) for i,f in enumerate(frames)],"arms":arms,
        "outcomes":[a["summary"]["status"] for a in arms],"counters":{"storage_replay":"no_matching_or_common_search"},
        "failure":{"code":"none"}}
    return seal(body)


def _rows_guard(rows,kind,cap):
    require(type(rows) is list and len(rows)<=cap,"row_count")
    for row in rows:
        require(type(row) is list,"row_type");row_size(row,kind)


def _bounded_object(value,cap):
    """Exact capped scalar/punctuation count before serializing a whole section."""
    total=0;tokens=0
    def charge(count):
        nonlocal total
        total+=count;byte_guard(total,cap,"section_byte_limit")
    def visit(item,depth=0):
        nonlocal tokens
        tokens+=1;require(tokens<=TOKEN_CAP,"token_limit");require(depth<=DEPTH_CAP,"depth_limit")
        if type(item) is dict:
            require(len(item)<=256,"object_field_limit");charge(2+max(0,len(item)-1))
            for key,child in item.items():
                require(type(key) is str and len(key)<=1024 and key.isascii(),"object_key")
                charge(len(canonical(key))+1);visit(child,depth+1)
        elif type(item) in (list,tuple):
            require(len(item)<=8704,"object_list_limit");charge(2+max(0,len(item)-1))
            for child in item:visit(child,depth+1)
        else:
            if type(item) is str:require(item.isascii() and len(item)<=STRING_CAP,"string_limit")
            elif type(item) is int:require(len(str(abs(item)))<=20,"integer_token_limit")
            elif type(item) is float:require(math.isfinite(item),"nonfinite_json")
            else:require(item is None or type(item) is bool,"object_value")
            charge(len(canonical(item)))
    visit(value)
    return total


def account(record):
    """Disjoint row/payload terms and four bounded remainder sections."""
    source={k:record[k] for k in ("bindings","runtime","treatment")}
    summary={"arms":[{"directed_state":a["directed_state"],**a["summary"]} for a in record["arms"]],"outcomes":record["outcomes"],"counters":record["counters"]}
    failure=record["failure"]
    source_bytes=_bounded_object(source,SECTION_CAPS["source"])
    summary_bytes=_bounded_object(summary,SECTION_CAPS["summary"])
    failure_bytes=_bounded_object(failure,SECTION_CAPS["failure"])
    blank={**record,"bindings":{},"runtime":{},"treatment":{},"outcomes":[],"counters":{},"failure":{},
        "arms":[{**a,"directed_state":"","forward":[],"reverse":[],"pairs":[],"summary":{}} for a in record["arms"]],
        "frames":[{**f,"sites":[],"table":{**f["table"],"data":""}} for f in record["frames"]]}
    framing_bytes=_bounded_object(blank,SECTION_CAPS["framing"]-1)+1
    sizes={"source":source_bytes,"summary":summary_bytes,"failure":failure_bytes,"framing":framing_bytes}
    for key,count in sizes.items():byte_guard(count,SECTION_CAPS[key],"section_byte_limit")
    rows=sum(row_size(r,"site") for f in record["frames"] for r in f["sites"])
    rows+=sum(row_size(r,"directed") for a in record["arms"] for d in ("forward","reverse") for r in a[d])
    rows+=sum(row_size(r,"pair") for a in record["arms"] for r in a["pairs"])
    payload=sum(len(f["table"]["data"]) for f in record["frames"])
    # Overcounts empty-section punctuation, never undercounts wire bytes.
    bound=payload+rows+sum(sizes.values())
    byte_guard(bound,MAX_MODEL,"model_byte_limit")
    return {"sections":sizes,"row_bytes":rows,"base64_bytes":payload,"accounted_upper_bound":bound}


def seal(body):
    record={**body,"record_sha256":"0"*64}
    structure(record)
    account(record)
    record["record_sha256"]=sha(canonical(body))
    return record


def structure(record):
    exact_keys(record,("schema","treatment","bindings","runtime","size_bound","frames","arms","outcomes","counters","failure","record_sha256"))
    require(record["schema"]==SCHEMA and record["treatment"]==TREATMENT and record["size_bound"]==size_model(),"schema_treatment_bound")
    require(type(record["frames"]) is list and len(record["frames"])==2,"frame_count")
    require(type(record["arms"]) is list and len(record["arms"])==2,"arm_count")
    counts=[]
    for role,frame in enumerate(record["frames"]):
        exact_keys(frame,("shape","frame_sha256","source_mask_sha256","usable_mask_sha256","sites","table"))
        require(type(frame["shape"]) is list and len(frame["shape"])==2,"frame_shape")
        for d in frame["shape"]:integer(d,1,65535)
        for key in ("frame_sha256","source_mask_sha256","usable_mask_sha256"):digest(frame[key])
        count=validate_table_header(frame["table"],role);counts.append(count)
        _rows_guard(frame["sites"],"site",(256,4096)[role]);require(len(frame["sites"])==count,"site_table_count")
        previous=None;seen=set()
        for row in frame["sites"]:
            require(len(row)==6,"site_row_shape");identity,x,y,zero,neighbors,bin_index=row
            digest(identity);integer(x,0,min(65535,frame["shape"][1]-1));integer(y,0,min(65535,frame["shape"][0]-1))
            require(x%16==y%16==8 and scd._window_in_bounds(x,y,tuple(frame["shape"])),"site_grid_footprint")
            require(type(zero) is bool,"site_zero_type");integer(neighbors,0,8);integer(bin_index,0,15)
            require(bin_index==_bin(x,y,frame["shape"]),"site_bin")
            require(identity==scd._site_id(frame["frame_sha256"],frame["source_mask_sha256"],frame["usable_mask_sha256"],x,y),"site_id")
            order=(y,x,identity);require(previous is None or order>previous,"site_order");previous=order
            require(identity not in seen,"duplicate_site");seen.add(identity)
    scd.check_site_caps(*counts)
    for arm,name in zip(record["arms"],("A","B")):
        exact_keys(arm,("arm","directed_state","forward","reverse","pairs","summary"));require(arm["arm"]==name,"arm_order")
        require(arm["directed_state"] in ("evaluated","skipped_empty_domain"),"directed_state")
        _rows_guard(arm["forward"],"directed",counts[0]);_rows_guard(arm["reverse"],"directed",counts[1]);_rows_guard(arm["pairs"],"pair",min(counts))
    digest(record["record_sha256"])


def _work(work,cap):
    exact_keys(work,("limit","requested","charged","processed","uncharged","unprocessed"))
    integer(work["limit"],0,cap)
    for key in ("requested","charged","processed","uncharged","unprocessed"):integer(work[key],0,cap+8)
    require(work["processed"]<=work["charged"]<=work["limit"] and work["requested"]>=work["charged"],"work_order")
    require(work["uncharged"]==work["requested"]-work["charged"] and work["unprocessed"]==work["charged"]-work["processed"],"work_closure")


def quartet_rank(indices, count):
    """Zero-based lexicographic 4-combination position, no search/replay."""
    rank=0;start=0
    for position,index in enumerate(indices):
        remaining=4-position
        rank+=math.comb(count-start,remaining)-math.comb(count-index,remaining)
        start=index+1
    return rank


def _binding(record,frames):
    binding=record["bindings"]
    exact_keys(binding,("sources","source_pins","diagnostic_outer_sha256","diagnostic_nested_sha256","contract_sha256","source_sha256","runtime_sha256","authorization_sha256","binding_sha256"))
    for key in set(binding)-{"sources","source_pins"}:digest(binding[key])
    require(binding["contract_sha256"]==CONTRACT_SHA256 and binding["source_sha256"]==ACCEPTED_SOURCE_SHA256,"source_contract_identity")
    require(binding["binding_sha256"]==sha(canonical({k:v for k,v in binding.items() if k!="binding_sha256"})),"source_binding_digest")
    require(type(binding["source_pins"]) is list and len(binding["source_pins"])<=256,"source_pin_count")
    seen=set()
    for row in binding["source_pins"]:
        require(type(row) is list and len(row)==2 and type(row[0]) is str and row[0].isascii() and len(row[0])<=300,"source_pin_row")
        require(len(canonical(row))+1<=384 and row[0] not in seen,"source_pin_bound");seen.add(row[0]);digest(row[1])
    require(type(record["runtime"]) is dict and set(record["runtime"])<=RUNTIME_KEYS,"runtime_keys")
    for key,value in record["runtime"].items():
        require(type(value) is str and value.isascii() and 0<len(value)<=128,"runtime_value")
        if key.endswith("sha256"):digest(value)
    require(binding["runtime_sha256"]==sha(canonical(record["runtime"])),"runtime_binding")
    require(type(binding["sources"]) is list and len(binding["sources"])==2,"source_count")
    for source,frame in zip(binding["sources"],frames):
        exact_keys(source,("frame_id","encoded_sha256","encoded_length","decoded_shape","decoded_dtype","decoded_rgb_sha256","ingress_sha256","endpoint","region_id","packed_mask_sha256","full_native_bool_sha256","usable_mask_sha256","usable_helper_sha256"))
        require(type(source["frame_id"]) is str and source["frame_id"].isascii() and 0<len(source["frame_id"])<=128,"frame_id")
        integer(source["encoded_length"],0,8388608)
        require(source["decoded_shape"]==[*frame.plan.shape,3] and source["decoded_dtype"]=="|u1","decoded_layout")
        require(type(source["endpoint"]) is list and len(source["endpoint"])==2 and source["endpoint"][0]==source["frame_id"],"endpoint")
        integer(source["endpoint"][1],0,65535)
        for key in ("encoded_sha256","decoded_rgb_sha256","ingress_sha256","region_id","packed_mask_sha256","full_native_bool_sha256","usable_mask_sha256","usable_helper_sha256"):digest(source[key])
        require(source["decoded_rgb_sha256"]==frame.plan.frame_sha256 and source["full_native_bool_sha256"]==frame.plan.source_mask_sha256 and source["usable_mask_sha256"]==frame.plan.usable_mask_sha256,"source_frame_binding")
    require(len({s["frame_id"] for s in binding["sources"]})==2,"source_frame_identity")


def validate(record):
    # Structural/section/site checks on BOTH frames precede either decode.
    structure(record);account(record)
    exact_keys(record["failure"],("code",));require(record["failure"]["code"]=="none","failure_code")
    require(record["counters"]=={"storage_replay":"no_matching_or_common_search"},"storage_replay_label")
    body=dict(record);seal_hash=body.pop("record_sha256");require(sha(canonical(body))==seal_hash,"record_digest")
    # Source/runtime schemas and binding joins precede both payload decodes.
    metadata_frames=[]
    for source in record["frames"]:
        rows=source["sites"]
        plan=scd.DomainPlan(scd.CONTRACT_VERSION,tuple(source["shape"]),source["frame_sha256"],source["source_mask_sha256"],source["usable_mask_sha256"],0,len(rows),tuple(scd.Site(r[0],r[1],r[2]) for r in rows))
        metadata_frames.append(SharedFrame(plan,None,None,None,None))
    _binding(record,metadata_frames)
    frames=[]
    for role,source in enumerate(record["frames"]):
        array=decode_table(source["table"],role);rows=source["sites"]
        central=np.asarray([r[3] for r in rows],bool);neighbors=np.asarray([r[4] for r in rows],np.int64)
        zeros=np.all(array==0,axis=2)
        require(np.array_equal(central,zeros[:,4]) and np.array_equal(neighbors,np.sum(zeros,axis=1)-zeros[:,4]),"zero_metadata")
        plan=scd.DomainPlan(scd.CONTRACT_VERSION,tuple(source["shape"]),source["frame_sha256"],source["source_mask_sha256"],source["usable_mask_sha256"],0,len(rows),tuple(scd.Site(r[0],r[1],r[2]) for r in rows))
        frames.append(SharedFrame(plan,array,central,neighbors,sha(array.tobytes(order="C"))))
    _binding(record,frames)
    binding=scd.source_binding(*(f.plan for f in frames));verified=[]
    for arm,name in zip(record["arms"],("A","B")):
        derived=[derive(f,name) for f in frames];domains=[list(map(int,a.shared_indices)) for a in derived]
        summary=arm["summary"]
        exact_keys(summary,("status","forward_counts","reverse_counts","witness","comparison_work","support","domain_hashes","effective_hashes","distance_matrix_sha256"))
        require(type(summary["status"]) is str and summary["status"] in STATUSES,"arm_status");digest(summary["distance_matrix_sha256"])
        require(summary["support"]==[_compact_support(a) for a in derived],"support_binding")
        for support in summary["support"]:
            exact_keys(support,("count","bins","bbox","first","first_ready","all_general"))
            integer(support["count"],0,4096)
            for value in support["bins"]:integer(value,0,4096)
            for value in support["bbox"]:integer(value,0,65535)
            for value in support["first"]:integer(value,0,4095)
            require(type(support["first_ready"]) is bool and type(support["all_general"]) is bool,"support_boolean")
        require(summary["domain_hashes"]==[sha(canonical([a.sites[i].site_id for i in d])) for a,d in zip(derived,domains)],"domain_binding")
        require(summary["effective_hashes"]==[effective_hash(f,name,a.descriptors) for f,a in zip(frames,derived)],"effective_descriptor_binding")
        skipped=not all(domains)
        require(arm["directed_state"]==("skipped_empty_domain" if skipped else "evaluated"),"directed_state_binding")
        if skipped:
            require(summary["distance_matrix_sha256"]==sha(canonical({"shape":[len(d) for d in domains],"dtype":"<f8"})+b"\x00"),"empty_matrix_binding")
        maps=[];distances=[]
        for rows,origin,target,key in ((arm["forward"],0,1,"forward_counts"),(arm["reverse"],1,0,"reverse_counts")):
            require(len(rows)==(0 if skipped else len(domains[origin])),"directed_domain_count")
            counts={"origins":len(domains[origin]),**{s:0 for s in NEAREST_STATUSES}};admitted={};values={}
            require([r[0] for r in rows]==([] if skipped else domains[origin]),"directed_origin_domain")
            exact_keys(summary[key],counts)
            for count in summary[key].values():integer(count,0,4096)
            for row in rows:
                require(len(row)==6,"directed_row_shape");i,j,k,d0,d1,status=row
                integer(i,0,max(0,len(frames[origin].plan.sites)-1));require(type(status) is str and status in NEAREST_STATUSES,"nearest_status")
                if len(domains[target])<2:
                    require(type(j) is int and type(k) is int and j==k==-1 and d0 is d1 is None and status=="absent_second_neighbor","absent_neighbor")
                else:
                    require(type(j) is int and type(k) is int and j in domains[target] and k in domains[target] and j!=k,"nearest_reference")
                    first,second=distance(d0),distance(d1);require(first<=second,"nearest_order")
                    expected="nearest_distance_tie" if first==second else ("ratio_accepted" if first<scd.RATIO_SQUARED*second else "ratio_rejected")
                    require(status==expected,"nearest_status_binding");values[i]=d0
                    if expected=="ratio_accepted":admitted[i]=j
                counts[status]+=1
            require(summary[key]==counts,"directed_counter_binding");maps.append(admitted);distances.append(values)
        expected={(i,j) for i,j in maps[0].items() if maps[1].get(j)==i};seen=set();pair_objects=[]
        for row in arm["pairs"]:
            require(len(row)==4,"pair_row_shape");identity,i,j,d=row
            digest(identity);require(type(i) is int and type(j) is int and (i,j) in expected and (i,j) not in seen,"reciprocal_join")
            seen.add((i,j));require(d==distances[0][i] and d==distances[1][j],"pair_distance_binding")
            p=scd.make_pair_record(frames[0].plan.sites[i].site_id,frames[1].plan.sites[j].site_id,frames[0].plan.sites[i].xy,frames[1].plan.sites[j].xy,distance(d))
            require(p["pair_id"]==identity,"pair_identity");pair_objects.append(p)
        require(seen==expected and [p["pair_id"] for p in pair_objects]==sorted(p["pair_id"] for p in pair_objects),"pair_domain_order")
        work=summary["comparison_work"];_work(work,scd.COMPARISON_LIMIT)
        require(work["requested"]==work["charged"]==work["processed"]==len(domains[0])*len(domains[1]) and work["uncharged"]==work["unprocessed"]==0,"comparison_work_binding")
        witness=summary["witness"];exact_keys(witness,("status","pairs","source_binding_sha256","node_work","determinant_work"))
        require(witness["source_binding_sha256"]==binding,"witness_source_binding")
        _work(witness["node_work"],4096);_work(witness["determinant_work"],32768)
        nw,dw=witness["node_work"],witness["determinant_work"];require(dw["charged"]%8==0 and dw["uncharged"] in (0,8),"determinant_batch")
        # Reciprocal pairs have unique site endpoints and coordinates, so each
        # completed tuple evaluates all eight determinants (also if collinear).
        require(dw["processed"]==dw["charged"]==8*nw["processed"],"determinant_node_binding")
        combinations=math.comb(len(pair_objects),4)
        require(nw["requested"]<=combinations,"witness_combination_bound")
        if skipped:
            require(all(w[k]==0 for w in (nw,dw,work) for k in ("requested","charged","processed","uncharged","unprocessed")),"skipped_work")
        require(type(witness["pairs"]) is list and len(witness["pairs"])<=4,"witness_rows")
        if witness["status"]=="found":
            refs=witness["pairs"];require(len(refs)==4 and all(type(i) is int and 0<=i<len(pair_objects) for i in refs) and len(set(refs))==4,"witness_refs")
            require(all(a<b for a,b in zip(refs,refs[1:])),"witness_reference_order")
            require(nw["processed"]==quartet_rank(refs,len(pair_objects))+1,"witness_traversal_position")
            quad=[pair_objects[i] for i in refs]
            require(len({p["left_id"] for p in quad})==len({p["right_id"] for p in quad})==4 and scd.good_four([p["left_xy"] for p in quad]) and scd.good_four([p["right_xy"] for p in quad]),"witness_geometry")
            require(nw["charged"]>=1 and dw["processed"]>=8 and nw["uncharged"]==nw["unprocessed"]==dw["uncharged"]==dw["unprocessed"]==0,"found_work")
        elif witness["status"]=="exhaustive_no_witness":
            require(not witness["pairs"] and nw["processed"]==math.comb(len(pair_objects),4) and nw["uncharged"]==nw["unprocessed"]==dw["uncharged"]==dw["unprocessed"]==0,"exhaustive_work")
        else:
            require(witness["status"]=="budget_inconclusive" and not witness["pairs"] and nw["processed"]<math.comb(len(pair_objects),4),"inconclusive_work")
            require((nw["uncharged"]==1 and nw["charged"]==nw["limit"] and nw["unprocessed"]==0 and dw["uncharged"]==0) or
                (dw["uncharged"]==8 and dw["charged"]+8>dw["limit"] and nw["unprocessed"]==1 and nw["uncharged"]==0),"refused_work")
        expected_status=("insufficient_support" if not all(domains) else "candidate_discrimination_witness" if witness["status"]=="found" else
            "witness_budget_inconclusive" if witness["status"]=="budget_inconclusive" else "reciprocal_pairs_without_common_witness" if pair_objects else "no_admitted_pairs")
        require(summary["status"]==expected_status,"outcome_binding");verified.append(expected_status)
    require(record["outcomes"]==verified,"paired_outcomes")
    return {"frames":frames,"storage_descriptor_elements":sum(len(f.plan.sites)*320 for f in frames),
        "storage_matching_comparisons":0,"storage_common_search_nodes":0,
        "storage_witness_verification_determinants":sum(8 for a in record["arms"] if a["summary"]["witness"]["status"]=="found")}


def dumps(record):
    structure(record);account(record);validate(record)
    wire=canonical(record)+b"\n";byte_guard(len(wire),EVIDENCE_CAP,"evidence_byte_limit")
    require(len(wire)<=account(record)["accounted_upper_bound"],"accounting_proof")
    _Lexical(wire).scan()
    return wire


def loads(data):
    record=parse(data);validate(record);return record


def read(path):
    with Path(path).open("rb") as stream:data=stream.read(EVIDENCE_CAP+1)
    byte_guard(len(data),EVIDENCE_CAP,"evidence_byte_limit")
    return loads(data)


def artifact_bytes(directory):
    return sum(p.stat().st_size for p in Path(directory).iterdir() if p.is_file())


def capped_readback(path,expected_length,reader=None):
    """Read at most one byte past the declared wire size; never an entire file."""
    integer(expected_length,0,EVIDENCE_CAP-1)
    limit=expected_length+1
    if reader is None:
        with Path(path).open("rb") as stream:data=stream.read(limit)
    else:data=reader(Path(path),limit)
    require(type(data) is bytes and len(data)==expected_length,"publication_readback_length")
    return data


def publish(path,record,*,writer=None,flush=None,fsync=None,reader=None,link=None):
    """Single synchronous synthetic transaction; hardlink aliases count twice."""
    path=Path(path);temporary=path.with_name("."+path.name+".partial")
    require(not path.exists() and not temporary.exists(),"publication_collision")
    wire=dumps(record);current=artifact_bytes(path.parent)
    byte_guard(current+2*len(wire),MAX_PUBLICATION,"publication_model_limit")
    byte_guard(current+2*len(wire),AGGREGATE_CAP,"aggregate_byte_limit")
    created=False;peak=current
    try:
        with temporary.open("xb") as stream:
            created=True
            wrote=stream.write(wire) if writer is None else writer(stream,wire)
            require(wrote==len(wire),"publication_short_write")
            (stream.flush if flush is None else lambda:flush(stream))()
            (os.fsync if fsync is None else fsync)(stream.fileno())
        received=capped_readback(temporary,len(wire),reader)
        require(received==wire,"publication_readback");loads(received)
        (os.link if link is None else link)(temporary,path)
        peak=artifact_bytes(path.parent);byte_guard(peak,MAX_PUBLICATION,"publication_model_limit")
        received=capped_readback(path,len(wire),reader)
        require(received==wire,"publication_readback");loads(received)
        return {"bytes":len(wire),"sha256":sha(wire),"aggregate_peak":peak}
    except OSError:
        raise Refusal("publication_io") from None
    finally:
        if created:temporary.unlink()
