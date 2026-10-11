"""Bounded evaluation-only sidecar; importing does no image/run/evidence IO."""
from __future__ import annotations

import ast
import base64
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import threading
import zlib
from contextlib import contextmanager
from contextvars import ContextVar

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "docs/reports/PLANE-PERIOD-SUPPORT-01-2026-10-05.md"
EVIDENCE = ROOT / "docs/reports/PLANE-PERIOD-SUPPORT-01-evidence-2026-10-05.jsonl.gz"
MARKER = "PLANE_PERIOD_SUPPORT_EXECUTION_STATE: NOT_STARTED"
SCHEMA = "mc-photogrammetry-plane-period-support-v1"
TARGET_READINGS = (("image-0000", 2), ("image-0001", 3), ("image-0002", 2), ("image-0002", 3))
TARGET_MATCHES = (
    ("image-0000", 2, 1, .25, "image-0001", 3, 1, .125, 3),
    ("image-0000", 2, 1, .5, "image-0002", 2, 1, .25, 1),
    ("image-0001", 3, 1, 3., "image-0002", 3, 1, .125, 2),
)
BASELINE = ".tmp-geom-phase-capture-r8-20261004/observation.json"
BASELINE_SHA = "c3d2ee4ad3e803754d7297ab0c09b7994b17c24de345fda281c0b5cb62888ff8"
BASELINE_GRAPH_SHA = "0a64c5e73b62b376f41e337a756c61af880d72cee1a90992001a42d00d9f39cd"
BASELINE_TRACE_SHA = "738c9dc09dc17cf64a74e91472cb59a45bbc9917de479cd4f33f80551becd736"
BASELINE_ORDERED_SHA = "28ad4ac48021c727c4b6e7bf283e9965514dc3016c3c16919a18367cf0868105"
CAPS = dict(frames=3, target_readings=4, rounds=3, alternatives=16, candidates=64,
            signature_witnesses=64, matcher_witnesses=3, events=4096,
            compressed_total=128*1024**2, decoded_total=512*1024**2,
            compressed_member=32*1024**2, decoded_member=128*1024**2,
            native_array=32*1024**2, extra_memory=256*1024**2,
            rectified_samples=262144, rectification_taps=1048576,
            signature_memberships=1048576, matcher_pairs=1048576)
_MEMORY = ContextVar("period_support_replay_memory", default=None)


@contextmanager
def memory_scope(budget):
    token=_MEMORY.set(budget)
    try: yield
    finally: _MEMORY.reset(token)


def memory_reserve(amount,detail):
    from mc_photogrammetry.reconstruction import lattice
    budget=_MEMORY.get()
    if budget is None:
        state=lattice._period_support_state()
        budget=state.memory if state is not None else lattice._PeriodSupportMemory(CAPS["extra_memory"])
    try: budget.reserve(int(amount),detail)
    except lattice._PeriodSupportIncomplete as error: raise CaptureRefusal(error.reason,error.detail) from error


def json_bound(value,depth=0,_nodes=None):
    """Worst ASCII JSON length before any base64/string/canonical allocation."""
    if _nodes is None: _nodes=[0]
    _nodes[0]+=1
    require(depth<=24 and _nodes[0]<=100000,"cap_exhausted","json_structure")
    if isinstance(value,np.ndarray): return 2048+4*((value.nbytes+2)//3)
    if isinstance(value,bytes): return 64+4*((len(value)+2)//3)
    if isinstance(value,str): return 2+12*len(value)  # surrogate-pair escapes
    if isinstance(value,dict):
        require(len(value)<=100000,"cap_exhausted","json_structure")
        return 2+sum(json_bound(k,depth+1,_nodes)+json_bound(v,depth+1,_nodes)+2 for k,v in value.items())
    if isinstance(value,(tuple,list)):
        require(len(value)<=100000,"cap_exhausted","json_structure")
        return 2+sum(json_bound(v,depth+1,_nodes)+1 for v in value)
    if isinstance(value,int): return 4+(value.bit_length()*30103)//100000
    return 64


class CaptureRefusal(RuntimeError):
    def __init__(self, reason, detail=""):
        self.reason, self.detail = reason, str(detail)[:256]
        super().__init__(self.detail or reason)


def require(condition, reason="identity_drift", detail=""):
    if not condition:
        raise CaptureRefusal(reason, detail)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(domain, value):
    return hashlib.sha256((SCHEMA + ":" + domain + "\0").encode() + canonical(value)).hexdigest()


def array_record(array):
    require(isinstance(array, np.ndarray), "unsupported_dtype", "array_type")
    require(array.dtype.kind in "biuf" and array.dtype.itemsize in (1,2,4,8), "unsupported_dtype", array.dtype.str)
    require(array.nbytes <= CAPS["native_array"], "cap_exhausted", "native_array")
    memory_reserve(12*array.nbytes+1024**2,"native_encoding")
    raw = array.tobytes(order="C")
    value = dict(dtype=array.dtype.str, byte_order=array.dtype.str[0], shape=list(array.shape),
                 byte_length=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                 payload_base64=base64.b64encode(raw).decode("ascii"))
    value["descriptor_sha256"] = digest("native-array", value)
    return {"native_array": value}


def encode(value):
    if isinstance(value, np.ndarray): return array_record(value)
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            return {"mapping_entries": [[encode(key),encode(item)] for key,item in value.items()]}
        return {key: encode(item) for key,item in value.items()}
    if isinstance(value, (tuple,list)): return [encode(item) for item in value]
    if isinstance(value, bytes): return {"native_bytes": base64.b64encode(value).decode("ascii")}
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": "positive_infinity" if value > 0 else "negative_infinity" if value < 0 else "nan"}
    return value


def array_decode(record):
    keys={"dtype","byte_order","shape","byte_length","sha256","payload_base64","descriptor_sha256"}
    require(type(record) is dict and set(record)==keys, "malformed_descriptor", "keys")
    memory_reserve(3*json_bound(record)+4096,"native_descriptor_hash")
    metadata={key:item for key,item in record.items() if key!="descriptor_sha256"}
    require(record["descriptor_sha256"]==digest("native-array",metadata), "digest_mismatch", "descriptor")
    try: dtype=np.dtype(record["dtype"])
    except (TypeError,ValueError) as error: raise CaptureRefusal("unsupported_dtype") from error
    require(dtype.kind in "biuf" and dtype.itemsize in (1,2,4,8), "unsupported_dtype")
    require(dtype.str==record["dtype"] and record["byte_order"]==dtype.str[0], "malformed_descriptor", "byte_order")
    shape=record["shape"]
    require(type(shape) is list and len(shape)<=4 and all(type(v) is int and v>=0 for v in shape), "malformed_descriptor", "shape")
    n=math.prod(shape)*dtype.itemsize
    require(type(record["byte_length"]) is int and n==record["byte_length"], "malformed_descriptor", "length")
    require(n<=CAPS["native_array"], "cap_exhausted", "native_array")
    require(type(record["payload_base64"]) is str and len(record["payload_base64"])==4*((n+2)//3), "malformed_descriptor", "encoded_length")
    memory_reserve(n+len(record["payload_base64"])+4096,"native_decoding")
    try: raw=base64.b64decode(record["payload_base64"],validate=True)
    except (ValueError,base64.binascii.Error) as error: raise CaptureRefusal("malformed_descriptor","base64") from error
    require(len(raw)==n and hashlib.sha256(raw).hexdigest()==record["sha256"], "digest_mismatch", "native_bytes")
    return np.frombuffer(raw,dtype=dtype).reshape(shape)


def decode(value):
    if isinstance(value,list): return tuple(decode(item) for item in value)
    if isinstance(value,dict):
        if set(value)=={"mapping_entries"}:
            entries=[(decode(key),decode(item)) for key,item in value["mapping_entries"]]
            result=dict(entries)
            require(len(result)==len(entries),"ambiguous_join","mapping_keys")
            return result
        if set(value)=={"native_array"}: return array_decode(value["native_array"])
        if set(value)=={"native_bytes"}:
            memory_reserve(2*len(value["native_bytes"])+4096,"native_bytes_decoding")
            return base64.b64decode(value["native_bytes"],validate=True)
        if set(value)=={"nonfinite"}:
            require(value["nonfinite"] in ("positive_infinity","negative_infinity","nan"),"malformed_descriptor")
            return {"positive_infinity":float("inf"),"negative_infinity":-float("inf"),"nan":float("nan")}[value["nonfinite"]]
        return {key:decode(item) for key,item in value.items()}
    return value


def fields(record, name):
    require(type(record) is dict and set(record)=={"record_type","fields"} and record["record_type"]==name,
            "ambiguous_join", name)
    return record["fields"]


def validate_region_plan(record,trusted_shape):
    """Bind scalar allocation dimensions before any region constructor/decode."""
    value=fields(record,"PlaneRegionEvidence")
    require(isinstance(trusted_shape,(tuple,list)) and len(trusted_shape)==2
            and all(type(v) is int and v>0 for v in trusted_shape),"missing_dependency","trusted_region_shape")
    shape=value.get("image_shape"); crop=value.get("crop_shape"); bbox=value.get("bbox_xyxy")
    require(isinstance(shape,(tuple,list)) and len(shape)==2 and tuple(shape)==tuple(trusted_shape)
            and all(type(v) is int and v>0 for v in shape),"identity_drift","region_source_shape")
    require(isinstance(crop,(tuple,list)) and len(crop)==2 and all(type(v) is int and v>0 for v in crop)
            and isinstance(bbox,(tuple,list)) and len(bbox)==4 and all(type(v) is int for v in bbox),
            "identity_drift","region_crop_shape")
    h,w=shape; ch,cw=crop; x0,y0,x1,y1=bbox
    require(0<=x0<x1<=w and 0<=y0<y1<=h and (ch,cw)==(y1-y0,x1-x0),
            "identity_drift","region_crop_bounds")
    pixels=h*w; bits=ch*cw; packed=(bits+7)//8; payload=value.get("packed_mask_base64")
    require(pixels<=CAPS["native_array"] and type(value.get("pixel_count")) is int
            and 0<value["pixel_count"]<=bits,"cap_exhausted","region_pixels")
    require(type(payload) is str and len(payload)==4*((packed+2)//3),"identity_drift","region_packed_length")
    # Full bool mask, unpacked crop and padding, canonical base64/hash buffers,
    # row/view objects and constructor bookkeeping. Dimensions are now trusted.
    memory_reserve(pixels+12*bits+12*packed+65536,"region_decode_planned_shapes")


def materialize(record,trusted_shapes=None):
    """Replay only whitelisted value records, never numerical producers."""
    from mc_photogrammetry.reconstruction import lattice_planes as lp
    from mc_photogrammetry.pipeline import image_derived_signatures as sig
    types={name:getattr(lp,name) for name in ("PlaneReading","PlaneRectification",
            "PlaneRegionEvidence","PeriodicEnergyPhaseEvidence","DistanceAlternative",
            "PlaneSignature","PlaneRegionFootprint")}
    types["SignatureEvidence"]=sig.SignatureEvidence
    if isinstance(record,dict) and set(record)=={"record_type","fields"}:
        require(record["record_type"] in types,"missing_dependency","record_type")
        if record["record_type"]=="PlaneReading" and record["fields"].get("region_evidence") is not None:
            frame=record["fields"].get("frame")
            require(trusted_shapes is not None and frame in trusted_shapes,"missing_dependency","reading_trusted_shape")
            validate_region_plan(record["fields"]["region_evidence"],trusted_shapes[frame])
        return types[record["record_type"]](**{k:materialize(v,trusted_shapes) for k,v in record["fields"].items()})
    if isinstance(record,tuple): return tuple(materialize(v,trusted_shapes) for v in record)
    return record


def reading_record_order_key(record,trusted_shapes=None):
    from mc_photogrammetry.reconstruction import lattice_planes as lp
    return lp.reading_order_key(materialize(record,trusted_shapes))


def verify_geometry(g,map_x,map_y,valid):
    n=g["samples"]; centre=g["centre"]; extent=g["extent"]
    require(type(n) is int and n>=2 and n*n<=CAPS["rectified_samples"],"cap_exhausted","projection_samples")
    require(all(isinstance(a,np.ndarray) and a.shape==(n,n) for a in (map_x,map_y,valid))
            and map_x.dtype==map_y.dtype==np.dtype(np.float32) and valid.dtype==np.bool_,
            "identity_drift","projection_shapes")
    axis=g["axis"]; matrix=g["axis_matrix"]
    require(type(axis) is int and axis in (0,1,2) and g["side"] in (-1.,1.)
            and isinstance(centre,(tuple,list)) and len(centre)==2
            and isinstance(g["principal_point"],(tuple,list)) and len(g["principal_point"])==2,
            "identity_drift","projection_scalars")
    require((isinstance(matrix,np.ndarray) and matrix.shape==(3,3)) or
            (isinstance(matrix,(tuple,list)) and len(matrix)==3
             and all(isinstance(row,(tuple,list)) and len(row)==3 for row in matrix)),
            "identity_drift","projection_axes_shape")
    require(all(isinstance(v,(int,float,np.integer,np.floating)) and not isinstance(v,(bool,np.bool_))
                and math.isfinite(float(v)) for row in matrix for v in row),
            "identity_drift","projection_axes_values")
    require(all(type(v) in (int,float) and math.isfinite(v) for v in (*centre,*g["principal_point"],extent,g["focal"]))
            and extent>0 and g["focal"]>0,"identity_drift","projection_finite")
    memory_reserve(640*n*n+128*n+65536,"projection_planned_shapes")
    axes=np.asarray(matrix); others=[i for i in range(3) if i!=axis]
    require(np.isfinite(axes).all(),"identity_drift","projection_axes_finite")
    u,v=np.meshgrid(np.linspace(centre[0]-extent,centre[0]+extent,n),
                    np.linspace(centre[1]-extent,centre[1]+extent,n))
    points=g["side"]*axes[axis][None,None,:]+u[...,None]*axes[others[0]][None,None,:]+v[...,None]*axes[others[1]][None,None,:]
    depth=points[...,2]; expected=depth>1e-6
    with np.errstate(divide="ignore",invalid="ignore"):
        x=g["principal_point"][0]+points[...,0]*g["focal"]/depth
        y=g["principal_point"][1]-points[...,1]*g["focal"]/depth
    require(np.array_equal(valid,expected) and np.array_equal(map_x,np.where(expected,x,-1.).astype(np.float32))
            and np.array_equal(map_y,np.where(expected,y,-1.).astype(np.float32)),"identity_drift","projection_maps")


def verify_window(mask,source,axis,window):
    from mc_photogrammetry.reconstruction import lattice
    ys,xs=np.nonzero(mask); matrix=source["axis_matrix"]; focal=source["focal"]; cx,cy=source["principal_point"]
    rays=np.stack(((xs-cx)/focal,-(ys-cy)/focal,np.ones(len(xs))),axis=1)
    normal=matrix[axis]; others=[a for a in range(3) if a!=axis]
    incidence=(rays/np.linalg.norm(rays,axis=1)[:,None])@(normal/np.linalg.norm(normal))
    projection=rays@normal
    side=1. if np.count_nonzero(projection>1e-6)>=np.count_nonzero(projection<-1e-6) else -1.
    stable=incidence*side>lattice.GRAZING_INCIDENCE
    require(np.any(stable),"identity_drift","window_incidence")
    points=rays[stable]/(projection[stable]*side)[:,None]
    u=points@matrix[others[0]]; v=points@matrix[others[1]]
    expected=(float((u.min()+u.max())/2),float((v.min()+v.max())/2),
              float(max(u.max()-u.min(),v.max()-v.min())/2)*1.05,side)
    require(expected==window,"identity_drift","proposal_window_geometry")


def registered_reading_policy():
    from mc_photogrammetry.reconstruction import lattice_planes as lp
    from mc_photogrammetry.pipeline import image_derived_lattice_measure as measure
    return dict(min_structural_share=lp.MIN_STRUCTURAL_SHARE,max_readings=lp.MAX_READINGS,
                max_per_axis_side=lp.MAX_READINGS_PER_AXIS_SIDE,max_rectified_side=measure.LATTICE_MAX_RECTIFIED_SIDE)


def verify_phase(reading,patch):
    phase=fields(reading["phase_evidence"],"PeriodicEnergyPhaseEvidence"); period=reading["period_pixels"]
    values=np.asarray(patch,dtype=float)
    for axis,profile in enumerate((values.sum(axis=0),values.sum(axis=1))):
        mass=float(profile.sum()); reason=None
        if mass<=1e-9: result=(None,0.,mass,0.,"empty_profile")
        else:
            response=np.sum(profile*np.exp(-1j*(2.*np.pi*np.arange(len(profile),dtype=float)/period)))
            magnitude=float(abs(response)); concentration=float(np.clip(magnitude/mass,0.,1.))
            if period<2.: value=None; reason="insufficient_period"
            elif magnitude<=1e-12: value=None; reason="zero_resultant"
            else:
                value=float((-np.angle(response)/(2.*np.pi)*period)%period)
                if value>=period-max(1e-12,period*1e-12): value=0.
            result=(value,magnitude,mass,concentration,reason)
        require(result==tuple(phase[key][axis] for key in ("phase_uv_pixels","response_magnitude_uv",
                  "profile_mass_uv","concentration_uv","unavailable_reason_uv")),"identity_drift","phase_energy_arithmetic")
    require(reading["phase_pixels"]==tuple(float(p if p is not None else 0.) for p in phase["phase_uv_pixels"]),"identity_drift","legacy_phase_binding")


def fixed_taps(map_x,map_y,shape):
    # This decodes the pinned interpolation table; it never samples an image.
    xy,table=cv2.convertMaps(map_x,map_y,cv2.CV_16SC2)
    fx=table.astype(np.int64)&31; fy=table.astype(np.int64)>>5
    h,w=shape
    for dx,dy,weight in ((0,0,(32-fx)*(32-fy)),(1,0,fx*(32-fy)),
                         (0,1,(32-fx)*fy),(1,1,fx*fy)):
        x=xy[...,0].astype(np.int64)+dx; y=xy[...,1].astype(np.int64)+dy
        inside=(x>=0)&(x<w)&(y>=0)&(y<h)
        yield np.where(inside,y*w+x,-1),weight


def verify_remap_values(source,map_x,map_y,result):
    # Explicitly lead-authorized evaluator-only replay AFTER production/output
    # seal. This verifies pixels, never claims the legacy tap table is exact.
    # Registered execution is separately blocked on active-kernel provenance.
    memory_reserve(3*int(result.nbytes)+512*1024,"postproduction_remap_output_and_byte_comparison")
    expected=cv2.remap(source,map_x,map_y,cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    require(expected.dtype==result.dtype and expected.shape==result.shape
            and memoryview(expected).tobytes()==memoryview(result).tobytes(),"identity_drift","remap_pixels")


def verify_native_taps(witness):
    expected_ptr=[0]; indices=[]; weights=[]
    tap_maps=tuple(fixed_taps(witness.map_x,witness.map_y,witness.source_shape))
    for sample in range(witness.map_x.size):
        for native,weight in tap_maps:
            value=int(weight.flat[sample])
            if value:
                indices.append(int(native.flat[sample])); weights.append(value)
        expected_ptr.append(len(indices))
    require(np.array_equal(expected_ptr,witness.tap_ptr) and np.array_equal(indices,witness.tap_source_indices)
            and np.array_equal(weights,witness.tap_weight_numerators),"identity_drift","native_tap_table")


def verify_period_arithmetic(payload,signal):
    from mc_photogrammetry.reconstruction import lattice
    reading=payload["estimate"]
    if signal.size<8 or float(signal.var())<lattice.MINIMUM_CONTRAST:
        require(reading is None and payload["correlation"] is None,"identity_drift","contrast_refusal"); return
    corr=np.correlate(signal,signal,mode="full")[len(signal)-1:]
    if corr[0]<=0:
        require(reading is None,"identity_drift","correlation_refusal"); return
    corr=corr/corr[0]
    require(np.array_equal(corr,payload["correlation"]),"identity_drift","signal_correlation")
    low=max(int(np.floor(payload["minimum_period"])),1)
    high=min(int(np.ceil(payload["maximum_period"] or len(signal)/3.)),len(corr)-1)
    peaks=np.flatnonzero((corr[1:-1]>=corr[:-2])&(corr[1:-1]>=corr[2:]))+1
    peaks=peaks[(peaks>=low)&(peaks<high)]
    require(np.array_equal(peaks,payload["peaks"]),"identity_drift","correlation_peaks")
    if high<=low or not len(peaks) or float(corr[peaks].max())<=0:
        require(reading is None,"identity_drift","peak_refusal"); return
    choices=peaks[corr[peaks]>=lattice.HARMONIC_SHARE*float(corr[peaks].max())]
    require(np.array_equal(choices,payload["harmonic_candidates"]),"identity_drift","harmonic_candidates")
    peak=int(choices[0]) if len(choices) else int(peaks[np.argmax(corr[peaks])]); strength=float(corr[peak])
    period=float(peak)
    if 1<=peak<len(corr)-1:
        before,centre,after=corr[peak-1:peak+2]; den=before-2.*centre+after
        if abs(den)>1e-12: period=float(peak+.5*(before-after)/den)
    require(reading is not None,"identity_drift","period_missing")
    r=fields(reading,"PeriodEstimate"); angles=2.*np.pi*np.arange(len(signal))/period
    phase=float((float(np.angle(np.sum(signal*np.exp(-1j*angles))))/(2.*np.pi))*period%period)
    require(r["period_pixels"]==period and r["strength"]==strength and r["phase_pixels"]==phase,"identity_drift","period_peak_interpolation_phase")


def verify_warp_values(witness,signature,cells,coverage):
    rows,columns=witness.target_shape; srcrows,srccols=signature["cells"].shape
    rr,cc=np.mgrid[:rows,:columns]; points=np.stack((cc.astype(float),rr.astype(float),np.ones((rows,columns))),axis=0)
    mapped=witness.homography@points.reshape(3,-1)
    with np.errstate(divide="ignore",invalid="ignore"):
        u=(mapped[0]/mapped[2]).reshape(rows,columns); v=(mapped[1]/mapped[2]).reshape(rows,columns)
    valid=np.isfinite(u)&np.isfinite(v)&(u>=-.5)&(u<=srccols-.5)&(v>=-.5)&(v<=srcrows-.5)
    sr=np.full((rows,columns),-1,dtype=np.int64); sc=sr.copy()
    sr[valid]=np.clip(np.rint(v[valid]).astype(int),0,srcrows-1); sc[valid]=np.clip(np.rint(u[valid]).astype(int),0,srccols-1)
    require(np.array_equal(u,witness.mapped_u,equal_nan=True) and np.array_equal(v,witness.mapped_v,equal_nan=True)
            and np.array_equal(valid,witness.geometric_valid) and np.array_equal(sr,witness.source_rows)
            and np.array_equal(sc,witness.source_columns),"identity_drift","warp_geometry")
    expected_cells=np.zeros((rows,columns),dtype=np.float32); expected_coverage=np.zeros((rows,columns),dtype=bool)
    expected_cells[valid]=signature["cells"][sr[valid],sc[valid]]; expected_coverage[valid]=signature["coverage"][sr[valid],sc[valid]]
    require(np.array_equal(cells,expected_cells) and np.array_equal(coverage,expected_coverage),"identity_drift","warp_signature_values")


def sealed_event(index, previous_sha, run_id, kind, context, payload):
    record=dict(schema=SCHEMA,index=index,previous_sha256=previous_sha,run_id=run_id,
                kind=kind,context=encode(context),payload=encode(payload))
    return dict(record,event_sha256=digest("event",record))


def validate_event(record,index,previous_sha,run_id):
    require(type(record) is dict and set(record)=={"schema","index","previous_sha256","run_id","kind","context","payload","event_sha256"}, "identity_drift","event_keys")
    require(record["schema"]==SCHEMA and record["index"]==index and record["previous_sha256"]==previous_sha and record["run_id"]==run_id, "identity_drift","event_order")
    memory_reserve(3*json_bound(record)+4096,"event_hash")
    body={key:item for key,item in record.items() if key!="event_sha256"}
    require(digest("event",body)==record["event_sha256"],"digest_mismatch","event")
    return decode(record["context"]),decode(record["payload"])


class _CompressedSink:
    def __init__(self,journal): self.journal=journal; self.count=0
    def write(self,data):
        journal=self.journal
        require(self.count+len(data)<=journal.caps["compressed_member"],"cap_exhausted","compressed_member")
        require(journal.total+len(data)<=journal.caps["compressed_total"],"cap_exhausted","compressed_total")
        require(journal.file.tell()==journal.total,"write_failure","append_offset")
        written=journal.file.write(data)
        require(written==len(data),"write_failure","short_write")
        journal.sha.update(data); journal.total+=written; self.count+=written
        return written
    def flush(self): pass


class Journal:
    """Single append owner, independent complete gzip member per sealed event."""
    def __init__(self,file,*,run_id,caps=None,durable=False):
        self.file,self.run_id,self.caps=file,run_id,dict(CAPS if caps is None else caps)
        self.durable=durable; self.total=self.decoded_total=0; self.members=[]
        self.sha=hashlib.sha256(); self.previous="0"*64; self.failed=False
        require(file.tell()==0,"write_failure","nonempty_initial_offset")
        file.seek(0,2); require(file.tell()==0,"write_failure","nonempty_file"); file.seek(0)
        self.thread=threading.get_ident()
        from mc_photogrammetry.reconstruction import lattice
        state=lattice._period_support_state()
        self.memory=state.memory if state is not None else lattice._PeriodSupportMemory(self.caps["extra_memory"])

    def append(self,kind,context,payload):
        from mc_photogrammetry.reconstruction import lattice
        state=lattice._period_support_state()
        # A journal is usually created before capture scope. Join its prior
        # reservations into the single production-observer ledger exactly once.
        try:
            if state is not None and self.memory is not state.memory:
                state.memory.reserve(self.memory.used,"journal_previous_allocations")
                self.memory=state.memory
            with memory_scope(self.memory):
                return self._append(kind,context,payload)
        except Exception:
            self.failed=True
            raise

    def _append(self,kind,context,payload):
        require(not self.failed and threading.get_ident()==self.thread,"write_failure","journal_owner")
        require(len(self.members)<self.caps["events"],"cap_exhausted","events")
        bound=json_bound(dict(context=context,payload=payload,kind=kind))+4096
        # Encoding tree + two canonical UTF8 buffers + worst-width JSON unicode
        # strings, encoder pieces, compression state and later readback overlap.
        memory_reserve(12*bound+2*1024**2,"sealed_event_encoding_and_overlap")
        record=sealed_event(len(self.members),self.previous,self.run_id,kind,context,payload)
        offset=self.total; decoded_count=0; sink=_CompressedSink(self)
        try:
            with gzip.GzipFile(filename="",mode="wb",fileobj=sink,compresslevel=6,mtime=0) as output:
                encoder=json.JSONEncoder(sort_keys=True,separators=(",",":"),allow_nan=False)
                for piece in encoder.iterencode(record):
                    for start in range(0,len(piece),65536):
                        raw=piece[start:start+65536].encode("utf-8")
                        decoded_count+=len(raw)
                        require(decoded_count+1<=self.caps["decoded_member"],"cap_exhausted","decoded_member")
                        require(self.decoded_total+decoded_count+1<=self.caps["decoded_total"],"cap_exhausted","decoded_total")
                        output.write(raw)
                output.write(b"\n"); decoded_count+=1
            self.file.flush()
            if self.durable: os.fsync(self.file.fileno())
            member=dict(offset=offset,length=self.total-offset,decoded_length=decoded_count,
                        event_sha256=record["event_sha256"])
            memory_reserve(member["length"]+1024,"compressed_readback")
            position=self.file.tell(); self.file.seek(offset); compressed=self.file.read(member["length"]); self.file.seek(position)
            member["compressed_sha256"]=hashlib.sha256(compressed).hexdigest()
            replay=_decode_member(compressed,self.caps,expected_decoded=decoded_count)
            require(replay==record,"digest_mismatch","member_readback")
            self.members.append(member); self.decoded_total+=decoded_count; self.previous=record["event_sha256"]
            return record["event_sha256"]
        except Exception:
            self.failed=True
            raise


def _decode_member(data,caps,*,expected_decoded=None):
    if _MEMORY.get() is None:
        from mc_photogrammetry.reconstruction import lattice
        with memory_scope(lattice._PeriodSupportMemory(caps["extra_memory"])):
            return _decode_member_impl(data,caps,expected_decoded=expected_decoded)
    return _decode_member_impl(data,caps,expected_decoded=expected_decoded)


def _decode_member_impl(data,caps,*,expected_decoded=None):
    require(0<len(data)<=caps["compressed_member"],"cap_exhausted","compressed_member")
    memory_reserve(512*1024,"gzip_decoder_state")
    decoder=zlib.decompressobj(31); pieces=[]; length=0
    try:
        for offset in range(0,len(data),65536):
            memory_reserve(65536+1024,"compressed_chunk_before_slice")
            chunk=data[offset:offset+65536]
            while chunk:
                bound=min(65536,caps["decoded_member"]-length+1)
                memory_reserve(3*bound+4*65536+4096,"decoded_chunk_before_allocation")
                out=decoder.decompress(chunk,bound)
                length+=len(out)
                require(length<=caps["decoded_member"],"cap_exhausted","decoded_member")
                pieces.append(out); chunk=decoder.unconsumed_tail
                if decoder.eof:
                    require(not decoder.unused_data and offset+65536>=len(data),"digest_mismatch","trailing_member_bytes")
                    break
        require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,"digest_mismatch","gzip_trailer")
    except zlib.error as error: raise CaptureRefusal("digest_mismatch","gzip_crc_or_trailer") from error
    require(expected_decoded is None or length==expected_decoded,"digest_mismatch","decoded_length")
    memory_reserve(length+1024,"decoded_join_before_allocation")
    raw=b"".join(pieces)
    require(raw.endswith(b"\n") and raw.count(b"\n")==1,"identity_drift","one_event_per_member")
    # Count punctuation even inside strings: over-counting is conservative.
    # Each value/container/key gets 512B (above pinned CPython object/table
    # overhead); unicode/UTF8 parser temporaries get 8 bytes per source byte.
    nodes=sum(raw.count(token) for token in (b",",b":",b"[",b"{",b"]",b"}"))+1
    memory_reserve(8*length+512*nodes+4096,"json_tree_before_allocation")
    try: return json.loads(raw)
    except (ValueError,UnicodeError) as error: raise CaptureRefusal("identity_drift","json") from error


def replay(file,members,*,run_id,expected_sha256,caps=None,consumer=None):
    from mc_photogrammetry.reconstruction import lattice
    caps=dict(CAPS if caps is None else caps)
    with memory_scope(lattice._PeriodSupportMemory(caps["extra_memory"])):
        return _replay(file,members,run_id=run_id,expected_sha256=expected_sha256,caps=caps,consumer=consumer)


def _replay(file,members,*,run_id,expected_sha256,caps=None,consumer=None):
    caps=dict(CAPS if caps is None else caps); offset=decoded_total=0; sha=hashlib.sha256(); previous="0"*64
    require(0<len(members)<=caps["events"],"cap_exhausted","event_count")
    for index,member in enumerate(members):
        require(member["offset"]==offset,"identity_drift","member_order")
        length=member["length"]
        require(type(length) is int and 0<length<=caps["compressed_member"],"cap_exhausted","compressed_member")
        require(offset+length<=caps["compressed_total"],"cap_exhausted","compressed_total")
        memory_reserve(length+1024,"compressed_member_before_read")
        file.seek(offset); compressed=file.read(length)
        require(len(compressed)==length and hashlib.sha256(compressed).hexdigest()==member["compressed_sha256"],"digest_mismatch","member_bytes")
        record=_decode_member(compressed,caps,expected_decoded=member["decoded_length"])
        context,payload=validate_event(record,index,previous,run_id)
        require(record["event_sha256"]==member["event_sha256"],"digest_mismatch","member_event")
        decoded_total+=member["decoded_length"]
        require(decoded_total<=caps["decoded_total"],"cap_exhausted","decoded_total")
        sha.update(compressed); offset+=length; previous=record["event_sha256"]
        if consumer is not None: consumer(record["kind"],context,payload)
    require(file.read(1)==b"","digest_mismatch","extra_members_or_junk")
    require(sha.hexdigest()==expected_sha256,"digest_mismatch","full_stream")
    return dict(status="verified",events=len(members),compressed_bytes=offset,
                decoded_bytes=decoded_total,evidence_sha256=sha.hexdigest(),last_event_sha256=previous)


class Observer:
    def __init__(self,journal,*,run_id,signature_targets=TARGET_READINGS,match_targets=TARGET_MATCHES):
        self.journal,self.run_id=journal,run_id
        self.signature_targets,self.match_targets=tuple(signature_targets),tuple(match_targets)
        self.incomplete=None; self.finished=False; self.frames=set(); self.candidates=set(); self.rounds={}
        self.trusted_ingress={}

    def __call__(self,kind,context,payload):
        require(context.get("run_id")==self.run_id,"identity_drift","foreign_run")
        if kind=="frame_source":
            frame=payload["frame"]
            require(frame not in self.frames,"ambiguous_join","frame_source")
            require(len(self.frames)<CAPS["frames"],"cap_exhausted","frames"); self.frames.add(frame)
            require(payload.get("ingress") is not None,"missing_dependency","trusted_ingress")
            self.trusted_ingress[frame]=dict(payload["ingress"])
        if kind=="candidate_begin":
            identity=tuple(context["candidate_id"])
            require(identity not in self.candidates,"ambiguous_join","candidate_id")
            require(len(self.candidates)<CAPS["candidates"],"cap_exhausted","candidates"); self.candidates.add(identity)
        if kind=="measurement_round_begin":
            identity=tuple(context["candidate_id"]); expected=self.rounds.get(identity,0)
            require(payload["round_index"]==expected and expected<CAPS["rounds"],"cap_exhausted","rounds")
            self.rounds[identity]=expected+1
        if kind=="distance_alternatives": require(len(payload["evaluated"])<=CAPS["alternatives"],"cap_exhausted","alternatives")
        self.journal.append(kind,context,payload)

    def finish_capture(self,incomplete):
        self.incomplete=incomplete
        self.finished=True


class SemanticReplay:
    """Streaming arithmetic/identity checks, never estimator/rectifier/matcher calls."""
    def __init__(self,*,target_readings=TARGET_READINGS,target_matches=TARGET_MATCHES,trusted_ingress=None):
        self.target_readings=tuple(target_readings); self.target_matches=tuple(target_matches)
        self.candidates={}; self.readings={}; self.round=None; self.signature_parts={}
        self.signature_bindings={}; self.matcher_parts={}; self.matcher_pairs={}
        self.final_matches=None; self.solver_matches=None; self.core=None; self.solution=None; self.solver_parameters=None
        self.source_frames={}; self.frame_masks={}; self.selected_candidates={}
        self.trusted_ingress={} if trusted_ingress is None else trusted_ingress
        self.measure_sources={}; self.original_indices={}; self.signatures={}; self.structural_sources={}

    def __call__(self,kind,context,payload):
        from mc_photogrammetry.reconstruction import lattice, lattice_planes as lp, plane_graph as pg
        trusted_shapes={frame:source["shape"] for frame,source in self.source_frames.items()}
        def max_samples(value):
            if isinstance(value,np.ndarray): return int(value.size)
            if isinstance(value,dict): return max((max_samples(v) for v in value.values()),default=0)
            if isinstance(value,(tuple,list)): return max((max_samples(v) for v in value),default=0)
            return 0
        # Includes independent replay maps, tap lists/sets, native arithmetic
        # intermediates and materialized value-record overhead. Freed objects
        # stay charged; retained source/round/signature state cannot escape cap.
        multiplier=2048 if kind=="signature_sampling" else 1024 if "rectification" in kind else 256
        memory_reserve(multiplier*max_samples(payload)+1024**2,"semantic_"+kind)
        identity=tuple(context.get("candidate_id",()))
        if kind=="frame_source":
            frame=payload["frame"]; require(frame not in self.source_frames,"ambiguous_join","source_frame")
            image=payload["rgb"]; pin=payload["input_identity"]
            require(image.ndim==3 and image.shape[2]==3 and image.dtype==np.uint8,"identity_drift","rgb")
            require(pin["frame_id"]==frame and (pin["height"],pin["width"])==image.shape[:2],"identity_drift","source_shape")
            ingress=payload.get("ingress"); trusted=self.trusted_ingress.get(frame)
            require(trusted is not None and same(ingress,trusted),"identity_drift","trusted_ingress")
            actual=hashlib.sha256(memoryview(image)).hexdigest()
            require(ingress["byte_sha256"]==pin["sha256"] and ingress["frame_id"]==frame
                    and tuple(ingress["decoded_shape"])==image.shape and ingress["decoded_dtype"]==image.dtype.str
                    and ingress["decoded_pixel_sha256"]==payload["decoded_sha256"]==actual,"identity_drift","ingress_rgb")
            self.source_frames[frame]=dict(pin=pin,decoded_sha256=actual,shape=image.shape[:2],rgb=image)
        elif kind=="frame_structural_source":
            frame=payload["frame"]
            require(frame in self.source_frames and frame not in self.structural_sources,
                    "ambiguous_join","structural_source")
            segments=payload["segments"]; weights=payload["segment_weights"]
            require(segments.ndim==2 and segments.shape[1]==4 and weights.shape==(len(segments),)
                    and np.isfinite(segments).all() and np.isfinite(weights).all(),
                    "identity_drift","structural_segments")
            h,w=self.source_frames[frame]["shape"]
            memory_reserve(8192*len(segments)+64*h*w+1024**2,"structural_source_replay")
            grey=cv2.cvtColor(self.source_frames[frame]["rgb"],cv2.COLOR_RGB2GRAY)
            usable=payload["usable"]
            require(usable.dtype==np.bool_ and usable.shape==(h,w),"identity_drift","structural_source_mask")
            grey[~usable]=0
            require(same(grey,payload["grey"]),"identity_drift","structural_source_pixels")
            from mc_photogrammetry.reconstruction import calibration
            fractions=np.linspace(.15,.85,calibration.CONTRAST_SAMPLES)[None,:]
            x0,y0=segments[:,0,None],segments[:,1,None]
            dx,dy=segments[:,2]-segments[:,0],segments[:,3]-segments[:,1]
            length=np.maximum(np.hypot(dx,dy),1e-9)
            xs=x0+fractions*dx[:,None]; ys=y0+fractions*dy[:,None]
            nx,ny=(-dy/length)[:,None],(dx/length)[:,None]
            def sample(px,py):
                rr=np.clip(np.rint(py).astype(int),0,h-1)
                cc=np.clip(np.rint(px).astype(int),0,w-1)
                return grey[rr,cc]
            near=sample(xs+nx*calibration.CONTRAST_OFFSET,ys+ny*calibration.CONTRAST_OFFSET)
            far=sample(xs-nx*calibration.CONTRAST_OFFSET,ys-ny*calibration.CONTRAST_OFFSET)
            contrast=np.abs(near.astype(float)-far.astype(float)).mean(axis=1)
            reference=float(np.median(contrast)) if len(contrast) else 0.
            if reference<=1e-6 and len(contrast): reference=float(np.mean(contrast))
            expected_weights=(np.ones(contrast.shape) if reference<=1e-6 else
                              np.clip(contrast/(2.*reference),calibration.CONTRAST_FLOOR,1.))
            require(same(expected_weights,weights),"identity_drift","structural_contrast_weights")
            # Post-seal raster arithmetic only. No detector/measurement call.
            expected=np.zeros((h,w),dtype=np.float32)
            order=np.lexsort((segments[:,3],segments[:,2],segments[:,1],segments[:,0]))
            for index in order:
                weight=float(weights[index])
                if weight<=0.: continue
                x0,y0,x1,y1=(float(v) for v in segments[index,:4])
                cv2.line(expected,(int(round(x0)),int(round(y0))),(int(round(x1)),int(round(y1))),
                         weight,1,lineType=cv2.LINE_8)
            require(same(expected,payload["structural_weights"]),"identity_drift","structural_raster")
            self.structural_sources[frame]=payload
        elif kind=="frame_measure_source":
            frame=payload["frame"]; usable=payload["usable"]
            require(same(payload.get("selection_policy"),registered_reading_policy()),
                    "identity_drift","effective_reading_policy")
            require(usable.dtype==np.bool_ and usable.shape==payload["grey"].shape==payload["structural_weights"].shape,"identity_drift","frame_measure_shape")
            require(frame in self.source_frames,"missing_dependency","measurement_source_rgb")
            expected=cv2.cvtColor(self.source_frames[frame]["rgb"],cv2.COLOR_RGB2GRAY); expected[~usable]=0
            require(np.array_equal(expected,payload["grey"]),"identity_drift","source_grayscale")
            if frame in self.structural_sources:
                structural=self.structural_sources[frame]
                require(same(payload["structural_weights"],structural["structural_weights"])
                        and same(payload["usable"],structural["usable"]),
                        "identity_drift","measurement_structural_source")
                cal=fields(structural["calibration"],"CalibrationResult")
                require(payload["focal"]==cal["focal_pixels"] and cal["width"]==usable.shape[1]
                        and cal["height"]==usable.shape[0],"identity_drift","measurement_camera")
                expected_axes=lp.axis_directions(cal["yaw_degrees"],cal["pitch_degrees"],cal["roll_degrees"])
                principal=cal["principal_point"] or (cal["width"]/2.,cal["height"]/2.)
                require(same(expected_axes,payload["axis_matrix"]) and same(principal,payload["principal_point"]),
                        "identity_drift","measurement_camera_basis")
            self.measure_sources[frame]=payload
            self.frame_masks[frame]=(usable.shape,hashlib.sha256(usable.tobytes()).hexdigest())
        elif kind=="candidate_begin":
            require(len(identity)==2 and identity not in self.candidates,"ambiguous_join","candidate")
            evidence=payload["region_evidence"]
            require(evidence is not None,"missing_dependency","proposal_region")
            require(identity[0] in trusted_shapes,"missing_dependency","candidate_frame")
            validate_region_plan(evidence,trusted_shapes[identity[0]])
            region=lp.PlaneRegionEvidence(**fields(evidence,"PlaneRegionEvidence"))
            mask=lp._decode_region_evidence(region,payload["axis"],payload["polygon"])
            require(mask.shape==region.image_shape,"identity_drift","region_shape")
            require(identity[0] in self.measure_sources and mask.shape==self.frame_masks[identity[0]][0],"missing_dependency","candidate_frame")
            require(not np.any(mask & ~self.measure_sources[identity[0]]["usable"]),"identity_drift","region_usable")
            self.candidates[identity]=dict(axis=payload["axis"],region_id=region.source_region_id,
                polygon=payload["polygon"],rounds=0,reading=None,alternatives=None,mask=mask,remaps=[])
        elif kind=="candidate_window":
            candidate=self.candidates[identity]; window=payload["window"]
            require(fields(payload["region_evidence"],"PlaneRegionEvidence")["source_region_id"]==candidate["region_id"],"identity_drift","window_region")
            require(len(window)==4 and np.isfinite(window).all() and window[2]>0 and window[3] in (-1.,1.),"identity_drift","candidate_window")
            verify_window(candidate["mask"],self.measure_sources[identity[0]],candidate["axis"],window)
            candidate["window"]=window
        elif kind=="measurement_rectification":
            candidate=self.candidates[identity]; source=self.measure_sources[identity[0]]
            verify_geometry(payload,payload["map_x"],payload["map_y"],payload["depth_valid"])
            require(payload["axis"]==candidate["axis"] and same(payload["axis_matrix"],source["axis_matrix"])
                    and payload["focal"]==source["focal"] and payload["principal_point"]==source["principal_point"],"identity_drift","measurement_projection")
            candidates=(source["grey"],candidate["mask"].astype(np.float32),source["structural_weights"])
            matching=[array for array in candidates if array.dtype.str==payload["source_dtype"]
                      and hashlib.sha256(memoryview(array)).hexdigest()==payload["source_sha256"]]
            require(matching,"identity_drift","rectification_source")
            verify_remap_values(matching[0],payload["map_x"],payload["map_y"],payload["result"])
            candidate["remaps"].append(payload)
        elif kind=="measurement_round_begin":
            require(identity in self.candidates,"identity_drift","round_candidate")
            candidate=self.candidates[identity]; index=payload["round_index"]
            require(index==candidate["rounds"] and index<CAPS["rounds"],"identity_drift","round_order")
            candidate["rounds"]+=1
            require("window" in candidate,"missing_dependency","candidate_window")
            require((payload["centre"][0],payload["centre"][1],payload["extent"],payload["side"])==candidate["window"],"identity_drift","round_window")
            self.round=dict(candidate=identity,index=index,samples=payload["samples"],profiles={},estimates={})
        elif kind=="profile" and context.get("profile_use")=="directional":
            require(self.round is not None and self.round["candidate"]==identity,"identity_drift","profile_round")
            along=payload["along"]; require(along in (0,1) and along not in self.round["profiles"],"ambiguous_join","profile_direction")
            self.round["profiles"][along]=payload["profile"]
        elif kind=="profile" and context.get("profile_use")=="supra":
            candidate=self.candidates[identity]; last=candidate.get("last_round")
            require(last is not None,"missing_dependency","supra_measurement")
            along=payload["along"]
            require(along==last["direction"] and same(payload["profile"],last["profiles"][along]),
                    "identity_drift","supra_profile_join")
            candidate["supra_profile"]=(float(context["multiple"]),along,payload["profile"])
        elif kind=="period_estimate":
            require(self.round is not None and self.round["candidate"]==identity,"identity_drift","estimate_round")
            along=payload["along"]; require(along in self.round["profiles"] and along not in self.round["estimates"],"ambiguous_join","estimate_direction")
            profile=self.round["profiles"][along]; signal=profile-profile.mean()
            require(np.array_equal(signal,payload["profile"]),"identity_drift","centered_profile")
            verify_period_arithmetic(payload,signal)
            estimate=payload["estimate"]
            if estimate is not None:
                reading=fields(estimate,"PeriodEstimate"); correlation=payload["correlation"]
                require(correlation is not None and correlation[0]==1 and np.isfinite(correlation).all(),"identity_drift","period_correlation")
                peaks=payload["peaks"]; choices=payload["harmonic_candidates"]
                require(len(peaks)>0 and len(choices)>0 and reading["strength"]>0,"identity_drift","period_peak")
                # Supports are captured from the same normalized native correlation.
                for multiple,support in reading["block_supports"]:
                    lag=int(round(reading["period_pixels"]*multiple))
                    require(0<lag<len(correlation)-1,"identity_drift","block_lag")
                    expected=float(correlation[max(lag-2,0):lag+3].max()/reading["strength"])
                    require(support==expected,"identity_drift","block_support")
            self.round["estimates"][along]=estimate
        elif kind=="period_reconciliation":
            require(self.round is not None and set(self.round["estimates"])=={0,1},"missing_dependency","two_period_directions")
            require(payload["readings"]==tuple(self.round["estimates"][a] for a in (0,1)),"identity_drift","reconcile_readings")
            found=[(along,reading) for along,reading in enumerate(payload["readings"]) if reading is not None]
            if not found: expected=None
            elif len(found)==1: expected=found[0]
            else:
                short,long_=sorted(found,key=lambda pair:fields(pair[1],"PeriodEstimate")["period_pixels"])
                factor=fields(long_[1],"PeriodEstimate")["period_pixels"]/fields(short[1],"PeriodEstimate")["period_pixels"]
                nearest=round(factor)
                expected=short if nearest>=2 and abs(factor-nearest)<=payload["tolerance"] else max(found,key=lambda pair:fields(pair[1],"PeriodEstimate")["strength"])
            require(payload["selected"]==expected,"identity_drift","reconciliation_winner")
            self.round["winner"]=expected
        elif kind=="measurement_round":
            require(self.round is not None and self.round["candidate"]==identity and self.round["index"]==payload["round_index"],"identity_drift","round_binding")
            samples=payload["samples"]; patch=payload["patch"]; weights=payload["region_weights"]
            remaps=self.candidates[identity]["remaps"]
            require(len(remaps)>=2 and same(remaps[-2]["result"],patch) and same(remaps[-1]["result"],weights),"identity_drift","round_remap_patches")
            require(all(r["samples"]==samples and r["extent"]==payload["extent"] and r["centre"]==payload["centre"] for r in remaps[-2:]),"identity_drift","round_remap_window")
            require(samples==self.round["samples"] and patch.shape[:2]==weights.shape==(samples,samples),"identity_drift","patch_samples")
            for along,profile in self.round["profiles"].items():
                grey=patch if patch.ndim==2 else patch.mean(axis=2)
                gradient=np.abs(np.diff(grey,axis=1-along)); mask=weights[:,:-1] if along==0 else weights[:-1,:]
                total=mask.sum(axis=along); covered=(gradient*mask).sum(axis=along)
                expected=np.divide(covered,total,out=np.zeros_like(covered),where=total>0)
                require(np.array_equal(expected,profile),"identity_drift","patch_profile")
            winner=self.round.get("winner")
            require(payload["estimate"]==(None if winner is None else winner[1]) and payload["winning_direction"]==(None if winner is None else winner[0]),"identity_drift","round_winner")
            self.candidates[identity]["last_round"]=dict(samples=samples,extent=payload["extent"],
                estimate=payload["estimate"],direction=payload["winning_direction"],profiles=dict(self.round["profiles"]))
        elif kind=="supra_support":
            candidate=self.candidates[identity]; correlation=payload["correlation"]; lag=payload["lag"]
            signal=payload["signal"]
            last=candidate.get("last_round"); profile=candidate.pop("supra_profile",None)
            require(last is not None and profile is not None,"missing_dependency","supra_profile")
            require(profile[:2]==(float(payload["multiple"]),payload["along"])
                    and payload["period"]==fields(last["estimate"],"PeriodEstimate")["period_pixels"]
                    and same(signal,profile[2]-profile[2].mean()),"identity_drift","supra_signal_join")
            if correlation is not None:
                expected=np.correlate(signal,signal,mode="full")[len(signal)-1:]; expected/=expected[0]
                require(np.array_equal(expected,correlation) and lag==int(round(payload["period"]*payload["multiple"])),"identity_drift","supra_correlation")
            support=0. if lag is None or correlation is None or not 0<lag<len(correlation) else float(max(0.,correlation[lag]))
            candidate.setdefault("supra",{})[float(payload["multiple"])]=support
        elif kind=="distance_alternatives":
            candidate=self.candidates[identity]; period=fields(payload["estimate"],"PeriodEstimate")["period_pixels"]
            last=candidate.get("last_round")
            require(last is not None and same(payload["estimate"],last["estimate"])
                    and payload["samples"]==last["samples"] and payload["extent"]==last["extent"],
                    "identity_drift","distance_measurement_join")
            world=period*(2.*payload["extent"]/(payload["samples"]-1))
            require(world==payload["world_period"],"identity_drift","pixel_step")
            expected={1.:1.}
            for multiple,support in fields(payload["estimate"],"PeriodEstimate")["block_supports"]:
                if multiple>0 and math.isfinite(support) and support>=0: expected[1./multiple]=support
            expected.update(candidate.get("supra",{}))
            require(tuple(expected.items())==payload["evaluated"],"identity_drift","evaluated_support_order")
            ranked=tuple(sorted(expected.items(),key=lambda item:(-item[1],-item[0])))
            require(ranked==payload["ranked"] and payload["prune_share"]==lp.SUPPORT_PRUNE and payload["prune_floor"]==lp.SUPPORT_PRUNE*ranked[0][1],"identity_drift","prune_order")
            retained=tuple((b,b/world,support) for b,support in ranked if support>=payload["prune_floor"])
            actual=tuple((v["blocks_per_period"],v["distance_blocks"],v["support"]) for v in (fields(item,"DistanceAlternative") for item in payload["retained"]))
            require(retained==actual,"identity_drift","distance_support_pruning")
            candidate["alternatives"]=payload["retained"]
        elif kind=="candidate_reading":
            candidate=self.candidates[identity]; reading=fields(payload["reading"],"PlaneReading")
            policy=self.measure_sources[identity[0]]["selection_policy"]
            require(type(payload.get("eligible")) is bool
                    and payload["eligible"]==(reading["structural_share"]>=float(policy["min_structural_share"])),
                    "identity_drift","candidate_eligibility")
            candidate["eligible"]=payload["eligible"]
            require(fields(reading["region_evidence"],"PlaneRegionEvidence")["source_region_id"]==candidate["region_id"],"identity_drift","reading_region")
            if candidate["axis"]==1:
                last=candidate.get("last_round"); require(last is not None,"missing_dependency","reading_round")
                require(reading["samples"]==last["samples"] and reading["extent"]==last["extent"] and reading["period_pixels"]==fields(last["estimate"],"PeriodEstimate")["period_pixels"],"identity_drift","reading_measurement")
                require(reading["alternatives"]==candidate["alternatives"],"identity_drift","reading_alternatives")
                require(payload["structural_energy_patch"].shape==(reading["samples"],reading["samples"]),"identity_drift","structural_patch")
                require(candidate["remaps"] and same(candidate["remaps"][-1]["result"],payload["structural_energy_patch"]),"identity_drift","structural_remap_binding")
                verify_phase(reading,payload["structural_energy_patch"])
                share=lp._structural_share(payload["structural_energy_patch"],reading["period_pixels"])
                cells=int(reading["samples"]//max(reading["period_pixels"],1.))
                strength=fields(last["estimate"],"PeriodEstimate")["strength"]
                terms=(min(cells/8.,1.),min(max(float(strength),0.)/.5,1.),min(share/.5,1.))
                confidence=float(np.clip(.4*terms[2]+.3*terms[0]+.3*terms[1],0.,1.))
                require(reading["coverage_cells"]==cells and reading["structural_share"]==share
                        and payload["confidence_terms"]==terms and reading["confidence"]==confidence,
                        "identity_drift","reading_confidence")
                rect=fields(reading["rectification"],"PlaneRectification")
                source=self.measure_sources[identity[0]]; window=candidate["window"]
                require(reading["axis"]==candidate["axis"] and reading["side"]==window[3]
                        and tuple(rect["centre"])==tuple(window[:2])
                        and same(np.asarray(rect["axis_matrix"]),source["axis_matrix"])
                        and rect["focal"]==source["focal"] and rect["principal_point"]==source["principal_point"],
                        "identity_drift","reading_projection")
            phase=reading["phase_evidence"]
            require(phase is not None,"missing_dependency","phase_evidence")
            phase=fields(phase,"PeriodicEnergyPhaseEvidence")
            require(phase["source_region_id"]==candidate["region_id"] and phase["samples"]==reading["samples"] and phase["period_pixels"]==reading["period_pixels"],"identity_drift","phase_binding")
            candidate["reading"]=payload["reading"]
        elif kind=="frame_reading_selection":
            ranked=payload["ranked"]; selected=payload["selected"]
            policy=self.measure_sources[payload["frame"]]["selection_policy"]
            require(all(payload.get(name)==value for name,value in policy.items()),"identity_drift","selection_policy_join")
            for cid,eligible,reading in ranked:
                candidate=self.candidates[tuple(cid)]
                require(candidate["reading"]==reading and type(eligible) is bool
                        and eligible==candidate["eligible"],"identity_drift","ranked_candidate_eligibility")
            def key(row):
                cid,eligible,record=row; reading=fields(record,"PlaneReading")
                return (not eligible,reading["axis"],reading["side"],-reading["confidence"],reading["polygon"] or (),fields(reading["region_evidence"],"PlaneRegionEvidence")["source_region_id"])
            require(tuple(sorted(ranked,key=key))==ranked,"identity_drift","candidate_rank")
            count={}; expected=[]
            for cid,eligible,reading in ranked:
                r=fields(reading,"PlaneReading"); pair=(r["axis"],r["side"])
                if count.get(pair,0)>=payload["max_per_axis_side"]: continue
                count[pair]=count.get(pair,0)+1; expected.append((cid,reading))
                if len(expected)>=payload["max_readings"]: break
            expected.sort(key=lambda item:key((item[0],True,item[1]))[1:])
            require(tuple(expected)==selected,"identity_drift","selected_candidates")
            for cid,reading in selected:
                seal=digest("reading",encode(reading)); require(seal not in self.selected_candidates,"ambiguous_join","reading_digest")
                self.selected_candidates[seal]=tuple(cid)
        elif kind=="lattice_final":
            for original_index,reading in enumerate(payload["readings"]):
                self.original_indices[digest("reading",encode(reading))]=original_index
            grouped={}
            for reading in payload["readings"]: grouped.setdefault(fields(reading,"PlaneReading")["frame"],[]).append(reading)
            for frame,group in grouped.items():
              for position,reading in enumerate(sorted(group,key=lambda r:reading_record_order_key(r,trusted_shapes))):
                seal=digest("reading",encode(reading)); require(seal in self.selected_candidates,"missing_dependency","reading_candidate")
                require((frame,position) not in self.readings,"ambiguous_join","canonical_reading")
                self.readings[(frame,position)]=reading
        elif kind=="signature_rectification":
            sid=tuple(context["signature_identity"]); parts=self.signature_parts.setdefault(sid,{"rectifications":[]})
            require(payload["rectification_index"]==len(parts["rectifications"]),"identity_drift","signature_rectification_order")
            metadata=dict(payload["metadata"]); arrays=payload["arrays"]
            for name,array in arrays.items():
                description=metadata.pop(name)
                require(same(description,dict(dtype=array.dtype.str,shape=list(array.shape),sha256=hashlib.sha256(array.tobytes()).hexdigest())),"digest_mismatch","rectification_native")
            witness=lattice.RectificationWitnessV1(**dict(metadata,**arrays))
            verify_geometry(payload["geometry"],witness.map_x,witness.map_y,witness.depth_valid)
            source=self.measure_sources.get(sid[0]); require(source is not None,"missing_dependency","signature_source")
            sources=(self.source_frames[sid[0]]["rgb"],source["usable"].astype(np.float32))
            if (sid[0],sid[1]) in self.readings:
                reading=fields(self.readings[(sid[0],sid[1])],"PlaneReading")
                if reading["region_evidence"] is not None:
                    validate_region_plan(reading["region_evidence"],trusted_shapes[sid[0]])
                    region=lp.PlaneRegionEvidence(**fields(reading["region_evidence"],"PlaneRegionEvidence"))
                    sources=(*sources,(lp._decode_region_evidence(region,reading["axis"],reading["polygon"]) & source["usable"]).astype(np.float32))
                g=payload["geometry"]; rect=fields(reading["rectification"],"PlaneRectification")
                require(g["axis"]==reading["axis"] and g["side"]==reading["side"]
                        and same(g["axis_matrix"],np.asarray(rect["axis_matrix"])) and g["focal"]==rect["focal"]
                        and g["principal_point"]==rect["principal_point"],"identity_drift","signature_projection")
            index=payload["rectification_index"]; require(index<len(sources),"identity_drift","signature_source_role")
            native=sources[index]
            require(native.dtype.str==payload["source_dtype"] and hashlib.sha256(memoryview(native)).hexdigest()==payload["source_sha256"],"identity_drift","signature_source_hash")
            verify_remap_values(native,witness.map_x,witness.map_y,payload["result"])
            verify_native_taps(witness)
            h,w=witness.source_shape
            require(np.all((witness.tap_source_indices>=-1)&(witness.tap_source_indices<h*w)),"identity_drift","source_tap_bounds")
            require(np.all(np.diff(witness.tap_ptr)>=0) and np.all(witness.tap_weight_numerators>0),"identity_drift","tap_weights")
            parts["rectifications"].append(witness)
            parts.setdefault("outputs",[]).append(payload["result"])
            parts.setdefault("geometries",[]).append(payload["geometry"])
        elif kind=="signature_sampling":
            sid=tuple(context["signature_identity"]); parts=self.signature_parts[sid]; rects=parts["rectifications"]
            metadata=payload["metadata"]; signature=fields(payload["signature"],"PlaneSignature")
            require((sid[0],sid[1]) in self.readings and signature["reading"]==self.readings[(sid[0],sid[1])],"identity_drift","signature_reading")
            grey=payload["grayscale"]; gy,gx=np.gradient(grey); energy=np.hypot(gx,gy)
            rgb=parts["outputs"][0].astype(np.float32)
            expected=rgb.mean(axis=2) if rgb.ndim==3 else rgb
            require(np.array_equal(expected,grey) and same(parts["outputs"][1],payload["coverage_patch"]),"identity_drift","signature_remap_output")
            require(np.array_equal(gy,payload["gradient_y"]) and np.array_equal(gx,payload["gradient_x"]) and np.array_equal(energy,payload["energy"]),"identity_drift","signature_derivatives")
            witness=lp.SignatureSamplingWitnessV1(
                metadata["schema"],rects[0],rects[1],rects[2] if len(rects)==3 else None,
                metadata["grayscale_sha256"],metadata["gradient_y_sha256"],metadata["gradient_x_sha256"],metadata["energy_sha256"],metadata["coverage_patch_sha256"],metadata["gradient_edge_order"],metadata["subsample_factor"],metadata["seam_band"],tuple(tuple(r) for r in metadata["cell_rectangles"]),tuple(tuple(r) for r in metadata["interior_rectangles"]),payload["cell_source_ptr"],payload["cell_source_indices"],metadata["witness_sha256"])
            for name,key in (("grayscale","grayscale_sha256"),("gradient_y","gradient_y_sha256"),("gradient_x","gradient_x_sha256"),("energy","energy_sha256"),("coverage_patch","coverage_patch_sha256")):
                require(hashlib.sha256(payload[name].tobytes()).hexdigest()==metadata[key],"digest_mismatch",key)
            require(same(witness.to_record(),metadata),"digest_mismatch","signature_witness")
            rows,columns=signature["cells"].shape; require(rows*columns==len(witness.cell_rectangles),"identity_drift","cell_count")
            covered=payload["coverage_patch"]>.5
            sub=witness.subsample_factor; band=witness.seam_band
            require(sub>=1 and band==max(1,sub//4) and energy.shape==(rows*sub,columns*sub),"identity_drift","signature_grid")
            r=fields(signature["reading"],"PlaneReading"); alts=r["alternatives"]
            require(0<=sid[2]<len(alts) and fields(alts[sid[2]],"DistanceAlternative")["blocks_per_period"]==sid[3]==signature["blocks_per_period"],"identity_drift","signature_alternative")
            d=fields(alts[sid[2]],"DistanceAlternative")["distance_blocks"]
            expected_centre=tuple((signature["origin_uv"][a]+(columns if a==0 else rows)/2)/d for a in (0,1))
            require(all(g["samples"]==rows*sub and g["centre"]==expected_centre and g["extent"]==rows/(2*d) for g in parts["geometries"]),"identity_drift","signature_grid_projection")
            for index,(rectangle,inside) in enumerate(zip(witness.cell_rectangles,witness.interior_rectangles)):
                r0,r1,c0,c1=rectangle; ir0,ir1,ic0,ic1=inside; row,column=divmod(index,columns)
                require(rectangle==(row*sub,(row+1)*sub,column*sub,(column+1)*sub)
                        and inside==((r0+band,r1-band,c0+band,c1-band) if sub>2*band else rectangle),"identity_drift","signature_cell_rectangles")
                require(signature["cells"][row,column]==np.float32(energy[r0:r1,c0:c1].sum()) and signature["interior"][row,column]==np.float32(energy[ir0:ir1,ic0:ic1].sum()),"identity_drift","cell_energy")
                require(bool(signature["coverage"][row,column])==bool(np.count_nonzero(covered[r0:r1,c0:c1])*2>=energy[r0:r1,c0:c1].size),"identity_drift","cell_coverage")
                members=set(); height,width=grey.shape
                for rr in range(r0,r1):
                    ys=(0,1) if rr==0 else (height-2,height-1) if rr==height-1 else (rr-1,rr+1)
                    for cc in range(c0,c1):
                        xs=(0,1) if cc==0 else (width-2,width-1) if cc==width-1 else (cc-1,cc+1)
                        for yr,xc in ({(yr,cc) for yr in ys}|{(rr,xc) for xc in xs}):
                            sample=yr*width+xc; begin,end=rects[0].tap_ptr[sample:sample+2]
                            members.update(int(v) for v in rects[0].tap_source_indices[begin:end])
                begin,end=witness.cell_source_ptr[index:index+2]
                require(np.array_equal(np.asarray(sorted(members),dtype=np.int64),witness.cell_source_indices[begin:end]),"identity_drift","signature_memberships")
            if signature["region_footprint"] is not None:
                foot=fields(signature["region_footprint"],"PlaneRegionFootprint")
                inside=parts["outputs"][2]>.5
                counts=bytes(int(np.count_nonzero(inside[a:b,c:d])) for a,b,c,d in witness.cell_rectangles)
                valid=bytes(int(np.count_nonzero(covered[a:b,c:d])) for a,b,c,d in witness.cell_rectangles)
                require(foot["inside_region_sample_counts"]==counts and foot["valid_sample_counts"]==valid
                        and foot["origin_uv"]==signature["origin_uv"] and foot["blocks_per_period"]==sid[3],"identity_drift","signature_footprint")
            parts["signature"]=payload["signature"]; parts["witness_sha256"]=witness.witness_sha256
            del parts["rectifications"]  # Native remaps/taps are never accumulated across signatures.
        elif kind=="signature_binding":
            sid=tuple(context["signature_identity"]); evidence=fields(payload["evidence"],"SignatureEvidence")
            require(same(evidence["signature"],self.signature_parts[sid]["signature"]),"identity_drift","signature_binding")
            from mc_photogrammetry.pipeline import image_derived_signatures as sigmod
            native=materialize(payload["evidence"],trusted_shapes)
            require(native.signature_sha256==evidence["signature_sha256"] and evidence["frame_id"]==sid[0]
                    and evidence["alternative_index"]==sid[2] and evidence["blocks_per_period"]==sid[3]
                    and evidence["canonical_index"]==self.original_indices[digest("reading",encode(evidence["signature"]["fields"]["reading"]))],"identity_drift","signature_identity_digest")
            h,w=self.frame_masks[sid[0]][0]
            usable=self.measure_sources[sid[0]]["usable"]
            expected_mask=sigmod.UsableMask(sid[0],h,w,usable.tobytes()).mask_sha256
            require(payload["mask_sha256"]==expected_mask,"identity_drift","signature_mask_binding")
            ordered_masks=[]
            for frame in self.source_frames:
                require(frame in self.measure_sources,"missing_dependency","all_signature_masks")
                mask=self.measure_sources[frame]["usable"]; hh,ww=mask.shape
                ordered_masks.append(sigmod.UsableMask(frame,hh,ww,mask.tobytes()))
            require(payload["masks_sha256"]==sigmod._masks_digest(tuple(ordered_masks)),
                    "identity_drift","signature_mask_set")
            require(sid not in self.signature_bindings,"ambiguous_join","signature_binding_duplicate")
            self.signature_bindings[sid]=evidence["signature_sha256"]
            self.signatures[sid]=evidence["signature"]
            self.signature_parts.pop(sid)
        elif kind=="matcher_warp":
            mid=tuple(context["match_identity"]); require(mid not in self.matcher_parts,"ambiguous_join","matcher_warp")
            values=fields(payload["witness"],"WarpSignatureWitnessV1"); witness=lp.WarpSignatureWitnessV1(**values)
            self.matcher_parts[mid]=(witness,payload["warped_cells"],payload["warped_coverage"])
        elif kind=="matcher_pair":
            mid=tuple(context["match_identity"]); require(mid in self.matcher_parts and mid not in self.matcher_pairs,"ambiguous_join","matcher_pair")
            warp,warped_cells,warped_coverage=self.matcher_parts.pop(mid); metadata=payload["metadata"]
            for offset,name in ((0,"left_signature"),(4,"right_signature")):
                matching=[sig for sid,sig in self.signatures.items() if sid[:2]==(mid[offset],mid[offset+1]) and sid[3]==mid[offset+3]]
                require(len(matching)==1 and same(matching[0],payload[name]),"identity_drift","matcher_emitted_signature")
            expected_h=pg.cell_grid_homography(materialize(payload["left_signature"],trusted_shapes),
                materialize(payload["right_signature"],trusted_shapes),metadata["quadrant"])
            require(expected_h is not None and same(expected_h,payload["homography"]),
                    "identity_drift","matcher_signature_homography")
            verify_warp_values(warp,fields(payload["left_signature"],"PlaneSignature"),warped_cells,warped_coverage)
            right_signature=fields(payload["right_signature"],"PlaneSignature")
            require(same(right_signature["coverage"],payload["right_coverage"]),"identity_drift","matcher_right_coverage")
            require(tuple(metadata["right_grid_shape"])==right_signature["cells"].shape==payload["right_coverage"].shape,
                    "identity_drift","matcher_right_grid_shape")
            witness=pg.MatcherPairWitnessV1(metadata["schema"],metadata["algorithm"],metadata["max_shift_cells"],metadata["quadrant"],payload["homography"],metadata["warp_witness_sha256"],tuple(metadata["canvas_shape"]),tuple(metadata["selected_shift_rc"]),metadata["selected_support"],payload["logical_pairs"],metadata["witness_sha256"],warp,right_grid_shape=tuple(metadata["right_grid_shape"]))
            require(same(witness.to_record(),metadata),"digest_mismatch","matcher_witness")
            require(np.array_equal(warped_coverage,payload["warped_coverage"]),"identity_drift","warp_coverage")
            rs,cs=witness.selected_shift_rc; expected=[]; rows,columns=payload["right_coverage"].shape
            for r in range(rows):
                for c in range(columns):
                    lr,lc=r+witness.max_shift_cells+rs,c+witness.max_shift_cells+cs
                    if 0<=lr<warped_coverage.shape[0] and 0<=lc<warped_coverage.shape[1] and warped_coverage[lr,lc] and payload["right_coverage"][r,c]:
                        expected.append((int(warp.source_rows[lr,lc]),int(warp.source_columns[lr,lc]),r,c))
            require(np.array_equal(np.asarray(expected,dtype=np.int64).reshape(-1,4),witness.logical_pairs),"identity_drift","logical_pairs")
            match=fields(payload["match"],"PlaneMatch")
            require((match["left"],mid[1],match["left_axis"],match["left_b"],match["right"],mid[5],match["right_axis"],match["right_b"],match["quadrant_of_right"])==mid,"identity_drift","exact_match_selector")
            require(match["overlap_cells"]==witness.selected_support and match["cell_shift_rc"]==witness.selected_shift_rc,"identity_drift","selected_match_support")
            self.matcher_pairs[mid]=match
        elif kind=="matcher_final":
            self.final_matches=payload["matches"]
            for (frame,index),reading in payload["indexed_readings"]:
                require(self.readings.get((frame,index))==reading,"identity_drift","canonical_reading_order")
            for mid,pair in self.matcher_pairs.items():
                matches=[fields(value,"PlaneMatch") for value in self.final_matches if match_identity(value)==mid]
                require(len(matches)==1,"ambiguous_join","selected_match_retained")
                current=dict(matches[0]); current["left_plane"]=current["right_plane"]=-1
                require(current==pair,"identity_drift","pair_to_final_match")
        elif kind=="solver_call":
            require(self.final_matches is not None and self.final_matches==payload["matches"],"identity_drift","solver_match_order")
            self.solver_matches=payload["matches"]
            names=("integer_tolerance","min_resultant","ambiguous_margin","quadrant_constraints")
            require(self.solver_parameters is None and all(name in payload for name in names),
                    "missing_dependency","effective_solver_parameters")
            self.solver_parameters={name:payload[name] for name in names}
            require(all(same(self.readings.get((fields(r,"PlaneReading")["frame"],i)),r)
                        for frame in set(fields(r,"PlaneReading")["frame"] for r in payload["readings"])
                        for i,r in enumerate(sorted((r for r in payload["readings"] if fields(r,"PlaneReading")["frame"]==frame),key=lambda r:reading_record_order_key(r,trusted_shapes)))),"identity_drift","solver_readings")
        elif kind=="solver_result":
            require(self.solver_matches is not None and self.solution is None,"ambiguous_join","solver_result_order")
            self.solution=payload["solution"]
        elif kind=="core_final":
            require(self.core is None,"ambiguous_join","core_final"); self.core=payload
            identities=payload.get("signature_identities",())
            for sid,sha in self.signature_bindings.items():
                reading=self.readings[sid[:2]]; original=self.original_indices[digest("reading",encode(reading))]
                matching=[row for row in identities if row["frame_id"]==sid[0]
                          and row["canonical_index"]==original and row["alternative_index"]==sid[2]]
                require(len(matching)==1 and matching[0]["signature_sha256"]==sha,"identity_drift","final_signature_identity")

    def finish(self,*,baseline=None):
        require(self.core is not None and self.solution is not None and self.solver_matches is not None,"missing_dependency","core_chain")
        require(all(target in self.readings for target in self.target_readings),"missing_dependency","target_readings")
        require(set(self.matcher_pairs)==set(self.target_matches),"missing_dependency","selected_matches")
        require(not self.signature_parts and not self.matcher_parts,"missing_dependency","unfinished_witness")
        expected={(frame,index,a,float(fields(alt,"DistanceAlternative")["blocks_per_period"]))
                  for frame,index in self.target_readings
                  for a,alt in enumerate(fields(self.readings[(frame,index)],"PlaneReading")["alternatives"])}
        require(set(self.signature_bindings)==expected,"missing_dependency","all_retained_signature_alternatives")
        require(set(self.structural_sources)==set(self.source_frames),"missing_dependency","all_structural_sources")
        if baseline is not None:
            verify_core_chain(self.core,self.solution,self.solver_matches,self.solver_parameters)
            require(self.core["graph_sha256"]==BASELINE_GRAPH_SHA and self.core["trace_sha256"]==BASELINE_TRACE_SHA,"baseline_drift","core_digest")
            require(same(self.core["graph"],baseline["graph"]) and same(self.core["trace"],baseline["trace"]) and same(self.core["solver_failure_observation"],baseline["solver_failure_observation"]),"baseline_drift","core_records")
        # V1 membership replay is a legacy-table consistency check. It does not
        # certify the installed OpenCV5 float-map kernel's source membership.
        require_supported_kernel()
        return dict(status="complete",target_readings=len(self.target_readings),selected_matches=len(self.matcher_pairs),
                    signatures=len(self.signature_bindings),graph_sha256=self.core["graph_sha256"],trace_sha256=self.core["trace_sha256"])


def same(left,right):
    if isinstance(left,np.ndarray) or isinstance(right,np.ndarray):
        return (isinstance(left,np.ndarray) and isinstance(right,np.ndarray)
                and left.dtype==right.dtype and left.shape==right.shape and np.array_equal(left,right))
    if isinstance(left,dict) or isinstance(right,dict):
        return isinstance(left,dict) and isinstance(right,dict) and set(left)==set(right) and all(same(left[key],right[key]) for key in left)
    if isinstance(left,(tuple,list)) or isinstance(right,(tuple,list)):
        return isinstance(left,(tuple,list)) and isinstance(right,(tuple,list)) and len(left)==len(right) and all(same(a,b) for a,b in zip(left,right))
    return left==right


def match_identity(value):
    match=fields(value,"PlaneMatch")
    return (match["left"],match["left_plane"],match["left_axis"],match["left_b"],
            match["right"],match["right_plane"],match["right_axis"],match["right_b"],match["quadrant_of_right"])


def runtime_pins():
    import PIL
    binaries={}
    for path in sorted(Path(cv2.__file__).parent.glob("*.pyd")):
        with path.open("rb") as file: binaries[path.name]=dict(byte_length=path.stat().st_size,sha256=_file_digest(file))
    return dict(python=sys.version,implementation=platform.python_implementation(),
                numpy=np.__version__,opencv=cv2.__version__,pillow=PIL.__version__,
                opencv_binaries=binaries,optimized=cv2.useOptimized(),cpu_features=cv2.getCPUFeaturesLine(),
                opencl_enabled=cv2.ocl.useOpenCL(),ipp_enabled=cv2.ipp.useIPP(),
                ipp_not_exact=cv2.ipp.useIPP_NotExact(),ipp_version=cv2.ipp.getIppVersion(),
                opencv_build_sha256=hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest())


def require_supported_kernel():
    # There is no verified matching binary/build proof of IPP_CALLS_ENFORCED
    # in this package. CPU source agreement and impulse tests cannot fill that
    # gap. No one-shot can be consumed while this condition remains unresolved.
    raise CaptureRefusal("unsupported_kernel","active HAL/CPU dispatch not certified; IPP_CALLS_ENFORCED provenance missing")


def dependency_hashes():
    """Static local-import closure, including function-local imports and package initializers."""
    pending=[Path(__file__).resolve(),ROOT/"scripts/run_image_derived_graph_stage.py"]
    seen={}; source=ROOT/"src"
    def add_module(module):
        if not module.startswith("mc_photogrammetry"): return
        parts=module.split(".")
        for size in range(1,len(parts)+1):
            package=source.joinpath(*parts[:size],"__init__.py")
            if package.is_file(): pending.append(package)
        path=source.joinpath(*parts).with_suffix(".py")
        if path.is_file(): pending.append(path)
    while pending:
        path=pending.pop().resolve()
        relative=path.relative_to(ROOT).as_posix()
        if relative in seen: continue
        raw=path.read_bytes(); seen[relative]=hashlib.sha256(raw).hexdigest()
        tree=ast.parse(raw.decode("utf-8-sig"))
        module_parts=path.relative_to(source).with_suffix("").parts if path.is_relative_to(source) else ()
        package_parts=module_parts[:-1]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                for alias in node.names: add_module(alias.name)
            elif isinstance(node,ast.ImportFrom):
                base=node.module or ""
                if node.level:
                    base=".".join((*package_parts[:len(package_parts)-node.level+1],*base.split(".")))
                add_module(base)
                for alias in node.names:
                    if alias.name!="*": add_module(base+"."+alias.name)
    return dict(sorted(seen.items()))


def freeze_preregistration():
    value=dict(schema=SCHEMA,study="PLANE-PERIOD-SUPPORT-01",mode="observer_only",
        targets=dict(readings=TARGET_READINGS,matches=TARGET_MATCHES),caps=CAPS,
        reading_policy=registered_reading_policy(),
        baseline=dict(path=BASELINE,file_sha256=BASELINE_SHA,graph_sha256=BASELINE_GRAPH_SHA,
                      trace_sha256=BASELINE_TRACE_SHA,ordered_match_identity_sha256=BASELINE_ORDERED_SHA),
        evidence_path=EVIDENCE.relative_to(ROOT).as_posix(),
        future_stage_directory="docs/reports/PLANE-PERIOD-SUPPORT-01-run-2026-10-05",
        source_sha256=dependency_hashes(),runtime=runtime_pins(),
        protocol=dict(one_shot="exclusive durable NOT_STARTED->RUNNING before baseline/RGB load; no retry",
            private_context="run/thread owned; no nesting; finally reset; interrupts propagate",
            numerical_calls="unchanged args/counts/data/decision/return/core hashes; no legacy sink activation",
            diagnostic_work="only target witness mapping from existing locals; failures latch incomplete",
            replay="complete bounded gzip/CRC/order/seal/native/semantic replay before interpretation; final digest recheck",
            remap_output="post-production detached cv2.remap replay only after first complete stream verification; native dtype/shape/bytes equality; no epsilon and no extra production sampling",
            support_kernel="UNSUPPORTED: exact installed HAL/CPU dispatch and nonzero kernel support unproven; unconditional unsupported_kernel before claim_run and semantic completion; V1 preserved as legacy-table witness only; V2 not implemented",
            memory="single monotone 256MiB preallocation ledger per capture/replay; scalar/container/native snapshots, witness work, encoding, readback, bounded decode, JSON tree and semantic temporaries charged; freed buffers remain charged; conservative early cap refusal",
            interpretation="require identical sealed R8 core/failure/38 ordered matches; otherwise baseline_drift; no replacement selection"))
    # JSON-normalize tuples so the in-report registry has one exact representation.
    value=json.loads(canonical(value)); return dict(value,preregistration_sha256=digest("preregistration",value))


def verify_preregistration(value):
    require(value==freeze_preregistration(),"missing_dependency","source_runtime_preregistration_drift")


def load_preregistration(report=REPORT):
    text=Path(report).read_text(encoding="utf-8")
    start=text.index("<!-- PLANE_PERIOD_SUPPORT_PREREG -->")
    opening=text.index("```json\n",start)+len("```json\n")
    closing=text.index("\n```",opening)
    value=json.loads(text[opening:closing]); verify_preregistration(value)
    return value


def claim_run(report=REPORT,evidence=EVIDENCE):
    """Durably consume authorization before any baseline/image loading."""
    require(not Path(evidence).exists(),"write_failure","evidence_already_exists")
    with Path(report).open("r+b") as file:
        import msvcrt
        file.seek(0); msvcrt.locking(file.fileno(),msvcrt.LK_NBLCK,1)
        try:
            file.seek(0); raw=file.read(); old=MARKER.encode()
            require(raw.count(old)==1 and b"PLANE_PERIOD_SUPPORT_EXECUTION_STATE: RUNNING" not in raw and b"PLANE_PERIOD_SUPPORT_EXECUTION_STATE: COMPLETE" not in raw,"identity_drift","one_shot_consumed")
            require(not Path(evidence).exists(),"write_failure","evidence_already_exists")
            raw=raw.replace(old,b"PLANE_PERIOD_SUPPORT_EXECUTION_STATE: RUNNING",1)
            file.seek(0); require(file.write(raw)==len(raw),"write_failure","marker_write")
            file.truncate(); file.flush(); os.fsync(file.fileno())
        finally:
            file.seek(0); msvcrt.locking(file.fileno(),msvcrt.LK_UNLCK,1)


def _import_stage():
    path=ROOT/"scripts/run_image_derived_graph_stage.py"
    spec=importlib.util.spec_from_file_location("period_support_graph_stage",path)
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
    return module


def _complete_report(result,report=REPORT):
    text=Path(report).read_text(encoding="utf-8")
    marker="PLANE_PERIOD_SUPPORT_EXECUTION_STATE: RUNNING"
    require(text.count(marker)==1,"identity_drift","complete_marker")
    text=text.replace(marker,"PLANE_PERIOD_SUPPORT_EXECUTION_STATE: COMPLETE",1)
    text+="\n\nFrozen one-shot result:\n```json\n"+json.dumps(result,sort_keys=True,indent=2)+"\n```\n"
    with Path(report).open("r+b") as file:
        raw=text.encode(); require(file.write(raw)==len(raw),"write_failure","report_write")
        file.truncate(); file.flush(); os.fsync(file.fileno())


def execute_registered():
    prereg=load_preregistration(); run_id=prereg["preregistration_sha256"]
    require_supported_kernel()
    claim_run()
    baseline_path=ROOT/BASELINE
    require(baseline_path.stat().st_size<=16*1024**2,"cap_exhausted","baseline_file")
    raw=baseline_path.read_bytes(); require(hashlib.sha256(raw).hexdigest()==BASELINE_SHA,"baseline_drift","baseline_file")
    baseline=json.loads(raw)
    require(baseline["graph_sha256"]==BASELINE_GRAPH_SHA and baseline["trace_sha256"]==BASELINE_TRACE_SHA,"baseline_drift","baseline_core")
    inputs=baseline["inputs"]; require(len(inputs)==3,"baseline_drift","frame_count")
    stage=_import_stage(); directory=ROOT/prereg["future_stage_directory"]
    require(not directory.exists(),"write_failure","stage_directory_exists")
    directory.mkdir()
    with EVIDENCE.open("x+b") as file:
        journal=Journal(file,run_id=run_id,durable=True)
        journal.append("registration",dict(run_id=run_id),dict(preregistration=prereg))
        observer=Observer(journal,run_id=run_id)
        output=stage.execute_graph_stage(png_paths=[row["path"] for row in inputs],
            expected_sha256=[row["sha256"] for row in inputs],
            project_path=directory/"stage.mcscan",output_path=directory/"observation.json",
            telemetry_path=directory/"telemetry.json",_period_support_observer=observer)
        require(observer.finished and observer.incomplete is None and not journal.failed,"capture_incomplete",str(observer.incomplete))
        require(output.get("execution_complete") is True,"baseline_drift","stage_refusal")
        # First complete structural/digest/native replay without arithmetic or
        # cv2 sampling. Only a verified full stream permits semantic evaluation.
        replay_result=replay(file,journal.members,run_id=run_id,expected_sha256=journal.sha.hexdigest())
        semantic=SemanticReplay(trusted_ingress=observer.trusted_ingress)
        semantic_replay=replay(file,journal.members,run_id=run_id,expected_sha256=replay_result["evidence_sha256"],consumer=semantic)
        require(semantic_replay==replay_result,"digest_mismatch","semantic_stream_changed")
        semantic_result=semantic.finish(baseline=baseline)
        file.seek(0); final_sha=hashlib.file_digest(file,"sha256").hexdigest() if hasattr(hashlib,"file_digest") else _file_digest(file)
        require(final_sha==replay_result["evidence_sha256"],"digest_mismatch","final_artifact")
        result=dict(schema=SCHEMA,status="complete",preregistration_sha256=run_id,
                    trusted_ingress=observer.trusted_ingress,
                    trusted_ingress_sha256=digest("trusted-ingress",observer.trusted_ingress),
                    replay=replay_result,semantic=semantic_result,members=journal.members,
                    evidence_path=prereg["evidence_path"],numerical_intervention=False)
    _complete_report(result)
    return result


def _file_digest(file):
    value=hashlib.sha256()
    for chunk in iter(lambda:file.read(1024*1024),b""): value.update(chunk)
    return value.hexdigest()


def main():
    import argparse
    parser=argparse.ArgumentParser(description="Internal observer-only evaluation; separate lead authorization required.")
    parser.add_argument("--execute-lead-authorized-once",action="store_true")
    args=parser.parse_args()
    if not args.execute_lead_authorized_once:
        print(json.dumps(dict(status="NOT_STARTED",preregistration=freeze_preregistration()),sort_keys=True)); return 0
    try:
        print(json.dumps(execute_registered(),sort_keys=True)); return 0
    except CaptureRefusal as error:
        print(json.dumps(dict(status="incomplete",reason=error.reason,detail=error.detail),sort_keys=True)); return 2




def verify_solve_parameters(parameters,context,frames):
    from mc_photogrammetry.pipeline import image_derived_graph as graph
    require(type(parameters) is dict and set(parameters)=={"integer_tolerance","min_resultant","ambiguous_margin","quadrant_constraints"},
            "missing_dependency","effective_solver_parameters")
    require(type(context) is dict,"missing_dependency","solve_context")
    try: graph.validate_solver_solve_context(context,expected_frames=frames)
    except (ValueError,TypeError,KeyError) as error: raise CaptureRefusal("identity_drift","solve_context") from error
    require(context["status"]=="complete" and not context.get("invalid_fields"),"missing_dependency","complete_solve_context")
    for name in ("integer_tolerance","min_resultant","ambiguous_margin"):
        value=parameters[name]
        require(type(value) in (int,float) and math.isfinite(value) and value==context[name],
                "identity_drift","solve_parameter:"+name)
    constraints=parameters["quadrant_constraints"]
    require(type(constraints) is tuple and len(constraints)<=graph.SOLVER_SOLVE_CONTEXT_MAX_CONSTRAINTS
            and all(type(row) is tuple and len(row)==3 and type(row[0]) is str
                    and type(row[1]) is str and type(row[2]) is int and row[2] in (0,1,2,3) for row in constraints),
            "identity_drift","solve_constraint_shape")
    rows=[dict(left=row[0],right=row[1],yaw=row[2]) for row in constraints]
    require(rows==context["constraints"],"identity_drift","solve_quadrant_constraints")


def verify_core_chain(core,solution,matches,solver_parameters=None):
    from mc_photogrammetry.pipeline import image_derived_graph as graph
    memory_reserve(8*json_bound(core)+1024**2,"core_normalization_before_allocation")
    normalized=json.loads(canonical(encode(core)))
    require(graph._sha(normalized["graph"]["payload"])==normalized["graph"]["result_sha256"]==core["graph_sha256"],"digest_mismatch","graph_core")
    trace=normalized["trace"]; trace_body={key:item for key,item in trace.items() if key!="attribution_sha256"}
    require(hashlib.sha256(canonical(trace_body)).hexdigest()==trace["attribution_sha256"]==core["trace_sha256"],"digest_mismatch","trace_core")
    observation=normalized["solver_failure_observation"]
    require(type(observation) is dict,"missing_dependency","failure_observation")
    body={key:item for key,item in observation.items() if key!="observation_sha256"}
    require(hashlib.sha256(canonical(body)).hexdigest()==observation["observation_sha256"],"digest_mismatch","failure_observation")
    verify_solve_parameters(solver_parameters,observation.get("solve_context"),
                            tuple(fields(solution,"ComponentSolution")["frames"]))
    phase=observation["phase_capture"]; inputs=observation["solver_input_bindings"]
    graph.validate_solver_phase_capture(phase); graph.validate_solver_input_bindings(inputs)
    identities=[]
    for value in matches:
        match=fields(value,"PlaneMatch")
        identities.append([match["left"],match["left_plane"],match["left_axis"],match["right"],match["right_plane"],match["right_axis"],match["quadrant_of_right"],float(match["left_b"]),float(match["right_b"])])
    ordered_sha=graph._sha(dict(domain=graph.SOLVER_INPUT_BINDINGS_INPUT_MATCH_IDENTITY_DOMAIN,
                               schema=graph.SOLVER_INPUT_BINDINGS_INPUT_MATCH_IDENTITY_SCHEMA,payload=identities))
    require(len(matches)==38 and ordered_sha==BASELINE_ORDERED_SHA==inputs["input_match_identity_sha256"],"baseline_drift","ordered_38_matches")
    require(phase["source_sha256"]==inputs["failed_assignment_sha256"],"identity_drift","phase_assignment")
    returned=fields(solution,"ComponentSolution")
    require(phase["translations"]==[[frame,list(returned["translations"][frame])] for frame in phase["frames"]],"identity_drift","returned_translations")
    require(phase["assignments"]==[returned["quadrants"][frame] for frame in phase["frames"]],"identity_drift","returned_assignment")
    require(returned["reason"]==observation["raw"]["reason"],"identity_drift","returned_failure")

if __name__=="__main__": raise SystemExit(main())
